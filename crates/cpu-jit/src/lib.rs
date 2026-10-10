//! A64 execution using the vendored Dynarmic recompiler.
mod callbacks;
mod ffi;
pub mod metrics;
use metrics::{Counter, Timer, record};
mod invalidation;
mod invocation;
mod state;
mod store_continuation;
pub use state::ThreadState;

use nixe_cpu::execution::{
    ArchitecturalTimer, ControlRequest, CpuControl, CpuExit, CpuFault, CpuFaultKind,
    ExecutionReport, MemoryBinding, SchedulerRequest, VcpuEventState,
};
use nixe_cpu::location::LocationDescriptor;
use nixe_cpu::memory::{CpuMemory, ExecutionMemory, InstructionMemory};
use nixe_cpu::profile::ProcessCpuContext;
use nixe_memory::{GuestVirtualAddress, MemoryInvalidationSource};
use std::collections::HashSet;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Weak};

#[derive(Debug)]
pub struct JitError(Box<str>);
impl std::fmt::Display for JitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}
impl std::error::Error for JitError {}

// The monitor is internally synchronized by Dynarmic. It outlives every core,
// including scheduler control handles retained after a worker is dropped.
struct Monitor(*mut c_void);
unsafe impl Send for Monitor {}
unsafe impl Sync for Monitor {}
impl Drop for Monitor {
    fn drop(&mut self) {
        unsafe { ffi::nixe_dynarmic_monitor_destroy(self.0) }
    }
}
struct Native {
    pointer: *mut c_void,
    _process: Arc<JitProcess>,
    context_owned: AtomicBool,
}
// Only HaltExecution may touch live execution concurrently. ThreadState owns
// stopped registers exclusively; moving it into JitThread's synchronous run
// transfers that authority to the worker until return. A different state cannot
// enter while context_owned is set. Callback borrows last only for that run.
unsafe impl Send for Native {}
unsafe impl Sync for Native {}
impl Drop for Native {
    fn drop(&mut self) {
        unsafe { ffi::nixe_dynarmic_destroy(self.pointer) }
    }
}

pub struct JitProcess {
    cpu: ProcessCpuContext,
    memory: Arc<ExecutionMemory>,
    monitor: Arc<Monitor>,
    cores: Mutex<Vec<Weak<Native>>>,
    stopped: AtomicBool,
    frequency: u32,
    active: AtomicUsize,
}
impl JitProcess {
    pub fn new(
        cpu: ProcessCpuContext,
        memory: Arc<ExecutionMemory>,
        timer_frequency: u64,
    ) -> Result<Self, JitError> {
        if memory
            .direct_address_space_view(cpu.address_space_id())
            .is_none()
        {
            return Err(JitError("Dynarmic requires a direct memory arena".into()));
        }
        let frequency = u32::try_from(timer_frequency)
            .ok()
            .filter(|f| *f != 0)
            .ok_or_else(|| {
                JitError("architectural timer frequency must fit a nonzero u32".into())
            })?;
        let monitor = unsafe { ffi::nixe_dynarmic_monitor_create(64) };
        if monitor.is_null() {
            return Err(JitError(ffi::error()));
        }
        Ok(Self {
            cpu,
            memory,
            monitor: Arc::new(Monitor(monitor)),
            cores: Mutex::new(Vec::new()),
            stopped: AtomicBool::new(false),
            active: AtomicUsize::new(0),
            frequency,
        })
    }
    pub fn request_stop(&self) -> Result<(), JitError> {
        self.stopped.store(true, Ordering::Release);
        for core in self.cores.lock().unwrap().iter().filter_map(Weak::upgrade) {
            unsafe { ffi::nixe_dynarmic_halt(core.pointer) };
        }
        Ok(())
    }
    pub fn try_shutdown(&self) -> Result<bool, JitError> {
        self.request_stop()?;
        Ok(self.active.load(Ordering::Acquire) == 0)
    }
}

/// Only diagnostic/replay consumers request an eager compact context. SVC
/// arguments/results use the retained ThreadState directly on the stopped core.
pub struct JitRunRequest<'a> {
    pub state: &'a mut ThreadState,
    pub instruction_budget: u64,
    pub timer: &'a dyn ArchitecturalTimer,
    pub events: &'a VcpuEventState,
    pub capture_context: bool,
}

pub struct JitThread {
    process: Arc<JitProcess>,
    native: Arc<Native>,
    control: CpuControl,
    invalidations: invalidation::Invalidations,
    blocked_fetches: HashSet<u64>,
}
impl JitThread {
    pub fn new(process: Arc<JitProcess>) -> Result<Self, JitError> {
        let arena = process
            .memory
            .direct_address_space_view(process.cpu.address_space_id())
            .unwrap();
        if !arena.address_space_size.is_power_of_two() {
            return Err(JitError(
                "Dynarmic arena size must be a power of two".into(),
            ));
        }
        let mut cores = process.cores.lock().unwrap();
        if process.stopped.load(Ordering::Acquire) {
            return Err(JitError("Dynarmic process is stopping".into()));
        }
        let id = cores
            .iter()
            .position(|c| c.strong_count() == 0)
            .unwrap_or(cores.len());
        if id >= 64 {
            return Err(JitError(
                "Dynarmic process exceeds 64 execution cores".into(),
            ));
        }
        let register = |key| match nixe_cpu::semantics::a64::runtime_register_read(
            process.cpu.platform(),
            key,
        ) {
            Some(nixe_cpu::semantics::a64::RuntimeRegisterRead::Constant(value)) => value as u32,
            _ => unreachable!("platform defines CTR_EL0 and DCZID_EL0"),
        };
        let pointer = unsafe {
            ffi::nixe_dynarmic_create(
                callbacks::table(),
                process.monitor.0,
                id,
                arena.base,
                arena.address_space_size.trailing_zeros() as usize,
                process.frequency,
                register(0xd53b_0020),
                register(0xd53b_00e0),
            )
        };
        if pointer.is_null() {
            return Err(JitError(ffi::error()));
        }
        let native = Arc::new(Native {
            pointer,
            _process: process.clone(),
            context_owned: AtomicBool::new(false),
        });
        if id == cores.len() {
            cores.push(Arc::downgrade(&native));
        } else {
            cores[id] = Arc::downgrade(&native);
        }
        drop(cores);
        let interrupt = native.clone();
        let control = CpuControl::with_interrupt(Arc::new(move || unsafe {
            ffi::nixe_dynarmic_halt(interrupt.pointer)
        }));
        let cursor = process.memory.invalidation_cursor();
        Ok(Self {
            process,
            native,
            control,
            invalidations: invalidation::Invalidations::new(cursor),
            blocked_fetches: HashSet::new(),
        })
    }
    pub fn control(&self) -> CpuControl {
        self.control.clone()
    }
    pub fn clear_local_exclusive_reservation(&mut self) {
        unsafe { ffi::nixe_dynarmic_clear_exclusive(self.native.pointer) }
    }
    pub fn synchronize_address_space(
        &mut self,
        binding: MemoryBinding<'_>,
    ) -> Result<(), JitError> {
        if binding.address_space != self.process.cpu.address_space_id()
            || binding.memory.execution_gate_identity()
                != self.process.memory.execution_gate_identity()
        {
            return Err(JitError(
                "Dynarmic execution memory differs from its process binding".into(),
            ));
        }
        let process = &self.process;
        let _lease = process.memory.acquire_execution_lease();
        self.invalidations
            .synchronize(&self.process, &self.native)
            .map_err(JitError)?;
        self.control
            .acknowledge_invalidation(self.invalidations.cursor().get());
        Ok(())
    }
    // Keep full snapshots and checked completion frames off resident RAM/SVC
    // entries. Only interrupted ordinary stores and MMIO need this consumer.
    #[cold]
    fn complete_cold_instruction(
        &self,
        state: &mut ThreadState,
        visibility: Option<nixe_memory::CanonicalBackingPage>,
        store_resume: Option<GuestVirtualAddress>,
        timer: &dyn ArchitecturalTimer,
        events: &VcpuEventState,
        progress: u64,
    ) -> Result<(nixe_cpu_interpreter::InstructionStep, u64), CpuFault> {
        let mut saved = state.snapshot();
        let word = match self.process.memory.fetch32(
            self.process.cpu.address_space_id(),
            GuestVirtualAddress::new(saved.pc()),
        ) {
            Ok(word) => word,
            Err(fault) => {
                return Ok((
                    nixe_cpu_interpreter::InstructionStep::Exit(CpuExit::FetchFault { fault }),
                    0,
                ));
            }
        };
        if let Some(page) = visibility {
            page.prepare_cpu_access()
                .map_err(|error| fault(error.to_string(), state, progress))?;
        }
        let store = store_resume.map(|address| {
            store_continuation::StoreContinuation::new(self.process.memory.as_ref(), address)
        });
        let memory: &dyn CpuMemory = store
            .as_ref()
            .map_or(self.process.memory.as_ref() as &dyn CpuMemory, |store| {
                store
            });
        let monitor = std::cell::RefCell::new(Default::default());
        let context = nixe_cpu_interpreter::InterpreterContext::new(
            self.process.cpu,
            memory,
            &monitor,
            timer,
            events,
        );
        let result = nixe_cpu_interpreter::execute_one_with_context(context, &mut saved, word.bits)
            .map_err(|error| fault(error.to_string(), state, progress))?;
        if store.as_ref().is_some_and(|store| !store.reached()) {
            return Err(fault(
                "checked store did not reach its native continuation",
                state,
                progress,
            ));
        }
        *state = saved.into();
        Ok((result, 1))
    }

    fn invalidate(native: &Native, address: u64, size: usize) -> Result<(), Box<str>> {
        record(Counter::InvalidationCalls, 1);
        record(Counter::InvalidationBytes, size as u64);
        record(Counter::FullInvalidations, u64::from(size == 0));
        let _measurement = Timer::new(Counter::InvalidationNanoseconds);
        if unsafe { ffi::nixe_dynarmic_invalidate(native.pointer, address, size) } {
            Ok(())
        } else {
            Err(ffi::error())
        }
    }
    pub fn run_slice(
        &mut self,
        worker: &mut nixe_cpu_direct_memory::NativeWorker,
        request: JitRunRequest<'_>,
    ) -> Result<ExecutionReport, CpuFault> {
        let JitRunRequest {
            state,
            instruction_budget,
            timer,
            events,
            capture_context,
        } = request;
        if !state.is_available() {
            return Err(unavailable_state_fault(
                "architectural state is unavailable",
                0,
            ));
        }
        let process = &self.process;
        process.active.fetch_add(1, Ordering::AcqRel);
        struct Active<'a>(&'a AtomicUsize);
        impl Drop for Active<'_> {
            fn drop(&mut self) {
                self.0.fetch_sub(1, Ordering::AcqRel);
            }
        }
        let _active = Active(&process.active);
        let mut progress = 0;
        if instruction_budget == 0 {
            return Ok(report(CpuExit::BudgetExhausted, 0, state, capture_context));
        }
        if instruction_budget > i64::MAX as u64 {
            return Err(fault("instruction budget exceeds i64::MAX", state, 0));
        }
        loop {
            // Clear before consuming scheduler requests, so a racing request is
            // either observed here or remains set in Dynarmic's atomic halt word.
            unsafe { ffi::nixe_dynarmic_clear_halt(self.native.pointer) };
            if self.process.stopped.load(Ordering::Acquire) {
                state.discard();
                return Err(unavailable_state_fault(
                    "Dynarmic process is stopping",
                    progress,
                ));
            }
            if let Some(pending) = self.control.take_pending() {
                // CodeInvalidation is acknowledged only after the memory log
                // has been accepted by the native cache, never on preemption.
                if pending.contains(ControlRequest::Preempt) {
                    return Ok(report(CpuExit::Safepoint, progress, state, capture_context));
                }
            }
            let mask = events.take_pending_interrupts();
            if mask != 0 {
                return Ok(report(
                    CpuExit::PendingEvent { mask },
                    progress,
                    state,
                    capture_context,
                ));
            }
            if progress >= instruction_budget {
                return Ok(report(
                    CpuExit::BudgetExhausted,
                    progress,
                    state,
                    capture_context,
                ));
            }
            let lease = process.memory.acquire_execution_lease();
            self.invalidations
                .synchronize(process, &self.native)
                .map_err(|e| fault(e, state, progress))?;
            self.control
                .acknowledge_invalidation(self.invalidations.cursor().get());
            state
                .select(&self.native)
                .map_err(|error| fault(error.to_string(), state, progress))?;
            let active = self.control.enter_execution();
            let mut context = callbacks::Context {
                memory: &process.memory,
                cpu: process.cpu,
                timer,
                blocked_fetches: &mut self.blocked_fetches,
                visibility: None,
                device_access: false,
                store_resume: None,
                data_fault: None,
                fatal: None,
            };
            let mut exit = ffi::Exit::default();
            record(Counter::NativeEntries, 1);
            let native_measurement = Timer::new(Counter::NativeRunNanoseconds);
            let ok = invocation::run(
                worker,
                process
                    .memory
                    .direct_address_space_view(process.cpu.address_space_id())
                    .unwrap(),
                self.native.pointer,
                &mut context,
                instruction_budget - progress,
                &mut exit,
            )
            .map_err(|e| {
                state.discard();
                let _ = self.process.request_stop();
                unavailable_state_fault(e, progress)
            })?;
            drop(native_measurement);
            record(Counter::GuestTicks, exit.ticks);
            if !ok {
                // A C++ failure does not establish stopped register authority.
                // Discard the owner rather than publishing the old save area.
                state.discard();
                let message = ffi::error();
                let _ = self.process.request_stop();
                return Err(unavailable_state_fault(message, progress));
            }
            let data_fault = context.data_fault;
            let fatal = context.fatal;
            let visibility = context.visibility;
            let device_access = context.device_access;
            let store_resume = context.store_resume;
            drop(active);
            drop(lease);
            progress = progress.saturating_add(exit.ticks);
            if let Some(error) = fatal {
                return Err(fault(error, state, progress));
            }
            if store_resume.is_none()
                && let Some(page) = visibility
            {
                page.prepare_cpu_access()
                    .map_err(|error| fault(error.to_string(), state, progress))?;
                continue;
            }
            if device_access || store_resume.is_some() {
                let (step, ticks) = self.complete_cold_instruction(
                    state,
                    visibility,
                    store_resume,
                    timer,
                    events,
                    progress,
                )?;
                progress = progress.saturating_add(ticks);
                match step {
                    nixe_cpu_interpreter::InstructionStep::Continue => continue,
                    nixe_cpu_interpreter::InstructionStep::Exit(stop) => {
                        let precise =
                            matches!(stop, CpuExit::FetchFault { .. } | CpuExit::DataFault { .. });
                        return Ok(report(stop, progress, state, capture_context || precise));
                    }
                }
            }
            let source = self.location(if exit.kind == 0 { state.pc() } else { exit.pc });
            if let Some(fault) = data_fault {
                return Ok(report(
                    CpuExit::DataFault {
                        source: self.location(state.pc()),
                        fault,
                    },
                    progress,
                    state,
                    true,
                ));
            }
            let stop = match exit.kind {
                0 => {
                    // The interrupted core has not consumed newer memory records.
                    // Retire scheduler requests, leaving their invalidation epoch
                    // unacknowledged until synchronization on resume or shutdown.
                    let _ = self.control.take_pending();
                    if progress >= instruction_budget {
                        CpuExit::BudgetExhausted
                    } else {
                        CpuExit::Safepoint
                    }
                }
                1 => self.unsupported(exit.pc),
                2 => CpuExit::SupervisorCall {
                    source,
                    immediate: exit.detail,
                },
                3 => match exit.detail {
                    3 => CpuExit::Scheduled {
                        source,
                        request: SchedulerRequest::WaitForInterrupt,
                    },
                    4 if events.consume_event() => continue,
                    4 => CpuExit::Scheduled {
                        source,
                        request: SchedulerRequest::WaitForEvent,
                    },
                    5 => CpuExit::Scheduled {
                        source,
                        request: SchedulerRequest::SendEvent,
                    },
                    6 => {
                        events.signal_event();
                        continue;
                    }
                    7 => CpuExit::Scheduled {
                        source,
                        request: SchedulerRequest::Yield,
                    },
                    8 => CpuExit::ArchitecturalException {
                        source,
                        kind: nixe_cpu::exception::ExceptionKind::Breakpoint,
                        syndrome: self
                            .process
                            .memory
                            .fetch32(self.process.cpu.address_space_id(), source.pc)
                            .ok()
                            .map(|word| u64::from((word.bits >> 5) & 0xffff)),
                    },
                    9 => match self
                        .process
                        .memory
                        .fetch32(self.process.cpu.address_space_id(), source.pc)
                    {
                        Err(fault) => CpuExit::FetchFault { fault },
                        Ok(_) if self.blocked_fetches.remove(&exit.pc) => {
                            Self::invalidate(&self.native, exit.pc, 4)
                                .map_err(|error| fault(error, state, progress))?;
                            continue;
                        }
                        Ok(_) => {
                            return Err(fault(
                                "Dynarmic reported a fetch fault for readable code",
                                state,
                                progress,
                            ));
                        }
                    },
                    _ => self.unsupported(exit.pc),
                },
                4 | 5 => {
                    // EL0 cannot issue set/way or whole-I-cache operations.
                    // DC ZVA also respects the selected platform's DCZID.DZP.
                    // https://developer.arm.com/documentation/ddi0601/2025-12/AArch64-Registers/DCZID-EL0--Data-Cache-Zero-ID-Register
                    let trapped = (exit.kind == 4
                        && (matches!(exit.detail, 0 | 2 | 6)
                            || (exit.detail == 8
                                && process.cpu.platform().user_cache_maintenance_prohibited())))
                        || (exit.kind == 5 && exit.detail != 0);
                    if trapped {
                        state.set_pc(state.pc().wrapping_sub(4));
                        return Ok(report(
                            CpuExit::ArchitecturalException {
                                source: self.location(state.pc()),
                                kind: nixe_cpu::exception::ExceptionKind::SystemRegisterTrap,
                                syndrome: None,
                            },
                            progress,
                            state,
                            true,
                        ));
                    }
                    if let Err(fault) = callbacks::maintain(&process.memory, process.cpu, &exit) {
                        state.set_pc(state.pc().wrapping_sub(4));
                        return Ok(report(
                            CpuExit::DataFault {
                                source: self.location(state.pc()),
                                fault,
                            },
                            progress,
                            state,
                            true,
                        ));
                    }
                    continue;
                }
                _ => return Err(fault("unknown Dynarmic exit", state, progress)),
            };
            return Ok(report(stop, progress, state, capture_context));
        }
    }
    fn location(&self, pc: u64) -> LocationDescriptor {
        LocationDescriptor::new(GuestVirtualAddress::new(pc), self.process.cpu.profile_id())
    }
    fn unsupported(&self, pc: u64) -> CpuExit {
        match self.process.memory.fetch32(
            self.process.cpu.address_space_id(),
            GuestVirtualAddress::new(pc),
        ) {
            Err(fault) => CpuExit::FetchFault { fault },
            Ok(word) => {
                use nixe_cpu::decode::{self, DecodeResult};
                match decode::decode(
                    self.process.cpu.decoder(),
                    self.location(pc),
                    word.bits.into(),
                ) {
                    DecodeResult::Decoded(decoded)
                    | DecodeResult::RecognizedUnimplemented(decoded) => {
                        CpuExit::UnsupportedSemantics {
                            source: decoded.location,
                            encoding: decoded.encoding,
                            disassembly: format!(
                                "{} (unsupported by Dynarmic A64)",
                                decode::disassemble(&decoded.instruction)
                            )
                            .into(),
                            coverage_id: decoded.instruction.coverage_id(),
                        }
                    }
                    DecodeResult::Unallocated {
                        instruction,
                        reason,
                    }
                    | DecodeResult::Reserved {
                        instruction,
                        reason,
                        ..
                    } => CpuExit::UnallocatedEncoding {
                        error: nixe_cpu::error::UnallocatedEncoding::new(instruction, reason),
                    },
                }
            }
        }
    }
}
fn fault(message: impl Into<Box<str>>, state: &ThreadState, progress: u64) -> CpuFault {
    CpuFault {
        backend: "dynarmic",
        kind: CpuFaultKind::Internal,
        progress,
        message: message.into(),
        context: Some(Box::new(state.register_context())),
    }
}
fn unavailable_state_fault(message: impl Into<Box<str>>, progress: u64) -> CpuFault {
    CpuFault {
        backend: "dynarmic",
        kind: CpuFaultKind::Internal,
        progress,
        message: message.into(),
        context: None,
    }
}
fn report(
    stop: CpuExit,
    progress: u64,
    state: &ThreadState,
    capture_context: bool,
) -> ExecutionReport {
    let capture_context = capture_context
        || !matches!(
            stop,
            CpuExit::SupervisorCall { .. }
                | CpuExit::BudgetExhausted
                | CpuExit::Safepoint
                | CpuExit::PendingEvent { .. }
                | CpuExit::Scheduled { .. }
        );
    ExecutionReport {
        stop,
        progress,
        context: capture_context.then(|| state.register_context()),
    }
}
