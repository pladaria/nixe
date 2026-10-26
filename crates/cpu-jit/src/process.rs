//! Canonical process/vCPU execution. Only a real native memory dependency parks
//! this worker; C++ frames and address-space admission have ended before it waits.
use crate::{
    JitError, JitRunRequest,
    execution::{Configuration, Core, Event},
};
use nixe_cpu::{
    decode::{self, DecodeResult},
    error::UnallocatedEncoding,
    exception::ExceptionKind,
    execution::{ControlRequest, CpuControl, CpuExit, CpuFault, ExecutionReport, SchedulerRequest},
    location::LocationDescriptor,
    memory::{DataAccessFault, DataAccessFaultReason},
    profile::ProcessCpuContext,
};
use nixe_cpu_direct_memory::NativeWorker;
use nixe_memory::{AddressSpace, GuestVirtualAddress, MemoryError};
use std::sync::{
    Arc, Condvar, Mutex, OnceLock, Weak,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

pub struct Process {
    cpu: ProcessCpuContext,
    space: Arc<AddressSpace>,
    frequency: u32,
    stopped: AtomicBool,
    active: AtomicUsize,
    controls: Mutex<Vec<CpuControl>>,
    waits: Mutex<Vec<Weak<Notification>>>,
    failure: OnceLock<Arc<MemoryError>>,
}
pub struct Thread {
    process: Arc<Process>,
    core: Core,
}
#[derive(Default)]
struct Notification {
    ready: Mutex<bool>,
    changed: Condvar,
}
impl Notification {
    fn signal(&self) {
        *self.ready.lock().unwrap() = true;
        self.changed.notify_one();
    }
    fn waker(self: &Arc<Self>) -> Arc<dyn Fn() + Send + Sync> {
        let notification = Arc::clone(self);
        Arc::new(move || notification.signal())
    }
}
impl Process {
    pub fn new(
        cpu: ProcessCpuContext,
        space: Arc<AddressSpace>,
        timer_frequency: u64,
    ) -> Result<Arc<Self>, JitError> {
        if cpu.address_space_id() != space.identity() {
            return Err(JitError(
                "CPU process belongs to another address space".into(),
            ));
        }
        let frequency = u32::try_from(timer_frequency)
            .ok()
            .filter(|value| *value != 0)
            .ok_or_else(|| {
                JitError("architectural timer frequency must fit a nonzero u32".into())
            })?;
        Ok(Arc::new(Self {
            cpu,
            space,
            frequency,
            stopped: AtomicBool::new(false),
            active: AtomicUsize::new(0),
            controls: Mutex::new(Vec::new()),
            waits: Mutex::new(Vec::new()),
            failure: OnceLock::new(),
        }))
    }
    pub fn create_thread(self: &Arc<Self>) -> Result<Thread, JitError> {
        if self.stopped.load(Ordering::Acquire) {
            return Err(JitError("JIT process is stopping".into()));
        }
        let core = Core::new(
            Arc::clone(&self.space),
            Configuration {
                cpu: self.cpu,
                timer_frequency: self.frequency,
            },
        )?;
        let mut controls = self.controls.lock().unwrap();
        if self.stopped.load(Ordering::Acquire) {
            return Err(JitError("JIT process is stopping".into()));
        }
        controls.push(core.control());
        Ok(Thread {
            process: Arc::clone(self),
            core,
        })
    }
    fn fail(&self, error: Arc<MemoryError>) -> Arc<MemoryError> {
        let _ = self.failure.set(error);
        self.request_stop();
        Arc::clone(self.failure.get().unwrap())
    }
    pub fn request_stop(&self) {
        self.stopped.store(true, Ordering::Release);
        let controls = self.controls.lock().unwrap().clone();
        for control in controls {
            control.request(ControlRequest::Preempt);
        }
        let waits: Vec<_> = self
            .waits
            .lock()
            .unwrap()
            .iter()
            .filter_map(Weak::upgrade)
            .collect();
        for wait in waits {
            wait.signal();
        }
    }
    pub fn try_shutdown(&self) -> Result<bool, MemoryError> {
        self.request_stop();
        if self.active.load(Ordering::Acquire) != 0 {
            return Ok(false);
        }
        self.space.close()?;
        Ok(true)
    }
    fn notification(&self) -> Arc<Notification> {
        let notification = Arc::new(Notification::default());
        let mut waits = self.waits.lock().unwrap();
        waits.retain(|wait| wait.strong_count() != 0);
        waits.push(Arc::downgrade(&notification));
        notification
    }
    fn wait<T>(
        &self,
        notification: &Notification,
        mut poll: impl FnMut() -> Option<Result<T, Arc<MemoryError>>>,
    ) -> Result<T, Arc<MemoryError>> {
        loop {
            if self.stopped.load(Ordering::Acquire) {
                return Err(self.failure.get().cloned().unwrap_or_else(|| {
                    Arc::new(MemoryError::Execution(
                        "JIT process stopped during a memory dependency".into(),
                    ))
                }));
            }
            if let Some(result) = poll() {
                return result;
            }
            let mut ready = notification.ready.lock().unwrap();
            if !*ready {
                ready = notification.changed.wait(ready).unwrap();
            }
            *ready = false;
        }
    }
}
struct Active<'a>(&'a AtomicUsize);
impl Drop for Active<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}
impl Thread {
    pub fn control(&self) -> CpuControl {
        self.core.control()
    }
    pub fn clear_local_exclusive_reservation(&mut self) {
        self.core.clear_exclusive();
    }
    pub fn run_slice(
        &mut self,
        worker: &mut NativeWorker,
        request: JitRunRequest<'_>,
    ) -> Result<ExecutionReport, CpuFault> {
        let JitRunRequest {
            state,
            instruction_budget,
            timer,
            events,
        } = request;
        let process = &self.process;
        process.active.fetch_add(1, Ordering::AcqRel);
        let _active = Active(&process.active);
        let mut progress = 0u64;
        if !state.is_available() || process.stopped.load(Ordering::Acquire) {
            state.discard();
            return Err(crate::unavailable_state_fault(
                "JIT process/context is unavailable",
                progress,
            ));
        }
        if instruction_budget > i64::MAX as u64 {
            return Err(crate::fault(
                "instruction budget exceeds i64::MAX",
                state,
                progress,
            ));
        }
        if instruction_budget == 0 {
            return Ok(crate::report(CpuExit::BudgetExhausted, progress, state));
        }
        self.core
            .select_state(state)
            .map_err(|error| crate::fault(error.to_string(), state, progress))?;
        loop {
            if process.stopped.load(Ordering::Acquire) {
                state.discard();
                return Err(crate::unavailable_state_fault(
                    "JIT process is stopping",
                    progress,
                ));
            }
            let interrupts = events.take_pending_interrupts();
            if interrupts != 0 {
                return Ok(crate::report(
                    CpuExit::PendingEvent { mask: interrupts },
                    progress,
                    state,
                ));
            }
            if progress >= instruction_budget {
                return Ok(crate::report(CpuExit::BudgetExhausted, progress, state));
            }
            let boundary = match self.core.run(worker, instruction_budget - progress, timer) {
                Ok(boundary) => boundary,
                Err(error) => {
                    let error =
                        process.fail(Arc::new(MemoryError::Execution(error.to_string().into())));
                    state.discard();
                    return Err(crate::unavailable_state_fault(error.to_string(), progress));
                }
            };
            progress = progress.saturating_add(boundary.ticks);
            if let Some(control) = boundary.control {
                if control.contains(ControlRequest::Preempt) {
                    return Ok(crate::report(CpuExit::Safepoint, progress, state));
                }
                continue;
            }
            if process.stopped.load(Ordering::Acquire) {
                state.discard();
                return Err(crate::unavailable_state_fault(
                    "JIT process is stopping",
                    progress,
                ));
            }
            if !boundary.code.is_empty() {
                for capture in boundary.code {
                    let mut read = capture
                        .start(&process.space)
                        .map_err(|error| crate::fault(error.to_string(), state, progress))?;
                    let resolved = if let Some(result) = read.poll() {
                        result
                    } else {
                        let notification = process.notification();
                        read.set_waker(notification.waker());
                        process.wait(&notification, || read.poll())
                    };
                    let resolved = match resolved {
                        Ok(resolved) => resolved,
                        Err(error) => {
                            let error = process.fail(read.cancel().unwrap_or(error));
                            state.discard();
                            return Err(crate::unavailable_state_fault(
                                error.to_string(),
                                progress,
                            ));
                        }
                    };
                    self.core
                        .install_code(resolved)
                        .map_err(|error| crate::fault(error.to_string(), state, progress))?;
                }
                continue;
            }
            if let Some(mut pending) = boundary.memory {
                let completed = if let Some(result) = pending.poll(process.space.machine()) {
                    result
                } else {
                    let notification = process.notification();
                    pending.set_waker(notification.waker());
                    process.wait(&notification, || pending.poll(process.space.machine()))
                };
                let completed = match completed {
                    Ok(completed) => completed,
                    Err(error) => {
                        let error = process.fail(pending.cancel().unwrap_or(error));
                        state.discard();
                        return Err(crate::unavailable_state_fault(error.to_string(), progress));
                    }
                };
                if process.stopped.load(Ordering::Acquire) {
                    state.discard();
                    return Err(crate::unavailable_state_fault(
                        "JIT process is stopping",
                        progress,
                    ));
                }
                let (step, ticks) = self
                    .core
                    .complete(completed)
                    .map_err(|error| crate::fault(error.to_string(), state, progress))?;
                progress = progress.saturating_add(ticks);
                match step {
                    None => continue,
                    Some(stop) => {
                        return Ok(crate::report(stop, progress, state));
                    }
                }
            }
            let location = |pc| {
                LocationDescriptor::new(GuestVirtualAddress::new(pc), process.cpu.profile_id())
            };
            if let Some(fault) = boundary.fault {
                state.set_pc(fault.pc.get());
                let address = match &fault.cause {
                    MemoryError::Unmapped(address)
                    | MemoryError::PermissionDenied { address, .. } => *address,
                    _ => fault.address,
                };
                return Ok(crate::report(
                    CpuExit::DataFault {
                        source: location(fault.pc.get()),
                        fault: DataAccessFault::new(
                            process.space.identity(),
                            address,
                            fault.kind,
                            DataAccessFaultReason::from(&fault.cause),
                        ),
                    },
                    progress,
                    state,
                ));
            }
            if let Some(fault) = boundary.fetch_fault {
                return Ok(crate::report(
                    CpuExit::FetchFault { fault },
                    progress,
                    state,
                ));
            }
            if let Some(start) = boundary.invalidate_code {
                process
                    .space
                    .invalidate_code(start, 64)
                    .map_err(|error| crate::fault(error.to_string(), state, progress))?;
                continue;
            }
            let event = boundary.event;
            let source = location(event.pc);
            let stop = match event.kind {
                0 => {
                    if progress >= instruction_budget {
                        CpuExit::BudgetExhausted
                    } else {
                        // Mapping/tracking quiescence does not end the guest
                        // quantum. Re-enter after publication; Core::run checks
                        // real scheduler requests before native execution.
                        crate::metrics::record(crate::metrics::Counter::InternalHaltResumes, 1);
                        continue;
                    }
                }
                2 => CpuExit::SupervisorCall {
                    source,
                    immediate: event.detail,
                },
                3 => match event.detail {
                    3 => CpuExit::Scheduled {
                        source,
                        request: SchedulerRequest::WaitForInterrupt,
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
                        kind: ExceptionKind::Breakpoint,
                        syndrome: event.word.map(|word| u64::from((word >> 5) & 0xffff)),
                    },
                    _ => unsupported(process.cpu, event)?,
                },
                4 | 5 => {
                    let trapped = (event.kind == 4
                        && (matches!(event.detail, 0 | 2 | 6)
                            || (event.detail == 8
                                && process.cpu.platform().user_cache_maintenance_prohibited())))
                        || (event.kind == 5 && event.detail != 0);
                    state.set_pc(event.pc);
                    if trapped {
                        CpuExit::ArchitecturalException {
                            source,
                            kind: ExceptionKind::SystemRegisterTrap,
                            syndrome: None,
                        }
                    } else {
                        return Err(crate::fault(
                            format!(
                                "unsupported native cache operation: kind={} detail={}",
                                event.kind, event.detail
                            ),
                            state,
                            progress,
                        ));
                    }
                }
                _ => {
                    return Err(crate::fault(
                        "unknown canonical native event",
                        state,
                        progress,
                    ));
                }
            };
            return Ok(crate::report(stop, progress, state));
        }
    }
}
fn unsupported(cpu: ProcessCpuContext, event: Event) -> Result<CpuExit, CpuFault> {
    let source = LocationDescriptor::new(GuestVirtualAddress::new(event.pc), cpu.profile_id());
    let Some(word) = event.word else {
        return Err(crate::unavailable_state_fault(
            "native exception has no captured opcode",
            0,
        ));
    };
    Ok(match decode::decode(cpu.decoder(), source, word.into()) {
        DecodeResult::Decoded(decoded) | DecodeResult::RecognizedUnimplemented(decoded) => {
            CpuExit::UnsupportedSemantics {
                source,
                encoding: decoded.encoding,
                disassembly: decode::disassemble(&decoded.instruction).to_string().into(),
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
            error: UnallocatedEncoding::new(instruction, reason),
        },
    })
}

#[cfg(test)]
mod tests;
