//! Process/vCPU execution with memory dependencies resolved by native callbacks.
use crate::{
    JitError, JitRunRequest,
    execution::{Configuration, Core, Event},
};
use nixe_cpu::{
    decode::{self, DecodeResult},
    error::UnallocatedEncoding,
    exception::ExceptionKind,
    execution::{CpuControl, CpuExit, CpuFault, ExecutionReport},
    location::LocationDescriptor,
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
#[derive(Debug)]
pub enum MemoryAccessFailure {
    Stopped,
    Memory(Arc<MemoryError>),
}
impl std::fmt::Display for MemoryAccessFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Stopped => f.write_str("JIT process stopped during a memory dependency"),
            Self::Memory(error) => error.fmt(f),
        }
    }
}
impl std::error::Error for MemoryAccessFailure {}
impl From<MemoryError> for MemoryAccessFailure {
    fn from(error: MemoryError) -> Self {
        Self::Memory(Arc::new(error))
    }
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
    fn stopped_fault(&self) -> CpuFault {
        if let Some(error) = self.failure.get() {
            return crate::unavailable_state_fault(error.to_string());
        }
        let mut fault = crate::unavailable_state_fault("JIT process is stopping");
        fault.kind = nixe_cpu::execution::CpuFaultKind::Stopped;
        fault
    }

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
    pub(crate) fn fail(&self, error: Arc<MemoryError>) -> Arc<MemoryError> {
        let _ = self.failure.set(error);
        self.request_stop();
        Arc::clone(self.failure.get().unwrap())
    }
    pub fn request_stop(&self) {
        self.stopped.store(true, Ordering::Release);
        let controls = self.controls.lock().unwrap().clone();
        for control in controls {
            control.request_preemption();
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
    pub(crate) fn resolve_access(
        &self,
        access: Result<nixe_memory::Access<nixe_memory::AccessResult>, MemoryError>,
    ) -> Result<nixe_memory::AccessResult, MemoryAccessFailure> {
        let mut pending = match access.map_err(MemoryAccessFailure::from)? {
            nixe_memory::Access::Ready(value) => return Ok(value),
            nixe_memory::Access::Pending(pending) => pending,
        };
        if let Some(result) = pending.poll() {
            return result.map_err(|error| MemoryAccessFailure::Memory(self.fail(error)));
        }
        let notification = self.notification();
        pending.set_waker(notification.waker());
        let result = self.wait(&notification, || pending.poll());
        match result {
            Ok(value) => Ok(value),
            Err(MemoryAccessFailure::Memory(error)) => {
                Err(MemoryAccessFailure::Memory(self.fail(error)))
            }
            Err(MemoryAccessFailure::Stopped) => Err(MemoryAccessFailure::Stopped),
        }
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
    ) -> Result<T, MemoryAccessFailure> {
        loop {
            if let Some(result) = poll() {
                return result.map_err(MemoryAccessFailure::Memory);
            }
            if self.stopped.load(Ordering::Acquire) {
                return Err(self
                    .failure
                    .get()
                    .cloned()
                    .map_or(MemoryAccessFailure::Stopped, MemoryAccessFailure::Memory));
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
    /// Kernel accesses on this core share native memory dependency cancellation.
    pub fn read_memory(
        &self,
        address: GuestVirtualAddress,
        bytes: usize,
    ) -> Result<u128, MemoryAccessFailure> {
        match self
            .process
            .space
            .read_value(address, bytes)
            .map_err(MemoryAccessFailure::from)?
        {
            nixe_memory::Access::Ready(value) => Ok(value),
            nixe_memory::Access::Pending(pending) => {
                match self
                    .process
                    .resolve_access(Ok(nixe_memory::Access::Pending(pending)))?
                {
                    nixe_memory::AccessResult::Read(value) => {
                        let mut scalar = [0; 16];
                        scalar[..value.len()].copy_from_slice(&value);
                        Ok(u128::from_le_bytes(scalar))
                    }
                    _ => unreachable!("scalar read completion is checked by memory"),
                }
            }
        }
    }

    pub fn write_memory(
        &self,
        address: GuestVirtualAddress,
        bytes: usize,
        value: u128,
    ) -> Result<(), MemoryAccessFailure> {
        match self
            .process
            .space
            .write_value(address, bytes, value)
            .map_err(MemoryAccessFailure::from)?
        {
            nixe_memory::Access::Ready(()) => Ok(()),
            nixe_memory::Access::Pending(pending) => {
                match self
                    .process
                    .resolve_access(Ok(nixe_memory::Access::Pending(pending)))?
                {
                    nixe_memory::AccessResult::Written => Ok(()),
                    _ => unreachable!("scalar write completion is checked by memory"),
                }
            }
        }
    }

    pub fn control(&self) -> CpuControl {
        self.core.control()
    }
    pub fn run_slice(
        &mut self,
        worker: &mut NativeWorker,
        request: JitRunRequest<'_>,
    ) -> Result<ExecutionReport, CpuFault> {
        let JitRunRequest {
            state,
            timer,
            events,
        } = request;
        let process = &self.process;
        process.active.fetch_add(1, Ordering::AcqRel);
        let _active = Active(&process.active);
        if !state.is_available() {
            state.discard();
            return Err(crate::unavailable_state_fault(
                "JIT process/context is unavailable",
            ));
        }
        if process.stopped.load(Ordering::Acquire) {
            state.discard();
            return Err(process.stopped_fault());
        }
        self.core
            .select_state(state)
            .map_err(|error| crate::fault(error.to_string(), state))?;
        self.core.clear_exclusive();
        loop {
            if process.stopped.load(Ordering::Acquire) {
                state.discard();
                return Err(process.stopped_fault());
            }
            let interrupts = events.take_pending_interrupts();
            if interrupts != 0 {
                return Ok(crate::report(
                    CpuExit::PendingEvent { mask: interrupts },
                    state,
                ));
            }
            let boundary = match self.core.run(worker, timer, process) {
                Ok(boundary) => boundary,
                Err(MemoryAccessFailure::Stopped) => {
                    // Native interruption is control flow, not a memory failure.
                    // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/arm/dynarmic/arm_dynarmic_64.cpp#L437-L439
                    state.discard();
                    return Err(process.stopped_fault());
                }
                Err(MemoryAccessFailure::Memory(error)) => {
                    let error = process.fail(error);
                    state.discard();
                    return Err(crate::unavailable_state_fault(error.to_string()));
                }
            };
            if boundary.preempted {
                return Ok(crate::report(CpuExit::Safepoint, state));
            }
            if process.stopped.load(Ordering::Acquire) {
                state.discard();
                return Err(process.stopped_fault());
            }
            if !boundary.code.is_empty() {
                for capture in boundary.code {
                    let mut read = capture
                        .start(&process.space)
                        .map_err(|error| crate::fault(error.to_string(), state))?;
                    let resolved = if let Some(result) = read.poll() {
                        result.map_err(MemoryAccessFailure::Memory)
                    } else {
                        let notification = process.notification();
                        read.set_waker(notification.waker());
                        process.wait(&notification, || read.poll())
                    };
                    let resolved = match resolved {
                        Ok(resolved) => resolved,
                        Err(MemoryAccessFailure::Stopped) => {
                            let fault = match read.cancel() {
                                Some(error) => {
                                    crate::unavailable_state_fault(process.fail(error).to_string())
                                }
                                None => process.stopped_fault(),
                            };
                            state.discard();
                            return Err(fault);
                        }
                        Err(MemoryAccessFailure::Memory(error)) => {
                            let error = process.fail(read.cancel().unwrap_or(error));
                            state.discard();
                            return Err(crate::unavailable_state_fault(error.to_string()));
                        }
                    };
                    self.core
                        .install_code(resolved)
                        .map_err(|error| crate::fault(error.to_string(), state))?;
                }
                continue;
            }
            let location = |pc| {
                LocationDescriptor::new(GuestVirtualAddress::new(pc), process.cpu.profile_id())
            };
            if let Some(fault) = boundary.fetch_fault {
                return Ok(crate::report(CpuExit::FetchFault { fault }, state));
            }
            let event = boundary.event;
            let source = location(event.pc);
            let stop = match event.kind {
                0 => {
                    crate::metrics::record(crate::metrics::Counter::InternalHaltResumes, 1);
                    continue;
                }
                1 => unsupported(process.cpu, event)?,
                2 => CpuExit::SupervisorCall {
                    source,
                    immediate: event.detail,
                },
                3 => match event.detail {
                    8 => CpuExit::ArchitecturalException {
                        source,
                        kind: ExceptionKind::Breakpoint,
                        syndrome: event.word.map(|word| u64::from((word >> 5) & 0xffff)),
                    },
                    _ => unsupported(process.cpu, event)?,
                },
                _ => {
                    return Err(crate::fault("unknown canonical native event", state));
                }
            };
            return Ok(crate::report(stop, state));
        }
    }
}
fn unsupported(cpu: ProcessCpuContext, event: Event) -> Result<CpuExit, CpuFault> {
    let source = LocationDescriptor::new(GuestVirtualAddress::new(event.pc), cpu.profile_id());
    let Some(word) = event.word else {
        return Err(crate::unavailable_state_fault(
            "native exception has no captured opcode",
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
