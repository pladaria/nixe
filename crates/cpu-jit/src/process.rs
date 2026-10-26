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
    ) -> Result<nixe_memory::AccessResult, Arc<MemoryError>> {
        let mut pending = match access.map_err(Arc::new)? {
            nixe_memory::Access::Ready(value) => return Ok(value),
            nixe_memory::Access::Pending(pending) => pending,
        };
        if let Some(result) = pending.poll() {
            return result.map_err(|error| self.fail(error));
        }
        let notification = self.notification();
        pending.set_waker(notification.waker());
        let result = self.wait(&notification, || pending.poll());
        match result {
            Ok(value) => Ok(value),
            Err(error) => Err(self.fail(error)),
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
    ) -> Result<T, Arc<MemoryError>> {
        loop {
            if let Some(result) = poll() {
                return result;
            }
            if self.stopped.load(Ordering::Acquire) {
                return Err(self.failure.get().cloned().unwrap_or_else(|| {
                    Arc::new(MemoryError::Execution(
                        "JIT process stopped during a memory dependency".into(),
                    ))
                }));
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
        if !state.is_available() || process.stopped.load(Ordering::Acquire) {
            state.discard();
            return Err(crate::unavailable_state_fault(
                "JIT process/context is unavailable",
            ));
        }
        self.core
            .select_state(state)
            .map_err(|error| crate::fault(error.to_string(), state))?;
        self.core.clear_exclusive();
        loop {
            if process.stopped.load(Ordering::Acquire) {
                state.discard();
                return Err(crate::unavailable_state_fault("JIT process is stopping"));
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
                Err(error) => {
                    let error =
                        process.fail(Arc::new(MemoryError::Execution(error.to_string().into())));
                    state.discard();
                    return Err(crate::unavailable_state_fault(error.to_string()));
                }
            };
            if boundary.preempted {
                return Ok(crate::report(CpuExit::Safepoint, state));
            }
            if process.stopped.load(Ordering::Acquire) {
                state.discard();
                return Err(crate::unavailable_state_fault("JIT process is stopping"));
            }
            if !boundary.code.is_empty() {
                for capture in boundary.code {
                    let mut read = capture
                        .start(&process.space)
                        .map_err(|error| crate::fault(error.to_string(), state))?;
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
