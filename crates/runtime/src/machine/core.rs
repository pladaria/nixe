//! A physical core selects guest fibers; each guest keeps its kernel call stack.
//! https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/core/cpu_manager.cpp
use super::*;
use crate::fiber::HostContext;
use crate::process::execution::{CoreExecution, CpuThread};
use crate::{ExceptionDispatchContext, ExceptionThreadContext, ExecutionStop, ThreadObject};
use corosensei::CoroutineResult;
use nixe_scheduler::{Completion, Lease};
use std::{cell::Cell, ptr};

thread_local! {
    static CURRENT: Cell<*mut PhysicalCore> = const { Cell::new(ptr::null_mut()) };
}
struct Binding(*mut PhysicalCore);
impl Binding {
    fn enter(core: &mut PhysicalCore) -> Self {
        Self(CURRENT.with(|current| current.replace(core)))
    }
}
impl Drop for Binding {
    fn drop(&mut self) {
        CURRENT.with(|current| current.set(self.0));
    }
}
/// The borrow ends before any fiber suspension. Kernel stacks never retain a
/// physical-core reference across migration, as Eden's CurrentPhysicalCore.
pub(crate) fn with_current<R>(f: impl FnOnce(&mut PhysicalCore) -> R) -> R {
    CURRENT.with(|current| {
        let pointer = current.get();
        assert!(
            !pointer.is_null(),
            "guest execution requires a physical core"
        );
        // SAFETY: the core binds itself during exclusive guest-context execution.
        // Every caller completes this borrow before yielding to the scheduler.
        f(unsafe { &mut *pointer })
    })
}
pub(crate) fn current_lease() -> Lease {
    with_current(|core| core.resident.as_ref().expect("selected guest fiber").lease)
}

/// Kernel user-memory accesses finish synchronously, including accepted atomic
/// updates. Only a real kernel object wait suspends the guest stack, as Eden's
/// address-arbiter and condition-variable ReadFromUser/WriteToUser paths.
/// Keep servicing native mapping interrupts while waiting for coherence.
/// https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/core/hle/kernel/k_condition_variable.cpp
pub(crate) fn wait_io<F: std::future::Future>(future: F) -> F::Output {
    use std::task::{Context, Poll, Waker};
    let mut future = std::pin::pin!(future);
    if let Poll::Ready(result) = future
        .as_mut()
        .poll(&mut Context::from_waker(Waker::noop()))
    {
        return result;
    }
    let signal = with_current(|core| core.signal.clone());
    let waker = Waker::from(signal.clone());
    loop {
        let observed = signal.sequence();
        with_current(|core| {
            core.admit_guest();
        });
        if let Poll::Ready(result) = future.as_mut().poll(&mut Context::from_waker(&waker)) {
            return result;
        }
        signal.wait(observed);
    }
}

struct Resident {
    lease: Lease,
    process: Arc<RunnableProcess>,
    thread: Arc<ThreadObject>,
    cpu: CpuThread,
    generation: u64,
    interrupted: u64,
    switch_requested: bool,
}
pub(crate) struct PhysicalCore {
    id: VirtualCpuId,
    shared: Arc<Shared>,
    signal: Arc<CoreSignal>,
    recovery: nixe_cpu_direct_memory::NativeWorker,
    native: BTreeMap<ProcessId, (Arc<RunnableProcess>, Option<CpuThread>)>,
    resident: Option<Resident>,
    clean_shutdown: bool,
    retiring: VecDeque<ProcessId>,
    serviced: u64,
    completions: u64,
}
pub(super) fn run<D: ExceptionDispatcher>(
    id: VirtualCpuId,
    index: usize,
    shared: Arc<Shared>,
    dispatcher: Arc<D>,
) {
    let signal = shared.signals[index].clone();
    let mut core = PhysicalCore {
        id,
        shared,
        completions: signal.completions(),
        signal,
        recovery: Default::default(),
        native: BTreeMap::new(),
        resident: None,
        clean_shutdown: false,
        retiring: VecDeque::new(),
        serviced: u64::MAX,
    };
    core.run(dispatcher);
}
impl PhysicalCore {
    fn service_control(&mut self) {
        if self.signal.sequence() != self.serviced {
            let (sequence, mut retirements) = self.signal.take_control();
            self.serviced = sequence;
            self.retiring.append(&mut retirements);
        }
        for _ in 0..self.retiring.len() {
            let process_id = self.retiring.pop_front().unwrap();
            if self
                .resident
                .as_ref()
                .is_some_and(|r| r.lease.process == process_id)
            {
                self.retiring.push_back(process_id);
            } else if let Some((process, cpu)) = self.native.remove(&process_id) {
                drop(cpu);
                process.retire_core(self.id);
                self.shared.status_changed();
            }
        }
    }
    fn run<D: ExceptionDispatcher>(&mut self, dispatcher: Arc<D>) {
        loop {
            self.service_control();
            let observed = self.signal.sequence();
            if self.shared.stopped.load(Ordering::Acquire) {
                break;
            }
            if self.shared.paused.load(Ordering::Acquire) {
                self.signal
                    .acknowledge(self.shared.pause_generation.load(Ordering::Acquire));
                self.signal.wait(observed);
                continue;
            }
            let lease = match self.shared.scheduler.claim(self.id) {
                Ok(Some(lease)) => lease,
                Ok(None) => {
                    self.signal.wait(observed);
                    continue;
                }
                Err(error) => {
                    self.shared.fail(error.into());
                    break;
                }
            };
            let process = self
                .shared
                .process(lease.process)
                .expect("registered ready process");
            let generation = process.generation();
            if process.lifecycle() != nixe_scheduler::ProcessLifecycle::Running {
                let _ = self.shared.scheduler.complete(lease, Completion::Ready);
                self.finish_termination(&process);
                continue;
            }
            let cpu = match self.native.entry(lease.process) {
                std::collections::btree_map::Entry::Occupied(mut entry) => {
                    Ok(entry.get_mut().1.take().expect("idle core interface"))
                }
                std::collections::btree_map::Entry::Vacant(entry) => {
                    process.create_core(self.id).inspect(|_| {
                        entry.insert((process.clone(), None));
                    })
                }
            };
            let cpu = match cpu {
                Ok(cpu) => cpu,
                Err(fault) => {
                    let _ = self.shared.scheduler.complete(lease, Completion::Ready);
                    if process.generation() == generation
                        || fault.kind != nixe_cpu::execution::CpuFaultKind::Stopped
                    {
                        self.shared.fail(RuntimeError::Execution {
                            process: lease.process,
                            thread: lease.thread,
                            diagnostic: fault.to_string().into(),
                        });
                        process.finish_process(nixe_scheduler::ProcessLifecycle::Faulted, None);
                    }
                    self.finish_termination(&process);
                    continue;
                }
            };
            let thread = process
                .thread(lease.thread)
                .expect("running kernel reference");
            let fiber = thread
                .take_host_context()
                .map_or_else(|| HostContext::new(thread.clone(), dispatcher.clone()), Ok);
            let mut fiber = match fiber {
                Ok(fiber) => fiber,
                Err(error) => {
                    self.native.get_mut(&lease.process).unwrap().1 = Some(cpu);
                    let _ = self.shared.scheduler.complete(lease, Completion::Faulted);
                    process.finish_process(nixe_scheduler::ProcessLifecycle::Faulted, None);
                    self.shared.fail(RuntimeError::Startup(error));
                    self.finish_termination(&process);
                    continue;
                }
            };
            self.resident = Some(Resident {
                lease,
                process: process.clone(),
                thread: thread.clone(),
                cpu,
                generation,
                interrupted: observed,
                switch_requested: false,
            });
            let result = {
                let _binding = Binding::enter(self);
                fiber.resume()
            };
            let (completion, suspended) = match result {
                CoroutineResult::Yield(completion) => (completion, true),
                CoroutineResult::Return(completion) => (completion, false),
            };
            let current = self.resident.take().unwrap();
            self.native.get_mut(&lease.process).unwrap().1 = Some(current.cpu);
            // Publish the saved stack before releasing its execution lease: a
            // concurrent wake may already have selected another physical core.
            if suspended && process.generation() == generation && !thread.is_signalled() {
                thread.restore_host_context(fiber);
            } else {
                drop(fiber);
            }
            if let Err(error) = self.shared.scheduler.complete(lease, completion) {
                self.shared.fail(error.into());
            }
            if matches!(completion, Completion::Exited | Completion::Faulted) {
                self.shared.finalize_thread_waits(&process, thread.id());
                thread.signal();
            }
            self.finish_termination(&process);
        }
        self.clean_shutdown = self.shared.stopped.load(Ordering::Acquire);
    }
    fn finish_termination(&self, process: &Arc<RunnableProcess>) {
        if let Err(error) = self.shared.finish_termination(process) {
            self.shared.fail(error);
        }
    }
    fn admit_guest(&mut self) -> bool {
        self.service_control();
        let current = self.resident.as_mut().expect("guest context owns a lease");
        let sequence = self.signal.sequence();
        if sequence != current.interrupted {
            if let Err(error) =
                current
                    .process
                    .handle_interrupt(&current.thread, self.id, &current.cpu)
            {
                let stopping = current.process.generation() != current.generation
                    && matches!(
                        error.downcast_ref::<nixe_cpu_jit::process::MemoryAccessFailure>(),
                        Some(nixe_cpu_jit::process::MemoryAccessFailure::Stopped)
                    );
                if !stopping {
                    self.shared.fail(RuntimeError::Execution {
                        process: current.lease.process,
                        thread: current.lease.thread,
                        diagnostic: error.to_string().into(),
                    });
                    current
                        .process
                        .finish_process(nixe_scheduler::ProcessLifecycle::Faulted, None);
                }
            }
            current.interrupted = sequence;
            current.switch_requested = self
                .shared
                .scheduler
                .selection(self.id)
                .expect("physical core")
                .thread
                != Some(current.thread.id());
        }
        !current.switch_requested
            && current.process.generation() == current.generation
            && !self.shared.stopped.load(Ordering::Acquire)
            && !self.shared.paused.load(Ordering::Acquire)
    }
    fn execute(
        &mut self,
        execution: &mut CoreExecution,
    ) -> Option<Result<crate::ExecutionReport, nixe_cpu::execution::CpuFault>> {
        let current = self.resident.as_mut().unwrap();
        self.signal.bind_native(Some(current.cpu.control()));
        if self.signal.sequence() != current.interrupted {
            self.signal.bind_native(None);
            return None;
        }
        let report = execution.run(&mut self.recovery, &mut current.cpu);
        self.signal.bind_native(None);
        self.completions = self.completions.wrapping_add(1);
        self.signal.record_completion(self.completions);
        Some(report)
    }
}

struct GuestExecution {
    thread: Arc<ThreadObject>,
    execution: CoreExecution,
}
impl Drop for GuestExecution {
    fn drop(&mut self) {
        if self.execution.thread.is_available() {
            self.execution.thread.leave_core();
            self.thread.restore_state(self.execution.thread.take());
        }
    }
}
pub(crate) fn run_guest<D: ExceptionDispatcher>(
    thread: Arc<ThreadObject>,
    dispatcher: Arc<D>,
) -> Completion {
    let process = thread.owner().expect("kernel thread owns process").clone();
    let generation = process.generation();
    let (clock, frequency) = process.execution_environment();
    let mut guest = GuestExecution {
        execution: CoreExecution {
            thread: thread
                .take_state()
                .expect("first guest context owns saved registers"),
            virtual_clock: clock,
            architectural_timer_frequency: frequency,
        },
        thread: thread.clone(),
    };
    loop {
        if process.generation() != generation {
            return Completion::Ready;
        }
        if !with_current(PhysicalCore::admit_guest) {
            crate::fiber::suspend(&thread, &mut guest.execution.thread, Completion::Ready);
            continue;
        }
        let Some(report) = with_current(|core| core.execute(&mut guest.execution)) else {
            continue;
        };
        let report = match report {
            Ok(report) => report,
            Err(error) => {
                let stopping = process.generation() != generation
                    && error.kind == nixe_cpu::execution::CpuFaultKind::Stopped;
                if !stopping {
                    with_current(|core| {
                        core.shared.fail(RuntimeError::Execution {
                            process: core.resident.as_ref().unwrap().lease.process,
                            thread: thread.id(),
                            diagnostic: error.to_string().into(),
                        })
                    });
                    process.finish_process(nixe_scheduler::ProcessLifecycle::Faulted, None);
                }
                return if stopping {
                    Completion::Ready
                } else {
                    Completion::Faulted
                };
            }
        };
        if process.generation() != generation {
            return Completion::Ready;
        }
        match report.stop {
            ExecutionStop::Safepoint => continue,
            ExecutionStop::SupervisorCall { source, immediate } => {
                if immediate == 7 && thread.loader_return == Some(source.pc) {
                    let exit_code =
                        guest
                            .execution
                            .thread
                            .read_x(nixe_cpu::state::a64::A64Register::General(
                                nixe_cpu::state::a64::A64GeneralRegister::new(0).unwrap(),
                            ));
                    process.finish_process(
                        nixe_scheduler::ProcessLifecycle::Exited,
                        Some(crate::ProcessExit {
                            cause: crate::ProcessExitCause::LoaderReturned,
                            exit_code,
                            source: Some(source),
                            thread_id: thread.thread_id(),
                            context: Some(Box::new(guest.execution.thread.register_context())),
                            frames: Box::new([]),
                        }),
                    );
                    return Completion::Ready;
                }
                let core = current_lease().vcpu;
                let observed = with_current(|core| core.signal.sequence());
                let thread_context = ExceptionThreadContext::new(
                    thread.id(),
                    core,
                    &thread,
                    &mut guest.execution.thread,
                    thread.tls_base,
                );
                let mut context =
                    ExceptionDispatchContext::new(source, &process, thread_context, generation);
                let request = crate::ExceptionDispatchRequest::new(
                    source,
                    nixe_cpu::exception::ExceptionKind::SupervisorCall,
                    Some(u64::from(immediate)),
                );
                let outcome = dispatcher
                    .dispatch(&mut context, request)
                    .map_fault(|fault| Box::new(fault) as DispatchFault);
                if let crate::ExceptionDispatchOutcome::Reject { diagnostic, level } = &outcome {
                    with_current(|core| {
                        core.shared.record_rejection(
                            &process,
                            thread.id(),
                            source,
                            generation,
                            *level,
                            diagnostic.to_string(),
                        )
                    });
                }
                match process.apply_supervisor_call_outcome(
                    &thread,
                    source,
                    &mut guest.execution.thread,
                    outcome,
                ) {
                    ExceptionHandlingResult::Resumed | ExceptionHandlingResult::Rejected(_) => {
                        with_current(|core| {
                            if core.signal.sequence() != observed
                                && let Err(error) = core
                                    .shared
                                    .scheduler
                                    .resume_kernel(core.resident.as_ref().unwrap().lease)
                            {
                                core.shared.fail(error.into());
                            }
                        });
                    }
                    ExceptionHandlingResult::Fault(fault) => {
                        with_current(|core| {
                            core.shared.fail(RuntimeError::Execution {
                                process: core.resident.as_ref().unwrap().lease.process,
                                thread: thread.id(),
                                diagnostic: format!(
                                    "Horizon SVC dispatch failed: {fault}; source=[{source}]"
                                )
                                .into(),
                            })
                        });
                        return Completion::Faulted;
                    }
                    ExceptionHandlingResult::Terminated { .. } => return Completion::Exited,
                }
            }
            _ => {
                with_current(|core| {
                    core.shared.fail(RuntimeError::Execution {
                        process: core.resident.as_ref().unwrap().lease.process,
                        thread: thread.id(),
                        diagnostic: report.to_string().into(),
                    })
                });
                process.finish_process(nixe_scheduler::ProcessLifecycle::Faulted, None);
                return Completion::Faulted;
            }
        }
    }
}
impl Drop for PhysicalCore {
    fn drop(&mut self) {
        self.signal.bind_native(None);
        if !self.clean_shutdown {
            self.shared.fail_core();
        }
        if let Some(current) = self.resident.take() {
            self.native.get_mut(&current.lease.process).unwrap().1 = Some(current.cpu);
            let _ = self
                .shared
                .scheduler
                .complete(current.lease, Completion::Faulted);
            current
                .process
                .finish_process(nixe_scheduler::ProcessLifecycle::Faulted, None);
            self.shared
                .finalize_thread_waits(&current.process, current.thread.id());
            current.thread.signal();
            self.finish_termination(&current.process);
        }
        for (_, (process, cpu)) in std::mem::take(&mut self.native) {
            drop(cpu);
            process.retire_core(self.id);
        }
        if let Err(error) = std::mem::take(&mut self.recovery).finish() {
            self.shared.fail(RuntimeError::Execution {
                process: ProcessId::new(0),
                thread: GuestThreadId::new(0),
                diagnostic: error.to_string().into(),
            });
        }
        self.shared.status_changed();
    }
}
