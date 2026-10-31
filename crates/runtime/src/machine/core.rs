use super::*;
use crate::process::execution::{CoreExecution, CpuThread};
use crate::{ExceptionDispatchContext, ExceptionThreadContext, ExecutionStop, ThreadObject};
use nixe_scheduler::{Completion, Lease};
use std::task::Poll;

struct Resident {
    lease: Lease,
    process: Arc<RunnableProcess>,
    thread: Arc<ThreadObject>,
    // The selected thread keeps its core's interface until a real context switch.
    // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/physical_core.cpp#L26-L161
    cpu: CpuThread,
    execution: CoreExecution,
    generation: u64,
    interrupted: u64,
}

struct PhysicalCore {
    id: VirtualCpuId,
    shared: Arc<Shared>,
    signal: Arc<crate::kernel_operation::CoreSignal>,
    queue: Arc<KernelQueue>,

    recovery: nixe_cpu_direct_memory::NativeWorker,
    native: BTreeMap<ProcessId, (Arc<RunnableProcess>, Option<CpuThread>)>,
    resident: Option<Resident>,
    clean_shutdown: bool,
    retiring: VecDeque<ProcessId>,
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
        queue: KernelQueue::new(Waker::from(signal.clone())),
        signal,

        recovery: Default::default(),
        native: BTreeMap::new(),
        resident: None,
        clean_shutdown: false,
        retiring: VecDeque::new(),
    };
    core.run(dispatcher);
}
impl PhysicalCore {
    fn run<D: ExceptionDispatcher>(&mut self, dispatcher: Arc<D>) {
        let id = self.id;
        let shared = &self.shared;
        let signal = &self.signal;
        let queue = &self.queue;
        let recovery = &mut self.recovery;
        let native = &mut self.native;
        let resident = &mut self.resident;
        let retiring = &mut self.retiring;
        let mut serviced = u64::MAX;
        let mut completions = signal.completions();
        loop {
            let mut observed = signal.sequence();
            if observed != serviced {
                let (sequence, mut retirements) = signal.take_control();
                observed = sequence;
                serviced = sequence;
                retiring.append(&mut retirements);
                drain(shared, queue);
            }
            if resident
                .as_ref()
                .is_some_and(|current| current.process.generation() != current.generation)
            {
                let current = resident.take().unwrap();
                let process = current.process.clone();
                release(shared, native, current, Completion::Ready);
                if let Err(error) = shared.finish_termination(&process) {
                    shared.fail(error);
                }
            }
            for _ in 0..retiring.len() {
                let process_id = retiring.pop_front().expect("owned retirement request");
                if resident
                    .as_ref()
                    .is_some_and(|current| current.lease.process == process_id)
                {
                    retiring.push_back(process_id);
                } else if let Some((process, cpu)) = native.remove(&process_id) {
                    drop(cpu);
                    process.retire_core(id);
                    shared.status_changed();
                }
            }
            let interrupt_sequence = signal.sequence();
            let mut rescheduled = false;
            if let Some(current) = resident.as_mut()
                && current.process.generation() == current.generation
                && interrupt_sequence != current.interrupted
            {
                if let Err(error) =
                    current
                        .process
                        .handle_interrupt(&current.thread, id, &current.cpu)
                {
                    let stopping = current.process.generation() != current.generation
                        && matches!(
                            error.downcast_ref::<nixe_cpu_jit::process::MemoryAccessFailure>(),
                            Some(nixe_cpu_jit::process::MemoryAccessFailure::Stopped)
                        );
                    if !stopping {
                        shared.fail(RuntimeError::Execution {
                            process: current.lease.process,
                            thread: current.lease.thread,
                            diagnostic: error.to_string().into(),
                        });
                        current
                            .process
                            .finish_process(nixe_scheduler::ProcessLifecycle::Faulted, None);
                    }
                }
                current.interrupted = interrupt_sequence;
                rescheduled = true;
            }
            if shared.stopped.load(Ordering::Acquire) {
                break;
            }
            if shared.paused.load(Ordering::Acquire) {
                if let Some(current) = resident.take() {
                    release(shared, native, current, Completion::Ready);
                    continue;
                }
                signal.acknowledge(shared.pause_generation.load(Ordering::Acquire));
                signal.wait(observed);
                continue;
            }
            if resident.as_ref().is_some_and(|current| {
                current.process.generation() != current.generation
                    || (rescheduled
                        && shared
                            .scheduler
                            .selection(id)
                            .expect("physical core")
                            .thread
                            != Some(current.lease.thread))
            }) {
                let current = resident.take().unwrap();
                let process = current.process.clone();
                release(shared, native, current, Completion::Ready);
                if let Err(error) = shared.finish_termination(&process) {
                    shared.fail(error);
                }
                continue;
            }
            if resident.is_none() {
                let lease = match shared.scheduler.claim(id) {
                    Ok(Some(lease)) => lease,
                    Ok(None) => {
                        signal.wait(observed);
                        continue;
                    }
                    Err(error) => {
                        shared.fail(error.into());
                        break;
                    }
                };
                let process = shared
                    .process(lease.process)
                    .expect("ready process is registered");
                let generation = process.generation();
                if process.lifecycle() != nixe_scheduler::ProcessLifecycle::Running {
                    let _ = shared.scheduler.complete(lease, Completion::Ready);
                    if let Err(error) = shared.finish_termination(&process) {
                        shared.fail(error);
                    }
                    continue;
                }
                let cpu = match native.entry(lease.process) {
                    std::collections::btree_map::Entry::Occupied(mut entry) => {
                        entry.get_mut().1.take().expect("idle native context")
                    }
                    std::collections::btree_map::Entry::Vacant(entry) => {
                        match process.create_core(id) {
                            Ok(cpu) => {
                                entry.insert((process.clone(), None));
                                cpu
                            }
                            Err(fault) => {
                                let _ = shared.scheduler.complete(lease, Completion::Ready);
                                let stopping = process.generation() != generation
                                    && fault.kind == nixe_cpu::execution::CpuFaultKind::Stopped;
                                if !stopping {
                                    shared.fail(RuntimeError::Execution {
                                        process: lease.process,
                                        thread: lease.thread,
                                        diagnostic: fault.to_string().into(),
                                    });
                                    process.finish_process(
                                        nixe_scheduler::ProcessLifecycle::Faulted,
                                        None,
                                    );
                                }
                                if let Err(error) = shared.finish_termination(&process) {
                                    shared.fail(error);
                                }
                                continue;
                            }
                        }
                    }
                };
                let thread = process
                    .thread(lease.thread)
                    .expect("running reference retains thread");
                let state = thread
                    .take_state()
                    .expect("claimed thread owns its saved registers");
                let (clock, frequency) = process.execution_environment();
                *resident = Some(Resident {
                    lease,
                    generation,
                    interrupted: observed,
                    process,
                    thread: thread.clone(),
                    cpu,
                    execution: CoreExecution {
                        thread: state,
                        virtual_clock: clock,
                        architectural_timer_frequency: frequency,
                    },
                });
                let current = resident.as_mut().unwrap();
                if current.process.generation() != current.generation {
                    continue;
                }
                if let Some(operation) = current.thread.take_wait() {
                    let source = operation.source();
                    let thread_context = ExceptionThreadContext::new(
                        current.thread.id(),
                        id,
                        &current.thread,
                        &mut current.execution.thread,
                        current.thread.tls_base,
                    );
                    let mut context = ExceptionDispatchContext::new(
                        source,
                        &current.process,
                        thread_context,
                        queue,
                        current.generation,
                    );
                    let outcome = operation.resume(&mut context);
                    if let Err(error) = finish_call(
                        shared,
                        native,
                        resident,
                        source,
                        outcome,
                        signal.sequence() != observed,
                    ) {
                        shared.fail(error);
                        break;
                    }
                    continue;
                }
            }
            let current = resident.as_mut().unwrap();
            let cpu = &mut current.cpu;
            signal.bind_native(Some(cpu.control()));
            // Check admission after publishing the native interrupt destination.
            if shared.paused.load(Ordering::Acquire)
                || shared.stopped.load(Ordering::Acquire)
                || signal.sequence() != observed
                || current.process.generation() != current.generation
            {
                signal.bind_native(None);
                continue;
            }
            let report = current.execution.run(recovery, cpu);
            signal.bind_native(None);
            completions = completions.wrapping_add(1);
            signal.record_completion(completions);
            let report = match report {
                Ok(report) => report,
                Err(error) => {
                    let current = resident.take().unwrap();
                    let process = current.process.clone();
                    let stopping = process.lifecycle() != nixe_scheduler::ProcessLifecycle::Running
                        && error.kind == nixe_cpu::execution::CpuFaultKind::Stopped;
                    if !stopping {
                        shared.fail(RuntimeError::Execution {
                            process: current.lease.process,
                            thread: current.lease.thread,
                            diagnostic: error.to_string().into(),
                        });
                        process.finish_process(nixe_scheduler::ProcessLifecycle::Faulted, None);
                    }
                    release(shared, native, current, Completion::Ready);
                    if let Err(error) = shared.finish_termination(&process) {
                        shared.fail(error);
                    }
                    continue;
                }
            };
            if current.process.generation() != current.generation {
                continue;
            }
            match report.stop {
                ExecutionStop::Safepoint => {}
                ExecutionStop::SupervisorCall { source, immediate } => {
                    if immediate == 7 && current.thread.loader_return == Some(source.pc) {
                        let result_code = current.execution.thread.read_x(
                            nixe_cpu::state::a64::A64Register::General(
                                nixe_cpu::state::a64::A64GeneralRegister::new(0).unwrap(),
                            ),
                        );
                        current.process.finish_process(
                            nixe_scheduler::ProcessLifecycle::Exited,
                            Some(crate::ProcessExit {
                                cause: crate::ProcessExitCause::LoaderReturned,
                                exit_code: result_code,
                                source: Some(source),
                                thread_id: current.thread.thread_id(),
                                context: Some(Box::new(
                                    current.execution.thread.register_context(),
                                )),
                                frames: Box::new([]),
                            }),
                        );
                        continue;
                    }
                    let context_thread = ExceptionThreadContext::new(
                        current.thread.id(),
                        id,
                        &current.thread,
                        &mut current.execution.thread,
                        current.thread.tls_base,
                    );
                    let mut context = ExceptionDispatchContext::new(
                        source,
                        &current.process,
                        context_thread,
                        queue,
                        current.generation,
                    );
                    let request = crate::ExceptionDispatchRequest::new(
                        source,
                        nixe_cpu::exception::ExceptionKind::SupervisorCall,
                        Some(u64::from(immediate)),
                    );
                    let outcome = dispatcher
                        .dispatch(&mut context, request)
                        .map_fault(|fault| Box::new(fault) as DispatchFault);
                    if let Err(error) = finish_call(
                        shared,
                        native,
                        resident,
                        source,
                        outcome,
                        signal.sequence() != observed,
                    ) {
                        shared.fail(error);
                        break;
                    }
                }
                _ => {
                    let current = resident.take().unwrap();
                    let process = current.process.clone();
                    shared.fail(RuntimeError::Execution {
                        process: current.lease.process,
                        thread: current.lease.thread,
                        diagnostic: report.to_string().into(),
                    });
                    process.finish_process(nixe_scheduler::ProcessLifecycle::Faulted, None);
                    release(shared, native, current, Completion::Ready);
                    if let Err(error) = shared.finish_termination(&process) {
                        shared.fail(error);
                    }
                }
            }
        }
        self.clean_shutdown = shared.stopped.load(Ordering::Acquire);
    }
}
fn finish_call(
    shared: &Shared,
    native: &mut BTreeMap<ProcessId, (Arc<RunnableProcess>, Option<CpuThread>)>,
    resident: &mut Option<Resident>,
    source: nixe_cpu::location::LocationDescriptor,
    outcome: crate::ExceptionDispatchOutcome<DispatchFault>,
    interrupted: bool,
) -> Result<(), RuntimeError> {
    let current = resident
        .as_mut()
        .expect("kernel call owns the selected context");
    if let crate::ExceptionDispatchOutcome::Reject { diagnostic, level } = &outcome {
        shared.record_rejection(
            &current.process,
            current.thread.id(),
            source,
            current.generation,
            *level,
            diagnostic.to_string(),
        );
    }
    let result = current.process.apply_supervisor_call_outcome(
        &current.thread,
        source,
        &mut current.execution.thread,
        outcome,
    );
    match result {
        ExceptionHandlingResult::Resumed | ExceptionHandlingResult::Rejected(_) => {
            if interrupted {
                shared.scheduler.resume_kernel(current.lease)?;
            }
        }
        ExceptionHandlingResult::Suspended => {
            if current.thread.pending_wait().is_none() {
                shared.fail(RuntimeError::Execution {
                    process: current.lease.process,
                    thread: current.lease.thread,
                    diagnostic: "blocked call has no owned continuation".into(),
                });
                current
                    .process
                    .finish_process(nixe_scheduler::ProcessLifecycle::Faulted, None);
            }
            let current = resident.take().unwrap();
            release(shared, native, current, Completion::Waiting);
        }
        result => {
            let current = resident.take().unwrap();
            let process = current.process.clone();
            let thread = current.thread.clone();
            let failed = if let ExceptionHandlingResult::Fault(fault) = result {
                shared.fail(RuntimeError::Execution {
                    process: current.lease.process,
                    thread: current.lease.thread,
                    diagnostic: format!("Horizon SVC dispatch failed: {fault}; source=[{source}]")
                        .into(),
                });
                true
            } else {
                false
            };
            release(
                shared,
                native,
                current,
                if failed {
                    Completion::Faulted
                } else {
                    Completion::Exited
                },
            );
            shared.finalize_thread_waits(&process, thread.id());
            thread.signal();
            if let Err(error) = shared.finish_termination(&process) {
                shared.fail(error);
            }
        }
    }
    Ok(())
}

impl Drop for PhysicalCore {
    fn drop(&mut self) {
        self.signal.bind_native(None);
        if !self.clean_shutdown {
            self.shared.fail_core();
        }
        if let Some(current) = self.resident.take() {
            let process = current.process.clone();
            let thread = current.thread.clone();
            release(
                &self.shared,
                &mut self.native,
                current,
                if self.clean_shutdown {
                    Completion::Ready
                } else {
                    Completion::Faulted
                },
            );
            if !self.clean_shutdown {
                process.terminate_thread(thread.id(), 0, crate::ExceptionTerminationScope::Process);
                self.shared.finalize_thread_waits(&process, thread.id());
                thread.signal();
            }
            if let Err(error) = self.shared.finish_termination(&process) {
                self.shared.fail(error);
            }
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
        if !self.clean_shutdown {
            self.queue.adopt(self.shared.events.clone());
            self.shared
                .abandoned
                .lock()
                .unwrap()
                .push(self.queue.clone());
        }
        self.shared.status_changed();
    }
}

fn release(
    shared: &Shared,
    native: &mut BTreeMap<ProcessId, (Arc<RunnableProcess>, Option<CpuThread>)>,
    mut current: Resident,
    completion: Completion,
) {
    if current.execution.thread.is_available() {
        current.execution.thread.leave_core();
        current.thread.restore_state(current.execution.thread);
    }
    native
        .get_mut(&current.lease.process)
        .expect("retained native process")
        .1 = Some(current.cpu);
    if let Err(error) = shared.scheduler.complete(current.lease, completion) {
        shared.fail(error.into());
    }
    if current.process.lifecycle() != nixe_scheduler::ProcessLifecycle::Running {
        shared.status_changed();
    }
}

pub(super) fn drain(shared: &Shared, queue: &Arc<KernelQueue>) {
    // Resolve each accepted notification even when another caller has failed.
    // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/k_thread_queue.cpp#L21-L35
    for _ in 0..queue.begin_batch() {
        let operation = queue.pop().expect("one physical-core consumer");
        let thread = operation.caller().clone();
        if !thread
            .pending_wait()
            .is_some_and(|pending| Arc::ptr_eq(&pending, &operation))
        {
            continue;
        }
        let process = thread.owner().expect("accepted call owns its process");
        let completed = match operation.clone().poll() {
            Poll::Pending => continue,
            Poll::Ready(completed) => completed,
        };
        let completed = match completed {
            Ok(completed) => completed,
            Err(fault) => {
                thread.clear_wait(&operation);
                process.finish_process(nixe_scheduler::ProcessLifecycle::Faulted, None);
                if let Err(error) = shared.finish_termination(process) {
                    shared.fail(error);
                }
                shared.fail(RuntimeError::Execution {
                    process: ProcessId::new(process.process_id()),
                    thread: thread.id(),
                    diagnostic: format!("{fault}; source=[{}]", operation.source()).into(),
                });
                continue;
            }
        };
        if process.lifecycle() != nixe_scheduler::ProcessLifecycle::Running
            || matches!(completed, crate::WaitCompletion::Cancelled)
        {
            thread.clear_wait(&operation);
            shared.status_changed();
            continue;
        }
        let Some(token) = operation.token() else {
            thread.clear_wait(&operation);
            continue;
        };
        if !shared.scheduler.thread(thread.id()).is_some_and(|view| {
            view.active_wait == Some(token)
                && view.lifecycle == nixe_scheduler::ThreadLifecycle::Waiting
        }) {
            thread.clear_wait(&operation);
            continue;
        }
        if let crate::WaitCompletion::Return { x0, x1 } = completed {
            use nixe_cpu::state::a64::{A64GeneralRegister, A64Register};
            {
                let mut state = thread.registers();
                state.write_x(
                    A64Register::General(A64GeneralRegister::new(0).unwrap()),
                    x0,
                );
                if let Some(value) = x1 {
                    state.write_x(
                        A64Register::General(A64GeneralRegister::new(1).unwrap()),
                        value,
                    );
                }
            }
            thread.clear_wait(&operation);
        }
        // Scalar replies are already installed. Complex continuations retain
        // their canonical request and execute only after the scheduler's claim.
        // https://github.com/eden-emulator/mirror/blob/master/src/core/hle/kernel/physical_core.cpp
        if let Err(error) = shared.scheduler.finish_wait(token) {
            if !matches!(error, nixe_scheduler::SchedulerError::StaleWake(_)) {
                shared.fail(error.into());
            }
            thread.clear_wait(&operation);
            shared.status_changed();
        }
    }
}
