use std::collections::{BTreeMap, VecDeque};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, TrySendError, sync_channel};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use nixe_cpu::execution::CpuProcessId;
use nixe_scheduler::{Lease, ProcessId, SchedulerError, SchedulerState, VirtualCpuId};
use std::sync::Arc;

use crate::process::execution::{CpuThread, VcpuExecutionState};
use crate::{ExecutionReport, ProcessExecutionError};

const WORKER_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(5);

pub(super) struct WorkerRequest {
    pub(super) lease: Lease,
    pub(super) cpu_thread: WorkerCpuThreadKey,
    pub(super) execution: VcpuExecutionState,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) struct WorkerCpuThreadKey {
    pub(super) process: ProcessId,
    pub(super) cpu_process: CpuProcessId,
}

pub(crate) struct WorkerResult {
    pub(super) lease: Lease,
    pub(super) execution: VcpuExecutionState,
    pub(super) outcome: Result<ExecutionReport, WorkerRunFailure>,
    pub(super) interrupt: Result<bool, InterruptFailure>,
}

pub(super) enum InterruptFailure {
    Stopped,
    Memory(Arc<nixe_memory::MemoryError>),
    Scheduler(SchedulerError),
}
impl From<nixe_cpu_jit::process::MemoryAccessFailure> for InterruptFailure {
    fn from(error: nixe_cpu_jit::process::MemoryAccessFailure) -> Self {
        match error {
            nixe_cpu_jit::process::MemoryAccessFailure::Stopped => Self::Stopped,
            nixe_cpu_jit::process::MemoryAccessFailure::Memory(error) => Self::Memory(error),
        }
    }
}

pub(super) enum WorkerRunFailure {
    Execution(ProcessExecutionError),
    Worker(WorkerFailure),
}

pub(super) struct WorkerDispatchFailure {
    pub(super) failure: WorkerFailure,
    pub(super) request: WorkerRequest,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WorkerFailure {
    BackendPanicked,
    Lost(VirtualCpuId),
    StaleResult {
        expected: Lease,
        received: Lease,
    },
    Stopped,
    CpuThreadUnavailable {
        process: ProcessId,
        vcpu: VirtualCpuId,
    },
    TeardownTimedOut(VirtualCpuId),
    NativeWorkerTeardownFailed {
        vcpu: VirtualCpuId,
        message: Box<str>,
    },
}

enum WorkerCommand {
    Install {
        key: WorkerCpuThreadKey,
        thread: CpuThread,
        reply: SyncSender<Result<(), CpuThread>>,
    },
    RetireProcess {
        process: ProcessId,
        reply: SyncSender<usize>,
    },
    Run(WorkerRequest),
    Interrupt {
        request: WorkerRequest,
        report: Box<ExecutionReport>,
    },
    Shutdown,
}

struct WorkerHandle {
    commands: SyncSender<WorkerCommand>,
    thread: Option<JoinHandle<Result<(), nixe_cpu_direct_memory::FaultRuntimeError>>>,
    shutdown_sent: bool,
}

pub(super) struct VcpuWorkerPool {
    workers: BTreeMap<VirtualCpuId, WorkerHandle>,
    stop_requested: bool,
    pub(super) pending_results: VecDeque<WorkerResult>,
}

impl VcpuWorkerPool {
    pub(super) fn start(
        vcpus: impl IntoIterator<Item = VirtualCpuId>,
        completion: SyncSender<crate::external_event::CoordinatorEvent>,
        scheduler: Arc<SchedulerState>,
    ) -> Result<Self, std::io::Error> {
        let mut workers: BTreeMap<VirtualCpuId, WorkerHandle> = BTreeMap::new();
        for vcpu in vcpus {
            let (commands, receiver) = sync_channel(1);
            let completion = completion.clone();
            let scheduler = scheduler.clone();
            let thread = match std::thread::Builder::new()
                .name(format!("nixe-vcpu-{}", vcpu.get()))
                .spawn(move || worker_main(receiver, completion, scheduler))
            {
                Ok(thread) => thread,
                Err(error) => {
                    for worker in workers.values_mut() {
                        let _ = worker.commands.send(WorkerCommand::Shutdown);
                        if let Some(thread) = worker.thread.take() {
                            let _ = thread.join();
                        }
                    }
                    return Err(error);
                }
            };
            workers.insert(
                vcpu,
                WorkerHandle {
                    commands,
                    thread: Some(thread),
                    shutdown_sent: false,
                },
            );
        }
        Ok(Self {
            workers,
            stop_requested: false,
            pending_results: VecDeque::new(),
        })
    }

    pub(super) fn dispatch(
        &self,
        request: WorkerRequest,
        report: Option<ExecutionReport>,
    ) -> Result<(), Box<WorkerDispatchFailure>> {
        if self.stop_requested {
            return Err(Box::new(WorkerDispatchFailure {
                failure: WorkerFailure::Stopped,
                request,
            }));
        }
        let vcpu = request.lease.vcpu;
        let Some(worker) = self.workers.get(&vcpu) else {
            return Err(Box::new(WorkerDispatchFailure {
                failure: WorkerFailure::Lost(vcpu),
                request,
            }));
        };
        nixe_memory::trace::detail_event(
            "cpu_dispatch",
            request.lease.generation.get(),
            request.lease.thread.get(),
            u64::from(vcpu.get()),
        );
        let command = match report {
            Some(report) => WorkerCommand::Interrupt {
                request,
                report: Box::new(report),
            },
            None => WorkerCommand::Run(request),
        };
        worker.commands.send(command).map_err(|error| {
            let request = match error.0 {
                WorkerCommand::Run(request) | WorkerCommand::Interrupt { request, .. } => request,
                _ => unreachable!("dispatch only sends execution commands"),
            };
            Box::new(WorkerDispatchFailure {
                failure: WorkerFailure::Lost(vcpu),
                request,
            })
        })
    }

    pub(super) fn install_cpu_thread(
        &self,
        vcpu: VirtualCpuId,
        key: WorkerCpuThreadKey,
        thread: CpuThread,
    ) -> Result<(), WorkerFailure> {
        let worker = self.workers.get(&vcpu).ok_or(WorkerFailure::Lost(vcpu))?;
        let (reply, result) = sync_channel(1);
        worker
            .commands
            .send(WorkerCommand::Install { key, thread, reply })
            .map_err(|_| WorkerFailure::Lost(vcpu))?;
        match result.recv().map_err(|_| WorkerFailure::Lost(vcpu))? {
            Ok(()) => Ok(()),
            Err(_) => Err(WorkerFailure::CpuThreadUnavailable {
                process: key.process,
                vcpu,
            }),
        }
    }

    pub(super) fn retire_process(
        &self,
        vcpu: VirtualCpuId,
        process: ProcessId,
    ) -> Result<usize, WorkerFailure> {
        let worker = self.workers.get(&vcpu).ok_or(WorkerFailure::Lost(vcpu))?;
        let (reply, result) = sync_channel(1);
        let deadline = Instant::now() + WORKER_SHUTDOWN_TIMEOUT;
        let mut command = WorkerCommand::RetireProcess { process, reply };
        loop {
            match worker.commands.try_send(command) {
                Ok(()) => break,
                Err(TrySendError::Disconnected(_)) => {
                    return Err(WorkerFailure::Lost(vcpu));
                }
                Err(TrySendError::Full(returned)) if Instant::now() < deadline => {
                    command = returned;
                    std::thread::yield_now();
                }
                Err(TrySendError::Full(_)) => {
                    return Err(WorkerFailure::TeardownTimedOut(vcpu));
                }
            }
        }
        result
            .recv_timeout(WORKER_SHUTDOWN_TIMEOUT)
            .map_err(|error| match error {
                RecvTimeoutError::Timeout => WorkerFailure::TeardownTimedOut(vcpu),
                RecvTimeoutError::Disconnected => WorkerFailure::Lost(vcpu),
            })
    }

    pub(super) fn shutdown(&mut self) -> Result<(), WorkerFailure> {
        self.shutdown_with_timeout(WORKER_SHUTDOWN_TIMEOUT)
    }

    fn shutdown_with_timeout(&mut self, timeout: Duration) -> Result<(), WorkerFailure> {
        if self.workers.values().all(|worker| worker.thread.is_none()) {
            return Ok(());
        }
        self.stop_requested = true;
        let deadline = Instant::now() + timeout;
        for (vcpu, worker) in &mut self.workers {
            while !worker.shutdown_sent {
                match worker.commands.try_send(WorkerCommand::Shutdown) {
                    Ok(()) | Err(TrySendError::Disconnected(_)) => {
                        worker.shutdown_sent = true;
                    }
                    Err(TrySendError::Full(_)) if Instant::now() < deadline => {
                        std::thread::yield_now();
                    }
                    Err(TrySendError::Full(_)) => {
                        return Err(WorkerFailure::TeardownTimedOut(*vcpu));
                    }
                }
            }
        }
        let mut failure = None;
        for (vcpu, worker) in &mut self.workers {
            while worker
                .thread
                .as_ref()
                .is_some_and(|thread| !thread.is_finished())
                && Instant::now() < deadline
            {
                std::thread::park_timeout(Duration::from_millis(1));
            }
            match worker.thread.take() {
                Some(thread) if thread.is_finished() => match thread.join() {
                    Ok(Ok(())) => {}
                    Ok(Err(error)) => {
                        failure.get_or_insert(WorkerFailure::NativeWorkerTeardownFailed {
                            vcpu: *vcpu,
                            message: error.to_string().into_boxed_str(),
                        });
                    }
                    Err(_) => {
                        failure.get_or_insert(WorkerFailure::Lost(*vcpu));
                    }
                },
                Some(thread) => {
                    worker.thread = Some(thread);
                    failure.get_or_insert(WorkerFailure::TeardownTimedOut(*vcpu));
                }
                None => {}
            }
        }
        failure.map_or(Ok(()), Err)
    }
}

impl Drop for VcpuWorkerPool {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

fn worker_main(
    commands: Receiver<WorkerCommand>,
    results: SyncSender<crate::external_event::CoordinatorEvent>,
    scheduler: Arc<SchedulerState>,
) -> Result<(), nixe_cpu_direct_memory::FaultRuntimeError> {
    let mut native_worker = nixe_cpu_direct_memory::NativeWorker::default();
    let mut cpu_threads = BTreeMap::new();
    while let Ok(command) = commands.recv() {
        let (mut request, report) = match command {
            WorkerCommand::Install { key, thread, reply } => {
                let result = match cpu_threads.entry(key) {
                    std::collections::btree_map::Entry::Vacant(entry) => {
                        entry.insert(thread);
                        Ok(())
                    }
                    std::collections::btree_map::Entry::Occupied(_) => Err(thread),
                };
                let _ = reply.send(result);
                continue;
            }
            WorkerCommand::RetireProcess { process, reply } => {
                let keys: Vec<_> = cpu_threads
                    .keys()
                    .filter(|key| key.process == process)
                    .copied()
                    .collect();
                for key in &keys {
                    cpu_threads.remove(key);
                }
                let _ = reply.send(keys.len());
                continue;
            }
            WorkerCommand::Run(request) => (request, None),
            WorkerCommand::Interrupt { request, report } => (request, Some(*report)),
            WorkerCommand::Shutdown => break,
        };
        let run_trace = nixe_memory::trace::span(
            "cpu_run",
            request.lease.generation.get(),
            request.lease.thread.get(),
            u64::from(request.lease.vcpu.get()),
        );
        let kernel_interrupt = report.is_some();
        let mut interrupt = Ok(false);
        let run = || {
            let thread = cpu_threads.get_mut(&request.cpu_thread).ok_or(
                ProcessExecutionError::BackendUnavailable {
                    process: request.cpu_thread.cpu_process,
                },
            )?;
            let report = match report {
                Some(report) => report,
                None => request.execution.run(&mut native_worker, thread)?,
            };
            if kernel_interrupt
                || matches!(
                    report.stop,
                    crate::ExecutionStop::Safepoint | crate::ExecutionStop::PendingEvent { .. }
                )
            {
                interrupt = handle_interrupt(&scheduler, &request, thread).map(|()| true);
            }
            Ok(report)
        };
        let outcome = catch_worker_panic(run);
        drop(run_trace);
        nixe_memory::trace::detail_event(
            "cpu_result",
            request.lease.generation.get(),
            request.lease.thread.get(),
            u64::from(request.lease.vcpu.get()),
        );
        if results
            .send(crate::external_event::CoordinatorEvent::Completed(
                WorkerResult {
                    lease: request.lease,
                    execution: request.execution,
                    outcome,
                    interrupt,
                },
            ))
            .is_err()
        {
            break;
        }
    }
    drop(cpu_threads);
    native_worker.finish()
}

// The interrupted core keeps its native context/lease until TLS publication.
// Device waits share the JIT process's existing stop/cancellation mechanism.
// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/k_interrupt_manager.cpp
fn handle_interrupt(
    scheduler: &SchedulerState,
    request: &WorkerRequest,
    cpu: &CpuThread,
) -> Result<(), InterruptFailure> {
    let tls = request.execution.tls_base;
    if cpu
        .read_memory(tls.checked_add(0x100).expect("bounded TLS"), 2)
        .map_err(InterruptFailure::from)?
        != 0
        && scheduler
            .pinned_thread(request.lease.process, request.lease.vcpu)
            .map_err(InterruptFailure::Scheduler)?
            .is_none()
    {
        scheduler
            .pin_thread(request.lease.vcpu, request.lease.thread)
            .map_err(InterruptFailure::Scheduler)?;
        cpu.write_memory(tls.checked_add(0x102).expect("bounded TLS"), 2, 1)
            .map_err(InterruptFailure::from)?;
    }
    Ok(())
}

fn catch_worker_panic(
    run: impl FnOnce() -> Result<ExecutionReport, ProcessExecutionError>,
) -> Result<ExecutionReport, WorkerRunFailure> {
    match catch_unwind(AssertUnwindSafe(run)) {
        Ok(result) => result.map_err(WorkerRunFailure::Execution),
        Err(_) => Err(WorkerRunFailure::Worker(WorkerFailure::BackendPanicked)),
    }
}
