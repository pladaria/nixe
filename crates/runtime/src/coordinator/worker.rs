use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, TrySendError, sync_channel};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use nixe_cpu::execution::CpuProcessId;
use nixe_scheduler::{Lease, ProcessId, VirtualCpuId};

use crate::process::execution::{CpuThread, CpuThreadTeardownState, VcpuExecutionState};
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

pub(super) struct WorkerResult {
    pub(super) lease: Lease,
    pub(super) execution: VcpuExecutionState,
    pub(super) outcome: Result<ExecutionReport, WorkerRunFailure>,
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
    CpuThreadTeardownFailed {
        process: ProcessId,
        vcpu: VirtualCpuId,
        fault: Box<nixe_cpu::execution::CpuFault>,
    },
    TeardownTimedOut(VirtualCpuId),
}

enum WorkerCommand {
    Install {
        key: WorkerCpuThreadKey,
        thread: CpuThread,
        reply: SyncSender<Result<(), CpuThread>>,
    },
    RetireProcess {
        process: ProcessId,
        preparation: CpuThreadTeardownState,
        reply: SyncSender<Result<usize, nixe_cpu::execution::CpuFault>>,
    },
    ClearLocalExclusive {
        key: WorkerCpuThreadKey,
        reply: SyncSender<bool>,
    },
    Run(WorkerRequest),
    Shutdown,
}

struct WorkerHandle {
    commands: SyncSender<WorkerCommand>,
    results: Receiver<WorkerResult>,
    thread: Option<JoinHandle<()>>,
    shutdown_sent: bool,
}

pub(super) struct VcpuWorkerPool {
    workers: BTreeMap<VirtualCpuId, WorkerHandle>,
    stop_requested: bool,
}

impl VcpuWorkerPool {
    pub(super) fn start(
        vcpus: impl IntoIterator<Item = VirtualCpuId>,
        serialize_execution: bool,
    ) -> Result<Self, std::io::Error> {
        let global_permit = serialize_execution.then(|| Arc::new(Mutex::new(())));
        let mut workers: BTreeMap<VirtualCpuId, WorkerHandle> = BTreeMap::new();
        for vcpu in vcpus {
            let (commands, receiver) = sync_channel(1);
            let (completion, results) = sync_channel(1);
            let permit = global_permit.clone();
            let thread = match std::thread::Builder::new()
                .name(format!("nixe-vcpu-{}", vcpu.get()))
                .spawn(move || worker_main(receiver, completion, permit))
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
                    results,
                    thread: Some(thread),
                    shutdown_sent: false,
                },
            );
        }
        Ok(Self {
            workers,
            stop_requested: false,
        })
    }

    pub(super) fn dispatch(
        &self,
        request: WorkerRequest,
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
        worker
            .commands
            .send(WorkerCommand::Run(request))
            .map_err(|error| {
                let WorkerCommand::Run(request) = error.0 else {
                    unreachable!("dispatch only sends run commands")
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
        preparation: CpuThreadTeardownState,
    ) -> Result<usize, WorkerFailure> {
        let worker = self.workers.get(&vcpu).ok_or(WorkerFailure::Lost(vcpu))?;
        let (reply, result) = sync_channel(1);
        let deadline = Instant::now() + WORKER_SHUTDOWN_TIMEOUT;
        let mut command = WorkerCommand::RetireProcess {
            process,
            preparation,
            reply,
        };
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
            })?
            .map_err(|fault| WorkerFailure::CpuThreadTeardownFailed {
                process,
                vcpu,
                fault: Box::new(fault),
            })
    }

    pub(super) fn clear_local_exclusive(
        &self,
        vcpu: VirtualCpuId,
        key: WorkerCpuThreadKey,
    ) -> Result<(), WorkerFailure> {
        let worker = self.workers.get(&vcpu).ok_or(WorkerFailure::Lost(vcpu))?;
        let (reply, result) = sync_channel(1);
        worker
            .commands
            .send(WorkerCommand::ClearLocalExclusive { key, reply })
            .map_err(|_| WorkerFailure::Lost(vcpu))?;
        result
            .recv()
            .map_err(|_| WorkerFailure::Lost(vcpu))?
            .then_some(())
            .ok_or(WorkerFailure::CpuThreadUnavailable {
                process: key.process,
                vcpu,
            })
    }

    pub(super) fn receive(&self, vcpu: VirtualCpuId) -> Result<WorkerResult, WorkerFailure> {
        self.workers
            .get(&vcpu)
            .ok_or(WorkerFailure::Lost(vcpu))?
            .results
            .recv()
            .map_err(|_| WorkerFailure::Lost(vcpu))
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
                Some(thread) if thread.is_finished() => {
                    if thread.join().is_err() {
                        failure.get_or_insert(WorkerFailure::Lost(*vcpu));
                    }
                }
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
    results: SyncSender<WorkerResult>,
    global_permit: Option<Arc<Mutex<()>>>,
) {
    let mut cpu_threads = BTreeMap::new();
    while let Ok(command) = commands.recv() {
        let mut request = match command {
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
            WorkerCommand::RetireProcess {
                process,
                preparation,
                reply,
            } => {
                let keys: Vec<_> = cpu_threads
                    .keys()
                    .filter(|key| key.process == process)
                    .copied()
                    .collect();
                let prepared = keys.iter().try_for_each(|key| {
                    preparation.prepare(
                        cpu_threads
                            .get_mut(key)
                            .expect("a collected CPU thread key remains installed"),
                    )
                });
                if prepared.is_ok() {
                    for key in &keys {
                        cpu_threads.remove(key);
                    }
                }
                let _ = reply.send(prepared.map(|()| keys.len()));
                continue;
            }
            WorkerCommand::ClearLocalExclusive { key, reply } => {
                let found = if let Some(thread) = cpu_threads.get_mut(&key) {
                    thread.clear_local_exclusive_reservation();
                    true
                } else {
                    false
                };
                let _ = reply.send(found);
                continue;
            }
            WorkerCommand::Run(request) => request,
            WorkerCommand::Shutdown => break,
        };
        let run = || {
            let thread = cpu_threads.get_mut(&request.cpu_thread).ok_or(
                ProcessExecutionError::BackendUnavailable {
                    process: request.cpu_thread.cpu_process,
                },
            )?;
            request.execution.run(thread)
        };
        let outcome = if let Some(permit) = &global_permit {
            let _permit = permit
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            catch_worker_panic(run)
        } else {
            catch_worker_panic(run)
        };
        if results
            .send(WorkerResult {
                lease: request.lease,
                execution: request.execution,
                outcome,
            })
            .is_err()
        {
            break;
        }
    }
}

fn catch_worker_panic(
    run: impl FnOnce() -> Result<ExecutionReport, ProcessExecutionError>,
) -> Result<ExecutionReport, WorkerRunFailure> {
    match catch_unwind(AssertUnwindSafe(run)) {
        Ok(result) => result.map_err(WorkerRunFailure::Execution),
        Err(_) => Err(WorkerRunFailure::Worker(WorkerFailure::BackendPanicked)),
    }
}
