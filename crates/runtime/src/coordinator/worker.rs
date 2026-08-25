use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, TrySendError, sync_channel};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use nixe_cpu_engine::{EngineDomainId, EngineExecutor};
use nixe_scheduler::{Lease, ProcessId, VirtualCpuId};

use crate::process::execution::{ExecutorTeardownState, VcpuExecutionState};
use crate::{ExecutionReport, ProcessExecutionError};

const WORKER_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(5);

pub(super) struct WorkerRequest {
    pub(super) lease: Lease,
    pub(super) executor: WorkerExecutorKey,
    pub(super) fallback: Option<WorkerExecutorKey>,
    pub(super) execution: VcpuExecutionState,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) struct WorkerExecutorKey {
    pub(super) process: ProcessId,
    pub(super) domain: EngineDomainId,
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
    EnginePanicked,
    Lost(VirtualCpuId),
    StaleResult {
        expected: Lease,
        received: Lease,
    },
    Stopped,
    ExecutorUnavailable {
        process: ProcessId,
        vcpu: VirtualCpuId,
    },
    ExecutorTeardownFailed {
        process: ProcessId,
        vcpu: VirtualCpuId,
        fault: Box<nixe_cpu_engine::EngineFault>,
    },
    TeardownTimedOut(VirtualCpuId),
}

enum WorkerCommand {
    Install {
        key: WorkerExecutorKey,
        executor: Box<dyn EngineExecutor>,
        reply: SyncSender<Result<(), Box<dyn EngineExecutor>>>,
    },
    RetireProcess {
        process: ProcessId,
        preparation: ExecutorTeardownState,
        reply: SyncSender<Result<usize, nixe_cpu_engine::EngineFault>>,
    },
    ClearLocalExclusive {
        key: WorkerExecutorKey,
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

    pub(super) fn install_executor(
        &self,
        vcpu: VirtualCpuId,
        key: WorkerExecutorKey,
        executor: Box<dyn EngineExecutor>,
    ) -> Result<(), WorkerFailure> {
        let worker = self.workers.get(&vcpu).ok_or(WorkerFailure::Lost(vcpu))?;
        let (reply, result) = sync_channel(1);
        worker
            .commands
            .send(WorkerCommand::Install {
                key,
                executor,
                reply,
            })
            .map_err(|_| WorkerFailure::Lost(vcpu))?;
        match result.recv().map_err(|_| WorkerFailure::Lost(vcpu))? {
            Ok(()) => Ok(()),
            Err(_) => Err(WorkerFailure::ExecutorUnavailable {
                process: key.process,
                vcpu,
            }),
        }
    }

    pub(super) fn retire_process(
        &self,
        vcpu: VirtualCpuId,
        process: ProcessId,
        preparation: ExecutorTeardownState,
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
            .map_err(|fault| WorkerFailure::ExecutorTeardownFailed {
                process,
                vcpu,
                fault: Box::new(fault),
            })
    }

    pub(super) fn clear_local_exclusive(
        &self,
        vcpu: VirtualCpuId,
        key: WorkerExecutorKey,
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
            .ok_or(WorkerFailure::ExecutorUnavailable {
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
    let mut executors = BTreeMap::new();
    while let Ok(command) = commands.recv() {
        let mut request = match command {
            WorkerCommand::Install {
                key,
                executor,
                reply,
            } => {
                let result = match executors.entry(key) {
                    std::collections::btree_map::Entry::Vacant(entry) => {
                        entry.insert(executor);
                        Ok(())
                    }
                    std::collections::btree_map::Entry::Occupied(_) => Err(executor),
                };
                let _ = reply.send(result);
                continue;
            }
            WorkerCommand::RetireProcess {
                process,
                preparation,
                reply,
            } => {
                let keys: Vec<_> = executors
                    .keys()
                    .filter(|key| key.process == process)
                    .copied()
                    .collect();
                let prepared = keys.iter().try_for_each(|key| {
                    preparation.prepare(
                        executors
                            .get_mut(key)
                            .expect("a collected executor key remains installed")
                            .as_mut(),
                    )
                });
                if prepared.is_ok() {
                    for key in &keys {
                        executors.remove(key);
                    }
                }
                let _ = reply.send(prepared.map(|()| keys.len()));
                continue;
            }
            WorkerCommand::ClearLocalExclusive { key, reply } => {
                let found = if let Some(executor) = executors.get_mut(&key) {
                    executor.clear_local_exclusive_reservation();
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
            let fallback_boundary = crate::process::execution::current_location(
                request.execution.cpu,
                &request.execution.thread,
            );
            let executor = executors.get_mut(&request.executor).ok_or(
                ProcessExecutionError::ExecutorUnavailable {
                    engine: request.executor.domain,
                },
            )?;
            let primary_engine = executor.descriptor().id;
            let report = request.execution.run(executor.as_mut())?;
            let crate::ExecutionStop::InterpretOne { source } = report.stop else {
                return Ok(report);
            };
            if source != fallback_boundary || report.instructions_executed != 0 {
                return Err(ProcessExecutionError::InvalidFallbackBoundary {
                    engine: primary_engine,
                    expected: fallback_boundary,
                    requested: source,
                });
            }
            let Some(fallback_key) = request.fallback else {
                return Err(ProcessExecutionError::FallbackUnavailable {
                    engine: primary_engine,
                });
            };
            let fallback = executors.get_mut(&fallback_key).ok_or(
                ProcessExecutionError::FallbackUnavailable {
                    engine: primary_engine,
                },
            )?;
            let fallback_report = request.execution.run_with_budget(fallback.as_mut(), 1)?;
            if matches!(
                fallback_report.stop,
                crate::ExecutionStop::InterpretOne { .. }
            ) || fallback_report.instructions_executed != 1
            {
                return Err(ProcessExecutionError::FallbackUnavailable {
                    engine: fallback.descriptor().id,
                });
            }
            Ok(ExecutionReport {
                instructions_executed: report
                    .instructions_executed
                    .saturating_add(fallback_report.instructions_executed),
                ..fallback_report
            })
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
        Err(_) => Err(WorkerRunFailure::Worker(WorkerFailure::EnginePanicked)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nixe_cpu::location::ExecutionState;
    use nixe_cpu::memory::ExecutionMemory;
    use nixe_cpu::profile::ProcessCpuContext;
    use nixe_cpu::state::ThreadCpuState;
    use nixe_cpu_engine::{
        EngineCapabilities, EngineDescriptor, EngineExecutor, EngineExecutorId, EngineId,
        EngineKind, RunRequest,
    };
    use nixe_memory::AddressSpaceId;
    use nixe_scheduler::{GuestThreadId, LeaseGeneration, ProcessId};

    struct PanickingExecutor;

    struct BlockingExecutor {
        entered: SyncSender<()>,
        release: Receiver<()>,
    }

    impl EngineExecutor for PanickingExecutor {
        fn descriptor(&self) -> EngineDescriptor {
            EngineDescriptor {
                id: EngineId::new(404),
                name: "panicking-test-engine".into(),
                kind: EngineKind::Test,
                capabilities: EngineCapabilities::default(),
            }
        }

        fn executor_id(&self) -> EngineExecutorId {
            EngineExecutorId::new(1)
        }

        fn run_slice(
            &mut self,
            _request: RunRequest<'_>,
        ) -> Result<ExecutionReport, nixe_cpu_engine::EngineFault> {
            panic!("injected engine panic")
        }

        fn clear_local_exclusive_reservation(&mut self) {}
    }

    impl EngineExecutor for BlockingExecutor {
        fn descriptor(&self) -> EngineDescriptor {
            EngineDescriptor {
                id: EngineId::new(405),
                name: "blocking-test-engine".into(),
                kind: EngineKind::Test,
                capabilities: EngineCapabilities::default(),
            }
        }

        fn executor_id(&self) -> EngineExecutorId {
            EngineExecutorId::new(1)
        }

        fn run_slice(
            &mut self,
            request: RunRequest<'_>,
        ) -> Result<ExecutionReport, nixe_cpu_engine::EngineFault> {
            self.entered.send(()).unwrap();
            self.release.recv().unwrap();
            Ok(ExecutionReport {
                instructions_executed: 0,
                stop: nixe_cpu_engine::EngineExit::Safepoint,
                context: request.state.register_context(),
            })
        }

        fn clear_local_exclusive_reservation(&mut self) {}
    }

    fn request(vcpu: VirtualCpuId) -> WorkerRequest {
        let config = crate::ProcessBuildConfig::default();
        let cpu = ProcessCpuContext::new(config.cpu_profile, AddressSpaceId::new(1));
        let thread = ThreadCpuState::new(cpu.thread_configuration(ExecutionState::A64).unwrap());
        WorkerRequest {
            lease: Lease {
                process: ProcessId::new(1),
                thread: GuestThreadId::new(1),
                vcpu,
                generation: LeaseGeneration::new(1),
            },
            executor: WorkerExecutorKey {
                process: ProcessId::new(1),
                domain: nixe_cpu_engine::EngineDomainId::new(1),
            },
            fallback: None,
            execution: VcpuExecutionState {
                thread,
                cpu,
                memory: Arc::new(ExecutionMemory::new()),
                virtual_clock: crate::VirtualClock::default(),
                architectural_timer_frequency: 19_200_000,
                address_space_end: nixe_memory::GuestVirtualAddress::new(1_u64 << 39),
                instruction_budget: 1,
                loader_return: None,
                events: nixe_cpu_engine::VcpuEventState::default(),
            },
        }
    }

    #[test]
    fn workers_park_contain_panics_and_shutdown_idempotently() {
        let vcpu = VirtualCpuId::new(2);
        let mut pool = VcpuWorkerPool::start([vcpu], true).unwrap();
        pool.install_executor(
            vcpu,
            WorkerExecutorKey {
                process: ProcessId::new(1),
                domain: nixe_cpu_engine::EngineDomainId::new(1),
            },
            Box::new(PanickingExecutor),
        )
        .unwrap();
        assert!(pool.dispatch(request(vcpu)).is_ok());
        let result = pool.receive(vcpu).unwrap();
        assert!(matches!(
            result.outcome,
            Err(WorkerRunFailure::Worker(WorkerFailure::EnginePanicked))
        ));
        pool.shutdown().unwrap();
        pool.shutdown().unwrap();
        assert!(matches!(pool.receive(vcpu), Err(WorkerFailure::Lost(id)) if id == vcpu));
    }

    #[test]
    fn worker_shutdown_reports_a_bound_instead_of_waiting_for_a_stuck_engine() {
        let vcpu = VirtualCpuId::new(2);
        let mut pool = VcpuWorkerPool::start([vcpu], true).unwrap();
        let (entered_tx, entered_rx) = sync_channel(1);
        let (release_tx, release_rx) = sync_channel(1);
        pool.install_executor(
            vcpu,
            WorkerExecutorKey {
                process: ProcessId::new(1),
                domain: nixe_cpu_engine::EngineDomainId::new(1),
            },
            Box::new(BlockingExecutor {
                entered: entered_tx,
                release: release_rx,
            }),
        )
        .unwrap();
        assert!(pool.dispatch(request(vcpu)).is_ok());
        entered_rx.recv().unwrap();
        assert!(matches!(
            pool.shutdown_with_timeout(Duration::from_millis(10)),
            Err(WorkerFailure::TeardownTimedOut(id)) if id == vcpu
        ));
        release_tx.send(()).unwrap();
        let _ = pool.receive(vcpu).unwrap();
        pool.shutdown().unwrap();
    }
}
