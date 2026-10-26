use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, TrySendError, sync_channel};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use nixe_cpu::execution::CpuProcessId;
use nixe_scheduler::{GuestThreadId, Lease, ProcessId, VirtualCpuId};

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

struct WorkerCpuThread {
    cpu: CpuThread,
    guest: Option<GuestThreadId>,
}

impl WorkerCpuThread {
    fn select_guest(&mut self, guest: GuestThreadId) {
        if self.guest != Some(guest) {
            // Exclusive reservations belong to execution on this guest thread,
            // not to the register snapshot or the next occupant of this core.
            self.cpu.clear_local_exclusive_reservation();
            self.guest = Some(guest);
        }
    }
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
    ClearLocalExclusive {
        key: WorkerCpuThreadKey,
        reply: SyncSender<bool>,
    },
    Run(WorkerRequest),
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
    results: Receiver<WorkerResult>,
    pending_results: BTreeMap<VirtualCpuId, WorkerResult>,
}

impl VcpuWorkerPool {
    pub(super) fn start(
        vcpus: impl IntoIterator<Item = VirtualCpuId>,
    ) -> Result<Self, std::io::Error> {
        let vcpus: Vec<_> = vcpus.into_iter().collect();
        let (completion, results) = sync_channel(vcpus.len().max(1));
        let mut workers: BTreeMap<VirtualCpuId, WorkerHandle> = BTreeMap::new();
        for vcpu in vcpus {
            let (commands, receiver) = sync_channel(1);
            let completion = completion.clone();
            let thread = match std::thread::Builder::new()
                .name(format!("nixe-vcpu-{}", vcpu.get()))
                .spawn(move || worker_main(receiver, completion))
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
            results,
            pending_results: BTreeMap::new(),
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
        nixe_memory::trace::detail_event(
            "cpu_dispatch",
            request.lease.generation.get(),
            request.lease.thread.get(),
            u64::from(vcpu.get()),
        );
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

    pub(super) fn receive_for(
        &mut self,
        vcpu: VirtualCpuId,
        timeout: Option<Duration>,
    ) -> Result<Option<WorkerResult>, WorkerFailure> {
        if !self.workers.contains_key(&vcpu) {
            return Err(WorkerFailure::Lost(vcpu));
        }
        if let Some(result) = self.pending_results.remove(&vcpu) {
            return Ok(Some(result));
        }
        let deadline = timeout.map(|timeout| Instant::now() + timeout);
        loop {
            let result = self.receive_result(
                deadline.map(|deadline| deadline.saturating_duration_since(Instant::now())),
            )?;
            let Some(result) = result else {
                return Ok(None);
            };
            if result.lease.vcpu == vcpu {
                return Ok(Some(result));
            }
            self.pending_results.insert(result.lease.vcpu, result);
        }
    }

    pub(super) fn receive_any(
        &mut self,
        timeout: Option<Duration>,
    ) -> Result<Option<WorkerResult>, WorkerFailure> {
        if let Some((_, result)) = self.pending_results.pop_first() {
            return Ok(Some(result));
        }
        self.receive_result(timeout)
    }

    fn receive_result(
        &self,
        timeout: Option<Duration>,
    ) -> Result<Option<WorkerResult>, WorkerFailure> {
        match timeout {
            Some(timeout) => match self.results.recv_timeout(timeout) {
                Ok(result) => Ok(Some(result)),
                Err(RecvTimeoutError::Timeout) => Ok(None),
                Err(RecvTimeoutError::Disconnected) => Err(WorkerFailure::Stopped),
            },
            None => self
                .results
                .recv()
                .map(Some)
                .map_err(|_| WorkerFailure::Stopped),
        }
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
    results: SyncSender<WorkerResult>,
) -> Result<(), nixe_cpu_direct_memory::FaultRuntimeError> {
    let mut native_worker = nixe_cpu_direct_memory::NativeWorker::default();
    let mut cpu_threads = BTreeMap::new();
    while let Ok(command) = commands.recv() {
        let mut request = match command {
            WorkerCommand::Install { key, thread, reply } => {
                let result = match cpu_threads.entry(key) {
                    std::collections::btree_map::Entry::Vacant(entry) => {
                        entry.insert(WorkerCpuThread {
                            cpu: thread,
                            guest: None,
                        });
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
                    let mut thread = cpu_threads.remove(key).unwrap();
                    thread.cpu.clear_local_exclusive_reservation();
                }
                let _ = reply.send(keys.len());
                continue;
            }
            WorkerCommand::ClearLocalExclusive { key, reply } => {
                let found = if let Some(thread) = cpu_threads.get_mut(&key) {
                    thread.cpu.clear_local_exclusive_reservation();
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
        let run_trace = nixe_memory::trace::span(
            "cpu_run",
            request.lease.generation.get(),
            request.lease.thread.get(),
            u64::from(request.lease.vcpu.get()),
        );
        let run = || {
            let thread = cpu_threads.get_mut(&request.cpu_thread).ok_or(
                ProcessExecutionError::BackendUnavailable {
                    process: request.cpu_thread.cpu_process,
                },
            )?;
            thread.select_guest(request.lease.thread);
            request.execution.run(&mut native_worker, &mut thread.cpu)
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
    drop(cpu_threads);
    native_worker.finish()
}

fn catch_worker_panic(
    run: impl FnOnce() -> Result<ExecutionReport, ProcessExecutionError>,
) -> Result<ExecutionReport, WorkerRunFailure> {
    match catch_unwind(AssertUnwindSafe(run)) {
        Ok(result) => result.map_err(WorkerRunFailure::Execution),
        Err(_) => Err(WorkerRunFailure::Worker(WorkerFailure::BackendPanicked)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nixe_cpu::{
        execution::{ArchitecturalTimer, TimerSnapshot, VcpuEventState},
        platform::TargetPlatform,
        profile::ProcessCpuContext,
        state::a64::{A64GeneralRegister, A64Register},
    };
    use nixe_cpu_jit::{JitProcess, JitRunRequest, ThreadState};
    use nixe_memory::{
        Access, AddressSpace, AddressWidth, AllocationRange, GuestVirtualAddress, Machine,
        MemoryPermissions, PAGE_SIZE,
    };

    struct FixedTimer;
    impl ArchitecturalTimer for FixedTimer {
        fn snapshot(&self) -> TimerSnapshot {
            TimerSnapshot {
                counter: 0,
                frequency: 19_200_000,
            }
        }
    }
    fn x(index: u8) -> A64Register {
        A64Register::General(A64GeneralRegister::new(index).unwrap())
    }

    #[test]
    fn guest_switch_clears_exclusives_but_same_guest_resume_preserves_them() {
        let machine = Machine::new(2 * PAGE_SIZE).unwrap();
        let account = machine.account(2 * PAGE_SIZE).unwrap();
        let code = [0xc85f7c23_u32, 0x14000001, 0xc8047c23, 0xd4000001];
        let image = account.allocate(PAGE_SIZE).unwrap();
        machine
            .write_resolved(
                &[AllocationRange::new(image.pin(), 0, code.len() as u64 * 4).unwrap()],
                &code
                    .into_iter()
                    .flat_map(u32::to_le_bytes)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let space = AddressSpace::new(machine, AddressWidth::Bits32).unwrap();
        space
            .map(
                GuestVirtualAddress::new(0x1000),
                image,
                0,
                PAGE_SIZE,
                MemoryPermissions::READ_EXECUTE,
            )
            .unwrap();
        space
            .map(
                GuestVirtualAddress::new(0x2000),
                account.allocate(PAGE_SIZE).unwrap(),
                0,
                PAGE_SIZE,
                MemoryPermissions::READ_WRITE,
            )
            .unwrap();
        let process = JitProcess::new(
            ProcessCpuContext::for_platform(TargetPlatform::Switch1, space.identity()),
            space.clone(),
            19_200_000,
        )
        .unwrap();
        let mut core = WorkerCpuThread {
            cpu: Box::new(process.create_thread().unwrap()),
            guest: None,
        };
        let mut worker = nixe_cpu_direct_memory::NativeWorker::default();
        let mut state = ThreadState::default();
        state.write_x(x(1), 0x2000);
        let mut run = |core: &mut WorkerCpuThread, guest, pc, value| {
            core.select_guest(GuestThreadId::new(guest));
            state.set_pc(pc);
            state.write_x(x(3), value);
            let jit = &mut core.cpu;
            let report = jit
                .run_slice(
                    &mut worker,
                    JitRunRequest {
                        state: &mut state,
                        instruction_budget: 2,
                        timer: &FixedTimer,
                        events: &VcpuEventState::default(),
                    },
                )
                .unwrap();
            if pc == 0x1000 {
                assert_eq!(report.stop, nixe_cpu::execution::CpuExit::BudgetExhausted);
            } else {
                assert!(matches!(
                    report.stop,
                    nixe_cpu::execution::CpuExit::SupervisorCall { .. }
                ));
            }
            state.read_x(x(4))
        };
        run(&mut core, 1, 0x1000, 0);
        assert_eq!(run(&mut core, 2, 0x1008, 7), 1);
        run(&mut core, 2, 0x1000, 0);
        assert_eq!(run(&mut core, 2, 0x1008, 7), 0);
        assert!(matches!(
            space
                .read_value(GuestVirtualAddress::new(0x2000), 8)
                .unwrap(),
            Access::Ready(7)
        ));
    }
}
