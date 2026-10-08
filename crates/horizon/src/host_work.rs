//! Bounded host work with retained mappings and replies published before wakeup.
use crate::ipc_wire::IpcWireError;
use nixe_memory::{
    CanonicalBackingRange, CanonicalWriteBatch, GuestVirtualAddress, MemoryPermissions,
};
use nixe_runtime::{
    EventObject, ExceptionProcessContext, ExternalEventSource, ReadableEventObject,
    WritableEventObject,
};
use std::{
    collections::BTreeMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread::{self, JoinHandle},
};

type Key = (u64, u64);
type Job = Box<dyn FnOnce() + Send>;
type Finalize =
    Box<dyn FnOnce(&mut ExceptionProcessContext<'_>) -> Result<(), IpcWireError> + Send>;
struct Completion {
    readable: ReadableEventObject,
    writable: WritableEventObject,
    result: Mutex<Option<Result<Option<Finalize>, IpcWireError>>>,
}
impl std::fmt::Debug for Completion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Completion").finish_non_exhaustive()
    }
}
impl Completion {
    fn new() -> Arc<Self> {
        let (writable, readable) =
            EventObject::create_pair_with_source(ExternalEventSource::Device);
        Arc::new(Self {
            readable,
            writable,
            result: Mutex::new(None),
        })
    }
}
#[derive(Clone, Debug)]
pub(crate) struct PendingHostWork(Arc<Completion>);
impl PendingHostWork {
    pub(crate) fn wake_event(&self) -> ReadableEventObject {
        self.0.readable.clone()
    }
}
impl PartialEq for PendingHostWork {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
impl Eq for PendingHostWork {}
struct Pool {
    sender: Option<mpsc::SyncSender<Job>>,
    workers: Vec<JoinHandle<()>>,
    capacity: Arc<Completion>,
}
impl Pool {
    fn new(name: &str, threads: usize) -> Self {
        let (sender, receiver) = mpsc::sync_channel::<Job>(8);
        let receiver = Arc::new(Mutex::new(receiver));
        let capacity = Completion::new();
        let workers = (0..threads)
            .map(|index| {
                let receiver = receiver.clone();
                let capacity = capacity.clone();
                thread::Builder::new()
                    .name(format!("nixe-{name}-{index}"))
                    .spawn(move || {
                        loop {
                            let job = receiver.lock().unwrap().recv();
                            let Ok(job) = job else { break };
                            capacity.writable.signal();
                            job();
                        }
                    })
                    .expect("failed to start host work owner")
            })
            .collect();
        Self {
            sender: Some(sender),
            workers,
            capacity,
        }
    }
}
impl Drop for Pool {
    fn drop(&mut self) {
        self.sender.take();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}
pub(crate) struct HostWorkSystem {
    pending: Mutex<BTreeMap<Key, Arc<Completion>>>,
    cancelled: Arc<AtomicBool>,
    graphics: Pool,
    graphics_failed: Arc<AtomicBool>,
    graphics_failure: Arc<Mutex<Option<IpcWireError>>>,
    storage: Pool,
}
impl std::fmt::Debug for HostWorkSystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostWorkSystem").finish_non_exhaustive()
    }
}
impl Default for HostWorkSystem {
    fn default() -> Self {
        Self {
            pending: Mutex::new(BTreeMap::new()),
            cancelled: Arc::new(AtomicBool::new(false)),
            graphics: Pool::new("gpu-frontend", 1),
            graphics_failed: Arc::default(),
            graphics_failure: Arc::default(),
            storage: Pool::new("storage", 1),
        }
    }
}
impl Drop for HostWorkSystem {
    fn drop(&mut self) {
        self.cancelled.store(true, Ordering::Release);
    }
}
impl HostWorkSystem {
    pub(crate) fn require_graphics_healthy(&self) -> Result<(), IpcWireError> {
        if self.graphics_failed.load(Ordering::Acquire) {
            return Err(self
                .graphics_failure
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .clone());
        }
        Ok(())
    }

    /// Acceptance returns before interpretation. The nvdrv permit already bounds
    /// active work; mutations sharing frontend state retain the gate until finish.
    /// Publish a fatal execution error before releasing that gate, never a fence.
    pub(crate) fn submit_graphics(
        &self,
        frontend: u64,
        guard: WorkGuard,
        run: impl FnOnce() -> Result<(), IpcWireError> + Send + 'static,
    ) -> Result<(), IpcWireError> {
        self.require_graphics_healthy()?;
        let failed = self.graphics_failed.clone();
        let failure = self.graphics_failure.clone();
        let cancelled = self.cancelled.clone();
        let job: Job = Box::new(move || {
            let _trace = nixe_trace::Span::new("gpu.frontend_job", frontend, 0);
            if !cancelled.load(Ordering::Acquire) && !failed.load(Ordering::Acquire) {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)).unwrap_or(
                    Err(IpcWireError::Internal("graphics frontend owner panicked")),
                );
                if let Err(error) = result {
                    *failure.lock().unwrap() = Some(error);
                    failed.store(true, Ordering::Release);
                    guard.abort();
                    return;
                }
            } else {
                guard.abort();
                return;
            }
            drop(guard);
        });
        self.graphics
            .sender
            .as_ref()
            .unwrap()
            .try_send(job)
            .map_err(|_| IpcWireError::Internal("reserved graphics work could not be accepted"))?;
        nixe_trace::event("gpu.frontend_accepted", frontend, 0);
        Ok(())
    }
    pub(crate) fn poll(
        &self,
        key: Key,
        process: &mut ExceptionProcessContext<'_>,
    ) -> Option<Result<(), IpcWireError>> {
        let mut pending = self.pending.lock().unwrap();
        let completion = pending.get(&key)?;
        let result = completion.result.lock().unwrap().take();
        match result {
            Some(result) => {
                pending.remove(&key);
                drop(pending);
                let result = result.and_then(|finalize| match finalize {
                    Some(finalize) => finalize(process),
                    None => Ok(()),
                });
                nixe_trace::event("ipc.host_complete", key.1, key.0);
                Some(result)
            }
            None => Some(Err(IpcWireError::PendingHostWork(PendingHostWork(
                completion.clone(),
            )))),
        }
    }
    fn submit(
        &self,
        key: Key,
        job: impl FnOnce(&AtomicBool) -> Result<Option<Finalize>, IpcWireError> + Send + 'static,
    ) -> Result<(), IpcWireError> {
        let pool = &self.storage;
        let completion = Completion::new();
        let finished = completion.clone();
        let cancelled = self.cancelled.clone();
        let job: Job = Box::new(move || {
            let _span = nixe_trace::Span::new("ipc.host_job", key.1, key.0);
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                if cancelled.load(Ordering::Acquire) {
                    return Ok(None);
                }
                job(&cancelled)
            }))
            .unwrap_or(Err(IpcWireError::Internal("host work owner panicked")));
            *finished.result.lock().unwrap() = Some(result);
            nixe_trace::event("ipc.host_ready", key.1, key.0);
            finished.writable.signal();
        });
        let mut pending = self.pending.lock().unwrap();
        pool.capacity.readable.clear();
        nixe_trace::event("ipc.host_enqueue", key.1, key.0);
        match pool.sender.as_ref().expect("live pool").try_send(job) {
            Ok(()) => {
                assert!(pending.insert(key, completion.clone()).is_none());
                Err(IpcWireError::PendingHostWork(PendingHostWork(completion)))
            }
            Err(mpsc::TrySendError::Full(_)) => {
                nixe_trace::event("ipc.host_retry", key.1, key.0);
                Err(IpcWireError::PendingHostWork(PendingHostWork(
                    pool.capacity.clone(),
                )))
            }
            Err(mpsc::TrySendError::Disconnected(_)) => {
                nixe_trace::event("ipc.host_failed", key.1, key.0);
                Err(IpcWireError::Internal("host work owner stopped"))
            }
        }
    }
}

pub(crate) struct AsyncReply<'a> {
    pub(crate) system: &'a HostWorkSystem,
    pub(crate) key: Key,
    pub(crate) address: GuestVirtualAddress,
    pub(crate) size: usize,
}
impl AsyncReply<'_> {
    pub(crate) fn submit(
        &self,
        process: &ExceptionProcessContext<'_>,
        job: impl FnOnce(&AtomicBool) -> Result<Vec<u8>, IpcWireError> + Send + 'static,
    ) -> Result<(), IpcWireError> {
        let reply = retain_output(process, self.address.get(), self.size as u64)?;
        self.system.submit(self.key, move |cancelled| {
            let response = job(cancelled)?;
            if !cancelled.load(Ordering::Acquire) {
                write_retained(&reply, 0, &response)?;
            }
            Ok(None)
        })
    }
    /// The worker prepares an object; only the coordinator allocates process handles.
    /// Its ready event schedules finalization, and the reply is committed before
    /// the suspended SVC returns or the guest caller can run.
    pub(crate) fn submit_prepared<T: Send + 'static>(
        &self,
        process: &ExceptionProcessContext<'_>,
        prepare: impl FnOnce() -> Result<T, IpcWireError> + Send + 'static,
        finalize: impl FnOnce(
            &mut ExceptionProcessContext<'_>,
            T,
        ) -> Result<(Vec<u8>, Option<u32>), IpcWireError>
        + Send
        + 'static,
    ) -> Result<(), IpcWireError> {
        let reply = retain_output(process, self.address.get(), self.size as u64)?;
        self.system.submit(self.key, move |_| {
            let value = prepare()?;
            Ok(Some(Box::new(move |process| {
                let (response, handle) = finalize(process, value)?;
                if let Err(error) = write_retained(&reply, 0, &response) {
                    if let Some(handle) = handle {
                        let _ = process.handles_mut().close(handle);
                    }
                    return Err(error);
                }
                Ok(())
            })))
        })
    }
}
pub(crate) fn retain_output(
    process: &ExceptionProcessContext<'_>,
    address: u64,
    size: u64,
) -> Result<CanonicalBackingRange, IpcWireError> {
    process
        .canonical_memory()
        .translate_canonical_range(
            process.cpu().address_space_id(),
            GuestVirtualAddress::new(address),
            size,
            MemoryPermissions::WRITE,
        )
        .map_err(|_| IpcWireError::Malformed("asynchronous output is not writable canonical RAM"))
}
pub(crate) fn write_retained(
    range: &CanonicalBackingRange,
    offset: u64,
    bytes: &[u8],
) -> Result<(), IpcWireError> {
    let mut batch = CanonicalWriteBatch::new();
    batch
        .stage(range, offset, bytes)
        .and_then(|()| batch.commit())
        .map_err(|_| IpcWireError::Internal("asynchronous output visibility failed"))
}

/// Service operations sharing frontend state suspend instead of taking a busy lock.
pub(crate) struct WorkGate {
    busy: AtomicBool,
    completion: Arc<Completion>,
    after: Mutex<Vec<Job>>,
}
impl std::fmt::Debug for WorkGate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkGate").finish_non_exhaustive()
    }
}
impl Default for WorkGate {
    fn default() -> Self {
        Self {
            busy: AtomicBool::new(false),
            completion: Completion::new(),
            after: Mutex::new(Vec::new()),
        }
    }
}
impl WorkGate {
    /// Order a consumer after the currently accepted frontend job without
    /// suspending its caller. The callback only enqueues backend work.
    pub(crate) fn after_current(&self, job: impl FnOnce() + Send + 'static) {
        let mut after = self.after.lock().unwrap();
        if self.busy.load(Ordering::Acquire) {
            after.push(Box::new(job));
        } else {
            job();
        }
    }
    pub(crate) fn wait(&self) -> Option<PendingHostWork> {
        self.busy
            .load(Ordering::Acquire)
            .then(|| PendingHostWork(self.completion.clone()))
    }
    /// Shutdown only: producers have stopped, no frontend/state lock is held.
    pub(crate) fn finish(&self) {
        while self.busy.load(Ordering::Acquire) {
            let _ = self.completion.readable.wait(None);
        }
    }
    pub(crate) fn begin(self: &Arc<Self>) -> WorkGuard {
        let _after = self.after.lock().unwrap();
        self.completion.readable.clear();
        assert!(!self.busy.swap(true, Ordering::AcqRel));
        WorkGuard(self.clone(), true)
    }
}
pub(crate) struct WorkGuard(Arc<WorkGate>, bool);
impl WorkGuard {
    fn abort(mut self) {
        self.1 = false;
    }
}
impl Drop for WorkGuard {
    fn drop(&mut self) {
        let mut after = self.0.after.lock().unwrap();
        for job in after.drain(..) {
            if self.1 {
                job();
            }
        }
        self.0.busy.store(false, Ordering::Release);
        self.0.completion.writable.signal();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    #[test]
    fn consumers_follow_frontend_acceptance_without_blocking_the_caller() {
        let gate = Arc::new(WorkGate::default());
        let guard = gate.begin();
        let order = Arc::new(Mutex::new(Vec::new()));
        for id in 0..3 {
            let order = order.clone();
            gate.after_current(move || order.lock().unwrap().push(id));
        }
        assert!(order.lock().unwrap().is_empty());
        assert!(gate.wait().is_some());
        drop(guard);
        assert_eq!(*order.lock().unwrap(), [0, 1, 2]);
        assert!(gate.wait().is_none());
        let next = order.clone();
        gate.after_current(move || next.lock().unwrap().push(3));
        assert_eq!(*order.lock().unwrap(), [0, 1, 2, 3]);
        drop(gate.begin());
    }
    #[test]
    fn failed_frontend_discards_dependent_consumers() {
        let gate = Arc::new(WorkGate::default());
        let guard = gate.begin();
        gate.after_current(|| panic!("a failed producer must not export its image"));
        guard.abort();
        assert!(gate.wait().is_none());
        let guard = gate.begin();
        let (reply, result) = mpsc::channel();
        gate.after_current(move || reply.send(()).unwrap());
        drop(guard);
        assert_eq!(result.try_recv(), Ok(()));
    }
    #[test]
    fn storage_is_bounded_fifo_and_does_not_block_graphics() {
        let system = HostWorkSystem::default();
        let (entered, started) = mpsc::sync_channel(1);
        let (release, wait) = mpsc::sync_channel(1);
        assert!(matches!(
            system.submit((1, 1), move |_| {
                entered.send(()).unwrap();
                wait.recv().unwrap();
                Ok(None)
            }),
            Err(IpcWireError::PendingHostWork(_))
        ));
        started.recv_timeout(Duration::from_secs(2)).unwrap();
        let order = Arc::new(Mutex::new(Vec::new()));
        let (ordered, done) = mpsc::sync_channel(1);
        for id in 2..10 {
            let order = order.clone();
            let ordered = ordered.clone();
            assert!(matches!(
                system.submit((1, id), move |_| {
                    order.lock().unwrap().push(id);
                    if id == 9 {
                        ordered.send(()).unwrap();
                    }
                    Ok(None)
                }),
                Err(IpcWireError::PendingHostWork(_))
            ));
        }
        assert!(matches!(
            system.submit((1, 10), |_| panic!("a rejected job must never execute")),
            Err(IpcWireError::PendingHostWork(_))
        ));
        assert!(!system.pending.lock().unwrap().contains_key(&(1, 10)));
        let (finished, ready) = mpsc::sync_channel(1);
        system
            .submit_graphics(1, Arc::new(WorkGate::default()).begin(), move || {
                finished.send(()).unwrap();
                Ok(())
            })
            .unwrap();
        ready.recv_timeout(Duration::from_secs(2)).unwrap();
        release.send(()).unwrap();
        done.recv_timeout(Duration::from_secs(2)).unwrap();
        drop(system);
        assert_eq!(*order.lock().unwrap(), (2..10).collect::<Vec<_>>());
    }
    #[test]
    fn graphics_acceptance_does_not_wait_and_failure_precedes_gate_release() {
        let system = HostWorkSystem::default();
        let gate = Arc::new(WorkGate::default());
        let (entered, started) = mpsc::sync_channel(1);
        let (release, wait) = mpsc::sync_channel(1);
        system
            .submit_graphics(1, gate.begin(), move || {
                entered.send(()).unwrap();
                wait.recv().unwrap();
                Err(IpcWireError::Internal("retained frontend failure"))
            })
            .unwrap();
        started.recv_timeout(Duration::from_secs(2)).unwrap();
        assert!(
            gate.wait().is_some(),
            "acceptance must not release shared state"
        );
        assert!(
            system.pending.lock().unwrap().is_empty(),
            "no suspended IPC reply"
        );
        assert!(system.require_graphics_healthy().is_ok());
        let (consumer, consumed) = mpsc::channel();
        gate.after_current(move || {
            let _ = consumer.send(());
        });
        release.send(()).unwrap();
        gate.finish();
        assert_eq!(consumed.try_recv(), Err(mpsc::TryRecvError::Disconnected));
        assert_eq!(
            system.require_graphics_healthy(),
            Err(IpcWireError::Internal("retained frontend failure"))
        );
        assert!(
            system
                .submit_graphics(1, gate.begin(), || panic!("failed owner must reject work"))
                .is_err()
        );
        assert!(gate.wait().is_none());
    }
    #[test]
    fn completion_is_ready_only_after_job_writes_and_retains_failures() {
        let system = HostWorkSystem::default();
        let written = Arc::new(AtomicBool::new(false));
        let output = written.clone();
        let error = system
            .submit((7, 3), move |_| {
                output.store(true, Ordering::Release);
                Err(IpcWireError::Internal("test host failure"))
            })
            .unwrap_err();
        let IpcWireError::PendingHostWork(wait) = error else {
            panic!("accepted work must suspend")
        };
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        loop {
            if let Some(result) = wait.0.result.lock().unwrap().as_ref() {
                assert!(written.load(Ordering::Acquire));
                assert!(matches!(
                    result,
                    Err(IpcWireError::Internal("test host failure"))
                ));
                break;
            }
            assert!(std::time::Instant::now() < deadline);
            thread::yield_now();
        }
    }
    #[test]
    fn cancellation_discards_queued_work_and_joins_the_running_owner() {
        let system = HostWorkSystem::default();
        let (entered, started) = mpsc::sync_channel(1);
        let (release, wait) = mpsc::sync_channel(1);
        system
            .submit((1, 1), move |_| {
                entered.send(()).unwrap();
                wait.recv().unwrap();
                Ok(None)
            })
            .unwrap_err();
        started.recv_timeout(Duration::from_secs(2)).unwrap();
        let called = Arc::new(AtomicBool::new(false));
        let executed = called.clone();
        system
            .submit((1, 2), move |_| {
                executed.store(true, Ordering::Release);
                Ok(None)
            })
            .unwrap_err();
        system.cancelled.store(true, Ordering::Release);
        release.send(()).unwrap();
        drop(system);
        assert!(!called.load(Ordering::Acquire));
    }
    #[test]
    fn gate_retains_wakeup_until_frontend_owner_releases_it() {
        let gate = Arc::new(WorkGate::default());
        assert!(gate.wait().is_none());
        let guard = gate.begin();
        let waiting = gate.wait().unwrap();
        assert_eq!(waiting, gate.wait().unwrap());
        drop(guard);
        assert!(gate.wait().is_none());
        drop(gate.begin());
        assert!(gate.wait().is_none());
    }
}
