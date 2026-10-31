//! Persistent physical cores with local exception handling.
use crate::kernel_operation::{CoreSignal, KernelQueue};
use crate::{
    DispatchFault, ExceptionDispatcher, ExceptionHandlingResult, ExternalEvent, ExternalEventInbox,
    ExternalEventSender, ExternalEventSource, ProcessTeardownReport, RunnableProcess,
};
use nixe_scheduler::{
    CoreSet, GuestThreadId, ProcessId, ScheduledThreadConfig, SchedulerState, VirtualCpuId,
};
use std::{
    collections::{BTreeMap, VecDeque},
    sync::{
        Arc, Mutex, RwLock,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    task::Waker,
    thread::JoinHandle,
    time::Duration,
};
mod core;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProcessRegistration {
    pub priority: i32,
    pub ideal_vcpu: Option<VirtualCpuId>,
    pub affinity: CoreSet,
}

#[derive(Debug)]
pub enum RuntimeError {
    ForeignKernel,
    UnknownProcess(ProcessId),
    Scheduler(nixe_scheduler::SchedulerError),
    Host(crate::ExternalEventSendError),
    Startup(std::io::Error),
    Execution {
        process: ProcessId,
        thread: GuestThreadId,
        diagnostic: Box<str>,
    },
    CorePanicked(VirtualCpuId),
    ProcessTeardown {
        process: ProcessId,
        failure: crate::ProcessTeardownFailure,
    },
}
impl std::fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "runtime failure: {self:?}")
    }
}
impl std::error::Error for RuntimeError {}
impl From<nixe_scheduler::SchedulerError> for RuntimeError {
    fn from(error: nixe_scheduler::SchedulerError) -> Self {
        Self::Scheduler(error)
    }
}
impl From<crate::ExternalEventSendError> for RuntimeError {
    fn from(error: crate::ExternalEventSendError) -> Self {
        Self::Host(error)
    }
}

struct Shared {
    scheduler: Arc<SchedulerState>,
    processes: RwLock<BTreeMap<ProcessId, Arc<RunnableProcess>>>,
    signals: Box<[Arc<CoreSignal>]>,
    stopped: AtomicBool,
    paused: AtomicBool,
    pause_generation: AtomicU64,
    faults: Mutex<VecDeque<RuntimeError>>,
    rejected: Mutex<BTreeMap<String, log::Level>>,
    abandoned: Mutex<Vec<Arc<KernelQueue>>>,
    failed: AtomicBool,
    events: ExternalEventSender,
}
impl Shared {
    fn notify_all(&self) {
        for signal in &self.signals {
            signal.notify();
        }
    }
    fn process(&self, id: ProcessId) -> Option<Arc<RunnableProcess>> {
        self.processes.read().unwrap().get(&id).cloned()
    }
    fn fail_core(&self) {
        if self.failed.swap(true, Ordering::AcqRel) {
            return;
        }
        let processes: Vec<_> = self.processes.read().unwrap().values().cloned().collect();
        for process in processes {
            process.finish_process(nixe_scheduler::ProcessLifecycle::Faulted, None);
            self.close_admission(&process);
        }
        self.notify_all();
    }
    fn drain_abandoned(&self) {
        let queues = self.abandoned.lock().unwrap().clone();
        for queue in queues {
            core::drain(self, &queue);
        }
    }
    fn record_rejection(
        &self,
        process: &RunnableProcess,
        thread: GuestThreadId,
        source: nixe_cpu::location::LocationDescriptor,
        generation: u64,
        level: log::Level,
        diagnostic: String,
    ) {
        let report = {
            let mut rejected = self.rejected.lock().unwrap();
            match rejected.entry(diagnostic.clone()) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(level);
                    true
                }
                std::collections::btree_map::Entry::Occupied(mut entry) if level < *entry.get() => {
                    entry.insert(level);
                    true
                }
                _ => false,
            }
        };
        if report {
            log::log!(
                level,
                "guest operation returned a Horizon error: {diagnostic}; \
                 process={} thread={} lifecycle={:?} admitted_generation={} generation={} \
                 source=[{source}]",
                process.process_id(),
                thread.get(),
                process.lifecycle(),
                generation,
                process.generation(),
            );
        }
    }
    fn status_changed(&self) {
        self.events.notify_host_service(ExternalEventSource::Worker);
    }
    fn fail(&self, error: RuntimeError) {
        self.faults.lock().unwrap().push_back(error);
        self.status_changed();
    }
    fn finalize_thread_waits(&self, process: &RunnableProcess, thread: GuestThreadId) {
        self.scheduler.cancel_thread_property(thread);
        process.address_waits().lock().release_thread(thread);
        while let Some((process, waiter)) = self.scheduler.take_held_waiter(thread) {
            if let Some(owner) = self.process(process) {
                owner.address_waits().lock().cancel_held_waiter(waiter);
            }
        }
    }
    fn close_admission(&self, process: &RunnableProcess) {
        let id = ProcessId::new(process.process_id());
        let closed = self.scheduler.request_process_termination(id);
        process.request_execution_stop();
        if closed {
            for signal in &self.signals {
                signal.retire_process(id);
            }
        }
    }
    fn finish_termination(&self, process: &Arc<RunnableProcess>) -> Result<(), RuntimeError> {
        let id = ProcessId::new(process.process_id());
        if process.lifecycle() != nixe_scheduler::ProcessLifecycle::Terminating {
            return Ok(());
        }
        self.close_admission(process);
        if self
            .scheduler
            .active_leases()
            .any(|lease| lease.process == id)
        {
            return Ok(());
        }
        let Some(lifecycle) = process.take_termination() else {
            return Ok(());
        };
        let exit_code = process.exit().map_or(0, |exit| exit.exit_code);
        for thread in process.threads().ids() {
            if self.scheduler.thread(thread).is_some_and(|view| {
                !matches!(
                    view.lifecycle,
                    nixe_scheduler::ThreadLifecycle::Exited
                        | nixe_scheduler::ThreadLifecycle::Faulted
                )
            }) {
                let object = process.terminate_thread(
                    thread,
                    exit_code,
                    crate::ExceptionTerminationScope::Process,
                );
                self.finalize_thread_waits(process, thread);
                self.scheduler.terminate(thread, false)?;
                if let Some(object) = object {
                    object.signal();
                }
            }
        }
        process.finish_termination(lifecycle);
        self.notify_all();
        self.status_changed();
        Ok(())
    }
}

/// The supervisor owns lifecycle only. CPU and kernel progress belongs to cores.
pub struct Runtime {
    shared: Arc<Shared>,
    inbox: ExternalEventInbox,
    workers: Vec<(VirtualCpuId, JoinHandle<()>)>,
    next_process: u64,
    host_stop: bool,
    started: bool,
}
impl Runtime {
    pub fn try_new(scheduler: Arc<SchedulerState>) -> Result<Self, RuntimeError> {
        let inbox = ExternalEventInbox::bounded(1024)?;
        scheduler
            .start_timers(Waker::from(Arc::new(inbox.sender())))
            .map_err(RuntimeError::Startup)?;
        let shared = Arc::new(Shared {
            signals: scheduler
                .profile()
                .vcpus()
                .iter()
                .map(|_| Arc::new(CoreSignal::default()))
                .collect(),
            scheduler,
            processes: RwLock::default(),
            stopped: AtomicBool::new(false),
            paused: AtomicBool::new(false),
            pause_generation: AtomicU64::new(0),
            faults: Mutex::default(),
            rejected: Mutex::default(),
            abandoned: Mutex::default(),
            failed: AtomicBool::new(false),
            events: inbox.sender(),
        });
        Ok(Self {
            shared,
            inbox,
            workers: Vec::new(),
            next_process: 1,
            host_stop: false,
            started: false,
        })
    }
    pub fn scheduler(&self) -> &Arc<SchedulerState> {
        &self.shared.scheduler
    }
    pub fn event_sender(&self) -> ExternalEventSender {
        self.inbox.sender()
    }
    pub fn process(&self, id: ProcessId) -> Option<Arc<RunnableProcess>> {
        self.shared.process(id)
    }
    pub fn register_process(
        &mut self,
        mut process: RunnableProcess,
        registration: ProcessRegistration,
    ) -> Result<ProcessId, RuntimeError> {
        if !Arc::ptr_eq(process.scheduler(), self.scheduler()) {
            return Err(RuntimeError::ForeignKernel);
        }
        let id = ProcessId::new(self.next_process);
        self.next_process = self
            .next_process
            .checked_add(1)
            .expect("process identity exhausted");
        process.assign_process_id(id);
        let thread = self.scheduler().create_thread(ScheduledThreadConfig {
            process: id,
            base_priority: registration.priority,
            ideal_vcpu: registration.ideal_vcpu,
            affinity: registration.affinity,
        })?;
        process
            .assign_main_thread_id(thread)
            .map_err(|error| RuntimeError::Execution {
                process: id,
                thread,
                diagnostic: error.to_string().into(),
            })?;
        process.address_waits().bind(id, self.scheduler().clone());
        process
            .events
            .set(self.inbox.sender())
            .expect("process registers once");
        let process = Arc::new(process);
        process.main_thread().bind_owner(&process);
        process
            .main_thread()
            .bind_retirement(id, self.inbox.sender());
        self.shared.processes.write().unwrap().insert(id, process);
        self.scheduler().make_ready(thread)?;
        Ok(id)
    }
    pub fn start<D: ExceptionDispatcher>(
        &mut self,
        dispatcher: Arc<D>,
    ) -> Result<(), RuntimeError> {
        assert!(!self.started, "cores start once");
        self.started = true;
        self.scheduler().bind_core_wakers(
            self.shared
                .signals
                .iter()
                .cloned()
                .map(Waker::from)
                .collect(),
        );
        for (index, descriptor) in self.shared.scheduler.profile().vcpus().iter().enumerate() {
            let id = descriptor.id();
            let shared = self.shared.clone();
            let dispatcher = dispatcher.clone();
            let worker = std::thread::Builder::new()
                .name(format!("nixe-vcpu-{}", id.get()))
                .spawn(move || {
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        core::run(id, index, shared.clone(), dispatcher)
                    }));
                    if result.is_err() {
                        if let Some(lease) = shared.scheduler.lease_for_vcpu(id) {
                            let _ = shared
                                .scheduler
                                .complete(lease, nixe_scheduler::Completion::Faulted);
                            if let Some(process) = shared.process(lease.process) {
                                process.finish_process(
                                    nixe_scheduler::ProcessLifecycle::Faulted,
                                    None,
                                );
                                let _ = shared.finish_termination(&process);
                            }
                        }
                        shared.fail(RuntimeError::CorePanicked(id));
                    }
                    shared.signals[index].finish();
                });
            let worker = match worker {
                Ok(worker) => worker,
                Err(error) => {
                    for signal in &self.shared.signals[index..] {
                        signal.finish();
                    }
                    self.shared.fail_core();
                    return Err(RuntimeError::Startup(error));
                }
            };
            self.workers.push((id, worker));
        }
        Ok(())
    }
    pub fn rejected_svc_kinds(&self) -> usize {
        self.shared.rejected.lock().unwrap().len()
    }
    pub fn completions(&self) -> u64 {
        self.shared
            .signals
            .iter()
            .map(|signal| signal.completions())
            .sum()
    }
    pub fn host_stop_requested(&self) -> bool {
        self.host_stop
    }
    pub fn host_paused(&self) -> bool {
        self.shared.paused.load(Ordering::Acquire)
    }
    pub fn emulation_elapsed(&self) -> Duration {
        self.scheduler().clock().host_elapsed()
    }
    fn handle_event(&mut self, event: ExternalEvent) -> Result<(), RuntimeError> {
        match event {
            ExternalEvent::HostStop => self.host_stop = true,
            ExternalEvent::HostPause(paused) => self.set_paused(paused)?,
            ExternalEvent::HostService(ExternalEventSource::Timer) => {
                if self.scheduler().acknowledge_timer_notification() {
                    return Err(RuntimeError::Execution {
                        process: ProcessId::new(0),
                        thread: GuestThreadId::new(0),
                        diagnostic: "timer worker failed".into(),
                    });
                }
            }
            ExternalEvent::HostService(_) | ExternalEvent::ThreadRetirement => {}
        }
        self.shared.drain_abandoned();
        self.finalize_retired_threads()
    }
    pub fn drain_external_events(&mut self) -> Result<(), RuntimeError> {
        while let Some(event) = self.inbox.try_recv()? {
            self.handle_event(event)?;
        }
        self.check_faults()
    }
    fn wait_for_event(&mut self) -> Result<(), RuntimeError> {
        let event = self.inbox.recv()?;
        self.handle_event(event)?;
        Ok(())
    }
    pub fn wait_events(&mut self, timeout: Duration) -> Result<(), RuntimeError> {
        if let Some(event) = self.inbox.recv_timeout(timeout)? {
            self.handle_event(event)?;
        }
        self.check_faults()
    }
    fn check_faults(&self) -> Result<(), RuntimeError> {
        if let Some(error) = self.shared.faults.lock().unwrap().pop_front() {
            return Err(error);
        }
        self.inbox.check_error().map_err(Into::into)
    }
    fn finalize_retired_threads(&self) -> Result<(), RuntimeError> {
        while let Some(retired) = self.inbox.sender.take_retired_thread() {
            let Some(view) = self.scheduler().thread(retired.thread) else {
                continue;
            };
            debug_assert_eq!(view.process, retired.process);
            if let Some(owner) = retired.owner.upgrade() {
                self.shared.finalize_thread_waits(&owner, retired.thread);
                owner
                    .retire_thread(retired.thread)
                    .expect("retained canonical thread");
            }
            self.scheduler().unregister(retired.thread)?;
        }
        Ok(())
    }
    pub fn set_paused(&mut self, paused: bool) -> Result<(), RuntimeError> {
        let generation = self.shared.pause_generation.fetch_add(1, Ordering::AcqRel) + 1;
        self.shared.paused.store(paused, Ordering::Release);
        self.scheduler().clock().set_paused(paused);
        self.shared.notify_all();
        if paused && self.started {
            for (descriptor, signal) in self
                .scheduler()
                .profile()
                .vcpus()
                .iter()
                .zip(&self.shared.signals)
            {
                if !signal.wait_acknowledged(generation) {
                    return Err(RuntimeError::CorePanicked(descriptor.id()));
                }
            }
        }
        Ok(())
    }
    pub fn quiesce(&mut self) -> Result<(), RuntimeError> {
        self.set_paused(true)
    }
    pub fn wait_while_host_paused(&mut self) -> Result<(), RuntimeError> {
        while self.host_paused() && !self.host_stop {
            // Host failures remain observable while guest execution is paused.
            // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/qt_common/render/emu_thread.cpp#L51-L80
            self.check_faults()?;
            self.wait_for_event()?;
        }
        self.check_faults()
    }
    pub fn terminate_process(&mut self, id: ProcessId) -> Result<bool, RuntimeError> {
        let process = self.process(id).ok_or(RuntimeError::UnknownProcess(id))?;
        let changed = process.terminate_from_host();
        self.shared.close_admission(&process);
        loop {
            self.shared.finish_termination(&process)?;
            if process.lifecycle() != nixe_scheduler::ProcessLifecycle::Terminating {
                break;
            }
            self.wait_for_event()?;
        }
        Ok(changed)
    }
    pub fn remove_process(&mut self, id: ProcessId) -> Result<ProcessTeardownReport, RuntimeError> {
        self.terminate_process(id)?;
        let process = self.process(id).ok_or(RuntimeError::UnknownProcess(id))?;
        self.shared.drain_abandoned();
        while process.has_native_threads()
            || process
                .threads()
                .iter()
                .any(|(_, thread)| thread.pending_wait().is_some())
        {
            self.wait_for_event()?;
        }
        self.shared.processes.write().unwrap().remove(&id);
        let report = process
            .try_teardown()
            .map_err(|failure| RuntimeError::ProcessTeardown {
                process: id,
                failure,
            });
        self.finalize_retired_threads()?;
        report
    }
    pub fn shutdown(&mut self) -> Result<(), RuntimeError> {
        let processes: Vec<_> = self
            .shared
            .processes
            .read()
            .unwrap()
            .keys()
            .copied()
            .collect();
        let mut first = None;
        for process in processes {
            if let Err(error) = self.remove_process(process) {
                first.get_or_insert(error);
            }
        }
        self.shared.stopped.store(true, Ordering::Release);
        self.shared.notify_all();
        for (id, worker) in self.workers.drain(..) {
            if worker.join().is_err() {
                first.get_or_insert(RuntimeError::CorePanicked(id));
            }
        }
        if self.scheduler().stop_timers().is_err() {
            return Err(RuntimeError::CorePanicked(VirtualCpuId::new(0)));
        }
        if let Err(error) = self.check_faults() {
            first.get_or_insert(error);
        }
        first.map_or(Ok(()), Err)
    }
}
impl Drop for Runtime {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

#[cfg(test)]
mod tests;
