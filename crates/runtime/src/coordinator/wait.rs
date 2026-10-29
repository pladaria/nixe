use super::*;

impl RuntimeCoordinator {
    pub(super) fn finalize_retired_threads(&mut self) -> Result<(), CoordinatorError> {
        while let Some(retired) = self.inbox.sender.take_retired_thread() {
            let (process, thread) = (retired.process, retired.thread);
            let Some(view) = self.scheduler.thread(thread) else {
                // Forced process/machine teardown already retired this identity.
                continue;
            };
            debug_assert_eq!(view.process, process);
            debug_assert!(matches!(
                view.lifecycle,
                nixe_scheduler::ThreadLifecycle::Created
                    | nixe_scheduler::ThreadLifecycle::Exited
                    | nixe_scheduler::ThreadLifecycle::Faulted
            ));
            if self.processes.contains_key(&process) {
                self.finalize_thread_waits(process, thread);
            }
            if let Some(owner) = retired.owner.upgrade() {
                owner
                    .retire_thread(thread)
                    .expect("canonical thread is registered");
            }
            self.scheduler.unregister(thread)?;
        }
        Ok(())
    }

    #[must_use]
    pub fn event_sender(&self) -> ExternalEventSender {
        self.inbox.sender()
    }

    #[must_use]
    pub const fn host_stop_requested(&self) -> bool {
        self.host_stop_requested
    }

    #[must_use]
    pub fn host_paused(&self) -> bool {
        self.virtual_clock.is_paused()
    }

    /// Shared emulation timeline for host services; excludes host pauses.
    #[must_use]
    pub fn emulation_elapsed(&self) -> std::time::Duration {
        self.virtual_clock.host_elapsed()
    }

    /// Waits on ingress while paused, ignoring frozen guest deadlines. Host
    /// stop and device notifications remain live; no guest work is dispatched.
    pub fn wait_while_host_paused(&mut self) -> Result<(), CoordinatorError> {
        while self.host_paused() && !self.host_stop_requested {
            let event = self.inbox.recv()?;
            self.accept_ingress(event, &mut CoordinatorDrainReport::default())?;
        }
        Ok(())
    }

    /// Completes only the current generation after its owned result is installed.
    pub fn complete_wait(&mut self, token: WakeToken) -> Result<bool, CoordinatorError> {
        self.apply_wake(token)
    }

    #[must_use]
    pub fn virtual_time_ns(&self) -> u64 {
        self.virtual_clock.scheduler_time_ns()
    }

    pub fn advance_virtual_time(&mut self, nanoseconds: u64) -> Result<usize, CoordinatorError> {
        Ok(self
            .scheduler
            .advance_time(self.virtual_time_ns().saturating_add(nanoseconds)))
    }

    /// Drains host ingress without sleeping; guest readiness is already kernel-owned.
    pub fn drain_external_events(&mut self) -> Result<CoordinatorDrainReport, CoordinatorError> {
        let mut report = CoordinatorDrainReport::default();
        while let Some(event) = self.deferred_events.pop_front() {
            self.apply_external_event(event, &mut report)?;
        }
        while let Some(event) = self.inbox.try_recv()? {
            self.accept_ingress(event, &mut report)?;
        }
        Ok(report)
    }

    fn accept_ingress(
        &mut self,
        event: crate::external_event::CoordinatorEvent,
        report: &mut CoordinatorDrainReport,
    ) -> Result<(), CoordinatorError> {
        match event {
            crate::external_event::CoordinatorEvent::External(event) => {
                self.apply_external_event(event, report)
            }
            crate::external_event::CoordinatorEvent::Completed(result) => {
                self.workers.pending_results.push_back(result);
                self.finalize_retired_threads()
            }
        }
    }

    pub(super) fn apply_external_event(
        &mut self,
        event: crate::SequencedExternalEvent,
        report: &mut CoordinatorDrainReport,
    ) -> Result<(), CoordinatorError> {
        self.inbox.consume_external();
        report.received += 1;
        report.first_sequence.get_or_insert(event.sequence);
        report.last_sequence = Some(event.sequence);
        match event.event {
            ExternalEvent::HostStop => self.host_stop_requested = true,
            ExternalEvent::HostPause(paused) => {
                if paused {
                    self.quiesce()?;
                }
                self.virtual_clock.set_paused(paused);
            }
            ExternalEvent::HostService(crate::ExternalEventSource::Timer) => {
                if self.scheduler.acknowledge_timer_notification() {
                    return Err(CoordinatorError::TimerWorkerFailed);
                }
            }
            ExternalEvent::HostService(_) | ExternalEvent::ThreadRetirement => {}
        }
        self.finalize_retired_threads()
    }

    pub(super) fn apply_wake(&mut self, token: WakeToken) -> Result<bool, CoordinatorError> {
        match self.scheduler.finish_wait(token) {
            Ok(_) => Ok(true),
            Err(SchedulerError::StaleWake(_)) => Ok(false),
            Err(error) => Err(error.into()),
        }
    }

    // Finalization follows the canonical held nodes into each waiter's process.
    // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/k_thread.cpp#L352-L380
    pub(super) fn finalize_thread_waits(&mut self, process: ProcessId, thread: GuestThreadId) {
        self.scheduler.cancel_thread_property(thread);
        self.processes[&process]
            .address_waits()
            .lock()
            .release_thread(thread);
        while let Some((process, waiter)) = self.scheduler.take_held_waiter(thread) {
            self.processes[&process]
                .address_waits()
                .lock()
                .cancel_held_waiter(waiter);
        }
    }
}
