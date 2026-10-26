use super::*;

impl RuntimeCoordinator {
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
            if let Some(event) = self.inbox.recv_timeout(std::time::Duration::from_secs(1))? {
                self.accept_ingress(event, &mut CoordinatorDrainReport::default())?;
            }
        }
        Ok(())
    }

    /// Atomically publishes either immediate readiness or a generation-safe wait.
    pub fn register_wait(
        &mut self,
        thread: GuestThreadId,
        readiness: Readiness,
    ) -> Result<Option<WakeToken>, CoordinatorError> {
        let decision = self
            .scheduler
            .apply(SchedulerCommand::RegisterWait { thread, readiness })?;
        match decision {
            SchedulerDecision::ReadyImmediately(_) => Ok(None),
            SchedulerDecision::WaitRegistered(token) => Ok(Some(token)),
            _ => unreachable!("wait registration has a dedicated decision"),
        }
    }

    /// Completes only the current generation after its owned result is installed.
    pub fn complete_wait(&mut self, token: WakeToken) -> Result<bool, CoordinatorError> {
        self.apply_wake(token, false)
    }

    /// Cancels exactly one current wait generation and releases its observers
    /// and deadline. A readiness wake which already won cannot be cancelled.
    pub fn cancel_wait(&mut self, token: WakeToken) -> Result<bool, CoordinatorError> {
        self.apply_wake(token, true)
    }

    #[must_use]
    pub fn virtual_time_ns(&self) -> u64 {
        self.virtual_clock.scheduler_time_ns()
    }

    /// Suspends a caller until a monotonic scheduler deadline. Non-positive
    /// Horizon values 0, -1 and -2 select the three explicit yield modes.
    pub fn sleep_thread(
        &mut self,
        thread: GuestThreadId,
        nanoseconds: i64,
    ) -> Result<Option<WakeToken>, CoordinatorError> {
        if nanoseconds <= 0 {
            let migration = match nanoseconds {
                0 => Some(nixe_scheduler::YieldMigration::None),
                -1 => Some(nixe_scheduler::YieldMigration::Suggested),
                -2 => Some(nixe_scheduler::YieldMigration::Any),
                _ => None,
            };
            self.scheduler.apply(match migration {
                Some(migration) => SchedulerCommand::Yield { thread, migration },
                None => SchedulerCommand::MakeReady(thread),
            })?;
            return Ok(None);
        }
        let token = self.register_timed_wait(thread, Some(nanoseconds as u64))?;
        Ok(Some(token))
    }

    pub fn register_timed_wait(
        &mut self,
        thread: GuestThreadId,
        timeout_ns: Option<u64>,
    ) -> Result<WakeToken, CoordinatorError> {
        let deadline = timeout_ns.map(|timeout_ns| {
            let deadline = self.virtual_time_ns().saturating_add(timeout_ns);
            let sequence = self.next_deadline_sequence;
            (deadline, sequence)
        });
        if deadline.is_some() {
            self.next_deadline_sequence = self
                .next_deadline_sequence
                .checked_add(1)
                .ok_or(CoordinatorError::DeadlineSequenceExhausted)?;
        }
        let token = self
            .register_wait(thread, Readiness::Pending)?
            .expect("a pending wait returns a wake token");
        if let Some(deadline) = deadline {
            self.deadlines.insert(deadline, token);
        }
        Ok(token)
    }

    /// Registers object readiness and its optional virtual deadline as one
    /// coordinator-owned wait. Event observers are cancelled with the wait.
    pub fn register_event_wait(
        &mut self,
        thread: GuestThreadId,
        events: impl IntoIterator<Item = crate::ReadableEventObject>,
        timeout_ns: Option<u64>,
        source: crate::ExternalEventSource,
    ) -> Result<WakeToken, CoordinatorError> {
        let token = self.register_timed_wait(thread, timeout_ns)?;
        for event in events {
            if let Err(error) = self
                .inbox
                .sender()
                .watch_readable_event(event, None, token, source)
            {
                self.release_wait_resources(thread);
                let _ = self.scheduler.apply(SchedulerCommand::CancelWait(token));
                return Err(CoordinatorError::ExternalEvent(error));
            }
        }
        Ok(token)
    }

    pub fn advance_virtual_time(&mut self, nanoseconds: u64) -> Result<usize, CoordinatorError> {
        let target = self.virtual_time_ns().saturating_add(nanoseconds);
        self.virtual_clock.advance_scheduler_to(target);
        self.wake_due_deadlines()
    }

    pub(super) fn fast_forward_to_next_deadline(&mut self) -> Result<bool, CoordinatorError> {
        // Live audio/input/device completion follows host time. Jumping ahead
        // would turn a pending device operation into an immediate guest timeout.
        if self.virtual_clock.mode() == crate::VirtualClockMode::Realtime {
            return Ok(false);
        }
        let Some((&(deadline, _), _)) = self.deadlines.first_key_value() else {
            return Ok(false);
        };
        self.virtual_clock.advance_scheduler_to(deadline);
        self.wake_due_deadlines()?;
        Ok(true)
    }

    pub(super) fn service_preemption(&mut self) {
        let Some(deadline) = self.next_preemption_ns else {
            return;
        };
        let now = self.virtual_time_ns();
        if now < deadline {
            return;
        }
        let interval = self.scheduler.profile().preemption_interval_ns().unwrap();
        // Missed host callbacks do not mean guest threads actually ran intervening
        // turns. Rotate once and preserve the periodic phase, as CoreTiming does.
        self.next_preemption_ns = Some(now.saturating_add(interval - (now - deadline) % interval));
        self.scheduler
            .apply(SchedulerCommand::Rotate)
            .expect("rotation is infallible");
    }

    pub(super) fn wake_due_deadlines(&mut self) -> Result<usize, CoordinatorError> {
        self.service_preemption();
        if self.deadlines.is_empty() {
            return Ok(0);
        }
        let now = self.virtual_time_ns();
        let mut woken = 0;
        while let Some((&key, &token)) = self.deadlines.first_key_value() {
            if key.0 > now {
                break;
            }
            self.deadlines.remove(&key);
            let handoff = self.scheduler.thread(token.thread).is_some_and(|thread| {
                thread.active_wait == Some(token)
                    && self.processes.get(&thread.process).is_some_and(|process| {
                        process.address_waits().lock().defer_timeout(token.thread)
                    })
            });
            if handoff {
                continue;
            }
            if self.apply_wake(token, false)? {
                // Address waits complete at timer expiry, before a later mutex
                // unlock or condition signal can transfer ownership to them.
                let process_id = self.scheduler.thread(token.thread).unwrap().process;
                if self
                    .processes
                    .get_mut(&process_id)
                    .unwrap()
                    .address_waits()
                    .lock()
                    .expire_thread_wait(token.thread)
                {
                    self.priority_donations
                        .retain(|donation| donation.waiter != token.thread);
                    self.recompute_effective_priorities()?;
                }
                woken += 1;
            }
        }
        Ok(woken)
    }

    /// Drains the bounded ingress without sleeping. Late and duplicate wakeups
    /// are counted and ignored after their generation loses the race.
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
                Ok(())
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
            ExternalEvent::HostService(_) => {}
            ExternalEvent::Wake { token, .. } => {
                if self.apply_wake(token, false)? {
                    report.woken += 1;
                } else {
                    report.stale += 1;
                }
            }
            ExternalEvent::CancelWait(token) => {
                if self.apply_wake(token, true)? {
                    report.cancelled += 1;
                } else {
                    report.stale += 1;
                }
            }
        }
        Ok(())
    }

    pub(super) fn apply_wake(
        &mut self,
        token: WakeToken,
        cancelled: bool,
    ) -> Result<bool, CoordinatorError> {
        self.release_wait_resources_if_current(token);
        if self.scheduler.thread(token.thread).is_none() {
            return Ok(false);
        }
        let command = if cancelled {
            SchedulerCommand::CancelWait(token)
        } else {
            SchedulerCommand::Wake(token)
        };
        match self.scheduler.apply(command) {
            Ok(_) => Ok(true),
            Err(SchedulerError::StaleWake(_)) => Ok(false),
            Err(error) => Err(error.into()),
        }
    }

    pub(super) fn release_wait_resources_if_current(&mut self, token: WakeToken) {
        if self
            .scheduler
            .thread(token.thread)
            .is_some_and(|thread| thread.active_wait == Some(token))
        {
            self.release_wait_resources(token.thread);
        } else {
            self.inbox.sender().cancel_watchers(token);
        }
    }

    pub(super) fn release_wait_resources(&mut self, thread: GuestThreadId) {
        let Some(token) = self
            .scheduler
            .thread(thread)
            .and_then(|thread| thread.active_wait)
        else {
            return;
        };
        self.inbox.sender().cancel_watchers(token);
        self.deadlines.retain(|_, candidate| *candidate != token);
    }
}
