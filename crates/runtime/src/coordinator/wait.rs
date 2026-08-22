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

    #[must_use]
    pub fn virtual_time_ns(&self) -> u64 {
        self.virtual_clock.scheduler_time_ns()
    }

    /// Suspends a caller until a deterministic virtual deadline. Non-positive
    /// Horizon sleep values are scheduler yields and become ready immediately.
    pub fn sleep_thread(
        &mut self,
        thread: GuestThreadId,
        nanoseconds: i64,
    ) -> Result<Option<WakeToken>, CoordinatorError> {
        if nanoseconds <= 0 {
            self.make_thread_ready(thread)?;
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
        let Some((&(deadline, _), _)) = self.deadlines.first_key_value() else {
            return Ok(false);
        };
        self.virtual_clock.advance_scheduler_to(deadline);
        self.wake_due_deadlines()?;
        Ok(true)
    }

    pub(super) fn wake_due_deadlines(&mut self) -> Result<usize, CoordinatorError> {
        let due: Vec<_> = self
            .deadlines
            .range(..=(self.virtual_time_ns(), u64::MAX))
            .map(|(key, token)| (*key, *token))
            .collect();
        let mut woken = 0;
        for (key, token) in due {
            self.deadlines.remove(&key);
            if self.apply_wake(token, false)? {
                woken += 1;
            }
        }
        Ok(woken)
    }

    /// Drains the bounded ingress without sleeping. Late and duplicate wakeups
    /// are counted and ignored after their generation loses the race.
    pub fn drain_external_events(&mut self) -> Result<CoordinatorDrainReport, CoordinatorError> {
        let mut report = CoordinatorDrainReport::default();
        while let Some(event) = self.inbox.try_recv_sequenced()? {
            self.apply_external_event(event, &mut report)?;
        }
        Ok(report)
    }

    /// Waits for external ingress only until the composition root must service
    /// another host-owned deadline. Expiry is observable as `None` and never
    /// manufactures guest readiness or advances virtual time.
    pub fn wait_for_external_event_for(
        &mut self,
        timeout: std::time::Duration,
    ) -> Result<Option<CoordinatorDrainReport>, CoordinatorError> {
        let Some(event) = self.inbox.recv_sequenced_timeout(timeout)? else {
            return Ok(None);
        };
        let mut report = CoordinatorDrainReport::default();
        self.apply_external_event(event, &mut report)?;
        Ok(Some(report))
    }

    pub(super) fn apply_external_event(
        &mut self,
        event: crate::SequencedExternalEvent,
        report: &mut CoordinatorDrainReport,
    ) -> Result<(), CoordinatorError> {
        self.record_external_event(event);
        report.received += 1;
        report.first_sequence.get_or_insert(event.sequence);
        report.last_sequence = Some(event.sequence);
        match event.event {
            ExternalEvent::HostStop => self.host_stop_requested = true,
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
