use super::*;

impl RuntimeCoordinator {
    /// Dispatch ready cores and wait for a CPU completion, host service event or
    /// the caller's next service deadline. Other native cores remain in flight.
    pub fn run_next(
        &mut self,
        host_wait: std::time::Duration,
    ) -> Result<Option<CoordinatorExecution>, CoordinatorError> {
        if self.host_paused() || self.host_stop_requested {
            return Ok(None);
        }
        if let Some(execution) = self.completed_executions.pop_front() {
            return Ok(Some(execution));
        }
        let deadline = std::time::Instant::now() + host_wait;
        loop {
            self.wake_due_deadlines()?;
            let idle: Vec<_> = self.scheduler.idle_vcpus().collect();
            for vcpu in idle {
                let SchedulerDecision::Selected(lease) =
                    self.scheduler.apply(SchedulerCommand::Select(vcpu))?
                else {
                    unreachable!()
                };
                if let Some(lease) = lease {
                    self.dispatch_worker(lease)?;
                }
            }
            self.interrupt_rescheduled_cores();
            if let Some(result) = self.workers.pending_results.pop_front() {
                let lease = self
                    .scheduler
                    .active_leases()
                    .find(|lease| lease.vcpu == result.lease.vcpu)
                    .ok_or(CoordinatorError::InFlightLease(result.lease))?;
                return self.complete_worker(lease, result).map(Some);
            }
            if self.scheduler.active_leases().next().is_none()
                && self.fast_forward_to_next_deadline()?
            {
                continue;
            }
            if let Some(event) = self.deferred_events.pop_front() {
                self.apply_external_event(event, &mut CoordinatorDrainReport::default())?;
                return Ok(None);
            }
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            let timeout = self
                .worker_wait_timeout()
                .map_or(remaining, |timeout| timeout.min(remaining));
            match self.inbox.recv_timeout(timeout)? {
                Some(crate::external_event::CoordinatorEvent::Completed(result)) => {
                    self.workers.pending_results.push_back(result);
                }
                Some(crate::external_event::CoordinatorEvent::External(event)) => {
                    self.apply_external_event(event, &mut CoordinatorDrainReport::default())?;
                    return Ok(None);
                }
                None if std::time::Instant::now() >= deadline => return Ok(None),
                None => {}
            }
        }
    }

    /// Returns every register owner to the coordinator before migration,
    /// process termination or teardown. Native contexts remain resident until
    /// an actual switch or full-state consumer. Preemption precedes waiting.
    pub fn quiesce(&mut self) -> Result<(), CoordinatorError> {
        let leases: Vec<_> = self.scheduler.active_leases().collect();
        for process in leases
            .iter()
            .map(|lease| lease.process)
            .collect::<BTreeSet<_>>()
        {
            self.processes
                .get_mut(&process)
                .unwrap()
                .request_safepoint();
        }
        let mut first_error = None;
        for lease in leases {
            match self.receive_worker(lease) {
                Ok(execution) => self.completed_executions.push_back(execution),
                Err(CoordinatorError::Execution {
                    error: ProcessExecutionError::ConcurrentProcessStop { .. },
                    ..
                }) => {}
                Err(error) => {
                    first_error.get_or_insert(error);
                }
            }
        }
        first_error.map_or(Ok(()), Err)
    }

    fn interrupt_rescheduled_cores(&mut self) {
        self.scheduler.rebalance();
        for lease in self.scheduler.preemptions() {
            self.processes[&lease.process].request_vcpu_safepoint(lease.vcpu);
        }
    }

    fn worker_wait_timeout(&self) -> Option<std::time::Duration> {
        if self.virtual_clock.mode() != crate::VirtualClockMode::Realtime {
            return None;
        }
        let deadline = self
            .next_preemption_ns
            .into_iter()
            .chain(
                self.deadlines
                    .first_key_value()
                    .map(|((deadline, _), _)| *deadline),
            )
            .min()?;
        Some(std::time::Duration::from_nanos(
            deadline.saturating_sub(self.virtual_time_ns()),
        ))
    }

    fn dispatch_worker(&mut self, lease: Lease) -> Result<(), CoordinatorError> {
        let events = self
            .vcpu_events
            .get(&lease.vcpu)
            .expect("a scheduler lease references a configured vCPU")
            .clone();
        let execution = self
            .processes
            .get_mut(&lease.process)
            .ok_or(CoordinatorError::UnknownProcess(lease.process))
            .and_then(|process| {
                process
                    .begin_thread_execution(lease.thread, lease.vcpu, events)
                    .map_err(|error| CoordinatorError::Execution {
                        process: lease.process,
                        thread: lease.thread,
                        error,
                    })
            });
        let execution = match execution {
            Ok(execution) => execution,
            Err(error) => {
                self.complete_failed_worker_lease(lease)?;
                return Err(error);
            }
        };
        let process = self
            .processes
            .get(&lease.process)
            .expect("the dispatched process remains registered");
        let cpu_thread = WorkerCpuThreadKey {
            process: lease.process,
            cpu_process: process.cpu_process_id(),
        };
        if let Err(failure) = self.workers.dispatch(WorkerRequest {
            lease,
            cpu_thread,
            execution,
        }) {
            self.processes
                .get_mut(&lease.process)
                .expect("the dispatched process remains registered")
                .abort_thread_execution(lease.thread, lease.vcpu, Some(failure.request.execution));
            self.complete_failed_worker_lease(lease)?;
            return Err(CoordinatorError::Worker(failure.failure));
        }
        Ok(())
    }

    pub(super) fn receive_worker(
        &mut self,
        expected: Lease,
    ) -> Result<CoordinatorExecution, CoordinatorError> {
        let worker_result = loop {
            if let Some(index) = self
                .workers
                .pending_results
                .iter()
                .position(|result| result.lease.vcpu == expected.vcpu)
            {
                break self.workers.pending_results.remove(index).unwrap();
            }
            match self.inbox.recv_timeout(
                self.worker_wait_timeout()
                    .unwrap_or(std::time::Duration::from_secs(5)),
            )? {
                Some(crate::external_event::CoordinatorEvent::Completed(result)) => {
                    self.workers.pending_results.push_back(result);
                }
                Some(crate::external_event::CoordinatorEvent::External(event)) => {
                    self.deferred_events.push_back(event)
                }
                None => {
                    self.wake_due_deadlines()?;
                    self.interrupt_rescheduled_cores();
                }
            }
        };
        self.complete_worker(expected, worker_result)
    }

    fn complete_worker(
        &mut self,
        expected: Lease,
        worker_result: worker::WorkerResult,
    ) -> Result<CoordinatorExecution, CoordinatorError> {
        nixe_memory::trace::detail_event(
            "cpu_result_received",
            worker_result.lease.generation.get(),
            worker_result.lease.thread.get(),
            u64::from(worker_result.lease.vcpu.get()),
        );
        if worker_result.lease != expected {
            self.processes
                .get_mut(&expected.process)
                .expect("the dispatched process remains registered")
                .abort_thread_execution(expected.thread, expected.vcpu, None);
            self.complete_failed_worker_lease(expected)?;
            return Err(CoordinatorError::Worker(WorkerFailure::StaleResult {
                expected,
                received: worker_result.lease,
            }));
        }
        let result = match worker_result.outcome {
            Ok(report) => Ok(report),
            Err(WorkerRunFailure::Execution(error)) => Err(error),
            Err(WorkerRunFailure::Worker(failure)) => {
                self.processes
                    .get_mut(&expected.process)
                    .expect("the dispatched process remains registered")
                    .abort_thread_execution(expected.thread, expected.vcpu, None);
                self.complete_failed_worker_lease(expected)?;
                return Err(CoordinatorError::Worker(failure));
            }
        };
        let result = self
            .processes
            .get_mut(&expected.process)
            .expect("the dispatched process remains registered")
            .finish_thread_execution(
                expected.thread,
                expected.vcpu,
                worker_result.execution,
                result,
            );
        let completion = match &result {
            Ok(report) => self.completion_for_stop(&report.stop)?,
            Err(ProcessExecutionError::ConcurrentProcessStop { .. }) => Completion::Ready,
            Err(_) => Completion::Faulted,
        };
        self.scheduler.apply(SchedulerCommand::Complete {
            lease: expected,
            outcome: completion,
        })?;
        result
            .map(|report| CoordinatorExecution {
                lease: expected,
                report,
            })
            .map_err(|error| CoordinatorError::Execution {
                process: expected.process,
                thread: expected.thread,
                error,
            })
    }

    fn complete_failed_worker_lease(&mut self, lease: Lease) -> Result<(), CoordinatorError> {
        self.scheduler
            .apply(SchedulerCommand::Complete {
                lease,
                outcome: Completion::Faulted,
            })
            .map(|_| ())
            .map_err(Into::into)
    }

    fn completion_for_stop(
        &mut self,
        stop: &ExecutionStop,
    ) -> Result<Completion, CoordinatorError> {
        let completion = match stop {
            ExecutionStop::Safepoint | ExecutionStop::PendingEvent { .. } => Completion::Ready,
            ExecutionStop::LoaderReturn { .. } => Completion::Exited,
            ExecutionStop::FetchFault { .. } | ExecutionStop::UnsupportedSemantics { .. } => {
                Completion::Faulted
            }
            ExecutionStop::SupervisorCall { .. } => Completion::KernelCall,
            _ => Completion::Waiting,
        };
        Ok(completion)
    }
}
