use super::*;

impl RuntimeCoordinator {
    /// Reconciles one completed parallel slice without waiting for other cores.
    pub fn run_next_adaptive(&mut self) -> Result<Option<CoordinatorExecution>, CoordinatorError> {
        let execution = self.run_with_budget(None)?;
        if let Some(execution) = &execution {
            self.budgets
                .get_mut(&execution.lease.vcpu)
                .unwrap()
                .observe(matches!(
                    execution.report.stop,
                    ExecutionStop::BudgetExhausted
                ));
        }
        Ok(execution)
    }

    /// Dispatches idle cores and returns the first completed slice. Other
    /// leases remain in flight and are reconciled by subsequent calls.
    pub fn run_next(
        &mut self,
        instruction_budget: u64,
    ) -> Result<Option<CoordinatorExecution>, CoordinatorError> {
        self.run_with_budget(Some(instruction_budget))
    }

    fn run_with_budget(
        &mut self,
        instruction_budget: Option<u64>,
    ) -> Result<Option<CoordinatorExecution>, CoordinatorError> {
        if self.host_paused() {
            return Ok(None);
        }
        if let Some(execution) = self.completed_executions.pop_front() {
            return Ok(Some(execution));
        }
        self.wake_due_deadlines()?;
        loop {
            let idle: Vec<_> = self.scheduler.idle_vcpus().collect();
            for vcpu in idle {
                let SchedulerDecision::Selected(lease) =
                    self.scheduler.apply(SchedulerCommand::Select(vcpu))?
                else {
                    unreachable!("select commands always produce a selected decision")
                };
                if let Some(lease) = lease {
                    let budget = instruction_budget.unwrap_or_else(|| self.budgets[&vcpu].current);
                    self.dispatch_worker(lease, budget)?;
                }
            }
            if self.scheduler.active_leases().next().is_some() {
                break;
            }
            if !self.fast_forward_to_next_deadline()? {
                return Ok(None);
            }
        }
        self.interrupt_rescheduled_cores();
        let result = self.receive_any_with_deadlines()?;
        let lease = self
            .scheduler
            .active_leases()
            .find(|lease| lease.vcpu == result.lease.vcpu)
            .ok_or(CoordinatorError::InFlightLease(result.lease))?;
        self.complete_worker(lease, result).map(Some)
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

    fn receive_any_with_deadlines(&mut self) -> Result<worker::WorkerResult, CoordinatorError> {
        loop {
            if let Some(result) = self
                .workers
                .receive_any(self.worker_wait_timeout())
                .map_err(CoordinatorError::Worker)?
            {
                return Ok(result);
            }
            self.wake_due_deadlines()?;
            self.interrupt_rescheduled_cores();
        }
    }

    fn dispatch_worker(
        &mut self,
        lease: Lease,
        instruction_budget: u64,
    ) -> Result<(), CoordinatorError> {
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
                    .begin_thread_execution(lease.thread, lease.vcpu, instruction_budget, events)
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
        let received = loop {
            match self
                .workers
                .receive_for(expected.vcpu, self.worker_wait_timeout())
            {
                Ok(None) => {
                    self.wake_due_deadlines()?;
                    self.interrupt_rescheduled_cores();
                }
                result => break result,
            }
        };
        let worker_result = match received {
            Ok(Some(result)) => result,
            Ok(None) => unreachable!("timeouts are serviced above"),
            Err(failure) => {
                self.processes
                    .get_mut(&expected.process)
                    .expect("the dispatched process remains registered")
                    .lose_thread_execution();
                self.complete_failed_worker_lease(expected)?;
                return Err(CoordinatorError::Worker(failure));
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
            Ok(report) => self.completion_for_stop(expected, &report.stop)?,
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
        lease: Lease,
        stop: &ExecutionStop,
    ) -> Result<Completion, CoordinatorError> {
        let completion = match stop {
            ExecutionStop::BudgetExhausted
            | ExecutionStop::Safepoint
            | ExecutionStop::PendingEvent { .. } => Completion::Ready,
            ExecutionStop::LoaderReturn { .. } => Completion::Exited,
            ExecutionStop::FetchFault { .. } | ExecutionStop::UnsupportedSemantics { .. } => {
                Completion::Faulted
            }
            ExecutionStop::Scheduled { request, .. } => match request {
                SchedulerRequest::Yield => Completion::Preempted,
                SchedulerRequest::WaitForEvent => {
                    let events = self
                        .vcpu_events
                        .get(&lease.vcpu)
                        .expect("a scheduler lease references a configured vCPU");
                    if events.consume_event() {
                        Completion::Ready
                    } else {
                        self.cpu_waits.insert(
                            lease.thread,
                            CpuWait {
                                vcpu: lease.vcpu,
                                request: *request,
                            },
                        );
                        Completion::Waiting
                    }
                }
                SchedulerRequest::WaitForInterrupt => {
                    let events = self
                        .vcpu_events
                        .get(&lease.vcpu)
                        .expect("a scheduler lease references a configured vCPU");
                    if events.interrupts_pending() {
                        Completion::Ready
                    } else {
                        self.cpu_waits.insert(
                            lease.thread,
                            CpuWait {
                                vcpu: lease.vcpu,
                                request: *request,
                            },
                        );
                        Completion::Waiting
                    }
                }
                SchedulerRequest::SendEvent => {
                    self.send_event()?;
                    Completion::Ready
                }
            },
            ExecutionStop::SupervisorCall { .. } => Completion::KernelCall,
            _ => Completion::Waiting,
        };
        Ok(completion)
    }
}
