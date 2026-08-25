use super::*;

impl RuntimeCoordinator {
    /// Executes one deterministic slice using the runtime-owned adaptive
    /// quantum. Exact-budget entry points remain available for replay and
    /// deterministic tests.
    pub fn run_next_adaptive(&mut self) -> Result<Option<CoordinatorExecution>, CoordinatorError> {
        let execution = self.run_next(self.adaptive_budget.current)?;
        self.adaptive_budget
            .observe(execution.as_ref().is_some_and(|execution| {
                matches!(execution.report.stop, ExecutionStop::BudgetExhausted)
            }));
        Ok(execution)
    }

    /// Executes one parallel wave using the runtime-owned adaptive quantum.
    pub fn run_parallel_wave_adaptive(
        &mut self,
    ) -> Result<Vec<CoordinatorExecution>, CoordinatorError> {
        let executions = self.run_parallel_wave(self.adaptive_budget.current)?;
        self.adaptive_budget.observe(
            !executions.is_empty()
                && executions.iter().all(|execution| {
                    matches!(execution.report.stop, ExecutionStop::BudgetExhausted)
                }),
        );
        Ok(executions)
    }

    /// Executes at most one deterministic slice and returns its scheduler lease.
    pub fn run_next(
        &mut self,
        instruction_budget: u64,
    ) -> Result<Option<CoordinatorExecution>, CoordinatorError> {
        if let Some(lease) = self.scheduler.active_leases().next() {
            return Err(CoordinatorError::InFlightLease(lease));
        }
        let replay_dispatch = self.replay_dispatches.front().copied();
        let select = replay_dispatch.map_or(SchedulerCommand::SelectNext, |(_, lease, _)| {
            SchedulerCommand::Select(lease.vcpu)
        });
        let Some(lease) = self.select_with_deadline(select)? else {
            return Ok(None);
        };
        let instruction_budget = if let Some((sequence, expected, budget)) = replay_dispatch {
            if lease != expected {
                self.scheduler.apply(SchedulerCommand::Complete {
                    lease,
                    outcome: Completion::Preempted,
                })?;
                return Err(CoordinatorError::ReplayLeaseMismatch {
                    sequence,
                    expected,
                    observed: lease,
                });
            }
            self.replay_dispatches.pop_front();
            budget
        } else {
            instruction_budget
        };
        self.dispatch_worker(lease, instruction_budget)?;
        self.receive_worker(lease).map(Some)
    }

    /// Dispatches one bounded wave across every currently idle emulated vCPU.
    /// The method is available only in explicitly selected parallel mode and
    /// does not return until every dispatched lease has been reconciled.
    pub fn run_parallel_wave(
        &mut self,
        instruction_budget: u64,
    ) -> Result<Vec<CoordinatorExecution>, CoordinatorError> {
        if self.execution_mode != VcpuExecutionMode::Parallel {
            return Err(CoordinatorError::ParallelModeRequired);
        }
        if let Some(lease) = self.scheduler.active_leases().next() {
            return Err(CoordinatorError::InFlightLease(lease));
        }
        let mut dispatched = Vec::new();
        let mut first_error = None;
        loop {
            let idle: Vec<_> = self.scheduler.idle_vcpus().collect();
            for vcpu in idle {
                let SchedulerDecision::Selected(lease) =
                    self.scheduler.apply(SchedulerCommand::Select(vcpu))?
                else {
                    unreachable!("select commands always produce a selected decision")
                };
                let Some(lease) = lease else {
                    continue;
                };
                match self.dispatch_worker(lease, instruction_budget) {
                    Ok(()) => dispatched.push(lease),
                    Err(error) => {
                        first_error.get_or_insert(error);
                    }
                }
            }
            if !dispatched.is_empty()
                || first_error.is_some()
                || !self.fast_forward_to_next_deadline()?
            {
                break;
            }
        }

        let mut executions = Vec::with_capacity(dispatched.len());
        for lease in dispatched {
            match self.receive_worker(lease) {
                Ok(execution) => executions.push(execution),
                Err(error) => {
                    first_error.get_or_insert(error);
                }
            }
        }
        first_error.map_or(Ok(executions), Err)
    }

    fn select_with_deadline(
        &mut self,
        command: SchedulerCommand,
    ) -> Result<Option<Lease>, CoordinatorError> {
        let SchedulerDecision::Selected(lease) = self.scheduler.apply(command.clone())? else {
            unreachable!("select commands always produce a selected decision")
        };
        if lease.is_some() || !self.fast_forward_to_next_deadline()? {
            return Ok(lease);
        }
        let SchedulerDecision::Selected(lease) = self.scheduler.apply(command)? else {
            unreachable!("select commands always produce a selected decision")
        };
        Ok(lease)
    }

    fn dispatch_worker(
        &mut self,
        lease: Lease,
        instruction_budget: u64,
    ) -> Result<(), CoordinatorError> {
        self.record_dispatch(lease, instruction_budget);

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
        let executor = WorkerExecutorKey {
            process: lease.process,
            domain: process.engine_domain_id(),
        };
        let fallback = process
            .fallback_engine_domain_id()
            .map(|domain| WorkerExecutorKey {
                process: lease.process,
                domain,
            });
        if let Err(failure) = self.workers.dispatch(WorkerRequest {
            lease,
            executor,
            fallback,
            execution,
        }) {
            self.processes
                .get_mut(&lease.process)
                .expect("the dispatched process remains registered")
                .abort_thread_execution(lease.thread, lease.vcpu, failure.request.execution);
            self.complete_failed_worker_lease(lease)?;
            return Err(CoordinatorError::Worker(failure.failure));
        }
        Ok(())
    }

    fn receive_worker(
        &mut self,
        expected: Lease,
    ) -> Result<CoordinatorExecution, CoordinatorError> {
        let worker_result = match self.workers.receive(expected.vcpu) {
            Ok(result) => result,
            Err(failure) => {
                self.processes
                    .get_mut(&expected.process)
                    .expect("the dispatched process remains registered")
                    .lose_thread_execution();
                self.complete_failed_worker_lease(expected)?;
                return Err(CoordinatorError::Worker(failure));
            }
        };
        if worker_result.lease != expected {
            self.processes
                .get_mut(&expected.process)
                .expect("the dispatched process remains registered")
                .abort_thread_execution(expected.thread, expected.vcpu, worker_result.execution);
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
                    .abort_thread_execution(
                        expected.thread,
                        expected.vcpu,
                        worker_result.execution,
                    );
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
            Err(_) => Completion::Faulted,
        };
        if let Ok(report) = &result {
            self.record_completion(expected, report);
        }
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
        self.record_dispatch_sequences.remove(&lease.vcpu);
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
            _ => Completion::Waiting,
        };
        Ok(completion)
    }
}
