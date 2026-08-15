use super::*;

impl RuntimeCoordinator {
    /// Executes at most one deterministic slice and returns its scheduler lease.
    pub fn run_next(
        &mut self,
        instruction_budget: u64,
    ) -> Result<Option<CoordinatorExecution>, CoordinatorError> {
        if let Some(lease) = self.in_flight.values().next().copied() {
            return Err(CoordinatorError::InFlightLease(lease));
        }
        let replay_dispatch = self.replay_dispatches.front().copied();
        let select = replay_dispatch.map_or(SchedulerCommand::SelectNext, |(_, lease, _)| {
            SchedulerCommand::Select(lease.vcpu)
        });
        let SchedulerDecision::Selected(lease) = self.scheduler.apply(select.clone())? else {
            unreachable!("select commands always produce a selected decision")
        };
        let lease = match lease {
            Some(lease) => lease,
            None => {
                if !self.fast_forward_to_next_deadline()? {
                    return Ok(None);
                }
                let SchedulerDecision::Selected(lease) = self.scheduler.apply(select)? else {
                    unreachable!("select commands always produce a selected decision")
                };
                let Some(lease) = lease else {
                    return Ok(None);
                };
                lease
            }
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
        self.in_flight.insert(lease.vcpu, lease);
        let slot = self
            .vcpu_slots
            .get_mut(&lease.vcpu)
            .expect("scheduler leases only configured vCPUs");
        slot.begin(lease);
        self.record_dispatch(lease, instruction_budget);
        let execution_result = match self.processes.get_mut(&lease.process) {
            Some(process) => {
                process.begin_thread_execution(lease.thread, lease.vcpu, instruction_budget)
            }
            None => {
                self.scheduler.apply(SchedulerCommand::Complete {
                    lease,
                    outcome: Completion::Faulted,
                })?;
                self.in_flight.remove(&lease.vcpu);
                self.vcpu_slots
                    .get_mut(&lease.vcpu)
                    .expect("leased vCPU slot remains configured")
                    .finish(lease);
                return Err(CoordinatorError::UnknownProcess(lease.process));
            }
        };
        let execution = match execution_result {
            Ok(execution) => execution,
            Err(error) => {
                self.complete_failed_worker_lease(lease)?;
                return Err(CoordinatorError::Execution {
                    process: lease.process,
                    thread: lease.thread,
                    error,
                });
            }
        };
        let executor = WorkerExecutorKey {
            process: lease.process,
            domain: self
                .processes
                .get(&lease.process)
                .expect("the dispatched process remains registered")
                .engine_domain_id(),
        };
        let fallback = self
            .processes
            .get(&lease.process)
            .and_then(RunnableProcess::fallback_engine_domain_id)
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
        let worker_result = match self.workers.receive(lease.vcpu) {
            Ok(result) => result,
            Err(failure) => {
                self.processes
                    .get_mut(&lease.process)
                    .expect("the dispatched process remains registered")
                    .lose_thread_execution(lease.thread);
                self.complete_failed_worker_lease(lease)?;
                return Err(CoordinatorError::Worker(failure));
            }
        };
        if worker_result.lease != lease {
            let received = worker_result.lease;
            self.processes
                .get_mut(&lease.process)
                .expect("the dispatched process remains registered")
                .abort_thread_execution(lease.thread, lease.vcpu, worker_result.execution);
            self.complete_failed_worker_lease(lease)?;
            return Err(CoordinatorError::Worker(WorkerFailure::StaleResult {
                expected: lease,
                received,
            }));
        }
        let result = match worker_result.outcome {
            Ok(report) => self
                .processes
                .get_mut(&lease.process)
                .expect("the dispatched process remains registered")
                .finish_thread_execution(
                    lease.thread,
                    lease.vcpu,
                    worker_result.execution,
                    Ok(report),
                ),
            Err(WorkerRunFailure::Execution(error)) => self
                .processes
                .get_mut(&lease.process)
                .expect("the dispatched process remains registered")
                .finish_thread_execution(
                    lease.thread,
                    lease.vcpu,
                    worker_result.execution,
                    Err(error),
                ),
            Err(WorkerRunFailure::Worker(failure)) => {
                self.processes
                    .get_mut(&lease.process)
                    .expect("the dispatched process remains registered")
                    .abort_thread_execution(lease.thread, lease.vcpu, worker_result.execution);
                self.complete_failed_worker_lease(lease)?;
                return Err(CoordinatorError::Worker(failure));
            }
        };
        let completion = match &result {
            Ok(report) => completion_for_stop(&report.stop),
            Err(_) => Completion::Faulted,
        };
        if let Ok(report) = &result {
            self.record_completion(lease, report);
        }
        let completion_result = self.scheduler.apply(SchedulerCommand::Complete {
            lease,
            outcome: completion,
        });
        self.in_flight.remove(&lease.vcpu);
        self.vcpu_slots
            .get_mut(&lease.vcpu)
            .expect("leased vCPU slot remains configured")
            .finish(lease);
        completion_result?;
        result
            .map(|report| Some(CoordinatorExecution { lease, report }))
            .map_err(|error| CoordinatorError::Execution {
                process: lease.process,
                thread: lease.thread,
                error,
            })
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
        if let Some(lease) = self.in_flight.values().next().copied() {
            return Err(CoordinatorError::InFlightLease(lease));
        }
        let mut dispatch_order = Vec::new();
        let mut first_error = None;
        loop {
            let idle: Vec<_> = self.scheduler.idle_vcpus().collect();
            for vcpu in idle {
                let SchedulerDecision::Selected(selected) =
                    self.scheduler.apply(SchedulerCommand::Select(vcpu))?
                else {
                    unreachable!("select commands always produce a selected decision")
                };
                let Some(lease) = selected else {
                    continue;
                };
                self.in_flight.insert(vcpu, lease);
                self.vcpu_slots
                    .get_mut(&vcpu)
                    .expect("scheduler leases only configured vCPUs")
                    .begin(lease);
                self.record_dispatch(lease, instruction_budget);
                let execution = self
                    .processes
                    .get_mut(&lease.process)
                    .ok_or(CoordinatorError::UnknownProcess(lease.process))?
                    .begin_thread_execution(lease.thread, vcpu, instruction_budget)
                    .map_err(|error| CoordinatorError::Execution {
                        process: lease.process,
                        thread: lease.thread,
                        error,
                    });
                let execution = match execution {
                    Ok(execution) => execution,
                    Err(error) => {
                        self.complete_failed_worker_lease(lease)?;
                        first_error.get_or_insert(error);
                        continue;
                    }
                };
                let executor = WorkerExecutorKey {
                    process: lease.process,
                    domain: self
                        .processes
                        .get(&lease.process)
                        .expect("the dispatched process remains registered")
                        .engine_domain_id(),
                };
                let fallback = self
                    .processes
                    .get(&lease.process)
                    .and_then(RunnableProcess::fallback_engine_domain_id)
                    .map(|domain| WorkerExecutorKey {
                        process: lease.process,
                        domain,
                    });
                match self.workers.dispatch(WorkerRequest {
                    lease,
                    executor,
                    fallback,
                    execution,
                }) {
                    Ok(()) => {
                        dispatch_order.push(lease);
                    }
                    Err(failure) => {
                        self.processes
                            .get_mut(&lease.process)
                            .expect("the dispatched process remains registered")
                            .abort_thread_execution(
                                lease.thread,
                                lease.vcpu,
                                failure.request.execution,
                            );
                        self.complete_failed_worker_lease(lease)?;
                        first_error.get_or_insert(CoordinatorError::Worker(failure.failure));
                    }
                }
            }

            if !dispatch_order.is_empty()
                || first_error.is_some()
                || !self.fast_forward_to_next_deadline()?
            {
                break;
            }
        }

        let mut executions = Vec::with_capacity(dispatch_order.len());
        for expected_lease in dispatch_order {
            let worker_result = match self.workers.receive(expected_lease.vcpu) {
                Ok(result) => result,
                Err(failure) => {
                    self.processes
                        .get_mut(&expected_lease.process)
                        .expect("the dispatched process remains registered")
                        .lose_thread_execution(expected_lease.thread);
                    self.complete_failed_worker_lease(expected_lease)?;
                    first_error.get_or_insert(CoordinatorError::Worker(failure));
                    continue;
                }
            };
            let lease = worker_result.lease;
            if lease != expected_lease {
                self.processes
                    .get_mut(&expected_lease.process)
                    .expect("the dispatched process remains registered")
                    .abort_thread_execution(
                        expected_lease.thread,
                        expected_lease.vcpu,
                        worker_result.execution,
                    );
                self.complete_failed_worker_lease(expected_lease)?;
                first_error.get_or_insert(CoordinatorError::Worker(WorkerFailure::StaleResult {
                    expected: expected_lease,
                    received: lease,
                }));
                continue;
            }
            let result = match worker_result.outcome {
                Ok(report) => self
                    .processes
                    .get_mut(&lease.process)
                    .expect("the dispatched process remains registered")
                    .finish_thread_execution(
                        lease.thread,
                        lease.vcpu,
                        worker_result.execution,
                        Ok(report),
                    ),
                Err(WorkerRunFailure::Execution(error)) => self
                    .processes
                    .get_mut(&lease.process)
                    .expect("the dispatched process remains registered")
                    .finish_thread_execution(
                        lease.thread,
                        lease.vcpu,
                        worker_result.execution,
                        Err(error),
                    ),
                Err(WorkerRunFailure::Worker(failure)) => {
                    self.processes
                        .get_mut(&lease.process)
                        .expect("the dispatched process remains registered")
                        .abort_thread_execution(lease.thread, lease.vcpu, worker_result.execution);
                    self.complete_failed_worker_lease(lease)?;
                    first_error.get_or_insert(CoordinatorError::Worker(failure));
                    continue;
                }
            };
            let completion = match &result {
                Ok(report) => completion_for_stop(&report.stop),
                Err(_) => Completion::Faulted,
            };
            if let Ok(report) = &result {
                self.record_completion(lease, report);
            }
            self.scheduler.apply(SchedulerCommand::Complete {
                lease,
                outcome: completion,
            })?;
            self.in_flight.remove(&lease.vcpu);
            self.vcpu_slots
                .get_mut(&lease.vcpu)
                .expect("leased vCPU slot remains configured")
                .finish(lease);
            match result {
                Ok(report) => executions.push(CoordinatorExecution { lease, report }),
                Err(error) => {
                    first_error.get_or_insert(CoordinatorError::Execution {
                        process: lease.process,
                        thread: lease.thread,
                        error,
                    });
                }
            }
        }
        first_error.map_or(Ok(executions), Err)
    }

    fn complete_failed_worker_lease(&mut self, lease: Lease) -> Result<(), CoordinatorError> {
        self.record_dispatch_sequences.remove(&lease.vcpu);
        self.scheduler.apply(SchedulerCommand::Complete {
            lease,
            outcome: Completion::Faulted,
        })?;
        self.in_flight.remove(&lease.vcpu);
        self.vcpu_slots
            .get_mut(&lease.vcpu)
            .expect("leased vCPU slot remains configured")
            .finish(lease);
        Ok(())
    }
}
