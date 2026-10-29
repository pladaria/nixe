use super::*;

impl RuntimeCoordinator {
    /// Dispatch ready cores and wait for a CPU completion, host service event or
    /// the caller's next service deadline. Other native cores remain in flight.
    pub fn run_next(
        &mut self,
        host_wait: std::time::Duration,
    ) -> Result<Option<CoordinatorExecution>, CoordinatorError> {
        self.finalize_retired_threads()?;
        if self.host_paused() || self.host_stop_requested {
            return Ok(None);
        }
        if let Some(execution) = self.completed_executions.pop_front() {
            return Ok(Some(execution));
        }
        let deadline = std::time::Instant::now() + host_wait;
        loop {
            let idle = self.scheduler.idle_vcpus();
            for vcpu in idle {
                let lease = self.scheduler.claim(vcpu)?;
                if let Some(lease) = lease {
                    self.dispatch_worker(lease, None)?;
                }
            }
            self.interrupt_rescheduled_cores();
            if let Some(result) = self.workers.pending_results.pop_front() {
                let lease = self
                    .scheduler
                    .lease_for_vcpu(result.lease.vcpu)
                    .ok_or(CoordinatorError::InFlightLease(result.lease))?;
                if let Some(execution) = self.complete_worker(lease, result)? {
                    return Ok(Some(execution));
                }
                continue;
            }
            if let Some(event) = self.deferred_events.pop_front() {
                let internal = matches!(
                    event.event,
                    ExternalEvent::ThreadRetirement
                        | ExternalEvent::HostService(crate::ExternalEventSource::Timer)
                );
                self.apply_external_event(event, &mut CoordinatorDrainReport::default())?;
                if internal {
                    continue;
                }
                return Ok(None);
            }
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            let timeout = remaining;
            match self.inbox.recv_timeout(timeout)? {
                Some(crate::external_event::CoordinatorEvent::Completed(result)) => {
                    self.workers.pending_results.push_back(result);
                }
                Some(crate::external_event::CoordinatorEvent::External(event)) => {
                    let internal = matches!(
                        event.event,
                        ExternalEvent::ThreadRetirement
                            | ExternalEvent::HostService(crate::ExternalEventSource::Timer)
                    );
                    self.apply_external_event(event, &mut CoordinatorDrainReport::default())?;
                    if internal {
                        continue;
                    }
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
        let leases = self.scheduler.active_leases().collect();
        self.quiesce_leases(leases)
    }

    pub(super) fn quiesce_process(&mut self, process: ProcessId) -> Result<(), CoordinatorError> {
        let leases = self
            .scheduler
            .active_leases()
            .filter(|lease| lease.process == process)
            .collect();
        self.quiesce_leases(leases)
    }

    fn quiesce_leases(&mut self, leases: Vec<Lease>) -> Result<(), CoordinatorError> {
        for process in leases
            .iter()
            .map(|lease| lease.process)
            .collect::<BTreeSet<_>>()
        {
            self.processes.get(&process).unwrap().request_safepoint();
        }
        let mut first_error = None;
        for lease in leases {
            match self.receive_worker(lease) {
                Ok(execution) => self.completed_executions.push_back(execution),
                Err(CoordinatorError::Execution {
                    error: ProcessExecutionError::ConcurrentProcessStop { .. },
                    ..
                }) => {}
                Err(CoordinatorError::Execution {
                    process,
                    error: ProcessExecutionError::Cpu { fault },
                    ..
                }) if fault.kind == nixe_cpu::execution::CpuFaultKind::Stopped
                    && matches!(
                        self.processes[&process].lifecycle(),
                        nixe_scheduler::ProcessLifecycle::Terminating
                            | nixe_scheduler::ProcessLifecycle::Exited
                            | nixe_scheduler::ProcessLifecycle::Faulted
                    ) => {}
                Err(error) => {
                    first_error.get_or_insert(error);
                }
            }
        }
        first_error.map_or(Ok(()), Err)
    }

    fn interrupt_rescheduled_cores(&mut self) {
        for core in self.scheduler.take_reschedule_requests().iter() {
            if let Some(lease) = self.scheduler.lease_for_vcpu(core)
                && self
                    .scheduler
                    .selection(core)
                    .expect("configured core")
                    .thread
                    != Some(lease.thread)
            {
                self.processes[&lease.process].request_vcpu_safepoint(core);
            }
        }
    }

    fn dispatch_worker(
        &mut self,
        lease: Lease,
        report: Option<crate::ExecutionReport>,
    ) -> Result<(), CoordinatorError> {
        let events = self
            .vcpu_events
            .get(&lease.vcpu)
            .expect("a scheduler lease references a configured vCPU")
            .clone();
        let execution = self
            .processes
            .get(&lease.process)
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
        if let Err(failure) = self.workers.dispatch(
            WorkerRequest {
                lease,
                cpu_thread,
                execution,
            },
            report,
        ) {
            self.processes
                .get(&lease.process)
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
        loop {
            let worker_result = loop {
                if let Some(index) = self
                    .workers
                    .pending_results
                    .iter()
                    .position(|result| result.lease.vcpu == expected.vcpu)
                {
                    break self.workers.pending_results.remove(index).unwrap();
                }
                match self.inbox.recv()? {
                    crate::external_event::CoordinatorEvent::Completed(result) => {
                        self.workers.pending_results.push_back(result);
                    }
                    crate::external_event::CoordinatorEvent::External(event) => {
                        if matches!(
                            event.event,
                            ExternalEvent::HostService(crate::ExternalEventSource::Timer)
                        ) {
                            self.apply_external_event(
                                event,
                                &mut CoordinatorDrainReport::default(),
                            )?;
                            self.interrupt_rescheduled_cores();
                        } else {
                            self.deferred_events.push_back(event);
                        }
                    }
                }
            };
            if let Some(execution) = self.complete_worker(expected, worker_result)? {
                return Ok(execution);
            }
        }
    }

    fn complete_worker(
        &mut self,
        expected: Lease,
        worker_result: worker::WorkerResult,
    ) -> Result<Option<CoordinatorExecution>, CoordinatorError> {
        nixe_memory::trace::detail_event(
            "cpu_result_received",
            worker_result.lease.generation.get(),
            worker_result.lease.thread.get(),
            u64::from(worker_result.lease.vcpu.get()),
        );
        if worker_result.lease != expected {
            self.processes
                .get(&expected.process)
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
                    .get(&expected.process)
                    .expect("the dispatched process remains registered")
                    .abort_thread_execution(expected.thread, expected.vcpu, None);
                self.complete_failed_worker_lease(expected)?;
                return Err(CoordinatorError::Worker(failure));
            }
        };
        let result = self
            .processes
            .get(&expected.process)
            .expect("the dispatched process remains registered")
            .finish_thread_execution(
                expected.thread,
                expected.vcpu,
                worker_result.execution,
                result,
            );
        let interrupt_handled = match worker_result.interrupt {
            Ok(handled) => handled,
            // Host termination may interrupt a genuine outstanding dependency.
            Err(worker::InterruptFailure::Stopped)
                if matches!(
                    result,
                    Err(ProcessExecutionError::ConcurrentProcessStop { .. })
                ) =>
            {
                true
            }
            Err(failure) => {
                self.complete_failed_worker_lease(expected)?;
                return Err(match failure {
                    worker::InterruptFailure::Stopped => {
                        CoordinatorError::Worker(WorkerFailure::Stopped)
                    }
                    worker::InterruptFailure::Memory(fault) => CoordinatorError::ThreadLocal {
                        thread: expected.thread,
                        fault,
                    },
                    worker::InterruptFailure::Scheduler(error) => error.into(),
                });
            }
        };
        let completion = match &result {
            Ok(report) => match report.stop {
                ExecutionStop::SupervisorCall { .. } => {
                    Completion::KernelCall { interrupt_handled }
                }
                ExecutionStop::Safepoint | ExecutionStop::PendingEvent { .. } => Completion::Ready,
                ExecutionStop::LoaderReturn { .. } => Completion::Exited,
                ExecutionStop::FetchFault { .. } | ExecutionStop::UnsupportedSemantics { .. } => {
                    Completion::Faulted
                }
                _ => Completion::Waiting,
            },
            Err(ProcessExecutionError::ConcurrentProcessStop { .. }) => Completion::Ready,
            Err(ProcessExecutionError::Cpu { fault })
                if fault.kind == nixe_cpu::execution::CpuFaultKind::Stopped
                    && matches!(
                        self.processes[&expected.process].lifecycle(),
                        nixe_scheduler::ProcessLifecycle::Terminating
                            | nixe_scheduler::ProcessLifecycle::Exited
                    ) =>
            {
                Completion::Exited
            }
            Err(_) => Completion::Faulted,
        };
        let terminated = if matches!(completion, Completion::Exited | Completion::Faulted) {
            let process = self.processes.get(&expected.process).unwrap();
            let thread = process
                .thread(expected.thread)
                .expect("running kernel object");
            process.finish_thread(&thread);
            self.finalize_thread_waits(expected.process, expected.thread);
            Some(thread)
        } else {
            None
        };
        if !self.scheduler.complete(expected, completion)? {
            // The SVC raced its interrupt. Continue the kernel boundary on its
            // original core; no guest instruction or SVC effect is replayed.
            self.dispatch_worker(expected, Some(result.expect("kernel-call completion")))?;
            return Ok(None);
        }
        if let Some(thread) = terminated {
            thread.signal();
        }
        if matches!(
            &result,
            Err(ProcessExecutionError::ConcurrentProcessStop { .. })
        ) || matches!(completion, Completion::Exited | Completion::Faulted)
        {
            self.finish_process_termination(expected.process)?;
        }
        result
            .map(|report| {
                Some(CoordinatorExecution {
                    lease: expected,
                    report,
                })
            })
            .map_err(|error| CoordinatorError::Execution {
                process: expected.process,
                thread: expected.thread,
                error,
            })
    }

    fn complete_failed_worker_lease(&mut self, lease: Lease) -> Result<(), CoordinatorError> {
        let thread = self
            .processes
            .get(&lease.process)
            .unwrap()
            .thread(lease.thread)
            .expect("failed worker retains its kernel object");
        self.processes
            .get(&lease.process)
            .unwrap()
            .finish_thread(&thread);
        self.finalize_thread_waits(lease.process, lease.thread);
        self.scheduler.complete(lease, Completion::Faulted)?;
        thread.signal();
        self.finish_process_termination(lease.process)?;
        Ok(())
    }
}

#[cfg(test)]
#[path = "tests/execution.rs"]
mod tests;
