use super::*;

impl RunnableProcess {
    /// Routes and atomically applies one supervisor-call decision.
    ///
    /// A normal handler must return [`ExceptionResume::Next`]; this method then
    /// advances past the SVC exactly once. Retry is explicit, suspension keeps
    /// its selected continuation non-runnable, and faults retain the SVC source
    /// for deterministic diagnostics.
    pub fn route_supervisor_call<D: ExceptionDispatcher>(
        &mut self,
        stop: &ExecutionStop,
        dispatcher: &mut D,
    ) -> Result<ExceptionHandlingResult<D::Fault>, ExceptionRouteError> {
        self.route_supervisor_call_for(
            self.main_thread_id,
            nixe_scheduler::VirtualCpuId::new(0),
            stop,
            dispatcher,
        )
    }

    /// Routes an exception to the explicit thread and vCPU from a completed
    /// scheduler lease.
    pub fn route_supervisor_call_for<D: ExceptionDispatcher>(
        &mut self,
        thread_id: nixe_scheduler::GuestThreadId,
        vcpu: nixe_scheduler::VirtualCpuId,
        stop: &ExecutionStop,
        dispatcher: &mut D,
    ) -> Result<ExceptionHandlingResult<D::Fault>, ExceptionRouteError> {
        let request = stop
            .exception_dispatch_request()
            .filter(|request| request.kind() == nixe_cpu::exception::ExceptionKind::SupervisorCall)
            .ok_or(ExceptionRouteError::NotSupervisorCall)?;
        let selected = self
            .threads
            .get(thread_id)
            .ok_or(ExceptionRouteError::UnknownThread(thread_id))?;
        if selected.lifecycle != nixe_scheduler::ThreadLifecycle::Waiting {
            return Err(ExceptionRouteError::ProcessNotSuspended {
                status: self.execution_status(),
            });
        }
        let current = execution::current_location(self.cpu, selected.state());
        if request.source() != current {
            return Err(ExceptionRouteError::SourceMismatch {
                requested: request.source(),
                current,
            });
        }
        let thread = self
            .threads
            .get_mut(thread_id)
            .expect("the selected thread was validated");
        let handle = thread.handle;
        let object = thread.object.clone();
        let process = ExceptionProcessContext::new(
            ExceptionProcessMetadata {
                process_id: self.process_id,
                cpu: self.cpu,
                address_space_limit: self.address_space.exclusive_limit(),
                memory_layout: self.memory_layout,
                random_entropy: self.random_entropy,
                initial_memory_size: self.initial_memory_size,
            },
            &mut self.heap_size,
            ExceptionProcessResources {
                memory: &self.memory,
                mapping_control: &self.execution,
                canonical_memory: self.memory.as_ref(),
                mounts: &self.mounts,
                handles: &mut self.handles,
                address_waits: &mut self.address_waits,
            },
        );
        let thread =
            ExceptionThreadContext::new(thread_id, vcpu, object, handle, thread.state_mut());
        let mut context = ExceptionDispatchContext::new(process, thread);
        let outcome = dispatcher.dispatch(&mut context, request);
        self.apply_supervisor_call_outcome(thread_id, request.source(), outcome)
    }

    fn apply_supervisor_call_outcome<F>(
        &mut self,
        thread_id: nixe_scheduler::GuestThreadId,
        source: LocationDescriptor,
        outcome: ExceptionDispatchOutcome<F>,
    ) -> Result<ExceptionHandlingResult<F>, ExceptionRouteError> {
        match outcome {
            ExceptionDispatchOutcome::Resume(continuation) => {
                let target = supervisor_call_continuation(source, continuation)?;
                let cpu = self.cpu;
                install_continuation(
                    cpu,
                    self.thread_mut(thread_id)
                        .ok_or(ExceptionRouteError::UnknownThread(thread_id))?
                        .state_mut(),
                    target,
                )?;
                let thread = self
                    .thread_mut(thread_id)
                    .ok_or(ExceptionRouteError::UnknownThread(thread_id))?;
                nixe_scheduler::transition_thread(
                    &mut thread.lifecycle,
                    nixe_scheduler::ThreadLifecycle::Ready,
                )
                .expect("a waiting exception thread may resume");
                thread.wait_reason = None;
                Ok(ExceptionHandlingResult::Resumed)
            }
            ExceptionDispatchOutcome::Suspend(continuation) => {
                let target = supervisor_call_continuation(source, continuation)?;
                let cpu = self.cpu;
                let thread = self
                    .thread_mut(thread_id)
                    .ok_or(ExceptionRouteError::UnknownThread(thread_id))?;
                install_continuation(cpu, thread.state_mut(), target)?;
                thread.continuation = Some(match continuation {
                    ExceptionResume::Retry => nixe_scheduler::Continuation::Retry,
                    ExceptionResume::Next => nixe_scheduler::Continuation::Next,
                    ExceptionResume::At(target) => {
                        nixe_scheduler::Continuation::Address(target.pc.get())
                    }
                });
                Ok(ExceptionHandlingResult::Suspended)
            }
            ExceptionDispatchOutcome::Reject { diagnostic } => {
                let target = supervisor_call_continuation(source, ExceptionResume::Next)?;
                let cpu = self.cpu;
                install_continuation(
                    cpu,
                    self.thread_mut(thread_id)
                        .ok_or(ExceptionRouteError::UnknownThread(thread_id))?
                        .state_mut(),
                    target,
                )?;
                let thread = self
                    .thread_mut(thread_id)
                    .ok_or(ExceptionRouteError::UnknownThread(thread_id))?;
                nixe_scheduler::transition_thread(
                    &mut thread.lifecycle,
                    nixe_scheduler::ThreadLifecycle::Ready,
                )
                .expect("a rejected call resumes its waiting thread");
                thread.wait_reason = None;
                Ok(ExceptionHandlingResult::Rejected(diagnostic))
            }
            ExceptionDispatchOutcome::Terminate {
                scope,
                exit_code,
                reason,
            } => {
                let cpu = self.cpu;
                install_continuation(
                    cpu,
                    self.thread_mut(thread_id)
                        .ok_or(ExceptionRouteError::UnknownThread(thread_id))?
                        .state_mut(),
                    source,
                )?;
                let exit = ProcessExit {
                    cause: match reason {
                        ExceptionTerminationReason::Break { reason, info, size } => {
                            ProcessExitCause::GuestBreak { reason, info, size }
                        }
                        ExceptionTerminationReason::Requested => match scope {
                            ExceptionTerminationScope::CurrentThread => {
                                ProcessExitCause::LastThreadExited
                            }
                            ExceptionTerminationScope::Process => {
                                ProcessExitCause::ProcessRequested
                            }
                        },
                    },
                    exit_code,
                    source: Some(source),
                    thread_id: thread_id.get(),
                };
                let terminate_process = scope == ExceptionTerminationScope::Process
                    || !self.threads.iter().any(|(id, thread)| {
                        *id != thread_id
                            && !matches!(
                                thread.lifecycle,
                                nixe_scheduler::ThreadLifecycle::Exited
                                    | nixe_scheduler::ThreadLifecycle::Faulted
                            )
                    });
                if terminate_process {
                    nixe_scheduler::transition_process(
                        &mut self.lifecycle,
                        nixe_scheduler::ProcessLifecycle::Terminating,
                    )
                    .expect("a live process may terminate");
                    nixe_scheduler::transition_process(
                        &mut self.lifecycle,
                        nixe_scheduler::ProcessLifecycle::Exited,
                    )
                    .expect("a terminating process may exit");
                    self.process_exit = Some(exit);
                }
                let thread = self
                    .thread_mut(thread_id)
                    .ok_or(ExceptionRouteError::UnknownThread(thread_id))?;
                nixe_scheduler::transition_thread(
                    &mut thread.lifecycle,
                    nixe_scheduler::ThreadLifecycle::Terminating,
                )
                .expect("a waiting exception thread may terminate");
                nixe_scheduler::transition_thread(
                    &mut thread.lifecycle,
                    nixe_scheduler::ThreadLifecycle::Exited,
                )
                .expect("a terminating exception thread may exit");
                thread.exit = Some(ThreadExit {
                    requested_scope: scope,
                    exit_code,
                    source: Some(source),
                });
                thread.object.signal();
                Ok(ExceptionHandlingResult::Terminated {
                    scope,
                    exit_code,
                    reason,
                })
            }
            ExceptionDispatchOutcome::Fault(fault) => {
                let cpu = self.cpu;
                install_continuation(
                    cpu,
                    self.thread_mut(thread_id)
                        .ok_or(ExceptionRouteError::UnknownThread(thread_id))?
                        .state_mut(),
                    source,
                )?;
                let has_other_live = self.threads.iter().any(|(id, thread)| {
                    *id != thread_id
                        && !matches!(
                            thread.lifecycle,
                            nixe_scheduler::ThreadLifecycle::Exited
                                | nixe_scheduler::ThreadLifecycle::Faulted
                        )
                });
                if !has_other_live {
                    nixe_scheduler::transition_process(
                        &mut self.lifecycle,
                        nixe_scheduler::ProcessLifecycle::Faulted,
                    )
                    .expect("a live process may fault");
                }
                let thread = self
                    .thread_mut(thread_id)
                    .ok_or(ExceptionRouteError::UnknownThread(thread_id))?;
                nixe_scheduler::transition_thread(
                    &mut thread.lifecycle,
                    nixe_scheduler::ThreadLifecycle::Faulted,
                )
                .expect("a waiting exception thread may fault");
                thread.object.signal();
                Ok(ExceptionHandlingResult::Fault(fault))
            }
        }
    }
}
