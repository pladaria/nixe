use super::*;

const MAX_EXIT_STACK_FRAMES: usize = 32;
impl RunnableProcess {
    pub(crate) fn route_supervisor_call_for<D: ExceptionDispatcher>(
        self: &Arc<Self>,
        thread_id: nixe_scheduler::GuestThreadId,
        vcpu: nixe_scheduler::VirtualCpuId,
        stop: &ExecutionStop,
        dispatcher: &mut D,
    ) -> Result<(Arc<crate::ThreadObject>, ExceptionHandlingResult<D::Fault>), ExceptionRouteError>
    {
        let request = stop
            .exception_dispatch_request()
            .filter(|request| request.kind() == nixe_cpu::exception::ExceptionKind::SupervisorCall)
            .ok_or(ExceptionRouteError::NotSupervisorCall)?;
        let thread = self
            .thread(thread_id)
            .ok_or(ExceptionRouteError::UnknownThread(thread_id))?;
        let state = thread
            .take_state()
            .expect("stopped thread owns its registers");
        let current = execution::current_location(self.cpu, &state);
        if request.source() != current {
            thread.restore_state(state);
            return Err(ExceptionRouteError::SourceMismatch {
                requested: request.source(),
                current,
            });
        }
        Ok(
            self.dispatch_context(thread, vcpu, request.source(), state, |context| {
                dispatcher.dispatch(context, request)
            }),
        )
    }
    pub(crate) fn resume_supervisor_call_for<F>(
        self: &Arc<Self>,
        thread_id: nixe_scheduler::GuestThreadId,
        vcpu: nixe_scheduler::VirtualCpuId,
        source: LocationDescriptor,
        continuation: impl FnOnce(&mut ExceptionDispatchContext<'_>) -> ExceptionDispatchOutcome<F>,
    ) -> Result<(Arc<crate::ThreadObject>, ExceptionHandlingResult<F>), ExceptionRouteError> {
        let thread = self
            .thread(thread_id)
            .ok_or(ExceptionRouteError::UnknownThread(thread_id))?;
        let state = thread
            .take_state()
            .expect("stopped thread owns its registers");
        Ok(self.dispatch_context(thread, vcpu, source, state, continuation))
    }
    fn dispatch_context<F>(
        self: &Arc<Self>,
        thread: Arc<crate::ThreadObject>,
        vcpu: nixe_scheduler::VirtualCpuId,
        source: LocationDescriptor,
        mut state: GuestCpuState,
        operation: impl FnOnce(&mut ExceptionDispatchContext<'_>) -> ExceptionDispatchOutcome<F>,
    ) -> (Arc<crate::ThreadObject>, ExceptionHandlingResult<F>) {
        // Continuations intentionally retain the original SVC location after PC advancement.
        let thread_context =
            ExceptionThreadContext::new(thread.id(), vcpu, &thread, &mut state, thread.tls_base);
        let mut context = ExceptionDispatchContext::new(source, self, thread_context);
        let outcome = operation(&mut context);
        let result = self.apply_supervisor_call_outcome(&thread, source, &mut state, outcome);
        thread.restore_state(state);
        (thread, result)
    }
    fn apply_supervisor_call_outcome<F>(
        &self,
        thread: &Arc<crate::ThreadObject>,
        source: LocationDescriptor,
        state: &mut GuestCpuState,
        outcome: ExceptionDispatchOutcome<F>,
    ) -> ExceptionHandlingResult<F> {
        match outcome {
            ExceptionDispatchOutcome::Resume(continuation) => {
                state.set_pc(supervisor_call_pc(source, continuation));
                ExceptionHandlingResult::Resumed
            }
            ExceptionDispatchOutcome::Suspend(continuation) => {
                state.set_pc(supervisor_call_pc(source, continuation));
                ExceptionHandlingResult::Suspended
            }
            ExceptionDispatchOutcome::Reject { diagnostic } => {
                state.set_pc(supervisor_call_pc(source, ExceptionResume::Next));
                ExceptionHandlingResult::Rejected(diagnostic)
            }
            ExceptionDispatchOutcome::Terminate {
                scope,
                exit_code,
                reason,
            } => {
                state.set_pc(source.pc.get());
                thread.set_exit(ThreadExit {
                    requested_scope: scope,
                    exit_code,
                    source: Some(source),
                });
                let last = self.finish_thread(thread);
                if scope == ExceptionTerminationScope::Process || last {
                    let context = state.register_context();
                    let frames = self.capture_exit_stack_frames(thread.id(), &context);
                    self.finish_process(
                        nixe_scheduler::ProcessLifecycle::Exited,
                        Some(ProcessExit {
                            cause: match reason {
                                ExceptionTerminationReason::Break {
                                    reason,
                                    info,
                                    size,
                                    payload,
                                } => ProcessExitCause::GuestBreak {
                                    reason,
                                    info,
                                    size,
                                    payload,
                                },
                                ExceptionTerminationReason::Requested => {
                                    if scope == ExceptionTerminationScope::CurrentThread {
                                        ProcessExitCause::LastThreadExited
                                    } else {
                                        ProcessExitCause::ProcessRequested
                                    }
                                }
                            },
                            exit_code,
                            source: Some(source),
                            thread_id: thread.thread_id(),
                            context: Some(Box::new(context)),
                            frames,
                        }),
                    );
                }
                ExceptionHandlingResult::Terminated {
                    scope,
                    exit_code,
                    reason,
                }
            }
            ExceptionDispatchOutcome::Fault(fault) => {
                state.set_pc(source.pc.get());
                if self.finish_thread(thread) {
                    self.finish_process(nixe_scheduler::ProcessLifecycle::Faulted, None);
                }
                ExceptionHandlingResult::Fault(fault)
            }
        }
    }
    fn capture_exit_stack_frames(
        &self,
        thread_id: nixe_scheduler::GuestThreadId,
        context: &nixe_cpu::state::RegisterContext,
    ) -> Box<[GuestStackFrame]> {
        let Some(thread) = self.thread(thread_id) else {
            return Box::new([]);
        };
        let stack_top = thread.stack_top.get();
        // CreateThread supplies only the top of a guest-owned stack. Its
        // allocation's lower bound is unknown; walk the active stack above SP
        // and let bounded memory reads reject unmapped frame records.
        let stack_bottom = if thread.stack_bottom == thread.stack_top {
            context.sp
        } else {
            thread.stack_bottom.get()
        };
        let mut frame_pointer = context.x[29];
        let mut frames = Vec::with_capacity(MAX_EXIT_STACK_FRAMES);
        while frames.len() < MAX_EXIT_STACK_FRAMES
            && frame_pointer.is_multiple_of(16)
            && frame_pointer >= stack_bottom
            && frame_pointer
                .checked_add(16)
                .is_some_and(|end| end <= stack_top)
        {
            let mut record = [0_u8; 16];
            if !matches!(
                self.memory
                    .space()
                    .read_into(GuestVirtualAddress::new(frame_pointer), &mut record),
                Ok(nixe_memory::Access::Ready(()))
            ) {
                break;
            }
            let previous = u64::from_le_bytes(record[..8].try_into().unwrap());
            let return_address = u64::from_le_bytes(record[8..].try_into().unwrap());
            frames.push(GuestStackFrame {
                frame_pointer,
                return_address,
            });
            if previous <= frame_pointer {
                break;
            }
            frame_pointer = previous;
        }
        frames.into_boxed_slice()
    }
}
