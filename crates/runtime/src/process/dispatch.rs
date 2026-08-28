use super::*;

impl RunnableProcess {
    /// Returns the process exit record retained until teardown.
    #[must_use]
    pub const fn exit(&self) -> Option<&ProcessExit> {
        self.process_exit.as_ref()
    }

    /// Returns the selected concrete CPU backend name.
    #[must_use]
    pub const fn cpu_backend_name(&self) -> &'static str {
        self.execution.backend_name()
    }

    pub(crate) const fn cpu_process_id(&self) -> nixe_cpu::execution::CpuProcessId {
        self.execution.process_id()
    }

    pub(crate) fn create_worker_cpu_thread(
        &mut self,
        vcpu: nixe_scheduler::VirtualCpuId,
    ) -> Result<execution::CpuThread, nixe_cpu::execution::CpuFault> {
        self.execution.create_worker_cpu_thread(vcpu)
    }

    pub(crate) fn request_execution_stop(&mut self) -> Result<(), nixe_cpu::execution::CpuFault> {
        self.execution.request_stop()
    }

    pub(crate) fn cpu_thread_teardown_state(&self) -> super::execution::CpuThreadTeardownState {
        self.execution.cpu_thread_teardown_state(
            std::sync::Arc::clone(&self.memory),
            self.main_thread().state().clone(),
        )
    }

    pub(crate) fn complete_cpu_thread_retirement(
        &mut self,
    ) -> Result<(), nixe_cpu::execution::CpuFault> {
        self.execution
            .complete_cpu_thread_retirement(self.memory.invalidation_cursor())
    }

    #[cfg(test)]
    pub(crate) fn request_safepoint(&mut self) {
        self.execution.request_safepoint();
    }

    /// Marks the process exited. Resource release occurs in
    /// [`Self::try_teardown`] or when the process is dropped.
    pub(crate) fn terminate_from_host(&mut self) -> bool {
        let thread_id = self.main_thread().object.thread_id();
        let exit = ProcessExit {
            cause: ProcessExitCause::HostRequested,
            exit_code: 0,
            source: None,
            thread_id,
            context: None,
            frames: Box::new([]),
        };
        let terminated = !matches!(
            self.lifecycle,
            nixe_scheduler::ProcessLifecycle::Exited | nixe_scheduler::ProcessLifecycle::Faulted
        );
        if terminated {
            nixe_scheduler::transition_process(
                &mut self.lifecycle,
                nixe_scheduler::ProcessLifecycle::Terminating,
            )
            .expect("a live process can terminate");
            nixe_scheduler::transition_process(
                &mut self.lifecycle,
                nixe_scheduler::ProcessLifecycle::Exited,
            )
            .expect("a terminating process can exit");
            self.process_exit = Some(exit);
            self.main_thread_mut().exit = Some(ThreadExit {
                requested_scope: ExceptionTerminationScope::Process,
                exit_code: 0,
                source: None,
            });
        }
        terminated
    }

    #[cfg(test)]
    pub(crate) fn run(
        &mut self,
        instruction_budget: u64,
    ) -> Result<ExecutionReport, ProcessExecutionError> {
        let vcpu = nixe_scheduler::VirtualCpuId::new(0);
        let mut cpu_thread = self
            .execution
            .create_worker_cpu_thread(vcpu)
            .map_err(|fault| ProcessExecutionError::Cpu { fault })?;
        self.run_with_cpu_thread(&mut cpu_thread, instruction_budget)
    }

    #[cfg(test)]
    pub(crate) fn run_with_cpu_thread(
        &mut self,
        thread: &mut execution::CpuThread,
        instruction_budget: u64,
    ) -> Result<ExecutionReport, ProcessExecutionError> {
        let vcpu = nixe_scheduler::VirtualCpuId::new(0);
        let events = nixe_cpu::execution::VcpuEventState::default();
        let mut execution =
            self.begin_thread_execution(self.main_thread_id, vcpu, instruction_budget, events)?;
        let result = execution.run(thread);
        self.finish_thread_execution(self.main_thread_id, vcpu, execution, result)
    }

    pub(crate) fn begin_thread_execution(
        &mut self,
        thread_id: nixe_scheduler::GuestThreadId,
        _vcpu: nixe_scheduler::VirtualCpuId,
        instruction_budget: u64,
        events: nixe_cpu::execution::VcpuEventState,
    ) -> Result<execution::VcpuExecutionState, ProcessExecutionError> {
        let Some(selected) = self.threads.get(thread_id) else {
            return Err(ProcessExecutionError::UnknownThread(thread_id));
        };
        let loader_return = selected.loader_return;
        let (virtual_clock, architectural_timer_frequency, cpu, address_space_end) =
            self.execution.execution_environment();
        let thread = self
            .threads
            .get_mut(thread_id)
            .expect("the selected thread was validated");
        let state = thread
            .take_state()
            .expect("a ready scheduler thread owns resident CPU state");
        Ok(execution::VcpuExecutionState {
            thread: state,
            cpu,
            memory: std::sync::Arc::clone(&self.memory),
            virtual_clock,
            architectural_timer_frequency,
            address_space_end,
            instruction_budget,
            loader_return,
            events,
        })
    }

    pub(crate) fn finish_thread_execution(
        &mut self,
        thread_id: nixe_scheduler::GuestThreadId,
        _vcpu: nixe_scheduler::VirtualCpuId,
        execution: execution::VcpuExecutionState,
        result: Result<ExecutionReport, ProcessExecutionError>,
    ) -> Result<ExecutionReport, ProcessExecutionError> {
        let execution::VcpuExecutionState { thread: state, .. } = execution;
        let concurrent_stop =
            (self.lifecycle != nixe_scheduler::ProcessLifecycle::Running).then_some(self.lifecycle);
        let thread = self
            .threads
            .get_mut(thread_id)
            .expect("the executed thread remains registered");
        thread.restore_state(state);
        if let Some(lifecycle) = concurrent_stop {
            return Err(ProcessExecutionError::ConcurrentProcessStop {
                lifecycle,
                context: Box::new(thread.state().register_context()),
            });
        }
        let report = match result {
            Ok(report) => report,
            Err(error) => {
                if self.lifecycle == nixe_scheduler::ProcessLifecycle::Running {
                    nixe_scheduler::transition_process(
                        &mut self.lifecycle,
                        nixe_scheduler::ProcessLifecycle::Faulted,
                    )
                    .expect("a running process may fault");
                }
                return Err(error);
            }
        };
        if let ExecutionStop::LoaderReturn {
            source,
            result_code,
        } = &report.stop
        {
            let object_thread_id = thread.object.thread_id();
            let exit = ProcessExit {
                cause: ProcessExitCause::LoaderReturned,
                exit_code: *result_code,
                source: Some(*source),
                thread_id: object_thread_id,
                context: Some(Box::new(thread.state().register_context())),
                frames: Box::new([]),
            };
            nixe_scheduler::transition_process(
                &mut self.lifecycle,
                nixe_scheduler::ProcessLifecycle::Terminating,
            )
            .expect("a running process may terminate");
            nixe_scheduler::transition_process(
                &mut self.lifecycle,
                nixe_scheduler::ProcessLifecycle::Exited,
            )
            .expect("a terminating process may exit");
            self.process_exit = Some(exit);
            let thread = self
                .threads
                .get_mut(thread_id)
                .expect("the executed thread remains registered");
            thread.exit = Some(ThreadExit {
                requested_scope: ExceptionTerminationScope::Process,
                exit_code: *result_code,
                source: Some(*source),
            });
        }
        Ok(report)
    }

    pub(crate) fn abort_thread_execution(
        &mut self,
        thread_id: nixe_scheduler::GuestThreadId,
        _vcpu: nixe_scheduler::VirtualCpuId,
        execution: execution::VcpuExecutionState,
    ) {
        let execution::VcpuExecutionState { thread: state, .. } = execution;
        let thread = self
            .threads
            .get_mut(thread_id)
            .expect("the failed worker's thread remains registered");
        thread.restore_state(state);
        if self.lifecycle == nixe_scheduler::ProcessLifecycle::Running {
            nixe_scheduler::transition_process(
                &mut self.lifecycle,
                nixe_scheduler::ProcessLifecycle::Faulted,
            )
            .expect("a running process may fault");
        }
    }

    pub(crate) fn lose_thread_execution(&mut self) {
        if self.lifecycle == nixe_scheduler::ProcessLifecycle::Running {
            nixe_scheduler::transition_process(
                &mut self.lifecycle,
                nixe_scheduler::ProcessLifecycle::Faulted,
            )
            .expect("a running process may fault when its worker is lost");
        }
    }
}
