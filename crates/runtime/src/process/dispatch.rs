use super::*;

impl RunnableProcess {
    /// Returns the process exit record retained until teardown.
    #[must_use]
    pub const fn exit(&self) -> Option<ProcessExit> {
        self.process_exit
    }

    /// Returns the selected execution-engine descriptor.
    #[must_use]
    pub fn engine_descriptor(&self) -> nixe_cpu_engine::EngineDescriptor {
        self.execution.engine_descriptor()
    }

    pub(crate) fn engine_requirements(
        &self,
        parallel: bool,
        vcpu_count: usize,
    ) -> nixe_cpu_engine::EngineCapabilities {
        let mut required = nixe_cpu_engine::EngineCapabilities {
            deterministic_execution: !parallel,
            concurrent_executors: parallel,
            max_safepoint_instructions: parallel
                .then(|| std::num::NonZeroU64::new(u64::MAX).unwrap()),
            acknowledged_invalidation: parallel,
            max_concurrent_executors: parallel.then(|| {
                std::num::NonZeroUsize::new(vcpu_count)
                    .expect("a runtime scheduler profile has at least one vCPU")
            }),
            ..Default::default()
        };
        for (_, thread) in self.threads.iter() {
            match thread.state().execution_state() {
                nixe_cpu::location::ExecutionState::A64 => required.a64 = true,
                nixe_cpu::location::ExecutionState::A32 => required.a32 = true,
                nixe_cpu::location::ExecutionState::T32 => required.t32 = true,
            }
        }
        required
    }

    pub(crate) fn engine_domain_id(&self) -> nixe_cpu_engine::EngineDomainId {
        self.execution.domain_id()
    }

    pub(crate) fn fallback_engine_domain_id(&self) -> Option<nixe_cpu_engine::EngineDomainId> {
        self.execution.fallback_domain_id()
    }

    pub(crate) fn create_worker_executors(
        &mut self,
        vcpu: nixe_scheduler::VirtualCpuId,
    ) -> Result<execution::WorkerExecutors, nixe_cpu_engine::EngineFault> {
        self.execution.create_worker_executors(vcpu)
    }

    #[cfg(test)]
    pub(crate) fn request_safepoint(&mut self) {
        self.execution.request_safepoint();
    }

    #[cfg(test)]
    pub(crate) fn post_event(&self, mask: u32) {
        self.execution.post_event(mask);
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
        let mut executor = self
            .execution
            .create_worker_executors(vcpu)
            .map_err(|fault| ProcessExecutionError::Engine { fault })?
            .primary;
        self.run_with_executor(executor.as_mut(), instruction_budget)
    }

    #[cfg(test)]
    pub(crate) fn run_with_executor(
        &mut self,
        executor: &mut dyn nixe_cpu_engine::EngineExecutor,
        instruction_budget: u64,
    ) -> Result<ExecutionReport, ProcessExecutionError> {
        let vcpu = nixe_scheduler::VirtualCpuId::new(0);
        let mut execution =
            self.begin_thread_execution(self.main_thread_id, vcpu, instruction_budget)?;
        let result = execution.run(executor);
        self.finish_thread_execution(self.main_thread_id, vcpu, execution, result)
    }

    pub(crate) fn begin_thread_execution(
        &mut self,
        thread_id: nixe_scheduler::GuestThreadId,
        _vcpu: nixe_scheduler::VirtualCpuId,
        instruction_budget: u64,
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
