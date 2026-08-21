use super::*;

impl RunnableProcess {
    /// Returns the host-side lifecycle state of this process.
    #[must_use]
    pub fn execution_status(&self) -> ProcessExecutionStatus {
        match self.lifecycle {
            nixe_scheduler::ProcessLifecycle::Exited => ProcessExecutionStatus::Exited,
            nixe_scheduler::ProcessLifecycle::Faulted => ProcessExecutionStatus::Faulted,
            _ => match self.main_thread().lifecycle {
                nixe_scheduler::ThreadLifecycle::Running => ProcessExecutionStatus::Running,
                nixe_scheduler::ThreadLifecycle::Waiting => ProcessExecutionStatus::Suspended,
                nixe_scheduler::ThreadLifecycle::Suspended => ProcessExecutionStatus::Suspended,
                nixe_scheduler::ThreadLifecycle::Exited => ProcessExecutionStatus::Exited,
                nixe_scheduler::ThreadLifecycle::Faulted => ProcessExecutionStatus::Faulted,
                _ => ProcessExecutionStatus::Ready,
            },
        }
    }

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
            precise_instruction_budget: true,
            instruction_trace: self.execution.instruction_trace_enabled(),
            canonical_state_version: 1,
            deterministic_execution: !parallel,
            precise_exceptions: true,
            engine_handoff: true,
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

    pub(crate) fn take_worker_executor(
        &mut self,
        vcpu: nixe_scheduler::VirtualCpuId,
    ) -> Result<Box<dyn nixe_cpu_engine::EngineExecutor>, nixe_cpu_engine::EngineFault> {
        self.execution.lease_executor(vcpu)
    }

    pub(crate) fn restore_worker_executor(
        &mut self,
        vcpu: nixe_scheduler::VirtualCpuId,
        executor: Box<dyn nixe_cpu_engine::EngineExecutor>,
    ) {
        self.execution.restore_executor(vcpu, executor);
    }

    pub(crate) fn take_worker_fallback_executor(
        &mut self,
        vcpu: nixe_scheduler::VirtualCpuId,
    ) -> Result<Option<Box<dyn nixe_cpu_engine::EngineExecutor>>, nixe_cpu_engine::EngineFault>
    {
        self.execution.lease_fallback_executor(vcpu)
    }

    pub(crate) fn restore_worker_fallback_executor(
        &mut self,
        vcpu: nixe_scheduler::VirtualCpuId,
        executor: Box<dyn nixe_cpu_engine::EngineExecutor>,
    ) {
        self.execution.restore_fallback_executor(vcpu, executor);
    }

    pub(crate) fn prepare_engine_switch(
        &mut self,
        vcpus: impl IntoIterator<Item = nixe_scheduler::VirtualCpuId>,
        provider: &dyn nixe_cpu_engine::EngineProvider,
    ) -> Result<execution::PreparedEngineSwitch, nixe_cpu_engine::HandoffFailure> {
        let memory = nixe_cpu_engine::DomainMemoryBinding {
            address_space: self.cpu.address_space_id(),
            end_exclusive: nixe_memory::GuestVirtualAddress::new(
                self.address_space.exclusive_limit(),
            ),
            memory: self.memory.as_ref(),
            invalidation_generation: self.memory.mapping_epoch().get(),
            dirty_generation: self.memory.content_mutation_epoch().get(),
        };
        self.execution
            .prepare_provider_switch(self.cpu, memory, vcpus, provider)
    }

    pub(crate) fn complete_engine_switch(
        &mut self,
        prepared: &mut execution::PreparedEngineSwitch,
    ) -> Result<(), nixe_cpu_engine::HandoffFailure> {
        let memory = nixe_cpu_engine::DomainMemoryBinding {
            address_space: self.cpu.address_space_id(),
            end_exclusive: nixe_memory::GuestVirtualAddress::new(
                self.address_space.exclusive_limit(),
            ),
            memory: self.memory.as_ref(),
            invalidation_generation: self.memory.mapping_epoch().get(),
            dirty_generation: self.memory.content_mutation_epoch().get(),
        };
        self.execution.complete_provider_switch(prepared, memory)
    }

    pub(crate) fn reactivate_engine_after_switch_failure(
        &mut self,
    ) -> Result<(), nixe_cpu_engine::EngineFault> {
        self.execution.reactivate_after_switch_failure()
    }

    pub(crate) fn commit_engine_switch(
        &mut self,
        prepared: execution::PreparedEngineSwitch,
    ) -> (
        nixe_cpu_engine::StateCommitBarrier,
        Box<dyn nixe_cpu_engine::EngineDomain>,
    ) {
        self.execution.commit_provider_switch(prepared)
    }

    #[cfg(test)]
    pub(crate) fn request_safepoint(&mut self) {
        self.execution.request_safepoint();
    }

    #[cfg(test)]
    pub(crate) fn post_event(&self, mask: u32) {
        self.execution.post_event(mask);
    }

    pub(crate) fn resume_thread(&mut self, id: nixe_scheduler::GuestThreadId) -> bool {
        if self.lifecycle != nixe_scheduler::ProcessLifecycle::Running {
            return false;
        }
        let Some(thread) = self.threads.get_mut(id) else {
            return false;
        };
        if thread.lifecycle != nixe_scheduler::ThreadLifecycle::Waiting {
            return false;
        }
        nixe_scheduler::transition_thread(
            &mut thread.lifecycle,
            nixe_scheduler::ThreadLifecycle::Ready,
        )
        .expect("runtime thread and scheduler lifecycles remain synchronized");
        thread.wait_reason = None;
        true
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
            let thread_lifecycle = self.main_thread().lifecycle;
            if !matches!(
                thread_lifecycle,
                nixe_scheduler::ThreadLifecycle::Exited | nixe_scheduler::ThreadLifecycle::Faulted
            ) {
                nixe_scheduler::transition_thread(
                    &mut self.main_thread_mut().lifecycle,
                    nixe_scheduler::ThreadLifecycle::Terminating,
                )
                .expect("a live main thread can terminate");
                nixe_scheduler::transition_thread(
                    &mut self.main_thread_mut().lifecycle,
                    nixe_scheduler::ThreadLifecycle::Exited,
                )
                .expect("a terminating main thread can exit");
            }
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
        self.run_thread(
            self.main_thread_id,
            nixe_scheduler::VirtualCpuId::new(0),
            instruction_budget,
        )
    }

    #[cfg(test)]
    pub(crate) fn run_thread(
        &mut self,
        thread_id: nixe_scheduler::GuestThreadId,
        vcpu: nixe_scheduler::VirtualCpuId,
        instruction_budget: u64,
    ) -> Result<ExecutionReport, ProcessExecutionError> {
        let mut executor = self
            .execution
            .lease_executor(vcpu)
            .map_err(|fault| ProcessExecutionError::Engine { fault })?;
        let mut execution = match self.begin_thread_execution(thread_id, vcpu, instruction_budget) {
            Ok(execution) => execution,
            Err(error) => {
                self.execution.restore_executor(vcpu, executor);
                return Err(error);
            }
        };
        let result = execution.run(executor.as_mut());
        self.execution.restore_executor(vcpu, executor);
        self.finish_thread_execution(thread_id, vcpu, execution, result)
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
        if selected.lifecycle != nixe_scheduler::ThreadLifecycle::Ready {
            return Err(ProcessExecutionError::NotRunnable {
                status: self.execution_status(),
                context: Box::new(selected.state().register_context()),
            });
        }
        let loader_return = selected.loader_return;
        let (virtual_clock, architectural_timer_frequency, cpu, address_space_end) =
            self.execution.execution_environment();
        let thread = self
            .threads
            .get_mut(thread_id)
            .expect("the selected thread was validated");
        nixe_scheduler::transition_thread(
            &mut thread.lifecycle,
            nixe_scheduler::ThreadLifecycle::Running,
        )
        .expect("a ready thread can run");
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
        let concurrent_stop = (self.lifecycle != nixe_scheduler::ProcessLifecycle::Running)
            .then(|| self.execution_status());
        let thread = self
            .threads
            .get_mut(thread_id)
            .expect("the executed thread remains registered");
        thread.restore_state(state);
        if let Some(status) = concurrent_stop {
            if thread.lifecycle == nixe_scheduler::ThreadLifecycle::Running {
                nixe_scheduler::transition_thread(
                    &mut thread.lifecycle,
                    nixe_scheduler::ThreadLifecycle::Faulted,
                )
                .expect("an in-flight thread may stop after its process");
            }
            return Err(ProcessExecutionError::ConcurrentProcessStop {
                status,
                context: Box::new(thread.state().register_context()),
            });
        }
        let report = match result {
            Ok(report) => report,
            Err(error) => {
                nixe_scheduler::transition_thread(
                    &mut thread.lifecycle,
                    nixe_scheduler::ThreadLifecycle::Faulted,
                )
                .expect("a running thread may fault");
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
        let target = match report.stop {
            ExecutionStop::BudgetExhausted
            | ExecutionStop::Safepoint
            | ExecutionStop::PendingEvent { .. } => nixe_scheduler::ThreadLifecycle::Ready,
            ExecutionStop::FetchFault { .. } | ExecutionStop::UnsupportedSemantics { .. } => {
                nixe_scheduler::ThreadLifecycle::Faulted
            }
            ExecutionStop::LoaderReturn { .. } => nixe_scheduler::ThreadLifecycle::Exited,
            _ => nixe_scheduler::ThreadLifecycle::Waiting,
        };
        if target == nixe_scheduler::ThreadLifecycle::Exited {
            nixe_scheduler::transition_thread(
                &mut thread.lifecycle,
                nixe_scheduler::ThreadLifecycle::Terminating,
            )
            .expect("a running thread may terminate");
        }
        nixe_scheduler::transition_thread(&mut thread.lifecycle, target)
            .expect("engine exits define legal running-thread transitions");
        thread.wait_reason = (thread.lifecycle == nixe_scheduler::ThreadLifecycle::Waiting)
            .then_some(nixe_scheduler::WaitReason::Scheduler);
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
        if thread.lifecycle == nixe_scheduler::ThreadLifecycle::Running {
            nixe_scheduler::transition_thread(
                &mut thread.lifecycle,
                nixe_scheduler::ThreadLifecycle::Faulted,
            )
            .expect("a running thread may fault");
        }
        if self.lifecycle == nixe_scheduler::ProcessLifecycle::Running {
            nixe_scheduler::transition_process(
                &mut self.lifecycle,
                nixe_scheduler::ProcessLifecycle::Faulted,
            )
            .expect("a running process may fault");
        }
    }

    pub(crate) fn lose_thread_execution(&mut self, thread_id: nixe_scheduler::GuestThreadId) {
        let thread = self
            .threads
            .get_mut(thread_id)
            .expect("the lost worker's thread remains registered");
        if thread.lifecycle == nixe_scheduler::ThreadLifecycle::Running {
            nixe_scheduler::transition_thread(
                &mut thread.lifecycle,
                nixe_scheduler::ThreadLifecycle::Faulted,
            )
            .expect("a running thread may fault when its worker is lost");
        }
        if self.lifecycle == nixe_scheduler::ProcessLifecycle::Running {
            nixe_scheduler::transition_process(
                &mut self.lifecycle,
                nixe_scheduler::ProcessLifecycle::Faulted,
            )
            .expect("a running process may fault when its worker is lost");
        }
    }
}
