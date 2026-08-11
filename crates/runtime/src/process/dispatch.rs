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

    pub(crate) fn engine_domain_id(&self) -> nixe_cpu_engine::EngineDomainId {
        self.execution.domain_id()
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

    pub(crate) fn set_worker_executors_resident(&mut self, resident: bool) {
        self.execution.set_worker_resident(resident);
    }

    pub(crate) fn prepare_engine_switch(
        &mut self,
        vcpus: impl IntoIterator<Item = nixe_scheduler::VirtualCpuId>,
        provider: &dyn nixe_cpu_engine::EngineProvider,
    ) -> Result<execution::PreparedEngineSwitch, nixe_cpu_engine::HandoffFailure> {
        let memory = nixe_cpu_engine::MemorySynchronizationRecord {
            address_space: self.cpu.address_space_id(),
            invalidation_generation: self.memory.mapping_epoch().get(),
            dirty_generation: self.memory.content_generation_watermark(),
        };
        self.execution
            .prepare_provider_switch(self.cpu, memory, vcpus, provider)
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

    /// Requests a stop before the next reference-engine instruction.
    pub fn request_safepoint(&mut self) {
        self.execution.request_safepoint();
    }

    /// Publishes runtime event bits to be observed at the next safepoint.
    pub fn post_event(&self, mask: u32) {
        self.execution.post_event(mask);
    }

    /// Resumes a process suspended by an exception or scheduling instruction.
    pub fn resume(&mut self) -> bool {
        self.resume_thread(self.main_thread_id)
    }

    /// Resumes one explicit guest thread. Scheduler-facing code must use this
    /// operation rather than the main-thread compatibility adapter.
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
        .expect("runtime and compatibility execution lifecycles remain synchronized");
        thread.wait_reason = None;
        true
    }

    /// Marks the process exited. Resource release occurs in [`Self::teardown`]
    /// or when the process is dropped.
    pub fn terminate(&mut self) -> bool {
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

    /// Runs one bounded slice through the injected CPU engine domain.
    pub fn run(
        &mut self,
        instruction_budget: u64,
    ) -> Result<ExecutionReport, ProcessExecutionError> {
        self.run_thread(
            self.main_thread_id,
            nixe_scheduler::VirtualCpuId::new(0),
            instruction_budget,
        )
    }

    /// Runs one bounded slice for the thread and emulated vCPU selected by the
    /// scheduler. This is the production execution entry point.
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
        let (virtual_clock, architectural_timer_frequency, cpu) =
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

    /// Deprecated compatibility name for [`Self::run`]. New orchestration must
    /// use the engine-neutral method; this wrapper is removed after migration.
    pub fn run_reference(
        &mut self,
        instruction_budget: u64,
    ) -> Result<ExecutionReport, ProcessExecutionError> {
        self.run(instruction_budget)
    }

    /// Failure-atomically replaces the process execution domain at a canonical
    /// state boundary. The old domain remains installed if preparation fails.
    pub fn switch_engine(
        &mut self,
        provider: &dyn nixe_cpu_engine::EngineProvider,
    ) -> Result<nixe_cpu_engine::StateCommitBarrier, nixe_cpu_engine::HandoffFailure> {
        let memory = nixe_cpu_engine::MemorySynchronizationRecord {
            address_space: self.cpu.address_space_id(),
            invalidation_generation: self.memory.mapping_epoch().get(),
            dirty_generation: self.memory.content_generation_watermark(),
        };
        self.execution.switch_provider(self.cpu, memory, provider)
    }
}
