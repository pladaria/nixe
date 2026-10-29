use super::*;

impl RunnableProcess {
    pub fn exit(&self) -> Option<ProcessExit> {
        self.state.lock().unwrap().exit.clone()
    }
    pub(crate) fn cpu_process_id(&self) -> nixe_cpu::execution::CpuProcessId {
        self.execution.lock().unwrap().process_id()
    }
    pub(crate) fn create_worker_cpu_thread(
        &self,
        vcpu: nixe_scheduler::VirtualCpuId,
    ) -> Result<execution::CpuThread, nixe_cpu::execution::CpuFault> {
        self.execution
            .lock()
            .unwrap()
            .create_worker_cpu_thread(vcpu)
    }
    pub(crate) fn request_execution_stop(&self) -> Result<(), nixe_cpu::execution::CpuFault> {
        self.execution.lock().unwrap().request_stop()
    }
    pub(crate) fn complete_cpu_thread_retirement(&self) {
        for (_, thread) in self.threads.iter() {
            thread.materialize(false);
        }
        let mut execution = self.execution.lock().unwrap();
        execution.residents.clear();
        execution.complete_cpu_thread_retirement();
    }
    pub(crate) fn request_safepoint(&self) {
        self.execution.lock().unwrap().request_safepoint();
    }
    pub(super) fn finish_process(
        &self,
        lifecycle: nixe_scheduler::ProcessLifecycle,
        exit: Option<ProcessExit>,
    ) {
        let mut state = self.state.lock().unwrap();
        if matches!(
            state.lifecycle,
            nixe_scheduler::ProcessLifecycle::Exited | nixe_scheduler::ProcessLifecycle::Faulted
        ) || (state.lifecycle == nixe_scheduler::ProcessLifecycle::Terminating
            && state.termination.is_none())
        {
            return;
        }
        if state.lifecycle != nixe_scheduler::ProcessLifecycle::Terminating {
            self.generation
                .fetch_add(1, std::sync::atomic::Ordering::Release);
        }
        state.lifecycle = nixe_scheduler::ProcessLifecycle::Terminating;
        state.termination = Some(lifecycle);
        state.exit = exit;
        drop(state);
        self.cancel_waits();
    }
    pub(crate) fn take_termination(&self) -> Option<nixe_scheduler::ProcessLifecycle> {
        self.state.lock().unwrap().termination.take()
    }
    pub(crate) fn finish_termination(&self, lifecycle: nixe_scheduler::ProcessLifecycle) {
        // Child waits/native execution have stopped. Close owned endpoints before
        // publishing termination; exported objects retain their own references.
        // https://github.com/eden-emulator/mirror/blob/master/src/core/hle/kernel/k_process.cpp
        self.handles.close_all();
        self.state.lock().unwrap().lifecycle = lifecycle;
    }
    pub(crate) fn terminate_from_host(&self) -> bool {
        let mut state = self.state.lock().unwrap();
        if state.lifecycle != nixe_scheduler::ProcessLifecycle::Running {
            return false;
        }
        state.lifecycle = nixe_scheduler::ProcessLifecycle::Terminating;
        state.termination = Some(nixe_scheduler::ProcessLifecycle::Exited);
        self.generation
            .fetch_add(1, std::sync::atomic::Ordering::Release);
        state.exit = Some(ProcessExit {
            cause: ProcessExitCause::HostRequested,
            exit_code: 0,
            source: None,
            thread_id: self.main_thread_id.get(),
            context: None,
            frames: Box::new([]),
        });
        drop(state);
        self.cancel_waits();
        true
    }
    pub(crate) fn cancel_waits(&self) {
        for (_, thread) in self.threads.iter() {
            thread.cancel_wait();
        }
    }
    pub(crate) fn begin_thread_execution(
        &self,
        id: nixe_scheduler::GuestThreadId,
        vcpu: nixe_scheduler::VirtualCpuId,
        events: nixe_cpu::execution::VcpuEventState,
    ) -> Result<execution::VcpuExecutionState, ProcessExecutionError> {
        let mut execution = self.execution.lock().unwrap();
        if execution
            .residents
            .iter()
            .any(|(core, guest)| *core != vcpu && *guest == id)
            && let Some(thread) = self.thread(id)
        {
            thread.materialize(true);
        }
        if let Some(previous) = execution.residents.insert(vcpu, id)
            && previous != id
            && let Some(thread) = self.thread(previous)
        {
            thread.materialize(true);
        }
        execution
            .residents
            .retain(|core, guest| *core == vcpu || *guest != id);
        let (virtual_clock, architectural_timer_frequency) = execution.execution_environment();
        drop(execution);
        let thread = self
            .thread(id)
            .ok_or(ProcessExecutionError::UnknownThread(id))?;
        let state = thread
            .take_state()
            .expect("ready kernel thread owns its registers");
        Ok(execution::VcpuExecutionState {
            thread: state,
            tls_base: thread.tls_base,
            virtual_clock,
            architectural_timer_frequency,
            events,
        })
    }
    pub(crate) fn finish_thread_execution(
        &self,
        id: nixe_scheduler::GuestThreadId,
        _vcpu: nixe_scheduler::VirtualCpuId,
        execution: execution::VcpuExecutionState,
        result: Result<ExecutionReport, ProcessExecutionError>,
    ) -> Result<ExecutionReport, ProcessExecutionError> {
        let execution::VcpuExecutionState { thread: state, .. } = execution;
        let lifecycle = self.lifecycle();
        let concurrent_stop =
            (lifecycle != nixe_scheduler::ProcessLifecycle::Running).then_some(lifecycle);
        let thread = self
            .thread(id)
            .expect("running reference retains the executed thread");
        let current = match &result {
            Ok(_) => true,
            Err(ProcessExecutionError::Cpu { fault }) => fault.context.is_some(),
            Err(_) => false,
        };
        if current {
            thread.restore_state(state);
        }
        let mut report = match result {
            Ok(report) => report,
            Err(error) => {
                if concurrent_stop.is_none()
                    || !matches!(&error, ProcessExecutionError::Cpu { fault } if fault.kind == nixe_cpu::execution::CpuFaultKind::Stopped)
                {
                    self.finish_process(nixe_scheduler::ProcessLifecycle::Faulted, None);
                }
                return Err(error);
            }
        };
        if let Some(lifecycle) = concurrent_stop {
            return Err(ProcessExecutionError::ConcurrentProcessStop {
                lifecycle,
                context: Box::new(thread.registers().register_context()),
            });
        }
        if let ExecutionStop::SupervisorCall {
            source,
            immediate: 7,
        } = &report.stop
            && thread.loader_return == Some(source.pc)
        {
            report.stop = ExecutionStop::LoaderReturn {
                source: *source,
                result_code: thread.registers().read_x(A64Register::General(
                    nixe_cpu::state::a64::A64GeneralRegister::new(0).unwrap(),
                )),
            };
        }
        if matches!(
            report.stop,
            ExecutionStop::FetchFault { .. } | ExecutionStop::UnsupportedSemantics { .. }
        ) {
            self.finish_process(nixe_scheduler::ProcessLifecycle::Faulted, None);
        }
        if let ExecutionStop::LoaderReturn {
            source,
            result_code,
        } = &report.stop
        {
            let context = thread.registers().register_context();
            report.context = Some(context.clone());
            self.finish_process(
                nixe_scheduler::ProcessLifecycle::Exited,
                Some(ProcessExit {
                    cause: ProcessExitCause::LoaderReturned,
                    exit_code: *result_code,
                    source: Some(*source),
                    thread_id: thread.thread_id(),
                    context: Some(Box::new(context)),
                    frames: Box::new([]),
                }),
            );
            thread.set_exit(ThreadExit {
                requested_scope: ExceptionTerminationScope::Process,
                exit_code: *result_code,
                source: Some(*source),
            });
        }
        Ok(report)
    }
    pub(crate) fn abort_thread_execution(
        &self,
        id: nixe_scheduler::GuestThreadId,
        _vcpu: nixe_scheduler::VirtualCpuId,
        execution: Option<execution::VcpuExecutionState>,
    ) {
        let thread = self.thread(id).expect("failed worker retains its thread");
        if let Some(execution::VcpuExecutionState { thread: state, .. }) = execution {
            thread.restore_state(state);
        }
        self.finish_process(nixe_scheduler::ProcessLifecycle::Faulted, None);
    }
}
