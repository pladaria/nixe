use super::*;

impl RunnableProcess {
    pub fn exit(&self) -> Option<ProcessExit> {
        self.state.lock().unwrap().exit.clone()
    }
    /// Publish the SDK preemption flag while this physical core owns the context.
    // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/k_interrupt_manager.cpp
    pub fn handle_interrupt(
        &self,
        thread: &crate::ThreadObject,
        core: nixe_scheduler::VirtualCpuId,
        cpu: &nixe_cpu_jit::JitThread,
    ) -> Result<(), crate::DispatchFault> {
        let tls = thread.tls_base;
        if cpu.read_memory(tls.checked_add(0x100).expect("bounded TLS"), 2)? != 0
            && self
                .scheduler
                .pinned_thread(nixe_scheduler::ProcessId::new(self.process_id()), core)?
                .is_none()
        {
            self.scheduler.pin_thread(core, thread.id())?;
            cpu.write_memory(tls.checked_add(0x102).expect("bounded TLS"), 2, 1)?;
        }
        Ok(())
    }

    pub fn create_core(
        &self,
        vcpu: nixe_scheduler::VirtualCpuId,
    ) -> Result<execution::CpuThread, nixe_cpu::execution::CpuFault> {
        let thread =
            self.execution
                .create_thread()
                .map_err(|error| nixe_cpu::execution::CpuFault {
                    backend: "jit",
                    kind: if matches!(error, nixe_cpu_jit::JitError::Stopped) {
                        nixe_cpu::execution::CpuFaultKind::Stopped
                    } else {
                        nixe_cpu::execution::CpuFaultKind::Internal
                    },
                    message: error.to_string().into(),
                    context: None,
                })?;
        self.native_cores
            .fetch_or(1 << vcpu.get(), std::sync::atomic::Ordering::Release);
        Ok(Box::new(thread))
    }
    pub(crate) fn request_execution_stop(&self) {
        self.execution.request_stop();
    }
    pub fn retire_core(&self, core: nixe_scheduler::VirtualCpuId) {
        self.native_cores
            .fetch_and(!(1 << core.get()), std::sync::atomic::Ordering::Release);
    }
    pub(crate) fn has_native_threads(&self) -> bool {
        self.native_cores.load(std::sync::atomic::Ordering::Acquire) != 0
    }
    pub(crate) fn execution_environment(&self) -> (crate::VirtualClock, u64) {
        (
            self.scheduler.clock().clone(),
            self.execution.timer_frequency(),
        )
    }
    pub fn finish_process(
        &self,
        lifecycle: nixe_scheduler::ProcessLifecycle,
        exit: Option<ProcessExit>,
    ) {
        let mut state = self.state.lock().unwrap();
        // Keep the first definitive exit; a last-thread reservation has no exit yet.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/k_process.cpp#L541-L581
        if matches!(
            state.lifecycle,
            nixe_scheduler::ProcessLifecycle::Exited | nixe_scheduler::ProcessLifecycle::Faulted
        ) || (state.lifecycle == nixe_scheduler::ProcessLifecycle::Terminating
            && (state.termination != Some(nixe_scheduler::ProcessLifecycle::Exited)
                || state.exit.is_some()))
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
}
