use super::*;

impl RuntimeCoordinator {
    pub(super) fn finish_process_termination(
        &mut self,
        process_id: ProcessId,
    ) -> Result<(), CoordinatorError> {
        let process = self
            .processes
            .get(&process_id)
            .ok_or(CoordinatorError::UnknownProcess(process_id))?
            .clone();
        if process.lifecycle() != nixe_scheduler::ProcessLifecycle::Terminating {
            return Ok(());
        }
        if self
            .scheduler
            .active_leases()
            .any(|lease| lease.process == process_id)
        {
            process.request_safepoint();
            return Ok(());
        }
        let Some(lifecycle) = process.take_termination() else {
            return Ok(());
        };
        self.completed_executions
            .retain(|execution| execution.lease.process != process_id);
        let exit_code = process.exit().map_or(0, |exit| exit.exit_code);
        for thread in process.threads().ids() {
            if self.scheduler.thread(thread).is_some_and(|view| {
                !matches!(
                    view.lifecycle,
                    nixe_scheduler::ThreadLifecycle::Exited
                        | nixe_scheduler::ThreadLifecycle::Faulted
                )
            }) {
                let object = process.terminate_thread_from_coordinator(
                    thread,
                    exit_code,
                    crate::ExceptionTerminationScope::Process,
                );
                self.finalize_thread_waits(process_id, thread);
                self.scheduler.terminate(thread, false)?;
                if let Some(object) = object {
                    object.signal()
                }
            }
        }
        process.finish_termination(lifecycle);
        Ok(())
    }
}
