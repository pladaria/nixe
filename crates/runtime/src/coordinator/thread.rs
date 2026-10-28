use super::*;

impl RuntimeCoordinator {
    /// Failure-atomically constructs and registers a created guest thread.
    pub fn create_thread(
        &mut self,
        process_id: ProcessId,
        request: ThreadCreateRequest,
    ) -> Result<ThreadCreation, ThreadCreateError> {
        if !self
            .scheduler
            .profile()
            .priorities()
            .contains(request.priority)
        {
            return Err(ThreadCreateError::InvalidPriority(request.priority));
        }
        if let Some(vcpu) = request
            .affinity
            .iter()
            .find(|vcpu| !self.scheduler.profile().contains(*vcpu))
        {
            return Err(ThreadCreateError::InvalidVirtualCpu(vcpu));
        }
        if let Some(ideal) = request.ideal_vcpu
            && !request.affinity.contains(ideal)
        {
            return Err(ThreadCreateError::InvalidVirtualCpu(ideal));
        }
        self.processes
            .get(&process_id)
            .ok_or(ThreadCreateError::Internal)?
            .validate_thread_request(&request)?;
        let id = self
            .scheduler
            .create_thread(ScheduledThreadConfig {
                process: process_id,
                base_priority: request.priority,
                ideal_vcpu: request.ideal_vcpu,
                affinity: request.affinity,
            })
            .map_err(|error| match error {
                SchedulerError::ThreadIdExhausted => ThreadCreateError::IdentityExhausted,
                _ => ThreadCreateError::Internal,
            })?;
        match self
            .processes
            .get_mut(&process_id)
            .expect("validated process")
            .commit_created_thread(id, &request, self.inbox.sender())
        {
            Ok(creation) => Ok(creation),
            Err(error) => {
                self.scheduler
                    .unregister(id)
                    .expect("created thread has no register owner");
                Err(error)
            }
        }
    }

    /// Starts exactly one process-local thread object. Created threads are
    /// published to the ready queue in one coordinator transaction.
    pub fn start_thread(&mut self, object_id: u64) -> Result<GuestThreadId, ThreadOperationError> {
        let thread = GuestThreadId::new(object_id);
        let info = self
            .scheduler
            .thread(thread)
            .ok_or(ThreadOperationError::InvalidState)?;
        if info.lifecycle != nixe_scheduler::ThreadLifecycle::Created {
            return Err(ThreadOperationError::InvalidState);
        }
        let runtime = self
            .processes
            .get_mut(&info.process)
            .ok_or(ThreadOperationError::Internal)?;
        runtime.start_thread(thread)?;
        self.scheduler
            .make_ready(thread)
            .map_err(|_| ThreadOperationError::Internal)?;
        Ok(thread)
    }

    pub fn thread_scheduling_info(
        &self,
        object_id: u64,
    ) -> Result<nixe_scheduler::ScheduledThreadView, ThreadOperationError> {
        let id = GuestThreadId::new(object_id);
        self.scheduler
            .thread(id)
            .ok_or(ThreadOperationError::InvalidHandle)
    }

    pub fn thread_cpu_state(
        &self,
        object_id: u64,
    ) -> Result<nixe_cpu::state::ThreadCpuState, ThreadOperationError> {
        let id = GuestThreadId::new(object_id);
        let view = self
            .scheduler
            .thread(id)
            .ok_or(ThreadOperationError::InvalidHandle)?;
        if view.lifecycle == nixe_scheduler::ThreadLifecycle::Running {
            return Err(ThreadOperationError::InvalidState);
        }
        let process_id = view.process;
        let process = self
            .processes
            .get(&process_id)
            .ok_or(ThreadOperationError::Internal)?;
        let thread = process.thread(id).ok_or(ThreadOperationError::Internal)?;
        thread
            .state
            .as_ref()
            .filter(|state| state.is_available())
            .map(|state| state.snapshot())
            .ok_or(ThreadOperationError::InvalidState)
    }

    pub fn set_thread_priority(
        &mut self,
        caller: Option<GuestThreadId>,
        object_id: u64,
        priority: i32,
    ) -> Result<(), ThreadOperationError> {
        self.scheduler
            .set_priority(GuestThreadId::new(object_id), priority, caller)
            .map_err(|error| match error {
                SchedulerError::UnknownThread(_) => ThreadOperationError::InvalidHandle,
                SchedulerError::InvalidPriority(_) => ThreadOperationError::InvalidState,
                _ => ThreadOperationError::Internal,
            })?;
        Ok(())
    }

    pub(super) fn terminate_remaining_process_threads(
        &mut self,
        process_id: ProcessId,
        current: GuestThreadId,
        exit_code: u64,
    ) -> Result<(), CoordinatorRouteError> {
        if self
            .scheduler
            .active_leases()
            .any(|lease| lease.process == process_id)
        {
            self.quiesce()
                .map_err(|error| CoordinatorRouteError::Coordinator(Box::new(error)))?;
        }
        self.completed_executions
            .retain(|execution| execution.lease.process != process_id);
        let targets: Vec<_> = self
            .processes
            .get(&process_id)
            .ok_or(CoordinatorRouteError::UnknownProcess(process_id))?
            .threads()
            .iter()
            .filter_map(|(id, _)| {
                (*id != current
                    && self.scheduler.thread(*id).is_some_and(|thread| {
                        !matches!(
                            thread.lifecycle,
                            nixe_scheduler::ThreadLifecycle::Exited
                                | nixe_scheduler::ThreadLifecycle::Faulted
                        )
                    }))
                .then_some(*id)
            })
            .collect();
        for thread in targets {
            self.finalize_thread_waits(process_id, thread);
            self.scheduler
                .terminate(thread, false)
                .map_err(CoordinatorRouteError::Scheduler)?;
            self.processes
                .get_mut(&process_id)
                .expect("the process was validated")
                .terminate_thread_from_coordinator(
                    thread,
                    exit_code,
                    crate::ExceptionTerminationScope::Process,
                );
        }
        Ok(())
    }
}
