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
        let (id, next_thread_id) = self
            .thread_ids
            .candidate()
            .ok_or(ThreadCreateError::IdentityExhausted)?;
        let creation = self
            .processes
            .get_mut(&process_id)
            .expect("the process was validated")
            .commit_created_thread(id, &request)?;
        let registration =
            self.scheduler
                .apply(SchedulerCommand::Register(ScheduledThreadConfig {
                    process: process_id,
                    thread: id,
                    base_priority: request.priority,
                    effective_priority: request.priority,
                    ideal_vcpu: request.ideal_vcpu,
                    affinity: request.affinity,
                }));
        if registration.is_err() {
            self.processes
                .get_mut(&process_id)
                .expect("the process was validated")
                .rollback_created_thread(id);
            return Err(ThreadCreateError::Internal);
        }
        self.thread_ids.commit(next_thread_id);
        Ok(creation)
    }

    /// Starts exactly one process-local thread object. Created threads are
    /// published to the ready queue in one coordinator transaction.
    pub fn start_thread(&mut self, object_id: u64) -> Result<GuestThreadId, ThreadOperationError> {
        let thread = GuestThreadId::new(object_id);
        if self
            .scheduler
            .thread(thread)
            .is_none_or(|thread| thread.lifecycle != nixe_scheduler::ThreadLifecycle::Created)
        {
            return Err(ThreadOperationError::InvalidState);
        }
        self.scheduler
            .apply(SchedulerCommand::MakeReady(thread))
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
        let process_id = self
            .scheduler
            .thread(id)
            .ok_or(ThreadOperationError::InvalidHandle)?
            .process;
        let process = self
            .processes
            .get(&process_id)
            .ok_or(ThreadOperationError::Internal)?;
        process
            .thread(id)
            .map(|thread| thread.state().clone())
            .ok_or(ThreadOperationError::Internal)
    }

    pub fn set_thread_priority(
        &mut self,
        object_id: u64,
        priority: i32,
    ) -> Result<(), ThreadOperationError> {
        let info = self.thread_scheduling_info(object_id)?;
        if !self.scheduler.profile().priorities().contains(priority) {
            return Err(ThreadOperationError::InvalidState);
        }
        self.processes
            .get(
                &self
                    .scheduler
                    .thread(info.thread)
                    .ok_or(ThreadOperationError::InvalidHandle)?
                    .process,
            )
            .ok_or(ThreadOperationError::Internal)?
            .validate_thread_policy(priority, &info.affinity)
            .map_err(|_| ThreadOperationError::InvalidState)?;
        self.scheduler
            .apply(SchedulerCommand::SetPriority {
                thread: info.thread,
                priority,
            })
            .map_err(|_| ThreadOperationError::Internal)?;
        self.recompute_effective_priorities()
            .map_err(|_| ThreadOperationError::Internal)
    }

    pub fn set_thread_affinity(
        &mut self,
        object_id: u64,
        ideal_vcpu: Option<VirtualCpuId>,
        affinity: CoreSet,
    ) -> Result<(), ThreadOperationError> {
        let info = self.thread_scheduling_info(object_id)?;
        let owner_process = self
            .scheduler
            .thread(info.thread)
            .ok_or(ThreadOperationError::InvalidHandle)?
            .process;
        self.processes
            .get(&owner_process)
            .ok_or(ThreadOperationError::Internal)?
            .validate_thread_policy(info.base_priority, &affinity)
            .map_err(|_| ThreadOperationError::InvalidState)?;
        self.migrate_thread(info.thread, ideal_vcpu, affinity)
            .map_err(|_| ThreadOperationError::InvalidState)
    }

    pub fn set_thread_activity(
        &mut self,
        object_id: u64,
        paused: bool,
    ) -> Result<(), ThreadOperationError> {
        let info = self.thread_scheduling_info(object_id)?;
        self.scheduler
            .apply(SchedulerCommand::SetActivity {
                thread: info.thread,
                paused,
            })
            .map(|_| ())
            .map_err(|_| ThreadOperationError::InvalidState)
    }

    pub fn inherit_thread_priority(
        &mut self,
        owner_object_id: u64,
        waiter_object_id: u64,
        donation_key: u64,
    ) -> Result<(), ThreadOperationError> {
        let owner = self.thread_scheduling_info(owner_object_id)?;
        let waiter = self.thread_scheduling_info(waiter_object_id)?;
        self.priority_donations.insert(PriorityDonation {
            owner: owner.thread,
            waiter: waiter.thread,
            key: donation_key,
        });
        self.recompute_effective_priorities()
            .map_err(|_| ThreadOperationError::InvalidState)
    }

    pub fn restore_thread_priority(
        &mut self,
        object_id: u64,
        donation_key: u64,
    ) -> Result<(), ThreadOperationError> {
        let info = self.thread_scheduling_info(object_id)?;
        self.priority_donations
            .retain(|donation| donation.owner != info.thread || donation.key != donation_key);
        self.recompute_effective_priorities()
            .map_err(|_| ThreadOperationError::InvalidState)
    }

    pub fn reap_thread(&mut self, object_id: u64) -> Result<(), ThreadOperationError> {
        let thread = GuestThreadId::new(object_id);
        let view = self
            .scheduler
            .thread(thread)
            .ok_or(ThreadOperationError::InvalidHandle)?;
        if !matches!(
            view.lifecycle,
            nixe_scheduler::ThreadLifecycle::Exited | nixe_scheduler::ThreadLifecycle::Faulted
        ) {
            return Err(ThreadOperationError::InvalidState);
        }
        let process_id = view.process;
        self.scheduler
            .apply(SchedulerCommand::Unregister(thread))
            .map_err(|_| ThreadOperationError::Internal)?;
        self.processes
            .get_mut(&process_id)
            .ok_or(ThreadOperationError::Internal)?
            .reap_exited_thread(thread)
            .map_err(|_| ThreadOperationError::Internal)
    }

    pub(super) fn recompute_effective_priorities(&mut self) -> Result<(), SchedulerError> {
        let mut priorities = BTreeMap::new();
        for process in self.processes.values() {
            for (thread, _) in process.threads().iter() {
                let view = self
                    .scheduler
                    .thread(*thread)
                    .ok_or(SchedulerError::UnknownThread(*thread))?;
                priorities.insert(*thread, view.base_priority);
            }
        }
        for _ in 0..priorities.len() {
            let mut changed = false;
            for donation in &self.priority_donations {
                let Some(waiter) = priorities.get(&donation.waiter).copied() else {
                    continue;
                };
                let Some(owner) = priorities.get_mut(&donation.owner) else {
                    continue;
                };
                if waiter < *owner {
                    *owner = waiter;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        for (thread, priority) in priorities {
            let view = self
                .scheduler
                .thread(thread)
                .ok_or(SchedulerError::UnknownThread(thread))?;
            if view.effective_priority != priority {
                self.scheduler
                    .apply(SchedulerCommand::SetEffectivePriority { thread, priority })?;
            }
        }
        Ok(())
    }

    pub(super) fn terminate_remaining_process_threads(
        &mut self,
        process_id: ProcessId,
        current: GuestThreadId,
        exit_code: u64,
    ) -> Result<(), CoordinatorRouteError> {
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
            self.release_wait_resources(thread);
            self.processes
                .get_mut(&process_id)
                .expect("the process was validated")
                .address_waits_mut()
                .release_thread(thread);
            self.priority_donations
                .retain(|donation| donation.owner != thread && donation.waiter != thread);
            self.scheduler
                .apply(SchedulerCommand::Terminate {
                    thread,
                    faulted: false,
                })
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
        self.recompute_effective_priorities()
            .map_err(CoordinatorRouteError::Scheduler)?;
        Ok(())
    }

    /// Applies a topology migration and consumes its executor-local effect at
    /// the same coordinator boundary.
    pub fn migrate_thread(
        &mut self,
        thread: GuestThreadId,
        ideal_vcpu: Option<VirtualCpuId>,
        affinity: CoreSet,
    ) -> Result<(), CoordinatorError> {
        let process = self
            .scheduler
            .thread(thread)
            .ok_or(SchedulerError::UnknownThread(thread))?
            .process;
        let decision = self.scheduler.apply(SchedulerCommand::Migrate {
            thread,
            ideal_vcpu,
            affinity,
        })?;
        if let SchedulerDecision::Migrated {
            effect: nixe_scheduler::MigrationEffect::ClearOldLocalExclusive { old_vcpu },
            ..
        } = decision
        {
            let domain = self
                .processes
                .get(&process)
                .expect("scheduled thread has an owning process")
                .engine_domain_id();
            self.workers
                .clear_local_exclusive(
                    old_vcpu,
                    super::worker::WorkerExecutorKey { process, domain },
                )
                .map_err(CoordinatorError::Worker)?;
        }
        Ok(())
    }
}
