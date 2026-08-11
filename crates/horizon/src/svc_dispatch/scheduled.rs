use super::*;

impl HorizonSvcDispatcher {
    /// Routes one scheduled Horizon SVC and commits its runtime-side effect
    /// before returning control to the application composition root.
    pub fn route_scheduled_supervisor_call(
        &mut self,
        coordinator: &mut nixe_runtime::RuntimeCoordinator,
        lease: nixe_scheduler::Lease,
        stop: &nixe_runtime::ExecutionStop,
    ) -> Result<nixe_runtime::ExceptionHandlingResult<HorizonSvcFault>, HorizonScheduledDispatchError>
    {
        self.synchronize_virtual_time(coordinator.virtual_time_ns());
        let handling = coordinator
            .route_supervisor_call(lease, stop, self)
            .map_err(HorizonScheduledDispatchError::Route)?;
        if let nixe_runtime::ExceptionHandlingResult::Terminated { scope, .. } = &handling {
            match scope {
                nixe_runtime::ExceptionTerminationScope::CurrentThread => {
                    self.release_thread_synchronization(lease.thread);
                }
                nixe_runtime::ExceptionTerminationScope::Process => {
                    let threads: Vec<_> = coordinator
                        .process(lease.process)
                        .into_iter()
                        .flat_map(|process| process.threads().iter().map(|(id, _)| *id))
                        .collect();
                    for thread in threads {
                        self.release_thread_synchronization(thread);
                    }
                }
            }
        }
        if self
            .apply_pending_runtime_request(coordinator, lease.process, lease.thread)
            .map_err(HorizonScheduledDispatchError::Runtime)?
        {
            return Ok(nixe_runtime::ExceptionHandlingResult::Resumed);
        }
        if handling == nixe_runtime::ExceptionHandlingResult::Suspended {
            if let Some((events, timeout)) = self.take_thread_wait(lease.thread.get()) {
                let timeout_ns =
                    timeout.map(|duration| u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX));
                coordinator
                    .register_event_wait(
                        lease.thread,
                        events,
                        timeout_ns,
                        nixe_runtime::ExternalEventSource::Device,
                    )
                    .map_err(HorizonScheduledDispatchError::Coordinator)?;
            } else {
                // Retry-style SVCs have committed their kernel-side effect and
                // intentionally re-enter the guest without an external wait.
                coordinator
                    .make_thread_ready(lease.thread)
                    .map_err(HorizonScheduledDispatchError::Coordinator)?;
            }
        }
        Ok(handling)
    }

    fn release_thread_synchronization(&mut self, thread: GuestThreadId) {
        let thread_id = thread.get();
        self.pending_runtime_requests.remove(&thread);
        self.pending_wakes.remove(&thread_id);
        self.wait_deadlines
            .retain(|(candidate, _), _| *candidate != thread_id);
    }
}

#[derive(Debug)]
pub enum HorizonScheduledDispatchError {
    Route(nixe_runtime::CoordinatorRouteError),
    Coordinator(nixe_runtime::CoordinatorError),
    Runtime(HorizonSvcFault),
}

impl std::fmt::Display for HorizonScheduledDispatchError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Route(error) => error.fmt(formatter),
            Self::Coordinator(error) => error.fmt(formatter),
            Self::Runtime(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for HorizonScheduledDispatchError {}
