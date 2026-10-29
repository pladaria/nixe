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
        let (event, number) = match stop {
            nixe_runtime::ExecutionStop::SupervisorCall { immediate, .. } => {
                ("svc_dispatch", u64::from(*immediate))
            }
            _ => ("cpu_exception_dispatch", u64::MAX),
        };
        let _trace =
            nixe_memory::trace::span(event, lease.generation.get(), lease.thread.get(), number);
        let handling = coordinator
            .route_supervisor_call(lease, stop, self)
            .map_err(HorizonScheduledDispatchError::Route)?;
        self.finish_scheduled_call(coordinator, lease.process, lease.thread, handling)
    }

    pub(super) fn finish_scheduled_call(
        &mut self,
        coordinator: &mut nixe_runtime::RuntimeCoordinator,
        _process: ProcessId,
        thread: GuestThreadId,
        handling: nixe_runtime::ExceptionHandlingResult<HorizonSvcFault>,
    ) -> Result<nixe_runtime::ExceptionHandlingResult<HorizonSvcFault>, HorizonScheduledDispatchError>
    {
        if matches!(handling, nixe_runtime::ExceptionHandlingResult::Suspended) {
            if self
                .register_memory_wait(coordinator, thread)
                .map_err(HorizonScheduledDispatchError::Coordinator)?
            {
                return Ok(handling);
            }
            return Err(HorizonScheduledDispatchError::Coordinator(
                nixe_runtime::CoordinatorError::MissingContinuation(thread),
            ));
        }
        Ok(handling)
    }
}

#[derive(Debug)]
pub enum HorizonScheduledDispatchError {
    Route(nixe_runtime::CoordinatorRouteError),
    Coordinator(nixe_runtime::CoordinatorError),
}

impl std::fmt::Display for HorizonScheduledDispatchError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Route(error) => error.fmt(formatter),
            Self::Coordinator(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for HorizonScheduledDispatchError {}
