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
        self.synchronize_virtual_time(coordinator.virtual_time_ns());
        let handling = coordinator
            .route_supervisor_call(lease, stop, self)
            .map_err(HorizonScheduledDispatchError::Route)?;
        self.finish_scheduled_call(coordinator, lease.process, lease.thread, handling)
    }

    pub(super) fn finish_scheduled_call(
        &mut self,
        coordinator: &mut nixe_runtime::RuntimeCoordinator,
        process: ProcessId,
        thread: GuestThreadId,
        handling: nixe_runtime::ExceptionHandlingResult<HorizonSvcFault>,
    ) -> Result<nixe_runtime::ExceptionHandlingResult<HorizonSvcFault>, HorizonScheduledDispatchError>
    {
        if let nixe_runtime::ExceptionHandlingResult::Terminated { scope, .. } = &handling {
            match scope {
                nixe_runtime::ExceptionTerminationScope::CurrentThread => {
                    self.release_thread_synchronization(thread);
                }
                nixe_runtime::ExceptionTerminationScope::Process => {
                    let threads: Vec<_> = coordinator
                        .process(process)
                        .into_iter()
                        .flat_map(|process| process.threads().iter().map(|(id, _)| *id))
                        .collect();
                    for thread in threads {
                        self.release_thread_synchronization(thread);
                    }
                }
            }
        }
        if let Some(handling) = self
            .apply_pending_runtime_request(coordinator, process, thread)
            .map_err(HorizonScheduledDispatchError::Runtime)?
        {
            return Ok(handling);
        }
        if matches!(handling, nixe_runtime::ExceptionHandlingResult::Suspended) {
            if self
                .register_memory_wait(coordinator, thread)
                .map_err(HorizonScheduledDispatchError::Coordinator)?
            {
                return Ok(handling);
            }
            if let Some((events, timeout)) = self.take_thread_wait(thread.get()) {
                let timeout_ns =
                    timeout.map(|duration| u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX));
                coordinator
                    .register_event_wait(
                        thread,
                        events,
                        timeout_ns,
                        nixe_runtime::ExternalEventSource::Device,
                    )
                    .map_err(HorizonScheduledDispatchError::Coordinator)?;
            } else {
                // Retry-style SVCs have committed their kernel-side effect and
                // intentionally re-enter the guest without an external wait.
                coordinator
                    .make_thread_ready(thread)
                    .map_err(HorizonScheduledDispatchError::Coordinator)?;
            }
        }
        Ok(handling)
    }

    pub(super) fn release_thread_synchronization(&mut self, thread: GuestThreadId) {
        let thread_id = thread.get();
        self.pending_runtime_requests.remove(&thread);
        self.cancel_memory_wait(thread);
        self.pending_wakes.remove(&thread_id);
        self.cancellable_waits.remove(&thread_id);
        self.cancelled_waits.remove(&thread_id);
        self.pending_wait_cancellations.remove(&thread_id);
        self.synchronization_waits.remove(&thread_id);
        self.session_requests.remove(&thread_id);
        self.reply_waits.remove(&thread_id);
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
