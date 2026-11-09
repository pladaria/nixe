pub(crate) fn test_kernel() -> std::sync::Arc<nixe_scheduler::SchedulerState> {
    use nixe_scheduler::{
        MachineSchedulerProfile, PriorityRange, SchedulerState, VirtualCpuDescriptor, VirtualCpuId,
    };
    SchedulerState::new(
        MachineSchedulerProfile::new(
            (0..8)
                .map(|id| VirtualCpuDescriptor::new(VirtualCpuId::new(id), 0))
                .collect(),
            PriorityRange::new(0, 63).unwrap(),
        )
        .unwrap(),
        crate::VirtualClock::new(crate::VirtualClockMode::Fixed { unix_seconds: 0 }),
    )
}

pub(crate) fn poll_session(
    session: &crate::SessionObject,
    owner: crate::SessionRequestOwner,
) -> Result<Option<crate::SessionMessage>, crate::SessionError> {
    use std::{
        future::Future,
        pin::Pin,
        task::{Context, Poll, Waker},
    };
    match Pin::new(&mut session.response(owner)).poll(&mut Context::from_waker(Waker::noop())) {
        Poll::Pending => Ok(None),
        Poll::Ready(result) => result.map(Some),
    }
}

pub(crate) fn watch_future<F: std::future::Future + Unpin>(
    mut future: F,
    callback: std::sync::Arc<dyn Fn() + Send + Sync>,
) -> F {
    use std::{
        pin::Pin,
        task::{Context, Wake, Waker},
    };
    struct Notify(std::sync::Arc<dyn Fn() + Send + Sync>);
    impl Wake for Notify {
        fn wake(self: std::sync::Arc<Self>) {
            (self.0)();
        }
    }
    let waker = Waker::from(std::sync::Arc::new(Notify(callback)));
    assert!(
        Pin::new(&mut future)
            .poll(&mut Context::from_waker(&waker))
            .is_pending()
    );
    future
}
