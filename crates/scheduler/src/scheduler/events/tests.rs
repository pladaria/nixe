use super::*;
use crate::{PriorityRange, VirtualClock, VirtualClockMode, VirtualCpuDescriptor};
use std::{
    future::Future,
    sync::{Mutex, Weak, mpsc},
    task::Wake,
    time::Duration,
};

fn kernel(mode: VirtualClockMode) -> Arc<SchedulerState> {
    SchedulerState::new(
        MachineSchedulerProfile::new(
            vec![VirtualCpuDescriptor::new(VirtualCpuId::new(0), 0)],
            PriorityRange::new(0, 63).unwrap(),
        )
        .unwrap(),
        VirtualClock::new(mode),
    )
}
fn fixed() -> Arc<SchedulerState> {
    kernel(VirtualClockMode::Fixed { unix_seconds: 0 })
}
fn caller(kernel: &Arc<SchedulerState>, priority: i32) -> GuestThreadId {
    let thread = kernel
        .create_thread(ScheduledThreadConfig {
            process: ProcessId::new(1),
            base_priority: priority,
            ideal_vcpu: Some(VirtualCpuId::new(0)),
            affinity: kernel.profile().all_cores(),
        })
        .unwrap();
    kernel.make_ready(thread).unwrap();
    let lease = kernel.claim(VirtualCpuId::new(0)).unwrap().unwrap();
    assert_eq!(lease.thread, thread);
    kernel.complete(lease, Completion::Waiting).unwrap();
    thread
}
fn pending(
    kernel: &Arc<SchedulerState>,
    thread: GuestThreadId,
    events: &[ReadableEventObject],
    timeout: Option<u64>,
) -> Arc<KernelWait> {
    match kernel
        .wait_on_objects(thread, events, timeout, true)
        .unwrap()
    {
        WaitStart::Pending(wait) => wait,
        result => panic!("expected a pending wait: {result:?}"),
    }
}
#[test]
fn signal_before_park_selects_the_first_duplicate_argument_and_unlinks_every_object() {
    let kernel = fixed();
    let thread = caller(&kernel, 20);
    let (a, ar) = EventObject::create_pair(&kernel);
    let (b, br) = EventObject::create_pair(&kernel);
    let wait = pending(&kernel, thread, &[ar.clone(), br.clone(), ar, br], None);
    b.signal();
    assert_eq!(wait.phase(), WaitPhase::Resolved(WaitOutcome::Signalled(1)));
    a.signal();
    assert_eq!(wait.outcome(), Some(WaitOutcome::Signalled(1)));
    assert_eq!(
        Pin::new(&mut std::pin::pin!(wait.wait())).poll(&mut Context::from_waker(Waker::noop())),
        Poll::Ready(WaitOutcome::Signalled(1))
    );
    kernel.finish_wait(wait.token()).unwrap();
    kernel.stop_timers().unwrap();
}
struct OrderedNotify {
    kernel: Weak<SchedulerState>,
    order: Arc<Mutex<Vec<u64>>>,
    id: u64,
}
impl Wake for OrderedNotify {
    fn wake(self: Arc<Self>) {
        assert!(self.kernel.upgrade().unwrap().state.try_lock().is_ok());
        self.order.lock().unwrap().push(self.id);
    }
}
#[test]
fn manual_signal_preserves_fifo_when_middle_wait_nodes_are_unlinked() {
    let kernel = fixed();
    let (signal, event) = EventObject::create_pair(&kernel);
    let (_, other) = EventObject::create_pair(&kernel);
    let order = Arc::new(Mutex::new(Vec::new()));
    let mut waits = Vec::new();
    let mut futures = Vec::new();
    for id in 1..=3 {
        let thread = caller(&kernel, 20);
        let wait = pending(&kernel, thread, &[other.clone(), event.clone()], None);
        let owned = wait.clone();
        let mut future = Box::pin(async move { owned.wait().await });
        let waker = Waker::from(Arc::new(OrderedNotify {
            kernel: Arc::downgrade(&kernel),
            order: order.clone(),
            id,
        }));
        assert!(
            future
                .as_mut()
                .poll(&mut Context::from_waker(&waker))
                .is_pending()
        );
        waits.push(wait);
        futures.push(future);
    }
    signal.signal();
    assert_eq!(*order.lock().unwrap(), [1, 2, 3]);
    for wait in waits {
        assert_eq!(wait.outcome(), Some(WaitOutcome::Signalled(1)));
        kernel.finish_wait(wait.token()).unwrap();
    }
    kernel.stop_timers().unwrap();
}
#[test]
fn auto_reset_uses_current_effective_priority_then_fifo_and_consumes_one_signal() {
    let kernel = fixed();
    let (signal, event) = EventObject::create_auto_reset_pair(&kernel);
    let mut waits = Vec::new();
    for priority in [10, 20, 10] {
        let thread = caller(&kernel, priority);
        waits.push(pending(&kernel, thread, std::slice::from_ref(&event), None));
    }
    kernel
        .set_priority(waits[1].token().thread, 5, None)
        .unwrap();
    for selected in [1, 0, 2] {
        signal.signal();
        assert_eq!(waits[selected].outcome(), Some(WaitOutcome::Signalled(0)));
        assert!(!event.is_signalled());
        kernel.finish_wait(waits[selected].token()).unwrap();
    }
    kernel.stop_timers().unwrap();
}
#[test]
fn readiness_precedes_zero_timeout_and_only_a_blocking_wait_consumes_cancellation() {
    let kernel = fixed();
    let thread = caller(&kernel, 20);
    let (signal, event) = EventObject::create_auto_reset_pair(&kernel);
    kernel.cancel_synchronization(thread).unwrap();
    signal.signal();
    assert!(matches!(
        kernel
            .wait_on_objects(thread, std::slice::from_ref(&event), Some(0), true)
            .unwrap(),
        WaitStart::Ready(WaitOutcome::Signalled(0))
    ));
    assert!(matches!(
        kernel
            .wait_on_objects(thread, std::slice::from_ref(&event), Some(0), true)
            .unwrap(),
        WaitStart::Ready(WaitOutcome::TimedOut)
    ));
    assert!(matches!(
        kernel
            .wait_on_objects(thread, &[event], None, true)
            .unwrap(),
        WaitStart::Ready(WaitOutcome::Cancelled)
    ));
    assert!(kernel.thread(thread).unwrap().active_wait.is_none());
    kernel.stop_timers().unwrap();
}
#[test]
fn an_auto_reset_host_claim_survives_clear_without_consuming_another_signal() {
    let kernel = fixed();
    let (signal, event) = EventObject::create_auto_reset_pair(&kernel);
    let mut first = event.wait_async();
    let mut second = event.wait_async();
    let mut context = Context::from_waker(Waker::noop());
    assert!(Pin::new(&mut first).poll(&mut context).is_pending());
    assert!(Pin::new(&mut second).poll(&mut context).is_pending());
    signal.signal();
    event.clear();
    assert!(Pin::new(&mut first).poll(&mut context).is_ready());
    assert!(Pin::new(&mut second).poll(&mut context).is_pending());
    signal.signal();
    assert!(Pin::new(&mut second).poll(&mut context).is_ready());
    signal.signal();
    assert!(
        Pin::new(&mut event.wait_async())
            .poll(&mut context)
            .is_ready()
    );
    assert!(!event.is_signalled());
    kernel.stop_timers().unwrap();
}
#[test]
fn cancelling_or_exiting_unlinks_every_object_and_late_signals_cannot_reopen_the_generation() {
    for outcome in [WaitOutcome::Cancelled, WaitOutcome::Terminated] {
        let kernel = fixed();
        let thread = caller(&kernel, 20);
        let (signal, event) = EventObject::create_pair(&kernel);
        let wait = pending(&kernel, thread, &[event.clone(), event], Some(100));
        assert!(kernel.resolve_wait(wait.token(), outcome));
        signal.signal();
        kernel.advance_time(100);
        assert_eq!(wait.outcome(), Some(outcome));
        assert!(!wait.commit());
        kernel.finish_wait(wait.token()).unwrap();
        kernel.stop_timers().unwrap();
    }
}
#[test]
fn fixed_time_and_pause_use_the_same_deadline_queue_without_advancing_devices() {
    let kernel = fixed();
    let thread = caller(&kernel, 20);
    let (_, event) = EventObject::create_pair(&kernel);
    let wait = pending(&kernel, thread, &[event], Some(100));
    kernel.clock().set_paused(true);
    assert_eq!(kernel.advance_time(200), 0);
    assert_eq!(kernel.clock().scheduler_time_ns(), 0);
    assert_eq!(wait.outcome(), None);
    kernel.clock().set_paused(false);
    kernel.advance_time(99);
    assert_eq!(wait.outcome(), None);
    kernel.advance_time(100);
    assert_eq!(wait.outcome(), Some(WaitOutcome::TimedOut));
    assert_eq!(kernel.clock().elapsed(), Duration::from_nanos(100));
    kernel.finish_wait(wait.token()).unwrap();
    kernel.stop_timers().unwrap();
}
struct Notify(mpsc::Sender<()>);
impl Wake for Notify {
    fn wake(self: Arc<Self>) {
        self.0.send(()).unwrap();
    }
}
#[test]
fn a_device_signal_notifies_an_idle_kernel_without_cpu_completion() {
    let kernel = fixed();
    let thread = caller(&kernel, 20);
    let (signal, event) = EventObject::create_pair(&kernel);
    let wait = pending(&kernel, thread, &[event], None);
    let (send, receive) = mpsc::channel();
    let waker = Waker::from(Arc::new(Notify(send)));
    let mut future = std::pin::pin!(wait.wait());
    assert!(
        future
            .as_mut()
            .poll(&mut Context::from_waker(&waker))
            .is_pending()
    );
    assert!(kernel.claim(VirtualCpuId::new(0)).unwrap().is_none());
    std::thread::spawn(move || signal.signal()).join().unwrap();
    receive.recv_timeout(Duration::from_secs(1)).unwrap();
    assert_eq!(wait.outcome(), Some(WaitOutcome::Signalled(0)));
    assert_eq!(
        kernel.thread(thread).unwrap().lifecycle,
        ThreadLifecycle::Waiting
    );
    kernel.finish_wait(wait.token()).unwrap();
    kernel.stop_timers().unwrap();
}

#[test]
fn shutdown_resolves_untimed_waits_and_releases_the_events_retaining_the_kernel() {
    let kernel = fixed();
    let weak = Arc::downgrade(&kernel);
    let thread = caller(&kernel, 20);
    let (_, event) = EventObject::create_pair(&kernel);
    let wait = pending(&kernel, thread, &[event], None);
    kernel.stop_timers().unwrap();
    assert_eq!(wait.outcome(), Some(WaitOutcome::Terminated));
    drop(kernel);
    assert!(weak.upgrade().is_none());
}
