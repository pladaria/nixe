use super::*;
use std::{
    future::Future,
    pin::Pin,
    sync::{
        Arc, Weak,
        atomic::{AtomicUsize, Ordering},
    },
    task::{Context, Poll, Wake, Waker},
};

struct Notification {
    scheduler: Weak<SchedulerState>,
    calls: AtomicUsize,
}
impl Wake for Notification {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }
    fn wake_by_ref(self: &Arc<Self>) {
        // Reentering the real scheduler proves notification occurs after unlock.
        self.scheduler.upgrade().unwrap().thread_count();
        self.calls.fetch_add(1, Ordering::Relaxed);
    }
}
fn notification(s: &Arc<SchedulerState>) -> (Arc<Notification>, Waker) {
    let n = Arc::new(Notification {
        scheduler: Arc::downgrade(s),
        calls: AtomicUsize::new(0),
    });
    (n.clone(), Waker::from(n))
}
fn poll<F: Future + Unpin>(future: &mut F, waker: &Waker) -> Poll<F::Output> {
    Pin::new(future).poll(&mut Context::from_waker(waker))
}
fn kernel_call(s: &SchedulerState, caller: GuestThreadId, core: u32) {
    let lease = lease(s, core);
    assert_eq!(lease.thread, caller);
    assert!(
        s.complete(
            lease,
            Completion::KernelCall {
                interrupt_handled: true
            }
        )
        .unwrap()
    );
}

#[test]
fn an_unpolled_or_recursive_acquisition_cannot_release_an_existing_guard() {
    let s = scheduler(1);
    let caller = add(&s, 1, 30, Some(0), &[0]);
    kernel_call(&s, caller, 0);
    let guard = s.try_thread_property(caller, caller).unwrap().unwrap();
    drop(s.lock_thread_property(caller, caller));
    assert!(s.try_thread_property(caller, caller).unwrap().is_none());
    let (_, waker) = notification(&s);
    let mut recursive = s.lock_thread_property(caller, caller);
    assert!(
        matches!(poll(&mut recursive, &waker), Poll::Ready(Err(SchedulerError::PriorityCycle(id))) if id == caller)
    );
    drop(recursive);
    assert!(s.try_thread_property(caller, caller).unwrap().is_none());
    drop(guard);
    assert!(s.try_thread_property(caller, caller).unwrap().is_some());
}

#[test]
fn property_handoff_preserves_wait_generations_until_results_are_installed() {
    let s = scheduler(2);
    let target = add(&s, 1, 50, Some(0), &[0]);
    let first = add(&s, 1, 30, Some(1), &[1]);
    s.pin_thread(VirtualCpuId::new(0), target).unwrap();
    kernel_call(&s, first, 1);
    let guard = s.try_thread_property(first, target).unwrap().unwrap();
    s.set_activity(target, true).unwrap();
    let (first_wake, waker) = notification(&s);
    let mut quiescence = Box::pin(guard.wait_quiescent(None));
    assert!(poll(&mut quiescence, &waker).is_pending());
    let first_token = s.register_wait(first).unwrap().token();
    s.set_activity(first, true).unwrap();
    let second = add(&s, 2, 10, Some(1), &[1]);
    kernel_call(&s, second, 1);
    let (second_wake, waker2) = notification(&s);
    let mut acquire = s.lock_thread_property(second, target);
    assert!(poll(&mut acquire, &waker2).is_pending());
    let second_token = s.register_wait(second).unwrap().token();
    assert_eq!(s.thread(first).unwrap().effective_priority, 10);
    s.unpin_thread(target).unwrap();
    assert_eq!(first_wake.calls.load(Ordering::Relaxed), 1);
    assert!(matches!(poll(&mut quiescence, &waker), Poll::Ready(Ok(()))));
    assert_eq!(s.thread(first).unwrap().active_wait, Some(first_token));
    assert_eq!(s.thread(first).unwrap().lifecycle, ThreadLifecycle::Waiting);
    drop(quiescence);
    drop(guard);
    assert_eq!(second_wake.calls.load(Ordering::Relaxed), 1);
    assert_eq!(s.thread(first).unwrap().effective_priority, 30);
    assert_eq!(
        s.thread(second).unwrap().lifecycle,
        ThreadLifecycle::Waiting
    );
    let Poll::Ready(Ok(next)) = poll(&mut acquire, &waker2) else {
        panic!("handoff must grant ownership");
    };
    s.finish_wait(first_token).unwrap();
    assert_eq!(
        s.thread(first).unwrap().lifecycle,
        ThreadLifecycle::Suspended
    );
    assert_eq!(s.thread(second).unwrap().active_wait, Some(second_token));
    drop(next);
    s.finish_wait(second_token).unwrap();
    assert_eq!(selected(&s, 1), Some(second));
}

#[test]
fn cancelling_a_contender_before_or_after_handoff_releases_the_real_node() {
    for handoff in [false, true] {
        let s = scheduler(2);
        let target = add(&s, 1, 50, Some(0), &[0]);
        let first = add(&s, 1, 30, Some(1), &[1]);
        kernel_call(&s, first, 1);
        let guard = s.try_thread_property(first, target).unwrap().unwrap();
        s.register_wait(first).unwrap().token();
        let second = add(&s, 1, 10, Some(1), &[1]);
        kernel_call(&s, second, 1);
        let (_, waker) = notification(&s);
        let mut second_lock = s.lock_thread_property(second, target);
        assert!(poll(&mut second_lock, &waker).is_pending());
        s.register_wait(second).unwrap().token();
        let third = add(&s, 1, 20, Some(1), &[1]);
        kernel_call(&s, third, 1);
        let mut third_lock = s.lock_thread_property(third, target);
        assert!(poll(&mut third_lock, &waker).is_pending());
        s.register_wait(third).unwrap().token();
        if handoff {
            drop(guard);
            drop(second_lock);
        } else {
            drop(second_lock);
            assert_eq!(s.thread(first).unwrap().effective_priority, 20);
            drop(guard);
        }
        let Poll::Ready(Ok(next)) = poll(&mut third_lock, &waker) else {
            panic!("remaining contender must own the lock");
        };
        assert_eq!(s.priority_donation_count(), 0);
        drop(next);
    }
}

#[test]
fn terminating_an_owner_hands_off_and_stale_guards_cannot_release_its_replacement() {
    let s = scheduler(2);
    let target = add(&s, 1, 50, Some(0), &[0]);
    let first = add(&s, 1, 30, Some(1), &[1]);
    kernel_call(&s, first, 1);
    let guard = s.try_thread_property(first, target).unwrap().unwrap();
    s.register_wait(first).unwrap().token();
    let second = add(&s, 1, 10, Some(1), &[1]);
    kernel_call(&s, second, 1);
    let (_, waker) = notification(&s);
    let mut acquire = s.lock_thread_property(second, target);
    assert!(poll(&mut acquire, &waker).is_pending());
    s.register_wait(second).unwrap().token();
    s.terminate(first, false).unwrap();
    s.unregister(first).unwrap();
    let replacement = add(&s, 1, 40, Some(1), &[1]);
    assert_ne!(first, replacement);
    let Poll::Ready(Ok(next)) = poll(&mut acquire, &waker) else {
        panic!("live waiter must receive the lock");
    };
    drop(guard);
    assert!(
        s.try_thread_property(replacement, target)
            .unwrap()
            .is_none()
    );
    drop(next);
    assert!(
        s.try_thread_property(replacement, target)
            .unwrap()
            .is_some()
    );
}

#[test]
fn retiring_a_pin_owner_waiting_on_its_own_property_cannot_resurrect_the_slot() {
    for remove in [false, true] {
        let s = scheduler(1);
        let target = add(&s, 1, 30, Some(0), &[0]);
        s.pin_thread(VirtualCpuId::new(0), target).unwrap();
        kernel_call(&s, target, 0);
        let guard = s.try_thread_property(target, target).unwrap().unwrap();
        let (_, waker) = notification(&s);
        let mut wait = Box::pin(guard.wait_quiescent(None));
        assert!(poll(&mut wait, &waker).is_pending());
        s.register_wait(target).unwrap().token();
        if !remove {
            s.terminate(target, false).unwrap();
        }
        s.unregister(target).unwrap();
        drop(wait);
        drop(guard);
        assert_eq!(selected(&s, 0), None);
        assert!(s.active_waits().next().is_none());
        let replacement = add(&s, 2, 20, Some(0), &[0]);
        assert_ne!(target, replacement);
        assert_eq!(selected(&s, 0), Some(replacement));
    }
}

#[test]
fn affinity_preserves_unassigned_suggestions_until_a_real_migration_search() {
    let s = scheduler(2);
    let remote = add(&s, 1, 20, Some(1), &[1]);
    let target = add(&s, 1, 20, Some(0), &[0, 1]);
    let local = add(&s, 1, 20, Some(0), &[0]);
    s.yield_thread(target, YieldMigration::Any).unwrap();
    assert_eq!(selected(&s, 0), Some(local));
    s.migrate(
        target,
        IdealCore::Core(VirtualCpuId::new(1)),
        s.profile().core_set([VirtualCpuId::new(1)]).unwrap(),
    )
    .unwrap();
    s.yield_thread(remote, YieldMigration::None).unwrap();
    assert_eq!(selected(&s, 1), Some(remote));
    s.complete(lease(&s, 1), Completion::Waiting).unwrap();
    assert_eq!(selected(&s, 1), Some(target));
}

#[test]
fn a_registered_pin_wait_survives_native_boundaries_and_blocking_until_unpin() {
    for block in [false, true] {
        let s = scheduler(2);
        let target = add(&s, 1, 50, Some(0), &[0]);
        let caller = add(&s, 1, 30, Some(1), &[1]);
        s.pin_thread(VirtualCpuId::new(0), target).unwrap();
        let target_lease = lease(&s, 0);
        kernel_call(&s, caller, 1);
        let guard = s.try_thread_property(caller, target).unwrap().unwrap();
        let allowed = s.profile().core_set([VirtualCpuId::new(1)]).unwrap();
        s.migrate(target, IdealCore::Core(VirtualCpuId::new(1)), allowed)
            .unwrap();
        let (notified, waker) = notification(&s);
        let mut wait = Box::pin(guard.wait_quiescent(Some(allowed)));
        assert!(poll(&mut wait, &waker).is_pending());
        let token = s.register_wait(caller).unwrap().token();
        s.complete(
            target_lease,
            Completion::KernelCall {
                interrupt_handled: true,
            },
        )
        .unwrap();
        assert!(!guard.is_quiescent(Some(allowed)).unwrap());
        assert!(poll(&mut wait, &waker).is_pending());
        if block {
            s.register_wait(target).unwrap().token();
            assert!(poll(&mut wait, &waker).is_pending());
        }
        assert_eq!(notified.calls.load(Ordering::Relaxed), 0);
        assert_eq!(s.thread(caller).unwrap().active_wait, Some(token));
        s.unpin_thread(target).unwrap();
        assert_eq!(notified.calls.load(Ordering::Relaxed), 1);
        assert!(matches!(poll(&mut wait, &waker), Poll::Ready(Ok(()))));
        assert_eq!(s.thread(caller).unwrap().active_wait, Some(token));
        drop(wait);
        drop(guard);
        s.finish_wait(token).unwrap();
    }
}

#[test]
fn a_mask_change_to_an_already_blocked_pinned_thread_needs_no_unpin_wait() {
    let s = scheduler(2);
    let target = add(&s, 1, 50, Some(0), &[0]);
    let caller = add(&s, 1, 30, Some(1), &[1]);
    s.pin_thread(VirtualCpuId::new(0), target).unwrap();
    s.complete(lease(&s, 0), Completion::Waiting).unwrap();
    s.register_wait(target).unwrap().token();
    kernel_call(&s, caller, 1);
    let guard = s.try_thread_property(caller, target).unwrap().unwrap();
    let allowed = s.profile().core_set([VirtualCpuId::new(1)]).unwrap();
    s.migrate(target, IdealCore::Core(VirtualCpuId::new(1)), allowed)
        .unwrap();
    assert!(guard.is_quiescent(Some(allowed)).unwrap());
    let (_, waker) = notification(&s);
    let mut wait = Box::pin(guard.wait_quiescent(Some(allowed)));
    assert!(matches!(poll(&mut wait, &waker), Poll::Ready(Ok(()))));
    assert_eq!(
        s.pinned_thread(ProcessId::new(1), VirtualCpuId::new(0))
            .unwrap(),
        Some(target)
    );
}

#[test]
fn a_suspended_final_property_recipient_waits_for_resume_or_a_real_kernel_waiter() {
    for kernel_waiter in [false, true] {
        let s = scheduler(2);
        let target = add(&s, 1, 50, Some(0), &[0]);
        let first = add(&s, 1, 30, Some(1), &[1]);
        kernel_call(&s, first, 1);
        let guard = s.try_thread_property(first, target).unwrap().unwrap();
        let first_token = s.register_wait(first).unwrap().token();
        let second = add(&s, 1, 20, Some(1), &[1]);
        kernel_call(&s, second, 1);
        let (notified, waker) = notification(&s);
        let mut acquire = s.lock_thread_property(second, target);
        assert!(poll(&mut acquire, &waker).is_pending());
        let token = s.register_wait(second).unwrap().token();
        s.set_activity(second, true).unwrap();
        drop(guard);
        s.finish_wait(first_token).unwrap();
        assert_eq!(notified.calls.load(Ordering::Relaxed), 0);
        assert!(poll(&mut acquire, &waker).is_pending());
        assert_eq!(s.thread(second).unwrap().active_wait, Some(token));
        let mut contender = None;
        if kernel_waiter {
            let third = add(&s, 1, 10, Some(1), &[1]);
            kernel_call(&s, third, 1);
            let (_, waker3) = notification(&s);
            let mut lock = s.lock_thread_property(third, target);
            assert!(poll(&mut lock, &waker3).is_pending());
            s.register_wait(third).unwrap().token();
            contender = Some(lock);
        } else {
            s.set_activity(second, false).unwrap();
        }
        assert_eq!(notified.calls.load(Ordering::Relaxed), 1);
        let Poll::Ready(Ok(next)) = poll(&mut acquire, &waker) else {
            panic!("eligible recipient must continue its actual operation");
        };
        assert_eq!(s.thread(second).unwrap().paused, kernel_waiter);
        drop(next);
        s.finish_wait(token).unwrap();
        drop(contender);
        assert_eq!(s.priority_donation_count(), 0);
    }
}

#[test]
fn quiescence_completion_respects_caller_suspension_and_real_context_release() {
    for pinned in [false, true] {
        let s = scheduler(2);
        let target = add(&s, 1, 50, Some(0), &[0]);
        let caller = add(&s, 1, 30, Some(1), &[1]);
        if pinned {
            s.pin_thread(VirtualCpuId::new(0), target).unwrap();
        }
        let executing = lease(&s, 0);
        kernel_call(&s, caller, 1);
        let guard = s.try_thread_property(caller, target).unwrap().unwrap();
        s.set_activity(target, true).unwrap();
        let (notified, waker) = notification(&s);
        let mut wait = Box::pin(guard.wait_quiescent(None));
        assert!(poll(&mut wait, &waker).is_pending());
        let token = s.register_wait(caller).unwrap().token();
        s.set_activity(caller, true).unwrap();
        if pinned {
            s.unpin_thread(target).unwrap();
            // The caller is suspended after its real pin wait ended. A later
            // pin must not restart that completed dependency when it resumes.
            s.pin_thread(VirtualCpuId::new(0), target).unwrap();
        }
        assert!(poll(&mut wait, &waker).is_pending());
        s.complete(executing, Completion::Ready).unwrap();
        assert_eq!(notified.calls.load(Ordering::Relaxed), 0);
        assert!(poll(&mut wait, &waker).is_pending());
        s.set_activity(caller, false).unwrap();
        assert_eq!(notified.calls.load(Ordering::Relaxed), 1);
        assert!(matches!(poll(&mut wait, &waker), Poll::Ready(Ok(()))));
        drop(wait);
        drop(guard);
        s.finish_wait(token).unwrap();
    }
}
