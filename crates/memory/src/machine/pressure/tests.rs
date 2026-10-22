use super::*;
use std::{sync::atomic::AtomicUsize, task::Wake};

struct WakeProbe {
    machine: Machine,
    wakes: Arc<AtomicUsize>,
}
impl Wake for WakeProbe {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }
    fn wake_by_ref(self: &Arc<Self>) {
        assert!(self.machine.0.state.try_lock().is_ok());
        assert!(self.machine.0.pressure.registry.try_lock().is_ok());
        self.wakes.fetch_add(1, Ordering::Relaxed);
    }
}
impl Drop for WakeProbe {
    fn drop(&mut self) {
        assert!(self.machine.0.state.try_lock().is_ok());
        assert!(self.machine.0.pressure.registry.try_lock().is_ok());
    }
}
fn waker(machine: &Machine, wakes: &Arc<AtomicUsize>) -> Waker {
    Waker::from(Arc::new(WakeProbe {
        machine: machine.clone(),
        wakes: Arc::clone(wakes),
    }))
}

#[test]
fn physical_retirement_wakes_owned_allocation_and_cancellation_drops_wakers_unlocked() {
    let machine = Machine::new(PAGE_SIZE).unwrap();
    let account = machine.account(PAGE_SIZE).unwrap();
    let old = account.allocate(PAGE_SIZE).unwrap();
    let pin = old.pin();
    drop(old);
    let wakes = Arc::new(AtomicUsize::new(0));
    let first = waker(&machine, &wakes);
    let mut future = Box::pin(account.allocate_async(PAGE_SIZE));
    assert!(
        future
            .as_mut()
            .poll(&mut Context::from_waker(&first))
            .is_pending()
    );
    assert_eq!(machine.pending_allocation_count(), 1);
    drop(first);
    let second = waker(&machine, &wakes);
    assert!(
        future
            .as_mut()
            .poll(&mut Context::from_waker(&second))
            .is_pending()
    );
    // Replacing the final old waker runs WakeProbe::drop outside registry locks.
    assert_eq!(machine.pending_allocation_count(), 1);
    drop(pin);
    assert_eq!(wakes.load(Ordering::Relaxed), 1);
    assert_eq!(machine.pending_allocation_count(), 0);
    let Poll::Ready(Ok(new)) = future.as_mut().poll(&mut Context::from_waker(&second)) else {
        panic!("the final old pin released the requested physical RAM");
    };
    drop(new);
    let old = account.allocate(PAGE_SIZE).unwrap();
    let pin = old.pin();
    drop(old);
    let mut cancelled = Box::pin(account.allocate_async(PAGE_SIZE));
    assert!(
        cancelled
            .as_mut()
            .poll(&mut Context::from_waker(&second))
            .is_pending()
    );
    drop(second);
    drop(cancelled);
    assert_eq!(machine.pending_allocation_count(), 0);
    drop(pin);
    assert_eq!(wakes.load(Ordering::Relaxed), 1);
    assert_eq!(machine.statistics().allocated_bytes, 0);
}

#[test]
fn original_native_failure_wakes_every_allocation_without_waiting_for_retirement() {
    let machine = Machine::new(PAGE_SIZE).unwrap();
    let account = machine.account(PAGE_SIZE).unwrap();
    let old = account.allocate(PAGE_SIZE).unwrap();
    let pin = old.pin();
    drop(old);
    let wakes = Arc::new(AtomicUsize::new(0));
    let wake = waker(&machine, &wakes);
    let mut first = Box::pin(account.allocate_async(PAGE_SIZE));
    let mut second = Box::pin(account.allocate_contiguous_async(PAGE_SIZE, PAGE_SIZE));
    assert!(
        first
            .as_mut()
            .poll(&mut Context::from_waker(&wake))
            .is_pending()
    );
    assert!(
        second
            .as_mut()
            .poll(&mut Context::from_waker(&wake))
            .is_pending()
    );
    assert_eq!(machine.pending_allocation_count(), 2);
    let cause = Arc::new(MemoryError::Invalid("original native submission failure"));
    machine.fail_pending_allocations(Arc::clone(&cause));
    machine.fail_pending_allocations(Arc::new(MemoryError::Invalid("later teardown failure")));
    assert_eq!(wakes.load(Ordering::Relaxed), 2);
    assert_eq!(machine.pending_allocation_count(), 0);
    drop(pin); // More free RAM cannot turn the failed dependency into success.
    for future in [&mut first, &mut second] {
        let Poll::Ready(Err(error)) = future.as_mut().poll(&mut Context::from_waker(&wake)) else {
            panic!("terminal failure must complete the original allocation continuation");
        };
        assert!(Arc::ptr_eq(&error, &cause));
    }
    assert_eq!(machine.statistics().allocated_bytes, 0);
}

#[test]
fn insufficient_releasable_ram_is_exhaustion_rather_than_a_retirement_wait() {
    let machine = Machine::new(3 * PAGE_SIZE).unwrap();
    let live_account = machine.account(3 * PAGE_SIZE).unwrap();
    let retained_account = machine.account(PAGE_SIZE).unwrap();
    let caller = machine.account(3 * PAGE_SIZE).unwrap();
    let _live = live_account.allocate(2 * PAGE_SIZE).unwrap();
    let old = retained_account.allocate(PAGE_SIZE).unwrap();
    let _pin = old.pin();
    drop(old);
    let wake = Waker::noop();
    let mut future = Box::pin(caller.allocate_async(2 * PAGE_SIZE));
    assert!(
        matches!(future.as_mut().poll(&mut Context::from_waker(wake)), Poll::Ready(Err(error)) if matches!(*error, MemoryError::PhysicalExhaustion { .. }))
    );
    assert_eq!(machine.pending_allocation_count(), 0);
}

#[test]
fn contiguous_pressure_checks_live_fragmentation_before_waiting_for_native_pins() {
    let machine = Machine::new(8 * PAGE_SIZE).unwrap();
    let account = machine.account(8 * PAGE_SIZE).unwrap();
    let caller = machine.account(8 * PAGE_SIZE).unwrap();
    let mut owners: Vec<_> = (0..8)
        .map(|_| Some(account.allocate(PAGE_SIZE).unwrap()))
        .collect();
    let mut retained = Vec::new();
    for index in [0, 2, 4, 6] {
        retained.push(Some(owners[index].as_ref().unwrap().pin()));
        owners[index] = None;
    }
    let wake = Waker::noop();
    let mut blocked = Box::pin(caller.allocate_contiguous_async(2 * PAGE_SIZE, PAGE_SIZE));
    assert!(
        matches!(blocked.as_mut().poll(&mut Context::from_waker(wake)), Poll::Ready(Err(error)) if matches!(*error, MemoryError::PhysicalExhaustion { .. }))
    );
    assert_eq!(machine.pending_allocation_count(), 0);
    owners[3] = None;
    let mut possible = Box::pin(caller.allocate_contiguous_async(2 * PAGE_SIZE, PAGE_SIZE));
    assert!(
        possible
            .as_mut()
            .poll(&mut Context::from_waker(wake))
            .is_pending()
    );
    retained[1] = None;
    let Poll::Ready(Ok(allocation)) = possible.as_mut().poll(&mut Context::from_waker(wake)) else {
        panic!("retired neighboring pages must satisfy the contiguous request");
    };
    assert_eq!(
        allocation
            .pin()
            .physical_extents(0, allocation.bytes())
            .collect::<Vec<_>>(),
        [Extent {
            start: ArenaOffset::new(2 * PAGE_SIZE),
            bytes: 2 * PAGE_SIZE
        }]
    );
}
