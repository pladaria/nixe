//! Owned, FIFO heap mutations. A physical-memory wait holds no kernel lock.
use super::*;
use std::{
    future::Future,
    pin::Pin,
    task::{Context, Poll, Waker},
};

#[derive(Default)]
pub(super) struct HeapRequests {
    next: u64,
    active: Option<u64>,
    waiters: BTreeMap<u64, Arc<Waker>>,
}
impl HeapRequests {
    pub(super) fn wakers(&self) -> Vec<Arc<Waker>> {
        self.waiters.values().cloned().collect()
    }
}
enum Phase {
    Waiting,
    Allocating {
        old: u64,
        allocation: AllocationFuture,
    },
    Done,
}
/// The kernel retains this operation when an allocating SVC releases its core.
/// Concurrent heap requests consume the current heap only on acquiring their
/// turn; cancellation never replays an allocation or a mapping publication.
pub struct HeapResize {
    memory: Arc<ProcessMemory>,
    bytes: u64,
    ticket: Option<u64>,
    phase: Phase,
}
impl ProcessMemory {
    pub fn resize_heap(self: &Arc<Self>, bytes: u64) -> HeapResize {
        HeapResize {
            memory: Arc::clone(self),
            bytes,
            ticket: None,
            phase: Phase::Waiting,
        }
    }
}
impl HeapResize {
    fn claim(&mut self, context: &Context<'_>) -> Result<bool, MappingError> {
        let incoming = context.waker().clone();
        let mut requests = self.memory.heap_requests.lock().unwrap();
        let ticket = match self.ticket {
            Some(ticket) => ticket,
            None => {
                requests.next = requests
                    .next
                    .checked_add(1)
                    .ok_or(MemoryError::Invalid("heap request identity exhausted"))?;
                self.ticket = Some(requests.next);
                requests.next
            }
        };
        if requests.active.is_none()
            && requests
                .waiters
                .first_key_value()
                .is_none_or(|(&first, _)| first == ticket)
        {
            requests.active = Some(ticket);
        }
        let acquired = requests.active == Some(ticket);
        // As KLightLock's uncontended path, an owner which can run needs no
        // waiter node. A physical allocation wait registers its observer below.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/k_light_lock.cpp#L41-L53
        let previous = if acquired {
            None
        } else {
            requests.waiters.insert(ticket, Arc::new(incoming))
        };
        drop(requests);
        drop(previous);
        Ok(acquired)
    }
    fn release(&mut self) {
        let Some(ticket) = self.ticket.take() else {
            return;
        };
        let mut requests = self.memory.heap_requests.lock().unwrap();
        let removed = requests.waiters.remove(&ticket);
        if requests.active == Some(ticket) {
            requests.active = None;
        }
        let next = if requests.active.is_none() {
            requests
                .waiters
                .first_key_value()
                .map(|(_, wake)| Arc::clone(wake))
        } else {
            None
        };
        drop(requests);
        drop(removed);
        if let Some(wake) = next {
            wake.wake_by_ref();
        }
    }
    fn finish(
        &mut self,
        result: Result<GuestVirtualAddress, MappingError>,
    ) -> Poll<Result<GuestVirtualAddress, MappingError>> {
        self.phase = Phase::Done;
        self.release();
        Poll::Ready(result)
    }
    fn pending(&mut self) -> Poll<Result<GuestVirtualAddress, MappingError>> {
        // close() may have notified the previous executor before this poll
        // registered its current waker. Recheck after every actual registration.
        if self.memory.closed.load(Ordering::Acquire) {
            self.finish(Err(MemoryError::Invalid("process memory is closed").into()))
        } else {
            Poll::Pending
        }
    }
}
impl Future for HeapResize {
    type Output = Result<GuestVirtualAddress, MappingError>;
    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        assert!(
            !matches!(this.phase, Phase::Done),
            "completed heap operation was polled again"
        );
        if this.memory.closed.load(Ordering::Acquire) {
            return this.finish(Err(MemoryError::Invalid("process memory is closed").into()));
        }
        if !this.bytes.is_multiple_of(PAGE_SIZE) || this.bytes > this.memory.heap_region.size() {
            return this.finish(Err(MappingError::InvalidRange));
        }
        match this.claim(context) {
            Ok(false) => return this.pending(),
            Err(error) => return this.finish(Err(error)),
            Ok(true) => {}
        }
        let base = this.memory.heap_region.base();
        let mapping = Mapping {
            kind: MemoryKind::Heap,
            permissions: MemoryPermissions::READ_WRITE,
            uncached: false,
            locked: false,
            permission_locked: false,
        };
        if matches!(this.phase, Phase::Waiting) {
            let old = this.memory.heap_size();
            if this.bytes == old {
                return this.finish(Ok(base));
            }
            if this.bytes < old {
                let result = this.memory.unmap(
                    base.checked_add(this.bytes).unwrap(),
                    old - this.bytes,
                    mapping,
                );
                if result.is_ok() {
                    this.memory.heap_bytes.store(this.bytes, Ordering::Release);
                }
                return this.finish(result.map(|()| base));
            }
            this.phase = Phase::Allocating {
                old,
                allocation: this.memory.account.allocate_async(this.bytes - old),
            };
        }
        let Phase::Allocating { old, allocation } = &mut this.phase else {
            unreachable!()
        };
        match Pin::new(allocation).poll(context) {
            Poll::Pending => {
                let incoming = Arc::new(context.waker().clone());
                let previous = this
                    .memory
                    .heap_requests
                    .lock()
                    .unwrap()
                    .waiters
                    .insert(this.ticket.expect("active heap owner"), incoming);
                drop(previous);
                this.pending()
            }
            Poll::Ready(Err(error)) => this.finish(Err(MappingError::Memory(error))),
            Poll::Ready(Ok(allocation)) => {
                let result = this.memory.map(
                    base.checked_add(*old).unwrap(),
                    allocation,
                    0,
                    this.bytes - *old,
                    mapping,
                );
                if result.is_ok() {
                    this.memory.heap_bytes.store(this.bytes, Ordering::Release);
                }
                this.finish(result.map(|()| base))
            }
        }
    }
}
impl Drop for HeapResize {
    fn drop(&mut self) {
        self.release();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{sync::atomic::AtomicUsize, task::Wake};
    #[derive(Default)]
    struct WakeCount(AtomicUsize);
    impl Wake for WakeCount {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
        fn wake_by_ref(self: &Arc<Self>) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }
    fn poll(
        request: &mut HeapResize,
        wake: &Arc<WakeCount>,
    ) -> Poll<Result<GuestVirtualAddress, MappingError>> {
        Pin::new(request).poll(&mut Context::from_waker(&Waker::from(Arc::clone(wake))))
    }
    fn fixture() -> Arc<ProcessMemory> {
        let machine = nixe_memory::Machine::new(4 * PAGE_SIZE).unwrap();
        Arc::new(
            ProcessMemory::new(
                machine.account(4 * PAGE_SIZE).unwrap(),
                machine.account(4 * PAGE_SIZE).unwrap(),
                AddressWidth::Bits32,
                crate::ProcessVirtualRegion::new(
                    GuestVirtualAddress::new(16 * PAGE_SIZE),
                    4 * PAGE_SIZE,
                ),
                0,
            )
            .unwrap(),
        )
    }
    fn pending_growth(
        memory: &Arc<ProcessMemory>,
        wake: &Arc<WakeCount>,
    ) -> (HeapResize, Vec<AllocationRange>, Allocation) {
        assert!(matches!(
            poll(&mut memory.resize_heap(2 * PAGE_SIZE), wake),
            Poll::Ready(Ok(_))
        ));
        let retained = memory
            .space
            .capture(
                memory.heap_region.base().checked_add(PAGE_SIZE).unwrap(),
                8,
                MemoryPermissions::READ,
            )
            .unwrap();
        assert!(matches!(
            poll(&mut memory.resize_heap(PAGE_SIZE), wake),
            Poll::Ready(Ok(_))
        ));
        let outside = memory.account.allocate(2 * PAGE_SIZE).unwrap();
        let mut growth = memory.resize_heap(2 * PAGE_SIZE);
        assert!(poll(&mut growth, wake).is_pending());
        assert_eq!(memory.account.used_bytes(), 3 * PAGE_SIZE);
        assert_eq!(
            memory.account.machine().statistics().allocated_bytes,
            4 * PAGE_SIZE
        );
        (growth, retained, outside)
    }
    #[test]
    fn queued_heap_mutations_use_the_committed_predecessor_and_wait_only_for_retained_ram() {
        let memory = fixture();
        let wake = Arc::new(WakeCount::default());
        let (mut first, retained, outside) = pending_growth(&memory, &wake);
        let next_wake = Arc::new(WakeCount::default());
        let mut second = memory.resize_heap(0);
        assert!(poll(&mut second, &next_wake).is_pending());
        assert_eq!(memory.heap_size(), PAGE_SIZE);
        drop(retained);
        assert!(wake.0.load(Ordering::Relaxed) != 0);
        assert!(matches!(poll(&mut first, &wake), Poll::Ready(Ok(_))));
        assert_eq!(memory.heap_size(), 2 * PAGE_SIZE);
        assert!(next_wake.0.load(Ordering::Relaxed) != 0);
        assert!(matches!(poll(&mut second, &next_wake), Poll::Ready(Ok(_))));
        assert_eq!(memory.heap_size(), 0);
        assert_eq!(memory.account.used_bytes(), 2 * PAGE_SIZE);
        drop(outside);
        memory.close().unwrap();
        assert_eq!(memory.account.machine().statistics().allocated_bytes, 0);
    }
    #[test]
    fn cancellation_releases_both_the_active_resize_and_an_unclaimed_queue_head() {
        let memory = fixture();
        let wake = Arc::new(WakeCount::default());
        let (first, retained, outside) = pending_growth(&memory, &wake);
        let mut second = memory.resize_heap(3 * PAGE_SIZE);
        let mut third = memory.resize_heap(0);
        let second_wake = Arc::new(WakeCount::default());
        let third_wake = Arc::new(WakeCount::default());
        assert!(poll(&mut second, &second_wake).is_pending());
        assert!(poll(&mut third, &third_wake).is_pending());
        drop(first);
        assert!(second_wake.0.load(Ordering::Relaxed) != 0);
        drop(second);
        assert!(third_wake.0.load(Ordering::Relaxed) != 0);
        assert!(matches!(poll(&mut third, &third_wake), Poll::Ready(Ok(_))));
        assert_eq!(memory.heap_size(), 0);
        assert_eq!(memory.account.machine().pending_allocation_count(), 0);
        drop(retained);
        drop(outside);
        assert_eq!(memory.account.machine().statistics().allocated_bytes, 0);
    }
    #[test]
    fn allocation_failure_keeps_the_original_device_error_and_does_not_publish_growth() {
        let memory = fixture();
        let wake = Arc::new(WakeCount::default());
        let (mut request, retained, outside) = pending_growth(&memory, &wake);
        let original = Arc::new(MemoryError::Execution("native submission failed".into()));
        memory
            .account
            .machine()
            .fail_pending_allocations(Arc::clone(&original));
        let Poll::Ready(Err(MappingError::Memory(error))) = poll(&mut request, &wake) else {
            panic!("pending allocation must report the owner failure")
        };
        assert!(Arc::ptr_eq(&error, &original));
        assert_eq!(memory.heap_size(), PAGE_SIZE);
        assert_eq!(memory.account.used_bytes(), 3 * PAGE_SIZE);
        memory.close().unwrap();
        drop(retained);
        drop(outside);
        assert_eq!(memory.account.machine().statistics().allocated_bytes, 0);
    }

    #[test]
    fn closing_memory_wakes_the_active_allocation_wait_and_queued_heap_mutations() {
        let memory = fixture();
        let owner_wake = Arc::new(WakeCount::default());
        let (mut owner, retained, outside) = pending_growth(&memory, &owner_wake);
        let queued_wake = Arc::new(WakeCount::default());
        let mut queued = memory.resize_heap(0);
        assert!(poll(&mut queued, &queued_wake).is_pending());
        memory.close().unwrap();
        assert!(owner_wake.0.load(Ordering::Relaxed) != 0);
        assert!(queued_wake.0.load(Ordering::Relaxed) != 0);
        assert!(matches!(poll(&mut owner, &owner_wake), Poll::Ready(Err(_))));
        assert!(matches!(
            poll(&mut queued, &queued_wake),
            Poll::Ready(Err(_))
        ));
        drop(retained);
        drop(outside);
        assert_eq!(memory.account.machine().pending_allocation_count(), 0);
        assert_eq!(memory.account.machine().statistics().allocated_bytes, 0);
    }

    #[test]
    fn memory_close_during_waker_replacement_cannot_leave_the_new_executor_asleep() {
        use std::mem::ManuallyDrop;
        use std::sync::{Weak, atomic::AtomicBool};
        use std::task::{RawWaker, RawWakerVTable};

        struct CloseOnClone {
            memory: Weak<ProcessMemory>,
            closed: AtomicBool,
            wakes: Arc<WakeCount>,
        }
        unsafe fn clone(data: *const ()) -> RawWaker {
            // Every raw waker owns one Arc. Borrowing a callback keeps its
            // original reference alive; clone returns one additional reference.
            let state = ManuallyDrop::new(unsafe { Arc::from_raw(data.cast::<CloseOnClone>()) });
            if !state.closed.swap(true, Ordering::AcqRel) {
                state.memory.upgrade().unwrap().close().unwrap();
            }
            RawWaker::new(Arc::into_raw(Arc::clone(&state)).cast(), &VTABLE)
        }
        unsafe fn wake(data: *const ()) {
            let state = unsafe { Arc::from_raw(data.cast::<CloseOnClone>()) };
            state.wakes.0.fetch_add(1, Ordering::Relaxed);
        }
        unsafe fn wake_by_ref(data: *const ()) {
            let state = ManuallyDrop::new(unsafe { Arc::from_raw(data.cast::<CloseOnClone>()) });
            state.wakes.0.fetch_add(1, Ordering::Relaxed);
        }
        unsafe fn release(data: *const ()) {
            drop(unsafe { Arc::from_raw(data.cast::<CloseOnClone>()) });
        }
        const VTABLE: RawWakerVTable = RawWakerVTable::new(clone, wake, wake_by_ref, release);

        let memory = fixture();
        let old_wake = Arc::new(WakeCount::default());
        let (mut owner, retained, outside) = pending_growth(&memory, &old_wake);
        // Keep all RAM resident after address-space closure, so physical
        // retirement cannot accidentally repair a missing close notification.
        let live = memory
            .space()
            .capture(
                memory.heap_region.base(),
                PAGE_SIZE,
                MemoryPermissions::READ,
            )
            .unwrap();
        let current_wake = Arc::new(WakeCount::default());
        let state = Arc::new(CloseOnClone {
            memory: Arc::downgrade(&memory),
            closed: AtomicBool::new(false),
            wakes: current_wake.clone(),
        });
        // SAFETY: VTABLE preserves Arc ownership and all callback state is Sync.
        let waker = unsafe { Waker::from_raw(RawWaker::new(Arc::into_raw(state).cast(), &VTABLE)) };
        let result = Pin::new(&mut owner).poll(&mut Context::from_waker(&waker));
        assert!(
            matches!(result, Poll::Ready(Err(_))) || current_wake.0.load(Ordering::Relaxed) != 0,
            "a close must finish or notify the current executor"
        );
        drop(owner);
        drop(waker);
        drop(live);
        drop(retained);
        drop(outside);
        assert_eq!(memory.account.machine().pending_allocation_count(), 0);
        assert_eq!(memory.account.machine().statistics().allocated_bytes, 0);
    }
}
