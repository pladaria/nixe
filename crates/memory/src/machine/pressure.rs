//! Physical pressure suspends the actual allocation, retaining no guest stack.
use super::*;
use std::{
    future::Future,
    pin::Pin,
    sync::OnceLock,
    task::{Context, Poll, Waker},
};
#[cfg(test)]
mod tests;

#[derive(Default)]
pub(super) struct AllocationPressure {
    revision: AtomicU64,
    has_waiters: AtomicBool,
    registry: Mutex<Waiters>,
    failure: OnceLock<Arc<MemoryError>>,
}
#[derive(Default)]
struct Waiters {
    next: u64,
    waiters: HashMap<u64, Waker>,
}
impl AllocationPressure {
    pub(super) fn retired(&self) {
        self.revision
            .fetch_update(Ordering::Release, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .expect("physical retirement revision exhausted");
        if !self.has_waiters.load(Ordering::Acquire) {
            return;
        }
        let waiters = {
            let mut registry = self.registry.lock().unwrap();
            self.has_waiters.store(false, Ordering::Release);
            std::mem::take(&mut registry.waiters)
        };
        for (_, wake) in waiters {
            wake.wake();
        }
    }
    fn remove(&self, id: u64) {
        let old = {
            let mut registry = self.registry.lock().unwrap();
            let old = registry.waiters.remove(&id);
            if registry.waiters.is_empty() {
                self.has_waiters.store(false, Ordering::Release);
            }
            old
        };
        drop(old);
    }
}

/// A loader/kernel continuation owns this future. Polling never blocks an OS
/// worker; physical retirement or terminal failure wakes a pending allocation. The
/// owner is asked to advance native uses outside all memory locks.
pub struct AllocationFuture {
    account: MemoryAccount,
    bytes: u64,
    alignment: Option<u64>,
    registration: Option<u64>,
    notified: Option<u64>,
    finished: bool,
}
impl MemoryAccount {
    pub fn allocate_async(&self, bytes: u64) -> AllocationFuture {
        self.allocation_future(bytes, None)
    }
    pub fn allocate_contiguous_async(&self, bytes: u64, alignment: u64) -> AllocationFuture {
        self.allocation_future(bytes, Some(alignment))
    }
    fn allocation_future(&self, bytes: u64, alignment: Option<u64>) -> AllocationFuture {
        AllocationFuture {
            account: self.clone(),
            bytes,
            alignment,
            registration: None,
            notified: None,
            finished: false,
        }
    }
}
impl Machine {
    /// Terminal graphics failure cancels allocations waiting on native pins.
    /// Preserve the owner's original error object through every continuation.
    pub fn fail_pending_allocations(&self, error: Arc<MemoryError>) {
        let pressure = &self.0.pressure;
        let _ = pressure.failure.set(error);
        let waiters = {
            let mut registry = pressure.registry.lock().unwrap();
            pressure.has_waiters.store(false, Ordering::Release);
            std::mem::take(&mut registry.waiters)
        };
        for (_, wake) in waiters {
            wake.wake();
        }
    }
    pub fn pending_allocation_count(&self) -> usize {
        self.0.pressure.registry.lock().unwrap().waiters.len()
    }
}
impl Future for AllocationFuture {
    type Output = Result<Allocation, Arc<MemoryError>>;
    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        assert!(
            !this.finished,
            "allocation continuation polled after completion"
        );
        loop {
            let machine = &this.account.machine;
            let pressure = &machine.0.pressure;
            if let Some(error) = pressure.failure.get() {
                this.finished = true;
                if let Some(id) = this.registration.take() {
                    pressure.remove(id);
                }
                return Poll::Ready(Err(Arc::clone(error)));
            }
            let revision = pressure.revision.load(Ordering::Acquire);
            let result = match this.alignment {
                Some(alignment) => this.account.allocate_contiguous(this.bytes, alignment),
                None => this.account.allocate(this.bytes),
            };
            if !matches!(result, Err(MemoryError::RetainedAllocations { .. })) {
                this.finished = true;
                if let Some(id) = this.registration.take() {
                    pressure.remove(id);
                }
                return Poll::Ready(result.map_err(Arc::new));
            }
            // The revision closes retirement between the failed allocation and
            // registration. Clone/drop user wakers outside the registry lock.
            let incoming = context.waker().clone();
            let old = {
                let mut registry = pressure.registry.lock().unwrap();
                if pressure.failure.get().is_some()
                    || pressure.revision.load(Ordering::Acquire) != revision
                {
                    continue;
                }
                let id = match this.registration {
                    Some(id) => id,
                    None => {
                        let Some(id) = registry.next.checked_add(1) else {
                            this.finished = true;
                            return Poll::Ready(Err(Arc::new(MemoryError::Invalid(
                                "allocation continuation identity exhausted",
                            ))));
                        };
                        registry.next = id;
                        this.registration = Some(id);
                        id
                    }
                };
                pressure.has_waiters.store(true, Ordering::Release);
                if registry
                    .waiters
                    .get(&id)
                    .is_some_and(|old| old.will_wake(&incoming))
                {
                    None
                } else {
                    registry.waiters.insert(id, incoming)
                }
            };
            drop(old);
            if this.notified != Some(revision) {
                this.notified = Some(revision);
                let endpoint = machine
                    .device_endpoint()
                    .lock()
                    .unwrap()
                    .as_ref()
                    .and_then(Weak::upgrade);
                if let Some(endpoint) = endpoint {
                    endpoint.allocation_pressure(machine.identity(), this.bytes);
                }
            }
            return Poll::Pending;
        }
    }
}
impl Drop for AllocationFuture {
    fn drop(&mut self) {
        if let Some(id) = self.registration.take() {
            self.account.machine.0.pressure.remove(id);
        }
    }
}
