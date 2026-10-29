//! A DSP block or an IPC update owns the renderer state; device waits hold no
//! kernel, service or renderer mutex. Other callers await the returned state.
use super::update::RendererUpdateState;
use nixe_runtime::{EventObject, ReadableEventObject, WritableEventObject};
use std::{
    ops::{Deref, DerefMut},
    sync::{Arc, Condvar, Mutex},
};

#[derive(Debug)]
pub(crate) struct UpdateSlot {
    value: Mutex<Option<RendererUpdateState>>,
    available: Condvar,
    ready: ReadableEventObject,
    signal: WritableEventObject,
}
impl UpdateSlot {
    pub fn new(kernel: &Arc<nixe_scheduler::SchedulerState>) -> Self {
        let (signal, ready) = EventObject::create_auto_reset_pair(kernel);
        signal.signal();
        Self {
            value: Mutex::new(Some(RendererUpdateState::default())),
            available: Condvar::new(),
            ready,
            signal,
        }
    }
}
impl UpdateSlot {
    pub fn try_take(self: &Arc<Self>) -> Option<UpdateLease> {
        let value = {
            let mut state = self.value.lock().unwrap();
            self.ready.clear();
            state.take()
        }?;
        Some(UpdateLease {
            owner: self.clone(),
            value: Some(value),
        })
    }
    pub async fn take(self: &Arc<Self>) -> UpdateLease {
        loop {
            if let Some(state) = self.try_take() {
                return state;
            }
            self.ready.wait_async().await;
        }
    }
    /// The dedicated DSP worker sleeps on the actual state mutex; guest calls use take().
    pub fn take_blocking(self: &Arc<Self>) -> UpdateLease {
        let mut state = self.value.lock().unwrap();
        while state.is_none() {
            state = self.available.wait(state).unwrap();
        }
        self.ready.clear();
        UpdateLease {
            owner: self.clone(),
            value: state.take(),
        }
    }
}
pub(crate) struct UpdateLease {
    owner: Arc<UpdateSlot>,
    value: Option<RendererUpdateState>,
}
impl Deref for UpdateLease {
    type Target = RendererUpdateState;
    fn deref(&self) -> &Self::Target {
        self.value.as_ref().unwrap()
    }
}
impl DerefMut for UpdateLease {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.value.as_mut().unwrap()
    }
}
impl Drop for UpdateLease {
    fn drop(&mut self) {
        let value = self.value.take().unwrap();
        let notifications = {
            let mut state = self.owner.value.lock().unwrap();
            let old = state.replace(value);
            assert!(old.is_none(), "renderer state has one executing owner");
            self.owner.signal.publish_readiness(true)
        };
        self.owner.available.notify_one();
        notifications.notify();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        future::Future,
        sync::atomic::{AtomicUsize, Ordering},
        task::{Context, Poll, Wake, Waker},
    };

    struct Notifications(AtomicUsize);
    impl Wake for Notifications {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
        fn wake_by_ref(self: &Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    #[test]
    fn returned_state_wakes_async_callers_and_cancelled_callers_do_not_own_it() {
        let slot = Arc::new(UpdateSlot::new(&crate::test_memory::kernel()));
        let mut lease = slot.try_take().unwrap();
        lease.frames = 42;
        let notifications = Arc::new(Notifications(AtomicUsize::new(0)));
        let waker = Waker::from(notifications.clone());
        let mut context = Context::from_waker(&waker);
        let mut next = Box::pin(slot.take());
        let mut cancelled = Box::pin(slot.take());
        assert!(next.as_mut().poll(&mut context).is_pending());
        assert!(cancelled.as_mut().poll(&mut context).is_pending());
        drop(cancelled);
        assert!(slot.value.try_lock().is_ok());
        drop(lease);
        assert!(notifications.0.load(Ordering::SeqCst) > 0);
        let Poll::Ready(lease) = next.as_mut().poll(&mut context) else {
            panic!("returned DSP state was lost");
        };
        assert_eq!(lease.frames, 42);
        assert!(slot.try_take().is_none());
        drop(lease);
        assert_eq!(slot.try_take().unwrap().frames, 42);
    }
}
