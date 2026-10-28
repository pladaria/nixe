//! Per-thread property ownership, with the kernel's existing inheritance nodes.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/k_light_lock.cpp
use super::*;
use std::{
    future::Future,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll, Waker},
};

/// Acquires the target's activity/core-mask/context lock. Dropping a blocked
/// acquisition removes its actual inheritance node, including a won handoff.
pub struct ThreadPropertyLock {
    scheduler: Arc<SchedulerState>,
    caller: GuestThreadId,
    target: GuestThreadId,
    phase: LockPhase,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum LockPhase {
    New,
    Waiting,
    Acquired,
}

#[derive(Debug)]
pub struct ThreadPropertyGuard {
    scheduler: Arc<SchedulerState>,
    caller: GuestThreadId,
    target: GuestThreadId,
}

impl SchedulerState {
    pub fn try_thread_property(
        self: &Arc<Self>,
        caller: GuestThreadId,
        target: GuestThreadId,
    ) -> Result<Option<ThreadPropertyGuard>, SchedulerError> {
        self.edit(|state| {
            let caller_index = state.slots.index(caller)?;
            let target_index = state.slots.index(target)?;
            if !state.acquire_property(caller_index, target_index)? {
                return Ok(None);
            }
            Ok(Some(ThreadPropertyGuard {
                scheduler: self.clone(),
                caller,
                target,
            }))
        })
    }
    pub fn lock_thread_property(
        self: &Arc<Self>,
        caller: GuestThreadId,
        target: GuestThreadId,
    ) -> ThreadPropertyLock {
        ThreadPropertyLock {
            scheduler: self.clone(),
            caller,
            target,
            phase: LockPhase::New,
        }
    }

    /// Cancellation/termination unwinds the owned kernel operation before its
    /// ordinary guest-address held locks and saved thread state are retired.
    pub fn cancel_thread_property(&self, caller: GuestThreadId) {
        self.end_thread_property(caller, true);
    }
    fn end_thread_property(&self, caller: GuestThreadId, notify: bool) {
        self.edit(|state| {
            if let Ok(caller) = state.slots.index(caller) {
                state.cancel_property(caller, &self.profile, notify);
            }
            Ok(())
        })
        .expect("property cancellation cannot fail");
    }
}

impl Future for ThreadPropertyLock {
    type Output = Result<ThreadPropertyGuard, SchedulerError>;
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let result = self.scheduler.edit(|state| {
            let caller = state.slots.index(self.caller)?;
            let target = state.slots.index(self.target)?;
            let owned = self.phase == LockPhase::Waiting
                && state.slots[target].property_owner == Some(caller);
            if owned || state.acquire_property(caller, target)? {
                // A suspended kernel continuation retains a won handoff until
                // resume or a real kernel waiter defers suspension.
                if state.slots[caller].suspended() {
                    state.register_kernel_waker(caller, cx.waker());
                    return Ok(false);
                }
                state.take_kernel_waker(caller, false);
                return Ok(true);
            }
            let owner = state.slots[target]
                .property_owner
                .expect("contended property lock");
            state.add_waiter(
                state.slots[target].config.process,
                LockKey::ThreadProperty(self.target),
                owner,
                caller,
                &self.scheduler.profile,
                Some(self.caller),
            )?;
            state.register_kernel_waker(caller, cx.waker());
            Ok(false)
        });
        match result {
            Ok(false) => {
                self.phase = LockPhase::Waiting;
                Poll::Pending
            }
            Ok(true) => {
                self.phase = LockPhase::Acquired;
                Poll::Ready(Ok(ThreadPropertyGuard {
                    scheduler: self.scheduler.clone(),
                    caller: self.caller,
                    target: self.target,
                }))
            }
            Err(error) => Poll::Ready(Err(error)),
        }
    }
}
impl Drop for ThreadPropertyLock {
    fn drop(&mut self) {
        if self.phase == LockPhase::Waiting {
            self.scheduler.end_thread_property(self.caller, false);
        }
    }
}

impl ThreadPropertyGuard {
    pub fn is_quiescent(&self, allowed: Option<CoreSet>) -> Result<bool, SchedulerError> {
        self.scheduler.edit(|state| {
            let target = state.slots.index(self.target)?;
            Ok(state
                .property_wait(target, allowed, &self.scheduler.profile, true)
                .is_none())
        })
    }
    /// A pause waits for both unpinning and the source's real register release.
    /// A core-mask change waits for a current owner outside the allowed mask,
    /// including a pinned core temporarily stopped at an ordinary SVC boundary.
    /// Once registered, a pin dependency ends only on unpin or termination.
    pub async fn wait_quiescent(&self, allowed: Option<CoreSet>) -> Result<(), SchedulerError> {
        let mut wait_for_pin = true;
        std::future::poll_fn(|cx| {
            self.scheduler
                .edit(|state| {
                    let caller = state.slots.index(self.caller)?;
                    let target = state.slots.index(self.target)?;
                    if state.slots[caller].waiting_on.is_none()
                        && let Some(membership) = state.property_wait(
                            target,
                            allowed,
                            &self.scheduler.profile,
                            wait_for_pin,
                        )
                    {
                        if matches!(membership, waits::Membership::Pinned(_)) {
                            // Continue after this unpin, as the blocked reference
                            // call does, even if the target is pinned again later.
                            wait_for_pin = false;
                        }
                        state.slots[caller].waiting_on = Some(membership);
                        state.insert_wait_node(caller, membership);
                    }
                    if state.slots[caller].waiting_on.is_none() && !state.slots[caller].suspended()
                    {
                        state.take_kernel_waker(caller, false);
                        return Ok(Poll::Ready(()));
                    }
                    state.register_kernel_waker(caller, cx.waker());
                    Ok(Poll::Pending)
                })
                .map_or_else(|error| Poll::Ready(Err(error)), |poll| poll.map(Ok))
        })
        .await
    }
}
impl Drop for ThreadPropertyGuard {
    fn drop(&mut self) {
        self.scheduler.end_thread_property(self.caller, false);
    }
}

impl State {
    fn acquire_property(&mut self, caller: Index, target: Index) -> Result<bool, SchedulerError> {
        if matches!(
            self.slots[caller].lifecycle,
            ThreadLifecycle::Exited | ThreadLifecycle::Faulted
        ) {
            return Err(self.invalid(caller));
        }
        if self.slots[target].property_owner.is_some() {
            return Ok(false);
        }
        debug_assert!(self.slots[caller].property_target.is_none());
        self.slots[target].property_owner = Some(caller);
        self.slots[caller].property_target = Some(target);
        Ok(true)
    }
    fn property_wait(
        &self,
        target: Index,
        allowed: Option<CoreSet>,
        profile: &MachineSchedulerProfile,
        wait_for_pin: bool,
    ) -> Option<waits::Membership> {
        let thread = &self.slots[target];
        if matches!(
            thread.lifecycle,
            ThreadLifecycle::Exited | ThreadLifecycle::Faulted
        ) {
            return None;
        }
        let busy = match allowed {
            Some(mask) => {
                thread.lease.is_some_and(|lease| !mask.contains(lease.vcpu))
                    || wait_for_pin
                        && thread.pinned_core.is_some_and(|core| {
                            self.cores[core].selected == Some(target)
                                && !mask.contains(profile.vcpus()[core].id())
                        })
            }
            None => wait_for_pin && thread.pinned_core.is_some() || thread.lease.is_some(),
        };
        busy.then_some(if wait_for_pin && thread.pinned_core.is_some() {
            waits::Membership::Pinned(target)
        } else {
            waits::Membership::ContextRelease(target)
        })
    }
    fn register_kernel_waker(&mut self, caller: Index, waker: &Waker) {
        if self.slots[caller]
            .kernel_waker
            .as_ref()
            .is_none_or(|old| !old.will_wake(waker))
            && let Some(old) = self.slots[caller].kernel_waker.replace(waker.clone())
        {
            self.deferred_wakers.push((old, false));
        }
    }
    fn take_kernel_waker(&mut self, caller: Index, notify: bool) {
        if let Some(waker) = self.slots[caller].kernel_waker.take() {
            self.deferred_wakers.push((waker, notify));
        }
    }
    pub(super) fn wake_kernel_operation(&mut self, caller: Index) {
        if self.slots[caller].waiting_on.is_none() && !self.slots[caller].suspended() {
            self.take_kernel_waker(caller, true);
        }
    }
    pub(super) fn notify_context_release(
        &mut self,
        target: Index,
        profile: &MachineSchedulerProfile,
    ) {
        // Only the property owner can await this target's saved registers.
        if let Some(caller) = self.slots[target].property_owner
            && self.slots[caller].waiting_on == Some(waits::Membership::ContextRelease(target))
        {
            self.remove_waiter(caller, profile, None);
            self.wake_kernel_operation(caller);
        }
    }
    pub(super) fn notify_pin_waiters(&mut self, target: Index, profile: &MachineSchedulerProfile) {
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/kernel/k_thread.cpp#L545-L548
        while let Some(caller) = self.slots[target].pin_waiters.first {
            self.remove_waiter(caller, profile, None);
            self.wake_kernel_operation(caller);
        }
    }
    pub(super) fn cancel_property(
        &mut self,
        caller: Index,
        profile: &MachineSchedulerProfile,
        notify: bool,
    ) {
        if self.slots[caller].kernel_waker.is_some() {
            self.remove_waiter(caller, profile, None);
            self.take_kernel_waker(caller, notify);
        }
        let Some(target) = self.slots[caller].property_target.take() else {
            return;
        };
        debug_assert_eq!(self.slots[target].property_owner, Some(caller));
        self.slots[target].property_owner = None;
        let process = self.slots[target].config.process;
        let key = LockKey::ThreadProperty(self.slots[target].id);
        if let Some(next) = self.release_lock(process, key, caller, profile).next {
            let next = self.slots.index(next).expect("live kernel waiter");
            self.slots[target].property_owner = Some(next);
            self.slots[next].property_target = Some(target);
            self.wake_kernel_operation(next);
        }
    }
}
