//! One retained wait generation and one terminal result for all producers.
use super::*;
use std::sync::{
    Arc, OnceLock, Weak,
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
};
use std::task::Poll;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WaitOutcome {
    Ready,
    Signalled(u32),
    TimedOut,
    Cancelled,
    Terminated,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WaitPhase {
    Registering,
    Waiting,
    Resolved(WaitOutcome),
    Cancelled(WaitOutcome),
}

/// The scheduler owns state changes. The atomic word lets the timer discard
/// stale heap entries without nesting the timer and scheduler locks.
pub struct KernelWait {
    token: WakeToken,
    pub(super) kernel: Weak<SchedulerState>,
    phase: AtomicU64,
    timer_count: OnceLock<Arc<AtomicUsize>>,
    effects: AtomicUsize,
    deferred_timeout: AtomicBool,
}

const TIMER_ARMED: u64 = 1 << 63;

impl std::fmt::Debug for KernelWait {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("KernelWait")
            .field("token", &self.token)
            .field("phase", &self.phase())
            .finish()
    }
}

impl KernelWait {
    pub(super) fn new(token: WakeToken, kernel: Weak<SchedulerState>) -> Self {
        Self {
            token,
            kernel,
            phase: AtomicU64::new(0),
            timer_count: OnceLock::new(),
            effects: AtomicUsize::new(0),
            deferred_timeout: AtomicBool::new(false),
        }
    }

    pub fn token(&self) -> WakeToken {
        self.token
    }

    pub fn outcome(&self) -> Option<WaitOutcome> {
        match self.phase.load(Ordering::Acquire) & !TIMER_ARMED {
            0 | 1 => None,
            2 => Some(WaitOutcome::Ready),
            3 => Some(WaitOutcome::TimedOut),
            4 => Some(WaitOutcome::Cancelled),
            5 => Some(WaitOutcome::Terminated),
            value => Some(WaitOutcome::Signalled((value - 0x100) as u32)),
        }
    }

    pub fn phase(&self) -> WaitPhase {
        match self.phase.load(Ordering::Acquire) & !TIMER_ARMED {
            0 => WaitPhase::Registering,
            1 => WaitPhase::Waiting,
            4 => WaitPhase::Cancelled(WaitOutcome::Cancelled),
            5 => WaitPhase::Cancelled(WaitOutcome::Terminated),
            _ => WaitPhase::Resolved(self.outcome().expect("terminal kernel wait")),
        }
    }

    pub fn commit(&self) -> bool {
        let _ = self
            .phase
            .fetch_update(Ordering::Release, Ordering::Relaxed, |phase| {
                ((phase & !TIMER_ARMED) == 0).then_some(phase | 1)
            });
        self.outcome().is_none()
    }

    // Called only with the canonical scheduler lock held.
    pub(super) fn resolve(&self, outcome: WaitOutcome) -> bool {
        let encoded = match outcome {
            WaitOutcome::Ready => 2,
            WaitOutcome::TimedOut => 3,
            WaitOutcome::Cancelled => 4,
            WaitOutcome::Terminated => 5,
            WaitOutcome::Signalled(index) => 0x100 + u64::from(index),
        };
        let mut phase = self.phase.load(Ordering::Relaxed);
        loop {
            if phase & !TIMER_ARMED > 1 {
                return false;
            }
            match self.phase.compare_exchange_weak(
                phase,
                encoded,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(previous) => {
                    if previous & TIMER_ARMED != 0 {
                        self.timer_count
                            .get()
                            .expect("armed kernel deadline")
                            .fetch_sub(1, Ordering::Release);
                    }
                    return true;
                }
                Err(observed) => phase = observed,
            }
        }
    }

    pub(super) fn arm_timer(&self, count: &Arc<AtomicUsize>) -> bool {
        self.timer_count
            .set(count.clone())
            .expect("one deadline per kernel wait");
        count.fetch_add(1, Ordering::Relaxed);
        if self
            .phase
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |phase| {
                ((phase & !TIMER_ARMED) <= 1).then_some(phase | TIMER_ARMED)
            })
            .is_ok()
        {
            true
        } else {
            count.fetch_sub(1, Ordering::Release);
            false
        }
    }

    pub(super) fn has_timer(&self) -> bool {
        self.timer_count.get().is_some()
    }

    pub async fn wait(&self) -> WaitOutcome {
        std::future::poll_fn(|context| {
            if let Some(outcome) = self.outcome().filter(|outcome| {
                *outcome == WaitOutcome::Terminated || self.effects.load(Ordering::Acquire) == 0
            }) {
                return Poll::Ready(outcome);
            }
            let Some(kernel) = self.kernel.upgrade() else {
                return Poll::Ready(WaitOutcome::Terminated);
            };
            kernel
                .edit(|state| {
                    if let Some(outcome) = self.outcome().filter(|outcome| {
                        *outcome == WaitOutcome::Terminated
                            || self.effects.load(Ordering::Acquire) == 0
                    }) {
                        return Ok(Poll::Ready(outcome));
                    }
                    let index = state.slots.index(self.token.thread)?;
                    if state.slots[index].wait_token() != Some(self.token) {
                        return Err(SchedulerError::StaleWake(self.token));
                    }
                    let slot = &mut state.slots[index].wait_waker;
                    if slot
                        .as_ref()
                        .is_none_or(|old| !old.will_wake(context.waker()))
                        && let Some(old) = slot.replace(context.waker().clone())
                    {
                        state.deferred_wakers.push((old, false));
                    }
                    Ok(Poll::Pending)
                })
                .expect("retained current kernel wait")
        })
        .await
    }
}

impl SchedulerState {
    pub fn resolve_wait(&self, token: WakeToken, outcome: WaitOutcome) -> bool {
        self.edit(|state| {
            let Ok(index) = state.slots.index(token.thread) else {
                return Ok(false);
            };
            if state.slots[index].wait_token() != Some(token) {
                return Ok(false);
            }
            Ok(state.resolve_wait(index, outcome, &self.profile))
        })
        .expect("kernel wait resolution")
    }

    pub fn resolve_wait_deferred(
        &self,
        token: WakeToken,
        outcome: WaitOutcome,
    ) -> (bool, super::events::EventNotifications) {
        let (result, notifications) = self.edit_deferred(|state| {
            let Ok(index) = state.slots.index(token.thread) else {
                return Ok(false);
            };
            if state.slots[index].wait_token() != Some(token) {
                return Ok(false);
            }
            Ok(state.resolve_wait(index, outcome, &self.profile))
        });
        (result.expect("kernel wait resolution"), notifications)
    }
}

impl State {
    pub(super) fn resolve_wait(
        &mut self,
        index: Index,
        outcome: WaitOutcome,
        profile: &MachineSchedulerProfile,
    ) -> bool {
        let Some(wait) = self.slots[index].wait.as_ref() else {
            return false;
        };
        if outcome == WaitOutcome::TimedOut && wait.effects.load(Ordering::Acquire) != 0 {
            wait.deferred_timeout.store(true, Ordering::Release);
            return false;
        }
        if !wait.resolve(outcome) {
            return false;
        }
        self.timer_invalidated |= wait.has_timer();
        let effects = wait.effects.load(Ordering::Acquire);
        #[cfg(feature = "performance-counters")]
        {
            self.statistics.waits_resolved += 1;
            self.slots[index].resolved_at_ns = Some(self.clock.scheduler_time_ns());
        }
        self.unlink_event_wait(index);
        self.remove_waiter(index, profile, None);
        self.slots[index].cancellable = false;
        if (effects == 0 || outcome == WaitOutcome::Terminated)
            && let Some(waker) = self.slots[index].wait_waker.take()
        {
            self.deferred_wakers.push((waker, true));
        }
        true
    }
}

impl Drop for KernelWait {
    fn drop(&mut self) {
        if self.phase.load(Ordering::Relaxed) & TIMER_ARMED != 0 {
            self.timer_count
                .get()
                .expect("armed kernel deadline")
                .fetch_sub(1, Ordering::Release);
        }
    }
}

/// A userspace atomic effect retains the same exclusion as its wait generation.
/// An expired timer is applied when that accepted effect has published its result.
#[derive(Debug)]
pub struct WaitEffect {
    wait: Arc<KernelWait>,
}
impl WaitEffect {
    pub fn registration(&self) -> &Arc<KernelWait> {
        &self.wait
    }
}
impl State {
    pub(super) fn claim_wait_effect(&self, index: Index) -> Option<WaitEffect> {
        let wait = self.slots[index].wait.as_ref()?;
        if wait.outcome().is_some() {
            return None;
        }
        wait.effects.fetch_add(1, Ordering::Release);
        Some(WaitEffect { wait: wait.clone() })
    }
}
impl Drop for WaitEffect {
    fn drop(&mut self) {
        let Some(kernel) = self.wait.kernel.upgrade() else {
            return;
        };
        kernel
            .edit(|state| {
                let remaining = self.wait.effects.fetch_sub(1, Ordering::AcqRel) - 1;
                if remaining != 0 {
                    return Ok(());
                }
                let Ok(index) = state.slots.index(self.wait.token.thread) else {
                    return Ok(());
                };
                if state.slots[index].wait_token() != Some(self.wait.token) {
                    return Ok(());
                }
                if self.wait.deferred_timeout.swap(false, Ordering::AcqRel)
                    && self.wait.outcome().is_none()
                {
                    state.resolve_wait(index, WaitOutcome::TimedOut, &kernel.profile);
                } else if self.wait.outcome().is_some()
                    && let Some(waker) = state.slots[index].wait_waker.take()
                {
                    state.deferred_wakers.push((waker, true));
                }
                Ok(())
            })
            .expect("retained wait effect");
    }
}
