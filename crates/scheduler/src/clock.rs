//! Shared kernel clock for scheduling, guest time and host services.

use std::fmt::{Debug, Formatter};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Initial wall-clock policy selected by the application runtime.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum VirtualClockMode {
    /// Anchor the wall clock to the host once and advance it monotonically.
    Realtime,
    /// Freeze virtual wall and monotonic time at a caller-provided timestamp.
    Fixed { unix_seconds: i64 },
}

#[derive(Debug)]
struct VirtualClockState {
    mode: VirtualClockMode,
    wall_anchor: i64,
    monotonic_anchor: Instant,
    scheduler_time_ns: AtomicU64,
    // Running: accumulated paused nanoseconds. Paused: flag + frozen elapsed.
    // One atomic word keeps reads lock-free and transitions indivisible.
    pause: AtomicU64,
    timer_epoch: Mutex<u64>,
    timer_changed: Condvar,
}

const PAUSED: u64 = 1 << 63;
fn pause_transition(state: u64, now: u64, paused: bool) -> u64 {
    if paused {
        PAUSED | now.saturating_sub(state)
    } else {
        now.saturating_sub(state & !PAUSED)
    }
}

/// One cloneable clock source shared by the scheduler and emulated services.
///
/// Realtime mode samples `SystemTime` only at construction. Subsequent reads
/// use `Instant`, so host wall-clock adjustments cannot make guest time move
/// backwards.
#[derive(Clone)]
pub struct VirtualClock {
    state: Arc<VirtualClockState>,
}

impl VirtualClock {
    /// Creates a virtual clock using the requested initial policy.
    #[must_use]
    pub fn new(mode: VirtualClockMode) -> Self {
        let wall_anchor = match mode {
            VirtualClockMode::Realtime => host_unix_seconds(),
            VirtualClockMode::Fixed { unix_seconds } => unix_seconds,
        };
        Self {
            state: Arc::new(VirtualClockState {
                mode,
                wall_anchor,
                monotonic_anchor: Instant::now(),
                scheduler_time_ns: AtomicU64::new(0),
                pause: AtomicU64::new(0),
                timer_epoch: Mutex::new(0),
                timer_changed: Condvar::new(),
            }),
        }
    }

    /// Returns the virtual POSIX wall-clock timestamp.
    #[must_use]
    pub fn unix_seconds(&self) -> i64 {
        self.state
            .wall_anchor
            .saturating_add(duration_seconds(self.elapsed()))
    }

    /// Returns monotonic elapsed time since this clock was created.
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        match self.state.mode {
            VirtualClockMode::Realtime => self.host_elapsed(),
            VirtualClockMode::Fixed { .. } => {
                Duration::from_nanos(self.state.scheduler_time_ns.load(Ordering::Acquire))
            }
        }
    }

    /// Host service time excluding pauses, also available with a fixed guest clock.
    #[must_use]
    pub fn host_elapsed(&self) -> Duration {
        loop {
            let state = self.state.pause.load(Ordering::Acquire);
            if state & PAUSED != 0 {
                return Duration::from_nanos(state & !PAUSED);
            }
            let now = self.host_elapsed_ns();
            if self.state.pause.load(Ordering::Acquire) == state {
                return Duration::from_nanos(now.saturating_sub(state));
            }
        }
    }

    fn host_elapsed_ns(&self) -> u64 {
        self.state
            .monotonic_anchor
            .elapsed()
            .as_nanos()
            .min(u128::from(PAUSED - 1)) as u64
    }

    /// Freezes/resumes the shared process clock, excluding host pause duration.
    pub fn set_paused(&self, paused: bool) {
        let mut epoch = self.timer_epoch();
        let mut state = self.state.pause.load(Ordering::Acquire);
        loop {
            if (state & PAUSED != 0) == paused {
                return;
            }
            let next = pause_transition(state, self.host_elapsed_ns(), paused);
            match self.state.pause.compare_exchange_weak(
                state,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    *epoch = epoch.wrapping_add(1);
                    self.state.timer_changed.notify_all();
                    return;
                }
                Err(observed) => state = observed,
            }
        }
    }

    #[must_use]
    pub fn is_paused(&self) -> bool {
        self.state.pause.load(Ordering::Acquire) & PAUSED != 0
    }

    /// Returns the selected wall-clock policy.
    #[must_use]
    pub fn mode(&self) -> VirtualClockMode {
        self.state.mode
    }

    /// Scheduler time follows the same monotonic host clock in realtime mode,
    /// so host device events can arrive before guest wait deadlines expire.
    /// Fixed-clock runs advance only through their explicit diagnostic driver.
    #[must_use]
    pub fn scheduler_time_ns(&self) -> u64 {
        let scheduled = self.state.scheduler_time_ns.load(Ordering::Acquire);
        match self.state.mode {
            VirtualClockMode::Realtime => {
                scheduled.max(self.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64)
            }
            VirtualClockMode::Fixed { .. } => scheduled,
        }
    }

    /// Advances diagnostic guest time monotonically, preserving pause semantics.
    pub fn advance_scheduler_to(&self, nanoseconds: u64) {
        let mut epoch = self.timer_epoch();
        if self.is_paused() {
            return;
        }
        let previous = self
            .state
            .scheduler_time_ns
            .fetch_max(nanoseconds, Ordering::AcqRel);
        if previous < nanoseconds {
            *epoch = epoch.wrapping_add(1);
            self.state.timer_changed.notify_all();
        }
    }

    pub(crate) fn timer_epoch(&self) -> MutexGuard<'_, u64> {
        self.state
            .timer_epoch
            .lock()
            .expect("timer clock lock poisoned")
    }

    pub(crate) fn wake_timers(&self) {
        let mut epoch = self.timer_epoch();
        *epoch = epoch.wrapping_add(1);
        self.state.timer_changed.notify_all();
    }

    pub(crate) fn wait_for_timer_change<'a>(
        &self,
        epoch: MutexGuard<'a, u64>,
        timeout: Option<Duration>,
    ) -> MutexGuard<'a, u64> {
        match timeout {
            Some(timeout) => {
                self.state
                    .timer_changed
                    .wait_timeout(epoch, timeout)
                    .expect("timer clock lock poisoned")
                    .0
            }
            None => self
                .state
                .timer_changed
                .wait(epoch)
                .expect("timer clock lock poisoned"),
        }
    }
}

impl Debug for VirtualClock {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("VirtualClock")
            .field("mode", &self.state.mode)
            .field("wall_anchor", &self.state.wall_anchor)
            .finish_non_exhaustive()
    }
}

impl Default for VirtualClock {
    fn default() -> Self {
        Self::new(VirtualClockMode::Realtime)
    }
}

fn host_unix_seconds() -> i64 {
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(duration) => duration_seconds(duration),
        Err(error) => duration_seconds(error.duration()).saturating_neg(),
    }
}

fn duration_seconds(duration: Duration) -> i64 {
    i64::try_from(duration.as_secs()).unwrap_or(i64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_pauses_exclude_only_the_paused_host_duration() {
        let frozen = pause_transition(0, 100, true);
        assert_eq!(frozen, PAUSED | 100);
        let offset = pause_transition(frozen, 1100, false);
        assert_eq!(offset, 1000);
        assert_eq!(1500 - offset, 500);
        let frozen = pause_transition(offset, 1600, true);
        assert_eq!(frozen, PAUSED | 600);
        assert_eq!(pause_transition(frozen, 3600, false), 3000);
    }

    #[test]
    fn paused_clock_is_shared_and_freezes_both_guest_and_service_timelines() {
        let clock = VirtualClock::default();
        clock.set_paused(true);
        let frozen = clock.elapsed();
        let scheduler = clock.scheduler_time_ns();
        let clone = clock.clone();
        std::thread::sleep(Duration::from_millis(2));
        clone.set_paused(true); // Repeated requests must not reset the anchor.
        assert_eq!(clone.elapsed(), frozen);
        assert_eq!(clone.host_elapsed(), frozen);
        assert_eq!(clone.scheduler_time_ns(), scheduler);
        clone.set_paused(false);
        assert!(!clock.is_paused());
        assert!(clock.elapsed() >= frozen);
        let fixed = VirtualClock::new(VirtualClockMode::Fixed { unix_seconds: 7 });
        fixed.set_paused(true);
        let host = fixed.host_elapsed();
        assert_eq!(fixed.elapsed(), Duration::ZERO);
        assert_eq!(fixed.unix_seconds(), 7);
        assert_eq!(fixed.host_elapsed(), host);
    }

    #[test]
    fn fixed_mode_freezes_wall_and_monotonic_time() {
        let clock = VirtualClock::new(VirtualClockMode::Fixed {
            unix_seconds: 1_234,
        });

        assert_eq!(clock.unix_seconds(), 1_234);
        assert_eq!(
            clock.mode(),
            VirtualClockMode::Fixed {
                unix_seconds: 1_234
            }
        );
        assert_eq!(clock.elapsed(), Duration::ZERO);
    }

    #[test]
    fn realtime_clock_is_anchored_near_the_host_time() {
        let before = host_unix_seconds();
        let clock = VirtualClock::new(VirtualClockMode::Realtime);
        let after = host_unix_seconds();

        assert!((before..=after.saturating_add(1)).contains(&clock.unix_seconds()));
    }
}
