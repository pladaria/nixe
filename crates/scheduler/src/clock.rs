//! Shared kernel clock for scheduling, guest time and host services.

use std::fmt::{Debug, Formatter};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
mod tsc;

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
    host: HostClock,
    scheduler_time_ns: AtomicU64,
    // Running: host counter at virtual time zero. Paused: flag + frozen elapsed.
    // One atomic word keeps reads lock-free and transitions indivisible.
    pause: AtomicU64,
    timer_epoch: Mutex<u64>,
    timer_changed: Condvar,
}

const PAUSED: u64 = 1 << 63;

#[derive(Debug)]
enum HostClock {
    Monotonic(Instant),
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    Tsc(tsc::Scale),
}

impl HostClock {
    fn new(mode: VirtualClockMode) -> (Self, u64) {
        #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
        if mode == VirtualClockMode::Realtime
            && let Some(scale) = tsc::Scale::host()
        {
            let base = tsc::read();
            if base < PAUSED {
                return (Self::Tsc(scale), base);
            }
        }
        let _ = mode;
        (Self::Monotonic(Instant::now()), 0)
    }

    fn sample(&self) -> u64 {
        match self {
            Self::Monotonic(anchor) => {
                anchor.elapsed().as_nanos().min(u128::from(PAUSED - 1)) as u64
            }
            #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
            Self::Tsc(_) => tsc::read(),
        }
    }

    fn nanoseconds(&self, elapsed: u64) -> u64 {
        match self {
            Self::Monotonic(_) => elapsed,
            #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
            Self::Tsc(scale) => scale.nanoseconds(elapsed),
        }
    }
}

/// The shared running base or paused value and the scale for an inline x64 counter.
/// Retain this handle for as long as generated code can read the base.
#[derive(Clone)]
pub struct TscCounter {
    state: Arc<VirtualClockState>,
    multiplier: u64,
}

impl TscCounter {
    /// Bit 63 marks a paused clock. The low bits are then frozen elapsed TSC ticks.
    /// While running, the word is the TSC value mapped to virtual time zero.
    pub fn base_ptr(&self) -> *const u64 {
        self.state.pause.as_ptr().cast_const()
    }

    /// The high half of an unsigned product by this value gives guest ticks.
    pub const fn multiplier(&self) -> u64 {
        self.multiplier
    }
}

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
/// use a monotonic host counter, so host wall-clock adjustments cannot make guest time move
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
        let (host, base) = HostClock::new(mode);
        Self {
            state: Arc::new(VirtualClockState {
                mode,
                wall_anchor,
                host,
                scheduler_time_ns: AtomicU64::new(0),
                pause: AtomicU64::new(base),
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
        Duration::from_nanos(self.elapsed_ns())
    }

    /// Returns monotonic elapsed time in nanoseconds.
    #[must_use]
    pub fn elapsed_ns(&self) -> u64 {
        match self.state.mode {
            VirtualClockMode::Realtime => self.host_elapsed_ns(),
            VirtualClockMode::Fixed { .. } => self.state.scheduler_time_ns.load(Ordering::Acquire),
        }
    }

    /// Returns an inline counter only when this clock uses the host TSC.
    #[must_use]
    pub fn tsc_counter(&self, frequency: u32) -> Option<TscCounter> {
        #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
        if let HostClock::Tsc(scale) = self.state.host {
            return Some(TscCounter {
                state: Arc::clone(&self.state),
                multiplier: scale.multiplier(u64::from(frequency))?,
            });
        }
        let _ = frequency;
        None
    }

    /// Host service time excluding pauses, also available with a fixed guest clock.
    #[must_use]
    pub fn host_elapsed(&self) -> Duration {
        Duration::from_nanos(self.host_elapsed_ns())
    }

    fn host_elapsed_ns(&self) -> u64 {
        let host = &self.state.host;
        loop {
            let state = self.state.pause.load(Ordering::Acquire);
            if state & PAUSED != 0 {
                return host.nanoseconds(state & !PAUSED);
            }
            let now = host.sample();
            if self.state.pause.load(Ordering::Acquire) == state {
                return host.nanoseconds(now.saturating_sub(state));
            }
        }
    }

    /// Freezes/resumes the shared process clock, excluding host pause duration.
    pub fn set_paused(&self, paused: bool) {
        let mut epoch = self.timer_epoch();
        let state = self.state.pause.load(Ordering::Relaxed);
        if (state & PAUSED != 0) == paused {
            return;
        }
        // The epoch mutex serializes writers; atomic publication is for readers.
        let next = pause_transition(state, self.state.host.sample(), paused);
        self.state.pause.store(next, Ordering::Release);
        *epoch = epoch.wrapping_add(1);
        self.state.timer_changed.notify_all();
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
            VirtualClockMode::Realtime => scheduled.max(self.host_elapsed_ns()),
            VirtualClockMode::Fixed { .. } => scheduled,
        }
    }

    /// Advances diagnostic guest time monotonically, preserving pause semantics.
    pub fn advance_scheduler_to(&self, nanoseconds: u64) {
        let mut epoch = self.timer_epoch();
        if self.is_paused() {
            return;
        }
        let previous = self.state.scheduler_time_ns.load(Ordering::Relaxed);
        if previous < nanoseconds {
            self.state
                .scheduler_time_ns
                .store(nanoseconds, Ordering::Release);
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
    fn concurrent_diagnostic_advances_remain_monotonic_and_respect_pause() {
        let clock = VirtualClock::new(VirtualClockMode::Fixed { unix_seconds: 0 });
        let start = std::sync::Barrier::new(4);
        std::thread::scope(|scope| {
            for worker in 0..4 {
                let clock = &clock;
                let start = &start;
                scope.spawn(move || {
                    start.wait();
                    for target in (worker..=1024).step_by(4) {
                        clock.advance_scheduler_to(target);
                        assert!(clock.scheduler_time_ns() >= target);
                    }
                });
            }
        });
        assert_eq!(clock.elapsed(), Duration::from_nanos(1024));
        clock.set_paused(true);
        std::thread::scope(|scope| {
            for target in 1025..=1028 {
                let clock = &clock;
                scope.spawn(move || clock.advance_scheduler_to(target));
            }
        });
        assert_eq!(clock.elapsed(), Duration::from_nanos(1024));
        clock.set_paused(false);
        clock.advance_scheduler_to(2048);
        assert_eq!(clock.elapsed(), Duration::from_nanos(2048));
    }

    #[test]
    fn realtime_clock_is_anchored_near_the_host_time() {
        let before = host_unix_seconds();
        let clock = VirtualClock::new(VirtualClockMode::Realtime);
        let after = host_unix_seconds();

        assert!((before..=after.saturating_add(1)).contains(&clock.unix_seconds()));
    }
}
