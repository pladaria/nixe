//! Host-independent virtual wall and monotonic clock state.

use std::fmt::{Debug, Formatter};
use std::sync::Arc;
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
}

/// One cloneable clock source shared by all services of an emulated process.
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
            }),
        }
    }

    /// Returns the virtual POSIX wall-clock timestamp.
    #[must_use]
    pub fn unix_seconds(&self) -> i64 {
        self.state
            .wall_anchor
            .saturating_add(match self.state.mode {
                VirtualClockMode::Realtime => duration_seconds(self.elapsed()),
                VirtualClockMode::Fixed { .. } => 0,
            })
    }

    /// Returns monotonic elapsed time since this clock was created.
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        match self.state.mode {
            VirtualClockMode::Realtime => self.state.monotonic_anchor.elapsed(),
            VirtualClockMode::Fixed { .. } => Duration::ZERO,
        }
    }

    /// Returns the selected wall-clock policy.
    #[must_use]
    pub fn mode(&self) -> VirtualClockMode {
        self.state.mode
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
