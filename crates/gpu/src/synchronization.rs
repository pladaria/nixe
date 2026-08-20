//! Device-neutral guest timeline identities and reservation rules.

use std::collections::VecDeque;
use std::fmt::{Display, Formatter};

/// Half of the 32-bit counter domain.
///
/// Modular values separated by exactly this distance have no unique order.
const AMBIGUOUS_DISTANCE: u32 = 1 << 31;
const MAX_OUTSTANDING_DISTANCE: u64 = (AMBIGUOUS_DISTANCE - 1) as u64;

/// Guest-visible identity of one hardware syncpoint.
///
/// The owning console frontend validates the hardware-specific ID range.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct GuestSyncpointId(u32);

impl GuestSyncpointId {
    /// Creates an ID already validated by the owning frontend.
    #[must_use]
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    /// Returns the guest-visible numeric representation.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

impl Display for GuestSyncpointId {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "syncpoint={}", self.0)
    }
}

/// One value of a wrapping 32-bit guest syncpoint counter.
///
/// NVIDIA host1x syncpoints use wrapping 32-bit counters and compare a
/// threshold using the sign bit of the modular distance. Nixe rejects the
/// exactly-half-range case rather than assigning it an arbitrary order.
/// Reference: Linux v6.12 `drivers/gpu/host1x/syncpt.c`,
/// `host1x_syncpt_is_expired`:
/// <https://github.com/torvalds/linux/blob/v6.12/drivers/gpu/host1x/syncpt.c#L238-L249>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct GuestSyncpointValue(u32);

impl GuestSyncpointValue {
    /// Creates a value from its guest-visible bit pattern.
    #[must_use]
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    /// Returns the guest-visible bit pattern.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }

    /// Adds increments using the hardware counter's explicit rollover rule.
    #[must_use]
    pub const fn wrapping_add(self, increments: u32) -> Self {
        Self(self.0.wrapping_add(increments))
    }

    /// Compares two values inside a valid less-than-half-range window.
    ///
    /// `Greater` means `self` is ahead of `other` in timeline order. Values
    /// exactly half a counter domain apart are ambiguous and rejected.
    pub const fn checked_cmp(
        self,
        other: Self,
    ) -> Result<std::cmp::Ordering, SyncpointComparisonError> {
        let distance = self.0.wrapping_sub(other.0);
        if distance == 0 {
            Ok(std::cmp::Ordering::Equal)
        } else if distance == AMBIGUOUS_DISTANCE {
            Err(SyncpointComparisonError {
                left: self,
                right: other,
            })
        } else if distance < AMBIGUOUS_DISTANCE {
            Ok(std::cmp::Ordering::Greater)
        } else {
            Ok(std::cmp::Ordering::Less)
        }
    }

    /// Returns whether this counter has reached a guest threshold.
    ///
    /// This is the directional host1x expiration test, not a total ordering:
    /// a threshold exactly half a counter range ahead is not expired. The
    /// distinction matters because [`Self::checked_cmp`] deliberately rejects
    /// that ambiguous pair while a hardware wait still has defined behavior.
    /// Reference: Linux v6.12 `drivers/gpu/host1x/syncpt.c`,
    /// `host1x_syncpt_is_expired`:
    /// <https://github.com/torvalds/linux/blob/v6.12/drivers/gpu/host1x/syncpt.c#L238-L249>
    #[must_use]
    pub const fn has_reached(self, threshold: Self) -> bool {
        self.0.wrapping_sub(threshold.0) & AMBIGUOUS_DISTANCE == 0
    }
}

impl Display for GuestSyncpointValue {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

/// A modular syncpoint comparison has no unique ordering.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SyncpointComparisonError {
    pub left: GuestSyncpointValue,
    pub right: GuestSyncpointValue,
}

impl Display for SyncpointComparisonError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "guest syncpoint values are exactly half a counter range apart: left={} right={}",
            self.left, self.right
        )
    }
}

impl std::error::Error for SyncpointComparisonError {}

/// Guest-visible point on one syncpoint.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GuestTimelinePoint {
    syncpoint: GuestSyncpointId,
    value: GuestSyncpointValue,
}

impl GuestTimelinePoint {
    /// Combines a syncpoint identity with one guest-visible counter value.
    #[must_use]
    pub const fn new(syncpoint: GuestSyncpointId, value: GuestSyncpointValue) -> Self {
        Self { syncpoint, value }
    }

    /// Returns the syncpoint containing this point.
    #[must_use]
    pub const fn syncpoint(self) -> GuestSyncpointId {
        self.syncpoint
    }

    /// Returns the wrapping guest-visible counter value.
    #[must_use]
    pub const fn value(self) -> GuestSyncpointValue {
        self.value
    }
}

impl Display for GuestTimelinePoint {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}:{}", self.syncpoint, self.value)
    }
}

/// Stable emulator identity for the owner of timeline mutations.
///
/// It is not a Horizon process object, host pointer, or backend handle.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct TimelineOwnerId(u64);

impl TimelineOwnerId {
    /// Creates an identity from a value assigned by the frontend owner.
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the owner-assigned numeric representation.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl Display for TimelineOwnerId {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "timeline-owner=0x{:016x}", self.0)
    }
}

/// Frontend-assigned lifetime identity of one guest timeline instance.
///
/// A frontend must not reuse this identity while any reservation from the old
/// instance can remain in flight. Including it in every reservation prevents a
/// stale token from matching a recreated syncpoint with the same numeric ID.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct TimelineInstanceId(u64);

impl TimelineInstanceId {
    /// Creates a lifetime identity assigned by the owning frontend.
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the frontend-assigned numeric representation.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl Display for TimelineInstanceId {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "timeline-instance=0x{:016x}", self.0)
    }
}

/// One owned reservation of future increments on a guest timeline.
///
/// The private logical position is monotonic across guest counter rollover.
/// Callers cannot construct or retarget reservations themselves.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ReservedTimelinePoint {
    syncpoint: GuestSyncpointId,
    instance: TimelineInstanceId,
    reservation: u64,
    owner: TimelineOwnerId,
    increments: u32,
    logical_position: u64,
    point: GuestTimelinePoint,
}

impl ReservedTimelinePoint {
    /// Returns the guest-visible endpoint of this reservation.
    #[must_use]
    pub const fn point(&self) -> GuestTimelinePoint {
        self.point
    }

    /// Returns the owner permitted to complete or cancel this reservation.
    #[must_use]
    pub const fn owner(&self) -> TimelineOwnerId {
        self.owner
    }

    /// Returns how many guest increments lead to this reservation's endpoint.
    #[must_use]
    pub const fn increments(&self) -> u32 {
        self.increments
    }

    /// Returns the timeline-local reservation identity.
    #[must_use]
    pub const fn reservation_id(&self) -> u64 {
        self.reservation
    }

    /// Compares reservations on the same guest timeline without wraparound.
    pub fn checked_cmp(
        &self,
        other: &Self,
    ) -> Result<std::cmp::Ordering, TimelinePointComparisonError> {
        if self.syncpoint != other.syncpoint || self.instance != other.instance {
            return Err(TimelinePointComparisonError {
                left_syncpoint: self.syncpoint,
                left_instance: self.instance,
                right_syncpoint: other.syncpoint,
                right_instance: other.instance,
            });
        }
        Ok(self.logical_position.cmp(&other.logical_position))
    }
}

impl Display for ReservedTimelinePoint {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "reservation={} owner={} point={}",
            self.reservation, self.owner, self.point
        )
    }
}

/// Reserved points from distinct timeline instances have no shared order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TimelinePointComparisonError {
    pub left_syncpoint: GuestSyncpointId,
    pub left_instance: TimelineInstanceId,
    pub right_syncpoint: GuestSyncpointId,
    pub right_instance: TimelineInstanceId,
}

impl Display for TimelinePointComparisonError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "reserved points belong to different guest timelines: left={}/{} right={}/{}",
            self.left_syncpoint, self.left_instance, self.right_syncpoint, self.right_instance
        )
    }
}

impl std::error::Error for TimelinePointComparisonError {}

/// Neutral state for one owned, wrapping guest syncpoint timeline.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GuestTimeline {
    syncpoint: GuestSyncpointId,
    instance: TimelineInstanceId,
    owner: TimelineOwnerId,
    initial_value: GuestSyncpointValue,
    current_position: u64,
    reserved_position: u64,
    next_reservation: u64,
    reservations: VecDeque<ReservedTimelinePoint>,
}

impl GuestTimeline {
    /// Creates an empty timeline whose mutations belong exclusively to owner.
    #[must_use]
    pub const fn new(
        syncpoint: GuestSyncpointId,
        instance: TimelineInstanceId,
        owner: TimelineOwnerId,
        initial_value: GuestSyncpointValue,
    ) -> Self {
        Self {
            syncpoint,
            instance,
            owner,
            initial_value,
            current_position: 0,
            reserved_position: 0,
            next_reservation: 1,
            reservations: VecDeque::new(),
        }
    }

    /// Returns this timeline's guest syncpoint identity.
    #[must_use]
    pub const fn syncpoint(&self) -> GuestSyncpointId {
        self.syncpoint
    }

    /// Returns the lifetime identity which prevents stale reservation reuse.
    #[must_use]
    pub const fn instance(&self) -> TimelineInstanceId {
        self.instance
    }

    /// Returns the sole owner permitted to reserve and advance this timeline.
    #[must_use]
    pub const fn owner(&self) -> TimelineOwnerId {
        self.owner
    }

    /// Returns the currently completed guest-visible point.
    #[must_use]
    pub fn current_point(&self) -> GuestTimelinePoint {
        self.point_at(self.current_position)
    }

    /// Returns the latest reserved point, or the current point when idle.
    #[must_use]
    pub fn latest_reserved_point(&self) -> GuestTimelinePoint {
        self.point_at(self.reserved_position)
    }

    /// Returns the number of reserved increments not yet completed.
    #[must_use]
    pub const fn outstanding_increments(&self) -> u64 {
        self.reserved_position - self.current_position
    }

    /// Returns the number of outstanding owned reservations.
    #[must_use]
    pub fn reservation_count(&self) -> usize {
        self.reservations.len()
    }

    /// Returns whether the current guest counter has reached `threshold`.
    #[must_use]
    pub fn has_reached(&self, threshold: GuestSyncpointValue) -> bool {
        self.current_point().value().has_reached(threshold)
    }

    /// Applies one immediate frontend-owned counter increment.
    ///
    /// Immediate increments cannot overtake reserved backend work. A frontend
    /// encountering that state must preserve ordering through its submission
    /// coordinator rather than publishing speculative progress.
    pub fn increment_immediate(
        &mut self,
        owner: TimelineOwnerId,
    ) -> Result<GuestTimelinePoint, TimelineIncrementError> {
        self.require_owner(owner)
            .map_err(TimelineIncrementError::WrongOwner)?;
        if let Some(pending) = self.reservations.front() {
            return Err(TimelineIncrementError::PendingReservation {
                reservation: pending.reservation_id(),
            });
        }
        let reservation = self
            .reserve(owner, 1)
            .map_err(TimelineIncrementError::Reservation)?;
        self.advance(owner, &reservation)
            .map_err(TimelineIncrementError::Advance)
    }

    /// Reserves a non-empty future interval without advancing guest progress.
    ///
    /// The complete outstanding window stays below half of the 32-bit domain,
    /// preserving an unambiguous modular order for every visible endpoint.
    pub fn reserve(
        &mut self,
        owner: TimelineOwnerId,
        increments: u32,
    ) -> Result<ReservedTimelinePoint, TimelineReservationError> {
        self.require_owner(owner)
            .map_err(TimelineReservationError::WrongOwner)?;
        if increments == 0 {
            return Err(TimelineReservationError::ZeroIncrements);
        }

        let outstanding = self.outstanding_increments();
        let requested = u64::from(increments);
        if outstanding + requested > MAX_OUTSTANDING_DISTANCE {
            return Err(TimelineReservationError::WindowExhausted {
                outstanding,
                requested: increments,
            });
        }
        let logical_position = self
            .reserved_position
            .checked_add(requested)
            .ok_or(TimelineReservationError::LogicalPositionExhausted)?;
        let reservation = self.next_reservation;
        let next_reservation = reservation
            .checked_add(1)
            .ok_or(TimelineReservationError::ReservationIdentityExhausted)?;
        self.reservations
            .try_reserve(1)
            .map_err(|_| TimelineReservationError::ResourceExhausted)?;
        let point = ReservedTimelinePoint {
            syncpoint: self.syncpoint,
            instance: self.instance,
            reservation,
            owner,
            increments,
            logical_position,
            point: self.point_at(logical_position),
        };

        self.reserved_position = logical_position;
        self.next_reservation = next_reservation;
        self.reservations.push_back(point.clone());
        Ok(point)
    }

    /// Advances through the oldest complete reservation.
    ///
    /// This method only enforces neutral timeline ordering and ownership. The
    /// completion coordinator added by later blocks is responsible for calling
    /// it only after host completion and required visibility transitions.
    pub fn advance(
        &mut self,
        owner: TimelineOwnerId,
        completed: &ReservedTimelinePoint,
    ) -> Result<GuestTimelinePoint, TimelineAdvanceError> {
        self.validate_advance(owner, completed)?;
        self.current_position = completed.logical_position;
        self.reservations.pop_front();
        Ok(self.current_point())
    }

    /// Validates publication without changing guest-visible progress.
    ///
    /// Completion coordinators use this before applying memory visibility so
    /// a stale or mismatched timeline cannot partially publish device writes.
    pub fn validate_advance(
        &self,
        owner: TimelineOwnerId,
        completed: &ReservedTimelinePoint,
    ) -> Result<(), TimelineAdvanceError> {
        self.require_owner(owner)
            .map_err(TimelineAdvanceError::WrongOwner)?;
        if completed.syncpoint != self.syncpoint {
            return Err(TimelineAdvanceError::WrongSyncpoint {
                expected: self.syncpoint,
                observed: completed.syncpoint,
            });
        }
        if completed.instance != self.instance {
            return Err(TimelineAdvanceError::WrongInstance {
                expected: self.instance,
                observed: completed.instance,
            });
        }
        if completed.owner != owner {
            return Err(TimelineAdvanceError::WrongOwner(OwnerMismatch {
                expected: owner,
                observed: completed.owner,
            }));
        }

        let Some(next) = self.reservations.front() else {
            return Err(TimelineAdvanceError::UnknownReservation {
                reservation: completed.reservation,
            });
        };
        if next.reservation != completed.reservation || next != completed {
            let pending = self
                .reservations
                .iter()
                .any(|reservation| reservation == completed);
            return if pending {
                Err(TimelineAdvanceError::OutOfOrder {
                    expected: next.reservation,
                    observed: completed.reservation,
                })
            } else {
                Err(TimelineAdvanceError::UnknownReservation {
                    reservation: completed.reservation,
                })
            };
        }

        Ok(())
    }

    fn point_at(&self, logical_position: u64) -> GuestTimelinePoint {
        GuestTimelinePoint::new(
            self.syncpoint,
            self.initial_value.wrapping_add(logical_position as u32),
        )
    }

    fn require_owner(&self, owner: TimelineOwnerId) -> Result<(), OwnerMismatch> {
        if owner == self.owner {
            Ok(())
        } else {
            Err(OwnerMismatch {
                expected: self.owner,
                observed: owner,
            })
        }
    }
}

/// Attempted timeline mutation by an identity without authority.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OwnerMismatch {
    pub expected: TimelineOwnerId,
    pub observed: TimelineOwnerId,
}

impl Display for OwnerMismatch {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "guest timeline owner mismatch: expected {} observed {}",
            self.expected, self.observed
        )
    }
}

/// Failure to reserve future guest timeline progress.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TimelineReservationError {
    WrongOwner(OwnerMismatch),
    ZeroIncrements,
    WindowExhausted { outstanding: u64, requested: u32 },
    LogicalPositionExhausted,
    ReservationIdentityExhausted,
    ResourceExhausted,
}

impl Display for TimelineReservationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WrongOwner(error) => error.fmt(formatter),
            Self::ZeroIncrements => {
                formatter.write_str("guest timeline reservation has zero increments")
            }
            Self::WindowExhausted {
                outstanding,
                requested,
            } => write!(
                formatter,
                "guest timeline reservation exceeds the unambiguous window: outstanding={outstanding} requested={requested}"
            ),
            Self::LogicalPositionExhausted => {
                formatter.write_str("guest timeline logical position is exhausted")
            }
            Self::ReservationIdentityExhausted => {
                formatter.write_str("guest timeline reservation identities are exhausted")
            }
            Self::ResourceExhausted => {
                formatter.write_str("host resources for guest timeline reservations are exhausted")
            }
        }
    }
}

impl std::error::Error for TimelineReservationError {}

/// Failure to apply an immediate frontend-owned syncpoint increment.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TimelineIncrementError {
    WrongOwner(OwnerMismatch),
    PendingReservation { reservation: u64 },
    Reservation(TimelineReservationError),
    Advance(TimelineAdvanceError),
}

impl Display for TimelineIncrementError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WrongOwner(error) => error.fmt(formatter),
            Self::PendingReservation { reservation } => write!(
                formatter,
                "immediate guest syncpoint increment would overtake reservation={reservation}"
            ),
            Self::Reservation(error) => error.fmt(formatter),
            Self::Advance(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for TimelineIncrementError {}

/// Failure to publish one reserved timeline point.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TimelineAdvanceError {
    WrongOwner(OwnerMismatch),
    WrongSyncpoint {
        expected: GuestSyncpointId,
        observed: GuestSyncpointId,
    },
    WrongInstance {
        expected: TimelineInstanceId,
        observed: TimelineInstanceId,
    },
    UnknownReservation {
        reservation: u64,
    },
    OutOfOrder {
        expected: u64,
        observed: u64,
    },
}

impl Display for TimelineAdvanceError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WrongOwner(error) => error.fmt(formatter),
            Self::WrongSyncpoint { expected, observed } => write!(
                formatter,
                "guest timeline reservation belongs to a different syncpoint: expected {expected} observed {observed}"
            ),
            Self::WrongInstance { expected, observed } => write!(
                formatter,
                "guest timeline reservation belongs to a stale timeline instance: expected {expected} observed {observed}"
            ),
            Self::UnknownReservation { reservation } => {
                write!(
                    formatter,
                    "guest timeline reservation is not pending: reservation={reservation}"
                )
            }
            Self::OutOfOrder { expected, observed } => write!(
                formatter,
                "guest timeline reservation completed out of order: expected={expected} observed={observed}"
            ),
        }
    }
}

impl std::error::Error for TimelineAdvanceError {}

#[cfg(test)]
mod tests {
    use super::*;

    const ID: GuestSyncpointId = GuestSyncpointId::new(5);
    const INSTANCE: TimelineInstanceId = TimelineInstanceId::new(9);
    const OWNER: TimelineOwnerId = TimelineOwnerId::new(0x22);

    #[test]
    fn modular_comparison_handles_rollover_and_rejects_half_range() {
        let before_wrap = GuestSyncpointValue::new(u32::MAX);
        let after_wrap = GuestSyncpointValue::new(0);

        assert_eq!(
            after_wrap.checked_cmp(before_wrap),
            Ok(std::cmp::Ordering::Greater)
        );
        assert!(after_wrap.has_reached(before_wrap));
        assert!(!GuestSyncpointValue::new(0).has_reached(GuestSyncpointValue::new(1 << 31)));
        assert_eq!(
            before_wrap.checked_cmp(after_wrap),
            Ok(std::cmp::Ordering::Less)
        );
        assert_eq!(
            GuestSyncpointValue::new(0).checked_cmp(GuestSyncpointValue::new(1 << 31)),
            Err(SyncpointComparisonError {
                left: GuestSyncpointValue::new(0),
                right: GuestSyncpointValue::new(1 << 31),
            })
        );
    }

    #[test]
    fn immediate_increment_cannot_overtake_reserved_work() {
        let mut timeline = GuestTimeline::new(ID, INSTANCE, OWNER, GuestSyncpointValue::new(9));
        assert_eq!(
            timeline.increment_immediate(OWNER).unwrap().value().get(),
            10
        );
        let pending = timeline.reserve(OWNER, 2).unwrap();
        assert_eq!(
            timeline.increment_immediate(OWNER),
            Err(TimelineIncrementError::PendingReservation {
                reservation: pending.reservation_id(),
            })
        );
    }

    #[test]
    fn reservations_are_monotonic_across_guest_counter_rollover() {
        let mut timeline =
            GuestTimeline::new(ID, INSTANCE, OWNER, GuestSyncpointValue::new(u32::MAX - 1));
        let first = timeline.reserve(OWNER, 1).unwrap();
        let second = timeline.reserve(OWNER, 2).unwrap();

        assert_eq!(first.increments(), 1);
        assert_eq!(second.increments(), 2);
        assert_eq!(first.point().value().get(), u32::MAX);
        assert_eq!(second.point().value().get(), 1);
        assert_eq!(second.checked_cmp(&first), Ok(std::cmp::Ordering::Greater));
        assert_eq!(timeline.current_point().value().get(), u32::MAX - 1);
        assert_eq!(timeline.latest_reserved_point(), second.point());
        assert_eq!(timeline.outstanding_increments(), 3);

        assert_eq!(timeline.advance(OWNER, &first), Ok(first.point()));
        assert_eq!(timeline.advance(OWNER, &second), Ok(second.point()));
        assert_eq!(timeline.outstanding_increments(), 0);
        assert_eq!(timeline.reservation_count(), 0);
    }

    #[test]
    fn ownership_and_submission_order_are_enforced() {
        let other = TimelineOwnerId::new(0x23);
        let mut timeline = GuestTimeline::new(ID, INSTANCE, OWNER, GuestSyncpointValue::new(10));
        assert_eq!(
            timeline.reserve(other, 1),
            Err(TimelineReservationError::WrongOwner(OwnerMismatch {
                expected: OWNER,
                observed: other,
            }))
        );

        let first = timeline.reserve(OWNER, 1).unwrap();
        let second = timeline.reserve(OWNER, 1).unwrap();
        assert_eq!(
            timeline.advance(OWNER, &second),
            Err(TimelineAdvanceError::OutOfOrder {
                expected: first.reservation_id(),
                observed: second.reservation_id(),
            })
        );
        assert_eq!(timeline.advance(OWNER, &first), Ok(first.point()));
        assert_eq!(
            timeline.advance(OWNER, &first),
            Err(TimelineAdvanceError::UnknownReservation {
                reservation: first.reservation_id(),
            })
        );
    }

    #[test]
    fn reservations_cannot_enter_the_ambiguous_window() {
        let mut timeline = GuestTimeline::new(ID, INSTANCE, OWNER, GuestSyncpointValue::new(0));
        let first = timeline.reserve(OWNER, u32::MAX >> 1).unwrap();
        assert_eq!(first.point().value().get(), u32::MAX >> 1);
        assert_eq!(
            timeline.reserve(OWNER, 1),
            Err(TimelineReservationError::WindowExhausted {
                outstanding: u64::from(u32::MAX >> 1),
                requested: 1,
            })
        );
        assert_eq!(
            GuestSyncpointValue::new(0).checked_cmp(first.point().value()),
            Ok(std::cmp::Ordering::Less)
        );
    }

    #[test]
    fn points_on_different_syncpoints_are_not_comparable() {
        let mut left = GuestTimeline::new(ID, INSTANCE, OWNER, GuestSyncpointValue::new(0));
        let mut right = GuestTimeline::new(
            GuestSyncpointId::new(6),
            TimelineInstanceId::new(10),
            OWNER,
            GuestSyncpointValue::new(0),
        );
        let left = left.reserve(OWNER, 1).unwrap();
        let right = right.reserve(OWNER, 1).unwrap();

        assert_eq!(
            left.checked_cmp(&right),
            Err(TimelinePointComparisonError {
                left_syncpoint: ID,
                left_instance: INSTANCE,
                right_syncpoint: GuestSyncpointId::new(6),
                right_instance: TimelineInstanceId::new(10),
            })
        );
    }

    #[test]
    fn stale_reservation_from_recreated_syncpoint_is_rejected() {
        let mut old = GuestTimeline::new(ID, INSTANCE, OWNER, GuestSyncpointValue::new(0));
        let stale = old.reserve(OWNER, 1).unwrap();
        let replacement_instance = TimelineInstanceId::new(10);
        let mut replacement =
            GuestTimeline::new(ID, replacement_instance, OWNER, GuestSyncpointValue::new(0));
        let current = replacement.reserve(OWNER, 1).unwrap();

        assert_eq!(
            replacement.advance(OWNER, &stale),
            Err(TimelineAdvanceError::WrongInstance {
                expected: replacement_instance,
                observed: INSTANCE,
            })
        );
        assert_eq!(replacement.advance(OWNER, &current), Ok(current.point()));
    }
}
