//! Host-independent propagation from backend completion to guest timelines.

use std::collections::VecDeque;
use std::fmt::{Display, Formatter};
use std::sync::Arc;

use nixe_memory::{
    CanonicalBackingRange, DeviceAccessDeclaration, DeviceVisibilityPoint, VisibilityCoordinator,
    VisibilityError,
};

use crate::{
    BackendSubmissionToken, FrontendSubmissionId, GuestTimeline, GuestTimelinePoint,
    HostCompletion, ReservedTimelinePoint, TimelineAdvanceError, TimelineOwnerId,
    VisibilityCompletion,
};

/// One canonical range written by an accepted backend submission.
///
/// The range retains canonical page identities rather than CPU addresses or
/// host pointers. The injected coordinator owns any later upload, download,
/// cache, or coherent-memory operation needed by `nixe-memory`.
#[derive(Clone)]
pub struct SubmissionWrite {
    range: CanonicalBackingRange,
    declaration: DeviceAccessDeclaration,
    coordinator: Arc<dyn VisibilityCoordinator>,
}

impl SubmissionWrite {
    /// Validates and retains one declared device write.
    pub fn new(
        range: CanonicalBackingRange,
        declaration: DeviceAccessDeclaration,
        coordinator: Arc<dyn VisibilityCoordinator>,
    ) -> Result<Self, CompletionRegistrationError> {
        if !declaration.kind().writes() {
            return Err(CompletionRegistrationError::DeclarationDoesNotWrite);
        }
        Ok(Self {
            range,
            declaration,
            coordinator,
        })
    }

    /// Returns the point at which the device-produced contents become valid.
    #[must_use]
    pub fn visibility_point(&self) -> DeviceVisibilityPoint {
        self.declaration
            .cpu_visible_at()
            .expect("validated submission write has a CPU visibility point")
    }
}

/// Completion metadata retained for one accepted neutral submission.
pub struct CompletionSubmission {
    frontend: FrontendSubmissionId,
    backend: BackendSubmissionToken,
    reservation: ReservedTimelinePoint,
    visibility_point: DeviceVisibilityPoint,
    writes: Box<[SubmissionWrite]>,
}

impl CompletionSubmission {
    /// Creates a submission whose writes all belong to one visibility point.
    pub fn new(
        frontend: FrontendSubmissionId,
        backend: BackendSubmissionToken,
        reservation: ReservedTimelinePoint,
        visibility_point: DeviceVisibilityPoint,
        writes: Vec<SubmissionWrite>,
    ) -> Result<Self, CompletionRegistrationError> {
        if let Some(write) = writes
            .iter()
            .find(|write| write.visibility_point() != visibility_point)
        {
            return Err(CompletionRegistrationError::VisibilityPointMismatch {
                expected: visibility_point,
                observed: write.visibility_point(),
            });
        }
        Ok(Self {
            frontend,
            backend,
            reservation,
            visibility_point,
            writes: writes.into_boxed_slice(),
        })
    }

    /// Returns the frontend identity used by diagnostics and capture.
    #[must_use]
    pub const fn frontend(&self) -> FrontendSubmissionId {
        self.frontend
    }

    /// Returns the opaque token owned by the selected backend.
    #[must_use]
    pub const fn backend(&self) -> BackendSubmissionToken {
        self.backend
    }

    /// Returns the guest point reserved for this submission.
    #[must_use]
    pub fn guest_point(&self) -> GuestTimelinePoint {
        self.reservation.point()
    }
}

/// Host-independent source of backend completion observations.
///
/// Implementations may query a real host queue or a deterministic test
/// timeline. They report only completion of the supplied opaque token and
/// cannot mutate guest timelines or canonical visibility state.
pub trait BackendCompletionSource {
    /// Returns whether the backend has completed `submission`.
    fn has_completed(
        &mut self,
        submission: BackendSubmissionToken,
    ) -> Result<bool, BackendCompletionError>;
}

/// Failure while asking the selected backend for completion state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BackendCompletionError(Box<str>);

impl BackendCompletionError {
    /// Creates a backend-independent diagnostic without exposing host objects.
    #[must_use]
    pub fn new(message: impl Into<Box<str>>) -> Self {
        Self(message.into())
    }
}

impl Display for BackendCompletionError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(formatter)
    }
}

impl std::error::Error for BackendCompletionError {}

struct PendingCompletion {
    submission: CompletionSubmission,
    host_completion: Option<HostCompletion>,
}

/// Ordered completion state for submissions targeting one guest timeline.
///
/// Backend tokens may complete out of order, but publication is constrained by
/// reservation order. A caller signals Horizon events only from the returned
/// [`PublishedSubmission`], after this coordinator has established memory
/// visibility and advanced the guest timeline.
pub struct SubmissionCompletionQueue {
    owner: TimelineOwnerId,
    pending: VecDeque<PendingCompletion>,
}

impl SubmissionCompletionQueue {
    /// Creates an empty queue owned by the same identity as its guest timeline.
    #[must_use]
    pub const fn new(owner: TimelineOwnerId) -> Self {
        Self {
            owner,
            pending: VecDeque::new(),
        }
    }

    /// Returns the number of submissions not yet published to the guest.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Registers one submission without claiming backend or guest progress.
    pub fn enqueue(
        &mut self,
        submission: CompletionSubmission,
    ) -> Result<(), CompletionRegistrationError> {
        if submission.reservation.owner() != self.owner {
            return Err(CompletionRegistrationError::WrongOwner {
                expected: self.owner,
                observed: submission.reservation.owner(),
            });
        }
        if self
            .pending
            .iter()
            .any(|pending| pending.submission.frontend == submission.frontend)
        {
            return Err(CompletionRegistrationError::DuplicateFrontendSubmission(
                submission.frontend,
            ));
        }
        if self
            .pending
            .iter()
            .any(|pending| pending.submission.backend == submission.backend)
        {
            return Err(CompletionRegistrationError::DuplicateBackendSubmission(
                submission.backend,
            ));
        }
        if let Some(previous) = self.pending.back() {
            let ordering = previous
                .submission
                .reservation
                .checked_cmp(&submission.reservation)
                .map_err(|_| CompletionRegistrationError::DifferentTimeline)?;
            if ordering != std::cmp::Ordering::Less {
                return Err(CompletionRegistrationError::ReservationOutOfOrder);
            }
        }
        self.pending
            .try_reserve(1)
            .map_err(|_| CompletionRegistrationError::ResourceExhausted)?;
        self.pending.push_back(PendingCompletion {
            submission,
            host_completion: None,
        });
        Ok(())
    }

    /// Samples every pending token, retaining out-of-order host completions.
    ///
    /// This operation never performs visibility work and never advances the
    /// guest timeline.
    pub fn observe_backend(
        &mut self,
        source: &mut impl BackendCompletionSource,
    ) -> Result<(), CompletionPropagationError> {
        for pending in &mut self.pending {
            if pending.host_completion.is_none()
                && source
                    .has_completed(pending.submission.backend)
                    .map_err(CompletionPropagationError::Backend)?
            {
                pending.host_completion = Some(HostCompletion::new(pending.submission.backend));
            }
        }
        Ok(())
    }

    /// Publishes the oldest host-complete submission, if one is ready.
    ///
    /// Every declared write is first transferred into `nixe-memory`'s
    /// visibility state. Only after all transitions succeed is the reservation
    /// advanced. A failed transition invalidates every declared write range and
    /// leaves the guest fence incomplete.
    pub fn publish_next(
        &mut self,
        timeline: &mut GuestTimeline,
    ) -> Result<Option<PublishedSubmission>, CompletionPropagationError> {
        if timeline.owner() != self.owner {
            return Err(CompletionPropagationError::WrongTimelineOwner {
                expected: self.owner,
                observed: timeline.owner(),
            });
        }
        let Some(pending) = self.pending.front() else {
            return Ok(None);
        };
        let Some(host_completion) = pending.host_completion else {
            return Ok(None);
        };

        timeline
            .validate_advance(self.owner, &pending.submission.reservation)
            .map_err(CompletionPropagationError::Timeline)?;
        for write in &pending.submission.writes {
            if let Err(error) = write
                .range
                .complete_device_write(write.declaration, Arc::clone(&write.coordinator))
            {
                let invalidation = invalidate_writes(&pending.submission.writes).err();
                return Err(CompletionPropagationError::Visibility {
                    submission: pending.submission.frontend,
                    error,
                    invalidation,
                });
            }
        }
        let visibility = VisibilityCompletion::new(pending.submission.visibility_point);
        let guest_point = timeline
            .advance(self.owner, &pending.submission.reservation)
            .map_err(CompletionPropagationError::Timeline)?;
        let frontend = pending.submission.frontend;
        let backend = pending.submission.backend;
        self.pending.pop_front();
        Ok(Some(PublishedSubmission {
            frontend,
            backend,
            host_completion,
            visibility,
            guest_point,
        }))
    }

    /// Drops every unobservable submission during owner teardown.
    ///
    /// The owning frontend must tear down the associated timeline at the same
    /// boundary; retained ranges and backend tokens are released here without
    /// fabricating guest progress.
    pub fn clear(&mut self) -> usize {
        let removed = self.pending.len();
        self.pending.clear();
        removed
    }
}

fn invalidate_writes(writes: &[SubmissionWrite]) -> Result<(), VisibilityError> {
    let mut first_error = None;
    for write in writes {
        if let Err(error) = write.range.invalidate_visibility()
            && first_error.is_none()
        {
            first_error = Some(error);
        }
    }
    if let Some(error) = first_error {
        Err(error)
    } else {
        Ok(())
    }
}

/// Evidence returned only after all three completion domains were crossed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PublishedSubmission {
    frontend: FrontendSubmissionId,
    backend: BackendSubmissionToken,
    host_completion: HostCompletion,
    visibility: VisibilityCompletion,
    guest_point: GuestTimelinePoint,
}

impl PublishedSubmission {
    #[must_use]
    pub const fn frontend(self) -> FrontendSubmissionId {
        self.frontend
    }

    #[must_use]
    pub const fn backend(self) -> BackendSubmissionToken {
        self.backend
    }

    #[must_use]
    pub const fn host_completion(self) -> HostCompletion {
        self.host_completion
    }

    #[must_use]
    pub const fn visibility(self) -> VisibilityCompletion {
        self.visibility
    }

    #[must_use]
    pub const fn guest_point(self) -> GuestTimelinePoint {
        self.guest_point
    }
}

/// Invalid submission registration at the neutral completion boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompletionRegistrationError {
    DeclarationDoesNotWrite,
    VisibilityPointMismatch {
        expected: DeviceVisibilityPoint,
        observed: DeviceVisibilityPoint,
    },
    WrongOwner {
        expected: TimelineOwnerId,
        observed: TimelineOwnerId,
    },
    DuplicateFrontendSubmission(FrontendSubmissionId),
    DuplicateBackendSubmission(BackendSubmissionToken),
    DifferentTimeline,
    ReservationOutOfOrder,
    ResourceExhausted,
}

impl Display for CompletionRegistrationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DeclarationDoesNotWrite => {
                formatter.write_str("submission visibility declaration does not write")
            }
            Self::VisibilityPointMismatch { expected, observed } => write!(
                formatter,
                "submission write visibility point mismatch: expected {expected} observed {observed}"
            ),
            Self::WrongOwner { expected, observed } => write!(
                formatter,
                "submission timeline owner mismatch: expected {expected} observed {observed}"
            ),
            Self::DuplicateFrontendSubmission(submission) => {
                write!(formatter, "duplicate {submission}")
            }
            Self::DuplicateBackendSubmission(submission) => {
                write!(formatter, "duplicate {submission}")
            }
            Self::DifferentTimeline => {
                formatter.write_str("submission reservations belong to different timelines")
            }
            Self::ReservationOutOfOrder => {
                formatter.write_str("submission reservations are not in increasing order")
            }
            Self::ResourceExhausted => {
                formatter.write_str("host resources for completion registration are exhausted")
            }
        }
    }
}

impl std::error::Error for CompletionRegistrationError {}

/// Failure before a host-complete submission can become guest-observable.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CompletionPropagationError {
    Backend(BackendCompletionError),
    WrongTimelineOwner {
        expected: TimelineOwnerId,
        observed: TimelineOwnerId,
    },
    Visibility {
        submission: FrontendSubmissionId,
        error: VisibilityError,
        invalidation: Option<VisibilityError>,
    },
    Timeline(TimelineAdvanceError),
}

impl Display for CompletionPropagationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Backend(error) => write!(formatter, "backend completion query failed: {error}"),
            Self::WrongTimelineOwner { expected, observed } => write!(
                formatter,
                "completion queue timeline owner mismatch: expected {expected} observed {observed}"
            ),
            Self::Visibility {
                submission,
                error,
                invalidation,
            } => {
                write!(
                    formatter,
                    "{submission} visibility propagation failed: {error}"
                )?;
                if let Some(invalidation) = invalidation {
                    write!(
                        formatter,
                        "; range invalidation also failed: {invalidation}"
                    )?;
                }
                Ok(())
            }
            Self::Timeline(error) => {
                write!(formatter, "guest timeline publication failed: {error}")
            }
        }
    }
}

impl std::error::Error for CompletionPropagationError {}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::sync::Mutex;

    use nixe_memory::{
        CanonicalAllocation, CpuVisibilityRequest, DeviceVisibilityRequest, MemoryPermissions,
        NonCpuDeviceId, VisibilityCoordinatorError, VisibilityState,
    };

    use super::*;
    use crate::{GuestSyncpointId, GuestSyncpointValue, TimelineInstanceId};

    const OWNER: TimelineOwnerId = TimelineOwnerId::new(4);
    const POINT: DeviceVisibilityPoint = DeviceVisibilityPoint::new(12);

    #[derive(Default)]
    struct ManualCompletionDriver {
        completed: BTreeSet<BackendSubmissionToken>,
    }

    impl ManualCompletionDriver {
        fn complete(&mut self, submission: BackendSubmissionToken) {
            self.completed.insert(submission);
        }
    }

    impl BackendCompletionSource for ManualCompletionDriver {
        fn has_completed(
            &mut self,
            submission: BackendSubmissionToken,
        ) -> Result<bool, BackendCompletionError> {
            Ok(self.completed.contains(&submission))
        }
    }

    #[derive(Default)]
    struct RecordingVisibility {
        downloads: Mutex<Vec<CpuVisibilityRequest>>,
    }

    impl VisibilityCoordinator for RecordingVisibility {
        fn make_device_visible(
            &self,
            _request: DeviceVisibilityRequest,
            _canonical_bytes: &[u8],
        ) -> Result<(), VisibilityCoordinatorError> {
            Ok(())
        }

        fn make_cpu_visible(
            &self,
            request: CpuVisibilityRequest,
        ) -> Result<Box<[u8]>, VisibilityCoordinatorError> {
            self.downloads.lock().unwrap().push(request);
            Ok(vec![0x5a; request.size].into_boxed_slice())
        }
    }

    fn timeline(initial: u32) -> GuestTimeline {
        GuestTimeline::new(
            GuestSyncpointId::new(2),
            TimelineInstanceId::new(3),
            OWNER,
            GuestSyncpointValue::new(initial),
        )
    }

    fn submission(
        timeline: &mut GuestTimeline,
        frontend: u64,
        backend: u64,
        writes: Vec<SubmissionWrite>,
    ) -> CompletionSubmission {
        let reservation = timeline.reserve(OWNER, 1).unwrap();
        CompletionSubmission::new(
            FrontendSubmissionId::new(frontend),
            BackendSubmissionToken::new(backend),
            reservation,
            POINT,
            writes,
        )
        .unwrap()
    }

    #[test]
    fn incomplete_host_work_cannot_publish_visibility_or_guest_progress() {
        let allocation = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
        let range = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let visibility: Arc<dyn VisibilityCoordinator> = Arc::new(RecordingVisibility::default());
        let declaration = DeviceAccessDeclaration::write(
            NonCpuDeviceId::new(8),
            DeviceVisibilityPoint::new(11),
            POINT,
        )
        .unwrap();
        let write = SubmissionWrite::new(range.clone(), declaration, visibility).unwrap();
        let mut timeline = timeline(0);
        let mut queue = SubmissionCompletionQueue::new(OWNER);
        queue
            .enqueue(submission(&mut timeline, 1, 10, vec![write]))
            .unwrap();
        let mut backend = ManualCompletionDriver::default();

        queue.observe_backend(&mut backend).unwrap();
        assert_eq!(queue.publish_next(&mut timeline), Ok(None));
        assert_eq!(
            timeline.current_point().value(),
            GuestSyncpointValue::new(0)
        );
        assert_eq!(
            range.segments()[0].visibility_state(),
            VisibilityState::Clean
        );

        backend.complete(BackendSubmissionToken::new(10));
        queue.observe_backend(&mut backend).unwrap();
        assert_eq!(
            timeline.current_point().value(),
            GuestSyncpointValue::new(0)
        );
        assert_eq!(
            range.segments()[0].visibility_state(),
            VisibilityState::Clean
        );
        let published = queue.publish_next(&mut timeline).unwrap().unwrap();
        assert_eq!(
            published.host_completion().submission(),
            published.backend()
        );
        assert_eq!(published.visibility().point(), POINT);
        assert_eq!(published.guest_point().value(), GuestSyncpointValue::new(1));
        assert_eq!(
            range.segments()[0].visibility_state(),
            VisibilityState::GpuNewer {
                device: NonCpuDeviceId::new(8),
                visible_at: POINT,
            }
        );
    }

    #[test]
    fn out_of_order_host_completion_waits_for_guest_reservation_order_across_wrap() {
        let mut timeline = timeline(u32::MAX - 1);
        let mut queue = SubmissionCompletionQueue::new(OWNER);
        queue
            .enqueue(submission(&mut timeline, 1, 10, Vec::new()))
            .unwrap();
        queue
            .enqueue(submission(&mut timeline, 2, 11, Vec::new()))
            .unwrap();
        let mut backend = ManualCompletionDriver::default();

        backend.complete(BackendSubmissionToken::new(11));
        queue.observe_backend(&mut backend).unwrap();
        assert_eq!(queue.publish_next(&mut timeline), Ok(None));
        assert_eq!(
            timeline.current_point().value(),
            GuestSyncpointValue::new(u32::MAX - 1)
        );

        backend.complete(BackendSubmissionToken::new(10));
        queue.observe_backend(&mut backend).unwrap();
        assert_eq!(
            queue
                .publish_next(&mut timeline)
                .unwrap()
                .unwrap()
                .guest_point()
                .value(),
            GuestSyncpointValue::new(u32::MAX)
        );
        assert_eq!(
            queue
                .publish_next(&mut timeline)
                .unwrap()
                .unwrap()
                .guest_point()
                .value(),
            GuestSyncpointValue::new(0)
        );
    }

    #[test]
    fn visibility_failure_keeps_the_guest_fence_incomplete_and_invalidates_writes() {
        let allocation = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
        allocation.write(0, &[1]).unwrap();
        let range = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let visibility: Arc<dyn VisibilityCoordinator> = Arc::new(RecordingVisibility::default());
        let declaration = DeviceAccessDeclaration::write(
            NonCpuDeviceId::new(8),
            DeviceVisibilityPoint::new(11),
            POINT,
        )
        .unwrap();
        let write = SubmissionWrite::new(range.clone(), declaration, visibility).unwrap();
        let mut timeline = timeline(0);
        let mut queue = SubmissionCompletionQueue::new(OWNER);
        queue
            .enqueue(submission(&mut timeline, 1, 10, vec![write]))
            .unwrap();
        let mut backend = ManualCompletionDriver::default();
        backend.complete(BackendSubmissionToken::new(10));
        queue.observe_backend(&mut backend).unwrap();

        assert!(matches!(
            queue.publish_next(&mut timeline),
            Err(CompletionPropagationError::Visibility { .. })
        ));
        assert_eq!(
            timeline.current_point().value(),
            GuestSyncpointValue::new(0)
        );
        assert_eq!(queue.pending_count(), 1);
        assert_eq!(
            range.segments()[0].visibility_state(),
            VisibilityState::Invalid
        );
    }

    #[test]
    fn teardown_drops_completed_and_incomplete_work_without_advancing_the_guest() {
        let mut timeline = timeline(7);
        let mut queue = SubmissionCompletionQueue::new(OWNER);
        queue
            .enqueue(submission(&mut timeline, 1, 10, Vec::new()))
            .unwrap();
        queue
            .enqueue(submission(&mut timeline, 2, 11, Vec::new()))
            .unwrap();
        let mut backend = ManualCompletionDriver::default();
        backend.complete(BackendSubmissionToken::new(10));
        queue.observe_backend(&mut backend).unwrap();

        assert_eq!(queue.clear(), 2);
        assert_eq!(queue.pending_count(), 0);
        assert_eq!(
            timeline.current_point().value(),
            GuestSyncpointValue::new(7)
        );
    }
}
