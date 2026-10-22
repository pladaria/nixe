//! Deterministic Switch 1 frontend submission scheduling.
//!
//! The initial policy deliberately serializes work in acceptance order. Queue
//! identities are independent of channels so the representation supports many
//! channels without accidentally granting any channel exclusive ownership.
//! This layer owns no Horizon descriptor and cannot signal guest completion.

use std::collections::{HashSet, VecDeque};
use std::fmt::{Display, Formatter};

use nixe_gpu::{FrontendSubmissionId, GuestTimelinePoint, ReservedTimelinePoint};

use crate::{
    MaxwellAddressSpaceId, MaxwellChannelId, MaxwellChannelSchedulingPolicy,
    MaxwellFrontendDispatchError, MaxwellGpfifoCapture, MaxwellGpfifoSourceError,
    MaxwellGpfifoSourceLocation, MaxwellGpuMemory, MaxwellValidatedGpfifoSubmission,
};

/// Global acceptance order assigned by one deterministic scheduler.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct MaxwellSchedulerSequence(u64);

impl MaxwellSchedulerSequence {
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl Display for MaxwellSchedulerSequence {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "scheduler-sequence=0x{:016x}", self.0)
    }
}

/// Furthest ordering boundary crossed by a scheduled submission.
///
/// Cache maintenance, decoded resource visibility, GPU writes, host
/// completion, and fence publication intentionally remain later distinct
/// stages. T6 cannot cross them before T7 identifies packet semantics and
/// resource accesses.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellSubmissionOrderingStage {
    /// CPU-produced command bytes were resolved to retained canonical ranges.
    CommandSourcesRetained,
    /// The complete retained source was selected for frontend packet decoding.
    FrontendDispatched,
    /// Verified packet cache-maintenance operations have completed.
    CacheMaintenanceComplete,
    /// Every decoded resource read is visible to the selected device.
    DeviceReadsVisible,
    /// Backend execution completed, without yet publishing written memory.
    GpuExecutionComplete,
    /// GPU-produced bytes reached their declared canonical visibility points.
    GpuWritesVisible,
    /// The reserved T5 point was published after every preceding stage.
    FencePublished,
    /// Queued ownership was released without claiming any later stage.
    Cancelled,
}

impl Display for MaxwellSubmissionOrderingStage {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::CommandSourcesRetained => "command-sources-retained",
            Self::FrontendDispatched => "frontend-dispatched",
            Self::CacheMaintenanceComplete => "cache-maintenance-complete",
            Self::DeviceReadsVisible => "device-reads-visible",
            Self::GpuExecutionComplete => "gpu-execution-complete",
            Self::GpuWritesVisible => "gpu-writes-visible",
            Self::FencePublished => "fence-published",
            Self::Cancelled => "cancelled",
        })
    }
}

/// One accepted submission and its explicit synchronization ownership.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaxwellScheduledSubmission {
    sequence: MaxwellSchedulerSequence,
    channel: MaxwellChannelId,
    dependency: Option<GuestTimelinePoint>,
    completion: Option<ReservedTimelinePoint>,
    stage: MaxwellSubmissionOrderingStage,
    submission: MaxwellValidatedGpfifoSubmission,
}

impl MaxwellScheduledSubmission {
    #[must_use]
    pub const fn sequence(&self) -> MaxwellSchedulerSequence {
        self.sequence
    }

    #[must_use]
    pub const fn channel(&self) -> MaxwellChannelId {
        self.channel
    }

    #[must_use]
    pub const fn frontend(&self) -> FrontendSubmissionId {
        self.submission.frontend()
    }

    #[must_use]
    pub const fn dependency(&self) -> Option<GuestTimelinePoint> {
        self.dependency
    }

    #[must_use]
    pub const fn completion(&self) -> Option<&ReservedTimelinePoint> {
        self.completion.as_ref()
    }

    #[must_use]
    pub const fn stage(&self) -> MaxwellSubmissionOrderingStage {
        self.stage
    }

    #[must_use]
    pub const fn submission(&self) -> &MaxwellValidatedGpfifoSubmission {
        &self.submission
    }
}

/// Complete work transferred to the Maxwell packet consumer.
///
/// Owning the scheduled object keeps its mapping snapshots, canonical backing,
/// dependency, and unreported fence reservation alive at the boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaxwellFrontendDispatch {
    scheduled: MaxwellScheduledSubmission,
}

impl MaxwellFrontendDispatch {
    #[must_use]
    pub const fn scheduled(&self) -> &MaxwellScheduledSubmission {
        &self.scheduled
    }

    #[must_use]
    pub fn capture(&self) -> MaxwellGpfifoCapture {
        self.scheduled.submission.capture()
    }

    /// Stops at the first packet boundary without decoding or executing it.
    pub fn unsupported_boundary(
        self,
    ) -> Result<MaxwellFrontendDispatchBoundary, MaxwellGpfifoSourceError> {
        let location = self.scheduled.submission.first_packet_location()?;
        Ok(match location {
            Some(location) => MaxwellFrontendDispatchBoundary::FirstPacket {
                dispatch: Box::new(self),
                location,
            },
            None => MaxwellFrontendDispatchBoundary::EmptySubmission {
                dispatch: Box::new(self),
            },
        })
    }
}

/// The exact frontend boundary which T7 must consume next.
#[derive(Clone, Debug)]
pub enum MaxwellFrontendDispatchBoundary {
    FirstPacket {
        dispatch: Box<MaxwellFrontendDispatch>,
        location: MaxwellGpfifoSourceLocation,
    },
    EmptySubmission {
        dispatch: Box<MaxwellFrontendDispatch>,
    },
    /// T7 decoded and dispatched the retained command stream and stopped at
    /// its precise semantic boundary. The scheduled work remains owned here,
    /// so its completion reservation cannot be published accidentally.
    Frontend {
        dispatch: Box<MaxwellFrontendDispatch>,
        failure: MaxwellFrontendDispatchError,
    },
}

impl MaxwellFrontendDispatchBoundary {
    #[must_use]
    pub const fn dispatch(&self) -> &MaxwellFrontendDispatch {
        match self {
            Self::FirstPacket { dispatch, .. }
            | Self::EmptySubmission { dispatch }
            | Self::Frontend { dispatch, .. } => dispatch,
        }
    }

    #[must_use]
    pub fn first_packet(&self) -> Option<MaxwellGpfifoSourceLocation> {
        match self {
            Self::FirstPacket { location, .. } => Some(*location),
            Self::EmptySubmission { .. } => None,
            Self::Frontend { dispatch, .. } => dispatch
                .scheduled()
                .submission()
                .first_packet_location()
                .ok()
                .flatten(),
        }
    }

    #[must_use]
    pub const fn frontend_failure(&self) -> Option<&MaxwellFrontendDispatchError> {
        match self {
            Self::Frontend { failure, .. } => Some(failure),
            Self::FirstPacket { .. } | Self::EmptySubmission { .. } => None,
        }
    }
}

impl Display for MaxwellFrontendDispatchBoundary {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        let scheduled = self.dispatch().scheduled();
        write!(
            formatter,
            "{} {} {} stage={} ",
            scheduled.sequence(),
            scheduled.channel(),
            scheduled.frontend(),
            scheduled.stage()
        )?;
        if let Some(dependency) = scheduled.dependency() {
            write!(formatter, "dependency={dependency} ")?;
        }
        if let Some(completion) = scheduled.completion() {
            write!(formatter, "completion=[{completion}] ")?;
        }
        match self {
            Self::FirstPacket { location, .. } => {
                write!(formatter, "first-unsupported-packet=[{location}]")
            }
            Self::EmptySubmission { .. } => {
                formatter.write_str("empty-submission completion semantics are unavailable")
            }
            Self::Frontend { failure, .. } => write!(formatter, "frontend-failure=[{failure}]"),
        }
    }
}

/// Failure before a submission can cross the packet-consumer boundary.
#[derive(Clone, Debug)]
pub enum MaxwellScheduleError {
    UnsupportedPolicy,
    ChannelMismatch {
        channel: MaxwellChannelId,
        submission: MaxwellChannelId,
    },
    DuplicateFrontendSubmission(FrontendSubmissionId),
    MissingWaitDependency,
    UnexpectedWaitDependency,
    MissingFenceReservation,
    UnexpectedFenceReservation,
    WrongFenceSyncpoint,
    PendingDependency(GuestTimelinePoint),
    Source(MaxwellGpfifoSourceError),
    SequenceExhausted,
    ResourceExhausted,
}

impl Display for MaxwellScheduleError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedPolicy => {
                formatter.write_str("channel scheduling policy is unsupported")
            }
            Self::ChannelMismatch {
                channel,
                submission,
            } => write!(
                formatter,
                "submission channel does not match scheduler input: channel={channel} submission={submission}"
            ),
            Self::DuplicateFrontendSubmission(submission) => {
                write!(formatter, "duplicate {submission}")
            }
            Self::MissingWaitDependency => {
                formatter.write_str("fence-wait submission has no typed dependency")
            }
            Self::UnexpectedWaitDependency => {
                formatter.write_str("submission without fence-wait has a dependency")
            }
            Self::MissingFenceReservation => formatter
                .write_str("completion-producing submission has no reserved timeline point"),
            Self::UnexpectedFenceReservation => formatter
                .write_str("submission without a completion increment has a fence reservation"),
            Self::WrongFenceSyncpoint => formatter
                .write_str("submission fence reservation does not belong to the channel syncpoint"),
            Self::PendingDependency(point) => write!(
                formatter,
                "submission dependency has not been reached: {point}"
            ),
            Self::Source(error) => error.fmt(formatter),
            Self::SequenceExhausted => {
                formatter.write_str("scheduler sequence identities are exhausted")
            }
            Self::ResourceExhausted => {
                formatter.write_str("host resources for Maxwell scheduling are exhausted")
            }
        }
    }
}

impl std::error::Error for MaxwellScheduleError {}

/// Deterministic FIFO scheduler shared by all channels of one frontend owner.
#[derive(Clone, Debug)]
pub struct MaxwellScheduler {
    next_sequence: u64,
    queued: VecDeque<MaxwellScheduledSubmission>,
    frontend_ids: HashSet<FrontendSubmissionId>,
}

impl Default for MaxwellScheduler {
    fn default() -> Self {
        Self {
            next_sequence: 1,
            queued: VecDeque::new(),
            frontend_ids: HashSet::new(),
        }
    }
}

impl MaxwellScheduler {
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.queued.len()
    }

    #[must_use]
    pub fn next_address_space(&self) -> Option<MaxwellAddressSpaceId> {
        self.queued
            .front()
            .map(|scheduled| scheduled.submission.address_space())
    }

    /// Preflights semantic state and host capacity before a timeline mutates.
    ///
    /// The caller may reserve the T5 fence after this returns. With exclusive
    /// access to the scheduler, the following [`Self::enqueue`] cannot need a
    /// collection allocation or discover a mode mismatch.
    pub fn prepare_enqueue(
        &mut self,
        channel: &crate::MaxwellChannelConfiguration,
        submission: &MaxwellValidatedGpfifoSubmission,
        dependency: Option<GuestTimelinePoint>,
        will_reserve_completion: bool,
    ) -> Result<(), MaxwellScheduleError> {
        self.validate_shape(channel, submission, dependency, will_reserve_completion)?;
        self.next_sequence
            .checked_add(1)
            .ok_or(MaxwellScheduleError::SequenceExhausted)?;
        self.queued
            .try_reserve(1)
            .map_err(|_| MaxwellScheduleError::ResourceExhausted)?;
        self.frontend_ids
            .try_reserve(1)
            .map_err(|_| MaxwellScheduleError::ResourceExhausted)?;
        Ok(())
    }

    /// Enqueues complete work without reading commands or claiming progress.
    pub fn enqueue(
        &mut self,
        channel: &crate::MaxwellChannelConfiguration,
        submission: MaxwellValidatedGpfifoSubmission,
        dependency: Option<GuestTimelinePoint>,
        completion: Option<ReservedTimelinePoint>,
    ) -> Result<MaxwellSchedulerSequence, MaxwellScheduleError> {
        self.validate_enqueue(channel, &submission, dependency, completion.as_ref())?;
        let sequence = MaxwellSchedulerSequence::new(self.next_sequence);
        let next_sequence = self
            .next_sequence
            .checked_add(1)
            .ok_or(MaxwellScheduleError::SequenceExhausted)?;
        self.queued
            .try_reserve(1)
            .map_err(|_| MaxwellScheduleError::ResourceExhausted)?;
        self.frontend_ids
            .try_reserve(1)
            .map_err(|_| MaxwellScheduleError::ResourceExhausted)?;
        self.frontend_ids.insert(submission.frontend());
        self.queued.push_back(MaxwellScheduledSubmission {
            sequence,
            channel: channel.id(),
            dependency,
            completion,
            stage: MaxwellSubmissionOrderingStage::CommandSourcesRetained,
            submission,
        });
        self.next_sequence = next_sequence;
        Ok(sequence)
    }

    /// Dispatches only the globally oldest submission.
    ///
    /// The dependency is sampled by the Horizon timeline owner and supplied as
    /// evidence. Source mappings are then revalidated atomically before the
    /// scheduler transfers ownership to the packet consumer.
    pub fn dispatch_next(
        &mut self,
        dependency_reached: bool,
        address_space: &MaxwellGpuMemory,
    ) -> Result<Option<MaxwellFrontendDispatch>, MaxwellScheduleError> {
        let Some(next) = self.queued.front() else {
            return Ok(None);
        };
        if let Some(dependency) = next.dependency
            && !dependency_reached
        {
            return Err(MaxwellScheduleError::PendingDependency(dependency));
        }
        next.submission
            .validate_sources(address_space)
            .map_err(MaxwellScheduleError::Source)?;
        let mut scheduled = self.queued.pop_front().expect("front was present");
        self.frontend_ids.remove(&scheduled.frontend());
        scheduled.stage = MaxwellSubmissionOrderingStage::FrontendDispatched;
        Ok(Some(MaxwellFrontendDispatch { scheduled }))
    }

    /// Cancels queued work owned by one channel without publishing its fence.
    pub fn cancel_channel(&mut self, channel: MaxwellChannelId) -> Vec<MaxwellScheduledSubmission> {
        let mut cancelled = Vec::new();
        self.queued.retain(|submission| {
            if submission.channel == channel {
                self.frontend_ids.remove(&submission.frontend());
                let mut submission = submission.clone();
                submission.stage = MaxwellSubmissionOrderingStage::Cancelled;
                cancelled.push(submission);
                false
            } else {
                true
            }
        });
        cancelled
    }

    /// Releases all retained work at frontend-owner teardown.
    pub fn clear(&mut self) -> usize {
        let removed = self.queued.len();
        self.queued.clear();
        self.frontend_ids.clear();
        removed
    }

    fn validate_enqueue(
        &self,
        channel: &crate::MaxwellChannelConfiguration,
        submission: &MaxwellValidatedGpfifoSubmission,
        dependency: Option<GuestTimelinePoint>,
        completion: Option<&ReservedTimelinePoint>,
    ) -> Result<(), MaxwellScheduleError> {
        self.validate_shape(channel, submission, dependency, completion.is_some())?;
        if let Some(completion) = completion
            && channel.syncpoint() != Some(completion.point().syncpoint())
        {
            return Err(MaxwellScheduleError::WrongFenceSyncpoint);
        }
        Ok(())
    }

    fn validate_shape(
        &self,
        channel: &crate::MaxwellChannelConfiguration,
        submission: &MaxwellValidatedGpfifoSubmission,
        dependency: Option<GuestTimelinePoint>,
        has_completion: bool,
    ) -> Result<(), MaxwellScheduleError> {
        if channel.scheduling_policy() != MaxwellChannelSchedulingPolicy::DeterministicSingleQueue {
            return Err(MaxwellScheduleError::UnsupportedPolicy);
        }
        if channel.id() != submission.channel() {
            return Err(MaxwellScheduleError::ChannelMismatch {
                channel: channel.id(),
                submission: submission.channel(),
            });
        }
        if self.frontend_ids.contains(&submission.frontend()) {
            return Err(MaxwellScheduleError::DuplicateFrontendSubmission(
                submission.frontend(),
            ));
        }
        let mode = submission.decoded().mode();
        match (mode.fence_wait(), dependency) {
            (true, None) => return Err(MaxwellScheduleError::MissingWaitDependency),
            (false, Some(_)) => return Err(MaxwellScheduleError::UnexpectedWaitDependency),
            _ => {}
        }
        let produces_completion = mode.fence_get() || mode.fence_increment_value();
        match (produces_completion, has_completion) {
            (true, false) => return Err(MaxwellScheduleError::MissingFenceReservation),
            (false, true) => return Err(MaxwellScheduleError::UnexpectedFenceReservation),
            _ => {}
        }
        if has_completion && channel.syncpoint().is_none() {
            return Err(MaxwellScheduleError::WrongFenceSyncpoint);
        }
        Ok(())
    }
}
