//! Transactional lowering boundary for one completely decoded submission.

use std::fmt::{Display, Formatter};

use nixe_gpu::{FrontendSubmissionId, GuestTimelinePoint, ReservedTimelinePoint};
use nixe_memory::MemoryPermissions;

use crate::{
    MaxwellComputeInlineToMemoryUpload, MaxwellComputeSynchronizationError,
    MaxwellComputeSynchronizationPlan, MaxwellEngineOperation, MaxwellEnginePacketDispatch,
    MaxwellGpuAccessError, MaxwellGpuAddressSpace, MaxwellMethodSource, MaxwellResolvedRange,
    MaxwellThreeDInlineConstantBufferUpload, MaxwellThreeDLoweredWork, MaxwellThreeDLoweringCache,
    MaxwellThreeDLoweringError, MaxwellThreeDResourceError, MaxwellThreeDSynchronizationError,
    MaxwellThreeDSynchronizationPlan, lower_maxwell_compute_synchronization,
    lower_maxwell_three_d_synchronization, preflight_maxwell_three_d_operation_unnegotiated,
    resolve_maxwell_three_d_resources,
};

/// One ordered operation whose inputs have been resolved without side effects.
pub enum MaxwellSubmissionExecutionStep {
    ComputeInlineToMemory {
        upload: MaxwellComputeInlineToMemoryUpload,
        target: MaxwellResolvedRange,
    },
    ComputeSynchronization(MaxwellComputeSynchronizationPlan),
    ThreeDInlineConstantBuffer {
        upload: MaxwellThreeDInlineConstantBufferUpload,
        target: MaxwellResolvedRange,
    },
    ThreeD(MaxwellThreeDLoweredWork),
    ThreeDSynchronization(MaxwellThreeDSynchronizationPlan),
}

/// Complete neutral plan awaiting backend negotiation, execution, and completion.
///
/// Only the guest-visible completion point is copied. The unforgeable
/// reservation remains owned by the scheduled dispatch until backend work and
/// memory visibility have completed.
pub struct MaxwellSubmissionExecutionPlan {
    steps: Box<[MaxwellSubmissionExecutionStep]>,
    staged_cache: MaxwellThreeDLoweringCache,
    completion: Option<GuestTimelinePoint>,
}

impl MaxwellSubmissionExecutionPlan {
    #[must_use]
    pub fn steps(&self) -> &[MaxwellSubmissionExecutionStep] {
        &self.steps
    }

    #[must_use]
    pub const fn staged_cache(&self) -> &MaxwellThreeDLoweringCache {
        &self.staged_cache
    }

    #[must_use]
    pub const fn completion(&self) -> Option<GuestTimelinePoint> {
        self.completion
    }
}

/// Failure before any guest write, backend submission, or fence publication.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MaxwellSubmissionExecutionError {
    InlineAddress {
        source: MaxwellMethodSource,
        error: MaxwellGpuAccessError,
    },
    ComputeSynchronization(MaxwellComputeSynchronizationError),
    ThreeDResource(MaxwellThreeDResourceError),
    ThreeDLowering(MaxwellThreeDLoweringError),
    ThreeDSynchronization(MaxwellThreeDSynchronizationError),
    MissingCompletionSignal {
        reserved: GuestTimelinePoint,
    },
    DuplicateCompletionSignal {
        reserved: GuestTimelinePoint,
    },
}

impl Display for MaxwellSubmissionExecutionError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InlineAddress { source, error } => {
                write!(
                    formatter,
                    "inline upload target is invalid: {source}: {error}"
                )
            }
            Self::ComputeSynchronization(error) => Display::fmt(error, formatter),
            Self::ThreeDResource(error) => Display::fmt(error, formatter),
            Self::ThreeDLowering(error) => Display::fmt(error, formatter),
            Self::ThreeDSynchronization(error) => Display::fmt(error, formatter),
            Self::MissingCompletionSignal { reserved } => write!(
                formatter,
                "submission has reserved completion {reserved} but emitted no matching syncpoint increment"
            ),
            Self::DuplicateCompletionSignal { reserved } => write!(
                formatter,
                "submission emitted more than one syncpoint increment for reserved completion {reserved}"
            ),
        }
    }
}

impl std::error::Error for MaxwellSubmissionExecutionError {}

/// Preflights a decoded submission in original packet and method-effect order.
///
/// This function clones the frontend cache and returns the candidate. It never
/// writes an inline payload, invokes a backend, changes scheduler state, or
/// consumes/publishes `completion`.
pub fn preflight_maxwell_submission_execution(
    packets: &[MaxwellEnginePacketDispatch],
    address_space: &MaxwellGpuAddressSpace,
    frontend: FrontendSubmissionId,
    predecessors: Vec<FrontendSubmissionId>,
    completion: Option<&ReservedTimelinePoint>,
    cache: &MaxwellThreeDLoweringCache,
) -> Result<MaxwellSubmissionExecutionPlan, MaxwellSubmissionExecutionError> {
    let mut staged_cache = cache.clone();
    let mut steps = Vec::new();
    let mut prior_work_pending = false;
    let mut completion_signal_count = 0_u8;

    for operation in packets
        .iter()
        .flat_map(MaxwellEnginePacketDispatch::ordered_operations)
    {
        match operation {
            MaxwellEngineOperation::ComputeInlineToMemory(upload) => {
                let target = resolve_inline_target(
                    address_space,
                    upload.address().get(),
                    upload.offset(),
                    upload.source(),
                )?;
                steps.push(MaxwellSubmissionExecutionStep::ComputeInlineToMemory {
                    upload: *upload,
                    target,
                });
                prior_work_pending = true;
            }
            MaxwellEngineOperation::ComputeSynchronization(operation) => {
                let plan = lower_maxwell_compute_synchronization(operation, prior_work_pending)
                    .map_err(MaxwellSubmissionExecutionError::ComputeSynchronization)?;
                if matches!(plan, MaxwellComputeSynchronizationPlan::Neutral) {
                    prior_work_pending = false;
                }
                steps.push(MaxwellSubmissionExecutionStep::ComputeSynchronization(plan));
            }
            MaxwellEngineOperation::ThreeDInlineConstantBuffer(upload) => {
                let target = resolve_inline_target(
                    address_space,
                    upload.address().get(),
                    upload.offset(),
                    upload.source(),
                )?;
                steps.push(MaxwellSubmissionExecutionStep::ThreeDInlineConstantBuffer {
                    upload: *upload,
                    target,
                });
                prior_work_pending = true;
            }
            MaxwellEngineOperation::ThreeD(operation) => {
                let resources = resolve_maxwell_three_d_resources(operation.state(), address_space)
                    .map_err(MaxwellSubmissionExecutionError::ThreeDResource)?;
                let plan = preflight_maxwell_three_d_operation_unnegotiated(
                    operation.state(),
                    &resources,
                    operation.trigger(),
                    None,
                    frontend,
                    predecessors.clone(),
                    &staged_cache,
                )
                .map_err(MaxwellSubmissionExecutionError::ThreeDLowering)?;
                let work = plan
                    .stage_cache(&mut staged_cache)
                    .map_err(MaxwellSubmissionExecutionError::ThreeDLowering)?;
                steps.push(MaxwellSubmissionExecutionStep::ThreeD(work));
                prior_work_pending = true;
            }
            MaxwellEngineOperation::ThreeDSynchronization(operation) => {
                let plan = lower_maxwell_three_d_synchronization(operation, completion)
                    .map_err(MaxwellSubmissionExecutionError::ThreeDSynchronization)?;
                if let MaxwellThreeDSynchronizationPlan::IncrementSyncpoint {
                    completion: reserved,
                    ..
                } = plan
                {
                    completion_signal_count = completion_signal_count.saturating_add(1);
                    if completion_signal_count > 1 {
                        return Err(MaxwellSubmissionExecutionError::DuplicateCompletionSignal {
                            reserved,
                        });
                    }
                }
                steps.push(MaxwellSubmissionExecutionStep::ThreeDSynchronization(plan));
                prior_work_pending = false;
            }
        }
    }

    if completion_signal_count == 0
        && let Some(completion) = completion
    {
        return Err(MaxwellSubmissionExecutionError::MissingCompletionSignal {
            reserved: completion.point(),
        });
    }

    Ok(MaxwellSubmissionExecutionPlan {
        steps: steps.into_boxed_slice(),
        staged_cache,
        completion: completion.map(ReservedTimelinePoint::point),
    })
}

fn resolve_inline_target(
    address_space: &MaxwellGpuAddressSpace,
    base: u64,
    offset: u32,
    source: MaxwellMethodSource,
) -> Result<MaxwellResolvedRange, MaxwellSubmissionExecutionError> {
    let base = address_space
        .address(base)
        .map_err(MaxwellGpuAccessError::Address)
        .map_err(|error| MaxwellSubmissionExecutionError::InlineAddress { source, error })?;
    let target = address_space
        .checked_add(base, u64::from(offset))
        .map_err(MaxwellGpuAccessError::Address)
        .map_err(|error| MaxwellSubmissionExecutionError::InlineAddress { source, error })?;
    address_space
        .resolve_range(target, size_of::<u32>() as u64, MemoryPermissions::WRITE)
        .map_err(|error| MaxwellSubmissionExecutionError::InlineAddress { source, error })
}

#[cfg(test)]
mod tests {
    use nixe_gpu::{
        FrontendSubmissionId, GpuVirtualAddress, GuestSyncpointId, GuestSyncpointValue,
        GuestTimeline, MappingGeneration, TimelineInstanceId, TimelineOwnerId,
    };

    use super::*;
    use crate::{
        MaxwellAddressSpaceId, MaxwellAddressSpaceInitialization, MaxwellChannelId,
        MaxwellChannelOwner, MaxwellGpfifoSourceLocation, MaxwellGpuChannel, MaxwellMappingId,
        MaxwellPushbufferWord, SWITCH_1_GM20B_PROFILE, decode_maxwell_pushbuffer,
        dispatch_maxwell_engine_packet,
    };

    fn packet(
        subchannel: u32,
        method_dword: u32,
        arguments: &[u32],
    ) -> crate::MaxwellDecodedPushbuffer {
        let mut words = Vec::with_capacity(arguments.len() + 1);
        words.push(Ok(MaxwellPushbufferWord::new(
            (1 << 29) | ((arguments.len() as u32) << 16) | (subchannel << 13) | method_dword,
            location(0),
        )));
        words.extend(arguments.iter().enumerate().map(|(index, argument)| {
            Ok(MaxwellPushbufferWord::new(
                *argument,
                location(index as u32 + 1),
            ))
        }));
        decode_maxwell_pushbuffer(words).unwrap()
    }

    fn location(word_offset: u32) -> MaxwellGpfifoSourceLocation {
        MaxwellGpfifoSourceLocation {
            channel: MaxwellChannelId::new(1),
            frontend: FrontendSubmissionId::new(2),
            entry_index: 0,
            pushbuffer: GpuVirtualAddress::try_new(0x8000, 40).unwrap(),
            word_offset: u64::from(word_offset),
            mapping: MaxwellMappingId::new(1),
            generation: MappingGeneration::new(1),
        }
    }

    fn address_space() -> MaxwellGpuAddressSpace {
        let mut address_space =
            MaxwellGpuAddressSpace::new(MaxwellAddressSpaceId::new(1), SWITCH_1_GM20B_PROFILE);
        address_space
            .initialize(MaxwellAddressSpaceInitialization::default())
            .unwrap();
        address_space
    }

    fn reservation() -> (GuestTimeline, ReservedTimelinePoint) {
        let owner = TimelineOwnerId::new(7);
        let mut timeline = GuestTimeline::new(
            GuestSyncpointId::new(1),
            TimelineInstanceId::new(1),
            owner,
            GuestSyncpointValue::new(0),
        );
        let reservation = timeline.reserve(owner, 1).unwrap();
        (timeline, reservation)
    }

    #[test]
    fn empty_preflight_is_neutral_without_a_completion() {
        let plan = preflight_maxwell_submission_execution(
            &[],
            &address_space(),
            FrontendSubmissionId::new(2),
            Vec::new(),
            None,
            &MaxwellThreeDLoweringCache::default(),
        )
        .unwrap();
        assert!(plan.steps().is_empty());
        assert_eq!(plan.completion(), None);
        assert_eq!(plan.staged_cache().revision(), 0);
    }

    #[test]
    fn reserved_completion_requires_an_exact_signal_without_publication() {
        let (timeline, reservation) = reservation();
        let before = timeline.current_point();
        assert!(matches!(
            preflight_maxwell_submission_execution(
                &[],
                &address_space(),
                FrontendSubmissionId::new(2),
                Vec::new(),
                Some(&reservation),
                &MaxwellThreeDLoweringCache::default(),
            ),
            Err(MaxwellSubmissionExecutionError::MissingCompletionSignal { reserved })
                if reserved == reservation.point()
        ));
        assert_eq!(timeline.current_point(), before);
    }

    #[test]
    fn matching_syncpoint_is_retained_once_and_never_published() {
        let mut channel = MaxwellGpuChannel::new(
            MaxwellChannelId::new(1),
            MaxwellChannelOwner::new(1),
            SWITCH_1_GM20B_PROFILE,
        );
        let bind = packet(0, 0, &[SWITCH_1_GM20B_PROFILE.classes().three_d().0]);
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(2),
            &bind.packets()[0],
        )
        .unwrap();
        let increment = packet(0, 0x02c8 / 4, &[1]);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(2),
            &increment.packets()[0],
        )
        .unwrap();
        let (timeline, reservation) = reservation();
        let before = timeline.current_point();
        let plan = preflight_maxwell_submission_execution(
            std::slice::from_ref(&dispatch),
            &address_space(),
            FrontendSubmissionId::new(2),
            Vec::new(),
            Some(&reservation),
            &MaxwellThreeDLoweringCache::default(),
        )
        .unwrap();
        assert_eq!(plan.completion(), Some(reservation.point()));
        assert!(matches!(
            plan.steps(),
            [MaxwellSubmissionExecutionStep::ThreeDSynchronization(
                MaxwellThreeDSynchronizationPlan::IncrementSyncpoint { completion, .. }
            )] if *completion == reservation.point()
        ));
        assert_eq!(timeline.current_point(), before);

        assert!(matches!(
            preflight_maxwell_submission_execution(
                &[dispatch.clone(), dispatch],
                &address_space(),
                FrontendSubmissionId::new(2),
                Vec::new(),
                Some(&reservation),
                &MaxwellThreeDLoweringCache::default(),
            ),
            Err(MaxwellSubmissionExecutionError::DuplicateCompletionSignal { reserved })
                if reserved == reservation.point()
        ));
    }
}
