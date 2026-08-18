//! Bounded, pointer-free Maxwell frontend capture and deterministic replay.
//!
//! Captures begin after T6 has validated and retained every GPFIFO source. They
//! contain only command words and stable guest/frontend identities; mappings,
//! canonical allocations, scheduler reservations, and host pointers are never
//! required by replay.

use std::fmt::{Display, Formatter};

use crate::pushbuffer::packet::SubmissionWords;
use crate::{
    GpuProfileId, MaxwellChannelFrontendState, MaxwellChannelId, MaxwellComputeState,
    MaxwellDecodedPushbuffer, MaxwellEngineDispatchError, MaxwellEnginePacketDispatch,
    MaxwellFrontendDispatch, MaxwellGpfifoCapture, MaxwellGpuAddressSpace, MaxwellGpuChannel,
    MaxwellPushbufferDecodeError, MaxwellPushbufferWord, MaxwellThreeDState,
    decode_maxwell_pushbuffer, dispatch_maxwell_engine_packet,
};

/// Maximum command words retained in an in-memory failure capture.
pub const MAXWELL_FRONTEND_CAPTURE_WORDS: usize = 4_096;

/// Redistributable, pointer-free command data at the validated frontend boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaxwellFrontendCapture {
    profile: GpuProfileId,
    channel: MaxwellChannelId,
    initial_frontend: MaxwellChannelFrontendState,
    initial_compute: MaxwellComputeState,
    initial_three_d: MaxwellThreeDState,
    submission: MaxwellGpfifoCapture,
    total_words: usize,
    words: Box<[MaxwellPushbufferWord]>,
    source_complete: bool,
}

impl MaxwellFrontendCapture {
    #[must_use]
    pub const fn profile(&self) -> GpuProfileId {
        self.profile
    }

    #[must_use]
    pub const fn channel(&self) -> MaxwellChannelId {
        self.channel
    }

    #[must_use]
    pub const fn initial_frontend(&self) -> MaxwellChannelFrontendState {
        self.initial_frontend
    }

    #[must_use]
    pub const fn initial_compute(&self) -> &MaxwellComputeState {
        &self.initial_compute
    }

    #[must_use]
    pub const fn initial_three_d(&self) -> &MaxwellThreeDState {
        &self.initial_three_d
    }

    #[must_use]
    pub const fn submission(&self) -> &MaxwellGpfifoCapture {
        &self.submission
    }

    #[must_use]
    pub const fn total_words(&self) -> usize {
        self.total_words
    }

    #[must_use]
    pub fn words(&self) -> &[MaxwellPushbufferWord] {
        &self.words
    }

    #[must_use]
    pub const fn omitted_words(&self) -> usize {
        self.total_words.saturating_sub(self.words.len())
    }

    #[must_use]
    pub const fn is_complete(&self) -> bool {
        self.source_complete && self.omitted_words() == 0
    }
}

impl Display for MaxwellFrontendCapture {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "profile={} {} words={} shown={} submission=[{}]",
            self.profile,
            self.channel,
            self.total_words,
            self.words.len(),
            self.submission
        )?;
        if let Some(first) = self.words.first() {
            write!(
                formatter,
                " first-word=[{} value={:#010x}]",
                first.location(),
                first.value()
            )?;
        }
        if self.omitted_words() != 0 {
            write!(formatter, " words-truncated={}", self.omitted_words())?;
        }
        if !self.source_complete {
            formatter.write_str(" source-incomplete=true")?;
        }
        Ok(())
    }
}

/// First semantic boundary reached while consuming or replaying a capture.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MaxwellFrontendFailure {
    EmptySubmission,
    PacketDecode(MaxwellPushbufferDecodeError),
    EngineDispatch(MaxwellEngineDispatchError),
    ExecutionUnavailable,
}

impl Display for MaxwellFrontendFailure {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptySubmission => {
                formatter.write_str("empty submission completion semantics are unavailable")
            }
            Self::PacketDecode(error) => write!(formatter, "packet decode failed: {error}"),
            Self::EngineDispatch(error) => {
                write!(formatter, "class method dispatch failed: {error}")
            }
            Self::ExecutionUnavailable => formatter.write_str(
                "decoded frontend work requires a later neutral execution and completion layer",
            ),
        }
    }
}

/// Deterministic packet prefix committed before the first fatal boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaxwellFrontendReplay {
    packets: Box<[MaxwellEnginePacketDispatch]>,
    failure: MaxwellFrontendFailure,
}

impl MaxwellFrontendReplay {
    #[must_use]
    pub fn packets(&self) -> &[MaxwellEnginePacketDispatch] {
        &self.packets
    }

    #[must_use]
    pub const fn failure(&self) -> &MaxwellFrontendFailure {
        &self.failure
    }
}

/// Capture construction or replay precondition failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MaxwellFrontendCaptureError {
    ProfileMismatch {
        expected: GpuProfileId,
        actual: GpuProfileId,
    },
    ChannelMismatch {
        expected: MaxwellChannelId,
        actual: MaxwellChannelId,
    },
    FrontendStateMismatch {
        channel: MaxwellChannelId,
    },
    EngineStateMismatch {
        channel: MaxwellChannelId,
    },
    TruncatedCapture {
        total_words: usize,
        retained_words: usize,
    },
    IncompleteSourceCapture,
    ResourceExhausted,
}

impl Display for MaxwellFrontendCaptureError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ProfileMismatch { expected, actual } => write!(
                formatter,
                "capture profile mismatch: expected={expected} actual={actual}"
            ),
            Self::ChannelMismatch { expected, actual } => write!(
                formatter,
                "capture channel mismatch: expected={expected} actual={actual}"
            ),
            Self::FrontendStateMismatch { channel } => {
                write!(
                    formatter,
                    "capture initial frontend state mismatch: {channel}"
                )
            }
            Self::EngineStateMismatch { channel } => {
                write!(
                    formatter,
                    "capture initial Maxwell engine state mismatch: {channel}"
                )
            }
            Self::TruncatedCapture {
                total_words,
                retained_words,
            } => write!(
                formatter,
                "capture is incomplete and cannot be replayed: words={total_words} retained={retained_words}"
            ),
            Self::IncompleteSourceCapture => formatter.write_str(
                "capture stopped on a retained-source failure and cannot reproduce it without mappings",
            ),
            Self::ResourceExhausted => {
                formatter.write_str("frontend capture exhausted host resources")
            }
        }
    }
}

impl std::error::Error for MaxwellFrontendCaptureError {}

/// Captures and decodes one retained dispatch, then stops at its first T7 boundary.
pub fn capture_maxwell_frontend_dispatch(
    dispatch: &MaxwellFrontendDispatch,
    channel: &mut MaxwellGpuChannel,
    address_space: &MaxwellGpuAddressSpace,
) -> (MaxwellFrontendCapture, MaxwellFrontendReplay) {
    let initial = CapturedInitialState {
        frontend: channel.frontend(),
        compute: channel.compute().clone(),
        three_d: channel.three_d().clone(),
    };
    let submission = dispatch.scheduled().submission();
    let mut recording = RecordingWords::new(SubmissionWords::new(submission, address_space));
    let decoded = decode_maxwell_pushbuffer(&mut recording);
    let source_complete = !matches!(&decoded, Err(MaxwellPushbufferDecodeError::Source(_)));
    let capture = recording.finish(
        channel.profile_id(),
        channel.id(),
        initial,
        dispatch.capture(),
        source_complete,
    );
    let replay = match decoded {
        Ok(decoded) => dispatch_decoded(channel, submission.frontend(), &decoded),
        Err(error) => MaxwellFrontendReplay {
            packets: Box::new([]),
            failure: MaxwellFrontendFailure::PacketDecode(error),
        },
    };
    (capture, replay)
}

/// Replays a complete capture without mappings, scheduler state, or backing storage.
pub fn replay_maxwell_frontend_capture(
    capture: &MaxwellFrontendCapture,
    channel: &mut MaxwellGpuChannel,
) -> Result<MaxwellFrontendReplay, MaxwellFrontendCaptureError> {
    if channel.profile_id() != capture.profile {
        return Err(MaxwellFrontendCaptureError::ProfileMismatch {
            expected: capture.profile,
            actual: channel.profile_id(),
        });
    }
    if channel.id() != capture.channel {
        return Err(MaxwellFrontendCaptureError::ChannelMismatch {
            expected: capture.channel,
            actual: channel.id(),
        });
    }
    if channel.frontend() != capture.initial_frontend {
        return Err(MaxwellFrontendCaptureError::FrontendStateMismatch {
            channel: channel.id(),
        });
    }
    if channel.compute() != &capture.initial_compute
        || channel.three_d() != &capture.initial_three_d
    {
        return Err(MaxwellFrontendCaptureError::EngineStateMismatch {
            channel: channel.id(),
        });
    }
    if capture.omitted_words() != 0 {
        return Err(MaxwellFrontendCaptureError::TruncatedCapture {
            total_words: capture.total_words,
            retained_words: capture.words.len(),
        });
    }
    if !capture.source_complete {
        return Err(MaxwellFrontendCaptureError::IncompleteSourceCapture);
    }
    let decoded = decode_maxwell_pushbuffer(capture.words.iter().copied().map(Ok));
    Ok(match decoded {
        Ok(decoded) => dispatch_decoded(channel, capture.submission.frontend(), &decoded),
        Err(error) => MaxwellFrontendReplay {
            packets: Box::new([]),
            failure: MaxwellFrontendFailure::PacketDecode(error),
        },
    })
}

fn dispatch_decoded(
    channel: &mut MaxwellGpuChannel,
    submission: nixe_gpu::FrontendSubmissionId,
    decoded: &MaxwellDecodedPushbuffer,
) -> MaxwellFrontendReplay {
    if decoded.packets().is_empty() {
        return MaxwellFrontendReplay {
            packets: Box::new([]),
            failure: MaxwellFrontendFailure::EmptySubmission,
        };
    }
    let mut packets = Vec::new();
    if packets.try_reserve_exact(decoded.packets().len()).is_err() {
        return MaxwellFrontendReplay {
            packets: Box::new([]),
            failure: MaxwellFrontendFailure::EngineDispatch(
                MaxwellEngineDispatchError::ResourceExhausted,
            ),
        };
    }
    for packet in decoded.packets() {
        match dispatch_maxwell_engine_packet(channel, submission, packet) {
            Ok(packet) => packets.push(packet),
            Err(error) => {
                return MaxwellFrontendReplay {
                    packets: packets.into_boxed_slice(),
                    failure: MaxwellFrontendFailure::EngineDispatch(error),
                };
            }
        }
    }
    MaxwellFrontendReplay {
        packets: packets.into_boxed_slice(),
        failure: MaxwellFrontendFailure::ExecutionUnavailable,
    }
}

struct RecordingWords<I> {
    inner: I,
    total_words: usize,
    words: Vec<MaxwellPushbufferWord>,
}

struct CapturedInitialState {
    frontend: MaxwellChannelFrontendState,
    compute: MaxwellComputeState,
    three_d: MaxwellThreeDState,
}

impl<I> RecordingWords<I> {
    fn new(inner: I) -> Self {
        Self {
            inner,
            total_words: 0,
            words: Vec::new(),
        }
    }

    fn finish(
        self,
        profile: GpuProfileId,
        channel: MaxwellChannelId,
        initial: CapturedInitialState,
        submission: MaxwellGpfifoCapture,
        source_complete: bool,
    ) -> MaxwellFrontendCapture {
        MaxwellFrontendCapture {
            profile,
            channel,
            initial_frontend: initial.frontend,
            initial_compute: initial.compute,
            initial_three_d: initial.three_d,
            submission,
            total_words: self.total_words,
            words: self.words.into_boxed_slice(),
            source_complete,
        }
    }
}

impl<I> Iterator for RecordingWords<I>
where
    I: Iterator<Item = Result<MaxwellPushbufferWord, crate::MaxwellGpfifoSourceError>>,
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        match self.inner.next()? {
            Ok(word) => {
                self.total_words = self.total_words.saturating_add(1);
                if self.words.len() < MAXWELL_FRONTEND_CAPTURE_WORDS {
                    if self.words.try_reserve(1).is_err() {
                        return Some(Err(crate::MaxwellGpfifoSourceError::ResourceExhausted));
                    }
                    self.words.push(word);
                }
                Some(Ok(word))
            }
            Err(error) => Some(Err(error)),
        }
    }
}

#[cfg(test)]
mod tests {
    use nixe_gpu::FrontendSubmissionId;
    use nixe_memory::{CanonicalAllocation, MemoryPermissions};

    use super::*;
    use crate::{
        MaxwellAddressSpaceId, MaxwellAddressSpaceInitialization, MaxwellAllocationId,
        MaxwellChannelOwner, MaxwellGpfifoSubmitRequest, MaxwellMapRequest, MaxwellScheduler,
        SWITCH_1_GM20B_PROFILE, decode_gpfifo_submission, resolve_gpfifo_submission,
    };

    const HARDWARE_FORMAT: u32 = 1 << 2;

    fn method_header(opcode: u32, method: u32, subchannel: u32, count: u32) -> u32 {
        opcode << 29 | count << 16 | subchannel << 13 | method
    }

    fn descriptor(address: u64, words: u32) -> [u8; 8] {
        let mut bytes = [0; 8];
        bytes[..4].copy_from_slice(&(address as u32).to_le_bytes());
        bytes[4..].copy_from_slice(&(((address >> 32) as u32) | words << 10).to_le_bytes());
        bytes
    }

    fn channel(address_space: MaxwellAddressSpaceId) -> MaxwellGpuChannel {
        let mut channel = MaxwellGpuChannel::new(
            MaxwellChannelId::new(7),
            MaxwellChannelOwner::new(1),
            SWITCH_1_GM20B_PROFILE,
        );
        channel.bind_address_space(address_space).unwrap();
        channel
    }

    fn scheduled_fixture(
        entries: &[&[u32]],
    ) -> (
        CanonicalAllocation,
        MaxwellGpuAddressSpace,
        MaxwellGpuChannel,
        MaxwellFrontendDispatch,
    ) {
        let byte_count = entries.iter().map(|entry| entry.len() * 4).sum::<usize>();
        let allocation_size = byte_count.next_multiple_of(0x1000).max(0x1000);
        let allocation = CanonicalAllocation::zeroed(allocation_size, 0x1000).unwrap();
        let mut byte_offset = 0;
        for entry in entries {
            let mut bytes = Vec::with_capacity(entry.len() * 4);
            for word in *entry {
                bytes.extend_from_slice(&word.to_le_bytes());
            }
            allocation.write(byte_offset, &bytes).unwrap();
            byte_offset += bytes.len();
        }
        let mut address_space =
            MaxwellGpuAddressSpace::new(MaxwellAddressSpaceId::new(3), SWITCH_1_GM20B_PROFILE);
        address_space
            .initialize(MaxwellAddressSpaceInitialization::default())
            .unwrap();
        let mapping = address_space
            .map(MaxwellMapRequest {
                allocation: MaxwellAllocationId::new(9),
                backing: allocation
                    .backing_range(MemoryPermissions::READ_WRITE)
                    .unwrap(),
                backing_offset: 0,
                size: u64::try_from(allocation_size).unwrap(),
                allocation_alignment: 0x1000,
                page_size: 0x1000,
                kind: 0,
                cacheable: false,
                permissions: MemoryPermissions::READ_WRITE,
                fixed_offset: None,
            })
            .unwrap();
        let channel = channel(address_space.id());
        let mut descriptors = Vec::with_capacity(entries.len() * 8);
        let mut entry_offset = 0_u64;
        for entry in entries {
            descriptors.extend_from_slice(&descriptor(
                mapping.offset().get() + entry_offset,
                u32::try_from(entry.len()).unwrap(),
            ));
            entry_offset += u64::try_from(entry.len() * 4).unwrap();
        }
        let decoded = decode_gpfifo_submission(
            SWITCH_1_GM20B_PROFILE,
            16,
            MaxwellGpfifoSubmitRequest {
                entry_count: u32::try_from(entries.len()).unwrap(),
                flags: HARDWARE_FORMAT,
                fence_id: 0,
                fence_value: 0,
            },
            &descriptors,
        )
        .unwrap();
        let submission = resolve_gpfifo_submission(
            &channel,
            FrontendSubmissionId::new(5),
            decoded,
            &address_space,
        )
        .unwrap();
        let mut scheduler = MaxwellScheduler::default();
        scheduler.enqueue(&channel, submission, None, None).unwrap();
        let dispatch = scheduler
            .dispatch_next(true, &address_space)
            .unwrap()
            .unwrap();
        (allocation, address_space, channel, dispatch)
    }

    #[test]
    fn capture_replay_preserves_packets_names_arguments_sources_and_failure() {
        let first_entry = [method_header(1, 0, 0, 1)];
        let second_entry = [
            SWITCH_1_GM20B_PROFILE.classes().three_d().0,
            method_header(1, 0x100 / 4, 0, 1),
            0xfeed_beef,
            method_header(4, 0x110 / 4, 0, 0),
        ];
        let (_allocation, address_space, mut channel, dispatch) =
            scheduled_fixture(&[&first_entry, &second_entry]);
        let initial = channel.clone();
        let (capture, first) =
            capture_maxwell_frontend_dispatch(&dispatch, &mut channel, &address_space);

        assert!(capture.is_complete());
        assert_eq!(capture.total_words(), 5);
        assert_eq!(capture.words()[0].location().entry_index, 0);
        assert_eq!(capture.words()[1].location().entry_index, 1);
        assert_eq!(first.packets().len(), 3);
        assert_eq!(
            first.packets()[1].methods()[0].metadata().method_name(),
            "NO_OPERATION"
        );
        assert_eq!(
            first.packets()[1].methods()[0].method().source().argument(),
            0xfeed_beef
        );
        assert_eq!(
            first.packets()[2].methods()[0].metadata().method_name(),
            "WAIT_FOR_IDLE"
        );
        assert!(matches!(
            first.packets()[2].methods()[0].effect(),
            crate::MaxwellEngineMethodEffect::ThreeDSynchronizationTrigger(
                crate::MaxwellThreeDSynchronizationTrigger::WaitForIdle { value: 0, .. }
            )
        ));
        assert!(matches!(
            first.failure(),
            MaxwellFrontendFailure::ExecutionUnavailable
        ));

        let mut replay_channel = initial;
        let replay = replay_maxwell_frontend_capture(&capture, &mut replay_channel).unwrap();
        assert_eq!(replay, first);
        assert_eq!(replay_channel.frontend(), channel.frontend());
    }

    #[test]
    fn first_method_rejection_preserves_initial_state_and_replay_requires_it() {
        let words = [
            method_header(1, 0, 0, 1),
            SWITCH_1_GM20B_PROFILE.classes().three_d().0 | (1 << 31),
        ];
        let (_allocation, address_space, mut channel, dispatch) = scheduled_fixture(&[&words]);
        let initial = channel.clone();
        let (capture, first) =
            capture_maxwell_frontend_dispatch(&dispatch, &mut channel, &address_space);
        assert!(matches!(
            first.failure(),
            MaxwellFrontendFailure::EngineDispatch(MaxwellEngineDispatchError::Binding(
                crate::MaxwellMethodDispatchError::InvalidSetObjectValue { .. }
            ))
        ));
        assert_eq!(channel.frontend(), initial.frontend());

        channel.reset_subchannel_bindings();
        let mut mismatched = initial.clone();
        let bind = crate::decode_maxwell_pushbuffer([
            Ok(MaxwellPushbufferWord::new(
                method_header(1, 0, 0, 1),
                capture.words()[0].location(),
            )),
            Ok(MaxwellPushbufferWord::new(
                SWITCH_1_GM20B_PROFILE.classes().three_d().0,
                capture.words()[1].location(),
            )),
        ])
        .unwrap();
        crate::dispatch_maxwell_engine_packet(
            &mut mismatched,
            capture.submission().frontend(),
            &bind.packets()[0],
        )
        .unwrap();
        assert!(matches!(
            replay_maxwell_frontend_capture(&capture, &mut mismatched),
            Err(MaxwellFrontendCaptureError::FrontendStateMismatch { .. })
        ));

        let mut engine_mismatched = initial.clone();
        crate::dispatch_maxwell_engine_packet(
            &mut engine_mismatched,
            capture.submission().frontend(),
            &bind.packets()[0],
        )
        .unwrap();
        let point_size = crate::decode_maxwell_pushbuffer([
            Ok(MaxwellPushbufferWord::new(
                method_header(1, 0x1518 / 4, 0, 1),
                capture.words()[0].location(),
            )),
            Ok(MaxwellPushbufferWord::new(
                0x3f80_0000,
                capture.words()[1].location(),
            )),
        ])
        .unwrap();
        crate::dispatch_maxwell_engine_packet(
            &mut engine_mismatched,
            capture.submission().frontend(),
            &point_size.packets()[0],
        )
        .unwrap();
        engine_mismatched.reset_subchannel_bindings();
        assert_eq!(engine_mismatched.frontend(), capture.initial_frontend());
        assert!(matches!(
            replay_maxwell_frontend_capture(&capture, &mut engine_mismatched),
            Err(MaxwellFrontendCaptureError::EngineStateMismatch { .. })
        ));

        let mut compute_mismatched = initial;
        let compute_bind = crate::decode_maxwell_pushbuffer([
            Ok(MaxwellPushbufferWord::new(
                method_header(1, 0, 1, 1),
                capture.words()[0].location(),
            )),
            Ok(MaxwellPushbufferWord::new(
                SWITCH_1_GM20B_PROFILE.classes().compute().0,
                capture.words()[1].location(),
            )),
        ])
        .unwrap();
        crate::dispatch_maxwell_engine_packet(
            &mut compute_mismatched,
            capture.submission().frontend(),
            &compute_bind.packets()[0],
        )
        .unwrap();
        let local_memory_upper = crate::decode_maxwell_pushbuffer([
            Ok(MaxwellPushbufferWord::new(
                method_header(1, 0x0790 / 4, 1, 1),
                capture.words()[0].location(),
            )),
            Ok(MaxwellPushbufferWord::new(4, capture.words()[1].location())),
        ])
        .unwrap();
        crate::dispatch_maxwell_engine_packet(
            &mut compute_mismatched,
            capture.submission().frontend(),
            &local_memory_upper.packets()[0],
        )
        .unwrap();
        compute_mismatched.reset_subchannel_bindings();
        assert_eq!(compute_mismatched.frontend(), capture.initial_frontend());
        assert!(matches!(
            replay_maxwell_frontend_capture(&capture, &mut compute_mismatched),
            Err(MaxwellFrontendCaptureError::EngineStateMismatch { .. })
        ));
    }

    #[test]
    fn bounded_capture_never_replays_an_omitted_suffix() {
        let words = vec![0; MAXWELL_FRONTEND_CAPTURE_WORDS + 1];
        let (_allocation, address_space, mut channel, dispatch) = scheduled_fixture(&[&words]);
        let initial = channel.clone();
        let (capture, first) =
            capture_maxwell_frontend_dispatch(&dispatch, &mut channel, &address_space);
        assert!(matches!(
            first.failure(),
            MaxwellFrontendFailure::ExecutionUnavailable
        ));
        assert_eq!(capture.omitted_words(), 1);
        let mut replay_channel = initial;
        assert!(matches!(
            replay_maxwell_frontend_capture(&capture, &mut replay_channel),
            Err(MaxwellFrontendCaptureError::TruncatedCapture {
                total_words: 4097,
                retained_words: 4096
            })
        ));
        assert_eq!(replay_channel.frontend(), capture.initial_frontend());
    }
}
