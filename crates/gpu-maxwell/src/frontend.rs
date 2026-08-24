//! Direct retained-source consumption for one Maxwell frontend submission.

use std::fmt::{Display, Formatter};

use crate::engines::{MaxwellEngineStreamError, stream_maxwell_engine_packet};
use crate::execution::MaxwellSubmissionPlanner;
use crate::pushbuffer::packet::SubmissionWords;
use crate::{
    MaxwellDecodedPushbuffer, MaxwellEngineDispatchError, MaxwellFrontendDispatch,
    MaxwellGpuAddressSpace, MaxwellGpuChannel, MaxwellPushbufferDecodeError,
    MaxwellSubmissionExecutionError, MaxwellSubmissionExecutionPlan, MaxwellThreeDLoweringCache,
    decode_maxwell_pushbuffer,
};
use nixe_gpu::{FrontendSubmissionId, ReservedTimelinePoint};

/// Hard bound for a failure-only command dump.
pub const MAXWELL_FRONTEND_DIAGNOSTIC_WORDS: usize = 4_096;

/// Raw command prefix reconstructed from retained sources after a failure.
pub struct MaxwellFrontendDiagnostic {
    total_words: usize,
    words: Box<[u32]>,
}

impl MaxwellFrontendDiagnostic {
    #[must_use]
    pub const fn total_words(&self) -> usize {
        self.total_words
    }

    #[must_use]
    pub const fn words(&self) -> &[u32] {
        &self.words
    }

    #[must_use]
    pub const fn is_complete(&self) -> bool {
        self.total_words == self.words.len()
    }
}

/// First semantic failure while consuming a retained command stream.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MaxwellFrontendDispatchError {
    EmptySubmission,
    PacketDecode(MaxwellPushbufferDecodeError),
    EngineDispatch(Box<MaxwellEngineDispatchError>),
    Execution(Box<MaxwellSubmissionExecutionError>),
}

impl Display for MaxwellFrontendDispatchError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptySubmission => formatter.write_str("empty Maxwell submission"),
            Self::PacketDecode(error) => write!(formatter, "packet decode failed: {error}"),
            Self::EngineDispatch(error) => write!(formatter, "engine dispatch failed: {error}"),
            Self::Execution(error) => write!(formatter, "frontend lowering failed: {error}"),
        }
    }
}

impl std::error::Error for MaxwellFrontendDispatchError {}

/// Decodes, dispatches, and lowers a retained submission exactly once.
///
/// Each packet is lowered before the next packet mutates channel state. Only
/// the completed neutral plan survives this function; packet objects and their
/// temporary trigger snapshots do not cross the frontend boundary.
pub fn lower_maxwell_frontend(
    dispatch: &MaxwellFrontendDispatch,
    channel: &mut MaxwellGpuChannel,
    address_space: &MaxwellGpuAddressSpace,
    frontend: FrontendSubmissionId,
    predecessors: Vec<FrontendSubmissionId>,
    completion: Option<&ReservedTimelinePoint>,
    cache: &mut MaxwellThreeDLoweringCache,
) -> Result<MaxwellSubmissionExecutionPlan, MaxwellFrontendDispatchError> {
    let submission = dispatch.scheduled().submission();
    let decoded = decode_maxwell_pushbuffer(SubmissionWords::new(submission, address_space))
        .map_err(MaxwellFrontendDispatchError::PacketDecode)?;
    if decoded.packets().is_empty() {
        return Err(MaxwellFrontendDispatchError::EmptySubmission);
    }
    lower_maxwell_pushbuffer(
        &decoded,
        channel,
        address_space,
        frontend,
        predecessors,
        completion,
        cache,
    )
}

/// Applies and lowers an already decoded pushbuffer through the current
/// streaming frontend path.
pub fn lower_maxwell_pushbuffer(
    decoded: &MaxwellDecodedPushbuffer,
    channel: &mut MaxwellGpuChannel,
    address_space: &MaxwellGpuAddressSpace,
    frontend: FrontendSubmissionId,
    predecessors: Vec<FrontendSubmissionId>,
    completion: Option<&ReservedTimelinePoint>,
    cache: &mut MaxwellThreeDLoweringCache,
) -> Result<MaxwellSubmissionExecutionPlan, MaxwellFrontendDispatchError> {
    if decoded.packets().is_empty() {
        return Err(MaxwellFrontendDispatchError::EmptySubmission);
    }
    let mut planner =
        MaxwellSubmissionPlanner::new(address_space, frontend, predecessors, completion, cache);
    let (mut mme_methods, mut mme_parameters) = planner.take_mme_scratch();
    for packet in decoded.packets() {
        stream_maxwell_engine_packet(
            channel,
            frontend,
            packet,
            None,
            &mut mme_methods,
            &mut mme_parameters,
            &mut |event| planner.push_event(event),
        )
        .map_err(|error| match error {
            MaxwellEngineStreamError::Dispatch(error) => {
                MaxwellFrontendDispatchError::EngineDispatch(error)
            }
            MaxwellEngineStreamError::Consumer(error) => {
                MaxwellFrontendDispatchError::Execution(Box::new(error))
            }
        })?;
    }
    planner.recycle_mme_scratch(mme_methods, mme_parameters);
    planner
        .finish()
        .map_err(|error| MaxwellFrontendDispatchError::Execution(Box::new(error)))
}

/// Reconstructs a bounded raw command prefix from sources already retained by
/// a failed dispatch. Normal submissions never call this function.
pub fn diagnose_maxwell_frontend(
    dispatch: &MaxwellFrontendDispatch,
) -> Result<MaxwellFrontendDiagnostic, Box<str>> {
    let submission = dispatch.scheduled().submission();
    let total_words = submission
        .pushbuffers()
        .iter()
        .try_fold(0_usize, |total, pushbuffer| {
            usize::try_from(pushbuffer.entry().word_count())
                .ok()
                .and_then(|words| total.checked_add(words))
        })
        .ok_or_else(|| Box::<str>::from("Maxwell diagnostic word count overflows"))?;
    let retained_words = total_words.min(MAXWELL_FRONTEND_DIAGNOSTIC_WORDS);
    let mut words = Vec::new();
    words
        .try_reserve_exact(retained_words)
        .map_err(|_| Box::<str>::from("Maxwell diagnostic allocation failed"))?;
    for pushbuffer in submission.pushbuffers() {
        let remaining_words = retained_words - words.len();
        if remaining_words == 0 {
            break;
        }
        let entry_words = usize::try_from(pushbuffer.entry().word_count())
            .map_err(|_| Box::<str>::from("Maxwell diagnostic entry size overflows"))?;
        let read_words = remaining_words.min(entry_words);
        let read_bytes = read_words
            .checked_mul(4)
            .ok_or_else(|| Box::<str>::from("Maxwell diagnostic byte count overflows"))?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(read_bytes)
            .map_err(|_| Box::<str>::from("Maxwell diagnostic allocation failed"))?;
        bytes.resize(read_bytes, 0);
        let mut copied = 0;
        for segment in pushbuffer.source().segments() {
            let count = usize::try_from(segment.size())
                .unwrap_or(usize::MAX)
                .min(read_bytes - copied);
            segment
                .mapping()
                .backing()
                .read(segment.backing_offset(), &mut bytes[copied..copied + count])
                .map_err(|error| error.to_string().into_boxed_str())?;
            copied += count;
            if copied == read_bytes {
                break;
            }
        }
        if copied != read_bytes {
            return Err("retained Maxwell diagnostic source is incomplete".into());
        }
        words.extend(
            bytes
                .chunks_exact(4)
                .map(|word| u32::from_le_bytes(word.try_into().expect("word chunk is complete"))),
        );
    }
    Ok(MaxwellFrontendDiagnostic {
        total_words,
        words: words.into_boxed_slice(),
    })
}
