//! Direct retained-source consumption for one Maxwell frontend submission.

use std::fmt::{Display, Formatter};

use crate::pushbuffer::packet::SubmissionWords;
use crate::{
    MaxwellEngineDispatchError, MaxwellEnginePacketDispatch, MaxwellFrontendDispatch,
    MaxwellGpuAddressSpace, MaxwellGpuChannel, MaxwellPushbufferDecodeError,
    decode_maxwell_pushbuffer, dispatch_maxwell_engine_packet,
};

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
    ResourceExhausted,
}

impl Display for MaxwellFrontendDispatchError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptySubmission => formatter.write_str("empty Maxwell submission"),
            Self::PacketDecode(error) => write!(formatter, "packet decode failed: {error}"),
            Self::EngineDispatch(error) => write!(formatter, "engine dispatch failed: {error}"),
            Self::ResourceExhausted => {
                formatter.write_str("Maxwell frontend dispatch exhausted host resources")
            }
        }
    }
}

impl std::error::Error for MaxwellFrontendDispatchError {}

/// Decodes and dispatches a retained submission exactly once.
///
/// Successful work retains only execution-relevant packets. Source mappings
/// remain owned by `dispatch`, and typed failures already carry their precise
/// command location, so normal execution does not clone engine state or record
/// a parallel replay stream.
pub fn dispatch_maxwell_frontend(
    dispatch: &MaxwellFrontendDispatch,
    channel: &mut MaxwellGpuChannel,
    address_space: &MaxwellGpuAddressSpace,
) -> Result<Box<[MaxwellEnginePacketDispatch]>, MaxwellFrontendDispatchError> {
    let submission = dispatch.scheduled().submission();
    let decoded = decode_maxwell_pushbuffer(SubmissionWords::new(submission, address_space))
        .map_err(MaxwellFrontendDispatchError::PacketDecode)?;
    if decoded.packets().is_empty() {
        return Err(MaxwellFrontendDispatchError::EmptySubmission);
    }
    let mut packets = Vec::new();
    packets
        .try_reserve_exact(decoded.packets().len())
        .map_err(|_| MaxwellFrontendDispatchError::ResourceExhausted)?;
    for packet in decoded.packets() {
        packets.push(
            dispatch_maxwell_engine_packet(channel, submission.frontend(), packet)
                .map_err(|error| MaxwellFrontendDispatchError::EngineDispatch(Box::new(error)))?,
        );
    }
    Ok(packets.into_boxed_slice())
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
