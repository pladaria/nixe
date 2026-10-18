//! Direct retained-source consumption for one Maxwell frontend submission.

use std::fmt::{Display, Formatter};

use crate::engines::{MaxwellEngineStreamError, stream_maxwell_engine_packet};
use crate::execution::{MaxwellFrontend, MaxwellSubmissionPlanner};
pub use crate::pushbuffer::packet::MaxwellCommandSourceScratch;
use crate::pushbuffer::packet::{DecodedPackets, SubmissionWords};
use crate::{
    MaxwellEngineDispatchError, MaxwellFrontendDispatch, MaxwellGpuAddressSpace, MaxwellGpuChannel,
    MaxwellPushbufferDecodeError, MaxwellSubmissionExecutionError, MaxwellSubmissionExecutionPlan,
    MaxwellSubmissionExecutionStep,
};

/// Hard bound for a failure-only command dump.
// Retain up to 256 KiB of commands so ordinary multi-entry submissions
// include the failing packet rather than only their initialization prefix.
// This allocation occurs exclusively on the diagnostic failure path.
pub const MAXWELL_FRONTEND_DIAGNOSTIC_WORDS: usize = 65_536;

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
    Delivery(Box<str>),
    PacketDecode(MaxwellPushbufferDecodeError),
    EngineDispatch(Box<MaxwellEngineDispatchError>),
    Execution(Box<MaxwellSubmissionExecutionError>),
}

impl Display for MaxwellFrontendDispatchError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptySubmission => formatter.write_str("empty Maxwell submission"),
            Self::Delivery(error) => write!(formatter, "frontend prefix delivery failed: {error}"),
            Self::PacketDecode(error) => write!(formatter, "packet decode failed: {error}"),
            Self::EngineDispatch(error) => write!(formatter, "engine dispatch failed: {error}"),
            Self::Execution(error) => write!(formatter, "frontend lowering failed: {error}"),
        }
    }
}

impl std::error::Error for MaxwellFrontendDispatchError {}

/// Decodes, dispatches, and lowers a retained submission exactly once.
///
/// Consumes completed effects before decoding subsequent operations.
/// The returned tail has passed completion-signal validation; only then may
/// its backend consumer seal the frontend delivery and publish its fence.
#[allow(
    clippy::too_many_arguments,
    reason = "command state, source capture and native consumption boundaries"
)]
pub fn stream_maxwell_frontend<L: MaxwellFrontend>(
    dispatch: &MaxwellFrontendDispatch,
    channel: &mut MaxwellGpuChannel,
    address_space: &MaxwellGpuAddressSpace,
    cache: &mut L,
    sources: &mut MaxwellCommandSourceScratch,
    synchronize: &mut dyn FnMut() -> Result<(), Box<str>>,
    consume: &mut impl FnMut(
        std::vec::Drain<'_, MaxwellSubmissionExecutionStep<L::Work>>,
    ) -> Result<(), Box<str>>,
) -> Result<MaxwellSubmissionExecutionPlan<L::Work>, MaxwellFrontendDispatchError> {
    let submission = dispatch.scheduled().submission();
    let packet_scratch = sources.take_packet_scratch();
    let mut packets = DecodedPackets::with_scratch(
        SubmissionWords::new(submission, address_space, sources, synchronize),
        packet_scratch,
    );
    let mut planner = MaxwellSubmissionPlanner::new(
        address_space,
        dispatch.scheduled().frontend(),
        dispatch.scheduled().completion(),
        cache,
    );
    if submission.decoded().mode().fence_get() {
        planner.set_driver_completion_increments(2);
    }
    let result = lower_packets_streamed(&mut packets, channel, planner, consume);
    let packet_scratch = packets.into_scratch();
    sources.recycle_packet_scratch(packet_scratch);
    result
}

fn lower_packets_streamed<
    L: MaxwellFrontend,
    I: Iterator<Item = Result<crate::MaxwellPushbufferWord, crate::MaxwellGpfifoSourceError>>,
>(
    packets: &mut DecodedPackets<I>,
    channel: &mut MaxwellGpuChannel,
    mut planner: MaxwellSubmissionPlanner<'_, L>,
    consume: &mut impl FnMut(
        std::vec::Drain<'_, MaxwellSubmissionExecutionStep<L::Work>>,
    ) -> Result<(), Box<str>>,
) -> Result<MaxwellSubmissionExecutionPlan<L::Work>, MaxwellFrontendDispatchError> {
    #[cfg(feature = "performance-counters")]
    let started = std::time::Instant::now();
    #[cfg(feature = "performance-counters")]
    let mut consumed = std::time::Duration::ZERO;
    let frontend = planner.frontend();
    let (mut mme_methods, mut mme_parameters) = planner.take_mme_scratch();
    let mut any_packet = false;
    while let Some(packet) = packets.next() {
        let packet = packet.map_err(MaxwellFrontendDispatchError::PacketDecode)?;
        any_packet = true;
        stream_maxwell_engine_packet(
            channel,
            frontend,
            &packet,
            None,
            &mut mme_methods,
            &mut mme_parameters,
            &mut |event| {
                if let Some(steps) = planner.before_event(&event) {
                    #[cfg(feature = "performance-counters")]
                    let consume_started = std::time::Instant::now();
                    consume(steps).map_err(MaxwellFrontendDispatchError::Delivery)?;
                    #[cfg(feature = "performance-counters")]
                    {
                        consumed += consume_started.elapsed();
                    }
                }
                planner
                    .push_event(event)
                    .map_err(|error| MaxwellFrontendDispatchError::Execution(Box::new(error)))?;
                if let Some(steps) = planner.take_ready_steps() {
                    #[cfg(feature = "performance-counters")]
                    let consume_started = std::time::Instant::now();
                    consume(steps).map_err(MaxwellFrontendDispatchError::Delivery)?;
                    #[cfg(feature = "performance-counters")]
                    {
                        consumed += consume_started.elapsed();
                    }
                }
                Ok(())
            },
        )
        .map_err(|error| match error {
            MaxwellEngineStreamError::Dispatch(error) => {
                MaxwellFrontendDispatchError::EngineDispatch(error)
            }
            MaxwellEngineStreamError::Consumer(error) => error,
        })?;
        if let Some(steps) = planner.finish_packet() {
            #[cfg(feature = "performance-counters")]
            let consume_started = std::time::Instant::now();
            consume(steps).map_err(MaxwellFrontendDispatchError::Delivery)?;
            #[cfg(feature = "performance-counters")]
            {
                consumed += consume_started.elapsed();
            }
        }
        packets.recycle(packet);
    }
    planner.recycle_mme_scratch(mme_methods, mme_parameters);
    if !any_packet {
        return Err(MaxwellFrontendDispatchError::EmptySubmission);
    }
    let plan = planner
        .finish()
        .map_err(|error| MaxwellFrontendDispatchError::Execution(Box::new(error)))?;
    #[cfg(feature = "performance-counters")]
    {
        nixe_gpu::metrics::record(nixe_gpu::metrics::Counter::FrontendSubmissions, 1);
        nixe_gpu::metrics::record(
            nixe_gpu::metrics::Counter::FrontendLoweringNanoseconds,
            started.elapsed().saturating_sub(consumed).as_nanos() as u64,
        );
    }
    Ok(plan)
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

#[cfg(test)]
mod tests;
