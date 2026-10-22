//! Direct retained-source consumption for one Maxwell frontend submission.

use std::fmt::{Display, Formatter};

use crate::engines::{MaxwellEngineStreamError, stream_maxwell_engine_packet};
use crate::execution::{MaxwellFrontend, MaxwellSubmissionPlanner};
pub use crate::pushbuffer::packet::MaxwellCommandSourceScratch;
use crate::pushbuffer::packet::{DecodedPackets, SubmissionWords};
use crate::{
    MaxwellEngineDispatchError, MaxwellFrontendDispatch, MaxwellGpuChannel, MaxwellGpuMemory,
    MaxwellPushbufferDecodeError, MaxwellSubmissionExecutionError, MaxwellSubmissionExecutionPlan,
    MaxwellSubmissionExecutionStep,
};

/// First semantic failure while consuming a retained command stream.
#[derive(Clone, Debug)]
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
    address_space: &MaxwellGpuMemory,
    cache: &mut L,
    sources: &mut MaxwellCommandSourceScratch,
    access: &mut nixe_gpu::MemoryAccess<'_>,
    synchronize: &mut dyn FnMut() -> Result<(), Box<str>>,
    consume: &mut impl FnMut(
        std::vec::Drain<'_, MaxwellSubmissionExecutionStep<L::Work>>,
    ) -> Result<(), Box<str>>,
) -> Result<MaxwellSubmissionExecutionPlan<L::Work>, MaxwellFrontendDispatchError> {
    let submission = dispatch.scheduled().submission();
    let access = std::cell::RefCell::new(access);
    let mut source_access = |ranges, operation| (access.borrow_mut())(ranges, operation);
    let mut effect_access = |ranges, operation| (access.borrow_mut())(ranges, operation);
    let packet_scratch = sources.take_packet_scratch();
    let mut packets = DecodedPackets::with_scratch(
        SubmissionWords::new(
            submission,
            address_space,
            sources,
            synchronize,
            &mut source_access,
        ),
        packet_scratch,
    );
    let mut planner = MaxwellSubmissionPlanner::new(
        address_space,
        dispatch.scheduled().frontend(),
        dispatch.scheduled().completion(),
        cache,
        &mut effect_access,
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
    mut planner: MaxwellSubmissionPlanner<'_, '_, L>,
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
        .map_err(|error| {
            log::debug!("failed Maxwell packet: {packet:x?}; MME parameters={mme_parameters:x?}; emitted={mme_methods:x?}");
            match error {
                MaxwellEngineStreamError::Dispatch(error) => {
                    MaxwellFrontendDispatchError::EngineDispatch(error)
                }
                MaxwellEngineStreamError::Consumer(error) => error,
            }
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
