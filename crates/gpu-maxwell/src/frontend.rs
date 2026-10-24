//! Direct retained-source consumption for one Maxwell frontend submission.

use std::fmt::{Display, Formatter};

use crate::engines::{MaxwellEngineStreamError, stream_maxwell_engine_packet};
use crate::execution::{MaxwellCommandEffects, MaxwellFrontend};
use crate::pushbuffer::packet::CommandStream;
pub use crate::pushbuffer::packet::MaxwellCommandSourceScratch;
use crate::{
    MaxwellEngineDispatchError, MaxwellFrontendDispatch, MaxwellGpuChannel, MaxwellGpuMemory,
    MaxwellPushbufferDecodeError, MaxwellSubmissionExecutionError,
};

/// First semantic failure while consuming a command stream.
#[derive(Clone, Debug)]
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

/// Fetch each entry and consume engine effects in order. Packet continuation
/// survives entry fetches; control work runs at effects and bounded word quanta.
pub async fn stream_maxwell_frontend<L: MaxwellFrontend, H: crate::MaxwellCommandHost>(
    dispatch: &MaxwellFrontendDispatch,
    channel: &mut MaxwellGpuChannel,
    address_space: &MaxwellGpuMemory,
    cache: &std::cell::RefCell<L>,
    sources: &mut MaxwellCommandSourceScratch,
    host: &mut H,
) -> Result<(), MaxwellFrontendDispatchError> {
    #[cfg(feature = "performance-counters")]
    let started = std::time::Instant::now();
    let submission = dispatch.scheduled().submission();
    let frontend = dispatch.scheduled().frontend();
    let mut effects = MaxwellCommandEffects::new(
        address_space,
        frontend,
        dispatch.scheduled().completion(),
        cache,
        host,
    );
    if submission.decoded().mode().fence_get() {
        effects.set_driver_completion_increments(2);
    }
    let mut mme_methods = None;
    let mut mme_parameters = effects.take_mme_scratch();
    let mut stream = CommandStream::default();
    let mut quantum = 0;
    let mut any = false;
    let execution_error = |error| MaxwellFrontendDispatchError::Execution(Box::new(error));
    let result = async {
        loop {
            if stream.needs_entry() && stream.next_entry_waits(submission) {
                crate::engines::finish_maxwell_engine_stream(channel, &mut async |event| {
                    effects.push_event(event).await?;
                    effects.control()
                })
                .await
                .map_err(execution_error)?;
            }
            if stream.needs_entry()
                && !stream
                    .load_entry(submission, address_space, sources, effects.host)
                    .map_err(|error| MaxwellFrontendDispatchError::PacketDecode(error.into()))?
            {
                break;
            }
            let (packet, words) = stream
                .next(submission, 4096 - quantum)
                .map_err(MaxwellFrontendDispatchError::PacketDecode)?;
            any = true;
            stream_maxwell_engine_packet(
                channel,
                frontend,
                packet,
                #[cfg(test)]
                None,
                &mut mme_methods,
                &mut mme_parameters,
                &mut async |event| {
                    effects.push_event(event).await.map_err(execution_error)?;
                    effects.control().map_err(execution_error)
                },
            )
            .await
            .map_err(|error| match error {
                MaxwellEngineStreamError::Dispatch(error) => {
                    MaxwellFrontendDispatchError::EngineDispatch(error)
                }
                MaxwellEngineStreamError::Consumer(error) => error,
            })?;
            quantum += words;
            if quantum >= 4096 {
                effects.control().map_err(execution_error)?;
                quantum = 0;
            }
        }
        stream
            .finish(sources)
            .map_err(MaxwellFrontendDispatchError::PacketDecode)?;
        crate::engines::finish_maxwell_engine_stream(channel, &mut async |event| {
            effects.push_event(event).await?;
            effects.control()
        })
        .await
        .map_err(execution_error)?;
        Ok(())
    }
    .await;
    effects.recycle_mme_scratch(mme_parameters);
    result?;
    if !any {
        return Err(MaxwellFrontendDispatchError::EmptySubmission);
    }
    #[cfg(feature = "performance-counters")]
    {
        nixe_gpu::metrics::record(nixe_gpu::metrics::Counter::FrontendSubmissions, 1);
        nixe_gpu::metrics::record(
            nixe_gpu::metrics::Counter::FrontendLoweringNanoseconds,
            started
                .elapsed()
                .saturating_sub(effects.consumed)
                .as_nanos() as u64,
        );
    }
    effects.finish().map_err(execution_error)?;
    // A short list is also a control boundary; many short lists must not reset
    // the fairness quantum without ever servicing a waiting CPU request.
    host.control().map_err(execution_error)
}
