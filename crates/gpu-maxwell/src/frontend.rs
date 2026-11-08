//! Direct retained-source consumption for one Maxwell frontend submission.

use std::fmt::{Display, Formatter};

use crate::engines::{MaxwellEngineStreamError, process_maxwell_commands};
use crate::execution::{ChannelExecution, MaxwellFrontend};
use crate::pushbuffer::dma::CommandReader;
pub use crate::pushbuffer::dma::MaxwellCommandSourceScratch;
use crate::{
    MaxwellEngineDispatchError, MaxwellFrontendDispatch, MaxwellGpfifoSourceError,
    MaxwellGpuChannel, MaxwellGpuMemory, MaxwellSubmissionExecutionError,
};

/// First semantic failure while consuming a command stream.
#[derive(Clone, Debug)]
pub enum MaxwellFrontendDispatchError {
    // Effects and MME sends carry this error type through their synchronous
    // results. Keep packet provenance behind the failure-only allocation too.
    CommandSource(Box<MaxwellGpfifoSourceError>),
    EngineDispatch(Box<MaxwellEngineDispatchError>),
    Execution(Box<MaxwellSubmissionExecutionError>),
}

impl Display for MaxwellFrontendDispatchError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CommandSource(error) => write!(formatter, "command source failed: {error}"),
            Self::EngineDispatch(error) => write!(formatter, "engine dispatch failed: {error}"),
            Self::Execution(error) => write!(formatter, "frontend lowering failed: {error}"),
        }
    }
}

impl std::error::Error for MaxwellFrontendDispatchError {}

/// Fetch each entry and consume engine effects in order. The channel DMA state
/// survives entry fetches; entries check shutdown without servicing CPU work.
/// CPU requests run after native draws or at the outer GPU inbox boundary.
/// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/dma_pusher.cpp#L34-L45
/// The executor binds the channel before entering DispatchCalls.
pub fn stream_maxwell_frontend<L: MaxwellFrontend, H: crate::MaxwellCommandHost>(
    dispatch: &MaxwellFrontendDispatch,
    channel: &mut MaxwellGpuChannel,
    address_space: &std::cell::RefCell<MaxwellGpuMemory>,
    cache: &std::cell::RefCell<L>,
    sources: &mut MaxwellCommandSourceScratch,
    host: &mut H,
) -> Result<(), MaxwellFrontendDispatchError> {
    #[cfg(feature = "performance-counters")]
    let started = std::time::Instant::now();
    let submission = dispatch.scheduled().submission();
    let frontend = dispatch.scheduled().frontend();
    let mut effects = ChannelExecution::new(
        address_space,
        frontend,
        dispatch.scheduled().completion(),
        cache,
        host,
    );
    if submission.decoded().mode().fence_get() {
        effects.set_driver_completion_increments(2);
    }
    let mut commands = std::mem::take(&mut channel.commands);
    let mut stream = CommandReader::new(address_space.borrow().machine().clone());
    let execution_error = |error| MaxwellFrontendDispatchError::Execution(Box::new(error));
    let result = (|| {
        loop {
            if effects.host.is_stopped() {
                return Err(execution_error(MaxwellSubmissionExecutionError::Native(
                    "GPU owner is shutting down".into(),
                )));
            }
            if !stream
                .load_entry(submission, &address_space.borrow(), sources, effects.host)
                .map_err(|error| MaxwellFrontendDispatchError::CommandSource(Box::new(error)))?
            {
                break;
            }
            let engine_error = |error| match error {
                MaxwellEngineStreamError::Dispatch(error) => {
                    MaxwellFrontendDispatchError::EngineDispatch(error)
                }
                MaxwellEngineStreamError::Execution(error) => {
                    MaxwellFrontendDispatchError::Execution(error)
                }
            };
            let mut index = 0;
            process_maxwell_commands(channel, &stream, &mut commands, &mut index, &mut effects)
                .map_err(engine_error)?;
        }
        Ok(())
    })();
    stream.finish(sources);
    channel.commands = commands;
    result?;
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
    effects
        .finish(
            submission.decoded().mode().fence_get() && !submission.decoded().mode().suppress_wfi(),
        )
        .map_err(execution_error)?;
    // The outer GPU loop handles pending requests after returning the channel
    // and finishing its command list, including lists with no native draws.
    Ok(())
}
