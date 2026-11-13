//! Expanded ordered methods; the transport owns the persistent word storage.
use crate::*;
use nixe_gpu::GpuMethodId;
use nixe_gpu::ordered::{ENGINE, HOST, LOCATION, MACRO_END, METHOD, MULTI, ORIGIN, RAW};
#[cfg(feature = "performance-counters")]
use std::time::Duration;

pub trait MaxwellMethodSink {
    fn fetch(
        &mut self,
        request: &mut MaxwellCommandFetch,
    ) -> Result<bool, MaxwellFrontendDispatchError>;
    fn acquire(
        &mut self,
        source: MaxwellMethodSource,
        address: u64,
        payload: u32,
        operation: u8,
    ) -> Result<bool, MaxwellFrontendDispatchError>;
    fn write(
        &mut self,
        prefix: &[u64],
        bytes: &[u8],
    ) -> Result<(), MaxwellSubmissionExecutionError>;
    fn flush(&mut self);
    fn note_gpu_writer(&mut self);
    fn note_host_method(&mut self);
    fn note_sync(&mut self);
    fn is_stopped(&self) -> bool;
    #[cfg(feature = "performance-counters")]
    fn blocked_time(&self) -> Duration;
    #[cfg(feature = "performance-counters")]
    fn mme_time(&mut self, active: Duration);
}
pub(crate) struct Writer<'a, H: MaxwellMethodSink> {
    pub host: &'a mut H,
    origin: Option<MaxwellGpfifoSourceLocation>,
    offset: u64,
    engine: Option<(crate::engines::Engine, MaxwellPushbufferSubchannel)>,
}
impl<'a, H: MaxwellMethodSink> Writer<'a, H> {
    pub fn new(host: &'a mut H) -> Self {
        Self {
            host,
            origin: None,
            offset: u64::MAX,
            engine: None,
        }
    }
    pub fn flush(&mut self) {
        self.host.flush();
    }
    fn location(
        &mut self,
        source: MaxwellMethodSource,
    ) -> Result<(), MaxwellSubmissionExecutionError> {
        let mut origin = source.location();
        origin.word_offset = 0;
        if self.origin != Some(origin) {
            self.host.write(
                &[
                    ORIGIN | (u64::from(origin.entry_index) << 32),
                    origin.pushbuffer.get(),
                ],
                &[],
            )?;
            self.origin = Some(origin);
            self.offset = u64::MAX;
        }
        if self.offset != source.location().word_offset {
            self.offset = source.location().word_offset;
            self.host.write(&[LOCATION | (self.offset << 32)], &[])?;
        }
        Ok(())
    }
    pub fn select(
        &mut self,
        engine: crate::engines::Engine,
        source: MaxwellMethodSource,
        bind: bool,
    ) -> Result<(), MaxwellSubmissionExecutionError> {
        let selected = (engine, source.subchannel());
        if bind || self.engine != Some(selected) {
            self.location(source)?;
            self.host.write(
                &[ENGINE
                    | (u64::from(source.subchannel().get()) << 3)
                    | ((engine as u64) << 8)
                    | (u64::from(bind) << 16)],
                &[],
            )?;
            self.engine = Some(selected);
        }
        Ok(())
    }
    pub fn host_effect(
        &mut self,
        source: MaxwellMethodSource,
        effect: crate::MaxwellHostSynchronizationKind,
    ) -> Result<(), MaxwellSubmissionExecutionError> {
        use crate::MaxwellHostSynchronizationKind as Effect;
        self.location(source)?;
        let (kind, a, b) = match effect {
            Effect::SemaphoreAcquire { .. } => unreachable!("acquires suspend the command channel"),
            Effect::WaitForIdle { all } => (0, u64::from(all), 0),
            Effect::ReleaseFences => (1, 0, 0),
            Effect::SignalReference => (2, 0, 0),
            Effect::IncrementSyncpoint { syncpoint } => (3, u64::from(syncpoint.get()), 0),
            Effect::SemaphoreRelease {
                address,
                payload,
                short,
                wait,
            } => (
                4,
                address,
                u64::from(payload) | (u64::from(short) << 32) | (u64::from(wait) << 33),
            ),
            Effect::L2SysmemInvalidate { operand_high } => (5, u64::from(operand_high), 0),
            Effect::L2FlushDirty { operand_high } => (6, u64::from(operand_high), 0),
        };
        self.host.write(
            &[
                HOST | (kind << 3)
                    | (u64::from(source.subchannel().get()) << 6)
                    | (u64::from(source.method().0) << 9)
                    | (u64::from(source.argument()) << 32),
                a,
                b,
            ],
            &[],
        )
    }
    /// Ordinary writes need only their encoded word, without diagnostic metadata.
    pub fn raw(
        &mut self,
        subchannel: MaxwellPushbufferSubchannel,
        method: GpuMethodId,
        value: u32,
    ) -> Result<(), MaxwellSubmissionExecutionError> {
        self.host.write(
            &[RAW
                | (u64::from(subchannel.get()) << 3)
                | (u64::from(method.0) << 8)
                | (u64::from(value) << 32)],
            &[],
        )
    }
    pub fn method(
        &mut self,
        source: MaxwellMethodSource,
    ) -> Result<(), MaxwellSubmissionExecutionError> {
        self.location(source)?;
        self.host.write(
            &[METHOD
                | (u64::from(source.subchannel().get()) << 3)
                | (u64::from(source.method().0) << 8)
                | (u64::from(source.argument()) << 32)],
            &[],
        )
    }
    pub fn end_macro(
        &mut self,
        source: MaxwellMethodSource,
    ) -> Result<(), MaxwellSubmissionExecutionError> {
        self.location(source)?;
        self.host.write(&[MACRO_END], &[])
    }
    pub fn multi(
        &mut self,
        source: MaxwellMethodSource,
        data: crate::MaxwellDmaArguments<'_>,
    ) -> Result<(), MaxwellSubmissionExecutionError> {
        let mut cursor = 0;
        while cursor < data.len() {
            let count = (data.len() - cursor).min(8192);
            self.location(data.source(cursor, source.method(), source.subchannel()))?;
            let prefix = [MULTI
                | (u64::from(source.subchannel().get()) << 3)
                | (u64::from(source.method().0) << 8)
                | ((count as u64) << 32)];
            self.host
                .write(&prefix, data.slice(cursor..cursor + count).bytes())?;
            cursor += count;
        }
        Ok(())
    }
}

/// Only the preparation thread constructs this cursor. Completion accounting
/// survives packet batches, including macros and DMA packets split across entries.
pub struct MaxwellMethodExecution<'a, L: MaxwellFrontend, H: MaxwellCommandHost> {
    channel: &'a mut MaxwellGpuChannel,
    effects: crate::execution::ChannelExecution<'a, L, H>,
    origin: &'a mut MaxwellGpfifoSourceLocation,
    engine: &'a mut Option<crate::engines::Engine>,
}
/// Submission-local execution survives channel suspension without retaining any
/// mutable renderer borrow. Canonical registers stay with the hardware channel.
pub struct MaxwellMethodState {
    origin: MaxwellGpfifoSourceLocation,
    engine: Option<crate::engines::Engine>,
    progress: crate::execution::ExecutionProgress,
}
impl MaxwellMethodState {
    pub fn new(dispatch: &MaxwellFrontendDispatch, space: &MaxwellGpuMemory) -> Self {
        let mut progress = crate::execution::ExecutionProgress::default();
        progress.driver_completion_increments = if dispatch
            .scheduled()
            .submission()
            .decoded()
            .mode()
            .fence_get()
        {
            2
        } else {
            0
        };
        Self {
            origin: MaxwellGpfifoSourceLocation {
                channel: dispatch.scheduled().channel(),
                frontend: dispatch.scheduled().frontend(),
                entry_index: 0,
                pushbuffer: space.address(0).expect("zero GPU address"),
                word_offset: 0,
            },
            engine: None,
            progress,
        }
    }
}
impl<'a, L: MaxwellFrontend, H: MaxwellCommandHost> MaxwellMethodExecution<'a, L, H> {
    pub fn new(
        dispatch: &'a MaxwellFrontendDispatch,
        channel: &'a mut MaxwellGpuChannel,
        space: &'a std::cell::RefCell<MaxwellGpuMemory>,
        rasterizer: &'a std::cell::RefCell<L>,
        host: &'a mut H,
        state: &'a mut MaxwellMethodState,
    ) -> Self {
        let effects = crate::execution::ChannelExecution::new(
            space,
            dispatch.scheduled().frontend(),
            dispatch.scheduled().completion(),
            rasterizer,
            host,
            &mut state.progress,
        );
        // Split the cursor fields from effect accounting so both are borrowed
        // by the execution without moving or copying channel state.
        Self {
            channel,
            effects,
            origin: &mut state.origin,
            engine: &mut state.engine,
        }
    }
    pub fn consume<T>(
        &mut self,
        batch: &mut nixe_gpu::ordered::Words<'_, T>,
    ) -> Result<(), MaxwellFrontendDispatchError> {
        while !batch.remaining().is_empty() {
            self.effects
                .host
                .service_progress()
                .map_err(|e| MaxwellFrontendDispatchError::Execution(Box::new(e)))?;
            let words = batch.remaining();
            let mut cursor = 0;
            let word = words[cursor];
            cursor += 1;
            match word & 7 {
                LOCATION => self.origin.word_offset = word >> 32,
                ORIGIN => {
                    self.origin.entry_index = (word >> 32) as u32;
                    self.origin.pushbuffer = self
                        .effects
                        .address_space
                        .borrow()
                        .address(words[cursor])
                        .expect("admitted command address");
                    cursor += 1;
                }
                MACRO_END => crate::engines::flush_macro(self.channel, &mut self.effects)
                    .map_err(dispatch_error)?,
                ENGINE => {
                    let engine = match (word >> 8) & 0xff {
                        0 => crate::engines::Engine::ThreeD,
                        1 => crate::engines::Engine::Compute,
                        2 => crate::engines::Engine::InlineToMemory,
                        3 => crate::engines::Engine::TwoD,
                        4 => crate::engines::Engine::Copy,
                        _ => unreachable!("admitted engine"),
                    };
                    let subchannel =
                        MaxwellPushbufferSubchannel::try_new(((word >> 3) & 7) as u8).unwrap();
                    let source = MaxwellMethodSource::dma(
                        *self.origin,
                        subchannel,
                        GpuMethodId(0),
                        engine.class().0,
                    );
                    crate::engines::select_engine(
                        self.channel,
                        source,
                        engine,
                        word & (1 << 16) != 0,
                    )
                    .map_err(dispatch_error)?;
                    *self.engine = Some(engine);
                    if engine != crate::engines::Engine::ThreeD {
                        self.effects.sync_guest_memory();
                    }
                }
                HOST => {
                    self.effects.sync_guest_memory();
                    use crate::MaxwellHostSynchronizationKind as Effect;
                    let a = words[cursor];
                    let b = words[cursor + 1];
                    cursor += 2;
                    let effect = match (word >> 3) & 7 {
                        0 => Effect::WaitForIdle { all: a != 0 },
                        1 => Effect::ReleaseFences,
                        2 => Effect::SignalReference,
                        3 => Effect::IncrementSyncpoint {
                            syncpoint: nixe_gpu::GuestSyncpointId::new(a as u32),
                        },
                        4 => Effect::SemaphoreRelease {
                            address: a,
                            payload: b as u32,
                            short: b & (1 << 32) != 0,
                            wait: b & (1 << 33) != 0,
                        },
                        5 => Effect::L2SysmemInvalidate {
                            operand_high: a as u8,
                        },
                        6 => Effect::L2FlushDirty {
                            operand_high: a as u8,
                        },
                        _ => unreachable!("admitted host effect"),
                    };
                    let source = MaxwellMethodSource::dma(
                        *self.origin,
                        MaxwellPushbufferSubchannel::try_new(((word >> 6) & 7) as u8).unwrap(),
                        GpuMethodId(((word >> 9) & 0xffff) as u32),
                        (word >> 32) as u32,
                    );
                    crate::engines::execute_host(self.channel, source, effect, &mut self.effects)
                        .map_err(dispatch_error)?;
                }
                RAW => {
                    let state = self.channel.three_d_mut();
                    let method = GpuMethodId(((word >> 8) & 0xffff) as u32);
                    let argument = (word >> 32) as u32;
                    if state.operation_state().register_words()[(method.0 / 4) as usize] != argument
                    {
                        crate::engines::threed::draw::before_write(
                            state,
                            method,
                            &mut self.effects,
                        )
                        .map_err(|e| MaxwellFrontendDispatchError::Execution(Box::new(e)))?;
                        state.write_register(method, argument);
                    }
                }
                kind => {
                    let method = GpuMethodId(((word >> 8) & 0xffff) as u32);
                    let subchannel =
                        MaxwellPushbufferSubchannel::try_new(((word >> 3) & 7) as u8).unwrap();
                    let argument = (word >> 32) as u32;
                    let source =
                        MaxwellMethodSource::dma(*self.origin, subchannel, method, argument);
                    if kind == METHOD
                        && *self.engine == Some(crate::engines::Engine::ThreeD)
                        && (0x2390..=0x23fc).contains(&method.0)
                        && words.get(1).is_some_and(|next| {
                            next & 7 == METHOD
                                && ((next >> 3) & 7) == u64::from(subchannel.get())
                                && (0x2390..=0x23fc).contains(&(((next >> 8) & 0xffff) as u32))
                        })
                    {
                        let mut methods = [GpuMethodId(0); 64];
                        let mut arguments = [0u32; 64];
                        methods[0] = method;
                        arguments[0] = argument;
                        let mut count = 1;
                        while count < 64 && count < words.len() {
                            let next = words[count];
                            let method = GpuMethodId(((next >> 8) & 0xffff) as u32);
                            if next & 7 != METHOD
                                || ((next >> 3) & 7) != u64::from(subchannel.get())
                                || !(0x2390..=0x23fc).contains(&method.0)
                            {
                                break;
                            }
                            methods[count] = method;
                            arguments[count] = (next >> 32) as u32;
                            count += 1;
                        }
                        crate::engines::threed::call_constant_buffer_words(
                            self.channel,
                            source,
                            &methods[..count],
                            &arguments[..count],
                            &mut self.effects,
                        )
                        .map_err(dispatch_error)?;
                        cursor = count;
                    } else if kind == MULTI {
                        let count = argument as usize;
                        let data = &words[cursor..cursor + count.div_ceil(2)];
                        // The transport stores literal words in native little-endian order.
                        let bytes = unsafe {
                            std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), count * 4)
                        };
                        let args = crate::MaxwellDmaArguments::owned(*self.origin, bytes);
                        crate::engines::execute_multi(
                            self.channel,
                            self.engine.expect("selected engine"),
                            source,
                            args,
                            &mut self.effects,
                        )
                        .map_err(dispatch_error)?;
                        cursor += count.div_ceil(2);
                    } else {
                        crate::engines::execute_method(
                            self.channel,
                            self.engine.expect("selected engine"),
                            source,
                            &mut self.effects,
                        )
                        .map_err(dispatch_error)?;
                    }
                }
            }
            batch.advance(cursor);
        }
        Ok(())
    }
    pub fn finish(
        self,
        dispatch: &MaxwellFrontendDispatch,
    ) -> Result<(), MaxwellFrontendDispatchError> {
        let mode = dispatch.scheduled().submission().decoded().mode();
        self.effects
            .finish(mode.fence_get() && !mode.suppress_wfi())
            .map_err(|e| MaxwellFrontendDispatchError::Execution(Box::new(e)))
    }
}
pub(crate) fn dispatch_error(
    e: crate::engines::MaxwellEngineStreamError,
) -> MaxwellFrontendDispatchError {
    match e {
        crate::engines::MaxwellEngineStreamError::Dispatch(e) => {
            MaxwellFrontendDispatchError::EngineDispatch(e)
        }
        crate::engines::MaxwellEngineStreamError::Execution(e) => {
            MaxwellFrontendDispatchError::Execution(e)
        }
    }
}
