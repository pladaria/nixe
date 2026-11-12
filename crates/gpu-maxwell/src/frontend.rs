//! Direct retained-source consumption for one Maxwell frontend submission.

use std::fmt::{Display, Formatter};

use crate::engines::{Engine, MaxwellEngineStreamError};
use crate::ordered::{Writer, dispatch_error};
use crate::pushbuffer::dma::CommandReader;
pub use crate::pushbuffer::dma::MaxwellCommandFetch;
use crate::{
    MaxwellAddressSpaceId, MaxwellEngineDispatchError, MaxwellFrontendDispatch,
    MaxwellGpfifoSourceError, MaxwellMethodSink, MaxwellMethodSource,
    MaxwellSubmissionExecutionError,
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

/// The command thread owns packet continuation, macro code and macro-visible
/// registers. The preparation thread owns a separate canonical engine bank.
pub struct MaxwellCommandProcessor {
    profile: crate::MaxwellGpuProfile,
    bindings: crate::MaxwellChannelFrontendState,
    commands: crate::pushbuffer::dma::CommandState,
    acquire: Option<(MaxwellMethodSource, u64, u32, u8)>,
    three_d: crate::engines::threed::front::Registers,
}
impl MaxwellCommandProcessor {
    pub fn new(configuration: crate::MaxwellChannelConfiguration, jit: bool) -> Self {
        Self {
            profile: configuration.profile(),
            bindings: Default::default(),
            commands: Default::default(),
            acquire: None,
            three_d: crate::engines::threed::front::Registers::new(jit),
        }
    }
    fn call<H: MaxwellMethodSink>(
        &mut self,
        source: MaxwellMethodSource,
        last: bool,
        writer: &mut Writer<'_, H>,
    ) -> Result<(), MaxwellEngineStreamError> {
        let method = source.method();
        if method.0 < 0x100 {
            if self.commands.mme_method.is_some() {
                return Err(self.interrupted(source));
            }
            self.three_d.check_puller(source)?;
            let effect = crate::pushbuffer::dispatch::call_puller_method(
                &self.profile,
                source,
                &mut self.bindings,
            )
            .map_err(crate::MaxwellEngineDispatchError::from)?;
            if method.0 == 0 {
                writer.select(
                    self.bindings
                        .subchannel_engine(source.subchannel())
                        .expect("admitted engine binding"),
                    source,
                    true,
                )?;
            }
            if let Some(crate::MaxwellHostSynchronizationKind::SemaphoreAcquire {
                address,
                payload,
                operation,
            }) = effect
            {
                self.acquire = Some((source, address, payload, operation));
                writer.flush();
                writer.host.note_sync();
                return Ok(());
            }
            if let Some(effect) = effect {
                writer.host_effect(source, effect)?;
                writer.host.note_host_method();
                if matches!(
                    effect,
                    crate::MaxwellHostSynchronizationKind::WaitForIdle { .. }
                        | crate::MaxwellHostSynchronizationKind::ReleaseFences
                        | crate::MaxwellHostSynchronizationKind::SemaphoreRelease {
                            wait: true,
                            ..
                        }
                ) {
                    writer.host.note_sync();
                }
            }
            return Ok(());
        }
        let engine = self
            .bindings
            .subchannel_engine(source.subchannel())
            .ok_or_else(|| {
                crate::MaxwellEngineDispatchError::from(
                    crate::MaxwellMethodDispatchError::UnboundSubchannel { source },
                )
            })?;
        writer.select(engine, source, false)?;
        if engine == Engine::ThreeD && crate::engines::threed::is_mme_aperture(method) {
            self.admit_macro(source)?;
            self.commands.mme_parameters.push(source.argument());
            if last {
                self.run_macro(writer)?;
            }
        } else {
            if self.commands.mme_method.is_some() {
                return Err(self.interrupted(source));
            }
            if engine == Engine::ThreeD {
                self.three_d.write(source, writer)?;
            } else {
                writer.method(source)?;
                // Copy and compute effects can produce command lists.
                // Their register setup does not constitute a memory producer.
                match engine {
                    Engine::Compute if matches!(method.0, 0x110 | 0x1b4 | 0x2bc) => {
                        writer.host.note_gpu_writer()
                    }
                    Engine::Copy if method.0 == 0x300 => writer.host.note_gpu_writer(),
                    _ => {}
                }
            }
        }
        Ok(())
    }
    fn interrupted(&self, source: MaxwellMethodSource) -> MaxwellEngineStreamError {
        crate::MaxwellEngineDispatchError::InvalidMethodEncoding {
            source,
            method_name: "MME",
            reason: "macro arguments interrupted before the final parameter",
        }
        .into()
    }
    fn admit_macro(&mut self, source: MaxwellMethodSource) -> Result<(), MaxwellEngineStreamError> {
        let method = source.method();
        let expected = self
            .commands
            .mme_method
            .as_ref()
            .map_or(method.0, |s| s.method().0);
        let starts = self.commands.mme_method.is_none() && (method.0 - 0x3800) & 7 == 0;
        let continues = self.commands.mme_method.is_some()
            && (method.0 == expected || method.0 == expected + 4);
        if !starts && !continues {
            return Err(crate::MaxwellEngineDispatchError::MmeExecution {
                source,
                error: crate::MaxwellThreeDMmeExecutionError::DataWithoutCall,
            }
            .into());
        }
        if starts {
            self.commands.mme_method = Some(source);
            self.commands.mme_parameters.clear();
        }
        Ok(())
    }
    fn run_macro<H: MaxwellMethodSink>(
        &mut self,
        writer: &mut Writer<'_, H>,
    ) -> Result<(), MaxwellEngineStreamError> {
        let source = self.commands.mme_method.take().unwrap();
        self.three_d
            .run(source, &self.commands.mme_parameters, writer)?;
        self.commands.mme_parameters.clear();
        Ok(())
    }
    fn entry<H: MaxwellMethodSink>(
        &mut self,
        reader: &CommandReader,
        writer: &mut Writer<'_, H>,
        word_offset: &mut usize,
    ) -> Result<bool, MaxwellEngineStreamError> {
        let mut index = *word_offset;
        while index < reader.len() {
            if self.commands.dma.method_count == 0 {
                let word = reader.word(index);
                let immediate = match word >> 29 {
                    1 => {
                        self.commands.dma.set_state(word);
                        self.commands.dma.non_incrementing = false;
                        self.commands.dma.increment_once = false;
                        None
                    }
                    3 => {
                        self.commands.dma.set_state(word);
                        self.commands.dma.non_incrementing = true;
                        self.commands.dma.increment_once = false;
                        None
                    }
                    4 => {
                        self.commands.dma.set_method(word);
                        self.commands.dma.non_incrementing = true;
                        self.commands.dma.increment_once = false;
                        Some((word >> 16) & 0x1fff)
                    }
                    5 => {
                        self.commands.dma.set_state(word);
                        self.commands.dma.non_incrementing = false;
                        self.commands.dma.increment_once = true;
                        None
                    }
                    _ => {
                        index += 1;
                        continue;
                    }
                };
                if let Some(argument) = immediate {
                    let source = MaxwellMethodSource::dma(
                        reader.location_at(index * 4),
                        self.commands.dma.subchannel,
                        self.commands.dma.method,
                        argument,
                    );
                    self.call(source, true, writer)?;
                }
                index += 1;
                if self.acquire.is_some() {
                    *word_offset = index;
                    return Ok(false);
                }
                continue;
            }
            let method = self.commands.dma.method;
            let subchannel = self.commands.dma.subchannel;
            let count = (self.commands.dma.method_count as usize)
                .min(reader.len() - index)
                .min(if method.0 < 0x100 { 1 } else { usize::MAX });
            if self.commands.dma.non_incrementing {
                let data = reader.arguments(index, count);
                let source = data.source(0, method, subchannel);
                let last = count as u32 == self.commands.dma.method_count;
                let engine = self.bindings.subchannel_engine(subchannel);
                if let Some(engine) = engine {
                    writer.select(engine, source, false)?;
                }
                if engine == Some(Engine::ThreeD) && crate::engines::threed::is_mme_aperture(method)
                {
                    self.admit_macro(source)?;
                    data.append_arguments(&mut self.commands.mme_parameters)
                        .map_err(|_| crate::MaxwellEngineDispatchError::ResourceExhausted)?;
                    if last {
                        self.run_macro(writer)?;
                    }
                } else {
                    let multi =
                        if self.commands.mme_method.is_none() && engine == Some(Engine::ThreeD) {
                            self.three_d.multi(source, data, writer)?
                        } else if self.commands.mme_method.is_none()
                            && method.0 == 0x1b4
                            && matches!(engine, Some(Engine::Compute | Engine::InlineToMemory))
                        {
                            writer.multi(source, data)?;
                            if engine == Some(Engine::Compute) {
                                writer.host.note_gpu_writer();
                            }
                            true
                        } else {
                            false
                        };
                    if !multi {
                        for i in 0..count {
                            self.call(
                                data.source(i, method, subchannel),
                                last && i + 1 == count,
                                writer,
                            )?;
                        }
                    }
                }
                self.commands.dma.method_count -= count as u32;
                index += count;
            } else {
                let source = MaxwellMethodSource::dma(
                    reader.location_at(index * 4),
                    subchannel,
                    method,
                    reader.word(index),
                );
                self.call(source, self.commands.dma.method_count == 1, writer)?;
                self.commands.dma.method.0 += 4;
                self.commands.dma.non_incrementing |= self.commands.dma.increment_once;
                self.commands.dma.method_count -= 1;
                index += 1;
            }
            if self.acquire.is_some() {
                *word_offset = index;
                return Ok(false);
            }
        }
        *word_offset = 0;
        Ok(true)
    }
}
/// Fetch coherent command prefixes, then expand macros into ordered methods.
/// SYNC_PROCEED permits prefetch across segment boundaries. SYNC_WAIT requires
/// prior host methods to have run before fetching, including zero-length entries.
/// https://github.com/NVIDIA/open-gpu-doc/blob/ab27fc22db5de0d02a4cabe08e555663b62db4d4/manuals/volta/gv100/dev_pbdma.ref.txt#L104-L138
pub struct MaxwellCommandStream {
    space: MaxwellAddressSpaceId,
    origin: crate::MaxwellGpfifoSourceLocation,
    entries: std::sync::Arc<[crate::MaxwellGpfifoEntry]>,
    first: usize,
    fetch: Option<MaxwellCommandFetch>,
    ready: bool,
    entry: usize,
    offset: usize,
    word_offset: usize,
    bytes: Vec<u8>,
}
impl MaxwellCommandStream {
    pub fn new(
        dispatch: &MaxwellFrontendDispatch,
        space: MaxwellAddressSpaceId,
        bytes: Vec<u8>,
    ) -> Self {
        let submission = dispatch.scheduled().submission();
        let entries = submission.decoded().shared_entries();
        Self {
            space,
            origin: crate::MaxwellGpfifoSourceLocation {
                channel: submission.channel(),
                frontend: submission.frontend(),
                entry_index: 0,
                pushbuffer: nixe_gpu::GpuVirtualAddress::try_new(0, 40).expect("zero address"),
                word_offset: 0,
            },
            entries,
            first: 0,
            fetch: None,
            ready: false,
            entry: 0,
            offset: 0,
            word_offset: 0,
            bytes,
        }
    }
    pub fn into_storage(self) -> Vec<u8> {
        self.bytes
    }
    pub fn run<H: MaxwellMethodSink>(
        &mut self,
        channel: &mut MaxwellCommandProcessor,
        host: &mut H,
    ) -> Result<bool, MaxwellFrontendDispatchError> {
        let mut writer = Writer::new(host);
        'prefix: while self.first < self.entries.len() {
            writer.flush();
            if writer.host.is_stopped() {
                return Err(MaxwellFrontendDispatchError::Execution(Box::new(
                    MaxwellSubmissionExecutionError::Native(
                        "GPU command processor is shutting down".into(),
                    ),
                )));
            }
            if self.fetch.is_none() {
                let mut end = self.first + 1;
                let mut bytes = self.entries[self.first].byte_count();
                while end < self.entries.len()
                    && bytes < 1024 * 1024
                    && end - self.first < 1024
                    && self.entries[end].sync() != crate::MaxwellGpfifoSyncMode::Wait
                {
                    bytes += self.entries[end].byte_count();
                    end += 1;
                }
                self.fetch = Some(MaxwellCommandFetch {
                    space: self.space,
                    origin: crate::MaxwellGpfifoSourceLocation {
                        entry_index: self.first as u32,
                        pushbuffer: self.entries[self.first].address(),
                        ..self.origin
                    },
                    entries: std::sync::Arc::clone(&self.entries),
                    range: self.first..end,
                    bytes: std::mem::take(&mut self.bytes),
                    copied: 0,
                    generation: None,
                });
            }
            let fetched = self.fetch.as_mut().unwrap();
            if !self.ready {
                if !writer.host.fetch(fetched)? {
                    return Ok(false);
                }
                self.ready = true;
            }
            if self.entry < fetched.range.start {
                self.entry = fetched.range.start;
            }
            while self.entry < fetched.range.end {
                if let Some((source, address, payload, operation)) = channel.acquire {
                    if !writer.host.acquire(source, address, payload, operation)? {
                        return Ok(false);
                    }
                    channel.acquire = None;
                    // A producer may have filled the remainder while this
                    // acquire waited. Preserve consumed packet state, reread
                    // only the bytes following the acquire before resuming.
                    fetched.invalidate_tail(self.offset + self.word_offset * 4);
                    self.ready = false;
                    continue 'prefix;
                }
                let index = self.entry;
                let offset = self.offset;
                let entry = &fetched.entries[index];
                let next = offset + entry.byte_count() as usize;
                let reader = CommandReader::new(
                    crate::MaxwellGpfifoSourceLocation {
                        entry_index: index as u32,
                        pushbuffer: entry.address(),
                        ..self.origin
                    },
                    &fetched.bytes[offset..next],
                );
                if !channel
                    .entry(&reader, &mut writer, &mut self.word_offset)
                    .map_err(dispatch_error)?
                {
                    return Ok(false);
                }
                self.offset = next;
                self.entry += 1;
            }
            let fetched = self.fetch.take().unwrap();
            self.first = fetched.range.end;
            self.bytes = fetched.bytes;
            self.ready = false;
            self.offset = 0;
        }
        writer.flush();
        Ok(true)
    }
}
