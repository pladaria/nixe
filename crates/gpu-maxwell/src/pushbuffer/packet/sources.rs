//! Consume live contiguous command RAM; gather scattered/device sources once.
use super::*;
use crate::MaxwellResolvedRange;

/// Reusable storage owned by the GPU command processor, never a native worker.
#[derive(Default)]
pub struct MaxwellCommandSourceScratch {
    bytes: Vec<u8>,
}
impl MaxwellCommandSourceScratch {
    /// Keep the largest completed scratch when another channel ran during a wait.
    pub fn recycle(&mut self, completed: Self) {
        if completed.bytes.capacity() > self.bytes.capacity() {
            *self = completed;
        }
    }
}

/// One entry's live physical storage or coherent gathered bytes. A borrowed
/// method span cannot outlive this owner or its physical pin.
pub(crate) enum EntryWords {
    Ram {
        address: nixe_memory::HostAddress,
        _pin: nixe_memory::AllocationRange,
    },
    Gathered(Vec<u8>),
}
impl EntryWords {
    pub(crate) fn read(&self, offset: usize) -> u32 {
        match self {
            Self::Ram { address, .. } => {
                // SAFETY: the parser bounds each borrowed span by the pinned
                // entry and reads owned scalars without forming RAM references.
                unsafe { address.read_u32(offset) }
            }
            Self::Gathered(bytes) => {
                u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
            }
        }
    }
}

/// Hardware packet continuation and the current entry; no future entry is read.
#[derive(Default)]
pub(crate) struct CommandStream {
    next_entry: usize,
    entry: usize,
    offset: usize,
    end: usize,
    segment: usize,
    direct: Option<crate::MaxwellResolvedMapping>,
    source: Option<MaxwellResolvedRange>,
    words: Option<EntryWords>,
    packet: Option<MethodHeader>,
    consumed: u16,
    immediate: Option<MaxwellDecodedMethodPacket>,
}
impl CommandStream {
    pub(crate) fn needs_entry(&self) -> bool {
        self.offset == self.end
    }
    pub(crate) fn next_entry_waits(&self, submission: &MaxwellValidatedGpfifoSubmission) -> bool {
        for entry in &submission.decoded().entries()[self.next_entry..] {
            if entry.sync() == crate::MaxwellGpfifoSyncMode::Wait {
                return true;
            }
            if entry.word_count() != 0 {
                break;
            }
        }
        false
    }
    pub(crate) fn load_entry<H: crate::MaxwellCommandHost>(
        &mut self,
        submission: &MaxwellValidatedGpfifoSubmission,
        memory: &MaxwellGpuMemory,
        scratch: &mut MaxwellCommandSourceScratch,
        host: &mut H,
    ) -> Result<bool, MaxwellGpfifoSourceError> {
        if let Some(EntryWords::Gathered(mut bytes)) = self.words.take() {
            bytes.clear();
            scratch.bytes = bytes;
        }
        self.direct = None;
        self.source = None;
        while let Some(entry) = submission.decoded().entries().get(self.next_entry).copied() {
            self.entry = self.next_entry;
            self.next_entry += 1;
            self.offset = 0;
            self.end = entry.byte_count() as usize;
            self.segment = 0;
            if self.end == 0 {
                continue;
            }
            if let Some((segment, address, pin)) = memory
                .command_ram_direct(entry.address(), entry.byte_count())
                .map_err(|error| MaxwellGpfifoSourceError::Resolution {
                    channel: submission.channel(),
                    frontend: submission.frontend(),
                    entry_index: self.entry as u32,
                    pushbuffer: entry.address(),
                    address_space_generation: memory.mapping_generation(),
                    error,
                })?
            {
                self.direct = Some(segment);
                self.words = Some(EntryWords::Ram { address, _pin: pin });
                return Ok(true);
            }
            let source = memory
                .resolve_range(
                    entry.address(),
                    entry.byte_count(),
                    nixe_memory::MemoryPermissions::READ,
                )
                .map_err(|error| MaxwellGpfifoSourceError::Resolution {
                    channel: submission.channel(),
                    frontend: submission.frontend(),
                    entry_index: self.entry as u32,
                    pushbuffer: entry.address(),
                    address_space_generation: memory.mapping_generation(),
                    error,
                })?;
            let first = source
                .segments()
                .first()
                .expect("nonempty entry has a mapping");
            let location = MaxwellGpfifoSourceLocation {
                channel: submission.channel(),
                frontend: submission.frontend(),
                entry_index: self.entry as u32,
                pushbuffer: entry.address(),
                word_offset: 0,
                mapping: first.mapping().id(),
                generation: first.mapping().generation(),
            };
            self.words = Some(
                if let Some((address, pin)) = memory
                    .command_ram(&source)
                    .map_err(|error| MaxwellGpfifoSourceError::Access { location, error })?
                {
                    EntryWords::Ram { address, _pin: pin }
                } else {
                    scratch.bytes.clear();
                    scratch
                        .bytes
                        .try_reserve(self.end)
                        .map_err(|_| MaxwellGpfifoSourceError::ResourceExhausted)?;
                    scratch.bytes.resize(self.end, 0);
                    memory
                        .read_resolved(&source, &mut scratch.bytes, &mut |ranges, operation| {
                            host.access(ranges, operation)
                        })
                        .map_err(|error| MaxwellGpfifoSourceError::Access { location, error })?;
                    EntryWords::Gathered(std::mem::take(&mut scratch.bytes))
                },
            );
            self.source = Some(source);
            return Ok(true);
        }
        Ok(false)
    }
    fn location(
        &mut self,
        submission: &MaxwellValidatedGpfifoSubmission,
    ) -> (MaxwellGpfifoSourceLocation, usize) {
        let entry = submission.decoded().entries()[self.entry];
        let address = entry.address().get() + self.offset as u64;
        let segments = if let Some(segment) = &self.direct {
            std::slice::from_ref(segment)
        } else {
            self.source.as_ref().unwrap().segments()
        };
        while address >= segments[self.segment].gpu_offset().get() + segments[self.segment].size() {
            self.segment += 1;
        }
        let segment = &segments[self.segment];
        let mapping = segment.mapping();
        (
            MaxwellGpfifoSourceLocation {
                channel: submission.channel(),
                frontend: submission.frontend(),
                entry_index: self.entry as u32,
                pushbuffer: entry.address(),
                word_offset: self.offset as u64 / 4,
                mapping: mapping.id(),
                generation: mapping.generation(),
            },
            (segment.gpu_offset().get() + segment.size() - address) as usize / 4,
        )
    }
    pub(crate) fn next<'a>(
        &'a mut self,
        submission: &MaxwellValidatedGpfifoSubmission,
        budget: usize,
    ) -> Result<(PacketView<'a>, usize), MaxwellPushbufferDecodeError> {
        self.next_at(budget, |stream| stream.location(submission))
    }
    fn next_at<'a>(
        &'a mut self,
        budget: usize,
        mut location: impl FnMut(&mut Self) -> (MaxwellGpfifoSourceLocation, usize),
    ) -> Result<(PacketView<'a>, usize), MaxwellPushbufferDecodeError> {
        let mut header_words = 0;
        if self.packet.is_none() {
            let (source, _) = location(self);
            let word =
                MaxwellPushbufferWord::new(self.words.as_ref().unwrap().read(self.offset), source);
            self.offset += 4;
            header_words = 1;
            match decode_header(word)? {
                Header::Control(operation) => {
                    if operation == MaxwellPushbufferControl::EndSegment {
                        self.offset = self.end;
                    }
                    return Ok((PacketView::Control { operation, source }, 1));
                }
                Header::Methods(header)
                    if header.mode == MaxwellMethodPacketMode::Immediate || header.count == 0 =>
                {
                    self.immediate = Some(MaxwellDecodedMethodPacket {
                        mode: header.mode,
                        header: source,
                        subchannel: header.subchannel,
                        methods: if header.count == 0 {
                            PacketMethods::Empty
                        } else {
                            PacketMethods::One(MaxwellDecodedMethod {
                                method: method_id(header.first),
                                argument: header.immediate,
                                argument_source: source,
                            })
                        },
                    });
                    let packet = self.immediate.as_ref().unwrap();
                    return Ok((
                        PacketView::Methods {
                            header: source,
                            subchannel: packet.subchannel,
                            methods: packet.methods(),
                            last: true,
                        },
                        1,
                    ));
                }
                Header::Methods(header) => {
                    self.packet = Some(header);
                    self.consumed = 0;
                }
            }
        }
        let header = self.packet.unwrap();
        // A header at an entry's end does not authorize reading the next entry.
        // Return an empty fragment so the owner can consume pending effects first.
        if self.offset == self.end || budget == header_words {
            self.immediate = Some(MaxwellDecodedMethodPacket {
                mode: header.mode,
                header: header.source,
                subchannel: header.subchannel,
                methods: PacketMethods::Empty,
            });
            return Ok((
                PacketView::Methods {
                    header: header.source,
                    subchannel: header.subchannel,
                    methods: self.immediate.as_ref().unwrap().methods(),
                    last: false,
                },
                header_words,
            ));
        }
        let (source, available) = location(self);
        let count = usize::from(header.count - self.consumed)
            .min(available)
            .min((self.end - self.offset) / 4)
            .min(budget - header_words);
        let offset = self.offset;
        self.offset += count * 4;
        let first = method_dword(header.mode, header.first, self.consumed);
        let mode = if header.mode == MaxwellMethodPacketMode::IncrementOnce && self.consumed > 0 {
            MaxwellMethodPacketMode::NonIncrementing
        } else {
            header.mode
        };
        self.consumed += count as u16;
        let last = self.consumed == header.count;
        if last {
            self.packet = None;
        }
        Ok((
            PacketView::Methods {
                header: header.source,
                subchannel: header.subchannel,
                methods: MaxwellMethodSpan::live(
                    self.words.as_ref().unwrap(),
                    offset,
                    count,
                    first,
                    mode,
                    source,
                ),
                last,
            },
            count + header_words,
        ))
    }
    pub(crate) fn finish(
        &mut self,
        scratch: &mut MaxwellCommandSourceScratch,
    ) -> Result<(), MaxwellPushbufferDecodeError> {
        if let Some(EntryWords::Gathered(mut bytes)) = self.words.take() {
            bytes.clear();
            scratch.bytes = bytes;
        }
        if let Some(header) = self.packet {
            return Err(MaxwellPushbufferDecodeError::TruncatedPacket {
                source: header.source,
                expected_arguments: header.count,
                available_arguments: self.consumed,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
pub(super) fn fixture_packets(
    words: Vec<MaxwellPushbufferWord>,
) -> Result<MaxwellDecodedPushbuffer, MaxwellPushbufferDecodeError> {
    let mut stream = CommandStream::default();
    let mut packets = Vec::new();
    let mut arguments = Vec::new();
    let mut mode = MaxwellMethodPacketMode::Immediate;
    for entry in words.chunk_by(|a, b| a.location.entry_index == b.location.entry_index) {
        stream.words = Some(EntryWords::Gathered(
            entry
                .iter()
                .flat_map(|word| word.value.to_le_bytes())
                .collect(),
        ));
        stream.offset = 0;
        stream.end = entry.len() * 4;
        while !stream.needs_entry() {
            if stream.packet.is_none()
                && let Header::Methods(header) = decode_header(entry[stream.offset / 4])?
            {
                mode = header.mode;
            }
            // Each fixture word keeps its supplied provenance. All continuation,
            // truncation and END_SEGMENT decisions use the production parser.
            let (packet, _) =
                stream.next_at(4096, |stream| (entry[stream.offset / 4].location, 1))?;
            match packet {
                PacketView::Control { operation, source } => {
                    packets.push(MaxwellDecodedPacket::Control { operation, source })
                }
                PacketView::Methods {
                    header,
                    subchannel,
                    methods,
                    last,
                } => {
                    arguments.extend(methods.iter());
                    if last {
                        packets.push(MaxwellDecodedPacket::Methods(MaxwellDecodedMethodPacket {
                            mode,
                            header,
                            subchannel,
                            methods: PacketMethods::Fixture(
                                std::mem::take(&mut arguments).into_boxed_slice(),
                            ),
                        }));
                    }
                }
            }
        }
    }
    stream.finish(&mut MaxwellCommandSourceScratch::default())?;
    Ok(MaxwellDecodedPushbuffer {
        packets: packets.into_boxed_slice(),
    })
}
