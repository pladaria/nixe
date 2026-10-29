//! Consume one accepted entry with a compact DMA cursor.
use super::*;
use crate::MaxwellResolvedRange;
#[derive(Default)]
pub struct MaxwellCommandSourceScratch {
    bytes: Vec<u8>,
}
impl MaxwellCommandSourceScratch {
    pub fn recycle(&mut self, completed: Self) {
        if completed.bytes.capacity() > self.bytes.capacity() {
            *self = completed;
        }
    }
}
pub(crate) enum EntryWords {
    Ram {
        address: nixe_memory::HostAddress,
        _pin: nixe_memory::AllocationRange,
    },
    Gathered(Vec<u8>),
}
#[derive(Default)]
pub(crate) struct CommandStream {
    next_entry: usize,
    entry: usize,
    origin: Option<MaxwellGpfifoSourceLocation>,
    offset: usize,
    end: usize,
    direct: Option<crate::MaxwellResolvedMapping>,
    source: Option<MaxwellResolvedRange>,
    words: Option<EntryWords>,
    address: Option<nixe_memory::HostAddress>,
    packet: Option<MethodHeader>,
    consumed: u16,
    current: Option<MethodHeader>,
    last: bool,
    header_offset: usize,
    header_origin: Option<MaxwellGpfifoSourceLocation>,
    #[cfg(test)]
    locations: Vec<MaxwellGpfifoSourceLocation>,
}
impl CommandStream {
    fn set_words(&mut self, words: EntryWords) {
        self.address = Some(match &words {
            EntryWords::Ram { address, .. } => *address,
            EntryWords::Gathered(bytes) => {
                nixe_memory::HostAddress::from_pointer(bytes.as_ptr().cast_mut())
            }
        });
        self.words = Some(words);
    }
    fn retain_header_location(&mut self) {
        if self.packet.is_some() && self.header_origin.is_none() {
            self.header_origin = Some(self.location_at(self.header_offset));
        }
    }
    #[cfg(test)]
    pub(crate) fn fixture(packet: PacketView<'_>) -> Self {
        let (origin, words) = match packet {
            PacketView::Control { operation, source } => {
                let word = match operation {
                    MaxwellPushbufferControl::Nop => 0,
                    MaxwellPushbufferControl::EndSegment => 7 << 29,
                    MaxwellPushbufferControl::SetSubdeviceMask(mask) => {
                        (1 << 16) | u32::from(mask) << 4
                    }
                    MaxwellPushbufferControl::StoreSubdeviceMask(mask) => {
                        (2 << 16) | u32::from(mask) << 4
                    }
                    MaxwellPushbufferControl::UseSubdeviceMask => 3 << 16,
                };
                (source, vec![word])
            }
            PacketView::Methods {
                header,
                subchannel,
                methods,
                last,
            } => {
                let first = methods.first();
                let mode = if methods.is_empty() {
                    MaxwellMethodPacketMode::Incrementing
                } else {
                    methods.mode()
                };
                let opcode = match mode {
                    MaxwellMethodPacketMode::Incrementing => 1,
                    MaxwellMethodPacketMode::NonIncrementing => 3,
                    MaxwellMethodPacketMode::Immediate => 4,
                    MaxwellMethodPacketMode::IncrementOnce => 5,
                };
                let immediate = mode == MaxwellMethodPacketMode::Immediate;
                let count = if immediate {
                    first.unwrap().argument()
                } else {
                    (methods.len() + usize::from(!last)) as u32
                };
                let method = first.map_or(0, |m| m.method().0 / 4);
                let word = (opcode << 29)
                    | (count << 16)
                    | (u32::from(subchannel.map_or(0, |s| s.get())) << 13)
                    | method;
                let mut words = vec![word];
                if !immediate {
                    words.extend(methods.arguments());
                }
                let origin = first.map_or(header, |first| {
                    let mut source = first.argument_source();
                    if !immediate {
                        source.word_offset -= 1;
                    }
                    source
                });
                (origin, words)
            }
        };
        Self::fixture_words(origin, &words)
    }
    #[cfg(test)]
    pub(crate) fn fixture_words(origin: MaxwellGpfifoSourceLocation, words: &[u32]) -> Self {
        let bytes: Vec<_> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
        let mut stream = Self {
            origin: Some(origin),
            end: bytes.len(),
            ..Self::default()
        };
        stream.set_words(EntryWords::Gathered(bytes));
        stream
    }
    #[cfg(test)]
    pub(crate) fn rewind(&mut self) {
        self.offset = 0;
        self.packet = None;
        self.consumed = 0;
    }
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
        self.retain_header_location();
        if let Some(EntryWords::Gathered(bytes)) = self.words.take() {
            scratch.bytes = bytes;
        }
        self.direct = None;
        self.source = None;
        while let Some(entry) = submission.decoded().entries().get(self.next_entry).copied() {
            self.entry = self.next_entry;
            self.next_entry += 1;
            self.offset = 0;
            self.end = entry.byte_count() as usize;
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
                self.origin = Some(MaxwellGpfifoSourceLocation {
                    channel: submission.channel(),
                    frontend: submission.frontend(),
                    entry_index: self.entry as u32,
                    pushbuffer: entry.address(),
                    word_offset: 0,
                    mapping: segment.mapping().id(),
                    generation: segment.mapping().generation(),
                });
                self.direct = Some(segment);
                self.set_words(EntryWords::Ram { address, _pin: pin });
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
            self.set_words(
                if let Some((address, pin)) = memory
                    .command_ram(&source)
                    .map_err(|error| MaxwellGpfifoSourceError::Access { location, error })?
                {
                    EntryWords::Ram { address, _pin: pin }
                } else {
                    scratch
                        .bytes
                        .try_reserve(self.end.saturating_sub(scratch.bytes.len()))
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
            self.origin = Some(location);
            return Ok(true);
        }
        Ok(false)
    }

    pub(super) fn location_at(&self, offset: usize) -> MaxwellGpfifoSourceLocation {
        #[cfg(test)]
        if let Some(source) = self.locations.get(offset / 4) {
            return *source;
        }
        let mut origin = self.origin.unwrap();
        origin.word_offset += offset as u64 / 4;
        if let Some(range) = &self.source {
            let address = origin.pushbuffer.get() + offset as u64;
            let segments = range.segments();
            let index = segments.partition_point(|s| s.gpu_offset().get() + s.size() <= address);
            origin.mapping = segments[index].mapping().id();
            origin.generation = segments[index].mapping().generation();
        }
        origin
    }
    pub(crate) fn header_location(&self) -> MaxwellGpfifoSourceLocation {
        self.header_origin
            .unwrap_or_else(|| self.location_at(self.header_offset))
    }
    pub(crate) fn last(&self) -> bool {
        self.last
    }
    pub(crate) fn subchannel(&self) -> Option<MaxwellPushbufferSubchannel> {
        self.current.unwrap().subchannel
    }
    pub(crate) fn methods(&self) -> MaxwellMethodSpan<'_> {
        let current = self.current.as_ref().unwrap();
        let address = if current.mode == MaxwellMethodPacketMode::Immediate {
            nixe_memory::HostAddress::from_pointer(
                (&current.immediate as *const u32).cast_mut().cast(),
            )
        } else {
            nixe_memory::HostAddress::from_pointer(unsafe {
                self.address.unwrap().as_ptr().add(current.offset)
            })
        };
        MaxwellMethodSpan::live(
            self,
            address,
            current.offset,
            current.count as usize,
            current.first,
            current.mode,
        )
    }
    pub(super) fn gathered_bytes(&self, begin: usize, end: usize) -> Option<&[u8]> {
        if self.current.unwrap().mode == MaxwellMethodPacketMode::Immediate {
            return None;
        }
        match self.words.as_ref().unwrap() {
            EntryWords::Gathered(bytes) => Some(&bytes[begin..end]),
            _ => None,
        }
    }
    /// Advance hardware packet state in place; ordinary packets return no object.
    pub(crate) fn next(
        &mut self,
    ) -> Result<Option<MaxwellPushbufferControl>, MaxwellPushbufferDecodeError> {
        if self.packet.is_none() {
            self.header_offset = self.offset;
            self.header_origin = None;
            // SAFETY: the entry and its complete word count were accepted before
            // entering this loop. No reference to mutable guest RAM is formed.
            let word = unsafe { self.address.unwrap().read_u32(self.offset) };
            let header = decode_header(word, self.offset, || self.location_at(self.offset))?;
            self.offset += 4;
            match header {
                Header::Control(operation) => {
                    if operation == MaxwellPushbufferControl::EndSegment {
                        self.offset = self.end;
                    }
                    self.current = None;
                    return Ok(Some(operation));
                }
                Header::Methods(header)
                    if header.mode == MaxwellMethodPacketMode::Immediate || header.count == 0 =>
                {
                    self.current = Some(header);
                    self.last = true;
                    return Ok(None);
                }
                Header::Methods(header) => {
                    self.packet = Some(header);
                    self.consumed = 0;
                }
            }
        }
        let header = self.packet.unwrap();
        let count = usize::from(header.count - self.consumed).min((self.end - self.offset) / 4);
        self.current = Some(MethodHeader {
            offset: self.offset,
            first: method_dword(header.mode, header.first, self.consumed),
            count: count as u16,
            mode: if header.mode == MaxwellMethodPacketMode::IncrementOnce && self.consumed > 0 {
                MaxwellMethodPacketMode::NonIncrementing
            } else {
                header.mode
            },
            ..header
        });
        self.offset += count * 4;
        self.consumed += count as u16;
        self.last = self.consumed == header.count;
        if self.last {
            self.packet = None;
        }
        Ok(None)
    }
    pub(crate) fn finish(
        &mut self,
        scratch: &mut MaxwellCommandSourceScratch,
    ) -> Result<(), MaxwellPushbufferDecodeError> {
        if let Some(header) = self.packet {
            return Err(MaxwellPushbufferDecodeError::TruncatedPacket {
                source: self.header_location(),
                expected_arguments: header.count,
                available_arguments: self.consumed,
            });
        }
        if let Some(EntryWords::Gathered(bytes)) = self.words.take() {
            scratch.bytes = bytes;
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
        stream.retain_header_location();
        stream.set_words(EntryWords::Gathered(
            entry
                .iter()
                .flat_map(|word| word.value.to_le_bytes())
                .collect(),
        ));
        stream.origin = Some(entry[0].location);
        stream.locations = entry.iter().map(|w| w.location).collect();
        stream.offset = 0;
        stream.end = entry.len() * 4;
        while !stream.needs_entry() {
            let new_header = stream.packet.is_none();
            if let Some(operation) = stream.next()? {
                packets.push(MaxwellDecodedPacket::Control {
                    operation,
                    source: stream.header_location(),
                });
                continue;
            }
            if new_header {
                mode = stream.current.unwrap().mode;
            }
            arguments.extend(stream.methods().iter());
            if stream.last {
                packets.push(MaxwellDecodedPacket::Methods(MaxwellDecodedMethodPacket {
                    mode,
                    header: stream.header_location(),
                    subchannel: stream.subchannel(),
                    methods: std::mem::take(&mut arguments).into_boxed_slice(),
                }));
            }
        }
    }
    stream.finish(&mut MaxwellCommandSourceScratch::default())?;
    Ok(MaxwellDecodedPushbuffer {
        packets: packets.into_boxed_slice(),
    })
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_packet_crosses_mappings_without_fragmenting_or_losing_error_locations() {
        use crate::{
            MaxwellAddressSpaceId, MaxwellAddressSpaceInitialization, MaxwellAllocationId,
            MaxwellGpuAddressSpace, MaxwellMapRequest, SWITCH_1_GM20B_PROFILE,
        };
        use nixe_memory::MemoryPermissions;
        let machine = crate::test_memory::machine();
        let mut space = MaxwellGpuAddressSpace::new(
            machine.clone(),
            MaxwellAddressSpaceId::new(1),
            SWITCH_1_GM20B_PROFILE,
        );
        space
            .initialize(MaxwellAddressSpaceInitialization::default())
            .unwrap();
        let mut map = || {
            let allocation = crate::test_memory::allocate(&machine, 0x1000);
            space
                .map(MaxwellMapRequest {
                    allocation: MaxwellAllocationId::new(1),
                    backing: crate::test_memory::ranges(&allocation),
                    backing_offset: 0,
                    size: 0x1000,
                    allocation_alignment: 0x1000,
                    page_size: 0,
                    kind: 0,
                    cacheable: false,
                    permissions: MemoryPermissions::READ_WRITE,
                    fixed_offset: None,
                })
                .unwrap()
        };
        let left = map();
        let right = map();
        assert_eq!(right.offset().get(), left.offset().get() + 0x1000);
        let memory = space.memory();
        let start = memory.checked_add(left.offset(), 0xff8).unwrap();
        let range = memory
            .resolve_range(start, 16, MemoryPermissions::READ)
            .unwrap();
        let origin = MaxwellGpfifoSourceLocation {
            channel: crate::MaxwellChannelId::new(1),
            frontend: nixe_gpu::FrontendSubmissionId::new(1),
            entry_index: 0,
            pushbuffer: start,
            word_offset: 0,
            mapping: left.id(),
            generation: left.generation(),
        };
        let mut stream = CommandStream::fixture_words(
            origin,
            &[(1 << 29) | (3 << 16) | (0x0a00 / 4), 11, 22, 33],
        );
        stream.source = Some(range);
        assert_eq!(stream.next().unwrap(), None);
        assert!(stream.last());
        let methods = stream.methods();
        assert_eq!(methods.len(), 3);
        assert_eq!(methods.argument_at(2), 33);
        assert_eq!(methods.get(0).unwrap().argument_source().mapping, left.id());
        assert_eq!(
            methods.get(1).unwrap().argument_source().mapping,
            right.id()
        );
        assert_eq!(methods.get(2).unwrap().argument_source().word_offset, 3);
        assert!(stream.needs_entry());
        stream
            .finish(&mut MaxwellCommandSourceScratch::default())
            .unwrap();
    }
}
