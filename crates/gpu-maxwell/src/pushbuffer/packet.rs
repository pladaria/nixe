//! Side-effect-free Maxwell pushbuffer packet decoding.
//!
//! Packet syntax is decoded before any channel, subchannel, class, or backend
//! state can be observed or changed. The bit layout and command semantics are
//! defined by NVIDIA's public FIFO DMA reference:
//! https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/manuals/volta/gv100/dev_ram.ref.txt#L893-L1227
//! The same GF100+ encodings, including increment-once, are independently
//! described by the pinned envytools DMA-pusher reference:
//! https://github.com/envytools/envytools/blob/f102b82381f3f11cee113d16374c87091db039d9/docs/hw/fifo/dma-pusher.rst#gf100-commands

use std::fmt::{Display, Formatter};

use nixe_gpu::GpuMethodId;

use crate::{
    MaxwellGpfifoSourceError, MaxwellGpfifoSourceLocation, MaxwellGpuAddressSpace,
    MaxwellValidatedGpfifoSubmission,
};

const METHOD_ADDRESS_MASK: u32 = 0x0fff;
const METHOD_RESERVED_BIT: u32 = 1 << 12;
const SUBCHANNEL_SHIFT: u32 = 13;
const SUBCHANNEL_MASK: u32 = 0x7;
const COUNT_SHIFT: u32 = 16;
const COUNT_MASK: u32 = 0x1fff;
const SECONDARY_OPCODE_SHIFT: u32 = 29;
const SECONDARY_OPCODE_MASK: u32 = 0x7;
const MAX_METHOD_DWORD: u32 = METHOD_ADDRESS_MASK;

const OPCODE_GROUP_ZERO: u8 = 0;
const OPCODE_INCREMENTING: u8 = 1;
const OPCODE_NON_INCREMENTING: u8 = 3;
const OPCODE_IMMEDIATE: u8 = 4;
const OPCODE_INCREMENT_ONCE: u8 = 5;
const OPCODE_END_SEGMENT: u8 = 7;

const CONTROL_SET_SUBDEVICE_MASK: u16 = 1;
const CONTROL_STORE_SUBDEVICE_MASK: u16 = 2;
const CONTROL_USE_SUBDEVICE_MASK: u16 = 3;

#[derive(Clone, Copy)]
struct MethodEncoding {
    opcode: u8,
    mode: MaxwellMethodPacketMode,
}

const METHOD_ENCODINGS: [MethodEncoding; 4] = [
    MethodEncoding {
        opcode: OPCODE_INCREMENTING,
        mode: MaxwellMethodPacketMode::Incrementing,
    },
    MethodEncoding {
        opcode: OPCODE_NON_INCREMENTING,
        mode: MaxwellMethodPacketMode::NonIncrementing,
    },
    MethodEncoding {
        opcode: OPCODE_IMMEDIATE,
        mode: MaxwellMethodPacketMode::Immediate,
    },
    MethodEncoding {
        opcode: OPCODE_INCREMENT_ONCE,
        mode: MaxwellMethodPacketMode::IncrementOnce,
    },
];

/// One command word paired with its exact retained T6 source.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellPushbufferWord {
    value: u32,
    location: MaxwellGpfifoSourceLocation,
}

impl MaxwellPushbufferWord {
    #[must_use]
    pub const fn new(value: u32, location: MaxwellGpfifoSourceLocation) -> Self {
        Self { value, location }
    }

    #[must_use]
    pub const fn value(self) -> u32 {
        self.value
    }

    #[must_use]
    pub const fn location(self) -> MaxwellGpfifoSourceLocation {
        self.location
    }
}

/// Hardware subchannel selected by a method header.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct MaxwellPushbufferSubchannel(u8);

impl MaxwellPushbufferSubchannel {
    #[must_use]
    pub const fn try_new(value: u8) -> Option<Self> {
        if value <= SUBCHANNEL_MASK as u8 {
            Some(Self(value))
        } else {
            None
        }
    }

    #[must_use]
    pub const fn get(self) -> u8 {
        self.0
    }
}

impl Display for MaxwellPushbufferSubchannel {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "subchannel={}", self.0)
    }
}

/// Method-address evolution selected by one packet header.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellMethodPacketMode {
    Incrementing,
    NonIncrementing,
    Immediate,
    IncrementOnce,
}

impl MaxwellMethodPacketMode {
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Incrementing => "incrementing",
            Self::NonIncrementing => "non-incrementing",
            Self::Immediate => "immediate",
            Self::IncrementOnce => "increment-once",
        }
    }
}

impl Display for MaxwellMethodPacketMode {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.label())
    }
}

/// One expanded method write. Its address is the byte-address used by class
/// method tables, not the dword-address stored in the packet header.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellDecodedMethod {
    method: GpuMethodId,
    argument: u32,
    argument_source: MaxwellGpfifoSourceLocation,
}

impl MaxwellDecodedMethod {
    #[must_use]
    pub const fn method(self) -> GpuMethodId {
        self.method
    }

    #[must_use]
    pub const fn argument(self) -> u32 {
        self.argument
    }

    #[must_use]
    pub const fn argument_source(self) -> MaxwellGpfifoSourceLocation {
        self.argument_source
    }
}

/// A complete, syntactically validated compressed method sequence.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaxwellDecodedMethodPacket {
    mode: MaxwellMethodPacketMode,
    header: MaxwellGpfifoSourceLocation,
    subchannel: Option<MaxwellPushbufferSubchannel>,
    methods: Box<[MaxwellDecodedMethod]>,
}

impl MaxwellDecodedMethodPacket {
    #[must_use]
    pub const fn mode(&self) -> MaxwellMethodPacketMode {
        self.mode
    }

    #[must_use]
    pub const fn header(&self) -> MaxwellGpfifoSourceLocation {
        self.header
    }

    /// Zero-count packets ignore all remaining header fields in hardware and
    /// therefore deliberately expose no subchannel.
    #[must_use]
    pub const fn subchannel(&self) -> Option<MaxwellPushbufferSubchannel> {
        self.subchannel
    }

    #[must_use]
    pub fn methods(&self) -> &[MaxwellDecodedMethod] {
        &self.methods
    }
}

/// Syntactically decoded PBDMA control instruction.
///
/// Decoding does not execute subdevice-mask behavior. That semantic boundary
/// remains explicit for later frontend dispatch work.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellPushbufferControl {
    Nop,
    SetSubdeviceMask(u16),
    StoreSubdeviceMask(u16),
    UseSubdeviceMask,
    EndSegment,
}

/// One complete packet or control instruction.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MaxwellDecodedPacket {
    Methods(MaxwellDecodedMethodPacket),
    Control {
        operation: MaxwellPushbufferControl,
        source: MaxwellGpfifoSourceLocation,
    },
}

impl MaxwellDecodedPacket {
    #[must_use]
    pub const fn source(&self) -> MaxwellGpfifoSourceLocation {
        match self {
            Self::Methods(packet) => packet.header(),
            Self::Control { source, .. } => *source,
        }
    }
}

/// Fully decoded command stream. Constructing this value has no frontend side
/// effects; callers cannot observe a valid prefix when a later packet fails.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaxwellDecodedPushbuffer {
    packets: Box<[MaxwellDecodedPacket]>,
}

impl MaxwellDecodedPushbuffer {
    #[must_use]
    pub fn packets(&self) -> &[MaxwellDecodedPacket] {
        &self.packets
    }
}

/// Failure while reading or validating packet syntax.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MaxwellPushbufferDecodeError {
    Source(MaxwellGpfifoSourceError),
    UnknownEncoding {
        source: MaxwellGpfifoSourceLocation,
        word: u32,
        secondary_opcode: u8,
    },
    InvalidControlEncoding {
        source: MaxwellGpfifoSourceLocation,
        word: u32,
    },
    ReservedMethodBit {
        source: MaxwellGpfifoSourceLocation,
        word: u32,
    },
    MethodRangeOverflow {
        source: MaxwellGpfifoSourceLocation,
        mode: MaxwellMethodPacketMode,
        first_method: GpuMethodId,
        count: u16,
    },
    TruncatedPacket {
        source: MaxwellGpfifoSourceLocation,
        expected_arguments: u16,
        available_arguments: u16,
    },
    ResourceExhausted,
}

impl Display for MaxwellPushbufferDecodeError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Source(error) => write!(formatter, "pushbuffer source failed: {error}"),
            Self::UnknownEncoding {
                source,
                word,
                secondary_opcode,
            } => write!(
                formatter,
                "unknown Maxwell packet encoding: {source} word={word:#010x} secondary-opcode={secondary_opcode}"
            ),
            Self::InvalidControlEncoding { source, word } => write!(
                formatter,
                "invalid Maxwell control encoding: {source} word={word:#010x}"
            ),
            Self::ReservedMethodBit { source, word } => write!(
                formatter,
                "Maxwell method header sets reserved bit 12: {source} word={word:#010x}"
            ),
            Self::MethodRangeOverflow {
                source,
                mode,
                first_method,
                count,
            } => write!(
                formatter,
                "Maxwell method sequence exceeds the method range: {source} mode={mode} {first_method} count={count}"
            ),
            Self::TruncatedPacket {
                source,
                expected_arguments,
                available_arguments,
            } => write!(
                formatter,
                "truncated Maxwell method packet: {source} expected-arguments={expected_arguments} available-arguments={available_arguments}"
            ),
            Self::ResourceExhausted => {
                formatter.write_str("Maxwell packet decoding exhausted host resources")
            }
        }
    }
}

impl std::error::Error for MaxwellPushbufferDecodeError {}

impl From<MaxwellGpfifoSourceError> for MaxwellPushbufferDecodeError {
    fn from(error: MaxwellGpfifoSourceError) -> Self {
        Self::Source(error)
    }
}

/// Decodes a complete stream atomically from source-tagged words.
///
/// Method data may cross GPFIFO pushbuffer-segment boundaries, as permitted by
/// the NVIDIA reference. `EndSegment` discards only the remaining words of its
/// own GPFIFO entry.
pub fn decode_maxwell_pushbuffer<I>(
    words: I,
) -> Result<MaxwellDecodedPushbuffer, MaxwellPushbufferDecodeError>
where
    I: IntoIterator<Item = Result<MaxwellPushbufferWord, MaxwellGpfifoSourceError>>,
{
    let mut words = words.into_iter().peekable();
    let mut packets = Vec::new();

    while let Some(header) = next_word(&mut words)? {
        let packet = decode_packet(header, &mut words)?;
        let end_entry = matches!(
            packet,
            MaxwellDecodedPacket::Control {
                operation: MaxwellPushbufferControl::EndSegment,
                ..
            }
        )
        .then_some(header.location.entry_index);
        packets
            .try_reserve(1)
            .map_err(|_| MaxwellPushbufferDecodeError::ResourceExhausted)?;
        packets.push(packet);

        if let Some(end_entry) = end_entry {
            skip_entry_tail(&mut words, end_entry)?;
        }
    }

    Ok(MaxwellDecodedPushbuffer {
        packets: packets.into_boxed_slice(),
    })
}

/// Reads the retained T6 sources and decodes their logical command stream.
/// No channel, scheduler, fence, class, or backend state is mutated.
pub fn decode_maxwell_submission(
    submission: &MaxwellValidatedGpfifoSubmission,
    address_space: &MaxwellGpuAddressSpace,
) -> Result<MaxwellDecodedPushbuffer, MaxwellPushbufferDecodeError> {
    submission.validate_sources(address_space)?;
    decode_maxwell_pushbuffer(SubmissionWords::new(submission, address_space))
}

fn decode_packet<I>(
    header: MaxwellPushbufferWord,
    words: &mut std::iter::Peekable<I>,
) -> Result<MaxwellDecodedPacket, MaxwellPushbufferDecodeError>
where
    I: Iterator<Item = Result<MaxwellPushbufferWord, MaxwellGpfifoSourceError>>,
{
    let word = header.value;
    if word == 0 {
        return Ok(control_packet(MaxwellPushbufferControl::Nop, header));
    }

    let opcode = ((word >> SECONDARY_OPCODE_SHIFT) & SECONDARY_OPCODE_MASK) as u8;
    if opcode == OPCODE_GROUP_ZERO {
        return decode_group_zero_control(header);
    }
    if opcode == OPCODE_END_SEGMENT {
        return Ok(control_packet(MaxwellPushbufferControl::EndSegment, header));
    }
    let Some(encoding) = METHOD_ENCODINGS
        .iter()
        .copied()
        .find(|encoding| encoding.opcode == opcode)
    else {
        return Err(MaxwellPushbufferDecodeError::UnknownEncoding {
            source: header.location,
            word,
            secondary_opcode: opcode,
        });
    };

    decode_method_packet(header, encoding.mode, words)
}

fn decode_group_zero_control(
    word: MaxwellPushbufferWord,
) -> Result<MaxwellDecodedPacket, MaxwellPushbufferDecodeError> {
    let opcode = (word.value >> 16) as u16;
    let low = word.value as u16;
    let operation = match opcode {
        CONTROL_SET_SUBDEVICE_MASK if low & 0xf == 0 => {
            MaxwellPushbufferControl::SetSubdeviceMask(low >> 4)
        }
        CONTROL_STORE_SUBDEVICE_MASK if low & 0xf == 0 => {
            MaxwellPushbufferControl::StoreSubdeviceMask(low >> 4)
        }
        CONTROL_USE_SUBDEVICE_MASK if low == 0 => MaxwellPushbufferControl::UseSubdeviceMask,
        _ => {
            return Err(MaxwellPushbufferDecodeError::InvalidControlEncoding {
                source: word.location,
                word: word.value,
            });
        }
    };
    Ok(control_packet(operation, word))
}

fn decode_method_packet<I>(
    header: MaxwellPushbufferWord,
    mode: MaxwellMethodPacketMode,
    words: &mut std::iter::Peekable<I>,
) -> Result<MaxwellDecodedPacket, MaxwellPushbufferDecodeError>
where
    I: Iterator<Item = Result<MaxwellPushbufferWord, MaxwellGpfifoSourceError>>,
{
    let encoded_count = ((header.value >> COUNT_SHIFT) & COUNT_MASK) as u16;
    let count = if mode == MaxwellMethodPacketMode::Immediate {
        1
    } else {
        encoded_count
    };
    if count == 0 {
        return Ok(MaxwellDecodedPacket::Methods(MaxwellDecodedMethodPacket {
            mode,
            header: header.location,
            subchannel: None,
            methods: Box::new([]),
        }));
    }
    if header.value & METHOD_RESERVED_BIT != 0 {
        return Err(MaxwellPushbufferDecodeError::ReservedMethodBit {
            source: header.location,
            word: header.value,
        });
    }

    let first_method_dword = header.value & METHOD_ADDRESS_MASK;
    validate_method_range(header.location, mode, first_method_dword, count)?;
    let subchannel =
        MaxwellPushbufferSubchannel(((header.value >> SUBCHANNEL_SHIFT) & SUBCHANNEL_MASK) as u8);
    let mut methods = Vec::new();
    methods
        .try_reserve_exact(usize::from(count))
        .map_err(|_| MaxwellPushbufferDecodeError::ResourceExhausted)?;

    if mode == MaxwellMethodPacketMode::Immediate {
        methods.push(MaxwellDecodedMethod {
            method: method_id(first_method_dword),
            argument: (header.value >> COUNT_SHIFT) & COUNT_MASK,
            argument_source: header.location,
        });
    } else {
        for argument_index in 0..count {
            let Some(argument) = next_word(words)? else {
                return Err(MaxwellPushbufferDecodeError::TruncatedPacket {
                    source: header.location,
                    expected_arguments: count,
                    available_arguments: argument_index,
                });
            };
            methods.push(MaxwellDecodedMethod {
                method: method_id(method_dword(mode, first_method_dword, argument_index)),
                argument: argument.value,
                argument_source: argument.location,
            });
        }
    }

    Ok(MaxwellDecodedPacket::Methods(MaxwellDecodedMethodPacket {
        mode,
        header: header.location,
        subchannel: Some(subchannel),
        methods: methods.into_boxed_slice(),
    }))
}

fn validate_method_range(
    source: MaxwellGpfifoSourceLocation,
    mode: MaxwellMethodPacketMode,
    first_method_dword: u32,
    count: u16,
) -> Result<(), MaxwellPushbufferDecodeError> {
    let last = match mode {
        MaxwellMethodPacketMode::Incrementing => first_method_dword
            .checked_add(u32::from(count) - 1)
            .filter(|last| *last <= MAX_METHOD_DWORD),
        MaxwellMethodPacketMode::IncrementOnce if count > 1 => first_method_dword
            .checked_add(1)
            .filter(|last| *last <= MAX_METHOD_DWORD),
        MaxwellMethodPacketMode::NonIncrementing
        | MaxwellMethodPacketMode::Immediate
        | MaxwellMethodPacketMode::IncrementOnce => Some(first_method_dword),
    };
    if last.is_none() {
        return Err(MaxwellPushbufferDecodeError::MethodRangeOverflow {
            source,
            mode,
            first_method: method_id(first_method_dword),
            count,
        });
    }
    Ok(())
}

const fn method_dword(mode: MaxwellMethodPacketMode, first: u32, index: u16) -> u32 {
    match mode {
        MaxwellMethodPacketMode::Incrementing => first + index as u32,
        MaxwellMethodPacketMode::IncrementOnce if index != 0 => first + 1,
        MaxwellMethodPacketMode::NonIncrementing
        | MaxwellMethodPacketMode::IncrementOnce
        | MaxwellMethodPacketMode::Immediate => first,
    }
}

const fn method_id(dword: u32) -> GpuMethodId {
    GpuMethodId(dword * 4)
}

const fn control_packet(
    operation: MaxwellPushbufferControl,
    word: MaxwellPushbufferWord,
) -> MaxwellDecodedPacket {
    MaxwellDecodedPacket::Control {
        operation,
        source: word.location,
    }
}

fn next_word<I>(
    words: &mut std::iter::Peekable<I>,
) -> Result<Option<MaxwellPushbufferWord>, MaxwellPushbufferDecodeError>
where
    I: Iterator<Item = Result<MaxwellPushbufferWord, MaxwellGpfifoSourceError>>,
{
    words.next().transpose().map_err(Into::into)
}

fn skip_entry_tail<I>(
    words: &mut std::iter::Peekable<I>,
    entry_index: u32,
) -> Result<(), MaxwellPushbufferDecodeError>
where
    I: Iterator<Item = Result<MaxwellPushbufferWord, MaxwellGpfifoSourceError>>,
{
    loop {
        match words.peek() {
            Some(Ok(word)) if word.location.entry_index == entry_index => {
                words.next();
            }
            Some(Err(_)) => {
                next_word(words)?;
            }
            _ => return Ok(()),
        }
    }
}

pub(crate) struct SubmissionWords<'a> {
    submission: &'a MaxwellValidatedGpfifoSubmission,
    address_space: &'a MaxwellGpuAddressSpace,
    next_entry: usize,
    current_entry: u32,
    current_bytes: Vec<u8>,
    current_offset: usize,
    failed: bool,
}

impl<'a> SubmissionWords<'a> {
    pub(crate) const fn new(
        submission: &'a MaxwellValidatedGpfifoSubmission,
        address_space: &'a MaxwellGpuAddressSpace,
    ) -> Self {
        Self {
            submission,
            address_space,
            next_entry: 0,
            current_entry: 0,
            current_bytes: Vec::new(),
            current_offset: 0,
            failed: false,
        }
    }

    fn load_next_entry(&mut self) -> Result<bool, MaxwellGpfifoSourceError> {
        let Some(pushbuffer) = self.submission.pushbuffers().get(self.next_entry) else {
            return Ok(false);
        };
        let byte_count = usize::try_from(pushbuffer.entry().byte_count())
            .map_err(|_| MaxwellGpfifoSourceError::ArithmeticOverflow)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(byte_count)
            .map_err(|_| MaxwellGpfifoSourceError::ResourceExhausted)?;
        bytes.resize(byte_count, 0);
        self.submission.read_pushbuffer(
            self.address_space,
            pushbuffer.entry_index(),
            &mut bytes,
        )?;
        self.current_entry = pushbuffer.entry_index();
        self.current_bytes = bytes;
        self.current_offset = 0;
        self.next_entry += 1;
        Ok(true)
    }
}

impl Iterator for SubmissionWords<'_> {
    type Item = Result<MaxwellPushbufferWord, MaxwellGpfifoSourceError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.failed {
            return None;
        }
        while self.current_offset == self.current_bytes.len() {
            match self.load_next_entry() {
                Ok(true) => {}
                Ok(false) => return None,
                Err(error) => {
                    self.failed = true;
                    return Some(Err(error));
                }
            }
        }

        let word_offset = self.current_offset / 4;
        let value = u32::from_le_bytes(
            self.current_bytes[self.current_offset..self.current_offset + 4]
                .try_into()
                .expect("GPFIFO word counts always produce complete dwords"),
        );
        self.current_offset += 4;
        let word_offset = match u64::try_from(word_offset) {
            Ok(offset) => offset,
            Err(_) => {
                self.failed = true;
                return Some(Err(MaxwellGpfifoSourceError::ArithmeticOverflow));
            }
        };
        match self
            .submission
            .word_location(self.current_entry, word_offset)
        {
            Ok(location) => Some(Ok(MaxwellPushbufferWord::new(value, location))),
            Err(error) => {
                self.failed = true;
                Some(Err(error))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use nixe_gpu::{FrontendSubmissionId, GpuVirtualAddress, MappingGeneration};
    use nixe_memory::{CanonicalAllocation, MemoryPermissions};

    use super::*;
    use crate::{
        MaxwellAddressSpaceId, MaxwellAddressSpaceInitialization, MaxwellAllocationId,
        MaxwellChannelId, MaxwellChannelOwner, MaxwellGpfifoSubmitRequest, MaxwellGpuChannel,
        MaxwellMapRequest, MaxwellMappingId, SWITCH_1_GM20B_PROFILE, decode_gpfifo_submission,
        resolve_gpfifo_submission,
    };

    fn location(entry: u32, word: u64) -> MaxwellGpfifoSourceLocation {
        MaxwellGpfifoSourceLocation {
            channel: MaxwellChannelId::new(3),
            frontend: FrontendSubmissionId::new(7),
            entry_index: entry,
            pushbuffer: GpuVirtualAddress::try_new(0x8000 + entry as u64 * 0x100, 40).unwrap(),
            word_offset: word,
            mapping: MaxwellMappingId::new(11 + entry as u64),
            generation: MappingGeneration::new(5),
        }
    }

    fn sourced(value: u32, entry: u32, word: u64) -> MaxwellPushbufferWord {
        MaxwellPushbufferWord::new(value, location(entry, word))
    }

    fn decode(
        words: &[MaxwellPushbufferWord],
    ) -> Result<MaxwellDecodedPushbuffer, MaxwellPushbufferDecodeError> {
        decode_maxwell_pushbuffer(words.iter().copied().map(Ok))
    }

    const fn method_header(opcode: u32, method: u32, subchannel: u32, count: u32) -> u32 {
        opcode << 29 | count << 16 | subchannel << 13 | method
    }

    #[test]
    fn decodes_incrementing_non_incrementing_and_increment_once_packets() {
        let words = [
            sourced(method_header(1, 0x100, 2, 3), 0, 0),
            sourced(10, 0, 1),
            sourced(11, 0, 2),
            sourced(12, 0, 3),
            sourced(method_header(3, 0x200, 4, 2), 0, 4),
            sourced(20, 0, 5),
            sourced(21, 0, 6),
            sourced(method_header(5, 0x300, 6, 3), 0, 7),
            sourced(30, 0, 8),
            sourced(31, 0, 9),
            sourced(32, 0, 10),
        ];
        let decoded = decode(&words).unwrap();
        assert_eq!(decoded.packets().len(), 3);

        let MaxwellDecodedPacket::Methods(incrementing) = &decoded.packets()[0] else {
            panic!("expected method packet");
        };
        assert_eq!(incrementing.mode(), MaxwellMethodPacketMode::Incrementing);
        assert_eq!(incrementing.subchannel().unwrap().get(), 2);
        assert_eq!(
            incrementing
                .methods()
                .iter()
                .map(|method| (method.method().0, method.argument()))
                .collect::<Vec<_>>(),
            [(0x400, 10), (0x404, 11), (0x408, 12)]
        );

        let MaxwellDecodedPacket::Methods(non_incrementing) = &decoded.packets()[1] else {
            panic!("expected method packet");
        };
        assert_eq!(
            non_incrementing
                .methods()
                .iter()
                .map(|method| method.method().0)
                .collect::<Vec<_>>(),
            [0x800, 0x800]
        );

        let MaxwellDecodedPacket::Methods(increment_once) = &decoded.packets()[2] else {
            panic!("expected method packet");
        };
        assert_eq!(
            increment_once
                .methods()
                .iter()
                .map(|method| method.method().0)
                .collect::<Vec<_>>(),
            [0xc00, 0xc04, 0xc04]
        );
        assert_eq!(
            increment_once.methods()[2].argument_source(),
            location(0, 10)
        );
    }

    #[test]
    fn immediate_packet_uses_inline_thirteen_bit_argument() {
        let word = method_header(4, 0x234, 7, 0x1abc);
        let decoded = decode(&[sourced(word, 0, 4)]).unwrap();
        let MaxwellDecodedPacket::Methods(packet) = &decoded.packets()[0] else {
            panic!("expected method packet");
        };
        assert_eq!(packet.mode(), MaxwellMethodPacketMode::Immediate);
        assert_eq!(packet.subchannel().unwrap().get(), 7);
        assert_eq!(packet.methods()[0].method(), GpuMethodId(0x8d0));
        assert_eq!(packet.methods()[0].argument(), 0x1abc);
        assert_eq!(packet.methods()[0].argument_source(), location(0, 4));
    }

    #[test]
    fn method_and_subchannel_endpoints_decode_without_widening_the_fields() {
        let decoded = decode(&[
            sourced(method_header(1, 0, 0, 1), 0, 0),
            sourced(1, 0, 1),
            sourced(method_header(1, 0xfff, 7, 1), 0, 2),
            sourced(2, 0, 3),
        ])
        .unwrap();
        let MaxwellDecodedPacket::Methods(first) = &decoded.packets()[0] else {
            panic!("expected first method packet");
        };
        let MaxwellDecodedPacket::Methods(last) = &decoded.packets()[1] else {
            panic!("expected last method packet");
        };
        assert_eq!(first.subchannel().unwrap().get(), 0);
        assert_eq!(first.methods()[0].method(), GpuMethodId(0));
        assert_eq!(last.subchannel().unwrap().get(), 7);
        assert_eq!(last.methods()[0].method(), GpuMethodId(0x3ffc));
    }

    #[test]
    fn zero_count_packet_ignores_reserved_and_address_fields() {
        let decoded = decode(&[sourced(method_header(1, 0x1fff, 7, 0), 0, 0)]).unwrap();
        let MaxwellDecodedPacket::Methods(packet) = &decoded.packets()[0] else {
            panic!("expected method packet");
        };
        assert_eq!(packet.subchannel(), None);
        assert!(packet.methods().is_empty());
    }

    #[test]
    fn decodes_control_packets_and_end_segment_discards_only_its_entry_tail() {
        let words = [
            sourced(0, 0, 0),
            sourced(0x0001_abc0, 0, 1),
            sourced(0x0002_1230, 0, 2),
            sourced(0x0003_0000, 0, 3),
            sourced(7 << 29, 0, 4),
            sourced(2 << 29, 0, 5),
            sourced(method_header(4, 0x100, 1, 9), 1, 0),
        ];
        let decoded = decode(&words).unwrap();
        assert_eq!(decoded.packets().len(), 6);
        assert_eq!(
            decoded.packets()[1],
            control_packet(MaxwellPushbufferControl::SetSubdeviceMask(0xabc), words[1])
        );
        assert_eq!(
            decoded.packets()[2],
            control_packet(
                MaxwellPushbufferControl::StoreSubdeviceMask(0x123),
                words[2]
            )
        );
        assert_eq!(
            decoded.packets()[3],
            control_packet(MaxwellPushbufferControl::UseSubdeviceMask, words[3])
        );
        assert!(matches!(
            decoded.packets()[5],
            MaxwellDecodedPacket::Methods(_)
        ));
    }

    #[test]
    fn method_payload_may_cross_pushbuffer_segments() {
        let words = [
            sourced(method_header(1, 0x100, 0, 2), 0, 9),
            sourced(0xaaaa, 1, 0),
            sourced(0xbbbb, 1, 1),
        ];
        let decoded = decode(&words).unwrap();
        let MaxwellDecodedPacket::Methods(packet) = &decoded.packets()[0] else {
            panic!("expected method packet");
        };
        assert_eq!(packet.methods()[0].argument_source(), location(1, 0));
        assert_eq!(packet.methods()[1].argument_source(), location(1, 1));
    }

    #[test]
    fn rejects_unknown_reserved_and_invalid_control_encodings() {
        assert!(matches!(
            decode(&[sourced(2 << 29, 0, 0)]),
            Err(MaxwellPushbufferDecodeError::UnknownEncoding {
                secondary_opcode: 2,
                ..
            })
        ));
        assert!(matches!(
            decode(&[
                sourced(method_header(1, 0x1000, 0, 1), 0, 0),
                sourced(0, 0, 1)
            ]),
            Err(MaxwellPushbufferDecodeError::ReservedMethodBit { .. })
        ));
        assert!(matches!(
            decode(&[sourced(0x0001_0001, 0, 0)]),
            Err(MaxwellPushbufferDecodeError::InvalidControlEncoding { .. })
        ));
    }

    #[test]
    fn rejects_truncation_and_incrementing_method_overflow_at_the_header() {
        let truncated = sourced(method_header(3, 0x100, 0, 3), 0, 4);
        assert_eq!(
            decode(&[truncated, sourced(1, 0, 5)]),
            Err(MaxwellPushbufferDecodeError::TruncatedPacket {
                source: location(0, 4),
                expected_arguments: 3,
                available_arguments: 1,
            })
        );

        assert!(matches!(
            decode(&[
                sourced(method_header(1, 0xfff, 0, 2), 0, 8),
                sourced(1, 0, 9),
                sourced(2, 0, 10),
            ]),
            Err(MaxwellPushbufferDecodeError::MethodRangeOverflow {
                source,
                first_method: GpuMethodId(0x3ffc),
                count: 2,
                ..
            }) if source == location(0, 8)
        ));
        assert!(matches!(
            decode(&[
                sourced(method_header(5, 0xfff, 0, 2), 0, 0),
                sourced(1, 0, 1),
                sourced(2, 0, 2),
            ]),
            Err(MaxwellPushbufferDecodeError::MethodRangeOverflow { .. })
        ));
    }

    #[test]
    fn complete_stream_decode_never_returns_a_valid_prefix() {
        let words = [
            sourced(method_header(4, 0x100, 0, 1), 0, 0),
            sourced(method_header(3, 0x200, 0, 2), 0, 1),
            sourced(0x55, 0, 2),
        ];
        assert!(matches!(
            decode(&words),
            Err(MaxwellPushbufferDecodeError::TruncatedPacket {
                source,
                available_arguments: 1,
                ..
            }) if source == location(0, 1)
        ));
    }

    #[test]
    fn source_errors_prevent_any_decoded_result() {
        let input = [
            Ok(sourced(method_header(4, 0x100, 0, 1), 0, 0)),
            Err(MaxwellGpfifoSourceError::ResourceExhausted),
        ];
        assert_eq!(
            decode_maxwell_pushbuffer(input),
            Err(MaxwellPushbufferDecodeError::Source(
                MaxwellGpfifoSourceError::ResourceExhausted
            ))
        );
    }

    #[test]
    fn retained_submission_decoding_reads_canonical_bytes_and_preserves_mapping_source() {
        let command_words = [method_header(1, 0x100, 2, 2), 0x1122_3344, 0x5566_7788];
        let allocation = CanonicalAllocation::zeroed(0x1000, 0x1000).unwrap();
        let command_bytes = command_words
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<_>>();
        allocation.write(0, &command_bytes).unwrap();

        let mut address_space =
            MaxwellGpuAddressSpace::new(MaxwellAddressSpaceId::new(2), SWITCH_1_GM20B_PROFILE);
        address_space
            .initialize(MaxwellAddressSpaceInitialization::default())
            .unwrap();
        let mapping = address_space
            .map(MaxwellMapRequest {
                allocation: MaxwellAllocationId::new(4),
                backing: allocation
                    .backing_range(MemoryPermissions::READ_WRITE)
                    .unwrap(),
                backing_offset: 0,
                size: 0x1000,
                allocation_alignment: 0x1000,
                page_size: 0x1000,
                kind: 0,
                cacheable: false,
                permissions: MemoryPermissions::READ_WRITE,
                fixed_offset: None,
            })
            .unwrap();
        let address = mapping.offset().get();
        let mut descriptor = [0_u8; 8];
        descriptor[..4].copy_from_slice(&(address as u32).to_le_bytes());
        descriptor[4..].copy_from_slice(
            &(((address >> 32) as u32 | ((command_words.len() as u32) << 10)).to_le_bytes()),
        );
        let decoded_gpfifo = decode_gpfifo_submission(
            SWITCH_1_GM20B_PROFILE,
            8,
            MaxwellGpfifoSubmitRequest {
                entry_count: 1,
                flags: 1 << 2,
                fence_id: 0,
                fence_value: 0,
            },
            &descriptor,
        )
        .unwrap();
        let mut channel = MaxwellGpuChannel::new(
            MaxwellChannelId::new(9),
            MaxwellChannelOwner::new(1),
            SWITCH_1_GM20B_PROFILE,
        );
        channel.bind_address_space(address_space.id()).unwrap();
        let retained = resolve_gpfifo_submission(
            &channel,
            FrontendSubmissionId::new(12),
            decoded_gpfifo,
            &address_space,
        )
        .unwrap();

        let decoded = decode_maxwell_submission(&retained, &address_space).unwrap();
        let MaxwellDecodedPacket::Methods(packet) = &decoded.packets()[0] else {
            panic!("expected method packet");
        };
        assert_eq!(packet.methods()[0].argument(), 0x1122_3344);
        assert_eq!(packet.methods()[1].argument(), 0x5566_7788);
        assert_eq!(packet.header().mapping, mapping.id());
        assert_eq!(packet.header().generation, mapping.generation());
        assert_eq!(packet.methods()[1].argument_source().word_offset, 2);

        address_space.unmap(mapping.offset()).unwrap();
        assert!(matches!(
            decode_maxwell_submission(&retained, &address_space),
            Err(MaxwellPushbufferDecodeError::Source(
                MaxwellGpfifoSourceError::StaleMapping { .. }
            ))
        ));
    }
}
