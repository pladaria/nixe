//! Side-effect-free Maxwell pushbuffer packet decoding.
//!
//! Packet syntax is decoded before any channel, subchannel, class, or backend
//! state can be observed or changed. The bit layout and command semantics are
//! defined by NVIDIA's public FIFO DMA reference:
//! https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/manuals/volta/gv100/dev_ram.ref.txt#L893-L1227
//! The same GF100+ encodings, including increment-once, are independently
//! described by the pinned envytools DMA-pusher reference:
//! https://github.com/envytools/envytools/blob/f102b82381f3f11cee113d16374c87091db039d9/docs/hw/fifo/dma-pusher.rst#gf100-commands

mod methods;
use methods::PacketMethods;
pub use methods::{MaxwellMethodIterator, MaxwellMethodSpan};
mod sources;
pub(crate) use sources::CommandStream;
pub use sources::MaxwellCommandSourceScratch;

use std::fmt::{Display, Formatter};

use nixe_gpu::GpuMethodId;

use crate::{
    MaxwellGpfifoSourceError, MaxwellGpfifoSourceLocation, MaxwellGpuMemory,
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

/// One command word paired with its exact source location.
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
    methods: PacketMethods,
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
    pub fn methods(&self) -> MaxwellMethodSpan<'_> {
        MaxwellMethodSpan::new(self)
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

#[derive(Clone, Copy)]
pub(crate) enum PacketView<'a> {
    Methods {
        header: MaxwellGpfifoSourceLocation,
        subchannel: Option<MaxwellPushbufferSubchannel>,
        methods: MaxwellMethodSpan<'a>,
        last: bool,
    },
    Control {
        operation: MaxwellPushbufferControl,
        source: MaxwellGpfifoSourceLocation,
    },
}
#[cfg(test)]
impl MaxwellDecodedPacket {
    pub(crate) fn view(&self) -> PacketView<'_> {
        match self {
            Self::Methods(packet) => PacketView::Methods {
                header: packet.header(),
                subchannel: packet.subchannel(),
                methods: packet.methods(),
                last: true,
            },
            Self::Control { operation, source } => PacketView::Control {
                operation: *operation,
                source: *source,
            },
        }
    }
}

/// Owned diagnostic fixture for a complete packet or control instruction.
#[cfg(test)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MaxwellDecodedPacket {
    Methods(MaxwellDecodedMethodPacket),
    Control {
        operation: MaxwellPushbufferControl,
        source: MaxwellGpfifoSourceLocation,
    },
}

#[cfg(test)]
impl MaxwellDecodedPacket {
    #[must_use]
    pub const fn source(&self) -> MaxwellGpfifoSourceLocation {
        match self {
            Self::Methods(packet) => packet.header(),
            Self::Control { source, .. } => *source,
        }
    }
}

/// Owned encoded-input fixture for engine unit tests.
#[cfg(test)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaxwellDecodedPushbuffer {
    packets: Box<[MaxwellDecodedPacket]>,
}

#[cfg(test)]
impl MaxwellDecodedPushbuffer {
    #[must_use]
    pub fn packets(&self) -> &[MaxwellDecodedPacket] {
        &self.packets
    }
}

/// Failure while reading or validating packet syntax.
#[derive(Clone, Debug)]
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
#[cfg(test)]
pub fn decode_maxwell_pushbuffer<I>(
    words: I,
) -> Result<MaxwellDecodedPushbuffer, MaxwellPushbufferDecodeError>
where
    I: IntoIterator<Item = Result<MaxwellPushbufferWord, MaxwellGpfifoSourceError>>,
{
    sources::fixture_packets(words.into_iter().collect::<Result<Vec<_>, _>>()?)
}

#[derive(Clone, Copy)]
struct MethodHeader {
    source: MaxwellGpfifoSourceLocation,
    mode: MaxwellMethodPacketMode,
    first: u32,
    count: u16,
    immediate: u32,
    subchannel: Option<MaxwellPushbufferSubchannel>,
}
enum Header {
    Control(MaxwellPushbufferControl),
    Methods(MethodHeader),
}
fn decode_header(header: MaxwellPushbufferWord) -> Result<Header, MaxwellPushbufferDecodeError> {
    let word = header.value;
    if word == 0 {
        return Ok(Header::Control(MaxwellPushbufferControl::Nop));
    }
    let opcode = ((word >> SECONDARY_OPCODE_SHIFT) & SECONDARY_OPCODE_MASK) as u8;
    if opcode == OPCODE_GROUP_ZERO {
        return decode_group_zero_control(header).map(Header::Control);
    }
    if opcode == OPCODE_END_SEGMENT {
        return Ok(Header::Control(MaxwellPushbufferControl::EndSegment));
    }
    let Some(encoding) = METHOD_ENCODINGS
        .iter()
        .find(|encoding| encoding.opcode == opcode)
    else {
        return Err(MaxwellPushbufferDecodeError::UnknownEncoding {
            source: header.location,
            word,
            secondary_opcode: opcode,
        });
    };
    let mode = encoding.mode;
    let encoded = ((word >> COUNT_SHIFT) & COUNT_MASK) as u16;
    let count = if mode == MaxwellMethodPacketMode::Immediate {
        1
    } else {
        encoded
    };
    if count > 0 && word & METHOD_RESERVED_BIT != 0 {
        return Err(MaxwellPushbufferDecodeError::ReservedMethodBit {
            source: header.location,
            word,
        });
    }
    let first = word & METHOD_ADDRESS_MASK;
    if count > 0 {
        validate_method_range(header.location, mode, first, count)?;
    }
    Ok(Header::Methods(MethodHeader {
        source: header.location,
        mode,
        first,
        count,
        immediate: u32::from(encoded),
        subchannel: (count > 0).then_some(MaxwellPushbufferSubchannel(
            ((word >> SUBCHANNEL_SHIFT) & SUBCHANNEL_MASK) as u8,
        )),
    }))
}

fn decode_group_zero_control(
    word: MaxwellPushbufferWord,
) -> Result<MaxwellPushbufferControl, MaxwellPushbufferDecodeError> {
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
    Ok(operation)
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

#[cfg(test)]
mod tests {
    fn control_packet(
        operation: super::MaxwellPushbufferControl,
        word: super::MaxwellPushbufferWord,
    ) -> super::MaxwellDecodedPacket {
        super::MaxwellDecodedPacket::Control {
            operation,
            source: word.location,
        }
    }

    use nixe_gpu::{FrontendSubmissionId, GpuVirtualAddress, MappingGeneration};

    use super::*;
    use crate::{MaxwellChannelId, MaxwellMappingId};

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
            increment_once.methods().get(2).unwrap().argument_source(),
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
        assert_eq!(
            packet.methods().get(0).unwrap().method(),
            GpuMethodId(0x8d0)
        );
        assert_eq!(packet.methods().get(0).unwrap().argument(), 0x1abc);
        assert_eq!(
            packet.methods().get(0).unwrap().argument_source(),
            location(0, 4)
        );
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
        assert_eq!(first.methods().get(0).unwrap().method(), GpuMethodId(0));
        assert_eq!(last.subchannel().unwrap().get(), 7);
        assert_eq!(last.methods().get(0).unwrap().method(), GpuMethodId(0x3ffc));
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
        assert_eq!(
            packet.methods().get(0).unwrap().argument_source(),
            location(1, 0)
        );
        assert_eq!(
            packet.methods().get(1).unwrap().argument_source(),
            location(1, 1)
        );
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
        let Err(MaxwellPushbufferDecodeError::TruncatedPacket {
            source,
            expected_arguments,
            available_arguments,
        }) = decode(&[truncated, sourced(1, 0, 5)])
        else {
            panic!("expected truncated packet")
        };
        assert_eq!(source, location(0, 4));
        assert_eq!((expected_arguments, available_arguments), (3, 1));

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
}
