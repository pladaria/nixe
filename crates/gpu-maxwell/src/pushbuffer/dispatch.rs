//! Maxwell subchannel binding and source-preserving method dispatch.
//!
//! This layer consumes only fully decoded T7 packets. It owns frontend class
//! selection but no engine method tables, mappings, Horizon ABI, or backend
//! objects. Engine-specific interpretation begins at the next dispatch layer.

use std::fmt::{Display, Formatter};

use nixe_gpu::{FrontendSubmissionId, GpuClassId, GpuMethodId};

use crate::{
    MaxwellChannelFrontendState, MaxwellChannelId, MaxwellDecodedMethod,
    MaxwellDecodedMethodPacket, MaxwellDecodedPacket, MaxwellGpfifoSourceLocation,
    MaxwellGpuChannel, MaxwellGpuProfile, MaxwellPushbufferControl, MaxwellPushbufferSubchannel,
};

#[cfg(test)]
use crate::MaxwellDecodedPushbuffer;

/// Byte address of the PBDMA SetObject host method.
pub const MAXWELL_SET_OBJECT_METHOD: GpuMethodId = GpuMethodId(0);
/// Byte address of the legacy channel `MEM_OP_A` host method accepted by GM20B.
pub const MAXWELL_LEGACY_MEM_OP_A_METHOD: GpuMethodId = GpuMethodId(0x28);
/// Byte address of the legacy channel `MEM_OP_B` host method accepted by GM20B.
pub const MAXWELL_LEGACY_MEM_OP_B_METHOD: GpuMethodId = GpuMethodId(0x2c);

const HOST_METHOD_LIMIT: u32 = 0x100;
const SET_OBJECT_CLASS_MASK: u32 = 0xffff;
const SET_OBJECT_ENGINE_SHIFT: u32 = 16;
const SET_OBJECT_ENGINE_MASK: u32 = 0x1f;
const SET_OBJECT_DEFINED_MASK: u32 =
    SET_OBJECT_CLASS_MASK | (SET_OBJECT_ENGINE_MASK << SET_OBJECT_ENGINE_SHIFT);
const MEM_OP_A_DEFINED_MASK: u32 = 0xffff_fffc;
const MEM_OP_B_OPERAND_HIGH_MASK: u32 = 0xff;
const MEM_OP_B_OPERATION_SHIFT: u32 = 27;
const MEM_OP_B_OPERATION_MASK: u32 = 0x1f << MEM_OP_B_OPERATION_SHIFT;
const MEM_OP_B_DEFINED_MASK: u32 = MEM_OP_B_OPERATION_MASK | MEM_OP_B_OPERAND_HIGH_MASK;
const MEM_OP_B_L2_SYSMEM_INVALIDATE: u8 = 0x0e;
const MEM_OP_B_L2_FLUSH_DIRTY: u8 = 0x10;

/// Complete pointer-free source of one expanded Maxwell method write.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellMethodSource {
    channel: MaxwellChannelId,
    submission: FrontendSubmissionId,
    location: MaxwellGpfifoSourceLocation,
    subchannel: MaxwellPushbufferSubchannel,
    method: GpuMethodId,
    argument: u32,
}

impl MaxwellMethodSource {
    pub(crate) const fn emitted_by_mme(self, method: GpuMethodId, argument: u32) -> Self {
        Self {
            method,
            argument,
            ..self
        }
    }

    pub(crate) const fn with_effective_argument(self, argument: u32) -> Self {
        Self { argument, ..self }
    }

    #[must_use]
    pub const fn channel(self) -> MaxwellChannelId {
        self.channel
    }

    #[must_use]
    pub const fn submission(self) -> FrontendSubmissionId {
        self.submission
    }

    #[must_use]
    pub const fn location(self) -> MaxwellGpfifoSourceLocation {
        self.location
    }

    #[must_use]
    pub const fn subchannel(self) -> MaxwellPushbufferSubchannel {
        self.subchannel
    }

    #[must_use]
    pub const fn method(self) -> GpuMethodId {
        self.method
    }

    #[must_use]
    pub const fn argument(self) -> u32 {
        self.argument
    }
}

impl Display for MaxwellMethodSource {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{} {} {} argument={:#010x}",
            self.location, self.subchannel, self.method, self.argument
        )
    }
}

/// Effect of a verified SetObject write on the selected subchannel.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellSetObjectTransition {
    Bound,
    VerifiedExisting,
    Replaced { previous: GpuClassId },
}

/// Verified operation selected by a legacy channel `MEM_OP_B` write.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellHostMemoryOperation {
    /// Discard cached system-memory reads before later channel work proceeds.
    L2SysmemInvalidate { operand_high: u8 },
    /// Write back dirty device L2 data before later channel work proceeds.
    L2FlushDirty { operand_high: u8 },
}

/// Source-preserving semantic kind of one implemented Maxwell host method.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellHostMethod {
    LegacyMemOpA { operand_low: u32 },
    LegacyMemOpB(MaxwellHostMemoryOperation),
}

/// Semantic kind of a source-preserving method record.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellMethodDispatchKind {
    SetObject(MaxwellSetObjectTransition),
    HostMethod(MaxwellHostMethod),
    ClassMethod,
}

/// One method after a valid subchannel class has been established.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellMethodDispatch {
    source: MaxwellMethodSource,
    class: GpuClassId,
    kind: MaxwellMethodDispatchKind,
}

impl MaxwellMethodDispatch {
    pub(crate) const fn emitted_by_mme(source: MaxwellMethodSource, class: GpuClassId) -> Self {
        Self {
            source,
            class,
            kind: MaxwellMethodDispatchKind::ClassMethod,
        }
    }

    pub(crate) const fn with_effective_argument(self, argument: u32) -> Self {
        Self {
            source: self.source.with_effective_argument(argument),
            ..self
        }
    }

    #[must_use]
    pub const fn source(self) -> MaxwellMethodSource {
        self.source
    }

    #[must_use]
    pub const fn class(self) -> GpuClassId {
        self.class
    }

    #[must_use]
    pub const fn kind(self) -> MaxwellMethodDispatchKind {
        self.kind
    }
}

impl Display for MaxwellMethodDispatch {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{} {} kind={:?}",
            self.source, self.class, self.kind
        )
    }
}

/// Test observation of methods decoded and applied while dispatching a packet.
#[cfg(test)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaxwellPacketDispatch {
    methods: Box<[MaxwellMethodDispatch]>,
}

#[cfg(test)]
impl MaxwellPacketDispatch {
    #[must_use]
    pub fn methods(&self) -> &[MaxwellMethodDispatch] {
        &self.methods
    }
}

/// Failure at the class-binding boundary, before engine method execution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MaxwellMethodDispatchError {
    SourceIdentityMismatch {
        source: MaxwellGpfifoSourceLocation,
        channel: MaxwellChannelId,
        submission: FrontendSubmissionId,
    },
    MissingPacketSubchannel {
        source: MaxwellGpfifoSourceLocation,
    },
    InvalidSetObjectValue {
        source: MaxwellMethodSource,
    },
    UnsupportedSetObjectEngine {
        source: MaxwellMethodSource,
        engine: u8,
    },
    UnsupportedClassForSubchannel {
        source: MaxwellMethodSource,
        class: GpuClassId,
        expected: Option<GpuClassId>,
    },
    UnsupportedHostMethod {
        source: MaxwellMethodSource,
    },
    InvalidHostMethodValue {
        source: MaxwellMethodSource,
        defined_mask: u32,
    },
    UnsupportedHostMemoryOperation {
        source: MaxwellMethodSource,
        operation: u8,
    },
    UnboundSubchannel {
        source: MaxwellMethodSource,
    },
    UnsupportedControl {
        source: MaxwellGpfifoSourceLocation,
        operation: MaxwellPushbufferControl,
    },
    ResourceExhausted,
}

impl Display for MaxwellMethodDispatchError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SourceIdentityMismatch {
                source,
                channel,
                submission,
            } => write!(
                formatter,
                "Maxwell method source belongs to another dispatch: expected=[{channel} {submission}] actual=[{source}]"
            ),
            Self::MissingPacketSubchannel { source } => {
                write!(
                    formatter,
                    "non-empty method packet has no subchannel: {source}"
                )
            }
            Self::InvalidSetObjectValue { source } => write!(
                formatter,
                "SetObject argument sets reserved bits outside ClassId[15:0] and EngineId[20:16]: {source}"
            ),
            Self::UnsupportedSetObjectEngine { source, engine } => write!(
                formatter,
                "SetObject engine is not available in the Switch 1 profile: {source} engine-id={engine}"
            ),
            Self::UnsupportedClassForSubchannel {
                source,
                class,
                expected,
            } => {
                write!(
                    formatter,
                    "SetObject class is unsupported by the selected Switch 1 engine: {source} requested-{class}"
                )?;
                match expected {
                    Some(expected) => write!(formatter, " expected-{expected}"),
                    None => formatter.write_str(" subchannel-has-no-hardware-engine"),
                }
            }
            Self::UnsupportedHostMethod { source } => {
                write!(
                    formatter,
                    "Maxwell host method semantics are unavailable: {source}"
                )
            }
            Self::InvalidHostMethodValue {
                source,
                defined_mask,
            } => write!(
                formatter,
                "Maxwell host method argument sets bits outside its verified field mask: {source} defined-mask={defined_mask:#010x}"
            ),
            Self::UnsupportedHostMemoryOperation { source, operation } => write!(
                formatter,
                "Maxwell host memory-operation semantics are unavailable: {source} operation={operation:#04x}"
            ),
            Self::UnboundSubchannel { source } => {
                write!(
                    formatter,
                    "Maxwell method targets an unbound subchannel: {source} class=unbound"
                )
            }
            Self::UnsupportedControl { source, operation } => write!(
                formatter,
                "Maxwell pushbuffer control semantics are unavailable: {source} operation={operation:?}"
            ),
            Self::ResourceExhausted => {
                formatter.write_str("Maxwell method dispatch exhausted host resources")
            }
        }
    }
}

impl std::error::Error for MaxwellMethodDispatchError {}

pub(crate) enum MaxwellMethodStreamError<E> {
    Dispatch(MaxwellMethodDispatchError),
    Consumer(E),
}

/// Applies class binding and streams methods without retaining a second packet.
pub(crate) fn stream_maxwell_packet_methods<E>(
    channel: &mut MaxwellGpuChannel,
    submission: FrontendSubmissionId,
    packet: &MaxwellDecodedPacket,
    consume: &mut impl FnMut(&mut MaxwellGpuChannel, MaxwellMethodDispatch) -> Result<(), E>,
) -> Result<(), MaxwellMethodStreamError<E>> {
    match packet {
        MaxwellDecodedPacket::Methods(packet) => {
            stream_method_packet(channel, submission, packet, consume)
        }
        MaxwellDecodedPacket::Control { operation, source } => {
            validate_source_identity(channel.id(), submission, *source)
                .map_err(MaxwellMethodStreamError::Dispatch)?;
            match operation {
                MaxwellPushbufferControl::Nop | MaxwellPushbufferControl::EndSegment => Ok(()),
                MaxwellPushbufferControl::SetSubdeviceMask(_)
                | MaxwellPushbufferControl::StoreSubdeviceMask(_)
                | MaxwellPushbufferControl::UseSubdeviceMask => {
                    Err(MaxwellMethodStreamError::Dispatch(
                        MaxwellMethodDispatchError::UnsupportedControl {
                            source: *source,
                            operation: *operation,
                        },
                    ))
                }
            }
        }
    }
}

#[cfg(test)]
pub fn dispatch_maxwell_packet(
    channel: &mut MaxwellGpuChannel,
    submission: FrontendSubmissionId,
    packet: &MaxwellDecodedPacket,
) -> Result<MaxwellPacketDispatch, MaxwellMethodDispatchError> {
    let mut methods = Vec::new();
    stream_maxwell_packet_methods(channel, submission, packet, &mut |_, method| {
        methods.push(method);
        Ok::<(), std::convert::Infallible>(())
    })
    .map_err(|error| match error {
        MaxwellMethodStreamError::Dispatch(error) => error,
        MaxwellMethodStreamError::Consumer(never) => match never {},
    })?;
    Ok(MaxwellPacketDispatch {
        methods: methods.into_boxed_slice(),
    })
}

/// Dispatches decoded packets and methods in stream order for tests.
#[cfg(test)]
pub fn dispatch_maxwell_pushbuffer(
    channel: &mut MaxwellGpuChannel,
    submission: FrontendSubmissionId,
    pushbuffer: &MaxwellDecodedPushbuffer,
) -> Result<Box<[MaxwellPacketDispatch]>, MaxwellMethodDispatchError> {
    let mut packets = Vec::new();
    packets
        .try_reserve_exact(pushbuffer.packets().len())
        .map_err(|_| MaxwellMethodDispatchError::ResourceExhausted)?;
    for packet in pushbuffer.packets() {
        packets.push(dispatch_maxwell_packet(channel, submission, packet)?);
    }
    Ok(packets.into_boxed_slice())
}

fn stream_method_packet<E>(
    channel: &mut MaxwellGpuChannel,
    submission: FrontendSubmissionId,
    packet: &MaxwellDecodedMethodPacket,
    consume: &mut impl FnMut(&mut MaxwellGpuChannel, MaxwellMethodDispatch) -> Result<(), E>,
) -> Result<(), MaxwellMethodStreamError<E>> {
    validate_source_identity(channel.id(), submission, packet.header())
        .map_err(MaxwellMethodStreamError::Dispatch)?;
    if packet.methods().is_empty() {
        return Ok(());
    }
    let subchannel = packet
        .subchannel()
        .ok_or(MaxwellMethodDispatchError::MissingPacketSubchannel {
            source: packet.header(),
        })
        .map_err(MaxwellMethodStreamError::Dispatch)?;
    let channel_id = channel.id();
    let profile = channel.profile();
    for method in packet.methods() {
        let method = dispatch_method(
            channel_id,
            profile,
            submission,
            subchannel,
            *method,
            channel.frontend_mut(),
        )
        .map_err(MaxwellMethodStreamError::Dispatch)?;
        consume(channel, method).map_err(MaxwellMethodStreamError::Consumer)?;
    }
    Ok(())
}

fn dispatch_method(
    channel: MaxwellChannelId,
    profile: MaxwellGpuProfile,
    submission: FrontendSubmissionId,
    subchannel: MaxwellPushbufferSubchannel,
    method: MaxwellDecodedMethod,
    frontend: &mut MaxwellChannelFrontendState,
) -> Result<MaxwellMethodDispatch, MaxwellMethodDispatchError> {
    let source = MaxwellMethodSource {
        channel,
        submission,
        location: method.argument_source(),
        subchannel,
        method: method.method(),
        argument: method.argument(),
    };
    validate_source_identity(channel, submission, source.location)?;
    if method.method() == MAXWELL_SET_OBJECT_METHOD {
        return preflight_set_object(profile, source, frontend);
    }
    if method.method() == MAXWELL_LEGACY_MEM_OP_A_METHOD {
        return preflight_legacy_mem_op_a(profile, source, frontend);
    }
    if method.method() == MAXWELL_LEGACY_MEM_OP_B_METHOD {
        return preflight_legacy_mem_op_b(profile, source);
    }
    if method.method().0 < HOST_METHOD_LIMIT {
        return Err(MaxwellMethodDispatchError::UnsupportedHostMethod { source });
    }
    let class = frontend
        .subchannel_binding(subchannel)
        .ok_or(MaxwellMethodDispatchError::UnboundSubchannel { source })?;
    Ok(MaxwellMethodDispatch {
        source,
        class,
        kind: MaxwellMethodDispatchKind::ClassMethod,
    })
}

fn preflight_legacy_mem_op_a(
    profile: MaxwellGpuProfile,
    source: MaxwellMethodSource,
    frontend: &mut MaxwellChannelFrontendState,
) -> Result<MaxwellMethodDispatch, MaxwellMethodDispatchError> {
    // GM20B's B06F header moved these fields to MEM_OP_C/D, but Switch's
    // captured Nouveau stream exercises the legacy A/B compatibility aperture.
    // The field layouts are published in NVIDIA's A16F and B06F class headers:
    // https://github.com/NVIDIA/open-gpu-kernel-modules/blob/580.126.09/src/common/sdk/nvidia/inc/class/cla16f.h
    // https://github.com/NVIDIA/open-gpu-kernel-modules/blob/580.126.09/src/common/sdk/nvidia/inc/class/clb06f.h
    if source.argument & !MEM_OP_A_DEFINED_MASK != 0 {
        return Err(MaxwellMethodDispatchError::InvalidHostMethodValue {
            source,
            defined_mask: MEM_OP_A_DEFINED_MASK,
        });
    }
    let operand_low = source.argument & MEM_OP_A_DEFINED_MASK;
    frontend.set_legacy_mem_op_a(operand_low);
    Ok(MaxwellMethodDispatch {
        source,
        class: profile.classes().gpfifo(),
        kind: MaxwellMethodDispatchKind::HostMethod(MaxwellHostMethod::LegacyMemOpA {
            operand_low,
        }),
    })
}

fn preflight_legacy_mem_op_b(
    profile: MaxwellGpuProfile,
    source: MaxwellMethodSource,
) -> Result<MaxwellMethodDispatch, MaxwellMethodDispatchError> {
    if source.argument & !MEM_OP_B_DEFINED_MASK != 0 {
        return Err(MaxwellMethodDispatchError::InvalidHostMethodValue {
            source,
            defined_mask: MEM_OP_B_DEFINED_MASK,
        });
    }
    let operation = ((source.argument & MEM_OP_B_OPERATION_MASK) >> MEM_OP_B_OPERATION_SHIFT) as u8;
    let operation = match operation {
        MEM_OP_B_L2_SYSMEM_INVALIDATE => MaxwellHostMemoryOperation::L2SysmemInvalidate {
            operand_high: (source.argument & MEM_OP_B_OPERAND_HIGH_MASK) as u8,
        },
        MEM_OP_B_L2_FLUSH_DIRTY => MaxwellHostMemoryOperation::L2FlushDirty {
            operand_high: (source.argument & MEM_OP_B_OPERAND_HIGH_MASK) as u8,
        },
        operation => {
            return Err(MaxwellMethodDispatchError::UnsupportedHostMemoryOperation {
                source,
                operation,
            });
        }
    };
    Ok(MaxwellMethodDispatch {
        source,
        class: profile.classes().gpfifo(),
        kind: MaxwellMethodDispatchKind::HostMethod(MaxwellHostMethod::LegacyMemOpB(operation)),
    })
}

fn validate_source_identity(
    channel: MaxwellChannelId,
    submission: FrontendSubmissionId,
    source: MaxwellGpfifoSourceLocation,
) -> Result<(), MaxwellMethodDispatchError> {
    if source.channel != channel || source.frontend != submission {
        return Err(MaxwellMethodDispatchError::SourceIdentityMismatch {
            source,
            channel,
            submission,
        });
    }
    Ok(())
}

fn preflight_set_object(
    profile: MaxwellGpuProfile,
    source: MaxwellMethodSource,
    frontend: &mut MaxwellChannelFrontendState,
) -> Result<MaxwellMethodDispatch, MaxwellMethodDispatchError> {
    // NVIDIA defines SetObject as NV_UDMA_OBJECT at byte method 0. It verifies
    // the class supported by the fixed engine selected by the graphics-runlist
    // subchannel; it does not select a host class and class zero is not a reset
    // command:
    // https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/manuals/volta/gv100/dev_pbdma.ref.txt#L3146-L3187
    // The fixed graphics subchannel mapping is documented in the paired RAM
    // reference at the same pinned revision:
    // https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/manuals/volta/gv100/dev_ram.ref.txt#L948-L975
    // Switch 1's public class table records the GM20B extension carrying
    // ClassId[15:0] and EngineId[20:16]. The advertised profile contains one
    // instance of each graphics engine, so only engine zero is currently
    // meaningful:
    // https://switchbrew.org/w/index.php?title=GPU_Classes&oldid=12790
    if source.argument & !SET_OBJECT_DEFINED_MASK != 0 {
        return Err(MaxwellMethodDispatchError::InvalidSetObjectValue { source });
    }
    let engine = ((source.argument >> SET_OBJECT_ENGINE_SHIFT) & SET_OBJECT_ENGINE_MASK) as u8;
    if engine != 0 {
        return Err(MaxwellMethodDispatchError::UnsupportedSetObjectEngine { source, engine });
    }
    let class = GpuClassId(source.argument & SET_OBJECT_CLASS_MASK);
    let expected = graphics_subchannel_class(profile, source.subchannel);
    if expected != Some(class) {
        return Err(MaxwellMethodDispatchError::UnsupportedClassForSubchannel {
            source,
            class,
            expected,
        });
    }
    let transition = match frontend.bind_subchannel(source.subchannel, class) {
        None => MaxwellSetObjectTransition::Bound,
        Some(previous) if previous == class => MaxwellSetObjectTransition::VerifiedExisting,
        Some(previous) => MaxwellSetObjectTransition::Replaced { previous },
    };
    Ok(MaxwellMethodDispatch {
        source,
        class,
        kind: MaxwellMethodDispatchKind::SetObject(transition),
    })
}

const fn graphics_subchannel_class(
    profile: MaxwellGpuProfile,
    subchannel: MaxwellPushbufferSubchannel,
) -> Option<GpuClassId> {
    let classes = profile.classes();
    match subchannel.get() {
        0 => Some(classes.three_d()),
        1 => Some(classes.compute()),
        2 => Some(classes.inline_to_memory()),
        3 => Some(classes.two_d()),
        4 => Some(classes.dma_copy()),
        5..=7 => None,
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use nixe_gpu::{GpuVirtualAddress, MappingGeneration};

    use super::*;
    use crate::{
        MaxwellChannelOwner, MaxwellGpfifoSourceLocation, MaxwellMappingId, MaxwellPushbufferWord,
        SWITCH_1_GM20B_PROFILE, decode_maxwell_pushbuffer,
    };

    fn channel() -> MaxwellGpuChannel {
        MaxwellGpuChannel::new(
            MaxwellChannelId::new(7),
            MaxwellChannelOwner::new(1),
            SWITCH_1_GM20B_PROFILE,
        )
    }

    fn location(word: u64) -> MaxwellGpfifoSourceLocation {
        MaxwellGpfifoSourceLocation {
            channel: MaxwellChannelId::new(7),
            frontend: FrontendSubmissionId::new(11),
            entry_index: 2,
            pushbuffer: GpuVirtualAddress::try_new(0x8000, 40).unwrap(),
            word_offset: word,
            mapping: MaxwellMappingId::new(3),
            generation: MappingGeneration::new(5),
        }
    }

    fn word(value: u32, offset: u64) -> MaxwellPushbufferWord {
        MaxwellPushbufferWord::new(value, location(offset))
    }

    const fn header(opcode: u32, method: u32, subchannel: u32, count: u32) -> u32 {
        opcode << 29 | count << 16 | subchannel << 13 | method
    }

    fn decode(words: &[MaxwellPushbufferWord]) -> MaxwellDecodedPushbuffer {
        decode_maxwell_pushbuffer(words.iter().copied().map(Ok)).unwrap()
    }

    #[test]
    fn set_object_binds_and_revalidates_the_expected_engine_class() {
        let class = SWITCH_1_GM20B_PROFILE.classes().three_d();
        let decoded = decode(&[
            word(header(1, 0, 0, 1), 0),
            word(class.0, 1),
            word(header(1, 0, 0, 1), 2),
            word(class.0, 3),
        ]);
        let mut channel = channel();
        let dispatched =
            dispatch_maxwell_pushbuffer(&mut channel, FrontendSubmissionId::new(11), &decoded)
                .unwrap();

        assert_eq!(dispatched.len(), 2);
        assert_eq!(
            dispatched[0].methods()[0].kind(),
            MaxwellMethodDispatchKind::SetObject(MaxwellSetObjectTransition::Bound)
        );
        assert_eq!(
            dispatched[1].methods()[0].kind(),
            MaxwellMethodDispatchKind::SetObject(MaxwellSetObjectTransition::VerifiedExisting)
        );
        assert_eq!(
            channel
                .frontend()
                .subchannel_binding(MaxwellPushbufferSubchannel::try_new(0).unwrap()),
            Some(class)
        );
    }

    #[test]
    fn failing_packet_keeps_the_successfully_dispatched_frontend_prefix() {
        let class = SWITCH_1_GM20B_PROFILE.classes().three_d();
        let subchannel = MaxwellPushbufferSubchannel::try_new(0).unwrap();
        let decoded = decode(&[word(header(1, 0, 0, 2), 0), word(class.0, 1), word(0, 2)]);
        let mut channel = channel();

        let error = dispatch_maxwell_packet(
            &mut channel,
            FrontendSubmissionId::new(11),
            &decoded.packets()[0],
        )
        .unwrap_err();

        assert!(matches!(
            error,
            MaxwellMethodDispatchError::UnsupportedHostMethod { .. }
        ));
        assert_eq!(
            channel.frontend().subchannel_binding(subchannel),
            Some(class)
        );
    }

    #[test]
    fn class_methods_preserve_complete_source_provenance() {
        let class = SWITCH_1_GM20B_PROFILE.classes().three_d();
        let decoded = decode(&[
            word(header(1, 0, 0, 1), 0),
            word(class.0, 1),
            word(header(1, 0x100, 0, 1), 2),
            word(0x1234_5678, 3),
        ]);
        let mut channel = channel();
        let dispatched =
            dispatch_maxwell_pushbuffer(&mut channel, FrontendSubmissionId::new(11), &decoded)
                .unwrap();
        let method = dispatched[1].methods()[0];

        assert_eq!(method.class(), class);
        assert_eq!(method.kind(), MaxwellMethodDispatchKind::ClassMethod);
        assert_eq!(method.source().channel(), MaxwellChannelId::new(7));
        assert_eq!(method.source().submission(), FrontendSubmissionId::new(11));
        assert_eq!(method.source().location(), location(3));
        assert_eq!(method.source().subchannel().get(), 0);
        assert_eq!(method.source().method(), GpuMethodId(0x400));
        assert_eq!(method.source().argument(), 0x1234_5678);
        assert!(method.to_string().contains("mapping-generation=5"));
    }

    #[test]
    fn ordinary_method_on_unbound_subchannel_is_rejected_without_state_change() {
        let decoded = decode(&[word(header(1, 0x100, 2, 1), 0), word(9, 1)]);
        let mut channel = channel();
        let before = channel.frontend();
        assert!(matches!(
            dispatch_maxwell_packet(
                &mut channel,
                FrontendSubmissionId::new(11),
                &decoded.packets()[0]
            ),
            Err(MaxwellMethodDispatchError::UnboundSubchannel { source })
                if source.location() == location(1)
                    && source.subchannel().get() == 2
                    && source.method() == GpuMethodId(0x400)
        ));
        assert_eq!(channel.frontend(), before);
    }

    #[test]
    fn retained_source_identity_cannot_be_relabelled_for_another_submission() {
        let class = SWITCH_1_GM20B_PROFILE.classes().three_d();
        let decoded = decode(&[word(header(1, 0, 0, 1), 0), word(class.0, 1)]);
        let mut channel = channel();
        assert!(matches!(
            dispatch_maxwell_packet(
                &mut channel,
                FrontendSubmissionId::new(12),
                &decoded.packets()[0]
            ),
            Err(MaxwellMethodDispatchError::SourceIdentityMismatch {
                source,
                submission,
                ..
            }) if source == location(0) && submission == FrontendSubmissionId::new(12)
        ));
        assert_eq!(channel.frontend(), MaxwellChannelFrontendState::default());
    }

    #[test]
    fn later_invalid_set_object_keeps_the_valid_prefix() {
        let class = SWITCH_1_GM20B_PROFILE.classes().three_d();
        let decoded = decode(&[
            word(header(3, 0, 0, 2), 0),
            word(class.0, 1),
            word(class.0 | 0x0020_0000, 2),
        ]);
        let mut channel = channel();
        let before = channel.frontend();
        assert!(matches!(
            dispatch_maxwell_packet(
                &mut channel,
                FrontendSubmissionId::new(11),
                &decoded.packets()[0]
            ),
            Err(MaxwellMethodDispatchError::InvalidSetObjectValue { source })
                if source.location() == location(2)
        ));
        assert_ne!(channel.frontend(), before);
    }

    #[test]
    fn wrong_engine_class_and_software_subchannels_are_typed_failures() {
        let compute = SWITCH_1_GM20B_PROFILE.classes().compute();
        for (subchannel, expected) in [
            (0, Some(SWITCH_1_GM20B_PROFILE.classes().three_d())),
            (5, None),
        ] {
            let decoded = decode(&[word(header(1, 0, subchannel, 1), 0), word(compute.0, 1)]);
            let mut channel = channel();
            assert!(matches!(
                dispatch_maxwell_packet(
                    &mut channel,
                    FrontendSubmissionId::new(11),
                    &decoded.packets()[0]
                ),
                Err(MaxwellMethodDispatchError::UnsupportedClassForSubchannel {
                    class,
                    expected: actual,
                    ..
                }) if class == compute && actual == expected
            ));
        }
    }

    #[test]
    fn nonzero_set_object_engine_is_not_silently_discarded() {
        let class = SWITCH_1_GM20B_PROFILE.classes().three_d();
        let decoded = decode(&[
            word(header(1, 0, 0, 1), 0),
            word(class.0 | (1 << SET_OBJECT_ENGINE_SHIFT), 1),
        ]);
        let mut channel = channel();
        assert!(matches!(
            dispatch_maxwell_packet(
                &mut channel,
                FrontendSubmissionId::new(11),
                &decoded.packets()[0]
            ),
            Err(MaxwellMethodDispatchError::UnsupportedSetObjectEngine { engine: 1, .. })
        ));
        assert_eq!(channel.frontend(), MaxwellChannelFrontendState::default());
    }

    #[test]
    fn channel_reset_clears_bindings_without_fabricating_set_object_zero() {
        let class = SWITCH_1_GM20B_PROFILE.classes().three_d();
        let decoded = decode(&[word(header(1, 0, 0, 1), 0), word(class.0, 1)]);
        let mut channel = channel();
        dispatch_maxwell_pushbuffer(&mut channel, FrontendSubmissionId::new(11), &decoded).unwrap();
        channel.reset_subchannel_bindings();
        assert_eq!(
            channel
                .frontend()
                .subchannel_binding(MaxwellPushbufferSubchannel::try_new(0).unwrap()),
            None
        );
    }

    #[test]
    fn captured_legacy_mem_op_b_flushes_l2_without_a_subchannel_binding() {
        let decoded = decode(&[word(0x2001_c00b, 0), word(0x8000_0000, 1)]);
        let mut channel = channel();
        let dispatched = dispatch_maxwell_packet(
            &mut channel,
            FrontendSubmissionId::new(11),
            &decoded.packets()[0],
        )
        .unwrap();

        let method = dispatched.methods()[0];
        assert_eq!(method.class(), SWITCH_1_GM20B_PROFILE.classes().gpfifo());
        assert_eq!(method.source().subchannel().get(), 6);
        assert_eq!(method.source().method(), MAXWELL_LEGACY_MEM_OP_B_METHOD);
        assert_eq!(
            method.kind(),
            MaxwellMethodDispatchKind::HostMethod(MaxwellHostMethod::LegacyMemOpB(
                MaxwellHostMemoryOperation::L2FlushDirty { operand_high: 0 },
            ))
        );
        assert_eq!(
            channel
                .frontend()
                .subchannel_binding(method.source().subchannel()),
            None
        );
    }

    #[test]
    fn captured_legacy_mem_op_b_invalidates_stale_l2_system_memory_reads() {
        let decoded = decode(&[word(0x2001_c00b, 0), word(0x7000_0000, 1)]);
        let mut channel = channel();
        let dispatched = dispatch_maxwell_packet(
            &mut channel,
            FrontendSubmissionId::new(11),
            &decoded.packets()[0],
        )
        .unwrap();

        let method = dispatched.methods()[0];
        assert_eq!(method.source().subchannel().get(), 6);
        assert_eq!(method.source().method(), MAXWELL_LEGACY_MEM_OP_B_METHOD);
        assert_eq!(
            method.kind(),
            MaxwellMethodDispatchKind::HostMethod(MaxwellHostMethod::LegacyMemOpB(
                MaxwellHostMemoryOperation::L2SysmemInvalidate { operand_high: 0 },
            ))
        );
    }

    #[test]
    fn legacy_mem_op_a_and_b_commit_atomically_and_preserve_operands() {
        let decoded = decode(&[
            word(header(1, 0x28 / 4, 6, 2), 0),
            word(0x1234_5678, 1),
            word(0x8000_00ab, 2),
        ]);
        let mut channel = channel();
        let dispatched = dispatch_maxwell_packet(
            &mut channel,
            FrontendSubmissionId::new(11),
            &decoded.packets()[0],
        )
        .unwrap();

        assert_eq!(channel.frontend().legacy_mem_op_a(), Some(0x1234_5678));
        assert_eq!(
            dispatched.methods()[0].kind(),
            MaxwellMethodDispatchKind::HostMethod(MaxwellHostMethod::LegacyMemOpA {
                operand_low: 0x1234_5678,
            })
        );
        assert_eq!(
            dispatched.methods()[1].kind(),
            MaxwellMethodDispatchKind::HostMethod(MaxwellHostMethod::LegacyMemOpB(
                MaxwellHostMemoryOperation::L2FlushDirty { operand_high: 0xab },
            ))
        );
    }

    #[test]
    fn invalid_or_unsupported_legacy_mem_ops_fail_after_applying_valid_prefix() {
        let invalid = decode(&[
            word(header(1, 0x28 / 4, 6, 2), 0),
            word(0x1234_5678, 1),
            word(0x8000_0100, 2),
        ]);
        let mut channel = channel();
        assert!(matches!(
            dispatch_maxwell_packet(
                &mut channel,
                FrontendSubmissionId::new(11),
                &invalid.packets()[0]
            ),
            Err(MaxwellMethodDispatchError::InvalidHostMethodValue { source, .. })
                if source.location() == location(2)
        ));
        assert_ne!(channel.frontend().legacy_mem_op_a(), None);

        let unsupported = decode(&[word(header(1, 0x2c / 4, 6, 1), 0), word(0x2800_0000, 1)]);
        assert!(matches!(
            dispatch_maxwell_packet(
                &mut channel,
                FrontendSubmissionId::new(11),
                &unsupported.packets()[0]
            ),
            Err(MaxwellMethodDispatchError::UnsupportedHostMemoryOperation { operation: 5, .. })
        ));
    }

    #[test]
    fn unsupported_host_and_subdevice_controls_remain_fatal_boundaries() {
        let host = decode(&[word(header(4, 2, 0, 7), 0)]);
        let control = decode(&[word(1 << 16, 0)]);
        let mut channel = channel();
        assert!(matches!(
            dispatch_maxwell_packet(
                &mut channel,
                FrontendSubmissionId::new(11),
                &host.packets()[0]
            ),
            Err(MaxwellMethodDispatchError::UnsupportedHostMethod { source })
                if source.method() == GpuMethodId(8)
        ));
        assert!(matches!(
            dispatch_maxwell_packet(
                &mut channel,
                FrontendSubmissionId::new(11),
                &control.packets()[0]
            ),
            Err(MaxwellMethodDispatchError::UnsupportedControl { .. })
        ));
    }

    #[test]
    fn zero_count_packet_and_verified_controls_have_no_binding_effect() {
        let decoded = decode(&[
            word(header(1, 0xfff, 7, 0), 0),
            word(0, 1),
            word(7 << 29, 2),
        ]);
        let mut channel = channel();
        let dispatched =
            dispatch_maxwell_pushbuffer(&mut channel, FrontendSubmissionId::new(11), &decoded)
                .unwrap();
        assert_eq!(dispatched.len(), 3);
        assert!(dispatched.iter().all(|packet| packet.methods().is_empty()));
        assert_eq!(channel.frontend(), MaxwellChannelFrontendState::default());
    }
}
