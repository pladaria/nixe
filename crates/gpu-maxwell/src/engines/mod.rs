//! Declarative Maxwell class-method routing.
//!
//! This layer validates every class method in a decoded packet before the
//! packet's binding state is committed. It contains no Horizon ABI, guest
//! mappings, scheduler state, or host-backend objects.

mod three_d;

use std::fmt::{Display, Formatter};

use nixe_gpu::{FrontendSubmissionId, GpuClassId, GpuMethodId};

use crate::{
    MaxwellDecodedPacket, MaxwellDecodedPushbuffer, MaxwellGpuChannel, MaxwellMethodDispatch,
    MaxwellMethodDispatchError, MaxwellMethodDispatchKind, MaxwellMethodSource,
    MaxwellPacketDispatch, commit_maxwell_packet, preflight_maxwell_packet,
};

/// Execution layer required by a known method whose semantics are unavailable.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellEngineCapability {
    NeutralExecution,
    HostBackend,
}

impl Display for MaxwellEngineCapability {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::NeutralExecution => "neutral-execution",
            Self::HostBackend => "host-backend",
        })
    }
}

/// Stable metadata for one verified Maxwell class method.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellEngineMethodMetadata {
    class: GpuClassId,
    class_name: &'static str,
    method: GpuMethodId,
    method_name: &'static str,
}

impl MaxwellEngineMethodMetadata {
    pub(crate) const fn new(
        class: GpuClassId,
        class_name: &'static str,
        method: GpuMethodId,
        method_name: &'static str,
    ) -> Self {
        Self {
            class,
            class_name,
            method,
            method_name,
        }
    }

    #[must_use]
    pub const fn class(self) -> GpuClassId {
        self.class
    }

    #[must_use]
    pub const fn class_name(self) -> &'static str {
        self.class_name
    }

    #[must_use]
    pub const fn method(self) -> GpuMethodId {
        self.method
    }

    #[must_use]
    pub const fn method_name(self) -> &'static str {
        self.method_name
    }
}

/// Host-independent effect of one implemented frontend method.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellEngineMethodEffect {
    NoOperation,
}

/// One named, validated class method ready for an atomic packet commit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellEngineMethodDispatch {
    method: MaxwellMethodDispatch,
    metadata: MaxwellEngineMethodMetadata,
    effect: MaxwellEngineMethodEffect,
}

impl MaxwellEngineMethodDispatch {
    pub(crate) const fn new(
        method: MaxwellMethodDispatch,
        metadata: MaxwellEngineMethodMetadata,
        effect: MaxwellEngineMethodEffect,
    ) -> Self {
        Self {
            method,
            metadata,
            effect,
        }
    }

    #[must_use]
    pub const fn method(self) -> MaxwellMethodDispatch {
        self.method
    }

    #[must_use]
    pub const fn metadata(self) -> MaxwellEngineMethodMetadata {
        self.metadata
    }

    #[must_use]
    pub const fn effect(self) -> MaxwellEngineMethodEffect {
        self.effect
    }
}

/// Complete engine preflight paired with its class-binding preflight.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaxwellEnginePacketDispatch {
    binding: MaxwellPacketDispatch,
    methods: Box<[MaxwellEngineMethodDispatch]>,
}

impl MaxwellEnginePacketDispatch {
    #[must_use]
    pub const fn binding(&self) -> &MaxwellPacketDispatch {
        &self.binding
    }

    #[must_use]
    pub fn methods(&self) -> &[MaxwellEngineMethodDispatch] {
        &self.methods
    }
}

/// Typed class-dispatch boundary. Missing coverage is never a guest result.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MaxwellEngineDispatchError {
    Binding(MaxwellMethodDispatchError),
    UnsupportedClass {
        source: MaxwellMethodSource,
        class: GpuClassId,
        class_name: &'static str,
    },
    UnknownClass {
        source: MaxwellMethodSource,
        class: GpuClassId,
    },
    UnsupportedMethod {
        source: MaxwellMethodSource,
        metadata: &'static MaxwellEngineMethodMetadata,
    },
    UnknownMethod {
        source: MaxwellMethodSource,
        class_name: &'static str,
    },
    InvalidMethodValue {
        source: MaxwellMethodSource,
        metadata: &'static MaxwellEngineMethodMetadata,
        defined_mask: u32,
    },
    MissingCapability {
        source: MaxwellMethodSource,
        metadata: &'static MaxwellEngineMethodMetadata,
        capability: MaxwellEngineCapability,
    },
    ResourceExhausted,
}

impl From<MaxwellMethodDispatchError> for MaxwellEngineDispatchError {
    fn from(error: MaxwellMethodDispatchError) -> Self {
        Self::Binding(error)
    }
}

impl Display for MaxwellEngineDispatchError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Binding(error) => write!(formatter, "class binding failed: {error}"),
            Self::UnsupportedClass {
                source,
                class,
                class_name,
            } => write!(
                formatter,
                "known Maxwell class has no implemented handler: {source} {class} class-name={class_name}"
            ),
            Self::UnknownClass { source, class } => write!(
                formatter,
                "unknown Maxwell class reached method dispatch: {source} {class}"
            ),
            Self::UnsupportedMethod { source, metadata } => write!(
                formatter,
                "known Maxwell method is not implemented: {source} class-name={} method-name={}",
                metadata.class_name, metadata.method_name
            ),
            Self::UnknownMethod { source, class_name } => write!(
                formatter,
                "unknown Maxwell class method: {source} class-name={class_name}"
            ),
            Self::InvalidMethodValue {
                source,
                metadata,
                defined_mask,
            } => write!(
                formatter,
                "Maxwell method argument sets bits outside its verified field mask: {source} class-name={} method-name={} defined-mask={defined_mask:#010x}",
                metadata.class_name, metadata.method_name
            ),
            Self::MissingCapability {
                source,
                metadata,
                capability,
            } => write!(
                formatter,
                "Maxwell method requires an unavailable execution capability: {source} class-name={} method-name={} capability={capability}",
                metadata.class_name, metadata.method_name
            ),
            Self::ResourceExhausted => {
                formatter.write_str("Maxwell engine dispatch exhausted host resources")
            }
        }
    }
}

impl std::error::Error for MaxwellEngineDispatchError {}

/// Validates binding and every engine method without mutating channel state.
pub fn preflight_maxwell_engine_packet(
    channel: &MaxwellGpuChannel,
    submission: FrontendSubmissionId,
    packet: &MaxwellDecodedPacket,
) -> Result<MaxwellEnginePacketDispatch, MaxwellEngineDispatchError> {
    let binding = preflight_maxwell_packet(channel, submission, packet)?;
    let mut methods = Vec::new();
    methods
        .try_reserve_exact(binding.methods().len())
        .map_err(|_| MaxwellEngineDispatchError::ResourceExhausted)?;

    for method in binding.methods() {
        if method.kind() == MaxwellMethodDispatchKind::ClassMethod {
            methods.push(preflight_class_method(channel, *method)?);
        }
    }

    Ok(MaxwellEnginePacketDispatch {
        binding,
        methods: methods.into_boxed_slice(),
    })
}

/// Commits a fully validated packet. Implemented T7-C effects are stateless.
pub fn commit_maxwell_engine_packet(
    channel: &mut MaxwellGpuChannel,
    dispatch: &MaxwellEnginePacketDispatch,
) -> Result<(), MaxwellEngineDispatchError> {
    commit_maxwell_packet(channel, &dispatch.binding)?;
    Ok(())
}

/// Preflights and atomically commits one packet.
pub fn dispatch_maxwell_engine_packet(
    channel: &mut MaxwellGpuChannel,
    submission: FrontendSubmissionId,
    packet: &MaxwellDecodedPacket,
) -> Result<MaxwellEnginePacketDispatch, MaxwellEngineDispatchError> {
    let dispatch = preflight_maxwell_engine_packet(channel, submission, packet)?;
    commit_maxwell_engine_packet(channel, &dispatch)?;
    Ok(dispatch)
}

/// Dispatches packets in order with per-packet atomicity.
pub fn dispatch_maxwell_engine_pushbuffer(
    channel: &mut MaxwellGpuChannel,
    submission: FrontendSubmissionId,
    pushbuffer: &MaxwellDecodedPushbuffer,
) -> Result<Box<[MaxwellEnginePacketDispatch]>, MaxwellEngineDispatchError> {
    let mut packets = Vec::new();
    packets
        .try_reserve_exact(pushbuffer.packets().len())
        .map_err(|_| MaxwellEngineDispatchError::ResourceExhausted)?;
    for packet in pushbuffer.packets() {
        packets.push(dispatch_maxwell_engine_packet(channel, submission, packet)?);
    }
    Ok(packets.into_boxed_slice())
}

fn preflight_class_method(
    channel: &MaxwellGpuChannel,
    method: MaxwellMethodDispatch,
) -> Result<MaxwellEngineMethodDispatch, MaxwellEngineDispatchError> {
    let classes = channel.profile().classes();
    let class = method.class();
    if class == three_d::CLASS {
        return three_d::preflight(method);
    }

    let class_name = if class == classes.compute() {
        Some("MAXWELL_COMPUTE_B")
    } else if class == classes.dma_copy() {
        Some("MAXWELL_DMA_COPY_A")
    } else if class == classes.inline_to_memory() {
        Some("MAXWELL_INLINE_TO_MEMORY_A")
    } else if class == classes.two_d() {
        Some("FERMI_TWOD_A")
    } else if class == classes.gpfifo() {
        Some("MAXWELL_CHANNEL_GPFIFO_A")
    } else {
        None
    };
    match class_name {
        Some(class_name) => Err(MaxwellEngineDispatchError::UnsupportedClass {
            source: method.source(),
            class,
            class_name,
        }),
        None => Err(MaxwellEngineDispatchError::UnknownClass {
            source: method.source(),
            class,
        }),
    }
}

#[cfg(test)]
mod tests {
    use nixe_gpu::{GpuVirtualAddress, MappingGeneration};

    use super::*;
    use crate::{
        MaxwellChannelId, MaxwellChannelOwner, MaxwellGpfifoSourceLocation, MaxwellMappingId,
        MaxwellPushbufferWord, SWITCH_1_GM20B_PROFILE, decode_maxwell_pushbuffer,
    };

    fn channel() -> MaxwellGpuChannel {
        MaxwellGpuChannel::new(
            MaxwellChannelId::new(7),
            MaxwellChannelOwner::new(1),
            SWITCH_1_GM20B_PROFILE,
        )
    }

    fn word(
        value: u32,
        index: u32,
    ) -> Result<MaxwellPushbufferWord, crate::MaxwellGpfifoSourceError> {
        Ok(MaxwellPushbufferWord::new(
            value,
            MaxwellGpfifoSourceLocation {
                channel: MaxwellChannelId::new(7),
                frontend: FrontendSubmissionId::new(3),
                entry_index: 0,
                pushbuffer: GpuVirtualAddress::try_new(0x8000, 40).unwrap(),
                word_offset: u64::from(index),
                mapping: MaxwellMappingId::new(2),
                generation: MappingGeneration::new(1),
            },
        ))
    }

    fn packet(method_dword: u32, argument: u32) -> MaxwellDecodedPushbuffer {
        packet_on_subchannel(0, method_dword, argument)
    }

    fn packet_on_subchannel(
        subchannel: u32,
        method_dword: u32,
        argument: u32,
    ) -> MaxwellDecodedPushbuffer {
        decode_maxwell_pushbuffer([
            word((1 << 29) | (1 << 16) | (subchannel << 13) | method_dword, 0),
            word(argument, 1),
        ])
        .unwrap()
    }

    fn bind_three_d(channel: &mut MaxwellGpuChannel) {
        let decoded = packet(0, three_d::CLASS.0);
        dispatch_maxwell_engine_packet(
            channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
    }

    #[test]
    fn three_d_no_operation_is_named_and_implemented() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        let decoded = packet(0x100 / 4, 0xfeed_beef);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();

        assert_eq!(dispatch.methods().len(), 1);
        assert_eq!(dispatch.methods()[0].metadata().class_name(), "MAXWELL_B");
        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "NO_OPERATION"
        );
        assert_eq!(
            dispatch.methods()[0].effect(),
            MaxwellEngineMethodEffect::NoOperation
        );
    }

    #[test]
    fn taxonomy_separates_unsupported_invalid_capability_and_unknown_methods() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        let cases = [
            (0x104, 0, "known Maxwell method is not implemented"),
            (
                0x124,
                4,
                "argument sets bits outside its verified field mask",
            ),
            (0x124, 3, "requires an unavailable execution capability"),
            (0x2ffc, 0, "unknown Maxwell class method"),
        ];
        for (method, argument, expected) in cases {
            let decoded = packet(method / 4, argument);
            let error = preflight_maxwell_engine_packet(
                &channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0],
            )
            .unwrap_err();
            assert!(error.to_string().contains(expected), "{error}");
        }
    }

    #[test]
    fn known_unimplemented_class_is_distinct_from_method_coverage() {
        let mut channel = channel();
        let compute = channel.profile().classes().compute();
        let bind = packet_on_subchannel(1, 0, compute.0);
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &bind.packets()[0],
        )
        .unwrap();
        let method = packet_on_subchannel(1, 0x100 / 4, 0);
        let error = preflight_maxwell_engine_packet(
            &channel,
            FrontendSubmissionId::new(3),
            &method.packets()[0],
        )
        .unwrap_err();
        assert!(matches!(
            error,
            MaxwellEngineDispatchError::UnsupportedClass {
                class_name: "MAXWELL_COMPUTE_B",
                ..
            }
        ));
    }
}
