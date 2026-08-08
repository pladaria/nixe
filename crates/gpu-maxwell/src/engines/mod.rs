//! Declarative Maxwell class-method routing.
//!
//! This layer validates every class method in a decoded packet before the
//! packet's binding state is committed. It contains no Horizon ABI, guest
//! mappings, scheduler state, or host-backend objects.

mod threed;
mod twod;

pub use threed::{
    MAXWELL_BIND_GROUP_COUNT, MAXWELL_COLOR_TARGET_COUNT, MAXWELL_CONSTANT_BUFFER_SLOT_COUNT,
    MAXWELL_PIPELINE_SHADER_COUNT, MAXWELL_SCISSOR_COUNT,
    MAXWELL_THREE_D_SM_TIMEOUT_COUNTER_BIT_MAX, MAXWELL_VERTEX_ATTRIBUTE_COUNT,
    MAXWELL_VERTEX_STREAM_COUNT, MAXWELL_VIEWPORT_COUNT, MaxwellThreeDAliasedLineWidthEnable,
    MaxwellThreeDAttachmentReadiness, MaxwellThreeDBegin, MaxwellThreeDBindGroupState,
    MaxwellThreeDBlendEnableCommon, MaxwellThreeDBlendFactor, MaxwellThreeDBlendOp,
    MaxwellThreeDClearState, MaxwellThreeDClearSurface, MaxwellThreeDColorCompressionMode,
    MaxwellThreeDColorMask, MaxwellThreeDColorTargetFormat, MaxwellThreeDColorTargetSelection,
    MaxwellThreeDColorTargetState, MaxwellThreeDCompareOp, MaxwellThreeDConstantBufferBinding,
    MaxwellThreeDConstantBufferSelectorState, MaxwellThreeDCoverageState,
    MaxwellThreeDCoverageStateWrite, MaxwellThreeDCsaaEnable, MaxwellThreeDCullFace,
    MaxwellThreeDDepthStencilFormat, MaxwellThreeDDepthStencilTargetState,
    MaxwellThreeDDescriptorPoolState, MaxwellThreeDDirtySubresource,
    MaxwellThreeDDirtySubresources, MaxwellThreeDFixedFunctionRegister,
    MaxwellThreeDFixedFunctionState, MaxwellThreeDFixedFunctionValue,
    MaxwellThreeDFixedFunctionWrite, MaxwellThreeDFrontFace, MaxwellThreeDImageKind,
    MaxwellThreeDImageLayout, MaxwellThreeDIndexBufferState, MaxwellThreeDIndexElementSize,
    MaxwellThreeDLineState, MaxwellThreeDLineStateWrite, MaxwellThreeDLoweredWork,
    MaxwellThreeDLoweringCache, MaxwellThreeDLoweringError, MaxwellThreeDLoweringPlan,
    MaxwellThreeDMappingReference, MaxwellThreeDOperationTrigger,
    MaxwellThreeDPipelineBindingState, MaxwellThreeDPointSize, MaxwellThreeDPolygonMode,
    MaxwellThreeDPreservedImageLayout, MaxwellThreeDPrimitiveState, MaxwellThreeDPrimitiveTopology,
    MaxwellThreeDRasterState, MaxwellThreeDRawValue, MaxwellThreeDRectangle, MaxwellThreeDRegister,
    MaxwellThreeDRegisterOrigin, MaxwellThreeDRenderEnableMode, MaxwellThreeDRenderEnableState,
    MaxwellThreeDRenderEnableStateWrite, MaxwellThreeDRenderTargetState,
    MaxwellThreeDRenderTargetWrite, MaxwellThreeDResolvedBuffer, MaxwellThreeDResolvedImage,
    MaxwellThreeDResolvedResource, MaxwellThreeDResolvedResources, MaxwellThreeDResourceAccess,
    MaxwellThreeDResourceAlias, MaxwellThreeDResourceError, MaxwellThreeDResourceRole,
    MaxwellThreeDSampleMode, MaxwellThreeDSamplerBindingMode, MaxwellThreeDScissorState,
    MaxwellThreeDShadeMode, MaxwellThreeDShaderBindingState, MaxwellThreeDShaderBindingWrite,
    MaxwellThreeDShaderExecutionState, MaxwellThreeDShaderExecutionStateWrite,
    MaxwellThreeDShaderResourceUse, MaxwellThreeDShaderStage, MaxwellThreeDSmTimeoutCounterBit,
    MaxwellThreeDState, MaxwellThreeDStateWrite, MaxwellThreeDStencilOp,
    MaxwellThreeDTranslatedShader, MaxwellThreeDTranslatedShaders, MaxwellThreeDUnresolvedAddress,
    MaxwellThreeDVertexArrayPrimitiveRestartEnable, MaxwellThreeDVertexAttributeFormat,
    MaxwellThreeDVertexComponentWidths, MaxwellThreeDVertexInputState,
    MaxwellThreeDVertexInputWrite, MaxwellThreeDVertexNumericalType,
    MaxwellThreeDVertexStreamFormat, MaxwellThreeDVertexStreamState,
    MaxwellThreeDViewportClipControl, MaxwellThreeDViewportState,
    MaxwellThreeDViewportTransformState, MaxwellThreeDViewportZClipRange,
    MaxwellThreeDVisibleCallLimit, MaxwellThreeDZCompressionMode, MaxwellThreeDZCullState,
    MaxwellThreeDZCullStateWrite, MaxwellThreeDZCullStatsEnable,
    preflight_maxwell_three_d_operation, resolve_maxwell_three_d_resources,
};
pub use twod::{
    MAXWELL_TWO_D_CORRAL_SIZE_MAX, MAXWELL_TWO_D_NOTIFY_ADDRESS_UPPER_MAX, MaxwellTwoDClipEnable,
    MaxwellTwoDColorKeyEnable, MaxwellTwoDNotifyAddressLower, MaxwellTwoDNotifyAddressUpper,
    MaxwellTwoDNotifyState, MaxwellTwoDNotifyStateWrite, MaxwellTwoDOperation,
    MaxwellTwoDPixelsFromMemoryCorralSize, MaxwellTwoDPixelsFromMemorySafeOverlap,
    MaxwellTwoDPixelsFromMemoryState, MaxwellTwoDPixelsFromMemoryStateWrite,
    MaxwellTwoDProcessingClusters, MaxwellTwoDRegister, MaxwellTwoDRegisterOrigin,
    MaxwellTwoDRenderEnableMode, MaxwellTwoDRenderEnableState, MaxwellTwoDRenderEnableStateWrite,
    MaxwellTwoDState, MaxwellTwoDStateWrite,
};

use std::fmt::{Display, Formatter};

use nixe_gpu::{FrontendSubmissionId, GpuClassId, GpuMethodId};

use crate::{
    MaxwellDecodedPacket, MaxwellDecodedPushbuffer, MaxwellGpuChannel, MaxwellMethodDispatch,
    MaxwellMethodDispatchError, MaxwellMethodDispatchKind, MaxwellMethodSource,
    MaxwellPacketDispatch, preflight_maxwell_packet,
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
    TwoDState(MaxwellTwoDStateWrite),
    ThreeDState(MaxwellThreeDStateWrite),
    ThreeDTrigger(MaxwellThreeDOperationTrigger),
    ThreeDStateAndTrigger {
        state: MaxwellThreeDStateWrite,
        trigger: MaxwellThreeDOperationTrigger,
    },
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
    two_d_before: MaxwellTwoDState,
    two_d_after: MaxwellTwoDState,
    three_d_before: MaxwellThreeDState,
    three_d_after: MaxwellThreeDState,
    methods: Box<[MaxwellEngineMethodDispatch]>,
    operations: Box<[MaxwellThreeDTriggeredOperation]>,
}

/// One execution trigger paired with the exact candidate state at that method.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaxwellThreeDTriggeredOperation {
    trigger: MaxwellThreeDOperationTrigger,
    state: MaxwellThreeDState,
}

impl MaxwellThreeDTriggeredOperation {
    #[must_use]
    pub const fn trigger(&self) -> MaxwellThreeDOperationTrigger {
        self.trigger
    }

    #[must_use]
    pub const fn state(&self) -> &MaxwellThreeDState {
        &self.state
    }
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

    #[must_use]
    pub const fn two_d_before(&self) -> &MaxwellTwoDState {
        &self.two_d_before
    }

    #[must_use]
    pub const fn two_d_after(&self) -> &MaxwellTwoDState {
        &self.two_d_after
    }

    #[must_use]
    pub const fn three_d_before(&self) -> &MaxwellThreeDState {
        &self.three_d_before
    }

    #[must_use]
    pub const fn three_d_after(&self) -> &MaxwellThreeDState {
        &self.three_d_after
    }

    #[must_use]
    pub fn operations(&self) -> &[MaxwellThreeDTriggeredOperation] {
        &self.operations
    }
}

/// Typed class-dispatch boundary. Missing coverage is never a guest result.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MaxwellEngineDispatchError {
    Binding(MaxwellMethodDispatchError),
    EngineStateChanged {
        channel: crate::MaxwellChannelId,
    },
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
    InvalidMethodEncoding {
        source: MaxwellMethodSource,
        method_name: &'static str,
        reason: &'static str,
    },
    ContradictoryState {
        source: Option<MaxwellMethodSource>,
        reason: &'static str,
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
            Self::EngineStateChanged { channel } => write!(
                formatter,
                "Maxwell channel engine state changed after packet preflight: {channel}"
            ),
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
            Self::InvalidMethodEncoding {
                source,
                method_name,
                reason,
            } => write!(
                formatter,
                "Maxwell method has an invalid verified encoding: {source} class-name=MAXWELL_B method-name={method_name} reason={reason}"
            ),
            Self::ContradictoryState { source, reason } => {
                write!(formatter, "contradictory Maxwell 3D state")?;
                if let Some(source) = source {
                    write!(formatter, ": {source}")?;
                }
                write!(formatter, " reason={reason}")
            }
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
    let two_d_before = channel.two_d().clone();
    let mut two_d_after = two_d_before.clone();
    let three_d_before = channel.three_d().clone();
    let mut three_d_after = three_d_before.clone();
    let mut methods = Vec::new();
    let mut operations = Vec::new();
    methods
        .try_reserve_exact(binding.methods().len())
        .map_err(|_| MaxwellEngineDispatchError::ResourceExhausted)?;

    for method in binding.methods() {
        if method.kind() == MaxwellMethodDispatchKind::ClassMethod {
            let method =
                preflight_class_method(channel, *method, &mut two_d_after, &mut three_d_after)?;
            let trigger = match method.effect() {
                MaxwellEngineMethodEffect::ThreeDTrigger(trigger)
                | MaxwellEngineMethodEffect::ThreeDStateAndTrigger { trigger, .. } => Some(trigger),
                _ => None,
            };
            methods.push(method);
            if let Some(trigger) = trigger {
                operations
                    .try_reserve(1)
                    .map_err(|_| MaxwellEngineDispatchError::ResourceExhausted)?;
                operations.push(MaxwellThreeDTriggeredOperation {
                    trigger,
                    state: three_d_after.clone(),
                });
            }
        }
    }
    three_d_after.validate_cross_registers().map_err(|error| {
        MaxwellEngineDispatchError::ContradictoryState {
            source: error.source,
            reason: error.reason,
        }
    })?;

    Ok(MaxwellEnginePacketDispatch {
        binding,
        two_d_before,
        two_d_after,
        three_d_before,
        three_d_after,
        methods: methods.into_boxed_slice(),
        operations: operations.into_boxed_slice(),
    })
}

/// Commits binding and engine state together after revalidating both snapshots.
pub fn commit_maxwell_engine_packet(
    channel: &mut MaxwellGpuChannel,
    dispatch: &MaxwellEnginePacketDispatch,
) -> Result<(), MaxwellEngineDispatchError> {
    if channel.id() != dispatch.binding.channel()
        || channel.frontend() != dispatch.binding.frontend_before()
    {
        return Err(MaxwellMethodDispatchError::FrontendStateChanged {
            channel: channel.id(),
        }
        .into());
    }
    if channel.two_d() != &dispatch.two_d_before || channel.three_d() != &dispatch.three_d_before {
        return Err(MaxwellEngineDispatchError::EngineStateChanged {
            channel: channel.id(),
        });
    }
    channel.replace_frontend(dispatch.binding.frontend_after());
    channel.replace_two_d(dispatch.two_d_after.clone());
    channel.replace_three_d(dispatch.three_d_after.clone());
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
    two_d: &mut MaxwellTwoDState,
    three_d: &mut MaxwellThreeDState,
) -> Result<MaxwellEngineMethodDispatch, MaxwellEngineDispatchError> {
    let classes = channel.profile().classes();
    let class = method.class();
    if class == threed::CLASS {
        return threed::preflight(method, three_d);
    }
    if class == twod::CLASS {
        return twod::preflight(method, two_d);
    }

    let class_name = if class == classes.compute() {
        Some("MAXWELL_COMPUTE_B")
    } else if class == classes.dma_copy() {
        Some("MAXWELL_DMA_COPY_A")
    } else if class == classes.inline_to_memory() {
        Some("MAXWELL_INLINE_TO_MEMORY_A")
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
    use nixe_gpu::{
        BackendCapabilities, BackendFeatures, BackendLimits, GpuCommand, GpuVirtualAddress,
        ImageFormat, MappingGeneration, QueryKind, RenderPassOperation, SampleCount, ShaderId,
        ShaderStage,
    };
    use nixe_memory::{CanonicalAllocation, CanonicalBackingRange, MemoryPermissions};

    use super::*;
    use crate::{
        MaxwellAddressSpaceId, MaxwellAddressSpaceInitialization, MaxwellAllocationId,
        MaxwellChannelId, MaxwellChannelOwner, MaxwellGpfifoSourceLocation, MaxwellGpuAddressSpace,
        MaxwellGpuMapping, MaxwellMapRequest, MaxwellMappingId, MaxwellPushbufferWord,
        SWITCH_1_GM20B_PROFILE, decode_maxwell_pushbuffer,
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

    fn incrementing_packet(method_dword: u32, arguments: &[u32]) -> MaxwellDecodedPushbuffer {
        incrementing_packet_on_subchannel(0, method_dword, arguments)
    }

    fn incrementing_packet_on_subchannel(
        subchannel: u32,
        method_dword: u32,
        arguments: &[u32],
    ) -> MaxwellDecodedPushbuffer {
        let mut words = Vec::with_capacity(arguments.len() + 1);
        words.push(word(
            (1 << 29) | ((arguments.len() as u32) << 16) | (subchannel << 13) | method_dword,
            0,
        ));
        words.extend(
            arguments
                .iter()
                .enumerate()
                .map(|(index, argument)| word(*argument, index as u32 + 1)),
        );
        decode_maxwell_pushbuffer(words).unwrap()
    }

    fn bind_two_d(channel: &mut MaxwellGpuChannel) {
        let decoded = packet_on_subchannel(3, 0, twod::CLASS.0);
        dispatch_maxwell_engine_packet(
            channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
    }

    fn bind_three_d(channel: &mut MaxwellGpuChannel) {
        let decoded = packet(0, threed::CLASS.0);
        dispatch_maxwell_engine_packet(
            channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
    }

    fn program_three_d(channel: &mut MaxwellGpuChannel, method: u32, argument: u32) {
        let decoded = packet(method / 4, argument);
        dispatch_maxwell_engine_packet(
            channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
    }

    fn resource_address_space() -> MaxwellGpuAddressSpace {
        let mut address_space =
            MaxwellGpuAddressSpace::new(MaxwellAddressSpaceId::new(9), SWITCH_1_GM20B_PROFILE);
        address_space
            .initialize(MaxwellAddressSpaceInitialization::default())
            .unwrap();
        address_space
    }

    fn map_resource(
        address_space: &mut MaxwellGpuAddressSpace,
        backing: CanonicalBackingRange,
        allocation: u64,
        kind: u8,
    ) -> MaxwellGpuMapping {
        let size = backing.size();
        address_space
            .map(MaxwellMapRequest {
                allocation: MaxwellAllocationId::new(allocation),
                backing,
                backing_offset: 0,
                size,
                allocation_alignment: 0x1000,
                page_size: 0,
                kind,
                cacheable: true,
                permissions: MemoryPermissions::READ_WRITE,
                fixed_offset: None,
            })
            .unwrap()
    }

    fn lowering_capabilities(features: BackendFeatures) -> BackendCapabilities {
        BackendCapabilities::new(
            features,
            [ImageFormat::Rgba8Unorm, ImageFormat::Bgra8Unorm],
            [SampleCount::One],
            [ShaderStage::Vertex, ShaderStage::Fragment],
            std::iter::empty::<QueryKind>(),
            BackendLimits {
                max_color_attachments: 8,
                max_descriptor_bindings: 32,
                max_compute_workgroups: [1, 1, 1],
            },
        )
    }

    fn color_target_selection_raw(count: u8, targets: [u8; 8]) -> u32 {
        targets
            .into_iter()
            .enumerate()
            .fold(u32::from(count), |raw, (index, target)| {
                raw | (u32::from(target) << (4 + index * 3))
            })
    }

    fn program_color_target(
        channel: &mut MaxwellGpuChannel,
        target: u8,
        address: u64,
        format: u32,
    ) {
        let base = 0x0800 + u32::from(target) * 0x40;
        for (offset, argument) in [
            (0x00, (address >> 32) as u32),
            (0x04, address as u32),
            (0x08, 64),
            (0x0c, 32),
            (0x10, format),
            (0x14, 0),
            (0x18, 1),
            (0x1c, 0),
            (0x20, 0),
        ] {
            program_three_d(channel, base + offset, argument);
        }
    }

    fn program_basic_draw_state(channel: &mut MaxwellGpuChannel, vertex: u64) {
        for (method, argument) in [
            (0x1c00, 0x1010),
            (0x1c04, (vertex >> 32) as u32),
            (0x1c08, vertex as u32),
            (0x1f00, (vertex >> 32) as u32),
            (0x1f04, (vertex + 0xff) as u32),
            (0x1160, 0x3820_0000),
            (0x0d74, 0),
            (0x1618, 4),
            (0x1970, 4),
            (0x12e4, 0),
            (0x135c, 0),
            (0x2000, 0x11),
            (0x2010, 0),
            (0x2040, 0x51),
            (0x2050, 1),
            (0x15d0, 0),
        ] {
            program_three_d(channel, method, argument);
        }
    }

    fn translated_graphics_shaders() -> MaxwellThreeDTranslatedShaders {
        MaxwellThreeDTranslatedShaders::new(
            vec![
                MaxwellThreeDTranslatedShader::new(ShaderStage::Vertex, ShaderId::new(1), 7),
                MaxwellThreeDTranslatedShader::new(ShaderStage::Fragment, ShaderId::new(2), 9),
            ],
            Vec::new(),
        )
        .unwrap()
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
    fn two_d_notify_address_upper_is_bounded_state_without_notification_effects() {
        let mut channel = channel();
        bind_two_d(&mut channel);
        let render_enable_before = channel.two_d().render_enable().clone();
        let pixels_from_memory_before = channel.two_d().pixels_from_memory().clone();
        let three_d_before = channel.three_d().clone();
        assert_eq!(
            channel.two_d().notify().address_upper().origin(),
            MaxwellTwoDRegisterOrigin::Unset
        );
        assert_eq!(
            MaxwellTwoDNotifyAddressUpper::new(
                MAXWELL_TWO_D_NOTIFY_ADDRESS_UPPER_MAX.saturating_add(1)
            ),
            None
        );

        for argument in [0, MAXWELL_TWO_D_NOTIFY_ADDRESS_UPPER_MAX] {
            let decoded = packet_on_subchannel(3, 0x0104 / 4, argument);
            let dispatch = dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0],
            )
            .unwrap();
            let source = dispatch.methods()[0].method().source();
            let value = MaxwellTwoDNotifyAddressUpper::new(argument).unwrap();
            let register = channel.two_d().notify().address_upper();

            assert_eq!(dispatch.methods()[0].metadata().class(), twod::CLASS);
            assert_eq!(
                dispatch.methods()[0].metadata().method_name(),
                "SET_NOTIFY_A"
            );
            assert_eq!(
                dispatch.methods()[0].effect(),
                MaxwellEngineMethodEffect::TwoDState(MaxwellTwoDStateWrite::Notify(
                    MaxwellTwoDNotifyStateWrite::AddressUpper { value, source }
                ))
            );
            assert!(dispatch.operations().is_empty());
            assert_eq!(register.origin(), MaxwellTwoDRegisterOrigin::Programmed);
            assert_eq!(register.raw(), Some(argument));
            assert_eq!(register.value().copied(), Some(value));
            assert_eq!(register.source(), Some(source));
            assert_eq!(value.get(), argument);
            assert_eq!(channel.two_d().render_enable(), &render_enable_before);
            assert_eq!(
                channel.two_d().pixels_from_memory(),
                &pixels_from_memory_before
            );
            assert_eq!(channel.three_d(), &three_d_before);
        }
    }

    #[test]
    fn invalid_two_d_notify_address_upper_rejects_atomically() {
        let mut channel = channel();
        bind_two_d(&mut channel);

        for argument in [0x0200_0000, u32::MAX] {
            let frontend_before = channel.frontend();
            let two_d_before = channel.two_d().clone();
            let three_d_before = channel.three_d().clone();
            let decoded = packet_on_subchannel(3, 0x0104 / 4, argument);

            assert!(matches!(
                dispatch_maxwell_engine_packet(
                    &mut channel,
                    FrontendSubmissionId::new(3),
                    &decoded.packets()[0]
                ),
                Err(MaxwellEngineDispatchError::InvalidMethodValue {
                    source,
                    defined_mask: MAXWELL_TWO_D_NOTIFY_ADDRESS_UPPER_MAX,
                    ..
                }) if source.argument() == argument
            ));
            assert_eq!(channel.frontend(), frontend_before);
            assert_eq!(channel.two_d(), &two_d_before);
            assert_eq!(channel.three_d(), &three_d_before);
        }
    }

    #[test]
    fn two_d_notify_address_lower_accepts_its_complete_bit_domain() {
        let mut channel = channel();
        bind_two_d(&mut channel);
        let three_d_before = channel.three_d().clone();
        assert_eq!(
            channel.two_d().notify().address_lower().origin(),
            MaxwellTwoDRegisterOrigin::Unset
        );

        for argument in [0, 0x0820_2010, u32::MAX] {
            let decoded = packet_on_subchannel(3, 0x0108 / 4, argument);
            let dispatch = dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0],
            )
            .unwrap();
            let source = dispatch.methods()[0].method().source();
            let value = MaxwellTwoDNotifyAddressLower::new(argument);
            let register = channel.two_d().notify().address_lower();

            assert_eq!(
                dispatch.methods()[0].metadata().method_name(),
                "SET_NOTIFY_B"
            );
            assert_eq!(
                dispatch.methods()[0].effect(),
                MaxwellEngineMethodEffect::TwoDState(MaxwellTwoDStateWrite::Notify(
                    MaxwellTwoDNotifyStateWrite::AddressLower { value, source }
                ))
            );
            assert!(dispatch.operations().is_empty());
            assert_eq!(register.origin(), MaxwellTwoDRegisterOrigin::Programmed);
            assert_eq!(register.raw(), Some(argument));
            assert_eq!(register.value().copied(), Some(value));
            assert_eq!(register.source(), Some(source));
            assert_eq!(value.get(), argument);
            assert_eq!(
                channel.two_d().notify().address_upper().origin(),
                MaxwellTwoDRegisterOrigin::Unset
            );
            assert_eq!(channel.three_d(), &three_d_before);
        }
    }

    #[test]
    fn incrementing_notify_address_fragments_commit_together() {
        let mut channel = channel();
        bind_two_d(&mut channel);
        let decoded = incrementing_packet_on_subchannel(3, 0x0104 / 4, &[1, 0x0820_2010]);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();

        assert_eq!(dispatch.methods().len(), 2);
        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_NOTIFY_A"
        );
        assert_eq!(
            dispatch.methods()[1].metadata().method_name(),
            "SET_NOTIFY_B"
        );
        assert!(dispatch.operations().is_empty());
        assert_eq!(channel.two_d().notify().address_upper().raw(), Some(1));
        assert_eq!(
            channel.two_d().notify().address_lower().raw(),
            Some(0x0820_2010)
        );
        assert_eq!(
            channel.two_d().notify().address_upper().source(),
            Some(dispatch.methods()[0].method().source())
        );
        assert_eq!(
            channel.two_d().notify().address_lower().source(),
            Some(dispatch.methods()[1].method().source())
        );
    }

    #[test]
    fn unsupported_notify_trigger_discards_both_address_fragments() {
        let mut channel = channel();
        bind_two_d(&mut channel);
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = incrementing_packet_on_subchannel(3, 0x0104 / 4, &[1, 0x0820_2010, 0]);

        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::UnknownMethod {
                source,
                class_name: "FERMI_TWOD_A",
            }) if source.method() == GpuMethodId(0x010c)
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }

    #[test]
    fn two_d_processing_cluster_values_are_typed_and_retain_their_source() {
        let mut channel = channel();
        bind_two_d(&mut channel);
        let three_d_before = channel.three_d().clone();
        assert_eq!(
            channel.two_d().processing_clusters().origin(),
            MaxwellTwoDRegisterOrigin::Unset
        );

        for (argument, expected) in [
            (0, MaxwellTwoDProcessingClusters::All),
            (1, MaxwellTwoDProcessingClusters::One),
        ] {
            let decoded = packet_on_subchannel(3, 0x0260 / 4, argument);
            let dispatch = dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0],
            )
            .unwrap();
            let source = dispatch.methods()[0].method().source();
            let register = channel.two_d().processing_clusters();

            assert_eq!(dispatch.methods()[0].metadata().class(), twod::CLASS);
            assert_eq!(
                dispatch.methods()[0].metadata().method_name(),
                "SET_NUM_PROCESSING_CLUSTERS"
            );
            assert_eq!(
                dispatch.methods()[0].effect(),
                MaxwellEngineMethodEffect::TwoDState(MaxwellTwoDStateWrite::ProcessingClusters {
                    value: expected,
                    source,
                })
            );
            assert_eq!(register.origin(), MaxwellTwoDRegisterOrigin::Programmed);
            assert_eq!(register.raw(), Some(argument));
            assert_eq!(register.value().copied(), Some(expected));
            assert_eq!(register.source(), Some(source));
            assert_eq!(channel.three_d(), &three_d_before);
        }
    }

    #[test]
    fn two_d_render_enable_modes_are_typed_state_without_condition_evaluation() {
        let mut channel = channel();
        bind_two_d(&mut channel);
        let pixels_from_memory_before = channel.two_d().pixels_from_memory().clone();
        let three_d_before = channel.three_d().clone();
        assert_eq!(
            channel.two_d().render_enable().mode().origin(),
            MaxwellTwoDRegisterOrigin::Unset
        );

        for (argument, expected) in [
            (0, MaxwellTwoDRenderEnableMode::Disabled),
            (1, MaxwellTwoDRenderEnableMode::Enabled),
            (2, MaxwellTwoDRenderEnableMode::Conditional),
            (3, MaxwellTwoDRenderEnableMode::RenderIfEqual),
            (4, MaxwellTwoDRenderEnableMode::RenderIfNotEqual),
        ] {
            let decoded = packet_on_subchannel(3, 0x026c / 4, argument);
            let dispatch = dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0],
            )
            .unwrap();
            let source = dispatch.methods()[0].method().source();
            let register = channel.two_d().render_enable().mode();

            assert_eq!(dispatch.methods()[0].metadata().class(), twod::CLASS);
            assert_eq!(
                dispatch.methods()[0].metadata().method_name(),
                "SET_RENDER_ENABLE_C"
            );
            assert_eq!(
                dispatch.methods()[0].effect(),
                MaxwellEngineMethodEffect::TwoDState(MaxwellTwoDStateWrite::RenderEnable(
                    MaxwellTwoDRenderEnableStateWrite::Mode {
                        value: expected,
                        source,
                    }
                ))
            );
            assert!(dispatch.operations().is_empty());
            assert_eq!(register.origin(), MaxwellTwoDRegisterOrigin::Programmed);
            assert_eq!(register.raw(), Some(argument));
            assert_eq!(register.value().copied(), Some(expected));
            assert_eq!(register.source(), Some(source));
            assert_eq!(
                channel.two_d().pixels_from_memory(),
                &pixels_from_memory_before
            );
            assert_eq!(channel.three_d(), &three_d_before);
        }
    }

    #[test]
    fn invalid_two_d_render_enable_modes_are_rejected_atomically() {
        let mut channel = channel();
        bind_two_d(&mut channel);

        for argument in [5, 6, 7, 8, u32::MAX] {
            let frontend_before = channel.frontend();
            let two_d_before = channel.two_d().clone();
            let three_d_before = channel.three_d().clone();
            let decoded = packet_on_subchannel(3, 0x026c / 4, argument);

            assert!(matches!(
                dispatch_maxwell_engine_packet(
                    &mut channel,
                    FrontendSubmissionId::new(3),
                    &decoded.packets()[0]
                ),
                Err(MaxwellEngineDispatchError::InvalidMethodValue {
                    source,
                    defined_mask: 7,
                    ..
                }) if source.argument() == argument
            ));
            assert_eq!(channel.frontend(), frontend_before);
            assert_eq!(channel.two_d(), &two_d_before);
            assert_eq!(channel.three_d(), &three_d_before);
        }
    }

    #[test]
    fn unsupported_render_enable_address_methods_remain_typed_fatal_errors() {
        let mut channel = channel();
        bind_two_d(&mut channel);

        for method in [0x0264, 0x0268] {
            let frontend_before = channel.frontend();
            let two_d_before = channel.two_d().clone();
            let three_d_before = channel.three_d().clone();
            let decoded = packet_on_subchannel(3, method / 4, 0);

            assert!(matches!(
                dispatch_maxwell_engine_packet(
                    &mut channel,
                    FrontendSubmissionId::new(3),
                    &decoded.packets()[0]
                ),
                Err(MaxwellEngineDispatchError::UnknownMethod {
                    source,
                    class_name: "FERMI_TWOD_A",
                }) if source.method() == GpuMethodId(method)
            ));
            assert_eq!(channel.frontend(), frontend_before);
            assert_eq!(channel.two_d(), &two_d_before);
            assert_eq!(channel.three_d(), &three_d_before);
        }
    }

    #[test]
    fn two_d_operation_values_are_typed_state_without_execution() {
        let mut channel = channel();
        bind_two_d(&mut channel);
        let three_d_before = channel.three_d().clone();
        assert_eq!(
            channel.two_d().operation().origin(),
            MaxwellTwoDRegisterOrigin::Unset
        );

        for (argument, expected) in [
            (0, MaxwellTwoDOperation::SourceCopyAnd),
            (1, MaxwellTwoDOperation::RasterOperationAnd),
            (2, MaxwellTwoDOperation::BlendAnd),
            (3, MaxwellTwoDOperation::SourceCopy),
            (4, MaxwellTwoDOperation::RasterOperation),
            (5, MaxwellTwoDOperation::SourceCopyPremultiplied),
            (6, MaxwellTwoDOperation::BlendPremultiplied),
        ] {
            let decoded = packet_on_subchannel(3, 0x02ac / 4, argument);
            let dispatch = dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0],
            )
            .unwrap();
            let source = dispatch.methods()[0].method().source();
            let register = channel.two_d().operation();

            assert_eq!(dispatch.methods()[0].metadata().class(), twod::CLASS);
            assert_eq!(
                dispatch.methods()[0].metadata().method_name(),
                "SET_OPERATION"
            );
            assert_eq!(
                dispatch.methods()[0].effect(),
                MaxwellEngineMethodEffect::TwoDState(MaxwellTwoDStateWrite::Operation {
                    value: expected,
                    source,
                })
            );
            assert!(dispatch.operations().is_empty());
            assert_eq!(register.origin(), MaxwellTwoDRegisterOrigin::Programmed);
            assert_eq!(register.raw(), Some(argument));
            assert_eq!(register.value().copied(), Some(expected));
            assert_eq!(register.source(), Some(source));
            assert_eq!(channel.three_d(), &three_d_before);
        }
    }

    #[test]
    fn two_d_clip_enable_values_are_typed_state_without_execution() {
        let mut channel = channel();
        bind_two_d(&mut channel);
        let three_d_before = channel.three_d().clone();
        assert_eq!(
            channel.two_d().clip_enable().origin(),
            MaxwellTwoDRegisterOrigin::Unset
        );

        for (argument, expected) in [
            (0, MaxwellTwoDClipEnable::Disabled),
            (1, MaxwellTwoDClipEnable::Enabled),
        ] {
            let decoded = packet_on_subchannel(3, 0x0290 / 4, argument);
            let dispatch = dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0],
            )
            .unwrap();
            let source = dispatch.methods()[0].method().source();
            let register = channel.two_d().clip_enable();

            assert_eq!(dispatch.methods()[0].metadata().class(), twod::CLASS);
            assert_eq!(
                dispatch.methods()[0].metadata().method_name(),
                "SET_CLIP_ENABLE"
            );
            assert_eq!(
                dispatch.methods()[0].effect(),
                MaxwellEngineMethodEffect::TwoDState(MaxwellTwoDStateWrite::ClipEnable {
                    value: expected,
                    source,
                })
            );
            assert!(dispatch.operations().is_empty());
            assert_eq!(register.origin(), MaxwellTwoDRegisterOrigin::Programmed);
            assert_eq!(register.raw(), Some(argument));
            assert_eq!(register.value().copied(), Some(expected));
            assert_eq!(register.source(), Some(source));
            assert_eq!(channel.three_d(), &three_d_before);
        }
    }

    #[test]
    fn invalid_two_d_clip_enable_rejects_without_mutating_channel_state() {
        let mut channel = channel();
        bind_two_d(&mut channel);
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet_on_subchannel(3, 0x0290 / 4, 2);

        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::InvalidMethodValue {
                source,
                defined_mask: 1,
                ..
            }) if source.argument() == 2
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }

    #[test]
    fn two_d_color_key_enable_values_are_typed_state_without_execution() {
        let mut channel = channel();
        bind_two_d(&mut channel);
        let three_d_before = channel.three_d().clone();
        assert_eq!(
            channel.two_d().color_key_enable().origin(),
            MaxwellTwoDRegisterOrigin::Unset
        );
        assert_eq!(
            channel.two_d().clip_enable().origin(),
            MaxwellTwoDRegisterOrigin::Unset
        );

        for (argument, expected) in [
            (0, MaxwellTwoDColorKeyEnable::Disabled),
            (1, MaxwellTwoDColorKeyEnable::Enabled),
        ] {
            let decoded = packet_on_subchannel(3, 0x029c / 4, argument);
            let dispatch = dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0],
            )
            .unwrap();
            let source = dispatch.methods()[0].method().source();
            let register = channel.two_d().color_key_enable();

            assert_eq!(dispatch.methods()[0].metadata().class(), twod::CLASS);
            assert_eq!(
                dispatch.methods()[0].metadata().method_name(),
                "SET_COLOR_KEY_ENABLE"
            );
            assert_eq!(
                dispatch.methods()[0].effect(),
                MaxwellEngineMethodEffect::TwoDState(MaxwellTwoDStateWrite::ColorKeyEnable {
                    value: expected,
                    source,
                })
            );
            assert!(dispatch.operations().is_empty());
            assert_eq!(register.origin(), MaxwellTwoDRegisterOrigin::Programmed);
            assert_eq!(register.raw(), Some(argument));
            assert_eq!(register.value().copied(), Some(expected));
            assert_eq!(register.source(), Some(source));
            assert_eq!(
                channel.two_d().clip_enable().origin(),
                MaxwellTwoDRegisterOrigin::Unset
            );
            assert_eq!(channel.three_d(), &three_d_before);
        }
    }

    #[test]
    fn invalid_two_d_color_key_enable_rejects_without_mutating_channel_state() {
        let mut channel = channel();
        bind_two_d(&mut channel);
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet_on_subchannel(3, 0x029c / 4, 2);

        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::InvalidMethodValue {
                source,
                defined_mask: 1,
                ..
            }) if source.argument() == 2
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }

    #[test]
    fn unsupported_method_after_color_key_enable_discards_the_packet_prefix() {
        let mut channel = channel();
        bind_two_d(&mut channel);
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = incrementing_packet_on_subchannel(3, 0x029c / 4, &[1, 0]);

        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::UnknownMethod {
                source,
                class_name: "FERMI_TWOD_A",
            }) if source.method() == GpuMethodId(0x02a0)
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }

    #[test]
    fn two_d_corral_size_is_bounded_source_preserving_state_without_execution() {
        let mut channel = channel();
        bind_two_d(&mut channel);
        let state_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        assert_eq!(
            channel.two_d().pixels_from_memory().corral_size().origin(),
            MaxwellTwoDRegisterOrigin::Unset
        );

        for argument in [0, 0x3f, u32::from(MAXWELL_TWO_D_CORRAL_SIZE_MAX)] {
            let decoded = packet_on_subchannel(3, 0x0884 / 4, argument);
            let dispatch = dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0],
            )
            .unwrap();
            let source = dispatch.methods()[0].method().source();
            let value = MaxwellTwoDPixelsFromMemoryCorralSize::new(argument as u16).unwrap();
            let register = channel.two_d().pixels_from_memory().corral_size();

            assert_eq!(dispatch.methods()[0].metadata().class(), twod::CLASS);
            assert_eq!(
                dispatch.methods()[0].metadata().method_name(),
                "SET_PIXELS_FROM_MEMORY_CORRAL_SIZE"
            );
            assert_eq!(
                dispatch.methods()[0].effect(),
                MaxwellEngineMethodEffect::TwoDState(MaxwellTwoDStateWrite::PixelsFromMemory(
                    MaxwellTwoDPixelsFromMemoryStateWrite::CorralSize { value, source }
                ))
            );
            assert!(dispatch.operations().is_empty());
            assert_eq!(register.origin(), MaxwellTwoDRegisterOrigin::Programmed);
            assert_eq!(register.raw(), Some(argument));
            assert_eq!(register.value().copied(), Some(value));
            assert_eq!(register.source(), Some(source));
            assert_eq!(value.get(), argument as u16);
            assert_eq!(channel.two_d().clip_enable(), state_before.clip_enable());
            assert_eq!(
                channel.two_d().color_key_enable(),
                state_before.color_key_enable()
            );
            assert_eq!(channel.three_d(), &three_d_before);
        }
    }

    #[test]
    fn invalid_two_d_corral_size_rejects_without_mutating_channel_state() {
        let mut channel = channel();
        bind_two_d(&mut channel);

        for argument in [0x0400, u32::MAX] {
            let frontend_before = channel.frontend();
            let two_d_before = channel.two_d().clone();
            let three_d_before = channel.three_d().clone();
            let decoded = packet_on_subchannel(3, 0x0884 / 4, argument);

            assert!(matches!(
                dispatch_maxwell_engine_packet(
                    &mut channel,
                    FrontendSubmissionId::new(3),
                    &decoded.packets()[0]
                ),
                Err(MaxwellEngineDispatchError::InvalidMethodValue {
                    source,
                    defined_mask: 0x03ff,
                    ..
                }) if source.argument() == argument
            ));
            assert_eq!(channel.frontend(), frontend_before);
            assert_eq!(channel.two_d(), &two_d_before);
            assert_eq!(channel.three_d(), &three_d_before);
        }
    }

    #[test]
    fn two_d_safe_overlap_values_are_typed_state_without_execution() {
        let mut channel = channel();
        bind_two_d(&mut channel);
        let three_d_before = channel.three_d().clone();
        assert_eq!(
            channel.two_d().pixels_from_memory().safe_overlap().origin(),
            MaxwellTwoDRegisterOrigin::Unset
        );

        for (argument, expected) in [
            (0, MaxwellTwoDPixelsFromMemorySafeOverlap::Disabled),
            (1, MaxwellTwoDPixelsFromMemorySafeOverlap::Enabled),
        ] {
            let decoded = packet_on_subchannel(3, 0x0888 / 4, argument);
            let dispatch = dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0],
            )
            .unwrap();
            let source = dispatch.methods()[0].method().source();
            let register = channel.two_d().pixels_from_memory().safe_overlap();

            assert_eq!(dispatch.methods()[0].metadata().class(), twod::CLASS);
            assert_eq!(
                dispatch.methods()[0].metadata().method_name(),
                "SET_PIXELS_FROM_MEMORY_SAFE_OVERLAP"
            );
            assert_eq!(
                dispatch.methods()[0].effect(),
                MaxwellEngineMethodEffect::TwoDState(MaxwellTwoDStateWrite::PixelsFromMemory(
                    MaxwellTwoDPixelsFromMemoryStateWrite::SafeOverlap {
                        value: expected,
                        source,
                    }
                ))
            );
            assert!(dispatch.operations().is_empty());
            assert_eq!(register.origin(), MaxwellTwoDRegisterOrigin::Programmed);
            assert_eq!(register.raw(), Some(argument));
            assert_eq!(register.value().copied(), Some(expected));
            assert_eq!(register.source(), Some(source));
            assert_eq!(
                channel.two_d().pixels_from_memory().corral_size().origin(),
                MaxwellTwoDRegisterOrigin::Unset
            );
            assert_eq!(channel.three_d(), &three_d_before);
        }
    }

    #[test]
    fn invalid_two_d_safe_overlap_rejects_without_mutating_channel_state() {
        let mut channel = channel();
        bind_two_d(&mut channel);

        for argument in [2, u32::MAX] {
            let frontend_before = channel.frontend();
            let two_d_before = channel.two_d().clone();
            let three_d_before = channel.three_d().clone();
            let decoded = packet_on_subchannel(3, 0x0888 / 4, argument);

            assert!(matches!(
                dispatch_maxwell_engine_packet(
                    &mut channel,
                    FrontendSubmissionId::new(3),
                    &decoded.packets()[0]
                ),
                Err(MaxwellEngineDispatchError::InvalidMethodValue {
                    source,
                    defined_mask: 1,
                    ..
                }) if source.argument() == argument
            ));
            assert_eq!(channel.frontend(), frontend_before);
            assert_eq!(channel.two_d(), &two_d_before);
            assert_eq!(channel.three_d(), &three_d_before);
        }
    }

    #[test]
    fn unsupported_pixels_from_memory_method_remains_a_typed_fatal_error() {
        let mut channel = channel();
        bind_two_d(&mut channel);
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet_on_subchannel(3, 0x0880 / 4, 0);

        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::UnknownMethod {
                source,
                class_name: "FERMI_TWOD_A",
            }) if source.method() == GpuMethodId(0x0880)
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }

    #[test]
    fn invalid_two_d_operation_values_are_rejected_atomically() {
        let mut channel = channel();
        bind_two_d(&mut channel);

        for argument in [7, 8] {
            let frontend_before = channel.frontend();
            let two_d_before = channel.two_d().clone();
            let three_d_before = channel.three_d().clone();
            let decoded = packet_on_subchannel(3, 0x02ac / 4, argument);

            assert!(matches!(
                dispatch_maxwell_engine_packet(
                    &mut channel,
                    FrontendSubmissionId::new(3),
                    &decoded.packets()[0]
                ),
                Err(MaxwellEngineDispatchError::InvalidMethodValue {
                    source,
                    defined_mask: 7,
                    ..
                }) if source.argument() == argument
            ));
            assert_eq!(channel.frontend(), frontend_before);
            assert_eq!(channel.two_d(), &two_d_before);
            assert_eq!(channel.three_d(), &three_d_before);
        }
    }

    #[test]
    fn unsupported_method_after_two_d_operation_discards_the_packet_prefix() {
        let mut channel = channel();
        bind_two_d(&mut channel);
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = incrementing_packet_on_subchannel(3, 0x02ac / 4, &[3, 0]);

        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::UnknownMethod {
                source,
                class_name: "FERMI_TWOD_A",
            }) if source.method() == GpuMethodId(0x02b0)
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }

    #[test]
    fn invalid_two_d_value_rejects_without_mutating_any_channel_state() {
        let mut channel = channel();
        bind_two_d(&mut channel);
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet_on_subchannel(3, 0x0260 / 4, 2);

        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::InvalidMethodValue {
                source,
                defined_mask: 1,
                ..
            }) if source.argument() == 2
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }

    #[test]
    fn unsupported_two_d_suffix_discards_the_valid_packet_prefix() {
        let mut channel = channel();
        bind_two_d(&mut channel);
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = incrementing_packet_on_subchannel(3, 0x0260 / 4, &[1, 0]);

        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::UnknownMethod {
                source,
                class_name: "FERMI_TWOD_A",
            }) if source.method() == GpuMethodId(0x0264)
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }

    #[test]
    fn commit_rejects_intervening_two_d_state_without_partial_publish() {
        let mut channel = channel();
        bind_two_d(&mut channel);
        let first = packet_on_subchannel(3, 0x0260 / 4, 0);
        let prepared = preflight_maxwell_engine_packet(
            &channel,
            FrontendSubmissionId::new(3),
            &first.packets()[0],
        )
        .unwrap();
        let intervening = packet_on_subchannel(3, 0x0260 / 4, 1);
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &intervening.packets()[0],
        )
        .unwrap();
        let committed_two_d = channel.two_d().clone();
        let committed_three_d = channel.three_d().clone();

        assert!(matches!(
            commit_maxwell_engine_packet(&mut channel, &prepared),
            Err(MaxwellEngineDispatchError::EngineStateChanged { .. })
        ));
        assert_eq!(channel.two_d(), &committed_two_d);
        assert_eq!(channel.three_d(), &committed_three_d);
    }

    #[test]
    fn three_d_render_enable_modes_are_typed_and_engine_owned() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        let two_d_before = channel.two_d().clone();
        assert_eq!(
            channel.three_d().render_enable().mode().origin(),
            MaxwellThreeDRegisterOrigin::Unset
        );

        for (argument, expected) in [
            (0, MaxwellThreeDRenderEnableMode::Disabled),
            (1, MaxwellThreeDRenderEnableMode::Enabled),
            (2, MaxwellThreeDRenderEnableMode::Conditional),
            (3, MaxwellThreeDRenderEnableMode::RenderIfEqual),
            (4, MaxwellThreeDRenderEnableMode::RenderIfNotEqual),
        ] {
            let decoded = packet(0x1558 / 4, argument);
            let dispatch = dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0],
            )
            .unwrap();
            let source = dispatch.methods()[0].method().source();
            let register = channel.three_d().render_enable().mode();

            assert_eq!(dispatch.methods()[0].metadata().class(), threed::CLASS);
            assert_eq!(
                dispatch.methods()[0].metadata().method_name(),
                "SET_RENDER_ENABLE_C"
            );
            assert_eq!(
                dispatch.methods()[0].effect(),
                MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::RenderEnable(
                    MaxwellThreeDRenderEnableStateWrite::Mode {
                        value: expected,
                        source,
                    }
                ))
            );
            assert!(dispatch.operations().is_empty());
            assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
            assert_eq!(register.raw(), Some(argument));
            assert_eq!(register.value().copied(), Some(expected));
            assert_eq!(register.source(), Some(source));
            assert_eq!(channel.two_d(), &two_d_before);
        }
    }

    #[test]
    fn invalid_three_d_render_enable_modes_are_rejected_atomically() {
        let mut channel = channel();
        bind_three_d(&mut channel);

        for argument in [5, 6, 7, 8, u32::MAX] {
            let frontend_before = channel.frontend();
            let two_d_before = channel.two_d().clone();
            let three_d_before = channel.three_d().clone();
            let decoded = packet(0x1558 / 4, argument);

            assert!(matches!(
                dispatch_maxwell_engine_packet(
                    &mut channel,
                    FrontendSubmissionId::new(3),
                    &decoded.packets()[0]
                ),
                Err(MaxwellEngineDispatchError::InvalidMethodValue {
                    source,
                    defined_mask: 7,
                    ..
                }) if source.argument() == argument
            ));
            assert_eq!(channel.frontend(), frontend_before);
            assert_eq!(channel.two_d(), &two_d_before);
            assert_eq!(channel.three_d(), &three_d_before);
        }
    }

    #[test]
    fn unsupported_three_d_render_enable_address_methods_remain_typed_fatal_errors() {
        let mut channel = channel();
        bind_three_d(&mut channel);

        for method in [0x1550, 0x1554] {
            let frontend_before = channel.frontend();
            let three_d_before = channel.three_d().clone();
            let decoded = packet(method / 4, 0);

            assert!(matches!(
                dispatch_maxwell_engine_packet(
                    &mut channel,
                    FrontendSubmissionId::new(3),
                    &decoded.packets()[0]
                ),
                Err(MaxwellEngineDispatchError::UnknownMethod {
                    source,
                    class_name: "MAXWELL_B",
                }) if source.method() == GpuMethodId(method)
            ));
            assert_eq!(channel.frontend(), frontend_before);
            assert_eq!(channel.three_d(), &three_d_before);
        }
    }

    #[test]
    fn non_enabled_three_d_render_modes_stop_before_neutral_lowering() {
        for (argument, expected) in [
            (0, MaxwellThreeDRenderEnableMode::Disabled),
            (2, MaxwellThreeDRenderEnableMode::Conditional),
            (3, MaxwellThreeDRenderEnableMode::RenderIfEqual),
            (4, MaxwellThreeDRenderEnableMode::RenderIfNotEqual),
        ] {
            let mut channel = channel();
            bind_three_d(&mut channel);
            let decoded = packet(0x1558 / 4, argument);
            let dispatch = dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0],
            )
            .unwrap();
            let resources =
                resolve_maxwell_three_d_resources(channel.three_d(), &resource_address_space())
                    .unwrap();
            let cache = MaxwellThreeDLoweringCache::default();
            let cache_before = cache.clone();

            assert!(matches!(
                preflight_maxwell_three_d_operation(
                    channel.three_d(),
                    &resources,
                    MaxwellThreeDOperationTrigger::ClearSurface {
                        source: dispatch.methods()[0].method().source(),
                    },
                    None,
                    FrontendSubmissionId::new(10),
                    Vec::new(),
                    &lowering_capabilities(BackendFeatures::empty()),
                    &cache,
                ),
                Err(MaxwellThreeDLoweringError::UnsupportedRenderEnableMode(mode))
                    if mode == expected
            ));
            assert_eq!(cache, cache_before);
        }
    }

    #[test]
    fn visible_call_limit_is_typed_source_preserving_execution_state() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        let two_d_before = channel.two_d().clone();
        let timeout_before = channel
            .three_d()
            .shader_execution()
            .sm_timeout_counter_bit()
            .to_owned();
        assert_eq!(
            channel
                .three_d()
                .shader_execution()
                .visible_call_limit()
                .origin(),
            MaxwellThreeDRegisterOrigin::Unset
        );

        for (argument, expected, limit) in [
            (0, MaxwellThreeDVisibleCallLimit::Calls0, Some(0)),
            (1, MaxwellThreeDVisibleCallLimit::Calls1, Some(1)),
            (2, MaxwellThreeDVisibleCallLimit::Calls2, Some(2)),
            (3, MaxwellThreeDVisibleCallLimit::Calls4, Some(4)),
            (4, MaxwellThreeDVisibleCallLimit::Calls8, Some(8)),
            (5, MaxwellThreeDVisibleCallLimit::Calls16, Some(16)),
            (6, MaxwellThreeDVisibleCallLimit::Calls32, Some(32)),
            (7, MaxwellThreeDVisibleCallLimit::Calls64, Some(64)),
            (8, MaxwellThreeDVisibleCallLimit::Calls128, Some(128)),
            (15, MaxwellThreeDVisibleCallLimit::NoCheck, None),
        ] {
            let decoded = packet(0x0d64 / 4, argument);
            let dispatch = dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0],
            )
            .unwrap();
            let source = dispatch.methods()[0].method().source();
            let register = channel.three_d().shader_execution().visible_call_limit();

            assert_eq!(
                dispatch.methods()[0].metadata().method_name(),
                "SET_API_VISIBLE_CALL_LIMIT"
            );
            assert_eq!(
                dispatch.methods()[0].effect(),
                MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::ShaderExecution(
                    MaxwellThreeDShaderExecutionStateWrite::VisibleCallLimit {
                        value: expected,
                        source,
                    }
                ))
            );
            assert!(dispatch.operations().is_empty());
            assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
            assert_eq!(register.raw(), Some(argument));
            assert_eq!(register.value().copied(), Some(expected));
            assert_eq!(register.source(), Some(source));
            assert_eq!(expected.raw(), argument);
            assert_eq!(expected.limit(), limit);
            assert_eq!(
                channel
                    .three_d()
                    .shader_execution()
                    .sm_timeout_counter_bit(),
                &timeout_before
            );
            assert_eq!(channel.two_d(), &two_d_before);
        }
    }

    #[test]
    fn invalid_visible_call_limits_and_packet_suffix_are_rejected_atomically() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        program_three_d(&mut channel, 0x0d64, 8);

        for argument in [9, 10, 11, 12, 13, 14, 0x10, 0x8000_0000, u32::MAX] {
            let frontend_before = channel.frontend();
            let two_d_before = channel.two_d().clone();
            let three_d_before = channel.three_d().clone();
            let decoded = packet(0x0d64 / 4, argument);

            assert!(matches!(
                dispatch_maxwell_engine_packet(
                    &mut channel,
                    FrontendSubmissionId::new(3),
                    &decoded.packets()[0]
                ),
                Err(MaxwellEngineDispatchError::InvalidMethodValue {
                    source,
                    defined_mask: 0x0f,
                    ..
                }) if source.argument() == argument
            ));
            assert_eq!(channel.frontend(), frontend_before);
            assert_eq!(channel.two_d(), &two_d_before);
            assert_eq!(channel.three_d(), &three_d_before);
        }

        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = incrementing_packet(0x0d64 / 4, &[15, 0]);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
                if source.method() == GpuMethodId(0x0d68)
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }

    #[test]
    fn active_visible_call_limits_block_only_draws_before_cache_effects() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        let address_space = resource_address_space();
        let capabilities = lowering_capabilities(BackendFeatures::empty());
        let cache = MaxwellThreeDLoweringCache::default();

        for (argument, expected) in [
            (0, MaxwellThreeDVisibleCallLimit::Calls0),
            (8, MaxwellThreeDVisibleCallLimit::Calls128),
        ] {
            program_three_d(&mut channel, 0x0d64, argument);
            let source = channel
                .three_d()
                .shader_execution()
                .visible_call_limit()
                .source()
                .unwrap();
            let resources =
                resolve_maxwell_three_d_resources(channel.three_d(), &address_space).unwrap();
            let cache_before = cache.clone();

            assert!(matches!(
                preflight_maxwell_three_d_operation(
                    channel.three_d(),
                    &resources,
                    MaxwellThreeDOperationTrigger::DrawVertexArray {
                        source,
                        vertex_count: 3,
                    },
                    None,
                    FrontendSubmissionId::new(10),
                    Vec::new(),
                    &capabilities,
                    &cache,
                ),
                Err(MaxwellThreeDLoweringError::UnsupportedVisibleCallLimitSemantics(limit))
                    if limit == expected
            ));
            assert_eq!(cache, cache_before);
        }

        let clear = packet(0x19d0 / 4, 0x3c);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &clear.packets()[0],
        )
        .unwrap();
        let triggered = &dispatch.operations()[0];
        let resources =
            resolve_maxwell_three_d_resources(triggered.state(), &address_space).unwrap();
        let cache_before = cache.clone();
        assert!(matches!(
            preflight_maxwell_three_d_operation(
                triggered.state(),
                &resources,
                triggered.trigger(),
                None,
                FrontendSubmissionId::new(11),
                Vec::new(),
                &capabilities,
                &cache,
            ),
            Err(MaxwellThreeDLoweringError::IncompleteClear(
                "horizontal rectangle"
            ))
        ));
        assert_eq!(cache, cache_before);
    }

    #[test]
    fn visible_call_no_check_does_not_invent_a_draw_limit() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        program_three_d(&mut channel, 0x0d64, 15);
        let source = channel
            .three_d()
            .shader_execution()
            .visible_call_limit()
            .source()
            .unwrap();
        let resources =
            resolve_maxwell_three_d_resources(channel.three_d(), &resource_address_space())
                .unwrap();
        let cache = MaxwellThreeDLoweringCache::default();
        let cache_before = cache.clone();
        let result = preflight_maxwell_three_d_operation(
            channel.three_d(),
            &resources,
            MaxwellThreeDOperationTrigger::DrawVertexArray {
                source,
                vertex_count: 3,
            },
            None,
            FrontendSubmissionId::new(10),
            Vec::new(),
            &lowering_capabilities(BackendFeatures::empty()),
            &cache,
        );

        assert!(!matches!(
            result,
            Err(MaxwellThreeDLoweringError::UnsupportedVisibleCallLimitSemantics(_))
        ));
        assert_eq!(cache, cache_before);
    }

    #[test]
    fn zcull_stats_enable_is_typed_source_preserving_isolated_three_d_state() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let coverage_before = channel.three_d().coverage().clone();
        assert_eq!(
            channel.three_d().zcull().stats_enable().origin(),
            MaxwellThreeDRegisterOrigin::Unset
        );

        for (argument, expected) in [
            (0, MaxwellThreeDZCullStatsEnable::Disabled),
            (1, MaxwellThreeDZCullStatsEnable::Enabled),
        ] {
            let decoded = packet(0x151c / 4, argument);
            let dispatch = dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0],
            )
            .unwrap();
            let source = dispatch.methods()[0].method().source();
            let register = channel.three_d().zcull().stats_enable();

            assert_eq!(
                dispatch.methods()[0].metadata().method_name(),
                "SET_ZCULL_STATS"
            );
            assert_eq!(
                dispatch.methods()[0].effect(),
                MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::ZCull(
                    MaxwellThreeDZCullStateWrite::StatsEnable {
                        value: expected,
                        source,
                    }
                ))
            );
            assert!(dispatch.operations().is_empty());
            assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
            assert_eq!(register.raw(), Some(argument));
            assert_eq!(register.value().copied(), Some(expected));
            assert_eq!(register.source(), Some(source));
            assert_eq!(expected.raw(), argument);
            assert_eq!(channel.frontend(), frontend_before);
            assert_eq!(channel.two_d(), &two_d_before);
            assert_eq!(channel.three_d().coverage(), &coverage_before);
        }
    }

    #[test]
    fn invalid_zcull_stats_values_and_packet_suffix_are_rejected_atomically() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        program_three_d(&mut channel, 0x151c, 1);

        for argument in [2, 3, 0x8000_0000, u32::MAX] {
            let frontend_before = channel.frontend();
            let two_d_before = channel.two_d().clone();
            let three_d_before = channel.three_d().clone();
            let decoded = packet(0x151c / 4, argument);

            assert!(matches!(
                dispatch_maxwell_engine_packet(
                    &mut channel,
                    FrontendSubmissionId::new(3),
                    &decoded.packets()[0]
                ),
                Err(MaxwellEngineDispatchError::InvalidMethodValue {
                    source,
                    defined_mask: 1,
                    ..
                }) if source.argument() == argument
            ));
            assert_eq!(channel.frontend(), frontend_before);
            assert_eq!(channel.two_d(), &two_d_before);
            assert_eq!(channel.three_d(), &three_d_before);
        }

        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = incrementing_packet(0x151c / 4, &[0, 0]);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
                if source.method() == GpuMethodId(0x1520)
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }

    #[test]
    fn enabled_zcull_stats_block_only_draws_before_cache_effects() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        let address_space = resource_address_space();
        let capabilities = lowering_capabilities(BackendFeatures::empty());
        let cache = MaxwellThreeDLoweringCache::default();

        program_three_d(&mut channel, 0x151c, 1);
        let source = channel.three_d().zcull().stats_enable().source().unwrap();
        let resources =
            resolve_maxwell_three_d_resources(channel.three_d(), &address_space).unwrap();
        let cache_before = cache.clone();
        let error = match preflight_maxwell_three_d_operation(
            channel.three_d(),
            &resources,
            MaxwellThreeDOperationTrigger::DrawVertexArray {
                source,
                vertex_count: 3,
            },
            None,
            FrontendSubmissionId::new(10),
            Vec::new(),
            &capabilities,
            &cache,
        ) {
            Ok(_) => panic!("enabled Z-cull statistics unexpectedly allowed a draw"),
            Err(error) => error,
        };
        assert!(matches!(
            &error,
            MaxwellThreeDLoweringError::UnsupportedZCullStatsSemantics
        ));
        assert_eq!(
            error.to_string(),
            "MAXWELL_B enabled Z-cull statistics have no implemented counter accumulation, visibility, or reporting semantics"
        );
        assert_eq!(cache, cache_before);

        let clear = packet(0x19d0 / 4, 0x3c);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &clear.packets()[0],
        )
        .unwrap();
        let triggered = &dispatch.operations()[0];
        let resources =
            resolve_maxwell_three_d_resources(triggered.state(), &address_space).unwrap();
        assert!(matches!(
            preflight_maxwell_three_d_operation(
                triggered.state(),
                &resources,
                triggered.trigger(),
                None,
                FrontendSubmissionId::new(11),
                Vec::new(),
                &capabilities,
                &cache,
            ),
            Err(MaxwellThreeDLoweringError::IncompleteClear(
                "horizontal rectangle"
            ))
        ));
        assert_eq!(cache, cache_before);

        program_three_d(&mut channel, 0x151c, 0);
        let source = channel.three_d().zcull().stats_enable().source().unwrap();
        let resources =
            resolve_maxwell_three_d_resources(channel.three_d(), &address_space).unwrap();
        let result = preflight_maxwell_three_d_operation(
            channel.three_d(),
            &resources,
            MaxwellThreeDOperationTrigger::DrawVertexArray {
                source,
                vertex_count: 3,
            },
            None,
            FrontendSubmissionId::new(12),
            Vec::new(),
            &capabilities,
            &cache,
        );
        assert!(!matches!(
            result,
            Err(MaxwellThreeDLoweringError::UnsupportedZCullStatsSemantics)
        ));
        assert_eq!(cache, cache_before);
    }

    #[test]
    fn sm_timeout_counter_bit_is_bounded_source_preserving_state() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        let two_d_before = channel.two_d().clone();
        assert_eq!(
            channel
                .three_d()
                .shader_execution()
                .sm_timeout_counter_bit()
                .origin(),
            MaxwellThreeDRegisterOrigin::Unset
        );

        for argument in [0, 0x17, MAXWELL_THREE_D_SM_TIMEOUT_COUNTER_BIT_MAX] {
            let decoded = packet(0x0de4 / 4, argument);
            let dispatch = dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0],
            )
            .unwrap();
            let source = dispatch.methods()[0].method().source();
            let value = MaxwellThreeDSmTimeoutCounterBit::new(argument).unwrap();
            let register = channel
                .three_d()
                .shader_execution()
                .sm_timeout_counter_bit();

            assert_eq!(
                dispatch.methods()[0].metadata().method_name(),
                "SET_SM_TIMEOUT_INTERVAL"
            );
            assert_eq!(
                dispatch.methods()[0].effect(),
                MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::ShaderExecution(
                    MaxwellThreeDShaderExecutionStateWrite::SmTimeoutCounterBit { value, source }
                ))
            );
            assert!(dispatch.operations().is_empty());
            assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
            assert_eq!(register.raw(), Some(argument));
            assert_eq!(register.value().copied(), Some(value));
            assert_eq!(register.source(), Some(source));
            assert_eq!(u32::from(value.get()), argument);
            assert_eq!(channel.two_d(), &two_d_before);
        }
    }

    #[test]
    fn sm_timeout_reserved_bits_are_rejected_atomically() {
        let mut channel = channel();
        bind_three_d(&mut channel);

        for argument in [0x40, 0x80, u32::MAX] {
            let frontend_before = channel.frontend();
            let two_d_before = channel.two_d().clone();
            let three_d_before = channel.three_d().clone();
            let decoded = packet(0x0de4 / 4, argument);

            assert!(matches!(
                dispatch_maxwell_engine_packet(
                    &mut channel,
                    FrontendSubmissionId::new(3),
                    &decoded.packets()[0]
                ),
                Err(MaxwellEngineDispatchError::InvalidMethodValue {
                    source,
                    defined_mask: MAXWELL_THREE_D_SM_TIMEOUT_COUNTER_BIT_MAX,
                    ..
                }) if source.argument() == argument
            ));
            assert_eq!(channel.frontend(), frontend_before);
            assert_eq!(channel.two_d(), &two_d_before);
            assert_eq!(channel.three_d(), &three_d_before);
        }
    }

    #[test]
    fn programmed_sm_timeout_stops_shader_execution_before_cache_publication() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        let decoded = packet(0x0de4 / 4, 0x17);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
        let resources =
            resolve_maxwell_three_d_resources(channel.three_d(), &resource_address_space())
                .unwrap();
        let cache = MaxwellThreeDLoweringCache::default();
        let cache_before = cache.clone();

        assert!(matches!(
            preflight_maxwell_three_d_operation(
                channel.three_d(),
                &resources,
                MaxwellThreeDOperationTrigger::DrawVertexArray {
                    source: dispatch.methods()[0].method().source(),
                    vertex_count: 3,
                },
                None,
                FrontendSubmissionId::new(10),
                Vec::new(),
                &lowering_capabilities(BackendFeatures::empty()),
                &cache,
            ),
            Err(MaxwellThreeDLoweringError::UnsupportedSmTimeoutIntervalSemantics(value))
                if value.get() == 0x17
        ));
        assert_eq!(cache, cache_before);
    }

    #[test]
    fn csaa_enable_is_typed_source_preserving_state_isolated_from_multisample() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        let fixed_function_before = channel.three_d().fixed_function().clone();
        let two_d_before = channel.two_d().clone();
        assert_eq!(
            channel.three_d().coverage().csaa_enable().origin(),
            MaxwellThreeDRegisterOrigin::Unset
        );

        for (argument, expected) in [
            (0, MaxwellThreeDCsaaEnable::Disabled),
            (1, MaxwellThreeDCsaaEnable::Enabled),
        ] {
            let decoded = packet(0x15b4 / 4, argument);
            let dispatch = dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0],
            )
            .unwrap();
            let source = dispatch.methods()[0].method().source();
            let register = channel.three_d().coverage().csaa_enable();

            assert_eq!(
                dispatch.methods()[0].metadata().method_name(),
                "CSAA_ENABLE"
            );
            assert_eq!(
                dispatch.methods()[0].effect(),
                MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::Coverage(
                    MaxwellThreeDCoverageStateWrite::CsaaEnable {
                        value: expected,
                        source,
                    }
                ))
            );
            assert!(dispatch.operations().is_empty());
            assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
            assert_eq!(register.raw(), Some(argument));
            assert_eq!(register.value().copied(), Some(expected));
            assert_eq!(register.source(), Some(source));
            assert_eq!(channel.three_d().fixed_function(), &fixed_function_before);
            assert_eq!(channel.two_d(), &two_d_before);
        }
    }

    #[test]
    fn invalid_csaa_values_and_packet_suffix_are_rejected_atomically() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        program_three_d(&mut channel, 0x15b4, 0);

        for argument in [2, 3, 0x8000_0000, u32::MAX] {
            let frontend_before = channel.frontend();
            let two_d_before = channel.two_d().clone();
            let three_d_before = channel.three_d().clone();
            let decoded = packet(0x15b4 / 4, argument);

            assert!(matches!(
                dispatch_maxwell_engine_packet(
                    &mut channel,
                    FrontendSubmissionId::new(3),
                    &decoded.packets()[0]
                ),
                Err(MaxwellEngineDispatchError::InvalidMethodValue {
                    source,
                    defined_mask: 1,
                    ..
                }) if source.argument() == argument
            ));
            assert_eq!(channel.frontend(), frontend_before);
            assert_eq!(channel.two_d(), &two_d_before);
            assert_eq!(channel.three_d(), &three_d_before);
        }

        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = incrementing_packet(0x15b4 / 4, &[1, 0]);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
                if source.method() == GpuMethodId(0x15b8)
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }

    #[test]
    fn vertex_array_restart_is_typed_source_preserving_and_isolated_from_indexed_restart() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        let two_d_before = channel.two_d().clone();
        assert_eq!(
            channel
                .three_d()
                .vertex_input()
                .primitive()
                .vertex_array_restart_enabled()
                .origin(),
            MaxwellThreeDRegisterOrigin::Unset
        );

        program_three_d(&mut channel, 0x1644, 1);
        program_three_d(&mut channel, 0x1648, 0xfeed_beef);
        let indexed_enable_before = channel
            .three_d()
            .vertex_input()
            .primitive()
            .restart_enabled()
            .to_owned();
        let indexed_index_before = channel
            .three_d()
            .vertex_input()
            .primitive()
            .restart_index()
            .to_owned();

        for (argument, expected) in [
            (0, MaxwellThreeDVertexArrayPrimitiveRestartEnable::Disabled),
            (1, MaxwellThreeDVertexArrayPrimitiveRestartEnable::Enabled),
        ] {
            let decoded = packet(0x0de8 / 4, argument);
            let dispatch = dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0],
            )
            .unwrap();
            let source = dispatch.methods()[0].method().source();
            let primitive = channel.three_d().vertex_input().primitive();
            let register = primitive.vertex_array_restart_enabled();

            assert_eq!(
                dispatch.methods()[0].metadata().method_name(),
                "SET_DA_PRIMITIVE_RESTART_VERTEX_ARRAY"
            );
            assert_eq!(
                dispatch.methods()[0].effect(),
                MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::VertexInput(
                    MaxwellThreeDVertexInputWrite::VertexArrayPrimitiveRestartEnable {
                        value: expected,
                        source,
                    }
                ))
            );
            assert!(dispatch.operations().is_empty());
            assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
            assert_eq!(register.raw(), Some(argument));
            assert_eq!(register.value().copied(), Some(expected));
            assert_eq!(register.source(), Some(source));
            assert_eq!(expected.raw(), argument);
            assert_eq!(primitive.restart_enabled(), &indexed_enable_before);
            assert_eq!(primitive.restart_index(), &indexed_index_before);
            assert_eq!(channel.two_d(), &two_d_before);
        }

        let vertex_array_before = channel
            .three_d()
            .vertex_input()
            .primitive()
            .vertex_array_restart_enabled()
            .to_owned();
        program_three_d(&mut channel, 0x1644, 0);
        program_three_d(&mut channel, 0x1648, 7);
        assert_eq!(
            channel
                .three_d()
                .vertex_input()
                .primitive()
                .vertex_array_restart_enabled(),
            &vertex_array_before
        );
    }

    #[test]
    fn invalid_vertex_array_restart_values_and_packet_suffix_are_atomic() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        program_three_d(&mut channel, 0x0de8, 0);

        for argument in [2, 3, 0x8000_0000, u32::MAX] {
            let frontend_before = channel.frontend();
            let two_d_before = channel.two_d().clone();
            let three_d_before = channel.three_d().clone();
            let decoded = packet(0x0de8 / 4, argument);

            assert!(matches!(
                dispatch_maxwell_engine_packet(
                    &mut channel,
                    FrontendSubmissionId::new(3),
                    &decoded.packets()[0]
                ),
                Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                    source,
                    method_name: "SET_DA_PRIMITIVE_RESTART_VERTEX_ARRAY",
                    ..
                }) if source.argument() == argument
            ));
            assert_eq!(channel.frontend(), frontend_before);
            assert_eq!(channel.two_d(), &two_d_before);
            assert_eq!(channel.three_d(), &three_d_before);
        }

        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = incrementing_packet(0x0de8 / 4, &[1, 0]);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
                if source.method() == GpuMethodId(0x0dec)
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }

    #[test]
    fn shade_mode_is_typed_source_preserving_and_part_of_pipeline_identity() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        program_three_d(&mut channel, 0x12cc, 0);
        let depth_test_before = channel
            .three_d()
            .fixed_function()
            .register(MaxwellThreeDFixedFunctionRegister::DepthTestEnable)
            .to_owned();
        let two_d_before = channel.two_d().clone();
        let unset_dependencies = channel.three_d().pipeline_dependencies(&[]);
        assert_eq!(
            channel
                .three_d()
                .fixed_function()
                .register(MaxwellThreeDFixedFunctionRegister::ShadeMode)
                .origin(),
            MaxwellThreeDRegisterOrigin::Unset
        );

        let mut previous_dependencies = unset_dependencies;
        for (argument, expected) in [
            (0x1d00, MaxwellThreeDShadeMode::Flat),
            (0x1d01, MaxwellThreeDShadeMode::Smooth),
        ] {
            let decoded = packet(0x12d4 / 4, argument);
            let dispatch = dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0],
            )
            .unwrap();
            let source = dispatch.methods()[0].method().source();
            let register = channel
                .three_d()
                .fixed_function()
                .register(MaxwellThreeDFixedFunctionRegister::ShadeMode);

            assert_eq!(
                dispatch.methods()[0].metadata().method_name(),
                "SET_SHADE_MODE"
            );
            assert_eq!(
                dispatch.methods()[0].effect(),
                MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::FixedFunction(
                    MaxwellThreeDFixedFunctionWrite::Register {
                        register: MaxwellThreeDFixedFunctionRegister::ShadeMode,
                        value: MaxwellThreeDFixedFunctionValue::ShadeMode(expected),
                        source,
                    }
                ))
            );
            assert!(dispatch.operations().is_empty());
            assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
            assert_eq!(register.raw(), Some(argument));
            assert_eq!(
                register.value().copied(),
                Some(MaxwellThreeDFixedFunctionValue::ShadeMode(expected))
            );
            assert_eq!(register.source(), Some(source));
            assert_eq!(expected.raw(), argument);
            assert_eq!(
                channel
                    .three_d()
                    .fixed_function()
                    .register(MaxwellThreeDFixedFunctionRegister::DepthTestEnable),
                &depth_test_before
            );
            assert_eq!(channel.two_d(), &two_d_before);

            let dependencies = channel.three_d().pipeline_dependencies(&[]);
            assert_ne!(dependencies, previous_dependencies);
            previous_dependencies = dependencies;
        }
    }

    #[test]
    fn invalid_shade_modes_and_packet_suffix_are_atomic() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        program_three_d(&mut channel, 0x12d4, 0x1d00);

        for argument in [0, 1, 0x1cff, 0x1d02, 0x8000_0000, u32::MAX] {
            let frontend_before = channel.frontend();
            let two_d_before = channel.two_d().clone();
            let three_d_before = channel.three_d().clone();
            let decoded = packet(0x12d4 / 4, argument);
            assert!(matches!(
                dispatch_maxwell_engine_packet(
                    &mut channel,
                    FrontendSubmissionId::new(3),
                    &decoded.packets()[0]
                ),
                Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                    source,
                    method_name: "SET_SHADE_MODE",
                    ..
                }) if source.argument() == argument
            ));
            assert_eq!(channel.frontend(), frontend_before);
            assert_eq!(channel.two_d(), &two_d_before);
            assert_eq!(channel.three_d(), &three_d_before);
        }

        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = incrementing_packet(0x12d4 / 4, &[0x1d01, 0]);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
                if source.method() == GpuMethodId(0x12d8)
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }

    #[test]
    fn shade_mode_is_consumed_only_by_draws_before_cache_or_backend_effects() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        let address_space = resource_address_space();
        let capabilities = lowering_capabilities(BackendFeatures::empty());
        let cache = MaxwellThreeDLoweringCache::default();

        for (argument, expected) in [
            (0x1d00, MaxwellThreeDShadeMode::Flat),
            (0x1d01, MaxwellThreeDShadeMode::Smooth),
        ] {
            program_three_d(&mut channel, 0x12d4, argument);
            let source = channel
                .three_d()
                .fixed_function()
                .register(MaxwellThreeDFixedFunctionRegister::ShadeMode)
                .source()
                .unwrap();
            let resources =
                resolve_maxwell_three_d_resources(channel.three_d(), &address_space).unwrap();
            let cache_before = cache.clone();
            assert!(matches!(
                preflight_maxwell_three_d_operation(
                    channel.three_d(),
                    &resources,
                    MaxwellThreeDOperationTrigger::DrawVertexArray {
                        source,
                        vertex_count: 3,
                    },
                    None,
                    FrontendSubmissionId::new(10),
                    Vec::new(),
                    &capabilities,
                    &cache,
                ),
                Err(MaxwellThreeDLoweringError::UnsupportedShadeModeSemantics(mode))
                    if mode == expected
            ));
            assert_eq!(cache, cache_before);
        }

        let clear = packet(0x19d0 / 4, 0x3c);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &clear.packets()[0],
        )
        .unwrap();
        let triggered = &dispatch.operations()[0];
        let resources =
            resolve_maxwell_three_d_resources(triggered.state(), &address_space).unwrap();
        let cache_before = cache.clone();
        assert!(matches!(
            preflight_maxwell_three_d_operation(
                triggered.state(),
                &resources,
                triggered.trigger(),
                None,
                FrontendSubmissionId::new(11),
                Vec::new(),
                &capabilities,
                &cache,
            ),
            Err(MaxwellThreeDLoweringError::IncompleteClear(
                "horizontal rectangle"
            ))
        ));
        assert_eq!(cache, cache_before);
    }

    #[test]
    fn common_blend_enable_is_typed_source_preserving_and_family_isolated() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        program_three_d(&mut channel, 0x12e4, 1);
        program_three_d(&mut channel, 0x1360, 1);
        program_three_d(&mut channel, 0x1e00, 0);
        let per_target_mode_before = channel
            .three_d()
            .fixed_function()
            .register(MaxwellThreeDFixedFunctionRegister::BlendPerTargetEnable)
            .to_owned();
        let per_target_enable_before = *channel.three_d().fixed_function().blend_enable();
        let per_target_state_before = *channel.three_d().fixed_function().per_target_blend();
        let two_d_before = channel.two_d().clone();
        assert_eq!(
            channel
                .three_d()
                .fixed_function()
                .blend_enable_common()
                .origin(),
            MaxwellThreeDRegisterOrigin::Unset
        );

        for (argument, expected) in [
            (0, MaxwellThreeDBlendEnableCommon::Disabled),
            (1, MaxwellThreeDBlendEnableCommon::Enabled),
        ] {
            let decoded = packet(0x135c / 4, argument);
            let dispatch = dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0],
            )
            .unwrap();
            let source = dispatch.methods()[0].method().source();
            let fixed = channel.three_d().fixed_function();
            let register = fixed.blend_enable_common();

            assert_eq!(
                dispatch.methods()[0].metadata().method_name(),
                "SET_BLEND_ENABLE_COMMON"
            );
            assert_eq!(
                dispatch.methods()[0].effect(),
                MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::FixedFunction(
                    MaxwellThreeDFixedFunctionWrite::BlendEnableCommon {
                        value: expected,
                        source,
                    }
                ))
            );
            assert!(dispatch.operations().is_empty());
            assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
            assert_eq!(register.raw(), Some(argument));
            assert_eq!(register.value().copied(), Some(expected));
            assert_eq!(register.source(), Some(source));
            assert_eq!(expected.raw(), argument);
            assert_eq!(
                fixed.register(MaxwellThreeDFixedFunctionRegister::BlendPerTargetEnable),
                &per_target_mode_before
            );
            assert_eq!(fixed.blend_enable(), &per_target_enable_before);
            assert_eq!(fixed.per_target_blend(), &per_target_state_before);
            assert_eq!(channel.two_d(), &two_d_before);
        }

        let common_before = channel
            .three_d()
            .fixed_function()
            .blend_enable_common()
            .to_owned();
        program_three_d(&mut channel, 0x12e4, 0);
        program_three_d(&mut channel, 0x1360, 0);
        program_three_d(&mut channel, 0x1e00, 1);
        assert_eq!(
            channel.three_d().fixed_function().blend_enable_common(),
            &common_before
        );
    }

    #[test]
    fn invalid_common_blend_enable_values_and_packet_suffix_are_atomic() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        program_three_d(&mut channel, 0x135c, 0);

        for argument in [2, 3, 0x8000_0000, u32::MAX] {
            let frontend_before = channel.frontend();
            let two_d_before = channel.two_d().clone();
            let three_d_before = channel.three_d().clone();
            let decoded = packet(0x135c / 4, argument);
            assert!(matches!(
                dispatch_maxwell_engine_packet(
                    &mut channel,
                    FrontendSubmissionId::new(3),
                    &decoded.packets()[0]
                ),
                Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                    source,
                    method_name: "SET_BLEND_ENABLE_COMMON",
                    ..
                }) if source.argument() == argument
            ));
            assert_eq!(channel.frontend(), frontend_before);
            assert_eq!(channel.two_d(), &two_d_before);
            assert_eq!(channel.three_d(), &three_d_before);
        }

        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = incrementing_packet(0x135c / 4, &[1, 2]);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                source,
                method_name: "SET_BLEND",
                ..
            }) if source.method() == GpuMethodId(0x1360)
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }

    #[test]
    fn draw_resolves_common_and_per_target_blend_state_before_effects() {
        let target_allocation = CanonicalAllocation::zeroed(0x10000, 0x1000).unwrap();
        let mut address_space = resource_address_space();
        let target = map_resource(
            &mut address_space,
            target_allocation
                .backing_range(MemoryPermissions::READ_WRITE)
                .unwrap(),
            73,
            0xfe,
        )
        .offset()
        .get();
        let mut channel = channel();
        bind_three_d(&mut channel);
        program_color_target(&mut channel, 0, target, 0xd5);
        program_three_d(&mut channel, 0x15d0, 0);
        program_three_d(&mut channel, 0x121c, 1);
        let cache = MaxwellThreeDLoweringCache::default();
        let capabilities = lowering_capabilities(BackendFeatures::empty());
        let source = channel
            .three_d()
            .render_targets()
            .color_target_selection()
            .source()
            .unwrap();

        let preflight = |channel: &MaxwellGpuChannel| {
            let resources =
                resolve_maxwell_three_d_resources(channel.three_d(), &address_space).unwrap();
            preflight_maxwell_three_d_operation(
                channel.three_d(),
                &resources,
                MaxwellThreeDOperationTrigger::DrawVertexArray {
                    source,
                    vertex_count: 3,
                },
                None,
                FrontendSubmissionId::new(10),
                Vec::new(),
                &capabilities,
                &cache,
            )
        };

        assert!(matches!(
            preflight(&channel),
            Err(MaxwellThreeDLoweringError::IncompleteBlendState {
                target: None,
                field: "SET_BLEND_STATE_PER_TARGET"
            })
        ));
        program_three_d(&mut channel, 0x12e4, 0);
        assert!(matches!(
            preflight(&channel),
            Err(MaxwellThreeDLoweringError::IncompleteBlendState {
                target: None,
                field: "SET_BLEND_ENABLE_COMMON"
            })
        ));
        program_three_d(&mut channel, 0x135c, 0);
        assert!(matches!(
            preflight(&channel),
            Err(MaxwellThreeDLoweringError::ShaderTranslationRequired)
        ));

        program_three_d(&mut channel, 0x135c, 1);
        assert!(matches!(
            preflight(&channel),
            Err(MaxwellThreeDLoweringError::IncompleteBlendState {
                target: None,
                field: "SET_BLEND_SEPARATE_FOR_ALPHA"
            })
        ));
        for (method, argument) in [(0x133c, 1), (0x1340, 1), (0x1344, 1), (0x1348, 1)] {
            program_three_d(&mut channel, method, argument);
        }
        assert!(matches!(
            preflight(&channel),
            Err(MaxwellThreeDLoweringError::IncompleteBlendState {
                target: None,
                field: "SET_BLEND_OP_ALPHA"
            })
        ));
        for (method, argument) in [(0x134c, 1), (0x1350, 1), (0x1358, 1)] {
            program_three_d(&mut channel, method, argument);
        }
        let cache_before = cache.clone();
        assert!(matches!(
            preflight(&channel),
            Err(MaxwellThreeDLoweringError::UnsupportedBlendSemantics { target: None })
        ));
        assert_eq!(cache, cache_before);

        program_three_d(&mut channel, 0x12e4, 1);
        assert!(matches!(
            preflight(&channel),
            Err(MaxwellThreeDLoweringError::IncompleteBlendState {
                target: Some(0),
                field: "SET_BLEND(i)"
            })
        ));
        program_three_d(&mut channel, 0x1360, 0);
        assert!(matches!(
            preflight(&channel),
            Err(MaxwellThreeDLoweringError::ShaderTranslationRequired)
        ));
        program_three_d(&mut channel, 0x1360, 1);
        assert!(matches!(
            preflight(&channel),
            Err(MaxwellThreeDLoweringError::IncompleteBlendState {
                target: Some(0),
                field: "SET_BLEND_PER_TARGET_SEPARATE_FOR_ALPHA"
            })
        ));
        for (method, argument) in [(0x1e00, 1), (0x1e04, 1), (0x1e08, 1), (0x1e0c, 1)] {
            program_three_d(&mut channel, method, argument);
        }
        assert!(matches!(
            preflight(&channel),
            Err(MaxwellThreeDLoweringError::IncompleteBlendState {
                target: Some(0),
                field: "SET_BLEND_PER_TARGET_OP_ALPHA"
            })
        ));
        for (method, argument) in [(0x1e10, 1), (0x1e14, 1), (0x1e18, 1)] {
            program_three_d(&mut channel, method, argument);
        }
        assert!(matches!(
            preflight(&channel),
            Err(MaxwellThreeDLoweringError::UnsupportedBlendSemantics { target: Some(0) })
        ));
        assert_eq!(cache, cache_before);

        let clear = packet(0x19d0 / 4, 0x3c);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &clear.packets()[0],
        )
        .unwrap();
        let triggered = &dispatch.operations()[0];
        let resources =
            resolve_maxwell_three_d_resources(triggered.state(), &address_space).unwrap();
        assert!(matches!(
            preflight_maxwell_three_d_operation(
                triggered.state(),
                &resources,
                triggered.trigger(),
                None,
                FrontendSubmissionId::new(11),
                Vec::new(),
                &capabilities,
                &cache,
            ),
            Err(MaxwellThreeDLoweringError::IncompleteClear(
                "horizontal rectangle"
            ))
        ));
        assert_eq!(cache, cache_before);
    }

    #[test]
    fn vertex_array_restart_is_consumed_only_by_non_indexed_draws() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        program_three_d(&mut channel, 0x121c, 0);
        let cache = MaxwellThreeDLoweringCache::default();
        let capabilities = lowering_capabilities(BackendFeatures::empty());

        program_three_d(&mut channel, 0x0de8, 0);
        let source = channel
            .three_d()
            .vertex_input()
            .primitive()
            .vertex_array_restart_enabled()
            .source()
            .unwrap();
        let resources =
            resolve_maxwell_three_d_resources(channel.three_d(), &resource_address_space())
                .unwrap();
        assert!(matches!(
            preflight_maxwell_three_d_operation(
                channel.three_d(),
                &resources,
                MaxwellThreeDOperationTrigger::DrawVertexArray {
                    source,
                    vertex_count: 3,
                },
                None,
                FrontendSubmissionId::new(10),
                Vec::new(),
                &capabilities,
                &cache,
            ),
            Err(MaxwellThreeDLoweringError::ShaderTranslationRequired)
        ));

        program_three_d(&mut channel, 0x0de8, 1);
        let source = channel
            .three_d()
            .vertex_input()
            .primitive()
            .vertex_array_restart_enabled()
            .source()
            .unwrap();
        let cache_before = cache.clone();
        assert!(matches!(
            preflight_maxwell_three_d_operation(
                channel.three_d(),
                &resources,
                MaxwellThreeDOperationTrigger::DrawVertexArray {
                    source,
                    vertex_count: 3,
                },
                None,
                FrontendSubmissionId::new(11),
                Vec::new(),
                &capabilities,
                &cache,
            ),
            Err(MaxwellThreeDLoweringError::UnsupportedVertexArrayPrimitiveRestartSemantics)
        ));
        assert_eq!(cache, cache_before);

        let clear = packet(0x19d0 / 4, 0x3c);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &clear.packets()[0],
        )
        .unwrap();
        let triggered = &dispatch.operations()[0];
        let clear_resources =
            resolve_maxwell_three_d_resources(triggered.state(), &resource_address_space())
                .unwrap();
        assert!(matches!(
            preflight_maxwell_three_d_operation(
                triggered.state(),
                &clear_resources,
                triggered.trigger(),
                None,
                FrontendSubmissionId::new(12),
                Vec::new(),
                &capabilities,
                &cache,
            ),
            Err(MaxwellThreeDLoweringError::IncompleteClear(
                "horizontal rectangle"
            ))
        ));
        assert_eq!(cache, cache_before);
    }

    #[test]
    fn aliased_line_width_selector_is_typed_source_preserving_isolated_state() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        let fixed_function_before = channel.three_d().fixed_function().clone();
        let coverage_before = channel.three_d().coverage().clone();
        let two_d_before = channel.two_d().clone();
        assert_eq!(
            channel
                .three_d()
                .line()
                .aliased_line_width_enable()
                .origin(),
            MaxwellThreeDRegisterOrigin::Unset
        );

        for (argument, expected) in [
            (0, MaxwellThreeDAliasedLineWidthEnable::Disabled),
            (1, MaxwellThreeDAliasedLineWidthEnable::Enabled),
        ] {
            let decoded = packet(0x020c / 4, argument);
            let dispatch = dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0],
            )
            .unwrap();
            let source = dispatch.methods()[0].method().source();
            let register = channel.three_d().line().aliased_line_width_enable();

            assert_eq!(
                dispatch.methods()[0].metadata().method_name(),
                "SET_ALIASED_LINE_WIDTH_ENABLE"
            );
            assert_eq!(
                dispatch.methods()[0].effect(),
                MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::Line(
                    MaxwellThreeDLineStateWrite::AliasedLineWidthEnable {
                        value: expected,
                        source,
                    }
                ))
            );
            assert!(dispatch.operations().is_empty());
            assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
            assert_eq!(register.raw(), Some(argument));
            assert_eq!(register.value().copied(), Some(expected));
            assert_eq!(register.source(), Some(source));
            assert_eq!(channel.three_d().fixed_function(), &fixed_function_before);
            assert_eq!(channel.three_d().coverage(), &coverage_before);
            assert_eq!(channel.two_d(), &two_d_before);
        }
    }

    #[test]
    fn invalid_aliased_line_width_values_and_packet_suffix_are_atomic() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        program_three_d(&mut channel, 0x020c, 0);

        for argument in [2, 3, 0x8000_0000, u32::MAX] {
            let frontend_before = channel.frontend();
            let two_d_before = channel.two_d().clone();
            let three_d_before = channel.three_d().clone();
            let decoded = packet(0x020c / 4, argument);

            assert!(matches!(
                dispatch_maxwell_engine_packet(
                    &mut channel,
                    FrontendSubmissionId::new(3),
                    &decoded.packets()[0]
                ),
                Err(MaxwellEngineDispatchError::InvalidMethodValue {
                    source,
                    defined_mask: 1,
                    ..
                }) if source.argument() == argument
            ));
            assert_eq!(channel.frontend(), frontend_before);
            assert_eq!(channel.two_d(), &two_d_before);
            assert_eq!(channel.three_d(), &three_d_before);
        }

        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = incrementing_packet(0x020c / 4, &[1, 0]);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
                if source.method() == GpuMethodId(0x0210)
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }

    #[test]
    fn aliased_line_width_is_consumed_only_by_line_rasterization() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        program_three_d(&mut channel, 0x121c, 0);
        program_three_d(&mut channel, 0x1618, 1);
        let resources =
            resolve_maxwell_three_d_resources(channel.three_d(), &resource_address_space())
                .unwrap();
        let cache = MaxwellThreeDLoweringCache::default();
        let capabilities = lowering_capabilities(BackendFeatures::empty());
        let source = channel
            .three_d()
            .vertex_input()
            .primitive()
            .begin()
            .source()
            .unwrap();

        assert!(matches!(
            preflight_maxwell_three_d_operation(
                channel.three_d(),
                &resources,
                MaxwellThreeDOperationTrigger::DrawVertexArray {
                    source,
                    vertex_count: 2,
                },
                None,
                FrontendSubmissionId::new(10),
                Vec::new(),
                &capabilities,
                &cache,
            ),
            Err(MaxwellThreeDLoweringError::IncompleteDraw(
                "SET_ALIASED_LINE_WIDTH_ENABLE"
            ))
        ));

        program_three_d(&mut channel, 0x020c, 0);
        assert!(matches!(
            preflight_maxwell_three_d_operation(
                channel.three_d(),
                &resources,
                MaxwellThreeDOperationTrigger::DrawVertexArray {
                    source,
                    vertex_count: 2,
                },
                None,
                FrontendSubmissionId::new(11),
                Vec::new(),
                &capabilities,
                &cache,
            ),
            Err(MaxwellThreeDLoweringError::IncompleteDraw(
                "SET_LINE_WIDTH_FLOAT"
            ))
        ));
        program_three_d(&mut channel, 0x13b0, 0x3f80_0000);
        assert!(matches!(
            preflight_maxwell_three_d_operation(
                channel.three_d(),
                &resources,
                MaxwellThreeDOperationTrigger::DrawVertexArray {
                    source,
                    vertex_count: 2,
                },
                None,
                FrontendSubmissionId::new(12),
                Vec::new(),
                &capabilities,
                &cache,
            ),
            Err(MaxwellThreeDLoweringError::ShaderTranslationRequired)
        ));

        program_three_d(&mut channel, 0x020c, 1);
        let cache_before = cache.clone();
        assert!(matches!(
            preflight_maxwell_three_d_operation(
                channel.three_d(),
                &resources,
                MaxwellThreeDOperationTrigger::DrawVertexArray {
                    source,
                    vertex_count: 2,
                },
                None,
                FrontendSubmissionId::new(13),
                Vec::new(),
                &capabilities,
                &cache,
            ),
            Err(MaxwellThreeDLoweringError::UnsupportedAliasedLineWidthSemantics)
        ));
        assert_eq!(cache, cache_before);

        program_three_d(&mut channel, 0x1618, 4);
        program_three_d(&mut channel, 0x0dac, 0x1b02);
        program_three_d(&mut channel, 0x0db0, 0x1b02);
        assert!(matches!(
            preflight_maxwell_three_d_operation(
                channel.three_d(),
                &resources,
                MaxwellThreeDOperationTrigger::DrawVertexArray {
                    source,
                    vertex_count: 3,
                },
                None,
                FrontendSubmissionId::new(14),
                Vec::new(),
                &capabilities,
                &cache,
            ),
            Err(MaxwellThreeDLoweringError::ShaderTranslationRequired)
        ));

        program_three_d(&mut channel, 0x1618, 0);
        assert!(matches!(
            preflight_maxwell_three_d_operation(
                channel.three_d(),
                &resources,
                MaxwellThreeDOperationTrigger::DrawVertexArray {
                    source,
                    vertex_count: 1,
                },
                None,
                FrontendSubmissionId::new(15),
                Vec::new(),
                &capabilities,
                &cache,
            ),
            Err(MaxwellThreeDLoweringError::ShaderTranslationRequired)
        ));

        program_three_d(&mut channel, 0x1618, 4);
        program_three_d(&mut channel, 0x0dac, 0x1b01);
        assert!(matches!(
            preflight_maxwell_three_d_operation(
                channel.three_d(),
                &resources,
                MaxwellThreeDOperationTrigger::DrawVertexArray {
                    source,
                    vertex_count: 3,
                },
                None,
                FrontendSubmissionId::new(16),
                Vec::new(),
                &capabilities,
                &cache,
            ),
            Err(MaxwellThreeDLoweringError::UnsupportedAliasedLineWidthSemantics)
        ));
        assert_eq!(cache, cache_before);

        // Polygon mode does not turn point primitives into line primitives.
        program_three_d(&mut channel, 0x1618, 0);
        assert!(matches!(
            preflight_maxwell_three_d_operation(
                channel.three_d(),
                &resources,
                MaxwellThreeDOperationTrigger::DrawVertexArray {
                    source,
                    vertex_count: 1,
                },
                None,
                FrontendSubmissionId::new(17),
                Vec::new(),
                &capabilities,
                &cache,
            ),
            Err(MaxwellThreeDLoweringError::ShaderTranslationRequired)
        ));
    }

    #[test]
    fn aliased_line_width_selector_does_not_change_clear_semantics() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        program_three_d(&mut channel, 0x020c, 1);
        let clear = packet(0x19d0 / 4, 0x3c);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &clear.packets()[0],
        )
        .unwrap();
        let triggered = &dispatch.operations()[0];
        let resources =
            resolve_maxwell_three_d_resources(triggered.state(), &resource_address_space())
                .unwrap();

        assert!(matches!(
            preflight_maxwell_three_d_operation(
                triggered.state(),
                &resources,
                triggered.trigger(),
                None,
                FrontendSubmissionId::new(18),
                Vec::new(),
                &lowering_capabilities(BackendFeatures::empty()),
                &MaxwellThreeDLoweringCache::default(),
            ),
            Err(MaxwellThreeDLoweringError::IncompleteClear(
                "horizontal rectangle"
            ))
        ));
    }

    #[test]
    fn csaa_only_blocks_draws_when_explicitly_enabled_and_never_blocks_clear() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        program_three_d(&mut channel, 0x121c, 0);
        let resources =
            resolve_maxwell_three_d_resources(channel.three_d(), &resource_address_space())
                .unwrap();
        let cache = MaxwellThreeDLoweringCache::default();

        program_three_d(&mut channel, 0x15b4, 0);
        let source = channel.three_d().coverage().csaa_enable().source().unwrap();
        assert!(matches!(
            preflight_maxwell_three_d_operation(
                channel.three_d(),
                &resources,
                MaxwellThreeDOperationTrigger::DrawVertexArray {
                    source,
                    vertex_count: 3,
                },
                None,
                FrontendSubmissionId::new(10),
                Vec::new(),
                &lowering_capabilities(BackendFeatures::empty()),
                &cache,
            ),
            Err(MaxwellThreeDLoweringError::ShaderTranslationRequired)
        ));

        program_three_d(&mut channel, 0x15b4, 1);
        let source = channel.three_d().coverage().csaa_enable().source().unwrap();
        let cache_before = cache.clone();
        assert!(matches!(
            preflight_maxwell_three_d_operation(
                channel.three_d(),
                &resources,
                MaxwellThreeDOperationTrigger::DrawVertexArray {
                    source,
                    vertex_count: 3,
                },
                None,
                FrontendSubmissionId::new(11),
                Vec::new(),
                &lowering_capabilities(BackendFeatures::empty()),
                &cache,
            ),
            Err(MaxwellThreeDLoweringError::UnsupportedCsaaSemantics)
        ));
        assert_eq!(cache, cache_before);

        let clear = packet(0x19d0 / 4, 0x3c);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &clear.packets()[0],
        )
        .unwrap();
        let triggered = &dispatch.operations()[0];
        assert!(matches!(
            preflight_maxwell_three_d_operation(
                triggered.state(),
                &resources,
                triggered.trigger(),
                None,
                FrontendSubmissionId::new(12),
                Vec::new(),
                &lowering_capabilities(BackendFeatures::empty()),
                &cache,
            ),
            Err(MaxwellThreeDLoweringError::IncompleteClear(
                "horizontal rectangle"
            ))
        ));
        assert_eq!(cache, cache_before);
    }

    #[test]
    fn coverage_line_and_vertex_array_restart_selectors_separate_pipeline_identity() {
        let vertex_allocation = CanonicalAllocation::zeroed(0x4000, 0x1000).unwrap();
        let target_allocation = CanonicalAllocation::zeroed(0x10000, 0x1000).unwrap();
        let mut address_space = resource_address_space();
        let vertex = map_resource(
            &mut address_space,
            vertex_allocation
                .backing_range(MemoryPermissions::READ_WRITE)
                .unwrap(),
            71,
            0,
        )
        .offset()
        .get();
        let target = map_resource(
            &mut address_space,
            target_allocation
                .backing_range(MemoryPermissions::READ_WRITE)
                .unwrap(),
            72,
            0xfe,
        )
        .offset()
        .get();
        let mut channel = channel();
        bind_three_d(&mut channel);
        program_basic_draw_state(&mut channel, vertex);
        program_color_target(&mut channel, 0, target, 0xd5);
        program_three_d(&mut channel, 0x121c, 1);
        let shaders = translated_graphics_shaders();
        let capabilities =
            lowering_capabilities(BackendFeatures::DRAW.union(BackendFeatures::RENDER_PASS));
        let mut cache = MaxwellThreeDLoweringCache::default();
        let draw = packet(0x0d78 / 4, 3);

        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &draw.packets()[0],
        )
        .unwrap();
        let triggered = &dispatch.operations()[0];
        let resources =
            resolve_maxwell_three_d_resources(triggered.state(), &address_space).unwrap();
        let plan = preflight_maxwell_three_d_operation(
            triggered.state(),
            &resources,
            triggered.trigger(),
            Some(&shaders),
            FrontendSubmissionId::new(20),
            Vec::new(),
            &capabilities,
            &cache,
        )
        .unwrap();
        plan.commit_cache(&mut cache).unwrap();
        let pipeline_count = cache.pipeline_count();

        program_three_d(&mut channel, 0x15b4, 0);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &draw.packets()[0],
        )
        .unwrap();
        let triggered = &dispatch.operations()[0];
        let resources =
            resolve_maxwell_three_d_resources(triggered.state(), &address_space).unwrap();
        let plan = preflight_maxwell_three_d_operation(
            triggered.state(),
            &resources,
            triggered.trigger(),
            Some(&shaders),
            FrontendSubmissionId::new(21),
            Vec::new(),
            &capabilities,
            &cache,
        )
        .unwrap();
        assert!(plan.resource_creations().iter().any(|creation| matches!(
            creation,
            nixe_gpu::BackendResourceCreateInfo::Pipeline { .. }
        )));
        assert!(!plan.resource_creations().iter().any(|creation| matches!(
            creation,
            nixe_gpu::BackendResourceCreateInfo::RenderPass { .. }
        )));
        plan.commit_cache(&mut cache).unwrap();
        assert_eq!(cache.pipeline_count(), pipeline_count + 1);

        let pipeline_count = cache.pipeline_count();
        program_three_d(&mut channel, 0x020c, 0);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &draw.packets()[0],
        )
        .unwrap();
        let triggered = &dispatch.operations()[0];
        let resources =
            resolve_maxwell_three_d_resources(triggered.state(), &address_space).unwrap();
        let plan = preflight_maxwell_three_d_operation(
            triggered.state(),
            &resources,
            triggered.trigger(),
            Some(&shaders),
            FrontendSubmissionId::new(22),
            Vec::new(),
            &capabilities,
            &cache,
        )
        .unwrap();
        assert!(plan.resource_creations().iter().any(|creation| matches!(
            creation,
            nixe_gpu::BackendResourceCreateInfo::Pipeline { .. }
        )));
        assert!(!plan.resource_creations().iter().any(|creation| matches!(
            creation,
            nixe_gpu::BackendResourceCreateInfo::RenderPass { .. }
        )));
        plan.commit_cache(&mut cache).unwrap();
        assert_eq!(cache.pipeline_count(), pipeline_count + 1);

        let pipeline_count = cache.pipeline_count();
        program_three_d(&mut channel, 0x0de8, 0);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &draw.packets()[0],
        )
        .unwrap();
        let triggered = &dispatch.operations()[0];
        let resources =
            resolve_maxwell_three_d_resources(triggered.state(), &address_space).unwrap();
        let plan = preflight_maxwell_three_d_operation(
            triggered.state(),
            &resources,
            triggered.trigger(),
            Some(&shaders),
            FrontendSubmissionId::new(23),
            Vec::new(),
            &capabilities,
            &cache,
        )
        .unwrap();
        assert!(plan.resource_creations().iter().any(|creation| matches!(
            creation,
            nixe_gpu::BackendResourceCreateInfo::Pipeline { .. }
        )));
        assert!(!plan.resource_creations().iter().any(|creation| matches!(
            creation,
            nixe_gpu::BackendResourceCreateInfo::RenderPass { .. }
        )));
        plan.commit_cache(&mut cache).unwrap();
        assert_eq!(cache.pipeline_count(), pipeline_count + 1);
    }

    #[test]
    fn disabled_blend_pipeline_identity_tracks_only_effective_active_selectors() {
        let vertex_allocation = CanonicalAllocation::zeroed(0x4000, 0x1000).unwrap();
        let target_allocation = CanonicalAllocation::zeroed(0x10000, 0x1000).unwrap();
        let mut address_space = resource_address_space();
        let vertex = map_resource(
            &mut address_space,
            vertex_allocation
                .backing_range(MemoryPermissions::READ_WRITE)
                .unwrap(),
            74,
            0,
        )
        .offset()
        .get();
        let target = map_resource(
            &mut address_space,
            target_allocation
                .backing_range(MemoryPermissions::READ_WRITE)
                .unwrap(),
            75,
            0xfe,
        )
        .offset()
        .get();
        let mut channel = channel();
        bind_three_d(&mut channel);
        program_basic_draw_state(&mut channel, vertex);
        program_color_target(&mut channel, 0, target, 0xd5);
        program_three_d(&mut channel, 0x121c, 1);
        let shaders = translated_graphics_shaders();
        let capabilities =
            lowering_capabilities(BackendFeatures::DRAW.union(BackendFeatures::RENDER_PASS));
        let mut cache = MaxwellThreeDLoweringCache::default();
        let draw = packet(0x0d78 / 4, 3);

        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &draw.packets()[0],
        )
        .unwrap();
        let triggered = &dispatch.operations()[0];
        let resources =
            resolve_maxwell_three_d_resources(triggered.state(), &address_space).unwrap();
        preflight_maxwell_three_d_operation(
            triggered.state(),
            &resources,
            triggered.trigger(),
            Some(&shaders),
            FrontendSubmissionId::new(30),
            Vec::new(),
            &capabilities,
            &cache,
        )
        .unwrap()
        .commit_cache(&mut cache)
        .unwrap();
        let pipeline_count = cache.pipeline_count();

        // Per-target state and common equations are inactive while common
        // blending is selected and explicitly disabled.
        for (method, argument) in [
            (0x1364, 1),
            (0x1e20, 1),
            (0x133c, 1),
            (0x1340, 1),
            (0x1344, 1),
            (0x1348, 1),
        ] {
            program_three_d(&mut channel, method, argument);
        }
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &draw.packets()[0],
        )
        .unwrap();
        let triggered = &dispatch.operations()[0];
        let resources =
            resolve_maxwell_three_d_resources(triggered.state(), &address_space).unwrap();
        let plan = preflight_maxwell_three_d_operation(
            triggered.state(),
            &resources,
            triggered.trigger(),
            Some(&shaders),
            FrontendSubmissionId::new(31),
            Vec::new(),
            &capabilities,
            &cache,
        )
        .unwrap();
        assert!(!plan.resource_creations().iter().any(|creation| matches!(
            creation,
            nixe_gpu::BackendResourceCreateInfo::Pipeline { .. }
        )));
        plan.commit_cache(&mut cache).unwrap();
        assert_eq!(cache.pipeline_count(), pipeline_count);

        // Selecting per-target state and explicitly disabling the active
        // target changes the effective pipeline state.
        program_three_d(&mut channel, 0x12e4, 1);
        program_three_d(&mut channel, 0x1360, 0);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &draw.packets()[0],
        )
        .unwrap();
        let triggered = &dispatch.operations()[0];
        let resources =
            resolve_maxwell_three_d_resources(triggered.state(), &address_space).unwrap();
        let plan = preflight_maxwell_three_d_operation(
            triggered.state(),
            &resources,
            triggered.trigger(),
            Some(&shaders),
            FrontendSubmissionId::new(32),
            Vec::new(),
            &capabilities,
            &cache,
        )
        .unwrap();
        assert!(plan.resource_creations().iter().any(|creation| matches!(
            creation,
            nixe_gpu::BackendResourceCreateInfo::Pipeline { .. }
        )));
        plan.commit_cache(&mut cache).unwrap();
        assert_eq!(cache.pipeline_count(), pipeline_count + 1);
        let pipeline_count = cache.pipeline_count();

        // Common and unselected target selectors are now inactive.
        program_three_d(&mut channel, 0x135c, 1);
        program_three_d(&mut channel, 0x1364, 0);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &draw.packets()[0],
        )
        .unwrap();
        let triggered = &dispatch.operations()[0];
        let resources =
            resolve_maxwell_three_d_resources(triggered.state(), &address_space).unwrap();
        let plan = preflight_maxwell_three_d_operation(
            triggered.state(),
            &resources,
            triggered.trigger(),
            Some(&shaders),
            FrontendSubmissionId::new(33),
            Vec::new(),
            &capabilities,
            &cache,
        )
        .unwrap();
        assert!(!plan.resource_creations().iter().any(|creation| matches!(
            creation,
            nixe_gpu::BackendResourceCreateInfo::Pipeline { .. }
        )));
        plan.commit_cache(&mut cache).unwrap();
        assert_eq!(cache.pipeline_count(), pipeline_count);
    }

    #[test]
    fn z_compression_selector_is_typed_depth_state_without_an_operation() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        let color_before = channel.three_d().render_targets().color().clone();
        let two_d_before = channel.two_d().clone();
        assert_eq!(
            channel
                .three_d()
                .render_targets()
                .depth_stencil()
                .compression()
                .origin(),
            MaxwellThreeDRegisterOrigin::Unset
        );

        for (argument, expected) in [
            (0, MaxwellThreeDZCompressionMode::Disabled),
            (1, MaxwellThreeDZCompressionMode::Enabled),
        ] {
            let decoded = packet(0x19cc / 4, argument);
            let dispatch = dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0],
            )
            .unwrap();
            let source = dispatch.methods()[0].method().source();
            let register = channel
                .three_d()
                .render_targets()
                .depth_stencil()
                .compression();

            assert_eq!(
                dispatch.methods()[0].metadata().method_name(),
                "SET_Z_COMPRESSION"
            );
            assert_eq!(
                dispatch.methods()[0].effect(),
                MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::RenderTarget(
                    MaxwellThreeDRenderTargetWrite::DepthCompression {
                        value: expected,
                        source,
                    }
                ))
            );
            assert!(dispatch.operations().is_empty());
            assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
            assert_eq!(register.raw(), Some(argument));
            assert_eq!(register.value().copied(), Some(expected));
            assert_eq!(register.source(), Some(source));
            assert_eq!(expected.raw(), argument);
            assert_eq!(channel.three_d().render_targets().color(), &color_before);
            assert_eq!(channel.two_d(), &two_d_before);
        }

        let resources =
            resolve_maxwell_three_d_resources(channel.three_d(), &resource_address_space())
                .unwrap();
        assert!(resources.resources().is_empty());
    }

    #[test]
    fn invalid_z_compression_values_are_rejected_atomically() {
        let mut channel = channel();
        bind_three_d(&mut channel);

        for argument in [2, 3, u32::MAX] {
            let frontend_before = channel.frontend();
            let two_d_before = channel.two_d().clone();
            let three_d_before = channel.three_d().clone();
            let decoded = packet(0x19cc / 4, argument);

            assert!(matches!(
                dispatch_maxwell_engine_packet(
                    &mut channel,
                    FrontendSubmissionId::new(3),
                    &decoded.packets()[0]
                ),
                Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                    source,
                    method_name: "SET_Z_COMPRESSION",
                    reason: "reserved bits are set",
                }) if source.argument() == argument
            ));
            assert_eq!(channel.frontend(), frontend_before);
            assert_eq!(channel.two_d(), &two_d_before);
            assert_eq!(channel.three_d(), &three_d_before);
        }
    }

    #[test]
    fn enabled_z_compression_stops_only_operations_that_consume_depth() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        let compression = packet(0x19cc / 4, 1);
        let compression_dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &compression.packets()[0],
        )
        .unwrap();
        program_three_d(&mut channel, 0x121c, 0);
        let resources =
            resolve_maxwell_three_d_resources(channel.three_d(), &resource_address_space())
                .unwrap();
        let cache = MaxwellThreeDLoweringCache::default();
        let cache_before = cache.clone();

        assert!(matches!(
            preflight_maxwell_three_d_operation(
                channel.three_d(),
                &resources,
                MaxwellThreeDOperationTrigger::DrawVertexArray {
                    source: compression_dispatch.methods()[0].method().source(),
                    vertex_count: 3,
                },
                None,
                FrontendSubmissionId::new(10),
                Vec::new(),
                &lowering_capabilities(BackendFeatures::empty()),
                &cache,
            ),
            Err(MaxwellThreeDLoweringError::ShaderTranslationRequired)
        ));
        assert_eq!(cache, cache_before);

        let clear = packet(0x19d0 / 4, 3);
        let clear_dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &clear.packets()[0],
        )
        .unwrap();
        let triggered = &clear_dispatch.operations()[0];
        assert!(matches!(
            preflight_maxwell_three_d_operation(
                triggered.state(),
                &resources,
                triggered.trigger(),
                None,
                FrontendSubmissionId::new(11),
                Vec::new(),
                &lowering_capabilities(BackendFeatures::empty()),
                &cache,
            ),
            Err(MaxwellThreeDLoweringError::UnsupportedZCompressionSemantics)
        ));
        assert_eq!(cache, cache_before);
    }

    #[test]
    fn color_compression_selectors_are_typed_and_isolated_per_target() {
        for target in 0..MAXWELL_COLOR_TARGET_COUNT as u8 {
            let mut channel = channel();
            bind_three_d(&mut channel);
            let depth_before = channel.three_d().render_targets().depth_stencil().clone();
            let two_d_before = channel.two_d().clone();

            for (argument, expected) in [
                (0, MaxwellThreeDColorCompressionMode::Disabled),
                (1, MaxwellThreeDColorCompressionMode::Enabled),
            ] {
                let method = 0x19e0 + u32::from(target) * 4;
                let decoded = packet(method / 4, argument);
                let dispatch = dispatch_maxwell_engine_packet(
                    &mut channel,
                    FrontendSubmissionId::new(3),
                    &decoded.packets()[0],
                )
                .unwrap();
                let source = dispatch.methods()[0].method().source();
                let targets = channel.three_d().render_targets().color();
                let register = targets[target as usize].compression();

                assert_eq!(
                    dispatch.methods()[0].metadata().method_name(),
                    "SET_COLOR_COMPRESSION"
                );
                assert_eq!(
                    dispatch.methods()[0].effect(),
                    MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::RenderTarget(
                        MaxwellThreeDRenderTargetWrite::ColorCompression {
                            target,
                            value: expected,
                            source,
                        }
                    ))
                );
                assert!(dispatch.operations().is_empty());
                assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
                assert_eq!(register.raw(), Some(argument));
                assert_eq!(register.value().copied(), Some(expected));
                assert_eq!(register.source(), Some(source));
                assert_eq!(expected.raw(), argument);
                for (other, state) in targets.iter().enumerate() {
                    if other != target as usize {
                        assert_eq!(
                            state.compression().origin(),
                            MaxwellThreeDRegisterOrigin::Unset
                        );
                    }
                }
                assert_eq!(
                    channel.three_d().render_targets().depth_stencil(),
                    &depth_before
                );
                assert_eq!(channel.two_d(), &two_d_before);
            }

            let resources =
                resolve_maxwell_three_d_resources(channel.three_d(), &resource_address_space())
                    .unwrap();
            assert!(resources.resources().is_empty());
        }
    }

    #[test]
    fn invalid_color_compression_values_are_rejected_atomically() {
        for target in 0..MAXWELL_COLOR_TARGET_COUNT as u8 {
            for argument in [2, 3, u32::MAX] {
                let mut channel = channel();
                bind_three_d(&mut channel);
                let frontend_before = channel.frontend();
                let two_d_before = channel.two_d().clone();
                let three_d_before = channel.three_d().clone();
                let method = 0x19e0 + u32::from(target) * 4;
                let decoded = packet(method / 4, argument);

                assert!(matches!(
                    dispatch_maxwell_engine_packet(
                        &mut channel,
                        FrontendSubmissionId::new(3),
                        &decoded.packets()[0]
                    ),
                    Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                        source,
                        method_name: "SET_COLOR_COMPRESSION",
                        reason: "reserved bits are set",
                    }) if source.argument() == argument && source.method() == GpuMethodId(method)
                ));
                assert_eq!(channel.frontend(), frontend_before);
                assert_eq!(channel.two_d(), &two_d_before);
                assert_eq!(channel.three_d(), &three_d_before);
            }
        }
    }

    #[test]
    fn color_compression_execution_is_target_specific_and_typed() {
        let allocation = CanonicalAllocation::zeroed(0x10000, 0x1000).unwrap();
        let mut address_space = resource_address_space();
        let mapping = map_resource(
            &mut address_space,
            allocation
                .backing_range(MemoryPermissions::READ_WRITE)
                .unwrap(),
            41,
            0xfe,
        );
        let address = mapping.offset().get();
        let mut channel = channel();
        bind_three_d(&mut channel);
        for (method, argument) in [
            (0x0800, (address >> 32) as u32),
            (0x0804, address as u32),
            (0x0808, 64),
            (0x080c, 32),
            (0x0810, 0xd5),
            (0x0814, 0),
            (0x0818, 1),
            (0x081c, 0),
            (0x0820, 0),
            (0x15d0, 0),
            (0x19e0, 1),
            (0x121c, 1),
            (0x12e4, 0),
            (0x135c, 0),
        ] {
            program_three_d(&mut channel, method, argument);
        }
        let resources =
            resolve_maxwell_three_d_resources(channel.three_d(), &address_space).unwrap();
        assert!(
            resources
                .resources()
                .iter()
                .any(|resource| { resource.role() == MaxwellThreeDResourceRole::ColorTarget(0) })
        );
        let cache = MaxwellThreeDLoweringCache::default();
        let cache_before = cache.clone();

        let draw_source = channel.three_d().render_targets().color()[0]
            .compression()
            .source()
            .unwrap();
        assert!(matches!(
            preflight_maxwell_three_d_operation(
                channel.three_d(),
                &resources,
                MaxwellThreeDOperationTrigger::DrawVertexArray {
                    source: draw_source,
                    vertex_count: 3,
                },
                None,
                FrontendSubmissionId::new(10),
                Vec::new(),
                &lowering_capabilities(BackendFeatures::empty()),
                &cache,
            ),
            Err(MaxwellThreeDLoweringError::UnsupportedColorCompressionSemantics { target: 0 })
        ));
        assert_eq!(cache, cache_before);

        let clear = packet(0x19d0 / 4, 0x3c);
        let clear_dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &clear.packets()[0],
        )
        .unwrap();
        let triggered = &clear_dispatch.operations()[0];
        assert!(matches!(
            preflight_maxwell_three_d_operation(
                triggered.state(),
                &resources,
                triggered.trigger(),
                None,
                FrontendSubmissionId::new(11),
                Vec::new(),
                &lowering_capabilities(BackendFeatures::empty()),
                &cache,
            ),
            Err(MaxwellThreeDLoweringError::UnsupportedColorCompressionSemantics { target: 0 })
        ));
        assert_eq!(cache, cache_before);
    }

    #[test]
    fn color_compression_does_not_block_a_different_clear_target() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        program_three_d(&mut channel, 0x19e4, 1);
        program_three_d(&mut channel, 0x121c, 1);
        let resources =
            resolve_maxwell_three_d_resources(channel.three_d(), &resource_address_space())
                .unwrap();
        let clear = packet(0x19d0 / 4, 0x3c);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &clear.packets()[0],
        )
        .unwrap();
        let triggered = &dispatch.operations()[0];

        assert!(matches!(
            preflight_maxwell_three_d_operation(
                triggered.state(),
                &resources,
                triggered.trigger(),
                None,
                FrontendSubmissionId::new(12),
                Vec::new(),
                &lowering_capabilities(BackendFeatures::empty()),
                &MaxwellThreeDLoweringCache::default(),
            ),
            Err(MaxwellThreeDLoweringError::IncompleteClear(
                "horizontal rectangle"
            ))
        ));
    }

    #[test]
    fn color_target_selection_retains_all_fields_for_counts_zero_through_eight() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        let targets = [7, 0, 6, 1, 5, 2, 4, 3];

        for count in 0..=8 {
            let argument = color_target_selection_raw(count, targets);
            let decoded = packet(0x121c / 4, argument);
            let dispatch = dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0],
            )
            .unwrap();
            let source = dispatch.methods()[0].method().source();
            let register = channel.three_d().render_targets().color_target_selection();
            let selection = register.value().copied().unwrap();

            assert_eq!(
                dispatch.methods()[0].metadata().method_name(),
                "SET_CT_SELECT"
            );
            assert!(dispatch.operations().is_empty());
            assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
            assert_eq!(register.raw(), Some(argument));
            assert_eq!(register.source(), Some(source));
            assert_eq!(selection.target_count(), count);
            assert_eq!(selection.targets(), targets);
            assert_eq!(selection.active_targets(), &targets[..usize::from(count)]);
            assert_eq!(selection.raw(), argument);
        }
    }

    #[test]
    fn malformed_color_target_selection_and_packet_suffix_are_atomic() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        let valid = color_target_selection_raw(2, [1, 0, 7, 6, 5, 4, 3, 2]);
        program_three_d(&mut channel, 0x121c, valid);

        for argument in [9, 15, 0x1000_0000, u32::MAX] {
            let frontend_before = channel.frontend();
            let two_d_before = channel.two_d().clone();
            let three_d_before = channel.three_d().clone();
            let decoded = packet(0x121c / 4, argument);

            assert!(matches!(
                dispatch_maxwell_engine_packet(
                    &mut channel,
                    FrontendSubmissionId::new(3),
                    &decoded.packets()[0]
                ),
                Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                    source,
                    method_name: "SET_CT_SELECT",
                    ..
                }) if source.argument() == argument
            ));
            assert_eq!(channel.frontend(), frontend_before);
            assert_eq!(channel.two_d(), &two_d_before);
            assert_eq!(channel.three_d(), &three_d_before);
        }

        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = incrementing_packet(0x121c / 4, &[1, 0]);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
                if source.method() == GpuMethodId(0x1220)
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }

    #[test]
    fn draw_rejects_missing_disabled_incomplete_and_duplicate_color_routes() {
        let cache = MaxwellThreeDLoweringCache::default();
        let capabilities = lowering_capabilities(BackendFeatures::empty());
        let address_space = resource_address_space();
        let mut channel = channel();
        bind_three_d(&mut channel);

        for (argument, expected) in [
            (
                color_target_selection_raw(1, [3, 0, 0, 0, 0, 0, 0, 0]),
                MaxwellThreeDLoweringError::ColorTargetRouteUnprogrammed { slot: 0, target: 3 },
            ),
            (
                color_target_selection_raw(2, [3, 3, 0, 0, 0, 0, 0, 0]),
                MaxwellThreeDLoweringError::DuplicateColorTargetRoute { target: 3 },
            ),
        ] {
            program_three_d(&mut channel, 0x121c, argument);
            let resources =
                resolve_maxwell_three_d_resources(channel.three_d(), &address_space).unwrap();
            let source = channel
                .three_d()
                .render_targets()
                .color_target_selection()
                .source()
                .unwrap();
            let result = preflight_maxwell_three_d_operation(
                channel.three_d(),
                &resources,
                MaxwellThreeDOperationTrigger::DrawVertexArray {
                    source,
                    vertex_count: 3,
                },
                None,
                FrontendSubmissionId::new(10),
                Vec::new(),
                &capabilities,
                &cache,
            );
            let error = match result {
                Err(error) => error,
                Ok(_) => panic!("invalid color-target route unexpectedly lowered"),
            };
            assert_eq!(error.to_string(), expected.to_string());
        }

        program_three_d(&mut channel, 0x0810, 0);
        program_three_d(&mut channel, 0x121c, 1);
        let resources =
            resolve_maxwell_three_d_resources(channel.three_d(), &address_space).unwrap();
        let source = channel
            .three_d()
            .render_targets()
            .color_target_selection()
            .source()
            .unwrap();
        assert!(matches!(
            preflight_maxwell_three_d_operation(
                channel.three_d(),
                &resources,
                MaxwellThreeDOperationTrigger::DrawVertexArray {
                    source,
                    vertex_count: 3,
                },
                None,
                FrontendSubmissionId::new(11),
                Vec::new(),
                &capabilities,
                &cache,
            ),
            Err(MaxwellThreeDLoweringError::ColorTargetRouteDisabled { slot: 0, target: 0 })
        ));

        program_three_d(&mut channel, 0x0810, 0xd5);
        assert!(matches!(
            preflight_maxwell_three_d_operation(
                channel.three_d(),
                &resources,
                MaxwellThreeDOperationTrigger::DrawVertexArray {
                    source,
                    vertex_count: 3,
                },
                None,
                FrontendSubmissionId::new(12),
                Vec::new(),
                &capabilities,
                &cache,
            ),
            Err(MaxwellThreeDLoweringError::ColorTargetRouteIncomplete { slot: 0, target: 0 })
        ));
    }

    #[test]
    fn three_d_register_write_uses_private_candidate_then_atomic_commit() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        let decoded = packet(0x1518 / 4, 0x3fc0_0000);
        let before = channel.three_d().clone();

        let prepared = preflight_maxwell_engine_packet(
            &channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
        assert_eq!(
            channel.three_d().raster().point_size().origin(),
            MaxwellThreeDRegisterOrigin::Unset
        );
        let after = prepared.three_d_after().clone();
        assert_eq!(
            after.raster().point_size().origin(),
            MaxwellThreeDRegisterOrigin::Programmed
        );
        assert_eq!(after.raster().point_size().raw(), Some(0x3fc0_0000));
        assert_eq!(
            after.raster().point_size().value().copied(),
            Some(MaxwellThreeDPointSize::from_bits(0x3fc0_0000))
        );
        assert_eq!(
            prepared.methods()[0].effect(),
            MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::PointSize {
                value: MaxwellThreeDPointSize::from_bits(0x3fc0_0000),
                source: prepared.methods()[0].method().source(),
            })
        );

        commit_maxwell_engine_packet(&mut channel, &prepared).unwrap();
        assert_ne!(channel.three_d(), &before);
        assert_eq!(channel.three_d(), &after);
    }

    #[test]
    fn enumerated_register_values_are_checked_before_state_changes() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        let valid = packet(0x0d7c / 4, 1);
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &valid.packets()[0],
        )
        .unwrap();
        assert_eq!(
            channel.three_d().viewport().z_clip_range().value().copied(),
            Some(MaxwellThreeDViewportZClipRange::ZeroToPositiveW)
        );

        let before = channel.three_d().clone();
        let invalid = packet(0x0d7c / 4, 2);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &invalid.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::InvalidMethodValue {
                defined_mask: 1,
                ..
            })
        ));
        assert_eq!(channel.three_d(), &before);
    }

    #[test]
    fn render_target_state_distinguishes_unset_disabled_ready_and_profile_unavailable() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        assert_eq!(
            channel.three_d().render_targets().color()[0].readiness(true),
            MaxwellThreeDAttachmentReadiness::Unprogrammed
        );

        let disabled = packet(0x0810 / 4, 0);
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &disabled.packets()[0],
        )
        .unwrap();
        assert_eq!(
            channel.three_d().render_targets().color()[0].readiness(true),
            MaxwellThreeDAttachmentReadiness::Disabled
        );

        let complete =
            incrementing_packet(0x0800 / 4, &[0, 0x0080_0000, 1280, 720, 0xd5, 0, 1, 0, 0]);
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &complete.packets()[0],
        )
        .unwrap();
        let target = &channel.three_d().render_targets().color()[0];
        assert_eq!(
            target.readiness(true),
            MaxwellThreeDAttachmentReadiness::Ready
        );
        assert_eq!(
            target.readiness(false),
            MaxwellThreeDAttachmentReadiness::ProfileUnavailable
        );
        assert_eq!(target.address_lower().value(), Some(&0x0080_0000));
        assert_eq!(target.format().raw(), Some(0xd5));
    }

    #[test]
    fn render_target_encodings_and_cross_register_contradictions_reject_atomically() {
        let mut channel = channel();
        bind_three_d(&mut channel);

        let malformed_layout = packet(0x0814 / 4, 0x1001);
        let before = channel.three_d().clone();
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &malformed_layout.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding { .. })
        ));
        assert_eq!(channel.three_d(), &before);

        let three_dimensional = packet(0x0814 / 4, 0x1_0000);
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &three_dimensional.packets()[0],
        )
        .unwrap();
        let before_layer = channel.three_d().clone();
        let nonzero_layer = packet(0x0820 / 4, 1);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &nonzero_layer.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::ContradictoryState { .. })
        ));
        assert_eq!(channel.three_d(), &before_layer);
    }

    #[test]
    fn clear_and_fixed_function_state_preserve_typed_values_and_sources() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        for (method, argument) in [
            (0x0d6c, (640_u32 << 16) | 10),
            (0x0d70, (480_u32 << 16) | 20),
            (0x0d80, 0x3f80_0000),
            (0x0d90, 0x3f00_0000),
            (0x0da0, 0x7f),
            (0x19d0, 0x3c),
            (0x12cc, 1),
            (0x130c, 0x203),
            (0x1380, 1),
            (0x1384, 0x1e00),
            (0x1390, 0x207),
            (0x1598, 0x1e01),
            (0x15a4, 0x201),
            (0x133c, 1),
            (0x1340, 0x8006),
            (0x1344, 0x4302),
            (0x1e00, 1),
            (0x1e04, 0x8006),
            (0x1e08, 0x4302),
            (0x1918, 1),
            (0x191c, 0x901),
            (0x1920, 0x405),
            (0x1a00, 0x1101),
        ] {
            let decoded = packet(method / 4, argument);
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0],
            )
            .unwrap();
        }

        let state = channel.three_d();
        let clear = state.render_targets().clear();
        assert_eq!(clear.horizontal().value().unwrap().min, 10);
        assert_eq!(clear.horizontal().value().unwrap().max, 640);
        assert_eq!(clear.last_surface().value().unwrap().color_mask(), 0xf);
        assert_eq!(
            clear.last_surface().source().unwrap().method(),
            GpuMethodId(0x19d0)
        );
        assert_eq!(
            state
                .fixed_function()
                .register(MaxwellThreeDFixedFunctionRegister::DepthCompare)
                .value(),
            Some(&MaxwellThreeDFixedFunctionValue::Compare(
                MaxwellThreeDCompareOp::LessEqual
            ))
        );
        assert_eq!(
            state.fixed_function().color_mask()[0].value(),
            Some(&MaxwellThreeDColorMask {
                red: true,
                green: false,
                blue: true,
                alpha: true,
            })
        );
        assert_eq!(
            state
                .fixed_function()
                .register(MaxwellThreeDFixedFunctionRegister::FrontStencilFail)
                .value(),
            Some(&MaxwellThreeDFixedFunctionValue::StencilOp(
                MaxwellThreeDStencilOp::Keep
            ))
        );
        assert_eq!(
            state.fixed_function().per_target_blend()[0][1].value(),
            Some(&MaxwellThreeDFixedFunctionValue::BlendOp(
                MaxwellThreeDBlendOp::Add
            ))
        );
    }

    #[test]
    fn malformed_scissor_suffix_discards_the_whole_candidate() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        let decoded = incrementing_packet(0x0e00 / 4, &[1, (100 << 16) | 5, (10 << 16) | 20]);
        let before = channel.three_d().clone();
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding { .. })
        ));
        assert_eq!(channel.three_d(), &before);
    }

    #[test]
    fn vertex_stream_attributes_and_begin_state_remain_unresolved_and_typed() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        for (method, argument) in [
            (0x1c00, 0x1010),
            (0x1c04, 0),
            (0x1c08, 0x2000),
            (0x1c0c, 1),
            (0x1f00, 0),
            (0x1f04, 0x20ff),
            (0x1160, 0x3820_0000),
            (0x1164, 0x40),
            (0x0d74, 7),
            (0x1618, 4),
            (0x1970, 4),
        ] {
            let decoded = packet(method / 4, argument);
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0],
            )
            .unwrap();
        }

        let input = channel.three_d().vertex_input();
        let stream = &input.streams()[0];
        assert_eq!(stream.address().unwrap().get(), 0x2000);
        assert_eq!(stream.limit().unwrap().get(), 0x20ff);
        assert_eq!(stream.format().value().unwrap().stride(), 16);
        assert!(stream.format().value().unwrap().enabled());
        let attribute = input.attributes()[0].value().unwrap();
        assert!(attribute.enabled());
        assert_eq!(attribute.stream(), 0);
        assert_eq!(attribute.component_widths().unwrap().byte_size(), 16);
        assert!(!input.attributes()[1].value().unwrap().enabled());
        assert_eq!(input.primitive().vertex_array_start().value(), Some(&7));
        assert_eq!(input.primitive().begin().value().unwrap().topology(), 4);
        assert_eq!(input.primitive().topology().value().unwrap().raw(), 4);
    }

    #[test]
    fn malformed_vertex_suffix_and_index_relationships_reject_atomically() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        let malformed = incrementing_packet(0x1c00 / 4, &[0x1010, 0, 0x2000, 1, 0x2000]);
        let before = channel.three_d().clone();
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &malformed.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding { .. })
        ));
        assert_eq!(channel.three_d(), &before);

        for (method, argument) in [(0x17c8, 0), (0x17cc, 0x1000), (0x17d0, 0), (0x17d4, 0x100e)] {
            let decoded = packet(method / 4, argument);
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0],
            )
            .unwrap();
        }
        let before_size = channel.three_d().clone();
        let size = packet(0x17d8 / 4, 2);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &size.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::ContradictoryState { .. })
        ));
        assert_eq!(channel.three_d(), &before_size);
    }

    #[test]
    fn shader_bindings_snapshot_selectors_and_preserve_stage_visibility() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        for (method, argument) in [
            (0x2000, 0x11),
            (0x2010, 2),
            (0x2380, 0x100),
            (0x2384, 0),
            (0x2388, 0x4000),
            (0x2450, 0x31),
            (0x1574, 0),
            (0x1578, 0x8000),
            (0x157c, 3),
            (0x155c, 0),
            (0x1560, 0xa000),
            (0x1564, 7),
            (0x1234, 1),
            (0x2608, 3),
        ] {
            let decoded = packet(method / 4, argument);
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0],
            )
            .unwrap();
        }

        let bindings = channel.three_d().shader_bindings();
        let constant = bindings.groups()[2].constant_buffers()[3].unwrap();
        assert!(constant.enabled());
        assert_eq!(constant.address().unwrap().get(), 0x4000);
        assert_eq!(constant.size(), Some(0x100));
        assert!(bindings.stage_visibility(2)[MaxwellThreeDShaderStage::Vertex as usize]);
        assert_eq!(bindings.texture_headers().address().unwrap().get(), 0x8000);
        assert_eq!(bindings.samplers().maximum_index().value(), Some(&7));
        assert_eq!(
            bindings.sampler_binding().value(),
            Some(&MaxwellThreeDSamplerBindingMode::ViaTextureHeader)
        );
    }

    #[test]
    fn incomplete_bindings_and_misaligned_descriptor_tables_do_not_commit() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        let before = channel.three_d().clone();
        let bind = packet(0x2410 / 4, 1);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &bind.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding { .. })
        ));
        assert_eq!(channel.three_d(), &before);

        for (method, argument) in [(0x1574, 0), (0x1578, 0x8001)] {
            let decoded = packet(method / 4, argument);
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0],
            )
            .unwrap();
        }
        let before_maximum = channel.three_d().clone();
        let maximum = packet(0x157c / 4, 0);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &maximum.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::ContradictoryState { .. })
        ));
        assert_eq!(channel.three_d(), &before_maximum);
    }

    #[test]
    fn resource_snapshot_resolves_buffers_aliases_and_content_generations() {
        let allocation = CanonicalAllocation::zeroed(0x4000, 0x1000).unwrap();
        let backing = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let mut address_space = resource_address_space();
        let first = map_resource(&mut address_space, backing.clone(), 11, 0);
        let alias = map_resource(&mut address_space, backing, 11, 0);
        let mut channel = channel();
        bind_three_d(&mut channel);

        let vertex = first.offset().get();
        for (method, argument) in [
            (0x1c00, 0x1010),
            (0x1c04, (vertex >> 32) as u32),
            (0x1c08, vertex as u32),
            (0x1f00, (vertex >> 32) as u32),
            (0x1f04, (vertex + 0xff) as u32),
        ] {
            program_three_d(&mut channel, method, argument);
        }
        let constant = alias.offset().get();
        for (method, argument) in [
            (0x2380, 0x100),
            (0x2384, (constant >> 32) as u32),
            (0x2388, constant as u32),
            (0x2410, 1),
        ] {
            program_three_d(&mut channel, method, argument);
        }

        allocation.write(0, &[0x5a]).unwrap();
        let mut resolved =
            resolve_maxwell_three_d_resources(channel.three_d(), &address_space).unwrap();
        assert_eq!(resolved.resources().len(), 2);
        assert_eq!(resolved.aliases().len(), 1);
        let MaxwellThreeDResolvedResource::Buffer(vertex) = &resolved.resources()[0] else {
            panic!("vertex stream must resolve as a buffer");
        };
        assert_eq!(vertex.role(), MaxwellThreeDResourceRole::VertexStream(0));
        assert_eq!(vertex.view().size(), 0x100);
        assert_eq!(
            vertex.view().backing().range().segments()[0].content_generation(),
            allocation
                .backing_range(MemoryPermissions::READ_WRITE)
                .unwrap()
                .segments()[0]
                .content_generation()
        );
        assert_eq!(
            resolved.mark_image_dirty(0),
            Err(MaxwellThreeDResourceError::NotAnImage { resource: 0 })
        );

        address_space.unmap(first.offset()).unwrap();
        assert!(matches!(
            resolved.validate_mappings(&address_space),
            Err(MaxwellThreeDResourceError::StaleMapping { mapping, .. })
                if mapping == first.id()
        ));
    }

    #[test]
    fn resource_snapshot_preserves_block_linear_targets_and_tracks_dirty_images() {
        let allocation = CanonicalAllocation::zeroed(0x10000, 0x1000).unwrap();
        let backing = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let mut address_space = resource_address_space();
        let mapping = map_resource(&mut address_space, backing, 12, 0xfe);
        let address = mapping.offset().get();
        let mut channel = channel();
        bind_three_d(&mut channel);
        for (method, argument) in [
            (0x0800, (address >> 32) as u32),
            (0x0804, address as u32),
            (0x0808, 64),
            (0x080c, 32),
            (0x0810, 0xd5),
            (0x0814, 0),
            (0x0818, 1),
            (0x081c, 0),
            (0x0820, 0),
            (0x15d0, 0),
            (0x121c, 1),
        ] {
            program_three_d(&mut channel, method, argument);
        }

        let mut resolved =
            resolve_maxwell_three_d_resources(channel.three_d(), &address_space).unwrap();
        assert_eq!(resolved.resources().len(), 1);
        let MaxwellThreeDResolvedResource::Image(image) = &resolved.resources()[0] else {
            panic!("color target must resolve as an image");
        };
        assert_eq!(
            image.description().format(),
            nixe_gpu::ImageFormat::Rgba8Unorm
        );
        assert_eq!(image.source().size(), 0x2000);
        assert_eq!(
            image.guest_layout().layout(),
            nixe_gpu::ImageMemoryLayout::BlockLinear(nixe_gpu::BlockLinearLayout {
                block_width_log2: 0,
                block_height_log2: 0,
                block_depth_log2: 0,
                layer_stride: 0x2000,
            })
        );
        assert!(!resolved.dirty_subresources().contains(0));
        resolved.mark_image_dirty(0).unwrap();
        assert!(resolved.dirty_subresources().contains(0));
        let dirty = resolved.dirty_subresources().entries().next().unwrap();
        assert_eq!(dirty.resource(), 0);
        assert_eq!(dirty.subresources().plane, 0);
        assert_eq!(dirty.subresources().mip_level, 0);
        assert_eq!(dirty.subresources().base_layer, 0);
        assert_eq!(dirty.subresources().layer_count, 1);
        resolved.clear_image_dirty(0);
        assert!(!resolved.dirty_subresources().contains(0));
    }

    #[test]
    fn resource_resolution_rejects_the_complete_snapshot_atomically() {
        let allocation = CanonicalAllocation::zeroed(0x10000, 0x1000).unwrap();
        let backing = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let mut address_space = resource_address_space();
        let mapping = map_resource(&mut address_space, backing, 13, 0);
        let address = mapping.offset().get();
        let mut state_channel = channel();
        bind_three_d(&mut state_channel);
        for (method, argument) in [
            (0x1c00, 0x1010),
            (0x1c04, (address >> 32) as u32),
            (0x1c08, address as u32),
            (0x1f00, (address >> 32) as u32),
            (0x1f04, (address + 0xff) as u32),
            (0x1574, 0),
            (0x1578, 0x4000),
            (0x157c, 1),
        ] {
            program_three_d(&mut state_channel, method, argument);
        }
        let state = state_channel.three_d().clone();

        assert!(matches!(
            resolve_maxwell_three_d_resources(&state, &address_space),
            Err(MaxwellThreeDResourceError::Resolution {
                role: MaxwellThreeDResourceRole::TextureHeaders,
                ..
            })
        ));
        assert_eq!(state_channel.three_d(), &state);

        let mut image_channel = channel();
        bind_three_d(&mut image_channel);
        for (method, argument) in [
            (0x0800, (address >> 32) as u32),
            (0x0804, address as u32),
            (0x0808, 64),
            (0x080c, 32),
            (0x0810, 0xd5),
            (0x0814, 0),
            (0x0818, 1),
            (0x081c, 0),
            (0x0820, 0),
            (0x15d0, 0),
            (0x121c, 1),
        ] {
            program_three_d(&mut image_channel, method, argument);
        }
        assert!(matches!(
            resolve_maxwell_three_d_resources(image_channel.three_d(), &address_space),
            Err(MaxwellThreeDResourceError::UnsupportedKind {
                role: MaxwellThreeDResourceRole::ColorTarget(0),
                expected: 0xfe,
                actual: 0,
            })
        ));
    }

    #[test]
    fn clear_trigger_lowers_atomically_and_cache_publication_is_generation_checked() {
        let allocation = CanonicalAllocation::zeroed(0x10000, 0x1000).unwrap();
        let mut address_space = resource_address_space();
        let mapping = map_resource(
            &mut address_space,
            allocation
                .backing_range(MemoryPermissions::READ_WRITE)
                .unwrap(),
            21,
            0xfe,
        );
        let address = mapping.offset().get();
        let mut channel = channel();
        bind_three_d(&mut channel);
        for (method, argument) in [
            (0x0800, (address >> 32) as u32),
            (0x0804, address as u32),
            (0x0808, 64),
            (0x080c, 32),
            (0x0810, 0xd5),
            (0x0814, 0),
            (0x0818, 1),
            (0x081c, 0),
            (0x0820, 0),
            (0x15d0, 0),
            (0x0d6c, 64 << 16),
            (0x0d70, 32 << 16),
            (0x0d80, 0x3f80_0000),
            (0x0d84, 0x3f00_0000),
            (0x0d88, 0),
            (0x0d8c, 0x3f80_0000),
            // Draw routing intentionally names an unprogrammed target. The
            // clear trigger below carries and consumes its own MRT selector.
            (
                0x121c,
                color_target_selection_raw(1, [7, 0, 0, 0, 0, 0, 0, 0]),
            ),
        ] {
            program_three_d(&mut channel, method, argument);
        }
        let clear = packet(0x19d0 / 4, 0x3c);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &clear.packets()[0],
        )
        .unwrap();
        assert_eq!(dispatch.operations().len(), 1);
        assert!(matches!(
            dispatch.methods()[0].effect(),
            MaxwellEngineMethodEffect::ThreeDStateAndTrigger { .. }
        ));
        let triggered = &dispatch.operations()[0];
        let resources =
            resolve_maxwell_three_d_resources(triggered.state(), &address_space).unwrap();
        let capabilities = lowering_capabilities(BackendFeatures::CLEAR);
        let mut cache = MaxwellThreeDLoweringCache::default();
        let first = preflight_maxwell_three_d_operation(
            triggered.state(),
            &resources,
            triggered.trigger(),
            None,
            FrontendSubmissionId::new(10),
            Vec::new(),
            &capabilities,
            &cache,
        )
        .unwrap();
        let stale = preflight_maxwell_three_d_operation(
            triggered.state(),
            &resources,
            triggered.trigger(),
            None,
            FrontendSubmissionId::new(11),
            Vec::new(),
            &capabilities,
            &cache,
        )
        .unwrap();
        assert_eq!(first.resource_creations().len(), 2);
        assert_eq!(first.dirty_images(), &[0]);
        assert!(matches!(
            first.submission().operations()[0].command(),
            GpuCommand::Clear(_)
        ));
        let committed = first.commit_cache(&mut cache).unwrap();
        assert_eq!(committed.dirty_images(), &[0]);
        assert_eq!(cache.revision(), 1);
        assert_eq!(cache.view_count(), 1);
        assert!(matches!(
            stale.commit_cache(&mut cache),
            Err(MaxwellThreeDLoweringError::CacheChanged {
                expected: 0,
                actual: 1
            })
        ));

        allocation.write(0, &[0x7f]).unwrap();
        let refreshed =
            resolve_maxwell_three_d_resources(triggered.state(), &address_space).unwrap();
        let refreshed_plan = preflight_maxwell_three_d_operation(
            triggered.state(),
            &refreshed,
            triggered.trigger(),
            None,
            FrontendSubmissionId::new(13),
            Vec::new(),
            &capabilities,
            &cache,
        )
        .unwrap();
        assert_eq!(refreshed_plan.resource_creations().len(), 1);
        assert!(matches!(
            refreshed_plan.resource_invalidations(),
            [nixe_gpu::ResourceDependency::Image(_)]
        ));
        refreshed_plan.commit_cache(&mut cache).unwrap();
        assert_eq!(cache.view_count(), 1);

        let insufficient = lowering_capabilities(BackendFeatures::empty());
        let before = cache.clone();
        assert!(matches!(
            preflight_maxwell_three_d_operation(
                triggered.state(),
                &resources,
                triggered.trigger(),
                None,
                FrontendSubmissionId::new(12),
                Vec::new(),
                &insufficient,
                &cache,
            ),
            Err(MaxwellThreeDLoweringError::Capability(_))
        ));
        assert_eq!(cache, before);
    }

    #[test]
    fn draw_lowering_requires_t10_evidence_and_emits_complete_neutral_pass() {
        let vertex_allocation = CanonicalAllocation::zeroed(0x4000, 0x1000).unwrap();
        let target_allocation = CanonicalAllocation::zeroed(0x10000, 0x1000).unwrap();
        let mut address_space = resource_address_space();
        let vertex_mapping = map_resource(
            &mut address_space,
            vertex_allocation
                .backing_range(MemoryPermissions::READ_WRITE)
                .unwrap(),
            31,
            0,
        );
        let target_mapping = map_resource(
            &mut address_space,
            target_allocation
                .backing_range(MemoryPermissions::READ_WRITE)
                .unwrap(),
            32,
            0xfe,
        );
        let vertex = vertex_mapping.offset().get();
        let target = target_mapping.offset().get();
        let mut channel = channel();
        bind_three_d(&mut channel);
        for (method, argument) in [
            (0x1c00, 0x1010),
            (0x1c04, (vertex >> 32) as u32),
            (0x1c08, vertex as u32),
            (0x1f00, (vertex >> 32) as u32),
            (0x1f04, (vertex + 0xff) as u32),
            (0x1160, 0x3820_0000),
            (0x0d74, 0),
            (0x1618, 4),
            (0x1970, 4),
            (0x12e4, 0),
            (0x135c, 0),
            (0x2000, 0x11),
            (0x2010, 0),
            (0x2040, 0x51),
            (0x2050, 1),
            (0x0800, (target >> 32) as u32),
            (0x0804, target as u32),
            (0x0808, 64),
            (0x080c, 32),
            (0x0810, 0xd5),
            (0x0814, 0),
            (0x0818, 1),
            (0x081c, 0),
            (0x0820, 0),
            (0x15d0, 0),
            (0x121c, 1),
        ] {
            program_three_d(&mut channel, method, argument);
        }
        let draw = packet(0x0d78 / 4, 3);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &draw.packets()[0],
        )
        .unwrap();
        let triggered = &dispatch.operations()[0];
        let resources =
            resolve_maxwell_three_d_resources(triggered.state(), &address_space).unwrap();
        let capabilities =
            lowering_capabilities(BackendFeatures::DRAW.union(BackendFeatures::RENDER_PASS));
        let mut cache = MaxwellThreeDLoweringCache::default();
        assert!(matches!(
            preflight_maxwell_three_d_operation(
                triggered.state(),
                &resources,
                triggered.trigger(),
                None,
                FrontendSubmissionId::new(20),
                Vec::new(),
                &capabilities,
                &cache,
            ),
            Err(MaxwellThreeDLoweringError::ShaderTranslationRequired)
        ));

        let shaders = MaxwellThreeDTranslatedShaders::new(
            vec![
                MaxwellThreeDTranslatedShader::new(ShaderStage::Vertex, ShaderId::new(1), 7),
                MaxwellThreeDTranslatedShader::new(ShaderStage::Fragment, ShaderId::new(2), 9),
            ],
            Vec::new(),
        )
        .unwrap();
        let plan = preflight_maxwell_three_d_operation(
            triggered.state(),
            &resources,
            triggered.trigger(),
            Some(&shaders),
            FrontendSubmissionId::new(20),
            Vec::new(),
            &capabilities,
            &cache,
        )
        .unwrap();
        let commands = plan
            .submission()
            .operations()
            .iter()
            .map(|operation| operation.command())
            .collect::<Vec<_>>();
        assert!(matches!(
            commands.as_slice(),
            [
                GpuCommand::RenderPass(RenderPassOperation::Begin { .. }),
                GpuCommand::Draw(_),
                GpuCommand::RenderPass(RenderPassOperation::End { .. })
            ]
        ));
        assert_eq!(plan.dirty_images().len(), 1);
        assert!(plan.resource_creations().iter().any(|creation| matches!(
            creation,
            nixe_gpu::BackendResourceCreateInfo::Pipeline { .. }
        )));
        plan.commit_cache(&mut cache).unwrap();

        vertex_allocation.write(0, &[1, 2, 3, 4]).unwrap();
        let vertex_refreshed =
            resolve_maxwell_three_d_resources(triggered.state(), &address_space).unwrap();
        let vertex_plan = preflight_maxwell_three_d_operation(
            triggered.state(),
            &vertex_refreshed,
            triggered.trigger(),
            Some(&shaders),
            FrontendSubmissionId::new(21),
            Vec::new(),
            &capabilities,
            &cache,
        )
        .unwrap();
        assert!(matches!(
            vertex_plan.resource_invalidations(),
            [nixe_gpu::ResourceDependency::Buffer(_)]
        ));
        assert!(
            !vertex_plan
                .resource_creations()
                .iter()
                .any(|creation| matches!(
                    creation,
                    nixe_gpu::BackendResourceCreateInfo::Pipeline { .. }
                ))
        );
        vertex_plan.commit_cache(&mut cache).unwrap();

        target_allocation.write(0, &[5, 6, 7, 8]).unwrap();
        let target_refreshed =
            resolve_maxwell_three_d_resources(triggered.state(), &address_space).unwrap();
        let target_plan = preflight_maxwell_three_d_operation(
            triggered.state(),
            &target_refreshed,
            triggered.trigger(),
            Some(&shaders),
            FrontendSubmissionId::new(22),
            Vec::new(),
            &capabilities,
            &cache,
        )
        .unwrap();
        assert!(
            target_plan
                .resource_invalidations()
                .iter()
                .any(|dependency| matches!(dependency, nixe_gpu::ResourceDependency::Pipeline(_)))
        );
        assert!(
            target_plan
                .resource_invalidations()
                .iter()
                .any(|dependency| matches!(dependency, nixe_gpu::ResourceDependency::Image(_)))
        );
        assert!(
            target_plan
                .resource_creations()
                .iter()
                .any(|creation| matches!(
                    creation,
                    nixe_gpu::BackendResourceCreateInfo::Pipeline { .. }
                ))
        );
    }

    #[test]
    fn draw_routing_is_ordered_exact_and_separates_render_pass_and_pipeline_caches() {
        let vertex_allocation = CanonicalAllocation::zeroed(0x4000, 0x1000).unwrap();
        let target_zero_allocation = CanonicalAllocation::zeroed(0x10000, 0x1000).unwrap();
        let target_one_allocation = CanonicalAllocation::zeroed(0x10000, 0x1000).unwrap();
        let mut address_space = resource_address_space();
        let vertex = map_resource(
            &mut address_space,
            vertex_allocation
                .backing_range(MemoryPermissions::READ_WRITE)
                .unwrap(),
            51,
            0,
        )
        .offset()
        .get();
        let target_zero = map_resource(
            &mut address_space,
            target_zero_allocation
                .backing_range(MemoryPermissions::READ_WRITE)
                .unwrap(),
            52,
            0xfe,
        )
        .offset()
        .get();
        let target_one = map_resource(
            &mut address_space,
            target_one_allocation
                .backing_range(MemoryPermissions::READ_WRITE)
                .unwrap(),
            53,
            0xfe,
        )
        .offset()
        .get();
        let mut channel = channel();
        bind_three_d(&mut channel);
        program_basic_draw_state(&mut channel, vertex);
        program_color_target(&mut channel, 0, target_zero, 0xd5);
        program_color_target(&mut channel, 1, target_one, 0xcf);
        let shaders = translated_graphics_shaders();
        let capabilities =
            lowering_capabilities(BackendFeatures::DRAW.union(BackendFeatures::RENDER_PASS));
        let mut cache = MaxwellThreeDLoweringCache::default();

        program_three_d(
            &mut channel,
            0x121c,
            color_target_selection_raw(1, [1, 7, 6, 5, 4, 3, 2, 0]),
        );
        let draw = packet(0x0d78 / 4, 3);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &draw.packets()[0],
        )
        .unwrap();
        let triggered = &dispatch.operations()[0];
        let resources =
            resolve_maxwell_three_d_resources(triggered.state(), &address_space).unwrap();
        let target_one_index = resources
            .resources()
            .iter()
            .position(|resource| resource.role() == MaxwellThreeDResourceRole::ColorTarget(1))
            .unwrap();
        let target_zero_index = resources
            .resources()
            .iter()
            .position(|resource| resource.role() == MaxwellThreeDResourceRole::ColorTarget(0))
            .unwrap();
        let plan = preflight_maxwell_three_d_operation(
            triggered.state(),
            &resources,
            triggered.trigger(),
            Some(&shaders),
            FrontendSubmissionId::new(30),
            Vec::new(),
            &capabilities,
            &cache,
        )
        .unwrap();
        assert_eq!(plan.dirty_images(), &[target_one_index]);
        assert_eq!(
            plan.resource_creations()
                .iter()
                .filter(|creation| matches!(
                    creation,
                    nixe_gpu::BackendResourceCreateInfo::Image { .. }
                ))
                .count(),
            1
        );
        let first_attachments = plan
            .submission()
            .operations()
            .iter()
            .find_map(|operation| match operation.command() {
                GpuCommand::RenderPass(RenderPassOperation::Begin { attachments, .. }) => {
                    Some(attachments.as_ref())
                }
                _ => None,
            })
            .unwrap();
        assert_eq!(first_attachments.len(), 1);
        assert_eq!(first_attachments[0].format, ImageFormat::Bgra8Unorm);
        plan.commit_cache(&mut cache).unwrap();

        target_zero_allocation.write(0, &[1, 2, 3, 4]).unwrap();
        let unselected_refresh =
            resolve_maxwell_three_d_resources(triggered.state(), &address_space).unwrap();
        let unselected_plan = preflight_maxwell_three_d_operation(
            triggered.state(),
            &unselected_refresh,
            triggered.trigger(),
            Some(&shaders),
            FrontendSubmissionId::new(33),
            Vec::new(),
            &capabilities,
            &cache,
        )
        .unwrap();
        assert!(unselected_plan.resource_invalidations().is_empty());
        assert!(
            !unselected_plan.resource_creations().iter().any(|creation| {
                matches!(
                    creation,
                    nixe_gpu::BackendResourceCreateInfo::Image { .. }
                        | nixe_gpu::BackendResourceCreateInfo::Pipeline { .. }
                        | nixe_gpu::BackendResourceCreateInfo::RenderPass { .. }
                )
            })
        );

        program_three_d(
            &mut channel,
            0x121c,
            color_target_selection_raw(2, [1, 0, 7, 6, 5, 4, 3, 2]),
        );
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &draw.packets()[0],
        )
        .unwrap();
        let triggered = &dispatch.operations()[0];
        let resources =
            resolve_maxwell_three_d_resources(triggered.state(), &address_space).unwrap();
        let plan = preflight_maxwell_three_d_operation(
            triggered.state(),
            &resources,
            triggered.trigger(),
            Some(&shaders),
            FrontendSubmissionId::new(31),
            Vec::new(),
            &capabilities,
            &cache,
        )
        .unwrap();
        assert_eq!(plan.dirty_images(), &[target_one_index, target_zero_index]);
        let attachments = plan
            .submission()
            .operations()
            .iter()
            .find_map(|operation| match operation.command() {
                GpuCommand::RenderPass(RenderPassOperation::Begin { attachments, .. }) => {
                    Some(attachments.as_ref())
                }
                _ => None,
            })
            .unwrap();
        assert_eq!(
            attachments
                .iter()
                .map(|attachment| attachment.format)
                .collect::<Vec<_>>(),
            [ImageFormat::Bgra8Unorm, ImageFormat::Rgba8Unorm]
        );
        plan.commit_cache(&mut cache).unwrap();
        let pipeline_count = cache.pipeline_count();

        program_three_d(
            &mut channel,
            0x121c,
            color_target_selection_raw(2, [0, 1, 7, 6, 5, 4, 3, 2]),
        );
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &draw.packets()[0],
        )
        .unwrap();
        let triggered = &dispatch.operations()[0];
        let resources =
            resolve_maxwell_three_d_resources(triggered.state(), &address_space).unwrap();
        let plan = preflight_maxwell_three_d_operation(
            triggered.state(),
            &resources,
            triggered.trigger(),
            Some(&shaders),
            FrontendSubmissionId::new(32),
            Vec::new(),
            &capabilities,
            &cache,
        )
        .unwrap();
        assert_eq!(plan.dirty_images(), &[target_zero_index, target_one_index]);
        assert!(plan.resource_creations().iter().any(|creation| matches!(
            creation,
            nixe_gpu::BackendResourceCreateInfo::RenderPass { description, .. }
                if description.attachments().iter().map(|attachment| attachment.format).eq([
                    ImageFormat::Rgba8Unorm,
                    ImageFormat::Bgra8Unorm,
                ])
        )));
        assert!(plan.resource_creations().iter().any(|creation| matches!(
            creation,
            nixe_gpu::BackendResourceCreateInfo::Pipeline { .. }
        )));
        plan.commit_cache(&mut cache).unwrap();
        assert_eq!(cache.pipeline_count(), pipeline_count + 1);
    }

    #[test]
    fn draw_alias_validation_ignores_unselected_targets_and_rejects_selected_aliases() {
        let vertex_allocation = CanonicalAllocation::zeroed(0x4000, 0x1000).unwrap();
        let shared_target_allocation = CanonicalAllocation::zeroed(0x10000, 0x1000).unwrap();
        let shared_backing = shared_target_allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let mut address_space = resource_address_space();
        let vertex = map_resource(
            &mut address_space,
            vertex_allocation
                .backing_range(MemoryPermissions::READ_WRITE)
                .unwrap(),
            61,
            0,
        )
        .offset()
        .get();
        let target_zero = map_resource(&mut address_space, shared_backing.clone(), 62, 0xfe)
            .offset()
            .get();
        let target_one = map_resource(&mut address_space, shared_backing, 63, 0xfe)
            .offset()
            .get();
        let mut channel = channel();
        bind_three_d(&mut channel);
        program_basic_draw_state(&mut channel, vertex);
        program_color_target(&mut channel, 0, target_zero, 0xd5);
        program_color_target(&mut channel, 1, target_one, 0xd5);
        let shaders = translated_graphics_shaders();
        let capabilities =
            lowering_capabilities(BackendFeatures::DRAW.union(BackendFeatures::RENDER_PASS));
        let cache = MaxwellThreeDLoweringCache::default();
        let draw = packet(0x0d78 / 4, 3);

        program_three_d(&mut channel, 0x121c, 1);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &draw.packets()[0],
        )
        .unwrap();
        let triggered = &dispatch.operations()[0];
        let resources =
            resolve_maxwell_three_d_resources(triggered.state(), &address_space).unwrap();
        assert!(resources.aliases().iter().any(|alias| {
            let first = resources.resources()[alias.first()].role();
            let second = resources.resources()[alias.second()].role();
            [first, second].contains(&MaxwellThreeDResourceRole::ColorTarget(0))
                && [first, second].contains(&MaxwellThreeDResourceRole::ColorTarget(1))
        }));
        preflight_maxwell_three_d_operation(
            triggered.state(),
            &resources,
            triggered.trigger(),
            Some(&shaders),
            FrontendSubmissionId::new(40),
            Vec::new(),
            &capabilities,
            &cache,
        )
        .unwrap();

        program_three_d(
            &mut channel,
            0x121c,
            color_target_selection_raw(2, [0, 1, 0, 0, 0, 0, 0, 0]),
        );
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &draw.packets()[0],
        )
        .unwrap();
        let triggered = &dispatch.operations()[0];
        let resources =
            resolve_maxwell_three_d_resources(triggered.state(), &address_space).unwrap();
        assert!(matches!(
            preflight_maxwell_three_d_operation(
                triggered.state(),
                &resources,
                triggered.trigger(),
                Some(&shaders),
                FrontendSubmissionId::new(41),
                Vec::new(),
                &capabilities,
                &cache,
            ),
            Err(MaxwellThreeDLoweringError::AliasedDrawResources { .. })
        ));
        assert_eq!(cache, MaxwellThreeDLoweringCache::default());
    }

    #[test]
    fn invalid_method_suffix_discards_candidate_state() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        let decoded = incrementing_packet(0x1518 / 4, &[0x3f80_0000, 0, 0]);
        let frontend_before = channel.frontend();
        let three_d_before = channel.three_d().clone();

        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
                if source.method() == GpuMethodId(0x1520)
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }

    #[test]
    fn commit_rejects_intervening_engine_or_binding_state_without_partial_publish() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        let first = packet(0x1518 / 4, 0x3f80_0000);
        let prepared = preflight_maxwell_engine_packet(
            &channel,
            FrontendSubmissionId::new(3),
            &first.packets()[0],
        )
        .unwrap();
        let intervening = packet(0x1518 / 4, 0x4000_0000);
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &intervening.packets()[0],
        )
        .unwrap();
        let committed_intervening = channel.three_d().clone();
        assert!(matches!(
            commit_maxwell_engine_packet(&mut channel, &prepared),
            Err(MaxwellEngineDispatchError::EngineStateChanged { .. })
        ));
        assert_eq!(channel.three_d(), &committed_intervening);

        let prepared = preflight_maxwell_engine_packet(
            &channel,
            FrontendSubmissionId::new(3),
            &first.packets()[0],
        )
        .unwrap();
        channel.reset_subchannel_bindings();
        let state_before_failed_commit = channel.three_d().clone();
        assert!(matches!(
            commit_maxwell_engine_packet(&mut channel, &prepared),
            Err(MaxwellEngineDispatchError::Binding(
                MaxwellMethodDispatchError::FrontendStateChanged { .. }
            ))
        ));
        assert_eq!(channel.three_d(), &state_before_failed_commit);
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
