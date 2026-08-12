//! Declarative Maxwell class-method routing.
//!
//! This layer validates every class method in a decoded packet before the
//! packet's binding state is committed. It contains no Horizon ABI, guest
//! mappings, scheduler state, or host-backend objects.

mod compute;
mod threed;
mod twod;

pub use compute::{
    MAXWELL_COMPUTE_CWD_REF_COUNTER_COUNT, MaxwellComputeAddress,
    MaxwellComputeBindlessTextureConstantBufferSlot, MaxwellComputeCwdRefCounterIndex,
    MaxwellComputeCwdRefCounterState, MaxwellComputeCwdRefCounterValue,
    MaxwellComputeDescriptorPoolState, MaxwellComputeInlineToMemoryLaunch,
    MaxwellComputeInlineToMemoryLayout, MaxwellComputeInlineToMemoryPendingTransfer,
    MaxwellComputeInlineToMemoryState, MaxwellComputeInlineToMemoryUpload,
    MaxwellComputeLocalMemoryAllocation, MaxwellComputeLocalMemoryState,
    MaxwellComputeOperationTrigger, MaxwellComputeProgramState, MaxwellComputeRegister,
    MaxwellComputeRegisterOrigin, MaxwellComputeShaderCacheInvalidation, MaxwellComputeSmCount,
    MaxwellComputeSpaVersion, MaxwellComputeState, MaxwellComputeStateWrite,
    MaxwellComputeSynchronizationPlan, MaxwellComputeTriggeredOperation,
    MaxwellShaderCacheInvalidation, lower_maxwell_compute_synchronization,
};
pub use threed::{
    MAXWELL_BIND_GROUP_COUNT, MAXWELL_COLOR_TARGET_COUNT, MAXWELL_CONSTANT_BUFFER_SLOT_COUNT,
    MAXWELL_PIPELINE_SHADER_COUNT, MAXWELL_SCISSOR_COUNT,
    MAXWELL_THREE_D_MME_CAPTURED_INSTRUCTION_WORDS, MAXWELL_THREE_D_MME_CAPTURED_START_ADDRESSES,
    MAXWELL_THREE_D_MME_EMITTED_METHOD_LIMIT, MAXWELL_THREE_D_MME_EXECUTION_INSTRUCTION_LIMIT,
    MAXWELL_THREE_D_PRIMITIVE_AREA_MAX, MAXWELL_THREE_D_SHADER_LOCAL_MEMORY_PER_WARP_SIZE_MAX,
    MAXWELL_THREE_D_SM_TIMEOUT_COUNTER_BIT_MAX, MAXWELL_VERTEX_ATTRIBUTE_COUNT,
    MAXWELL_VERTEX_STREAM_COUNT, MAXWELL_VIEWPORT_COUNT, MAXWELL_WINDOW_CLIP_COUNT,
    MaxwellThreeDAliasedLineWidthEnable, MaxwellThreeDAlphaFraction,
    MaxwellThreeDAttachmentReadiness, MaxwellThreeDAttributeDefaultVector,
    MaxwellThreeDAttributeDefaults, MaxwellThreeDBegin, MaxwellThreeDBindGroupState,
    MaxwellThreeDBlendControlState, MaxwellThreeDBlendEnableCommon, MaxwellThreeDBlendFactor,
    MaxwellThreeDBlendFloatPixelKillEnable, MaxwellThreeDBlendOp,
    MaxwellThreeDBlendPerFormatEnable, MaxwellThreeDBlendZeroTimesAnythingIsZero,
    MaxwellThreeDClearState, MaxwellThreeDClearSurface, MaxwellThreeDClearSurfaceControl,
    MaxwellThreeDClipIdTestEnable, MaxwellThreeDColorCompressionMode, MaxwellThreeDColorMask,
    MaxwellThreeDColorReductionFp16Threshold, MaxwellThreeDColorReductionSrgb8Threshold,
    MaxwellThreeDColorReductionState, MaxwellThreeDColorReductionStateWrite,
    MaxwellThreeDColorReductionThresholdsEnable, MaxwellThreeDColorReductionThresholdsFp16,
    MaxwellThreeDColorReductionThresholdsSrgb8, MaxwellThreeDColorReductionThresholdsUnorm8,
    MaxwellThreeDColorReductionThresholdsUnorm10, MaxwellThreeDColorReductionThresholdsUnorm16,
    MaxwellThreeDColorTargetFormat, MaxwellThreeDColorTargetSelection,
    MaxwellThreeDColorTargetState, MaxwellThreeDCompareOp,
    MaxwellThreeDConditionalLoadConstantBuffer, MaxwellThreeDConstantBufferBinding,
    MaxwellThreeDConstantBufferLoadState, MaxwellThreeDConstantBufferSelectorState,
    MaxwellThreeDCoverageState, MaxwellThreeDCoverageStateWrite, MaxwellThreeDCsaaEnable,
    MaxwellThreeDCullFace, MaxwellThreeDDepthStencilFormat, MaxwellThreeDDepthStencilTargetState,
    MaxwellThreeDDescriptorPoolState, MaxwellThreeDDirectlyAddressableMemory,
    MaxwellThreeDDirtySubresource, MaxwellThreeDDirtySubresources, MaxwellThreeDEdgeFlag,
    MaxwellThreeDFixedFunctionRegister, MaxwellThreeDFixedFunctionState,
    MaxwellThreeDFixedFunctionValue, MaxwellThreeDFixedFunctionWrite,
    MaxwellThreeDFlushPendingWrites, MaxwellThreeDFrontFace, MaxwellThreeDImageKind,
    MaxwellThreeDImageLayout, MaxwellThreeDIndexBufferState, MaxwellThreeDIndexElementSize,
    MaxwellThreeDInlineConstantBufferUpload, MaxwellThreeDL2CacheEvictionPolicy,
    MaxwellThreeDLineState, MaxwellThreeDLineStateWrite, MaxwellThreeDLoweredWork,
    MaxwellThreeDLoweringCache, MaxwellThreeDLoweringError, MaxwellThreeDLoweringPlan,
    MaxwellThreeDMappingReference, MaxwellThreeDMmeExecutionError, MaxwellThreeDMmeExecutionReport,
    MaxwellThreeDMmeInstruction, MaxwellThreeDMmeLoadError, MaxwellThreeDMmeRam,
    MaxwellThreeDMmeRamAddress, MaxwellThreeDMmeState, MaxwellThreeDMmeStateWrite,
    MaxwellThreeDOperationTrigger, MaxwellThreeDPatchSize, MaxwellThreeDPipelineBindingState,
    MaxwellThreeDPointCenterMode, MaxwellThreeDPointSize, MaxwellThreeDPointSpriteOrigin,
    MaxwellThreeDPointSpriteRMode, MaxwellThreeDPointSpriteSelect, MaxwellThreeDPolygonMode,
    MaxwellThreeDPreservedImageLayout, MaxwellThreeDPrimitiveCircularBufferThrottle,
    MaxwellThreeDPrimitiveState, MaxwellThreeDPrimitiveTopology, MaxwellThreeDProgramRegionState,
    MaxwellThreeDPsOutputSampleMaskUsage, MaxwellThreeDRasterBoundingBox,
    MaxwellThreeDRasterBoundingBoxMode, MaxwellThreeDRasterState, MaxwellThreeDRawValue,
    MaxwellThreeDRectangle, MaxwellThreeDRegister, MaxwellThreeDRegisterOrigin,
    MaxwellThreeDRenderEnableMode, MaxwellThreeDRenderEnableState,
    MaxwellThreeDRenderEnableStateWrite, MaxwellThreeDRenderTargetLayer,
    MaxwellThreeDRenderTargetLayerControl, MaxwellThreeDRenderTargetState,
    MaxwellThreeDRenderTargetWrite, MaxwellThreeDResolvedBuffer, MaxwellThreeDResolvedImage,
    MaxwellThreeDResolvedResource, MaxwellThreeDResolvedResources, MaxwellThreeDResourceAccess,
    MaxwellThreeDResourceAlias, MaxwellThreeDResourceError, MaxwellThreeDResourceRole,
    MaxwellThreeDRopL2CacheRequest, MaxwellThreeDRopL2CacheState,
    MaxwellThreeDRopL2CacheStateWrite, MaxwellThreeDSampleMode, MaxwellThreeDSamplerBindingMode,
    MaxwellThreeDScissorState, MaxwellThreeDSeparateFragmentData, MaxwellThreeDShadeMode,
    MaxwellThreeDShaderBindingState, MaxwellThreeDShaderBindingWrite,
    MaxwellThreeDShaderExecutionState, MaxwellThreeDShaderExecutionStateWrite,
    MaxwellThreeDShaderLocalMemoryPerWarpSize, MaxwellThreeDShaderLocalMemoryState,
    MaxwellThreeDShaderResourceUse, MaxwellThreeDShaderStage, MaxwellThreeDSmTimeoutCounterBit,
    MaxwellThreeDState, MaxwellThreeDStateWrite, MaxwellThreeDStencilOp,
    MaxwellThreeDSynchronizationError, MaxwellThreeDSynchronizationOperation,
    MaxwellThreeDSynchronizationPlan, MaxwellThreeDSynchronizationTrigger,
    MaxwellThreeDSyncpointCondition, MaxwellThreeDSyncpointIncrement,
    MaxwellThreeDTextureCacheLines, MaxwellThreeDTextureDataCacheInvalidation,
    MaxwellThreeDTranslatedShader, MaxwellThreeDTranslatedShaders,
    MaxwellThreeDUnnegotiatedLoweringPlan, MaxwellThreeDUnorm8, MaxwellThreeDUnresolvedAddress,
    MaxwellThreeDVertexArrayPrimitiveRestartEnable, MaxwellThreeDVertexAssemblyState,
    MaxwellThreeDVertexAttributeFormat, MaxwellThreeDVertexComponentWidths,
    MaxwellThreeDVertexIdUsesArrayStart, MaxwellThreeDVertexInputState,
    MaxwellThreeDVertexInputWrite, MaxwellThreeDVertexNumericalType,
    MaxwellThreeDVertexStreamFormat, MaxwellThreeDVertexStreamState,
    MaxwellThreeDVertexStreamSubstituteState, MaxwellThreeDViewportClipControl,
    MaxwellThreeDViewportScaleOffsetEnable, MaxwellThreeDViewportState,
    MaxwellThreeDViewportTransformState, MaxwellThreeDViewportZClipRange,
    MaxwellThreeDVisibleCallLimit, MaxwellThreeDWindowClipState, MaxwellThreeDWindowClipType,
    MaxwellThreeDZCompressionMode, MaxwellThreeDZCullRegionId, MaxwellThreeDZCullState,
    MaxwellThreeDZCullStateWrite, MaxwellThreeDZCullStatsEnable,
    lower_maxwell_three_d_synchronization, preflight_maxwell_three_d_operation,
    preflight_maxwell_three_d_operation_unnegotiated, resolve_maxwell_three_d_resources,
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
    MaxwellAamVersionRange, MaxwellDecodedPacket, MaxwellDecodedPushbuffer, MaxwellGpuChannel,
    MaxwellHostMemoryOperation, MaxwellHostMethod, MaxwellMethodDispatch,
    MaxwellMethodDispatchError, MaxwellMethodDispatchKind, MaxwellMethodSource,
    MaxwellPacketDispatch, MaxwellShaderProgramHeaderVersionRange, preflight_maxwell_packet,
};

/// Execution layer required by a known method whose semantics are unavailable.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellEngineCapability {
    NeutralExecution,
    HostBackend,
}

#[cfg(test)]
mod tests;

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
    AamCompatibilityCheck {
        requested: MaxwellAamVersionRange,
        supported: MaxwellAamVersionRange,
    },
    ShaderProgramHeaderCompatibilityCheck {
        requested: MaxwellShaderProgramHeaderVersionRange,
        supported: MaxwellShaderProgramHeaderVersionRange,
    },
    HostMemoryOperandLow {
        operand_low: u32,
    },
    HostSynchronization(MaxwellHostMemoryOperation),
    TwoDState(MaxwellTwoDStateWrite),
    ComputeState(MaxwellComputeStateWrite),
    ComputeTrigger(MaxwellComputeOperationTrigger),
    ComputeStateAndInlineToMemoryUpload {
        state: MaxwellComputeStateWrite,
        upload: MaxwellComputeInlineToMemoryUpload,
    },
    ThreeDState(MaxwellThreeDStateWrite),
    ThreeDTrigger(MaxwellThreeDOperationTrigger),
    ThreeDSynchronizationTrigger(MaxwellThreeDSynchronizationTrigger),
    ThreeDStateAndTrigger {
        state: MaxwellThreeDStateWrite,
        trigger: MaxwellThreeDOperationTrigger,
    },
    ThreeDStateAndInlineConstantBufferUpload {
        state: MaxwellThreeDStateWrite,
        upload: MaxwellThreeDInlineConstantBufferUpload,
    },
    MmeMacroCall {
        macro_index: u8,
        parameter_count: u16,
        report: MaxwellThreeDMmeExecutionReport,
    },
    MmeMacroData {
        macro_index: u8,
    },
}

/// One execution-relevant effect in exact pushbuffer order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MaxwellEngineOperation {
    HostSynchronization(MaxwellHostSynchronizationOperation),
    ComputeInlineToMemory(MaxwellComputeInlineToMemoryUpload),
    ComputeSynchronization(Box<MaxwellComputeTriggeredOperation>),
    ThreeDInlineConstantBuffer(MaxwellThreeDInlineConstantBufferUpload),
    ThreeD(Box<MaxwellThreeDTriggeredOperation>),
    ThreeDSynchronization(Box<MaxwellThreeDSynchronizationOperation>),
}

/// One validated host cache operation at its exact pushbuffer source.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellHostSynchronizationOperation {
    source: MaxwellMethodSource,
    operation: MaxwellHostMemoryOperation,
}

impl MaxwellHostSynchronizationOperation {
    #[must_use]
    pub const fn source(self) -> MaxwellMethodSource {
        self.source
    }

    #[must_use]
    pub const fn operation(self) -> MaxwellHostMemoryOperation {
        self.operation
    }
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
    compute_before: MaxwellComputeState,
    compute_after: MaxwellComputeState,
    two_d_before: MaxwellTwoDState,
    two_d_after: MaxwellTwoDState,
    three_d_before: MaxwellThreeDState,
    three_d_after: MaxwellThreeDState,
    methods: Box<[MaxwellEngineMethodDispatch]>,
    ordered_operations: Box<[MaxwellEngineOperation]>,
    compute_operations: Box<[MaxwellComputeTriggeredOperation]>,
    synchronization_operations: Box<[MaxwellThreeDSynchronizationOperation]>,
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
    pub fn ordered_operations(&self) -> &[MaxwellEngineOperation] {
        &self.ordered_operations
    }

    #[must_use]
    pub const fn compute_before(&self) -> &MaxwellComputeState {
        &self.compute_before
    }

    #[must_use]
    pub const fn compute_after(&self) -> &MaxwellComputeState {
        &self.compute_after
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

    #[must_use]
    pub fn compute_operations(&self) -> &[MaxwellComputeTriggeredOperation] {
        &self.compute_operations
    }

    #[must_use]
    pub fn synchronization_operations(&self) -> &[MaxwellThreeDSynchronizationOperation] {
        &self.synchronization_operations
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
    InvalidComputeMethodEncoding {
        source: MaxwellMethodSource,
        method_name: &'static str,
        reason: &'static str,
    },
    MmeRamLoad {
        source: MaxwellMethodSource,
        ram: MaxwellThreeDMmeRam,
        error: MaxwellThreeDMmeLoadError,
    },
    MmeExecution {
        source: MaxwellMethodSource,
        error: MaxwellThreeDMmeExecutionError,
    },
    IncompatibleShaderProgramHeaderVersion {
        source: MaxwellMethodSource,
        requested: MaxwellShaderProgramHeaderVersionRange,
        supported: MaxwellShaderProgramHeaderVersionRange,
    },
    IncompatibleAamVersion {
        source: MaxwellMethodSource,
        requested: MaxwellAamVersionRange,
        supported: MaxwellAamVersionRange,
    },
    UnsupportedConditionalConstantBufferLoad {
        source: MaxwellMethodSource,
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
            Self::InvalidComputeMethodEncoding {
                source,
                method_name,
                reason,
            } => write!(
                formatter,
                "Maxwell method has an invalid verified encoding: {source} class-name=MAXWELL_COMPUTE_B method-name={method_name} reason={reason}"
            ),
            Self::MmeRamLoad { source, ram, error } => write!(
                formatter,
                "MAXWELL_B MME RAM load exceeds implemented host coverage: {source} ram={ram:?} error={error:?}"
            ),
            Self::MmeExecution { source, error } => write!(
                formatter,
                "MAXWELL_B MME execution failed: {source} error={error:?}"
            ),
            Self::IncompatibleShaderProgramHeaderVersion {
                source,
                requested,
                supported,
            } => write!(
                formatter,
                "Maxwell shader-program-header version check is incompatible with the bound profile: {source} requested={}..={} supported={}..={}",
                requested.oldest_supported().raw(),
                requested.current().raw(),
                supported.oldest_supported().raw(),
                supported.current().raw()
            ),
            Self::IncompatibleAamVersion {
                source,
                requested,
                supported,
            } => write!(
                formatter,
                "Maxwell AAM version check is incompatible with the bound profile: {source} requested={}..={} supported={}..={}",
                requested.oldest_supported().raw(),
                requested.current().raw(),
                supported.oldest_supported().raw(),
                supported.current().raw()
            ),
            Self::UnsupportedConditionalConstantBufferLoad { source } => write!(
                formatter,
                "conditional Maxwell constant-buffer load semantics are unavailable: {source}"
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
    let compute_before = channel.compute().clone();
    let mut compute_after = compute_before.clone();
    let two_d_before = channel.two_d().clone();
    let mut two_d_after = two_d_before.clone();
    let three_d_before = channel.three_d().clone();
    let mut three_d_after = three_d_before.clone();
    let mut methods = Vec::new();
    let mut ordered_operations = Vec::new();
    let mut compute_operations = Vec::new();
    let mut synchronization_operations = Vec::new();
    let mut operations = Vec::new();
    methods
        .try_reserve_exact(binding.methods().len())
        .map_err(|_| MaxwellEngineDispatchError::ResourceExhausted)?;

    let mut method_index = 0;
    while method_index < binding.methods().len() {
        let method = binding.methods()[method_index];
        if let MaxwellMethodDispatchKind::HostMethod(host) = method.kind() {
            let method = preflight_host_method(method, host);
            if let MaxwellEngineMethodEffect::HostSynchronization(operation) = method.effect() {
                ordered_operations
                    .try_reserve(1)
                    .map_err(|_| MaxwellEngineDispatchError::ResourceExhausted)?;
                ordered_operations.push(MaxwellEngineOperation::HostSynchronization(
                    MaxwellHostSynchronizationOperation {
                        source: method.method().source(),
                        operation,
                    },
                ));
            }
            methods.push(method);
            method_index += 1;
            continue;
        }
        if method.kind() == MaxwellMethodDispatchKind::ClassMethod
            && method.class() == threed::CLASS
            && threed::is_mme_aperture(method.source().method())
        {
            let first_method = method.source().method().0;
            if (first_method - 0x3800) & 7 != 0 {
                return Err(MaxwellEngineDispatchError::MmeExecution {
                    source: method.source(),
                    error: MaxwellThreeDMmeExecutionError::DataWithoutCall,
                });
            }
            let data_method = first_method + 4;
            let mut end = method_index + 1;
            while end < binding.methods().len() {
                let next = binding.methods()[end];
                let next_method = next.source().method().0;
                if next.kind() != MaxwellMethodDispatchKind::ClassMethod
                    || next.class() != threed::CLASS
                    || (next_method != first_method && next_method != data_method)
                {
                    break;
                }
                end += 1;
            }
            let macro_preflight = threed::preflight_mme_call(
                channel.profile(),
                &binding.methods()[method_index..end],
                &mut three_d_after,
            )?;
            methods
                .try_reserve(macro_preflight.methods.len())
                .map_err(|_| MaxwellEngineDispatchError::ResourceExhausted)?;
            operations
                .try_reserve(macro_preflight.operations.len())
                .map_err(|_| MaxwellEngineDispatchError::ResourceExhausted)?;
            synchronization_operations
                .try_reserve(macro_preflight.synchronization_operations.len())
                .map_err(|_| MaxwellEngineDispatchError::ResourceExhausted)?;
            ordered_operations
                .try_reserve(macro_preflight.ordered_operations.len())
                .map_err(|_| MaxwellEngineDispatchError::ResourceExhausted)?;
            methods.extend(macro_preflight.methods);
            operations.extend(macro_preflight.operations);
            synchronization_operations.extend(macro_preflight.synchronization_operations);
            ordered_operations.extend(macro_preflight.ordered_operations);
            method_index = end;
            continue;
        }
        if method.kind() == MaxwellMethodDispatchKind::ClassMethod {
            let method = preflight_class_method(
                channel,
                method,
                &mut compute_after,
                &mut two_d_after,
                &mut three_d_after,
            )?;
            let trigger = match method.effect() {
                MaxwellEngineMethodEffect::ThreeDTrigger(trigger)
                | MaxwellEngineMethodEffect::ThreeDStateAndTrigger { trigger, .. } => Some(trigger),
                _ => None,
            };
            let compute_trigger = match method.effect() {
                MaxwellEngineMethodEffect::ComputeTrigger(trigger) => Some(trigger),
                _ => None,
            };
            let synchronization_trigger = match method.effect() {
                MaxwellEngineMethodEffect::ThreeDSynchronizationTrigger(trigger) => Some(trigger),
                _ => None,
            };
            let inline_operation = match method.effect() {
                MaxwellEngineMethodEffect::ComputeStateAndInlineToMemoryUpload {
                    upload, ..
                } => Some(MaxwellEngineOperation::ComputeInlineToMemory(upload)),
                MaxwellEngineMethodEffect::ThreeDStateAndInlineConstantBufferUpload {
                    upload,
                    ..
                } => Some(MaxwellEngineOperation::ThreeDInlineConstantBuffer(upload)),
                _ => None,
            };
            methods.push(method);
            if let Some(operation) = inline_operation {
                ordered_operations
                    .try_reserve(1)
                    .map_err(|_| MaxwellEngineDispatchError::ResourceExhausted)?;
                ordered_operations.push(operation);
            }
            if let Some(trigger) = compute_trigger {
                compute_operations
                    .try_reserve(1)
                    .map_err(|_| MaxwellEngineDispatchError::ResourceExhausted)?;
                let operation =
                    MaxwellComputeTriggeredOperation::new(trigger, compute_after.clone());
                ordered_operations
                    .try_reserve(1)
                    .map_err(|_| MaxwellEngineDispatchError::ResourceExhausted)?;
                ordered_operations.push(MaxwellEngineOperation::ComputeSynchronization(Box::new(
                    operation.clone(),
                )));
                compute_operations.push(operation);
            }
            if let Some(trigger) = trigger {
                operations
                    .try_reserve(1)
                    .map_err(|_| MaxwellEngineDispatchError::ResourceExhausted)?;
                let operation = MaxwellThreeDTriggeredOperation {
                    trigger,
                    state: three_d_after.clone(),
                };
                ordered_operations
                    .try_reserve(1)
                    .map_err(|_| MaxwellEngineDispatchError::ResourceExhausted)?;
                ordered_operations
                    .push(MaxwellEngineOperation::ThreeD(Box::new(operation.clone())));
                operations.push(operation);
            }
            if let Some(trigger) = synchronization_trigger {
                synchronization_operations
                    .try_reserve(1)
                    .map_err(|_| MaxwellEngineDispatchError::ResourceExhausted)?;
                let operation =
                    MaxwellThreeDSynchronizationOperation::new(trigger, three_d_after.clone());
                ordered_operations
                    .try_reserve(1)
                    .map_err(|_| MaxwellEngineDispatchError::ResourceExhausted)?;
                ordered_operations.push(MaxwellEngineOperation::ThreeDSynchronization(Box::new(
                    operation.clone(),
                )));
                synchronization_operations.push(operation);
            }
        }
        method_index += 1;
    }
    three_d_after.validate_cross_registers().map_err(|error| {
        MaxwellEngineDispatchError::ContradictoryState {
            source: error.source,
            reason: error.reason,
        }
    })?;

    Ok(MaxwellEnginePacketDispatch {
        binding,
        compute_before,
        compute_after,
        two_d_before,
        two_d_after,
        three_d_before,
        three_d_after,
        methods: methods.into_boxed_slice(),
        ordered_operations: ordered_operations.into_boxed_slice(),
        compute_operations: compute_operations.into_boxed_slice(),
        synchronization_operations: synchronization_operations.into_boxed_slice(),
        operations: operations.into_boxed_slice(),
    })
}

fn preflight_host_method(
    method: MaxwellMethodDispatch,
    host: MaxwellHostMethod,
) -> MaxwellEngineMethodDispatch {
    match host {
        MaxwellHostMethod::LegacyMemOpA { operand_low } => MaxwellEngineMethodDispatch::new(
            method,
            MaxwellEngineMethodMetadata::new(
                method.class(),
                "MAXWELL_CHANNEL_GPFIFO_A(legacy-host-compatibility)",
                method.source().method(),
                "MEM_OP_A",
            ),
            MaxwellEngineMethodEffect::HostMemoryOperandLow { operand_low },
        ),
        MaxwellHostMethod::LegacyMemOpB(operation) => MaxwellEngineMethodDispatch::new(
            method,
            MaxwellEngineMethodMetadata::new(
                method.class(),
                "MAXWELL_CHANNEL_GPFIFO_A(legacy-host-compatibility)",
                method.source().method(),
                "MEM_OP_B",
            ),
            MaxwellEngineMethodEffect::HostSynchronization(operation),
        ),
    }
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
    if channel.compute() != &dispatch.compute_before
        || channel.two_d() != &dispatch.two_d_before
        || channel.three_d() != &dispatch.three_d_before
    {
        return Err(MaxwellEngineDispatchError::EngineStateChanged {
            channel: channel.id(),
        });
    }
    channel.replace_frontend(dispatch.binding.frontend_after());
    channel.replace_compute(dispatch.compute_after.clone());
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
    compute: &mut MaxwellComputeState,
    two_d: &mut MaxwellTwoDState,
    three_d: &mut MaxwellThreeDState,
) -> Result<MaxwellEngineMethodDispatch, MaxwellEngineDispatchError> {
    let classes = channel.profile().classes();
    let class = method.class();
    if class == compute::CLASS {
        return compute::preflight(class, method, compute);
    }
    if class == threed::CLASS {
        return threed::preflight(channel.profile(), method, three_d);
    }
    if class == twod::CLASS {
        return twod::preflight(method, two_d);
    }

    let class_name = if class == classes.dma_copy() {
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
