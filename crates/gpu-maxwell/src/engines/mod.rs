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
    lower_maxwell_compute_synchronization,
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
    MaxwellMethodDispatch, MaxwellMethodDispatchError, MaxwellMethodDispatchKind,
    MaxwellMethodSource, MaxwellPacketDispatch, MaxwellShaderProgramHeaderVersionRange,
    preflight_maxwell_packet,
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
    AamCompatibilityCheck {
        requested: MaxwellAamVersionRange,
        supported: MaxwellAamVersionRange,
    },
    ShaderProgramHeaderCompatibilityCheck {
        requested: MaxwellShaderProgramHeaderVersionRange,
        supported: MaxwellShaderProgramHeaderVersionRange,
    },
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
    ComputeInlineToMemory(MaxwellComputeInlineToMemoryUpload),
    ComputeSynchronization(Box<MaxwellComputeTriggeredOperation>),
    ThreeDInlineConstantBuffer(MaxwellThreeDInlineConstantBufferUpload),
    ThreeD(Box<MaxwellThreeDTriggeredOperation>),
    ThreeDSynchronization(Box<MaxwellThreeDSynchronizationOperation>),
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

#[cfg(test)]
mod tests {
    use nixe_gpu::{
        BackendCapabilities, BackendFeatures, BackendLimits, GpuCommand, GpuVirtualAddress,
        GuestSyncpointId, GuestSyncpointValue, GuestTimeline, ImageFormat, MappingGeneration,
        QueryKind, RenderPassOperation, SampleCount, ShaderId, ShaderStage, TimelineInstanceId,
        TimelineOwnerId,
    };
    use nixe_memory::{CanonicalAllocation, CanonicalBackingRange, MemoryPermissions};

    use super::*;
    use crate::{
        MaxwellAamVersion, MaxwellAamVersionRange, MaxwellAddressSpaceId,
        MaxwellAddressSpaceInitialization, MaxwellAllocationId, MaxwellChannelId,
        MaxwellChannelOwner, MaxwellGpfifoSourceLocation, MaxwellGpuAddressSpace,
        MaxwellGpuMapping, MaxwellMapRequest, MaxwellMappingId, MaxwellPushbufferWord,
        MaxwellShaderProgramHeaderVersion, SWITCH_1_GM20B_PROFILE, decode_maxwell_pushbuffer,
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

    fn increment_once_packet(method_dword: u32, arguments: &[u32]) -> MaxwellDecodedPushbuffer {
        increment_once_packet_on_subchannel(0, method_dword, arguments)
    }

    fn increment_once_packet_on_subchannel(
        subchannel: u32,
        method_dword: u32,
        arguments: &[u32],
    ) -> MaxwellDecodedPushbuffer {
        let mut words = Vec::with_capacity(arguments.len() + 1);
        words.push(word(
            (5 << 29) | ((arguments.len() as u32) << 16) | (subchannel << 13) | method_dword,
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

    fn load_mme_program(channel: &mut MaxwellGpuChannel, macro_index: u8, code: &[u32]) {
        let start = 0x100 + u32::from(macro_index) * 0x10;
        let start_packet = incrementing_packet(0x011c / 4, &[u32::from(macro_index), start]);
        dispatch_maxwell_engine_packet(
            channel,
            FrontendSubmissionId::new(3),
            &start_packet.packets()[0],
        )
        .unwrap();
        let mut arguments = Vec::with_capacity(code.len() + 1);
        arguments.push(start);
        arguments.extend_from_slice(code);
        let instruction_packet = increment_once_packet(0x0114 / 4, &arguments);
        dispatch_maxwell_engine_packet(
            channel,
            FrontendSubmissionId::new(3),
            &instruction_packet.packets()[0],
        )
        .unwrap();
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

    fn non_incrementing_packet_on_subchannel(
        subchannel: u32,
        method_dword: u32,
        arguments: &[u32],
    ) -> MaxwellDecodedPushbuffer {
        let mut words = Vec::with_capacity(arguments.len() + 1);
        words.push(word(
            (3 << 29) | ((arguments.len() as u32) << 16) | (subchannel << 13) | method_dword,
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

    fn bind_compute(channel: &mut MaxwellGpuChannel) {
        let decoded = packet_on_subchannel(1, 0, compute::CLASS.0);
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
            (0x0308, 3),
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
                MaxwellThreeDTranslatedShader::new(
                    ShaderStage::Vertex,
                    ShaderId::new(1),
                    7,
                    MaxwellThreeDDirectlyAddressableMemory::Size48KiB,
                ),
                MaxwellThreeDTranslatedShader::new(
                    ShaderStage::Fragment,
                    ShaderId::new(2),
                    9,
                    MaxwellThreeDDirectlyAddressableMemory::Size48KiB,
                ),
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
    fn alpha_fraction_is_typed_source_preserving_raster_state() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        let fixed_function_before = channel.three_d().fixed_function().clone();
        let render_targets_before = channel.three_d().render_targets().clone();
        let viewport_before = channel.three_d().viewport().clone();
        let point_size_before = *channel.three_d().raster().point_size();
        let two_d_before = channel.two_d().clone();
        let mut previous_dependencies = channel.three_d().pipeline_dependencies(&[]);
        assert_eq!(
            channel.three_d().raster().alpha_fraction().origin(),
            MaxwellThreeDRegisterOrigin::Unset
        );

        for argument in [0, 0x3f, 0xff] {
            let decoded = packet(0x074c / 4, argument);
            let dispatch = dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0],
            )
            .unwrap();
            let source = dispatch.methods()[0].method().source();
            let value = MaxwellThreeDAlphaFraction::new(argument as u8);
            let register = channel.three_d().raster().alpha_fraction();

            assert_eq!(
                dispatch.methods()[0].metadata().method_name(),
                "SET_ALPHA_FRACTION"
            );
            assert_eq!(
                dispatch.methods()[0].effect(),
                MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::AlphaFraction {
                    value,
                    source,
                })
            );
            assert!(dispatch.operations().is_empty());
            assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
            assert_eq!(register.raw(), Some(argument));
            assert_eq!(register.value().copied(), Some(value));
            assert_eq!(register.source(), Some(source));
            assert_eq!(value.raw(), argument as u8);
            assert_eq!(channel.three_d().raster().point_size(), &point_size_before);
            assert_eq!(channel.three_d().fixed_function(), &fixed_function_before);
            assert_eq!(channel.three_d().render_targets(), &render_targets_before);
            assert_eq!(channel.three_d().viewport(), &viewport_before);
            assert_eq!(channel.two_d(), &two_d_before);

            let dependencies = channel.three_d().pipeline_dependencies(&[]);
            assert_ne!(dependencies, previous_dependencies);
            previous_dependencies = dependencies;
        }
    }

    #[test]
    fn invalid_alpha_fraction_values_and_packet_suffix_are_rejected_atomically() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        program_three_d(&mut channel, 0x074c, 0x3f);

        for argument in [0x0000_0100, 0x0000_ff00, 0x8000_0000, u32::MAX] {
            let frontend_before = channel.frontend();
            let two_d_before = channel.two_d().clone();
            let three_d_before = channel.three_d().clone();
            let decoded = packet(0x074c / 4, argument);

            assert!(matches!(
                dispatch_maxwell_engine_packet(
                    &mut channel,
                    FrontendSubmissionId::new(3),
                    &decoded.packets()[0]
                ),
                Err(MaxwellEngineDispatchError::InvalidMethodValue {
                    source,
                    defined_mask: 0x0000_00ff,
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
        let decoded = incrementing_packet(0x074c / 4, &[0xff, 0]);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
                if source.method() == GpuMethodId(0x0750)
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }

    #[test]
    fn raster_bounding_box_is_typed_source_preserving_and_pipeline_neutral() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        let dependencies = channel.three_d().pipeline_dependencies(&[]);
        let two_d_before = channel.two_d().clone();
        assert_eq!(
            channel.three_d().raster().bounding_box().origin(),
            MaxwellThreeDRegisterOrigin::Unset
        );

        for (argument, mode, pad) in [
            (0, MaxwellThreeDRasterBoundingBoxMode::BoundingBox, 0),
            (1, MaxwellThreeDRasterBoundingBoxMode::FullViewport, 0),
            (0x60, MaxwellThreeDRasterBoundingBoxMode::BoundingBox, 6),
            (
                0x0ff1,
                MaxwellThreeDRasterBoundingBoxMode::FullViewport,
                u8::MAX,
            ),
        ] {
            let decoded = packet(0x02ec / 4, argument);
            let dispatch = dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0],
            )
            .unwrap();
            let source = dispatch.methods()[0].method().source();
            let value = MaxwellThreeDRasterBoundingBox::new(mode, pad);
            let register = channel.three_d().raster().bounding_box();

            assert_eq!(
                dispatch.methods()[0].metadata().method_name(),
                "SET_RASTER_BOUNDING_BOX"
            );
            assert_eq!(
                dispatch.methods()[0].effect(),
                MaxwellEngineMethodEffect::ThreeDState(
                    MaxwellThreeDStateWrite::RasterBoundingBox { value, source }
                )
            );
            assert!(dispatch.operations().is_empty());
            assert_eq!(value.mode(), mode);
            assert_eq!(value.pad(), pad);
            assert_eq!(value.raw(), argument);
            assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
            assert_eq!(register.raw(), Some(argument));
            assert_eq!(register.value(), Some(&value));
            assert_eq!(register.source(), Some(source));
            assert_eq!(channel.three_d().pipeline_dependencies(&[]), dependencies);
            assert_eq!(channel.two_d(), &two_d_before);
        }

        let macro_index = 8;
        let method_dword = 0x02ec / 4;
        let set_method = 1 | (2 << 4) | (method_dword << 14);
        let send_parameter_and_exit = (4 << 4) | (1 << 7) | (1 << 11);
        load_mme_program(
            &mut channel,
            macro_index,
            &[set_method, send_parameter_and_exit, 0x11],
        );
        let call = packet((0x3800 + u32::from(macro_index) * 8) / 4, 0x60);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &call.packets()[0],
        )
        .unwrap();
        assert_eq!(
            dispatch.methods()[0].effect(),
            MaxwellEngineMethodEffect::MmeMacroCall {
                macro_index,
                parameter_count: 1,
                report: MaxwellThreeDMmeExecutionReport {
                    instructions: 3,
                    emitted_methods: 1,
                },
            }
        );
        let register = channel.three_d().raster().bounding_box();
        let source = register.source().unwrap();
        assert_eq!(register.raw(), Some(0x60));
        assert_eq!(source.method(), GpuMethodId(0x02ec));
        assert_eq!(
            source.location(),
            dispatch.methods()[0].method().source().location()
        );
        assert_eq!(channel.three_d().pipeline_dependencies(&[]), dependencies);
    }

    #[test]
    fn invalid_raster_bounding_box_values_and_packet_suffix_are_atomic() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        program_three_d(&mut channel, 0x02ec, 0x60);

        for argument in [2, 4, 8, 0x1000, 0x8000_0000, u32::MAX] {
            let frontend_before = channel.frontend();
            let two_d_before = channel.two_d().clone();
            let three_d_before = channel.three_d().clone();
            let decoded = packet(0x02ec / 4, argument);
            assert!(matches!(
                dispatch_maxwell_engine_packet(
                    &mut channel,
                    FrontendSubmissionId::new(3),
                    &decoded.packets()[0]
                ),
                Err(MaxwellEngineDispatchError::InvalidMethodValue {
                    source,
                    defined_mask: 0x0000_0ff1,
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
        let decoded = incrementing_packet(0x02ec / 4, &[0x60, 0]);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
                if source.method() == GpuMethodId(0x02f0)
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }

    #[test]
    fn sph_version_check_is_typed_profile_validated_and_state_neutral() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        let supported = channel.profile().shader().sph_versions();

        for (argument, current, oldest_supported) in [
            (0x0003_0003, 3, 3),
            (0x0003_0004, 4, 3),
            (0x0002_0003, 3, 2),
            (0x0002_0004, 4, 2),
        ] {
            let frontend_before = channel.frontend();
            let two_d_before = channel.two_d().clone();
            let three_d_before = channel.three_d().clone();
            let decoded = packet(0x16a8 / 4, argument);
            let dispatch = dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0],
            )
            .unwrap();
            let requested = MaxwellShaderProgramHeaderVersionRange::new(
                MaxwellShaderProgramHeaderVersion::new(current),
                MaxwellShaderProgramHeaderVersion::new(oldest_supported),
            );

            assert_eq!(
                dispatch.methods()[0].metadata().method_name(),
                "CHECK_SPH_VERSION"
            );
            assert_eq!(
                dispatch.methods()[0].effect(),
                MaxwellEngineMethodEffect::ShaderProgramHeaderCompatibilityCheck {
                    requested,
                    supported,
                }
            );
            assert!(dispatch.operations().is_empty());
            assert_eq!(requested.raw(), argument);
            assert_eq!(channel.two_d(), &two_d_before);
            assert_eq!(channel.three_d(), &three_d_before);
            assert_eq!(channel.frontend(), frontend_before);
        }
    }

    #[test]
    fn malformed_incompatible_and_suffixed_sph_checks_are_rejected_atomically() {
        let mut channel = channel();
        bind_three_d(&mut channel);

        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x16a8 / 4, 0x0003_0002);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                source,
                method_name: "CHECK_SPH_VERSION",
                ..
            }) if source.argument() == 0x0003_0002
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);

        for argument in [0x0002_0002, 0x0004_0004] {
            let frontend_before = channel.frontend();
            let two_d_before = channel.two_d().clone();
            let three_d_before = channel.three_d().clone();
            let decoded = packet(0x16a8 / 4, argument);

            assert!(matches!(
                dispatch_maxwell_engine_packet(
                    &mut channel,
                    FrontendSubmissionId::new(3),
                    &decoded.packets()[0]
                ),
                Err(
                    MaxwellEngineDispatchError::IncompatibleShaderProgramHeaderVersion {
                        source,
                        requested,
                        supported,
                    }
                ) if source.argument() == argument
                    && requested.raw() == argument
                    && supported.raw() == 0x0003_0003
            ));
            assert_eq!(channel.frontend(), frontend_before);
            assert_eq!(channel.two_d(), &two_d_before);
            assert_eq!(channel.three_d(), &three_d_before);
        }

        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = incrementing_packet(0x16a8 / 4, &[0x0003_0003, 0]);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
                if source.method() == GpuMethodId(0x16ac)
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }

    #[test]
    fn aam_version_check_is_typed_profile_validated_and_state_neutral() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        let supported = channel.profile().aam_versions();

        for (argument, current, oldest_supported) in [
            (0x0002_0002, 2, 2),
            (0x0002_0003, 3, 2),
            (0x0001_0002, 2, 1),
            (0x0001_0003, 3, 1),
        ] {
            let frontend_before = channel.frontend();
            let two_d_before = channel.two_d().clone();
            let three_d_before = channel.three_d().clone();
            let decoded = packet(0x1794 / 4, argument);
            let dispatch = dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0],
            )
            .unwrap();
            let requested = MaxwellAamVersionRange::new(
                MaxwellAamVersion::new(current),
                MaxwellAamVersion::new(oldest_supported),
            );

            assert_eq!(
                dispatch.methods()[0].metadata().method_name(),
                "CHECK_AAM_VERSION"
            );
            assert_eq!(
                dispatch.methods()[0].effect(),
                MaxwellEngineMethodEffect::AamCompatibilityCheck {
                    requested,
                    supported,
                }
            );
            assert!(dispatch.operations().is_empty());
            assert_eq!(requested.raw(), argument);
            assert_eq!(channel.frontend(), frontend_before);
            assert_eq!(channel.two_d(), &two_d_before);
            assert_eq!(channel.three_d(), &three_d_before);
        }
    }

    #[test]
    fn malformed_incompatible_and_suffixed_aam_checks_are_rejected_atomically() {
        let mut channel = channel();
        bind_three_d(&mut channel);

        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x1794 / 4, 0x0003_0002);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                source,
                method_name: "CHECK_AAM_VERSION",
                ..
            }) if source.argument() == 0x0003_0002
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);

        for argument in [0x0001_0001, 0x0003_0003] {
            let frontend_before = channel.frontend();
            let two_d_before = channel.two_d().clone();
            let three_d_before = channel.three_d().clone();
            let decoded = packet(0x1794 / 4, argument);

            assert!(matches!(
                dispatch_maxwell_engine_packet(
                    &mut channel,
                    FrontendSubmissionId::new(3),
                    &decoded.packets()[0]
                ),
                Err(MaxwellEngineDispatchError::IncompatibleAamVersion {
                    source,
                    requested,
                    supported,
                }) if source.argument() == argument
                    && requested.raw() == argument
                    && supported.raw() == 0x0002_0002
            ));
            assert_eq!(channel.frontend(), frontend_before);
            assert_eq!(channel.two_d(), &two_d_before);
            assert_eq!(channel.three_d(), &three_d_before);
        }

        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = incrementing_packet(0x1794 / 4, &[0x0002_0002, 0]);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
                if source.method() == GpuMethodId(0x1798)
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }

    #[test]
    fn rop_l2_cache_controls_are_typed_source_preserving_independent_state() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        let methods = [
            (
                0x0218,
                "SET_L2_CACHE_CONTROL_FOR_ROP_PREFETCH_READ_REQUESTS",
                MaxwellThreeDRopL2CacheRequest::PrefetchRead,
            ),
            (
                0x10fc,
                "SET_L2_CACHE_CONTROL_FOR_ROP_NONINTERLOCKED_READ_REQUESTS",
                MaxwellThreeDRopL2CacheRequest::NoninterlockedRead,
            ),
            (
                0x1290,
                "SET_L2_CACHE_CONTROL_FOR_ROP_INTERLOCKED_READ_REQUESTS",
                MaxwellThreeDRopL2CacheRequest::InterlockedRead,
            ),
            (
                0x12d8,
                "SET_L2_CACHE_CONTROL_FOR_ROP_NONINTERLOCKED_WRITE_REQUESTS",
                MaxwellThreeDRopL2CacheRequest::NoninterlockedWrite,
            ),
            (
                0x12dc,
                "SET_L2_CACHE_CONTROL_FOR_ROP_INTERLOCKED_WRITE_REQUESTS",
                MaxwellThreeDRopL2CacheRequest::InterlockedWrite,
            ),
        ];
        let requests = methods.map(|(_, _, request)| request);
        let two_d_before = channel.two_d().clone();
        let pipeline_dependencies_before = channel.three_d().pipeline_dependencies(&[]);

        for request in requests {
            assert_eq!(
                channel.three_d().rop_l2_cache().policy(request).origin(),
                MaxwellThreeDRegisterOrigin::Unset
            );
        }

        for (method, method_name, request) in methods {
            for (argument, value) in [
                (0x00, MaxwellThreeDL2CacheEvictionPolicy::EvictFirst),
                (0x10, MaxwellThreeDL2CacheEvictionPolicy::EvictNormal),
                (0x20, MaxwellThreeDL2CacheEvictionPolicy::EvictLast),
            ] {
                let registers_before =
                    requests.map(|other| *channel.three_d().rop_l2_cache().policy(other));
                let decoded = packet(method / 4, argument);
                let dispatch = dispatch_maxwell_engine_packet(
                    &mut channel,
                    FrontendSubmissionId::new(3),
                    &decoded.packets()[0],
                )
                .unwrap();
                let source = dispatch.methods()[0].method().source();
                let register = channel.three_d().rop_l2_cache().policy(request);

                assert_eq!(dispatch.methods()[0].metadata().method_name(), method_name);
                assert_eq!(
                    dispatch.methods()[0].effect(),
                    MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::RopL2Cache(
                        MaxwellThreeDRopL2CacheStateWrite::Policy {
                            request,
                            value,
                            source,
                        }
                    ))
                );
                assert!(dispatch.operations().is_empty());
                assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
                assert_eq!(register.raw(), Some(argument));
                assert_eq!(register.value().copied(), Some(value));
                assert_eq!(register.source(), Some(source));
                assert_eq!(value.encoded(), argument);

                for (index, other) in requests.into_iter().enumerate() {
                    if other != request {
                        assert_eq!(
                            channel.three_d().rop_l2_cache().policy(other),
                            &registers_before[index]
                        );
                    }
                }
                assert_eq!(channel.two_d(), &two_d_before);
                assert_eq!(
                    channel.three_d().pipeline_dependencies(&[]),
                    pipeline_dependencies_before
                );
            }
        }
    }

    #[test]
    fn invalid_rop_l2_cache_policies_and_packet_suffix_are_rejected_atomically() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        let methods = [0x0218, 0x10fc, 0x1290, 0x12d8, 0x12dc];
        for method in methods {
            program_three_d(&mut channel, method, 0x10);
        }

        for method in methods {
            for argument in [0x30, 0x01, 0x40, 0x8000_0010, u32::MAX] {
                let frontend_before = channel.frontend();
                let two_d_before = channel.two_d().clone();
                let three_d_before = channel.three_d().clone();
                let decoded = packet(method / 4, argument);

                assert!(matches!(
                    dispatch_maxwell_engine_packet(
                        &mut channel,
                        FrontendSubmissionId::new(3),
                        &decoded.packets()[0]
                    ),
                    Err(MaxwellEngineDispatchError::InvalidMethodValue {
                        source,
                        defined_mask: 0x0000_0030,
                        ..
                    }) if source.argument() == argument
                ));
                assert_eq!(channel.frontend(), frontend_before);
                assert_eq!(channel.two_d(), &two_d_before);
                assert_eq!(channel.three_d(), &three_d_before);
            }
        }

        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = incrementing_packet(0x0218 / 4, &[0x20, 0]);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
                if source.method() == GpuMethodId(0x021c)
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
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
    fn render_enable_control_is_typed_source_preserving_and_pipeline_neutral() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        let mode_before = *channel.three_d().render_enable().mode();
        let two_d_before = channel.two_d().clone();
        let dependencies_before = channel.three_d().pipeline_dependencies(&[]);

        for (argument, value) in [
            (0, MaxwellThreeDConditionalLoadConstantBuffer::Disabled),
            (1, MaxwellThreeDConditionalLoadConstantBuffer::Enabled),
        ] {
            let decoded = packet(0x030c / 4, argument);
            let dispatch = dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0],
            )
            .unwrap();
            let source = dispatch.methods()[0].method().source();
            let register = channel
                .three_d()
                .render_enable()
                .conditional_load_constant_buffer();

            assert_eq!(
                dispatch.methods()[0].metadata().method_name(),
                "SET_RENDER_ENABLE_CONTROL"
            );
            assert_eq!(
                dispatch.methods()[0].effect(),
                MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::RenderEnable(
                    MaxwellThreeDRenderEnableStateWrite::ConditionalLoadConstantBuffer {
                        value,
                        source,
                    }
                ))
            );
            assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
            assert_eq!(register.raw(), Some(argument));
            assert_eq!(register.value(), Some(&value));
            assert_eq!(register.source(), Some(source));
            assert_eq!(value.raw(), argument);
            assert_eq!(channel.three_d().render_enable().mode(), &mode_before);
            assert_eq!(
                channel.three_d().pipeline_dependencies(&[]),
                dependencies_before
            );
            assert_eq!(channel.two_d(), &two_d_before);
        }
    }

    #[test]
    fn invalid_render_enable_controls_are_rejected_atomically() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        program_three_d(&mut channel, 0x030c, 0);

        for argument in [2, 3, 0x10, 0x8000_0000, u32::MAX] {
            let frontend_before = channel.frontend();
            let two_d_before = channel.two_d().clone();
            let three_d_before = channel.three_d().clone();
            let decoded = packet(0x030c / 4, argument);

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
    fn enabled_conditional_load_stops_before_neutral_lowering() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        program_three_d(&mut channel, 0x1558, 1);
        let decoded = packet(0x030c / 4, 0);
        let disabled = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
        assert_eq!(
            disabled.methods()[0].metadata().method_name(),
            "SET_RENDER_ENABLE_CONTROL"
        );
        let clear = packet(0x19d0 / 4, 0x3c);
        let clear_dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &clear.packets()[0],
        )
        .unwrap();
        let disabled_triggered = &clear_dispatch.operations()[0];
        let resources = resolve_maxwell_three_d_resources(
            disabled_triggered.state(),
            &resource_address_space(),
        )
        .unwrap();
        let cache = MaxwellThreeDLoweringCache::default();
        assert!(matches!(
            preflight_maxwell_three_d_operation(
                disabled_triggered.state(),
                &resources,
                disabled_triggered.trigger(),
                None,
                FrontendSubmissionId::new(10),
                Vec::new(),
                &lowering_capabilities(BackendFeatures::empty()),
                &cache,
            ),
            Err(MaxwellThreeDLoweringError::IncompleteClear(
                "horizontal rectangle"
            ))
        ));

        let decoded = packet(0x030c / 4, 1);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_RENDER_ENABLE_CONTROL"
        );
        let clear_dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &clear.packets()[0],
        )
        .unwrap();
        let enabled_triggered = &clear_dispatch.operations()[0];
        let resources =
            resolve_maxwell_three_d_resources(enabled_triggered.state(), &resource_address_space())
                .unwrap();
        let cache_before = cache.clone();

        assert!(matches!(
            preflight_maxwell_three_d_operation(
                enabled_triggered.state(),
                &resources,
                enabled_triggered.trigger(),
                None,
                FrontendSubmissionId::new(10),
                Vec::new(),
                &lowering_capabilities(BackendFeatures::empty()),
                &cache,
            ),
            Err(MaxwellThreeDLoweringError::UnsupportedConditionalLoadConstantBufferSemantics)
        ));
        assert_eq!(cache, cache_before);
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
    fn l1_configuration_is_typed_source_preserving_shader_memory_state() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        let two_d_before = channel.two_d().clone();
        let visible_call_before = channel
            .three_d()
            .shader_execution()
            .visible_call_limit()
            .to_owned();
        let mut previous_dependencies = channel.three_d().pipeline_dependencies(&[]);
        assert_eq!(
            channel
                .three_d()
                .shader_execution()
                .l1_configuration()
                .origin(),
            MaxwellThreeDRegisterOrigin::Unset
        );

        for (argument, expected, bytes) in [
            (
                1,
                MaxwellThreeDDirectlyAddressableMemory::Size16KiB,
                16 * 1024,
            ),
            (
                3,
                MaxwellThreeDDirectlyAddressableMemory::Size48KiB,
                48 * 1024,
            ),
        ] {
            let decoded = packet(0x0308 / 4, argument);
            let dispatch = dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0],
            )
            .unwrap();
            let source = dispatch.methods()[0].method().source();
            let register = channel.three_d().shader_execution().l1_configuration();

            assert_eq!(
                dispatch.methods()[0].metadata().method_name(),
                "SET_L1_CONFIGURATION"
            );
            assert_eq!(
                dispatch.methods()[0].effect(),
                MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::ShaderExecution(
                    MaxwellThreeDShaderExecutionStateWrite::L1Configuration {
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
            assert_eq!(expected.bytes(), bytes);
            assert_eq!(
                channel.three_d().shader_execution().visible_call_limit(),
                &visible_call_before
            );
            assert_eq!(channel.two_d(), &two_d_before);

            let dependencies = channel.three_d().pipeline_dependencies(&[]);
            assert_ne!(dependencies, previous_dependencies);
            previous_dependencies = dependencies;
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
            resolve_maxwell_three_d_resources(triggered.state(), &resource_address_space())
                .unwrap();
        let cache = MaxwellThreeDLoweringCache::default();
        let cache_before = cache.clone();
        assert!(matches!(
            preflight_maxwell_three_d_operation(
                triggered.state(),
                &resources,
                triggered.trigger(),
                None,
                FrontendSubmissionId::new(10),
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
    fn invalid_l1_configurations_and_packet_suffix_are_rejected_atomically() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        program_three_d(&mut channel, 0x0308, 1);

        for argument in [0, 2, 4, 5, 6, 7, 8, 0x8000_0000, u32::MAX] {
            let frontend_before = channel.frontend();
            let two_d_before = channel.two_d().clone();
            let three_d_before = channel.three_d().clone();
            let decoded = packet(0x0308 / 4, argument);

            assert!(matches!(
                dispatch_maxwell_engine_packet(
                    &mut channel,
                    FrontendSubmissionId::new(3),
                    &decoded.packets()[0]
                ),
                Err(MaxwellEngineDispatchError::InvalidMethodValue {
                    source,
                    defined_mask: 0x07,
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
        let decoded = incrementing_packet(0x0308 / 4, &[3, 2]);
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
            }) if source.method() == GpuMethodId(0x030c) && source.argument() == 2
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }

    #[test]
    fn shader_local_memory_block_is_typed_source_preserving_and_shader_scoped() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        let two_d_before = channel.two_d().clone();
        let inactive_dependencies = channel.three_d().pipeline_dependencies(&[]);

        let region = incrementing_packet(0x0790 / 4, &[0x04, 0x0008_0000, 0, 0x0408_0000, 0]);
        let region_dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &region.packets()[0],
        )
        .unwrap();
        assert_eq!(
            region_dispatch
                .methods()
                .iter()
                .map(|method| method.metadata().method_name())
                .collect::<Vec<_>>(),
            [
                "SET_SHADER_LOCAL_MEMORY_A",
                "SET_SHADER_LOCAL_MEMORY_B",
                "SET_SHADER_LOCAL_MEMORY_C",
                "SET_SHADER_LOCAL_MEMORY_D",
                "SET_SHADER_LOCAL_MEMORY_E",
            ]
        );

        let window = packet(0x077c / 4, 0xff00_0000);
        let window_dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &window.packets()[0],
        )
        .unwrap();
        assert_eq!(
            window_dispatch.methods()[0].metadata().method_name(),
            "SET_SHADER_LOCAL_MEMORY_WINDOW"
        );

        let local = channel.three_d().shader_execution().shader_local_memory();
        assert_eq!(local.address().unwrap().get(), 0x04_0008_0000);
        assert_eq!(local.size(), Some(0x0408_0000));
        assert_eq!(local.address_upper().raw(), Some(4));
        assert_eq!(
            local.address_upper().source(),
            Some(region_dispatch.methods()[0].method().source())
        );
        assert_eq!(local.address_lower().raw(), Some(0x0008_0000));
        assert_eq!(local.size_upper().raw(), Some(0));
        assert_eq!(local.size_lower().raw(), Some(0x0408_0000));
        let per_warp = local.default_size_per_warp();
        assert_eq!(per_warp.raw(), Some(0));
        assert_eq!(per_warp.value().unwrap().bytes(), 0);
        assert_eq!(
            per_warp.source(),
            Some(region_dispatch.methods()[4].method().source())
        );
        assert_eq!(local.window_base_address().raw(), Some(0xff00_0000));
        assert_eq!(
            local.window_base_address().source(),
            Some(window_dispatch.methods()[0].method().source())
        );
        assert_eq!(
            channel.three_d().pipeline_dependencies(&[]),
            inactive_dependencies
        );
        assert_eq!(channel.two_d(), &two_d_before);

        program_three_d(&mut channel, 0x2000, 0x11);
        let active_dependencies = channel.three_d().pipeline_dependencies(&[]);
        program_three_d(&mut channel, 0x077c, 0xfe00_0000);
        assert_ne!(
            channel.three_d().pipeline_dependencies(&[]),
            active_dependencies
        );
    }

    #[test]
    fn shader_local_memory_fields_ranges_and_packets_are_atomic() {
        let mut channel = channel();
        bind_three_d(&mut channel);

        for (method, argument, mask) in [
            (0x0790, 0x100, 0xff),
            (0x0798, 0x40, 0x3f),
            (
                0x07a0,
                MAXWELL_THREE_D_SHADER_LOCAL_MEMORY_PER_WARP_SIZE_MAX + 1,
                MAXWELL_THREE_D_SHADER_LOCAL_MEMORY_PER_WARP_SIZE_MAX,
            ),
        ] {
            let frontend_before = channel.frontend();
            let two_d_before = channel.two_d().clone();
            let three_d_before = channel.three_d().clone();
            let decoded = packet(method / 4, argument);
            assert!(matches!(
                dispatch_maxwell_engine_packet(
                    &mut channel,
                    FrontendSubmissionId::new(3),
                    &decoded.packets()[0]
                ),
                Err(MaxwellEngineDispatchError::InvalidMethodValue {
                    source,
                    defined_mask,
                    ..
                }) if source.argument() == argument && defined_mask == mask
            ));
            assert_eq!(channel.frontend(), frontend_before);
            assert_eq!(channel.two_d(), &two_d_before);
            assert_eq!(channel.three_d(), &three_d_before);
        }

        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let invalid_suffix = incrementing_packet(
            0x0790 / 4,
            &[
                4,
                0x0008_0000,
                0,
                0x0408_0000,
                MAXWELL_THREE_D_SHADER_LOCAL_MEMORY_PER_WARP_SIZE_MAX + 1,
            ],
        );
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &invalid_suffix.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::InvalidMethodValue { source, .. })
                if source.method() == GpuMethodId(0x07a0)
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);

        program_three_d(&mut channel, 0x0790, 0xff);
        program_three_d(&mut channel, 0x0794, u32::MAX);
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let overflowing_size = incrementing_packet(0x0798 / 4, &[0, 2]);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &overflowing_size.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::ContradictoryState {
                source: Some(source),
                reason: "shader-local-memory region exceeds the 40-bit GPU address space",
            }) if source.method() == GpuMethodId(0x079c)
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }

    #[test]
    fn active_shader_local_memory_blocks_only_draws_before_effects() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        program_three_d(&mut channel, 0x121c, 0);
        program_three_d(&mut channel, 0x2000, 0x11);
        program_three_d(&mut channel, 0x0790, 4);
        let resources =
            resolve_maxwell_three_d_resources(channel.three_d(), &resource_address_space())
                .unwrap();
        let cache = MaxwellThreeDLoweringCache::default();
        let partial_source = channel
            .three_d()
            .shader_execution()
            .shader_local_memory()
            .address_upper()
            .source()
            .unwrap();
        assert!(matches!(
            preflight_maxwell_three_d_operation(
                channel.three_d(),
                &resources,
                MaxwellThreeDOperationTrigger::DrawVertexArray {
                    source: partial_source,
                    vertex_count: 3,
                },
                None,
                FrontendSubmissionId::new(9),
                Vec::new(),
                &lowering_capabilities(BackendFeatures::empty()),
                &cache,
            ),
            Err(MaxwellThreeDLoweringError::IncompleteDraw(
                "SET_SHADER_LOCAL_MEMORY_A-D"
            ))
        ));

        for (method, argument) in [
            (0x0794, 0x0008_0000),
            (0x0798, 0),
            (0x079c, 0x0408_0000),
            (0x07a0, 0),
            (0x077c, 0xff00_0000),
        ] {
            program_three_d(&mut channel, method, argument);
        }
        let source = channel
            .three_d()
            .shader_execution()
            .shader_local_memory()
            .default_size_per_warp()
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
                FrontendSubmissionId::new(10),
                Vec::new(),
                &lowering_capabilities(BackendFeatures::empty()),
                &cache,
            ),
            Err(MaxwellThreeDLoweringError::ShaderTranslationRequired)
        ));

        program_three_d(&mut channel, 0x07a0, 0x100);
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
            Err(MaxwellThreeDLoweringError::UnsupportedShaderLocalMemorySemantics {
                default_size_per_warp,
            }) if default_size_per_warp.bytes() == 0x100
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
    fn active_zcull_region_is_typed_source_preserving_and_pipeline_neutral() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let stats_before = *channel.three_d().zcull().stats_enable();
        assert_eq!(
            channel.three_d().zcull().active_region().origin(),
            MaxwellThreeDRegisterOrigin::Unset
        );

        for argument in [0, 1, 0x3f] {
            let dependencies_before = channel.three_d().pipeline_dependencies(&[]);
            let decoded = packet(0x1590 / 4, argument);
            let dispatch = dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0],
            )
            .unwrap();
            let method = dispatch.methods()[0];
            let source = method.method().source();
            let register = channel.three_d().zcull().active_region();
            let value = register.value().copied().unwrap();

            assert_eq!(method.metadata().method_name(), "SET_ACTIVE_ZCULL_REGION");
            assert_eq!(
                method.effect(),
                MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::ZCull(
                    MaxwellThreeDZCullStateWrite::ActiveRegion { value, source }
                ))
            );
            assert!(dispatch.operations().is_empty());
            assert_eq!(value.id(), argument as u8);
            assert_eq!(value.raw(), argument);
            assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
            assert_eq!(register.raw(), Some(argument));
            assert_eq!(register.source(), Some(source));
            assert_eq!(channel.three_d().zcull().stats_enable(), &stats_before);
            assert_eq!(
                channel.three_d().pipeline_dependencies(&[]),
                dependencies_before
            );
            assert_eq!(channel.frontend(), frontend_before);
            assert_eq!(channel.two_d(), &two_d_before);
        }
    }

    #[test]
    fn active_zcull_region_reserved_bits_and_packet_suffix_are_rejected_atomically() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        program_three_d(&mut channel, 0x1590, 0x3f);

        for argument in [0x40, 0x80, 0x8000_0000, u32::MAX] {
            let frontend_before = channel.frontend();
            let two_d_before = channel.two_d().clone();
            let three_d_before = channel.three_d().clone();
            let decoded = packet(0x1590 / 4, argument);

            assert!(matches!(
                dispatch_maxwell_engine_packet(
                    &mut channel,
                    FrontendSubmissionId::new(3),
                    &decoded.packets()[0]
                ),
                Err(MaxwellEngineDispatchError::InvalidMethodValue {
                    source,
                    defined_mask: 0x3f,
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
        let decoded = incrementing_packet(0x1590 / 4, &[0, 0]);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
                if source.method() == GpuMethodId(0x1594)
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }

    #[test]
    fn active_zcull_region_without_region_storage_does_not_change_draw_or_clear_semantics() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        program_three_d(&mut channel, 0x121c, 0);
        program_three_d(&mut channel, 0x1590, 0x3f);
        let resources =
            resolve_maxwell_three_d_resources(channel.three_d(), &resource_address_space())
                .unwrap();
        let capabilities = lowering_capabilities(BackendFeatures::empty());
        let cache = MaxwellThreeDLoweringCache::default();
        let cache_before = cache.clone();
        let source = channel.three_d().zcull().active_region().source().unwrap();

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
    fn ps_output_sample_mask_usage_is_typed_source_preserving_coverage_state() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        let fixed_function_before = channel.three_d().fixed_function().clone();
        let csaa_before = *channel.three_d().coverage().csaa_enable();
        let two_d_before = channel.two_d().clone();
        assert_eq!(
            channel
                .three_d()
                .coverage()
                .ps_output_sample_mask_usage()
                .origin(),
            MaxwellThreeDRegisterOrigin::Unset
        );

        for argument in 0..=3 {
            let decoded = packet(0x0300 / 4, argument);
            let dispatch = dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0],
            )
            .unwrap();
            let source = dispatch.methods()[0].method().source();
            let register = channel.three_d().coverage().ps_output_sample_mask_usage();
            let value = register.value().copied().unwrap();

            assert_eq!(
                dispatch.methods()[0].metadata().method_name(),
                "SET_PS_OUTPUT_SAMPLE_MASK_USAGE"
            );
            assert_eq!(
                dispatch.methods()[0].effect(),
                MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::Coverage(
                    MaxwellThreeDCoverageStateWrite::PsOutputSampleMaskUsage { value, source }
                ))
            );
            assert_eq!(value.enabled(), argument & 1 != 0);
            assert_eq!(value.qualify_by_anti_alias_enable(), argument & 2 != 0);
            assert_eq!(value.effective(Some(false)), Some(matches!(argument, 1)));
            assert_eq!(value.effective(Some(true)), Some(matches!(argument, 1 | 3)));
            assert_eq!(
                value.effective(None),
                match argument {
                    1 => Some(true),
                    3 => None,
                    _ => Some(false),
                }
            );
            assert_eq!(register.raw(), Some(argument));
            assert_eq!(register.value().copied(), Some(value));
            assert_eq!(register.source(), Some(source));
            assert_eq!(channel.three_d().coverage().csaa_enable(), &csaa_before);
            assert_eq!(channel.three_d().fixed_function(), &fixed_function_before);
            assert_eq!(channel.two_d(), &two_d_before);
            assert!(dispatch.operations().is_empty());
        }
    }

    #[test]
    fn primitive_circular_buffer_throttle_is_typed_source_preserving_and_pipeline_neutral() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        let two_d_before = channel.two_d().clone();
        assert_eq!(
            channel
                .three_d()
                .vertex_input()
                .primitive()
                .circular_buffer_throttle()
                .origin(),
            MaxwellThreeDRegisterOrigin::Unset
        );

        for argument in [0, 1, MAXWELL_THREE_D_PRIMITIVE_AREA_MAX] {
            let dependencies_before = channel.three_d().pipeline_dependencies(&[]);
            let decoded = packet(0x02d0 / 4, argument);
            let dispatch = dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0],
            )
            .unwrap();
            let source = dispatch.methods()[0].method().source();
            let register = channel
                .three_d()
                .vertex_input()
                .primitive()
                .circular_buffer_throttle();
            let value = register.value().copied().unwrap();

            assert_eq!(
                dispatch.methods()[0].metadata().method_name(),
                "SET_PRIM_CIRCULAR_BUFFER_THROTTLE"
            );
            assert_eq!(
                dispatch.methods()[0].effect(),
                MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::VertexInput(
                    MaxwellThreeDVertexInputWrite::PrimitiveCircularBufferThrottle {
                        value,
                        source,
                    }
                ))
            );
            assert_eq!(value.primitive_area(), argument);
            assert_eq!(register.raw(), Some(argument));
            assert_eq!(register.source(), Some(source));
            assert_eq!(
                channel.three_d().pipeline_dependencies(&[]),
                dependencies_before
            );
            assert_eq!(channel.two_d(), &two_d_before);
            assert!(dispatch.operations().is_empty());
        }
    }

    #[test]
    fn unorm8_color_reduction_thresholds_are_typed_source_preserving_and_isolated() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        let fixed_function_before = channel.three_d().fixed_function().clone();
        let render_targets_before = channel.three_d().render_targets().clone();
        let two_d_before = channel.two_d().clone();
        let pipeline_dependencies_before = channel.three_d().pipeline_dependencies(&[]);
        assert_eq!(
            channel
                .three_d()
                .color_reduction()
                .thresholds_unorm8()
                .origin(),
            MaxwellThreeDRegisterOrigin::Unset
        );

        for (argument, all_hit_once, all_covered) in [
            (0, 0, 0),
            (0x0000_00ff, 0xff, 0),
            (0x00ff_0000, 0, 0xff),
            (0x00ff_00ff, 0xff, 0xff),
        ] {
            let decoded = packet(0x10cc / 4, argument);
            let dispatch = dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0],
            )
            .unwrap();
            let source = dispatch.methods()[0].method().source();
            let value = MaxwellThreeDColorReductionThresholdsUnorm8::new(
                MaxwellThreeDUnorm8::new(all_hit_once),
                MaxwellThreeDUnorm8::new(all_covered),
            );
            let register = channel.three_d().color_reduction().thresholds_unorm8();

            assert_eq!(
                dispatch.methods()[0].metadata().method_name(),
                "SET_REDUCE_COLOR_THRESHOLDS_UNORM8"
            );
            assert_eq!(
                dispatch.methods()[0].effect(),
                MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::ColorReduction(
                    MaxwellThreeDColorReductionStateWrite::ThresholdsUnorm8 { value, source }
                ))
            );
            assert!(dispatch.operations().is_empty());
            assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
            assert_eq!(register.raw(), Some(argument));
            assert_eq!(register.value().copied(), Some(value));
            assert_eq!(register.source(), Some(source));
            assert_eq!(value.raw(), argument);
            assert_eq!(value.all_covered_all_hit_once().raw(), all_hit_once);
            assert_eq!(value.all_covered().raw(), all_covered);
            assert_eq!(channel.three_d().fixed_function(), &fixed_function_before);
            assert_eq!(channel.three_d().render_targets(), &render_targets_before);
            assert_eq!(channel.two_d(), &two_d_before);
            assert_eq!(
                channel.three_d().pipeline_dependencies(&[]),
                pipeline_dependencies_before
            );
        }

        let unorm10_before = *channel.three_d().color_reduction().thresholds_unorm10();
        program_three_d(&mut channel, 0x10cc, 0x0056_0078);
        assert_eq!(
            channel.three_d().color_reduction().thresholds_unorm10(),
            &unorm10_before
        );
    }

    #[test]
    fn unorm10_color_reduction_thresholds_are_typed_source_preserving_and_independent() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        program_three_d(&mut channel, 0x10cc, 0x0012_0034);
        let unorm8_before = *channel.three_d().color_reduction().thresholds_unorm8();
        let fixed_function_before = channel.three_d().fixed_function().clone();
        let render_targets_before = channel.three_d().render_targets().clone();
        let two_d_before = channel.two_d().clone();
        let pipeline_dependencies_before = channel.three_d().pipeline_dependencies(&[]);
        assert_eq!(
            channel
                .three_d()
                .color_reduction()
                .thresholds_unorm10()
                .origin(),
            MaxwellThreeDRegisterOrigin::Unset
        );

        for (argument, all_hit_once, all_covered) in [
            (0, 0, 0),
            (0x0000_00ff, 0xff, 0),
            (0x00ff_0000, 0, 0xff),
            (0x00ff_00ff, 0xff, 0xff),
        ] {
            let decoded = packet(0x10e0 / 4, argument);
            let dispatch = dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0],
            )
            .unwrap();
            let source = dispatch.methods()[0].method().source();
            let value = MaxwellThreeDColorReductionThresholdsUnorm10::new(
                MaxwellThreeDUnorm8::new(all_hit_once),
                MaxwellThreeDUnorm8::new(all_covered),
            );
            let register = channel.three_d().color_reduction().thresholds_unorm10();

            assert_eq!(
                dispatch.methods()[0].metadata().method_name(),
                "SET_REDUCE_COLOR_THRESHOLDS_UNORM10"
            );
            assert_eq!(
                dispatch.methods()[0].effect(),
                MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::ColorReduction(
                    MaxwellThreeDColorReductionStateWrite::ThresholdsUnorm10 { value, source }
                ))
            );
            assert!(dispatch.operations().is_empty());
            assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
            assert_eq!(register.raw(), Some(argument));
            assert_eq!(register.value().copied(), Some(value));
            assert_eq!(register.source(), Some(source));
            assert_eq!(value.raw(), argument);
            assert_eq!(value.all_covered_all_hit_once().raw(), all_hit_once);
            assert_eq!(value.all_covered().raw(), all_covered);
            assert_eq!(
                channel.three_d().color_reduction().thresholds_unorm8(),
                &unorm8_before
            );
            assert_eq!(channel.three_d().fixed_function(), &fixed_function_before);
            assert_eq!(channel.three_d().render_targets(), &render_targets_before);
            assert_eq!(channel.two_d(), &two_d_before);
            assert_eq!(
                channel.three_d().pipeline_dependencies(&[]),
                pipeline_dependencies_before
            );
        }
    }

    #[test]
    fn unorm16_color_reduction_thresholds_are_typed_source_preserving_and_independent() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        program_three_d(&mut channel, 0x10cc, 0x0012_0034);
        program_three_d(&mut channel, 0x10e0, 0x0056_0078);
        let unorm8_before = *channel.three_d().color_reduction().thresholds_unorm8();
        let unorm10_before = *channel.three_d().color_reduction().thresholds_unorm10();
        let fixed_function_before = channel.three_d().fixed_function().clone();
        let render_targets_before = channel.three_d().render_targets().clone();
        let two_d_before = channel.two_d().clone();
        let pipeline_dependencies_before = channel.three_d().pipeline_dependencies(&[]);
        assert_eq!(
            channel
                .three_d()
                .color_reduction()
                .thresholds_unorm16()
                .origin(),
            MaxwellThreeDRegisterOrigin::Unset
        );

        for (argument, all_hit_once, all_covered) in [
            (0, 0, 0),
            (0x0000_00ff, 0xff, 0),
            (0x00ff_0000, 0, 0xff),
            (0x00ff_00ff, 0xff, 0xff),
        ] {
            let decoded = packet(0x10e4 / 4, argument);
            let dispatch = dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0],
            )
            .unwrap();
            let source = dispatch.methods()[0].method().source();
            let value = MaxwellThreeDColorReductionThresholdsUnorm16::new(
                MaxwellThreeDUnorm8::new(all_hit_once),
                MaxwellThreeDUnorm8::new(all_covered),
            );
            let register = channel.three_d().color_reduction().thresholds_unorm16();

            assert_eq!(
                dispatch.methods()[0].metadata().method_name(),
                "SET_REDUCE_COLOR_THRESHOLDS_UNORM16"
            );
            assert_eq!(
                dispatch.methods()[0].effect(),
                MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::ColorReduction(
                    MaxwellThreeDColorReductionStateWrite::ThresholdsUnorm16 { value, source }
                ))
            );
            assert!(dispatch.operations().is_empty());
            assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
            assert_eq!(register.raw(), Some(argument));
            assert_eq!(register.value().copied(), Some(value));
            assert_eq!(register.source(), Some(source));
            assert_eq!(value.raw(), argument);
            assert_eq!(value.all_covered_all_hit_once().raw(), all_hit_once);
            assert_eq!(value.all_covered().raw(), all_covered);
            assert_eq!(
                channel.three_d().color_reduction().thresholds_unorm8(),
                &unorm8_before
            );
            assert_eq!(
                channel.three_d().color_reduction().thresholds_unorm10(),
                &unorm10_before
            );
            assert_eq!(channel.three_d().fixed_function(), &fixed_function_before);
            assert_eq!(channel.three_d().render_targets(), &render_targets_before);
            assert_eq!(channel.two_d(), &two_d_before);
            assert_eq!(
                channel.three_d().pipeline_dependencies(&[]),
                pipeline_dependencies_before
            );
        }

        let unorm16_before = *channel.three_d().color_reduction().thresholds_unorm16();
        program_three_d(&mut channel, 0x10e0, 0x009a_00bc);
        assert_eq!(
            channel.three_d().color_reduction().thresholds_unorm16(),
            &unorm16_before
        );
    }

    #[test]
    fn fp16_color_reduction_thresholds_are_typed_source_preserving_and_independent() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        program_three_d(&mut channel, 0x10cc, 0x0012_0034);
        program_three_d(&mut channel, 0x10e0, 0x0056_0078);
        program_three_d(&mut channel, 0x10e4, 0x009a_00bc);
        let unorm8_before = *channel.three_d().color_reduction().thresholds_unorm8();
        let unorm10_before = *channel.three_d().color_reduction().thresholds_unorm10();
        let unorm16_before = *channel.three_d().color_reduction().thresholds_unorm16();
        let fixed_function_before = channel.three_d().fixed_function().clone();
        let render_targets_before = channel.three_d().render_targets().clone();
        let two_d_before = channel.two_d().clone();
        let pipeline_dependencies_before = channel.three_d().pipeline_dependencies(&[]);
        assert_eq!(
            channel
                .three_d()
                .color_reduction()
                .thresholds_fp16()
                .origin(),
            MaxwellThreeDRegisterOrigin::Unset
        );

        for (argument, all_hit_once, all_covered) in [
            (0, 0, 0),
            (0x0000_00ff, 0xff, 0),
            (0x00ff_0000, 0, 0xff),
            (0x00ff_00ff, 0xff, 0xff),
        ] {
            let decoded = packet(0x10ec / 4, argument);
            let dispatch = dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0],
            )
            .unwrap();
            let source = dispatch.methods()[0].method().source();
            let value = MaxwellThreeDColorReductionThresholdsFp16::new(
                MaxwellThreeDColorReductionFp16Threshold::new(all_hit_once),
                MaxwellThreeDColorReductionFp16Threshold::new(all_covered),
            );
            let register = channel.three_d().color_reduction().thresholds_fp16();

            assert_eq!(
                dispatch.methods()[0].metadata().method_name(),
                "SET_REDUCE_COLOR_THRESHOLDS_FP16"
            );
            assert_eq!(
                dispatch.methods()[0].effect(),
                MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::ColorReduction(
                    MaxwellThreeDColorReductionStateWrite::ThresholdsFp16 { value, source }
                ))
            );
            assert!(dispatch.operations().is_empty());
            assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
            assert_eq!(register.raw(), Some(argument));
            assert_eq!(register.value().copied(), Some(value));
            assert_eq!(register.source(), Some(source));
            assert_eq!(value.raw(), argument);
            assert_eq!(value.all_covered_all_hit_once().raw(), all_hit_once);
            assert_eq!(value.all_covered().raw(), all_covered);
            assert_eq!(
                channel.three_d().color_reduction().thresholds_unorm8(),
                &unorm8_before
            );
            assert_eq!(
                channel.three_d().color_reduction().thresholds_unorm10(),
                &unorm10_before
            );
            assert_eq!(
                channel.three_d().color_reduction().thresholds_unorm16(),
                &unorm16_before
            );
            assert_eq!(channel.three_d().fixed_function(), &fixed_function_before);
            assert_eq!(channel.three_d().render_targets(), &render_targets_before);
            assert_eq!(channel.two_d(), &two_d_before);
            assert_eq!(
                channel.three_d().pipeline_dependencies(&[]),
                pipeline_dependencies_before
            );
        }
    }

    #[test]
    fn srgb8_color_reduction_thresholds_are_typed_source_preserving_and_independent() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        program_three_d(&mut channel, 0x10cc, 0x0012_0034);
        program_three_d(&mut channel, 0x10e0, 0x0056_0078);
        program_three_d(&mut channel, 0x10e4, 0x009a_00bc);
        program_three_d(&mut channel, 0x10ec, 0x00de_00f0);
        let unorm8_before = *channel.three_d().color_reduction().thresholds_unorm8();
        let unorm10_before = *channel.three_d().color_reduction().thresholds_unorm10();
        let unorm16_before = *channel.three_d().color_reduction().thresholds_unorm16();
        let fp16_before = *channel.three_d().color_reduction().thresholds_fp16();
        let fixed_function_before = channel.three_d().fixed_function().clone();
        let render_targets_before = channel.three_d().render_targets().clone();
        let two_d_before = channel.two_d().clone();
        let pipeline_dependencies_before = channel.three_d().pipeline_dependencies(&[]);
        assert_eq!(
            channel
                .three_d()
                .color_reduction()
                .thresholds_srgb8()
                .origin(),
            MaxwellThreeDRegisterOrigin::Unset
        );

        for (argument, all_hit_once, all_covered) in [
            (0, 0, 0),
            (0x0000_00ff, 0xff, 0),
            (0x00ff_0000, 0, 0xff),
            (0x00ff_00ff, 0xff, 0xff),
        ] {
            let decoded = packet(0x10f0 / 4, argument);
            let dispatch = dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0],
            )
            .unwrap();
            let source = dispatch.methods()[0].method().source();
            let value = MaxwellThreeDColorReductionThresholdsSrgb8::new(
                MaxwellThreeDColorReductionSrgb8Threshold::new(all_hit_once),
                MaxwellThreeDColorReductionSrgb8Threshold::new(all_covered),
            );
            let register = channel.three_d().color_reduction().thresholds_srgb8();

            assert_eq!(
                dispatch.methods()[0].metadata().method_name(),
                "SET_REDUCE_COLOR_THRESHOLDS_SRGB8"
            );
            assert_eq!(
                dispatch.methods()[0].effect(),
                MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::ColorReduction(
                    MaxwellThreeDColorReductionStateWrite::ThresholdsSrgb8 { value, source }
                ))
            );
            assert!(dispatch.operations().is_empty());
            assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
            assert_eq!(register.raw(), Some(argument));
            assert_eq!(register.value().copied(), Some(value));
            assert_eq!(register.source(), Some(source));
            assert_eq!(value.raw(), argument);
            assert_eq!(value.all_covered_all_hit_once().raw(), all_hit_once);
            assert_eq!(value.all_covered().raw(), all_covered);
            assert_eq!(
                channel.three_d().color_reduction().thresholds_unorm8(),
                &unorm8_before
            );
            assert_eq!(
                channel.three_d().color_reduction().thresholds_unorm10(),
                &unorm10_before
            );
            assert_eq!(
                channel.three_d().color_reduction().thresholds_unorm16(),
                &unorm16_before
            );
            assert_eq!(
                channel.three_d().color_reduction().thresholds_fp16(),
                &fp16_before
            );
            assert_eq!(channel.three_d().fixed_function(), &fixed_function_before);
            assert_eq!(channel.three_d().render_targets(), &render_targets_before);
            assert_eq!(channel.two_d(), &two_d_before);
            assert_eq!(
                channel.three_d().pipeline_dependencies(&[]),
                pipeline_dependencies_before
            );
        }
    }

    #[test]
    fn invalid_color_reduction_values_and_packet_suffix_are_rejected_atomically() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        program_three_d(&mut channel, 0x10cc, 0x0000_00ff);

        program_three_d(&mut channel, 0x10e0, 0x00ff_0000);
        program_three_d(&mut channel, 0x10e4, 0x0000_00ff);
        program_three_d(&mut channel, 0x10ec, 0x0000_00ff);
        program_three_d(&mut channel, 0x10f0, 0x0000_00ff);

        for method in [0x10cc, 0x10e0, 0x10e4, 0x10ec, 0x10f0] {
            for argument in [0x0000_0100, 0x0000_ff00, 0x0100_0000, 0xff00_0000] {
                let frontend_before = channel.frontend();
                let two_d_before = channel.two_d().clone();
                let three_d_before = channel.three_d().clone();
                let decoded = packet(method / 4, argument);

                assert!(matches!(
                    dispatch_maxwell_engine_packet(
                        &mut channel,
                        FrontendSubmissionId::new(3),
                        &decoded.packets()[0]
                    ),
                    Err(MaxwellEngineDispatchError::InvalidMethodValue {
                        source,
                        defined_mask: 0x00ff_00ff,
                        ..
                    }) if source.argument() == argument
                ));
                assert_eq!(channel.frontend(), frontend_before);
                assert_eq!(channel.two_d(), &two_d_before);
                assert_eq!(channel.three_d(), &three_d_before);
            }
        }

        for argument in [2, 3, u32::MAX] {
            let three_d_before = channel.three_d().clone();
            let decoded = packet(0x0d9c / 4, argument);
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
            assert_eq!(channel.three_d(), &three_d_before);
        }

        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = incrementing_packet(0x10cc / 4, &[0x00ff_00ff, 0]);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
                if source.method() == GpuMethodId(0x10d0)
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);

        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = incrementing_packet(0x10e0 / 4, &[0x00ff_00ff, 0x0000_0100]);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::InvalidMethodValue {
                source,
                defined_mask: 0x00ff_00ff,
                ..
            }) if source.method() == GpuMethodId(0x10e4)
                && source.argument() == 0x0000_0100
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);

        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = incrementing_packet(0x10e4 / 4, &[0x00ff_00ff, 0]);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
                if source.method() == GpuMethodId(0x10e8)
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);

        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = incrementing_packet(0x10ec / 4, &[0x00ff_00ff, 0x0000_0100]);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::InvalidMethodValue {
                source,
                defined_mask: 0x00ff_00ff,
                ..
            }) if source.method() == GpuMethodId(0x10f0)
                && source.argument() == 0x0000_0100
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);

        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = incrementing_packet(0x10f0 / 4, &[0x00ff_00ff, 0]);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
                if source.method() == GpuMethodId(0x10f4)
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
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
    fn invalid_ps_output_sample_mask_usage_and_packet_suffix_are_atomic() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        program_three_d(&mut channel, 0x0300, 3);

        for argument in [4, 7, 0x8000_0000, u32::MAX] {
            let frontend_before = channel.frontend();
            let two_d_before = channel.two_d().clone();
            let three_d_before = channel.three_d().clone();
            let decoded = packet(0x0300 / 4, argument);

            assert!(matches!(
                dispatch_maxwell_engine_packet(
                    &mut channel,
                    FrontendSubmissionId::new(3),
                    &decoded.packets()[0]
                ),
                Err(MaxwellEngineDispatchError::InvalidMethodValue {
                    source,
                    defined_mask: 3,
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
        let decoded = incrementing_packet(0x0300 / 4, &[0, 0]);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::MissingCapability { source, .. })
                if source.method() == GpuMethodId(0x0304)
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }

    #[test]
    fn invalid_primitive_circular_buffer_throttle_and_packet_suffix_are_atomic() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        program_three_d(&mut channel, 0x02d0, MAXWELL_THREE_D_PRIMITIVE_AREA_MAX);

        for argument in [0x0040_0000, 0x8000_0000, u32::MAX] {
            let frontend_before = channel.frontend();
            let two_d_before = channel.two_d().clone();
            let three_d_before = channel.three_d().clone();
            let decoded = packet(0x02d0 / 4, argument);

            assert!(matches!(
                dispatch_maxwell_engine_packet(
                    &mut channel,
                    FrontendSubmissionId::new(3),
                    &decoded.packets()[0]
                ),
                Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                    source,
                    method_name: "SET_PRIM_CIRCULAR_BUFFER_THROTTLE",
                    reason: "reserved bits are set",
                }) if source.argument() == argument
            ));
            assert_eq!(channel.frontend(), frontend_before);
            assert_eq!(channel.two_d(), &two_d_before);
            assert_eq!(channel.three_d(), &three_d_before);
        }

        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = incrementing_packet(0x02d0 / 4, &[0, 0]);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
                if source.method() == GpuMethodId(0x02d4)
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }

    #[test]
    fn patch_size_is_typed_source_preserving_and_reserved_bits_are_atomic() {
        let mut channel = channel();
        bind_three_d(&mut channel);

        for argument in [0, 1, 3, 32, 0xff] {
            let decoded = packet(0x0dcc / 4, argument);
            let dispatch = dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0],
            )
            .unwrap();
            let source = dispatch.methods()[0].method().source();
            let value = MaxwellThreeDPatchSize::new(argument as u8);
            let register = channel.three_d().vertex_input().primitive().patch_size();

            assert_eq!(dispatch.methods()[0].metadata().method_name(), "SET_PATCH");
            assert_eq!(
                dispatch.methods()[0].effect(),
                MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::VertexInput(
                    MaxwellThreeDVertexInputWrite::PatchSize { value, source }
                ))
            );
            assert!(dispatch.operations().is_empty());
            assert_eq!(value.control_points(), argument as u8);
            assert_eq!(value.raw(), argument);
            assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
            assert_eq!(register.raw(), Some(argument));
            assert_eq!(register.value(), Some(&value));
            assert_eq!(register.source(), Some(source));
        }

        for argument in [0x100, 0x8000_0000, u32::MAX] {
            let frontend_before = channel.frontend();
            let two_d_before = channel.two_d().clone();
            let three_d_before = channel.three_d().clone();
            let decoded = packet(0x0dcc / 4, argument);
            assert!(matches!(
                dispatch_maxwell_engine_packet(
                    &mut channel,
                    FrontendSubmissionId::new(3),
                    &decoded.packets()[0]
                ),
                Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                    source,
                    method_name: "SET_PATCH",
                    reason: "reserved bits are set",
                }) if source.argument() == argument
            ));
            assert_eq!(channel.frontend(), frontend_before);
            assert_eq!(channel.two_d(), &two_d_before);
            assert_eq!(channel.three_d(), &three_d_before);
        }

        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = incrementing_packet(0x0dcc / 4, &[3, 0]);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
                if source.method() == GpuMethodId(0x0dd0)
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }

    #[test]
    fn point_rasterization_state_is_typed_source_preserving_and_atomic() {
        let mut channel = channel();
        bind_three_d(&mut channel);

        for (argument, r_mode, origin, texture_mask) in [
            (
                0,
                MaxwellThreeDPointSpriteRMode::Zero,
                MaxwellThreeDPointSpriteOrigin::Bottom,
                0,
            ),
            (
                0x000d,
                MaxwellThreeDPointSpriteRMode::FromR,
                MaxwellThreeDPointSpriteOrigin::Top,
                1,
            ),
            (
                0x1ffe,
                MaxwellThreeDPointSpriteRMode::FromS,
                MaxwellThreeDPointSpriteOrigin::Top,
                0x03ff,
            ),
        ] {
            let decoded = packet(0x1604 / 4, argument);
            let dispatch = dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0],
            )
            .unwrap();
            let source = dispatch.methods()[0].method().source();
            let register = channel.three_d().raster().point_sprite_select();
            let value = register.value().copied().unwrap();

            assert_eq!(
                dispatch.methods()[0].metadata().method_name(),
                "SET_POINT_SPRITE_SELECT"
            );
            assert_eq!(value.r_mode(), r_mode);
            assert_eq!(value.origin(), origin);
            assert_eq!(value.generated_texture_mask(), texture_mask);
            assert_eq!(value.raw(), argument);
            for texture in 0..10 {
                assert_eq!(
                    value.generates_texture(texture),
                    texture_mask & (1 << texture) != 0
                );
            }
            assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
            assert_eq!(register.raw(), Some(argument));
            assert_eq!(register.source(), Some(source));
        }

        for (argument, mode) in [
            (0, MaxwellThreeDPointCenterMode::OpenGl),
            (1, MaxwellThreeDPointCenterMode::Direct3D),
        ] {
            let decoded = packet(0x165c / 4, argument);
            let dispatch = dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0],
            )
            .unwrap();
            let source = dispatch.methods()[0].method().source();
            let register = channel.three_d().raster().point_center_mode();

            assert_eq!(
                dispatch.methods()[0].metadata().method_name(),
                "SET_POINT_CENTER_MODE"
            );
            assert_eq!(register.value(), Some(&mode));
            assert_eq!(register.raw(), Some(mode.raw()));
            assert_eq!(register.source(), Some(source));
        }

        for (method, argument, mask) in [
            (0x1604, 0x1fff, 0x1fff),
            (0x1604, 0x2000, 0x1fff),
            (0x165c, 2, 1),
        ] {
            let frontend_before = channel.frontend();
            let two_d_before = channel.two_d().clone();
            let three_d_before = channel.three_d().clone();
            let decoded = packet(method / 4, argument);
            assert!(matches!(
                dispatch_maxwell_engine_packet(
                    &mut channel,
                    FrontendSubmissionId::new(3),
                    &decoded.packets()[0]
                ),
                Err(MaxwellEngineDispatchError::InvalidMethodValue {
                    source,
                    defined_mask,
                    ..
                }) if source.argument() == argument && defined_mask == mask
            ));
            assert_eq!(channel.frontend(), frontend_before);
            assert_eq!(channel.two_d(), &two_d_before);
            assert_eq!(channel.three_d(), &three_d_before);
        }

        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = incrementing_packet(0x1604 / 4, &[0, 0x100]);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                source,
                method_name: "SET_PROGRAM_REGION_A",
                ..
            }) if source.method() == GpuMethodId(0x1608)
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }

    #[test]
    fn edge_flag_state_is_typed_source_preserving_and_atomic() {
        let mut channel = channel();
        bind_three_d(&mut channel);

        for (argument, expected) in [
            (0, MaxwellThreeDEdgeFlag::Disabled),
            (1, MaxwellThreeDEdgeFlag::Enabled),
        ] {
            let decoded = packet(0x15e4 / 4, argument);
            let dispatch = dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0],
            )
            .unwrap();
            let source = dispatch.methods()[0].method().source();
            let register = channel.three_d().raster().edge_flag();

            assert_eq!(
                dispatch.methods()[0].metadata().method_name(),
                "SET_EDGE_FLAG"
            );
            assert_eq!(register.value(), Some(&expected));
            assert_eq!(register.raw(), Some(expected.raw()));
            assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
            assert_eq!(register.source(), Some(source));
        }

        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = packet(0x15e4 / 4, 2);
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

        let decoded = incrementing_packet(0x15e4 / 4, &[0, 0]);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
                if source.method() == GpuMethodId(0x15e8)
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
        let decoded = incrementing_packet(0x12d4 / 4, &[0x1d01, 0, 0, 0]);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
                if source.method() == GpuMethodId(0x12e0)
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
    fn blend_controls_are_typed_source_preserving_and_family_isolated() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        let common_before = *channel.three_d().fixed_function().blend_enable_common();
        let per_target_enable_before = *channel.three_d().fixed_function().blend_enable();
        let per_target_state_before = *channel.three_d().fixed_function().per_target_blend();
        let two_d_before = channel.two_d().clone();

        for (argument, value) in [
            (0, MaxwellThreeDBlendPerFormatEnable::Disabled),
            (0x10, MaxwellThreeDBlendPerFormatEnable::Enabled),
        ] {
            let pixel_kill_before = *channel
                .three_d()
                .fixed_function()
                .blend_controls()
                .float_pixel_kill_enable();
            let zero_product_before = *channel
                .three_d()
                .fixed_function()
                .blend_controls()
                .zero_times_anything_is_zero();
            let decoded = packet(0x1140 / 4, argument);
            let dispatch = dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0],
            )
            .unwrap();
            let source = dispatch.methods()[0].method().source();
            let fixed = channel.three_d().fixed_function();
            let register = fixed.blend_controls().per_format_enable();

            assert_eq!(
                dispatch.methods()[0].metadata().method_name(),
                "SET_BLEND_PER_FORMAT_ENABLE"
            );
            assert_eq!(
                dispatch.methods()[0].effect(),
                MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::FixedFunction(
                    MaxwellThreeDFixedFunctionWrite::BlendPerFormatEnable { value, source }
                ))
            );
            assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
            assert_eq!(register.raw(), Some(argument));
            assert_eq!(register.value().copied(), Some(value));
            assert_eq!(register.source(), Some(source));
            assert_eq!(value.raw(), argument);
            assert_eq!(
                fixed.blend_controls().float_pixel_kill_enable(),
                &pixel_kill_before
            );
            assert_eq!(
                fixed.blend_controls().zero_times_anything_is_zero(),
                &zero_product_before
            );
        }

        for (argument, value) in [
            (0, MaxwellThreeDBlendFloatPixelKillEnable::Disallowed),
            (1, MaxwellThreeDBlendFloatPixelKillEnable::Allowed),
        ] {
            let per_format_before = *channel
                .three_d()
                .fixed_function()
                .blend_controls()
                .per_format_enable();
            let zero_product_before = *channel
                .three_d()
                .fixed_function()
                .blend_controls()
                .zero_times_anything_is_zero();
            let decoded = packet(0x0fdc / 4, argument);
            let dispatch = dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0],
            )
            .unwrap();
            let source = dispatch.methods()[0].method().source();
            let fixed = channel.three_d().fixed_function();
            let register = fixed.blend_controls().float_pixel_kill_enable();

            assert_eq!(
                dispatch.methods()[0].metadata().method_name(),
                "SET_BLEND_OPT_CONTROL"
            );
            assert_eq!(
                dispatch.methods()[0].effect(),
                MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::FixedFunction(
                    MaxwellThreeDFixedFunctionWrite::BlendFloatPixelKillEnable { value, source }
                ))
            );
            assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
            assert_eq!(register.raw(), Some(argument));
            assert_eq!(register.value().copied(), Some(value));
            assert_eq!(register.source(), Some(source));
            assert_eq!(value.raw(), argument);
            assert_eq!(
                fixed.blend_controls().per_format_enable(),
                &per_format_before
            );
            assert_eq!(
                fixed.blend_controls().zero_times_anything_is_zero(),
                &zero_product_before
            );
        }

        for (argument, value) in [
            (0, MaxwellThreeDBlendZeroTimesAnythingIsZero::Disabled),
            (1, MaxwellThreeDBlendZeroTimesAnythingIsZero::Enabled),
        ] {
            let per_format_before = *channel
                .three_d()
                .fixed_function()
                .blend_controls()
                .per_format_enable();
            let pixel_kill_before = *channel
                .three_d()
                .fixed_function()
                .blend_controls()
                .float_pixel_kill_enable();
            let decoded = packet(0x19c0 / 4, argument);
            let dispatch = dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0],
            )
            .unwrap();
            let source = dispatch.methods()[0].method().source();
            let fixed = channel.three_d().fixed_function();
            let register = fixed.blend_controls().zero_times_anything_is_zero();

            assert_eq!(
                dispatch.methods()[0].metadata().method_name(),
                "SET_BLEND_FLOAT_OPTION"
            );
            assert_eq!(
                dispatch.methods()[0].effect(),
                MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::FixedFunction(
                    MaxwellThreeDFixedFunctionWrite::BlendZeroTimesAnythingIsZero { value, source }
                ))
            );
            assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
            assert_eq!(register.raw(), Some(argument));
            assert_eq!(register.value().copied(), Some(value));
            assert_eq!(register.source(), Some(source));
            assert_eq!(value.raw(), argument);
            assert_eq!(
                fixed.blend_controls().per_format_enable(),
                &per_format_before
            );
            assert_eq!(
                fixed.blend_controls().float_pixel_kill_enable(),
                &pixel_kill_before
            );
        }

        let fixed = channel.three_d().fixed_function();
        assert_eq!(fixed.blend_enable_common(), &common_before);
        assert_eq!(fixed.blend_enable(), &per_target_enable_before);
        assert_eq!(fixed.per_target_blend(), &per_target_state_before);
        assert_eq!(channel.two_d(), &two_d_before);
    }

    #[test]
    fn blend_control_pipeline_dependencies_are_semantic_and_ignore_optimization_permission() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        program_three_d(&mut channel, 0x12e4, 0);
        program_three_d(&mut channel, 0x135c, 0);
        program_three_d(&mut channel, 0x1140, 0);
        program_three_d(&mut channel, 0x0fdc, 0);
        program_three_d(&mut channel, 0x19c0, 0);

        let disabled_dependencies = channel.three_d().pipeline_dependencies(&[0]);
        program_three_d(&mut channel, 0x1140, 0x10);
        program_three_d(&mut channel, 0x0fdc, 1);
        program_three_d(&mut channel, 0x19c0, 1);
        assert_eq!(
            channel.three_d().pipeline_dependencies(&[0]),
            disabled_dependencies
        );

        program_three_d(&mut channel, 0x1140, 0);
        program_three_d(&mut channel, 0x0fdc, 0);
        program_three_d(&mut channel, 0x19c0, 0);
        program_three_d(&mut channel, 0x135c, 1);
        let enabled_dependencies = channel.three_d().pipeline_dependencies(&[0]);

        program_three_d(&mut channel, 0x0fdc, 1);
        assert_eq!(
            channel.three_d().pipeline_dependencies(&[0]),
            enabled_dependencies
        );
        program_three_d(&mut channel, 0x1140, 0x10);
        let per_format_dependencies = channel.three_d().pipeline_dependencies(&[0]);
        assert_ne!(per_format_dependencies, enabled_dependencies);
        program_three_d(&mut channel, 0x19c0, 1);
        assert_ne!(
            channel.three_d().pipeline_dependencies(&[0]),
            per_format_dependencies
        );
    }

    #[test]
    fn invalid_blend_controls_and_packet_suffix_are_rejected_atomically() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        for (method, argument) in [(0x1140, 0x10), (0x0fdc, 1), (0x19c0, 1)] {
            program_three_d(&mut channel, method, argument);
        }

        for (method, method_name, arguments) in [
            (
                0x1140,
                "SET_BLEND_PER_FORMAT_ENABLE",
                [1, 0x20, 0x11, u32::MAX],
            ),
            (
                0x0fdc,
                "SET_BLEND_OPT_CONTROL",
                [2, 0x10, 0x8000_0000, u32::MAX],
            ),
            (
                0x19c0,
                "SET_BLEND_FLOAT_OPTION",
                [2, 0x10, 0x8000_0000, u32::MAX],
            ),
        ] {
            for argument in arguments {
                let frontend_before = channel.frontend();
                let two_d_before = channel.two_d().clone();
                let three_d_before = channel.three_d().clone();
                let decoded = packet(method / 4, argument);
                assert!(matches!(
                    dispatch_maxwell_engine_packet(
                        &mut channel,
                        FrontendSubmissionId::new(3),
                        &decoded.packets()[0]
                    ),
                    Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                        source,
                        method_name: actual,
                        ..
                    }) if actual == method_name && source.argument() == argument
                ));
                assert_eq!(channel.frontend(), frontend_before);
                assert_eq!(channel.two_d(), &two_d_before);
                assert_eq!(channel.three_d(), &three_d_before);
            }
        }

        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = incrementing_packet(0x1140 / 4, &[0, 0, 0]);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
                if source.method() == GpuMethodId(0x1148)
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
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
    fn ps_output_sample_mask_usage_obeys_aa_and_never_blocks_clear() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        program_three_d(&mut channel, 0x121c, 0);
        let resources =
            resolve_maxwell_three_d_resources(channel.three_d(), &resource_address_space())
                .unwrap();
        let cache = MaxwellThreeDLoweringCache::default();

        program_three_d(&mut channel, 0x1534, 0);
        program_three_d(&mut channel, 0x0300, 3);
        let inactive_dependencies = channel.three_d().pipeline_dependencies(&[]);
        let source = channel
            .three_d()
            .coverage()
            .ps_output_sample_mask_usage()
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
                FrontendSubmissionId::new(10),
                Vec::new(),
                &lowering_capabilities(BackendFeatures::empty()),
                &cache,
            ),
            Err(MaxwellThreeDLoweringError::ShaderTranslationRequired)
        ));

        program_three_d(&mut channel, 0x0300, 0);
        assert_eq!(
            channel.three_d().pipeline_dependencies(&[]),
            inactive_dependencies
        );

        program_three_d(&mut channel, 0x0300, 3);
        program_three_d(&mut channel, 0x1534, 1);
        assert_ne!(
            channel.three_d().pipeline_dependencies(&[]),
            inactive_dependencies
        );
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
            Err(MaxwellThreeDLoweringError::UnsupportedPsOutputSampleMaskSemantics)
        ));
        assert_eq!(cache, cache_before);

        program_three_d(&mut channel, 0x1534, 0);
        program_three_d(&mut channel, 0x0300, 1);
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
                &lowering_capabilities(BackendFeatures::empty()),
                &cache,
            ),
            Err(MaxwellThreeDLoweringError::UnsupportedPsOutputSampleMaskSemantics)
        ));

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
                FrontendSubmissionId::new(13),
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
    fn primitive_circular_buffer_throttle_does_not_change_draw_or_clear_semantics() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        program_three_d(&mut channel, 0x121c, 0);
        program_three_d(&mut channel, 0x02d0, MAXWELL_THREE_D_PRIMITIVE_AREA_MAX);
        let resources =
            resolve_maxwell_three_d_resources(channel.three_d(), &resource_address_space())
                .unwrap();
        let cache = MaxwellThreeDLoweringCache::default();
        let cache_before = cache.clone();
        let source = channel
            .three_d()
            .vertex_input()
            .primitive()
            .circular_buffer_throttle()
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
                FrontendSubmissionId::new(10),
                Vec::new(),
                &lowering_capabilities(BackendFeatures::empty()),
                &cache,
            ),
            Err(MaxwellThreeDLoweringError::ShaderTranslationRequired)
        ));

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
                FrontendSubmissionId::new(11),
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
    fn patch_size_is_consumed_only_by_patch_draws_and_never_by_clears() {
        let mut missing_channel = channel();
        bind_three_d(&mut missing_channel);
        program_three_d(&mut missing_channel, 0x121c, 0);
        program_three_d(&mut missing_channel, 0x1618, 14);
        let missing = missing_channel.three_d().clone();

        let mut channel = channel();
        bind_three_d(&mut channel);
        program_three_d(&mut channel, 0x121c, 0);
        program_three_d(&mut channel, 0x2080, 0x20);
        program_three_d(&mut channel, 0x20c0, 0x30);
        program_three_d(&mut channel, 0x1618, 4);
        let dependencies_without_patch = channel.three_d().pipeline_dependencies(&[]);
        program_three_d(&mut channel, 0x0dcc, 3);
        assert_eq!(
            channel.three_d().pipeline_dependencies(&[]),
            dependencies_without_patch
        );
        let resources =
            resolve_maxwell_three_d_resources(channel.three_d(), &resource_address_space())
                .unwrap();
        let cache = MaxwellThreeDLoweringCache::default();
        let capabilities = lowering_capabilities(BackendFeatures::empty());
        let source = channel
            .three_d()
            .vertex_input()
            .primitive()
            .patch_size()
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
                FrontendSubmissionId::new(10),
                Vec::new(),
                &capabilities,
                &cache,
            ),
            Err(MaxwellThreeDLoweringError::ShaderTranslationRequired)
        ));

        program_three_d(&mut channel, 0x1618, 14);
        let patch_three_dependencies = channel.three_d().pipeline_dependencies(&[]);
        program_three_d(&mut channel, 0x0dcc, 4);
        assert_ne!(
            channel.three_d().pipeline_dependencies(&[]),
            patch_three_dependencies
        );
        let source = channel
            .three_d()
            .vertex_input()
            .primitive()
            .begin()
            .source()
            .unwrap();
        let cache_before = cache.clone();
        assert!(matches!(
            preflight_maxwell_three_d_operation(
                channel.three_d(),
                &resources,
                MaxwellThreeDOperationTrigger::DrawVertexArray {
                    source,
                    vertex_count: 4,
                },
                None,
                FrontendSubmissionId::new(11),
                Vec::new(),
                &capabilities,
                &cache,
            ),
            Err(MaxwellThreeDLoweringError::UnsupportedPatchSemantics(size))
                if size.control_points() == 4
        ));
        assert_eq!(cache, cache_before);

        program_three_d(&mut channel, 0x0dcc, 0);
        assert!(matches!(
            preflight_maxwell_three_d_operation(
                channel.three_d(),
                &resources,
                MaxwellThreeDOperationTrigger::DrawVertexArray {
                    source,
                    vertex_count: 4,
                },
                None,
                FrontendSubmissionId::new(12),
                Vec::new(),
                &capabilities,
                &cache,
            ),
            Err(MaxwellThreeDLoweringError::InvalidPatchSize(size))
                if size.control_points() == 0
        ));

        // A fresh channel proves that patch topology without SET_PATCH is
        // incomplete rather than silently assuming a control-point count.
        let missing_source = missing.vertex_input().primitive().begin().source().unwrap();
        assert!(matches!(
            preflight_maxwell_three_d_operation(
                &missing,
                &resources,
                MaxwellThreeDOperationTrigger::DrawVertexArray {
                    source: missing_source,
                    vertex_count: 3,
                },
                None,
                FrontendSubmissionId::new(13),
                Vec::new(),
                &capabilities,
                &cache,
            ),
            Err(MaxwellThreeDLoweringError::IncompleteDraw("SET_PATCH"))
        ));

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
                FrontendSubmissionId::new(14),
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
    fn point_rasterization_state_is_consumed_only_by_point_draws_and_never_by_clears() {
        let mut passthrough = channel();
        let mut channel = channel();
        bind_three_d(&mut channel);
        program_three_d(&mut channel, 0x121c, 0);
        program_three_d(&mut channel, 0x1618, 4);
        let triangle_dependencies = channel.three_d().pipeline_dependencies(&[]);
        program_three_d(&mut channel, 0x1604, 0x000d);
        program_three_d(&mut channel, 0x165c, 1);
        assert_eq!(
            channel.three_d().pipeline_dependencies(&[]),
            triangle_dependencies
        );

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

        program_three_d(&mut channel, 0x1618, 0);
        let generated_dependencies = channel.three_d().pipeline_dependencies(&[]);
        program_three_d(&mut channel, 0x1604, 0);
        let passthrough_dependencies = channel.three_d().pipeline_dependencies(&[]);
        assert_ne!(generated_dependencies, passthrough_dependencies);
        program_three_d(&mut channel, 0x1604, 0x000d);
        let source = channel
            .three_d()
            .vertex_input()
            .primitive()
            .begin()
            .source()
            .unwrap();
        let cache_before = cache.clone();
        assert!(matches!(
            preflight_maxwell_three_d_operation(
                channel.three_d(),
                &resources,
                MaxwellThreeDOperationTrigger::DrawVertexArray {
                    source,
                    vertex_count: 1,
                },
                None,
                FrontendSubmissionId::new(11),
                Vec::new(),
                &capabilities,
                &cache,
            ),
            Err(MaxwellThreeDLoweringError::UnsupportedPointSpriteCoordinatesSemantics(
                select
            )) if select.generated_texture_mask() == 1
                && select.r_mode() == MaxwellThreeDPointSpriteRMode::FromR
                && select.origin() == MaxwellThreeDPointSpriteOrigin::Top
        ));
        assert_eq!(cache, cache_before);

        program_three_d(&mut channel, 0x1604, 0);
        assert!(matches!(
            preflight_maxwell_three_d_operation(
                channel.three_d(),
                &resources,
                MaxwellThreeDOperationTrigger::DrawVertexArray {
                    source,
                    vertex_count: 1,
                },
                None,
                FrontendSubmissionId::new(12),
                Vec::new(),
                &capabilities,
                &cache,
            ),
            Err(MaxwellThreeDLoweringError::UnsupportedPointCenterSemantics(
                MaxwellThreeDPointCenterMode::Direct3D
            ))
        ));

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
                FrontendSubmissionId::new(13),
                Vec::new(),
                &capabilities,
                &cache,
            ),
            Err(MaxwellThreeDLoweringError::IncompleteClear(
                "horizontal rectangle"
            ))
        ));
        assert_eq!(cache, cache_before);

        bind_three_d(&mut passthrough);
        program_three_d(&mut passthrough, 0x121c, 0);
        program_three_d(&mut passthrough, 0x1618, 0);
        program_three_d(&mut passthrough, 0x1604, 0);
        let resources =
            resolve_maxwell_three_d_resources(passthrough.three_d(), &resource_address_space())
                .unwrap();
        let source = passthrough
            .three_d()
            .vertex_input()
            .primitive()
            .begin()
            .source()
            .unwrap();
        assert!(matches!(
            preflight_maxwell_three_d_operation(
                passthrough.three_d(),
                &resources,
                MaxwellThreeDOperationTrigger::DrawVertexArray {
                    source,
                    vertex_count: 1,
                },
                None,
                FrontendSubmissionId::new(14),
                Vec::new(),
                &capabilities,
                &cache,
            ),
            Err(MaxwellThreeDLoweringError::ShaderTranslationRequired)
        ));
    }

    #[test]
    fn edge_flag_is_consumed_only_by_non_fill_polygon_draws_and_never_by_clears() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        program_three_d(&mut channel, 0x121c, 0);
        program_three_d(&mut channel, 0x1618, 4);
        program_three_d(&mut channel, 0x0dac, 0x1b02);
        program_three_d(&mut channel, 0x0db0, 0x1b02);
        let fill_dependencies = channel.three_d().pipeline_dependencies(&[]);
        program_three_d(&mut channel, 0x15e4, 0);
        assert_eq!(
            channel.three_d().pipeline_dependencies(&[]),
            fill_dependencies
        );

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

        program_three_d(&mut channel, 0x0dac, 0x1b01);
        let disabled_dependencies = channel.three_d().pipeline_dependencies(&[]);
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
            Err(MaxwellThreeDLoweringError::UnsupportedEdgeFlagSemantics(
                MaxwellThreeDEdgeFlag::Disabled
            ))
        ));
        assert_eq!(cache, cache_before);

        program_three_d(&mut channel, 0x15e4, 1);
        assert_ne!(
            channel.three_d().pipeline_dependencies(&[]),
            disabled_dependencies
        );

        program_three_d(&mut channel, 0x1618, 0);
        program_three_d(&mut channel, 0x15e4, 0);
        let point_disabled_dependencies = channel.three_d().pipeline_dependencies(&[]);
        program_three_d(&mut channel, 0x15e4, 1);
        assert_eq!(
            channel.three_d().pipeline_dependencies(&[]),
            point_disabled_dependencies
        );

        program_three_d(&mut channel, 0x15e4, 0);
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
    fn color_reduction_only_blocks_draws_when_explicitly_enabled_and_never_blocks_clear() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        program_three_d(&mut channel, 0x121c, 0);
        program_three_d(&mut channel, 0x10cc, 0x0000_00ff);
        program_three_d(&mut channel, 0x10e0, 0x0000_00ff);
        program_three_d(&mut channel, 0x10e4, 0x0000_00ff);
        let resources =
            resolve_maxwell_three_d_resources(channel.three_d(), &resource_address_space())
                .unwrap();
        let cache = MaxwellThreeDLoweringCache::default();
        let threshold_source = channel
            .three_d()
            .color_reduction()
            .thresholds_unorm8()
            .source()
            .unwrap();
        assert!(matches!(
            preflight_maxwell_three_d_operation(
                channel.three_d(),
                &resources,
                MaxwellThreeDOperationTrigger::DrawVertexArray {
                    source: threshold_source,
                    vertex_count: 3,
                },
                None,
                FrontendSubmissionId::new(9),
                Vec::new(),
                &lowering_capabilities(BackendFeatures::empty()),
                &cache,
            ),
            Err(MaxwellThreeDLoweringError::ShaderTranslationRequired)
        ));

        for (argument, expected) in [
            (0, MaxwellThreeDColorReductionThresholdsEnable::Disabled),
            (1, MaxwellThreeDColorReductionThresholdsEnable::Enabled),
        ] {
            let decoded = packet(0x0d9c / 4, argument);
            let dispatch = dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0],
            )
            .unwrap();
            let source = dispatch.methods()[0].method().source();
            let register = channel.three_d().color_reduction().enable();
            assert_eq!(
                dispatch.methods()[0].metadata().method_name(),
                "SET_REDUCE_COLOR_THRESHOLDS_ENABLE"
            );
            assert_eq!(register.raw(), Some(argument));
            assert_eq!(register.value().copied(), Some(expected));
            assert_eq!(register.source(), Some(source));
            assert!(dispatch.operations().is_empty());
        }

        program_three_d(&mut channel, 0x0d9c, 0);
        let source = channel
            .three_d()
            .color_reduction()
            .enable()
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
                FrontendSubmissionId::new(10),
                Vec::new(),
                &lowering_capabilities(BackendFeatures::empty()),
                &cache,
            ),
            Err(MaxwellThreeDLoweringError::ShaderTranslationRequired)
        ));

        program_three_d(&mut channel, 0x0d9c, 1);
        let source = channel
            .three_d()
            .color_reduction()
            .enable()
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
                &lowering_capabilities(BackendFeatures::empty()),
                &cache,
            ),
            Err(MaxwellThreeDLoweringError::UnsupportedColorReductionSemantics)
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
    fn render_target_layer_is_typed_source_preserving_and_conditionally_dependent() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        let neutral_dependencies = channel.three_d().pipeline_dependencies(&[0]);

        for (argument, layer, control, affects_draw_layering) in [
            (0, 0, MaxwellThreeDRenderTargetLayerControl::Fixed, false),
            (
                0xffff,
                u16::MAX,
                MaxwellThreeDRenderTargetLayerControl::Fixed,
                true,
            ),
            (
                0x0001_0000,
                0,
                MaxwellThreeDRenderTargetLayerControl::GeometryShader,
                true,
            ),
            (
                0x0001_ffff,
                u16::MAX,
                MaxwellThreeDRenderTargetLayerControl::GeometryShader,
                true,
            ),
        ] {
            let decoded = packet(0x15cc / 4, argument);
            let dispatch = dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0],
            )
            .unwrap();
            let source = dispatch.methods()[0].method().source();
            let value = MaxwellThreeDRenderTargetLayer::new(layer, control);
            let register = channel.three_d().render_targets().render_target_layer();

            assert_eq!(
                dispatch.methods()[0].metadata().method_name(),
                "SET_RT_LAYER"
            );
            assert_eq!(
                dispatch.methods()[0].effect(),
                MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::RenderTarget(
                    MaxwellThreeDRenderTargetWrite::RenderTargetLayer { value, source }
                ))
            );
            assert!(dispatch.operations().is_empty());
            assert_eq!(value.layer(), layer);
            assert_eq!(value.control(), control);
            assert_eq!(value.raw(), argument);
            assert_eq!(value.affects_draw_layering(), affects_draw_layering);
            assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
            assert_eq!(register.raw(), Some(argument));
            assert_eq!(register.value(), Some(&value));
            assert_eq!(register.source(), Some(source));
            assert_eq!(
                channel.three_d().pipeline_dependencies(&[0]) == neutral_dependencies,
                !affects_draw_layering
            );
        }
    }

    #[test]
    fn render_target_layer_reserved_bits_and_packet_suffix_are_atomic() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        program_three_d(&mut channel, 0x15cc, 0);

        for argument in [0x0002_0000, 0x8000_0000, u32::MAX] {
            let frontend_before = channel.frontend();
            let two_d_before = channel.two_d().clone();
            let three_d_before = channel.three_d().clone();
            let decoded = packet(0x15cc / 4, argument);
            assert!(matches!(
                dispatch_maxwell_engine_packet(
                    &mut channel,
                    FrontendSubmissionId::new(3),
                    &decoded.packets()[0]
                ),
                Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                    source,
                    method_name: "SET_RT_LAYER",
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
        let decoded = incrementing_packet(0x15cc / 4, &[1, 0x10]);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                source,
                method_name: "SET_ANTI_ALIAS",
                ..
            }) if source.method() == GpuMethodId(0x15d0)
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
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
    fn clear_surface_control_is_typed_source_preserving_and_pipeline_neutral() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        let two_d_before = channel.two_d().clone();

        for combination in 0_u32..16 {
            let argument = (combination & 1)
                | ((combination & 2) << 3)
                | ((combination & 4) << 6)
                | ((combination & 8) << 9);
            let dependencies_before = channel.three_d().pipeline_dependencies(&[]);
            let decoded = packet(0x10f8 / 4, argument);
            let dispatch = dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0],
            )
            .unwrap();
            let method = dispatch.methods()[0];
            let source = method.method().source();
            let register = channel.three_d().render_targets().clear().surface_control();
            let value = register.value().copied().unwrap();

            assert_eq!(method.metadata().method_name(), "SET_CLEAR_SURFACE_CONTROL");
            assert_eq!(
                method.effect(),
                MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::RenderTarget(
                    MaxwellThreeDRenderTargetWrite::ClearSurfaceControl { value, source }
                ))
            );
            assert!(dispatch.operations().is_empty());
            assert_eq!(value.raw(), argument);
            assert_eq!(value.respect_stencil_mask(), combination & 1 != 0);
            assert_eq!(value.use_clear_rect(), combination & 2 != 0);
            assert_eq!(value.use_scissor_zero(), combination & 4 != 0);
            assert_eq!(value.use_viewport_clip_zero(), combination & 8 != 0);
            assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
            assert_eq!(register.raw(), Some(argument));
            assert_eq!(register.source(), Some(source));
            assert_eq!(
                channel.three_d().pipeline_dependencies(&[]),
                dependencies_before
            );
            assert_eq!(channel.two_d(), &two_d_before);
        }
    }

    #[test]
    fn clear_surface_control_reserved_bits_and_packet_suffix_are_rejected_atomically() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        program_three_d(&mut channel, 0x10f8, 0x1111);

        for argument in [2, 0x20, 0x200, 0x2000, 0x8000_0000, u32::MAX] {
            let frontend_before = channel.frontend();
            let two_d_before = channel.two_d().clone();
            let three_d_before = channel.three_d().clone();
            let decoded = packet(0x10f8 / 4, argument);
            assert!(matches!(
                dispatch_maxwell_engine_packet(
                    &mut channel,
                    FrontendSubmissionId::new(3),
                    &decoded.packets()[0]
                ),
                Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                    source,
                    method_name: "SET_CLEAR_SURFACE_CONTROL",
                    reason: "reserved control bits are set",
                }) if source.argument() == argument
            ));
            assert_eq!(channel.frontend(), frontend_before);
            assert_eq!(channel.two_d(), &two_d_before);
            assert_eq!(channel.three_d(), &three_d_before);
        }

        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = incrementing_packet(0x10f8 / 4, &[0, 0, 0]);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
                if source.method() == GpuMethodId(0x1100)
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }

    #[test]
    fn unsupported_clear_surface_modifiers_fail_only_when_consumed() {
        let preflight = |control, surface| {
            let mut channel = channel();
            bind_three_d(&mut channel);
            program_three_d(&mut channel, 0x10f8, control);
            let clear = packet(0x19d0 / 4, surface);
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
            match preflight_maxwell_three_d_operation(
                triggered.state(),
                &resources,
                triggered.trigger(),
                None,
                FrontendSubmissionId::new(10),
                Vec::new(),
                &lowering_capabilities(BackendFeatures::empty()),
                &MaxwellThreeDLoweringCache::default(),
            ) {
                Ok(_) => panic!("unsupported clear modifier unexpectedly lowered"),
                Err(error) => error,
            }
        };

        assert!(matches!(
            preflight(0x0100, 0x3c),
            MaxwellThreeDLoweringError::UnsupportedClearScissorSemantics
        ));
        assert!(matches!(
            preflight(0x1000, 0x3c),
            MaxwellThreeDLoweringError::UnsupportedClearViewportClipSemantics
        ));
        assert!(matches!(
            preflight(0x0001, 0x03),
            MaxwellThreeDLoweringError::UnsupportedClearStencilMaskSemantics
        ));
        assert!(!matches!(
            preflight(0x0001, 0x3c),
            MaxwellThreeDLoweringError::UnsupportedClearStencilMaskSemantics
        ));
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
    fn window_clip_type_is_typed_and_rejects_unknown_values_atomically() {
        let mut channel = channel();
        bind_three_d(&mut channel);

        for (argument, expected) in [
            (0, MaxwellThreeDWindowClipType::Inclusive),
            (1, MaxwellThreeDWindowClipType::Exclusive),
            (2, MaxwellThreeDWindowClipType::ClipAll),
        ] {
            let decoded = packet(0x1950 / 4, argument);
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
                .register(MaxwellThreeDFixedFunctionRegister::WindowClipType);

            assert_eq!(expected.raw(), argument);
            assert_eq!(register.raw(), Some(argument));
            assert_eq!(
                register.value(),
                Some(&MaxwellThreeDFixedFunctionValue::WindowClipType(expected))
            );
            assert_eq!(register.source(), Some(source));
        }

        for argument in [3, 0x8000_0000, u32::MAX] {
            let frontend_before = channel.frontend();
            let two_d_before = channel.two_d().clone();
            let three_d_before = channel.three_d().clone();
            let decoded = packet(0x1950 / 4, argument);
            assert!(matches!(
                dispatch_maxwell_engine_packet(
                    &mut channel,
                    FrontendSubmissionId::new(3),
                    &decoded.packets()[0]
                ),
                Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                    source,
                    method_name: "SET_WINDOW_CLIP_TYPE",
                    reason: "unknown window clip type",
                }) if source.argument() == argument
            ));
            assert_eq!(channel.frontend(), frontend_before);
            assert_eq!(channel.two_d(), &two_d_before);
            assert_eq!(channel.three_d(), &three_d_before);
        }
    }

    #[test]
    fn window_clip_packet_programs_all_eight_typed_source_preserving_pairs() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        program_three_d(&mut channel, 0x1950, 0);
        program_three_d(&mut channel, 0x194c, 0);
        assert_eq!(
            channel
                .three_d()
                .fixed_function()
                .register(MaxwellThreeDFixedFunctionRegister::WindowClipType)
                .value(),
            Some(&MaxwellThreeDFixedFunctionValue::WindowClipType(
                MaxwellThreeDWindowClipType::Inclusive
            ))
        );
        let dependencies_before = channel.three_d().pipeline_dependencies(&[]);
        let arguments = std::array::from_fn::<_, 16, _>(|word| {
            let region = word / 2;
            let minimum = (region * 10 + word % 2) as u32;
            let maximum = minimum + 100;
            (maximum << 16) | minimum
        });
        let decoded = incrementing_packet(0x0d00 / 4, &arguments);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();

        assert_eq!(dispatch.methods().len(), 16);
        assert!(dispatch.operations().is_empty());
        for (region_index, region) in channel
            .three_d()
            .fixed_function()
            .window_clip()
            .iter()
            .enumerate()
        {
            for (vertical, register, word) in [
                (false, region.horizontal(), region_index * 2),
                (true, region.vertical(), region_index * 2 + 1),
            ] {
                let method = dispatch.methods()[word];
                let source = method.method().source();
                let expected = MaxwellThreeDRectangle {
                    min: (region_index * 10 + usize::from(vertical)) as u16,
                    max: (region_index * 10 + usize::from(vertical) + 100) as u16,
                };
                let method_name = if vertical {
                    "SET_WINDOW_CLIP_VERTICAL"
                } else {
                    "SET_WINDOW_CLIP_HORIZONTAL"
                };

                assert_eq!(method.metadata().method_name(), method_name);
                assert_eq!(
                    method.effect(),
                    MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::FixedFunction(
                        MaxwellThreeDFixedFunctionWrite::WindowClipRectangle {
                            region: region_index as u8,
                            vertical,
                            value: expected,
                            source,
                        }
                    ))
                );
                assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
                assert_eq!(register.raw(), Some(arguments[word]));
                assert_eq!(register.value(), Some(&expected));
                assert_eq!(register.source(), Some(source));
            }
        }
        assert_eq!(
            channel.three_d().pipeline_dependencies(&[]),
            dependencies_before
        );
    }

    #[test]
    fn malformed_window_clip_rectangle_discards_the_whole_packet() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        let valid = incrementing_packet(0x0d00 / 4, &[0; 16]);
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &valid.packets()[0],
        )
        .unwrap();
        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let mut arguments = [0; 16];
        arguments[9] = (1 << 16) | 2;
        let malformed = incrementing_packet(0x0d00 / 4, &arguments);

        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &malformed.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                source,
                method_name: "SET_WINDOW_CLIP_VERTICAL",
                reason: "rectangle minimum exceeds maximum",
            }) if source.method() == GpuMethodId(0x0d24)
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }

    #[test]
    fn window_clip_dependencies_and_draw_error_follow_enable_while_clear_is_independent() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        program_three_d(&mut channel, 0x121c, 0);
        program_three_d(&mut channel, 0x1950, 0);
        program_three_d(&mut channel, 0x194c, 0);
        let dependencies_disabled = channel.three_d().pipeline_dependencies(&[]);
        program_three_d(&mut channel, 0x0d00, (100 << 16) | 10);
        program_three_d(&mut channel, 0x1950, 1);
        assert_eq!(
            channel.three_d().pipeline_dependencies(&[]),
            dependencies_disabled
        );

        program_three_d(&mut channel, 0x194c, 1);
        let dependencies_enabled = channel.three_d().pipeline_dependencies(&[]);
        assert_ne!(dependencies_enabled, dependencies_disabled);
        program_three_d(&mut channel, 0x0d04, (200 << 16) | 20);
        assert_ne!(
            channel.three_d().pipeline_dependencies(&[]),
            dependencies_enabled
        );

        let resources =
            resolve_maxwell_three_d_resources(channel.three_d(), &resource_address_space())
                .unwrap();
        let capabilities = lowering_capabilities(BackendFeatures::empty());
        let cache = MaxwellThreeDLoweringCache::default();
        let cache_before = cache.clone();
        let source = channel
            .three_d()
            .fixed_function()
            .register(MaxwellThreeDFixedFunctionRegister::WindowClipEnable)
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
                FrontendSubmissionId::new(10),
                Vec::new(),
                &capabilities,
                &cache,
            ),
            Err(MaxwellThreeDLoweringError::UnsupportedWindowClipSemantics)
        ));

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
    fn clip_id_test_enable_is_typed_source_preserving_and_atomic() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        let two_d_before = channel.two_d().clone();
        let window_clip_before = channel.three_d().fixed_function().window_clip().to_owned();
        assert_eq!(
            channel
                .three_d()
                .fixed_function()
                .register(MaxwellThreeDFixedFunctionRegister::ClipIdTestEnable)
                .origin(),
            MaxwellThreeDRegisterOrigin::Unset
        );

        for (argument, expected) in [
            (0, MaxwellThreeDClipIdTestEnable::Disabled),
            (1, MaxwellThreeDClipIdTestEnable::Enabled),
        ] {
            let decoded = packet(0x197c / 4, argument);
            let dispatch = dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0],
            )
            .unwrap();
            let method = dispatch.methods()[0];
            let source = method.method().source();
            let register = channel
                .three_d()
                .fixed_function()
                .register(MaxwellThreeDFixedFunctionRegister::ClipIdTestEnable);
            let value = MaxwellThreeDFixedFunctionValue::ClipIdTestEnable(expected);

            assert_eq!(method.metadata().method_name(), "SET_CLIP_ID_TEST");
            assert_eq!(
                method.effect(),
                MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::FixedFunction(
                    MaxwellThreeDFixedFunctionWrite::Register {
                        register: MaxwellThreeDFixedFunctionRegister::ClipIdTestEnable,
                        value,
                        source,
                    }
                ))
            );
            assert!(dispatch.operations().is_empty());
            assert_eq!(expected.raw(), argument);
            assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
            assert_eq!(register.raw(), Some(argument));
            assert_eq!(register.value(), Some(&value));
            assert_eq!(register.source(), Some(source));
            assert_eq!(
                channel.three_d().fixed_function().window_clip(),
                &window_clip_before
            );
            assert_eq!(channel.two_d(), &two_d_before);
        }

        for argument in [2, 3, 0x8000_0000, u32::MAX] {
            let frontend_before = channel.frontend();
            let two_d_before = channel.two_d().clone();
            let three_d_before = channel.three_d().clone();
            let decoded = packet(0x197c / 4, argument);
            assert!(matches!(
                dispatch_maxwell_engine_packet(
                    &mut channel,
                    FrontendSubmissionId::new(3),
                    &decoded.packets()[0]
                ),
                Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                    source,
                    method_name: "SET_CLIP_ID_TEST",
                    reason: "expected boolean 0 or 1",
                }) if source.argument() == argument
            ));
            assert_eq!(channel.frontend(), frontend_before);
            assert_eq!(channel.two_d(), &two_d_before);
            assert_eq!(channel.three_d(), &three_d_before);
        }

        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = incrementing_packet(0x197c / 4, &[0, 0]);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
                if source.method() == GpuMethodId(0x1980)
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }

    #[test]
    fn clip_id_test_only_blocks_draw_when_enabled_and_never_blocks_clear() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        program_three_d(&mut channel, 0x121c, 0);
        program_three_d(&mut channel, 0x197c, 0);
        let dependencies_disabled = channel.three_d().pipeline_dependencies(&[]);
        let resources =
            resolve_maxwell_three_d_resources(channel.three_d(), &resource_address_space())
                .unwrap();
        let capabilities = lowering_capabilities(BackendFeatures::empty());
        let cache = MaxwellThreeDLoweringCache::default();
        let cache_before = cache.clone();
        let disabled_source = channel
            .three_d()
            .fixed_function()
            .register(MaxwellThreeDFixedFunctionRegister::ClipIdTestEnable)
            .source()
            .unwrap();
        assert!(matches!(
            preflight_maxwell_three_d_operation(
                channel.three_d(),
                &resources,
                MaxwellThreeDOperationTrigger::DrawVertexArray {
                    source: disabled_source,
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

        program_three_d(&mut channel, 0x197c, 1);
        assert_ne!(
            channel.three_d().pipeline_dependencies(&[]),
            dependencies_disabled
        );
        let enabled_source = channel
            .three_d()
            .fixed_function()
            .register(MaxwellThreeDFixedFunctionRegister::ClipIdTestEnable)
            .source()
            .unwrap();
        assert!(matches!(
            preflight_maxwell_three_d_operation(
                channel.three_d(),
                &resources,
                MaxwellThreeDOperationTrigger::DrawVertexArray {
                    source: enabled_source,
                    vertex_count: 3,
                },
                None,
                FrontendSubmissionId::new(11),
                Vec::new(),
                &capabilities,
                &cache,
            ),
            Err(MaxwellThreeDLoweringError::UnsupportedClipIdTestSemantics)
        ));

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
    fn viewport_scale_offset_enable_is_typed_source_preserving_and_atomic() {
        let mut channel = channel();
        bind_three_d(&mut channel);

        for (argument, expected) in [
            (0, MaxwellThreeDViewportScaleOffsetEnable::Disabled),
            (1, MaxwellThreeDViewportScaleOffsetEnable::Enabled),
        ] {
            let decoded = packet(0x192c / 4, argument);
            let dispatch = dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0],
            )
            .unwrap();
            let method = dispatch.methods()[0];
            let source = method.method().source();
            let register = channel
                .three_d()
                .fixed_function()
                .register(MaxwellThreeDFixedFunctionRegister::ViewportScaleOffsetEnable);
            let value = MaxwellThreeDFixedFunctionValue::ViewportScaleOffsetEnable(expected);

            assert_eq!(method.metadata().method_name(), "SET_VIEWPORT_SCALE_OFFSET");
            assert_eq!(
                method.effect(),
                MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::FixedFunction(
                    MaxwellThreeDFixedFunctionWrite::Register {
                        register: MaxwellThreeDFixedFunctionRegister::ViewportScaleOffsetEnable,
                        value,
                        source,
                    }
                ))
            );
            assert!(dispatch.operations().is_empty());
            assert_eq!(expected.raw(), argument);
            assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
            assert_eq!(register.raw(), Some(argument));
            assert_eq!(register.value(), Some(&value));
            assert_eq!(register.source(), Some(source));
        }

        for argument in [2, 3, 0x8000_0000, u32::MAX] {
            let frontend_before = channel.frontend();
            let two_d_before = channel.two_d().clone();
            let three_d_before = channel.three_d().clone();
            let decoded = packet(0x192c / 4, argument);
            assert!(matches!(
                dispatch_maxwell_engine_packet(
                    &mut channel,
                    FrontendSubmissionId::new(3),
                    &decoded.packets()[0]
                ),
                Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                    source,
                    method_name: "SET_VIEWPORT_SCALE_OFFSET",
                    reason: "expected boolean 0 or 1",
                }) if source.argument() == argument
            ));
            assert_eq!(channel.frontend(), frontend_before);
            assert_eq!(channel.two_d(), &two_d_before);
            assert_eq!(channel.three_d(), &three_d_before);
        }

        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = incrementing_packet(0x192c / 4, &[0, 0]);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
                if source.method() == GpuMethodId(0x1930)
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }

    #[test]
    fn viewport_scale_offset_dependencies_and_draw_error_follow_enable_only() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        program_three_d(&mut channel, 0x121c, 0);
        program_three_d(&mut channel, 0x192c, 0);
        let dependencies_disabled = channel.three_d().pipeline_dependencies(&[]);

        program_three_d(&mut channel, 0x0a00, 1.0_f32.to_bits());
        program_three_d(&mut channel, 0x0a0c, 2.0_f32.to_bits());
        assert_eq!(
            channel.three_d().pipeline_dependencies(&[]),
            dependencies_disabled
        );

        let resources =
            resolve_maxwell_three_d_resources(channel.three_d(), &resource_address_space())
                .unwrap();
        let capabilities = lowering_capabilities(BackendFeatures::empty());
        let cache = MaxwellThreeDLoweringCache::default();
        let disabled_source = channel
            .three_d()
            .fixed_function()
            .register(MaxwellThreeDFixedFunctionRegister::ViewportScaleOffsetEnable)
            .source()
            .unwrap();
        assert!(matches!(
            preflight_maxwell_three_d_operation(
                channel.three_d(),
                &resources,
                MaxwellThreeDOperationTrigger::DrawVertexArray {
                    source: disabled_source,
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

        program_three_d(&mut channel, 0x192c, 1);
        let dependencies_enabled = channel.three_d().pipeline_dependencies(&[]);
        assert_ne!(dependencies_enabled, dependencies_disabled);
        program_three_d(&mut channel, 0x0a00, 3.0_f32.to_bits());
        assert_ne!(
            channel.three_d().pipeline_dependencies(&[]),
            dependencies_enabled
        );

        let enabled_source = channel
            .three_d()
            .fixed_function()
            .register(MaxwellThreeDFixedFunctionRegister::ViewportScaleOffsetEnable)
            .source()
            .unwrap();
        assert!(matches!(
            preflight_maxwell_three_d_operation(
                channel.three_d(),
                &resources,
                MaxwellThreeDOperationTrigger::DrawVertexArray {
                    source: enabled_source,
                    vertex_count: 3,
                },
                None,
                FrontendSubmissionId::new(11),
                Vec::new(),
                &capabilities,
                &cache,
            ),
            Err(MaxwellThreeDLoweringError::UnsupportedViewportScaleOffsetSemantics)
        ));

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
                &capabilities,
                &cache,
            ),
            Err(MaxwellThreeDLoweringError::IncompleteClear(
                "horizontal rectangle"
            ))
        ));
    }

    #[test]
    fn mme_ram_loads_capture_typed_programs_with_sources_and_auto_advance() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        let dependencies_before = channel.three_d().pipeline_dependencies(&[]);

        let start_packet = incrementing_packet(0x011c / 4, &[5, 7]);
        let start_dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &start_packet.packets()[0],
        )
        .unwrap();
        let start_pointer_source = start_dispatch.methods()[0].method().source();
        let start_source = start_dispatch.methods()[1].method().source();
        assert_eq!(
            start_dispatch.methods()[0].metadata().method_name(),
            "LOAD_MME_START_ADDRESS_RAM_POINTER"
        );
        assert_eq!(
            start_dispatch.methods()[1].metadata().method_name(),
            "LOAD_MME_START_ADDRESS_RAM"
        );
        assert_eq!(
            start_dispatch.methods()[0].effect(),
            MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::Mme(
                MaxwellThreeDMmeStateWrite::StartAddressPointer {
                    value: MaxwellThreeDMmeRamAddress::new(5),
                    source: start_pointer_source,
                }
            ))
        );
        assert_eq!(
            start_dispatch.methods()[1].effect(),
            MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::Mme(
                MaxwellThreeDMmeStateWrite::StartAddress {
                    index: MaxwellThreeDMmeRamAddress::new(5),
                    address: MaxwellThreeDMmeRamAddress::new(7),
                    source: start_source,
                }
            ))
        );

        let instruction_words = [0x0000_0301, 0x0000_0211, 0x0588_0021];
        let instruction_packet = increment_once_packet(
            0x0114 / 4,
            &[
                7,
                instruction_words[0],
                instruction_words[1],
                instruction_words[2],
            ],
        );
        let instruction_dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &instruction_packet.packets()[0],
        )
        .unwrap();
        let mme = channel.three_d().mme();

        assert!(start_dispatch.operations().is_empty());
        assert!(instruction_dispatch.operations().is_empty());
        assert_eq!(mme.instruction_pointer().raw(), Some(7));
        assert_eq!(
            mme.instruction_pointer().value(),
            Some(&MaxwellThreeDMmeRamAddress::new(7))
        );
        assert_eq!(
            mme.next_instruction_address(),
            Some(MaxwellThreeDMmeRamAddress::new(10))
        );
        assert_eq!(mme.instruction_count(), 3);
        for (word, expected) in instruction_words.into_iter().enumerate() {
            let address = MaxwellThreeDMmeRamAddress::new(7 + word as u32);
            let register = mme.instruction(address).unwrap();
            let source = instruction_dispatch.methods()[word + 1].method().source();
            assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
            assert_eq!(register.raw(), Some(expected));
            assert_eq!(
                register.value(),
                Some(&MaxwellThreeDMmeInstruction::new(expected))
            );
            assert_eq!(register.source(), Some(source));
        }
        assert_eq!(mme.start_address_pointer().raw(), Some(5));
        assert_eq!(
            mme.next_start_address_index(),
            Some(MaxwellThreeDMmeRamAddress::new(6))
        );
        let start = mme
            .start_address(MaxwellThreeDMmeRamAddress::new(5))
            .unwrap();
        assert_eq!(start.raw(), Some(7));
        assert_eq!(start.value(), Some(&MaxwellThreeDMmeRamAddress::new(7)));
        assert_eq!(start.source(), Some(start_source));
        assert_eq!(
            channel.three_d().pipeline_dependencies(&[]),
            dependencies_before
        );
    }

    #[test]
    fn mme_ram_load_failures_discard_the_whole_packet() {
        let mut channel = channel();
        bind_three_d(&mut channel);

        for (method, ram) in [
            (0x0118, MaxwellThreeDMmeRam::Instruction),
            (0x0120, MaxwellThreeDMmeRam::StartAddress),
        ] {
            let before = channel.three_d().clone();
            let decoded = packet(method / 4, 0);
            assert!(matches!(
                dispatch_maxwell_engine_packet(
                    &mut channel,
                    FrontendSubmissionId::new(3),
                    &decoded.packets()[0]
                ),
                Err(MaxwellEngineDispatchError::MmeRamLoad {
                    ram: actual,
                    error: MaxwellThreeDMmeLoadError::PointerUnset,
                    ..
                }) if actual == ram
            ));
            assert_eq!(channel.three_d(), &before);
        }

        for (method, ram) in [
            (0x0114, MaxwellThreeDMmeRam::Instruction),
            (0x011c, MaxwellThreeDMmeRam::StartAddress),
        ] {
            let before = channel.three_d().clone();
            let decoded = incrementing_packet(method / 4, &[u32::MAX, 0]);
            assert!(matches!(
                dispatch_maxwell_engine_packet(
                    &mut channel,
                    FrontendSubmissionId::new(3),
                    &decoded.packets()[0]
                ),
                Err(MaxwellEngineDispatchError::MmeRamLoad {
                    ram: actual,
                    error: MaxwellThreeDMmeLoadError::PointerOverflow,
                    ..
                }) if actual == ram
            ));
            assert_eq!(channel.three_d(), &before);
        }

        let before = channel.three_d().clone();
        let mut arguments = Vec::with_capacity(MAXWELL_THREE_D_MME_CAPTURED_INSTRUCTION_WORDS + 2);
        arguments.push(0);
        arguments.resize(
            MAXWELL_THREE_D_MME_CAPTURED_INSTRUCTION_WORDS + 2,
            0x0000_0201,
        );
        let decoded = increment_once_packet(0x0114 / 4, &arguments);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::MmeRamLoad {
                ram: MaxwellThreeDMmeRam::Instruction,
                error: MaxwellThreeDMmeLoadError::StorageLimitExceeded {
                    limit: MAXWELL_THREE_D_MME_CAPTURED_INSTRUCTION_WORDS,
                },
                ..
            })
        ));
        assert_eq!(channel.three_d(), &before);

        let mut arguments = Vec::with_capacity(MAXWELL_THREE_D_MME_CAPTURED_START_ADDRESSES + 2);
        arguments.push(0);
        arguments.resize(MAXWELL_THREE_D_MME_CAPTURED_START_ADDRESSES + 2, 0);
        let decoded = increment_once_packet(0x011c / 4, &arguments);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::MmeRamLoad {
                ram: MaxwellThreeDMmeRam::StartAddress,
                error: MaxwellThreeDMmeLoadError::StorageLimitExceeded {
                    limit: MAXWELL_THREE_D_MME_CAPTURED_START_ADDRESSES,
                },
                ..
            })
        ));
        assert_eq!(channel.three_d(), &before);
    }

    #[test]
    fn mme_macro_executes_captured_code_and_emits_validated_methods() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        let macro_index = 5;
        let point_size_method_dword = 0x1518 / 4;
        let set_method = 1 | (2 << 4) | (point_size_method_dword << 14);
        let send_parameter_and_exit = (4 << 4) | (1 << 7) | (1 << 11);
        load_mme_program(
            &mut channel,
            macro_index,
            &[set_method, send_parameter_and_exit, 0x11],
        );

        let argument = 2.5_f32.to_bits();
        let call = packet((0x3800 + u32::from(macro_index) * 8) / 4, argument);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &call.packets()[0],
        )
        .unwrap();

        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "CALL_MME_MACRO"
        );
        assert_eq!(
            dispatch.methods()[0].effect(),
            MaxwellEngineMethodEffect::MmeMacroCall {
                macro_index,
                parameter_count: 1,
                report: MaxwellThreeDMmeExecutionReport {
                    instructions: 3,
                    emitted_methods: 1,
                },
            }
        );
        let point_size = channel.three_d().raster().point_size();
        assert_eq!(point_size.raw(), Some(argument));
        let source = point_size.source().unwrap();
        assert_eq!(
            source.location(),
            dispatch.methods()[0].method().source().location()
        );
        assert_eq!(source.method(), GpuMethodId(0x1518));
        assert_eq!(source.argument(), argument);
        assert_eq!(
            channel
                .three_d()
                .raw_register(GpuMethodId(0x1518))
                .and_then(MaxwellThreeDRegister::raw),
            Some(argument)
        );
    }

    #[test]
    fn mme_call_data_supplies_additional_parameters() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        let macro_index = 2;
        let fetch_second_parameter = 1 | (2 << 8);
        let point_size_method_dword = 0x1518 / 4;
        let set_method = 1 | (2 << 4) | (point_size_method_dword << 14);
        let send_second_parameter_and_exit = (4 << 4) | (1 << 7) | (2 << 11);
        load_mme_program(
            &mut channel,
            macro_index,
            &[
                fetch_second_parameter,
                set_method,
                send_second_parameter_and_exit,
                0x11,
            ],
        );

        let argument = 4.0_f32.to_bits();
        let call = incrementing_packet(
            (0x3800 + u32::from(macro_index) * 8) / 4,
            &[0xdead_beef, argument],
        );
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &call.packets()[0],
        )
        .unwrap();

        assert_eq!(dispatch.methods().len(), 2);
        assert_eq!(
            dispatch.methods()[1].metadata().method_name(),
            "CALL_MME_DATA"
        );
        assert_eq!(
            dispatch.methods()[1].effect(),
            MaxwellEngineMethodEffect::MmeMacroData { macro_index }
        );
        assert_eq!(
            channel.three_d().raster().point_size().raw(),
            Some(argument)
        );
    }

    #[test]
    fn mme_reads_polygon_mode_reset_bits_until_guest_programming_overrides_them() {
        let mut channel = channel();
        bind_three_d(&mut channel);

        for (method, register) in [
            (0x0dac, MaxwellThreeDFixedFunctionRegister::FrontPolygonMode),
            (0x0db0, MaxwellThreeDFixedFunctionRegister::BackPolygonMode),
        ] {
            let raw = channel.three_d().raw_register(GpuMethodId(method)).unwrap();
            assert_eq!(raw.origin(), MaxwellThreeDRegisterOrigin::VerifiedReset);
            assert_eq!(raw.raw(), Some(0));
            assert_eq!(raw.value(), Some(&0));
            assert_eq!(raw.source(), None);

            let typed = channel.three_d().fixed_function().register(register);
            assert_eq!(typed.origin(), MaxwellThreeDRegisterOrigin::VerifiedReset);
            assert_eq!(typed.raw(), Some(0));
            assert_eq!(typed.value(), None);
            assert_eq!(typed.source(), None);
        }

        let macro_index = 3;
        let read_front = 5 | (1 << 4) | (2 << 8) | (0x036b << 14);
        let read_back_and_exit = 5 | (1 << 4) | (1 << 7) | (3 << 8) | (0x036c << 14);
        load_mme_program(
            &mut channel,
            macro_index,
            &[read_front, read_back_and_exit, 0x11],
        );
        let before_call = channel.three_d().clone();
        let call = packet((0x3800 + u32::from(macro_index) * 8) / 4, 0);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &call.packets()[0],
        )
        .unwrap();
        assert_eq!(
            dispatch.methods()[0].effect(),
            MaxwellEngineMethodEffect::MmeMacroCall {
                macro_index,
                parameter_count: 1,
                report: MaxwellThreeDMmeExecutionReport {
                    instructions: 3,
                    emitted_methods: 0,
                },
            }
        );
        assert_eq!(channel.three_d(), &before_call);

        program_three_d(&mut channel, 0x0dac, 0x1b02);
        let raw = channel.three_d().raw_register(GpuMethodId(0x0dac)).unwrap();
        assert_eq!(raw.origin(), MaxwellThreeDRegisterOrigin::Programmed);
        assert_eq!(raw.raw(), Some(0x1b02));
        assert!(raw.source().is_some());
        let typed = channel
            .three_d()
            .fixed_function()
            .register(MaxwellThreeDFixedFunctionRegister::FrontPolygonMode);
        assert_eq!(typed.origin(), MaxwellThreeDRegisterOrigin::Programmed);
        assert_eq!(
            typed.value(),
            Some(&MaxwellThreeDFixedFunctionValue::PolygonMode(
                MaxwellThreeDPolygonMode::Fill,
            ))
        );
        assert_eq!(
            channel
                .three_d()
                .raw_register(GpuMethodId(0x0db0))
                .unwrap()
                .origin(),
            MaxwellThreeDRegisterOrigin::VerifiedReset
        );
    }

    #[test]
    fn mme_reads_all_pipeline_shader_reset_headers_and_writes_override_one_slot() {
        let mut channel = channel();
        bind_three_d(&mut channel);

        for pipeline in 0..MAXWELL_PIPELINE_SHADER_COUNT {
            let method = 0x2000 + pipeline as u32 * 0x40;
            let raw = channel.three_d().raw_register(GpuMethodId(method)).unwrap();
            assert_eq!(raw.origin(), MaxwellThreeDRegisterOrigin::VerifiedReset);
            assert_eq!(raw.raw(), Some(0));
            assert_eq!(raw.value(), Some(&0));
            assert_eq!(raw.source(), None);

            let binding = &channel.three_d().shader_bindings().pipeline()[pipeline];
            assert_eq!(
                binding.enabled().origin(),
                MaxwellThreeDRegisterOrigin::VerifiedReset
            );
            assert_eq!(binding.enabled().raw(), Some(0));
            assert_eq!(binding.enabled().value(), Some(&false));
            assert_eq!(binding.enabled().source(), None);
            assert_eq!(
                binding.stage().origin(),
                MaxwellThreeDRegisterOrigin::VerifiedReset
            );
            assert_eq!(binding.stage().raw(), Some(0));
            assert_eq!(
                binding.stage().value(),
                Some(&MaxwellThreeDShaderStage::VertexCullBeforeFetch)
            );
            assert_eq!(binding.stage().source(), None);
            assert_eq!(binding.group().origin(), MaxwellThreeDRegisterOrigin::Unset);
        }

        let macro_index = 4;
        let read_pipeline_three = 5 | (1 << 4) | (2 << 8) | (0x0830 << 14);
        let read_pipeline_four_and_exit = 5 | (1 << 4) | (1 << 7) | (3 << 8) | (0x0840 << 14);
        load_mme_program(
            &mut channel,
            macro_index,
            &[read_pipeline_three, read_pipeline_four_and_exit, 0x11],
        );
        let before_call = channel.three_d().clone();
        let call = packet((0x3800 + u32::from(macro_index) * 8) / 4, 0);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &call.packets()[0],
        )
        .unwrap();
        assert_eq!(
            dispatch.methods()[0].effect(),
            MaxwellEngineMethodEffect::MmeMacroCall {
                macro_index,
                parameter_count: 1,
                report: MaxwellThreeDMmeExecutionReport {
                    instructions: 3,
                    emitted_methods: 0,
                },
            }
        );
        assert_eq!(channel.three_d(), &before_call);

        let before_invalid = channel.three_d().clone();
        let invalid = packet(0x20c0 / 4, 2);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &invalid.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                source,
                method_name: "SET_PIPELINE_SHADER",
                ..
            }) if source.method() == GpuMethodId(0x20c0)
        ));
        assert_eq!(channel.three_d(), &before_invalid);

        program_three_d(&mut channel, 0x20c0, 0x41);
        let raw = channel.three_d().raw_register(GpuMethodId(0x20c0)).unwrap();
        assert_eq!(raw.origin(), MaxwellThreeDRegisterOrigin::Programmed);
        assert_eq!(raw.raw(), Some(0x41));
        assert!(raw.source().is_some());
        let binding = &channel.three_d().shader_bindings().pipeline()[3];
        assert_eq!(
            binding.enabled().origin(),
            MaxwellThreeDRegisterOrigin::Programmed
        );
        assert_eq!(binding.enabled().value(), Some(&true));
        assert_eq!(
            binding.stage().origin(),
            MaxwellThreeDRegisterOrigin::Programmed
        );
        assert_eq!(
            binding.stage().value(),
            Some(&MaxwellThreeDShaderStage::Geometry)
        );
        assert_eq!(
            channel
                .three_d()
                .raw_register(GpuMethodId(0x2100))
                .unwrap()
                .origin(),
            MaxwellThreeDRegisterOrigin::VerifiedReset
        );
    }

    #[test]
    fn mme_emitted_draw_keeps_the_exact_candidate_snapshot() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        let macro_index = 6;
        let draw_method_dword = 0x0d78 / 4;
        let set_method = 1 | (2 << 4) | (draw_method_dword << 14);
        let send_parameter_and_exit = (4 << 4) | (1 << 7) | (1 << 11);
        load_mme_program(
            &mut channel,
            macro_index,
            &[set_method, send_parameter_and_exit, 0x11],
        );

        let call = packet((0x3800 + u32::from(macro_index) * 8) / 4, 3);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &call.packets()[0],
        )
        .unwrap();
        assert_eq!(dispatch.operations().len(), 1);
        let operation = &dispatch.operations()[0];
        assert_eq!(operation.state(), channel.three_d());
        assert!(matches!(
            operation.trigger(),
            MaxwellThreeDOperationTrigger::DrawVertexArray {
                source,
                vertex_count: 3,
            } if source.method() == GpuMethodId(0x0d78)
                && source.location() == dispatch.methods()[0].method().source().location()
        ));
    }

    #[test]
    fn mme_execution_errors_and_partial_emissions_are_atomic() {
        let mut channel = channel();
        bind_three_d(&mut channel);

        let data = packet(0x3804 / 4, 0);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &data.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::MmeExecution {
                error: MaxwellThreeDMmeExecutionError::DataWithoutCall,
                ..
            })
        ));

        let missing = packet(0x3800 / 4, 0);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &missing.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::MmeExecution {
                error: MaxwellThreeDMmeExecutionError::MissingStartAddress { macro_index: 0 },
                ..
            })
        ));

        let macro_index = 1;
        let point_size_method_dword = 0x1518 / 4;
        let set_point_size = 1 | (2 << 4) | (point_size_method_dword << 14);
        let send_parameter = (4 << 4) | (1 << 11);
        let set_recursive_method = 1 | (2 << 4) | (0x0e00 << 14);
        let send_recursive_and_exit = (4 << 4) | (1 << 7) | (1 << 11);
        load_mme_program(
            &mut channel,
            macro_index,
            &[
                set_point_size,
                send_parameter,
                set_recursive_method,
                send_recursive_and_exit,
                0x11,
            ],
        );
        let before = channel.three_d().clone();
        let call = packet((0x3800 + u32::from(macro_index) * 8) / 4, 1.0_f32.to_bits());
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &call.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::MmeExecution {
                error: MaxwellThreeDMmeExecutionError::RecursiveMacroCall {
                    method_dword: 0x0e00,
                },
                ..
            })
        ));
        assert_eq!(channel.three_d(), &before);
    }

    #[test]
    fn mme_register_reads_and_execution_limit_fail_typed_and_atomically() {
        let mut channel = channel();
        bind_three_d(&mut channel);

        let read_macro = 3;
        let read_unset_register = 5 | (1 << 4) | (2 << 8) | (0x036d << 14);
        load_mme_program(&mut channel, read_macro, &[read_unset_register]);
        let before = channel.three_d().clone();
        let call = packet((0x3800 + u32::from(read_macro) * 8) / 4, 0);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &call.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::MmeExecution {
                error: MaxwellThreeDMmeExecutionError::RegisterReadUnavailable {
                    method_dword: 0x036d,
                },
                ..
            })
        ));
        assert_eq!(channel.three_d(), &before);

        let loop_macro = 4;
        let branch_to_self_without_delay = 7 | (1 << 5);
        load_mme_program(&mut channel, loop_macro, &[branch_to_self_without_delay]);
        let before = channel.three_d().clone();
        let call = packet((0x3800 + u32::from(loop_macro) * 8) / 4, 0);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &call.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::MmeExecution {
                error: MaxwellThreeDMmeExecutionError::InstructionLimitExceeded {
                    limit: MAXWELL_THREE_D_MME_EXECUTION_INSTRUCTION_LIMIT,
                },
                ..
            })
        ));
        assert_eq!(channel.three_d(), &before);
    }

    #[test]
    fn vertex_assembly_controls_are_typed_source_preserving_and_isolated() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        let input_before = channel.three_d().vertex_input().clone();
        let two_d_before = channel.two_d().clone();

        let decoded = packet(0x1610 / 4, 0x0e);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
        let defaults_source = dispatch.methods()[0].method().source();
        let defaults = channel
            .three_d()
            .vertex_input()
            .assembly()
            .attribute_defaults();
        let value = *defaults.value().unwrap();

        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_ATTRIBUTE_DEFAULT"
        );
        assert_eq!(
            dispatch.methods()[0].effect(),
            MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::VertexInput(
                MaxwellThreeDVertexInputWrite::AttributeDefaults {
                    value,
                    source: defaults_source,
                }
            ))
        );
        assert_eq!(defaults.origin(), MaxwellThreeDRegisterOrigin::Programmed);
        assert_eq!(defaults.raw(), Some(0x0e));
        assert_eq!(defaults.source(), Some(defaults_source));
        assert_eq!(
            value.color_front_diffuse(),
            MaxwellThreeDAttributeDefaultVector::Vector0001
        );
        assert_eq!(
            value.color_front_specular(),
            MaxwellThreeDAttributeDefaultVector::Vector0001
        );
        assert_eq!(
            value.generic_vector(),
            MaxwellThreeDAttributeDefaultVector::Vector0001
        );
        assert_eq!(
            value.fixed_function_texture(),
            MaxwellThreeDAttributeDefaultVector::Vector0001
        );
        assert_eq!(
            value.dx9_color0(),
            MaxwellThreeDAttributeDefaultVector::Vector0001
        );
        assert_eq!(
            value.dx9_color1_to_color15(),
            MaxwellThreeDAttributeDefaultVector::Vector0000
        );
        assert_eq!(
            channel
                .three_d()
                .vertex_input()
                .assembly()
                .vertex_id_uses_array_start(),
            input_before.assembly().vertex_id_uses_array_start()
        );

        let decoded = packet(0x164c / 4, 0x1000);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &decoded.packets()[0],
        )
        .unwrap();
        let vertex_id_source = dispatch.methods()[0].method().source();
        let input = channel.three_d().vertex_input();
        let vertex_id = input.assembly().vertex_id_uses_array_start();

        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_DA_OUTPUT"
        );
        assert_eq!(
            dispatch.methods()[0].effect(),
            MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::VertexInput(
                MaxwellThreeDVertexInputWrite::VertexIdUsesArrayStart {
                    value: MaxwellThreeDVertexIdUsesArrayStart::Enabled,
                    source: vertex_id_source,
                }
            ))
        );
        assert_eq!(vertex_id.origin(), MaxwellThreeDRegisterOrigin::Programmed);
        assert_eq!(vertex_id.raw(), Some(0x1000));
        assert_eq!(
            vertex_id.value(),
            Some(&MaxwellThreeDVertexIdUsesArrayStart::Enabled)
        );
        assert_eq!(vertex_id.source(), Some(vertex_id_source));
        assert_eq!(input.streams(), input_before.streams());
        assert_eq!(input.attributes(), input_before.attributes());
        assert_eq!(input.index(), input_before.index());
        assert_eq!(input.primitive(), input_before.primitive());
        assert_eq!(channel.two_d(), &two_d_before);
    }

    #[test]
    fn vertex_assembly_controls_are_draw_dependencies() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        let unset = channel.three_d().pipeline_dependencies(&[]);

        program_three_d(&mut channel, 0x1610, 0x0e);
        let defaults = channel.three_d().pipeline_dependencies(&[]);
        assert_ne!(defaults, unset);

        program_three_d(&mut channel, 0x164c, 0x1000);
        let vertex_id = channel.three_d().pipeline_dependencies(&[]);
        assert_ne!(vertex_id, defaults);

        program_three_d(&mut channel, 0x1610, 0x3f);
        assert_ne!(channel.three_d().pipeline_dependencies(&[]), vertex_id);
    }

    #[test]
    fn invalid_vertex_assembly_controls_and_packet_suffix_are_atomic() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        program_three_d(&mut channel, 0x1610, 0x0e);
        program_three_d(&mut channel, 0x164c, 0x1000);

        for (method, method_name, arguments) in [
            (
                0x1610,
                "SET_ATTRIBUTE_DEFAULT",
                [0x40, 0x100, 0x8000_0000, u32::MAX],
            ),
            (0x164c, "SET_DA_OUTPUT", [1, 0x2000, 0x1001, u32::MAX]),
        ] {
            for argument in arguments {
                let frontend_before = channel.frontend();
                let two_d_before = channel.two_d().clone();
                let three_d_before = channel.three_d().clone();
                let decoded = packet(method / 4, argument);
                assert!(matches!(
                    dispatch_maxwell_engine_packet(
                        &mut channel,
                        FrontendSubmissionId::new(3),
                        &decoded.packets()[0]
                    ),
                    Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                        source,
                        method_name: actual,
                        ..
                    }) if actual == method_name && source.argument() == argument
                ));
                assert_eq!(channel.frontend(), frontend_before);
                assert_eq!(channel.two_d(), &two_d_before);
                assert_eq!(channel.three_d(), &three_d_before);
            }
        }

        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = incrementing_packet(0x1648 / 4, &[0xdead_beef, 1]);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                source,
                method_name: "SET_DA_OUTPUT",
                ..
            }) if source.argument() == 1
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
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
    fn program_region_is_source_preserving_and_only_active_for_shader_pipelines() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        let two_d_before = channel.two_d().clone();
        let inactive_dependencies = channel.three_d().pipeline_dependencies(&[]);

        let lower = packet(0x160c / 4, 0);
        let lower_dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &lower.packets()[0],
        )
        .unwrap();
        let lower_source = lower_dispatch.methods()[0].method().source();
        let region = channel.three_d().shader_bindings().program_region();
        assert_eq!(
            lower_dispatch.methods()[0].metadata().method_name(),
            "SET_PROGRAM_REGION_B"
        );
        assert_eq!(
            lower_dispatch.methods()[0].effect(),
            MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::ShaderBinding(
                MaxwellThreeDShaderBindingWrite::ProgramRegionAddressLower {
                    value: 0,
                    source: lower_source,
                }
            ))
        );
        assert!(region.address().is_none());
        assert_eq!(region.address_lower().raw(), Some(0));
        assert_eq!(region.address_lower().source(), Some(lower_source));
        assert_eq!(
            channel.three_d().pipeline_dependencies(&[]),
            inactive_dependencies
        );

        let upper = packet(0x1608 / 4, 4);
        let upper_dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &upper.packets()[0],
        )
        .unwrap();
        let upper_source = upper_dispatch.methods()[0].method().source();
        let region = channel.three_d().shader_bindings().program_region();
        assert_eq!(
            upper_dispatch.methods()[0].metadata().method_name(),
            "SET_PROGRAM_REGION_A"
        );
        assert_eq!(region.address_upper().raw(), Some(4));
        assert_eq!(region.address_upper().source(), Some(upper_source));
        assert_eq!(region.address().unwrap().get(), 0x0000_0004_0000_0000);
        assert_eq!(
            channel.three_d().pipeline_dependencies(&[]),
            inactive_dependencies
        );
        assert_eq!(channel.two_d(), &two_d_before);

        program_three_d(&mut channel, 0x2000, 0x11);
        let active_dependencies = channel.three_d().pipeline_dependencies(&[]);
        program_three_d(&mut channel, 0x160c, 0x1000);
        assert_ne!(
            channel.three_d().pipeline_dependencies(&[]),
            active_dependencies
        );
    }

    #[test]
    fn invalid_program_region_upper_and_packet_suffix_are_atomic() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        program_three_d(&mut channel, 0x1608, 4);
        program_three_d(&mut channel, 0x160c, 0);

        for argument in [0x100, 0x8000_0000, u32::MAX] {
            let frontend_before = channel.frontend();
            let two_d_before = channel.two_d().clone();
            let three_d_before = channel.three_d().clone();
            let decoded = packet(0x1608 / 4, argument);
            assert!(matches!(
                dispatch_maxwell_engine_packet(
                    &mut channel,
                    FrontendSubmissionId::new(3),
                    &decoded.packets()[0]
                ),
                Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                    source,
                    method_name: "SET_PROGRAM_REGION_A",
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
        let decoded = incrementing_packet(0x1608 / 4, &[5, 0x1000, 0x40]);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                source,
                method_name: "SET_ATTRIBUTE_DEFAULT",
                ..
            }) if source.method() == GpuMethodId(0x1610)
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }

    #[test]
    fn vertex_stream_substitute_address_is_typed_source_preserving_and_pipeline_neutral() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        let two_d_before = channel.two_d().clone();
        let dependencies_before = channel.three_d().pipeline_dependencies(&[]);

        let lower = packet(0x0f88 / 4, 0x082c_3000);
        let lower_dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &lower.packets()[0],
        )
        .unwrap();
        let lower_method = lower_dispatch.methods()[0];
        let lower_source = lower_method.method().source();
        assert_eq!(
            lower_method.metadata().method_name(),
            "SET_VERTEX_STREAM_SUBSTITUTE_B"
        );
        assert_eq!(
            lower_method.effect(),
            MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::VertexInput(
                MaxwellThreeDVertexInputWrite::StreamSubstituteAddressLower {
                    value: 0x082c_3000,
                    source: lower_source,
                }
            ))
        );
        let substitute = channel.three_d().vertex_input().stream_substitute();
        assert!(substitute.address().is_none());
        assert_eq!(substitute.address_lower().raw(), Some(0x082c_3000));
        assert_eq!(substitute.address_lower().source(), Some(lower_source));

        let upper = packet(0x0f84 / 4, 0x7f);
        let upper_dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &upper.packets()[0],
        )
        .unwrap();
        let upper_method = upper_dispatch.methods()[0];
        let upper_source = upper_method.method().source();
        assert_eq!(
            upper_method.metadata().method_name(),
            "SET_VERTEX_STREAM_SUBSTITUTE_A"
        );
        assert_eq!(
            upper_method.effect(),
            MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::VertexInput(
                MaxwellThreeDVertexInputWrite::StreamSubstituteAddressUpper {
                    value: 0x7f,
                    source: upper_source,
                }
            ))
        );
        let substitute = channel.three_d().vertex_input().stream_substitute();
        assert_eq!(substitute.address_upper().raw(), Some(0x7f));
        assert_eq!(substitute.address_upper().source(), Some(upper_source));
        assert_eq!(substitute.address().unwrap().get(), 0x7f_082c_3000);
        assert_eq!(
            channel.three_d().pipeline_dependencies(&[]),
            dependencies_before
        );
        assert_eq!(channel.two_d(), &two_d_before);
    }

    #[test]
    fn invalid_vertex_stream_substitute_upper_and_packet_are_atomic() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        program_three_d(&mut channel, 0x0f84, 4);
        program_three_d(&mut channel, 0x0f88, 0x1000);

        for argument in [0x100, 0x8000_0000, u32::MAX] {
            let frontend_before = channel.frontend();
            let two_d_before = channel.two_d().clone();
            let three_d_before = channel.three_d().clone();
            let decoded = incrementing_packet(0x0f84 / 4, &[argument, 0x082c_3000]);
            assert!(matches!(
                dispatch_maxwell_engine_packet(
                    &mut channel,
                    FrontendSubmissionId::new(3),
                    &decoded.packets()[0]
                ),
                Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                    source,
                    method_name: "SET_VERTEX_STREAM_SUBSTITUTE_A",
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
        let decoded = incrementing_packet(0x0f84 / 4, &[0, 0x082c_3000, 0]);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
                if source.method() == GpuMethodId(0x0f8c)
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }

    #[test]
    fn active_shader_pipeline_requires_complete_program_region_but_clear_does_not() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        program_three_d(&mut channel, 0x121c, 0);
        program_three_d(&mut channel, 0x2000, 0x11);
        program_three_d(&mut channel, 0x160c, 0);
        let resources =
            resolve_maxwell_three_d_resources(channel.three_d(), &resource_address_space())
                .unwrap();
        let cache = MaxwellThreeDLoweringCache::default();
        let cache_before = cache.clone();
        let source = channel
            .three_d()
            .shader_bindings()
            .program_region()
            .address_lower()
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
                FrontendSubmissionId::new(10),
                Vec::new(),
                &lowering_capabilities(BackendFeatures::empty()),
                &cache,
            ),
            Err(MaxwellThreeDLoweringError::IncompleteDraw(
                "SET_PROGRAM_REGION_A/B"
            ))
        ));

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
                FrontendSubmissionId::new(11),
                Vec::new(),
                &lowering_capabilities(BackendFeatures::empty()),
                &cache,
            ),
            Err(MaxwellThreeDLoweringError::IncompleteClear(
                "horizontal rectangle"
            ))
        ));

        program_three_d(&mut channel, 0x1608, 4);
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
                &lowering_capabilities(BackendFeatures::empty()),
                &cache,
            ),
            Err(MaxwellThreeDLoweringError::ShaderTranslationRequired)
        ));
        assert_eq!(cache, cache_before);
    }

    #[test]
    fn constant_buffer_inline_load_tracks_typed_cursor_and_upload_effects() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        let selector = incrementing_packet(0x2380 / 4, &[0x10, 0, 0x82c_3000]);
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &selector.packets()[0],
        )
        .unwrap();
        let dependencies_before = channel.three_d().pipeline_dependencies(&[]);

        let load = incrementing_packet(0x238c / 4, &[0, 0x1122_3344, 0x5566_7788]);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &load.packets()[0],
        )
        .unwrap();
        assert_eq!(dispatch.methods().len(), 3);
        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "LOAD_CONSTANT_BUFFER_OFFSET"
        );
        let state = channel.three_d().shader_bindings().constant_buffer_load();
        assert_eq!(state.offset().value(), Some(&0));
        assert_eq!(
            state.offset().source(),
            Some(dispatch.methods()[0].method().source())
        );
        assert_eq!(state.next_offset(), Some(8));
        assert_eq!(state.last_data().value(), Some(&0x5566_7788));
        assert_eq!(
            state.last_data().source(),
            Some(dispatch.methods()[2].method().source())
        );

        for (index, (offset, value)) in [(0, 0x1122_3344), (4, 0x5566_7788)].into_iter().enumerate()
        {
            let method = dispatch.methods()[index + 1];
            assert_eq!(method.metadata().method_name(), "LOAD_CONSTANT_BUFFER");
            assert_eq!(
                method.effect(),
                MaxwellEngineMethodEffect::ThreeDStateAndInlineConstantBufferUpload {
                    state: MaxwellThreeDStateWrite::ShaderBinding(
                        MaxwellThreeDShaderBindingWrite::ConstantBufferLoadData {
                            value,
                            next_offset: offset + 4,
                            source: method.method().source(),
                        },
                    ),
                    upload: MaxwellThreeDInlineConstantBufferUpload::new(
                        MaxwellThreeDUnresolvedAddress::new(0, 0x82c_3000),
                        offset,
                        value,
                        method.method().source(),
                    ),
                }
            );
        }
        assert_eq!(
            channel.three_d().pipeline_dependencies(&[]),
            dependencies_before
        );
    }

    #[test]
    fn constant_buffer_inline_load_validates_fields_sequence_bounds_and_atomicity() {
        let mut channel = channel();
        bind_three_d(&mut channel);

        let before = channel.three_d().clone();
        let data_without_selector = packet(0x2390 / 4, 1);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &data_without_selector.packets()[0],
            ),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                method_name: "LOAD_CONSTANT_BUFFER",
                ..
            })
        ));
        assert_eq!(channel.three_d(), &before);

        let invalid_offset = packet(0x238c / 4, 0x1_0000);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &invalid_offset.packets()[0],
            ),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                method_name: "LOAD_CONSTANT_BUFFER_OFFSET",
                ..
            })
        ));
        assert_eq!(channel.three_d(), &before);

        for (method, argument) in [(0x2380, 4), (0x2384, 0), (0x2388, 0x1000)] {
            program_three_d(&mut channel, method, argument);
        }
        let before_packet = channel.three_d().clone();
        let overflow = increment_once_packet(0x238c / 4, &[0, 1, 2]);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &overflow.packets()[0],
            ),
            Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                source,
                method_name: "LOAD_CONSTANT_BUFFER",
                ..
            }) if source.argument() == 2
        ));
        assert_eq!(channel.three_d(), &before_packet);
    }

    #[test]
    fn conditional_constant_buffer_inline_load_is_an_explicit_host_boundary() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        for (method, argument) in [
            (0x2380, 4),
            (0x2384, 0),
            (0x2388, 0x1000),
            (0x238c, 0),
            (0x030c, 1),
        ] {
            program_three_d(&mut channel, method, argument);
        }
        let before = channel.three_d().clone();
        let data = packet(0x2390 / 4, 0xfeed_beef);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &data.packets()[0],
            ),
            Err(MaxwellEngineDispatchError::UnsupportedConditionalConstantBufferLoad {
                source,
            }) if source.argument() == 0xfeed_beef
        ));
        assert_eq!(channel.three_d(), &before);

        program_three_d(&mut channel, 0x030c, 0);
        program_three_d(&mut channel, 0x2390, 0xfeed_beef);
        assert_eq!(
            channel
                .three_d()
                .shader_bindings()
                .constant_buffer_load()
                .next_offset(),
            Some(4)
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
            (0x10f8, 0),
            (0x0d6c, (50 << 16) | 10),
            (0x0d70, (25 << 16) | 5),
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
        let GpuCommand::Clear(nixe_gpu::ClearOperation::Image { target, .. }) =
            first.submission().operations()[0].command()
        else {
            panic!("full-surface clear did not lower to an image clear");
        };
        assert_eq!(target.origin, nixe_gpu::ImageOrigin { x: 0, y: 0, z: 0 });
        assert_eq!(target.extent.width, 64);
        assert_eq!(target.extent.height, 32);

        program_three_d(&mut channel, 0x10f8, 0x10);
        let rectangular_clear = packet(0x19d0 / 4, 0x3c);
        let rectangular_dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &rectangular_clear.packets()[0],
        )
        .unwrap();
        let rectangular_trigger = &rectangular_dispatch.operations()[0];
        let rectangular_resources =
            resolve_maxwell_three_d_resources(rectangular_trigger.state(), &address_space).unwrap();
        let rectangular = preflight_maxwell_three_d_operation(
            rectangular_trigger.state(),
            &rectangular_resources,
            rectangular_trigger.trigger(),
            None,
            FrontendSubmissionId::new(12),
            Vec::new(),
            &capabilities,
            &MaxwellThreeDLoweringCache::default(),
        )
        .unwrap();
        let GpuCommand::Clear(nixe_gpu::ClearOperation::Image { target, .. }) =
            rectangular.submission().operations()[0].command()
        else {
            panic!("rectangular clear did not lower to an image clear");
        };
        assert_eq!(target.origin, nixe_gpu::ImageOrigin { x: 10, y: 5, z: 0 });
        assert_eq!(target.extent.width, 40);
        assert_eq!(target.extent.height, 20);
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
                MaxwellThreeDTranslatedShader::new(
                    ShaderStage::Vertex,
                    ShaderId::new(1),
                    7,
                    MaxwellThreeDDirectlyAddressableMemory::Size48KiB,
                ),
                MaxwellThreeDTranslatedShader::new(
                    ShaderStage::Fragment,
                    ShaderId::new(2),
                    9,
                    MaxwellThreeDDirectlyAddressableMemory::Size48KiB,
                ),
            ],
            Vec::new(),
        )
        .unwrap();
        let cache_before = cache.clone();
        assert!(matches!(
            preflight_maxwell_three_d_operation(
                triggered.state(),
                &resources,
                triggered.trigger(),
                Some(&shaders),
                FrontendSubmissionId::new(20),
                Vec::new(),
                &capabilities,
                &cache,
            ),
            Err(MaxwellThreeDLoweringError::IncompleteDraw(
                "SET_L1_CONFIGURATION"
            ))
        ));
        assert_eq!(cache, cache_before);

        program_three_d(&mut channel, 0x0308, 3);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &draw.packets()[0],
        )
        .unwrap();
        let triggered = &dispatch.operations()[0];
        let resources =
            resolve_maxwell_three_d_resources(triggered.state(), &address_space).unwrap();
        let incompatible_shaders = MaxwellThreeDTranslatedShaders::new(
            vec![
                MaxwellThreeDTranslatedShader::new(
                    ShaderStage::Vertex,
                    ShaderId::new(1),
                    7,
                    MaxwellThreeDDirectlyAddressableMemory::Size16KiB,
                ),
                MaxwellThreeDTranslatedShader::new(
                    ShaderStage::Fragment,
                    ShaderId::new(2),
                    9,
                    MaxwellThreeDDirectlyAddressableMemory::Size16KiB,
                ),
            ],
            Vec::new(),
        )
        .unwrap();
        let cache_before = cache.clone();
        assert!(matches!(
            preflight_maxwell_three_d_operation(
                triggered.state(),
                &resources,
                triggered.trigger(),
                Some(&incompatible_shaders),
                FrontendSubmissionId::new(20),
                Vec::new(),
                &capabilities,
                &cache,
            ),
            Err(
                MaxwellThreeDLoweringError::TranslatedShaderMemoryConfigurationMismatch {
                    configured: MaxwellThreeDDirectlyAddressableMemory::Size48KiB,
                    required: MaxwellThreeDDirectlyAddressableMemory::Size16KiB,
                    ..
                }
            )
        ));
        assert_eq!(cache, cache_before);
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
    fn ct_mrt_enable_is_typed_source_preserving_and_atomic() {
        let mut channel = channel();
        bind_three_d(&mut channel);

        for (argument, expected) in [
            (0, MaxwellThreeDSeparateFragmentData::Disabled),
            (1, MaxwellThreeDSeparateFragmentData::Enabled),
        ] {
            let decoded = packet(0x0fac / 4, argument);
            let dispatch = dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0],
            )
            .unwrap();
            let method = dispatch.methods()[0];
            let source = method.method().source();
            let register = channel.three_d().render_targets().separate_fragment_data();

            assert_eq!(method.metadata().method_name(), "SET_CT_MRT_ENABLE");
            assert_eq!(
                method.effect(),
                MaxwellEngineMethodEffect::ThreeDState(MaxwellThreeDStateWrite::RenderTarget(
                    MaxwellThreeDRenderTargetWrite::SeparateFragmentData {
                        value: expected,
                        source,
                    }
                ))
            );
            assert!(dispatch.operations().is_empty());
            assert_eq!(expected.raw(), argument);
            assert_eq!(register.origin(), MaxwellThreeDRegisterOrigin::Programmed);
            assert_eq!(register.raw(), Some(argument));
            assert_eq!(register.value(), Some(&expected));
            assert_eq!(register.source(), Some(source));
        }

        for argument in [2, 3, 0x8000_0000, u32::MAX] {
            let frontend_before = channel.frontend();
            let two_d_before = channel.two_d().clone();
            let three_d_before = channel.three_d().clone();
            let decoded = packet(0x0fac / 4, argument);
            assert!(matches!(
                dispatch_maxwell_engine_packet(
                    &mut channel,
                    FrontendSubmissionId::new(3),
                    &decoded.packets()[0]
                ),
                Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                    source,
                    method_name: "SET_CT_MRT_ENABLE",
                    reason: "undefined boolean encoding or reserved bits",
                }) if source.argument() == argument
            ));
            assert_eq!(channel.frontend(), frontend_before);
            assert_eq!(channel.two_d(), &two_d_before);
            assert_eq!(channel.three_d(), &three_d_before);
        }

        let frontend_before = channel.frontend();
        let two_d_before = channel.two_d().clone();
        let three_d_before = channel.three_d().clone();
        let decoded = incrementing_packet(0x0fac / 4, &[0, 0]);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::UnknownMethod { source, .. })
                if source.method() == GpuMethodId(0x0fb0)
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.two_d(), &two_d_before);
        assert_eq!(channel.three_d(), &three_d_before);
    }

    #[test]
    fn ct_mrt_enable_only_affects_multi_target_draws_and_not_clears() {
        let vertex_allocation = CanonicalAllocation::zeroed(0x4000, 0x1000).unwrap();
        let target_zero_allocation = CanonicalAllocation::zeroed(0x10000, 0x1000).unwrap();
        let target_one_allocation = CanonicalAllocation::zeroed(0x10000, 0x1000).unwrap();
        let mut address_space = resource_address_space();
        let vertex = map_resource(
            &mut address_space,
            vertex_allocation
                .backing_range(MemoryPermissions::READ_WRITE)
                .unwrap(),
            91,
            0,
        )
        .offset()
        .get();
        let target_zero = map_resource(
            &mut address_space,
            target_zero_allocation
                .backing_range(MemoryPermissions::READ_WRITE)
                .unwrap(),
            92,
            0xfe,
        )
        .offset()
        .get();
        let target_one = map_resource(
            &mut address_space,
            target_one_allocation
                .backing_range(MemoryPermissions::READ_WRITE)
                .unwrap(),
            93,
            0xfe,
        )
        .offset()
        .get();
        let mut channel = channel();
        bind_three_d(&mut channel);
        program_basic_draw_state(&mut channel, vertex);
        program_color_target(&mut channel, 0, target_zero, 0xd5);
        program_color_target(&mut channel, 1, target_one, 0xcf);
        program_three_d(
            &mut channel,
            0x121c,
            color_target_selection_raw(2, [0, 1, 0, 0, 0, 0, 0, 0]),
        );

        program_three_d(&mut channel, 0x0fac, 0);
        let single_target_disabled = channel.three_d().pipeline_dependencies(&[0]);
        let multi_target_disabled = channel.three_d().pipeline_dependencies(&[0, 1]);
        program_three_d(&mut channel, 0x0fac, 1);
        assert_eq!(
            channel.three_d().pipeline_dependencies(&[0]),
            single_target_disabled
        );
        assert_ne!(
            channel.three_d().pipeline_dependencies(&[0, 1]),
            multi_target_disabled
        );

        let shaders = translated_graphics_shaders();
        let capabilities =
            lowering_capabilities(BackendFeatures::DRAW.union(BackendFeatures::RENDER_PASS));
        let cache = MaxwellThreeDLoweringCache::default();
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
            FrontendSubmissionId::new(10),
            Vec::new(),
            &capabilities,
            &cache,
        )
        .unwrap();

        program_three_d(&mut channel, 0x0fac, 0);
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
                FrontendSubmissionId::new(11),
                Vec::new(),
                &capabilities,
                &cache,
            ),
            Err(MaxwellThreeDLoweringError::UnsupportedReplicatedColorTargetOutputSemantics)
        ));
        assert_eq!(cache, MaxwellThreeDLoweringCache::default());

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
                &capabilities,
                &cache,
            ),
            Err(MaxwellThreeDLoweringError::IncompleteClear(
                "horizontal rectangle"
            ))
        ));
        assert_eq!(cache, MaxwellThreeDLoweringCache::default());
    }

    #[test]
    fn render_target_layer_only_blocks_effective_layered_draws() {
        let vertex_allocation = CanonicalAllocation::zeroed(0x4000, 0x1000).unwrap();
        let target_allocation = CanonicalAllocation::zeroed(0x10000, 0x1000).unwrap();
        let mut address_space = resource_address_space();
        let vertex = map_resource(
            &mut address_space,
            vertex_allocation
                .backing_range(MemoryPermissions::READ_WRITE)
                .unwrap(),
            94,
            0,
        )
        .offset()
        .get();
        let target = map_resource(
            &mut address_space,
            target_allocation
                .backing_range(MemoryPermissions::READ_WRITE)
                .unwrap(),
            95,
            0xfe,
        )
        .offset()
        .get();
        let mut channel = channel();
        bind_three_d(&mut channel);
        program_basic_draw_state(&mut channel, vertex);
        program_color_target(&mut channel, 0, target, 0xd5);
        program_three_d(&mut channel, 0x121c, 1);
        let capabilities =
            lowering_capabilities(BackendFeatures::DRAW.union(BackendFeatures::RENDER_PASS));
        let cache = MaxwellThreeDLoweringCache::default();

        program_three_d(&mut channel, 0x15cc, 0);
        let resources =
            resolve_maxwell_three_d_resources(channel.three_d(), &address_space).unwrap();
        let source = channel
            .three_d()
            .render_targets()
            .render_target_layer()
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
                FrontendSubmissionId::new(10),
                Vec::new(),
                &capabilities,
                &cache,
            ),
            Err(MaxwellThreeDLoweringError::ShaderTranslationRequired)
        ));

        for argument in [1, 0x0001_0000, 0x0001_0040] {
            program_three_d(&mut channel, 0x15cc, argument);
            let source = channel
                .three_d()
                .render_targets()
                .render_target_layer()
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
                Err(MaxwellThreeDLoweringError::UnsupportedRenderTargetLayerSemantics(value))
                    if value.raw() == argument
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
        assert!(matches!(
            preflight_maxwell_three_d_operation(
                triggered.state(),
                &resources,
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
    }

    #[test]
    fn compute_shader_memory_state_is_typed_source_preserving_and_atomic() {
        let mut channel = channel();
        bind_compute(&mut channel);
        let three_d_before = channel.three_d().clone();
        let two_d_before = channel.two_d().clone();

        let upper = packet_on_subchannel(1, 0x0790 / 4, 4);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &upper.packets()[0],
        )
        .unwrap();
        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_SHADER_LOCAL_MEMORY_A"
        );
        assert!(matches!(
            dispatch.methods()[0].effect(),
            MaxwellEngineMethodEffect::ComputeState(MaxwellComputeStateWrite::AddressUpper {
                value: 4,
                ..
            })
        ));
        let memory = channel.compute().shader_local_memory();
        assert_eq!(memory.address(), None);
        assert_eq!(memory.address_upper().raw(), Some(4));
        assert_eq!(memory.address_upper().value(), Some(&4));
        assert_eq!(
            memory.address_upper().origin(),
            MaxwellComputeRegisterOrigin::Programmed
        );
        assert!(memory.address_upper().source().is_some());

        let lower = packet_on_subchannel(1, 0x0794 / 4, 0x0008_0000);
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &lower.packets()[0],
        )
        .unwrap();
        assert_eq!(
            channel.compute().shader_local_memory().address(),
            Some(MaxwellComputeAddress::new(4, 0x0008_0000))
        );
        assert_eq!(
            channel
                .compute()
                .shader_local_memory()
                .address()
                .unwrap()
                .get(),
            0x0000_0004_0008_0000
        );

        for (method, arguments) in [
            (0x02e4, [0, 0x0040_8000, 0xff]),
            (0x02f0, [0, 0x0040_8000, 0xff]),
        ] {
            let decoded = incrementing_packet_on_subchannel(1, method / 4, &arguments);
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0],
            )
            .unwrap();
        }
        let memory = channel.compute().shader_local_memory();
        for allocation in [memory.non_throttled(), memory.throttled()] {
            assert_eq!(allocation.size(), Some(0x0040_8000));
            assert_eq!(allocation.max_sm_count().value().unwrap().get(), 0xff);
            assert!(allocation.size_upper().source().is_some());
            assert!(allocation.size_lower().source().is_some());
            assert!(allocation.max_sm_count().source().is_some());
        }

        for (method, argument, name) in [
            (0x077c, 0xff00_0000, "SET_SHADER_LOCAL_MEMORY_WINDOW"),
            (0x0214, 0xfe00_0000, "SET_SHADER_SHARED_MEMORY_WINDOW"),
        ] {
            let decoded = packet_on_subchannel(1, method / 4, argument);
            let dispatch = dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0],
            )
            .unwrap();
            assert_eq!(dispatch.methods()[0].metadata().method_name(), name);
        }
        let memory = channel.compute().shader_local_memory();
        assert_eq!(memory.local_window_base().value(), Some(&0xff00_0000));
        assert_eq!(memory.shared_window_base().value(), Some(&0xfe00_0000));
        assert_eq!(channel.three_d(), &three_d_before);
        assert_eq!(channel.two_d(), &two_d_before);

        for (method, argument, mask) in [
            (0x0790, 0x100, 0xff),
            (0x02e4, 0x100, 0xff),
            (0x02ec, 0x200, 0x1ff),
            (0x02f0, 0x100, 0xff),
            (0x02f8, 0x200, 0x1ff),
        ] {
            let frontend_before = channel.frontend();
            let compute_before = channel.compute().clone();
            let decoded = packet_on_subchannel(1, method / 4, argument);
            assert!(matches!(
                dispatch_maxwell_engine_packet(
                    &mut channel,
                    FrontendSubmissionId::new(3),
                    &decoded.packets()[0]
                ),
                Err(MaxwellEngineDispatchError::InvalidMethodValue {
                    source,
                    defined_mask,
                    ..
                }) if source.argument() == argument && defined_mask == mask
            ));
            assert_eq!(channel.frontend(), frontend_before);
            assert_eq!(channel.compute(), &compute_before);
        }

        let frontend_before = channel.frontend();
        let compute_before = channel.compute().clone();
        let decoded = incrementing_packet_on_subchannel(1, 0x02e4 / 4, &[1, 2, 0x200]);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::InvalidMethodValue { source, .. })
                if source.method() == GpuMethodId(0x02ec)
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.compute(), &compute_before);
    }

    #[test]
    fn compute_program_state_is_typed_source_preserving_and_atomic() {
        let mut channel = channel();
        bind_compute(&mut channel);

        let lower = packet_on_subchannel(1, 0x160c / 4, 0x0123_4000);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &lower.packets()[0],
        )
        .unwrap();
        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_PROGRAM_REGION_B"
        );
        assert!(matches!(
            dispatch.methods()[0].effect(),
            MaxwellEngineMethodEffect::ComputeState(
                MaxwellComputeStateWrite::ProgramRegionAddressLower {
                    value: 0x0123_4000,
                    ..
                }
            )
        ));
        assert_eq!(channel.compute().program().region_address(), None);

        let upper = packet_on_subchannel(1, 0x1608 / 4, 4);
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &upper.packets()[0],
        )
        .unwrap();
        let program = channel.compute().program();
        assert_eq!(
            program.region_address(),
            Some(MaxwellComputeAddress::new(4, 0x0123_4000))
        );
        assert_eq!(
            program.region_address().unwrap().get(),
            0x0000_0004_0123_4000
        );
        assert_eq!(program.region_address_upper().raw(), Some(4));
        assert_eq!(program.region_address_lower().raw(), Some(0x0123_4000));
        assert_eq!(
            program.region_address_upper().origin(),
            MaxwellComputeRegisterOrigin::Programmed
        );
        assert!(program.region_address_upper().source().is_some());
        assert!(program.region_address_lower().source().is_some());

        let spa = packet_on_subchannel(1, 0x0310 / 4, 0x0400);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &spa.packets()[0],
        )
        .unwrap();
        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_SPA_VERSION"
        );
        let version = *channel.compute().program().spa_version().value().unwrap();
        assert_eq!(version.major(), 4);
        assert_eq!(version.minor(), 0);
        assert_eq!(version.raw(), 0x0400);
        assert_eq!(
            channel.compute().program().spa_version().raw(),
            Some(0x0400)
        );
        assert!(channel.compute().program().spa_version().source().is_some());

        for (method, argument, mask) in [(0x1608, 0x100, 0xff), (0x0310, 0x1_0000, 0xffff)] {
            let compute_before = channel.compute().clone();
            let decoded = packet_on_subchannel(1, method / 4, argument);
            assert!(matches!(
                dispatch_maxwell_engine_packet(
                    &mut channel,
                    FrontendSubmissionId::new(3),
                    &decoded.packets()[0]
                ),
                Err(MaxwellEngineDispatchError::InvalidMethodValue {
                    source,
                    defined_mask,
                    ..
                }) if source.argument() == argument && defined_mask == mask
            ));
            assert_eq!(channel.compute(), &compute_before);
        }

        let frontend_before = channel.frontend();
        let compute_before = channel.compute().clone();
        let decoded = non_incrementing_packet_on_subchannel(1, 0x0310 / 4, &[0x0501, 0x1_0000]);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &decoded.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::InvalidMethodValue { source, .. })
                if source.method() == GpuMethodId(0x0310)
                    && source.argument() == 0x1_0000
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.compute(), &compute_before);
    }

    #[test]
    fn compute_descriptor_pools_are_typed_source_preserving_and_atomic() {
        let mut channel = channel();
        bind_compute(&mut channel);

        let header_lower = packet_on_subchannel(1, 0x1578 / 4, 0x0410_0000);
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &header_lower.packets()[0],
        )
        .unwrap();
        assert_eq!(channel.compute().texture_headers().address(), None);

        let header_upper = packet_on_subchannel(1, 0x1574 / 4, 4);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &header_upper.packets()[0],
        )
        .unwrap();
        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_TEX_HEADER_POOL_A"
        );
        assert!(matches!(
            dispatch.methods()[0].effect(),
            MaxwellEngineMethodEffect::ComputeState(
                MaxwellComputeStateWrite::TextureHeaderAddressUpper { value: 4, .. }
            )
        ));
        let header_maximum = packet_on_subchannel(1, 0x157c / 4, 0x003f_ffff);
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &header_maximum.packets()[0],
        )
        .unwrap();
        let headers = channel.compute().texture_headers();
        assert_eq!(
            headers.address(),
            Some(MaxwellComputeAddress::new(4, 0x0410_0000))
        );
        assert_eq!(headers.address().unwrap().get(), 0x0000_0004_0410_0000);
        assert_eq!(headers.maximum_index().value(), Some(&0x003f_ffff));
        assert!(headers.address_upper().source().is_some());
        assert!(headers.address_lower().source().is_some());
        assert!(headers.maximum_index().source().is_some());

        let sampler =
            incrementing_packet_on_subchannel(1, 0x155c / 4, &[4, 0x0411_0000, 0x000f_ffff]);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &sampler.packets()[0],
        )
        .unwrap();
        assert_eq!(dispatch.methods().len(), 3);
        assert_eq!(
            dispatch.methods()[2].metadata().method_name(),
            "SET_TEX_SAMPLER_POOL_C"
        );
        let samplers = channel.compute().samplers();
        assert_eq!(
            samplers.address(),
            Some(MaxwellComputeAddress::new(4, 0x0411_0000))
        );
        assert_eq!(samplers.maximum_index().value(), Some(&0x000f_ffff));
        assert!(samplers.address_upper().source().is_some());
        assert!(samplers.address_lower().source().is_some());
        assert!(samplers.maximum_index().source().is_some());

        for (method, argument, mask) in [
            (0x155c, 0x100, 0xff),
            (0x1564, 0x0010_0000, 0x000f_ffff),
            (0x1574, 0x100, 0xff),
            (0x157c, 0x0040_0000, 0x003f_ffff),
        ] {
            let compute_before = channel.compute().clone();
            let decoded = packet_on_subchannel(1, method / 4, argument);
            assert!(matches!(
                dispatch_maxwell_engine_packet(
                    &mut channel,
                    FrontendSubmissionId::new(3),
                    &decoded.packets()[0]
                ),
                Err(MaxwellEngineDispatchError::InvalidMethodValue {
                    source,
                    defined_mask,
                    ..
                }) if source.argument() == argument && defined_mask == mask
            ));
            assert_eq!(channel.compute(), &compute_before);
        }

        let frontend_before = channel.frontend();
        let compute_before = channel.compute().clone();
        let invalid_header =
            incrementing_packet_on_subchannel(1, 0x1574 / 4, &[5, 0x1234_0000, 0x0040_0000]);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &invalid_header.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::InvalidMethodValue { source, .. })
                if source.method() == GpuMethodId(0x157c)
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.compute(), &compute_before);
    }

    #[test]
    fn compute_bindless_texture_slot_is_typed_source_preserving_and_atomic() {
        let mut channel = channel();
        bind_compute(&mut channel);
        assert_eq!(
            channel
                .compute()
                .bindless_texture_constant_buffer_slot()
                .origin(),
            MaxwellComputeRegisterOrigin::Unset
        );

        let slots = non_incrementing_packet_on_subchannel(1, 0x2608 / 4, &[0, 7]);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &slots.packets()[0],
        )
        .unwrap();
        assert_eq!(dispatch.methods().len(), 2);
        assert_eq!(
            dispatch.methods()[1].metadata().method_name(),
            "SET_BINDLESS_TEXTURE"
        );
        assert!(matches!(
            dispatch.methods()[1].effect(),
            MaxwellEngineMethodEffect::ComputeState(
                MaxwellComputeStateWrite::BindlessTextureConstantBufferSlot {
                    value,
                    source,
                }
            ) if value.get() == 7
                && source.method() == GpuMethodId(0x2608)
                && source.argument() == 7
        ));
        let slot = channel.compute().bindless_texture_constant_buffer_slot();
        assert_eq!(slot.value().unwrap().get(), 7);
        assert_eq!(slot.raw(), Some(7));
        assert_eq!(slot.origin(), MaxwellComputeRegisterOrigin::Programmed);
        assert_eq!(slot.source().unwrap().method(), GpuMethodId(0x2608));

        let compute_before = channel.compute().clone();
        let invalid = packet_on_subchannel(1, 0x2608 / 4, 8);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &invalid.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::InvalidMethodValue {
                source,
                defined_mask: 0x0000_0007,
                ..
            }) if source.argument() == 8
        ));
        assert_eq!(channel.compute(), &compute_before);

        let frontend_before = channel.frontend();
        let invalid_packet = non_incrementing_packet_on_subchannel(1, 0x2608 / 4, &[6, 8]);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &invalid_packet.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::InvalidMethodValue { source, .. })
                if source.argument() == 8
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.compute(), &compute_before);
    }

    #[test]
    fn compute_inline_to_memory_pitch_upload_is_typed_ordered_and_atomic() {
        let mut channel = channel();
        bind_compute(&mut channel);

        let empty_before = channel.compute().clone();
        let unconfigured_launch = packet_on_subchannel(1, 0x01b0 / 4, 0x41);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &unconfigured_launch.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::InvalidComputeMethodEncoding {
                method_name: "LAUNCH_DMA",
                reason: "launch requires a complete destination address",
                ..
            })
        ));
        assert_eq!(channel.compute(), &empty_before);

        let address = incrementing_packet_on_subchannel(1, 0x0188 / 4, &[0, 0x082b_30c0]);
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &address.packets()[0],
        )
        .unwrap();
        let dimensions = incrementing_packet_on_subchannel(1, 0x0180 / 4, &[0x40, 1]);
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &dimensions.packets()[0],
        )
        .unwrap();

        let data = [0, 0, 1, 0, 0, 1, 1, 1, 2, 0, 3, 0, 2, 1, 3, 1];
        let mut launch_arguments = Vec::with_capacity(data.len() + 1);
        launch_arguments.push(0x41);
        launch_arguments.extend_from_slice(&data);
        let launch = increment_once_packet_on_subchannel(1, 0x01b0 / 4, &launch_arguments);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &launch.packets()[0],
        )
        .unwrap();

        assert_eq!(dispatch.methods().len(), 17);
        assert_eq!(dispatch.methods()[0].metadata().method_name(), "LAUNCH_DMA");
        assert!(matches!(
            dispatch.methods()[0].effect(),
            MaxwellEngineMethodEffect::ComputeState(
                MaxwellComputeStateWrite::InlineToMemoryLaunch {
                    value,
                    pending,
                    source,
                }
            ) if value.layout() == MaxwellComputeInlineToMemoryLayout::Pitch
                && value.system_memory_barrier_disabled()
                && pending.address().get() == 0x082b_30c0
                && pending.byte_length() == 0x40
                && source.argument() == 0x41
        ));
        for (index, expected) in data.into_iter().enumerate() {
            let method = dispatch.methods()[index + 1];
            assert_eq!(method.metadata().method_name(), "LOAD_INLINE_DATA");
            assert!(matches!(
                method.effect(),
                MaxwellEngineMethodEffect::ComputeStateAndInlineToMemoryUpload {
                    state: MaxwellComputeStateWrite::InlineToMemoryData {
                        value,
                        next_offset,
                        ..
                    },
                    upload,
                } if value == expected
                    && next_offset == (index as u32 + 1) * 4
                    && upload.address().get() == 0x082b_30c0
                    && upload.offset() == index as u32 * 4
                    && upload.value() == expected
                    && upload.source().method() == GpuMethodId(0x01b4)
            ));
        }
        let inline = channel.compute().inline_to_memory();
        assert_eq!(inline.address().unwrap().get(), 0x082b_30c0);
        assert_eq!(inline.line_length().value(), Some(&0x40));
        assert_eq!(inline.line_count().value(), Some(&1));
        assert_eq!(inline.launch().raw(), Some(0x41));
        assert_eq!(inline.last_data().value(), Some(&1));
        assert_eq!(inline.pending(), None);
        assert!(inline.address_upper().source().is_some());
        assert!(inline.address_lower().source().is_some());

        let short = packet_on_subchannel(1, 0x0180 / 4, 4);
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &short.packets()[0],
        )
        .unwrap();
        let frontend_before = channel.frontend();
        let compute_before = channel.compute().clone();
        let excessive =
            increment_once_packet_on_subchannel(1, 0x01b0 / 4, &[0x41, 0x1122_3344, 0x5566_7788]);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &excessive.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::InvalidComputeMethodEncoding {
                source,
                method_name: "LOAD_INLINE_DATA",
                ..
            }) if source.argument() == 0x5566_7788
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.compute(), &compute_before);

        let reserved = packet_on_subchannel(1, 0x01b0 / 4, 0x80);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &reserved.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::InvalidMethodValue {
                source,
                defined_mask: 0x0000_f37f,
                ..
            }) if source.argument() == 0x80
        ));
        assert_eq!(channel.compute(), &compute_before);
    }

    #[test]
    fn compute_shader_cache_invalidation_is_typed_ordered_and_atomic() {
        let mut channel = channel();
        bind_compute(&mut channel);

        let address = incrementing_packet_on_subchannel(1, 0x0188 / 4, &[0, 0x082b_30c0]);
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &address.packets()[0],
        )
        .unwrap();
        let dimensions = incrementing_packet_on_subchannel(1, 0x0180 / 4, &[4, 1]);
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &dimensions.packets()[0],
        )
        .unwrap();
        let upload = increment_once_packet_on_subchannel(1, 0x01b0 / 4, &[0x41, 0xdead_beef]);
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &upload.packets()[0],
        )
        .unwrap();
        let compute_before = channel.compute().clone();
        assert_eq!(
            compute_before.inline_to_memory().last_data().value(),
            Some(&0xdead_beef)
        );

        let invalidations = non_incrementing_packet_on_subchannel(1, 0x1698 / 4, &[0x1000, 0x1011]);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &invalidations.packets()[0],
        )
        .unwrap();
        assert_eq!(dispatch.methods().len(), 2);
        assert_eq!(dispatch.compute_operations().len(), 2);
        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "INVALIDATE_SHADER_CACHES_NO_WFI"
        );
        for (index, expected) in [
            MaxwellComputeShaderCacheInvalidation::new(false, false, true),
            MaxwellComputeShaderCacheInvalidation::new(true, true, true),
        ]
        .into_iter()
        .enumerate()
        {
            assert!(matches!(
                dispatch.methods()[index].effect(),
                MaxwellEngineMethodEffect::ComputeTrigger(
                    MaxwellComputeOperationTrigger::InvalidateShaderCachesNoWfi {
                        caches,
                        source,
                    }
                ) if caches == expected && source.method() == GpuMethodId(0x1698)
            ));
            assert_eq!(
                dispatch.compute_operations()[index].state(),
                &compute_before
            );
            assert_eq!(
                lower_maxwell_compute_synchronization(&dispatch.compute_operations()[index], true),
                MaxwellComputeSynchronizationPlan::InvalidateShaderCachesNoWfi { caches: expected }
            );
        }
        let captured = match dispatch.compute_operations()[0].trigger() {
            MaxwellComputeOperationTrigger::InvalidateShaderCachesNoWfi { caches, .. } => caches,
            _ => unreachable!(),
        };
        assert!(!captured.instruction());
        assert!(!captured.global_data());
        assert!(captured.constant());
        assert_eq!(channel.compute(), &compute_before);

        let frontend_before = channel.frontend();
        let invalid = non_incrementing_packet_on_subchannel(1, 0x1698 / 4, &[0x1000, 0x1002]);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &invalid.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::InvalidMethodValue {
                source,
                defined_mask: 0x0000_1011,
                ..
            }) if source.argument() == 0x1002
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.compute(), &compute_before);
    }

    #[test]
    fn compute_cwd_reference_counter_bank_is_typed_source_preserving_and_atomic() {
        let mut channel = channel();
        bind_compute(&mut channel);

        let arguments = (0..MAXWELL_COMPUTE_CWD_REF_COUNTER_COUNT)
            .rev()
            .map(|index| (0x0380 << 8) | index as u32)
            .collect::<Vec<_>>();
        let counters = non_incrementing_packet_on_subchannel(1, 0x0248 / 4, arguments.as_slice());
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &counters.packets()[0],
        )
        .unwrap();
        assert_eq!(
            dispatch.methods().len(),
            MAXWELL_COMPUTE_CWD_REF_COUNTER_COUNT
        );
        assert_eq!(
            dispatch.methods()[0].metadata().method_name(),
            "SET_CWD_REF_COUNTER"
        );
        assert!(matches!(
            dispatch.methods()[0].effect(),
            MaxwellEngineMethodEffect::ComputeState(
                MaxwellComputeStateWrite::CwdReferenceCounter {
                    index,
                    value,
                    ..
                }
            ) if index.get() == 63 && value.get() == 0x0380
        ));

        let bank = channel.compute().cwd_reference_counters();
        assert_eq!(bank.entries().len(), MAXWELL_COMPUTE_CWD_REF_COUNTER_COUNT);
        for index in 0..MAXWELL_COMPUTE_CWD_REF_COUNTER_COUNT {
            let index = MaxwellComputeCwdRefCounterIndex::new(index as u8).unwrap();
            let register = bank.entry(index);
            assert_eq!(register.value().unwrap().get(), 0x0380);
            assert_eq!(register.raw(), Some((0x0380 << 8) | u32::from(index.get())));
            assert_eq!(register.origin(), MaxwellComputeRegisterOrigin::Programmed);
            assert!(register.source().is_some());
        }
        assert_eq!(MaxwellComputeCwdRefCounterIndex::new(64), None);

        let overwrite = packet_on_subchannel(1, 0x0248 / 4, 0xffff << 8);
        dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &overwrite.packets()[0],
        )
        .unwrap();
        let bank = channel.compute().cwd_reference_counters();
        assert_eq!(
            bank.entry(MaxwellComputeCwdRefCounterIndex::new(0).unwrap())
                .value(),
            Some(&MaxwellComputeCwdRefCounterValue::new(0xffff))
        );
        assert_eq!(
            bank.entry(MaxwellComputeCwdRefCounterIndex::new(1).unwrap())
                .value(),
            Some(&MaxwellComputeCwdRefCounterValue::new(0x0380))
        );

        for argument in [0x0000_0040, 0x0100_0000] {
            let compute_before = channel.compute().clone();
            let invalid = packet_on_subchannel(1, 0x0248 / 4, argument);
            assert!(matches!(
                dispatch_maxwell_engine_packet(
                    &mut channel,
                    FrontendSubmissionId::new(3),
                    &invalid.packets()[0]
                ),
                Err(MaxwellEngineDispatchError::InvalidMethodValue {
                    source,
                    defined_mask: 0x00ff_ff3f,
                    ..
                }) if source.argument() == argument
            ));
            assert_eq!(channel.compute(), &compute_before);
        }

        let frontend_before = channel.frontend();
        let compute_before = channel.compute().clone();
        let invalid_packet =
            non_incrementing_packet_on_subchannel(1, 0x0248 / 4, &[0x0012_343f, 0x40]);
        assert!(matches!(
            dispatch_maxwell_engine_packet(
                &mut channel,
                FrontendSubmissionId::new(3),
                &invalid_packet.packets()[0]
            ),
            Err(MaxwellEngineDispatchError::InvalidMethodValue { source, .. })
                if source.argument() == 0x40
        ));
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.compute(), &compute_before);
    }

    #[test]
    fn compute_wait_for_idle_is_an_ordered_neutral_operation() {
        let mut channel = channel();
        bind_compute(&mut channel);
        let frontend_before = channel.frontend();
        let compute_before = channel.compute().clone();
        let three_d_before = channel.three_d().clone();
        let two_d_before = channel.two_d().clone();

        let waits = non_incrementing_packet_on_subchannel(1, 0x0110 / 4, &[0, 0xfeed_beef]);
        let dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &waits.packets()[0],
        )
        .unwrap();
        assert_eq!(dispatch.methods().len(), 2);
        assert_eq!(dispatch.compute_operations().len(), 2);
        assert!(dispatch.operations().is_empty());
        for (index, value) in [0, 0xfeed_beef].into_iter().enumerate() {
            assert_eq!(
                dispatch.methods()[index].metadata().method_name(),
                "WAIT_FOR_IDLE"
            );
            assert!(matches!(
                dispatch.methods()[index].effect(),
                MaxwellEngineMethodEffect::ComputeTrigger(
                    MaxwellComputeOperationTrigger::WaitForIdle {
                        value: actual,
                        source,
                    }
                ) if actual == value && source.argument() == value
            ));
            let operation = &dispatch.compute_operations()[index];
            assert!(matches!(
                operation.trigger(),
                MaxwellComputeOperationTrigger::WaitForIdle {
                    value: actual,
                    source,
                } if actual == value && source.argument() == value
            ));
            assert_eq!(operation.state(), &compute_before);
            assert_eq!(
                lower_maxwell_compute_synchronization(operation, false),
                MaxwellComputeSynchronizationPlan::WaitForIdle {
                    prior_work_pending: false,
                }
            );
            assert_eq!(
                lower_maxwell_compute_synchronization(operation, true),
                MaxwellComputeSynchronizationPlan::WaitForIdle {
                    prior_work_pending: true,
                }
            );
        }
        assert_eq!(channel.frontend(), frontend_before);
        assert_eq!(channel.compute(), &compute_before);
        assert_eq!(channel.three_d(), &three_d_before);
        assert_eq!(channel.two_d(), &two_d_before);
    }

    #[test]
    fn three_d_flush_and_syncpoint_increment_are_ordered_completion_operations() {
        let mut channel = channel();
        bind_three_d(&mut channel);
        let three_d_before = channel.three_d().clone();

        let flushes = non_incrementing_packet_on_subchannel(0, 0x1144 / 4, &[0, 1]);
        let flush_dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &flushes.packets()[0],
        )
        .unwrap();
        assert_eq!(flush_dispatch.methods().len(), 2);
        assert_eq!(flush_dispatch.synchronization_operations().len(), 2);
        assert!(flush_dispatch.operations().is_empty());
        for (index, expected) in [false, true].into_iter().enumerate() {
            assert_eq!(
                flush_dispatch.methods()[index].metadata().method_name(),
                "FLUSH_PENDING_WRITES"
            );
            let operation = &flush_dispatch.synchronization_operations()[index];
            assert!(matches!(
                operation.trigger(),
                MaxwellThreeDSynchronizationTrigger::FlushPendingWrites { request, source }
                    if request.sm_does_global_store() == expected
                        && source.argument() == u32::from(expected)
                        && source.method() == GpuMethodId(0x1144)
            ));
            assert_eq!(operation.state(), &three_d_before);
            assert_eq!(
                lower_maxwell_three_d_synchronization(operation, None),
                Ok(MaxwellThreeDSynchronizationPlan::FlushPendingWrites {
                    request: MaxwellThreeDFlushPendingWrites::new(expected),
                })
            );
        }

        let increments =
            non_incrementing_packet_on_subchannel(0, 0x02c8 / 4, &[1, (1 << 20) | (1 << 16) | 1]);
        let increment_dispatch = dispatch_maxwell_engine_packet(
            &mut channel,
            FrontendSubmissionId::new(3),
            &increments.packets()[0],
        )
        .unwrap();
        assert_eq!(increment_dispatch.synchronization_operations().len(), 2);
        let stream_out = &increment_dispatch.synchronization_operations()[0];
        assert!(matches!(
            stream_out.trigger(),
            MaxwellThreeDSynchronizationTrigger::IncrementSyncpoint { request, source }
                if request.syncpoint() == GuestSyncpointId::new(1)
                    && !request.clean_l2()
                    && request.condition() == MaxwellThreeDSyncpointCondition::StreamOutWritesDone
                    && source.method() == GpuMethodId(0x02c8)
        ));
        let rop = &increment_dispatch.synchronization_operations()[1];
        assert!(matches!(
            rop.trigger(),
            MaxwellThreeDSynchronizationTrigger::IncrementSyncpoint { request, .. }
                if request.syncpoint() == GuestSyncpointId::new(1)
                    && request.clean_l2()
                    && request.condition() == MaxwellThreeDSyncpointCondition::RopWritesDone
        ));

        let owner = TimelineOwnerId::new(9);
        let mut timeline = GuestTimeline::new(
            GuestSyncpointId::new(1),
            TimelineInstanceId::new(4),
            owner,
            GuestSyncpointValue::new(0),
        );
        let reservation = timeline.reserve(owner, 1).unwrap();
        assert!(matches!(
            lower_maxwell_three_d_synchronization(rop, Some(&reservation)),
            Ok(MaxwellThreeDSynchronizationPlan::IncrementSyncpoint {
                request,
                completion,
            }) if request.clean_l2()
                && request.condition() == MaxwellThreeDSyncpointCondition::RopWritesDone
                && completion == reservation.point()
        ));
        assert!(matches!(
            lower_maxwell_three_d_synchronization(rop, None),
            Err(MaxwellThreeDSynchronizationError::MissingCompletionReservation {
                requested,
                ..
            }) if requested == GuestSyncpointId::new(1)
        ));

        let mut wrong_timeline = GuestTimeline::new(
            GuestSyncpointId::new(2),
            TimelineInstanceId::new(5),
            owner,
            GuestSyncpointValue::new(0),
        );
        let wrong_reservation = wrong_timeline.reserve(owner, 1).unwrap();
        assert!(matches!(
            lower_maxwell_three_d_synchronization(rop, Some(&wrong_reservation)),
            Err(MaxwellThreeDSynchronizationError::WrongCompletionSyncpoint {
                requested,
                reserved,
                ..
            }) if requested == GuestSyncpointId::new(1)
                && reserved == GuestSyncpointId::new(2)
        ));
        assert_eq!(channel.three_d(), &three_d_before);
    }

    #[test]
    fn three_d_completion_reserved_bits_reject_the_whole_packet_atomically() {
        let mut channel = channel();
        bind_three_d(&mut channel);

        for (method, valid, invalid, mask) in [
            (0x1144, 0, 2, 0x0000_0001),
            (0x02c8, 0x0011_0001, 0x0011_1001, 0x0011_0fff),
        ] {
            let frontend_before = channel.frontend();
            let three_d_before = channel.three_d().clone();
            let packet = non_incrementing_packet_on_subchannel(0, method / 4, &[valid, invalid]);
            assert!(matches!(
                dispatch_maxwell_engine_packet(
                    &mut channel,
                    FrontendSubmissionId::new(3),
                    &packet.packets()[0]
                ),
                Err(MaxwellEngineDispatchError::InvalidMethodValue {
                    source,
                    defined_mask,
                    ..
                }) if source.argument() == invalid && defined_mask == mask
            ));
            assert_eq!(channel.frontend(), frontend_before);
            assert_eq!(channel.three_d(), &three_d_before);
        }
    }

    #[test]
    fn known_compute_class_distinguishes_missing_method_coverage() {
        let mut channel = channel();
        bind_compute(&mut channel);
        let method = packet_on_subchannel(1, 0x100 / 4, 0);
        let error = preflight_maxwell_engine_packet(
            &channel,
            FrontendSubmissionId::new(3),
            &method.packets()[0],
        )
        .unwrap_err();
        assert!(matches!(
            error,
            MaxwellEngineDispatchError::UnknownMethod {
                class_name: "MAXWELL_COMPUTE_B",
                ..
            }
        ));
    }
}
