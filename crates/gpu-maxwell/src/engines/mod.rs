//! Declarative Maxwell class-method routing.
//!
//! This layer applies decoded class methods to channel-owned state in stream
//! order and returns the first typed fatal boundary. It contains no Horizon
//! ABI, guest mappings, scheduler state, or host-backend objects.

mod compute;
mod dma_copy;
mod inline_to_memory;
mod spa;
mod threed;
pub(crate) use threed::lower_maxwell_three_d_operation_into_cache;
mod twod;

pub use spa::MaxwellSpaVersion;

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
    MaxwellComputeSpaVersion, MaxwellComputeState, MaxwellComputeSynchronizationPlan,
    MaxwellComputeTriggeredOperation, MaxwellShaderCacheInvalidation,
    lower_maxwell_compute_synchronization,
};
pub use dma_copy::{
    MaxwellDmaCopyComponentSource, MaxwellDmaCopyError, MaxwellDmaCopyMemoryLayout,
    MaxwellDmaCopyOperation, MaxwellDmaCopyRegister, MaxwellDmaCopyRegisterName,
    MaxwellDmaCopyRemap, MaxwellDmaCopyState,
};
pub use inline_to_memory::{
    MaxwellInlineToMemoryAddress, MaxwellInlineToMemoryLaunch,
    MaxwellInlineToMemoryPendingTransfer, MaxwellInlineToMemoryRegister,
    MaxwellInlineToMemorySemaphoreStructureSize, MaxwellInlineToMemoryState,
    MaxwellInlineToMemoryUpload,
};
#[cfg(test)]
use threed::resolve_maxwell_three_d_resources_for_roles_with_staged_writes;
pub(crate) use threed::resolve_maxwell_three_d_resources_for_roles_with_staged_writes_and_cache;
pub use threed::{
    MAXWELL_BIND_GROUP_COUNT, MAXWELL_COLOR_TARGET_COUNT, MAXWELL_CONSTANT_BUFFER_SLOT_COUNT,
    MAXWELL_PIPELINE_SHADER_COUNT, MAXWELL_POLYGON_STIPPLE_PATTERN_WORD_COUNT,
    MAXWELL_SAMPLE_LOCATION_GROUP_COUNT, MAXWELL_SAMPLE_LOCATIONS_PER_GROUP, MAXWELL_SCISSOR_COUNT,
    MAXWELL_TESSELLATION_LOD_COUNT, MAXWELL_THREE_D_MME_CAPTURED_INSTRUCTION_WORDS,
    MAXWELL_THREE_D_MME_CAPTURED_START_ADDRESSES, MAXWELL_THREE_D_MME_EMITTED_METHOD_LIMIT,
    MAXWELL_THREE_D_MME_EXECUTION_INSTRUCTION_LIMIT, MAXWELL_THREE_D_MME_SHADOW_SCRATCH_COUNT,
    MAXWELL_THREE_D_PRIMITIVE_AREA_MAX, MAXWELL_THREE_D_SHADER_LOCAL_MEMORY_PER_WARP_SIZE_MAX,
    MAXWELL_THREE_D_SM_TIMEOUT_COUNTER_BIT_MAX, MAXWELL_VERTEX_ATTRIBUTE_COUNT,
    MAXWELL_VERTEX_STREAM_COUNT, MAXWELL_VIEWPORT_COUNT, MAXWELL_WINDOW_CLIP_COUNT,
    MaxwellThreeDAliasedLineWidthEnable, MaxwellThreeDAlphaFraction,
    MaxwellThreeDAlphaToCoverageOverride, MaxwellThreeDAntiAliasedLineEnable,
    MaxwellThreeDApiMandatedEarlyZ, MaxwellThreeDAttachmentReadiness,
    MaxwellThreeDAttributeDefaultVector, MaxwellThreeDAttributeDefaults,
    MaxwellThreeDAttributePointSize, MaxwellThreeDBalancedPrimitiveWorkload, MaxwellThreeDBegin,
    MaxwellThreeDBeginInstance, MaxwellThreeDBindGroupState, MaxwellThreeDBlendControlState,
    MaxwellThreeDBlendEnableCommon, MaxwellThreeDBlendFactor,
    MaxwellThreeDBlendFloatPixelKillEnable, MaxwellThreeDBlendOp,
    MaxwellThreeDBlendPerFormatEnable, MaxwellThreeDBlendZeroTimesAnythingIsZero,
    MaxwellThreeDClearState, MaxwellThreeDClearSurface, MaxwellThreeDClearSurfaceControl,
    MaxwellThreeDClipIdTestEnable, MaxwellThreeDColorCompressionMode, MaxwellThreeDColorMask,
    MaxwellThreeDColorReductionFp16Threshold, MaxwellThreeDColorReductionSrgb8Threshold,
    MaxwellThreeDColorReductionState, MaxwellThreeDColorReductionThresholdsEnable,
    MaxwellThreeDColorReductionThresholdsFp16, MaxwellThreeDColorReductionThresholdsSrgb8,
    MaxwellThreeDColorReductionThresholdsUnorm8, MaxwellThreeDColorReductionThresholdsUnorm10,
    MaxwellThreeDColorReductionThresholdsUnorm16, MaxwellThreeDColorTargetFormat,
    MaxwellThreeDColorTargetSelection, MaxwellThreeDColorTargetState, MaxwellThreeDCompareOp,
    MaxwellThreeDCompressionThreshold, MaxwellThreeDConditionalLoadConstantBuffer,
    MaxwellThreeDConservativeRasterEnable, MaxwellThreeDConstantBufferBinding,
    MaxwellThreeDConstantBufferLoadState, MaxwellThreeDConstantBufferSelectorState,
    MaxwellThreeDConstantColorComponent, MaxwellThreeDConstantColorRenderingState,
    MaxwellThreeDConstantColorValue, MaxwellThreeDCounterState, MaxwellThreeDCoverageState,
    MaxwellThreeDCoverageToColor, MaxwellThreeDCsaaEnable, MaxwellThreeDCullFace,
    MaxwellThreeDDecompressSurface, MaxwellThreeDDepthStencilFormat,
    MaxwellThreeDDepthStencilTargetState, MaxwellThreeDDepthTargetCount,
    MaxwellThreeDDescriptorPoolState, MaxwellThreeDDirectlyAddressableMemory,
    MaxwellThreeDDirtySubresource, MaxwellThreeDDirtySubresources, MaxwellThreeDEdgeFlag,
    MaxwellThreeDFalconError, MaxwellThreeDFalconMaskedRegisterWrite, MaxwellThreeDFalconRegister,
    MaxwellThreeDFalconRegisterAddress, MaxwellThreeDFalconState, MaxwellThreeDFillViaTriangleMode,
    MaxwellThreeDFixedFunctionRegister, MaxwellThreeDFixedFunctionState,
    MaxwellThreeDFixedFunctionValue, MaxwellThreeDFlushPendingWrites, MaxwellThreeDFrontFace,
    MaxwellThreeDGuestImageFormat, MaxwellThreeDHybridAntiAliasCentroid,
    MaxwellThreeDHybridAntiAliasControl, MaxwellThreeDImageKind, MaxwellThreeDImageLayout,
    MaxwellThreeDIndexBufferState, MaxwellThreeDIndexElementSize,
    MaxwellThreeDInlineConstantBufferUpload, MaxwellThreeDInlineToMemoryCompletion,
    MaxwellThreeDInlineToMemoryLaunch, MaxwellThreeDInlineToMemoryLayout,
    MaxwellThreeDInlineToMemoryState, MaxwellThreeDInstrumentationState,
    MaxwellThreeDInstrumentationValue, MaxwellThreeDIteratedBlend,
    MaxwellThreeDIteratedBlendPassCount, MaxwellThreeDL2CacheEvictionPolicy,
    MaxwellThreeDL2CacheState, MaxwellThreeDLineState, MaxwellThreeDLineStippleParameters,
    MaxwellThreeDLogicOp, MaxwellThreeDLoweredWork, MaxwellThreeDLoweringCache,
    MaxwellThreeDLoweringError, MaxwellThreeDLoweringPlan, MaxwellThreeDMappingReference,
    MaxwellThreeDMmeExecutionError, MaxwellThreeDMmeInstruction, MaxwellThreeDMmeLoadError,
    MaxwellThreeDMmeRam, MaxwellThreeDMmeRamAddress, MaxwellThreeDMmeShadowRamControl,
    MaxwellThreeDMmeShadowRamError, MaxwellThreeDMmeShadowScratchIndex, MaxwellThreeDMmeState,
    MaxwellThreeDMutableMethodControl, MaxwellThreeDOperationTrigger, MaxwellThreeDPatchSize,
    MaxwellThreeDPipelineBindingState, MaxwellThreeDPixelShaderClampRange,
    MaxwellThreeDPixelShaderInterlockControl, MaxwellThreeDPixelShaderInterlockFragmentOrder,
    MaxwellThreeDPixelShaderInterlockMode, MaxwellThreeDPixelShaderInterlockTileSize,
    MaxwellThreeDPixelShaderSaturate, MaxwellThreeDPointCenterMode, MaxwellThreeDPointSize,
    MaxwellThreeDPointSpriteOrigin, MaxwellThreeDPointSpriteRMode, MaxwellThreeDPointSpriteSelect,
    MaxwellThreeDPolygonClipGeneratedEdge, MaxwellThreeDPolygonMode,
    MaxwellThreeDPostZPixelShaderImask, MaxwellThreeDPreservedImageLayout,
    MaxwellThreeDPrimitiveCircularBufferThrottle, MaxwellThreeDPrimitiveState,
    MaxwellThreeDPrimitiveTopology, MaxwellThreeDProgramRegionState, MaxwellThreeDProvokingVertex,
    MaxwellThreeDPsOutputSampleMaskUsage, MaxwellThreeDRasterBoundingBox,
    MaxwellThreeDRasterBoundingBoxMode, MaxwellThreeDRasterState, MaxwellThreeDRawValue,
    MaxwellThreeDRectangle, MaxwellThreeDRegister, MaxwellThreeDRegisterOrigin,
    MaxwellThreeDRenderEnableMode, MaxwellThreeDRenderEnableState,
    MaxwellThreeDRenderTargetIndexOffset, MaxwellThreeDRenderTargetLayer,
    MaxwellThreeDRenderTargetLayerControl, MaxwellThreeDRenderTargetState,
    MaxwellThreeDReportSemaphoreControl, MaxwellThreeDReportSemaphoreOperation,
    MaxwellThreeDReportSemaphorePipelineLocation, MaxwellThreeDReportSemaphoreRelease,
    MaxwellThreeDReportSemaphoreState, MaxwellThreeDReportSemaphoreStructureSize,
    MaxwellThreeDResolvedBuffer, MaxwellThreeDResolvedImage, MaxwellThreeDResolvedResource,
    MaxwellThreeDResolvedResources, MaxwellThreeDResolvedSampler, MaxwellThreeDResourceAccess,
    MaxwellThreeDResourceAlias, MaxwellThreeDResourceError, MaxwellThreeDResourceRole,
    MaxwellThreeDRopL2CacheRequest, MaxwellThreeDSampleLocation, MaxwellThreeDSampleLocationGroup,
    MaxwellThreeDSampleMode, MaxwellThreeDSamplerBindingMode, MaxwellThreeDScissorState,
    MaxwellThreeDSeparateFragmentData, MaxwellThreeDShadeMode, MaxwellThreeDShaderBindingState,
    MaxwellThreeDShaderCacheInvalidation, MaxwellThreeDShaderExceptionsEnable,
    MaxwellThreeDShaderExecutionState, MaxwellThreeDShaderLocalMemoryPerWarpSize,
    MaxwellThreeDShaderLocalMemoryState, MaxwellThreeDShaderResourceUse, MaxwellThreeDShaderStage,
    MaxwellThreeDShaderWatermarkRange, MaxwellThreeDShaderWatermarkTarget,
    MaxwellThreeDSmTimeoutCounterBit, MaxwellThreeDState, MaxwellThreeDStencilOp,
    MaxwellThreeDSubtilingPerfKnobA, MaxwellThreeDSubtilingPerfKnobB, MaxwellThreeDSurfaceClipAxis,
    MaxwellThreeDSynchronizationError, MaxwellThreeDSynchronizationOperation,
    MaxwellThreeDSynchronizationPlan, MaxwellThreeDSynchronizationTrigger,
    MaxwellThreeDSyncpointCondition, MaxwellThreeDSyncpointIncrement,
    MaxwellThreeDSystemMemoryVolatile, MaxwellThreeDTessellationLod,
    MaxwellThreeDTextureCacheInvalidation, MaxwellThreeDTextureCacheLines,
    MaxwellThreeDTextureCacheTarget, MaxwellThreeDTiledCacheFlushMode,
    MaxwellThreeDTiledCacheState, MaxwellThreeDTiledCacheTileSize,
    MaxwellThreeDTiledCacheUnknownConfig, MaxwellThreeDTirControl, MaxwellThreeDTirMode,
    MaxwellThreeDTirModulationComponentSelect, MaxwellThreeDTirModulationFunction,
    MaxwellThreeDTranslatedShader, MaxwellThreeDTranslatedShaders,
    MaxwellThreeDUnnegotiatedLoweringPlan, MaxwellThreeDUnorm8, MaxwellThreeDUnresolvedAddress,
    MaxwellThreeDVafL2CacheControl, MaxwellThreeDVertexArrayPrimitiveRestartEnable,
    MaxwellThreeDVertexAssemblyState, MaxwellThreeDVertexAttributeFormat,
    MaxwellThreeDVertexComponentWidths, MaxwellThreeDVertexIdUsesArrayStart,
    MaxwellThreeDVertexInputState, MaxwellThreeDVertexNumericalType,
    MaxwellThreeDVertexStreamFormat, MaxwellThreeDVertexStreamState,
    MaxwellThreeDVertexStreamSubstituteState, MaxwellThreeDViewportClipControl,
    MaxwellThreeDViewportCoordinateSwizzle, MaxwellThreeDViewportPixelCenter,
    MaxwellThreeDViewportScaleOffsetEnable, MaxwellThreeDViewportState,
    MaxwellThreeDViewportSwizzleComponent, MaxwellThreeDViewportTransformState,
    MaxwellThreeDViewportZClipRange, MaxwellThreeDVisibleCallLimit, MaxwellThreeDWindowClipState,
    MaxwellThreeDWindowClipType, MaxwellThreeDZCompressionMode, MaxwellThreeDZCullBounds,
    MaxwellThreeDZCullCriterion, MaxwellThreeDZCullEnable, MaxwellThreeDZCullRegionId,
    MaxwellThreeDZCullState, MaxwellThreeDZCullStatsEnable, MaxwellThreeDZCullStencilFunction,
    MaxwellThreeDZPassPixelCountEnable, lower_maxwell_three_d_synchronization,
    preflight_maxwell_three_d_operation, preflight_maxwell_three_d_operation_unnegotiated,
    resolve_maxwell_three_d_resources, resolve_maxwell_three_d_resources_for_roles,
};
pub use twod::{
    MAXWELL_TWO_D_CORRAL_SIZE_MAX, MAXWELL_TWO_D_NOTIFY_ADDRESS_UPPER_MAX, MaxwellTwoDBeta1,
    MaxwellTwoDBeta4, MaxwellTwoDBetaState, MaxwellTwoDClipEnable, MaxwellTwoDColorKeyEnable,
    MaxwellTwoDNotifyAddressLower, MaxwellTwoDNotifyAddressUpper, MaxwellTwoDNotifyState,
    MaxwellTwoDOperation, MaxwellTwoDPixelsFromMemoryCorralSize,
    MaxwellTwoDPixelsFromMemorySafeOverlap, MaxwellTwoDPixelsFromMemoryState,
    MaxwellTwoDProcessingClusters, MaxwellTwoDRegister, MaxwellTwoDRegisterOrigin,
    MaxwellTwoDRenderEnableMode, MaxwellTwoDRenderEnableState, MaxwellTwoDState,
};

use std::{
    fmt::{Display, Formatter},
    sync::Arc,
};

use nixe_gpu::{FrontendSubmissionId, GpuClassId, GpuMethodId};

use crate::{
    MaxwellAamVersionRange, MaxwellDecodedPacket, MaxwellDecodedPushbuffer, MaxwellGpuChannel,
    MaxwellHostMemoryOperation, MaxwellHostMethod, MaxwellMethodDispatch,
    MaxwellMethodDispatchError, MaxwellMethodDispatchKind, MaxwellMethodSource,
    MaxwellPacketDispatch, MaxwellShaderProgramHeaderVersionRange, dispatch_maxwell_packet,
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

/// One execution-relevant effect in exact pushbuffer order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MaxwellEngineOperation {
    HostSynchronization(MaxwellHostSynchronizationOperation),
    ComputeInlineToMemory(MaxwellComputeInlineToMemoryUpload),
    InlineToMemory(MaxwellInlineToMemoryUpload),
    DmaCopy(MaxwellDmaCopyOperation),
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

/// One named class method applied during direct packet dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellEngineMethodDispatch {
    method: MaxwellMethodDispatch,
    metadata: MaxwellEngineMethodMetadata,
}

impl MaxwellEngineMethodDispatch {
    pub(crate) const fn new(
        method: MaxwellMethodDispatch,
        metadata: MaxwellEngineMethodMetadata,
    ) -> Self {
        Self { method, metadata }
    }

    #[must_use]
    pub const fn method(self) -> MaxwellMethodDispatch {
        self.method
    }

    #[must_use]
    pub const fn metadata(self) -> MaxwellEngineMethodMetadata {
        self.metadata
    }
}

struct AppliedMethod {
    dispatch: MaxwellEngineMethodDispatch,
    operation: Option<MaxwellEngineOperation>,
}

impl AppliedMethod {
    const fn new(
        method: MaxwellMethodDispatch,
        metadata: MaxwellEngineMethodMetadata,
        operation: Option<MaxwellEngineOperation>,
    ) -> Self {
        Self {
            dispatch: MaxwellEngineMethodDispatch::new(method, metadata),
            operation,
        }
    }
}

/// Validated packet methods and ordered operations after immediate state commit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaxwellEnginePacketDispatch {
    binding: MaxwellPacketDispatch,
    methods: Box<[MaxwellEngineMethodDispatch]>,
    ordered_operations: Box<[MaxwellEngineOperation]>,
}

/// One execution trigger paired with the exact channel-state snapshot at that method.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaxwellThreeDTriggeredOperation {
    trigger: MaxwellThreeDOperationTrigger,
    state: Arc<MaxwellThreeDState>,
}

impl MaxwellThreeDTriggeredOperation {
    #[must_use]
    pub const fn trigger(&self) -> MaxwellThreeDOperationTrigger {
        self.trigger
    }

    #[must_use]
    pub fn state(&self) -> &MaxwellThreeDState {
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
    InvalidInlineToMemoryMethodEncoding {
        source: MaxwellMethodSource,
        method_name: &'static str,
        reason: &'static str,
    },
    InvalidDmaCopyMethodEncoding {
        source: MaxwellMethodSource,
        method_name: &'static str,
        reason: &'static str,
    },
    FalconFirmware {
        source: MaxwellMethodSource,
        error: MaxwellThreeDFalconError,
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
    MmeShadowRam {
        source: MaxwellMethodSource,
        error: MaxwellThreeDMmeShadowRamError,
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
            Self::InvalidInlineToMemoryMethodEncoding {
                source,
                method_name,
                reason,
            } => write!(
                formatter,
                "Maxwell method has an invalid verified encoding: {source} class-name=MAXWELL_INLINE_TO_MEMORY_A method-name={method_name} reason={reason}"
            ),
            Self::InvalidDmaCopyMethodEncoding {
                source,
                method_name,
                reason,
            } => write!(
                formatter,
                "Maxwell method has an invalid verified encoding: {source} class-name=MAXWELL_DMA_COPY_A method-name={method_name} reason={reason}"
            ),
            Self::FalconFirmware { source, error } => write!(
                formatter,
                "MAXWELL_B Falcon firmware call failed: {source} error={error:?}"
            ),
            Self::MmeRamLoad { source, ram, error } => write!(
                formatter,
                "MAXWELL_B MME RAM load exceeds implemented host coverage: {source} ram={ram:?} error={error:?}"
            ),
            Self::MmeExecution { source, error } => write!(
                formatter,
                "MAXWELL_B MME execution failed: {source} error={error:?}"
            ),
            Self::MmeShadowRam { source, error } => write!(
                formatter,
                "MAXWELL_B MME shadow-RAM transition failed: {source} error={error:?}"
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

/// Dispatches one packet directly against channel-owned engine state.
///
/// Unsupported semantics terminate the guest process, so successfully applied
/// method prefixes are intentionally not rolled back.
pub fn dispatch_maxwell_engine_packet(
    channel: &mut MaxwellGpuChannel,
    submission: FrontendSubmissionId,
    packet: &MaxwellDecodedPacket,
) -> Result<MaxwellEnginePacketDispatch, MaxwellEngineDispatchError> {
    let binding = dispatch_maxwell_packet(channel, submission, packet)?;
    let mut methods = Vec::new();
    let mut ordered_operations = Vec::new();
    methods
        .try_reserve_exact(binding.methods().len())
        .map_err(|_| MaxwellEngineDispatchError::ResourceExhausted)?;

    let mut method_index = 0;
    while method_index < binding.methods().len() {
        let method = binding.methods()[method_index];
        if let MaxwellMethodDispatchKind::HostMethod(host) = method.kind() {
            let applied = preflight_host_method(method, host);
            if let Some(operation) = applied.operation {
                ordered_operations
                    .try_reserve(1)
                    .map_err(|_| MaxwellEngineDispatchError::ResourceExhausted)?;
                ordered_operations.push(operation);
            }
            methods.push(applied.dispatch);
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
                channel.three_d_mut(),
            )?;
            methods
                .try_reserve(macro_preflight.methods.len())
                .map_err(|_| MaxwellEngineDispatchError::ResourceExhausted)?;
            ordered_operations
                .try_reserve(macro_preflight.ordered_operations.len())
                .map_err(|_| MaxwellEngineDispatchError::ResourceExhausted)?;
            methods.extend(macro_preflight.methods);
            ordered_operations.extend(macro_preflight.ordered_operations);
            method_index = end;
            continue;
        }
        if method.kind() == MaxwellMethodDispatchKind::ClassMethod {
            let applied = dispatch_class_method(channel, method)?;
            methods.push(applied.dispatch);
            if let Some(operation) = applied.operation {
                ordered_operations
                    .try_reserve(1)
                    .map_err(|_| MaxwellEngineDispatchError::ResourceExhausted)?;
                ordered_operations.push(operation);
            }
        }
        method_index += 1;
    }
    // Maxwell register programming is not transactional across pushbuffer
    // packets. Related address, limit, format, and selector fields may form an
    // inconsistent intermediate snapshot while the guest moves from one valid
    // configuration to the next. Keep method-local encoding checks here, but
    // defer relational validation until a draw or clear consumes its immutable
    // state snapshot in the neutral lowering preflight.
    Ok(MaxwellEnginePacketDispatch {
        binding,
        methods: methods.into_boxed_slice(),
        ordered_operations: ordered_operations.into_boxed_slice(),
    })
}

fn preflight_host_method(method: MaxwellMethodDispatch, host: MaxwellHostMethod) -> AppliedMethod {
    match host {
        MaxwellHostMethod::LegacyMemOpA { .. } => AppliedMethod::new(
            method,
            MaxwellEngineMethodMetadata::new(
                method.class(),
                "MAXWELL_CHANNEL_GPFIFO_A(legacy-host-compatibility)",
                method.source().method(),
                "MEM_OP_A",
            ),
            None,
        ),
        MaxwellHostMethod::LegacyMemOpB(operation) => AppliedMethod::new(
            method,
            MaxwellEngineMethodMetadata::new(
                method.class(),
                "MAXWELL_CHANNEL_GPFIFO_A(legacy-host-compatibility)",
                method.source().method(),
                "MEM_OP_B",
            ),
            Some(MaxwellEngineOperation::HostSynchronization(
                MaxwellHostSynchronizationOperation {
                    source: method.source(),
                    operation,
                },
            )),
        ),
    }
}

/// Dispatches packets and methods in stream order.
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

fn dispatch_class_method(
    channel: &mut MaxwellGpuChannel,
    method: MaxwellMethodDispatch,
) -> Result<AppliedMethod, MaxwellEngineDispatchError> {
    let classes = channel.profile().classes();
    let profile = channel.profile();
    let class = method.class();
    if class == compute::CLASS {
        return compute::preflight(class, method, channel.compute_mut());
    }
    if class == dma_copy::CLASS {
        return dma_copy::preflight(method, channel.dma_copy_mut());
    }
    if class == inline_to_memory::CLASS {
        return inline_to_memory::preflight(method, channel.inline_to_memory_mut());
    }
    if class == threed::CLASS {
        return threed::preflight(profile, method, channel.three_d_mut());
    }
    if class == twod::CLASS {
        return twod::preflight(method, channel.two_d_mut());
    }

    let class_name = if class == classes.gpfifo() {
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
