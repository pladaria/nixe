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
pub(crate) use threed::MaxwellThreeDFrontendState;
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
    MaxwellThreeDLoweringError, MaxwellThreeDMappingReference, MaxwellThreeDMmeExecutionError,
    MaxwellThreeDMmeInstruction, MaxwellThreeDMmeLoadError, MaxwellThreeDMmeRam,
    MaxwellThreeDMmeRamAddress, MaxwellThreeDMmeShadowRamControl, MaxwellThreeDMmeShadowRamError,
    MaxwellThreeDMmeShadowScratchIndex, MaxwellThreeDMmeState, MaxwellThreeDMutableMethodControl,
    MaxwellThreeDOperationTrigger, MaxwellThreeDPatchSize, MaxwellThreeDPipelineBindingState,
    MaxwellThreeDPixelShaderClampRange, MaxwellThreeDPixelShaderInterlockControl,
    MaxwellThreeDPixelShaderInterlockFragmentOrder, MaxwellThreeDPixelShaderInterlockMode,
    MaxwellThreeDPixelShaderInterlockTileSize, MaxwellThreeDPixelShaderSaturate,
    MaxwellThreeDPointCenterMode, MaxwellThreeDPointSize, MaxwellThreeDPointSpriteOrigin,
    MaxwellThreeDPointSpriteRMode, MaxwellThreeDPointSpriteSelect,
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
    MaxwellThreeDSynchronizationError, MaxwellThreeDSynchronizationPlan,
    MaxwellThreeDSynchronizationTrigger, MaxwellThreeDSyncpointCondition,
    MaxwellThreeDSyncpointIncrement, MaxwellThreeDSystemMemoryVolatile,
    MaxwellThreeDTessellationLod, MaxwellThreeDTextureCacheInvalidation,
    MaxwellThreeDTextureCacheLines, MaxwellThreeDTextureCacheTarget,
    MaxwellThreeDTiledCacheFlushMode, MaxwellThreeDTiledCacheState,
    MaxwellThreeDTiledCacheTileSize, MaxwellThreeDTiledCacheUnknownConfig, MaxwellThreeDTirControl,
    MaxwellThreeDTirMode, MaxwellThreeDTirModulationComponentSelect,
    MaxwellThreeDTirModulationFunction, MaxwellThreeDTranslatedShader,
    MaxwellThreeDTranslatedShaders, MaxwellThreeDUnorm8, MaxwellThreeDUnresolvedAddress,
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
    MaxwellThreeDZPassPixelCountEnable, lower_maxwell_three_d_operation,
    lower_maxwell_three_d_synchronization, resolve_maxwell_three_d_resources,
    resolve_maxwell_three_d_resources_for_roles,
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

use std::fmt::{Display, Formatter};

#[cfg(test)]
use std::sync::Arc;

use nixe_gpu::{FrontendSubmissionId, GpuClassId, GpuMethodId};

use crate::pushbuffer::dispatch::{MaxwellMethodStreamError, stream_maxwell_packet_methods};
use crate::{
    MaxwellAamVersionRange, MaxwellDecodedPacket, MaxwellGpuChannel, MaxwellHostMemoryOperation,
    MaxwellHostMethod, MaxwellMethodDispatch, MaxwellMethodDispatchError,
    MaxwellMethodDispatchKind, MaxwellMethodSource, MaxwellShaderProgramHeaderVersionRange,
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
    #[cfg(test)]
    pub const fn class(self) -> GpuClassId {
        self.class
    }

    #[must_use]
    #[cfg(test)]
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

/// One execution-relevant effect produced while applying a method.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum PendingEngineOperation {
    HostSynchronization(MaxwellHostSynchronizationOperation),
    ComputeInlineToMemory(MaxwellComputeInlineToMemoryUpload),
    InlineToMemory(MaxwellInlineToMemoryUpload),
    DmaCopy(MaxwellDmaCopyOperation),
    ComputeSynchronization(Box<MaxwellComputeTriggeredOperation>),
    ThreeDInlineConstantBuffer(MaxwellThreeDInlineConstantBufferUpload),
    ThreeD(MaxwellThreeDOperationTrigger),
    ThreeDSynchronization(MaxwellThreeDSynchronizationTrigger),
}

/// One execution effect paired with live state for immediate consumption.
pub(crate) struct MaxwellEngineEvent<'a> {
    pub(crate) operation: PendingEngineOperation,
    pub(crate) three_d: &'a MaxwellThreeDState,
}

/// One validated host cache operation at its exact pushbuffer source.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct MaxwellHostSynchronizationOperation {
    operation: MaxwellHostMemoryOperation,
}

impl MaxwellHostSynchronizationOperation {
    #[must_use]
    pub const fn operation(self) -> MaxwellHostMemoryOperation {
        self.operation
    }
}

/// One named class method applied during direct packet dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct MaxwellEngineMethodDispatch {
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
    #[cfg(test)]
    pub const fn method(self) -> MaxwellMethodDispatch {
        self.method
    }

    #[must_use]
    #[cfg(test)]
    pub const fn metadata(self) -> MaxwellEngineMethodMetadata {
        self.metadata
    }
}

struct AppliedMethod {
    dispatch: MaxwellEngineMethodDispatch,
    operation: Option<PendingEngineOperation>,
}

impl AppliedMethod {
    const fn new(
        method: MaxwellMethodDispatch,
        metadata: MaxwellEngineMethodMetadata,
        operation: Option<PendingEngineOperation>,
    ) -> Self {
        Self {
            dispatch: MaxwellEngineMethodDispatch::new(method, metadata),
            operation,
        }
    }
}

/// Test observation of validated methods and owned operation snapshots.
#[cfg(test)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaxwellEnginePacketDispatch {
    methods: Box<[MaxwellEngineMethodDispatch]>,
    ordered_operations: Box<[MaxwellEngineOperation]>,
}

#[cfg(test)]
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

/// Test snapshot of one 3D trigger.
#[cfg(test)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaxwellThreeDTriggeredOperation {
    trigger: MaxwellThreeDOperationTrigger,
    state: Arc<MaxwellThreeDState>,
}

#[cfg(test)]
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

#[cfg(test)]
impl MaxwellEnginePacketDispatch {
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

/// Failure while streaming one packet into its immediate consumer.
pub(crate) enum MaxwellEngineStreamError<E> {
    Dispatch(Box<MaxwellEngineDispatchError>),
    Consumer(E),
}

impl<E> MaxwellEngineStreamError<E> {
    fn dispatch(error: MaxwellEngineDispatchError) -> Self {
        Self::Dispatch(Box::new(error))
    }
}

fn emit_pending_operation<E>(
    operation: PendingEngineOperation,
    three_d: &MaxwellThreeDState,
    consume: &mut impl for<'a> FnMut(MaxwellEngineEvent<'a>) -> Result<(), E>,
) -> Result<(), E> {
    consume(MaxwellEngineEvent { operation, three_d })
}

/// Applies one packet and streams sparse execution effects in exact method order.
///
/// Unsupported semantics terminate the guest process, so successfully applied
/// method prefixes are intentionally not rolled back. A 3D effect borrows the
/// live state only for the callback and therefore cannot survive a following
/// register mutation.
pub(crate) fn stream_maxwell_engine_packet<E>(
    channel: &mut MaxwellGpuChannel,
    submission: FrontendSubmissionId,
    packet: &MaxwellDecodedPacket,
    methods: Option<&mut Vec<MaxwellEngineMethodDispatch>>,
    mme_methods: &mut Vec<MaxwellMethodDispatch>,
    mme_parameters: &mut Vec<u32>,
    consume: &mut impl for<'a> FnMut(MaxwellEngineEvent<'a>) -> Result<(), E>,
) -> Result<(), MaxwellEngineStreamError<E>> {
    let mut methods = methods;
    stream_maxwell_packet_methods(channel, submission, packet, &mut |channel, method| {
        if !mme_methods.is_empty() {
            let first_method = mme_methods[0].source().method().0;
            let next_method = method.source().method().0;
            let continues = method.kind() == MaxwellMethodDispatchKind::ClassMethod
                && method.class() == threed::CLASS
                && (next_method == first_method || next_method == first_method + 4);
            if continues {
                mme_methods.try_reserve(1).map_err(|_| {
                    MaxwellEngineStreamError::dispatch(
                        MaxwellEngineDispatchError::ResourceExhausted,
                    )
                })?;
                mme_methods.push(method);
                return Ok(());
            }
            flush_mme_methods(channel, mme_methods, mme_parameters, &mut methods, consume)?;
        }
        if let MaxwellMethodDispatchKind::HostMethod(host) = method.kind() {
            let applied = preflight_host_method(method, host);
            if let Some(operation) = applied.operation {
                emit_pending_operation(operation, channel.three_d(), consume)
                    .map_err(MaxwellEngineStreamError::Consumer)?;
            }
            if let Some(methods) = methods.as_deref_mut() {
                methods.push(applied.dispatch);
            }
            return Ok(());
        }
        if method.kind() == MaxwellMethodDispatchKind::ClassMethod
            && method.class() == threed::CLASS
            && threed::is_mme_aperture(method.source().method())
        {
            let first_method = method.source().method().0;
            if (first_method - 0x3800) & 7 != 0 {
                return Err(MaxwellEngineStreamError::dispatch(
                    MaxwellEngineDispatchError::MmeExecution {
                        source: method.source(),
                        error: MaxwellThreeDMmeExecutionError::DataWithoutCall,
                    },
                ));
            }
            mme_methods.try_reserve(1).map_err(|_| {
                MaxwellEngineStreamError::dispatch(MaxwellEngineDispatchError::ResourceExhausted)
            })?;
            mme_methods.push(method);
            return Ok(());
        }
        if method.kind() == MaxwellMethodDispatchKind::ClassMethod {
            let applied = dispatch_class_method(channel, method)
                .map_err(MaxwellEngineStreamError::dispatch)?;
            if let Some(operation) = applied.operation {
                emit_pending_operation(operation, channel.three_d(), consume)
                    .map_err(MaxwellEngineStreamError::Consumer)?;
            }
            if let Some(methods) = methods.as_deref_mut() {
                methods.push(applied.dispatch);
            }
        }
        Ok(())
    })
    .map_err(|error| match error {
        MaxwellMethodStreamError::Dispatch(error) => {
            MaxwellEngineStreamError::dispatch(error.into())
        }
        MaxwellMethodStreamError::Consumer(error) => error,
    })?;
    flush_mme_methods(channel, mme_methods, mme_parameters, &mut methods, consume)?;
    // Maxwell register programming is not transactional across pushbuffer
    // packets. Related address, limit, format, and selector fields may form an
    // inconsistent intermediate snapshot while the guest moves from one valid
    // configuration to the next. Keep method-local encoding checks here, but
    // defer relational validation until a draw or clear consumes its immutable
    // state snapshot in the neutral lowering preflight.
    Ok(())
}

fn flush_mme_methods<E>(
    channel: &mut MaxwellGpuChannel,
    pending: &mut Vec<MaxwellMethodDispatch>,
    parameters: &mut Vec<u32>,
    methods: &mut Option<&mut Vec<MaxwellEngineMethodDispatch>>,
    consume: &mut impl for<'a> FnMut(MaxwellEngineEvent<'a>) -> Result<(), E>,
) -> Result<(), MaxwellEngineStreamError<E>> {
    if pending.is_empty() {
        return Ok(());
    }
    threed::preflight_mme_call(
        channel.profile(),
        pending,
        parameters,
        channel.three_d_mut(),
        methods.as_deref_mut(),
        &mut |operation, state| emit_pending_operation(operation, state, consume),
    )
    .map_err(|error| match error {
        threed::MaxwellThreeDMmePreflightError::Dispatch(error) => {
            MaxwellEngineStreamError::Dispatch(error)
        }
        threed::MaxwellThreeDMmePreflightError::Consumer(error) => {
            MaxwellEngineStreamError::Consumer(error)
        }
    })?;
    pending.clear();
    Ok(())
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
            Some(PendingEngineOperation::HostSynchronization(
                MaxwellHostSynchronizationOperation { operation },
            )),
        ),
    }
}

/// Test collector for method semantics and owned trigger observations.
#[cfg(test)]
pub fn dispatch_maxwell_engine_packet(
    channel: &mut MaxwellGpuChannel,
    submission: FrontendSubmissionId,
    packet: &MaxwellDecodedPacket,
) -> Result<MaxwellEnginePacketDispatch, MaxwellEngineDispatchError> {
    let mut methods = Vec::new();
    let mut ordered_operations = Vec::new();
    let mut mme_methods = Vec::new();
    let mut mme_parameters = Vec::new();
    stream_maxwell_engine_packet(
        channel,
        submission,
        packet,
        Some(&mut methods),
        &mut mme_methods,
        &mut mme_parameters,
        &mut |event| {
            let operation = match event.operation {
                PendingEngineOperation::HostSynchronization(operation) => {
                    MaxwellEngineOperation::HostSynchronization(operation)
                }
                PendingEngineOperation::ComputeInlineToMemory(upload) => {
                    MaxwellEngineOperation::ComputeInlineToMemory(upload)
                }
                PendingEngineOperation::InlineToMemory(upload) => {
                    MaxwellEngineOperation::InlineToMemory(upload)
                }
                PendingEngineOperation::DmaCopy(operation) => {
                    MaxwellEngineOperation::DmaCopy(operation)
                }
                PendingEngineOperation::ComputeSynchronization(operation) => {
                    MaxwellEngineOperation::ComputeSynchronization(operation)
                }
                PendingEngineOperation::ThreeDInlineConstantBuffer(upload) => {
                    MaxwellEngineOperation::ThreeDInlineConstantBuffer(upload)
                }
                PendingEngineOperation::ThreeD(trigger) => {
                    MaxwellEngineOperation::ThreeD(Box::new(MaxwellThreeDTriggeredOperation {
                        trigger,
                        state: Arc::new(event.three_d.clone()),
                    }))
                }
                PendingEngineOperation::ThreeDSynchronization(trigger) => {
                    MaxwellEngineOperation::ThreeDSynchronization(Box::new(
                        MaxwellThreeDSynchronizationOperation {
                            trigger,
                            state: Arc::new(event.three_d.clone()),
                        },
                    ))
                }
            };
            ordered_operations.push(operation);
            Ok::<(), std::convert::Infallible>(())
        },
    )
    .map_err(|error| match error {
        MaxwellEngineStreamError::Dispatch(error) => *error,
        MaxwellEngineStreamError::Consumer(never) => match never {},
    })?;
    Ok(MaxwellEnginePacketDispatch {
        methods: methods.into_boxed_slice(),
        ordered_operations: ordered_operations.into_boxed_slice(),
    })
}

#[cfg(test)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaxwellThreeDSynchronizationOperation {
    trigger: MaxwellThreeDSynchronizationTrigger,
    state: Arc<MaxwellThreeDState>,
}

#[cfg(test)]
impl MaxwellThreeDSynchronizationOperation {
    pub const fn trigger(&self) -> MaxwellThreeDSynchronizationTrigger {
        self.trigger
    }

    pub fn state(&self) -> &MaxwellThreeDState {
        &self.state
    }
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
