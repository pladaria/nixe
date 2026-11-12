//! Declarative Maxwell class-method routing.
//!
//! This layer applies decoded class methods to channel-owned state in stream
//! order and returns the first typed fatal boundary. It contains no Horizon
//! ABI, guest mappings, scheduler state, or host-backend objects.

mod bulk;
pub mod compute;
mod dma_copy;
mod inline_to_memory;
pub mod lowering;
pub(crate) mod memory_copy;
mod spa;
pub mod threed;
pub(crate) use threed::MaxwellThreeDEngineState;
pub mod twod;

pub use spa::MaxwellSpaVersion;

pub use compute::{
    MAXWELL_COMPUTE_CWD_REF_COUNTER_COUNT, MaxwellComputeAddress,
    MaxwellComputeBindlessTextureConstantBufferSlot, MaxwellComputeCwdRefCounterIndex,
    MaxwellComputeCwdRefCounterState, MaxwellComputeCwdRefCounterValue,
    MaxwellComputeDescriptorPoolState, MaxwellComputeInlineToMemoryLaunch,
    MaxwellComputeInlineToMemoryLayout, MaxwellComputeInlineToMemoryPendingTransfer,
    MaxwellComputeInlineToMemoryState, MaxwellComputeInlineToMemoryUpload, MaxwellComputeLaunch,
    MaxwellComputeLaunchError, MaxwellComputeLocalMemoryAllocation, MaxwellComputeLocalMemoryState,
    MaxwellComputeOperationTrigger, MaxwellComputeProgramState, MaxwellComputeRegister,
    MaxwellComputeRegisterOrigin, MaxwellComputeShaderCacheInvalidation, MaxwellComputeSmCount,
    MaxwellComputeSpaVersion, MaxwellComputeState, MaxwellComputeSynchronizationPlan,
    MaxwellResolvedComputeLaunch, MaxwellShaderCacheInvalidation,
    lower_maxwell_compute_synchronization,
};

pub(crate) use compute::resolve_compute_launch;
pub use dma_copy::{MaxwellDmaCopyRegister, MaxwellDmaCopyRegisterName, MaxwellDmaCopyState};
pub use inline_to_memory::{
    MaxwellInlineToMemoryAddress, MaxwellInlineToMemoryLaunch,
    MaxwellInlineToMemoryPendingTransfer, MaxwellInlineToMemoryRegister,
    MaxwellInlineToMemorySemaphoreStructureSize, MaxwellInlineToMemoryState,
    MaxwellInlineToMemoryUpload,
};
pub use memory_copy::{
    MaxwellMemoryCopyComponentSource, MaxwellMemoryCopyError, MaxwellMemoryCopyLayout,
    MaxwellMemoryCopyOperation, MaxwellMemoryCopyRemap,
};
pub use threed::{
    MAXWELL_BIND_GROUP_COUNT, MAXWELL_COLOR_TARGET_COUNT, MAXWELL_CONSTANT_BUFFER_SLOT_COUNT,
    MAXWELL_PIPELINE_SHADER_COUNT, MAXWELL_POLYGON_STIPPLE_PATTERN_WORD_COUNT,
    MAXWELL_SAMPLE_LOCATION_GROUP_COUNT, MAXWELL_SAMPLE_LOCATIONS_PER_GROUP, MAXWELL_SCISSOR_COUNT,
    MAXWELL_TESSELLATION_LOD_COUNT, MAXWELL_THREE_D_MME_EMITTED_METHOD_LIMIT,
    MAXWELL_THREE_D_MME_EXECUTION_INSTRUCTION_LIMIT, MAXWELL_THREE_D_MME_SHADOW_SCRATCH_COUNT,
    MAXWELL_THREE_D_PRIMITIVE_AREA_MAX, MAXWELL_THREE_D_SHADER_LOCAL_MEMORY_PER_WARP_SIZE_MAX,
    MAXWELL_THREE_D_SM_TIMEOUT_COUNTER_BIT_MAX, MAXWELL_VERTEX_ATTRIBUTE_COUNT,
    MAXWELL_VERTEX_STREAM_COUNT, MAXWELL_VIEWPORT_COUNT, MAXWELL_WINDOW_CLIP_COUNT,
    MaxwellLoweringError, MaxwellShaderStage, MaxwellTessellationModeError,
    MaxwellThreeDAliasedLineWidthEnable, MaxwellThreeDAlphaFraction,
    MaxwellThreeDAlphaToCoverageDither, MaxwellThreeDAlphaToCoverageOverride,
    MaxwellThreeDAntiAliasedLineEnable, MaxwellThreeDApiMandatedEarlyZ,
    MaxwellThreeDAttachmentReadiness, MaxwellThreeDAttributeDefaultVector,
    MaxwellThreeDAttributeDefaults, MaxwellThreeDAttributePointSize,
    MaxwellThreeDBalancedPrimitiveWorkload, MaxwellThreeDBegin, MaxwellThreeDBeginInstance,
    MaxwellThreeDBindGroupState, MaxwellThreeDBlendControlState,
    MaxwellThreeDBlendFloatPixelKillEnable, MaxwellThreeDBlendPerFormatEnable,
    MaxwellThreeDBlendZeroTimesAnythingIsZero, MaxwellThreeDClearState, MaxwellThreeDClearSurface,
    MaxwellThreeDClipAxis, MaxwellThreeDClipIdTestEnable, MaxwellThreeDColorCompressionMode,
    MaxwellThreeDColorReductionFp16Threshold, MaxwellThreeDColorReductionSrgb8Threshold,
    MaxwellThreeDColorReductionState, MaxwellThreeDColorReductionThresholdsEnable,
    MaxwellThreeDColorReductionThresholdsFp16, MaxwellThreeDColorReductionThresholdsSrgb8,
    MaxwellThreeDColorReductionThresholdsUnorm8, MaxwellThreeDColorReductionThresholdsUnorm10,
    MaxwellThreeDColorReductionThresholdsUnorm16, MaxwellThreeDColorTargetFormat,
    MaxwellThreeDColorTargetState, MaxwellThreeDCompareOp, MaxwellThreeDCompressionThreshold,
    MaxwellThreeDConditionalLoadConstantBuffer, MaxwellThreeDConservativeRasterEnable,
    MaxwellThreeDConstantBufferBinding, MaxwellThreeDConstantBufferSelectorState,
    MaxwellThreeDCoverageState, MaxwellThreeDCoverageToColor, MaxwellThreeDCsaaEnable,
    MaxwellThreeDCullFace, MaxwellThreeDDecompressSurface, MaxwellThreeDDepthArrayControl,
    MaxwellThreeDDepthStencilFormat, MaxwellThreeDDepthStencilTargetState,
    MaxwellThreeDDepthTargetCount, MaxwellThreeDDescriptorPoolState,
    MaxwellThreeDDirectlyAddressableMemory, MaxwellThreeDDrawIndexBuffer, MaxwellThreeDDrawState,
    MaxwellThreeDEdgeFlag, MaxwellThreeDFillViaTriangleMode, MaxwellThreeDFixedFunctionRegister,
    MaxwellThreeDFixedFunctionState, MaxwellThreeDFixedFunctionValue,
    MaxwellThreeDFlushPendingWrites, MaxwellThreeDFrontFace, MaxwellThreeDGuestImageFormat,
    MaxwellThreeDHybridAntiAliasCentroid, MaxwellThreeDHybridAntiAliasControl,
    MaxwellThreeDImageKind, MaxwellThreeDImageLayout, MaxwellThreeDInlineToMemoryCompletion,
    MaxwellThreeDInlineToMemoryLaunch, MaxwellThreeDInlineToMemoryLayout,
    MaxwellThreeDInlineToMemoryState, MaxwellThreeDIteratedBlend,
    MaxwellThreeDIteratedBlendPassCount, MaxwellThreeDLineState,
    MaxwellThreeDLineStippleParameters, MaxwellThreeDLogicOp, MaxwellThreeDMmeExecutionError,
    MaxwellThreeDMmeInstruction, MaxwellThreeDMmeLoadError, MaxwellThreeDMmeRam,
    MaxwellThreeDMmeRamAddress, MaxwellThreeDMmeShadowRamControl, MaxwellThreeDMmeState,
    MaxwellThreeDMutableMethodControl, MaxwellThreeDPatchSize, MaxwellThreeDPipelineBindingState,
    MaxwellThreeDPixelShaderClampRange, MaxwellThreeDPixelShaderInterlockControl,
    MaxwellThreeDPixelShaderInterlockFragmentOrder, MaxwellThreeDPixelShaderInterlockMode,
    MaxwellThreeDPixelShaderInterlockTileSize, MaxwellThreeDPixelShaderSaturate,
    MaxwellThreeDPointCenterMode, MaxwellThreeDPointSize, MaxwellThreeDPointSpriteOrigin,
    MaxwellThreeDPointSpriteRMode, MaxwellThreeDPointSpriteSelect,
    MaxwellThreeDPolygonClipGeneratedEdge, MaxwellThreeDPolygonMode,
    MaxwellThreeDPostZPixelShaderImask, MaxwellThreeDPrimitiveCircularBufferThrottle,
    MaxwellThreeDPrimitiveState, MaxwellThreeDPrimitiveTopology, MaxwellThreeDProgramRegionState,
    MaxwellThreeDProvokingVertex, MaxwellThreeDPsOutputSampleMaskUsage,
    MaxwellThreeDRasterBoundingBox, MaxwellThreeDRasterBoundingBoxMode, MaxwellThreeDRasterState,
    MaxwellThreeDRawValue, MaxwellThreeDRectangle, MaxwellThreeDRegister,
    MaxwellThreeDRenderEnableMode, MaxwellThreeDRenderEnableOverride,
    MaxwellThreeDRenderEnableState, MaxwellThreeDRenderTargetIndexOffset,
    MaxwellThreeDRenderTargetLayer, MaxwellThreeDRenderTargetLayerControl,
    MaxwellThreeDRenderTargetState, MaxwellThreeDReportSemaphoreControl,
    MaxwellThreeDReportSemaphoreOperation, MaxwellThreeDReportSemaphorePipelineLocation,
    MaxwellThreeDReportSemaphoreState, MaxwellThreeDReportSemaphoreStructureSize,
    MaxwellThreeDReportSemaphoreWrite, MaxwellThreeDResolvedSampler, MaxwellThreeDResourceError,
    MaxwellThreeDResourceRole, MaxwellThreeDSampleLocation, MaxwellThreeDSampleLocationGroup,
    MaxwellThreeDSampleMode, MaxwellThreeDSamplerBindingMode, MaxwellThreeDScissorState,
    MaxwellThreeDSeparateFragmentData, MaxwellThreeDShadeMode, MaxwellThreeDShaderBindingState,
    MaxwellThreeDShaderCacheInvalidation, MaxwellThreeDShaderExceptionsEnable,
    MaxwellThreeDShaderExecutionState, MaxwellThreeDShaderLocalMemoryPerWarpSize,
    MaxwellThreeDShaderLocalMemoryState, MaxwellThreeDShaderWatermarkRange,
    MaxwellThreeDShaderWatermarkTarget, MaxwellThreeDSmTimeoutCounterBit, MaxwellThreeDState,
    MaxwellThreeDStencilOp, MaxwellThreeDSubtilingPerfKnobA, MaxwellThreeDSubtilingPerfKnobB,
    MaxwellThreeDSynchronizationError, MaxwellThreeDSynchronizationPlan,
    MaxwellThreeDSynchronizationTrigger, MaxwellThreeDSyncpointCondition,
    MaxwellThreeDSyncpointIncrement, MaxwellThreeDTessellationLod, MaxwellThreeDTessellationMode,
    MaxwellThreeDTextureCacheInvalidation, MaxwellThreeDTextureCacheLines,
    MaxwellThreeDTextureCacheTarget, MaxwellThreeDTiledCacheFlushMode, MaxwellThreeDTirControl,
    MaxwellThreeDTirMode, MaxwellThreeDTirModulationComponentSelect,
    MaxwellThreeDTirModulationFunction, MaxwellThreeDUnorm8, MaxwellThreeDUnresolvedAddress,
    MaxwellThreeDVertexArrayPrimitiveRestartEnable, MaxwellThreeDVertexAssemblyState,
    MaxwellThreeDVertexAttributeFormat, MaxwellThreeDVertexComponentWidths,
    MaxwellThreeDVertexIdUsesArrayStart, MaxwellThreeDVertexInputState,
    MaxwellThreeDVertexNumericalType, MaxwellThreeDVertexStreamFormat,
    MaxwellThreeDVertexStreamState, MaxwellThreeDVertexStreamSubstituteState,
    MaxwellThreeDViewportClipControl, MaxwellThreeDViewportCoordinateSwizzle,
    MaxwellThreeDViewportPixelCenter, MaxwellThreeDViewportScaleOffsetEnable,
    MaxwellThreeDViewportState, MaxwellThreeDViewportSwizzleComponent,
    MaxwellThreeDViewportTransformState, MaxwellThreeDViewportZClipRange,
    MaxwellThreeDVisibleCallLimit, MaxwellThreeDWindowClipState, MaxwellThreeDWindowClipType,
    MaxwellThreeDZCompressionMode, lower_maxwell_three_d_synchronization,
};
pub use twod::{
    MAXWELL_TWO_D_CORRAL_SIZE_MAX, MAXWELL_TWO_D_NOTIFY_ADDRESS_UPPER_MAX, MaxwellTwoDBeta1,
    MaxwellTwoDBeta4, MaxwellTwoDBetaState, MaxwellTwoDBlitOperation, MaxwellTwoDClipEnable,
    MaxwellTwoDColorKeyEnable, MaxwellTwoDNotifyAddressLower, MaxwellTwoDNotifyAddressUpper,
    MaxwellTwoDNotifyState, MaxwellTwoDOperation, MaxwellTwoDPixelsFromMemoryCorralSize,
    MaxwellTwoDPixelsFromMemorySafeOverlap, MaxwellTwoDPixelsFromMemoryState,
    MaxwellTwoDProcessingClusters, MaxwellTwoDRegister, MaxwellTwoDRegisterOrigin,
    MaxwellTwoDRenderEnableMode, MaxwellTwoDRenderEnableState, MaxwellTwoDSolidOperation,
    MaxwellTwoDState,
};

use std::fmt::{Display, Formatter};

use nixe_gpu::{GpuClassId, GpuMethodId};

use crate::{
    MaxwellAamVersionRange, MaxwellGpuChannel, MaxwellMethodDispatchError, MaxwellMethodSource,
    MaxwellShaderProgramHeaderVersionRange,
};

/// Direct engine selection owned by each PBDMA subchannel, as in DmaPusher.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub(crate) enum Engine {
    ThreeD,
    Compute,
    InlineToMemory,
    TwoD,
    Copy,
}
impl Engine {
    pub(crate) const fn class(self) -> GpuClassId {
        match self {
            Self::ThreeD => threed::CLASS,
            Self::Compute => compute::CLASS,
            Self::InlineToMemory => inline_to_memory::CLASS,
            Self::TwoD => twod::CLASS,
            Self::Copy => dma_copy::CLASS,
        }
    }
}

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
    pub const fn method(self) -> GpuMethodId {
        self.method
    }

    #[must_use]
    pub const fn method_name(self) -> &'static str {
        self.method_name
    }
}

/// Typed class-dispatch boundary. Missing coverage is never a guest result.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MaxwellEngineDispatchError {
    Binding(MaxwellMethodDispatchError),
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
    InvalidTwoDMethodEncoding {
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
            Self::InvalidTwoDMethodEncoding {
                source,
                method_name,
                reason,
            } => write!(
                formatter,
                "Maxwell 2D method cannot be represented: {source} class-name=FERMI_TWOD_A method-name={method_name} reason={reason}"
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
pub(crate) enum MaxwellEngineStreamError {
    Dispatch(Box<MaxwellEngineDispatchError>),
    Execution(Box<crate::MaxwellSubmissionExecutionError>),
}

impl MaxwellEngineStreamError {
    fn dispatch(error: MaxwellEngineDispatchError) -> Self {
        Self::Dispatch(Box::new(error))
    }
    fn execution(error: crate::MaxwellSubmissionExecutionError) -> Self {
        Self::Execution(Box::new(error))
    }
}

impl From<MaxwellEngineDispatchError> for MaxwellEngineStreamError {
    fn from(error: MaxwellEngineDispatchError) -> Self {
        Self::dispatch(error)
    }
}
impl From<crate::MaxwellSubmissionExecutionError> for MaxwellEngineStreamError {
    fn from(error: crate::MaxwellSubmissionExecutionError) -> Self {
        Self::execution(error)
    }
}

/// Consume expanded methods on the preparation owner. Macro ISA and shadow
/// replay have already run on the command owner; this bank owns all effects.
pub(crate) fn select_engine(
    channel: &mut MaxwellGpuChannel,
    source: MaxwellMethodSource,
    engine: Engine,
    bind: bool,
) -> Result<(), MaxwellEngineStreamError> {
    if let Some((_, armed)) = channel.three_d_mut().pending_notification
        && (bind || source.subchannel() != armed.subchannel() || engine != Engine::ThreeD)
    {
        return Err(pending_notification_switch(source).into());
    }
    Ok(())
}
pub(crate) fn execute_host<L: crate::MaxwellFrontend, H: crate::MaxwellCommandHost>(
    channel: &mut MaxwellGpuChannel,
    source: MaxwellMethodSource,
    effect: crate::MaxwellHostSynchronizationKind,
    execution: &mut crate::execution::ChannelExecution<'_, L, H>,
) -> Result<(), MaxwellEngineStreamError> {
    if let Some((_, armed)) = channel.three_d_mut().pending_notification
        && source.subchannel() != armed.subchannel()
    {
        return Err(pending_notification_switch(source).into());
    }
    execution.host_synchronization(source, effect)?;
    Ok(())
}
pub(crate) fn execute_method<L: crate::MaxwellFrontend, H: crate::MaxwellCommandHost>(
    channel: &mut MaxwellGpuChannel,
    engine: Engine,
    source: MaxwellMethodSource,
    execution: &mut crate::execution::ChannelExecution<'_, L, H>,
) -> Result<(), MaxwellEngineStreamError> {
    let method = source.method();
    match engine {
        Engine::ThreeD => {
            let (profile, state) = channel.three_d_with_profile_mut();
            threed::call_method(profile, method, source.argument(), state, execution, || {
                source
            })?;
        }
        Engine::Compute => compute::call_method(source, channel.compute_mut(), execution)?,
        Engine::Copy => {
            dma_copy::call_method(source, channel.dma_copy_mut(), execution)?;
        }
        Engine::InlineToMemory => {
            inline_to_memory::call_method(source, channel.inline_to_memory_mut(), execution)?;
        }
        Engine::TwoD => twod::call_method(source, channel.two_d_mut(), execution)?,
    }
    Ok(())
}
pub(crate) use bulk::execute_multi;
pub(crate) fn flush_macro<L: crate::MaxwellFrontend, H: crate::MaxwellCommandHost>(
    channel: &mut MaxwellGpuChannel,
    execution: &mut crate::execution::ChannelExecution<'_, L, H>,
) -> Result<(), MaxwellEngineStreamError> {
    threed::draw::flush(channel.three_d_mut(), execution, false)
        .map_err(MaxwellEngineStreamError::execution)
}
fn pending_notification_switch(source: MaxwellMethodSource) -> MaxwellEngineDispatchError {
    MaxwellEngineDispatchError::InvalidMethodEncoding {
        source,
        method_name: "NOTIFY",
        reason: "object or subchannel switched with a pending notification",
    }
}
