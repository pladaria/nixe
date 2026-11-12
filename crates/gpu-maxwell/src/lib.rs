//! Switch 1 Maxwell GPU frontend.
//!
//! This crate owns console GPU profile and command semantics. It depends only
//! on host-independent GPU contracts and never on Horizon or a host backend.

mod address_space;
mod device_memory;
pub use device_memory::{MaxwellDeviceMapping, MaxwellDeviceMemory};
mod channel;
pub mod engines;
mod execution;
mod frontend;
mod ordered;
pub use frontend::MaxwellCommandProcessor;
pub use ordered::{MaxwellMethodExecution, MaxwellMethodSink, MaxwellMethodState};
mod gpfifo;
mod presentation;
pub use presentation::{MissingGpuExecution, PresentationImageRequest};
mod profile;
mod pushbuffer;
mod scheduler;
mod shader;
mod shader_stage;

pub use address_space::{
    CommandRead, MAX_MAPPING_DUMP_ENTRIES, MaxwellAddressSpaceError, MaxwellAddressSpaceId,
    MaxwellAddressSpaceInitialization, MaxwellAllocationId, MaxwellCommandMemory,
    MaxwellGpuAccessError, MaxwellGpuAddressSpace, MaxwellGpuMapping, MaxwellGpuMemory,
    MaxwellGpuMemoryHandle, MaxwellGpuMemoryInterface, MaxwellMapRequest, MaxwellMappingDiagnostic,
    MaxwellMappingDump, MaxwellMappingId, MaxwellResolvedMapping, MaxwellResolvedRange,
    MaxwellSparseMapping, MaxwellSparseRemapRequest, MaxwellVaRegion, MaxwellVaReservation,
};
pub use channel::{
    MaxwellChannelConfiguration, MaxwellChannelDriverState, MaxwellChannelError,
    MaxwellChannelFrontendState, MaxwellChannelId, MaxwellChannelOwner, MaxwellChannelPriority,
    MaxwellChannelSchedulingPolicy, MaxwellChannelTimeout, MaxwellChannelTimeslice,
    MaxwellGpuChannel, MaxwellMemoryManagerId, MaxwellObjectContext, MaxwellZCullBinding,
    MaxwellZCullMode,
};
pub use engines::{
    MAXWELL_BIND_GROUP_COUNT, MAXWELL_COLOR_TARGET_COUNT, MAXWELL_COMPUTE_CWD_REF_COUNTER_COUNT,
    MAXWELL_CONSTANT_BUFFER_SLOT_COUNT, MAXWELL_PIPELINE_SHADER_COUNT,
    MAXWELL_POLYGON_STIPPLE_PATTERN_WORD_COUNT, MAXWELL_SAMPLE_LOCATION_GROUP_COUNT,
    MAXWELL_SAMPLE_LOCATIONS_PER_GROUP, MAXWELL_SCISSOR_COUNT, MAXWELL_TESSELLATION_LOD_COUNT,
    MAXWELL_THREE_D_MME_EMITTED_METHOD_LIMIT, MAXWELL_THREE_D_MME_EXECUTION_INSTRUCTION_LIMIT,
    MAXWELL_THREE_D_MME_SHADOW_SCRATCH_COUNT, MAXWELL_THREE_D_PRIMITIVE_AREA_MAX,
    MAXWELL_THREE_D_SHADER_LOCAL_MEMORY_PER_WARP_SIZE_MAX,
    MAXWELL_THREE_D_SM_TIMEOUT_COUNTER_BIT_MAX, MAXWELL_TWO_D_CORRAL_SIZE_MAX,
    MAXWELL_TWO_D_NOTIFY_ADDRESS_UPPER_MAX, MAXWELL_VERTEX_ATTRIBUTE_COUNT,
    MAXWELL_VERTEX_STREAM_COUNT, MAXWELL_VIEWPORT_COUNT, MAXWELL_WINDOW_CLIP_COUNT,
    MaxwellComputeAddress, MaxwellComputeBindlessTextureConstantBufferSlot,
    MaxwellComputeCwdRefCounterIndex, MaxwellComputeCwdRefCounterState,
    MaxwellComputeCwdRefCounterValue, MaxwellComputeDescriptorPoolState,
    MaxwellComputeInlineToMemoryLaunch, MaxwellComputeInlineToMemoryLayout,
    MaxwellComputeInlineToMemoryPendingTransfer, MaxwellComputeInlineToMemoryState,
    MaxwellComputeInlineToMemoryUpload, MaxwellComputeLaunch, MaxwellComputeLaunchError,
    MaxwellComputeLocalMemoryAllocation, MaxwellComputeLocalMemoryState,
    MaxwellComputeOperationTrigger, MaxwellComputeProgramState, MaxwellComputeRegister,
    MaxwellComputeRegisterOrigin, MaxwellComputeShaderCacheInvalidation, MaxwellComputeSmCount,
    MaxwellComputeSpaVersion, MaxwellComputeState, MaxwellComputeSynchronizationPlan,
    MaxwellDmaCopyRegister, MaxwellDmaCopyRegisterName, MaxwellDmaCopyState,
    MaxwellEngineCapability, MaxwellEngineDispatchError, MaxwellEngineMethodMetadata,
    MaxwellInlineToMemoryAddress, MaxwellInlineToMemoryLaunch,
    MaxwellInlineToMemoryPendingTransfer, MaxwellInlineToMemoryRegister,
    MaxwellInlineToMemorySemaphoreStructureSize, MaxwellInlineToMemoryState,
    MaxwellInlineToMemoryUpload, MaxwellLoweringError, MaxwellMemoryCopyComponentSource,
    MaxwellMemoryCopyError, MaxwellMemoryCopyLayout, MaxwellMemoryCopyOperation,
    MaxwellMemoryCopyRemap, MaxwellShaderCacheInvalidation, MaxwellShaderStage, MaxwellSpaVersion,
    MaxwellTessellationModeError, MaxwellThreeDAliasedLineWidthEnable, MaxwellThreeDAlphaFraction,
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
    MaxwellThreeDZCompressionMode, MaxwellTwoDBeta1, MaxwellTwoDBeta4, MaxwellTwoDBetaState,
    MaxwellTwoDBlitOperation, MaxwellTwoDClipEnable, MaxwellTwoDColorKeyEnable,
    MaxwellTwoDNotifyAddressLower, MaxwellTwoDNotifyAddressUpper, MaxwellTwoDNotifyState,
    MaxwellTwoDOperation, MaxwellTwoDPixelsFromMemoryCorralSize,
    MaxwellTwoDPixelsFromMemorySafeOverlap, MaxwellTwoDPixelsFromMemoryState,
    MaxwellTwoDProcessingClusters, MaxwellTwoDRegister, MaxwellTwoDRegisterOrigin,
    MaxwellTwoDRenderEnableMode, MaxwellTwoDRenderEnableState, MaxwellTwoDState,
    lower_maxwell_compute_synchronization, lower_maxwell_three_d_synchronization,
};
pub use execution::{MaxwellCanonicalExecutionError, MaxwellSubmissionExecutionError};

pub use frontend::{MaxwellCommandFetch, MaxwellCommandStream, MaxwellFrontendDispatchError};
pub use gpfifo::{
    MAXWELL_GPFIFO_ENTRY_SIZE, MaxwellDecodedGpfifoSubmission, MaxwellGpfifoDecodeError,
    MaxwellGpfifoEntry, MaxwellGpfifoFetchMode, MaxwellGpfifoLevel, MaxwellGpfifoSourceError,
    MaxwellGpfifoSourceLocation, MaxwellGpfifoSubmissionMode, MaxwellGpfifoSubmitRequest,
    MaxwellGpfifoSyncMode, MaxwellInvalidGpfifoSubmission, MaxwellUnsupportedGpfifoSubmission,
    MaxwellValidatedGpfifoSubmission, decode_gpfifo_submission, validate_gpfifo_submission,
};
pub use profile::{
    AddressBitCount, ChipName, GpuArchitecture, GpuBusType, GpuFeatureFlags, GpuImplementation,
    GpuPageSize, GpuPageSizeMask, GpuProfileId, GpuRevision, MaxwellAamVersion,
    MaxwellAamVersionRange, MaxwellCacheCapabilities, MaxwellChipsetIdentity,
    MaxwellClassCapabilities, MaxwellGpuProfile, MaxwellInterconnectCapabilities,
    MaxwellMemoryCapabilities, MaxwellProfileValidationError, MaxwellShaderCapabilities,
    MaxwellShaderProgramHeaderVersion, MaxwellShaderProgramHeaderVersionRange, MaxwellTopology,
    MaxwellVirtualAddressCapabilities, MaxwellZCullCapabilities, SWITCH_1_GM20B_PROFILE,
    ShaderVersion, maxwell_gpu_timestamp,
};
pub use pushbuffer::dispatch::{
    MAXWELL_LEGACY_MEM_OP_A_METHOD, MAXWELL_LEGACY_MEM_OP_B_METHOD, MAXWELL_SET_OBJECT_METHOD,
    MaxwellHostSynchronizationKind, MaxwellMethodDispatchError, MaxwellMethodSource,
};
pub(crate) use pushbuffer::dma::MaxwellDmaArguments;
pub use pushbuffer::dma::MaxwellPushbufferSubchannel;

pub use scheduler::{
    MaxwellFrontendDispatch, MaxwellFrontendDispatchBoundary, MaxwellScheduleError,
    MaxwellScheduledSubmission, MaxwellScheduler, MaxwellSchedulerSequence,
    MaxwellSubmissionOrderingStage,
};

pub use execution::{MaxwellCommandHost, MaxwellFrontend};

pub use engines::{MaxwellResolvedComputeLaunch, MaxwellTwoDSolidOperation};
pub use execution::MissingRenderer;

pub use shader::recompiler;

#[cfg(any(test, feature = "test-support"))]
pub mod test_memory;
