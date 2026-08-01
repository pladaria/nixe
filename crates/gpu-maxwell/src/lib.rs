//! Switch 1 Maxwell GPU frontend.
//!
//! This crate owns console GPU profile and command semantics. It depends only
//! on host-independent GPU contracts and never on Horizon or a host backend.

mod address_space;
mod capture;
mod channel;
mod engines;
mod gpfifo;
mod profile;
mod pushbuffer;
mod scheduler;

pub use address_space::{
    MAX_MAPPING_DUMP_ENTRIES, MaxwellAddressSpaceError, MaxwellAddressSpaceId,
    MaxwellAddressSpaceInitialization, MaxwellAllocationId, MaxwellGpuAccessError,
    MaxwellGpuAddressSpace, MaxwellGpuMapping, MaxwellMapRequest, MaxwellMappingDiagnostic,
    MaxwellMappingDump, MaxwellMappingId, MaxwellResolvedMapping, MaxwellResolvedRange,
    MaxwellSparseMapping, MaxwellSparseRemapRequest, MaxwellVaRegion, MaxwellVaReservation,
};
pub use capture::{
    MAXWELL_FRONTEND_CAPTURE_WORDS, MaxwellFrontendCapture, MaxwellFrontendCaptureError,
    MaxwellFrontendFailure, MaxwellFrontendReplay, capture_maxwell_frontend_dispatch,
    replay_maxwell_frontend_capture,
};
pub use channel::{
    MaxwellChannelError, MaxwellChannelFrontendState, MaxwellChannelId, MaxwellChannelOwner,
    MaxwellChannelPriority, MaxwellChannelSchedulingPolicy, MaxwellChannelTimeout,
    MaxwellChannelTimeslice, MaxwellGpuChannel, MaxwellMemoryManagerId, MaxwellObjectContext,
    MaxwellZCullBinding, MaxwellZCullMode,
};
pub use engines::{
    MAXWELL_BIND_GROUP_COUNT, MAXWELL_COLOR_TARGET_COUNT, MAXWELL_CONSTANT_BUFFER_SLOT_COUNT,
    MAXWELL_PIPELINE_SHADER_COUNT, MAXWELL_SCISSOR_COUNT,
    MAXWELL_THREE_D_SM_TIMEOUT_COUNTER_BIT_MAX, MAXWELL_TWO_D_CORRAL_SIZE_MAX,
    MAXWELL_TWO_D_NOTIFY_ADDRESS_UPPER_MAX, MAXWELL_VERTEX_ATTRIBUTE_COUNT,
    MAXWELL_VERTEX_STREAM_COUNT, MAXWELL_VIEWPORT_COUNT, MaxwellEngineCapability,
    MaxwellEngineDispatchError, MaxwellEngineMethodDispatch, MaxwellEngineMethodEffect,
    MaxwellEngineMethodMetadata, MaxwellEnginePacketDispatch, MaxwellThreeDAliasedLineWidthEnable,
    MaxwellThreeDAttachmentReadiness, MaxwellThreeDBegin, MaxwellThreeDBindGroupState,
    MaxwellThreeDBlendFactor, MaxwellThreeDBlendOp, MaxwellThreeDClearState,
    MaxwellThreeDClearSurface, MaxwellThreeDColorCompressionMode, MaxwellThreeDColorMask,
    MaxwellThreeDColorTargetFormat, MaxwellThreeDColorTargetSelection,
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
    MaxwellThreeDShaderBindingState, MaxwellThreeDShaderBindingWrite,
    MaxwellThreeDShaderExecutionState, MaxwellThreeDShaderExecutionStateWrite,
    MaxwellThreeDShaderResourceUse, MaxwellThreeDShaderStage, MaxwellThreeDSmTimeoutCounterBit,
    MaxwellThreeDState, MaxwellThreeDStateWrite, MaxwellThreeDStencilOp,
    MaxwellThreeDTranslatedShader, MaxwellThreeDTranslatedShaders, MaxwellThreeDTriggeredOperation,
    MaxwellThreeDUnresolvedAddress, MaxwellThreeDVertexAttributeFormat,
    MaxwellThreeDVertexComponentWidths, MaxwellThreeDVertexInputState,
    MaxwellThreeDVertexInputWrite, MaxwellThreeDVertexNumericalType,
    MaxwellThreeDVertexStreamFormat, MaxwellThreeDVertexStreamState,
    MaxwellThreeDViewportClipControl, MaxwellThreeDViewportState,
    MaxwellThreeDViewportTransformState, MaxwellThreeDViewportZClipRange,
    MaxwellThreeDZCompressionMode, MaxwellTwoDClipEnable, MaxwellTwoDColorKeyEnable,
    MaxwellTwoDNotifyAddressLower, MaxwellTwoDNotifyAddressUpper, MaxwellTwoDNotifyState,
    MaxwellTwoDNotifyStateWrite, MaxwellTwoDOperation, MaxwellTwoDPixelsFromMemoryCorralSize,
    MaxwellTwoDPixelsFromMemorySafeOverlap, MaxwellTwoDPixelsFromMemoryState,
    MaxwellTwoDPixelsFromMemoryStateWrite, MaxwellTwoDProcessingClusters, MaxwellTwoDRegister,
    MaxwellTwoDRegisterOrigin, MaxwellTwoDRenderEnableMode, MaxwellTwoDRenderEnableState,
    MaxwellTwoDRenderEnableStateWrite, MaxwellTwoDState, MaxwellTwoDStateWrite,
    commit_maxwell_engine_packet, dispatch_maxwell_engine_packet,
    dispatch_maxwell_engine_pushbuffer, preflight_maxwell_engine_packet,
    preflight_maxwell_three_d_operation, resolve_maxwell_three_d_resources,
};
pub use gpfifo::{
    MAXWELL_GPFIFO_CAPTURE_SOURCES, MAXWELL_GPFIFO_ENTRY_SIZE, MaxwellDecodedGpfifoSubmission,
    MaxwellGpfifoCapture, MaxwellGpfifoCaptureSource, MaxwellGpfifoDecodeError, MaxwellGpfifoEntry,
    MaxwellGpfifoFetchMode, MaxwellGpfifoLevel, MaxwellGpfifoSourceError,
    MaxwellGpfifoSourceLocation, MaxwellGpfifoSubmissionMode, MaxwellGpfifoSubmitRequest,
    MaxwellGpfifoSyncMode, MaxwellInvalidGpfifoSubmission, MaxwellRetainedPushbuffer,
    MaxwellUnsupportedGpfifoSubmission, MaxwellValidatedGpfifoSubmission, decode_gpfifo_submission,
    resolve_gpfifo_submission,
};
pub use profile::{
    AddressBitCount, ChipName, GpuArchitecture, GpuBusType, GpuFeatureFlags, GpuImplementation,
    GpuPageSize, GpuPageSizeMask, GpuProfileId, GpuRevision, MaxwellCacheCapabilities,
    MaxwellChipsetIdentity, MaxwellClassCapabilities, MaxwellGpuProfile,
    MaxwellInterconnectCapabilities, MaxwellMemoryCapabilities, MaxwellProfileValidationError,
    MaxwellShaderCapabilities, MaxwellTopology, MaxwellVirtualAddressCapabilities,
    MaxwellZCullCapabilities, SWITCH_1_GM20B_PROFILE, ShaderVersion,
};
pub use pushbuffer::dispatch::{
    MAXWELL_SET_OBJECT_METHOD, MaxwellMethodDispatch, MaxwellMethodDispatchError,
    MaxwellMethodDispatchKind, MaxwellMethodSource, MaxwellPacketDispatch,
    MaxwellSetObjectTransition, commit_maxwell_packet, dispatch_maxwell_packet,
    dispatch_maxwell_pushbuffer, preflight_maxwell_packet,
};
pub use pushbuffer::packet::{
    MaxwellDecodedMethod, MaxwellDecodedMethodPacket, MaxwellDecodedPacket,
    MaxwellDecodedPushbuffer, MaxwellMethodPacketMode, MaxwellPushbufferControl,
    MaxwellPushbufferDecodeError, MaxwellPushbufferSubchannel, MaxwellPushbufferWord,
    decode_maxwell_pushbuffer, decode_maxwell_submission,
};
pub use scheduler::{
    MaxwellFrontendDispatch, MaxwellFrontendDispatchBoundary, MaxwellScheduleError,
    MaxwellScheduledSubmission, MaxwellScheduler, MaxwellSchedulerSequence,
    MaxwellSubmissionOrderingStage,
};
