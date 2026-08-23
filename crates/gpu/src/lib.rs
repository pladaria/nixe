//! Host-independent GPU contracts and diagnostics.
//!
//! Console frontends and host backends meet at this boundary without sharing
//! Horizon ABI, console packet formats, or concrete host graphics objects.

mod access;
mod address;
mod allocation;
mod backend;
mod cache;
mod capability;
mod command;
mod diagnostics;
mod presentation;
mod resource;
mod runtime;
mod shader;
mod submission;
mod synchronization;
mod view;

pub use access::{
    AccessDescriptionError, AccessMode, AccessScope, AccessTarget, BufferRange, PipelineStages,
    QueryRange, ResourceAccess, ResourceDependency, ResourceTransition, ResourceUsage,
};
pub use address::{GpuVirtualAddress, GpuVirtualAddressError};
pub use allocation::{
    AllocationDescriptionError, BackingView, BackingViewError, GpuAllocationDescription,
    GpuAllocationId,
};
pub use backend::{
    AcceptedBackendSubmission, Backend, BackendDriver, BackendDriverError, BackendError,
    BackendResourceCreateInfo, BackendResourceHandle, BackendResourceKind,
    BackendResourceValidationError, BackendState,
};
pub use cache::{
    DEFAULT_BIND_GROUPS_PER_DESCRIPTOR_TABLE, DEFAULT_PERSISTENT_PIPELINE_CACHE_BYTES,
    DEFAULT_PIPELINE_CACHE_ENTRIES, DEFAULT_PIPELINE_VARIANTS_PER_RESOURCE,
    DEFAULT_SHADER_CACHE_ENTRIES, GpuCacheConfiguration, GpuCacheConfigurationError,
    MIN_SHADER_CACHE_ENTRIES, cache_fingerprint,
};
pub use capability::{
    BackendCapabilities, BackendCapabilityError, BackendFeatures, BackendLimits,
    CapabilityAgreement, CapabilityRequirement, CapabilityRequirements,
};
pub use command::{
    AlphaCompareOperation, AlphaTest, AttachmentLoad, AttachmentStore, BarrierOperation,
    BufferRegion, CacheMaintenanceOperation, ClearOperation, ClearValue, CommandDescriptionError,
    CopyOperation, DepthCompareOperation, DepthState, DispatchOperation, DrawArguments,
    DrawOperation, GpuCommand, GpuOperation, ImageOrigin, ImageRegion, IndexType,
    OperationSubmission, PrimitiveTopology, QueryOperation, RenderAttachment, RenderPassOperation,
    TriangleRasterization, VertexAttribute, VertexBufferLayout, VertexComponentCount,
    VertexComponentWidth, VertexFormat, VertexStepMode, ViewportTransform,
};
pub use diagnostics::{
    CpuVirtualAddress, GpfifoEntryIndex, GpuChannelId, GpuClassId, GpuMethodId,
    GraphicsAllocationId, GraphicsGapKind,
};
pub use nixe_memory::{CanonicalBackingSegment, MappingGeneration};
pub use presentation::{PresentationImageFormat, PresentationImageRequest, ResidentImage};
pub use resource::{
    AddressMode, BufferDescription, BufferId, DescriptorKind, DescriptorTableBinding,
    DescriptorTableDescription, DescriptorTableId, FilterMode, ImageDescription, ImageDimension,
    ImageExtent, ImageFormat, ImageId, ImageKind, PipelineDescription, PipelineId, PipelineKind,
    QueryKind, QueryPoolDescription, QueryPoolId, RenderPassAttachmentDescription,
    RenderPassDescription, RenderPassId, ResourceDescriptionError, SampleCount, SamplerDescription,
    SamplerId, ShaderDescription, ShaderId, ShaderStage,
};
pub use runtime::{
    BackendExecutionCompletion, BackendRuntime, BackendRuntimeError, BackendVisibilityRequester,
    NeutralBackendRuntime,
};
pub use shader::{
    ShaderBackendLoweringError, ShaderBackendModule, ShaderEvaluationError, ShaderEvaluationInputs,
    ShaderEvaluationResult, ShaderFloatComparison, ShaderFloatControl, ShaderInstruction,
    ShaderInterfaceElement, ShaderInterpolation, ShaderIoLocation, ShaderIr,
    ShaderIrConstructionError, ShaderMathAccuracy, ShaderNanMode, ShaderOperation, ShaderPredicate,
    ShaderPredicateSetOperation, ShaderRegister, ShaderResourceAccess, ShaderResourceKind,
    ShaderRoundingMode, ShaderScalarType, ShaderSourceLocation, ShaderSpecialFunction,
    ShaderTextureSampleOutput, ShaderVerificationError, VerifiedShaderIr, evaluate_shader_ir,
    lower_shader_ir_to_wgsl, lower_shader_ir_to_wgsl_with_vertex_pulling,
};
pub use submission::{
    BackendInstanceId, BackendSubmissionToken, FrontendSubmissionId, FrontendSubmissionSegment,
};
pub use synchronization::{
    GuestSyncpointId, GuestSyncpointValue, GuestTimeline, GuestTimelinePoint, OwnerMismatch,
    ReservedTimelinePoint, SyncpointComparisonError, TimelineAdvanceError, TimelineIncrementError,
    TimelineInstanceId, TimelineOwnerId, TimelinePointComparisonError, TimelineReservationError,
};
pub use view::{
    BlockLinearLayout, BufferView, BufferViewError, ComponentSwizzle, ImageMemoryKind,
    ImageMemoryLayout, ImageSubresourceBinding, ImageSubresourceRange, ImageView, ImageViewError,
    Swizzle,
};
