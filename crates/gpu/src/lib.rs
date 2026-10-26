//! Host-independent GPU contracts and diagnostics.
//!
//! Console frontends and host backends meet at this boundary without sharing
//! Horizon ABI, console packet formats, or concrete host graphics objects.

mod access;
mod address;
mod allocation;
pub mod authority;
pub mod physical;
mod working_set;
pub use working_set::WorkingSet;
mod color;
mod command;
mod conversion;
pub use color::{
    BlendComponent, BlendFactor, BlendOperation, ColorBlendState, ColorOutputState, ColorWriteMask,
};
pub use conversion::{ConversionImage, Nv12ColorTransform, Nv12Planes};
mod diagnostics;
mod execution;
mod presentation;
mod resource;
mod runtime;
pub use execution::{GpuExecution, MemoryAccess, OwnerAccess, PreparedStateId};
mod tessellation;
pub use tessellation::{
    TessellationControl, TessellationDomain, TessellationMode, TessellationOutput,
    TessellationSpacing, TessellationState, TessellationWinding,
};
mod submission;
mod synchronization;
mod view;

pub use access::{
    AccessDescriptionError, AccessMode, AccessScope, AccessTarget, BufferRange, PipelineStages,
    QueryRange, ResourceAccess, ResourceDependency, ResourceUsage,
};
pub use address::{
    GpuVirtualAddress, GpuVirtualAddressError, MappingGeneration, MappingGenerationExhausted,
};
pub use allocation::{
    AllocationDescriptionError, BackingView, BackingViewError, GpuAllocationDescription,
    GpuAllocationId,
};

pub use command::{
    AlphaCompareOperation, AlphaTest, AttachmentLoad, AttachmentStore, BlitFilter,
    CacheMaintenanceOperation, ClearValue, CommandDescriptionError, CullMode, DepthBias,
    DepthCompareOperation, DepthState, DrawArguments, FragmentDepthTransform, FrontFace,
    ImageOrigin, IndexType, LineRasterization, PrimitiveTopology, ScissorRect, StencilFaceState,
    StencilOperation, StencilTestState, TriangleRasterization, VertexAttribute,
    VertexComponentCount, VertexComponentWidth, VertexFormat, VertexPositionTransform,
    VertexStepMode, ViewportTransform,
};
pub use diagnostics::{
    CpuVirtualAddress, GpfifoEntryIndex, GpuChannelId, GpuClassId, GpuMethodId,
    GraphicsAllocationId, GraphicsGapKind,
};
pub use presentation::{
    PresentationImageFormat, PresentationImageId, PresentationImageRequest, PresentationLease,
    PresentationRelease, ResidentImage,
};
pub use resource::{
    AddressMode, BufferDescription, BufferId, DescriptorTableId, FilterMode, ImageDescription,
    ImageDimension, ImageExtent, ImageFormat, ImageId, ImageKind, PipelineId, QueryKind,
    QueryPoolId, RenderPassId, ResourceDescriptionError, SampleCount, SamplerDescription,
    SamplerId, ShaderId, ShaderStage,
};
pub use runtime::{BackendExecutionCompletion, BackendRuntimeError};
pub use submission::{BackendInstanceId, BackendSubmissionToken, FrontendSubmissionId};
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

pub mod cache_counters;
/// Diagnostic counters compiled only when explicitly enabled.
pub mod metrics;
pub use nixe_memory::trace;

pub mod texture_decode;

pub mod native;

#[cfg(feature = "performance-counters")]
pub mod compilation_capture;

pub mod compilation;
pub mod shader_compilation;
pub mod shader_encoding;
pub mod shader_identity;
pub mod shader_module;
