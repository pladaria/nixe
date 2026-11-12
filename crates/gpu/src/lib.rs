//! Host-independent GPU contracts and diagnostics.
//!
//! Console frontends and host backends meet at this boundary without sharing
//! Horizon ABI, console packet formats, or concrete host graphics objects.

mod access;
mod address;
mod allocation;
pub mod memory_tracker;
pub mod physical;
mod working_set;
pub use working_set::{Equivalent, WorkingSet};
mod command;
mod conversion;
pub use conversion::{ConversionImage, Nv12ColorTransform, Nv12Planes};
mod diagnostics;
mod execution;
pub mod ordered;
mod presentation;
mod resource;
mod runtime;
pub use execution::{FenceAction, GpuExecution, MemoryAccess, OwnerMemory};
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
    CacheMaintenanceOperation, ClearValue, CommandDescriptionError, DepthCompareOperation,
    ImageOrigin, PrimitiveTopology, ScissorRect, VertexAttribute, VertexComponentCount,
    VertexComponentWidth, VertexFormat, VertexPositionTransform, VertexStepMode,
};
pub use diagnostics::{
    CpuVirtualAddress, GpfifoEntryIndex, GpuChannelId, GpuClassId, GpuMethodId,
    GraphicsAllocationId, GraphicsGapKind,
};
pub use presentation::{
    FrameCrop, FrameTransform, PresentationImageFormat, PresentationImageId, PresentationLease,
    PresentationRelease, ResidentImage,
};
pub use resource::{
    AddressMode, BufferDescription, BufferId, DescriptorTableId, FilterMode, ImageDescription,
    ImageDimension, ImageExtent, ImageFormat, ImageId, ImageKind, PipelineId, QueryKind,
    QueryPoolId, RenderPassId, ResourceDescriptionError, SampleCount, SamplerDescription,
    SamplerId, SamplerReduction, ShaderId, ShaderStage,
};
pub use runtime::BackendRuntimeError;
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
pub mod frame_metrics;
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
