//! Independent Vulkan instance/device ownership, capability selection and memory.
//! No wgpu objects, HAL assumptions, or guest decoder dependencies cross this crate.

pub mod cache;
mod device;
mod dispatch;
mod error;
mod extensions;
mod image_layout;
mod instance;
mod memory;
mod profile;
mod queue;
mod render_pass;
pub(crate) mod residency;
pub mod smooth_motion;
mod transfer;
pub use memory::NativeImageDescription;
pub use residency::{
    AccessScope, AttachmentId, AttachmentLoad, AttachmentStore, BufferId, BufferRange, ImageId,
    ImagePlane, ImageViewDescription, ImageViewId, Residency, ResidencyStatistics, VulkanResidency,
};
mod validation;
pub use gpu_allocator::MemoryLocation;
pub use memory::{AllocatedBuffer, AllocatedImage, AllocatedImageView, AllocationMode};

pub use device::{DeviceOptions, QueueSelection, SurfaceProfile, VulkanDevice};
pub use error::VulkanError;
pub use instance::{NativeWindow, SurfaceTarget};
pub use nixe_shader_recompiler::profile::ResolutionScaling;
pub(crate) const RESOLUTION_SCALES: [ResolutionScaling; 4] = [
    ResolutionScaling::new(1, 0),
    ResolutionScaling::new(3, 1),
    ResolutionScaling::new(2, 0),
    ResolutionScaling::new(3, 0),
];
pub use profile::{
    ApiPath, ApiPreference, DeviceProfile, DynamicStateSupport, FloatControls, PortabilitySupport,
    SubgroupSupport,
};
pub use queue::{
    CompletedReadback, QueueUsage, ReadbackTicket, RecordingStatistics, WorkStatus, WorkTicket,
};
pub use validation::{ValidationMode, ValidationMonitor};

pub use residency::execution::{
    AttachmentSource, Dispatch, Draw, GraphicsParameters, HostBinding, HostBuffer, HostGraphics,
    HostImage, HostProgram, IndexBuffer, PreparedTargets, RenderAttachment, RenderingTargets,
    VertexBuffer,
};

mod native;
pub mod presentation;
pub use native::{
    MaxwellVulkanRasterizer, NativeExecutionStatistics, NativeFrontendStatistics,
    NativeGpuExecution,
};

#[cfg(test)]
mod test_shaders;

#[cfg(test)]
mod test_device {
    use crate as nixe_gpu_vulkan;
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/support/device.rs"
    ));
}

mod slot_vector;
