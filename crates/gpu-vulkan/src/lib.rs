//! Independent Vulkan instance/device ownership, capability selection and memory.
//! No wgpu objects, HAL assumptions, or guest decoder dependencies cross this crate.

pub mod cache;
mod device;
mod dispatch;
mod error;
mod extensions;
#[cfg(feature = "binding-probe")]
mod helpers;
#[cfg(feature = "binding-probe")]
mod probe;
#[cfg(feature = "binding-probe")]
pub use probe::BindingMeasurement;
mod instance;
mod memory;
#[cfg(feature = "binding-probe")]
pub use helpers::BufferBinding;
mod profile;
mod queue;
pub(crate) mod residency;
mod transfer;
pub use memory::NativeImageDescription;
pub use residency::{
    AccessScope, AttachmentId, AttachmentLoad, AttachmentStore, BarrierStatistics, BufferId,
    BufferRange, ImageId, ImagePlane, ImageViewDescription, ImageViewId, ResidencyStatistics,
    VulkanResidency,
};
mod validation;
pub use gpu_allocator::MemoryLocation;
pub use memory::{AllocatedBuffer, AllocatedImage, AllocatedImageView, AllocationMode};

pub use device::{DeviceOptions, QueueSelection, SurfaceProfile, VulkanDevice};
pub use error::VulkanError;
pub use instance::{NativeWindow, SurfaceTarget};
pub use profile::{
    ApiPath, ApiPreference, DeviceProfile, DynamicStateSupport, FloatControls, PortabilitySupport,
    SubgroupSupport, SwapchainMaintenance,
};
pub use queue::{QueueUsage, ReadbackTicket, RecordingStatistics, WorkStatus, WorkTicket};
pub use validation::{ValidationMode, ValidationMonitor};

pub use residency::execution::{
    AttachmentSource, Binding, Dispatch, Draw, GraphicsParameters, IndexBuffer, PreparedDraw,
    PreparedGraphics, PreparedProgram, PreparedTargets, RenderAttachment, RenderingTargets,
    VertexBuffer,
};

pub mod vertex;

mod native;
pub mod presentation;
pub use native::{NativeExecutionStatistics, NativeGpuExecution};

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
