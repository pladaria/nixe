use ash::vk;
use std::fmt::{Display, Formatter};

#[derive(Debug)]
pub enum VulkanError {
    Loader(Box<str>),
    Compilation(Box<str>),
    TextureDecode(Box<str>),
    Vulkan {
        operation: &'static str,
        result: vk::Result,
    },
    Allocation(gpu_allocator::AllocationError),
    WindowHandle(Box<str>),
    MissingInstanceExtension(String),
    ValidationUnavailable,
    ApiVersion {
        available: u32,
    },
    NoSuitableDevice(Vec<String>),
    InvalidResource(&'static str),
    Validation(Vec<String>),
    QueueStopped,
    SubmissionFailed {
        tick: u64,
        cause: Box<VulkanError>,
    },
    Memory(std::sync::Arc<nixe_memory::MemoryError>),
}

impl VulkanError {
    pub(crate) fn vk(operation: &'static str, result: vk::Result) -> Self {
        Self::Vulkan { operation, result }
    }
}

impl Display for VulkanError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TextureDecode(e) => write!(f,"Vulkan texture decode: {e}"),
            Self::Compilation(e) => write!(f,"Vulkan shader/pipeline compilation: {e}"),
            Self::Loader(e) => write!(f, "cannot load the Vulkan loader: {e}"),
            Self::Vulkan { operation, result } => write!(f, "{operation} failed: {result:?}"),
            Self::Allocation(e) => write!(f, "Vulkan allocation failed: {e}"),
            Self::WindowHandle(e) => write!(f, "cannot obtain native window handles: {e}"),
            Self::MissingInstanceExtension(e) => write!(f, "Vulkan instance extension required by the surface is unavailable: {e}"),
            Self::ValidationUnavailable => f.write_str("Vulkan validation requested, but VK_LAYER_KHRONOS_validation or VK_EXT_debug_utils is unavailable; install a Vulkan SDK/validation layer or configure VK_LAYER_PATH"),
            Self::ApiVersion { available } => write!(f, "Vulkan 1.2 or newer is required; loader provides {}.{}", vk::api_version_major(*available), vk::api_version_minor(*available)),
            Self::NoSuitableDevice(reasons) => write!(f, "no device meets the Vulkan profile: {}", reasons.join("; ")),
            Self::InvalidResource(e) => write!(f, "invalid Vulkan resource: {e}"),
            Self::Validation(e) => write!(f, "Vulkan validation reported errors: {}", e.join("; ")),
            Self::Memory(error) => write!(f, "Vulkan physical memory: {error}"),
            Self::QueueStopped => f.write_str("Vulkan queue is stopped after shutdown or a terminal device error"),
            Self::SubmissionFailed { tick, cause } => write!(f, "Vulkan submission tick {tick} failed: {cause}"),
        }
    }
}
impl std::error::Error for VulkanError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::SubmissionFailed { cause, .. } => Some(cause.as_ref()),
            Self::Memory(cause) => Some(cause.as_ref()),
            _ => None,
        }
    }
}
impl From<gpu_allocator::AllocationError> for VulkanError {
    fn from(e: gpu_allocator::AllocationError) -> Self {
        Self::Allocation(e)
    }
}

impl From<nixe_memory::MemoryError> for VulkanError {
    fn from(error: nixe_memory::MemoryError) -> Self {
        Self::Memory(std::sync::Arc::new(error))
    }
}
impl From<std::sync::Arc<nixe_memory::MemoryError>> for VulkanError {
    fn from(error: std::sync::Arc<nixe_memory::MemoryError>) -> Self {
        Self::Memory(error)
    }
}
