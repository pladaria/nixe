use ash::vk;
use std::fmt::{Display, Formatter};

#[derive(Debug)]
pub enum VulkanError {
    Loader(Box<str>),
    Compilation(Box<str>),
    TextureDecode(Box<str>),
    ExternalVideo(Box<str>),
    Vulkan {
        operation: &'static str,
        result: vk::Result,
    },
    Allocation(gpu_allocator::AllocationError),
    WindowHandle(Box<str>),
    MissingInstanceExtension(String),
    SurfaceMaintenanceUnavailable,
    SwapchainMaintenanceUnavailable,
    ValidationUnavailable,
    ApiVersion {
        available: u32,
    },
    NoSuitableDevice(Vec<String>),
    InvalidResource(&'static str),
    Validation(Vec<String>),
    QueuePressure {
        pending: Option<crate::WorkTicket>,
        retained_readbacks: usize,
    },
    StagingRequestTooLarge {
        requested: u64,
        maximum: u64,
    },
    QueueStopped,
    SubmissionFailed {
        tick: u64,
        cause: Box<VulkanError>,
    },
    DescriptorPressure,
    Memory(std::sync::Arc<nixe_memory::MemoryError>),
    ResidencyPressure {
        requested: u64,
        allocated: u64,
        budget: u64,
    },
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
            Self::ExternalVideo(e) => write!(f,"Vulkan external video: {e}"),
            Self::Compilation(e) => write!(f,"Vulkan shader/pipeline compilation: {e}"),
            Self::Loader(e) => write!(f, "cannot load the Vulkan loader: {e}"),
            Self::Vulkan { operation, result } => write!(f, "{operation} failed: {result:?}"),
            Self::Allocation(e) => write!(f, "Vulkan allocation failed: {e}"),
            Self::WindowHandle(e) => write!(f, "cannot obtain native window handles: {e}"),
            Self::MissingInstanceExtension(e) => write!(f, "Vulkan instance extension required by the surface is unavailable: {e}"),
            // First upstream loader extension allowlist with EXT surface maintenance:
            // https://github.com/KhronosGroup/Vulkan-Loader/blob/v1.3.237/loader/generated/vk_loader_extensions.c
            Self::SurfaceMaintenanceUnavailable => f.write_str("The Vulkan runtime does not expose VK_EXT_surface_maintenance1 or VK_KHR_surface_maintenance1, required for windowed rendering. Use Khronos Vulkan Loader 1.3.237 or newer (EXT support) and a compatible GPU driver."),
            Self::SwapchainMaintenanceUnavailable => f.write_str("The selected GPU/driver does not provide the swapchainMaintenance1 feature with compatible KHR/EXT surface extensions, required for windowed rendering. Update your GPU driver or select a supported GPU."),
            Self::ValidationUnavailable => f.write_str("Vulkan validation requested, but VK_LAYER_KHRONOS_validation or VK_EXT_debug_utils is unavailable; install a Vulkan SDK/validation layer or configure VK_LAYER_PATH"),
            Self::ApiVersion { available } => write!(f, "Vulkan 1.2 or newer is required; loader provides {}.{}", vk::api_version_major(*available), vk::api_version_minor(*available)),
            Self::NoSuitableDevice(reasons) => write!(f, "no device meets the Vulkan profile: {}", reasons.join("; ")),
            Self::InvalidResource(e) => write!(f, "invalid Vulkan resource: {e}"),
            Self::Validation(e) => write!(f, "Vulkan validation reported errors: {}", e.join("; ")),
            Self::QueuePressure { pending, retained_readbacks } => write!(f, "Vulkan staging/command budget is full; pending work: {pending:?}, unread readbacks: {retained_readbacks}"),
            Self::StagingRequestTooLarge { requested, maximum } => write!(f, "Vulkan staging request of {requested} bytes exceeds the bounded allocation size {maximum}; split the transfer into chunks"),
            Self::Memory(error) => write!(f, "Vulkan physical memory: {error}"),
            Self::ResidencyPressure { requested, allocated, budget } => write!(f, "Vulkan residency budget exhausted: requested {requested}, live/in-flight allocations {allocated}, budget {budget}; authoritative or leased contents cannot be discarded"),
            Self::DescriptorPressure => f.write_str("Vulkan descriptor slot budget is full; flush and continue in a recycled command slot"),
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
