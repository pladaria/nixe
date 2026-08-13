//! Accelerated `wgpu` implementation of Nixe's host-independent GPU contract.
//!
//! This crate is the only place where neutral resources become `wgpu` objects.
//! Switch profiles and Maxwell packets deliberately remain outside this layer.

mod driver;
mod visibility;

use std::fmt::{Display, Formatter};
use std::sync::Arc;

pub use driver::WgpuBackendDriver;
pub use visibility::WgpuVisibilityCoordinator;

use nixe_gpu::{
    Backend, BackendCapabilities, BackendFeatures, BackendInstanceId, BackendLimits, ImageFormat,
    NeutralBackendRuntime, QueryKind, SampleCount, ShaderStage, SynchronousBackendRuntime,
};
use nixe_memory::NonCpuDeviceId;
use wgpu::{
    Backends, DeviceDescriptor, ExperimentalFeatures, Instance, InstanceDescriptor, MemoryHints,
    PowerPreference, RequestAdapterOptions, Trace,
};

/// Host API requested for accelerated execution.
///
/// Variants remain explicit even when a build only contains Vulkan, so adding
/// Metal or another `wgpu` backend never changes the neutral GPU contract.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HostBackend {
    Vulkan,
    Metal,
    Direct3D12,
    OpenGl,
}

impl HostBackend {
    const fn wgpu(self) -> Backends {
        match self {
            Self::Vulkan => Backends::VULKAN,
            Self::Metal => Backends::METAL,
            Self::Direct3D12 => Backends::DX12,
            Self::OpenGl => Backends::GL,
        }
    }
}

/// Immutable backend initialization policy selected by the composition root.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WgpuBackendConfiguration {
    pub host_backend: HostBackend,
    pub power_preference: WgpuPowerPreference,
    pub force_fallback_adapter: bool,
}

impl Default for WgpuBackendConfiguration {
    fn default() -> Self {
        Self {
            host_backend: HostBackend::Vulkan,
            power_preference: WgpuPowerPreference::HighPerformance,
            force_fallback_adapter: false,
        }
    }
}

/// API-independent adapter power preference.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WgpuPowerPreference {
    None,
    LowPower,
    HighPerformance,
}

impl WgpuPowerPreference {
    const fn wgpu(self) -> PowerPreference {
        match self {
            Self::None => PowerPreference::None,
            Self::LowPower => PowerPreference::LowPower,
            Self::HighPerformance => PowerPreference::HighPerformance,
        }
    }
}

/// Stable host-adapter information useful for diagnostics and captures.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WgpuAdapterInformation {
    pub name: Box<str>,
    pub driver: Box<str>,
    pub backend: HostBackend,
}

/// Result of accelerated backend initialization.
pub struct InitializedWgpuBackend {
    pub backend: Backend<WgpuBackendDriver>,
    pub visibility: Arc<WgpuVisibilityCoordinator>,
    pub adapter: WgpuAdapterInformation,
}

impl InitializedWgpuBackend {
    /// Erases only the concrete host driver type for injection into a console
    /// frontend. The resulting runtime still exposes exclusively neutral GPU
    /// contracts.
    #[must_use]
    pub fn into_runtime(self) -> Box<dyn NeutralBackendRuntime> {
        let device = self.visibility.device();
        let visibility: Arc<dyn nixe_memory::VisibilityCoordinator> = self.visibility;
        Box::new(SynchronousBackendRuntime::new(
            self.backend,
            device,
            visibility,
        ))
    }
}

/// Creates a validated accelerated backend with immutable host capabilities.
pub fn initialize_backend(
    instance_id: BackendInstanceId,
    device_id: NonCpuDeviceId,
    configuration: WgpuBackendConfiguration,
) -> Result<InitializedWgpuBackend, WgpuBackendInitializationError> {
    pollster::block_on(initialize_backend_async(
        instance_id,
        device_id,
        configuration,
    ))
}

async fn initialize_backend_async(
    instance_id: BackendInstanceId,
    device_id: NonCpuDeviceId,
    configuration: WgpuBackendConfiguration,
) -> Result<InitializedWgpuBackend, WgpuBackendInitializationError> {
    if configuration.host_backend != HostBackend::Vulkan {
        return Err(WgpuBackendInitializationError::BackendNotCompiled(
            configuration.host_backend,
        ));
    }
    let mut descriptor = InstanceDescriptor::new_without_display_handle();
    descriptor.backends = configuration.host_backend.wgpu();
    let instance = Instance::new(descriptor);
    let adapter = instance
        .request_adapter(&RequestAdapterOptions {
            power_preference: configuration.power_preference.wgpu(),
            force_fallback_adapter: configuration.force_fallback_adapter,
            compatible_surface: None,
            apply_limit_buckets: false,
        })
        .await
        .map_err(|error| WgpuBackendInitializationError::Adapter(error.to_string().into()))?;
    let info = adapter.get_info();
    let required_limits = adapter.limits();
    let (device, queue) = adapter
        .request_device(&DeviceDescriptor {
            label: Some("Nixe accelerated GPU backend"),
            required_features: wgpu::Features::empty(),
            required_limits: required_limits.clone(),
            experimental_features: ExperimentalFeatures::disabled(),
            memory_hints: MemoryHints::Performance,
            trace: Trace::Off,
        })
        .await
        .map_err(|error| WgpuBackendInitializationError::Device(error.to_string().into()))?;
    let capabilities = capabilities(&adapter, &required_limits);
    let visibility = Arc::new(WgpuVisibilityCoordinator::new(device_id));
    let driver = WgpuBackendDriver::new(device, queue, Arc::clone(&visibility));
    Ok(InitializedWgpuBackend {
        backend: Backend::new(instance_id, capabilities, driver),
        visibility,
        adapter: WgpuAdapterInformation {
            name: info.name.into(),
            driver: info.driver.into(),
            backend: configuration.host_backend,
        },
    })
}

fn capabilities(adapter: &wgpu::Adapter, limits: &wgpu::Limits) -> BackendCapabilities {
    let formats = ALL_IMAGE_FORMATS.into_iter().filter(|format| {
        let Some(host) = driver::texture_format(*format) else {
            return false;
        };
        let allowed = adapter.get_texture_format_features(host).allowed_usages;
        allowed.contains(
            wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
        )
    });
    BackendCapabilities::new(
        BackendFeatures::COPY
            .union(BackendFeatures::CLEAR)
            .union(BackendFeatures::DRAW)
            .union(BackendFeatures::INDEXED_DRAW)
            .union(BackendFeatures::BARRIER)
            .union(BackendFeatures::RENDER_PASS),
        formats,
        [SampleCount::One],
        [ShaderStage::Vertex, ShaderStage::Fragment],
        std::iter::empty::<QueryKind>(),
        BackendLimits {
            max_color_attachments: limits.max_color_attachments.min(u32::from(u8::MAX)) as u8,
            max_descriptor_bindings: limits.max_bindings_per_bind_group,
            max_compute_workgroups: [limits.max_compute_workgroups_per_dimension; 3],
        },
    )
}

const ALL_IMAGE_FORMATS: [ImageFormat; 16] = [
    ImageFormat::R8Unorm,
    ImageFormat::Rg8Unorm,
    ImageFormat::Rgba8Unorm,
    ImageFormat::Rgba8Srgb,
    ImageFormat::Bgra8Unorm,
    ImageFormat::Bgra8Srgb,
    ImageFormat::R16Float,
    ImageFormat::Rg16Float,
    ImageFormat::Rgba16Float,
    ImageFormat::R32Float,
    ImageFormat::Rg32Float,
    ImageFormat::Rgba32Float,
    ImageFormat::Depth16Unorm,
    ImageFormat::Depth24UnormStencil8Uint,
    ImageFormat::Depth32Float,
    ImageFormat::Depth32FloatStencil8Uint,
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WgpuBackendInitializationError {
    BackendNotCompiled(HostBackend),
    Adapter(Box<str>),
    Device(Box<str>),
}

impl Display for WgpuBackendInitializationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BackendNotCompiled(backend) => {
                write!(
                    formatter,
                    "requested wgpu backend is not compiled: {backend:?}"
                )
            }
            Self::Adapter(error) => write!(formatter, "wgpu adapter selection failed: {error}"),
            Self::Device(error) => write!(formatter, "wgpu device creation failed: {error}"),
        }
    }
}

impl std::error::Error for WgpuBackendInitializationError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_vulkan_selection_is_explicitly_rejected_in_this_build() {
        let error = initialize_backend(
            BackendInstanceId::new(1),
            NonCpuDeviceId::new(1),
            WgpuBackendConfiguration {
                host_backend: HostBackend::Metal,
                ..WgpuBackendConfiguration::default()
            },
        )
        .err()
        .expect("Metal is intentionally not compiled");
        assert_eq!(
            error,
            WgpuBackendInitializationError::BackendNotCompiled(HostBackend::Metal)
        );
    }
}
