//! Accelerated `wgpu` implementation of Nixe's host-independent GPU contract.
//!
//! This crate is the only place where neutral resources become `wgpu` objects.
//! Switch profiles and Maxwell packets deliberately remain outside this layer.

mod driver;
mod visibility;

use std::fmt::{Display, Formatter};
use std::sync::Arc;

use driver::WgpuBackendDriver;
use visibility::WgpuVisibilityCoordinator;

use nixe_gpu::{
    Backend, BackendCapabilities, BackendFeatures, BackendInstanceId, BackendLimits,
    BackendRuntime, ImageFormat, NeutralBackendRuntime, QueryKind, SampleCount, ShaderStage,
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
            host_backend: default_host_backend(),
            power_preference: WgpuPowerPreference::HighPerformance,
            force_fallback_adapter: false,
        }
    }
}

const fn default_host_backend() -> HostBackend {
    #[cfg(target_os = "macos")]
    {
        HostBackend::Metal
    }
    #[cfg(not(target_os = "macos"))]
    {
        HostBackend::Vulkan
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
    runtime: Box<dyn NeutralBackendRuntime>,
    pub adapter: WgpuAdapterInformation,
}

impl InitializedWgpuBackend {
    /// Erases only the concrete host driver type for injection into a console
    /// frontend. The resulting runtime still exposes exclusively neutral GPU
    /// contracts.
    #[must_use]
    pub fn into_runtime(self) -> Box<dyn NeutralBackendRuntime> {
        self.runtime
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
    if !matches!(
        configuration.host_backend,
        HostBackend::Vulkan | HostBackend::Metal
    ) {
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
    let required_features = requested_device_features(adapter.features());
    let (device, queue) = adapter
        .request_device(&DeviceDescriptor {
            label: Some("Nixe accelerated GPU backend"),
            required_features,
            required_limits: required_limits.clone(),
            experimental_features: ExperimentalFeatures::disabled(),
            memory_hints: MemoryHints::Performance,
            trace: Trace::Off,
        })
        .await
        .map_err(|error| WgpuBackendInitializationError::Device(error.to_string().into()))?;
    let capabilities = capabilities(&adapter, &required_limits, required_features);
    let visibility = Arc::new(WgpuVisibilityCoordinator::new(device_id));
    let driver = WgpuBackendDriver::new(device, queue, Arc::clone(&visibility));
    let backend = Backend::new(instance_id, capabilities, driver);
    let visibility: Arc<dyn nixe_memory::VisibilityCoordinator> = visibility;
    Ok(InitializedWgpuBackend {
        runtime: Box::new(BackendRuntime::new(backend, device_id, visibility)),
        adapter: WgpuAdapterInformation {
            name: info.name.into(),
            driver: info.driver.into(),
            backend: configuration.host_backend,
        },
    })
}

fn requested_device_features(adapter_features: wgpu::Features) -> wgpu::Features {
    adapter_features & wgpu::Features::FLOAT32_FILTERABLE
}

fn required_features_for_image_format(format: ImageFormat) -> wgpu::Features {
    match format {
        // The neutral format capability currently covers every supported use of
        // a format. Keep the float32 family out of that contract unless wgpu can
        // also represent its filtered sampled-image use exactly.
        ImageFormat::R32Float | ImageFormat::Rg32Float | ImageFormat::Rgba32Float => {
            wgpu::Features::FLOAT32_FILTERABLE
        }
        _ => wgpu::Features::empty(),
    }
}

fn capabilities(
    adapter: &wgpu::Adapter,
    limits: &wgpu::Limits,
    enabled_features: wgpu::Features,
) -> BackendCapabilities {
    let formats = ALL_IMAGE_FORMATS.into_iter().filter(|format| {
        if !enabled_features.contains(required_features_for_image_format(*format)) {
            return false;
        }
        let Some(host) = driver::texture_format(*format) else {
            return false;
        };
        let allowed = adapter.get_texture_format_features(host).allowed_usages;
        allowed.contains(driver::required_texture_usages(*format))
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
    fn device_enables_float32_filtering_only_when_the_adapter_supports_it() {
        assert_eq!(
            requested_device_features(wgpu::Features::FLOAT32_FILTERABLE),
            wgpu::Features::FLOAT32_FILTERABLE
        );
        assert!(requested_device_features(wgpu::Features::empty()).is_empty());
    }

    #[test]
    fn float32_image_formats_require_the_filtering_feature() {
        for format in [
            ImageFormat::R32Float,
            ImageFormat::Rg32Float,
            ImageFormat::Rgba32Float,
        ] {
            assert_eq!(
                required_features_for_image_format(format),
                wgpu::Features::FLOAT32_FILTERABLE
            );
        }
        assert!(required_features_for_image_format(ImageFormat::Rgba16Float).is_empty());
        assert!(required_features_for_image_format(ImageFormat::Rgba8Unorm).is_empty());
    }

    #[test]
    fn unsupported_backend_selection_is_explicitly_rejected_in_this_build() {
        let error = initialize_backend(
            BackendInstanceId::new(1),
            NonCpuDeviceId::new(1),
            WgpuBackendConfiguration {
                host_backend: HostBackend::Direct3D12,
                ..WgpuBackendConfiguration::default()
            },
        )
        .err()
        .expect("Direct3D 12 is intentionally not compiled");
        assert_eq!(
            error,
            WgpuBackendInitializationError::BackendNotCompiled(HostBackend::Direct3D12)
        );
    }
}
