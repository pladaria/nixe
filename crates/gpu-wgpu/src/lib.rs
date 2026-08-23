//! Accelerated `wgpu` implementation of Nixe's host-independent GPU contract.
//!
//! This crate is the only place where neutral resources become `wgpu` objects.
//! Switch profiles and Maxwell packets deliberately remain outside this layer.

mod driver;
mod visibility;

use std::fmt::{Display, Formatter};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use driver::WgpuBackendDriver;
use visibility::WgpuVisibilityCoordinator;

use nixe_gpu::{
    Backend, BackendCapabilities, BackendFeatures, BackendInstanceId, BackendLimits,
    BackendRuntime, GpuCacheConfiguration, ImageFormat, NeutralBackendRuntime, QueryKind,
    ResidentImage, SampleCount, ShaderStage,
};
use nixe_memory::NonCpuDeviceId;
use wgpu::{
    Adapter, Backends, Device, DeviceDescriptor, ExperimentalFeatures, Instance,
    InstanceDescriptor, MemoryHints, PowerPreference, Queue, RequestAdapterOptions, Trace,
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
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WgpuBackendConfiguration {
    pub host_backend: HostBackend,
    pub power_preference: WgpuPowerPreference,
    pub force_fallback_adapter: bool,
    pub pipeline_cache_directory: Option<PathBuf>,
    pub cache: GpuCacheConfiguration,
}

impl Default for WgpuBackendConfiguration {
    fn default() -> Self {
        Self {
            host_backend: default_host_backend(),
            power_preference: WgpuPowerPreference::HighPerformance,
            force_fallback_adapter: false,
            pipeline_cache_directory: default_pipeline_cache_directory(),
            cache: GpuCacheConfiguration::default(),
        }
    }
}

fn default_pipeline_cache_directory() -> Option<PathBuf> {
    #[cfg(target_os = "windows")]
    {
        std::env::var_os("LOCALAPPDATA")
            .map(PathBuf::from)
            .map(|root| root.join("Nixe").join("Cache"))
    }
    #[cfg(target_os = "macos")]
    {
        std::env::var_os("HOME")
            .map(PathBuf::from)
            .map(|root| root.join("Library").join("Caches").join("Nixe"))
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        std::env::var_os("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|root| PathBuf::from(root).join(".cache")))
            .map(|root| root.join("nixe"))
    }
    #[cfg(not(any(unix, target_os = "windows")))]
    {
        None
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
    presentation: WgpuPresentationContext,
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

    /// Returns the host context shared with the window presenter. Cloning these
    /// WGPU handles retains the same device and queue; it never creates a
    /// second host graphics device.
    #[must_use]
    pub fn presentation_context(&self) -> WgpuPresentationContext {
        self.presentation.clone()
    }
}

/// WGPU objects shared by accelerated execution and host presentation.
#[derive(Clone, Debug)]
pub struct WgpuPresentationContext {
    backend: BackendInstanceId,
    instance: Instance,
    adapter: Adapter,
    device: Device,
    queue: Queue,
}

impl WgpuPresentationContext {
    #[must_use]
    pub const fn backend(&self) -> BackendInstanceId {
        self.backend
    }

    #[must_use]
    pub const fn instance(&self) -> &Instance {
        &self.instance
    }

    #[must_use]
    pub const fn adapter(&self) -> &Adapter {
        &self.adapter
    }

    #[must_use]
    pub const fn device(&self) -> &Device {
        &self.device
    }

    #[must_use]
    pub const fn queue(&self) -> &Queue {
        &self.queue
    }
}

#[derive(Debug)]
pub(crate) struct WgpuResidentImage {
    texture: wgpu::Texture,
}

impl WgpuResidentImage {
    pub(crate) fn new(texture: wgpu::Texture) -> Self {
        Self { texture }
    }
}

/// Resolves an opaque resident image for the matching WGPU presenter.
///
/// A different backend payload returns `None`; concrete WGPU objects never
/// cross the neutral GPU or Horizon contracts.
#[must_use]
pub fn resident_texture(image: &ResidentImage) -> Option<&wgpu::Texture> {
    image
        .payload::<WgpuResidentImage>()
        .map(|resident| &resident.texture)
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
    let pipeline_cache_path = configuration
        .pipeline_cache_directory
        .as_deref()
        .map(|directory| pipeline_cache_path(directory, configuration.host_backend, &info));
    let pipeline_cache = if required_features.contains(wgpu::Features::PIPELINE_CACHE) {
        let data = pipeline_cache_path
            .as_deref()
            .map(|path| {
                load_pipeline_cache(path, configuration.cache.persistent_pipeline_cache_bytes())
            })
            .transpose()?
            .flatten();
        if let (Some(path), Some(data)) = (&pipeline_cache_path, &data) {
            log::info!(
                "loaded WGPU pipeline cache: path={} bytes={}",
                path.display(),
                data.len()
            );
        }
        // SAFETY: cache bytes are read only from Nixe's private, versioned cache
        // file and that file contains only data previously returned by this
        // `wgpu` API. `fallback` rejects incompatible adapter or driver data.
        Some(unsafe {
            device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                label: Some("Nixe persistent pipeline cache"),
                data: data.as_deref(),
                fallback: true,
            })
        })
    } else {
        log::debug!("WGPU adapter does not expose persistent pipeline-cache support");
        None
    };
    let presentation = WgpuPresentationContext {
        backend: instance_id,
        instance: instance.clone(),
        adapter: adapter.clone(),
        device: device.clone(),
        queue: queue.clone(),
    };
    let driver = WgpuBackendDriver::new(
        device,
        queue,
        Arc::clone(&visibility),
        pipeline_cache,
        pipeline_cache_path,
        configuration.cache,
    );
    let backend = Backend::new(instance_id, capabilities, driver);
    let visibility: Arc<dyn nixe_memory::VisibilityCoordinator> = visibility;
    Ok(InitializedWgpuBackend {
        runtime: Box::new(BackendRuntime::new(backend, device_id, visibility)),
        presentation,
        adapter: WgpuAdapterInformation {
            name: info.name.into(),
            driver: info.driver.into(),
            backend: configuration.host_backend,
        },
    })
}

fn requested_device_features(adapter_features: wgpu::Features) -> wgpu::Features {
    adapter_features & (wgpu::Features::FLOAT32_FILTERABLE | wgpu::Features::PIPELINE_CACHE)
}

const PIPELINE_CACHE_MAGIC: &[u8] = b"NIXE-WGPU-CACHE\x01";
fn pipeline_cache_path(
    directory: &Path,
    backend: HostBackend,
    adapter: &wgpu::AdapterInfo,
) -> PathBuf {
    let backend = match backend {
        HostBackend::Vulkan => "vulkan",
        HostBackend::Metal => "metal",
        HostBackend::Direct3D12 => "d3d12",
        HostBackend::OpenGl => "opengl",
    };
    directory.join(format!(
        "wgpu-30-{backend}-{:08x}-{:08x}.bin",
        adapter.vendor, adapter.device
    ))
}

fn load_pipeline_cache(
    path: &Path,
    maximum_bytes: u64,
) -> Result<Option<Vec<u8>>, WgpuBackendInitializationError> {
    let file = match std::fs::File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            log::debug!(
                "no persistent WGPU pipeline cache found: path={}",
                path.display()
            );
            return Ok(None);
        }
        Err(error) => return Err(cache_error(path, error)),
    };
    let bound = maximum_bytes
        .checked_add(PIPELINE_CACHE_MAGIC.len() as u64)
        .ok_or_else(|| {
            WgpuBackendInitializationError::PipelineCache(
                "configured pipeline cache bound cannot be represented".into(),
            )
        })?;
    if file
        .metadata()
        .map_err(|error| cache_error(path, error))?
        .len()
        > bound
    {
        return Err(WgpuBackendInitializationError::PipelineCache(
            format!(
                "pipeline cache is larger than its configured {} byte bound: {}",
                maximum_bytes,
                path.display(),
            )
            .into(),
        ));
    }
    let mut data = Vec::new();
    use std::io::Read;
    file.take(bound.checked_add(1).ok_or_else(|| {
        WgpuBackendInitializationError::PipelineCache(
            "configured pipeline cache read bound cannot be represented".into(),
        )
    })?)
    .read_to_end(&mut data)
    .map_err(|error| cache_error(path, error))?;
    if data.len() as u64 > bound {
        return Err(WgpuBackendInitializationError::PipelineCache(
            format!(
                "pipeline cache grew beyond its configured {} byte bound: {}",
                maximum_bytes,
                path.display(),
            )
            .into(),
        ));
    }
    let Some(data) = data.strip_prefix(PIPELINE_CACHE_MAGIC) else {
        log::warn!(
            "ignoring incompatible WGPU pipeline cache: path={}",
            path.display()
        );
        return Ok(None);
    };
    Ok(Some(data.to_vec()))
}

fn cache_error(path: &Path, error: std::io::Error) -> WgpuBackendInitializationError {
    WgpuBackendInitializationError::PipelineCache(
        format!("cannot read pipeline cache {}: {error}", path.display()).into(),
    )
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
    PipelineCache(Box<str>),
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
            Self::PipelineCache(error) => write!(formatter, "wgpu pipeline cache failed: {error}"),
        }
    }
}

impl std::error::Error for WgpuBackendInitializationError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_cache_loader_accepts_only_versioned_bounded_payloads() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("pipeline.bin");
        assert_eq!(load_pipeline_cache(&path, 1024).unwrap(), None);

        std::fs::write(&path, b"unrelated cache data").unwrap();
        assert_eq!(load_pipeline_cache(&path, 1024).unwrap(), None);

        let payload = b"driver-owned-cache";
        let mut stored = PIPELINE_CACHE_MAGIC.to_vec();
        stored.extend_from_slice(payload);
        std::fs::write(&path, stored).unwrap();
        assert_eq!(
            load_pipeline_cache(&path, 1024).unwrap().as_deref(),
            Some(payload.as_slice())
        );

        std::fs::write(&path, vec![0; 1025 + PIPELINE_CACHE_MAGIC.len()]).unwrap();
        let error = load_pipeline_cache(&path, 1024).unwrap_err();
        assert!(error.to_string().contains("larger than its configured"));
    }

    #[test]
    fn device_enables_optional_features_only_when_the_adapter_supports_them() {
        assert_eq!(
            requested_device_features(
                wgpu::Features::FLOAT32_FILTERABLE | wgpu::Features::PIPELINE_CACHE
            ),
            wgpu::Features::FLOAT32_FILTERABLE | wgpu::Features::PIPELINE_CACHE
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
