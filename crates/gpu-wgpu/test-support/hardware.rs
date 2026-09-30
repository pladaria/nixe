//! Shared test-only physical-GPU policy. Never execute a software rasterizer.
#![allow(dead_code)] // Each test binary uses a different subset of these helpers.

use nixe_gpu_wgpu::{HostBackend, InitializedWgpuBackend, WgpuBackendConfiguration};

pub fn is_physical(kind: wgpu::DeviceType) -> bool {
    matches!(
        kind,
        wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::IntegratedGpu
    )
}

pub fn adapter(instance: &wgpu::Instance, backends: wgpu::Backends) -> Option<wgpu::Adapter> {
    let adapters = pollster::block_on(instance.enumerate_adapters(backends));
    let selected = adapters
        .into_iter()
        .filter(|adapter| is_physical(adapter.get_info().device_type))
        .min_by_key(|adapter| match adapter.get_info().device_type {
            wgpu::DeviceType::DiscreteGpu => 0,
            _ => 1,
        });
    if selected.is_none() {
        eprintln!(
            "SKIP: no physical GPU for {backends:?}; software/virtual/unknown adapters are excluded"
        );
    }
    selected
}

pub fn available(backends: wgpu::Backends) -> bool {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends,
        flags: wgpu::InstanceFlags::empty(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    adapter(&instance, backends).is_some()
}

pub fn initialize_backend(
    instance_id: nixe_gpu::BackendInstanceId,
    device_id: nixe_memory::NonCpuDeviceId,
    configuration: WgpuBackendConfiguration,
) -> Option<InitializedWgpuBackend> {
    let backends = match configuration.host_backend {
        HostBackend::Vulkan => wgpu::Backends::VULKAN,
        HostBackend::Metal => wgpu::Backends::METAL,
        other => panic!("unsupported test backend: {other:?}"),
    };
    if !available(backends) {
        return None;
    }
    let backend = nixe_gpu_wgpu::initialize_backend(instance_id, device_id, configuration)
        .expect("physical GPU exists: backend initialization errors must fail, not skip");
    assert!(
        is_physical(
            backend
                .presentation_context()
                .device()
                .adapter_info()
                .device_type
        ),
        "production selected a nonphysical adapter despite an available physical GPU"
    );
    Some(backend)
}

pub fn native_capabilities(wireframe: bool) -> Option<nixe_gpu_wgpu::VulkanNativeCapabilities> {
    let backend = initialize_backend(
        nixe_gpu::BackendInstanceId::new(1),
        nixe_memory::NonCpuDeviceId::new(1),
        WgpuBackendConfiguration {
            host_backend: HostBackend::Vulkan,
            pipeline_cache_directory: None,
            ..Default::default()
        },
    )?;
    let Some(caps) = backend.adapter.native_vulkan else {
        eprintln!("SKIP: physical GPU has no native Vulkan interop support");
        return None;
    };
    if !caps.tessellation_shader {
        eprintln!("SKIP: physical GPU lacks tessellationShader");
        return None;
    }
    if wireframe
        && !(caps.raster.wireframe
            && caps.raster.wide_lines
            && caps.raster.rectangular_lines
            && caps.raster.smooth_lines
            && f32::from_bits(caps.raster.line_width_range_bits[0]) <= 1.0
            && f32::from_bits(caps.raster.line_width_range_bits[1]) >= 4.0)
    {
        eprintln!(
            "SKIP: physical GPU lacks required width-1/4 rectangular/smooth wireframe: {:?}",
            caps.raster
        );
        return None;
    }
    Some(caps)
}

#[test]
fn adapter_policy_excludes_software_virtual_and_unknown_devices() {
    assert!(is_physical(wgpu::DeviceType::DiscreteGpu));
    assert!(is_physical(wgpu::DeviceType::IntegratedGpu));
    assert!(!is_physical(wgpu::DeviceType::Cpu));
    assert!(!is_physical(wgpu::DeviceType::VirtualGpu));
    assert!(!is_physical(wgpu::DeviceType::Other));
}
