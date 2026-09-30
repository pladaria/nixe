use ash::vk;
use nixe_gpu::{SpirvFloat32Capabilities, SpirvFloat64Capabilities};

use super::fma;
use crate::{VulkanNativeCapabilities, VulkanTessellationLimits, WgpuBackendInitializationError};

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

#[cfg(test)]
static LIVE_IMPORTED_DEVICES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

pub(crate) struct CreatedDevice {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub capabilities: VulkanNativeCapabilities,
}

struct ImportedDevice {
    raw: ash::Device,
    // HAL can release its instance reference before calling its drop callback.
    // Retain the instance through vkDestroyDevice, including import failure.
    _instance: wgpu::Instance,
}

impl Drop for ImportedDevice {
    fn drop(&mut self) {
        // SAFETY: exactly one RAII owner; HAL's callback runs after its children
        // have been released. The captured owner also drops if HAL import fails.
        unsafe { self.raw.destroy_device(None) };
        #[cfg(test)]
        LIVE_IMPORTED_DEVICES.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
    }
}

fn device_error(error: impl std::fmt::Display) -> WgpuBackendInitializationError {
    WgpuBackendInitializationError::Device(format!("native Vulkan device: {error}").into())
}

/// Creates the backend's sole logical device. None means the selected adapter
/// cannot expose this native boundary; its ordinary wgpu path remains available.
/// The descriptor is the existing adapter-supported wgpu request, not a second
/// independently selected feature/limit policy.
pub(crate) fn create_device(
    instance: &wgpu::Instance,
    adapter: &wgpu::Adapter,
    descriptor: &wgpu::DeviceDescriptor<'_>,
) -> Result<Option<CreatedDevice>, WgpuBackendInitializationError> {
    // SAFETY: all raw queries and creation use this live adapter/instance. HAL's
    // required features/extensions are preserved and only supported native bits
    // are added. No wgpu call is made while holding its adapter HAL guard.
    // https://github.com/gfx-rs/wgpu/blob/v30.0.0/wgpu-hal/src/vulkan/adapter.rs
    let (open, capabilities) = unsafe {
        let Some(hal) = adapter.as_hal::<wgpu::hal::api::Vulkan>() else {
            return Ok(None);
        };
        let properties = hal.physical_device_capabilities().properties();
        if properties.api_version < vk::API_VERSION_1_1
            || hal.shared_instance().instance_api_version() < vk::API_VERSION_1_1
        {
            // The native emitter targets SPIR-V 1.3/Vulkan 1.1. Do not raise the
            // minimum version of ordinary wgpu rendering to match it.
            return Ok(None);
        }
        let raw_instance = hal.shared_instance().raw_instance();
        let physical = hal.raw_physical_device();
        // Match HAL 30's queue selection, including its presentation assumptions.
        let family = 0;
        let families = raw_instance.get_physical_device_queue_family_properties(physical);
        if !families.first().is_some_and(|f| {
            f.queue_count != 0
                && f.queue_flags
                    .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
        }) {
            return Ok(None);
        }
        let mut extensions = hal.required_device_extensions(descriptor.required_features);
        let supported = hal.physical_device_capabilities();
        let float_controls = properties.api_version >= vk::API_VERSION_1_2
            || supported.supports_extension(ash::khr::shader_float_controls::NAME);
        let mut floats = vk::PhysicalDeviceFloatControlsProperties::default();
        if float_controls {
            let mut query = vk::PhysicalDeviceProperties2::default().push_next(&mut floats);
            raw_instance.get_physical_device_properties2(physical, &mut query);
            if properties.api_version < vk::API_VERSION_1_2 {
                add_extension(&mut extensions, ash::khr::shader_float_controls::NAME);
            }
        }
        let mut fma_query = fma::Features::default();
        if supported.supports_extension(fma::NAME) {
            let mut query = vk::PhysicalDeviceFeatures2::default().push_next(&mut fma_query);
            raw_instance.get_physical_device_features2(physical, &mut query);
        }
        // Only float32 needs fused native arithmetic. Wide underflow repair
        // uses separate exact multiplication and compensated addition.
        let mut fma_enabled = fma::Features::default();
        fma_enabled.float32 = fma_query.float32;
        if fma_enabled.float32 != vk::FALSE {
            add_extension(&mut extensions, fma::NAME);
        }
        let mut features = hal.physical_device_features(&extensions, descriptor.required_features);
        // KHR and EXT expose the same feature structure. Keep the whole feature
        // family optional; lack of smooth lines must not disable ordinary wgpu.
        // https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_line_rasterization.html
        let line_extension = [
            ash::khr::line_rasterization::NAME,
            ash::ext::line_rasterization::NAME,
        ]
        .into_iter()
        .find(|name| supported.supports_extension(name));
        let mut line_features = vk::PhysicalDeviceLineRasterizationFeaturesKHR::default();
        if let Some(name) = line_extension {
            let mut query = vk::PhysicalDeviceFeatures2::default().push_next(&mut line_features);
            raw_instance.get_physical_device_features2(physical, &mut query);
            // Only enable the two modes represented by the neutral contract.
            line_features.bresenham_lines = vk::FALSE;
            line_features.stippled_rectangular_lines = vk::FALSE;
            line_features.stippled_bresenham_lines = vk::FALSE;
            line_features.stippled_smooth_lines = vk::FALSE;
            add_extension(&mut extensions, name);
        }
        let core = enabled_core(
            features.get_core(),
            hal.get_physical_device_features().get_core(),
        );
        let capabilities = VulkanNativeCapabilities {
            standard_sample_locations: properties.limits.standard_sample_locations != vk::FALSE,
            tessellation_shader: core.tessellation_shader != vk::FALSE,
            raster: crate::VulkanRasterCapabilities {
                wireframe: core.fill_mode_non_solid != vk::FALSE,
                wide_lines: core.wide_lines != vk::FALSE,
                rectangular_lines: line_features.rectangular_lines != vk::FALSE,
                smooth_lines: line_features.smooth_lines != vk::FALSE,
                line_width_range_bits: properties.limits.line_width_range.map(f32::to_bits),
            },
            float32: SpirvFloat32Capabilities {
                denorm_preserve: floats.shader_denorm_preserve_float32 != vk::FALSE,
                rounding_mode_rte: floats.shader_rounding_mode_rte_float32 != vk::FALSE,
                signed_zero_inf_nan_preserve: floats.shader_signed_zero_inf_nan_preserve_float32
                    != vk::FALSE,
                fused_multiply_add: fma_enabled.float32 != vk::FALSE,
            },
            float64: SpirvFloat64Capabilities {
                enabled: core.shader_float64 != vk::FALSE,
                rounding_mode_rte: floats.shader_rounding_mode_rte_float64 != vk::FALSE,
                signed_zero_inf_nan_preserve: floats.shader_signed_zero_inf_nan_preserve_float64
                    != vk::FALSE,
            },
            tessellation_limits: limits(properties.limits),
            graphics_limits: crate::VulkanGraphicsLimits {
                resources_per_stage: properties.limits.max_per_stage_resources,
                storage_buffers_per_stage: properties
                    .limits
                    .max_per_stage_descriptor_storage_buffers,
                storage_buffers_per_set: properties.limits.max_descriptor_set_storage_buffers,
                storage_buffer_range: properties.limits.max_storage_buffer_range,
                vertex_input_attributes: properties.limits.max_vertex_input_attributes,
                vertex_input_bindings: properties.limits.max_vertex_input_bindings,
                vertex_input_attribute_offset: properties.limits.max_vertex_input_attribute_offset,
                vertex_input_binding_stride: properties.limits.max_vertex_input_binding_stride,
                vertex_output_components: properties.limits.max_vertex_output_components,
                fragment_input_components: properties.limits.max_fragment_input_components,
            },
            robust_buffer_access: core.robust_buffer_access != vk::FALSE,
            full_draw_index_uint32: core.full_draw_index_uint32 != vk::FALSE,
            // Match HAL's public-source mapping, not the neutral format name:
            // https://github.com/gfx-rs/wgpu/blob/v30.0.0/wgpu-hal/src/vulkan/adapter.rs
            // https://github.com/gfx-rs/wgpu/blob/v30.0.0/wgpu-hal/src/vulkan/conv.rs
            depth24_stencil8_uses_float32: !raw_instance
                .get_physical_device_format_properties(physical, vk::Format::D24_UNORM_S8_UINT)
                .optimal_tiling_features
                .contains(
                    vk::FormatFeatureFlags::SAMPLED_IMAGE
                        | vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
                ),
        };
        let priorities = [1.0];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family)
            .queue_priorities(&priorities)];
        let names: Vec<_> = extensions.iter().map(|e| e.as_ptr()).collect();
        // add_to_device_create installs HAL's core pointer; replace it AFTER
        // this call or tessellationShader would silently be lost.
        let mut info = features
            .add_to_device_create(
                vk::DeviceCreateInfo::default()
                    .queue_create_infos(&queues)
                    .enabled_extension_names(&names),
            )
            .enabled_features(&core);
        if fma_enabled.float32 != vk::FALSE {
            info = info.push_next(&mut fma_enabled);
        }
        if line_extension.is_some() {
            info = info.push_next(&mut line_features);
        }
        let raw = raw_instance
            .create_device(physical, &info, None)
            .map_err(device_error)?;
        let owner = ImportedDevice {
            raw: raw.clone(),
            _instance: instance.clone(),
        };
        #[cfg(test)]
        LIVE_IMPORTED_DEVICES.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let open = hal
            .device_from_raw(
                raw,
                Some(Box::new(move || drop(owner))),
                &extensions,
                descriptor.required_features,
                &descriptor.required_limits,
                &descriptor.memory_hints,
                family,
                0,
            )
            .map_err(device_error)?;
        (open, capabilities)
    };
    // SAFETY: the imported HAL device was created from this adapter using the
    // exact descriptor and HAL's required extensions/features. Guard is released.
    let (device, queue) =
        unsafe { adapter.create_device_from_hal::<wgpu::hal::api::Vulkan>(open, descriptor) }
            .map_err(device_error)?;
    Ok(Some(CreatedDevice {
        device,
        queue,
        capabilities,
    }))
}

fn add_extension(extensions: &mut Vec<&'static std::ffi::CStr>, name: &'static std::ffi::CStr) {
    if !extensions.contains(&name) {
        extensions.push(name);
    }
}

fn enabled_core(
    mut required: vk::PhysicalDeviceFeatures,
    supported: vk::PhysicalDeviceFeatures,
) -> vk::PhysicalDeviceFeatures {
    required.tessellation_shader = supported.tessellation_shader;
    required.shader_float64 = supported.shader_float64;
    // Native uint32 indices stay resident; do not scan them to prove a reduced
    // maxDrawIndexedIndexValue on devices lacking the full-range feature.
    // https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceFeatures.html
    required.full_draw_index_uint32 = supported.full_draw_index_uint32;
    required.fill_mode_non_solid = supported.fill_mode_non_solid;
    required.wide_lines = supported.wide_lines;
    required
}

fn limits(l: vk::PhysicalDeviceLimits) -> VulkanTessellationLimits {
    VulkanTessellationLimits {
        generation_level: l.max_tessellation_generation_level,
        patch_size: l.max_tessellation_patch_size,
        control_per_vertex_input_components: l.max_tessellation_control_per_vertex_input_components,
        control_per_vertex_output_components: l
            .max_tessellation_control_per_vertex_output_components,
        control_per_patch_output_components: l.max_tessellation_control_per_patch_output_components,
        control_total_output_components: l.max_tessellation_control_total_output_components,
        evaluation_input_components: l.max_tessellation_evaluation_input_components,
        evaluation_output_components: l.max_tessellation_evaluation_output_components,
    }
}
