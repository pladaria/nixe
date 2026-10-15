//! Immutable host facts and initialization-time choices, never guest semantics.
use crate::extensions::{self, ShaderFma, UnifiedLayouts};
use ash::{Instance, vk};
use std::{collections::HashSet, ffi::CStr};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ApiPath {
    Core13,
    Khr12,
}
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ApiPreference {
    #[default]
    Auto,
    Compatibility12,
}
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SwapchainMaintenance {
    #[default]
    Disabled,
    Khr,
    Ext,
}

#[derive(Debug)]
pub struct DeviceProfile {
    pub name: String,
    pub device_type: vk::PhysicalDeviceType,
    pub api_version: u32,
    pub api_path: ApiPath,
    pub driver_version: u32,
    pub vendor_id: u32,
    pub device_id: u32,
    pub device_uuid: [u8; 16],
    pub pipeline_cache_uuid: [u8; 16],
    pub pipeline_creation_feedback: bool,
    pub pipeline_creation_cache_control: bool,
    pub limits: vk::PhysicalDeviceLimits,
    pub core: vk::PhysicalDeviceFeatures,
    pub float_controls: FloatControls,
    pub subgroup: SubgroupSupport,
    pub dynamic_state: DynamicStateSupport,
    pub graphics_pipeline_library: bool,
    pub graphics_pipeline_library_fast_linking: bool,
    pub graphics_pipeline_library_independent_interpolation: bool,
    pub descriptor_indexing: bool,
    pub runtime_descriptor_array: bool,
    pub buffer_device_address: bool,
    pub descriptor_buffer: bool,
    pub descriptor_buffer_image_layout_ignored: bool,
    pub descriptor_buffer_offset_alignment: u64,
    pub storage_buffer_descriptor_size: usize,
    pub separate_depth_stencil_layouts: bool,
    pub shader_object: bool,
    pub transform_feedback: bool,
    pub provoking_vertex_last: bool,
    pub index_uint8: bool,
    pub fill_rectangle: bool,
    pub custom_border_color: bool,
    pub custom_border_color_without_format: bool,
    pub sampler_filter_minmax: bool,
    pub rectangular_lines: bool,
    pub smooth_lines: bool,
    pub bresenham_lines: bool,
    pub unified_image_layouts: bool,
    pub swapchain_maintenance: SwapchainMaintenance,
    pub swapchain_mutable_format: bool,
    pub portability: Option<PortabilitySupport>,
    /// Enabled API extensions; resource-specific import/export support is queried
    /// when an external allocation is created, not inferred from these bits.
    pub external_memory_fd: bool,
    pub external_semaphore_fd: bool,
    pub enabled_extensions: Vec<String>,
}
#[derive(Debug)]
pub struct FloatControls {
    pub float32_denorm_preserve: bool,
    pub float32_denorm_flush_to_zero: bool,
    pub float32_signed_zero_inf_nan_preserve: bool,
    pub float32_round_to_nearest_even: bool,
    pub float32_round_toward_zero: bool,
    pub float64_denorm_preserve: bool,
    pub float64_signed_zero_inf_nan_preserve: bool,
    pub float64_round_to_nearest_even: bool,
    pub fma_float16: bool,
    pub fma_float32: bool,
    pub fma_float64: bool,
}
#[derive(Debug)]
pub struct SubgroupSupport {
    pub size: u32,
    pub min_size: u32,
    pub max_size: u32,
    pub stages: vk::ShaderStageFlags,
    pub operations: vk::SubgroupFeatureFlags,
    pub required_size_stages: vk::ShaderStageFlags,
    pub size_control: bool,
    pub compute_full_subgroups: bool,
}
#[derive(Debug)]
pub struct DynamicStateSupport {
    pub base: bool,
    pub base2: bool,
    pub logic_op: bool,
    pub patch_control_points: bool,
    pub vertex_input: bool,
    pub polygon_mode: bool,
    pub depth_clamp_enable: bool,
    pub rasterization_samples: bool,
    pub sample_mask: bool,
    pub alpha_to_coverage_enable: bool,
    pub color_blend_enable: bool,
    pub color_blend_equation: bool,
    pub color_write_mask: bool,
}
#[derive(Debug)]
pub struct PortabilitySupport {
    pub triangle_fans: bool,
    pub image_view_format_swizzle: bool,
    pub mutable_comparison_samplers: bool,
    pub events: bool,
    pub tessellation_isolines: bool,
    pub tessellation_point_mode: bool,
    pub separate_stencil_mask_ref: bool,
}

pub(crate) struct ExtensionSet(HashSet<String>);
impl ExtensionSet {
    pub fn new(properties: &[vk::ExtensionProperties]) -> Self {
        Self(
            properties
                .iter()
                .map(|p| {
                    // SAFETY: Vulkan returns a NUL-terminated extension name array.
                    unsafe { CStr::from_ptr(p.extension_name.as_ptr()) }
                        .to_string_lossy()
                        .into_owned()
                })
                .collect(),
        )
    }
    pub fn contains(&self, name: &CStr) -> bool {
        self.0
            .contains(name.to_str().expect("static Vulkan name is UTF-8"))
    }
}

/// Prefer a unified family; use a separate present family only when required.
pub(crate) fn choose_queues(
    families: &[vk::QueueFamilyProperties],
    present: Option<&[bool]>,
) -> Option<(u32, Option<u32>)> {
    let graphics = |i: usize| {
        families[i].queue_count != 0
            && families[i]
                .queue_flags
                .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
    };
    if let Some(present) = present {
        if let Some(i) = (0..families.len())
            .find(|&i| graphics(i) && present[i] && families[i].queue_count >= 2)
            .or_else(|| (0..families.len()).find(|&i| graphics(i) && present[i]))
        {
            return Some((i as u32, Some(i as u32)));
        }
        let graphics = (0..families.len()).find(|&i| graphics(i))?;
        let present = (0..families.len()).find(|&i| families[i].queue_count != 0 && present[i])?;
        Some((graphics as u32, Some(present as u32)))
    } else {
        (0..families.len())
            .find(|&i| graphics(i))
            .map(|i| (i as u32, None))
    }
}

pub(crate) fn baseline_missing(
    version: u32,
    timeline: bool,
    synchronization2: bool,
    dynamic_rendering: bool,
    ext: &ExtensionSet,
) -> Vec<&'static str> {
    let mut missing = Vec::new();
    if version < vk::API_VERSION_1_2 {
        missing.push("Vulkan 1.2");
    }
    if !timeline {
        missing.push("timelineSemaphore feature");
    }
    if !synchronization2 {
        missing.push("synchronization2 feature");
    }
    if !dynamic_rendering {
        missing.push("dynamicRendering feature");
    }
    if version < vk::API_VERSION_1_3 {
        if !ext.contains(ash::khr::synchronization2::NAME) {
            missing.push("VK_KHR_synchronization2");
        }
        if !ext.contains(ash::khr::dynamic_rendering::NAME) {
            missing.push("VK_KHR_dynamic_rendering");
        }
    }
    missing
}

// Each optional structure is queried/enabled only with its extension/dependencies.
macro_rules! optional_features {
    ($($field:ident: $ty:ty => $name:expr),+ $(,)?) => {
        #[derive(Default)]
        struct FeatureChains { subgroup: bool, cache_control:bool, $($field: bool,)+ }
        #[derive(Default)]
        pub(crate) struct FeatureSet {
            chains: FeatureChains,
            pub core: vk::PhysicalDeviceFeatures,
            pub v12: vk::PhysicalDeviceVulkan12Features<'static>,
            pub sync: vk::PhysicalDeviceSynchronization2Features<'static>,
            pub rendering: vk::PhysicalDeviceDynamicRenderingFeatures<'static>,
            pub subgroup: vk::PhysicalDeviceSubgroupSizeControlFeatures<'static>,
            pub cache_control:vk::PhysicalDevicePipelineCreationCacheControlFeatures<'static>,
            $(pub $field: $ty,)+
            pub names: Vec<&'static CStr>,
        }
        impl FeatureSet {
            pub unsafe fn query(instance: &Instance, physical: vk::PhysicalDevice, ext: &ExtensionSet,
                version: u32, maintenance: SwapchainMaintenance) -> Self {
                let mut result = Self::default();
                let mut query = vk::PhysicalDeviceFeatures2::default().push_next(&mut result.v12);
                if version >= vk::API_VERSION_1_3 || ext.contains(ash::khr::synchronization2::NAME) { query = query.push_next(&mut result.sync); }
                if version >= vk::API_VERSION_1_3 || ext.contains(ash::khr::dynamic_rendering::NAME) { query = query.push_next(&mut result.rendering); }
                if version >= vk::API_VERSION_1_3 || ext.contains(ash::ext::subgroup_size_control::NAME) {
                    result.chains.subgroup = true;
                    query = query.push_next(&mut result.subgroup);
                    if version < vk::API_VERSION_1_3 { result.names.push(ash::ext::subgroup_size_control::NAME); }
                }
                $(if let Some(name) = ($name)(version, maintenance, ext) {
                    result.names.push(name);
                    result.chains.$field = true;
                    query = query.push_next(&mut result.$field);
                })+
                // The optional feature was promoted into Vulkan 1.3. A driver
                // need not advertise the old extension to expose/enable it.
                if version >= vk::API_VERSION_1_3 || ext.contains(ash::ext::pipeline_creation_cache_control::NAME) {
                    if version < vk::API_VERSION_1_3 {result.names.push(ash::ext::pipeline_creation_cache_control::NAME);}
                    result.chains.cache_control = true;
                    query = query.push_next(&mut result.cache_control);
                }
                // SAFETY: all structures live until this synchronous query returns;
                // each is defined by the supported API version or queried extension.
                unsafe { instance.get_physical_device_features2(physical, &mut query); }
                result.core = query.features;
                result.v12.p_next = std::ptr::null_mut();
                result.sync.p_next = std::ptr::null_mut();
                result.rendering.p_next = std::ptr::null_mut();
                result.subgroup.p_next = std::ptr::null_mut();
                result.cache_control.p_next=std::ptr::null_mut();
                $(result.$field.p_next = std::ptr::null_mut();)+
                result
            }
            pub fn enable_info(&mut self) -> vk::DeviceCreateInfo<'_> {
                let mut info = vk::DeviceCreateInfo::default().enabled_features(&self.core)
                    .push_next(&mut self.v12).push_next(&mut self.sync).push_next(&mut self.rendering);
                if self.chains.subgroup { info = info.push_next(&mut self.subgroup); }
                if self.chains.cache_control {info=info.push_next(&mut self.cache_control);}
                $(if self.chains.$field { info = info.push_next(&mut self.$field); })+
                info
            }
        }
    }
}

// Extension dependency selection is cold initialization work; enabled chains
// retain their selection explicitly, including KHR/EXT aliases.
fn supported(ext: &ExtensionSet, name: &'static CStr) -> Option<&'static CStr> {
    ext.contains(name).then_some(name)
}
macro_rules! fixed {
    ($name:expr) => {
        |_: u32, _: SwapchainMaintenance, e: &ExtensionSet| supported(e, $name)
    };
}
optional_features! {
    state1: vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT<'static> => fixed!(ash::ext::extended_dynamic_state::NAME),
    state2: vk::PhysicalDeviceExtendedDynamicState2FeaturesEXT<'static> => fixed!(ash::ext::extended_dynamic_state2::NAME),
    state3: vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT<'static> => fixed!(ash::ext::extended_dynamic_state3::NAME),
    vertex_input: vk::PhysicalDeviceVertexInputDynamicStateFeaturesEXT<'static> => fixed!(ash::ext::vertex_input_dynamic_state::NAME),
    gpl: vk::PhysicalDeviceGraphicsPipelineLibraryFeaturesEXT<'static> => fixed!(ash::ext::graphics_pipeline_library::NAME),
    descriptors: vk::PhysicalDeviceDescriptorBufferFeaturesEXT<'static> => fixed!(ash::ext::descriptor_buffer::NAME),
    objects: vk::PhysicalDeviceShaderObjectFeaturesEXT<'static> => fixed!(ash::ext::shader_object::NAME),
    feedback: vk::PhysicalDeviceTransformFeedbackFeaturesEXT<'static> => fixed!(ash::ext::transform_feedback::NAME),
    index8: vk::PhysicalDeviceIndexTypeUint8FeaturesKHR<'static> => |_, _, e: &ExtensionSet| supported(e, ash::khr::index_type_uint8::NAME).or_else(|| supported(e, ash::ext::index_type_uint8::NAME)),
    custom_border: vk::PhysicalDeviceCustomBorderColorFeaturesEXT<'static> => fixed!(ash::ext::custom_border_color::NAME),
    provoking: vk::PhysicalDeviceProvokingVertexFeaturesEXT<'static> => fixed!(ash::ext::provoking_vertex::NAME),
    portability: vk::PhysicalDevicePortabilitySubsetFeaturesKHR<'static> => fixed!(ash::khr::portability_subset::NAME),
    layouts: UnifiedLayouts<'static> => fixed!(extensions::UNIFIED_LAYOUTS),
    fma: ShaderFma<'static> => fixed!(extensions::SHADER_FMA),
    lines: vk::PhysicalDeviceLineRasterizationFeaturesKHR<'static> => |_, _, e: &ExtensionSet| supported(e, ash::khr::line_rasterization::NAME).or_else(|| supported(e, ash::ext::line_rasterization::NAME)),
    maintenance: vk::PhysicalDeviceSwapchainMaintenance1FeaturesEXT<'static> => |_, m, _: &ExtensionSet| match m { SwapchainMaintenance::Khr => Some(extensions::SWAPCHAIN_MAINTENANCE_KHR), SwapchainMaintenance::Ext => Some(ash::ext::swapchain_maintenance1::NAME), SwapchainMaintenance::Disabled => None },
}

impl FeatureSet {
    pub fn finish_enabling(
        &mut self,
        ext: &ExtensionSet,
        version: u32,
        maintenance: SwapchainMaintenance,
    ) {
        if version < vk::API_VERSION_1_3 {
            self.names.extend([
                ash::khr::synchronization2::NAME,
                ash::khr::dynamic_rendering::NAME,
            ]);
        }
        if self
            .names
            .contains(&ash::ext::graphics_pipeline_library::NAME)
        {
            if ext.contains(ash::khr::pipeline_library::NAME) {
                self.names.push(ash::khr::pipeline_library::NAME);
            } else {
                self.names
                    .retain(|n| *n != ash::ext::graphics_pipeline_library::NAME);
                self.gpl.graphics_pipeline_library = vk::FALSE;
                self.chains.gpl = false;
            }
        }
        if self.v12.buffer_device_address == vk::FALSE {
            self.names
                .retain(|n| *n != ash::ext::descriptor_buffer::NAME);
            self.descriptors = Default::default();
            self.chains.descriptors = false;
        }
        if ext.contains(ash::nv::fill_rectangle::NAME) {
            self.names.push(ash::nv::fill_rectangle::NAME);
        }
        // Enable only dynamic state with semantics supplied by the existing core
        // feature set. Other EDS3 fields need their own optional extensions.
        self.state3 = vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT::default()
            .extended_dynamic_state3_polygon_mode(
                self.state3.extended_dynamic_state3_polygon_mode != 0,
            )
            .extended_dynamic_state3_depth_clamp_enable(
                self.state3.extended_dynamic_state3_depth_clamp_enable != 0,
            )
            .extended_dynamic_state3_rasterization_samples(
                self.state3.extended_dynamic_state3_rasterization_samples != 0,
            )
            .extended_dynamic_state3_sample_mask(
                self.state3.extended_dynamic_state3_sample_mask != 0,
            )
            .extended_dynamic_state3_alpha_to_coverage_enable(
                self.state3.extended_dynamic_state3_alpha_to_coverage_enable != 0,
            )
            .extended_dynamic_state3_color_blend_enable(
                self.state3.extended_dynamic_state3_color_blend_enable != 0,
            )
            .extended_dynamic_state3_color_blend_equation(
                self.state3.extended_dynamic_state3_color_blend_equation != 0,
            )
            .extended_dynamic_state3_color_write_mask(
                self.state3.extended_dynamic_state3_color_write_mask != 0,
            );
        self.descriptors.descriptor_buffer_capture_replay = vk::FALSE;
        self.descriptors.descriptor_buffer_push_descriptors = vk::FALSE;
        self.core.robust_buffer_access = vk::FALSE;
        self.core.sparse_binding = vk::FALSE;
        self.core.sparse_residency_buffer = vk::FALSE;
        self.core.sparse_residency_image2_d = vk::FALSE;
        self.core.sparse_residency_image3_d = vk::FALSE;
        self.core.sparse_residency2_samples = vk::FALSE;
        self.core.sparse_residency4_samples = vk::FALSE;
        self.core.sparse_residency8_samples = vk::FALSE;
        self.core.sparse_residency16_samples = vk::FALSE;
        self.core.sparse_residency_aliased = vk::FALSE;
        self.v12.buffer_device_address_capture_replay = vk::FALSE;
        self.v12.buffer_device_address_multi_device = vk::FALSE;
        self.fma.float16 &= self.v12.shader_float16;
        self.fma.float64 &= self.core.shader_float64;
        self.layouts.video = vk::FALSE;
        if maintenance != SwapchainMaintenance::Disabled
            && self.maintenance.swapchain_maintenance1 == 0
        {
            self.names.retain(|n| {
                *n != extensions::SWAPCHAIN_MAINTENANCE_KHR
                    && *n != ash::ext::swapchain_maintenance1::NAME
            });
            self.chains.maintenance = false;
        }
        if version < vk::API_VERSION_1_3 && ext.contains(ash::ext::pipeline_creation_feedback::NAME)
        {
            self.names.push(ash::ext::pipeline_creation_feedback::NAME);
        }
        for name in [
            ash::khr::external_memory_fd::NAME,
            ash::khr::external_semaphore_fd::NAME,
        ] {
            if ext.contains(name) {
                self.names.push(name);
            }
        }
    }

    pub unsafe fn profile(
        &self,
        instance: &Instance,
        physical: vk::PhysicalDevice,
        version: u32,
        maintenance: SwapchainMaintenance,
    ) -> DeviceProfile {
        let mut floats = vk::PhysicalDeviceFloatControlsProperties::default();
        let mut id = vk::PhysicalDeviceIDProperties::default();
        let mut subgroup = vk::PhysicalDeviceSubgroupProperties::default();
        let mut subgroup_control = vk::PhysicalDeviceSubgroupSizeControlProperties::default();
        let mut gpl = vk::PhysicalDeviceGraphicsPipelineLibraryPropertiesEXT::default();
        let mut descriptors = vk::PhysicalDeviceDescriptorBufferPropertiesEXT::default();
        let mut props = vk::PhysicalDeviceProperties2::default()
            .push_next(&mut floats)
            .push_next(&mut id)
            .push_next(&mut subgroup);
        if self.chains.subgroup {
            props = props.push_next(&mut subgroup_control);
        }
        if self
            .names
            .contains(&ash::ext::graphics_pipeline_library::NAME)
        {
            props = props.push_next(&mut gpl);
        }
        if self.names.contains(&ash::ext::descriptor_buffer::NAME) {
            props = props.push_next(&mut descriptors);
        }
        // SAFETY: supported property structures are retained for the full query.
        unsafe {
            instance.get_physical_device_properties2(physical, &mut props);
        }
        let properties = props.properties;
        DeviceProfile {
            // SAFETY: physical device properties include a NUL-terminated name.
            name: unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }
                .to_string_lossy()
                .into_owned(),
            device_type: properties.device_type,
            api_version: version,
            api_path: if version >= vk::API_VERSION_1_3 {
                ApiPath::Core13
            } else {
                ApiPath::Khr12
            },
            driver_version: properties.driver_version,
            vendor_id: properties.vendor_id,
            device_id: properties.device_id,
            device_uuid: id.device_uuid,
            pipeline_cache_uuid: properties.pipeline_cache_uuid,
            pipeline_creation_feedback: version >= vk::API_VERSION_1_3
                || self
                    .names
                    .contains(&ash::ext::pipeline_creation_feedback::NAME),
            pipeline_creation_cache_control: self.cache_control.pipeline_creation_cache_control
                != 0,
            limits: properties.limits,
            core: self.core,
            float_controls: FloatControls {
                float32_denorm_preserve: floats.shader_denorm_preserve_float32 != 0,
                float32_denorm_flush_to_zero: floats.shader_denorm_flush_to_zero_float32 != 0,
                float32_signed_zero_inf_nan_preserve: floats
                    .shader_signed_zero_inf_nan_preserve_float32
                    != 0,
                float32_round_to_nearest_even: floats.shader_rounding_mode_rte_float32 != 0,
                float32_round_toward_zero: floats.shader_rounding_mode_rtz_float32 != 0,
                float64_denorm_preserve: floats.shader_denorm_preserve_float64 != 0,
                float64_signed_zero_inf_nan_preserve: floats
                    .shader_signed_zero_inf_nan_preserve_float64
                    != 0,
                float64_round_to_nearest_even: floats.shader_rounding_mode_rte_float64 != 0,
                fma_float16: self.fma.float16 != 0,
                fma_float32: self.fma.float32 != 0,
                fma_float64: self.fma.float64 != 0,
            },
            subgroup: SubgroupSupport {
                size: subgroup.subgroup_size,
                min_size: if self.subgroup.subgroup_size_control != 0 {
                    subgroup_control.min_subgroup_size
                } else {
                    subgroup.subgroup_size
                },
                max_size: if self.subgroup.subgroup_size_control != 0 {
                    subgroup_control.max_subgroup_size
                } else {
                    subgroup.subgroup_size
                },
                stages: subgroup.supported_stages,
                operations: subgroup.supported_operations,
                required_size_stages: subgroup_control.required_subgroup_size_stages,
                size_control: self.subgroup.subgroup_size_control != 0,
                compute_full_subgroups: self.subgroup.compute_full_subgroups != 0,
            },
            dynamic_state: DynamicStateSupport {
                base: version >= vk::API_VERSION_1_3 || self.state1.extended_dynamic_state != 0,
                base2: version >= vk::API_VERSION_1_3 || self.state2.extended_dynamic_state2 != 0,
                logic_op: self.state2.extended_dynamic_state2_logic_op != 0,
                patch_control_points: self.state2.extended_dynamic_state2_patch_control_points != 0,
                vertex_input: self.vertex_input.vertex_input_dynamic_state != 0,
                polygon_mode: self.state3.extended_dynamic_state3_polygon_mode != 0,
                depth_clamp_enable: self.state3.extended_dynamic_state3_depth_clamp_enable != 0,
                rasterization_samples: self.state3.extended_dynamic_state3_rasterization_samples
                    != 0,
                sample_mask: self.state3.extended_dynamic_state3_sample_mask != 0,
                alpha_to_coverage_enable: self
                    .state3
                    .extended_dynamic_state3_alpha_to_coverage_enable
                    != 0,
                color_blend_enable: self.state3.extended_dynamic_state3_color_blend_enable != 0,
                color_blend_equation: self.state3.extended_dynamic_state3_color_blend_equation != 0,
                color_write_mask: self.state3.extended_dynamic_state3_color_write_mask != 0,
            },
            graphics_pipeline_library: self.gpl.graphics_pipeline_library != 0,
            graphics_pipeline_library_fast_linking: gpl.graphics_pipeline_library_fast_linking != 0,
            graphics_pipeline_library_independent_interpolation: gpl
                .graphics_pipeline_library_independent_interpolation_decoration
                != 0,
            descriptor_indexing: self.v12.descriptor_indexing != 0,
            runtime_descriptor_array: self.v12.runtime_descriptor_array != 0,
            buffer_device_address: self.v12.buffer_device_address != 0,
            descriptor_buffer: self.descriptors.descriptor_buffer != 0,
            descriptor_buffer_image_layout_ignored: self
                .descriptors
                .descriptor_buffer_image_layout_ignored
                != 0,
            descriptor_buffer_offset_alignment: descriptors.descriptor_buffer_offset_alignment,
            storage_buffer_descriptor_size: descriptors.storage_buffer_descriptor_size,
            separate_depth_stencil_layouts: self.v12.separate_depth_stencil_layouts != 0,
            shader_object: self.objects.shader_object != 0,
            transform_feedback: self.feedback.transform_feedback != 0,
            provoking_vertex_last: self.provoking.provoking_vertex_last != 0,
            index_uint8: self.index8.index_type_uint8 != 0,
            fill_rectangle: self.names.contains(&ash::nv::fill_rectangle::NAME),
            custom_border_color: self.custom_border.custom_border_colors != 0,
            custom_border_color_without_format: self
                .custom_border
                .custom_border_color_without_format
                != 0,
            sampler_filter_minmax: self.v12.sampler_filter_minmax != 0,
            rectangular_lines: self.lines.rectangular_lines != 0,
            smooth_lines: self.lines.smooth_lines != 0,
            bresenham_lines: self.lines.bresenham_lines != 0,
            unified_image_layouts: self.layouts.layouts != 0,
            swapchain_mutable_format: self
                .names
                .contains(&ash::khr::swapchain_mutable_format::NAME),
            swapchain_maintenance: if self.maintenance.swapchain_maintenance1 != 0 {
                maintenance
            } else {
                SwapchainMaintenance::Disabled
            },
            portability: self
                .names
                .contains(&ash::khr::portability_subset::NAME)
                .then_some(PortabilitySupport {
                    triangle_fans: self.portability.triangle_fans != 0,
                    image_view_format_swizzle: self.portability.image_view_format_swizzle != 0,
                    mutable_comparison_samplers: self.portability.mutable_comparison_samplers != 0,
                    events: self.portability.events != 0,
                    tessellation_isolines: self.portability.tessellation_isolines != 0,
                    tessellation_point_mode: self.portability.tessellation_point_mode != 0,
                    separate_stencil_mask_ref: self.portability.separate_stencil_mask_ref != 0,
                }),
            external_memory_fd: self.names.contains(&ash::khr::external_memory_fd::NAME),
            external_semaphore_fd: self.names.contains(&ash::khr::external_semaphore_fd::NAME),
            enabled_extensions: self
                .names
                .iter()
                .map(|name| name.to_string_lossy().into_owned())
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn family(flags: vk::QueueFlags) -> vk::QueueFamilyProperties {
        vk::QueueFamilyProperties {
            queue_flags: flags,
            queue_count: 1,
            ..Default::default()
        }
    }
    #[test]
    fn queues_use_actual_present_support_and_prefer_a_unified_family() {
        let both = vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE;
        let families = [family(vk::QueueFlags::TRANSFER), family(both), family(both)];
        assert_eq!(
            choose_queues(&families, Some(&[false, false, true])),
            Some((2, Some(2)))
        );
        assert_eq!(
            choose_queues(&families, Some(&[true, false, false])),
            Some((1, Some(0)))
        );
        assert_eq!(choose_queues(&families, None), Some((1, None)));
        assert_eq!(choose_queues(&families, Some(&[false; 3])), None);
        let mut queues = [family(both), family(both)];
        queues[1].queue_count = 2;
        assert_eq!(
            choose_queues(&queues, Some(&[true, true])),
            Some((1, Some(1)))
        );
        assert_eq!(choose_queues(&queues, None), Some((0, None)));
    }
    #[test]
    fn version_never_substitutes_for_required_feature_bits() {
        let ext = ExtensionSet(HashSet::new());
        assert_eq!(
            baseline_missing(vk::API_VERSION_1_3, false, true, false, &ext),
            ["timelineSemaphore feature", "dynamicRendering feature"]
        );
        assert_eq!(
            baseline_missing(vk::API_VERSION_1_2, true, true, true, &ext),
            ["VK_KHR_synchronization2", "VK_KHR_dynamic_rendering"]
        );
    }
}
