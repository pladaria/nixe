//! Optional native execution within the primary wgpu backend. No separate queue.

#[cfg(not(target_os = "macos"))]
pub(crate) mod vulkan;

/// Enabled Vulkan device capabilities, not a claim that all corresponding guest
/// semantics have an executable backend path. Missing features remain optional
/// for ordinary wgpu rendering and are checked when native shaders consume them.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VulkanNativeCapabilities {
    pub tessellation_shader: bool,
    pub raster: VulkanRasterCapabilities,
    pub float32: nixe_gpu::SpirvFloat32Capabilities,
    pub float64: nixe_gpu::SpirvFloat64Capabilities,
    pub tessellation_limits: VulkanTessellationLimits,
    pub graphics_limits: VulkanGraphicsLimits,
    /// HAL's enabled core robustness, required by native dynamic buffer reads.
    pub robust_buffer_access: bool,
    /// Full uint32 index values without CPU inspection of the index stream.
    pub full_draw_index_uint32: bool,
    /// Exact wgpu-hal 30 Depth24PlusStencil8 physical format selection.
    pub depth24_stencil8_uses_float32: bool,
}

/// Enabled line features and device limits, queried once at device creation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VulkanRasterCapabilities {
    pub wireframe: bool,
    pub wide_lines: bool,
    pub rectangular_lines: bool,
    pub smooth_lines: bool,
    pub line_width_range_bits: [u32; 2],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VulkanGraphicsLimits {
    pub storage_buffers_per_stage: u32,
    pub resources_per_stage: u32,
    pub storage_buffers_per_set: u32,
    pub storage_buffer_range: u32,
    pub vertex_input_attributes: u32,
    pub vertex_input_bindings: u32,
    pub vertex_input_attribute_offset: u32,
    pub vertex_input_binding_stride: u32,
    pub vertex_output_components: u32,
    pub fragment_input_components: u32,
}

/// Physical Vulkan limits, retained once per device rather than queried per draw.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VulkanTessellationLimits {
    pub generation_level: u32,
    pub patch_size: u32,
    pub control_per_vertex_input_components: u32,
    pub control_per_vertex_output_components: u32,
    pub control_per_patch_output_components: u32,
    pub control_total_output_components: u32,
    pub evaluation_input_components: u32,
    pub evaluation_output_components: u32,
}
