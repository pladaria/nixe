//! Owned pipeline state; dynamic values are canonicalized out of cache keys.
use ash::vk;
use std::sync::Arc;
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct BlendState {
    pub enabled: bool,
    pub src_color: vk::BlendFactor,
    pub dst_color: vk::BlendFactor,
    pub color_op: vk::BlendOp,
    pub src_alpha: vk::BlendFactor,
    pub dst_alpha: vk::BlendFactor,
    pub alpha_op: vk::BlendOp,
    pub write_mask: vk::ColorComponentFlags,
}
impl BlendState {
    pub fn opaque() -> Self {
        Self {
            enabled: false,
            src_color: vk::BlendFactor::ONE,
            dst_color: vk::BlendFactor::ZERO,
            color_op: vk::BlendOp::ADD,
            src_alpha: vk::BlendFactor::ONE,
            dst_alpha: vk::BlendFactor::ZERO,
            alpha_op: vk::BlendOp::ADD,
            write_mask: vk::ColorComponentFlags::RGBA,
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct VertexBinding {
    pub binding: u32,
    pub stride: u32,
    pub rate: vk::VertexInputRate,
}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct VertexAttribute {
    pub location: u32,
    pub binding: u32,
    pub format: vk::Format,
    pub offset: u32,
}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct StencilState {
    pub fail: vk::StencilOp,
    pub pass: vk::StencilOp,
    pub depth_fail: vk::StencilOp,
    pub compare: vk::CompareOp,
}
impl Default for StencilState {
    fn default() -> Self {
        Self {
            fail: vk::StencilOp::KEEP,
            pass: vk::StencilOp::KEEP,
            depth_fail: vk::StencilOp::KEEP,
            compare: vk::CompareOp::ALWAYS,
        }
    }
}
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct GraphicsState {
    pub vertex_bindings: Arc<[VertexBinding]>,
    pub vertex_attributes: Arc<[VertexAttribute]>,
    /// Native (binding, instance divisor) entries; absent entries use one.
    pub vertex_divisors: Arc<[(u32, u32)]>,
    pub topology: vk::PrimitiveTopology,
    pub primitive_restart: bool,
    pub patch_points: u32,
    pub colors: Arc<[(vk::Format, BlendState)]>,
    pub depth_format: vk::Format,
    pub stencil_format: vk::Format,
    pub samples: vk::SampleCountFlags,
    pub sample_mask: u64,
    pub alpha_to_coverage: bool,
    pub polygon: vk::PolygonMode,
    pub cull: vk::CullModeFlags,
    pub front_face: vk::FrontFace,
    pub depth_clamp: bool,
    pub raster_discard: bool,
    pub depth_bias: bool,
    pub depth_test: bool,
    pub depth_write: bool,
    pub depth_compare: vk::CompareOp,
    pub depth_bounds: bool,
    pub stencil_test: bool,
    pub front: StencilState,
    pub back: StencilState,
    pub logic_enabled: bool,
    pub logic_op: vk::LogicOp,
    pub viewport_count: u32,
    pub provoking: vk::ProvokingVertexModeEXT,
    pub line_mode: vk::LineRasterizationModeEXT,
}
impl GraphicsState {
    pub fn color(format: vk::Format) -> Self {
        Self {
            vertex_bindings: Arc::from([]),
            vertex_attributes: Arc::from([]),
            vertex_divisors: Arc::from([]),
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            primitive_restart: false,
            patch_points: 0,
            colors: Arc::from([(format, BlendState::opaque())]),
            depth_format: vk::Format::UNDEFINED,
            stencil_format: vk::Format::UNDEFINED,
            samples: vk::SampleCountFlags::TYPE_1,
            sample_mask: u64::MAX,
            alpha_to_coverage: false,
            polygon: vk::PolygonMode::FILL,
            cull: vk::CullModeFlags::NONE,
            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
            depth_clamp: false,
            raster_discard: false,
            depth_bias: false,
            depth_test: false,
            depth_write: false,
            depth_compare: vk::CompareOp::ALWAYS,
            depth_bounds: false,
            stencil_test: false,
            front: Default::default(),
            back: Default::default(),
            logic_enabled: false,
            logic_op: vk::LogicOp::COPY,
            viewport_count: 1,
            provoking: vk::ProvokingVertexModeEXT::FIRST_VERTEX,
            line_mode: vk::LineRasterizationModeEXT::DEFAULT,
        }
    }
}

impl GraphicsState {
    pub(crate) fn normalized(mut self, dynamic: &[vk::DynamicState]) -> Self {
        if dynamic.contains(&vk::DynamicState::CULL_MODE) {
            self.cull = vk::CullModeFlags::NONE;
            self.front_face = vk::FrontFace::COUNTER_CLOCKWISE;
            self.depth_test = false;
            self.depth_write = false;
            self.depth_compare = vk::CompareOp::ALWAYS;
            self.depth_bounds = false;
            self.stencil_test = false;
            self.front = StencilState::default();
            self.back = StencilState::default();
        }
        if dynamic.contains(&vk::DynamicState::PRIMITIVE_TOPOLOGY)
            && dynamic.contains(&vk::DynamicState::PRIMITIVE_RESTART_ENABLE)
        {
            self.topology = topology_class(self.topology);
        }
        if dynamic.contains(&vk::DynamicState::PRIMITIVE_RESTART_ENABLE) {
            self.primitive_restart = false;
            self.raster_discard = false;
            self.depth_bias = false;
        }
        if dynamic.contains(&vk::DynamicState::VERTEX_INPUT_EXT) {
            self.vertex_bindings = Arc::from([]);
            self.vertex_attributes = Arc::from([]);
            self.vertex_divisors = Arc::from([]);
        } else if dynamic.contains(&vk::DynamicState::VERTEX_INPUT_BINDING_STRIDE) {
            for binding in Arc::make_mut(&mut self.vertex_bindings) {
                binding.stride = 0;
            }
        }
        if dynamic.contains(&vk::DynamicState::COLOR_BLEND_ENABLE_EXT) {
            self.colors = self
                .colors
                .iter()
                .map(|&(format, _)| (format, BlendState::opaque()))
                .collect();
        }
        self
    }
}

/// Eden collapses triangle assembly variants only when both EDS1 and EDS2
/// are present. Line strips retain their own representative.
/// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/fixed_pipeline_state.cpp
pub(crate) fn topology_class(topology: vk::PrimitiveTopology) -> vk::PrimitiveTopology {
    match topology {
        vk::PrimitiveTopology::TRIANGLE_STRIP | vk::PrimitiveTopology::TRIANGLE_FAN => {
            vk::PrimitiveTopology::TRIANGLE_LIST
        }
        vk::PrimitiveTopology::LINE_STRIP_WITH_ADJACENCY => {
            vk::PrimitiveTopology::LINE_LIST_WITH_ADJACENCY
        }
        vk::PrimitiveTopology::TRIANGLE_STRIP_WITH_ADJACENCY => {
            vk::PrimitiveTopology::TRIANGLE_LIST_WITH_ADJACENCY
        }
        _ => topology,
    }
}
/// Matches Eden IsPrimitiveRestartSupported for the actual native topology.
/// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/vk_rasterizer.cpp
pub(crate) fn restart_supported(topology: vk::PrimitiveTopology, support: [bool; 2]) -> bool {
    match topology {
        vk::PrimitiveTopology::PATCH_LIST => support[1],
        vk::PrimitiveTopology::POINT_LIST
        | vk::PrimitiveTopology::LINE_LIST
        | vk::PrimitiveTopology::TRIANGLE_LIST
        | vk::PrimitiveTopology::LINE_LIST_WITH_ADJACENCY
        | vk::PrimitiveTopology::TRIANGLE_LIST_WITH_ADJACENCY => support[0],
        _ => true,
    }
}
