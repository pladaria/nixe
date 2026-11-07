//! Owned native parameters passed to the Vulkan recording worker.
use super::*;
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct DynamicState {
    pub cull: vk::CullModeFlags,
    pub front_face: vk::FrontFace,
    pub depth_test: bool,
    pub depth_write: bool,
    pub depth_compare: vk::CompareOp,
    pub depth_bounds: bool,
    pub stencil_test: bool,
    pub front: crate::cache::StencilState,
    pub back: crate::cache::StencilState,
    pub topology: vk::PrimitiveTopology,
    pub primitive_restart: bool,
    pub raster_discard: bool,
    pub depth_bias: bool,
}
impl From<&crate::cache::GraphicsState> for DynamicState {
    fn from(state: &crate::cache::GraphicsState) -> Self {
        Self {
            cull: state.cull,
            front_face: state.front_face,
            depth_test: state.depth_test,
            depth_write: state.depth_write,
            depth_compare: state.depth_compare,
            depth_bounds: state.depth_bounds,
            stencil_test: state.stencil_test,
            front: state.front,
            back: state.back,
            topology: state.topology,
            primitive_restart: state.primitive_restart,
            raster_discard: state.raster_discard,
            depth_bias: state.depth_bias,
        }
    }
}
