//! Owned native parameters passed to the Vulkan recording worker.
use super::*;
#[derive(Clone, Copy)]
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
        }
    }
}
