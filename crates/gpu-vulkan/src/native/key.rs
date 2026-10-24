//! Pipeline selection uses only shader specialization and immutable native state.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/vk_pipeline_cache.cpp
use super::*;
use std::hash::{Hash, Hasher};
#[derive(Clone)]
pub(super) struct GraphicsKey {
    pub parameters: Arc<GuestDrawParameters>,
    shader: ContentId,
    dynamic: bool,
}
impl GraphicsKey {
    pub fn new(draw: &GuestDrawState, dynamic: bool) -> Self {
        Self {
            parameters: Arc::clone(&draw.parameters),
            shader: draw.program.shader.identity,
            dynamic,
        }
    }
}
fn identity(p: &GuestDrawParameters, dynamic: bool) -> impl Eq + Hash + '_ {
    let tessellation = p.tessellation.map(|mut t| {
        if matches!(
            t.control,
            nixe_gpu::TessellationControl::DefaultLevels { .. }
        ) {
            t.control = nixe_gpu::TessellationControl::DefaultLevels {
                outer: [0; 4],
                inner: [0; 2],
                defined: 0,
            };
        }
        t
    });
    let raster = match p.raster {
        nixe_gpu::TriangleRasterization::Wireframe { smooth, .. } => {
            nixe_gpu::TriangleRasterization::Wireframe {
                width_bits: 0,
                smooth,
            }
        }
        other => other,
    };
    let stencil = p.stencil.map(|s| {
        (
            s.front.fail,
            s.front.pass,
            s.front.depth_fail,
            s.front.compare,
            s.back.fail,
            s.back.pass,
            s.back.depth_fail,
            s.back.compare,
        )
    });
    (
        (
            &p.vertices,
            &p.color_formats,
            p.depth_stencil_format,
            p.stencil.is_some(),
            p.samples,
            p.topology,
            p.restart,
            p.provoking_first,
            raster,
            p.line.map(|l| l.smooth),
        ),
        (
            &p.color_outputs,
            p.sample_mask,
            p.alpha_to_coverage,
            p.depth_bias.is_some(),
            (!dynamic).then_some((p.cull, p.front_face, p.depth, stencil)),
        ),
        (
            p.shader_state,
            p.alpha_test,
            tessellation,
            p.viewport
                .is_some_and(|v| v.depth_clip_negative_one_to_one()),
            p.viewport.and_then(|v| v.fragment_depth_transform()),
        ),
    )
}
impl PartialEq for GraphicsKey {
    fn eq(&self, other: &Self) -> bool {
        self.shader == other.shader
            && self.dynamic == other.dynamic
            && identity(&self.parameters, self.dynamic)
                == identity(&other.parameters, other.dynamic)
    }
}
impl Eq for GraphicsKey {}
impl Hash for GraphicsKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.shader.hash(state);
        self.dynamic.hash(state);
        identity(&self.parameters, self.dynamic).hash(state);
    }
}
