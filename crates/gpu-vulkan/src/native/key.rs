//! Static pipeline identity has fixed slots, independent of resource bindings.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/vk_pipeline_cache.cpp
use super::*;
use nixe_gpu::{
    AlphaTest, ColorOutputState, CullMode, DepthState, FragmentDepthTransform, FrontFace,
    ImageFormat, PrimitiveTopology, SampleCount, StencilTestState, TessellationState,
    TriangleRasterization, VertexStepMode,
};

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(super) struct GraphicsKey {
    shader: ContentId,
    vertices: [u64; 32],
    attributes: [u64; 32],
    colors: [Option<(ImageFormat, ColorOutputState)>; 8],
    depth_stencil_format: Option<ImageFormat>,
    stencil: bool,
    samples: SampleCount,
    topology: PrimitiveTopology,
    restart: bool,
    provoking_first: bool,
    raster: TriangleRasterization,
    smooth_line: Option<bool>,
    sample_mask: u64,
    alpha_to_coverage: bool,
    depth_bias: bool,
    fixed_state: Option<(CullMode, FrontFace, DepthState, Option<StencilTestState>)>,
    shader_state: GuestShaderState,
    alpha_test: Option<AlphaTest>,
    tessellation: Option<TessellationState>,
    negative_depth: bool,
    fragment_depth_transform: Option<FragmentDepthTransform>,
}
impl GraphicsKey {
    pub fn new(draw: &GuestDrawState, dynamic: bool) -> Result<Self, VulkanError> {
        let p = &draw.parameters;
        let tessellation = p.tessellation.map(|mut t| {
            if matches!(
                t.control,
                nixe_gpu::TessellationControl::DefaultLevels { .. }
            ) {
                t.control = nixe_gpu::TessellationControl::DefaultLevels {
                    outer: [0; 4],
                    inner: [0; 2],
                };
            }
            t
        });
        let raster = match p.raster {
            TriangleRasterization::Wireframe { smooth, .. } => TriangleRasterization::Wireframe {
                width_bits: 0,
                smooth,
            },
            other => other,
        };
        let stencil = p.stencil.map(|mut s| {
            for face in [&mut s.front, &mut s.back] {
                face.read_mask = 0;
                face.write_mask = 0;
                face.reference = 0;
            }
            s
        });
        let mut key = Self {
            shader: draw.program.shader.identity,
            vertices: [0; 32],
            attributes: [0; 32],
            colors: [None; 8],
            depth_stencil_format: p.depth_stencil_format,
            stencil: p.stencil.is_some(),
            samples: p.samples,
            topology: p.topology,
            restart: p.restart,
            provoking_first: p.provoking_first,
            raster,
            smooth_line: p.line.map(|l| l.smooth),
            sample_mask: p.sample_mask,
            alpha_to_coverage: p.alpha_to_coverage,
            depth_bias: p.depth_bias.is_some(),
            fixed_state: (!dynamic).then_some((p.cull, p.front_face, p.depth, stencil)),
            shader_state: p.shader_state,
            alpha_test: p.alpha_test,
            tessellation,
            negative_depth: p
                .viewport
                .is_some_and(|v| v.depth_clip_negative_one_to_one()),
            fragment_depth_transform: p.viewport.and_then(|v| v.fragment_depth_transform()),
        };
        for vertex in &p.vertices {
            let slot = key.vertices.get_mut(vertex.binding as usize).ok_or(
                VulkanError::InvalidResource("vertex binding exceeds Maxwell's 32 slots"),
            )?;
            if *slot != 0 {
                return Err(VulkanError::InvalidResource("duplicate vertex binding"));
            }
            *slot = 1 << 63
                | u64::from(vertex.stride)
                | (u64::from(vertex.step == VertexStepMode::Instance) << 32);
            for attribute in &vertex.attributes {
                let slot = key
                    .attributes
                    .get_mut(attribute.shader_location as usize)
                    .ok_or(VulkanError::InvalidResource(
                        "vertex attribute exceeds Maxwell's 32 slots",
                    ))?;
                if *slot != 0 {
                    return Err(VulkanError::InvalidResource("duplicate vertex attribute"));
                }
                // Dense scalar slots avoid pointer trees and padding-dependent
                // identities. All currently supported native vertex formats fit u8.
                let fetch = vertex::VertexFetchFormat::new(attribute.format);
                let format = u8::try_from(fetch.format.as_raw()).map_err(|_| {
                    VulkanError::InvalidResource("vertex format exceeds pipeline key encoding")
                })?;
                let offset = u32::try_from(attribute.offset).map_err(|_| {
                    VulkanError::InvalidResource("vertex offset exceeds Vulkan limits")
                })?;
                let scaled = fetch
                    .scaled32
                    .map_or(0, |signed| if signed { 2u64 } else { 1 });
                *slot = 1 << 63
                    | u64::from(offset)
                    | (u64::from(format) << 32)
                    | (u64::from(vertex.binding) << 40)
                    | (scaled << 45);
            }
        }
        if p.color_formats.len() > key.colors.len()
            || p.color_formats.len() != p.color_outputs.len()
        {
            return Err(VulkanError::InvalidResource("invalid color output count"));
        }
        for (slot, (&format, &output)) in key
            .colors
            .iter_mut()
            .zip(p.color_formats.iter().zip(&p.color_outputs))
        {
            *slot = Some((format, output));
        }
        Ok(key)
    }
    pub fn matches_source(&self, source: &GuestDrawState, parameters: PreparedStateId) -> bool {
        parameters == source.parameters.identity && self.shader == source.program.shader.identity
    }
}
