//! Fixed-point NV12 is a host utility, compiled through the ordinary pipeline
//! cache and recorded as an ordinary raster workload. External import is separate.
use super::*;
use crate::{
    Binding, Draw, GraphicsParameters, PreparedDraw, RenderAttachment, RenderingTargets, cache::*,
};
use nixe_gpu::{
    Nv12ColorTransform, ShaderResourceKind, ShaderStage,
    ssa::{ResourceId, spirv},
};
use std::sync::OnceLock;
pub(crate) fn stage(
    stage: ShaderStage,
    bytes: &[u8],
    bindings: Box<[spirv::Binding]>,
    push: u32,
) -> CompiledStage {
    CompiledStage::new(
        stage,
        spirv::Module {
            words: bytes
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                .collect(),
            bindings,
            push_constant_bytes: push,
        },
        None,
        0,
        None,
    )
}
pub struct PreparedNv12 {
    draw: PreparedDraw,
    parameters: [u8; 80],
}
/// Request ahead of consumption, exactly like a guest graphics pipeline. The
/// owner polls the returned publication; no synchronous compilation in recording.
pub fn nv12_pipeline(
    compiler: &PipelineCompiler,
    format: vk::Format,
) -> Result<Pending<NativePipeline>, CompileError> {
    if !matches!(
        format,
        vk::Format::R8G8B8A8_UNORM | vk::Format::B8G8R8A8_UNORM
    ) {
        return Err(CompileError::Failed(
            "NV12 output requires RGBA8/BGRA8 UNORM".into(),
        ));
    }
    static SHADER: OnceLock<Arc<ShaderArtifact>> = OnceLock::new();
    let shader = SHADER.get_or_init(|| {
        Arc::new(ShaderArtifact::new(Box::new([
            stage(
                ShaderStage::Vertex,
                include_bytes!("../../shaders/spv/solid.vert.spv"),
                Box::new([]),
                0,
            ),
            stage(
                ShaderStage::Fragment,
                include_bytes!("../../shaders/spv/nv12.frag.spv"),
                (0..2)
                    .map(|binding| spirv::Binding {
                        resource: ResourceId::new(binding),
                        binding,
                        kind: ShaderResourceKind::SampledImage,
                        readable: true,
                        writable: false,
                    })
                    .collect(),
                80,
            ),
        ])))
    });
    compiler.request(PipelineRecipe::Graphics {
        shader: Arc::clone(shader),
        state: GraphicsState::color(format),
    })
}
impl VulkanResidency {
    pub fn prepare_nv12(
        &mut self,
        pipeline: Arc<NativePipeline>,
        sources: [ImageViewId; 2],
        destination: ImageId,
        plane: usize,
        transform: Nv12ColorTransform,
    ) -> Result<PreparedNv12, VulkanError> {
        let t = transform;
        if t.shift > 15
            || t.clamp[0] < 0
            || t.clamp[1] > 1023
            || t.clamp[0] > t.clamp[1]
            || t.coefficients
                .iter()
                .flatten()
                .any(|c| !(-524288..=524287).contains(c))
        {
            return Err(VulkanError::InvalidResource(
                "NV12 fixed-point coefficients, shift or clamp exceed the supported integer contract",
            ));
        }
        let extent = self
            .image(destination)?
            .planes
            .get(plane)
            .ok_or(VulkanError::InvalidResource(
                "NV12 destination plane is absent",
            ))?
            .extent;
        for (index, view) in sources.iter().enumerate() {
            let image = self.image(view.image)?;
            let d = image
                .views
                .get(view.view)
                .ok_or(VulkanError::InvalidResource("NV12 source view is absent"))?;
            let format = if index == 0 {
                vk::Format::R8_UNORM
            } else {
                vk::Format::R8G8_UNORM
            };
            let end = [
                t.source_origin[0].checked_add(extent.width),
                t.source_origin[1].checked_add(extent.height),
            ];
            if image.description.format!=format || d.format!=format || d.view_type!=vk::ImageViewType::TYPE_2D || d.base_mip!=0 || d.mip_count!=1 || d.base_layer!=0 || d.layer_count!=1 || d.swizzle!=[vk::ComponentSwizzle::IDENTITY;4]
                || end[0].is_none_or(|v|if index==0 {v} else {v.div_ceil(2)}>image.description.extent.width) || end[1].is_none_or(|v|if index==0 {v} else {v.div_ceil(2)}>image.description.extent.height) {
                return Err(VulkanError::InvalidResource("NV12 needs complete R8/RG8 source views and a bounded source crop"));
            }
        }
        let format = self.image(destination)?.description.format;
        if !matches!(
            format,
            vk::Format::R8G8B8A8_UNORM | vk::Format::B8G8R8A8_UNORM
        ) {
            return Err(VulkanError::InvalidResource(
                "NV12 destination requires RGBA8/BGRA8 UNORM",
            ));
        }
        let program = self.prepare_program(pipeline, &sources.map(Binding::Image))?;
        let graphics = self.prepare_graphics(
            program,
            GraphicsParameters::new(
                GraphicsState::color(format),
                vk::Extent2D {
                    width: extent.width,
                    height: extent.height,
                },
            ),
        )?;
        let targets = self.prepare_targets(RenderingTargets {
            area: transfers::full_rect(self.image(destination)?.planes[plane]),
            colors: Box::new([Some(RenderAttachment::color(
                destination,
                plane,
                AttachmentLoad::Discard,
                AttachmentStore::Store,
            ))]),
            depth: None,
            stencil: None,
        })?;
        let draw = self.prepare_draw(graphics, targets, Box::new([]), None)?;
        let words: Vec<_> = t
            .coefficients
            .into_iter()
            .flatten()
            .map(|v| v as u32)
            .chain([
                t.source_origin[0],
                t.source_origin[1],
                t.shift,
                u32::from(t.enabled),
            ])
            .chain([t.clamp[0] as u32, t.clamp[1] as u32, u32::from(t.alpha), 0])
            .collect();
        let parameters = std::array::from_fn(|i| words[i / 4].to_le_bytes()[i % 4]);
        Ok(PreparedNv12 { draw, parameters })
    }
    pub fn convert_nv12(&mut self, prepared: &PreparedNv12) -> Result<WorkTicket, VulkanError> {
        self.draw(
            &prepared.draw,
            Draw::Direct {
                vertices: 3,
                instances: 1,
                first_vertex: 0,
                first_instance: 0,
            },
            &prepared.parameters,
        )
    }
}
