//! CPU visibility reads individual samples, never an averaged color resolve.
//! Host utilities use the renderer's shared pipeline compiler and normal recorder.
use super::*;
use crate::{Draw, GraphicsParameters, RenderAttachment, RenderingTargets, cache::*};
use nixe_gpu::{
    ShaderStage,
    shader_module::{self as spirv, ResourceId},
};
use std::sync::OnceLock;

pub(super) struct SampleConversion {
    compiler: Arc<PipelineCompiler>,
    descriptors: DescriptorCache,
    pipelines: [Option<Arc<NativePipeline>>; 2],
}
fn artifact(stencil: bool) -> Arc<ShaderArtifact> {
    static FLOAT: OnceLock<Arc<ShaderArtifact>> = OnceLock::new();
    static STENCIL: OnceLock<Arc<ShaderArtifact>> = OnceLock::new();
    Arc::clone((if stencil { &STENCIL } else { &FLOAT }).get_or_init(|| {
        Arc::new(ShaderArtifact::new(Box::new([
            conversion::stage(
                ShaderStage::Vertex,
                include_bytes!("../../shaders/spv/solid.vert.spv"),
                Box::new([]),
                0,
            ),
            conversion::stage(
                ShaderStage::Fragment,
                if stencil {
                    include_bytes!("../../shaders/spv/sample-stencil-readback.frag.spv")
                } else {
                    include_bytes!("../../shaders/spv/sample-readback.frag.spv")
                },
                Box::new([spirv::Binding {
                    resource: ResourceId::new(0),
                    binding: 0,
                    kind: spirv::DescriptorKind::SampledImage,
                    count: 1,
                    sampler: None,
                    readable: true,
                    writable: false,
                }]),
                16,
            ),
        ])))
    }))
}
impl Residency<'_> {
    pub fn enable_sample_conversion(&mut self, compiler: Arc<PipelineCompiler>) {
        self.texture_cache.sample_conversion = Some(SampleConversion {
            compiler,
            descriptors: DescriptorCache::new(&self.runtime.device),
            pipelines: [None, None],
        });
    }
    pub(super) fn sample_readback_plane(
        &mut self,
        id: ImageId,
        plane: ImagePlane,
    ) -> Result<AttachmentId, VulkanError> {
        let mut conversion =
            self.texture_cache
                .sample_conversion
                .take()
                .ok_or(VulkanError::InvalidResource(
                    "multisample CPU visibility needs the shared pipeline compiler",
                ))?;
        self.runtime.constructing.push(Resource::Image(id));
        let result = self.record_sample_plane(id, plane, &mut conversion);
        self.unpin(Resource::Image(id));
        self.texture_cache.sample_conversion = Some(conversion);
        result
    }
    fn record_sample_plane(
        &mut self,
        id: ImageId,
        plane: ImagePlane,
        conversion: &mut SampleConversion,
    ) -> Result<AttachmentId, VulkanError> {
        let entry = self.texture_cache.get(id);
        let p = plane;
        let grid = entry.sample_grid;
        let format = match entry.description.format {
            vk::Format::R8G8B8A8_SRGB => vk::Format::R8G8B8A8_UNORM,
            vk::Format::B8G8R8A8_SRGB => vk::Format::B8G8R8A8_UNORM,
            format => format,
        };
        let stencil = p.aspect == vk::ImageAspectFlags::STENCIL;
        let kind = match (p.aspect, format) {
            (vk::ImageAspectFlags::STENCIL, _) => 0u32,
            (
                vk::ImageAspectFlags::DEPTH,
                vk::Format::D24_UNORM_S8_UINT | vk::Format::X8_D24_UNORM_PACK32,
            ) => 1,
            (
                vk::ImageAspectFlags::DEPTH,
                vk::Format::D32_SFLOAT | vk::Format::D32_SFLOAT_S8_UINT,
            ) => 2,
            (vk::ImageAspectFlags::COLOR, vk::Format::R8G8B8A8_UNORM) => 0,
            (vk::ImageAspectFlags::COLOR, vk::Format::B8G8R8A8_UNORM) => 3,
            _ => {
                return Err(VulkanError::InvalidResource(
                    "multisample readback has no exact encoder for this format/aspect",
                ));
            }
        };
        let target_format = if stencil {
            vk::Format::R8_UINT
        } else {
            vk::Format::R32_UINT
        };
        let pipeline = if let Some(pipeline) = &conversion.pipelines[stencil as usize] {
            Arc::clone(pipeline)
        } else {
            let pending = conversion
                .compiler
                .request(PipelineRecipe::Graphics {
                    shader: artifact(stencil),
                    state: GraphicsState::color(target_format),
                })
                .map_err(|e| VulkanError::Compilation(e.to_string().into()))?;
            let pipeline = pending;
            conversion.pipelines[stencil as usize] = Some(Arc::clone(&pipeline));
            pipeline
        };
        let source = self.image_view(
            id,
            ImageViewDescription {
                size: p.extent,
                format,
                view_type: vk::ImageViewType::TYPE_2D,
                aspects: p.aspect,
                base_mip: p.mip_level,
                mip_count: 1,
                base_layer: p.array_layer,
                layer_count: 1,
                swizzle: [vk::ComponentSwizzle::IDENTITY; 4],
            },
        )?;
        // Download the native samples before refreshing CPU-modified contents,
        // as TextureCache::DownloadMemory does in the reference implementation.
        self.shader_image(id)?;
        let layout = vk::ImageLayout::GENERAL;
        let record = conversion.descriptors.get(
            &pipeline.layout,
            &[DescriptorResource::Image {
                view: self.image_view_lease(source)?,
                layout,
            }],
        )?;
        let extent = vk::Extent2D {
            width: p.extent.width.div_ceil(p.block_extent.width),
            height: p.rows(),
        };
        let target = self.attachment_image(NativeImageDescription {
            image_type: vk::ImageType::TYPE_2D,
            format: target_format,
            extent: vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            flags: vk::ImageCreateFlags::empty(),
        })?;
        let graphics = self.prepare_host_graphics(
            pipeline,
            record,
            GraphicsParameters::new(GraphicsState::color(target_format), extent),
        )?;
        let targets = self.prepare_targets(RenderingTargets {
            area: vk::Rect2D {
                offset: Default::default(),
                extent,
            },
            colors: Box::new([Some(RenderAttachment {
                source: crate::AttachmentSource::ResidentOnly(target),
                load: vk::AttachmentLoadOp::DONT_CARE,
                store: vk::AttachmentStoreOp::STORE,
                clear: Default::default(),
            })]),
            depth: None,
            stencil: None,
        })?;
        let push: Vec<_> = [grid[0], grid[1], 0, kind]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect();
        self.draw(
            &graphics,
            &targets,
            Draw::Direct {
                vertices: 3,
                instances: 1,
                first_vertex: 0,
                first_instance: 0,
            },
            &push,
        )?;
        Ok(target)
    }
}
