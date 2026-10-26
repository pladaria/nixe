//! Presentation and the bounded egui mesh contract use the ordinary native
//! pipeline/descriptor caches, command slots, allocator and timeline.
use super::*;
use crate::cache::*;
use ash::vk;
use nixe_gpu::{
    ShaderStage,
    shader_module::{self as spirv, ResourceId},
};
use rustc_hash::FxHashMap as HashMap;
pub(crate) fn color_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .level_count(1)
        .layer_count(1)
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn transition(
    device: &mut VulkanDevice,
    image: vk::Image,
    old: vk::ImageLayout,
    new: vk::ImageLayout,
    source: vk::PipelineStageFlags2,
    reads: vk::AccessFlags2,
    destination: vk::PipelineStageFlags2,
    writes: vk::AccessFlags2,
) {
    let barriers = [vk::ImageMemoryBarrier2::default()
        .image(image)
        .subresource_range(color_range())
        .old_layout(old)
        .new_layout(new)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .src_stage_mask(source)
        .src_access_mask(reads)
        .dst_stage_mask(destination)
        .dst_access_mask(writes)];
    unsafe {
        device
            .dispatch
            .barrier(&vk::DependencyInfo::default().image_memory_barriers(&barriers));
    }
}
fn shader(kind: u8) -> Arc<ShaderArtifact> {
    use std::sync::OnceLock;
    static SCREEN: OnceLock<Arc<ShaderArtifact>> = OnceLock::new();
    static PACKED: OnceLock<Arc<ShaderArtifact>> = OnceLock::new();
    static UI: OnceLock<Arc<ShaderArtifact>> = OnceLock::new();
    let cell = match kind {
        0 => &SCREEN,
        1 => &PACKED,
        _ => &UI,
    };
    Arc::clone(cell.get_or_init(|| {
        let stage = crate::residency::helper_stage;
        let vertex: &[u8] = if kind == 2 {
            include_bytes!("../../shaders/spv/ui.vert.spv")
        } else {
            include_bytes!("../../shaders/spv/screen.vert.spv")
        };
        let fragment: &[u8] = match kind {
            0 => include_bytes!("../../shaders/spv/screen.frag.spv"),
            1 => include_bytes!("../../shaders/spv/screen-packed.frag.spv"),
            _ => include_bytes!("../../shaders/spv/ui.frag.spv"),
        };
        let bindings = [
            spirv::DescriptorKind::SampledImage,
            spirv::DescriptorKind::Sampler,
        ]
        .into_iter()
        .enumerate()
        .map(|(i, kind)| spirv::Binding {
            resource: ResourceId::new(i as u32),
            binding: i as u32,
            kind,
            count: 1,
            sampler: None,
            readable: true,
            writable: false,
        })
        .collect();
        Arc::new(ShaderArtifact::new(Box::new([
            stage(
                ShaderStage::Vertex,
                vertex,
                Box::new([]),
                if kind == 2 { 8 } else { 0 },
            ),
            stage(
                ShaderStage::Fragment,
                fragment,
                bindings,
                if kind == 2 { 0 } else { 32 },
            ),
        ])))
    }))
}
fn state(format: vk::Format, kind: u8) -> GraphicsState {
    let mut state = GraphicsState::color(format);
    if kind == 2 {
        state.vertex_bindings = std::sync::Arc::from([VertexBinding {
            binding: 0,
            stride: 20,
            rate: vk::VertexInputRate::VERTEX,
        }]);
        state.vertex_attributes = std::sync::Arc::from([
            VertexAttribute {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: 0,
            },
            VertexAttribute {
                location: 1,
                binding: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: 8,
            },
            VertexAttribute {
                location: 2,
                binding: 0,
                format: vk::Format::R8G8B8A8_UNORM,
                offset: 16,
            },
        ]);
        std::sync::Arc::make_mut(&mut state.colors)[0].1 = BlendState {
            enabled: true,
            src_color: vk::BlendFactor::ONE,
            dst_color: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
            color_op: vk::BlendOp::ADD,
            src_alpha: vk::BlendFactor::ONE,
            dst_alpha: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
            alpha_op: vk::BlendOp::ADD,
            write_mask: vk::ColorComponentFlags::RGBA,
        };
    }
    state
}
struct Texture {
    view: Arc<AllocatedImageView>,
    size: [usize; 2],
    sampler: Arc<Sampler>,
}
pub(crate) struct ScreenRenderer {
    pipelines: HashMap<(vk::Format, u8), Arc<NativePipeline>>,
    linear: Arc<Sampler>,
    nearest: Arc<Sampler>,
    textures: HashMap<egui::TextureId, Texture>,
    vertices: Vec<u8>,
    indices: Vec<u8>,
}
impl ScreenRenderer {
    pub fn new(device: &VulkanDevice) -> Result<Self, VulkanError> {
        let make = |filter| {
            Sampler::new(
                device,
                &vk::SamplerCreateInfo::default()
                    .mag_filter(filter)
                    .min_filter(filter)
                    .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .max_lod(0.),
            )
        };
        Ok(Self {
            pipelines: HashMap::default(),
            linear: make(vk::Filter::LINEAR)?,
            nearest: make(vk::Filter::NEAREST)?,
            textures: HashMap::default(),
            vertices: Vec::new(),
            indices: Vec::new(),
        })
    }
    pub fn pipeline(
        &mut self,
        compiler: &PipelineCompiler,
        format: vk::Format,
        kind: u8,
    ) -> Result<Arc<NativePipeline>, VulkanError> {
        if let std::collections::hash_map::Entry::Vacant(entry) =
            self.pipelines.entry((format, kind))
        {
            let pending = compiler
                .request(PipelineRecipe::Graphics {
                    shader: shader(kind),
                    state: state(format, kind),
                })
                .map_err(|e| VulkanError::Compilation(e.to_string().into()))?;
            entry.insert(pending);
        }
        Ok(Arc::clone(&self.pipelines[&(format, kind)]))
    }
    pub fn update_textures(
        &mut self,
        cache: &mut VulkanResidency,
        ui: &mut UiFrame,
        samplers: &mut SamplerCache,
    ) -> Result<(), VulkanError> {
        for (id, deltas) in &ui.textures.set {
            for delta in deltas {
                let egui::ImageData::Color(image) = &delta.image;
                let bytes: Vec<_> = image.pixels.iter().flat_map(|p| p.to_array()).collect();
                let position = delta.pos.unwrap_or([0, 0]);
                if delta.pos.is_none() {
                    if self.textures.len() >= 64 && !self.textures.contains_key(id) {
                        return Err(VulkanError::InvalidResource(
                            "UI texture count exceeds host budget",
                        ));
                    }
                    let extent = [image.size[0] as u32, image.size[1] as u32];
                    if extent.contains(&0) || extent.iter().any(|s| *s > 4096) {
                        return Err(VulkanError::InvalidResource(
                            "UI texture extent exceeds host budget",
                        ));
                    }
                    let native = cache.allocate_host_image(extent, vk::Format::R8G8B8A8_UNORM)?;
                    let view = native.view_2d(vk::Format::R8G8B8A8_UNORM)?;
                    self.textures.insert(
                        *id,
                        Texture {
                            view,
                            size: image.size,
                            sampler: Arc::clone(&self.linear),
                        },
                    );
                }
                if delta.options.mipmap_mode.is_some() {
                    return Err(VulkanError::InvalidResource(
                        "native UI textures do not generate mipmaps",
                    ));
                }
                let filter = |f| {
                    if f == egui::TextureFilter::Nearest {
                        vk::Filter::NEAREST
                    } else {
                        vk::Filter::LINEAR
                    }
                };
                let wrap = match delta.options.wrap_mode {
                    egui::TextureWrapMode::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
                    egui::TextureWrapMode::Repeat => vk::SamplerAddressMode::REPEAT,
                    egui::TextureWrapMode::MirroredRepeat => {
                        vk::SamplerAddressMode::MIRRORED_REPEAT
                    }
                };
                let sampler = samplers.get(
                    &vk::SamplerCreateInfo::default()
                        .min_filter(filter(delta.options.minification))
                        .mag_filter(filter(delta.options.magnification))
                        .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                        .address_mode_u(wrap)
                        .address_mode_v(wrap)
                        .address_mode_w(wrap)
                        .max_lod(0.),
                )?;
                self.textures
                    .get_mut(id)
                    .ok_or(VulkanError::InvalidResource(
                        "UI partial texture update has no allocation",
                    ))?
                    .sampler = sampler;
                let texture = self.textures.get(id).ok_or(VulkanError::InvalidResource(
                    "UI partial texture update has no allocation",
                ))?;
                if position[0]
                    .checked_add(image.size[0])
                    .is_none_or(|end| end > texture.size[0])
                    || position[1]
                        .checked_add(image.size[1])
                        .is_none_or(|end| end > texture.size[1])
                {
                    return Err(VulkanError::InvalidResource(
                        "UI texture update exceeds allocation",
                    ));
                }
                cache.device_mut().end_rendering();
                let device = cache.device_mut();
                let slice = device
                    .submission
                    .reserve_upload(&mut device.dispatch, &bytes)?;
                let device = cache.device_mut();

                let old = if delta.pos.is_none() {
                    vk::ImageLayout::UNDEFINED
                } else {
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                };
                transition(
                    device,
                    texture.view.image.handle,
                    old,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::PipelineStageFlags2::FRAGMENT_SHADER,
                    vk::AccessFlags2::SHADER_SAMPLED_READ,
                    vk::PipelineStageFlags2::TRANSFER,
                    vk::AccessFlags2::TRANSFER_WRITE,
                );
                let copy = vk::BufferImageCopy::default()
                    .buffer_offset(slice.offset)
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .layer_count(1),
                    )
                    .image_offset(vk::Offset3D {
                        x: position[0] as i32,
                        y: position[1] as i32,
                        z: 0,
                    })
                    .image_extent(vk::Extent3D {
                        width: image.size[0] as u32,
                        height: image.size[1] as u32,
                        depth: 1,
                    });
                unsafe {
                    device.dispatch.cmd_copy_buffer_to_image(
                        device.submission.staging_buffer(&slice).handle,
                        texture.view.image.handle,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &[copy],
                    );
                }
                transition(
                    device,
                    texture.view.image.handle,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    vk::PipelineStageFlags2::TRANSFER,
                    vk::AccessFlags2::TRANSFER_WRITE,
                    vk::PipelineStageFlags2::FRAGMENT_SHADER,
                    vk::AccessFlags2::SHADER_SAMPLED_READ,
                );
                let work = cache.device_mut().work_ticket()?;
                cache
                    .device_mut()
                    .retire_image_views(vec![Arc::clone(&texture.view)], work)?;
            }
        }
        ui.textures.set.clear();
        Ok(())
    }
    pub fn free_textures(&mut self, ui: &mut UiFrame) {
        for id in ui.textures.free.drain() {
            self.textures.remove(&id);
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub fn screen(
        &self,
        cache: &mut VulkanResidency,
        descriptors: &mut DescriptorCache,
        pipeline: Arc<NativePipeline>,
        frame: &frames::Snapshot,
        source: &Source,
        target: vk::ImageView,
        extent: vk::Extent2D,
        format: vk::Format,
        letterbox: bool,
    ) -> Result<(), VulkanError> {
        if source.crop.width == 0
            || source.crop.height == 0
            || source
                .crop
                .left
                .checked_add(source.crop.width)
                .is_none_or(|x| x > frame.extent[0])
            || source
                .crop
                .top
                .checked_add(source.crop.height)
                .is_none_or(|y| y > frame.extent[1])
        {
            return Err(VulkanError::InvalidResource(
                "presentation crop exceeds native source extent",
            ));
        }
        let sampler = if frame.format == vk::Format::R16_UINT {
            &self.nearest
        } else {
            &self.linear
        };
        let record = descriptors.get(
            &pipeline.layout,
            &[
                DescriptorResource::Image {
                    view: Arc::clone(&frame.view),
                    layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                },
                DescriptorResource::Sampler(Arc::clone(sampler)),
            ],
        )?;
        let mut parameters = GraphicsParameters::new(
            state(
                format,
                if frame.format == vk::Format::R16_UINT {
                    1
                } else {
                    0
                },
            ),
            extent,
        );
        if letterbox {
            let [width, height] = source.dimensions().map(u64::from);
            let ow = u64::from(extent.width);
            let oh = u64::from(extent.height);
            let (w, h) = if ow * height <= oh * width {
                (ow, ow * height / width)
            } else {
                (oh * width / height, oh)
            };
            parameters.viewports[0] = vk::Viewport {
                x: ((ow - w) / 2) as f32,
                y: ((oh - h) / 2) as f32,
                width: w as f32,
                height: h as f32,
                min_depth: 0.,
                max_depth: 1.,
            };
        }
        let graphics = cache.prepare_host_graphics(pipeline, record, parameters)?;
        let mut push = [0u8; 32];
        for (i, value) in [
            source.crop.left as f32 / frame.extent[0] as f32,
            source.crop.top as f32 / frame.extent[1] as f32,
            source.crop.width as f32 / frame.extent[0] as f32,
            source.crop.height as f32 / frame.extent[1] as f32,
        ]
        .into_iter()
        .enumerate()
        {
            push[i * 4..i * 4 + 4].copy_from_slice(&value.to_le_bytes());
        }
        let bits = u32::from(source.transform.flip_horizontal)
            | (u32::from(source.transform.flip_vertical) << 1)
            | (u32::from(source.transform.rotate_90_clockwise) << 2);
        for (i, value) in [
            bits,
            0,
            u32::from(frame.encoding == nixe_gpu::PresentationImageFormat::Rgbx8),
            u32::from(frame.encoding == nixe_gpu::PresentationImageFormat::Rgb565),
        ]
        .into_iter()
        .enumerate()
        {
            push[16 + i * 4..20 + i * 4].copy_from_slice(&value.to_le_bytes());
        }
        cache.bind_host_graphics(&graphics, &push)?;
        let device = cache.device_mut();

        let attachments = [vk::RenderingAttachmentInfo::default()
            .image_view(target)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0., 0., 0., 1.],
                },
            })];
        let info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: Default::default(),
                extent,
            })
            .layer_count(1)
            .color_attachments(&attachments);
        unsafe {
            device.dispatch.begin_rendering(&info);
            device.dispatch.cmd_draw(3, 1, 0, 0);
            device.dispatch.end_rendering();
        }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub fn ui(
        &mut self,
        cache: &mut VulkanResidency,
        descriptors: &mut DescriptorCache,
        pipeline: Arc<NativePipeline>,
        ui: &mut UiFrame,
        target: vk::ImageView,
        extent: vk::Extent2D,
        format: vk::Format,
    ) -> Result<(), VulkanError> {
        if !ui.pixels_per_point.is_finite() || ui.pixels_per_point <= 0. {
            return Err(VulkanError::InvalidResource("invalid UI pixel scale"));
        }
        self.vertices.clear();
        self.indices.clear();
        let mut ranges = Vec::new();
        for primitive in &ui.primitives {
            let egui::epaint::Primitive::Mesh(mesh) = &primitive.primitive else {
                return Err(VulkanError::InvalidResource(
                    "native UI supports meshes, not custom callbacks",
                ));
            };
            if !mesh.is_valid() {
                return Err(VulkanError::InvalidResource("invalid UI mesh indices"));
            }
            let base = (self.vertices.len() / 20) as i32;
            let first = (self.indices.len() / 4) as u32;
            for vertex in mesh.vertices.iter() {
                for value in [vertex.pos.x, vertex.pos.y, vertex.uv.x, vertex.uv.y] {
                    self.vertices.extend_from_slice(&value.to_le_bytes());
                }
                self.vertices.extend_from_slice(&vertex.color.to_array());
            }
            for index in &mesh.indices {
                self.indices.extend_from_slice(&index.to_le_bytes());
            }
            ranges.push((
                mesh.texture_id,
                primitive.clip_rect,
                first,
                mesh.indices.len() as u32,
                base,
            ));
        }
        if self.vertices.len() + self.indices.len() > 4 * 1024 * 1024 {
            return Err(VulkanError::InvalidResource(
                "UI meshes exceed host transient budget",
            ));
        }
        if self.indices.is_empty() {
            self.free_textures(ui);
            return Ok(());
        }
        // The owner staging arena also serves vertex/index fetch directly;
        // these slices cannot recycle before this command slot completes.
        let device = cache.device_mut();
        let slot = device.begin_recording()?;
        let vertices = device
            .submission
            .reserve_upload(&mut device.dispatch, &self.vertices)?;
        let indices = device
            .submission
            .reserve_upload(&mut device.dispatch, &self.indices)?;
        let vertex_handle = cache
            .device_mut()
            .submission
            .staging_buffer(&vertices)
            .handle;
        let index_handle = cache
            .device_mut()
            .submission
            .staging_buffer(&indices)
            .handle;
        let mut push = [0u8; 8];
        push[..4].copy_from_slice(&(extent.width as f32 / ui.pixels_per_point).to_le_bytes());
        push[4..8].copy_from_slice(&(extent.height as f32 / ui.pixels_per_point).to_le_bytes());
        cache.device_mut().end_rendering();
        cache.device_mut().begin_recording()?;

        let dependency = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(
                vk::AccessFlags2::COLOR_ATTACHMENT_READ | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            )];
        unsafe {
            cache
                .device_mut()
                .dispatch
                .barrier(&vk::DependencyInfo::default().memory_barriers(&dependency));
        }
        let attachments = [vk::RenderingAttachmentInfo::default()
            .image_view(target)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(vk::AttachmentStoreOp::STORE)];
        let info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: Default::default(),
                extent,
            })
            .layer_count(1)
            .color_attachments(&attachments);
        unsafe {
            cache.device_mut().dispatch.begin_rendering(&info);
        }
        let draw_result = (|| -> Result<(), VulkanError> {
            for (texture, clip, first, count, base) in ranges {
                let texture = self
                    .textures
                    .get(&texture)
                    .ok_or(VulkanError::InvalidResource(
                        "UI draw references an absent texture",
                    ))?;
                let record = descriptors.get(
                    &pipeline.layout,
                    &[
                        DescriptorResource::Image {
                            view: Arc::clone(&texture.view),
                            layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        },
                        DescriptorResource::Sampler(Arc::clone(&texture.sampler)),
                    ],
                )?;
                let mut parameters = GraphicsParameters::new(state(format, 2), extent);
                let min = (clip.min.to_vec2() * ui.pixels_per_point)
                    .floor()
                    .max(egui::Vec2::ZERO);
                let max = (clip.max.to_vec2() * ui.pixels_per_point)
                    .ceil()
                    .min(egui::vec2(extent.width as f32, extent.height as f32));
                if max.x <= min.x || max.y <= min.y {
                    continue;
                }
                parameters.scissors[0] = vk::Rect2D {
                    offset: vk::Offset2D {
                        x: min.x as i32,
                        y: min.y as i32,
                    },
                    extent: vk::Extent2D {
                        width: (max.x - min.x) as u32,
                        height: (max.y - min.y) as u32,
                    },
                };
                let graphics =
                    cache.prepare_host_graphics(Arc::clone(&pipeline), record, parameters)?;
                cache.bind_host_graphics(&graphics, &push)?;
                let device = cache.device_mut();

                unsafe {
                    device.dispatch.cmd_bind_vertex_buffers(
                        0,
                        &[vertex_handle],
                        &[vertices.offset],
                    );
                    device.dispatch.cmd_bind_index_buffer(
                        index_handle,
                        indices.offset,
                        vk::IndexType::UINT32,
                    );
                    device.dispatch.cmd_draw_indexed(count, 1, first, base, 0);
                }
                let work = cache.device_mut().work_ticket()?;
                cache
                    .device_mut()
                    .retire_image_views(vec![Arc::clone(&texture.view)], work)?;
            }
            Ok(())
        })();
        unsafe {
            cache.device_mut().dispatch.end_rendering();
        }
        // Host UI changed raw vertex/index binds; invalidate guest bind knowledge.
        cache.device_mut().submission.slots[slot]
            .vertex_bindings
            .clear();
        cache.device_mut().submission.slots[slot].index_binding = None;
        draw_result?;
        self.free_textures(ui);
        Ok(())
    }
}
