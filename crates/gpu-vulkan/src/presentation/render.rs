//! Presentation and the bounded egui mesh contract use the ordinary native
//! pipeline/descriptor caches, command slots, allocator and timeline.
use super::filters::{self, FilterTarget, Program};
use super::*;
use crate::cache::*;
use ash::vk;
use nixe_video::ScalingFilter;
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
fn state(format: vk::Format, ui: bool) -> GraphicsState {
    let mut state = GraphicsState::color(format);
    if ui {
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
    sampler: Sampler,
}
pub(crate) struct LayerImage {
    pub view: Arc<AllocatedImageView>,
    pub extent: [u32; 2],
    pub format: vk::Format,
    pub encoding: nixe_gpu::PresentationImageFormat,
    pub crop: nixe_gpu::FrameCrop,
    pub transform: nixe_gpu::FrameTransform,
}
impl LayerImage {
    pub fn dimensions(&self) -> [u32; 2] {
        if self.transform.rotate_90_clockwise {
            [self.crop.height, self.crop.width]
        } else {
            [self.crop.width, self.crop.height]
        }
    }
    pub fn content_rect(&self, extent: [u32; 2]) -> nixe_gpu::FrameCrop {
        let [width, height] = self.dimensions().map(u64::from);
        let [ow, oh] = extent.map(u64::from);
        let (w, h) = if ow * height <= oh * width {
            (ow, ow * height / width)
        } else {
            (oh * width / height, oh)
        };
        nixe_gpu::FrameCrop {
            left: ((ow - w) / 2) as u32,
            top: ((oh - h) / 2) as u32,
            width: w as u32,
            height: h as u32,
        }
    }
}
pub(crate) struct ScreenRenderer {
    pipelines: HashMap<(vk::Format, Program), NativePipeline>,
    pub(super) filter: ScalingFilter,
    half: bool,
    linear: Sampler,
    nearest: Sampler,
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
            filter: device.scaling_filter,
            half: device.profile().shader_compiler.host.support_float16,
            linear: make(vk::Filter::LINEAR)?,
            nearest: make(vk::Filter::NEAREST)?,
            textures: HashMap::default(),
            vertices: Vec::new(),
            indices: Vec::new(),
        })
    }
    pub(super) fn pipeline(
        &mut self,
        compiler: &PipelineCompiler,
        format: vk::Format,
        program: Program,
    ) -> Result<NativePipeline, VulkanError> {
        if let std::collections::hash_map::Entry::Vacant(entry) =
            self.pipelines.entry((format, program))
        {
            let pending = compiler
                .request(PipelineRecipe::Graphics {
                    shader: program.shader(),
                    state: state(format, program == Program::Ui),
                })
                .map_err(|e| VulkanError::Compilation(e.to_string().into()))?;
            entry.insert(pending);
        }
        Ok((self.pipelines[&(format, program)]).clone())
    }
    pub fn update_textures(
        &mut self,
        cache: &mut crate::residency::Residency<'_>,
        ui: &mut UiFrame,
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
                            sampler: self.linear.clone(),
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
                let sampler = (cache.texture_cache.samplers.get(
                    &vk::SamplerCreateInfo::default()
                        .min_filter(filter(delta.options.minification))
                        .mag_filter(filter(delta.options.magnification))
                        .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                        .address_mode_u(wrap)
                        .address_mode_v(wrap)
                        .address_mode_w(wrap)
                        .max_lod(0.),
                )?)
                .clone();
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
                cache.device_mut().end_rendering()?;
                let device = cache.device_mut();
                let slice = device.submission.reserve_upload(&bytes)?;
                let device = cache.device_mut();

                let old = if delta.pos.is_none() {
                    vk::ImageLayout::UNDEFINED
                } else {
                    vk::ImageLayout::GENERAL
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
                    vk::ImageLayout::GENERAL,
                    vk::PipelineStageFlags2::TRANSFER,
                    vk::AccessFlags2::TRANSFER_WRITE,
                    vk::PipelineStageFlags2::FRAGMENT_SHADER,
                    vk::AccessFlags2::SHADER_SAMPLED_READ,
                );
                let work = cache.device_mut().current_work();
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
        cache: &mut crate::residency::Residency<'_>,
        descriptors: &mut DescriptorCache,
        pipeline: NativePipeline,
        frame: &LayerImage,
        target: &Arc<crate::render_pass::Framebuffer>,
        extent: vk::Extent2D,
        format: vk::Format,
        letterbox: bool,
        filter: ScalingFilter,
    ) -> Result<(), VulkanError> {
        if frame.crop.width == 0
            || frame.crop.height == 0
            || frame
                .crop
                .left
                .checked_add(frame.crop.width)
                .is_none_or(|x| x > frame.extent[0])
            || frame
                .crop
                .top
                .checked_add(frame.crop.height)
                .is_none_or(|y| y > frame.extent[1])
        {
            return Err(VulkanError::InvalidResource(
                "presentation crop exceeds native source extent",
            ));
        }
        let mut parameters = GraphicsParameters::new(state(format, false), extent);
        if letterbox {
            let [width, height] = frame.dimensions().map(u64::from);
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
        let mut push = [0u8; 40];
        for (i, value) in [
            frame.crop.left as f32 / frame.extent[0] as f32,
            frame.crop.top as f32 / frame.extent[1] as f32,
            frame.crop.width as f32 / frame.extent[0] as f32,
            frame.crop.height as f32 / frame.extent[1] as f32,
        ]
        .into_iter()
        .enumerate()
        {
            push[i * 4..i * 4 + 4].copy_from_slice(&value.to_le_bytes());
        }
        let bits = u32::from(frame.transform.flip_horizontal)
            | (u32::from(frame.transform.flip_vertical) << 1)
            | (u32::from(frame.transform.rotate_90_clockwise) << 2);
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
        let mut size = [
            parameters.viewports[0].width,
            parameters.viewports[0].height,
        ];
        if frame.transform.rotate_90_clockwise {
            size.swap(0, 1);
        }
        // Area sampling integrates only the cropped source, even with letterboxing.
        size[0] *= frame.extent[0] as f32 / frame.crop.width as f32;
        size[1] *= frame.extent[1] as f32 / frame.crop.height as f32;
        for (i, value) in size.into_iter().enumerate() {
            push[32 + i * 4..36 + i * 4].copy_from_slice(&value.to_le_bytes());
        }
        self.draw_source(
            cache,
            descriptors,
            pipeline,
            frame,
            target,
            extent,
            parameters,
            &push,
            filter == ScalingFilter::Nearest,
        )
    }
    pub fn screen_pipeline(
        &mut self,
        compiler: &PipelineCompiler,
        format: vk::Format,
        filter: ScalingFilter,
        frame: &LayerImage,
    ) -> Result<NativePipeline, VulkanError> {
        self.pipeline(
            compiler,
            format,
            Program::filter(filter, frame.format == vk::Format::R16_UINT, self.half),
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn draw_source(
        &self,
        cache: &mut crate::residency::Residency<'_>,
        descriptors: &mut DescriptorCache,
        pipeline: NativePipeline,
        frame: &LayerImage,
        target: &Arc<crate::render_pass::Framebuffer>,
        extent: vk::Extent2D,
        parameters: GraphicsParameters,
        push: &[u8],
        nearest: bool,
    ) -> Result<(), VulkanError> {
        let sampler = if nearest || frame.format == vk::Format::R16_UINT {
            &self.nearest
        } else {
            &self.linear
        };
        let record = descriptors.get(
            &pipeline.layout,
            &[
                DescriptorResource::Image {
                    view: Arc::clone(&frame.view),
                    layout: vk::ImageLayout::GENERAL,
                },
                DescriptorResource::Sampler(sampler.clone()),
            ],
        )?;
        let graphics = cache.prepare_host_graphics(pipeline, record, parameters)?;
        cache.bind_host_graphics(&graphics, push)?;
        let device = cache.device_mut();
        let mut pass = target.key;
        pass.color_clear = 1;
        device.begin_render_pass(
            target,
            pass,
            vk::Rect2D {
                offset: Default::default(),
                extent,
            },
            &[vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0., 0., 0., 1.],
                },
            }],
        )?;
        unsafe {
            device.dispatch.cmd_draw(3, 1, 0, 0);
            device.dispatch.end_rendering();
        }
        Ok(())
    }
    pub(super) fn fsr(
        &mut self,
        cache: &mut crate::residency::Residency<'_>,
        compiler: &PipelineCompiler,
        descriptors: &mut DescriptorCache,
        source: &LayerImage,
        targets: &[FilterTarget; 2],
    ) -> Result<LayerImage, VulkanError> {
        let mut layer = LayerImage {
            view: Arc::clone(&source.view),
            extent: source.extent,
            format: source.format,
            encoding: source.encoding,
            crop: source.crop,
            transform: nixe_gpu::FrameTransform::default(),
        };
        for (index, target) in targets.iter().enumerate() {
            let program = if index == 0 {
                Program::Easu {
                    packed: source.format == vk::Format::R16_UINT,
                    half: self.half,
                }
            } else {
                Program::Rcas { half: self.half }
            };
            let pipeline = self.pipeline(compiler, vk::Format::R16G16B16A16_SFLOAT, program)?;
            let push = filters::fsr_constants(&layer, target.extent, index == 0);
            let extent = vk::Extent2D {
                width: target.extent[0],
                height: target.extent[1],
            };
            cache.device_mut().end_rendering()?;
            // The frame slot is reused only after its previous graphics and WSI work completes.
            transition(
                cache.device_mut(),
                target.view.image.handle,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
                vk::PipelineStageFlags2::NONE,
                vk::AccessFlags2::NONE,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            );
            self.draw_source(
                cache,
                descriptors,
                pipeline,
                &layer,
                &target.framebuffer,
                extent,
                GraphicsParameters::new(state(vk::Format::R16G16B16A16_SFLOAT, false), extent),
                &push,
                false,
            )?;
            transition(
                cache.device_mut(),
                target.view.image.handle,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::GENERAL,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                vk::PipelineStageFlags2::FRAGMENT_SHADER,
                vk::AccessFlags2::SHADER_SAMPLED_READ,
            );
            let work = cache.device_mut().current_work();
            cache
                .device_mut()
                .retire_image_views(vec![Arc::clone(&target.view)], work)?;
            layer = target.layer(source.encoding, nixe_gpu::FrameTransform::default());
        }
        layer.transform = source.transform;
        Ok(layer)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn ui(
        &mut self,
        cache: &mut crate::residency::Residency<'_>,
        descriptors: &mut DescriptorCache,
        pipeline: NativePipeline,
        ui: &mut UiFrame,
        target: &Arc<crate::render_pass::Framebuffer>,
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
        let slot = device.recording_slot()?;
        let vertices = device.submission.reserve_upload(&self.vertices)?;
        let indices = device.submission.reserve_upload(&self.indices)?;
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
        cache.device_mut().end_rendering()?;

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
        cache.device_mut().begin_render_pass(
            target,
            target.key,
            vk::Rect2D {
                offset: Default::default(),
                extent,
            },
            &[],
        )?;
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
                            layout: vk::ImageLayout::GENERAL,
                        },
                        DescriptorResource::Sampler((texture.sampler).clone()),
                    ],
                )?;
                let mut parameters = GraphicsParameters::new(state(format, true), extent);
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
                let graphics = cache.prepare_host_graphics(pipeline.clone(), record, parameters)?;
                cache.bind_host_graphics(&graphics, &push)?;
                let device = cache.device_mut();

                unsafe {
                    if device.profile().dynamic_state.base {
                        device.dispatch.cmd_bind_vertex_buffers2(
                            0,
                            &[vertex_handle],
                            &[vertices.offset],
                            &[vertices.size],
                            &[std::mem::size_of::<egui::epaint::Vertex>() as u64],
                        );
                    } else {
                        device.dispatch.cmd_bind_vertex_buffers(
                            0,
                            &[vertex_handle],
                            &[vertices.offset],
                        );
                    }
                    device.dispatch.cmd_bind_index_buffer(
                        index_handle,
                        indices.offset,
                        vk::IndexType::UINT32,
                    );
                    device.dispatch.cmd_draw_indexed(count, 1, first, base, 0);
                }
                let work = cache.device_mut().current_work();
                cache
                    .device_mut()
                    .retire_image_views(vec![Arc::clone(&texture.view)], work)?;
            }
            Ok(())
        })();
        unsafe {
            cache.device_mut().dispatch.end_rendering();
        }
        // Host UI changed raw vertex binds; invalidate the current guest channel.
        cache.device_mut().submission.slots[slot].vertex_channel = None;
        draw_result?;
        self.free_textures(ui);
        Ok(())
    }
}
