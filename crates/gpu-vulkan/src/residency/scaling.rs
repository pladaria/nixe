//! Original and scaled texture storage, selected by the owning texture cache.
//! https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/texture_cache/texture_cache.h#L365
use super::*;
use crate::cache::{
    DescriptorCache, DescriptorResource, GraphicsState, NativePipeline, PipelineCompiler,
    PipelineRecipe, ShaderArtifact,
};
use crate::{AttachmentSource, Draw, GraphicsParameters, RenderAttachment, RenderingTargets};
use nixe_gpu::{
    ShaderStage,
    shader_module::{Binding, DescriptorKind, ResourceId},
};
use std::sync::OnceLock;
use std::time::Duration;

pub(super) struct Helpers {
    compiler: Arc<PipelineCompiler>,
    descriptors: DescriptorCache,
    pipelines: HashMap<(vk::Format, vk::SampleCountFlags, u8), NativePipeline>,
}

fn shader(kind: u8, multisample: bool) -> Arc<ShaderArtifact> {
    static SHADERS: [OnceLock<Arc<ShaderArtifact>>; 10] = [const { OnceLock::new() }; 10];
    let index = kind as usize + usize::from(multisample) * 5;
    Arc::clone(SHADERS[index].get_or_init(|| {
        let modules: [&[u8]; 10] = [
            include_bytes!("../../shaders/spv/scale-color.frag.spv"),
            include_bytes!("../../shaders/spv/scale-uint.frag.spv"),
            include_bytes!("../../shaders/spv/scale-sint.frag.spv"),
            include_bytes!("../../shaders/spv/scale-depth.frag.spv"),
            include_bytes!("../../shaders/spv/scale-depth-stencil.frag.spv"),
            include_bytes!("../../shaders/spv/scale-color-msaa.frag.spv"),
            include_bytes!("../../shaders/spv/scale-uint-msaa.frag.spv"),
            include_bytes!("../../shaders/spv/scale-sint-msaa.frag.spv"),
            include_bytes!("../../shaders/spv/scale-depth-msaa.frag.spv"),
            include_bytes!("../../shaders/spv/scale-depth-stencil-msaa.frag.spv"),
        ];
        Arc::new(ShaderArtifact::new(Box::new([
            helper_stage(
                ShaderStage::Vertex,
                include_bytes!("../../shaders/spv/solid.vert.spv"),
                Box::new([]),
                0,
            ),
            helper_stage(
                ShaderStage::Fragment,
                modules[index],
                (0..if kind == 4 { 2 } else { 1 })
                    .map(|i| Binding {
                        resource: ResourceId::new(i),
                        binding: i,
                        kind: if multisample {
                            DescriptorKind::SampledImage
                        } else {
                            DescriptorKind::CombinedImageSampler
                        },
                        count: 1,
                        sampler: None,
                        readable: true,
                        writable: false,
                    })
                    .collect(),
                8,
            ),
        ])))
    }))
}

fn color_kind(format: vk::Format) -> u8 {
    use vk::Format as F;
    match format {
        F::R8_UINT
        | F::R8G8_UINT
        | F::R8G8B8_UINT
        | F::R8G8B8A8_UINT
        | F::B8G8R8A8_UINT
        | F::A8B8G8R8_UINT_PACK32
        | F::A2R10G10B10_UINT_PACK32
        | F::A2B10G10R10_UINT_PACK32
        | F::R16_UINT
        | F::R16G16_UINT
        | F::R16G16B16_UINT
        | F::R16G16B16A16_UINT
        | F::R32_UINT
        | F::R32G32_UINT
        | F::R32G32B32_UINT
        | F::R32G32B32A32_UINT => 1,
        F::R8_SINT
        | F::R8G8_SINT
        | F::R8G8B8_SINT
        | F::R8G8B8A8_SINT
        | F::B8G8R8A8_SINT
        | F::A8B8G8R8_SINT_PACK32
        | F::A2R10G10B10_SINT_PACK32
        | F::A2B10G10R10_SINT_PACK32
        | F::R16_SINT
        | F::R16G16_SINT
        | F::R16G16B16_SINT
        | F::R16G16B16A16_SINT
        | F::R32_SINT
        | F::R32G32_SINT
        | F::R32G32B32_SINT
        | F::R32G32B32A32_SINT => 2,
        _ => 0,
    }
}

impl TextureCache {
    pub(super) fn image_can_rescale(&mut self, id: ImageId) -> bool {
        let entry = &mut self.images[id];
        if !entry.can_rescale {
            return false;
        }
        if entry.rescaled || entry.checking_rescale || entry.rescale_checked {
            return true;
        }
        entry.checking_rescale = true;
        let aliases: smallvec::SmallVec<[_; 8]> = entry.aliases.iter().map(|a| a.image).collect();
        let result = aliases
            .into_iter()
            .all(|alias| self.image_can_rescale(alias));
        self.images[id].checking_rescale = false;
        self.images[id].rescale_checked = result;
        result
    }
    fn invalidate_scale(&mut self, id: ImageId) {
        self.images[id].scale_tick = self.images[id].scale_tick.max(self.frame_tick + 1);
        let removed = std::mem::take(&mut self.images[id].views);
        self.bindings.invalidate_image_views(&removed);
        self.remove_framebuffers(&removed);
        self.binding_revision += 1;
        let views = removed
            .into_iter()
            .map(|view| self.views.erase(view).native)
            .collect();
        let entry = &self.images[id];
        self.retired
            .push(Arc::clone(&entry.native), views, entry.contents.last_use);
    }
}
impl Residency<'_> {
    pub(crate) fn enable_scaling(&mut self, compiler: Arc<PipelineCompiler>) {
        self.texture_cache.scaling = Some(Helpers {
            compiler,
            descriptors: DescriptorCache::new(&self.runtime.device),
            pipelines: Default::default(),
        });
    }
    pub(crate) fn resolution(&self) -> crate::ResolutionScaling {
        self.runtime.device.resolution_scale
    }
    pub(crate) fn change_resolution(
        &mut self,
        resolution: crate::ResolutionScaling,
    ) -> Result<(), VulkanError> {
        let images: Vec<_> = self
            .texture_cache
            .images
            .iter()
            .filter_map(|(id, image)| image.scaled.is_some().then_some(id))
            .collect();
        let retained = self.runtime.constructing.len();
        self.runtime
            .constructing
            .extend(images.iter().copied().map(Resource::Image));
        let result = (|| {
            // Preserve every physical version with the old ratio before
            // replacing scaled storage. Alias modification order stays intact.
            self.runtime.device.end_rendering()?;
            for &image in &images {
                self.scale_image(image, false, false)?;
            }
            let work = self.flush()?;
            self.runtime.device.wait_for_work(work, Duration::MAX)?;
            for image in images {
                let entry = &mut self.texture_cache.images[image];
                entry.scaled = None;
                entry.scaled_initialized = false;
            }
            self.runtime.device.resolution_scale = resolution;
            for slot in &mut self.runtime.device.submission.slots {
                slot.graphics_state = Default::default();
            }
            self.texture_cache.bindings.reset_resolution();
            Ok(())
        })();
        self.runtime.constructing.truncate(retained);
        result
    }
    pub(crate) fn is_rescaling(&self) -> bool {
        self.texture_cache.bindings.channel.rescaled
    }
    pub(crate) fn image_is_rescaled(&self, view: ImageViewId) -> bool {
        self.texture_cache
            .get(self.texture_cache.views[view].image)
            .rescaled
    }
    pub(crate) fn image_extent(&self, id: ImageId, mut logical: vk::Extent3D) -> vk::Extent3D {
        if self.texture_cache.get(id).rescaled {
            let scale = self.resolution();
            logical.width = scale.scale(logical.width);
            logical.height = scale.scale(logical.height);
        }
        logical
    }
    pub(crate) fn image_rect(&self, id: ImageId, rect: vk::Rect2D) -> vk::Rect2D {
        if self.texture_cache.get(id).rescaled {
            self.scale_rect(rect)
        } else {
            rect
        }
    }
    pub(crate) fn scale_rect(&self, rect: vk::Rect2D) -> vk::Rect2D {
        let scale = self.resolution();
        let x = scale.scale_signed(rect.offset.x);
        let y = scale.scale_signed(rect.offset.y);
        let right = scale.scale_signed(rect.offset.x + rect.extent.width as i32);
        let bottom = scale.scale_signed(rect.offset.y + rect.extent.height as i32);
        vk::Rect2D {
            offset: vk::Offset2D { x, y },
            extent: vk::Extent2D {
                width: (right - x) as u32,
                height: (bottom - y) as u32,
            },
        }
    }
    /// Match all render targets, including depth, before choosing their views.
    pub(crate) fn rescale_render_targets(&mut self) -> Result<(), VulkanError> {
        if !self.resolution().active {
            return Ok(());
        }
        let ids: smallvec::SmallVec<[_; 9]> = self
            .texture_cache
            .bindings
            .channel
            .render_targets
            .views
            .iter()
            .flatten()
            .map(|&view| self.texture_cache.views[view].image)
            .collect();
        let mut can_rescale = !ids.is_empty();
        let mut any_rescaled = false;
        let mut rating = 0;
        for &id in &ids {
            can_rescale &= self.texture_cache.image_can_rescale(id);
            let entry = self.texture_cache.get(id);
            any_rescaled |= entry.rescaled
                || sync::native_image_aspects(entry.description.format)
                    != vk::ImageAspectFlags::COLOR;
            rating = rating.max(
                entry.scale_rating + u32::from(entry.scale_tick <= self.texture_cache.frame_tick),
            );
        }
        let rescaled = can_rescale && (any_rescaled || rating >= 2);
        rating = if !can_rescale {
            1
        } else if rescaled {
            2
        } else {
            rating
        };
        let next_tick = self.texture_cache.frame_tick + 1;
        for id in ids {
            let entry = &mut self.texture_cache.images[id];
            entry.scale_rating = rating;
            entry.scale_tick = next_tick;
            self.scale_image(id, rescaled, false)?;
        }
        let bank = &mut self.texture_cache.bindings.channel;
        if bank.rescaled != rescaled {
            bank.rescaled = rescaled;
            bank.render_targets_changed();
        }
        Ok(())
    }
    pub(super) fn match_image_scales(
        &mut self,
        source: ImageId,
        destination: ImageId,
    ) -> Result<(), VulkanError> {
        if self.texture_cache.get(source).rescaled == self.texture_cache.get(destination).rescaled {
            return Ok(());
        }
        let up = self.texture_cache.image_can_rescale(source)
            && self.texture_cache.image_can_rescale(destination);
        self.scale_image(source, up, false)?;
        self.scale_image(destination, up, false)
    }
    /// Fermi blits can convert between different scales. Resolves require
    /// matching extents and promote the single-sample destination and aliases.
    /// https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/texture_cache/texture_cache.h#L701
    pub(super) fn rescale_blit_images(
        &mut self,
        source: ImageId,
        destination: ImageId,
        resolve: bool,
    ) -> Result<(), VulkanError> {
        if self.texture_cache.get(source).rescaled == self.texture_cache.get(destination).rescaled {
            return Ok(());
        }
        if self.texture_cache.image_can_rescale(source) {
            self.scale_image(source, true, false)?;
            if resolve {
                self.texture_cache.images[destination].can_rescale = true;
                let aliases: smallvec::SmallVec<[_; 8]> = self
                    .texture_cache
                    .get(destination)
                    .aliases
                    .iter()
                    .map(|a| a.image)
                    .collect();
                for id in aliases {
                    self.texture_cache.images[id].can_rescale = true;
                }
            }
        }
        if self.texture_cache.image_can_rescale(destination) {
            self.scale_image(destination, true, false)?;
        }
        if resolve
            && self.texture_cache.get(source).rescaled
                != self.texture_cache.get(destination).rescaled
        {
            self.scale_image(source, false, false)?;
            self.scale_image(destination, false, false)?;
        }
        Ok(())
    }
    pub(super) fn scale_image(
        &mut self,
        id: ImageId,
        up: bool,
        ignore: bool,
    ) -> Result<(), VulkanError> {
        self.switch_image_scale(id, up, ignore, true)
    }
    /// RefreshContents temporarily selects original storage without invalidating
    /// views when the same scaled representation will immediately be restored.
    pub(super) fn switch_image_scale(
        &mut self,
        id: ImageId,
        up: bool,
        ignore: bool,
        invalidate: bool,
    ) -> Result<(), VulkanError> {
        if self.texture_cache.get(id).rescaled == up {
            return Ok(());
        }
        self.runtime.device.realize_deferred_clear()?;
        let scale = self.resolution();
        let description = self.texture_cache.get(id).description;
        if up && self.texture_cache.get(id).scaled.is_none() {
            let mut scaled = description;
            scaled.extent.width = scale.scale(scaled.extent.width);
            scaled.extent.height = scale.scale(scaled.extent.height);
            let owner = Arc::clone(&self.runtime.device.owner);
            let native = allocate_image_charged(
                &owner,
                scaled,
                AllocationMode::Managed,
                "scaled resident image",
                |bytes| self.charge(bytes),
            )?;
            self.texture_cache.images[id].scaled = Some(Arc::new(native));
            self.runtime.statistics.image_allocations += 1;
        }
        let source = self.use_image(id)?;
        let entry = &mut self.texture_cache.images[id];
        if entry.rescaled {
            entry.scaled_initialized = entry.initialized;
        } else {
            entry.original_initialized = entry.initialized;
        }
        let target = if up {
            Arc::clone(entry.scaled.as_ref().unwrap())
        } else {
            Arc::clone(&entry.original)
        };
        let initialized = if up {
            &mut entry.scaled_initialized
        } else {
            &mut entry.original_initialized
        };
        if !*initialized {
            self.runtime.barriers.queue_image(sync::initialize_image(
                target.handle,
                sync::native_image_aspects(description.format),
            ));
            *initialized = true;
        }
        if !ignore {
            self.runtime.barriers.emit(&mut self.runtime.device)?;
            self.blit_scale(id, source, target.handle, up)?;
        }
        let entry = &mut self.texture_cache.images[id];
        entry.native = target;
        entry.rescaled = up;
        entry.initialized = true;
        if invalidate {
            self.texture_cache.invalidate_scale(id);
        }
        Ok(())
    }
    fn blit_scale(
        &mut self,
        id: ImageId,
        source: vk::Image,
        destination: vk::Image,
        up: bool,
    ) -> Result<(), VulkanError> {
        let entry = self.texture_cache.get(id);
        let d = entry.description;
        let aspects = sync::native_image_aspects(d.format);
        let properties = self
            .runtime
            .device
            .format_properties(d.format)
            .optimal_tiling_features;
        if d.samples != vk::SampleCountFlags::TYPE_1
            || !properties
                .contains(vk::FormatFeatureFlags::BLIT_SRC | vk::FormatFeatureFlags::BLIT_DST)
        {
            return self.blit_scale_helper(id, source, destination, up);
        }
        let scale = self.resolution();
        let from = vk::Offset3D {
            x: d.extent.width as i32,
            y: d.extent.height as i32,
            z: 1,
        };
        let to = vk::Offset3D {
            x: scale.scale(d.extent.width) as i32,
            y: scale.scale(d.extent.height) as i32,
            z: 1,
        };
        let (from, to) = if up { (from, to) } else { (to, from) };
        let mut copies = smallvec::SmallVec::<[vk::ImageBlit; 2]>::new();
        for aspect in [
            vk::ImageAspectFlags::COLOR,
            vk::ImageAspectFlags::DEPTH,
            vk::ImageAspectFlags::STENCIL,
        ] {
            if !aspects.contains(aspect) {
                continue;
            }
            let sub = vk::ImageSubresourceLayers::default()
                .aspect_mask(aspect)
                .layer_count(d.array_layers);
            copies.push(
                vk::ImageBlit::default()
                    .src_subresource(sub)
                    .dst_subresource(sub)
                    .src_offsets([vk::Offset3D::default(), from])
                    .dst_offsets([vk::Offset3D::default(), to]),
            );
        }
        let range = vk::ImageSubresourceRange::default()
            .aspect_mask(aspects)
            .level_count(d.mip_levels)
            .layer_count(d.array_layers);
        let filter = if aspects == vk::ImageAspectFlags::COLOR
            && properties.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR)
        {
            vk::Filter::LINEAR
        } else {
            vk::Filter::NEAREST
        };
        unsafe {
            self.runtime.device.dispatch.blit_image(
                source,
                range,
                destination,
                range,
                &copies,
                filter,
            );
        }
        Ok(())
    }

    fn blit_scale_helper(
        &mut self,
        id: ImageId,
        source: vk::Image,
        destination: vk::Image,
        up: bool,
    ) -> Result<(), VulkanError> {
        let mut helpers = self
            .texture_cache
            .scaling
            .take()
            .ok_or(VulkanError::InvalidResource(
                "image scaling requires the shared pipeline compiler",
            ))?;
        let result = self.record_scale_helper(id, source, destination, up, &mut helpers);
        self.texture_cache.scaling = Some(helpers);
        result
    }
    fn record_scale_helper(
        &mut self,
        id: ImageId,
        source: vk::Image,
        destination: vk::Image,
        up: bool,
        helpers: &mut Helpers,
    ) -> Result<(), VulkanError> {
        let entry = self.texture_cache.get(id);
        let d = entry.description;
        let source_image = Arc::clone(&entry.native);
        let target_image = if up {
            Arc::clone(entry.scaled.as_ref().unwrap())
        } else {
            Arc::clone(&entry.original)
        };
        debug_assert_eq!(source_image.handle, source);
        debug_assert_eq!(target_image.handle, destination);
        let aspects = sync::native_image_aspects(d.format);
        let depth = aspects.contains(vk::ImageAspectFlags::DEPTH);
        let stencil = aspects.contains(vk::ImageAspectFlags::STENCIL)
            && self
                .runtime
                .device
                .profile
                .enabled_extensions
                .iter()
                .any(|s| s == "VK_EXT_shader_stencil_export");
        let kind = if depth {
            if stencil { 4 } else { 3 }
        } else {
            color_kind(d.format)
        };
        let multisample = d.samples != vk::SampleCountFlags::TYPE_1;
        let mut state = GraphicsState::color(d.format);
        state.samples = d.samples;
        if depth {
            state.colors = Arc::from([]);
            state.depth_format = d.format;
            state.depth_test = true;
            state.depth_write = true;
            if stencil {
                state.stencil_format = d.format;
                state.stencil_test = true;
                state.front.pass = vk::StencilOp::REPLACE;
                state.back.pass = vk::StencilOp::REPLACE;
            }
        }
        let key = (d.format, d.samples, kind);
        let pipeline = if let Some(pipeline) = helpers.pipelines.get(&key) {
            pipeline.clone()
        } else {
            let pipeline = helpers
                .compiler
                .request(PipelineRecipe::Graphics {
                    shader: shader(kind, multisample),
                    state: state.clone(),
                })
                .map_err(|e| VulkanError::Compilation(e.to_string().into()))?;
            helpers.pipelines.insert(key, pipeline.clone());
            pipeline
        };
        let scale = self.resolution();
        let mut extent = vk::Extent2D {
            width: d.extent.width,
            height: d.extent.height,
        };
        if up {
            extent.width = scale.scale(extent.width);
            extent.height = scale.scale(extent.height);
        }
        let filter = if !depth && kind == 0 {
            vk::Filter::LINEAR
        } else {
            vk::Filter::NEAREST
        };
        let sampler = self
            .texture_cache
            .samplers
            .get(
                &vk::SamplerCreateInfo::default()
                    .min_filter(filter)
                    .mag_filter(filter)
                    .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
            )?
            .clone();
        let view = |image: Arc<AllocatedImage>,
                    aspects,
                    layer|
         -> Result<Arc<crate::AllocatedImageView>, VulkanError> {
            let range = vk::ImageSubresourceRange::default()
                .aspect_mask(aspects)
                .base_array_layer(layer)
                .layer_count(1)
                .level_count(1);
            let handle = unsafe {
                image.owner.device.create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(image.handle)
                        .format(d.format)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .subresource_range(range),
                    None,
                )
            }
            .map_err(|e| VulkanError::vk("scaling image view", e))?;
            Ok(Arc::new(crate::AllocatedImageView::from_handle(
                image, handle, range,
            )))
        };
        for layer in 0..d.array_layers {
            self.runtime.barriers.queue_image(
                vk::ImageMemoryBarrier2::default()
                    .image(source)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(aspects)
                            .level_count(1)
                            .base_array_layer(layer)
                            .layer_count(1),
                    )
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                    .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ),
            );
            self.runtime.barriers.emit(&mut self.runtime.device)?;
            let source_view = view(
                Arc::clone(&source_image),
                if depth {
                    vk::ImageAspectFlags::DEPTH
                } else {
                    vk::ImageAspectFlags::COLOR
                },
                layer,
            )?;
            let resource = |view| {
                if multisample {
                    DescriptorResource::Image {
                        view,
                        layout: vk::ImageLayout::GENERAL,
                    }
                } else {
                    DescriptorResource::CombinedImageSampler {
                        view,
                        layout: vk::ImageLayout::GENERAL,
                        sampler: sampler.clone(),
                    }
                }
            };
            let mut resources: smallvec::SmallVec<[DescriptorResource; 2]> =
                smallvec::smallvec![resource(source_view)];
            if stencil {
                resources.push(resource(view(
                    Arc::clone(&source_image),
                    vk::ImageAspectFlags::STENCIL,
                    layer,
                )?));
            }
            let record = helpers.descriptors.get(&pipeline.layout, &resources)?;
            let target = self
                .texture_cache
                .attachments
                .insert(attachment::AttachmentEntry {
                    view: view(Arc::clone(&target_image), aspects, layer)?,
                    description: NativeImageDescription {
                        extent: vk::Extent3D {
                            width: extent.width,
                            height: extent.height,
                            depth: 1,
                        },
                        array_layers: 1,
                        ..d
                    },
                    initialized: true,
                    aspects,
                    last_use: Some(self.runtime.device.current_work()),
                });
            let attachment = RenderAttachment {
                source: AttachmentSource::ResidentOnly(target),
                load: vk::AttachmentLoadOp::DONT_CARE,
                store: vk::AttachmentStoreOp::STORE,
                clear: Default::default(),
            };
            let targets = self.prepare_targets(RenderingTargets {
                area: vk::Rect2D {
                    offset: Default::default(),
                    extent,
                },
                colors: if depth {
                    Box::new([])
                } else {
                    Box::new([Some(attachment)])
                },
                depth: depth.then_some(attachment),
                stencil: stencil.then_some(attachment),
            })?;
            let graphics = self.prepare_host_graphics(
                pipeline.clone(),
                record,
                GraphicsParameters::new(state.clone(), extent),
            )?;
            let push: Vec<_> = [extent.width as f32, extent.height as f32]
                .into_iter()
                .flat_map(f32::to_le_bytes)
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
            self.runtime.device.end_rendering()?;
            self.release_attachment(target)?;
        }
        Ok(())
    }
}
