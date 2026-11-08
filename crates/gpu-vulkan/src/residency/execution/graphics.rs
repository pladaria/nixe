use super::*;
mod clear;
pub(crate) use clear::{DeferredClear, RenderPassState};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VertexBuffer {
    pub binding: u32,
    pub buffer: BufferId,
    pub offset: u64,
    pub size: u64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndexBuffer {
    pub buffer: BufferId,
    pub offset: u64,
    pub size: u64,
    pub kind: vk::IndexType,
}
/// A native runtime-generated index window. LUT storage belongs to the buffer
/// runtime; scratch reservations belong to the scheduler's current tick.
pub(crate) struct NativeIndexBuffer {
    pub(crate) buffer: vk::Buffer,
    pub(crate) offset: u64,
    pub(crate) size: u64,
    pub(crate) kind: vk::IndexType,
}
#[derive(Clone, Copy, Debug)]
pub enum Draw {
    Direct {
        vertices: u32,
        instances: u32,
        first_vertex: u32,
        first_instance: u32,
    },
    Indexed {
        indices: u32,
        instances: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    },
    Indirect {
        buffer: BufferId,
        offset: u64,
        count: u32,
        stride: u32,
        indexed: bool,
    },
}
impl Draw {
    pub(crate) fn is_empty(self) -> bool {
        match self {
            Self::Direct {
                vertices,
                instances,
                ..
            } => vertices == 0 || instances == 0,
            Self::Indexed {
                indices, instances, ..
            } => indices == 0 || instances == 0,
            Self::Indirect { count, .. } => count == 0,
        }
    }
}
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum AttachmentSource {
    Canonical { view: ImageViewId, plane: usize },
    ResidentOnly(super::super::attachment::AttachmentId),
}
#[derive(Clone, Copy)]
pub struct RenderAttachment {
    pub source: AttachmentSource,
    pub load: vk::AttachmentLoadOp,
    pub store: vk::AttachmentStoreOp,
    pub clear: vk::ClearValue,
}
impl RenderAttachment {
    pub fn color(
        view: ImageViewId,
        plane: usize,
        load: AttachmentLoad,
        store: AttachmentStore,
    ) -> Self {
        let (load, color) = match load {
            AttachmentLoad::Load => (vk::AttachmentLoadOp::LOAD, [0.; 4]),
            AttachmentLoad::Clear(color) => (vk::AttachmentLoadOp::CLEAR, color),
            AttachmentLoad::Discard => (vk::AttachmentLoadOp::DONT_CARE, [0.; 4]),
        };
        Self {
            source: AttachmentSource::Canonical { view, plane },
            load,
            store: match store {
                AttachmentStore::Store => vk::AttachmentStoreOp::STORE,
                AttachmentStore::Discard => vk::AttachmentStoreOp::DONT_CARE,
            },
            clear: vk::ClearValue {
                color: vk::ClearColorValue { float32: color },
            },
        }
    }
}
pub struct RenderingTargets {
    pub area: vk::Rect2D,
    pub colors: Box<[Option<RenderAttachment>]>,
    pub depth: Option<RenderAttachment>,
    pub stencil: Option<RenderAttachment>,
}
struct Attachment {
    description: RenderAttachment,
    plane: ImagePlane,
    view: Arc<crate::AllocatedImageView>,
    format: vk::Format,
    samples: vk::SampleCountFlags,
    size: vk::Extent3D,
}
pub struct PreparedTargets {
    binding_revision: AtomicU64,
    area: vk::Rect2D,
    colors: Box<[Option<Attachment>]>,
    depth: Option<Attachment>,
    stencil: Option<Attachment>,
    framebuffer: Arc<crate::render_pass::Framebuffer>,
    pass: crate::render_pass::Key,
    clear_values: Box<[vk::ClearValue]>,
    maps: smallvec::SmallVec<[(u64, u64); 9]>,
    images: smallvec::SmallVec<[(vk::Image, vk::ImageSubresourceRange); 9]>,
    attachment_operations: bool,
}
impl PreparedTargets {
    fn new(
        owner: Arc<crate::device::DeviceOwner>,
        binding_revision: u64,
        area: vk::Rect2D,
        colors: Box<[Option<Attachment>]>,
        depth: Option<Attachment>,
        stencil: Option<Attachment>,
        maps: smallvec::SmallVec<[(u64, u64); 9]>,
    ) -> Result<Self, VulkanError> {
        let mut pass = crate::render_pass::Key::default();
        let mut views = Vec::new();
        let mut retained = Vec::new();
        let mut clear_values = Vec::new();
        let mut extent = vk::Extent2D {
            width: u32::MAX,
            height: u32::MAX,
        };
        let mut capture = |a: &Attachment| {
            extent.width = extent.width.min(a.size.width);
            extent.height = extent.height.min(a.size.height);
            views.push(a.view.handle);
            retained.push(Arc::clone(&a.view));
            clear_values.push(a.description.clear);
        };
        for (i, a) in colors.iter().enumerate() {
            let Some(a) = a else {
                continue;
            };
            pass.colors[i] = a.format;
            pass.samples = a.samples;
            if a.description.load == vk::AttachmentLoadOp::CLEAR {
                pass.color_clear |= 1 << i;
            }
            if a.description.load == vk::AttachmentLoadOp::DONT_CARE {
                pass.color_dont_care |= 1 << i;
            }
            if a.description.store == vk::AttachmentStoreOp::DONT_CARE {
                pass.color_discard |= 1 << i;
            }
            capture(a);
        }
        if let Some(a) = depth.as_ref().or(stencil.as_ref()) {
            pass.depth = a.format;
            pass.samples = a.samples;
            capture(a);
            let clear = clear_values.last_mut().unwrap();
            if let Some(d) = &depth {
                pass.depth_load = d.description.load;
                pass.depth_store = d.description.store;
                clear.depth_stencil.depth = unsafe { d.description.clear.depth_stencil.depth };
            }
            if let Some(st) = &stencil {
                pass.stencil_load = st.description.load;
                pass.stencil_store = st.description.store;
                clear.depth_stencil.stencil = unsafe { st.description.clear.depth_stencil.stencil };
            }
        }
        // Eden's framebuffer dimensions follow the requested render area,
        // clamped to the logical attachment views rather than backing images.
        extent.width = extent
            .width
            .min((area.offset.x as u32 + area.extent.width).max(1));
        extent.height = extent
            .height
            .min((area.offset.y as u32 + area.extent.height).max(1));
        // Eden Framebuffer captures native images/ranges at construction.
        // EndRenderPass consumes this span without rebuilding target metadata.
        // https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/renderer_vulkan/vk_texture_cache.cpp#L3045-L3155
        let mut images = smallvec::SmallVec::new();
        let mut attachment_operations = false;
        for a in colors
            .iter()
            .flatten()
            .chain(depth.iter())
            .chain(stencil.iter())
        {
            if a.samples != pass.samples || !Arc::ptr_eq(&a.view.image.owner, &owner) {
                return Err(VulkanError::InvalidResource(
                    "framebuffer attachments use incompatible sample counts or devices",
                ));
            }
            attachment_operations |= a.description.load != vk::AttachmentLoadOp::LOAD
                || a.description.store != vk::AttachmentStoreOp::STORE;
            let image = a.view.image.handle;
            let range =
                commands::plane_range(a.plane).aspect_mask(sync::native_image_aspects(a.format));
            if !images
                .iter()
                .any(|(old, r): &(vk::Image, vk::ImageSubresourceRange)| {
                    *old == image
                        && r.aspect_mask == range.aspect_mask
                        && r.base_mip_level == range.base_mip_level
                        && r.level_count == range.level_count
                        && r.base_array_layer == range.base_array_layer
                        && r.layer_count == range.layer_count
                })
            {
                images.push((image, range));
            }
        }
        let framebuffer =
            unsafe { crate::render_pass::Framebuffer::new(owner, pass, &views, extent, retained)? };
        Ok(Self {
            binding_revision: binding_revision.into(),
            area,
            colors,
            depth,
            stencil,
            framebuffer,
            pass,
            clear_values: clear_values.into_boxed_slice(),
            maps,
            images,
            attachment_operations,
        })
    }
    pub(crate) fn render_area(&self) -> vk::Rect2D {
        self.area
    }
    pub(crate) fn has_color(&self, slot: usize) -> bool {
        self.colors.get(slot).is_some_and(Option::is_some)
    }
    pub(crate) fn depth_stencil_aspects(&self) -> vk::ImageAspectFlags {
        let mut aspects = vk::ImageAspectFlags::empty();
        if self.depth.is_some() {
            aspects |= vk::ImageAspectFlags::DEPTH;
        }
        if self.stencil.is_some() {
            aspects |= vk::ImageAspectFlags::STENCIL;
        }
        aspects
    }
    fn same_framebuffer(&self, other: &Self) -> bool {
        self.framebuffer.handle == other.framebuffer.handle
            && self.framebuffer.render_pass == other.framebuffer.render_pass
            && self.area == other.area
    }
    fn attachments(&self) -> impl Iterator<Item = &Attachment> {
        self.colors
            .iter()
            .filter_map(Option::as_ref)
            .chain(self.depth.iter())
            .chain(self.stencil.iter())
    }
}
fn read_only_state(aspect: vk::ImageAspectFlags, state: &GraphicsState) -> bool {
    if aspect == vk::ImageAspectFlags::DEPTH {
        !state.depth_write
    } else if aspect == vk::ImageAspectFlags::STENCIL {
        !state.stencil_test
            || [state.front, state.back].iter().all(|s| {
                [s.fail, s.pass, s.depth_fail]
                    .iter()
                    .all(|op| *op == vk::StencilOp::KEEP)
            })
    } else {
        false
    }
}

/// Image selections consumed by ConfigureDraw. Buffer selections stay in
/// BufferCache; this span follows Eden's FillImageViews/CheckFeedbackLoop.
pub(crate) struct ShaderImage {
    pub(crate) view: ImageViewId,
    pub(crate) descriptor: usize,
    // The texture cache owns guest samplers; HostProgram owns utility samplers.
    // Configure captures native handles after those actual owners retain them.
    pub(crate) sampler: Option<vk::Sampler>,
}
impl Residency<'_> {
    pub(crate) fn targets_current(&self, targets: &PreparedTargets) -> bool {
        if targets.binding_revision.load(Ordering::Relaxed) == self.texture_cache.binding_revision {
            return true;
        }
        if targets
            .attachments()
            .any(|attachment| match attachment.description.source {
                AttachmentSource::Canonical { view, .. } => {
                    self.texture_cache.views.get(view).is_none_or(|current| {
                        !self.texture_cache.binding_current(current.image)
                            || !Arc::ptr_eq(&current.native, &attachment.view)
                    })
                }
                AttachmentSource::ResidentOnly(id) => self
                    .texture_cache
                    .attachments
                    .get(id)
                    .is_none_or(|current| !Arc::ptr_eq(&current.view, &attachment.view)),
            })
        {
            return false;
        }
        targets
            .binding_revision
            .store(self.texture_cache.binding_revision, Ordering::Relaxed);
        true
    }
    fn rendering_matches(&self, targets: &Arc<PreparedTargets>) -> bool {
        self.runtime.device.render_pass_matches(targets)
    }

    pub fn prepare_targets(
        &mut self,
        targets: RenderingTargets,
    ) -> Result<Arc<PreparedTargets>, VulkanError> {
        if targets.colors.len() > 8
            || targets.colors.len()
                > self.runtime.device.profile().limits.max_color_attachments as usize
        {
            return Err(VulkanError::InvalidResource(
                "rendering targets exceed supported color attachment slots",
            ));
        }
        let area = targets.area;
        let colors = targets
            .colors
            .iter()
            .map(|a| {
                a.map(|a| self.prepare_attachment(a, area, vk::ImageAspectFlags::COLOR))
                    .transpose()
            })
            .collect::<Result<Box<[_]>, _>>()?;
        let depth = targets
            .depth
            .map(|a| self.prepare_attachment(a, area, vk::ImageAspectFlags::DEPTH))
            .transpose()?;
        let stencil = targets
            .stencil
            .map(|a| self.prepare_attachment(a, area, vk::ImageAspectFlags::STENCIL))
            .transpose()?;
        self.create_framebuffer(area, colors, depth, stencil)
    }

    /// Construct cached native attachment metadata directly from the current
    /// view slots. Host RenderingTargets never enter this cache-owned path.
    /// https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/texture_cache/texture_cache.h#L569-L582
    pub(in crate::residency) fn framebuffer_from_targets(
        &mut self,
        key: crate::residency::texture_bindings::FramebufferKey,
    ) -> Result<Arc<PreparedTargets>, VulkanError> {
        let mut size = key.size;
        for view in key.views.iter().copied().flatten() {
            let extent = self.texture_cache.views[view].description.size;
            size[0] = size[0].min(extent.width);
            size[1] = size[1].min(extent.height);
        }
        let area = vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent: vk::Extent2D {
                width: size[0],
                height: size[1],
            },
        };
        let color_count = key.views[..8]
            .iter()
            .rposition(Option::is_some)
            .map_or(0, |i| i + 1);
        if color_count > self.runtime.device.profile().limits.max_color_attachments as usize {
            return Err(VulkanError::InvalidResource(
                "framebuffer exceeds native color attachment slots",
            ));
        }
        let mut attachment = |view, aspect| {
            let plane = self.image_view_plane(view, aspect)?;
            self.prepare_attachment(
                RenderAttachment {
                    source: AttachmentSource::Canonical { view, plane },
                    load: vk::AttachmentLoadOp::LOAD,
                    store: vk::AttachmentStoreOp::STORE,
                    clear: vk::ClearValue::default(),
                },
                area,
                aspect,
            )
        };
        let colors = key.views[..color_count]
            .iter()
            .map(|&view| {
                view.map(|view| attachment(view, vk::ImageAspectFlags::COLOR))
                    .transpose()
            })
            .collect::<Result<Box<[_]>, VulkanError>>()?;
        let mut depth = None;
        let mut stencil = None;
        if let Some(view) = key.views[8] {
            let aspects = self.texture_cache.views[view].description.aspects;
            if aspects.contains(vk::ImageAspectFlags::DEPTH) {
                let plane = self.image_view_plane(view, vk::ImageAspectFlags::DEPTH)?;
                depth = Some(self.prepare_attachment(
                    RenderAttachment {
                        source: AttachmentSource::Canonical { view, plane },
                        load: vk::AttachmentLoadOp::LOAD,
                        store: vk::AttachmentStoreOp::STORE,
                        clear: vk::ClearValue::default(),
                    },
                    area,
                    vk::ImageAspectFlags::DEPTH,
                )?);
            }
            if aspects.contains(vk::ImageAspectFlags::STENCIL) {
                let plane = self.image_view_plane(view, vk::ImageAspectFlags::STENCIL)?;
                stencil = Some(self.prepare_attachment(
                    RenderAttachment {
                        source: AttachmentSource::Canonical { view, plane },
                        load: vk::AttachmentLoadOp::LOAD,
                        store: vk::AttachmentStoreOp::STORE,
                        clear: vk::ClearValue::default(),
                    },
                    area,
                    vk::ImageAspectFlags::STENCIL,
                )?);
            }
        }
        self.create_framebuffer(area, colors, depth, stencil)
    }

    fn create_framebuffer(
        &mut self,
        area: vk::Rect2D,
        colors: Box<[Option<Attachment>]>,
        depth: Option<Attachment>,
        stencil: Option<Attachment>,
    ) -> Result<Arc<PreparedTargets>, VulkanError> {
        if let (Some(d), Some(s)) = (&depth, &stencil)
            && d.view.handle != s.view.handle
        {
            return Err(VulkanError::InvalidResource(
                "depth and stencil attachments must use the same native view",
            ));
        }
        let mut maps = smallvec::SmallVec::new();
        for a in colors
            .iter()
            .flatten()
            .chain(depth.iter())
            .chain(stencil.iter())
        {
            if let AttachmentSource::Canonical { view, .. } = a.description.source {
                maps.extend_from_slice(
                    &self
                        .texture_cache
                        .get(self.texture_cache.views[view].image)
                        .maps,
                );
            }
        }
        Ok(Arc::new(PreparedTargets::new(
            Arc::clone(&self.runtime.device.owner),
            self.texture_cache.binding_revision,
            area,
            colors,
            depth,
            stencil,
            maps,
        )?))
    }

    fn prepare_attachment(
        &mut self,
        a: RenderAttachment,
        area: vk::Rect2D,
        aspect: vk::ImageAspectFlags,
    ) -> Result<Attachment, VulkanError> {
        let (description, p, view, size) = match a.source {
            AttachmentSource::Canonical { view, plane } => {
                let image = self.texture_cache.views[view].image;
                let p = self.checked_image_rect(image, plane, area)?;
                let mut description = self.texture_cache.get(image).description;
                let d = self.image_view_description(view)?;
                if (area.offset.x as u32).saturating_add(area.extent.width) > d.size.width
                    || (area.offset.y as u32).saturating_add(area.extent.height) > d.size.height
                {
                    return Err(VulkanError::InvalidResource(
                        "render area exceeds the logical image view",
                    ));
                }
                if d.base_mip != p.mip_level
                    || d.base_layer != p.array_layer + p.depth_slice
                    || !d.aspects.contains(aspect)
                {
                    return Err(VulkanError::InvalidResource(
                        "attachment view does not select its physical plane",
                    ));
                }
                description.format = d.format;
                (description, p, self.image_view_lease(view)?, d.size)
            }
            AttachmentSource::ResidentOnly(id) => {
                let entry = self.attachment(id)?;
                let d = entry.description;
                let (block_extent, bytes_per_block) = super::super::format::copy_block(
                    d.format, aspect,
                )
                .ok_or(VulkanError::InvalidResource(
                    "resident-only attachment format/aspect mismatch",
                ))?;
                if area.offset.x < 0
                    || area.offset.y < 0
                    || area.extent.width == 0
                    || area.extent.height == 0
                    || (area.offset.x as u32)
                        .checked_add(area.extent.width)
                        .is_none_or(|v| v > d.extent.width)
                    || (area.offset.y as u32)
                        .checked_add(area.extent.height)
                        .is_none_or(|v| v > d.extent.height)
                {
                    return Err(VulkanError::InvalidResource(
                        "resident-only render area exceeds the image",
                    ));
                }
                let p = ImagePlane {
                    depth_slice: 0,
                    offset: 0,
                    row_pitch: 0,
                    mip_level: 0,
                    array_layer: 0,
                    aspect,
                    extent: d.extent,
                    block_extent,
                    bytes_per_block,
                };
                (d, p, Arc::clone(&entry.view), d.extent)
            }
        };
        if p.aspect != aspect || p.extent.depth != 1 {
            return Err(VulkanError::InvalidResource(
                "attachment plane has the wrong aspect or depth",
            ));
        }
        let usage = if aspect == vk::ImageAspectFlags::COLOR {
            vk::ImageUsageFlags::COLOR_ATTACHMENT
        } else {
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
        };
        if !description.usage.contains(usage) {
            return Err(VulkanError::InvalidResource("image lacks attachment usage"));
        }
        if !matches!(
            a.load,
            vk::AttachmentLoadOp::LOAD
                | vk::AttachmentLoadOp::CLEAR
                | vk::AttachmentLoadOp::DONT_CARE
        ) || !matches!(
            a.store,
            vk::AttachmentStoreOp::STORE | vk::AttachmentStoreOp::DONT_CARE
        ) {
            return Err(VulkanError::InvalidResource(
                "attachment load/store operation is not enabled",
            ));
        }
        if aspect == vk::ImageAspectFlags::DEPTH && a.load == vk::AttachmentLoadOp::CLEAR {
            let value = unsafe { a.clear.depth_stencil.depth };
            if !value.is_finite() || !(0. ..=1.).contains(&value) {
                return Err(VulkanError::InvalidResource(
                    "attachment depth clear requires a finite normalized value",
                ));
            }
        }
        Ok(Attachment {
            description: a,
            plane: p,
            view,
            format: description.format,
            samples: description.samples,
            size,
        })
    }

    fn vertex_binding(&self, slot: u32) -> Result<VertexBuffer, VulkanError> {
        self.runtime
            .buffer_bindings
            .channel
            .vertex_buffers
            .get(slot as usize)
            .copied()
            .flatten()
            .ok_or(VulkanError::InvalidResource(
                "shader consumes a null vertex buffer",
            ))
    }
    fn validate_framebuffer(
        &self,
        state: &GraphicsState,
        targets: &PreparedTargets,
    ) -> Result<(), VulkanError> {
        if !self.targets_current(targets) {
            return Err(VulkanError::InvalidResource(
                "framebuffer selects a retired image",
            ));
        }
        if state.colors.len() != targets.colors.len()
            || state
                .colors
                .iter()
                .zip(&targets.colors)
                .any(|((format, _), a)| {
                    *format != a.as_ref().map_or(vk::Format::UNDEFINED, |a| a.format)
                })
            || state.depth_format
                != targets
                    .depth
                    .as_ref()
                    .map_or(vk::Format::UNDEFINED, |a| a.format)
            || state.stencil_format
                != targets
                    .stencil
                    .as_ref()
                    .map_or(vk::Format::UNDEFINED, |a| a.format)
            || targets.attachments().any(|a| {
                a.samples != state.samples
                    || !Arc::ptr_eq(&a.view.image.owner, &self.runtime.device.owner)
            })
        {
            return Err(VulkanError::InvalidResource(
                "rendering attachments differ from the compiled formats/sample count/device",
            ));
        }
        Ok(())
    }
    fn draw_feedback(
        &self,
        id: ImageId,
        range: vk::ImageSubresourceRange,
        scope: AccessScope,
        targets: &PreparedTargets,
        read_only: &impl Fn(vk::ImageAspectFlags) -> Result<bool, VulkanError>,
    ) -> Result<vk::ImageAspectFlags, VulkanError> {
        let mut feedback_aspects = vk::ImageAspectFlags::empty();
        {
            let overlaps = |a: &Attachment| {
                matches!(a.description.source,AttachmentSource::Canonical {view,..} if self.texture_cache.views[view].image==id)
                    && (range.base_mip_level..range.base_mip_level + range.level_count)
                        .contains(&a.plane.mip_level)
                    && (range.base_array_layer..range.base_array_layer + range.layer_count)
                        .contains(&a.plane.array_layer)
            };
            for a in targets.attachments().filter(|a| overlaps(a)) {
                // Load/store operations can touch adjacent depth/stencil
                // aspects too. Both must be read-only before sharing reads.
                if scope.writes() {
                    return Err(VulkanError::InvalidResource(
                        "shader storage writes cannot alias a rendering attachment",
                    ));
                }
                let preserve = a.description.load == vk::AttachmentLoadOp::LOAD
                    && a.description.store == vk::AttachmentStoreOp::STORE;
                if !(preserve && read_only(a.plane.aspect)?) {
                    if !self.runtime.device.profile().attachment_feedback_loop
                        || !self
                            .texture_cache
                            .get(id)
                            .description
                            .usage
                            .contains(vk::ImageUsageFlags::ATTACHMENT_FEEDBACK_LOOP_EXT)
                    {
                        return Err(VulkanError::InvalidResource(
                            "attachment feedback requires Vulkan feedback-loop support",
                        ));
                    }
                    feedback_aspects |= range.aspect_mask;
                }
            }
        }
        Ok(feedback_aspects)
    }
    pub(crate) fn retain_target_resources(&mut self, targets: &PreparedTargets) {
        self.runtime
            .constructing
            .extend(
                targets
                    .attachments()
                    .filter_map(|a| match a.description.source {
                        AttachmentSource::Canonical { view, .. } => {
                            Some(Resource::Image(self.texture_cache.views[view].image))
                        }
                        AttachmentSource::ResidentOnly(_) => None,
                    }),
            );
    }
    /// Consume immediately, synchronizing contents independently of cached bindings.
    pub fn draw(
        &mut self,
        graphics: &HostGraphics,
        targets: &Arc<PreparedTargets>,
        draw: Draw,
        push: &[u8],
    ) -> Result<WorkTicket, VulkanError> {
        // Host utilities accept independently constructed pipeline/framebuffer
        // captures. Guest draws derive both from their owning caches directly.
        self.validate_framebuffer(
            graphics
                .program
                .pipeline
                .graphics
                .as_ref()
                .expect("graphics pipeline"),
            targets,
        )?;
        let retained = self.runtime.constructing.len();
        self.retain_host_program(&graphics.program);
        self.retain_target_resources(targets);
        for binding in graphics.parameters.state.vertex_bindings.iter() {
            let vertex = self.vertex_binding(binding.binding)?;
            self.runtime
                .constructing
                .push(Resource::Buffer(vertex.buffer));
        }
        if matches!(
            draw,
            Draw::Indexed { .. } | Draw::Indirect { indexed: true, .. }
        ) && let Some(index) = self.runtime.buffer_bindings.channel.index_buffer
        {
            self.runtime
                .constructing
                .push(Resource::Buffer(index.buffer));
        }
        self.runtime
            .constructing
            .extend(std::iter::once(&draw).filter_map(|draw| match draw {
                Draw::Indirect { buffer, .. } => Some(Resource::Buffer(*buffer)),
                _ => None,
            }));
        let result = (|| {
            if !draw.is_empty() {
                let enabled = graphics
                    .parameters
                    .state
                    .vertex_bindings
                    .iter()
                    .fold(0u32, |mask, binding| mask | (1 << binding.binding));
                self.bind_host_geometry_buffers(None, enabled, draw, None, |binding| {
                    graphics
                        .parameters
                        .state
                        .vertex_bindings
                        .iter()
                        .find(|v| v.binding == binding)
                        .expect("enabled host vertex input")
                        .stride
                })?;
            }
            let mut descriptors = if draw.is_empty() {
                DescriptorQueue {
                    record: None,
                    dynamic_offsets: None,
                    update: None,
                }
            } else {
                let descriptors = self.stream_graphics(&graphics.program)?;
                self.host_program_access(&graphics.program)?;
                descriptors
            };
            if !draw.is_empty() && descriptors.record.is_none() && descriptors.update.is_none() {
                descriptors.record = self.host_descriptor_record(&graphics.program)?;
            }
            if descriptors.dynamic_offsets.is_none() {
                descriptors.dynamic_offsets = Some(graphics.program.dynamic_offsets.clone());
            }
            let mut images = smallvec::SmallVec::<[ShaderImage; 16]>::new();
            for &descriptor in &graphics.program.pipeline.image_bindings {
                if let Some(binding) = graphics.program.bindings.get(descriptor) {
                    match binding {
                        HostBinding::Image(view) => images.push(ShaderImage {
                            view: view.id,
                            descriptor,
                            sampler: None,
                        }),
                        HostBinding::CombinedImageSampler { image, sampler } => {
                            images.push(ShaderImage {
                                view: image.id,
                                descriptor,
                                sampler: Some(sampler.handle),
                            })
                        }
                        _ => unreachable!("compiled shader image slot"),
                    }
                }
            }
            let result = self.record_draw_inner(
                &graphics.program.pipeline,
                targets,
                &images,
                descriptors,
                draw,
                push,
                |aspect| Ok(read_only_state(aspect, &graphics.parameters.state)),
                |residency| {
                    residency.graphics_state(&graphics.program.pipeline, &graphics.parameters)
                },
            )?;
            if !draw.is_empty() {
                self.host_program_written(&graphics.program)?;
            }
            Ok(result)
        })();
        self.runtime.constructing.truncate(retained);
        result
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_draw_inner(
        &mut self,
        pipeline: &NativePipeline,
        targets: &Arc<PreparedTargets>,
        images: &[ShaderImage],
        descriptors: DescriptorQueue,
        draw: Draw,
        push: &[u8],
        read_only: impl Fn(vk::ImageAspectFlags) -> Result<bool, VulkanError>,
        record_state: impl FnOnce(&mut Self) -> Result<(), VulkanError>,
    ) -> Result<WorkTicket, VulkanError> {
        // Configure consumes the current image IDs, as Eden CheckFeedbackLoop
        // does; there is no second draw-owned set of descriptor/geometry slots.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/vk_graphics_pipeline.cpp
        let mut feedback_aspects = vk::ImageAspectFlags::empty();
        for image in images {
            let range = self.texture_cache.views[image.view].native.range;
            let scope = pipeline.resources[image.descriptor];
            let aspects = self.draw_feedback(
                self.texture_cache.views[image.view].image,
                range,
                scope,
                targets,
                &read_only,
            )?;
            feedback_aspects |= aspects;
        }
        // Scheduler::RequestRenderpass continues the current framebuffer even
        // when a shader writes storage resources. Guest memory barriers and
        // operations outside a pass own those dependencies, not shader metadata.
        // Explicit host attachment loads/stores still execute once per call.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/vk_scheduler.cpp#L204-L223
        let attachment_operations = targets.attachment_operations;
        if attachment_operations
            || !feedback_aspects.is_empty()
            || (!self.rendering_matches(targets)
                && !self.runtime.device.deferred_clear_matches(targets))
        {
            self.runtime.device.end_rendering()?;
        }
        let empty = draw.is_empty();
        let DescriptorQueue {
            record,
            dynamic_offsets,
            update,
        } = descriptors;
        let indirect = if !empty {
            if let Draw::Indirect {
                buffer,
                offset,
                count,
                stride,
                indexed,
            } = draw
            {
                let element = if indexed { 20 } else { 16 };
                if count > self.runtime.device.profile().limits.max_draw_indirect_count
                    || count > 1 && self.runtime.device.profile().core.multi_draw_indirect == 0
                    || count > 1 && (stride < element || !stride.is_multiple_of(4))
                {
                    return Err(VulkanError::InvalidResource(
                        "indirect draw count/stride exceeds enabled native multiDrawIndirect contract",
                    ));
                }
                // Indirect firstInstance is a guest producer's contract: native support
                // is required rather than reading back and rewriting its commands.
                if self
                    .runtime
                    .device
                    .profile()
                    .core
                    .draw_indirect_first_instance
                    == 0
                {
                    return Err(VulkanError::InvalidResource(
                        "indirect draws require drawIndirectFirstInstance",
                    ));
                }
                Some(self.indirect_buffer(
                    buffer,
                    offset,
                    u64::from(count - 1) * u64::from(stride) + u64::from(element),
                )?)
            } else {
                None
            }
        } else {
            None
        };
        // UpdateRenderTargets prepares each color image and the combined
        // depth/stencil image once. Draw coverage never invalidates an image.
        // https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/texture_cache/texture_cache.h#L457-L490
        for a in targets
            .colors
            .iter()
            .filter_map(Option::as_ref)
            .chain(targets.depth.as_ref().or(targets.stencil.as_ref()))
        {
            if let AttachmentSource::Canonical { view, .. } = a.description.source {
                let image = self.texture_cache.views[view].image;
                let clear_aspects = if a.plane.aspect == vk::ImageAspectFlags::COLOR {
                    if a.description.load == vk::AttachmentLoadOp::CLEAR {
                        vk::ImageAspectFlags::COLOR
                    } else {
                        vk::ImageAspectFlags::empty()
                    }
                } else {
                    targets
                        .depth
                        .iter()
                        .chain(targets.stencil.iter())
                        .filter(|a| a.description.load == vk::AttachmentLoadOp::CLEAR)
                        .fold(vk::ImageAspectFlags::empty(), |aspects, a| {
                            aspects | a.plane.aspect
                        })
                };
                let invalidate = !clear_aspects.is_empty()
                    && self.is_full_image_clear(image, a.plane, targets.area, clear_aspects);
                self.prepare_image(image, true, invalidate)?;
            }
        }
        // Framebuffer operations in one rendering instance already execute in
        // rasterization order. Only unchanged native attachments can continue;
        // imports and real native dependencies terminate the scope.
        // https://docs.vulkan.org/spec/latest/chapters/primsrast.html#primsrast-order
        if !self.runtime.barriers.is_empty() {
            self.runtime.device.end_rendering()?;
        }
        let continuing = self.rendering_matches(targets);
        if !continuing && !self.runtime.device.deferred_clear_matches(targets) {
            for a in targets.attachments() {
                match a.description.source {
                    AttachmentSource::Canonical { view, .. } => {
                        let image = self.texture_cache.views[view].image;
                        self.use_image(image)?;
                    }
                    AttachmentSource::ResidentOnly(id) => {
                        self.use_attachment(id)?;
                    }
                }
            }
            self.runtime.barriers.emit(&mut self.runtime.device)?;
        }
        if empty {
        } else {
            self.runtime.device.configure_pipeline(
                pipeline,
                push,
                record.as_ref(),
                dynamic_offsets.as_deref().unwrap_or(&[]),
                update,
            )?;
            self.set_feedback_loop(feedback_aspects);
        }
        // Eden updates dynamic state before recording the draw, including an
        // empty draw. Consuming a channel's flags must update native state even
        // when this draw has no vertices, so the next draw can reuse it.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/vk_rasterizer.cpp
        record_state(self)?;
        if let Some(offsets) = dynamic_offsets {
            self.runtime.descriptor_offsets = offsets;
        }
        self.runtime.device.request_render_pass(targets)?;
        {
            let device = &mut self.runtime.device;
            {
                let raw = &mut device.dispatch;
                unsafe {
                    if !empty {
                        device.submission.statistics.draw_calls += 1;
                        {
                            match draw {
                                Draw::Direct {
                                    vertices,
                                    instances,
                                    first_vertex,
                                    first_instance,
                                } => {
                                    raw.cmd_draw(vertices, instances, first_vertex, first_instance)
                                }
                                Draw::Indexed {
                                    indices,
                                    instances,
                                    first_index,
                                    vertex_offset,
                                    first_instance,
                                } => raw.cmd_draw_indexed(
                                    indices,
                                    instances,
                                    first_index,
                                    vertex_offset,
                                    first_instance,
                                ),
                                Draw::Indirect {
                                    offset,
                                    count,
                                    stride,
                                    indexed,
                                    ..
                                } => {
                                    if indexed {
                                        raw.cmd_draw_indexed_indirect(
                                            indirect.unwrap(),
                                            offset,
                                            count,
                                            stride,
                                        );
                                    } else {
                                        raw.cmd_draw_indirect(
                                            indirect.unwrap(),
                                            offset,
                                            count,
                                            stride,
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        };
        if attachment_operations {
            self.runtime.device.end_rendering()?;
        }
        Ok(self.runtime.device.current_work())
    }
}
