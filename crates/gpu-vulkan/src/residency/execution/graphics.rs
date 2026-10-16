use super::*;
#[derive(Clone, Copy, Debug)]
pub struct VertexBuffer {
    pub binding: u32,
    pub buffer: BufferId,
    pub offset: u64,
    pub size: u64,
}
#[derive(Clone, Copy, Debug)]
pub struct IndexBuffer {
    pub buffer: BufferId,
    pub offset: u64,
    pub size: u64,
    pub kind: vk::IndexType,
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
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum AttachmentSource {
    Canonical { image: ImageId, plane: usize },
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
        image: ImageId,
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
            source: AttachmentSource::Canonical { image, plane },
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
}
pub struct PreparedTargets {
    area: vk::Rect2D,
    colors: Box<[Option<Attachment>]>,
    depth: Option<Attachment>,
    stencil: Option<Attachment>,
    native_colors: Box<[vk::RenderingAttachmentInfo<'static>]>,
    native_depth: Option<vk::RenderingAttachmentInfo<'static>>,
    native_stencil: Option<vk::RenderingAttachmentInfo<'static>>,
}
impl PreparedTargets {
    fn compatible(&self, other: &Self) -> bool {
        let same = |a: &Option<Attachment>, b: &Option<Attachment>| match (a, b) {
            (None, None) => true,
            (Some(a), Some(b)) => {
                a.description.source == b.description.source
                    && a.view.handle == b.view.handle
                    && a.plane == b.plane
                    && a.samples == b.samples
                    && a.format == b.format
            }
            _ => false,
        };
        self.area == other.area
            && self.colors.len() == other.colors.len()
            && self
                .colors
                .iter()
                .zip(&other.colors)
                .all(|(a, b)| same(a, b))
            && same(&self.depth, &other.depth)
            && same(&self.stencil, &other.stencil)
    }
    fn attachments(&self) -> impl Iterator<Item = &Attachment> {
        self.colors
            .iter()
            .filter_map(Option::as_ref)
            .chain(self.depth.iter())
            .chain(self.stencil.iter())
    }
}
pub struct PreparedDraw {
    batchable: bool,
    graphics: PreparedGraphics,
    targets: Arc<PreparedTargets>,
    vertices: Box<[VertexBuffer]>,
    _vertex_leases: Box<[Arc<AllocatedBuffer>]>,
    index: Option<(IndexBuffer, Arc<AllocatedBuffer>)>,
}
impl PreparedDraw {
    pub(crate) fn can_group(&self) -> bool {
        self.batchable && !self.shader_writes()
    }

    pub(crate) fn shader_writes(&self) -> bool {
        self.graphics.program.uses.iter().any(|u| match u {
            Use::Buffer { scope, .. } | Use::Image { scope, .. } => scope.writes(),
        })
    }
}
impl VulkanResidency {
    pub(crate) fn targets_current(&self, targets: &PreparedTargets) -> bool {
        targets
            .attachments()
            .all(|attachment| match attachment.description.source {
                AttachmentSource::Canonical { image, .. } => self.image(image).is_ok(),
                AttachmentSource::ResidentOnly(id) => self.attachment(id).is_ok(),
            })
    }
    fn rendering_matches(&self, targets: &Arc<PreparedTargets>) -> bool {
        self.device
            .submission
            .recording
            .and_then(|index| self.device.submission.slots[index].active_targets.as_ref())
            .is_some_and(|active| Arc::ptr_eq(active, targets) || active.compatible(targets))
    }

    pub fn prepare_targets(
        &mut self,
        targets: RenderingTargets,
    ) -> Result<Arc<PreparedTargets>, VulkanError> {
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
        if let (Some(d), Some(s)) = (&depth, &stencil)
            && d.view.handle != s.view.handle
        {
            return Err(VulkanError::InvalidResource(
                "depth and stencil attachments must use the same native view",
            ));
        }
        let native = |a: &Attachment| {
            vk::RenderingAttachmentInfo::default()
                .image_view(a.view.handle)
                .image_layout(vk::ImageLayout::GENERAL)
                .load_op(a.description.load)
                .store_op(a.description.store)
                .clear_value(a.description.clear)
        };
        let native_colors = colors
            .iter()
            .map(|a| {
                a.as_ref()
                    .map_or_else(vk::RenderingAttachmentInfo::default, native)
            })
            .collect();
        let native_depth = depth.as_ref().map(native);
        let native_stencil = stencil.as_ref().map(native);
        Ok(Arc::new(PreparedTargets {
            area,
            colors,
            depth,
            stencil,
            native_colors,
            native_depth,
            native_stencil,
        }))
    }
    fn prepare_attachment(
        &mut self,
        a: RenderAttachment,
        area: vk::Rect2D,
        aspect: vk::ImageAspectFlags,
    ) -> Result<Attachment, VulkanError> {
        let (description, p, view) = match a.source {
            AttachmentSource::Canonical { image, plane } => {
                let p = self.checked_image_rect(image, plane, area)?;
                let description = self.image(image)?.description;
                let aspects = if aspect != vk::ImageAspectFlags::COLOR
                    && super::super::format::copy_block(
                        description.format,
                        vk::ImageAspectFlags::DEPTH,
                    )
                    .is_some()
                    && super::super::format::copy_block(
                        description.format,
                        vk::ImageAspectFlags::STENCIL,
                    )
                    .is_some()
                {
                    vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
                } else {
                    aspect
                };
                let view = self.image_view(
                    image,
                    ImageViewDescription {
                        format: description.format,
                        view_type: vk::ImageViewType::TYPE_2D,
                        aspects,
                        base_mip: p.mip_level,
                        mip_count: 1,
                        base_layer: p.array_layer,
                        layer_count: 1,
                        swizzle: [vk::ComponentSwizzle::IDENTITY; 4],
                    },
                )?;
                (description, p, self.image_view_lease(view)?)
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
                    offset: 0,
                    row_pitch: 0,
                    mip_level: 0,
                    array_layer: 0,
                    aspect,
                    extent: d.extent,
                    block_extent,
                    bytes_per_block,
                };
                (d, p, Arc::clone(&entry.view))
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
        })
    }

    pub fn prepare_draw(
        &self,
        graphics: PreparedGraphics,
        targets: Arc<PreparedTargets>,
        vertices: Box<[VertexBuffer]>,
        index: Option<IndexBuffer>,
    ) -> Result<PreparedDraw, VulkanError> {
        let state = &graphics.parameters.state;
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
                a.samples != state.samples || !Arc::ptr_eq(&a.view.image.owner, &self.device.owner)
            })
        {
            return Err(VulkanError::InvalidResource(
                "rendering attachments differ from the compiled formats/sample count/device",
            ));
        }
        for resource in &graphics.program.uses {
            if let Use::Image { id, plane, .. } = resource && targets.attachments().any(|a| {
                matches!(a.description.source,AttachmentSource::Canonical {image,..} if image==*id)
                    && a.plane.mip_level == plane.mip_level
                    && a.plane.array_layer == plane.array_layer
            }) {
                return Err(VulkanError::InvalidResource(
                    "attachment/shader feedback requires an exact snapshot or enabled feedback-loop lowering",
                ));
            }
        }
        if vertices.len() != state.vertex_bindings.len()
            || state
                .vertex_bindings
                .iter()
                .any(|b| vertices.iter().filter(|v| v.binding == b.binding).count() != 1)
        {
            return Err(VulkanError::InvalidResource(
                "vertex streams do not match compiled binding slots",
            ));
        }
        let vertex_leases = vertices
            .iter()
            .map(|v| {
                self.check_buffer_range(v.buffer, v.offset, v.size)?;
                let buffer = self.buffer_lease(v.buffer)?;
                if !buffer.usage.contains(vk::BufferUsageFlags::VERTEX_BUFFER) {
                    return Err(VulkanError::InvalidResource(
                        "vertex stream lacks VERTEX_BUFFER usage",
                    ));
                }
                Ok(buffer)
            })
            .collect::<Result<_, _>>()?;
        let index = index.map(|i| {
            let size = match i.kind { vk::IndexType::UINT16 => 2, vk::IndexType::UINT32 => 4, vk::IndexType::UINT8_EXT if self.device.profile().index_uint8 => 1, _ => return Err(VulkanError::InvalidResource("native index type requires UINT16/UINT32; frontend must widen unsupported UINT8")) };
            self.check_buffer_range(i.buffer,i.offset,i.size)?;
            if !i.offset.is_multiple_of(size) || !self.buffer(i.buffer)?.usage.contains(vk::BufferUsageFlags::INDEX_BUFFER) { return Err(VulkanError::InvalidResource("index stream requires aligned INDEX_BUFFER storage")); }
            Ok((i,self.buffer_lease(i.buffer)?))
        }).transpose()?;
        let attachment_images: Vec<_> = targets
            .attachments()
            .filter_map(|a| {
                if let AttachmentSource::Canonical { image, .. } = a.description.source {
                    Some(Resource::Image(image))
                } else {
                    None
                }
            })
            .collect();
        let overlaps = |backing: &CanonicalBackingRange, offset: u64, size: u64| {
            // A buffer binding often uses a tiny window of a large allocation.
            // Its segment cursor already locates those pages directly; walking
            // the entire allocation first makes every draw scale with heap size.
            // Whole mapped images keep the compact page rejection so disjoint
            // textures never expand their per-texel transfer permutation.
            if offset == 0
                && size == backing.size()
                && !attachment_images
                    .iter()
                    .any(|&target| backing.may_share_pages(&self.contents(target).unwrap().backing))
            {
                return false;
            }
            backing
                .subrange_segments(offset, size)
                .is_ok_and(|segments| {
                    segments.into_iter().any(|(logical, segment)| {
                        let start = segment.offset() + offset.saturating_sub(logical);
                        let end = segment.offset() + segment.size().min(offset + size - logical);
                        // Filter resource identity before expanding page mappings.
                        // Previous images sharing this page are irrelevant to draw
                        // feedback and can contain thousands of packed aspect runs.
                        self.index.pages.get(&segment.page()).is_some_and(|page| {
                            attachment_images
                                .iter()
                                .flat_map(|&r| page.bindings_for(r))
                                .any(|b| {
                                    b.exact_iter()
                                        .any(|span| span.start < end && start < span.end)
                                })
                        })
                    })
                })
        };
        let batchable = targets.attachments().all(|a| {
            a.description.load == vk::AttachmentLoadOp::LOAD
                && a.description.store == vk::AttachmentStoreOp::STORE
        }) && !vertices.iter().any(|v| {
            overlaps(
                &self.buffer(v.buffer).unwrap().contents.backing,
                v.offset,
                v.size,
            )
        }) && !index.as_ref().is_some_and(|i| {
            overlaps(
                &self.buffer(i.0.buffer).unwrap().contents.backing,
                i.0.offset,
                i.0.size,
            )
        }) && !graphics.program.uses.iter().any(|u| match *u {
            Use::Buffer {
                id, offset, size, ..
            } => overlaps(&self.buffer(id).unwrap().contents.backing, offset, size),
            Use::Image { id, .. } => {
                let b = &self.image(id).unwrap().contents.backing;
                overlaps(b, 0, b.size())
            }
        });
        Ok(PreparedDraw {
            batchable,
            graphics,
            targets,
            vertices,
            _vertex_leases: vertex_leases,
            index,
        })
    }
    pub fn draw(
        &mut self,
        prepared: &PreparedDraw,
        draw: Draw,
        push: &[u8],
    ) -> Result<WorkTicket, VulkanError> {
        self.draws(prepared, std::slice::from_ref(&draw), push)
    }
    /// One compatible immutable state and several direct arguments. Shader
    /// writes are a real inter-draw hazard and require separate preparation.
    pub fn draws(
        &mut self,
        prepared: &PreparedDraw,
        draws: &[Draw],
        push: &[u8],
    ) -> Result<WorkTicket, VulkanError> {
        self.record_draws(prepared, draws, push, None)
    }
    pub(crate) fn draws_streamed(
        &mut self,
        prepared: &PreparedDraw,
        draws: &[Draw],
        push: &[u8],
        descriptors: &mut DescriptorCache,
    ) -> Result<WorkTicket, VulkanError> {
        self.record_draws(prepared, draws, push, Some(descriptors))
    }
    fn record_draws(
        &mut self,
        prepared: &PreparedDraw,
        draws: &[Draw],
        push: &[u8],
        descriptors: Option<&mut DescriptorCache>,
    ) -> Result<WorkTicket, VulkanError> {
        let result = self.record_draws_inner(prepared, draws, push, descriptors);
        self.device.submission.release_stream_uploads();
        result
    }
    fn record_draws_inner(
        &mut self,
        prepared: &PreparedDraw,
        draws: &[Draw],
        push: &[u8],
        descriptors: Option<&mut DescriptorCache>,
    ) -> Result<WorkTicket, VulkanError> {
        if draws.is_empty() {
            return Err(VulkanError::InvalidResource("empty draw group"));
        }
        if draws.len() > 1
            && (!prepared.can_group() || draws.iter().any(|d| matches!(d, Draw::Indirect { .. })))
        {
            return Err(VulkanError::InvalidResource(
                "draw group requires direct arguments and read-only shader resources",
            ));
        }
        if !prepared.can_group() || !self.rendering_matches(&prepared.targets) {
            self.device.end_rendering();
        }
        let is_empty = |d: &Draw| match *d {
            Draw::Direct {
                vertices,
                instances,
                ..
            } => vertices == 0 || instances == 0,
            Draw::Indexed {
                indices, instances, ..
            } => indices == 0 || instances == 0,
            Draw::Indirect { count, .. } => count == 0,
        };
        let draw = *draws.iter().find(|d| !is_empty(d)).unwrap_or(&draws[0]);
        let graphics = &prepared.graphics;
        let empty = match draw {
            Draw::Direct {
                vertices,
                instances,
                ..
            } => vertices == 0 || instances == 0,
            Draw::Indexed {
                indices, instances, ..
            } => indices == 0 || instances == 0,
            Draw::Indirect { count, .. } => count == 0,
        };
        for &draw in draws {
            match draw {
                Draw::Indexed {
                    indices,
                    first_index,
                    ..
                } if !is_empty(&draw) => {
                    let (i, _) = prepared.index.as_ref().ok_or(VulkanError::InvalidResource(
                        "indexed draw has no index stream",
                    ))?;
                    let element = match i.kind {
                        vk::IndexType::UINT16 => 2,
                        vk::IndexType::UINT8_EXT => 1,
                        _ => 4,
                    };
                    if (u64::from(first_index) + u64::from(indices)) * element > i.size {
                        return Err(VulkanError::InvalidResource(
                            "indexed draw exceeds bound index window",
                        ));
                    }
                }
                _ => {}
            }
        }
        let streaming = descriptors.is_some() && prepared.can_group() && !empty;
        let (record, mask) = if streaming {
            self.stream_program(&graphics.program, descriptors.unwrap())?
        } else {
            (None, Vec::new())
        };
        let mut vertex_bindings = Vec::with_capacity(prepared.vertices.len());
        let indirect = if !empty {
            self.program_access_except(&graphics.program, &mask)?;
            for vertex in &prepared.vertices {
                if streaming
                    && let Some((buffer, offset)) =
                        self.stream_buffer(vertex.buffer, vertex.offset, vertex.size, 4)?
                {
                    vertex_bindings.push((buffer.handle, offset));
                    continue;
                }
                self.ensure_buffer(vertex.buffer)?;
                let handle = self.buffer_access_range(
                    vertex.buffer,
                    vertex.offset,
                    vertex.size,
                    AccessScope::new(
                        vk::PipelineStageFlags2::VERTEX_ATTRIBUTE_INPUT,
                        vk::AccessFlags2::VERTEX_ATTRIBUTE_READ,
                    ),
                )?;
                vertex_bindings.push((handle, vertex.offset));
            }
            // Input order matches the immutable vertex lease array.
            if let Some((index, _)) = &prepared.index {
                self.ensure_buffer(index.buffer)?;
                self.buffer_access_range(
                    index.buffer,
                    index.offset,
                    index.size,
                    AccessScope::new(
                        vk::PipelineStageFlags2::INDEX_INPUT,
                        vk::AccessFlags2::INDEX_READ,
                    ),
                )?;
            }
            if let Draw::Indirect {
                buffer,
                offset,
                count,
                stride,
                indexed,
            } = draw
            {
                if indexed && prepared.index.is_none() {
                    return Err(VulkanError::InvalidResource(
                        "indexed indirect draw has no index stream",
                    ));
                }
                let element = if indexed { 20 } else { 16 };
                if count > self.device.profile().limits.max_draw_indirect_count
                    || count > 1 && self.device.profile().core.multi_draw_indirect == 0
                    || count > 1 && (stride < element || !stride.is_multiple_of(4))
                {
                    return Err(VulkanError::InvalidResource(
                        "indirect draw count/stride exceeds enabled native multiDrawIndirect contract",
                    ));
                }
                // Indirect firstInstance is a guest producer's contract: native support
                // is required rather than reading back and rewriting its commands.
                if self.device.profile().core.draw_indirect_first_instance == 0 {
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
        for a in prepared.targets.attachments() {
            match a.description.source {
                AttachmentSource::Canonical { image, .. } => {
                    if a.description.load == vk::AttachmentLoadOp::LOAD {
                        self.ensure_image_initialized(image, a.plane, prepared.targets.area)?;
                    } else {
                        self.ensure_image(image)?;
                    }
                    self.prepare_image_rect(image, a.plane, prepared.targets.area)?;
                }
                AttachmentSource::ResidentOnly(id) => {
                    if a.description.load == vk::AttachmentLoadOp::LOAD
                        && self
                            .attachment(id)?
                            .undefined
                            .iter()
                            .any(|r| transfers::intersects(*r, prepared.targets.area))
                    {
                        return Err(VulkanError::InvalidResource(
                            "attachment LOAD consumes undefined or discarded samples",
                        ));
                    }
                }
            }
        }
        // Framebuffer operations in one rendering instance already execute in
        // rasterization order. Only unchanged native attachments can continue;
        // shader writes/feedback, imports and real hazards terminate the scope.
        // https://docs.vulkan.org/spec/latest/chapters/primsrast.html#primsrast-order
        if !self.barriers.is_empty() {
            self.device.end_rendering();
        }
        let continuing = self.rendering_matches(&prepared.targets);
        if !continuing {
            for a in prepared.targets.attachments() {
                let (stages, access) = if a.plane.aspect == vk::ImageAspectFlags::COLOR {
                    (
                        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                        vk::AccessFlags2::COLOR_ATTACHMENT_READ
                            | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                    )
                } else {
                    (
                        vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                            | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                        vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
                            | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    )
                };
                let scope = AccessScope::new(stages, access);
                match a.description.source {
                    AttachmentSource::Canonical { image, .. } => {
                        self.image_access_subresources(
                            image,
                            commands::plane_range(a.plane),
                            scope,
                            vk::ImageLayout::GENERAL,
                        )?;
                    }
                    AttachmentSource::ResidentOnly(id) => {
                        self.attachment_access(id, scope, vk::ImageLayout::GENERAL)?;
                    }
                }
            }
            self.barriers.emit(&mut self.device);
        }
        let command = if empty {
            let slot = self.device.submission.begin()?;
            self.device.submission.slots[slot].command
        } else {
            let command = self.program_bind_record(
                &graphics.program,
                push,
                record.as_ref().or(graphics.program.record.as_ref()),
            )?;
            self.graphics_state(graphics)?;
            command
        };
        let index = self.device.submission.recording.unwrap();
        let slot = &mut self.device.submission.slots[index];
        if slot
            .target_handles
            .insert(Arc::as_ptr(&prepared.targets) as usize)
        {
            slot.native_targets.push(Arc::clone(&prepared.targets));
        }
        // Vertex/index leases can survive removal from the residency index.
        // Residency last_use also prevents release before this reservation retires.
        let raw = &self.device.owner.device;
        unsafe {
            if !empty {
                for (v, &value) in prepared.vertices.iter().zip(&vertex_bindings) {
                    if slot.vertex_bindings.get(&v.binding) != Some(&value) {
                        self.device.submission.statistics.vertex_binds += 1;
                        raw.cmd_bind_vertex_buffers(command, v.binding, &[value.0], &[value.1]);
                        slot.vertex_bindings.insert(v.binding, value);
                    }
                }
                if let Some((i, buffer)) = &prepared.index {
                    let value = (buffer.handle, i.offset, i.kind);
                    if slot.index_binding != Some(value) {
                        self.device.submission.statistics.index_binds += 1;
                        raw.cmd_bind_index_buffer(command, buffer.handle, i.offset, i.kind);
                        slot.index_binding = Some(value);
                    }
                }
            }
            if !continuing {
                let targets = &prepared.targets;
                let mut info = vk::RenderingInfo::default()
                    .render_area(targets.area)
                    .layer_count(1)
                    .color_attachments(&targets.native_colors);
                if let Some(depth) = &targets.native_depth {
                    info = info.depth_attachment(depth);
                }
                if let Some(stencil) = &targets.native_stencil {
                    info = info.stencil_attachment(stencil);
                }
                // No render-pass object, framebuffer cache, HAL encoder or backend segment.
                // https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdBeginRendering.html
                self.device.submission.statistics.rendering_scopes += 1;
                self.device.dispatch.begin_rendering(command, &info);
                slot.active_targets = Some(Arc::clone(&prepared.targets));
            }
            if !empty {
                for &draw in draws {
                    match draw {
                        Draw::Direct {
                            vertices,
                            instances,
                            first_vertex,
                            first_instance,
                        } => {
                            raw.cmd_draw(command, vertices, instances, first_vertex, first_instance)
                        }
                        Draw::Indexed {
                            indices,
                            instances,
                            first_index,
                            vertex_offset,
                            first_instance,
                        } => raw.cmd_draw_indexed(
                            command,
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
                                    command,
                                    indirect.unwrap(),
                                    offset,
                                    count,
                                    stride,
                                );
                            } else {
                                raw.cmd_draw_indirect(
                                    command,
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
        if !prepared.can_group() {
            self.device.end_rendering();
        }
        if !empty {
            self.program_written(&graphics.program)?;
        }
        for a in prepared.targets.attachments() {
            match a.description.source {
                AttachmentSource::Canonical { image, .. } => {
                    if continuing {
                        // The first draw already published this exact coverage
                        // at the current ticket. A continued LOAD/STORE scope
                        // permits no shader writes, attachment feedback, native
                        // imports or intervening commands that can consume or
                        // replace that authority. CPU visibility also ends and
                        // submits rendering before exposing bytes. Keep version
                        // changes per draw, without walking every target page.
                        self.advance_image_producer(image)?;
                        self.statistics.attachment_publication_hits += 1;
                    } else if !(empty && a.description.load == vk::AttachmentLoadOp::DONT_CARE) {
                        self.finish_image_rect(image, a.plane, prepared.targets.area)?;
                    }
                    if a.description.store == vk::AttachmentStoreOp::DONT_CARE
                        || empty && a.description.load == vk::AttachmentLoadOp::DONT_CARE
                    {
                        self.mark_image_discarded(image, a.plane, prepared.targets.area);
                    }
                }
                AttachmentSource::ResidentOnly(id) => {
                    if a.description.store == vk::AttachmentStoreOp::DONT_CARE
                        || empty && a.description.load == vk::AttachmentLoadOp::DONT_CARE
                    {
                        self.attachment_discarded(id, prepared.targets.area);
                    } else {
                        self.attachment_defined(id, prepared.targets.area);
                    }
                }
            }
        }
        self.device.work_ticket()
    }
}
