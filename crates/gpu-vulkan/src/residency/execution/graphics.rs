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
#[derive(Clone)]
struct Attachment {
    description: RenderAttachment,
    plane: ImagePlane,
    view: Arc<crate::AllocatedImageView>,
    format: vk::Format,
    samples: vk::SampleCountFlags,
}
#[derive(Clone)]
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
            && self.native_depth.map(|a| a.store_op) == other.native_depth.map(|a| a.store_op)
            && self.native_stencil.map(|a| a.store_op) == other.native_stencil.map(|a| a.store_op)
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
fn read_only_attachment(a: &Attachment, state: &GraphicsState) -> bool {
    let preserve = a.description.load == vk::AttachmentLoadOp::LOAD
        && a.description.store == vk::AttachmentStoreOp::STORE;
    preserve
        && if a.plane.aspect == vk::ImageAspectFlags::DEPTH {
            !state.depth_write
        } else if a.plane.aspect == vk::ImageAspectFlags::STENCIL {
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
enum InlineIndex {
    Stream(Arc<[u8]>),
    // A payload larger than one ring arena needs one contiguous binding. Give
    // it dedicated mapped storage rather than repeatedly exhausting the ring.
    Buffer(Arc<AllocatedBuffer>),
}
impl InlineIndex {
    fn size(&self) -> u64 {
        match self {
            Self::Stream(bytes) => bytes.len() as u64,
            Self::Buffer(buffer) => buffer.size(),
        }
    }
}
pub struct PreparedDraw {
    batchable: bool,
    graphics: PreparedGraphics,
    targets: Arc<PreparedTargets>,
    vertices: Box<[VertexBuffer]>,
    _vertex_leases: Box<[Arc<AllocatedBuffer>]>,
    index: Option<(IndexBuffer, Arc<AllocatedBuffer>)>,
    inline_index: Option<InlineIndex>,
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
    /// Consume a draw run's CPU inputs at one shared capture boundary. The
    /// returned versions remain immutable; later CPU writes belong to a later
    /// read boundary. Intervening GPU producers still invalidate captured data.
    /// Fixed-arena admission bounds both the copies and their retained leases.
    pub(crate) fn capture_draw_inputs<'a>(
        &mut self,
        draws: impl Iterator<Item = &'a PreparedDraw>,
    ) -> Result<buffer::DrawInputBatch, VulkanError> {
        let limits = self.device.profile().limits;
        let mut seen = rustc_hash::FxHashMap::<(BufferId, u64, u64), usize>::default();
        let mut requests: Vec<Option<buffer::BufferStreamRequest>> = Vec::new();
        let mut add = |request: buffer::BufferStreamRequest| {
            use std::collections::hash_map::Entry;
            match seen.entry((request.id, request.offset, request.size)) {
                Entry::Occupied(index) => {
                    let old = requests[*index.get()].as_mut().unwrap();
                    old.alignment = old.alignment.max(request.alignment);
                }
                Entry::Vacant(entry) => {
                    entry.insert(requests.len());
                    requests.push(Some(request));
                }
            }
        };
        for draw in draws.filter(|draw| draw.can_group()) {
            for usage in &draw.graphics.program.uses {
                if let Use::Buffer {
                    id,
                    offset,
                    size,
                    scope,
                } = *usage
                    && !scope.writes()
                {
                    add(buffer::BufferStreamRequest {
                        id,
                        offset,
                        size,
                        alignment: scope.buffer_alignment(&limits),
                    });
                }
            }
            for vertex in &draw.vertices {
                add(buffer::BufferStreamRequest {
                    id: vertex.buffer,
                    offset: vertex.offset,
                    size: vertex.size,
                    alignment: 4,
                });
            }
        }
        self.capture_buffer_inputs(&requests)
    }
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

    pub(crate) fn prepare_inline_index_draw(
        &mut self,
        graphics: PreparedGraphics,
        targets: Arc<PreparedTargets>,
        vertices: Box<[VertexBuffer]>,
        bytes: Arc<[u8]>,
    ) -> Result<PreparedDraw, VulkanError> {
        if bytes.is_empty() || !bytes.len().is_multiple_of(4) {
            return Err(VulkanError::InvalidResource(
                "inline UINT32 indices require a nonempty aligned payload",
            ));
        }
        let mut draw = self.prepare_draw(graphics, targets, vertices, None)?;
        draw.inline_index = Some(
            if bytes.len() as u64 <= self.device.submission.configuration.arena_bytes {
                InlineIndex::Stream(bytes)
            } else {
                let buffer = crate::memory::allocate_buffer(
                    &self.device.owner,
                    bytes.len() as u64,
                    vk::BufferUsageFlags::INDEX_BUFFER,
                    crate::MemoryLocation::CpuToGpu,
                    crate::AllocationMode::Managed,
                    "nixe large inline index stream",
                )?;
                unsafe { buffer.write_range(0, &bytes) }?;
                self.statistics.cpu_streamed_bytes += bytes.len() as u64;
                InlineIndex::Buffer(Arc::new(buffer))
            },
        );
        Ok(draw)
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
        // Read-only depth/stencil can also be sampled. STORE is allowed to
        // write even when the pipeline disables depth/stencil updates; use NONE
        // to preserve the original producer and avoid a render-pass data race.
        // https://docs.vulkan.org/spec/latest/chapters/renderpass.html#renderpass-common-data-races
        let mut targets = targets;
        if targets
            .attachments()
            .any(|a| read_only_attachment(a, state))
        {
            let targets = Arc::make_mut(&mut targets);
            for (attachment, native) in [
                (&targets.depth, &mut targets.native_depth),
                (&targets.stencil, &mut targets.native_stencil),
            ] {
                if attachment
                    .as_ref()
                    .is_some_and(|a| read_only_attachment(a, state))
                {
                    native.as_mut().unwrap().store_op = vk::AttachmentStoreOp::NONE;
                }
            }
        }
        for resource in &graphics.program.uses {
            if let Use::Image { id, plane, scope } = resource {
                for a in targets.attachments().filter(|a| {
                    matches!(a.description.source,AttachmentSource::Canonical {image,..} if image==*id)
                        && a.plane.mip_level == plane.mip_level && a.plane.array_layer == plane.array_layer
                }) {
                    // Load/store operations can touch adjacent depth/stencil
                    // aspects too. Both must be read-only before sharing reads.
                    if scope.writes() || !read_only_attachment(a, state) {
                        return Err(VulkanError::InvalidResource(
                            "attachment/shader feedback requires an exact snapshot or enabled feedback-loop lowering",
                        ));
                    }
                }
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
            inline_index: None,
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
        self.record_draws(prepared, draws, push, false, None)
    }
    pub(crate) fn draws_streamed(
        &mut self,
        prepared: &PreparedDraw,
        draws: &[Draw],
        push: &[u8],
        inputs: Option<&buffer::DrawInputBatch>,
    ) -> Result<WorkTicket, VulkanError> {
        self.record_draws(prepared, draws, push, true, inputs)
    }
    fn record_draws(
        &mut self,
        prepared: &PreparedDraw,
        draws: &[Draw],
        push: &[u8],
        streaming: bool,
        inputs: Option<&buffer::DrawInputBatch>,
    ) -> Result<WorkTicket, VulkanError> {
        loop {
            let result = self.record_draws_inner(prepared, draws, push, streaming, inputs);
            self.device.submission.release_stream_uploads();
            if matches!(result, Err(VulkanError::DescriptorPressure)) {
                // Descriptor allocation precedes draw emission. Submit the
                // full slot and retry this draw in a fresh bounded slot; no
                // guest draw or side effect has been emitted by the failed try.
                self.flush()?;
                continue;
            }
            return result;
        }
    }
    fn record_draws_inner(
        &mut self,
        prepared: &PreparedDraw,
        draws: &[Draw],
        push: &[u8],
        streaming: bool,
        inputs: Option<&buffer::DrawInputBatch>,
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
                    let (element, size) = match (&prepared.index, &prepared.inline_index) {
                        (Some((i, _)), None) => (
                            match i.kind {
                                vk::IndexType::UINT16 => 2,
                                vk::IndexType::UINT8_EXT => 1,
                                _ => 4,
                            },
                            i.size,
                        ),
                        (None, Some(index)) => (4, index.size()),
                        _ => {
                            return Err(VulkanError::InvalidResource(
                                "indexed draw has no unique index stream",
                            ));
                        }
                    };
                    if (u64::from(first_index) + u64::from(indices)) * element > size {
                        return Err(VulkanError::InvalidResource(
                            "indexed draw exceeds bound index window",
                        ));
                    }
                }
                _ => {}
            }
        }
        // Inline command bytes use the command slot's existing INDEX_BUFFER
        // upload arena. No canonical-memory import, resource object or readback.
        // Slot retirement keeps the mapped bytes alive until GPU completion.
        let inline_binding = if empty {
            None
        } else {
            match &prepared.inline_index {
                Some(InlineIndex::Stream(bytes)) => {
                    // Each stream fits a fresh base arena, so bounded pressure can
                    // flush/wait for a real reusable slot and make forward progress.
                    let (slot, slice) = self.reserve_upload(bytes, 4)?;
                    self.device.submission.retain_stream_upload(slot);
                    let buffer = self.device.submission.slots[slot].upload_lease(slice);
                    self.statistics.cpu_streamed_bytes += bytes.len() as u64;
                    Some((buffer, slice.offset, vk::IndexType::UINT32))
                }
                Some(InlineIndex::Buffer(buffer)) => {
                    Some((Arc::clone(buffer), 0, vk::IndexType::UINT32))
                }
                None => None,
            }
        };
        let streaming = streaming && prepared.can_group() && !empty;
        let streams = if streaming {
            self.stream_graphics(&graphics.program, &prepared.vertices, inputs)?
        } else {
            GraphicsStreams {
                dynamic_offsets: None,
                record: None,
                program_mask: Vec::new(),
                vertices: vec![None; prepared.vertices.len()],
            }
        };
        let record = streams.record;
        let mut vertex_bindings = Vec::with_capacity(prepared.vertices.len());
        let indirect = if !empty {
            self.program_access_except(&graphics.program, &streams.program_mask)?;
            for (vertex, stream) in prepared.vertices.iter().zip(streams.vertices) {
                if let Some((buffer, offset)) = stream {
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
                if indexed && prepared.index.is_none() && prepared.inline_index.is_none() {
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
        let mut color_loads = None::<Vec<vk::RenderingAttachmentInfo<'static>>>;
        for a in prepared.targets.attachments() {
            match a.description.source {
                AttachmentSource::Canonical { image, .. } => {
                    if a.description.load == vk::AttachmentLoadOp::LOAD {
                        let replacement = if a.plane.aspect == vk::ImageAspectFlags::COLOR
                            && !empty
                            && !self.image(image)?.discarded.is_empty()
                        {
                            self.ensure_image(image)?;
                            prepared
                                .targets
                                .colors
                                .iter()
                                .position(|color| {
                                    color.as_ref().is_some_and(|color| std::ptr::eq(color, a))
                                })
                                .filter(|&index| {
                                    let state = &graphics.parameters.state;
                                    let blend = &state.colors[index].1;
                                    !state.logic_enabled
                                        && !blend.enabled
                                        && blend.write_mask
                                            == (vk::ColorComponentFlags::R
                                                | vk::ColorComponentFlags::G
                                                | vk::ColorComponentFlags::B
                                                | vk::ColorComponentFlags::A)
                                })
                        } else {
                            None
                        };
                        // LOAD preserves existing texels; it does not mean the
                        // guest reads the destination. Replacement draws can
                        // write an incompletely initialized target. Keep LOAD
                        // when it preserves defined texels; discard only an
                        // entirely undefined render area.
                        // https://docs.vulkan.org/refpages/latest/refpages/source/VkAttachmentLoadOp.html
                        let discard = replacement.filter(|_| {
                            self.image(image).unwrap().discarded.iter().any(
                                |&(plane, undefined)| {
                                    self.image(image).unwrap().planes[plane] == a.plane
                                        && transfers::contains_rect(
                                            undefined,
                                            prepared.targets.area,
                                        )
                                },
                            )
                        });
                        if let Some(index) = discard {
                            color_loads
                                .get_or_insert_with(|| prepared.targets.native_colors.to_vec())
                                [index]
                                .load_op = vk::AttachmentLoadOp::DONT_CARE;
                        } else if replacement.is_none() {
                            self.ensure_image_initialized(image, a.plane, prepared.targets.area)?;
                        }
                    } else {
                        self.ensure_image(image)?;
                    }
                    if !read_only_attachment(a, &graphics.parameters.state) {
                        self.prepare_image_rect(image, a.plane, prepared.targets.area)?;
                    }
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
        let record = if empty || record.is_some() {
            record
        } else {
            self.program_record(&graphics.program)?
        };
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
                let access = if read_only_attachment(a, &graphics.parameters.state) {
                    vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
                } else {
                    access
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
            let slot = self.device.begin_recording()?;
            self.device.submission.slots[slot].command
        } else {
            let command = self.program_bind_record(
                &graphics.program,
                push,
                record.as_ref(),
                streams
                    .dynamic_offsets
                    .as_deref()
                    .unwrap_or(&graphics.program.dynamic_offsets),
            )?;
            self.graphics_state(graphics)?;
            command
        };
        if let Some(offsets) = streams.dynamic_offsets {
            self.descriptor_offsets = offsets;
        }
        let index = self.device.submission.recording.unwrap();
        let slot = &mut self.device.submission.slots[index];
        if let Some(InlineIndex::Buffer(buffer)) = &prepared.inline_index {
            // The frontend/prepared cache may drop this draw before submission.
            // Its actual command slot retains the dedicated buffer through retirement.
            slot.retired_buffers.push(Arc::clone(buffer));
        }
        if slot
            .target_handles
            .insert(Arc::as_ptr(&prepared.targets) as usize)
        {
            slot.native_targets.push(Arc::clone(&prepared.targets));
        }
        // Vertex/index leases can survive removal from the residency index.
        // Residency last_use also prevents release before this reservation retires.
        let raw = &self.device.dispatch;
        unsafe {
            if !empty {
                for (v, &value) in prepared.vertices.iter().zip(&vertex_bindings) {
                    if slot.vertex_bindings.get(&v.binding) != Some(&value) {
                        self.device.submission.statistics.vertex_binds += 1;
                        raw.cmd_bind_vertex_buffers(command, v.binding, &[value.0], &[value.1]);
                        slot.vertex_bindings.insert(v.binding, value);
                    }
                }
                if let Some(value) = inline_binding
                    .as_ref()
                    .map(|(buffer, offset, kind)| (buffer.handle, *offset, *kind))
                    .or_else(|| {
                        prepared
                            .index
                            .as_ref()
                            .map(|(i, buffer)| (buffer.handle, i.offset, i.kind))
                    })
                    && slot.index_binding != Some(value)
                {
                    self.device.submission.statistics.index_binds += 1;
                    raw.cmd_bind_index_buffer(command, value.0, value.1, value.2);
                    slot.index_binding = Some(value);
                }
            }
            if !continuing {
                let targets = &prepared.targets;
                let mut info = vk::RenderingInfo::default()
                    .render_area(targets.area)
                    .layer_count(1)
                    .color_attachments(color_loads.as_deref().unwrap_or(&targets.native_colors));
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
                self.device.submission.statistics.draw_calls += draws.len() as u64;
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
            if read_only_attachment(a, &graphics.parameters.state) {
                continue;
            }
            match a.description.source {
                AttachmentSource::Canonical { image, .. } => {
                    let incomplete = a.description.load == vk::AttachmentLoadOp::LOAD
                        && self
                            .image(image)?
                            .discarded
                            .iter()
                            .any(|&(index, undefined)| {
                                self.image(image).unwrap().planes[index] == a.plane
                                    && transfers::intersects(undefined, prepared.targets.area)
                            });
                    if incomplete {
                        // STORE preserves old defined contents and the draw
                        // can produce only inside its scissors. Do not publish
                        // undefined padding as newly written guest bytes.
                        let regions = preserved_draw_rects(
                            prepared.targets.area,
                            self.image(image)?
                                .discarded
                                .iter()
                                .filter_map(|&(index, rect)| {
                                    (self.image(image).unwrap().planes[index] == a.plane)
                                        .then_some(rect)
                                }),
                            &graphics.parameters.scissors,
                        );
                        for rect in regions {
                            self.finish_image_rect(image, a.plane, rect)?;
                        }
                    } else if continuing {
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

/// The cold path for a target that still has undefined texels. Ordinary draws
/// retain the single-rectangle publication and continued-rendering fast path.
fn preserved_draw_rects(
    area: vk::Rect2D,
    undefined: impl Iterator<Item = vk::Rect2D>,
    scissors: &[vk::Rect2D],
) -> Vec<vk::Rect2D> {
    let mut produced = vec![area];
    for undefined in undefined {
        let mut untouched = vec![undefined];
        for &scissor in scissors {
            untouched = untouched
                .into_iter()
                .flat_map(|rect| transfers::subtract_rect(rect, scissor))
                .collect();
        }
        for untouched in untouched {
            produced = produced
                .into_iter()
                .flat_map(|rect| transfers::subtract_rect(rect, untouched))
                .collect();
        }
    }
    produced
}
