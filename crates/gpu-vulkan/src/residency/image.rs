//! Native image representations, compatible views and GPU-only byte realignment.
use super::recording::scratch_read_barrier;
use super::*;
impl VulkanResidency {
    /// Image reconstruction consumes complete transfer blocks. Tracking partial
    /// bytes as separate invalid intervals cannot save a transfer and explodes
    /// metadata for packed depth/stencil aliases. Producer authority stays exact.
    pub(super) fn image_invalidation_window(
        &self,
        id: ImageId,
        offset: u64,
        length: u64,
    ) -> (u64, u64) {
        let image = self.image(id).expect("indexed image");
        let planes = image.decoded.as_ref().map_or(&image.planes, |d| &d.planes);
        let plane = planes
            .iter()
            .map(|&plane| image.storage_plane(plane))
            .find(|plane| offset >= plane.offset && offset < plane.end().unwrap())
            .expect("indexed image plane");
        let block = u64::from(plane.bytes_per_block);
        let first = offset - plane.offset;
        let last = first + length - 1;
        let start = plane.offset
            + first / plane.row_pitch * plane.row_pitch
            + first % plane.row_pitch / block * block;
        let end = plane.offset
            + last / plane.row_pitch * plane.row_pitch
            + (last % plane.row_pitch + 1).div_ceil(block) * block;
        (start, end - start)
    }
    pub fn image_for(
        &mut self,
        backing: CanonicalBackingRange,
        description: NativeImageDescription,
        planes: Vec<ImagePlane>,
    ) -> Result<ImageId, VulkanError> {
        self.create_image(backing, description, planes, true)
    }
    /// Allocate an undefined native target for an actual device producer.
    /// LOAD/read consumption is checked; allocation is never a fabricated clear.
    pub fn image_for_device_write(
        &mut self,
        backing: CanonicalBackingRange,
        description: NativeImageDescription,
        planes: Vec<ImagePlane>,
    ) -> Result<ImageId, VulkanError> {
        self.create_image(backing, description, planes, false)
    }
    fn create_image(
        &mut self,
        backing: CanonicalBackingRange,
        mut description: NativeImageDescription,
        planes: Vec<ImagePlane>,
        initialize: bool,
    ) -> Result<ImageId, VulkanError> {
        description.usage |= vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
        let id = self.create_image_representation(
            backing,
            description,
            planes,
            initialize,
            None,
            [1, 1],
        )?;
        if initialize {
            self.ensure_image(id)?;
        }
        Ok(id)
    }
    /// Register canonical storage without importing bytes ahead of consumption.
    /// Full device writes validate their own coverage; an actual LOAD or shader
    /// read imports the remaining canonical contents through ensure_image.
    pub fn image_for_deferred_import(
        &mut self,
        backing: CanonicalBackingRange,
        mut description: NativeImageDescription,
        planes: Vec<ImagePlane>,
    ) -> Result<ImageId, VulkanError> {
        description.usage |= vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
        self.create_image_representation(backing, description, planes, true, None, [1, 1])
    }
    /// Device-authored samples share canonical ownership with ordinary images.
    /// Their sample grid is supplied by the frontend, independently of host layout.
    pub fn multisample_image_for(
        &mut self,
        backing: CanonicalBackingRange,
        mut description: NativeImageDescription,
        planes: Vec<ImagePlane>,
        sample_grid: [u32; 2],
    ) -> Result<ImageId, VulkanError> {
        description.usage |= vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
        // CPU sample visibility reads encoded bits through a compatible UNORM
        // view, avoiding an approximate inverse of the sampler's sRGB decode.
        if matches!(
            description.format,
            vk::Format::R8G8B8A8_SRGB | vk::Format::B8G8R8A8_SRGB
        ) {
            description.flags |= vk::ImageCreateFlags::MUTABLE_FORMAT;
        }
        self.create_image_representation(backing, description, planes, false, None, sample_grid)
    }
    pub(super) fn create_image_representation(
        &mut self,
        backing: CanonicalBackingRange,
        description: NativeImageDescription,
        planes: Vec<ImagePlane>,
        initialize: bool,
        decoded: Option<super::decoded_texture::DecodedImage>,
        sample_grid: [u32; 2],
    ) -> Result<ImageId, VulkanError> {
        if description.samples != vk::SampleCountFlags::TYPE_1 && initialize {
            return Err(VulkanError::InvalidResource(
                "multisample canonical import requires sample conversion",
            ));
        }
        if let Some(decoded) = &decoded {
            let mut encoded = description;
            encoded.format = decoded.format;
            validate_planes(&backing, encoded, &decoded.planes, [1, 1])?;
        } else {
            validate_planes(&backing, description, &planes, sample_grid)?;
        }
        self.drain_notifications();
        let first_bytes = backing
            .snapshot_subrange(planes[0].offset, 1)
            .map_err(memory_error)?;
        let first = &first_bytes.segments()[0];
        let candidate = self.index.pages.get(&first.page()).and_then(|page| {
            page.resources().find_map(|resource| {
                let Resource::Image(id) = *resource else {
                    return None;
                };
                let entry = self.image(id).ok()?;
                // A framebuffer consumer needs only a subset of the producer's
                // usages. Reuse that exact representation instead of rebuilding
                // it through a buffer alias merely to remove usage flags.
                let compatible = NativeImageDescription {
                    usage: entry.description.usage,
                    flags: entry.description.flags,
                    ..description
                };
                (entry.description == compatible
                    && entry.description.usage.contains(description.usage)
                    && entry.description.flags.contains(description.flags)
                    && entry.decoded == decoded
                    && entry.sample_grid == sample_grid
                    && entry.planes == planes
                    && same_bytes(&entry.contents.backing, &backing))
                .then_some(id)
            })
        });
        if let Some(id) = candidate {
            self.touch(Resource::Image(id));
            self.statistics.cache_hits += 1;
            return Ok(id);
        }
        let owner = Arc::clone(&self.device.owner);
        let native = allocate_image_charged(
            &owner,
            description,
            AllocationMode::Managed,
            "resident image",
            |bytes| self.reserve(bytes),
        )?;
        let cpu = CanonicalCpuWriteDependency::capture(&backing).map_err(memory_error)?;
        let slot = self
            .images
            .iter()
            .position(|slot| slot.entry.is_none() && slot.generation < u32::MAX)
            .unwrap_or(self.images.len());
        if slot == self.images.len() {
            self.images.push(Slot {
                generation: 0,
                entry: None,
            });
        }
        let generation = self.images[slot].generation + 1;
        let id = ImageId {
            instance: self.instance,
            slot,
            generation,
        };
        self.images[slot] = Slot {
            generation,
            entry: Some(ImageEntry {
                contents: Contents {
                    backing: backing.clone(),
                    cpu,
                    invalid: if initialize {
                        vec![(0, backing.size())]
                    } else {
                        Vec::new()
                    },
                    last_use: None,
                    touched: 0,
                },
                native: Arc::new(native),
                description,
                planes: planes.clone(),
                sample_grid,
                views: Vec::new(),
                native_views: Vec::new(),
                access: sync::image_states(
                    &planes,
                    description.format,
                    self.device.profile().separate_depth_stencil_layouts,
                ),
                version: 0,
                producer: None,
                leases: 0,
                decoded,
                write_plans: Vec::new(),
                discarded: if initialize {
                    Vec::new()
                } else {
                    planes
                        .iter()
                        .enumerate()
                        .map(|(index, &plane)| (index, super::transfers::full_rect(plane)))
                        .collect()
                },
            }),
        };
        let index_planes = self.image(id)?.decoded.as_ref().map_or_else(
            || self.image(id).unwrap().planes.clone(),
            |d| d.planes.clone(),
        );
        self.index_image_planes(id, &index_planes)?;
        self.touch(Resource::Image(id));
        Ok(id)
    }
    pub fn ensure_image(&mut self, id: ImageId) -> Result<(), VulkanError> {
        let entry = self.image(id)?;
        if entry.contents.invalid.is_empty()
            && entry.contents.cpu.remains_current()
            && entry.discarded.is_empty()
        {
            return Ok(());
        }
        if self.refresh_native_alias(id)? {
            return Ok(());
        }
        let entry = self.image(id)?;
        if entry.contents.invalid.is_empty() && entry.contents.cpu.remains_current() {
            return Ok(());
        }
        if entry.description.samples != vk::SampleCountFlags::TYPE_1 {
            return Err(VulkanError::InvalidResource(
                "multisample image reconstruction requires explicit sample conversion",
            ));
        }
        if entry.leases != 0 {
            return Err(VulkanError::InvalidResource(
                "leased image version needs an ordered presentation read or a separate GPU snapshot before overwrite",
            ));
        }
        self.constructing.push(Resource::Image(id));
        let result = self.refresh_image(id);
        self.unpin(Resource::Image(id));
        result
    }
    /// Reconstruct an opaque shape alias only when a real read/LOAD needs its
    /// undefined texels. A partial destination write must not repeatedly prove
    /// or import the untouched remainder of a large image.
    pub(super) fn ensure_image_initialized(
        &mut self,
        id: ImageId,
        plane: ImagePlane,
        rect: vk::Rect2D,
    ) -> Result<(), VulkanError> {
        self.ensure_image(id)?;
        if self.image(id)?.discarded.iter().any(|&(index, undefined)| {
            self.image(id).unwrap().planes[index] == plane && transfers::intersects(undefined, rect)
        }) && self.device_alias_has_producers(id)?
        {
            let entry = self.images[id.slot].entry.as_mut().expect("image");
            entry.contents.invalid = vec![(0, entry.contents.backing.size())];
            self.ensure_image(id)?;
        }
        self.require_image_initialized(id, plane, rect)
    }
    fn device_alias_has_producers(&mut self, id: ImageId) -> Result<bool, VulkanError> {
        let entry = self.image(id)?;
        if entry.description.samples != vk::SampleCountFlags::TYPE_1 || entry.decoded.is_some() {
            return Ok(false);
        }
        let backing = entry.contents.backing.clone();
        self.drain_notifications();
        for backing_page in backing.pages() {
            if !self.gpu_owned(backing_page.identity(), backing_page.visibility_state()) {
                return Ok(false);
            }
            let page = self
                .index
                .pages
                .get_mut(&backing_page.identity())
                .expect("indexed image");
            let groups: Vec<_> = page.bindings_for(Resource::Image(id)).cloned().collect();
            for spans in groups {
                if !page.has_complete_producer(&spans) {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }
    /// Reinterpret an exact native producer without downloading guest bytes.
    /// Copy preserves each sample's bits; resolving before an sRGB reinterpretation
    /// would average in the wrong color space.
    fn refresh_native_alias(&mut self, id: ImageId) -> Result<bool, VulkanError> {
        let target = self.image(id)?;
        if target.leases != 0 || target.decoded.is_some() {
            return Ok(false);
        }
        let first_bytes = target
            .contents
            .backing
            .snapshot_subrange(target.planes[0].offset, 1)
            .map_err(memory_error)?;
        let source = self
            .index
            .pages
            .get(&first_bytes.segments()[0].page())
            .and_then(|page| {
                page.resources().find_map(|resource| {
                    let Resource::Image(source) = *resource else {
                        return None;
                    };
                    if source == id {
                        return None;
                    }
                    let entry = self.image(source).ok()?;
                    let compatible = NativeImageDescription {
                        format: target.description.format,
                        usage: target.description.usage,
                        flags: target.description.flags,
                        extent: target.description.extent,
                        ..entry.description
                    };
                    let formats_compatible = entry.description.format == target.description.format
                        || target.planes.iter().all(|p| {
                            p.aspect == vk::ImageAspectFlags::COLOR
                                && p.block_extent
                                    == vk::Extent2D {
                                        width: 1,
                                        height: 1,
                                    }
                                && format::copy_block(entry.description.format, p.aspect)
                                    == format::copy_block(target.description.format, p.aspect)
                        });
                    let same_mapping = entry.planes == target.planes
                        && same_bytes(&entry.contents.backing, &target.contents.backing);
                    let rectangular_mapping =
                        match (entry.planes.as_slice(), target.planes.as_slice()) {
                            ([a], [b])
                                if entry.description.samples == vk::SampleCountFlags::TYPE_1
                                    && a.offset == 0
                                    && b.offset == 0
                                    && a.extent.depth == 1
                                    && b.extent.depth == 1
                                    && a.extent.width >= b.extent.width
                                    && a.extent.height >= b.extent.height
                                    && (b.extent.width.is_multiple_of(b.block_extent.width)
                                        || b.extent.width == a.extent.width)
                                    && (b.extent.height.is_multiple_of(b.block_extent.height)
                                        || b.extent.height == a.extent.height)
                                    && a.mip_level == b.mip_level
                                    && a.array_layer == b.array_layer
                                    && a.aspect == b.aspect
                                    && a.block_extent == b.block_extent
                                    && a.bytes_per_block == b.bytes_per_block =>
                            {
                                entry.contents.backing.same_rectangular_bytes(
                                    a.row_pitch,
                                    &target.contents.backing,
                                    b.row_pitch,
                                    b.row_bytes(),
                                    b.rows(),
                                )
                            }
                            _ => false,
                        };
                    (compatible == target.description
                        && formats_compatible
                        && entry.sample_grid == target.sample_grid
                        && (same_mapping || rectangular_mapping)
                        && entry.decoded.is_none()
                        && entry.producer.is_some()
                        && !entry.discarded.iter().any(|&(plane, undefined)| {
                            target.planes.get(plane).is_none_or(|p| {
                                transfers::intersects(undefined, transfers::full_rect(*p))
                            })
                        })
                        && entry.contents.invalid.is_empty()
                        && entry.contents.cpu.remains_current())
                    .then_some(source)
                })
            });
        let Some(source) = source else {
            return Ok(false);
        };
        let planes = target.planes.clone();
        // Establish the destination observer before reading the producer. CPU
        // writes racing the copy must remain dirty, just as for buffer imports.
        let dependency =
            CanonicalCpuWriteDependency::capture(&target.contents.backing).map_err(memory_error)?;
        if !self.image(source)?.contents.cpu.remains_current() {
            return Ok(false);
        }
        // Size-compatible color formats and identical depth/stencil formats can
        // be copied with equal sample counts. This is a representation refresh,
        // not a new canonical writer.
        // https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdCopyImage.html
        for p in planes {
            let (src, src_layout) = self.image_access_subresources(
                source,
                commands::plane_range(p),
                AccessScope {
                    stages: vk::PipelineStageFlags2::COPY,
                    access: vk::AccessFlags2::TRANSFER_READ,
                },
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            )?;
            let (dst, dst_layout) = self.image_access_subresources(
                id,
                commands::plane_range(p),
                AccessScope {
                    stages: vk::PipelineStageFlags2::COPY,
                    access: vk::AccessFlags2::TRANSFER_WRITE,
                },
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            )?;
            let subresource = vk::ImageSubresourceLayers::default()
                .aspect_mask(p.aspect)
                .mip_level(p.mip_level)
                .base_array_layer(p.array_layer)
                .layer_count(1);
            let region = vk::ImageCopy::default()
                .src_subresource(subresource)
                .dst_subresource(subresource)
                .extent(p.extent);
            self.barriers.emit(&mut self.device);
            let slot = self.device.begin_recording()?;
            unsafe {
                self.device.owner.device.cmd_copy_image(
                    self.device.submission.slots[slot].command,
                    src,
                    src_layout,
                    dst,
                    dst_layout,
                    &[region],
                );
            }
        }
        let work = self.device.work_ticket()?;
        let entry = self.images[id.slot].entry.as_mut().expect("image");
        entry.contents.cpu = dependency;
        entry.contents.invalid.clear();
        entry.discarded.clear();
        entry.producer = Some(work);
        entry.version = entry
            .version
            .checked_add(1)
            .ok_or(VulkanError::InvalidResource("image version exhausted"))?;
        Ok(true)
    }
    /// A write-only operation must observe CPU changes without importing the
    /// contents it will overwrite. Untouched aspects/regions stay invalid until
    /// an actual reader consumes them; a complete clear validates its own bytes.
    pub(super) fn observe_image_cpu_writes(&mut self, id: ImageId) -> Result<(), VulkanError> {
        let entry = self.image(id)?;
        if entry.contents.cpu.remains_current() {
            return Ok(());
        }
        let backing = entry.contents.backing.clone();
        let dependency = entry.contents.cpu.clone();
        self.contents_mut(Resource::Image(id))
            .invalidate_intervals(&[(0, backing.size())]);
        self.drain_notifications();
        let cpu_pages: Vec<_> = backing
            .pages()
            .filter(|page| !self.gpu_owned(page.identity(), page.visibility_state()))
            .map(|page| page.identity())
            .collect();
        for page in cpu_pages {
            if let Some(index) = self.index.pages.get_mut(&page) {
                index.clear_writes();
            }
        }
        // CPU imports do not own the physical page's preserved baseline: an
        // earlier alias may already have prepared a write in this operation.
        // Canonical visibility transitions and final eviction retire it.
        dependency.rearm().map_err(memory_error)?;
        Ok(())
    }
    fn refresh_image(&mut self, id: ImageId) -> Result<(), VulkanError> {
        if self.image(id)?.decoded.is_some() {
            return self.refresh_decoded_image(id);
        }
        let entry = self.image(id)?;
        let backing = entry.contents.backing.clone();
        let dependency = entry.contents.cpu.clone();
        let mut dirty = entry.contents.invalid.clone();
        let direct_plane = match entry.planes.as_slice() {
            [p] if p.offset == 0
                && p.extent.depth == 1
                && p.row_pitch == p.row_bytes()
                && p.end() == Some(backing.size())
                && p.row_pitch <= self.device.submission.configuration.arena_bytes
                && (dirty.is_empty() || dirty == [(0, backing.size())])
                && backing
                    .pages()
                    .all(|p| !self.gpu_owned(p.identity(), p.visibility_state())) =>
            {
                Some(*p)
            }
            _ => None,
        };
        // This snapshot establishes the image observer BEFORE native imports.
        // CPU writes after it stay dirty even if the buffer import sees them.
        let snapshots = if let Some(p) = direct_plane {
            dependency
                .snapshot_with_resolver(
                    &backing,
                    if dirty.is_empty() {
                        CpuWriteSnapshotSelection::DirtyPages
                    } else {
                        CpuWriteSnapshotSelection::All
                    },
                    p.row_pitch,
                    &mut |_, _| {
                        Err(nixe_memory::VisibilityCoordinatorError::new(
                            "CPU image snapshot crossed concurrent GPU ownership",
                        ))
                    },
                )
                .map_err(memory_error)?
        } else {
            let imports = self.cpu_imports(&backing, &dependency, &[])?;
            dirty.extend(imports.gpu_invalidated);
            imports.snapshots
        };
        for (offset, bytes) in &snapshots {
            dirty.push((*offset, bytes.len() as u64));
        }
        normalize(&mut dirty);
        self.images[id.slot]
            .entry
            .as_mut()
            .expect("image")
            .contents
            .invalid = dirty.clone();
        if dirty.is_empty() {
            return Ok(());
        }
        if let Some(plane) = direct_plane {
            // CPU-authored texels already exist in this immutable snapshot.
            // Stage them once; a persistent canonical buffer would duplicate
            // both the snapshot and the GPU transfer, and index tiled fragments.
            for page in backing.pages() {
                if let Some(indexed) = self.index.pages.get_mut(&page.identity()) {
                    indexed.clear_writes();
                }
            }
            for (offset, bytes) in snapshots {
                self.upload_cpu_image_rows(id, plane, offset, &bytes)?;
            }
        } else {
            let source = self.buffer_for(backing, vk::BufferUsageFlags::TRANSFER_SRC)?;
            self.constructing.push(Resource::Buffer(source));
            let result = self.copy_rows_to_image(source, 0, id, &dirty);
            self.unpin(Resource::Buffer(source));
            result?;
        }
        let producer = self.device.work_ticket()?;
        let entry = self.images[id.slot].entry.as_mut().expect("image");
        entry.producer = Some(producer);
        entry.contents.invalid.clear();
        entry.version = entry
            .version
            .checked_add(1)
            .ok_or(VulkanError::InvalidResource(
                "image content version exhausted",
            ))?;
        Ok(())
    }
    fn upload_cpu_image_rows(
        &mut self,
        id: ImageId,
        plane: ImagePlane,
        offset: u64,
        bytes: &[u8],
    ) -> Result<(), VulkanError> {
        debug_assert!(offset.is_multiple_of(plane.row_pitch));
        debug_assert!((bytes.len() as u64).is_multiple_of(plane.row_pitch));
        let rows = self.device.submission.configuration.arena_bytes / plane.row_pitch;
        let chunk_bytes = (rows * plane.row_pitch) as usize;
        for (chunk, bytes) in bytes.chunks(chunk_bytes).enumerate() {
            let row = (offset + (chunk * chunk_bytes) as u64) / plane.row_pitch;
            let y = row as u32 * plane.block_extent.height;
            let height = (bytes.len() as u64 / plane.row_pitch) as u32 * plane.block_extent.height;
            let height = height.min(plane.extent.height - y);
            let (slot, slice) =
                self.reserve_upload(bytes, u64::from(plane.bytes_per_block).max(4))?;
            let (image, layout) = self.image_access_subresources(
                id,
                commands::plane_range(plane),
                AccessScope::new(
                    vk::PipelineStageFlags2::COPY,
                    vk::AccessFlags2::TRANSFER_WRITE,
                ),
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            )?;
            let region = vk::BufferImageCopy::default()
                .buffer_offset(slice.offset)
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(plane.aspect)
                        .mip_level(plane.mip_level)
                        .base_array_layer(plane.array_layer)
                        .layer_count(1),
                )
                .image_offset(vk::Offset3D {
                    x: 0,
                    y: y as i32,
                    z: 0,
                })
                .image_extent(vk::Extent3D {
                    width: plane.extent.width,
                    height,
                    depth: 1,
                });
            self.barriers.emit(&mut self.device);
            unsafe {
                self.device.owner.device.cmd_copy_buffer_to_image(
                    self.device.submission.slots[slot].command,
                    self.device.submission.slots[slot]
                        .upload_buffer(slice)
                        .handle,
                    image,
                    layout,
                    &[region],
                );
            }
            self.define_image_rect(
                id,
                plane,
                vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: y as i32 },
                    extent: vk::Extent2D {
                        width: plane.extent.width,
                        height,
                    },
                },
            );
            self.statistics.cpu_uploaded_bytes += bytes.len() as u64;
        }
        Ok(())
    }
    fn copy_rows_to_image(
        &mut self,
        source: BufferId,
        source_offset: u64,
        image: ImageId,
        dirty: &[(u64, u64)],
    ) -> Result<(), VulkanError> {
        let planes = self.image(image)?.planes.clone();
        let mut pieces = Vec::new();
        for plane in planes {
            for row in 0..plane.rows() {
                let offset = plane.offset + plane.row_pitch * u64::from(row);
                if dirty.iter().any(|&(start, length)| {
                    start < offset + plane.row_bytes() && offset < start + length
                }) {
                    self.check_buffer_range(source, source_offset + offset, plane.row_bytes())?;
                    pieces.push(ImagePiece {
                        plane,
                        row,
                        rows: 1,
                        first: 0,
                        blocks: plane.extent.width.div_ceil(plane.block_extent.width),
                        skip: 0,
                        logical: offset,
                        length: plane.row_bytes(),
                    });
                }
            }
        }
        let pieces = coalesce_dense_rows(pieces);
        if pieces.is_empty() {
            return Ok(());
        }
        let scratch = if pieces
            .iter()
            .all(|piece| piece.direct(source_offset + piece.logical))
        {
            None
        } else {
            let (size, slices) = packed_slices(&pieces)?;
            let scratch =
                self.temporary_buffer(size, &[Resource::Buffer(source), Resource::Image(image)])?;
            Some((scratch, slices))
        };
        let src = self.buffer(source)?.native.handle;
        for piece in &pieces {
            self.buffer_access_range(
                source,
                source_offset + piece.logical,
                piece.length,
                AccessScope::new(
                    vk::PipelineStageFlags2::COPY,
                    vk::AccessFlags2::TRANSFER_READ,
                ),
            )?;
        }
        let (dst, layout) =
            self.image_regions_access(image, &pieces, vk::AccessFlags2::TRANSFER_WRITE)?;
        let index = self.device.begin_recording()?;
        let command = self.device.submission.slots[index].command;
        let regions = if let Some((scratch, slices)) = &scratch {
            let copies: Vec<_> = pieces
                .iter()
                .zip(slices)
                .map(|(piece, &start)| {
                    vk::BufferCopy::default()
                        .src_offset(source_offset + piece.logical)
                        .dst_offset(start)
                        .size(piece.length)
                })
                .collect();
            self.barriers.emit(&mut self.device);
            unsafe {
                self.device
                    .owner
                    .device
                    .cmd_copy_buffer(command, src, scratch.handle, &copies);
                scratch_read_barrier(&self.device, command, scratch.handle, scratch.size());
            }
            pieces
                .iter()
                .zip(slices)
                .map(|(piece, &start)| piece.region(start))
                .collect::<Vec<_>>()
        } else {
            pieces
                .iter()
                .map(|piece| piece.region(source_offset + piece.logical))
                .collect()
        };
        self.barriers.emit(&mut self.device);
        unsafe {
            self.device.owner.device.cmd_copy_buffer_to_image(
                command,
                scratch.as_ref().map_or(src, |(buffer, _)| buffer.handle),
                dst,
                layout,
                &regions,
            );
        }
        if let Some((scratch, _)) = scratch {
            let work = self.device.work_ticket()?;
            self.device.retire_buffer(scratch, work)?;
        }
        for piece in &pieces {
            self.define_image_rect(
                image,
                piece.plane,
                vk::Rect2D {
                    offset: vk::Offset2D {
                        x: 0,
                        y: (piece.row * piece.plane.block_extent.height) as i32,
                    },
                    extent: vk::Extent2D {
                        width: piece.plane.extent.width,
                        height: (piece.rows * piece.plane.block_extent.height).min(
                            piece.plane.extent.height - piece.row * piece.plane.block_extent.height,
                        ),
                    },
                },
            );
        }
        self.statistics.gpu_copied_bytes += pieces.iter().map(|piece| piece.length).sum::<u64>();
        Ok(())
    }
    /// Transfer-authored image bytes become authoritative only for texel rows;
    /// canonical pitch padding and disjoint writers remain untouched.
    pub fn write_image_from_buffer(
        &mut self,
        image: ImageId,
        source: BufferId,
        source_offset: u64,
    ) -> Result<WorkTicket, VulkanError> {
        self.ensure_buffer(source)?;
        self.ensure_image(image)?;
        if self.image(image)?.leases != 0 {
            return Err(VulkanError::InvalidResource(
                "cannot overwrite a leased image",
            ));
        }
        let planes = self.image(image)?.planes.clone();
        for &plane in &planes {
            self.prepare_image_rect(image, plane, transfers::full_rect(plane))?;
        }
        let size = self.image(image)?.contents.backing.size();
        self.copy_rows_to_image(source, source_offset, image, &[(0, size)])?;
        for plane in planes {
            // Ordered rows can refer to the same canonical bytes. Publishing a
            // merged logical interval would erase the earlier row's dependency
            // on the later producer in that same image.
            for row in 0..plane.rows() {
                let rect = vk::Rect2D {
                    offset: vk::Offset2D {
                        x: 0,
                        y: (row * plane.block_extent.height) as i32,
                    },
                    extent: vk::Extent2D {
                        width: plane.extent.width,
                        height: plane
                            .block_extent
                            .height
                            .min(plane.extent.height - row * plane.block_extent.height),
                    },
                };
                for (offset, size) in self.canonical_image_rect_bytes(image, plane, rect)? {
                    self.record_authority(Resource::Image(image), offset, size)?;
                }
            }
        }
        let entry = self.images[image.slot].entry.as_mut().expect("image");
        // Repeated canonical offsets can invalidate other rows in this same
        // image. Preserve those dependencies; only each exact producer interval
        // was validated by record_authority.
        entry.producer = entry.contents.last_use;
        entry.discarded.clear();
        entry.version = entry
            .version
            .checked_add(1)
            .ok_or(VulkanError::InvalidResource(
                "image content version exhausted",
            ))?;
        self.device.work_ticket()
    }
    pub fn image_view(
        &mut self,
        image: ImageId,
        description: ImageViewDescription,
    ) -> Result<ImageViewId, VulkanError> {
        let entry = self.image(image)?;
        if let Some(view) = entry.views.iter().position(|view| *view == description) {
            return Ok(ImageViewId { image, view });
        }
        let d = description;
        let same_format = d.format == entry.description.format;
        let rgba_pair = matches!(
            (entry.description.format, d.format),
            (vk::Format::R8G8B8A8_UNORM, vk::Format::R8G8B8A8_SRGB)
                | (vk::Format::R8G8B8A8_SRGB, vk::Format::R8G8B8A8_UNORM)
                | (vk::Format::B8G8R8A8_UNORM, vk::Format::B8G8R8A8_SRGB)
                | (vk::Format::B8G8R8A8_SRGB, vk::Format::B8G8R8A8_UNORM)
        );
        if !same_format
            && (!rgba_pair
                || !entry
                    .description
                    .flags
                    .contains(vk::ImageCreateFlags::MUTABLE_FORMAT))
        {
            return Err(VulkanError::InvalidResource(
                "image view format requires a supported mutable-format compatibility class",
            ));
        }
        if d.mip_count == 0
            || d.layer_count == 0
            || d.aspects.is_empty()
            || d.base_mip
                .checked_add(d.mip_count)
                .is_none_or(|end| end > entry.description.mip_levels)
            || d.base_layer
                .checked_add(d.layer_count)
                .is_none_or(|end| end > entry.description.array_layers)
        {
            return Err(VulkanError::InvalidResource(
                "image view exceeds resident subresources",
            ));
        }
        let aspects = entry
            .planes
            .iter()
            .fold(vk::ImageAspectFlags::empty(), |aspects, plane| {
                aspects | plane.aspect
            });
        if !aspects.contains(d.aspects) {
            return Err(VulkanError::InvalidResource(
                "image view aspect has no native plane",
            ));
        }
        for level in d.base_mip..d.base_mip + d.mip_count {
            for layer in d.base_layer..d.base_layer + d.layer_count {
                for aspect in [
                    vk::ImageAspectFlags::COLOR,
                    vk::ImageAspectFlags::DEPTH,
                    vk::ImageAspectFlags::STENCIL,
                ] {
                    if d.aspects.contains(aspect)
                        && !entry.planes.iter().any(|plane| {
                            plane.mip_level == level
                                && plane.array_layer == layer
                                && plane.aspect == aspect
                        })
                    {
                        return Err(VulkanError::InvalidResource(
                            "image view references an uninitialized native subresource",
                        ));
                    }
                }
            }
        }
        match d.view_type {
            vk::ImageViewType::TYPE_2D if d.layer_count == 1 => {}
            vk::ImageViewType::TYPE_2D_ARRAY => {}
            vk::ImageViewType::CUBE | vk::ImageViewType::CUBE_ARRAY
                if entry
                    .description
                    .flags
                    .contains(vk::ImageCreateFlags::CUBE_COMPATIBLE)
                    && d.layer_count.is_multiple_of(6)
                    && (d.view_type != vk::ImageViewType::CUBE || d.layer_count == 6) => {}
            _ => {
                return Err(VulkanError::InvalidResource(
                    "image view type is incompatible with resident image/layers",
                ));
            }
        }
        if self
            .device
            .profile()
            .portability
            .as_ref()
            .is_some_and(|p| !p.image_view_format_swizzle)
            && d.swizzle != [vk::ComponentSwizzle::IDENTITY; 4]
        {
            return Err(VulkanError::InvalidResource(
                "component swizzle requires imageViewFormatSwizzle on this portability device",
            ));
        }
        let info = vk::ImageViewCreateInfo::default()
            .image(entry.native.handle)
            .view_type(d.view_type)
            .format(d.format)
            .components(vk::ComponentMapping {
                r: d.swizzle[0],
                g: d.swizzle[1],
                b: d.swizzle[2],
                a: d.swizzle[3],
            })
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(d.aspects)
                    .base_mip_level(d.base_mip)
                    .level_count(d.mip_count)
                    .base_array_layer(d.base_layer)
                    .layer_count(d.layer_count),
            );
        // Alternate views must support the inherited usage on this device.
        // https://docs.vulkan.org/refpages/latest/refpages/source/VkImageViewCreateInfo.html
        let _supported = unsafe {
            self.device
                .owner
                .instance
                .instance
                .get_physical_device_image_format_properties(
                    self.device.owner.physical,
                    d.format,
                    entry.description.image_type,
                    vk::ImageTiling::OPTIMAL,
                    entry.description.usage,
                    entry.description.flags,
                )
        }
        .map_err(|e| VulkanError::vk("image view format usage", e))?;
        let handle = unsafe { self.device.owner.device.create_image_view(&info, None) }
            .map_err(|e| VulkanError::vk("vkCreateImageView", e))?;
        let entry = self.images[image.slot].entry.as_mut().expect("image");
        let view = entry.views.len();
        entry.views.push(description);
        entry
            .native_views
            .push(Arc::new(crate::AllocatedImageView::from_handle(
                Arc::clone(&entry.native),
                handle,
            )));
        Ok(ImageViewId { image, view })
    }
    pub fn lease_image(&mut self, image: ImageId) -> Result<PresentationLease, VulkanError> {
        self.ensure_image(image)?;
        self.flush()?;
        let entry = self.image(image)?;
        let producer = entry
            .producer
            .ok_or(VulkanError::InvalidResource(
                "image has no producing native work",
            ))?
            .submission();
        let version = entry.version;
        self.images[image.slot]
            .entry
            .as_mut()
            .expect("image")
            .leases += 1;
        Ok(PresentationLease::new(
            PresentationImageId {
                slot: image.slot as u64,
                generation: image.generation,
            },
            version,
            producer,
            self.release_sender.clone(),
        ))
    }
    pub(super) fn image_pieces(
        &self,
        id: ImageId,
        offset: u64,
        length: u64,
    ) -> Result<Vec<ImagePiece>, VulkanError> {
        if self.image(id)?.description.samples != vk::SampleCountFlags::TYPE_1 {
            return Err(VulkanError::InvalidResource(
                "multisample samples cannot be copied as ordinary buffer texels",
            ));
        }
        let end = offset
            .checked_add(length)
            .ok_or(VulkanError::InvalidResource("image interval overflow"))?;
        let mut result = Vec::new();
        let mut consumed = 0;
        for &plane in &self.image(id)?.planes {
            // Authority often names only a few bytes of one packed texel.
            // Locate its rows arithmetically instead of scanning the full mip
            // again for every physical fragment of a CPU visibility request.
            if end <= plane.offset {
                continue;
            }
            let first_row = offset.saturating_sub(plane.offset) / plane.row_pitch;
            let last_row = (end - plane.offset).div_ceil(plane.row_pitch);
            for row in first_row.min(u64::from(plane.rows())) as u32
                ..last_row.min(u64::from(plane.rows())) as u32
            {
                let row_start = plane.offset + plane.row_pitch * u64::from(row);
                let start = offset.max(row_start);
                let finish = end.min(row_start + plane.row_bytes());
                if start >= finish {
                    continue;
                }
                let block = u64::from(plane.bytes_per_block);
                let first = (start - row_start) / block;
                let last = (finish - row_start).div_ceil(block);
                result.push(ImagePiece {
                    plane,
                    row,
                    rows: 1,
                    first: first as u32,
                    blocks: (last - first) as u32,
                    skip: start - row_start - first * block,
                    logical: start,
                    length: finish - start,
                });
                consumed += finish - start;
            }
        }
        for piece in &result {
            self.require_image_initialized(
                id,
                piece.plane,
                vk::Rect2D {
                    offset: vk::Offset2D {
                        x: (piece.first * piece.plane.block_extent.width) as i32,
                        y: (piece.row * piece.plane.block_extent.height) as i32,
                    },
                    extent: vk::Extent2D {
                        width: (piece.blocks * piece.plane.block_extent.width).min(
                            piece.plane.extent.width - piece.first * piece.plane.block_extent.width,
                        ),
                        height: piece.plane.block_extent.height.min(
                            piece.plane.extent.height - piece.row * piece.plane.block_extent.height,
                        ),
                    },
                },
            )?;
        }
        result.sort_unstable_by_key(|piece| piece.logical);
        if consumed != length {
            return Err(VulkanError::InvalidResource(
                "image producer interval crosses padding or missing plane",
            ));
        }
        Ok(coalesce_dense_rows(result))
    }
    pub(super) fn image_regions_access(
        &mut self,
        id: ImageId,
        pieces: &[ImagePiece],
        access: vk::AccessFlags2,
    ) -> Result<(vk::Image, vk::ImageLayout), VulkanError> {
        let mut result = (vk::Image::null(), vk::ImageLayout::UNDEFINED);
        let conventional = if access == vk::AccessFlags2::TRANSFER_READ {
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL
        } else {
            vk::ImageLayout::TRANSFER_DST_OPTIMAL
        };
        for piece in pieces {
            result = self.image_access_subresources(
                id,
                super::commands::plane_range(piece.plane),
                AccessScope::new(vk::PipelineStageFlags2::COPY, access),
                conventional,
            )?;
        }
        Ok(result)
    }
    /// Download whole transfer blocks once, then scatter only authoritative
    /// bytes on the CPU. D24 padding and independent stencil/alias producers
    /// must never become canonical merely because a native transfer reads them.
    pub(super) fn image_read_pieces(
        &self,
        id: ImageId,
        writes: &[Authority],
        limit: u64,
    ) -> Result<Vec<ImagePiece>, VulkanError> {
        let mut pieces = Vec::new();
        for write in writes {
            for piece in self.image_pieces(id, write.offset, write.end - write.start)? {
                let block = u64::from(piece.plane.bytes_per_block);
                for row in piece.row..piece.row + piece.rows {
                    pieces.push(ImagePiece {
                        row,
                        rows: 1,
                        skip: 0,
                        logical: piece.plane.offset
                            + u64::from(row) * piece.plane.row_pitch
                            + u64::from(piece.first) * block,
                        length: u64::from(piece.blocks) * block,
                        ..piece
                    });
                }
            }
        }
        pieces.sort_unstable_by_key(|piece| piece.logical);
        let mut merged: Vec<ImagePiece> = Vec::new();
        for piece in pieces {
            if let Some(last) = merged.last_mut()
                && last.plane == piece.plane
                && last.row == piece.row
                && piece.first <= last.first + last.blocks
            {
                last.blocks =
                    (piece.first + piece.blocks).max(last.first + last.blocks) - last.first;
                last.length = u64::from(last.blocks) * u64::from(last.plane.bytes_per_block);
            } else {
                merged.push(piece);
            }
        }
        let mut bounded = Vec::new();
        for piece in merged {
            let block = u64::from(piece.plane.bytes_per_block);
            let blocks =
                u32::try_from((limit / block).min(u64::from(u32::MAX))).map_err(memory_error)?;
            if blocks == 0 {
                return Err(VulkanError::InvalidResource(
                    "readback arena cannot hold an image block",
                ));
            }
            let mut consumed = 0;
            while consumed < piece.blocks {
                let count = blocks.min(piece.blocks - consumed);
                bounded.push(ImagePiece {
                    first: piece.first + consumed,
                    blocks: count,
                    logical: piece.logical + u64::from(consumed) * block,
                    length: u64::from(count) * block,
                    ..piece
                });
                consumed += count;
            }
        }
        Ok(bounded)
    }
    pub(super) fn image_pieces_buffer(
        &mut self,
        id: ImageId,
        pieces: &[ImagePiece],
        references: &[Resource],
    ) -> Result<(vk::Buffer, Vec<u64>), VulkanError> {
        let (size, slices) = packed_slices(pieces)?;
        let mut references = references.to_vec();
        references.push(Resource::Image(id));
        let scratch = self.temporary_buffer(size, &references)?;
        let mut source = vk::Image::null();
        let mut layout = vk::ImageLayout::UNDEFINED;
        for piece in pieces {
            (source, layout) = self.image_access_subresources(
                id,
                super::commands::plane_range(piece.plane),
                AccessScope::new(
                    vk::PipelineStageFlags2::COPY,
                    vk::AccessFlags2::TRANSFER_READ,
                ),
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            )?;
        }
        let index = self.device.begin_recording()?;
        let command = self.device.submission.slots[index].command;
        let regions: Vec<_> = pieces
            .iter()
            .zip(&slices)
            .map(|(piece, &offset)| piece.region(offset))
            .collect();
        self.barriers.emit(&mut self.device);
        unsafe {
            self.device.owner.device.cmd_copy_image_to_buffer(
                command,
                source,
                layout,
                scratch.handle,
                &regions,
            );
            scratch_read_barrier(&self.device, command, scratch.handle, size);
        }
        let handle = scratch.handle;
        let work = self.device.work_ticket()?;
        self.device.retire_buffer(scratch, work)?;
        Ok((handle, slices))
    }
}

#[derive(Clone, Copy)]
pub(super) struct ImagePiece {
    pub(super) plane: ImagePlane,
    pub(super) row: u32,
    pub(super) rows: u32,
    pub(super) first: u32,
    pub(super) blocks: u32,
    pub(super) skip: u64,
    pub(super) logical: u64,
    pub(super) length: u64,
}
impl ImagePiece {
    pub(super) fn direct(self, offset: u64) -> bool {
        self.skip == 0
            && self.length
                == u64::from(self.blocks)
                    * u64::from(self.plane.bytes_per_block)
                    * u64::from(self.rows)
            && offset.is_multiple_of(u64::from(self.plane.bytes_per_block).max(4))
    }
    pub(super) fn region(self, offset: u64) -> vk::BufferImageCopy {
        let plane = self.plane;
        let x = self.first * plane.block_extent.width;
        let y = self.row * plane.block_extent.height;
        vk::BufferImageCopy::default()
            .buffer_offset(offset)
            .image_subresource(plane.layers())
            .image_offset(vk::Offset3D {
                x: x as i32,
                y: y as i32,
                z: 0,
            })
            .image_extent(vk::Extent3D {
                width: (self.blocks * plane.block_extent.width).min(plane.extent.width - x),
                height: (plane.block_extent.height * self.rows).min(plane.extent.height - y),
                depth: 1,
            })
    }
}
fn coalesce_dense_rows(pieces: Vec<ImagePiece>) -> Vec<ImagePiece> {
    let mut merged: Vec<ImagePiece> = Vec::with_capacity(pieces.len());
    for piece in pieces {
        if let Some(last) = merged.last_mut()
            && last.plane == piece.plane
            && last.first == 0
            && piece.first == 0
            && last.skip == 0
            && piece.skip == 0
            && last.blocks == piece.blocks
            && last.plane.row_pitch == last.plane.row_bytes()
            && last.length == last.plane.row_bytes() * u64::from(last.rows)
            && piece.length == piece.plane.row_bytes()
            && last.row + last.rows == piece.row
            && last.logical + last.length == piece.logical
        {
            last.rows += piece.rows;
            last.length += piece.length;
        } else {
            merged.push(piece);
        }
    }
    merged
}
fn packed_slices(pieces: &[ImagePiece]) -> Result<(u64, Vec<u64>), VulkanError> {
    let mut size = 0;
    let mut slices = Vec::with_capacity(pieces.len());
    for piece in pieces {
        size = crate::queue::align_up(size, u64::from(piece.plane.bytes_per_block).max(4)).ok_or(
            VulkanError::InvalidResource("packed image copy size overflow"),
        )?;
        slices.push(size);
        size = size
            .checked_add(
                u64::from(piece.blocks)
                    * u64::from(piece.plane.bytes_per_block)
                    * u64::from(piece.rows),
            )
            .ok_or(VulkanError::InvalidResource(
                "packed image copy size overflow",
            ))?;
    }
    Ok((size, slices))
}

fn validate_planes(
    backing: &CanonicalBackingRange,
    description: NativeImageDescription,
    planes: &[ImagePlane],
    sample_grid: [u32; 2],
) -> Result<(), VulkanError> {
    if planes.is_empty() {
        return Err(VulkanError::InvalidResource(
            "image residency needs native copy planes",
        ));
    }
    if description.image_type != vk::ImageType::TYPE_2D
        || sample_grid[0] == 0
        || sample_grid[1] == 0
        || sample_grid[0].checked_mul(sample_grid[1]) != Some(description.samples.as_raw())
        || description.samples.as_raw().count_ones() != 1
        || description.samples != vk::SampleCountFlags::TYPE_1 && description.mip_levels != 1
    {
        return Err(VulkanError::InvalidResource(
            "image residency requires 2D planes and a valid frontend sample grid",
        ));
    }
    let mut prior = 0;
    let mut subresources = std::collections::HashSet::new();
    for plane in planes {
        if !subresources.insert((plane.mip_level, plane.array_layer, plane.aspect.as_raw())) {
            return Err(VulkanError::InvalidResource(
                "duplicate native image subresource plane",
            ));
        }
        if plane.extent.width == 0
            || plane.extent.height == 0
            || plane.extent.depth != 1
            || plane.block_extent.width == 0
            || plane.block_extent.height == 0
            || plane.bytes_per_block == 0
            || plane.aspect.as_raw().count_ones() != 1
            || plane.mip_level >= description.mip_levels
            || plane.array_layer >= description.array_layers
        {
            return Err(VulkanError::InvalidResource(
                "invalid native image copy plane",
            ));
        }
        if format::copy_block(description.format, plane.aspect)
            != Some((plane.block_extent, plane.bytes_per_block))
        {
            return Err(VulkanError::InvalidResource(
                "native copy plane geometry does not match format/aspect; packed or unsupported encodings require conversion recording",
            ));
        }
        let width = description
            .extent
            .width
            .checked_shr(plane.mip_level)
            .unwrap_or(0)
            .max(1);
        let height = description
            .extent
            .height
            .checked_shr(plane.mip_level)
            .unwrap_or(0)
            .max(1);
        if plane.extent.width != width
            || plane.extent.height != height
            || plane.mip_level >= u32::BITS
        {
            return Err(VulkanError::InvalidResource(
                "image copy plane extent does not match native mip extent",
            ));
        }
        let storage = ImagePlane {
            extent: vk::Extent3D {
                width: width
                    .checked_mul(sample_grid[0])
                    .ok_or(VulkanError::InvalidResource("sample width overflow"))?,
                height: height
                    .checked_mul(sample_grid[1])
                    .ok_or(VulkanError::InvalidResource("sample height overflow"))?,
                depth: 1,
            },
            ..*plane
        };
        if plane.row_pitch < storage.row_bytes()
            || plane.offset < prior
            || storage.end().is_none_or(|end| end > backing.size())
        {
            return Err(VulkanError::InvalidResource(
                "overlapping or out-of-bounds image sample planes",
            ));
        }
        prior = storage.end().expect("checked sample extent");
    }
    Ok(())
}

impl VulkanResidency {
    pub fn image_view_lease(
        &self,
        id: ImageViewId,
    ) -> Result<Arc<crate::AllocatedImageView>, VulkanError> {
        self.image(id.image)?
            .native_views
            .get(id.view)
            .cloned()
            .ok_or(VulkanError::InvalidResource("stale image view"))
    }
}
