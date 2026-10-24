//! Native copies consume the shared ordered access state before their commands.
//! Canonical producer tracking remains independent of host barrier emission.
use super::*;
use crate::ReadbackTicket;
impl VulkanResidency {
    /// Capture dirty RAM directly into one mapped upload batch, bounded by the
    /// arena capacity. Region offsets retain their order across chunk boundaries.
    /// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/buffer_cache/buffer_cache.h
    pub(super) fn upload_buffer_captures(
        &mut self,
        id: BufferId,
        inputs: &[nixe_memory::AllocationRange],
        copies: &[nixe_memory::PhysicalCapture],
        windows: &[(u64, usize, usize)],
        packed: usize,
    ) -> Result<(), VulkanError> {
        let limit = self.device.submission.configuration.arena_bytes as usize;
        let mut captures = Vec::new();
        let mut regions = Vec::new();
        let mut retained = Vec::new();
        let mut input_index = 0;
        let mut input_offset = 0;
        for begin in (0..packed).step_by(limit) {
            let end = (begin + limit).min(packed);
            retained.clear();
            let mut remaining = (end - begin) as u64;
            while remaining != 0 {
                let input = inputs.get(input_index).ok_or(VulkanError::InvalidResource(
                    "buffer upload capture exceeds retained inputs",
                ))?;
                let bytes = remaining.min(input.bytes() - input_offset);
                retained.push(input.slice(input_offset, bytes)?);
                remaining -= bytes;
                input_offset += bytes;
                if input_offset == input.bytes() {
                    input_index += 1;
                    input_offset = 0;
                }
            }
            captures.clear();
            let first = copies
                .partition_point(|copy| copy.destination + copy.source.bytes as usize <= begin);
            for copy in &copies[first..] {
                if copy.destination >= end {
                    break;
                }
                let start = copy.destination.max(begin);
                let stop = (copy.destination + copy.source.bytes as usize).min(end);
                captures.push(nixe_memory::PhysicalCapture {
                    source: nixe_memory::Extent {
                        start: nixe_memory::ArenaOffset::new(
                            copy.source.start.get() + (start - copy.destination) as u64,
                        ),
                        bytes: (stop - start) as u64,
                    },
                    destination: start - begin,
                });
            }
            // Resolve lazy literals/hazards before reserving the capture's slot:
            // those operations may consume staging or submit earlier commands.
            regions.clear();
            let first = windows.partition_point(|&(_, start, bytes)| start + bytes <= begin);
            for &(offset, start, bytes) in &windows[first..] {
                if start >= end {
                    break;
                }
                let low = start.max(begin);
                let high = (start + bytes).min(end);
                let destination = offset + (low - start) as u64;
                self.buffer_access_range(
                    id,
                    destination,
                    (high - low) as u64,
                    AccessScope::new(
                        vk::PipelineStageFlags2::COPY,
                        vk::AccessFlags2::TRANSFER_WRITE,
                    ),
                )?;
                regions.push(
                    vk::BufferCopy::default()
                        .src_offset((low - begin) as u64)
                        .dst_offset(destination)
                        .size((high - low) as u64),
                );
            }
            let (slot, slice) = self.reserve_upload_space((end - begin) as u64, 4)?;
            let upload = self.device.submission.slots[slot].upload_buffer(slice);
            unsafe {
                upload.write_with(slice.offset, slice.size, |destination| {
                    self.memory
                        .capture_resource_scattered(&retained, &captures, destination)
                        .map_err(memory_error)
                })?;
            }
            let source = upload.handle;
            for region in &mut regions {
                region.src_offset += slice.offset;
            }
            let work = self.device.work_ticket()?;
            self.contents_mut(Resource::Buffer(id)).last_use = Some(work);
            self.barriers.emit(&mut self.device);
            unsafe {
                self.device.dispatch.cmd_copy_buffer(
                    self.device.submission.slots[slot].command,
                    source,
                    self.buffer(id)?.native.handle,
                    &regions,
                );
            }
            for region in &regions {
                self.buffer_mut(id)
                    .validate_interval(region.dst_offset, region.size);
            }
            self.invalidate_cpu_resources(&[])?;
            self.statistics.cpu_uploaded_bytes += (end - begin) as u64;
        }
        Ok(())
    }
    pub(super) fn check_buffer_range(
        &self,
        id: BufferId,
        offset: u64,
        size: u64,
    ) -> Result<(), VulkanError> {
        if size == 0
            || offset.checked_add(size).is_none_or(|end| {
                end > self
                    .buffer(id)
                    .map(|entry| entry.native.size())
                    .unwrap_or(0)
            })
        {
            return Err(VulkanError::InvalidResource(
                "buffer transfer exceeds resident interval",
            ));
        }
        Ok(())
    }
    pub(super) fn reserve_upload(
        &mut self,
        bytes: &[u8],
        alignment: u64,
    ) -> Result<(usize, crate::queue::Slice), VulkanError> {
        let (slot, slice) = self.reserve_upload_space(bytes.len() as u64, alignment)?;
        let buffer = self.device.submission.slots[slot].upload_buffer(slice);
        unsafe {
            buffer.write_range(slice.offset, bytes)?;
        }
        Ok((slot, slice))
    }
    pub(super) fn reserve_upload_space(
        &mut self,
        size: u64,
        alignment: u64,
    ) -> Result<(usize, crate::queue::Slice), VulkanError> {
        loop {
            match self.device.submission.reserve_slice(size, true, alignment) {
                Ok(reservation) => return Ok(reservation),
                Err(error @ VulkanError::QueuePressure { pending, .. }) => {
                    let flushed = self.flush()?;
                    let Some(work) = pending.or(flushed) else {
                        return Err(error);
                    };
                    self.statistics.staging_pressure_waits += 1;
                    let _wait =
                        nixe_gpu::trace::span("staging_pressure_wait", work.tick(), size, 0);
                    self.device.wait_for_work(work, std::time::Duration::MAX)?;
                }
                Err(error) => return Err(error),
            }
        }
    }
    pub(super) fn upload_buffer(
        &mut self,
        id: BufferId,
        offset: u64,
        bytes: &[u8],
    ) -> Result<(), VulkanError> {
        self.check_buffer_range(id, offset, bytes.len() as u64)?;
        // Chunking is a staging policy, never a guest buffer-size restriction.
        let limit = self.device.submission.configuration.arena_bytes as usize;
        for (part, bytes) in bytes.chunks(limit).enumerate() {
            let (index, slice) = self.reserve_upload(bytes, 4)?;
            let target = self.buffer_access_range(
                id,
                offset + (part * limit) as u64,
                bytes.len() as u64,
                AccessScope::new(
                    vk::PipelineStageFlags2::COPY,
                    vk::AccessFlags2::TRANSFER_WRITE,
                ),
            )?;
            let upload = self.device.submission.slots[index]
                .upload_buffer(slice)
                .handle;
            let regions = [vk::BufferCopy::default()
                .src_offset(slice.offset)
                .dst_offset(offset + (part * limit) as u64)
                .size(bytes.len() as u64)];
            self.barriers.emit(&mut self.device);
            unsafe {
                self.device.dispatch.cmd_copy_buffer(
                    self.device.submission.slots[index].command,
                    upload,
                    target,
                    &regions,
                );
            }
            self.buffer_mut(id)
                .validate_interval(offset + (part * limit) as u64, bytes.len() as u64);
            self.statistics.cpu_uploaded_bytes += bytes.len() as u64;
        }
        Ok(())
    }
    pub(super) fn copy_source(
        &mut self,
        source: Resource,
        offset: u64,
        destination: BufferId,
        destination_offset: u64,
        size: u64,
    ) -> Result<(), VulkanError> {
        self.check_buffer_range(destination, destination_offset, size)?;
        if let Resource::Buffer(source) = source {
            self.check_buffer_range(source, offset, size)?;
            // Merges copy distinct entries; self aliases are snapshotted by refresh_buffer.
            debug_assert_ne!(source, destination);
            let src = self.buffer_access_range(
                source,
                offset,
                size,
                AccessScope::new(
                    vk::PipelineStageFlags2::COPY,
                    vk::AccessFlags2::TRANSFER_READ,
                ),
            )?;
            let dst = self.buffer_access_range(
                destination,
                destination_offset,
                size,
                AccessScope::new(
                    vk::PipelineStageFlags2::COPY,
                    vk::AccessFlags2::TRANSFER_WRITE,
                ),
            )?;
            let index = self.device.begin_recording()?;
            let command = self.device.submission.slots[index].command;
            let work = self.device.work_ticket()?;
            self.contents_mut(Resource::Buffer(source)).last_use = Some(work);
            self.contents_mut(Resource::Buffer(destination)).last_use = Some(work);
            self.barriers.emit(&mut self.device);
            unsafe {
                self.device.dispatch.cmd_copy_buffer(
                    command,
                    src,
                    dst,
                    &[vk::BufferCopy::default()
                        .src_offset(offset)
                        .dst_offset(destination_offset)
                        .size(size)],
                );
            }
            self.statistics.gpu_copied_bytes += size;
        } else if let Resource::Image(source) = source {
            let pieces = self.image_pieces(source, offset, size)?;
            let dst = self.buffer_access_range(
                destination,
                destination_offset,
                size,
                AccessScope::new(
                    vk::PipelineStageFlags2::COPY,
                    vk::AccessFlags2::TRANSFER_WRITE,
                ),
            )?;
            if pieces
                .iter()
                .all(|piece| piece.direct(destination_offset + piece.logical - offset))
            {
                let (src, layout) =
                    self.image_regions_access(source, &pieces, vk::AccessFlags2::TRANSFER_READ)?;
                let regions: Vec<_> = pieces
                    .iter()
                    .map(|piece| piece.region(destination_offset + piece.logical - offset))
                    .collect();
                let index = self.device.begin_recording()?;
                self.barriers.emit(&mut self.device);
                unsafe {
                    self.device.dispatch.cmd_copy_image_to_buffer(
                        self.device.submission.slots[index].command,
                        src,
                        layout,
                        dst,
                        &regions,
                    );
                }
            } else {
                let (scratch, slices) =
                    self.image_pieces_buffer(source, &pieces, &[Resource::Buffer(destination)])?;
                let regions: Vec<_> = pieces
                    .iter()
                    .zip(slices)
                    .map(|(piece, start)| {
                        vk::BufferCopy::default()
                            .src_offset(start + piece.skip)
                            .dst_offset(destination_offset + piece.logical - offset)
                            .size(piece.length)
                    })
                    .collect();
                let index = self.device.begin_recording()?;
                self.barriers.emit(&mut self.device);
                unsafe {
                    self.device.dispatch.cmd_copy_buffer(
                        self.device.submission.slots[index].command,
                        scratch,
                        dst,
                        &regions,
                    );
                }
            }
            self.statistics.gpu_copied_bytes += size;
        }
        self.buffer_mut(destination)
            .validate_interval(destination_offset, size);
        Ok(())
    }
    /// Bound native command/temporary-allocation count when canonical tiling or
    /// packed aspects yield many small image-to-buffer intervals. Buffer copies
    /// scatter exact bytes; unowned padding and other aspect producers survive.
    /// Destinations must be ordered and disjoint, as in canonical restoration.
    pub(super) fn copy_image_buffer_regions(
        &mut self,
        source: ImageId,
        destination: BufferId,
        copies: &[(u64, u64, u64)],
    ) -> Result<(), VulkanError> {
        debug_assert!(copies.windows(2).all(|c| c[0].1 + c[0].2 <= c[1].1));
        if copies.len() == 1 {
            let (from, to, size) = copies[0];
            return self.copy_source(Resource::Image(source), from, destination, to, size);
        }
        if copies.is_empty() {
            return Ok(());
        }
        let begin = copies.iter().map(|c| c.1).min().unwrap();
        let end = copies.iter().map(|c| c.1 + c.2).max().unwrap();
        self.buffer_access_range(
            destination,
            begin,
            end - begin,
            AccessScope::new(
                vk::PipelineStageFlags2::COPY,
                vk::AccessFlags2::TRANSFER_WRITE,
            ),
        )?;
        let target = Arc::clone(&self.buffer(destination)?.native);
        self.constructing.push(Resource::Buffer(destination));
        let result = self.copy_image_bytes(source, &target, copies);
        self.unpin(Resource::Buffer(destination));
        result?;
        let written: Vec<_> = copies.iter().map(|&(_, to, size)| (to, size)).collect();
        self.buffer_mut(destination).validate_intervals(&written);
        Ok(())
    }
    /// Copy exact encoded intersections into a native transfer destination.
    /// Temporary layout conversions have no physical identity or cache entry.
    pub(super) fn copy_image_bytes(
        &mut self,
        source: ImageId,
        destination: &Arc<AllocatedBuffer>,
        copies: &[(u64, u64, u64)],
    ) -> Result<(), VulkanError> {
        let mut pieces = Vec::new();
        let mut offsets = Vec::new();
        for &(from, to, size) in copies {
            for piece in self.image_pieces(source, from, size)? {
                offsets.push(to + piece.logical - from);
                pieces.push(piece);
            }
        }
        if pieces.is_empty() {
            return Ok(());
        }
        let dst = destination.handle;
        if pieces.iter().zip(&offsets).all(|(p, &to)| p.direct(to)) {
            let (src, layout) =
                self.image_regions_access(source, &pieces, vk::AccessFlags2::TRANSFER_READ)?;
            let regions: Vec<_> = pieces
                .iter()
                .zip(&offsets)
                .map(|(p, &to)| p.region(to))
                .collect();
            let slot = self.device.begin_recording()?;
            let work = self.device.work_ticket()?;
            self.device.retire_buffer(Arc::clone(destination), work)?;
            self.barriers.emit(&mut self.device);
            unsafe {
                self.device.dispatch.cmd_copy_image_to_buffer(
                    self.device.submission.slots[slot].command,
                    src,
                    layout,
                    dst,
                    &regions,
                );
            }
        } else {
            let (scratch, slices) = self.image_pieces_buffer(source, &pieces, &[])?;
            let regions: Vec<_> = pieces
                .iter()
                .zip(&offsets)
                .zip(slices)
                .map(|((p, &to), from)| {
                    vk::BufferCopy::default()
                        .src_offset(from + p.skip)
                        .dst_offset(to)
                        .size(p.length)
                })
                .collect();
            let slot = self.device.begin_recording()?;
            let work = self.device.work_ticket()?;
            self.device.retire_buffer(Arc::clone(destination), work)?;
            self.barriers.emit(&mut self.device);
            unsafe {
                self.device.dispatch.cmd_copy_buffer(
                    self.device.submission.slots[slot].command,
                    scratch,
                    dst,
                    &regions,
                );
            }
        }
        self.statistics.gpu_copied_bytes += copies.iter().map(|&(_, _, size)| size).sum::<u64>();
        Ok(())
    }
    pub(super) fn read_buffer_native(
        &mut self,
        id: BufferId,
        offset: u64,
        size: u64,
    ) -> Result<Vec<u8>, VulkanError> {
        self.check_buffer_range(id, offset, size)?;
        let mut result =
            Vec::with_capacity(usize::try_from(size).map_err(|_| {
                VulkanError::InvalidResource("readback exceeds host address space")
            })?);
        let limit = self.device.submission.configuration.arena_bytes;
        let mut copied = 0;
        while copied < size {
            let length = (size - copied).min(limit);
            let ticket = self.download_buffer(id, offset + copied, length)?;
            self.flush()?;
            result.extend(self.device.readback(ticket)?);
            self.statistics.native_readback_bytes += length;
            copied += length;
        }
        Ok(result)
    }
    /// Native transfer batches name exact owned requests. Source fragments are
    /// gathered into one request buffer; publication keeps the original tokens.
    pub(super) fn record_access_readbacks(
        &mut self,
        request: nixe_memory::AccessRequestId,
        writes: &[nixe_gpu::authority::ResolvedBytes<Resource>],
    ) -> Result<(), VulkanError> {
        use super::cpu_readback::{Scatter, Transfer};
        let mut images = rustc_hash::FxHashMap::<ImageId, Vec<_>>::default();
        let mut buffers = Vec::new();
        let limit = self.device.submission.configuration.arena_bytes;
        for &write in writes {
            let producer = write.producer.ok_or(VulkanError::InvalidResource(
                "native readback has no device producer",
            ))?;
            match producer.resource {
                Resource::Buffer(id) => {
                    if let Some(known) = self.buffer(id)?.known.iter().find(|known| {
                        known.offset <= producer.native
                            && producer.native + write.bytes
                                <= known.offset + known.bytes.len() as u64
                    }) {
                        let literal = Arc::clone(&known.bytes);
                        let start = (producer.native - known.offset) as usize;
                        let end = start.checked_add(write.bytes as usize).ok_or(
                            VulkanError::InvalidResource("literal readback interval overflows"),
                        )?;
                        let bytes = literal.get(start..end).ok_or(VulkanError::InvalidResource(
                            "literal readback exceeds its producer",
                        ))?;
                        self.cpu_readbacks
                            .get_mut(&request)
                            .expect("registered access")
                            .bytes
                            [write.logical as usize..(write.logical + write.bytes) as usize]
                            .copy_from_slice(bytes);
                    } else {
                        let mut copied = 0;
                        while copied < write.bytes {
                            let length = (write.bytes - copied).min(limit);
                            buffers.push((
                                id,
                                producer.native + copied,
                                write.logical + copied,
                                length,
                            ));
                            copied += length;
                        }
                    }
                }
                Resource::Image(id) => images.entry(id).or_default().push(write),
            }
        }
        let mut next = 0;
        while next < buffers.len() {
            let first = next;
            let mut bytes = 0;
            while let Some(&(_, _, _, length)) = buffers.get(next) {
                if bytes + length > limit {
                    break;
                }
                bytes += length;
                next += 1;
            }
            let (index, slice, ticket) = self.reserve_readback(bytes, 4)?;
            let command = self.device.submission.slots[index].command;
            let target = self.device.submission.slots[index]
                .readback_buffer(slice)
                .handle;
            let mut copies = Vec::new();
            let mut packed = 0;
            for &(id, offset, destination, length) in &buffers[first..next] {
                let source = self.buffer_access_range(
                    id,
                    offset,
                    length,
                    AccessScope::new(
                        vk::PipelineStageFlags2::COPY,
                        vk::AccessFlags2::TRANSFER_READ,
                    ),
                )?;
                self.barriers.emit(&mut self.device);
                unsafe {
                    self.device.dispatch.cmd_copy_buffer(
                        command,
                        source,
                        target,
                        &[vk::BufferCopy::default()
                            .src_offset(offset)
                            .dst_offset(slice.offset + packed)
                            .size(length)],
                    );
                }
                copies.push(Scatter {
                    source: packed as usize,
                    destination: destination as usize,
                    length: length as usize,
                });
                packed += length;
            }
            host_read_barrier(&self.device, command, target, slice.offset, bytes);
            self.enqueue_cpu_transfer(
                request,
                Transfer {
                    ticket,
                    bytes,
                    copies,
                    windows: Vec::new(),
                },
            );
        }
        for (id, writes) in images {
            if self.record_cached_image_authorities(request, id, &writes)? {
                continue;
            }
            let pieces = self.image_read_pieces(id, &writes, limit)?;
            let mut ordered: Vec<_> = writes.iter().collect();
            ordered.sort_unstable_by_key(|w| w.producer.expect("device source").native);
            let mut next = 0;
            let mut active = Vec::new();
            let mut reads = Vec::new();
            for piece in pieces {
                let finish = piece.logical + piece.length;
                // The native pieces are ordered and disjoint. Each source
                // enters once and is visited only while it intersects a piece.
                // Repeated physical aliases can still have separate targets.
                while let Some(&write) = ordered.get(next) {
                    if write.producer.expect("device source").native >= finish {
                        break;
                    }
                    active.push(write);
                    next += 1;
                }
                active.retain(|w| {
                    w.producer.expect("device source").native + w.bytes > piece.logical
                });
                let mut copies = Vec::new();
                for w in &active {
                    let offset = w.producer.expect("device readback source").native;
                    let start = offset.max(piece.logical);
                    let end = (offset + w.bytes).min(piece.logical + piece.length);
                    if start < end {
                        copies.push(Scatter {
                            source: (start - piece.logical) as usize,
                            destination: (w.logical + start - offset) as usize,
                            length: (end - start) as usize,
                        });
                    }
                }
                reads.push((None, piece, copies));
            }
            self.record_image_reads(request, id, reads)?;
        }
        Ok(())
    }

    pub(super) fn record_image_reads(
        &mut self,
        request: nixe_memory::AccessRequestId,
        id: ImageId,
        reads: Vec<(
            Option<super::readback::WindowKey>,
            super::image::ImagePiece,
            Vec<super::cpu_readback::Scatter>,
        )>,
    ) -> Result<(), VulkanError> {
        use super::cpu_readback::Transfer;
        let limit = self.device.submission.configuration.arena_bytes;
        let multisample = self.image(id)?.description.samples != vk::SampleCountFlags::TYPE_1;
        let mut reads = reads.into_iter().peekable();
        while reads.peek().is_some() {
            let mut bytes = 0;
            let mut alignment = 4;
            let mut batch = Vec::new();
            while let Some((_, piece, _)) = reads.peek() {
                let align = u64::from(piece.plane.bytes_per_block).max(4);
                let offset = crate::queue::align_up(bytes, align).ok_or(
                    VulkanError::InvalidResource("image readback alignment overflow"),
                )?;
                if offset + piece.length > limit {
                    break;
                }
                bytes = offset + piece.length;
                alignment = alignment.max(align);
                batch.push((offset, reads.next().unwrap()));
            }
            if batch.is_empty() {
                return Err(VulkanError::InvalidResource(
                    "image readback block exceeds the staging arena",
                ));
            }
            let mut converted = Vec::new();
            if multisample {
                for (_, (_, piece, _)) in &batch {
                    converted.push(self.sample_readback_window(id, *piece)?);
                }
            }
            let (index, slice, ticket) = self.reserve_readback(bytes, alignment)?;
            let command = self.device.submission.slots[index].command;
            let buffer = self.device.submission.slots[index]
                .readback_buffer(slice)
                .handle;
            if multisample {
                for ((offset, (_, piece, _)), attachment) in batch.iter().zip(converted) {
                    let image = self.attachment_access(
                        attachment,
                        AccessScope::new(
                            vk::PipelineStageFlags2::COPY,
                            vk::AccessFlags2::TRANSFER_READ,
                        ),
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    )?;
                    self.barriers.emit(&mut self.device);
                    let region = vk::BufferImageCopy::default()
                        .buffer_offset(slice.offset + offset)
                        .image_subresource(
                            vk::ImageSubresourceLayers::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .layer_count(1),
                        )
                        .image_extent(vk::Extent3D {
                            width: piece.blocks,
                            height: piece.rows,
                            depth: 1,
                        });
                    unsafe {
                        self.device.dispatch.cmd_copy_image_to_buffer(
                            command,
                            image,
                            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                            buffer,
                            &[region],
                        );
                    }
                    self.release_attachment(attachment)?;
                }
            } else {
                let pieces: Vec<_> = batch.iter().map(|(_, (_, piece, _))| *piece).collect();
                let (image, layout) =
                    self.image_regions_access(id, &pieces, vk::AccessFlags2::TRANSFER_READ)?;
                let regions: Vec<_> = batch
                    .iter()
                    .map(|(offset, (_, piece, _))| piece.region(slice.offset + offset))
                    .collect();
                self.barriers.emit(&mut self.device);
                unsafe {
                    self.device
                        .dispatch
                        .cmd_copy_image_to_buffer(command, image, layout, buffer, &regions);
                }
            }
            host_read_barrier(&self.device, command, buffer, slice.offset, bytes);
            let mut copies = Vec::new();
            let mut windows = Vec::new();
            for (offset, (key, piece, scatter)) in batch {
                copies.extend(scatter.into_iter().map(|mut copy| {
                    copy.source += offset as usize;
                    copy
                }));
                if let Some(key) = key {
                    windows.push((key, offset as usize, piece.length as usize));
                }
            }
            self.enqueue_cpu_transfer(
                request,
                Transfer {
                    ticket,
                    bytes,
                    copies,
                    windows,
                },
            );
        }
        Ok(())
    }
    pub(super) fn download_buffer(
        &mut self,
        id: BufferId,
        offset: u64,
        size: u64,
    ) -> Result<ReadbackTicket, VulkanError> {
        let (index, slice, ticket) = self.reserve_readback(size, 4)?;
        let source = self.buffer_access_range(
            id,
            offset,
            size,
            AccessScope::new(
                vk::PipelineStageFlags2::COPY,
                vk::AccessFlags2::TRANSFER_READ,
            ),
        )?;
        let command = self.device.submission.slots[index].command;
        let target = self.device.submission.slots[index]
            .readback_buffer(slice)
            .handle;
        self.barriers.emit(&mut self.device);
        unsafe {
            self.device.dispatch.cmd_copy_buffer(
                command,
                source,
                target,
                &[vk::BufferCopy::default()
                    .src_offset(offset)
                    .dst_offset(slice.offset)
                    .size(size)],
            );
        }
        let barriers = [vk::BufferMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COPY)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::HOST)
            .dst_access_mask(vk::AccessFlags2::HOST_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(target)
            .offset(slice.offset)
            .size(size)];
        self.barriers.emit(&mut self.device);
        unsafe {
            self.device.dispatch.barrier(
                command,
                &vk::DependencyInfo::default().buffer_memory_barriers(&barriers),
            );
        }
        Ok(ticket)
    }
}

pub(super) use crate::transfer::host_read_barrier;
pub(super) fn scratch_read_barrier(
    device: &VulkanDevice,
    command: vk::CommandBuffer,
    buffer: vk::Buffer,
    size: u64,
) {
    let barriers = [vk::BufferMemoryBarrier2::default()
        .buffer(buffer)
        .size(size)
        .src_stage_mask(vk::PipelineStageFlags2::COPY)
        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::COPY)
        .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)];
    unsafe {
        device.dispatch.barrier(
            command,
            &vk::DependencyInfo::default().buffer_memory_barriers(&barriers),
        );
    }
}

impl VulkanResidency {
    pub(crate) fn copy_physical_buffers(
        &mut self,
        source: &Arc<PhysicalMapping>,
        destination: &Arc<PhysicalMapping>,
        regions: &[nixe_gpu::native::BufferCopyRegion],
    ) -> Result<(), VulkanError> {
        let retained = self.constructing.len();
        let result = self.copy_physical_buffers_inner(source, destination, regions);
        self.constructing.truncate(retained);
        result
    }
    fn copy_physical_buffers_inner(
        &mut self,
        source: &Arc<PhysicalMapping>,
        destination: &Arc<PhysicalMapping>,
        regions: &[nixe_gpu::native::BufferCopyRegion],
    ) -> Result<(), VulkanError> {
        if regions.is_empty() {
            return Ok(());
        }
        for region in regions {
            if region.bytes == 0
                || region
                    .source
                    .checked_add(region.bytes)
                    .is_none_or(|end| end > source.bytes())
                || region
                    .destination
                    .checked_add(region.bytes)
                    .is_none_or(|end| end > destination.bytes())
            {
                return Err(VulkanError::InvalidResource(
                    "native copy exceeds physical inputs",
                ));
            }
        }
        let (source, destination) = loop {
            let src = self.buffer_for_deferred_import(
                Arc::clone(source),
                vk::BufferUsageFlags::TRANSFER_SRC,
            )?;
            self.constructing.push(Resource::Buffer(src.buffer));
            let dst = self.buffer_for_deferred_import(
                Arc::clone(destination),
                vk::BufferUsageFlags::TRANSFER_DST,
            )?;
            if self.buffer(src.buffer).is_ok() {
                break (src, dst);
            }
        };
        let resources = [
            Resource::Buffer(source.buffer),
            Resource::Buffer(destination.buffer),
        ];
        self.constructing.extend(resources);
        let result = (|| {
            let mut read: Vec<_> = regions
                .iter()
                .map(|r| (source.offset + r.source, r.bytes))
                .collect();
            let mut written: Vec<_> = regions
                .iter()
                .map(|r| (destination.offset + r.destination, r.bytes))
                .collect();
            normalize(&mut read);
            normalize(&mut written);
            for &(offset, bytes) in &read {
                self.ensure_buffer_range(source.buffer, offset, bytes)?;
            }
            let mut next = 0;
            let overlapping = source.buffer == destination.buffer
                && read.iter().any(|&(offset, bytes)| {
                    while next < written.len() && written[next].0 + written[next].1 <= offset {
                        next += 1;
                    }
                    written
                        .get(next)
                        .is_some_and(|&(start, _)| start < offset + bytes)
                });
            let scratch_bytes = read.iter().map(|&(_, bytes)| bytes).sum();
            let scratch = if overlapping {
                Some(Arc::new(self.temporary_buffer(scratch_bytes, &resources)?))
            } else {
                None
            };
            for &(offset, bytes) in &read {
                self.buffer_access_range(
                    source.buffer,
                    offset,
                    bytes,
                    AccessScope::new(
                        vk::PipelineStageFlags2::COPY,
                        vk::AccessFlags2::TRANSFER_READ,
                    ),
                )?;
            }
            let source_handle = self.buffer(source.buffer)?.native.handle;
            let mut packed = Vec::new();
            if let Some(scratch) = &scratch {
                let mut offset = 0;
                for &(start, bytes) in &read {
                    packed.push(
                        vk::BufferCopy::default()
                            .src_offset(start)
                            .dst_offset(offset)
                            .size(bytes),
                    );
                    offset += bytes;
                }
                let slot = self.device.begin_recording()?;
                self.barriers.emit(&mut self.device);
                let command = self.device.submission.slots[slot].command;
                let work = self.device.work_ticket()?;
                self.device.retire_buffer(Arc::clone(scratch), work)?;
                unsafe {
                    self.device.dispatch.cmd_copy_buffer(
                        command,
                        source_handle,
                        scratch.handle,
                        &packed,
                    );
                    scratch_read_barrier(&self.device, command, scratch.handle, scratch_bytes);
                }
                self.statistics.gpu_copied_bytes += scratch_bytes;
            }
            let mut writers = Vec::with_capacity(written.len());
            for &(offset, bytes) in &written {
                writers.push(self.prepare_write(
                    Resource::Buffer(destination.buffer),
                    offset,
                    bytes,
                )?);
                self.buffer_access_range(
                    destination.buffer,
                    offset,
                    bytes,
                    AccessScope::new(
                        vk::PipelineStageFlags2::COPY,
                        vk::AccessFlags2::TRANSFER_WRITE,
                    ),
                )?;
            }
            let copies: Vec<_> = regions
                .iter()
                .map(|region| {
                    vk::BufferCopy::default()
                        .src_offset(if scratch.is_some() {
                            let start = source.offset + region.source;
                            let i = packed.partition_point(|copy| copy.src_offset <= start) - 1;
                            packed[i].dst_offset + start - packed[i].src_offset
                        } else {
                            source.offset + region.source
                        })
                        .dst_offset(destination.offset + region.destination)
                        .size(region.bytes)
                })
                .collect();
            let slot = self.device.begin_recording()?;
            self.barriers.emit(&mut self.device);
            let target = self.buffer(destination.buffer)?.native.handle;
            let command = self.device.submission.slots[slot].command;
            if let Some(scratch) = &scratch {
                let work = self.device.work_ticket()?;
                self.device.retire_buffer(Arc::clone(scratch), work)?;
            }
            let source_handle = scratch
                .as_ref()
                .map_or(source_handle, |scratch| scratch.handle);
            let work = self.device.work_ticket()?;
            self.contents_mut(Resource::Buffer(source.buffer)).last_use = Some(work);
            self.contents_mut(Resource::Buffer(destination.buffer))
                .last_use = Some(work);
            self.publish_effect(&writers, |device| unsafe {
                device
                    .dispatch
                    .cmd_copy_buffer(command, source_handle, target, &copies);
            })?;
            for (offset, bytes) in written {
                self.finish_buffer_write(destination.buffer, offset, bytes)?;
            }
            self.statistics.gpu_copied_bytes += regions.iter().map(|r| r.bytes).sum::<u64>();
            Ok(())
        })();
        for resource in resources.into_iter().rev() {
            self.unpin(resource);
        }
        result
    }
}
