//! Native copies consume the shared ordered access state before their commands.
//! Canonical producer tracking remains independent of host barrier emission.
use super::*;
use crate::ReadbackTicket;
impl VulkanResidency {
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
        loop {
            match self
                .device
                .submission
                .reserve_upload_aligned(bytes, alignment)
            {
                Ok(reservation) => return Ok(reservation),
                Err(error @ VulkanError::QueuePressure { pending, .. }) => {
                    // Submit an oversized resource's recorded prefix, then wait
                    // only for work that can release bounded staging capacity.
                    let flushed = self.flush()?;
                    let Some(work) = pending.or(flushed) else {
                        return Err(error);
                    };
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
            self.contents_mut(Resource::Buffer(id))
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
            if source == destination && offset == destination_offset {
                self.contents_mut(Resource::Buffer(destination))
                    .validate_interval(destination_offset, size);
                return Ok(());
            }
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
            if source == destination {
                // Vulkan disallows overlapping source/destination memory. A
                // GPU temporary preserves ordered repeated-canonical offsets.
                let scratch = self.temporary_buffer(
                    size,
                    &[Resource::Buffer(source), Resource::Buffer(destination)],
                )?;
                self.barriers.emit(&mut self.device);
                unsafe {
                    self.device.dispatch.cmd_copy_buffer(
                        command,
                        src,
                        scratch.handle,
                        &[vk::BufferCopy::default().src_offset(offset).size(size)],
                    );
                    scratch_read_barrier(&self.device, command, scratch.handle, size);
                    self.device.dispatch.cmd_copy_buffer(
                        command,
                        scratch.handle,
                        dst,
                        &[vk::BufferCopy::default()
                            .dst_offset(destination_offset)
                            .size(size)],
                    );
                }
                let work = self.device.work_ticket()?;
                self.device.retire_buffer(scratch, work)?;
            } else {
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
        self.contents_mut(Resource::Buffer(destination))
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
        let mut pieces = Vec::new();
        let mut offsets = Vec::new();
        for &(from, to, size) in copies {
            self.check_buffer_range(destination, to, size)?;
            for piece in self.image_pieces(source, from, size)? {
                offsets.push(to + piece.logical - from);
                pieces.push(piece);
            }
        }
        let Some(&start) = offsets.first() else {
            return Ok(());
        };
        let end = offsets.last().unwrap() + pieces.last().unwrap().length;
        let dst = self.buffer_access_range(
            destination,
            start,
            end - start,
            AccessScope::new(
                vk::PipelineStageFlags2::COPY,
                vk::AccessFlags2::TRANSFER_WRITE,
            ),
        )?;
        if pieces.iter().zip(&offsets).all(|(p, &to)| p.direct(to)) {
            let (src, layout) =
                self.image_regions_access(source, &pieces, vk::AccessFlags2::TRANSFER_READ)?;
            let regions: Vec<_> = pieces
                .iter()
                .zip(&offsets)
                .map(|(p, &to)| p.region(to))
                .collect();
            let slot = self.device.begin_recording()?;
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
            let (scratch, slices) =
                self.image_pieces_buffer(source, &pieces, &[Resource::Buffer(destination)])?;
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
        let written: Vec<_> = copies.iter().map(|&(_, to, size)| (to, size)).collect();
        self.contents_mut(Resource::Buffer(destination))
            .validate_intervals(&written);
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
            let src = self.buffer_access_range(
                id,
                offset + copied,
                length,
                AccessScope::new(
                    vk::PipelineStageFlags2::COPY,
                    vk::AccessFlags2::TRANSFER_READ,
                ),
            )?;
            let ticket = self.download_buffer(src, offset + copied, length)?;
            self.flush()?;
            result.extend(self.device.readback(ticket)?);
            self.statistics.native_readback_bytes += length;
            copied += length;
        }
        Ok(result)
    }
    /// Record one demanded page's compatible transfers without consuming them.
    pub(super) fn record_cpu_authorities(
        &mut self,
        request: CpuVisibilityRequest,
        writes: &[Authority],
    ) -> Result<(), VulkanError> {
        use super::cpu_readback::{Scatter, Transfer};
        let mut images = rustc_hash::FxHashMap::<ImageId, Vec<Authority>>::default();
        let mut buffers = Vec::new();
        let limit = self.device.submission.configuration.arena_bytes;
        for &write in writes {
            match write.source {
                Resource::Buffer(id) => {
                    if let Some(known) = &self.buffer(id)?.known {
                        nixe_gpu::trace::detail_event("readback_kind", 0, write.start, write.end);
                        let bytes = Arc::clone(&known.bytes);
                        self.cpu_readbacks.get_mut(&request).unwrap().bytes
                            [write.start as usize..write.end as usize]
                            .copy_from_slice(
                                &bytes[write.offset as usize
                                    ..(write.offset + write.end - write.start) as usize],
                            );
                    } else {
                        nixe_gpu::trace::detail_event("readback_kind", 1, write.start, write.end);
                        let mut copied = 0;
                        while copied < write.end - write.start {
                            let length = (write.end - write.start - copied).min(limit);
                            buffers.push(Authority {
                                start: write.start + copied,
                                end: write.start + copied + length,
                                offset: write.offset + copied,
                                ..write
                            });
                            copied += length;
                        }
                    }
                }
                Resource::Image(id) => {
                    nixe_gpu::trace::detail_event("readback_kind", 2, write.start, write.end);
                    images.entry(id).or_default().push(write);
                }
            }
        }
        let mut next = 0;
        while next < buffers.len() {
            let first = next;
            let mut bytes = 0;
            while let Some(w) = buffers.get(next) {
                if bytes + w.end - w.start > limit {
                    break;
                }
                bytes += w.end - w.start;
                next += 1;
            }
            let (index, slice, ticket) = self.reserve_cpu_readback(bytes, 4)?;
            let command = self.device.submission.slots[index].command;
            let target = self.device.submission.slots[index]
                .readback_buffer(slice)
                .handle;
            let mut copies = Vec::new();
            let mut packed = 0;
            for w in &buffers[first..next] {
                let Resource::Buffer(id) = w.source else {
                    unreachable!()
                };
                let length = w.end - w.start;
                let source = self.buffer_access_range(
                    id,
                    w.offset,
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
                            .src_offset(w.offset)
                            .dst_offset(slice.offset + packed)
                            .size(length)],
                    );
                }
                copies.push(Scatter {
                    source: packed as usize,
                    destination: w.start as usize,
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
            let mut reads = Vec::new();
            for piece in pieces {
                let mut copies = Vec::new();
                for w in &writes {
                    let start = w.offset.max(piece.logical);
                    let end = (w.offset + w.end - w.start).min(piece.logical + piece.length);
                    if start < end {
                        copies.push(Scatter {
                            source: (start - piece.logical) as usize,
                            destination: (w.start + start - w.offset) as usize,
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
        request: CpuVisibilityRequest,
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
            let mut converted = Vec::new();
            if multisample {
                for (_, (_, piece, _)) in &batch {
                    converted.push(self.sample_readback_window(id, *piece)?);
                }
            }
            let (index, slice, ticket) = self.reserve_cpu_readback(bytes, alignment)?;
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
    fn download_buffer(
        &mut self,
        source: vk::Buffer,
        offset: u64,
        size: u64,
    ) -> Result<ReadbackTicket, VulkanError> {
        let (index, slice, ticket) = self.device.submission.reserve_readback(size)?;
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

pub(super) fn host_read_barrier(
    device: &VulkanDevice,
    command: vk::CommandBuffer,
    buffer: vk::Buffer,
    offset: u64,
    size: u64,
) {
    let barriers = [vk::BufferMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::COPY)
        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::HOST)
        .dst_access_mask(vk::AccessFlags2::HOST_READ)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .buffer(buffer)
        .offset(offset)
        .size(size)];
    unsafe {
        device.dispatch.barrier(
            command,
            &vk::DependencyInfo::default().buffer_memory_barriers(&barriers),
        );
    }
}
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
