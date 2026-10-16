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
                self.device.owner.device.cmd_copy_buffer(
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
            let index = self.device.submission.begin()?;
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
                    self.device.owner.device.cmd_copy_buffer(
                        command,
                        src,
                        scratch.handle,
                        &[vk::BufferCopy::default().src_offset(offset).size(size)],
                    );
                    scratch_read_barrier(&self.device, command, scratch.handle, size);
                    self.device.owner.device.cmd_copy_buffer(
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
                    self.device.owner.device.cmd_copy_buffer(
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
                let index = self.device.submission.begin()?;
                self.barriers.emit(&mut self.device);
                unsafe {
                    self.device.owner.device.cmd_copy_image_to_buffer(
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
                let index = self.device.submission.begin()?;
                self.barriers.emit(&mut self.device);
                unsafe {
                    self.device.owner.device.cmd_copy_buffer(
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
    /// Read each image's transfer blocks in bounded batches. Byte authority is
    /// applied only when scattering into the preserved CPU page baseline.
    pub(super) fn read_authorities(
        &mut self,
        writes: &[Authority],
        target: &mut [u8],
    ) -> Result<(), VulkanError> {
        // A canonical page commonly has one image producer. Consume that
        // slice directly instead of allocating and hashing every tiled run.
        if let Some(first) = writes.first()
            && let Resource::Image(id) = first.source
            && writes.iter().all(|write| write.source == first.source)
        {
            self.read_image_authorities(id, writes, target)?;
            self.statistics.canonical_readback_bytes += writes
                .iter()
                .map(|write| write.end - write.start)
                .sum::<u64>();
            return Ok(());
        }
        let mut images = rustc_hash::FxHashMap::<ImageId, Vec<Authority>>::default();
        let mut buffers = Vec::new();
        let limit = self.device.submission.configuration.arena_bytes;
        for &write in writes {
            match write.source {
                Resource::Buffer(_) => {
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
                Resource::Image(id) => images.entry(id).or_default().push(write),
            }
        }
        let mut next = 0;
        while next < buffers.len() {
            let begin = next;
            let mut bytes = 0;
            let mut regions = Vec::new();
            while let Some(write) = buffers.get(next) {
                let length = write.end - write.start;
                if bytes + length > limit {
                    break;
                }
                let Resource::Buffer(id) = write.source else {
                    unreachable!()
                };
                let source = self.buffer_access_range(
                    id,
                    write.offset,
                    length,
                    AccessScope::new(
                        vk::PipelineStageFlags2::COPY,
                        vk::AccessFlags2::TRANSFER_READ,
                    ),
                )?;
                regions.push((
                    source,
                    vk::BufferCopy::default()
                        .src_offset(write.offset)
                        .dst_offset(bytes)
                        .size(length),
                ));
                bytes += length;
                next += 1;
            }
            let (index, slice, ticket) = self.device.submission.reserve_readback(bytes)?;
            let command = self.device.submission.slots[index].command;
            let buffer = self.device.submission.slots[index]
                .readback_buffer(slice)
                .handle;
            self.barriers.emit(&mut self.device);
            for (source, mut region) in regions {
                region.dst_offset += slice.offset;
                unsafe {
                    self.device
                        .owner
                        .device
                        .cmd_copy_buffer(command, source, buffer, &[region]);
                }
            }
            host_read_barrier(&self.device, command, buffer, slice.offset, bytes);
            self.flush()?;
            let data = self.device.readback(ticket)?;
            self.statistics.native_readback_bytes += bytes;
            let mut packed = 0;
            for write in &buffers[begin..next] {
                let length = (write.end - write.start) as usize;
                target[write.start as usize..write.end as usize]
                    .copy_from_slice(&data[packed..packed + length]);
                packed += length;
            }
        }
        for (id, writes) in images {
            self.read_image_authorities(id, &writes, target)?;
        }
        self.statistics.canonical_readback_bytes += writes
            .iter()
            .map(|write| write.end - write.start)
            .sum::<u64>();
        Ok(())
    }
    fn read_image_authorities(
        &mut self,
        id: ImageId,
        writes: &[Authority],
        target: &mut [u8],
    ) -> Result<(), VulkanError> {
        if self.cached_image_authorities(id, writes, target)? {
            return Ok(());
        }
        let limit = self.device.submission.configuration.arena_bytes;
        let mut writes = writes.to_vec();
        writes.sort_unstable_by_key(|write| write.offset);
        let pieces = self.image_read_pieces(id, &writes, limit)?;
        let mut next = 0;
        let mut first_write = 0;
        while next < pieces.len() {
            let begin = next;
            let mut bytes = 0;
            let mut alignment = 4;
            let mut offsets = Vec::new();
            while let Some(piece) = pieces.get(next) {
                let align = u64::from(piece.plane.bytes_per_block).max(4);
                let offset = crate::queue::align_up(bytes, align).ok_or(
                    VulkanError::InvalidResource("image readback alignment overflow"),
                )?;
                if offset + piece.length > limit {
                    break;
                }
                offsets.push(offset);
                bytes = offset + piece.length;
                alignment = alignment.max(align);
                next += 1;
            }
            let batch = &pieces[begin..next];
            let (image, layout) =
                self.image_regions_access(id, batch, vk::AccessFlags2::TRANSFER_READ)?;
            let (index, slice, ticket) = self
                .device
                .submission
                .reserve_readback_aligned(bytes, alignment)?;
            let command = self.device.submission.slots[index].command;
            let buffer = self.device.submission.slots[index]
                .readback_buffer(slice)
                .handle;
            let regions: Vec<_> = batch
                .iter()
                .zip(&offsets)
                .map(|(piece, offset)| piece.region(slice.offset + offset))
                .collect();
            self.barriers.emit(&mut self.device);
            unsafe {
                self.device
                    .owner
                    .device
                    .cmd_copy_image_to_buffer(command, image, layout, buffer, &regions);
            }
            host_read_barrier(&self.device, command, buffer, slice.offset, bytes);
            self.flush()?;
            let data = self.device.readback(ticket)?;
            self.statistics.native_readback_bytes += bytes;
            for (piece, offset) in batch.iter().zip(offsets) {
                let end = piece.logical + piece.length;
                while first_write < writes.len()
                    && writes[first_write].offset + writes[first_write].end
                        - writes[first_write].start
                        <= piece.logical
                {
                    first_write += 1;
                }
                for write in &writes[first_write..] {
                    if write.offset >= end {
                        break;
                    }
                    let begin = write.offset.max(piece.logical);
                    let stop = (write.offset + write.end - write.start).min(end);
                    if begin < stop {
                        let destination = (write.start + begin - write.offset) as usize;
                        let source = (offset + begin - piece.logical) as usize;
                        let length = (stop - begin) as usize;
                        target[destination..destination + length]
                            .copy_from_slice(&data[source..source + length]);
                    }
                }
            }
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
            self.device.owner.device.cmd_copy_buffer(
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
