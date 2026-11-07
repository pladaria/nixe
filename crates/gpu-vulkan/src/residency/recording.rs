//! Native copy operations own their dependencies; cache use tracks lifetime,
//! byte provenance and legal upload reordering, independently of those barriers.
use super::*;
use crate::ReadbackTicket;
impl Residency<'_> {
    pub(super) fn record_buffer_upload(
        &mut self,
        id: BufferId,
        slice: crate::queue::Slice,
        mut regions: smallvec::SmallVec<[vk::BufferCopy; 8]>,
    ) -> Result<(), VulkanError> {
        let entry = self.buffer_cache.get(id);
        let reorder = self.runtime.device.submission.staging.is_stream(&slice)
            && regions
                .iter()
                .all(|r| !entry.used.used(r.dst_offset, r.size));
        if reorder {
            let work = self.runtime.device.current_work();
            let entry = self.buffer_cache.get_mut(id);
            entry.contents.last_use = Some(work);
            self.touch(Resource::Buffer(id));
        } else {
            for region in &regions {
                self.use_buffer(id, region.dst_offset, region.size)?;
            }
            self.runtime.barriers.emit(&mut self.runtime.device)?;
        }
        let source = self.runtime.device.submission.staging_buffer(&slice).handle;
        let destination = self.buffer_cache.get(id).native.handle;
        for region in &mut regions {
            region.src_offset += slice.offset;
        }
        {
            let device = &mut self.runtime.device;
            unsafe {
                if reorder {
                    device
                        .dispatch
                        .cmd_upload_buffer(source, destination, &regions);
                } else {
                    device
                        .dispatch
                        .cmd_copy_buffer(source, destination, &regions, true);
                }
            }
        };
        self.buffer_cache.get_mut(id).write_tick = self.runtime.device.current_work().tick();
        for region in regions {
            self.buffer_cache
                .get_mut(id)
                .invalid
                .subtract(region.dst_offset, region.size);
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
            || offset
                .checked_add(size)
                .is_none_or(|end| end > self.buffer_cache.get(id).native.size())
        {
            return Err(VulkanError::InvalidResource(
                "buffer transfer exceeds resident interval",
            ));
        }
        Ok(())
    }
    pub(super) fn upload_buffer(
        &mut self,
        id: BufferId,
        offset: u64,
        bytes: &[u8],
    ) -> Result<(), VulkanError> {
        self.check_buffer_range(id, offset, bytes.len() as u64)?;
        let slice = self
            .runtime
            .device
            .submission
            .reserve_upload_aligned(bytes, 4)?;
        self.record_buffer_upload(
            id,
            slice,
            smallvec::smallvec![
                vk::BufferCopy::default()
                    .dst_offset(offset)
                    .size(bytes.len() as u64),
            ],
        )?;
        self.runtime.statistics.cpu_uploaded_bytes += bytes.len() as u64;
        Ok(())
    }
    pub(super) fn copy_buffer(
        &mut self,
        source: BufferId,
        offset: u64,
        destination: BufferId,
        destination_offset: u64,
        size: u64,
    ) -> Result<(), VulkanError> {
        self.check_buffer_range(destination, destination_offset, size)?;
        self.check_buffer_range(source, offset, size)?;
        self.check_buffer_range(source, offset, size)?;
        // Merges copy distinct entries; self aliases are snapshotted by refresh_buffer.
        debug_assert_ne!(source, destination);
        let src = self.use_buffer(source, offset, size)?;
        let dst = self.use_buffer(destination, destination_offset, size)?;

        let work = self.runtime.device.current_work();
        self.contents_mut(Resource::Buffer(source)).last_use = Some(work);
        self.contents_mut(Resource::Buffer(destination)).last_use = Some(work);
        self.runtime.barriers.emit(&mut self.runtime.device)?;
        unsafe {
            self.runtime.device.dispatch.cmd_copy_buffer(
                src,
                dst,
                &[vk::BufferCopy::default()
                    .src_offset(offset)
                    .dst_offset(destination_offset)
                    .size(size)],
                true,
            );
        }
        self.runtime.statistics.gpu_copied_bytes += size;
        self.buffer_cache.get_mut(destination).write_tick =
            self.runtime.device.current_work().tick();

        Ok(())
    }
    pub(super) fn read_buffer_native(
        &mut self,
        id: BufferId,
        offset: u64,
        size: u64,
    ) -> Result<Vec<u8>, VulkanError> {
        self.check_buffer_range(id, offset, size)?;
        let ticket = self.download_buffer(id, offset, size)?;
        self.flush()?;
        self.runtime
            .device
            .wait_for_work(ticket.work(), std::time::Duration::MAX)?;
        let bytes = ticket.completed()?.to_vec();
        self.runtime.statistics.native_readback_bytes += size;
        Ok(bytes)
    }
    pub(super) fn download_buffer(
        &mut self,
        id: BufferId,
        offset: u64,
        size: u64,
    ) -> Result<ReadbackTicket, VulkanError> {
        let (slice, ticket) = self
            .runtime
            .device
            .submission
            .reserve_readback_aligned(size, 4)?;
        let source = self.use_buffer(id, offset, size)?;

        let target = self.runtime.device.submission.staging_buffer(&slice).handle;
        self.runtime.barriers.emit(&mut self.runtime.device)?;
        unsafe {
            self.runtime.device.dispatch.cmd_copy_buffer(
                source,
                target,
                &[vk::BufferCopy::default()
                    .src_offset(offset)
                    .dst_offset(slice.offset)
                    .size(size)],
                true,
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
        self.runtime.barriers.emit(&mut self.runtime.device)?;
        unsafe {
            self.runtime
                .device
                .dispatch
                .barrier(&vk::DependencyInfo::default().buffer_memory_barriers(&barriers));
        }
        Ok(ticket)
    }
}
