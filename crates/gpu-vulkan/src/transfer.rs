//! Minimal real copy recorder sharing the owner timeline, arenas and retirement.
use crate::{AllocationMode, MemoryLocation, ReadbackTicket, VulkanDevice, VulkanError};
use ash::vk;
impl VulkanDevice {
    /// Bring-up proof: upload -> device buffer -> readback, synchronized by sync2
    /// and a real timeline signal/wait. This is not guest batch execution.
    pub fn enqueue_buffer_readback(&mut self, bytes: &[u8]) -> Result<ReadbackTicket, VulkanError> {
        if bytes.is_empty() {
            return Err(VulkanError::InvalidResource("empty bring-up transfer"));
        }
        let device_buffer = self.allocate_buffer(
            bytes.len() as u64,
            vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
            AllocationMode::Managed,
            "transfer device buffer",
        )?;
        let (index, source_slice, destination_slice, ticket) =
            self.submission.reserve_pair(&mut self.dispatch, bytes)?;
        let command = self.submission.slots[index].command;
        let slot = &self.submission.slots[index];
        let upload = slot.upload_buffer(source_slice).handle;
        let download = slot.readback_buffer(destination_slice).handle;
        let middle = device_buffer.handle;
        let region = [vk::BufferCopy::default()
            .src_offset(source_slice.offset)
            .size(bytes.len() as u64)];
        unsafe {
            self.dispatch
                .cmd_copy_buffer(command, upload, middle, &region);
        }
        let barrier = [vk::BufferMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COPY)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COPY)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(middle)
            .size(vk::WHOLE_SIZE)];
        unsafe {
            self.dispatch.barrier(
                command,
                &vk::DependencyInfo::default().buffer_memory_barriers(&barrier),
            );
            let region = [vk::BufferCopy::default()
                .dst_offset(destination_slice.offset)
                .size(bytes.len() as u64)];
            self.dispatch
                .cmd_copy_buffer(command, middle, download, &region);
        }
        host_read_barrier(
            self,
            command,
            download,
            destination_slice.offset,
            destination_slice.size,
        );
        self.retire_buffer(device_buffer, ticket.work())?;
        Ok(ticket)
    }

    /// Bring-up proof for a real optimal-tiled image and dedicated memory.
    pub fn enqueue_image_readback(
        &mut self,
        width: u32,
        height: u32,
        bytes: &[u8],
        mode: AllocationMode,
    ) -> Result<ReadbackTicket, VulkanError> {
        let expected = u64::from(width)
            .checked_mul(u64::from(height))
            .and_then(|v| v.checked_mul(4));
        if expected != Some(bytes.len() as u64) || bytes.is_empty() {
            return Err(VulkanError::InvalidResource(
                "RGBA8 transfer size does not match image extent",
            ));
        }
        let image = self.allocate_image_2d(
            width,
            height,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST,
            mode,
            "transfer image",
        )?;
        let (index, source_slice, destination_slice, ticket) =
            self.submission.reserve_pair(&mut self.dispatch, bytes)?;
        let command = self.submission.slots[index].command;
        let slot = &self.submission.slots[index];
        let upload = slot.upload_buffer(source_slice).handle;
        let download = slot.readback_buffer(destination_slice).handle;
        let image_handle = image.handle;
        let range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);
        let (destination, source) = if self.profile().unified_image_layouts {
            (vk::ImageLayout::GENERAL, vk::ImageLayout::GENERAL)
        } else {
            (
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            )
        };
        let barrier = [vk::ImageMemoryBarrier2::default()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(destination)
            .dst_stage_mask(vk::PipelineStageFlags2::COPY)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image_handle)
            .subresource_range(range)];
        let region = [vk::BufferImageCopy::default()
            .buffer_offset(source_slice.offset)
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1),
            )
            .image_extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })];
        unsafe {
            self.dispatch.barrier(
                command,
                &vk::DependencyInfo::default().image_memory_barriers(&barrier),
            );
            self.dispatch.cmd_copy_buffer_to_image(
                command,
                upload,
                image_handle,
                destination,
                &region,
            );
        }
        let barrier = [vk::ImageMemoryBarrier2::default()
            .old_layout(destination)
            .new_layout(source)
            .src_stage_mask(vk::PipelineStageFlags2::COPY)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COPY)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image_handle)
            .subresource_range(range)];
        unsafe {
            self.dispatch.barrier(
                command,
                &vk::DependencyInfo::default().image_memory_barriers(&barrier),
            );
            let region = [vk::BufferImageCopy {
                buffer_offset: destination_slice.offset,
                ..region[0]
            }];
            self.dispatch.cmd_copy_image_to_buffer(
                command,
                image_handle,
                source,
                download,
                &region,
            );
        }
        host_read_barrier(
            self,
            command,
            download,
            destination_slice.offset,
            destination_slice.size,
        );
        self.retire_image(image, ticket.work())?;
        Ok(ticket)
    }
    pub fn transfer_round_trip(&mut self, bytes: &[u8]) -> Result<Vec<u8>, VulkanError> {
        let ticket = self.enqueue_buffer_readback(bytes)?;
        // This diagnostic also verifies the normalized dynamic-rendering entry
        // points. Guest transfer recording adds no empty rendering scopes.
        let index = self.submission.recording.expect("diagnostic recording");
        let command = self.submission.slots[index].command;
        let info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: vk::Extent2D {
                    width: 1,
                    height: 1,
                },
            })
            .layer_count(1);
        unsafe {
            self.dispatch.begin_rendering(command, &info);
            self.dispatch.end_rendering(command);
        }
        self.readback(ticket)
    }
    pub fn image_round_trip(
        &mut self,
        width: u32,
        height: u32,
        bytes: &[u8],
        mode: AllocationMode,
    ) -> Result<Vec<u8>, VulkanError> {
        let ticket = self.enqueue_image_readback(width, height, bytes, mode)?;
        self.readback(ticket)
    }
}
pub(crate) fn host_read_barrier(
    device: &mut VulkanDevice,
    command: vk::CommandBuffer,
    buffer: vk::Buffer,
    offset: u64,
    size: u64,
) {
    // Publish the download to the host and order the next GPU write when this
    // completed arena slice is recycled. A counter query alone adds no barrier.
    let barrier = [vk::BufferMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::COPY)
        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::HOST | vk::PipelineStageFlags2::COPY)
        .dst_access_mask(vk::AccessFlags2::HOST_READ | vk::AccessFlags2::TRANSFER_WRITE)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .buffer(buffer)
        .offset(offset)
        .size(size)];
    // SAFETY: submitted resources/recording buffer belong to this device; the
    // caller waits the timeline before mapping readback bytes.
    unsafe {
        device.dispatch.barrier(
            command,
            &vk::DependencyInfo::default().buffer_memory_barriers(&barrier),
        );
    }
}
