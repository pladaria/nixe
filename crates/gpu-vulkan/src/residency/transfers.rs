//! Native transfers and exact canonical write footprints.
use super::commands::plane_range;
use super::*;
impl VulkanResidency {
    /// Exact byte-range fill. Native aligned words cover the interior; at most
    /// six staging bytes preserve boundary bytes without a read-modify-write
    /// shader, CPU readback, or an oversized zero upload.
    pub fn fill_buffer_range(
        &mut self,
        id: BufferId,
        offset: u64,
        size: u64,
        value: u32,
    ) -> Result<WorkTicket, VulkanError> {
        self.ensure_buffer(id)?;
        self.check_buffer_range(id, offset, size)?;
        let pattern = value.to_le_bytes();
        let prefix = ((4 - offset % 4) % 4).min(size);
        if prefix != 0 {
            let mut copied = 0;
            while copied < prefix {
                let upload = self.write_buffer(
                    id,
                    offset + copied,
                    &pattern[copied as usize..prefix as usize],
                )?;
                copied += upload.consumed as u64;
            }
        }
        let middle = (size - prefix) / 4 * 4;
        if middle != 0 {
            self.fill_buffer(
                id,
                offset + prefix,
                middle,
                value.rotate_right((prefix % 4) as u32 * 8),
            )?;
        }
        let tail = size - prefix - middle;
        if tail != 0 {
            let bytes: Vec<_> = (0..tail)
                .map(|i| pattern[((prefix + middle + i) % 4) as usize])
                .collect();
            let mut copied = 0;
            while copied < tail {
                let upload = self.write_buffer(
                    id,
                    offset + prefix + middle + copied,
                    &bytes[copied as usize..],
                )?;
                copied += upload.consumed as u64;
            }
        }
        self.device.work_ticket()
    }
    pub fn fill_buffer(
        &mut self,
        id: BufferId,
        offset: u64,
        size: u64,
        value: u32,
    ) -> Result<WorkTicket, VulkanError> {
        self.ensure_buffer(id)?;
        self.check_buffer_range(id, offset, size)?;
        if !offset.is_multiple_of(4) || !size.is_multiple_of(4) {
            return Err(VulkanError::InvalidResource(
                "native fill requires a four-byte aligned interval; byte masks require a conversion helper",
            ));
        }
        self.prepare_write(Resource::Buffer(id), offset, size)?;
        let buffer = self.buffer_access_range(
            id,
            offset,
            size,
            AccessScope::new(
                vk::PipelineStageFlags2::CLEAR,
                vk::AccessFlags2::TRANSFER_WRITE,
            ),
        )?;
        let index = self.device.submission.begin()?;
        self.barriers.emit(&self.device);
        unsafe {
            self.device.owner.device.cmd_fill_buffer(
                self.device.submission.slots[index].command,
                buffer,
                offset,
                size,
                value,
            );
        }
        self.record_authority(Resource::Buffer(id), offset, size)?;
        self.device.work_ticket()
    }
    pub(super) fn checked_image_rect(
        &self,
        id: ImageId,
        plane: usize,
        rect: vk::Rect2D,
    ) -> Result<ImagePlane, VulkanError> {
        let p = *self
            .image(id)?
            .planes
            .get(plane)
            .ok_or(VulkanError::InvalidResource("missing image plane"))?;
        if rect.offset.x < 0
            || rect.offset.y < 0
            || rect.extent.width == 0
            || rect.extent.height == 0
            || (rect.offset.x as u32)
                .checked_add(rect.extent.width)
                .is_none_or(|end| end > p.extent.width)
            || (rect.offset.y as u32)
                .checked_add(rect.extent.height)
                .is_none_or(|end| end > p.extent.height)
        {
            return Err(VulkanError::InvalidResource(
                "image region exceeds native plane",
            ));
        }
        if !(rect.offset.x as u32).is_multiple_of(p.block_extent.width)
            || !(rect.offset.y as u32).is_multiple_of(p.block_extent.height)
            || !rect.extent.width.is_multiple_of(p.block_extent.width)
                && rect.offset.x as u32 + rect.extent.width != p.extent.width
            || !rect.extent.height.is_multiple_of(p.block_extent.height)
                && rect.offset.y as u32 + rect.extent.height != p.extent.height
        {
            return Err(VulkanError::InvalidResource(
                "native image copy region splits a compressed block; conversion recording is required",
            ));
        }
        Ok(p)
    }
    pub(super) fn image_rect_bytes(p: ImagePlane, rect: vk::Rect2D) -> Vec<(u64, u64)> {
        let start = rect.offset.y as u32 / p.block_extent.height;
        let rows = rect.extent.height.div_ceil(p.block_extent.height);
        let x =
            u64::from(rect.offset.x as u32 / p.block_extent.width) * u64::from(p.bytes_per_block);
        let size = u64::from(rect.extent.width.div_ceil(p.block_extent.width))
            * u64::from(p.bytes_per_block);
        let mut intervals = Vec::with_capacity(rows as usize);
        for row in start..start + rows {
            intervals.push((p.offset + u64::from(row) * p.row_pitch + x, size));
        }
        normalize(&mut intervals);
        intervals
    }
    pub(super) fn canonical_image_rect_bytes(
        &self,
        id: ImageId,
        p: ImagePlane,
        rect: vk::Rect2D,
    ) -> Result<Vec<(u64, u64)>, VulkanError> {
        let p = self.image(id)?.storage_plane(p);
        let rect = self.image(id)?.storage_rect(rect);
        if p.aspect == vk::ImageAspectFlags::DEPTH
            && matches!(
                self.image(id)?.description.format,
                vk::Format::D24_UNORM_S8_UINT | vk::Format::X8_D24_UNORM_PACK32
            )
        {
            // Vulkan's D24 transfer element has an unused high byte. It is not
            // a depth producer and must not overwrite a canonical padding or
            // independently authored stencil byte with unspecified copy data.
            // https://docs.vulkan.org/refpages/latest/refpages/source/VkFormat.html
            let mut ranges = Vec::with_capacity((rect.extent.width * rect.extent.height) as usize);
            for y in rect.offset.y as u32..rect.offset.y as u32 + rect.extent.height {
                for x in rect.offset.x as u32..rect.offset.x as u32 + rect.extent.width {
                    ranges.push((p.offset + u64::from(y) * p.row_pitch + u64::from(x) * 4, 3));
                }
            }
            Ok(ranges)
        } else {
            Ok(Self::image_rect_bytes(p, rect))
        }
    }
    pub(super) fn prepare_image_rect(
        &mut self,
        id: ImageId,
        p: ImagePlane,
        rect: vk::Rect2D,
    ) -> Result<(), VulkanError> {
        if self.image(id)?.decoded.is_some() {
            return Err(VulkanError::InvalidResource(
                "decoded compressed textures are read-only; writes require the encoded canonical representation",
            ));
        }
        if self.image(id)?.leases != 0 {
            return Err(VulkanError::InvalidResource(
                "image write would overwrite a leased version",
            ));
        }
        self.observe_image_cpu_writes(id)?;
        self.prepare_image_write(id, p, rect)
    }
    pub(super) fn finish_image_rect(
        &mut self,
        id: ImageId,
        p: ImagePlane,
        rect: vk::Rect2D,
    ) -> Result<(), VulkanError> {
        self.publish_image_write(id, p, rect)?;
        let work = self.device.work_ticket()?;
        self.define_image_rect(id, p, rect);
        let entry = self.images[id.slot].entry.as_mut().expect("image");
        entry.producer = Some(work);
        entry.version = entry
            .version
            .checked_add(1)
            .ok_or(VulkanError::InvalidResource("image version exhausted"))?;
        Ok(())
    }
    pub(super) fn define_image_rect(&mut self, id: ImageId, p: ImagePlane, rect: vk::Rect2D) {
        let entry = self.images[id.slot].entry.as_mut().expect("image");
        let plane = entry
            .planes
            .iter()
            .position(|plane| *plane == p)
            .expect("plane identity");
        if entry.discarded.is_empty() {
            return;
        }
        let mut remaining = Vec::new();
        for (index, undefined) in entry.discarded.drain(..) {
            if index != plane {
                remaining.push((index, undefined));
            } else {
                remaining.extend(
                    subtract_rect(undefined, rect)
                        .into_iter()
                        .map(|rect| (index, rect)),
                );
            }
        }
        entry.discarded = remaining;
    }
    pub(super) fn mark_image_discarded(&mut self, id: ImageId, p: ImagePlane, rect: vk::Rect2D) {
        let entry = self.images[id.slot].entry.as_mut().expect("image");
        let plane = entry
            .planes
            .iter()
            .position(|plane| *plane == p)
            .expect("plane");
        entry.discarded.push((plane, rect));
    }
    pub(super) fn require_image_initialized(
        &self,
        id: ImageId,
        p: ImagePlane,
        rect: vk::Rect2D,
    ) -> Result<(), VulkanError> {
        let entry = self.image(id)?;
        if entry
            .discarded
            .iter()
            .any(|&(plane, undefined)| entry.planes[plane] == p && intersects(undefined, rect))
        {
            log::debug!(
                "undefined native image {:?} plane {:?} requested {:?}; discarded={:?}",
                id,
                p,
                rect,
                self.image(id)?.discarded
            );
            return Err(VulkanError::InvalidResource(
                "image read/attachment LOAD consumes undefined or discarded texels; an actual producer is required",
            ));
        }
        Ok(())
    }
    pub fn clear_color_image(
        &mut self,
        id: ImageId,
        plane: usize,
        value: vk::ClearColorValue,
    ) -> Result<WorkTicket, VulkanError> {
        let p = *self
            .image(id)?
            .planes
            .get(plane)
            .ok_or(VulkanError::InvalidResource("missing clear plane"))?;
        if p.aspect != vk::ImageAspectFlags::COLOR
            || p.block_extent
                != (vk::Extent2D {
                    width: 1,
                    height: 1,
                })
        {
            return Err(VulkanError::InvalidResource(
                "native color clear requires an uncompressed color plane",
            ));
        }
        let rect = full_rect(p);
        self.prepare_image_rect(id, p, rect)?;
        let (image, layout) = self.image_access_subresources(
            id,
            plane_range(p),
            AccessScope::new(
                vk::PipelineStageFlags2::CLEAR,
                vk::AccessFlags2::TRANSFER_WRITE,
            ),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        )?;
        self.barriers.emit(&self.device);
        let index = self.device.submission.begin()?;
        unsafe {
            self.device.owner.device.cmd_clear_color_image(
                self.device.submission.slots[index].command,
                image,
                layout,
                &value,
                &[plane_range(p)],
            );
        }
        self.finish_image_rect(id, p, rect)?;
        self.device.work_ticket()
    }
    pub fn clear_depth_stencil(
        &mut self,
        id: ImageId,
        plane: usize,
        depth: f32,
        stencil: u32,
    ) -> Result<WorkTicket, VulkanError> {
        let p = *self
            .image(id)?
            .planes
            .get(plane)
            .ok_or(VulkanError::InvalidResource("missing clear aspect"))?;
        if !p
            .aspect
            .intersects(vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL)
        {
            return Err(VulkanError::InvalidResource(
                "depth/stencil clear consumes a color plane",
            ));
        }
        if p.aspect == vk::ImageAspectFlags::DEPTH
            && (!depth.is_finite() || !(0.0..=1.0).contains(&depth))
        {
            return Err(VulkanError::InvalidResource(
                "depth clear outside [0,1] requires an enabled unrestricted-depth-range lowering",
            ));
        }
        let rect = full_rect(p);
        self.prepare_image_rect(id, p, rect)?;
        let (image, layout) = self.image_access_subresources(
            id,
            plane_range(p),
            AccessScope::new(
                vk::PipelineStageFlags2::CLEAR,
                vk::AccessFlags2::TRANSFER_WRITE,
            ),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        )?;
        self.barriers.emit(&self.device);
        let index = self.device.submission.begin()?;
        unsafe {
            self.device.owner.device.cmd_clear_depth_stencil_image(
                self.device.submission.slots[index].command,
                image,
                layout,
                // Vulkan validates depth even when only stencil is selected.
                // https://docs.vulkan.org/spec/latest/chapters/clears.html#VUID-VkClearDepthStencilValue-depth-00022
                &vk::ClearDepthStencilValue {
                    depth: if p.aspect == vk::ImageAspectFlags::DEPTH {
                        depth
                    } else {
                        0.0
                    },
                    stencil,
                },
                &[plane_range(p)],
            );
        }
        self.finish_image_rect(id, p, rect)?;
        self.device.work_ticket()
    }
    /// Resolve native samples directly into canonical single-sample storage.
    /// https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdResolveImage.html
    pub fn resolve_image(
        &mut self,
        source: ImageId,
        source_plane: usize,
        source_rect: vk::Rect2D,
        destination: ImageId,
        destination_plane: usize,
        destination_offset: vk::Offset2D,
    ) -> Result<WorkTicket, VulkanError> {
        self.ensure_image(source)?;
        let src = self.checked_image_rect(source, source_plane, source_rect)?;
        let rect = vk::Rect2D {
            offset: destination_offset,
            extent: source_rect.extent,
        };
        let dst = self.checked_image_rect(destination, destination_plane, rect)?;
        let s = self.image(source)?.description;
        let d = self.image(destination)?.description;
        if s.samples == vk::SampleCountFlags::TYPE_1
            || d.samples != vk::SampleCountFlags::TYPE_1
            || s.format != d.format
            || src.aspect != vk::ImageAspectFlags::COLOR
            || dst.aspect != vk::ImageAspectFlags::COLOR
        {
            return Err(VulkanError::InvalidResource(
                "color resolve requires matching formats and a multisample source",
            ));
        }
        self.ensure_image_initialized(source, src, source_rect)?;
        if rect != full_rect(dst) {
            self.ensure_image(destination)?;
        }
        self.prepare_image_rect(destination, dst, rect)?;
        let (source, source_layout) = self.image_access_subresources(
            source,
            plane_range(src),
            AccessScope::new(
                vk::PipelineStageFlags2::RESOLVE,
                vk::AccessFlags2::TRANSFER_READ,
            ),
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        )?;
        let (destination_handle, destination_layout) = self.image_access_subresources(
            destination,
            plane_range(dst),
            AccessScope::new(
                vk::PipelineStageFlags2::RESOLVE,
                vk::AccessFlags2::TRANSFER_WRITE,
            ),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        )?;
        let region = vk::ImageResolve::default()
            .src_subresource(src.layers())
            .dst_subresource(dst.layers())
            .src_offset(vk::Offset3D {
                x: source_rect.offset.x,
                y: source_rect.offset.y,
                z: 0,
            })
            .dst_offset(vk::Offset3D {
                x: rect.offset.x,
                y: rect.offset.y,
                z: 0,
            })
            .extent(vk::Extent3D {
                width: rect.extent.width,
                height: rect.extent.height,
                depth: 1,
            });
        self.barriers.emit(&self.device);
        let slot = self.device.submission.begin()?;
        unsafe {
            self.device.owner.device.cmd_resolve_image(
                self.device.submission.slots[slot].command,
                source,
                source_layout,
                destination_handle,
                destination_layout,
                &[region],
            );
        }
        self.finish_image_rect(destination, dst, rect)?;
        self.device.work_ticket()
    }
    pub fn copy_image_region(
        &mut self,
        source: ImageId,
        source_plane: usize,
        source_rect: vk::Rect2D,
        destination: ImageId,
        destination_plane: usize,
        destination_offset: vk::Offset2D,
    ) -> Result<WorkTicket, VulkanError> {
        self.ensure_image(source)?;
        self.ensure_image(destination)?;
        let src = self.checked_image_rect(source, source_plane, source_rect)?;
        let rect = vk::Rect2D {
            offset: destination_offset,
            extent: source_rect.extent,
        };
        let dst = self.checked_image_rect(destination, destination_plane, rect)?;
        self.ensure_image_initialized(source, src, source_rect)?;
        if src.aspect != dst.aspect
            || src.block_extent != dst.block_extent
            || src.bytes_per_block != dst.bytes_per_block
        {
            return Err(VulkanError::InvalidResource(
                "image copy requires compatible native aspect/block representations",
            ));
        }
        if source == destination
            && src.mip_level == dst.mip_level
            && src.array_layer == dst.array_layer
        {
            self.prepare_image_rect(destination, dst, rect)?;
            let row_bytes = u64::from(source_rect.extent.width.div_ceil(src.block_extent.width))
                * u64::from(src.bytes_per_block);
            let size =
                row_bytes * u64::from(source_rect.extent.height.div_ceil(src.block_extent.height));
            let scratch = self.temporary_buffer(size, &[Resource::Image(source)])?;
            let (handle, read_layout) = self.image_access_subresources(
                source,
                plane_range(src),
                AccessScope::new(
                    vk::PipelineStageFlags2::COPY,
                    vk::AccessFlags2::TRANSFER_READ,
                ),
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            )?;
            self.barriers.emit(&self.device);
            let index = self.device.submission.begin()?;
            let command = self.device.submission.slots[index].command;
            let extent = vk::Extent3D {
                width: rect.extent.width,
                height: rect.extent.height,
                depth: 1,
            };
            let region = vk::BufferImageCopy::default()
                .image_subresource(src.layers())
                .image_offset(vk::Offset3D {
                    x: source_rect.offset.x,
                    y: source_rect.offset.y,
                    z: 0,
                })
                .image_extent(extent);
            unsafe {
                self.device.owner.device.cmd_copy_image_to_buffer(
                    command,
                    handle,
                    read_layout,
                    scratch.handle,
                    &[region],
                );
            }
            super::recording::scratch_read_barrier(&self.device, command, scratch.handle, size);
            let (_, write_layout) = self.image_access_subresources(
                destination,
                plane_range(dst),
                AccessScope::new(
                    vk::PipelineStageFlags2::COPY,
                    vk::AccessFlags2::TRANSFER_WRITE,
                ),
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            )?;
            self.barriers.emit(&self.device);
            let region = vk::BufferImageCopy::default()
                .image_subresource(dst.layers())
                .image_offset(vk::Offset3D {
                    x: rect.offset.x,
                    y: rect.offset.y,
                    z: 0,
                })
                .image_extent(extent);
            unsafe {
                self.device.owner.device.cmd_copy_buffer_to_image(
                    command,
                    scratch.handle,
                    handle,
                    write_layout,
                    &[region],
                );
            }
            let work = self.device.work_ticket()?;
            self.device.retire_buffer(scratch, work)?;
            self.finish_image_rect(destination, dst, rect)?;
            return Ok(work);
        }
        self.prepare_image_rect(destination, dst, rect)?;
        let (source_handle, source_layout) = self.image_access_subresources(
            source,
            plane_range(src),
            AccessScope::new(
                vk::PipelineStageFlags2::COPY,
                vk::AccessFlags2::TRANSFER_READ,
            ),
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        )?;
        let (destination_handle, destination_layout) = self.image_access_subresources(
            destination,
            plane_range(dst),
            AccessScope::new(
                vk::PipelineStageFlags2::COPY,
                vk::AccessFlags2::TRANSFER_WRITE,
            ),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        )?;
        let region = vk::ImageCopy::default()
            .src_subresource(src.layers())
            .dst_subresource(dst.layers())
            .src_offset(vk::Offset3D {
                x: source_rect.offset.x,
                y: source_rect.offset.y,
                z: 0,
            })
            .dst_offset(vk::Offset3D {
                x: rect.offset.x,
                y: rect.offset.y,
                z: 0,
            })
            .extent(vk::Extent3D {
                width: rect.extent.width,
                height: rect.extent.height,
                depth: 1,
            });
        self.barriers.emit(&self.device);
        let index = self.device.submission.begin()?;
        unsafe {
            self.device.owner.device.cmd_copy_image(
                self.device.submission.slots[index].command,
                source_handle,
                source_layout,
                destination_handle,
                destination_layout,
                &[region],
            );
        }
        self.finish_image_rect(destination, dst, rect)?;
        self.device.work_ticket()
    }
}
pub(super) fn full_rect(p: ImagePlane) -> vk::Rect2D {
    vk::Rect2D {
        offset: vk::Offset2D::default(),
        extent: vk::Extent2D {
            width: p.extent.width,
            height: p.extent.height,
        },
    }
}
pub(super) fn intersects(a: vk::Rect2D, b: vk::Rect2D) -> bool {
    a.offset.x < b.offset.x + b.extent.width as i32
        && b.offset.x < a.offset.x + a.extent.width as i32
        && a.offset.y < b.offset.y + b.extent.height as i32
        && b.offset.y < a.offset.y + a.extent.height as i32
}
pub(super) fn subtract_rect(a: vk::Rect2D, b: vk::Rect2D) -> Vec<vk::Rect2D> {
    if !intersects(a, b) {
        return vec![a];
    }
    let left = a.offset.x.max(b.offset.x);
    let right = (a.offset.x + a.extent.width as i32).min(b.offset.x + b.extent.width as i32);
    let top = a.offset.y.max(b.offset.y);
    let bottom = (a.offset.y + a.extent.height as i32).min(b.offset.y + b.extent.height as i32);
    [
        (
            a.offset.x,
            a.offset.y,
            a.extent.width as i32,
            top - a.offset.y,
        ),
        (
            a.offset.x,
            bottom,
            a.extent.width as i32,
            a.offset.y + a.extent.height as i32 - bottom,
        ),
        (a.offset.x, top, left - a.offset.x, bottom - top),
        (
            right,
            top,
            a.offset.x + a.extent.width as i32 - right,
            bottom - top,
        ),
    ]
    .into_iter()
    .filter(|&(_, _, width, height)| width > 0 && height > 0)
    .map(|(x, y, width, height)| vk::Rect2D {
        offset: vk::Offset2D { x, y },
        extent: vk::Extent2D {
            width: width as u32,
            height: height as u32,
        },
    })
    .collect()
}
