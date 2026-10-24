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
        self.check_buffer_range(id, offset, size)?;
        if !offset.is_multiple_of(4) || !size.is_multiple_of(4) {
            return Err(VulkanError::InvalidResource(
                "native fill requires a four-byte aligned interval; byte masks require a conversion helper",
            ));
        }
        let writer = self.prepare_write(Resource::Buffer(id), offset, size)?;
        let buffer = self.buffer_access_range(
            id,
            offset,
            size,
            AccessScope::new(
                vk::PipelineStageFlags2::CLEAR,
                vk::AccessFlags2::TRANSFER_WRITE,
            ),
        )?;
        let index = self.device.begin_recording()?;
        self.barriers.emit(&mut self.device);
        self.publish_effect(&[writer], |device| unsafe {
            device.dispatch.cmd_fill_buffer(
                device.submission.slots[index].command,
                buffer,
                offset,
                size,
                value,
            );
        })?;
        self.finish_buffer_write(id, offset, size)?;
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
    pub(super) fn prepare_image_rect(
        &mut self,
        id: ImageId,
        p: ImagePlane,
        rect: vk::Rect2D,
    ) -> Result<WriterId, VulkanError> {
        if self.image(id)?.decoded.is_some() {
            return Err(VulkanError::InvalidResource(
                "decoded compressed textures are read-only; writes require the encoded canonical representation",
            ));
        }
        // Exact transfers only author this rectangle. Conditional shader writes
        // prepare initialized contents in program/attachment consumption instead.
        self.invalidate_cpu_resources(&[])?;
        self.prepare_image_write(id, p, rect)
    }
    pub(super) fn finish_image_rect(
        &mut self,
        id: ImageId,
        p: ImagePlane,
        rect: vk::Rect2D,
    ) -> Result<(), VulkanError> {
        let full = self.image(id)?.planes.len() == 1 && rect == full_rect(p);
        if full {
            if self
                .image(id)?
                .contents
                .geometry
                .iter()
                .all(|f| f.mapping().is_injective())
                && self
                    .image(id)?
                    .contents
                    .device
                    .iter()
                    .any(ResourceObservation::is_invalid)
            {
                let geometry = self.image(id)?.contents.geometry.clone();
                let observed = geometry
                    .into_iter()
                    .map(|f| self.authority.observe_native(Resource::Image(id), f))
                    .collect::<Result<Vec<_>, _>>()?;
                self.images[id.slot]
                    .entry
                    .as_mut()
                    .expect("image")
                    .contents
                    .device = observed;
            }
            self.images[id.slot]
                .entry
                .as_mut()
                .expect("image")
                .synchronized = true;
        }
        // Bound cached replay plans after their commands have been published.
        // Authority retains any released writer that still owns newest bytes.
        if self.image(id)?.write_plans.len() > 64 {
            let plans =
                std::mem::take(&mut self.images[id.slot].entry.as_mut().unwrap().write_plans);
            for (_, writer) in plans {
                self.authority.release_writer(writer)?;
            }
        }
        self.track_image_cpu(id)?;
        self.define_image_rect(id, p, rect);
        self.advance_image_producer(id)
    }
    /// The native content version changes for every accepted write, including
    /// writes whose physical coverage is already current in a rendering scope.
    pub(super) fn advance_image_producer(&mut self, id: ImageId) -> Result<(), VulkanError> {
        let work = self.device.work_ticket()?;
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
        let contains = |outer: vk::Rect2D, inner: vk::Rect2D| {
            outer.offset.x <= inner.offset.x
                && outer.offset.y <= inner.offset.y
                && outer.offset.x as i64 + i64::from(outer.extent.width)
                    >= inner.offset.x as i64 + i64::from(inner.extent.width)
                && outer.offset.y as i64 + i64::from(outer.extent.height)
                    >= inner.offset.y as i64 + i64::from(inner.extent.height)
        };
        if entry
            .discarded
            .iter()
            .any(|&(p, old)| p == plane && contains(old, rect))
        {
            return;
        }
        entry
            .discarded
            .retain(|&(p, old)| p != plane || !contains(rect, old));
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
        let writer = self.prepare_image_rect(id, p, rect)?;
        let (image, layout) = self.image_access_subresources(
            id,
            plane_range(p),
            AccessScope::new(
                vk::PipelineStageFlags2::CLEAR,
                vk::AccessFlags2::TRANSFER_WRITE,
            ),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        )?;
        self.barriers.emit(&mut self.device);
        let index = self.device.begin_recording()?;
        self.publish_effect(&[writer], |device| unsafe {
            device.dispatch.cmd_clear_color_image(
                device.submission.slots[index].command,
                image,
                layout,
                &value,
                &[plane_range(p)],
            );
        })?;
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
        let writer = self.prepare_image_rect(id, p, rect)?;
        let (image, layout) = self.image_access_subresources(
            id,
            plane_range(p),
            AccessScope::new(
                vk::PipelineStageFlags2::CLEAR,
                vk::AccessFlags2::TRANSFER_WRITE,
            ),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        )?;
        self.barriers.emit(&mut self.device);
        let index = self.device.begin_recording()?;
        self.publish_effect(&[writer], |device| unsafe {
            device.dispatch.cmd_clear_depth_stencil_image(
                device.submission.slots[index].command,
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
        })?;
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
        let writer = self.prepare_image_rect(destination, dst, rect)?;
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
        self.barriers.emit(&mut self.device);
        let slot = self.device.begin_recording()?;
        self.publish_effect(&[writer], |device| unsafe {
            device.dispatch.cmd_resolve_image(
                device.submission.slots[slot].command,
                source,
                source_layout,
                destination_handle,
                destination_layout,
                &[region],
            );
        })?;
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
        let src = self.checked_image_rect(source, source_plane, source_rect)?;
        let rect = vk::Rect2D {
            offset: destination_offset,
            extent: source_rect.extent,
        };
        let dst = self.checked_image_rect(destination, destination_plane, rect)?;
        self.ensure_image_initialized(source, src, source_rect)?;
        if self.image(source)?.description.samples != self.image(destination)?.description.samples
            || !format::copy_compatible(
                self.image(source)?.description.format,
                self.image(destination)?.description.format,
                src.aspect,
            )
            || src.aspect != dst.aspect
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
            if self.image(source)?.description.samples != vk::SampleCountFlags::TYPE_1 {
                return Err(VulkanError::InvalidResource(
                    "overlapping multisample image copy requires a sample-preserving snapshot",
                ));
            }
            let writer = self.prepare_image_rect(destination, dst, rect)?;
            let row_bytes = u64::from(source_rect.extent.width.div_ceil(src.block_extent.width))
                * u64::from(src.bytes_per_block);
            let size =
                row_bytes * u64::from(source_rect.extent.height.div_ceil(src.block_extent.height));
            let scratch = Arc::new(self.temporary_buffer(size, &[Resource::Image(source)])?);
            let (handle, read_layout) = self.image_access_subresources(
                source,
                plane_range(src),
                AccessScope::new(
                    vk::PipelineStageFlags2::COPY,
                    vk::AccessFlags2::TRANSFER_READ,
                ),
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            )?;
            self.barriers.emit(&mut self.device);
            let index = self.device.begin_recording()?;
            let command = self.device.submission.slots[index].command;
            let work = self.device.work_ticket()?;
            self.device.retire_buffer(Arc::clone(&scratch), work)?;
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
                self.device.dispatch.cmd_copy_image_to_buffer(
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
            self.barriers.emit(&mut self.device);
            let region = vk::BufferImageCopy::default()
                .image_subresource(dst.layers())
                .image_offset(vk::Offset3D {
                    x: rect.offset.x,
                    y: rect.offset.y,
                    z: 0,
                })
                .image_extent(extent);
            self.publish_effect(&[writer], |device| unsafe {
                device.dispatch.cmd_copy_buffer_to_image(
                    command,
                    scratch.handle,
                    handle,
                    write_layout,
                    &[region],
                );
            })?;
            let work = self.device.work_ticket()?;
            self.device.retire_buffer(scratch, work)?;
            self.finish_image_rect(destination, dst, rect)?;
            return Ok(work);
        }
        let writer = self.prepare_image_rect(destination, dst, rect)?;
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
        self.barriers.emit(&mut self.device);
        let index = self.device.begin_recording()?;
        self.publish_effect(&[writer], |device| unsafe {
            device.dispatch.cmd_copy_image(
                device.submission.slots[index].command,
                source_handle,
                source_layout,
                destination_handle,
                destination_layout,
                &[region],
            );
        })?;
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
pub(super) fn contains_rect(outer: vk::Rect2D, inner: vk::Rect2D) -> bool {
    let end = |r: vk::Rect2D| {
        [
            i64::from(r.offset.x) + i64::from(r.extent.width),
            i64::from(r.offset.y) + i64::from(r.extent.height),
        ]
    };
    outer.offset.x <= inner.offset.x
        && outer.offset.y <= inner.offset.y
        && end(outer)[0] >= end(inner)[0]
        && end(outer)[1] >= end(inner)[1]
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
