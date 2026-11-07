//! Native transfers and exact canonical write footprints.
use super::commands::plane_range;
use super::*;
impl Residency<'_> {
    pub(super) fn transfer_planes(
        &self,
        source: ImageId,
        destination: ImageId,
        planes: &[(usize, usize)],
        source_rect: vk::Rect2D,
        destination_rect: vk::Rect2D,
    ) -> Result<smallvec::SmallVec<[(ImagePlane, ImagePlane); 2]>, VulkanError> {
        if planes.is_empty() {
            return Err(VulkanError::InvalidResource("image transfer has no planes"));
        }
        let mut selected = smallvec::SmallVec::new();
        for &(source_plane, destination_plane) in planes {
            let src = self.checked_image_rect(source, source_plane, source_rect)?;
            let dst = self.checked_image_rect(destination, destination_plane, destination_rect)?;
            if src.aspect != dst.aspect
                || selected
                    .iter()
                    .any(|(_, prior): &(ImagePlane, ImagePlane)| same_subresource(*prior, dst))
            {
                return Err(VulkanError::InvalidResource(
                    "image transfer requires matching aspects and distinct destination subresources",
                ));
            }
            selected.push((src, dst));
        }
        Ok(selected)
    }
    pub(super) fn image_planes_transfer(
        &mut self,
        id: ImageId,
        planes: impl IntoIterator<Item = ImagePlane>,
    ) -> Result<(vk::Image, vk::ImageSubresourceRange), VulkanError> {
        let mut range = vk::ImageSubresourceRange::default();
        for plane in planes {
            if range.level_count == 0 {
                range = plane_range(plane);
            } else {
                let mip = range.base_mip_level.min(plane.mip_level);
                let layer = range.base_array_layer.min(plane.array_layer);
                range.level_count =
                    (range.base_mip_level + range.level_count).max(plane.mip_level + 1) - mip;
                range.layer_count =
                    (range.base_array_layer + range.layer_count).max(plane.array_layer + 1) - layer;
                range.base_mip_level = mip;
                range.base_array_layer = layer;
                range.aspect_mask |= plane.aspect;
            }
        }
        self.transfer_image(id, range)
    }
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
            self.write_buffer(id, offset, &pattern[..prefix as usize])?;
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
            self.write_buffer(id, offset + prefix + middle, &bytes)?;
        }
        Ok(self.runtime.device.current_work())
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
        if self.buffer_cache.get(id).device_address.is_some() {
            self.ensure_buffer_range(id, offset, size)?;
        }
        self.prepare_write(Resource::Buffer(id), offset, size)?;
        let buffer = self.use_buffer(id, offset, size)?;
        self.runtime.barriers.emit(&mut self.runtime.device)?;
        {
            let device = &mut self.runtime.device;
            unsafe {
                device.dispatch.cmd_fill_buffer(buffer, offset, size, value);
            }
        };
        self.finish_buffer_write(id, offset, size)?;
        Ok(self.runtime.device.current_work())
    }
    pub(super) fn checked_image_rect(
        &self,
        id: ImageId,
        plane: usize,
        rect: vk::Rect2D,
    ) -> Result<ImagePlane, VulkanError> {
        let p = *self
            .texture_cache
            .get(id)
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
    pub fn clear_color_image(
        &mut self,
        id: ImageId,
        plane: usize,
        value: vk::ClearColorValue,
    ) -> Result<WorkTicket, VulkanError> {
        let p = *self
            .texture_cache
            .get(id)
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
        if self.texture_cache.get(id).description.image_type == vk::ImageType::TYPE_3D
            && self.texture_cache.get(id).description.extent.depth > 1
        {
            let view = self.attachment_view(id, plane)?;
            return self.clear_view_rect(
                view,
                plane,
                full_rect(p),
                vk::ClearValue { color: value },
            );
        }
        let rect = full_rect(p);
        let invalidate = self.is_full_image_clear(id, p, rect, p.aspect);
        self.prepare_image(id, true, invalidate)?;
        let (image, layout) = self.transfer_image(id, plane_range(p))?;
        self.runtime.barriers.emit(&mut self.runtime.device)?;
        {
            let device = &mut self.runtime.device;
            unsafe {
                device
                    .dispatch
                    .clear_color_image(image, layout, &value, &[plane_range(p)]);
            }
        };
        Ok(self.runtime.device.current_work())
    }
    pub fn clear_depth_stencil(
        &mut self,
        id: ImageId,
        plane: usize,
        depth: f32,
        stencil: u32,
    ) -> Result<WorkTicket, VulkanError> {
        let p = *self
            .texture_cache
            .get(id)
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
        let invalidate = self.is_full_image_clear(id, p, rect, p.aspect);
        self.prepare_image(id, true, invalidate)?;
        let (image, layout) = self.transfer_image(id, plane_range(p))?;
        self.runtime.barriers.emit(&mut self.runtime.device)?;
        {
            let device = &mut self.runtime.device;
            unsafe {
                device.dispatch.clear_depth_stencil_image(
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
        };
        Ok(self.runtime.device.current_work())
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
        self.prepare_image(source, false, false)?;
        let src = self.checked_image_rect(source, source_plane, source_rect)?;
        let rect = vk::Rect2D {
            offset: destination_offset,
            extent: source_rect.extent,
        };
        let dst = self.checked_image_rect(destination, destination_plane, rect)?;
        let s = self.texture_cache.get(source).description;
        let d = self.texture_cache.get(destination).description;
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
        self.prepare_image(destination, true, false)?;
        let (source, source_layout) = self.transfer_image(source, plane_range(src))?;
        let (destination_handle, destination_layout) =
            self.transfer_image(destination, plane_range(dst))?;
        let region = vk::ImageResolve::default()
            .src_subresource(src.layers())
            .dst_subresource(dst.layers())
            .src_offset(vk::Offset3D {
                x: source_rect.offset.x,
                y: source_rect.offset.y,
                z: src.depth_slice as i32,
            })
            .dst_offset(vk::Offset3D {
                x: rect.offset.x,
                y: rect.offset.y,
                z: dst.depth_slice as i32,
            })
            .extent(vk::Extent3D {
                width: rect.extent.width,
                height: rect.extent.height,
                depth: 1,
            });
        self.runtime.barriers.emit(&mut self.runtime.device)?;
        {
            let device = &mut self.runtime.device;
            unsafe {
                device.dispatch.resolve_image(
                    source,
                    source_layout,
                    destination_handle,
                    destination_layout,
                    &[region],
                );
            }
        };
        Ok(self.runtime.device.current_work())
    }
    /// Copy all selected aspects in one native command. Overlapping self-copies
    /// capture every source plane before publishing any destination plane.
    /// https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdCopyImage.html
    pub fn copy_image_regions(
        &mut self,
        source: ImageId,
        destination: ImageId,
        planes: &[(usize, usize)],
        source_rect: vk::Rect2D,
        destination_offset: vk::Offset2D,
    ) -> Result<WorkTicket, VulkanError> {
        self.prepare_image(source, false, false)?;
        let rect = vk::Rect2D {
            offset: destination_offset,
            extent: source_rect.extent,
        };
        let selected = self.transfer_planes(source, destination, planes, source_rect, rect)?;
        let source_description = self.texture_cache.get(source).description;
        let destination_description = self.texture_cache.get(destination).description;
        for &(src, dst) in &selected {
            if source_description.samples != destination_description.samples
                || !format::copy_compatible(
                    source_description.format,
                    destination_description.format,
                    src.aspect,
                )
                || src.block_extent != dst.block_extent
                || src.bytes_per_block != dst.bytes_per_block
            {
                return Err(VulkanError::InvalidResource(
                    "image copy requires compatible native aspect/block representations",
                ));
            }
        }
        let snapshot = source == destination
            && selected
                .iter()
                .any(|(src, _)| selected.iter().any(|(_, dst)| same_subresource(*src, *dst)));
        if snapshot && source_description.samples != vk::SampleCountFlags::TYPE_1 {
            return Err(VulkanError::InvalidResource(
                "overlapping multisample image copy requires a sample-preserving snapshot",
            ));
        }
        self.prepare_image(destination, true, false)?;

        let extent = vk::Extent3D {
            width: rect.extent.width,
            height: rect.extent.height,
            depth: 1,
        };
        if snapshot {
            let mut size = 0;
            let mut reads = smallvec::SmallVec::<[vk::BufferImageCopy; 2]>::new();
            let mut writes = smallvec::SmallVec::<[vk::BufferImageCopy; 2]>::new();
            for &(src, dst) in &selected {
                let offset = crate::queue::align_up(size, 4).ok_or(
                    VulkanError::InvalidResource("image snapshot alignment overflow"),
                )?;
                let row_bytes =
                    u64::from(source_rect.extent.width.div_ceil(src.block_extent.width))
                        * u64::from(src.bytes_per_block);
                let bytes = row_bytes
                    * u64::from(source_rect.extent.height.div_ceil(src.block_extent.height));
                size = offset
                    .checked_add(bytes)
                    .ok_or(VulkanError::InvalidResource("image snapshot size overflow"))?;
                reads.push(
                    vk::BufferImageCopy::default()
                        .buffer_offset(offset)
                        .image_subresource(src.layers())
                        .image_offset(vk::Offset3D {
                            x: source_rect.offset.x,
                            y: source_rect.offset.y,
                            z: src.depth_slice as i32,
                        })
                        .image_extent(extent),
                );
                writes.push(
                    vk::BufferImageCopy::default()
                        .buffer_offset(offset)
                        .image_subresource(dst.layers())
                        .image_offset(vk::Offset3D {
                            x: rect.offset.x,
                            y: rect.offset.y,
                            z: dst.depth_slice as i32,
                        })
                        .image_extent(extent),
                );
            }
            let scratch = Arc::new(self.temporary_buffer(size, &[Resource::Image(source)])?);
            let (handle, read_layout) =
                self.image_planes_transfer(source, selected.iter().map(|(src, _)| *src))?;
            self.runtime.barriers.emit(&mut self.runtime.device)?;
            let work = self.runtime.device.current_work();
            self.runtime
                .device
                .retire_buffer(Arc::clone(&scratch), work)?;
            unsafe {
                self.runtime.device.dispatch.copy_image_to_buffer(
                    handle,
                    read_layout,
                    scratch.handle,
                    &reads,
                );
            }

            let (_, write_layout) =
                self.image_planes_transfer(destination, selected.iter().map(|(_, dst)| *dst))?;
            self.runtime.barriers.emit(&mut self.runtime.device)?;
            {
                let device = &mut self.runtime.device;
                unsafe {
                    device.dispatch.copy_buffer_to_image(
                        scratch.handle,
                        handle,
                        write_layout,
                        &writes,
                    );
                }
            };
            let work = self.runtime.device.current_work();
            self.runtime.device.retire_buffer(scratch, work)?;
        } else {
            let (source_handle, source_layout) =
                self.image_planes_transfer(source, selected.iter().map(|(src, _)| *src))?;
            let (destination_handle, destination_layout) =
                self.image_planes_transfer(destination, selected.iter().map(|(_, dst)| *dst))?;
            let regions: smallvec::SmallVec<[_; 2]> = selected
                .iter()
                .map(|&(src, dst)| {
                    vk::ImageCopy::default()
                        .src_subresource(src.layers())
                        .dst_subresource(dst.layers())
                        .src_offset(vk::Offset3D {
                            x: source_rect.offset.x,
                            y: source_rect.offset.y,
                            z: src.depth_slice as i32,
                        })
                        .dst_offset(vk::Offset3D {
                            x: rect.offset.x,
                            y: rect.offset.y,
                            z: dst.depth_slice as i32,
                        })
                        .extent(extent)
                })
                .collect();
            self.runtime.barriers.emit(&mut self.runtime.device)?;
            {
                let device = &mut self.runtime.device;
                unsafe {
                    device.dispatch.copy_image(
                        source_handle,
                        source_layout,
                        destination_handle,
                        destination_layout,
                        &regions,
                    );
                }
            };
        }
        Ok(self.runtime.device.current_work())
    }
}
pub(super) fn same_subresource(a: ImagePlane, b: ImagePlane) -> bool {
    a.aspect.intersects(b.aspect)
        && a.mip_level == b.mip_level
        && a.array_layer == b.array_layer
        && a.depth_slice == b.depth_slice
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
