//! DMA uses registered cache entries before SafeReadCachedWrite fallback.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/buffer_cache/buffer_cache.h#L200-L275
use super::*;
use nixe_gpu_maxwell::{MaxwellGpuMemory, MaxwellMemoryCopyLayout, MaxwellMemoryCopyOperation};

fn failure(error: impl std::fmt::Display) -> VulkanError {
    VulkanError::Compilation(error.to_string().into())
}
impl Residency<'_> {
    pub(crate) fn accelerate_dma(
        &mut self,
        operation: &MaxwellMemoryCopyOperation,
        space: &MaxwellGpuMemory,
    ) -> Result<bool, VulkanError> {
        if operation.has_remap() {
            self.runtime.statistics.dma_cpu_fallbacks += 1;
            return Ok(false);
        }
        let accelerated = match (operation.source_layout(), operation.destination_layout()) {
            (MaxwellMemoryCopyLayout::Pitch { .. }, MaxwellMemoryCopyLayout::Pitch { .. })
                if operation.height() == 1 =>
            {
                let source = space.address(operation.source_address()).map_err(failure)?;
                let destination = space
                    .address(operation.destination_address())
                    .map_err(failure)?;
                let source = space.device_address(source).map_err(failure)?;
                let destination = space.device_address(destination).map_err(failure)?;
                self.dma_buffer_copy(source, destination, u64::from(operation.width()))
            }
            (
                MaxwellMemoryCopyLayout::Pitch { .. },
                MaxwellMemoryCopyLayout::BlockLinear { .. },
            ) => self.dma_buffer_image(operation, space, true),
            (
                MaxwellMemoryCopyLayout::BlockLinear { .. },
                MaxwellMemoryCopyLayout::Pitch { .. },
            ) => self.dma_buffer_image(operation, space, false),
            _ => Ok(false),
        }?;
        self.runtime.statistics.dma_cpu_fallbacks += u64::from(!accelerated);
        Ok(accelerated)
    }
    fn dma_buffer_copy(
        &mut self,
        source: u64,
        destination: u64,
        bytes: u64,
    ) -> Result<bool, VulkanError> {
        const PAGE: u64 = 1 << 16;
        let registered = |address: u64| {
            (address / PAGE..(address + bytes).div_ceil(PAGE))
                .any(|page| self.buffer_cache.buffer_index.at(page * PAGE).is_some())
        };
        if !registered(source) && !registered(destination) {
            return Ok(false);
        }
        self.buffer_cache.clear_download(destination, bytes);
        let retained = self.runtime.constructing.len();
        let result = self.record_dma_buffer_copy(source, destination, bytes);
        self.runtime.constructing.truncate(retained);
        result.map(|_| {
            self.runtime.statistics.dma_buffer_copies += 1;
            true
        })
    }
    fn record_dma_buffer_copy(
        &mut self,
        source_address: u64,
        destination_address: u64,
        bytes: u64,
    ) -> Result<(), VulkanError> {
        let (source, destination) = loop {
            let src = self.device_buffer(
                source_address,
                bytes,
                vk::BufferUsageFlags::TRANSFER_SRC,
                false,
                false,
            )?;
            self.runtime.constructing.push(Resource::Buffer(src.buffer));
            let dst = self.device_buffer(
                destination_address,
                bytes,
                vk::BufferUsageFlags::TRANSFER_DST,
                false,
                false,
            )?;
            if self.buffer_cache.buffers.contains(src.buffer) {
                break (src, dst);
            }
        };
        self.runtime
            .constructing
            .push(Resource::Buffer(destination.buffer));
        self.ensure_buffer_range(source.buffer, source.offset, bytes)?;
        self.ensure_buffer_range(destination.buffer, destination.offset, bytes)?;
        // Save source intervals before subtracting an overlapping destination.
        let modified: smallvec::SmallVec<[(u64, u64); 4]> = self
            .buffer_cache
            .gpu_modified_ranges
            .intersections(source_address, bytes)
            .map(|(address, size)| (destination_address + address - source_address, size))
            .collect();
        self.buffer_cache
            .gpu_modified_ranges
            .subtract(destination_address, bytes);
        for &(address, size) in &modified {
            self.buffer_cache.gpu_modified_ranges.insert(address, size);
            self.buffer_cache
                .uncommitted_gpu_modified_ranges
                .insert(address, size);
        }
        let source_handle = self.use_buffer(source.buffer, source.offset, bytes)?;
        let overlap = source.buffer == destination.buffer
            && source.offset < destination.offset + bytes
            && destination.offset < source.offset + bytes;
        let scratch = if overlap {
            let scratch =
                Arc::new(self.temporary_buffer(bytes, &[Resource::Buffer(source.buffer)])?);
            self.runtime.barriers.emit(&mut self.runtime.device)?;
            unsafe {
                self.runtime.device.dispatch.cmd_copy_buffer(
                    source_handle,
                    scratch.handle,
                    &[vk::BufferCopy::default()
                        .src_offset(source.offset)
                        .size(bytes)],
                    true,
                );
            }
            Some(scratch)
        } else {
            None
        };
        let destination_handle = self.use_buffer(destination.buffer, destination.offset, bytes)?;
        self.runtime.barriers.emit(&mut self.runtime.device)?;
        let work = self.runtime.device.current_work();
        unsafe {
            self.runtime.device.dispatch.cmd_copy_buffer(
                scratch.as_ref().map_or(source_handle, |s| s.handle),
                destination_handle,
                &[vk::BufferCopy::default()
                    .src_offset(if overlap { 0 } else { source.offset })
                    .dst_offset(destination.offset)
                    .size(bytes)],
                true,
            );
        }
        if let Some(scratch) = scratch {
            self.runtime.device.retire_buffer(scratch, work)?;
        }
        self.contents_mut(Resource::Buffer(source.buffer)).last_use = Some(work);
        let entry = self.buffer_cache.get_mut(destination.buffer);

        entry.write_tick = work.tick();
        entry.invalid.subtract(destination.offset, bytes);
        entry.contents.last_use = Some(work);
        if !modified.is_empty() {
            self.buffer_cache
                .memory_tracker
                .gpu_write(destination_address, bytes);
        }
        self.runtime.statistics.gpu_copied_bytes += bytes;
        // Eden also mirrors RAM with UnsafeReadWrite, without invalidating the
        // native destination or downloading the GPU-produced source intervals.
        let mut copy = std::mem::take(&mut self.buffer_cache.dma_buffer);
        copy.resize(bytes as usize, 0);
        let device = &self.runtime.device_memory;
        let result = device
            .reader()
            .read_block_unsafe(source_address, &mut copy)
            .and_then(|()| {
                device
                    .write_block_unsafe(destination_address, nixe_memory::WriteSource::Bytes(&copy))
            });
        self.buffer_cache.dma_buffer = copy;
        result.map_err(Into::into)
    }
}

impl Residency<'_> {
    fn dma_buffer_image(
        &mut self,
        operation: &MaxwellMemoryCopyOperation,
        space: &MaxwellGpuMemory,
        upload: bool,
    ) -> Result<bool, VulkanError> {
        let (image_address, image_size, image_layout, buffer_address, buffer_layout) = if upload {
            (
                operation.destination_address(),
                operation.destination_range_size(),
                operation.destination_layout(),
                operation.source_address(),
                operation.source_layout(),
            )
        } else {
            (
                operation.source_address(),
                operation.source_range_size(),
                operation.source_layout(),
                operation.destination_address(),
                operation.destination_layout(),
            )
        };
        let Some((image, [level, layer])) =
            self.dma_image_id(space, image_address, image_size, image_layout, upload)?
        else {
            return Ok(false);
        };
        let entry = self.texture_cache.get(image);
        let mut planes = entry
            .planes
            .iter()
            .filter(|plane| plane.mip_level == level && plane.array_layer == layer);
        let Some(&plane) = planes.next() else {
            return Ok(false);
        };
        // Vulkan copies one native aspect. Encoded/decoded and packed multiple-
        // aspect representations use the engine's exact CPU transformation.
        if planes.next().is_some()
            || entry.decoded.is_some()
            || plane.block_extent
                != (vk::Extent2D {
                    width: 1,
                    height: 1,
                })
            || (upload && entry.description.samples != vk::SampleCountFlags::TYPE_1)
        {
            return Ok(false);
        }
        let stored = entry.storage_plane(plane);
        let multisample = entry.description.samples != vk::SampleCountFlags::TYPE_1;
        let MaxwellMemoryCopyLayout::BlockLinear { x, y, .. } = image_layout else {
            unreachable!()
        };
        let MaxwellMemoryCopyLayout::Pitch { pitch } = buffer_layout else {
            unreachable!()
        };
        let bpp = plane.bytes_per_block;
        if !x.is_multiple_of(bpp)
            || !operation.width().is_multiple_of(bpp)
            || !pitch.is_multiple_of(bpp)
        {
            return Ok(false);
        }
        let rect = vk::Rect2D {
            offset: vk::Offset2D {
                x: (x / bpp) as i32,
                y: y as i32,
            },
            extent: vk::Extent2D {
                width: operation.width() / bpp,
                height: operation.height(),
            },
        };
        if x / bpp + rect.extent.width > stored.extent.width
            || y + rect.extent.height > stored.extent.height
        {
            return Ok(false);
        }
        let buffer_size = u64::from(pitch) * u64::from(operation.height());
        let gpu = space.address(buffer_address).map_err(failure)?;
        if space
            .max_continuous_range(gpu, buffer_size)
            .map_err(failure)?
            < buffer_size
        {
            return Ok(false);
        }
        let Ok(address) = space.device_address(gpu) else {
            return Ok(false);
        };
        let retained = self.runtime.constructing.len();
        self.runtime.constructing.push(Resource::Image(image));
        let result = self.record_dma_buffer_image(
            operation,
            image,
            plane,
            stored,
            rect,
            address,
            pitch,
            buffer_size,
            multisample,
            upload,
        );
        self.runtime.constructing.truncate(retained);
        result
    }
    #[allow(clippy::too_many_arguments)]
    fn record_dma_buffer_image(
        &mut self,
        operation: &MaxwellMemoryCopyOperation,
        image: ImageId,
        plane: ImagePlane,
        stored: ImagePlane,
        rect: vk::Rect2D,
        address: u64,
        pitch: u32,
        buffer_size: u64,
        multisample: bool,
        upload: bool,
    ) -> Result<bool, VulkanError> {
        // ObtainBuffer(FullSynchronize) preserves the target's untouched rows
        // and padding, as well as GPU-produced source data on image uploads.
        let buffer = self.device_buffer(
            address,
            buffer_size,
            if upload {
                vk::BufferUsageFlags::TRANSFER_SRC
            } else {
                vk::BufferUsageFlags::TRANSFER_DST
            },
            true,
            false,
        )?;
        self.runtime
            .constructing
            .push(Resource::Buffer(buffer.buffer));
        if !buffer
            .offset
            .is_multiple_of(u64::from(plane.bytes_per_block))
            || !buffer.offset.is_multiple_of(4)
        {
            return Ok(false);
        }
        self.scale_image(image, false, false)?;
        self.prepare_image(image, upload, false)?;
        let mut region = vk::BufferImageCopy::default()
            .buffer_row_length(pitch / plane.bytes_per_block)
            .buffer_image_height(operation.height())
            .image_subresource(plane.layers())
            .image_offset(vk::Offset3D {
                x: rect.offset.x,
                y: rect.offset.y,
                z: 0,
            })
            .image_extent(vk::Extent3D {
                width: rect.extent.width,
                height: rect.extent.height,
                depth: 1,
            });
        if upload {
            let source = self.use_buffer(buffer.buffer, buffer.offset, buffer_size)?;
            let (destination, layout) = self.image_planes_transfer(image, [plane])?;
            self.runtime.barriers.emit(&mut self.runtime.device)?;
            region.buffer_offset = buffer.offset;
            unsafe {
                self.runtime.device.dispatch.copy_buffer_to_image(
                    source,
                    destination,
                    layout,
                    &[region],
                );
            }
            self.runtime.statistics.dma_image_uploads += 1;
        } else {
            let converted = if multisample {
                Some(self.sample_readback_plane(image, stored)?)
            } else {
                None
            };
            let (source, layout) = if let Some(attachment) = converted {
                region.image_subresource = vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1);
                self.transfer_attachment(attachment)?
            } else {
                self.image_planes_transfer(image, [plane])?
            };
            let destination = self.use_buffer(buffer.buffer, buffer.offset, buffer_size)?;
            let (slice, ticket) = self
                .runtime
                .device
                .submission
                .reserve_readback_aligned(buffer_size, 64)?;
            let staging = self.runtime.device.submission.staging_buffer(&slice).handle;
            let mapping = self
                .runtime
                .device_memory
                .buffer_mapping(address, buffer_size)?;
            let mut inputs = Vec::new();
            // Only bytes actually copied from the image are published. Vulkan
            // leaves row padding undefined in the direct DMA staging copy.
            for row in 0..u64::from(operation.height()) {
                let start = row * u64::from(pitch);
                let bytes = u64::from(operation.width());
                for span in mapping.spans() {
                    let first = start.max(span.logical);
                    let last = (start + bytes).min(span.logical + span.bytes);
                    if first < last {
                        let mut offset = first;
                        for input in mapping.pin(first, last - first)? {
                            let length = input.bytes();
                            inputs.push((offset, input));
                            offset += length;
                        }
                    }
                }
            }
            self.runtime.barriers.emit(&mut self.runtime.device)?;
            region.buffer_offset = buffer.offset;
            unsafe {
                self.runtime.device.dispatch.copy_image_to_buffer(
                    source,
                    layout,
                    destination,
                    &[region],
                );
            }
            region.buffer_offset = slice.offset;
            unsafe {
                self.runtime.device.dispatch.copy_image_to_buffer(
                    source,
                    layout,
                    staging,
                    &[region],
                );
            }
            crate::transfer::host_read_barrier(
                &mut self.runtime.device,
                staging,
                slice.offset,
                buffer_size,
            );
            if let Some(attachment) = converted {
                self.release_attachment(attachment)?;
            }
            self.finish_buffer_write(buffer.buffer, buffer.offset, buffer_size)?;
            self.buffer_cache.async_downloads.add(address, buffer_size);
            self.texture_cache.pending_downloads.push(
                super::texture_download::PendingDownload::Buffer(
                    super::buffer_download::Download::image_copy(
                        ticket,
                        address,
                        buffer_size,
                        inputs,
                    ),
                ),
            );
        }
        self.runtime.statistics.dma_image_downloads += u64::from(!upload);
        self.runtime.statistics.gpu_copied_bytes +=
            u64::from(operation.width()) * u64::from(operation.height());
        Ok(true)
    }
}
