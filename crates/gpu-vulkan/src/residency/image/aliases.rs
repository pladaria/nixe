//! Persistent image alias copies, selected by image modification ticks.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/texture_cache/image_base.cpp#L157-L253
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/texture_cache/texture_cache.h#L2451-L2518
use super::*;
use nixe_gpu::physical::{ImageFootprint, ImageStorage, WriteDomain};

fn same_storage(a: ImageFootprint, b: ImageFootprint) -> bool {
    if a.packed_bytes != b.packed_bytes
        || a.aspect_offset != b.aspect_offset
        || a.aspect_bytes != b.aspect_bytes
    {
        return false;
    }
    match (a.storage, b.storage) {
        (ImageStorage::Pitch { row_pitch: a }, ImageStorage::Pitch { row_pitch: b }) => a == b,
        (
            ImageStorage::BlockLinear {
                block_height_log2: ah,
                block_depth_log2: ad,
            },
            ImageStorage::BlockLinear {
                block_height_log2: bh,
                block_depth_log2: bd,
            },
        )
        | (
            ImageStorage::BlockLinearAmpere {
                block_height_log2: ah,
                block_depth_log2: ad,
            },
            ImageStorage::BlockLinearAmpere {
                block_height_log2: bh,
                block_depth_log2: bd,
            },
        ) => {
            ah == bh
                && ad == bd
                && (u64::from(a.columns) * u64::from(a.packed_bytes)).div_ceil(64)
                    == (u64::from(b.columns) * u64::from(b.packed_bytes)).div_ceil(64)
        }
        _ => false,
    }
}
fn same_backing(a: &Footprint, b: &Footprint) -> bool {
    if a.mapping().machine() != b.mapping().machine() {
        return false;
    }
    let ar = a.bounds();
    let br = b.bounds();
    let bytes = (ar.end - ar.start).min(br.end - br.start);
    fn spans(
        mapping: &nixe_memory::PhysicalMapping,
        start: u64,
        bytes: u64,
    ) -> impl Iterator<Item = (u64, nixe_memory::AllocationId, u64, u64)> + '_ {
        mapping.spans().iter().filter_map(move |span| {
            let begin = span.logical.max(start);
            let end = (span.logical + span.bytes).min(start + bytes);
            if begin >= end {
                return None;
            }
            Some((
                begin - start,
                span.allocation,
                span.start.get() + begin - span.logical,
                end - begin,
            ))
        })
    }
    let mut lhs = spans(a.mapping(), ar.start, bytes).peekable();
    lhs.peek().is_some() && lhs.eq(spans(b.mapping(), br.start, bytes))
}
impl Residency<'_> {
    pub(super) fn register_image_aliases(&mut self, id: ImageId) -> Result<(), VulkanError> {
        let mut candidates = smallvec::SmallVec::<[ImageId; 16]>::new();
        for &(address, size) in &self.texture_cache.get(id).maps {
            for page in address >> 20..=(address + size - 1) >> 20 {
                if let Some(ids) = self.texture_cache.image_pages.get(&page) {
                    for &other in ids {
                        if other != id && !candidates.contains(&other) {
                            candidates.push(other);
                        }
                    }
                }
            }
        }
        for other in candidates {
            let target = self.texture_cache.get(id);
            let source = self.texture_cache.get(other);
            if target.decoded.is_some()
                || source.decoded.is_some()
                || target.description.image_type != source.description.image_type
                || target.description.samples != source.description.samples
                || target.sample_grid != source.sample_grid
            {
                continue;
            }
            let mut forward = Vec::new();
            let mut backward = Vec::new();
            for (&destination, target_geometry) in
                target.planes.iter().zip(&target.contents.geometry)
            {
                let WriteDomain::Image(target_domain) = target_geometry.domain() else {
                    unreachable!("image geometry");
                };
                for (&from, source_geometry) in source.planes.iter().zip(&source.contents.geometry)
                {
                    let WriteDomain::Image(source_domain) = source_geometry.domain() else {
                        unreachable!("image geometry");
                    };
                    let direct = from.aspect == destination.aspect
                        && format::copy_compatible(
                            source.description.format,
                            target.description.format,
                            from.aspect,
                        );
                    let depth_alias = source.description.samples == vk::SampleCountFlags::TYPE_1
                        && matches!(
                            (source.description.format, target.description.format),
                            (vk::Format::R32_SFLOAT, vk::Format::D32_SFLOAT)
                                | (vk::Format::D32_SFLOAT, vk::Format::R32_SFLOAT)
                                | (vk::Format::R16_UNORM, vk::Format::D16_UNORM)
                                | (vk::Format::D16_UNORM, vk::Format::R16_UNORM)
                        );
                    if (!direct && !depth_alias)
                        || from.block_extent != destination.block_extent
                        || !same_storage(target_domain, source_domain)
                    {
                        continue;
                    }
                    if !same_backing(target_geometry, source_geometry) {
                        continue;
                    }
                    let extent = vk::Extent3D {
                        width: destination.extent.width.min(from.extent.width),
                        height: destination.extent.height.min(from.extent.height),
                        depth: 1,
                    };
                    let from_offset = vk::Offset3D {
                        x: 0,
                        y: 0,
                        z: from.depth_slice as i32,
                    };
                    let to_offset = vk::Offset3D {
                        x: 0,
                        y: 0,
                        z: destination.depth_slice as i32,
                    };
                    forward.push(PlaneCopies {
                        source: from,
                        destination,
                        copies: vec![
                            vk::ImageCopy::default()
                                .src_subresource(from.layers())
                                .dst_subresource(destination.layers())
                                .src_offset(from_offset)
                                .dst_offset(to_offset)
                                .extent(extent),
                        ],
                    });
                    backward.push(PlaneCopies {
                        source: destination,
                        destination: from,
                        copies: vec![
                            vk::ImageCopy::default()
                                .src_subresource(destination.layers())
                                .dst_subresource(from.layers())
                                .src_offset(to_offset)
                                .dst_offset(from_offset)
                                .extent(extent),
                        ],
                    });
                }
            }
            if !forward.is_empty() {
                self.texture_cache.images[id].aliases.push(ImageAlias {
                    image: other,
                    copies: forward.into(),
                });
                self.texture_cache.images[other].aliases.push(ImageAlias {
                    image: id,
                    copies: backward.into(),
                });
            }
        }
        Ok(())
    }
    // Select one scale for the destination and all newer alias producers before
    // copying them in modification order. Scale changes can retire image views.
    // https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/texture_cache/texture_cache.h#L2470-L2520
    pub(super) fn synchronize_image_aliases(&mut self, id: ImageId) -> Result<(), VulkanError> {
        let target = self.texture_cache.get(id);
        let tick = target.modification_tick;
        let mut modified = target.gpu_modified;
        let mut any_rescaled = target.rescaled;
        let mut newest = tick;
        let mut aliases = smallvec::SmallVec::<[(ImageId, Arc<[PlaneCopies]>, u64); 8]>::new();
        for alias in &target.aliases {
            let source = self.texture_cache.get(alias.image);
            if source.modification_tick > tick {
                newest = newest.max(source.modification_tick);
                modified |= source.gpu_modified;
                any_rescaled |= source.rescaled;
                aliases.push((
                    alias.image,
                    Arc::clone(&alias.copies),
                    source.modification_tick,
                ));
            }
        }
        if aliases.is_empty() {
            return Ok(());
        }
        let retained = self.runtime.constructing.len();
        self.runtime
            .constructing
            .extend(aliases.iter().map(|alias| Resource::Image(alias.0)));
        let result: Result<(), VulkanError> = (|| {
            let can_rescale = self.texture_cache.image_can_rescale(id);
            if any_rescaled {
                self.scale_image(id, can_rescale, false)?;
            }
            aliases.sort_unstable_by_key(|alias| alias.2);
            for (source, copies, _) in aliases {
                if self.resolution().active && any_rescaled {
                    self.scale_image(source, can_rescale, false)?;
                }
                self.copy_alias_regions(source, id, &copies)?;
            }
            Ok(())
        })();
        self.runtime.constructing.truncate(retained);
        result?;
        let target = &mut self.texture_cache.images[id];
        target.modification_tick = newest;
        if modified && !target.gpu_modified {
            for &(address, bytes) in &target.maps {
                self.runtime.device_memory.mark_gpu_written(address, bytes);
            }
        }
        target.gpu_modified = modified;
        Ok(())
    }
    pub(super) fn copy_alias_regions(
        &mut self,
        source: ImageId,
        id: ImageId,
        planes: &[PlaneCopies],
    ) -> Result<(), VulkanError> {
        let (native, layout) =
            self.image_planes_transfer(source, planes.iter().map(|plane| plane.source))?;
        let (target, target_layout) =
            self.image_planes_transfer(id, planes.iter().map(|plane| plane.destination))?;
        self.runtime.barriers.emit(&mut self.runtime.device)?;
        let mut copies: smallvec::SmallVec<[vk::ImageCopy; 16]> = planes
            .iter()
            .flat_map(|plane| plane.copies.iter().copied())
            .collect();
        if self.texture_cache.get(source).rescaled {
            let scale = self.resolution();
            for copy in &mut copies {
                copy.src_offset.x = scale.scale_signed(copy.src_offset.x);
                copy.src_offset.y = scale.scale_signed(copy.src_offset.y);
                copy.dst_offset.x = scale.scale_signed(copy.dst_offset.x);
                copy.dst_offset.y = scale.scale_signed(copy.dst_offset.y);
                copy.extent.width = scale.scale(copy.extent.width);
                copy.extent.height = scale.scale(copy.extent.height);
            }
        }
        if planes
            .iter()
            .any(|plane| plane.source.aspect != plane.destination.aspect)
        {
            // Color/depth formats have equal texel bits but cannot use
            // vkCmdCopyImage. A device-local buffer preserves those bits, with
            // no CPU readback or wait. The staging pool retains it to completion.
            // https://docs.vulkan.org/spec/latest/chapters/copies.html#copies-images
            let mut size = 0u64;
            let mut reads = smallvec::SmallVec::<[vk::BufferImageCopy; 16]>::new();
            let mut writes = smallvec::SmallVec::<[vk::BufferImageCopy; 16]>::new();
            for copy in &copies {
                let bytes_per_texel = planes[0].source.bytes_per_block;
                let offset = crate::queue::align_up(size, 4).ok_or(
                    VulkanError::InvalidResource("alias copy alignment overflow"),
                )?;
                let bytes = u64::from(copy.extent.width)
                    .checked_mul(u64::from(copy.extent.height))
                    .and_then(|v| v.checked_mul(u64::from(bytes_per_texel)))
                    .ok_or(VulkanError::InvalidResource("alias copy size overflow"))?;
                size = offset
                    .checked_add(bytes)
                    .ok_or(VulkanError::InvalidResource("alias copy size overflow"))?;
                reads.push(
                    vk::BufferImageCopy::default()
                        .buffer_offset(offset)
                        .image_subresource(copy.src_subresource)
                        .image_offset(copy.src_offset)
                        .image_extent(copy.extent),
                );
                writes.push(
                    vk::BufferImageCopy::default()
                        .buffer_offset(offset)
                        .image_subresource(copy.dst_subresource)
                        .image_offset(copy.dst_offset)
                        .image_extent(copy.extent),
                );
            }
            let scratch = self.runtime.device.submission.reserve_slice(
                size,
                crate::queue::staging::Usage::DeviceLocal,
                4,
            )?;
            for region in reads.iter_mut().chain(writes.iter_mut()) {
                region.buffer_offset += scratch.offset;
            }
            unsafe {
                self.runtime.device.dispatch.copy_image_to_buffer(
                    native,
                    layout,
                    scratch.buffer,
                    &reads,
                );
                self.runtime.device.dispatch.copy_buffer_to_image(
                    scratch.buffer,
                    target,
                    target_layout,
                    &writes,
                );
            }
            self.runtime.statistics.alias_copy_commands += 2;
        } else {
            unsafe {
                self.runtime.device.dispatch.copy_image(
                    native,
                    layout,
                    target,
                    target_layout,
                    &copies,
                );
            }
            self.runtime.statistics.alias_copy_commands += 1;
        }
        self.runtime.statistics.alias_copy_regions += copies.len() as u64;
        for plane in planes {
            let dst = plane.destination;
            for copy in &plane.copies {
                self.runtime.statistics.gpu_copied_bytes +=
                    u64::from(copy.extent.width.div_ceil(dst.block_extent.width))
                        * u64::from(copy.extent.height.div_ceil(dst.block_extent.height))
                        * u64::from(dst.bytes_per_block);
            }
        }
        Ok(())
    }
}

impl TextureCache {
    pub(in crate::residency) fn unregister_image_aliases(&mut self, id: ImageId) {
        let aliases = std::mem::take(&mut self.images[id].aliases);
        for alias in aliases {
            if let Some(entry) = self.images.get_mut(alias.image) {
                entry.aliases.retain(|alias| alias.image != id);
            }
        }
    }
}
