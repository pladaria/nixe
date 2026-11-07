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
                    if from.aspect != destination.aspect
                        || from.block_extent != destination.block_extent
                        || !format::copy_compatible(
                            source.description.format,
                            target.description.format,
                            from.aspect,
                        )
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
    pub(super) fn synchronize_image_aliases(&mut self, id: ImageId) -> Result<(), VulkanError> {
        let target = self.texture_cache.get(id);
        let tick = target.modification_tick;
        let mut modified = target.gpu_modified;
        let mut newest = tick;
        let mut aliases = smallvec::SmallVec::<[(ImageId, Arc<[PlaneCopies]>, u64); 8]>::new();
        for alias in &target.aliases {
            let source = self.texture_cache.get(alias.image);
            if source.modification_tick > tick {
                newest = newest.max(source.modification_tick);
                modified |= source.gpu_modified;
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
        aliases.sort_unstable_by_key(|alias| alias.2);
        let retained = self.runtime.constructing.len();
        self.runtime
            .constructing
            .extend(aliases.iter().map(|alias| Resource::Image(alias.0)));
        let result: Result<(), VulkanError> = (|| {
            for (source, copies, _) in aliases {
                self.copy_alias_regions(source, id, &copies)?;
            }
            Ok(())
        })();
        self.runtime.constructing.truncate(retained);
        result?;
        let target = &mut self.texture_cache.images[id];
        target.modification_tick = newest;
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
        let copies: smallvec::SmallVec<[vk::ImageCopy; 16]> = planes
            .iter()
            .flat_map(|plane| plane.copies.iter().copied())
            .collect();
        unsafe {
            self.runtime
                .device
                .dispatch
                .copy_image(native, layout, target, target_layout, &copies);
        }
        self.runtime.statistics.alias_copy_commands += 1;
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
