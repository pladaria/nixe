//! Alias reconciliation consumes newest physical intersections without registering
//! a second, linear representation of an image.
use super::*;
use nixe_gpu::authority::ResolvedBytes;

impl VulkanResidency {
    pub(super) fn refresh_image_intersections(
        &mut self,
        id: ImageId,
        parts: &[ResolvedBytes<Resource>],
        bytes: u64,
    ) -> Result<(), VulkanError> {
        let retained = self.constructing.len();
        self.constructing.push(Resource::Image(id));
        self.constructing
            .extend(parts.iter().filter_map(|p| p.producer.map(|p| p.resource)));
        let result = self.refresh_intersections_inner(id, parts, bytes);
        self.constructing.truncate(retained);
        result
    }
    fn refresh_intersections_inner(
        &mut self,
        id: ImageId,
        parts: &[ResolvedBytes<Resource>],
        bytes: u64,
    ) -> Result<(), VulkanError> {
        if self.copy_native_intersections(id, parts)? {
            return Ok(());
        }
        let entry = self.image(id)?;
        let import_cpu = entry.canonical_import;
        let geometry = entry.contents.geometry.clone();
        let mut changed = Vec::new();
        for part in parts {
            let needs_copy = match part.producer {
                Some(source) => {
                    !(entry.synchronized && source.sequence.get() <= entry.synchronized_sequence)
                        && (source.resource != Resource::Image(id) || source.native != part.logical)
                }
                None => import_cpu && !entry.synchronized,
            };
            if !needs_copy {
                continue;
            }
            for plane in entry.planes.iter() {
                let begin = part.logical.max(plane.offset);
                let end = (part.logical + part.bytes).min(plane.end().unwrap());
                if begin >= end {
                    continue;
                }
                let block = u64::from(plane.bytes_per_block);
                let first = (begin - plane.offset) / plane.row_pitch;
                let last = (end - plane.offset).div_ceil(plane.row_pitch);
                for row in first..last {
                    let base = plane.offset + row * plane.row_pitch;
                    let low = begin.max(base) - base;
                    let high = end.min(base + plane.row_bytes()) - base;
                    if low < high {
                        let start = base + low / block * block;
                        let stop = base + high.div_ceil(block) * block;
                        changed.push((start, stop - start));
                    }
                }
            }
        }
        normalize(&mut changed);
        let Some(&(base, _)) = changed.first() else {
            return Ok(());
        };
        let &(last, length) = changed.last().unwrap();
        let scratch_bytes = last + length - base;
        debug_assert!(last + length <= bytes);
        let mut selected = Vec::new();
        for part in parts {
            if !import_cpu && part.producer.is_none() {
                continue;
            }
            let first = changed.partition_point(|&(start, size)| start + size <= part.logical);
            for &(start, size) in &changed[first..] {
                if start >= part.logical + part.bytes {
                    break;
                }
                let low = start.max(part.logical);
                let high = (start + size).min(part.logical + part.bytes);
                let mut piece = *part;
                let skip = low - part.logical;
                piece.start = nixe_memory::ArenaOffset::new(part.start.get() + skip);
                piece.logical = low;
                piece.bytes = high - low;
                if let Some(producer) = &mut piece.producer {
                    producer.native += skip;
                }
                selected.push(piece);
            }
        }
        let mut dirty: Vec<_> = selected
            .iter()
            .map(|part| (part.logical, part.bytes))
            .collect();
        normalize(&mut dirty);
        let scratch = Arc::new(self.temporary_buffer(scratch_bytes, &[])?);
        let mut captures = Vec::new();
        let mut windows = Vec::new();
        let mut packed = 0usize;
        let mut images = rustc_hash::FxHashMap::<ImageId, Vec<(u64, u64, u64)>>::default();
        let mut buffers = rustc_hash::FxHashMap::<BufferId, Vec<vk::BufferCopy>>::default();
        for part in &selected {
            if let Some(producer) = part.producer {
                match producer.resource {
                    Resource::Image(source) => {
                        let copies = images.entry(source).or_default();
                        if let Some((from, to, size)) = copies.last_mut()
                            && *from + *size == producer.native
                            && *to + *size == (part.logical - base)
                        {
                            *size += part.bytes;
                        } else {
                            copies.push((producer.native, (part.logical - base), part.bytes));
                        }
                    }
                    Resource::Buffer(source) => {
                        let copies = buffers.entry(source).or_default();
                        if let Some(last) = copies.last_mut()
                            && last.src_offset + last.size == producer.native
                            && last.dst_offset + last.size == (part.logical - base)
                        {
                            last.size += part.bytes;
                        } else {
                            copies.push(
                                vk::BufferCopy::default()
                                    .src_offset(producer.native)
                                    .dst_offset(part.logical - base)
                                    .size(part.bytes),
                            );
                        }
                    }
                }
            } else if import_cpu {
                captures.push(nixe_memory::PhysicalCapture {
                    source: nixe_memory::Extent {
                        start: part.start,
                        bytes: part.bytes,
                    },
                    destination: packed,
                });
                windows.push(((part.logical - base), packed, part.bytes as usize));
                packed += part.bytes as usize;
            }
        }
        if packed != 0 {
            let mut pages = Vec::new();
            for footprint in &geometry {
                footprint.mapping().observed_pages(footprint, &mut pages)?;
            }
            let inputs = nixe_gpu::physical::pin_selected_pages(pages)?;
            let mut captured = vec![0; packed];
            self.memory
                .capture_resource_scattered(&inputs, &captures, &mut captured)?;
            let slice = self.device.submission.reserve_upload_aligned(
                &mut self.device.dispatch,
                &captured,
                4,
            )?;
            let copies: Vec<_> = windows
                .iter()
                .map(|&(to, offset, bytes)| {
                    vk::BufferCopy::default()
                        .src_offset(slice.offset + offset as u64)
                        .dst_offset(to)
                        .size(bytes as u64)
                })
                .collect();
            let work = self.device.work_ticket()?;
            self.device.retire_buffer(Arc::clone(&scratch), work)?;
            self.barriers.emit(&mut self.device);
            let source = self.device.submission.staging_buffer(&slice).handle;
            unsafe {
                self.device
                    .dispatch
                    .cmd_copy_buffer(source, scratch.handle, &copies);
            }
            self.statistics.cpu_uploaded_bytes += captured.len() as u64;
        }
        for (source, copies) in buffers {
            for copy in &copies {
                self.buffer_access_range(
                    source,
                    copy.src_offset,
                    copy.size,
                    AccessScope::new(
                        vk::PipelineStageFlags2::COPY,
                        vk::AccessFlags2::TRANSFER_READ,
                    ),
                )?;
            }
            let native = self.buffer(source)?.native.handle;
            self.device.begin_recording()?;
            let work = self.device.work_ticket()?;
            self.device.retire_buffer(Arc::clone(&scratch), work)?;
            self.barriers.emit(&mut self.device);
            unsafe {
                self.device
                    .dispatch
                    .cmd_copy_buffer(native, scratch.handle, &copies);
            }
            self.statistics.gpu_copied_bytes += copies.iter().map(|c| c.size).sum::<u64>();
        }
        for (source, copies) in images {
            self.copy_image_bytes(source, &scratch, &copies)?;
        }
        self.device.begin_recording()?;

        scratch_read_barrier(&mut self.device, scratch.handle, scratch_bytes);
        self.copy_rows_to_image(&scratch, base, id, &dirty)
    }

    /// Eden synchronizes aliases at consumption with native image copies. Here
    /// exact authority additionally restricts each copy to the newest byte domain.
    /// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/texture_cache/texture_cache.h
    fn copy_native_intersections(
        &mut self,
        id: ImageId,
        parts: &[ResolvedBytes<Resource>],
    ) -> Result<bool, VulkanError> {
        let target = self.image(id)?;
        let mut groups = rustc_hash::FxHashMap::<
            (ImageId, u32, u32, u32, u32, u32, u32, u32),
            (ImagePlane, ImagePlane, Vec<vk::ImageCopy>),
        >::default();
        for part in parts {
            let Some(producer) = part.producer else {
                if target.canonical_import && !target.synchronized {
                    return Ok(false);
                }
                continue;
            };
            if target.synchronized && producer.sequence.get() <= target.synchronized_sequence {
                continue;
            }
            if producer.resource == Resource::Image(id) && producer.native == part.logical {
                continue;
            }
            let Resource::Image(source) = producer.resource else {
                return Ok(false);
            };
            if source == id {
                return Ok(false);
            }
            let source_entry = self.image(source)?;
            if source_entry.description.samples != target.description.samples {
                return Ok(false);
            }
            let mut from = producer.native;
            let mut to = part.logical;
            let mut remaining = part.bytes;
            while remaining != 0 {
                let Some(&src) = source_entry
                    .planes
                    .iter()
                    .find(|p| from >= p.offset && from < p.end().unwrap())
                else {
                    return Ok(false);
                };
                let Some(&dst) = target
                    .planes
                    .iter()
                    .find(|p| to >= p.offset && to < p.end().unwrap())
                else {
                    return Ok(false);
                };
                if src.block_extent != dst.block_extent
                    || src.bytes_per_block != dst.bytes_per_block
                    || src.aspect != dst.aspect
                    || !format::copy_compatible(
                        source_entry.description.format,
                        target.description.format,
                        dst.aspect,
                    )
                {
                    return Ok(false);
                }
                let block = u64::from(src.bytes_per_block);
                let sx = (from - src.offset) % src.row_pitch;
                let dx = (to - dst.offset) % dst.row_pitch;
                if sx >= src.row_bytes()
                    || dx >= dst.row_bytes()
                    || !sx.is_multiple_of(block)
                    || !dx.is_multiple_of(block)
                {
                    return Ok(false);
                }
                let size = remaining
                    .min(src.row_bytes() - sx)
                    .min(dst.row_bytes() - dx);
                if !size.is_multiple_of(block) {
                    return Ok(false);
                }
                let sy = ((from - src.offset) / src.row_pitch) as u32 * src.block_extent.height;
                let dy = ((to - dst.offset) / dst.row_pitch) as u32 * dst.block_extent.height;
                let sx = (sx / block) as u32 * src.block_extent.width;
                let dx = (dx / block) as u32 * dst.block_extent.width;
                let width = (size / block) as u32 * dst.block_extent.width;
                let width = width.min(src.extent.width - sx).min(dst.extent.width - dx);
                let height = src
                    .block_extent
                    .height
                    .min(src.extent.height - sy)
                    .min(dst.extent.height - dy);
                let copy = vk::ImageCopy::default()
                    .src_subresource(src.layers())
                    .dst_subresource(dst.layers())
                    .src_offset(vk::Offset3D {
                        x: sx as i32,
                        y: sy as i32,
                        z: src.depth_slice as i32,
                    })
                    .dst_offset(vk::Offset3D {
                        x: dx as i32,
                        y: dy as i32,
                        z: dst.depth_slice as i32,
                    })
                    .extent(vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    });
                groups
                    .entry((
                        source,
                        src.aspect.as_raw(),
                        src.mip_level,
                        src.array_layer,
                        src.depth_slice,
                        dst.mip_level,
                        dst.array_layer,
                        dst.depth_slice,
                    ))
                    .or_insert_with(|| (src, dst, Vec::new()))
                    .2
                    .push(copy);
                from += size;
                to += size;
                remaining -= size;
            }
        }
        for ((source, ..), (src, dst, mut copies)) in groups {
            coalesce_copies(&mut copies);
            self.copy_alias_regions(source, src, id, dst, &copies)?;
        }
        Ok(true)
    }

    pub(super) fn copy_alias_regions(
        &mut self,
        source: ImageId,
        src: ImagePlane,
        id: ImageId,
        dst: ImagePlane,
        copies: &[vk::ImageCopy],
    ) -> Result<(), VulkanError> {
        let (native, layout) = self.image_access_subresources(
            source,
            commands::plane_range(src),
            AccessScope::new(
                vk::PipelineStageFlags2::COPY,
                vk::AccessFlags2::TRANSFER_READ,
            ),
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        )?;
        let (target, target_layout) = self.image_access_subresources(
            id,
            commands::plane_range(dst),
            AccessScope::new(
                vk::PipelineStageFlags2::COPY,
                vk::AccessFlags2::TRANSFER_WRITE,
            ),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        )?;
        self.device.begin_recording()?;
        self.barriers.emit(&mut self.device);
        unsafe {
            self.device
                .dispatch
                .cmd_copy_image(native, layout, target, target_layout, copies);
        }
        self.statistics.alias_copy_commands += 1;
        self.statistics.alias_copy_regions += copies.len() as u64;
        for copy in copies {
            self.define_image_rect(
                id,
                dst,
                vk::Rect2D {
                    offset: vk::Offset2D {
                        x: copy.dst_offset.x,
                        y: copy.dst_offset.y,
                    },
                    extent: vk::Extent2D {
                        width: copy.extent.width,
                        height: copy.extent.height,
                    },
                },
            );
            self.statistics.gpu_copied_bytes +=
                u64::from(copy.extent.width.div_ceil(dst.block_extent.width))
                    * u64::from(copy.extent.height.div_ceil(dst.block_extent.height))
                    * u64::from(dst.bytes_per_block);
        }
        Ok(())
    }
}

// Physical tiling splits a rectangular write into many disjoint byte intervals.
// Recover row runs first, then rectangles with the same source translation.
fn coalesce_copies(copies: &mut Vec<vk::ImageCopy>) {
    for vertical in [false, true] {
        copies.sort_unstable_by_key(|c| {
            let (s, d, e) = (c.src_offset, c.dst_offset, c.extent);
            [
                i64::from(s.x) - i64::from(d.x),
                i64::from(s.y) - i64::from(d.y),
                i64::from(s.z),
                i64::from(d.z),
                i64::from(e.depth),
                if vertical {
                    i64::from(d.x)
                } else {
                    i64::from(d.y)
                },
                if vertical {
                    i64::from(e.width)
                } else {
                    i64::from(e.height)
                },
                if vertical {
                    i64::from(d.y)
                } else {
                    i64::from(d.x)
                },
            ]
        });
        let mut count = 0;
        for index in 0..copies.len() {
            let next = copies[index];
            if count > 0 {
                let previous = &mut copies[count - 1];
                let contiguous = previous.src_offset.z == next.src_offset.z
                    && previous.dst_offset.z == next.dst_offset.z
                    && previous.extent.depth == next.extent.depth
                    && if vertical {
                        previous.src_offset.x == next.src_offset.x
                            && previous.dst_offset.x == next.dst_offset.x
                            && previous.extent.width == next.extent.width
                            && previous.src_offset.y + previous.extent.height as i32
                                == next.src_offset.y
                            && previous.dst_offset.y + previous.extent.height as i32
                                == next.dst_offset.y
                    } else {
                        previous.src_offset.y == next.src_offset.y
                            && previous.dst_offset.y == next.dst_offset.y
                            && previous.extent.height == next.extent.height
                            && previous.src_offset.x + previous.extent.width as i32
                                == next.src_offset.x
                            && previous.dst_offset.x + previous.extent.width as i32
                                == next.dst_offset.x
                    };
                if contiguous {
                    if vertical {
                        previous.extent.height += next.extent.height;
                    } else {
                        previous.extent.width += next.extent.width;
                    }
                    continue;
                }
            }
            copies[count] = next;
            count += 1;
        }
        copies.truncate(count);
    }
}
