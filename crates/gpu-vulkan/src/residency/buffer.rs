//! Physical buffer interpretations, immutable uploads and native alias reconstruction.
use super::*;
/// A retained native buffer and the addressed subrange within it.
#[derive(Clone, Copy, Debug)]
pub struct BufferRange {
    pub buffer: BufferId,
    pub offset: u64,
    pub size: u64,
}
#[derive(Clone, Copy)]
pub(super) struct BufferStreamRequest {
    pub id: BufferId,
    pub offset: u64,
    pub size: u64,
    pub alignment: u64,
}
pub(super) type BufferStream = Option<(Arc<AllocatedBuffer>, u64)>;
impl VulkanResidency {
    pub fn buffer_for(
        &mut self,
        backing: Arc<PhysicalMapping>,
        usage: vk::BufferUsageFlags,
    ) -> Result<BufferRange, VulkanError> {
        self.create_buffer(backing, usage, true)
    }
    /// Resolve retained identity without importing bytes until ordered use.
    pub fn buffer_for_deferred_import(
        &mut self,
        backing: Arc<PhysicalMapping>,
        usage: vk::BufferUsageFlags,
    ) -> Result<BufferRange, VulkanError> {
        self.create_buffer(backing, usage, false)
    }
    fn create_buffer(
        &mut self,
        requested: Arc<PhysicalMapping>,
        usage: vk::BufferUsageFlags,
        initialize: bool,
    ) -> Result<BufferRange, VulkanError> {
        let _trace = nixe_gpu::trace::span("buffer_lookup", requested.bytes(), 0, 0);
        self.invalidate_cpu_resources(&[])?;
        // A buffer's physical coverage is independent of the current binding's
        // role. Vulkan storage is allocated for all ordinary buffer consumers.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/buffer_cache/buffer_cache.h
        let usage = usage
            | vk::BufferUsageFlags::VERTEX_BUFFER
            | vk::BufferUsageFlags::INDEX_BUFFER
            | vk::BufferUsageFlags::INDIRECT_BUFFER
            | vk::BufferUsageFlags::UNIFORM_BUFFER
            | vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::UNIFORM_TEXEL_BUFFER
            | vk::BufferUsageFlags::STORAGE_TEXEL_BUFFER;
        // Look up retained coverage before expanding or pinning a new mapping.
        // Any containing interpretation must cover the first requested byte.
        let mut found = None;
        self.authority.visit_resource_overlaps(
            nixe_memory::Extent {
                start: requested.spans()[0].start,
                bytes: 1,
            },
            |resource, _, _| {
                if found.is_some() {
                    return;
                }
                let Resource::Buffer(id) = resource else {
                    return;
                };
                let Ok(entry) = self.buffer(id) else { return };
                let [old] = entry.contents.geometry.as_slice() else {
                    return;
                };
                if entry.binding_valid
                    && entry.usage.contains(usage)
                    && matches!(old.domain(), nixe_gpu::physical::WriteDomain::Buffer { .. })
                    && let Some(offset) = old.mapping().containing_offset(&requested)
                {
                    found = Some((id, offset));
                }
            },
        );
        if let Some((id, offset)) = found {
            if initialize {
                self.ensure_buffer_range(id, offset, requested.bytes())?;
            }
            self.touch(Resource::Buffer(id));
            self.statistics.cache_hits += 1;
            return Ok(BufferRange {
                buffer: id,
                offset,
                size: requested.bytes(),
            });
        }
        let mut mapping = requested.expanded_buffer(nixe_memory::PAGE_SIZE)?;
        let mut overlaps = Vec::new();
        let mut growth = 0u32;
        loop {
            let footprint = Footprint::new(
                Arc::clone(&mapping),
                nixe_gpu::physical::WriteDomain::Buffer {
                    start: 0,
                    bytes: mapping.bytes(),
                },
            )?;
            let mut expanded = false;
            for resource in self.authority.resource_candidates(&footprint) {
                let Resource::Buffer(id) = resource else {
                    continue;
                };
                let entry = self.buffer(id)?;
                if !entry.binding_valid {
                    continue;
                }
                let [old] = entry.contents.geometry.as_slice() else {
                    continue;
                };
                if !matches!(old.domain(), nixe_gpu::physical::WriteDomain::Buffer { .. }) {
                    continue;
                }
                if overlaps.contains(&id) {
                    continue;
                }
                let joined = join_buffers(&mapping, old.mapping())?;
                let Some(joined) = joined else { continue };
                if entry.contents.touched + 1 >= self.epoch {
                    growth = growth.max(entry.growth.saturating_add(1));
                }
                overlaps.push(id);
                if joined.bytes() != mapping.bytes() {
                    mapping = joined;
                    expanded = true;
                }
            }
            if growth >= 4 && !overlaps.is_empty() {
                let grown = mapping.expanded_buffer(1024 * 1024)?;
                if grown.bytes() != mapping.bytes() {
                    mapping = grown;
                    expanded = true;
                }
            }
            if !expanded {
                break;
            }
        }
        let offset = mapping
            .containing_offset(&requested)
            .ok_or(VulkanError::InvalidResource(
                "buffer merge changed request addressing",
            ))?;
        let bytes = mapping.bytes();
        let footprint = Footprint::new(
            Arc::clone(&mapping),
            nixe_gpu::physical::WriteDomain::Buffer { start: 0, bytes },
        )?;
        self.constructing
            .extend(overlaps.iter().copied().map(Resource::Buffer));
        let created = self.allocate_buffer_entry(footprint, bytes, usage);
        for &id in overlaps.iter().rev() {
            self.unpin(Resource::Buffer(id));
        }
        let id = created?;
        self.constructing.push(Resource::Buffer(id));
        let merged = (|| {
            for old in overlaps {
                if old == id {
                    continue;
                }
                let _merge =
                    nixe_gpu::trace::span("buffer_merge", old.slot as u64, id.slot as u64, 0);
                let old_mapping = Arc::clone(self.buffer(old)?.contents.geometry[0].mapping());
                let destination = mapping
                    .containing_offset(&old_mapping)
                    .ok_or(VulkanError::InvalidResource("buffer merge lost overlap"))?;
                let old_bytes = old_mapping.bytes();
                // Preserve native contents without downloading to RAM. Dirty
                // CPU ranges remain dirty in the new entry until consumption.
                let invalid = self.buffer(id)?.invalid.clone();
                let old_current = self.buffer(old)?.contents.observations_current();
                self.copy_source(Resource::Buffer(old), 0, id, destination, old_bytes)?;
                self.buffer_mut(id).invalid = invalid;
                let valid = if old_current {
                    self.buffer(old)?.invalid.clone()
                } else {
                    vec![(0, old_bytes)]
                };
                let mut cursor = 0;
                for (begin, size) in valid.into_iter().chain([(old_bytes, 0)]) {
                    if begin > cursor {
                        self.buffer_mut(id)
                            .validate_interval(destination + cursor, begin - cursor);
                    }
                    cursor = cursor.max(begin + size);
                }
                let tick = nixe_gpu::authority::NativeTick(self.device.work_ticket()?.tick());
                self.authority.relocate_buffer(
                    Resource::Buffer(old),
                    Resource::Buffer(id),
                    tick,
                )?;
                let plans =
                    std::mem::take(&mut self.buffers[old.slot].entry.as_mut().unwrap().write_plans);
                self.buffers[id.slot]
                    .entry
                    .as_mut()
                    .unwrap()
                    .write_plans
                    .extend(
                        plans
                            .into_iter()
                            .map(|(start, size, writer)| (start + destination, size, writer)),
                    );
                if let Some(writes) = self.buffer_writes.remove(&old) {
                    self.buffer_writes.entry(id).or_default().extend(
                        writes
                            .into_iter()
                            .map(|(start, size)| (destination + start, size)),
                    );
                }
                self.retire_entry(Resource::Buffer(old))?;
                self.statistics.buffer_merges += 1;
            }
            self.buffers[id.slot].entry.as_mut().unwrap().growth = growth;
            let footprint = Arc::clone(&self.buffer(id)?.contents.geometry[0]);
            let observation = self
                .authority
                .observe_native(Resource::Buffer(id), footprint)?;
            self.contents_mut(Resource::Buffer(id)).device = vec![observation];
            if initialize {
                self.ensure_buffer_range(id, offset, requested.bytes())?;
            }
            self.touch(Resource::Buffer(id));
            Ok(BufferRange {
                buffer: id,
                offset,
                size: requested.bytes(),
            })
        })();
        self.unpin(Resource::Buffer(id));
        merged
    }
    fn allocate_buffer_entry(
        &mut self,
        footprint: Arc<Footprint>,
        bytes: u64,
        usage: vk::BufferUsageFlags,
    ) -> Result<BufferId, VulkanError> {
        if usage.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
            && !self.device.profile().buffer_device_address
        {
            return Err(VulkanError::InvalidResource(
                "buffer device address not enabled",
            ));
        }
        let usage = usage | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
        let geometry = vec![footprint];
        let native = self.allocate_buffer(bytes, usage)?;
        #[cfg(feature = "binding-probe")]
        let address = if usage.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS) {
            unsafe {
                self.device.owner.device.get_buffer_device_address(
                    &vk::BufferDeviceAddressInfo::default().buffer(native.handle),
                )
            }
        } else {
            0
        };
        let slot = self.free_buffers.pop().unwrap_or(self.buffers.len());
        if slot == self.buffers.len() {
            self.buffers.push(Slot {
                generation: 0,
                entry: None,
            });
        }
        let generation =
            self.buffers[slot]
                .generation
                .checked_add(1)
                .ok_or(VulkanError::InvalidResource(
                    "native buffer generation exhausted",
                ))?;
        let id = BufferId {
            instance: self.instance,
            slot,
            generation,
        };
        self.buffers[slot] = Slot {
            generation,
            entry: Some(BufferEntry {
                known: Vec::new(),
                growth: 0,
                binding_valid: true,
                write_plans: Vec::new(),
                invalid: vec![(0, bytes)],
                contents: Contents {
                    lru: None,
                    geometry: geometry.clone(),
                    device: Vec::new(),
                    last_use: None,
                    touched: 0,
                },
                native: Arc::new(native),
                usage,
                #[cfg(feature = "binding-probe")]
                address,
                access: sync::BufferAccess::new(bytes),
            }),
        };
        self.authority
            .register(Resource::Buffer(id), geometry.clone())
            .map_err(memory_error)?;

        let observations = geometry
            .into_iter()
            .map(|f| self.authority.observe_native(Resource::Buffer(id), f))
            .collect::<Result<Vec<_>, _>>()?;
        self.buffers[id.slot]
            .entry
            .as_mut()
            .expect("buffer")
            .contents
            .device = observations;
        self.statistics.buffer_allocations += 1;
        self.touch(Resource::Buffer(id));
        Ok(id)
    }
    /// Match the cache/stream preference over recent guest frames, independently
    /// of host presentation frequency. Alignment copies remain mandatory.
    /// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/buffer_cache/buffer_cache.h
    pub(crate) fn tick_uniform_cache(&mut self) {
        let hits: u64 = self.uniform_cache_hits.iter().sum();
        let shots: u64 = self.uniform_cache_shots.iter().sum();
        self.uniform_stream_limit = if hits * 256 < shots * 251 { 4096 } else { 0 };
        self.uniform_cache_hits.copy_within(..17, 1);
        self.uniform_cache_shots.copy_within(..17, 1);
        self.uniform_cache_hits[0] = 0;
        self.uniform_cache_shots[0] = 0;
    }
    /// Frequently rewritten CPU uniforms receive immutable upload slices. They
    /// stay dirty and do not install page observers or quiesce running cores.
    pub(super) fn stream_buffer(
        &mut self,
        request: BufferStreamRequest,
    ) -> Result<BufferStream, VulkanError> {
        self.check_buffer_range(request.id, request.offset, request.size)?;
        if request.size > 64 * 1024
            || request.size > self.device.submission.configuration.arena_bytes
        {
            return Ok(None);
        }
        let entry = self.buffer(request.id)?;
        let [full] = entry.contents.geometry.as_slice() else {
            return Ok(None);
        };
        if !matches!(
            full.domain(),
            nixe_gpu::physical::WriteDomain::Buffer { .. }
        ) || self
            .authority
            .has_buffer_producer(full.mapping(), request.offset, request.size)?
        {
            return Ok(None);
        }
        let mapping = Arc::clone(full.mapping());
        let result = (|| {
            mapping.pin_into(request.offset, request.size, &mut self.stream_inputs)?;
            let (slot, slice) =
                self.reserve_upload_space(request.size, request.alignment.max(4))?;
            let upload = self.device.submission.slots[slot].upload_buffer(slice);
            unsafe {
                upload.write_with(slice.offset, request.size, |bytes| {
                    self.memory
                        .capture_stream(&self.stream_inputs, bytes)
                        .map_err(memory_error)
                })?;
            }
            self.device.submission.retain_stream_upload(slot);
            let native = self.device.submission.slots[slot].upload_lease(slice);
            self.touch(Resource::Buffer(request.id));
            self.statistics.cpu_streamed_bytes += request.size;
            Ok(Some((native, slice.offset)))
        })();
        self.stream_inputs.clear();
        result
    }
    pub(super) fn aligned_uniform_copy(
        &mut self,
        id: BufferId,
        offset: u64,
        size: u64,
    ) -> Result<Arc<AllocatedBuffer>, VulkanError> {
        self.ensure_buffer_range(id, offset, size)?;
        let copy = Arc::new(self.allocate_buffer(
            size,
            vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        )?);
        let source = self.buffer_access_range(
            id,
            offset,
            size,
            AccessScope::new(
                vk::PipelineStageFlags2::COPY,
                vk::AccessFlags2::TRANSFER_READ,
            ),
        )?;
        let slot = self.device.begin_recording()?;
        self.barriers.emit(&mut self.device);
        let command = self.device.submission.slots[slot].command;
        unsafe {
            self.device.dispatch.cmd_copy_buffer(
                command,
                source,
                copy.handle,
                &[vk::BufferCopy::default().src_offset(offset).size(size)],
            );
            let barrier = [vk::BufferMemoryBarrier2::default()
                .buffer(copy.handle)
                .size(size)
                .src_stage_mask(vk::PipelineStageFlags2::COPY)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .dst_access_mask(vk::AccessFlags2::UNIFORM_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)];
            self.device.dispatch.barrier(
                command,
                &vk::DependencyInfo::default().buffer_memory_barriers(&barrier),
            );
        }
        // The recording slot owns the transfer; its descriptor record extends
        // the allocation through the actual shader consumer if preparation submits.
        let work = self.device.work_ticket()?;
        self.device.retire_buffer(Arc::clone(&copy), work)?;
        self.statistics.gpu_copied_bytes += size;
        Ok(copy)
    }
    pub(super) fn track_buffer_cpu(
        &mut self,
        id: BufferId,
        offset: u64,
        bytes: u64,
    ) -> Result<(), VulkanError> {
        let contents = &self.buffer(id)?.contents;
        let full = &contents.geometry[0];
        let end = offset + bytes;
        let spans = full.mapping().spans();
        let first = spans.partition_point(|span| span.logical + span.bytes <= offset);
        if spans[first..]
            .iter()
            .take_while(|span| span.logical < end)
            .all(|span| {
                let start = offset.max(span.logical);
                let stop = end.min(span.logical + span.bytes);
                self.memory.resource_pages_tracked(nixe_memory::Extent {
                    start: nixe_memory::ArenaOffset::new(span.start.get() + start - span.logical),
                    bytes: stop - start,
                })
            })
        {
            return Ok(());
        }
        let inputs = full.mapping().pin(offset, bytes)?;
        self.memory
            .capture_resource_scattered(&inputs, &[], &mut [])?;
        Ok(())
    }
    pub(crate) fn buffer_binding_current(&self, id: BufferId) -> bool {
        self.buffer(id).is_ok_and(|entry| entry.binding_valid)
    }
    pub fn ensure_buffer(&mut self, id: BufferId) -> Result<(), VulkanError> {
        self.ensure_buffer_range(id, 0, self.buffer(id)?.native.size())
    }
    pub(super) fn ensure_buffer_range(
        &mut self,
        id: BufferId,
        offset: u64,
        size: u64,
    ) -> Result<(), VulkanError> {
        self.check_buffer_range(id, offset, size)?;
        if !self.buffer_binding_current(id) {
            return Err(VulkanError::InvalidResource(
                "buffer binding has retired physical coverage",
            ));
        }
        self.invalidate_cpu_resources(&[])?;
        let entry = self.buffer(id)?;
        let first = entry
            .invalid
            .partition_point(|&(start, bytes)| start + bytes <= offset);
        if entry.contents.observations_current()
            && entry
                .invalid
                .get(first)
                .is_none_or(|&(start, _)| start >= offset + size)
        {
            return Ok(());
        }
        self.constructing.push(Resource::Buffer(id));
        let result = self.refresh_buffer(id, offset, size);
        self.unpin(Resource::Buffer(id));
        result
    }
    fn refresh_buffer(
        &mut self,
        id: BufferId,
        offset: u64,
        requested_size: u64,
    ) -> Result<(), VulkanError> {
        let _trace = nixe_gpu::trace::span(
            "buffer_refresh",
            id.slot as u64,
            self.buffer(id)?.native.size(),
            0,
        );
        self.invalidate_cpu_resources(&[])?;
        let mut geometry = self.buffer(id)?.contents.geometry.clone();
        let size = self.buffer(id)?.native.size();
        let changed = self
            .buffer(id)?
            .contents
            .device
            .iter()
            .any(ResourceObservation::is_invalid);
        if changed {
            let device = geometry
                .iter()
                .map(|f| {
                    self.authority
                        .observe_native(Resource::Buffer(id), Arc::clone(f))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let contents = self.contents_mut(Resource::Buffer(id));
            contents.device = device;
        }
        let page = nixe_memory::PAGE_SIZE;
        let begin = offset / page * page;
        let end = ((offset + requested_size).div_ceil(page) * page).min(size);
        if !changed {
            geometry[0] = Footprint::new(
                Arc::clone(geometry[0].mapping()),
                nixe_gpu::physical::WriteDomain::Buffer {
                    start: begin,
                    bytes: end - begin,
                },
            )?;
        }
        let mut parts = Vec::new();
        for footprint in &geometry {
            parts.extend(self.authority.resolve_footprint(footprint)?);
        }
        parts.sort_unstable_by_key(|part| part.logical);
        let mut copies = Vec::new();
        let mut windows: Vec<(u64, usize, usize)> = Vec::new();
        let mut packed = 0;
        let mut inputs = Vec::new();
        let invalid = self.buffer(id)?.invalid.clone();
        for part in parts.iter().filter(|p| p.producer.is_none()) {
            let first = invalid
                .partition_point(|&(offset, bytes)| offset + bytes <= part.logical.max(begin));
            for &(offset, bytes) in &invalid[first..] {
                if offset >= (part.logical + part.bytes).min(end) {
                    break;
                }
                let begin = part.logical.max(offset).max(begin);
                let end = (part.logical + part.bytes).min(offset + bytes).min(end);
                if begin >= end {
                    continue;
                }
                let source = nixe_memory::ArenaOffset::new(part.start.get() + begin - part.logical);
                let mapping = geometry[0].mapping();
                let span = mapping
                    .spans()
                    .partition_point(|span| span.logical + span.bytes <= begin);
                let pin = mapping.pin_span(span, source, end - begin)?;
                inputs
                    .push(pin.ok_or(VulkanError::InvalidResource("CPU upload input has retired"))?);
                copies.push(nixe_memory::PhysicalCapture {
                    source: nixe_memory::Extent {
                        start: nixe_memory::ArenaOffset::new(
                            part.start.get() + begin - part.logical,
                        ),
                        bytes: end - begin,
                    },
                    destination: packed,
                });
                if let Some((offset, _, bytes)) = windows.last_mut()
                    && *offset + *bytes as u64 == begin
                {
                    *bytes += (end - begin) as usize;
                } else {
                    windows.push((begin, packed, (end - begin) as usize));
                }
                packed += (end - begin) as usize;
            }
        }
        let self_alias = parts.iter().any(|part| {
            part.producer
                .is_some_and(|p| p.resource == Resource::Buffer(id) && p.native != part.logical)
        });
        let preserved = if self_alias {
            let bytes = self.buffer(id)?.native.size();
            let scratch = Arc::new(self.temporary_buffer(bytes, &[Resource::Buffer(id)])?);
            let source = self.buffer_access_range(
                id,
                0,
                bytes,
                AccessScope::new(
                    vk::PipelineStageFlags2::COPY,
                    vk::AccessFlags2::TRANSFER_READ,
                ),
            )?;
            let slot = self.device.begin_recording()?;
            let work = self.device.work_ticket()?;
            self.device.retire_buffer(Arc::clone(&scratch), work)?;
            self.barriers.emit(&mut self.device);
            unsafe {
                let command = self.device.submission.slots[slot].command;
                self.device.dispatch.cmd_copy_buffer(
                    command,
                    source,
                    scratch.handle,
                    &[vk::BufferCopy::default().size(bytes)],
                );
                super::recording::scratch_read_barrier(
                    &mut self.device,
                    command,
                    scratch.handle,
                    bytes,
                );
            }
            Some(scratch)
        } else {
            None
        };
        self.upload_buffer_captures(id, &inputs, &copies, &windows, packed)?;
        let native_valid: Vec<_> = parts
            .iter()
            .filter(|p| p.producer.is_some())
            .map(|p| (p.logical, p.bytes))
            .collect();
        let mut images = rustc_hash::FxHashMap::<ImageId, Vec<(u64, u64, u64)>>::default();
        let mut buffers: Vec<(Resource, u64, u64, u64)> = Vec::new();
        for part in parts {
            let Some(source) = part.producer else {
                continue;
            };
            if source.resource == Resource::Buffer(id) && source.native == part.logical {
                continue;
            }
            if let Resource::Image(image) = source.resource {
                let copies = images.entry(image).or_default();
                if let Some((from, to, bytes)) = copies.last_mut()
                    && *from + *bytes == source.native
                    && *to + *bytes == part.logical
                {
                    *bytes += part.bytes;
                } else {
                    copies.push((source.native, part.logical, part.bytes));
                }
            } else if let Some((resource, from, to, bytes)) = buffers.last_mut()
                && *resource == source.resource
                && *from + *bytes == source.native
                && *to + *bytes == part.logical
            {
                *bytes += part.bytes;
            } else {
                buffers.push((source.resource, source.native, part.logical, part.bytes));
            }
        }
        for (source, from, to, bytes) in buffers {
            if source == Resource::Buffer(id) {
                let scratch = preserved.as_ref().expect("preserved native self alias");
                let target = self.buffer_access_range(
                    id,
                    to,
                    bytes,
                    AccessScope::new(
                        vk::PipelineStageFlags2::COPY,
                        vk::AccessFlags2::TRANSFER_WRITE,
                    ),
                )?;
                let slot = self.device.begin_recording()?;
                let work = self.device.work_ticket()?;
                self.device.retire_buffer(Arc::clone(scratch), work)?;
                self.barriers.emit(&mut self.device);
                unsafe {
                    self.device.dispatch.cmd_copy_buffer(
                        self.device.submission.slots[slot].command,
                        scratch.handle,
                        target,
                        &[vk::BufferCopy::default()
                            .src_offset(from)
                            .dst_offset(to)
                            .size(bytes)],
                    );
                }
            } else {
                self.copy_source(source, from, id, to, bytes)?;
            }
        }
        for (image, copies) in images {
            self.copy_image_buffer_regions(image, id, &copies)?;
        }
        let entry = self.buffers[id.slot].entry.as_mut().expect("buffer");
        entry.validate_intervals(&native_valid);
        // Publish the captured version before consuming racing CPU stores. The
        // persistent page watch keeps future stores dirty without a new observer.
        self.invalidate_cpu_resources(&[])?;
        Ok(())
    }
    pub fn buffer_lease(&self, id: BufferId) -> Result<Arc<AllocatedBuffer>, VulkanError> {
        Ok(Arc::clone(&self.buffer(id)?.native))
    }
}

fn join_buffers(
    a: &Arc<PhysicalMapping>,
    b: &Arc<PhysicalMapping>,
) -> Result<Option<Arc<PhysicalMapping>>, VulkanError> {
    if a.containing_offset(b).is_some() {
        return Ok(Some(Arc::clone(a)));
    }
    if b.containing_offset(a).is_some() {
        return Ok(Some(Arc::clone(b)));
    }
    for (first, second) in [(a, b), (b, a)] {
        let head = &second.spans()[0];
        for span in first.spans() {
            if span.allocation != head.allocation
                || head.start < span.start
                || head.start.get() >= span.start.get() + span.bytes
            {
                continue;
            }
            let base = span.logical + head.start.get() - span.start.get();
            let overlap = first.bytes() - base;
            if overlap >= second.bytes() {
                continue;
            }
            let suffix = first.slice(base, overlap)?;
            let prefix = second.slice(0, overlap)?;
            if suffix.containing_offset(&prefix) != Some(0) {
                continue;
            }
            return Ok(Some(PhysicalMapping::join(&[
                Arc::clone(first),
                second.slice(overlap, second.bytes() - overlap)?,
            ])?));
        }
    }
    Ok(None)
}
