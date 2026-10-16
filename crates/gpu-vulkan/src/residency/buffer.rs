//! Physical buffer lookup, CPU epochs and GPU alias reconstruction.
use super::*;
pub(super) struct StreamedBuffer {
    work: WorkTicket,
    window: (u64, u64),
    buffer: Arc<AllocatedBuffer>,
    offset: u64,
}
#[derive(Clone)]
pub(super) struct ImageBufferImport {
    source: ImageId,
    // Proven immutable physical correspondence, in destination byte order.
    // Pixel freshness is checked separately when replaying the native copies.
    // Related alias-copy design: https://github.com/eden-emulator/mirror/blob/master/src/video_core/texture_cache/texture_cache.h
    copies: Arc<[(u64, u64, u64)]>,
}
impl VulkanResidency {
    pub fn buffer_for(
        &mut self,
        backing: CanonicalBackingRange,
        usage: vk::BufferUsageFlags,
    ) -> Result<BufferId, VulkanError> {
        self.create_buffer(backing, usage, true)
    }
    /// Prepare identity/allocation without reading guest bytes or recording a
    /// transfer. The ordered draw/dispatch consumer calls ensure_buffer once.
    pub fn buffer_for_deferred_import(
        &mut self,
        backing: CanonicalBackingRange,
        usage: vk::BufferUsageFlags,
    ) -> Result<BufferId, VulkanError> {
        self.create_buffer(backing, usage, false)
    }
    fn create_buffer(
        &mut self,
        backing: CanonicalBackingRange,
        usage: vk::BufferUsageFlags,
        initialize: bool,
    ) -> Result<BufferId, VulkanError> {
        self.drain_notifications();
        if usage.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
            && !self.device.profile().buffer_device_address
        {
            return Err(VulkanError::InvalidResource(
                "buffer device address not enabled",
            ));
        }
        let usage = usage | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
        let key = backing.byte_mapping_key();
        let candidate = self.buffer_views.get(&key).and_then(|views| {
            views.iter().find_map(|&id| {
                let entry = self.buffer(id).ok()?;
                entry.usage.contains(usage).then_some(id)
            })
        });
        if let Some(id) = candidate {
            if initialize {
                self.ensure_buffer(id)?;
            }
            self.touch(Resource::Buffer(id));
            self.statistics.cache_hits += 1;
            return Ok(id);
        }
        let cpu = CanonicalCpuWriteDependency::capture(&backing).map_err(memory_error)?;
        let native = self.allocate_buffer(backing.size(), usage)?;
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
        let generation = self.buffers[slot].generation + 1;
        let id = BufferId {
            instance: self.instance,
            slot,
            generation,
        };
        self.buffers[slot] = Slot {
            generation,
            entry: Some(BufferEntry {
                streaming_cpu: None,
                streamed: None,
                image_import: None,
                contents: Contents {
                    invalid: vec![(0, backing.size())],
                    backing: backing.clone(),
                    cpu,
                    last_use: None,
                    touched: 0,
                },
                native: Arc::new(native),
                usage,
                #[cfg(feature = "binding-probe")]
                address,
                access: sync::BufferAccess::new(backing.size()),
            }),
        };
        self.index.insert(Resource::Buffer(id), &backing);
        self.buffer_views.entry(key).or_default().push(id);
        if initialize {
            self.ensure_buffer(id)?;
        }
        self.touch(Resource::Buffer(id));
        Ok(id)
    }
    /// Small CPU-authored read snapshots can be consumed directly from the
    /// mapped submission arena. Every version has a distinct, atom-isolated
    /// region retained until timeline completion; GPU-authored data and large
    /// working sets continue through the native device buffer and exact hazards.
    pub(super) fn stream_buffer(
        &mut self,
        id: BufferId,
        offset: u64,
        size: u64,
        alignment: u64,
    ) -> Result<Option<(Arc<AllocatedBuffer>, u64)>, VulkanError> {
        const MAX_STREAM_BYTES: u64 = 64 * 1024;
        self.check_buffer_range(id, offset, size)?;
        if size > MAX_STREAM_BYTES || size > self.device.submission.configuration.arena_bytes {
            return Ok(None);
        }
        let backing = self
            .buffer(id)?
            .contents
            .backing
            .snapshot_subrange(offset, size)
            .map_err(memory_error)?;
        if backing
            .pages()
            .any(|page| self.gpu_owned(page.identity(), page.visibility_state()))
        {
            return Ok(None);
        }
        let work = self.device.work_ticket()?;
        if let Some(entry) = &self.buffers[id.slot].entry
            && let Some(stream) = &entry.streamed
            && stream.work == work
            && stream.window == (offset, size)
            && entry
                .streaming_cpu
                .as_ref()
                .is_some_and(CanonicalCpuWriteDependency::remains_current)
        {
            let binding = (Arc::clone(&stream.buffer), stream.offset);
            self.statistics.cpu_stream_hits += 1;
            self.touch(Resource::Buffer(id));
            let slot = self.device.submission.recording.expect("stream recording");
            self.device.submission.retain_stream_upload(slot);
            return Ok(Some(binding));
        }
        // Descriptor records can outlive a submission. Restrict direct streams
        // to the fixed arenas; retaining overflow allocations in that cache
        // would escape the queue's transient overflow accounting.
        let slot = self.device.submission.begin()?;
        if let Some(arena) = &self.device.submission.slots[slot].upload
            && crate::queue::align_up(
                arena.cursor,
                alignment.max(4).max(arena.buffer.host_alignment()),
            )
            .and_then(|start| start.checked_add(size))
            .is_none_or(|end| end > arena.buffer.size())
        {
            return Ok(None);
        }
        let dependency = if let Some(dependency) = &self.buffer(id)?.streaming_cpu {
            dependency.clone()
        } else {
            let dependency =
                CanonicalCpuWriteDependency::capture(&self.buffer(id)?.contents.backing)
                    .map_err(memory_error)?;
            self.buffers[id.slot].entry.as_mut().unwrap().streaming_cpu = Some(dependency.clone());
            dependency
        };
        // This observer is independent of the persistent native buffer's dirty
        // proof: streaming must not acknowledge data that buffer never imported.
        let mut snapshots = dependency
            .snapshot_with_resolver(&backing, CpuWriteSnapshotSelection::All, 1, &mut |_, _| {
                Err(nixe_memory::VisibilityCoordinatorError::new(
                    "streamed CPU snapshot crossed concurrent GPU ownership",
                ))
            })
            .map_err(memory_error)?;
        let (_, bytes) = snapshots.pop().expect("complete CPU stream snapshot");
        let (slot, slice) = self
            .device
            .submission
            .reserve_upload_aligned(&bytes, alignment.max(4))?;
        self.device.submission.retain_stream_upload(slot);
        let buffer = self.device.submission.slots[slot].upload_lease(slice);
        let work = self.device.work_ticket()?;
        let contents = self.contents_mut(Resource::Buffer(id));
        contents.last_use = Some(work);
        contents.touched = work.tick();
        self.buffers[id.slot].entry.as_mut().unwrap().streamed = Some(StreamedBuffer {
            work,
            window: (offset, size),
            buffer: Arc::clone(&buffer),
            offset: slice.offset,
        });
        self.statistics.cpu_streamed_bytes += size;
        // Queue submission makes these already-flushed host writes visible;
        // no transfer command or in-rendering pipeline barrier is required.
        // https://docs.vulkan.org/spec/latest/chapters/synchronization.html#synchronization-submission-host-writes
        Ok(Some((buffer, slice.offset)))
    }
    /// Warm buffers do one CPU summary check and no page walk or transfer.
    pub fn ensure_buffer(&mut self, id: BufferId) -> Result<(), VulkanError> {
        let entry = self.buffer(id)?;
        if entry.contents.invalid.is_empty() && entry.contents.cpu.remains_current() {
            return Ok(());
        }
        self.constructing.push(Resource::Buffer(id));
        let result = self.refresh_buffer(id);
        self.unpin(Resource::Buffer(id));
        result
    }
    fn refresh_buffer(&mut self, id: BufferId) -> Result<(), VulkanError> {
        self.drain_notifications();
        let backing = self.buffer(id)?.contents.backing.clone();
        let dependency = self.buffer(id)?.contents.cpu.clone();
        let invalid = self.buffer(id)?.contents.invalid.clone();
        let snapshots = self.cpu_snapshots(&backing, &dependency, &invalid)?;
        let contents = &mut self.buffers[id.slot]
            .entry
            .as_mut()
            .expect("buffer")
            .contents;
        contents.invalid.extend(
            snapshots
                .iter()
                .map(|(offset, bytes)| (*offset, bytes.len() as u64)),
        );
        normalize(&mut contents.invalid);
        let invalid = contents.invalid.clone();
        for (offset, bytes) in snapshots {
            {
                let range = backing
                    .snapshot_subrange(offset, bytes.len() as u64)
                    .map_err(memory_error)?;
                let mut visible = self.visibility.pages.lock().unwrap();
                for backing in range.pages() {
                    if let Some(page) = self.index.pages.get_mut(&backing.identity()) {
                        page.clear_writes();
                    }
                    visible.remove(&backing.identity());
                }
            }
            self.upload_buffer(id, offset, &bytes)?;
        }
        for (offset, length) in invalid {
            self.restore_gpu_bytes(id, offset, length)?;
        }
        self.buffers[id.slot]
            .entry
            .as_mut()
            .expect("buffer")
            .contents
            .invalid
            .clear();
        Ok(())
    }
    /// One CPU range retains all ordered occurrences of CPU-visible pages.
    /// Per-segment DirtyPages requests would acknowledge a repeated page before
    /// its other occurrences were copied. Mixed ownership needs a final rearm
    /// for the excluded GPU pages. A fully CPU-visible range's dirty snapshot
    /// already updates its whole observer and retains adaptive streaming state.
    pub(super) fn cpu_snapshots(
        &self,
        backing: &CanonicalBackingRange,
        dependency: &CanonicalCpuWriteDependency,
        invalid: &[(u64, u64)],
    ) -> Result<nixe_memory::CanonicalByteSnapshots, VulkanError> {
        let mut cpu_segments = Vec::new();
        let mut mapping = Vec::<(u64, u64, u64)>::new();
        let mut logical = 0;
        let mut compressed = 0;
        // Classify physical pages once. CPU-produced tiled framebuffers can
        // contain hundreds of logical fragments per page, but need no new
        // topology or offset remapping when every page is CPU-visible.
        let gpu_pages: rustc_hash::FxHashSet<_> = backing
            .pages()
            .filter(|page| self.gpu_owned(page.identity(), page.visibility_state()))
            .map(|page| page.identity())
            .collect();
        let all_cpu = gpu_pages.is_empty();
        if all_cpu && (invalid.is_empty() || invalid == [(0, backing.size())]) {
            // A dirty-only request already returns ordered disjoint windows;
            // a complete invalidation needs one full snapshot, not overlapping
            // dirty and forced copies of the same bytes.
            let selection = if invalid.is_empty() {
                CpuWriteSnapshotSelection::DirtyPages
            } else {
                CpuWriteSnapshotSelection::All
            };
            return dependency.snapshot_with_resolver(backing, selection, 1, &mut |_, _| {
                Err(nixe_memory::VisibilityCoordinatorError::new(
                    "CPU snapshot crossed concurrent GPU ownership; owner must resolve visibility inline",
                ))
            }).map_err(memory_error);
        }
        if all_cpu {
            mapping.push((0, 0, backing.size()));
        } else {
            for segment in backing.segments() {
                if !gpu_pages.contains(&segment.page()) {
                    cpu_segments.push(segment.clone());
                    if let Some(last) = mapping.last_mut()
                        && last.0 + last.2 == compressed
                        && last.1 + last.2 == logical
                    {
                        last.2 += segment.size();
                    } else {
                        mapping.push((compressed, logical, segment.size()));
                    }
                    compressed += segment.size();
                }
                logical += segment.size();
            }
        }
        let mut ranges = Vec::new();
        for &(start, size) in invalid {
            let end = start + size;
            // Mapping runs are ordered even when the invalid intervals are not.
            // Start at the first overlap instead of checking every run for each
            // interval in a mixed CPU/GPU view.
            let first = mapping.partition_point(|&(_, logical, length)| logical + length <= start);
            for &(_, logical, length) in &mapping[first..] {
                if logical >= end {
                    break;
                }
                let begin = start.max(logical);
                let finish = end.min(logical + length);
                if begin < finish {
                    ranges.push((
                        begin,
                        backing
                            .snapshot_subrange(begin, finish - begin)
                            .map_err(memory_error)?,
                    ));
                }
            }
        }
        let cpu_range = if all_cpu {
            Some(backing.clone())
        } else if cpu_segments.is_empty() {
            None
        } else {
            Some(CanonicalBackingRange::new(cpu_segments).map_err(memory_error)?)
        };
        let mut requests = Vec::with_capacity(
            ranges.len() + usize::from(cpu_range.is_some()) + usize::from(!all_cpu),
        );
        if let Some(range) = &cpu_range {
            requests.push(CpuWriteSnapshotRequest {
                dependency,
                range,
                selection: CpuWriteSnapshotSelection::DirtyPages,
                alignment: 1,
            });
        }
        for (_, range) in &ranges {
            requests.push(CpuWriteSnapshotRequest {
                dependency,
                range,
                selection: CpuWriteSnapshotSelection::All,
                alignment: 1,
            });
        }
        if !all_cpu {
            requests.push(CpuWriteSnapshotRequest {
                dependency,
                range: backing,
                selection: CpuWriteSnapshotSelection::Rearm,
                alignment: 1,
            });
        }
        let snapshots = CanonicalCpuWriteDependency::snapshot_batch_with_resolver(&requests, &mut |_, _| Err(nixe_memory::VisibilityCoordinatorError::new("CPU snapshot crossed concurrent GPU ownership; owner must resolve visibility inline"))).map_err(memory_error)?;
        let mut results = Vec::new();
        let mut responses = snapshots.into_iter();
        if all_cpu {
            results.extend(responses.next().expect("CPU dirty request"));
        } else if cpu_range.is_some() {
            // DirtyPages produces ordered, disjoint logical windows. Both
            // those compressed windows and this mapping advance monotonically.
            let mut first_mapping = 0;
            for (offset, bytes) in responses.next().expect("CPU dirty request") {
                let end = offset + bytes.len() as u64;
                while first_mapping < mapping.len()
                    && mapping[first_mapping].0 + mapping[first_mapping].2 <= offset
                {
                    first_mapping += 1;
                }
                for &(compressed, logical, size) in &mapping[first_mapping..] {
                    if compressed >= end {
                        break;
                    }
                    let start = offset.max(compressed);
                    let finish = end.min(compressed + size);
                    if start < finish {
                        results.push((
                            logical + start - compressed,
                            bytes[(start - offset) as usize..(finish - offset) as usize].into(),
                        ));
                    }
                }
            }
        }
        for ((base, _), parts) in ranges.iter().zip(responses) {
            for (offset, bytes) in parts {
                results.push((base + offset, bytes));
            }
        }
        results.sort_unstable_by_key(|(offset, _)| *offset);
        let mut merged: Vec<(u64, Vec<u8>)> = Vec::with_capacity(results.len());
        for (offset, bytes) in results {
            if let Some((start, previous)) = merged.last_mut()
                && offset <= *start + previous.len() as u64
            {
                let overlap =
                    (*start + previous.len() as u64 - offset).min(bytes.len() as u64) as usize;
                if overlap < bytes.len() {
                    // Retain capacity while accumulating adjacent windows.
                    // Rebuilding a boxed prefix at each join copied O(n^2)
                    // bytes for a long sequence of small fragments.
                    previous.extend_from_slice(&bytes[overlap..]);
                }
            } else {
                merged.push((offset, bytes.into_vec()));
            }
        }
        Ok(merged
            .into_iter()
            .map(|(offset, bytes)| (offset, bytes.into_boxed_slice()))
            .collect())
    }
    fn restore_gpu_bytes(
        &mut self,
        id: BufferId,
        offset: u64,
        length: u64,
    ) -> Result<(), VulkanError> {
        if let Some(import) = self.buffer(id)?.image_import.clone()
            && self.image_is_current_for_buffer_import(import.source)
        {
            self.statistics.image_import_plan_hits += 1;
            let end = offset + length;
            let first = import
                .copies
                .partition_point(|&(_, to, size)| to + size <= offset);
            for &(from, to, size) in &import.copies[first..] {
                if to >= end {
                    break;
                }
                let begin = to.max(offset);
                let stop = (to + size).min(end);
                self.copy_source(
                    Resource::Image(import.source),
                    from + begin - to,
                    id,
                    begin,
                    stop - begin,
                )?;
            }
            return Ok(());
        }
        let backing = &self.buffer(id)?.contents.backing;
        let native_source = backing.pages().next().and_then(|first| {
            self.index
                .pages
                .get(&first.identity())?
                .resources()
                .find_map(|resource| {
                    let Resource::Image(image) = *resource else {
                        return None;
                    };
                    let entry = self.image(image).ok()?;
                    let [plane] = entry.planes.as_slice() else {
                        return None;
                    };
                    (entry.description.samples == vk::SampleCountFlags::TYPE_1
                        && entry.decoded.is_none()
                        && plane.aspect == vk::ImageAspectFlags::COLOR
                        && plane.offset == 0
                        && plane.row_pitch == plane.row_bytes()
                        && plane.end() == Some(backing.size())
                        && entry.discarded.is_empty()
                        && entry.contents.invalid.is_empty()
                        && entry.contents.cpu.remains_current()
                        && same_bytes(&entry.contents.backing, backing))
                    .then_some(image)
                })
        });
        if let Some(source) = native_source {
            // An up-to-date dense native image already proves every logical
            // byte of this buffer. Copy its requested window directly, rather
            // than resolving the same tiled physical permutation per frame.
            return self.copy_source(Resource::Image(source), offset, id, offset, length);
        }
        enum Source {
            Device(Resource, u64),
            Baseline(CanonicalPageId, u64),
        }
        struct Run {
            destination: u64,
            length: u64,
            source: Source,
        }
        let backing = self.buffer(id)?.contents.backing.clone();
        let range = backing
            .snapshot_subrange(offset, length)
            .map_err(memory_error)?;
        let end = offset + length;
        let mut runs = Vec::new();
        for backing_page in range.pages() {
            let physical = backing_page.identity();
            if !self.gpu_owned(physical, backing_page.visibility_state()) {
                continue;
            }
            let page = self.index.pages.get_mut(&physical).expect("indexed buffer");
            page.materialize_writes();
            for spans in page.bindings_for(Resource::Buffer(id)) {
                // Immutable buffer bindings and current producers are both in
                // physical order. Merge them once per page, instead of a binary
                // search and ownership lock for every tiny tiled fragment.
                let mut first = 0;
                for span in spans.exact_iter() {
                    let logical = span.logical.max(offset);
                    let finish = (span.logical + span.end - span.start).min(end);
                    if logical >= finish {
                        continue;
                    }
                    let start = span.start + logical - span.logical;
                    let stop = start + finish - logical;
                    while first < page.writes.len() && page.writes[first].end <= start {
                        first += 1;
                    }
                    let mut cursor = start;
                    for write in &page.writes[first..] {
                        if write.start >= stop {
                            break;
                        }
                        let begin = write.start.max(start);
                        let finish = write.end.min(stop);
                        if cursor < begin {
                            runs.push(Run {
                                destination: logical + cursor - start,
                                length: begin - cursor,
                                source: Source::Baseline(physical, cursor),
                            });
                        }
                        runs.push(Run {
                            destination: logical + begin - start,
                            length: finish - begin,
                            source: Source::Device(
                                write.source,
                                write.offset + begin - write.start,
                            ),
                        });
                        cursor = finish;
                    }
                    if cursor < stop {
                        runs.push(Run {
                            destination: logical + cursor - start,
                            length: stop - cursor,
                            source: Source::Baseline(physical, cursor),
                        });
                    }
                }
            }
        }
        // Canonical aliases may repeat physical bytes. Emit in logical order,
        // preserving ordered in-place buffer snapshots and contiguous copies.
        runs.sort_unstable_by_key(|run| run.destination);
        if offset == 0
            && length == backing.size()
            && let Some(Run {
                source: Source::Device(Resource::Image(image), _),
                ..
            }) = runs.first()
            && self.image_is_current_for_buffer_import(*image)
        {
            let image = *image;
            let mut copies = Vec::<(u64, u64, u64)>::new();
            let mut cursor = 0;
            for run in &runs {
                let Source::Device(Resource::Image(source), from) = run.source else {
                    break;
                };
                if source != image || run.destination != cursor {
                    break;
                }
                cursor += run.length;
                if let Some((previous, to, size)) = copies.last_mut()
                    && *previous + *size == from
                    && *to + *size == run.destination
                {
                    *size += run.length;
                } else {
                    copies.push((from, run.destination, run.length));
                }
            }
            // Cache small native copy plans, never a second expanded texel map.
            // Larger/heterogeneous reconstructions retain exact demand handling.
            if cursor == length && std::mem::size_of_val(copies.as_slice()) <= 16 * 1024 {
                self.buffers[id.slot].entry.as_mut().unwrap().image_import =
                    Some(ImageBufferImport {
                        source: image,
                        copies: copies.into(),
                    });
            }
        }
        let mut baseline = None;
        let mut copy: Option<(Resource, u64, u64, u64)> = None;
        for run in runs {
            match run.source {
                Source::Baseline(page, offset) => {
                    self.flush_restore_copy(id, &mut copy)?;
                    self.append_baseline(
                        id,
                        page,
                        offset,
                        run.length,
                        run.destination,
                        &mut baseline,
                    )?;
                }
                Source::Device(source, from) => {
                    self.flush_baseline(id, &mut baseline)?;
                    if let Some((previous, offset, to, size)) = &mut copy
                        && *previous == source
                        && source != Resource::Buffer(id)
                        && *offset + *size == from
                        && *to + *size == run.destination
                    {
                        *size += run.length;
                    } else {
                        self.flush_restore_copy(id, &mut copy)?;
                        copy = Some((source, from, run.destination, run.length));
                    }
                }
            }
        }
        self.flush_restore_copy(id, &mut copy)?;
        self.flush_baseline(id, &mut baseline)
    }
    fn image_is_current_for_buffer_import(&self, id: ImageId) -> bool {
        self.image(id).is_ok_and(|entry| {
            entry.description.samples == vk::SampleCountFlags::TYPE_1
                && entry.decoded.is_none()
                && entry.discarded.is_empty()
                && entry.contents.invalid.is_empty()
                && entry.contents.cpu.remains_current()
        })
    }
    /// Canonical page/GOB boundaries do not require separate native commands.
    /// Merge only simultaneous contiguous source and destination runs; in-place
    /// buffer moves keep their original ordered snapshot semantics.
    fn flush_restore_copy(
        &mut self,
        destination: BufferId,
        pending: &mut Option<(Resource, u64, u64, u64)>,
    ) -> Result<(), VulkanError> {
        if let Some((source, offset, destination_offset, size)) = pending.take() {
            self.copy_source(source, offset, destination, destination_offset, size)?;
        }
        Ok(())
    }
    /// Adjacent logical holes share one bounded upload even when the guest
    /// mapping has millions of tiny physical fragments. Flush before a GPU
    /// overlay to preserve the original ordered reconstruction semantics.
    fn append_baseline(
        &mut self,
        id: BufferId,
        page: CanonicalPageId,
        offset: u64,
        size: u64,
        destination: u64,
        pending: &mut Option<(u64, Vec<u8>)>,
    ) -> Result<(), VulkanError> {
        let limit = self.device.submission.configuration.arena_bytes;
        let mut copied = 0;
        while copied < size {
            if pending.as_ref().is_some_and(|(start, bytes)| {
                *start + bytes.len() as u64 != destination + copied || bytes.len() as u64 == limit
            }) {
                self.flush_baseline(id, pending)?;
            }
            let (_, bytes) = pending.get_or_insert_with(|| (destination + copied, Vec::new()));
            let length = (size - copied).min(limit - bytes.len() as u64);
            {
                let pages = self.visibility.pages.lock().unwrap();
                let page = pages.get(&page).ok_or(VulkanError::InvalidResource(
                    "GPU authority has no untouched CPU page baseline",
                ))?;
                bytes.extend_from_slice(
                    &page.bytes[(offset + copied) as usize..(offset + copied + length) as usize],
                );
            }
            copied += length;
        }
        Ok(())
    }
    fn flush_baseline(
        &mut self,
        id: BufferId,
        pending: &mut Option<(u64, Vec<u8>)>,
    ) -> Result<(), VulkanError> {
        if let Some((offset, bytes)) = pending.take() {
            self.upload_buffer(id, offset, &bytes)?;
        }
        Ok(())
    }
}

impl VulkanResidency {
    /// Resolve once when binding identity changes; content synchronization remains
    /// the residency recorder's responsibility on every consumed access.
    pub fn buffer_lease(&self, id: BufferId) -> Result<Arc<AllocatedBuffer>, VulkanError> {
        Ok(Arc::clone(&self.buffer(id)?.native))
    }
}
