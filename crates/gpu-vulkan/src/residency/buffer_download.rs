//! Ordered buffer downloads retain physical destinations and publication masks.
//! https://github.com/zortos293/Lichii/blob/b4b830c8faf1fb5d7b2ad1b892117f4cfd708546/src/video_core/buffer_cache/buffer_cache.h#L539-L841
use super::*;
use crate::transfer::host_read_barrier;

/// A publication boundary, independent of the native tick (several copies can
/// share a submission). Read prefixes consume earlier reactive copies; DMA
/// boundaries identify individual stores in the texture publication order.
#[derive(Clone, Copy)]
pub(super) struct DownloadEnd {
    serial: u64,
    work: WorkTicket,
}
impl DownloadEnd {
    pub(super) fn work(self) -> WorkTicket {
        self.work
    }
}
pub(super) struct Download {
    ticket: crate::ReadbackTicket,
    publications: Vec<Publication>,
    write_ranges: RangeSet,
}
// Reactive reads retain the requested snapshot even after a newer GPU write.
// Fence copies instead discard bytes superseded by either CPU or GPU writes.
pub(super) struct ReadDownload {
    download: Download,
    clean_ranges: RangeSet,
}
struct Publication {
    address: u64,
    bytes: u64,
    inputs: Vec<(u64, nixe_memory::AllocationRange)>,
    offset: usize,
}
// Retain logical offsets across holes in device memory and pin the actual
// physical destinations, rather than resolving them again at publication.
fn download_inputs(
    mapping: &PhysicalMapping,
) -> Result<Vec<(u64, nixe_memory::AllocationRange)>, nixe_memory::MemoryError> {
    let mut inputs = Vec::new();
    for span in mapping.spans() {
        let mut offset = span.logical;
        for input in mapping.pin(span.logical, span.bytes)? {
            let bytes = input.bytes();
            inputs.push((offset, input));
            offset += bytes;
        }
    }
    Ok(inputs)
}
impl Download {
    fn new(ticket: crate::ReadbackTicket, publications: Vec<Publication>) -> Self {
        let mut ranges = RangeSet::default();
        for publication in &publications {
            ranges.insert(publication.address, publication.bytes);
        }
        Self {
            ticket,
            publications,
            write_ranges: ranges,
        }
    }
    fn bounds(&self) -> (u64, u64) {
        let start = self
            .publications
            .iter()
            .map(|part| part.address)
            .min()
            .unwrap();
        let end = self
            .publications
            .iter()
            .map(|part| part.address + part.bytes)
            .max()
            .unwrap();
        (start, end)
    }
}
impl ReadDownload {
    fn new(download: Download) -> Self {
        Self {
            clean_ranges: download.write_ranges.clone(),
            download,
        }
    }
    pub(super) fn image_copy(
        ticket: crate::ReadbackTicket,
        address: u64,
        bytes: u64,
        inputs: Vec<(u64, nixe_memory::AllocationRange)>,
    ) -> Self {
        Self::new(Download::new(
            ticket,
            vec![Publication {
                address,
                bytes,
                inputs,
                offset: 0,
            }],
        ))
    }
}
// Bounded, sorted pages learned from CPU reads of GPU-written data. Reports
// outside these pages need no extra buffer-cache bookkeeping.
// https://github.com/zortos293/Lichii/blob/b4b830c8faf1fb5d7b2ad1b892117f4cfd708546/src/video_core/buffer_cache/buffer_cache.h#L876-L917
pub(super) struct ReadBackPage {
    address: u64,
    last_read: u64,
    written: bool,
}
impl BufferCache {
    pub(super) fn read_needs_download(&self, address: u64, bytes: u64) -> bool {
        (self.download_policy == BufferDownloadPolicy::OnRead
            && self.gpu_modified_ranges.intersects(address, bytes))
            || self.pending_read(address, bytes).is_some()
    }
    /// GPU ownership does not imply CPU consumption. Cached pages already trap
    /// native reads; a separate reference also protects partially CPU-dirty pages.
    /// https://github.com/zortos293/Lichii/blob/646b5d17135a2b2701ccc7a959afd0bbbf2f6160/src/video_core/buffer_cache/buffer_cache_base.h
    pub(super) fn queue_fence_download(
        &mut self,
        address: u64,
        bytes: u64,
    ) -> Result<(), VulkanError> {
        if self.download_policy == BufferDownloadPolicy::AtFence {
            self.pending_download_ranges.insert(address, bytes);
            return Ok(());
        }
        let page = nixe_memory::PAGE_SIZE;
        let begin = address / page * page;
        let end = (address + bytes).div_ceil(page) * page;
        if !self.read_trap_pages.contains(begin, end - begin) {
            let mut next = begin;
            let mut added = smallvec::SmallVec::<[(u64, u64); 4]>::new();
            for (at, size) in self.read_trap_pages.intersections(begin, end - begin) {
                if next < at {
                    added.push((next, at - next));
                }
                next = at + size;
            }
            if next < end {
                added.push((next, end - next));
            }
            for (at, size) in added {
                self.device_memory.update_pages_cached_count(at, size, 1)?;
                self.read_trap_pages.insert(at, size);
            }
        }
        // Notify only for newly armed pages. The read callbacks consume these
        // marks independently of the references keeping mixed CPU/GPU pages cached.
        // https://github.com/zortos293/Lichii/blob/646b5d17135a2b2701ccc7a959afd0bbbf2f6160/src/video_core/buffer_cache/read_trap_tracker.h
        if !self.read_armed_ranges.contains(begin, end - begin) {
            self.read_armed_ranges.insert(begin, end - begin);
            self.device_memory.invalidate_read_areas();
        }
        Ok(())
    }
    pub(super) fn release_clean_read_traps(&mut self, address: u64, bytes: u64) {
        let page = nixe_memory::PAGE_SIZE;
        let begin = address / page * page;
        let end = (address + bytes).div_ceil(page) * page;
        let mut clean = RangeSet::from_range(begin, end - begin);
        for (at, size) in self.gpu_modified_ranges.intersections(begin, end - begin) {
            let start = at / page * page;
            clean.subtract(start, (at + size).div_ceil(page) * page - start);
        }
        for (at, size) in clean.iter() {
            for (start, length) in self.read_trap_pages.intersections(at, size) {
                self.device_memory
                    .update_pages_cached_count(start, length, -1)
                    .expect("release GPU read protection");
            }
            self.read_trap_pages.subtract(at, size);
            self.read_armed_ranges.subtract(at, size);
            self.memory_tracker.clear_gpu(at, size);
        }
    }
    pub(super) fn note_read_back(&mut self, address: u64, bytes: u64) {
        const MAX_PAGES: usize = 64;
        let page_size = nixe_memory::PAGE_SIZE;
        let begin = address / page_size * page_size;
        let end = (address + bytes).div_ceil(page_size) * page_size;
        let modified = self.gpu_modified_ranges.intersects(begin, end - begin);
        if !modified && self.read_back_pages.is_empty() {
            return;
        }
        for page in (begin..end).step_by(page_size as usize) {
            let index = match self
                .read_back_pages
                .binary_search_by_key(&page, |entry| entry.address)
            {
                Ok(index) => index,
                Err(_) if !modified => continue,
                Err(mut index) => {
                    if self.read_back_pages.len() == MAX_PAGES {
                        let oldest = self
                            .read_back_pages
                            .iter()
                            .enumerate()
                            .min_by_key(|(_, entry)| entry.last_read)
                            .unwrap()
                            .0;
                        self.read_back_pages.remove(oldest);
                        if oldest < index {
                            index -= 1;
                        }
                    }
                    self.read_back_pages.insert(
                        index,
                        ReadBackPage {
                            address: page,
                            last_read: 0,
                            written: true,
                        },
                    );
                    index
                }
            };
            let entry = &mut self.read_back_pages[index];
            if entry.written {
                entry.written = false;
                self.read_back_reads += 1;
                entry.last_read = self.read_back_reads;
            }
        }
    }
    pub(super) fn note_report_pending(&mut self, address: u64, report: &Arc<query::Query>) -> bool {
        let page = address / nixe_memory::PAGE_SIZE * nixe_memory::PAGE_SIZE;
        if self
            .read_back_pages
            .binary_search_by_key(&page, |entry| entry.address)
            .is_err()
        {
            return false;
        }
        self.pending_reports.push(Arc::clone(report));
        true
    }
    pub(super) fn mark_written(
        &mut self,
        id: BufferId,
        offset: u64,
        bytes: u64,
    ) -> Result<(), VulkanError> {
        let address = self.buffers[id]
            .device_address
            .ok_or(VulkanError::InvalidResource(
                "GPU buffer writes require a mapped device-address range",
            ))?
            + offset;
        #[cfg(feature = "performance-counters")]
        nixe_gpu::trace::detail_event(
            "readback_producer",
            address,
            bytes,
            self.buffers[id].write_tick,
        );
        self.note_uniform_gpu_write(address, bytes);
        self.memory_tracker.gpu_write(address, bytes);
        self.note_command_write(id, address, bytes);
        self.outdate_downloads(address, bytes);
        self.gpu_modified_ranges.insert(address, bytes);
        self.queue_fence_download(address, bytes)?;
        Ok(())
    }
    pub(super) fn outdate_downloads(&mut self, address: u64, bytes: u64) {
        // A later GPU producer supersedes ordinary fence publications. Reactive
        // readers still receive their captured bytes, but cannot clean this write.
        for download in self.buffer_downloads.iter_mut().flatten() {
            download.write_ranges.subtract(address, bytes);
        }
        for (_, download) in self
            .read_downloads
            .iter_mut()
            .chain(&mut self.dma_downloads)
        {
            download.clean_ranges.subtract(address, bytes);
        }
        self.read_download_ranges.subtract(address, bytes);
        for report in &self.pending_reports {
            report.outdate(address, bytes);
        }
        let page_size = nixe_memory::PAGE_SIZE;
        let begin = address / page_size * page_size;
        let first = self
            .read_back_pages
            .partition_point(|entry| entry.address < begin);
        for entry in self.read_back_pages[first..]
            .iter_mut()
            .take_while(|entry| entry.address < address + bytes)
        {
            entry.written = true;
        }
    }
    fn clear_fence_downloads(&mut self, address: u64, bytes: u64) {
        for download in self.buffer_downloads.iter_mut().flatten() {
            download.write_ranges.subtract(address, bytes);
        }
        self.pending_download_ranges.subtract(address, bytes);
    }
    pub(super) fn clear_download(&mut self, address: u64, bytes: u64) {
        self.clear_fence_downloads(address, bytes);
        for (_, download) in self
            .read_downloads
            .iter_mut()
            .chain(&mut self.dma_downloads)
        {
            download.download.write_ranges.subtract(address, bytes);
            download.clean_ranges.subtract(address, bytes);
        }
        self.read_download_ranges.subtract(address, bytes);
    }
    fn next_download_end(&mut self, work: WorkTicket) -> DownloadEnd {
        self.download_serial = self
            .download_serial
            .checked_add(1)
            .expect("buffer download serial exhausted");
        DownloadEnd {
            serial: self.download_serial,
            work,
        }
    }
    fn enqueue_read_download(&mut self, download: Download) -> DownloadEnd {
        let end = self.next_download_end(download.ticket.work());
        // A clean proof may be reused through any device alias. Invalidate
        // the captured publication destinations before adding the snapshot.
        for publication in &download.publications {
            for (_, input) in &publication.inputs {
                self.device_memory.invalidate_owner_reads(input);
            }
        }
        self.read_download_ranges
            .extend(download.write_ranges.iter());
        #[cfg(feature = "performance-counters")]
        nixe_gpu::trace::detail_event("readback_prefix", end.serial, end.work.tick(), 1);
        self.read_downloads
            .push_back((end, ReadDownload::new(download)));
        self.pending_read_count
            .fetch_add(1, std::sync::atomic::Ordering::Release);
        end
    }
    pub(super) fn enqueue_dma_download(&mut self, download: ReadDownload) -> DownloadEnd {
        let end = self.next_download_end(download.download.ticket.work());
        self.dma_downloads.push_back((end, download));
        end
    }
    pub(super) fn publish_dma_download(&mut self, end: DownloadEnd) -> Result<(), VulkanError> {
        self.publish_completed_reads()?;
        let index = self
            .dma_downloads
            .iter()
            .rposition(|(pending, _)| pending.serial == end.serial)
            .ok_or(VulkanError::InvalidResource(
                "missing DMA download publication",
            ))?;
        let (_, download) = self.dma_downloads.remove(index).unwrap();
        self.publish_read_download(download)
    }
    pub(crate) fn visibility_work(&self) -> Option<WorkTicket> {
        self.buffer_downloads
            .back()
            .and_then(Option::as_ref)
            .map(|download| download.ticket.work())
    }
    /// Only reactive reads participate here. Ordinary fence publication retains
    /// its existing preemptive CPU-read policy.
    pub(super) fn pending_read(&self, address: u64, bytes: u64) -> Option<DownloadEnd> {
        if self
            .pending_read_count
            .load(std::sync::atomic::Ordering::Relaxed)
            == 0
        {
            return None;
        }
        let pending = |address, bytes| {
            self.read_downloads
                .iter()
                .rev()
                .find_map(|(end, download)| {
                    download
                        .download
                        .write_ranges
                        .intersects(address, bytes)
                        .then_some(*end)
                })
        };
        let mut result = pending(address, bytes);
        // All device aliases share the same physical RAM destination. A write
        // through another mapping must also follow earlier reactive copies.
        self.device_memory
            .apply_device_aliases(address, bytes, |alias, bytes| {
                if let Some(end) = pending(alias, bytes)
                    && result.is_none_or(|previous| previous.serial < end.serial)
                {
                    result = Some(end);
                }
            });
        result
    }
    pub(super) fn publish_reads_through(&mut self, end: DownloadEnd) -> Result<(), VulkanError> {
        while self
            .read_downloads
            .front()
            .is_some_and(|(next, _)| next.serial <= end.serial)
        {
            let (end, download) = self.read_downloads.pop_front().unwrap();
            #[cfg(feature = "performance-counters")]
            nixe_gpu::trace::detail_event("readback_finish_prefix", end.serial, end.work.tick(), 1);
            #[cfg(not(feature = "performance-counters"))]
            let _ = end;
            // Retire the reader even if publication fails during cancellation.
            let result = self.publish_read_download(download);
            self.pending_read_count
                .fetch_sub(1, std::sync::atomic::Ordering::Release);
            result?;
        }
        Ok(())
    }
    pub(super) fn publish_completed_reads(&mut self) -> Result<(), VulkanError> {
        while let Some((end, download)) = self.read_downloads.front() {
            if !download.download.ticket.is_complete()? {
                break;
            }
            self.publish_reads_through(*end)?;
        }
        Ok(())
    }
    pub(crate) fn pop_async_flush(&mut self) -> Result<(), VulkanError> {
        self.publish_completed_reads()?;
        if let Some(Some(download)) = self.buffer_downloads.pop_front() {
            self.publish_fence_download(download)?;
        }
        Ok(())
    }
    // The caller holds the landing guard through publication and ownership
    // updates. Staging remains pinned until all of its borrowed bytes are used.
    fn write_download(&mut self, download: Download) -> Result<RangeSet, VulkanError> {
        for (address, bytes) in download.write_ranges.iter() {
            self.note_uniform_gpu_write(address, bytes);
        }
        #[cfg(feature = "performance-counters")]
        let _trace =
            nixe_gpu::trace::span("readback_publication", download.ticket.work().tick(), 0, 0);
        let staging = download.ticket.completed()?;
        let memory = self.device_memory.machine();
        for publication in download.publications {
            self.native_readback_bytes += publication.bytes;
            let bytes =
                &staging[publication.offset..publication.offset + publication.bytes as usize];
            for &(offset, ref input) in &publication.inputs {
                let offset = offset as usize;
                let length = input.bytes() as usize;
                let address = publication.address + offset as u64;
                if !download.write_ranges.intersects(address, length as u64) {
                    continue;
                }
                memory.write_download(input, &bytes[offset..offset + length], || {
                    Ok(download
                        .write_ranges
                        .intersections(address, length as u64)
                        .map(|(start, size)| {
                            #[cfg(feature = "performance-counters")]
                            nixe_gpu::trace::detail_event("readback_published", start, size, 0);
                            self.canonical_readback_bytes += size;
                            (start - address) as usize..(start - address + size) as usize
                        })
                        .collect())
                })?;
            }
        }
        Ok(download.write_ranges)
    }
    fn publish_fence_download(&mut self, mut download: Download) -> Result<(), VulkanError> {
        let memory = self.device_memory.machine().clone();
        let (begin, end) = download.bounds();
        let landing = memory.begin_gpu_download(begin, end);
        for (address, bytes) in memory.pending_gpu_dirty_memory().iter() {
            #[cfg(feature = "performance-counters")]
            if download.write_ranges.intersects(address, bytes) {
                memory.note_download_write_overlap();
            }
            download.write_ranges.subtract(address, bytes);
        }
        let published = self.write_download(download)?;
        // A mapping transition can hold the alias lock while waiting for a
        // native store that overlaps this landing. Release the stores before
        // changing page protections, with the completed bytes already visible.
        drop(landing);
        for (address, bytes) in published.iter() {
            self.gpu_modified_ranges.subtract(address, bytes);
            self.release_clean_read_traps(address, bytes);
        }
        Ok(())
    }
    fn publish_read_download(&mut self, mut read: ReadDownload) -> Result<(), VulkanError> {
        let memory = self.device_memory.machine().clone();
        let (begin, end) = read.download.bounds();
        let landing = memory.begin_gpu_download(begin, end);
        for (address, bytes) in memory.pending_gpu_dirty_memory().iter() {
            #[cfg(feature = "performance-counters")]
            if read.download.write_ranges.intersects(address, bytes) {
                memory.note_download_write_overlap();
            }
            read.download.write_ranges.subtract(address, bytes);
            read.clean_ranges.subtract(address, bytes);
            self.read_download_ranges.subtract(address, bytes);
        }
        self.write_download(read.download)?;
        drop(landing);
        // Keep ownership when an older report can still overwrite these bytes.
        for report in &self.pending_reports {
            if let Some((address, bytes)) = report.pending_overwrite() {
                read.clean_ranges.subtract(address, bytes);
                self.read_download_ranges.subtract(address, bytes);
            }
        }
        for (address, bytes) in read.clean_ranges.iter() {
            self.gpu_modified_ranges.subtract(address, bytes);
            self.read_download_ranges.subtract(address, bytes);
            self.release_clean_read_traps(address, bytes);
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum DownloadKind {
    Fence,
    Read,
}

impl Residency<'_> {
    /// Record exact dirty intervals in one staging reservation, grouped by
    /// source buffer. Completion is consumed separately, directly from staging.
    fn record_buffer_downloads(
        &mut self,
        ranges: impl IntoIterator<Item = (u64, u64)>,
        kind: DownloadKind,
    ) -> Result<Option<Download>, VulkanError> {
        let mut publications = Vec::new();
        let mut groups: Vec<(BufferId, Vec<vk::BufferCopy>)> = Vec::new();
        let mut total = 0u64;
        let page = nixe_memory::PAGE_SIZE;
        let mut previous_end = 0;
        for (address, bytes) in ranges {
            #[cfg(feature = "performance-counters")]
            nixe_gpu::trace::detail_event("readback_requested", address, bytes, kind as u64);
            if bytes == 0 {
                continue;
            }
            // Coalesce tracker pages shared by adjacent exact dirty intervals.
            let mut at = (address / page * page).max(previous_end);
            let end = (address + bytes).div_ceil(page) * page;
            previous_end = end;
            while at < end {
                let Some(id) = self.buffer_cache.buffer_index.at(at) else {
                    at = ((at / 65536 + 1) * 65536).min(end);
                    continue;
                };
                let entry = &self.buffer_cache.buffers[id];
                let base = entry.device_address.expect("DAddr buffer index entry");
                let stop = (base + entry.native.size()).min(end);
                #[cfg(feature = "performance-counters")]
                let provenance = {
                    use ash::vk::Handle;
                    (entry.native.handle.as_raw(), entry.write_tick)
                };
                // GPU ownership is exact. CPU-dirty neighbours on this page do
                // not make the remaining GPU bytes safe to read from RAM.
                {
                    let (start, size) = (at, stop - at);
                    #[cfg(feature = "performance-counters")]
                    for (at, count) in self
                        .buffer_cache
                        .read_download_ranges
                        .intersections(start, size)
                    {
                        nixe_gpu::trace::detail_event("readback_covered", at, count, kind as u64);
                    }
                    match kind {
                        DownloadKind::Fence => {}
                        DownloadKind::Read => {
                            self.buffer_cache.clear_fence_downloads(start, size);
                            // Texture-owned DMA stores must not restore older
                            // bytes after this read's snapshot has landed.
                            for (_, download) in &mut self.buffer_cache.dma_downloads {
                                download.download.write_ranges.subtract(start, size);
                                download.clean_ranges.subtract(start, size);
                            }
                        }
                    }
                    let mut add_copy = |start: u64, size: u64| -> Result<(), VulkanError> {
                        #[cfg(feature = "performance-counters")]
                        {
                            nixe_gpu::trace::detail_event(
                                "readback_source",
                                start,
                                provenance.0,
                                provenance.1,
                            );
                        }
                        let footprint = self.runtime.device_memory.read_mapping(start, size)?;
                        let inputs = download_inputs(&footprint)?;
                        let target = total;
                        total = crate::queue::align_up(
                            total.checked_add(size).ok_or(VulkanError::InvalidResource(
                                "buffer download size overflow",
                            ))?,
                            64,
                        )
                        .ok_or(VulkanError::InvalidResource(
                            "buffer download alignment overflow",
                        ))?;
                        let copy = vk::BufferCopy::default()
                            .src_offset(start - base)
                            .dst_offset(target)
                            .size(size);
                        if let Some((_, copies)) =
                            groups.last_mut().filter(|(buffer, _)| *buffer == id)
                        {
                            copies.push(copy);
                        } else {
                            groups.push((id, vec![copy]));
                        }
                        publications.push(Publication {
                            address: start,
                            bytes: size,
                            inputs,
                            offset: target as usize,
                        });
                        Ok(())
                    };
                    for (start, size) in self
                        .buffer_cache
                        .gpu_modified_ranges
                        .intersections(start, size)
                    {
                        if matches!(kind, DownloadKind::Read) {
                            let mut next = start;
                            for (covered, count) in self
                                .buffer_cache
                                .read_download_ranges
                                .intersections(start, size)
                            {
                                if next < covered {
                                    add_copy(next, covered - next)?;
                                }
                                next = next.max(covered + count);
                            }
                            if next < start + size {
                                add_copy(next, start + size - next)?;
                            }
                        } else {
                            add_copy(start, size)?;
                        }
                    }
                }
                at = stop;
            }
        }
        if publications.is_empty() {
            return Ok(None);
        }
        let (slice, ticket) = self
            .runtime
            .device
            .submission
            .reserve_readback_aligned(total, 64)?;
        #[cfg(feature = "performance-counters")]
        for part in &publications {
            nixe_gpu::trace::detail_event(
                "readback_copy",
                part.address,
                part.bytes,
                ticket.work().tick(),
            );
            if nixe_gpu::trace::detailed_active() {
                for (offset, input) in &part.inputs {
                    let mut logical = part.address + offset;
                    for extent in input.physical_extents() {
                        nixe_gpu::trace::detail_event(
                            "readback_physical",
                            logical,
                            extent.start.get(),
                            extent.bytes,
                        );
                        logical += extent.bytes;
                    }
                }
            }
        }
        let target = self.runtime.device.submission.staging_buffer(&slice).handle;
        let mut sources = Vec::with_capacity(groups.len());
        for (id, copies) in &mut groups {
            let mut source = None;
            for copy in copies {
                source = Some(self.use_buffer(*id, copy.src_offset, copy.size)?);
                copy.dst_offset += slice.offset;
            }
            sources.push(source.expect("nonempty copy group"));
        }
        self.runtime.barriers.emit(&mut self.runtime.device)?;
        unsafe {
            self.runtime.device.dispatch.copy_barrier(false);
        }
        for ((_, copies), source) in groups.iter().zip(sources) {
            unsafe {
                self.runtime
                    .device
                    .dispatch
                    .cmd_copy_buffer(source, target, copies, false);
            }
        }
        unsafe {
            self.runtime.device.dispatch.copy_barrier(true);
        }
        host_read_barrier(&mut self.runtime.device, target, slice.offset, total);
        Ok(Some(Download::new(ticket, publications)))
    }
    /// Copy only gaps in current reactive snapshots. Fence downloads retain a
    /// separate publication order and never become dependencies of this reader.
    pub(super) fn prepare_buffer_read(
        &mut self,
        address: u64,
        bytes: u64,
    ) -> Result<Option<DownloadEnd>, VulkanError> {
        #[cfg(feature = "performance-counters")]
        let _trace = nixe_gpu::trace::span("read_buffer_prepare", address, bytes, 0);
        if bytes == 0 {
            return Ok(None);
        }
        self.buffer_cache.publish_completed_reads()?;
        let page = nixe_memory::PAGE_SIZE;
        let begin = address / page * page;
        let end = (address + bytes).div_ceil(page) * page;
        if !self.buffer_cache.download_cpu_read_ranges
            && !self
                .buffer_cache
                .gpu_modified_ranges
                .intersects(begin, end - begin)
        {
            // Physical-page read marks are conservative and survive publication.
            // Clean pages need no temporary interval set, buffer lookup or staging scan.
            // An older reader can still own an ordered snapshot of these bytes.
            return Ok(self.buffer_cache.pending_read(address, bytes));
        }
        self.submit_buffer_reads(&RangeSet::from_range(address, bytes))?;
        Ok(self.buffer_cache.pending_read(address, bytes))
    }

    fn submit_buffer_reads(&mut self, ranges: &RangeSet) -> Result<(), VulkanError> {
        // Only a newly observed CPU read asks to include learned ranges.
        let mut learned;
        let ranges = if std::mem::take(&mut self.buffer_cache.download_cpu_read_ranges) {
            learned = ranges.clone();
            for (at, size) in self.buffer_cache.cpu_read_ranges.iter() {
                learned.extend(
                    self.buffer_cache
                        .gpu_modified_ranges
                        .intersections(at, size),
                );
            }
            &learned
        } else {
            ranges
        };
        if let Some(download) = self.record_buffer_downloads(ranges.iter(), DownloadKind::Read)? {
            self.buffer_cache.enqueue_read_download(download);
            // Submit before handing the wait to the CPU; an idle owner must not
            // strand a reader whose producer needs no further guest commands.
            self.flush()?;
        }
        Ok(())
    }

    pub(super) fn read_buffer_spans(&mut self, ranges: RangeSet) -> Result<(), VulkanError> {
        self.buffer_cache.publish_completed_reads()?;
        // All ranges observe this owner boundary. Record them before any
        // producer can advance, then publish the required prefix once.
        self.submit_buffer_reads(&ranges)?;
        let end = ranges
            .iter()
            .filter_map(|(address, bytes)| self.buffer_cache.pending_read(address, bytes))
            .max_by_key(|end| end.serial);
        if let Some(end) = end {
            self.complete_buffer_read(end)?;
        }
        Ok(())
    }
    /// A dependent owner read consumes the same ordered snapshots as CPU reads.
    /// Copies already pending for the current version cover the read; only gaps
    /// need another reservation/submission. A later producer removes its bytes
    /// from that coverage and therefore still requires a new, ordered copy.
    #[cfg_attr(feature = "performance-counters", track_caller)]
    pub(crate) fn read_buffer_memory(
        &mut self,
        address: u64,
        bytes: u64,
    ) -> Result<(), VulkanError> {
        #[cfg(feature = "performance-counters")]
        let _trace = nixe_gpu::trace::span("read_buffer_sync", address, bytes, 0);
        if let Some(end) = self.prepare_buffer_read(address, bytes)? {
            self.complete_buffer_read(end)?;
        }
        Ok(())
    }
    fn complete_buffer_read(&mut self, end: DownloadEnd) -> Result<(), VulkanError> {
        self.runtime
            .device
            .wait_for_work(end.work(), std::time::Duration::MAX)?;
        self.buffer_cache.publish_reads_through(end)
    }
    pub(super) fn finish_buffer_reads(
        &mut self,
        address: u64,
        bytes: u64,
    ) -> Result<(), VulkanError> {
        if let Some(end) = self.buffer_cache.pending_read(address, bytes) {
            self.complete_buffer_read(end)?;
        }
        Ok(())
    }
    pub(crate) fn commit_buffer_downloads(&mut self) -> Result<(), VulkanError> {
        let ranges = std::mem::take(&mut self.buffer_cache.pending_download_ranges);
        let download = self.record_buffer_downloads(ranges.iter(), DownloadKind::Fence)?;
        self.buffer_cache.buffer_downloads.push_back(download);
        Ok(())
    }
}
