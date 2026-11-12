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
        self.memory_tracker.gpu_write(address, bytes);
        self.note_command_write(id, address, bytes);
        self.outdate_downloads(address, bytes);
        self.gpu_modified_ranges.insert(address, bytes);
        self.pending_download_ranges.insert(address, bytes);
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
        self.read_download_ranges
            .extend(download.write_ranges.iter());
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
            let (_, download) = self.read_downloads.pop_front().unwrap();
            // Retire the reader even if publication fails during cancellation.
            let result = self.publish_read_download(download);
            self.pending_read_count
                .fetch_sub(1, std::sync::atomic::Ordering::Release);
            result?;
        }
        Ok(())
    }
    fn publish_completed_reads(&mut self) -> Result<(), VulkanError> {
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
        let _landing = memory.begin_gpu_download(begin, end);
        for (address, bytes) in memory.pending_gpu_dirty_memory().iter() {
            download.write_ranges.subtract(address, bytes);
        }
        for (address, bytes) in self.write_download(download)?.iter() {
            self.gpu_modified_ranges.subtract(address, bytes);
        }
        Ok(())
    }
    fn publish_read_download(&mut self, mut read: ReadDownload) -> Result<(), VulkanError> {
        let memory = self.device_memory.machine().clone();
        let (begin, end) = read.download.bounds();
        let _landing = memory.begin_gpu_download(begin, end);
        for (address, bytes) in memory.pending_gpu_dirty_memory().iter() {
            read.download.write_ranges.subtract(address, bytes);
            read.clean_ranges.subtract(address, bytes);
            self.read_download_ranges.subtract(address, bytes);
        }
        self.write_download(read.download)?;
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
        }
        for (address, bytes) in read.clean_ranges.iter() {
            let page = nixe_memory::PAGE_SIZE;
            for at in (address / page..(address + bytes).div_ceil(page)).map(|p| p * page) {
                if !self.gpu_modified_ranges.intersects(at, page) {
                    self.memory_tracker.clear_gpu(at, page);
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum DownloadKind {
    Fence,
    Read,
    Synchronous,
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
                for (start, size) in self.buffer_cache.memory_tracker.downloads(at, stop - at) {
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
                        DownloadKind::Synchronous => {
                            self.buffer_cache.clear_download(start, size);
                            self.buffer_cache.memory_tracker.clear_gpu(start, size);
                        }
                    }
                    let mut add_copy = |start: u64, size: u64| -> Result<(), VulkanError> {
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
        self.buffer_cache.publish_completed_reads()?;
        if let Some(download) =
            self.record_buffer_downloads([(address, bytes)], DownloadKind::Read)?
        {
            self.buffer_cache.enqueue_read_download(download);
            // Submit before handing the wait to the CPU; an idle owner must not
            // strand a reader whose producer needs no further guest commands.
            self.flush()?;
        }
        Ok(self.buffer_cache.pending_read(address, bytes))
    }
    /// Owner-dependent reads synchronize all modified bytes of the affected
    /// tracker pages and replace older publications before preparation resumes.
    pub(crate) fn download_buffer_memory(
        &mut self,
        address: u64,
        bytes: u64,
    ) -> Result<(), VulkanError> {
        if let Some(download) =
            self.record_buffer_downloads([(address, bytes)], DownloadKind::Synchronous)?
        {
            let work = download.ticket.work();
            self.flush()?;
            self.runtime
                .device
                .wait_for_work(work, std::time::Duration::MAX)?;
            self.buffer_cache.publish_fence_download(download)?;
        }
        Ok(())
    }
    pub(super) fn finish_buffer_reads(
        &mut self,
        address: u64,
        bytes: u64,
    ) -> Result<(), VulkanError> {
        if let Some(end) = self.buffer_cache.pending_read(address, bytes) {
            self.runtime
                .device
                .wait_for_work(end.work(), std::time::Duration::MAX)?;
            self.buffer_cache.publish_reads_through(end)?;
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
