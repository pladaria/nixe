//! BufferCache owns asynchronous DAddr downloads and their publication masks.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/buffer_cache/buffer_cache.h#L586-L716
use super::*;
use crate::transfer::host_read_barrier;

pub(super) struct Download {
    ticket: crate::ReadbackTicket,
    publications: Vec<Publication>,
}
struct Publication {
    address: u64,
    bytes: u64,
    inputs: Vec<(u64, nixe_memory::AllocationRange)>,
    offset: usize,
}
// Keep logical offsets when the device interval contains unmapped pages.
// DeviceMemory::WriteBlockUnsafe skips those pages rather than compacting data.
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
    pub(super) fn image_copy(
        ticket: crate::ReadbackTicket,
        address: u64,
        bytes: u64,
        inputs: Vec<(u64, nixe_memory::AllocationRange)>,
    ) -> Self {
        Self {
            ticket,
            publications: vec![Publication {
                address,
                bytes,
                inputs,
                offset: 0,
            }],
        }
    }
    pub(super) fn work(&self) -> WorkTicket {
        self.ticket.work()
    }
}
impl BufferCache {
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
        self.gpu_modified_ranges.insert(address, bytes);
        self.uncommitted_gpu_modified_ranges.insert(address, bytes);
        Ok(())
    }
    pub(super) fn clear_download(&mut self, address: u64, bytes: u64) {
        self.async_downloads.delete_all(address, bytes);
        self.uncommitted_gpu_modified_ranges
            .subtract(address, bytes);
        for ranges in &mut self.committed_gpu_modified_ranges {
            ranges.subtract(address, bytes);
        }
    }
    pub(super) fn accumulate_flushes(&mut self) {
        if !self.uncommitted_gpu_modified_ranges.is_empty() {
            self.committed_gpu_modified_ranges
                .push(std::mem::take(&mut self.uncommitted_gpu_modified_ranges));
        }
    }
    pub(crate) fn visibility_work(&self) -> Option<WorkTicket> {
        self.buffer_downloads
            .back()
            .and_then(Option::as_ref)
            .map(Download::work)
    }
}
impl Residency<'_> {
    /// Safe owner reads follow Eden DownloadBufferMemory: clear GPU dirtiness,
    /// cancel older asynchronous publications, copy exact modified intervals,
    /// finish native work, and consume the completed staging directly.
    /// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/buffer_cache/buffer_cache.h#L1878-L1953
    pub(crate) fn download_buffer_memory(
        &mut self,
        address: u64,
        bytes: u64,
    ) -> Result<(), VulkanError> {
        let mut at = address;
        let end = address + bytes;
        while at < end {
            let Some(slot) = self.buffer_cache.buffer_index.at(at) else {
                at = ((at / 65536 + 1) * 65536).min(end);
                continue;
            };
            let entry = &self.buffer_cache.buffers[slot];
            let base = entry.device_address.expect("DAddr buffer index entry");
            let stop = (base + entry.native.size()).min(end);
            let id = slot;
            let mut copies = Vec::new();
            let mut inputs = Vec::new();
            let mut total = 0;
            for (start, size) in self
                .buffer_cache
                .memory_tracker
                .downloads(at, stop - at, true)
            {
                for (start, size) in self
                    .buffer_cache
                    .gpu_modified_ranges
                    .intersections(start, size)
                {
                    let mapping = self.runtime.device_memory.read_mapping(start, size)?;
                    inputs.push(download_inputs(&mapping)?);
                    copies.push(
                        vk::BufferCopy::default()
                            .src_offset(start - base)
                            .dst_offset(total)
                            .size(size),
                    );
                    total += size.next_multiple_of(64);
                }
                self.buffer_cache.clear_download(start, size);
                self.buffer_cache.gpu_modified_ranges.subtract(start, size);
            }
            if total != 0 {
                let (slice, ticket) = self
                    .runtime
                    .device
                    .submission
                    .reserve_readback_aligned(total, 64)?;
                let destination = self.runtime.device.submission.staging_buffer(&slice).handle;
                let mut source = None;
                for copy in &mut copies {
                    source = Some(self.use_buffer(id, copy.src_offset, copy.size)?);
                    copy.dst_offset += slice.offset;
                }
                self.runtime.barriers.emit(&mut self.runtime.device)?;
                unsafe {
                    self.runtime.device.dispatch.cmd_copy_buffer(
                        source.unwrap(),
                        destination,
                        &copies,
                        true,
                    );
                }
                host_read_barrier(&mut self.runtime.device, destination, slice.offset, total);
                self.flush()?;
                self.runtime
                    .device
                    .wait_for_work(ticket.work(), std::time::Duration::MAX)?;
                let staging = ticket.completed()?;
                for (copy, inputs) in copies.iter().zip(inputs) {
                    let source = (copy.dst_offset - slice.offset) as usize;
                    for (offset, input) in inputs {
                        let offset = offset as usize;
                        let mut source =
                            &staging[source + offset..source + offset + input.bytes() as usize];
                        self.runtime.memory.write_resources(
                            std::slice::from_ref(&input),
                            |writer| {
                                for extent in input.physical_extents() {
                                    let size = extent.bytes as usize;
                                    writer.write(extent, &source[..size])?;
                                    source = &source[size..];
                                }
                                Ok(())
                            },
                        )?;
                        self.buffer_cache.canonical_readback_bytes += input.bytes();
                    }
                    self.buffer_cache.native_readback_bytes += copy.size;
                }
            }
            at = stop;
        }
        Ok(())
    }
    pub(crate) fn commit_buffer_downloads(&mut self) -> Result<(), VulkanError> {
        self.buffer_cache.accumulate_flushes();
        let mut pending = std::mem::take(&mut self.buffer_cache.committed_gpu_modified_ranges);
        for index in 0..pending.len() {
            let (earlier, later) = pending.split_at_mut(index + 1);
            for ranges in later {
                for (address, bytes) in ranges.iter() {
                    earlier[index].subtract(address, bytes);
                }
            }
        }
        let mut publications = Vec::new();
        let mut groups: Vec<(BufferId, Vec<vk::BufferCopy>)> = Vec::new();
        let mut total = 0u64;
        for ranges in &pending {
            for (address, bytes) in ranges.iter() {
                let mut at = address;
                let end = address + bytes;
                while at < end {
                    let Some(slot) = self.buffer_cache.buffer_index.at(at) else {
                        at = ((at / 65536 + 1) * 65536).min(end);
                        continue;
                    };
                    let entry = &self.buffer_cache.buffers[slot];
                    let base = entry.device_address.expect("DAddr buffer index entry");
                    let stop = (base + entry.native.size()).min(end);
                    let id = slot;
                    let downloads =
                        self.buffer_cache
                            .memory_tracker
                            .downloads(at, stop - at, false);
                    for (start, size) in downloads {
                        for (start, size) in self
                            .buffer_cache
                            .gpu_modified_ranges
                            .intersections(start, size)
                        {
                            let offset = start - base;
                            let footprint = self.runtime.device_memory.read_mapping(start, size)?;
                            let inputs = download_inputs(&footprint)?;
                            let target = total;
                            total = crate::queue::align_up(
                                total.checked_add(size).ok_or(VulkanError::InvalidResource(
                                    "fence download size overflow",
                                ))?,
                                64,
                            )
                            .ok_or(VulkanError::InvalidResource(
                                "fence download alignment overflow",
                            ))?;
                            let copy = vk::BufferCopy::default()
                                .src_offset(offset)
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
                        }
                    }
                    at = stop;
                }
            }
        }
        if publications.is_empty() {
            self.buffer_cache.buffer_downloads.push_back(None);
            pending.clear();
            self.buffer_cache.committed_gpu_modified_ranges = pending;
            return Ok(());
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
        for publication in &publications {
            self.buffer_cache
                .async_downloads
                .add(publication.address, publication.bytes);
        }
        self.buffer_cache.buffer_downloads.push_back(Some(Download {
            ticket,
            publications,
        }));
        pending.clear();
        self.buffer_cache.committed_gpu_modified_ranges = pending;
        Ok(())
    }
}

impl BufferCache {
    pub(crate) fn pop_async_flush(&mut self) -> Result<(), VulkanError> {
        if let Some(Some(download)) = self.buffer_downloads.pop_front() {
            self.publish_download(download)?;
        }
        Ok(())
    }
    pub(super) fn publish_download(&mut self, download: Download) -> Result<(), VulkanError> {
        let staging = download.ticket.completed()?;
        for publication in download.publications {
            self.native_readback_bytes += publication.bytes;
            let offset = publication.offset;
            let bytes = &staging[offset..offset + publication.bytes as usize];
            let memory = self.device_memory.machine().clone();
            let buffers = &mut *self;
            for &(offset, ref input) in &publication.inputs {
                let offset = offset as usize;
                let length = input.bytes() as usize;
                let address = publication.address + offset as u64;
                memory.write_download(input, &bytes[offset..offset + length], || {
                    let selected = buffers
                        .async_downloads
                        .intersections(address, length as u64)
                        .map(|(start, size, _)| {
                            buffers.canonical_readback_bytes += size;
                            (start - address) as usize..(start - address + size) as usize
                        })
                        .collect();
                    Ok(selected)
                })?;
            }
            let cache = &mut *self;
            cache.async_downloads.subtract(
                publication.address,
                publication.bytes,
                |start, size| cache.gpu_modified_ranges.subtract(start, size),
            );
        }
        drop(staging);
        Ok(())
    }
}
