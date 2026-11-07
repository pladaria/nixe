//! BufferCache owns asynchronous DAddr downloads and their publication masks.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/buffer_cache/buffer_cache.h#L586-L716
use super::*;
use crate::transfer::host_read_barrier;

pub(super) struct Download {
    work: Option<WorkTicket>,
    ticket: Option<crate::ReadbackTicket>,
    publications: Vec<Publication>,
}
struct Publication {
    address: u64,
    bytes: u64,
    inputs: Vec<(u64, nixe_memory::AllocationRange)>,
    source: Source,
}
enum Source {
    Native { offset: usize },
    Literal { bytes: Arc<[u8]>, offset: usize },
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
            work: Some(ticket.work()),
            ticket: Some(ticket),
            publications: vec![Publication {
                address,
                bytes,
                inputs,
                source: Source::Native { offset: 0 },
            }],
        }
    }
    pub(super) fn work(&self) -> Option<WorkTicket> {
        self.work
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
            .and_then(Download::work)
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
                            let literal = self.buffer_cache.get(id).known.iter().find(|known| {
                                known.offset <= offset
                                    && offset + size <= known.offset + known.bytes.len() as u64
                            });
                            let source = if let Some(literal) = literal {
                                Source::Literal {
                                    bytes: Arc::clone(&literal.bytes),
                                    offset: (offset - literal.offset) as usize,
                                }
                            } else {
                                let target = total;
                                total = crate::queue::align_up(
                                    total.checked_add(size).ok_or(VulkanError::InvalidResource(
                                        "fence download size overflow",
                                    ))?,
                                    64,
                                )
                                .ok_or(
                                    VulkanError::InvalidResource(
                                        "fence download alignment overflow",
                                    ),
                                )?;
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
                                Source::Native {
                                    offset: target as usize,
                                }
                            };
                            publications.push(Publication {
                                address: start,
                                bytes: size,
                                inputs,
                                source,
                            });
                        }
                    }
                    at = stop;
                }
            }
        }
        let mut work = None;
        let ticket = if total != 0 {
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
            work = Some(ticket.work());
            Some(ticket)
        } else {
            None
        };
        for publication in &publications {
            self.buffer_cache
                .async_downloads
                .add(publication.address, publication.bytes);
        }
        self.buffer_cache
            .buffer_downloads
            .push_back((!publications.is_empty()).then_some(Download {
                work,
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
        let staging = download
            .ticket
            .map(crate::ReadbackTicket::completed)
            .transpose()?;
        for publication in download.publications {
            let bytes = match &publication.source {
                Source::Native { offset } => {
                    self.native_readback_bytes += publication.bytes;
                    &staging.as_ref().expect("native download reservation")
                        [*offset..*offset + publication.bytes as usize]
                }
                Source::Literal { bytes, offset } => {
                    &bytes[*offset..*offset + publication.bytes as usize]
                }
            };
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::residency::tests::{backing_allocation, buffer, cache, cpu_read};

    #[test]
    fn safe_owner_read_downloads_current_buffer_and_cancels_older_publications() {
        use nixe_memory::{AccessOperation, AccessResult};
        let Some(mut owner) = cache(crate::ApiPreference::Auto, 16 * 1024 * 1024) else {
            return;
        };
        let (_allocation, mapping) = backing_allocation(owner.device_memory(), 4096, 0x19);
        let mut cache = owner.access();
        let validation = cache.device().validation_monitor();
        let id = buffer(&mut cache, Arc::clone(&mapping));
        cache.fill_buffer(id, 4, 8, 0x37373737).unwrap();
        cache.commit_buffer_downloads().unwrap();
        cache.flush().unwrap();
        cache.fill_buffer(id, 4, 4, 0x42424242).unwrap();
        let inputs = mapping.pin(5, 1).unwrap();
        let AccessResult::Read(bytes) = cache
            .access_memory(&inputs, &mut AccessOperation::Read(Vec::new()))
            .unwrap()
        else {
            panic!("expected safe read");
        };
        assert_eq!(bytes, [0x42]);
        assert!(cache.buffer_cache.gpu_modified_ranges.is_empty());
        assert!(
            cache
                .buffer_cache
                .uncommitted_gpu_modified_ranges
                .is_empty()
        );
        crate::residency::tests::wait_buffer_downloads(&mut cache).unwrap();
        let bytes = cpu_read(&mut cache, &mapping);
        assert_eq!(&bytes[4..8], &[0x42; 4]);
        assert_eq!(&bytes[8..12], &[0x37; 4]);
        assert_eq!(&bytes[12..], &[0x19; 4084]);
        // Subsequent native writes still become dirty and publish at their fence.
        cache.fill_buffer(id, 8, 4, 0x71717171).unwrap();
        cache.commit_buffer_downloads().unwrap();
        cache.flush().unwrap();
        crate::residency::tests::wait_buffer_downloads(&mut cache).unwrap();
        assert_eq!(
            &cpu_read(&mut cache, &mapping)[4..12],
            &[0x42, 0x42, 0x42, 0x42, 0x71, 0x71, 0x71, 0x71]
        );
        drop(cache);
        drop(owner);
        validation.check().unwrap();
    }

    #[test]
    fn fence_batch_groups_regions_and_retains_later_overlapping_producers() {
        let Some(mut cache_owner) = cache(crate::ApiPreference::Auto, 16 * 1024 * 1024) else {
            return;
        };
        let mut cache = cache_owner.access();
        let validation = cache.runtime.device.validation_monitor();
        let first = backing_allocation(cache.device_memory(), 4096, 0x19);
        let second = backing_allocation(cache.device_memory(), 4096, 0x19);
        let ids = [
            buffer(&mut cache, Arc::clone(&first.1)),
            buffer(&mut cache, Arc::clone(&second.1)),
        ];
        for (index, &id) in ids.iter().enumerate() {
            cache
                .fill_buffer(id, 4, 8, 0x37373737 + index as u32)
                .unwrap();
            cache
                .fill_buffer(id, 80, 12, 0x42424242 + index as u32)
                .unwrap();
        }
        cache.commit_buffer_downloads().unwrap();
        assert_eq!(cache.buffer_cache.buffer_downloads.len(), 1);
        let batch = cache.buffer_cache.buffer_downloads[0].as_ref().unwrap();
        // Eden ForEachDownloadRange reports whole dirty pages for each
        // committed interval. Both small regions on a page are therefore
        // included by each query, with independent async overlap references.
        assert_eq!(batch.ticket.as_ref().unwrap().slice.size, 512);
        assert_eq!(batch.publications.len(), 8);
        assert!(
            batch
                .publications
                .iter()
                .all(|p| matches!(p.source, Source::Native { offset } if offset % 64 == 0))
        );
        assert_eq!(cache.runtime.device.queue_usage().retained_readbacks, 1);
        let prefix = batch.work.unwrap();
        cache.flush().unwrap();
        cache.fill_buffer(ids[0], 4, 4, 0x71717171).unwrap();
        cache.commit_buffer_downloads().unwrap();
        cache.flush().unwrap();
        cache
            .runtime
            .device
            .wait_for_work(prefix, std::time::Duration::MAX)
            .unwrap();
        cache.buffer_cache.pop_async_flush().unwrap();
        assert_eq!(cache.buffer_cache.buffer_downloads.len(), 1);
        assert_eq!(cache.runtime.device.queue_usage().retained_readbacks, 1);
        crate::residency::tests::wait_buffer_downloads(&mut cache).unwrap();
        assert_eq!(cache.runtime.device.queue_usage().retained_readbacks, 0);
        let a = cpu_read(&mut cache, &first.1);
        let b = cpu_read(&mut cache, &second.1);
        assert_eq!(&a[4..8], &[0x71; 4]);
        assert_eq!(&a[8..12], &[0x37; 4]);
        assert_eq!(&a[80..92], &[0x42; 12]);
        assert_eq!(&b[4..12], &0x37373738u32.to_le_bytes().repeat(2));
        assert_eq!(&b[80..92], &0x42424243u32.to_le_bytes().repeat(3));
        assert_eq!(&a[12..80], &[0x19; 68]);
        drop(cache);
        drop(cache_owner);
        validation.check().unwrap();
    }
}

#[cfg(test)]
mod cpu_store_tests {
    use super::*;
    use nixe_memory::{
        AccessOperation, AddressSpace, AddressWidth, GuestVirtualAddress, MemoryPermissions,
    };

    #[test]
    fn explicit_cpu_invalidation_cancels_its_cache_line_before_fence_publication() {
        let Some(mut owner) =
            super::super::tests::cache(crate::ApiPreference::Auto, 16 * 1024 * 1024)
        else {
            return;
        };
        let memory = owner.device_memory().clone();
        let machine = memory.machine().clone();
        let allocation = machine.account(4096).unwrap().allocate(4096).unwrap();
        let space = AddressSpace::new(machine.clone(), AddressWidth::Bits32).unwrap();
        let base = GuestVirtualAddress::new(0x10000);
        space
            .map(base, allocation, 0, 4096, MemoryPermissions::READ_WRITE)
            .unwrap();
        let inputs = space
            .capture(base, 4096, MemoryPermissions::READ_WRITE)
            .unwrap();
        machine.write_resolved(&inputs, &[0x19; 4096]).unwrap();
        let reservation = memory.reserve(65536, 65536).unwrap();
        let pin = memory
            .map(reservation.address(), inputs.clone().into(), true)
            .unwrap();
        let mut cache = owner.access();
        let validation = cache.device().validation_monitor();
        let id = super::super::tests::buffer(&mut cache, Arc::clone(pin.physical()));
        cache.fill_buffer(id, 4, 8, 0x37373737).unwrap();
        cache.fill_buffer(id, 80, 8, 0x44444444).unwrap();
        cache.commit_buffer_downloads().unwrap();
        cache.flush().unwrap();
        drop(cache);

        machine.enable_cpu_write_collection().unwrap();
        let core = space.register_core(Arc::new(|| {}));
        let native = core.enter_native().unwrap();
        let target = GuestVirtualAddress::new(base.get() + 6);
        assert!(!space.write_native(&native, target, &[0x55, 0x66]).unwrap());
        let operation = space
            .capture_native(&native, target, 2, MemoryPermissions::WRITE)
            .unwrap();
        let cached = space.has_cached_pages(&native, target, 2).unwrap();
        native.suspend();
        assert!(matches!(
            native
                .access_captured(&operation, cached, AccessOperation::Write(vec![0x55, 0x66]))
                .unwrap()
                .into_ready()
                .unwrap(),
            nixe_memory::AccessResult::Written(_)
        ));
        native.resume().unwrap();
        drop(native);

        machine.gather_gpu_dirty_memory(|address, bytes| {
            nixe_memory::device_memory::DeviceMemoryInterface::invalidate_region(
                &*owner.caches,
                address,
                bytes,
            );
        });
        let mut cache = owner.access();
        crate::residency::tests::wait_buffer_downloads(&mut cache).unwrap();
        let mut bytes = [0; 128];
        pin.physical().capture_stream(0, &mut bytes).unwrap();
        assert_eq!(
            &bytes[4..12],
            &[0x19, 0x19, 0x55, 0x66, 0x19, 0x19, 0x19, 0x19]
        );
        assert_eq!(&bytes[..4], &[0x19; 4]);
        assert_eq!(&bytes[12..64], &[0x19; 52]);
        assert_eq!(&bytes[80..88], &[0x44; 8]);
        machine.gather_gpu_dirty_memory(|_, _| panic!("unexpected pending CPU invalidation"));
        machine.disable_cpu_write_collection();
        drop(cache);
        drop(owner);
        validation.check().unwrap();
    }
}
