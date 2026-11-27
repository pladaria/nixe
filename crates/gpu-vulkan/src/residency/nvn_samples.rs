//! Ordered CPU publication of Vulkan visibility segments and NVN sample reports.
//! https://github.com/scarletnx/ScarletNX/blob/6930e50/src/nx/gpu/vk/release.c
use super::*;
use crate::queue::NvnQueryCopy;
use nixe_memory::{AllocationRange, PAGE_SIZE};
use std::cell::RefCell;
use std::collections::{BTreeMap, VecDeque, btree_map::Entry};

const REPORT_BYTES: u64 = 16;

struct Report {
    address: u64,
    timestamp: u64,
    target: Vec<AllocationRange>,
}

enum Event {
    Reset,
    Report(Report),
}

struct Batch {
    work: WorkTicket,
    copies: Vec<NvnQueryCopy>,
    events: Vec<(u64, Event)>,
}

#[derive(Default)]
struct DeviceIndex {
    generation: u64,
    intervals: BTreeMap<u64, (u64, WorkTicket)>,
}

impl DeviceIndex {
    fn insert(
        &mut self,
        memory: &nixe_memory::device_memory::DeviceMemory,
        extent: nixe_memory::Extent,
        work: WorkTicket,
    ) {
        for (address, bytes) in memory.device_ranges(extent) {
            self.intervals.insert(address, (bytes, work));
        }
    }
}

#[derive(Default)]
pub(super) struct SampleReports {
    events: Vec<(u64, Event)>,
    submitted: VecDeque<Batch>,
    position: u64,
    total: u64,
    // Physical intervals remain valid across DAddr aliases and remapping.
    latest: BTreeMap<u64, (u64, WorkTicket)>,
    trapped: BTreeMap<u64, usize>,
    device_index: RefCell<DeviceIndex>,
}

impl SampleReports {
    pub(super) fn has_events(&self) -> bool {
        !self.events.is_empty()
    }

    pub(super) fn producer(
        &self,
        memory: &nixe_memory::device_memory::DeviceMemory,
        address: u64,
        bytes: u64,
    ) -> Option<WorkTicket> {
        if self.latest.is_empty() || bytes == 0 {
            return None;
        }
        // Rebuild from pending reports only when a device mapping changes.
        // Lookup is independent of the number of pages in the input buffer.
        let mut index = self.device_index.borrow_mut();
        loop {
            let generation = memory.generation();
            if index.generation != generation {
                index.intervals.clear();
                for (&start, &(bytes, work)) in &self.latest {
                    index.insert(
                        memory,
                        nixe_memory::Extent {
                            start: nixe_memory::ArenaOffset::new(start),
                            bytes,
                        },
                        work,
                    );
                }
                index.generation = generation;
            }
            let work = index
                .intervals
                .range(address.saturating_sub(REPORT_BYTES - 1)..address + bytes)
                .filter_map(|(&at, &(size, work))| (at + size > address).then_some(work))
                .max_by_key(|work| work.tick());
            if memory.generation() == generation {
                return work;
            }
        }
    }

    fn register(
        &mut self,
        memory: &nixe_memory::device_memory::DeviceMemory,
        position: u64,
        work: WorkTicket,
        report: Report,
    ) -> Result<(), VulkanError> {
        if self.latest.is_empty() {
            let index = self.device_index.get_mut();
            index.generation = memory.generation();
            index.intervals.clear();
        }
        let address = report.address;
        for page in report_pages(address) {
            match self.trapped.entry(page) {
                Entry::Occupied(mut count) => *count.get_mut() += 1,
                Entry::Vacant(slot) => {
                    memory.update_pages_cached_count(page, PAGE_SIZE, 1)?;
                    slot.insert(1);
                }
            }
        }
        for extent in report
            .target
            .iter()
            .flat_map(AllocationRange::physical_extents)
        {
            self.latest.insert(extent.start.get(), (extent.bytes, work));
            let index = self.device_index.get_mut();
            if index.generation == memory.generation() {
                index.insert(memory, extent, work);
            }
        }
        memory.invalidate_read_areas();
        self.events.push((position, Event::Report(report)));
        Ok(())
    }
}

fn report_pages(address: u64) -> impl Iterator<Item = u64> {
    (address / PAGE_SIZE..(address + REPORT_BYTES).div_ceil(PAGE_SIZE)).map(|page| page * PAGE_SIZE)
}

impl BufferCache {
    pub(super) fn has_pending_nvn_samples(&self) -> bool {
        !self.nvn_samples.latest.is_empty()
    }

    pub(super) fn nvn_sample_producer(&self, address: u64, bytes: u64) -> Option<WorkTicket> {
        self.nvn_samples
            .producer(&self.device_memory, address, bytes)
    }

    pub(crate) fn publish_nvn_samples(&mut self, completed: u64) -> Result<(), VulkanError> {
        while self
            .nvn_samples
            .submitted
            .front()
            .is_some_and(|batch| batch.work.tick() <= completed)
        {
            let batch = self.nvn_samples.submitted.pop_front().unwrap();
            let mut events = batch.events.into_iter().peekable();
            for copy in batch.copies {
                let data = copy.ticket.completed()?;
                self.native_readback_bytes += data.len() as u64;
                for &word in data.as_chunks::<8>().0 {
                    self.apply_nvn_sample_events(&mut events, batch.work)?;
                    self.nvn_samples.total = self
                        .nvn_samples
                        .total
                        .wrapping_add(u64::from_le_bytes(word));
                    self.nvn_samples.position += 1;
                }
            }
            self.apply_nvn_sample_events(&mut events, batch.work)?;
            debug_assert!(events.next().is_none());
        }
        Ok(())
    }

    fn apply_nvn_sample_events(
        &mut self,
        events: &mut std::iter::Peekable<impl Iterator<Item = (u64, Event)>>,
        work: WorkTicket,
    ) -> Result<(), VulkanError> {
        while events
            .peek()
            .is_some_and(|(position, _)| *position == self.nvn_samples.position)
        {
            match events.next().unwrap().1 {
                Event::Reset => self.nvn_samples.total = 0,
                Event::Report(report) => self.publish_nvn_sample(report, work)?,
            }
        }
        Ok(())
    }

    fn publish_nvn_sample(&mut self, report: Report, work: WorkTicket) -> Result<(), VulkanError> {
        let mut value = [0; REPORT_BYTES as usize];
        value[..8].copy_from_slice(&self.nvn_samples.total.to_le_bytes());
        value[8..].copy_from_slice(&report.timestamp.to_le_bytes());
        let memory = self.device_memory.clone();
        query::write_report(memory.machine(), &report.target, &value)?;
        self.canonical_readback_bytes += REPORT_BYTES;
        for extent in report
            .target
            .iter()
            .flat_map(AllocationRange::physical_extents)
        {
            let finished = self
                .nvn_samples
                .latest
                .get(&extent.start.get())
                .is_some_and(|(_, newer)| *newer == work);
            for (address, bytes) in memory.device_ranges(extent) {
                self.write_memory(address, bytes);
                let index = self.nvn_samples.device_index.get_mut();
                if finished && index.generation == memory.generation() {
                    index.intervals.remove(&address);
                }
            }
            if finished {
                self.nvn_samples.latest.remove(&extent.start.get());
            }
        }
        for page in report_pages(report.address) {
            if let Entry::Occupied(mut entry) = self.nvn_samples.trapped.entry(page) {
                *entry.get_mut() -= 1;
                if *entry.get() == 0 {
                    entry.remove();
                    memory.update_pages_cached_count(page, PAGE_SIZE, -1)?;
                }
            }
        }
        Ok(())
    }
}

impl Residency<'_> {
    pub(crate) fn reset_nvn_samples(&mut self) {
        let position = self.runtime.device.nvn_query_boundary();
        self.buffer_cache
            .nvn_samples
            .events
            .push((position, Event::Reset));
    }

    pub(crate) fn write_nvn_sample_report(
        &mut self,
        address: u64,
        timestamp: u64,
    ) -> Result<(), VulkanError> {
        let memory = self.runtime.device_memory.clone();
        let target = memory
            .read_mapping(address, REPORT_BYTES)?
            .pin(0, REPORT_BYTES)?;
        let mut aliases = RangeSet::from_range(address, REPORT_BYTES);
        memory.apply_device_aliases(address, REPORT_BYTES, |alias, bytes| {
            aliases.insert(alias, bytes)
        });
        for (alias, bytes) in aliases.iter() {
            self.finish_buffer_reads(alias, bytes)?;
        }
        self.preserve_texture_writes(&aliases, || {
            Ok(target
                .iter()
                .flat_map(AllocationRange::physical_extents)
                .map(|extent| (extent.start.get(), extent.bytes))
                .collect())
        })?;
        for (alias, bytes) in aliases.iter() {
            self.buffer_cache.claim_cpu_write(alias, bytes);
            self.texture_cache.write_memory(alias, bytes);
            self.shader_cache.invalidate_region(alias, bytes);
            if self.caches.query_pages.any(alias, bytes) {
                self.caches
                    .query_cache
                    .lock()
                    .unwrap()
                    .invalidate_region(alias, bytes);
            }
        }
        for extent in target.iter().flat_map(AllocationRange::physical_extents) {
            self.caches
                .ordered_store_pages
                .mark(extent.start.get(), extent.bytes);
        }
        let position = self.runtime.device.nvn_query_boundary();
        let work = self.runtime.device.current_work();
        memory.mark_gpu_written(address, REPORT_BYTES);
        self.buffer_cache.nvn_samples.register(
            &memory,
            position,
            work,
            Report {
                address,
                timestamp,
                target,
            },
        )
    }

    pub(super) fn stage_nvn_samples(&mut self) -> Result<(), VulkanError> {
        let copies = self.runtime.device.take_nvn_query_copies()?;
        let events = std::mem::take(&mut self.buffer_cache.nvn_samples.events);
        if !events.is_empty() || !copies.is_empty() {
            let work = self.runtime.device.current_work();
            let count = events
                .iter()
                .filter(|(_, event)| matches!(event, Event::Report(_)))
                .count();
            nixe_gpu::trace::event(
                "nvn_sample_readback_stage",
                work.tick(),
                count as u64 * REPORT_BYTES,
                count as u64,
            );
            self.buffer_cache.nvn_samples.submitted.push_back(Batch {
                work,
                copies,
                events,
            });
            self.caches
                .nvn_sample_batches
                .store(true, std::sync::atomic::Ordering::Release);
        }
        Ok(())
    }

    pub(super) fn resolve_nvn_sample_access(
        &mut self,
        address: u64,
        bytes: u64,
    ) -> Result<bool, VulkanError> {
        let Some(work) = self.buffer_cache.nvn_sample_producer(address, bytes) else {
            return Ok(false);
        };
        nixe_gpu::trace::event("nvn_sample_readback_wait", work.tick(), address, bytes);
        if work.tick() > self.runtime.device.scheduled_tick() {
            self.flush()?;
        }
        self.runtime
            .device
            .wait_for_work(work, std::time::Duration::MAX)?;
        self.buffer_cache
            .publish_nvn_samples(self.caches.progress.completed())?;
        Ok(true)
    }
}
