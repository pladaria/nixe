//! Safe owner reads flush the actual buffer and texture caches before RAM access.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/memory_manager.cpp#L350-L410
use super::*;
use nixe_memory::{AccessOperation, AccessResult, AllocationRange, DeviceAccessRequest};

#[cfg(test)]
mod tests;

// Direct mapping bounds owner storage; collisions only replace evidence.
pub(super) const CLEAN_READ_SLOTS: usize = 1024;
#[derive(Clone, Copy, Default)]
pub(super) struct CleanRead {
    start: u64,
    end: u64,
    generation: u64,
}

impl Residency<'_> {
    pub(crate) fn write_constant_span(
        &mut self,
        address: u64,
        bytes: &[u8],
    ) -> Result<(), VulkanError> {
        let memory = self.runtime.device_memory.clone();
        if memory.may_be_gpu_written(address, bytes.len() as u64) {
            let mut ranges = RangeSet::from_range(address, bytes.len() as u64);
            memory.apply_device_aliases(address, bytes.len() as u64, |a, b| ranges.insert(a, b));
            for (a, b) in ranges.iter() {
                self.finish_buffer_reads(a, b)?;
            }
            self.preserve_texture_writes(&ranges, || {
                Ok(memory
                    .read_mapping(address, bytes.len() as u64)?
                    .spans()
                    .iter()
                    .map(|span| (span.start.get(), span.bytes))
                    .collect())
            })?;
            {
                // Fence callbacks publish query values independently of the owner.
                // Cancel older reports before exposing the replacement RAM bytes.
                let mut queries = self.caches.query_cache.lock().unwrap();
                for (a, b) in ranges.iter() {
                    queries.invalidate_region(a, b);
                }
            }
            // Retire old pending publications before the RAM store can be seen.
            // Invalidating the native bindings remains deferred to consumption.
            for (a, b) in ranges.iter() {
                self.buffer_cache.claim_cpu_write(a, b);
            }
        }
        memory.write_contiguous(address, nixe_memory::WriteSource::Bytes(bytes))?;
        Ok(())
    }
    pub(crate) fn write_constant_regions(
        &mut self,
        writes: &[(u64, u64)],
    ) -> Result<(), VulkanError> {
        let mut ranges = smallvec::SmallVec::<[(u64, u64); 16]>::new();
        for &(address, bytes) in writes {
            ranges.push((address, bytes));
            self.runtime
                .device_memory
                .apply_device_aliases(address, bytes, |a, b| {
                    ranges.push((a, b));
                });
        }
        // Claim the complete batch, including physical aliases, before deciding
        // whether a partially overwritten native page still needs an upload.
        self.invalidate_cpu_ranges(ranges.iter().copied(), RamWrite::Constant)
    }

    fn has_read_producer(&self, address: u64, bytes: u64) -> bool {
        // Reactive buffer downloads collect every dirty interval on the input
        // pages. A clean word beside a dirty buffer must still trigger that batch.
        let page = nixe_memory::PAGE_SIZE;
        let start = address & !(page - 1);
        let end = (address + bytes).div_ceil(page) * page;
        self.buffer_cache
            .gpu_modified_ranges
            .intersects(start, end - start)
            || self.buffer_cache.pending_read(address, bytes).is_some()
            || self.texture_cache.region_gpu_modified(address, bytes)
            || (self.caches.query_pages.any(address, bytes)
                && self
                    .caches
                    .query_cache
                    .lock()
                    .unwrap()
                    .must_flush_region(address, bytes))
    }

    pub(super) fn physical_read_is_clean(&self, address: u64, bytes: u64) -> bool {
        if self.has_read_producer(address, bytes) {
            return false;
        }
        let mut clean = true;
        self.runtime
            .device_memory
            .apply_device_aliases(address, bytes, |at, size| {
                clean = clean && !self.has_read_producer(at, size);
            });
        clean
    }

    #[cfg_attr(feature = "performance-counters", track_caller)]
    pub(crate) fn flush_read_span(&mut self, address: u64, bytes: u64) -> Result<(), VulkanError> {
        use nixe_memory::device_memory::OwnerReadSpan;
        let span = self.runtime.device_memory.owner_read_span(address, bytes);
        if matches!(span, OwnerReadSpan::Unwritten) {
            return Ok(());
        }
        #[cfg(feature = "performance-counters")]
        let clean = {
            self.runtime.statistics.owner_historical_reads += 1;
            let clean = self.physical_read_is_clean(address, bytes);
            if clean {
                self.runtime.statistics.owner_clean_reads += 1;
                let page = nixe_memory::PAGE_SIZE;
                if bytes <= page - (address % page)
                    && self.physical_read_is_clean(address & !(page - 1), page)
                {
                    self.runtime.statistics.owner_clean_pages += 1;
                }
            }
            clean
        };
        if let OwnerReadSpan::Page { physical, reusable } = span
            && !self.buffer_cache.download_cpu_read_ranges
            && self.runtime.device.submission.deferred_clear.is_none()
        {
            let page = nixe_memory::PAGE_SIZE;
            let index = (physical / page) as usize % CLEAN_READ_SLOTS;
            let entry = self.runtime.clean_reads[index];
            let generation = self.runtime.device_memory.generation();
            if reusable
                && entry.generation == generation
                && entry.start <= physical
                && physical + bytes <= entry.end
            {
                #[cfg(feature = "performance-counters")]
                {
                    assert!(
                        clean,
                        "owner read hit overlaps a GPU producer at {address:#x} ({bytes} bytes)"
                    );
                    self.runtime.statistics.owner_read_hits += 1;
                }
                return Ok(());
            }
            // Prove wider coverage without downloading any adjacent bytes. If
            // a neighbour is dirty, only the requested interval can be reused.
            let base = address & !(page - 1);
            let range = if self.physical_read_is_clean(base, page) {
                Some((physical & !(page - 1), page))
            } else if (address != base || bytes != page)
                && self.physical_read_is_clean(address, bytes)
            {
                Some((physical, bytes))
            } else {
                None
            };
            if let Some((start, size)) = range {
                // Mapping changes can introduce another device alias. Discard
                // evidence from a walk across two generations.
                if self.runtime.device_memory.generation() == generation
                    && self
                        .runtime
                        .device_memory
                        .physical_address(address)
                        .is_some_and(|p| p.get() == physical)
                {
                    self.runtime.clean_reads[index] = CleanRead {
                        start,
                        end: start + size,
                        generation,
                    };
                    if !reusable {
                        self.runtime.device_memory.remember_owner_read(physical);
                    }
                    #[cfg(feature = "performance-counters")]
                    {
                        self.runtime.statistics.owner_read_fills += 1;
                    }
                    return Ok(());
                }
            }
        }
        #[cfg(feature = "performance-counters")]
        let _trace = nixe_gpu::trace::span("read_coherent", address, bytes, 0);
        self.flush_region(address, bytes)?;
        let memory = self.runtime.device_memory.clone();
        let mut result = Ok(());
        memory.apply_device_aliases(address, bytes, |alias, size| {
            if result.is_ok() {
                result = self.flush_region(alias, size);
            }
        });
        result
    }
    /// Synchronous owner writes borrow the upload and its physical leases.
    pub(crate) fn write_captured_memory(
        &mut self,
        capture: &nixe_memory::CapturedMemory,
        bytes: &[u8],
    ) -> Result<(), VulkanError> {
        if capture.bytes() != bytes.len() as u64 {
            return Err(VulkanError::InvalidResource(
                "owner write size differs from capture",
            ));
        }
        self.runtime.device.realize_deferred_clear()?;
        let mut ranges = RangeSet::default();
        for i in 0..capture.len() {
            for extent in capture[i].physical_extents() {
                for (address, bytes) in self.runtime.device_memory.device_ranges(extent) {
                    ranges.insert(address, bytes);
                }
            }
        }
        self.preserve_texture_writes(&ranges, || {
            Ok((0..capture.len())
                .flat_map(|i| capture[i].physical_extents())
                .map(|extent| (extent.start.get(), extent.bytes))
                .collect())
        })?;
        {
            let mut queries = self.caches.query_cache.lock().unwrap();
            for (address, bytes) in ranges.iter() {
                queries.invalidate_region(address, bytes);
            }
        }
        let mut offset = 0;
        for i in 0..capture.len() {
            let end = offset + capture[i].bytes() as usize;
            self.runtime
                .memory
                .write_resolved(std::slice::from_ref(&capture[i]), &bytes[offset..end])?;
            offset = end;
        }
        self.write_cpu_ranges(ranges.iter())
    }
    /// Read directly into owner-provided storage after resolving GPU authority.
    /// The capture retains every physical span for the complete access.
    #[cfg_attr(feature = "performance-counters", track_caller)]
    pub(crate) fn read_captured_memory(
        &mut self,
        capture: &nixe_memory::CapturedMemory,
        destination: &mut [u8],
    ) -> Result<(), VulkanError> {
        #[cfg(feature = "performance-counters")]
        let _trace =
            nixe_gpu::trace::span("read_captured", capture.bytes(), capture.len() as u64, 0);
        if capture.bytes() != destination.len() as u64 {
            return Err(VulkanError::InvalidResource(
                "owner read size differs from its capture",
            ));
        }
        self.runtime.device.realize_deferred_clear()?;
        for index in 0..capture.len() {
            for extent in capture[index].physical_extents() {
                for (address, bytes) in self.runtime.device_memory.device_ranges(extent) {
                    self.flush_region(address, bytes)?;
                }
            }
        }
        let mut offset = 0;
        for index in 0..capture.len() {
            let range = &capture[index];
            let end = offset + range.bytes() as usize;
            self.runtime
                .memory
                .capture_stream(std::slice::from_ref(range), &mut destination[offset..end])?;
            offset = end;
        }
        Ok(())
    }
    pub(crate) fn flush_region(&mut self, address: u64, bytes: u64) -> Result<(), VulkanError> {
        self.runtime.device.realize_deferred_clear()?;
        self.download_texture_memory(address, bytes)?;
        self.read_buffer_memory(address, bytes)?;
        self.caches
            .query_cache
            .lock()
            .unwrap()
            .flush_region(address, bytes)
    }

    pub(crate) fn progress_native_work(&mut self) -> Result<(), VulkanError> {
        self.runtime.device.poll_progress()?;
        self.buffer_cache.publish_completed_reads()
    }

    pub(crate) fn queue_device_access(
        &mut self,
        mut request: DeviceAccessRequest,
    ) -> Result<(), VulkanError> {
        if request.is_cancelled() {
            request.finish(Err(Arc::new(nixe_memory::MemoryError::Execution(
                "GPU memory request was cancelled".into(),
            ))));
            return Ok(());
        }
        let result = {
            let (ranges, operation) = request.parts_mut();
            self.access_memory(ranges, operation)
        };
        match result {
            Ok(result) => {
                request.finish(Ok(result));
                Ok(())
            }
            Err(error) => {
                let cause = match &error {
                    VulkanError::Memory(cause) => Arc::clone(cause),
                    other => Arc::new(nixe_memory::MemoryError::Execution(
                        other.to_string().into(),
                    )),
                };
                request.finish(Err(cause));
                Err(error)
            }
        }
    }

    pub(crate) fn access_memory(
        &mut self,
        ranges: &[AllocationRange],
        operation: &mut AccessOperation,
    ) -> Result<AccessResult, VulkanError> {
        self.runtime.device.realize_deferred_clear()?;
        if let AccessOperation::PrepareRead(area) = operation {
            if area.start >= area.end || area.end > 1 << self.runtime.device_memory.address_bits() {
                return Err(VulkanError::InvalidResource(
                    "invalid rasterizer flush area",
                ));
            }
            #[cfg(feature = "performance-counters")]
            let _trace =
                nixe_gpu::trace::span("read_cpu_prepare", area.start, area.end - area.start, 0);
            // Buffer-only reads return after submission. A query store has a
            // real dependency on overlapping read copies and resolves them
            // before publishing its value on this ordered owner boundary.
            self.download_texture_memory(area.start, area.end - area.start)?;
            let end = self.prepare_buffer_read(area.start, area.end - area.start)?;
            if self
                .caches
                .query_cache
                .lock()
                .unwrap()
                .must_flush_region(area.start, area.end - area.start)
            {
                self.finish_buffer_reads(area.start, area.end - area.start)?;
                self.caches
                    .query_cache
                    .lock()
                    .unwrap()
                    .flush_region(area.start, area.end - area.start)?;
            }
            return Ok(AccessResult::ReadPrepared(Box::new(ReadCompletion {
                caches: Arc::clone(self.caches),
                end,
            })));
        }
        // Scalar CPU reads use GetFlushArea above. Other coherent reads share
        // the texture, buffer and query publication order.
        let mut device_ranges = RangeSet::default();
        for extent in ranges.iter().flat_map(AllocationRange::physical_extents) {
            for (address, bytes) in self.runtime.device_memory.device_ranges(extent) {
                device_ranges.insert(address, bytes);
            }
        }
        if let AccessOperation::Atomic(_) = operation {
            // A successful atomic replaces RAM and invalidates the image even
            // when just its addressed word was already mirrored by DMA.
            for (address, bytes) in device_ranges.iter() {
                self.flush_region(address, bytes)?;
            }
        }
        if matches!(
            operation,
            AccessOperation::Write(_)
                | AccessOperation::Publish { .. }
                | AccessOperation::Atomic(_)
        ) {
            // Texture invalidation replaces a whole representation. Preserve
            // its untouched bytes before a partial canonical write. Buffer
            // invalidation already retains exact unwritten native intervals.
            self.preserve_texture_writes(&device_ranges, || {
                Ok(ranges
                    .iter()
                    .flat_map(AllocationRange::physical_extents)
                    .map(|extent| (extent.start.get(), extent.bytes))
                    .collect())
            })?;
        } else {
            for (address, bytes) in device_ranges.iter() {
                self.flush_region(address, bytes)?;
            }
        }
        let result = self
            .runtime
            .memory
            .apply_resolved_access(ranges, operation)?;
        if matches!(
            result,
            AccessResult::Written(_)
                | AccessResult::Published(_)
                | AccessResult::Atomic { written: true, .. }
        ) {
            self.write_cpu_ranges(device_ranges.iter())?;
        }
        Ok(result)
    }
}

// The owner captures the exact prefix. A subsequent request cannot extend this
// reader's GPU wait; query publication has already been resolved in order.
struct ReadCompletion {
    caches: Arc<cache_owners::CacheOwners>,
    end: Option<buffer_download::DownloadEnd>,
}
impl std::fmt::Debug for ReadCompletion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReadCompletion")
            .field("work", &self.end.map(|end| end.work()))
            .finish()
    }
}
impl nixe_memory::PreparedRead for ReadCompletion {
    fn finish(self: Box<Self>) -> Result<(), nixe_memory::MemoryError> {
        // Release the cache lock for the actual wait. Another reader or fence
        // can publish the prefix first; consuming it again is then a no-op.
        // https://github.com/zortos293/Lichii/blob/b4b830c8faf1fb5d7b2ad1b892117f4cfd708546/src/video_core/buffer_cache/buffer_cache.h#L789-L819
        let finish = || {
            if let Some(end) = self.end {
                #[cfg(feature = "performance-counters")]
                let _trace = nixe_gpu::trace::span("read_cpu_wait", end.work().tick(), 0, 0);
                self.caches
                    .timeline
                    .wait(&self.caches.progress, end.work().tick())?;
                self.caches
                    .buffer_cache
                    .lock()
                    .unwrap()
                    .publish_reads_through(end)?;
            }
            Ok(())
        };
        finish().map_err(|error: VulkanError| {
            nixe_memory::MemoryError::Execution(error.to_string().into())
        })
    }
}
