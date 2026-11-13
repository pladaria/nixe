//! Guest query values belong to the query cache, independently of buffer downloads.
//! CounterReport queues CPU stores through SignalFence/SyncOperation; NotifyWFI
//! synchronizes their values into native buffers for subsequent GPU consumers.
//! https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/query_cache/query_cache.h
use super::*;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};

pub(crate) struct Query {
    address: u64,
    machine: Machine,
    target: Vec<nixe_memory::AllocationRange>,
    bytes: Arc<[u8]>,
    invalidated: AtomicBool,
    rewritten: AtomicBool,
    gpu_rewritten: AtomicBool,
}
impl Query {
    pub(super) fn outdate(&self, address: u64, bytes: u64) {
        if self.address < address + bytes && address < self.address + self.bytes.len() as u64 {
            self.gpu_rewritten.store(true, Ordering::Relaxed);
        }
    }
    pub(super) fn pending_overwrite(&self) -> Option<(u64, u64)> {
        (self.gpu_rewritten.load(Ordering::Relaxed) && !self.invalidated.load(Ordering::Relaxed))
            .then_some((self.address, self.bytes.len() as u64))
    }
    fn value(&self) -> &[u8] {
        &self.bytes[..if self.bytes.len() == 4 { 4 } else { 8 }]
    }
    fn write(&self, bytes: &[u8]) -> Result<(), nixe_memory::MemoryError> {
        self.machine.write_resources(&self.target, |writer| {
            // Publish a long report's timestamp before its completion value.
            let parts = if bytes.len() == 16 {
                [8..16, 0..8]
            } else {
                [0..bytes.len(), 0..0]
            };
            for part in parts {
                let mut offset = 0usize;
                for extent in self
                    .target
                    .iter()
                    .flat_map(nixe_memory::AllocationRange::physical_extents)
                {
                    let begin = offset.max(part.start);
                    let end = (offset + extent.bytes as usize).min(part.end);
                    if begin < end {
                        writer.write(
                            nixe_memory::Extent {
                                start: nixe_memory::ArenaOffset::new(
                                    extent.start.get() + (begin - offset) as u64,
                                ),
                                bytes: (end - begin) as u64,
                            },
                            &bytes[begin..end],
                        )?;
                    }
                    offset += extent.bytes as usize;
                }
            }
            Ok(())
        })
    }
}
pub(crate) struct QueryCache {
    uniform_sync: Arc<std::sync::atomic::AtomicU64>,
    cached: BTreeMap<u64, Arc<Query>>,
    pending_sync: Vec<Arc<Query>>,
    pub(super) canonical_readback_bytes: u64,
}
impl QueryCache {
    pub(super) fn new(uniform_sync: Arc<std::sync::atomic::AtomicU64>) -> Self {
        Self {
            uniform_sync,
            cached: Default::default(),
            pending_sync: Vec::new(),
            canonical_readback_bytes: 0,
        }
    }
    pub(super) fn invalidate_region(&mut self, address: u64, bytes: u64) {
        let end = address.saturating_add(bytes);
        let start = address.saturating_sub(15);
        let keys: Vec<_> = self
            .cached
            .range(start..end)
            .filter_map(|(&at, query)| (at + query.bytes.len() as u64 > address).then_some(at))
            .collect();
        for key in keys {
            if let Some(query) = self.cached.remove(&key) {
                query.invalidated.store(true, Ordering::Relaxed);
            }
        }
    }
    fn unregister(&mut self, query: &Arc<Query>) {
        if self
            .cached
            .get(&query.address)
            .is_some_and(|current| Arc::ptr_eq(current, query))
        {
            self.cached.remove(&query.address);
        }
    }
    fn publish(
        &mut self,
        query: &Arc<Query>,
        result: Result<(), Arc<nixe_memory::MemoryError>>,
    ) -> Result<(), Arc<nixe_memory::MemoryError>> {
        let result = result.and_then(|()| {
            if !query.invalidated.load(Ordering::Relaxed) {
                query.write(&query.bytes).map_err(Arc::new)?;
                self.uniform_sync.fetch_add(1, Ordering::Release);
                self.canonical_readback_bytes += query.bytes.len() as u64;
            }
            Ok(())
        });
        if result.is_err() {
            query.invalidated.store(true, Ordering::Relaxed);
        }
        self.unregister(query);
        result
    }
    pub(super) fn must_flush_region(&self, address: u64, bytes: u64) -> bool {
        self.cached
            .range(address.saturating_sub(15)..address.saturating_add(bytes))
            .any(|(&at, query)| {
                at + query.bytes.len() as u64 > address
                    && !query.invalidated.load(Ordering::Relaxed)
            })
    }
    pub(super) fn flush_region(&mut self, address: u64, bytes: u64) -> Result<(), VulkanError> {
        let end = address.saturating_add(bytes);
        for (&at, query) in self.cached.range(address.saturating_sub(15)..end) {
            if at + query.bytes.len() as u64 > address && !query.invalidated.load(Ordering::Relaxed)
            {
                query.write(query.value())?;
                self.uniform_sync.fetch_add(1, Ordering::Release);
                self.canonical_readback_bytes += query.value().len() as u64;
            }
        }
        Ok(())
    }
}
impl Residency<'_> {
    pub(crate) fn report_query(
        &mut self,
        target: Vec<nixe_memory::AllocationRange>,
        bytes: Arc<[u8]>,
    ) -> Result<nixe_gpu::FenceAction, VulkanError> {
        self.caches.sync_guest_memory();
        if !matches!(bytes.len(), 4 | 8 | 16) {
            return Err(VulkanError::InvalidResource(
                "report requires four, eight or sixteen bytes",
            ));
        }
        let mapping = PhysicalMapping::new(&target)?;
        if mapping.bytes() != bytes.len() as u64 {
            return Err(VulkanError::InvalidResource(
                "query payload differs from retained destination",
            ));
        }
        let address = self
            .runtime
            .device_memory
            .find_address(&mapping, 0..mapping.bytes())?
            .ok_or(VulkanError::InvalidResource(
                "query destination has no device address",
            ))?;
        self.runtime
            .device_memory
            .mark_gpu_written(address, bytes.len() as u64);
        self.caches.command_pages.draw_write(
            &self.runtime.device_memory,
            address,
            bytes.len() as u64,
        );
        let query = Arc::new(Query {
            address,
            machine: self.runtime.memory.clone(),
            target,
            bytes,
            invalidated: AtomicBool::new(false),
            rewritten: AtomicBool::new(false),
            gpu_rewritten: AtomicBool::new(false),
        });
        let report_noted = self.buffer_cache.note_report_pending(address, &query);
        let caches = Arc::clone(self.caches);
        {
            let mut cache = caches.query_cache.lock().unwrap();
            if let Some(previous) = cache.cached.insert(address, Arc::clone(&query)) {
                previous.rewritten.store(true, Ordering::Relaxed);
            }
            cache.pending_sync.push(Arc::clone(&query));
        }
        Ok(Box::new(move |result| {
            // A report must land after overlapping reactive copies. The wait
            // uses only the shared timeline, with no cache lock held.
            // https://github.com/zortos293/Lichii/blob/b4b830c8faf1fb5d7b2ad1b892117f4cfd708546/src/video_core/query_cache/query_cache.h#L684-L694
            let result = result.and_then(|()| {
                caches
                    .finish_buffer_reads(query.address, query.bytes.len() as u64)
                    .map_err(|error| {
                        Arc::new(nixe_memory::MemoryError::Execution(
                            error.to_string().into(),
                        ))
                    })
            });
            // Only learned readback pages need a buffer-cache lock at report
            // publication. Keep that lock before the query lock, as on the owner.
            let mut buffers = report_noted.then(|| caches.buffer_cache.lock().unwrap());
            let result = caches.query_cache.lock().unwrap().publish(&query, result);
            if let Some(buffers) = buffers.as_mut() {
                if let Some((address, bytes)) = query.pending_overwrite()
                    && buffers.gpu_modified_ranges.intersects(address, bytes)
                {
                    buffers.pending_download_ranges.insert(address, bytes);
                }
                buffers
                    .pending_reports
                    .retain(|pending| !Arc::ptr_eq(pending, &query));
            }
            result
        }))
    }
    pub(super) fn sync_query_values(&mut self) -> Result<(), VulkanError> {
        let pending = std::mem::take(&mut self.caches.query_cache.lock().unwrap().pending_sync);
        for query in pending {
            if query.invalidated.load(Ordering::Relaxed) || query.rewritten.load(Ordering::Relaxed)
            {
                continue;
            }
            let value = query.value();
            let range = self.device_buffer(
                query.address,
                value.len() as u64,
                vk::BufferUsageFlags::TRANSFER_DST,
                true,
                false,
            )?;
            // SyncValues obtains a synchronized buffer without marking it as a
            // downloadable GPU write: the query's fence owns CPU publication.
            self.upload_buffer(range.buffer, range.offset, value)?;
        }
        Ok(())
    }
}
