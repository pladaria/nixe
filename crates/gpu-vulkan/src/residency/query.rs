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
}
impl Query {
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
#[derive(Default)]
pub(crate) struct QueryCache {
    cached: BTreeMap<u64, Arc<Query>>,
    pending_sync: Vec<Arc<Query>>,
    pub(super) canonical_readback_bytes: u64,
}
impl QueryCache {
    pub(super) fn invalidate_region(&mut self, address: u64, bytes: u64) {
        let end = address.saturating_add(bytes);
        let start = address.saturating_sub(3);
        let keys: Vec<_> = self
            .cached
            .range(start..end)
            .filter_map(|(&at, _)| (at + 4 > address).then_some(at))
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
    pub(super) fn flush_region(&mut self, address: u64, bytes: u64) -> Result<(), VulkanError> {
        let end = address.saturating_add(bytes);
        for (&at, query) in self.cached.range(address.saturating_sub(3)..end) {
            if at + 4 > address && !query.invalidated.load(Ordering::Relaxed) {
                query.write(query.value())?;
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
        if !matches!(bytes.len(), 4 | 16) {
            return Err(VulkanError::InvalidResource(
                "query report requires four or sixteen bytes",
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
        let query = Arc::new(Query {
            address,
            machine: self.runtime.memory.clone(),
            target,
            bytes,
            invalidated: AtomicBool::new(false),
            rewritten: AtomicBool::new(false),
        });
        let caches = Arc::clone(self.caches);
        {
            let mut cache = caches.query_cache.lock().unwrap();
            if let Some(previous) = cache.cached.insert(address, Arc::clone(&query)) {
                previous.rewritten.store(true, Ordering::Relaxed);
            }
            cache.pending_sync.push(Arc::clone(&query));
        }
        Ok(Box::new(move |result| {
            caches.query_cache.lock().unwrap().publish(&query, result)
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
