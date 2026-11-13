//! Shader identities and physical-page invalidation shared by instruction sets.
use nixe_memory::device_memory::DeviceMemory;
use rustc_hash::FxHashMap as HashMap;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

const PAGE_BITS: u32 = 14;
const PAGE_SIZE: u64 = 1 << PAGE_BITS;

/// Immutable shader facts. Current channel selections retain these facts,
/// independently of cache page marks and owned compiler environments.
#[derive(Debug)]
pub struct ShaderInfo {
    pub unique_hash: u64,
    pub size_bytes: usize,
    valid: AtomicBool,
}
impl ShaderInfo {
    pub fn new(unique_hash: u64, size_bytes: usize) -> Self {
        Self {
            unique_hash,
            size_bytes,
            valid: AtomicBool::new(true),
        }
    }
    pub fn is_valid(&self) -> bool {
        self.valid.load(Ordering::Acquire)
    }
}
struct Entry {
    isa: nixe_shader_recompiler::environment::ShaderIsa,
    address: u64,
    end: u64,
    info: Arc<ShaderInfo>,
    memory_marked: AtomicBool,
}
#[derive(Default)]
struct Invalidations {
    pages: HashMap<u64, Vec<Arc<Entry>>>,
    pending: Vec<Arc<Entry>>,
}
struct Cache {
    memory: DeviceMemory,
    counters: crate::cache_counters::CacheCounters,
    lookup: Mutex<HashMap<(u64, nixe_shader_recompiler::environment::ShaderIsa), Arc<Entry>>>,
    invalidations: Mutex<Invalidations>,
}
impl Drop for Cache {
    fn drop(&mut self) {
        for entry in self.lookup.get_mut().unwrap().values() {
            if !entry.memory_marked.load(Ordering::Relaxed) {
                continue;
            }
            self.memory
                .update_pages_cached_count(entry.address, entry.end - entry.address, -1)
                .expect("registered shader cache pages");
        }
    }
}
#[derive(Clone)]
pub struct ShaderCache(Arc<Cache>);
impl ShaderCache {
    pub fn new(memory: DeviceMemory) -> Self {
        Self(Arc::new(Cache {
            memory,
            counters: crate::cache_counters::CacheCounters::new(0, 0),
            lookup: Mutex::new(HashMap::default()),
            invalidations: Mutex::new(Invalidations::default()),
        }))
    }
    pub fn cache_counters(&self) -> crate::cache_counters::CacheCounters {
        self.0.counters.clone()
    }
    pub fn try_get(
        &self,
        address: u64,
        isa: nixe_shader_recompiler::environment::ShaderIsa,
    ) -> Option<Arc<ShaderInfo>> {
        self.0
            .lookup
            .lock()
            .unwrap()
            .get(&(address, isa))
            .filter(|entry| entry.info.is_valid())
            .map(|entry| Arc::clone(&entry.info))
    }
    pub fn register(
        &self,
        address: u64,
        isa: nixe_shader_recompiler::environment::ShaderIsa,
        info: ShaderInfo,
    ) -> nixe_shader_recompiler::Result<Arc<ShaderInfo>> {
        let info = Arc::new(info);
        let bytes = info.size_bytes as u64;
        let end = address
            .checked_add(bytes)
            .ok_or_else(|| invalid("Shader cache range overflows"))?;
        let entry = Arc::new(Entry {
            isa,
            address,
            end,
            info: Arc::clone(&info),
            memory_marked: AtomicBool::new(true),
        });
        // Register publishes one address entry and its page index together;
        // only these cache-owned entries mark device pages cached.
        let mut invalidations = self.0.invalidations.lock().unwrap();
        let mut lookup = self.0.lookup.lock().unwrap();
        Self::retire_pending(&self.0.counters, &mut invalidations, &mut lookup);
        if let Some(old) = lookup.get(&(address, isa)) {
            return Ok(Arc::clone(&old.info));
        }
        self.0
            .memory
            .update_pages_cached_count(address, bytes, 1)
            .map_err(|e| invalid(e.to_string()))?;
        lookup.insert((address, isa), Arc::clone(&entry));
        self.0.counters.miss();
        self.0.counters.add_entries(1);
        for page in address >> PAGE_BITS..end.div_ceil(PAGE_SIZE) {
            invalidations
                .pages
                .entry(page)
                .or_default()
                .push(Arc::clone(&entry));
        }
        Ok(info)
    }
    /// CPU writes remove invalidated lookups immediately. Unmaps only unmark
    /// and detach page entries; their lookups retire at the next normal
    /// invalidation, as ShaderCache::OnCacheInvalidation/RemovePendingShaders.
    pub fn invalidate_region(&self, address: u64, bytes: u64) {
        let mut invalidations = self.0.invalidations.lock().unwrap();
        self.invalidate_pages(&mut invalidations, address, bytes);
        if invalidations.pending.is_empty() {
            return;
        }
        let mut lookup = self.0.lookup.lock().unwrap();
        Self::retire_pending(&self.0.counters, &mut invalidations, &mut lookup);
    }
    fn retire_pending(
        counters: &crate::cache_counters::CacheCounters,
        invalidations: &mut Invalidations,
        lookup: &mut HashMap<(u64, nixe_shader_recompiler::environment::ShaderIsa), Arc<Entry>>,
    ) {
        invalidations.pending.sort_unstable_by_key(Arc::as_ptr);
        invalidations.pending.dedup_by(|a, b| Arc::ptr_eq(a, b));
        let count = invalidations.pending.len();
        for entry in invalidations.pending.drain(..) {
            lookup.remove(&(entry.address, entry.isa));
        }
        counters.remove_entries(count);
        counters.evict(count as u64);
    }
    pub fn on_cache_invalidation(&self, address: u64, bytes: u64) {
        self.invalidate_pages(&mut self.0.invalidations.lock().unwrap(), address, bytes);
    }
    fn invalidate_pages(&self, invalidations: &mut Invalidations, address: u64, bytes: u64) {
        if bytes == 0 {
            return;
        }
        let end = address + bytes;
        for page in address >> PAGE_BITS..end.div_ceil(PAGE_SIZE) {
            loop {
                let entry = invalidations
                    .pages
                    .get(&page)
                    .and_then(|entries| {
                        entries
                            .iter()
                            .find(|entry| entry.address < end && address < entry.end)
                    })
                    .cloned();
                let Some(entry) = entry else {
                    break;
                };
                entry.info.valid.store(false, Ordering::Release);
                if entry.memory_marked.swap(false, Ordering::Relaxed) {
                    self.0
                        .memory
                        .update_pages_cached_count(entry.address, entry.end - entry.address, -1)
                        .expect("invalidated shader cache pages");
                }
                for entry_page in entry.address >> PAGE_BITS..entry.end.div_ceil(PAGE_SIZE) {
                    let entries = invalidations
                        .pages
                        .get_mut(&entry_page)
                        .expect("registered shader invalidation page");
                    let index = entries
                        .iter()
                        .position(|old| Arc::ptr_eq(old, &entry))
                        .expect("registered shader page entry");
                    entries.remove(index);
                }
                invalidations.pending.push(entry);
            }
        }
    }
}

fn invalid(message: impl Into<String>) -> nixe_shader_recompiler::CompileError {
    nixe_shader_recompiler::CompileError::new(
        nixe_shader_recompiler::diagnostics::FailureCategory::InvalidArgument,
        message,
    )
}
