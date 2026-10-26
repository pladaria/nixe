//! Read-only cache telemetry shared with the host diagnostics thread.
//! Writers publish on cache mutations; warm lookups need no atomic operations.
use std::sync::{
    Arc,
    atomic::{AtomicU64, AtomicUsize, Ordering},
};

#[derive(Clone)]
pub struct CacheCounters(Arc<Counters>);

struct Counters {
    entries: AtomicUsize,
    bytes: AtomicUsize,
    misses: AtomicU64,
    evictions: AtomicU64,
    entry_limit: usize,
    byte_limit: usize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CacheSnapshot {
    pub entries: usize,
    pub bytes: usize,
    pub entry_limit: usize,
    pub byte_limit: usize,
    pub misses: u64,
    pub evictions: u64,
}

impl CacheCounters {
    pub fn new(entry_limit: usize, byte_limit: usize) -> Self {
        Self(Arc::new(Counters {
            entries: AtomicUsize::new(0),
            bytes: AtomicUsize::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            entry_limit,
            byte_limit,
        }))
    }

    pub fn publish_usage(&self, entries: usize, bytes: usize) {
        self.0.entries.store(entries, Ordering::Relaxed);
        self.0.bytes.store(bytes, Ordering::Relaxed);
    }

    pub fn miss(&self) {
        self.0.misses.fetch_add(1, Ordering::Relaxed);
    }

    pub fn add_entries(&self, count: usize) {
        self.0.entries.fetch_add(count, Ordering::Relaxed);
    }

    pub fn remove_entries(&self, count: usize) {
        self.0.entries.fetch_sub(count, Ordering::Relaxed);
    }

    pub fn evict(&self, count: u64) {
        if count != 0 {
            self.0.evictions.fetch_add(count, Ordering::Relaxed);
        }
    }

    /// Concurrent mutations may straddle this sample. No cache ownership or
    /// GPU resources are retained, and sampling never locks an execution owner.
    pub fn snapshot(&self) -> CacheSnapshot {
        CacheSnapshot {
            entries: self.0.entries.load(Ordering::Relaxed),
            bytes: self.0.bytes.load(Ordering::Relaxed),
            entry_limit: self.0.entry_limit,
            byte_limit: self.0.byte_limit,
            misses: self.0.misses.load(Ordering::Relaxed),
            evictions: self.0.evictions.load(Ordering::Relaxed),
        }
    }
}
