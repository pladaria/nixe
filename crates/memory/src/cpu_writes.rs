//! Per-core GPUDirtyMemoryManager: DAddr blocks with 64-byte dirty masks.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/gpu_dirty_memory_manager.h
use crate::{Extent, Machine, MemoryError};
use std::cell::RefCell;
use std::sync::{
    Arc, Mutex, RwLock,
    atomic::{AtomicBool, AtomicU64, Ordering},
};

const EMPTY: u64 = (u32::MAX as u64) << 32;
const BLOCK_BITS: u32 = 11;
const BLOCK_BYTES: u64 = 1 << BLOCK_BITS;
const ALIGN_BITS: u32 = 6;
const ALIGN_BYTES: u64 = 1 << ALIGN_BITS;
type WriteBatch = Arc<Vec<(u64, u64)>>;
#[derive(Default)]
pub(crate) struct CpuWrites {
    enabled: AtomicBool,
    cores: RwLock<Vec<Arc<CpuWriteCollector>>>,
    applying: Mutex<Vec<WriteBatch>>,
    landing: Landing,
}

// Publication runs under the buffer cache lock, so only one landing is active.
// Store/load fences ensure that a collected CPU write is either seen by the
// download's snapshot or stores its bytes after that download has landed.
// https://github.com/zortos293/Lichii/blob/b4b830c8faf1fb5d7b2ad1b892117f4cfd708546/src/video_core/rasterizer_download_area.h#L45-L113
#[derive(Default)]
struct Landing {
    sequence: AtomicU64,
    begin: AtomicU64,
    end: AtomicU64,
}
pub struct DownloadLanding<'a> {
    landing: &'a Landing,
}
impl Drop for DownloadLanding<'_> {
    fn drop(&mut self) {
        self.landing.sequence.fetch_add(1, Ordering::Release);
    }
}
impl Landing {
    fn wait(&self, address: u64, bytes: u64) {
        std::sync::atomic::fence(Ordering::SeqCst);
        let sequence = self.sequence.load(Ordering::Acquire);
        if sequence & 1 == 0 {
            return;
        }
        let begin = self.begin.load(Ordering::Relaxed);
        let end = self.end.load(Ordering::Relaxed);
        std::sync::atomic::fence(Ordering::Acquire);
        if self.sequence.load(Ordering::Acquire) != sequence
            || address >= end
            || address + bytes <= begin
        {
            return;
        }
        while self.sequence.load(Ordering::Acquire) == sequence {
            std::thread::yield_now();
        }
    }
}
struct ApplyingWrites<'a> {
    owner: &'a CpuWrites,
    ranges: WriteBatch,
}
impl Drop for ApplyingWrites<'_> {
    fn drop(&mut self) {
        self.owner
            .applying
            .lock()
            .unwrap()
            .retain(|ranges| !Arc::ptr_eq(ranges, &self.ranges));
    }
}
pub(crate) struct CpuWriteCollector {
    current: AtomicU64,
    pub(crate) last_gpu_page: AtomicU64,
    journal: Mutex<Journal>,
}
impl Default for CpuWriteCollector {
    fn default() -> Self {
        Self {
            current: AtomicU64::new(EMPTY),
            last_gpu_page: AtomicU64::new(u64::MAX),
            journal: Mutex::default(),
        }
    }
}
struct Journal {
    back: Vec<u64>,
    front: Vec<u64>,
}
impl Default for Journal {
    fn default() -> Self {
        Self {
            back: Vec::with_capacity(256),
            front: Vec::with_capacity(256),
        }
    }
}
impl Machine {
    pub fn enable_cpu_write_collection(&self) -> Result<(), MemoryError> {
        self.cpu_writes()
            .enabled
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| MemoryError::Invalid("CPU writes already have a graphics consumer"))?;
        Ok(())
    }
    pub fn disable_cpu_write_collection(&self) {
        self.cpu_writes().enabled.store(false, Ordering::Release);
        for core in self.cpu_writes().cores.read().unwrap().iter() {
            let mut journal = core.journal.lock().unwrap();
            core.current.store(EMPTY, Ordering::Release);
            core.last_gpu_page.store(u64::MAX, Ordering::Release);
            journal.back.clear();
            journal.front.clear();
        }
    }
    /// GPU::InvalidateGPUCache gathers actual device-address notifications.
    /// There is one graphics consumer; a fence publication never gathers stores.
    pub fn gather_gpu_dirty_memory<E>(
        &self,
        invalidate: impl FnOnce(&[(u64, u64)]) -> Result<(), E>,
    ) -> Result<(), E> {
        // Keep the batch visible while invalidation acquires the cache locks.
        // Snapshotters take this mutex only to copy metadata, never to wait on
        // an invalidation callback (which could be waiting for their cache lock).
        let owner = self.cpu_writes();
        let batch = {
            let mut applying = owner.applying.lock().unwrap();
            let mut ranges = Vec::new();
            for core in owner.cores.read().unwrap().iter() {
                core.gather(&mut |address, bytes| ranges.push((address, bytes)));
            }
            if ranges.is_empty() {
                return Ok(());
            }
            let ranges = Arc::new(ranges);
            applying.push(Arc::clone(&ranges));
            ApplyingWrites { owner, ranges }
        };
        invalidate(&batch.ranges)
    }
    /// The caller must serialize buffer publications and keep this guard until
    /// all stores finish. It must not wait for native CPU quiescence in the scope.
    pub fn begin_gpu_download(&self, begin: u64, end: u64) -> DownloadLanding<'_> {
        let landing = &self.cpu_writes().landing;
        landing.begin.store(begin, Ordering::Relaxed);
        landing.end.store(end, Ordering::Relaxed);
        let previous = landing.sequence.fetch_add(1, Ordering::SeqCst);
        assert_eq!(
            previous & 1,
            0,
            "overlapping buffer download publication scopes"
        );
        std::sync::atomic::fence(Ordering::SeqCst);
        DownloadLanding { landing }
    }
    /// Include writes still collecting and batches currently being applied.
    /// Taking this snapshot does not consume either source.
    pub fn pending_gpu_dirty_memory(&self) -> crate::RangeSet {
        let owner = self.cpu_writes();
        let applying = owner.applying.lock().unwrap();
        let mut ranges = crate::RangeSet::default();
        for core in owner.cores.read().unwrap().iter() {
            core.pending(&mut |address, bytes| ranges.insert(address, bytes));
        }
        for batch in &*applying {
            ranges.extend(batch.iter().copied());
        }
        ranges
    }
}
impl CpuWrites {
    pub(crate) fn register(&self) -> Arc<CpuWriteCollector> {
        let mut cores = self.cores.write().unwrap();
        if let Some(core) = cores.iter().find(|core| {
            Arc::strong_count(core) == 1
                && core.current.load(Ordering::Acquire) == EMPTY
                && core.journal.lock().unwrap().back.is_empty()
        }) {
            return Arc::clone(core);
        }
        let core = Arc::new(CpuWriteCollector::default());
        cores.push(Arc::clone(&core));
        core
    }
}
impl CpuWriteCollector {
    fn visit(record: u64, visit: &mut impl FnMut(u64, u64)) {
        if record == EMPTY {
            return;
        }
        let base = (record >> 32) << BLOCK_BITS;
        let mut bits = record as u32;
        while bits != 0 {
            let low = bits.trailing_zeros();
            let high = low + (bits >> low).trailing_ones();
            visit(
                base + (u64::from(low) << ALIGN_BITS),
                u64::from(high - low) << ALIGN_BITS,
            );
            bits = if high == 32 {
                0
            } else {
                bits & (u32::MAX << high)
            };
        }
    }
    fn pending(&self, visit: &mut impl FnMut(u64, u64)) {
        let journal = self.journal.lock().unwrap();
        for &record in &journal.back {
            Self::visit(record, visit);
        }
        Self::visit(self.current.load(Ordering::Acquire), visit);
    }
    fn collect(&self, mut address: u64, mut bytes: u64) {
        while bytes != 0 {
            let count = bytes.min(BLOCK_BYTES - (address & (BLOCK_BYTES - 1)));
            let low = (address & (BLOCK_BYTES - 1)) >> ALIGN_BITS;
            let high = ((address & (BLOCK_BYTES - 1)) + count).div_ceil(ALIGN_BYTES);
            let mask = ((u32::MAX >> (32 - high)) & (u32::MAX << low)) as u64;
            let next = ((address >> BLOCK_BITS) << 32) | mask;
            loop {
                let current = self.current.load(Ordering::Acquire);
                if current >> 32 == next >> 32 {
                    if current | next == current {
                        break;
                    }
                    if self
                        .current
                        .compare_exchange_weak(
                            current,
                            current | next,
                            Ordering::Release,
                            Ordering::Relaxed,
                        )
                        .is_ok()
                    {
                        break;
                    }
                } else if current == EMPTY {
                    if self
                        .current
                        .compare_exchange_weak(current, next, Ordering::Release, Ordering::Relaxed)
                        .is_ok()
                    {
                        break;
                    }
                } else {
                    let mut journal = self.journal.lock().unwrap();
                    let previous = self.current.swap(next, Ordering::AcqRel);
                    if previous != EMPTY {
                        journal.back.push(previous);
                    }
                    break;
                }
            }
            address += count;
            bytes -= count;
        }
    }
    fn gather(&self, invalidate: &mut impl FnMut(u64, u64)) {
        let mut front = {
            let mut journal = self.journal.lock().unwrap();
            let current = self.current.swap(EMPTY, Ordering::AcqRel);
            let spare = std::mem::take(&mut journal.front);
            let mut front = std::mem::replace(&mut journal.back, spare);
            if current != EMPTY {
                front.push(current);
            }
            front
        };
        for &record in &front {
            Self::visit(record, invalidate);
        }
        front.clear();
        self.journal.lock().unwrap().front = front;
    }
    /// Native execution or captured ranges retain RAM through the store. The
    /// core's device manager translates journal addresses without another lookup.
    pub(crate) fn write(
        &self,
        machine: &Machine,
        device: &crate::device_memory::DeviceMemory,
        extents: &[Extent],
        scratch: &RefCell<Vec<u32>>,
        write: impl FnOnce(),
    ) -> Result<bool, MemoryError> {
        if !machine.cpu_writes().enabled.load(Ordering::Acquire) {
            return Ok(false);
        }
        for &extent in extents {
            device.apply_on_physical(extent, scratch, |address, bytes| {
                self.collect(address, bytes);
                machine.cpu_writes().landing.wait(address, bytes);
            });
        }
        write();
        Ok(true)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn gather(core: &CpuWriteCollector) -> Vec<(u64, u64)> {
        let mut ranges = Vec::new();
        core.gather(&mut |address, size| ranges.push((address, size)));
        ranges
    }
    #[test]
    fn dirty_masks_round_to_64_bytes_and_keep_gaps() {
        let core = CpuWriteCollector::default();
        core.collect(0x10003, 3);
        core.collect(0x1000a, 2);
        core.collect(0x10080, 1);
        assert_eq!(gather(&core), [(0x10000, 64), (0x10080, 64)]);
        assert!(gather(&core).is_empty());
    }
    #[test]
    fn stores_split_at_2k_blocks_and_4k_pages() {
        let core = CpuWriteCollector::default();
        core.collect(0x107ff, 2);
        core.collect(0x10fff, 2);
        assert_eq!(
            gather(&core),
            [(0x107c0, 64), (0x10800, 64), (0x10fc0, 64), (0x11000, 64)]
        );
    }
    #[test]
    fn changing_blocks_keeps_notification_order_without_sorting_or_pin_capture() {
        let core = CpuWriteCollector::default();
        for address in [0x30000, 0x10000, 0x30010] {
            core.collect(address, 4);
        }
        assert_eq!(gather(&core), [(0x30000, 64), (0x10000, 64), (0x30000, 64)]);
    }
    #[test]
    fn gathering_and_collecting_concurrently_does_not_lose_dirty_blocks() {
        let core = Arc::new(CpuWriteCollector::default());
        let producer = Arc::clone(&core);
        let finished = Arc::new(AtomicBool::new(false));
        let done = Arc::clone(&finished);
        let thread = std::thread::spawn(move || {
            for index in 0..4096 {
                producer.collect(0x10000 + index * 64, 4);
            }
            done.store(true, Ordering::Release);
        });
        let mut seen = crate::RangeSet::default();
        while !finished.load(Ordering::Acquire) {
            core.gather(&mut |address, size| seen.insert(address, size));
            std::thread::yield_now();
        }
        thread.join().unwrap();
        core.gather(&mut |address, size| seen.insert(address, size));
        assert_eq!(seen.iter().collect::<Vec<_>>(), [(0x10000, 4096 * 64)]);
    }
}
