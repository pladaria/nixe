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
#[derive(Default)]
pub(crate) struct CpuWrites {
    enabled: AtomicBool,
    cores: RwLock<Vec<Arc<CpuWriteCollector>>>,
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
    pub fn gather_gpu_dirty_memory(&self, mut invalidate: impl FnMut(u64, u64)) {
        for core in self.cpu_writes().cores.read().unwrap().iter() {
            core.gather(&mut invalidate);
        }
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
            let base = (record >> 32) << BLOCK_BITS;
            let mut bits = record as u32;
            while bits != 0 {
                let low = bits.trailing_zeros();
                let high = low + (bits >> low).trailing_ones();
                invalidate(
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
                self.collect(address, bytes)
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
    #[test]
    fn independent_cores_keep_their_current_blocks_and_shutdown_clears_batches() {
        let machine = Machine::new(4096).unwrap();
        machine.enable_cpu_write_collection().unwrap();
        let first = machine.cpu_writes().register();
        let second = machine.cpu_writes().register();
        first.collect(0x10000, 4);
        second.collect(0x20000, 4);
        let mut seen = Vec::new();
        machine.gather_gpu_dirty_memory(|address, size| seen.push((address, size)));
        assert_eq!(seen, [(0x10000, 64), (0x20000, 64)]);
        first.collect(0x30000, 4);
        second.collect(0x40000, 4);
        machine.disable_cpu_write_collection();
        machine.gather_gpu_dirty_memory(|_, _| panic!("shutdown left pending stores"));
    }
}
