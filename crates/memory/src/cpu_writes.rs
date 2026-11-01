//! CPU dirty bytes use a compact per-core atomic accumulator and recycled batches.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/gpu_dirty_memory_manager.h
use crate::{AllocationRange, ArenaOffset, Machine, MemoryError, PAGE_SIZE};
use std::{
    collections::BTreeMap,
    sync::{
        Arc, Mutex, RwLock,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};

// Each bit represents one byte, preserving mixed CPU/device byte ownership.
// Native page-table physical indices fit u32. The compact 32-byte-address
// accumulator covers 128 GiB; larger physical addresses use normal access.
const EMPTY: u64 = u64::MAX;
const BLOCK_BYTES: u64 = 32;
#[derive(Default)]
pub(crate) struct CpuWrites {
    enabled: AtomicBool,
    cores: RwLock<Vec<Arc<CpuWriteCollector>>>,
}
pub(crate) struct CpuWriteCollector {
    current: AtomicU64,
    dirty: AtomicBool,
    journal: Mutex<Journal>,
}
impl Default for CpuWriteCollector {
    fn default() -> Self {
        Self {
            current: AtomicU64::new(EMPTY),
            dirty: AtomicBool::new(false),
            journal: Mutex::default(),
        }
    }
}
#[derive(Default)]
struct Batch {
    records: Vec<u64>,
    pages: BTreeMap<u64, AllocationRange>,
}
#[derive(Default)]
struct Journal {
    pending: Batch,
    spare: Batch,
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
            journal.pending.records.clear();
            journal.pending.pages.clear();
            core.dirty.store(false, Ordering::Release);
        }
    }
    pub fn collect_cpu_writes(&self) -> Result<Vec<AllocationRange>, MemoryError> {
        let cores = self.cpu_writes().cores.read().unwrap();
        let mut result = Vec::new();
        for core in cores.iter() {
            if !core.dirty.load(Ordering::Acquire) {
                continue;
            }
            let mut pending = {
                let mut journal = core.journal.lock().unwrap();
                let last = core.current.swap(EMPTY, Ordering::AcqRel);
                if last != EMPTY {
                    journal.pending.records.push(last);
                }
                let spare = std::mem::take(&mut journal.spare);
                let pending = std::mem::replace(&mut journal.pending, spare);
                core.dirty.store(false, Ordering::Release);
                pending
            };
            append_ranges(&mut pending, &mut result)?;
            pending.records.clear();
            pending.pages.clear();
            core.journal.lock().unwrap().spare = pending;
        }
        Ok(result)
    }
}
impl CpuWrites {
    pub(crate) fn register(&self) -> Arc<CpuWriteCollector> {
        let mut cores = self.cores.write().unwrap();
        if let Some(core) = cores
            .iter()
            .find(|core| Arc::strong_count(core) == 1 && !core.dirty.load(Ordering::Acquire))
        {
            return core.clone();
        }
        let core = Arc::new(CpuWriteCollector::default());
        cores.push(core.clone());
        core
    }
}
fn record(address: u64, bytes: u64) -> u64 {
    let mask = u32::MAX >> (32 - bytes);
    ((address / BLOCK_BYTES) << 32) | (u64::from(mask) << (address % BLOCK_BYTES))
}
impl CpuWriteCollector {
    /// Native mapping admission and physical guards retain every target page and
    /// exclude captures/downloads until the original store completes. Publishing
    /// its dirty mask before the store therefore cannot expose incomplete RAM.
    pub(crate) fn write(
        &self,
        machine: &Machine,
        address: u64,
        size: usize,
        physical: impl Fn(u64) -> ArenaOffset,
        mut capture: impl FnMut(u64) -> Result<AllocationRange, MemoryError>,
        write: impl FnOnce(),
    ) -> Result<bool, MemoryError> {
        if !machine.cpu_writes().enabled.load(Ordering::Acquire) {
            return Ok(false);
        }
        debug_assert!(size > 0 && size <= 16);
        let end = address + size as u64;
        let start = physical(address & !(PAGE_SIZE - 1)).get() + address % PAGE_SIZE;
        if start / BLOCK_BYTES >= u64::from(u32::MAX) {
            return Ok(false);
        }
        if start % BLOCK_BYTES + size as u64 <= BLOCK_BYTES {
            let added = record(start, size as u64);
            let mut current = self.current.load(Ordering::Acquire);
            while current >> 32 == added >> 32 {
                match self.current.compare_exchange_weak(
                    current,
                    current | added,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => {
                        write();
                        return Ok(true);
                    }
                    Err(next) => current = next,
                }
            }
        }
        let mut journal = self.journal.lock().unwrap();
        if !machine.cpu_writes().enabled.load(Ordering::Acquire) {
            return Ok(false);
        }
        // Capture both pages before publishing any byte of a cross-page store.
        let mut targets = [(0, 0); 2];
        let mut count = 0;
        for page in (address & !(PAGE_SIZE - 1)..end).step_by(PAGE_SIZE as usize) {
            let base = physical(page).get();
            let last = base + end.min(page + PAGE_SIZE) - page - 1;
            if last / BLOCK_BYTES >= u64::from(u32::MAX) {
                return Ok(false);
            }
            if let std::collections::btree_map::Entry::Vacant(entry) =
                journal.pending.pages.entry(base)
            {
                entry.insert(capture(page)?);
            }
            targets[count] = (
                base + address.max(page) - page,
                end.min(page + PAGE_SIZE) - address.max(page),
            );
            count += 1;
        }
        for &(mut start, mut bytes) in &targets[..count] {
            while bytes != 0 {
                let size = bytes.min(BLOCK_BYTES - start % BLOCK_BYTES);
                let added = record(start, size);
                let current = self.current.load(Ordering::Acquire);
                if current >> 32 == added >> 32 {
                    self.current.store(current | added, Ordering::Release);
                } else {
                    let old = self.current.swap(added, Ordering::AcqRel);
                    if old != EMPTY {
                        journal.pending.records.push(old);
                    }
                }
                start += size;
                bytes -= size;
            }
        }
        self.dirty.store(true, Ordering::Release);
        drop(journal);
        write();
        Ok(true)
    }
}
fn append_ranges(batch: &mut Batch, result: &mut Vec<AllocationRange>) -> Result<(), MemoryError> {
    batch.records.sort_unstable_by_key(|record| record >> 32);
    let mut write = 0;
    for read in 0..batch.records.len() {
        let record = batch.records[read];
        if write != 0 && batch.records[write - 1] >> 32 == record >> 32 {
            batch.records[write - 1] |= record;
        } else {
            batch.records[write] = record;
            write += 1;
        }
    }
    batch.records.truncate(write);
    let mut run: Option<(u64, u64)> = None;
    let mut emit = |(begin, end): (u64, u64)| {
        let page = &batch.pages[&(begin & !(PAGE_SIZE - 1))];
        result.push(AllocationRange::new(
            page.allocation().clone(),
            page.offset() + begin % PAGE_SIZE,
            end - begin,
        )?);
        Ok::<_, MemoryError>(())
    };
    for &record in &batch.records {
        let base = (record >> 32) * BLOCK_BYTES;
        let mut bits = record as u32;
        while bits != 0 {
            let low = bits.trailing_zeros();
            let high = low + (bits >> low).trailing_ones();
            let begin = base + u64::from(low);
            let end = base + u64::from(high);
            match run {
                Some((start, previous))
                    if previous == begin && start / PAGE_SIZE == begin / PAGE_SIZE =>
                {
                    run = Some((start, end))
                }
                previous => {
                    if let Some(previous) = previous {
                        emit(previous)?;
                    }
                    run = Some((begin, end));
                }
            }
            bits = if high == 32 {
                0
            } else {
                bits & (u32::MAX << high)
            };
        }
    }
    if let Some(run) = run {
        emit(run)?;
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AddressSpace, AddressWidth, GuestVirtualAddress, MemoryPermissions};
    use std::sync::Arc;

    fn fixture() -> (Machine, Arc<AddressSpace>, AllocationRange) {
        let machine = Machine::new(2 * PAGE_SIZE).unwrap();
        let allocation = machine
            .account(2 * PAGE_SIZE)
            .unwrap()
            .allocate(2 * PAGE_SIZE)
            .unwrap();
        let range = AllocationRange::new(allocation.pin(), 0, 2 * PAGE_SIZE).unwrap();
        let space = AddressSpace::new(machine.clone(), AddressWidth::Bits32).unwrap();
        for base in [PAGE_SIZE, 4 * PAGE_SIZE] {
            space
                .map(
                    GuestVirtualAddress::new(base),
                    allocation.clone(),
                    0,
                    2 * PAGE_SIZE,
                    MemoryPermissions::READ_WRITE,
                )
                .unwrap();
        }
        machine
            .publish_device_write(std::slice::from_ref(&range), || Ok(()))
            .unwrap();
        machine.enable_cpu_write_collection().unwrap();
        (machine, space, range)
    }

    #[test]
    fn native_aliases_collect_exact_bytes_and_do_not_copy_untouched_gpu_bytes() {
        let (machine, space, range) = fixture();
        let core = space.register_core(Arc::new(|| {}));
        let native = core.enter_native().unwrap();
        for (offset, bytes) in [
            (3, vec![7; 2]),
            (5, vec![8; 4]),
            (63, vec![9; 3]),
            (4094, vec![10; 4]),
        ] {
            assert!(
                space
                    .write_native(
                        &native,
                        GuestVirtualAddress::new(PAGE_SIZE + offset),
                        &bytes
                    )
                    .unwrap()
            );
        }
        assert!(
            space
                .write_native(
                    &native,
                    GuestVirtualAddress::new(4 * PAGE_SIZE + 4),
                    &[11; 2]
                )
                .unwrap()
        );
        drop(native);
        let dirty = machine.collect_cpu_writes().unwrap();
        assert_eq!(
            dirty
                .iter()
                .map(|r| (r.offset(), r.bytes()))
                .collect::<Vec<_>>(),
            [(3, 6), (63, 3), (4094, 2), (4096, 2)]
        );
        let mut ram = vec![0; (2 * PAGE_SIZE) as usize];
        machine.capture_stream(&[range], &mut ram).unwrap();
        assert_eq!(&ram[3..9], &[7, 11, 11, 8, 8, 8]);
        assert_eq!(&ram[62..67], &[0, 9, 9, 9, 0]);
        assert_eq!(&ram[4093..4099], &[0, 10, 10, 10, 10, 0]);
        assert!(machine.collect_cpu_writes().unwrap().is_empty());
        machine.disable_cpu_write_collection();
    }

    #[test]
    fn collection_is_detached_and_cores_have_independent_store_buffers() {
        let (machine, space, range) = fixture();
        let core = space.register_core(Arc::new(|| {}));
        let other = space.register_core(Arc::new(|| {}));
        let native = core.enter_native().unwrap();
        let another = other.enter_native().unwrap();
        let snapshot = machine.collect_cpu_writes().unwrap();
        assert!(
            space
                .write_native(&native, GuestVirtualAddress::new(PAGE_SIZE + 3), &[7; 4])
                .unwrap()
        );
        {
            let _held = native.cpu_writes().journal.lock().unwrap();
            assert!(
                space
                    .write_native(&another, GuestVirtualAddress::new(PAGE_SIZE + 12), &[8])
                    .unwrap()
            );
        }
        let dirty = machine.collect_cpu_writes().unwrap();
        assert_eq!(dirty.len(), 2);
        assert!(
            space
                .write_native(&native, GuestVirtualAddress::new(PAGE_SIZE + 7), &[9])
                .unwrap()
        );
        drop(native);
        drop(another);
        assert!(snapshot.is_empty());
        assert_eq!(machine.collect_cpu_writes().unwrap().len(), 1);
        let mut ram = vec![0; (2 * PAGE_SIZE) as usize];
        machine.capture_stream(&[range], &mut ram).unwrap();
        assert_eq!(&ram[3..8], &[7, 7, 7, 7, 9]);
        assert_eq!(ram[12], 8);
        machine.disable_cpu_write_collection();
    }

    #[test]
    fn repeated_stores_in_one_block_do_not_lock_the_journal() {
        let (machine, space, _) = fixture();
        let core = space.register_core(Arc::new(|| {}));
        let native = core.enter_native().unwrap();
        assert!(
            space
                .write_native(&native, GuestVirtualAddress::new(PAGE_SIZE + 3), &[1])
                .unwrap()
        );
        {
            let _held = native.cpu_writes().journal.lock().unwrap();
            assert!(
                space
                    .write_native(&native, GuestVirtualAddress::new(PAGE_SIZE + 4), &[2])
                    .unwrap()
            );
        }
        drop(native);
        let dirty = machine.collect_cpu_writes().unwrap();
        assert_eq!(
            dirty
                .iter()
                .map(|r| (r.offset(), r.bytes()))
                .collect::<Vec<_>>(),
            [(3, 2)]
        );
        machine.disable_cpu_write_collection();
    }

    #[test]
    fn download_selection_excludes_only_its_destination_and_preserves_cpu_bytes() {
        let region = 2 * 1024 * 1024;
        let machine = Machine::new(2 * region).unwrap();
        let allocation = machine
            .account(2 * region)
            .unwrap()
            .allocate(2 * region)
            .unwrap();
        let range = AllocationRange::new(allocation.pin(), 0, 2 * region).unwrap();
        let space = AddressSpace::new(machine.clone(), AddressWidth::Bits32).unwrap();
        space
            .map(
                GuestVirtualAddress::new(PAGE_SIZE),
                allocation,
                0,
                2 * region,
                MemoryPermissions::READ_WRITE,
            )
            .unwrap();
        machine
            .publish_device_write(std::slice::from_ref(&range), || Ok(()))
            .unwrap();
        machine.enable_cpu_write_collection().unwrap();
        let core = space.register_core(Arc::new(|| {}));
        let native = core.enter_native().unwrap();
        assert!(
            space
                .write_native(&native, GuestVirtualAddress::new(PAGE_SIZE + 3), &[77, 78])
                .unwrap()
        );
        let target = range.slice(0, 16).unwrap();
        let protections = machine.tracking_statistics().protection_calls;
        machine
            .write_download(&target, &[11; 16], || {
                assert!(
                    !space
                        .write_native(&native, GuestVirtualAddress::new(PAGE_SIZE + 7), &[99])
                        .unwrap()
                );
                assert!(
                    space
                        .write_native(&native, GuestVirtualAddress::new(PAGE_SIZE + region), &[42])
                        .unwrap()
                );
                let dirty = machine.collect_cpu_writes()?;
                assert_eq!(
                    dirty
                        .iter()
                        .map(|r| (r.offset(), r.bytes()))
                        .collect::<Vec<_>>(),
                    [(3, 2), (region, 1)]
                );
                Ok(vec![0..3, 5..16])
            })
            .unwrap();
        assert_eq!(machine.tracking_statistics().protection_calls, protections);
        // Destination admission reopens, including while the native invocation lives.
        assert!(
            space
                .write_native(&native, GuestVirtualAddress::new(PAGE_SIZE + 7), &[99])
                .unwrap()
        );
        drop(native);
        let mut actual = [0; 16];
        machine.capture_stream(&[target], &mut actual).unwrap();
        let mut expected = [11; 16];
        expected[3..5].copy_from_slice(&[77, 78]);
        expected[7] = 99;
        assert_eq!(actual, expected);
        machine.disable_cpu_write_collection();
    }

    #[test]
    fn dirty_page_pins_survive_unmapping_and_are_released_on_failure() {
        let (machine, space, range) = fixture();
        let core = space.register_core(Arc::new(|| {}));
        let native = core.enter_native().unwrap();
        assert!(
            space
                .write_native(&native, GuestVirtualAddress::new(PAGE_SIZE), &[1])
                .unwrap()
        );
        drop(native);
        space
            .unmap(GuestVirtualAddress::new(PAGE_SIZE), 2 * PAGE_SIZE)
            .unwrap();
        space
            .unmap(GuestVirtualAddress::new(4 * PAGE_SIZE), 2 * PAGE_SIZE)
            .unwrap();
        drop(range);
        assert_eq!(machine.statistics().retained_bytes, PAGE_SIZE);
        machine.disable_cpu_write_collection();
        assert_eq!(machine.statistics().allocated_bytes, 0);
    }

    #[test]
    fn alternating_pages_stay_compact_and_gather_once_per_dirty_run() {
        let (machine, space, _) = fixture();
        let core = space.register_core(Arc::new(|| {}));
        let native = core.enter_native().unwrap();
        for i in 0..1024 {
            assert!(
                space
                    .write_native(
                        &native,
                        GuestVirtualAddress::new(PAGE_SIZE + (i % 2) * PAGE_SIZE + 4),
                        &[7; 4]
                    )
                    .unwrap()
            );
        }
        let journal = native.cpu_writes().journal.lock().unwrap();
        assert_eq!(journal.pending.pages.len(), 2);
        println!(
            "compact CPU writes: accepted=1024 records={} record_bytes={} retained_pages={}",
            journal.pending.records.len(),
            journal.pending.records.len() * std::mem::size_of::<u64>(),
            journal.pending.pages.len()
        );
        drop(journal);
        drop(native);
        let ranges = machine.collect_cpu_writes().unwrap();
        assert_eq!(ranges.len(), 2);
        assert_eq!(ranges.iter().map(|r| r.bytes()).sum::<u64>(), 8);
        machine.disable_cpu_write_collection();
    }
}
