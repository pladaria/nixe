//! CPU overwrites of device-dependent RAM are collected before graphics effects.
//! The journal contains dirty bytes, not device ownership or native versions.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/gpu_dirty_memory_manager.h
//! Dirty granules are individual bytes here to preserve mixed GPU/CPU coverage.
use crate::{AllocationRange, ArenaOffset, Machine, MemoryError, PAGE_SIZE};
use std::sync::{
    Arc, Mutex, RwLock,
    atomic::{AtomicBool, Ordering},
};

// Bounded retained RAM/metadata while the command owner is compiling or busy.
const MAX_PAGES: usize = 256;
struct DirtyPage {
    physical: ArenaOffset,
    range: AllocationRange,
    bytes: [u64; PAGE_SIZE as usize / 64],
}
#[derive(Default)]
pub(crate) struct CpuWrites {
    enabled: AtomicBool,
    cores: RwLock<Vec<Arc<CpuWriteCollector>>>,
}
/// A core owns its collector; another core never contends on its store path.
#[derive(Default)]
pub(crate) struct CpuWriteCollector {
    pages: Mutex<CollectorPages>,
    dirty: AtomicBool,
}
#[derive(Default)]
struct CollectorPages {
    pending: Vec<DirtyPage>,
    spare: Vec<DirtyPage>,
}
impl Machine {
    pub fn enable_cpu_write_collection(&self) -> Result<(), MemoryError> {
        self.cpu_writes()
            .enabled
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| MemoryError::Invalid("CPU writes already have a graphics consumer"))?;
        Ok(())
    }
    /// Stop collection before the device owner drops, including failure paths.
    pub fn disable_cpu_write_collection(&self) {
        self.cpu_writes().enabled.store(false, Ordering::Release);
        for core in self.cpu_writes().cores.read().unwrap().iter() {
            core.pages.lock().unwrap().pending.clear();
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
                let mut pages = core.pages.lock().unwrap();
                let spare = std::mem::take(&mut pages.spare);
                let pending = std::mem::replace(&mut pages.pending, spare);
                core.dirty.store(false, Ordering::Release);
                pending
            };
            append_ranges(&pending, &mut result)?;
            pending.clear();
            core.pages.lock().unwrap().spare = pending;
        }
        Ok(result)
    }
}
impl CpuWrites {
    pub(crate) fn register(&self) -> Arc<CpuWriteCollector> {
        let mut cores = self.cores.write().unwrap();
        // Reuse drained slots once their native context has gone away.
        if let Some(core) = cores.iter().find(|core| {
            Arc::strong_count(core) == 1 && core.pages.lock().unwrap().pending.is_empty()
        }) {
            return core.clone();
        }
        let core = Arc::new(CpuWriteCollector::default());
        cores.push(core.clone());
        core
    }
}
impl CpuWriteCollector {
    /// The caller retains native admission and has checked every target page
    /// is writable and unobserved. `capture` pins each complete physical page;
    /// `write` executes the original store once while the owner is excluded.
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
        let Ok(mut collected) = self.pages.try_lock() else {
            return Ok(false);
        };
        let pages = &mut collected.pending;
        if !machine.cpu_writes().enabled.load(Ordering::Acquire) {
            return Ok(false);
        }
        if size > 16 {
            return Ok(false);
        }
        let end = address + size as u64;
        let count = (end.div_ceil(PAGE_SIZE) - address / PAGE_SIZE) as usize;
        let mut last = pages.last().map(|page| page.physical);
        let mut missing = 0;
        for start in (address & !(PAGE_SIZE - 1)..end).step_by(PAGE_SIZE as usize) {
            let page = physical(start);
            missing += usize::from(last != Some(page));
            last = Some(page);
        }
        if missing > MAX_PAGES - pages.len() {
            return Ok(false);
        }
        // Validate/capture the entire write before executing its first byte.
        // Native callbacks are at most 16 bytes, hence at most two pages.
        let mut targets = [(0, 0, 0); 2];
        for (target, start) in targets
            .iter_mut()
            .zip((address & !(PAGE_SIZE - 1)..end).step_by(PAGE_SIZE as usize))
        {
            let physical = physical(start);
            let index = if pages.last().is_some_and(|page| page.physical == physical) {
                pages.len() - 1
            } else {
                let range = capture(start)?;
                pages.push(DirtyPage {
                    physical,
                    range,
                    bytes: [0; PAGE_SIZE as usize / 64],
                });
                pages.len() - 1
            };
            *target = (
                index,
                address.max(start) - start,
                end.min(start + PAGE_SIZE) - start,
            );
        }
        write();
        for &(index, begin, end) in &targets[..count] {
            for word in begin / 64..end.div_ceil(64) {
                let low = begin.saturating_sub(word * 64);
                let high = (end - word * 64).min(64);
                pages[index].bytes[word as usize] |= (u64::MAX << low) & (u64::MAX >> (64 - high));
            }
        }
        self.dirty.store(true, Ordering::Release);
        Ok(true)
    }
}

fn append_ranges(
    pages: &[DirtyPage],
    result: &mut Vec<AllocationRange>,
) -> Result<(), MemoryError> {
    for page in pages {
        let mut run: Option<(u64, u64)> = None;
        let mut emit = |(begin, end): (u64, u64)| {
            result.push(AllocationRange::new(
                page.range.allocation().clone(),
                page.range.offset() + begin,
                end - begin,
            )?);
            Ok::<_, MemoryError>(())
        };
        for (index, &word) in page.bytes.iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                let low = bits.trailing_zeros();
                let high = low + (bits >> low).trailing_ones();
                let begin = index as u64 * 64 + u64::from(low);
                let end = index as u64 * 64 + u64::from(high);
                match run {
                    Some((start, previous_end)) if previous_end == begin => {
                        run = Some((start, end))
                    }
                    previous => {
                        if let Some(previous) = previous {
                            emit(previous)?;
                        }
                        run = Some((begin, end));
                    }
                }
                bits = if high == 64 {
                    0
                } else {
                    bits & (u64::MAX << high)
                };
            }
        }
        if let Some(run) = run {
            emit(run)?;
        }
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
            [(3, 6), (63, 3), (4094, 2), (4096, 2), (4, 2)]
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
            let _held = native.cpu_writes().pages.lock().unwrap();
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
    fn bounded_collection_reuses_current_page_and_defers_new_cross_page_stores_intact() {
        let capacity = (MAX_PAGES as u64 + 1) * PAGE_SIZE;
        let machine = Machine::new(capacity).unwrap();
        let allocation = machine
            .account(capacity)
            .unwrap()
            .allocate(capacity)
            .unwrap();
        let range = AllocationRange::new(allocation.pin(), 0, capacity).unwrap();
        let space = AddressSpace::new(machine.clone(), AddressWidth::Bits32).unwrap();
        space
            .map(
                GuestVirtualAddress::new(PAGE_SIZE),
                allocation,
                0,
                capacity,
                MemoryPermissions::READ_WRITE,
            )
            .unwrap();
        machine
            .publish_device_write(std::slice::from_ref(&range), || Ok(()))
            .unwrap();
        machine.enable_cpu_write_collection().unwrap();
        let core = space.register_core(Arc::new(|| {}));
        let native = core.enter_native().unwrap();
        for page in 0..MAX_PAGES {
            assert!(
                space
                    .write_native(
                        &native,
                        GuestVirtualAddress::new((page as u64 + 1) * PAGE_SIZE),
                        &[17]
                    )
                    .unwrap()
            );
        }
        assert!(
            space
                .write_native(
                    &native,
                    GuestVirtualAddress::new(MAX_PAGES as u64 * PAGE_SIZE),
                    &[18]
                )
                .unwrap()
        );
        assert!(
            !space
                .write_native(&native, GuestVirtualAddress::new(capacity - 8), &[99; 16])
                .unwrap()
        );
        drop(native);
        let mut unchanged = [0; 16];
        machine
            .capture_stream(
                &[range.slice(capacity - PAGE_SIZE - 8, 16).unwrap()],
                &mut unchanged,
            )
            .unwrap();
        assert_eq!(unchanged, [0; 16]);
        assert_eq!(machine.collect_cpu_writes().unwrap().len(), MAX_PAGES);
        machine.disable_cpu_write_collection();
    }
}
