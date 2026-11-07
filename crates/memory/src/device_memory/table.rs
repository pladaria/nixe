//! Canonical device PTEs in a zero-backed virtual reservation. Mapping metadata
//! and heap pin counts are cold; readers index these entries directly.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/device_memory_manager.h
use super::*;
use crate::SparseArray;
const CPU_BITS: u32 = 39;
const CPU_MASK: u64 = (1 << CPU_BITS) - 1;

#[derive(Debug)]
#[repr(C)]
pub(super) struct TrackedEntry {
    cpu_backing: AtomicU64,
    // Physical page + 1 and its CPU continuity are one publication. Zero is
    // unmapped. Atomic fields permit shared map handles without racing Rust data.
    mapping: AtomicU64,
}
#[derive(Clone, Copy)]
pub(super) struct TrackedPage {
    pub(super) physical: u32,
    pub(super) continuity: u32,
}
impl TrackedEntry {
    pub(super) fn page(&self) -> TrackedPage {
        let value = self.mapping.load(Ordering::Acquire);
        TrackedPage {
            physical: value as u32,
            continuity: (value >> 32) as u32,
        }
    }
    pub(super) fn set_physical(&self, physical: u32) {
        self.mapping
            .fetch_update(Ordering::Release, Ordering::Relaxed, |previous| {
                Some((previous & !u64::from(u32::MAX)) | u64::from(physical))
            })
            .unwrap();
    }
    fn set_continuity(&self, pages: u32) {
        self.mapping
            .fetch_update(Ordering::Release, Ordering::Relaxed, |previous| {
                Some((previous & u64::from(u32::MAX)) | (u64::from(pages) << 32))
            })
            .unwrap();
    }
    pub(super) fn cpu_backing(&self) -> (u32, u64) {
        let value = self.cpu_backing.load(Ordering::Acquire);
        ((value >> CPU_BITS) as u32, value & CPU_MASK)
    }
    pub(super) fn set_cpu_backing(&self, process: u32, address: u64) {
        assert!(
            address <= CPU_MASK && u64::from(process) <= (u64::MAX >> CPU_BITS),
            "device CPU backing exceeds the hardware address/ASID fields"
        );
        self.cpu_backing.store(
            address | (u64::from(process) << CPU_BITS),
            Ordering::Release,
        );
    }
}
#[derive(Debug)]
pub(super) struct TrackedTable {
    entries: SparseArray<TrackedEntry>,
}
impl TrackedTable {
    pub(super) fn new(machine: &Machine, address_bits: u32) -> Self {
        // SAFETY: TrackedEntry is two zero-valid atomic integers without Drop.
        Self {
            entries: unsafe { SparseArray::new(machine, 1usize << (address_bits - 12)) },
        }
    }
    #[inline]
    pub(super) fn entry(&self, address: u64) -> &TrackedEntry {
        &self.entries[(address >> 12) as usize]
    }
    pub(super) fn track_continuity(&self, address: u64, physical: &PhysicalMapping) {
        let spans = physical.spans();
        let mut index = spans.len();
        let mut next = None;
        let mut continuity = 0;
        // TrackContinuityImpl walks backwards over every CPU page, including
        // holes. Reusing a device range must clear continuity left in a hole by
        // its previous CPU backing; Unmap itself deliberately retains it.
        for page in (0..physical.bytes() / PAGE_SIZE).rev() {
            let offset = page * PAGE_SIZE;
            while index != 0 && spans[index - 1].logical > offset {
                index -= 1;
            }
            let current = index
                .checked_sub(1)
                .map(|index| &spans[index])
                .filter(|span| offset < span.logical + span.bytes)
                .map(|span| span.start.get() + offset - span.logical);
            continuity = if current
                .zip(next)
                .is_some_and(|(current, next)| current + PAGE_SIZE == next)
            {
                continuity + 1
            } else {
                0
            };
            self.entry(address + offset).set_continuity(continuity);
            next = current;
        }
    }
}
