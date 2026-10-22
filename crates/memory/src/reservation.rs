//! Physical reservation identity shared by exclusive and atomic operations.
//! Ordinary stores stay direct. Exclusive/atomic writers share reservation
//! versions, and exclusive commit compares its captured value atomically.
//! https://developer.arm.com/documentation/ddi0487/latest/
use crate::{AllocationId, ArenaOffset, MachineId, PAGE_SIZE};
use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

pub(crate) const GRANULE_BYTES: usize = 64;
/// A retained device resource can invalidate exclusive observations without
/// walking all its physical pages or retaining guest RAM. Device invalidation
/// is conservative within an allocation; exclusive/atomic stores remain granule
/// precise. Arm permits a store-exclusive to fail without a conflicting store.
#[derive(Clone, Default)]
pub struct DeviceWriteEpoch(Arc<AtomicU64>);
impl DeviceWriteEpoch {
    pub fn mark_device_modified(&self) {
        if self.0.load(Ordering::Acquire) != 0 {
            self.0
                .fetch_update(Ordering::Release, Ordering::Relaxed, |value| {
                    value.checked_add(1)
                })
                .expect("device reservation epoch exhausted");
        }
    }
    pub(crate) fn activate(&self) {
        let _ = self
            .0
            .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire);
    }
}
/// Native emitter ABI: canonical page pointer, lifetime identity, device epoch
/// pointer, then one
/// aligned version/lock word per physical reservation granule. These are host
/// metadata atomics, never references into concurrently writable guest RAM.
#[repr(C, align(64))]
pub(crate) struct PhysicalMonitor {
    canonical_base: usize,
    identity: u64,
    device_epoch: *const AtomicU64,
    words: [AtomicU64; PAGE_SIZE as usize / GRANULE_BYTES],
    _epoch: DeviceWriteEpoch,
}
// The pointer references owned atomic metadata, never guest bytes. _epoch keeps
// it alive until this monitor is retired after the last physical allocation pin.
unsafe impl Send for PhysicalMonitor {}
unsafe impl Sync for PhysicalMonitor {}
const _: () = {
    assert!(std::mem::offset_of!(PhysicalMonitor, identity) == 8);
    assert!(std::mem::offset_of!(PhysicalMonitor, device_epoch) == 16);
    assert!(std::mem::offset_of!(PhysicalMonitor, words) == 24);
};
pub(crate) struct LockedGranule<'a> {
    word: &'a AtomicU64,
    version: u64,
    written: bool,
}
impl PhysicalMonitor {
    fn new(canonical_base: usize, epoch: DeviceWriteEpoch) -> Self {
        static NEXT_MONITOR: AtomicU64 = AtomicU64::new(1);
        let identity = NEXT_MONITOR
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .expect("physical reservation lifetime identity exhausted");
        Self {
            canonical_base,
            identity,
            device_epoch: Arc::as_ptr(&epoch.0),
            words: std::array::from_fn(|_| AtomicU64::new(2)),
            _epoch: epoch,
        }
    }
    pub(crate) fn canonical_base(&self) -> usize {
        self.canonical_base
    }
    pub fn identity(&self) -> u64 {
        self.identity
    }
    pub fn device_version(&self) -> u64 {
        self._epoch.0.load(Ordering::Acquire)
    }
    pub fn lock(&self, page_offset: usize) -> LockedGranule<'_> {
        let word = &self.words[page_offset / GRANULE_BYTES];
        loop {
            let before = word.fetch_or(1, Ordering::Acquire);
            if before & 1 == 0 {
                return LockedGranule {
                    word,
                    version: before,
                    written: false,
                };
            }
            while word.load(Ordering::Relaxed) & 1 != 0 {
                std::hint::spin_loop();
            }
        }
    }
}
impl LockedGranule<'_> {
    pub fn version(&self) -> u64 {
        self.version
    }
    pub fn written(&mut self) {
        self.written = true;
    }
}
impl Drop for LockedGranule<'_> {
    fn drop(&mut self) {
        let next = self
            .version
            .checked_add(if self.written { 2 } else { 0 })
            .expect("physical reservation version exhausted");
        self.word.store(next, Ordering::Release);
    }
}
#[derive(Default)]
pub(crate) struct Monitors(Mutex<HashMap<u64, Box<PhysicalMonitor>>>);
impl Monitors {
    /// Caller has quiesced virtual and physical accesses under the machine
    /// mapping/protection writer; insertion never moves an existing monitor.
    pub fn ensure(&self, page: u64, canonical: usize, epoch: &DeviceWriteEpoch) {
        self.0
            .lock()
            .unwrap()
            .entry(page)
            .or_insert_with(|| Box::new(PhysicalMonitor::new(canonical, epoch.clone())));
    }
    /// # Safety
    /// Caller retains the physical allocation, directly or through an admitted
    /// virtual mapping. Final pin release is the only monitor removal point.
    pub unsafe fn get(&self, page: u64) -> Option<&PhysicalMonitor> {
        let pointer = self
            .0
            .lock()
            .unwrap()
            .get(&page)
            .map(|page| std::ptr::from_ref(&**page))?;
        Some(unsafe { &*pointer })
    }
    pub fn remove(&self, page: u64) {
        self.0.lock().unwrap().remove(&page);
    }
    pub fn bytes(&self) -> usize {
        self.0.lock().unwrap().len() * std::mem::size_of::<PhysicalMonitor>()
    }
}
/// A reservation retains identity and version, not guest RAM. Unmapping/freeing
/// its old allocation cannot keep it alive or resurrect it after recycling.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Reservation {
    pub(crate) machine: MachineId,
    pub(crate) allocation: AllocationId,
    pub(crate) physical: ArenaOffset,
    pub(crate) bytes: u8,
    pub(crate) monitor: u64,
    pub(crate) version: u64,
    pub(crate) device_version: u64,
    pub(crate) expected: [u64; 2],
}
/// Owned native store-exclusive proof. Its byte address and width must survive
/// the cold callback along with identity/version and the observed value; two offsets in one physical
/// page can otherwise have equal versions without naming the same reservation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct NativeReservation {
    pub identity: u64,
    pub version: u64,
    pub device_version: u64,
    pub canonical_address: u64,
    pub bytes: u64,
    pub expected: [u64; 2],
}
const _: () = assert!(std::mem::size_of::<NativeReservation>() == 56);
