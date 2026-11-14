//! BufferCache's bounded GPU virtual segment cache and deferred invalidations.
//! The shared invalidation member is reachable by CPU mapping callbacks; only
//! the graphics owner consumes entries, under its existing buffer-cache guard.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/buffer_cache/virtual_range_cache.h
use super::*;
use nixe_gpu::native::GuestBufferRange;
use nixe_gpu::native::ImageAddressSpace;
use nixe_gpu_maxwell::MaxwellAddressSpaceId;
use std::sync::{
    Mutex,
    atomic::{AtomicBool, Ordering},
};
const MAX_ENTRIES: usize = 8192;
const MAX_DEFERRED: usize = 4096;
#[derive(Clone, Copy)]
struct Unmap {
    address_space: MaxwellAddressSpaceId,
    address: u64,
    bytes: u64,
}
#[derive(Default)]
struct Pending {
    ranges: Vec<Unmap>,
    overflow: bool,
}
#[derive(Default)]
pub(super) struct Invalidations {
    pending: Mutex<Pending>,
    has_deferred: AtomicBool,
}
impl Invalidations {
    pub(super) fn unmap(&self, address_space: MaxwellAddressSpaceId, address: u64, bytes: u64) {
        if bytes == 0 {
            return;
        }
        let mut pending = self.pending.lock().unwrap();
        if let Some(last) = pending.ranges.last_mut()
            && last.address_space == address_space
            && last.address + last.bytes == address
        {
            last.bytes += bytes;
        } else if pending.ranges.len() >= MAX_DEFERRED {
            pending.ranges.clear();
            pending.overflow = true;
        } else {
            pending.ranges.push(Unmap {
                address_space,
                address,
                bytes,
            });
        }
        self.has_deferred.store(true, Ordering::Release);
    }
}
struct Entry {
    address_space: u64,
    generation: u64,
    address: u64,
    bytes: u64,
    segments: smallvec::SmallVec<[GuestBufferRange; 8]>,
}
#[derive(Default)]
pub(super) struct VirtualRangeCache {
    entries: HashMap<u64, Entry>,
    pub(super) invalidations: Arc<Invalidations>,
}
impl VirtualRangeCache {
    fn apply_deferred(&mut self) {
        let pending = {
            let mut pending = self.invalidations.pending.lock().unwrap();
            self.invalidations
                .has_deferred
                .store(false, Ordering::Release);
            std::mem::take(&mut *pending)
        };
        if pending.overflow {
            self.entries.clear();
            return;
        }
        if pending.ranges.is_empty() || self.entries.is_empty() {
            return;
        }
        self.entries.retain(|_, entry| {
            !pending.ranges.iter().any(|unmap| {
                unmap.address_space.get() == entry.address_space
                    && entry.address < unmap.address + unmap.bytes
                    && unmap.address < entry.address + entry.bytes
            })
        });
    }
    pub(super) fn query(
        &mut self,
        memory: &impl ImageAddressSpace,
        address: u64,
        bytes: u64,
        generation: u64,
    ) -> Result<&[GuestBufferRange], VulkanError> {
        if self.invalidations.has_deferred.load(Ordering::Acquire) {
            self.apply_deferred();
        }
        if self.entries.len() > MAX_ENTRIES {
            self.entries.clear();
        }
        let address_space = memory.image_space_id();
        let key = (address_space << 48) ^ address;
        use std::collections::hash_map::Entry as MapEntry;
        let entry = match self.entries.entry(key) {
            MapEntry::Occupied(occupied) => {
                let entry = occupied.into_mut();
                if entry.address_space != address_space
                    || entry.address != address
                    || entry.bytes != bytes
                    || entry.generation != generation
                {
                    *entry = Self::resolve(memory, address, bytes, generation)?;
                }
                entry
            }
            MapEntry::Vacant(vacant) => {
                vacant.insert(Self::resolve(memory, address, bytes, generation)?)
            }
        };
        Ok(&entry.segments)
    }
    fn resolve(
        memory: &impl ImageAddressSpace,
        address: u64,
        bytes: u64,
        generation: u64,
    ) -> Result<Entry, VulkanError> {
        let mut entry = Entry {
            address_space: memory.image_space_id(),
            generation,
            address,
            bytes,
            segments: smallvec::SmallVec::new(),
        };
        let mut expected = address;
        for (gpu, size) in memory.image_mapped_ranges(address, bytes)? {
            if gpu != expected || size == 0 || size > u64::from(u32::MAX) {
                break;
            }
            let Some(device) = memory
                .image_device_address(gpu)?
                .filter(|&device| device != 0)
            else {
                break;
            };
            entry.segments.push(GuestBufferRange {
                address: device,
                size,
            });
            expected += size;
        }
        if expected != address + bytes {
            entry.segments.clear();
        }
        Ok(entry)
    }
}
