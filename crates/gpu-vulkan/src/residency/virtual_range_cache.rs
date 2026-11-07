//! BufferCache's bounded GPU virtual segment cache and deferred invalidations.
//! The shared invalidation member is reachable by CPU mapping callbacks; only
//! the graphics owner consumes entries, under its existing buffer-cache guard.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/buffer_cache/virtual_range_cache.h
use super::*;
use nixe_gpu::GpuVirtualAddress;
use nixe_gpu::native::GuestBufferRange;
use nixe_gpu_maxwell::{MaxwellAddressSpaceId, MaxwellGpuMemory};
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
    address_space: MaxwellAddressSpaceId,
    address: GpuVirtualAddress,
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
                unmap.address_space == entry.address_space
                    && entry.address.get() < unmap.address + unmap.bytes
                    && unmap.address < entry.address.get() + entry.bytes
            })
        });
    }
    pub(super) fn query(
        &mut self,
        memory: &MaxwellGpuMemory,
        address: GpuVirtualAddress,
        bytes: u64,
    ) -> Result<&[GuestBufferRange], VulkanError> {
        if self.invalidations.has_deferred.load(Ordering::Acquire) {
            self.apply_deferred();
        }
        if self.entries.len() > MAX_ENTRIES {
            self.entries.clear();
        }
        let address_space = memory.id();
        let key = (address_space.get() << 48) ^ address.get();
        use std::collections::hash_map::Entry as MapEntry;
        let entry = match self.entries.entry(key) {
            MapEntry::Occupied(occupied) => {
                let entry = occupied.into_mut();
                if entry.address_space != address_space
                    || entry.address != address
                    || entry.bytes != bytes
                {
                    *entry = Self::resolve(memory, address, bytes)?;
                }
                entry
            }
            MapEntry::Vacant(vacant) => vacant.insert(Self::resolve(memory, address, bytes)?),
        };
        Ok(&entry.segments)
    }
    fn resolve(
        memory: &MaxwellGpuMemory,
        address: GpuVirtualAddress,
        bytes: u64,
    ) -> Result<Entry, VulkanError> {
        let fail = |error: nixe_gpu_maxwell::MaxwellGpuAccessError| {
            VulkanError::Compilation(error.to_string().into())
        };
        let mut entry = Entry {
            address_space: memory.id(),
            address,
            bytes,
            segments: smallvec::SmallVec::new(),
        };
        let mut expected = address.get();
        for (gpu, size) in memory.get_submapped_range(address, bytes).map_err(fail)? {
            if gpu.get() != expected || size == 0 || size > u64::from(u32::MAX) {
                break;
            }
            let device = match memory.device_address(gpu) {
                Ok(device) if device != 0 => device,
                Ok(_) | Err(nixe_gpu_maxwell::MaxwellGpuAccessError::UnmappedAddress { .. }) => {
                    break;
                }
                Err(error) => return Err(fail(error)),
            };
            entry.segments.push(GuestBufferRange {
                address: device,
                size,
            });
            expected += size;
        }
        if expected != address.get() + bytes {
            entry.segments.clear();
        }
        Ok(entry)
    }
}
