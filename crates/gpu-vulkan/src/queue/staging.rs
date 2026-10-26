//! Streaming uploads and recyclable staging buffers, independent of command slots.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/vk_staging_buffer_pool.cpp
use super::*;
use std::sync::atomic::AtomicBool;

pub(crate) const STREAM_BYTES: u64 = 256 * 1024 * 1024;
const REGIONS: usize = 16;
pub(crate) const REGION_BYTES: u64 = STREAM_BYTES / REGIONS as u64;
// CPU readback locality is independent of staging admission or submission.
pub(crate) const READBACK_WINDOW_BYTES: u64 = 1024 * 1024;

#[derive(Clone)]
enum Source {
    Stream(Arc<AllocatedBuffer>),
    Pooled(Arc<Entry>),
}
#[derive(Clone)]
pub(crate) struct Slice {
    source: Source,
    pub offset: u64,
    pub size: u64,
}
impl std::fmt::Debug for Slice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Slice")
            .field("offset", &self.offset)
            .field("size", &self.size)
            .finish_non_exhaustive()
    }
}
struct Entry {
    buffer: Arc<AllocatedBuffer>,
    tick: AtomicU64,
    deferred: AtomicBool,
    upload: bool,
}
impl Entry {
    fn reusable(&self, completed: u64) -> bool {
        !self.deferred.load(Ordering::Relaxed)
            && self.tick.load(Ordering::Relaxed) <= completed
            && Arc::strong_count(&self.buffer) == 1
    }
}
#[derive(Default)]
struct Level {
    entries: Vec<Arc<Entry>>,
    next: usize,
    delete: usize,
}
pub(crate) struct StagingPool {
    owner: Arc<DeviceOwner>,
    stream: Option<Arc<AllocatedBuffer>>,
    cursor: u64,
    region: Option<usize>,
    ticks: [u64; REGIONS],
    deferred: u16,
    uploads: [Level; 64],
    downloads: [Level; 64],
    delete_level: usize,
}
impl StagingPool {
    pub fn new(owner: Arc<DeviceOwner>) -> Self {
        Self {
            owner,
            stream: None,
            cursor: 0,
            region: None,
            ticks: [0; REGIONS],
            deferred: 0,
            uploads: std::array::from_fn(|_| Level::default()),
            downloads: std::array::from_fn(|_| Level::default()),
            delete_level: 0,
        }
    }
    fn active(&self, begin: usize, end: usize, completed: u64) -> bool {
        (begin..end).any(|i| self.ticks[i] > completed || self.deferred & (1 << i) != 0)
    }
    pub fn reserve(
        &mut self,
        size: u64,
        alignment: u64,
        upload: bool,
        tick: u64,
        completed: u64,
    ) -> Result<Slice, VulkanError> {
        if size == 0 || !alignment.is_power_of_two() {
            return Err(VulkanError::InvalidResource(
                "invalid staging size/alignment",
            ));
        }
        if upload && size <= REGION_BYTES {
            if self.stream.is_none() {
                self.stream = Some(Arc::new(self.allocate(STREAM_BYTES, true)?));
            }
            let alignment = alignment
                .max(self.stream.as_ref().unwrap().host_alignment())
                .max(256);
            let start = align_up(self.cursor, alignment)
                .ok_or(VulkanError::InvalidResource("staging offset overflow"))?;
            let end = start
                .checked_add(size)
                .ok_or(VulkanError::InvalidResource("staging size overflow"))?;
            let wrapped = end > STREAM_BYTES;
            let offset = if wrapped { 0 } else { start };
            let first = (offset / REGION_BYTES) as usize;
            let last = ((offset + size - 1) / REGION_BYTES) as usize;
            // Only the unused tail of the current region can be appended while
            // its previous slices are active. Crossing/wrapping checks every
            // newly entered region. Failed reservations do not move the cursor.
            let begin = if !wrapped && self.region == Some(first) {
                first + 1
            } else {
                first
            };
            if !self.active(begin, last + 1, completed) {
                self.ticks[first..=last].fill(tick);
                self.cursor = align_up(offset + size, alignment)
                    .ok_or(VulkanError::InvalidResource("staging offset overflow"))?;
                self.region = Some(last);
                return Ok(Slice {
                    source: Source::Stream(Arc::clone(self.stream.as_ref().unwrap())),
                    offset,
                    size,
                });
            }
        }

        let capacity = size
            .max(alignment)
            .max(self.owner.non_coherent_atom_size)
            .checked_next_power_of_two()
            .ok_or(VulkanError::InvalidResource("staging capacity overflow"))?;
        let level = capacity.ilog2() as usize;
        let cache = if upload {
            &mut self.uploads[level]
        } else {
            &mut self.downloads[level]
        };
        let index = (cache.next..cache.entries.len())
            .chain(0..cache.next)
            .find(|&i| {
                Arc::strong_count(&cache.entries[i]) == 1 && cache.entries[i].reusable(completed)
            });
        let entry = if let Some(index) = index {
            let entry = &cache.entries[index];
            entry.tick.store(tick, Ordering::Relaxed);
            entry.deferred.store(!upload, Ordering::Relaxed);
            cache.next = (index + 1) % cache.entries.len();
            Arc::clone(entry)
        } else {
            let entry = Arc::new(Entry {
                buffer: Arc::new(self.allocate(capacity, upload)?),
                tick: AtomicU64::new(tick),
                deferred: AtomicBool::new(!upload),
                upload,
            });
            let cache = if upload {
                &mut self.uploads[level]
            } else {
                &mut self.downloads[level]
            };
            cache.entries.push(Arc::clone(&entry));
            entry
        };
        Ok(Slice {
            source: Source::Pooled(entry),
            offset: 0,
            size,
        })
    }
    fn allocate(&self, size: u64, upload: bool) -> Result<AllocatedBuffer, VulkanError> {
        let (usage, location, label) = if upload {
            (
                vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::VERTEX_BUFFER
                    | vk::BufferUsageFlags::INDEX_BUFFER
                    | vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::UNIFORM_BUFFER
                    | vk::BufferUsageFlags::UNIFORM_TEXEL_BUFFER
                    | vk::BufferUsageFlags::STORAGE_TEXEL_BUFFER,
                MemoryLocation::CpuToGpu,
                "nixe upload staging",
            )
        } else {
            (
                vk::BufferUsageFlags::TRANSFER_DST,
                MemoryLocation::GpuToCpu,
                "nixe readback staging",
            )
        };
        allocate_buffer(
            &self.owner,
            size,
            usage,
            location,
            AllocationMode::Managed,
            label,
        )
    }
    pub fn mark_used(&mut self, slice: &Slice, tick: u64) {
        match &slice.source {
            Source::Stream(_) => {
                let regions = self.regions(slice);
                for i in 0..REGIONS {
                    if regions & (1 << i) != 0 {
                        self.ticks[i] = self.ticks[i].max(tick);
                    }
                }
            }
            Source::Pooled(entry) => {
                entry.tick.fetch_max(tick, Ordering::Relaxed);
            }
        }
    }
    pub fn buffer<'a>(&self, slice: &'a Slice) -> &'a Arc<AllocatedBuffer> {
        match &slice.source {
            Source::Stream(buffer) => buffer,
            Source::Pooled(entry) => &entry.buffer,
        }
    }
    pub fn is_stream(&self, slice: &Slice) -> bool {
        matches!(slice.source, Source::Stream(_))
    }
    pub fn regions(&self, slice: &Slice) -> u16 {
        if !self.is_stream(slice) {
            return 0;
        }
        let first = slice.offset / REGION_BYTES;
        let last = (slice.offset + slice.size - 1) / REGION_BYTES;
        (((1u32 << (last - first + 1)) - 1) << first) as u16
    }
    pub fn defer_regions(&mut self, regions: u16) {
        self.deferred |= regions;
    }
    pub fn release_regions(&mut self, tick: u64) {
        for i in 0..REGIONS {
            if self.deferred & (1 << i) != 0 {
                self.ticks[i] = self.ticks[i].max(tick);
            }
        }
        self.deferred = 0;
    }
    pub fn valid_readback(&self, slice: &Slice, tick: u64) -> bool {
        match &slice.source {
            Source::Pooled(entry) => {
                !entry.upload
                    && entry.deferred.load(Ordering::Relaxed)
                    && entry.tick.load(Ordering::Relaxed) == tick
            }
            _ => false,
        }
    }
    pub fn retained_readbacks(&self) -> usize {
        self.downloads
            .iter()
            .flat_map(|level| &level.entries)
            .filter(|entry| entry.deferred.load(Ordering::Relaxed))
            .count()
    }
    pub fn release_readback(&mut self, slice: &Slice) {
        let Source::Pooled(entry) = &slice.source else {
            unreachable!("readback owns download staging")
        };
        entry.deferred.store(false, Ordering::Relaxed);
    }
    pub fn tick_frame(&mut self, completed: u64) {
        self.delete_level = (self.delete_level + 1) % 64;
        for cache in [
            &mut self.uploads[self.delete_level],
            &mut self.downloads[self.delete_level],
        ] {
            let end = (cache.delete + 16).min(cache.entries.len());
            let mut write = cache.delete;
            for read in cache.delete..end {
                let entry = &cache.entries[read];
                if Arc::strong_count(entry) != 1 || !entry.reusable(completed) {
                    cache.entries.swap(write, read);
                    write += 1;
                }
            }
            cache.entries.drain(write..end);
            cache.delete = if end < cache.entries.len() { end } else { 0 };
            cache.next = cache.next.min(cache.entries.len());
        }
    }
    pub fn clear(&mut self) {
        self.stream = None;
        for level in self.uploads.iter_mut().chain(&mut self.downloads) {
            for entry in &level.entries {
                entry.deferred.store(false, Ordering::Relaxed);
            }
            level.entries.clear();
            level.next = 0;
            level.delete = 0;
        }
        self.cursor = 0;
        self.region = None;
        self.ticks.fill(0);
        self.deferred = 0;
    }
    pub fn stream_capacity(&self) -> u64 {
        self.stream.as_ref().map_or(0, |b| b.size())
    }
    pub fn pooled_capacity(&self) -> u64 {
        self.uploads
            .iter()
            .chain(&self.downloads)
            .flat_map(|level| &level.entries)
            .map(|e| e.buffer.size())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn compaction_preserves_deferred_downloads_and_discards_historical_holes() {
        let Some(device) = crate::test_device::device_for_test(crate::ApiPreference::Auto) else {
            return;
        };
        let validation = device.validation_monitor();
        let mut pool = StagingPool::new(Arc::clone(&device.owner));
        let mut live = None;
        for i in 0..256 {
            let slice = pool.reserve(256, 4, false, 1, 0).unwrap();
            if i == 127 {
                live = Some(slice);
            } else {
                pool.release_readback(&slice);
            }
        }
        let live = live.unwrap();
        let native = Arc::clone(pool.buffer(&live));
        let level = 256u64.max(device.owner.non_coherent_atom_size).ilog2() as usize;
        for _ in 0..64 * 40 {
            pool.tick_frame(1);
        }
        assert_eq!(pool.downloads[level].entries.len(), 1);
        assert_eq!(pool.retained_readbacks(), 1);
        assert!(pool.valid_readback(&live, 1));
        assert!(Arc::ptr_eq(pool.buffer(&live), &native));
        pool.release_readback(&live);
        drop(live);
        drop(native);
        for _ in 0..64 {
            pool.tick_frame(1);
        }
        assert!(pool.downloads[level].entries.is_empty());
        let next = pool.reserve(256, 4, false, 2, 1).unwrap();
        assert_eq!(pool.downloads[level].entries.len(), 1);
        pool.release_readback(&next);
        drop(next);
        drop(pool);
        drop(device);
        validation.check().unwrap();
    }
}
