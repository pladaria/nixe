//! Streaming uploads and recyclable staging buffers, independent of command slots.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/vk_staging_buffer_pool.cpp
use super::*;
use std::sync::atomic::AtomicBool;

pub(crate) const STREAM_BYTES: u64 = 256 * 1024 * 1024;
const REGIONS: usize = 16;
pub(crate) const REGION_BYTES: u64 = STREAM_BYTES / REGIONS as u64;
// CPU readback locality is independent of staging admission or submission.

/// Actual staging roles, as Eden MemoryUsage. Device-local scratch is never
/// mapped for CPU publication and retires by the same native timeline.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Usage {
    Upload,
    Download,
    DeviceLocal,
}

/// Non-owning pool reference, as Eden's StagingBufferRef. The pool keeps
/// allocations alive by submission tick; only a cross-thread readback takes
/// ownership of the deferred allocation itself.
#[derive(Clone, Copy, Debug)]
enum Source {
    Stream,
    Pooled {
        usage: Usage,
        level: u32,
        index: u64,
    },
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct Slice {
    source: Source,
    // Exposed mapped address, not an owning Rust reference. The pool's tick or
    // deferred readback guard retains it; only the unsafe consumption APIs
    // below can dereference the accepted span.
    mapped_address: usize,
    pub buffer: vk::Buffer,
    pub offset: u64,
    pub size: u64,
}
pub(super) struct ReadbackAllocation(Arc<Entry>);
impl ReadbackAllocation {
    pub(super) fn valid(&self, tick: u64) -> bool {
        self.0.usage == Usage::Download
            && self.0.deferred.load(Ordering::Acquire)
            && self.0.tick.load(Ordering::Relaxed) == tick
    }
    pub(super) fn buffer(&self) -> &AllocatedBuffer {
        &self.0.buffer
    }
    /// Caller proved native completion and owns this deferred reservation.
    pub(super) unsafe fn completed_bytes(&self, slice: &Slice) -> Result<&[u8], VulkanError> {
        if slice.mapped_address == 0 {
            return unsafe { self.buffer().completed_bytes(slice.offset, slice.size) };
        }
        // SAFETY: Request established these bounds; the readback guard retains
        // the mapped allocation and excludes recycling throughout consumption.
        Ok(unsafe {
            std::slice::from_raw_parts(
                std::ptr::with_exposed_provenance(slice.mapped_address),
                slice.size as usize,
            )
        })
    }
}
impl Drop for ReadbackAllocation {
    fn drop(&mut self) {
        self.0.deferred.store(false, Ordering::Release);
    }
}
struct Entry {
    index: u64,
    buffer: Arc<AllocatedBuffer>,
    mapped_address: usize,
    tick: AtomicU64,
    deferred: AtomicBool,
    usage: Usage,
}
impl Entry {
    fn reusable(&self, completed: u64) -> bool {
        !self.deferred.load(Ordering::Acquire)
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
    stream_pointer: usize,
    cursor: u64,
    used_cursor: u64,
    free_cursor: u64,
    ticks: [u64; REGIONS],
    next_index: u64,
    uploads: [Level; 64],
    downloads: [Level; 64],
    device_local: [Level; 64],
    delete_level: usize,
}
impl StagingPool {
    pub fn new(owner: Arc<DeviceOwner>) -> Result<Self, VulkanError> {
        let mut pool = Self {
            owner,
            stream: None,
            stream_pointer: 0,
            cursor: 0,
            used_cursor: 0,
            free_cursor: 0,
            ticks: [0; REGIONS],
            next_index: 0,
            uploads: std::array::from_fn(|_| Level::default()),
            downloads: std::array::from_fn(|_| Level::default()),
            device_local: std::array::from_fn(|_| Level::default()),
            delete_level: 0,
        };
        let stream = Arc::new(pool.allocate(STREAM_BYTES, Usage::Upload)?);
        pool.stream_pointer = stream.coherent_mapped_address();
        pool.stream = Some(stream);
        Ok(pool)
    }
    fn active(&self, begin: usize, end: usize, completed: u64) -> bool {
        (begin..end).any(|i| self.ticks[i] > completed)
    }
    pub fn reserve(
        &mut self,
        size: u64,
        alignment: u64,
        usage: Usage,
        tick: u64,
        completed: u64,
    ) -> Result<Slice, VulkanError> {
        if size == 0 || !alignment.is_power_of_two() {
            return Err(VulkanError::InvalidResource(
                "invalid staging size/alignment",
            ));
        }
        if usage == Usage::Upload && size <= REGION_BYTES && alignment <= REGION_BYTES {
            let alignment = alignment
                .max(self.stream.as_ref().unwrap().host_alignment())
                .max(256);
            let start = align_up(self.cursor, alignment)
                .ok_or(VulkanError::InvalidResource("staging offset overflow"))?;
            let end = start
                .checked_add(size)
                .ok_or(VulkanError::InvalidResource("staging size overflow"))?;
            // GetStreamBuffer only stamps regions when leaving them. Appending
            // within the current region does not revalidate or rewrite its tick.
            // https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/renderer_vulkan/vk_staging_buffer_pool.cpp#L134-L174
            if !self.active(
                (self.free_cursor / REGION_BYTES) as usize + 1,
                ((end / REGION_BYTES) as usize + 1).min(REGIONS),
                completed,
            ) {
                self.ticks
                    [(self.used_cursor / REGION_BYTES) as usize..(start / REGION_BYTES) as usize]
                    .fill(tick);
                self.used_cursor = start;
                self.free_cursor = self.free_cursor.max(end);
                let offset = if end >= STREAM_BYTES {
                    self.ticks[(self.used_cursor / REGION_BYTES) as usize..].fill(tick);
                    if self.active(0, (size / REGION_BYTES) as usize + 1, completed) {
                        self.cursor = start;
                        return self.reserve_pooled(size, alignment, usage, tick, completed);
                    }
                    // A pooled fallback must not make a later request treat
                    // the still-busy first region as already reserved here.
                    self.used_cursor = 0;
                    self.free_cursor = size;
                    0
                } else {
                    start
                };
                self.cursor = align_up(offset + size, alignment)
                    .ok_or(VulkanError::InvalidResource("staging offset overflow"))?;
                return Ok(Slice {
                    source: Source::Stream,
                    mapped_address: if self.stream_pointer == 0 {
                        0
                    } else {
                        self.stream_pointer + offset as usize
                    },
                    buffer: self.stream.as_ref().unwrap().handle,
                    offset,
                    size,
                });
            }
        }
        self.reserve_pooled(size, alignment, usage, tick, completed)
    }
    fn reserve_pooled(
        &mut self,
        size: u64,
        alignment: u64,
        usage: Usage,
        tick: u64,
        completed: u64,
    ) -> Result<Slice, VulkanError> {
        let capacity = size
            .max(alignment)
            .max(self.owner.non_coherent_atom_size)
            .checked_next_power_of_two()
            .ok_or(VulkanError::InvalidResource("staging capacity overflow"))?;
        let level = capacity.ilog2() as usize;
        let cache = match usage {
            Usage::Upload => &mut self.uploads[level],
            Usage::Download => &mut self.downloads[level],
            Usage::DeviceLocal => &mut self.device_local[level],
        };
        let index = (cache.next..cache.entries.len())
            .chain(0..cache.next)
            .find(|&i| {
                Arc::strong_count(&cache.entries[i]) == 1 && cache.entries[i].reusable(completed)
            });
        let entry = if let Some(index) = index {
            let entry = &cache.entries[index];
            entry.tick.store(tick, Ordering::Relaxed);
            entry
                .deferred
                .store(usage == Usage::Download, Ordering::Relaxed);
            cache.next = (index + 1) % cache.entries.len();
            entry
        } else {
            let buffer = Arc::new(self.allocate(capacity, usage)?);
            let entry = Arc::new(Entry {
                index: self.next_index,
                mapped_address: buffer.coherent_mapped_address(),
                buffer,
                tick: AtomicU64::new(tick),
                deferred: AtomicBool::new(usage == Usage::Download),
                usage,
            });
            let cache = match usage {
                Usage::Upload => &mut self.uploads[level],
                Usage::Download => &mut self.downloads[level],
                Usage::DeviceLocal => &mut self.device_local[level],
            };
            self.next_index += 1;
            cache.entries.push(entry);
            cache.entries.last().unwrap()
        };
        Ok(Slice {
            source: Source::Pooled {
                usage,
                level: level as u32,
                index: entry.index,
            },
            buffer: entry.buffer.handle,
            mapped_address: entry.mapped_address,
            offset: 0,
            size,
        })
    }
    fn allocate(&self, size: u64, usage: Usage) -> Result<AllocatedBuffer, VulkanError> {
        let (usage, location, label) = match usage {
            Usage::Upload => (
                vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::VERTEX_BUFFER
                    | vk::BufferUsageFlags::INDEX_BUFFER
                    | vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::UNIFORM_BUFFER
                    | vk::BufferUsageFlags::UNIFORM_TEXEL_BUFFER
                    | vk::BufferUsageFlags::STORAGE_TEXEL_BUFFER,
                MemoryLocation::CpuToGpu,
                "nixe upload staging",
            ),
            Usage::Download => (
                vk::BufferUsageFlags::TRANSFER_DST,
                MemoryLocation::GpuToCpu,
                "nixe readback staging",
            ),
            Usage::DeviceLocal => (
                vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::INDEX_BUFFER,
                MemoryLocation::GpuOnly,
                "nixe device-local staging",
            ),
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
    fn entry(&self, source: Source) -> &Arc<Entry> {
        let Source::Pooled {
            usage,
            level,
            index,
        } = source
        else {
            unreachable!("stream reference has no pooled entry")
        };
        let cache = match usage {
            Usage::Upload => &self.uploads,
            Usage::Download => &self.downloads,
            Usage::DeviceLocal => &self.device_local,
        };
        // Like FreeDeferred, use the stable allocation identity: TickFrame can
        // compact a level after other references have already been reserved.
        cache[level as usize]
            .entries
            .iter()
            .find(|entry| entry.index == index)
            .expect("live staging allocation")
    }
    pub(crate) fn buffer(&self, slice: &Slice) -> &Arc<AllocatedBuffer> {
        match slice.source {
            Source::Stream => self.stream.as_ref().expect("allocated stream"),
            source => &self.entry(source).buffer,
        }
    }
    /// Fill a fresh reservation directly from its mapped span, as
    /// StagingBufferRef::mapped_span. No mapped bytes escape the fill callback.
    ///
    /// # Safety
    /// The reservation belongs to this live pool, is still retained, and has
    /// no concurrent host or queued GPU users of the bytes being filled.
    pub(crate) unsafe fn write_with<T>(
        &mut self,
        slice: &Slice,
        fill: impl FnOnce(&mut [u8]) -> Result<T, VulkanError>,
    ) -> Result<T, VulkanError> {
        if slice.mapped_address == 0 {
            // Non-coherent memory still needs atom-isolated publication.
            return unsafe {
                self.buffer(slice)
                    .write_with(slice.offset, slice.size, fill)
            };
        }
        // SAFETY: Request accepted the exact span; the caller owns its fresh
        // reservation. Coherent host writes need no additional publication.
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(
                std::ptr::with_exposed_provenance_mut(slice.mapped_address),
                slice.size as usize,
            )
        };
        fill(bytes)
    }
    pub(super) fn retain_readback(&self, slice: &Slice) -> ReadbackAllocation {
        let entry = self.entry(slice.source);
        assert_eq!(entry.usage, Usage::Download);
        ReadbackAllocation(Arc::clone(entry))
    }
    pub fn is_stream(&self, slice: &Slice) -> bool {
        matches!(slice.source, Source::Stream)
    }
    pub fn retained_readbacks(&self) -> usize {
        self.downloads
            .iter()
            .flat_map(|level| &level.entries)
            .filter(|entry| entry.deferred.load(Ordering::Relaxed))
            .count()
    }
    pub fn tick_frame(&mut self, completed: u64) {
        self.delete_level = (self.delete_level + 1) % 64;
        for cache in [
            &mut self.uploads[self.delete_level],
            &mut self.downloads[self.delete_level],
            &mut self.device_local[self.delete_level],
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
        self.stream_pointer = 0;
        for level in self
            .uploads
            .iter_mut()
            .chain(&mut self.downloads)
            .chain(&mut self.device_local)
        {
            level.entries.clear();
            level.next = 0;
            level.delete = 0;
        }
        self.cursor = 0;
        self.used_cursor = 0;
        self.free_cursor = 0;
        self.ticks.fill(0);
    }
    pub fn stream_capacity(&self) -> u64 {
        self.stream.as_ref().map_or(0, |b| b.size())
    }
    pub fn pooled_capacity(&self) -> u64 {
        self.uploads
            .iter()
            .chain(&self.downloads)
            .chain(&self.device_local)
            .flat_map(|level| &level.entries)
            .map(|e| e.buffer.size())
            .sum()
    }
}
