//! Native caches share the machine's physical resource authority. Only actual
//! accepted operations retain guest inputs; cached interpretations remain weak.
mod affine;
pub use affine::affine_pipeline;
mod buffer;
pub use buffer::BufferRange;
mod buffer_download;
pub(crate) mod commands;
mod conversion;
pub(crate) use conversion::stage as helper_stage;
mod budget;
mod cpu_readback;
mod decoded_texture;
pub(crate) mod execution;
mod format;
mod image;
pub(crate) mod presentation;
#[cfg(feature = "binding-probe")]
pub(crate) mod probe_commands;
mod raster_transfer;
mod readback;
mod recording;
mod sample_readback;
mod transfers;
pub use commands::{AttachmentLoad, AttachmentStore};
pub use conversion::{PreparedNv12, nv12_pipeline};
mod sync;
pub use sync::{AccessScope, BarrierPolicy, BarrierStatistics, LayoutPolicy, RecordingPolicy};
pub(crate) mod video;
use crate::{
    AllocatedBuffer, AllocatedImage, AllocationMode, MemoryLocation, VulkanDevice, VulkanError,
    WorkTicket,
    memory::{
        AllocationBudget, AllocationCharge, NativeImageDescription, allocate_buffer_charged,
        allocate_image_charged,
    },
};
use ash::vk;
use nixe_gpu::BackendInstanceId;
use nixe_gpu::authority::Authority;
use nixe_gpu::authority::{ResourceObservation, WriterId};
use nixe_gpu::physical::{Footprint, PhysicalMapping};
use nixe_memory::Machine;
use std::sync::Arc;

macro_rules! cache_id {
    ($name:ident) => {
        #[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
        pub struct $name {
            instance: BackendInstanceId,
            slot: usize,
            generation: u32,
        }
    };
}
cache_id!(BufferId);
cache_id!(ImageId);
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum Resource {
    Buffer(BufferId),
    Image(ImageId),
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ImageViewId {
    pub(crate) image: ImageId,
    view: usize,
}
mod attachment;
pub use attachment::AttachmentId;
struct Slot<T> {
    generation: u32,
    entry: Option<T>,
}
#[derive(Clone, Copy)]
struct LruLinks {
    previous: Option<Resource>,
    next: Option<Resource>,
}
struct Contents {
    lru: Option<LruLinks>,
    geometry: Vec<Arc<Footprint>>,
    device: Vec<ResourceObservation>,
    last_use: Option<WorkTicket>,
    touched: u64,
}
impl Contents {
    fn observations_current(&self) -> bool {
        self.device
            .iter()
            .all(|observation| !observation.is_invalid())
    }
}
/// A recorder consumes a bounded prefix; the execution owner retains the rest
/// under its original admission permit and continues after staging pressure.
#[must_use]
#[derive(Clone, Copy, Debug)]
pub struct BufferUpload {
    pub work: WorkTicket,
    pub consumed: usize,
}
struct KnownBufferBytes {
    offset: u64,
    bytes: Arc<[u8]>,
    /// A completion effect needs no physical transfer until a GPU access.
    pending_upload: bool,
}
struct BufferEntry {
    invalid: Vec<(u64, u64)>,
    /// Immutable host-authored payload, valid until another native producer.
    /// CPU publication still waits for the actual producing ticket, including
    /// when the physical buffer has never been consumed by the GPU.
    known: Vec<KnownBufferBytes>,
    growth: u32,
    binding_valid: bool,
    write_plans: Vec<(u64, u64, WriterId)>,
    contents: Contents,
    native: Arc<AllocatedBuffer>,
    usage: vk::BufferUsageFlags,
    #[cfg(feature = "binding-probe")]
    address: vk::DeviceAddress,
    access: sync::BufferAccess,
}
impl BufferEntry {
    fn validate_interval(&mut self, start: u64, size: u64) {
        self.validate_intervals(&[(start, size)]);
    }
    /// Subtract ordered, non-overlapping written intervals in one sweep.
    /// Packed depth footprints can have one three-byte interval per texel.
    fn validate_intervals(&mut self, written: &[(u64, u64)]) {
        if self.invalid.is_empty() {
            return;
        }
        let mut invalid = Vec::new();
        let mut next = 0;
        for (offset, length) in self.invalid.drain(..) {
            let mut cursor = offset;
            let stop = offset + length;
            while next < written.len() && written[next].0 + written[next].1 <= cursor {
                next += 1;
            }
            while let Some(&(start, size)) = written.get(next) {
                if start >= stop {
                    break;
                }
                if cursor < start {
                    invalid.push((cursor, start - cursor));
                }
                cursor = cursor.max(start + size);
                if cursor >= stop {
                    break;
                }
                next += 1;
            }
            if cursor < stop {
                invalid.push((cursor, stop - cursor));
            }
        }
        self.invalid = invalid;
    }
}
pub(crate) struct ImageEntry {
    synchronized: bool,
    synchronized_sequence: u64,
    cpu_tracked: bool,
    pub(crate) guest: Option<nixe_gpu::native::GuestImage>,
    binding_valid: bool,
    contents: Contents,
    native: Arc<AllocatedImage>,
    pub(crate) description: NativeImageDescription,
    pub(crate) planes: Arc<[ImagePlane]>,
    sample_grid: [u32; 2],
    views: Vec<ImageViewDescription>,
    native_views: Vec<Arc<crate::AllocatedImageView>>,
    access: Vec<sync::ImageAccess>,
    version: u64,
    producer: Option<WorkTicket>,
    discarded: Vec<(usize, vk::Rect2D)>,
    decoded: Option<decoded_texture::DecodedImage>,
    canonical_import: bool,
    write_plans: rustc_hash::FxHashMap<(usize, nixe_gpu::physical::ByteRect), WriterId>,
}
impl ImageEntry {
    fn current(&self) -> bool {
        self.synchronized
            && self
                .contents
                .device
                .iter()
                .all(|observer| !observer.is_invalid())
    }
    pub(crate) fn view_format(&self, requested: vk::Format) -> vk::Format {
        if self.decoded.is_some() {
            self.description.format
        } else {
            requested
        }
    }
    fn storage_plane(&self, mut plane: ImagePlane) -> ImagePlane {
        plane.extent.width *= self.sample_grid[0];
        plane.extent.height *= self.sample_grid[1];
        plane
    }
    fn storage_rect(&self, mut rect: vk::Rect2D) -> vk::Rect2D {
        rect.offset.x *= self.sample_grid[0] as i32;
        rect.offset.y *= self.sample_grid[1] as i32;
        rect.extent.width *= self.sample_grid[0];
        rect.extent.height *= self.sample_grid[1];
        rect
    }
}
/// Native copy plane in the canonical byte stream. Guest block-linear decoding
/// and other guest encodings are legalized by the selected frontend, not this cache.
/// Row padding remains canonical CPU storage; GPU image writes own texel bytes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ImagePlane {
    pub offset: u64,
    pub row_pitch: u64,
    pub mip_level: u32,
    pub array_layer: u32,
    pub aspect: vk::ImageAspectFlags,
    pub extent: vk::Extent3D,
    pub block_extent: vk::Extent2D,
    pub bytes_per_block: u32,
}
impl ImagePlane {
    fn row_bytes(self) -> u64 {
        u64::from(self.extent.width.div_ceil(self.block_extent.width))
            * u64::from(self.bytes_per_block)
    }
    fn rows(self) -> u32 {
        self.extent.height.div_ceil(self.block_extent.height)
    }
    fn end(self) -> Option<u64> {
        self.offset
            .checked_add(self.row_pitch.checked_mul(u64::from(self.rows() - 1))?)
            .and_then(|v| v.checked_add(self.row_bytes()))
    }
    fn layers(self) -> vk::ImageSubresourceLayers {
        vk::ImageSubresourceLayers::default()
            .aspect_mask(self.aspect)
            .mip_level(self.mip_level)
            .base_array_layer(self.array_layer)
            .layer_count(1)
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ImageViewDescription {
    pub format: vk::Format,
    pub view_type: vk::ImageViewType,
    pub aspects: vk::ImageAspectFlags,
    pub base_mip: u32,
    pub mip_count: u32,
    pub base_layer: u32,
    pub layer_count: u32,
    pub swizzle: [vk::ComponentSwizzle; 4],
}
#[derive(Clone, Copy, Debug, Default)]
pub struct ResidencyStatistics {
    pub attachment_publication_hits: u64,
    pub cpu_uploaded_bytes: u64,
    pub cpu_streamed_bytes: u64,
    pub gpu_copied_bytes: u64,
    pub alias_copy_commands: u64,
    pub alias_copy_regions: u64,
    pub decoded_upload_commands: u64,
    /// Exact bytes transferred to host memory, excluding GPU-only realignment.
    pub native_readback_bytes: u64,
    pub canonical_readback_bytes: u64,
    pub cache_hits: u64,
    pub buffer_allocations: u64,
    pub image_allocations: u64,
    pub image_layout_plans: u64,
    pub image_view_allocations: u64,
    pub buffer_merges: u64,
    pub staging_pressure_waits: u64,
    pub evictions: u64,
}
/// Concrete single-owner storage, selected once at startup. No generic backend
/// handle registry, virtual resource dispatch or guest-ISA branch in its hot path.
pub struct VulkanResidency {
    pub(crate) device: VulkanDevice,
    instance: BackendInstanceId,
    buffers: Vec<Slot<BufferEntry>>,
    free_buffers: Vec<usize>,
    epoch: u64,
    lru_head: Option<Resource>,
    lru_tail: Option<Resource>,
    lru_scan: Option<Resource>,
    budget_ceiling: u64,
    collection_target: u64,
    maintenance_at: std::time::Instant,
    images: Vec<Slot<ImageEntry>>,
    free_images: Vec<usize>,
    attachments: Vec<Slot<attachment::AttachmentEntry>>,
    free_attachments: Vec<usize>,
    memory: Machine,
    authority: Authority<Resource>,
    budget: AllocationBudget,
    statistics: ResidencyStatistics,
    image_readbacks: readback::ImageReadbacks,
    cpu_readbacks: rustc_hash::FxHashMap<nixe_memory::AccessRequestId, cpu_readback::CpuReadback>,
    buffer_writes: rustc_hash::FxHashMap<BufferId, Vec<(u64, u64)>>,
    buffer_downloads: std::collections::VecDeque<buffer_download::Download>,
    barriers: sync::Barriers,
    decoder: nixe_gpu::texture_decode::BlockDecoder,
    pub(crate) video: video::VideoTransfers,
    sample_conversion: Option<sample_readback::SampleConversion>,
    #[cfg(feature = "binding-probe")]
    helpers: crate::helpers::Helpers,
    #[cfg(feature = "binding-probe")]
    descriptor_dispatch: Option<ash::ext::descriptor_buffer::Device>,
    barrier_policy: BarrierPolicy,
    unified_layouts: bool,
    /// Entries needed by a cold reconstruction cannot be evicted by its allocations.
    pub(crate) constructing: Vec<Resource>,
    descriptor_resources: Vec<crate::cache::DescriptorResource>,
    descriptor_offsets: Vec<u32>,
    streamed_uses: Vec<usize>,
    stream_inputs: Vec<nixe_memory::AllocationRange>,
    uniform_cache_hits: [u64; 18],
    uniform_cache_shots: [u64; 18],
    uniform_stream_limit: u64,
}
impl VulkanResidency {
    pub(crate) fn memory(&self) -> &Machine {
        &self.memory
    }
    pub(crate) unsafe fn owner_memory_written(
        &mut self,
        extents: &[(nixe_memory::AllocationId, nixe_memory::Extent)],
    ) -> Result<(), VulkanError> {
        // RAM is already updated under the original mapping guard. Exact native
        // coverage changes now; dirty-page cache invalidation is consumed at the
        // next resource access, just like the cached-write accumulator.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/memory_manager.cpp
        self.authority
            .cpu_write_extents(extents.iter().copied())
            .map_err(memory_error)?;
        unsafe {
            self.authority
                .retire_unused_device_extents(&self.memory, extents.iter().copied())
        }
        .map_err(memory_error)
    }
    pub(crate) fn consume_cpu_writes(
        &mut self,
        ranges: Vec<nixe_memory::AllocationRange>,
    ) -> Result<(), VulkanError> {
        self.invalidate_cpu_resources(&ranges)?;
        if !ranges.is_empty() {
            self.authority.cpu_write(&ranges)?;
            self.authority
                .retire_unused_device_pages(&self.memory, &ranges)?;
        }
        Ok(())
    }
    fn invalidate_cpu_resources(
        &mut self,
        ranges: &[nixe_memory::AllocationRange],
    ) -> Result<(), VulkanError> {
        let modified = self.memory.take_cpu_modified_pages();
        let buffers = &mut self.buffers;
        let images = &mut self.images;
        let mut failure = None;
        let mut touched = smallvec::SmallVec::<[usize; 16]>::new();
        for extent in modified.into_iter().chain(
            ranges
                .iter()
                .flat_map(nixe_memory::AllocationRange::physical_extents),
        ) {
            self.authority
                .visit_resource_overlaps(extent, |resource, footprint, span| {
                    if let Resource::Image(id) = resource {
                        let entry = images[id.slot].entry.as_mut().expect("registered image");
                        if let Err(error) =
                            footprint.visit(span, extent.start, extent.bytes, |_| {
                                entry.synchronized = false;
                                entry.cpu_tracked = false;
                                false
                            })
                        {
                            failure = Some(error);
                        }
                        return;
                    }
                    let Resource::Buffer(id) = resource else {
                        unreachable!()
                    };
                    let entry = buffers[id.slot].entry.as_mut().expect("registered buffer");
                    if entry.invalid.as_slice() == [(0, entry.native.size())] {
                        return;
                    }
                    touched.push(id.slot);
                    if matches!(
                        footprint.domain(),
                        nixe_gpu::physical::WriteDomain::Buffer { .. }
                    ) {
                        if let Err(error) =
                            footprint.visit(span, extent.start, extent.bytes, |part| {
                                entry.invalid.push((part.native, part.bytes));
                                true
                            })
                        {
                            failure = Some(error);
                        }
                    } else {
                        entry.invalid.clear();
                        entry.invalid.push((0, entry.native.size()));
                    }
                });
        }
        if let Some(error) = failure {
            return Err(memory_error(error));
        }
        touched.sort_unstable();
        touched.dedup();
        for slot in touched {
            normalize(&mut buffers[slot].entry.as_mut().unwrap().invalid);
        }
        Ok(())
    }
    pub fn new(memory: Machine, device: VulkanDevice, bytes: u64) -> Result<Self, VulkanError> {
        if bytes == 0 {
            return Err(VulkanError::InvalidResource("empty residency budget"));
        }
        let instance = device.submission.ticket(0).submission().instance();
        let unified_layouts = device.profile().unified_image_layouts;
        #[cfg(feature = "binding-probe")]
        let descriptor_dispatch = device.profile().descriptor_buffer.then(|| {
            ash::ext::descriptor_buffer::Device::new(
                &device.owner.instance.instance,
                &device.owner.device,
            )
        });
        let mut cache = Self {
            device,
            instance,
            buffers: Vec::new(),
            free_buffers: Vec::new(),
            epoch: 0,
            lru_head: None,
            lru_tail: None,
            lru_scan: None,
            budget_ceiling: bytes,
            collection_target: bytes / 8 * 7,
            maintenance_at: std::time::Instant::now(),
            images: Vec::new(),
            free_images: Vec::new(),
            attachments: Vec::new(),
            free_attachments: Vec::new(),
            authority: Authority::new(memory.identity()),
            memory,
            budget: AllocationBudget::new(bytes),
            statistics: ResidencyStatistics::default(),
            image_readbacks: readback::ImageReadbacks::default(),
            cpu_readbacks: Default::default(),
            buffer_writes: Default::default(),
            buffer_downloads: Default::default(),
            barriers: sync::Barriers::default(),
            decoder: Default::default(),
            video: Default::default(),
            sample_conversion: None,
            #[cfg(feature = "binding-probe")]
            helpers: crate::helpers::Helpers::default(),
            #[cfg(feature = "binding-probe")]
            descriptor_dispatch,
            barrier_policy: BarrierPolicy::default(),
            unified_layouts,
            constructing: Vec::new(),
            descriptor_resources: Vec::new(),
            descriptor_offsets: Vec::new(),
            streamed_uses: Vec::new(),
            stream_inputs: Vec::new(),
            uniform_cache_hits: [0; 18],
            uniform_cache_shots: [0; 18],
            uniform_stream_limit: 4096,
        };
        cache.maintain_budget(true)?;
        Ok(cache)
    }
    pub fn device(&self) -> &VulkanDevice {
        &self.device
    }
    pub(crate) fn device_mut(&mut self) -> &mut VulkanDevice {
        &mut self.device
    }
    pub fn statistics(&self) -> ResidencyStatistics {
        self.statistics
    }
    pub fn allocated_bytes(&self) -> u64 {
        self.budget.used()
    }
    /// Cold accounting includes allocator slabs and the separately bounded
    /// staging arenas. The resource budget charges native memory requirements,
    /// not logical image texel sizes; allocator capacity includes fragmentation.
    pub fn allocator_capacity_bytes(&self) -> u64 {
        self.device
            .owner
            .allocator
            .lock()
            .unwrap()
            .as_ref()
            .expect("live allocator")
            .capacity()
    }
    fn contents(&self, resource: Resource) -> Result<&Contents, VulkanError> {
        match resource {
            Resource::Buffer(id) => Ok(&self.buffer(id)?.contents),
            Resource::Image(id) => Ok(&self.image(id)?.contents),
        }
    }
    fn contents_mut(&mut self, resource: Resource) -> &mut Contents {
        match resource {
            Resource::Buffer(id) => {
                &mut self.buffers[id.slot]
                    .entry
                    .as_mut()
                    .expect("live indexed buffer")
                    .contents
            }
            Resource::Image(id) => {
                &mut self.images[id.slot]
                    .entry
                    .as_mut()
                    .expect("live indexed image")
                    .contents
            }
        }
    }
    fn buffer_mut(&mut self, id: BufferId) -> &mut BufferEntry {
        self.buffers[id.slot].entry.as_mut().expect("live buffer")
    }
    fn buffer(&self, id: BufferId) -> Result<&BufferEntry, VulkanError> {
        self.buffers
            .get(id.slot)
            .filter(|slot| id.instance == self.instance && slot.generation == id.generation)
            .and_then(|slot| slot.entry.as_ref())
            .ok_or(VulkanError::InvalidResource(
                "stale or foreign buffer cache ID",
            ))
    }
    pub(crate) fn image(&self, id: ImageId) -> Result<&ImageEntry, VulkanError> {
        self.images
            .get(id.slot)
            .filter(|slot| id.instance == self.instance && slot.generation == id.generation)
            .and_then(|slot| slot.entry.as_ref())
            .ok_or(VulkanError::InvalidResource(
                "stale or foreign image cache ID",
            ))
    }
    fn reserve(&mut self, bytes: u64) -> Result<Option<AllocationCharge>, VulkanError> {
        if let Some(charge) = self.budget.reserve(bytes) {
            return Ok(Some(charge));
        }
        self.device.poll_progress()?;
        // Descriptor lookups are reconstructible; completed slot caches must
        // not pin otherwise evictable resources under residency pressure.
        // In-flight records retain their leases until actual GPU completion.
        for slot in &mut self.device.submission.slots {
            if let Some(descriptors) = &mut slot.stream_descriptors {
                descriptors.trim();
            }
        }
        while self
            .budget
            .used()
            .checked_add(bytes)
            .is_none_or(|used| used > self.budget.limit)
        {
            if self.video.evict_idle(self.device.completed_tick()) {
                continue;
            }
            if !self.evict_one()? {
                self.flush()?;
                let completed = self.device.completed_tick();
                let oldest = self
                    .device
                    .submission
                    .slots
                    .iter()
                    .map(|slot| slot.tick)
                    .filter(|&tick| tick > completed && tick <= self.device.scheduled_tick())
                    .min();
                if let Some(tick) = oldest {
                    self.device.wait_for_work(
                        self.device.submission.ticket(tick),
                        std::time::Duration::MAX,
                    )?;
                    self.progress_readbacks()?;
                    for slot in &mut self.device.submission.slots {
                        if let Some(descriptors) = &mut slot.stream_descriptors {
                            descriptors.trim();
                        }
                    }
                    continue;
                }
                return Err(VulkanError::ResidencyPressure {
                    requested: bytes,
                    allocated: self.budget.used(),
                    budget: self.budget.limit,
                });
            }
        }
        Ok(self.budget.reserve(bytes))
    }
    fn retire_entry(&mut self, resource: Resource) -> Result<(), VulkanError> {
        self.unlink_lru(resource);
        match resource {
            Resource::Buffer(id) => {
                let entry = self.buffers[id.slot].entry.take().expect("candidate");
                if self.buffers[id.slot].generation < u32::MAX {
                    self.free_buffers.push(id.slot);
                }
                if let Some(last_use) = entry.contents.last_use {
                    self.device.retire_buffer(entry.native, last_use)?;
                }
            }
            Resource::Image(id) => {
                let entry = self.images[id.slot].entry.take().expect("candidate");
                if self.images[id.slot].generation < u32::MAX {
                    self.free_images.push(id.slot);
                }
                if let Some(last_use) = entry.contents.last_use {
                    self.device
                        .retire_image_views(entry.native_views, last_use)?;
                    self.device.retire_image(entry.native, last_use)?;
                }
            }
        }
        Ok(())
    }
    pub(crate) fn check_machine(&self, machine: nixe_memory::MachineId) -> Result<(), VulkanError> {
        if machine != self.memory.identity() {
            return Err(VulkanError::InvalidResource(
                "memory notification belongs to another machine",
            ));
        }
        Ok(())
    }
    pub(crate) fn retire_physical_memory(
        &mut self,
        machine: nixe_memory::MachineId,
        allocation: nixe_memory::AllocationId,
        extents: &[nixe_memory::Extent],
    ) -> Result<(), VulkanError> {
        let resources = self
            .authority
            .retire_physical_ranges(machine, allocation, extents)?;
        for resource in resources {
            if self.authority.mapping(resource, 0).is_none() {
                self.retire_entry(resource)?;
            } else {
                match resource {
                    Resource::Buffer(id) => {
                        let entry = self.buffers[id.slot].entry.as_mut().unwrap();
                        entry.binding_valid = false;
                        entry.invalid = vec![(0, entry.native.size())];
                    }
                    Resource::Image(id) => {
                        let entry = self.images[id.slot].entry.as_mut().unwrap();
                        entry.binding_valid = false;
                        entry.synchronized = false;
                        entry.cpu_tracked = false;
                    }
                }
            }
        }
        Ok(())
    }
    fn unpin(&mut self, resource: Resource) {
        let index = self
            .constructing
            .iter()
            .rposition(|item| *item == resource)
            .expect("active construction pin");
        self.constructing.remove(index);
    }
    fn temporary_buffer(
        &mut self,
        size: u64,
        references: &[Resource],
    ) -> Result<AllocatedBuffer, VulkanError> {
        self.constructing.extend_from_slice(references);
        let result = self.allocate_buffer(
            size,
            vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
        );
        for &resource in references.iter().rev() {
            self.unpin(resource);
        }
        result
    }
    fn allocate_buffer(
        &mut self,
        size: u64,
        usage: vk::BufferUsageFlags,
    ) -> Result<AllocatedBuffer, VulkanError> {
        let owner = Arc::clone(&self.device.owner);
        allocate_buffer_charged(
            &owner,
            size,
            usage,
            MemoryLocation::GpuOnly,
            AllocationMode::Managed,
            "resident buffer",
            |bytes| self.reserve(bytes),
        )
    }
    fn wait_writer(&mut self, writer: WriterId) -> Result<WriterId, VulkanError> {
        while self
            .authority
            .writer_dependencies(writer)
            .is_some_and(|ids| !ids.is_empty())
        {
            self.progress_device_accesses()?;
            if self
                .authority
                .writer_dependencies(writer)
                .is_none_or(|ids| ids.is_empty())
            {
                break;
            }
            self.flush()?;
            self.progress_device_accesses()?;
            if self
                .authority
                .writer_dependencies(writer)
                .is_none_or(|ids| ids.is_empty())
            {
                break;
            }
            let point = self.readback_point().ok_or(VulkanError::InvalidResource(
                "writer dependency has no native progress point",
            ))?;
            self.device.wait_for_work(
                self.device.submission.ticket(point),
                std::time::Duration::MAX,
            )?;
        }
        Ok(writer)
    }
    fn publish_effect<T>(
        &mut self,
        writers: &[WriterId],
        append: impl FnOnce(&mut VulkanDevice) -> T,
    ) -> Result<T, VulkanError> {
        if writers.is_empty() {
            return Ok(append(&mut self.device));
        }
        let mut unique = smallvec::SmallVec::<[WriterId; 16]>::new();
        for &writer in writers {
            if !unique.contains(&writer) {
                unique.push(writer);
            }
        }
        let tick = nixe_gpu::authority::NativeTick(self.device.work_ticket()?.tick());
        let memory = self.memory.clone();
        // The detached collection orders preceding CPU stores before this effect.
        // Stores collected at the next boundary are later than this effect.
        self.consume_cpu_writes(memory.collect_cpu_writes()?)?;
        self.authority
            .publish_writes(&self.memory, &unique, || {
                Ok((append(&mut self.device), tick))
            })
            .map(|(value, _)| value)
            .map_err(memory_error)
    }
    fn prepare_write(
        &mut self,
        resource: Resource,
        offset: u64,
        bytes: u64,
    ) -> Result<WriterId, VulkanError> {
        let Resource::Buffer(id) = resource else {
            return Err(VulkanError::InvalidResource(
                "linear write requires a buffer",
            ));
        };
        self.check_buffer_range(id, offset, bytes)?;
        // Whole literal overwrites can discard their pending upload. Partial
        // overwrites first preserve the untouched bytes in native storage.
        let partial = self.buffer(id)?.known.iter().any(|literal| {
            let end = literal.offset + literal.bytes.len() as u64;
            literal.pending_upload
                && literal.offset < offset + bytes
                && offset < end
                && !(offset <= literal.offset && offset + bytes >= end)
        });
        if partial {
            self.buffer_access_range(
                id,
                offset,
                bytes,
                AccessScope::new(
                    vk::PipelineStageFlags2::COPY,
                    vk::AccessFlags2::TRANSFER_READ,
                ),
            )?;
        }
        self.buffers[id.slot]
            .entry
            .as_mut()
            .unwrap()
            .known
            .retain(|literal| {
                literal.offset >= offset + bytes
                    || literal.offset + literal.bytes.len() as u64 <= offset
            });
        let old = self
            .buffer(id)?
            .write_plans
            .iter()
            .find(|(start, size, _)| *start == offset && *size == bytes)
            .map(|entry| entry.2);
        let writer = if let Some(writer) = old {
            writer
        } else {
            let [geometry] = self.buffer(id)?.contents.geometry.as_slice() else {
                return Err(VulkanError::InvalidResource(
                    "linear write requires one physical footprint",
                ));
            };
            let domain = Footprint::new(
                Arc::clone(geometry.mapping()),
                nixe_gpu::physical::WriteDomain::Buffer {
                    start: offset,
                    bytes,
                },
            )?;
            let writer = self.authority.create_writer(resource, domain)?;
            self.buffers[id.slot]
                .entry
                .as_mut()
                .expect("buffer")
                .write_plans
                .push((offset, bytes, writer));
            writer
        };
        self.wait_writer(writer)
    }
    fn finish_buffer_write(
        &mut self,
        id: BufferId,
        offset: u64,
        bytes: u64,
    ) -> Result<(), VulkanError> {
        let full = offset == 0 && bytes == self.buffer(id)?.native.size();
        let renew = full
            && self
                .buffer(id)?
                .contents
                .geometry
                .iter()
                .all(|f| f.mapping().is_injective())
            && self
                .buffer(id)?
                .contents
                .device
                .iter()
                .any(ResourceObservation::is_invalid);
        if renew {
            let geometry = self.buffer(id)?.contents.geometry.clone();
            let observed = geometry
                .into_iter()
                .map(|f| self.authority.observe_native(Resource::Buffer(id), f))
                .collect::<Result<Vec<_>, _>>()?;
            self.buffers[id.slot]
                .entry
                .as_mut()
                .expect("buffer")
                .contents
                .device = observed;
        }
        self.track_buffer_cpu(id, offset, bytes)?;
        let work = self.device.work_ticket()?;
        let entry = self.buffers[id.slot].entry.as_mut().expect("buffer");
        entry.validate_interval(offset, bytes);
        entry.contents.last_use = Some(work);
        self.buffer_writes
            .entry(id)
            .or_default()
            .push((offset, bytes));
        Ok(())
    }
    pub fn write_buffer(
        &mut self,
        id: BufferId,
        offset: u64,
        bytes: &[u8],
    ) -> Result<BufferUpload, VulkanError> {
        let consumed = bytes
            .len()
            .min(self.device.submission.configuration.arena_bytes as usize);
        let bytes = &bytes[..consumed];
        let (slot, slice) = self.reserve_upload(bytes, 4)?;
        let upload = self.device.submission.slots[slot]
            .upload_buffer(slice)
            .handle;
        let writer = self.prepare_write(Resource::Buffer(id), offset, consumed as u64)?;
        let target = self.buffer_access_range(
            id,
            offset,
            consumed as u64,
            AccessScope::new(
                vk::PipelineStageFlags2::COPY,
                vk::AccessFlags2::TRANSFER_WRITE,
            ),
        )?;
        let command = self.device.submission.slots[slot].command;
        self.barriers.emit(&mut self.device);
        self.publish_effect(&[writer], |device| unsafe {
            device.dispatch.cmd_copy_buffer(
                command,
                upload,
                target,
                &[vk::BufferCopy::default()
                    .src_offset(slice.offset)
                    .dst_offset(offset)
                    .size(consumed as u64)],
            );
        })?;
        self.finish_buffer_write(id, offset, consumed as u64)?;
        Ok(BufferUpload {
            work: self.device.work_ticket()?,
            consumed,
        })
    }
    pub(crate) fn report_memory(
        &mut self,
        target: Vec<nixe_memory::AllocationRange>,
        offset: u64,
        bytes: &Arc<[u8]>,
        fence: bool,
    ) -> Result<(), VulkanError> {
        let target = nixe_memory::capture_subrange(&target, offset, bytes.len() as u64)?;
        let mapping = PhysicalMapping::new(&target)?;
        let id = self.buffer_for_deferred_import(
            mapping,
            vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
        )?;
        let writer = self.prepare_write(Resource::Buffer(id.buffer), id.offset, id.size)?;
        self.publish_effect(&[writer], |_| ())?;
        self.finish_buffer_write(id.buffer, id.offset, id.size)?;
        self.buffers[id.buffer.slot]
            .entry
            .as_mut()
            .expect("buffer")
            .known
            .push(KnownBufferBytes {
                offset: id.offset,
                bytes: Arc::clone(bytes),
                pending_upload: true,
            });
        // Query reports accumulate until a real fence commits the producing
        // prefix. They must not create a submission/completion per timestamp.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/fence_manager.h
        if fence {
            self.commit_buffer_downloads()?;
        }
        Ok(())
    }
    pub fn flush(&mut self) -> Result<Option<WorkTicket>, VulkanError> {
        self.flush_with_sync(&[], &[])
    }
    pub(crate) fn flush_with_sync(
        &mut self,
        waits: &[vk::SemaphoreSubmitInfo<'_>],
        signals: &[vk::SemaphoreSubmitInfo<'_>],
    ) -> Result<Option<WorkTicket>, VulkanError> {
        self.commit_buffer_downloads()?;
        self.progress_readbacks()?;
        if self.device.submission.recording.is_some() {
            self.barriers.emit(&mut self.device);
        }
        let work = match unsafe {
            self.device.submission.flush(
                self.device.queue,
                &mut self.device.dispatch,
                waits,
                signals,
            )
        } {
            Ok(work) => work,
            Err(error) => {
                let cause = match &error {
                    VulkanError::Memory(cause) => Arc::clone(cause),
                    _ => Arc::new(nixe_memory::MemoryError::Execution(
                        error.to_string().into(),
                    )),
                };
                self.authority.fail_accesses(cause);
                return Err(error);
            }
        };
        Ok(work)
    }
    fn preserve(&mut self, resource: Resource) -> Result<(), VulkanError> {
        let geometry = self.contents(resource)?.geometry.clone();
        let mut pins = Vec::new();
        for footprint in geometry {
            for part in self.authority.resolve_footprint(&footprint)? {
                if part
                    .producer
                    .is_some_and(|producer| producer.resource == resource && !producer.materialized)
                    && let Some(range) =
                        footprint
                            .mapping()
                            .pin_physical(part.allocation, part.start, part.bytes)?
                {
                    pins.push(range);
                }
            }
        }
        for range in pins {
            self.access_memory(vec![range], nixe_memory::AccessOperation::Read)?;
        }
        Ok(())
    }
    pub fn read_buffer(
        &mut self,
        id: BufferId,
        offset: u64,
        size: u64,
    ) -> Result<Vec<u8>, VulkanError> {
        self.ensure_buffer_range(id, offset, size)?;
        self.read_buffer_native(id, offset, size)
    }
}
impl Drop for VulkanResidency {
    fn drop(&mut self) {
        // Drain accepted native uses before entries destroy their objects. The
        // queue's fatal/device-lost teardown policy remains the single authority.
        if let Err(error) = self.device.shutdown()
            && !self.device.submission.slots.is_empty()
        {
            log::error!("Cannot safely destroy Vulkan residency before queue completion: {error}");
            std::process::abort();
        }
        self.authority
            .fail_accesses(Arc::new(nixe_memory::MemoryError::Execution(
                "Vulkan resource owner stopped".into(),
            )));
    }
}
fn normalize(intervals: &mut Vec<(u64, u64)>) {
    intervals.sort_unstable_by_key(|&(start, _)| start);
    let mut used = 0;
    for index in 0..intervals.len() {
        let (start, size) = intervals[index];
        if used != 0 && start <= intervals[used - 1].0 + intervals[used - 1].1 {
            let last = &mut intervals[used - 1];
            last.1 = (start + size).max(last.0 + last.1) - last.0;
        } else {
            intervals[used] = (start, size);
            used += 1;
        }
    }
    intervals.truncate(used);
}
fn memory_error(error: nixe_memory::MemoryError) -> VulkanError {
    VulkanError::from(error)
}

#[cfg(test)]
pub(crate) mod tests;

#[cfg(test)]
mod commands_tests;
