//! Specialized native caches. Canonical memory owns identity and CPU epochs;
//! this owner keeps exact GPU producers and reconstructible representations.
mod affine;
pub use affine::affine_pipeline;
mod buffer;
pub(crate) mod commands;
mod conversion;
pub(crate) use conversion::stage as helper_stage;
mod decoded_texture;
pub(crate) mod execution;
mod footprint;
mod format;
mod image;
mod index;
mod ownership;
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
mod visibility;
use crate::{
    AllocatedBuffer, AllocatedImage, AllocationMode, MemoryLocation, VulkanDevice, VulkanError,
    WorkTicket,
    memory::{
        AllocationBudget, AllocationCharge, NativeImageDescription, allocate_buffer_charged,
        allocate_image_charged,
    },
};
use ash::vk;
use index::{Authority, CanonicalIndex, Resource};
use nixe_gpu::{BackendInstanceId, PresentationImageId, PresentationLease, PresentationRelease};
use nixe_memory::{
    CanonicalBackingRange, CanonicalCpuWriteDependency, CanonicalPageId, CpuVisibilityRequest,
    CpuWriteSnapshotRequest, CpuWriteSnapshotSelection, DeviceAccessDeclaration,
    DeviceVisibilityPoint, NonCpuDeviceId, VisibilityState,
};
use std::sync::{Arc, mpsc};
pub use visibility::VulkanVisibility;

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
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ImageViewId {
    image: ImageId,
    view: usize,
}
mod attachment;
pub use attachment::AttachmentId;
struct Slot<T> {
    generation: u32,
    entry: Option<T>,
}
struct Contents {
    backing: CanonicalBackingRange,
    cpu: CanonicalCpuWriteDependency,
    invalid: Vec<(u64, u64)>,
    last_use: Option<WorkTicket>,
    touched: u64,
}
impl Contents {
    /// Union two normalized interval sets in one ordered sweep. A fully dirty
    /// image already covers every later partial alias invalidation in O(1).
    fn invalidate_intervals(&mut self, added: &[(u64, u64)]) {
        if added.is_empty() {
            return;
        }
        if let [(start, size)] = self.invalid.as_slice()
            && *start <= added[0].0
            && *start + *size >= added.last().unwrap().0 + added.last().unwrap().1
        {
            return;
        }
        let mut old = std::mem::take(&mut self.invalid).into_iter().peekable();
        let mut added = added.iter().copied().peekable();
        while old.peek().is_some() || added.peek().is_some() {
            let (start, size) = if old
                .peek()
                .is_some_and(|o| added.peek().is_none_or(|a| o.0 <= a.0))
            {
                old.next().unwrap()
            } else {
                added.next().unwrap()
            };
            if let Some(last) = self.invalid.last_mut()
                && start <= last.0 + last.1
            {
                last.1 = (start + size).max(last.0 + last.1) - last.0;
            } else {
                self.invalid.push((start, size));
            }
        }
    }
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
/// A recorder consumes a bounded prefix; the execution owner retains the rest
/// under its original admission permit and continues after staging pressure.
#[must_use]
#[derive(Clone, Copy, Debug)]
pub struct BufferUpload {
    pub work: WorkTicket,
    pub consumed: usize,
}
struct BufferEntry {
    streaming_cpu: Option<CanonicalCpuWriteDependency>,
    streamed: Option<buffer::StreamedBuffer>,
    image_import: Option<buffer::ImageBufferImport>,
    contents: Contents,
    native: Arc<AllocatedBuffer>,
    usage: vk::BufferUsageFlags,
    #[cfg(feature = "binding-probe")]
    address: vk::DeviceAddress,
    access: sync::BufferAccess,
}
pub(crate) struct ImageEntry {
    contents: Contents,
    native: Arc<AllocatedImage>,
    pub(crate) description: NativeImageDescription,
    pub(crate) planes: Vec<ImagePlane>,
    sample_grid: [u32; 2],
    views: Vec<ImageViewDescription>,
    native_views: Vec<Arc<crate::AllocatedImageView>>,
    access: Vec<sync::ImageAccess>,
    version: u64,
    producer: Option<WorkTicket>,
    leases: usize,
    discarded: Vec<(usize, vk::Rect2D)>,
    decoded: Option<decoded_texture::DecodedImage>,
    write_plans: Vec<(ImagePlane, vk::Rect2D, Box<footprint::ImageWritePlan>)>,
}
impl ImageEntry {
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
    pub cpu_stream_hits: u64,
    pub gpu_copied_bytes: u64,
    /// Exact bytes transferred to host memory, excluding GPU-only realignment.
    pub native_readback_bytes: u64,
    pub canonical_readback_bytes: u64,
    pub cache_hits: u64,
    /// Reused physical image-to-buffer copy plans, independent of pixel epochs.
    pub image_import_plan_hits: u64,
    pub evictions: u64,
}
/// Concrete single-owner storage, selected once at startup. No generic backend
/// handle registry, virtual resource dispatch or guest-ISA branch in its hot path.
pub struct VulkanResidency {
    pub(crate) device: VulkanDevice,
    instance: BackendInstanceId,
    buffers: Vec<Slot<BufferEntry>>,
    buffer_views: rustc_hash::FxHashMap<nixe_memory::CanonicalByteMappingKey, Vec<BufferId>>,
    free_buffers: Vec<usize>,
    images: Vec<Slot<ImageEntry>>,
    attachments: Vec<Slot<attachment::AttachmentEntry>>,
    index: CanonicalIndex,
    visibility: Arc<VulkanVisibility>,
    acknowledgements: mpsc::Receiver<CpuVisibilityRequest>,
    releases: mpsc::Receiver<PresentationRelease>,
    release_sender: mpsc::Sender<PresentationRelease>,
    budget: AllocationBudget,
    pending_writes: Vec<CanonicalBackingRange>,
    published_writes: usize,
    statistics: ResidencyStatistics,
    image_readbacks: readback::ImageReadbacks,
    barriers: sync::Barriers,
    decoder: nixe_gpu::texture_decode::BlockDecoder,
    sample_conversion: Option<sample_readback::SampleConversion>,
    #[cfg(feature = "binding-probe")]
    helpers: crate::helpers::Helpers,
    #[cfg(feature = "binding-probe")]
    descriptor_dispatch: Option<ash::ext::descriptor_buffer::Device>,
    barrier_policy: BarrierPolicy,
    unified_layouts: bool,
    /// Entries needed by a cold reconstruction cannot be evicted by its allocations.
    constructing: Vec<Resource>,
}
impl VulkanResidency {
    pub fn new(device: VulkanDevice, bytes: u64) -> Result<Self, VulkanError> {
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
        let (sender, acknowledgements) = mpsc::channel();
        let (release_sender, releases) = mpsc::channel();
        Ok(Self {
            device,
            instance,
            buffers: Vec::new(),
            buffer_views: Default::default(),
            free_buffers: Vec::new(),
            images: Vec::new(),
            attachments: Vec::new(),
            index: CanonicalIndex::default(),
            visibility: Arc::new(VulkanVisibility::new(sender)),
            acknowledgements,
            releases,
            release_sender,
            budget: AllocationBudget::new(bytes),
            pending_writes: Vec::new(),
            published_writes: 0,
            statistics: ResidencyStatistics::default(),
            image_readbacks: readback::ImageReadbacks::default(),
            barriers: sync::Barriers::default(),
            decoder: Default::default(),
            sample_conversion: None,
            #[cfg(feature = "binding-probe")]
            helpers: crate::helpers::Helpers::default(),
            #[cfg(feature = "binding-probe")]
            descriptor_dispatch,
            barrier_policy: BarrierPolicy::default(),
            unified_layouts,
            constructing: Vec::new(),
        })
    }
    pub fn device(&self) -> &VulkanDevice {
        &self.device
    }
    pub(crate) fn device_mut(&mut self) -> &mut VulkanDevice {
        &mut self.device
    }
    pub fn visibility(&self) -> Arc<VulkanVisibility> {
        Arc::clone(&self.visibility)
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
    fn gpu_owned(&self, page: CanonicalPageId, state: VisibilityState) -> bool {
        matches!(state, VisibilityState::GpuNewer { .. })
            || self.index.pages.get(&page).is_some_and(|page| {
                page.has_unsubmitted_write(self.device.queue_usage().submitted_tick)
            })
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
    fn touch(&mut self, resource: Resource) {
        // Batch age updates with real native use, not just cold creation order.
        let age = self.device.submission.submitted.saturating_add(1);
        self.contents_mut(resource).touched = age;
    }
    fn drain_notifications(&mut self) {
        while let Ok(ack) = self.acknowledgements.try_recv() {
            if let Some(page) = self.index.pages.get_mut(&ack.page) {
                page.acknowledge(ack.visible_at);
            }
        }
        while let Ok(release) = self.releases.try_recv() {
            if release.producer.instance() != self.instance {
                continue;
            }
            if let Some(slot) = self.images.get_mut(release.image.slot as usize)
                && slot.generation == release.image.generation
                && let Some(entry) = &mut slot.entry
                && entry.version == release.content_version
            {
                entry.leases = entry
                    .leases
                    .checked_sub(1)
                    .expect("one release per presentation lease");
            }
        }
    }
    fn authoritative(&self, resource: Resource) -> bool {
        self.index
            .pages
            .values()
            .any(|page| page.has_source(resource))
    }
    fn reserve(&mut self, bytes: u64) -> Result<Option<AllocationCharge>, VulkanError> {
        self.drain_notifications();
        if let Some(charge) = self.budget.reserve(bytes) {
            return Ok(Some(charge));
        }
        self.device.poll_progress()?;
        while self
            .budget
            .used()
            .checked_add(bytes)
            .is_none_or(|used| used > self.budget.limit)
        {
            if !self.evict_one()? {
                return Err(VulkanError::ResidencyPressure {
                    requested: bytes,
                    allocated: self.budget.used(),
                    budget: self.budget.limit,
                });
            }
        }
        Ok(self.budget.reserve(bytes))
    }
    pub fn evict_one(&mut self) -> Result<bool, VulkanError> {
        self.drain_notifications();
        let candidate = self
            .buffers
            .iter()
            .enumerate()
            .filter_map(|(slot, item)| {
                item.entry
                    .as_ref()
                    .filter(|e| Arc::strong_count(&e.native) == 1)
                    .map(|entry| {
                        (
                            Resource::Buffer(BufferId {
                                instance: self.instance,
                                slot,
                                generation: item.generation,
                            }),
                            &entry.contents,
                        )
                    })
            })
            .chain(self.images.iter().enumerate().filter_map(|(slot, item)| {
                item.entry
                    .as_ref()
                    .filter(|entry| {
                        entry.native_views.iter().all(|v| Arc::strong_count(v) == 1)
                            && Arc::strong_count(&entry.native) == 1 + entry.native_views.len()
                            && entry.leases == 0
                            && entry
                                .access
                                .iter()
                                .all(|cell| cell.ownership == ownership::Ownership::Local)
                    })
                    .map(|entry| {
                        (
                            Resource::Image(ImageId {
                                instance: self.instance,
                                slot,
                                generation: item.generation,
                            }),
                            &entry.contents,
                        )
                    })
            }))
            .filter(|(resource, _)| {
                !self.constructing.contains(resource) && !self.authoritative(*resource)
            })
            .min_by_key(|(_, contents)| contents.touched)
            .map(|(resource, _)| resource);
        let Some(resource) = candidate else {
            return Ok(false);
        };
        match resource {
            Resource::Buffer(id) => {
                let entry = self.buffers[id.slot].entry.take().expect("candidate");
                let key = entry.contents.backing.byte_mapping_key();
                let views = self
                    .buffer_views
                    .get_mut(&key)
                    .expect("indexed buffer view");
                views.retain(|view| *view != id);
                if views.is_empty() {
                    self.buffer_views.remove(&key);
                }
                if self.buffers[id.slot].generation < u32::MAX {
                    self.free_buffers.push(id.slot);
                }
                self.index.remove(resource, &entry.contents.backing);
                if let Some(last_use) = entry.contents.last_use {
                    self.device.retire_buffer(entry.native, last_use)?;
                }
            }
            Resource::Image(id) => {
                let entry = self.images[id.slot].entry.take().expect("candidate");
                self.index.remove(resource, &entry.contents.backing);
                if let Some(last_use) = entry.contents.last_use {
                    self.device
                        .retire_image_views(entry.native_views, last_use)?;
                    self.device.retire_image(entry.native, last_use)?;
                }
            }
        }
        self.statistics.evictions += 1;
        Ok(true)
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
    fn invalidate_aliases(
        &mut self,
        page: CanonicalPageId,
        start: u64,
        end: u64,
        producer: Resource,
        producer_offset: u64,
    ) {
        let bindings = self
            .index
            .pages
            .get(&page)
            .map(|page| {
                page.bindings()
                    .filter(|binding| binding.start < end && start < binding.end)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        for binding in bindings {
            let begin = start.max(binding.start);
            let finish = end.min(binding.end);
            if begin < finish
                && (binding.resource != producer
                    || binding.offset + begin - binding.start != producer_offset + begin - start)
            {
                let offset = binding.offset + begin - binding.start;
                let window = match binding.resource {
                    Resource::Image(id) => {
                        self.image_invalidation_window(id, offset, finish - begin)
                    }
                    Resource::Buffer(_) => (offset, finish - begin),
                };
                let contents = self.contents_mut(binding.resource);
                contents.invalidate_intervals(&[window]);
            }
        }
    }
    fn record_authority(
        &mut self,
        resource: Resource,
        offset: u64,
        size: u64,
    ) -> Result<(), VulkanError> {
        if let Resource::Image(id) = resource
            && self.image(id)?.decoded.is_some()
        {
            return Err(VulkanError::InvalidResource(
                "decoded compressed textures cannot author encoded canonical bytes",
            ));
        }
        let range = self
            .contents(resource)?
            .backing
            .snapshot_subrange(offset, size)
            .map_err(memory_error)?;
        let work = self.device.work_ticket()?;
        let point = DeviceVisibilityPoint::new(work.tick());
        let mut logical = offset;
        for segment in range.segments() {
            self.invalidate_aliases(
                segment.page(),
                segment.offset(),
                segment.offset() + segment.size(),
                resource,
                logical,
            );
            CanonicalIndex::replace(
                self.index
                    .pages
                    .get_mut(&segment.page())
                    .expect("indexed producer"),
                Authority {
                    start: segment.offset(),
                    end: segment.offset() + segment.size(),
                    source: resource,
                    offset: logical,
                    point,
                },
            );
            logical += segment.size();
        }
        self.pending_writes.push(range);
        self.contents_mut(resource).last_use = Some(work);
        self.contents_mut(resource).validate_interval(offset, size);
        Ok(())
    }
    fn prepare_write(
        &mut self,
        resource: Resource,
        offset: u64,
        size: u64,
    ) -> Result<(), VulkanError> {
        let range = self
            .contents(resource)?
            .backing
            .snapshot_subrange(offset, size)
            .map_err(memory_error)?;
        let tick = self.device.work_ticket()?.tick();
        CanonicalBackingRange::prepare_resident_device_accesses(
            [(
                &range,
                DeviceAccessDeclaration::write(
                    NonCpuDeviceId::new(self.instance.get()),
                    DeviceVisibilityPoint::new(tick),
                    DeviceVisibilityPoint::new(tick),
                )
                .map_err(memory_error)?,
            )],
            self.visibility.clone(),
        )
        .map_err(memory_error)
    }
    /// Records a real native transfer write; flush publishes canonical GPU
    /// ownership only after vkQueueSubmit2 has accepted this work.
    pub fn write_buffer(
        &mut self,
        id: BufferId,
        offset: u64,
        bytes: &[u8],
    ) -> Result<BufferUpload, VulkanError> {
        self.ensure_buffer(id)?;
        let consumed = bytes
            .len()
            .min(self.device.submission.configuration.arena_bytes as usize);
        let bytes = &bytes[..consumed];
        self.prepare_write(Resource::Buffer(id), offset, consumed as u64)?;
        self.upload_buffer(id, offset, bytes)?;
        self.record_authority(Resource::Buffer(id), offset, consumed as u64)?;
        Ok(BufferUpload {
            work: self.device.work_ticket()?,
            consumed,
        })
    }
    pub fn copy_buffer(
        &mut self,
        source: BufferId,
        source_offset: u64,
        destination: BufferId,
        destination_offset: u64,
        size: u64,
    ) -> Result<WorkTicket, VulkanError> {
        self.ensure_buffer(source)?;
        self.ensure_buffer(destination)?;
        self.prepare_write(Resource::Buffer(destination), destination_offset, size)?;
        self.copy_source(
            Resource::Buffer(source),
            source_offset,
            destination,
            destination_offset,
            size,
        )?;
        self.record_authority(Resource::Buffer(destination), destination_offset, size)?;
        self.device.work_ticket()
    }
    /// Accepts the recorded producer without forcing a queue submission.
    /// Canonical readers must already see this pending ownership: their
    /// visibility request flushes this owner and waits for the real timeline
    /// before any CPU bytes are published. Retain all writes until submission
    /// so a failed queue submission invalidates even previously accepted ones.
    pub(crate) fn accept_recording(&mut self) -> Result<(), VulkanError> {
        if self.published_writes == self.pending_writes.len() {
            return Ok(());
        }
        let tick = self.device.work_ticket()?.tick();
        let declaration = DeviceAccessDeclaration::write(
            NonCpuDeviceId::new(self.instance.get()),
            DeviceVisibilityPoint::new(tick),
            DeviceVisibilityPoint::new(tick),
        )
        .map_err(memory_error)?;
        CanonicalBackingRange::publish_device_writes(
            self.pending_writes[self.published_writes..]
                .iter()
                .map(|range| (range, declaration)),
            self.visibility.clone(),
        )
        .map_err(memory_error)?;
        self.published_writes = self.pending_writes.len();
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
        self.barriers.emit(&mut self.device);
        let work = match unsafe {
            self.device
                .submission
                .flush(self.device.queue, &self.device.dispatch, waits, signals)
        } {
            Ok(work) => work,
            Err(error) => {
                let _ =
                    CanonicalBackingRange::invalidate_visibility_ranges(self.pending_writes.iter());
                return Err(error);
            }
        };
        if let Some(work) = work
            && self.published_writes != self.pending_writes.len()
        {
            let declaration = DeviceAccessDeclaration::write(
                NonCpuDeviceId::new(self.instance.get()),
                DeviceVisibilityPoint::new(work.tick()),
                DeviceVisibilityPoint::new(work.tick()),
            )
            .map_err(memory_error)?;
            CanonicalBackingRange::publish_device_writes(
                self.pending_writes[self.published_writes..]
                    .iter()
                    .map(|range| (range, declaration)),
                self.visibility.clone(),
            )
            .map_err(memory_error)?;
        }
        if work.is_some() {
            self.pending_writes.clear();
            self.published_writes = 0;
        }
        Ok(work)
    }
    /// Resolves this owner inline. The memory coordinator validates the returned
    /// version and calls release_cpu_page only after canonical publication.
    pub fn materialize_cpu_page(
        &mut self,
        request: CpuVisibilityRequest,
    ) -> Result<Box<[u8]>, VulkanError> {
        if request.device != NonCpuDeviceId::new(self.instance.get()) {
            return Err(VulkanError::InvalidResource(
                "CPU visibility requested from foreign Vulkan owner",
            ));
        }
        self.flush()?;
        let mut bytes = {
            let pages = self.visibility.pages.lock().unwrap();
            let page = pages
                .get(&request.page)
                .ok_or(VulkanError::InvalidResource(
                    "CPU visibility has no preserved page baseline",
                ))?;
            if page.bytes.len() != request.size {
                return Err(VulkanError::InvalidResource(
                    "CPU visibility page size mismatch",
                ));
            }
            if page
                .completed
                .is_some_and(|point| point >= request.visible_at)
            {
                return Ok(page.bytes.clone());
            }
            page.bytes.clone()
        };
        let writes = self
            .index
            .pages
            .get(&request.page)
            .map(|page| page.resolved_writes())
            .unwrap_or_default();
        let mut completed = request.visible_at;
        self.read_authorities(&writes, &mut bytes)?;
        for write in writes {
            completed = completed.max(write.point);
        }
        self.visibility.pages.lock().unwrap().insert(
            request.page,
            visibility::CpuPage {
                bytes: bytes.clone(),
                completed: Some(completed),
            },
        );
        Ok(bytes)
    }
    /// Explicit preservation for allocation pressure, distinct from automatic
    /// eviction of reconstructible entries. Only this resource's canonical
    /// pages are published; native downloads still cover exact producer bytes.
    pub fn preserve_buffer(&mut self, id: BufferId) -> Result<(), VulkanError> {
        self.preserve(Resource::Buffer(id))
    }
    pub fn preserve_image(&mut self, id: ImageId) -> Result<(), VulkanError> {
        self.preserve(Resource::Image(id))
    }
    fn preserve(&mut self, resource: Resource) -> Result<(), VulkanError> {
        self.flush()?;
        let backing = self.contents(resource)?.backing.clone();
        let mut visited = std::collections::HashSet::new();
        let mut logical = 0;
        for segment in backing.segments() {
            if visited.insert(segment.page())
                && self
                    .index
                    .pages
                    .get(&segment.page())
                    .is_some_and(|page| page.has_source(resource))
            {
                let mut byte = [0];
                backing
                    .read_with_resolver(logical, &mut byte, &mut |_, request| {
                        self.materialize_cpu_page(request).map_err(|error| {
                            nixe_memory::VisibilityCoordinatorError::new(error.to_string())
                        })
                    })
                    .map_err(memory_error)?;
            }
            if matches!(
                segment.visibility_state(),
                VisibilityState::Clean | VisibilityState::CpuNewer
            ) {
                if let Some(page) = self.index.pages.get_mut(&segment.page()) {
                    page.clear_writes();
                }
                self.visibility
                    .pages
                    .lock()
                    .unwrap()
                    .remove(&segment.page());
            }
            logical += segment.size();
        }
        self.drain_notifications();
        Ok(())
    }
    /// Diagnostic/read consumers request only the selected resource interval.
    pub fn read_buffer(
        &mut self,
        id: BufferId,
        offset: u64,
        size: u64,
    ) -> Result<Vec<u8>, VulkanError> {
        self.ensure_buffer(id)?;
        self.flush()?;
        self.read_buffer_native(id, offset, size)
    }
}
impl Drop for VulkanResidency {
    fn drop(&mut self) {
        // Drain accepted native uses before entries destroy their objects. The
        // queue's fatal/device-lost teardown policy remains the single authority.
        if let Err(error) = self.device.shutdown()
            && !self.device.submission.is_device_lost()
        {
            log::error!("Cannot safely destroy Vulkan residency before queue completion: {error}");
            std::process::abort();
        }
        let device = NonCpuDeviceId::new(self.instance.get());
        let pages = self
            .buffers
            .iter()
            .filter_map(|slot| slot.entry.as_ref())
            .map(|entry| &entry.contents.backing)
            .chain(
                self.images
                    .iter()
                    .filter_map(|slot| slot.entry.as_ref())
                    .map(|entry| &entry.contents.backing),
            )
            .flat_map(CanonicalBackingRange::pages)
            .filter(|page| {
                matches!(page.visibility_state(), VisibilityState::GpuNewer { device: owner, .. } if owner == device)
            });
        let _ = CanonicalBackingRange::invalidate_visibility_pages(pages);
    }
}
fn same_bytes(a: &CanonicalBackingRange, b: &CanonicalBackingRange) -> bool {
    a.same_byte_mapping(b)
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
fn memory_error(error: impl std::fmt::Display) -> VulkanError {
    VulkanError::CanonicalMemory(error.to_string().into())
}

#[cfg(test)]
pub(crate) mod tests;

#[cfg(test)]
mod commands_tests;
