//! One timeline governs submission, transient reuse and deferred destruction.
use crate::{
    AllocatedBuffer, AllocatedImage, AllocationMode, MemoryLocation, VulkanDevice, VulkanError,
    device::DeviceOwner, memory::allocate_buffer,
};
use ash::vk;
pub(crate) mod progress;
#[cfg(feature = "performance-counters")]
mod timing;
use nixe_gpu::{BackendInstanceId, BackendSubmissionToken};
use std::{
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

#[derive(Clone, Copy, Debug)]
pub struct QueueConfiguration {
    pub command_slots: usize,
    /// Capacity of each upload/readback arena in a command slot. Allocated lazily.
    pub arena_bytes: u64,
    /// Additional staging capacity per slot, shared by uploads and readbacks.
    pub overflow_bytes: u64,
}
impl Default for QueueConfiguration {
    fn default() -> Self {
        Self {
            command_slots: 8,
            arena_bytes: 4 * 1024 * 1024,
            overflow_bytes: 16 * 1024 * 1024,
        }
    }
}
impl QueueConfiguration {
    pub(crate) fn validate(self) -> Result<(), VulkanError> {
        if self.command_slots == 0
            || self.arena_bytes == 0
            || self
                .arena_bytes
                .checked_mul(2)
                .and_then(|v| v.checked_add(self.overflow_bytes))
                .and_then(|v| v.checked_mul(self.command_slots as u64))
                .is_none()
        {
            return Err(VulkanError::InvalidResource(
                "invalid command/staging budget",
            ));
        }
        Ok(())
    }
}

/// Reserved host work, distinct from acceptance, completion or guest publication.
/// Slot identity is internal; the monotonically increasing tick never recycles.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WorkTicket {
    instance: BackendInstanceId,
    tick: u64,
}
impl WorkTicket {
    pub fn tick(self) -> u64 {
        self.tick
    }
    /// Identifies this host work. An execution owner can associate several
    /// distinct logical completion tokens with one batched host ticket.
    pub fn submission(self) -> BackendSubmissionToken {
        BackendSubmissionToken::new(self.instance, self.tick, 1)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorkStatus {
    Pending,
    Complete,
}
#[must_use = "consume or explicitly discard the reserved readback"]
#[derive(Debug)]
pub struct ReadbackTicket {
    pub(crate) work: WorkTicket,
    pub(crate) slot: usize,
    pub(crate) entry: usize,
}
impl ReadbackTicket {
    pub fn work(&self) -> WorkTicket {
        self.work
    }
}

#[derive(Debug)]
pub struct QueueUsage {
    pub scheduled_tick: u64,
    pub submitted_tick: u64,
    pub completed_tick: u64,
    pub command_slots: usize,
    pub arena_capacity: u64,
    pub overflow_capacity: u64,
    pub oversized_capacity: u64,
    pub retained_readbacks: usize,
    pub retired_buffers: usize,
    pub retired_images: usize,
}

pub(crate) struct Arena {
    pub buffer: Arc<AllocatedBuffer>,
    pub cursor: u64,
}
impl Arena {
    pub(crate) fn reserve_aligned(&mut self, size: u64, alignment: u64) -> Option<u64> {
        let start = align_up(self.cursor, self.buffer.host_alignment().max(alignment))?;
        let end = start.checked_add(size)?;
        if end > self.buffer.size() {
            return None;
        }
        // Atom isolation applies even before the next allocation is made.
        self.cursor = align_up(end, self.buffer.host_alignment())?;
        Some(start)
    }
}
pub(crate) fn align_up(value: u64, alignment: u64) -> Option<u64> {
    value
        .checked_add(alignment - 1)
        .map(|v| v & !(alignment - 1))
}

#[derive(Clone, Copy)]
pub(crate) struct Slice {
    pub overflow: Option<usize>,
    pub offset: u64,
    pub size: u64,
}
pub(crate) struct Slot {
    pub stream_descriptors: Option<crate::cache::DescriptorCache>,
    #[cfg(feature = "performance-counters")]
    timing: Option<timing::Queries>,
    #[cfg(feature = "binding-probe")]
    owner: Arc<DeviceOwner>,
    pub native: Arc<crate::dispatch::CommandSlot>,
    pub tick: u64,
    stream_read_until: u64,
    pending_stream_read: bool,
    pub upload: Option<Arena>,
    pub readback: Option<Arena>,
    overflow_capacity: u64,
    pub upload_overflow: Vec<Arena>,
    pub readback_overflow: Vec<Arena>,
    pub readbacks: Vec<Option<Slice>>,
    unread_count: usize,
    pub retired_buffers: Vec<Arc<AllocatedBuffer>>,
    pub retired_images: Vec<Arc<AllocatedImage>>,
    pub retained_views: Vec<Arc<crate::AllocatedImageView>>,
    pub native_pipelines: Vec<Arc<crate::cache::NativePipeline>>,
    pub native_bindings: Vec<Arc<crate::cache::BindingRecord>>,
    pub pipeline_handles: std::collections::HashSet<usize>,
    pub binding_handles: std::collections::HashSet<vk::DescriptorSet>,
    pub bound_graphics_set: vk::DescriptorSet,
    pub bound_compute_offsets: Vec<u32>,
    pub bound_graphics_offsets: Vec<u32>,
    pub bound_compute_layout: vk::PipelineLayout,
    pub bound_graphics_layout: vk::PipelineLayout,
    #[cfg(feature = "binding-probe")]
    pub probe_queries: Vec<Arc<crate::probe::ProbeQueries>>,
    #[cfg(feature = "binding-probe")]
    pub descriptors: crate::residency::probe_commands::Descriptors,
    pub bound_compute: usize,
    pub bound_graphics: usize,
    pub vertex_bindings: rustc_hash::FxHashMap<u32, (vk::Buffer, u64)>,
    pub index_binding: Option<(vk::Buffer, u64, vk::IndexType)>,
    pub push_layout: vk::PipelineLayout,
    pub push_stages: vk::ShaderStageFlags,
    pub push_bytes: Vec<u8>,
    pub native_targets: Vec<Arc<crate::PreparedTargets>>,
    pub active_targets: Option<Arc<crate::PreparedTargets>>,
    pub target_handles: std::collections::HashSet<usize>,
    pub graphics_parameters: Option<Arc<crate::GraphicsParameters>>,
    pub feedback_aspects: Option<vk::ImageAspectFlags>,
    pub bound_set: vk::DescriptorSet,
    #[cfg(feature = "binding-probe")]
    pub bound_descriptor_buffer: vk::DeviceAddress,
    #[cfg(feature = "binding-probe")]
    pub bound_descriptor_offset: u64,
}
impl Slot {
    fn new(owner: Arc<DeviceOwner>, family: u32) -> Result<Self, VulkanError> {
        let slot = Self {
            stream_descriptors: None,
            #[cfg(feature = "performance-counters")]
            timing: timing::Queries::new(Arc::clone(&owner), family)?,
            native: crate::dispatch::CommandSlot::new(Arc::clone(&owner), family),
            #[cfg(feature = "binding-probe")]
            owner,
            tick: 0,
            stream_read_until: 0,
            pending_stream_read: false,
            upload: None,
            readback: None,
            overflow_capacity: 0,
            upload_overflow: Vec::new(),
            readback_overflow: Vec::new(),
            readbacks: Vec::new(),
            unread_count: 0,
            retired_buffers: Vec::new(),
            retired_images: Vec::new(),
            retained_views: Vec::new(),
            native_pipelines: Vec::new(),
            native_bindings: Vec::new(),
            pipeline_handles: Default::default(),
            binding_handles: Default::default(),
            bound_graphics_set: vk::DescriptorSet::null(),
            bound_compute_offsets: Vec::new(),
            bound_graphics_offsets: Vec::new(),
            bound_compute_layout: vk::PipelineLayout::null(),
            bound_graphics_layout: vk::PipelineLayout::null(),
            #[cfg(feature = "binding-probe")]
            probe_queries: Vec::new(),
            #[cfg(feature = "binding-probe")]
            descriptors: crate::residency::probe_commands::Descriptors::new(),
            bound_compute: 0,
            bound_graphics: 0,
            vertex_bindings: rustc_hash::FxHashMap::default(),
            index_binding: None,
            push_layout: vk::PipelineLayout::null(),
            push_stages: vk::ShaderStageFlags::empty(),
            push_bytes: Vec::new(),
            native_targets: Vec::new(),
            active_targets: None,
            target_handles: Default::default(),
            graphics_parameters: None,
            feedback_aspects: None,
            bound_set: vk::DescriptorSet::null(),
            #[cfg(feature = "binding-probe")]
            bound_descriptor_buffer: 0,
            #[cfg(feature = "binding-probe")]
            bound_descriptor_offset: u64::MAX,
        };
        Ok(slot)
    }
    pub fn upload_buffer(&self, slice: Slice) -> &AllocatedBuffer {
        match slice.overflow {
            Some(i) => &self.upload_overflow[i].buffer,
            None => &self.upload.as_ref().expect("upload arena").buffer,
        }
    }
    pub(crate) fn upload_lease(&self, slice: Slice) -> Arc<AllocatedBuffer> {
        match slice.overflow {
            Some(index) => Arc::clone(&self.upload_overflow[index].buffer),
            None => Arc::clone(&self.upload.as_ref().expect("upload arena").buffer),
        }
    }
    pub fn readback_buffer(&self, slice: Slice) -> &AllocatedBuffer {
        match slice.overflow {
            Some(i) => &self.readback_overflow[i].buffer,
            None => &self.readback.as_ref().expect("readback arena").buffer,
        }
    }
    fn unread(&self) -> usize {
        self.unread_count
    }
    fn reusable(&self, completed: u64) -> bool {
        !self.pending_stream_read
            && self.tick.max(self.stream_read_until) <= completed
            && self.unread() == 0
    }
}
#[cfg(feature = "binding-probe")]
impl Drop for Slot {
    fn drop(&mut self) {
        // QueueState proves completion/device loss before slots are dropped.
        self.descriptors.destroy(&self.owner);
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RecordingStatistics {
    pub stream_descriptor_hits: u64,
    pub stream_descriptor_allocations: u64,
    pub pipeline_binds: u64,
    pub descriptor_binds: u64,
    pub vertex_binds: u64,
    pub index_binds: u64,
    pub push_updates: u64,
    pub graphics_updates: u64,
    pub rendering_scopes: u64,
    pub draw_calls: u64,
}
pub(crate) struct QueueState {
    recording_handle: crate::dispatch::RecordingHandle,
    pub(crate) progress: Arc<progress::SubmissionProgress>,
    pub statistics: RecordingStatistics,
    owner: Arc<DeviceOwner>,
    pub timeline: vk::Semaphore,
    instance: BackendInstanceId,
    family: u32,
    pub(crate) configuration: QueueConfiguration,
    pub slots: Vec<Slot>,
    pub recording: Option<usize>,
    pending_submit_deadline: Option<Instant>,
    pub scheduled: u64,
    pub completed: u64,
    pending_stream_reads: Vec<usize>,
    stopped: bool,
    lost: bool,
}
impl QueueState {
    pub(crate) fn end_rendering(&mut self, dispatch: &mut crate::dispatch::Dispatch) {
        if let Some(index) = self.recording
            && self.slots[index].active_targets.take().is_some()
        {
            unsafe { dispatch.end_rendering() };
        }
    }
    pub fn new(
        owner: Arc<DeviceOwner>,
        family: u32,
        configuration: QueueConfiguration,
        _dispatch: &crate::dispatch::Dispatch,
    ) -> Result<Self, VulkanError> {
        static NEXT_INSTANCE: AtomicU64 = AtomicU64::new(1);
        let instance = NEXT_INSTANCE
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| v.checked_add(1))
            .map_err(|_| VulkanError::InvalidResource("Vulkan instance identity exhausted"))?;
        let mut kind = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let info = vk::SemaphoreCreateInfo::default().push_next(&mut kind);
        let timeline = unsafe { owner.device.create_semaphore(&info, None) }
            .map_err(|e| VulkanError::vk("vkCreateSemaphore(owner timeline)", e))?;
        let state = Self {
            recording_handle: _dispatch.recording_handle(),
            progress: Arc::new(progress::SubmissionProgress::default()),
            owner,
            timeline,
            instance: BackendInstanceId::new(instance),
            family,
            configuration,
            slots: Vec::new(),
            recording: None,
            pending_submit_deadline: None,
            statistics: Default::default(),
            scheduled: 0,
            completed: 0,
            pending_stream_reads: Vec::new(),
            stopped: false,
            lost: false,
        };
        state.owner.name(timeline, "nixe submission timeline")?;
        Ok(state)
    }
    pub(crate) fn is_device_lost(&self) -> bool {
        self.lost
            || self.recording_handle.is_device_lost()
            || self
                .progress
                .failure()
                .is_some_and(|f| f.cause.device_lost())
    }
    pub(crate) fn ticket(&self, tick: u64) -> WorkTicket {
        WorkTicket {
            instance: self.instance,
            tick,
        }
    }
    fn check_running(&self) -> Result<(), VulkanError> {
        if self.stopped {
            return Err(VulkanError::QueueStopped);
        }
        if let Some(failure) = self.progress.failure() {
            return Err(failure.error());
        }
        self.recording_handle.check_failure()
    }
    fn error(&mut self, operation: &'static str, result: vk::Result) -> VulkanError {
        self.stopped = true;
        self.lost |= result == vk::Result::ERROR_DEVICE_LOST;
        self.progress.fail(
            self.completed + 1,
            progress::NativeFailure::Vulkan { operation, result },
        );
        self.progress.failure().unwrap().error()
    }
    fn check_ticket(&self, ticket: WorkTicket) -> Result<(), VulkanError> {
        let last = self.scheduled + u64::from(self.recording.is_some());
        if ticket.instance != self.instance || ticket.tick == 0 || ticket.tick > last {
            return Err(VulkanError::InvalidResource(
                "foreign or unissued queue ticket",
            ));
        }
        Ok(())
    }
    pub fn poll(&mut self) -> Result<u64, VulkanError> {
        self.refresh_completed()?;
        self.check_running()?;
        Ok(self.completed)
    }
    fn refresh_completed(&mut self) -> Result<u64, VulkanError> {
        let submitted = self.progress.submitted();
        if self.completed >= submitted {
            return Ok(self.completed);
        }
        let completed = unsafe { self.owner.device.get_semaphore_counter_value(self.timeline) }
            .map_err(|e| self.error("vkGetSemaphoreCounterValue", e))?;
        if completed.min(submitted) > self.completed {
            #[cfg(feature = "performance-counters")]
            self.retire_timings(completed.min(submitted))?;
            self.progress.observe_completed(completed.min(submitted));
            self.reclaim(completed.min(submitted));
        }
        Ok(self.completed)
    }
    fn poll_work(&mut self, ticket: WorkTicket) -> Result<WorkStatus, VulkanError> {
        self.check_ticket(ticket)?;
        if ticket.tick <= self.completed {
            return Ok(WorkStatus::Complete);
        }
        self.progress.check(ticket.tick)?;
        if self.stopped {
            return Err(VulkanError::QueueStopped);
        }
        if ticket.tick > self.progress.submitted() {
            return Ok(WorkStatus::Pending);
        }
        self.refresh_completed()?;
        Ok(if ticket.tick <= self.completed {
            WorkStatus::Complete
        } else {
            WorkStatus::Pending
        })
    }
    #[cfg(feature = "performance-counters")]
    fn retire_timings(&mut self, completed: u64) -> Result<(), VulkanError> {
        for slot in &mut self.slots {
            if slot.tick <= completed
                && let Some(timing) = &mut slot.timing
            {
                timing.retire()?;
            }
        }
        Ok(())
    }
    fn reclaim(&mut self, completed: u64) {
        if completed > self.completed {
            nixe_gpu::trace::detail_event("queue_complete", completed, self.completed, 0);
        }
        self.completed = self.completed.max(completed);
        for slot in &mut self.slots {
            if !slot.pending_stream_read && slot.tick.max(slot.stream_read_until) <= self.completed
            {
                slot.retired_buffers.clear();
                slot.retired_images.clear();
                slot.retained_views.clear();
                slot.graphics_parameters = None;
                slot.feedback_aspects = None;
                slot.native_targets.clear();
                slot.target_handles.clear();
                slot.native_bindings.clear();
                slot.native_pipelines.clear();
                slot.pipeline_handles.clear();
                slot.binding_handles.clear();
                #[cfg(feature = "binding-probe")]
                slot.probe_queries.clear();
                slot.overflow_capacity -= slot
                    .upload_overflow
                    .iter()
                    .filter(|arena| arena.buffer.size() <= self.configuration.overflow_bytes)
                    .map(|arena| arena.buffer.size())
                    .sum::<u64>();
                slot.upload_overflow.clear();
                if slot.unread() == 0 {
                    slot.overflow_capacity -= slot
                        .readback_overflow
                        .iter()
                        .filter(|arena| arena.buffer.size() <= self.configuration.overflow_bytes)
                        .map(|arena| arena.buffer.size())
                        .sum::<u64>();
                    slot.readback_overflow.clear();
                }
            }
        }
    }
    /// A transfer may submit while a draw is still preparing its inputs. Keep
    /// its mapped streams alive until the command that actually reads them,
    /// rather than recycling them at the earlier upload submission's signal.
    pub(crate) fn retain_stream_upload(&mut self, slot: usize) {
        if !self.slots[slot].pending_stream_read {
            self.slots[slot].pending_stream_read = true;
            self.pending_stream_reads.push(slot);
        }
    }
    pub(crate) fn release_stream_uploads(&mut self) {
        let consumer = self.scheduled + u64::from(self.recording.is_some());
        let mut reclaim = false;
        for index in self.pending_stream_reads.drain(..) {
            let slot = &mut self.slots[index];
            slot.stream_read_until = slot.stream_read_until.max(consumer);
            slot.pending_stream_read = false;
            reclaim |= slot.tick.max(slot.stream_read_until) <= self.completed;
        }
        if reclaim {
            self.reclaim(self.completed);
        }
    }
    fn pressure(&self) -> VulkanError {
        VulkanError::QueuePressure {
            pending: self
                .slots
                .iter()
                .map(|slot| slot.tick.max(slot.stream_read_until))
                .filter(|&tick| tick > self.completed && tick <= self.scheduled)
                .min()
                .map(|tick| self.ticket(tick)),
            retained_readbacks: self.slots.iter().map(Slot::unread).sum(),
        }
    }
    pub fn begin(
        &mut self,
        dispatch: &mut crate::dispatch::Dispatch,
    ) -> Result<usize, VulkanError> {
        self.check_running()?;
        if let Some(index) = self.recording {
            return Ok(index);
        }
        // Poll only at batch acquisition/pressure, never per draw or arena slice.
        if !self.slots.iter().any(|slot| slot.reusable(self.completed)) {
            self.poll()?;
        }
        let index = match self
            .slots
            .iter()
            .position(|slot| slot.reusable(self.completed))
        {
            Some(index) => index,
            None if self.slots.len() < self.configuration.command_slots => {
                self.slots
                    .push(Slot::new(Arc::clone(&self.owner), self.family)?);
                self.slots.len() - 1
            }
            None => return Err(self.pressure()),
        };
        let tick = self
            .scheduled
            .checked_add(1)
            .ok_or(VulkanError::InvalidResource(
                "submission timeline exhausted",
            ))?;
        let slot = &mut self.slots[index];
        // The slot's previous GPU tick is complete. Reset/begin remain ordered
        // native jobs; no owner/worker round trip is needed to acquire a slot.
        let deadline = Instant::now() + Duration::from_millis(2);
        dispatch.begin_command_buffer(
            Arc::clone(&slot.native),
            tick,
            Arc::clone(&self.progress),
        )?;
        let slot = &mut self.slots[index];
        #[cfg(feature = "performance-counters")]
        if let Some(timing) = &mut slot.timing {
            timing.begin(dispatch);
        }
        slot.tick = tick;
        if let Some(arena) = &mut slot.upload {
            arena.cursor = 0;
        }
        if let Some(arena) = &mut slot.readback {
            arena.cursor = 0;
        }
        #[cfg(feature = "binding-probe")]
        slot.descriptors.reset(&self.owner)?;
        slot.bound_compute = 0;
        slot.bound_graphics = 0;
        slot.vertex_bindings.clear();
        slot.index_binding = None;
        slot.push_layout = vk::PipelineLayout::null();
        slot.push_stages = vk::ShaderStageFlags::empty();
        slot.push_bytes.clear();
        slot.graphics_parameters = None;
        slot.feedback_aspects = None;
        slot.bound_set = vk::DescriptorSet::null();
        slot.bound_graphics_set = vk::DescriptorSet::null();
        slot.bound_compute_offsets.clear();
        slot.bound_graphics_offsets.clear();
        slot.bound_compute_layout = vk::PipelineLayout::null();
        slot.bound_graphics_layout = vk::PipelineLayout::null();
        #[cfg(feature = "binding-probe")]
        {
            slot.bound_descriptor_buffer = 0;
            slot.bound_descriptor_offset = u64::MAX;
        }
        slot.readbacks.clear();
        // Immutable descriptors remain valid across command-buffer reuse.
        // Their bounded cache recycles unleased pages on pressure; clearing it
        // here would discard unchanged streaming bindings on every submission.
        self.recording = Some(index);
        self.pending_submit_deadline = Some(deadline);
        Ok(index)
    }
    pub fn reserve_upload(
        &mut self,
        dispatch: &mut crate::dispatch::Dispatch,
        bytes: &[u8],
    ) -> Result<(usize, Slice), VulkanError> {
        self.reserve_upload_aligned(dispatch, bytes, 4)
    }
    pub(crate) fn reserve_upload_aligned(
        &mut self,
        dispatch: &mut crate::dispatch::Dispatch,
        bytes: &[u8],
        alignment: u64,
    ) -> Result<(usize, Slice), VulkanError> {
        let (index, slice) = self.reserve_slice(dispatch, bytes.len() as u64, true, alignment)?;
        let slot = &mut self.slots[index];
        let buffer = match slice.overflow {
            Some(i) => &mut slot.upload_overflow[i].buffer,
            None => &mut slot.upload.as_mut().expect("upload arena").buffer,
        };
        unsafe { buffer.write_range(slice.offset, bytes) }?;
        Ok((index, slice))
    }
    pub fn reserve_readback(
        &mut self,
        dispatch: &mut crate::dispatch::Dispatch,
        size: u64,
    ) -> Result<(usize, Slice, ReadbackTicket), VulkanError> {
        self.reserve_readback_aligned(dispatch, size, 4)
    }
    pub fn reserve_readback_aligned(
        &mut self,
        dispatch: &mut crate::dispatch::Dispatch,
        size: u64,
        alignment: u64,
    ) -> Result<(usize, Slice, ReadbackTicket), VulkanError> {
        if !alignment.is_power_of_two() {
            return Err(VulkanError::InvalidResource(
                "staging alignment must be a nonzero power of two",
            ));
        }
        let (index, slice) = self.reserve_slice(dispatch, size, false, alignment)?;
        let slot = &mut self.slots[index];
        let entry = slot.readbacks.len();
        slot.readbacks.push(Some(slice));
        slot.unread_count += 1;
        Ok((
            index,
            slice,
            ReadbackTicket {
                work: self.ticket(self.scheduled + 1),
                slot: index,
                entry,
            },
        ))
    }
    pub(crate) fn reserve_slice(
        &mut self,
        dispatch: &mut crate::dispatch::Dispatch,
        size: u64,
        upload: bool,
        alignment: u64,
    ) -> Result<(usize, Slice), VulkanError> {
        if size == 0 {
            return Err(VulkanError::InvalidResource("empty staging transfer"));
        }
        let minimum = align_up(size, self.owner.non_coherent_atom_size.max(4))
            .ok_or(VulkanError::InvalidResource("staging capacity overflow"))?;
        let oversized = minimum
            > self
                .configuration
                .arena_bytes
                .max(self.configuration.overflow_bytes);
        let index = self.begin(dispatch)?;
        let (location, usage, label) = if upload {
            (
                MemoryLocation::CpuToGpu,
                vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::VERTEX_BUFFER
                    | vk::BufferUsageFlags::INDEX_BUFFER
                    | vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::UNIFORM_BUFFER
                    | vk::BufferUsageFlags::UNIFORM_TEXEL_BUFFER
                    | vk::BufferUsageFlags::STORAGE_TEXEL_BUFFER,
                "nixe upload arena",
            )
        } else {
            (
                MemoryLocation::GpuToCpu,
                vk::BufferUsageFlags::TRANSFER_DST,
                "nixe readback arena",
            )
        };
        let slot = &mut self.slots[index];
        let arena = if upload {
            &mut slot.upload
        } else {
            &mut slot.readback
        };
        if !oversized && minimum <= self.configuration.arena_bytes && arena.is_none() {
            *arena = Some(Arena {
                buffer: Arc::new(allocate_buffer(
                    &self.owner,
                    self.configuration.arena_bytes,
                    usage,
                    location,
                    AllocationMode::Dedicated,
                    label,
                )?),
                cursor: 0,
            });
        }
        if !oversized
            && let Some(offset) = arena
                .as_mut()
                .and_then(|arena| arena.reserve_aligned(size, alignment))
        {
            return Ok((
                index,
                Slice {
                    overflow: None,
                    offset,
                    size,
                },
            ));
        }
        let overflow = if upload {
            &mut slot.upload_overflow
        } else {
            &mut slot.readback_overflow
        };
        // Suballocate the current overflow arena as well. Small texture rows
        // must not create one VkBuffer/VkDeviceMemory allocation per request.
        if !oversized
            && let Some(arena) = overflow.last_mut()
            && let Some(offset) = arena.reserve_aligned(size, alignment)
        {
            return Ok((
                index,
                Slice {
                    overflow: Some(overflow.len() - 1),
                    offset,
                    size,
                },
            ));
        }
        let capacity = if oversized {
            minimum
        } else {
            let remaining = self.configuration.overflow_bytes - slot.overflow_capacity;
            if minimum > remaining {
                return Err(self.pressure());
            }
            minimum.max(self.configuration.arena_bytes).min(remaining)
        };
        let buffer = allocate_buffer(
            &self.owner,
            capacity,
            usage,
            location,
            AllocationMode::Dedicated,
            label,
        )?;
        let slot = &mut self.slots[index];
        let overflow = if upload {
            &mut slot.upload_overflow
        } else {
            &mut slot.readback_overflow
        };
        let entry = overflow.len();
        overflow.push(Arena {
            buffer: Arc::new(buffer),
            cursor: minimum,
        });
        if !oversized {
            slot.overflow_capacity += capacity;
        }
        Ok((
            index,
            Slice {
                overflow: Some(entry),
                offset: 0,
                size,
            },
        ))
    }
    pub unsafe fn flush(
        &mut self,
        queue: vk::Queue,
        dispatch: &mut crate::dispatch::Dispatch,
        waits: &[vk::SemaphoreSubmitInfo<'_>],
        extra_signals: &[vk::SemaphoreSubmitInfo<'_>],
    ) -> Result<Option<WorkTicket>, VulkanError> {
        self.check_running()?;
        let Some(index) = self.recording else {
            return Ok(None);
        };
        if waits
            .iter()
            .chain(extra_signals)
            .any(|info| !info.p_next.is_null())
        {
            return Err(VulkanError::InvalidResource(
                "queue semaphore extensions require owned native parameters",
            ));
        }
        self.end_rendering(dispatch);
        let slot = &self.slots[index];
        let tick = slot.tick;
        #[cfg(feature = "performance-counters")]
        {
            let slot = &mut self.slots[index];
            if let Some(timing) = &mut slot.timing {
                timing.end(dispatch);
            }
        }
        if extra_signals.len() > 1 {
            return Err(VulkanError::InvalidResource(
                "owner submission supports one presentation signal",
            ));
        }
        let signals = [
            vk::SemaphoreSubmitInfo::default()
                .semaphore(self.timeline)
                .value(tick)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
            extra_signals.first().copied().unwrap_or_default(),
        ];
        // Capture native parameters now. The worker closes and submits this
        // command buffer in FIFO order while the owner prepares following work.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/vk_scheduler.cpp
        nixe_gpu::trace::detail_event("queue_schedule", tick, 0, 0);
        let result = dispatch.submit(crate::dispatch::QueueSubmission {
            queue,
            waits: waits
                .iter()
                .map(|info| {
                    vk::SemaphoreSubmitInfo::default()
                        .semaphore(info.semaphore)
                        .value(info.value)
                        .stage_mask(info.stage_mask)
                        .device_index(info.device_index)
                })
                .collect(),
            signals: signals[..1 + extra_signals.len()]
                .iter()
                .map(|info| {
                    vk::SemaphoreSubmitInfo::default()
                        .semaphore(info.semaphore)
                        .value(info.value)
                        .stage_mask(info.stage_mask)
                        .device_index(info.device_index)
                })
                .collect(),
            tick,
            progress: Arc::clone(&self.progress),
        });
        if result.is_err() {
            self.progress.fail(
                tick,
                self.recording_handle
                    .native_failure()
                    .unwrap_or(progress::NativeFailure::Stopped),
            );
            return Err(self.progress.failure().unwrap().error());
        }
        self.scheduled = tick;
        self.recording = None;
        self.pending_submit_deadline = None;
        Ok(Some(self.ticket(tick)))
    }
    fn wait_for_completion(
        &mut self,
        ticket: WorkTicket,
        timeout: Duration,
    ) -> Result<bool, VulkanError> {
        let deadline = Instant::now().checked_add(timeout);
        let _trace = nixe_gpu::trace::span(
            "timeline_wait",
            ticket.tick,
            timeout.as_nanos().min(u128::from(u64::MAX)) as u64,
            self.completed,
        );
        self.check_ticket(ticket)?;
        if timeout.is_zero() {
            return Ok(self.poll_work(ticket)? == WorkStatus::Complete);
        }
        if ticket.tick > self.scheduled {
            return Err(VulkanError::InvalidResource(
                "wait target is still recording; flush it before waiting",
            ));
        }
        if ticket.tick <= self.completed {
            return Ok(true);
        }
        self.progress.check(ticket.tick)?;
        if !self.progress.wait_submitted(ticket.tick, deadline)? {
            return Ok(false);
        }
        let semaphores = [self.timeline];
        let values = [ticket.tick];
        let info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        // A timeout never establishes completion or releases any in-flight use.
        // https://docs.vulkan.org/refpages/latest/refpages/source/vkWaitSemaphores.html
        #[cfg(feature = "performance-counters")]
        let waited_at = std::time::Instant::now();
        let result = loop {
            self.progress.check(ticket.tick)?;
            let remaining = deadline.map_or(Duration::MAX, |d| {
                d.saturating_duration_since(Instant::now())
            });
            let slice = remaining.min(Duration::from_millis(50));
            let result = unsafe {
                self.owner
                    .device
                    .wait_semaphores(&info, slice.as_nanos() as u64)
            };
            if result != Err(vk::Result::TIMEOUT) || remaining <= slice {
                break result;
            }
        };
        #[cfg(feature = "performance-counters")]
        {
            nixe_gpu::metrics::record(nixe_gpu::metrics::Counter::HostQueueWaits, 1);
            nixe_gpu::metrics::record(
                nixe_gpu::metrics::Counter::HostQueueWaitNanoseconds,
                waited_at.elapsed().as_nanos() as u64,
            );
        }
        match result {
            Ok(()) => {
                #[cfg(feature = "performance-counters")]
                self.retire_timings(ticket.tick)?;
                self.progress.observe_completed(ticket.tick);
                self.reclaim(ticket.tick);
                Ok(true)
            }
            Err(vk::Result::TIMEOUT) => Ok(false),
            Err(e) => Err(self.error("vkWaitSemaphores", e)),
        }
    }
    pub fn usage(&self) -> QueueUsage {
        QueueUsage {
            scheduled_tick: self.scheduled,
            submitted_tick: self.progress.submitted(),
            completed_tick: self.completed,
            command_slots: self.slots.len(),
            arena_capacity: self
                .slots
                .iter()
                .flat_map(|slot| [slot.upload.as_ref(), slot.readback.as_ref()])
                .flatten()
                .map(|arena| arena.buffer.size())
                .sum(),
            overflow_capacity: self.slots.iter().map(|slot| slot.overflow_capacity).sum(),
            oversized_capacity: self
                .slots
                .iter()
                .flat_map(|slot| slot.upload_overflow.iter().chain(&slot.readback_overflow))
                .filter(|arena| arena.buffer.size() > self.configuration.overflow_bytes)
                .map(|arena| arena.buffer.size())
                .sum(),
            retained_readbacks: self.slots.iter().map(Slot::unread).sum(),
            retired_buffers: self
                .slots
                .iter()
                .map(|slot| slot.retired_buffers.len())
                .sum(),
            retired_images: self
                .slots
                .iter()
                .map(|slot| slot.retired_images.len())
                .sum(),
        }
    }
}
impl QueueState {
    fn stop(&mut self) -> Result<(), VulkanError> {
        if self.stopped && self.slots.is_empty() {
            return Ok(());
        }
        let prior_failure = self.progress.failure();
        self.stopped = true;
        self.progress.cancel_pending();
        // Shutdown is the explicit worker-drain boundary. Canceled jobs release
        // their parameters; a native call already in progress may still succeed.
        let drain = self.recording_handle.finish();
        self.lost |= self.is_device_lost();
        let submitted = self.progress.submitted();
        if !self.lost && self.completed < submitted {
            let semaphores = [self.timeline];
            let values = [submitted];
            let info = vk::SemaphoreWaitInfo::default()
                .semaphores(&semaphores)
                .values(&values);
            match unsafe { self.owner.device.wait_semaphores(&info, u64::MAX) } {
                Ok(()) => {
                    self.progress.observe_completed(submitted);
                    self.reclaim(submitted);
                }
                Err(vk::Result::ERROR_DEVICE_LOST) => self.lost = true,
                Err(error) => return Err(self.error("shutdown timeline wait", error)),
            }
        }
        self.recording = None;
        self.pending_submit_deadline = None;
        self.slots.clear();
        self.pending_stream_reads.clear();
        if let Some(failure) = prior_failure.or_else(|| self.progress.failure())
            && !matches!(failure.cause, progress::NativeFailure::Stopped)
        {
            return Err(failure.error());
        }
        match drain {
            Err(_)
                if self.progress.cancelled()
                    && self
                        .recording_handle
                        .native_failure()
                        .is_some_and(|failure| {
                            matches!(failure, progress::NativeFailure::Stopped)
                        }) =>
            {
                Ok(())
            }
            other => other,
        }
    }
}
impl Drop for QueueState {
    fn drop(&mut self) {
        let result = self.stop();
        if !self.slots.is_empty() {
            // Never destroy handles whose actual native use remains unproven.
            log::error!(
                "cannot drain Vulkan queue during destruction: {:?}",
                result.err()
            );
            std::process::abort();
        }
        unsafe { self.owner.device.destroy_semaphore(self.timeline, None) };
    }
}

impl VulkanDevice {
    /// Ordering queries must not assemble diagnostic resource-usage totals.
    #[inline]
    pub(crate) fn submitted_tick(&self) -> u64 {
        self.submission.progress.submitted()
    }
    #[inline]
    pub(crate) fn scheduled_tick(&self) -> u64 {
        self.submission.scheduled
    }
    #[inline]
    pub(crate) fn completed_tick(&self) -> u64 {
        self.submission.completed
    }
    pub(crate) fn preceding_work(&self) -> Option<WorkTicket> {
        let tick = self
            .submission
            .recording
            .map_or(self.submission.scheduled, |slot| {
                self.submission.slots[slot].tick
            });
        (tick != 0).then(|| self.submission.ticket(tick))
    }
    pub(crate) fn submission_instance(&self) -> BackendInstanceId {
        self.submission.instance
    }
    pub(crate) fn end_rendering(&mut self) {
        self.submission.end_rendering(&mut self.dispatch);
    }
    /// Acquire recording capacity with bounded storage. The queue's low-level
    /// begin remains nonblocking; native recording waits only for a submitted
    /// use that can free a slot. Unconsumed readbacks still report pressure.
    pub(crate) fn begin_recording(&mut self) -> Result<usize, VulkanError> {
        loop {
            match self.submission.begin(&mut self.dispatch) {
                Ok(index) => return Ok(index),
                Err(VulkanError::QueuePressure {
                    pending: Some(work),
                    ..
                }) => {
                    let _trace = nixe_gpu::trace::span("resource_pressure_wait", work.tick(), 0, 0);
                    self.wait_for_work(work, Duration::MAX)?;
                }
                Err(error) => return Err(error),
            }
        }
    }

    /// Reserve an ordered completion point, including a real signal for work
    /// with no commands. A flush accepts it; host polling alone is not guest success.
    pub fn work_ticket(&mut self) -> Result<WorkTicket, VulkanError> {
        self.begin_recording()?;
        Ok(self.submission.ticket(self.submission.scheduled + 1))
    }
    pub fn pending_submit_deadline(&self) -> Option<Instant> {
        self.submission.pending_submit_deadline
    }
    /// Submit recorded commands and signal the single timeline, without waiting.
    pub fn flush(&mut self) -> Result<Option<WorkTicket>, VulkanError> {
        unsafe {
            self.submission
                .flush(self.queue, &mut self.dispatch, &[], &[])
        }
    }
    pub fn poll_progress(&mut self) -> Result<u64, VulkanError> {
        self.submission.poll()
    }
    /// Inspect one ticket without flushing or waiting for recording/submission.
    pub fn poll_work(&mut self, ticket: WorkTicket) -> Result<WorkStatus, VulkanError> {
        self.submission.poll_work(ticket)
    }
    /// Wait only for sealed work. Call flush explicitly before waiting on a
    /// recording ticket; the timeout covers acknowledgement and GPU completion.
    pub fn wait_for_work(
        &mut self,
        ticket: WorkTicket,
        timeout: Duration,
    ) -> Result<bool, VulkanError> {
        self.submission.wait_for_completion(ticket, timeout)
    }
    pub fn recording_statistics(&self) -> RecordingStatistics {
        self.submission.statistics
    }
    pub fn queue_usage(&self) -> QueueUsage {
        self.submission.usage()
    }
    /// Read exactly the reserved bytes after completion; unconsumed readbacks
    /// pin their arena even when the GPU has finished. No stale range can recycle.
    pub fn readback(&mut self, ticket: ReadbackTicket) -> Result<Vec<u8>, VulkanError> {
        self.validate_readback(&ticket)?;
        if ticket.work.tick > self.submission.scheduled {
            self.flush()?;
        }
        self.wait_for_work(ticket.work, Duration::MAX)?;
        let slot = &mut self.submission.slots[ticket.slot];
        let slice = slot.readbacks[ticket.entry]
            .take()
            .expect("validated readback");
        slot.unread_count -= 1;
        let result = unsafe {
            slot.readback_buffer(slice)
                .read_range(slice.offset, slice.size)
        };
        self.submission.reclaim(self.submission.completed);
        result
    }
    pub fn discard_readback(&mut self, ticket: ReadbackTicket) -> Result<(), VulkanError> {
        self.validate_readback(&ticket)?;
        self.submission.slots[ticket.slot].readbacks[ticket.entry] = None;
        self.submission.slots[ticket.slot].unread_count -= 1;
        self.submission.reclaim(self.submission.completed);
        Ok(())
    }
    fn validate_readback(&self, ticket: &ReadbackTicket) -> Result<(), VulkanError> {
        self.submission.check_ticket(ticket.work)?;
        if self.submission.slots.get(ticket.slot).is_none_or(|slot| {
            slot.tick != ticket.work.tick
                || slot.readbacks.get(ticket.entry).is_none_or(Option::is_none)
        }) {
            return Err(VulkanError::InvalidResource(
                "stale or consumed readback ticket",
            ));
        }
        Ok(())
    }
    pub fn retire_buffer(
        &mut self,
        buffer: impl Into<Arc<AllocatedBuffer>>,
        last_use: WorkTicket,
    ) -> Result<(), VulkanError> {
        let buffer = buffer.into();
        self.submission.check_ticket(last_use)?;
        if !Arc::ptr_eq(&buffer.owner, &self.owner) {
            return Err(VulkanError::InvalidResource("foreign buffer retirement"));
        }
        if last_use.tick > self.submission.completed {
            let slot = self
                .submission
                .slots
                .iter_mut()
                .find(|slot| slot.tick == last_use.tick)
                .expect("pending use retains command slot");
            slot.retired_buffers.push(buffer);
        }
        Ok(())
    }
    pub(crate) fn retire_image(
        &mut self,
        image: impl Into<Arc<AllocatedImage>>,
        last_use: WorkTicket,
    ) -> Result<(), VulkanError> {
        let image = image.into();
        self.submission.check_ticket(last_use)?;
        if !Arc::ptr_eq(&image.owner, &self.owner) {
            return Err(VulkanError::InvalidResource("foreign image retirement"));
        }
        if last_use.tick > self.submission.completed {
            self.submission
                .slots
                .iter_mut()
                .find(|slot| slot.tick == last_use.tick)
                .expect("pending use retains command slot")
                .retired_images
                .push(image);
        }
        Ok(())
    }
    pub(crate) fn retire_image_views(
        &mut self,
        views: Vec<Arc<crate::AllocatedImageView>>,
        last_use: WorkTicket,
    ) -> Result<(), VulkanError> {
        self.submission.check_ticket(last_use)?;
        if last_use.tick > self.submission.completed {
            self.submission
                .slots
                .iter_mut()
                .find(|s| s.tick == last_use.tick)
                .expect("pending image view slot")
                .retained_views
                .extend(views);
        }
        Ok(())
    }
    /// Cancel unsubmitted recording and drain accepted work, without publishing
    /// canceled guest reservations. Dropping the device uses the same last tick.
    pub fn shutdown(&mut self) -> Result<(), VulkanError> {
        self.dispatch.discard_pending();
        self.submission.stop()
    }
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

#[cfg(test)]
mod tests;
