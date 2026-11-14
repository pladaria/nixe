//! One timeline governs submission, transient reuse and deferred destruction.
use crate::{
    AllocatedBuffer, AllocatedImage, AllocationMode, MemoryLocation, VulkanDevice, VulkanError,
    device::DeviceOwner, memory::allocate_buffer,
};
use ash::vk;
pub(crate) mod progress;
pub(crate) mod staging;
pub(crate) use staging::Slice;
mod frame_metrics;
mod queries;
mod timing;
use nixe_gpu::{BackendInstanceId, BackendSubmissionToken};
use std::{
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

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
pub struct ReadbackTicket {
    pub(crate) work: WorkTicket,
    pub(crate) slice: Slice,
    allocation: staging::ReadbackAllocation,
    progress: Arc<progress::SubmissionProgress>,
}
impl std::fmt::Debug for ReadbackTicket {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReadbackTicket")
            .field("work", &self.work)
            .field("slice", &self.slice)
            .finish()
    }
}
impl ReadbackTicket {
    /// Query completion without mapping or invalidating the staging allocation.
    pub(crate) fn is_complete(&self) -> Result<bool, VulkanError> {
        self.progress.check(self.work.tick)?;
        Ok(self.work.tick <= self.progress.completed())
    }
    /// Consume this reservation after its native timeline has completed.
    /// Staging ownership and progress travel with the reservation, so the
    /// fencing thread needs neither the recorder nor its mutable staging pool.
    pub fn completed(self) -> Result<CompletedReadback, VulkanError> {
        if !self.is_complete()? {
            return Err(VulkanError::InvalidResource(
                "readback work has not completed",
            ));
        }
        if !self.allocation.valid(self.work.tick) {
            return Err(VulkanError::InvalidResource(
                "stale or cancelled readback reservation",
            ));
        }
        let bytes = unsafe { self.allocation.completed_bytes(&self.slice)? };
        let pointer = std::ptr::NonNull::new(bytes.as_ptr().cast_mut()).expect("mapped readback");
        let length = bytes.len();
        Ok(CompletedReadback {
            _ticket: self,
            pointer,
            length,
        })
    }
    /// Borrow a retained reservation only after native completion. The pool
    /// cannot recycle its atom-isolated storage while the ticket is alive.
    pub(crate) fn try_completed(&self) -> Result<Option<&[u8]>, VulkanError> {
        if !self.is_complete()? {
            return Ok(None);
        }
        if !self.allocation.valid(self.work.tick) {
            return Err(VulkanError::InvalidResource(
                "stale or cancelled readback reservation",
            ));
        }
        Ok(Some(unsafe {
            self.allocation.completed_bytes(&self.slice)?
        }))
    }
    pub fn work(&self) -> WorkTicket {
        self.work
    }
}

/// Mapped bytes from completed GPU work. Owning the reservation prevents staging
/// recycling through consumption, cancellation, early returns and unwinding.
/// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/vk_staging_buffer_pool.cpp
pub struct CompletedReadback {
    _ticket: ReadbackTicket,
    pointer: std::ptr::NonNull<u8>,
    length: usize,
}
impl std::ops::Deref for CompletedReadback {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        // The ticket owns the atom-isolated mapped allocation. Its GPU writes
        // completed before construction and no subsequent producer can reuse it.
        unsafe { std::slice::from_raw_parts(self.pointer.as_ptr(), self.length) }
    }
}
impl AsRef<[u8]> for CompletedReadback {
    fn as_ref(&self) -> &[u8] {
        self
    }
}

#[derive(Debug)]
pub struct QueueUsage {
    pub scheduled_tick: u64,
    pub submitted_tick: u64,
    pub completed_tick: u64,
    pub command_slots: usize,
    pub stream_capacity: u64,
    pub staging_capacity: u64,
    pub retained_readbacks: usize,
    pub retired_buffers: usize,
    pub retired_images: usize,
}

pub(crate) fn align_up(value: u64, alignment: u64) -> Option<u64> {
    value
        .checked_add(alignment - 1)
        .map(|v| v & !(alignment - 1))
}

pub(crate) struct Slot {
    timing: Option<timing::Queries>,
    frame: Option<u64>,
    pub native: Arc<crate::dispatch::CommandSlot>,
    pub tick: u64,
    pub retired_buffers: Vec<Arc<AllocatedBuffer>>,
    pub retired_images: Vec<Arc<AllocatedImage>>,
    pub framebuffers: Vec<Arc<crate::render_pass::Framebuffer>>,
    pub framebuffer_handles: rustc_hash::FxHashSet<vk::Framebuffer>,
    pub retained_views: Vec<Arc<crate::AllocatedImageView>>,
    pub bound_graphics_binding: usize,
    pub bound_compute_offsets: Vec<u32>,
    pub bound_graphics_offsets: Vec<u32>,
    pub bound_compute_layout: vk::PipelineLayout,
    pub bound_graphics_layout: vk::PipelineLayout,
    pub bound_compute: usize,
    pub bound_graphics: usize,
    pub vertex_channel: Option<u64>,
    pub push_layout: vk::PipelineLayout,
    pub push_stages: vk::ShaderStageFlags,
    pub push_bytes: Vec<u8>,
    pub active_render_pass: Option<crate::residency::execution::RenderPassState>,
    pub graphics_state: crate::native::maxwell::dynamic_state::CommandState,
    pub feedback_aspects: Option<vk::ImageAspectFlags>,
    pub bound_compute_binding: usize,
}
impl Slot {
    fn new(native: Arc<crate::dispatch::CommandSlot>) -> Result<Self, VulkanError> {
        let owner = Arc::clone(native.owner());
        let family = native.family();
        let slot = Self {
            timing: timing::Queries::new(Arc::clone(&owner), family)?,
            frame: None,
            native,
            tick: 0,
            retired_buffers: Vec::new(),
            retired_images: Vec::new(),
            framebuffers: Vec::new(),
            framebuffer_handles: Default::default(),
            retained_views: Vec::new(),
            bound_graphics_binding: 0,
            bound_compute_offsets: Vec::new(),
            bound_graphics_offsets: Vec::new(),
            bound_compute_layout: vk::PipelineLayout::null(),
            bound_graphics_layout: vk::PipelineLayout::null(),
            bound_compute: 0,
            bound_graphics: 0,
            vertex_channel: None,
            push_layout: vk::PipelineLayout::null(),
            push_stages: vk::ShaderStageFlags::empty(),
            push_bytes: Vec::new(),
            active_render_pass: None,
            graphics_state: Default::default(),
            feedback_aspects: None,
            bound_compute_binding: 0,
        };
        Ok(slot)
    }
    fn reusable(&self, completed: u64) -> bool {
        self.tick <= completed
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RecordingStatistics {
    pub descriptor_payload_hits: u64,
    pub descriptor_payload_captures: u64,
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
    pub(crate) samples: queries::Samples,
    pub frame_metrics: frame_metrics::Recorder,
    calibration: Option<timing::Calibration>,
    recording_handle: crate::dispatch::RecordingHandle,
    pub(crate) progress: Arc<progress::SubmissionProgress>,
    pub statistics: RecordingStatistics,
    #[cfg(feature = "performance-counters")]
    pub dependency_wait_ns: u64,
    #[cfg(feature = "performance-counters")]
    pub recording_wait_ns: u64,
    owner: Arc<DeviceOwner>,
    pub timeline: Arc<progress::Timeline>,
    pub(crate) wait_for_idle_event: vk::Event,
    instance: BackendInstanceId,
    family: u32,
    pub(crate) staging: staging::StagingPool,
    pub slots: Vec<Slot>,
    in_flight: std::collections::VecDeque<usize>,
    pub recording: Option<usize>,
    pub(crate) deferred_clear: Option<crate::residency::execution::DeferredClear>,
    next_slot: usize,
    pub scheduled: u64,
    submitted_serial: u64,
    last_flush: Option<Instant>,
    pub(crate) flush_pending: bool,
    pub completed: u64,
    stopped: bool,
    lost: bool,
}
impl QueueState {
    pub(crate) fn end_rendering(&mut self, dispatch: &mut crate::dispatch::Dispatch) {
        self.samples.pause(dispatch);
        if let Some(index) = self.recording
            && let Some(active) = self.slots[index].active_render_pass.take()
        {
            active.end(dispatch);
        }
    }
    pub fn new(
        owner: Arc<DeviceOwner>,
        family: u32,
        dispatch: &mut crate::dispatch::Dispatch,
    ) -> Result<Self, VulkanError> {
        static NEXT_INSTANCE: AtomicU64 = AtomicU64::new(1);
        let instance = NEXT_INSTANCE
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| v.checked_add(1))
            .map_err(|_| VulkanError::InvalidResource("Vulkan instance identity exhausted"))?;
        let staging = staging::StagingPool::new(Arc::clone(&owner))?;
        let calibration = timing::Calibration::new(&owner)?;
        let timeline = Arc::new(progress::Timeline::new(Arc::clone(&owner))?);
        let mut state = Self {
            samples: Default::default(),
            frame_metrics: frame_metrics::Recorder::new(),
            calibration,
            recording_handle: dispatch.recording_handle(),
            progress: Arc::new(progress::SubmissionProgress::default()),
            staging,
            owner,
            timeline: Arc::clone(&timeline),
            wait_for_idle_event: vk::Event::null(),
            instance: BackendInstanceId::new(instance),
            family,
            slots: Vec::new(),
            in_flight: Default::default(),
            recording: None,
            deferred_clear: None,
            next_slot: 0,
            statistics: Default::default(),
            #[cfg(feature = "performance-counters")]
            dependency_wait_ns: 0,
            #[cfg(feature = "performance-counters")]
            recording_wait_ns: 0,
            scheduled: 0,
            submitted_serial: 0,
            last_flush: None,
            flush_pending: false,
            completed: 0,
            stopped: false,
            lost: false,
        };
        // One device event is reused in queue order, as RasterizerVulkan's WFI
        // event. Queue shutdown drains both recording and GPU use before drop.
        // https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/renderer_vulkan/vk_rasterizer.cpp#L231
        state.wait_for_idle_event = unsafe {
            state
                .owner
                .device
                .create_event(&vk::EventCreateInfo::default(), None)
        }
        .map_err(|e| VulkanError::vk("vkCreateEvent(wait for idle)", e))?;
        state
            .owner
            .name(state.wait_for_idle_event, "nixe wait for idle")?;
        state
            .owner
            .name(timeline.handle, "nixe submission timeline")?;
        state.allocate_context(dispatch)?;
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
    fn slot_for_tick(&self, tick: u64) -> Option<usize> {
        if let Some(index) = self.recording
            && self.slots[index].tick == tick
        {
            return Some(index);
        }
        let first = self.slots[*self.in_flight.front()?].tick;
        let index = *self
            .in_flight
            .get(usize::try_from(tick.checked_sub(first)?).ok()?)?;
        (self.slots[index].tick == tick).then_some(index)
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
        let last = self.current_tick();
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
    fn reclaim_known(&mut self) -> Result<(), VulkanError> {
        let completed = self.progress.completed();
        if completed > self.completed {
            self.retire_timings(completed)?;
            self.reclaim(completed);
        }
        Ok(())
    }
    fn refresh_completed(&mut self) -> Result<u64, VulkanError> {
        self.reclaim_known()?;
        let submitted = self.progress.submitted();
        if self.completed >= submitted {
            return Ok(self.completed);
        }
        let completed = unsafe {
            self.owner
                .device
                .get_semaphore_counter_value(self.timeline.handle)
        }
        .map_err(|e| self.error("vkGetSemaphoreCounterValue", e))?;
        if completed.min(submitted) > self.completed {
            self.retire_timings(completed.min(submitted))?;
            self.progress.observe_completed(completed.min(submitted));
            self.reclaim(completed.min(submitted));
        }
        Ok(self.completed)
    }
    fn poll_work(&mut self, ticket: WorkTicket) -> Result<WorkStatus, VulkanError> {
        self.check_ticket(ticket)?;
        self.reclaim_known()?;
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
    fn retire_timings(&mut self, completed: u64) -> Result<(), VulkanError> {
        for &index in &self.in_flight {
            let slot = &mut self.slots[index];
            if slot.tick > completed {
                break;
            }
            let interval = if let Some(timing) = &mut slot.timing {
                timing.retire(slot.tick, self.calibration.as_ref())?
            } else {
                None
            };
            if let Some(frame) = slot.frame.take() {
                self.frame_metrics
                    .retire(frame, slot.tick, interval, self.calibration.as_ref());
            }
        }
        Ok(())
    }
    pub fn frame_idle(&mut self, start: Instant, end: Instant, recorded: Option<Instant>) {
        self.frame_metrics
            .idle(start, end, self.scheduled, recorded);
    }
    pub fn begin_composite(
        &mut self,
        dispatch: &mut crate::dispatch::Dispatch,
        pacing: nixe_gpu::frame_metrics::FramePacing,
    ) {
        self.frame_metrics.composite(pacing);
        self.begin_frame_work(dispatch);
    }
    pub fn begin_frame_work(&mut self, dispatch: &mut crate::dispatch::Dispatch) {
        self.frame_metrics.begin_cpu();
        // Host uploads or coherency work may already occupy an unsubmitted
        // command slot. Attribute that shared submission without splitting it.
        if let Some(index) = self.recording {
            let slot = &mut self.slots[index];
            if slot.frame.is_none() {
                slot.frame = self.frame_metrics.begin_submission();
                if let Some(timing) = &mut slot.timing {
                    timing.enable(dispatch);
                }
            }
        }
    }
    pub fn end_frame(&mut self, producer: u64, swap_interval: u32) {
        self.frame_metrics
            .end_cpu(producer, swap_interval, self.calibration.as_ref());
    }
    fn reclaim(&mut self, completed: u64) {
        if completed > self.completed {
            nixe_gpu::trace::event("queue_complete", completed, self.completed, 0);
        }
        self.completed = self.completed.max(completed);
        while let Some(&index) = self.in_flight.front() {
            if self.slots[index].tick > self.completed {
                break;
            }
            self.in_flight.pop_front();
            let slot = &mut self.slots[index];
            {
                slot.retired_buffers.clear();
                slot.retired_images.clear();
                slot.framebuffers.clear();
                slot.framebuffer_handles.clear();
                slot.retained_views.clear();
                slot.graphics_state = Default::default();
                slot.feedback_aspects = None;
            }
        }
    }
    // Scheduler always owns a context for CurrentTick. Allocation occurs only
    // at construction and after sealing a submission, never at resource use.
    // https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/renderer_vulkan/vk_scheduler.cpp#L40-L67
    fn allocate_context(
        &mut self,
        dispatch: &mut crate::dispatch::Dispatch,
    ) -> Result<(), VulkanError> {
        self.check_running()?;
        debug_assert!(self.recording.is_none());
        self.reclaim_known()?;
        let find = |slots: &[Slot], next: usize, completed: u64| {
            (next..slots.len())
                .chain(0..next)
                .find(|&i| slots[i].reusable(completed))
        };
        let index = if let Some(index) = find(&self.slots, self.next_slot, self.completed) {
            index
        } else {
            self.poll()?;
            if let Some(index) = find(&self.slots, self.next_slot, self.completed) {
                index
            } else {
                let first = self.slots.len();
                for native in
                    crate::dispatch::CommandSlot::allocate(Arc::clone(&self.owner), self.family)
                {
                    self.slots.push(Slot::new(native)?);
                }
                first
            }
        };
        self.next_slot = (index + 1) % self.slots.len();
        let tick = self
            .scheduled
            .checked_add(1)
            .ok_or(VulkanError::InvalidResource(
                "submission timeline exhausted",
            ))?;
        let slot = &mut self.slots[index];
        // The slot's previous GPU tick is complete. Reset/begin remain ordered
        // native jobs; no owner/worker round trip is needed to acquire a slot.
        dispatch.begin_command_buffer(
            Arc::clone(&slot.native),
            tick,
            Arc::clone(&self.progress),
            self.timeline.handle,
        )?;
        let slot = &mut self.slots[index];
        // Frame ownership is attached by begin_frame_work or when this
        // context is sealed. Eagerly opening a context must not hold the
        // preceding frame's measurement open through the next frame.
        slot.frame = None;
        if let Some(timing) = &mut slot.timing {
            timing.begin(
                dispatch,
                self.frame_metrics.is_active() || cfg!(feature = "performance-counters"),
            );
        }
        slot.tick = tick;
        slot.bound_compute = 0;
        slot.bound_graphics = 0;
        slot.vertex_channel = None;
        slot.push_layout = vk::PipelineLayout::null();
        slot.push_stages = vk::ShaderStageFlags::empty();
        slot.push_bytes.clear();
        slot.graphics_state = Default::default();
        slot.feedback_aspects = None;
        slot.bound_compute_binding = 0;
        slot.bound_graphics_binding = 0;
        slot.bound_compute_offsets.clear();
        slot.bound_graphics_offsets.clear();
        slot.bound_compute_layout = vk::PipelineLayout::null();
        slot.bound_graphics_layout = vk::PipelineLayout::null();
        self.recording = Some(index);
        self.submitted_serial = dispatch.record_serial();
        Ok(())
    }
    pub fn reserve_upload(&mut self, bytes: &[u8]) -> Result<Slice, VulkanError> {
        self.reserve_upload_aligned(bytes, 4)
    }
    pub(crate) fn reserve_upload_aligned(
        &mut self,
        bytes: &[u8],
        alignment: u64,
    ) -> Result<Slice, VulkanError> {
        let slice = self.reserve_slice(bytes.len() as u64, staging::Usage::Upload, alignment)?;
        unsafe {
            self.staging.write_with(&slice, |destination| {
                destination.copy_from_slice(bytes);
                Ok(())
            })
        }?;
        Ok(slice)
    }
    pub fn reserve_readback(&mut self, size: u64) -> Result<(Slice, ReadbackTicket), VulkanError> {
        self.reserve_readback_aligned(size, 4)
    }
    pub fn reserve_readback_aligned(
        &mut self,
        size: u64,
        alignment: u64,
    ) -> Result<(Slice, ReadbackTicket), VulkanError> {
        if !alignment.is_power_of_two() {
            return Err(VulkanError::InvalidResource(
                "staging alignment must be a nonzero power of two",
            ));
        }
        let slice = self.reserve_slice(size, staging::Usage::Download, alignment)?;
        Ok((
            slice,
            ReadbackTicket {
                work: self.ticket(self.scheduled + 1),
                slice,
                allocation: self.staging.retain_readback(&slice),
                progress: Arc::clone(&self.progress),
            },
        ))
    }
    pub(crate) fn reserve_slice(
        &mut self,
        size: u64,
        usage: staging::Usage,
        alignment: u64,
    ) -> Result<Slice, VulkanError> {
        if self.stopped {
            return Err(VulkanError::QueueStopped);
        }
        let slice = self.staging.reserve(
            size,
            alignment,
            usage,
            self.current_tick(),
            self.completed.max(self.progress.completed()),
        )?;
        Ok(slice)
    }
    pub(crate) fn staging_buffer(&self, slice: &Slice) -> &Arc<AllocatedBuffer> {
        self.staging.buffer(slice)
    }
    #[inline]
    fn current_tick(&self) -> u64 {
        self.scheduled + 1
    }
    /// Empty contexts keep their tick for explicit fences. Command-list ends
    /// submit only actual recorded work, and nearby lists may share that tick.
    pub(crate) fn finish_command_list(&mut self, dispatch: &crate::dispatch::Dispatch) -> bool {
        if self.deferred_clear.is_none() && dispatch.record_serial() == self.submitted_serial {
            return false;
        }
        !self.defer_flush()
    }
    pub(crate) fn defer_flush(&mut self) -> bool {
        if self
            .last_flush
            .is_some_and(|at| at.elapsed() < Duration::from_micros(500))
        {
            self.flush_pending = true;
            true
        } else {
            false
        }
    }
    pub unsafe fn flush(
        &mut self,
        queue: vk::Queue,
        dispatch: &mut crate::dispatch::Dispatch,
        waits: &[vk::SemaphoreSubmitInfo<'_>],
        extra_signals: &[vk::SemaphoreSubmitInfo<'_>],
    ) -> Result<WorkTicket, VulkanError> {
        self.check_running()?;
        let index = self.recording.ok_or(VulkanError::QueueStopped)?;
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
        if self.slots[index].frame.is_none() {
            self.slots[index].frame = self.frame_metrics.begin_submission();
        }
        let slot = &self.slots[index];
        let tick = slot.tick;
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
                .semaphore(self.timeline.handle)
                .value(tick)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
            extra_signals.first().copied().unwrap_or_default(),
        ];
        // Capture native parameters now. The worker closes and submits this
        // command buffer in FIFO order while the owner prepares following work.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/vk_scheduler.cpp
        nixe_gpu::trace::event("queue_schedule", tick, 0, 0);
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
        self.in_flight.push_back(index);
        self.scheduled = tick;
        self.progress.scheduled_successfully(tick);
        self.last_flush = Some(Instant::now());
        self.flush_pending = false;
        self.recording = None;
        self.allocate_context(dispatch)?;
        Ok(self.ticket(tick))
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
        self.reclaim_known()?;
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
        #[cfg(feature = "performance-counters")]
        let submitted_at = Instant::now();
        let submitted = self.progress.wait_submitted(ticket.tick, deadline)?;
        #[cfg(feature = "performance-counters")]
        {
            self.recording_wait_ns += submitted_at.elapsed().as_nanos() as u64;
        }
        if !submitted {
            return Ok(false);
        }
        let semaphores = [self.timeline.handle];
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
            self.dependency_wait_ns += waited_at.elapsed().as_nanos() as u64;
            nixe_gpu::metrics::record(nixe_gpu::metrics::Counter::HostQueueWaits, 1);
            nixe_gpu::metrics::record(
                nixe_gpu::metrics::Counter::HostQueueWaitNanoseconds,
                waited_at.elapsed().as_nanos() as u64,
            );
        }
        match result {
            Ok(()) => {
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
            stream_capacity: self.staging.stream_capacity(),
            staging_capacity: self.staging.pooled_capacity(),
            retained_readbacks: self.staging.retained_readbacks(),
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
            let semaphores = [self.timeline.handle];
            let values = [submitted];
            let info = vk::SemaphoreWaitInfo::default()
                .semaphores(&semaphores)
                .values(&values);
            match unsafe { self.owner.device.wait_semaphores(&info, u64::MAX) } {
                Ok(()) => {
                    self.retire_timings(submitted)?;
                    self.progress.observe_completed(submitted);
                    self.reclaim(submitted);
                }
                Err(vk::Result::ERROR_DEVICE_LOST) => self.lost = true,
                Err(error) => return Err(self.error("shutdown timeline wait", error)),
            }
        }
        self.recording = None;
        self.slots.clear();
        self.in_flight.clear();
        self.staging.clear();
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
        unsafe {
            self.owner
                .device
                .destroy_event(self.wait_for_idle_event, None);
        }
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
    pub(crate) fn end_rendering(&mut self) -> Result<(), VulkanError> {
        self.realize_deferred_clear()?;
        self.submission.end_rendering(&mut self.dispatch);
        Ok(())
    }
    /// Index of the scheduler's already allocated command context. A stopped
    /// device has no context; this query never allocates or admits native work.
    #[inline]
    pub(crate) fn recording_slot(&self) -> Result<usize, VulkanError> {
        self.submission.recording.ok_or(VulkanError::QueueStopped)
    }
    /// The scheduler's current ordered completion point, as CurrentTick. A
    /// flush seals this tick even when it contains no native commands.
    #[inline]
    pub fn current_work(&self) -> WorkTicket {
        self.submission.ticket(self.submission.current_tick())
    }
    /// Submit recorded commands and signal the single timeline, without waiting.
    pub fn flush(&mut self) -> Result<WorkTicket, VulkanError> {
        self.realize_deferred_clear()?;
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
            let index = self
                .submission
                .slot_for_tick(last_use.tick)
                .expect("pending use retains command slot");
            let slot = &mut self.submission.slots[index];
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
            let index = self
                .submission
                .slot_for_tick(last_use.tick)
                .expect("pending image use");
            self.submission.slots[index].retired_images.push(image);
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
            let index = self
                .submission
                .slot_for_tick(last_use.tick)
                .expect("pending image view use");
            self.submission.slots[index].retained_views.extend(views);
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
