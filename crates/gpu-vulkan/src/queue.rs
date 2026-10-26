//! One timeline governs submission, transient reuse and deferred destruction.
use crate::{
    AllocatedBuffer, AllocatedImage, AllocationMode, MemoryLocation, VulkanDevice, VulkanError,
    device::DeviceOwner, memory::allocate_buffer,
};
use ash::vk;
pub(crate) mod progress;
pub(crate) mod staging;
pub(crate) use staging::Slice;
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
    pub(crate) slice: Slice,
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
    #[cfg(feature = "performance-counters")]
    timing: Option<timing::Queries>,
    #[cfg(feature = "binding-probe")]
    owner: Arc<DeviceOwner>,
    pub native: Arc<crate::dispatch::CommandSlot>,
    pub tick: u64,
    pub retired_buffers: Vec<Arc<AllocatedBuffer>>,
    pub retired_images: Vec<Arc<AllocatedImage>>,
    pub retained_views: Vec<Arc<crate::AllocatedImageView>>,
    pub native_pipelines: Vec<Arc<crate::cache::NativePipeline>>,
    pub native_bindings: Vec<Arc<crate::cache::BindingRecord>>,
    pub pipeline_handles: std::collections::HashSet<usize>,
    pub binding_handles: std::collections::HashSet<usize>,
    pub bound_graphics_binding: usize,
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
    pub bound_compute_binding: usize,
}
impl Slot {
    fn new(native: Arc<crate::dispatch::CommandSlot>) -> Result<Self, VulkanError> {
        #[cfg(any(feature = "performance-counters", feature = "binding-probe"))]
        let owner = Arc::clone(native.owner());
        #[cfg(feature = "performance-counters")]
        let family = native.family();
        let slot = Self {
            #[cfg(feature = "performance-counters")]
            timing: timing::Queries::new(Arc::clone(&owner), family)?,
            native,
            #[cfg(feature = "binding-probe")]
            owner,
            tick: 0,
            retired_buffers: Vec::new(),
            retired_images: Vec::new(),
            retained_views: Vec::new(),
            native_pipelines: Vec::new(),
            native_bindings: Vec::new(),
            pipeline_handles: Default::default(),
            binding_handles: Default::default(),
            bound_graphics_binding: 0,
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
            bound_compute_binding: 0,
        };
        Ok(slot)
    }
    fn reusable(&self, completed: u64) -> bool {
        self.tick <= completed
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
    recording_handle: crate::dispatch::RecordingHandle,
    pub(crate) progress: Arc<progress::SubmissionProgress>,
    pub statistics: RecordingStatistics,
    owner: Arc<DeviceOwner>,
    pub timeline: vk::Semaphore,
    instance: BackendInstanceId,
    family: u32,
    pub(crate) staging: staging::StagingPool,
    pub slots: Vec<Slot>,
    in_flight: std::collections::VecDeque<usize>,
    pub recording: Option<usize>,
    next_slot: usize,
    pub scheduled: u64,
    pub completed: u64,
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
            staging: staging::StagingPool::new(Arc::clone(&owner)),
            owner,
            timeline,
            instance: BackendInstanceId::new(instance),
            family,
            slots: Vec::new(),
            in_flight: Default::default(),
            recording: None,
            next_slot: 0,
            statistics: Default::default(),
            scheduled: 0,
            completed: 0,
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
    fn reclaim_known(&mut self) -> Result<(), VulkanError> {
        let completed = self.progress.completed();
        if completed > self.completed {
            #[cfg(feature = "performance-counters")]
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
    #[cfg(feature = "performance-counters")]
    fn retire_timings(&mut self, completed: u64) -> Result<(), VulkanError> {
        for &index in &self.in_flight {
            let slot = &mut self.slots[index];
            if slot.tick > completed {
                break;
            }
            if let Some(timing) = &mut slot.timing {
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
        while let Some(&index) = self.in_flight.front() {
            if self.slots[index].tick > self.completed {
                break;
            }
            self.in_flight.pop_front();
            let slot = &mut self.slots[index];
            {
                slot.retired_buffers.clear();
                slot.retired_images.clear();
                slot.retained_views.clear();
                slot.graphics_parameters = None;
                slot.feedback_aspects = None;
                slot.native_targets.clear();
                slot.target_handles.clear();
                slot.native_bindings.clear();
                slot.native.retire_descriptor_pools();
                slot.native_pipelines.clear();
                slot.pipeline_handles.clear();
                slot.binding_handles.clear();
                #[cfg(feature = "binding-probe")]
                slot.probe_queries.clear();
            }
        }
    }
    /// Prepared mapped inputs can outlive an intervening dependency submit.
    /// Their stream regions, not the completed command context, stay reserved.
    pub(crate) fn retain_stream_upload(&mut self, slice: &Slice) {
        self.staging.defer_regions(self.staging.regions(slice));
    }
    pub(crate) fn release_stream_uploads(&mut self) {
        let consumer = self.scheduled + u64::from(self.recording.is_some());
        self.staging.release_regions(consumer);
    }
    pub fn begin(
        &mut self,
        dispatch: &mut crate::dispatch::Dispatch,
    ) -> Result<usize, VulkanError> {
        self.check_running()?;
        if let Some(index) = self.recording {
            return Ok(index);
        }
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
            self.timeline,
        )?;
        let slot = &mut self.slots[index];
        #[cfg(feature = "performance-counters")]
        if let Some(timing) = &mut slot.timing {
            timing.begin(dispatch);
        }
        slot.tick = tick;
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
        slot.bound_compute_binding = 0;
        slot.bound_graphics_binding = 0;
        slot.bound_compute_offsets.clear();
        slot.bound_graphics_offsets.clear();
        slot.bound_compute_layout = vk::PipelineLayout::null();
        slot.bound_graphics_layout = vk::PipelineLayout::null();
        self.recording = Some(index);
        Ok(index)
    }
    pub fn reserve_upload(
        &mut self,
        dispatch: &mut crate::dispatch::Dispatch,
        bytes: &[u8],
    ) -> Result<Slice, VulkanError> {
        self.reserve_upload_aligned(dispatch, bytes, 4)
    }
    pub(crate) fn reserve_upload_aligned(
        &mut self,
        dispatch: &mut crate::dispatch::Dispatch,
        bytes: &[u8],
        alignment: u64,
    ) -> Result<Slice, VulkanError> {
        let slice = self.reserve_slice(dispatch, bytes.len() as u64, true, alignment)?;
        let buffer = self.staging_buffer(&slice);
        unsafe { buffer.write_range(slice.offset, bytes) }?;
        Ok(slice)
    }
    pub fn reserve_readback(
        &mut self,
        dispatch: &mut crate::dispatch::Dispatch,
        size: u64,
    ) -> Result<(Slice, ReadbackTicket), VulkanError> {
        self.reserve_readback_aligned(dispatch, size, 4)
    }
    pub fn reserve_readback_aligned(
        &mut self,
        dispatch: &mut crate::dispatch::Dispatch,
        size: u64,
        alignment: u64,
    ) -> Result<(Slice, ReadbackTicket), VulkanError> {
        if !alignment.is_power_of_two() {
            return Err(VulkanError::InvalidResource(
                "staging alignment must be a nonzero power of two",
            ));
        }
        let slice = self.reserve_slice(dispatch, size, false, alignment)?;
        Ok((
            slice.clone(),
            ReadbackTicket {
                work: self.ticket(self.scheduled + 1),
                slice,
            },
        ))
    }
    pub(crate) fn reserve_slice(
        &mut self,
        dispatch: &mut crate::dispatch::Dispatch,
        size: u64,
        upload: bool,
        alignment: u64,
    ) -> Result<Slice, VulkanError> {
        let index = self.begin(dispatch)?;
        let slice = self.staging.reserve(
            size,
            alignment,
            upload,
            self.slots[index].tick,
            self.completed.max(self.progress.completed()),
        )?;
        Ok(slice)
    }
    pub(crate) fn staging_buffer<'a>(&mut self, slice: &'a Slice) -> &'a Arc<AllocatedBuffer> {
        let tick = self
            .recording
            .map_or(self.scheduled, |i| self.slots[i].tick);
        self.staging.mark_used(slice, tick);
        self.staging.buffer(slice)
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
        self.in_flight.push_back(index);
        self.scheduled = tick;
        self.recording = None;
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
    pub(crate) fn begin_recording(&mut self) -> Result<usize, VulkanError> {
        self.submission.begin(&mut self.dispatch)
    }

    /// Reserve an ordered completion point, including a real signal for work
    /// with no commands. A flush accepts it; host polling alone is not guest success.
    pub fn work_ticket(&mut self) -> Result<WorkTicket, VulkanError> {
        self.begin_recording()?;
        Ok(self.submission.ticket(self.submission.scheduled + 1))
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
    /// retain their allocation after GPU completion. No unread range can recycle.
    pub fn readback(&mut self, ticket: ReadbackTicket) -> Result<Vec<u8>, VulkanError> {
        self.validate_readback(&ticket)?;
        if ticket.work.tick > self.submission.scheduled {
            self.flush()?;
        }
        self.wait_for_work(ticket.work, Duration::MAX)?;
        let slice = ticket.slice;
        let result = unsafe {
            self.submission
                .staging
                .buffer(&slice)
                .read_range(slice.offset, slice.size)
        };
        self.submission.staging.release_readback(&slice);
        self.submission.reclaim(self.submission.completed);
        result
    }
    pub fn discard_readback(&mut self, ticket: ReadbackTicket) -> Result<(), VulkanError> {
        self.validate_readback(&ticket)?;
        self.submission.staging.release_readback(&ticket.slice);
        self.submission.reclaim(self.submission.completed);
        Ok(())
    }
    fn validate_readback(&self, ticket: &ReadbackTicket) -> Result<(), VulkanError> {
        self.submission.check_ticket(ticket.work)?;
        if !self
            .submission
            .staging
            .valid_readback(&ticket.slice, ticket.work.tick)
        {
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
