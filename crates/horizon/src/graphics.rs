//! Switch display-service state shared by VI, Binder, and nvdrv sessions.

use nixe_gpu_maxwell::PresentationImageRequest;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt::{Display, Formatter};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex, Weak};
use std::task::{Context, Poll, Waker};

use nixe_gpu::{
    BlockLinearLayout, GuestSyncpointId, GuestSyncpointValue, GuestTimelinePoint,
    ImageMemoryLayout, PresentationImageFormat,
};
use nixe_gpu::{FrameCrop, FrameTransform};
use nixe_runtime::{EventObject, ReadableEventObject, WritableEventObject};
use nixe_video::FrameMailbox;
mod conductor;

use crate::parcel::{ParcelError, ParcelReader, ParcelWriter};
use crate::{
    GraphicsEventSource, NvDrvSession, NvMapExportedId, NvMapImageViewMetadata, NvMapObject,
    NvMapPlaneMetadata,
};

const DEFAULT_DISPLAY_ID: u64 = 1;
const DEFAULT_WIDTH: u32 = 1280;
const DEFAULT_HEIGHT: u32 = 720;

/// VI refresh event, kept type-distinct from queue and GPU events even though
/// all three currently use runtime kernel-event primitives.
#[derive(Debug)]
struct ViVsyncEvent {
    video: Weak<Mutex<VideoState>>,
    source: GraphicsEventSource,
    writable: WritableEventObject,
    readable: ReadableEventObject,
}

impl ViVsyncEvent {
    fn new(video: &VideoSystem, display_id: u64) -> Arc<Self> {
        let (writable, readable) = EventObject::create_pair(&video.kernel);
        let event = Arc::new(Self {
            video: Arc::downgrade(&video.state),
            source: GraphicsEventSource::ViVsync { display_id },
            writable,
            readable,
        });
        video
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .vsync_events
            .insert(Arc::as_ptr(&event) as usize, event.writable.clone());
        event
    }

    fn readable(&self) -> ReadableEventObject {
        debug_assert!(matches!(self.source, GraphicsEventSource::ViVsync { .. }));
        self.readable.clone()
    }
}

// Register and unlink per application-display client, as VI does in Eden.
// The registry retains only the writable kernel handle, never the client.
// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/service/vi/application_display_service.cpp
impl Drop for ViVsyncEvent {
    fn drop(&mut self) {
        if let Some(video) = self.video.upgrade() {
            video
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .vsync_events
                .remove(&(self as *const Self as usize));
        }
    }
}

/// Producer native handle. HOS signals this event on configuration/cancellation;
/// free-slot waits belong to the independent dequeue condition below.
/// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/service/nvnflinger/buffer_queue_producer.cpp
#[derive(Clone, Debug)]
struct BufferQueueWaitEvent {
    source: GraphicsEventSource,
    writable: WritableEventObject,
    readable: ReadableEventObject,
}

impl BufferQueueWaitEvent {
    fn new(kernel: &Arc<nixe_scheduler::SchedulerState>, binder_id: i32) -> Self {
        let (writable, readable) = EventObject::create_pair(kernel);
        Self {
            source: GraphicsEventSource::BufferQueueAvailability { binder_id },
            writable,
            readable,
        }
    }

    fn signal(&self) {
        self.writable.signal();
    }

    fn readable(&self) -> ReadableEventObject {
        debug_assert!(matches!(
            self.source,
            GraphicsEventSource::BufferQueueAvailability { .. }
        ));
        self.readable.clone()
    }
}

/// Privilege of one VI root service connection.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ViServiceKind {
    Application,
    System,
    Manager,
}

impl ViServiceKind {
    pub(crate) fn from_name(name: &[u8]) -> Option<Self> {
        match name {
            b"vi:u" => Some(Self::Application),
            b"vi:s" => Some(Self::System),
            b"vi:m" => Some(Self::Manager),
            _ => None,
        }
    }

    pub(crate) const fn required_root_command(self) -> u32 {
        match self {
            Self::Application => 0,
            Self::System => 1,
            Self::Manager => 2,
        }
    }
}

/// Concrete interface represented by a VI session handle.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ViObjectKind {
    Root(ViServiceKind),
    ApplicationDisplay,
    BinderRelay,
    SystemDisplay,
    ManagerDisplay,
}

/// Handle-table object for a VI root or child interface.
#[derive(Clone, Debug)]
pub struct ViSession {
    kind: ViObjectKind,
    video: VideoSystem,
    client: Arc<Mutex<ViClient>>,
}

#[derive(Debug, Default)]
struct ViClient {
    opened_layers: BTreeSet<u64>,
    vsync_event: Option<Arc<ViVsyncEvent>>,
}

impl ViSession {
    pub(crate) fn new(kind: ViObjectKind, video: VideoSystem) -> Self {
        Self {
            kind,
            video,
            client: Arc::default(),
        }
    }

    pub(crate) const fn kind(&self) -> ViObjectKind {
        self.kind
    }

    pub(crate) fn video(&self) -> &VideoSystem {
        &self.video
    }

    pub(crate) fn vsync_event(
        &self,
        display_id: u64,
    ) -> Result<ReadableEventObject, crate::HorizonIpcResult> {
        VideoSystem::display_resolution(display_id)
            .ok_or(crate::HorizonIpcResult::SF_PRECONDITION_VIOLATION)?;
        let mut client = self
            .client
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if client.vsync_event.is_some() {
            return Err(crate::HorizonIpcResult::VI_PERMISSION_DENIED);
        }
        let event = ViVsyncEvent::new(&self.video, display_id);
        let readable = event.readable();
        client.vsync_event = Some(event);
        Ok(readable)
    }

    pub(crate) fn open_layer(&self, layer_id: u64) -> Option<LayerState> {
        let layer = self.video.layer(layer_id)?;
        self.client
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .opened_layers
            .insert(layer_id)
            .then_some(layer)
    }

    // Close the application connection, not the AM/manager-owned layer.
    // The Binder producer retains its identity across a resolution change.
    // https://github.com/eden-emulator/mirror/blob/d16735f5b618942136d6ab53466e3be0a382c30a/src/core/hle/service/vi/application_display_service.cpp
    // https://github.com/eden-emulator/mirror/blob/d16735f5b618942136d6ab53466e3be0a382c30a/src/core/hle/service/vi/container.cpp
    pub(crate) fn close_layer(&self, layer_id: u64) -> bool {
        if !self
            .client
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .opened_layers
            .remove(&layer_id)
        {
            return false;
        }
        let mut state = self
            .video
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(layer) = state.layers.get(&layer_id) else {
            return false;
        };
        let binder_id = layer.binder_id;
        if let Some(queue) = state.queues.get_mut(&binder_id) {
            queue.lock().unwrap().abandon();
        }
        true
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LayerState {
    pub(crate) id: u64,
    pub(crate) binder_id: i32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) visible: bool,
    pub(crate) scaling_mode: u32,
    pub(crate) x: f32,
    pub(crate) y: f32,
    pub(crate) z: i64,
}

#[derive(Debug)]
struct VideoState {
    next_layer_id: u64,
    layers: BTreeMap<u64, LayerState>,
    queues: BTreeMap<i32, Arc<Mutex<BufferQueue>>>,
    vsync_events: BTreeMap<usize, WritableEventObject>,
    presenter: Option<FrameMailbox>,
    nvdrv: NvDrvSession,
    next_frame_sequence: u64,
    composition_frame: u64,
    pending_composites: Vec<std::sync::mpsc::Receiver<Result<(), Box<str>>>>,
}

/// Cloneable process display state shared across independently opened services.
#[derive(Clone, Debug)]
pub struct VideoSystem {
    kernel: Arc<nixe_scheduler::SchedulerState>,
    state: Arc<Mutex<VideoState>>,
    conductor: Arc<std::sync::OnceLock<Result<conductor::Conductor, Box<str>>>>,
}

/// Resources released when one emulated process graphics system is torn down.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct GraphicsTeardownReport {
    pub layers_released: usize,
    pub queues_released: usize,
    pub pending_frames_released: usize,
    pub device_fds_released: usize,
    pub allocations_released: usize,
}

impl VideoSystem {
    #[must_use]
    pub fn new(
        machine: nixe_memory::Machine,
        kernel: &Arc<nixe_scheduler::SchedulerState>,
    ) -> Self {
        Self::with_nvdrv(kernel, None, NvDrvSession::new(machine, kernel))
    }

    /// Construct channel graphics state beside its renderer on the GPU owner.
    pub fn with_gpu_frontend<E, L, R>(
        device_memory: nixe_gpu_maxwell::MaxwellDeviceMemory,
        kernel: &Arc<nixe_scheduler::SchedulerState>,
        presenter: Option<FrameMailbox>,
        mme_jit: bool,
        initialize: impl FnOnce() -> Result<(E, R), nixe_gpu::BackendRuntimeError> + Send + 'static,
        frontend: impl FnOnce(std::rc::Rc<std::cell::RefCell<Option<E>>>) -> L + Send + 'static,
    ) -> Result<(Self, R), nixe_gpu::BackendRuntimeError>
    where
        L: nixe_gpu_maxwell::MaxwellFrontend,
        R: Send + 'static,
        E: nixe_gpu::GpuExecution<
                Conversion = nixe_gpu::native::NativeNv12Conversion,
                ImageRequest = nixe_gpu_maxwell::PresentationImageRequest,
            >,
    {
        let (nvdrv, report) =
            NvDrvSession::with_gpu_frontend(device_memory, kernel, mme_jit, initialize, frontend)?;
        Ok((Self::with_nvdrv(kernel, presenter, nvdrv), report))
    }

    fn with_nvdrv(
        kernel: &Arc<nixe_scheduler::SchedulerState>,
        presenter: Option<FrameMailbox>,
        nvdrv: NvDrvSession,
    ) -> Self {
        Self {
            kernel: kernel.clone(),
            state: Arc::new(Mutex::new(VideoState {
                next_layer_id: 1,
                layers: BTreeMap::new(),
                queues: BTreeMap::new(),
                vsync_events: BTreeMap::new(),
                presenter,
                nvdrv,
                next_frame_sequence: 1,
                composition_frame: 0,
                pending_composites: Vec::new(),
            })),
            conductor: Arc::default(),
        }
    }

    pub(crate) fn start_conductor(
        &self,
        events: nixe_runtime::ExternalEventSender,
    ) -> Result<(), FramebufferError> {
        self.conductor
            .get_or_init(|| {
                conductor::Conductor::new(
                    Arc::clone(&self.kernel),
                    Arc::downgrade(&self.state),
                    events,
                )
                .map_err(|error| error.to_string().into_boxed_str())
            })
            .as_ref()
            .map(|_| ())
            .map_err(|error| FramebufferError::Backend(error.clone()))
    }
    pub(crate) fn check_conductor(&self) -> Result<(), FramebufferError> {
        match self.conductor.get() {
            Some(Ok(conductor)) => conductor.check(),
            Some(Err(error)) => Err(FramebufferError::Backend(error.clone())),
            None => Ok(()),
        }
    }

    /// Returns the number of live VI layers for diagnostics and acceptance
    /// tests without exposing mutable compositor state.
    #[must_use]
    pub fn active_layer_count(&self) -> usize {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .layers
            .len()
    }

    /// Deterministically releases process-owned display and driver resources.
    ///
    /// Host presentation resources are owned by `nixe-video-winit`, not this
    /// guest graphics system. The frontend releases those resources when its
    /// worker-finished event is handled.
    #[must_use]
    pub fn teardown(&self) -> GraphicsTeardownReport {
        if let Some(Ok(conductor)) = self.conductor.get() {
            conductor.stop();
        }
        let nvdrv = self.nvdrv().teardown();
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let report = GraphicsTeardownReport {
            layers_released: state.layers.len(),
            queues_released: state.queues.len(),
            pending_frames_released: state
                .queues
                .values()
                .map(|q| q.lock().unwrap().pending.len())
                .sum(),
            device_fds_released: nvdrv.device_fds_released,
            allocations_released: nvdrv.allocations_released,
        };
        state.layers.clear();
        for queue in state.queues.values() {
            queue.lock().unwrap().abandon();
        }
        state.queues.clear();
        report
    }

    pub(crate) fn nvdrv(&self) -> NvDrvSession {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .nvdrv
            .clone()
    }

    pub(crate) fn open_display(&self, name: &[u8]) -> Option<u64> {
        let end = name
            .iter()
            .position(|byte| *byte == 0)
            .unwrap_or(name.len());
        if &name[..end] == b"Default" {
            Some(DEFAULT_DISPLAY_ID)
        } else {
            None
        }
    }

    pub(crate) const fn display_resolution(display_id: u64) -> Option<(u32, u32)> {
        if display_id == DEFAULT_DISPLAY_ID {
            Some((DEFAULT_WIDTH, DEFAULT_HEIGHT))
        } else {
            None
        }
    }

    pub(crate) fn create_layer(&self, display_id: u64) -> Option<LayerState> {
        Self::display_resolution(display_id)?;
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let id = state.next_layer_id;
        state.next_layer_id = state.next_layer_id.checked_add(1)?;
        let binder_id = i32::try_from(id).ok()?;
        let layer = LayerState {
            id,
            binder_id,
            width: DEFAULT_WIDTH,
            height: DEFAULT_HEIGHT,
            visible: true,
            scaling_mode: 0,
            x: 0.0,
            y: 0.0,
            z: 0,
        };
        state.layers.insert(id, layer.clone());
        state.queues.insert(
            binder_id,
            Arc::new(Mutex::new(BufferQueue::new(&self.kernel, binder_id))),
        );
        Some(layer)
    }

    pub(crate) fn layer(&self, layer_id: u64) -> Option<LayerState> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .layers
            .get(&layer_id)
            .cloned()
    }

    pub(crate) fn remove_layer(&self, layer_id: u64) -> bool {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(layer) = state.layers.remove(&layer_id) else {
            return false;
        };
        if let Some(queue) = state.queues.remove(&layer.binder_id) {
            queue.lock().unwrap().abandon();
        }
        true
    }

    pub(crate) fn set_layer_scaling_mode(&self, layer_id: u64, mode: u32) -> bool {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(layer) = state.layers.get_mut(&layer_id) else {
            return false;
        };
        layer.scaling_mode = mode;
        true
    }

    pub(crate) fn set_layer_position(&self, layer_id: u64, x: f32, y: f32) -> bool {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(layer) = state.layers.get_mut(&layer_id) else {
            return false;
        };
        layer.x = x;
        layer.y = y;
        true
    }

    pub(crate) fn set_layer_size(&self, layer_id: u64, width: u32, height: u32) -> bool {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(layer) = state.layers.get_mut(&layer_id) else {
            return false;
        };
        layer.width = width;
        layer.height = height;
        true
    }

    pub(crate) fn set_layer_z(&self, layer_id: u64, z: i64) -> bool {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(layer) = state.layers.get_mut(&layer_id) else {
            return false;
        };
        layer.z = z;
        true
    }

    pub(crate) fn binder_event(&self, binder_id: i32) -> Option<ReadableEventObject> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .queues
            .get(&binder_id)
            .map(|queue| queue.lock().unwrap().buffer_wait_event.readable())
    }

    pub(crate) fn adjust_binder_refcount(
        &self,
        binder_id: i32,
        add_value: i32,
        reference_type: i32,
    ) -> bool {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.queues.get_mut(&binder_id).is_some_and(|queue| {
            queue
                .lock()
                .unwrap()
                .adjust_refcount(add_value, reference_type)
        })
    }

    pub(crate) fn transact_binder<'a>(
        &self,
        binder_id: i32,
        code: u32,
        encoded: &'a [u8],
    ) -> Result<BinderCall<'a>, ParcelError> {
        let queue = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .queues
            .get(&binder_id)
            .cloned()
            .ok_or(ParcelError::Malformed("unknown Binder producer object"))?;
        Ok(BinderCall {
            queue,
            code,
            encoded,
            waiter: None,
        })
    }

    /// Resolves one producer reservation to canonical nvmap storage and makes
    /// it visible to the compositor without reading it before its acquire
    /// fence. The slot transition is committed only after every descriptor
    /// boundary has been validated.
    pub(crate) fn queue_graphic_buffer(
        &self,
        binder_id: i32,
        request: QueuedBufferRequest,
    ) -> Result<(), FramebufferError> {
        let requested_interval = request.input.swap_interval;
        nixe_memory::trace::event(
            "requested_interval",
            ((binder_id as u32 as u64) << 32) | request.slot as u32 as u64,
            u64::from(requested_interval),
            u64::from(request.input.swap_interval),
        );
        nixe_memory::trace::event(
            "queue_buffer",
            ((binder_id as u32 as u64) << 32) | request.slot as u32 as u64,
            u64::from(request.input.swap_interval),
            0,
        );
        let reserved_slot = request.slot;
        let result = (|| {
            let primary = request.buffer.primary_plane();
            let crop = effective_crop(request.input.crop, primary.width, primary.height);
            validate_present_descriptor(&request.buffer, Some(crop))?;
            let transform = frame_transform(request.input.transform)?;
            let nvdrv = self.nvdrv();
            for fence in request.input.acquire_fences.iter() {
                if !nvdrv.knows_guest_timeline_point(fence.point) {
                    return Err(FramebufferError::Malformed(
                        "QueueBuffer acquire fence references an unknown GPU syncpoint",
                    ));
                }
            }
            let object = nvdrv
                .nvmap_object_by_id(NvMapExportedId::new(request.buffer.nvmap_id))
                .ok_or(FramebufferError::Malformed(
                    "queued graphic buffer references an unknown nvmap ID",
                ))?;
            if request.buffer.total_size > object.size() {
                return Err(FramebufferError::Malformed(
                    "queued graphic buffer exceeds its nvmap allocation",
                ));
            }
            object
                .validate_image_view(&request.buffer.nvmap_view_metadata())
                .map_err(FramebufferError::NvMap)?;

            let presenting = {
                let state = self
                    .state
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                state.presenter.is_some()
            };
            let mut presentation_source = request.presentation_source;
            let presentation = if presenting {
                let mut source = match &presentation_source {
                    Some(source) => source.clone(),
                    None => presentable_image_request(&request.buffer, &object, &nvdrv)?,
                };
                source.crop = frame_crop(crop)?;
                source.transform = transform;
                presentation_source = Some(source.clone());
                Some(source)
            } else {
                None
            };

            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let queue = state
                .queues
                .get_mut(&binder_id)
                .ok_or(FramebufferError::Malformed(
                    "unknown Binder producer object",
                ))?;
            let mut queue = queue.lock().unwrap();
            if !queue.commit_queue(
                request.slot,
                presentation_source,
                request.input.sticky_transform,
            ) {
                return Err(FramebufferError::Malformed(
                    "QueueBuffer reservation lost slot ownership",
                ));
            }
            queue.pending.push_back(PendingFrame {
                slot: request.slot,
                acquire_fences: request.input.acquire_fences,
                swap_interval: request.input.swap_interval,
                presentation,
            });
            drop(queue);
            drop(state);
            Ok(())
        })();

        if result.is_err() {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(queue) = state.queues.get_mut(&binder_id) {
                queue.lock().unwrap().rollback_queue(reserved_slot);
            }
        }
        result
    }

    fn poll_composite_registration(state: &mut VideoState) -> Result<bool, FramebufferError> {
        if !state.layers.values().any(|layer| layer.visible) {
            return Ok(true);
        }
        // Bound in-flight compositions by the previous sync registration;
        // guest fences and renderer publication retain their GPU-owned lives.
        // https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/gpu.cpp#L248-L295
        while let Some(registration) = state.pending_composites.last() {
            match registration.try_recv() {
                Ok(result) => result.map_err(FramebufferError::Backend)?,
                Err(std::sync::mpsc::TryRecvError::Empty) => return Ok(false),
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    return Err(FramebufferError::Backend(
                        "GPU composition registration was cancelled".into(),
                    ));
                }
            }
            state.pending_composites.pop();
        }
        Ok(true)
    }
    fn compose(state: &mut VideoState, registration: &Waker) -> Result<(), FramebufferError> {
        if !state.layers.values().any(|layer| layer.visible) {
            for event in state.vsync_events.values() {
                event.signal();
            }
            return Ok(());
        }
        let display_sequence = state.composition_frame.saturating_add(1);
        let VideoState {
            queues,
            nvdrv,
            presenter,
            next_frame_sequence,
            composition_frame,
            pending_composites,
            ..
        } = state;
        for (&binder_id, queue) in &*queues {
            let mut queue = queue.lock().unwrap();
            if queue.acquired.is_some() {
                continue;
            }
            let Some(pending) = queue.pending.front() else {
                continue;
            };
            if pending.presentation.is_none()
                && !pending
                    .acquire_fences
                    .iter()
                    .all(|f| nvdrv.guest_timeline_point_reached(f.point).unwrap_or(false))
            {
                continue;
            }
            let pending = queue.pending.pop_front().unwrap();
            queue.acquire(pending.slot, *composition_frame, pending.swap_interval)?;
            let source_id = ((binder_id as u32 as u64) << 32) | pending.slot as u32 as u64;
            nixe_memory::trace::event(
                "vi_acquire",
                source_id,
                display_sequence,
                u64::from(pending.swap_interval),
            );
            if let Some(request) = pending.presentation {
                let sequence = *next_frame_sequence;
                *next_frame_sequence = sequence.saturating_add(1);
                let registration = nvdrv
                    .request_composite(
                        crate::nvdrv::Composite {
                            request,
                            swap_interval: pending.swap_interval,
                            presenter: presenter
                                .as_ref()
                                .expect("resident composition retains its presenter")
                                .clone(),
                            sequence,
                            source: source_id,
                            display_sequence,
                        },
                        pending.acquire_fences.iter().map(|f| f.point).collect(),
                        registration.clone(),
                    )
                    .map_err(FramebufferError::Backend)?;
                pending_composites.push(registration);
            } else {
                nixe_memory::trace::event(
                    "vi_latch",
                    source_id,
                    *next_frame_sequence,
                    display_sequence,
                );
                log::debug!(
                    "discarded headless frame from Binder producer {binder_id} slot {} at VSync {display_sequence}",
                    pending.slot
                );
            }
        }
        // HardwareComposer counts completed compositions, including empty
        // ones. Coalesced host timer ticks do not fabricate compositions.
        *composition_frame = composition_frame.saturating_add(1);
        // Release against the advanced counter in this composition, as Eden
        // does, so interval1 does not occupy a producer slot for another tick.
        // https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/core/hle/service/nvnflinger/hardware_composer.cpp#L165-L175
        for (&binder_id, queue) in queues {
            if let Some(slot) = queue.lock().unwrap().release_acquired(*composition_frame) {
                nixe_memory::trace::event(
                    "buffer_release",
                    ((binder_id as u32 as u64) << 32) | slot as u32 as u64,
                    display_sequence,
                    0,
                );
            }
        }
        for event in state.vsync_events.values() {
            event.signal();
        }
        Ok(())
    }
}

const IGRAPHIC_BUFFER_PRODUCER: &str = "android.gui.IGraphicBufferProducer";
// libnx's NvMultiFence is a u32 count followed by four { id, value } fences.
const MAX_BUFFER_SLOTS: i32 = 64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SlotOwnership {
    Free,
    Dequeued,
    Queueing,
    Queued,
    Acquired,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct GraphicBufferPlane {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) color_format: u64,
    pub(crate) layout: u32,
    pub(crate) pitch: u32,
    pub(crate) offset: u32,
    pub(crate) kind: u32,
    pub(crate) block_height_log2: u32,
    pub(crate) scan_format: u32,
    pub(crate) second_field_offset: u32,
    pub(crate) flags: u64,
    pub(crate) size: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct GraphicBuffer {
    width: u32,
    height: u32,
    window_format: u32,
    pub(crate) nvmap_id: u32,
    pub(crate) stride_pixels: u32,
    pub(crate) format: u32,
    pub(crate) external_format: u32,
    pub(crate) usage: u32,
    pub(crate) total_size: u32,
    pub(crate) planes: Box<[GraphicBufferPlane]>,
}

impl GraphicBuffer {
    fn primary_plane(&self) -> &GraphicBufferPlane {
        &self.planes[0]
    }

    pub(crate) fn nvmap_view_metadata(&self) -> NvMapImageViewMetadata {
        let primary = self.primary_plane();
        NvMapImageViewMetadata::new(
            primary.width,
            primary.height,
            self.format,
            primary.kind,
            primary.layout,
            primary.block_height_log2,
            self.planes
                .iter()
                .map(|plane| {
                    NvMapPlaneMetadata::new(u64::from(plane.offset), plane.size, plane.pitch)
                })
                .collect(),
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct NvFence {
    point: GuestTimelinePoint,
}

// Keep the producer's four fence entries inline, as android::Fence does. Both
// the queued item and its reusable slot retain the same guest fence values.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct NvMultiFence {
    count: usize,
    entries: [NvFence; 4],
}

impl Default for NvMultiFence {
    fn default() -> Self {
        Self {
            count: 0,
            entries: [NvFence {
                point: GuestTimelinePoint::new(
                    GuestSyncpointId::new(u32::MAX),
                    GuestSyncpointValue::new(0),
                ),
            }; 4],
        }
    }
}

impl std::ops::Deref for NvMultiFence {
    type Target = [NvFence];
    fn deref(&self) -> &Self::Target {
        &self.entries[..self.count]
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct QueueBufferInput {
    timestamp: i64,
    auto_timestamp: bool,
    crop: CropRect,
    scaling_mode: u32,
    transform: u32,
    sticky_transform: u32,
    swap_interval: u32,
    acquire_fences: NvMultiFence,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CropRect {
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
}

#[derive(Clone, Debug)]
pub(crate) struct QueuedBufferRequest {
    slot: i32,
    buffer: GraphicBuffer,
    presentation_source: Option<PresentationImageRequest>,
    input: QueueBufferInput,
}

#[derive(Clone, Debug)]
struct BufferSlot {
    ownership: SlotOwnership,
    frame_number: u64,
    buffer: GraphicBuffer,
    // RequestBuffer must return the complete registered descriptor, including
    // native-handle fields not consumed by the presentation path.
    flattened_buffer: Box<[u8]>,
    presentation_source: Option<PresentationImageRequest>,
    release_fences: NvMultiFence,
}

#[derive(Debug)]
struct BufferQueue {
    connected_api: Option<i32>,
    abandoned: bool,
    default_width: u32,
    default_height: u32,
    default_format: u32,
    sticky_transform: u32,
    next_waiter: u64,
    dequeue_waiters: BTreeMap<u64, Waker>,
    override_max_buffer_count: i32,
    frame_counter: u64,
    weak_references: i64,
    strong_references: i64,
    slots: BTreeMap<i32, BufferSlot>,
    buffer_wait_event: BufferQueueWaitEvent,
    acquired: Option<AcquiredBuffer>,
    pending: VecDeque<PendingFrame>,
}

#[derive(Debug)]
struct AcquiredBuffer {
    slot: i32,
    release_frame: u64,
}

#[derive(Clone, Debug)]
pub(crate) struct BinderTransaction {
    pub(crate) reply: Vec<u8>,
    pub(crate) queued: Option<QueuedBufferRequest>,
}

#[derive(Debug)]
struct PendingFrame {
    slot: i32,
    acquire_fences: NvMultiFence,
    swap_interval: u32,
    presentation: Option<PresentationImageRequest>,
}

/// One accepted producer operation. A free-slot wait releases the producer
/// mutex and suspends its IPC continuation, just as Eden's dequeue condition
/// releases core->mutex. It never uses the native handle exposed to the game.
/// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/service/nvnflinger/buffer_queue_producer.cpp#L125-L228
pub(crate) struct BinderCall<'a> {
    queue: Arc<Mutex<BufferQueue>>,
    code: u32,
    encoded: &'a [u8],
    waiter: Option<u64>,
}

impl Future for BinderCall<'_> {
    type Output = Result<BinderTransaction, ParcelError>;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        let mut queue = this.queue.lock().unwrap();
        match queue.transact(this.code, this.encoded) {
            Ok(None) => {
                let id = *this.waiter.get_or_insert_with(|| {
                    queue.next_waiter = queue
                        .next_waiter
                        .checked_add(1)
                        .expect("producer dequeue waiter identity exhausted");
                    queue.next_waiter
                });
                if queue
                    .dequeue_waiters
                    .get(&id)
                    .is_none_or(|waker| !waker.will_wake(context.waker()))
                {
                    queue.dequeue_waiters.insert(id, context.waker().clone());
                }
                Poll::Pending
            }
            result => {
                if let Some(id) = this.waiter.take() {
                    queue.dequeue_waiters.remove(&id);
                }
                Poll::Ready(result.map(|result| result.expect("completed producer transaction")))
            }
        }
    }
}

impl Drop for BinderCall<'_> {
    fn drop(&mut self) {
        if let Some(id) = self.waiter.take() {
            self.queue.lock().unwrap().dequeue_waiters.remove(&id);
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FramebufferError {
    Malformed(&'static str),
    Unsupported(&'static str),
    Backend(Box<str>),
    NvMap(crate::NvMapViewError),
}

impl Display for FramebufferError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Malformed(reason) => write!(formatter, "malformed queued image: {reason}"),
            Self::Unsupported(reason) => {
                write!(formatter, "unsupported presentation semantic: {reason}")
            }
            Self::Backend(reason) => write!(formatter, "GPU presentation export failed: {reason}"),
            Self::NvMap(error) => write!(formatter, "queued nvmap image access failed: {error}"),
        }
    }
}

impl std::error::Error for FramebufferError {}

impl BufferQueue {
    fn new(kernel: &Arc<nixe_scheduler::SchedulerState>, binder_id: i32) -> Self {
        Self {
            connected_api: None,
            abandoned: false,
            default_width: 1,
            default_height: 1,
            default_format: 1,
            sticky_transform: 0,
            next_waiter: 0,
            dequeue_waiters: BTreeMap::new(),
            override_max_buffer_count: 0,
            frame_counter: 0,
            weak_references: 0,
            strong_references: 0,
            slots: BTreeMap::new(),
            buffer_wait_event: BufferQueueWaitEvent::new(kernel, binder_id),
            acquired: None,
            pending: VecDeque::new(),
        }
    }

    fn adjust_refcount(&mut self, add_value: i32, reference_type: i32) -> bool {
        let references = match reference_type {
            0 => &mut self.weak_references,
            1 => &mut self.strong_references,
            _ => return false,
        };
        let Some(updated) = references.checked_add(i64::from(add_value)) else {
            return false;
        };
        if updated < 0 {
            return false;
        }
        *references = updated;
        true
    }

    fn transact(
        &mut self,
        code: u32,
        encoded: &[u8],
    ) -> Result<Option<BinderTransaction>, ParcelError> {
        let mut reader = ParcelReader::decode(encoded)?;
        reader.read_interface_token(IGRAPHIC_BUFFER_PRODUCER)?;
        let mut writer = ParcelWriter::default();
        let mut queued = None;
        match code {
            1 => {
                let slot_index = reader.read_i32()?;
                // RequestBuffer returns presence, the flattened GraphicBuffer,
                // then status. Only a connected producer's dequeued slot is valid.
                // https://android.googlesource.com/platform/frameworks/native/+/778b6f4/libs/gui/IGraphicBufferProducer.cpp
                // https://android.googlesource.com/platform/frameworks/native/+/eb7980c/libs/gui/BufferQueueProducer.cpp
                let slot = self
                    .slots
                    .get(&slot_index)
                    .filter(|slot| slot.ownership == SlotOwnership::Dequeued);
                if self.connected_api.is_none() {
                    writer.write_i32(0);
                    writer.write_i32(-19);
                } else if let Some(slot) = slot {
                    writer.write_i32(1);
                    writer.write_flattened(&slot.flattened_buffer)?;
                    writer.write_i32(0);
                } else {
                    writer.write_i32(0);
                    writer.write_i32(-22);
                }
            }
            2 => {
                let count = reader.read_i32()?;
                let status = if !(0..=MAX_BUFFER_SLOTS).contains(&count)
                    || self.slots.values().any(|slot| {
                        matches!(
                            slot.ownership,
                            SlotOwnership::Dequeued | SlotOwnership::Queueing
                        )
                    }) {
                    -22
                } else {
                    self.override_max_buffer_count = count;
                    self.signal_dequeue_condition();
                    if count != 0 {
                        self.buffer_wait_event.signal();
                    }
                    0
                };
                writer.write_i32(status);
            }
            3 => {
                let _async = reader.read_i32()?;
                let _width = reader.read_u32()?;
                let _height = reader.read_u32()?;
                let _format = reader.read_i32()?;
                let _usage = reader.read_u32()?;
                if self.abandoned || self.connected_api.is_none() {
                    writer.write_i32(-1);
                    writer.write_i32(1);
                    writer.write_flattened(&encode_nv_multi_fence(&NvMultiFence::default()))?;
                    writer.write_i32(-19);
                    return Ok(Some(BinderTransaction {
                        reply: writer.finish()?,
                        queued: None,
                    }));
                }
                let max_count = self.max_buffer_count();
                if self.override_max_buffer_count == 0
                    && self.slots.values().any(|slot| {
                        matches!(
                            slot.ownership,
                            SlotOwnership::Dequeued | SlotOwnership::Queueing
                        )
                    })
                {
                    writer.write_i32(-1);
                    writer.write_i32(1);
                    writer.write_flattened(&encode_nv_multi_fence(&NvMultiFence::default()))?;
                    writer.write_i32(-38);
                    return Ok(Some(BinderTransaction {
                        reply: writer.finish()?,
                        queued: None,
                    }));
                }
                // Reuse the least recently queued free slot, preserving the
                // producer's buffer rotation rather than favoring low indices.
                // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/service/nvnflinger/buffer_queue_producer.cpp#L126-L176
                let free = self
                    .slots
                    .iter_mut()
                    .filter(|(index, slot)| {
                        **index < max_count && slot.ownership == SlotOwnership::Free
                    })
                    .min_by_key(|(_, slot)| slot.frame_number);
                if let Some((&slot_index, slot)) = free {
                    slot.ownership = SlotOwnership::Dequeued;
                    writer.write_i32(slot_index);
                    // Always return a valid fence object, including NoFence.
                    // A reused slot carries the producer's previous GPU fence.
                    writer.write_i32(1);
                    writer.write_flattened(&encode_nv_multi_fence(&std::mem::take(
                        &mut slot.release_fences,
                    )))?;
                    writer.write_i32(0);
                } else {
                    return Ok(None);
                }
            }
            7 => {
                let slot_index = reader.read_i32()?;
                let input = parse_queue_buffer_input(reader.read_flattened()?)?;
                let Some(slot) = self.slots.get_mut(&slot_index) else {
                    return Err(ParcelError::Malformed(
                        "QueueBuffer references an unknown slot",
                    ));
                };
                if slot.ownership != SlotOwnership::Dequeued {
                    return Err(ParcelError::Malformed(
                        "QueueBuffer slot is not producer-owned",
                    ));
                }
                slot.ownership = SlotOwnership::Queueing;
                slot.release_fences = input.acquire_fences;
                queued = Some(QueuedBufferRequest {
                    slot: slot_index,
                    buffer: slot.buffer.clone(),
                    presentation_source: slot.presentation_source.clone(),
                    input,
                });
                self.write_buffer_output(&mut writer);
                writer.write_i32(0);
            }
            8 => {
                let slot_index = reader.read_i32()?;
                let fences = parse_nv_multi_fence(reader.read_flattened()?)?;
                let Some(slot) = self.slots.get_mut(&slot_index) else {
                    return Err(ParcelError::Malformed(
                        "CancelBuffer references an unknown slot",
                    ));
                };
                if slot.ownership != SlotOwnership::Dequeued {
                    return Err(ParcelError::Malformed(
                        "CancelBuffer slot is not producer-owned",
                    ));
                }
                slot.release_fences = fences;
                slot.frame_number = 0;
                slot.ownership = SlotOwnership::Free;
                self.signal_dequeue_condition();
                self.buffer_wait_event.signal();
            }
            9 => {
                let what = reader.read_i32()?;
                let value = match what {
                    0 => Some(self.default_width),
                    1 => Some(self.default_height),
                    2 => Some(self.default_format),
                    // HOS disables the extra async buffer and has minUndequeued=0.
                    3 | 10 => Some(0),
                    9 => Some(u32::from(self.pending_count() > 1)),
                    11 => Some(self.sticky_transform),
                    // This is the pinned producer's DefaultDataSpace query.
                    12 => Some(self.max_buffer_count() as u32),
                    _ => None,
                };
                writer.write_u32(value.unwrap_or(0));
                writer.write_i32(if self.abandoned {
                    -19
                } else if value.is_some() {
                    0
                } else {
                    -22
                });
            }
            10 => {
                let listener_present = reader.read_i32()?;
                let api = reader.read_i32()?;
                let _controlled_by_app = reader.read_i32()?;
                // BufferQueue accepts EGL, CPU, media and camera producer APIs.
                // The connected API must match at Disconnect; it does not select
                // the host rendering backend.
                // https://android.googlesource.com/platform/frameworks/native/+/android-8.1.0_r1/libs/gui/BufferQueueProducer.cpp
                log::trace!(
                    "Binder Connect: listener={listener_present} api={api} connected-api={:?}",
                    self.connected_api
                );
                if listener_present != 0 {
                    return Err(ParcelError::Unsupported(
                        "Binder producer listeners are not implemented",
                    ));
                }
                if !(1..=4).contains(&api) || self.connected_api.is_some() {
                    self.write_buffer_output(&mut writer);
                    writer.write_i32(-22);
                } else {
                    self.connected_api = Some(api);
                    self.write_buffer_output(&mut writer);
                    writer.write_i32(0);
                }
            }
            11 => {
                let api = reader.read_i32()?;
                let status = if !(1..=4).contains(&api) {
                    -22
                } else {
                    match self.connected_api {
                        Some(connected) if connected == api => 0,
                        Some(_) => -22,
                        None => -19,
                    }
                };
                if status == 0 {
                    self.connected_api = None;
                    self.override_max_buffer_count = 0;
                    self.frame_counter = 0;
                    self.acquired = None;
                    self.pending.clear();
                    self.slots.clear();
                    self.signal_dequeue_condition();
                    self.buffer_wait_event.signal();
                }
                writer.write_i32(status);
            }
            14 => {
                let slot_index = reader.read_i32()?;
                if !(0..MAX_BUFFER_SLOTS).contains(&slot_index) {
                    return Err(ParcelError::Malformed(
                        "preallocated buffer slot is out of range",
                    ));
                }
                let has_buffer = reader.read_i32()?;
                if has_buffer == 0 {
                    self.slots.remove(&slot_index);
                } else if has_buffer == 1 {
                    let flattened = reader.read_flattened()?;
                    let buffer = parse_graphic_buffer(flattened)?;
                    self.default_width = buffer.width;
                    self.default_height = buffer.height;
                    self.default_format = buffer.window_format;
                    self.slots.insert(
                        slot_index,
                        BufferSlot {
                            ownership: SlotOwnership::Free,
                            frame_number: 0,
                            buffer,
                            flattened_buffer: flattened.into(),
                            presentation_source: None,
                            release_fences: NvMultiFence::default(),
                        },
                    );
                    self.override_max_buffer_count = self.slots.len() as i32;
                } else {
                    return Err(ParcelError::Malformed(
                        "preallocated buffer presence flag is invalid",
                    ));
                }
                self.pending.retain(|frame| frame.slot != slot_index);
                if self.acquired.as_ref().is_some_and(|a| a.slot == slot_index) {
                    self.acquired = None;
                }
                self.signal_dequeue_condition();
                self.buffer_wait_event.signal();
            }
            _ => {
                return Err(ParcelError::Unsupported(
                    "unsupported IGraphicBufferProducer transaction",
                ));
            }
        }
        match writer.finish() {
            Ok(reply) => Ok(Some(BinderTransaction { reply, queued })),
            Err(error) => {
                if let Some(request) = queued {
                    self.rollback_queue(request.slot);
                }
                Err(error)
            }
        }
    }

    fn pending_count(&self) -> u32 {
        u32::try_from(
            self.slots
                .values()
                .filter(|slot| {
                    matches!(
                        slot.ownership,
                        SlotOwnership::Queueing | SlotOwnership::Queued
                    )
                })
                .count(),
        )
        .unwrap_or(u32::MAX)
    }

    fn max_buffer_count(&self) -> i32 {
        if self.override_max_buffer_count != 0 {
            return self.override_max_buffer_count;
        }
        self.slots
            .iter()
            .filter(|(_, slot)| {
                matches!(
                    slot.ownership,
                    SlotOwnership::Dequeued | SlotOwnership::Queueing | SlotOwnership::Queued
                )
            })
            .map(|(&index, _)| index + 1)
            .max()
            .unwrap_or(2)
            .max(2)
    }

    fn signal_dequeue_condition(&mut self) {
        for (_, waker) in std::mem::take(&mut self.dequeue_waiters) {
            waker.wake();
        }
    }

    fn write_buffer_output(&self, writer: &mut ParcelWriter) {
        writer.write_u32(self.default_width);
        writer.write_u32(self.default_height);
        writer.write_u32(0);
        writer.write_u32(self.pending_count());
    }

    fn abandon(&mut self) {
        self.abandoned = true;
        self.pending.clear();
        self.acquired = None;
        for slot in self.slots.values_mut() {
            if matches!(
                slot.ownership,
                SlotOwnership::Queued | SlotOwnership::Acquired
            ) {
                slot.ownership = SlotOwnership::Free;
                slot.release_fences = NvMultiFence::default();
            }
        }
        self.signal_dequeue_condition();
    }
    fn acquire(&mut self, slot: i32, frame: u64, interval: u32) -> Result<(), FramebufferError> {
        if self.acquired.is_some() {
            return Err(FramebufferError::Malformed(
                "producer already has an acquired framebuffer",
            ));
        }
        let entry = self
            .slots
            .get_mut(&slot)
            .ok_or(FramebufferError::Malformed(
                "acquired framebuffer lost its slot",
            ))?;
        if entry.ownership != SlotOwnership::Queued {
            return Err(FramebufferError::Malformed(
                "framebuffer acquisition requires a queued slot",
            ));
        }
        entry.ownership = SlotOwnership::Acquired;
        self.acquired = Some(AcquiredBuffer {
            slot,
            release_frame: frame.saturating_add(u64::from(interval)),
        });
        self.signal_dequeue_condition();
        Ok(())
    }
    fn release_acquired(&mut self, frame: u64) -> Option<i32> {
        let acquired = self.acquired.as_ref()?;
        if acquired.release_frame > frame {
            return None;
        }
        let slot_index = self.acquired.take().unwrap().slot;
        let slot = self
            .slots
            .get_mut(&slot_index)
            .expect("acquired framebuffer slot");
        assert_eq!(slot.ownership, SlotOwnership::Acquired);
        slot.ownership = SlotOwnership::Free;
        // The pinned consumer retains the queued fence here. Dequeue returns
        // it to the producer, then resets the slot's fence to NoFence.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/core/hle/service/nvnflinger/buffer_queue_consumer.cpp#L145-L150
        self.signal_dequeue_condition();
        Some(slot_index)
    }

    fn commit_queue(
        &mut self,
        slot_index: i32,
        presentation_source: Option<PresentationImageRequest>,
        sticky_transform: u32,
    ) -> bool {
        let Some(slot) = self.slots.get_mut(&slot_index) else {
            return false;
        };
        if slot.ownership != SlotOwnership::Queueing {
            return false;
        }
        if presentation_source.is_some() {
            slot.presentation_source = presentation_source;
        }
        self.frame_counter += 1;
        slot.frame_number = self.frame_counter;
        slot.ownership = SlotOwnership::Queued;
        self.sticky_transform = sticky_transform;
        self.signal_dequeue_condition();
        true
    }

    fn rollback_queue(&mut self, slot_index: i32) {
        if let Some(slot) = self.slots.get_mut(&slot_index)
            && slot.ownership == SlotOwnership::Queueing
        {
            slot.ownership = SlotOwnership::Dequeued;
        }
    }
}

fn parse_graphic_buffer(bytes: &[u8]) -> Result<GraphicBuffer, ParcelError> {
    const PREFIX_WORDS: usize = 10;
    const SURFACE_WORDS: usize = 22;
    if bytes.len() < PREFIX_WORDS * 4 {
        return Err(ParcelError::Malformed(
            "flattened graphic buffer is truncated",
        ));
    }
    let word = |index: usize| -> Result<u32, ParcelError> {
        let offset = index
            .checked_mul(4)
            .ok_or(ParcelError::Malformed("graphic buffer offset overflows"))?;
        Ok(u32::from_le_bytes(
            bytes
                .get(offset..offset + 4)
                .ok_or(ParcelError::Malformed("graphic buffer is truncated"))?
                .try_into()
                .unwrap(),
        ))
    };
    if word(0)? != 0x4742_4652 || word(8)? != 0 {
        return Err(ParcelError::Malformed(
            "flattened graphic buffer header is invalid",
        ));
    }
    let num_ints = usize::try_from(word(9)?)
        .map_err(|_| ParcelError::Malformed("graphic buffer integer count overflows"))?;
    if PREFIX_WORDS
        .checked_add(num_ints)
        .and_then(|words| words.checked_mul(4))
        .is_none_or(|size| size > bytes.len())
    {
        return Err(ParcelError::Malformed(
            "graphic buffer integer payload is truncated",
        ));
    }
    let int = |index: usize| word(PREFIX_WORDS + index);
    if int(3)? != 0xdaff_caff {
        return Err(ParcelError::Unsupported(
            "NvGraphicBuffer metadata is unsupported",
        ));
    }
    let plane_count = usize::try_from(int(11)?)
        .map_err(|_| ParcelError::Malformed("graphic buffer plane count overflows"))?;
    if !(1..=3).contains(&plane_count)
        || 13_usize
            .checked_add(plane_count.saturating_mul(SURFACE_WORDS))
            .is_none_or(|required| required > num_ints)
    {
        return Err(ParcelError::Unsupported(
            "NvGraphicBuffer plane metadata is unsupported",
        ));
    }
    let mut planes = Vec::with_capacity(plane_count);
    for plane_index in 0..plane_count {
        let base = 13 + plane_index * SURFACE_WORDS;
        planes.push(GraphicBufferPlane {
            width: int(base)?,
            height: int(base + 1)?,
            color_format: u64::from(int(base + 2)?) | (u64::from(int(base + 3)?) << 32),
            layout: int(base + 4)?,
            pitch: int(base + 5)?,
            offset: int(base + 7)?,
            kind: int(base + 8)?,
            block_height_log2: int(base + 9)?,
            scan_format: int(base + 10)?,
            second_field_offset: int(base + 11)?,
            flags: u64::from(int(base + 12)?) | (u64::from(int(base + 13)?) << 32),
            size: u64::from(int(base + 14)?) | (u64::from(int(base + 15)?) << 32),
        });
    }
    Ok(GraphicBuffer {
        width: word(1)?,
        height: word(2)?,
        window_format: word(4)?,
        nvmap_id: int(1)?,
        stride_pixels: int(9)?,
        format: int(7)?,
        external_format: int(8)?,
        usage: int(6)?,
        total_size: int(10)?,
        planes: planes.into_boxed_slice(),
    })
}

fn parse_queue_buffer_input(bytes: &[u8]) -> Result<QueueBufferInput, ParcelError> {
    // Exact BqBufferInput/NvMultiFence layout used by pinned libnx:
    // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/services/buffer_producer.h
    const REQUIRED_SIZE: usize = 84;
    if bytes.len() < REQUIRED_SIZE {
        return Err(ParcelError::Malformed("QueueBuffer input is truncated"));
    }
    let u32_at = |offset: usize| u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
    let i32_at = |offset: usize| i32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
    let acquire_fences = parse_nv_multi_fence(&bytes[48..84])?;
    nixe_memory::trace::event(
        "queue_options",
        u64::from(u32_at(40)),
        u64::from(u32_at(8)),
        u64::from_le_bytes(bytes[0..8].try_into().unwrap()),
    );
    Ok(QueueBufferInput {
        timestamp: i64::from_le_bytes(bytes[0..8].try_into().unwrap()),
        auto_timestamp: i32_at(8) != 0,
        crop: CropRect {
            left: i32_at(12),
            top: i32_at(16),
            right: i32_at(20),
            bottom: i32_at(24),
        },
        scaling_mode: u32_at(28),
        transform: u32_at(32),
        sticky_transform: u32_at(36),
        swap_interval: u32_at(44),
        acquire_fences,
    })
}

fn parse_nv_multi_fence(bytes: &[u8]) -> Result<NvMultiFence, ParcelError> {
    if bytes.len() < 36 {
        return Err(ParcelError::Malformed("NvMultiFence is truncated"));
    }
    let u32_at = |offset: usize| u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
    let count = usize::try_from(u32_at(0))
        .map_err(|_| ParcelError::Malformed("NvMultiFence count overflows"))?;
    if count > 4 {
        return Err(ParcelError::Malformed(
            "NvMultiFence contains too many fences",
        ));
    }
    let mut result = NvMultiFence {
        count,
        ..NvMultiFence::default()
    };
    for (index, fence) in result.entries[..count].iter_mut().enumerate() {
        let offset = 4 + index * 8;
        fence.point = GuestTimelinePoint::new(
            GuestSyncpointId::new(u32_at(offset)),
            GuestSyncpointValue::new(u32_at(offset + 4)),
        );
    }
    Ok(result)
}

fn encode_nv_multi_fence(fences: &[NvFence]) -> [u8; 36] {
    debug_assert!(fences.len() <= 4);
    let mut bytes = [0_u8; 36];
    bytes[..4].copy_from_slice(&u32::try_from(fences.len()).unwrap().to_le_bytes());
    for (index, fence) in fences.iter().enumerate() {
        let offset = 4 + index * 8;
        bytes[offset..offset + 4].copy_from_slice(&fence.point.syncpoint().get().to_le_bytes());
        bytes[offset + 4..offset + 8].copy_from_slice(&fence.point.value().get().to_le_bytes());
    }
    bytes
}

fn validate_present_descriptor(
    buffer: &GraphicBuffer,
    crop: Option<CropRect>,
) -> Result<(), FramebufferError> {
    const MAX_DIMENSION: u32 = 8192;
    if buffer.planes.len() != 1 {
        return Err(FramebufferError::Unsupported(
            "multi-plane presentation images are not implemented",
        ));
    }
    let plane = buffer.primary_plane();
    let bytes_per_pixel = present_format(buffer, plane)?.bytes_per_pixel();
    let required_size = required_present_plane_size(plane, bytes_per_pixel);
    if plane.width == 0
        || plane.height == 0
        || plane.width > MAX_DIMENSION
        || plane.height > MAX_DIMENSION
        || plane.block_height_log2 > 5
        || buffer.stride_pixels < plane.width
        || plane.pitch < plane.width.saturating_mul(bytes_per_pixel)
        || (plane.layout == 3
            && (!plane.pitch.is_multiple_of(64)
                || !(plane.kind == 0xfe || (plane.kind == 0xdb && bytes_per_pixel == 4))))
        || (plane.layout != 1 && plane.layout != 3)
        || required_size.is_none_or(|required| required > plane.size)
    {
        return Err(FramebufferError::Malformed(
            "queued image dimensions or memory layout are invalid",
        ));
    }
    if let Some(crop) = crop {
        validate_crop(crop, plane.width, plane.height)?;
    }
    Ok(())
}

fn required_present_plane_size(plane: &GraphicBufferPlane, bytes_per_pixel: u32) -> Option<u64> {
    let row_bytes = u64::from(plane.width).checked_mul(u64::from(bytes_per_pixel))?;
    if plane.layout == 1 {
        return u64::from(plane.height.checked_sub(1)?)
            .checked_mul(u64::from(plane.pitch))?
            .checked_add(row_bytes);
    }
    let block_height_gobs = 1_u64.checked_shl(plane.block_height_log2)?;
    let block_rows = 8_u64.checked_mul(block_height_gobs)?;
    let block_row_count = u64::from(plane.height).div_ceil(block_rows);
    block_row_count
        .checked_mul(u64::from(plane.pitch) / 64)?
        .checked_mul(512)?
        .checked_mul(block_height_gobs)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PresentFormat {
    Rgba8,
    Rgbx8,
    Bgra8,
    Rgb565,
    Rgba4444,
}

impl PresentFormat {
    const fn bytes_per_pixel(self) -> u32 {
        match self {
            Self::Rgba8 | Self::Rgbx8 | Self::Bgra8 => 4,
            Self::Rgb565 | Self::Rgba4444 => 2,
        }
    }
}

fn present_format(
    buffer: &GraphicBuffer,
    plane: &GraphicBufferPlane,
) -> Result<PresentFormat, FramebufferError> {
    // Android/NvColor values from pinned libnx nvidia/types.h and
    // framebuffer.c:
    // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/nvidia/types.h
    match (buffer.format, plane.color_format) {
        (1, 0x0100_532120) => Ok(PresentFormat::Rgba8),
        (2, 0x010a_532120) => Ok(PresentFormat::Rgbx8),
        (5, 0x0100_d12120) => Ok(PresentFormat::Bgra8),
        (4, 0x010a_881210) => Ok(PresentFormat::Rgb565),
        (7, 0x0100_531510) => Ok(PresentFormat::Rgba4444),
        _ => Err(FramebufferError::Unsupported(
            "queued image pixel format has no verified presentation conversion",
        )),
    }
}

fn presentable_image_request(
    buffer: &GraphicBuffer,
    object: &NvMapObject,
    nvdrv: &NvDrvSession,
) -> Result<PresentationImageRequest, FramebufferError> {
    let plane = buffer.primary_plane();
    // C32_2CRA may be scanned out from its rendered representation. Its plane
    // size includes compression metadata/alignment, outside the texel footprint.
    // Never reinterpret those opaque bytes as an ordinary block-linear import.
    // https://github.com/NVIDIA/open-gpu-kernel-modules/blob/580.95.05/src/common/inc/swref/published/maxwell/gm107/dev_mmu.h#L99
    let allow_canonical_import = plane.kind != 0xdb;
    let pixel_size = if allow_canonical_import {
        plane.size
    } else {
        required_present_plane_size(plane, 4).ok_or(FramebufferError::Malformed(
            "queued image storage size overflows",
        ))?
    };
    let format = match present_format(buffer, plane)? {
        PresentFormat::Rgba8 => PresentationImageFormat::Rgba8,
        PresentFormat::Rgbx8 => PresentationImageFormat::Rgbx8,
        PresentFormat::Bgra8 => PresentationImageFormat::Bgra8,
        PresentFormat::Rgb565 => PresentationImageFormat::Rgb565,
        PresentFormat::Rgba4444 => PresentationImageFormat::Rgba4444,
    };
    let layout = if plane.layout == 1 {
        ImageMemoryLayout::PitchLinear {
            row_pitch: u64::from(plane.pitch),
            layer_stride: pixel_size,
        }
    } else {
        ImageMemoryLayout::BlockLinear(BlockLinearLayout {
            block_width_log2: 0,
            block_height_log2: u8::try_from(plane.block_height_log2)
                .expect("validated block height fits u8"),
            block_depth_log2: 0,
            layer_stride: pixel_size,
        })
    };
    let mapping = nvdrv
        .pin_framebuffer(object)
        .map_err(|error| FramebufferError::Backend(error.to_string().into()))?;
    Ok(PresentationImageRequest {
        crop: FrameCrop {
            left: 0,
            top: 0,
            width: plane.width,
            height: plane.height,
        },
        transform: FrameTransform::default(),
        allow_canonical_import,
        mapping,
        offset: u64::from(plane.offset),
        size: pixel_size,
        width: plane.width,
        height: plane.height,
        format,
        layout,
        row_pitch: plane.pitch,
    })
}

fn frame_crop(crop: CropRect) -> Result<FrameCrop, FramebufferError> {
    Ok(FrameCrop {
        left: u32::try_from(crop.left)
            .map_err(|_| FramebufferError::Malformed("QueueBuffer crop left overflows"))?,
        top: u32::try_from(crop.top)
            .map_err(|_| FramebufferError::Malformed("QueueBuffer crop top overflows"))?,
        width: u32::try_from(crop.right - crop.left)
            .map_err(|_| FramebufferError::Malformed("QueueBuffer crop width overflows"))?,
        height: u32::try_from(crop.bottom - crop.top)
            .map_err(|_| FramebufferError::Malformed("QueueBuffer crop height overflows"))?,
    })
}

fn frame_transform(transform: u32) -> Result<FrameTransform, FramebufferError> {
    if transform & !0x7 != 0 {
        return Err(FramebufferError::Unsupported(
            "QueueBuffer transform contains unsupported bits",
        ));
    }
    Ok(FrameTransform {
        flip_horizontal: transform & 0x1 != 0,
        flip_vertical: transform & 0x2 != 0,
        rotate_90_clockwise: transform & 0x4 != 0,
    })
}

fn validate_crop(crop: CropRect, width: u32, height: u32) -> Result<(), FramebufferError> {
    if crop.left < 0
        || crop.top < 0
        || crop.right <= crop.left
        || crop.bottom <= crop.top
        || u32::try_from(crop.right).map_or(true, |right| right > width)
        || u32::try_from(crop.bottom).map_or(true, |bottom| bottom > height)
    {
        return Err(FramebufferError::Malformed("QueueBuffer crop is invalid"));
    }
    Ok(())
}

fn effective_crop(crop: CropRect, width: u32, height: u32) -> CropRect {
    // libnx zero-initializes NWindow crop state and forwards it unchanged;
    // this sentinel means that the consumer uses the complete buffer:
    // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/display/native_window.c
    if crop
        == (CropRect {
            left: 0,
            top: 0,
            right: 0,
            bottom: 0,
        })
    {
        CropRect {
            left: 0,
            top: 0,
            right: i32::try_from(width).unwrap_or(i32::MAX),
            bottom: i32::try_from(height).unwrap_or(i32::MAX),
        }
    } else {
        crop
    }
}

pub(crate) fn encode_native_window(binder_id: i32) -> Vec<u8> {
    // NativeWindow contains a complete flattened Binder handle and an object
    // offset table. Reading only the ID at payload offset 8 (as libnx does)
    // misses the fields required by Android Parcel::readStrongBinder.
    // https://switchbrew.org/wiki/Display_services#NativeWindow
    let mut parcel = vec![0_u8; 0x3c];
    parcel[0..4].copy_from_slice(&0x28_u32.to_le_bytes());
    parcel[4..8].copy_from_slice(&16_u32.to_le_bytes());
    parcel[8..12].copy_from_slice(&4_u32.to_le_bytes());
    parcel[12..16].copy_from_slice(&0x38_u32.to_le_bytes());
    parcel[16..20].copy_from_slice(&2_u32.to_le_bytes()); // Remote Binder handle.
    parcel[24..28].copy_from_slice(&binder_id.to_le_bytes());
    parcel[40..48].copy_from_slice(b"dispdrv\0");
    // The sole object table entry at 0x38 is zero: the payload-relative offset
    // of the flattened Binder handle. All reserved fields remain zero.
    parcel
}
