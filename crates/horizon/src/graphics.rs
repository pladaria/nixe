//! Switch display-service state shared by VI, Binder, and nvdrv sessions.

use std::collections::{BTreeMap, VecDeque};
use std::fmt::{Display, Formatter};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use nixe_gpu::{GuestSyncpointId, GuestSyncpointValue, GuestTimelinePoint, NeutralBackendRuntime};
use nixe_runtime::{EventObject, ExternalEventSource, ReadableEventObject, WritableEventObject};
use nixe_video::{DisplayClock, Frame, FrameMailbox};

use crate::parcel::{ParcelError, ParcelReader, ParcelWriter};
use crate::{
    GraphicsEventSource, NvDrvSession, NvMapExportedId, NvMapImageView, NvMapImageViewMetadata,
    NvMapPlaneMetadata,
};

const DEFAULT_DISPLAY_ID: u64 = 1;
const DEFAULT_WIDTH: u32 = 1280;
const DEFAULT_HEIGHT: u32 = 720;

/// VI refresh event, kept type-distinct from queue and GPU events even though
/// all three currently use runtime kernel-event primitives.
#[derive(Clone, Debug)]
struct ViVsyncEvent {
    source: GraphicsEventSource,
    writable: WritableEventObject,
    readable: ReadableEventObject,
}

impl ViVsyncEvent {
    fn new(display_id: u64) -> Self {
        let (writable, readable) =
            EventObject::create_pair_with_source(ExternalEventSource::Display);
        Self {
            source: GraphicsEventSource::ViVsync { display_id },
            writable,
            readable,
        }
    }

    fn signal(&self) {
        self.writable.signal();
    }

    fn readable(&self) -> ReadableEventObject {
        debug_assert!(matches!(self.source, GraphicsEventSource::ViVsync { .. }));
        self.readable.clone()
    }
}

/// BufferQueue slot-availability event, never reused as a VI or GPU event.
#[derive(Clone, Debug)]
struct BufferQueueAvailabilityEvent {
    source: GraphicsEventSource,
    writable: WritableEventObject,
    readable: ReadableEventObject,
}

impl BufferQueueAvailabilityEvent {
    fn new(binder_id: i32) -> Self {
        let (writable, readable) =
            EventObject::create_pair_with_source(ExternalEventSource::Display);
        Self {
            source: GraphicsEventSource::BufferQueueAvailability { binder_id },
            writable,
            readable,
        }
    }

    fn signal(&self) {
        self.writable.signal();
    }

    fn clear(&self) {
        self.writable.clear();
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
}

impl ViSession {
    pub(crate) fn new(kind: ViObjectKind, video: VideoSystem) -> Self {
        Self { kind, video }
    }

    pub(crate) const fn kind(&self) -> ViObjectKind {
        self.kind
    }

    pub(crate) fn video(&self) -> &VideoSystem {
        &self.video
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
    queues: BTreeMap<i32, BufferQueue>,
    vsync_event: ViVsyncEvent,
    display_clock: DisplayClock,
    mailbox: FrameMailbox,
    nvdrv: NvDrvSession,
    next_frame_sequence: u64,
    pending_frames: VecDeque<PendingFrame>,
}

/// Cloneable process display state shared across independently opened services.
#[derive(Clone, Debug)]
pub struct VideoSystem {
    state: Arc<Mutex<VideoState>>,
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

impl Default for VideoSystem {
    fn default() -> Self {
        Self::new(FrameMailbox::default())
    }
}

impl VideoSystem {
    #[must_use]
    pub fn new(mailbox: FrameMailbox) -> Self {
        Self::new_with_gpu_backend(mailbox, None)
    }

    /// Constructs the process graphics system with a composition-root-selected
    /// neutral GPU backend. Horizon never observes its concrete host API.
    #[must_use]
    pub fn with_gpu_backend(
        mailbox: FrameMailbox,
        backend: Box<dyn NeutralBackendRuntime>,
    ) -> Self {
        Self::new_with_gpu_backend(mailbox, Some(backend))
    }

    fn new_with_gpu_backend(
        mailbox: FrameMailbox,
        backend: Option<Box<dyn NeutralBackendRuntime>>,
    ) -> Self {
        Self {
            state: Arc::new(Mutex::new(VideoState {
                next_layer_id: 1,
                layers: BTreeMap::new(),
                queues: BTreeMap::new(),
                vsync_event: ViVsyncEvent::new(DEFAULT_DISPLAY_ID),
                display_clock: DisplayClock::new(60)
                    .expect("the fixed Switch display refresh is non-zero"),
                mailbox,
                nvdrv: backend.map_or_else(NvDrvSession::new, NvDrvSession::with_gpu_backend),
                next_frame_sequence: 1,
                pending_frames: VecDeque::new(),
            })),
        }
    }

    #[must_use]
    pub fn mailbox(&self) -> FrameMailbox {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .mailbox
            .clone()
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
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let nvdrv = state.nvdrv.teardown();
        let report = GraphicsTeardownReport {
            layers_released: state.layers.len(),
            queues_released: state.queues.len(),
            pending_frames_released: state.pending_frames.len(),
            device_fds_released: nvdrv.device_fds_released,
            allocations_released: nvdrv.allocations_released,
        };
        state.layers.clear();
        state.queues.clear();
        state.pending_frames.clear();
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
        state.queues.insert(binder_id, BufferQueue::new(binder_id));
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
        state.queues.remove(&layer.binder_id);
        state
            .pending_frames
            .retain(|pending| pending.binder_id != layer.binder_id);
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

    pub(crate) fn vsync_event(&self, display_id: u64) -> Option<ReadableEventObject> {
        Self::display_resolution(display_id)?;
        Some(
            self.state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .vsync_event
                .readable(),
        )
    }

    pub(crate) fn binder_event(&self, binder_id: i32) -> Option<ReadableEventObject> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .queues
            .get(&binder_id)
            .map(|queue| queue.available_event.readable())
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
        state
            .queues
            .get_mut(&binder_id)
            .is_some_and(|queue| queue.adjust_refcount(add_value, reference_type))
    }

    pub(crate) fn transact_binder(
        &self,
        binder_id: i32,
        code: u32,
        encoded: &[u8],
    ) -> Result<BinderTransaction, ParcelError> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let queue = state
            .queues
            .get_mut(&binder_id)
            .ok_or(ParcelError::Malformed("unknown Binder producer object"))?;
        queue.transact(code, encoded)
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
        let reserved_slot = request.slot;
        let result = (|| {
            let primary = request.buffer.primary_plane();
            let crop = effective_crop(request.input.crop, primary.width, primary.height);
            validate_present_descriptor(&request.buffer, Some(crop))?;
            transformed_dimensions(crop, request.input.transform)?;
            let nvdrv = self.nvdrv();
            for fence in &request.input.acquire_fences {
                if nvdrv.guest_timeline_point_reached(fence.point).is_none() {
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
            let view = object
                .image_view(request.buffer.nvmap_view_metadata())
                .map_err(FramebufferError::NvMap)?;

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
            if !queue.commit_queue(request.slot) {
                return Err(FramebufferError::Malformed(
                    "QueueBuffer reservation lost slot ownership",
                ));
            }
            state.pending_frames.push_back(PendingFrame {
                binder_id,
                slot: request.slot,
                source: PendingFrameSource::GuestImage {
                    buffer: request.buffer,
                    input: request.input,
                    view,
                },
            });
            Ok(())
        })();

        if result.is_err() {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(queue) = state.queues.get_mut(&binder_id) {
                queue.rollback_queue(reserved_slot);
            }
        }
        result
    }

    /// Advances guest display timing independently from host monitor VSync.
    pub fn advance(&self, elapsed: Duration) -> Result<u64, FramebufferError> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let ticks = state.display_clock.advance(elapsed);
        if ticks.crossed != 0 {
            state.vsync_event.signal();
        }
        for _ in 0..ticks.crossed {
            let Some(pending) = state.pending_frames.front().cloned() else {
                break;
            };
            let frame = match &pending.source {
                PendingFrameSource::GuestImage {
                    buffer,
                    input,
                    view,
                } => {
                    if !input.acquire_fences.iter().all(|fence| {
                        state
                            .nvdrv
                            .guest_timeline_point_reached(fence.point)
                            .unwrap_or(false)
                    }) {
                        break;
                    }
                    let bytes = view.read_plane(0).map_err(FramebufferError::NvMap)?;
                    let primary = buffer.primary_plane();
                    let crop = effective_crop(input.crop, primary.width, primary.height);
                    let pixels =
                        decode_present_image(buffer, &bytes, Some((crop, input.transform)))?;
                    let dimensions = transformed_dimensions(crop, input.transform)?;
                    let sequence = state.next_frame_sequence;
                    state.next_frame_sequence = state.next_frame_sequence.saturating_add(1);
                    Arc::new(
                        Frame::new_xrgb8888(dimensions.0, dimensions.1, sequence, pixels).map_err(
                            |_| {
                                FramebufferError::Malformed("composed frame dimensions are invalid")
                            },
                        )?,
                    )
                }
            };
            state.pending_frames.pop_front();
            state.mailbox.publish(frame);
            log::debug!(
                "latched frame from Binder producer {} slot {} at VSync {}",
                pending.binder_id,
                pending.slot,
                ticks.latest_sequence
            );
            if let Some(queue) = state.queues.get_mut(&pending.binder_id) {
                let _ = queue.release(pending.slot);
            }
        }
        Ok(ticks.crossed)
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

#[derive(Clone, Debug, Eq, PartialEq)]
struct QueueBufferInput {
    timestamp: i64,
    auto_timestamp: bool,
    crop: CropRect,
    scaling_mode: u32,
    transform: u32,
    sticky_transform: u32,
    swap_interval: u32,
    acquire_fences: Box<[NvFence]>,
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
    input: QueueBufferInput,
}

#[derive(Clone, Debug)]
struct BufferSlot {
    ownership: SlotOwnership,
    buffer: GraphicBuffer,
    release_fences: Box<[NvFence]>,
}

#[derive(Debug)]
struct BufferQueue {
    connected: bool,
    weak_references: i64,
    strong_references: i64,
    slots: BTreeMap<i32, BufferSlot>,
    available_event: BufferQueueAvailabilityEvent,
}

#[derive(Clone, Debug)]
pub(crate) struct BinderTransaction {
    pub(crate) reply: Vec<u8>,
    pub(crate) queued: Option<QueuedBufferRequest>,
}

#[derive(Clone, Debug)]
struct PendingFrame {
    binder_id: i32,
    slot: i32,
    source: PendingFrameSource,
}

#[derive(Clone, Debug)]
enum PendingFrameSource {
    GuestImage {
        buffer: GraphicBuffer,
        input: QueueBufferInput,
        view: NvMapImageView,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FramebufferError {
    Malformed(&'static str),
    Unsupported(&'static str),
    NvMap(crate::NvMapViewError),
}

impl Display for FramebufferError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Malformed(reason) => write!(formatter, "malformed queued image: {reason}"),
            Self::Unsupported(reason) => {
                write!(formatter, "unsupported presentation semantic: {reason}")
            }
            Self::NvMap(error) => write!(formatter, "queued nvmap image access failed: {error}"),
        }
    }
}

impl std::error::Error for FramebufferError {}

impl BufferQueue {
    fn new(binder_id: i32) -> Self {
        Self {
            connected: false,
            weak_references: 0,
            strong_references: 0,
            slots: BTreeMap::new(),
            available_event: BufferQueueAvailabilityEvent::new(binder_id),
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

    fn transact(&mut self, code: u32, encoded: &[u8]) -> Result<BinderTransaction, ParcelError> {
        let mut reader = ParcelReader::decode(encoded)?;
        reader.read_interface_token(IGRAPHIC_BUFFER_PRODUCER)?;
        let mut writer = ParcelWriter::default();
        let mut queued = None;
        match code {
            1 => {
                let slot = reader.read_i32()?;
                let status = if self.slots.contains_key(&slot) {
                    0
                } else {
                    -75
                };
                writer.write_i32(0);
                writer.write_i32(status);
            }
            3 => {
                let _async = reader.read_i32()?;
                let _width = reader.read_u32()?;
                let _height = reader.read_u32()?;
                let _format = reader.read_i32()?;
                let _usage = reader.read_u32()?;
                let free = self
                    .slots
                    .iter_mut()
                    .find(|(_, slot)| slot.ownership == SlotOwnership::Free);
                if let Some((&slot_index, slot)) = free {
                    slot.ownership = SlotOwnership::Dequeued;
                    writer.write_i32(slot_index);
                    // Android's producer protocol makes the fence optional, but
                    // libnx leaves its local NvMultiFence uninitialized when it
                    // is absent. VI therefore always supplies a valid empty one.
                    writer.write_i32(1);
                    writer.write_flattened(&encode_nv_multi_fence(&slot.release_fences))?;
                    writer.write_i32(0);
                    if !self
                        .slots
                        .values()
                        .any(|slot| slot.ownership == SlotOwnership::Free)
                    {
                        self.available_event.clear();
                    }
                } else {
                    writer.write_i32(-1);
                    writer.write_i32(0);
                    writer.write_i32(-11);
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
                queued = Some(QueuedBufferRequest {
                    slot: slot_index,
                    buffer: slot.buffer.clone(),
                    input,
                });
                write_buffer_output(&mut writer, self.pending_count());
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
                slot.ownership = SlotOwnership::Free;
                self.available_event.signal();
            }
            9 => {
                let _what = reader.read_i32()?;
                writer.write_i32(0);
                writer.write_i32(0);
            }
            10 => {
                let listener_present = reader.read_i32()?;
                let api = reader.read_i32()?;
                let _controlled_by_app = reader.read_i32()?;
                if listener_present != 0 || api != 2 || self.connected {
                    write_buffer_output(&mut writer, self.pending_count());
                    writer.write_i32(-22);
                } else {
                    self.connected = true;
                    write_buffer_output(&mut writer, self.pending_count());
                    writer.write_i32(0);
                    self.update_availability();
                }
            }
            11 => {
                let api = reader.read_i32()?;
                let status = if api == 2 && self.connected { 0 } else { -22 };
                if status == 0 {
                    self.connected = false;
                    self.slots.clear();
                    self.available_event.clear();
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
                    if self.slots.contains_key(&slot_index) {
                        return Err(ParcelError::Malformed(
                            "preallocated buffer slot already exists",
                        ));
                    }
                    self.slots.insert(
                        slot_index,
                        BufferSlot {
                            ownership: SlotOwnership::Free,
                            buffer,
                            release_fences: Box::default(),
                        },
                    );
                } else {
                    return Err(ParcelError::Malformed(
                        "preallocated buffer presence flag is invalid",
                    ));
                }
                self.update_availability();
            }
            _ => {
                return Err(ParcelError::Unsupported(
                    "unsupported IGraphicBufferProducer transaction",
                ));
            }
        }
        match writer.finish() {
            Ok(reply) => Ok(BinderTransaction { reply, queued }),
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

    fn update_availability(&self) {
        if self
            .slots
            .values()
            .any(|slot| slot.ownership == SlotOwnership::Free)
        {
            self.available_event.signal();
        } else {
            self.available_event.clear();
        }
    }

    fn release(&mut self, slot_index: i32) -> bool {
        let Some(slot) = self.slots.get_mut(&slot_index) else {
            return false;
        };
        if slot.ownership != SlotOwnership::Queued {
            return false;
        }
        slot.ownership = SlotOwnership::Free;
        // Composition is complete synchronously before this transition, so no
        // future guest GPU point is required for the next producer dequeue.
        slot.release_fences = Box::default();
        self.available_event.signal();
        true
    }

    fn commit_queue(&mut self, slot_index: i32) -> bool {
        let Some(slot) = self.slots.get_mut(&slot_index) else {
            return false;
        };
        if slot.ownership != SlotOwnership::Queueing {
            return false;
        }
        slot.ownership = SlotOwnership::Queued;
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

fn write_buffer_output(writer: &mut ParcelWriter, pending: u32) {
    writer.write_u32(DEFAULT_WIDTH);
    writer.write_u32(DEFAULT_HEIGHT);
    writer.write_u32(0);
    writer.write_u32(pending);
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

fn parse_nv_multi_fence(bytes: &[u8]) -> Result<Box<[NvFence]>, ParcelError> {
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
    Ok((0..count)
        .map(|index| {
            let offset = 4 + index * 8;
            NvFence {
                point: GuestTimelinePoint::new(
                    GuestSyncpointId::new(u32_at(offset)),
                    GuestSyncpointValue::new(u32_at(offset + 4)),
                ),
            }
        })
        .collect::<Vec<_>>()
        .into_boxed_slice())
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
    if plane.width == 0
        || plane.height == 0
        || plane.width > MAX_DIMENSION
        || plane.height > MAX_DIMENSION
        || plane.block_height_log2 > 5
        || buffer.stride_pixels < plane.width
        || plane.pitch < plane.width.saturating_mul(bytes_per_pixel)
        || (plane.layout == 3 && (!plane.pitch.is_multiple_of(64) || plane.kind != 0xfe))
        || (plane.layout != 1 && plane.layout != 3)
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

fn transformed_dimensions(crop: CropRect, transform: u32) -> Result<(u32, u32), FramebufferError> {
    // HAL applies horizontal/vertical flips first and ROT_90 clockwise after
    // them. These exact bits and ordering are recorded by pinned libnx:
    // https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/display/types.h
    if transform & !0x7 != 0 {
        return Err(FramebufferError::Unsupported(
            "QueueBuffer transform contains unsupported bits",
        ));
    }
    let width = u32::try_from(crop.right - crop.left)
        .map_err(|_| FramebufferError::Malformed("QueueBuffer crop width overflows"))?;
    let height = u32::try_from(crop.bottom - crop.top)
        .map_err(|_| FramebufferError::Malformed("QueueBuffer crop height overflows"))?;
    Ok(if transform & 0x4 != 0 {
        (height, width)
    } else {
        (width, height)
    })
}

fn decode_present_image(
    buffer: &GraphicBuffer,
    bytes: &[u8],
    presentation: Option<(CropRect, u32)>,
) -> Result<Vec<u32>, FramebufferError> {
    validate_present_descriptor(buffer, presentation.map(|value| value.0))?;
    let plane = buffer.primary_plane();
    let format = present_format(buffer, plane)?;
    let crop = presentation.map_or(
        CropRect {
            left: 0,
            top: 0,
            right: i32::try_from(plane.width).unwrap_or(i32::MAX),
            bottom: i32::try_from(plane.height).unwrap_or(i32::MAX),
        },
        |value| value.0,
    );
    let transform = presentation.map_or(0, |value| value.1);
    let dimensions = transformed_dimensions(crop, transform)?;
    let crop_width = u32::try_from(crop.right - crop.left).unwrap();
    let crop_height = u32::try_from(crop.bottom - crop.top).unwrap();
    let pixel_count = usize::try_from(dimensions.0)
        .ok()
        .and_then(|width| {
            usize::try_from(dimensions.1)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .ok_or(FramebufferError::Malformed(
            "queued image dimensions overflow",
        ))?;
    let mut pixels = vec![0_u32; pixel_count];
    let width_in_gobs = u64::from(plane.pitch / 64);
    let block_height_gobs = 1_u64 << plane.block_height_log2;
    for source_y in 0..crop_height {
        for source_x in 0..crop_width {
            let x = u64::from(u32::try_from(crop.left).unwrap() + source_x);
            let y = u64::from(u32::try_from(crop.top).unwrap() + source_y);
            let byte_x = x * u64::from(format.bytes_per_pixel());
            let address = if plane.layout == 1 {
                y * u64::from(plane.pitch) + byte_x
            } else {
                (y / (8 * block_height_gobs)) * 512 * block_height_gobs * width_in_gobs
                    + (byte_x / 64) * 512 * block_height_gobs
                    + ((y % (8 * block_height_gobs)) / 8) * 512
                    + ((byte_x % 64) / 32) * 256
                    + ((y % 8) / 2) * 64
                    + ((byte_x % 32) / 16) * 32
                    + (y % 2) * 16
                    + byte_x % 16
            };
            let address = usize::try_from(address)
                .map_err(|_| FramebufferError::Malformed("image address overflows"))?;
            let needed = usize::try_from(format.bytes_per_pixel()).unwrap();
            let texel = bytes
                .get(address..address + needed)
                .ok_or(FramebufferError::Malformed(
                    "queued image backing is truncated",
                ))?;
            let (red, green, blue) = match format {
                PresentFormat::Rgba8 | PresentFormat::Rgbx8 => (
                    u32::from(texel[0]),
                    u32::from(texel[1]),
                    u32::from(texel[2]),
                ),
                PresentFormat::Bgra8 => (
                    u32::from(texel[2]),
                    u32::from(texel[1]),
                    u32::from(texel[0]),
                ),
                PresentFormat::Rgb565 => {
                    let packed = u16::from_le_bytes(texel.try_into().unwrap());
                    (
                        u32::from((packed >> 11) & 0x1f) * 255 / 31,
                        u32::from((packed >> 5) & 0x3f) * 255 / 63,
                        u32::from(packed & 0x1f) * 255 / 31,
                    )
                }
                PresentFormat::Rgba4444 => {
                    let packed = u16::from_le_bytes(texel.try_into().unwrap());
                    (
                        u32::from(packed & 0xf) * 17,
                        u32::from((packed >> 4) & 0xf) * 17,
                        u32::from((packed >> 8) & 0xf) * 17,
                    )
                }
            };
            let transformed_x = if transform & 0x1 != 0 {
                crop_width - 1 - source_x
            } else {
                source_x
            };
            let transformed_y = if transform & 0x2 != 0 {
                crop_height - 1 - source_y
            } else {
                source_y
            };
            let (output_x, output_y) = if transform & 0x4 != 0 {
                (crop_height - 1 - transformed_y, transformed_x)
            } else {
                (transformed_x, transformed_y)
            };
            let output_index = usize::try_from(output_y)
                .ok()
                .and_then(|y| {
                    usize::try_from(dimensions.0)
                        .ok()
                        .and_then(|w| y.checked_mul(w))
                })
                .and_then(|row| {
                    usize::try_from(output_x)
                        .ok()
                        .and_then(|x| row.checked_add(x))
                })
                .ok_or(FramebufferError::Malformed("RGB565 output index overflows"))?;
            pixels[output_index] = (red << 16) | (green << 8) | blue;
        }
    }
    Ok(pixels)
}

pub(crate) fn encode_native_window(binder_id: i32) -> [u8; 0x1c] {
    let mut parcel = [0_u8; 0x1c];
    parcel[0..4].copy_from_slice(&12_u32.to_le_bytes());
    parcel[4..8].copy_from_slice(&16_u32.to_le_bytes());
    parcel[8..12].copy_from_slice(&0_u32.to_le_bytes());
    parcel[12..16].copy_from_slice(&28_u32.to_le_bytes());
    parcel[24..28].copy_from_slice(&binder_id.to_le_bytes());
    parcel
}

#[cfg(test)]
mod tests {
    use nixe_cpu::memory::{
        CpuMemory, ExecutionMemory, MemoryAccess, MemoryAccessSize, MemoryMappingPurpose,
        MemoryValue, ProcessMemory,
    };
    use nixe_memory::{AddressSpaceId, GuestVirtualAddress, MemoryPermissions};

    use super::*;

    fn rgba_pitch_buffer(nvmap_id: u32) -> GraphicBuffer {
        GraphicBuffer {
            nvmap_id,
            stride_pixels: 4,
            format: 1,
            external_format: 1,
            usage: 0x100,
            total_size: 0x1000,
            planes: vec![GraphicBufferPlane {
                width: 4,
                height: 4,
                color_format: 0x0100_532120,
                layout: 1,
                pitch: 16,
                offset: 0,
                kind: 0,
                block_height_log2: 0,
                scan_format: 0,
                second_field_offset: 0,
                flags: 0,
                size: 64,
            }]
            .into_boxed_slice(),
        }
    }

    fn video_with_rgba_nvmap() -> (VideoSystem, i32, GraphicBuffer) {
        let video = VideoSystem::default();
        let layer = video.create_layer(DEFAULT_DISPLAY_ID).unwrap();
        let nvdrv = video.nvdrv();
        let memory = ExecutionMemory::new();
        let address_space = AddressSpaceId::new(44);
        let address = GuestVirtualAddress::new(0x4000);
        memory
            .resize_zeroed_mapping(
                address_space,
                address,
                0,
                0x1000,
                MemoryPermissions::READ_WRITE,
                MemoryMappingPurpose::Normal,
            )
            .unwrap();
        for index in 0..16_usize {
            let x = index % 4;
            let y = index / 4;
            let pixel = if x <= y { 0xff00_00ff_u32 } else { 0xffff_0000 };
            memory
                .write(
                    address_space,
                    GuestVirtualAddress::new(address.get() + u64::try_from(index * 4).unwrap()),
                    MemoryAccess::normal(MemoryAccessSize::Word),
                    MemoryValue::U32(pixel),
                )
                .unwrap();
        }
        nvdrv.initialize();
        let fd = nvdrv.open(b"/dev/nvmap", 44).unwrap();
        let mut create = [0_u8; 8];
        create[..4].copy_from_slice(&0x1000_u32.to_le_bytes());
        let (created, error) = nvdrv
            .ioctl(fd, super::super::nvdrv::IOCTL_NVMAP_CREATE, &create)
            .unwrap();
        assert_eq!(error, super::super::nvdrv::NV_SUCCESS);
        let handle = u32::from_le_bytes(created[4..8].try_into().unwrap());
        let mut allocate = [0_u8; 32];
        allocate[..4].copy_from_slice(&handle.to_le_bytes());
        allocate[4..8].copy_from_slice(&0x4000_0000_u32.to_le_bytes());
        allocate[12..16].copy_from_slice(&0x1000_u32.to_le_bytes());
        allocate[24..32].copy_from_slice(&address.get().to_le_bytes());
        assert_eq!(
            nvdrv
                .ioctl_with_memory(fd, 0xc020_0104, &allocate, 44, address_space, &memory)
                .unwrap()
                .1,
            super::super::nvdrv::NV_SUCCESS
        );
        let mut get_id = [0_u8; 8];
        get_id[4..8].copy_from_slice(&handle.to_le_bytes());
        let (get_id, error) = nvdrv.ioctl(fd, 0xc008_010e, &get_id).unwrap();
        assert_eq!(error, super::super::nvdrv::NV_SUCCESS);
        let exported_id = u32::from_le_bytes(get_id[..4].try_into().unwrap());
        (video, layer.binder_id, rgba_pitch_buffer(exported_id))
    }

    #[test]
    fn native_window_places_binder_id_in_the_libnx_payload_slot() {
        let encoded = encode_native_window(7);
        assert_eq!(u32::from_le_bytes(encoded[0..4].try_into().unwrap()), 12);
        assert_eq!(u32::from_le_bytes(encoded[4..8].try_into().unwrap()), 16);
        assert_eq!(i32::from_le_bytes(encoded[24..28].try_into().unwrap()), 7);
    }

    #[test]
    fn layers_have_stable_distinct_binder_objects() {
        let video = VideoSystem::default();
        let first = video.create_layer(DEFAULT_DISPLAY_ID).unwrap();
        let second = video.create_layer(DEFAULT_DISPLAY_ID).unwrap();
        assert_ne!(first.id, second.id);
        assert_ne!(first.binder_id, second.binder_id);
        assert_eq!(video.layer(first.id), Some(first));
    }

    #[test]
    fn layer_geometry_and_binder_references_are_retained() {
        let video = VideoSystem::default();
        let layer = video.create_layer(DEFAULT_DISPLAY_ID).unwrap();

        assert!(video.set_layer_position(layer.id, 12.5, -4.0));
        assert!(video.set_layer_size(layer.id, 640, 360));
        assert!(video.set_layer_z(layer.id, -3));
        let updated = video.layer(layer.id).unwrap();
        assert_eq!((updated.x, updated.y), (12.5, -4.0));
        assert_eq!((updated.width, updated.height, updated.z), (640, 360, -3));

        assert!(video.adjust_binder_refcount(layer.binder_id, 1, 0));
        assert!(video.adjust_binder_refcount(layer.binder_id, 1, 1));
        assert!(video.adjust_binder_refcount(layer.binder_id, -1, 1));
        assert!(!video.adjust_binder_refcount(layer.binder_id, -1, 1));
        assert!(!video.adjust_binder_refcount(layer.binder_id, 1, 2));
    }

    #[test]
    fn graphics_teardown_releases_layers_queues_and_nvdrv_state() {
        let video = VideoSystem::default();
        let layer = video.create_layer(DEFAULT_DISPLAY_ID).unwrap();
        let nvdrv = video.nvdrv();
        nvdrv.initialize();
        let map_fd = nvdrv.open(b"/dev/nvmap", 1).unwrap();
        let _gpu_fd = nvdrv.open(b"/dev/nvhost-ctrl-gpu", 1).unwrap();
        let mut create = [0_u8; 8];
        create[..4].copy_from_slice(&0x1000_u32.to_le_bytes());
        nvdrv
            .ioctl(map_fd, super::super::nvdrv::IOCTL_NVMAP_CREATE, &create)
            .unwrap();
        let mut tpc_masks = [0_u8; 24];
        tpc_masks[0..4].copy_from_slice(&8_u32.to_le_bytes());
        tpc_masks[8..16].copy_from_slice(&1_u64.to_le_bytes());
        assert_eq!(
            nvdrv.ioctl(_gpu_fd, 0xc018_4706, &tpc_masks).unwrap().1,
            super::super::nvdrv::NV_SUCCESS
        );

        assert_eq!(video.active_layer_count(), 1);
        assert!(video.binder_event(layer.binder_id).is_some());
        assert_eq!(
            video.teardown(),
            GraphicsTeardownReport {
                layers_released: 1,
                queues_released: 1,
                pending_frames_released: 0,
                device_fds_released: 2,
                allocations_released: 1,
            }
        );
        assert_eq!(video.active_layer_count(), 0);
        assert!(video.binder_event(layer.binder_id).is_none());
        assert_eq!(video.teardown(), GraphicsTeardownReport::default());
    }

    #[test]
    fn libnx_gob_layout_decodes_to_xrgb8888() {
        let width = 32_u32;
        let height = 8_u32;
        let pitch = 64_u32;
        let mut linear = vec![0_u8; usize::try_from(pitch * height).unwrap()];
        let colors = [0xf800_u16, 0x07e0, 0x001f, 0xffff];
        for y in 0..height {
            for x in 0..width {
                let packed = colors[usize::try_from((x + y) % 4).unwrap()];
                let offset = usize::try_from(y * pitch + x * 2).unwrap();
                linear[offset..offset + 2].copy_from_slice(&packed.to_le_bytes());
            }
        }
        // This is the 16-byte group permutation used by pinned libnx's
        // _convertGobTo16Bx2, independent from the address equation under test.
        let mut swizzled = vec![0_u8; 512];
        for index in 0..32_u32 {
            let y = ((index >> 1) & 0x06) | (index & 0x01);
            let x = ((index << 3) & 0x10) | ((index << 1) & 0x20);
            let source = usize::try_from(y * pitch + x).unwrap();
            let destination = usize::try_from(index * 16).unwrap();
            swizzled[destination..destination + 16].copy_from_slice(&linear[source..source + 16]);
        }
        let buffer = GraphicBuffer {
            nvmap_id: 1,
            stride_pixels: width,
            format: 4,
            external_format: 4,
            usage: 0,
            total_size: 512,
            planes: vec![GraphicBufferPlane {
                width,
                height,
                color_format: 0x010a_881210,
                layout: 3,
                pitch,
                offset: 0,
                kind: 0xfe,
                block_height_log2: 0,
                scan_format: 0,
                second_field_offset: 0,
                flags: 0,
                size: 512,
            }]
            .into_boxed_slice(),
        };
        let decoded = decode_present_image(&buffer, &swizzled, None).unwrap();
        assert_eq!(&decoded[..4], &[0xff0000, 0x00ff00, 0x0000ff, 0xffffff]);
        assert_eq!(
            &decoded[usize::try_from(width).unwrap()..][..4],
            &[0x00ff00, 0x0000ff, 0xffffff, 0xff0000]
        );
    }

    #[test]
    fn unsupported_framebuffer_formats_are_not_guest_malformed_data() {
        let buffer = GraphicBuffer {
            nvmap_id: 1,
            stride_pixels: 1,
            format: 1,
            external_format: 1,
            usage: 0,
            total_size: 4,
            planes: vec![GraphicBufferPlane {
                width: 1,
                height: 1,
                color_format: 0,
                layout: 1,
                pitch: 4,
                offset: 0,
                kind: 0,
                block_height_log2: 0,
                scan_format: 0,
                second_field_offset: 0,
                flags: 0,
                size: 4,
            }]
            .into_boxed_slice(),
        };
        assert!(matches!(
            decode_present_image(&buffer, &[0; 4], None),
            Err(FramebufferError::Unsupported(_))
        ));
    }

    #[test]
    fn rgba_crop_and_rotation_produce_the_expected_host_image() {
        let buffer = GraphicBuffer {
            nvmap_id: 3,
            stride_pixels: 3,
            format: 1,
            external_format: 1,
            usage: 0x100,
            total_size: 24,
            planes: vec![GraphicBufferPlane {
                width: 3,
                height: 2,
                color_format: 0x0100_532120,
                layout: 1,
                pitch: 12,
                offset: 0,
                kind: 0,
                block_height_log2: 0,
                scan_format: 0,
                second_field_offset: 0,
                flags: 0,
                size: 24,
            }]
            .into_boxed_slice(),
        };
        let bytes = [
            1, 0, 0, 255, 2, 0, 0, 255, 3, 0, 0, 255, 4, 0, 0, 255, 5, 0, 0, 255, 6, 0, 0, 255,
        ];
        let crop = CropRect {
            left: 0,
            top: 0,
            right: 3,
            bottom: 2,
        };

        assert_eq!(
            decode_present_image(&buffer, &bytes, Some((crop, 4))).unwrap(),
            vec![0x040000, 0x010000, 0x050000, 0x020000, 0x060000, 0x030000]
        );
        assert_eq!(transformed_dimensions(crop, 4).unwrap(), (2, 3));
        assert_eq!(
            effective_crop(
                CropRect {
                    left: 0,
                    top: 0,
                    right: 0,
                    bottom: 0,
                },
                3,
                2,
            ),
            crop
        );
    }

    #[test]
    fn queue_buffer_input_preserves_crop_transform_timing_and_fences() {
        let mut bytes = [0_u8; 84];
        bytes[0..8].copy_from_slice(&123_i64.to_le_bytes());
        bytes[8..12].copy_from_slice(&1_i32.to_le_bytes());
        for (offset, value) in [(12, 2_i32), (16, 3), (20, 10), (24, 20)] {
            bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
        bytes[28..32].copy_from_slice(&7_u32.to_le_bytes());
        bytes[32..36].copy_from_slice(&5_u32.to_le_bytes());
        bytes[36..40].copy_from_slice(&4_u32.to_le_bytes());
        bytes[44..48].copy_from_slice(&1_u32.to_le_bytes());
        bytes[48..52].copy_from_slice(&1_u32.to_le_bytes());
        bytes[52..56].copy_from_slice(&9_u32.to_le_bytes());
        bytes[56..60].copy_from_slice(&44_u32.to_le_bytes());

        let input = parse_queue_buffer_input(&bytes).unwrap();
        assert_eq!(input.timestamp, 123);
        assert!(input.auto_timestamp);
        assert_eq!(
            input.crop,
            CropRect {
                left: 2,
                top: 3,
                right: 10,
                bottom: 20
            }
        );
        assert_eq!(
            (input.scaling_mode, input.transform, input.sticky_transform),
            (7, 5, 4)
        );
        assert_eq!(input.swap_interval, 1);
        assert_eq!(input.acquire_fences[0].point.to_string(), "syncpoint=9:44");
    }

    #[test]
    fn graphic_buffer_parser_preserves_all_registered_planes_and_usage() {
        let mut words = vec![0_u32; 10 + 81];
        words[0] = 0x4742_4652;
        words[9] = 81;
        let int = |index: usize| 10 + index;
        words[int(1)] = 77;
        words[int(3)] = 0xdaff_caff;
        words[int(6)] = 0x1234_5678;
        words[int(7)] = 1;
        words[int(8)] = 5;
        words[int(9)] = 128;
        words[int(10)] = 0x6000;
        words[int(11)] = 3;
        for plane in 0..3_usize {
            let base = 13 + plane * 22;
            words[int(base)] = 128 >> plane;
            words[int(base + 1)] = 64 >> plane;
            words[int(base + 2)] = 0x532120 + u32::try_from(plane).unwrap();
            words[int(base + 3)] = 1;
            words[int(base + 4)] = if plane == 0 { 3 } else { 1 };
            words[int(base + 5)] = 512 >> plane;
            words[int(base + 7)] = u32::try_from(plane * 0x2000).unwrap();
            words[int(base + 8)] = if plane == 0 { 0xfe } else { 0 };
            words[int(base + 9)] = u32::try_from(plane).unwrap();
            words[int(base + 10)] = 9;
            words[int(base + 11)] = 10;
            words[int(base + 12)] = 11;
            words[int(base + 13)] = 12;
            words[int(base + 14)] = 0x2000;
        }
        let bytes = words
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();

        let parsed = parse_graphic_buffer(&bytes).unwrap();
        assert_eq!(
            (parsed.nvmap_id, parsed.usage, parsed.external_format),
            (77, 0x1234_5678, 5)
        );
        assert_eq!(parsed.planes.len(), 3);
        assert_eq!(
            (parsed.planes[2].width, parsed.planes[2].offset),
            (32, 0x4000)
        );
        assert_eq!(
            (
                parsed.planes[1].scan_format,
                parsed.planes[1].second_field_offset
            ),
            (9, 10)
        );
        assert_eq!(parsed.planes[0].flags, (u64::from(12_u32) << 32) | 11);
    }

    #[test]
    fn queue_reservation_rolls_back_without_releasing_producer_ownership() {
        let buffer = GraphicBuffer {
            nvmap_id: 1,
            stride_pixels: 1,
            format: 1,
            external_format: 1,
            usage: 0,
            total_size: 4,
            planes: vec![GraphicBufferPlane {
                width: 1,
                height: 1,
                color_format: 0x0100_532120,
                layout: 1,
                pitch: 4,
                offset: 0,
                kind: 0,
                block_height_log2: 0,
                scan_format: 0,
                second_field_offset: 0,
                flags: 0,
                size: 4,
            }]
            .into_boxed_slice(),
        };
        let mut queue = BufferQueue::new(4);
        queue.slots.insert(
            2,
            BufferSlot {
                ownership: SlotOwnership::Queueing,
                buffer,
                release_fences: Box::default(),
            },
        );

        queue.rollback_queue(2);
        assert_eq!(queue.slots[&2].ownership, SlotOwnership::Dequeued);
        assert!(!queue.release(2));
        queue.slots.get_mut(&2).unwrap().ownership = SlotOwnership::Queueing;
        assert!(queue.commit_queue(2));
        assert!(!queue.commit_queue(2));
        assert!(queue.release(2));
        assert_eq!(queue.slots[&2].ownership, SlotOwnership::Free);
    }

    #[test]
    fn delayed_gpu_fence_blocks_composition_but_not_vsync_or_slot_ownership() {
        let (video, binder_id, buffer) = video_with_rgba_nvmap();
        let syncpoint = GuestSyncpointId::new(9);
        video.nvdrv().install_test_timeline(syncpoint);
        {
            let mut state = video.state.lock().unwrap();
            state.queues.get_mut(&binder_id).unwrap().slots.insert(
                0,
                BufferSlot {
                    ownership: SlotOwnership::Queueing,
                    buffer: buffer.clone(),
                    release_fences: Box::default(),
                },
            );
        }
        video
            .queue_graphic_buffer(
                binder_id,
                QueuedBufferRequest {
                    slot: 0,
                    buffer,
                    input: QueueBufferInput {
                        timestamp: 0,
                        auto_timestamp: true,
                        crop: CropRect {
                            left: 0,
                            top: 0,
                            right: 4,
                            bottom: 4,
                        },
                        scaling_mode: 0,
                        transform: 0,
                        sticky_transform: 0,
                        swap_interval: 1,
                        acquire_fences: vec![NvFence {
                            point: GuestTimelinePoint::new(syncpoint, GuestSyncpointValue::new(1)),
                        }]
                        .into_boxed_slice(),
                    },
                },
            )
            .unwrap();

        assert_eq!(video.advance(Duration::from_millis(17)).unwrap(), 1);
        assert_eq!(video.mailbox().statistics().published, 0);
        assert_eq!(
            video.state.lock().unwrap().queues[&binder_id].slots[&0].ownership,
            SlotOwnership::Queued
        );

        video.nvdrv().advance_test_timeline(syncpoint);
        assert_eq!(video.advance(Duration::from_millis(34)).unwrap(), 1);
        let frame = video.mailbox().take_latest().unwrap();
        assert_eq!(
            frame.pixels(),
            &[
                0xff0000, 0x0000ff, 0x0000ff, 0x0000ff, 0xff0000, 0xff0000, 0x0000ff, 0x0000ff,
                0xff0000, 0xff0000, 0xff0000, 0x0000ff, 0xff0000, 0xff0000, 0xff0000, 0xff0000,
            ]
        );
        assert_eq!(frame.content_hash64(), 0x2e24_7804_7487_ddaf);
        assert_eq!(
            video.state.lock().unwrap().queues[&binder_id].slots[&0].ownership,
            SlotOwnership::Free
        );
    }

    #[test]
    fn teardown_drops_queued_images_without_publishing_or_releasing_afterwards() {
        let (video, binder_id, buffer) = video_with_rgba_nvmap();
        {
            let mut state = video.state.lock().unwrap();
            state.queues.get_mut(&binder_id).unwrap().slots.insert(
                0,
                BufferSlot {
                    ownership: SlotOwnership::Queueing,
                    buffer: buffer.clone(),
                    release_fences: Box::default(),
                },
            );
        }
        video
            .queue_graphic_buffer(
                binder_id,
                QueuedBufferRequest {
                    slot: 0,
                    buffer,
                    input: QueueBufferInput {
                        timestamp: 0,
                        auto_timestamp: true,
                        crop: CropRect {
                            left: 0,
                            top: 0,
                            right: 4,
                            bottom: 4,
                        },
                        scaling_mode: 0,
                        transform: 0,
                        sticky_transform: 0,
                        swap_interval: 1,
                        acquire_fences: Box::default(),
                    },
                },
            )
            .unwrap();
        let report = video.teardown();
        assert_eq!(report.pending_frames_released, 1);
        assert_eq!(video.advance(Duration::from_secs(1)).unwrap(), 60);
        assert_eq!(video.mailbox().statistics().published, 0);
    }
}
