//! Switch display-service state shared by VI, Binder, and nvdrv sessions.

use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use nixe_runtime::{EventObject, ReadableEventObject, WritableEventObject};
use nixe_video::{DisplayClock, Frame, FrameMailbox};

use crate::parcel::{ParcelError, ParcelReader, ParcelWriter};
use crate::{NvDrvSession, NvMapImageViewMetadata, NvMapPlaneMetadata};

const DEFAULT_DISPLAY_ID: u64 = 1;
const DEFAULT_WIDTH: u32 = 1280;
const DEFAULT_HEIGHT: u32 = 720;

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

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct LayerState {
    pub(crate) id: u64,
    pub(crate) binder_id: i32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) visible: bool,
    pub(crate) scaling_mode: u32,
}

#[derive(Debug)]
struct VideoState {
    next_layer_id: u64,
    layers: BTreeMap<u64, LayerState>,
    queues: BTreeMap<i32, BufferQueue>,
    vsync_writable: WritableEventObject,
    vsync_readable: ReadableEventObject,
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
        let (vsync_writable, vsync_readable) = EventObject::create_pair();
        Self {
            state: Arc::new(Mutex::new(VideoState {
                next_layer_id: 1,
                layers: BTreeMap::new(),
                queues: BTreeMap::new(),
                vsync_writable,
                vsync_readable,
                display_clock: DisplayClock::new(60)
                    .expect("the fixed Switch display refresh is non-zero"),
                mailbox,
                nvdrv: NvDrvSession::new(),
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
        };
        state.layers.insert(id, layer.clone());
        state.queues.insert(binder_id, BufferQueue::new());
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

    pub(crate) fn vsync_event(&self, display_id: u64) -> Option<ReadableEventObject> {
        Self::display_resolution(display_id)?;
        Some(
            self.state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .vsync_readable
                .clone(),
        )
    }

    pub(crate) fn binder_event(&self, binder_id: i32) -> Option<ReadableEventObject> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .queues
            .get(&binder_id)
            .map(|queue| queue.available_readable.clone())
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
            .ok_or(ParcelError("unknown Binder producer object"))?;
        queue.transact(code, encoded)
    }

    pub(crate) fn queue_software_frame(
        &self,
        binder_id: i32,
        slot: i32,
        buffer: &GraphicBuffer,
        bytes: &[u8],
    ) -> Result<(), FramebufferError> {
        let pixels = decode_rgb565_block_linear(buffer, bytes)?;
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let sequence = state.next_frame_sequence;
        state.next_frame_sequence = state.next_frame_sequence.saturating_add(1);
        let frame = Frame::new_xrgb8888(buffer.width, buffer.height, sequence, pixels)
            .map_err(|_| FramebufferError("decoded frame dimensions are invalid"))?;
        state.pending_frames.push_back(PendingFrame {
            binder_id,
            slot,
            frame: Arc::new(frame),
        });
        log::debug!(
            "queued software frame {sequence} from Binder producer {binder_id} slot {slot} ({}x{})",
            buffer.width,
            buffer.height
        );
        Ok(())
    }

    /// Advances guest display timing independently from host monitor VSync.
    pub fn advance(&self, elapsed: Duration) -> u64 {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let ticks = state.display_clock.advance(elapsed);
        if ticks.crossed != 0 {
            state.vsync_writable.signal();
        }
        for _ in 0..ticks.crossed {
            let Some(pending) = state.pending_frames.pop_front() else {
                break;
            };
            state.mailbox.publish(pending.frame);
            log::debug!(
                "latched software frame from Binder producer {} slot {} at VSync {}",
                pending.binder_id,
                pending.slot,
                ticks.latest_sequence
            );
            if let Some(queue) = state.queues.get_mut(&pending.binder_id) {
                let _ = queue.release(pending.slot);
            }
        }
        ticks.crossed
    }
}

const IGRAPHIC_BUFFER_PRODUCER: &str = "android.gui.IGraphicBufferProducer";
// libnx's NvMultiFence is a u32 count followed by four { id, value } fences.
const EMPTY_NV_MULTI_FENCE: [u8; 36] = [0; 36];
const MAX_BUFFER_SLOTS: i32 = 64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SlotOwnership {
    Free,
    Dequeued,
    Queued,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct GraphicBuffer {
    pub(crate) nvmap_id: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) stride_pixels: u32,
    pub(crate) format: u32,
    pub(crate) total_size: u32,
    pub(crate) color_format: u64,
    pub(crate) layout: u32,
    pub(crate) pitch: u32,
    pub(crate) offset: u32,
    pub(crate) kind: u32,
    pub(crate) block_height_log2: u32,
    pub(crate) plane_size: u64,
}

impl GraphicBuffer {
    pub(crate) fn nvmap_view_metadata(&self) -> NvMapImageViewMetadata {
        NvMapImageViewMetadata::new(
            self.width,
            self.height,
            self.format,
            self.kind,
            self.layout,
            self.block_height_log2,
            vec![NvMapPlaneMetadata::new(
                u64::from(self.offset),
                self.plane_size,
                self.pitch,
            )],
        )
    }
}

#[derive(Clone, Debug)]
struct BufferSlot {
    ownership: SlotOwnership,
    buffer: GraphicBuffer,
}

#[derive(Debug)]
struct BufferQueue {
    connected: bool,
    slots: BTreeMap<i32, BufferSlot>,
    available_writable: WritableEventObject,
    available_readable: ReadableEventObject,
}

#[derive(Clone, Debug)]
pub(crate) struct BinderTransaction {
    pub(crate) reply: Vec<u8>,
    pub(crate) queued: Option<(i32, GraphicBuffer)>,
}

#[derive(Clone, Debug)]
struct PendingFrame {
    binder_id: i32,
    slot: i32,
    frame: Arc<Frame>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct FramebufferError(pub(crate) &'static str);

impl BufferQueue {
    fn new() -> Self {
        let (available_writable, available_readable) = EventObject::create_pair();
        Self {
            connected: false,
            slots: BTreeMap::new(),
            available_writable,
            available_readable,
        }
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
                    writer.write_flattened(&EMPTY_NV_MULTI_FENCE)?;
                    writer.write_i32(0);
                    if !self
                        .slots
                        .values()
                        .any(|slot| slot.ownership == SlotOwnership::Free)
                    {
                        self.available_writable.clear();
                    }
                } else {
                    writer.write_i32(-1);
                    writer.write_i32(0);
                    writer.write_i32(-11);
                }
            }
            7 => {
                let slot_index = reader.read_i32()?;
                let _input = reader.read_flattened()?;
                let Some(slot) = self.slots.get_mut(&slot_index) else {
                    return Err(ParcelError("QueueBuffer references an unknown slot"));
                };
                if slot.ownership != SlotOwnership::Dequeued {
                    return Err(ParcelError("QueueBuffer slot is not producer-owned"));
                }
                slot.ownership = SlotOwnership::Queued;
                queued = Some((slot_index, slot.buffer.clone()));
                write_buffer_output(&mut writer, self.pending_count());
                writer.write_i32(0);
            }
            8 => {
                let slot_index = reader.read_i32()?;
                let _fence = reader.read_flattened()?;
                let Some(slot) = self.slots.get_mut(&slot_index) else {
                    return Err(ParcelError("CancelBuffer references an unknown slot"));
                };
                if slot.ownership != SlotOwnership::Dequeued {
                    return Err(ParcelError("CancelBuffer slot is not producer-owned"));
                }
                slot.ownership = SlotOwnership::Free;
                self.available_writable.signal();
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
                    self.available_writable.clear();
                }
                writer.write_i32(status);
            }
            14 => {
                let slot_index = reader.read_i32()?;
                if !(0..MAX_BUFFER_SLOTS).contains(&slot_index) {
                    return Err(ParcelError("preallocated buffer slot is out of range"));
                }
                let has_buffer = reader.read_i32()?;
                if has_buffer == 0 {
                    self.slots.remove(&slot_index);
                } else if has_buffer == 1 {
                    let flattened = reader.read_flattened()?;
                    let buffer = parse_graphic_buffer(flattened)?;
                    if self.slots.contains_key(&slot_index) {
                        return Err(ParcelError("preallocated buffer slot already exists"));
                    }
                    self.slots.insert(
                        slot_index,
                        BufferSlot {
                            ownership: SlotOwnership::Free,
                            buffer,
                        },
                    );
                } else {
                    return Err(ParcelError("preallocated buffer presence flag is invalid"));
                }
                self.update_availability();
            }
            _ => {
                return Err(ParcelError(
                    "unsupported IGraphicBufferProducer transaction",
                ));
            }
        }
        Ok(BinderTransaction {
            reply: writer.finish()?,
            queued,
        })
    }

    fn pending_count(&self) -> u32 {
        u32::try_from(
            self.slots
                .values()
                .filter(|slot| slot.ownership == SlotOwnership::Queued)
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
            self.available_writable.signal();
        } else {
            self.available_writable.clear();
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
        self.available_writable.signal();
        true
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
    const REQUIRED_INTS: usize = 29;
    if bytes.len() < PREFIX_WORDS * 4 {
        return Err(ParcelError("flattened graphic buffer is truncated"));
    }
    let word = |index: usize| -> Result<u32, ParcelError> {
        let offset = index
            .checked_mul(4)
            .ok_or(ParcelError("graphic buffer offset overflows"))?;
        Ok(u32::from_le_bytes(
            bytes
                .get(offset..offset + 4)
                .ok_or(ParcelError("graphic buffer is truncated"))?
                .try_into()
                .unwrap(),
        ))
    };
    if word(0)? != 0x4742_4652 || word(8)? != 0 {
        return Err(ParcelError("flattened graphic buffer header is invalid"));
    }
    let num_ints = usize::try_from(word(9)?)
        .map_err(|_| ParcelError("graphic buffer integer count overflows"))?;
    if num_ints < REQUIRED_INTS
        || PREFIX_WORDS
            .checked_add(num_ints)
            .and_then(|words| words.checked_mul(4))
            .is_none_or(|size| size > bytes.len())
    {
        return Err(ParcelError("graphic buffer integer payload is truncated"));
    }
    let int = |index: usize| word(PREFIX_WORDS + index);
    if int(3)? != 0xdaff_caff || int(11)? != 1 {
        return Err(ParcelError("NvGraphicBuffer metadata is unsupported"));
    }
    let color_format = u64::from(int(15)?) | (u64::from(int(16)?) << 32);
    let plane_size = u64::from(int(27)?) | (u64::from(int(28)?) << 32);
    Ok(GraphicBuffer {
        nvmap_id: int(1)?,
        width: int(13)?,
        height: int(14)?,
        stride_pixels: int(9)?,
        format: int(7)?,
        total_size: int(10)?,
        color_format,
        layout: int(17)?,
        pitch: int(18)?,
        offset: int(20)?,
        kind: int(21)?,
        block_height_log2: int(22)?,
        plane_size,
    })
}

fn decode_rgb565_block_linear(
    buffer: &GraphicBuffer,
    bytes: &[u8],
) -> Result<Vec<u32>, FramebufferError> {
    const PIXEL_FORMAT_RGB_565: u32 = 4;
    const NV_LAYOUT_BLOCK_LINEAR: u32 = 3;
    const NV_KIND_GENERIC_16BX2: u32 = 0xfe;
    const NV_COLOR_FORMAT_R5G6B5: u64 = 0x010a_881210;
    const MAX_DIMENSION: u32 = 8192;
    if buffer.format != PIXEL_FORMAT_RGB_565
        || buffer.layout != NV_LAYOUT_BLOCK_LINEAR
        || buffer.kind != NV_KIND_GENERIC_16BX2
        || buffer.color_format != NV_COLOR_FORMAT_R5G6B5
    {
        return Err(FramebufferError(
            "queued image is not supported block-linear RGB565",
        ));
    }
    if buffer.width == 0
        || buffer.height == 0
        || buffer.width > MAX_DIMENSION
        || buffer.height > MAX_DIMENSION
        || buffer.block_height_log2 > 5
        || buffer.stride_pixels < buffer.width
        || buffer.pitch < buffer.width.saturating_mul(2)
        || !buffer.pitch.is_multiple_of(64)
    {
        return Err(FramebufferError("queued RGB565 dimensions are invalid"));
    }
    let pixel_count = usize::try_from(buffer.width)
        .ok()
        .and_then(|width| {
            usize::try_from(buffer.height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .ok_or(FramebufferError("queued RGB565 dimensions overflow"))?;
    let mut pixels = vec![0_u32; pixel_count];
    let width_in_gobs = u64::from(buffer.pitch / 64);
    let block_height_gobs = 1_u64 << buffer.block_height_log2;
    for y in 0..u64::from(buffer.height) {
        for x in 0..u64::from(buffer.width) {
            let byte_x = x * 2;
            let address = (y / (8 * block_height_gobs)) * 512 * block_height_gobs * width_in_gobs
                + (byte_x / 64) * 512 * block_height_gobs
                + ((y % (8 * block_height_gobs)) / 8) * 512
                + ((byte_x % 64) / 32) * 256
                + ((y % 8) / 2) * 64
                + ((byte_x % 32) / 16) * 32
                + (y % 2) * 16
                + byte_x % 16;
            let address = usize::try_from(address)
                .map_err(|_| FramebufferError("RGB565 swizzle address overflows"))?;
            let packed = u16::from_le_bytes(
                bytes
                    .get(address..address + 2)
                    .ok_or(FramebufferError("RGB565 backing is truncated"))?
                    .try_into()
                    .unwrap(),
            );
            let red = u32::from((packed >> 11) & 0x1f) * 255 / 31;
            let green = u32::from((packed >> 5) & 0x3f) * 255 / 63;
            let blue = u32::from(packed & 0x1f) * 255 / 31;
            let output_index = usize::try_from(y)
                .ok()
                .and_then(|y| {
                    usize::try_from(buffer.width)
                        .ok()
                        .and_then(|width| y.checked_mul(width))
                })
                .and_then(|row| usize::try_from(x).ok().and_then(|x| row.checked_add(x)))
                .ok_or(FramebufferError("RGB565 output index overflows"))?;
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
    use super::*;

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
            width,
            height,
            stride_pixels: width,
            format: 4,
            total_size: 512,
            color_format: 0x010a_881210,
            layout: 3,
            pitch,
            offset: 0,
            kind: 0xfe,
            block_height_log2: 0,
            plane_size: 512,
        };
        let decoded = decode_rgb565_block_linear(&buffer, &swizzled).unwrap();
        assert_eq!(&decoded[..4], &[0xff0000, 0x00ff00, 0x0000ff, 0xffffff]);
        assert_eq!(
            &decoded[usize::try_from(width).unwrap()..][..4],
            &[0x00ff00, 0x0000ff, 0xffffff, 0xff0000]
        );
    }
}
