//! NVDEC packet reconstruction and retained decoded frames consumed by VIC.
//! Layout: https://github.com/averne/FFmpeg/blob/nvtegra/libavutil/nvdec_drv.h
//! NVDEC TBL uses 64x8-byte GOBs and at least two GOBs per block:
//! https://github.com/averne/FFmpeg/blob/nvtegra/libavutil/hwcontext_nvtegra.c
use super::Mapping;
use nixe_memory::PhysicalMapping;
use nixe_memory::{
    Access, AccessOperation, AccessResult, AllocationId, AllocationRange, ArenaOffset, MachineId,
};
use std::collections::{BTreeMap, VecDeque};
use std::ffi::{c_int, c_void};
use std::sync::{Arc, Mutex};

unsafe extern "C" {
    fn nixe_vp9_create() -> *mut c_void;
    fn nixe_vp9_destroy(headers: *mut c_void);
    fn nixe_nvdec_send(
        decoder: *mut c_void,
        headers: *mut c_void,
        picture: *const u8,
        entropy: *const u8,
        tiles: *const u8,
        length: usize,
        offsets: *const u64,
        origin: u64,
    ) -> c_int;
}
use crate::video_codec::Decoder;
pub(super) use crate::video_codec::Frame;

/// Decoders publish retained frames; VIC consumes them without borrowing a
/// decoder or serializing different devices' command execution.
/// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/host1x/host1x.h#L36-L96
#[derive(Debug, Default)]
pub(super) struct FrameQueue(
    Mutex<BTreeMap<super::super::NvDrvFileDescriptor, VecDeque<(SurfaceIdentity, Frame)>>>,
);
impl FrameQueue {
    fn open(&self, fd: super::super::NvDrvFileDescriptor) {
        self.0.lock().unwrap().insert(fd, VecDeque::new());
    }
    pub(super) fn close(&self, fd: super::super::NvDrvFileDescriptor) {
        let retired = self.0.lock().unwrap().remove(&fd);
        drop(retired);
    }
    fn clear(&self, fd: super::super::NvDrvFileDescriptor) {
        if let Some(frames) = self.0.lock().unwrap().get_mut(&fd) {
            frames.clear();
        }
    }
    fn push(&self, fd: super::super::NvDrvFileDescriptor, surface: SurfaceIdentity, frame: Frame) {
        if let Some(frames) = self.0.lock().unwrap().get_mut(&fd) {
            if frames.len() >= 100 {
                frames.pop_front();
            }
            frames.push_back((surface, frame));
        }
    }
    pub(super) fn pop(
        &self,
        fd: &mut Option<super::super::NvDrvFileDescriptor>,
        surface: impl FnOnce() -> Result<SurfaceIdentity, String>,
    ) -> Result<Option<Frame>, String> {
        let mut queues = self.0.lock().unwrap();
        if fd.is_some_and(|fd| !queues.contains_key(&fd)) {
            *fd = None;
        }
        if fd.is_none() {
            let surface = surface()?;
            *fd = queues.iter().find_map(|(&fd, frames)| {
                frames.iter().any(|(id, _)| *id == surface).then_some(fd)
            });
        }
        Ok(fd
            .and_then(|fd| queues.get_mut(&fd))
            .and_then(VecDeque::pop_front)
            .map(|(_, frame)| frame))
    }
}
pub(super) type Mappings = Arc<BTreeMap<u32, Mapping>>;

pub(super) fn region(
    mappings: &Mappings,
    address: u64,
    length: usize,
) -> Result<(&Arc<PhysicalMapping>, u64), String> {
    let (&base, mapping) = mappings
        .range(..=u32::try_from(address).map_err(|_| "Host1x address exceeds 32 bits")?)
        .next_back()
        .ok_or_else(|| format!("unmapped Host1x address {address:#x}"))?;
    let offset = address - u64::from(base);
    if offset
        .checked_add(length as u64)
        .is_none_or(|end| end > u64::from(mapping.object.size()))
    {
        return Err(format!(
            "Host1x memory access outside object: address={address:#x} length={length:#x}"
        ));
    }
    Ok((&mapping.physical, offset))
}
/// Runs only on the Host1x worker. No driver, channel or nvmap lock is held
/// while another device completes this exact physical operation.
fn access(ranges: &[AllocationRange], operation: AccessOperation) -> Result<AccessResult, String> {
    let machine = ranges
        .first()
        .ok_or("Host1x access has no physical spans")?
        .allocation()
        .machine();
    match machine
        .access_captured(ranges, operation)
        .map_err(|e| e.to_string())?
    {
        Access::Ready(result) => Ok(result),
        Access::Pending(pending) => pending.wait().map_err(|e| e.to_string()),
    }
}
pub(super) fn read(mappings: &Mappings, address: u64, length: usize) -> Result<Vec<u8>, String> {
    let (physical, offset) = region(mappings, address, length)?;
    let ranges = physical
        .pin(offset, length as u64)
        .map_err(|e| e.to_string())?;
    let AccessResult::Read(bytes) = access(&ranges, AccessOperation::Read(Vec::new()))? else {
        unreachable!("read returns bytes")
    };
    Ok(bytes)
}
/// Callers provide coalesced runs in storage order. One write names only
/// modified bytes. Padding and masked components are not
/// read back or overwritten merely because they share pages with the output.
pub(super) fn write_spans(
    physical: &PhysicalMapping,
    offset: u64,
    bytes: &[u8],
    spans: Vec<std::ops::Range<usize>>,
) -> Result<(), String> {
    debug_assert!(spans.windows(2).all(|pair| pair[0].end <= pair[1].start));
    let mut ranges = Vec::with_capacity(spans.len());
    let mut payload = Vec::with_capacity(spans.iter().map(|span| span.end - span.start).sum());
    for span in spans {
        let data = bytes
            .get(span.clone())
            .ok_or("Host1x write exceeds captured output")?;
        ranges.extend(
            physical
                .pin(offset + span.start as u64, data.len() as u64)
                .map_err(|e| e.to_string())?,
        );
        payload.extend_from_slice(data);
    }
    if ranges.is_empty() {
        return Ok(());
    }
    assert!(matches!(
        access(&ranges, AccessOperation::Write(payload))?,
        AccessResult::Written(_)
    ));
    Ok(())
}

pub(super) fn word(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}
pub(super) fn surface_identity(
    mappings: &Mappings,
    address: u64,
) -> Result<(MachineId, AllocationId, ArenaOffset), String> {
    let (physical, offset) = region(mappings, address, 1)?;
    let spans = physical.slice(offset, 1).map_err(|e| e.to_string())?;
    let span = spans
        .spans()
        .first()
        .ok_or("NVDEC surface has no physical identity")?;
    Ok((physical.machine(), span.allocation, span.start))
}
pub(super) fn gob_offset(x: usize, y: usize, pitch: usize, log2_height: u32) -> usize {
    let height = 1usize << log2_height;
    (y / (8 * height)) * pitch.div_ceil(64) * 512 * height
        + (x / 64) * 512 * height
        + (y % (8 * height) / 8) * 512
        + (x % 64 / 32) * 256
        + (y % 8 / 2) * 64
        + (x % 32 / 16) * 32
        + (y % 2) * 16
        + x % 16
}

pub(super) type SurfaceIdentity = (MachineId, AllocationId, ArenaOffset);
#[derive(Debug)]
struct Output {
    identity: SurfaceIdentity,
    width: usize,
    height: usize,
}
#[derive(Debug)]
pub(super) struct NvDec {
    decoder: Option<Decoder>,
    vp9: *mut c_void,
    codec: u32,
    device: Option<[u8; 16]>,
    h264_initialized: bool,
    next_origin: u64,
    outputs: BTreeMap<u64, Output>,
    frames: Arc<FrameQueue>,
    fd: super::super::NvDrvFileDescriptor,
}
impl Drop for NvDec {
    fn drop(&mut self) {
        unsafe { nixe_vp9_destroy(self.vp9) };
    }
}
impl NvDec {
    pub(super) fn new(
        device: Option<[u8; 16]>,
        frames: Arc<FrameQueue>,
        fd: super::super::NvDrvFileDescriptor,
    ) -> Self {
        frames.open(fd);
        Self {
            decoder: None,
            vp9: std::ptr::null_mut(),
            codec: 0,
            device,
            h264_initialized: false,
            next_origin: 1,
            outputs: BTreeMap::new(),
            frames,
            fd,
        }
    }
    fn select_codec(&mut self, codec: u32) -> Result<(), String> {
        if self.codec == codec {
            return Ok(());
        }
        let decoder = Decoder::new(codec, self.device)?;
        self.decoder = Some(decoder);
        unsafe { nixe_vp9_destroy(self.vp9) };
        self.vp9 = std::ptr::null_mut();
        if codec == 9 {
            self.vp9 = unsafe { nixe_vp9_create() };
            if self.vp9.is_null() {
                return Err("cannot allocate VP9 reconstruction state".into());
            }
        }
        self.codec = codec;
        self.h264_initialized = false;
        self.outputs.clear();
        self.frames.clear(self.fd);
        Ok(())
    }
    fn error(&self) -> String {
        self.decoder.as_ref().unwrap().error()
    }
    pub(super) fn execute(&mut self, registers: &[u32], mappings: &Mappings) -> Result<(), String> {
        self.select_codec(registers[0x80])?;
        if self.codec == 3 {
            return self.execute_h264(registers, mappings);
        }
        let picture = read(mappings, u64::from(registers[0x101]) << 8, 256)?;
        let entropy = read(mappings, u64::from(registers[0x170]) << 8, 3744)?;
        if word(&picture, 0x34) != 0 {
            return Err("encrypted NVDEC VP9 input is unsupported".into());
        }
        let layout = word(&picture, 0x40);
        if layout & 3 != 0 {
            return Err(format!(
                "unsupported NVDEC surface tile format {}",
                layout & 3
            ));
        }
        let block_height = (layout >> 2 & 7) + 1;
        if block_height > 5 {
            return Err("invalid NVDEC GOB height".into());
        }
        let dim = |offset| {
            usize::from(u16::from_le_bytes(
                picture[offset..offset + 2].try_into().unwrap(),
            ))
        };
        let width = dim(0x60);
        let height = dim(0x62);
        let luma_pitch = dim(0x64);
        let chroma_pitch = dim(0x66);
        if width == 0
            || height == 0
            || width > 16384
            || height > 16384
            || luma_pitch < width
            || chroma_pitch < width.div_ceil(2) * 2
            || luma_pitch % 64 != 0
            || chroma_pitch % 64 != 0
        {
            return Err("invalid NVDEC VP9 frame dimensions/pitches".into());
        }
        let length = word(&picture, 0x30) as usize;
        if length == 0 || length > i32::MAX as usize {
            return Err("invalid NVDEC bitstream size".into());
        }
        let tiles = read(mappings, u64::from(registers[0x102]) << 8, length)?;
        let offsets = std::array::from_fn::<_, 4, _>(|i| u64::from(registers[0x10c + i]) << 8);
        let origin = self.next_origin;
        self.next_origin = self
            .next_origin
            .checked_add(1)
            .ok_or("NVDEC frame origin overflow")?;
        self.outputs.insert(
            origin,
            Output {
                identity: surface_identity(mappings, offsets[3])?,
                width,
                height,
            },
        );
        log::trace!(
            "NVDEC VP9 input: origin={origin} dimensions={width}x{height} bytes={length} pitch={luma_pitch}/{chroma_pitch} block-height={block_height}"
        );
        let result = unsafe {
            nixe_nvdec_send(
                self.decoder.as_mut().unwrap().raw(),
                self.vp9,
                picture.as_ptr(),
                entropy.as_ptr(),
                tiles.as_ptr(),
                tiles.len(),
                offsets.as_ptr(),
                origin,
            )
        };
        if result < 0 {
            return Err(format!(
                "VP9 reconstruction/decode failed: {}",
                self.error()
            ));
        }
        if result == 0 {
            // A hidden frame updates references but produces no presentation.
            self.outputs.remove(&origin);
        }
        self.receive()?;
        Ok(())
    }
    fn execute_h264(&mut self, registers: &[u32], mappings: &Mappings) -> Result<(), String> {
        let context = read(mappings, u64::from(registers[0x101]) << 8, 0x2fc)?;
        let picture = super::h264::Picture::decode(&context, registers)?;
        if picture.stream_length == 0 || picture.stream_length > i32::MAX as usize - 512 {
            return Err("invalid NVDEC H.264 bitstream length".into());
        }
        let stream = read(
            mappings,
            u64::from(registers[0x102]) << 8,
            picture.stream_length,
        )?;
        let packet = if !self.h264_initialized || picture.frame_number == 0 {
            let mut packet = super::h264::headers(&context)?;
            packet.extend_from_slice(&stream);
            packet
        } else {
            stream
        };
        self.h264_initialized = true;
        let origin = self.next_origin;
        self.next_origin = self
            .next_origin
            .checked_add(1)
            .ok_or("NVDEC frame origin overflow")?;
        self.outputs.insert(
            origin,
            Output {
                identity: surface_identity(mappings, picture.luma)?,
                width: picture.width,
                height: picture.height,
            },
        );
        log::trace!(
            "NVDEC H.264 input: origin={origin} frame={} dimensions={}x{} bytes={} pitch={}/{} block-height={}",
            picture.frame_number,
            picture.width,
            picture.height,
            packet.len(),
            picture.luma_pitch,
            picture.chroma_pitch,
            picture.block_height
        );
        self.decoder
            .as_mut()
            .unwrap()
            .send(&packet, origin)
            .map_err(|e| format!("H.264 reconstruction/decode failed: {e}"))?;
        // H.264 may retain pictures for display reordering. Keep their exact
        // submitted surfaces until FFmpeg returns each packet's origin.
        self.receive()
    }
    fn receive(&mut self) -> Result<(), String> {
        // Drain before another send, as required by FFmpeg's send/receive protocol.
        while let Some((decoded_origin, frame)) = self.decoder.as_mut().unwrap().receive()? {
            let decoded_width = frame.width;
            let decoded_height = frame.height;
            let output = self
                .outputs
                .remove(&decoded_origin)
                .ok_or("NVDEC returned an unknown frame origin")?;
            if output.width != frame.width || output.height != frame.height {
                return Err("NVDEC output dimensions do not match submitted surface".into());
            }
            // Presentation order is FFmpeg receive order, including H.264 B
            // pictures. Pin the frame, not a native image or a copy of guest RAM.
            self.frames.push(self.fd, output.identity, frame);
            log::trace!(
                "NVDEC output: origin={decoded_origin} dimensions={decoded_width}x{decoded_height}"
            );
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::nvdrv::nvmap::{NvMapAllocationMetadata, NvMapCpuMapping, NvMapObjects, NvMapOwner};
    use nixe_memory::{AddressSpaceId, GuestVirtualAddress};

    #[test]
    fn reference_identity_survives_iova_reassignment_and_distinguishes_reused_addresses() {
        let mut nvmap = NvMapObjects::default();
        let owner = NvMapOwner::new(1);
        let mut objects = Vec::new();
        for base in [0x1000, 0x2000] {
            let handle = nvmap.create(owner, 4096).unwrap();
            let machine = crate::test_memory::machine();
            let storage = machine.account(4096).unwrap().allocate(4096).unwrap();
            let input = nixe_memory::AllocationRange::new(storage.pin(), 0, 4096).unwrap();
            nvmap
                .allocate(
                    owner,
                    handle,
                    NvMapAllocationMetadata::new(0, 0, 4096, 0),
                    NvMapCpuMapping::new(AddressSpaceId::new(1), GuestVirtualAddress::new(base)),
                    Arc::from([input.clone()]),
                )
                .unwrap();
            objects.push(
                nvmap
                    .object_snapshot_by_owned_handle(owner, handle)
                    .unwrap(),
            );
        }
        let original = Arc::new(BTreeMap::from([(
            0x1000,
            Mapping {
                physical: nixe_memory::PhysicalMapping::new(objects[0].ranges().unwrap()).unwrap(),
                object: objects[0].clone(),
                references: 1,
            },
        )]));
        let remapped = Arc::new(BTreeMap::from([
            (
                0x1000,
                Mapping {
                    physical: nixe_memory::PhysicalMapping::new(objects[1].ranges().unwrap())
                        .unwrap(),
                    object: objects[1].clone(),
                    references: 1,
                },
            ),
            (
                0x3000,
                Mapping {
                    physical: nixe_memory::PhysicalMapping::new(objects[0].ranges().unwrap())
                        .unwrap(),
                    object: objects[0].clone(),
                    references: 1,
                },
            ),
        ]));
        assert_eq!(
            surface_identity(&original, 0x1020).unwrap(),
            surface_identity(&remapped, 0x3020).unwrap()
        );
        assert_ne!(
            surface_identity(&original, 0x1020).unwrap(),
            surface_identity(&remapped, 0x1020).unwrap()
        );
        assert_ne!(
            surface_identity(&remapped, 0x3020).unwrap(),
            surface_identity(&remapped, 0x3021).unwrap()
        );
    }
    #[test]
    fn gob_addresses_cover_every_byte_once_across_rows_and_block_heights() {
        for log2 in 0..=5 {
            let pitch = 192;
            let rows = 8 << log2;
            let mut seen = vec![false; pitch * rows];
            for y in 0..rows {
                for x in 0..pitch {
                    let offset = gob_offset(x, y, pitch, log2);
                    assert!(!seen[offset]);
                    seen[offset] = true;
                }
            }
            assert!(seen.into_iter().all(|b| b));
            assert_eq!(gob_offset(0, rows, pitch, log2), pitch * rows);
        }
        assert_eq!(gob_offset(16, 0, 64, 1), 32);
        assert_eq!(gob_offset(0, 1, 64, 1), 16);
        assert_eq!(gob_offset(32, 0, 64, 1), 256);
        assert_eq!(gob_offset(0, 8, 64, 1), 512);
    }
}
