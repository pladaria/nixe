//! NVDEC packet reconstruction and retained decoded frames consumed by VIC.
//! Layout: https://github.com/averne/FFmpeg/blob/nvtegra/libavutil/nvdec_drv.h
//! NVDEC TBL uses 64x8-byte GOBs and at least two GOBs per block:
//! https://github.com/averne/FFmpeg/blob/nvtegra/libavutil/hwcontext_nvtegra.c
use super::Mapping;
use nixe_gpu::physical::PhysicalMapping;
use nixe_memory::{
    Access, AccessOperation, AccessResult, AllocationId, AllocationRange, ArenaOffset, MachineId,
};
use std::borrow::Cow;
use std::collections::{BTreeMap, VecDeque};
use std::ffi::{CStr, c_char, c_int, c_void};
use std::sync::Arc;

unsafe extern "C" {
    fn nixe_nvdec_create(codec: u32, uuid: *const u8) -> *mut c_void;
    fn nixe_video_hardware_supported(codec: u32, uuid: *const u8) -> c_int;
    fn nixe_nvdec_send_packet(
        decoder: *mut c_void,
        packet: *const u8,
        length: usize,
        origin: u64,
    ) -> c_int;
    fn nixe_video_error() -> *const c_char;
    fn nixe_video_frame_free(frame: *mut c_void);
    fn nixe_video_frame_planes(
        frame: *const c_void,
        planes: *mut *const u8,
        strides: *mut i32,
    ) -> c_int;
    fn nixe_nvdec_destroy(decoder: *mut c_void);
    fn nixe_nvdec_error(decoder: *const c_void) -> *const c_char;
    fn nixe_nvdec_send(
        decoder: *mut c_void,
        picture: *const u8,
        entropy: *const u8,
        tiles: *const u8,
        length: usize,
        offsets: *const u64,
        origin: u64,
    ) -> c_int;
    fn nixe_nvdec_receive(
        decoder: *mut c_void,
        origin: *mut u64,
        width: *mut u32,
        height: *mut u32,
        retained_frame: *mut *mut c_void,
    ) -> c_int;
}

#[derive(Debug)]
pub(super) struct Frame {
    raw: *mut c_void,
    pub(super) width: usize,
    pub(super) height: usize,
    planes: [*const u8; 3],
    strides: [usize; 3],
}
// Host1x owns each immutable AVFrame reference until VIC consumes it. Borrowed
// plane slices cannot outlive that reference; only owned uploads cross threads.
impl Drop for Frame {
    fn drop(&mut self) {
        unsafe { nixe_video_frame_free(self.raw) };
    }
}
impl Frame {
    fn new(raw: *mut c_void, width: usize, height: usize) -> Result<Self, String> {
        if raw.is_null() {
            return Err("NVDEC returned no retained frame".into());
        }
        let mut frame = Self {
            raw,
            width,
            height,
            planes: [std::ptr::null(); 3],
            strides: [0; 3],
        };
        let mut strides = [0; 3];
        let count = unsafe {
            nixe_video_frame_planes(raw, frame.planes.as_mut_ptr(), strides.as_mut_ptr())
        };
        if count < 0 {
            return Err(video_error().into());
        }
        let widths = [
            width,
            if count == 2 {
                width.div_ceil(2) * 2
            } else {
                width.div_ceil(2)
            },
            width.div_ceil(2),
        ];
        for i in 0..count as usize {
            if frame.planes[i].is_null() || strides[i] < widths[i] as i32 {
                return Err("invalid decoded host video plane".into());
            }
            frame.strides[i] = strides[i] as usize;
        }
        Ok(frame)
    }
    pub(super) fn luma(&self) -> &[u8] {
        // SAFETY: FFmpeg owns all visible rows with their validated positive
        // stride. Excluding final row padding keeps the slice within its plane.
        unsafe {
            std::slice::from_raw_parts(
                self.planes[0],
                (self.height - 1) * self.strides[0] + self.width,
            )
        }
    }
    pub(super) fn chroma_pitch(&self) -> usize {
        if self.planes[2].is_null() {
            self.strides[1]
        } else {
            self.width.div_ceil(2) * 2
        }
    }
    pub(super) fn luma_pitch(&self) -> usize {
        self.strides[0]
    }
    pub(super) fn chroma(&self) -> Cow<'_, [u8]> {
        let width = self.width.div_ceil(2) * 2;
        let rows = self.height.div_ceil(2);
        if self.planes[2].is_null() {
            // SAFETY: receive validated the NV12 plane and its complete extent.
            Cow::Borrowed(unsafe {
                std::slice::from_raw_parts(self.planes[1], (rows - 1) * self.strides[1] + width)
            })
        } else {
            // The CPU compositor consumes semiplanar UV. Convert only software
            // planar output; CUDA already supplies an interleaved host plane.
            let mut uv = vec![0; width * rows];
            for (y, row) in uv.chunks_exact_mut(width).enumerate() {
                for (x, value) in row.iter_mut().enumerate() {
                    unsafe {
                        *value = *self.planes[1 + x % 2].add(y * self.strides[1 + x % 2] + x / 2);
                    }
                }
            }
            Cow::Owned(uv)
        }
    }
    /// Own only the visible pixels needed by the asynchronous graphics owner.
    /// CPU VIC reads the retained planes directly instead of making this copy.
    pub(super) fn linear_planes(
        &self,
        width: usize,
        height: usize,
    ) -> Result<[Arc<[u8]>; 2], String> {
        if width == 0 || width > self.width || height == 0 || height > self.height {
            return Err("VIC source exceeds decoded frame dimensions".into());
        }
        let widths = [width, width.div_ceil(2) * 2];
        let heights = [height, height.div_ceil(2)];
        Ok(std::array::from_fn(|plane| {
            let width = widths[plane];
            let mut bytes = Arc::<[u8]>::new_uninit_slice(width * heights[plane]);
            for (y, row) in Arc::get_mut(&mut bytes)
                .unwrap()
                .chunks_exact_mut(width)
                .enumerate()
            {
                if plane == 0 || self.planes[2].is_null() {
                    // SAFETY: the validated FFmpeg row contains `width` bytes;
                    // the unique output row has exactly that writable extent.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            self.planes[plane].add(y * self.strides[plane]),
                            row.as_mut_ptr().cast::<u8>(),
                            width,
                        );
                    }
                } else {
                    for (x, value) in row.iter_mut().enumerate() {
                        unsafe {
                            value.write(
                                *self.planes[1 + x % 2].add(y * self.strides[1 + x % 2] + x / 2),
                            );
                        }
                    }
                }
            }
            // SAFETY: every output byte was initialized in the row loop above.
            unsafe { bytes.assume_init() }
        }))
    }
}

fn video_error() -> Box<str> {
    unsafe { CStr::from_ptr(nixe_video_error()) }
        .to_string_lossy()
        .into_owned()
        .into_boxed_str()
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
    let AccessResult::Read(bytes) = access(&ranges, AccessOperation::Read)? else {
        unreachable!("read returns bytes")
    };
    Ok(bytes.into_vec())
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
        access(&ranges, AccessOperation::Write(payload.into_boxed_slice()))?,
        AccessResult::Written
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
    decoder: *mut c_void,
    codec: u32,
    device: Option<[u8; 16]>,
    h264_initialized: bool,
    next_origin: u64,
    outputs: BTreeMap<u64, Output>,
    surfaces: BTreeMap<[SurfaceIdentity; 2], u64>,
    frames: VecDeque<(SurfaceIdentity, Frame)>,
}
impl Drop for NvDec {
    fn drop(&mut self) {
        unsafe { nixe_nvdec_destroy(self.decoder) };
    }
}
impl NvDec {
    pub(super) fn new(device: Option<[u8; 16]>) -> Self {
        Self {
            decoder: std::ptr::null_mut(),
            codec: 0,
            device,
            h264_initialized: false,
            next_origin: 1,
            outputs: BTreeMap::new(),
            surfaces: BTreeMap::new(),
            frames: VecDeque::new(),
        }
    }
    pub(super) fn has_frame(&self, identity: SurfaceIdentity) -> bool {
        self.frames.iter().any(|(surface, _)| *surface == identity)
    }
    pub(super) fn pop_frame(&mut self) -> Option<Frame> {
        // The VIC channel identifies its decoder once. FFmpeg receive order
        // already accounts for H.264 display reordering.
        self.frames.pop_front().map(|(_, frame)| frame)
    }
    fn select_codec(&mut self, codec: u32) -> Result<(), String> {
        if self.codec == codec {
            return Ok(());
        }
        if !matches!(codec, 3 | 9) {
            return Err(format!("unsupported NVDEC codec {codec:#x}"));
        }
        unsafe { nixe_nvdec_destroy(self.decoder) };
        self.decoder = std::ptr::null_mut();
        self.codec = 0;
        self.h264_initialized = false;
        self.outputs.clear();
        self.surfaces.clear();
        self.frames.clear();
        let name = if codec == 3 { "H.264" } else { "VP9" };
        let device = match self.device {
            Some(uuid) => match unsafe { nixe_video_hardware_supported(codec, uuid.as_ptr()) } {
                1 => Some(uuid),
                0 => None,
                _ => {
                    return Err(format!(
                        "cannot query hardware video decoding: {}",
                        video_error()
                    ));
                }
            },
            None => None,
        };
        let decoder = unsafe {
            nixe_nvdec_create(
                codec,
                device
                    .as_ref()
                    .map_or(std::ptr::null(), |uuid| uuid.as_ptr()),
            )
        };
        if decoder.is_null() {
            return Err(format!(
                "cannot initialize FFmpeg {name} decoder: {}",
                unsafe { CStr::from_ptr(nixe_video_error()) }.to_string_lossy()
            ));
        }
        log::info!(
            "{name} decoder selected: {}",
            if device.is_some() {
                "CUDA/NVDEC hardware, retained frames"
            } else {
                "software"
            }
        );
        self.decoder = decoder;
        self.codec = codec;
        Ok(())
    }
    fn error(&self) -> String {
        unsafe { CStr::from_ptr(nixe_nvdec_error(self.decoder)) }
            .to_string_lossy()
            .into_owned()
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
        // IOVAs are routinely unmapped, reused and reassigned. The software DPB
        // follows canonical physical surface identity rather than transient device VAs.
        let mut references = [0; 4];
        for (index, address) in offsets.iter().enumerate() {
            let key = [
                surface_identity(mappings, *address)?,
                surface_identity(mappings, u64::from(registers[0x11d + index]) << 8)?,
            ];
            let next = self.surfaces.len() as u64 + 1;
            references[index] = *self.surfaces.entry(key).or_insert(next);
        }
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
                self.decoder,
                picture.as_ptr(),
                entropy.as_ptr(),
                tiles.as_ptr(),
                tiles.len(),
                references.as_ptr(),
                origin,
            )
        };
        if result < 0 {
            return Err(format!(
                "VP9 reconstruction/decode failed: {}",
                self.error()
            ));
        }
        self.receive()?;
        if self.outputs.contains_key(&origin) {
            return Err(
                "NVDEC visible VP9 frame did not produce a surface before completion".into(),
            );
        }
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
        if unsafe { nixe_nvdec_send_packet(self.decoder, packet.as_ptr(), packet.len(), origin) }
            < 0
        {
            return Err(format!(
                "H.264 reconstruction/decode failed: {}",
                self.error()
            ));
        }
        // H.264 may retain pictures for display reordering. Keep their exact
        // submitted surfaces until FFmpeg returns each packet's origin.
        self.receive()
    }
    fn receive(&mut self) -> Result<(), String> {
        // Drain before another send, as required by FFmpeg's send/receive protocol.
        loop {
            let mut decoded_origin = 0;
            let mut decoded_width = 0;
            let mut decoded_height = 0;
            let mut retained_frame = std::ptr::null_mut();
            let result = unsafe {
                nixe_nvdec_receive(
                    self.decoder,
                    &mut decoded_origin,
                    &mut decoded_width,
                    &mut decoded_height,
                    &mut retained_frame,
                )
            };
            if result < 0 {
                return Err(format!("NVDEC receive failed: {}", self.error()));
            }
            if result == 0 {
                break;
            }
            let frame = Frame::new(
                retained_frame,
                decoded_width as usize,
                decoded_height as usize,
            )?;
            let output = self
                .outputs
                .remove(&decoded_origin)
                .ok_or("NVDEC returned an unknown frame origin")?;
            if output.width != frame.width || output.height != frame.height {
                return Err("NVDEC output dimensions do not match submitted surface".into());
            }
            // Presentation order is FFmpeg receive order, including H.264 B
            // pictures. Pin the frame, not a native image or a copy of guest RAM.
            let limit = 100;
            if self.frames.len() >= limit {
                self.frames.pop_front();
            }
            self.frames.push_back((output.identity, frame));
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
    fn h264_reconstructed_headers_retain_frames_including_reordered_surfaces() {
        // Original 16x16 red frames encoded with libx264, high profile, QP20,
        // bframes=0, ref=4. Only slices remain: the guest supplies SPS/PPS fields
        // through NVDEC's context rather than the original container headers.
        let slices: [&[u8]; 3] = [
            &[
                0x65, 0x88, 0x84, 0x3f, 0xfe, 0xbd, 0x67, 0xe6, 0x58, 0xdf, 0x1c, 0x74, 0x91, 0x6d,
                0x28, 0x43, 0xda, 0x81,
            ],
            &[0x41, 0x9a, 0x3b, 0x1f, 0xfe, 0xc0],
            &[0x41, 0x9a, 0x57, 0xe1, 0x0c, 0x99, 0x4c, 0x3f, 0xfe, 0xc1],
        ];
        let machine = crate::test_memory::machine();
        let allocation = machine.account(0x10000).unwrap().allocate(0x10000).unwrap();
        let input = AllocationRange::new(allocation.pin(), 0, 0x10000).unwrap();
        machine
            .write_resolved(std::slice::from_ref(&input), &vec![0xa5; 0x10000])
            .unwrap();
        let mut nvmap = NvMapObjects::default();
        let owner = NvMapOwner::new(1);
        let handle = nvmap.create(owner, 0x10000).unwrap();
        nvmap
            .allocate(
                owner,
                handle,
                NvMapAllocationMetadata::new(0, 0, 4096, 0),
                NvMapCpuMapping::new(AddressSpaceId::new(1), GuestVirtualAddress::new(0x1000)),
                Arc::from([input.clone()]),
            )
            .unwrap();
        let object = nvmap
            .object_snapshot_by_owned_handle(owner, handle)
            .unwrap();
        let mappings = Arc::new(BTreeMap::from([(
            0x1000,
            Mapping {
                physical: PhysicalMapping::new(object.ranges().unwrap()).unwrap(),
                object,
                references: 1,
            },
        )]));
        let mut context = vec![0; 0x2fc];
        for (offset, value) in [
            (8, 1_u32),
            (12, 1),
            (16, 1),
            (0x18, 1),
            (0x20, 3),
            (0x28, 1),
            (0x30, 1),
            (0x34, 64),
            (0x38, 64),
            (0x44, 1),
            (0x50, 1),
        ] {
            context[0x58 + offset..0x5c + offset].copy_from_slice(&value.to_le_bytes());
        }
        context[0x1c0..0x2a0].fill(16);
        let flags =
            2_u64 | 4 | (1 << 12) | (2 << 14) | (58 << 16) | (30 << 22) | (30 << 27) | (4 << 34);
        let mut registers = [0; 0x200];
        registers[0x80] = 3;
        registers[0x101] = 0x10;
        registers[0x102] = 0x14;
        registers[0x10c + 4] = 0x30;
        registers[0x11d + 4] = 0x60;
        let mut decoder = NvDec::new(None);
        let fd = super::super::super::NvDrvFileDescriptor::new(7);
        let mut vic = super::super::vic::Vic::default();
        // Repeated IDR also exercises header regeneration when the movie loops.
        for frame in [0, 1, 2, 0] {
            let mut packet = vec![0, 0, 1];
            packet.extend_from_slice(slices[frame]);
            context[0x48..0x4c].copy_from_slice(&(packet.len() as u32).to_le_bytes());
            context[0xb0..0xb8].copy_from_slice(&(flags | ((frame as u64) << 46)).to_le_bytes());
            machine
                .write_resolved(&[input.slice(0, context.len() as u64).unwrap()], &context)
                .unwrap();
            machine
                .write_resolved(&[input.slice(0x400, packet.len() as u64).unwrap()], &packet)
                .unwrap();
            decoder.execute(&registers, &mappings).unwrap();
            assert!(decoder.outputs.is_empty());
            {
                let frame = &decoder.frames.front().unwrap().1;
                assert_eq!(frame.luma().as_ptr(), frame.planes[0]);
                let planes = frame.linear_planes(frame.width, frame.height).unwrap();
                assert!(planes[0].iter().all(|&v| v == 81));
                assert!(planes[1].chunks_exact(2).all(|uv| uv == [90, 240]));
                // Crop both width and coded height before an owned GPU upload.
                let cropped = frame.linear_planes(13, 13).unwrap();
                assert_eq!(cropped[0].len(), 13 * 13);
                assert_eq!(cropped[1].len(), 14 * 7);
                assert!(cropped[0].iter().all(|&v| v == 81));
                assert!(cropped[1].chunks_exact(2).all(|uv| uv == [90, 240]));
            }
            // Consume the actual retained frame through CPU VIC. Its FFmpeg
            // strides differ from the guest surface pitches; input RAM still
            // contains the sentinel rather than a materialized decode surface.
            let nv12 = frame % 2 == 0;
            let mut config = vec![0; 0x610];
            for (offset, value) in [
                (0x10, 4u32),
                (0x18, 15 << 16),
                (0x1c, 12 << 16),
                (0x20, if nv12 { 68 } else { 31 }),
                (0x24, 15 | 12 << 14),
                (0x28, if nv12 { 31 | 15 << 14 } else { 19 | 15 << 14 }),
                (0x2c, 15 | 7 << 14),
                (0x90, 0x101),
                (0xa0, 1023 << 10),
                (0xa4, 1023),
                (0xb4, 15 << 16),
                (0xbc, 12 << 16),
                (0xc0, 15 << 16),
                (0xc4, 12 << 16),
                (0xd0, 68 | 1 << 11 | 1 << 15),
                (0xd4, 15 | 15 << 14),
                (0xd8, 63 | 15 << 14),
                (0xdc, 31 | 7 << 14),
                (0x130, 1023),
                (0x134, 0x2050),
            ] {
                config[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
            }
            machine
                .write_resolved(&[input.slice(0x800, config.len() as u64).unwrap()], &config)
                .unwrap();
            machine
                .write_resolved(&[input.slice(0xb000, 0x3000).unwrap()], &vec![0xa5; 0x3000])
                .unwrap();
            let mut vic_registers = [0; 0x446];
            vic_registers[0x1c2] = 0x18;
            // Once bound, VIC consumes its decoder FIFO even when the guest
            // supplies a different/unmapped decode surface for a later picture.
            vic_registers[0x100] = if vic.decoder.is_none() {
                0x31
            } else {
                0xffff00
            };
            vic_registers[0x101] = 0x61;
            vic_registers[0x1c8] = 0xc0;
            vic_registers[0x1c9] = 0xd0;
            let mut decoders = BTreeMap::from([(fd, decoder)]);
            vic.execute(&vic_registers, &mappings, None, &mut decoders)
                .unwrap();
            // A repeated VIC execute has no new decoder frame and must retain
            // the previous picture instead of reading the unmaterialized input.
            vic.execute(&vic_registers, &mappings, None, &mut decoders)
                .unwrap();
            decoder = decoders.remove(&fd).unwrap();
            assert!(decoder.frames.is_empty());
            let pitch = if nv12 { 32 } else { 80 };
            let mut pixels = vec![0; pitch * 16];
            crate::test_memory::read(
                &[input.slice(0xb000, pixels.len() as u64).unwrap()],
                &mut pixels,
            );
            let mut expected = vec![0xa5; pixels.len()];
            for y in 0..13 {
                if nv12 {
                    expected[y * pitch..y * pitch + 16].fill(81);
                } else {
                    for pixel in expected[y * pitch..y * pitch + 64].chunks_exact_mut(4) {
                        pixel.copy_from_slice(&[81, 90, 240, 255]);
                    }
                }
            }
            assert_eq!(pixels, expected);
            if nv12 {
                let mut uv = vec![0; 32 * 8];
                crate::test_memory::read(&[input.slice(0xc000, uv.len() as u64).unwrap()], &mut uv);
                let mut expected = vec![0xa5; uv.len()];
                for y in 0..7 {
                    for pair in expected[y * 32..y * 32 + 16].chunks_exact_mut(2) {
                        pair.copy_from_slice(&[90, 240]);
                    }
                }
                assert_eq!(uv, expected);
            }
        }
        assert!(
            decoder
                .select_codec(0xff)
                .unwrap_err()
                .contains("unsupported NVDEC codec")
        );
        decoder.select_codec(9).unwrap();
        assert!(!decoder.h264_initialized);
        decoder.select_codec(3).unwrap();
        assert!(!decoder.h264_initialized);

        // Eight synthetic gray levels with two B pictures between references.
        // Decode order differs from display order; VIC must consume FFmpeg
        // receive order rather than selecting a picture by its decode surface.
        let reordered: [&[u8]; 8] = [
            &[
                0x65, 0x88, 0x84, 0x00, 0xff, 0xfe, 0xbd, 0x67, 0xe6, 0x59, 0x29, 0x1a, 0xdd, 0x24,
                0x5b, 0x49, 0xae, 0x1a, 0x81,
            ],
            &[
                0x41, 0x9a, 0x23, 0x6c, 0x7f, 0x5f, 0xcb, 0xff, 0xff, 0xe7, 0xcd, 0x56, 0x94, 0xf8,
                0xb2, 0x1a, 0x16, 0xf6, 0x96, 0xe0, 0x58,
            ],
            &[0x41, 0x9e, 0x41, 0x78, 0xaf, 0xda, 0x81],
            &[0x01, 0x9e, 0x62, 0x6a, 0x49, 0xff, 0xdb, 0x80],
            &[
                0x41, 0x9a, 0x66, 0x35, 0x08, 0x2d, 0x13, 0x29, 0x81, 0xff, 0x90, 0xff, 0x9e, 0x14,
                0x7f, 0xff, 0xf3, 0x43, 0xdb, 0xc0, 0xf4, 0x18, 0x5e, 0x61, 0x17, 0x61,
            ],
            &[0x41, 0x9e, 0x84, 0x6e, 0x51, 0x12, 0xd7, 0xda, 0x81],
            &[0x01, 0x9e, 0xa5, 0x6a, 0x49, 0xff, 0xdb, 0x81],
            &[
                0x41, 0x9a, 0xa7, 0x35, 0x08, 0x2d, 0x93, 0x29, 0x81, 0xff, 0x56, 0x5a, 0xff, 0xff,
                0xf3, 0x4a, 0xd2, 0x9a, 0x43, 0xca, 0x62, 0xf6, 0x85,
            ],
        ];
        let flags = (flags & !(3 << 14)) | (2 << 32);
        context[0x58..0x5c].copy_from_slice(&2_u32.to_le_bytes());
        for (i, slice) in reordered.into_iter().enumerate() {
            let frame = [0, 1, 2, 3, 3, 4, 5, 5][i];
            registers[0x10c + 4] = 0x30 + i as u32 * 8;
            registers[0x11d + 4] = 0x80 + i as u32 * 8;
            let mut packet = vec![0, 0, 1];
            packet.extend_from_slice(slice);
            context[0x48..0x4c].copy_from_slice(&(packet.len() as u32).to_le_bytes());
            context[0xb0..0xb8].copy_from_slice(&(flags | (frame << 46)).to_le_bytes());
            machine
                .write_resolved(&[input.slice(0, context.len() as u64).unwrap()], &context)
                .unwrap();
            machine
                .write_resolved(&[input.slice(0x400, packet.len() as u64).unwrap()], &packet)
                .unwrap();
            decoder.execute(&registers, &mappings).unwrap();
        }
        let produced = decoder.frames.len();
        assert!(produced >= 5);
        let expected = [48, 64, 80, 96, 112, 128, 144, 160];
        for &value in &expected[..produced] {
            let frame = decoder.pop_frame().unwrap();
            assert_eq!(frame.luma()[0], value);
        }
        assert!(decoder.frames.is_empty());
    }

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
                physical: nixe_gpu::physical::PhysicalMapping::new(objects[0].ranges().unwrap())
                    .unwrap(),
                object: objects[0].clone(),
                references: 1,
            },
        )]));
        let remapped = Arc::new(BTreeMap::from([
            (
                0x1000,
                Mapping {
                    physical: nixe_gpu::physical::PhysicalMapping::new(
                        objects[1].ranges().unwrap(),
                    )
                    .unwrap(),
                    object: objects[1].clone(),
                    references: 1,
                },
            ),
            (
                0x3000,
                Mapping {
                    physical: nixe_gpu::physical::PhysicalMapping::new(
                        objects[0].ranges().unwrap(),
                    )
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
