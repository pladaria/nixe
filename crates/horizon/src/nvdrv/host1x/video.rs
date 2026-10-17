//! NVDEC packet reconstruction and actual decoded surface writes.
//! Layout: https://github.com/averne/FFmpeg/blob/nvtegra/libavutil/nvdec_drv.h
//! NVDEC TBL uses 64x8-byte GOBs and at least two GOBs per block:
//! https://github.com/averne/FFmpeg/blob/nvtegra/libavutil/hwcontext_nvtegra.c
use super::Mapping;
use nixe_memory::{CanonicalBackingRange, CanonicalPageId, CanonicalWriteBatch};
use std::collections::BTreeMap;
use std::ffi::{CStr, c_char, c_int, c_void};
use std::sync::Arc;

unsafe extern "C" {
    fn nixe_nvdec_create(uuid: *const u8) -> *mut c_void;
    fn nixe_video_hardware_supported(uuid: *const u8) -> c_int;
    fn nixe_video_frame_context(frame: *const c_void) -> usize;
    fn nixe_video_transfer_create(
        frame: *const c_void,
        fd: i32,
        semaphore_fd: i32,
        size: u64,
        uuid: *const u8,
        pitch: u64,
        chroma: u64,
        initialize: c_int,
    ) -> *mut c_void;
    fn nixe_video_transfer_enqueue(target: *mut c_void, frame: *const c_void) -> c_int;
    fn nixe_video_transfer_wait(target: *mut c_void) -> c_int;
    fn nixe_video_transfer_free(target: *mut c_void);
    fn nixe_video_error() -> *const c_char;
    fn nixe_video_frame_free(frame: *mut c_void);
    fn nixe_video_frame_copy(
        frame: *mut c_void,
        fd: i32,
        semaphore_fd: i32,
        allocation_size: u64,
        uuid: *const u8,
        row_pitch: u64,
        chroma_offset: u64,
        initialize: c_int,
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
        planes: *mut *const u8,
        strides: *mut i32,
        device_frame: *mut *mut c_void,
    ) -> c_int;
}

#[derive(Debug)]
struct CudaFrame(*mut c_void);
// Immutable AVFrame reference pins its CUDA pixels and device context. Transfers
// execute on the backend owner; the producer cannot recycle the pinned surface.
unsafe impl Send for CudaFrame {}
unsafe impl Sync for CudaFrame {}
impl Drop for CudaFrame {
    fn drop(&mut self) {
        unsafe { nixe_video_frame_free(self.0) };
    }
}
impl nixe_gpu::DeviceImageSource for CudaFrame {
    fn transfer_context(&self) -> usize {
        unsafe { nixe_video_frame_context(self.0) }
    }
    fn import_nv12(
        &self,
        target: nixe_gpu::ExternalImageBuffer,
    ) -> Result<Box<dyn nixe_gpu::DeviceImageTransfer>, Box<str>> {
        let transfer = unsafe {
            nixe_video_transfer_create(
                self.0,
                target.fd,
                target.semaphore_fd,
                target.allocation_size,
                target.device_uuid.as_ptr(),
                target.row_pitch,
                target.chroma_offset,
                c_int::from(target.initialize),
            )
        };
        if transfer.is_null() {
            return Err(video_error());
        }
        Ok(Box::new(CudaImageTransfer(transfer)))
    }
    fn enqueue_nv12(&self, target: &mut dyn nixe_gpu::DeviceImageTransfer) -> Result<(), Box<str>> {
        let target = (target as &mut dyn std::any::Any)
            .downcast_mut::<CudaImageTransfer>()
            .ok_or_else(|| {
                Box::<str>::from("decoded frame and transfer use different producer APIs")
            })?;
        if unsafe { nixe_video_transfer_enqueue(target.0, self.0) } < 0 {
            Err(video_error())
        } else {
            Ok(())
        }
    }
    fn copy_nv12(&self, target: nixe_gpu::ExternalImageBuffer) -> Result<(), Box<str>> {
        let result = unsafe {
            nixe_video_frame_copy(
                self.0,
                target.fd,
                target.semaphore_fd,
                target.allocation_size,
                target.device_uuid.as_ptr(),
                target.row_pitch,
                target.chroma_offset,
                c_int::from(target.initialize),
            )
        };
        if result < 0 {
            Err(unsafe { CStr::from_ptr(nixe_video_error()) }
                .to_string_lossy()
                .into_owned()
                .into_boxed_str())
        } else {
            Ok(())
        }
    }
}

fn video_error() -> Box<str> {
    unsafe { CStr::from_ptr(nixe_video_error()) }
        .to_string_lossy()
        .into_owned()
        .into_boxed_str()
}

#[derive(Debug)]
struct CudaImageTransfer(*mut c_void);
// Imports retain their device context. The concrete Vulkan owner serializes
// copies; destruction occurs only after its actual consumer completion.
unsafe impl Send for CudaImageTransfer {}
impl nixe_gpu::DeviceImageTransfer for CudaImageTransfer {
    fn wait(&mut self) -> Result<(), Box<str>> {
        if unsafe { nixe_video_transfer_wait(self.0) } < 0 {
            Err(video_error())
        } else {
            Ok(())
        }
    }
}
impl Drop for CudaImageTransfer {
    fn drop(&mut self) {
        unsafe { nixe_video_transfer_free(self.0) };
    }
}

pub(super) type Mappings = Arc<BTreeMap<u32, Mapping>>;

pub(super) fn region(
    mappings: &Mappings,
    address: u64,
    length: usize,
) -> Result<(CanonicalBackingRange, u64), String> {
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
    Ok((
        mapping
            .object
            .backing()
            .ok_or("Host1x object has no canonical backing")?
            .clone(),
        offset,
    ))
}
pub(super) fn read(mappings: &Mappings, address: u64, length: usize) -> Result<Vec<u8>, String> {
    let (backing, offset) = region(mappings, address, length)?;
    let mut bytes = vec![0; length];
    backing
        .read(offset, &mut bytes)
        .map_err(|e| e.to_string())?;
    Ok(bytes)
}
pub(super) fn word(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}
fn surface_identity(mappings: &Mappings, address: u64) -> Result<(CanonicalPageId, u64), String> {
    let (backing, offset) = region(mappings, address, 1)?;
    let mut skip = offset;
    for segment in backing.segments() {
        if skip < segment.size() {
            return Ok((segment.page(), segment.offset() + skip));
        }
        skip -= segment.size();
    }
    Err("NVDEC surface has no canonical byte identity".into())
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

#[derive(Debug)]
struct Output {
    mappings: Mappings,
    luma: u64,
    chroma: u64,
    width: usize,
    height: usize,
    luma_pitch: usize,
    chroma_pitch: usize,
    block_height: u32,
}
#[derive(Debug)]
pub(super) struct NvDec {
    decoder: *mut c_void,
    next_origin: u64,
    outputs: BTreeMap<u64, Output>,
    surfaces: BTreeMap<[(CanonicalPageId, u64); 2], u64>,
}
// The codec is exclusively owned by one Host1x worker and never accessed concurrently.
unsafe impl Send for NvDec {}
impl Drop for NvDec {
    fn drop(&mut self) {
        unsafe { nixe_nvdec_destroy(self.decoder) };
    }
}
impl NvDec {
    pub(super) fn new(device: Option<[u8; 16]>) -> Result<Self, String> {
        let device = match device {
            Some(uuid) => match unsafe { nixe_video_hardware_supported(uuid.as_ptr()) } {
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
                device
                    .as_ref()
                    .map_or(std::ptr::null(), |uuid| uuid.as_ptr()),
            )
        };
        if decoder.is_null() {
            return Err(format!(
                "cannot initialize FFmpeg VP9 decoder: {}",
                unsafe { CStr::from_ptr(nixe_video_error()) }.to_string_lossy()
            ));
        }
        log::info!(
            "VP9 decoder selected: {}",
            if device.is_some() {
                "CUDA/NVDEC hardware, device-resident NV12"
            } else {
                "software"
            }
        );
        Ok(Self {
            decoder,
            next_origin: 1,
            outputs: BTreeMap::new(),
            surfaces: BTreeMap::new(),
        })
    }
    fn error(&self) -> String {
        unsafe { CStr::from_ptr(nixe_nvdec_error(self.decoder)) }
            .to_string_lossy()
            .into_owned()
    }
    pub(super) fn execute(
        &mut self,
        registers: &[u32],
        mappings: &Mappings,
        gpu: Option<&super::super::gpu_executor::NvDrvGpuExecutor>,
    ) -> Result<(), String> {
        if registers[0x80] != 9 {
            return Err(format!("unsupported NVDEC codec {:#x}", registers[0x80]));
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
                mappings: Arc::clone(mappings),
                luma: offsets[3],
                chroma: u64::from(registers[0x11d + 3]) << 8,
                width,
                height,
                luma_pitch,
                chroma_pitch,
                block_height,
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
        // Drain before another send, as required by FFmpeg's send/receive protocol.
        loop {
            let mut decoded_origin = 0;
            let mut decoded_width = 0;
            let mut decoded_height = 0;
            let mut planes = [std::ptr::null(); 3];
            let mut strides = [0; 3];
            let mut device_frame = std::ptr::null_mut();
            let result = unsafe {
                nixe_nvdec_receive(
                    self.decoder,
                    &mut decoded_origin,
                    &mut decoded_width,
                    &mut decoded_height,
                    planes.as_mut_ptr(),
                    strides.as_mut_ptr(),
                    &mut device_frame,
                )
            };
            if result < 0 {
                return Err(format!("VP9 receive failed: {}", self.error()));
            }
            if result == 0 {
                break;
            }
            let device_frame = (!device_frame.is_null()).then(|| CudaFrame(device_frame));
            let output = self
                .outputs
                .remove(&decoded_origin)
                .ok_or("NVDEC returned an unknown frame origin")?;
            if output.width != decoded_width as usize
                || output.height != decoded_height as usize
                || (device_frame.is_none()
                    && (strides[0] < output.width as i32
                        || strides[1] < output.width.div_ceil(2) as i32
                        || strides[2] < output.width.div_ceil(2) as i32
                        || planes.iter().any(|p| p.is_null())))
            {
                return Err("NVDEC output dimensions/planes do not match submitted surface".into());
            }
            if let Some(frame) = device_frame {
                let gpu = gpu.ok_or("hardware video requires an accelerated GPU backend")?;
                gpu.upload_device_planes(
                    output.images()?,
                    nixe_gpu::DeviceImageFrame {
                        width: decoded_width,
                        height: decoded_height,
                        source: Arc::new(frame),
                    },
                )?;
            } else if let Some(gpu) = gpu {
                output.upload(planes, strides, gpu)?;
            } else {
                output.write(planes, strides)?;
            }
            log::trace!(
                "NVDEC VP9 output: origin={decoded_origin} dimensions={decoded_width}x{decoded_height}"
            );
        }
        if self.outputs.contains_key(&origin) {
            return Err("NVDEC visible frame did not produce a surface before completion".into());
        }
        Ok(())
    }
}
impl Output {
    fn upload(
        &self,
        planes: [*const u8; 3],
        strides: [i32; 3],
        gpu: &super::super::gpu_executor::NvDrvGpuExecutor,
    ) -> Result<(), String> {
        let images = self.images()?;
        let bytes = self.linear_planes(planes, strides);
        gpu.upload_decoded_planes(
            images,
            bytes,
            [self.width as u32, self.width.div_ceil(2) as u32],
        )
    }

    fn images(&self) -> Result<[nixe_gpu::ConversionImage; 2], String> {
        let surface = |chroma| super::surface::Surface {
            width: if chroma {
                self.width.div_ceil(2) * 2
            } else {
                self.width
            },
            height: if chroma {
                self.height.div_ceil(2)
            } else {
                self.height
            },
            pitch: if chroma {
                self.chroma_pitch
            } else {
                self.luma_pitch
            },
            rows: if chroma {
                self.height.div_ceil(2)
            } else {
                self.height
            },
            kind: 1,
            block_height: self.block_height,
            format: 68,
        };
        let images = [
            surface(false).gpu_image(
                &self.mappings,
                self.luma,
                nixe_gpu::ImageFormat::R8Unorm,
                true,
            )?,
            surface(true).gpu_image(
                &self.mappings,
                self.chroma,
                nixe_gpu::ImageFormat::Rg8Unorm,
                true,
            )?,
        ];
        Ok(images)
    }

    // The FFI frame is borrowed only until the next receive. Own linear planes
    // across the frontend/backend queues without swizzling canonical memory.
    fn linear_planes(&self, planes: [*const u8; 3], strides: [i32; 3]) -> [Arc<[u8]>; 2] {
        let mut luma = Arc::<[u8]>::new_uninit_slice(self.width * self.height);
        let chroma_width = self.width.div_ceil(2);
        let mut uv = Arc::<[u8]>::new_uninit_slice(chroma_width * 2 * self.height.div_ceil(2));
        for (y, row) in Arc::get_mut(&mut luma)
            .unwrap()
            .chunks_exact_mut(self.width)
            .enumerate()
        {
            // SAFETY: receive validates positive strides and complete plane
            // dimensions; this copy finishes before the next receive.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    planes[0].add(y * strides[0] as usize),
                    row.as_mut_ptr().cast::<u8>(),
                    self.width,
                );
            }
        }
        for (y, row) in Arc::get_mut(&mut uv)
            .unwrap()
            .chunks_exact_mut(chroma_width * 2)
            .enumerate()
        {
            for x in 0..chroma_width {
                // SAFETY: both chroma planes have the validated row extents.
                unsafe {
                    row[x * 2].write(*planes[1].add(y * strides[1] as usize + x));
                    row[x * 2 + 1].write(*planes[2].add(y * strides[2] as usize + x));
                }
            }
        }
        // SAFETY: every byte of both tightly packed planes was initialized above.
        unsafe { [luma.assume_init(), uv.assume_init()] }
    }

    fn write(&self, planes: [*const u8; 3], strides: [i32; 3]) -> Result<(), String> {
        let mut writes = CanonicalWriteBatch::new();
        for chroma in [false, true] {
            let pitch = if chroma {
                self.chroma_pitch
            } else {
                self.luma_pitch
            };
            let width = if chroma {
                self.width.div_ceil(2) * 2
            } else {
                self.width
            };
            let rows = if chroma {
                self.height.div_ceil(2)
            } else {
                self.height
            };
            let length = pitch * rows.next_multiple_of(8 << self.block_height);
            let address = if chroma { self.chroma } else { self.luma };
            let (backing, offset) = region(&self.mappings, address, length)?;
            let mut bytes = vec![0; length];
            backing
                .read(offset, &mut bytes)
                .map_err(|e| e.to_string())?;
            // A GOB stores each 16-byte horizontal run contiguously. Copy runs
            // rather than recomputing the block-linear address for every byte.
            // Preserve padding in the backing snapshot, including partial runs.
            for y in 0..rows {
                // SAFETY: receive validated positive strides and complete plane
                // dimensions; FFmpeg keeps this borrowed frame alive until the
                // next receive. The luma row contains at least `width` bytes.
                let luma = if chroma {
                    None
                } else {
                    Some(unsafe {
                        std::slice::from_raw_parts(planes[0].add(y * strides[0] as usize), width)
                    })
                };
                for x in (0..width).step_by(16) {
                    let count = (width - x).min(16);
                    let target = gob_offset(x, y, pitch, self.block_height);
                    if let Some(luma) = luma {
                        bytes[target..target + count].copy_from_slice(&luma[x..x + count]);
                    } else {
                        let mut interleaved = [0; 16];
                        for (i, value) in interleaved[..count].iter_mut().enumerate() {
                            let channel = (x + i) % 2;
                            // SAFETY: the verified chroma rows contain ceil(width/2)
                            // values in each source plane, with positive strides.
                            *value = unsafe {
                                *planes[1 + channel]
                                    .add(y * strides[1 + channel] as usize + (x + i) / 2)
                            };
                        }
                        bytes[target..target + count].copy_from_slice(&interleaved[..count]);
                    }
                }
            }
            writes
                .stage(&backing, offset, &bytes)
                .map_err(|e| e.to_string())?;
        }
        writes.commit_ordered().map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nvdrv::nvmap::{NvMapAllocationMetadata, NvMapCpuMapping, NvMapObjects, NvMapOwner};
    use nixe_memory::{
        AddressSpaceId, CanonicalAllocation, GuestVirtualAddress, MemoryPermissions,
    };
    #[test]
    fn decoded_gob_runs_preserve_partial_rows_chroma_and_padding() {
        let allocation = CanonicalAllocation::zeroed(0x10000, 4096).unwrap();
        let mut nvmap = NvMapObjects::default();
        let owner = NvMapOwner::new(1);
        let handle = nvmap.create(owner, 0x10000).unwrap();
        nvmap
            .allocate(
                owner,
                handle,
                NvMapAllocationMetadata::new(0, 0, 4096, 0),
                NvMapCpuMapping::new(AddressSpaceId::new(1), GuestVirtualAddress::new(0x1000)),
                allocation
                    .backing_range(MemoryPermissions::READ_WRITE)
                    .unwrap(),
            )
            .unwrap();
        let mappings = Arc::new(BTreeMap::from([(
            0x1000,
            Mapping {
                object: nvmap
                    .object_snapshot_by_owned_handle(owner, handle)
                    .unwrap(),
                references: 1,
            },
        )]));
        let width = 67usize;
        let height = 37usize;
        let y = (0..73 * height).map(|i| (i * 7) as u8).collect::<Vec<_>>();
        let u = (0..37 * height.div_ceil(2))
            .map(|i| (i * 3) as u8)
            .collect::<Vec<_>>();
        let v = (0..39 * height.div_ceil(2))
            .map(|i| (255usize.wrapping_sub(i * 5)) as u8)
            .collect::<Vec<_>>();
        allocation.write(0, &vec![0xa5; 0x10000]).unwrap();
        let output = Output {
            mappings,
            luma: 0x2000,
            chroma: 0x5000,
            width,
            height,
            luma_pitch: 128,
            chroma_pitch: 128,
            block_height: 2,
        };
        let planes = [y.as_ptr(), u.as_ptr(), v.as_ptr()];
        let linear = output.linear_planes(planes, [73, 37, 39]);
        output.write(planes, [73, 37, 39]).unwrap();
        for chroma in [false, true] {
            let rows = if chroma { height.div_ceil(2) } else { height };
            let w = if chroma { width.div_ceil(2) * 2 } else { width };
            let length = 128 * rows.next_multiple_of(32);
            let mut expected = vec![0xa5; length];
            for row in 0..rows {
                for x in 0..w {
                    expected[gob_offset(x, row, 128, 2)] = if !chroma {
                        y[row * 73 + x]
                    } else if x % 2 == 0 {
                        u[row * 37 + x / 2]
                    } else {
                        v[row * 39 + x / 2]
                    };
                }
            }
            let mut actual = vec![0; length];
            allocation
                .read(if chroma { 0x4000 } else { 0x1000 }, &mut actual)
                .unwrap();
            assert_eq!(actual, expected);
            let pitch = if chroma { width.div_ceil(2) * 2 } else { width };
            for row in 0..rows {
                for x in 0..w {
                    assert_eq!(
                        linear[usize::from(chroma)][row * pitch + x],
                        expected[gob_offset(x, row, 128, 2)]
                    );
                }
            }
        }
    }

    #[test]
    fn reference_identity_survives_iova_reassignment_and_distinguishes_reused_addresses() {
        let mut nvmap = NvMapObjects::default();
        let owner = NvMapOwner::new(1);
        let mut objects = Vec::new();
        for base in [0x1000, 0x2000] {
            let handle = nvmap.create(owner, 4096).unwrap();
            let storage = CanonicalAllocation::zeroed(4096, 4096).unwrap();
            nvmap
                .allocate(
                    owner,
                    handle,
                    NvMapAllocationMetadata::new(0, 0, 4096, 0),
                    NvMapCpuMapping::new(AddressSpaceId::new(1), GuestVirtualAddress::new(base)),
                    storage
                        .backing_range(MemoryPermissions::READ_WRITE)
                        .unwrap(),
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
                object: objects[0].clone(),
                references: 1,
            },
        )]));
        let remapped = Arc::new(BTreeMap::from([
            (
                0x1000,
                Mapping {
                    object: objects[1].clone(),
                    references: 1,
                },
            ),
            (
                0x3000,
                Mapping {
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
