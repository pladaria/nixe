//! Shared packet decoding and immutable frames, independent of guest device APIs.
//! https://ffmpeg.org/doxygen/trunk/group__lavc__encdec.html
use std::{
    borrow::Cow,
    ffi::{CStr, c_char, c_int, c_void},
    sync::Arc,
};
unsafe extern "C" {
    fn nixe_decoder_create(codec: u32, uuid: *const u8) -> *mut c_void;
    fn nixe_video_hardware_supported(codec: u32, uuid: *const u8) -> c_int;
    fn nixe_decoder_send_packet(
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
    fn nixe_decoder_destroy(decoder: *mut c_void);
    fn nixe_decoder_error(decoder: *const c_void) -> *const c_char;
    fn nixe_decoder_flush(decoder: *mut c_void);
    fn nixe_decoder_receive(
        decoder: *mut c_void,
        origin: *mut u64,
        width: *mut u32,
        height: *mut u32,
        retained_frame: *mut *mut c_void,
    ) -> c_int;
}

#[derive(Clone, Copy)]
pub(crate) enum Plane<'a> {
    Packed { data: &'a [u8], stride: usize },
    Chroma(PlanarChroma<'a>),
}

#[derive(Clone, Copy)]
pub(crate) struct PlanarChroma<'a> {
    pub u: &'a [u8],
    pub v: &'a [u8],
    pub u_stride: usize,
    pub v_stride: usize,
}

impl PlanarChroma<'_> {
    pub fn interleave(self, row: usize, column: usize, output: &mut [u8]) {
        let u = &self.u[row * self.u_stride + column / 2..];
        let v = &self.v[row * self.v_stride + column / 2..];
        let (u, output) = if !column.is_multiple_of(2) && !output.is_empty() {
            output[0] = v[0];
            (&u[1..], &mut output[1..])
        } else {
            (u, output)
        };
        let v = &v[column % 2..];
        let count = output.len() / 2;
        let mut pairs = output.chunks_exact_mut(2);
        for ((pair, a), b) in pairs.by_ref().zip(&u[..count]).zip(&v[..count]) {
            pair.copy_from_slice(&[*a, *b]);
        }
        if let [last] = pairs.into_remainder() {
            *last = u[count];
        }
    }
}

#[derive(Debug)]
pub(crate) struct Frame {
    raw: *mut c_void,
    pub(crate) width: usize,
    pub(crate) height: usize,
    planes: [*const u8; 3],
    strides: [usize; 3],
}
// A pending publication moves its owned AVFrame reference to the writer.
// Each plane slice is borrowed only while that writer is copying pixels.
// SAFETY: this value owns an immutable AVFrame reference. FFmpeg buffer
// references are thread-safe; no mutable plane access crosses the handoff.
// https://ffmpeg.org/doxygen/trunk/group__lavu__buffer.html
unsafe impl Send for Frame {}

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
    pub(crate) fn luma(&self) -> &[u8] {
        // SAFETY: FFmpeg owns all visible rows with their validated positive
        // stride. Excluding final row padding keeps the slice within its plane.
        unsafe {
            std::slice::from_raw_parts(
                self.planes[0],
                (self.height - 1) * self.strides[0] + self.width,
            )
        }
    }
    pub(crate) fn chroma_pitch(&self) -> usize {
        if self.planes[2].is_null() {
            self.strides[1]
        } else {
            self.width.div_ceil(2) * 2
        }
    }
    pub(crate) fn luma_pitch(&self) -> usize {
        self.strides[0]
    }
    pub(crate) fn chroma_plane(&self) -> Plane<'_> {
        let samples = self.width.div_ceil(2);
        let rows = self.height.div_ceil(2);
        let plane = |i: usize, width| {
            // SAFETY: receive validated the positive stride and FFmpeg retains
            // these rows, including the final visible bytes (without padding).
            unsafe {
                std::slice::from_raw_parts(self.planes[i], (rows - 1) * self.strides[i] + width)
            }
        };
        if self.planes[2].is_null() {
            Plane::Packed {
                data: plane(1, samples * 2),
                stride: self.strides[1],
            }
        } else {
            Plane::Chroma(PlanarChroma {
                u: plane(1, samples),
                v: plane(2, samples),
                u_stride: self.strides[1],
                v_stride: self.strides[2],
            })
        }
    }

    pub(crate) fn chroma(&self) -> Cow<'_, [u8]> {
        match self.chroma_plane() {
            Plane::Packed { data, .. } => Cow::Borrowed(data),
            Plane::Chroma(uv) => {
                let width = self.width.div_ceil(2) * 2;
                let mut bytes = vec![0; width * self.height.div_ceil(2)];
                for (y, row) in bytes.chunks_exact_mut(width).enumerate() {
                    uv.interleave(y, 0, row);
                }
                Cow::Owned(bytes)
            }
        }
    }
    /// Own only the visible pixels needed by the asynchronous graphics owner.
    /// CPU VIC reads the retained planes directly instead of making this copy.
    pub(crate) fn linear_planes(
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

#[derive(Debug)]
pub(crate) struct Decoder {
    raw: *mut c_void,
}
// SAFETY: a decoder has one owner; FFmpeg contexts are never accessed concurrently.
unsafe impl Send for Decoder {}
impl Drop for Decoder {
    fn drop(&mut self) {
        unsafe { nixe_decoder_destroy(self.raw) };
    }
}
impl Decoder {
    pub fn new(codec: u32, device: Option<[u8; 16]>) -> Result<Self, String> {
        let name = match codec {
            3 => "H.264",
            9 => "VP9",
            _ => return Err(format!("unsupported video codec {codec}")),
        };
        let device = match device {
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
        let raw = unsafe {
            nixe_decoder_create(
                codec,
                device
                    .as_ref()
                    .map_or(std::ptr::null(), |uuid| uuid.as_ptr()),
            )
        };
        if raw.is_null() {
            return Err(format!(
                "cannot initialize FFmpeg {name} decoder: {}",
                video_error()
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
        Ok(Self { raw })
    }
    /// The NVDEC reconstruction bridge feeds the same owned codec context.
    pub fn raw(&mut self) -> *mut c_void {
        self.raw
    }
    pub fn error(&self) -> String {
        unsafe { CStr::from_ptr(nixe_decoder_error(self.raw)) }
            .to_string_lossy()
            .into_owned()
    }
    pub fn send(&mut self, packet: &[u8], origin: u64) -> Result<(), String> {
        if packet.is_empty() || packet.len() > i32::MAX as usize || origin == i64::MIN as u64 {
            return Err("invalid video packet size or origin".into());
        }
        // The token round-trips through FFmpeg's signed PTS field. Zero and
        // negative timestamps are valid; only AV_NOPTS_VALUE is reserved.
        if unsafe { nixe_decoder_send_packet(self.raw, packet.as_ptr(), packet.len(), origin) } < 0
        {
            return Err(self.error());
        }
        Ok(())
    }
    pub fn finish(&mut self) -> Result<(), String> {
        if unsafe { nixe_decoder_send_packet(self.raw, std::ptr::null(), 0, 0) } < 0 {
            return Err(self.error());
        }
        Ok(())
    }
    pub fn flush(&mut self) {
        unsafe { nixe_decoder_flush(self.raw) };
    }
    pub fn receive(&mut self) -> Result<Option<(u64, Frame)>, String> {
        let (mut origin, mut width, mut height) = (0, 0, 0);
        let mut raw = std::ptr::null_mut();
        let result = unsafe {
            nixe_decoder_receive(self.raw, &mut origin, &mut width, &mut height, &mut raw)
        };
        match result {
            0 => Ok(None),
            1 => Ok(Some((
                origin,
                Frame::new(raw, width as usize, height as usize)?,
            ))),
            _ => Err(self.error()),
        }
    }
}
