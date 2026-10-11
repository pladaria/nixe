//! Tegra VIC progressive surface conversion. All conversion coefficients come from the guest.
//! Public field layout and fixed point rectangles:
//! https://github.com/averne/FFmpeg/blob/caeec83b791be08ed43468a2ef426d6901d51c78/libavutil/vic_drv.h
//! https://github.com/averne/FFmpeg/blob/caeec83b791be08ed43468a2ef426d6901d51c78/libavutil/hwcontext_nvtegra.c
use super::video::{Mappings, gob_offset, read, region, word};
use nixe_memory::CanonicalWriteBatch;

fn wide(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}
fn bits(value: u64, start: u32, count: u32) -> u64 {
    (value >> start) & ((1 << count) - 1)
}
fn signed20(value: u64, shift: u32) -> i64 {
    ((bits(value, shift, 20) as i64) << 44) >> 44
}
#[derive(Clone, Copy)]
struct Matrix {
    coefficients: [[i64; 4]; 3],
    shift: u32,
    enabled: bool,
}
impl Matrix {
    fn decode(bytes: &[u8]) -> Self {
        let columns = std::array::from_fn::<_, 4, _>(|i| wide(bytes, i * 8));
        Self {
            coefficients: std::array::from_fn(|r| {
                std::array::from_fn(|c| signed20(columns[c], r as u32 * 20))
            }),
            shift: (columns[0] >> 60) as u32,
            enabled: columns[1] >> 63 != 0,
        }
    }
    fn apply(&self, pixel: [i64; 4]) -> [i64; 4] {
        if !self.enabled {
            return pixel;
        }
        let mut result = pixel;
        for (row, value) in result[..3].iter_mut().enumerate() {
            let c = self.coefficients[row];
            *value =
                (((pixel[0] * c[0] + pixel[1] * c[1] + pixel[2] * c[2]) >> self.shift) + c[3]) >> 8;
        }
        result
    }
}
struct Surface {
    width: usize,
    height: usize,
    pitch: usize,
    rows: usize,
    kind: u32,
    block_height: u32,
    format: u32,
}
impl Surface {
    fn gpu_image(
        &self,
        mappings: &Mappings,
        address: u64,
        format: nixe_gpu::ImageFormat,
        plane: bool,
    ) -> Result<nixe_gpu::ConversionImage, String> {
        use nixe_gpu::*;
        let (whole, offset) = region(mappings, address, self.length())?;
        let (_, mapping) = mappings
            .range(..=u32::try_from(address).map_err(|_| "VIC address exceeds 32 bits")?)
            .next_back()
            .ok_or("VIC mapping was lost")?;
        let allocation =
            GpuAllocationDescription::new(whole.size(), 1).map_err(|e| e.to_string())?;
        let bytes = usize::from(
            format
                .plane_bytes_per_texel(0)
                .ok_or("invalid VIC GPU image format")?,
        );
        // NVDEC may allocate a wider GOB stride than its visible source. Plane
        // textures retain that stride; the validated source crop selects texels.
        let width = if plane && self.kind == 1 {
            self.pitch / bytes
        } else {
            self.width / if plane { bytes } else { 1 }
        };
        let description = ImageDescription::new(
            ImageDimension::Two,
            ImageExtent::new(width as u32, self.height as u32, 1).map_err(|e| e.to_string())?,
            format,
            ImageKind::Color,
            1,
            1,
            SampleCount::One,
        )
        .map_err(|e| e.to_string())?;
        let backing = BackingView::new(
            GpuAllocationId::new(mapping.object.id().raw()),
            allocation,
            offset,
            whole
                .snapshot_subrange(offset, self.length() as u64)
                .map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        let layout = if self.kind == 1 {
            ImageMemoryLayout::BlockLinear(BlockLinearLayout {
                block_width_log2: 0,
                block_height_log2: self.block_height as u8,
                block_depth_log2: 0,
                layer_stride: self.length() as u64,
            })
        } else {
            ImageMemoryLayout::PitchLinear {
                row_pitch: self.pitch as u64,
                layer_stride: self.length() as u64,
            }
        };
        let view = ImageView::new(
            ImageId::new(0),
            description,
            Swizzle::IDENTITY,
            vec![(
                ImageSubresourceRange {
                    mip_level: 0,
                    base_layer: 0,
                    layer_count: 1,
                    plane: 0,
                },
                layout,
                backing,
            )],
        )
        .map_err(|e| e.to_string())?;
        Ok(ConversionImage {
            description,
            allocation,
            view,
        })
    }
    fn decode(bytes: &[u8], bytes_per_pixel: usize) -> Result<Self, String> {
        let layout = word(bytes, 0);
        let size = word(bytes, 4);
        let allocation = word(bytes, 8);
        let width = (size & 0x3fff) as usize + 1;
        let height = (size >> 14 & 0x3fff) as usize + 1;
        let pitch = ((allocation & 0x3fff) as usize + 1) * bytes_per_pixel;
        let rows = (allocation >> 14 & 0x3fff) as usize + 1;
        let kind = layout >> 11 & 15;
        let block_height = layout >> 15 & 15;
        if kind > 1 || block_height > 5 || width * bytes_per_pixel > pitch || height > rows {
            return Err("invalid/unsupported VIC surface layout".into());
        }
        Ok(Self {
            width,
            height,
            pitch: if kind == 1 {
                pitch.next_multiple_of(64)
            } else {
                pitch.next_multiple_of(16)
            },
            rows,
            kind,
            block_height,
            format: layout & 127,
        })
    }
    fn length(&self) -> usize {
        self.pitch
            * if self.kind == 1 {
                self.rows.next_multiple_of(8 << self.block_height)
            } else {
                self.rows
            }
    }
    fn offset(&self, x: usize, y: usize) -> usize {
        if self.kind == 1 {
            gob_offset(x, y, self.pitch, self.block_height)
        } else {
            y * self.pitch + x
        }
    }
    fn read_linear(&self, mappings: &Mappings, address: u64) -> Result<Vec<u8>, String> {
        let stored = read(mappings, address, self.length())?;
        if self.kind == 0 {
            return Ok(stored);
        }
        let mut linear = vec![0; self.pitch * self.height];
        for y in 0..self.height {
            for x in (0..self.width).step_by(16) {
                let count = (self.width - x).min(16);
                let source = self.offset(x, y);
                let target = y * self.pitch + x;
                linear[target..target + count].copy_from_slice(&stored[source..source + count]);
            }
        }
        Ok(linear)
    }
}
fn factor(
    selector: u32,
    k1: i64,
    k2: i64,
    src_alpha: i64,
    dst_alpha: i64,
    channel: usize,
) -> Result<i64, String> {
    if channel < 3 {
        match selector {
            0 => Ok(k1),
            1 => Ok(k1 * dst_alpha / 1023),
            2 => Ok(k1 * (1023 - dst_alpha) / 1023),
            3 => Ok(k1 * src_alpha / 1023),
            4 => Ok(0),
            _ => Err("invalid VIC source color blend factor".into()),
        }
    } else {
        match selector {
            0 => Ok(k1),
            1 => Ok(k2),
            2 => Ok(k1 * (1023 - dst_alpha) / 1023),
            3 => Ok(0),
            _ => Err("unsupported VIC source alpha blend factor".into()),
        }
    }
}
fn dst_factor(
    selector: u32,
    k1: i64,
    k2: i64,
    src_alpha: i64,
    dst_alpha: i64,
    channel: usize,
) -> Result<i64, String> {
    if channel < 3 {
        match selector {
            0 => Ok(k1),
            1 => Ok(k2),
            2 => Ok(k1 * dst_alpha / 1023),
            3 => Ok(k1 * (1023 - dst_alpha) / 1023),
            4 => Ok(k1 * (1023 - src_alpha) / 1023),
            5 => Ok(0),
            6 => Ok(1023),
            _ => Err("invalid VIC destination color blend factor".into()),
        }
    } else {
        match selector {
            0 => Ok(k2),
            1 => Ok(k1 * (1023 - src_alpha) / 1023),
            2 => Ok(0),
            3 => Ok(1023),
            _ => Err("unsupported VIC destination alpha blend factor".into()),
        }
    }
}
/// Exact S12.8 matrix evaluation for an 8-bit NV12 source. Precompute the
/// separable products once; retain both signed shifts and the guest translation.
struct Nv12Matrix {
    products: [[[i64; 256]; 3]; 3],
    matrix: Matrix,
}
impl Nv12Matrix {
    fn new(matrix: Matrix) -> Self {
        Self {
            products: std::array::from_fn(|row| {
                std::array::from_fn(|component| {
                    std::array::from_fn(|value| {
                        (value as i64 * 4) * matrix.coefficients[row][component]
                    })
                })
            }),
            matrix,
        }
    }
    fn rgb(&self, yuv: [u8; 3], minimum: i64, maximum: i64) -> [u8; 3] {
        std::array::from_fn(|row| {
            let value = if self.matrix.enabled {
                let p = &self.products[row];
                (((p[0][yuv[0] as usize] + p[1][yuv[1] as usize] + p[2][yuv[2] as usize])
                    >> self.matrix.shift)
                    + self.matrix.coefficients[row][3])
                    >> 8
            } else {
                i64::from(yuv[row]) * 4
            };
            (value.clamp(minimum, maximum).clamp(0, 1023) >> 2) as u8
        })
    }
}

// SSE4.1 retains the VIC integer arithmetic exactly, processing four pixels.
// Signed 20-bit coefficients multiplied by three 10-bit components sum to
// at most 3 * 1020 * 524288, so every intermediate fits signed 32 bits.
// https://doc.rust-lang.org/core/arch/x86_64/fn._mm_mullo_epi32.html
// https://doc.rust-lang.org/core/arch/x86_64/fn._mm_sra_epi32.html
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn rgba4(
    matrix: Matrix,
    luma: &[u8],
    uv: &[u8],
    x: usize,
    minimum: i64,
    maximum: i64,
    alpha: u8,
    bgra: bool,
) -> [u8; 16] {
    use std::arch::x86_64::*;
    let ys = u32::from_le_bytes(luma[x..x + 4].try_into().unwrap());
    let y = _mm_slli_epi32::<2>(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(ys as i32)));
    let u = _mm_slli_epi32::<2>(_mm_setr_epi32(
        uv[x / 2 * 2] as i32,
        uv[x.div_ceil(2) * 2] as i32,
        uv[(x + 2) / 2 * 2] as i32,
        uv[(x + 3) / 2 * 2] as i32,
    ));
    let v = _mm_slli_epi32::<2>(_mm_setr_epi32(
        uv[x / 2 * 2 + 1] as i32,
        uv[x.div_ceil(2) * 2 + 1] as i32,
        uv[(x + 2) / 2 * 2 + 1] as i32,
        uv[(x + 3) / 2 * 2 + 1] as i32,
    ));
    let input = [y, u, v];
    let mut rgb = input;
    for row in 0..3 {
        let mut value = input[row];
        if matrix.enabled {
            let c = matrix.coefficients[row];
            value = _mm_mullo_epi32(y, _mm_set1_epi32(c[0] as i32));
            value = _mm_add_epi32(value, _mm_mullo_epi32(u, _mm_set1_epi32(c[1] as i32)));
            value = _mm_add_epi32(value, _mm_mullo_epi32(v, _mm_set1_epi32(c[2] as i32)));
            value = _mm_sra_epi32(value, _mm_cvtsi32_si128(matrix.shift as i32));
            value = _mm_add_epi32(value, _mm_set1_epi32(c[3] as i32));
            value = _mm_srai_epi32::<8>(value);
        }
        value = _mm_max_epi32(value, _mm_set1_epi32(minimum as i32));
        value = _mm_min_epi32(value, _mm_set1_epi32(maximum as i32));
        value = _mm_max_epi32(value, _mm_setzero_si128());
        value = _mm_min_epi32(value, _mm_set1_epi32(1023));
        rgb[row] = _mm_srli_epi32::<2>(value);
    }
    if bgra {
        rgb.swap(0, 2);
    }
    let packed = _mm_or_si128(
        _mm_or_si128(rgb[0], _mm_slli_epi32::<8>(rgb[1])),
        _mm_or_si128(
            _mm_slli_epi32::<16>(rgb[2]),
            _mm_set1_epi32(i32::from(alpha) << 24),
        ),
    );
    let mut bytes = [0; 16];
    // SAFETY: bytes provides exactly the 16 writable bytes required by storeu.
    unsafe {
        _mm_storeu_si128(bytes.as_mut_ptr().cast(), packed);
    }
    bytes
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[allow(clippy::too_many_arguments)]
unsafe fn convert_opaque_nv12_sse41(
    bytes: &mut [u8],
    output: &Surface,
    input: &Surface,
    chroma: &Surface,
    luma: &[u8],
    uv: &[u8],
    color: Matrix,
    minimum: i64,
    maximum: i64,
    alpha: u8,
    left: usize,
    top: usize,
) {
    let scalar = Nv12Matrix::new(color);
    let bgra = matches!(output.format, 32 | 36);
    // Keep the complete raster loop inside the feature-enabled function. Calling
    // a target_feature function separately for every four pixels prevents normal
    // inlining/loop-invariant hoisting across the target-feature boundary.
    for y in 0..output.height {
        let sy = top + y;
        let luma = &luma[sy * input.pitch..];
        let uv = &uv[(sy / 2) * chroma.pitch..];
        for x0 in (0..output.width).step_by(4) {
            let count = (output.width - x0).min(4);
            let mut packed = [0; 16];
            if count == 4 {
                // SAFETY: this whole function requires SSE4.1; validated surface
                // extents/crop provide complete luma pixels and their NV12 pairs.
                packed =
                    unsafe { rgba4(color, luma, uv, left + x0, minimum, maximum, alpha, bgra) };
            } else {
                for i in 0..count {
                    let x = left + x0 + i;
                    let mut rgb = scalar.rgb(
                        [luma[x], uv[x / 2 * 2], uv[x / 2 * 2 + 1]],
                        minimum,
                        maximum,
                    );
                    if bgra {
                        rgb.swap(0, 2);
                    }
                    packed[i * 4..i * 4 + 3].copy_from_slice(&rgb);
                    packed[i * 4 + 3] = alpha;
                }
            }
            let target = output.offset(x0 * 4, y);
            bytes[target..target + count * 4].copy_from_slice(&packed[..count * 4]);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn write_opaque_nv12(
    output: &Surface,
    input: &Surface,
    chroma: &Surface,
    luma: &[u8],
    uv: &[u8],
    color: Matrix,
    minimum: i64,
    maximum: i64,
    alpha: u8,
    left: usize,
    top: usize,
    mappings: &Mappings,
    address: u64,
) -> Result<(), String> {
    // VIDEO_PROBE_BEGIN
    let opaque_start = std::time::Instant::now();
    // VIDEO_PROBE_END
    let matrix = Nv12Matrix::new(color);
    #[cfg(target_arch = "x86_64")]
    let simd = std::is_x86_feature_detected!("sse4.1");
    let (backing, offset) = region(mappings, address, output.length())?;
    let mut bytes = vec![0; output.length()];
    backing
        .read(offset, &mut bytes)
        .map_err(|e| e.to_string())?;
    // VIDEO_PROBE_BEGIN
    let opaque_read_us = opaque_start.elapsed().as_micros();
    let opaque_kernel_start = std::time::Instant::now();
    // VIDEO_PROBE_END
    #[cfg(target_arch = "x86_64")]
    if simd {
        // SAFETY: the runtime feature check covers the complete raster loop.
        unsafe {
            convert_opaque_nv12_sse41(
                &mut bytes, output, input, chroma, luma, uv, color, minimum, maximum, alpha, left,
                top,
            );
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    let simd = false;
    if !simd {
        for y in 0..output.height {
            let sy = top + y;
            for x0 in (0..output.width).step_by(4) {
                let count = (output.width - x0).min(4);
                let mut packed = [0; 16];
                for i in 0..count {
                    let sx = left + x0 + i;
                    let cx = sx / 2 * 2;
                    let cy = sy / 2;
                    let mut rgb = matrix.rgb(
                        [
                            luma[sy * input.pitch + sx],
                            uv[cy * chroma.pitch + cx],
                            uv[cy * chroma.pitch + cx + 1],
                        ],
                        minimum,
                        maximum,
                    );
                    if matches!(output.format, 32 | 36) {
                        rgb.swap(0, 2);
                    }
                    packed[i * 4..i * 4 + 3].copy_from_slice(&rgb);
                    packed[i * 4 + 3] = alpha;
                }
                let target = output.offset(x0 * 4, y);
                bytes[target..target + count * 4].copy_from_slice(&packed[..count * 4]);
            }
        }
    }
    // VIDEO_PROBE_BEGIN
    let opaque_kernel_us = opaque_kernel_start.elapsed().as_micros();
    let opaque_write_start = std::time::Instant::now();
    // VIDEO_PROBE_END
    let mut writes = CanonicalWriteBatch::new();
    writes
        .stage(&backing, offset, &bytes)
        .map_err(|e| e.to_string())?;
    writes.commit_ordered().map_err(|e| e.to_string())?;
    // VIDEO_PROBE_BEGIN
    if std::env::var_os("NIXE_VIDEO_AUDIT").is_some() {
        log::debug!(
            "VIDEO_PROBE vic-stages: read={opaque_read_us} kernel={opaque_kernel_us} write={}",
            opaque_write_start.elapsed().as_micros()
        );
    }
    // VIDEO_PROBE_END
    Ok(())
}

#[cfg(test)]
pub(super) fn execute(registers: &[u32], mappings: &Mappings) -> Result<(), String> {
    execute_with_gpu(registers, mappings, None)
}

pub(super) fn execute_with_gpu(
    registers: &[u32],
    mappings: &Mappings,
    gpu: Option<&super::super::gpu_executor::NvDrvGpuExecutor>,
) -> Result<(), String> {
    // VIDEO_PROBE_BEGIN
    let gpu = gpu.filter(|_| std::env::var_os("NIXE_CPU_VIC").is_none());
    // VIDEO_PROBE_END
    // VIDEO_PROBE_BEGIN
    let probe_start = std::time::Instant::now();
    // VIDEO_PROBE_END
    let config = read(mappings, u64::from(registers[0x1c2]) << 8, 0x610)?;
    let output = Surface::decode(&config[0x20..0x30], 4)?;
    if !matches!(output.format, 31 | 32 | 35 | 36) {
        return Err(format!(
            "unsupported VIC output pixel format {}",
            output.format
        ));
    }
    let out_config = wide(&config, 0x10);
    if bits(out_config, 46, 5) != 0 {
        return Err("VIC output regamma/flip/transpose is not implemented".into());
    }
    let alpha_mode = bits(out_config, 0, 3);
    let alpha_slot = bits(out_config, 3, 3) as usize;
    if alpha_mode > 5 {
        return Err("invalid VIC output alpha mode".into());
    }
    let background = [
        bits(out_config, 16, 10) as i64,
        bits(out_config, 26, 10) as i64,
        bits(out_config, 36, 10) as i64,
        bits(out_config, 6, 10) as i64,
    ];
    let final_matrix = Matrix::decode(&config[0x30..0x50]);
    let single_slot = (0..8)
        .filter(|slot| wide(&config, 0x90 + slot * 0xb0) & 1 != 0)
        .count()
        == 1;
    let mut pixels = Vec::new();
    let mut selected_alpha = Vec::new();
    let target_x = word(&config, 0x18);
    let target_y = word(&config, 0x1c);
    let target_left = (target_x & 0x3fff) as usize;
    let target_right = (target_x >> 16 & 0x3fff) as usize;
    let target_top = (target_y & 0x3fff) as usize;
    let target_bottom = (target_y >> 16 & 0x3fff) as usize;
    for slot in 0..8 {
        let start = 0x90 + slot * 0xb0;
        let state = &config[start..start + 0xb0];
        let flags = wide(state, 0);
        if flags & 1 == 0 {
            continue;
        }
        let controls = wide(state, 16);
        log::trace!(
            "VIC slot: index={slot} config={:016x?} surface={:08x?}",
            (0..7).map(|i| wide(state, i * 8)).collect::<Vec<_>>(),
            (0..4)
                .map(|i| word(state, 0x40 + i * 4))
                .collect::<Vec<_>>()
        );
        if flags & 0xfe != 0
            || bits(flags, 9, 11) != 0
            || bits(flags, 24, 12) != 0
            || bits(controls, 43, 16) != 0
            || bits(wide(state, 0x58), 40, 1) != 0
        {
            return Err(format!(
                "unsupported VIC slot processing: flags={flags:#x} controls={controls:#x}"
            ));
        }
        if flags & (1 << 8) == 0 {
            return Err("VIC enabled slot lacks its current input field".into());
        }
        let input = Surface::decode(&state[0x40..0x50], 1)?;
        if input.format != 68 {
            return Err(format!(
                "unsupported VIC input pixel format {}",
                input.format
            ));
        }
        let chroma_size = word(state, 0x4c);
        let chroma = Surface {
            width: input.width.div_ceil(2) * 2,
            height: input.height.div_ceil(2),
            pitch: (((chroma_size & 0x3fff) as usize + 1) * 2)
                .next_multiple_of(if input.kind == 1 { 64 } else { 16 }),
            rows: (chroma_size >> 14 & 0x3fff) as usize + 1,
            kind: input.kind,
            block_height: input.block_height,
            format: input.format,
        };
        if chroma.pitch < chroma.width || chroma.rows < chroma.height {
            return Err("invalid VIC chroma allocation".into());
        }
        let left = (word(state, 0x20) & 0x3fffffff) as usize;
        let right = (word(state, 0x24) & 0x3fffffff) as usize;
        let top = (word(state, 0x28) & 0x3fffffff) as usize;
        let bottom = (word(state, 0x2c) & 0x3fffffff) as usize;
        let dx = word(state, 0x30);
        let dy = word(state, 0x34);
        let dl = (dx & 0x3fff) as usize;
        let dr = (dx >> 16 & 0x3fff) as usize;
        let dt = (dy & 0x3fff) as usize;
        let db = (dy >> 16 & 0x3fff) as usize;
        if right < left
            || bottom < top
            || dr < dl
            || db < dt
            || [left, right, top, bottom].iter().any(|v| v & 0xffff != 0)
            || right / 65536 - left / 65536 != dr - dl
            || bottom / 65536 - top / 65536 != db - dt
        {
            return Err("VIC scaling/subpixel rectangles are not implemented".into());
        }
        if right / 65536 >= input.width || bottom / 65536 >= input.height {
            return Err("VIC source rectangle exceeds surface".into());
        }
        let color = Matrix::decode(&state[0x60..0x80]);
        let gamut = Matrix::decode(&state[0x80..0xa0]);
        let minimum = bits(controls, 0, 10) as i64;
        let maximum = bits(controls, 10, 10) as i64;
        if minimum > maximum {
            return Err("invalid VIC soft clamp range".into());
        }
        let alpha = bits(controls, 32, 10) as i64;
        let blend = &state[0xa0..0xb0];
        let constants = word(blend, 0);
        let factors = word(blend, 4);
        let overrides = word(blend, 12);
        if overrides & 0x000f_f000 != 0 {
            return Err("VIC blend channel overrides/masks are not implemented".into());
        }
        let k1 = (constants & 1023) as i64;
        let k2 = (constants >> 16 & 1023) as i64;
        let selectors = [
            factors & 7,
            factors >> 4 & 7,
            factors >> 8 & 7,
            factors >> 12 & 7,
        ];
        // VIDEO_PROBE_BEGIN
        if std::env::var_os("NIXE_VIDEO_AUDIT").is_some() {
            log::debug!(
                "VIDEO_PROBE vic-config: dimensions={}x{} color={:?}/{}:{} gamut={:?}/{}:{} blend={selectors:?} constants={k1}/{k2} alpha={alpha}",
                output.width,
                output.height,
                color.coefficients,
                color.shift,
                color.enabled,
                gamut.coefficients,
                gamut.shift,
                gamut.enabled
            );
        }
        // VIDEO_PROBE_END
        let x0 = dl.max(target_left);
        let x1 = dr.min(target_right).min(output.width - 1);
        let y0 = dt.max(target_top);
        let y1 = db.min(target_bottom).min(output.height - 1);
        if x0 > x1 || y0 > y1 {
            continue;
        }
        // A single full-surface opaque NV12 slot does not need a compositor
        // framebuffer, alpha array, blend divides or a second full-pixel pass.
        // Select this from actual VIC state; matrices, clamp, format, source
        // crop and alpha still have their exact guest-defined semantics.
        let source_alpha = alpha.clamp(minimum, maximum);
        if single_slot
            && !gamut.enabled
            && !final_matrix.enabled
            && x0 == 0
            && y0 == 0
            && x1 + 1 == output.width
            && y1 + 1 == output.height
            && factor(selectors[0], k1, k2, source_alpha, background[3], 0)? == 1023
            && dst_factor(selectors[1], k1, k2, source_alpha, background[3], 0)? == 0
        {
            let sf = factor(selectors[2], k1, k2, source_alpha, background[3], 3)?;
            let df = dst_factor(selectors[3], k1, k2, source_alpha, background[3], 3)?;
            let composited_alpha =
                ((source_alpha * sf + background[3] * df + 511) / 1023).clamp(0, 1023);
            let final_alpha = if matches!(output.format, 35 | 36) {
                1023
            } else {
                match alpha_mode {
                    0 => 1023,
                    1 => background[3],
                    2 | 4 => composited_alpha,
                    3 | 5 => {
                        if slot == alpha_slot {
                            source_alpha
                        } else {
                            0
                        }
                    }
                    _ => unreachable!(),
                }
            };
            // VIDEO_PROBE_BEGIN
            if std::env::var_os("NIXE_VIC_NOOP").is_some() {
                log::debug!("VIDEO_PROBE vic-noop");
                return Ok(());
            }
            // VIDEO_PROBE_END
            if let Some(gpu) = gpu
                && (output.kind == 0 || output.pitch == (output.width * 4).next_multiple_of(64))
            {
                let format = if matches!(output.format, 32 | 36) {
                    nixe_gpu::ImageFormat::Bgra8Unorm
                } else {
                    nixe_gpu::ImageFormat::Rgba8Unorm
                };
                let images = [
                    input.gpu_image(
                        mappings,
                        u64::from(registers[0x100 + slot * 24]) << 8,
                        nixe_gpu::ImageFormat::R8Unorm,
                        true,
                    )?,
                    chroma.gpu_image(
                        mappings,
                        u64::from(registers[0x101 + slot * 24]) << 8,
                        nixe_gpu::ImageFormat::Rg8Unorm,
                        true,
                    )?,
                    output.gpu_image(mappings, u64::from(registers[0x1c8]) << 8, format, false)?,
                ];
                gpu.convert_nv12(
                    images,
                    nixe_gpu::Nv12ColorTransform {
                        coefficients: color.coefficients.map(|row| row.map(|c| c as i32)),
                        shift: color.shift,
                        enabled: color.enabled,
                        clamp: [minimum as i32, maximum as i32],
                        alpha: (final_alpha.clamp(0, 1023) >> 2) as u8,
                        source_origin: [
                            (left / 65536 + x0 - dl) as u32,
                            (top / 65536 + y0 - dt) as u32,
                        ],
                    },
                )?;
                // VIDEO_PROBE_BEGIN
                if std::env::var_os("NIXE_GPU_VIC_DELAY").is_some() {
                    std::thread::sleep(std::time::Duration::from_millis(12));
                }
                if std::env::var_os("NIXE_GPU_VIC_READBACK").is_some() {
                    let (backing, offset) = region(mappings, u64::from(registers[0x1c8]) << 8, output.length())?;
                    let mut pixels = vec![0; output.length()];
                    backing.read(offset, &mut pixels).map_err(|e| e.to_string())?;
                }
                if std::env::var_os("NIXE_VIDEO_AUDIT").is_some() {
                    log::debug!(
                        "VIDEO_PROBE vic-gpu: dimensions={}x{} us={}",
                        output.width,
                        output.height,
                        probe_start.elapsed().as_micros()
                    );
                }
                // VIDEO_PROBE_END
                return Ok(());
            }
            let luma = input.read_linear(mappings, u64::from(registers[0x100 + slot * 24]) << 8)?;
            let uv = chroma.read_linear(mappings, u64::from(registers[0x101 + slot * 24]) << 8)?;
            write_opaque_nv12(
                &output,
                &input,
                &chroma,
                &luma,
                &uv,
                color,
                minimum,
                maximum,
                (final_alpha.clamp(0, 1023) >> 2) as u8,
                left / 65536 + x0 - dl,
                top / 65536 + y0 - dt,
                mappings,
                u64::from(registers[0x1c8]) << 8,
            )?;
            // VIDEO_PROBE_BEGIN
            if std::env::var_os("NIXE_VIDEO_AUDIT").is_some() {
                log::debug!(
                    "VIDEO_PROBE vic-opaque: dimensions={}x{} us={}",
                    output.width,
                    output.height,
                    probe_start.elapsed().as_micros()
                );
            }
            // VIDEO_PROBE_END
            return Ok(());
        }
        if pixels.is_empty() {
            pixels.resize(output.width * output.height, background);
            selected_alpha.resize(pixels.len(), 0);
        }
        let luma = input.read_linear(mappings, u64::from(registers[0x100 + slot * 24]) << 8)?;
        let uv = chroma.read_linear(mappings, u64::from(registers[0x101 + slot * 24]) << 8)?;
        for y in y0..=y1 {
            for x in x0..=x1 {
                let sx = left / 65536 + x - dl;
                let sy = top / 65536 + y - dt;
                let cx = sx / 2 * 2;
                let cy = sy / 2;
                let source = [
                    i64::from(luma[sy * input.pitch + sx]) << 2,
                    i64::from(uv[cy * chroma.pitch + cx]) << 2,
                    i64::from(uv[cy * chroma.pitch + cx + 1]) << 2,
                    alpha,
                ];
                let source = gamut
                    .apply(color.apply(source))
                    .map(|v| v.clamp(minimum, maximum));
                let index = y * output.width + x;
                let destination = pixels[index];
                for channel in 0..4 {
                    let sf = factor(
                        selectors[if channel == 3 { 2 } else { 0 }],
                        k1,
                        k2,
                        source[3],
                        destination[3],
                        channel,
                    )?;
                    let df = dst_factor(
                        selectors[if channel == 3 { 3 } else { 1 }],
                        k1,
                        k2,
                        source[3],
                        destination[3],
                        channel,
                    )?;
                    pixels[index][channel] =
                        ((source[channel] * sf + destination[channel] * df + 511) / 1023)
                            .clamp(0, 1023);
                }
                if slot == alpha_slot {
                    selected_alpha[index] = source[3];
                }
            }
        }
    }
    // No enabled/intersecting slot still produces the configured background.
    if pixels.is_empty() {
        pixels.resize(output.width * output.height, background);
        selected_alpha.resize(pixels.len(), 0);
    }
    let address = u64::from(registers[0x1c8]) << 8;
    let length = output.length();
    let (backing, offset) = region(mappings, address, length)?;
    let mut bytes = vec![0; length];
    backing
        .read(offset, &mut bytes)
        .map_err(|e| e.to_string())?;
    for y in 0..output.height {
        for x0 in (0..output.width).step_by(4) {
            let count = (output.width - x0).min(4);
            let mut packed = [0; 16];
            for i in 0..count {
                let x = x0 + i;
                let index = y * output.width + x;
                let mut pixel = final_matrix.apply(pixels[index]);
                pixel[3] = match alpha_mode {
                    0 => 1023,
                    1 => background[3],
                    2 | 4 => pixel[3],
                    3 | 5 => selected_alpha[index],
                    _ => unreachable!(),
                };
                if matches!(output.format, 35 | 36) {
                    pixel[3] = 1023;
                }
                if matches!(output.format, 32 | 36) {
                    pixel.swap(0, 2);
                }
                for (c, value) in pixel.into_iter().enumerate() {
                    packed[i * 4 + c] = (value.clamp(0, 1023) >> 2) as u8;
                }
            }
            let target = output.offset(x0 * 4, y);
            bytes[target..target + count * 4].copy_from_slice(&packed[..count * 4]);
        }
    }
    let mut writes = CanonicalWriteBatch::new();
    writes
        .stage(&backing, offset, &bytes)
        .map_err(|e| e.to_string())?;
    writes.commit_ordered().map_err(|e| e.to_string())?;
    // VIDEO_PROBE_BEGIN
    if std::env::var_os("NIXE_VIDEO_AUDIT").is_some() {
        log::debug!(
            "VIDEO_PROBE vic: dimensions={}x{} us={}",
            output.width,
            output.height,
            probe_start.elapsed().as_micros()
        );
    }
    // VIDEO_PROBE_END
    log::trace!(
        "VIC output: dimensions={}x{} format={} layout={} block-height={}",
        output.width,
        output.height,
        output.format,
        output.kind,
        output.block_height
    );
    Ok(())
}

#[cfg(test)]
#[path = "vic_tests.rs"]
mod tests;
