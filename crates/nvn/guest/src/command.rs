use super::*;
use bytemuck::{Pod, Zeroable};
#[derive(Clone, Copy, Default, Pod, Zeroable)]
#[repr(C)]
// Header fields also touched by inline NVN commands in application modules.
// An empty inline pushbuffer directs draw calls through their resolved NVN entry.
struct Recorder {
    position: u64,
    inline_limit: u64,
    inline_dirty: u8,
    recording: u8,
    control_flags: u16,
    callback_enabled: u8,
    saved_flags: [u8; 3],
    limit: u64,
    control: u64,
    control_end: u64,
    command: u64,
    control_base: u64,
    begin: u64,
    first: u64,
    last: u64,
    capture_anchor: u64,
    device: u64,
    callback: u64,
    callback_data: u64,
    copy_strides: CopyStrides,
}
const _: () = assert!(
    core::mem::offset_of!(Recorder, begin) == 64
        && core::mem::offset_of!(Recorder, callback) == 104
);
pub fn initialize(c: &Call) -> Result<u64> {
    c.write(
        c.args[0],
        &Recorder {
            callback_enabled: 1,
            device: c.args[1],
            ..Default::default()
        },
    )?;
    Ok(1)
}
impl Recorder {
    fn request(&self, c: &Call, event: u64, bytes: u64) -> Result<u64> {
        if self.callback == 0 || (event == 0 && self.callback_enabled == 0) {
            return Err("NVN recording memory exhausted without refill callback");
        }
        c.refill
            .set(Some([self.callback, event, bytes, self.callback_data]));
        Ok(0)
    }
    fn has_control(&self, entries: u64) -> bool {
        self.control_end.saturating_sub(self.control) >= (entries + 1) * CONTROL_SIZE
    }
    fn entry(&mut self, c: &Call, address: u64, size: u64) -> Result<()> {
        if self.first == 0 {
            self.first = self.control
        }
        c.write(self.control, &Range { address, size })?;
        self.last = self.control;
        self.control += CONTROL_SIZE;
        Ok(())
    }
    fn close_range(&mut self, c: &Call) -> Result<()> {
        if self.position > self.begin {
            self.entry(c, self.begin, self.position - self.begin)?;
            self.begin = self.position;
            self.capture_anchor = self.position;
        }
        Ok(())
    }
}
pub fn finalize(c: &Call) -> Result<u64> {
    c.write(c.args[0], &Recorder::default())?;
    Ok(0)
}
macro_rules! recorder_field {
    ($name:ident,$field:ident,$value:expr) => {
        pub fn $name(c: &Call) -> Result<u64> {
            c.write(
                c.args[0] + core::mem::offset_of!(Recorder, $field) as u64,
                &$value(c),
            )?;
            Ok(0)
        }
    };
}
recorder_field!(memory_callback, callback, |c: &Call| c.args[1]);
recorder_field!(callback_data, callback_data, |c: &Call| c.args[1]);
recorder_field!(callback_enabled, callback_enabled, |c: &Call| u8::from(
    c.args[1] as u8 != 0
));
// The SDK stores two 32-bit pitches on the command buffer and reads them when
// recording each transfer. A setter never consumes command memory.
// https://github.com/seedds/nxnative/blob/a366365/dm/gfx/nvn/hle_nvn.c#L1692-L1705
const COPY_STRIDES: u64 = core::mem::offset_of!(Recorder, copy_strides) as u64;
pub fn set_copy_row_stride(c: &Call) -> Result<u64> {
    c.write(c.args[0] + COPY_STRIDES, &(c.args[1] as u32))?;
    Ok(0)
}
pub fn set_copy_image_stride(c: &Call) -> Result<u64> {
    c.write(c.args[0] + COPY_STRIDES + 4, &(c.args[1] as u32))?;
    Ok(0)
}
pub fn get_copy_row_stride(c: &Call) -> Result<u64> {
    c.read::<u32>(c.args[0] + COPY_STRIDES).map(u64::from)
}
pub fn get_copy_image_stride(c: &Call) -> Result<u64> {
    c.read::<u32>(c.args[0] + COPY_STRIDES + 4).map(u64::from)
}
pub fn command_used(c: &Call) -> Result<u64> {
    let r: Recorder = c.read(c.args[0])?;
    Ok(r.position - r.command)
}
pub fn control_used(c: &Call) -> Result<u64> {
    let r: Recorder = c.read(c.args[0])?;
    Ok(r.control - r.control_base)
}
pub fn add_commands(c: &Call) -> Result<u64> {
    let p: Pool = c.read(c.args[1])?;
    // A caller may round the final grant up to its minimum segment size.
    // Only the backed prefix is writable; the normal refill callback handles exhaustion.
    let remaining = p
        .bytes
        .checked_sub(c.args[2])
        .ok_or("command memory offset exceeds pool size")?;
    let mut r: Recorder = c.read(c.args[0])?;
    if r.recording != 0 && r.position > r.begin {
        if !r.has_control(1) {
            return r.request(c, 1, 48);
        }
        r.close_range(c)?;
    }
    r.command = p.cpu + c.args[2];
    r.position = r.command;
    r.begin = r.position;
    r.capture_anchor = r.position;
    r.limit = r.command + c.args[3].min(remaining);
    c.write(c.args[0], &r)?;
    Ok(0)
}
pub fn add_control(c: &Call) -> Result<u64> {
    let mut r: Recorder = c.read(c.args[0])?;
    if c.args[2] < 32 {
        return Err("control segment too small");
    }
    if r.recording != 0 && r.last != 0 {
        c.write(
            r.control,
            &Range {
                address: c.args[1],
                size: LINK,
            },
        )?;
    }
    r.control = c.args[1];
    r.control_base = r.control;
    r.control_end = r
        .control
        .checked_add(c.args[2])
        .ok_or("control segment overflow")?;
    c.write(c.args[0], &r)?;
    Ok(0)
}
pub fn begin(c: &Call) -> Result<u64> {
    let mut r: Recorder = c.read(c.args[0])?;
    if r.recording != 0 {
        return Err("already recording");
    }
    r.recording = 1;
    r.begin = r.position;
    r.capture_anchor = r.position;
    r.first = 0;
    r.last = 0;
    c.write(c.args[0], &r)?;
    Ok(0)
}
// A call retains the source command ranges. The application can reuse their
// control storage as soon as this function returns. A copy owns the bytes too.
// https://github.com/seedds/nxnative/blob/a366365/dm/gfx/nvn/hle_nvn.c#L1237
pub fn call_commands(c: &Call) -> Result<u64> {
    append_commands::<false>(c)
}
pub fn copy_commands(c: &Call) -> Result<u64> {
    append_commands::<true>(c)
}
fn append_commands<const COPY: bool>(c: &Call) -> Result<u64> {
    let mut r: Recorder = c.read(c.args[0])?;
    if r.recording == 0 {
        return Err("outside recording");
    }
    for i in 0..u64::from(c.args[1] as u32) {
        let mut cursor: u64 = c.read(c.args[2] + 8 * i)?;
        loop {
            let Range { address, size } = c.read(cursor)?;
            if size == LINK {
                cursor = address;
                continue;
            }
            let bytes = size & !(END | CALLED);
            if bytes != 0 {
                if COPY {
                    r.copy_range(c, address, bytes, size & CALLED != 0)?;
                } else {
                    if r.position > r.begin {
                        r.ensure_control(c)?;
                        r.close_range(c)?;
                    }
                    r.ensure_control(c)?;
                    r.entry(c, address, bytes | CALLED)?;
                }
            }
            if size & END != 0 {
                break;
            }
            cursor += CONTROL_SIZE;
        }
    }
    c.write(c.args[0], &r)?;
    Ok(0)
}
impl Recorder {
    #[cold]
    fn refill_now(&mut self, c: &Call, event: u64, bytes: u64) -> Result<()> {
        if self.callback == 0 || (event == 0 && self.callback_enabled == 0) {
            return Err("NVN recording memory exhausted without refill callback");
        }
        c.write(c.args[0], self)?;
        let callback: unsafe extern "C" fn(u64, u64, u64, u64) =
            unsafe { core::mem::transmute(self.callback) };
        unsafe { callback(c.args[0], event, bytes, self.callback_data) };
        *self = c.read(c.args[0])?;
        Ok(())
    }
    fn ensure_control(&mut self, c: &Call) -> Result<()> {
        if !self.has_control(1) {
            self.refill_now(c, 1, 2 * CONTROL_SIZE)?;
            if !self.has_control(1) {
                return Err("insufficient control memory after refill");
            }
        }
        Ok(())
    }
    fn copy_range(
        &mut self,
        c: &Call,
        mut source: u64,
        mut bytes: u64,
        called: bool,
    ) -> Result<()> {
        if !called && bytes <= self.limit - self.position {
            // A complete source range already contains its submit offsets.
            if self.position != self.begin {
                self.ensure_control(c)?;
                self.close_range(c)?;
            }
            self.ensure_control(c)?;
            unsafe {
                core::ptr::copy(
                    source as *const u8,
                    self.position as *mut u8,
                    bytes as usize,
                )
            };
            self.position += bytes;
            return self.close_range(c);
        }
        // A callback may split this range at a different record boundary.
        // Rebuild its links in the destination and keep each record indivisible.
        while bytes != 0 {
            let header: u64 = c.read(source)?;
            let size = (record_words(header) as u64 + 1) * 8;
            if size > bytes {
                if called {
                    break;
                }
                return Err("truncated copied command");
            }
            if size > self.limit - self.position {
                self.refill_now(c, 0, size)?;
                if size > self.limit - self.position {
                    return Err("insufficient command memory after refill");
                }
            }
            unsafe {
                core::ptr::copy(source as *const u8, self.position as *mut u8, size as usize)
            };
            c.write(self.position + 4, &0u32)?;
            if needs_capture(record_tag(header)) {
                link_capture(c, self.position, self.capture_anchor)?;
                self.capture_anchor = self.position;
            }
            self.position += size;
            source += size;
            bytes -= size;
        }
        Ok(())
    }
}
pub fn end(c: &Call) -> Result<u64> {
    let mut r: Recorder = c.read(c.args[0])?;
    if r.recording == 0 {
        return Err("outside recording");
    }
    if (r.position > r.begin || r.last == 0) && !r.has_control(1) {
        return r.request(c, 1, 32);
    }
    r.close_range(c)?;
    if r.last == 0 {
        r.entry(c, 0, 0)?
    }
    let size: u64 = c.read(r.last + 8)?;
    c.write(r.last + 8, &(size | END))?;
    r.recording = 0;
    c.write(c.args[0], &r)?;
    Ok(r.first)
}
#[inline(always)]
fn link_capture(c: &Call, position: u64, anchor: u64) -> Result<()> {
    if anchor != position {
        let words =
            u32::try_from((position - anchor) / 8).map_err(|_| "command range too large")?;
        c.write(anchor + 4, &words)?;
    }
    Ok(())
}
fn reserve(c: &Call, op: u32, words: usize) -> Result<Option<u64>> {
    let object = c.args[0];
    let position: u64 = c.read(object)?;
    let limit: u64 = c.read(object + core::mem::offset_of!(Recorder, limit) as u64)?;
    let bytes = (words as u64 + 1) * 8;
    if c.read::<u8>(object + core::mem::offset_of!(Recorder, recording) as u64)? == 0 {
        return Err("command outside recording");
    }
    if limit.saturating_sub(position) < bytes {
        c.read::<Recorder>(object)?.request(c, 0, bytes)?;
        return Ok(None);
    }
    if needs_capture(op) {
        let field = object + core::mem::offset_of!(Recorder, capture_anchor) as u64;
        link_capture(c, position, c.read(field)?)?;
        c.write(field, &position)?;
    }
    c.write(position, &(u64::from(op) | ((words as u64) << 8)))?;
    c.write(object, &(position + bytes))?;
    Ok(Some(position + 8))
}
fn record(c: &Call, op: u32, words: &[u64]) -> Result<u64> {
    if let Some(dst) = reserve(c, op, words.len())? {
        c.write_bytes(dst, bytemuck::cast_slice(words))?;
    }
    Ok(0)
}
macro_rules! simple {
    ($name:ident,$op:ident,$n:literal) => {
        pub fn $name(c: &Call) -> Result<u64> {
            record(c, op::$op, &c.args[1..=$n])
        }
    };
}
fn bind_pool(c: &Call, op: u32) -> Result<u64> {
    let pool: DescriptorPool = c.read(c.args[1])?;
    record(c, op, &[pool.base, pool.count.into()])
}
pub fn sampler_pool(c: &Call) -> Result<u64> {
    bind_pool(c, op::SAMPLER_POOL)
}
pub fn texture_pool(c: &Call) -> Result<u64> {
    bind_pool(c, op::TEXTURE_POOL)
}
pub fn program(c: &Call) -> Result<u64> {
    let mask = c.args[2] & 63;
    if c.args[1] == 0 {
        return record(c, op::PROGRAM, &[mask]);
    }
    let p: Program = c.read(c.args[1])?;
    let stages = p
        .stages
        .get(..p.count as usize)
        .ok_or("invalid program stage count")?;
    let stages = bytemuck::cast_slice::<Shader, u64>(stages);
    if let Some(dst) = reserve(c, op::PROGRAM, 1 + stages.len())? {
        c.write(dst, &mask)?;
        c.write_bytes(dst + 8, bytemuck::cast_slice(stages))?;
    }
    Ok(0)
}
fn pair(low: u64, high: u64) -> u64 {
    u64::from(low as u32) | (high << 32)
}
macro_rules! rectangle {
    ($name:ident,$tag:ident) => {
        pub fn $name(c: &Call) -> Result<u64> {
            record(
                c,
                op::$tag,
                &[pair(c.args[1], c.args[2]), pair(c.args[3], c.args[4])],
            )
        }
    };
}
rectangle!(scissor, SCISSOR);
rectangle!(viewport, VIEWPORT);
macro_rules! buffer_binding {
    ($name:ident,$tag:ident) => {
        pub fn $name(c: &Call) -> Result<u64> {
            record(
                c,
                op::$tag,
                &[pair(c.args[1], c.args[2]), c.args[3], c.args[4]],
            )
        }
    };
}
simple!(barrier, BARRIER, 1);
simple!(vertex, VERTEX, 3);
buffer_binding!(uniform, UNIFORM);
simple!(dispatch, DISPATCH, 3);
simple!(deferred, DEFERRED, 4);
simple!(report, REPORT, 2);
simple!(reset_counter, RESET_COUNTER, 1);
pub fn tiled_cache_hint(_: &Call) -> Result<u64> {
    // @SIMPLIFICATION Vulkan manages on-chip tile binning and flushing. Resource
    // visibility is expressed separately by NVN barriers and attachment dependencies.
    Ok(0)
}
buffer_binding!(storage, STORAGE);
simple!(condition, CONDITION, 2);
pub fn enable_render(c: &Call) -> Result<u64> {
    record(
        c,
        op::CONDITION,
        &[if c.args[1] as u8 != 0 { 2 } else { 3 }, 0],
    )
}
pub fn scratch_memory(_: &Call) -> Result<u64> {
    // Vulkan allocates shader scratch space as part of pipeline execution.
    Ok(0)
}
pub fn update(c: &Call) -> Result<u64> {
    let [_, buffer, size, off, bytes, data, ..] = c.args;
    if off.checked_add(bytes).is_none_or(|end| end > size) || size > 65536 {
        return Err("uniform update exceeds range");
    }
    let n = bytes.div_ceil(8) as usize;
    if let Some(dst) = reserve(c, op::UPDATE, 2 + n)? {
        c.write(dst, &[buffer + off, bytes])?;
        if bytes % 8 != 0 {
            c.write(dst + 8 * (1 + n as u64), &0u64)?;
        }
        if bytes != 0 {
            unsafe { core::ptr::copy(data as *const u8, (dst + 16) as *mut u8, bytes as usize) };
        }
    }
    Ok(0)
}
fn indexed_draw(c: &Call, base: u64, first: u64, instances: u64) -> Result<u64> {
    record(
        c,
        op::INDEXED,
        &[
            pair(c.args[1], c.args[2]),
            pair(c.args[3], base),
            c.args[4],
            pair(first, instances),
        ],
    )
}
pub fn indexed_instances(c: &Call) -> Result<u64> {
    indexed_draw(c, c.args[5], c.args[6], c.args[7])
}
pub fn elements(c: &Call) -> Result<u64> {
    indexed_draw(c, 0, 0, 1)
}
pub fn indexed(c: &Call) -> Result<u64> {
    indexed_draw(c, c.args[5], 0, 1)
}
fn array_draw(c: &Call, first: u64, instances: u64) -> Result<u64> {
    record(
        c,
        op::DRAW,
        &[
            pair(c.args[1], c.args[2]),
            u64::from(c.args[3] as u32),
            pair(first, instances),
        ],
    )
}
pub fn draw_instances(c: &Call) -> Result<u64> {
    array_draw(c, c.args[4], c.args[5])
}
pub fn draw(c: &Call) -> Result<u64> {
    array_draw(c, 0, 1)
}
macro_rules! bind_state {
    ($fn:ident,$tag:ident,$ty:ty) => {
        pub fn $fn(c: &Call) -> Result<u64> {
            record(c, op::$tag, &[c.read::<$ty>(c.args[1])? as u64])
        }
    };
}
bind_state!(blend, BLEND, u64);
bind_state!(color, COLOR, u32);
bind_state!(mask, MASK, u32);
bind_state!(depth, DEPTH, u64);
bind_state!(multisample, MULTISAMPLE, u32);
simple!(clamp, CLAMP, 1);
simple!(discard, DISCARD, 1);
simple!(stencil_ref, REF, 2);
simple!(stencil_write, WRITE_MASK, 2);
simple!(stencil_compare, COMPARE_MASK, 2);
pub fn point(c: &Call) -> Result<u64> {
    record(c, op::POINT, &[u64::from(c.float[0].to_bits())])
}
pub fn line(c: &Call) -> Result<u64> {
    record(c, op::LINE, &[u64::from(c.float[0].to_bits())])
}
pub fn bounds(c: &Call) -> Result<u64> {
    record(
        c,
        op::BOUNDS,
        &[c.args[1], bytemuck::cast([c.float[0], c.float[1]])],
    )
}
pub fn offset(c: &Call) -> Result<u64> {
    record(
        c,
        op::OFFSET,
        &[
            bytemuck::cast([c.float[1], c.float[2]]),
            u64::from(c.float[0].to_bits()),
        ],
    )
}
bind_state!(poly, POLY, u32);
pub fn blend_color(c: &Call) -> Result<u64> {
    record(c, op::BLEND_COLOR, &c.read::<[u64; 2]>(c.args[1])?)
}
pub fn depth_range(c: &Call) -> Result<u64> {
    record(
        c,
        op::DEPTH_RANGE,
        &[bytemuck::cast([c.float[0], c.float[1]])],
    )
}
pub fn clear_depth(c: &Call) -> Result<u64> {
    record(
        c,
        op::CLEAR_DEPTH,
        &[c.float[0].to_bits() as u64, c.args[1], c.args[2], c.args[3]],
    )
}
pub fn swizzles(c: &Call) -> Result<u64> {
    if c.args[1] != 0 || c.args[2] != 1 {
        return Err("multiple viewport swizzles are unsupported");
    }
    let sw: [u32; 4] = c.read(c.args[3])?;
    let flip = match sw {
        [0, 2, 4, 6] => false,
        [0, 3, 4, 6] => true,
        _ => return Err("unsupported viewport swizzle"),
    };
    record(c, op::SWIZZLES, &[flip.into()])
}
pub fn texture(c: &Call) -> Result<u64> {
    bind_handle(c, c.args[2] as u32)
}
pub fn image(c: &Call) -> Result<u64> {
    let slot = c.args[2] as u32;
    if slot >= IMAGE_HANDLE_SLOTS {
        return Err("invalid storage image slot");
    }
    bind_handle(c, IMAGE_HANDLE_BASE_SLOT + slot)
}
fn bind_handle(c: &Call, row: u32) -> Result<u64> {
    record(
        c,
        op::TEXTURE,
        &[
            u64::from(c.args[1] as u32) | (u64::from(row) << 32),
            c.args[3],
        ],
    )
}
pub fn targets(c: &Call) -> Result<u64> {
    let count = c.args[1] as u32;
    if count > 8 {
        return Err("invalid color target count");
    }
    let mut attachments = ArrayVec::<Texture, 9>::new();
    attachments.push(if c.args[4] == 0 {
        Texture::default()
    } else {
        texels::view(c, c.args[4], c.args[5])?
    });
    for i in 0..u64::from(count) {
        let texture = c.read(c.args[2] + 8 * i)?;
        let view = if c.args[3] != 0 {
            c.read(c.args[3] + 8 * i)?
        } else {
            0
        };
        attachments.push(if texture == 0 {
            Texture::default()
        } else {
            texels::view(c, texture, view)?
        });
    }
    record(c, op::TARGETS, bytemuck::cast_slice(&attachments))
}
pub fn clear(c: &Call) -> Result<u64> {
    let rgba: [u64; 2] = c.read(c.args[2])?;
    record(c, op::CLEAR, &[c.args[1], rgba[0], rgba[1], c.args[3]])
}
fn array(c: &Call, op: u32, stride: usize) -> Result<u64> {
    let n = c.args[1] as u32 as usize;
    if n > 32 {
        return Err("too many vertex inputs");
    }
    let mut buf = [0u64; 33];
    buf[0] = n as u64;
    c.read_bytes(
        c.args[2],
        &mut bytemuck::cast_slice_mut(&mut buf[1..])[..n * stride],
    )?;
    record(c, op, &buf[..1 + (n * stride).div_ceil(8)])
}
pub fn copy_texture(c: &Call) -> Result<u64> {
    let t = texels::view(c, c.args[2], c.args[3])?;
    let region: [u32; 6] = c.read(c.args[4])?;
    let strides: CopyStrides = c.read(c.args[0] + COPY_STRIDES)?;
    let strided = strides != CopyStrides::default();
    if let Some(dst) = reserve(
        c,
        op::COPY_TEXTURE,
        5 + size_of::<Texture>() / 8 + usize::from(strided),
    )? {
        c.write(dst, &[c.args[1], c.args[5]])?;
        c.write(dst + 16, &t)?;
        let tail = dst + 16 + size_of::<Texture>() as u64;
        c.write(tail, &region)?;
        if strided {
            c.write(tail + size_of::<[u32; 6]>() as u64, &strides)?;
        }
    }
    Ok(0)
}
pub fn attributes(c: &Call) -> Result<u64> {
    array(c, op::ATTRIBUTES, 4)
}
pub fn streams(c: &Call) -> Result<u64> {
    array(c, op::STREAMS, 8)
}

// Record pointed-to values now, as the original SDK does when emitting methods.
fn viewport_array(c: &Call, tag: u32, stride: usize) -> Result<u64> {
    let first = c.args[1] as u32;
    let count = c.args[2] as u32;
    if first > 16 || count > 16 - first {
        return Err("invalid viewport array range");
    }
    if count == 0 {
        return Ok(0);
    }
    let words = count as usize * stride;
    if let Some(dst) = reserve(c, tag, 1 + words)? {
        c.write(dst, &u64::from(first))?;
        let src = c.args[3] as *const u8;
        unsafe {
            core::ptr::copy(src, (dst + 8) as *mut u8, words * 8);
        }
    }
    Ok(0)
}
pub fn viewports(c: &Call) -> Result<u64> {
    viewport_array(c, op::VIEWPORTS, 2)
}
pub fn scissors(c: &Call) -> Result<u64> {
    viewport_array(c, op::SCISSORS, 2)
}
pub fn depth_ranges(c: &Call) -> Result<u64> {
    viewport_array(c, op::DEPTH_RANGES, 1)
}
simple!(conservative, CONSERVATIVE, 1);
pub fn clear_texture(c: &Call) -> Result<u64> {
    let texture = texels::view(c, c.args[1], c.args[2])?;
    let region: [u32; 6] = c.read(c.args[3])?;
    let color: [u32; 4] = c.read(c.args[4])?;
    if let Some(dst) = reserve(c, op::CLEAR_IMAGE, size_of::<Texture>() / 8 + 6)? {
        c.write(dst, &texture)?;
        let tail = dst + size_of::<Texture>() as u64;
        c.write(tail, &region)?;
        c.write(tail + 24, &color)?;
        c.write(tail + 40, &c.args[5])?;
    }
    Ok(0)
}
pub fn copy_images(c: &Call) -> Result<u64> {
    let source = texels::view(c, c.args[1], c.args[2])?;
    let target = texels::view(c, c.args[4], c.args[5])?;
    let source_region: [u32; 6] = c.read(c.args[3])?;
    let target_region: [u32; 6] = c.read(c.args[6])?;
    if let Some(dst) = reserve(c, op::COPY_IMAGES, 2 * size_of::<Texture>() / 8 + 7)? {
        c.write(dst, &[source, target])?;
        let tail = dst + 2 * size_of::<Texture>() as u64;
        c.write(tail, &[source_region, target_region])?;
        c.write(tail + 48, &c.args[7])?;
    }
    Ok(0)
}
