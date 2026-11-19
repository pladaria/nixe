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
    reserved: u64,
    device: u64,
    callback: u64,
    callback_data: u64,
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
            self.begin = self.position
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
    c.args[1] as u32 != 0
));
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
            let bytes = size & !END;
            if bytes != 0 {
                if COPY {
                    r.copy_range(c, address, bytes)?;
                } else {
                    if r.position > r.begin {
                        r.ensure_control(c)?;
                        r.close_range(c)?;
                    }
                    r.ensure_control(c)?;
                    r.entry(c, address, bytes)?;
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
    fn copy_range(&mut self, c: &Call, mut source: u64, mut bytes: u64) -> Result<()> {
        while bytes != 0 {
            let available = self.limit - self.position;
            let mut n = bytes;
            if n > available {
                n = 0;
                while n < bytes {
                    let header: u64 = c.read(source + n)?;
                    let record = ((header >> 32) + 1) * 8;
                    if record > bytes - n {
                        return Err("truncated copied command");
                    }
                    if record > available - n {
                        if n == 0 {
                            self.refill_now(c, 0, record)?;
                            if self.limit - self.position < record {
                                return Err("insufficient command memory after refill");
                            }
                        }
                        break;
                    }
                    n += record;
                }
            }
            if n == 0 {
                continue;
            }
            unsafe { core::ptr::copy(source as *const u8, self.position as *mut u8, n as usize) };
            self.position += n;
            source += n;
            bytes -= n;
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
    c.write(position, &(u64::from(op) | ((words as u64) << 32)))?;
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
simple!(scissor, SCISSOR, 4);
simple!(viewport, VIEWPORT, 4);
simple!(barrier, BARRIER, 1);
simple!(vertex, VERTEX, 3);
simple!(uniform, UNIFORM, 4);
simple!(dispatch, DISPATCH, 3);
simple!(deferred, DEFERRED, 4);
simple!(report, REPORT, 2);
simple!(storage, STORAGE, 4);
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
simple!(indexed_instances, INDEXED, 7);
simple!(draw_instances, DRAW, 5);
pub fn elements(c: &Call) -> Result<u64> {
    record(
        c,
        op::INDEXED,
        &[c.args[1], c.args[2], c.args[3], c.args[4], 0, 0, 1],
    )
}
pub fn indexed(c: &Call) -> Result<u64> {
    record(
        c,
        op::INDEXED,
        &[c.args[1], c.args[2], c.args[3], c.args[4], c.args[5], 0, 1],
    )
}
pub fn draw(c: &Call) -> Result<u64> {
    record(c, op::DRAW, &[c.args[1], c.args[2], c.args[3], 0, 1])
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
    record(
        c,
        op::TEXTURE,
        &[(c.args[1] as u32 as u64) | (c.args[2] << 32), c.args[3]],
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
    if let Some(dst) = reserve(c, op::COPY_TEXTURE, 5 + size_of::<Texture>() / 8)? {
        c.write(dst, &[c.args[1], c.args[5]])?;
        c.write(dst + 16, &t)?;
        c.write(dst + 16 + size_of::<Texture>() as u64, &region)?;
    }
    Ok(0)
}
pub fn attributes(c: &Call) -> Result<u64> {
    array(c, op::ATTRIBUTES, 4)
}
pub fn streams(c: &Call) -> Result<u64> {
    array(c, op::STREAMS, 8)
}
