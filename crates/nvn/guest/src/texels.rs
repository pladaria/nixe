//! CPU transfers address the same pitched or block-linear allocation as sampling
//! and rendering. Only bytes inside the specified rectangle are modified.
use super::*;

pub fn defaults(c: &Call) -> Result<u64> {
    c.write(
        c.args[0],
        &View {
            target: 255,
            mode: 255,
            swizzle: [255; 4],
            ..Default::default()
        },
    )?;
    Ok(0)
}
macro_rules! set {
    ($name:ident,$field:ident,$val:expr) => {
        pub fn $name(c: &Call) -> Result<u64> {
            c.write(
                c.args[0] + core::mem::offset_of!(View, $field) as u64,
                &$val(c),
            )?;
            Ok(0)
        }
    };
}
set!(target, target, |c: &Call| c.args[1] as u8);
set!(format, format, |c: &Call| c.args[1] as u16);
set!(mode, mode, |c: &Call| c.args[1] as u8);
// Each range has its own presence bit, including an explicitly selected zero range.
pub fn levels(c: &Call) -> Result<u64> {
    let mut v: View = c.read(c.args[0])?;
    v.first = c.args[1] as u8;
    v.levels = c.args[2] as u8;
    v.range_flags |= 1;
    c.write(c.args[0], &v)?;
    Ok(0)
}
pub fn layers(c: &Call) -> Result<u64> {
    let mut v: View = c.read(c.args[0])?;
    v.layer = c.args[1] as u16;
    v.layers = c.args[2] as u16;
    v.range_flags |= 2;
    c.write(c.args[0], &v)?;
    Ok(0)
}
pub fn get_levels(c: &Call) -> Result<u64> {
    let v: View = c.read(c.args[0])?;
    c.write(c.args[1], &u32::from(v.first))?;
    c.write(c.args[2], &u32::from(v.levels))?;
    Ok(u64::from(v.range_flags & 1 != 0))
}
pub fn get_layers(c: &Call) -> Result<u64> {
    let v: View = c.read(c.args[0])?;
    c.write(c.args[1], &u32::from(v.layer))?;
    c.write(c.args[2], &u32::from(v.layers))?;
    Ok(u64::from(v.range_flags & 2 != 0))
}
pub fn get_target(c: &Call) -> Result<u64> {
    let v: View = c.read(c.args[0])?;
    let set = v.target != 255;
    c.write(c.args[1], &u32::from(if set { v.target } else { 0 }))?;
    Ok(u64::from(set))
}
set!(swizzle, swizzle, |c: &Call| [
    c.args[1] as u8,
    c.args[2] as u8,
    c.args[3] as u8,
    c.args[4] as u8
]);
pub fn cache(_: &Call) -> Result<u64> {
    // @SIMPLIFICATION CPU reads and writes go through the shared memory coherency owner, including
    // pending GPU downloads. There is no separate data cache for mapped texels.
    Ok(0)
}
pub fn read(c: &Call) -> Result<u64> {
    transfer(c, false)
}
pub fn write(c: &Call) -> Result<u64> {
    transfer(c, true)
}

fn transfer(c: &Call, upload: bool) -> Result<u64> {
    let t = view(c, c.args[0], c.args[1])?;
    let base: u64 = c.read(c.args[0] + size_of::<Texture>() as u64)?;
    let [x, y, z, w, h, d]: [u32; 6] = c.read(c.args[2])?;
    let l = t.layout()?;
    let m = l.mips[t.first_level as usize];
    let [width, height] = t.view_extent();
    let depth = t.layer_count();
    if x > width || w > width - x || y > height || h > height - y || z > depth || d > depth - z {
        return Err("texture region outside selected subresource");
    }
    if w == 0 || h == 0 || d == 0 {
        return Ok(0);
    }
    let (bpp, bw, bh) = t.block()?;
    if !x.is_multiple_of(bw) || !y.is_multiple_of(bh) {
        return Err("unaligned compressed region");
    }
    let (x, y, w, h) = (
        (x / bw * bpp) as usize,
        (y / bh) as usize,
        (w.div_ceil(bw) * bpp) as usize,
        h.div_ceil(bh) as usize,
    );
    for dz in 0..d {
        let plane = u32::from(t.first_layer) + z + dz;
        let (layer, slice) = if t.target == 2 {
            (0, plane as usize)
        } else {
            (u64::from(plane) * l.layer_stride, 0)
        };
        let base = base + m.offset;
        let packed = c.args[3] + u64::from(dz) * (w * h) as u64;
        let layer_rows = if t.flags & 0x50 != 0 {
            height.div_ceil(bh) as usize
        } else {
            (height.div_ceil(bh) as usize).next_multiple_of(8 << m.block_height)
        };
        transfer_slice(
            upload,
            base + layer,
            packed,
            m,
            t.flags & 0x50 != 0,
            [x, y, w, h],
            slice,
            layer_rows,
        )?;
    }
    Ok(0)
}
#[allow(clippy::too_many_arguments)]
fn transfer_slice(
    upload: bool,
    base: u64,
    src: u64,
    layout: Mip,
    linear: bool,
    [x, y, w, h]: [usize; 4],
    slice: usize,
    layer_rows: usize,
) -> Result<()> {
    let pitch = layout.pitch as usize;

    let band = if linear {
        1
    } else {
        8usize << layout.block_height
    };
    let bd = 1usize << layout.block_depth;
    for dy in 0..h {
        let row = y + dy;
        let packed = src + (dy * w) as u64;
        let slab = ((slice / bd * layer_rows + row / band * band) * pitch * bd
            + slice % bd * band * 64) as u64;
        let mut done = 0;
        while done < w {
            let col = x + done;
            let (offset, count) = if linear {
                (((slice * layer_rows + row) * pitch + col) as u64, w - done)
            } else {
                let y = row % band;
                let gob = (col / 64) * 64 * band * bd
                    + (y / 8) * 512
                    + ((col & 32) << 3)
                    + ((y & 6) << 5)
                    + ((col & 16) << 1)
                    + ((y & 1) << 4)
                    + (col & 15);
                (slab + gob as u64, (16 - col % 16).min(w - done))
            };
            let a = packed + done as u64;
            let b = base + offset;
            let (from, to) = if upload { (a, b) } else { (b, a) };
            unsafe { core::ptr::copy(from as *const u8, to as *mut u8, count) };
            done += count;
        }
    }
    Ok(())
}

pub(crate) fn view(c: &Call, texture: u64, view: u64) -> Result<Texture> {
    crate::view::apply(
        c.read(texture)?,
        (view != 0).then(|| c.read(view)).transpose()?,
    )
}
