//! CPU transfers address the same pitched or block-linear allocation as sampling
//! and rendering. Only bytes inside the specified rectangle are modified.
use super::*;

pub fn defaults(c: &Call) -> Result<u64> {
    c.write(c.args[0], &[-1i32, -1, 0, 0])?;
    Ok(0)
}
pub fn format(c: &Call) -> Result<u64> {
    c.write(c.args[0], &(c.args[1] as i32))?;
    Ok(0)
}
pub fn target(c: &Call) -> Result<u64> {
    c.write(c.args[0] + 4, &(c.args[1] as i32))?;
    Ok(0)
}
pub fn levels(c: &Call) -> Result<u64> {
    c.write(c.args[0] + 8, &[c.args[1] as i32, c.args[2] as i32])?;
    Ok(0)
}
pub(crate) fn view(c: &Call, texture: u64, view: u64) -> Result<Texture> {
    let mut t: Texture = c.read(texture)?;
    if view != 0 {
        let [fmt, target, first, count]: [i32; 4] = c.read(view)?;
        if first != 0
            || !(0..=i32::from(t.levels)).contains(&count)
            || (target != -1 && target != i32::from(t.target))
        {
            return Err(
                format!("unsupported texture view target={target} levels={first}+{count}").into(),
            );
        }
        if fmt != -1 {
            let orig = t.texel_bytes()?;
            t.format = fmt as u16;
            if t.texel_bytes()? != orig {
                return Err("incompatible texture view format".into());
            }
        }
    }
    Ok(t)
}
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
    let base: u64 = c.read(c.args[0] + 32)?;
    let [x, y, z, w, h, d]: [u32; 6] = c.read(c.args[2])?;
    if z != 0 || d != 1 || x > t.width || w > t.width - x || y > t.height || h > t.height - y {
        return Err(format!(
            "texture region {x},{y},{z}+{w},{h},{d} outside {}x{}",
            t.width, t.height
        )
        .into());
    }
    if w == 0 || h == 0 {
        return Ok(0);
    }
    let layout = t.layout()?;
    let bpp = t.texel_bytes()? as usize;
    let (x, y, w, h) = (x as usize * bpp, y as usize, w as usize * bpp, h as usize);
    let pitch = layout.pitch as usize;
    let src = c.args[3];
    let linear = t.flags & 0x50 != 0;
    let band = if linear {
        1
    } else {
        8usize << layout.block_height
    };
    let mut rows = vec![0; w * h.min(band)];
    let mut tiled = Vec::new();
    let mut done = 0;
    while done < h {
        let y0 = y + done;
        let n = (band - y0 % band).min(h - done);
        let rows = &mut rows[..w * n];
        let packed = src + (done * w) as u64;
        if upload {
            c.read_bytes(packed, rows)?;
        }
        let io = |addr: u64, bytes: &mut [u8]| {
            if upload {
                c.write_bytes(addr, bytes)
            } else {
                c.read_bytes(addr, bytes)
            }
        };
        if linear {
            io(base + (y0 * pitch + x) as u64, rows)?;
        } else {
            let slab = base + (y0 / band * band * pitch) as u64;
            let gob = |col: usize, row: usize| {
                (col / 64) * 64 * band
                    + (row / 8) * 512
                    + ((col & 32) << 3)
                    + ((row & 6) << 5)
                    + ((col & 16) << 1)
                    + ((row & 1) << 4)
                    + (col & 15)
            };
            // A complete band of GOBs can cross the address-space boundary in one call.
            let full = n == band && x.is_multiple_of(64) && w.is_multiple_of(64);
            let start = gob(x, 0);
            if full {
                tiled.resize(rows.len(), 0);
                if !upload {
                    io(slab + start as u64, &mut tiled)?;
                }
            }
            for (dy, row) in rows.chunks_exact_mut(w).enumerate() {
                let mut i = 0;
                while i < w {
                    let len = (16 - (x + i) % 16).min(w - i);
                    let at = gob(x + i, y0 % band + dy);
                    let span = &mut row[i..i + len];
                    if full {
                        let other = &mut tiled[at - start..at - start + len];
                        if upload {
                            other.copy_from_slice(span)
                        } else {
                            span.copy_from_slice(other)
                        }
                    } else {
                        io(slab + at as u64, span)?;
                    }
                    i += len;
                }
            }
            if full && upload {
                io(slab + start as u64, &mut tiled)?;
            }
        }
        if !upload {
            c.write_bytes(packed, rows)?;
        }
        done += n;
    }
    Ok(0)
}
