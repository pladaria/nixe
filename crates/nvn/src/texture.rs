use super::*;
use bytemuck::{Pod, Zeroable};
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Pod, Zeroable)]
#[repr(C)]
pub struct Texture {
    pub address: u64,
    pub stride: u32,
    pub width: u32,
    pub height: u32,
    pub flags: u32,
    pub format: u16,
    pub levels: u8,
    pub target: u8,
    pub samples: u32,
}
#[derive(Clone, Copy, Debug)]
pub struct Layout {
    pub bytes: u64,
    pub alignment: u32,
    pub block_height: u8,
    pub pitch: u32,
}
impl Texture {
    pub fn layout(&self) -> Result<Layout> {
        if self.target != 1
            || self.samples > 1
            || self.width == 0
            || self.height == 0
            || self.levels != 1
            || self.flags & !(0x01 | 0x08 | 0x10 | 0x40 | 0x200) != 0
        {
            return Err(format!("unsupported texture layout {self:?}").into());
        }
        let bpp = match self.format {
            0x25 => 4, // RGBA8
            0x95 => 2, // BGR565
            f => return Err(format!("unsupported texture format {f:#x}").into()),
        };
        let row = self.width.checked_mul(bpp).ok_or("texture row overflows")?;
        if self.flags & 0x50 != 0 {
            let alignment = if self.flags & 0x40 != 0 { 128 } else { 32 };
            let pitch = if self.stride != 0 {
                self.stride
            } else {
                row.checked_next_multiple_of(alignment)
                    .ok_or("texture pitch overflows")?
            };
            if pitch < row {
                return Err("linear texture stride is smaller than its row".into());
            }
            return Ok(Layout {
                bytes: u64::from(pitch) * u64::from(self.height),
                alignment,
                block_height: 0,
                pitch,
            });
        }
        let pitch = row
            .checked_next_multiple_of(64)
            .ok_or("block-linear texture pitch overflows")?;
        // https://github.com/seedds/nxnative/blob/a366365/dm/gfx/nvn/nvn_tex.c
        // Automatic block height for 2D Maxwell textures; MINIMAL_LAYOUT may
        // reduce it if the surface would otherwise acquire trailing block rows.
        let mut block_height = (u64::from(self.height) * 3 / 16).clamp(1, 16).ilog2() as u8;
        let size = |bh: u8| u64::from(pitch) * u64::from(self.height).next_multiple_of(8 << bh);
        if self.flags & 0x200 != 0 && size(block_height) != size(0) {
            block_height = 0
        }
        Ok(Layout {
            bytes: size(block_height),
            alignment: 512,
            block_height,
            pitch,
        })
    }
}
pub fn set_device(c: &Call) -> Result<u64> {
    c.write(c.args[0], &c.args[1])?;
    Ok(0)
}
pub fn defaults(c: &Call) -> Result<u64> {
    c.write(
        c.args[0] + 8,
        &Texture {
            target: 1,
            levels: 1,
            ..Default::default()
        },
    )?;
    Ok(0)
}
macro_rules! setter {
    ($name:ident,$field:ident,$ty:ty) => {
        pub fn $name(c: &Call) -> Result<u64> {
            c.write(
                c.args[0] + 8 + std::mem::offset_of!(Texture, $field) as u64,
                &(c.args[1] as $ty),
            )?;
            Ok(0)
        }
    };
}
setter!(flags, flags, u32);
setter!(target, target, u8);
setter!(levels, levels, u8);
setter!(format, format, u16);
setter!(samples, samples, u32);
setter!(stride, stride, u32);
pub fn size(c: &Call) -> Result<u64> {
    c.write(
        c.args[0] + 8 + std::mem::offset_of!(Texture, width) as u64,
        &[c.args[1] as u32, c.args[2] as u32],
    )?;
    Ok(0)
}
pub fn storage(c: &Call) -> Result<u64> {
    let p: pool::Pool = c.read(c.args[1])?;
    c.write(
        c.args[0] + 8,
        &p.gpu
            .checked_add(c.args[2])
            .ok_or("texture address overflows")?,
    )?;
    Ok(0)
}
pub fn alignment(c: &Call) -> Result<u64> {
    Ok(u64::from(
        c.read::<Texture>(c.args[0] + 8)?.layout()?.alignment,
    ))
}
pub fn storage_size(c: &Call) -> Result<u64> {
    Ok(c.read::<Texture>(c.args[0] + 8)?.layout()?.bytes)
}
pub fn initialize(c: &Call) -> Result<u64> {
    let t: Texture = c.read(c.args[1] + 8)?;
    t.layout()?;
    c.write(c.args[0], &t)?;
    Ok(1)
}
pub fn register_texture(c: &Call) -> Result<u64> {
    if c.args[3] != 0 {
        return Err("explicit NVN texture views are unsupported".into());
    }
    let t: Texture = c.read(c.args[2])?;
    c.write(pool::slot(c, c.args[0], c.args[1] as u32)?, &t)?;
    Ok(0)
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Pod, Zeroable)]
pub struct Sampler {
    pub min: u32,
    pub mag: u32,
    pub wrap: [u32; 3],
}
pub fn sampler_defaults(c: &Call) -> Result<u64> {
    c.write(
        c.args[0] + 8,
        &Sampler {
            min: 0,
            mag: 0,
            wrap: [1; 3],
        },
    )?;
    Ok(0)
}
pub fn filter(c: &Call) -> Result<u64> {
    c.write(c.args[0] + 8, &[c.args[1] as u32, c.args[2] as u32])?;
    Ok(0)
}
pub fn wrap(c: &Call) -> Result<u64> {
    c.write(
        c.args[0] + 16,
        &[c.args[1] as u32, c.args[2] as u32, c.args[3] as u32],
    )?;
    Ok(0)
}
pub fn sampler_initialize(c: &Call) -> Result<u64> {
    let s: Sampler = c.read(c.args[1] + 8)?;
    c.write(c.args[0] + 8, &s)?;
    Ok(1)
}
pub fn register_sampler(c: &Call) -> Result<u64> {
    let s: Sampler = c.read(c.args[2] + 8)?;
    c.write(pool::slot(c, c.args[0], c.args[1] as u32)?, &s)?;
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_linear_texture_respects_application_pitch() {
        let t = Texture {
            width: 424,
            height: 240,
            levels: 1,
            target: 1,
            flags: 0x10,
            stride: 1024,
            format: 0x95,
            ..Default::default()
        };
        let l = t.layout().unwrap();
        assert_eq!((l.pitch, l.bytes), (1024, 1024 * 240));
        assert!(Texture { stride: 640, ..t }.layout().is_err());
    }

    #[test]
    fn tiled_rgba_framebuffer_has_complete_gob_rows() {
        let t = Texture {
            width: 1920,
            height: 1080,
            levels: 1,
            target: 1,
            format: 0x25,
            ..Default::default()
        };
        let l = t.layout().unwrap();
        assert_eq!((l.block_height, l.pitch, l.bytes), (4, 7680, 0x870000));
    }
}
