use super::{Result, Value, Visitor, unsupported};
use crate::{
    ir::TextureInstInfo,
    types::{ImageFormat, TextureType},
};

impl Visitor<'_> {
    fn surface_coords(&mut self) -> Result<(Value, TextureInstInfo)> {
        let kind = self.field::<33, 3>();
        let (ty, count, array) = match kind {
            0 => (TextureType::Color1D, 1, false),
            1 => (TextureType::Buffer, 1, false),
            2 => (TextureType::ColorArray1D, 2, true),
            3 => (TextureType::Color2D, 2, false),
            4 => (TextureType::ColorArray2D, 3, true),
            5 => (TextureType::Color3D, 3, false),
            _ => return Err(unsupported("Surface dimension")),
        };
        let mut coords = [Value::None; 3];
        for (i, value) in coords.iter_mut().enumerate().take(count) {
            *value = self
                .emit
                .get_reg(self.instruction.reg8().offset(i as i32)?)?;
            if array && i == count - 1 {
                *value =
                    self.emit
                        .bit_field_extract(*value, Value::U32(0), Value::U32(16), false)?;
            }
        }
        let coords = if count == 1 {
            coords[0]
        } else {
            self.emit.composite_construct(&coords[..count])?
        };
        let mut info = TextureInstInfo::default();
        info.set_texture_type(ty);
        Ok((coords, info))
    }
    fn surface_handle(&mut self, bindless: bool) -> Result<Value> {
        if bindless {
            self.emit.get_reg(self.instruction.reg39())
        } else {
            Ok(Value::U32(self.field::<36, 13>() * 4))
        }
    }
    pub(super) fn surface_memory(&mut self, store: bool) -> Result<()> {
        if self.field::<49, 2>() != 0 {
            return Err(unsupported("Surface clamp mode"));
        }
        if self.field::<24, 2>() > 1 {
            return Err(unsupported("Surface cache mode"));
        }
        let typed = self.bit::<52>();
        if typed && self.bit::<23>() {
            return Err(unsupported("Surface byte-address mode"));
        }
        let size = self.field::<20, 3>();
        let format = if typed {
            ImageFormat::try_from(size + 1).map_err(|_| unsupported("Surface access size"))?
        } else {
            ImageFormat::Typeless
        };
        let (coords, mut info) = self.surface_coords()?;
        let handle = self.surface_handle(!self.bit::<51>())?;
        info.set_image_format(format);
        let reg = self.instruction.reg0();
        let count = match size {
            0..=4 => 1,
            5 => 2,
            6 => 4,
            _ => 0,
        };
        let mask = self.field::<20, 4>();
        if store {
            if !typed && mask != 15 {
                return Err(unsupported("Surface store component mask"));
            }
            let mut values = [Value::U32(0); 4];
            for (i, value) in values
                .iter_mut()
                .enumerate()
                .take(if typed { count } else { 4 })
            {
                *value = self.emit.get_reg(reg.offset(i as i32)?)?;
            }
            let color = self.emit.composite_construct(&values)?;
            self.emit.image_write(handle, coords, color, info)?;
        } else {
            let value = self.emit.image_read(handle, coords, info)?;
            let mask = if typed {
                (1 << count) - 1
            } else {
                if mask == 0 {
                    return Err(unsupported("Empty surface swizzle"));
                }
                let count = mask.count_ones();
                if !reg.is_aligned(if count == 3 { 4 } else { count as u8 }) {
                    return Err(unsupported("Unaligned surface destination"));
                }
                mask
            };
            let mut dest = reg;
            for component in 0..4 {
                if mask & (1 << component) == 0 {
                    continue;
                }
                let part = self.emit.composite_extract(value, component)?;
                self.emit.set_reg(dest, part)?;
                dest = dest.offset(1)?;
            }
        }
        Ok(())
    }
    pub(super) fn surface_atomic(&mut self, reduction: bool) -> Result<()> {
        if self.field::<49, 2>() != 0 {
            return Err(unsupported("Surface atomic clamp mode"));
        }
        let size = if reduction {
            self.field::<20, 3>()
        } else {
            self.field::<51, 3>()
        };
        if !matches!(size, 0 | 1 | 6) {
            return Err(unsupported("Surface atomic size other than 32-bit integer"));
        }
        let (coords, mut info) = self.surface_coords()?;
        info.set_image_format(ImageFormat::R32_UINT);
        let handle = self.surface_handle(if reduction {
            !self.bit::<51>()
        } else {
            self.bit::<54>()
        })?;
        let source = if reduction {
            self.instruction.reg0()
        } else {
            self.instruction.reg20()
        };
        let value = self.emit.get_reg(source)?;
        let op = if reduction {
            self.field::<24, 3>()
        } else {
            self.field::<29, 4>()
        };
        let result = match op {
            0 => self.emit.image_atomic_i_add(handle, coords, value, info)?,
            1 => self
                .emit
                .image_atomic_i_min(handle, coords, value, size == 1, info)?,
            2 => self
                .emit
                .image_atomic_i_max(handle, coords, value, size == 1, info)?,
            3 => self.emit.image_atomic_inc(handle, coords, value, info)?,
            4 => self.emit.image_atomic_dec(handle, coords, value, info)?,
            5 => self.emit.image_atomic_and(handle, coords, value, info)?,
            6 => self.emit.image_atomic_or(handle, coords, value, info)?,
            7 => self.emit.image_atomic_xor(handle, coords, value, info)?,
            8 => self
                .emit
                .image_atomic_exchange(handle, coords, value, info)?,
            _ => return Err(unsupported("Surface atomic operation")),
        };
        if !reduction {
            self.write(result)?;
        }
        Ok(())
    }
}
