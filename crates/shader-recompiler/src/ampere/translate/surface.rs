//! Native formatted surface access through descriptors in any constant buffer.
use super::*;
use crate::{ir::TextureInstInfo, types::TextureType};

impl Visitor<'_> {
    pub(super) fn surface_memory(&mut self) -> Result<()> {
        let i = self.instruction;
        let store = matches!(i.opcode(), 0xb9b | 0x99c);
        let bindless = matches!(i.opcode(), 0x998 | 0x99c);
        let mode = i.bits(64, 32) & if bindless { !0xfff } else { !0xf00 };
        let strong = store && !bindless && mode == 0x10a000;
        let mask = i.bits(72, 4);
        if mask == 0
            || (store && (mask != 15 || i.bits(16, 8) != 0))
            || mode
                != if store {
                    if strong { 0x10a000 } else { 0x100000 }
                } else {
                    0x1e0000
                }
            || i.bits(59, 2) != 0
            || i.bits(96, 32) & 0xc00001ff != 0
        {
            return Err(unsupported("Native formatted surface access mode"));
        }
        let (ty, count) = match i.bits(61, 3) {
            0 => (TextureType::Color1D, 1),
            1 => (TextureType::Buffer, 1),
            3 => (TextureType::Color2D, 2),
            4 => (TextureType::ColorArray2D, 3),
            5 => (TextureType::Color3D, 3),
            _ => return Err(unsupported("Native formatted surface dimension")),
        };
        let base = Reg(i.bits(24, 8) as u8);
        let mut coordinates = [Value::None; 3];
        for (index, coordinate) in coordinates.iter_mut().enumerate().take(count) {
            *coordinate = self.emit.get_reg(base.offset(index as i32)?)?;
            if ty == TextureType::ColorArray2D && index == 2 {
                *coordinate = self.emit.bit_field_extract(
                    *coordinate,
                    Value::U32(0),
                    Value::U32(16),
                    false,
                )?;
            }
        }
        let coordinates = if count == 1 {
            coordinates[0]
        } else {
            self.emit.composite_construct(&coordinates[..count])?
        };
        let handle = if bindless {
            self.reg(i.bits(64, 8))?
        } else {
            self.emit.get_cbuf(
                Value::U32(i.bits(54, 5)),
                Value::U32(i.bits(40, 14) * 4),
                32,
                false,
            )?
        };
        let mut info = TextureInstInfo::default();
        info.set_texture_type(ty);
        let register = Reg(i.bits(if store { 32 } else { 16 }, 8) as u8);
        if store {
            let mut values = [Value::None; 4];
            for (index, value) in values.iter_mut().enumerate() {
                *value = self.emit.get_reg(register.offset(index as i32)?)?;
            }
            let color = self.emit.composite_construct(&values)?;
            if strong {
                self.emit.device_memory_barrier()?;
            }
            self.emit.image_write(handle, coordinates, color, info)?;
            if strong {
                self.emit.device_memory_barrier()?;
            }
        } else {
            let count = mask.count_ones();
            if !register.is_aligned(if count == 3 { 4 } else { count as u8 }) {
                return Err(unsupported("Native surface destination alignment"));
            }
            let result = self.emit.image_read(handle, coordinates, info)?;
            let mut offset = 0;
            for component in 0..4 {
                if mask & (1 << component) != 0 {
                    let value = self.emit.composite_extract(result, component)?;
                    self.emit.set_reg(register.offset(offset)?, value)?;
                    offset += 1;
                }
            }
        }
        Ok(())
    }
}
