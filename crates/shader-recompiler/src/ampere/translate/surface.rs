//! Native formatted surface access through descriptors in any constant buffer.
//! Encoding: https://github.com/zortos293/Lichii/blob/646b5d17135a2b2701ccc7a959afd0bbbf2f6160/src/shader_recompiler/frontend/ampere/lower_to_maxwell.h#L1588-L1605
use super::*;
use crate::{ir::TextureInstInfo, types::TextureType};

impl Visitor<'_> {
    fn surface_coordinates(&mut self) -> Result<(Value, TextureInstInfo)> {
        let i = self.instruction;
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
        let mut info = TextureInstInfo::default();
        info.set_texture_type(ty);
        Ok((coordinates, info))
    }
    pub(super) fn surface_atomic(&mut self) -> Result<()> {
        let i = self.instruction;
        let operation = i.bits(87, 4);
        if i.bits(64, 32) & !0x07800000 != 0x001ee000
            || operation > 8
            || i.bits(59, 2) != 0
            || i.bits(54, 5) >= crate::artifact::MAX_CBUFS as u32
            || i.bits(40, 14) > 0x1fff
            || i.bits(96, 32) & 0xc00001ff != 0
        {
            return Err(unsupported("Native formatted surface atomic mode"));
        }
        let result = self.surface_atomic_operation(operation)?;
        self.write(result)
    }

    pub(super) fn surface_reduction(&mut self) -> Result<()> {
        // SURED performs the surface atomic operation without returning its old value.
        // https://docs.nvidia.com/cuda/parallel-thread-execution/#surface-instructions-sured
        // Native encoding: https://github.com/zortos293/Lichii/blob/646b5d17135a2b2701ccc7a959afd0bbbf2f6160/src/shader_recompiler/frontend/ampere/lower_to_maxwell.h#L1607-L1626
        let i = self.instruction;
        if i.bits(64, 32) & !0x03800000 != 0x0010e000
            || !matches!(i.bits(61, 3), 1 | 3 | 5)
            || i.bits(59, 2) != 0
            || i.bits(54, 5) >= crate::artifact::MAX_CBUFS as u32
            || i.bits(40, 14) > 0x1fff
        {
            return Err(unsupported("Native surface reduction mode"));
        }
        self.surface_atomic_operation(i.bits(87, 3))?;
        Ok(())
    }

    fn surface_atomic_operation(&mut self, operation: u32) -> Result<Value> {
        let i = self.instruction;
        let (coordinates, mut info) = self.surface_coordinates()?;
        info.set_image_format(crate::types::ImageFormat::R32_UINT);
        let handle = self.emit.get_cbuf(
            Value::U32(i.bits(54, 5)),
            Value::U32(i.bits(40, 14) * 4),
            32,
            false,
        )?;
        let value = self.reg(i.bits(32, 8))?;
        match operation {
            0 => self
                .emit
                .image_atomic_i_add(handle, coordinates, value, info),
            1 => self
                .emit
                .image_atomic_u_min(handle, coordinates, value, info),
            2 => self
                .emit
                .image_atomic_u_max(handle, coordinates, value, info),
            3 => self.emit.image_atomic_inc(handle, coordinates, value, info),
            4 => self.emit.image_atomic_dec(handle, coordinates, value, info),
            5 => self.emit.image_atomic_and(handle, coordinates, value, info),
            6 => self.emit.image_atomic_or(handle, coordinates, value, info),
            7 => self.emit.image_atomic_xor(handle, coordinates, value, info),
            _ => self
                .emit
                .image_atomic_exchange(handle, coordinates, value, info),
        }
    }

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
        let (coordinates, info) = self.surface_coordinates()?;
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
