use super::*;
use crate::{ir::TextureInstInfo, types::TextureType};

impl Visitor<'_> {
    pub(super) fn texture_sample(&mut self) -> Result<()> {
        let i = self.instruction;
        if i.bits(59, 1) != 0
            || i.bits(60, 1) != 1
            || i.bits(61, 3) != 1
            || i.bits(76, 4) != 0
            || i.bits(81, 3) != 7
            || i.bits(87, 3) != 0
        {
            return Err(unsupported("Native texture sample dimension/operands/mode"));
        }
        let mask = i.bits(72, 4);
        if mask == 0 {
            return Err(unsupported("Native texture sample has no components"));
        }
        // Native instructions select a descriptor buffer independently. Its load
        // stays explicit in IR so resource tracking preserves that exact source.
        let handle = self.emit.get_cbuf(
            Value::U32(i.bits(54, 5)),
            Value::U32(i.bits(40, 14) * 4),
            32,
            false,
        )?;
        let x = self.reg(i.bits(24, 8))?;
        let y = self.reg(i.bits(32, 8))?;
        let x = self.emit.bit_cast(Type::F32, x)?;
        let y = self.emit.bit_cast(Type::F32, y)?;
        let coords = self.emit.composite_construct(&[x, y])?;
        let mut info = TextureInstInfo::default();
        info.set_texture_type(TextureType::Color2D);
        // Bit 90 is NODEP, an instruction scheduling hint; it does not
        // change derivative semantics or the sampled value.
        let value = self.emit.image_sample_implicit_lod(
            handle,
            coords,
            Value::F32(0),
            Value::None,
            Value::None,
            info,
        )?;
        let first = Reg(i.bits(16, 8) as u8);
        let second = Reg(i.bits(64, 8) as u8);
        let mut output = 0;
        for component in 0..4 {
            if mask & (1 << component) == 0 {
                continue;
            }
            let dest = if second != Reg::RZ && output >= 2 {
                second.offset(output - 2)?
            } else if first == Reg::RZ {
                Reg::RZ
            } else {
                first.offset(output)?
            };
            let component = self.emit.composite_extract(value, component)?;
            let component = self.emit.bit_cast(Type::U32, component)?;
            self.emit.set_reg(dest, component)?;
            output += 1;
        }
        Ok(())
    }
}
