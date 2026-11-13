use super::*;

impl Visitor<'_> {
    pub(super) fn select(&mut self) -> Result<()> {
        let i = self.instruction;
        let floating = i.opcode() & 0x1ff == 8;
        if i.bits(64, 8) != 0
            || i.bits(74, 13) != 0
            || i.bits(91, 1) != 0
            || (!floating && i.bits(72, 2) != 0)
        {
            return Err(unsupported("Native select modifiers"));
        }
        let a = if floating {
            self.operand(0)?
        } else {
            self.reg(i.bits(24, 8))?
        };
        let b = if floating {
            self.operand(1)?
        } else {
            self.raw_b()?
        };
        let condition = self
            .emit
            .get_pred(Pred::new(i.bits(87, 3) as u8)?, i.bits(90, 1) != 0)?;
        let result = self.emit.select(condition, a, b)?;
        if floating {
            self.write_float(result)
        } else {
            self.write(result)
        }
    }
    pub(super) fn iadd3(&mut self) -> Result<()> {
        let i = self.instruction;
        if i.bits(73, 2) != 0
            || i.bits(76, 1) != 0
            || i.bits(77, 7) != 0x7f
            || i.bits(84, 7) != 0x7f
            || i.bits(91, 1) != 0
            || (i.opcode() == 0x210 && i.bits(40, 23) != 0)
            || (i.opcode() == 0xa10 && i.bits(62, 1) != 0)
        {
            return Err(unsupported("Native integer addition carry/extended mode"));
        }
        let mut a = self.reg(i.bits(24, 8))?;
        let mut b = self.raw_b()?;
        let mut c = self.reg(i.bits(64, 8))?;
        if i.bits(72, 1) != 0 {
            a = self.emit.i_neg(a)?;
        }
        if i.opcode() != 0x810 && i.bits(63, 1) != 0 {
            b = self.emit.i_neg(b)?;
        }
        if i.bits(75, 1) != 0 {
            c = self.emit.i_neg(c)?;
        }
        let sum = self.emit.i_add(a, b)?;
        let sum = self.emit.i_add(sum, c)?;
        self.write(sum)
    }
    pub(super) fn i2fp(&mut self) -> Result<()> {
        let i = self.instruction;
        if i.bits(64, 28) & !0xc400 != 0x00201000 {
            return Err(unsupported("Native integer-to-float format"));
        }
        let value = self.raw_b()?;
        let control = FpControl {
            rounding: match i.bits(78, 2) {
                0 => FpRounding::RN,
                1 => FpRounding::RM,
                2 => FpRounding::RP,
                _ => FpRounding::RZ,
            },
            ..Default::default()
        };
        let value = self
            .emit
            .convert_i_to_f(32, 32, i.bits(74, 1) != 0, value, control)?;
        self.write_float(value)
    }
    pub(super) fn mufu(&mut self) -> Result<()> {
        let i = self.instruction;
        if i.bits(73, 1) != 0 || i.bits(80, 12) != 0 {
            return Err(unsupported("Native special-function precision/mode"));
        }
        let mut value = self.operand(1)?;
        let operation = i.bits(74, 6);
        if operation <= 1 {
            // Native trigonometric inputs are turns; IR functions take radians.
            value = self.emit.fp_mul(
                value,
                Value::F32(std::f32::consts::TAU.to_bits()),
                FpControl::default(),
            )?;
        }
        let value = match operation {
            0 => self.emit.fp_cos(value)?,
            1 => self.emit.fp_sin(value)?,
            2 => self.emit.fp_exp2(value)?,
            3 => self.emit.fp_log2(value)?,
            4 => self.emit.fp_recip(value)?,
            5 => self.emit.fp_recip_sqrt(value)?,
            8 => self.emit.fp_sqrt(value)?,
            _ => return Err(unsupported("Native special-function operation")),
        };
        self.write_float(value)
    }
    pub(super) fn interpolate(&mut self) -> Result<()> {
        let i = self.instruction;
        if i.bits(32, 8) != 255 || i.bits(76, 2) != 0 || i.bits(79, 1) != 0 || i.bits(81, 3) != 7 {
            return Err(unsupported("Native interpolation mode/sample location"));
        }
        let attribute = Attribute::from_index(i.bits(64, 8) as u16)
            .ok_or_else(|| unsupported("Native interpolated attribute"))?;
        let mut value = self.emit.get_attribute(attribute)?;
        if attribute.is_generic() {
            use crate::types::PixelImap;
            let map = self
                .metadata
                .sph
                .generic_input_map(attribute.generic_index()? as usize);
            let mode = map
                .into_iter()
                .find(|m| *m != PixelImap::Unused)
                .unwrap_or(PixelImap::Unused);
            if matches!(mode, PixelImap::Unused | PixelImap::Perspective) {
                // The guest performs perspective division explicitly after IPA.
                let w = self.emit.get_attribute(Attribute::PositionW)?;
                value = self.emit.fp_mul(value, w, FpControl::default())?;
            }
        } else if attribute.is_legacy() {
            return Err(unsupported("Native legacy attribute interpolation"));
        }
        self.write_float(value)
    }
}
