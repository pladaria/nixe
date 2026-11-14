use super::*;

impl Visitor<'_> {
    pub(super) fn hmnmx(&mut self) -> Result<()> {
        let i = self.instruction;
        let immediate = i.opcode() == 0x840;
        let swizzle_a = i.bits(74, 2);
        let swizzle_b = if immediate { 0 } else { i.bits(60, 2) };
        if i.bits(76, 11) != 0 || i.bits(91, 1) != 0 || swizzle_a == 1 || swizzle_b == 1 {
            return Err(unsupported("Native half min/max mode"));
        }
        let a = self.reg(i.bits(24, 8))?;
        let b = if immediate {
            Value::U32(i.bits(32, 32))
        } else {
            self.reg(i.bits(32, 8))?
        };
        let a = crate::frontend::half::extract(&mut self.emit, a, swizzle_a)?;
        let b = crate::frontend::half::extract(&mut self.emit, b, swizzle_b)?;
        let predicate = self
            .emit
            .get_pred(Pred::new(i.bits(87, 3) as u8)?, i.bits(90, 1) != 0)?;
        let control = FpControl::default();
        let mut result = [Value::None; 2];
        for index in 0..2 {
            let a = self.emit.fp_convert(32, a[index], control)?;
            let b = self.emit.fp_convert(32, b[index], control)?;
            let a = self
                .emit
                .fp_abs_neg(a, i.bits(73, 1) != 0, i.bits(72, 1) != 0)?;
            let b = self.emit.fp_abs_neg(
                b,
                !immediate && i.bits(62, 1) != 0,
                !immediate && i.bits(63, 1) != 0,
            )?;
            let min = self.emit.fp_min(a, b, control)?;
            let max = self.emit.fp_max(a, b, control)?;
            result[index] = self.emit.select(predicate, min, max)?;
        }
        let pair = self.emit.composite_construct(&result)?;
        let value = self.emit.pack_half2x16(pair)?;
        self.write(value)
    }

    pub(super) fn hsetp(&mut self) -> Result<()> {
        let i = self.instruction;
        let immediate = i.opcode() == 0x434;
        if i.bits(64, 5) != 0 || i.bits(91, 1) != 0 || (i.opcode() == 0x234 && i.bits(40, 20) != 0)
        {
            return Err(unsupported("Native half comparison mode"));
        }
        let a = self.reg(i.bits(24, 8))?;
        let b = match i.opcode() {
            0x234 => self.reg(i.bits(32, 8))?,
            0x434 => Value::U32(i.bits(32, 32)),
            _ => self.constant()?,
        };
        let mut operands = [
            crate::frontend::half::extract(&mut self.emit, a, i.bits(74, 2))?,
            crate::frontend::half::extract(
                &mut self.emit,
                b,
                if immediate { 0 } else { i.bits(60, 2) },
            )?,
        ];
        crate::frontend::half::promote(&mut self.emit, &mut operands)?;
        let input = self
            .emit
            .get_pred(Pred::new(i.bits(87, 3) as u8)?, i.bits(90, 1) != 0)?;
        let control = FpControl {
            fmz_mode: if i.bits(80, 1) != 0 {
                FmzMode::FTZ
            } else {
                FmzMode::None
            },
            ..Default::default()
        };
        let mut results = [Value::None; 2];
        for (index, result) in results.iter_mut().enumerate() {
            let a =
                self.emit
                    .fp_abs_neg(operands[0][index], i.bits(73, 1) != 0, i.bits(72, 1) != 0)?;
            let b = self.emit.fp_abs_neg(
                operands[1][index],
                !immediate && i.bits(62, 1) != 0,
                !immediate && i.bits(63, 1) != 0,
            )?;
            let comparison =
                crate::frontend::float_compare(&mut self.emit, a, b, i.bits(76, 4), control)?;
            *result = self.combine(comparison, input, i.bits(69, 2))?;
        }
        if i.bits(71, 1) != 0 {
            results[0] = self.emit.logical_and(results[0], results[1])?;
            results[1] = self.emit.logical_not(results[0])?;
        }
        self.emit
            .set_pred(Pred::new(i.bits(81, 3) as u8)?, results[0])?;
        self.emit
            .set_pred(Pred::new(i.bits(84, 3) as u8)?, results[1])
    }

    pub(super) fn pack_floats(&mut self) -> Result<()> {
        let i = self.instruction;
        if i.bits(62, 2) != 0
            || i.bits(72, 6) != 0
            || i.bits(79, 13) != 0
            || (i.opcode() == 0x23e && i.bits(40, 22) != 0)
        {
            return Err(unsupported("Native float packing mode"));
        }
        let b = self.operand(1)?;
        let merge = i.bits(78, 1) != 0;
        let a = if merge {
            Value::F32(0)
        } else {
            self.operand(0)?
        };
        // PACK_AB places B in the low half and A in the high half. MERGE_C
        // preserves C's low half in the high output half without reconverting it.
        // https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/shader_recompiler/frontend/ampere/lower_to_maxwell.h#L600-L624
        let pair = self.emit.composite_construct(&[b, a])?;
        let mut value = self.emit.pack_half2x16(pair)?;
        if merge {
            let c = self.reg(i.bits(64, 8))?;
            value = self
                .emit
                .bit_field_insert(value, c, Value::U32(16), Value::U32(16))?;
        }
        self.write(value)
    }

    pub(super) fn swizzled_add(&mut self) -> Result<()> {
        let i = self.instruction;
        if i.bits(40, 24) != 0 || i.bits(72, 24) != 0 {
            return Err(unsupported("Native quad swizzled add mode"));
        }
        let a = self.reg(i.bits(24, 8))?;
        let b = self.reg(i.bits(64, 8))?;
        let a = self.emit.bit_cast(Type::F32, a)?;
        let b = self.emit.bit_cast(Type::F32, b)?;
        // Native lane choices are a+b, a-b, b-a, b. The semantic IR table
        // indexes the negative coefficient of a before that of b.
        let selectors = i.bits(32, 8);
        let coefficients = (selectors & 0x55) << 1 | (selectors & 0xaa) >> 1;
        let control = FpControl {
            no_contraction: false,
            rounding: FpRounding::RN,
            fmz_mode: FmzMode::None,
        };
        let value = self
            .emit
            .f_swizzle_add(a, b, Value::U32(coefficients), control)?;
        self.write_float(value)
    }
    pub(super) fn fset(&mut self) -> Result<()> {
        let i = self.instruction;
        let predicates = i.opcode() & 0x1ff == 0xb;
        if i.bits(64, 8) != 0
            || i.bits(91, 1) != u32::from(matches!(i.opcode(), 0xc0a | 0xc0b))
            || (!predicates && i.bits(81, 6) != 0)
        {
            return Err(unsupported("Native floating comparison output mode"));
        }
        let a = self.operand(0)?;
        let b = self.operand(1)?;
        let control = FpControl {
            fmz_mode: if i.bits(80, 1) != 0 {
                FmzMode::FTZ
            } else {
                FmzMode::None
            },
            ..Default::default()
        };
        let comparison =
            crate::frontend::float_compare(&mut self.emit, a, b, i.bits(76, 4), control)?;
        if predicates {
            self.predicate_pair(comparison)
        } else {
            let input = self
                .emit
                .get_pred(Pred::new(i.bits(87, 3) as u8)?, i.bits(90, 1) != 0)?;
            let condition = self.combine(comparison, input, i.bits(74, 2))?;
            let value = self
                .emit
                .select(condition, Value::F32(1.0f32.to_bits()), Value::F32(0))?;
            self.write_float(value)
        }
    }
    pub(super) fn fmnmx(&mut self) -> Result<()> {
        let i = self.instruction;
        if i.bits(74, 6) != 0
            || i.bits(82, 5) != 0
            || i.bits(91, 1) != u32::from(i.opcode() == 0xc09)
        {
            return Err(unsupported("Native floating min/max mode"));
        }
        let a = self.operand(0)?;
        let b = self.operand(1)?;
        let predicate = self
            .emit
            .get_pred(Pred::new(i.bits(87, 3) as u8)?, i.bits(90, 1) != 0)?;
        let control = FpControl {
            fmz_mode: if i.bits(80, 1) != 0 {
                FmzMode::FTZ
            } else {
                FmzMode::None
            },
            ..Default::default()
        };
        let min = self.emit.fp_min(a, b, control)?;
        let max = self.emit.fp_max(a, b, control)?;
        let mut value = self.emit.select(predicate, min, max)?;
        if i.bits(81, 1) != 0 {
            let nan_a = self.emit.fp_is_nan(a)?;
            let nan_b = self.emit.fp_is_nan(b)?;
            let unordered = self.emit.logical_or(nan_a, nan_b)?;
            value = self.emit.select(unordered, Value::F32(0x7fffffff), value)?;
        }
        self.write_float(value)
    }
    pub(super) fn frnd(&mut self) -> Result<()> {
        let i = self.instruction;
        if i.bits(24, 8) != 0 || i.bits(64, 32) & !0x1c000 != 0x201000 {
            return Err(unsupported("Native floating round format"));
        }
        let value = self.operand(1)?;
        let control = FpControl {
            fmz_mode: if i.bits(80, 1) != 0 {
                FmzMode::FTZ
            } else {
                FmzMode::None
            },
            ..Default::default()
        };
        let value = match i.bits(78, 2) {
            0 => self.emit.fp_round_even(value, control)?,
            1 => self.emit.fp_floor(value, control)?,
            2 => self.emit.fp_ceil(value, control)?,
            _ => self.emit.fp_trunc(value, control)?,
        };
        self.write_float(value)
    }
    pub(super) fn hadd2_float(&mut self) -> Result<()> {
        let i = self.instruction;
        if i.bits(24, 8) != 255
            || i.bits(64, 32) != 0x4100
            || i.bits(60, 2) < 2
            || i.bits(40, 20) != 0
        {
            return Err(unsupported("Native half addition mode"));
        }
        // HADD2.F32 with -RZ converts the selected half without a packed result.
        let packed = self.reg(i.bits(32, 8))?;
        let pair = self.emit.unpack_half2x16(packed)?;
        let value = self
            .emit
            .composite_extract(pair, usize::from(i.bits(60, 2) == 3))?;
        let value = self
            .emit
            .fp_abs_neg(value, i.bits(62, 1) != 0, i.bits(63, 1) != 0)?;
        self.write_float(value)
    }
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
        let uniform = matches!(i.opcode(), 0x290 | 0x890);
        let extended = i.bits(74, 1) != 0;
        let outputs = [i.bits(81, 3), i.bits(84, 3)];
        if i.bits(73, 1) != 0
            || i.bits(76, 1) != 0
            || i.bits(91, 1) != u32::from(uniform || i.opcode() == 0xc10)
            || (!extended && (i.bits(77, 4) != 15 || i.bits(87, 4) != 15))
            || (outputs[0] != 7 && outputs[0] == outputs[1])
            || (matches!(i.opcode(), 0x210 | 0x290) && i.bits(32, 32) & !0x800000ff != 0)
            || (i.opcode() == 0xa10 && i.bits(62, 1) != 0)
            || (i.opcode() == 0xc10 && i.bits(32, 32) & !0x8000003f != 0)
        {
            return Err(unsupported("Native integer addition mode"));
        }
        let mut values = [
            self.register(i.bits(24, 8), uniform)?,
            if i.opcode() == 0x290 {
                self.ureg(i.bits(32, 8))?
            } else {
                self.raw_b()?
            },
            self.register(i.bits(64, 8), uniform)?,
        ];
        let negations = [
            i.bits(72, 1) != 0,
            !matches!(i.opcode(), 0x810 | 0x890) && i.bits(63, 1) != 0,
            i.bits(75, 1) != 0,
        ];
        let carries = outputs != [7, 7];
        for (value, negate) in values.iter_mut().zip(negations) {
            if negate {
                *value = if extended || carries {
                    self.emit.bitwise_not(*value)?
                } else {
                    self.emit.i_neg(*value)?
                };
            }
        }
        let mut additional = [Value::U32(0); 2];
        if extended {
            for (value, bit) in additional.iter_mut().zip([87, 77]) {
                let predicate = self.emit.get_pred(
                    Self::predicate(i.bits(bit, 3), uniform)?,
                    i.bits(bit + 3, 1) != 0,
                )?;
                *value = self.emit.select(predicate, Value::U32(1), Value::U32(0))?;
            }
        } else if carries {
            additional[0] = Value::U32(negations.into_iter().filter(|v| *v).count() as u32);
        }
        let mut sum = values[0];
        let mut carry_count = Value::U32(0);
        for addend in values[1..].iter().chain(&additional).copied() {
            if addend == Value::U32(0) {
                continue;
            }
            let next = self.emit.i_add(sum, addend)?;
            if carries {
                let carry = self.emit.i_less_than(next, sum, false)?;
                let carry = self.emit.select(carry, Value::U32(1), Value::U32(0))?;
                carry_count = self.emit.i_add(carry_count, carry)?;
            }
            sum = next;
        }
        self.write_register(sum, uniform)?;
        for (index, output) in outputs.into_iter().enumerate() {
            if output != 7 {
                let predicate = self.emit.i_greater_than_equal(
                    carry_count,
                    Value::U32(index as u32 + 1),
                    false,
                )?;
                self.emit
                    .set_pred(Self::predicate(output, uniform)?, predicate)?;
            }
        }
        Ok(())
    }
    pub(super) fn i2f_narrow(&mut self) -> Result<()> {
        let i = self.instruction;
        let uniform = i.opcode() == 0xd06;
        let mode = i.bits(64, 32) & !0x08000400;
        let width = match mode {
            0x1000 => 8,
            0x101000 => 16,
            _ => return Err(unsupported("Native narrow integer-to-float format")),
        };
        let selector = i.bits(60, 2);
        if i.bits(24, 8) != 0
            || i.bits(32, 28) > if uniform { 63 } else { 255 }
            || i.bits(62, 2) != 0
            || i.bits(91, 1) != u32::from(uniform)
            || (width == 16 && selector > 1)
        {
            return Err(unsupported("Native narrow integer-to-float source"));
        }
        let signed = i.bits(74, 1) != 0;
        let source = self.register(i.bits(32, 8), uniform)?;
        let value = self.emit.bit_field_extract(
            source,
            Value::U32(selector * width),
            Value::U32(width),
            signed,
        )?;
        let value = self.emit.convert_i_to_f(
            32,
            32,
            signed,
            value,
            FpControl {
                rounding: FpRounding::RN,
                ..Default::default()
            },
        )?;
        self.write_float(value)
    }
    pub(super) fn i2fp(&mut self) -> Result<()> {
        let i = self.instruction;
        if i.bits(64, 28) & !0xc400
            != if i.opcode() == 0xc45 {
                0x08201000
            } else {
                0x00201000
            }
        {
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
        if i.bits(73, 1) != 0 || i.bits(80, 12) != if i.opcode() == 0xd08 { 0x800 } else { 0 } {
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
        let mut value = if i.opcode() == 0x526 && i.bits(24, 8) != 63 {
            // The uniform source is a byte offset; the immediate selects a
            // component. Keep the indexed load in IR, with its actual inputs.
            let offset = self.ureg(i.bits(24, 8))?;
            let offset = self.emit.i_add(offset, Value::U32(i.bits(64, 8) * 4))?;
            self.emit.get_attribute_indexed(offset)?
        } else {
            self.emit.get_attribute(attribute)?
        };
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
