use super::{Opcode, Result, Source, Value, Visitor, unsupported};

impl Visitor<'_> {
    pub(super) fn lea(&mut self, source: Source, high: bool) -> Result<()> {
        let base = self.source(source)?;
        let (scale, neg, x) = if !high {
            (self.field::<39, 5>(), self.bit::<45>(), self.bit::<46>())
        } else if matches!(source, Source::Cbuf) {
            (self.field::<51, 5>(), self.bit::<56>(), self.bit::<57>())
        } else {
            (self.field::<28, 5>(), self.bit::<37>(), self.bit::<38>())
        };
        if x {
            return Err(unsupported("LEA X"));
        }
        if self.field::<48, 3>() != 7 {
            return Err(unsupported("LEA predicate destination"));
        }
        if self.bit::<47>() {
            return Err(unsupported("LEA CC"));
        }
        let mut offset = self.a()?;
        if high {
            let upper = self.source(Source::Reg39)?;
            let pair = self.emit.composite_construct(&[offset, upper])?;
            offset = self.emit.pack_uint2x32(pair)?;
        }
        if neg {
            offset = self.emit.i_neg(offset)?;
        }
        let scaled = if high {
            let shifted = self
                .emit
                .shift_right_logical(offset, Value::U32(32 - scale))?;
            let pair = self.emit.unpack_uint2x32(shifted)?;
            self.emit.composite_extract(pair, 0)?
        } else {
            self.emit.shift_left_logical(offset, Value::U32(scale))?
        };
        let result = self.emit.i_add(base, scaled)?;
        self.write(result)
    }
    pub(super) fn mov(&mut self, source: Source, wide: bool) -> Result<()> {
        let value = self.source(source)?;
        let mask = if wide {
            self.field::<12, 4>()
        } else {
            self.field::<39, 4>()
        };
        // Masked moves have no destination write in the supported behavior.
        if matches!(mask, 1 | 15) {
            self.write(value)?;
        }
        Ok(())
    }
    pub(super) fn sel(&mut self, source: Source) -> Result<()> {
        let b = self.source(source)?;
        let pred = self.predicate::<39, 42>()?;
        let a = self.a()?;
        let value = self.emit.select(pred, a, b)?;
        self.write(value)
    }
    pub(super) fn iadd(&mut self, source: Source, wide: bool) -> Result<()> {
        let mut b = self.source(source)?;
        let (neg_a, po, sat, x, cc) = if wide {
            let po = self.field::<55, 2>() == 3;
            (
                !po && self.bit::<56>(),
                po,
                self.bit::<54>(),
                self.bit::<53>(),
                self.bit::<52>(),
            )
        } else {
            let po = self.field::<48, 2>() == 3;
            if !po && self.bit::<48>() {
                b = self.emit.i_neg(b)?;
            }
            (
                self.bit::<49>(),
                po,
                self.bit::<50>(),
                self.bit::<43>(),
                self.bit::<47>(),
            )
        };
        if sat {
            return Err(unsupported("IADD SAT"));
        }
        if x && po {
            return Err(unsupported("IADD X+PO"));
        }
        let mut a = self.a()?;
        if neg_a {
            a = self.emit.i_neg(a)?;
        }
        let mut result = self.emit.i_add(a, b)?;
        if x {
            let carry = self.emit.get_c_flag()?;
            let carry = self.emit.select(carry, Value::U32(1), Value::U32(0))?;
            result = self.emit.i_add(result, carry)?;
        }
        if po {
            result = self.emit.i_add(result, Value::U32(1))?;
        }
        if cc {
            if po {
                return Err(unsupported("IADD CC+PO"));
            }
            if x {
                return Err(unsupported("IADD X+CC"));
            }
            self.flags_arithmetic(result)?;
        }
        self.write(result)
    }
    fn half(&mut self, value: Value, half: u32) -> Result<Value> {
        match half {
            0 => Ok(value),
            1 | 2 => self.emit.bit_field_extract(
                value,
                Value::U32((half - 1) * 16),
                Value::U32(16),
                false,
            ),
            _ => Err(unsupported("Invalid integer half")),
        }
    }
    pub(super) fn iadd3(&mut self, source: Source, register_form: bool) -> Result<()> {
        let mut a = self.a()?;
        let mut b = self.source(source)?;
        let mut c = self.source(Source::Reg39)?;
        let shift = if register_form {
            a = self.half(a, self.field::<35, 2>())?;
            b = self.half(b, self.field::<33, 2>())?;
            c = self.half(c, self.field::<31, 2>())?;
            self.field::<37, 2>()
        } else {
            0
        };
        if self.bit::<51>() {
            a = self.emit.i_neg(a)?;
        }
        if self.bit::<50>() {
            b = self.emit.i_neg(b)?;
        }
        if self.bit::<49>() {
            c = self.emit.i_neg(c)?;
        }
        let mut lhs = self.emit.i_add(a, b)?;
        if self.bit::<48>() {
            if shift == 1 {
                return Err(unsupported("IADD3 X+RS"));
            }
            let carry = self.emit.get_c_flag()?;
            let carry = self.emit.select(carry, Value::U32(1), Value::U32(0))?;
            lhs = self.emit.i_add(lhs, carry)?;
        }
        let shifted = match shift {
            0 => lhs,
            1 => {
                let carry = self.emit.get_carry_from_op(lhs)?;
                let shifted = self.emit.shift_right_logical(lhs, Value::U32(16))?;
                let full = self.emit.i_add(shifted, Value::U32(0x10000))?;
                self.emit.select(carry, full, shifted)?
            }
            2 => self.emit.shift_left_logical(lhs, Value::U32(16))?,
            _ => return Err(unsupported("Invalid IADD3 shift")),
        };
        let result = self.emit.i_add(shifted, c)?;
        self.write(result)?;
        if self.bit::<47>() {
            if self.bit::<48>() {
                return Err(unsupported("IADD3 X+CC"));
            }
            self.flags_zs(result, true)?;
            let carry = self.emit.get_carry_from_op(result)?;
            self.emit.set_c_flag(carry)?;
            let first_overflow = self.emit.i_less_than(lhs, a, false)?;
            let overflow = self.emit.get_overflow_from_op(result)?;
            let overflow = self.emit.logical_or(overflow, first_overflow)?;
            self.emit.set_o_flag(overflow)?;
        }
        Ok(())
    }
    pub(super) fn iscadd(&mut self, source: Source, wide: bool) -> Result<()> {
        let mut b = self.source(source)?;
        let (cc, neg_a, neg_b, scale) = if wide {
            (self.bit::<52>(), false, false, self.field::<53, 5>())
        } else {
            (
                self.bit::<47>(),
                self.bit::<49>(),
                self.bit::<48>(),
                self.field::<39, 5>(),
            )
        };
        let po = neg_a && neg_b;
        let mut a = self.a()?;
        if po {
            b = self.emit.i_add(b, Value::U32(1))?;
        } else {
            if neg_a {
                a = self.emit.i_neg(a)?;
            }
            if neg_b {
                b = self.emit.i_neg(b)?;
            }
        }
        let a = self.emit.shift_left_logical(a, Value::U32(scale))?;
        let result = self.emit.i_add(a, b)?;
        self.write(result)?;
        if cc {
            self.flags_zs(result, true)?;
            let mut carry = self.emit.get_carry_from_op(result)?;
            let mut overflow = self.emit.get_overflow_from_op(result)?;
            if po {
                let extra_carry = self.emit.get_carry_from_op(b)?;
                let extra_overflow = self.emit.get_overflow_from_op(b)?;
                carry = self.emit.logical_or(carry, extra_carry)?;
                overflow = self.emit.logical_or(overflow, extra_overflow)?;
            }
            self.emit.set_c_flag(carry)?;
            self.emit.set_o_flag(overflow)?;
        }
        Ok(())
    }
    pub(super) fn imnmx(&mut self, source: Source) -> Result<()> {
        let b = self.source(source)?;
        if self.bit::<47>() {
            return Err(unsupported("IMNMX CC"));
        }
        if self.field::<43, 2>() != 0 {
            return Err(unsupported("IMNMX.MODE"));
        }
        let pred = self.predicate::<39, 42>()?;
        let a = self.a()?;
        let min = self.emit.i_min(a, b, self.bit::<48>())?;
        let max = self.emit.i_max(a, b, self.bit::<48>())?;
        let result = self.emit.select(pred, min, max)?;
        self.write(result)
    }
    pub(super) fn shift(&mut self, source: Source, right: bool) -> Result<()> {
        let shift = self.source(source)?;
        if self.bit::<43>() {
            return Err(unsupported("Shift X mode"));
        }
        if self.bit::<47>() {
            return Err(unsupported("Shift CC"));
        }
        let mut base = self.a()?;
        if right && self.bit::<40>() {
            base = self.emit.bit_reverse(base)?;
        }
        let wrapped = self.bit::<39>();
        let safe = if wrapped {
            self.emit.bitwise_and(shift, Value::U32(31))?
        } else {
            shift
        };
        let mut result = if !right {
            self.emit.shift_left_logical(base, safe)?
        } else if self.bit::<48>() {
            self.emit.shift_right_arithmetic(base, safe)?
        } else {
            self.emit.shift_right_logical(base, safe)?
        };
        if !wrapped {
            // The clamp's sign comes from the shifted result, including SHR.U32.
            let clamped = if right {
                let negative = self.emit.i_less_than(result, Value::U32(0), true)?;
                self.emit
                    .select(negative, Value::U32(u32::MAX), Value::U32(0))?
            } else {
                Value::U32(0)
            };
            let safe = self.emit.i_less_than(shift, Value::U32(32), false)?;
            result = self.emit.select(safe, result, clamped)?;
        }
        self.write(result)
    }
    pub(super) fn shf(&mut self, source: Source, right: bool) -> Result<()> {
        let shift = self.source(source)?;
        let high = self.source(Source::Reg39)?;
        if self.bit::<47>() {
            return Err(unsupported("SHF CC"));
        }
        if self.field::<48, 2>() != 0 {
            return Err(unsupported("SHF X Mode"));
        }
        let mode = self.field::<37, 2>();
        if mode == 1 {
            return Err(unsupported("SHF undefined MaxShift"));
        }
        let low = self.a()?;
        let pair = self.emit.composite_construct(&[low, high])?;
        let packed = self.emit.pack_uint2x32(pair)?;
        let max = if mode == 0 { 32 } else { 63 };
        let safe = if self.bit::<50>() {
            self.emit.bitwise_and(shift, Value::U32(max - 1))?
        } else {
            self.emit.u_min(shift, Value::U32(max))?
        };
        let shifted = if !right {
            self.emit.shift_left_logical(packed, safe)?
        } else if mode == 3 {
            self.emit.shift_right_arithmetic(packed, safe)?
        } else {
            self.emit.shift_right_logical(packed, safe)?
        };
        let pair = self.emit.unpack_uint2x32(shifted)?;
        let result = self.emit.composite_extract(pair, usize::from(!right))?;
        self.write(result)
    }
    pub(super) fn popc(&mut self, source: Source) -> Result<()> {
        let mut value = self.source(source)?;
        if self.bit::<40>() {
            value = self.emit.bitwise_not(value)?;
        }
        let result = self.emit.bit_count(value)?;
        self.write(result)
    }
    pub(super) fn flo(&mut self, source: Source) -> Result<()> {
        let mut value = self.source(source)?;
        if self.bit::<47>() {
            return Err(unsupported("FLO CC"));
        }
        if self.bit::<40>() {
            value = self.emit.bitwise_not(value)?;
        }
        let mut result = if self.bit::<48>() {
            self.emit.find_s_msb(value)?
        } else {
            self.emit.find_u_msb(value)?
        };
        if self.bit::<41>() {
            let missing = self.emit.i_equal(result, Value::U32(u32::MAX))?;
            let shift = self.emit.bitwise_xor(result, Value::U32(31))?;
            result = self.emit.select(missing, result, shift)?;
        }
        self.write(result)
    }
    pub(super) fn bfe(&mut self, source: Source) -> Result<()> {
        let src = self.source(source)?;
        let offset = self
            .emit
            .bit_field_extract(src, Value::U32(0), Value::U32(8), false)?;
        let count = self
            .emit
            .bit_field_extract(src, Value::U32(8), Value::U32(8), false)?;
        let empty = self.emit.i_equal(count, Value::U32(0))?;
        let end = self.emit.i_add(offset, count)?;
        let exceeds = self.emit.i_greater_than_equal(end, Value::U32(32), false)?;
        let replicate = self
            .emit
            .i_greater_than_equal(offset, Value::U32(32), false)?;
        let mut base = self.a()?;
        if self.bit::<40>() {
            base = self.emit.bit_reverse(base)?;
        }
        let mut result = self
            .emit
            .bit_field_extract(base, offset, count, self.bit::<48>())?;
        if self.bit::<48>() {
            let negative = self.emit.i_less_than(base, Value::U32(0), true)?;
            let replicated = self
                .emit
                .select(negative, Value::U32(u32::MAX), Value::U32(0))?;
            let sign = self
                .emit
                .bit_field_extract(base, Value::U32(31), Value::U32(1), false)?;
            result = self.emit.select(replicate, replicated, result)?;
            let extended =
                self.emit
                    .bit_field_insert(result, sign, Value::U32(31), Value::U32(1))?;
            result = self.emit.select(exceeds, extended, result)?;
        }
        result = self.emit.select(empty, Value::U32(0), result)?;
        self.write(result)?;
        if self.bit::<47>() {
            self.flags_zs(result, false)?;
            self.clear_co()?;
        }
        Ok(())
    }
    pub(super) fn bfi(&mut self, source: Source, base: Source) -> Result<()> {
        let src = self.source(source)?;
        let base = self.source(base)?;
        let offset = self
            .emit
            .bit_field_extract(src, Value::U32(0), Value::U32(8), false)?;
        let count = self
            .emit
            .bit_field_extract(src, Value::U32(8), Value::U32(8), false)?;
        let outside = self
            .emit
            .i_greater_than_equal(offset, Value::U32(32), false)?;
        let exceeds = self.emit.i_greater_than(count, Value::U32(32), false)?;
        let remaining = self.emit.i_sub(Value::U32(32), offset)?;
        let count = self.emit.select(exceeds, remaining, count)?;
        let insert = self.a()?;
        let result = self.emit.bit_field_insert(base, insert, offset, count)?;
        let result = self.emit.select(outside, base, result)?;
        self.write(result)?;
        if self.bit::<47>() {
            self.flags_zs(result, false)?;
            self.clear_co()?;
        }
        Ok(())
    }
    pub(super) fn xmad(&mut self, op: Opcode) -> Result<()> {
        let (b, c, mode, half_b, psl, merge, x) = match op {
            Opcode::XMAD_reg => (
                Source::Reg20,
                Source::Reg39,
                self.field::<50, 3>(),
                self.bit::<35>(),
                self.bit::<36>(),
                self.bit::<37>(),
                self.bit::<38>(),
            ),
            Opcode::XMAD_imm => (
                Source::Imm16,
                Source::Reg39,
                self.field::<50, 3>(),
                false,
                self.bit::<36>(),
                self.bit::<37>(),
                self.bit::<38>(),
            ),
            Opcode::XMAD_rc => (
                Source::Reg39,
                Source::Cbuf,
                self.field::<50, 2>(),
                self.bit::<52>(),
                false,
                false,
                self.bit::<54>(),
            ),
            Opcode::XMAD_cr => (
                Source::Cbuf,
                Source::Reg39,
                self.field::<50, 2>(),
                self.bit::<52>(),
                self.bit::<55>(),
                self.bit::<56>(),
                self.bit::<54>(),
            ),
            _ => unreachable!(),
        };
        let b = self.source(b)?;
        let c = self.source(c)?;
        if x {
            return Err(unsupported("XMAD X"));
        }
        let a = self.a()?;
        let a = self.emit.bit_field_extract(
            a,
            Value::U32(if self.bit::<53>() { 16 } else { 0 }),
            Value::U32(16),
            self.bit::<48>(),
        )?;
        let extracted_b = self.emit.bit_field_extract(
            b,
            Value::U32(if half_b { 16 } else { 0 }),
            Value::U32(16),
            self.bit::<49>(),
        )?;
        let mut product = self.emit.i_mul(a, extracted_b)?;
        if psl {
            product = self.emit.shift_left_logical(product, Value::U32(16))?;
        }
        let c = match mode {
            0 => c,
            1 | 2 => self.emit.bit_field_extract(
                c,
                Value::U32((mode - 1) * 16),
                Value::U32(16),
                false,
            )?,
            4 => {
                let high = self.emit.shift_left_logical(b, Value::U32(16))?;
                self.emit.i_add(high, c)?
            }
            3 => return Err(unsupported("XMAD CSFU")),
            _ => return Err(unsupported("Invalid XMAD select mode")),
        };
        let mut result = self.emit.i_add(product, c)?;
        if merge {
            let low = self
                .emit
                .bit_field_extract(b, Value::U32(0), Value::U32(16), false)?;
            result = self
                .emit
                .bit_field_insert(result, low, Value::U32(16), Value::U32(16))?;
        }
        if self.bit::<47>() {
            return Err(unsupported("XMAD CC"));
        }
        self.write(result)
    }
    pub(super) fn i2i(&mut self, source: Source) -> Result<()> {
        let src = self.source(source)?;
        let width = |format| match format {
            0 => Ok(8),
            1 => Ok(16),
            2 => Ok(32),
            _ => Err(unsupported("Invalid integer width")),
        };
        let source_width = width(self.field::<10, 2>())?;
        let destination_width = width(self.field::<8, 2>())?;
        let selector = self.field::<41, 3>();
        if (source_width == 16 && matches!(selector, 1 | 3))
            || (source_width == 32 && selector != 0)
        {
            return Err(unsupported(
                "Integer source width incompatible with selector",
            ));
        }
        let signed = self.bit::<13>();
        let dest_signed = self.bit::<12>();
        let mut src = self.emit.bit_field_extract(
            src,
            Value::U32(selector * 8),
            Value::U32(source_width),
            signed,
        )?;
        if self.bit::<49>() {
            src = self.emit.i_abs(src)?;
        }
        if self.bit::<45>() {
            src = self.emit.i_neg(src)?;
        }
        let result = if self.bit::<50>() {
            let min = if dest_signed && signed {
                u32::MAX << (destination_width - 1)
            } else {
                0
            };
            let max = u32::MAX >> (32 - destination_width + u32::from(dest_signed));
            if !dest_signed && signed {
                src = self.emit.s_max(Value::U32(0), src)?;
            }
            if dest_signed && signed {
                self.emit.s_clamp(src, Value::U32(min), Value::U32(max))?
            } else {
                self.emit.u_clamp(src, Value::U32(min), Value::U32(max))?
            }
        } else {
            self.emit
                .bit_field_extract(src, Value::U32(0), Value::U32(destination_width), false)?
        };
        self.write(result)?;
        if self.bit::<47>() {
            self.flags_zs(result, true)?;
            self.clear_co()?;
        }
        Ok(())
    }
}
