use super::*;

impl Visitor<'_> {
    pub(super) fn extended_multiply(&mut self) -> Result<()> {
        let i = self.instruction;
        let wide = matches!(i.opcode(), 0x225 | 0x625 | 0x825);
        let immediate = matches!(i.opcode(), 0x825 | 0x827);
        let constant_addend = i.opcode() == 0x625;
        let signed = i.bits(73, 1) != 0;
        let carry_output = i.bits(81, 3);
        let addend = if constant_addend { 255 } else { i.bits(64, 8) };
        let destination = i.bits(16, 8);
        if i.bits(64, 32) & !0x000e02ff != 0x07800000
            || (matches!(i.opcode(), 0x225 | 0x227) && i.bits(32, 32) > 255)
            || (matches!(i.opcode(), 0x625 | 0xa27) && i.bits(62, 2) != 0)
            || (constant_addend && i.bits(38, 16) > 0xfff8)
            || (addend != 255 && (addend & 1 != 0 || addend >= 254))
            || (wide && (destination & 1 != 0 || destination >= 254))
            || (signed && carry_output != 7)
        {
            return Err(unsupported("Native extended multiply mode"));
        }
        let a = self.reg(i.bits(24, 8))?;
        let b = if immediate {
            Value::U32(i.bits(32, 32))
        } else if i.opcode() == 0xa27 {
            self.constant()?
        } else {
            self.reg(i.bits(if constant_addend { 64 } else { 32 }, 8))?
        };
        let (add_low, add_high) = if constant_addend {
            let low = self.constant()?;
            let high = self.emit.get_cbuf(
                Value::U32(i.bits(54, 5)),
                Value::U32(i.bits(38, 16) + 4),
                32,
                false,
            )?;
            (low, high)
        } else {
            (
                self.reg(addend)?,
                self.reg(if addend == 255 { 255 } else { addend + 1 })?,
            )
        };
        let product = self.emit.i_mul_extended(a, b, signed)?;
        let low = self.emit.composite_extract(product, 0)?;
        let high = self.emit.composite_extract(product, 1)?;
        let low = self.emit.i_add(low, add_low)?;
        let carry = self.emit.get_carry_from_op(low)?;
        let carry = self.emit.select(carry, Value::U32(1), Value::U32(0))?;
        let high_sum = self.emit.i_add(high, add_high)?;
        let high = self.emit.i_add(high_sum, carry)?;
        if carry_output != 7 {
            let carry_a = self.emit.get_carry_from_op(high_sum)?;
            let carry_b = self.emit.get_carry_from_op(high)?;
            let carry = self.emit.logical_or(carry_a, carry_b)?;
            self.emit.set_pred(Pred::new(carry_output as u8)?, carry)?;
        }
        // Read every source before writing either result: destination pairs
        // may alias the multiplicands or the 64-bit addend.
        if wide {
            self.write(low)?;
            self.emit.set_reg(Reg((destination + 1) as u8), high)?;
            Ok(())
        } else {
            self.write(high)
        }
    }

    pub(super) fn permute_bytes(&mut self) -> Result<()> {
        let i = self.instruction;
        if i.bits(48, 16) != 0 || i.bits(72, 24) != 0 {
            return Err(unsupported("Native byte permutation mode"));
        }
        let a = self.reg(i.bits(24, 8))?;
        let b = self.reg(i.bits(64, 8))?;
        let mut result = Value::U32(0);
        // Each nibble selects one byte of b:a, optionally replicating its sign.
        // https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-prmt
        for byte in 0..4 {
            let selector = i.bits(32 + byte * 4, 4);
            let sign = selector & 8 != 0;
            let source = if selector & 4 != 0 { b } else { a };
            let position = (selector & 3) * 8 + if sign { 7 } else { 0 };
            let value = self.emit.bit_field_extract(
                source,
                Value::U32(position),
                Value::U32(if sign { 1 } else { 8 }),
                sign,
            )?;
            result =
                self.emit
                    .bit_field_insert(result, value, Value::U32(byte * 8), Value::U32(8))?;
        }
        self.write(result)
    }
    pub(super) fn bit_scan(&mut self) -> Result<()> {
        let i = self.instruction;
        let uniform_source = matches!(i.opcode(), 0xd00 | 0xd09);
        let population = i.opcode() & 0xff == 9;
        let mode = u32::from(uniform_source) << 27;
        if i.bits(24, 8) != 0
            || i.bits(32, 31) > if uniform_source { 63 } else { 255 }
            || if population {
                i.bits(64, 32) != mode
            } else {
                i.bits(64, 32) & !0x600 != (mode | 0xe0000)
            }
        {
            return Err(unsupported("Native bit-scan mode"));
        }
        let mut value = self.register(i.bits(32, 8), uniform_source)?;
        if i.bits(63, 1) != 0 {
            value = self.emit.bitwise_not(value)?;
        }
        let result = if population {
            self.emit.bit_count(value)?
        } else {
            let position = if i.bits(73, 1) != 0 {
                self.emit.find_s_msb(value)?
            } else {
                self.emit.find_u_msb(value)?
            };
            if i.bits(74, 1) != 0 {
                let missing = self.emit.i_equal(position, Value::U32(u32::MAX))?;
                let shift = self.emit.bitwise_xor(position, Value::U32(31))?;
                self.emit.select(missing, position, shift)?
            } else {
                position
            }
        };
        self.write(result)
    }
    pub(super) fn imad(&mut self) -> Result<()> {
        let i = self.instruction;
        let uniform = matches!(i.opcode(), 0x2a4 | 0x4a4 | 0x8a4);
        let form = i.bits(9, 3);
        let extended_zero = i.opcode() == 0x624
            && i.bits(24, 8) == 255
            && i.bits(64, 32) & !0x07800000 == 0x000e06ff;
        let extended_register = i.opcode() == 0x224
            && i.bits(24, 8) == 255
            && i.bits(32, 32) == 255
            && i.bits(64, 32) & !0x07800aff == 0x000e0400;
        let extended_add = i.opcode() == 0x824
            && i.bits(32, 32) == 1
            && i.bits(64, 32) & !0x078008ff == 0x000e0600;
        if extended_zero || extended_register || extended_add {
            let mut c = if extended_zero {
                self.constant()?
            } else {
                self.reg(i.bits(64, 8))?
            };
            if !extended_zero && i.bits(75, 1) != 0 {
                c = self.emit.bitwise_not(c)?;
            }
            let carry = self
                .emit
                .get_pred(Pred::new(i.bits(87, 3) as u8)?, i.bits(90, 1) != 0)?;
            let carry = self.emit.select(carry, Value::U32(1), Value::U32(0))?;
            let a = if extended_add {
                self.reg(i.bits(24, 8))?
            } else {
                Value::U32(0)
            };
            let sum = self.emit.i_add(a, c)?;
            let sum = self.emit.i_add(sum, carry)?;
            return self.write(sum);
        }
        let negate_register = matches!(form, 1 | 4) && i.bits(75, 1) != 0;
        let allowed = if matches!(form, 1 | 4) { 0xaff } else { 0x2ff };
        let uniform_operand = matches!(form, 6 | 7);
        if i.bits(64, 32) & !allowed != (0x078e0000 | (u32::from(uniform || uniform_operand) << 27))
            || (form == 1 && i.bits(32, 32) > 255)
            || (form == 3 && i.bits(62, 1) != 0)
            || (form == 5 && i.bits(62, 2) != 0)
            || (uniform_operand && i.bits(32, 32) & !if form == 7 { 0x8000003f } else { 0x3f } != 0)
        {
            return Err(unsupported(
                "Native integer multiply-add carry/extended mode",
            ));
        }
        let a = self.register(i.bits(24, 8), uniform)?;
        let (b, mut c) = match form {
            1 => (
                self.register(i.bits(32, 8), uniform)?,
                self.register(i.bits(64, 8), uniform)?,
            ),
            2 => (
                self.register(i.bits(64, 8), uniform)?,
                Value::U32(i.bits(32, 32)),
            ),
            3 => (self.reg(i.bits(64, 8))?, self.constant()?),
            4 => (
                Value::U32(i.bits(32, 32)),
                self.register(i.bits(64, 8), uniform)?,
            ),
            5 => (self.constant()?, self.reg(i.bits(64, 8))?),
            6 => (self.ureg(i.bits(32, 6))?, self.reg(i.bits(64, 8))?),
            7 => (self.reg(i.bits(64, 8))?, self.ureg(i.bits(32, 6))?),
            _ => unreachable!(),
        };
        if negate_register || (matches!(form, 3 | 7) && i.bits(63, 1) != 0) {
            c = self.emit.i_neg(c)?;
        }
        // The low word is identical for signed and unsigned multiplication.
        let product = self.emit.i_mul(a, b)?;
        let value = self.emit.i_add(product, c)?;
        self.write_register(value, uniform)
    }
    pub(super) fn lop3(&mut self) -> Result<()> {
        let i = self.instruction;
        let uniform = matches!(i.opcode(), 0x292 | 0x892);
        if i.bits(80, 1) != 0
            || i.bits(87, 5) != (15 | (u32::from(uniform || i.opcode() == 0xc12) << 4))
            || (matches!(i.opcode(), 0x212 | 0x292) && i.bits(40, 24) != 0)
            || (i.opcode() == 0xa12 && i.bits(62, 2) != 0)
        {
            return Err(unsupported("Native LUT predicate/source mode"));
        }
        let a = self.register(i.bits(24, 8), uniform)?;
        let b = if i.opcode() == 0x292 {
            self.ureg(i.bits(32, 8))?
        } else {
            self.raw_b()?
        };
        let c = self.register(i.bits(64, 8), uniform)?;
        let result = crate::frontend::lut::apply(&mut self.emit, a, b, c, i.bits(72, 8) as u8)?;
        let pred = self.emit.i_not_equal(result, Value::U32(0))?;
        self.write_register(result, uniform)?;
        self.emit
            .set_pred(Self::predicate(i.bits(81, 3), uniform)?, pred)
    }
    pub(super) fn isetp(&mut self) -> Result<()> {
        let i = self.instruction;
        if i.bits(64, 4) != 0
            || i.bits(79, 2) != 0
            || i.bits(91, 1) != u32::from(i.opcode() == 0xc0c)
        {
            return Err(unsupported("Native integer comparison mode"));
        }
        let a = self.reg(i.bits(24, 8))?;
        let b = self.raw_b()?;
        let op = i.bits(76, 3);
        let signed = i.bits(73, 1) != 0;
        let mut comparison = crate::frontend::integer_compare(&mut self.emit, a, b, op, signed)?;
        if i.bits(72, 1) != 0 {
            let low = self
                .emit
                .get_pred(Pred::new(i.bits(68, 3) as u8)?, i.bits(71, 1) != 0)?;
            comparison = match op {
                0 | 7 => comparison,
                2 => self.emit.logical_and(comparison, low)?,
                5 => self.emit.logical_or(comparison, low)?,
                _ => {
                    let equal = self.emit.i_equal(a, b)?;
                    let low = self.emit.logical_and(equal, low)?;
                    let strict = crate::frontend::integer_compare(
                        &mut self.emit,
                        a,
                        b,
                        if op < 4 { 1 } else { 4 },
                        signed,
                    )?;
                    self.emit.logical_or(strict, low)?
                }
            };
        }
        self.predicate_pair(comparison)
    }
    pub(super) fn imnmx(&mut self) -> Result<()> {
        let i = self.instruction;
        if i.bits(64, 32) & !0x04000200 != 0x03800000
            || (i.opcode() == 0x217 && i.bits(32, 32) > 255)
            || (i.opcode() == 0xa17 && i.bits(62, 2) != 0)
        {
            return Err(unsupported("Native integer min/max mode"));
        }
        let a = self.reg(i.bits(24, 8))?;
        let b = self.raw_b()?;
        let min = self.emit.i_min(a, b, i.bits(73, 1) != 0)?;
        let max = self.emit.i_max(a, b, i.bits(73, 1) != 0)?;
        let pred = self
            .emit
            .get_pred(Pred::new(i.bits(87, 3) as u8)?, i.bits(90, 1) != 0)?;
        let value = self.emit.select(pred, min, max)?;
        self.write(value)
    }
    fn scalar_shift(
        &mut self,
        value: Value,
        amount: Value,
        right: bool,
        signed: bool,
        wrap: bool,
    ) -> Result<Value> {
        let shift = if wrap {
            self.emit.bitwise_and(amount, Value::U32(31))?
        } else {
            self.emit.u_min(amount, Value::U32(31))?
        };
        let shifted = if !right {
            self.emit.shift_left_logical(value, shift)?
        } else if signed {
            self.emit.shift_right_arithmetic(value, shift)?
        } else {
            self.emit.shift_right_logical(value, shift)?
        };
        if wrap || signed {
            Ok(shifted)
        } else {
            let in_range = self.emit.i_less_than(amount, Value::U32(32), false)?;
            self.emit.select(in_range, shifted, Value::U32(0))
        }
    }
    pub(super) fn shift_variable(&mut self) -> Result<()> {
        let i = self.instruction;
        let low = i.bits(24, 8);
        let high = i.bits(64, 8);
        let wrap = i.bits(75, 1) != 0;
        if i.opcode() == 0x419 {
            if low != 255 || i.bits(64, 32) & !0x8ff != 0x10600 {
                return Err(unsupported("Native immediate-value funnel shift mode"));
            }
            let shift = self.reg(high)?;
            let value = self.scalar_shift(Value::U32(i.bits(32, 32)), shift, false, false, wrap)?;
            return self.write(value);
        }
        let shift = self.raw_b()?;
        if (i.opcode() == 0x219 && i.bits(32, 32) > 255)
            || (i.opcode() == 0xa19 && i.bits(62, 2) != 0)
        {
            return Err(unsupported("Native variable shift operand"));
        }
        if i.opcode() == 0x219 && i.bits(64, 32) & !0xff == 0x1200 {
            let low = self.reg(low)?;
            let high = self.reg(high)?;
            let pair = self.emit.composite_construct(&[low, high])?;
            let pair = self.emit.pack_uint2x32(pair)?;
            let limited = self.emit.u_min(shift, Value::U32(63))?;
            let shifted = self.emit.shift_right_logical(pair, limited)?;
            let pair = self.emit.unpack_uint2x32(shifted)?;
            let word = self.emit.composite_extract(pair, 0)?;
            let in_range = self.emit.i_less_than(shift, Value::U32(64), false)?;
            let value = self.emit.select(in_range, word, Value::U32(0))?;
            return self.write(value);
        }
        let right = i.bits(76, 1) != 0;
        let upper = i.bits(80, 1) != 0;
        let signed = i.bits(73, 2) == 2;
        let expected =
            if signed { 0x400 } else { 0x600 } | if i.opcode() == 0xc19 { 0x08000000 } else { 0 };
        if i.bits(64, 32) & !0x118ff != expected
            || (signed && (i.opcode() != 0x219 || !right || !upper || low != 255))
            || (!signed
                && if right {
                    if upper { low != 255 } else { high != 255 }
                } else if upper {
                    low != 255
                } else {
                    high != 255
                })
        {
            return Err(unsupported("Native variable funnel shift mode"));
        }
        let source = self.reg(if upper { high } else { low })?;
        let value = self.scalar_shift(source, shift, right, signed, wrap)?;
        self.write(value)
    }
    pub(super) fn shf(&mut self) -> Result<()> {
        let i = self.instruction;
        let mode = i.bits(73, 2);
        let shift = i.bits(32, 32);
        let uniform = i.opcode() == 0x899;
        if mode == 0
            || shift > 32
            || i.bits(64, 32) & !0x116ff != if uniform { 0x08000000 } else { 0 }
        {
            return Err(unsupported("Native funnel shift mode"));
        }
        let low = self.register(i.bits(24, 8), uniform)?;
        let high = self.register(i.bits(64, 8), uniform)?;
        let right = i.bits(76, 1) != 0;
        let high_result = i.bits(80, 1) != 0;
        // The outer word never depends on the other half of the pair. Keep
        // these operations at 32 bits, including the defined shift-by-32 result.
        if (!right && !high_result) || (right && high_result) {
            let value = if !right {
                if shift == 32 {
                    Value::U32(0)
                } else {
                    self.emit.shift_left_logical(low, Value::U32(shift))?
                }
            } else if mode == 2 {
                self.emit
                    .shift_right_arithmetic(high, Value::U32(shift.min(31)))?
            } else if shift == 32 {
                Value::U32(0)
            } else {
                self.emit.shift_right_logical(high, Value::U32(shift))?
            };
            return self.write_register(value, uniform);
        }
        let pair = self.emit.composite_construct(&[low, high])?;
        let packed = self.emit.pack_uint2x32(pair)?;
        let shifted = if i.bits(76, 1) == 0 {
            self.emit.shift_left_logical(packed, Value::U32(shift))?
        } else if mode == 2 {
            self.emit
                .shift_right_arithmetic(packed, Value::U32(shift))?
        } else {
            self.emit.shift_right_logical(packed, Value::U32(shift))?
        };
        let pair = self.emit.unpack_uint2x32(shifted)?;
        let value = self.emit.composite_extract(pair, i.bits(80, 1) as usize)?;
        self.write_register(value, uniform)
    }
    pub(super) fn lea(&mut self) -> Result<()> {
        let i = self.instruction;
        let high = i.bits(80, 1) != 0;
        let extended = i.bits(74, 1) != 0;
        let signed = i.bits(73, 1) != 0;
        let shift = i.bits(75, 5);
        let carry_out = i.bits(81, 3);
        if i.bits(72, 1) != 0
            || i.bits(84, 3) != 0
            || i.bits(91, 1) != u32::from(i.opcode() == 0xc11)
            || (i.opcode() == 0x211 && i.bits(32, 32) & !0x800000ff != 0)
            || (i.opcode() == 0xa11 && (i.bits(62, 1) != 0 || (high && i.bits(63, 1) != 0)))
            || (i.opcode() == 0xc11 && i.bits(32, 32) & !if high { 0x3f } else { 0x8000003f } != 0)
            || (high && (carry_out != 7 || (signed && i.bits(64, 8) != 255)))
            || (!high && (extended || i.bits(64, 8) != 255))
            || (!extended && i.bits(87, 3) != 7)
            || ((extended || carry_out != 7) && i.opcode() != 0x811 && i.bits(63, 1) != 0)
        {
            return Err(unsupported("Native LEA high/extended/sign mode"));
        }
        let a = self.reg(i.bits(24, 8))?;
        let shifted = if high {
            if signed {
                // A zero shift still reads the sign-extended high word; avoid
                // an undefined SPIR-V shift by 32.
                self.emit.shift_right_arithmetic(
                    a,
                    Value::U32(if shift == 0 { 31 } else { 32 - shift }),
                )?
            } else {
                let upper = self.reg(i.bits(64, 8))?;
                if shift == 0 {
                    upper
                } else {
                    let upper = self.emit.shift_left_logical(upper, Value::U32(shift))?;
                    let lower = self.emit.shift_right_logical(a, Value::U32(32 - shift))?;
                    self.emit.bitwise_or(upper, lower)?
                }
            }
        } else {
            self.emit.shift_left_logical(a, Value::U32(shift))?
        };
        let mut b = self.raw_b()?;
        if i.opcode() != 0x811 && i.bits(63, 1) != 0 {
            b = self.emit.i_neg(b)?;
        }
        let mut sum = self.emit.i_add(shifted, b)?;
        if extended {
            let carry = self
                .emit
                .get_pred(Pred::new(i.bits(87, 3) as u8)?, i.bits(90, 1) != 0)?;
            let carry = self.emit.select(carry, Value::U32(1), Value::U32(0))?;
            sum = self.emit.i_add(sum, carry)?;
        }
        self.write(sum)?;
        if carry_out != 7 {
            let carry = self.emit.i_less_than(sum, b, false)?;
            self.emit.set_pred(Pred::new(carry_out as u8)?, carry)?;
        }
        Ok(())
    }
    pub(super) fn ldc(&mut self) -> Result<()> {
        let i = self.instruction;
        let width = i.bits(73, 3);
        let mode = i.bits(78, 2);
        if width > 5
            || !matches!(mode, 0 | 2)
            || i.bits(64, 32) & !0xce00 != 0
            || i.bits(32, 32) & 0xf800003f != 0
        {
            return Err(unsupported("Native indexed constant load mode"));
        }
        let base = self.reg(i.bits(24, 8))?;
        let immediate = (i.bits(38, 16) as i16) as u32;
        let mut offset = self.emit.i_add(base, Value::U32(immediate))?;
        let mut bank = Value::U32(i.bits(54, 5));
        if mode == 2 {
            let segment = self.emit.shift_right_logical(offset, Value::U32(16))?;
            bank = self.emit.i_add(bank, segment)?;
            offset = self.emit.bitwise_and(offset, Value::U32(0xffff))?;
        }
        self.load_constant(bank, offset, width, false)
    }
    pub(super) fn load_constant(
        &mut self,
        bank: Value,
        offset: Value,
        width: u32,
        uniform: bool,
    ) -> Result<()> {
        let destination = self.instruction.bits(16, 8);
        let (bits, signed) = match width {
            0 => (8, false),
            1 => (8, true),
            2 => (16, false),
            3 => (16, true),
            4 => (32, false),
            5 => (64, false),
            _ => return Err(unsupported("Native constant load width")),
        };
        if bits == 64 && (destination & 1 != 0 || destination >= if uniform { 62 } else { 254 }) {
            return Err(unsupported("Unaligned native 64-bit constant destination"));
        }
        let value = self.emit.get_cbuf(bank, offset, bits, signed)?;
        if bits == 64 {
            for word in 0..2 {
                let value = self.emit.composite_extract(value, word)?;
                if uniform {
                    self.emit
                        .set_uniform_reg(destination + word as u32, value)?;
                } else {
                    self.emit
                        .set_reg(Reg((destination + word as u32) as u8), value)?;
                }
            }
            Ok(())
        } else {
            self.write_register(value, uniform)
        }
    }
    pub(super) fn p2r(&mut self) -> Result<()> {
        let i = self.instruction;
        let mask = i.bits(32, 32);
        if mask > 0x7f || i.bits(64, 32) & !0x3000 != 0 {
            return Err(unsupported("Native predicate-to-register mode"));
        }
        let offset = i.bits(76, 2) * 8;
        let base = self.reg(i.bits(24, 8))?;
        let mut value = self.emit.bitwise_and(base, Value::U32(!(mask << offset)))?;
        for bit in 0..7 {
            if mask & (1 << bit) != 0 {
                let condition = self.emit.get_pred(Pred::new(bit as u8)?, false)?;
                let selected =
                    self.emit
                        .select(condition, Value::U32(1 << (offset + bit)), Value::U32(0))?;
                value = self.emit.bitwise_or(value, selected)?;
            }
        }
        self.write(value)
    }

    pub(super) fn r2p(&mut self) -> Result<()> {
        let i = self.instruction;
        if i.bits(16, 8) != 0 || i.bits(32, 32) > 0x7f || i.bits(64, 32) & !0x3000 != 0 {
            return Err(unsupported("Native register-to-predicate mode"));
        }
        let value = self.reg(i.bits(24, 8))?;
        for bit in 0..7 {
            if i.bits(32, 32) & (1 << bit) != 0 {
                let selected = self
                    .emit
                    .bitwise_and(value, Value::U32(1 << (i.bits(76, 2) * 8 + bit)))?;
                let pred = self.emit.i_not_equal(selected, Value::U32(0))?;
                self.emit.set_pred(Pred::new(bit as u8)?, pred)?;
            }
        }
        Ok(())
    }
    pub(super) fn f2i(&mut self) -> Result<()> {
        let i = self.instruction;
        let uniform = i.opcode() == 0xd05;
        let mode = i.bits(64, 32) & !if uniform { 0x08000000 } else { 0 };
        // Integer widths have distinct encodings. Keep the conversion in U32
        // registers, with signed narrow results sign-extended after saturation.
        // https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/shader_recompiler/frontend/ampere/lower_to_maxwell.h#L2186-L2290
        let byte = mode & !0x10100 == 0x20e000;
        let short =
            mode & !0xc000 == 0x212800 || (i.opcode() == 0x305 && mode & !0x100 == 0x21e800);
        let word = mode & !0x1c100 == 0x203000;
        if i.bits(24, 8) != 0
            || !(byte || short || word)
            || (i.opcode() == 0x305 && i.bits(32, 32) & !0xc00000ff != 0)
            || (uniform && (i.bits(32, 32) > 63 || i.bits(91, 1) == 0))
        {
            return Err(unsupported("Native float-to-integer format"));
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
        let signed = i.bits(72, 1) != 0;
        // Convert only a representable value. In particular 2^31 and 2^32
        // round above the corresponding integer maxima in F32.
        let (min, max, saturation) = if byte && signed {
            (-128.0f32, 127.0f32, i8::MAX as u32)
        } else if byte {
            (0.0, 255.0, u8::MAX as u32)
        } else if short && signed {
            (-32768.0, 32767.0, i16::MAX as u32)
        } else if short {
            (0.0, 65535.0, u16::MAX as u32)
        } else if signed {
            (-2147483648.0f32, 2147483520.0f32, i32::MAX as u32)
        } else {
            (0.0, 4294967040.0, u32::MAX)
        };
        let finite = self.emit.fp_ordered(value, value)?;
        let value = self.emit.select(finite, value, Value::F32(0))?;
        let overflow = self.emit.fp_greater_than(
            value,
            Value::F32(max.to_bits()),
            FpControl::default(),
            true,
        )?;
        let bounded =
            self.emit
                .fp_clamp(value, Value::F32(min.to_bits()), Value::F32(max.to_bits()))?;
        let converted = self.emit.convert_f_to_i(32, signed, bounded)?;
        let value = self
            .emit
            .select(overflow, Value::U32(saturation), converted)?;
        self.write(value)
    }
}
