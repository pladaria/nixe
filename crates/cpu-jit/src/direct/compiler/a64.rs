use cranelift_codegen::ir::{self, InstBuilder, condcodes::IntCC, types};
use nixe_cpu::decode::a64::integer::{Instruction, Operands};
use nixe_cpu::semantics::immediate::{decode_a64_bit_masks, decode_a64_logical_immediate};
use nixe_memory::GuestVirtualAddress;

use super::{CraneliftTranslator, LazyFlags};
use crate::direct::DirectJitError;

impl CraneliftTranslator<'_, '_> {
    pub(super) fn emit_integer(
        &mut self,
        source: GuestVirtualAddress,
        instruction: Instruction,
        flags: &LazyFlags,
    ) -> Result<Option<LazyFlags>, DirectJitError> {
        match instruction {
            Instruction::MoveWide(fields) => self.emit_move_wide(fields),
            Instruction::AddSubImmediate(fields) => self.emit_add_sub_immediate(fields),
            Instruction::AddSubShifted(fields) => self.emit_add_sub_shifted(fields),
            Instruction::AddSubExtended(fields) => self.emit_add_sub_extended(fields),
            Instruction::AddSubCarry(fields) => self.emit_add_sub_carry(fields, flags),
            Instruction::LogicalImmediate(fields) => self.emit_logical_immediate(fields),
            Instruction::LogicalShifted(fields) => self.emit_logical_shifted(fields),
            Instruction::Bitfield(fields) => self.emit_bitfield(fields),
            Instruction::Extract(fields) => self.emit_extract(fields),
            Instruction::TwoSource(fields) => self.emit_two_source(fields),
            Instruction::ConditionalCompareRegister(fields)
            | Instruction::ConditionalCompareImmediate(fields) => {
                self.emit_conditional_compare(fields, flags)
            }
            Instruction::ConditionalSelect(fields) => self.emit_conditional_select(fields, flags),
            Instruction::ThreeSource(fields) => self.emit_three_source(fields),
            Instruction::OneSource(fields) => self.emit_one_source(fields),
            Instruction::Adr(fields) => self.emit_adr(source, fields, false),
            Instruction::Adrp(fields) => self.emit_adr(source, fields, true),
        }
    }

    // A64 integer data-processing semantics follow Arm DDI 0602 (2025-12),
    // Base Instructions. Decoded masks, shifts, and aliases are constants here:
    // https://developer.arm.com/documentation/ddi0602/2025-12/Base-Instructions/ADD--immediate---Add--immediate--
    // https://developer.arm.com/documentation/ddi0602/2025-12/Base-Instructions/ADD--shifted-register---Add--shifted-register--
    fn emit_move_wide(&mut self, fields: Operands) -> Result<Option<LazyFlags>, DirectJitError> {
        let ty = value_type(fields);
        let shift = u32::from(fields.opcode_2) * 16;
        let immediate = u64::from(fields.immediate_16) << shift;
        let opcode = u8::from(fields.subtract) * 2 + u8::from(fields.set_flags);
        let value = match opcode {
            0 => self
                .builder
                .ins()
                .iconst(ty, (!immediate & width_mask(fields)) as i64),
            2 => self.builder.ins().iconst(ty, immediate as i64),
            3 => {
                let old = self.read_integer(fields.rd, false, fields.width_64)?;
                let preserved = self
                    .builder
                    .ins()
                    .band_imm_u(old, !(0xffff_u64 << shift) as i64);
                self.builder.ins().bor_imm_u(preserved, immediate as i64)
            }
            _ => return Err(unsupported_subencoding("move-wide", fields)),
        };
        self.write_integer(fields.rd, false, value)?;
        Ok(None)
    }

    fn emit_add_sub_immediate(
        &mut self,
        fields: Operands,
    ) -> Result<Option<LazyFlags>, DirectJitError> {
        let lhs = self.read_integer(fields.rn, true, fields.width_64)?;
        let immediate = u64::from(fields.immediate_12) << if fields.n { 12 } else { 0 };
        let rhs = self
            .builder
            .ins()
            .iconst(value_type(fields), immediate as i64);
        self.emit_add_sub(fields, lhs, rhs, None, true)
    }

    fn emit_add_sub_shifted(
        &mut self,
        fields: Operands,
    ) -> Result<Option<LazyFlags>, DirectJitError> {
        let lhs = self.read_integer(fields.rn, false, fields.width_64)?;
        let rhs = self.read_integer(fields.rm, false, fields.width_64)?;
        let rhs = self.shift_immediate(rhs, fields.shift_kind, fields.shift_amount, false)?;
        self.emit_add_sub(fields, lhs, rhs, None, false)
    }

    fn emit_add_sub_extended(
        &mut self,
        fields: Operands,
    ) -> Result<Option<LazyFlags>, DirectJitError> {
        let lhs = self.read_integer(fields.rn, true, fields.width_64)?;
        let raw = self.read_integer(fields.rm, false, fields.width_64)?;
        let source_bits = match fields.extension & 3 {
            0 => 8,
            1 => 16,
            2 => 32,
            3 => 64,
            _ => unreachable!(),
        };
        if source_bits == 64 && !fields.width_64 {
            return Err(unsupported_subencoding("add-sub-extended", fields));
        }
        let source_type = match source_bits {
            8 => types::I8,
            16 => types::I16,
            32 => types::I32,
            64 => types::I64,
            _ => unreachable!(),
        };
        let value = if self.builder.func.dfg.value_type(raw) == source_type {
            raw
        } else {
            self.builder.ins().ireduce(source_type, raw)
        };
        let ty = value_type(fields);
        let extended = if source_type == ty {
            value
        } else if fields.extension & 4 == 0 {
            self.builder.ins().uextend(ty, value)
        } else {
            self.builder.ins().sextend(ty, value)
        };
        let rhs = if fields.small_shift == 0 {
            extended
        } else {
            self.builder
                .ins()
                .ishl_imm_u(extended, i64::from(fields.small_shift))
        };
        self.emit_add_sub(fields, lhs, rhs, None, true)
    }

    fn emit_add_sub_carry(
        &mut self,
        fields: Operands,
        flags: &LazyFlags,
    ) -> Result<Option<LazyFlags>, DirectJitError> {
        let lhs = self.read_integer(fields.rn, false, fields.width_64)?;
        let rhs = self.read_integer(fields.rm, false, fields.width_64)?;
        let carry = self.flag_c(flags);
        self.emit_add_sub(fields, lhs, rhs, Some(carry), false)
    }

    fn emit_add_sub(
        &mut self,
        fields: Operands,
        lhs: ir::Value,
        rhs: ir::Value,
        carry: Option<ir::Value>,
        register31_is_sp: bool,
    ) -> Result<Option<LazyFlags>, DirectJitError> {
        let width = value_width(fields);
        let (result, updated) = match (fields.subtract, carry) {
            (false, None) => {
                let result = self.builder.ins().iadd(lhs, rhs);
                let flags = LazyFlags::Add {
                    lhs,
                    rhs,
                    result,
                    width,
                };
                (result, flags)
            }
            (true, None) => {
                let result = self.builder.ins().isub(lhs, rhs);
                let flags = LazyFlags::Subtract {
                    lhs,
                    rhs,
                    result,
                    width,
                };
                (result, flags)
            }
            (false, Some(carry)) => {
                let carry_value = self.builder.ins().uextend(value_type(fields), carry);
                let partial = self.builder.ins().iadd(lhs, rhs);
                let result = self.builder.ins().iadd(partial, carry_value);
                let flags = LazyFlags::AddCarry {
                    lhs,
                    rhs,
                    carry,
                    result,
                    width,
                };
                (result, flags)
            }
            (true, Some(carry)) => {
                let carry_value = self.builder.ins().uextend(value_type(fields), carry);
                let one = self.builder.ins().iconst(value_type(fields), 1);
                let borrow = self.builder.ins().isub(one, carry_value);
                let partial = self.builder.ins().isub(lhs, rhs);
                let result = self.builder.ins().isub(partial, borrow);
                let flags = LazyFlags::SubtractCarry {
                    lhs,
                    rhs,
                    carry,
                    result,
                    width,
                };
                (result, flags)
            }
        };
        self.write_integer(fields.rd, register31_is_sp && !fields.set_flags, result)?;
        Ok(fields.set_flags.then_some(updated))
    }

    fn emit_logical_immediate(
        &mut self,
        fields: Operands,
    ) -> Result<Option<LazyFlags>, DirectJitError> {
        let immediate = decode_a64_logical_immediate(
            fields.n,
            fields.immediate_6_high,
            fields.shift_amount,
            value_width(fields),
        )
        .map_err(|_| unsupported_subencoding("logical-immediate", fields))?;
        let rhs = self
            .builder
            .ins()
            .iconst(value_type(fields), immediate as i64);
        self.emit_logical(fields, rhs)
    }

    fn emit_logical_shifted(
        &mut self,
        fields: Operands,
    ) -> Result<Option<LazyFlags>, DirectJitError> {
        let rhs = self.read_integer(fields.rm, false, fields.width_64)?;
        let rhs = self.shift_immediate(rhs, fields.shift_kind, fields.shift_amount, true)?;
        let rhs = if fields.invert {
            self.builder.ins().bnot(rhs)
        } else {
            rhs
        };
        self.emit_logical(fields, rhs)
    }

    fn emit_logical(
        &mut self,
        fields: Operands,
        rhs: ir::Value,
    ) -> Result<Option<LazyFlags>, DirectJitError> {
        let lhs = self.read_integer(fields.rn, false, fields.width_64)?;
        let opcode = u8::from(fields.subtract) * 2 + u8::from(fields.set_flags);
        let result = match opcode {
            0 | 3 => self.builder.ins().band(lhs, rhs),
            1 => self.builder.ins().bor(lhs, rhs),
            2 => self.builder.ins().bxor(lhs, rhs),
            _ => unreachable!(),
        };
        self.write_integer(fields.rd, false, result)?;
        Ok((opcode == 3).then_some(LazyFlags::Logical {
            result,
            width: value_width(fields),
        }))
    }

    // Bitfield masks are decoded once in Rust and become CLIF constants.
    // https://developer.arm.com/documentation/ddi0602/2025-12/Base-Instructions/BFM--Bitfield-move-
    fn emit_bitfield(&mut self, fields: Operands) -> Result<Option<LazyFlags>, DirectJitError> {
        let opcode = u8::from(fields.subtract) * 2 + u8::from(fields.set_flags);
        let masks = decode_a64_bit_masks(
            fields.n,
            fields.immediate_6_high,
            fields.shift_amount,
            value_width(fields),
            false,
        )
        .map_err(|_| unsupported_subencoding("bitfield", fields))?;
        let source = self.read_integer(fields.rn, false, fields.width_64)?;
        let destination = self.read_integer(fields.rd, false, fields.width_64)?;
        let rotated = self.rotate_right_immediate(source, fields.immediate_6_high);
        let bottom = if opcode == 1 {
            let preserved = self
                .builder
                .ins()
                .band_imm_u(destination, !masks.write_mask as i64);
            let inserted = self
                .builder
                .ins()
                .band_imm_u(rotated, masks.write_mask as i64);
            self.builder.ins().bor(preserved, inserted)
        } else {
            self.builder
                .ins()
                .band_imm_u(rotated, masks.write_mask as i64)
        };
        let top = match opcode {
            0 => {
                let shifted = self
                    .builder
                    .ins()
                    .ushr_imm_u(source, i64::from(fields.shift_amount));
                let bit = self.builder.ins().band_imm_u(shifted, 1);
                self.builder.ins().ineg(bit)
            }
            1 => destination,
            2 => self.builder.ins().iconst(value_type(fields), 0),
            _ => return Err(unsupported_subencoding("bitfield", fields)),
        };
        let upper = self.builder.ins().band_imm_u(top, !masks.test_mask as i64);
        let lower = self
            .builder
            .ins()
            .band_imm_u(bottom, masks.test_mask as i64);
        let result = self.builder.ins().bor(upper, lower);
        self.write_integer(fields.rd, false, result)?;
        Ok(None)
    }

    fn emit_extract(&mut self, fields: Operands) -> Result<Option<LazyFlags>, DirectJitError> {
        let low = self.read_integer(fields.rm, false, fields.width_64)?;
        let high = self.read_integer(fields.rn, false, fields.width_64)?;
        let lsb = fields.shift_amount;
        let result = if lsb == 0 {
            low
        } else {
            let low = self.builder.ins().ushr_imm_u(low, i64::from(lsb));
            let high = self
                .builder
                .ins()
                .ishl_imm_u(high, i64::from(value_width(fields) - lsb));
            self.builder.ins().bor(low, high)
        };
        self.write_integer(fields.rd, false, result)?;
        Ok(None)
    }

    fn emit_two_source(&mut self, fields: Operands) -> Result<Option<LazyFlags>, DirectJitError> {
        let lhs = self.read_integer(fields.rn, false, fields.width_64)?;
        let rhs = self.read_integer(fields.rm, false, fields.width_64)?;
        let result = match fields.shift_amount {
            2 => self.safe_divide(lhs, rhs, false),
            3 => self.safe_divide(lhs, rhs, true),
            8..=11 => {
                let amount = self
                    .builder
                    .ins()
                    .band_imm_u(rhs, i64::from(value_width(fields) - 1));
                match fields.shift_amount {
                    8 => self.builder.ins().ishl(lhs, amount),
                    9 => self.builder.ins().ushr(lhs, amount),
                    10 => self.builder.ins().sshr(lhs, amount),
                    11 => self.builder.ins().rotr(lhs, amount),
                    _ => unreachable!(),
                }
            }
            _ => {
                return Err(unsupported_subencoding(
                    "data-processing-two-source",
                    fields,
                ));
            }
        };
        self.write_integer(fields.rd, false, result)?;
        Ok(None)
    }

    // Conditional compare and select consume only the NZCV bits required by
    // their condition and keep the selected flag producer lazy.
    // https://developer.arm.com/documentation/ddi0602/2025-12/Base-Instructions/CCMP--register---Conditional-compare--register--
    // https://developer.arm.com/documentation/ddi0602/2025-12/Base-Instructions/CSEL--Conditional-select-
    fn emit_conditional_compare(
        &mut self,
        fields: Operands,
        flags: &LazyFlags,
    ) -> Result<Option<LazyFlags>, DirectJitError> {
        let predicate = self.emit_condition(
            nixe_cpu::semantics::conditions::Condition::from_encoding(fields.condition),
            flags,
        );
        let lhs = self.read_integer(fields.rn, false, fields.width_64)?;
        let rhs = if fields.immediate_form {
            self.builder
                .ins()
                .iconst(value_type(fields), i64::from(fields.rm))
        } else {
            self.read_integer(fields.rm, false, fields.width_64)?
        };
        let width = value_width(fields);
        let when_true = if fields.subtract {
            let result = self.builder.ins().isub(lhs, rhs);
            LazyFlags::Subtract {
                lhs,
                rhs,
                result,
                width,
            }
        } else {
            let result = self.builder.ins().iadd(lhs, rhs);
            LazyFlags::Add {
                lhs,
                rhs,
                result,
                width,
            }
        };
        Ok(Some(LazyFlags::Conditional {
            predicate,
            when_true: Box::new(when_true),
            when_false: u32::from(fields.nzcv),
        }))
    }

    fn emit_conditional_select(
        &mut self,
        fields: Operands,
        flags: &LazyFlags,
    ) -> Result<Option<LazyFlags>, DirectJitError> {
        let predicate = self.emit_condition(
            nixe_cpu::semantics::conditions::Condition::from_encoding(fields.condition),
            flags,
        );
        let when_true = self.read_integer(fields.rn, false, fields.width_64)?;
        let mut when_false = self.read_integer(fields.rm, false, fields.width_64)?;
        if fields.subtract {
            when_false = self.builder.ins().bnot(when_false);
        }
        if fields.bit10 {
            when_false = self.builder.ins().iadd_imm_s(when_false, 1);
        }
        let result = self.builder.ins().select(predicate, when_true, when_false);
        self.write_integer(fields.rd, false, result)?;
        Ok(None)
    }

    fn emit_three_source(&mut self, fields: Operands) -> Result<Option<LazyFlags>, DirectJitError> {
        let result = match fields.opcode_3 {
            0 => {
                let lhs = self.read_integer(fields.rn, false, fields.width_64)?;
                let rhs = self.read_integer(fields.rm, false, fields.width_64)?;
                let product = self.builder.ins().imul(lhs, rhs);
                let addend = self.read_integer(fields.ra, false, fields.width_64)?;
                if fields.subtract_product {
                    self.builder.ins().isub(addend, product)
                } else {
                    self.builder.ins().iadd(addend, product)
                }
            }
            1 => {
                let lhs = self.read_integer(fields.rn, false, false)?;
                let rhs = self.read_integer(fields.rm, false, false)?;
                let lhs = self.builder.ins().sextend(types::I64, lhs);
                let rhs = self.builder.ins().sextend(types::I64, rhs);
                let product = self.builder.ins().imul(lhs, rhs);
                let addend = self.read_integer(fields.ra, false, true)?;
                if fields.subtract_product {
                    self.builder.ins().isub(addend, product)
                } else {
                    self.builder.ins().iadd(addend, product)
                }
            }
            2 if fields.ra == 31 && !fields.subtract_product => {
                let lhs = self.read_integer(fields.rn, false, true)?;
                let rhs = self.read_integer(fields.rm, false, true)?;
                self.builder.ins().smulhi(lhs, rhs)
            }
            5 => {
                let lhs = self.read_integer(fields.rn, false, false)?;
                let rhs = self.read_integer(fields.rm, false, false)?;
                let lhs = self.builder.ins().uextend(types::I64, lhs);
                let rhs = self.builder.ins().uextend(types::I64, rhs);
                let product = self.builder.ins().imul(lhs, rhs);
                let addend = self.read_integer(fields.ra, false, true)?;
                if fields.subtract_product {
                    self.builder.ins().isub(addend, product)
                } else {
                    self.builder.ins().iadd(addend, product)
                }
            }
            6 if fields.ra == 31 && !fields.subtract_product => {
                let lhs = self.read_integer(fields.rn, false, true)?;
                let rhs = self.read_integer(fields.rm, false, true)?;
                self.builder.ins().umulhi(lhs, rhs)
            }
            _ => {
                return Err(unsupported_subencoding(
                    "data-processing-three-source",
                    fields,
                ));
            }
        };
        self.write_integer(fields.rd, false, result)?;
        Ok(None)
    }

    fn emit_one_source(&mut self, fields: Operands) -> Result<Option<LazyFlags>, DirectJitError> {
        let input = self.read_integer(fields.rn, false, fields.width_64)?;
        let result = match fields.shift_amount {
            0 => self.builder.ins().bitrev(input),
            1 => {
                let mask = if fields.width_64 {
                    0x00ff_00ff_00ff_00ff
                } else {
                    0x00ff_00ff
                };
                let low = self.builder.ins().band_imm_u(input, mask);
                let low = self.builder.ins().ishl_imm_u(low, 8);
                let high = self.builder.ins().ushr_imm_u(input, 8);
                let high = self.builder.ins().band_imm_u(high, mask);
                self.builder.ins().bor(low, high)
            }
            2 if fields.width_64 => {
                let swapped = self.builder.ins().bswap(input);
                self.builder.ins().rotl_imm_u(swapped, 32)
            }
            2 | 3 => self.builder.ins().bswap(input),
            4 => self.builder.ins().clz(input),
            5 => self.builder.ins().cls(input),
            _ => {
                return Err(unsupported_subencoding(
                    "data-processing-one-source",
                    fields,
                ));
            }
        };
        self.write_integer(fields.rd, false, result)?;
        Ok(None)
    }

    // ADR and ADRP targets are fully determined by the normalized immediate
    // and source PC, so no address arithmetic reaches native execution.
    // https://developer.arm.com/documentation/ddi0602/2025-12/Base-Instructions/ADR--Form-PC-relative-address-
    fn emit_adr(
        &mut self,
        source: GuestVirtualAddress,
        fields: Operands,
        page: bool,
    ) -> Result<Option<LazyFlags>, DirectJitError> {
        let displacement =
            nixe_cpu::semantics::a64::signed_immediate(u64::from(fields.adr_immediate), 21);
        let base = if page {
            source.get() & !0xfff
        } else {
            source.get()
        };
        let value = base.wrapping_add_signed(if page {
            displacement << 12
        } else {
            displacement
        });
        let value = self.builder.ins().iconst(types::I64, value as i64);
        self.write_register(fields.rd, value)?;
        Ok(None)
    }

    pub(super) fn integer_value(&mut self, value: ir::Value, width_64: bool) -> ir::Value {
        if width_64 {
            value
        } else {
            self.builder.ins().ireduce(types::I32, value)
        }
    }

    fn read_integer(
        &mut self,
        index: u8,
        register31_is_sp: bool,
        width_64: bool,
    ) -> Result<ir::Value, DirectJitError> {
        let value = self.read_register(index, register31_is_sp)?;
        Ok(self.integer_value(value, width_64))
    }

    pub(super) fn write_integer(
        &mut self,
        index: u8,
        register31_is_sp: bool,
        value: ir::Value,
    ) -> Result<(), DirectJitError> {
        let value = if self.builder.func.dfg.value_type(value) == types::I64 {
            value
        } else {
            self.builder.ins().uextend(types::I64, value)
        };
        self.write_register_with_sp(index, register31_is_sp, value)
    }

    fn shift_immediate(
        &mut self,
        value: ir::Value,
        kind: u8,
        amount: u8,
        allow_rotate: bool,
    ) -> Result<ir::Value, DirectJitError> {
        if amount == 0 {
            return Ok(value);
        }
        Ok(match kind {
            0 => self.builder.ins().ishl_imm_u(value, i64::from(amount)),
            1 => self.builder.ins().ushr_imm_u(value, i64::from(amount)),
            2 => self.builder.ins().sshr_imm_u(value, i64::from(amount)),
            3 if allow_rotate => {
                let bits = self.builder.func.dfg.value_type(value).bits();
                self.builder
                    .ins()
                    .rotl_imm_u(value, i64::from(bits) - i64::from(amount))
            }
            _ => return Err(DirectJitError::unsupported("invalid A64 shift kind")),
        })
    }

    fn rotate_right_immediate(&mut self, value: ir::Value, amount: u8) -> ir::Value {
        if amount == 0 {
            value
        } else {
            let bits = self.builder.func.dfg.value_type(value).bits();
            self.builder
                .ins()
                .rotl_imm_u(value, i64::from(bits) - i64::from(amount))
        }
    }

    fn safe_divide(&mut self, lhs: ir::Value, rhs: ir::Value, signed: bool) -> ir::Value {
        let ty = self.builder.func.dfg.value_type(lhs);
        let zero = self.builder.ins().iconst(ty, 0);
        let one = self.builder.ins().iconst(ty, 1);
        let divisor_zero = self.builder.ins().icmp_imm_s(IntCC::Equal, rhs, 0);
        let overflow = if signed {
            let minimum = self
                .builder
                .ins()
                .iconst(ty, (1_u64 << (ty.bits() - 1)) as i64);
            let lhs_minimum = self.builder.ins().icmp(IntCC::Equal, lhs, minimum);
            let rhs_negative_one = self.builder.ins().icmp_imm_s(IntCC::Equal, rhs, -1);
            self.builder.ins().band(lhs_minimum, rhs_negative_one)
        } else {
            self.builder.ins().iconst(types::I8, 0)
        };
        let exceptional = self.builder.ins().bor(divisor_zero, overflow);
        let safe_rhs = self.builder.ins().select(exceptional, one, rhs);
        let quotient = if signed {
            self.builder.ins().sdiv(lhs, safe_rhs)
        } else {
            self.builder.ins().udiv(lhs, safe_rhs)
        };
        let exceptional_result = if signed {
            self.builder.ins().select(overflow, lhs, zero)
        } else {
            zero
        };
        self.builder
            .ins()
            .select(exceptional, exceptional_result, quotient)
    }
}

const fn value_width(fields: Operands) -> u8 {
    if fields.width_64 { 64 } else { 32 }
}

const fn value_type(fields: Operands) -> ir::Type {
    if fields.width_64 {
        types::I64
    } else {
        types::I32
    }
}

const fn width_mask(fields: Operands) -> u64 {
    if fields.width_64 {
        u64::MAX
    } else {
        u32::MAX as u64
    }
}

fn unsupported_subencoding(name: &str, fields: Operands) -> DirectJitError {
    DirectJitError::unsupported(format!(
        "direct JIT does not implement this {name} subencoding: {fields:?}"
    ))
}
