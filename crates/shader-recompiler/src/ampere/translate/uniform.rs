//! Native uniform registers and predicates have their own SSA namespace. A
//! lane-to-uniform transfer is an explicit subgroup broadcast, not a register
//! alias or a reserved range of ordinary registers.
//! Encoding reference: https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/shader_recompiler/frontend/ampere/lower_to_maxwell.h
use super::*;

impl Visitor<'_> {
    pub(super) fn uniform_wide_multiply(&mut self) -> Result<()> {
        let i = self.instruction;
        let destination = i.bits(16, 8);
        let addend = i.bits(64, 8);
        if i.bits(32, 32) > 63
            || i.bits(64, 32) & !0xff != 0x0f8e0000
            || destination >= 62
            || destination & 1 != 0
            || (addend != 63 && (addend >= 62 || addend & 1 != 0))
        {
            return Err(unsupported("Native uniform wide multiply mode"));
        }
        let a = self.ureg(i.bits(24, 8))?;
        let b = self.ureg(i.bits(32, 8))?;
        let product = self.emit.i_mul_extended(a, b, false)?;
        let product_low = self.emit.composite_extract(product, 0)?;
        let product_high = self.emit.composite_extract(product, 1)?;
        let addend_low = self.ureg(addend)?;
        let addend_high = self.ureg(if addend == 63 { 63 } else { addend + 1 })?;
        let low = self.emit.i_add(product_low, addend_low)?;
        let carry = self.emit.get_carry_from_op(low)?;
        let carry = self.emit.select(carry, Value::U32(1), Value::U32(0))?;
        let high = self.emit.i_add(product_high, addend_high)?;
        let high = self.emit.i_add(high, carry)?;
        self.emit.set_uniform_reg(destination, low)?;
        self.emit.set_uniform_reg(destination + 1, high)
    }

    pub(super) fn vote(&mut self) -> Result<()> {
        let i = self.instruction;
        let uniform = i.opcode() == 0x886;
        if (uniform && i.bits(12, 4) != 7)
            || i.bits(24, 8) != 0
            || i.bits(32, 32) != 0
            || i.bits(64, 32) & !0x078e0300 != 0
            || i.bits(72, 2) > 2
        {
            return Err(unsupported("Native uniform vote mode"));
        }
        let input = self
            .emit
            .get_pred(Pred::new(i.bits(87, 3) as u8)?, i.bits(90, 1) != 0)?;
        let value = match i.bits(72, 2) {
            0 => self.emit.vote_all(input)?,
            1 => self.emit.vote_any(input)?,
            _ => self.emit.vote_equal(input)?,
        };
        let ballot = self.emit.subgroup_ballot(input)?;
        self.emit
            .set_pred(Self::predicate(i.bits(81, 3), uniform)?, value)?;
        self.write_register(ballot, uniform)
    }
    pub(super) fn uldc(&mut self) -> Result<()> {
        let i = self.instruction;
        let indexed = i.opcode() == 0xabb;
        let width = i.bits(73, 3);
        let immediate = i.bits(38, 16);
        let alignment = if width < 4 { 1 << (width / 2) } else { 4 };
        if width > 5
            || i.bits(16, 8) >= 63
            || i.bits(62, 2) != 0
            || i.bits(64, 32) & !0xe00 != if indexed { 0x08000000 } else { 0 }
            || (!indexed
                && (!immediate.is_multiple_of(alignment) || (width < 4 && immediate >= 0x8000)))
        {
            return Err(unsupported("Native uniform constant load mode"));
        }
        let offset = if indexed {
            let base = self.ureg(i.bits(24, 8))?;
            self.emit
                .i_add(base, Value::U32((immediate as i16) as u32))?
        } else {
            Value::U32(if width == 4 {
                immediate
            } else {
                (immediate as i16) as u32
            })
        };
        self.load_constant(Value::U32(i.bits(54, 5)), offset, width, true)
    }
    pub(super) fn ureg(&mut self, index: u32) -> Result<Value> {
        self.emit.get_uniform_reg(index)
    }
    pub(super) fn register(&mut self, index: u32, uniform: bool) -> Result<Value> {
        if uniform {
            self.ureg(index)
        } else {
            self.reg(index)
        }
    }
    pub(super) fn write_register(&mut self, value: Value, uniform: bool) -> Result<()> {
        if uniform {
            self.emit
                .set_uniform_reg(self.instruction.bits(16, 8), value)
        } else {
            self.write(value)
        }
    }
    pub(super) fn predicate(index: u32, uniform: bool) -> Result<Pred> {
        if uniform {
            Pred::uniform(index as u8)
        } else {
            Pred::new(index as u8)
        }
    }
    pub(super) fn umov(&mut self) -> Result<()> {
        let i = self.instruction;
        if i.bits(24, 8) != 0 || i.bits(64, 32) != if i.opcode() == 0xc82 { 0x08000000 } else { 0 }
        {
            return Err(unsupported("Native uniform move mode"));
        }
        let value = if i.opcode() == 0xc82 {
            self.ureg(i.bits(32, 32))?
        } else {
            Value::U32(i.bits(32, 32))
        };
        self.write_register(value, true)
    }
    pub(super) fn uisetp(&mut self) -> Result<()> {
        let i = self.instruction;
        if i.bits(64, 8) != 0x70
            || i.bits(72, 1) != 0
            || i.bits(79, 2) != 0
            || i.bits(84, 3) != 7
            || i.bits(91, 1) != 1
        {
            return Err(unsupported("Native uniform comparison mode"));
        }
        let a = self.ureg(i.bits(24, 8))?;
        let b = if i.opcode() == 0x88c {
            Value::U32(i.bits(32, 32))
        } else {
            self.ureg(i.bits(32, 32))?
        };
        let comparison = crate::frontend::integer_compare(
            &mut self.emit,
            a,
            b,
            i.bits(76, 3),
            i.bits(73, 1) != 0,
        )?;
        let input = self
            .emit
            .get_pred(Pred::uniform(i.bits(87, 3) as u8)?, i.bits(90, 1) != 0)?;
        let value = self.combine(comparison, input, i.bits(74, 2))?;
        self.emit
            .set_pred(Pred::uniform(i.bits(81, 3) as u8)?, value)
    }
    pub(super) fn r2ur(&mut self) -> Result<()> {
        let i = self.instruction;
        if i.bits(32, 32) != 0 || i.bits(64, 32) != 0x000e0000 {
            return Err(unsupported("Native register-to-uniform mode"));
        }
        let predicate = self
            .emit
            .get_pred(Pred::new(i.bits(12, 3) as u8)?, i.bits(15, 1) != 0)?;
        let mask = self.emit.subgroup_ballot(predicate)?;
        let negated = self.emit.i_neg(mask)?;
        let first_bit = self.emit.bitwise_and(mask, negated)?;
        let lane = self.emit.find_u_msb(first_bit)?;
        let source = self.reg(i.bits(24, 8))?;
        let mut value = self
            .emit
            .shuffle_index(source, lane, Value::U32(31), Value::U32(0))?;
        if i.bits(12, 4) != 7 {
            let any = self.emit.i_not_equal(mask, Value::U32(0))?;
            let old = self.ureg(i.bits(16, 8))?;
            value = self.emit.select(any, value, old)?;
        }
        self.write_register(value, true)
    }
    pub(super) fn plop3(&mut self) -> Result<()> {
        let i = self.instruction;
        let outputs = [i.bits(81, 3), i.bits(84, 3)];
        if i.bits(24, 8) != 0
            || i.bits(32, 32) != 0
            || i.bits(91, 5) != 0
            || (outputs[1] == 7 && i.bits(16, 8) != 0)
            || (outputs[0] != 7 && outputs[0] == outputs[1])
        {
            return Err(unsupported("Native predicate LUT mode"));
        }
        let mut inputs = [Value::None; 3];
        for (value, (bit, uniform)) in
            inputs
                .iter_mut()
                .zip([(87, false), (77, false), (68, i.bits(67, 1) != 0)])
        {
            *value = self.emit.get_pred(
                Self::predicate(i.bits(bit, 3), uniform)?,
                i.bits(bit + 3, 1) != 0,
            )?;
        }
        // Capture both results before updating either predicate (sources may
        // alias either destination).
        let tables = [i.bits(64, 3) | i.bits(72, 5) << 3, i.bits(16, 8)];
        let mut results = [Value::None; 2];
        for ((result, table), output) in results.iter_mut().zip(tables).zip(outputs) {
            if output == 7 {
                continue;
            }
            let mut terms =
                std::array::from_fn::<_, 8, _>(|bit| Value::U1(table & (1 << bit) != 0));
            for (level, condition) in inputs.into_iter().rev().enumerate() {
                for j in 0..(4 >> level) {
                    terms[j] = self
                        .emit
                        .select(condition, terms[j * 2 + 1], terms[j * 2])?;
                }
            }
            *result = terms[0];
        }
        for (output, result) in outputs.into_iter().zip(results) {
            if output != 7 {
                self.emit.set_pred(Pred::new(output as u8)?, result)?;
            }
        }
        Ok(())
    }
}
