use super::{Result, Source, Value, Visitor, unsupported};
use crate::ir::{FlowTest, Pred};

impl Visitor<'_> {
    pub(super) fn compare(&mut self, a: Value, b: Value, op: u32, signed: bool) -> Result<Value> {
        match op {
            0 => Ok(Value::U1(false)),
            1 => self.emit.i_less_than(a, b, signed),
            2 => self.emit.i_equal(a, b),
            3 => self.emit.i_less_than_equal(a, b, signed),
            4 => self.emit.i_greater_than(a, b, signed),
            5 => self.emit.i_not_equal(a, b),
            6 => self.emit.i_greater_than_equal(a, b, signed),
            7 => Ok(Value::U1(true)),
            _ => Err(unsupported("Invalid integer comparison")),
        }
    }
    fn extended_compare(&mut self, a: Value, b: Value, op: u32, signed: bool) -> Result<Value> {
        let carry = self.emit.get_c_flag()?;
        let carry = self.emit.select(carry, Value::U32(1), Value::U32(0))?;
        let z = self.emit.get_z_flag()?;
        let not_b = self.emit.bitwise_not(b)?;
        let difference = self.emit.i_add(a, not_b)?;
        let difference = self.emit.i_add(difference, carry)?;
        let flip = if signed {
            Value::U1(false)
        } else {
            let sa = self.emit.i_less_than(a, Value::U32(0), true)?;
            let sb = self.emit.i_less_than(b, Value::U32(0), true)?;
            self.emit.logical_xor(sa, sb)?
        };
        match op {
            0 => Ok(Value::U1(false)),
            1 | 3 | 4 | 6 => {
                let (yes, no) = match op {
                    1 | 3 => (6, 1),
                    4 => (3, 4),
                    _ => (1, 6),
                };
                let yes = self.compare(difference, Value::U32(0), yes, true)?;
                let no = self.compare(difference, Value::U32(0), no, true)?;
                let result = self.emit.select(flip, yes, no)?;
                if op == 1 {
                    return Ok(result);
                }
                let equal = self.emit.i_equal(difference, Value::U32(0))?;
                let z = if op == 4 {
                    self.emit.logical_not(z)?
                } else {
                    z
                };
                let equal = self.emit.logical_and(equal, z)?;
                self.emit.logical_or(result, equal)
            }
            2 => {
                let equal = self.emit.i_equal(difference, Value::U32(0))?;
                self.emit.logical_and(equal, z)
            }
            5 => {
                let unequal = self.emit.i_not_equal(difference, Value::U32(0))?;
                let equal = self.emit.i_equal(difference, Value::U32(0))?;
                let not_z = self.emit.logical_not(z)?;
                let extra = self.emit.logical_and(equal, not_z)?;
                self.emit.logical_or(unequal, extra)
            }
            7 => Ok(Value::U1(true)),
            _ => Err(unsupported("Invalid extended integer comparison")),
        }
    }
    pub(super) fn combine(&mut self, a: Value, b: Value, op: u32) -> Result<Value> {
        match op {
            0 => self.emit.logical_and(a, b),
            1 => self.emit.logical_or(a, b),
            2 => self.emit.logical_xor(a, b),
            _ => Err(unsupported("Invalid boolean operation")),
        }
    }
    pub(super) fn predicate_result(&mut self, value: Value, op: u32) -> Result<Value> {
        match op {
            0 => Ok(Value::U1(false)),
            1 => Ok(Value::U1(true)),
            2 => self.emit.i_equal(value, Value::U32(0)),
            3 => self.emit.i_not_equal(value, Value::U32(0)),
            _ => Err(unsupported("Invalid result predicate operation")),
        }
    }
    pub(super) fn predicate_pair(&mut self, comparison: Value) -> Result<()> {
        let pred = self.predicate::<39, 42>()?;
        let op = self.field::<45, 2>();
        let a = self.combine(comparison, pred, op)?;
        let inverse = self.emit.logical_not(comparison)?;
        let b = self.combine(inverse, pred, op)?;
        self.set_predicate::<3>(a)?;
        self.set_predicate::<0>(b)
    }
    fn predicate_register(&mut self, predicate: Value) -> Result<()> {
        let bf = self.bit::<44>();
        let pass = if bf { 0x3f800000 } else { u32::MAX };
        let result = self
            .emit
            .select(predicate, Value::U32(pass), Value::U32(0))?;
        self.write(result)?;
        if self.bit::<47>() {
            let zero = self.emit.i_equal(result, Value::U32(0))?;
            self.emit.set_z_flag(zero)?;
            let sign = if bf {
                Value::U1(false)
            } else {
                self.emit.logical_not(zero)?
            };
            self.emit.set_s_flag(sign)?;
            self.clear_co()?;
        }
        Ok(())
    }
    pub(super) fn icmp(&mut self, b: Source, c: Source) -> Result<()> {
        let b = self.source(b)?;
        let c = self.source(c)?;
        let predicate = self.compare(c, Value::U32(0), self.field::<49, 3>(), self.bit::<48>())?;
        let a = self.a()?;
        let result = self.emit.select(predicate, a, b)?;
        self.write(result)
    }
    pub(super) fn iset(&mut self, source: Source, predicates: bool) -> Result<()> {
        let b = self.source(source)?;
        let a = self.a()?;
        let comparison = if self.bit::<43>() {
            self.extended_compare(a, b, self.field::<49, 3>(), self.bit::<48>())?
        } else {
            self.compare(a, b, self.field::<49, 3>(), self.bit::<48>())?
        };
        if predicates {
            self.predicate_pair(comparison)
        } else {
            let pred = self.predicate::<39, 42>()?;
            let result = self.combine(comparison, pred, self.field::<45, 2>())?;
            if self.bit::<47>() && self.bit::<43>() {
                return Err(unsupported("ISET.CC + X"));
            }
            self.predicate_register(result)
        }
    }
    pub(super) fn cset(&mut self, predicates: bool) -> Result<()> {
        let condition = self
            .emit
            .get_flow_test_result(FlowTest::try_from(self.field::<8, 5>() as u8)?)?;
        if predicates {
            self.predicate_pair(condition)
        } else {
            let pred = self.predicate::<39, 42>()?;
            let value = self.combine(condition, pred, self.field::<45, 2>())?;
            self.predicate_register(value)
        }
    }
    pub(super) fn pset(&mut self, predicates: bool) -> Result<()> {
        let a = self.predicate::<12, 15>()?;
        let b = self.predicate::<29, 32>()?;
        let c = self.predicate::<39, 42>()?;
        let first = self.combine(a, b, self.field::<24, 2>())?;
        if predicates {
            let inverse = self.emit.logical_not(a)?;
            let second = self.combine(inverse, b, self.field::<24, 2>())?;
            let first = self.combine(first, c, self.field::<45, 2>())?;
            let second = self.combine(second, c, self.field::<45, 2>())?;
            self.set_predicate::<3>(first)?;
            self.set_predicate::<0>(second)
        } else {
            let result = self.combine(first, c, self.field::<45, 2>())?;
            self.predicate_register(result)
        }
    }
    fn get_flag(&mut self, index: u32) -> Result<Value> {
        match index {
            0 => self.emit.get_z_flag(),
            1 => self.emit.get_s_flag(),
            2 => self.emit.get_c_flag(),
            3 => self.emit.get_o_flag(),
            _ => unreachable!(),
        }
    }
    fn set_flag(&mut self, index: u32, value: Value) -> Result<()> {
        match index {
            0 => self.emit.set_z_flag(value)?,
            1 => self.emit.set_s_flag(value)?,
            2 => self.emit.set_c_flag(value)?,
            3 => self.emit.set_o_flag(value)?,
            _ => unreachable!(),
        };
        Ok(())
    }
    pub(super) fn p2r(&mut self) -> Result<()> {
        let mask = self.instruction.imm20();
        let cc = self.bit::<40>();
        let offset = self.field::<41, 2>() * 8;
        let mut bits = Value::U32(0);
        for index in 0..if cc { 4 } else { 7 } {
            if mask & (1 << index) == 0 {
                continue;
            }
            let condition = if cc {
                self.get_flag(index)?
            } else {
                self.emit.get_pred(Pred::new(index as u8)?, false)?
            };
            let bit =
                self.emit
                    .select(condition, Value::U32(1 << (index + offset)), Value::U32(0))?;
            bits = self.emit.bitwise_or(bits, bit)?;
        }
        let base = self.a()?;
        let masked = self.emit.bitwise_and(base, Value::U32(!(mask << offset)))?;
        let result = self.emit.bitwise_or(masked, bits)?;
        self.write(result)
    }
    pub(super) fn r2p(&mut self, source: Source) -> Result<()> {
        let mask = self.source(source)?;
        let src = self.a()?;
        let cc = self.bit::<40>();
        let offset = self.field::<41, 2>() * 8;
        for index in 0..if cc { 4 } else { 7 } {
            let bit = self.emit.bit_field_extract(
                src,
                Value::U32(offset + index),
                Value::U32(1),
                false,
            )?;
            let zero = self.emit.get_zero_from_op(bit)?;
            let bit = self.emit.logical_not(zero)?;
            let mask =
                self.emit
                    .bit_field_extract(mask, Value::U32(index), Value::U32(1), false)?;
            let preserve = self.emit.get_zero_from_op(mask)?;
            let old = if cc {
                self.get_flag(index)?
            } else {
                self.emit.get_pred(Pred::new(index as u8)?, false)?
            };
            let result = self.emit.select(preserve, old, bit)?;
            if cc {
                self.set_flag(index, result)?;
            } else {
                self.emit.set_pred(Pred::new(index as u8)?, result)?;
            }
        }
        Ok(())
    }
    pub(super) fn lop(&mut self, source: Source, wide: bool) -> Result<()> {
        let mut b = self.source(source)?;
        let (op, x, cc, invert_a, invert_b) = if wide {
            (
                self.field::<53, 2>(),
                self.bit::<57>(),
                self.bit::<52>(),
                self.bit::<55>(),
                self.bit::<56>(),
            )
        } else {
            (
                self.field::<41, 2>(),
                self.bit::<43>(),
                self.bit::<47>(),
                self.bit::<39>(),
                self.bit::<40>(),
            )
        };
        if x {
            return Err(unsupported("LOP X"));
        }
        let mut a = self.a()?;
        if invert_a {
            a = self.emit.bitwise_not(a)?;
        }
        if invert_b {
            b = self.emit.bitwise_not(b)?;
        }
        let result = match op {
            0 => self.emit.bitwise_and(a, b)?,
            1 => self.emit.bitwise_or(a, b)?,
            2 => self.emit.bitwise_xor(a, b)?,
            3 => b,
            _ => unreachable!(),
        };
        if !wide {
            let pred = self.predicate_result(result, self.field::<44, 2>())?;
            self.set_predicate::<48>(pred)?;
        }
        if cc {
            self.flags_zs(result, op != 3)?;
            self.clear_co()?;
        }
        self.write(result)
    }
    pub(super) fn lop3(&mut self, source: Source, register_form: bool) -> Result<()> {
        if register_form && self.bit::<38>() {
            return Err(unsupported("LOP3 X"));
        }
        if self.bit::<47>() {
            return Err(unsupported("LOP3 CC"));
        }
        let b = self.source(source)?;
        let c = self.source(Source::Reg39)?;
        let a = self.a()?;
        let lut = if register_form {
            self.field::<28, 8>()
        } else {
            self.field::<48, 8>()
        };
        let result = super::lut::apply(&mut self.emit, a, b, c, lut as u8)?;
        self.write(result)?;
        if register_form {
            let pred = self.predicate_result(result, self.field::<36, 2>())?;
            self.set_predicate::<48>(pred)?;
        }
        Ok(())
    }
}
