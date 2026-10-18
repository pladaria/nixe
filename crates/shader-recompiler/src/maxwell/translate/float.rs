use super::{Operands, Result, Source, Value, Visitor, unsupported};
use crate::ir::{FmzMode, FpControl, FpRounding};

pub(super) fn rounding(mode: u32) -> FpRounding {
    match mode {
        0 => FpRounding::RN,
        1 => FpRounding::RM,
        2 => FpRounding::RP,
        3 => FpRounding::RZ,
        _ => unreachable!(),
    }
}
pub(super) fn fmz(mode: u32) -> Result<FmzMode> {
    match mode {
        0 => Ok(FmzMode::None),
        1 | 2 => Ok(FmzMode::FTZ),
        _ => Err(unsupported("Invalid FMZ mode")),
    }
}
pub(super) fn ftz(enabled: bool) -> FmzMode {
    if enabled { FmzMode::FTZ } else { FmzMode::None }
}

impl Visitor<'_> {
    pub(super) fn float_source(&mut self, source: Source, double: bool) -> Result<Value> {
        let mut operands = Operands {
            emit: &mut self.emit,
            instruction: self.instruction,
        };
        if double {
            match source {
                Source::Reg0 => operands.read_f64(self.instruction.reg0()),
                Source::Reg20 => operands.double_reg20(),
                Source::Reg39 => operands.double_reg39(),
                Source::Cbuf => operands.double_cbuf(),
                Source::Imm20 => Ok(operands.double_imm20()),
                _ => Err(unsupported("Invalid double operand form")),
            }
        } else {
            match source {
                Source::Reg0 => operands.read_f32(self.instruction.reg0()),
                Source::Reg20 => operands.float_reg20(),
                Source::Reg39 => operands.float_reg39(),
                Source::Cbuf => operands.float_cbuf(),
                Source::Imm20 => Ok(operands.float_imm20()),
                Source::Imm32 => Ok(operands.float_imm32()),
                _ => Err(unsupported("Invalid float operand form")),
            }
        }
    }
    pub(super) fn float_a(&mut self, double: bool) -> Result<Value> {
        let mut operands = Operands {
            emit: &mut self.emit,
            instruction: self.instruction,
        };
        if double {
            operands.read_f64(self.instruction.reg8())
        } else {
            operands.float_reg8()
        }
    }
    pub(super) fn write_float(&mut self, value: Value, double: bool) -> Result<()> {
        let mut operands = Operands {
            emit: &mut self.emit,
            instruction: self.instruction,
        };
        if double {
            operands.write_pair(self.instruction.reg0(), value)
        } else {
            operands.write_f32(self.instruction.reg0(), value)
        }
    }
    pub(super) fn float_compare(
        &mut self,
        a: Value,
        b: Value,
        op: u32,
        control: FpControl,
    ) -> Result<Value> {
        let ordered = !matches!(op, 9..=14);
        match op {
            0 => Ok(Value::U1(false)),
            1 | 9 => self.emit.fp_less_than(a, b, control, ordered),
            2 | 10 => self.emit.fp_equal(a, b, control, ordered),
            3 | 11 => self.emit.fp_less_than_equal(a, b, control, ordered),
            4 | 12 => self.emit.fp_greater_than(a, b, control, ordered),
            5 | 13 => self.emit.fp_not_equal(a, b, control, ordered),
            6 | 14 => self.emit.fp_greater_than_equal(a, b, control, ordered),
            7 => self.emit.fp_ordered(a, b),
            8 => self.emit.fp_unordered(a, b),
            15 => Ok(Value::U1(true)),
            _ => Err(unsupported("Invalid floating comparison")),
        }
    }
    pub(super) fn fadd(&mut self, source: Source, double: bool, wide: bool) -> Result<()> {
        let b = self.float_source(source, double)?;
        let (sat, cc, flush, round, abs_a, neg_a, abs_b, neg_b) = if wide {
            (
                false,
                self.bit::<52>(),
                self.bit::<55>(),
                0,
                self.bit::<54>(),
                self.bit::<56>(),
                self.bit::<57>(),
                self.bit::<53>(),
            )
        } else {
            (
                self.bit::<50>() && !double,
                self.bit::<47>(),
                self.bit::<44>() && !double,
                self.field::<39, 2>(),
                self.bit::<46>(),
                self.bit::<48>(),
                self.bit::<49>(),
                self.bit::<45>(),
            )
        };
        if cc {
            return Err(unsupported("Floating add CC"));
        }
        let a = self.float_a(double)?;
        let a = self.emit.fp_abs_neg(a, abs_a, neg_a)?;
        let b = self.emit.fp_abs_neg(b, abs_b, neg_b)?;
        let control = FpControl {
            no_contraction: true,
            rounding: rounding(round),
            fmz_mode: ftz(flush),
        };
        let mut result = self.emit.fp_add(a, b, control)?;
        if sat {
            result = self.emit.fp_saturate(result)?;
        }
        self.write_float(result, double)
    }
    pub(super) fn fmul(&mut self, source: Source, double: bool, wide: bool) -> Result<()> {
        let b = self.float_source(source, double)?;
        if double {
            if self.bit::<47>() {
                return Err(unsupported("DMUL CC"));
            }
            let a = self.float_a(true)?;
            let a = self.emit.fp_abs_neg(a, false, self.bit::<48>())?;
            let control = FpControl {
                no_contraction: true,
                rounding: rounding(self.field::<39, 2>()),
                fmz_mode: FmzMode::None,
            };
            let result = self.emit.fp_mul(a, b, control)?;
            return self.write_float(result, true);
        }
        let (cc, mode, sat, round, scale, neg_b) = if wide {
            (
                self.bit::<52>(),
                self.field::<53, 2>(),
                self.bit::<55>(),
                0,
                0,
                false,
            )
        } else {
            (
                self.bit::<47>(),
                self.field::<44, 2>(),
                self.bit::<50>(),
                self.field::<39, 2>(),
                self.field::<41, 3>(),
                self.bit::<48>(),
            )
        };
        if cc {
            return Err(unsupported("FMUL CC"));
        }
        let mut a = self.float_a(false)?;
        if scale != 0 {
            if mode != 1 || round != 0 {
                return Err(unsupported("FMUL scale requires FTZ and RN"));
            }
            let factor = match scale {
                1 => 0.5f32,
                2 => 0.25,
                3 => 0.125,
                4 => 8.0,
                5 => 4.0,
                6 => 2.0,
                _ => return Err(unsupported("Invalid FMUL scale")),
            };
            a = self
                .emit
                .fp_mul(a, Value::F32(factor.to_bits()), FpControl::default())?;
        }
        let b = self.emit.fp_abs_neg(b, false, neg_b)?;
        let control = FpControl {
            no_contraction: true,
            rounding: rounding(round),
            fmz_mode: fmz(mode)?,
        };
        let mut result = self.emit.fp_mul(a, b, control)?;
        if mode == 2 && !sat {
            let zero_a = self
                .emit
                .fp_equal(a, Value::F32(0), FpControl::default(), true)?;
            let zero_b = self
                .emit
                .fp_equal(b, Value::F32(0), FpControl::default(), true)?;
            let zero = self.emit.logical_or(zero_a, zero_b)?;
            result = self.emit.select(zero, Value::F32(0), result)?;
        }
        if sat {
            result = self.emit.fp_saturate(result)?;
        }
        self.write_float(result, false)
    }
    pub(super) fn ffma(&mut self, b: Source, c: Source, double: bool, wide: bool) -> Result<()> {
        let b = self.float_source(b, double)?;
        let c = self.float_source(c, double)?;
        let (cc, mode, round, sat, neg_a, neg_b, neg_c) = if double {
            (
                self.bit::<47>(),
                0,
                self.field::<50, 2>(),
                false,
                false,
                self.bit::<48>(),
                self.bit::<49>(),
            )
        } else if wide {
            (
                self.bit::<52>(),
                self.field::<53, 2>(),
                0,
                self.bit::<55>(),
                self.bit::<56>(),
                false,
                self.bit::<57>(),
            )
        } else {
            (
                self.bit::<47>(),
                self.field::<53, 2>(),
                self.field::<51, 2>(),
                self.bit::<50>(),
                false,
                self.bit::<48>(),
                self.bit::<49>(),
            )
        };
        if cc {
            return Err(unsupported("Floating FMA CC"));
        }
        let a = self.float_a(double)?;
        let a = self.emit.fp_abs_neg(a, false, neg_a)?;
        let b = self.emit.fp_abs_neg(b, false, neg_b)?;
        let c = self.emit.fp_abs_neg(c, false, neg_c)?;
        let control = FpControl {
            no_contraction: true,
            rounding: rounding(round),
            fmz_mode: fmz(mode)?,
        };
        let mut result = self.emit.fp_fma(a, b, c, control)?;
        if mode == 2 && !sat {
            let zero_a = self
                .emit
                .fp_equal(a, Value::F32(0), FpControl::default(), true)?;
            let zero_b = self
                .emit
                .fp_equal(b, Value::F32(0), FpControl::default(), true)?;
            let zero = self.emit.logical_or(zero_a, zero_b)?;
            result = self.emit.select(zero, c, result)?;
        }
        if sat {
            result = self.emit.fp_saturate(result)?;
        }
        self.write_float(result, double)
    }
    pub(super) fn fmnmx(&mut self, source: Source, double: bool) -> Result<()> {
        let b = self.float_source(source, double)?;
        if self.bit::<47>() {
            return Err(unsupported("Floating min/max CC"));
        }
        let predicate = self.predicate::<39, 42>()?;
        let a = self.float_a(double)?;
        let a = self
            .emit
            .fp_abs_neg(a, self.bit::<46>(), self.bit::<48>())?;
        let b = self
            .emit
            .fp_abs_neg(b, self.bit::<49>(), self.bit::<45>())?;
        let control = if double {
            FpControl::default()
        } else {
            FpControl {
                fmz_mode: ftz(self.bit::<44>()),
                ..Default::default()
            }
        };
        let max = self.emit.fp_max(a, b, control)?;
        let min = self.emit.fp_min(a, b, control)?;
        let result = self.emit.select(predicate, min, max)?;
        self.write_float(result, double)
    }
    pub(super) fn fcmp(&mut self, b: Source, c: Source) -> Result<()> {
        let b = if matches!(b, Source::Imm20) {
            Value::U32(self.instruction.float_imm20_bits())
        } else {
            self.source(b)?
        };
        let c = self.float_source(c, false)?;
        let control = FpControl {
            fmz_mode: ftz(self.bit::<47>()),
            ..Default::default()
        };
        let condition = self.float_compare(c, Value::F32(0), self.field::<48, 4>(), control)?;
        let a = self.a()?;
        let result = self.emit.select(condition, a, b)?;
        self.write(result)
    }
    pub(super) fn fset(&mut self, source: Source, double: bool, predicates: bool) -> Result<()> {
        let b = self.float_source(source, double)?;
        let a = self.float_a(double)?;
        let (abs_a, neg_b, flush) = if predicates {
            (self.bit::<7>(), self.bit::<6>(), self.bit::<47>())
        } else {
            (self.bit::<54>(), self.bit::<53>(), self.bit::<55>())
        };
        let a = self.emit.fp_abs_neg(a, abs_a, self.bit::<43>())?;
        let b = self.emit.fp_abs_neg(b, self.bit::<44>(), neg_b)?;
        let control = if double {
            FpControl::default()
        } else {
            FpControl {
                fmz_mode: ftz(flush),
                ..Default::default()
            }
        };
        let condition = self.float_compare(a, b, self.field::<48, 4>(), control)?;
        if predicates {
            self.predicate_pair(condition)
        } else {
            let pred = self.predicate::<39, 42>()?;
            let condition = self.combine(condition, pred, self.field::<45, 2>())?;
            let pass = if self.bit::<52>() {
                0x3f800000
            } else {
                u32::MAX
            };
            let result = self
                .emit
                .select(condition, Value::U32(pass), Value::U32(0))?;
            self.write(result)?;
            if self.bit::<47>() {
                self.flags_zs(result, false)?;
                self.clear_co()?;
            }
            Ok(())
        }
    }
    pub(super) fn mufu(&mut self) -> Result<()> {
        let a = self.float_a(false)?;
        let a = self
            .emit
            .fp_abs_neg(a, self.bit::<46>(), self.bit::<48>())?;
        let mut value = match self.field::<20, 4>() {
            0 => self.emit.fp_cos(a)?,
            1 => self.emit.fp_sin(a)?,
            2 => self.emit.fp_exp2(a)?,
            3 => self.emit.fp_log2(a)?,
            4 => self.emit.fp_recip(a)?,
            5 => self.emit.fp_recip_sqrt(a)?,
            6 => return Err(unsupported("MUFU.RCP64H")),
            7 => return Err(unsupported("MUFU.RSQ64H")),
            8 => self.emit.fp_sqrt(a)?,
            _ => return Err(unsupported("Invalid MUFU operation")),
        };
        if self.bit::<50>() {
            value = self.emit.fp_saturate(value)?;
        }
        self.write_float(value, false)
    }
    pub(super) fn rro(&mut self, source: Source) -> Result<()> {
        let value = self.float_source(source, false)?;
        // Both reduction modes retain the operand with its sign modifiers.
        let result = self
            .emit
            .fp_abs_neg(value, self.bit::<49>(), self.bit::<45>())?;
        self.write_float(result, false)
    }
    pub(super) fn fswzadd(&mut self) -> Result<()> {
        let a = self.float_a(false)?;
        let b = self.float_source(Source::Reg20, false)?;
        let control = FpControl {
            no_contraction: false,
            rounding: rounding(self.field::<39, 2>()),
            fmz_mode: ftz(self.bit::<44>()),
        };
        // NDV shares the ordinary swizzle behavior.
        let result = self
            .emit
            .f_swizzle_add(a, b, Value::U32(self.field::<28, 8>()), control)?;
        self.write_float(result, false)?;
        if self.bit::<47>() {
            return Err(unsupported("FSWZADD CC"));
        }
        Ok(())
    }
}
