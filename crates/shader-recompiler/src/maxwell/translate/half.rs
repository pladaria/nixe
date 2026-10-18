use super::{Opcode, Result, Source, Value, Visitor};
use crate::{
    CompileError,
    diagnostics::FailureCategory,
    ir::{FmzMode, FpControl, Type},
};

fn precision(mode: u32) -> FmzMode {
    match mode {
        0 => FmzMode::None,
        1 => FmzMode::FTZ,
        2 => FmzMode::FMZ,
        _ => FmzMode::DontCare,
    }
}

impl Visitor<'_> {
    fn half_immediate(&self) -> Value {
        Value::U32(
            (self.field::<20, 9>() << 6)
                | (self.field::<29, 1>() << 15)
                | (self.field::<30, 9>() << 22)
                | (self.field::<56, 1>() << 31),
        )
    }
    pub(super) fn half_extract(&mut self, value: Value, swizzle: u32) -> Result<[Value; 2]> {
        if swizzle == 1 {
            let scalar = self.emit.bit_cast(Type::F32, value)?;
            return Ok([scalar; 2]);
        }
        let vector = self.emit.unpack_float2x16(value)?;
        let first = self
            .emit
            .composite_extract(vector, usize::from(swizzle == 3))?;
        let second = if swizzle == 0 {
            self.emit.composite_extract(vector, 1)?
        } else {
            first
        };
        Ok([first, second])
    }
    fn promote_half_operands<const N: usize>(
        &mut self,
        operands: &mut [[Value; 2]; N],
    ) -> Result<bool> {
        let ty = self.emit.ir.value_type(operands[0][0])?;
        let promotion = operands.iter().any(|pair| {
            self.emit
                .ir
                .value_type(pair[0])
                .is_ok_and(|other| other != ty)
        });
        if promotion {
            for pair in operands {
                if self.emit.ir.value_type(pair[0])? == Type::F16 {
                    for value in pair {
                        *value = self.emit.fp_convert(32, *value, FpControl::default())?;
                    }
                }
            }
        }
        Ok(promotion)
    }
    fn half_merge(&mut self, values: [Value; 2], merge: u32) -> Result<()> {
        if merge == 0
            && (self.emit.ir.value_type(values[0])? != Type::F16
                || self.emit.ir.value_type(values[1])? != Type::F16)
        {
            return Err(CompileError::new(
                FailureCategory::UnsupportedReferenceCase,
                "Packed-half merge requires F16 operands after arithmetic",
            ));
        }
        let result = match merge {
            0 => {
                let vector = self.emit.composite_construct(&values)?;
                self.emit.pack_float2x16(vector)?
            }
            1 => {
                let value = self.emit.fp_convert(32, values[0], FpControl::default())?;
                self.emit.bit_cast(Type::U32, value)?
            }
            2 | 3 => {
                let old = self.source(Source::Reg0)?;
                let vector = self.emit.unpack_float2x16(old)?;
                let index = (merge - 2) as usize;
                let insert = self
                    .emit
                    .fp_convert(16, values[index], FpControl::default())?;
                let vector = self.emit.composite_insert(vector, insert, index)?;
                self.emit.pack_float2x16(vector)?
            }
            _ => unreachable!(),
        };
        self.write(result)
    }
    pub(super) fn half_arithmetic(&mut self, opcode: Opcode) -> Result<()> {
        let multiply = matches!(
            opcode,
            Opcode::HMUL2_reg | Opcode::HMUL2_cbuf | Opcode::HMUL2_imm | Opcode::HMUL2_32I
        );
        let wide = matches!(opcode, Opcode::HMUL2_32I | Opcode::HADD2_32I);
        let register = matches!(opcode, Opcode::HMUL2_reg | Opcode::HADD2_reg);
        let constant = matches!(opcode, Opcode::HMUL2_cbuf | Opcode::HADD2_cbuf);
        let b = if wide {
            self.source(Source::Imm32)?
        } else if register {
            self.source(Source::Reg20)?
        } else if constant {
            self.source(Source::Cbuf)?
        } else {
            self.half_immediate()
        };
        let (merge, swizzle_a, swizzle_b, sat, abs_a, neg_a, abs_b, neg_b, mode) = if wide {
            (
                0,
                self.field::<53, 2>(),
                0,
                self.bit::<52>(),
                false,
                !multiply && self.bit::<56>(),
                false,
                false,
                if multiply {
                    self.field::<55, 2>()
                } else {
                    self.field::<55, 1>()
                },
            )
        } else {
            (
                self.field::<49, 2>(),
                self.field::<47, 2>(),
                if register {
                    self.field::<28, 2>()
                } else if constant {
                    1
                } else {
                    0
                },
                if register {
                    self.bit::<32>()
                } else {
                    self.bit::<52>()
                },
                self.bit::<44>(),
                (!multiply || !register) && self.bit::<43>(),
                if register {
                    self.bit::<30>()
                } else {
                    constant && self.bit::<54>()
                },
                if register {
                    self.bit::<31>()
                } else {
                    !multiply && constant && self.bit::<56>()
                },
                if multiply {
                    self.field::<39, 2>()
                } else {
                    self.field::<39, 1>()
                },
            )
        };
        let a = self.a()?;
        let mut inputs = [
            self.half_extract(a, swizzle_a)?,
            self.half_extract(b, swizzle_b)?,
        ];
        let promotion = self.promote_half_operands(&mut inputs)?;
        let control = FpControl {
            no_contraction: true,
            fmz_mode: precision(mode),
            ..Default::default()
        };
        let mut result = [Value::None; 2];
        for index in 0..2 {
            let a = self.emit.fp_abs_neg(inputs[0][index], abs_a, neg_a)?;
            let b = self.emit.fp_abs_neg(inputs[1][index], abs_b, neg_b)?;
            let mut value = if multiply {
                self.emit.fp_mul(a, b, control)?
            } else {
                self.emit.fp_add(a, b, control)?
            };
            if multiply && mode == 2 && !sat {
                if self.emit.ir.value_type(a)? != Type::F32 {
                    return Err(CompileError::new(
                        FailureCategory::InvalidArgument,
                        "Half FMZ compares F16 operands with an F32 zero",
                    ));
                }
                let za = self
                    .emit
                    .fp_equal(a, Value::F32(0), FpControl::default(), true)?;
                let zb = self
                    .emit
                    .fp_equal(b, Value::F32(0), FpControl::default(), true)?;
                let zero = self.emit.logical_or(za, zb)?;
                value = self.emit.select(zero, Value::F32(0), value)?;
            }
            if sat {
                value = self.emit.fp_saturate(value)?;
            }
            if promotion {
                value = self.emit.fp_convert(16, value, FpControl::default())?;
            }
            result[index] = value;
        }
        self.half_merge(result, merge)
    }
    pub(super) fn hfma(&mut self, opcode: Opcode) -> Result<()> {
        let (b, c, swizzle_b, swizzle_c, neg_b, neg_c, sat, mode) = match opcode {
            Opcode::HFMA2_reg => (
                self.source(Source::Reg20)?,
                self.source(Source::Reg39)?,
                self.field::<28, 2>(),
                self.field::<35, 2>(),
                self.bit::<31>(),
                self.bit::<30>(),
                self.bit::<32>(),
                self.field::<37, 2>(),
            ),
            Opcode::HFMA2_rc => (
                self.source(Source::Reg39)?,
                self.source(Source::Cbuf)?,
                self.field::<53, 2>(),
                1,
                self.bit::<56>(),
                self.bit::<51>(),
                self.bit::<52>(),
                self.field::<57, 2>(),
            ),
            Opcode::HFMA2_cr => (
                self.source(Source::Cbuf)?,
                self.source(Source::Reg39)?,
                1,
                self.field::<53, 2>(),
                self.bit::<56>(),
                self.bit::<51>(),
                self.bit::<52>(),
                self.field::<57, 2>(),
            ),
            Opcode::HFMA2_imm => (
                self.half_immediate(),
                self.source(Source::Reg39)?,
                0,
                self.field::<53, 2>(),
                false,
                self.bit::<51>(),
                self.bit::<52>(),
                self.field::<57, 2>(),
            ),
            Opcode::HFMA2_32I => (
                self.source(Source::Imm32)?,
                self.source(Source::Reg0)?,
                0,
                0,
                false,
                self.bit::<52>(),
                false,
                self.field::<55, 2>(),
            ),
            _ => unreachable!(),
        };
        let wide = opcode == Opcode::HFMA2_32I;
        let swizzle_a = if wide {
            self.field::<53, 2>()
        } else {
            self.field::<47, 2>()
        };
        let merge = if wide { 0 } else { self.field::<49, 2>() };
        let a = self.a()?;
        let mut inputs = [
            self.half_extract(a, swizzle_a)?,
            self.half_extract(b, swizzle_b)?,
            self.half_extract(c, swizzle_c)?,
        ];
        let promotion = self.promote_half_operands(&mut inputs)?;
        let control = FpControl {
            no_contraction: true,
            fmz_mode: precision(mode),
            ..Default::default()
        };
        let mut result = [Value::None; 2];
        for index in 0..2 {
            let a = inputs[0][index];
            let b = self.emit.fp_abs_neg(inputs[1][index], false, neg_b)?;
            let c = self.emit.fp_abs_neg(inputs[2][index], false, neg_c)?;
            let mut value = self.emit.fp_fma(a, b, c, control)?;
            if mode == 2 && !sat {
                if self.emit.ir.value_type(a)? != Type::F32 {
                    return Err(CompileError::new(
                        FailureCategory::InvalidArgument,
                        "Half FMA FMZ compares F16 operands with an F32 zero",
                    ));
                }
                let za = self
                    .emit
                    .fp_equal(a, Value::F32(0), FpControl::default(), true)?;
                let zb = self
                    .emit
                    .fp_equal(b, Value::F32(0), FpControl::default(), true)?;
                let zero = self.emit.logical_or(za, zb)?;
                value = self.emit.select(zero, c, value)?;
            }
            if sat {
                value = self.emit.fp_saturate(value)?;
            }
            if promotion {
                value = self.emit.fp_convert(16, value, FpControl::default())?;
            }
            result[index] = value;
        }
        self.half_merge(result, merge)
    }
    pub(super) fn hset(&mut self, source: Source, predicates: bool) -> Result<()> {
        let register = matches!(source, Source::Reg20);
        let constant = matches!(source, Source::Cbuf);
        let b = if matches!(source, Source::Imm20) {
            self.half_immediate()
        } else {
            self.source(source)?
        };
        let abs_b = if register {
            self.bit::<30>()
        } else {
            predicates && constant && self.bit::<54>()
        };
        let neg_b = if register {
            self.bit::<31>()
        } else {
            constant && self.bit::<56>()
        };
        let swizzle_b = if register {
            self.field::<28, 2>()
        } else if constant {
            1
        } else {
            0
        };
        let compare = if register {
            self.field::<35, 4>()
        } else {
            self.field::<49, 4>()
        };
        let mode = if register {
            self.bit::<49>()
        } else {
            self.bit::<53>()
        };
        let flush = if predicates {
            self.bit::<6>()
        } else if register {
            self.bit::<50>()
        } else {
            self.bit::<54>()
        };
        let a = self.a()?;
        let mut inputs = [
            self.half_extract(a, self.field::<47, 2>())?,
            self.half_extract(b, swizzle_b)?,
        ];
        self.promote_half_operands(&mut inputs)?;
        let pred = self.predicate::<39, 42>()?;
        let control = FpControl {
            fmz_mode: super::float::ftz(flush),
            ..Default::default()
        };
        let mut results = [Value::None; 2];
        for index in 0..2 {
            let a = self
                .emit
                .fp_abs_neg(inputs[0][index], self.bit::<44>(), self.bit::<43>())?;
            let b = self.emit.fp_abs_neg(inputs[1][index], abs_b, neg_b)?;
            let comparison = self.float_compare(a, b, compare, control)?;
            results[index] = self.combine(comparison, pred, self.field::<45, 2>())?;
        }
        if predicates {
            if mode {
                let value = self.emit.logical_and(results[0], results[1])?;
                self.set_predicate::<3>(value)?;
                let inverse = self.emit.logical_not(value)?;
                self.set_predicate::<0>(inverse)
            } else {
                self.set_predicate::<3>(results[0])?;
                self.set_predicate::<0>(results[1])
            }
        } else {
            let mask = if mode { 0x3c00 } else { 0xffff };
            let low = self
                .emit
                .select(results[0], Value::U32(mask), Value::U32(0))?;
            let high = self
                .emit
                .select(results[1], Value::U32(mask << 16), Value::U32(0))?;
            let result = self.emit.bitwise_or(low, high)?;
            self.write(result)
        }
    }
}
