use super::float::{ftz, rounding};
use super::{Operands, Result, Source, Value, Visitor, unsupported};
use crate::ir::{FmzMode, FpControl};

fn float_width(format: u32) -> Result<u32> {
    match format {
        1 => Ok(16),
        2 => Ok(32),
        3 => Ok(64),
        _ => Err(unsupported("Invalid floating-point format")),
    }
}

impl Visitor<'_> {
    fn write_numeric_float(&mut self, value: Value, width: u32) -> Result<()> {
        if width == 16 {
            let zero = self
                .emit
                .fp_convert(16, Value::F32(0), FpControl::default())?;
            let pair = self.emit.composite_construct(&[value, zero])?;
            let result = self.emit.pack_float2x16(pair)?;
            self.write(result)
        } else {
            self.write_float(value, width == 64)
        }
    }
    fn float_round(&mut self, value: Value, mode: u32, control: FpControl) -> Result<Value> {
        match mode {
            0 => self.emit.fp_round_even(value, control),
            1 => self.emit.fp_floor(value, control),
            2 => self.emit.fp_ceil(value, control),
            3 => self.emit.fp_trunc(value, control),
            _ => unreachable!(),
        }
    }
    pub(super) fn f2f(&mut self, source: Source) -> Result<()> {
        let src_width = float_width(self.field::<10, 2>())?;
        let dst_width = float_width(self.field::<8, 2>())?;
        let input = if src_width == 16 {
            let immediate = matches!(source, Source::Imm20);
            let raw = if immediate {
                let bits = self.field::<20, 16>();
                Value::U32(bits | bits << 16)
            } else {
                self.source(source)?
            };
            let vector = self.emit.unpack_float2x16(raw)?;
            let index = usize::from(if immediate {
                !self.bit::<41>()
            } else {
                self.bit::<41>()
            });
            let value = self.emit.composite_extract(vector, index)?;
            if immediate && self.bit::<56>() {
                return Err(unsupported("F2F negation bit on F16 immediate"));
            }
            value
        } else {
            self.float_source(source, src_width == 64)?
        };
        if self.bit::<47>() {
            return Err(unsupported("F2F CC"));
        }
        let mut value = self
            .emit
            .fp_abs_neg(input, self.bit::<49>(), self.bit::<45>())?;
        let any_double = src_width == 64 || dst_width == 64;
        let mut control = FpControl {
            fmz_mode: ftz(self.bit::<44>() && !any_double),
            ..Default::default()
        };
        if src_width != dst_width {
            control.rounding = rounding(self.field::<39, 2>());
            value = self.emit.fp_convert(dst_width, value, control)?;
        } else {
            match self.field::<39, 4>() & 0xb {
                0 | 3 => {
                    let zero = match src_width {
                        16 => self
                            .emit
                            .fp_convert(16, Value::F32(0), FpControl::default())?,
                        32 => Value::F32(0),
                        64 => Value::F64(0),
                        _ => unreachable!(),
                    };
                    value = self.emit.fp_add(value, zero, control)?;
                }
                mode @ 8..=11 => value = self.float_round(value, mode - 8, control)?,
                _ => return Err(unsupported("F2F rounding operation")),
            }
        }
        if self.bit::<50>() && !any_double {
            value = self.emit.fp_saturate(value)?;
        }
        self.write_numeric_float(value, dst_width)
    }
    pub(super) fn f2i(&mut self, source: Source) -> Result<()> {
        let src_width = float_width(self.field::<10, 2>())?;
        let dst_width = float_width(self.field::<8, 2>())?;
        let input = if src_width == 16 {
            let raw = self.source(source)?;
            let pair = self.emit.unpack_float2x16(raw)?;
            self.emit
                .composite_extract(pair, usize::from(self.bit::<41>()))?
        } else if src_width == 32 {
            self.float_source(source, false)?
        } else {
            let (low, high) = if matches!(source, Source::Cbuf) {
                let offset = super::super::bits::extract_signed::<20, 14>(self.instruction.0);
                let bank = self.field::<34, 5>();
                if bank >= 18 {
                    return Err(unsupported("F2I constant buffer binding out of bounds"));
                }
                if offset < 0 {
                    return Err(unsupported("F2I constant buffer offset out of bounds"));
                }
                if offset % 2 != 0 {
                    return Err(unsupported("Unaligned F2I F64 constant buffer offset"));
                }
                let high = self.emit.get_cbuf(
                    Value::U32(bank),
                    Value::U32(offset as u32 * 4 + 4),
                    32,
                    false,
                )?;
                (Value::U32(0), high)
            } else {
                let reg = self.instruction.reg20();
                (self.emit.get_reg(reg)?, self.emit.get_reg(reg.offset(1)?)?)
            };
            let pair = self.emit.composite_construct(&[low, high])?;
            self.emit.pack_double2x32(pair)?
        };
        let fmz_mode = if src_width == 32 && dst_width != 64 {
            ftz(self.bit::<44>())
        } else {
            FmzMode::DontCare
        };
        let control = FpControl {
            no_contraction: true,
            fmz_mode,
            ..Default::default()
        };
        let value = self
            .emit
            .fp_abs_neg(input, self.bit::<45>(), self.bit::<49>())?;
        let rounded = self.float_round(value, self.field::<39, 2>(), control)?;
        let signed = self.bit::<12>();
        let (min, max) = match (dst_width, signed) {
            (16, true) => (i16::MIN as f64, i16::MAX as f64),
            (16, false) => (0.0, u16::MAX as f64),
            (32, true) => (i32::MIN as f64, i32::MAX as f64),
            (32, false) => (0.0, u32::MAX as f64),
            (64, true) => (i64::MIN as f64, i64::MAX as f64),
            (64, false) => (0.0, u64::MAX as f64),
            _ => unreachable!(),
        };
        let (min, max) = match src_width {
            16 => (
                self.emit.fp_convert(
                    16,
                    Value::F32((min as f32).to_bits()),
                    FpControl::default(),
                )?,
                self.emit.fp_convert(
                    16,
                    Value::F32((max as f32).to_bits()),
                    FpControl::default(),
                )?,
            ),
            32 => (
                Value::F32((min as f32).to_bits()),
                Value::F32((max as f32).to_bits()),
            ),
            64 => (Value::F64(min.to_bits()), Value::F64(max.to_bits())),
            _ => unreachable!(),
        };
        let clamped = self.emit.fp_clamp(rounded, min, max)?;
        let bits = dst_width.max(32);
        let mut result = self.emit.convert_f_to_i(bits, signed, clamped)?;
        let special = (src_width == 64) != (dst_width == 64);
        let special_value = if special && dst_width == 32 {
            Some(Value::U32(0x80000000))
        } else if special && dst_width == 64 {
            Some(Value::U64(0x8000000000000000))
        } else if signed {
            Some(if bits == 64 {
                Value::U64(0)
            } else {
                Value::U32(0)
            })
        } else {
            None
        };
        if let Some(special) = special_value {
            let nan = self.emit.fp_is_nan(value)?;
            result = self.emit.select(nan, special, result)?;
        }
        if bits == 64 {
            Operands {
                emit: &mut self.emit,
                instruction: self.instruction,
            }
            .write_pair(self.instruction.reg0(), result)?;
        } else {
            self.write(result)?;
        }
        if self.bit::<47>() {
            return Err(unsupported("F2I CC"));
        }
        Ok(())
    }
    pub(super) fn i2f(&mut self, source: Source) -> Result<()> {
        let src_width = 8 << self.field::<10, 2>();
        let dst_width = float_width(self.field::<8, 2>())?;
        let mut src = if src_width == 64 {
            match source {
                Source::Reg20 => {
                    let reg = self.instruction.reg20();
                    let low = self.emit.get_reg(reg)?;
                    let high = self.emit.get_reg(reg.offset(1)?)?;
                    let pair = self.emit.composite_construct(&[low, high])?;
                    self.emit.pack_uint2x32(pair)?
                }
                Source::Cbuf => Operands {
                    emit: &mut self.emit,
                    instruction: self.instruction,
                }
                .packed_cbuf()?,
                Source::Imm20 => Value::U64(self.instruction.packed_imm20()),
                _ => unreachable!(),
            }
        } else {
            self.source(source)?
        };
        if self.bit::<47>() {
            return Err(unsupported("I2F CC"));
        }
        let selector = self.field::<41, 2>();
        let signed = self.bit::<13>();
        let absolute = self.bit::<49>();
        if src_width < 32 {
            if src_width == 16 && matches!(selector, 1 | 3) {
                return Err(unsupported("I2F invalid U16 selector"));
            }
            src = self.emit.bit_field_extract(
                src,
                Value::U32(selector * 8),
                Value::U32(src_width),
                signed,
            )?;
            if absolute {
                let mask = self
                    .emit
                    .shift_right_arithmetic(src, Value::U32(src_width - 1))?;
                let sum = self.emit.i_add(src, mask)?;
                let abs = self.emit.bitwise_xor(sum, mask)?;
                let least = self
                    .emit
                    .i_equal(src, Value::U32((1u32 << (src_width - 1)).wrapping_neg()))?;
                src = self.emit.select(least, src, abs)?;
            }
        } else {
            if selector != 0 {
                return Err(unsupported("I2F unexpected selector"));
            }
            if absolute && signed {
                src = self.emit.i_abs(src)?;
            }
        }
        let control = FpControl {
            rounding: rounding(self.field::<39, 2>()),
            ..Default::default()
        };
        let mut value =
            self.emit
                .convert_i_to_f(dst_width, src_width.max(32), signed, src, control)?;
        if self.bit::<45>() {
            let negative = self.emit.fp_neg(value)?;
            if absolute || !signed {
                value = negative;
            } else {
                let min = if src_width == 64 {
                    Value::U64(1 << 63)
                } else {
                    Value::U32((1u32 << (src_width - 1)).wrapping_neg())
                };
                let least = self.emit.i_equal(src, min)?;
                value = self.emit.select(least, value, negative)?;
            }
        }
        self.write_numeric_float(value, dst_width)
    }
}
