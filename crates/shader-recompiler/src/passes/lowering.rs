use crate::{
    CompileError, Result,
    arena::{BlockId, InstId},
    diagnostics::FailureCategory,
    ir::{Emitter, FpControl, Ir, Opcode as O, Program, Type, Value},
};

pub fn lower_fp16(ir: &mut Ir, program: &Program) -> Result<()> {
    for &block in &program.blocks {
        let mut index = 0;
        while let Some(&id) = ir.block(block)?.instructions().get(index) {
            ir.replace_opcode(id, super::lowering_tables::fp16(ir.inst(id)?.opcode()))?;
            index += 1;
        }
    }
    Ok(())
}

// Inserted instructions precede the current instruction and are already lowered.
fn visit(
    ir: &mut Ir,
    blocks: impl Iterator<Item = BlockId>,
    mut lower: impl FnMut(&mut Ir, BlockId, InstId, usize) -> Result<()>,
) -> Result<()> {
    for block in blocks {
        let mut index = 0;
        while let Some(&id) = ir.block(block)?.instructions().get(index) {
            let before = ir.block(block)?.instructions().len();
            lower(ir, block, id, index)?;
            index += 1 + ir.block(block)?.instructions().len() - before;
        }
    }
    Ok(())
}
fn field(e: &mut Emitter<'_>, value: Value, start: u32, width: u32) -> Result<Value> {
    e.bit_field_extract(value, Value::U32(start), Value::U32(width), false)
}
fn exponent(
    e: &mut Emitter<'_>,
    value: Value,
    delta: i32,
    source_max: u32,
    dest_max: u32,
) -> Result<Value> {
    let zero = e.i_equal(value, Value::U32(0))?;
    let biased = e.i_add(value, Value::U32(delta as u32))?;
    let subnormal = e.select(zero, Value::U32(0), biased)?;
    let special = e.i_equal(value, Value::U32(source_max))?;
    e.select(special, Value::U32(dest_max), subnormal)
}
fn assemble_float(
    e: &mut Emitter<'_>,
    sign: Value,
    exponent: Value,
    mantissa: Value,
    shift: u32,
) -> Result<Value> {
    let sign = e.shift_left_logical(sign, Value::U32(31))?;
    let exponent = e.shift_left_logical(exponent, Value::U32(shift))?;
    let body = e.bitwise_or(exponent, mantissa)?;
    e.bitwise_or(sign, body)
}
pub fn lower_fp64(ir: &mut Ir, program: &Program) -> Result<()> {
    visit(
        ir,
        program.blocks.iter().copied(),
        |ir, block, id, position| {
            let op = ir.inst(id)?.opcode();
            if !matches!(op, O::PackDouble2x32 | O::UnpackDouble2x32) {
                return ir.replace_opcode(id, super::lowering_tables::fp64(op));
            }
            let value = ir.inst(id)?.arg(0)?;
            let mut e = Emitter::at_position(ir, block, position)?;
            let result = if op == O::PackDouble2x32 {
                let low = e.composite_extract(value, 0)?;
                let high = e.composite_extract(value, 1)?;
                let sign = field(&mut e, high, 31, 1)?;
                let exp = field(&mut e, high, 20, 11)?;
                let high_mantissa = field(&mut e, high, 0, 20)?;
                let low_mantissa = field(&mut e, low, 29, 3)?;
                let high_mantissa = e.shift_left_logical(high_mantissa, Value::U32(3))?;
                let mantissa = e.bitwise_or(high_mantissa, low_mantissa)?;
                let exp = exponent(&mut e, exp, 896, 0x7ff, 0xff)?;
                let bits = assemble_float(&mut e, sign, exp, mantissa, 23)?;
                e.bit_cast(Type::F32, bits)?
            } else {
                let bits = e.bit_cast(Type::U32, value)?;
                let sign = field(&mut e, bits, 31, 1)?;
                let exp = field(&mut e, bits, 23, 8)?;
                let mantissa = field(&mut e, bits, 0, 23)?;
                let high_mantissa = field(&mut e, mantissa, 3, 20)?;
                let low_mantissa = field(&mut e, mantissa, 0, 3)?;
                let exp = exponent(&mut e, exp, -896, 0xff, 0x7ff)?;
                let low = e.shift_left_logical(low_mantissa, Value::U32(29))?;
                let high = assemble_float(&mut e, sign, exp, high_mantissa, 20)?;
                e.composite_construct(&[low, high])?
            };
            e.ir.replace_uses_with(id, result)
        },
    )
}
fn unpack(e: &mut Emitter<'_>, value: Value) -> Result<(Value, Value)> {
    if let Value::U64(value) = e.ir.resolve(value)? {
        Ok((Value::U32(value as u32), Value::U32((value >> 32) as u32)))
    } else {
        Ok((
            e.composite_extract(value, 0)?,
            e.composite_extract(value, 1)?,
        ))
    }
}
fn bit(e: &mut Emitter<'_>, value: Value) -> Result<Value> {
    e.select(value, Value::U32(1), Value::U32(0))
}
fn negate(e: &mut Emitter<'_>, lo: Value, hi: Value) -> Result<(Value, Value)> {
    let lo = e.bitwise_not(lo)?;
    let hi = e.bitwise_not(hi)?;
    let lo = e.i_add(lo, Value::U32(1))?;
    let carry = e.get_carry_from_op(lo)?;
    let carry = bit(e, carry)?;
    Ok((lo, e.i_add(hi, carry)?))
}
fn shift(e: &mut Emitter<'_>, op: O, lo: Value, hi: Value, amount: Value) -> Result<Value> {
    let left = op == O::ShiftLeftLogical64;
    let signed = op == O::ShiftRightArithmetic64;
    let short_lo = if left {
        e.shift_left_logical(lo, amount)?
    } else {
        e.shift_right_logical(lo, amount)?
    };
    let short_hi = if left {
        e.shift_left_logical(hi, amount)?
    } else if signed {
        e.shift_right_arithmetic(hi, amount)?
    } else {
        e.shift_right_logical(hi, amount)?
    };
    let extension = if signed {
        e.shift_right_arithmetic(hi, Value::U32(31))?
    } else {
        Value::U32(0)
    };
    let excess = e.i_sub(amount, Value::U32(32))?;
    let long = e.i_greater_than_equal(excess, Value::U32(0), true)?;
    let zero = e.i_equal(amount, Value::U32(0))?;
    let complement = e.i_sub(Value::U32(32), amount)?;
    let (long_lo, long_hi, short_lo, short_hi) = if left {
        let long_hi = e.shift_left_logical(lo, excess)?;
        let extract = e.bit_field_extract(lo, complement, amount, false)?;
        (
            Value::U32(0),
            long_hi,
            short_lo,
            e.bitwise_or(short_hi, extract)?,
        )
    } else {
        let long_lo = if signed {
            e.shift_right_arithmetic(hi, excess)?
        } else {
            e.shift_right_logical(hi, excess)?
        };
        let extract = e.bit_field_extract(hi, Value::U32(0), amount, false)?;
        (
            long_lo,
            extension,
            e.bit_field_insert(short_lo, extract, complement, amount)?,
            short_hi,
        )
    };
    let nonzero_lo = e.select(long, long_lo, short_lo)?;
    let nonzero_hi = e.select(long, long_hi, short_hi)?;
    let lo = e.select(zero, lo, nonzero_lo)?;
    let hi = e.select(zero, hi, nonzero_hi)?;
    e.composite_construct(&[lo, hi])
}
fn int_replacement(op: O) -> Option<O> {
    Some(match op {
        O::PackUint2x32 | O::UnpackUint2x32 => O::Identity,
        O::SharedAtomicExchange64 => O::SharedAtomicExchange32x2,
        O::GlobalAtomicIAdd64 => O::GlobalAtomicIAdd32x2,
        O::GlobalAtomicSMin64 => O::GlobalAtomicSMin32x2,
        O::GlobalAtomicUMin64 => O::GlobalAtomicUMin32x2,
        O::GlobalAtomicSMax64 => O::GlobalAtomicSMax32x2,
        O::GlobalAtomicUMax64 => O::GlobalAtomicUMax32x2,
        O::GlobalAtomicAnd64 => O::GlobalAtomicAnd32x2,
        O::GlobalAtomicOr64 => O::GlobalAtomicOr32x2,
        O::GlobalAtomicXor64 => O::GlobalAtomicXor32x2,
        O::GlobalAtomicExchange64 => O::GlobalAtomicExchange32x2,
        O::StorageAtomicIAdd64 => O::StorageAtomicIAdd32x2,
        O::StorageAtomicSMin64 => O::StorageAtomicSMin32x2,
        O::StorageAtomicUMin64 => O::StorageAtomicUMin32x2,
        O::StorageAtomicSMax64 => O::StorageAtomicSMax32x2,
        O::StorageAtomicUMax64 => O::StorageAtomicUMax32x2,
        O::StorageAtomicAnd64 => O::StorageAtomicAnd32x2,
        O::StorageAtomicOr64 => O::StorageAtomicOr32x2,
        O::StorageAtomicXor64 => O::StorageAtomicXor32x2,
        O::StorageAtomicExchange64 => O::StorageAtomicExchange32x2,
        O::BitCastU64F64 => O::UnpackDouble2x32,
        O::BitCastF64U64 => O::PackDouble2x32,
        _ => return None,
    })
}
pub fn lower_int64(ir: &mut Ir, program: &Program) -> Result<()> {
    visit(
        ir,
        program.post_order.iter().rev().copied(),
        |ir, block, id, position| {
            let inst = ir.inst(id)?;
            let op = inst.opcode();
            if let Some(replacement) = int_replacement(op) {
                return ir.replace_opcode(id, replacement);
            }
            let mut args = [Value::None; 3];
            for (i, arg) in args.iter_mut().enumerate().take(inst.num_args().min(3)) {
                *arg = inst.arg(i)?;
            }
            if matches!(
                op,
                O::IAdd64
                    | O::ISub64
                    | O::INeg64
                    | O::ShiftLeftLogical64
                    | O::ShiftRightLogical64
                    | O::ShiftRightArithmetic64
            ) && inst.has_associated_pseudo_operation()
            {
                return Err(CompileError::new(
                    FailureCategory::Unsupported,
                    format!("Cannot lower {op:?} with associated pseudo-results"),
                ));
            }
            let mut e = Emitter::at_position(ir, block, position)?;
            let result = match op {
                O::IAdd64 | O::ISub64 => {
                    let (a, b) = unpack(&mut e, args[0])?;
                    let (c, d) = unpack(&mut e, args[1])?;
                    let (lo, hi) = if op == O::IAdd64 {
                        let lo = e.i_add(a, c)?;
                        let carry = e.get_carry_from_op(lo)?;
                        let carry = bit(&mut e, carry)?;
                        let hi = e.i_add(b, d)?;
                        (lo, e.i_add(hi, carry)?)
                    } else {
                        let lo = e.i_sub(a, c)?;
                        let borrow = e.i_greater_than(lo, a, false)?;
                        let borrow = bit(&mut e, borrow)?;
                        let hi = e.i_sub(b, d)?;
                        (lo, e.i_sub(hi, borrow)?)
                    };
                    e.composite_construct(&[lo, hi])?
                }
                O::INeg64 | O::IAbs64 => {
                    let (lo, hi) = unpack(&mut e, args[0])?;
                    let (mut result_lo, mut result_hi) = negate(&mut e, lo, hi)?;
                    if op == O::IAbs64 {
                        let sign = e.bitwise_and(hi, Value::U32(0x80000000))?;
                        let negative = e.i_not_equal(sign, Value::U32(0))?;
                        result_lo = e.select(negative, result_lo, lo)?;
                        result_hi = e.select(negative, result_hi, hi)?;
                    }
                    e.composite_construct(&[result_lo, result_hi])?
                }
                O::ShiftLeftLogical64 | O::ShiftRightLogical64 | O::ShiftRightArithmetic64 => {
                    let (lo, hi) = unpack(&mut e, args[0])?;
                    shift(&mut e, op, lo, hi, args[1])?
                }
                O::SelectU64 => {
                    let (a, b) = unpack(&mut e, args[1])?;
                    let (c, d) = unpack(&mut e, args[2])?;
                    let lo = e.select(args[0], a, c)?;
                    let hi = e.select(args[0], b, d)?;
                    e.composite_construct(&[lo, hi])?
                }
                O::UndefU64 => e.composite_construct(&[Value::U32(0), Value::U32(0)])?,
                O::ConvertU64U32 => e.composite_construct(&[args[0], Value::U32(0)])?,
                O::ConvertU32U64 => unpack(&mut e, args[0])?.0,
                O::ConvertF16S64
                | O::ConvertF32S64
                | O::ConvertF64S64
                | O::ConvertF16U64
                | O::ConvertF32U64
                | O::ConvertF64U64 => {
                    let signed =
                        matches!(op, O::ConvertF16S64 | O::ConvertF32S64 | O::ConvertF64S64);
                    let width = match op {
                        O::ConvertF16S64 | O::ConvertF16U64 => 16,
                        O::ConvertF32S64 | O::ConvertF32U64 => 32,
                        _ => 64,
                    };
                    let (lo, hi) = unpack(&mut e, args[0])?;
                    let low = e.convert_i_to_f(32, 32, false, lo, FpControl::default())?;
                    let high = e.convert_i_to_f(32, 32, signed, hi, FpControl::default())?;
                    let value = e.fp_fma(
                        high,
                        Value::F32(4294967296.0f32.to_bits()),
                        low,
                        FpControl::default(),
                    )?;
                    e.fp_convert(width, value, FpControl::default())?
                }
                O::ConvertS64F16
                | O::ConvertS64F32
                | O::ConvertS64F64
                | O::ConvertU64F16
                | O::ConvertU64F32
                | O::ConvertU64F64 => {
                    let signed =
                        matches!(op, O::ConvertS64F16 | O::ConvertS64F32 | O::ConvertS64F64);
                    let value = e.fp_convert(32, args[0], FpControl::default())?;
                    let high = e.fp_mul(
                        value,
                        Value::F32((1.0f32 / 4294967296.0).to_bits()),
                        FpControl::default(),
                    )?;
                    let high = e.fp_floor(high, FpControl::default())?;
                    let hi = e.convert_f_to_i(32, signed, high)?;
                    let negative = e.fp_neg(Value::F32(4294967296.0f32.to_bits()))?;
                    let low = e.fp_fma(high, negative, value, FpControl::default())?;
                    let lo = e.convert_f_to_i(32, false, low)?;
                    e.composite_construct(&[lo, hi])?
                }
                _ => return Ok(()),
            };
            e.ir.replace_uses_with(id, result)
        },
    )
}
