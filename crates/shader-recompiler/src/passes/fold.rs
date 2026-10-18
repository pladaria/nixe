use crate::{
    CompileError, Environment, Result,
    arena::{BlockId, InstId},
    diagnostics::FailureCategory,
    ir::{Attribute, Emitter, FpControl, Ir, Opcode as O, Program, Value},
    types::{ByteOffset, CbufIndex, ReplaceConstant},
};

pub(super) fn node(ir: &Ir, value: Value, opcode: O) -> Result<Option<InstId>> {
    Ok(match ir.resolve(value)? {
        Value::Inst(id) if ir.inst(id)?.opcode() == opcode => Some(id),
        _ => None,
    })
}
pub(super) fn arg(ir: &Ir, id: InstId, index: usize) -> Result<Value> {
    ir.resolve(ir.inst(id)?.arg(index)?)
}
pub(super) fn equal_cbuf(ir: &Ir, a: Value, b: Value) -> Result<bool> {
    let (Some(a), Some(b)) = (node(ir, a, O::GetCbufU32)?, node(ir, b, O::GetCbufU32)?) else {
        return Ok(false);
    };
    Ok(ir.inst(a)?.arg(0)? == ir.inst(b)?.arg(0)? && ir.inst(a)?.arg(1)? == ir.inst(b)?.arg(1)?)
}
fn immediate(op: O, a: &[Value]) -> Result<Option<Value>> {
    let undefined = || {
        CompileError::new(
            FailureCategory::UnsupportedReferenceCase,
            format!("Undefined immediate operands for {op:?}: {a:?}"),
        )
    };
    let value = match (op, a) {
        (O::IMul32, [Value::U32(a), Value::U32(b)]) => Value::U32(a.wrapping_mul(*b)),
        (O::ISub32, [Value::U32(a), Value::U32(b)]) => Value::U32(a.wrapping_sub(*b)),
        (O::ShiftRightArithmetic32, [Value::U32(a), Value::U32(b)]) => {
            if *b >= 32 {
                return Err(undefined());
            }
            Value::U32((*a as i32 >> b) as u32)
        }
        (O::FPNeg32, [Value::F32(a)]) => Value::f32(-f32::from_bits(*a)),
        (O::FPAdd32, [Value::F32(a), Value::F32(b)]) => {
            Value::f32(f32::from_bits(*a) + f32::from_bits(*b))
        }
        (O::FPMul32, [Value::F32(a), Value::F32(b)]) => {
            Value::f32(f32::from_bits(*a) * f32::from_bits(*b))
        }
        (O::SLessThan, [Value::U32(a), Value::U32(b)]) => Value::U1((*a as i32) < (*b as i32)),
        (O::ULessThan, [Value::U32(a), Value::U32(b)]) => Value::U1(a < b),
        (O::SLessThanEqual, [Value::U32(a), Value::U32(b)]) => {
            Value::U1((*a as i32) <= (*b as i32))
        }
        (O::ULessThanEqual, [Value::U32(a), Value::U32(b)]) => Value::U1(a <= b),
        (O::SGreaterThan, [Value::U32(a), Value::U32(b)]) => Value::U1((*a as i32) > (*b as i32)),
        (O::UGreaterThan, [Value::U32(a), Value::U32(b)]) => Value::U1(a > b),
        (O::SGreaterThanEqual, [Value::U32(a), Value::U32(b)]) => {
            Value::U1((*a as i32) >= (*b as i32))
        }
        (O::UGreaterThanEqual, [Value::U32(a), Value::U32(b)]) => Value::U1(a >= b),
        (O::IEqual, [Value::U32(a), Value::U32(b)]) => Value::U1(a == b),
        (O::INotEqual, [Value::U32(a), Value::U32(b)]) => Value::U1(a != b),
        (O::BitwiseAnd32, [Value::U32(a), Value::U32(b)]) => Value::U32(a & b),
        (O::BitwiseOr32, [Value::U32(a), Value::U32(b)]) => Value::U32(a | b),
        (O::BitwiseXor32, [Value::U32(a), Value::U32(b)]) => Value::U32(a ^ b),
        (O::BitFieldUExtract, [Value::U32(a), Value::U32(shift), Value::U32(count)]) => {
            if u64::from(*shift) + u64::from(*count) > 32 {
                return Err(CompileError::new(
                    FailureCategory::Internal,
                    "Unsigned bitfield exceeds word",
                ));
            }
            if *shift >= 32 || *count >= 32 {
                return Err(undefined());
            }
            Value::U32((a >> shift) & ((1 << count) - 1))
        }
        (O::BitFieldSExtract, [Value::U32(a), Value::U32(shift), Value::U32(count)]) => {
            let back = u64::from(*shift) + u64::from(*count);
            if back > 32 || back == 0 || *count == 0 {
                return Err(CompileError::new(
                    FailureCategory::Internal,
                    "Signed bitfield has undefined shift",
                ));
            }
            Value::U32((a.wrapping_shl((32 - back) as u32) as i32 >> (32 - count)) as u32)
        }
        (
            O::BitFieldInsert,
            [
                Value::U32(a),
                Value::U32(b),
                Value::U32(offset),
                Value::U32(bits),
            ],
        ) => {
            if *bits >= 32 || *offset >= 32 {
                return Err(CompileError::new(
                    FailureCategory::Internal,
                    "Bitfield insertion has undefined shift",
                ));
            }
            Value::U32((a & !((!(u32::MAX << bits)) << offset)) | (b << offset))
        }
        _ => return Ok(None),
    };
    Ok(Some(value))
}
fn combine(a: Value, b: Value, op: O) -> Option<Value> {
    match (op, a, b) {
        (O::IAdd32, Value::U32(a), Value::U32(b)) => Some(Value::U32(a.wrapping_add(b))),
        (O::IAdd64, Value::U64(a), Value::U64(b)) => Some(Value::U64(a.wrapping_add(b))),
        (O::LogicalAnd, Value::U1(a), Value::U1(b)) => Some(Value::U1(a && b)),
        (O::LogicalOr, Value::U1(a), Value::U1(b)) => Some(Value::U1(a || b)),
        _ => None,
    }
}
fn commutative(ir: &mut Ir, id: InstId, op: O) -> Result<bool> {
    let a = arg(ir, id, 0)?;
    let b = arg(ir, id, 1)?;
    if let Some(value) = combine(a, b, op) {
        ir.replace_uses_with(id, value)?;
        return Ok(false);
    }
    if !matches!(a, Value::Inst(_)) && matches!(b, Value::Inst(_)) {
        if let Some(nested) = node(ir, b, op)?
            && let Some(combined) = combine(a, arg(ir, nested, 1)?, op)
        {
            ir.set_arg(id, 0, ir.inst(nested)?.arg(0)?)?;
            ir.set_arg(id, 1, combined)?;
        } else {
            ir.set_arg(id, 0, b)?;
            ir.set_arg(id, 1, a)?;
        }
    } else if matches!(a, Value::Inst(_))
        && !matches!(b, Value::Inst(_))
        && let Some(nested) = node(ir, a, op)?
        && let Some(combined) = combine(b, arg(ir, nested, 1)?, op)
    {
        ir.set_arg(id, 0, ir.inst(nested)?.arg(0)?)?;
        ir.set_arg(id, 1, combined)?;
    }
    Ok(true)
}
fn composite_pair(op: O) -> Option<(O, O)> {
    Some(match op {
        O::CompositeExtractU32x2 => (O::CompositeConstructU32x2, O::CompositeInsertU32x2),
        O::CompositeExtractU32x3 => (O::CompositeConstructU32x3, O::CompositeInsertU32x3),
        O::CompositeExtractU32x4 => (O::CompositeConstructU32x4, O::CompositeInsertU32x4),
        O::CompositeExtractF32x2 => (O::CompositeConstructF32x2, O::CompositeInsertF32x2),
        O::CompositeExtractF32x3 => (O::CompositeConstructF32x3, O::CompositeInsertF32x3),
        O::CompositeExtractF32x4 => (O::CompositeConstructF32x4, O::CompositeInsertF32x4),
        O::CompositeExtractF16x2 => (O::CompositeConstructF16x2, O::CompositeInsertF16x2),
        O::CompositeExtractF16x3 => (O::CompositeConstructF16x3, O::CompositeInsertF16x3),
        O::CompositeExtractF16x4 => (O::CompositeConstructF16x4, O::CompositeInsertF16x4),
        _ => return None,
    })
}
fn subtract(ir: &mut Ir, id: InstId) -> Result<()> {
    let a = arg(ir, id, 0)?;
    let b = arg(ir, id, 1)?;
    if !matches!((a, b), (Value::Inst(_), Value::Inst(_))) {
        return Ok(());
    }
    if equal_cbuf(ir, a, b)? {
        return ir.replace_uses_with(id, Value::U32(0));
    }
    let (add, cbuf) = if node(ir, b, O::IAdd32)?.is_some() {
        (b, a)
    } else {
        (a, b)
    };
    let Some(add) = node(ir, add, O::IAdd32)? else {
        return Ok(());
    };
    let mut left = arg(ir, add, 0)?;
    let mut right = arg(ir, add, 1)?;
    if ir.is_immediate(right)? {
        std::mem::swap(&mut left, &mut right);
    }
    if equal_cbuf(ir, right, cbuf)? {
        ir.replace_uses_with(id, left)?;
    }
    Ok(())
}
fn bitcast(ir: &mut Ir, id: InstId, op: O) -> Result<()> {
    let value = arg(ir, id, 0)?;
    match (op, value) {
        (O::BitCastF32U32, Value::U32(x)) => return ir.replace_uses_with(id, Value::F32(x)),
        (O::BitCastU32F32, Value::F32(x)) => return ir.replace_uses_with(id, Value::U32(x)),
        _ => (),
    }
    let reverse = if op == O::BitCastF32U32 {
        O::BitCastU32F32
    } else {
        O::BitCastF32U32
    };
    if let Some(cast) = node(ir, value, reverse)? {
        return ir.replace_uses_with(id, ir.inst(cast)?.arg(0)?);
    }
    let source = if op == O::BitCastF32U32 {
        node(ir, value, O::GetCbufU32)?
    } else {
        node(ir, value, O::GetAttribute)?
    };
    if let Some(source) = source {
        let a = ir.inst(source)?.arg(0)?;
        let b = ir.inst(source)?.arg(1)?;
        if op == O::BitCastU32F32
            && !matches!(
                a,
                Value::Attribute(
                    Attribute::PrimitiveId
                        | Attribute::InstanceId
                        | Attribute::VertexId
                        | Attribute::BaseVertex
                        | Attribute::BaseInstance
                        | Attribute::DrawID
                )
            )
        {
            return Ok(());
        }
        ir.replace_opcode(
            id,
            if op == O::BitCastF32U32 {
                O::GetCbufF32
            } else {
                O::GetAttributeU32
            },
        )?;
        ir.set_arg(id, 0, a)?;
        ir.set_arg(id, 1, b)?;
    }
    Ok(())
}
fn multiply(ir: &mut Ir, id: InstId) -> Result<()> {
    if FpControl::from_bits(ir.inst(id)?.flags)?.no_contraction {
        return Ok(());
    }
    let a = arg(ir, id, 0)?;
    let b = arg(ir, id, 1)?;
    if !matches!((a, b), (Value::Inst(_), Value::Inst(_))) {
        return Ok(());
    }
    if node(ir, a, O::YDirection)?.is_some() && node(ir, b, O::DPdyFine)?.is_some() {
        return ir.replace_uses_with(id, b);
    }
    if node(ir, b, O::YDirection)?.is_some() && node(ir, a, O::DPdyFine)?.is_some() {
        return ir.replace_uses_with(id, a);
    }
    let (Some(mul), Some(recip)) = (node(ir, a, O::FPMul32)?, node(ir, b, O::FPRecip32)?) else {
        return Ok(());
    };
    let (Some(attr_a), Some(attr_b)) = (
        node(ir, arg(ir, recip, 0)?, O::GetAttribute)?,
        node(ir, arg(ir, mul, 1)?, O::GetAttribute)?,
    ) else {
        return Ok(());
    };
    if ir.inst(attr_a)?.arg(0)? == ir.inst(attr_b)?.arg(0)? {
        ir.replace_uses_with(id, ir.inst(mul)?.arg(0)?)?;
    }
    Ok(())
}
fn cbuf<E: Environment>(
    env: &mut E,
    ir: &mut Ir,
    block: BlockId,
    id: InstId,
    position: usize,
) -> Result<()> {
    let (Value::U32(bank), Value::U32(offset)) = (arg(ir, id, 0)?, arg(ir, id, 1)?) else {
        return Ok(());
    };
    if env.metadata().has_hle_macro_state
        && let Some(replacement) =
            env.get_replace_const_buffer(CbufIndex(bank), ByteOffset(offset))?
    {
        let attribute = match replacement {
            ReplaceConstant::BaseInstance => Attribute::BaseInstance,
            ReplaceConstant::BaseVertex => Attribute::BaseVertex,
            ReplaceConstant::DrawID => Attribute::DrawID,
        };
        let integer = ir.inst(id)?.opcode() == O::GetCbufU32;
        let mut e = Emitter::at_position(ir, block, position)?;
        let value = if integer {
            e.get_attribute_u32(attribute)?
        } else {
            e.get_attribute(attribute)?
        };
        e.ir.replace_uses_with(id, value)?;
    }
    if env.metadata().is_proprietary_driver {
        // Consume the current instruction, including any preceding replacement.
        if ir.inst(id)?.num_args() < 2 {
            return Err(CompileError::new(
                FailureCategory::UnsupportedReferenceCase,
                "Driver constant folding follows an HLE replacement without cbuf operands",
            ));
        }
        let (Value::U32(bank), Value::U32(offset)) = (arg(ir, id, 0)?, arg(ir, id, 1)?) else {
            return Ok(());
        };
        if bank == 1 && offset < u16::MAX as u32 {
            let bits = env.read_cbuf_value(CbufIndex(bank), ByteOffset(offset))?;
            let value = if ir.inst(id)?.opcode() == O::GetCbufU32 {
                Value::U32(bits)
            } else {
                Value::F32(bits)
            };
            ir.replace_uses_with(id, value)?;
        }
    }
    Ok(())
}
fn fold<E: Environment>(
    env: &mut E,
    ir: &mut Ir,
    block: BlockId,
    id: InstId,
    position: usize,
) -> Result<()> {
    let op = ir.inst(id)?.opcode();
    if op == O::Phi {
        return Ok(());
    }
    let mut args = [Value::None; 5];
    let count = ir.inst(id)?.num_args();
    for (i, value) in args.iter_mut().enumerate().take(count) {
        *value = arg(ir, id, i)?;
    }
    if !ir.inst(id)?.has_associated_pseudo_operation()
        && let Some(value) = immediate(op, &args[..count])?
    {
        return ir.replace_uses_with(id, value);
    }
    match op {
        O::GetRegister if args[0] == Value::Reg(crate::ir::Reg::RZ) => {
            ir.replace_uses_with(id, Value::U32(0))
        }
        O::GetPred if args[0] == Value::Pred(crate::ir::Pred::PT) => {
            ir.replace_uses_with(id, Value::U1(true))
        }
        O::IAdd32 | O::IAdd64 | O::LogicalAnd | O::LogicalOr => {
            if matches!(op, O::IAdd32 | O::IAdd64) && ir.inst(id)?.has_associated_pseudo_operation()
            {
                return Ok(());
            }
            if !commutative(ir, id, op)? {
                return Ok(());
            }
            let rhs = arg(ir, id, 1)?;
            match (op, rhs) {
                (O::IAdd32, Value::U32(0))
                | (O::IAdd64, Value::U64(0))
                | (O::LogicalAnd, Value::U1(true))
                | (O::LogicalOr, Value::U1(false)) => {
                    ir.replace_uses_with(id, ir.inst(id)?.arg(0)?)
                }
                (O::LogicalAnd, Value::U1(false)) | (O::LogicalOr, Value::U1(true)) => {
                    ir.replace_uses_with(id, rhs)
                }
                (O::IAdd32, _) => super::fold_patterns::xmad(ir, block, id, position),
                _ => Ok(()),
            }
        }
        O::ISub32 => subtract(ir, id),
        O::SelectU1
        | O::SelectU8
        | O::SelectU16
        | O::SelectU32
        | O::SelectU64
        | O::SelectF16
        | O::SelectF32
        | O::SelectF64 => {
            if let Value::U1(condition) = args[0] {
                ir.replace_uses_with(id, args[if condition { 1 } else { 2 }])?;
            }
            Ok(())
        }
        O::FPAdd32 => {
            if matches!(args[0],Value::F32(bits) if bits&0x7fffffff==0) {
                ir.replace_uses_with(id, args[1])?;
            }
            if matches!(args[1],Value::F32(bits) if bits&0x7fffffff==0) {
                ir.replace_uses_with(id, args[0])?;
            }
            Ok(())
        }
        O::FPMul32 => multiply(ir, id),
        O::LogicalNot => {
            if let Value::U1(value) = args[0] {
                ir.replace_uses_with(id, Value::U1(!value))?;
            } else if let Some(inner) = node(ir, args[0], O::LogicalNot)? {
                ir.replace_uses_with(id, ir.inst(inner)?.arg(0)?)?;
            }
            Ok(())
        }
        O::BitCastF32U32 | O::BitCastU32F32 => bitcast(ir, id, op),
        O::PackHalf2x16 | O::UnpackHalf2x16 | O::PackFloat2x16 | O::UnpackFloat2x16 => {
            let inverse = match op {
                O::PackHalf2x16 => O::UnpackHalf2x16,
                O::UnpackHalf2x16 => O::PackHalf2x16,
                O::PackFloat2x16 => O::UnpackFloat2x16,
                _ => O::PackFloat2x16,
            };
            if let Some(inner) = node(ir, args[0], inverse)? {
                ir.replace_uses_with(id, ir.inst(inner)?.arg(0)?)?;
            }
            Ok(())
        }
        O::GetCbufU32 | O::GetCbufF32 => cbuf(env, ir, block, id, position),
        O::FSwizzleAdd => super::fold_patterns::derivative(ir, block, id, position),
        O::BindlessImageSampleImplicitLod
        | O::BoundImageSampleImplicitLod
        | O::ImageSampleImplicitLod => super::fold_patterns::gradient(ir, block, id, position),
        _ => {
            if let Some((construct, insert)) = composite_pair(op)
                && let Value::U32(index) = args[1]
            {
                let mut value = args[0];
                loop {
                    if let Some(constructor) = node(ir, value, construct)? {
                        ir.replace_uses_with(id, ir.inst(constructor)?.arg(index as usize)?)?;
                        break;
                    }
                    let Some(insertion) = node(ir, value, insert)? else {
                        break;
                    };
                    let Value::U32(second) = arg(ir, insertion, 2)? else {
                        break;
                    };
                    if index == second {
                        ir.replace_uses_with(id, ir.inst(insertion)?.arg(1)?)?;
                        break;
                    }
                    value = arg(ir, insertion, 0)?;
                }
            }
            Ok(())
        }
    }
}
pub fn constant_propagation<E: Environment>(
    env: &mut E,
    ir: &mut Ir,
    program: &Program,
) -> Result<()> {
    for &block in program.post_order.iter().rev() {
        let mut position = 0;
        while let Some(&id) = ir.block(block)?.instructions().get(position) {
            let before = ir.block(block)?.instructions().len();
            fold(env, ir, block, id, position)?;
            position += 1 + ir.block(block)?.instructions().len() - before;
        }
    }
    Ok(())
}
