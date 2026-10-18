use super::fold::{arg, equal_cbuf, node};
use crate::{
    CompileError, Result,
    arena::{BlockId, InstId},
    diagnostics::FailureCategory,
    ir::{Emitter, Ir, Opcode as O, TextureInstInfo, Type, Value},
    types::TextureType,
};

fn operands(ir: &Ir, value: Value, opcode: O) -> Option<[Value; 4]> {
    let id = node(ir, value, opcode).ok()??;
    let inst = ir.inst(id).ok()?;
    let mut values = [Value::None; 4];
    for (index, value) in values.iter_mut().enumerate().take(inst.num_args()) {
        *value = arg(ir, id, index).ok()?;
    }
    Some(values)
}
fn extract(ir: &Ir, value: Value, offset: u32) -> Option<Value> {
    let args = operands(ir, value, O::BitFieldUExtract)?;
    (args[1] == Value::U32(offset) && args[2] == Value::U32(16)).then_some(args[0])
}
fn shift16(ir: &Ir, value: Value) -> Option<Value> {
    let args = operands(ir, value, O::ShiftLeftLogical32)?;
    (args[1] == Value::U32(16)).then_some(args[0])
}
fn multiply(ir: &Ir, id: InstId) -> Option<(Value, Value)> {
    let lhs = shift16(ir, arg(ir, id, 0).ok()?)?;
    let a = operands(ir, lhs, O::IMul32)?;
    let b = operands(ir, arg(ir, id, 1).ok()?, O::IMul32)?;
    if a[1] != b[1] {
        return None;
    }
    let factor = extract(ir, a[0], 16)?;
    (factor == extract(ir, b[0], 0)?).then_some((factor, a[1]))
}
fn equal(ir: &Ir, values: [Value; 3]) -> Option<bool> {
    for pair in values.windows(2) {
        if pair[0] != pair[1] && !equal_cbuf(ir, pair[0], pair[1]).ok()? {
            return Some(false);
        }
    }
    Some(true)
}
fn multiply_add(ir: &Ir, id: InstId) -> Option<(Value, Value, Value)> {
    let high = operands(ir, shift16(ir, arg(ir, id, 0).ok()?)?, O::IMul32)?;
    let outer = operands(ir, arg(ir, id, 1).ok()?, O::IAdd32)?;
    let factor_b = extract(ir, high[0], 16)?;
    let packed = extract(ir, high[1], 16)?;
    let merge = operands(ir, packed, O::BitFieldInsert)?;
    if merge[2] != Value::U32(16) || merge[3] != Value::U32(16) || shift16(ir, outer[0])? != packed
    {
        return None;
    }
    let partial = operands(ir, merge[0], O::IMul32)?;
    let low_add = operands(ir, outer[1], O::IAdd32)?;
    let low = operands(ir, low_add[0], O::IMul32)?;
    let a = [
        extract(ir, partial[1], 16)?,
        extract(ir, low[1], 0)?,
        extract(ir, merge[1], 0)?,
    ];
    let b = [
        factor_b,
        extract(ir, partial[0], 0)?,
        extract(ir, low[0], 0)?,
    ];
    (equal(ir, a)? && equal(ir, b)?).then_some((a[0], b[1], low_add[1]))
}
pub(super) fn xmad(ir: &mut Ir, block: BlockId, id: InstId, position: usize) -> Result<()> {
    if let Some((a, b)) = multiply(ir, id) {
        let mut e = Emitter::at_position(ir, block, position)?;
        let value = e.i_mul(a, b)?;
        return e.ir.replace_uses_with(id, value);
    }
    if let Some((a, b, c)) = multiply_add(ir, id) {
        let mut e = Emitter::at_position(ir, block, position)?;
        let product = e.i_mul(a, b)?;
        let value = e.i_add(product, c)?;
        e.ir.replace_uses_with(id, value)?;
    }
    Ok(())
}
fn through_cast(ir: &Ir, value: Value, opcode: O) -> Result<Value> {
    let value = ir.resolve(value)?;
    if let Some(cast) = node(ir, value, opcode)? {
        arg(ir, cast, 0)
    } else {
        Ok(value)
    }
}
fn butterfly(ir: &Ir, value: Value) -> Result<u32> {
    if let Some(quad) = node(ir, value, O::QuadSwap)? {
        return Ok(if let Value::U32(direction) = arg(ir, quad, 1)? {
            direction.wrapping_add(1)
        } else {
            0
        });
    }
    if let Some(shuffle) = node(ir, value, O::ShuffleButterfly)?
        && let (Value::U32(index), Value::U32(3), Value::U32(28)) = (
            arg(ir, shuffle, 1)?,
            arg(ir, shuffle, 2)?,
            arg(ir, shuffle, 3)?,
        )
    {
        return Ok(index);
    }
    Ok(0)
}
pub(super) fn derivative(ir: &mut Ir, block: BlockId, id: InstId, position: usize) -> Result<()> {
    let mask = arg(ir, id, 2)?;
    if !matches!(mask, Value::U32(0x99 | 0xa5)) {
        return Ok(());
    }
    let a = through_cast(ir, arg(ir, id, 0)?, O::BitCastF32U32)?;
    let b = through_cast(ir, arg(ir, id, 1)?, O::BitCastF32U32)?;
    let lane = butterfly(ir, a)?;
    if lane == 0 {
        return Ok(());
    }
    let Value::Inst(shuffle) = ir.resolve(a)? else {
        return Ok(());
    };
    let c = through_cast(ir, arg(ir, shuffle, 0)?, O::BitCastU32F32)?;
    if b != c && !matches!((b,c),(Value::F32(b),Value::U32(c)) if b==c) {
        return Ok(());
    }
    let source = ir.inst(id)?.arg(1)?;
    let mut e = Emitter::at_position(ir, block, position)?;
    let result = match (mask, lane) {
        (Value::U32(0x99), 1) => e.d_pdx_fine(source)?,
        (Value::U32(0xa5), 2) => e.d_pdy_fine(source)?,
        _ => return Ok(()),
    };
    e.ir.replace_uses_with(id, result)
}
fn through_shuffle(ir: &Ir, value: Value) -> Result<Option<Value>> {
    let value = through_cast(ir, value, O::BitCastF32U32)?;
    if let Some(quad) = node(ir, value, O::QuadBroadcast)? {
        if !ir.is_immediate(arg(ir, quad, 1)?)? {
            return Ok(None);
        }
        return Ok(Some(through_cast(ir, arg(ir, quad, 0)?, O::BitCastU32F32)?));
    }
    let Some(shuffle) = node(ir, value, O::ShuffleIndex)? else {
        return Ok(None);
    };
    let (Value::U32(index), Value::U32(clamp), Value::U32(_)) = (
        arg(ir, shuffle, 1)?,
        arg(ir, shuffle, 2)?,
        arg(ir, shuffle, 3)?,
    ) else {
        return Ok(None);
    };
    if index != 3 && clamp != 3 {
        return Ok(None);
    }
    Ok(Some(through_cast(
        ir,
        arg(ir, shuffle, 0)?,
        O::BitCastU32F32,
    )?))
}
fn mask(ir: &Ir, id: InstId) -> Result<u32> {
    if let Value::U32(mask) = arg(ir, id, 2)? {
        Ok(mask)
    } else {
        Err(CompileError::new(
            FailureCategory::UnsupportedReferenceCase,
            "Gradient pattern has a non-immediate swizzle mask",
        ))
    }
}
fn pending(ir: &Ir, value: Value) -> Result<Option<(Value, Value, u32)>> {
    let Some(inner) = node(ir, value, O::FSwizzleAdd)? else {
        return Ok(None);
    };
    let (Some(a), Some(b)) = (
        through_shuffle(ir, arg(ir, inner, 0)?)?,
        through_shuffle(ir, arg(ir, inner, 1)?)?,
    ) else {
        return Ok(None);
    };
    Ok(Some((a, b, mask(ir, inner)?)))
}
fn gradient_components(ir: &Ir, coordinate: Value) -> Result<Option<[Value; 3]>> {
    let Some(outer) = node(ir, coordinate, O::FSwizzleAdd)? else {
        return Ok(None);
    };
    let a = arg(ir, outer, 0)?;
    let b = arg(ir, outer, 1)?;
    let outer_mask = mask(ir, outer)?;
    if ir.is_immediate(a)? || ir.is_immediate(b)? {
        return Ok(None);
    }
    let mut state = None;
    if let Some((x, y, mask)) = pending(ir, a)? {
        state = through_shuffle(ir, b)?.map(|z| ([z, x, y], 0, mask));
    }
    if let Some((x, y, mask)) = pending(ir, b)? {
        state = through_shuffle(ir, a)?.map(|z| ([z, x, y], 2, mask));
    }
    let Some((temporary, coordinate_index, inner_mask)) = state else {
        return Ok(None);
    };
    let mut zero_a = 0u32;
    let mut zero_b = 0u32;
    for lane in 0..4 {
        let a = (outer_mask >> (lane * 2)) & 3;
        let b = (inner_mask >> (lane * 2)) & 3;
        if a == 2 || b == 2 {
            return Ok(None);
        }
        zero_a |= u32::from(a == 3) << lane;
        zero_b |= u32::from(b == 3) << lane;
    }
    if zero_a.count_ones() != 2 || zero_b.count_ones() != 2 || zero_a == zero_b {
        return Ok(None);
    }
    let mut result = [temporary[coordinate_index], Value::None, Value::None];
    if coordinate_index == 0 {
        let horizontal = matches!(zero_b, 0b1010 | 0b0101);
        result[if horizontal { 1 } else { 2 }] = temporary[1];
        result[if horizontal { 2 } else { 1 }] = temporary[2];
    } else {
        result[if matches!(zero_b, 0b1010 | 0b0101) {
            1
        } else {
            2
        }] = temporary[1];
        result[if matches!(zero_a, 0b1010 | 0b0101) {
            1
        } else {
            2
        }] = temporary[0];
    }
    Ok(Some(result))
}
pub(super) fn gradient(ir: &mut Ir, block: BlockId, id: InstId, position: usize) -> Result<()> {
    let mut info = TextureInstInfo(ir.inst(id)?.flags);
    if !info.ndv_is_active()
        || info.texture_type() != TextureType::Color3D
        || !ir.is_immediate(ir.inst(id)?.arg(3)?)?
    {
        return Ok(());
    }
    let handle = ir.inst(id)?.arg(0)?;
    let coords = arg(ir, id, 1)?;
    let bias = arg(ir, id, 2)?;
    let indexed = ir.inst(id)?.opcode() == O::ImageSampleImplicitLod;
    let Value::Inst(coords) = coords else {
        return Err(CompileError::new(
            FailureCategory::UnsupportedReferenceCase,
            "Gradient pattern coordinates are not an instruction",
        ));
    };
    let mut results = [[Value::None; 3]; 3];
    let mut insertion = position;
    for (i, result) in results.iter_mut().enumerate() {
        let Some(values) = gradient_components(ir, arg(ir, coords, i)?)? else {
            return Ok(());
        };
        *result = values;
        let before = ir.block(block)?.instructions().len();
        let mut e = Emitter::at_position(ir, block, insertion)?;
        for value in result {
            if e.ir.value_type(*value)? == Type::U32 {
                *value = if let Value::U32(bits) = e.ir.resolve(*value)? {
                    Value::F32(bits)
                } else {
                    e.bit_cast(Type::F32, *value)?
                };
            }
        }
        insertion += e.ir.block(block)?.instructions().len() - before;
    }
    let clamp = if info.has_lod_clamp() {
        if let Value::Inst(bias) = bias {
            arg(ir, bias, 1)?
        } else {
            bias
        }
    } else {
        Value::None
    };
    let mut e = Emitter::at_position(ir, block, insertion)?;
    let coords = e.composite_construct(&[results[0][0], results[1][0], results[2][0]])?;
    let derivatives =
        e.composite_construct(&[results[0][1], results[0][2], results[1][1], results[1][2]])?;
    let extra = e.composite_construct(&[results[2][1], results[2][2]])?;
    info.set_num_derivatives(3);
    let result = e.image_gradient(handle, coords, derivatives, extra, clamp, info)?;
    if indexed && let Value::Inst(id) = result {
        e.ir.replace_opcode(id, O::ImageGradient)?;
    }
    e.ir.replace_uses_with(id, result)
}
