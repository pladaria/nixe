//! Shared packed-half operand semantics, independent of instruction encoding.
use crate::{
    Result,
    ir::{Emitter, FpControl, Type, Value},
};

pub(crate) fn extract(emit: &mut Emitter<'_>, value: Value, swizzle: u32) -> Result<[Value; 2]> {
    if swizzle == 1 {
        let scalar = emit.bit_cast(Type::F32, value)?;
        return Ok([scalar; 2]);
    }
    let vector = emit.unpack_float2x16(value)?;
    let first = emit.composite_extract(vector, usize::from(swizzle == 3))?;
    let second = if swizzle == 0 {
        emit.composite_extract(vector, 1)?
    } else {
        first
    };
    Ok([first, second])
}

pub(crate) fn promote<const N: usize>(
    emit: &mut Emitter<'_>,
    operands: &mut [[Value; 2]; N],
) -> Result<bool> {
    let ty = emit.ir.value_type(operands[0][0])?;
    let promotion = operands
        .iter()
        .any(|pair| emit.ir.value_type(pair[0]).is_ok_and(|other| other != ty));
    if promotion {
        for pair in operands {
            if emit.ir.value_type(pair[0])? == Type::F16 {
                for value in pair {
                    *value = emit.fp_convert(32, *value, FpControl::default())?;
                }
            }
        }
    }
    Ok(promotion)
}
