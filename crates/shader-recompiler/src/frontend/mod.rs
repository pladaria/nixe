pub mod asl;
pub mod flow;
pub(crate) mod half;
pub(crate) mod lut;
pub mod structure;

/// LOD queries return unclamped and clamped levels as signed/unsigned 8.8
/// fixed point. The host query orders those components in the opposite order.
// https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/shader_recompiler/frontend/maxwell/translate/impl/texture_mipmap_level.cpp#L95-L106
pub(crate) fn texture_lod_component(
    emit: &mut crate::ir::Emitter<'_>,
    query: crate::ir::Value,
    component: usize,
) -> crate::Result<crate::ir::Value> {
    use crate::ir::{FpControl, Value};
    let value = emit.composite_extract(query, component ^ 1)?;
    let scaled = emit.fp_mul(value, Value::f32(256.0), FpControl::default())?;
    let fixed = emit.convert_f_to_i(32, component == 0, scaled)?;
    emit.bitwise_and(fixed, Value::U32(0xffff))
}

/// Fragment export register layout is shared by the graphics header versions.
pub(crate) fn export_fragment(
    emit: &mut crate::ir::Emitter<'_>,
    metadata: &crate::environment::EnvironmentMetadata,
) -> crate::Result<()> {
    use crate::ir::{Reg, Type};
    if metadata.stage != crate::Stage::Fragment {
        return Ok(());
    }
    let mut reg = Reg(0);
    for target in 0..8 {
        if !metadata.sph.has_output_components(target) {
            continue;
        }
        for (component, enabled) in metadata
            .sph
            .enabled_output_components(target)
            .into_iter()
            .enumerate()
        {
            if enabled {
                let value = emit.get_reg(reg)?;
                let value = emit.bit_cast(Type::F32, value)?;
                emit.set_frag_color(target as u32, component as u32, value)?;
            }
            reg = reg.offset(1)?;
        }
    }
    if metadata.sph.fragment_sample_mask() {
        let value = emit.get_reg(reg)?;
        emit.set_sample_mask(value)?;
    }
    if metadata.sph.fragment_depth() {
        let value = emit.get_reg(reg.offset(1)?)?;
        let value = emit.bit_cast(Type::F32, value)?;
        emit.set_frag_depth(value)?;
    }
    Ok(())
}

pub(crate) fn float_compare(
    emit: &mut crate::ir::Emitter<'_>,
    a: crate::ir::Value,
    b: crate::ir::Value,
    op: u32,
    control: crate::ir::FpControl,
) -> crate::Result<crate::ir::Value> {
    let ordered = !matches!(op, 9..=14);
    match op {
        0 => Ok(crate::ir::Value::U1(false)),
        1 | 9 => emit.fp_less_than(a, b, control, ordered),
        2 | 10 => emit.fp_equal(a, b, control, ordered),
        3 | 11 => emit.fp_less_than_equal(a, b, control, ordered),
        4 | 12 => emit.fp_greater_than(a, b, control, ordered),
        5 | 13 => emit.fp_not_equal(a, b, control, ordered),
        6 | 14 => emit.fp_greater_than_equal(a, b, control, ordered),
        7 => emit.fp_ordered(a, b),
        8 => emit.fp_unordered(a, b),
        15 => Ok(crate::ir::Value::U1(true)),
        _ => Err(crate::CompileError::new(
            crate::diagnostics::FailureCategory::InvalidArgument,
            "Invalid floating comparison",
        )),
    }
}

pub(crate) fn integer_compare(
    emit: &mut crate::ir::Emitter<'_>,
    a: crate::ir::Value,
    b: crate::ir::Value,
    op: u32,
    signed: bool,
) -> crate::Result<crate::ir::Value> {
    match op {
        0 => Ok(crate::ir::Value::U1(false)),
        1 => emit.i_less_than(a, b, signed),
        2 => emit.i_equal(a, b),
        3 => emit.i_less_than_equal(a, b, signed),
        4 => emit.i_greater_than(a, b, signed),
        5 => emit.i_not_equal(a, b),
        6 => emit.i_greater_than_equal(a, b, signed),
        7 => Ok(crate::ir::Value::U1(true)),
        _ => Err(crate::CompileError::new(
            crate::diagnostics::FailureCategory::InvalidArgument,
            "Invalid integer comparison",
        )),
    }
}

/// Preserve fragment quad exchanges, including helper invocations, instead of
/// expressing derivative shuffles as general active-lane subgroup operations.
pub(crate) fn shuffle(
    emit: &mut crate::ir::Emitter<'_>,
    stage: crate::Stage,
    mode: u32,
    input: crate::ir::Value,
    index: crate::ir::Value,
    mask: crate::ir::Value,
) -> crate::Result<(crate::ir::Value, crate::ir::Value)> {
    use crate::ir::Value;
    if stage == crate::Stage::Fragment
        && mask == Value::U32((28 << 8) | 3)
        && let Value::U32(index) = index
    {
        let quad = match (mode, index) {
            (0, 0..=3) => Some(emit.quad_broadcast(input, Value::U32(index))?),
            (3, 1..=3) => Some(emit.quad_swap(input, Value::U32(index - 1))?),
            _ => None,
        };
        if let Some(value) = quad {
            return Ok((value, Value::U1(true)));
        }
    }
    let (clamp, segment) = if let Value::U32(mask) = mask {
        (Value::U32(mask & 31), Value::U32((mask >> 8) & 31))
    } else {
        (
            emit.bit_field_extract(mask, Value::U32(0), Value::U32(5), false)?,
            emit.bit_field_extract(mask, Value::U32(8), Value::U32(5), false)?,
        )
    };
    let value = match mode {
        0 => emit.shuffle_index(input, index, clamp, segment)?,
        1 => emit.shuffle_up(input, index, clamp, segment)?,
        2 => emit.shuffle_down(input, index, clamp, segment)?,
        _ => emit.shuffle_butterfly(input, index, clamp, segment)?,
    };
    let in_bounds = emit.get_in_bounds_from_op(value)?;
    Ok((value, in_bounds))
}
