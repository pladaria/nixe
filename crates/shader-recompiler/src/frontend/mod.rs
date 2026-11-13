pub mod asl;
pub mod flow;
pub mod structure;

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
