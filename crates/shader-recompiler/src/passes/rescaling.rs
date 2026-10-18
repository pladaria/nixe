use super::fold::{arg, node};
use crate::{
    Result,
    arena::{BlockId, InstId},
    ir::{Attribute, Emitter, Ir, Opcode as O, Program, TextureInstInfo, Value},
    profile::ResolutionScaling,
    types::{Stage, TextureType as T},
};
const MARKED: u32 = 0xdeadbeef;
fn rescalable(ty: T) -> bool {
    matches!(ty, T::Color2D | T::ColorArray2D | T::Color2DRect)
}
fn copy_instruction(e: &mut Emitter<'_>, id: InstId) -> Result<Value> {
    let inst = e.ir.inst(id)?;
    let op = inst.opcode();
    let flags = inst.flags;
    let args = (0..inst.num_args())
        .map(|i| inst.arg(i))
        .collect::<Result<Vec<_>>>()?;
    e.emit(op, &args, flags)
}
fn mark(ir: &mut Ir, block: BlockId, id: InstId) -> Result<()> {
    if !matches!(
        ir.inst(id)?.opcode(),
        O::ShuffleIndex | O::ShuffleUp | O::ShuffleDown | O::ShuffleButterfly
    ) {
        return Ok(());
    }
    let Some(cast) = node(ir, arg(ir, id, 0)?, O::BitCastU32F32)? else {
        return Ok(());
    };
    let Some(attribute) = node(ir, arg(ir, cast, 0)?, O::GetAttribute)? else {
        return Ok(());
    };
    if !matches!(
        arg(ir, attribute, 0)?,
        Value::Attribute(Attribute::PositionX | Attribute::PositionY)
    ) {
        return Ok(());
    }
    ir.inst_mut(attribute)?.flags = MARKED;
    let mut e = Emitter::before(ir, block, id)?;
    let value = copy_instruction(&mut e, id)?;
    let value = e.emit(O::ConvertF32U32, &[value], 0)?;
    let factor = e.emit(O::ResolutionDownFactor, &[], 0)?;
    let factor = e.emit(O::FPRecip32, &[factor], 0)?;
    let value = e.emit(O::FPMul32, &[value, factor], 0)?;
    e.ir.replace_uses_with(id, value)
}
fn scale(
    e: &mut Emitter<'_>,
    value: Value,
    enabled: Value,
    settings: &ResolutionScaling,
    down: bool,
) -> Result<Value> {
    let mut result = value;
    if down {
        if settings.down_shift != 0 {
            result = e.emit(
                O::ShiftLeftLogical32,
                &[result, Value::U32(settings.down_shift)],
                0,
            )?;
        }
        if settings.up_scale != 1 {
            result = e.emit(O::UDiv32, &[result, Value::U32(settings.up_scale)], 0)?;
        }
    } else {
        if settings.up_scale != 1 {
            result = e.i_mul(result, Value::U32(settings.up_scale))?;
        }
        if settings.down_shift != 0 {
            result = e.emit(
                O::ShiftRightArithmetic32,
                &[result, Value::U32(settings.down_shift)],
                0,
            )?;
        }
    }
    e.select(enabled, result, value)
}
fn subscale(
    e: &mut Emitter<'_>,
    value: Value,
    enabled: Value,
    attr: Attribute,
    settings: &ResolutionScaling,
) -> Result<Value> {
    let up = Value::f32(settings.up_factor);
    let down = Value::f32(settings.down_factor);
    let converted = e.emit(O::ConvertF32U32, &[value], 0)?;
    let base = e.emit(O::FPMul32, &[converted, up], 0)?;
    let coord = e.get_attribute(attr)?;
    let coarse = e.emit(O::FPMul32, &[coord, down], 0)?;
    let floor = e.emit(O::FPFloor32, &[coarse], 0)?;
    let floor = e.emit(O::FPMul32, &[up, floor], 0)?;
    let negative = e.emit(O::FPNeg32, &[floor], 0)?;
    let deviation = e.emit(O::FPAdd32, &[coord, negative], 0)?;
    let deviation = e.emit(O::FPAdd32, &[base, deviation], 0)?;
    let converted = e.emit(O::ConvertU32F32, &[deviation], 0)?;
    e.select(enabled, converted, value)
}
fn composite(
    e: &mut Emitter<'_>,
    id: InstId,
    index: usize,
    enabled: Value,
    settings: &ResolutionScaling,
    fragment: bool,
) -> Result<()> {
    let value = e.ir.inst(id)?.arg(index)?;
    if value == Value::None {
        return Ok(());
    }
    let ty = TextureInstInfo(e.ir.inst(id)?.flags).texture_type();
    let mut components = Vec::with_capacity(3);
    for component in 0..2 {
        let original = e.composite_extract(value, component)?;
        let scaled = if fragment && index == 1 {
            subscale(
                e,
                original,
                enabled,
                if component == 0 {
                    Attribute::PositionX
                } else {
                    Attribute::PositionY
                },
                settings,
            )?
        } else {
            scale(e, original, enabled, settings, false)?
        };
        components.push(scaled);
    }
    if index == 1 && ty == T::ColorArray2D {
        components.push(e.composite_extract(value, 2)?);
    }
    let result = e.composite_construct(&components)?;
    e.ir.set_arg(id, index, result)
}
fn visit(
    ir: &mut Ir,
    block: BlockId,
    id: InstId,
    fragment: bool,
    settings: &ResolutionScaling,
) -> Result<()> {
    let op = ir.inst(id)?.opcode();
    match op {
        O::GetAttribute
            if fragment
                && ir.inst(id)?.flags != MARKED
                && matches!(
                    arg(ir, id, 0)?,
                    Value::Attribute(Attribute::PositionX | Attribute::PositionY)
                ) =>
        {
            let Value::Attribute(attribute) = ir.inst(id)?.arg(0)? else {
                unreachable!()
            };
            let mut e = Emitter::before(ir, block, id)?;
            let factor = e.emit(O::ResolutionDownFactor, &[], 0)?;
            let coord = e.get_attribute(attribute)?;
            let scaled = e.emit(O::FPMul32, &[coord, factor], 0)?;
            e.ir.replace_uses_with(id, scaled)?;
        }
        O::SetAttribute
            if ir.inst(id)?.flags != MARKED
                && arg(ir, id, 0)? == Value::Attribute(Attribute::PointSize) =>
        {
            let value = ir.inst(id)?.arg(1)?;
            let mut e = Emitter::before(ir, block, id)?;
            let factor = e.emit(O::ResolutionDownFactor, &[], 0)?;
            let factor = e.emit(O::FPRecip32, &[factor], 0)?;
            let value = e.emit(O::FPMul32, &[value, factor], 0)?;
            e.ir.set_arg(id, 1, value)?;
        }
        O::ImageQueryDimensions => {
            let flags = TextureInstInfo(ir.inst(id)?.flags);
            let mut e = Emitter::before(ir, block, id)?;
            let enabled = e.emit(
                O::IsTextureScaled,
                &[Value::U32(flags.descriptor_index())],
                0,
            )?;
            if rescalable(flags.texture_type()) {
                let query = copy_instruction(&mut e, id)?;
                let mut values = [Value::None; 4];
                for (i, value) in values.iter_mut().enumerate() {
                    let component = e.composite_extract(query, i)?;
                    *value = if i < 2 {
                        scale(&mut e, component, enabled, settings, true)?
                    } else {
                        component
                    };
                }
                let result = e.composite_construct(&values)?;
                e.ir.replace_uses_with(id, result)?;
            }
        }
        O::ImageFetch | O::ImageRead => {
            let flags = TextureInstInfo(ir.inst(id)?.flags);
            if !rescalable(flags.texture_type()) {
                return Ok(());
            }
            let mut e = Emitter::before(ir, block, id)?;
            let enabled = e.emit(
                if op == O::ImageFetch {
                    O::IsTextureScaled
                } else {
                    O::IsImageScaled
                },
                &[Value::U32(flags.descriptor_index())],
                0,
            )?;
            composite(&mut e, id, 1, enabled, settings, fragment)?;
            if op == O::ImageFetch {
                composite(&mut e, id, 2, enabled, settings, false)?;
            }
        }
        _ => (),
    }
    Ok(())
}
pub fn rescaling(ir: &mut Ir, program: &Program, settings: &ResolutionScaling) -> Result<()> {
    let fragment = program.stage == Stage::Fragment;
    if fragment {
        for &block in &program.post_order {
            for id in ir.block(block)?.instructions().to_vec() {
                mark(ir, block, id)?;
            }
        }
    }
    for &block in &program.post_order {
        for id in ir.block(block)?.instructions().to_vec() {
            visit(ir, block, id, fragment, settings)?;
        }
    }
    Ok(())
}
