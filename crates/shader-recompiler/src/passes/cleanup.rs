use crate::{
    CompileError, Environment, Result,
    diagnostics::FailureCategory,
    ir::{Attribute, Emitter, FpControl, Ir, Opcode, Program, Type, Value},
    types::Stage,
};
use std::collections::{HashMap, HashSet};

pub fn dead_code_elimination(ir: &mut Ir, program: &Program) -> Result<()> {
    for &block in &program.post_order {
        let mut instructions = std::mem::take(ir.block_mut(block)?.instructions_mut());
        let mut kept = instructions.len();
        for index in (0..instructions.len()).rev() {
            let id = instructions[index];
            let inst = ir.inst(id)?;
            if !inst.has_uses() && !inst.may_have_side_effects() {
                ir.invalidate(id)?;
            } else {
                kept -= 1;
                instructions[kept] = id;
            }
        }
        let count = instructions.len() - kept;
        instructions.copy_within(kept.., 0);
        instructions.truncate(count);
        *ir.block_mut(block)?.instructions_mut() = instructions;
    }
    Ok(())
}
pub fn remove_identities(ir: &mut Ir, program: &Program) -> Result<()> {
    let mut detached = Vec::new();
    for &block in &program.blocks {
        let mut index = 0;
        while let Some(&id) = ir.block(block)?.instructions().get(index) {
            for arg_index in 0..ir.inst(id)?.num_args() {
                let before = ir.inst(id)?.arg(arg_index)?;
                let after = ir.resolve(before)?;
                if before != after {
                    ir.set_arg(id, arg_index, after)?;
                }
            }
            if matches!(ir.inst(id)?.opcode(), Opcode::Identity | Opcode::Void) {
                detached.push(id);
                ir.detach(block, id)?;
            } else {
                index += 1;
            }
        }
    }
    for id in detached {
        ir.invalidate(id)?;
    }
    Ok(())
}
pub fn verify_program(ir: &Ir, program: &Program) -> Result<()> {
    let invalid = |message| CompileError::new(FailureCategory::Internal, message);
    let mut uses = HashMap::new();
    let mut definitions = HashSet::new();
    for &block in &program.blocks {
        let mut non_phi = false;
        for &id in ir.block(block)?.instructions() {
            let inst = ir.inst(id)?;
            definitions.insert(id);
            let phi = inst.opcode() == Opcode::Phi;
            if phi && non_phi {
                return Err(invalid("Interleaved phi nodes"));
            }
            non_phi |= !phi;
            for index in 0..inst.num_args() {
                let value = inst.arg(index)?;
                if !phi {
                    let actual = ir.value_type(value)?;
                    let expected = inst.opcode().signature().arguments[index];
                    if actual != expected && actual != Type::Opaque && expected != Type::Opaque {
                        return Err(invalid("Invalid IR operand type"));
                    }
                }
                if ir.is_immediate(value)? {
                    continue;
                }
                let Value::Inst(producer) = value else {
                    return Err(invalid("Invalid instruction operand"));
                };
                *uses.entry(producer).or_insert(0i32) += 1;
                if !phi && !definitions.contains(&producer) {
                    return Err(invalid("Forward declaration in IR block"));
                }
            }
        }
    }
    for (id, count) in uses {
        if ir.inst(id)?.use_count() != count {
            return Err(invalid("Invalid IR use count"));
        }
    }
    Ok(())
}
pub fn position<E: Environment>(
    environment: &mut E,
    ir: &mut Ir,
    program: &mut Program,
) -> Result<()> {
    if environment.metadata().stage != Stage::VertexB
        || environment.read_viewport_transform_state()? != 0
    {
        return Ok(());
    }
    program.info.uses_render_area = true;
    for &block in &program.post_order {
        let mut index = 0;
        while let Some(&id) = ir.block(block)?.instructions().get(index) {
            let before = ir.block(block)?.instructions().len();
            let inst = ir.inst(id)?;
            if inst.opcode() == Opcode::SetAttribute {
                let attribute = inst.arg(0)?;
                if matches!(
                    attribute,
                    Value::Attribute(Attribute::PositionX | Attribute::PositionY)
                ) {
                    let value = inst.arg(1)?;
                    let mut e = Emitter::at_position(ir, block, index)?;
                    let extent = if attribute == Value::Attribute(Attribute::PositionX) {
                        e.render_area_width()?
                    } else {
                        e.render_area_height()?
                    };
                    let reciprocal = e.fp_recip(extent)?;
                    let scale = e.fp_mul(reciprocal, Value::f32(2.0), FpControl::default())?;
                    let result = e.fp_fma(value, scale, Value::f32(-1.0), FpControl::default())?;
                    e.ir.set_arg(id, 1, result)?;
                }
            }
            index += 1 + ir.block(block)?.instructions().len() - before;
        }
    }
    Ok(())
}
