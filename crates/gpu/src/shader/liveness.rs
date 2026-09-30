//! Backward liveness for straight-line programs with predicated instructions.
//! Undefined false-path values stay undefined; conditional writes cannot kill
//! a reaching definition needed on the other path. No speculative GPU work is
//! introduced to implement dead guest arithmetic (notably interpolation setup).
use super::*;

pub(super) fn live_instructions(
    ir: &ShaderIr,
) -> Result<Vec<bool>, (ShaderSourceLocation, &'static str)> {
    let mut end = ir.instructions.len();
    for (index, instruction) in ir.instructions.iter().enumerate() {
        if instruction.predicate == ShaderPredicate::Never {
            continue;
        }
        let reason = match instruction.operation {
            ShaderOperation::Branch { .. } => {
                Some("guest control flow requires CFG structurization")
            }
            ShaderOperation::Exit if instruction.predicate != ShaderPredicate::Always => {
                Some("conditional exit requires CFG structurization")
            }
            ShaderOperation::Exit => {
                end = index + 1;
                break;
            }
            _ => None,
        };
        if let Some(reason) = reason {
            return Err((instruction.source, reason));
        }
    }
    let mut keep = vec![false; end];
    let mut registers = Vec::<bool>::new();
    let mut predicates = 0_u8;
    for (index, instruction) in ir.instructions[..end].iter().enumerate().rev() {
        if instruction.predicate == ShaderPredicate::Never {
            continue;
        }
        let mut live_definition = false;
        instruction
            .operation
            .visit_destination_registers(|register| {
                live_definition |= registers
                    .get(usize::from(register.index()))
                    .copied()
                    .unwrap_or(false);
            });
        let predicate_definitions = match &instruction.operation {
            ShaderOperation::SetPredicateInteger32 { destinations, .. } => destinations
                .iter()
                .flatten()
                .fold(0, |mask, p| mask | (1 << p)),
            ShaderOperation::SetPredicateFloat32 { destination, .. } => 1 << destination,
            _ => 0,
        };
        // Exhaustive classification: adding a new store/atomic operation must
        // not silently classify it as removable. There is no shader FP status.
        let observable = match instruction.operation {
            ShaderOperation::StoreOutput { .. }
            | ShaderOperation::StoreControlPoint { .. }
            | ShaderOperation::PatchBarrier
            | ShaderOperation::Exit => true,
            ShaderOperation::Undefined32 { .. }
            | ShaderOperation::MoveImmediate32 { .. }
            | ShaderOperation::Move32 { .. }
            | ShaderOperation::FloatAbsolute32 { .. }
            | ShaderOperation::FloatNegate32 { .. }
            | ShaderOperation::ConvertIntegerToFloat32 { .. }
            | ShaderOperation::RoundFloat32ToIntegral { .. }
            | ShaderOperation::ConvertFloat32ToInteger { .. }
            | ShaderOperation::LoadInput { .. }
            | ShaderOperation::LoadControlPoint { .. }
            | ShaderOperation::LoadPatchOutput { .. }
            | ShaderOperation::Multiply32 { .. }
            | ShaderOperation::FloatMultiplyZero32 { .. }
            | ShaderOperation::Add32 { .. }
            | ShaderOperation::ShiftLeft32 { .. }
            | ShaderOperation::Bitwise32 { .. }
            | ShaderOperation::FloatMinMax32 { .. }
            | ShaderOperation::FusedMultiplyAdd32 { .. }
            | ShaderOperation::Reciprocal32 { .. }
            | ShaderOperation::ReciprocalSqrt32 { .. }
            | ShaderOperation::SpecialFunction32 { .. }
            | ShaderOperation::SetPredicateFloat32 { .. }
            | ShaderOperation::SetPredicateInteger32 { .. }
            | ShaderOperation::InterpolateInput { .. }
            | ShaderOperation::LoadConstantBuffer32 { .. }
            | ShaderOperation::LoadConstantBufferIndexed32 { .. }
            | ShaderOperation::SampleTexture2D { .. }
            | ShaderOperation::LoadTexture2D { .. }
            | ShaderOperation::SampleTexture2DArray { .. } => false,
            ShaderOperation::Branch { .. } => unreachable!("branches rejected before liveness"),
        };
        if !observable && !live_definition && predicates & predicate_definitions == 0 {
            continue;
        }
        keep[index] = true;
        if instruction.predicate == ShaderPredicate::Always {
            instruction
                .operation
                .visit_destination_registers(|register| {
                    if let Some(live) = registers.get_mut(usize::from(register.index())) {
                        *live = false;
                    }
                });
            predicates &= !predicate_definitions;
        }
        for register in instruction.operation.source_registers() {
            let index = usize::from(register.index());
            if index >= registers.len() {
                registers.resize(index + 1, false);
            }
            registers[index] = true;
        }
        let mut use_predicate = |predicate| {
            if let ShaderPredicate::Register { register, .. } = predicate {
                predicates |= 1 << register;
            }
        };
        use_predicate(instruction.predicate);
        match instruction.operation {
            ShaderOperation::SetPredicateInteger32 { accumulator, .. }
            | ShaderOperation::SetPredicateFloat32 { accumulator, .. } => {
                use_predicate(accumulator)
            }
            ShaderOperation::FloatMinMax32 { minimum, .. } => use_predicate(minimum),
            _ => {}
        }
    }
    Ok(keep)
}
