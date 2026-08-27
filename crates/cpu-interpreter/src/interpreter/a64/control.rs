use nixe_cpu::{
    decode::{DecodedOpcode, a64::control::Instruction},
    exception::ExceptionKind,
    location::{DecodedInstruction, LocationDescriptor},
    semantics::conditions::Condition,
    semantics::conditions::evaluate_a64,
    state::a64::A64State,
};

use super::{advance, sign_extend, zero_register};
use crate::interpreter::{InstructionStep, InterpreterError};

pub(super) fn execute(
    state: &mut A64State,
    decoded: &DecodedInstruction<DecodedOpcode>,
    instruction: Instruction,
) -> Result<InstructionStep, InterpreterError> {
    let fields = instruction.operands();
    let source = decoded.location;
    let next_pc = source.pc.get().wrapping_add(4);
    match instruction {
        Instruction::Nop(_) => advance(state),
        Instruction::BranchImmediate(_) => {
            state.set_pc(branch_target(source.pc.get(), fields.immediate_26, 26));
        }
        Instruction::BranchLinkImmediate(_) => {
            state.write_x(zero_register(30), next_pc);
            state.set_pc(branch_target(source.pc.get(), fields.immediate_26, 26));
        }
        Instruction::BranchRegister(_) => {
            let branch_kind = fields.branch_register_key;
            if !matches!(branch_kind, 0xd61f_0000 | 0xd63f_0000 | 0xd65f_0000) {
                return Err(super::super::unsupported(decoded));
            }
            let target = state.read_x(zero_register(fields.rn));
            if target & 3 != 0 {
                return Ok(InstructionStep::architectural_exception(
                    source,
                    ExceptionKind::AlignmentFault,
                    None,
                ));
            }
            if branch_kind == 0xd63f_0000 {
                state.write_x(zero_register(30), next_pc);
            }
            state.set_pc(target);
        }
        Instruction::ConditionalBranch(_) => {
            let taken = evaluate_a64(
                Condition::from_encoding(fields.condition),
                state.nzcv().bits(),
            );
            state.set_pc(if taken {
                branch_target(source.pc.get(), fields.immediate_19, 19)
            } else {
                next_pc
            });
        }
        Instruction::CompareBranch(_) => {
            let value = if fields.width_64 {
                state.read_x(zero_register(fields.rd))
            } else {
                u64::from(state.read_w(zero_register(fields.rd)))
            };
            state.set_pc(if (value != 0) == fields.nonzero {
                branch_target(source.pc.get(), fields.immediate_19, 19)
            } else {
                next_pc
            });
        }
        Instruction::TestBranch(_) => {
            let value = state.read_x(zero_register(fields.rd));
            let bit_set = value & (1_u64 << fields.bit_index) != 0;
            state.set_pc(if bit_set == fields.nonzero {
                branch_target(source.pc.get(), u32::from(fields.immediate_14), 14)
            } else {
                next_pc
            });
        }
        Instruction::SupervisorCall(_) => {
            return Ok(InstructionStep::supervisor_call(
                source,
                u32::from(fields.immediate_16),
            ));
        }
        Instruction::Breakpoint(_) => {
            return Ok(architectural_exception(
                source,
                ExceptionKind::Breakpoint,
                fields.immediate_16,
            ));
        }
    }
    Ok(InstructionStep::Continue)
}

fn architectural_exception(
    source: LocationDescriptor,
    kind: ExceptionKind,
    immediate: u16,
) -> InstructionStep {
    InstructionStep::architectural_exception(source, kind, Some(u64::from(immediate)))
}

fn branch_target(pc: u64, immediate: impl Into<u64>, bits: u8) -> u64 {
    pc.wrapping_add_signed(sign_extend(immediate.into(), bits) << 2)
}
