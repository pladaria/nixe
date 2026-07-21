//! A64 reference interpretation.

use crate::{
    address::GuestVirtualAddress,
    decode::{
        DecodedOpcode,
        a64::{A64Instruction, control::Instruction as ControlInstruction},
    },
    ir::{op::Condition, terminator::ExceptionKind},
    location::{DecodedInstruction, ExecutionState, LocationDescriptor},
    semantics::conditions::evaluate_a64,
    state::a64::{A64GeneralRegister, A64Register, A64State},
};

use super::{InterpreterError, InterpreterOutcome};

pub(super) fn execute(
    state: &mut A64State,
    decoded: &DecodedInstruction<DecodedOpcode>,
) -> Result<InterpreterOutcome, InterpreterError> {
    match crate::decode::a64::normalize(&decoded.instruction, decoded.encoding) {
        A64Instruction::Control(instruction) => execute_control(state, decoded, instruction),
        // Scalar integer, memory, system, and FP/SIMD reference handlers are
        // intentionally independent of IR and are added by instruction family.
        A64Instruction::System(_)
        | A64Instruction::Integer(_)
        | A64Instruction::Memory(_)
        | A64Instruction::FpSimd(_)
        | A64Instruction::RecognizedFallback { .. } => Err(super::unsupported(decoded)),
    }
}

fn execute_control(
    state: &mut A64State,
    decoded: &DecodedInstruction<DecodedOpcode>,
    instruction: ControlInstruction,
) -> Result<InterpreterOutcome, InterpreterError> {
    let fields = instruction.operands();
    let source = decoded.location;
    let next_pc = source.pc.get().wrapping_add(4);
    match instruction {
        ControlInstruction::Nop(_) => state.set_pc(next_pc),
        ControlInstruction::BranchImmediate(_) => {
            state.set_pc(branch_target(source.pc.get(), fields.immediate_26, 26));
        }
        ControlInstruction::BranchLinkImmediate(_) => {
            state.write_x(general(30), next_pc);
            state.set_pc(branch_target(source.pc.get(), fields.immediate_26, 26));
        }
        ControlInstruction::BranchRegister(_) => {
            let branch_kind = fields.branch_register_key;
            if !matches!(branch_kind, 0xd61f_0000 | 0xd63f_0000 | 0xd65f_0000) {
                return Err(super::unsupported(decoded));
            }
            let target = state.read_x(general(fields.rn));
            if target & 3 != 0 {
                return Ok(InterpreterOutcome::Exception {
                    source,
                    kind: ExceptionKind::AlignmentFault,
                    syndrome: None,
                });
            }
            if branch_kind == 0xd63f_0000 {
                state.write_x(general(30), next_pc);
            }
            state.set_pc(target);
        }
        ControlInstruction::ConditionalBranch(_) => {
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
        ControlInstruction::CompareBranch(_) => {
            let value = if fields.width_64 {
                state.read_x(general(fields.rd))
            } else {
                u64::from(state.read_w(general(fields.rd)))
            };
            let taken = (value != 0) == fields.nonzero;
            state.set_pc(if taken {
                branch_target(source.pc.get(), fields.immediate_19, 19)
            } else {
                next_pc
            });
        }
        ControlInstruction::TestBranch(_) => {
            let value = state.read_x(general(fields.rd));
            let bit_set = value & (1_u64 << fields.bit_index) != 0;
            state.set_pc(if bit_set == fields.nonzero {
                branch_target(source.pc.get(), u32::from(fields.immediate_14), 14)
            } else {
                next_pc
            });
        }
        ControlInstruction::SupervisorCall(_) => {
            return Ok(InterpreterOutcome::Exception {
                source,
                kind: ExceptionKind::SupervisorCall,
                syndrome: Some(u64::from(fields.immediate_16)),
            });
        }
        ControlInstruction::Breakpoint(_) => {
            return Ok(InterpreterOutcome::Exception {
                source,
                kind: ExceptionKind::Breakpoint,
                syndrome: Some(u64::from(fields.immediate_16)),
            });
        }
    }
    Ok(InterpreterOutcome::Resume(LocationDescriptor::new(
        GuestVirtualAddress::new(state.pc()),
        ExecutionState::A64,
        source.profile_id,
    )))
}

fn general(index: u8) -> A64Register {
    match A64GeneralRegister::new(index) {
        Some(register) => A64Register::General(register),
        None => A64Register::Zero,
    }
}

fn branch_target(pc: u64, immediate: impl Into<u64>, bits: u8) -> u64 {
    let immediate = immediate.into();
    let shift = 64 - bits;
    let displacement = (((immediate << shift) as i64) >> shift) << 2;
    pc.wrapping_add_signed(displacement)
}
