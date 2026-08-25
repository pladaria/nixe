//! A32-to-IR translation.

use crate::{
    decode::{
        DecodedOpcode,
        a32::{
            A32Instruction, control::Instruction as ControlInstruction,
            fp_simd::Instruction as FpSimdInstruction, integer::Instruction as IntegerInstruction,
            memory::Instruction as MemoryInstruction, normalize,
        },
    },
    exception::ExceptionKind,
    ir::{
        builder::{BuildError, IrBuilder},
        op::{Condition, FlagOperation, OperationKind, StateRegister},
        terminator::{ControlTarget, Terminator},
        types::IrType,
        value::{Immediate, Operand},
    },
    location::{DecodedInstruction, ExecutionState, LocationDescriptor},
};

use super::{aarch32, block::LiftOutcome};

pub(crate) fn lift(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
) -> LiftOutcome {
    lift_inner(builder, decoded).expect("A32 semantic construction must produce valid IR")
}

fn lift_inner(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
) -> Result<LiftOutcome, BuildError> {
    let normalized = normalize(&decoded.instruction, decoded.encoding);
    let predicate = evaluate_condition(builder, decoded, normalized.condition)?;
    let fallthrough = direct(decoded.location.pc.wrapping_offset(4), ExecutionState::A32);
    match normalized.instruction {
        A32Instruction::Control(instruction) => {
            lift_control(builder, decoded, predicate, fallthrough, instruction)
        }
        A32Instruction::Integer(IntegerInstruction::DataProcessing(instruction)) => {
            aarch32::lift_data_processing(
                builder,
                decoded.location,
                predicate,
                instruction,
                fallthrough,
                false,
            )
        }
        A32Instruction::Integer(IntegerInstruction::Multiply(instruction)) => {
            aarch32::lift_multiply(
                builder,
                decoded.location,
                predicate,
                instruction,
                fallthrough,
                false,
            )
        }
        A32Instruction::Integer(IntegerInstruction::MoveWide { rd, immediate, top }) => {
            aarch32::lift_move_wide(builder, decoded.location, predicate, rd, immediate, top)
        }
        A32Instruction::Memory(MemoryInstruction::Single(transfer)) => {
            aarch32::lift_single_transfer(
                builder,
                decoded.location,
                predicate,
                transfer,
                fallthrough,
            )
        }
        A32Instruction::Memory(MemoryInstruction::Multiple(transfer)) => {
            aarch32::lift_multiple_transfer(
                builder,
                decoded.location,
                predicate,
                transfer,
                fallthrough,
            )
        }
        A32Instruction::Memory(MemoryInstruction::Exclusive(transfer)) => {
            aarch32::lift_exclusive_transfer(builder, decoded.location, predicate, transfer)
        }
        A32Instruction::Memory(MemoryInstruction::AcquireRelease(transfer)) => {
            aarch32::lift_acquire_release_transfer(builder, decoded.location, predicate, transfer)
        }
        A32Instruction::FpSimd(FpSimdInstruction::Data(data)) => {
            aarch32::lift_vector_data(builder, decoded.location, predicate, data)
        }
        A32Instruction::FpSimd(FpSimdInstruction::Memory(transfer)) => {
            aarch32::lift_vector_transfer(builder, decoded.location, predicate, transfer)
        }
    }
}

fn lift_control(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    predicate: Operand,
    fallthrough: ControlTarget,
    instruction: ControlInstruction,
) -> Result<LiftOutcome, BuildError> {
    let source = decoded.location;
    Ok(match instruction {
        ControlInstruction::Nop => LiftOutcome::Continue,
        ControlInstruction::Branch { link, displacement } => {
            if link {
                write_link(
                    builder,
                    source,
                    predicate,
                    source.pc.get().wrapping_add(4) as u32,
                )?;
            }
            let target = direct(
                nixe_memory::GuestVirtualAddress::new(u64::from(
                    (source.pc.get() as u32)
                        .wrapping_add(8)
                        .wrapping_add_signed(displacement),
                )),
                ExecutionState::A32,
            );
            conditional_control(
                predicate,
                target,
                fallthrough,
                link,
                source.pc.wrapping_offset(4),
            )
        }
        ControlInstruction::Exchange { link, rm } => {
            if link {
                write_link(
                    builder,
                    source,
                    predicate,
                    source.pc.get().wrapping_add(4) as u32,
                )?;
            }
            let bits = aarch32::read_register(builder, source, rm, false)?;
            let address = builder
                .emit(
                    source,
                    &[IrType::Address],
                    OperationKind::Address(crate::ir::op::AddressOperation::FromInteger {
                        value: bits,
                        width: crate::ir::op::GuestAddressWidth::Bits32,
                    }),
                )?
                .iter()
                .next()
                .unwrap();
            conditional_control(
                predicate,
                ControlTarget::A32Interworking {
                    address: address.into(),
                },
                fallthrough,
                link,
                source.pc.wrapping_offset(4),
            )
        }
        ControlInstruction::BlxImmediate { displacement } => {
            write_link(
                builder,
                source,
                predicate,
                source.pc.get().wrapping_add(4) as u32,
            )?;
            let target = nixe_memory::GuestVirtualAddress::new(u64::from(
                (source.pc.get() as u32)
                    .wrapping_add(8)
                    .wrapping_add_signed(displacement)
                    & !1,
            ));
            conditional_control(
                predicate,
                direct(target, ExecutionState::T32),
                fallthrough,
                true,
                source.pc.wrapping_offset(4),
            )
        }
        ControlInstruction::Svc { immediate } => conditional_exception(
            predicate,
            source,
            ExceptionKind::SupervisorCall,
            Some(u64::from(immediate)),
            fallthrough,
        ),
        ControlInstruction::Breakpoint { immediate } => conditional_exception(
            predicate,
            source,
            ExceptionKind::Breakpoint,
            Some(u64::from(immediate)),
            fallthrough,
        ),
    })
}

fn write_link(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    predicate: Operand,
    value: u32,
) -> Result<(), BuildError> {
    let _ = aarch32::write_register(builder, source, 14, Immediate::I32(value).into(), predicate)?;
    Ok(())
}

fn conditional_control(
    predicate: Operand,
    target: ControlTarget,
    fallthrough: ControlTarget,
    call: bool,
    return_address: nixe_memory::GuestVirtualAddress,
) -> LiftOutcome {
    if predicate == Operand::Immediate(Immediate::I1(true)) {
        return if call {
            LiftOutcome::Terminate(Terminator::Call {
                target,
                return_address,
            })
        } else {
            LiftOutcome::Terminate(Terminator::Direct { target })
        };
    }
    if call {
        LiftOutcome::Terminate(Terminator::ConditionalCall {
            condition: predicate,
            target,
            fallthrough,
            return_address,
        })
    } else {
        LiftOutcome::Terminate(Terminator::Conditional {
            condition: predicate,
            taken: target,
            fallthrough,
        })
    }
}

fn conditional_exception(
    predicate: Operand,
    source: LocationDescriptor,
    kind: ExceptionKind,
    syndrome: Option<u64>,
    fallthrough: ControlTarget,
) -> LiftOutcome {
    if predicate == Operand::Immediate(Immediate::I1(true)) {
        return LiftOutcome::Terminate(Terminator::Exception {
            source,
            kind,
            syndrome,
        });
    }
    LiftOutcome::Terminate(Terminator::ConditionalException {
        condition: predicate,
        source,
        kind,
        syndrome,
        fallthrough,
    })
}

fn direct(pc: nixe_memory::GuestVirtualAddress, execution_state: ExecutionState) -> ControlTarget {
    ControlTarget::Direct {
        pc,
        execution_state,
    }
}

fn evaluate_condition(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    condition: Condition,
) -> Result<Operand, BuildError> {
    if condition == Condition::Al {
        return Ok(Immediate::I1(true).into());
    }
    if condition == Condition::Nv {
        return Ok(Immediate::I1(false).into());
    }
    let packed = builder
        .emit(
            decoded.location,
            &[IrType::I32],
            OperationKind::ReadState(StateRegister::A32Cpsr),
        )?
        .iter()
        .next()
        .unwrap();
    let flags = builder
        .emit(
            decoded.location,
            &[IrType::Flags],
            OperationKind::Flags(FlagOperation::FromPacked {
                value: packed.into(),
            }),
        )?
        .iter()
        .next()
        .unwrap();
    Ok(builder
        .emit(
            decoded.location,
            &[IrType::I1],
            OperationKind::Flags(FlagOperation::Evaluate {
                flags: flags.into(),
                condition,
            }),
        )?
        .iter()
        .next()
        .unwrap()
        .into())
}
