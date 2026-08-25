//! T32-to-IR translation.

use crate::{
    decode::{
        DecodedOpcode,
        t32::{
            T32Instruction, control::Instruction as ControlInstruction,
            integer::Instruction as IntegerInstruction, memory::Instruction as MemoryInstruction,
            normalize,
        },
    },
    exception::ExceptionKind,
    ir::{
        builder::{BuildError, IrBuilder},
        op::{
            FlagOperation, IntegerBinaryKind, IntegerPredicate, OperationKind, ScalarOperation,
            ShiftKind, StateRegister,
        },
        terminator::{ControlTarget, Terminator},
        types::IrType,
        value::{Immediate, Operand, Value},
    },
    location::{DecodedInstruction, ExecutionState, LocationDescriptor},
    state::a32::{Cpsr, ItState},
};

use super::{aarch32, block::LiftOutcome};

pub(crate) fn lift(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
) -> LiftOutcome {
    lift_inner(builder, decoded).expect("T32 semantic construction must produce valid IR")
}

fn lift_inner(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
) -> Result<LiftOutcome, BuildError> {
    let instruction = normalize(&decoded.instruction, decoded.encoding);
    if let T32Instruction::Control(ControlInstruction::It {
        first_condition,
        mask,
    }) = instruction
    {
        return lift_it(builder, decoded, first_condition, mask);
    }
    let predicate = current_it_condition(builder, decoded.location)?;
    let fallthrough = ControlTarget::Direct {
        pc: decoded
            .location
            .pc
            .wrapping_offset(i64::from(decoded.encoding.size().bytes())),
        execution_state: ExecutionState::T32,
    };
    match instruction {
        T32Instruction::Control(ControlInstruction::It { .. }) => {
            unreachable!("IT was handled before ordinary predication")
        }
        T32Instruction::Control(instruction) => {
            lift_control(builder, decoded, predicate, fallthrough, instruction)
        }
        T32Instruction::Integer(IntegerInstruction::DataProcessing(instruction)) => {
            aarch32::lift_data_processing(
                builder,
                decoded.location,
                predicate,
                instruction,
                fallthrough,
                true,
            )
        }
        T32Instruction::Integer(IntegerInstruction::Multiply(instruction)) => {
            aarch32::lift_multiply(
                builder,
                decoded.location,
                predicate,
                instruction,
                fallthrough,
                true,
            )
        }
        T32Instruction::Integer(IntegerInstruction::MoveWide { rd, immediate, top }) => {
            aarch32::lift_move_wide(builder, decoded.location, predicate, rd, immediate, top)
        }
        T32Instruction::Memory(MemoryInstruction::Single(transfer)) => {
            aarch32::lift_single_transfer(
                builder,
                decoded.location,
                predicate,
                transfer,
                fallthrough,
            )
        }
        T32Instruction::Memory(MemoryInstruction::Multiple(transfer)) => {
            aarch32::lift_multiple_transfer(
                builder,
                decoded.location,
                predicate,
                transfer,
                fallthrough,
            )
        }
    }
}

fn lift_control(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    mut predicate: Operand,
    fallthrough: ControlTarget,
    instruction: ControlInstruction,
) -> Result<LiftOutcome, BuildError> {
    let source = decoded.location;
    Ok(match instruction {
        ControlInstruction::Nop => LiftOutcome::Continue,
        ControlInstruction::Hint { .. } => {
            unreachable!("recognized-unimplemented T32 hints never reach the lifter")
        }
        ControlInstruction::It { .. } => unreachable!("IT was handled before predication"),
        ControlInstruction::Branch {
            condition,
            displacement,
        } => {
            if let Some(condition) = condition {
                let branch_condition = evaluate_encoded_condition(builder, source, condition)?;
                predicate = emit_one(
                    builder,
                    source,
                    IrType::I1,
                    OperationKind::Scalar(ScalarOperation::Binary {
                        kind: IntegerBinaryKind::And,
                        lhs: predicate,
                        rhs: branch_condition,
                    }),
                )?
                .into();
            }
            let target = ControlTarget::Direct {
                pc: nixe_memory::GuestVirtualAddress::new(u64::from(
                    (source.pc.get() as u32)
                        .wrapping_add(4)
                        .wrapping_add_signed(displacement),
                )),
                execution_state: ExecutionState::T32,
            };
            LiftOutcome::Terminate(Terminator::Conditional {
                condition: predicate,
                taken: target,
                fallthrough,
            })
        }
        ControlInstruction::Exchange { link, rm } => {
            if link {
                let _ = aarch32::write_register(
                    builder,
                    source,
                    14,
                    Immediate::I32((source.pc.get().wrapping_add(2) as u32) | 1).into(),
                    predicate,
                )?;
            }
            let bits = aarch32::read_register(builder, source, rm, false)?;
            let address = emit_one(
                builder,
                source,
                IrType::Address,
                OperationKind::Address(crate::ir::op::AddressOperation::FromInteger {
                    value: bits,
                    width: crate::ir::op::GuestAddressWidth::Bits32,
                }),
            )?;
            let target = ControlTarget::A32Interworking {
                address: address.into(),
            };
            if link {
                LiftOutcome::Terminate(Terminator::ConditionalCall {
                    condition: predicate,
                    target,
                    fallthrough,
                    return_address: source.pc.wrapping_offset(2),
                })
            } else {
                LiftOutcome::Terminate(Terminator::Conditional {
                    condition: predicate,
                    taken: target,
                    fallthrough,
                })
            }
        }
        ControlInstruction::BranchLink { displacement } => {
            let return_address = source.pc.wrapping_offset(4);
            let _ = aarch32::write_register(
                builder,
                source,
                14,
                Immediate::I32((return_address.get() as u32) | 1).into(),
                predicate,
            )?;
            let target = ControlTarget::Direct {
                pc: nixe_memory::GuestVirtualAddress::new(u64::from(
                    (source.pc.get() as u32)
                        .wrapping_add(4)
                        .wrapping_add_signed(displacement),
                )),
                execution_state: ExecutionState::T32,
            };
            LiftOutcome::Terminate(Terminator::ConditionalCall {
                condition: predicate,
                target,
                fallthrough,
                return_address,
            })
        }
        ControlInstruction::Svc { immediate } => {
            LiftOutcome::Terminate(Terminator::ConditionalException {
                condition: predicate,
                source,
                kind: ExceptionKind::SupervisorCall,
                syndrome: Some(u64::from(immediate)),
                fallthrough,
            })
        }
        ControlInstruction::Breakpoint { immediate } => {
            LiftOutcome::Terminate(Terminator::ConditionalException {
                condition: predicate,
                source,
                kind: ExceptionKind::Breakpoint,
                syndrome: Some(u64::from(immediate)),
                fallthrough,
            })
        }
    })
}

fn evaluate_encoded_condition(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    condition: u8,
) -> Result<Operand, BuildError> {
    let cpsr = read_cpsr(builder, source)?;
    let flags = emit_one(
        builder,
        source,
        IrType::Flags,
        OperationKind::Flags(FlagOperation::FromPacked { value: cpsr.into() }),
    )?;
    Ok(emit_one(
        builder,
        source,
        IrType::I1,
        OperationKind::Flags(FlagOperation::Evaluate {
            flags: flags.into(),
            condition: crate::ir::op::Condition::from_encoding(condition),
        }),
    )?
    .into())
}

fn lift_it(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    first_condition: u8,
    mask: u8,
) -> Result<LiftOutcome, BuildError> {
    let it_state = ItState::from_encoding(first_condition, mask)
        .expect("allocation validation rejects reserved IT encodings");

    let cpsr = read_cpsr(builder, decoded.location)?;
    let packed = pack_it_state(
        builder,
        decoded.location,
        cpsr.into(),
        Immediate::I32(u32::from(it_state.bits())).into(),
    )?;
    write_cpsr(builder, decoded.location, packed.into())?;
    Ok(LiftOutcome::Continue)
}

/// Returns the current IT predicate and advances ITSTATE before any observable
/// effect of the predicated instruction is emitted.
fn current_it_condition(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
) -> Result<Operand, BuildError> {
    let cpsr = read_cpsr(builder, source)?;
    let it_state = unpack_it_state(builder, source, cpsr.into())?;
    let encoded = scalar(
        builder,
        source,
        ScalarOperation::Shift {
            kind: ShiftKind::LogicalRight,
            value: it_state.into(),
            amount: Immediate::I32(4).into(),
        },
    )?;
    let inactive = emit_one(
        builder,
        source,
        IrType::I1,
        OperationKind::Scalar(ScalarOperation::Compare {
            predicate: IntegerPredicate::Equal,
            lhs: it_state.into(),
            rhs: Immediate::I32(0).into(),
        }),
    )?;
    let condition = emit_one(
        builder,
        source,
        IrType::I32,
        OperationKind::Scalar(ScalarOperation::Select {
            condition: inactive.into(),
            when_true: Immediate::I32(14).into(),
            when_false: encoded.into(),
        }),
    )?;
    let flags = emit_one(
        builder,
        source,
        IrType::Flags,
        OperationKind::Flags(FlagOperation::FromPacked { value: cpsr.into() }),
    )?;
    let predicate = emit_one(
        builder,
        source,
        IrType::I1,
        OperationKind::Flags(FlagOperation::EvaluateEncoded {
            flags: flags.into(),
            condition: condition.into(),
            nv_is_unconditional: false,
        }),
    )?;
    let advanced = advance_it_value(builder, source, it_state.into())?;
    let packed = pack_it_state(builder, source, cpsr.into(), advanced.into())?;
    write_cpsr(builder, source, packed.into())?;
    Ok(predicate.into())
}

fn advance_it_value(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    it_state: Operand,
) -> Result<Value, BuildError> {
    let low_three = scalar_binary(
        builder,
        source,
        IntegerBinaryKind::And,
        it_state,
        Immediate::I32(7).into(),
    )?;
    let is_last = emit_one(
        builder,
        source,
        IrType::I1,
        OperationKind::Scalar(ScalarOperation::Compare {
            predicate: IntegerPredicate::Equal,
            lhs: low_three.into(),
            rhs: Immediate::I32(0).into(),
        }),
    )?;
    let top = scalar_binary(
        builder,
        source,
        IntegerBinaryKind::And,
        it_state,
        Immediate::I32(0xe0).into(),
    )?;
    let shifted = scalar(
        builder,
        source,
        ScalarOperation::Shift {
            kind: ShiftKind::LogicalLeft,
            value: it_state,
            amount: Immediate::I32(1).into(),
        },
    )?;
    let low = scalar_binary(
        builder,
        source,
        IntegerBinaryKind::And,
        shifted.into(),
        Immediate::I32(0x1f).into(),
    )?;
    let next = scalar_binary(
        builder,
        source,
        IntegerBinaryKind::Or,
        top.into(),
        low.into(),
    )?;
    emit_one(
        builder,
        source,
        IrType::I32,
        OperationKind::Scalar(ScalarOperation::Select {
            condition: is_last.into(),
            when_true: Immediate::I32(0).into(),
            when_false: next.into(),
        }),
    )
}

fn unpack_it_state(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    cpsr: Operand,
) -> Result<Value, BuildError> {
    let low = scalar(
        builder,
        source,
        ScalarOperation::Shift {
            kind: ShiftKind::LogicalRight,
            value: cpsr,
            amount: Immediate::I32(25).into(),
        },
    )?;
    let low = scalar_binary(
        builder,
        source,
        IntegerBinaryKind::And,
        low.into(),
        Immediate::I32(3).into(),
    )?;
    let high = scalar(
        builder,
        source,
        ScalarOperation::Shift {
            kind: ShiftKind::LogicalRight,
            value: cpsr,
            amount: Immediate::I32(10).into(),
        },
    )?;
    let high = scalar_binary(
        builder,
        source,
        IntegerBinaryKind::And,
        high.into(),
        Immediate::I32(0x3f).into(),
    )?;
    let high = scalar(
        builder,
        source,
        ScalarOperation::Shift {
            kind: ShiftKind::LogicalLeft,
            value: high.into(),
            amount: Immediate::I32(2).into(),
        },
    )?;
    scalar_binary(
        builder,
        source,
        IntegerBinaryKind::Or,
        high.into(),
        low.into(),
    )
}

fn pack_it_state(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    cpsr: Operand,
    it_state: Operand,
) -> Result<Value, BuildError> {
    let cleared = scalar_binary(
        builder,
        source,
        IntegerBinaryKind::And,
        cpsr,
        Immediate::I32(!Cpsr::IT_MASK).into(),
    )?;
    let low = scalar_binary(
        builder,
        source,
        IntegerBinaryKind::And,
        it_state,
        Immediate::I32(3).into(),
    )?;
    let low = scalar(
        builder,
        source,
        ScalarOperation::Shift {
            kind: ShiftKind::LogicalLeft,
            value: low.into(),
            amount: Immediate::I32(25).into(),
        },
    )?;
    let high = scalar(
        builder,
        source,
        ScalarOperation::Shift {
            kind: ShiftKind::LogicalRight,
            value: it_state,
            amount: Immediate::I32(2).into(),
        },
    )?;
    let high = scalar(
        builder,
        source,
        ScalarOperation::Shift {
            kind: ShiftKind::LogicalLeft,
            value: high.into(),
            amount: Immediate::I32(10).into(),
        },
    )?;
    let packed = scalar_binary(
        builder,
        source,
        IntegerBinaryKind::Or,
        cleared.into(),
        low.into(),
    )?;
    scalar_binary(
        builder,
        source,
        IntegerBinaryKind::Or,
        packed.into(),
        high.into(),
    )
}

fn read_cpsr(builder: &mut IrBuilder, source: LocationDescriptor) -> Result<Value, BuildError> {
    emit_one(
        builder,
        source,
        IrType::I32,
        OperationKind::ReadState(StateRegister::A32Cpsr),
    )
}

fn write_cpsr(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    value: Operand,
) -> Result<(), BuildError> {
    builder.emit(
        source,
        &[],
        OperationKind::WriteState {
            register: StateRegister::A32Cpsr,
            value,
        },
    )?;
    Ok(())
}

fn scalar(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    operation: ScalarOperation,
) -> Result<Value, BuildError> {
    emit_one(
        builder,
        source,
        IrType::I32,
        OperationKind::Scalar(operation),
    )
}

fn scalar_binary(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    kind: IntegerBinaryKind,
    lhs: Operand,
    rhs: Operand,
) -> Result<Value, BuildError> {
    scalar(builder, source, ScalarOperation::Binary { kind, lhs, rhs })
}

fn emit_one(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    ty: IrType,
    kind: OperationKind,
) -> Result<Value, BuildError> {
    Ok(builder
        .emit(source, &[ty], kind)?
        .iter()
        .next()
        .expect("one result was requested"))
}
