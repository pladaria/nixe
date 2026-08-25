//! Encoding-independent AArch32-to-IR semantic construction.
//!
//! Predication, transfers, exclusive accesses, and VFP/Advanced SIMD follow
//! the Arm ARM AArch32 instruction definitions (DDI 0602):
//! https://developer.arm.com/documentation/ddi0602/latest/AArch32-Instructions
//! False predicates are represented explicitly so they cannot access memory,
//! alter the local exclusive monitor, or trigger an FP helper exit.

use crate::{
    decode::aarch32::{
        AcquireReleaseTransfer, DataOperation, DataProcessing, ExclusiveTransfer, MemoryOffset,
        MemorySize, MultipleTransfer, Multiply, Shift, ShiftAmount, ShifterOperand, SingleTransfer,
        VectorDataProcessing, VectorOperation as DecodedVectorOperation, VectorSize,
        VectorTransfer,
    },
    ir::{
        builder::{BuildError, IrBuilder},
        op::{
            AddressOperation, ByteOrder, EffectSet, ExclusiveOperation, GuestAddressWidth,
            HelperOperation, IntegerBinaryKind, LaneType, MemoryDescriptor, MemoryOperation,
            MemoryPrivilege, OperationEffects, OperationKind, RegisterIndex, ScalarOperation,
            StateRegister, VectorArrangement, VectorOperation, Volatility,
        },
        terminator::{ControlTarget, Terminator},
        types::IrType,
        value::{Immediate, Operand, Value},
    },
    location::{ExecutionState, LocationDescriptor},
    memory::{MemoryAccess, MemoryAccessClass, MemoryAccessSize, MemoryAlignment, MemoryOrdering},
    state::a32::A32GeneralRegister,
};

use super::block::LiftOutcome;

pub(super) fn read_register(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    index: u8,
    align_pc: bool,
) -> Result<Operand, BuildError> {
    if index == 15 {
        let visible = match source.execution_state {
            ExecutionState::A32 => source.pc.get().wrapping_add(8) as u32,
            ExecutionState::T32 => source.pc.get().wrapping_add(4) as u32,
            ExecutionState::A64 => unreachable!("AArch32 helper received A64 source"),
        };
        return Ok(Immediate::I32(if align_pc { visible & !3 } else { visible }).into());
    }
    Ok(emit_one(
        builder,
        source,
        IrType::I32,
        OperationKind::ReadState(StateRegister::A32R(
            A32GeneralRegister::new(index).expect("normalized AArch32 register"),
        )),
    )?
    .into())
}

pub(super) fn read_cpsr(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
) -> Result<Operand, BuildError> {
    Ok(emit_one(
        builder,
        source,
        IrType::I32,
        OperationKind::ReadState(StateRegister::A32Cpsr),
    )?
    .into())
}

fn write_state(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    register: StateRegister,
    value: Operand,
) -> Result<(), BuildError> {
    builder.emit(source, &[], OperationKind::WriteState { register, value })?;
    Ok(())
}

fn select(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    predicate: Operand,
    when_true: Operand,
    when_false: Operand,
) -> Result<Operand, BuildError> {
    Ok(emit_one(
        builder,
        source,
        when_true.ty(),
        OperationKind::Scalar(ScalarOperation::Select {
            condition: predicate,
            when_true,
            when_false,
        }),
    )?
    .into())
}

pub(super) fn write_register(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    index: u8,
    value: Operand,
    predicate: Operand,
) -> Result<Option<ControlTarget>, BuildError> {
    if index == 15 {
        let address = emit_one(
            builder,
            source,
            IrType::Address,
            OperationKind::Address(AddressOperation::FromInteger {
                value,
                width: GuestAddressWidth::Bits32,
            }),
        )?;
        return Ok(Some(ControlTarget::A32Interworking {
            address: address.into(),
            source,
        }));
    }
    let register = StateRegister::A32R(
        A32GeneralRegister::new(index).expect("normalized AArch32 destination"),
    );
    let old = emit_one(
        builder,
        source,
        IrType::I32,
        OperationKind::ReadState(register),
    )?;
    let selected = select(builder, source, predicate, value, old.into())?;
    write_state(builder, source, register, selected)?;
    Ok(None)
}

pub(super) fn lift_data_processing(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    predicate: Operand,
    instruction: DataProcessing,
    fallthrough: ControlTarget,
    suppress_flags: Operand,
) -> Result<LiftOutcome, BuildError> {
    let lhs = read_register(builder, source, instruction.rn, false)?;
    let old_destination = read_register(builder, source, instruction.rd, false)?;
    let cpsr = read_cpsr(builder, source)?;
    let (operand, shift_kind, shift_amount, shift_by_register, rotation) =
        match instruction.operand2 {
            ShifterOperand::Immediate { value, rotation } => (
                Immediate::I32(value).into(),
                0,
                Immediate::I32(0).into(),
                false,
                rotation,
            ),
            ShifterOperand::Register { rm, shift } => {
                let amount = match shift.amount {
                    ShiftAmount::Immediate(amount) => Immediate::I32(u32::from(amount)).into(),
                    ShiftAmount::Register(rs) => read_register(builder, source, rs, false)?,
                };
                (
                    read_register(builder, source, rm, false)?,
                    shift_kind_code(shift),
                    amount,
                    matches!(shift.amount, ShiftAmount::Register(_)),
                    0,
                )
            }
        };
    let results = helper(
        builder,
        source,
        "aarch32.data-processing",
        vec![
            predicate,
            old_destination,
            lhs,
            operand,
            shift_amount,
            cpsr,
            Immediate::I32(data_operation_code(instruction.operation)).into(),
            Immediate::I32(u32::from(instruction.set_flags)).into(),
            suppress_flags,
            Immediate::I32(u32::from(shift_kind)).into(),
            Immediate::I32(u32::from(shift_by_register)).into(),
            Immediate::I32(u32::from(rotation)).into(),
        ],
        &[IrType::I32, IrType::I32],
        OperationEffects::new(EffectSet::HELPER, false),
    )?;
    write_state(builder, source, StateRegister::A32Cpsr, results[1].into())?;
    if instruction.operation.is_test() {
        return Ok(LiftOutcome::Continue);
    }
    if let Some(target) = write_register(
        builder,
        source,
        instruction.rd,
        results[0].into(),
        predicate,
    )? {
        return Ok(LiftOutcome::Terminate(Terminator::Conditional {
            condition: predicate,
            taken: target,
            fallthrough,
        }));
    }
    Ok(LiftOutcome::Continue)
}

pub(super) fn lift_multiply(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    predicate: Operand,
    instruction: Multiply,
    fallthrough: ControlTarget,
    suppress_flags: Operand,
) -> Result<LiftOutcome, BuildError> {
    let old = read_register(builder, source, instruction.rd, false)?;
    let rm = read_register(builder, source, instruction.rm, false)?;
    let rs = read_register(builder, source, instruction.rs, false)?;
    let addend = if instruction.accumulate {
        read_register(builder, source, instruction.rn, false)?
    } else {
        Immediate::I32(0).into()
    };
    let cpsr = read_cpsr(builder, source)?;
    let results = helper(
        builder,
        source,
        "aarch32.multiply",
        vec![
            predicate,
            old,
            rm,
            rs,
            addend,
            cpsr,
            Immediate::I32(u32::from(instruction.set_flags)).into(),
            suppress_flags,
        ],
        &[IrType::I32, IrType::I32],
        OperationEffects::new(EffectSet::HELPER, false),
    )?;
    write_state(builder, source, StateRegister::A32Cpsr, results[1].into())?;
    if let Some(target) = write_register(
        builder,
        source,
        instruction.rd,
        results[0].into(),
        predicate,
    )? {
        return Ok(LiftOutcome::Terminate(Terminator::Conditional {
            condition: predicate,
            taken: target,
            fallthrough,
        }));
    }
    Ok(LiftOutcome::Continue)
}

pub(super) fn lift_move_wide(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    predicate: Operand,
    rd: u8,
    immediate: u16,
    top: bool,
) -> Result<LiftOutcome, BuildError> {
    let old = read_register(builder, source, rd, false)?;
    let value = if top {
        let low = emit_one(
            builder,
            source,
            IrType::I32,
            OperationKind::Scalar(ScalarOperation::Binary {
                kind: IntegerBinaryKind::And,
                lhs: old,
                rhs: Immediate::I32(0xffff).into(),
            }),
        )?;
        emit_one(
            builder,
            source,
            IrType::I32,
            OperationKind::Scalar(ScalarOperation::Binary {
                kind: IntegerBinaryKind::Or,
                lhs: low.into(),
                rhs: Immediate::I32(u32::from(immediate) << 16).into(),
            }),
        )?
        .into()
    } else {
        Immediate::I32(u32::from(immediate)).into()
    };
    let _ = write_register(builder, source, rd, value, predicate)?;
    Ok(LiftOutcome::Continue)
}

pub(super) fn lift_single_transfer(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    predicate: Operand,
    transfer: SingleTransfer,
    fallthrough: ControlTarget,
) -> Result<LiftOutcome, BuildError> {
    let base = read_register(builder, source, transfer.rn, transfer.rn == 15)?;
    let raw_offset = memory_offset(builder, source, transfer.offset)?;
    let offset = emit_one(
        builder,
        source,
        IrType::I32,
        OperationKind::Scalar(ScalarOperation::Binary {
            kind: if transfer.add {
                IntegerBinaryKind::Add
            } else {
                IntegerBinaryKind::Subtract
            },
            lhs: base,
            rhs: raw_offset,
        }),
    )?;
    let address_bits = if transfer.pre_index {
        offset.into()
    } else {
        base
    };
    let address = address(builder, source, address_bits)?;
    let descriptor = descriptor(
        transfer.size,
        MemoryOrdering::Relaxed,
        MemoryAccessClass::Normal,
    );
    if transfer.load {
        let old = read_register(builder, source, transfer.rt, false)?;
        let fallback = narrow(builder, source, old, descriptor.value_type())?;
        let loaded = emit_one(
            builder,
            source,
            descriptor.value_type(),
            OperationKind::Memory(MemoryOperation::GuardedLoad {
                predicate,
                address,
                fallback,
                descriptor,
            }),
        )?;
        let value = extend_load(builder, source, loaded.into(), transfer.signed)?;
        if let Some(target) = write_register(builder, source, transfer.rt, value, predicate)? {
            return Ok(LiftOutcome::Terminate(Terminator::Conditional {
                condition: predicate,
                taken: target,
                fallthrough,
            }));
        }
    } else {
        let value = read_register(builder, source, transfer.rt, false)?;
        let value = narrow(builder, source, value, descriptor.value_type())?;
        builder.emit(
            source,
            &[],
            OperationKind::Memory(MemoryOperation::GuardedStore {
                predicate,
                address,
                value,
                descriptor,
            }),
        )?;
    }
    if transfer.writeback {
        let _ = write_register(builder, source, transfer.rn, offset.into(), predicate)?;
    }
    Ok(LiftOutcome::Continue)
}

pub(super) fn lift_multiple_transfer(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    predicate: Operand,
    transfer: MultipleTransfer,
    fallthrough: ControlTarget,
) -> Result<LiftOutcome, BuildError> {
    let count = transfer.registers.count_ones();
    let base = read_register(builder, source, transfer.rn, false)?;
    let start_delta = if transfer.increment {
        if transfer.before { 4 } else { 0 }
    } else if transfer.before {
        -(i64::from(count) * 4)
    } else {
        -(i64::from(count.saturating_sub(1)) * 4)
    };
    let mut current = offset_i32(builder, source, base, start_delta)?;
    let descriptor = descriptor(
        MemorySize::Word,
        MemoryOrdering::Relaxed,
        MemoryAccessClass::Normal,
    );
    let mut pc_target = None;
    for register in 0_u8..16 {
        if transfer.registers & (1_u16 << register) == 0 {
            continue;
        }
        let memory_address = address(builder, source, current)?;
        if transfer.load {
            let old = read_register(builder, source, register, false)?;
            let loaded = emit_one(
                builder,
                source,
                IrType::I32,
                OperationKind::Memory(MemoryOperation::GuardedLoad {
                    predicate,
                    address: memory_address,
                    fallback: old,
                    descriptor,
                }),
            )?;
            pc_target =
                write_register(builder, source, register, loaded.into(), predicate)?.or(pc_target);
        } else {
            let value = read_register(builder, source, register, false)?;
            builder.emit(
                source,
                &[],
                OperationKind::Memory(MemoryOperation::GuardedStore {
                    predicate,
                    address: memory_address,
                    value,
                    descriptor,
                }),
            )?;
        }
        current = offset_i32(builder, source, current, 4)?;
    }
    if transfer.writeback {
        let delta = if transfer.increment {
            i64::from(count) * 4
        } else {
            -(i64::from(count) * 4)
        };
        let updated = offset_i32(builder, source, base, delta)?;
        let _ = write_register(builder, source, transfer.rn, updated, predicate)?;
    }
    if let Some(taken) = pc_target {
        return Ok(LiftOutcome::Terminate(Terminator::Conditional {
            condition: predicate,
            taken,
            fallthrough,
        }));
    }
    Ok(LiftOutcome::Continue)
}

pub(super) fn lift_exclusive_transfer(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    predicate: Operand,
    transfer: ExclusiveTransfer,
) -> Result<LiftOutcome, BuildError> {
    match transfer {
        ExclusiveTransfer::Load { size, rn, rt } => {
            let base = read_register(builder, source, rn, false)?;
            let address = address(builder, source, base)?;
            let descriptor =
                descriptor(size, MemoryOrdering::Relaxed, MemoryAccessClass::Exclusive);
            let old = read_register(builder, source, rt, false)?;
            let fallback = narrow(builder, source, old, descriptor.value_type())?;
            let loaded = emit_one(
                builder,
                source,
                descriptor.value_type(),
                OperationKind::Exclusive(ExclusiveOperation::GuardedLoad {
                    predicate,
                    address,
                    fallback,
                    descriptor,
                }),
            )?;
            let value = extend_load(builder, source, loaded.into(), false)?;
            let _ = write_register(builder, source, rt, value, predicate)?;
        }
        ExclusiveTransfer::Store {
            size,
            rn,
            rt,
            status,
        } => {
            let base = read_register(builder, source, rn, false)?;
            let address = address(builder, source, base)?;
            let descriptor =
                descriptor(size, MemoryOrdering::Relaxed, MemoryAccessClass::Exclusive);
            let value = read_register(builder, source, rt, false)?;
            let value = narrow(builder, source, value, descriptor.value_type())?;
            let old_status = read_register(builder, source, status, false)?;
            let fallback = emit_one(
                builder,
                source,
                IrType::I1,
                OperationKind::Scalar(ScalarOperation::Compare {
                    predicate: crate::ir::op::IntegerPredicate::Equal,
                    lhs: old_status,
                    rhs: Immediate::I32(0).into(),
                }),
            )?;
            let succeeded = emit_one(
                builder,
                source,
                IrType::I1,
                OperationKind::Exclusive(ExclusiveOperation::GuardedStore {
                    predicate,
                    address,
                    value,
                    fallback: fallback.into(),
                    descriptor,
                }),
            )?;
            let encoded = emit_one(
                builder,
                source,
                IrType::I32,
                OperationKind::Scalar(ScalarOperation::Select {
                    condition: succeeded.into(),
                    when_true: Immediate::I32(0).into(),
                    when_false: Immediate::I32(1).into(),
                }),
            )?;
            let _ = write_register(builder, source, status, encoded.into(), predicate)?;
        }
    }
    Ok(LiftOutcome::Continue)
}

pub(super) fn lift_acquire_release_transfer(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    predicate: Operand,
    transfer: AcquireReleaseTransfer,
) -> Result<LiftOutcome, BuildError> {
    match transfer {
        AcquireReleaseTransfer::Load { size, rn, rt } => {
            let base = read_register(builder, source, rn, false)?;
            let address = address(builder, source, base)?;
            let descriptor = descriptor(size, MemoryOrdering::Acquire, MemoryAccessClass::Normal);
            let old = read_register(builder, source, rt, false)?;
            let fallback = narrow(builder, source, old, descriptor.value_type())?;
            let loaded = emit_one(
                builder,
                source,
                descriptor.value_type(),
                OperationKind::Memory(MemoryOperation::GuardedLoad {
                    predicate,
                    address,
                    fallback,
                    descriptor,
                }),
            )?;
            let value = extend_load(builder, source, loaded.into(), false)?;
            let _ = write_register(builder, source, rt, value, predicate)?;
        }
        AcquireReleaseTransfer::Store { size, rn, rt } => {
            let base = read_register(builder, source, rn, false)?;
            let address = address(builder, source, base)?;
            let descriptor = descriptor(size, MemoryOrdering::Release, MemoryAccessClass::Normal);
            let value = read_register(builder, source, rt, false)?;
            let value = narrow(builder, source, value, descriptor.value_type())?;
            builder.emit(
                source,
                &[],
                OperationKind::Memory(MemoryOperation::GuardedStore {
                    predicate,
                    address,
                    value,
                    descriptor,
                }),
            )?;
        }
    }
    Ok(LiftOutcome::Continue)
}

pub(super) fn lift_vector_data(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    predicate: Operand,
    data: VectorDataProcessing,
) -> Result<LiftOutcome, BuildError> {
    let vector_type = if data.size == VectorSize::D {
        IrType::V64
    } else {
        IrType::V128
    };
    let lhs = read_vector(builder, source, data.size, data.vn)?;
    let rhs = read_vector(builder, source, data.size, data.vm)?;
    let old = read_vector(builder, source, data.size, data.vd)?;
    let result = match data.operation {
        DecodedVectorOperation::And
        | DecodedVectorOperation::BitClear
        | DecodedVectorOperation::Or
        | DecodedVectorOperation::ExclusiveOr
        | DecodedVectorOperation::Move => helper(
            builder,
            source,
            "aarch32.neon.bitwise",
            vec![
                lhs,
                rhs,
                Immediate::I32(vector_operation_code(data.operation)).into(),
            ],
            &[vector_type],
            OperationEffects::new(EffectSet::HELPER, false),
        )?[0],
        DecodedVectorOperation::AddInteger { lane_bits }
        | DecodedVectorOperation::SubtractInteger { lane_bits } => {
            let lane_type = match lane_bits {
                8 => LaneType::I8,
                16 => LaneType::I16,
                32 => LaneType::I32,
                64 => LaneType::I64,
                _ => unreachable!("normalized NEON lane width"),
            };
            let lanes = match data.size {
                VectorSize::D => 64 / lane_bits,
                VectorSize::Q => 128 / lane_bits,
            };
            emit_one(
                builder,
                source,
                vector_type,
                OperationKind::Vector(VectorOperation::Arithmetic {
                    kind: if matches!(data.operation, DecodedVectorOperation::AddInteger { .. }) {
                        IntegerBinaryKind::Add
                    } else {
                        IntegerBinaryKind::Subtract
                    },
                    arrangement: VectorArrangement {
                        lane_type,
                        lane_count: lanes,
                    },
                    lhs,
                    rhs,
                }),
            )?
        }
        DecodedVectorOperation::AddF32
        | DecodedVectorOperation::SubtractF32
        | DecodedVectorOperation::MultiplyF32 => {
            let fpscr = emit_one(
                builder,
                source,
                IrType::I32,
                OperationKind::ReadState(StateRegister::A32Fpscr),
            )?;
            let results = helper(
                builder,
                source,
                "aarch32.vfp.binary32-vector",
                vec![
                    predicate,
                    old,
                    lhs,
                    rhs,
                    fpscr.into(),
                    Immediate::I32(vector_operation_code(data.operation)).into(),
                    Immediate::I32(match data.size {
                        VectorSize::D => 2,
                        VectorSize::Q => 4,
                    })
                    .into(),
                ],
                &[vector_type, IrType::I32],
                OperationEffects::new(
                    EffectSet::HELPER
                        .union(EffectSet::READ_FPCR)
                        .union(EffectSet::WRITE_FPSR),
                    true,
                ),
            )?;
            let selected_fpscr =
                select(builder, source, predicate, results[1].into(), fpscr.into())?;
            write_state(builder, source, StateRegister::A32Fpscr, selected_fpscr)?;
            results[0]
        }
    };
    let selected = select(builder, source, predicate, result.into(), old)?;
    write_vector(builder, source, data.size, data.vd, selected)?;
    Ok(LiftOutcome::Continue)
}

pub(super) fn lift_vector_transfer(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    predicate: Operand,
    transfer: VectorTransfer,
) -> Result<LiftOutcome, BuildError> {
    let base = read_register(builder, source, transfer.rn, false)?;
    let address = address(builder, source, base)?;
    let descriptor = descriptor(
        MemorySize::Word,
        MemoryOrdering::Relaxed,
        MemoryAccessClass::Normal,
    );
    let descriptor = MemoryDescriptor {
        access: MemoryAccess::new(
            MemoryAccessSize::Doubleword,
            MemoryAlignment::Unaligned,
            MemoryOrdering::Relaxed,
            MemoryAccessClass::Normal,
        ),
        ..descriptor
    };
    let register =
        StateRegister::A32D(RegisterIndex::new(transfer.vd).expect("normalized D register"));
    let old = emit_one(
        builder,
        source,
        IrType::I64,
        OperationKind::ReadState(register),
    )?;
    if transfer.load {
        let loaded = emit_one(
            builder,
            source,
            IrType::I64,
            OperationKind::Memory(MemoryOperation::GuardedLoad {
                predicate,
                address,
                fallback: old.into(),
                descriptor,
            }),
        )?;
        write_state(builder, source, register, loaded.into())?;
    } else {
        builder.emit(
            source,
            &[],
            OperationKind::Memory(MemoryOperation::GuardedStore {
                predicate,
                address,
                value: old.into(),
                descriptor,
            }),
        )?;
    }
    if let Some(rm) = transfer.writeback_rm {
        let increment = if rm == 13 {
            Immediate::I32(8).into()
        } else {
            read_register(builder, source, rm, false)?
        };
        let updated = emit_one(
            builder,
            source,
            IrType::I32,
            OperationKind::Scalar(ScalarOperation::Binary {
                kind: IntegerBinaryKind::Add,
                lhs: base,
                rhs: increment,
            }),
        )?;
        let _ = write_register(builder, source, transfer.rn, updated.into(), predicate)?;
    }
    Ok(LiftOutcome::Continue)
}

fn memory_offset(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    offset: MemoryOffset,
) -> Result<Operand, BuildError> {
    match offset {
        MemoryOffset::Immediate(value) => Ok(Immediate::I32(value).into()),
        MemoryOffset::Register { rm, shift } => {
            let value = read_register(builder, source, rm, false)?;
            let amount = match shift.amount {
                ShiftAmount::Immediate(amount) => Immediate::I32(u32::from(amount)).into(),
                ShiftAmount::Register(rs) => read_register(builder, source, rs, false)?,
            };
            let cpsr = read_cpsr(builder, source)?;
            Ok(helper(
                builder,
                source,
                "aarch32.shift",
                vec![
                    value,
                    amount,
                    cpsr,
                    Immediate::I32(u32::from(shift_kind_code(shift))).into(),
                    Immediate::I32(u32::from(matches!(shift.amount, ShiftAmount::Register(_))))
                        .into(),
                ],
                &[IrType::I32],
                OperationEffects::new(EffectSet::HELPER, false),
            )?[0]
                .into())
        }
    }
}

fn address(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    value: Operand,
) -> Result<Operand, BuildError> {
    Ok(emit_one(
        builder,
        source,
        IrType::Address,
        OperationKind::Address(AddressOperation::FromInteger {
            value,
            width: GuestAddressWidth::Bits32,
        }),
    )?
    .into())
}

fn offset_i32(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    base: Operand,
    offset: i64,
) -> Result<Operand, BuildError> {
    Ok(emit_one(
        builder,
        source,
        IrType::I32,
        OperationKind::Scalar(ScalarOperation::Binary {
            kind: if offset < 0 {
                IntegerBinaryKind::Subtract
            } else {
                IntegerBinaryKind::Add
            },
            lhs: base,
            rhs: Immediate::I32(offset.unsigned_abs() as u32).into(),
        }),
    )?
    .into())
}

fn descriptor(
    size: MemorySize,
    ordering: MemoryOrdering,
    class: MemoryAccessClass,
) -> MemoryDescriptor {
    let size = match size {
        MemorySize::Byte => MemoryAccessSize::Byte,
        MemorySize::Halfword => MemoryAccessSize::Halfword,
        MemorySize::Word => MemoryAccessSize::Word,
    };
    MemoryDescriptor {
        access: MemoryAccess::new(size, MemoryAlignment::Unaligned, ordering, class),
        byte_order: ByteOrder::Little,
        volatility: Volatility::NonVolatile,
        privilege: MemoryPrivilege::Current,
    }
}

fn narrow(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    value: Operand,
    to: IrType,
) -> Result<Operand, BuildError> {
    if value.ty() == to {
        return Ok(value);
    }
    Ok(emit_one(
        builder,
        source,
        to,
        OperationKind::Scalar(ScalarOperation::Truncate { value, to }),
    )?
    .into())
}

fn extend_load(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    value: Operand,
    signed: bool,
) -> Result<Operand, BuildError> {
    if value.ty() == IrType::I32 {
        return Ok(value);
    }
    Ok(emit_one(
        builder,
        source,
        IrType::I32,
        OperationKind::Scalar(if signed {
            ScalarOperation::SignExtend {
                value,
                to: IrType::I32,
            }
        } else {
            ScalarOperation::ZeroExtend {
                value,
                to: IrType::I32,
            }
        }),
    )?
    .into())
}

fn read_vector(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    size: VectorSize,
    index: u8,
) -> Result<Operand, BuildError> {
    let first = emit_one(
        builder,
        source,
        IrType::I64,
        OperationKind::ReadState(StateRegister::A32D(
            RegisterIndex::new(if size == VectorSize::Q {
                index * 2
            } else {
                index
            })
            .unwrap(),
        )),
    )?;
    if size == VectorSize::D {
        return Ok(emit_one(
            builder,
            source,
            IrType::V64,
            OperationKind::Scalar(ScalarOperation::Bitcast {
                value: first.into(),
                to: IrType::V64,
            }),
        )?
        .into());
    }
    let second = if size == VectorSize::Q {
        emit_one(
            builder,
            source,
            IrType::I64,
            OperationKind::ReadState(StateRegister::A32D(
                RegisterIndex::new(index * 2 + 1).unwrap(),
            )),
        )?
        .into()
    } else {
        Immediate::I64(0).into()
    };
    Ok(helper(
        builder,
        source,
        "aarch32.vector.pack",
        vec![first.into(), second],
        &[IrType::V128],
        OperationEffects::new(EffectSet::HELPER, false),
    )?[0]
        .into())
}

fn write_vector(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    size: VectorSize,
    index: u8,
    value: Operand,
) -> Result<(), BuildError> {
    if size == VectorSize::D {
        let bits = emit_one(
            builder,
            source,
            IrType::I64,
            OperationKind::Scalar(ScalarOperation::Bitcast {
                value,
                to: IrType::I64,
            }),
        )?;
        return write_state(
            builder,
            source,
            StateRegister::A32D(RegisterIndex::new(index).unwrap()),
            bits.into(),
        );
    }
    let parts = helper(
        builder,
        source,
        "aarch32.vector.unpack",
        vec![value],
        &[IrType::I64, IrType::I64],
        OperationEffects::new(EffectSet::HELPER, false),
    )?;
    let first = if size == VectorSize::Q {
        index * 2
    } else {
        index
    };
    write_state(
        builder,
        source,
        StateRegister::A32D(RegisterIndex::new(first).unwrap()),
        parts[0].into(),
    )?;
    if size == VectorSize::Q {
        write_state(
            builder,
            source,
            StateRegister::A32D(RegisterIndex::new(first + 1).unwrap()),
            parts[1].into(),
        )?;
    }
    Ok(())
}

fn helper(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    name: &'static str,
    arguments: Vec<Operand>,
    results: &[IrType],
    effects: OperationEffects,
) -> Result<Box<[Value]>, BuildError> {
    Ok(builder
        .emit(
            source,
            results,
            OperationKind::Helper(HelperOperation {
                helper: name.into(),
                arguments: arguments.into_boxed_slice(),
                effects,
            }),
        )?
        .iter()
        .collect::<Vec<_>>()
        .into_boxed_slice())
}

fn emit_one(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    ty: IrType,
    kind: OperationKind,
) -> Result<Value, BuildError> {
    Ok(builder.emit(source, &[ty], kind)?.iter().next().unwrap())
}

fn data_operation_code(operation: DataOperation) -> u32 {
    match operation {
        DataOperation::And => 0,
        DataOperation::ExclusiveOr => 1,
        DataOperation::Subtract => 2,
        DataOperation::ReverseSubtract => 3,
        DataOperation::Add => 4,
        DataOperation::AddCarry => 5,
        DataOperation::SubtractCarry => 6,
        DataOperation::ReverseSubtractCarry => 7,
        DataOperation::Test => 8,
        DataOperation::TestExclusiveOr => 9,
        DataOperation::Compare => 10,
        DataOperation::CompareNegative => 11,
        DataOperation::Or => 12,
        DataOperation::Move => 13,
        DataOperation::BitClear => 14,
        DataOperation::MoveNot => 15,
    }
}

fn shift_kind_code(shift: Shift) -> u8 {
    use crate::semantics::shifts::A32ShiftKind;
    match shift.kind {
        A32ShiftKind::LogicalLeft => 0,
        A32ShiftKind::LogicalRight => 1,
        A32ShiftKind::ArithmeticRight => 2,
        A32ShiftKind::RotateRight => 3,
        A32ShiftKind::RotateRightExtended => 4,
    }
}

fn vector_operation_code(operation: DecodedVectorOperation) -> u32 {
    match operation {
        DecodedVectorOperation::Move => 0,
        DecodedVectorOperation::And => 1,
        DecodedVectorOperation::BitClear => 2,
        DecodedVectorOperation::Or => 3,
        DecodedVectorOperation::ExclusiveOr => 4,
        DecodedVectorOperation::AddInteger { .. } => 5,
        DecodedVectorOperation::SubtractInteger { .. } => 6,
        DecodedVectorOperation::AddF32 => 7,
        DecodedVectorOperation::SubtractF32 => 8,
        DecodedVectorOperation::MultiplyF32 => 9,
    }
}
