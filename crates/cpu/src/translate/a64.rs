//! A64-to-IR translation for the minimum viable instruction subset.

use crate::{
    address::GuestVirtualAddress,
    decode::DecodedOpcode,
    ir::{
        builder::{BuildError, IrBuilder},
        op::{
            BarrierAccess, BarrierDomain, BarrierOperation, ByteOrder, CacheMaintenanceKind,
            CacheMaintenanceOperation, Condition, EffectSet, ExclusiveOperation, FlagOperation,
            HelperOperation, IntegerBinaryKind, IntegerPredicate, MemoryDescriptor,
            MemoryOperation, OperationEffects, OperationKind, ScalarOperation, ShiftKind,
            StateRegister, Volatility,
        },
        terminator::{ControlTarget, ExceptionKind, Terminator},
        types::IrType,
        value::{Immediate, Operand, Value},
    },
    location::{DecodedInstruction, ExecutionState, LocationDescriptor},
    memory::{MemoryAccess, MemoryAccessClass, MemoryAccessSize, MemoryAlignment, MemoryOrdering},
    semantics::immediate::decode_a64_logical_immediate,
    state::a64::A64GeneralRegister,
};

use super::block::{
    LiftOutcome, conditional_terminator, direct_branch_target, emit_call, indirect_target,
};

#[derive(Clone, Copy)]
enum Register31 {
    Zero,
    StackPointer,
}

pub(crate) fn lift(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
) -> LiftOutcome {
    lift_inner(builder, decoded).expect("A64 lifter only emits verifier-compatible IR")
}

fn lift_inner(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
) -> Result<LiftOutcome, BuildError> {
    let bits = decoded.encoding.bits();
    let source = decoded.location;
    let outcome = match decoded.instruction.pattern().name {
        "nop" => LiftOutcome::Continue,
        "b" => LiftOutcome::Terminate(super::block::direct_branch(
            direct_branch_target(decoded).expect("decoded B has an aligned immediate target"),
        )),
        "bl" => {
            let target = direct_target(source, sign_extend(u64::from(bits & 0x03ff_ffff), 26) << 2);
            let return_address = next_pc(source);
            LiftOutcome::Terminate(emit_call(builder, source, target, return_address)?)
        }
        "branch-register" => lift_branch_register(builder, source, bits)?,
        "b.cond" => lift_conditional_branch(builder, source, bits)?,
        "compare-branch" => lift_compare_branch(builder, source, bits)?,
        "test-branch" => lift_test_branch(builder, source, bits)?,
        "svc" => LiftOutcome::Terminate(exception(
            source,
            ExceptionKind::SupervisorCall,
            Some(u64::from((bits >> 5) & 0xffff)),
        )),
        "brk" => LiftOutcome::Terminate(exception(
            source,
            ExceptionKind::Breakpoint,
            Some(u64::from((bits >> 5) & 0xffff)),
        )),
        "hint" => lift_hint(builder, decoded, bits)?,
        "mrs" => lift_mrs(builder, decoded, bits)?,
        "msr-register" => lift_msr(builder, decoded, bits)?,
        "barrier" => lift_barrier(builder, decoded, bits)?,
        "system" => lift_system(builder, decoded, bits)?,
        "move-wide" => lift_move_wide(builder, decoded, bits)?,
        "add-sub-immediate" => lift_add_sub_immediate(builder, decoded, bits)?,
        "add-sub-shifted" => lift_add_sub_shifted(builder, decoded, bits)?,
        "add-sub-extended" => lift_add_sub_extended(builder, decoded, bits)?,
        "add-sub-carry" => lift_add_sub_carry(builder, decoded, bits)?,
        "logical-immediate" => lift_logical_immediate(builder, decoded, bits)?,
        "logical-shifted" => lift_logical_shifted(builder, decoded, bits)?,
        "bitfield" => lift_bitfield(builder, decoded, bits)?,
        "extract" => lift_extract(builder, decoded, bits)?,
        "data-processing-two-source" => lift_two_source(builder, decoded, bits)?,
        "conditional-compare-register" | "conditional-compare-immediate" => {
            lift_conditional_compare(builder, decoded, bits)?
        }
        "conditional-select" => lift_conditional_select(builder, decoded, bits)?,
        "data-processing-three-source" => lift_three_source(builder, decoded, bits)?,
        "data-processing-one-source" => lift_one_source(builder, decoded, bits)?,
        "adr" | "adrp" => lift_adr(builder, decoded, bits)?,
        "load-literal" => lift_literal_load(builder, decoded, bits)?,
        "load-store-unsigned" => lift_load_store_unsigned(builder, decoded, bits)?,
        "load-store-unscaled" | "load-store-post-index" | "load-store-pre-index" => {
            lift_load_store_indexed(builder, decoded, bits)?
        }
        "load-store-register" => lift_load_store_register(builder, decoded, bits)?,
        "load-store-pair" => lift_load_store_pair(builder, decoded, bits)?,
        "load-acquire" | "store-release" => lift_acquire_release(builder, decoded, bits)?,
        "load-exclusive" | "store-exclusive" => lift_exclusive(builder, decoded, bits)?,
        "simd-bitwise"
        | "simd-integer"
        | "fp-scalar-two-source"
        | "fp-scalar-move"
        | "fp-compare-register"
        | "fp-compare-zero" => lift_fp_simd_compute(builder, decoded, bits)?,
        "fp-signed-int-to-float"
        | "fp-unsigned-int-to-float"
        | "fp-float-to-signed-int"
        | "fp-float-to-unsigned-int"
        | "fp-move-to-general"
        | "fp-move-from-general" => lift_fp_conversion(builder, decoded, bits)?,
        "fp-simd-load-store-unsigned"
        | "fp-simd-load-store-unscaled"
        | "fp-simd-load-store-post-index"
        | "fp-simd-load-store-pre-index"
        | "fp-simd-load-store-register"
        | "fp-simd-load-literal" => lift_fp_simd_memory(builder, decoded, bits)?,
        _ => interpret(decoded),
    };
    Ok(outcome)
}

fn interpret(decoded: &DecodedInstruction<DecodedOpcode>) -> LiftOutcome {
    LiftOutcome::Interpret(decoded.instruction.coverage_id())
}

fn next_pc(source: LocationDescriptor) -> GuestVirtualAddress {
    source
        .pc
        .checked_add(4)
        .expect("a fetched A64 instruction has a representable fallthrough")
}

fn direct_target(source: LocationDescriptor, displacement: i64) -> ControlTarget {
    ControlTarget::Direct {
        pc: source.pc.wrapping_offset(displacement),
        execution_state: ExecutionState::A64,
    }
}

fn sign_extend(value: u64, bits: u8) -> i64 {
    let shift = 64 - bits;
    ((value << shift) as i64) >> shift
}

fn condition(bits: u32) -> Condition {
    match bits & 0xf {
        0 => Condition::Eq,
        1 => Condition::Ne,
        2 => Condition::Cs,
        3 => Condition::Cc,
        4 => Condition::Mi,
        5 => Condition::Pl,
        6 => Condition::Vs,
        7 => Condition::Vc,
        8 => Condition::Hi,
        9 => Condition::Ls,
        10 => Condition::Ge,
        11 => Condition::Lt,
        12 => Condition::Gt,
        13 => Condition::Le,
        14 => Condition::Al,
        15 => Condition::Nv,
        _ => unreachable!(),
    }
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

fn read_gpr(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    index: u8,
    width: IrType,
    register31: Register31,
) -> Result<Operand, BuildError> {
    let value = match (index, register31) {
        (31, Register31::Zero) => {
            return Ok(match width {
                IrType::I32 => Immediate::I32(0).into(),
                IrType::I64 => Immediate::I64(0).into(),
                _ => unreachable!("A64 GPR width is 32 or 64 bits"),
            });
        }
        (31, Register31::StackPointer) => emit_one(
            builder,
            source,
            IrType::I64,
            OperationKind::ReadState(StateRegister::A64Sp),
        )?,
        (index, _) => emit_one(
            builder,
            source,
            IrType::I64,
            OperationKind::ReadState(StateRegister::A64X(
                A64GeneralRegister::new(index).expect("GPR field is five bits"),
            )),
        )?,
    };
    if width != IrType::I64 {
        Ok(emit_one(
            builder,
            source,
            width,
            OperationKind::Scalar(ScalarOperation::Truncate {
                value: value.into(),
                to: width,
            }),
        )?
        .into())
    } else {
        Ok(value.into())
    }
}

fn write_gpr(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    index: u8,
    value: Operand,
    register31: Register31,
) -> Result<(), BuildError> {
    if index == 31 && matches!(register31, Register31::Zero) {
        return Ok(());
    }
    let value = if value.ty() != IrType::I64 {
        emit_one(
            builder,
            source,
            IrType::I64,
            OperationKind::Scalar(ScalarOperation::ZeroExtend {
                value,
                to: IrType::I64,
            }),
        )?
        .into()
    } else {
        value
    };
    let register = if index == 31 {
        StateRegister::A64Sp
    } else {
        StateRegister::A64X(A64GeneralRegister::new(index).expect("GPR field is five bits"))
    };
    builder.emit(source, &[], OperationKind::WriteState { register, value })?;
    Ok(())
}

fn bitcast(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    value: Operand,
    to: IrType,
) -> Result<Operand, BuildError> {
    Ok(emit_one(
        builder,
        source,
        to,
        OperationKind::Scalar(ScalarOperation::Bitcast { value, to }),
    )?
    .into())
}

fn scalar(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    ty: IrType,
    operation: ScalarOperation,
) -> Result<Operand, BuildError> {
    Ok(emit_one(builder, source, ty, OperationKind::Scalar(operation))?.into())
}

fn binary(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    kind: IntegerBinaryKind,
    lhs: Operand,
    rhs: Operand,
) -> Result<Operand, BuildError> {
    scalar(
        builder,
        source,
        lhs.ty(),
        ScalarOperation::Binary { kind, lhs, rhs },
    )
}

fn read_flags(builder: &mut IrBuilder, source: LocationDescriptor) -> Result<Operand, BuildError> {
    let packed = emit_one(
        builder,
        source,
        IrType::I32,
        OperationKind::ReadState(StateRegister::A64Nzcv),
    )?;
    Ok(emit_one(
        builder,
        source,
        IrType::Flags,
        OperationKind::Flags(FlagOperation::FromPacked {
            value: packed.into(),
        }),
    )?
    .into())
}

fn evaluate_condition(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    condition: Condition,
) -> Result<Operand, BuildError> {
    let flags = read_flags(builder, source)?;
    Ok(emit_one(
        builder,
        source,
        IrType::I1,
        OperationKind::Flags(FlagOperation::Evaluate { flags, condition }),
    )?
    .into())
}

fn write_flags(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    flags: Operand,
) -> Result<(), BuildError> {
    let packed = emit_one(
        builder,
        source,
        IrType::I32,
        OperationKind::Flags(FlagOperation::Materialize { flags }),
    )?;
    builder.emit(
        source,
        &[],
        OperationKind::WriteState {
            register: StateRegister::A64Nzcv,
            value: packed.into(),
        },
    )?;
    Ok(())
}

fn arithmetic_flags(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    result: Operand,
    carry: Operand,
    overflow: Operand,
) -> Result<(), BuildError> {
    let flags = emit_one(
        builder,
        source,
        IrType::Flags,
        OperationKind::Flags(FlagOperation::FromArithmetic {
            result,
            carry,
            overflow,
        }),
    )?;
    write_flags(builder, source, flags.into())
}

fn logical_flags(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    result: Operand,
) -> Result<(), BuildError> {
    let flags = emit_one(
        builder,
        source,
        IrType::Flags,
        OperationKind::Flags(FlagOperation::FromLogical {
            result,
            carry: Immediate::I1(false).into(),
        }),
    )?;
    write_flags(builder, source, flags.into())
}

fn helper(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    name: &'static str,
    arguments: impl Into<Box<[Operand]>>,
    result_types: &[IrType],
    effects: OperationEffects,
) -> Result<Vec<Value>, BuildError> {
    Ok(builder
        .emit(
            source,
            result_types,
            OperationKind::Helper(HelperOperation {
                helper: name.into(),
                arguments: arguments.into(),
                effects,
            }),
        )?
        .iter()
        .collect())
}

fn lift_branch_register(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    bits: u32,
) -> Result<LiftOutcome, BuildError> {
    let masked = bits & 0xffff_fc1f;
    let rn = ((bits >> 5) & 0x1f) as u8;
    if !matches!(masked, 0xd61f_0000 | 0xd63f_0000 | 0xd65f_0000) {
        return Ok(LiftOutcome::Interpret(crate::coverage::CoverageId::new(5)));
    }
    let address_bits = read_gpr(builder, source, rn, IrType::I64, Register31::Zero)?;
    let address = bitcast(builder, source, address_bits, IrType::Address)?;
    let target = indirect_target(address, ExecutionState::A64);
    Ok(match masked {
        0xd61f_0000 => LiftOutcome::Terminate(Terminator::Indirect { target }),
        0xd63f_0000 => LiftOutcome::Terminate(emit_call(builder, source, target, next_pc(source))?),
        0xd65f_0000 => LiftOutcome::Terminate(Terminator::Return { target }),
        _ => unreachable!(),
    })
}

fn lift_conditional_branch(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    bits: u32,
) -> Result<LiftOutcome, BuildError> {
    let cond = evaluate_condition(builder, source, condition(bits))?;
    let displacement = sign_extend(u64::from((bits >> 5) & 0x7ffff), 19) << 2;
    Ok(LiftOutcome::Terminate(conditional_terminator(
        cond,
        direct_target(source, displacement),
        direct_target(source, 4),
    )))
}

fn lift_compare_branch(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    bits: u32,
) -> Result<LiftOutcome, BuildError> {
    let width = if bits >> 31 != 0 {
        IrType::I64
    } else {
        IrType::I32
    };
    let value = read_gpr(
        builder,
        source,
        (bits & 0x1f) as u8,
        width,
        Register31::Zero,
    )?;
    let zero = if width == IrType::I64 {
        Immediate::I64(0)
    } else {
        Immediate::I32(0)
    };
    let predicate = if bits & (1 << 24) == 0 {
        IntegerPredicate::Equal
    } else {
        IntegerPredicate::NotEqual
    };
    let condition = scalar(
        builder,
        source,
        IrType::I1,
        ScalarOperation::Compare {
            predicate,
            lhs: value,
            rhs: zero.into(),
        },
    )?;
    let displacement = sign_extend(u64::from((bits >> 5) & 0x7ffff), 19) << 2;
    Ok(LiftOutcome::Terminate(conditional_terminator(
        condition,
        direct_target(source, displacement),
        direct_target(source, 4),
    )))
}

fn lift_test_branch(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    bits: u32,
) -> Result<LiftOutcome, BuildError> {
    let bit_index = (((bits >> 31) & 1) << 5) | ((bits >> 19) & 0x1f);
    let value = read_gpr(
        builder,
        source,
        (bits & 0x1f) as u8,
        IrType::I64,
        Register31::Zero,
    )?;
    let tested = binary(
        builder,
        source,
        IntegerBinaryKind::And,
        value,
        Immediate::I64(1_u64 << bit_index).into(),
    )?;
    let predicate = if bits & (1 << 24) == 0 {
        IntegerPredicate::Equal
    } else {
        IntegerPredicate::NotEqual
    };
    let condition = scalar(
        builder,
        source,
        IrType::I1,
        ScalarOperation::Compare {
            predicate,
            lhs: tested,
            rhs: Immediate::I64(0).into(),
        },
    )?;
    let displacement = sign_extend(u64::from((bits >> 5) & 0x3fff), 14) << 2;
    Ok(LiftOutcome::Terminate(conditional_terminator(
        condition,
        direct_target(source, displacement),
        direct_target(source, 4),
    )))
}

fn exception(source: LocationDescriptor, kind: ExceptionKind, syndrome: Option<u64>) -> Terminator {
    Terminator::Exception {
        source,
        kind,
        syndrome,
    }
}

fn lift_hint(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    bits: u32,
) -> Result<LiftOutcome, BuildError> {
    let immediate = (bits >> 5) & 0x7f;
    if immediate == 0 {
        return Ok(LiftOutcome::Continue);
    }
    let name = match immediate {
        1 => "a64.hint.yield",
        2 => "a64.hint.wfe",
        3 => "a64.hint.wfi",
        4 => "a64.hint.sev",
        5 => "a64.hint.sevl",
        _ => return Ok(interpret(decoded)),
    };
    helper(
        builder,
        decoded.location,
        name,
        Box::<[Operand]>::default(),
        &[],
        OperationEffects::new(EffectSet::HELPER, false),
    )?;
    Ok(LiftOutcome::Continue)
}

fn system_register(bits: u32) -> Option<StateRegister> {
    match bits & 0xffff_ffe0 {
        0xd53b_4200 | 0xd51b_4200 => Some(StateRegister::A64Nzcv),
        0xd53b_4400 | 0xd51b_4400 => Some(StateRegister::A64Fpcr),
        0xd53b_4420 | 0xd51b_4420 => Some(StateRegister::A64Fpsr),
        0xd53b_d040 | 0xd51b_d040 => Some(StateRegister::A64TpidrEl0),
        0xd53b_d060 => Some(StateRegister::A64TpidrroEl0),
        _ => None,
    }
}

fn lift_mrs(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    bits: u32,
) -> Result<LiftOutcome, BuildError> {
    let Some(register) = system_register(bits) else {
        return Ok(interpret(decoded));
    };
    let value = emit_one(
        builder,
        decoded.location,
        register.ty(),
        OperationKind::ReadState(register),
    )?;
    let value = if register.ty() == IrType::I32 {
        scalar(
            builder,
            decoded.location,
            IrType::I64,
            ScalarOperation::ZeroExtend {
                value: value.into(),
                to: IrType::I64,
            },
        )?
    } else {
        value.into()
    };
    write_gpr(
        builder,
        decoded.location,
        (bits & 0x1f) as u8,
        value,
        Register31::Zero,
    )?;
    Ok(LiftOutcome::Continue)
}

fn lift_msr(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    bits: u32,
) -> Result<LiftOutcome, BuildError> {
    let Some(register) = system_register(bits) else {
        return Ok(interpret(decoded));
    };
    if register == StateRegister::A64TpidrroEl0 {
        return Ok(interpret(decoded));
    }
    let mut value = read_gpr(
        builder,
        decoded.location,
        (bits & 0x1f) as u8,
        IrType::I64,
        Register31::Zero,
    )?;
    if register.ty() == IrType::I32 {
        value = scalar(
            builder,
            decoded.location,
            IrType::I32,
            ScalarOperation::Truncate {
                value,
                to: IrType::I32,
            },
        )?;
    }
    if register == StateRegister::A64Nzcv {
        value = binary(
            builder,
            decoded.location,
            IntegerBinaryKind::And,
            value,
            Immediate::I32(0xf000_0000).into(),
        )?;
    }
    builder.emit(
        decoded.location,
        &[],
        OperationKind::WriteState { register, value },
    )?;
    Ok(LiftOutcome::Continue)
}

fn barrier_scope(option: u8) -> Option<(BarrierDomain, BarrierAccess)> {
    let access = match option & 3 {
        1 => BarrierAccess::Reads,
        2 => BarrierAccess::Writes,
        3 => BarrierAccess::ReadsAndWrites,
        _ => return None,
    };
    let domain = match option >> 2 {
        0 => BarrierDomain::OuterShareable,
        1 => BarrierDomain::NonShareable,
        2 => BarrierDomain::InnerShareable,
        3 => BarrierDomain::FullSystem,
        _ => return None,
    };
    Some((domain, access))
}

fn lift_barrier(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    bits: u32,
) -> Result<LiftOutcome, BuildError> {
    let opcode = (bits >> 5) & 7;
    let option = ((bits >> 8) & 0xf) as u8;
    let operation = match opcode {
        4 | 5 => {
            let Some((domain, access)) = barrier_scope(option) else {
                return Ok(interpret(decoded));
            };
            if opcode == 4 {
                BarrierOperation::DataSynchronization { domain, access }
            } else {
                BarrierOperation::DataMemory { domain, access }
            }
        }
        6 if option == 15 => BarrierOperation::InstructionSynchronization,
        _ => return Ok(interpret(decoded)),
    };
    builder.emit(decoded.location, &[], OperationKind::Barrier(operation))?;
    Ok(LiftOutcome::Continue)
}

fn lift_system(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    bits: u32,
) -> Result<LiftOutcome, BuildError> {
    let (kind, uses_address) = match bits & 0xffff_ffe0 {
        0xd508_7500 => (CacheMaintenanceKind::InstructionInvalidate, false), // IC IALLU
        0xd50b_7520 => (CacheMaintenanceKind::InstructionInvalidate, true),  // IC IVAU
        0xd508_7620 => (CacheMaintenanceKind::DataInvalidate, true),         // DC IVAC
        0xd50b_7b20 => (CacheMaintenanceKind::DataClean, true),              // DC CVAU
        0xd50b_7e20 => (CacheMaintenanceKind::DataCleanAndInvalidate, true), // DC CIVAC
        _ => return Ok(interpret(decoded)),
    };
    let address = if uses_address {
        let raw = read_gpr(
            builder,
            decoded.location,
            (bits & 0x1f) as u8,
            IrType::I64,
            Register31::Zero,
        )?;
        Some(bitcast(builder, decoded.location, raw, IrType::Address)?)
    } else {
        None
    };
    builder.emit(
        decoded.location,
        &[],
        OperationKind::CacheMaintenance(CacheMaintenanceOperation { kind, address }),
    )?;
    Ok(LiftOutcome::Continue)
}

fn integer_width(bits: u32) -> IrType {
    if bits >> 31 != 0 {
        IrType::I64
    } else {
        IrType::I32
    }
}

fn immediate_for(width: IrType, value: u64) -> Immediate {
    if width == IrType::I64 {
        Immediate::I64(value)
    } else {
        Immediate::I32(value as u32)
    }
}

fn lift_move_wide(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    bits: u32,
) -> Result<LiftOutcome, BuildError> {
    let width = integer_width(bits);
    let hw = (bits >> 21) & 3;
    if width == IrType::I32 && hw >= 2 {
        return Ok(interpret(decoded));
    }
    let shift = hw * 16;
    let imm = u64::from((bits >> 5) & 0xffff) << shift;
    let opc = (bits >> 29) & 3;
    let value: Operand = match opc {
        0 => immediate_for(width, !imm).into(), // MOVN, truncated by the immediate type
        2 => immediate_for(width, imm).into(),  // MOVZ
        3 => {
            let old = read_gpr(
                builder,
                decoded.location,
                (bits & 0x1f) as u8,
                width,
                Register31::Zero,
            )?;
            let mask = !(0xffff_u64 << shift);
            let retained = binary(
                builder,
                decoded.location,
                IntegerBinaryKind::And,
                old,
                immediate_for(width, mask).into(),
            )?;
            binary(
                builder,
                decoded.location,
                IntegerBinaryKind::Or,
                retained,
                immediate_for(width, imm).into(),
            )?
        }
        _ => return Ok(interpret(decoded)),
    };
    write_gpr(
        builder,
        decoded.location,
        (bits & 0x1f) as u8,
        value,
        Register31::Zero,
    )?;
    Ok(LiftOutcome::Continue)
}

struct AddSubSpec {
    width: IrType,
    subtract: bool,
    set_flags: bool,
    destination: u8,
    destination_register31: Register31,
}

fn emit_add_sub(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    lhs: Operand,
    rhs: Operand,
    spec: AddSubSpec,
) -> Result<(), BuildError> {
    let rhs = if spec.subtract {
        binary(
            builder,
            source,
            IntegerBinaryKind::Xor,
            rhs,
            immediate_for(spec.width, u64::MAX).into(),
        )?
    } else {
        rhs
    };
    let results: Vec<_> = builder
        .emit(
            source,
            &[spec.width, IrType::I1, IrType::I1],
            OperationKind::Scalar(ScalarOperation::AddWithCarry {
                lhs,
                rhs,
                carry_in: Immediate::I1(spec.subtract).into(),
            }),
        )?
        .iter()
        .collect();
    write_gpr(
        builder,
        source,
        spec.destination,
        results[0].into(),
        spec.destination_register31,
    )?;
    if spec.set_flags {
        arithmetic_flags(
            builder,
            source,
            results[0].into(),
            results[1].into(),
            results[2].into(),
        )?;
    }
    Ok(())
}

fn lift_add_sub_immediate(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    bits: u32,
) -> Result<LiftOutcome, BuildError> {
    let width = integer_width(bits);
    let lhs = read_gpr(
        builder,
        decoded.location,
        ((bits >> 5) & 0x1f) as u8,
        width,
        Register31::StackPointer,
    )?;
    let shift = if bits & (1 << 22) != 0 { 12 } else { 0 };
    let rhs = immediate_for(width, u64::from((bits >> 10) & 0xfff) << shift).into();
    let set_flags = bits & (1 << 29) != 0;
    emit_add_sub(
        builder,
        decoded.location,
        lhs,
        rhs,
        AddSubSpec {
            width,
            subtract: bits & (1 << 30) != 0,
            set_flags,
            destination: (bits & 0x1f) as u8,
            destination_register31: if set_flags {
                Register31::Zero
            } else {
                Register31::StackPointer
            },
        },
    )?;
    Ok(LiftOutcome::Continue)
}

fn shifted_register(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    bits: u32,
    width: IrType,
    index: u8,
) -> Result<Option<Operand>, BuildError> {
    let amount = (bits >> 10) & 0x3f;
    if width == IrType::I32 && amount >= 32 {
        return Ok(None);
    }
    let kind = match (bits >> 22) & 3 {
        0 => ShiftKind::LogicalLeft,
        1 => ShiftKind::LogicalRight,
        2 => ShiftKind::ArithmeticRight,
        3 => ShiftKind::RotateRight,
        _ => unreachable!(),
    };
    let value = read_gpr(builder, source, index, width, Register31::Zero)?;
    if amount == 0 {
        Ok(Some(value))
    } else {
        Ok(Some(scalar(
            builder,
            source,
            width,
            ScalarOperation::Shift {
                kind,
                value,
                amount: immediate_for(width, u64::from(amount)).into(),
            },
        )?))
    }
}

fn lift_add_sub_shifted(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    bits: u32,
) -> Result<LiftOutcome, BuildError> {
    if (bits >> 22) & 3 == 3 {
        return Ok(interpret(decoded));
    }
    let width = integer_width(bits);
    let Some(rhs) = shifted_register(
        builder,
        decoded.location,
        bits,
        width,
        ((bits >> 16) & 0x1f) as u8,
    )?
    else {
        return Ok(interpret(decoded));
    };
    let lhs = read_gpr(
        builder,
        decoded.location,
        ((bits >> 5) & 0x1f) as u8,
        width,
        Register31::Zero,
    )?;
    emit_add_sub(
        builder,
        decoded.location,
        lhs,
        rhs,
        AddSubSpec {
            width,
            subtract: bits & (1 << 30) != 0,
            set_flags: bits & (1 << 29) != 0,
            destination: (bits & 0x1f) as u8,
            destination_register31: Register31::Zero,
        },
    )?;
    Ok(LiftOutcome::Continue)
}

fn lift_add_sub_extended(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    bits: u32,
) -> Result<LiftOutcome, BuildError> {
    let width = integer_width(bits);
    let shift = (bits >> 10) & 7;
    if shift > 4 {
        return Ok(interpret(decoded));
    }
    let rm = read_gpr(
        builder,
        decoded.location,
        ((bits >> 16) & 0x1f) as u8,
        width,
        Register31::Zero,
    )?;
    let extension = ((bits >> 13) & 7) as u64;
    let result = helper(
        builder,
        decoded.location,
        "a64.extend-register",
        vec![
            rm,
            Immediate::I8(extension as u8).into(),
            Immediate::I8(shift as u8).into(),
        ],
        &[width],
        OperationEffects::default(),
    )?[0];
    let lhs = read_gpr(
        builder,
        decoded.location,
        ((bits >> 5) & 0x1f) as u8,
        width,
        Register31::StackPointer,
    )?;
    let set_flags = bits & (1 << 29) != 0;
    emit_add_sub(
        builder,
        decoded.location,
        lhs,
        result.into(),
        AddSubSpec {
            width,
            subtract: bits & (1 << 30) != 0,
            set_flags,
            destination: (bits & 0x1f) as u8,
            destination_register31: if set_flags {
                Register31::Zero
            } else {
                Register31::StackPointer
            },
        },
    )?;
    Ok(LiftOutcome::Continue)
}

fn carry_in(builder: &mut IrBuilder, source: LocationDescriptor) -> Result<Operand, BuildError> {
    evaluate_condition(builder, source, Condition::Cs)
}

fn lift_add_sub_carry(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    bits: u32,
) -> Result<LiftOutcome, BuildError> {
    let width = integer_width(bits);
    let lhs = read_gpr(
        builder,
        decoded.location,
        ((bits >> 5) & 31) as u8,
        width,
        Register31::Zero,
    )?;
    let mut rhs = read_gpr(
        builder,
        decoded.location,
        ((bits >> 16) & 31) as u8,
        width,
        Register31::Zero,
    )?;
    let subtract = bits & (1 << 30) != 0;
    if subtract {
        rhs = binary(
            builder,
            decoded.location,
            IntegerBinaryKind::Xor,
            rhs,
            immediate_for(width, u64::MAX).into(),
        )?;
    }
    let carry = carry_in(builder, decoded.location)?;
    let values: Vec<_> = builder
        .emit(
            decoded.location,
            &[width, IrType::I1, IrType::I1],
            OperationKind::Scalar(ScalarOperation::AddWithCarry {
                lhs,
                rhs,
                carry_in: carry,
            }),
        )?
        .iter()
        .collect();
    write_gpr(
        builder,
        decoded.location,
        (bits & 31) as u8,
        values[0].into(),
        Register31::Zero,
    )?;
    if bits & (1 << 29) != 0 {
        arithmetic_flags(
            builder,
            decoded.location,
            values[0].into(),
            values[1].into(),
            values[2].into(),
        )?;
    }
    Ok(LiftOutcome::Continue)
}

fn logical_result(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    opc: u32,
    lhs: Operand,
    rhs: Operand,
) -> Result<Operand, BuildError> {
    binary(
        builder,
        source,
        match opc {
            0 | 3 => IntegerBinaryKind::And,
            1 => IntegerBinaryKind::Or,
            2 => IntegerBinaryKind::Xor,
            _ => unreachable!(),
        },
        lhs,
        rhs,
    )
}

fn lift_logical_immediate(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    bits: u32,
) -> Result<LiftOutcome, BuildError> {
    let width = integer_width(bits);
    let size = if width == IrType::I64 { 64 } else { 32 };
    let Ok(immediate) = decode_a64_logical_immediate(
        bits & (1 << 22) != 0,
        ((bits >> 16) & 0x3f) as u8,
        ((bits >> 10) & 0x3f) as u8,
        size,
    ) else {
        return Ok(interpret(decoded));
    };
    let lhs = read_gpr(
        builder,
        decoded.location,
        ((bits >> 5) & 31) as u8,
        width,
        Register31::Zero,
    )?;
    let opc = (bits >> 29) & 3;
    let result = logical_result(
        builder,
        decoded.location,
        opc,
        lhs,
        immediate_for(width, immediate).into(),
    )?;
    write_gpr(
        builder,
        decoded.location,
        (bits & 31) as u8,
        result,
        Register31::Zero,
    )?;
    if opc == 3 {
        logical_flags(builder, decoded.location, result)?;
    }
    Ok(LiftOutcome::Continue)
}

fn lift_logical_shifted(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    bits: u32,
) -> Result<LiftOutcome, BuildError> {
    let width = integer_width(bits);
    let Some(mut rhs) = shifted_register(
        builder,
        decoded.location,
        bits,
        width,
        ((bits >> 16) & 31) as u8,
    )?
    else {
        return Ok(interpret(decoded));
    };
    if bits & (1 << 21) != 0 {
        rhs = binary(
            builder,
            decoded.location,
            IntegerBinaryKind::Xor,
            rhs,
            immediate_for(width, u64::MAX).into(),
        )?;
    }
    let lhs = read_gpr(
        builder,
        decoded.location,
        ((bits >> 5) & 31) as u8,
        width,
        Register31::Zero,
    )?;
    let opc = (bits >> 29) & 3;
    let result = logical_result(builder, decoded.location, opc, lhs, rhs)?;
    write_gpr(
        builder,
        decoded.location,
        (bits & 31) as u8,
        result,
        Register31::Zero,
    )?;
    if opc == 3 {
        logical_flags(builder, decoded.location, result)?;
    }
    Ok(LiftOutcome::Continue)
}

fn lift_bitfield(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    bits: u32,
) -> Result<LiftOutcome, BuildError> {
    let width = integer_width(bits);
    let n = bits & (1 << 22) != 0;
    if n != (width == IrType::I64) || (width == IrType::I32 && bits & 0x0000_8000 != 0) {
        return Ok(interpret(decoded));
    }
    let opc = (bits >> 29) & 3;
    if opc == 3 {
        return Ok(interpret(decoded));
    }
    let source_value = read_gpr(
        builder,
        decoded.location,
        ((bits >> 5) & 31) as u8,
        width,
        Register31::Zero,
    )?;
    let destination = read_gpr(
        builder,
        decoded.location,
        (bits & 31) as u8,
        width,
        Register31::Zero,
    )?;
    let name = match opc {
        0 => "a64.sbfm",
        1 => "a64.bfm",
        2 => "a64.ubfm",
        _ => unreachable!(),
    };
    let value = helper(
        builder,
        decoded.location,
        name,
        vec![
            destination,
            source_value,
            Immediate::I8(((bits >> 16) & 63) as u8).into(),
            Immediate::I8(((bits >> 10) & 63) as u8).into(),
        ],
        &[width],
        OperationEffects::default(),
    )?[0];
    write_gpr(
        builder,
        decoded.location,
        (bits & 31) as u8,
        value.into(),
        Register31::Zero,
    )?;
    Ok(LiftOutcome::Continue)
}

fn lift_extract(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    bits: u32,
) -> Result<LiftOutcome, BuildError> {
    let width = integer_width(bits);
    let lsb = (bits >> 10) & 63;
    if (bits & (1 << 22) != 0) != (width == IrType::I64) || (width == IrType::I32 && lsb >= 32) {
        return Ok(interpret(decoded));
    }
    let first = read_gpr(
        builder,
        decoded.location,
        ((bits >> 5) & 31) as u8,
        width,
        Register31::Zero,
    )?;
    let second = read_gpr(
        builder,
        decoded.location,
        ((bits >> 16) & 31) as u8,
        width,
        Register31::Zero,
    )?;
    let value = helper(
        builder,
        decoded.location,
        "a64.extr",
        vec![first, second, Immediate::I8(lsb as u8).into()],
        &[width],
        OperationEffects::default(),
    )?[0];
    write_gpr(
        builder,
        decoded.location,
        (bits & 31) as u8,
        value.into(),
        Register31::Zero,
    )?;
    Ok(LiftOutcome::Continue)
}

fn lift_two_source(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    bits: u32,
) -> Result<LiftOutcome, BuildError> {
    let width = integer_width(bits);
    let opcode = (bits >> 10) & 0x3f;
    let lhs = read_gpr(
        builder,
        decoded.location,
        ((bits >> 5) & 31) as u8,
        width,
        Register31::Zero,
    )?;
    let mut rhs = read_gpr(
        builder,
        decoded.location,
        ((bits >> 16) & 31) as u8,
        width,
        Register31::Zero,
    )?;
    if matches!(opcode, 8..=11) {
        rhs = binary(
            builder,
            decoded.location,
            IntegerBinaryKind::And,
            rhs,
            immediate_for(width, if width == IrType::I64 { 63 } else { 31 }).into(),
        )?;
    }
    let operation = match opcode {
        2 => ScalarOperation::Binary {
            kind: IntegerBinaryKind::UnsignedDivide,
            lhs,
            rhs,
        },
        3 => ScalarOperation::Binary {
            kind: IntegerBinaryKind::SignedDivide,
            lhs,
            rhs,
        },
        8 => ScalarOperation::Shift {
            kind: ShiftKind::LogicalLeft,
            value: lhs,
            amount: rhs,
        },
        9 => ScalarOperation::Shift {
            kind: ShiftKind::LogicalRight,
            value: lhs,
            amount: rhs,
        },
        10 => ScalarOperation::Shift {
            kind: ShiftKind::ArithmeticRight,
            value: lhs,
            amount: rhs,
        },
        11 => ScalarOperation::Shift {
            kind: ShiftKind::RotateRight,
            value: lhs,
            amount: rhs,
        },
        _ => return Ok(interpret(decoded)),
    };
    let value = scalar(builder, decoded.location, width, operation)?;
    write_gpr(
        builder,
        decoded.location,
        (bits & 31) as u8,
        value,
        Register31::Zero,
    )?;
    Ok(LiftOutcome::Continue)
}

fn proposed_compare_flags(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    width: IrType,
    lhs: Operand,
    mut rhs: Operand,
    subtract: bool,
) -> Result<Operand, BuildError> {
    if subtract {
        rhs = binary(
            builder,
            source,
            IntegerBinaryKind::Xor,
            rhs,
            immediate_for(width, u64::MAX).into(),
        )?;
    }
    let values: Vec<_> = builder
        .emit(
            source,
            &[width, IrType::I1, IrType::I1],
            OperationKind::Scalar(ScalarOperation::AddWithCarry {
                lhs,
                rhs,
                carry_in: Immediate::I1(subtract).into(),
            }),
        )?
        .iter()
        .collect();
    Ok(emit_one(
        builder,
        source,
        IrType::Flags,
        OperationKind::Flags(FlagOperation::FromArithmetic {
            result: values[0].into(),
            carry: values[1].into(),
            overflow: values[2].into(),
        }),
    )?
    .into())
}

fn lift_conditional_compare(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    bits: u32,
) -> Result<LiftOutcome, BuildError> {
    let width = integer_width(bits);
    let lhs = read_gpr(
        builder,
        decoded.location,
        ((bits >> 5) & 31) as u8,
        width,
        Register31::Zero,
    )?;
    let rhs = if bits & (1 << 11) != 0 {
        immediate_for(width, u64::from((bits >> 16) & 31)).into()
    } else {
        read_gpr(
            builder,
            decoded.location,
            ((bits >> 16) & 31) as u8,
            width,
            Register31::Zero,
        )?
    };
    let proposed = proposed_compare_flags(
        builder,
        decoded.location,
        width,
        lhs,
        rhs,
        bits & (1 << 30) != 0,
    )?;
    let fallback = emit_one(
        builder,
        decoded.location,
        IrType::Flags,
        OperationKind::Flags(FlagOperation::FromPacked {
            value: Immediate::I32((bits & 15) << 28).into(),
        }),
    )?;
    let cond = evaluate_condition(builder, decoded.location, condition(bits >> 12))?;
    let selected = scalar(
        builder,
        decoded.location,
        IrType::Flags,
        ScalarOperation::Select {
            condition: cond,
            when_true: proposed,
            when_false: fallback.into(),
        },
    )?;
    write_flags(builder, decoded.location, selected)?;
    Ok(LiftOutcome::Continue)
}

fn lift_conditional_select(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    bits: u32,
) -> Result<LiftOutcome, BuildError> {
    let width = integer_width(bits);
    let true_value = read_gpr(
        builder,
        decoded.location,
        ((bits >> 5) & 31) as u8,
        width,
        Register31::Zero,
    )?;
    let mut false_value = read_gpr(
        builder,
        decoded.location,
        ((bits >> 16) & 31) as u8,
        width,
        Register31::Zero,
    )?;
    let op = bits & (1 << 30) != 0;
    let op2 = bits & (1 << 10) != 0;
    if op {
        false_value = binary(
            builder,
            decoded.location,
            IntegerBinaryKind::Xor,
            false_value,
            immediate_for(width, u64::MAX).into(),
        )?;
    }
    if op2 {
        false_value = binary(
            builder,
            decoded.location,
            IntegerBinaryKind::Add,
            false_value,
            immediate_for(width, 1).into(),
        )?;
    }
    let cond = evaluate_condition(builder, decoded.location, condition(bits >> 12))?;
    let result = scalar(
        builder,
        decoded.location,
        width,
        ScalarOperation::Select {
            condition: cond,
            when_true: true_value,
            when_false: false_value,
        },
    )?;
    write_gpr(
        builder,
        decoded.location,
        (bits & 31) as u8,
        result,
        Register31::Zero,
    )?;
    Ok(LiftOutcome::Continue)
}

fn lift_three_source(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    bits: u32,
) -> Result<LiftOutcome, BuildError> {
    let width = integer_width(bits);
    let opcode = (bits >> 21) & 7;
    if opcode != 0 {
        if matches!(opcode, 2 | 6) && ((bits >> 10) & 31) != 31 {
            return Ok(interpret(decoded));
        }
        let name = match (opcode, bits & (1 << 15) != 0) {
            (1, false) => "a64.smaddl",
            (1, true) => "a64.smsubl",
            (2, false) => "a64.smulh",
            (5, false) => "a64.umaddl",
            (5, true) => "a64.umsubl",
            (6, false) => "a64.umulh",
            _ => return Ok(interpret(decoded)),
        };
        let operand_width = if matches!(opcode, 1 | 5) {
            IrType::I32
        } else {
            IrType::I64
        };
        let mut values = vec![
            read_gpr(
                builder,
                decoded.location,
                ((bits >> 5) & 31) as u8,
                operand_width,
                Register31::Zero,
            )?,
            read_gpr(
                builder,
                decoded.location,
                ((bits >> 16) & 31) as u8,
                operand_width,
                Register31::Zero,
            )?,
        ];
        if matches!(opcode, 1 | 5) {
            values.push(read_gpr(
                builder,
                decoded.location,
                ((bits >> 10) & 31) as u8,
                IrType::I64,
                Register31::Zero,
            )?);
        }
        let result = helper(
            builder,
            decoded.location,
            name,
            values,
            &[IrType::I64],
            OperationEffects::default(),
        )?[0];
        write_gpr(
            builder,
            decoded.location,
            (bits & 31) as u8,
            result.into(),
            Register31::Zero,
        )?;
        return Ok(LiftOutcome::Continue);
    }
    let lhs = read_gpr(
        builder,
        decoded.location,
        ((bits >> 5) & 31) as u8,
        width,
        Register31::Zero,
    )?;
    let rhs = read_gpr(
        builder,
        decoded.location,
        ((bits >> 16) & 31) as u8,
        width,
        Register31::Zero,
    )?;
    let addend = read_gpr(
        builder,
        decoded.location,
        ((bits >> 10) & 31) as u8,
        width,
        Register31::Zero,
    )?;
    let product = binary(
        builder,
        decoded.location,
        IntegerBinaryKind::Multiply,
        lhs,
        rhs,
    )?;
    let result = binary(
        builder,
        decoded.location,
        if bits & (1 << 15) != 0 {
            IntegerBinaryKind::Subtract
        } else {
            IntegerBinaryKind::Add
        },
        addend,
        product,
    )?;
    write_gpr(
        builder,
        decoded.location,
        (bits & 31) as u8,
        result,
        Register31::Zero,
    )?;
    Ok(LiftOutcome::Continue)
}

fn lift_one_source(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    bits: u32,
) -> Result<LiftOutcome, BuildError> {
    let width = integer_width(bits);
    let opcode = (bits >> 10) & 0x3f;
    let input = read_gpr(
        builder,
        decoded.location,
        ((bits >> 5) & 31) as u8,
        width,
        Register31::Zero,
    )?;
    let value = match opcode {
        0 => scalar(
            builder,
            decoded.location,
            width,
            ScalarOperation::ReverseBits { value: input },
        )?,
        4 => scalar(
            builder,
            decoded.location,
            width,
            ScalarOperation::CountLeadingZeros { value: input },
        )?,
        1 | 2 | 3 | 5 => {
            let name = match opcode {
                1 => "a64.rev16",
                2 => "a64.rev32",
                3 => "a64.rev",
                5 => "a64.cls",
                _ => unreachable!(),
            };
            helper(
                builder,
                decoded.location,
                name,
                vec![input],
                &[width],
                OperationEffects::default(),
            )?[0]
                .into()
        }
        _ => return Ok(interpret(decoded)),
    };
    write_gpr(
        builder,
        decoded.location,
        (bits & 31) as u8,
        value,
        Register31::Zero,
    )?;
    Ok(LiftOutcome::Continue)
}

fn lift_adr(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    bits: u32,
) -> Result<LiftOutcome, BuildError> {
    let immediate = sign_extend(u64::from(((bits >> 3) & 0x1ffffc) | ((bits >> 29) & 3)), 21);
    let address = if decoded.instruction.pattern().name == "adrp" {
        GuestVirtualAddress::new(decoded.location.pc.get() & !0xfff)
            .wrapping_offset(immediate << 12)
    } else {
        decoded.location.pc.wrapping_offset(immediate)
    };
    write_gpr(
        builder,
        decoded.location,
        (bits & 31) as u8,
        Immediate::I64(address.get()).into(),
        Register31::Zero,
    )?;
    Ok(LiftOutcome::Continue)
}

fn descriptor(
    size: MemoryAccessSize,
    ordering: MemoryOrdering,
    class: MemoryAccessClass,
) -> MemoryDescriptor {
    MemoryDescriptor {
        access: MemoryAccess::new(size, MemoryAlignment::Unaligned, ordering, class),
        byte_order: ByteOrder::Little,
        volatility: Volatility::NonVolatile,
    }
}

fn aligned_descriptor(
    size: MemoryAccessSize,
    ordering: MemoryOrdering,
    class: MemoryAccessClass,
) -> MemoryDescriptor {
    let mut descriptor = descriptor(size, ordering, class);
    descriptor.access.alignment = MemoryAlignment::Natural;
    descriptor
}

fn size_from_bits(size: u32) -> MemoryAccessSize {
    match size {
        0 => MemoryAccessSize::Byte,
        1 => MemoryAccessSize::Halfword,
        2 => MemoryAccessSize::Word,
        3 => MemoryAccessSize::Doubleword,
        _ => unreachable!(),
    }
}

fn address_add(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    base: Operand,
    offset: i64,
) -> Result<Operand, BuildError> {
    let raw = binary(
        builder,
        source,
        IntegerBinaryKind::Add,
        base,
        Immediate::I64(offset as u64).into(),
    )?;
    bitcast(builder, source, raw, IrType::Address)
}

fn base_address(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    rn: u8,
) -> Result<Operand, BuildError> {
    read_gpr(builder, source, rn, IrType::I64, Register31::StackPointer)
}

fn memory_transfer(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    bits: u32,
    address: Operand,
    descriptor: MemoryDescriptor,
) -> Result<bool, BuildError> {
    let opc = (bits >> 22) & 3;
    let rt = (bits & 31) as u8;
    if opc == 0 {
        let value = read_gpr(
            builder,
            source,
            rt,
            descriptor.value_type(),
            Register31::Zero,
        )?;
        builder.emit(
            source,
            &[],
            OperationKind::Memory(MemoryOperation::Store {
                address,
                value,
                descriptor,
            }),
        )?;
        return Ok(true);
    }
    if (opc >= 2 && descriptor.access.size == MemoryAccessSize::Doubleword)
        || (opc == 3 && descriptor.access.size == MemoryAccessSize::Word)
    {
        return Ok(false);
    }
    let loaded = emit_one(
        builder,
        source,
        descriptor.value_type(),
        OperationKind::Memory(MemoryOperation::Load {
            address,
            descriptor,
        }),
    )?;
    let destination_width = if opc == 2 || descriptor.access.size == MemoryAccessSize::Doubleword {
        IrType::I64
    } else {
        IrType::I32
    };
    let mut value: Operand = loaded.into();
    if descriptor.value_type() != destination_width {
        value = scalar(
            builder,
            source,
            destination_width,
            if matches!(opc, 2 | 3) {
                ScalarOperation::SignExtend {
                    value,
                    to: destination_width,
                }
            } else {
                ScalarOperation::ZeroExtend {
                    value,
                    to: destination_width,
                }
            },
        )?;
    }
    write_gpr(builder, source, rt, value, Register31::Zero)?;
    Ok(true)
}

fn lift_literal_load(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    bits: u32,
) -> Result<LiftOutcome, BuildError> {
    let opc = bits >> 30;
    if opc == 3 {
        return Ok(interpret(decoded)); // PRFM literal
    }
    let size = match opc {
        0 | 2 => MemoryAccessSize::Word,
        1 => MemoryAccessSize::Doubleword,
        3 => return Ok(interpret(decoded)), // PRFM literal
        _ => unreachable!(),
    };
    let address = bitcast(
        builder,
        decoded.location,
        Immediate::I64(
            decoded
                .location
                .pc
                .wrapping_offset(sign_extend(u64::from((bits >> 5) & 0x7ffff), 19) << 2)
                .get(),
        )
        .into(),
        IrType::Address,
    )?;
    let descriptor = descriptor(size, MemoryOrdering::Relaxed, MemoryAccessClass::Normal);
    let loaded = emit_one(
        builder,
        decoded.location,
        descriptor.value_type(),
        OperationKind::Memory(MemoryOperation::Load {
            address,
            descriptor,
        }),
    )?;
    let value = if opc == 2 {
        scalar(
            builder,
            decoded.location,
            IrType::I64,
            ScalarOperation::SignExtend {
                value: loaded.into(),
                to: IrType::I64,
            },
        )?
    } else {
        loaded.into()
    };
    write_gpr(
        builder,
        decoded.location,
        (bits & 31) as u8,
        value,
        Register31::Zero,
    )?;
    Ok(LiftOutcome::Continue)
}

fn lift_load_store_unsigned(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    bits: u32,
) -> Result<LiftOutcome, BuildError> {
    let size = size_from_bits(bits >> 30);
    let base = base_address(builder, decoded.location, ((bits >> 5) & 31) as u8)?;
    let address = address_add(
        builder,
        decoded.location,
        base,
        i64::from((bits >> 10) & 0xfff) * size.bytes() as i64,
    )?;
    if !memory_transfer(
        builder,
        decoded.location,
        bits,
        address,
        descriptor(size, MemoryOrdering::Relaxed, MemoryAccessClass::Normal),
    )? {
        return Ok(interpret(decoded));
    }
    Ok(LiftOutcome::Continue)
}

fn lift_load_store_indexed(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    bits: u32,
) -> Result<LiftOutcome, BuildError> {
    let size = size_from_bits(bits >> 30);
    let rn = ((bits >> 5) & 31) as u8;
    let rt = (bits & 31) as u8;
    let mode = decoded.instruction.pattern().name;
    if mode != "load-store-unscaled" && rn != 31 && rn == rt {
        return Ok(interpret(decoded));
    }
    let base = base_address(builder, decoded.location, rn)?;
    let offset = sign_extend(u64::from((bits >> 12) & 0x1ff), 9);
    let address = if mode == "load-store-pre-index" {
        address_add(builder, decoded.location, base, offset)?
    } else {
        bitcast(builder, decoded.location, base, IrType::Address)?
    };
    if !memory_transfer(
        builder,
        decoded.location,
        bits,
        address,
        descriptor(size, MemoryOrdering::Relaxed, MemoryAccessClass::Normal),
    )? {
        return Ok(interpret(decoded));
    }
    if mode != "load-store-unscaled" {
        let updated = binary(
            builder,
            decoded.location,
            IntegerBinaryKind::Add,
            base,
            Immediate::I64(offset as u64).into(),
        )?;
        write_gpr(
            builder,
            decoded.location,
            rn,
            updated,
            Register31::StackPointer,
        )?;
    }
    Ok(LiftOutcome::Continue)
}

fn lift_load_store_register(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    bits: u32,
) -> Result<LiftOutcome, BuildError> {
    let size = size_from_bits(bits >> 30);
    let base = base_address(builder, decoded.location, ((bits >> 5) & 31) as u8)?;
    let offset = read_gpr(
        builder,
        decoded.location,
        ((bits >> 16) & 31) as u8,
        IrType::I64,
        Register31::Zero,
    )?;
    let option = (bits >> 13) & 7;
    if option & 2 == 0 {
        return Ok(interpret(decoded));
    }
    let shift = if bits & (1 << 12) != 0 {
        size.bytes().trailing_zeros() as u8
    } else {
        0
    };
    let offset = helper(
        builder,
        decoded.location,
        "a64.load-store-register-offset",
        vec![
            offset,
            Immediate::I8(option as u8).into(),
            Immediate::I8(shift).into(),
        ],
        &[IrType::I64],
        OperationEffects::default(),
    )?[0];
    let raw = binary(
        builder,
        decoded.location,
        IntegerBinaryKind::Add,
        base,
        offset.into(),
    )?;
    let address = bitcast(builder, decoded.location, raw, IrType::Address)?;
    if !memory_transfer(
        builder,
        decoded.location,
        bits,
        address,
        descriptor(size, MemoryOrdering::Relaxed, MemoryAccessClass::Normal),
    )? {
        return Ok(interpret(decoded));
    }
    Ok(LiftOutcome::Continue)
}

fn lift_load_store_pair(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    bits: u32,
) -> Result<LiftOutcome, BuildError> {
    let opc = bits >> 30;
    if opc == 3 {
        return Ok(interpret(decoded));
    }
    let size = if matches!(opc, 0 | 1) {
        MemoryAccessSize::Word
    } else {
        MemoryAccessSize::Doubleword
    };
    let width = if size == MemoryAccessSize::Word {
        IrType::I32
    } else {
        IrType::I64
    };
    let rn = ((bits >> 5) & 31) as u8;
    let rt = (bits & 31) as u8;
    let rt2 = ((bits >> 10) & 31) as u8;
    let mode = (bits >> 23) & 3;
    let load = bits & (1 << 22) != 0;
    if (load && rt == rt2) || (matches!(mode, 1 | 3) && rn != 31 && (rn == rt || rn == rt2)) {
        return Ok(interpret(decoded));
    }
    let base = base_address(builder, decoded.location, rn)?;
    let offset = sign_extend(u64::from((bits >> 15) & 0x7f), 7) * size.bytes() as i64;
    let transfer_base = if mode == 3 {
        binary(
            builder,
            decoded.location,
            IntegerBinaryKind::Add,
            base,
            Immediate::I64(offset as u64).into(),
        )?
    } else {
        base
    };
    let first_address = bitcast(builder, decoded.location, transfer_base, IrType::Address)?;
    let second_address = address_add(
        builder,
        decoded.location,
        transfer_base,
        size.bytes() as i64,
    )?;
    if opc == 1 && !load {
        return Ok(interpret(decoded));
    }
    let descriptor = descriptor(size, MemoryOrdering::Relaxed, MemoryAccessClass::Normal);
    for (rt, address) in [
        ((bits & 31) as u8, first_address),
        (((bits >> 10) & 31) as u8, second_address),
    ] {
        if load {
            let mut value: Operand = emit_one(
                builder,
                decoded.location,
                width,
                OperationKind::Memory(MemoryOperation::Load {
                    address,
                    descriptor,
                }),
            )?
            .into();
            if opc == 1 {
                value = scalar(
                    builder,
                    decoded.location,
                    IrType::I64,
                    ScalarOperation::SignExtend {
                        value,
                        to: IrType::I64,
                    },
                )?;
            }
            write_gpr(builder, decoded.location, rt, value, Register31::Zero)?;
        } else {
            let value = read_gpr(builder, decoded.location, rt, width, Register31::Zero)?;
            builder.emit(
                decoded.location,
                &[],
                OperationKind::Memory(MemoryOperation::Store {
                    address,
                    value,
                    descriptor,
                }),
            )?;
        }
    }
    if matches!(mode, 1 | 3) {
        let updated = binary(
            builder,
            decoded.location,
            IntegerBinaryKind::Add,
            base,
            Immediate::I64(offset as u64).into(),
        )?;
        write_gpr(
            builder,
            decoded.location,
            rn,
            updated,
            Register31::StackPointer,
        )?;
    }
    Ok(LiftOutcome::Continue)
}

fn lift_acquire_release(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    bits: u32,
) -> Result<LiftOutcome, BuildError> {
    let size = size_from_bits(bits >> 30);
    let base = base_address(builder, decoded.location, ((bits >> 5) & 31) as u8)?;
    let address = bitcast(builder, decoded.location, base, IrType::Address)?;
    let ordering = if decoded.instruction.pattern().name == "load-acquire" {
        MemoryOrdering::Acquire
    } else {
        MemoryOrdering::Release
    };
    let descriptor = aligned_descriptor(size, ordering, MemoryAccessClass::Normal);
    let rt = (bits & 31) as u8;
    if decoded.instruction.pattern().name == "load-acquire" {
        let value = emit_one(
            builder,
            decoded.location,
            descriptor.value_type(),
            OperationKind::Memory(MemoryOperation::Load {
                address,
                descriptor,
            }),
        )?;
        write_gpr(
            builder,
            decoded.location,
            rt,
            value.into(),
            Register31::Zero,
        )?;
    } else {
        let value = read_gpr(
            builder,
            decoded.location,
            rt,
            descriptor.value_type(),
            Register31::Zero,
        )?;
        builder.emit(
            decoded.location,
            &[],
            OperationKind::Memory(MemoryOperation::Store {
                address,
                value,
                descriptor,
            }),
        )?;
    }
    Ok(LiftOutcome::Continue)
}

fn lift_exclusive(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    bits: u32,
) -> Result<LiftOutcome, BuildError> {
    let size = size_from_bits(bits >> 30);
    let base = base_address(builder, decoded.location, ((bits >> 5) & 31) as u8)?;
    let address = bitcast(builder, decoded.location, base, IrType::Address)?;
    let ordered = bits & (1 << 15) != 0;
    let ordering = match (decoded.instruction.pattern().name, ordered) {
        ("load-exclusive", true) => MemoryOrdering::Acquire,
        ("store-exclusive", true) => MemoryOrdering::Release,
        (_, false) => MemoryOrdering::Relaxed,
        _ => unreachable!(),
    };
    let descriptor = aligned_descriptor(size, ordering, MemoryAccessClass::Exclusive);
    if decoded.instruction.pattern().name == "load-exclusive" {
        let value = emit_one(
            builder,
            decoded.location,
            descriptor.value_type(),
            OperationKind::Exclusive(ExclusiveOperation::Load {
                address,
                descriptor,
            }),
        )?;
        write_gpr(
            builder,
            decoded.location,
            (bits & 31) as u8,
            value.into(),
            Register31::Zero,
        )?;
    } else {
        let value = read_gpr(
            builder,
            decoded.location,
            (bits & 31) as u8,
            descriptor.value_type(),
            Register31::Zero,
        )?;
        let succeeded = emit_one(
            builder,
            decoded.location,
            IrType::I1,
            OperationKind::Exclusive(ExclusiveOperation::Store {
                address,
                value,
                descriptor,
            }),
        )?;
        let status = scalar(
            builder,
            decoded.location,
            IrType::I32,
            ScalarOperation::Select {
                condition: succeeded.into(),
                when_true: Immediate::I32(0).into(),
                when_false: Immediate::I32(1).into(),
            },
        )?;
        write_gpr(
            builder,
            decoded.location,
            ((bits >> 16) & 31) as u8,
            status,
            Register31::Zero,
        )?;
    }
    Ok(LiftOutcome::Continue)
}

fn vector_read(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    index: u8,
) -> Result<Operand, BuildError> {
    Ok(emit_one(
        builder,
        source,
        IrType::V128,
        OperationKind::ReadState(StateRegister::A64V(
            crate::ir::op::RegisterIndex::new(index).unwrap(),
        )),
    )?
    .into())
}

fn vector_write(
    builder: &mut IrBuilder,
    source: LocationDescriptor,
    index: u8,
    value: Operand,
) -> Result<(), BuildError> {
    builder.emit(
        source,
        &[],
        OperationKind::WriteState {
            register: StateRegister::A64V(crate::ir::op::RegisterIndex::new(index).unwrap()),
            value,
        },
    )?;
    Ok(())
}

fn lift_fp_simd_compute(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    bits: u32,
) -> Result<LiftOutcome, BuildError> {
    let family = decoded.instruction.pattern().name;
    let first = vector_read(builder, decoded.location, ((bits >> 5) & 31) as u8)?;
    if matches!(family, "simd-bitwise" | "simd-integer" | "fp-scalar-move") {
        let mut arguments = vec![first];
        if family != "fp-scalar-move" {
            arguments.push(vector_read(
                builder,
                decoded.location,
                ((bits >> 16) & 31) as u8,
            )?);
            arguments.push(vector_read(builder, decoded.location, (bits & 31) as u8)?);
        }
        arguments.push(Immediate::I32(bits).into());
        let name = match family {
            "simd-bitwise" => "a64.simd.bitwise",
            "simd-integer" => "a64.simd.integer-arithmetic-compare",
            "fp-scalar-move" => "a64.fp.scalar-move",
            _ => unreachable!(),
        };
        let result = helper(
            builder,
            decoded.location,
            name,
            arguments,
            &[IrType::V128],
            OperationEffects::new(EffectSet::HELPER, false),
        )?[0];
        vector_write(builder, decoded.location, (bits & 31) as u8, result.into())?;
        return Ok(LiftOutcome::Continue);
    }

    let second = if family == "fp-compare-zero" {
        Immediate::V128(0).into()
    } else {
        vector_read(builder, decoded.location, ((bits >> 16) & 31) as u8)?
    };
    let fpcr = emit_one(
        builder,
        decoded.location,
        IrType::I32,
        OperationKind::ReadState(StateRegister::A64Fpcr),
    )?;
    let fpsr = emit_one(
        builder,
        decoded.location,
        IrType::I32,
        OperationKind::ReadState(StateRegister::A64Fpsr),
    )?;
    let result_types: &[IrType] = if family.starts_with("fp-compare") {
        &[IrType::Flags, IrType::I32]
    } else {
        &[IrType::V128, IrType::I32]
    };
    let results = helper(
        builder,
        decoded.location,
        if family.starts_with("fp-compare") {
            "a64.fp.scalar-compare"
        } else {
            "a64.fp.scalar-arithmetic"
        },
        vec![
            first,
            second,
            fpcr.into(),
            fpsr.into(),
            Immediate::I32(bits).into(),
        ],
        result_types,
        OperationEffects::new(
            EffectSet::READ_FPCR
                .union(EffectSet::WRITE_FPSR)
                .union(EffectSet::HELPER),
            false,
        ),
    )?;
    if family.starts_with("fp-compare") {
        write_flags(builder, decoded.location, results[0].into())?;
    } else {
        vector_write(
            builder,
            decoded.location,
            (bits & 31) as u8,
            results[0].into(),
        )?;
    }
    builder.emit(
        decoded.location,
        &[],
        OperationKind::WriteState {
            register: StateRegister::A64Fpsr,
            value: results[1].into(),
        },
    )?;
    Ok(LiftOutcome::Continue)
}

fn lift_fp_conversion(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    bits: u32,
) -> Result<LiftOutcome, BuildError> {
    let family = decoded.instruction.pattern().name;
    if (bits >> 22) & 3 > 1 {
        return Ok(interpret(decoded));
    }
    let width = integer_width(bits);
    let rn = ((bits >> 5) & 31) as u8;
    let rd = (bits & 31) as u8;

    if family == "fp-move-to-general" {
        let vector = vector_read(builder, decoded.location, rn)?;
        let result = helper(
            builder,
            decoded.location,
            "a64.fp.move-to-general",
            vec![vector, Immediate::I32(bits).into()],
            &[width],
            OperationEffects::new(EffectSet::HELPER, false),
        )?[0];
        write_gpr(
            builder,
            decoded.location,
            rd,
            result.into(),
            Register31::Zero,
        )?;
        return Ok(LiftOutcome::Continue);
    }
    if family == "fp-move-from-general" {
        let integer = read_gpr(builder, decoded.location, rn, width, Register31::Zero)?;
        let result = helper(
            builder,
            decoded.location,
            "a64.fp.move-from-general",
            vec![integer, Immediate::I32(bits).into()],
            &[IrType::V128],
            OperationEffects::new(EffectSet::HELPER, false),
        )?[0];
        vector_write(builder, decoded.location, rd, result.into())?;
        return Ok(LiftOutcome::Continue);
    }

    let fpcr = emit_one(
        builder,
        decoded.location,
        IrType::I32,
        OperationKind::ReadState(StateRegister::A64Fpcr),
    )?;
    let fpsr = emit_one(
        builder,
        decoded.location,
        IrType::I32,
        OperationKind::ReadState(StateRegister::A64Fpsr),
    )?;
    let effects = OperationEffects::new(
        EffectSet::READ_FPCR
            .union(EffectSet::WRITE_FPSR)
            .union(EffectSet::HELPER),
        false,
    );
    let results = if family.ends_with("int-to-float") {
        let integer = read_gpr(builder, decoded.location, rn, width, Register31::Zero)?;
        helper(
            builder,
            decoded.location,
            if family.starts_with("fp-signed") {
                "a64.fp.signed-int-to-float"
            } else {
                "a64.fp.unsigned-int-to-float"
            },
            vec![
                integer,
                fpcr.into(),
                fpsr.into(),
                Immediate::I32(bits).into(),
            ],
            &[IrType::V128, IrType::I32],
            effects,
        )?
    } else {
        let vector = vector_read(builder, decoded.location, rn)?;
        helper(
            builder,
            decoded.location,
            if family.contains("signed") {
                "a64.fp.float-to-signed-int"
            } else {
                "a64.fp.float-to-unsigned-int"
            },
            vec![
                vector,
                fpcr.into(),
                fpsr.into(),
                Immediate::I32(bits).into(),
            ],
            &[width, IrType::I32],
            effects,
        )?
    };
    if family.ends_with("int-to-float") {
        vector_write(builder, decoded.location, rd, results[0].into())?;
    } else {
        write_gpr(
            builder,
            decoded.location,
            rd,
            results[0].into(),
            Register31::Zero,
        )?;
    }
    builder.emit(
        decoded.location,
        &[],
        OperationKind::WriteState {
            register: StateRegister::A64Fpsr,
            value: results[1].into(),
        },
    )?;
    Ok(LiftOutcome::Continue)
}

fn lift_fp_simd_memory(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    bits: u32,
) -> Result<LiftOutcome, BuildError> {
    let family = decoded.instruction.pattern().name;
    let literal = family == "fp-simd-load-literal";
    let size = if literal {
        match bits >> 30 {
            0 => MemoryAccessSize::Word,
            1 => MemoryAccessSize::Doubleword,
            2 => MemoryAccessSize::Quadword,
            _ => return Ok(interpret(decoded)),
        }
    } else if bits & (1 << 23) != 0 {
        MemoryAccessSize::Quadword
    } else {
        size_from_bits(bits >> 30)
    };
    let rn = ((bits >> 5) & 31) as u8;
    let mut writeback = None;
    let address = if literal {
        let target = decoded
            .location
            .pc
            .wrapping_offset(sign_extend(u64::from((bits >> 5) & 0x7ffff), 19) << 2);
        bitcast(
            builder,
            decoded.location,
            Immediate::I64(target.get()).into(),
            IrType::Address,
        )?
    } else {
        let base = base_address(builder, decoded.location, rn)?;
        if family == "fp-simd-load-store-register" {
            let option = (bits >> 13) & 7;
            if option & 2 == 0 {
                return Ok(interpret(decoded));
            }
            let raw_offset = read_gpr(
                builder,
                decoded.location,
                ((bits >> 16) & 31) as u8,
                IrType::I64,
                Register31::Zero,
            )?;
            let shift = if bits & (1 << 12) != 0 {
                size.bytes().trailing_zeros() as u8
            } else {
                0
            };
            let offset = helper(
                builder,
                decoded.location,
                "a64.load-store-register-offset",
                vec![
                    raw_offset,
                    Immediate::I8(option as u8).into(),
                    Immediate::I8(shift).into(),
                ],
                &[IrType::I64],
                OperationEffects::default(),
            )?[0];
            let raw = binary(
                builder,
                decoded.location,
                IntegerBinaryKind::Add,
                base,
                offset.into(),
            )?;
            bitcast(builder, decoded.location, raw, IrType::Address)?
        } else {
            let offset = if family.ends_with("unsigned") {
                i64::from((bits >> 10) & 0xfff) * size.bytes() as i64
            } else {
                sign_extend(u64::from((bits >> 12) & 0x1ff), 9)
            };
            let transfer_base = if family.ends_with("post-index") {
                base
            } else {
                binary(
                    builder,
                    decoded.location,
                    IntegerBinaryKind::Add,
                    base,
                    Immediate::I64(offset as u64).into(),
                )?
            };
            if family.ends_with("pre-index") || family.ends_with("post-index") {
                writeback = Some(binary(
                    builder,
                    decoded.location,
                    IntegerBinaryKind::Add,
                    base,
                    Immediate::I64(offset as u64).into(),
                )?);
            }
            bitcast(builder, decoded.location, transfer_base, IrType::Address)?
        }
    };
    let descriptor = descriptor(size, MemoryOrdering::Relaxed, MemoryAccessClass::Normal);
    let rt = (bits & 31) as u8;
    if literal || bits & (1 << 22) != 0 {
        let raw = emit_one(
            builder,
            decoded.location,
            descriptor.value_type(),
            OperationKind::Memory(MemoryOperation::Load {
                address,
                descriptor,
            }),
        )?;
        let vector = helper(
            builder,
            decoded.location,
            "a64.simd.zero-extend-load",
            vec![raw.into()],
            &[IrType::V128],
            OperationEffects::default(),
        )?[0];
        vector_write(builder, decoded.location, rt, vector.into())?;
    } else {
        let vector = vector_read(builder, decoded.location, rt)?;
        let raw = helper(
            builder,
            decoded.location,
            "a64.simd.low-bits",
            vec![vector],
            &[descriptor.value_type()],
            OperationEffects::default(),
        )?[0];
        builder.emit(
            decoded.location,
            &[],
            OperationKind::Memory(MemoryOperation::Store {
                address,
                value: raw.into(),
                descriptor,
            }),
        )?;
    }
    if let Some(updated) = writeback {
        write_gpr(
            builder,
            decoded.location,
            rn,
            updated,
            Register31::StackPointer,
        )?;
    }
    Ok(LiftOutcome::Continue)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        address::{AddressSpaceId, GuestPhysicalPageId},
        ir::{block::BlockExitKind, op::OperationKind},
        location::ExecutionState,
        memory::{MemoryPermissions, SyntheticMemory},
        profile::GuestCpuProfile,
        translate::{BlockTranslationConfig, translate_block},
    };

    const SPACE: AddressSpaceId = AddressSpaceId::new(11);

    fn translate_with_profile(
        profile: GuestCpuProfile,
        words: &[u32],
    ) -> crate::ir::block::IrBlock {
        let mut memory = SyntheticMemory::new();
        assert!(memory.add_ram_page(GuestPhysicalPageId::new(1)));
        assert!(memory.map_page(
            SPACE,
            GuestVirtualAddress::new(0x1000),
            GuestPhysicalPageId::new(1),
            MemoryPermissions::READ_EXECUTE,
        ));
        for (index, word) in words.iter().enumerate() {
            assert!(memory.initialize_ram(
                GuestPhysicalPageId::new(1),
                index * 4,
                &word.to_le_bytes(),
            ));
        }
        translate_block(
            BlockTranslationConfig::default(),
            &profile,
            SPACE,
            LocationDescriptor::new(
                GuestVirtualAddress::new(0x1000),
                ExecutionState::A64,
                profile.id(),
            ),
            &memory,
        )
        .unwrap()
    }

    fn translate(words: &[u32]) -> crate::ir::block::IrBlock {
        translate_with_profile(GuestCpuProfile::switch_1(), words)
    }

    #[test]
    fn integer_loop_has_typed_flags_and_two_explicit_successors() {
        // movz x0,#3; subs x0,x0,#1; b.ne -4
        let block = translate(&[0xd280_0060, 0xf100_0400, 0x54ff_ffe1]);
        assert_eq!(block.metadata.guest_instruction_count, 3);
        assert!(
            block
                .operations
                .iter()
                .any(|operation| matches!(operation.kind, OperationKind::Flags(_)))
        );
        assert_eq!(block.metadata.exits.len(), 2);
        assert_eq!(
            block.metadata.exits[0].kind,
            BlockExitKind::ConditionalTaken
        );
        assert_eq!(
            block.metadata.exits[0].target,
            Some(GuestVirtualAddress::new(0x1004))
        );
        assert_eq!(
            block.metadata.exits[1].target,
            Some(GuestVirtualAddress::new(0x100c))
        );
    }

    #[test]
    fn function_call_and_return_update_lr_and_keep_indirect_target_in_guest_domain() {
        let call = translate(&[0x9400_0002]);
        assert!(
            matches!(call.terminator, Terminator::Call { return_address, .. } if return_address == GuestVirtualAddress::new(0x1004))
        );
        assert!(call.operations.iter().any(|operation| matches!(operation.kind, OperationKind::WriteState { register: StateRegister::A64X(register), .. } if register.index() == 30)));

        let ret = translate(&[0xd65f_03c0]);
        assert!(matches!(
            ret.terminator,
            Terminator::Return {
                target: ControlTarget::Indirect {
                    execution_state: ExecutionState::A64,
                    ..
                }
            }
        ));
    }

    #[test]
    fn stack_memory_and_svc_have_precise_ir_boundaries() {
        // str x0,[sp,#8]; ldr x1,[sp,#8]; svc #7
        let block = translate(&[0xf900_07e0, 0xf940_07e1, 0xd400_00e1]);
        assert_eq!(block.metadata.guest_instruction_count, 3);
        assert_eq!(
            block
                .operations
                .iter()
                .filter(|operation| matches!(operation.kind, OperationKind::Memory(_)))
                .count(),
            2
        );
        assert!(matches!(
            block.terminator,
            Terminator::Exception {
                kind: ExceptionKind::SupervisorCall,
                syndrome: Some(7),
                ..
            }
        ));
    }

    #[test]
    fn narrow_integer_memory_transfers_extend_and_truncate_at_gpr_boundaries() {
        let block = translate(&[
            0x3900_0020, // strb w0,[x1]
            0x3940_0022, // ldrb w2,[x1]
            0x3980_0023, // ldrsb x3,[x1]
            0x39c0_0024, // ldrsb w4,[x1]
            0x7900_0020, // strh w0,[x1]
            0x7940_0022, // ldrh w2,[x1]
            0x7980_0023, // ldrsh x3,[x1]
            0x79c0_0024, // ldrsh w4,[x1]
            0xd400_0001,
        ]);
        assert_eq!(block.metadata.guest_instruction_count, 9);
        assert_eq!(
            block
                .operations
                .iter()
                .filter(|operation| matches!(operation.kind, OperationKind::Memory(_)))
                .count(),
            8
        );
    }

    #[test]
    fn signed_literal_and_acquire_release_transfers_do_not_fall_back() {
        let block = translate(&[
            0x9800_0000, // ldrsw x0, literal
            0xc8df_fc20, // ldar x0,[x1]
            0xc89f_fc20, // stlr x0,[x1]
            0xd400_0001,
        ]);
        assert_eq!(block.metadata.guest_instruction_count, 4);
        assert_eq!(
            block
                .operations
                .iter()
                .filter(|operation| matches!(operation.kind, OperationKind::Memory(_)))
                .count(),
            3
        );
        assert!(matches!(
            block.terminator,
            Terminator::Exception {
                kind: ExceptionKind::SupervisorCall,
                ..
            }
        ));
    }

    #[test]
    fn barriers_cache_and_exclusives_remain_semantic_ir_operations() {
        let barrier = translate(&[
            0xd503_3bbf,
            0xd503_3fdf,
            0xd50b_7b20,
            0xc85f_7c20,
            0xc800_7c41,
            0xd400_0001,
        ]);
        assert!(barrier.operations.iter().any(|operation| matches!(
            operation.kind,
            OperationKind::Barrier(BarrierOperation::DataMemory { .. })
        )));
        assert!(barrier.operations.iter().any(|operation| matches!(
            operation.kind,
            OperationKind::Barrier(BarrierOperation::InstructionSynchronization)
        )));
        assert!(
            barrier
                .operations
                .iter()
                .any(|operation| matches!(operation.kind, OperationKind::CacheMaintenance(_)))
        );
        assert!(barrier.operations.iter().any(|operation| matches!(
            operation.kind,
            OperationKind::Exclusive(ExclusiveOperation::Load { .. })
        )));
        assert!(barrier.operations.iter().any(|operation| matches!(
            operation.kind,
            OperationKind::Exclusive(ExclusiveOperation::Store { .. })
        )));
    }

    #[test]
    fn representative_integer_families_form_one_verified_block() {
        let block = translate(&[
            0xd280_0020, // movz x0,#1
            0x8b01_0000, // add x0,x0,x1
            0x9a01_0000, // adc x0,x0,x1
            0xaa01_0000, // orr x0,x0,x1
            0xd340_fc00, // ubfm x0,x0,#0,#63
            0x93c1_0400, // extr x0,x0,x1,#1
            0x9ac1_2000, // lslv x0,x0,x1
            0x9a81_0000, // csel x0,x0,x1,eq
            0x9b01_0800, // madd x0,x0,x1,x2
            0xdac0_1000, // clz x0,x0
            0xd400_0001, // svc #0
        ]);
        assert_eq!(block.metadata.guest_instruction_count, 11);
        assert!(matches!(
            block.terminator,
            Terminator::Exception {
                kind: ExceptionKind::SupervisorCall,
                ..
            }
        ));
    }

    #[test]
    fn fp_and_simd_use_explicit_helpers_and_status_state() {
        use crate::profile::{CapabilityStatus, InstructionFeature};

        let profile = GuestCpuProfile::switch_1()
            .with_instruction_feature(InstructionFeature::AdvancedSimd, CapabilityStatus::Enabled);
        let block = translate_with_profile(
            profile,
            &[
                0x4e20_1c00, // and v0.16b,v0.16b,v0.16b
                0x4e20_8400, // add v0.16b,v0.16b,v0.16b
                0x1e60_4000, // fmov d0,d0
                0x1e61_2800, // fadd d0,d0,d1
                0x1e61_2000, // fcmp d0,d1
                0x9e62_0000, // scvtf d0,x0
                0x9e78_0001, // fcvtzs x1,d0
                0x9e66_0002, // fmov x2,d0
                0x9e67_0040, // fmov d0,x2
                0x3dc0_0000, // ldr q0,[x0]
                0xd400_0001,
            ],
        );
        assert_eq!(block.metadata.guest_instruction_count, 11);
        assert!(block.operations.iter().any(|operation| matches!(
            operation.kind,
            OperationKind::Helper(ref helper) if helper.helper.as_ref() == "a64.fp.scalar-arithmetic"
        )));
        assert!(block.operations.iter().any(|operation| matches!(
            operation.kind,
            OperationKind::WriteState {
                register: StateRegister::A64Fpsr,
                ..
            }
        )));
        assert!(block.operations.iter().any(|operation| matches!(
            operation.kind,
            OperationKind::WriteState {
                register: StateRegister::A64Nzcv,
                ..
            }
        )));
    }
}
