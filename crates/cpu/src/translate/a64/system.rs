use super::*;

use crate::{
    decode::{
        DecodedOpcode,
        a64::system::{Instruction as SystemInstruction, Operands as SystemOperands},
    },
    ir::builder::{BuildError, IrBuilder},
    location::DecodedInstruction,
};

use super::LiftOutcome;

pub(super) fn lift(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    instruction: SystemInstruction,
) -> Result<LiftOutcome, BuildError> {
    let fields = instruction.operands();
    match instruction {
        SystemInstruction::Hint(_) => lift_hint(builder, decoded, fields),
        SystemInstruction::ReadRegister(_) => lift_mrs(builder, decoded, fields),
        SystemInstruction::WriteRegister(_) => lift_msr(builder, decoded, fields),
        SystemInstruction::Barrier(_) => lift_barrier(builder, decoded, fields),
        SystemInstruction::System(_) => lift_system(builder, decoded, fields),
    }
}

fn lift_hint(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    fields: SystemOperands,
) -> Result<LiftOutcome, BuildError> {
    let immediate = u32::from(fields.hint);
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

fn system_register(system_key: u32) -> Option<StateRegister> {
    match system_key {
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
    fields: SystemOperands,
) -> Result<LiftOutcome, BuildError> {
    let Some(register) = system_register(fields.system_key) else {
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
        fields.rt,
        value,
        Register31::Zero,
    )?;
    Ok(LiftOutcome::Continue)
}

fn lift_msr(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    fields: SystemOperands,
) -> Result<LiftOutcome, BuildError> {
    let Some(register) = system_register(fields.system_key) else {
        return Ok(interpret(decoded));
    };
    if register == StateRegister::A64TpidrroEl0 {
        return Ok(interpret(decoded));
    }
    let mut value = read_gpr(
        builder,
        decoded.location,
        fields.rt,
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
    fields: SystemOperands,
) -> Result<LiftOutcome, BuildError> {
    let opcode = u32::from(fields.barrier_opcode);
    let option = fields.barrier_option;
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
    fields: SystemOperands,
) -> Result<LiftOutcome, BuildError> {
    let (kind, uses_address) = match fields.system_key {
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
            fields.rt,
            IrType::I64,
            Register31::Zero,
        )?;
        Some(guest_address_from_integer(builder, decoded.location, raw)?)
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
