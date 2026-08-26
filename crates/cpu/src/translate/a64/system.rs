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
    if let Some(operation) = crate::semantics::a64::hint_operation(fields.hint) {
        if operation == crate::semantics::a64::HintOperation::NoOperation {
            return Ok(LiftOutcome::Continue);
        }
        builder.emit(
            decoded.location,
            &[],
            OperationKind::ProcessorHint(operation),
        )?;
        return Ok(LiftOutcome::Continue);
    }
    if matches!(fields.hint, 32 | 34 | 36 | 38) {
        return Ok(LiftOutcome::Continue);
    }
    Ok(unsupported(decoded))
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
    let value = if let Some(register) = system_register(fields.system_key) {
        let value = if register == StateRegister::A64Nzcv {
            let flags = read_flags(builder, decoded.location)?;
            emit_one(
                builder,
                decoded.location,
                IrType::I32,
                OperationKind::Flags(FlagOperation::Materialize { flags }),
            )?
        } else {
            emit_one(
                builder,
                decoded.location,
                register.ty(),
                OperationKind::ReadState(register),
            )?
        };
        if register.ty() == IrType::I32 {
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
        }
    } else if matches!(
        fields.system_key,
        0xd53b_0020 | 0xd53b_00e0 | 0xd53b_e000 | 0xd53b_e020
    ) {
        emit_one(
            builder,
            decoded.location,
            IrType::I64,
            OperationKind::RuntimeRegisterRead(fields.system_key),
        )?
        .into()
    } else {
        return Ok(unsupported(decoded));
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
        return Ok(unsupported(decoded));
    };
    if register == StateRegister::A64TpidrroEl0 {
        return Ok(unsupported(decoded));
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
    if register == StateRegister::A64Nzcv {
        let flags = emit_one(
            builder,
            decoded.location,
            IrType::Flags,
            OperationKind::Flags(FlagOperation::FromPacked { value }),
        )?;
        write_flags(builder, decoded.location, flags.into())?;
    } else {
        builder.emit(
            decoded.location,
            &[],
            OperationKind::WriteState { register, value },
        )?;
    }
    Ok(LiftOutcome::Continue)
}

fn lift_barrier(
    builder: &mut IrBuilder,
    decoded: &DecodedInstruction<DecodedOpcode>,
    fields: SystemOperands,
) -> Result<LiftOutcome, BuildError> {
    let Some(operation) =
        crate::semantics::a64::barrier_operation(fields.barrier_opcode, fields.barrier_option)
    else {
        return Ok(unsupported(decoded));
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
        _ => return Ok(unsupported(decoded)),
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
