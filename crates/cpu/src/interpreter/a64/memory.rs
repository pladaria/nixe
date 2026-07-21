use crate::{
    address::GuestVirtualAddress,
    decode::{
        DecodedOpcode,
        a64::memory::{Instruction, Operands},
    },
    location::DecodedInstruction,
    memory::{
        CpuMemory, DataAccessFault, MemoryAccess, MemoryAccessClass, MemoryAccessSize,
        MemoryAlignment, MemoryOrdering, MemoryValue,
    },
    state::a64::A64State,
};

use super::{advance, read, resume, sign_extend, write};
use crate::interpreter::{InterpreterContext, InterpreterError, InterpreterOutcome};

type MemoryStep = Result<Option<()>, DataAccessFault>;

pub(super) fn execute(
    context: InterpreterContext<'_>,
    state: &mut A64State,
    decoded: &DecodedInstruction<DecodedOpcode>,
    instruction: Instruction,
) -> Result<InterpreterOutcome, InterpreterError> {
    let Some(memory) = context.memory() else {
        return Err(super::super::unsupported(decoded));
    };
    let address_space = context.process().address_space_id();
    let fields = instruction.operands();
    let result = match instruction {
        Instruction::Literal(_) => literal(memory, address_space, state, decoded, fields),
        Instruction::Unsigned(_) => unsigned(memory, address_space, state, fields),
        Instruction::Unscaled(_) | Instruction::PostIndex(_) | Instruction::PreIndex(_) => {
            indexed(memory, address_space, state, fields, instruction)
        }
        Instruction::Register(_) => register_offset(memory, address_space, state, fields),
        Instruction::Pair(_) => pair(memory, address_space, state, fields),
        Instruction::LoadAcquire(_) | Instruction::StoreRelease(_) => {
            acquire_release(memory, address_space, state, fields, instruction)
        }
        Instruction::LoadExclusive(_) | Instruction::StoreExclusive(_) => Ok(None),
    };
    match result {
        Ok(Some(())) => {
            advance(state);
            Ok(resume(state, decoded))
        }
        Ok(None) => Err(super::super::unsupported(decoded)),
        Err(fault) => Ok(InterpreterOutcome::DataAbort {
            source: decoded.location,
            fault,
        }),
    }
}

fn access(size: MemoryAccessSize, ordering: MemoryOrdering, aligned: bool) -> MemoryAccess {
    MemoryAccess::new(
        size,
        if aligned {
            MemoryAlignment::Natural
        } else {
            MemoryAlignment::Unaligned
        },
        ordering,
        MemoryAccessClass::Normal,
    )
}

fn size_from_bits(size: u8) -> MemoryAccessSize {
    match size {
        0 => MemoryAccessSize::Byte,
        1 => MemoryAccessSize::Halfword,
        2 => MemoryAccessSize::Word,
        3 => MemoryAccessSize::Doubleword,
        _ => unreachable!(),
    }
}

fn literal(
    memory: &dyn CpuMemory,
    address_space: crate::address::AddressSpaceId,
    state: &mut A64State,
    decoded: &DecodedInstruction<DecodedOpcode>,
    fields: Operands,
) -> MemoryStep {
    let (size, signed) = match fields.size {
        0 => (MemoryAccessSize::Word, false),
        1 => (MemoryAccessSize::Doubleword, false),
        2 => (MemoryAccessSize::Word, true),
        _ => return Ok(None),
    };
    let address = decoded
        .location
        .pc
        .wrapping_offset(sign_extend(u64::from(fields.immediate_19), 19) << 2);
    let value = memory.read(
        address_space,
        address,
        access(size, MemoryOrdering::Relaxed, false),
    )?;
    write_loaded(state, fields.rt, size, fields.opc, signed, value.value);
    Ok(Some(()))
}

fn unsigned(
    memory: &dyn CpuMemory,
    address_space: crate::address::AddressSpaceId,
    state: &mut A64State,
    fields: Operands,
) -> MemoryStep {
    let size = size_from_bits(fields.size);
    let base = read(state, fields.rn, 64, true);
    let address = GuestVirtualAddress::new(
        base.wrapping_add(u64::from(fields.immediate_12) * size.bytes() as u64),
    );
    transfer(
        memory,
        address_space,
        state,
        fields,
        address,
        size,
        access(size, MemoryOrdering::Relaxed, false),
    )
}

fn indexed(
    memory: &dyn CpuMemory,
    address_space: crate::address::AddressSpaceId,
    state: &mut A64State,
    fields: Operands,
    instruction: Instruction,
) -> MemoryStep {
    if !matches!(instruction, Instruction::Unscaled(_)) && fields.rn != 31 && fields.rn == fields.rt
    {
        return Ok(None);
    }
    let size = size_from_bits(fields.size);
    let base = read(state, fields.rn, 64, true);
    let offset = sign_extend(u64::from(fields.immediate_9), 9);
    let address = if matches!(instruction, Instruction::PreIndex(_)) {
        base.wrapping_add_signed(offset)
    } else {
        base
    };
    if transfer(
        memory,
        address_space,
        state,
        fields,
        GuestVirtualAddress::new(address),
        size,
        access(size, MemoryOrdering::Relaxed, false),
    )?
    .is_none()
    {
        return Ok(None);
    }
    if !matches!(instruction, Instruction::Unscaled(_)) {
        write(state, fields.rn, 64, true, base.wrapping_add_signed(offset));
    }
    Ok(Some(()))
}

fn register_offset(
    memory: &dyn CpuMemory,
    address_space: crate::address::AddressSpaceId,
    state: &mut A64State,
    fields: Operands,
) -> MemoryStep {
    if fields.option & 2 == 0 {
        return Ok(None);
    }
    let size = size_from_bits(fields.size);
    let raw = read(state, fields.rm, 64, false);
    let source_width = if fields.option & 1 == 0 { 32 } else { 64 };
    let mut offset = if source_width == 32 {
        u64::from(raw as u32)
    } else {
        raw
    };
    if fields.option & 4 != 0 {
        offset = sign_extend(offset, source_width) as u64;
    }
    if fields.scaled {
        offset = offset.wrapping_shl(size.bytes().trailing_zeros());
    }
    let address = GuestVirtualAddress::new(read(state, fields.rn, 64, true).wrapping_add(offset));
    transfer(
        memory,
        address_space,
        state,
        fields,
        address,
        size,
        access(size, MemoryOrdering::Relaxed, false),
    )
}

fn pair(
    memory: &dyn CpuMemory,
    address_space: crate::address::AddressSpaceId,
    state: &mut A64State,
    fields: Operands,
) -> MemoryStep {
    let (size, signed) = match fields.size {
        0 => (MemoryAccessSize::Word, false),
        1 if fields.load => (MemoryAccessSize::Word, true),
        2 => (MemoryAccessSize::Doubleword, false),
        _ => return Ok(None),
    };
    if (fields.load && fields.rt == fields.rt2)
        || (matches!(fields.mode, 1 | 3)
            && fields.rn != 31
            && (fields.rn == fields.rt || fields.rn == fields.rt2))
    {
        return Ok(None);
    }
    let base = read(state, fields.rn, 64, true);
    let offset = sign_extend(u64::from(fields.immediate_7), 7) * size.bytes() as i64;
    let transfer_base = if fields.mode == 3 {
        base.wrapping_add_signed(offset)
    } else {
        base
    };
    let first = GuestVirtualAddress::new(transfer_base);
    let second = first.wrapping_add(size.bytes() as u64);
    let descriptor = access(size, MemoryOrdering::Relaxed, false);
    if fields.load {
        // Delay register writes until both reads succeed, preserving precise
        // state for synthetic faults in the reference engine.
        let first_value = memory.read(address_space, first, descriptor)?.value;
        let second_value = memory.read(address_space, second, descriptor)?.value;
        write_loaded(
            state,
            fields.rt,
            size,
            u8::from(signed) * 2 + 1,
            signed,
            first_value,
        );
        write_loaded(
            state,
            fields.rt2,
            size,
            u8::from(signed) * 2 + 1,
            signed,
            second_value,
        );
    } else {
        memory.write(
            address_space,
            first,
            descriptor,
            register_value(state, fields.rt, size),
        )?;
        memory.write(
            address_space,
            second,
            descriptor,
            register_value(state, fields.rt2, size),
        )?;
    }
    if matches!(fields.mode, 1 | 3) {
        write(state, fields.rn, 64, true, base.wrapping_add_signed(offset));
    }
    Ok(Some(()))
}

fn acquire_release(
    memory: &dyn CpuMemory,
    address_space: crate::address::AddressSpaceId,
    state: &mut A64State,
    fields: Operands,
    instruction: Instruction,
) -> MemoryStep {
    let load = matches!(instruction, Instruction::LoadAcquire(_));
    let ordering = if load {
        MemoryOrdering::Acquire
    } else {
        MemoryOrdering::Release
    };
    let size = size_from_bits(fields.size);
    let address = GuestVirtualAddress::new(read(state, fields.rn, 64, true));
    let descriptor = access(size, ordering, true);
    if load {
        let value = memory.read(address_space, address, descriptor)?.value;
        write_loaded(state, fields.rt, size, 1, false, value);
    } else {
        memory.write(
            address_space,
            address,
            descriptor,
            register_value(state, fields.rt, size),
        )?;
    }
    Ok(Some(()))
}

fn transfer(
    memory: &dyn CpuMemory,
    address_space: crate::address::AddressSpaceId,
    state: &mut A64State,
    fields: Operands,
    address: GuestVirtualAddress,
    size: MemoryAccessSize,
    descriptor: MemoryAccess,
) -> MemoryStep {
    match fields.opc {
        0 => {
            memory.write(
                address_space,
                address,
                descriptor,
                register_value(state, fields.rt, size),
            )?;
        }
        1 => {
            let value = memory.read(address_space, address, descriptor)?.value;
            write_loaded(state, fields.rt, size, fields.opc, false, value);
        }
        2 if size != MemoryAccessSize::Doubleword => {
            let value = memory.read(address_space, address, descriptor)?.value;
            write_loaded(state, fields.rt, size, fields.opc, true, value);
        }
        3 if !matches!(size, MemoryAccessSize::Word | MemoryAccessSize::Doubleword) => {
            let value = memory.read(address_space, address, descriptor)?.value;
            write_loaded(state, fields.rt, size, fields.opc, true, value);
        }
        _ => return Ok(None),
    }
    Ok(Some(()))
}

fn register_value(state: &A64State, register: u8, size: MemoryAccessSize) -> MemoryValue {
    let value = read(
        state,
        register,
        if size == MemoryAccessSize::Doubleword {
            64
        } else {
            32
        },
        false,
    );
    match size {
        MemoryAccessSize::Byte => MemoryValue::U8(value as u8),
        MemoryAccessSize::Halfword => MemoryValue::U16(value as u16),
        MemoryAccessSize::Word => MemoryValue::U32(value as u32),
        MemoryAccessSize::Doubleword => MemoryValue::U64(value),
        MemoryAccessSize::Quadword => unreachable!("A64 scalar transfer is at most 64 bits"),
    }
}

fn write_loaded(
    state: &mut A64State,
    register: u8,
    size: MemoryAccessSize,
    opc: u8,
    signed: bool,
    value: MemoryValue,
) {
    let raw = match value {
        MemoryValue::U8(value) => u64::from(value),
        MemoryValue::U16(value) => u64::from(value),
        MemoryValue::U32(value) => u64::from(value),
        MemoryValue::U64(value) => value,
        MemoryValue::U128(_) => unreachable!("A64 scalar transfer is at most 64 bits"),
    };
    let destination_width = if opc == 2 || size == MemoryAccessSize::Doubleword || signed {
        64
    } else {
        32
    };
    let result = if signed {
        sign_extend(raw, (size.bytes() * 8) as u8) as u64
    } else {
        raw
    };
    write(state, register, destination_width, false, result);
}
