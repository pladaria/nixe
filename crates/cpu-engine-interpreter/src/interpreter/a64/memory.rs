use nixe_cpu::{
    decode::{
        DecodedOpcode,
        a64::memory::{Instruction, Operands},
    },
    location::DecodedInstruction,
    memory::{
        CpuMemory, DataAccessFault, MemoryAccess, MemoryAccessClass, MemoryAccessSize,
        MemoryAlignment, MemoryOrdering, MemoryValue,
    },
    semantics::a64::{
        LoadSpec, ScalarTransfer, literal_load, memory_size, pair_transfer, scalar_transfer,
    },
    state::a64::A64State,
};
use nixe_memory::{AddressSpaceId, GuestVirtualAddress};

use super::{advance, read, register_offset_address, resume, sign_extend, write};
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
        Instruction::LoadExclusive(_) | Instruction::StoreExclusive(_) => {
            exclusive(context, memory, address_space, state, fields, instruction)
        }
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

fn exclusive(
    context: InterpreterContext<'_>,
    memory: &dyn CpuMemory,
    address_space: AddressSpaceId,
    state: &mut A64State,
    fields: Operands,
    instruction: Instruction,
) -> MemoryStep {
    let Some(monitor) = context.exclusive_monitor() else {
        return Ok(None);
    };
    let size = memory_size(fields.size);
    let address = GuestVirtualAddress::new(read(state, fields.rn, 64, true));
    let load = matches!(instruction, Instruction::LoadExclusive(_));
    let ordering = match (load, fields.ordered) {
        (true, true) => MemoryOrdering::Acquire,
        (false, true) => MemoryOrdering::Release,
        (_, false) => MemoryOrdering::Relaxed,
    };
    let descriptor = MemoryAccess::new(
        size,
        MemoryAlignment::Natural,
        ordering,
        MemoryAccessClass::Exclusive,
    );
    if load {
        let (value, reservation) = memory.load_exclusive(address_space, address, descriptor)?;
        monitor.borrow_mut().reserve(reservation);
        write_loaded(
            state,
            fields.rt,
            size,
            LoadSpec::unsigned(size),
            value.value,
        );
    } else {
        let reservation = monitor.borrow().reservation();
        monitor.borrow_mut().clear();
        let succeeded = if let Some(reservation) = reservation {
            memory
                .store_exclusive(
                    address_space,
                    address,
                    descriptor,
                    register_value(state, fields.rt, size),
                    reservation,
                )?
                .1
        } else {
            false
        };
        write(state, fields.rm, 32, false, u64::from(!succeeded));
    }
    Ok(Some(()))
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

fn literal(
    memory: &dyn CpuMemory,
    address_space: AddressSpaceId,
    state: &mut A64State,
    decoded: &DecodedInstruction<DecodedOpcode>,
    fields: Operands,
) -> MemoryStep {
    let Some((size, load)) = literal_load(fields.size) else {
        return Ok(None);
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
    write_loaded(state, fields.rt, size, load, value.value);
    Ok(Some(()))
}

fn unsigned(
    memory: &dyn CpuMemory,
    address_space: AddressSpaceId,
    state: &mut A64State,
    fields: Operands,
) -> MemoryStep {
    let size = memory_size(fields.size);
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
    address_space: AddressSpaceId,
    state: &mut A64State,
    fields: Operands,
    instruction: Instruction,
) -> MemoryStep {
    if !matches!(instruction, Instruction::Unscaled(_)) && fields.rn != 31 && fields.rn == fields.rt
    {
        return Ok(None);
    }
    let size = memory_size(fields.size);
    let base = read(state, fields.rn, 64, true);
    let offset = sign_extend(u64::from(fields.immediate_9), 9);
    let address = if matches!(
        instruction,
        Instruction::Unscaled(_) | Instruction::PreIndex(_)
    ) {
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
    address_space: AddressSpaceId,
    state: &mut A64State,
    fields: Operands,
) -> MemoryStep {
    let size = memory_size(fields.size);
    let Some(address) = register_offset_address(
        state,
        fields.rn,
        fields.rm,
        fields.option,
        fields.scaled,
        size.bytes().trailing_zeros(),
    ) else {
        return Ok(None);
    };
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
    address_space: AddressSpaceId,
    state: &mut A64State,
    fields: Operands,
) -> MemoryStep {
    let Some((size, load_spec)) = pair_transfer(fields.size, fields.load) else {
        return Ok(None);
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
    let transfer_base = if matches!(fields.mode, 2 | 3) {
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
        write_loaded(state, fields.rt, size, load_spec, first_value);
        write_loaded(state, fields.rt2, size, load_spec, second_value);
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
    address_space: AddressSpaceId,
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
    let size = memory_size(fields.size);
    let address = GuestVirtualAddress::new(read(state, fields.rn, 64, true));
    let descriptor = access(size, ordering, true);
    if load {
        let value = memory.read(address_space, address, descriptor)?.value;
        write_loaded(state, fields.rt, size, LoadSpec::unsigned(size), value);
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
    address_space: AddressSpaceId,
    state: &mut A64State,
    fields: Operands,
    address: GuestVirtualAddress,
    size: MemoryAccessSize,
    descriptor: MemoryAccess,
) -> MemoryStep {
    match scalar_transfer(fields.opc, size) {
        Some(ScalarTransfer::Store) => {
            memory.write(
                address_space,
                address,
                descriptor,
                register_value(state, fields.rt, size),
            )?;
        }
        Some(ScalarTransfer::Load(load)) => {
            let value = memory.read(address_space, address, descriptor)?.value;
            write_loaded(state, fields.rt, size, load, value);
        }
        None => return Ok(None),
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
    load: LoadSpec,
    value: MemoryValue,
) {
    let raw = match value {
        MemoryValue::U8(value) => u64::from(value),
        MemoryValue::U16(value) => u64::from(value),
        MemoryValue::U32(value) => u64::from(value),
        MemoryValue::U64(value) => value,
        MemoryValue::U128(_) => unreachable!("A64 scalar transfer is at most 64 bits"),
    };
    let result = if load.signed {
        sign_extend(raw, (size.bytes() * 8) as u8) as u64
    } else {
        raw
    };
    write(state, register, load.destination_bits, false, result);
}
