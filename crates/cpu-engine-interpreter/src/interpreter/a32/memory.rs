use super::{InterpreterContext, InterpreterError};
use crate::interpreter::aarch32::{
    SemanticControl, execute_multiple, execute_single, read_register, write_register,
};
use nixe_cpu::{
    decode::{DecodedOpcode, a32::memory::Instruction, aarch32::ExclusiveTransfer},
    location::DecodedInstruction,
    memory::{
        CpuMemory, MemoryAccess, MemoryAccessClass, MemoryAccessSize, MemoryAlignment,
        MemoryOrdering, MemoryValue,
    },
    state::a32::A32State,
};
use nixe_memory::{AddressSpaceId, GuestVirtualAddress};

pub(super) enum Execution {
    Control(SemanticControl),
    Fault(nixe_cpu::memory::DataAccessFault),
}

pub(super) fn execute(
    context: InterpreterContext<'_>,
    state: &mut A32State,
    decoded: &DecodedInstruction<DecodedOpcode>,
    instruction: Instruction,
) -> Result<Execution, InterpreterError> {
    let Some(memory) = context.memory() else {
        return Err(super::super::unsupported(decoded));
    };
    let address_space = context.process().address_space_id();
    let result = match instruction {
        Instruction::Single(instruction) => {
            execute_single(memory, address_space, state, instruction)
        }
        Instruction::Multiple(instruction) => {
            execute_multiple(memory, address_space, state, instruction)
        }
        Instruction::Exclusive(transfer) => {
            exclusive(context, memory, address_space, state, transfer)
        }
        Instruction::AcquireRelease(transfer) => {
            acquire_release(memory, address_space, state, transfer)
        }
    };
    Ok(match result {
        Ok(control) => Execution::Control(control),
        Err(fault) => Execution::Fault(fault),
    })
}

fn exclusive(
    context: InterpreterContext<'_>,
    memory: &dyn CpuMemory,
    address_space: AddressSpaceId,
    state: &mut A32State,
    transfer: ExclusiveTransfer,
) -> Result<SemanticControl, nixe_cpu::memory::DataAccessFault> {
    let Some(monitor) = context.exclusive_monitor() else {
        return Ok(SemanticControl::Continue);
    };
    let address = GuestVirtualAddress::new(u64::from(read_register(state, transfer.rn, false)));
    let access = MemoryAccess::new(
        MemoryAccessSize::Word,
        MemoryAlignment::Natural,
        if transfer.load {
            MemoryOrdering::Acquire
        } else {
            MemoryOrdering::Release
        },
        MemoryAccessClass::Exclusive,
    );

    // Arm A-profile LDREX/STREX use a PE-local monitor and a physical-memory
    // reservation; see DDI0602 AArch32 LDREX and STREX instruction pages:
    // https://developer.arm.com/documentation/ddi0602/latest/AArch32-Instructions/LDREX
    // https://developer.arm.com/documentation/ddi0602/latest/AArch32-Instructions/STREX
    if transfer.load {
        let (value, reservation) = memory.load_exclusive(address_space, address, access)?;
        monitor.borrow_mut().reserve(reservation);
        let MemoryValue::U32(value) = value.value else {
            unreachable!("word exclusive load returns a 32-bit value")
        };
        write_register(state, transfer.rt, value)
            .expect("normalized exclusive destination is not PC");
    } else {
        let reservation = monitor.borrow().reservation();
        monitor.borrow_mut().clear();
        let succeeded = if let Some(reservation) = reservation {
            memory
                .store_exclusive(
                    address_space,
                    address,
                    access,
                    MemoryValue::U32(read_register(state, transfer.rt, false)),
                    reservation,
                )?
                .1
        } else {
            false
        };
        write_register(
            state,
            transfer
                .status
                .expect("exclusive store has a status register"),
            u32::from(!succeeded),
        )
        .expect("normalized exclusive status destination is not PC");
    }
    Ok(SemanticControl::Continue)
}

fn acquire_release(
    memory: &dyn CpuMemory,
    address_space: AddressSpaceId,
    state: &mut A32State,
    transfer: ExclusiveTransfer,
) -> Result<SemanticControl, nixe_cpu::memory::DataAccessFault> {
    let address = GuestVirtualAddress::new(u64::from(read_register(state, transfer.rn, false)));
    let access = MemoryAccess::new(
        MemoryAccessSize::Word,
        MemoryAlignment::Natural,
        if transfer.load {
            MemoryOrdering::Acquire
        } else {
            MemoryOrdering::Release
        },
        MemoryAccessClass::Normal,
    );

    // Arm A-profile LDA/STL provide acquire/release ordering without an
    // exclusive monitor; see DDI0602 AArch32 load-acquire/store-release pages:
    // https://developer.arm.com/documentation/ddi0602/latest/AArch32-Instructions/LDA
    // https://developer.arm.com/documentation/ddi0602/latest/AArch32-Instructions/STL
    if transfer.load {
        let value = memory.read(address_space, address, access)?.value;
        let MemoryValue::U32(value) = value else {
            unreachable!("word acquire load returns a 32-bit value")
        };
        write_register(state, transfer.rt, value)
            .expect("normalized acquire destination is not PC");
    } else {
        memory.write(
            address_space,
            address,
            access,
            MemoryValue::U32(read_register(state, transfer.rt, false)),
        )?;
    }
    Ok(SemanticControl::Continue)
}
