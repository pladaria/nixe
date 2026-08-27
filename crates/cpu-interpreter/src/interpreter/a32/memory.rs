use super::{InterpreterContext, InterpreterError};
use crate::interpreter::aarch32::{
    SemanticControl, execute_multiple, execute_single, read_register, write_register,
};
use nixe_cpu::{
    decode::{
        DecodedOpcode,
        a32::memory::Instruction,
        aarch32::{AcquireReleaseTransfer, ExclusiveTransfer, MemorySize},
    },
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
    _decoded: &DecodedInstruction<DecodedOpcode>,
    instruction: Instruction,
) -> Result<Execution, InterpreterError> {
    let memory = context.memory();
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
    let monitor = context.exclusive_monitor();

    // Arm A-profile LDREX/STREX use a PE-local monitor and a physical-memory
    // reservation; see DDI0602 AArch32 LDREX and STREX instruction pages:
    // https://developer.arm.com/documentation/ddi0602/latest/AArch32-Instructions/LDREX
    // https://developer.arm.com/documentation/ddi0602/latest/AArch32-Instructions/STREX
    match transfer {
        ExclusiveTransfer::Load { size, rn, rt } => {
            let address = transfer_address(state, rn);
            let access =
                transfer_access(size, MemoryOrdering::Relaxed, MemoryAccessClass::Exclusive);
            let (value, reservation) = memory.load_exclusive(address_space, address, access)?;
            monitor.borrow_mut().reserve(reservation);
            let MemoryValue::U32(value) = value.value else {
                unreachable!("word exclusive load returns a 32-bit value")
            };
            write_register(state, rt, value).expect("normalized exclusive destination is not PC");
        }
        ExclusiveTransfer::Store {
            size,
            rn,
            rt,
            status,
        } => {
            let address = transfer_address(state, rn);
            let access =
                transfer_access(size, MemoryOrdering::Relaxed, MemoryAccessClass::Exclusive);
            let reservation = monitor.borrow().reservation();
            monitor.borrow_mut().clear();
            let succeeded = if let Some(reservation) = reservation {
                memory
                    .store_exclusive(
                        address_space,
                        address,
                        access,
                        MemoryValue::U32(read_register(state, rt, false)),
                        reservation,
                    )?
                    .1
            } else {
                false
            };
            write_register(state, status, u32::from(!succeeded))
                .expect("normalized exclusive status destination is not PC");
        }
    }
    Ok(SemanticControl::Continue)
}

fn acquire_release(
    memory: &dyn CpuMemory,
    address_space: AddressSpaceId,
    state: &mut A32State,
    transfer: AcquireReleaseTransfer,
) -> Result<SemanticControl, nixe_cpu::memory::DataAccessFault> {
    // Arm A-profile LDA/STL provide acquire/release ordering without an
    // exclusive monitor; see DDI0602 AArch32 load-acquire/store-release pages:
    // https://developer.arm.com/documentation/ddi0602/latest/AArch32-Instructions/LDA
    // https://developer.arm.com/documentation/ddi0602/latest/AArch32-Instructions/STL
    match transfer {
        AcquireReleaseTransfer::Load { size, rn, rt } => {
            let address = transfer_address(state, rn);
            let access = transfer_access(size, MemoryOrdering::Acquire, MemoryAccessClass::Normal);
            let value = memory.read(address_space, address, access)?.value;
            let MemoryValue::U32(value) = value else {
                unreachable!("word acquire load returns a 32-bit value")
            };
            write_register(state, rt, value).expect("normalized acquire destination is not PC");
        }
        AcquireReleaseTransfer::Store { size, rn, rt } => {
            let address = transfer_address(state, rn);
            let access = transfer_access(size, MemoryOrdering::Release, MemoryAccessClass::Normal);
            memory.write(
                address_space,
                address,
                access,
                MemoryValue::U32(read_register(state, rt, false)),
            )?;
        }
    }
    Ok(SemanticControl::Continue)
}

fn transfer_address(state: &A32State, rn: u8) -> GuestVirtualAddress {
    GuestVirtualAddress::new(u64::from(read_register(state, rn, false)))
}

fn transfer_access(
    size: MemorySize,
    ordering: MemoryOrdering,
    class: MemoryAccessClass,
) -> MemoryAccess {
    let size = match size {
        MemorySize::Byte => MemoryAccessSize::Byte,
        MemorySize::Halfword => MemoryAccessSize::Halfword,
        MemorySize::Word => MemoryAccessSize::Word,
    };
    MemoryAccess::new(size, MemoryAlignment::Natural, ordering, class)
}
