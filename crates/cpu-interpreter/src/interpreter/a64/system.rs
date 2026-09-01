use nixe_cpu::{
    decode::{
        DecodedOpcode,
        a64::system::{Instruction, Operands},
    },
    location::DecodedInstruction,
    semantics::a64::{HintOperation, RuntimeRegisterRead},
    state::a64::{A64State, Nzcv},
};

use super::{advance, read, write};
use crate::interpreter::{InstructionStep, InterpreterContext, InterpreterError};
use nixe_cpu::execution::SchedulerRequest;

pub(super) fn execute(
    context: InterpreterContext<'_>,
    state: &mut A64State,
    decoded: &DecodedInstruction<DecodedOpcode>,
    instruction: Instruction,
) -> Result<InstructionStep, InterpreterError> {
    let fields = instruction.operands();
    if matches!(instruction, Instruction::Hint(_))
        && let Some(operation) =
            nixe_cpu::semantics::a64::hint_operation(context.process().platform(), fields.hint)
    {
        return execute_architectural_hint(context, state, decoded, operation);
    }
    let outcome = match instruction {
        Instruction::Hint(_) => execute_hint(fields),
        Instruction::ReadRegister(_) => execute_mrs(context, state, fields),
        Instruction::WriteRegister(_) => execute_msr(state, fields),
        Instruction::Barrier(_) => execute_barrier(context, fields),
        Instruction::ClearExclusive(_) => {
            context.exclusive_monitor().borrow_mut().clear();
            true
        }
        Instruction::System(_) => match execute_system(context, state, fields) {
            Ok(executed) => executed,
            Err(fault) => {
                return Ok(InstructionStep::data_fault(decoded.location, fault));
            }
        },
    };
    if !outcome {
        return Err(super::super::unsupported(decoded));
    }
    advance(state);
    Ok(InstructionStep::Continue)
}

fn execute_hint(fields: Operands) -> bool {
    match fields.hint {
        // The selected platform table has no BTI entry, so these encodings
        // retain their architectural HINT behavior.
        32 | 34 | 36 | 38 => true,
        _ => false,
    }
}

fn execute_architectural_hint(
    context: InterpreterContext<'_>,
    state: &mut A64State,
    decoded: &DecodedInstruction<DecodedOpcode>,
    operation: HintOperation,
) -> Result<InstructionStep, InterpreterError> {
    let source = decoded.location;
    let scheduled = |state: &mut A64State, request| {
        advance(state);
        Ok(InstructionStep::scheduled(source, request))
    };
    match operation {
        HintOperation::NoOperation => {}
        // https://developer.arm.com/documentation/ddi0601/2025-12/AArch64-Instructions/YIELD--Yield-
        HintOperation::Yield => return scheduled(state, SchedulerRequest::Yield),
        // https://developer.arm.com/documentation/ddi0601/2025-12/AArch64-Instructions/WFE--Wait-For-Event-
        HintOperation::WaitForEvent => {
            let events = context.vcpu_events();
            if events.consume_event() {
                advance(state);
                return Ok(InstructionStep::Continue);
            }
            return scheduled(state, SchedulerRequest::WaitForEvent);
        }
        // https://developer.arm.com/documentation/ddi0601/2025-12/AArch64-Instructions/WFI--Wait-For-Interrupt-
        HintOperation::WaitForInterrupt => {
            let events = context.vcpu_events();
            if events.interrupts_pending() {
                advance(state);
                return Ok(InstructionStep::Continue);
            }
            return scheduled(state, SchedulerRequest::WaitForInterrupt);
        }
        // https://developer.arm.com/documentation/ddi0601/2025-12/AArch64-Instructions/SEV--Send-Event-
        HintOperation::SendEvent => return scheduled(state, SchedulerRequest::SendEvent),
        // https://developer.arm.com/documentation/ddi0601/2025-12/AArch64-Instructions/SEVL--Send-Event-Local-
        HintOperation::SendEventLocal => {
            let events = context.vcpu_events();
            events.signal_event();
        }
    }
    advance(state);
    Ok(InstructionStep::Continue)
}

fn execute_mrs(context: InterpreterContext<'_>, state: &mut A64State, fields: Operands) -> bool {
    let value = match fields.system_key {
        0xd53b_4200 => u64::from(state.nzcv().bits()),
        0xd53b_4400 => u64::from(state.fpcr()),
        0xd53b_4420 => u64::from(state.fpsr()),
        0xd53b_d040 => state.tpidr_el0(),
        0xd53b_d060 => state.tpidrro_el0(),
        system_key => match nixe_cpu::semantics::a64::runtime_register_read(
            context.process().platform(),
            system_key,
        ) {
            Some(RuntimeRegisterRead::Constant(value)) => value,
            Some(RuntimeRegisterRead::TimerFrequency) => {
                let timer = context.architectural_timer();
                timer.frequency
            }
            Some(RuntimeRegisterRead::TimerCounter) => {
                let timer = context.architectural_timer();
                timer.counter
            }
            None => return false,
        },
    };
    write(state, fields.rt, 64, false, value);
    true
}

fn execute_msr(state: &mut A64State, fields: Operands) -> bool {
    let value = read(state, fields.rt, 64, false);
    match fields.system_key {
        0xd51b_4200 => state.set_nzcv(Nzcv::from_bits(value as u32)),
        0xd51b_4400 => state.set_fpcr(value as u32),
        0xd51b_4420 => state.set_fpsr(value as u32),
        0xd51b_d040 => state.set_tpidr_el0(value),
        // TPIDRRO_EL0 is runtime-owned and architecturally read-only here.
        0xd51b_d060 => return false,
        _ => return false,
    }
    true
}

fn execute_barrier(context: InterpreterContext<'_>, fields: Operands) -> bool {
    let Some(operation) =
        nixe_cpu::semantics::a64::barrier_operation(fields.barrier_opcode, fields.barrier_option)
    else {
        return false;
    };
    context.memory().memory_barrier(operation);
    true
}

fn execute_system(
    context: InterpreterContext<'_>,
    state: &A64State,
    fields: Operands,
) -> Result<bool, nixe_cpu::memory::DataAccessFault> {
    // Arm defines IC/DC operations as architectural cache-maintenance effects;
    // the CPU-memory owner synchronizes canonical bytes and derived engine code.
    // https://developer.arm.com/documentation/ddi0601/2025-12/AArch64-Instructions/IC-IVAU--Instruction-Cache-line-Invalidate-by-VA-to-PoU
    // https://developer.arm.com/documentation/ddi0601/2025-12/AArch64-Instructions/DC-CIVAC--Data-or-Unified-Cache-Line-Clean-and-Invalidate-by-VA-to-PoC
    let Some(operation) = nixe_cpu::semantics::a64::cache_maintenance_operation(fields.system_key)
    else {
        return Ok(false);
    };
    let memory = context.memory();
    let address = operation
        .uses_address
        .then(|| nixe_memory::GuestVirtualAddress::new(read(state, fields.rt, 64, false)));
    memory.maintain_cache(
        context.process().address_space_id(),
        operation.kind,
        address,
    )?;
    Ok(true)
}
