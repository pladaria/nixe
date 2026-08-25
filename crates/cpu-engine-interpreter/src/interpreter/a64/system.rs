use nixe_cpu::{
    decode::{
        DecodedOpcode,
        a64::system::{Instruction, Operands},
    },
    location::{DecodedInstruction, LocationDescriptor},
    profile::{CapabilityStatus, InstructionFeature},
    semantics::a64::{HintOperation, RuntimeRegisterRead},
    state::a64::{A64State, Nzcv},
};

use super::{advance, read, resume, write};
use crate::interpreter::{InterpreterContext, InterpreterError, InterpreterOutcome};
use nixe_cpu_engine::SchedulerRequest;

pub(super) fn execute(
    context: InterpreterContext<'_>,
    state: &mut A64State,
    decoded: &DecodedInstruction<DecodedOpcode>,
    instruction: Instruction,
) -> Result<InterpreterOutcome, InterpreterError> {
    let fields = instruction.operands();
    if matches!(instruction, Instruction::Hint(_))
        && let Some(operation) = nixe_cpu::semantics::a64::hint_operation(fields.hint)
    {
        return execute_architectural_hint(context, state, decoded, operation);
    }
    let outcome = match instruction {
        Instruction::Hint(_) => execute_hint(context, state, decoded.location, fields),
        Instruction::ReadRegister(_) => execute_mrs(context, state, fields),
        Instruction::WriteRegister(_) => execute_msr(state, fields),
        Instruction::Barrier(_) => execute_barrier(context, fields),
        Instruction::System(_) => match execute_system(context, state, fields) {
            Ok(executed) => executed,
            Err(fault) => {
                return Ok(InterpreterOutcome::DataAbort {
                    source: decoded.location,
                    fault,
                });
            }
        },
    };
    if !outcome {
        return Err(super::super::unsupported(decoded));
    }
    advance(state);
    Ok(resume(state, decoded))
}

fn execute_hint(
    context: InterpreterContext<'_>,
    _state: &mut A64State,
    _source: LocationDescriptor,
    fields: Operands,
) -> bool {
    match fields.hint {
        // BTI is encoded in the HINT space. On a profile where FEAT_BTI is
        // absent these encodings retain their architectural hint behavior;
        // enabled or unknown profiles require the future branch-type state.
        32 | 34 | 36 | 38 => matches!(
            context
                .process()
                .profile()
                .instruction_feature_status(InstructionFeature::BranchTargetIdentification),
            CapabilityStatus::Disabled
        ),
        _ => false,
    }
}

fn execute_architectural_hint(
    context: InterpreterContext<'_>,
    state: &mut A64State,
    decoded: &DecodedInstruction<DecodedOpcode>,
    operation: HintOperation,
) -> Result<InterpreterOutcome, InterpreterError> {
    let source = decoded.location;
    let scheduled = |state: &mut A64State, request| {
        advance(state);
        Ok(InterpreterOutcome::Scheduled { source, request })
    };
    match operation {
        HintOperation::NoOperation => {}
        // https://developer.arm.com/documentation/ddi0601/2025-12/AArch64-Instructions/YIELD--Yield-
        HintOperation::Yield => return scheduled(state, SchedulerRequest::Yield),
        // https://developer.arm.com/documentation/ddi0601/2025-12/AArch64-Instructions/WFE--Wait-For-Event-
        HintOperation::WaitForEvent => {
            let Some(events) = context.vcpu_events() else {
                return Err(super::super::unsupported(decoded));
            };
            if events.consume_event() {
                advance(state);
                return Ok(resume(state, decoded));
            }
            return scheduled(state, SchedulerRequest::WaitForEvent);
        }
        // https://developer.arm.com/documentation/ddi0601/2025-12/AArch64-Instructions/WFI--Wait-For-Interrupt-
        HintOperation::WaitForInterrupt => {
            let Some(events) = context.vcpu_events() else {
                return Err(super::super::unsupported(decoded));
            };
            if events.interrupts_pending() {
                advance(state);
                return Ok(resume(state, decoded));
            }
            return scheduled(state, SchedulerRequest::WaitForInterrupt);
        }
        // https://developer.arm.com/documentation/ddi0601/2025-12/AArch64-Instructions/SEV--Send-Event-
        HintOperation::SendEvent => return scheduled(state, SchedulerRequest::SendEvent),
        // https://developer.arm.com/documentation/ddi0601/2025-12/AArch64-Instructions/SEVL--Send-Event-Local-
        HintOperation::SendEventLocal => {
            let Some(events) = context.vcpu_events() else {
                return Err(super::super::unsupported(decoded));
            };
            events.signal_event();
        }
    }
    advance(state);
    Ok(resume(state, decoded))
}

fn execute_mrs(context: InterpreterContext<'_>, state: &mut A64State, fields: Operands) -> bool {
    let value = match fields.system_key {
        0xd53b_4200 => u64::from(state.nzcv().bits()),
        0xd53b_4400 => u64::from(state.fpcr()),
        0xd53b_4420 => u64::from(state.fpsr()),
        0xd53b_d040 => state.tpidr_el0(),
        0xd53b_d060 => state.tpidrro_el0(),
        system_key => match nixe_cpu::semantics::a64::runtime_register_read(
            context.process().profile(),
            system_key,
        ) {
            Some(RuntimeRegisterRead::Constant(value)) => value,
            Some(RuntimeRegisterRead::TimerFrequency) => {
                let Some(timer) = context.architectural_timer() else {
                    return false;
                };
                timer.frequency
            }
            Some(RuntimeRegisterRead::TimerCounter) => {
                let Some(timer) = context.architectural_timer() else {
                    return false;
                };
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
    if let Some(memory) = context.memory() {
        memory.memory_barrier(operation);
    } else {
        nixe_cpu::memory::apply_host_memory_barrier(operation);
    }
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
    let (kind, uses_address) = match fields.system_key {
        0xd508_7500 => (
            nixe_cpu::memory::CacheMaintenanceKind::InstructionInvalidate,
            false,
        ),
        0xd50b_7520 => (
            nixe_cpu::memory::CacheMaintenanceKind::InstructionInvalidate,
            true,
        ),
        0xd508_7620 => (nixe_cpu::memory::CacheMaintenanceKind::DataInvalidate, true),
        0xd50b_7b20 => (nixe_cpu::memory::CacheMaintenanceKind::DataClean, true),
        0xd50b_7e20 => (
            nixe_cpu::memory::CacheMaintenanceKind::DataCleanAndInvalidate,
            true,
        ),
        _ => return Ok(false),
    };
    let Some(memory) = context.memory() else {
        return Ok(false);
    };
    let address = uses_address
        .then(|| nixe_memory::GuestVirtualAddress::new(read(state, fields.rt, 64, false)));
    memory.maintain_cache(context.process().address_space_id(), kind, address)?;
    Ok(true)
}
