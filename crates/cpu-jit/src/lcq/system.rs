//! Typed system exits and their canonical FP/runtime completion.

use crate::abi::{FpSystemOperation, RuntimeSystemOperation};
use crate::jit_error::Error;
use nixe_cpu::decode::a64::system::Instruction;
use nixe_cpu::platform::TargetPlatform;
use nixe_cpu::semantics::a64::{
    HintOperation, RuntimeRegisterRead, hint_operation, runtime_register_read,
};
use nixe_cpu::state::a64::A64State;
use nixe_cpu::{
    exclusive::ExclusiveMonitorState,
    execution::{ArchitecturalTimer, SchedulerRequest, VcpuEventState},
    memory::{CpuMemory, DataAccessFault},
};
use nixe_memory::{AddressSpaceId, GuestVirtualAddress};

/// Switch 1 CIVAC can prove the canonical coherent-RAM case through the
/// existing readable direct alias. Other maintenance operations stay cold.
pub(crate) fn is_cache_probe(platform: TargetPlatform, instruction: Instruction) -> bool {
    platform == TargetPlatform::Switch1
        && matches!(instruction, Instruction::System(f) if f.system_key == 0xd50b_7e20)
}

pub(crate) fn runtime_boundary(
    platform: TargetPlatform,
    instruction: Instruction,
) -> Option<RuntimeSystemOperation> {
    use nixe_cpu::semantics::a64::{barrier_operation, cache_maintenance_operation};
    let f = instruction.operands();
    match instruction {
        Instruction::ReadRegister(_) => match runtime_register_read(platform, f.system_key)? {
            RuntimeRegisterRead::TimerCounter => {
                Some(RuntimeSystemOperation::TimerCounter { rt: f.rt })
            }
            RuntimeRegisterRead::TimerFrequency => {
                Some(RuntimeSystemOperation::TimerFrequency { rt: f.rt })
            }
            RuntimeRegisterRead::Constant(_) => None,
        },
        Instruction::Hint(_) => match hint_operation(platform, f.hint)? {
            HintOperation::NoOperation => None,
            operation => Some(RuntimeSystemOperation::Hint(operation)),
        },
        Instruction::Barrier(_) => barrier_operation(f.barrier_opcode, f.barrier_option)
            .map(RuntimeSystemOperation::Barrier),
        Instruction::ClearExclusive(_) => Some(RuntimeSystemOperation::ClearExclusive),
        Instruction::System(_) => {
            let operation = cache_maintenance_operation(f.system_key)?;
            Some(RuntimeSystemOperation::Cache {
                kind: operation.kind,
                address_register: operation.uses_address.then_some(f.rt),
            })
        }
        Instruction::WriteRegister(_) => None,
    }
}

/// Borrowed vCPU/runtime services, not another owner or native context layout.
pub(crate) struct RuntimeServices<'a> {
    pub address_space: AddressSpaceId,
    pub memory: &'a dyn CpuMemory,
    pub timer: &'a dyn ArchitecturalTimer,
    pub events: &'a VcpuEventState,
    pub exclusive: &'a mut ExclusiveMonitorState,
}

#[derive(Debug)]
pub(crate) enum CompletionError {
    Invalid(Error),
    Memory(DataAccessFault),
}

fn checked_register(rt: u8) -> Result<Option<usize>, CompletionError> {
    match rt {
        0..=30 => Ok(Some(usize::from(rt))),
        31 => Ok(None),
        _ => Err(CompletionError::Invalid(Error::internal(
            "invalid system exit register",
        ))),
    }
}

/// Consume a PRE-instruction system exit after gateway FP completion, epoch
/// release and release of the execution memory lease. In particular, cache
/// maintenance may rendezvous with execution and must not wait on this vCPU.
/// There is no native continuation to unwind across: ordinary Rust errors keep
/// the source PC and are attributed using the owning GuestExit. Advance PC only
/// after the operation succeeds; scheduling is a completed instruction too.
///
/// https://developer.arm.com/documentation/ddi0601/2025-12/AArch64-Instructions/WFE--Wait-For-Event-
/// https://developer.arm.com/documentation/ddi0601/2025-12/AArch64-Instructions/CLREX--Clear-Exclusive-
/// https://developer.arm.com/documentation/ddi0601/2025-12/AArch64-Instructions/DMB--Data-Memory-Barrier-
/// https://developer.arm.com/documentation/ddi0601/2025-12/AArch64-Instructions/IC-IVAU--Instruction-Cache-line-Invalidate-by-VA-to-PoU
pub(crate) fn complete_runtime(
    operation: RuntimeSystemOperation,
    state: &mut A64State,
    services: &mut RuntimeServices<'_>,
) -> Result<Option<SchedulerRequest>, CompletionError> {
    let mut scheduled = None;
    match operation {
        RuntimeSystemOperation::TimerCounter { rt }
        | RuntimeSystemOperation::TimerFrequency { rt } => {
            let destination = checked_register(rt)?;
            let snapshot = services.timer.snapshot();
            if let Some(index) = destination {
                state.general_register_storage_mut()[index] =
                    if matches!(operation, RuntimeSystemOperation::TimerCounter { .. }) {
                        snapshot.counter
                    } else {
                        snapshot.frequency
                    };
            }
        }
        RuntimeSystemOperation::Hint(hint) => match hint {
            HintOperation::NoOperation => {
                return Err(CompletionError::Invalid(Error::internal(
                    "no-op hint must be lowered inline",
                )));
            }
            HintOperation::Yield => scheduled = Some(SchedulerRequest::Yield),
            HintOperation::WaitForEvent => {
                if !services.events.consume_event() {
                    scheduled = Some(SchedulerRequest::WaitForEvent);
                }
            }
            HintOperation::WaitForInterrupt => {
                if !services.events.interrupts_pending() {
                    scheduled = Some(SchedulerRequest::WaitForInterrupt);
                }
            }
            HintOperation::SendEvent => scheduled = Some(SchedulerRequest::SendEvent),
            HintOperation::SendEventLocal => services.events.signal_event(),
        },
        RuntimeSystemOperation::Barrier(operation) => services.memory.memory_barrier(operation),
        RuntimeSystemOperation::ClearExclusive => services.exclusive.clear(),
        RuntimeSystemOperation::Cache {
            kind,
            address_register,
        } => {
            let address = match address_register {
                Some(rt) => Some(GuestVirtualAddress::new(match checked_register(rt)? {
                    Some(index) => state.general_register_storage_mut()[index],
                    None => 0,
                })),
                None => None,
            };
            services
                .memory
                .maintain_cache(services.address_space, kind, address)
                .map_err(CompletionError::Memory)?;
        }
    }
    state.set_pc(state.pc().wrapping_add(4));
    Ok(scheduled)
}

pub(crate) fn fp_boundary(instruction: Instruction) -> Option<FpSystemOperation> {
    let f = instruction.operands();
    match instruction {
        Instruction::ReadRegister(_) if f.system_key == 0xd53b_4420 => {
            Some(FpSystemOperation::ReadStatus { rt: f.rt })
        }
        Instruction::WriteRegister(_) if f.system_key == 0xd51b_4400 => {
            Some(FpSystemOperation::WriteControl { rt: f.rt })
        }
        Instruction::WriteRegister(_) if f.system_key == 0xd51b_4420 => {
            Some(FpSystemOperation::WriteStatus { rt: f.rt })
        }
        _ => None,
    }
}

pub(crate) fn is_inline(platform: TargetPlatform, instruction: Instruction) -> bool {
    let f = instruction.operands();
    match instruction {
        Instruction::ReadRegister(_) => {
            matches!(
                f.system_key,
                0xd53b_4200 | 0xd53b_4400 | 0xd53b_d040 | 0xd53b_d060
            ) || matches!(
                runtime_register_read(platform, f.system_key),
                Some(RuntimeRegisterRead::Constant(_))
            )
        }
        Instruction::WriteRegister(_) => matches!(f.system_key, 0xd51b_4200 | 0xd51b_d040),
        Instruction::Hint(_) => {
            matches!(
                hint_operation(platform, f.hint),
                Some(HintOperation::NoOperation)
            ) || matches!(f.hint, 32 | 34 | 36 | 38)
        }
        _ => false,
    }
}

/// Complete only after the canonical gateway has merged host FPSR and restored
/// the caller environment, and the Invocation/NativeFrame borrows have ended.
/// The exit PC still identifies the unexecuted system instruction. Reads see
/// completed status; replacements cannot be contaminated by the old segment.
///
/// Arm DDI 0601: MRS/MSR FPCR and FPSR.
/// https://developer.arm.com/documentation/ddi0601/2025-12/AArch64-Registers/FPCR--Floating-point-Control-Register
/// https://developer.arm.com/documentation/ddi0601/2025-12/AArch64-Registers/FPSR--Floating-point-Status-Register
pub(crate) fn complete_fp(operation: FpSystemOperation, state: &mut A64State) -> Result<(), Error> {
    let (FpSystemOperation::ReadStatus { rt }
    | FpSystemOperation::WriteControl { rt }
    | FpSystemOperation::WriteStatus { rt }) = operation;
    if rt > 31 {
        return Err(Error::internal("invalid FP system exit register"));
    }
    let input = if rt == 31 {
        0
    } else {
        state.general_register_storage_mut()[usize::from(rt)] as u32
    };
    match operation {
        FpSystemOperation::ReadStatus { .. } if rt != 31 => {
            let status = u64::from(state.fpsr());
            state.general_register_storage_mut()[usize::from(rt)] = status;
        }
        FpSystemOperation::ReadStatus { .. } => {}
        FpSystemOperation::WriteControl { .. } => state.set_fpcr(input),
        FpSystemOperation::WriteStatus { .. } => state.set_fpsr(input),
    }
    state.set_pc(state.pc().wrapping_add(4));
    Ok(())
}
