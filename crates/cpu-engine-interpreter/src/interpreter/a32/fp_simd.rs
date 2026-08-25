use super::{InterpreterContext, InterpreterError};
use crate::interpreter::aarch32::SemanticControl;
use nixe_cpu::{
    decode::{
        DecodedOpcode,
        a32::fp_simd::Instruction,
        aarch32::{VectorOperation, VectorSize},
    },
    location::DecodedInstruction,
    memory::{
        MemoryAccess, MemoryAccessClass, MemoryAccessSize, MemoryAlignment, MemoryOrdering,
        MemoryValue,
    },
    state::a32::A32State,
};
use nixe_memory::GuestVirtualAddress;

pub(super) enum Execution {
    Control(SemanticControl),
    Fault(nixe_cpu::memory::DataAccessFault),
    FloatingPointException,
}

pub(super) fn execute(
    context: InterpreterContext<'_>,
    state: &mut A32State,
    decoded: &DecodedInstruction<DecodedOpcode>,
    instruction: Instruction,
) -> Result<Execution, InterpreterError> {
    let Instruction::Data(data) = instruction else {
        let Instruction::Memory(transfer) = instruction else {
            unreachable!()
        };
        let Some(memory) = context.memory() else {
            return Err(super::super::unsupported(decoded));
        };
        let base = super::super::aarch32::read_register(state, transfer.rn, false);
        let address = GuestVirtualAddress::new(u64::from(base));
        let access = MemoryAccess::new(
            MemoryAccessSize::Doubleword,
            MemoryAlignment::Unaligned,
            MemoryOrdering::Relaxed,
            MemoryAccessClass::Normal,
        );
        let address_space = context.process().address_space_id();
        if transfer.load {
            match memory.read(address_space, address, access) {
                Ok(read) => {
                    let MemoryValue::U64(value) = read.value else {
                        unreachable!()
                    };
                    state.write_d(transfer.vd, value);
                }
                Err(fault) => return Ok(Execution::Fault(fault)),
            }
        } else {
            let value = state.read_d(transfer.vd).expect("normalized D register");
            if let Err(fault) =
                memory.write(address_space, address, access, MemoryValue::U64(value))
            {
                return Ok(Execution::Fault(fault));
            }
        }
        if let Some(rm) = transfer.writeback_rm {
            let increment = if rm == 13 {
                8
            } else {
                super::super::aarch32::read_register(state, rm, false)
            };
            super::super::aarch32::write_register(state, transfer.rn, base.wrapping_add(increment))
                .expect("NEON transfer base is an ordinary register");
        }
        return Ok(Execution::Control(SemanticControl::Continue));
    };
    let (lhs, rhs) = match data.size {
        VectorSize::D => (
            u128::from(state.read_d(data.vn).unwrap()),
            u128::from(state.read_d(data.vm).unwrap()),
        ),
        VectorSize::Q => (
            state.read_q(data.vn).unwrap(),
            state.read_q(data.vm).unwrap(),
        ),
    };
    let width = if data.size == VectorSize::D { 64 } else { 128 };
    let mask = if width == 64 {
        u128::from(u64::MAX)
    } else {
        u128::MAX
    };
    let result = match data.operation {
        VectorOperation::Move => rhs,
        VectorOperation::And => lhs & rhs,
        VectorOperation::BitClear => lhs & !rhs,
        VectorOperation::Or => lhs | rhs,
        VectorOperation::ExclusiveOr => lhs ^ rhs,
        VectorOperation::AddInteger { lane_bits } => lanes(lhs, rhs, width, lane_bits, false),
        VectorOperation::SubtractInteger { lane_bits } => lanes(lhs, rhs, width, lane_bits, true),
        VectorOperation::AddF32 | VectorOperation::SubtractF32 | VectorOperation::MultiplyF32 => {
            let operation = match data.operation {
                VectorOperation::AddF32 => nixe_cpu::semantics::a64_fp_simd::Binary32Operation::Add,
                VectorOperation::SubtractF32 => {
                    nixe_cpu::semantics::a64_fp_simd::Binary32Operation::Subtract
                }
                VectorOperation::MultiplyF32 => {
                    nixe_cpu::semantics::a64_fp_simd::Binary32Operation::Multiply
                }
                _ => unreachable!(),
            };
            let mut result = 0_u128;
            let mut status = nixe_cpu::semantics::floating_point::FpStatus::default();
            for shift in (0..width).step_by(32) {
                let (lane, lane_status) = nixe_cpu::semantics::a64_fp_simd::binary32(
                    operation,
                    (lhs >> shift) as u32,
                    (rhs >> shift) as u32,
                    state.fpscr(),
                );
                result |= u128::from(lane) << shift;
                merge_status(&mut status, lane_status);
            }
            if nixe_cpu::semantics::a64_fp_simd::fp_status_traps(status, state.fpscr()) {
                return Ok(Execution::FloatingPointException);
            }
            state.set_fpscr(
                state.fpscr() | nixe_cpu::semantics::a64_fp_simd::fp_status_bits(status),
            );
            result
        }
    } & mask;
    match data.size {
        VectorSize::D => {
            state.write_d(data.vd, result as u64);
        }
        VectorSize::Q => {
            state.write_q(data.vd, result);
        }
    }
    Ok(Execution::Control(SemanticControl::Continue))
}

fn merge_status(
    destination: &mut nixe_cpu::semantics::floating_point::FpStatus,
    source: nixe_cpu::semantics::floating_point::FpStatus,
) {
    destination.invalid_operation |= source.invalid_operation;
    destination.divide_by_zero |= source.divide_by_zero;
    destination.overflow |= source.overflow;
    destination.underflow |= source.underflow;
    destination.inexact |= source.inexact;
    destination.input_denormal |= source.input_denormal;
}

fn lanes(lhs: u128, rhs: u128, width: u8, lane_bits: u8, subtract: bool) -> u128 {
    let lane_mask = (1_u128 << lane_bits) - 1;
    let mut result = 0;
    for shift in (0..width).step_by(usize::from(lane_bits)) {
        let a = (lhs >> shift) & lane_mask;
        let b = (rhs >> shift) & lane_mask;
        let lane = if subtract {
            a.wrapping_sub(b)
        } else {
            a.wrapping_add(b)
        } & lane_mask;
        result |= lane << shift;
    }
    result
}
