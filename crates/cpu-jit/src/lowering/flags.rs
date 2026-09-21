//! Recipe shapes produced by the shared integer emitter, not guest semantics
//! or another IR. emit_integer checks this contract against its actual result
//! in debug builds. Operands of equal shapes become ordinary SSA parameters.

use crate::abi::LazyFlags;
use nixe_cpu::decode::a64::integer::Instruction;

pub(crate) fn integer_flag_shape(instruction: Instruction) -> Option<LazyFlags<()>> {
    let f = instruction.operands();
    let width = if f.width_64 { 64 } else { 32 };
    let arithmetic = |carry| match (f.subtract, carry) {
        (false, false) => LazyFlags::Add {
            lhs: (),
            rhs: (),
            result: (),
            width,
        },
        (true, false) => LazyFlags::Subtract {
            lhs: (),
            rhs: (),
            result: (),
            width,
        },
        (false, true) => LazyFlags::AddCarry {
            lhs: (),
            rhs: (),
            carry: (),
            result: (),
            width,
        },
        (true, true) => LazyFlags::SubtractCarry {
            lhs: (),
            rhs: (),
            carry: (),
            result: (),
            width,
        },
    };
    match instruction {
        Instruction::AddSubImmediate(_)
        | Instruction::AddSubShifted(_)
        | Instruction::AddSubExtended(_)
        | Instruction::AddSubCarry(_)
            if f.set_flags =>
        {
            Some(arithmetic(matches!(
                instruction,
                Instruction::AddSubCarry(_)
            )))
        }
        Instruction::LogicalImmediate(_) | Instruction::LogicalShifted(_)
            if f.subtract && f.set_flags =>
        {
            Some(LazyFlags::Logical { result: (), width })
        }
        Instruction::ConditionalCompareRegister(_)
        | Instruction::ConditionalCompareImmediate(_) => Some(LazyFlags::Conditional {
            predicate: (),
            when_true: Box::new(arithmetic(false)),
            when_false: u32::from(f.nzcv),
        }),
        _ => None,
    }
}
