use crate::decode::InstructionPattern;
use crate::decode::aarch32::{
    DataOperation, DataProcessing, Multiply, Shift, ShiftAmount, ShifterOperand,
    decode_immediate_shift,
};
use crate::semantics::shifts::A32ShiftKind;

use super::{NO_FEATURES, pattern};

pub static PATTERNS: &[InstructionPattern] = &[
    pattern(
        "multiply",
        0x0fc0_00f0,
        0x0000_0090,
        0x0001_0011,
        20,
        &[],
        NO_FEATURES,
    ),
    pattern(
        "movw",
        0x0ff0_0000,
        0x0300_0000,
        0x0001_0012,
        15,
        &[],
        NO_FEATURES,
    ),
    pattern(
        "movt",
        0x0ff0_0000,
        0x0340_0000,
        0x0001_0013,
        15,
        &[],
        NO_FEATURES,
    ),
    pattern(
        "data-processing",
        0x0c00_0000,
        0x0000_0000,
        0x0001_0010,
        0,
        &[],
        NO_FEATURES,
    ),
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Instruction {
    DataProcessing(DataProcessing),
    Multiply(Multiply),
    MoveWide { rd: u8, immediate: u16, top: bool },
}

pub(super) fn normalize(id: u32, bits: u32) -> Instruction {
    match id {
        0x0001_0010 => Instruction::DataProcessing(normalize_data_processing(bits)),
        0x0001_0011 => Instruction::Multiply(Multiply {
            rd: ((bits >> 16) & 0xf) as u8,
            rn: ((bits >> 12) & 0xf) as u8,
            rs: ((bits >> 8) & 0xf) as u8,
            rm: (bits & 0xf) as u8,
            accumulate: bits & (1 << 21) != 0,
            set_flags: bits & (1 << 20) != 0,
        }),
        0x0001_0012 | 0x0001_0013 => Instruction::MoveWide {
            rd: ((bits >> 12) & 0xf) as u8,
            immediate: ((((bits >> 16) & 0xf) << 12) | (bits & 0xfff)) as u16,
            top: id == 0x0001_0013,
        },
        _ => unreachable!(),
    }
}

fn normalize_data_processing(bits: u32) -> DataProcessing {
    let immediate = bits & (1 << 25) != 0;
    let operand2 = if immediate {
        let rotation = ((bits >> 8) & 0xf) as u8 * 2;
        ShifterOperand::Immediate {
            value: (bits as u8 as u32).rotate_right(u32::from(rotation)),
            rotation,
        }
    } else {
        let rm = (bits & 0xf) as u8;
        let kind_bits = ((bits >> 5) & 3) as u8;
        let shift = if bits & (1 << 4) == 0 {
            decode_immediate_shift(kind_bits, ((bits >> 7) & 0x1f) as u8)
        } else {
            let kind = match kind_bits {
                0 => A32ShiftKind::LogicalLeft,
                1 => A32ShiftKind::LogicalRight,
                2 => A32ShiftKind::ArithmeticRight,
                3 => A32ShiftKind::RotateRight,
                _ => unreachable!(),
            };
            Shift {
                kind,
                amount: ShiftAmount::Register(((bits >> 8) & 0xf) as u8),
            }
        };
        ShifterOperand::Register { rm, shift }
    };
    let operation = DataOperation::from_a32_opcode(((bits >> 21) & 0xf) as u8);
    DataProcessing {
        operation,
        set_flags: operation.is_test() || bits & (1 << 20) != 0,
        rn: ((bits >> 16) & 0xf) as u8,
        rd: ((bits >> 12) & 0xf) as u8,
        operand2,
    }
}
