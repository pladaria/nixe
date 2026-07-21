use super::{pattern16, pattern32};
use crate::{
    decode::aarch32::{
        DataOperation, DataProcessing, Multiply, Shift, ShiftAmount, ShifterOperand,
    },
    decode::{DecodeSupport, InstructionPattern, OperandField, OperandId, OperandKind},
    semantics::shifts::A32ShiftKind,
};

pub static PATTERNS_16: &[InstructionPattern] = &[
    pattern16(
        "movs",
        0xf800,
        0x2000,
        0x0002_0003,
        20,
        &[
            OperandField {
                id: OperandId::Destination,
                lsb: 8,
                width: 3,
                kind: OperandKind::Register(crate::decode::RegisterClass::A32General),
            },
            OperandField {
                id: OperandId::Immediate,
                lsb: 0,
                width: 8,
                kind: OperandKind::Unsigned,
            },
        ],
        DecodeSupport::Ready,
    ),
    pattern16(
        "shift-immediate",
        0xe000,
        0x0000,
        0x0002_0010,
        1,
        &[],
        DecodeSupport::Ready,
    ),
    pattern16(
        "add-subtract-three-register",
        0xf800,
        0x1800,
        0x0002_0011,
        10,
        &[],
        DecodeSupport::Ready,
    ),
    pattern16(
        "immediate-data-processing",
        0xe000,
        0x2000,
        0x0002_0012,
        1,
        &[],
        DecodeSupport::Ready,
    ),
    pattern16(
        "data-processing-register",
        0xfc00,
        0x4000,
        0x0002_0013,
        1,
        &[],
        DecodeSupport::Ready,
    ),
    pattern16(
        "special-data-processing",
        0xfc00,
        0x4400,
        0x0002_0014,
        1,
        &[],
        DecodeSupport::Ready,
    ),
    pattern16(
        "add-pc-sp",
        0xf000,
        0xa000,
        0x0002_0015,
        1,
        &[],
        DecodeSupport::Ready,
    ),
    pattern16(
        "adjust-sp",
        0xff00,
        0xb000,
        0x0002_0016,
        10,
        &[],
        DecodeSupport::Ready,
    ),
];
pub static PATTERNS_32: &[InstructionPattern] = &[
    pattern32(
        "movw",
        0xfbf0_8000,
        0xf240_0000,
        0x0002_0017,
        10,
        &[],
        DecodeSupport::Ready,
    ),
    pattern32(
        "movt",
        0xfbf0_8000,
        0xf2c0_0000,
        0x0002_0018,
        10,
        &[],
        DecodeSupport::Ready,
    ),
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Instruction {
    DataProcessing(DataProcessing),
    Multiply(Multiply),
    MoveWide { rd: u8, immediate: u16, top: bool },
}

pub(super) fn normalize(id: u32, bits: u32) -> Instruction {
    if id == 0x0002_0003 {
        return Instruction::DataProcessing(DataProcessing {
            operation: DataOperation::Move,
            set_flags: true,
            rn: 0,
            rd: ((bits >> 8) & 7) as u8,
            operand2: ShifterOperand::Immediate {
                value: bits & 0xff,
                rotation: 0,
            },
        });
    }
    if id == 0x0002_0013 && (bits >> 6) & 0xf == 13 {
        let rd = (bits & 7) as u8;
        return Instruction::Multiply(Multiply {
            rd,
            rn: 0,
            rs: ((bits >> 3) & 7) as u8,
            rm: rd,
            accumulate: false,
            set_flags: true,
        });
    }
    if matches!(id, 0x0002_0017 | 0x0002_0018) {
        let immediate = (((bits >> 16) & 0xf) << 12)
            | (((bits >> 26) & 1) << 11)
            | (((bits >> 12) & 7) << 8)
            | (bits & 0xff);
        return Instruction::MoveWide {
            rd: ((bits >> 8) & 0xf) as u8,
            immediate: immediate as u16,
            top: id == 0x0002_0018,
        };
    }
    Instruction::DataProcessing(match id {
        0x0002_0010 => {
            let op = ((bits >> 11) & 3) as u8;
            let kind = match op {
                0 => A32ShiftKind::LogicalLeft,
                1 => A32ShiftKind::LogicalRight,
                _ => A32ShiftKind::ArithmeticRight,
            };
            let encoded_amount = ((bits >> 6) & 0x1f) as u8;
            let amount = if op != 0 && encoded_amount == 0 {
                32
            } else {
                encoded_amount
            };
            DataProcessing {
                operation: DataOperation::Move,
                set_flags: true,
                rn: 0,
                rd: (bits & 7) as u8,
                operand2: ShifterOperand::Register {
                    rm: ((bits >> 3) & 7) as u8,
                    shift: Shift {
                        kind,
                        amount: ShiftAmount::Immediate(amount),
                    },
                },
            }
        }
        0x0002_0011 => DataProcessing {
            operation: if bits & (1 << 9) == 0 {
                DataOperation::Add
            } else {
                DataOperation::Subtract
            },
            set_flags: true,
            rn: ((bits >> 3) & 7) as u8,
            rd: (bits & 7) as u8,
            operand2: if bits & (1 << 10) == 0 {
                reg(((bits >> 6) & 7) as u8)
            } else {
                ShifterOperand::Immediate {
                    value: (bits >> 6) & 7,
                    rotation: 0,
                }
            },
        },
        0x0002_0012 => {
            let op = ((bits >> 11) & 3) as u8;
            DataProcessing {
                operation: [
                    DataOperation::Move,
                    DataOperation::Compare,
                    DataOperation::Add,
                    DataOperation::Subtract,
                ][usize::from(op)],
                set_flags: true,
                rn: ((bits >> 8) & 7) as u8,
                rd: ((bits >> 8) & 7) as u8,
                operand2: ShifterOperand::Immediate {
                    value: bits & 0xff,
                    rotation: 0,
                },
            }
        }
        0x0002_0013 => {
            let op = ((bits >> 6) & 0xf) as u8;
            let rd = (bits & 7) as u8;
            let rm = ((bits >> 3) & 7) as u8;
            if matches!(op, 2 | 3 | 4 | 7) {
                let kind = match op {
                    2 => A32ShiftKind::LogicalLeft,
                    3 => A32ShiftKind::LogicalRight,
                    4 => A32ShiftKind::ArithmeticRight,
                    _ => A32ShiftKind::RotateRight,
                };
                DataProcessing {
                    operation: DataOperation::Move,
                    set_flags: true,
                    rn: 0,
                    rd,
                    operand2: ShifterOperand::Register {
                        rm: rd,
                        shift: Shift {
                            kind,
                            amount: ShiftAmount::Register(rm),
                        },
                    },
                }
            } else if op == 9 {
                DataProcessing {
                    operation: DataOperation::ReverseSubtract,
                    set_flags: true,
                    rn: rm,
                    rd,
                    operand2: ShifterOperand::Immediate {
                        value: 0,
                        rotation: 0,
                    },
                }
            } else {
                let operation = match op {
                    0 => DataOperation::And,
                    1 => DataOperation::ExclusiveOr,
                    5 => DataOperation::AddCarry,
                    6 => DataOperation::SubtractCarry,
                    8 => DataOperation::Test,
                    10 => DataOperation::Compare,
                    11 => DataOperation::CompareNegative,
                    12 => DataOperation::Or,
                    14 => DataOperation::BitClear,
                    15 => DataOperation::MoveNot,
                    _ => unreachable!(),
                };
                DataProcessing {
                    operation,
                    set_flags: true,
                    rn: rd,
                    rd,
                    operand2: reg(rm),
                }
            }
        }
        0x0002_0014 => {
            let op = ((bits >> 8) & 3) as u8;
            let rd = (((bits >> 4) & 8) | (bits & 7)) as u8;
            DataProcessing {
                operation: [
                    DataOperation::Add,
                    DataOperation::Compare,
                    DataOperation::Move,
                    DataOperation::Move,
                ][usize::from(op)],
                set_flags: op == 1,
                rn: rd,
                rd,
                operand2: reg(((bits >> 3) & 0xf) as u8),
            }
        }
        0x0002_0015 => DataProcessing {
            operation: DataOperation::Add,
            set_flags: false,
            rn: if bits & (1 << 11) == 0 { 15 } else { 13 },
            rd: ((bits >> 8) & 7) as u8,
            operand2: ShifterOperand::Immediate {
                value: (bits & 0xff) << 2,
                rotation: 0,
            },
        },
        0x0002_0016 => DataProcessing {
            operation: if bits & 0x80 == 0 {
                DataOperation::Add
            } else {
                DataOperation::Subtract
            },
            set_flags: false,
            rn: 13,
            rd: 13,
            operand2: ShifterOperand::Immediate {
                value: (bits & 0x7f) << 2,
                rotation: 0,
            },
        },
        _ => unreachable!(),
    })
}

const fn reg(rm: u8) -> ShifterOperand {
    ShifterOperand::Register {
        rm,
        shift: Shift {
            kind: A32ShiftKind::LogicalLeft,
            amount: ShiftAmount::Immediate(0),
        },
    }
}
