use crate::decode::{InstructionPattern, OperandField, OperandId, OperandKind};

use super::{NO_FEATURES, pattern};

const NONE: &[OperandField] = &[];
const CONDITION: OperandField = OperandField {
    id: OperandId::Condition,
    lsb: 28,
    width: 4,
    kind: OperandKind::Unsigned,
};
const CONDITION_ONLY: &[OperandField] = &[CONDITION];
const BRANCH: &[OperandField] = &[
    OperandField {
        id: OperandId::Immediate,
        lsb: 0,
        width: 24,
        kind: OperandKind::SignedScaled { scale: 2 },
    },
    CONDITION,
];

pub static PATTERNS: &[InstructionPattern] = &[
    pattern(
        "nop",
        0x0fff_ffff,
        0x0320_f000,
        0x0001_0001,
        10,
        CONDITION_ONLY,
        NO_FEATURES,
    ),
    pattern(
        "b",
        0x0f00_0000,
        0x0a00_0000,
        0x0001_0002,
        1,
        BRANCH,
        NO_FEATURES,
    ),
    pattern(
        "bl",
        0x0f00_0000,
        0x0b00_0000,
        0x0001_0003,
        1,
        BRANCH,
        NO_FEATURES,
    ),
    pattern(
        "bx",
        0x0fff_fff0,
        0x012f_ff10,
        0x0001_0004,
        20,
        CONDITION_ONLY,
        NO_FEATURES,
    ),
    pattern(
        "blx-register",
        0x0fff_fff0,
        0x012f_ff30,
        0x0001_0005,
        20,
        CONDITION_ONLY,
        NO_FEATURES,
    ),
    pattern(
        "blx-immediate",
        0xfe00_0000,
        0xfa00_0000,
        0x0001_0006,
        20,
        NONE,
        NO_FEATURES,
    ),
    pattern(
        "svc",
        0x0f00_0000,
        0x0f00_0000,
        0x0001_0007,
        2,
        CONDITION_ONLY,
        NO_FEATURES,
    ),
    pattern(
        "bkpt",
        0x0ff0_00f0,
        0x0120_0070,
        0x0001_0008,
        30,
        CONDITION_ONLY,
        NO_FEATURES,
    ),
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Instruction {
    Nop,
    Branch { link: bool, displacement: i32 },
    Exchange { link: bool, rm: u8 },
    BlxImmediate { displacement: i32 },
    Svc { immediate: u32 },
    Breakpoint { immediate: u16 },
}

pub(super) fn normalize(id: u32, bits: u32) -> Instruction {
    match id {
        0x0001_0001 => Instruction::Nop,
        0x0001_0002 | 0x0001_0003 => Instruction::Branch {
            link: id == 0x0001_0003,
            displacement: (((bits & 0x00ff_ffff) << 8) as i32) >> 6,
        },
        0x0001_0004 | 0x0001_0005 => Instruction::Exchange {
            link: id == 0x0001_0005,
            rm: bits as u8 & 0xf,
        },
        0x0001_0006 => {
            let immediate = ((bits & 0x00ff_ffff) << 2) | ((bits >> 23) & 2);
            Instruction::BlxImmediate {
                displacement: ((immediate << 6) as i32) >> 6,
            }
        }
        0x0001_0007 => Instruction::Svc {
            immediate: bits & 0x00ff_ffff,
        },
        0x0001_0008 => Instruction::Breakpoint {
            immediate: (((bits >> 4) & 0xfff0) | (bits & 0xf)) as u16,
        },
        _ => unreachable!(),
    }
}
