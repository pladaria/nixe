use super::{pattern16, pattern32};
use crate::decode::{DecodeSupport, InstructionPattern, OperandField, OperandId, OperandKind};

const NONE: &[OperandField] = &[];
const IT_IMM: &[OperandField] = &[OperandField {
    id: OperandId::Immediate,
    lsb: 0,
    width: 8,
    kind: OperandKind::Unsigned,
}];
const B_IMM: &[OperandField] = &[OperandField {
    id: OperandId::Immediate,
    lsb: 0,
    width: 11,
    kind: OperandKind::SignedScaled { scale: 1 },
}];

pub static PATTERNS_16: &[InstructionPattern] = &[
    pattern16(
        "nop",
        0xffff,
        0xbf00,
        0x0002_0001,
        30,
        NONE,
        DecodeSupport::Ready,
    ),
    pattern16(
        "it",
        0xff00,
        0xbf00,
        0x0002_0005,
        20,
        IT_IMM,
        DecodeSupport::Ready,
    ),
    pattern16(
        "hint",
        0xff0f,
        0xbf00,
        0x0002_0006,
        10,
        NONE,
        DecodeSupport::Ready,
    ),
    pattern16(
        "b-conditional",
        0xf000,
        0xd000,
        0x0002_0007,
        1,
        NONE,
        DecodeSupport::Ready,
    ),
    pattern16(
        "b",
        0xf800,
        0xe000,
        0x0002_0002,
        1,
        B_IMM,
        DecodeSupport::Ready,
    ),
    pattern16(
        "branch-exchange",
        0xff00,
        0x4700,
        0x0002_0008,
        10,
        NONE,
        DecodeSupport::Ready,
    ),
    pattern16(
        "svc",
        0xff00,
        0xdf00,
        0x0002_000a,
        20,
        NONE,
        DecodeSupport::Ready,
    ),
    pattern16(
        "bkpt",
        0xff00,
        0xbe00,
        0x0002_000b,
        20,
        NONE,
        DecodeSupport::Ready,
    ),
];
pub static PATTERNS_32: &[InstructionPattern] = &[
    pattern32(
        "bl",
        0xf800_d000,
        0xf000_d000,
        0x0002_0009,
        5,
        NONE,
        DecodeSupport::Ready,
    ),
    pattern32(
        "nop.w",
        u32::MAX,
        0xf3af_8000,
        0x0002_0004,
        30,
        NONE,
        DecodeSupport::Ready,
    ),
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Instruction {
    Nop,
    It {
        first_condition: u8,
        mask: u8,
    },
    Hint {
        operation: u8,
    },
    Branch {
        condition: Option<u8>,
        displacement: i32,
    },
    Exchange {
        link: bool,
        rm: u8,
    },
    BranchLink {
        displacement: i32,
    },
    Svc {
        immediate: u8,
    },
    Breakpoint {
        immediate: u8,
    },
}

pub(super) fn normalize(id: u32, bits: u32) -> Instruction {
    match id {
        0x0002_0001 | 0x0002_0004 => Instruction::Nop,
        0x0002_0005 => Instruction::It {
            first_condition: ((bits >> 4) & 0xf) as u8,
            mask: (bits & 0xf) as u8,
        },
        0x0002_0006 => Instruction::Hint {
            operation: ((bits >> 4) & 0xf) as u8,
        },
        0x0002_0007 => Instruction::Branch {
            condition: Some(((bits >> 8) & 0xf) as u8),
            displacement: (((bits & 0xff) << 24) as i32) >> 23,
        },
        0x0002_0002 => Instruction::Branch {
            condition: None,
            displacement: (((bits & 0x7ff) << 21) as i32) >> 20,
        },
        0x0002_0008 => Instruction::Exchange {
            link: bits & 0x80 != 0,
            rm: ((bits >> 3) & 0xf) as u8,
        },
        0x0002_0009 => {
            let s = (bits >> 26) & 1;
            let j1 = (bits >> 13) & 1;
            let j2 = (bits >> 11) & 1;
            let i1 = (!(j1 ^ s)) & 1;
            let i2 = (!(j2 ^ s)) & 1;
            let immediate = (s << 24)
                | (i1 << 23)
                | (i2 << 22)
                | (((bits >> 16) & 0x3ff) << 12)
                | ((bits & 0x7ff) << 1);
            Instruction::BranchLink {
                displacement: ((immediate << 7) as i32) >> 7,
            }
        }
        0x0002_000a => Instruction::Svc {
            immediate: bits as u8,
        },
        0x0002_000b => Instruction::Breakpoint {
            immediate: bits as u8,
        },
        _ => unreachable!(),
    }
}
