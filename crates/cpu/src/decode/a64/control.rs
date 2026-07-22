//! Normalized control-flow and exception instructions.

use crate::decode::table::{InstructionPattern, OperandField, OperandId, OperandKind};

use super::{NO_FEATURES, pattern};

const B_FIELDS: &[OperandField] = &[OperandField {
    id: OperandId::Immediate,
    lsb: 0,
    width: 26,
    kind: OperandKind::SignedScaled { scale: 2 },
}];

pub(super) const PATTERNS: &[InstructionPattern] = &[
    pattern(
        "nop",
        u32::MAX,
        0xd503_201f,
        0x0000_0001,
        200,
        &[],
        NO_FEATURES,
    ),
    pattern(
        "b",
        0xfc00_0000,
        0x1400_0000,
        0x0000_0002,
        199,
        B_FIELDS,
        NO_FEATURES,
    ),
    pattern(
        "bl",
        0xfc00_0000,
        0x9400_0000,
        0x0000_0004,
        198,
        &[],
        NO_FEATURES,
    ),
    pattern(
        "br",
        0xffff_fc1f,
        0xd61f_0000,
        0x0000_0005,
        194,
        &[],
        NO_FEATURES,
    ),
    pattern(
        "blr",
        0xffff_fc1f,
        0xd63f_0000,
        0x0000_0044,
        194,
        &[],
        NO_FEATURES,
    ),
    pattern(
        "ret",
        0xffff_fc1f,
        0xd65f_0000,
        0x0000_0045,
        194,
        &[],
        NO_FEATURES,
    ),
    pattern(
        "eret",
        u32::MAX,
        0xd69f_03e0,
        0x0000_0046,
        194,
        &[],
        NO_FEATURES,
    ),
    pattern(
        "drps",
        u32::MAX,
        0xd6bf_03e0,
        0x0000_0047,
        194,
        &[],
        NO_FEATURES,
    ),
    pattern(
        "b.cond",
        0xff00_0010,
        0x5400_0000,
        0x0000_0006,
        197,
        &[],
        NO_FEATURES,
    ),
    pattern(
        "compare-branch",
        0x7e00_0000,
        0x3400_0000,
        0x0000_0007,
        78,
        &[],
        NO_FEATURES,
    ),
    pattern(
        "test-branch",
        0x7e00_0000,
        0x3600_0000,
        0x0000_0008,
        77,
        &[],
        NO_FEATURES,
    ),
    pattern(
        "svc",
        0xffe0_001f,
        0xd400_0001,
        0x0000_0009,
        196,
        &[],
        NO_FEATURES,
    ),
    pattern(
        "brk",
        0xffe0_001f,
        0xd420_0000,
        0x0000_000a,
        195,
        &[],
        NO_FEATURES,
    ),
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Operands {
    pub rd: u8,
    pub rn: u8,
    pub condition: u8,
    pub bit_index: u8,
    pub branch_register_key: u32,
    pub immediate_14: u16,
    pub immediate_16: u16,
    pub immediate_19: u32,
    pub immediate_26: u32,
    pub nonzero: bool,
    pub width_64: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Instruction {
    Nop(Operands),
    BranchImmediate(Operands),
    BranchLinkImmediate(Operands),
    BranchRegister(Operands),
    ConditionalBranch(Operands),
    CompareBranch(Operands),
    TestBranch(Operands),
    SupervisorCall(Operands),
    Breakpoint(Operands),
}

impl Instruction {
    #[must_use]
    pub const fn operands(self) -> Operands {
        match self {
            Self::Nop(value)
            | Self::BranchImmediate(value)
            | Self::BranchLinkImmediate(value)
            | Self::BranchRegister(value)
            | Self::ConditionalBranch(value)
            | Self::CompareBranch(value)
            | Self::TestBranch(value)
            | Self::SupervisorCall(value)
            | Self::Breakpoint(value) => value,
        }
    }
}

pub(super) fn normalize(semantic_id: u32, bits: u32) -> Instruction {
    let operands = Operands {
        rd: (bits & 0x1f) as u8,
        rn: ((bits >> 5) & 0x1f) as u8,
        condition: (bits & 0xf) as u8,
        bit_index: ((((bits >> 31) & 1) << 5) | ((bits >> 19) & 0x1f)) as u8,
        branch_register_key: bits & 0xffff_fc1f,
        immediate_14: ((bits >> 5) & 0x3fff) as u16,
        immediate_16: ((bits >> 5) & 0xffff) as u16,
        immediate_19: (bits >> 5) & 0x7ffff,
        immediate_26: bits & 0x03ff_ffff,
        nonzero: bits & (1 << 24) != 0,
        width_64: bits & (1 << 31) != 0,
    };
    match semantic_id {
        0x0000_0001 => Instruction::Nop(operands),
        0x0000_0002 => Instruction::BranchImmediate(operands),
        0x0000_0004 => Instruction::BranchLinkImmediate(operands),
        0x0000_0005 | 0x0000_0044..=0x0000_0047 => Instruction::BranchRegister(operands),
        0x0000_0006 => Instruction::ConditionalBranch(operands),
        0x0000_0007 => Instruction::CompareBranch(operands),
        0x0000_0008 => Instruction::TestBranch(operands),
        0x0000_0009 => Instruction::SupervisorCall(operands),
        0x0000_000a => Instruction::Breakpoint(operands),
        _ => unreachable!("control semantic ID was routed to the wrong family"),
    }
}
