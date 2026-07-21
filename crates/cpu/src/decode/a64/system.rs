//! Normalized hints, system-register, barrier, and cache instructions.

use crate::decode::table::{DecodeSupport, InstructionPattern};

use super::{NO_FEATURES, pattern};

pub(super) const PATTERNS: &[InstructionPattern] = &[
    pattern(
        "hint",
        0xffff_f01f,
        0xd503_201f,
        0x0000_000b,
        190,
        &[],
        NO_FEATURES,
        DecodeSupport::Ready,
    ),
    pattern(
        "mrs",
        0xfff0_0000,
        0xd530_0000,
        0x0000_000c,
        70,
        &[],
        NO_FEATURES,
        DecodeSupport::Ready,
    ),
    pattern(
        "msr-register",
        0xfff0_0000,
        0xd510_0000,
        0x0000_000d,
        69,
        &[],
        NO_FEATURES,
        DecodeSupport::Ready,
    ),
    pattern(
        "barrier",
        0xffff_f01f,
        0xd503_301f,
        0x0000_000e,
        189,
        &[],
        NO_FEATURES,
        DecodeSupport::Ready,
    ),
    pattern(
        "system",
        0xffc0_0000,
        0xd500_0000,
        0x0000_000f,
        20,
        &[],
        NO_FEATURES,
        DecodeSupport::Ready,
    ),
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Operands {
    pub rt: u8,
    pub hint: u8,
    pub barrier_opcode: u8,
    pub barrier_option: u8,
    pub system_key: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Instruction {
    Hint(Operands),
    ReadRegister(Operands),
    WriteRegister(Operands),
    Barrier(Operands),
    System(Operands),
}

impl Instruction {
    #[must_use]
    pub const fn operands(self) -> Operands {
        match self {
            Self::Hint(value)
            | Self::ReadRegister(value)
            | Self::WriteRegister(value)
            | Self::Barrier(value)
            | Self::System(value) => value,
        }
    }
}

pub(super) fn normalize(semantic_id: u32, bits: u32) -> Instruction {
    let operands = Operands {
        rt: (bits & 0x1f) as u8,
        hint: ((bits >> 5) & 0x7f) as u8,
        barrier_opcode: ((bits >> 5) & 7) as u8,
        barrier_option: ((bits >> 8) & 0xf) as u8,
        system_key: bits & 0xffff_ffe0,
    };
    match semantic_id {
        0x0000_000b => Instruction::Hint(operands),
        0x0000_000c => Instruction::ReadRegister(operands),
        0x0000_000d => Instruction::WriteRegister(operands),
        0x0000_000e => Instruction::Barrier(operands),
        0x0000_000f => Instruction::System(operands),
        _ => unreachable!("system semantic ID was routed to the wrong family"),
    }
}
