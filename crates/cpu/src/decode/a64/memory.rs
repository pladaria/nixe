//! Normalized integer memory instructions.

use crate::decode::table::InstructionPattern;

use super::{NO_FEATURES, pattern};

pub(super) const PATTERNS: &[InstructionPattern] = &[
    pattern(
        "load-literal",
        0x3b00_0000,
        0x1800_0000,
        0x0000_0022,
        61,
        &[],
        NO_FEATURES,
    ),
    pattern(
        "load-store-unsigned",
        0x3b00_0000,
        0x3900_0000,
        0x0000_0023,
        60,
        &[],
        NO_FEATURES,
    ),
    pattern(
        "load-store-unscaled",
        0x3b20_0c00,
        0x3800_0000,
        0x0000_0024,
        120,
        &[],
        NO_FEATURES,
    ),
    pattern(
        "load-store-post-index",
        0x3b20_0c00,
        0x3800_0400,
        0x0000_0025,
        119,
        &[],
        NO_FEATURES,
    ),
    pattern(
        "load-store-pre-index",
        0x3b20_0c00,
        0x3800_0c00,
        0x0000_0026,
        118,
        &[],
        NO_FEATURES,
    ),
    pattern(
        "load-store-register",
        0x3b20_0c00,
        0x3820_0800,
        0x0000_0027,
        117,
        &[],
        NO_FEATURES,
    ),
    pattern(
        "load-store-pair",
        0x3e00_0000,
        0x2800_0000,
        0x0000_0028,
        59,
        &[],
        NO_FEATURES,
    ),
    pattern(
        "load-acquire",
        0x3fe0_fc00,
        0x08c0_fc00,
        0x0000_0029,
        147,
        &[],
        NO_FEATURES,
    ),
    pattern(
        "store-release",
        0x3fe0_fc00,
        0x0880_fc00,
        0x0000_002a,
        146,
        &[],
        NO_FEATURES,
    ),
    pattern(
        "load-exclusive",
        0x3fe0_fc00,
        0x0840_7c00,
        0x0000_002b,
        145,
        &[],
        NO_FEATURES,
    ),
    pattern(
        "store-exclusive",
        0x3f20_fc00,
        0x0800_7c00,
        0x0000_002c,
        144,
        &[],
        NO_FEATURES,
    ),
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Operands {
    pub rt: u8,
    pub rn: u8,
    pub rt2: u8,
    pub rm: u8,
    pub size: u8,
    pub opc: u8,
    pub mode: u8,
    pub option: u8,
    pub immediate_7: u8,
    pub immediate_9: u16,
    pub immediate_12: u16,
    pub immediate_19: u32,
    pub load: bool,
    pub ordered: bool,
    pub scaled: bool,
}

macro_rules! instructions {
    ($($variant:ident),+ $(,)?) => {
        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        pub enum Instruction { $($variant(Operands)),+ }

        impl Instruction {
            #[must_use]
            pub const fn operands(self) -> Operands {
                match self { $(Self::$variant(value) => value,)+ }
            }
        }
    };
}

instructions!(
    Literal,
    Unsigned,
    Unscaled,
    PostIndex,
    PreIndex,
    Register,
    Pair,
    LoadAcquire,
    StoreRelease,
    LoadExclusive,
    StoreExclusive,
);

pub(super) fn normalize(semantic_id: u32, bits: u32) -> Instruction {
    let operands = Operands {
        rt: (bits & 0x1f) as u8,
        rn: ((bits >> 5) & 0x1f) as u8,
        rt2: ((bits >> 10) & 0x1f) as u8,
        rm: ((bits >> 16) & 0x1f) as u8,
        size: (bits >> 30) as u8,
        opc: ((bits >> 22) & 3) as u8,
        mode: ((bits >> 23) & 3) as u8,
        option: ((bits >> 13) & 7) as u8,
        immediate_7: ((bits >> 15) & 0x7f) as u8,
        immediate_9: ((bits >> 12) & 0x1ff) as u16,
        immediate_12: ((bits >> 10) & 0xfff) as u16,
        immediate_19: (bits >> 5) & 0x7ffff,
        load: bits & (1 << 22) != 0,
        ordered: bits & (1 << 15) != 0,
        scaled: bits & (1 << 12) != 0,
    };
    match semantic_id {
        0x0000_0022 => Instruction::Literal(operands),
        0x0000_0023 => Instruction::Unsigned(operands),
        0x0000_0024 => Instruction::Unscaled(operands),
        0x0000_0025 => Instruction::PostIndex(operands),
        0x0000_0026 => Instruction::PreIndex(operands),
        0x0000_0027 => Instruction::Register(operands),
        0x0000_0028 => Instruction::Pair(operands),
        0x0000_0029 => Instruction::LoadAcquire(operands),
        0x0000_002a => Instruction::StoreRelease(operands),
        0x0000_002b => Instruction::LoadExclusive(operands),
        0x0000_002c => Instruction::StoreExclusive(operands),
        _ => unreachable!("memory semantic ID was routed to the wrong family"),
    }
}
