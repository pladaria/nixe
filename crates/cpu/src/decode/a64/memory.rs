//! Normalized integer memory instructions.

use super::pattern;
use crate::decode::table::InstructionPattern;

pub(super) const PATTERNS: &[InstructionPattern] = &[
    pattern(
        "load-literal",
        0x3f00_0000,
        0x1800_0000,
        0x0000_0022,
        61,
        &[],
    ),
    pattern(
        "load-store-unsigned",
        0x3f00_0000,
        0x3900_0000,
        0x0000_0023,
        60,
        &[],
    ),
    pattern(
        "load-store-unscaled",
        0x3f20_0c00,
        0x3800_0000,
        0x0000_0024,
        120,
        &[],
    ),
    pattern(
        "load-store-post-index",
        0x3f20_0c00,
        0x3800_0400,
        0x0000_0025,
        119,
        &[],
    )
    .fixture32(0x3800_0401),
    pattern(
        "load-store-pre-index",
        0x3f20_0c00,
        0x3800_0c00,
        0x0000_0026,
        118,
        &[],
    )
    .fixture32(0x3800_0c01),
    pattern(
        "load-store-register",
        0x3f20_0c00,
        0x3820_0800,
        0x0000_0027,
        117,
        &[],
    )
    .fixture32(0xf861_6800),
    pattern(
        "load-store-pair",
        0x3e00_0000,
        0x2800_0000,
        0x0000_0028,
        59,
        &[],
    ),
    pattern(
        "load-acquire",
        0x3fe0_fc00,
        0x08c0_fc00,
        0x0000_0029,
        147,
        &[],
    ),
    pattern(
        "store-release",
        0x3fe0_fc00,
        0x0880_fc00,
        0x0000_002a,
        146,
        &[],
    ),
    pattern(
        "load-exclusive",
        0x3fe0_7c00,
        0x0840_7c00,
        0x0000_002b,
        145,
        &[],
    ),
    pattern(
        "store-exclusive",
        0x3f20_7c00,
        0x0800_7c00,
        0x0000_002c,
        144,
        &[],
    ),
    pattern(
        "load-exclusive-pair",
        0xbfe0_0000,
        0x8860_0000,
        0x0000_005e,
        150,
        &[],
    )
    .fixture32(0xc87f_0c20),
    pattern(
        "store-exclusive-pair",
        0xbf20_0000,
        0x8820_0000,
        0x0000_005f,
        149,
        &[],
    )
    .fixture32(0xc822_0c20),
    // Armv8.1-A FEAT_LSE atomic memory operations. The generic RMW pattern is
    // narrowed to allocated operations by the architectural allocation pass.
    // https://developer.arm.com/documentation/ddi0602/latest/Base-Instructions/LDADD--LDADDA--LDADDAL--LDADDL--Atomic-add-on-word-or-doubleword-in-memory-
    pattern(
        "atomic-read-modify-write",
        0x3f20_0c00,
        0x3820_0000,
        0x0000_002d,
        151,
        &[],
    )
    .fixture32(0xb8e0_0041),
    // https://developer.arm.com/documentation/ddi0602/latest/Base-Instructions/CAS--CASA--CASAL--CASL--Compare-and-swap-word-or-doubleword-in-memory-
    pattern(
        "compare-and-swap",
        0x3fa0_7c00,
        0x08a0_7c00,
        0x0000_002e,
        153,
        &[],
    )
    .fixture32(0x88e0_fc41),
    // https://developer.arm.com/documentation/ddi0602/latest/Base-Instructions/CASP--CASPA--CASPAL--CASPL--Compare-and-swap-pair-of-words-or-doublewords-in-memory-
    pattern(
        "compare-and-swap-pair",
        0xbfa0_7c00,
        0x0820_7c00,
        0x0000_002f,
        152,
        &[],
    )
    .fixture32(0x4860_fc82),
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
    pub acquire: bool,
    pub release: bool,
    pub atomic_opcode: u8,
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
    LoadExclusivePair,
    StoreExclusivePair,
    AtomicReadModifyWrite,
    CompareAndSwap,
    CompareAndSwapPair,
);

pub(super) fn normalize(instruction_id: u32, bits: u32) -> Instruction {
    let (acquire, release) = match instruction_id {
        0x0000_002d => (bits & (1 << 23) != 0, bits & (1 << 22) != 0),
        0x0000_002e | 0x0000_002f => (bits & (1 << 22) != 0, bits & (1 << 15) != 0),
        _ => (false, false),
    };
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
        acquire,
        release,
        atomic_opcode: ((bits >> 12) & 0xf) as u8,
    };
    match instruction_id {
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
        0x0000_002d => Instruction::AtomicReadModifyWrite(operands),
        0x0000_002e => Instruction::CompareAndSwap(operands),
        0x0000_002f => Instruction::CompareAndSwapPair(operands),
        0x0000_005e => Instruction::LoadExclusivePair(operands),
        0x0000_005f => Instruction::StoreExclusivePair(operands),
        _ => unreachable!("memory semantic ID was routed to the wrong family"),
    }
}
