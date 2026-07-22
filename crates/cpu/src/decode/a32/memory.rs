use crate::decode::InstructionPattern;
use crate::decode::aarch32::{
    ExclusiveTransfer, MemoryOffset, MemorySize, MultipleTransfer, SingleTransfer,
    decode_immediate_shift,
};

use super::{NO_FEATURES, pattern};

pub static PATTERNS: &[InstructionPattern] = &[
    pattern(
        "load-store-exclusive",
        0x0f00_00f0,
        0x0100_0090,
        0x0001_0024,
        30,
        &[],
        NO_FEATURES,
    ),
    pattern(
        "load-store-extra",
        0x0e00_0090,
        0x0000_0090,
        0x0001_0023,
        10,
        &[],
        NO_FEATURES,
    ),
    pattern(
        "load-store-single",
        0x0c00_0000,
        0x0400_0000,
        0x0001_0020,
        0,
        &[],
        NO_FEATURES,
    ),
    pattern(
        "load-store-multiple",
        0x0e00_0000,
        0x0800_0000,
        0x0001_0021,
        0,
        &[],
        NO_FEATURES,
    ),
    pattern(
        "load-store-acquire-release",
        0x0f00_00f0,
        0x0100_0080,
        0x0001_0022,
        25,
        &[],
        NO_FEATURES,
    ),
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Instruction {
    Single(SingleTransfer),
    Multiple(MultipleTransfer),
    Exclusive(ExclusiveTransfer),
    AcquireRelease(ExclusiveTransfer),
}

pub(super) fn normalize(id: u32, bits: u32) -> Instruction {
    match id {
        0x0001_0020 => Instruction::Single(single(bits, false)),
        0x0001_0023 => Instruction::Single(single(bits, true)),
        0x0001_0021 => Instruction::Multiple(MultipleTransfer {
            load: bits & (1 << 20) != 0,
            rn: ((bits >> 16) & 0xf) as u8,
            registers: bits as u16,
            increment: bits & (1 << 23) != 0,
            before: bits & (1 << 24) != 0,
            writeback: bits & (1 << 21) != 0,
        }),
        0x0001_0022 | 0x0001_0024 => {
            let load = bits & (1 << 20) != 0;
            let transfer = ExclusiveTransfer {
                load,
                size: MemorySize::Word,
                rn: ((bits >> 16) & 0xf) as u8,
                rt: if load {
                    ((bits >> 12) & 0xf) as u8
                } else {
                    (bits & 0xf) as u8
                },
                status: (!load).then_some(((bits >> 12) & 0xf) as u8),
                acquire: id == 0x0001_0022 && load,
                release: id == 0x0001_0022 && !load,
            };
            if id == 0x0001_0022 {
                Instruction::AcquireRelease(transfer)
            } else {
                Instruction::Exclusive(transfer)
            }
        }
        _ => unreachable!(),
    }
}

fn single(bits: u32, extra: bool) -> SingleTransfer {
    if extra {
        let immediate = bits & (1 << 22) != 0;
        let offset = if immediate {
            MemoryOffset::Immediate(((bits >> 4) & 0xf0) | (bits & 0xf))
        } else {
            MemoryOffset::Register {
                rm: (bits & 0xf) as u8,
                shift: decode_immediate_shift(0, 0),
            }
        };
        let op = ((bits >> 5) & 3) as u8;
        return SingleTransfer {
            load: bits & (1 << 20) != 0,
            signed: op >= 2,
            size: if op == 2 {
                MemorySize::Byte
            } else {
                MemorySize::Halfword
            },
            rn: ((bits >> 16) & 0xf) as u8,
            rt: ((bits >> 12) & 0xf) as u8,
            offset,
            add: bits & (1 << 23) != 0,
            pre_index: bits & (1 << 24) != 0,
            writeback: bits & (1 << 21) != 0 || bits & (1 << 24) == 0,
        };
    }
    let register = bits & (1 << 25) != 0;
    SingleTransfer {
        load: bits & (1 << 20) != 0,
        signed: false,
        size: if bits & (1 << 22) != 0 {
            MemorySize::Byte
        } else {
            MemorySize::Word
        },
        rn: ((bits >> 16) & 0xf) as u8,
        rt: ((bits >> 12) & 0xf) as u8,
        offset: if register {
            MemoryOffset::Register {
                rm: (bits & 0xf) as u8,
                shift: decode_immediate_shift(((bits >> 5) & 3) as u8, ((bits >> 7) & 0x1f) as u8),
            }
        } else {
            MemoryOffset::Immediate(bits & 0xfff)
        },
        add: bits & (1 << 23) != 0,
        pre_index: bits & (1 << 24) != 0,
        writeback: bits & (1 << 21) != 0 || bits & (1 << 24) == 0,
    }
}
