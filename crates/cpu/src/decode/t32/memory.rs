use super::{pattern16, pattern32};
use crate::decode::InstructionPattern;
use crate::decode::aarch32::{MemoryOffset, MemorySize, MultipleTransfer, SingleTransfer};

pub static PATTERNS_16: &[InstructionPattern] = &[
    pattern16("load-literal", 0xf800, 0x4800, 0x0002_0020, 5, &[]),
    pattern16("load-store-register", 0xf000, 0x5000, 0x0002_0021, 1, &[]),
    pattern16("load-store-immediate", 0xe000, 0x6000, 0x0002_0022, 1, &[]),
    pattern16("load-store-halfword", 0xf000, 0x8000, 0x0002_0023, 1, &[]),
    pattern16(
        "load-store-sp-relative",
        0xf000,
        0x9000,
        0x0002_0024,
        1,
        &[],
    ),
    pattern16("push", 0xfe00, 0xb400, 0x0002_0025, 10, &[]),
    pattern16("pop", 0xfe00, 0xbc00, 0x0002_0026, 10, &[]),
    pattern16("store-multiple", 0xf800, 0xc000, 0x0002_0027, 1, &[]),
    pattern16("load-multiple", 0xf800, 0xc800, 0x0002_0028, 1, &[]),
];
pub static PATTERNS_32: &[InstructionPattern] = &[
    pattern32(
        "load-word-immediate.w",
        0xfff0_0000,
        0xf8d0_0000,
        0x0002_0029,
        5,
        &[],
    ),
    pattern32(
        "store-word-immediate.w",
        0xfff0_0000,
        0xf8c0_0000,
        0x0002_002a,
        5,
        &[],
    ),
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Instruction {
    Single(SingleTransfer),
    Multiple(MultipleTransfer),
}

pub(super) fn normalize(id: u32, bits: u32) -> Instruction {
    match id {
        0x0002_0025 | 0x0002_0026 => {
            let load = id == 0x0002_0026;
            let extra = if bits & (1 << 8) != 0 {
                if load { 1 << 15 } else { 1 << 14 }
            } else {
                0
            };
            Instruction::Multiple(MultipleTransfer {
                load,
                rn: 13,
                registers: bits as u16 & 0xff | extra,
                increment: load,
                before: !load,
                writeback: true,
            })
        }
        0x0002_0027 | 0x0002_0028 => Instruction::Multiple(MultipleTransfer {
            load: id == 0x0002_0028,
            rn: ((bits >> 8) & 7) as u8,
            registers: bits as u16 & 0xff,
            increment: true,
            before: false,
            writeback: true,
        }),
        _ => Instruction::Single(single(id, bits)),
    }
}

fn single(id: u32, bits: u32) -> SingleTransfer {
    let (load, signed, size, rn, rt, offset) = match id {
        0x0002_0020 => (
            true,
            false,
            MemorySize::Word,
            15,
            ((bits >> 8) & 7) as u8,
            MemoryOffset::Immediate((bits & 0xff) << 2),
        ),
        0x0002_0021 => {
            let op = ((bits >> 9) & 7) as u8;
            let size = match op {
                0 | 4 => MemorySize::Word,
                1 | 5 | 7 => MemorySize::Halfword,
                _ => MemorySize::Byte,
            };
            (
                op >= 3,
                matches!(op, 3 | 7),
                size,
                ((bits >> 3) & 7) as u8,
                (bits & 7) as u8,
                MemoryOffset::Register {
                    rm: ((bits >> 6) & 7) as u8,
                    shift: crate::decode::aarch32::decode_immediate_shift(0, 0),
                },
            )
        }
        0x0002_0022 => {
            let byte = bits & (1 << 12) != 0;
            let load = bits & (1 << 11) != 0;
            let scale = if byte { 0 } else { 2 };
            (
                load,
                false,
                if byte {
                    MemorySize::Byte
                } else {
                    MemorySize::Word
                },
                ((bits >> 3) & 7) as u8,
                (bits & 7) as u8,
                MemoryOffset::Immediate(((bits >> 6) & 0x1f) << scale),
            )
        }
        0x0002_0023 => (
            bits & (1 << 11) != 0,
            false,
            MemorySize::Halfword,
            ((bits >> 3) & 7) as u8,
            (bits & 7) as u8,
            MemoryOffset::Immediate(((bits >> 6) & 0x1f) << 1),
        ),
        0x0002_0024 => (
            bits & (1 << 11) != 0,
            false,
            MemorySize::Word,
            13,
            ((bits >> 8) & 7) as u8,
            MemoryOffset::Immediate((bits & 0xff) << 2),
        ),
        0x0002_0029 | 0x0002_002a => (
            id == 0x0002_0029,
            false,
            MemorySize::Word,
            ((bits >> 16) & 0xf) as u8,
            ((bits >> 12) & 0xf) as u8,
            MemoryOffset::Immediate(bits & 0xfff),
        ),
        _ => unreachable!(),
    };
    SingleTransfer {
        load,
        signed,
        size,
        rn,
        rt,
        offset,
        add: true,
        pre_index: true,
        writeback: false,
    }
}
