use super::pattern;
use crate::decode::aarch32::{VectorDataProcessing, VectorOperation, VectorSize, VectorTransfer};
use crate::{decode::InstructionPattern, profile::InstructionFeature};

const SIMD: &[InstructionFeature] = &[InstructionFeature::AdvancedSimd];

pub static PATTERNS: &[InstructionPattern] = &[
    pattern(
        "vfp-data-processing",
        0x0e00_0a00,
        0x0e00_0a00,
        0x0001_0030,
        0,
        &[],
        SIMD,
    ),
    pattern(
        "neon-bitwise",
        0xfe00_0110,
        0xf200_0110,
        0x0001_0031,
        5,
        &[],
        SIMD,
    ),
    pattern(
        "neon-integer",
        0xfe00_0800,
        0xf200_0800,
        0x0001_0032,
        1,
        &[],
        SIMD,
    ),
    pattern(
        "neon-memory",
        0xff00_0000,
        0xf400_0000,
        0x0001_0033,
        5,
        &[],
        SIMD,
    ),
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Instruction {
    Data(VectorDataProcessing),
    Memory(VectorTransfer),
}

pub(super) fn normalize(id: u32, bits: u32) -> Instruction {
    if id == 0x0001_0033 {
        let rm = (bits & 0xf) as u8;
        return Instruction::Memory(VectorTransfer {
            load: bits & (1 << 21) != 0,
            rn: ((bits >> 16) & 0xf) as u8,
            vd: (((bits >> 12) & 0xf) | ((bits >> 18) & 0x10)) as u8,
            register_count: 1,
            writeback_rm: (rm != 15).then_some(rm),
        });
    }
    let q = bits & (1 << 6) != 0;
    let vd = (((bits >> 12) & 0xf) | ((bits >> 18) & 0x10)) as u8;
    let vn = (((bits >> 16) & 0xf) | ((bits >> 3) & 0x10)) as u8;
    let vm = ((bits & 0xf) | ((bits >> 1) & 0x10)) as u8;
    let operation = match id {
        0x0001_0030 => match (bits >> 20) & 3 {
            0 => VectorOperation::AddF32,
            1 => VectorOperation::SubtractF32,
            _ => VectorOperation::MultiplyF32,
        },
        0x0001_0031 => match (bits >> 20) & 3 {
            0 => VectorOperation::And,
            1 => VectorOperation::BitClear,
            2 => VectorOperation::Or,
            _ => VectorOperation::ExclusiveOr,
        },
        0x0001_0032 => {
            if bits & (1 << 21) == 0 {
                VectorOperation::AddInteger {
                    lane_bits: 8 << ((bits >> 20) & 3),
                }
            } else {
                VectorOperation::SubtractInteger {
                    lane_bits: 8 << ((bits >> 20) & 3),
                }
            }
        }
        _ => unreachable!(),
    };
    Instruction::Data(VectorDataProcessing {
        operation,
        size: if q { VectorSize::Q } else { VectorSize::D },
        vd: if q { vd / 2 } else { vd },
        vn: if q { vn / 2 } else { vn },
        vm: if q { vm / 2 } else { vm },
    })
}
