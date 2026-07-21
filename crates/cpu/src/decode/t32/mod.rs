//! Declarative T32 decoder and typed normalization.

pub(crate) mod control;
pub(crate) mod integer;
pub(crate) mod memory;

use super::{
    DecodeResult, DecodedOpcode,
    table::{DecodeSupport, DecoderTable, InstructionPattern, OperandField, SemanticId},
};
use crate::{
    coverage::CoverageId,
    error::InstructionDiagnostic,
    location::{ExecutionState, InstructionEncoding, InstructionSize, LocationDescriptor},
    profile::{GuestCpuProfile, InstructionFeature},
};
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum T32Instruction {
    Control(control::Instruction),
    Integer(integer::Instruction),
    Memory(memory::Instruction),
}

#[must_use]
pub fn normalize(opcode: &DecodedOpcode, encoding: InstructionEncoding) -> T32Instruction {
    let id = opcode.semantic_id().get();
    let bits = encoding.bits();
    match id {
        0x0002_0001 | 0x0002_0002 | 0x0002_0004..=0x0002_000b => {
            T32Instruction::Control(control::normalize(id, bits))
        }
        0x0002_0003 => T32Instruction::Integer(integer::normalize(id, bits)),
        0x0002_0010..=0x0002_0018 => T32Instruction::Integer(integer::normalize(id, bits)),
        0x0002_0020..=0x0002_002a => T32Instruction::Memory(memory::normalize(id, bits)),
        _ => unreachable!("T32 pattern lacks typed normalization"),
    }
}

pub(super) const NO_FEATURES: &[InstructionFeature] = &[];

#[allow(clippy::too_many_arguments)]
pub(super) const fn pattern16(
    name: &'static str,
    mask: u32,
    value: u32,
    id: u32,
    priority: u16,
    operands: &'static [OperandField],
    support: DecodeSupport,
) -> InstructionPattern {
    pattern(
        name,
        InstructionSize::Bits16,
        mask,
        value,
        id,
        priority,
        operands,
        support,
    )
}
#[allow(clippy::too_many_arguments)]
pub(super) const fn pattern32(
    name: &'static str,
    mask: u32,
    value: u32,
    id: u32,
    priority: u16,
    operands: &'static [OperandField],
    support: DecodeSupport,
) -> InstructionPattern {
    pattern(
        name,
        InstructionSize::Bits32,
        mask,
        value,
        id,
        priority,
        operands,
        support,
    )
}
#[allow(clippy::too_many_arguments)]
const fn pattern(
    name: &'static str,
    size: InstructionSize,
    mask: u32,
    value: u32,
    id: u32,
    priority: u16,
    operands: &'static [OperandField],
    support: DecodeSupport,
) -> InstructionPattern {
    InstructionPattern {
        name,
        execution_state: ExecutionState::T32,
        size,
        mask,
        value,
        operands,
        reserved_constraints: &[],
        required_features: NO_FEATURES,
        semantic_id: SemanticId::new(id),
        coverage_id: CoverageId::new(id),
        priority,
        support,
    }
}

static PATTERNS_16_CELL: OnceLock<Box<[InstructionPattern]>> = OnceLock::new();
static PATTERNS_32_CELL: OnceLock<Box<[InstructionPattern]>> = OnceLock::new();
static TABLE_16: OnceLock<DecoderTable> = OnceLock::new();
static TABLE_32: OnceLock<DecoderTable> = OnceLock::new();

#[must_use]
pub fn patterns_16() -> &'static [InstructionPattern] {
    PATTERNS_16_CELL.get_or_init(|| {
        let mut p = Vec::new();
        p.extend_from_slice(control::PATTERNS_16);
        p.extend_from_slice(integer::PATTERNS_16);
        p.extend_from_slice(memory::PATTERNS_16);
        p.into_boxed_slice()
    })
}
#[must_use]
pub fn patterns_32() -> &'static [InstructionPattern] {
    PATTERNS_32_CELL.get_or_init(|| {
        let mut p = Vec::new();
        p.extend_from_slice(control::PATTERNS_32);
        p.extend_from_slice(integer::PATTERNS_32);
        p.extend_from_slice(memory::PATTERNS_32);
        p.into_boxed_slice()
    })
}

pub(crate) fn decode(
    profile: &GuestCpuProfile,
    location: LocationDescriptor,
    encoding: InstructionEncoding,
) -> DecodeResult {
    match encoding.size() {
        InstructionSize::Bits16 => {
            if encoding.bits() & 0xff00 == 0xde00 {
                return DecodeResult::Unallocated {
                    instruction: InstructionDiagnostic::new(location, encoding),
                    reason: "permanently undefined T32 encoding",
                };
            }
            table_16().decode(profile, location, encoding)
        }
        InstructionSize::Bits32 => {
            if !crate::location::is_t32_32_bit_prefix((encoding.bits() >> 16) as u16) {
                return DecodeResult::Unallocated {
                    instruction: InstructionDiagnostic::new(location, encoding),
                    reason: "32-bit T32 encoding lacks a 32-bit prefix",
                };
            }
            table_32().decode(profile, location, encoding)
        }
    }
}

#[must_use]
pub fn table_16() -> &'static DecoderTable {
    TABLE_16
        .get_or_init(|| DecoderTable::compile(patterns_16()).expect("valid T32-16 decoder table"))
}
#[must_use]
pub fn table_32() -> &'static DecoderTable {
    TABLE_32
        .get_or_init(|| DecoderTable::compile(patterns_32()).expect("valid T32-32 decoder table"))
}
