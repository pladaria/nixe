//! Declarative A32 decoder and typed normalization.

pub(crate) mod control;
pub(crate) mod fp_simd;
pub(crate) mod integer;
pub(crate) mod memory;

use std::sync::OnceLock;

use crate::{
    coverage::CoverageId,
    location::{ExecutionState, InstructionEncoding, InstructionSize, LocationDescriptor},
    profile::{GuestCpuProfile, InstructionFeature},
};

use super::{
    DecodeResult, DecodedOpcode,
    table::{DecodeSupport, DecoderTable, InstructionPattern, OperandField, SemanticId},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum A32Instruction {
    Control(control::Instruction),
    Integer(integer::Instruction),
    Memory(memory::Instruction),
    FpSimd(fp_simd::Instruction),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NormalizedA32 {
    pub condition: crate::ir::op::Condition,
    pub instruction: A32Instruction,
}

#[must_use]
pub fn normalize(opcode: &DecodedOpcode, encoding: InstructionEncoding) -> NormalizedA32 {
    let id = opcode.semantic_id().get();
    let bits = encoding.bits();
    let instruction = match id {
        0x0001_0001..=0x0001_0008 => A32Instruction::Control(control::normalize(id, bits)),
        0x0001_0010..=0x0001_0013 => A32Instruction::Integer(integer::normalize(id, bits)),
        0x0001_0020..=0x0001_0024 => A32Instruction::Memory(memory::normalize(id, bits)),
        0x0001_0030..=0x0001_0033 => A32Instruction::FpSimd(fp_simd::normalize(id, bits)),
        _ => unreachable!("A32 pattern lacks typed normalization"),
    };
    let condition = if id == 0x0001_0006 || bits >> 28 == 0xf {
        crate::ir::op::Condition::Al
    } else {
        crate::ir::op::Condition::from_encoding((bits >> 28) as u8)
    };
    NormalizedA32 {
        condition,
        instruction,
    }
}

pub(super) const NO_FEATURES: &[InstructionFeature] = &[];

#[allow(clippy::too_many_arguments)]
pub(super) const fn pattern(
    name: &'static str,
    mask: u32,
    value: u32,
    id: u32,
    priority: u16,
    operands: &'static [OperandField],
    required_features: &'static [InstructionFeature],
    support: DecodeSupport,
) -> InstructionPattern {
    InstructionPattern {
        name,
        execution_state: ExecutionState::A32,
        size: InstructionSize::Bits32,
        mask,
        value,
        operands,
        reserved_constraints: &[],
        required_features,
        semantic_id: SemanticId::new(id),
        coverage_id: CoverageId::new(id),
        priority,
        support,
    }
}

static PATTERNS: OnceLock<Box<[InstructionPattern]>> = OnceLock::new();
static TABLE: OnceLock<DecoderTable> = OnceLock::new();

#[must_use]
pub fn patterns() -> &'static [InstructionPattern] {
    PATTERNS.get_or_init(|| {
        let mut patterns = Vec::new();
        patterns.extend_from_slice(control::PATTERNS);
        patterns.extend_from_slice(integer::PATTERNS);
        patterns.extend_from_slice(memory::PATTERNS);
        patterns.extend_from_slice(fp_simd::PATTERNS);
        patterns.into_boxed_slice()
    })
}

pub(crate) fn decode(
    profile: &GuestCpuProfile,
    location: LocationDescriptor,
    encoding: InstructionEncoding,
) -> DecodeResult {
    let result = table().decode(profile, location, encoding);
    if encoding.bits() >> 28 != 0xf {
        return result;
    }
    match &result {
        DecodeResult::Decoded(decoded) | DecodeResult::RecognizedUnimplemented(decoded)
            if matches!(
                decoded.instruction.semantic_id().get(),
                0x0001_0006 | 0x0001_0031..=0x0001_0033
            ) =>
        {
            result
        }
        _ => DecodeResult::Unallocated {
            instruction: crate::error::InstructionDiagnostic::new(location, encoding),
            reason: "encoding is not allocated in the A32 unconditional space",
        },
    }
}

#[must_use]
pub fn table() -> &'static DecoderTable {
    TABLE.get_or_init(|| DecoderTable::compile(patterns()).expect("valid A32 decoder table"))
}
