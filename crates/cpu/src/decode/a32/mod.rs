//! Declarative A32 decoder and typed normalization.

pub mod control;
pub mod fp_simd;
pub mod integer;
pub mod memory;

use std::sync::OnceLock;

use crate::{
    coverage::CoverageId,
    location::{ExecutionState, InstructionEncoding, InstructionSize, LocationDescriptor},
    profile::InstructionFeature,
};

use super::{
    DecodeResult, DecodedOpcode,
    table::{DecodeSupport, DecoderTable, InstructionPattern, LoweringAvailability, OperandField},
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
    pub condition: crate::semantics::conditions::Condition,
    pub instruction: A32Instruction,
}

#[must_use]
pub fn normalize(opcode: &DecodedOpcode, encoding: InstructionEncoding) -> NormalizedA32 {
    let id = opcode.coverage_id().get();
    let bits = encoding.bits();
    let instruction = match id {
        0x0001_0001..=0x0001_0008 => A32Instruction::Control(control::normalize(id, bits)),
        0x0001_0010..=0x0001_0013 => A32Instruction::Integer(integer::normalize(id, bits)),
        0x0001_0020..=0x0001_0024 => A32Instruction::Memory(memory::normalize(id, bits)),
        0x0001_0030..=0x0001_0033 => A32Instruction::FpSimd(fp_simd::normalize(id, bits)),
        _ => unreachable!("A32 pattern lacks typed normalization"),
    };
    let condition = if id == 0x0001_0006 || bits >> 28 == 0xf {
        crate::semantics::conditions::Condition::Al
    } else {
        crate::semantics::conditions::Condition::from_encoding((bits >> 28) as u8)
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
) -> InstructionPattern {
    InstructionPattern {
        name,
        execution_state: ExecutionState::A32,
        size: InstructionSize::Bits32,
        mask,
        value,
        operands,
        required_features,
        coverage_id: CoverageId::new(id),
        priority,
        decoder: DecodeSupport::Ready,
        lowering: LoweringAvailability::Missing,
        regression_fixture: None,
    }
}

static PATTERNS: OnceLock<Box<[InstructionPattern]>> = OnceLock::new();
static SWITCH_1_TABLE: OnceLock<DecoderTable> = OnceLock::new();
static SWITCH_2_TABLE: OnceLock<DecoderTable> = OnceLock::new();

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
    decoder: crate::platform::PlatformDecoder,
    location: LocationDescriptor,
    encoding: InstructionEncoding,
) -> DecodeResult {
    platform_table(decoder.platform()).decode(location, encoding)
}

fn platform_table(platform: crate::platform::TargetPlatform) -> &'static DecoderTable {
    let cell = match platform {
        crate::platform::TargetPlatform::Switch1 => &SWITCH_1_TABLE,
        crate::platform::TargetPlatform::Switch2 => &SWITCH_2_TABLE,
    };
    cell.get_or_init(|| {
        DecoderTable::compile_for_platform(patterns(), platform)
            .expect("valid platform A32 decoder table")
    })
}

#[must_use]
pub fn table() -> &'static DecoderTable {
    platform_table(crate::platform::TargetPlatform::Switch1)
}
