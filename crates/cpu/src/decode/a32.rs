//! Initial A32 declarative instruction table.

use std::sync::OnceLock;

use crate::{
    coverage::CoverageId,
    location::{ExecutionState, InstructionEncoding, InstructionSize, LocationDescriptor},
    profile::GuestCpuProfile,
};

use super::{
    DecodeResult,
    table::{
        DecodeSupport, DecoderTable, InstructionPattern, OperandField, OperandId, OperandKind,
        SemanticId,
    },
};

const NO_FIELDS: &[OperandField] = &[];
const NO_CONSTRAINTS: &[super::ReservedConstraint] = &[];
const NO_FEATURES: &[crate::profile::InstructionFeature] = &[];
const B_FIELDS: &[OperandField] = &[OperandField {
    id: OperandId::Immediate,
    lsb: 0,
    width: 24,
    kind: OperandKind::SignedScaled { scale: 2 },
}];

/// Minimal framework-validation table. ISA-family milestones extend it.
pub static PATTERNS: &[InstructionPattern] = &[
    InstructionPattern {
        name: "nop",
        execution_state: ExecutionState::A32,
        size: InstructionSize::Bits32,
        mask: u32::MAX,
        value: 0xe320_f000,
        operands: NO_FIELDS,
        reserved_constraints: NO_CONSTRAINTS,
        required_features: NO_FEATURES,
        semantic_id: SemanticId::new(0x0001_0001),
        coverage_id: CoverageId::new(0x0001_0001),
        priority: 0,
        support: DecodeSupport::Ready,
    },
    InstructionPattern {
        name: "b",
        execution_state: ExecutionState::A32,
        size: InstructionSize::Bits32,
        mask: 0xff00_0000,
        value: 0xea00_0000,
        operands: B_FIELDS,
        reserved_constraints: NO_CONSTRAINTS,
        required_features: NO_FEATURES,
        semantic_id: SemanticId::new(0x0001_0002),
        coverage_id: CoverageId::new(0x0001_0002),
        priority: 0,
        support: DecodeSupport::Ready,
    },
];

static TABLE: OnceLock<DecoderTable> = OnceLock::new();

pub(crate) fn decode(
    profile: &GuestCpuProfile,
    location: LocationDescriptor,
    encoding: InstructionEncoding,
) -> DecodeResult {
    table().decode(profile, location, encoding)
}

#[must_use]
pub fn table() -> &'static DecoderTable {
    TABLE.get_or_init(|| DecoderTable::compile(PATTERNS).expect("valid A32 decoder table"))
}
