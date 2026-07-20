//! Initial A64 declarative instruction table.

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
        RegisterClass, SemanticId,
    },
};

const NO_FIELDS: &[OperandField] = &[];
const NO_CONSTRAINTS: &[super::ReservedConstraint] = &[];
const NO_FEATURES: &[crate::profile::InstructionFeature] = &[];

const B_FIELDS: &[OperandField] = &[OperandField {
    id: OperandId::Immediate,
    lsb: 0,
    width: 26,
    kind: OperandKind::SignedScaled { scale: 2 },
}];
const ADD_IMMEDIATE_FIELDS: &[OperandField] = &[
    OperandField {
        id: OperandId::Destination,
        lsb: 0,
        width: 5,
        kind: OperandKind::Register(RegisterClass::A64General),
    },
    OperandField {
        id: OperandId::FirstSource,
        lsb: 5,
        width: 5,
        kind: OperandKind::Register(RegisterClass::A64General),
    },
    OperandField {
        id: OperandId::Immediate,
        lsb: 10,
        width: 12,
        kind: OperandKind::Unsigned,
    },
];

/// Minimal framework-validation table. ISA-family milestones extend it.
pub static PATTERNS: &[InstructionPattern] = &[
    InstructionPattern {
        name: "nop",
        execution_state: ExecutionState::A64,
        size: InstructionSize::Bits32,
        mask: u32::MAX,
        value: 0xd503_201f,
        operands: NO_FIELDS,
        reserved_constraints: NO_CONSTRAINTS,
        required_features: NO_FEATURES,
        semantic_id: SemanticId::new(0x0000_0001),
        coverage_id: CoverageId::new(0x0000_0001),
        priority: 0,
        support: DecodeSupport::Ready,
    },
    InstructionPattern {
        name: "b",
        execution_state: ExecutionState::A64,
        size: InstructionSize::Bits32,
        mask: 0xfc00_0000,
        value: 0x1400_0000,
        operands: B_FIELDS,
        reserved_constraints: NO_CONSTRAINTS,
        required_features: NO_FEATURES,
        semantic_id: SemanticId::new(0x0000_0002),
        coverage_id: CoverageId::new(0x0000_0002),
        priority: 0,
        support: DecodeSupport::Ready,
    },
    InstructionPattern {
        name: "add",
        execution_state: ExecutionState::A64,
        size: InstructionSize::Bits32,
        mask: 0xff40_0000,
        value: 0x9100_0000,
        operands: ADD_IMMEDIATE_FIELDS,
        reserved_constraints: NO_CONSTRAINTS,
        required_features: NO_FEATURES,
        semantic_id: SemanticId::new(0x0000_0003),
        coverage_id: CoverageId::new(0x0000_0003),
        priority: 0,
        support: DecodeSupport::RecognizedUnimplemented,
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

/// Returns the validated compiled table for consistency tests and diagnostics.
#[must_use]
pub fn table() -> &'static DecoderTable {
    TABLE.get_or_init(|| DecoderTable::compile(PATTERNS).expect("valid A64 decoder table"))
}
