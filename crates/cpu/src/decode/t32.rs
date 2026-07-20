//! Initial T32 declarative instruction tables.

use std::sync::OnceLock;

use crate::{
    coverage::CoverageId,
    error::InstructionDiagnostic,
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
const B16_FIELDS: &[OperandField] = &[OperandField {
    id: OperandId::Immediate,
    lsb: 0,
    width: 11,
    kind: OperandKind::SignedScaled { scale: 1 },
}];
const MOV16_FIELDS: &[OperandField] = &[
    OperandField {
        id: OperandId::Destination,
        lsb: 8,
        width: 3,
        kind: OperandKind::Register(super::RegisterClass::A32General),
    },
    OperandField {
        id: OperandId::Immediate,
        lsb: 0,
        width: 8,
        kind: OperandKind::Unsigned,
    },
];

pub static PATTERNS_16: &[InstructionPattern] = &[
    InstructionPattern {
        name: "nop",
        execution_state: ExecutionState::T32,
        size: InstructionSize::Bits16,
        mask: 0xffff,
        value: 0xbf00,
        operands: NO_FIELDS,
        reserved_constraints: NO_CONSTRAINTS,
        required_features: NO_FEATURES,
        semantic_id: SemanticId::new(0x0002_0001),
        coverage_id: CoverageId::new(0x0002_0001),
        priority: 0,
        support: DecodeSupport::RecognizedUnimplemented,
    },
    InstructionPattern {
        name: "b",
        execution_state: ExecutionState::T32,
        size: InstructionSize::Bits16,
        mask: 0xf800,
        value: 0xe000,
        operands: B16_FIELDS,
        reserved_constraints: NO_CONSTRAINTS,
        required_features: NO_FEATURES,
        semantic_id: SemanticId::new(0x0002_0002),
        coverage_id: CoverageId::new(0x0002_0002),
        priority: 0,
        support: DecodeSupport::RecognizedUnimplemented,
    },
    InstructionPattern {
        name: "movs",
        execution_state: ExecutionState::T32,
        size: InstructionSize::Bits16,
        mask: 0xf800,
        value: 0x2000,
        operands: MOV16_FIELDS,
        reserved_constraints: NO_CONSTRAINTS,
        required_features: NO_FEATURES,
        semantic_id: SemanticId::new(0x0002_0003),
        coverage_id: CoverageId::new(0x0002_0003),
        priority: 0,
        support: DecodeSupport::RecognizedUnimplemented,
    },
];

pub static PATTERNS_32: &[InstructionPattern] = &[InstructionPattern {
    name: "nop.w",
    execution_state: ExecutionState::T32,
    size: InstructionSize::Bits32,
    mask: u32::MAX,
    value: 0xf3af_8000,
    operands: NO_FIELDS,
    reserved_constraints: NO_CONSTRAINTS,
    required_features: NO_FEATURES,
    semantic_id: SemanticId::new(0x0002_0004),
    coverage_id: CoverageId::new(0x0002_0004),
    priority: 0,
    support: DecodeSupport::RecognizedUnimplemented,
}];

static TABLE_16: OnceLock<DecoderTable> = OnceLock::new();
static TABLE_32: OnceLock<DecoderTable> = OnceLock::new();

pub(crate) fn decode(
    profile: &GuestCpuProfile,
    location: LocationDescriptor,
    encoding: InstructionEncoding,
) -> DecodeResult {
    match encoding.size() {
        InstructionSize::Bits16 => table_16().decode(profile, location, encoding),
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
    TABLE_16.get_or_init(|| DecoderTable::compile(PATTERNS_16).expect("valid T32-16 decoder table"))
}

#[must_use]
pub fn table_32() -> &'static DecoderTable {
    TABLE_32.get_or_init(|| DecoderTable::compile(PATTERNS_32).expect("valid T32-32 decoder table"))
}
