//! Declarative A64 instruction table and authoritative frontend coverage.

pub mod control;
pub mod fp_simd;
pub mod integer;
pub mod memory;
pub mod system;

use std::sync::OnceLock;

use crate::{
    coverage::CoverageId,
    location::{InstructionEncoding, LocationDescriptor},
};

use super::{
    DecodeResult, DecodedOpcode,
    table::{DecodeSupport, DecoderTable, InstructionPattern, OperandField, RegressionFixture},
};

/// Fully normalized A64 instruction consumed by the family lifters.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum A64Instruction {
    Control(control::Instruction),
    System(system::Instruction),
    Integer(integer::Instruction),
    Memory(memory::Instruction),
    FpSimd(fp_simd::Instruction),
    RecognizedUnsupported { coverage_id: CoverageId },
}

/// Converts a table-classified A64 opcode into the typed lifter contract.
#[must_use]
pub fn normalize(opcode: &DecodedOpcode, encoding: InstructionEncoding) -> A64Instruction {
    let bits = encoding.bits();
    let instruction_id = opcode.coverage_id().get();
    if opcode.pattern().decoder == DecodeSupport::RecognizedUnimplemented {
        return A64Instruction::RecognizedUnsupported {
            coverage_id: opcode.coverage_id(),
        };
    }
    match instruction_id {
        0x0000_0001 | 0x0000_0002 | 0x0000_0004..=0x0000_000a | 0x0000_0044..=0x0000_0047 => {
            A64Instruction::Control(control::normalize(instruction_id, bits))
        }
        0x0000_000b..=0x0000_000f => {
            A64Instruction::System(system::normalize(instruction_id, bits))
        }
        0x0000_0003 | 0x0000_0010..=0x0000_001d | 0x0000_0020..=0x0000_0021 => {
            A64Instruction::Integer(integer::normalize(instruction_id, bits))
        }
        0x0000_0022..=0x0000_002f => {
            A64Instruction::Memory(memory::normalize(instruction_id, bits))
        }
        0x0000_0030..=0x0000_0043 | 0x0000_0048..=0x0000_005d | 0x0000_0060..=0x0000_00a0 => {
            A64Instruction::FpSimd(fp_simd::normalize(instruction_id, bits))
        }
        _ => unreachable!("A64 table contains an instruction without a typed family"),
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) const fn pattern(
    name: &'static str,
    mask: u32,
    value: u32,
    id: u32,
    priority: u16,
    operands: &'static [OperandField],
) -> InstructionPattern {
    InstructionPattern {
        name,
        mask,
        value,
        operands,
        coverage_id: CoverageId::new(id),
        priority,
        decoder: DecodeSupport::Ready,
        regression_fixture: Some(RegressionFixture {
            encoding: InstructionEncoding::from_u32(value),
        }),
    }
}

static PATTERNS: OnceLock<Box<[InstructionPattern]>> = OnceLock::new();
static SWITCH_1_TABLE: OnceLock<DecoderTable> = OnceLock::new();
static SWITCH_2_TABLE: OnceLock<DecoderTable> = OnceLock::new();

/// Returns the stable aggregate catalog compiled from family-owned patterns.
#[must_use]
pub fn patterns() -> &'static [InstructionPattern] {
    PATTERNS.get_or_init(|| {
        let mut patterns = Vec::new();
        patterns.extend_from_slice(control::PATTERNS);
        patterns.extend_from_slice(system::PATTERNS);
        patterns.extend_from_slice(integer::PATTERNS);
        patterns.extend_from_slice(memory::PATTERNS);
        patterns.extend_from_slice(fp_simd::PATTERNS);
        patterns.into_boxed_slice()
    })
}

pub(crate) fn decode(
    decoder: impl Into<crate::platform::PlatformDecoder>,
    location: LocationDescriptor,
    encoding: InstructionEncoding,
) -> DecodeResult {
    platform_table(decoder.into().platform()).decode(location, encoding)
}

fn platform_table(platform: crate::platform::TargetPlatform) -> &'static DecoderTable {
    let cell = match platform {
        crate::platform::TargetPlatform::Switch1 => &SWITCH_1_TABLE,
        crate::platform::TargetPlatform::Switch2 => &SWITCH_2_TABLE,
    };
    cell.get_or_init(|| {
        DecoderTable::compile_for_platform(patterns(), platform)
            .expect("valid platform A64 decoder table")
    })
}

/// Returns the validated compiled table for consistency tests and diagnostics.
#[must_use]
pub fn table() -> &'static DecoderTable {
    platform_table(crate::platform::TargetPlatform::Switch1)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use crate::{
        decode::{DecodeResult, table::DecodeSupport},
        location::LocationDescriptor,
        platform::TargetPlatform,
    };
    use nixe_memory::GuestVirtualAddress;

    use super::*;

    #[test]
    fn every_platform_catalog_entry_has_one_decodable_fixture_and_identity() {
        let unique_ids: BTreeSet<_> = patterns()
            .iter()
            .map(|pattern| pattern.coverage_id)
            .collect();
        assert_eq!(unique_ids.len(), patterns().len());

        for platform in [TargetPlatform::Switch1, TargetPlatform::Switch2] {
            let location =
                LocationDescriptor::new(GuestVirtualAddress::new(0), platform.profile_id());
            for pattern in patterns() {
                let fixture = pattern
                    .regression_fixture
                    .expect("every A64 catalog entry has a regression fixture");
                let decoded = platform_table(platform).decode(location, fixture.encoding);
                let (coverage_id, support) = match decoded {
                    DecodeResult::Decoded(decoded) => {
                        (decoded.instruction.coverage_id(), DecodeSupport::Ready)
                    }
                    DecodeResult::RecognizedUnimplemented(decoded) => (
                        decoded.instruction.coverage_id(),
                        DecodeSupport::RecognizedUnimplemented,
                    ),
                    rejected => panic!(
                        "{platform:?} rejected fixture for {}: {rejected:?}",
                        pattern.name
                    ),
                };
                assert_eq!(coverage_id, pattern.coverage_id);
                assert_eq!(support, pattern.decoder);
            }
        }
    }
}
