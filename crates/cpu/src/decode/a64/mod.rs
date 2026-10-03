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
        0x0000_000b..=0x0000_000f | 0x0000_001e => {
            A64Instruction::System(system::normalize(instruction_id, bits))
        }
        0x0000_0003 | 0x0000_0010..=0x0000_001d | 0x0000_0020..=0x0000_0021 => {
            A64Instruction::Integer(integer::normalize(instruction_id, bits))
        }
        0x0000_0022..=0x0000_002f | 0x0000_005e..=0x0000_005f => {
            A64Instruction::Memory(memory::normalize(instruction_id, bits))
        }
        0x0000_0030..=0x0000_0043 | 0x0000_0048..=0x0000_005d | 0x0000_0060..=0x0000_00a8 => {
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

    #[test]
    fn fminmaxnm_decodes_single_double_and_keeps_half_precision_unsupported() {
        for minimum in [false, true] {
            for platform in [TargetPlatform::Switch1, TargetPlatform::Switch2] {
                let location =
                    LocationDescriptor::new(GuestVirtualAddress::new(0), platform.profile_id());
                for (word, rd, rn, rm, opc) in
                    [(0x1e21_6802, 2, 0, 1, 0), (0x1e7f_6bff, 31, 31, 31, 1)]
                {
                    let word = word | (u32::from(minimum) << 12);
                    let encoding = InstructionEncoding::from_u32(word);
                    let DecodeResult::Decoded(decoded) = decode(platform, location, encoding)
                    else {
                        panic!("FMAXNM {word:08x} did not decode");
                    };
                    let instruction = normalize(&decoded.instruction, encoding);
                    let fields =
                        match instruction {
                            A64Instruction::FpSimd(fp_simd::Instruction::ScalarFloatMinNumber(
                                f,
                            )) if minimum => f,
                            A64Instruction::FpSimd(fp_simd::Instruction::ScalarFloatMaxNumber(
                                f,
                            )) if !minimum => f,
                            _ => panic!("{decoded:?}"),
                        };
                    assert_eq!(
                        (fields.rd, fields.rn, fields.rm, fields.opc),
                        (rd, rn, rm, opc)
                    );
                }
                assert!(!matches!(
                    decode(
                        platform,
                        location,
                        InstructionEncoding::from_u32(0x1ee1_6802 | (u32::from(minimum) << 12))
                    ),
                    DecodeResult::Decoded(_)
                ));
            }
        }
    }

    #[test]
    fn uaddlv_decodes_all_arrangements_and_rejects_reserved_sizes() {
        for platform in [TargetPlatform::Switch1, TargetPlatform::Switch2] {
            let location =
                LocationDescriptor::new(GuestVirtualAddress::new(0), platform.profile_id());
            for (size, full) in [(0, false), (0, true), (1, false), (1, true), (2, true)] {
                let word = 0x2e30_3800 | (size << 22) | (u32::from(full) << 30) | (31 << 5) | 31;
                let encoding = InstructionEncoding::from_u32(word);
                let DecodeResult::Decoded(decoded) = decode(platform, location, encoding) else {
                    panic!("UADDLV {word:08x} was not decoded");
                };
                let A64Instruction::FpSimd(fp_simd::Instruction::UnsignedAddLongAcrossVector(
                    fields,
                )) = normalize(&decoded.instruction, encoding)
                else {
                    panic!("{decoded:?}");
                };
                assert_eq!(
                    (fields.rd, fields.rn, fields.opc, fields.vector_128),
                    (31, 31, size as u8, full)
                );
            }
            for word in [0x2eb0_3800, 0x2ef0_3800, 0x6ef0_3800] {
                assert!(!matches!(
                    decode(platform, location, InstructionEncoding::from_u32(word)),
                    DecodeResult::Decoded(_)
                ));
            }
        }
    }

    #[test]
    fn fmul_vector_decodes_cube_operands_and_validates_shapes() {
        for platform in [TargetPlatform::Switch1, TargetPlatform::Switch2] {
            let location = LocationDescriptor::new(
                GuestVirtualAddress::new(0x7100_132c),
                platform.profile_id(),
            );
            for (word, wide, full) in [
                (0x6e3c_dfbd, false, true),
                (0x2e3c_dfbd, false, false),
                (0x6e7c_dfbd, true, true),
            ] {
                let encoding = InstructionEncoding::from_u32(word);
                let DecodeResult::Decoded(decoded) = decode(platform, location, encoding) else {
                    panic!("{word:08x}")
                };
                let A64Instruction::FpSimd(fp_simd::Instruction::VectorFloatMultiply(fields)) =
                    normalize(&decoded.instruction, encoding)
                else {
                    panic!("{decoded:?}")
                };
                assert_eq!((fields.rd, fields.rn, fields.rm), (29, 29, 28));
                assert_eq!(fields.opc, u8::from(wide));
                assert_eq!(fields.vector_128, full);
            }
            // Reserved 1D and unsupported FP16/FMULX must not decode as FMUL S/D.
            for word in [0x2e7c_dfbd, 0x6e5c_1fbd, 0x4e3c_dfbd] {
                assert!(!matches!(
                    decode(platform, location, InstructionEncoding::from_u32(word)),
                    DecodeResult::Decoded(_)
                ));
            }
            assert!(matches!(
                crate::decode::allocation::validate_a64(
                    crate::coverage::CoverageId::new(0xa3),
                    0x2e7c_dfbd
                ),
                crate::decode::table::AllocationStatus::Reserved(_)
            ));
        }
    }

    #[test]
    fn vector_add_decodes_mesh_alias_and_rejects_reserved_shapes() {
        let platform = TargetPlatform::Switch1;
        let location =
            LocationDescriptor::new(GuestVirtualAddress::new(0x7100_51cc), platform.profile_id());
        for (wide, full) in [(false, false), (false, true), (true, true)] {
            for subtract in [false, true] {
                let word = 0x0e37_d7ff
                    | (u32::from(full) << 30)
                    | (u32::from(wide) << 22)
                    | (u32::from(subtract) << 23);
                let DecodeResult::Decoded(decoded) =
                    decode(platform, location, InstructionEncoding::from_u32(word))
                else {
                    panic!("{word:08x}")
                };
                let A64Instruction::FpSimd(fp_simd::Instruction::VectorFloatAdd(fields)) =
                    normalize(&decoded.instruction, InstructionEncoding::from_u32(word))
                else {
                    panic!("{decoded:?}")
                };
                assert_eq!((fields.rd, fields.rn, fields.rm), (31, 31, 23));
                assert_eq!(fields.opc, u8::from(wide));
                assert_eq!(fields.vector_128, full);
                assert_eq!(
                    fields.float_add_operation,
                    Some(if subtract {
                        fp_simd::FloatAddOperation::Subtract
                    } else {
                        fp_simd::FloatAddOperation::Add
                    })
                );
            }
        }
        // Q=0 with 64-bit lanes is reserved; half precision stays unsupported.
        for word in [0x0e77_d7ff, 0x0ef7_d7ff, 0x4e37_17ff] {
            assert!(
                !matches!(
                    decode(platform, location, InstructionEncoding::from_u32(word)),
                    DecodeResult::Decoded(_)
                ),
                "{word:08x}"
            );
        }
    }

    #[test]
    fn fused_vector_decodes_mesh_operands_and_rejects_reserved_shapes() {
        let platform = TargetPlatform::Switch1;
        let location =
            LocationDescriptor::new(GuestVirtualAddress::new(0x7100_51c4), platform.profile_id());
        for (wide, full) in [(false, false), (false, true), (true, true)] {
            for subtract in [false, true] {
                let word = 0x0e3c_cfbf
                    | (u32::from(full) << 30)
                    | (u32::from(wide) << 22)
                    | (u32::from(subtract) << 23);
                let DecodeResult::Decoded(decoded) =
                    decode(platform, location, InstructionEncoding::from_u32(word))
                else {
                    panic!("{word:08x}")
                };
                let A64Instruction::FpSimd(fp_simd::Instruction::VectorFloatFused(fields)) =
                    normalize(&decoded.instruction, InstructionEncoding::from_u32(word))
                else {
                    panic!("{decoded:?}")
                };
                assert_eq!((fields.rd, fields.rn, fields.rm), (31, 29, 28));
                assert_eq!(fields.opc, u8::from(wide));
                assert_eq!(fields.vector_128, full);
                assert_eq!(fields.subtract, subtract);
            }
        }
        for word in [0x0e7c_cfbf, 0x0efc_cfbf] {
            assert!(!matches!(
                decode(platform, location, InstructionEncoding::from_u32(word)),
                DecodeResult::Decoded(_)
            ));
            assert!(matches!(
                crate::decode::allocation::validate_a64(
                    crate::coverage::CoverageId::new(0xa4),
                    word
                ),
                crate::decode::table::AllocationStatus::Reserved(_)
            ));
        }
        // Half precision is optional and must not alias the binary32 pattern.
        assert!(!matches!(
            decode(
                platform,
                location,
                InstructionEncoding::from_u32(0x4e3c_0fbf)
            ),
            DecodeResult::Decoded(_)
        ));
    }

    #[test]
    fn fmla_element_decodes_cube_operands_and_rejects_reserved_double_shapes() {
        let platform = TargetPlatform::Switch1;
        let location =
            LocationDescriptor::new(GuestVirtualAddress::new(0x7100_2064), platform.profile_id());
        for (word, wide, full, lane, subtract) in [
            (0x4f99_12fb, false, true, 0, false),
            (0x0fb9_5afb, false, false, 3, true),
            (0x4fd9_1afb, true, true, 1, false),
        ] {
            let DecodeResult::Decoded(decoded) =
                decode(platform, location, InstructionEncoding::from_u32(word))
            else {
                panic!("{word:08x}")
            };
            let A64Instruction::FpSimd(fp_simd::Instruction::VectorFloatFusedElement(fields)) =
                normalize(&decoded.instruction, InstructionEncoding::from_u32(word))
            else {
                panic!("{decoded:?}")
            };
            assert_eq!((fields.rd, fields.rn, fields.rm), (27, 23, 25));
            assert_eq!(fields.opc, u8::from(wide));
            assert_eq!(fields.vector_128, full);
            assert_eq!(fields.fp_element_lane, lane);
            assert_eq!(fields.subtract, subtract);
        }
        for word in [0x0fd9_12fb, 0x4ff9_12fb] {
            let decoded = decode(platform, location, InstructionEncoding::from_u32(word));
            assert!(
                !matches!(decoded, DecodeResult::Decoded(_)),
                "{word:08x}: {decoded:?}"
            );
            assert!(matches!(
                crate::decode::allocation::validate_a64(
                    crate::coverage::CoverageId::new(0xa2),
                    word
                ),
                crate::decode::table::AllocationStatus::Reserved(_)
            ));
        }
        // Optional half-precision must not accidentally execute as binary32.
        assert!(!matches!(
            decode(
                platform,
                location,
                InstructionEncoding::from_u32(0x4f19_12fb)
            ),
            DecodeResult::Decoded(_)
        ));
    }
}
