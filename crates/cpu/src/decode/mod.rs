//! Declarative, profile-aware instruction decoding.

pub mod a32;
pub mod a64;
pub mod aarch32;
pub mod t32;
pub mod table;

use core::fmt;

use crate::{
    location::{DecodedInstruction, ExecutionState, InstructionEncoding, LocationDescriptor},
    profile::GuestCpuProfile,
};

pub use table::{
    DecodeSupport, DecodedOpcode, DecodedOperands, InstructionPattern, OperandField, OperandId,
    OperandKind, OperandValue, RegisterClass, ReservedConstraint,
};

/// Exhaustive architectural classification returned by the decoder layer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DecodeResult {
    /// The encoding and operands are ready for a lifter or interpreter.
    Decoded(DecodedInstruction<DecodedOpcode>),
    /// The encoding belongs to no allocated entry in the current table.
    Unallocated {
        instruction: crate::error::InstructionDiagnostic,
        reason: &'static str,
    },
    /// A family matched, but an architectural reserved-bit rule failed.
    Reserved {
        instruction: crate::error::InstructionDiagnostic,
        name: &'static str,
        reason: &'static str,
    },
    /// The profile does not enable a required architectural feature.
    ProfileDisabled {
        instruction: crate::error::InstructionDiagnostic,
        name: &'static str,
        rejection: crate::profile::InstructionFeatureRejection,
    },
    /// The architecture and operands are known, but semantics are not present.
    RecognizedUnimplemented(DecodedInstruction<DecodedOpcode>),
}

/// Decodes one canonical instruction according to its execution state.
#[must_use]
pub fn decode(
    profile: &GuestCpuProfile,
    location: LocationDescriptor,
    encoding: InstructionEncoding,
) -> DecodeResult {
    if location.profile_id != profile.id() {
        return DecodeResult::Unallocated {
            instruction: crate::error::InstructionDiagnostic::new(location, encoding),
            reason: "location profile does not match decoder profile",
        };
    }
    if !profile
        .allowed_execution_states()
        .contains(location.execution_state)
    {
        return DecodeResult::Unallocated {
            instruction: crate::error::InstructionDiagnostic::new(location, encoding),
            reason: "execution state is unavailable in this profile",
        };
    }

    match location.execution_state {
        ExecutionState::A64 => a64::decode(profile, location, encoding),
        ExecutionState::A32 => a32::decode(profile, location, encoding),
        ExecutionState::T32 => t32::decode(profile, location, encoding),
    }
}

/// Display adapter for disassembly, deliberately independent of IR lifting.
pub struct Disassembly<'a>(&'a DecodedOpcode);

impl fmt::Display for Disassembly<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt_disassembly(f)
    }
}

/// Formats a decoded opcode without invoking or depending on a lifter.
#[must_use]
pub const fn disassemble(opcode: &DecodedOpcode) -> Disassembly<'_> {
    Disassembly(opcode)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::{
        address::GuestVirtualAddress,
        coverage::CoverageId,
        location::InstructionSize,
        profile::{CapabilityStatus, InstructionFeature},
    };

    fn location(profile: GuestCpuProfile, state: ExecutionState) -> LocationDescriptor {
        LocationDescriptor::new(GuestVirtualAddress::new(0x1000), state, profile.id())
    }

    fn opcode(result: DecodeResult) -> DecodedInstruction<DecodedOpcode> {
        match result {
            DecodeResult::Decoded(decoded) | DecodeResult::RecognizedUnimplemented(decoded) => {
                decoded
            }
            other => panic!("expected recognized instruction, got {other:?}"),
        }
    }

    // Encoding families and instruction semantics are traced to Arm DDI 0602
    // (A64) and Arm DDI 0597 (A32/T32). See `crates/cpu/tests/README.md`.
    // These are independent expectations, not output copied from an assembler.
    #[test]
    fn raw_encoding_goldens_cover_every_decoder_family() {
        let profile = GuestCpuProfile::switch_1();
        let cases = [
            (ExecutionState::A64, 0x1400_0000_u32.into(), "b"),
            (ExecutionState::A64, 0xd503_3bbf_u32.into(), "barrier"),
            (
                ExecutionState::A64,
                0x9100_4423_u32.into(),
                "add-sub-immediate",
            ),
            (
                ExecutionState::A64,
                0xf940_0020_u32.into(),
                "load-store-unsigned",
            ),
            (ExecutionState::A64, 0x4e20_1c00_u32.into(), "simd-bitwise"),
            (ExecutionState::A32, 0xeaff_ffff_u32.into(), "b"),
            (
                ExecutionState::A32,
                0xe3a0_0001_u32.into(),
                "data-processing",
            ),
            (
                ExecutionState::A32,
                0xe590_1000_u32.into(),
                "load-store-single",
            ),
            (ExecutionState::A32, 0xf200_0110_u32.into(), "neon-bitwise"),
            (ExecutionState::T32, 0xe7ff_u16.into(), "b"),
            (ExecutionState::T32, 0x237f_u16.into(), "movs"),
            (ExecutionState::T32, 0x4801_u16.into(), "load-literal"),
        ];

        for (state, encoding, expected_name) in cases {
            let decoded = opcode(decode(&profile, location(profile, state), encoding));
            assert_eq!(
                decoded.encoding, encoding,
                "decoder did not preserve raw encoding for {state} {encoding}"
            );
            assert_eq!(
                decoded.instruction.pattern().name,
                expected_name,
                "raw golden mismatch for {state} {encoding}"
            );
        }
    }

    #[test]
    fn representative_encodings_have_stable_golden_disassembly() {
        let profile = GuestCpuProfile::switch_1();
        let cases = [
            (
                ExecutionState::A64,
                InstructionEncoding::from_u32(0xd503_201f),
                "nop",
                CoverageId::new(0x0000_0001),
            ),
            (
                ExecutionState::A64,
                InstructionEncoding::from_u32(0x17ff_ffff),
                "b imm=#-4",
                CoverageId::new(0x0000_0002),
            ),
            (
                ExecutionState::A64,
                InstructionEncoding::from_u32(0xd503_3bbf),
                "barrier",
                CoverageId::new(0x0000_000e),
            ),
            (
                ExecutionState::A64,
                InstructionEncoding::from_u32(0x9100_4423),
                "add-sub-immediate",
                CoverageId::new(0x0000_0003),
            ),
            (
                ExecutionState::A64,
                InstructionEncoding::from_u32(0xf940_0020),
                "load-store-unsigned",
                CoverageId::new(0x0000_0023),
            ),
            (
                ExecutionState::A64,
                InstructionEncoding::from_u32(0x4e20_1c00),
                "simd-bitwise",
                CoverageId::new(0x0000_0030),
            ),
            (
                ExecutionState::A32,
                InstructionEncoding::from_u32(0xeaff_ffff),
                "b imm=#-4, cond=#14",
                CoverageId::new(0x0001_0002),
            ),
            (
                ExecutionState::A32,
                InstructionEncoding::from_u32(0xe3a0_0001),
                "data-processing",
                CoverageId::new(0x0001_0010),
            ),
            (
                ExecutionState::A32,
                InstructionEncoding::from_u32(0xe590_1000),
                "load-store-single",
                CoverageId::new(0x0001_0020),
            ),
            (
                ExecutionState::A32,
                InstructionEncoding::from_u32(0xf200_0110),
                "neon-bitwise",
                CoverageId::new(0x0001_0031),
            ),
            (
                ExecutionState::T32,
                InstructionEncoding::from_u16(0xe7ff),
                "b imm=#-2",
                CoverageId::new(0x0002_0002),
            ),
            (
                ExecutionState::T32,
                InstructionEncoding::from_u16(0x237f),
                "movs dst=r3, imm=#127",
                CoverageId::new(0x0002_0003),
            ),
            (
                ExecutionState::T32,
                InstructionEncoding::from_u16(0x4801),
                "load-literal",
                CoverageId::new(0x0002_0020),
            ),
            (
                ExecutionState::T32,
                InstructionEncoding::from_u32(0xf3af_8000),
                "nop.w",
                CoverageId::new(0x0002_0004),
            ),
        ];

        for (state, encoding, expected_text, expected_coverage) in cases {
            let decoded = opcode(decode(&profile, location(profile, state), encoding));
            assert_eq!(disassemble(&decoded.instruction).to_string(), expected_text);
            assert_eq!(decoded.instruction.coverage_id(), expected_coverage);
        }
    }

    #[test]
    fn all_shipped_tables_are_consistent_and_ids_are_globally_unique() {
        let tables = [
            a64::patterns(),
            a32::patterns(),
            t32::patterns_16(),
            t32::patterns_32(),
        ];
        let mut coverage = BTreeSet::new();
        let mut semantics = BTreeSet::new();
        for patterns in tables {
            let table = table::DecoderTable::compile(patterns).expect("consistent table");
            assert!(table.candidate_count(0) <= patterns.len());
            for pattern in patterns {
                assert!(
                    coverage.insert(pattern.coverage_id),
                    "duplicate coverage ID"
                );
                assert!(
                    semantics.insert(pattern.semantic_id),
                    "duplicate semantic ID"
                );
            }
        }
    }

    #[test]
    fn classifications_are_exhaustive_and_distinct() {
        let profile = GuestCpuProfile::switch_1();
        assert!(matches!(
            decode(
                &profile,
                location(profile, ExecutionState::A64),
                0xd503_201f_u32.into()
            ),
            DecodeResult::Decoded(_)
        ));
        assert!(matches!(
            decode(
                &profile,
                location(profile, ExecutionState::A64),
                0x1400_0000_u32.into()
            ),
            DecodeResult::Decoded(_)
        ));
        assert!(matches!(
            decode(
                &profile,
                location(profile, ExecutionState::A64),
                0_u32.into()
            ),
            DecodeResult::Unallocated { .. }
        ));
    }

    // Arm DDI 0487 defines feature-dependent instruction availability. This
    // test exercises the actual decoder gate rather than only the profile API.
    #[test]
    fn decoder_profile_feature_goldens_cover_enabled_disabled_and_unknown() {
        let enabled = GuestCpuProfile::switch_1();
        let disabled = enabled
            .with_instruction_feature(InstructionFeature::AdvancedSimd, CapabilityStatus::Disabled);
        let unknown = GuestCpuProfile::switch_2_native();
        let encoding = InstructionEncoding::from_u32(0x4e20_1c00); // AND V0.16B,V0.16B,V0.16B

        assert!(matches!(
            decode(&enabled, location(enabled, ExecutionState::A64), encoding),
            DecodeResult::Decoded(_)
        ));
        assert!(matches!(
            decode(
                &disabled,
                location(disabled, ExecutionState::A64),
                encoding
            ),
            DecodeResult::ProfileDisabled { rejection, .. }
                if rejection.feature == InstructionFeature::AdvancedSimd
                    && rejection.status == CapabilityStatus::Disabled
        ));
        assert!(matches!(
            decode(
                &unknown,
                location(unknown, ExecutionState::A64),
                encoding
            ),
            DecodeResult::ProfileDisabled { rejection, .. }
                if rejection.feature == InstructionFeature::AdvancedSimd
                    && rejection.status == CapabilityStatus::Unknown
        ));

        // Use a conditional-space VFP encoding so feature rejection remains
        // independently observable from A32 unconditional-space allocation.
        let a32_simd = InstructionEncoding::from_u32(0xee00_0a00);
        assert!(matches!(
            decode(&enabled, location(enabled, ExecutionState::A32), a32_simd),
            DecodeResult::RecognizedUnimplemented(_)
        ));
        assert!(matches!(
            decode(&disabled, location(disabled, ExecutionState::A32), a32_simd),
            DecodeResult::ProfileDisabled { .. }
        ));

        assert!(matches!(
            decode(&unknown, location(unknown, ExecutionState::A32), a32_simd),
            DecodeResult::Unallocated { .. }
        ));
    }

    #[test]
    fn arbitrary_encodings_are_total_and_keep_valid_fixed_operand_storage() {
        let profile = GuestCpuProfile::switch_1();

        for bits in u16::MIN..=u16::MAX {
            check_result(decode(
                &profile,
                location(profile, ExecutionState::T32),
                InstructionEncoding::from_u16(bits),
            ));
        }

        let mut bits = 0x6d2b_79f5_u32;
        for _ in 0..200_000 {
            bits ^= bits << 13;
            bits ^= bits >> 17;
            bits ^= bits << 5;
            check_result(decode(
                &profile,
                location(profile, ExecutionState::A64),
                InstructionEncoding::from_u32(bits),
            ));
            check_result(decode(
                &profile,
                location(profile, ExecutionState::A32),
                InstructionEncoding::from_u32(bits),
            ));
            if crate::location::is_t32_32_bit_prefix((bits >> 16) as u16) {
                check_result(decode(
                    &profile,
                    location(profile, ExecutionState::T32),
                    InstructionEncoding::from_u32(bits),
                ));
            }
        }
    }

    fn check_result(result: DecodeResult) {
        let decoded = match result {
            DecodeResult::Decoded(decoded) | DecodeResult::RecognizedUnimplemented(decoded) => {
                decoded
            }
            DecodeResult::Unallocated { .. }
            | DecodeResult::Reserved { .. }
            | DecodeResult::ProfileDisabled { .. } => return,
        };
        assert!(decoded.instruction.operands().len() <= 8);
        assert_eq!(decoded.encoding.size(), decoded.instruction.pattern().size);
        assert_eq!(
            decoded.location.execution_state,
            decoded.instruction.pattern().execution_state
        );
        for (_, operand) in decoded.instruction.operands().iter() {
            if let OperandValue::Register { index, .. } = operand {
                assert!(index < 32);
            }
        }
    }

    #[test]
    fn wrong_width_is_classified_without_reading_invalid_operands() {
        let profile = GuestCpuProfile::switch_1();
        let result = decode(
            &profile,
            location(profile, ExecutionState::A64),
            InstructionEncoding::from_u16(0x201f),
        );
        assert!(matches!(result, DecodeResult::Unallocated { .. }));
        assert_eq!(InstructionSize::Bits16.bytes(), 2);
    }
}
