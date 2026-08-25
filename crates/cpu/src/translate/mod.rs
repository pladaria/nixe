//! Translation from guest instructions to shared IR.

mod a32;
mod a64;
mod aarch32;
mod block;
pub mod observability;
pub mod region;
mod t32;

pub use observability::{
    RegionTranslationFailureReason, RegionTranslationReport, translate_raw_region_report,
    translate_region_report,
};
pub use region::{
    DEFAULT_MAX_BLOCKS_PER_REGION, DEFAULT_MAX_CODE_DEPENDENCIES_PER_REGION,
    DEFAULT_MAX_GUEST_BYTES_PER_REGION, DEFAULT_MAX_GUEST_INSTRUCTIONS_PER_BASIC_BLOCK,
    DEFAULT_MAX_GUEST_INSTRUCTIONS_PER_REGION, DEFAULT_MAX_IR_OPERATIONS_PER_REGION,
    MAX_BLOCKS_PER_REGION, MAX_CODE_DEPENDENCIES_PER_REGION, MAX_GUEST_BYTES_PER_REGION,
    MAX_GUEST_INSTRUCTIONS_PER_BASIC_BLOCK, MAX_GUEST_INSTRUCTIONS_PER_REGION,
    MAX_IR_OPERATIONS_PER_REGION, RegionTranslationConfig, translate_region,
};

#[cfg(test)]
mod normalization_tests {
    #[test]
    fn aarch32_lifters_cannot_decode_raw_instruction_bits() {
        let forbidden = concat!("encoding.", "bits()");
        for (state, source) in [
            ("A32", include_str!("a32.rs")),
            ("T32", include_str!("t32.rs")),
        ] {
            assert!(
                !source.contains(forbidden),
                "{state} lifter bypasses typed normalization"
            );
        }
    }
}
