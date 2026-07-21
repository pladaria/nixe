//! Translation from guest instructions to shared IR.

mod a32;
mod a64;
pub mod block;
mod t32;

pub use block::{
    AddressCalculationError, BlockTranslationConfig, MAX_GUEST_INSTRUCTIONS_PER_BLOCK,
    MAX_IR_OPERATIONS_PER_GUEST_INSTRUCTION, a32_interworking_target, conditional_terminator,
    direct_branch_target, emit_call, indirect_interworking_target, indirect_target,
    translate_block,
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
