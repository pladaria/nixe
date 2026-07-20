//! Translation from guest instructions to shared IR.

mod a32;
mod a64;
pub mod block;
mod t32;

pub use block::{
    AddressCalculationError, BlockTranslationConfig, a32_interworking_target,
    conditional_terminator, direct_branch_target, emit_call, indirect_interworking_target,
    indirect_target, translate_block,
};
