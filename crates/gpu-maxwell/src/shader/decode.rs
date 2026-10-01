//! Shared predicate, register-range, and temporary-register decoding rules.

use super::error::{MaxwellShaderTranslationError, malformed};
use crate::MaxwellShaderStage;
use nixe_gpu::{ShaderPredicate, ShaderRegister};

pub(super) const fn decode_predicate(encoding: u64) -> ShaderPredicate {
    decode_predicate_fields(encoding, 16, 19)
}

pub(super) const fn decode_predicate_fields(
    encoding: u64,
    register_bit: u32,
    inverted_bit: u32,
) -> ShaderPredicate {
    let register = ((encoding >> register_bit) & 0x7) as u8;
    let inverted = encoding & (1 << inverted_bit) != 0;
    if register == 7 {
        if inverted {
            ShaderPredicate::Never
        } else {
            ShaderPredicate::Always
        }
    } else {
        ShaderPredicate::Register { register, inverted }
    }
}

pub(super) fn allocate_shader_temporary(
    stage: MaxwellShaderStage,
    offset: u32,
    encoding: u64,
    detail: &'static str,
    next_temporary: &mut u16,
) -> Result<ShaderRegister, MaxwellShaderTranslationError> {
    if *next_temporary >= 256 {
        return Err(malformed(stage, offset, encoding, detail));
    }
    let register = ShaderRegister::new(*next_temporary);
    *next_temporary += 1;
    Ok(register)
}

pub(super) fn validate_register_range(
    stage: MaxwellShaderStage,
    offset: u32,
    encoding: u64,
    first: u8,
    count: u8,
    register_count: u8,
) -> Result<(), MaxwellShaderTranslationError> {
    if first == 0xff
        || first
            .checked_add(count)
            .is_none_or(|end| end > register_count)
    {
        return Err(malformed(
            stage,
            offset,
            encoding,
            "register range exceeds SET_PIPELINE_REGISTER_COUNT",
        ));
    }
    Ok(())
}
