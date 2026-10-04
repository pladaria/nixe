//! Maxwell exits, branches, and warp reconvergence targets.

use super::binary::{
    MAXWELL_INSTRUCTION_SIZE, MAXWELL_SCHEDULE_BUNDLE_SIZE, MAXWELL_SCHEDULE_CONTROL_SIZE,
};
use super::error::{MaxwellShaderTranslationError, malformed};
use crate::MaxwellShaderStage;
use nixe_gpu::ShaderSourceLocation;

pub(super) const fn is_exit(encoding: u64) -> bool {
    encoding >> 48 == 0xe300
}

pub(super) const fn is_set_sync_point(encoding: u64) -> bool {
    encoding >> 48 == 0xe290
}

pub(super) const fn is_branch(encoding: u64) -> bool {
    encoding >> 48 == 0xe240
}

pub(super) const fn is_synchronize(encoding: u64) -> bool {
    encoding >> 48 == 0xf0f8
}

pub(super) const fn is_dependency_barrier(encoding: u64) -> bool {
    encoding >> 48 == 0xf0f0
}

pub(super) fn validate_dependency_barrier(
    stage: MaxwellShaderStage,
    offset: u32,
    encoding: u64,
) -> Result<(), MaxwellShaderTranslationError> {
    // DEPBAR waits for the issuing warp's asynchronous register dependencies,
    // not for other invocations or for memory visibility. Our IR operations
    // produce their results synchronously and retain program order, so no
    // backend barrier is required. Retain its location for branch targets.
    // Maxwell field layout: devkitPro UAM's emitDEPBAR:
    // https://github.com/devkitPro/uam/blob/master/mesa-imported/codegen/nv50_ir_emit_gm107.cpp
    const FIELDS: u64 = 0xffff_0000_0000_0000 | (0x3ff << 20) | (0xf << 16) | 0x3f;
    if encoding & !FIELDS != 0 || (encoding >> 26) & 7 >= 6 {
        return Err(malformed(stage, offset, encoding, "invalid DEPBAR fields"));
    }
    Ok(())
}

pub(super) fn decode_shader_control_target(
    stage: MaxwellShaderStage,
    offset: u32,
    encoding: u64,
    code_size: u32,
) -> Result<ShaderSourceLocation, MaxwellShaderTranslationError> {
    // SM50 control flow stores a signed 24-bit byte displacement relative to
    // the following instruction. Field placement and PC bias follow Mesa
    // NAK's pinned set_rel_offset and SSY encoder:
    // https://gitlab.freedesktop.org/mesa/mesa/-/blob/2c9073912232b93eb9b60486edbd72d53e5f3d26/src/nouveau/compiler/nak/sm50.rs#L3007-L3041
    let raw = ((encoding >> 20) & 0x00ff_ffff) as i32;
    let displacement = (raw << 8) >> 8;
    let target =
        i64::from(offset) + i64::from(MAXWELL_INSTRUCTION_SIZE as u32) + i64::from(displacement);
    let target = u32::try_from(target).map_err(|_| {
        malformed(
            stage,
            offset,
            encoding,
            "shader control target lies outside the bounded program",
        )
    })?;
    let target = if target.is_multiple_of(MAXWELL_SCHEDULE_BUNDLE_SIZE as u32) {
        target
            .checked_add(MAXWELL_SCHEDULE_CONTROL_SIZE as u32)
            .ok_or_else(|| {
                malformed(
                    stage,
                    offset,
                    encoding,
                    "shader control target overflows after bundle normalization",
                )
            })?
    } else {
        target
    };
    let bundle_offset = target % MAXWELL_SCHEDULE_BUNDLE_SIZE as u32;
    if target >= code_size || !matches!(bundle_offset, 8 | 16 | 24) {
        return Err(malformed(
            stage,
            offset,
            encoding,
            "shader control target is not an executable instruction slot",
        ));
    }
    Ok(ShaderSourceLocation::new(target))
}

#[cfg(test)]
mod tests;
