//! Maxwell exits, branches, and warp reconvergence targets.

use super::binary::{
    MAXWELL_INSTRUCTION_SIZE, MAXWELL_SCHEDULE_BUNDLE_SIZE, MAXWELL_SCHEDULE_CONTROL_SIZE,
};
use super::error::{MaxwellShaderTranslationError, malformed};
use crate::MaxwellShaderStage;
use nixe_gpu::ShaderSourceLocation;

pub(super) const fn is_loop_push(encoding: u64) -> bool {
    matches!(encoding >> 48, 0xe2a0 | 0xe2b0)
}

pub(super) const fn is_loop_branch(encoding: u64) -> bool {
    matches!(encoding >> 48, 0xe340 | 0xe350)
}

/// Resolve the per-invocation control stack along actual CFG edges. In
/// particular, a break cannot inherit a PBK from a branch it did not execute.
/// Different paths may share a token branch only when its destination agrees.
/// https://github.com/eden-emulator/mirror/blob/master/src/shader_recompiler/frontend/maxwell/control_flow.cpp (Stack::Pop)
pub(super) fn stack_branch_targets(
    binary: &super::binary::MaxwellShaderBinary,
) -> Result<std::collections::BTreeMap<u32, ShaderSourceLocation>, MaxwellShaderTranslationError> {
    use super::decode::decode_predicate;
    use nixe_gpu::ShaderPredicate;
    use std::collections::{BTreeMap, BTreeSet, VecDeque};
    let stage = binary.stage();
    let size = (binary.bundles().len() * MAXWELL_SCHEDULE_BUNDLE_SIZE) as u32;
    let mut code = Vec::new();
    'code: for bundle in binary.bundles() {
        for (slot, encoding) in bundle.instructions.iter().copied().enumerate() {
            code.push((bundle.offset + 8 + slot as u32 * 8, encoding));
            if is_exit(encoding) && decode_predicate(encoding) == ShaderPredicate::Always {
                break 'code;
            }
        }
    }
    if !code.iter().any(|(_, word)| {
        is_loop_push(*word)
            || is_loop_branch(*word)
            || is_set_sync_point(*word)
            || is_synchronize(*word)
    }) {
        return Ok(BTreeMap::new());
    }
    let locations: BTreeMap<_, _> = code
        .iter()
        .enumerate()
        .map(|(index, (offset, _))| (*offset, index))
        .collect();
    let mut pending = VecDeque::from([(0usize, Vec::<(u16, u32)>::new())]);
    let mut visited = BTreeSet::new();
    let mut targets = BTreeMap::new();
    while let Some((index, mut stack)) = pending.pop_front() {
        if index >= code.len() || !visited.insert((index, stack.clone())) {
            continue;
        }
        let (offset, encoding) = code[index];
        if stack.len() > 64 || visited.len() > code.len() * 64 {
            return Err(malformed(
                stage,
                offset,
                encoding,
                "shader control stack cannot be statically bounded",
            ));
        }
        let predicate = decode_predicate(encoding);
        if encoding >> 48 == 0xe250 {
            let table = binary
                .indirect_tables
                .iter()
                .find(|table| table.offset == offset)
                .ok_or_else(|| {
                    malformed(
                        stage,
                        offset,
                        encoding,
                        "BRX table targets were not captured",
                    )
                })?;
            for (_, target) in &table.targets {
                let destination = *locations.get(&target.byte_offset()).ok_or_else(|| {
                    malformed(
                        stage,
                        offset,
                        encoding,
                        "BRX target is outside executable code",
                    )
                })?;
                pending.push_back((destination, stack.clone()));
            }
            continue;
        }
        let mut taken_stack = stack.clone();
        let target = if is_set_sync_point(encoding) || is_loop_push(encoding) {
            if encoding & (1 << 5) != 0 {
                return Err(malformed(
                    stage,
                    offset,
                    encoding,
                    "constant-buffer control-stack target is not implemented",
                ));
            }
            stack.push((
                (encoding >> 48) as u16,
                decode_shader_control_target(stage, offset, encoding, size)?.byte_offset(),
            ));
            None
        } else if is_synchronize(encoding) || is_loop_branch(encoding) {
            if encoding & 0x1f != 0xf {
                return Err(malformed(
                    stage,
                    offset,
                    encoding,
                    "control-stack branch condition flags are not implemented",
                ));
            }
            if predicate == ShaderPredicate::Never {
                None
            } else {
                let token = match encoding >> 48 {
                    0xe340 => 0xe2a0,
                    0xe350 => 0xe2b0,
                    _ => 0xe290,
                };
                let position = stack
                    .iter()
                    .rposition(|&(kind, _)| kind == token)
                    .ok_or_else(|| {
                        malformed(
                            stage,
                            offset,
                            encoding,
                            "control-stack branch has no executed matching push",
                        )
                    })?;
                let target = stack[position].1;
                taken_stack.truncate(position);
                if targets
                    .insert(offset, ShaderSourceLocation::new(target))
                    .is_some_and(|old| old.byte_offset() != target)
                {
                    return Err(malformed(
                        stage,
                        offset,
                        encoding,
                        "path-dependent control-stack branch requires CFG cloning",
                    ));
                }
                Some(target)
            }
        } else if is_branch(encoding) && predicate != ShaderPredicate::Never {
            Some(decode_shader_control_target(stage, offset, encoding, size)?.byte_offset())
        } else {
            None
        };
        if let Some(target) = target {
            let destination = *locations.get(&target).ok_or_else(|| {
                malformed(
                    stage,
                    offset,
                    encoding,
                    "control-stack target is outside executable code",
                )
            })?;
            pending.push_back((destination, taken_stack));
        }
        if (!is_exit(encoding) && target.is_none()) || predicate != ShaderPredicate::Always {
            // Metadata pushes are unconditional; their low bits are not an
            // instruction predicate. A taken branch uses its popped stack,
            // while its fallthrough retains the original one.
            if !is_exit(encoding) || predicate != ShaderPredicate::Always {
                pending.push_back((index + 1, stack));
            }
        }
    }
    Ok(targets)
}

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
