//! Architectural-state bridge between independently allocated native units.
//! Only emission walks maps; the executed bridge is straight-line native code.
use super::moves::{Copy, Emitter};
use super::{TransferError, canonical, emit_copies, emit_fast_transfer_validated, flags};
use crate::abi::{EntryContract, ExitStateMap, LazyFlags, NzcvLocation, ValueLocation};

/// Published units are immutable and their complete contracts were checked
/// before directory exposure. Dynamic admission/version checks remain with the
/// lifetime owner; construction here does not repeat exhaustive map validation.
pub(crate) fn emit_published_transfer(
    source: &crate::lifetime::unit::CodeUnit,
    state_map: u32,
    target: &crate::lifetime::unit::CodeUnit,
    entry: usize,
) -> Result<Vec<u8>, TransferError> {
    emit_chain_transfer_validated(
        &source.states[state_map as usize].state,
        &target.entries[entry].contract,
    )
}

#[cfg(test)]
pub(crate) fn emit_chain_transfer(
    source: &ExitStateMap,
    target: &EntryContract,
) -> Result<Vec<u8>, TransferError> {
    source.validate().map_err(TransferError::InvalidContract)?;
    target.validate().map_err(TransferError::InvalidContract)?;
    emit_chain_transfer_validated(source, target)
}

/// Preserve dirty values absent from the target in canonical homes, then install
/// its physical inputs. Missing source bindings are clean canonical inputs.
/// The target must treat writable inputs as potentially dirty at observations.
/// No PC store, call, FP transition, epoch change or runtime dirty-mask walk.
/// Pending host FPSR remains owned by the invocation, including when no target
/// binding names FPSR. The caller owns the final branch and both code lifetimes.
fn emit_chain_transfer_validated(
    source: &ExitStateMap,
    target: &EntryContract,
) -> Result<Vec<u8>, TransferError> {
    if source.abi != target.abi {
        return Err(TransferError::DifferentHostAbis);
    }
    let mut commit = source.clone();
    commit.dirty_live = source
        .dirty_live
        .without(target.live_in.union(target.discard));
    let mut copies = Vec::new();
    let mut missing = Vec::new();
    for binding in target.bindings.iter() {
        if let Some(input) = source.bindings.get_value(binding.value) {
            copies.push(Copy {
                source: input.location,
                destination: binding.location,
                bytes: binding.value.bytes(),
            });
        } else {
            missing.push(*binding);
        }
    }
    // An unbound pending host FPSR is not a software store. Neither entry nor
    // exit should end guest FP ownership merely because the next unit is integer.
    let stores_data = source.bindings.iter().any(|b| {
        !commit
            .dirty_live
            .intersection(b.value.state().unwrap())
            .is_empty()
    });
    let missing_flags = target.live_in.nzcv & !source.live.nzcv;
    if !stores_data && commit.dirty_live.nzcv == 0 && missing.is_empty() && missing_flags == 0 {
        return emit_fast_transfer_validated(source, target);
    }
    let mut emitter = Emitter::new(source.abi);
    let required = commit.dirty_live.nzcv | (target.live_in.nzcv & source.live.nzcv);
    // Capture host flags and deferred operands before writeback/copies can
    // clobber them. Packed values can be used directly unless canonical bits
    // must be merged; this avoids materialization on ordinary packed bridges.
    let packed = if required != 0 || missing_flags != 0 {
        if missing_flags == 0
            && let NzcvLocation::Packed(location)
            | NzcvLocation::Deferred(
                LazyFlags::Packed(location) | LazyFlags::Canonical(location),
            ) = source.nzcv
        {
            Some(location)
        } else {
            if required == 0 {
                flags::materialize(&mut emitter, &NzcvLocation::Canonical, target.live_in.nzcv);
            } else {
                flags::materialize(&mut emitter, &source.nzcv, required);
                if missing_flags != 0 {
                    canonical::fill_missing_flags(&mut emitter, required);
                }
            }
            Some(ValueLocation::Spill {
                offset: flags::RESULT,
                bytes: 4,
            })
        }
    } else {
        None
    };
    let nzcv = packed
        .map(NzcvLocation::Packed)
        .unwrap_or_else(|| source.nzcv.clone());
    let source_scratch = free_integer(
        source.abi,
        source.bindings.iter().map(|b| b.location),
        &nzcv,
    );
    canonical::writeback_with_scratch(&mut emitter, &commit, &nzcv, source_scratch);
    let host_target = if target.live_in.nzcv != 0 {
        let input = packed.unwrap();
        let destination = match target.nzcv {
            NzcvLocation::Packed(location) => location,
            NzcvLocation::Host { .. } => ValueLocation::Spill {
                offset: flags::RESULT,
                bytes: 4,
            },
            _ => unreachable!("validated entry"),
        };
        copies.push(Copy {
            source: input,
            destination,
            bytes: 4,
        });
        matches!(target.nzcv, NzcvLocation::Host { .. })
    } else {
        false
    };
    emit_copies(&mut emitter, source.abi, copies)?;
    let target_scratch = free_integer(
        target.abi,
        target.bindings.iter().map(|b| b.location),
        &target.nzcv,
    );
    canonical::load_missing(&mut emitter, &missing, target_scratch);
    if host_target && let NzcvLocation::Host { carry_inverted } = target.nzcv {
        flags::install_host(&mut emitter, carry_inverted);
    }
    Ok(emitter.finish())
}

/// Choose scratch at emission time, never by inspecting runtime state. These
/// are allocatable x86 GPRs; reserved frame, poll, arena, link and stack registers
/// are excluded. Protect packed flags as well as clean and dirty data bindings.
/// The existing save/restore adapter handles contracts occupying every GPR.
fn free_integer(
    abi: crate::abi::HostAbi,
    locations: impl Iterator<Item = ValueLocation>,
    nzcv: &NzcvLocation,
) -> Option<u8> {
    if abi != crate::abi::HostAbi::X86_64 {
        return None;
    }
    let mut occupied = 0u16;
    for location in locations.chain(match nzcv {
        NzcvLocation::Packed(location) => Some(*location),
        _ => None,
    }) {
        if let ValueLocation::Register {
            class: crate::abi::RegisterClass::Integer,
            index,
        } = location
        {
            occupied |= 1 << index;
        }
    }
    (0..16).find(|index| {
        occupied & (1 << index) == 0
            && ValueLocation::Register {
                class: crate::abi::RegisterClass::Integer,
                index: *index,
            }
            .valid(abi, 8)
    })
}
