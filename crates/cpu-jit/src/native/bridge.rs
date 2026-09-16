//! Architectural-state bridge between independently allocated native units.
//! Only emission walks maps; the executed bridge is straight-line native code.
use super::moves::{Copy, Emitter};
use super::{TransferError, canonical, emit_copies, emit_fast_transfer, flags};
use crate::abi::{EntryContract, ExitStateMap, NzcvLocation, ValueLocation};

/// Preserve dirty values absent from the target in canonical homes, then install
/// its physical inputs. Missing source bindings are clean canonical inputs.
/// The target must treat writable inputs as potentially dirty at observations.
/// No PC store, call, FP transition, epoch change or runtime dirty-mask walk.
/// Pending host FPSR remains owned by the invocation, including when no target
/// binding names FPSR. The caller owns the final branch and both code lifetimes.
pub fn emit_chain_transfer(
    source: &ExitStateMap,
    target: &EntryContract,
) -> Result<Vec<u8>, TransferError> {
    source.validate().map_err(TransferError::InvalidContract)?;
    target.validate().map_err(TransferError::InvalidContract)?;
    if source.abi != target.abi {
        return Err(TransferError::DifferentHostAbis);
    }
    let mut commit = source.clone();
    commit.dirty_live = source.dirty_live.without(target.live_in);
    let mut copies = Vec::new();
    let mut missing = Vec::new();
    for binding in &target.bindings {
        if let Some(input) = source
            .bindings
            .iter()
            .find(|input| input.value == binding.value)
        {
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
        return emit_fast_transfer(source, target);
    }
    let mut emitter = Emitter::new(source.abi);
    let required = commit.dirty_live.nzcv | (target.live_in.nzcv & source.live.nzcv);
    // Capture host flags and deferred operands before writeback/copies can
    // clobber them. Packed values can be used directly unless canonical bits
    // must be merged; this avoids materialization on ordinary packed bridges.
    let packed = if required != 0 || missing_flags != 0 {
        if missing_flags == 0
            && let NzcvLocation::Packed(location) = source.nzcv
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
    canonical::writeback(&mut emitter, &commit, &nzcv);
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
    canonical::load_missing(&mut emitter, &missing);
    if host_target && let NzcvLocation::Host { carry_inverted } = target.nzcv {
        flags::install_host(&mut emitter, carry_inverted);
    }
    Ok(emitter.finish())
}
