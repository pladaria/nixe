//! Shared physical exit adapters and link/poll patches. The tier supplies the
//! completed prefix; this module never derives work from a region's word index.

use super::staging::append;
use super::*;
use crate::abi::{BlockKey, ValueLocation};
use crate::lifetime::unit::TerminalTransfer;
use crate::native::emit_canonical_exit;

pub(crate) struct Patch {
    map: StateMap,
    adapter: Vec<u8>,
    pc: ValueLocation,
    target: Option<BlockKey>,
    completed: u16,
    probe_bytes: usize,
    operation: Vec<u8>,
    slice_adapter: Option<Vec<u8>>,
}

pub(crate) fn prepare(
    abi: HostAbi,
    key: BlockKey,
    allocated: &AllocatedBoundary<'_>,
    pending: &PendingExit,
    site: ExitSiteKey,
    completed: u16,
) -> Result<(Patch, StateRecord), Error> {
    let map = allocated.map;
    if map.entry || map.poll.is_some_and(|poll| poll.completed != completed) {
        return Err(Error::internal(
            "native exit checkpoint cost/shape mismatch",
        ));
    }
    let key = key
        .at(pending.guest.pc)
        .ok_or_else(|| Error::internal("invalid native exit source key"))?;
    let state = pending
        .state
        .allocate(abi, site.source, site.state_map, allocated)?;
    let pc = allocated
        .location(pending.pc_operand, types::I64)
        .map_err(fail)?;
    // Dispatch checkpoints charge once before either patch. PRE exits
    // still charge their partial prefix in the canonical adapter.
    let uncharged = if map.poll.is_some() { 0 } else { completed };
    let indirect = pending.static_target.is_none()
        && matches!(
            pending.guest.kind,
            EdgeKind::Indirect | EdgeKind::Call | EdgeKind::Return
        );
    let mut adapter = if pending.static_target.is_some() || indirect {
        crate::native::emit_dispatch_fallback(&state, pc, uncharged)
    } else {
        emit_canonical_exit(&state, pc, pending.reason, uncharged)
    }
    .map_err(fail)?;
    let operation = match pending.guest.kind {
        EdgeKind::Call => crate::native::rsb::emit_push(
            &state,
            key.at(GuestVirtualAddress::new(
                pending.guest.pc.get().wrapping_add(4),
            ))
            .unwrap(),
        )
        .map_err(fail)?,
        EdgeKind::Return => {
            crate::native::rsb::emit_return_update(&state, key, pc).map_err(fail)?
        }
        _ => Vec::new(),
    };
    // An indirect hot path already updates the RSB before a PIC miss.
    // An exhausted poll bypasses that hot path and needs its own update.
    let slice_adapter = if indirect && !operation.is_empty() {
        let mut cold = operation.clone();
        cold.extend_from_slice(&adapter);
        Some(cold)
    } else {
        None
    };
    let probe_bytes = if indirect {
        if map.poll.is_none() {
            return Err(Error::internal(
                "native indirect probe has no charged terminal checkpoint",
            ));
        }
        let mut probe = if pending.guest.kind == EdgeKind::Return {
            crate::native::rsb::emit_return_probe(&state, key, pc).map_err(fail)?
        } else {
            let mut probe = operation.clone();
            probe.extend(crate::native::pic::probe::emit(&state, key, pc).map_err(fail)?);
            probe
        };
        let length = probe.len();
        probe.append(&mut adapter);
        adapter = probe;
        length
    } else {
        let mut prefix = operation.clone();
        prefix.append(&mut adapter);
        adapter = prefix;
        0
    };

    let target = pending
        .static_target
        .map(|pc| {
            key.at(pc)
                .ok_or_else(|| Error::internal("invalid native exit target key"))
        })
        .transpose()?;
    Ok((
        Patch {
            map: map.clone(),
            adapter,
            pc,
            target,
            completed,
            probe_bytes,
            operation,
            slice_adapter,
        },
        StateRecord {
            native_offset: map.offset,
            state,
            exit: Some(pending.guest),
            transfer: None,
        },
    ))
}

impl Patch {
    pub(crate) fn append(self, bytes: &mut Vec<u8>, record: &mut StateRecord) -> Result<(), Error> {
        let Self {
            map,
            adapter,
            pc,
            target,
            completed,
            probe_bytes,
            operation,
            slice_adapter,
        } = self;
        let destination = append(bytes, &adapter);
        map.patch_exit(bytes, 0, destination as u64).map_err(fail)?;
        // A slice exit must bypass the PIC even if its target is cached.
        // Sample-only polls instead resume the already-charged hot patch.
        let fallback = destination + probe_bytes;
        if map.poll.is_some() {
            let slice = slice_adapter.map_or(fallback, |adapter| append(bytes, &adapter));
            let mut control = operation;
            control.extend(
                emit_canonical_exit(&record.state, pc, NativeExitReason::Control, 0)
                    .map_err(fail)?,
            );
            let control = append(bytes, &control);
            let start = append_poll(
                bytes,
                &map,
                &record.state,
                pc,
                [map.offset, slice as u32, control as u32],
            )?;
            map.patch_poll(bytes, 0, start as u64).map_err(fail)?;
        }
        record.transfer = Some(Box::new(TerminalTransfer {
            destination: pc,
            static_target: target,
            completed,
            patch_bytes: map.patch_bytes,
            fallback_offset: fallback as u32,
            poll_offset: map.poll.map(|poll| poll.offset),
        }));

        Ok(())
    }
}

/// Shared cold control/sample path for external terminals and internal SSA
/// checks. Continuations are [resume, slice, control]; none charges work again.
pub(crate) fn append_poll(
    bytes: &mut Vec<u8>,
    map: &StateMap,
    state: &ExitStateMap,
    pc: ValueLocation,
    targets: [u32; 3],
) -> Result<usize, Error> {
    let [resume, slice, control] = targets;
    let (poll, branches) = crate::native::emit_poll(state.abi);
    let start = append(bytes, &poll);
    let mut branch = map.clone();
    branch.poll = None;
    // Unlike terminal transfers, an internal check can have live optimizer
    // temporaries outside the architectural map. Preserve all volatiles.
    let (observe, continuations) =
        crate::native::observation::emit_callback(state, pc, map.poll.is_none()).map_err(fail)?;
    let sample = append(bytes, &observe);
    for (offset, target) in continuations.into_iter().zip([resume, control]) {
        branch.offset = sample as u32 + offset;
        branch
            .patch_exit(bytes, 0, u64::from(target))
            .map_err(fail)?;
    }
    for (offset, target) in branches.into_iter().zip([sample as u32, slice, control]) {
        branch.offset = start as u32 + offset;
        branch
            .patch_exit(bytes, 0, u64::from(target))
            .map_err(fail)?;
    }
    Ok(start)
}
