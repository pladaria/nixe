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
    poll_targets: Option<[usize; 2]>,
    observe: bool,
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
    if map.entry
        || map.poll.is_some_and(|poll| {
            poll.completed != completed || pending.reason != NativeExitReason::Dispatch
        })
    {
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
    let (mut adapter, mut poll_targets) = if map.poll.is_some() {
        let (code, targets) = crate::native::emit_polled_exit(&state, pc).map_err(fail)?;
        (code, Some(targets))
    } else {
        let adapter = if pending.static_target.is_some() || indirect {
            crate::native::emit_dispatch_fallback(&state, pc, uncharged)
        } else {
            emit_canonical_exit(&state, pc, pending.reason, uncharged)
        }
        .map_err(fail)?;
        (adapter, None)
    };
    let probe_bytes = if indirect {
        if map.poll.is_none() {
            return Err(Error::internal(
                "native indirect probe has no charged terminal checkpoint",
            ));
        }
        // RET uses its architectural target just like BR/BLR. The PIC checks
        // the source contract and destination PC before entering owned code.
        let mut probe = crate::native::pic::probe::emit(&state, pc).map_err(fail)?;
        let length = probe.len();
        poll_targets = poll_targets.map(|offsets| offsets.map(|offset| length + offset));
        probe.append(&mut adapter);
        adapter = probe;
        length
    } else {
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
            poll_targets,
            observe: pending.guest.kind != EdgeKind::Return,
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
            poll_targets,
            observe,
        } = self;
        let destination = append(bytes, &adapter);
        map.patch_exit(bytes, 0, destination as u64).map_err(fail)?;
        // A slice exit must bypass the PIC even if its target is cached.
        // Sample-only polls instead resume the already-charged hot patch.
        let fallback = destination + probe_bytes;
        if let Some([slice, control]) = poll_targets {
            let start = append_poll(
                bytes,
                &map,
                &record.state,
                pc,
                [
                    map.offset,
                    (destination + slice) as u32,
                    (destination + control) as u32,
                ],
                observe,
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
    observe: bool,
) -> Result<usize, Error> {
    let [resume, slice, control] = targets;
    let (poll, branches) = crate::native::emit_poll(state);
    let start = append(bytes, &poll);
    let mut branch = map.clone();
    branch.poll = None;
    let sample = if observe {
        let (callback, continuations) =
            crate::native::observation::emit_callback(state, pc).map_err(fail)?;
        let sample = append(bytes, &callback) as u32;
        for (offset, target) in continuations.into_iter().zip([resume, control]) {
            branch.offset = sample + offset;
            branch
                .patch_exit(bytes, 0, u64::from(target))
                .map_err(fail)?;
        }
        sample
    } else {
        // Internal cycles and external return fallbacks need no growth
        // observation. Keep the native control/slice poll and resume
        // directly, preserving every mapped register and the FP environment.
        resume
    };
    for (offset, target) in branches.into_iter().zip([sample, slice, control]) {
        branch.offset = start as u32 + offset;
        branch
            .patch_exit(bytes, 0, u64::from(target))
            .map_err(fail)?;
    }
    Ok(start)
}
