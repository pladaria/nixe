//! Executable transfer storage shared by static links and dynamic bridges.
//! The caller owns both immutable contracts throughout emission/installation.

use super::*;
use crate::executable::output::{Metadata, Output};
use nixe_memory::GuestVirtualAddress;

pub(super) enum Tail {
    StaticIsland,
    DynamicInline,
}

pub(super) fn install(
    process: &Lifetime,
    source: &CodeUnit,
    state_map: u32,
    target: &CodeUnit,
    target_entry: usize,
    tail_kind: Tail,
) -> Result<Option<Box<Installed>>, Error> {
    let entry = &target.entries[target_entry];
    let state = &source.states[state_map as usize];
    // Static BL reaches this bridge directly, bypassing its canonical fallback.
    // BLR pushes before its PIC probe; dynamic bridges must not push again.
    let mut transfer = if matches!(tail_kind, Tail::StaticIsland)
        && state.exit.is_some_and(|exit| exit.kind == EdgeKind::Call)
    {
        let continuation = source
            .instructions
            .get(0)
            .unwrap()
            .key
            .block_key()
            .at(GuestVirtualAddress::new(
                state.exit.unwrap().pc.get().wrapping_add(4),
            ))
            .ok_or(Error::InvalidUnit("unaligned call continuation"))?;
        crate::native::rsb::emit_push(&state.state, continuation)
            .map_err(|_| Error::InvalidUnit("cannot emit static call prediction"))?
    } else {
        Vec::new()
    };
    transfer.extend(
        crate::native::emit_chain_transfer(
            &source.states[state_map as usize].state,
            &entry.contract,
        )
        .map_err(|_| Error::InvalidUnit("cannot emit link state transfer"))?,
    );
    if transfer.is_empty() {
        return Ok(None);
    }
    let abi = source.code.metadata.abi;
    let (mut bytes, tail) = crate::native::link::bridge(abi, &transfer);
    // Nonfaulting canonical/fixed-frame storage only: no independent guest
    // instruction or fault table. A dynamic tail's worst-case bytes belong to
    // this span; only static transfers may request an out-of-line island.
    if matches!(tail_kind, Tail::DynamicInline) {
        let mut extended = bytes.into_vec();
        extended.resize(tail + crate::native::link::ISLAND_BYTES, 0);
        bytes = extended.into_boxed_slice();
    }
    let output = Output {
        bytes,
        alignment: 16,
        metadata: Metadata {
            abi,
            frame_extent: source
                .code
                .metadata
                .frame_extent
                .max(target.code.metadata.frame_extent),
            entries: Box::new([]),
            states: Box::new([]),
            faults: Box::new([]),
            traps: Box::new([]),
            relocations: Box::new([]),
        },
    };
    let address = target.code.allocation.address() + entry.fast_offset as usize;
    let installed = match tail_kind {
        Tail::StaticIsland => process
            .cache
            .install_with_branch(output, source.tier, tail, address),
        Tail::DynamicInline => {
            process
                .cache
                .install_with_inline_branch(output, source.tier, tail, address)
        }
    }?;
    Ok(Some(Box::new(installed)))
}
