//! Executable transfer storage shared by static links and dynamic bridges.
//! The caller owns both immutable contracts throughout emission/installation.

use super::*;
use crate::executable::output::{Metadata, Output};

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
    let _trace = nixe_trace::Span::new("cpu.bridge.emit_install", 0, 0);
    let entry = &target.entries[target_entry];
    let transfer = crate::native::emit_published_transfer(source, state_map, target, target_entry)
        .map_err(|_| Error::InvalidUnit("cannot emit link state transfer"))?;
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
            #[cfg(feature = "jit-profile")]
            regions: Box::new([]),
            #[cfg(feature = "jit-profile")]
            profile_body_length: 0,
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
    #[cfg(feature = "jit-profile")]
    let dynamic = matches!(tail_kind, Tail::DynamicInline);
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
    #[cfg(feature = "jit-profile")]
    crate::profiling::bridge(&installed, source, target, process.identity, dynamic);
    Ok(Some(Box::new(installed)))
}
