# Migration and removal map

[Specification index](README.md) · [Implementation tasks](tasks/README.md) · [Open review items](review-status.md)

Related contracts: [Architecture, objectives and invariants](architecture.md); [Final conformance gate](conformance.md).

## Migration map

The current source tree is an input to migration, not a compatibility target.
The completed architecture removes or replaces:

- multi-block breadth-first LCQ discovery in direct/region.rs;
- the 100-entry NativeLookupNode hotness counter and generated
  emit_promotion_check sequence;
- root-owned PublishedRegion HCQ publication;
- exact-root long HCQ compilation and its unbounded overlap;
- commit_state plus dispatch-cell load plus return_call_indirect on every static
  edge;
- generated traversal of the global bucket-chain for every dynamic edge;
- one append-only JITModule per compiler as native-code owner;
- logical native-byte accounting which cannot reclaim actual allocations;
- append-only retired code/fault metadata;
- COARSE_PROGRESS where it exists only to support the old unit boundary;
- test-only CLIF/native counters and production fields retained solely for
  implementation-detail assertions;
- old region-size, continuation, overlap and scheduler tests whose contract is
  no longer valid; and
- every adapter which keeps the old context-tail ABI callable after cutover.

The common decoder, exact instruction semantics, typed helper semantics,
dependency capture and verified FP behavior are reused where they satisfy this
specification. The memory/fault implementation is retained only to the extent
that it satisfies the [memory-authority](memory-authority.md) and [fault-retry](faults.md) contracts. These components may
be moved or reshaped, but they are not duplicated per tier or backend.
Per-entry retained-range and dependency copies are removed; exact code/mapping
dependencies remain once per CodeUnit because invalidation correctness requires
them.
