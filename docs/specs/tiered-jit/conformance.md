# Final conformance gate

[Specification index](README.md) · [Implementation tasks](tasks/README.md) · [Open review items](review-status.md)

Related contracts: [Architecture, objectives and invariants](architecture.md); [External performance acceptance](performance.md).

## Final conformance gate

The architecture is complete only when all of these statements are true:

- LCQ is a synchronous single-entry straight-line fragment bounded by its
  first terminator or emergency cut; HCQ is asynchronous, execution-informed
  and never makes the guest wait for admission/compilation/resources; shared
  finite-work safety rendezvous remain permitted.
- A native invocation crosses compatible units without repeated gateway,
  system ABI, full-state commit/reload or dispatch-slot loads on resolved
  static edges and indirect-cache hits.
- Static links, indirect dispatch and matched returns remain native and have a
  safe unlink path.
- HCQ region shape is deterministic for one snapshot, has one total
  2048-instruction cap and can be reshaped instead of preserving the first
  root's partition forever.
- Unrelated active/in-flight HCQ owners share no InstructionKey even when their
  LCQ roots overlap; only cutover- and epoch-protected predecessor/successor
  versions coexist.
- HCQ entries are real native labels with minimal live-in adapters; there is no
  ordinal dispatcher.
- Code, dispatch records, dynamic bridges, dependencies, state maps, links and
  fault/directory metadata have one bounded, reclaimable version lifetime; no
  code is retired before synchronized removal of every executable root.
- Only ordinary unobserved tracked RAM retries an identical native PC; MMIO,
  observed-code writes and guest faults reconstruct the exact plan stage and
  finish canonically once. Memory invalidation and FP/NZCV behavior are
  identical in both tiers.
- OpenToken, active epochs, UnitPins, page COW tables and segment generations
  prevent every lookup/publication/fault/reuse race named here.
- The executable cache is W^X, uses its exact charged-total soft/hard gates and
  cannot leave stale native-PC metadata or writable published aliases after
  reuse.
- Native 4096-page Linux x86-64 and Linux AArch64 execution satisfy the same
  [Task 0](tasks/00-baseline.md) baseline and the frozen external performance thresholds.
- Production contains no unsolicited telemetry, per-entry promotion counter,
  ownership probe, benchmark machinery, silent interpreter fallback, obsolete
  JIT route or code retained solely for legacy tests.
