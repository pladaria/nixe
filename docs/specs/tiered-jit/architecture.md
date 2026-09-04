# Architecture, objectives and invariants

[Specification index](README.md) · [Implementation tasks](tasks/README.md) · [Open review items](review-status.md)

Related contracts: [Supported baseline and generated manifest](baseline.md); [Policy defaults, layouts and safety bounds](policy-and-capacity.md); [Final conformance gate](conformance.md).

## Objective

The target is a correct, highly efficient JIT for commercial Nintendo Switch
software, with first-use latency and sustained execution competitive with the
best production emulators on x86-64 and AArch64. The priorities, in order, are:

1. Preserve exact guest semantics, native-fault retry, invalidation and FP
   state.
2. Minimize sustained hot-path work: guest branches should remain native,
   ordinary RAM accesses should remain ordinary host accesses, and compatible
   links should avoid the dispatcher and full architectural-state round trips.
3. Keep synchronous first-use compilation small and predictable.
4. Spend higher-quality compilation only on execution-proven hot code, without
   delaying the guest.
5. Bound and reclaim executable code and all metadata that shares its
   lifetime.
6. Retain only complexity which buys correctness or measurable execution,
   compilation or cache benefit.

Simplicity means removing accidental complexity. It does not mean preserving a
slower ABI, an append-only allocator or an inferior compiler boundary because
those are easier to implement.

## Scope and non-goals

This specification owns:

- the LCQ and HCQ unit shapes;
- the native gateway and fast-chain ABI;
- static linking, indirect dispatch and return prediction;
- functional hot-code selection;
- background compilation and duplicate-work arbitration;
- publication, version replacement and overlap policy;
- executable allocation, W^X, invalidation, retirement and reclamation;
- exact code/fault/dependency lifetime;
- x86-64 and AArch64 conformance; and
- migration and deletion of the superseded implementation.

The following are not part of this architecture:

- a persistent on-disk native-code cache;
- speculative value profiling, guards, deoptimization or trace trees;
- a title-specific policy database;
- an interpreter fallback hidden behind a JIT miss or compilation failure;
- exact retired-instruction observability;
- production compilation timers, overlap statistics, counters exported for
  diagnostics, benchmark modes or other unsolicited telemetry.

Tier-selection samples are functional compiler state, not observability. They
exist only to decide whether and what to compile, are updated only at an
already-required control poll, and are never logged or exposed.


## Architecture overview

~~~mermaid
flowchart TD
    G["system-ABI gateway<br/>once per native invocation"]
    N["bounded dispatch slot<br/>canonical authority"]
    L["LCQ basic block<br/>synchronous"]
    S["direct static chain"]
    D["per-vCPU PIC / RSB"]
    P["cold control poll<br/>functional sample"]
    Q["bounded HCQ queue"]
    W["parallel discovery and backend workers"]
    H["HCQ hot region<br/>real native entries"]
    C["segmented W^X code cache"]
    R["unlink + epoch retirement"]

    G --> N --> L
    L --> S --> L
    L --> D --> L
    L --> P --> Q --> W --> H --> C
    H --> S
    H --> D
    C --> R
    R --> N
~~~

The gateway establishes fast mode once. A native invocation may traverse many
LCQ and HCQ units without returning to Rust. Static links normally become
direct branches. Dynamic branches and returns use per-vCPU native caches.
Only a control-budget expiry, miss requiring compilation, architectural
boundary, unrecoverable fault or explicit runtime request leaves fast mode.

## Architectural invariants

1. One JIT guest-semantics and typed-helper implementation feeds both tiers.
   Tier-specific optimization may change representation but never semantics.
2. Each complete BlockKey has one current dispatch identity. Dispatch records
   are bounded and reclaimable; guest PC alone is never an identity.
3. The dispatch table is the authority and fallback; it is not a mandatory
   load on every resolved static edge.
4. Ordinary compatible cross-unit edges execute without Rust, a global lookup,
   a full A64State commit/reload, a system-ABI call or a native
   prologue/epilogue.
5. LCQ is always sufficient for correct progress. A vCPU never waits for HCQ
   discovery, compilation, queue capacity, allocator capacity or ownership.
   Like mapping changes, HCQ publication may request the common bounded
   maintenance rendezvous; that rendezvous is not an HCQ compile wait. HCQ never
   becomes a semantic fallback.
6. An active guest InstructionKey belongs to at most one HCQ family. LCQ may
   overlap it. Consecutive HCQ versions may coexist only while incoming roots
   are being cut over and the old version remains epoch-protected.
7. Generated code never queries HCQ ownership, code-cache capacity or
   dependency indexes.
8. Fast mapped RAM relies on the host mapping/protection contract defined here.
   No duplicate guest permission or ownership lookup is added to a normal
   memory access.
9. Code bytes, link bridges, dependency records, state maps and native-fault
   metadata share a versioned lifetime and cannot be reclaimed separately.
10. Unsupported guest semantics stop precisely. They do not silently enter the
    interpreter or an obsolete JIT.
