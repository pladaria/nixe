# Nixe tiered JIT architecture

Review status: **draft; technical review is still open**. The document split
and stable-release selection do not certify implementation readiness. Known
contradictions and unproved gates are recorded in [review status](review-status.md).

## Status and authority

This directory is one specification set and implementation sequence for Nixe's
in-memory JIT architecture. Its chapters define shared contracts; its task
files define the sequential implementation work. Read it with the repository
[CONTRIBUTING.md](../../../CONTRIBUTING.md), which governs all work. The current
code remains authoritative for current behavior; this set defines the proposed
target architecture. Other documents may link here but do not amend these
contracts or task gates.
References to the current source tree describe migration input, not immutable
architecture.

The implementation may replace crate boundaries, runtime data structures,
Cranelift integration, generated-code ABI, executable allocation, dispatch,
linking, tier selection and invalidation. If a requirement proves technically
impracticable, implementation stops and this specification is amended before a
different architecture is built. A compatibility implementation is not an
acceptable substitute.

The architecture is complete only after the superseded implementation and its
adapters, branches, abstractions and tests have been removed. Green tests for
the old architecture are not an acceptance criterion.

## Start here

1. Read [CONTRIBUTING.md](../../../CONTRIBUTING.md) and the
   [architecture and invariants](architecture.md).
2. Read [open review items](review-status.md); resolve applicable blockers
   before treating a task as ready for implementation or closure.
3. Open the next task in the [sequential task index](tasks/README.md). Its entry
   conditions identify the contracts required for that work.
4. Record actual implementation and verification evidence using the
   [task handoff format](tasks/README.md#completion-and-handoff).

## Contract map and reading order

Each topic has one canonical owner below. Tasks reference these contracts and
do not duplicate their layouts, constants or state machines. The list is the
architectural reading order, not permission to implement tasks out of sequence.

- [Architecture, objectives and invariants](architecture.md)
- [Supported baseline and generated manifest](baseline.md)
- [Evidence from reference engines](reference-evidence.md)
- [Policy defaults, layouts and safety bounds](policy-and-capacity.md)
- [Direct memory and fault authority](memory-authority.md)
- [Tagged state cells, keys and dispatch](runtime-state.md)
- [Code units, versions, registries and admission](units-and-registries.md)
- [Gateway, state transfer and helper ABI](native-abi.md)
- [Cranelift fork and backend contract](backend.md)
- [Control budget and functional sampling](sampling-and-budget.md)
- [Synchronous LCQ compiler](lcq.md)
- [Static links, indirect dispatch, returns and target protection](linking.md)
- [HCQ admission, region formation and reshape](hcq.md)
- [Compilation and publication pipeline](publication.md)
- [Executable cache, metadata and backend ownership](cache.md)
- [Coordinator phases, terminal control and signal installation](coordinator.md)
- [Maintenance records, cleanup and mapping requests](maintenance-records.md)
- [Cohort workspace, arbitration and mutation plans](cohort.md)
- [Shutdown requests, result ownership and wait protocols](shutdown-and-waits.md)
- [Request execution, terminal transfer and cohort handoff](coordinator-execution.md)
- [Code and mapping invalidation](invalidation.md)
- [Native fault transport and completion](faults.md)
- [Epoch reclamation and terminal teardown](epochs-and-shutdown.md)
- [Failure policy and concurrency rules](failure-policy.md)
- [Migration and removal map](migration.md)
- [External performance acceptance](performance.md)
- [Final conformance gate](conformance.md)

## Tracked artifacts and release pin

All normative documents and generated inputs live under
`docs/specs/tiered-jit/`. Task 0 creates `tiered-jit-baseline.toml`; Task 3
creates `tiered-jit-protocol.toml`. Their absence before those tasks is expected;
placeholder manifests must not be used to claim a gate passed. The entire
repository-root `notes/` directory stays Git-ignored without exceptions.

The upstream base is [Wasmtime v48.0.1](https://github.com/bytecodealliance/wasmtime/releases/tag/v48.0.1),
commit `7bac2c2775808aaec5d4aa5627a5e447b51102cf`, containing Cranelift `0.135.1`.
See the [baseline schema](baseline.md) and [fork contract](backend.md) for the
separate immutable Nixe fork pin and required proofs. Preparing the branch does
not modify the project's current Cargo dependencies or complete the backend
changes and their proofs.

Fork development uses branch `nixe` in `/home/pladaria/projects/wasmtime`,
initially created at that release commit. The [workspace protocol](backend.md#fork-workspace-and-development-branch)
defines creation/resumption and how tested branch commits become immutable
Cargo pins.

## Editing this set

Amend a contract in its owning chapter and update every affected task and
acceptance condition in the same change. A later paragraph or task does not
silently override a conflicting contract: record and resolve the conflict in
[review status](review-status.md). Preserve Task 0–10 identities when subdividing
work. Keep protocol step numbers local to their named protocol and use a
relative link when referring to a different chapter; avoid new unqualified
“above” or “below” references. Do not retain a second monolithic normative copy.
