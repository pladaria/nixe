# Technical review status

[Specification index](README.md) · [Implementation tasks](tasks/README.md)

Status: open. The preceding technical review was interrupted. Splitting its
working document and replacing the development base with a stable release
preserves the proposed contracts but does not resolve their outstanding
contradictions. No implementation task has been certified complete by this
documentation change. The items below are the known review backlog, not a
claim that the rest of the design has already passed a complete audit.

## Known blockers and required resolutions

| ID | Owning contracts | Required resolution before the affected task can close |
|---|---|---|
| TJ-R01 | [Backend](backend.md), [baseline](baseline.md) | Revalidate all proposed fork hooks, allocator constraints, consuming output APIs and structural bounds against stable v48.0.1. Source-level API presence alone does not prove the custom ABI feasible. |
| TJ-R02 | [Units/registries](units-and-registries.md), [cohort](cohort.md), [cache](cache.md) | Complete PageFaultTable COW ownership: precharge the alternate UnitPin-set block, provide a stable owner location for the shadow, separate logical count from immutable allocation capacity, and specify exact transfers and release of all three allocations. Recalculate layouts and charge. |
| TJ-R03 | [Cohort](cohort.md), [coordinator execution](coordinator-execution.md) | Close the exhaustive collision table for ordinary overlapping invalidation/deactivation/pressure/cutover/unlink claims; define direct effect-stream access and unique allocation accounting; remove duplicate cutoff sealing; preserve the VcpuDeactivate WaitingFaultIdle continuation at the final barrier. |
| TJ-R04 | [Cache](cache.md), [cohort](cohort.md), [policy/layouts](policy-and-capacity.md) | Reconcile Pressure victim selection with the bounded projection algorithm, define deterministic candidate ordering and effect accounting, and prove the visit bound without a hidden per-victim rescan. |
| TJ-R05 | [Coordinator](coordinator.md), [faults](faults.md), [epochs/shutdown](epochs-and-shutdown.md) | Correct AArch64 SA_EXPOSE_TAGBITS installation and saved-handler chaining; specify initial epoch/parking values; prove that exit/landing suffixes execute from permanent process text before releasing arena/epoch lifetime. |
| TJ-R06 | [Publication](publication.md), [runtime state](runtime-state.md), [HCQ](hcq.md), [shutdown/waits](shutdown-and-waits.md) | Make shape-version, LCQ/HCQ completion and parking-sequence publication edges explicit under their sole writer; complete tagged payload layouts and verify semantic publication cannot race notification or final cleanup. |
| TJ-R07 | [Maintenance records](maintenance-records.md), [coordinator execution](coordinator-execution.md) | Give LinkCleanupTicket exactly one executor and an explicit ownership transfer; reconcile deferred-install sequence exhaustion with the closed reason enums and terminal protocol. |
| TJ-R08 | [Units/registries](units-and-registries.md), [HCQ](hcq.md), [cache](cache.md) | Specify how registry generation survives slot decommit/reuse, bridge metadata-pin acquisition to owner detachment without nested locks, and define NegativeBuildCache/BuildFingerprint ownership without duplicate full fingerprints. |
| TJ-R09 | [Cache](cache.md), [linking](linking.md), [policy/layouts](policy-and-capacity.md) | Complete segment/island owner schemas, generation and mutable storage rules, and partial executable MAP_FIXED rollback/decommit behavior. |
| TJ-R10 | [Policy/layouts](policy-and-capacity.md), [cache](cache.md), [baseline](baseline.md) | Produce an exhaustive non-double-counted layout/ledger model, the largest-HCQ-reservation formula and the LCQ emergency resource vector; prove the stated fixed, soft and hard budgets together. Numerical feasibility remains conditional on this gate. |
| TJ-R11 | [Performance acceptance](performance.md) | Resolve path confinement, runner/tool bootstrap, candidate/reference artifact identities, smoke/commercial schemas, pre-exec lifecycle and CPU allocation, capture/PSS timing, qualification/outcome scope, teardown, interrupted sealing and offline verification into one executable protocol. |
| TJ-R12 | [Task 8](tasks/08-lifecycle-stress.md), [conformance](conformance.md) | Add the focused regression/oracle cases required by the resolved contracts, including shared-page COW, every collision tuple, Pressure duplicates/bounds and collective Applying failure injection; audit cross-file references after those changes. |

## Resolution record

Resolve each item in its owning contract and update affected task acceptance
criteria in the same change. Mark an item resolved only with links to the
resulting contract sections and the relevant proof/test gate; do not erase its
ID or turn a proposed fix into evidence that the fix has been implemented.
When the technical review is complete, explicitly update this status and the
index. Until then, implementation readiness must not be inferred from the
words “exact”, “complete” or “exhaustive” in an unreviewed draft paragraph.
