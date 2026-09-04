# Task 7: add family leases, boundary sampling and versioned reshape

[Specification index](../README.md) · [Task sequence and evidence](README.md) · [Open review items](../review-status.md)

## Entry conditions and required reading

Status: pending; this file is a specification, not an implementation completion record.

Prerequisite: [Task 6](06-background-admission.md) must be accepted first.

Read [CONTRIBUTING.md](../../../../CONTRIBUTING.md), the
[architecture and invariants](../architecture.md), the
[task execution contract](README.md), and these task-specific contracts before
editing production code: [HCQ admission, region formation and reshape](../hcq.md); [Code units, versions, registries and admission](../units-and-registries.md); [Compilation and publication pipeline](../publication.md); [Cohort workspace, arbitration and mutation plans](../cohort.md); [Code and mapping invalidation](../invalidation.md). Follow their related-contract links when
the change consumes a referenced protocol.

## Work

Enable BoundaryTable/ReshapeSnapshot and the four-sample request only now that a
current HCQ family exists. Implement ordered acquisition of exactly one or two
CurrentStable family replacement leases, reshape OwnerCell validation, mandatory
endpoint/foreign-collision trimming, structural negative completion and the
single no-fail predecessor/successor publication transaction. Zero matching
families is stale and routes useful LCQ work through ordinary admission.

The successor is CurrentPendingCutover; selected keys immediately name it,
dropped keys publish their pinned LCQ, predecessors become CutoverOld, and all
predecessor roots remain callable until the [Task 4](04-native-linking.md) rendezvous cuts them. Only
then may the successor become CurrentStable and the predecessors retire. An
invalidation/eviction during any lease/pending state follows Withdrawing and
exact-token cleanup, never an unconditional restore.

## Acceptance criteria

 unrelated workers still compile concurrently and no foreign
InstructionKey reaches backend work. A first-root partition can grow, shrink,
merge or repartition; both boundary endpoints remain mandatory. Identical over-
cap/disconnected shapes stay directly linked while their exact bounded negative
entry is resident, and may retry only after fingerprint change or deterministic
cache eviction. Stale workers cannot publish, free or clear newer ownership.
No CurrentPendingCutover family accepts another reshape; at most one successor
plus two immediate callable predecessors coexist, all old roots are cut before
retirement and every omitted member dispatches to its retained LCQ.
