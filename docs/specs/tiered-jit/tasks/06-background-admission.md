# Task 6: enable functional sampling and bounded background admission

[Specification index](../README.md) · [Task sequence and evidence](README.md) · [Open review items](../review-status.md)

## Entry conditions and required reading

Status: pending; this file is a specification, not an implementation completion record.

Prerequisite: [Task 5](05-hcq-regions.md) must be accepted first.

Read [CONTRIBUTING.md](../../../../CONTRIBUTING.md), the
[architecture and invariants](../architecture.md), the
[task execution contract](README.md), and these task-specific contracts before
editing production code: [Control budget and functional sampling](../sampling-and-budget.md); [HCQ admission, region formation and reshape](../hcq.md); [Failure policy and concurrency rules](../failure-policy.md); [Shutdown requests, result ownership and wait protocols](../shutdown-and-waits.md); [Epoch reclamation and terminal teardown](../epochs-and-shutdown.md). Follow their related-contract links when
the change consumes a referenced protocol.

## Work

Activate the 4096-instruction functional sample and HotSeedTable, but not
BoundaryTable. Implement only the runtime producer/integration which populates
[Task 5](05-hcq-regions.md)'s normalized AdmissionSnapshot and UnitPin set, plus ActiveBuildTable,
anchor-token deduplication, NegativeBuildCache, the preallocated queue, fixed
worker formula and seven-newest/one-oldest service. Workers invoke the exact
[Task 5](05-hcq-regions.md) pipeline. Implement nonblocking admission, zero-worker behavior, partial-
startup rollback, checked sequence exhaustion and exact success, structural,
transient, stale, fatal and shutdown cleanup.
Implement WorkerControlCell/WorkerJoinSet ownership and the worker-as-terminal-
driver/loser protocol; test partial startup and a worker winning shutdown while
another observes Requested.

## Acceptance criteria

 LCQ contains only its already-required budget sub/test and
no hotness load/store/RMW or promotion branch. Seven matching samples do not
enqueue; the eighth can publish exactly one matching CurrentStable family, and
another vCPU attaches to the same active fingerprint rather than duplicating
it. A worker never clears a sampler way: the next cold poll which observes a
matching preferred payload clears its own way. Structural rejection retains the
saturated way while its bounded negative entry survives, transient completion
retries from score seven on a real sample, and stale identity starts the new
record at score one. Workers never read/write per-vCPU tables or raw unpinned
registry data. Collision, queue fullness, zero workers, Closing, cache pressure,
invalidation and shutdown never block guest admission or publish a wrong
identity; no sample is logged/exported and BoundaryTable remains disabled.
