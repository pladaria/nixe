# Task 3: cut synchronous LCQ over as one vertical slice

[Specification index](../README.md) · [Task sequence and evidence](README.md) · [Open review items](../review-status.md)

## Entry conditions and required reading

Status: pending; this file is a specification, not an implementation completion record.

Prerequisite: [Task 2](02-lifetime-foundation.md) must be accepted first.

Read [CONTRIBUTING.md](../../../../CONTRIBUTING.md), the
[architecture and invariants](../architecture.md), the
[task execution contract](README.md), and these task-specific contracts before
editing production code: [Synchronous LCQ compiler](../lcq.md); [Control budget and functional sampling](../sampling-and-budget.md); [Direct memory and fault authority](../memory-authority.md); [Code and mapping invalidation](../invalidation.md); [Native fault transport and completion](../faults.md); [External performance acceptance](../performance.md). Follow their related-contract links when
the change consumes a referenced protocol.

## Work

Replace region BFS with one demanded straight-line fragment ending at the first
terminator or 512-instruction emergency cut. Implement its generational
same-key LcqBuildCell, ResourceCut/halving rule, opt_level=none/single_pass output,
complete canonical/fast entry metadata and permanent canonical exit fallbacks.
Enter it through the production gateway/dispatch index and implement signed
block-level control-budget accounting; functional tier sampling remains
disabled.

This vertical slice also implements executable provisional-copy/arming/final-
copy leases, all dependency indexes and the production signal/fault state
machine on top of [Task 2](02-lifetime-foundation.md)'s already-live capture/landing/TLS foundation. It adds
the five-way classifier, normal-stack resolver, suspended-retry protocol,
LcqBuild shutdown wake, occupied FaultTransition cleanup and the fault-owned
extensions of the already-complete address-space/permit terminal protocol.
Connect every
existing CPU and non-CPU backing writer to the one memory
authority. LCQ must support all five fault dispositions now: local ordinary-RAM
retry, canonical MMIO completion, observed-code write transition, precise guest
data abort and internal fault. Lower every MemoryEffectPlan class correctly;
[Task 8](08-lifecycle-stress.md) later exhausts combinations but does not introduce missing safety.
Cut dispatch to this LCQ in one migration transaction and make the old region/
context-tail executor unreachable before closing the task.

Before this task closes, and therefore before [Task 4](04-native-linking.md) or any comparative
profiling begins, the repository must track
`docs/specs/tiered-jit/tiered-jit-protocol.toml` and add workspace package
`tools/tiered-jit-perf` with shared protocol/schema code plus binaries
`tiered-jit-perf-runner` and `tiered-jit-perf-analyze`. The exact CI entrypoint
is `cargo run --locked --package nixe-tiered-jit-perf --bin
tiered-jit-perf-analyze -- check-protocol --protocol
docs/specs/tiered-jit/tiered-jit-protocol.toml`; package tests contain the synthetic
fixtures, checksum cases and golden ChaCha8 bootstrap-index vector. Both that
command and `cargo test --locked --package nixe-tiered-jit-perf --all-targets`
must pass, and `git ls-files --error-unmatch` must succeed for the protocol and
every package source/fixture.

## Acceptance criteria

 a cold key decodes no successor and no payload is reachable
until every exact code-page lease/alias is armed and its final bytes revalidated.
Barrier-controlled writes racing provisional copy, arming, final copy and
publication either stale the candidate or unlink it before visibility. Same-
key/current-version races compile once, exact waiters wake on every terminal
outcome and unrelated vCPUs compile concurrently. Ordinary dirty tracking
restores the identical native PC/context once; MMIO and observed-code stores
produce exactly one effect and continue canonically; invalid/compound accesses
match MemoryEffectPlan stage/writeback semantics. Zero/small slices and loops
reach the declared checkpoints, fault exits charge once, one-instruction
overflow fails precisely and longer overflow follows exact halving. Every
overlapping LCQ dependency invalidates. No BFS LCQ, process compiler mutex,
per-entry promotion sequence, old PublishedRegion route, JITModule-owned new
code or legacy context-tail executor is callable.
