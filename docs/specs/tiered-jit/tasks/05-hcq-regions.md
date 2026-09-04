# Task 5: build deterministic multi-entry HCQ with exclusive ownership

[Specification index](../README.md) · [Task sequence and evidence](README.md) · [Open review items](../review-status.md)

## Entry conditions and required reading

Status: pending; this file is a specification, not an implementation completion record.

Prerequisite: [Task 4](04-native-linking.md) must be accepted first.

Read [CONTRIBUTING.md](../../../../CONTRIBUTING.md), the
[architecture and invariants](../architecture.md), the
[task execution contract](README.md), and these task-specific contracts before
editing production code: [HCQ admission, region formation and reshape](../hcq.md); [Cranelift fork and backend contract](../backend.md); [Code units, versions, registries and admission](../units-and-registries.md); [Compilation and publication pipeline](../publication.md); [Executable cache, metadata and backend ownership](../cache.md). Follow their related-contract links when
the change consumes a referenced protocol.

## Work

Define and normalize immutable AdmissionSnapshot, define the post-discovery
OwnershipSnapshot, and build the production HCQ planner/compiler from those
injected values plus UnitPin-protected demanded-LCQ snapshots; automatic sampling and
workers remain disabled. Merge overlap by InstructionKey, form canonical
leaders, apply the fixed worklist/one 2048-instruction cap, freeze only actual
external entries, and reserve unowned keys with the normal OwnerCell
transaction before backend work. Publish the initial family as CurrentStable
through the already-safe [Task 2](02-lifetime-foundation.md)/4 lifecycle.

Reuse [Task 1](01-abi.md) liveness/state contracts so internal edges retain SSA and selected
entries load only true live-ins. Emit real selected labels with
opt_level=speed/backtracking; calls stay external. Implement reverse-ordinal
spill/segment/island trimming and exact structural rejection. A focused harness
calls this same production planner/compiler directly; it does not introduce a
test planner, alternate decoder or production forced-promotion branch.

## Acceptance criteria

 identical AdmissionSnapshot plus exact CodeUnit set yields
the identical canonical instruction/block/entry sequence, selected shape,
relocation kinds/addends and trim/rejection result. Final RX addresses and bytes
containing address-dependent relocations need not be identical.
Unexecuted successors are never decoded, overlapping LCQ roots contain one copy
per InstructionKey and no two unrelated builds begin backend work with the same
key. Every selected PC enters its exported real label; internal edges have no
adapter/canonical round trip or checkpoint except required backedges, and
coverage-only instructions create no dispatch slot. Forced spill/body/island
exhaustion recomputes the connected shape in reverse ordinal order and leaves
LCQ current on structural rejection. An actual allocator/fragmentation/608-MiB
failure after structural fit is transient, performs no trimming and creates no
negative-cache entry. Rollback clears only exact reservations. There is no
ordinal/br_table dispatcher, second decoder, trace/deoptimization machinery or
reachable HCQ without exclusive ownership.
