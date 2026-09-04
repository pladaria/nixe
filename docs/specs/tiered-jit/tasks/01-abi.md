# Task 1: freeze and prove the executable ABI and state contracts

[Specification index](../README.md) · [Task sequence and evidence](README.md) · [Open review items](../review-status.md)

## Entry conditions and required reading

Status: pending; this file is a specification, not an implementation completion record.

Prerequisite: [Task 0](00-baseline.md) must be accepted first.

Read [CONTRIBUTING.md](../../../../CONTRIBUTING.md), the
[architecture and invariants](../architecture.md), the
[task execution contract](README.md), and these task-specific contracts before
editing production code: [Supported baseline and generated manifest](../baseline.md); [Cranelift fork and backend contract](../backend.md); [Gateway, state transfer and helper ABI](../native-abi.md); [Direct memory and fault authority](../memory-authority.md); [Tagged state cells, keys and dispatch](../runtime-state.md); [Control budget and functional sampling](../sampling-and-budget.md). Follow their related-contract links when
the change consumes a referenced protocol.

## Work

Materialize NativeFrame, DispatchPayload, every checked identity authority,
Canonical/FastEntryContract, ExitSiteKey,
Physical/Exit/PreFaultStateMap, runtime HelperDescriptor derived byte-for-byte
from HelperCatalogEntryV1 and the exact native patch
shapes. Implement the one shared use/def, partial-write, dirty and boundary-
observation analysis and the compile-time NativeFrame/assembly offset checks.
Consume the [Task 0](00-baseline.md) key encodings/catalogs and connect its fork hooks to
production lowering; no test-only state-map source may survive.

[Task 1](01-abi.md)'s exit tests compile the **real production lowering** for every CoverageId
on both target ISAs with Fastalloc and Ion, compare its statepoint/patch/helper
contract with the catalog row, and require its actual body/island/stack demand
not to exceed [Task 0](00-baseline.md)'s frozen bound. A missing real lowering, a fixture still
reachable from production, or a bound excess blocks [Task 1](01-abi.md) and requires an
explicit baseline/specification amendment.

Integrate the [Task 0](00-baseline.md) production emitters and add semantic verifiers for the
system-ABI gateway, canonical landing, guest/host FP ownership, parallel-copy
planner and Leaf-helper veneer; a second encoder is forbidden. The gateway
exposes the exact call sites/relocations for the
[Task 2](02-lifetime-foundation.md) OpenToken-to-active-epoch primitive, but [Task 1](01-abi.md) neither wires nor executes
that primitive and does not invent a temporary executable allocator. Generate
and independently disassemble both target byte streams; run the parallel-copy,
state-map and helper-layout oracles without jumping to generated code.

## Acceptance criteria

 cross-target encoder tests prove that inter-fragment shapes
contain no gateway, prologue/epilogue, system-ABI call, explicit SP adjustment,
host return or unnecessary A64State store. Randomized final-allocation maps
reconstruct all required values, respect exact NativeFrame ranges and preserve
reserved registers. Helper-layout tests preserve stable SP, x30/system return,
lazy flags and guest FP state, stay within each 2048-byte stack half and reject
one-byte overflow. Versioned disassembly assertions and exported offsets agree
for every entry/patch/fault statepoint. There is one production StableEncode,
one semantic IR/use-def analysis, one typed helper table and no handwritten
helper signature or test-only liveness classifier. No [Task 1](01-abi.md) test maps an
executable page or claims to prove epoch ordering.
