# Task 2: build the bounded publication and lifetime foundation

[Specification index](../README.md) · [Task sequence and evidence](README.md) · [Open review items](../review-status.md)

## Entry conditions and required reading

Status: pending; this file is a specification, not an implementation completion record.

Prerequisite: [Task 1](01-abi.md) must be accepted first.

Read [CONTRIBUTING.md](../../../../CONTRIBUTING.md), the
[architecture and invariants](../architecture.md), the
[task execution contract](README.md), and these task-specific contracts before
editing production code: [Code units, versions, registries and admission](../units-and-registries.md); [Compilation and publication pipeline](../publication.md); [Executable cache, metadata and backend ownership](../cache.md); [Coordinator phases, terminal control and signal installation](../coordinator.md); [Maintenance records, cleanup and mapping requests](../maintenance-records.md); [Cohort workspace, arbitration and mutation plans](../cohort.md); [Shutdown requests, result ownership and wait protocols](../shutdown-and-waits.md); [Request execution, terminal transfer and cohort handoff](../coordinator-execution.md); [Epoch reclamation and terminal teardown](../epochs-and-shutdown.md). Follow their related-contract links when
the change consumes a referenced protocol.

## Work

Replace JITModule ownership with staged byte/relocation output and implement the
segmented dual-alias W^X allocator, exact charged ledger/reservations, metadata
arena, fixed Robin-Hood dispatch index, generational registries, CodeUnit/family
lifecycle, UnitPins and global/per-executor epochs. Implement per-host-page COW
PageFaultTables, segment-generation reuse and the one cache/pipeline coherence
wrapper for initial images and patches.

Implement TerminalControl, ShutdownRecord, NormalResultToken,
PrequeueProducerToken, OpenToken, typed MaintenanceRecord/cohort sequencing, queue-serialized
reopen, terminal Stopped and shutdown primitives. Implement the fallible-
prepare/no-fail DispatchPayload commit with synthetic native units containing no
inter-unit links. Every allocation and COW replacement needed after commit must
be reserved during preparation. The legacy executor may remain isolated for
one task, but none of these new objects is owned by JITModule or an append-only
side registry. Implement the production-general `ReclaimToSoftWatermark`
request/acknowledgement on this maintenance foundation.

This task also owns the base executor lifetime required by its native proof:
AddressSpace/vCPU registration, ArenaAdmissionToken, VcpuUseToken, FaultSlot
Idle, guarded SignalAltStack and landing-stack allocation, HostSignalControl
installation/chaining, minimal async capture/landing ABI, direct TLS descriptor
publication/detach and AArch64 PROT_BTI mapping. WorkerJoinSet exists with zero
enabled workers. [Task 2](02-lifetime-foundation.md) implements complete terminalization for every state it
can create: base AddressSpace/vCPU registry states, MappingRequestPermit and
MappingChange records, ReclaimControl, empty FaultTransition records, the
zero-worker JoinSet and both arenas. It does not yet classify a native fault or execute a
MemoryEffectPlan; those resolver semantics belong to [Task 3](03-lcq-cutover.md).

After those primitives exist, wire the [Task 1](01-abi.md) gateway to the real
OpenToken-to-active-epoch entry primitive and build the native two-fragment ABI
proof on Linux x86-64 and Linux AArch64. Through the production allocator and
encoders it enters once, crosses a test-prewired empty bridge, crosses a GPR/
vector cycle through transfer slots, invokes a typed test Leaf helper which
forces stack arguments, and exits canonically with exact GPR/vector/SP/PC, lazy
NZCV, FPCR and FPSR state. These test fragments do not implement a production
static/PIC root or link lifecycle. The proof and its scaffolding remain until
[Task 4](04-native-linking.md) exercises both bridge shapes through the production linker; [Task 4](04-native-linking.md) then
migrates the assertions and removes only the prewired test route.

## Acceptance criteria

 deterministic barrier tests pause a publisher before/after
OpenToken acquisition and prove it either commits before the Closing freeze
point and is included by invalidation, or changes no reachable state. A request
inserted at the final Closed/Open handoff belongs to the sealed cohort or wins
the next close; no wakeup is lost. Readers cannot observe an executable pointer
before their active epoch or a torn payload. Both native ABI proofs establish
that ordering and all [Task 1](01-abi.md) state/stack preservation claims. Failure injection
at every fallible precommit step leaves no root; from commit step 1 onward the
tail contains no fallible operation, including before the first payload
exchange. Old PageFaultTable readers and UnitPins prevent directory/span reuse,
then exact address/slot reuse succeeds after grace. At the 512/608/640 MiB
edges, a debug arena walk equals
`committed_executable_bytes + committed_metadata_bytes`, a reservation-table
walk separately equals `reserved_credit_bytes`, and their checked sum equals
`jit_charged_bytes`. Adjacent cleared spans appear as one free bitmap run;
empty segments/metadata pages
decommit, `/proc` inspection finds no RWX virtual mapping and every published
RW twin is PROT_NONE outside the audited writer window, cross-core execution
observes synchronized new code, and the new foundation contains no append-only
code/fault/retirement owner. A dedicated zero-worker test shuts down from every
base AddressSpace/permit/reclaim phase, reaches Applied, observes an empty
JoinSet and proves both arenas are unmapped; [Task 3](03-lcq-cutover.md) may not be required for
that test to terminate.
