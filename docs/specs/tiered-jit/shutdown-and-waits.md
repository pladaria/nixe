# Shutdown requests, result ownership and wait protocols

[Specification index](README.md) · [Implementation tasks](tasks/README.md) · [Open review items](review-status.md)

Related contracts: [Coordinator phases, terminal control and signal installation](coordinator.md); [Maintenance records, cleanup and mapping requests](maintenance-records.md); [Request execution, terminal transfer and cohort handoff](coordinator-execution.md); [Epoch reclamation and terminal teardown](epochs-and-shutdown.md).

Shutdown is deliberately not a MaintenanceRecord or RootMutationPlan. The
process owns one precharged, never-reused ShutdownRecord represented by
`shutdown_word: AtomicU64` and `shutdown_parking_sequence: AtomicU32`, initially
zero and one respectively. Its exact word layout is:

```text
bits 0..1    tag: 0=Idle, 1=Requested, 2=Claimed, 3=Applied
bits 2..17   nonzero TerminalCauseCode as repr(u16), or zero only for Idle
bits 18..49  cause-scoped TerminalDetailCode: repr(u32); zero=None
bits 50..63  zero; any nonzero bit is corruption
```

`TerminalCauseCode` has exactly these discriminants: `ExplicitShutdown=1`,
`JitCommitInvariant=2`, `IdentityExhausted=3`,
`PublicationSequenceExhausted=4`, `PhaseSequenceExhausted=5`,
`LedgerInvariant=6`, `SignalInstallFailure=7`, `SignalContextInvalid=8`,
`NestedNativeFault=9`, `HandlerCounterOverflow=10`,
`TlsOrAltStackCorruption=11`, `BackendInvariant=12`, `WorkerInvariant=13`,
`CacheCoherenceFailure=14`, `ProtectionFailure=15` and
`InternalInvariant=16`. Detail is exclusively a closed cause-scoped enum
discriminant, never a handle, pointer, string or dynamically allocated object.
[Task 0](tasks/00-baseline.md) freezes every detail-domain discriminant.
The initially required detail constants are
`CacheCoherenceFailure::CodeSync=1`, `LedgerInvariant::SequenceExhausted=1`,
`InternalInvariant::NativeFault=1` and
`InternalInvariant::TaggedStateCorruption=2`; all other call sites in this
version pass detail zero.
The decoder accepts exactly `Idle/cause=0/detail=0/reserved=0` or a non-Idle
tag with one listed nonzero cause, zero/one detail permitted by that cause's
generated closed table and zero reserved bits. A caller proposing an invalid
cause/detail never encodes it; before any shutdown claim it instead invokes
`latch_terminal(InternalInvariant, TaggedStateCorruption)`. Observing an
already-stored word outside the accepted table means first-writer state itself
is corrupt and immediately invokes raw `SYS_exit_group(70)`; it is never
normalized, masked, treated as Idle or overwritten with a later cause.

The only state edges are `Idle -> Requested(cause, detail) ->
Claimed(cause, detail) -> Applied(cause, detail)` and both later CASes preserve
bits 2..49 exactly. The first AcqRel CAS from zero fixes the sticky cause;
the edge itself does not yet certify completion of its required post-CAS tail.
The parking sequence is initialized to one and the three edges must publish
exactly 2, 3 and 4. `edge_complete(Idle|Requested|Claimed|Applied)` is therefore
respectively `parking_sequence >= 1|2|3|4`, observed with Acquire. Before each
CAS the owner arms `ShutdownWakeGuard` with that edge's fixed next value.
A failed CAS disarms it; after a successful CAS, a non-unwinding tail makes the
guard release-store the value and call raw
`FUTEX_WAKE_PRIVATE(INT_MAX)`. Its Drop completes an omitted post-CAS
publication/wake and raw-exits 70 on any impossible word/value mismatch, so
unwind cannot expose a semantic edge without its wake. The owner of a successor
edge must first acquire-observe completion of the predecessor: Requested-to-
Claimed requires sequence at least two, and Claimed-to-Applied at least three.
A caller which will take a successor edge, join completion or return a completed
shutdown outcome does not treat a non-Idle tag as ready until it acquire-
observes that tag's completion sequence; if the word advances meanwhile, it
waits for the newer tag's value. `latch_terminal` may return the explicitly
incomplete LatchedPending notification without waiting. A joiner loads
the word and sequence, rechecks both, and performs FUTEX_WAIT on that exact u32
with `FUTEX_WAIT_PRIVATE` in a loop until Applied is edge-complete. It has no request sequence, RoundTicket, reopen epoch,
generation or separately writable result cell. `shutdown_requested` is a
sticky wake hint which changes false to true and is never cleared; it is never
independent authority.

MappingChange, invalidation, pressure eviction, tier cutover and deactivation
are safety-critical. InstallLink is performance-only; failure or deferral leaves
the permanent fallback intact. Queue nodes are never allocated on the request
path: each PatchRecord has a performance record, each UnitLifecycle a distinct
safety-unlink record, each HcqFamily a distinct tier-cutover record, each
registered vCPU has separate precharged `FaultTransitionRecord` and
`VcpuDeactivateRecord` instances, and each address-space memory authority has
one serialized mapping record. The process additionally owns one precharged
`PressureRecord`, distinct from every unit/family record, whose plan storage is
two fixed bitmaps sized to the unit- and family-registry capacities plus
generational request metadata. Those bitmaps are exclusively the immutable
PressureEvictRequest selection input, not a RootMutationPlan; ClosingPrepare
expands them into the common multidomain workspace. Selected units use their own precharged
SafetyPayloadCells rather than duplicating payloads in the bitmaps. The separate
process ShutdownRecord is also precharged. An occupied record cannot be
overwritten or queued twice. A later
safety request either joins a subsuming exact safety result or uses its own
typed owner record; it never waits for a performance record. Terminal shutdown
subsumes every normal record. Concurrent pressure requests join the occupied
PressureRecord generation; after its terminal result the requester recomputes
the projection and either observes the target satisfied or claims the next Free
generation. An HCQ allocation never waits for it and returns transient pressure;
an LCQ capacity request may join/wait because LCQ reclamation is safety work.

`FaultTransitionRecord` is not an eighth RootMutationRequest variant and never
enters the maintenance queue. It is the vCPU-owned, preallocated reconstruction/
result scratch embedded exactly once in each `vcpu_runtime_slots` row. Its
frozen handle and payload are:

```text
FaultTransitionRecordHandleV1 {
  vcpu_slot:u32, reserved_zero:u32, vcpu_generation:u64,
  record_generation:u64, fault_record_id:u64,
  captured_fault_slot_sequence:u64
}
PlanValueV1 {
  source_kind:StateComponent|Temp, source_ordinal:u8,
  bit_width:u16, reserved_zero:u32, little_endian_bits:[u8;16]
}
FaultReconstructionV1 {
  unit:UnitHandle, code_version:u64, native_pc_offset:u32,
  state_map_id:u64, memory_plan_id:MemoryEffectPlanId,
  faulting_subaccess:u8, committed_stage_count:u8,
  prefault_value_count:u8, temp_value_count:u8,
  effective_address:u64,
  prefault_values:[PlanValueV1;64], temp_values:[PlanValueV1;32]
}
```

Unused values are all zero; populated values occur in MemoryEffectPlan order
and exactly match its declared widths. The record has a checked nonzero u64
record_generation and the closed non-atomic graph
`Empty -> Prepared(handle, FaultReconstructionV1) ->
WaitingMapping(MaintenanceRecordHandle, request_sequence) ->
Consumed(MaintenanceResult) -> Empty`. It is initially Empty/generation one;
the owner reserves the next generation before reuse and never wraps. It is
mutated only by the unique FaultSlot owner while that owner holds its
VcpuUseToken or SuspendedResumeUseToken, except for one immutable borrow by the
coordinator: the owner completes WaitingMapping before release-publishing the
MappingRecord Pending state; after acquiring that record the coordinator
validates every handle field and may read, but never write, the frozen
reconstruction. It ends that borrow before release-publishing MappingResult.
The owner acquire-observes the result before writing Consumed and then Empty.
Every edge validates exact FaultSlot/vCPU/record generations, so there is no
concurrent mutation or unguarded reusable handle. Its owner
acquires the address space's MappingRequestPermit, copies the typed request into
that address space's single `MappingChange` MaintenanceRecord, waits for and
copies the exact result, returns that record to Free, releases the permit and
then changes Consumed to Empty before releasing its final result/use token.
Terminal reads it only after use/resume, FaultSlot and NormalResultToken drains,
and requires every configured record Empty before arena unmap. Thus simultaneous fault and API mapping
changes are serialized through one queue node and one memory-authority
operation; no vCPU record can overwrite or bypass it.

Independent operations sharing one address-space mapping record are serialized
before any guest-visible memory effect by its checked `MappingRequestPermit`:
`Free(g) -> Owned(g, request_id) -> Free(g + 1)`. Stopping cancels/wakes the
owner's MappingRecord, but does not overwrite Owned; that owner consumes the
exact result and returns it to Free. Only terminal step 8, after the normal-
result drain proves no owner remains, changes Free to `Terminal(cause)`.
Acquisition and waiting occur only after the
caller has copied its reconstruction state, cleared its executor epoch and
released every UnitPin and subsystem lock. A busy caller waits on the permit's
checked generation and shutdown wake; it neither queues a second node nor
changes memory. The owner releases the permit only after its exact record result
is terminal and consumed. This serialization means record occupancy cannot lose
or reject a safety mapping operation.
Before joining a different terminal driver, a permit owner waits only for or
finishes its own MappingChange record, copies its result, releases its normal
result reference/token, returns the address-space record to its recyclable
state, and then releases MappingRequestPermit. If a fatal Applying tail owns
the permit, it completes the required no-fail mutation, records
TerminatedAfterMutation, performs that same local consumption/release and only
then calls `drive_or_join_terminal`. No caller waits for ShutdownRecord while
owning a mapping permit.

Within a nonterminal cohort, the total conflict precedence is
`InvalidateToUnavailable/MappingChange > VcpuDeactivate > PressureEvict >
CutoverToSuccessor > UnlinkOnly > InstallLink`. VcpuDeactivate competes only
for its vCPU/PIC marks; if MappingChange has already removed one of those PIC
roots, deactivation treats the absent exact root as satisfied rather than
recreating or failing it. The higher plan
changes an unclaimed lower Pending InstallLink directly to
AwaitingLinkCleanup(StaleNoOp(Superseded)) under the queue mutex. If the install
is already Claimed, cohort precedence changes Claimed to the same awaiting
state; its LinkCleanupTicket later releases any claimed pins outside the queue
mutex. Applying is never cancelled or rolled back; its
no-fail tail completes and the safety plan cuts the resulting root in its
sealed or next ticket. Lower safety plans are coalesced or become
StaleNoOp after the higher result; record occupancy can never make safety fail.
For one vCPU, a MappingChange initiated by FaultTransitionRecord is applied
before VcpuDeactivate.
If the latter finds the still-suspended retry described above, it performs only
the specified root cut/wake and enters WaitingFaultIdle; its next-ticket
continuation requires FaultSlot Idle before clearing PIC roots and completing
Inactive. The two requests never share or overwrite a record.

Every nonshutdown requester acquires a `NormalResultToken` before its first
maintenance-record or reclaimable-arena dereference. TerminalControl's
`normal_result_gate` is a CountedCloseGateV1 initialized open, never reopened,
and has limit COUNT_MASK. Acquisition is exactly `try_acquire`; success is
authority to dereference and failure reads the sticky terminal result using
TerminalControl only. At capacity a safety caller waits on that gate word
without any other lock and retries while it remains open, whereas a
performance-only caller leaves fallback unchanged. `latch_terminal` closes the
gate before publishing Requested, so a token acquired before close is counted
and a later acquisition cannot succeed; the terminal driver therefore waits
only for the finite pre-Requested holders. An asynchronous request drops the
token after its queue node owns all data; a synchronous request retains it
through copying its exact terminal result into caller-owned storage, then drops
its result-cell reference and token on every exit.

Any request path which can claim or fill a MaintenanceRecord, PatchRecord,
bridge, ActiveBuild cell or other cleanup-bearing state before that state is
visible to its owning queue must additionally own a `PrequeueProducerToken`,
unless it already owns an OpenToken through that entire interval.
TerminalControl's `prequeue_producer_gate` is another CountedCloseGateV1,
initialized open, never reopened and limited to COUNT_MASK. Acquisition is its
single `try_acquire` CAS; at capacity it uses the same safety-wait/performance-
fallback rule as NormalResultToken. `latch_terminal` closes it before Requested,
and failure thereafter touches no prequeue object. The owner
retains it from before its first prequeue claim until either (a) the queue-
visible record owns every immutable datum and cleanup ticket or (b) complete
local rollback, including any InstallPending bridge/span cleanup, has finished.
If Stopping appears during (b), registration is still open: the owner may
publish its counted AwaitingLinkCleanup before dropping the producer token, but
may not leave a hidden cleanup obligation. Publishers, HCQ admission and vCPU
deactivation which already retain an OpenToken do not increment this second
counter. The terminal driver drains both the OpenToken and prequeue gate low
counts to zero before its exhaustive record
conversion and only then closes Awaiting registration. This count uses the
typed release primitive: every checked decrement broadcasts (including both
MAX-to-MAX-1 and one-to-zero) and each waiter acquire-rechecks the count.
Because close and acquire modify the same gate word, there is no paused
load/increment window and no rejected post-drain increment.

All finite counters which another thread waits to reach zero use a typed RAII
release primitive. The two Awaiting counts wake exactly on one-to-zero.
CountedCloseGate words and other counters whose acquisition may wait for
capacity—requester_refcount, FamilyPin and UnitPin—broadcast after every
checked decrement, so both MAX-to-MAX-1 capacity
waiters and one-to-zero drain waiters make progress. A nonzero-to-zero
active-code-epoch store performs its epoch wake after
the release-store. Signal-handler counters use the equivalent async-signal-safe
atomic decrement plus raw `SYS_futex(FUTEX_WAKE_PRIVATE, INT_MAX)` and never
libc. Every zero transition broadcasts with `INT_MAX`, because several
terminal, deactivation or grace waiters may observe one counter; every waiter
loops on an acquire reload and tolerates spurious wakes. Every success,
rejection, rollback, fault suspension/resume and terminal path goes
through these primitives; an unmatched decrement or underflow is fatal
corruption.
Linux `FUTEX_WAIT_PRIVATE` is used only on an aligned native-endian
`AtomicU32`, never by casting an AtomicU64 or selecting an arbitrary half. Each
waitable TaggedPayloadCell, FaultSlot and active-code-epoch slot owns a checked
`parking_sequence: AtomicU32`, initialized to one.
Every successful transition of a waitable TaggedPayloadCell or FaultSlot, and
every active-epoch nonzero-to-zero publication, release-increments that sequence
before `FUTEX_WAKE_PRIVATE(INT_MAX)`; implementations do not decide whether a
particular waiter might benefit. A waiter copies the complete semantic
state, acquire-loads W, rechecks the complete predicate, and only then waits on
`&parking_sequence` with expected W; a racing publication produces EAGAIN or a
wake. It always loops and revalidates the full u64/generational state, so the
parking word is a notification hint, never authority. The notification reserve
is sixteen for every TaggedPayloadCell including LcqBuildCell, four for a
FaultSlot and one for an active-epoch slot. Normal increments stop at
`u32::MAX - reserve`; exhaustion
closes the corresponding admission and latches terminal before consuming the
reserve. Every remaining bounded terminal transition or zero publication uses
one distinct value, and the final value is consumed only when the waited
predicate is permanently terminal/zero. No word wraps; [Task 1](tasks/01-abi.md) loom models each
boundary.

Before each semantic state CAS/store, the transition owner preflights and arms
a `ParkingWakeGuard` for that exact reserved next parking value. CAS/store
failure disarms it without advancing the word. Success enters a non-unwinding
tail in which the guard release-publishes the preflighted value, performs the
futex wake and disarms. Its Drop performs any uncompleted suffix using only the
parking atomic/raw futex; an impossible mismatch or syscall failure raw-exits
70. Active-epoch zero uses the same specialized guard. Consequently unwind
cannot leave a changed semantic predicate with an unchanged parking sequence,
and the two-publication parity loom case above also pauses between semantic
publication and notification.

ArenaAdmissionToken instead release-clears its exact bit in one of four aligned
AtomicU32 bitmap words with `fetch_and(!bit)`, requires the previous word
contained that bit and performs `FUTEX_WAKE_PRIVATE(INT_MAX)` on that exact
word, and then releases its matching `arena_admission_gate` count. Terminal
first closes that gate and acquire-waits for `CLOSED|0`; it then acquire-scans
the four bitmap words in ascending order and requires all zero. A nonzero bit
at zero gate count is fatal unmatched-ownership corruption. Because the bit is
set only while a successfully acquired gate count is held, no post-close bit
publication exists and no bitmap futex wait is required for correctness. Every
bit clear still wakes its word for diagnostics and nonterminal same-slot
contenders. Active-code-epoch zero stores
and FaultSlot/LCQ transitions use their own parking sequence as specified
above.
