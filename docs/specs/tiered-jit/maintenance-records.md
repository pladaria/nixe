# Maintenance records, cleanup and mapping requests

[Specification index](README.md) · [Implementation tasks](tasks/README.md) · [Open review items](review-status.md)

Related contracts: [Coordinator phases, terminal control and signal installation](coordinator.md); [Cohort workspace, arbitration and mutation plans](cohort.md); [Shutdown requests, result ownership and wait protocols](shutdown-and-waits.md).

A `MaintenanceRecordHandleV1` is exactly
`{ owner_record_ordinal:u32, reserved_zero:u32, record_generation:u64 }` and
resolves through the fixed [OwnerRecordOrdinal function](cohort.md#object-domains-and-claim-identities). Every record
embeds `queue_prev` and `queue_next` explicit optional forms of that handle plus
`queue_owned:u8` and seven zero bytes. The one `MaintenanceQueueIndexV1` in the
`maintenance_queue_index` layout row is
`{ head:OptionalMaintenanceRecordHandle, tail:OptionalMaintenanceRecordHandle,
length:u32, reserved_zero:u32 }`; its structural capacity is exactly
3,408,065 records. It has no separately allocated node array.

Under only the maintenance-queue mutex, first insertion changes queue_owned
zero to one, appends at tail and increments length. Request sequences are
globally increasing, so walking next from head is strictly increasing sequence
order. Removal repairs both neighbors/head/tail, zeroes both links and
queue_owned, then decrements length. Deferred-to-Pending or WaitingFaultIdle-to-
Pending first removes its old participation if still linked, assigns the new
sequence/ticket and appends at tail. Prepared, Pending, Claimed, Applying,
Awaiting and unacknowledged terminal states remain linked while queue_owned;
Free is never linked. Every traversal validates handle generation, reciprocal
prev/next, state/queue_owned, length and a maximum of length steps; a cycle,
duplicate, nonincreasing sequence or orphan queue-owned record is
TaggedStateCorruption. This append/remove algorithm and ascending walk are the
only queue/index representation.

A `RoundTicket` contains a unique checked round ID and the exact already-
reserved admission epoch to publish on reopen. Shutdown uses no RoundTicket.
Under the maintenance-queue mutex, records arriving before the
current cutoff is sealed join that ticket; records arriving after it join one
precreated next ticket. Creating either ticket checks epoch capacity before the
record is accepted. Reopen performs no counter arithmetic. The tagged phase and
record sequence are part of every CAS, so a delayed requester from an older
round cannot close a later Open epoch.

All maintenance/build reasons below are closed `repr(u8)` domains; the shown
numbers are their only ABI-v1 discriminants:

```text
HcqRejectReason = OverInstructionCap=1 |
    DisconnectedMandatoryEndpoints=2 | BackendSpillCapacity=3 |
    SegmentCapacity=4 | IslandCapacity=5
ActiveBuildCancelReason = QueueContention=1 | QueueFull=2 | CancelToken=3 |
    AttachedTo(BuildToken)=4 | AnchorConflict=5 | StaleInput=6 |
    Shutdown=7 | OwnershipRace=8 | PublicationRace=9
StaleNoOpReason = Superseded=1 | PreparationFailed=2 |
    PredicateChanged=3 | CoalescedHigherPrecedence=4 |
    SourceUnavailable=5 | TargetUnavailable=6 | GenerationChanged=7
MaintenanceFailureCode = Capacity=1 | StaleGeneration=2 | PlanConflict=3 |
    MappingRejected=4 | CounterExhausted=5 | InvalidRequest=6 |
    RangeTooLarge=7
LinkCleanupReason = Stale=1 | Superseded=2 | PreparationFailed=3 |
    Capacity=4 | SequenceExhausted=5 | Cancelled=6 | Shutdown=7 |
    PredicateChanged=8
AppliedDetail = RootsCut=1 | PressureRootsCut=2 | CutoverComplete=3 |
    Unlinked=4 | MappingCommitted=5 | VcpuInactive=6 | LinkInstalled=7
MaintenanceResult =
    Applied(AppliedDetail) |
    StaleNoOp(StaleNoOpReason) |
    FailedBeforeMutation(MaintenanceFailureCode) |
    CancelledBeforeCommit |
    CancelledByStop(TerminalCauseCode) |
    TerminatedAfterMutation(TerminalCauseCode)
```

The payload-bearing ActiveBuild `AttachedTo` variant includes its complete
nonzero BuildToken after discriminant 4. Unknown values are corruption, never
forward-compatible fallbacks. The allowed terminal MaintenanceResult set is:

| request | allowed ordinary terminal results |
|---|---|
| InvalidateToUnavailable | Applied(RootsCut), StaleNoOp, FailedBeforeMutation, CancelledByStop, TerminatedAfterMutation |
| PressureEvict | Applied(PressureRootsCut), StaleNoOp, FailedBeforeMutation, CancelledByStop, TerminatedAfterMutation |
| CutoverToSuccessor | Applied(CutoverComplete), StaleNoOp before publication, CancelledBeforeCommit, CancelledByStop, TerminatedAfterMutation |
| UnlinkOnly | Applied(Unlinked), StaleNoOp, FailedBeforeMutation, CancelledByStop, TerminatedAfterMutation |
| MappingChange | Applied(MappingCommitted), StaleNoOp, FailedBeforeMutation, CancelledByStop, TerminatedAfterMutation |
| VcpuDeactivate | Applied(VcpuInactive), CancelledBeforeCommit, CancelledByStop, TerminatedAfterMutation |
| InstallLink | Applied(LinkInstalled), StaleNoOp, CancelledByStop, TerminatedAfterMutation |

Any other result/variant pair is TaggedStateCorruption. InstallLink preparation
failure is StaleNoOp(PreparationFailed) only after its LinkCleanupTicket drains;
it never exposes FailedBeforeMutation. A post-publication Cutover or post-
Deactivating vCPU cannot take an ordinary failure row.

Each PatchRecord embeds one stable, generational `LinkCleanupTicketV1` slot:

```text
LinkCleanupTicketV1 {
    ticket_generation: u64,                 // nonzero, checked
    patch: PatchRecordHandle,
    expected_patch_generation: u64,
    destination: Fallback | Dead,
    reason: LinkCleanupReason,
    final_result: StaleNoOp(StaleNoOpReason) |
                  CancelledByStop(TerminalCauseCode),
    bridge: None | Some(UnitHandle),
    pin_bundle: None | Some(CleanupPinBundleHandle),
    page_snapshot_bundle: None | Some(PageSnapshotBundleHandle),
    span_reservation: None | Some(ExecutableSpanReservationHandle),
    ledger_credit: None | Some(LedgerReservationHandle),
    ownership_mask: u16,
    counted_awaiting: bool,
    reserved_zero: [u8; 5],
}
```

Ownership-mask bits are exactly 0 source UnitPin, 1 target UnitPin, 2 bridge
UnitPin, 3 page snapshot bundle, 4 bridge cleanup reference, 5 executable span,
6 metadata/registry slots, 7 bridge quota permit and 8 ledger credit; bits
9..15 must be zero. Optional handles are fixed-layout `{tag:u8,zero:[u8;7],
slot:u32,zero2:u32,generation:u64}` values into precharged generational storage.
The ticket owner clears a bit only after releasing that exact resource and
revalidates the patch/ticket generations at every authority transition. The
ticket is noncopyable protocol state even though its atomic payload is copied
between alternating slots.

MaintenanceRecord, not a reason bit, is authoritative. It contains its checked
generation/sequence/ticket, criticality, a typed immutable
`RootMutationRequest` with exact generational payload handles, one precharged
plan cell and a stable result cell. ClosingPrepare fills the plan cell with the
corresponding immutable `RootMutationPlan`; request code and Closed never build
or infer a plan. The record follows:

```text
Free(g) -> Prepared(g, ReplacingPublisher(BuildToken), sequence, ticket)
        -> Pending(g, sequence, ticket)
Free(g) -> Prepared(g, VcpuDeactivator(VcpuHandle, expected_vcpu_generation:u64,
                                       VcpuDeactivationToken), sequence, ticket)
        -> Pending(g, sequence, ticket)
Free(g) ---------------------------------------------> Pending(g, sequence, ticket)
Prepared(ReplacingPublisher) -> CancelledBeforeCommitAwaitingBuilder(BuildToken)
Prepared(VcpuDeactivator) -> CancelledBeforeCommit
Pending -> Claimed(round)
Pending(InstallLink) -> AwaitingLinkCleanup(LinkCleanupTicket, StaleNoOp)
Claimed -> Applying | StaleNoOp | FailedBeforeMutation(error)
        | Deferred(old_sequence, reserved_sequence, reserved_ticket)
Applying -> Applied | TerminatedAfterMutation(cause)
Applying(VcpuDeactivate) -> WaitingFaultIdle(old_sequence, retry_root_cut=true)
Prepared(ReplacingPublisher) -> CancelledByStopAwaitingBuilder(BuildToken, cause)
Prepared(VcpuDeactivator) -> CancelledByStop(cause)
Pending|Claimed|Deferred(InstallLink)
    -> AwaitingLinkCleanup(LinkCleanupTicket, final_result)
Pending(nonlink)|Claimed(nonlink)|WaitingFaultIdle -> CancelledByStop(cause)
Deferred(old_sequence, reserved_sequence, reserved_ticket)
    -> Pending(g, reserved_sequence, reserved_ticket)
WaitingFaultIdle(old_sequence, true) -> Pending(g, new_sequence, new_ticket)
CancelledBeforeCommitAwaitingBuilder(BuildToken) -> CancelledBeforeCommit
CancelledByStopAwaitingBuilder(BuildToken, cause) -> CancelledByStop(cause)
AwaitingLinkCleanup(LinkCleanupTicket, final_result) -> final_result
Applied|StaleNoOp|FailedBeforeMutation|CancelledBeforeCommit|CancelledByStop
    -> Free(g + 1)
```

`PreparedOwnerV1` is a closed `repr(u8)` union with discriminants
`ReplacingPublisher=1` and `VcpuDeactivator=2` and exactly the payloads shown in
the graph; tags 0 and 3..255 are invalid. Only ReplacingPublisher can own an
AwaitingBuilder obligation. Queue scans always branch on this owner tag before
performing a Prepared transition, and [Task 0](tasks/00-baseline.md) freezes the union layout and both
discriminants.

The closure driver is the only Claimed writer. Under the queue mutex it performs
every check which can produce StaleNoOp, FailedBeforeMutation, Deferred or
AwaitingLinkCleanup, then transitions the complete accepted set to Applying and
release-publishes the cohort batch marker exactly as specified under closure
arbitration. No record waits until its own first root store and no per-plan
Claimed-to-Applying edge exists. Before the collective marker, no root/guest-
visible mutation is legal; afterward every Applying record belongs to the same
no-fail tail. A fatal platform failure records the one common
TerminatedAfterMutation cause for that batch and proceeds only to terminal
process cleanup, never reopen. CancelledByStop and
TerminatedAfterMutation are exact result-cell outcomes, not silent queue
removal. `Deferred(old_sequence, reserved_sequence, reserved_ticket)` is
terminal only for acknowledgement of that old sequence/ticket participation;
it is not completion of the logical request
and cannot wake its result waiter. `WaitingFaultIdle` is the sole permitted
post-mutation continuation: only VcpuDeactivate can enter it, only after the
retry root is cut and its owner is woken, and it likewise closes the old
sequence without completing the request. The owner-side Idle publication wakes
the queue; reassignment creates a new sequence before either state returns to
Pending. Before publishing WaitingFaultIdle, the first plan releases every
traversal pin and unconsumed shadow/reservation and resets its plan cell; it
retains only stable record/result identity while the real PIC roots remain
unchanged. The continuation's next ClosingPrepare builds and revalidates a new
plan from current generations and may not reuse any old-ticket root list or
payload. No other Applying record may defer or wait across a handoff.

WaitingFaultIdle by itself is not work for another maintenance round. Its owner
first release-stores FaultSlot Idle, then takes the queue mutex, revalidates the
exact vCPU/slot/record generations and CASes WaitingFaultIdle to Pending only
after assigning a new checked request sequence and current/next RoundTicket; it
sets control and, if the validated phase is Open, prepares the payload and
attempts only the Open-to-Closing CAS while holding that mutex. It then unlocks
before driving or joining the round. If shutdown changed the record to
CancelledByStop, the owner performs
no reassignment. If request-sequence, RoundTicket, DriverToken or reopen-epoch
capacity is exhausted after the phase-one root cut, the owner cannot report an
ordinary pre-mutation failure: it calls `latch_terminal` while still preserving
WaitingFaultIdle, releases the queue and calls `drive_or_join_terminal`; entry
to Stopping changes that record to CancelledByStop and shutdown completes the
deactivation. Thus WaitingFaultIdle is never stranded and the coordinator does
not spin through empty rounds while
waiting for owner-side canonicalization.

An AwaitingBuilder or AwaitingLinkCleanup state is not a terminal result and cannot advance record
reuse. The coordinator may cancel admission and wake the builder, but never
returns that builder's span, ledger credit or OwnerCell reservations. Exactly
the thread owning the matching BuildToken performs those staged cleanups after
dropping every compilation UnitPin, then CASes AwaitingBuilder to its final
Cancelled state and wakes result/ack waiters. If the builder's initial
Prepared-to-AwaitingBuilder CAS loses to the coordinator's identical-token
state, it still performs that one cleanup; any other token is fatal corruption.
The record cannot become Free until this acknowledgement, so neither side can
double-clean or reuse it. Terminal shutdown joins all builders before arena
reclamation.

TerminalControl's two checked-u32 Awaiting counters make those ownership gaps
explicit. Both counters initialize to zero and
`awaiting_registration_closed` initializes false. Under the queue mutex, a
producer first inspects the exact record. If it already is the matching
Awaiting state with `counted=true`, the BuildToken/LinkCleanupTicket owner
adopts only its cleanup obligation and does not increment. If it is Prepared or
the permitted pre-mutation link state and registration is still open, local
RAII `AwaitingRegistrationGuard::new` checked-increments the matching counter,
prepares a payload with counted=true, and attempts the exact CAS. CAS failure
lets guard Drop decrement and reloops; CAS success calls the non-unwinding
`guard.commit_to_state()` which merely disarms Drop. The guard object never
lives in or is copied through atomic payload limbs. Encountering an eligible
pre-Awaiting state after registration is closed is terminal-scan corruption;
closure forbids creating a new obligation, not observing or consuming one
already published.

Before attempting the exact `Awaiting(..., counted=true)`-to-final CAS, the
owner constructs an `AwaitingConsumptionGuard` containing the record parking
word, the selected Awaiting counter and one armed count obligation. Under the
queue mutex it preflights that the record's checked parking sequence can advance;
exhaustion before the CAS is an unrecoverable identity failure and invokes raw
`SYS_exit_group(70)` because leaving an Awaiting count stranded cannot be
terminalized safely. CAS failure disarms the guard without decrementing and
retries from the observed state. CAS success enters one non-unwinding/no-fail
tail: release-increment the record parking sequence and futex-wake all result
waiters, AcqRel-decrement the selected Awaiting counter exactly once, futex-wake
all counter waiters on one-to-zero, then disarm the guard. The guard's Drop
performs precisely the uncompleted suffix and may use only those atomics/futex;
an unexpected syscall/invariant failure raw-exits 70. Thus neither unwind nor
an asynchronous terminal request can expose a final record with a permanently
positive Awaiting count. No other path decrements it. AwaitingBuilder is
structurally bounded by 262,144 family
cutover records and AwaitingLinkCleanup by 2,097,152 patch records, both below
u32::MAX. Reaching either bound implies duplicate registration and latches
InternalInvariant/TaggedStateCorruption; it is never ordinary capacity pressure
and cannot leave a new Awaiting state uncounted. Stopping first prevents new records, closes HCQ
admission, performs every queue-locked conversion which can create Awaiting and
then release-stores `awaiting_registration_closed = true` while still holding
the queue mutex. Only after that producer-closure point does it unlock and
join/drain owners; both counters are then monotonic toward zero. Terminal waits
directly on each nonzero AtomicU32 value with the standard snapshot/recheck
futex loop and finally acquire-scans every record. Underflow, a false counted
bit or double consumption is fatal corruption.

Every record has one queue-owned-reference bit plus a checked-u32
`requester_refcount`. The initial synchronous requester, while exclusively
owning a Free-to-Prepared claim under its PrequeueProducerToken, initializes
that count to one before publishing the record. Every later safety/pressure
joiner holds the maintenance-queue mutex, validates the exact visible
generation/sequence, checked-increments the count and only then unlocks; a
terminal/recyclable transition and new reference acquisition therefore cannot
cross. There is no validate/increment/revalidate sequence on separate atomics.
The count is bounded by the low count of `normal_result_gate`; its exhaustion
uses the same safety-wait/performance-fallback rule. Release may atomically
decrement outside the queue mutex and wakes after every decrement, but a last
drop which could recycle reacquires the queue mutex and revalidates both the
queue-owned bit and exact terminal state. The driver clears the queue-owned bit after it has advanced
acknowledgement past that record's participation. For an asynchronous record
while ShutdownRecord is Idle, that clear may perform the exact terminal-to-
`Free(g + 1)` CAS and wake record
waiters. A synchronous terminal record remains generation/sequence-addressable
until every requester has copied the result and decremented the count; whichever
operation observes both queue-owned false and requester_refcount zero performs
the same CAS only while shutdown remains Idle. Once shutdown is non-Idle, last
drop only decrements/wakes and leaves the terminal tag nonreusable for terminal
scan; it never needs g+1. AwaitingBuilder, AwaitingLinkCleanup, Deferred and
WaitingFaultIdle are not recyclable states. Generation increment
failure latches terminal before reuse; no scanner or best-effort garbage
collector recycles records.

RootMutationRequest and RootMutationPlan have the same variant tag, exactly one
of `InvalidateToUnavailable`, `PressureEvict`, `CutoverToSuccessor`,
`UnlinkOnly`, `MappingChange`, `VcpuDeactivate` or `InstallLink`. The request
fixes the initiating identities and desired result. The prepared plan
additionally fixes every intended dispatch result, unit/family generation,
exact affected object set, immutable predicate over the coordinator's
per-cohort FaultSlot snapshot, prepared payload/COW shadow, acquired traversal
UnitPin, memory-authority operation and continuation/result cell. There is no
choice between an embedded set and an implementation-selected slice.

The request-side representation is the following closed schema. All enums are
`repr(u8)`, all reserved bytes are zero, all handles are validated with their
full generation before dereference, and [Task 0](tasks/00-baseline.md) freezes every discriminant and
field offset:

```text
DependencyCause = ExplicitApi=1 | ObservedCpuWrite=2 | ExternalWriter=3 |
    MappingRemoved=4 | MappingPermissionChanged=5 | BackingReplaced=6 |
    AddressSpaceDestroyed=7
UnlinkReason = Superseded=1 | Invalidated=2 | Pressure=3 |
    SourceDestroyed=4 | TargetUnavailable=5 | AddressSpaceDestroyed=6
ExternalWriterKind = Interpreter=1 | Dma=2 | Gpu=3 | Loader=4 |
    Debugger=5 | Service=6
MappingOperationKind = InstallRam=1 | InstallMmio=2 | Remove=3 | Protect=4 |
    ReplaceBacking=5 | ArmExecutableObservation=6 |
    CompleteObservedCodeWrite=7 | ExternalWrite=8 | DestroyAddressSpace=9
MappingContinuationV1 = ReturnToApi=1 |
    ResumeCanonical { vcpu:VcpuHandle,
                      fault_record:FaultTransitionRecordHandle,
                      continuation:BlockKey }=2 |
    AddressSpaceDestroyComplete=3

BuildFingerprintHandleV1 {
    slot_index:u32, reserved_zero:u32, slot_generation:u64
}
MappingOperationHandleV1 = repr(C, align(8)), size 16 {
    address_space_slot:u32, cell_index:u8, reserved_zero:[u8;3],
    cell_generation:u64
}
MappingGenerationVectorHandleV1 = repr(C, align(8)), size 48 {
    block:MetadataBlockHandleV1, length:u32, reserved_zero:u32
}
WriteBufferHandleV1 = repr(C, align(8)), size 48 {
    block:MetadataBlockHandleV1, byte_len:u32, reserved_zero:u32
}
PressureSelectionHandleV1 {
    pressure_record_generation:u64,
    unit_bitmap_generation:u64, family_bitmap_generation:u64,
    selected_unit_count:u32, selected_family_count:u32
}
MappingGenerationEntryV1 {
    guest_page_index:u64, physical_page_id:u64,
    mapping_generation:u64, content_generation:u64,
    protection_generation:u64, observation_generation:u64
}
LinkVersionPredicateV1 {
    patch_generation:u64,
    source_code_version:u64, target_code_version:u64,
    source_segment_generation:u64, target_segment_generation:u64
}

RootMutationRequestV1 =
  InvalidateToUnavailable {
    victim:UnitHandle, expected_code_version:u64,
    expected_reachability:u64, cause:DependencyCause
  }=1 |
  PressureEvict {
    requested_total_bytes:u64, input_ledger_sequence:u64,
    selection:PressureSelectionHandleV1
  }=2 |
  CutoverToSuccessor {
    successor:HcqFamilyHandle, expected_successor_version:u64,
    build_token:u64, build_fingerprint:BuildFingerprintHandleV1,
    predecessor_count:u8, reserved_zero:[u8;7],
    predecessors:[OptionalHcqFamilyHandle;2]
  }=3 |
  UnlinkOnly {
    victim:UnitHandle, expected_code_version:u64, reason:UnlinkReason
  }=4 |
  MappingChange {
    address_space:AddressSpaceHandle,
    operation:MappingOperationHandleV1,
    expected:MappingGenerationVectorHandleV1,
    continuation:MappingContinuationV1
  }=5 |
  VcpuDeactivate {
    vcpu:VcpuHandle, expected_vcpu_generation:u64,
    deactivation_token:u64
  }=6 |
  InstallLink {
    patch:PatchRecordHandle, source:UnitHandle, target:UnitHandle,
    expected:LinkVersionPredicateV1,
    bridge:OptionalUnitHandle
  }=7
```

`predecessor_count` is one or two, entries below it are distinct and sorted by
HcqFamilyId, and every remaining optional entry is None/zero. `build_token` and
`deactivation_token` are their distinct nonzero newtypes, not interchangeable
u64 values in Rust. InstallLink requires its optional bridge, when present, to
be the exact PublishReady StaticBridgeUnit held by its cleanup ticket. Pressure
selection bit i corresponds only to registry slot i; the counts equal bitmap
popcount and all unused bitmap tail bits are zero.

Each address-space slot embeds two alternating fixed
`MappingOperationCellV1`s. The referenced cell contains its kind plus one exact
payload: InstallRam/InstallMmio/ReplaceBacking name a checked
`BackingHandle { backing_id:u64, backing_generation:u64 }`, backing-page offset,
guest-page start/count and final permission bits; Remove names that guest-page
range; Protect names the range, final permissions and dirty-tracking arm flag;
ArmExecutableObservation names one sorted PhysicalPageId/alias set;
CompleteObservedCodeWrite names the exact FaultTransitionRecord and
MemoryEffectPlan stage; ExternalWrite names writer kind, physical-page byte
range and an immutable charged `WriteBufferHandleV1`; DestroyAddressSpace has no additional
per-batch union payload and instead uses the progress record below. Permission bits are exactly Read=bit0, Write=bit1, Execute=bit2 and
bits 3..7 zero. Page counts are nonzero, byte ranges are nonempty and checked
inside the arena/backing object, and all unused union bytes are zero.
Both cells begin Free with generation one and the slot's `next_mapping_cell`
begins zero. Claim requires the permit and cell mutex, chooses exactly that
index, preflights its generation successor, changes it Free-to-Prepared and
toggles `next_mapping_cell ^= 1` only after publication. The handle's explicit
cell_index selects the cell; matching only generation or deriving parity from
generation is forbidden. A cell becomes Free(g+1) only after its exact result,
buffer/vector owners and requester references are consumed; terminal leaves it
nonreusable. Validation also requires the separately carried AddressSpaceHandle
slot/generation match the cell's immutable owning slot.

Long DestroyAddressSpace and ExternalWrite operations use the following
persistent progress embedded in their selected MappingOperationCell; no caller-
stack cursor is authority:

```text
MappingBatchProgressV1 = repr(C, align(8)), size 64 {
    operation_id:u64, batch_generation:u64,
    next_offset:u64, end_exclusive:u64, committed_prefix:u64,
    current_batch_start:u64, current_batch_length:u32,
    state:u8, unit:u8, reserved_zero:u16, terminal_code:u64
}
MappingBatchState = Idle=0 | Prepared=1 | AwaitingResult=2 |
    Complete=3 | Failed=4 | CancelledByStop=5
MappingBatchUnit = GuestPage=1 | Byte=2
ExternalWriteOutcomeV1 = {
    operation_id:u64, committed_prefix_bytes:u64,
    terminal:Complete=1 | Failed(MaintenanceFailureCode)=2 |
             CancelledByStop(TerminalCauseCode)=3
}
DestroyAddressSpaceOutcomeV1 = {
    operation_id:u64, removed_mapping_count:u64,
    terminal:Complete=1 | CancelledByStop(TerminalCauseCode)=2
}
```

`operation_id` and every reused `batch_generation` are checked nonzero-u64
identities reserved with their successors before the first state change.
Destroy first changes the registry slot to Destroying and, under its memory-
authority mutex, snapshots `end_exclusive` as one past the highest currently
mapped guest page (zero for empty), with next_offset/committed_prefix zero and
GuestPage unit. Because Destroying rejects all new mapping operations, each
batch may deterministically walk the mapping authority from next_offset and
freeze the lowest at most 2,097,152 extant entries into its expected vector.
An empty suffix completes. A successful batch adds its exact entry count to
committed_prefix and advances next_offset to one past its last guest page; a
failure cannot advance either. It publishes Complete and terminalizes the
address-space slot only after the final suffix.

ExternalWrite initializes Byte unit, next_offset/committed_prefix zero and
end_exclusive to the checked requested byte length. While holding the one
MappingRequestPermit across the outer operation, it copies the next
`min(16,777,216, end-next)` bytes into a fresh pinned WriteBuffer, records the
exact start/length, and submits one ExternalWrite MappingChange. On success it
checked-adds exactly the memory plan's reported committed-prefix bytes; a full
chunk also advances next_offset by current_batch_length. A partial or failed
chunk publishes Failed with next_offset unchanged and returns
ExternalWriteOutcomeV1; the outcome's prefix is the sum of all prior complete
chunks plus the permitted visible prefix of the terminal chunk. Complete
requires both offsets equal end_exclusive. Cancellation returns the same stored
prefix. Prepared/Awaiting state, buffer owner and exact MaintenanceRecord handle
remain in the cell across every wait/handoff and are cleared exactly once after
the result is copied. A retry validates operation/batch IDs and never repeats a
completed chunk. These outer operations do not promise atomicity across
batches; all other mapping kinds require one vector/buffer within the stated
maxima and return RangeTooLarge otherwise.

The expected vector is sorted unique by guest_page_index and contains every
page/alias generation the operation predicates. It is immutable, charged and
pinned by the request before enqueue; its u32 length is the exact operation
scope, and insufficient metadata capacity returns FailedBeforeMutation(Capacity)
before any memory effect. ClosingPrepare creates the same-length resulting
vector in the pre-reserved plan cell, reserves every successor generation and
revalidates exact equality with expected. Thus neither request nor driver can
rescan an unspecified range, invoke a callback or substitute the current
mapping after enqueue. MappingResult is the closed union
`Committed { resulting:MappingGenerationVectorHandleV1 } |
Stale { first_mismatch_index:u32 } | Rejected { code:MaintenanceFailureCode }`;
the MaintenanceResult row fixes whether that value may be exposed.

Both vector and write-buffer wrappers resolve directly through their embedded
MetadataBlockHandle; there is no extra slot registry. Vector validation requires
PagedPayload type MappingGenerationVectorV1 and
`block.byte_len == length*size_of(MappingGenerationEntryV1)` with checked
arithmetic. Write-buffer validation requires its dedicated PagedPayload type,
nonzero `byte_len == block.byte_len <= 16,777,216`. The request owns one
MetadataBlockPin for each input block from before record publication through
the last preparation/read; the prepared plan owns the resulting-vector pin
until MappingResult ownership is transferred to the caller on Committed or
freed on every other result. Copying a synchronous Committed result transfers
that one pin into the returned typed owner; dropping that owner retires the
block normally. An ExternalWrite record releases its input buffer pin only
after the exact operation reaches a terminal result. Shutdown cancellation
uses the same owner paths and cannot free a block merely from its bare handle.
