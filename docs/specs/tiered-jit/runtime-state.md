# Tagged state cells, keys and dispatch

[Specification index](README.md) · [Implementation tasks](tasks/README.md) · [Open review items](review-status.md)

Related contracts: [Code units, versions, registries and admission](units-and-registries.md); [Shutdown requests, result ownership and wait protocols](shutdown-and-waits.md).

## Runtime data model

## Portable tagged-payload state cells

No conceptual state tuple in this document requires a 128-bit atomic. Except
for the separately specified phase word, packed FaultSlot word and one-way
ShutdownRecord word, every state
whose tag is atomically observed together with a full-width generation, token,
cause or handle uses the following `TaggedPayloadCell` representation:

- `state_word: AtomicU64` stores a four-bit tag in bits 0..3 and a nonzero,
  checked, strictly increasing 60-bit `publication_sequence` in bits 4..63;
- two fixed payload slots, selected by `publication_sequence & 1`, contain only
  fixed-size atomic integer/pointer limbs and an `AtomicU64 stamp`. A ready
  stamp is `publication_sequence << 1`; the low bit set means busy. A
  variable-sized immutable object is
  represented only by a generational handle to precharged stable storage;
- the named writer mutex for the object serializes every payload writer. The writer
  requires `next_sequence = current_sequence + 1`, AcqRel-swaps the inactive
  slot's stamp to `(next_sequence << 1) | 1`, release-writes every atomic
  payload limb, release-stores `next_sequence << 1`, and then compare-exchanges the
  exact old `state_word` to the new tag/sequence with Release success and
  Acquire failure ordering;
  and
- a reader acquire-loads `state_word`, acquire-reads the selected slot between
  two identical ready stamps equal to `publication_sequence << 1`, executing
  an acquire load for every payload limb and an acquire fence before the second
  acquire stamp load, then
acquire-loads the identical word again. It retries on any mismatch. It never
reads a non-atomic object concurrently with mutation.

The Release/Acquire ordering of every limb is mandatory, not an optimization
choice. If a slow reader observes any changed limb from a reuse two
publications later, that acquire synchronizes after the later busy-stamp store;
its second stamp observation therefore cannot legally accept the older ready
stamp. [Task 1](tasks/01-abi.md)'s loom/litmus suite pauses a reader between each pair of loads
while two same-parity slot reuses complete and proves that it returns either
complete old payload, complete new payload or Retry, never a mixture. A model
with only one intervening publication is insufficient.

Word/stamp coherence does not by itself extend object lifetime. The exhaustive
container guards are: VcpuStateCell uses ArenaAdmissionToken, NormalResultToken
or TerminalDriverGuard; FamilyLifecycleCell uses FamilyPin, the JIT-state mutex
or TerminalDriverGuard; LcqBuildCell uses a DispatchSlot owner/queue/waiter
reference or an active execution epoch; ActiveBuildStateCell uses its exact
initialization/owner/queue/worker/attach pin; MappingRequestPermitCell uses its
permit-owner reference plus NormalResultToken; MaintenanceRecordCell uses its
queue-owned bit, requester reference, PrequeueProducer owner or DriverGuard;
ReclaimControlCell uses its checked requester reference plus NormalResultToken,
its unique Running-owner reference or TerminalDriverGuard;
PatchRecordCell uses its source/patch-registry reference, LinkCleanupTicket or
sealed-plan pin; and UnitLifecycleCell uses UnitPin, directory/root/cleanup-
owner reference or the JIT-state mutex. TerminalDriverGuard keeps every backing
slab alive until the terminal unmap. No JitProcessHandle alone pins a reusable
slot. An indirect payload handle is not dereferenced until
the reader acquires its generation-matched pin/reference under the owning
authority and then revalidates the identical state word and payload; mismatch
drops the new reference and retries. Reclamation first publishes a
nonacquirable tag, waits the corresponding count to zero, and only then reuses
payload storage. A lock-free read followed later by an unguarded refcount
increment is forbidden.

Construction relaxed-initializes both slots and word while unreachable: slot 1
contains the complete initial payload with ready stamp `1 << 1`, slot 0 has
stamp zero, and `state_word = (1 << 4) | initial_tag`; release-publication of
the containing object is the first reader-visible event. An “exact-state CAS”
below therefore means: hold the named writer mutex,
validate the complete old word and validated payload, prepare the other slot,
and CAS the word. Only the word is the commit point; a failed CAS leaves the
prepared slot unreachable. The 60-bit publication sequence is ABA protection,
not a semantic generation and is never copied into a key. Normal transitions
may use only `1..=2^60-17`; the final sixteen values are reserved solely for
process-Stopping/exhaustion cleanup. Ordinary cancellation, rejection and
return-to-Free consume ordinary sequential values and may be reused normally;
a cell which enters the reserved range is never reused. Exhaustion before a normal transition
latches the process terminal cause before mutation and never wraps. The
longest terminal path is UnitLifecycle's `Published -> Superseded ->
UnlinkPending -> Unlinked -> Retired -> DirectoryDetached -> Reclaimed`;
including entry into terminal handling and one cleanup-failure marker consumes
at most nine publications. Every other state graph consumes at most six. Task
0 exhaustively explores each graph and fails if a terminal path needs more
than sixteen; terminal code may not invent a collapsed edge absent from that
graph. [Task 0](tasks/00-baseline.md) generates the fixed slot layouts and offset assertions; [Task 1](tasks/01-abi.md)
supplies a loom model for a writer, two readers, reuse and every reserved
terminal transition.

There are exactly two tag-only exceptions which neither change payload nor
publication sequence. First, the unique replacing publisher may CAS
`MaintenanceRecord Prepared(g, ReplacingPublisher(BuildToken), sequence,
ticket)` to the identical
payload tagged Pending while it holds the matching OpenToken and JIT-state
mutex. The coordinator may concurrently CAS Prepared to its builder-cancel
Awaiting state under the queue mutex; exactly one word CAS wins. Pending wins
before any unit-commit mutation and the OpenToken prevents ClosingFreeze until
that commit finishes; cancellation wins before mutation and obligates builder
cleanup. The winning CAS release-increments the record's parking sequence and
wakes waiters after publishing the tag; the sequence is only a notification
hint. This exception does not take the cell's normal writer mutex and exists
solely to avoid nesting the maintenance-queue and JIT-state mutexes. No other
publisher transition is legal. Second, a deactivation owner which already
reserved its exact `Prepared(g, VcpuDeactivator(VcpuHandle,
expected_vcpu_generation:u64, VcpuDeactivationToken), sequence, ticket)` record may CAS it
to Pending while
holding the slot-registration mutex and matching OpenToken immediately after
Active-to-Deactivating. Its payload already contains the request sequence and
RoundTicket, and the drained OpenToken excludes coordinator cancellation. It
uses the same parking/control publication. No other tag-only transition is
legal.

The following tag values and writer authorities are exhaustive:

```text
VcpuStateCell (slot-registration mutex):
  0 Inactive, 1 Active, 2 Deactivating,
  3 TerminalDeactivating, 4 TerminalInactive
FamilyLifecycleCell (JIT-state mutex):
  0 CurrentStable, 1 CurrentPendingCutover, 2 ReplacementReserved,
  3 CutoverOld, 4 Withdrawing, 5 FamilyRetired, 6 FamilyReclaimed
LcqBuildCell (JIT-state mutex):
  0 Idle, 1 Building, 2 Published, 3 Failed, 4 Stale
ActiveBuildStateCell (its stable per-cell `active_build_writer_mutex`; guest
  admission uses try_lock and treats contention as transient):
  0 Free, 1 Initializing, 2 AdmissionReserved, 3 Queued, 4 Building,
  5 Published, 6 Rejected, 7 Cancelled
MappingRequestPermitCell (memory-authority permit mutex):
  0 Free, 1 Owned, 2 Terminal
ReclaimControlCell (its process-stable `reclaim_control_mutex`):
  0 Free, 1 Running, 2 Applied, 3 Failed, 4 CancelledByStop
MaintenanceRecordCell (maintenance-queue mutex):
  0 Free, 1 Prepared, 2 Pending,
  3 CancelledBeforeCommitAwaitingBuilder, 4 Claimed, 5 Applying,
  6 StaleNoOp, 7 FailedBeforeMutation, 8 Deferred, 9 Applied,
  10 TerminatedAfterMutation, 11 WaitingFaultIdle,
  12 CancelledByStopAwaitingBuilder, 13 AwaitingLinkCleanup,
  14 CancelledBeforeCommit, 15 CancelledByStop
PatchRecordCell (maintenance-queue mutex):
  0 Fallback, 1 InstallPending, 2 InstallCleanupPending,
  3 Installed, 4 UnlinkPending, 5 Dead
UnitLifecycleCell (JIT-state mutex):
  0 Staging, 1 PublishReady, 2 Published, 3 Superseded,
  4 Invalidating, 5 UnlinkPending, 6 Unlinked, 7 Retired,
  8 DirectoryDetached, 9 Reclaimed, 10 Aborted
```

Each cell payload stores every parameter shown in its state diagram, including
the full checked-u64 semantic generation, BuildToken, ReachabilityVersion,
admission epoch, request ID, cause and `was_counted_active` flag as applicable.
Free-to-Initializing is serialized by that mutex and publishes only the new
cell generation and BuildToken; the winner alone fills the immutable identity
payload before publishing AdmissionReserved. An external cancellation of
Initializing or Building locks the writer mutex, revalidates the exact
generation/BuildToken and stores that token in
`cancel_build_token: AtomicU64`; zero means no request.
Free-to-Initializing resets it to zero under the same mutex before commit, and
an owner treats cancellation as requested only when the value equals its own
BuildToken. The unique owner writes the terminal outcome and advances the cell;
a delayed canceller cannot affect a reused cell. ActiveBuild semantic-anchor
mutation separately takes the JIT-state mutex; pending-queue insertion/removal
separately takes the pending-queue mutex. None is held with the cell mutex.
Admission publishes Queued before trying the queue mutex, then under that mutex
pushes only after revalidating the exact Queued word and
`cancel_build_token != BuildToken`; otherwise it terminalizes without pushing. A cancellation
may remove an existing queue handle under the queue mutex and, after unlocking,
terminalize the still-Queued cell under its writer mutex. The queue may
therefore briefly contain a stale generational handle, which a worker discards
after acquire validation, but never a handle to reused storage. MaintenanceRecord,
PatchRecord and UnitLifecycle use the same representation and the exact tags
above; [Task 0](tasks/00-baseline.md) verifies/emits those values and never assigns alternatives. A
payload field is never a second state authority.

## Keys and dispatch publication

The initial `BlockKey` is exactly:

```text
address_space_id: AddressSpaceId
guest_pc: u64                         // must be four-byte aligned
arena_size: u64                       // immutable for this address space
cpu_profile_id: CpuProfileId          // Switch1=1, Switch2=2
fpcr: u32                             // complete architectural value
backend_abi_version: u32              // NativeFrame/NixeFast version, initially 1
fork_api_version: u32                 // exactly NIXE_FORK_API_VERSION
host_isa: u8                          // x86_64=1, aarch64=2
host_feature_bits: [u64; 2]           // Task 0 assigned bit positions
```

The process compiler fixes host ISA/features; the gateway constructs the key
from current canonical PC/FPCR and immutable process/address-space fields. An
FPCR write is a canonical boundary, so fast-linked units always have equal
FPCR. Adding another backend-visible architectural mode requires a new checked
field, StableEncode update, ABI-version bump and [Task 0](tasks/00-baseline.md) manifest/proof update;
it cannot be hidden in mutable compiler state.

`ExecutionKey` is exactly BlockKey without `guest_pc`; all keys in one CodeUnit
must share it.

Mapping and code generations are not hidden in the key. They are explicit
version checks. `InstructionKey` is exactly `ExecutionKey` plus
`instruction_pc: u64`, which must be four-byte aligned; HCQ overlap is
arbitrated over these keys, not block roots or address hulls.

The remaining link identities are exactly:

```text
UnitHandle {
    slot_index: u32,
    reserved_zero: u32,
    slot_generation: u64,             // nonzero
}
HcqFamilyHandle {
    slot_index: u32,
    reserved_zero: u32,
    slot_generation: u64,             // nonzero
}
PatchRecordHandle {
    slot_index: u32,
    reserved_zero: u32,
    slot_generation: u64,             // nonzero
}
DispatchSlotHandleV1 {
    slot_index: u32,
    reserved_zero: u32,
    slot_generation: u64,             // nonzero
}
ActiveBuildHandleV1 {
    cell_index: u32,
    reserved_zero: u32,
    cell_generation: u64,             // nonzero
    build_token: u64,                 // nonzero BuildToken
}
CompactOptionalActiveBuildHandleV1 {
    cell_index_plus_one: u32,          // zero means None
    reserved_zero: u32,
    cell_generation: u64,             // zero iff None
    build_token: u64,                 // zero iff None
}
PicRootHandleV1 {
    slot_index: u32,
    reserved_zero: u32,
    slot_generation: u64,             // nonzero
}

ExitSiteKey {
    source_unit: UnitHandle,
    source_code_version: CodeVersion,
    exit_site_id: ExitSiteId,          // process-unique, nonzero
    exit_site_ordinal: u32,            // zero-based final-code order in unit
    source_instruction_pc: u64,        // four-byte aligned guest PC
    edge_kind: EdgeKind,
    exit_state_map_id: ExitStateMapId,
}

BridgeKey {
    host_isa: u8,
    host_feature_bits: [u64; 2],
    backend_abi_version: u32,
    fork_api_version: u32,
    landing_kind: NativeLandingKind,   // IndirectJump for every PIC bridge
    source_site: ExitSiteKey,
    target_block: BlockKey,
    target_reachability: ReachabilityVersion,
    target_code_version: CodeVersion,
    target_contract_id: FastEntryContractId,
}

StaticBridgeKey {
    source_code_version: CodeVersion,
    patch_record: PatchRecordHandle,
    target_code_version: CodeVersion,
    source_map_id: ExitStateMapId,
    target_contract_id: FastEntryContractId,
    landing_kind: NativeLandingKind,
}

HelperIslandKey {
    helper_id: HelperId,
    helper_binary_sha256: [u8; 32],
    landing_kind: NativeLandingKind,   // IndirectCall
}

PicProbeKey {
    source_site: ExitSiteKey,
    target_block: BlockKey,
}

ObservedSuccessor {
    target_block: BlockKey,
    edge_kind: EdgeKind,
}

ParticipatingFamily {
    family_id: HcqFamilyId,
    family_version: u64,
    code_version: CodeVersion,
}

BoundaryObservationKey {
    source: BoundaryEndpointIdentity,
    target: BoundaryEndpointIdentity,
}

CanonicalBlockRef {
    lcq_unit: UnitHandle,
    leader: InstructionKey,
}

MemberOriginKey {
    origin_block: BlockKey,
    lcq_code_version: CodeVersion,
    lcq_unit: UnitHandle,
}
```

`NativeLandingKind` is exactly `DirectOnly`, `IndirectJump`, `IndirectCall` or
`IndirectJumpOrCall`, with [Task 0](tasks/00-baseline.md) manifest discriminants. It describes how host
control can reach the first byte and therefore whether BTI/IBT landing bytes are
mandatory; it is part of every bridge/island key and StableEncode, not inferred
after allocation.

`EdgeKind` is exactly `Sequential`, `DirectTaken`, `ConditionalTaken`,
`ConditionalFallthrough`, `DirectCall`, `IndirectBranch`, `IndirectCall` or
`Return`; its checked discriminants are in the [Task 0](tasks/00-baseline.md) manifest. A resource cut
uses `Sequential`; a forced asynchronous control transition is not an edge
sample. Final-code exits are assigned ordinals in increasing exported native
offset, breaking an impossible equal-offset tie by increasing guest PC and then
EdgeKind discriminant. The verifier requires a one-to-one mapping between every
ordinal/ExitSiteId and one exported exit record; `source_instruction_pc` and
`edge_kind` must equal that record. Distinct source exits never share an
ExitSiteKey even when their maps and transfer semantics are byte-identical.

`CodeVersion`, `ReachabilityVersion`, `ExitSiteId`, `ExitStateMapId` and
`FastEntryContractId` are distinct nonzero-u64 newtypes, not interchangeable
integers. A BridgeKey's platform/ABI fields must equal those in both source and
target execution keys. Its source map ID must resolve in `source_unit` at
`source_code_version`, and its target contract ID must resolve in the selected
target at `target_code_version`; construction rejects rather than normalizes a
mismatch. These declarations are the field order consumed by StableEncode.
PicProbeKey, rather than an anonymous tuple, is the exact input to the private
PIC set hash. ObservedSuccessor is the element type of admission successor
arrays, and ParticipatingFamily is the element type of reshape family arrays.

The dispatch index is one JIT-state-mutex-protected Robin-Hood open-addressed
array of exactly 2,097,152 buckets and admits at most 1,048,576 live slots. A
bucket contains stable hash plus generational slot handle; BlockKey equality is
checked in the slot. Lookup starts at `hash & 0x1f_ffff`, uses ordinary
Robin-Hood probe-distance termination and is cold gateway/fallback work.
Deletion uses backward shifting, so there are no tombstones or resize/rehash
decisions. Construction precharges the bucket array. Reaching the live-slot
ceiling makes an HCQ-only new entry transient and an LCQ-required new entry a
precise `DispatchCapacityExceeded`; existing entries remain usable.

The bounded dispatch index maps BlockKey to a generational DispatchSlot. A slot
has exactly these logical fields: immutable `BlockKey`; checked nonzero u64 slot
generation; one pointer-sized `AtomicPtr<DispatchPayload>`; one coupled charged
`SafetyPayloadCell`; a generational `LcqBuildCell`; an `AtomicU64`
HCQ-anchor token where zero means unowned; and bounded waiter/queue reference
counts. A payload is an immutable, metadata-arena allocation containing:

- a globally unique ReachabilityVersion;
- the current LCQ entry, when resident;
- the preferred entry, which is the current HCQ entry when one exists and the
  LCQ entry otherwise; and
- the current HCQ family/version identity and complete BuildFingerprint, when
  present.

A publisher initializes the complete payload and all referenced objects, then
release-stores its pointer once. `DispatchAvailability` is the closed
`repr(u8)` domain `Available=0 | Unavailable=1 | TerminalUnavailable=2`.
`DispatchPayloadV1` is a fixed `repr(C)` record
containing `{reachability: u64, availability:DispatchAvailability,
reserved_zero:[u8;7], lcq: OptionalUnitHandle, preferred:
OptionalUnitHandle, hcq_family: OptionalHcqFamilyHandle,
build_fingerprint: OptionalBuildFingerprintHandle}`; option wrappers are
explicit tag-plus-zeroed-payload structs, not Rust niche layouts. Variable
fingerprint/member data is immutable generational storage referenced by the
handle. `SafetyPayloadCellV1` is `repr(C, align(64))` and contains exactly two
`DispatchPayloadV1` slots, two AtomicU64 ready/busy stamps and one AtomicU32
selected-slot generation plus explicit zero padding emitted by [Task 0](tasks/00-baseline.md). Its
manifest layout row records exact size, alignment and field offsets and the
DispatchSlot embeds exactly one such cell. It is not a root and stores one
complete replacement payload. A safety transition fills it before
mutation; after the Closed epoch drain, the displaced payload storage becomes
the next cell before reopen. Ordinary Open publication allocates its new
payload without consuming this cell, so every live slot always enters Open with
one safety replacement available even at the 640 MiB hard limit.

Available requires at least one LCQ/preferred UnitHandle and all optional
fields consistent with the named family. Unavailable has every optional handle
None and is a normal versioned result. TerminalUnavailable also has every
optional handle None, is constructed in the existing precharged
SafetyPayloadCell with the slot's current ReachabilityVersion, and may be
published only by TerminalDriverGuard after Stopping. It allocates no payload,
identity or epoch successor, is never reopened/reused and exists solely to cut
terminal admission safely when ReachabilityVersion is exhausted.

A reader first publishes its execution epoch,
uses the JIT-state mutex only to obtain the matching generational slot handle,
releases it, then acquire-loads the pointer and validates the slot generation.
The active epoch protects both slot and payload after lookup. Entry address,
CodeVersion and ReachabilityVersion are
therefore one coherent publication without a multiword atomic. Replaced
payloads are retired at the replacement execution epoch and remain allocated
until reader quiescence. The slot and its payload are not embedded in resolved
static links.

The LcqBuildCell is the state machine defined in [LCQ compilation](lcq.md); its state word and payload
name the exact semantic build-cell generation. HCQ admission CASes `hcq_anchor` from zero to its nonzero
BuildToken, then acquire-reloads the slot generation and payload and retains
the token only if the captured ReachabilityVersion is unchanged; otherwise it
CASes only its exact token back to zero. Payload replacement/invalidation
cancels or transfers the exact anchor before publishing a different
ReachabilityVersion. This avoids an assumed 128-bit atomic tuple.

DispatchSlot addresses are not process-lifetime ABI. Source fallbacks embed a
BlockKey and resolve it again; queued work carries keys and generational
handles. A slot is removed and epoch-retired when it has no resident entry,
compile/reservation state or queued job. A fallback does not pin a slot. This
prevents dynamic code and self-modifying workloads from growing an unbounded
node arena. Tier heat lives only in the fixed per-vCPU tables and generated
code never loads it. Under the JIT-state mutex, removal first clears/cancels the
two build states, backward-shift-removes the bucket and release-publishes the
null payload with the [epoch protocol](epochs-and-shutdown.md#epoch-reclamation). Only after payload/slot grace and
zero waiter/queue references is the checked slot generation advanced and the
storage reused with a new key.
