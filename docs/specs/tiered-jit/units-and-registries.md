# Code units, versions, registries and admission

[Specification index](README.md) · [Implementation tasks](tasks/README.md) · [Open review items](review-status.md)

Related contracts: [Tagged state cells, keys and dispatch](runtime-state.md); [Executable cache, metadata and backend ownership](cache.md); [Epoch reclamation and terminal teardown](epochs-and-shutdown.md).

## Code units and versions

Every runtime identity or semantic generation in this architecture is a
checked u64. Zero means absent/uninitialized; allocation begins at one and a
published value is never reused. Process-owned monotonic authorities allocate
AddressSpaceId, ReachabilityVersion, CodeUnitId, CodeVersion, HcqFamilyId,
BuildToken, FaultRecordId, ExitSiteId, ExitStateMapId and
FastEntryContractId, DriverToken,
VcpuDeactivationToken,
CodeUnit creation sequence, global invalidation/instruction-observation cursor,
global code epoch, admission/round epoch and maintenance request sequence.
Address-space or memory-authority owners allocate PhysicalPageId and mapping,
content, protection and observation generations. Each reusable
dispatch/unit/family/bridge/active-build/vCPU slot and executable segment owns a
u64 generation which is checked-incremented only after its preceding object has
passed the specified grace period. Family version and reservation generation
are likewise monotonically incremented by their owning family/slot. The owner
mutex named by this specification, or a single compare-exchange loop for an
atomic owner, is the sole allocation authority for each counter.

There is one process-global `global_hcq_shape_version: AtomicU64`, initialized
to one, whose sole writer holds the JIT-state mutex. It preflights and advances
exactly once in any commit which publishes/removes an LCQ, changes stable
InstructionKey ownership or changes current-family identity; an executable-
content observation-cursor change alone does not advance this version. Several
shape changes in one Open publication or one Closed transaction consume one
version. The writer reserves the successor before the first shape mutation,
performs every such mutation while admission is excluded, and
release-stores the reserved next version before unlocking; admission and
ownership snapshots acquire-load it inside their one JIT critical section and
publication revalidates exact captured objects rather than requiring an
unrelated later global version to remain equal. Reservation-only OwnerCell
changes do not advance the version and cancel their exact conflicting worker.
Exhaustion latches IdentityExhausted before
the first associated mutation. No memory authority allocates or aliases this
version, and every `HcqShapeVersion` in this document means this field.
Terminal root/owner/registry destruction after Stopping does not request or
publish another HcqShapeVersion: admission can never reopen and no future
snapshot exists. This is the sole shape-version mutation exception and lets
shape-version exhaustion cleanly shut down.

There is separately one process-global
`global_instruction_observation_cursor: AtomicU64`, initialized to one. It is
the sole value meant by “global invalidation cursor”, “executable-content
observation cursor” and `instruction_observation_cursor` throughout this
document; those phrases do not name three counters. Its sole writer owns
`MemoryTransactionGuard`. Before the first mapping, backing identity, guest
permission, physical-alias, page-content, protection-reason or observation-
generation mutation which can affect instruction fetch/leases, the writer
checked-reserves exactly cursor+1. It performs every object-local generation
and byte/PTE mutation, then release-stores that reserved cursor as the final
transaction publication before dropping the guard. Several such changes in
one MappingChange/observed-write transaction consume one cursor value. The two
explicit non-Closed local repair paths either do not affect executable content
and leave the cursor unchanged, or, for last-lease observation-reason
replacement, take this same guard and publish a successor before returning.
Exhaustion latches IdentityExhausted before any effect and never wraps.

Normal observation-cursor and per-page protection/observation generations use
only `1..=u64::MAX-1`; u64::MAX is reserved for the one terminal step-6 memory
batch. That batch, after admission/executors are permanently closed, writes the
reserved value to every surviving affected protection/observation generation
and release-stores the cursor u64::MAX once, without requesting a successor.
Mapping/content generations which terminal does not change retain their last
value. No post-terminal reader can acquire them. Exhaustion at MAX-1 can thus
enter typed shutdown and reach Applied rather than failing its own mandatory
protection recomputation.

The normal successor operation is one generated checked function:
`normal_observation_successor(R) = Some(R+1)` exactly when
`1 <= R <= u64::MAX-2`; zero, `u64::MAX-1` and `u64::MAX` return `None`.
The cursor is initialized to one. A `None` result latches
IdentityExhausted before changing any mapping, byte, root, pointer or local
generation. Consequently no normal commit ever stores `u64::MAX`; the terminal
step named above is its unique writer. Every phrase in this document such as
“reserve R+1” or “has a successor” invokes this function and therefore excludes
`R == u64::MAX-1`.

Final executable-lease acquisition takes MemoryTransactionGuard, copies and
pins the complete page/alias snapshot, acquire-loads this cursor after all
object-local generations, and returns that exact value before releasing the
guard. Admission/ownership snapshots acquire-load it inside their documented
JIT critical section only as fingerprint evidence; they never hold both
mutexes. A later cursor change does not alone cancel a build: preparation and
publication revalidate every captured page, alias, byte and generation. The
cursor is not a substitute for those predicates, does not advance
`global_hcq_shape_version`, and no JIT-state writer may modify it.

`DriverToken` is a distinct nonzero-u64 newtype. Each RoundTicket reserves one
fresh DriverToken before its request can be accepted; the successful
Open-to-Closing CAS returns `DriverGuard { phase_sequence, driver_token }` to
that exact caller, and every later driver action validates both fields. The
ShutdownRecord reserves its separate terminal DriverToken at process
construction, so identity exhaustion cannot prevent terminal entry. Tokens are
never reused. A normal DriverToken allocation failure occurs before mutation,
latches terminal intent and consumes only the already-reserved shutdown token.

Normal global-code epochs occupy `1..=u64::MAX-1`; `u64::MAX` is the terminal
Stopped marker and is never a retirement epoch. Every other identity may use
`u64::MAX` once but cannot request a successor, except for the explicitly
reserved terminal observation/protection values above. An operation preflights or
reserves every value it needs before its first shared-state or guest-visible
change. Exhausting an HCQ-only identity atomically disables new HCQ admission,
cancels exact pending builds and leaves installed code/LCQ correct. Exhausting
an address-space ID rejects creation; exhausting a mapping/content/protection
generation rejects that transition before memory changes; exhausting an
LCQ/reachability/lifecycle identity reports the named checked-capacity failure
before publication. Shutdown needs no successor epoch: during Stopping it
blocks admission, drains all executors/pins and reclaims, then release-stores
`Stopped(cause)` and the reserved terminal marker. Nonsemantic per-vCPU sample and
negative-cache creation sequences alone use the explicit clear-and-restart
rules below. No hash, pointer, Rust discriminant or truncated counter is an
identity substitute.

The internal `phase_sequence`, every
`TaggedPayloadCell.publication_sequence` and FaultSlot's 60-bit transition
sequence are ABA/publication mechanisms rather than runtime identities or
semantic generations. Their widths, normal ranges and reserved exhaustion
paths are the only packed-word exceptions. They are never encoded into a key,
used as a guest-visible result or substituted for the full checked-u64 identity
stored in the corresponding payload.

A CodeUnit is split into immutable `CodeUnitMetadata` and mutable
`UnitLifecycle`. `CodeUnitMetadata` is immutable after construction and owns:

- tier and backend ABI version;
- exact BlockKeys and InstructionKeys;
- canonical and fast native entry offsets;
- entry live-in contracts and exit dirty/live-out maps;
- exact copied instruction image;
- mapping and physical code dependencies;
- relocations, direct-link patch sites and source-local fallbacks;
- native-fault records;
- executable allocation and segment identity.

`UnitLifecycle` lives in the generational unit-registry slot and is the only
mutable state associated with that metadata. Publication and retirement epochs
belong to it, not to `CodeUnitMetadata`.

An HcqFamily owns the active optimized InstructionKeys and current CodeVersion
and has this separate checked lifecycle:

```text
new ordinary family: CurrentStable
new reshape family:  CurrentPendingCutover(predecessor handles)
CurrentStable -> ReplacementReserved(BuildToken)
ReplacementReserved -> CurrentStable                    [exact-token abort]
ReplacementReserved -> CutoverOld(successor handle)     [publication]
CurrentPendingCutover -> CurrentStable                   [all old roots cut]
CurrentStable|ReplacementReserved|CurrentPendingCutover -> Withdrawing
CutoverOld|Withdrawing -> FamilyRetired(retirement_epoch) -> FamilyReclaimed
```

A reshape names exactly one or two distinct CurrentStable family generations
and acquires their replacement leases in ascending HcqFamilyId order. Its
successor owns selected InstructionKeys immediately but remains
CurrentPendingCutover and cannot itself be reshaped until every predecessor
root is cut. CutoverOld is callable/epoch-protected but absent from the current
owner index and proceeds through its CodeUnit unlink/retirement lifecycle.
Invalidation or eviction may withdraw any current/reserved/pending state;
stale cleanup restores CurrentStable only by matching the exact reservation
token and state. `ReplacementReserved(family_version, BuildToken)` is the
validated tag/payload of the family's `FamilyLifecycleCell`; there is no separate
`reshape_token` or second lease authority. Its exact-token abort CAS is
`ReplacementReserved(v, token) -> CurrentStable(v)`. An invalidator may instead
win a transition from that exact reserved value to `Withdrawing`; a later
worker CAS then fails and must not restore it. `FamilyRetired` is entered only after its owner-index,
dispatch and member-unit roots have been removed. Its registry slot remains
occupied until every member unit and explicit family/build pin has passed
grace; `FamilyReclaimed` then releases that generational slot. Therefore a
replacement chain cannot accumulate: at most one successor and its one or two
immediate predecessors coexist for a cutover.

`FamilyPin` is a non-cloneable generational RAII reference backed by each
family slot's checked u32 pin count. Acquisition holds the JIT-state mutex,
validates the exact HcqFamilyHandle and an acquirable lifecycle
(CurrentStable, ReplacementReserved, CurrentPendingCutover, CutoverOld or
Withdrawing), checked-increments the count, and revalidates the same cell before
unlocking. FamilyRetired is nonacquirable. Release performs one AcqRel decrement
and wakes zero/capacity waiters. FamilyReclaimed requires the exact retired
generation, pin count zero and every member UnitPin/epoch grace complete under
the JIT-state mutex. ActiveBuild/reshape snapshots store explicit FamilyPins;
there is no implicit “build pin” or Arc which reclaim must discover.

`AdmissionFingerprint` is exactly one of:

```text
Ordinary {
    seed: BlockKey,
    seed_reachability: ReachabilityVersion,
    seed_lcq_code_version: CodeVersion,
    execution_key: ExecutionKey,
    hcq_shape_version: u64,
    instruction_observation_cursor: u64,
    normalized_successors: [ObservedSuccessor; 0..=4],
}
Reshape {
    source: BoundaryEndpointIdentity,
    target: BoundaryEndpointIdentity,
    execution_key: ExecutionKey,
    hcq_shape_version: u64,
    instruction_observation_cursor: u64,
    participating_families: [ParticipatingFamily; 1..=2],
}

BoundaryEndpointIdentity =
    LcqEndpoint {
        block: BlockKey,
        reachability: ReachabilityVersion,
        code_version: CodeVersion,
    }
  | HcqEndpoint {
        instruction: InstructionKey,
        family: HcqFamilyId,
        family_version: u64,
        code_version: CodeVersion,
    }

OwnershipObservation {
    instruction: InstructionKey,
    owner: None | Owned(HcqFamilyId, family_version),
}

OwnershipSnapshot {
    hcq_shape_version: u64,
    observations: [OwnershipObservation; 1..=2048],
}

BuildFingerprint {
    admission: AdmissionFingerprint,
    ownership: OwnershipSnapshot,
}
```

`BuildFingerprintHandleV1` is the generational slot handle defined in the
[request schema](maintenance-records.md); `OptionalBuildFingerprintHandle` is its explicit
tag/zero-payload wrapper. The fixed 266,404-slot fingerprint registry is
owned by the JIT-state mutex. Each slot contains its generation, state, payload
MetadataBlockHandle, one explicit payload-owner MetadataBlockPin and a checked-
u32 FingerprintPin count; a bare payload handle is never its lifetime. Its exact graph is
`Free(g) -> Staging(g, ActiveBuildHandle) ->
Published(g, HcqFamilyHandle) -> Retired(g, retirement_epoch) -> Free(g+1)` or
`Staging -> Retired` on reject/cancel. The ActiveBuild owner allocates one
slot/payload only after discovery has the complete fingerprint. Successful
publication atomically transfers that ownership to the family; all
DispatchPayloads reference the same handle rather than copying the variable
object. The ActiveBuild owner acquires the payload-owner pin with the allocation;
successful publication atomically transfers that one pin to the family, while
rejection/cancellation retains it until the Staging slot becomes nonacquirable.
Each of the 4096 NegativeBuildCache ways stores that same handle plus
one explicit owned FingerprintPin while resident; it never embeds a second
BuildFingerprint copy.

The immutable payload contains the full StableEncode-able BuildFingerprint and
is allocated from `build_fingerprint_payload_pages`; its OwnershipSnapshot is
bounded by 2048 observations and its admission successor list by four. A
FingerprintPin is acquired under JIT-state by validating handle/state,
checked-incrementing, then revalidating; release broadcasts every decrement.
Published is acquirable while the owning family is acquirable. Retirement first
makes the slot nonacquirable, removes negative-cache/dispatch/family references,
records the last relevant code epoch, then waits that epoch and FingerprintPin
count zero, releases exactly the slot's payload-owner MetadataBlockPin, retires/
decommits the PagedPayload, and only then advances the slot generation and
clears its allocation bit. Staging rejection/cancellation performs the same
owner-pin release after ActiveBuild no longer exposes the payload. Registry or
metadata-capacity failure is transient before backend/publication and can never
leave a dangling OptionalBuildFingerprintHandle. The slot, bitmap, directory
and payload pages have the four exact [layout rows](policy-and-capacity.md); no heap Arc or second
fingerprint store exists.

Source/target are directed and participating families are sorted by
HcqFamilyId. An internal HCQ InstructionKey need not have a DispatchSlot, so it
never fabricates a ReachabilityVersion. An ordinary admission fingerprint contains the
full normalized successor list offered by AdmissionSnapshot; reshape has no
such field and discovers only from the directed boundary plus pinned family/LCQ
images. Either fingerprint may cache FNV-1a of its complete StableEncode for
indexing, but that hash is not an identity field. HcqShapeVersion advances on
LCQ publication/removal or current HCQ ownership/family change, using the
reserve-mutate-release-publish order above. Equality is required only from
admission lookup through the one JIT-mutex OwnershipSnapshot capture; it proves
the stable owner map did not change while discovery established that snapshot.
After capture, an unrelated HcqShapeVersion advance changes future admission/
negative-cache identity but does not by itself discard this build: reservation
and publication revalidate every captured unit, owner, anchor and dependency
entry exactly and cancel only on one of those mismatches. The
ordinary successor-list field is exactly that offered ordered list, whether or
not a later ownership/capacity rule admits every item; neither variant depends
on a decision made after fingerprint lookup or contains raw counts/sample
sequence numbers. Hashes use
64-bit FNV-1a over each field's canonical little-endian byte encoding
(offset basis 14695981039346656037, prime 1099511628211 and wrapping
multiplication); hash equality is followed by full-field equality and is never
identity by itself.
Active admission, deduplication and negative-cache lookup/suppression are keyed
by AdmissionFingerprint. A published DispatchPayload's immutable identity and
the rejection evidence stored in a negative-cache value carry the complete
BuildFingerprint; every negative value requires
`value.admission == lookup_key`. Thus a profile, dependency, owner or observed-
successor change can form new work while an identical resident rejection cannot
spin; the full post-discovery fingerprint is never used as an index key before
it exists.

One shared `StableEncode` implementation supplies every hash/index named in
this specification. Unsigned integers use their declared fixed-width little-
endian bytes; signed integers use the same-width two's-complement little-endian
bytes; a nonzero-u64 newtype encodes exactly its inner u64 after its containing
field has selected the type; booleans use exactly one byte, 0 or 1. A struct
starts with its distinct u32 little-endian type-domain tag and then encodes
fields in declaration order. An enum likewise starts with its type-domain tag,
then its [Task 0](tasks/00-baseline.md) u32 little-endian variant discriminant, then only that variant's
payload fields in declaration order. An optional value uses one byte 0 for None
or 1 followed by its value for Some. A fixed array has no length prefix and
encodes exactly its declared number of elements; a variable sequence uses a u32
little-endian element count followed by its elements. A UTF-8 string uses its
u32 little-endian byte length followed by those exact bytes, with no NUL.
Integers outside their declared width, invalid bool/presence bytes and sequence/
string lengths above u32::MAX are rejected before encoding. Host pointers,
padding, anonymous tuples, Rust discriminants and randomized `Hash` state are
forbidden. [Task 0](tasks/00-baseline.md) checks golden encodings/hashes for BlockKey, InstructionKey,
ExitSiteKey, PicProbeKey, BuildFingerprint, BridgeKey, StaticBridgeKey,
HelperIslandKey, BoundaryObservationKey, CanonicalBlockRef, MemberOriginKey and
ParticipatingFamily on both hosts. Every one has a type-domain row.

Each CodeUnit follows exactly:

```text
Staging -> PublishReady -> Published
Published -> Superseded -> UnlinkPending -> Unlinked
Published -> Invalidating -> UnlinkPending -> Unlinked
Unlinked -> Retired(retirement_epoch) -> DirectoryDetached -> Reclaimed
Staging|PublishReady -> Aborted -> Reclaimed
```

`Staging` owns an unreachable RW span and all preallocated metadata.
`PublishReady` means relocation, executable protection, instruction-cache
synchronization, dependency validation and every fallible verification have
completed, but no entry address is reachable. `Published` means at least one
executable root may name the unit; a guest CodeUnit normally starts with a
DispatchPayload, while a BridgeUnit starts with its owning patch/PIC root. A
`Superseded` unit remains callable only
through roots which existed before cutover; `Superseded` and `Invalidating`
both forbid acquisition of every new static, PIC, bridge, dispatch or
suspended-retry root. `Aborted` was never reachable; after releasing
preparation pins/credits it returns its span and metadata directly, but its
already-issued CodeUnitId/CodeVersion are never reused.
`UnlinkPending` retains every existing root until its machine-code patch or PIC
clear is globally visible. `Unlinked` means no dispatch payload, static branch,
bridge target, PIC entry or suspended native-fault retry can newly enter the
unit. `Retired` records the reclamation epoch. `DirectoryDetached` means no
current native-PC table names the unit; retired copy-on-write table snapshots
may still own pins. `Reclaimed` is entered only after those snapshots, every
execution epoch and every explicit UnitPin are quiescent. Only `Reclaimed`
returns executable spans and generational slots for reuse.

The complete set of executable entry roots is: current DispatchPayload entries;
installed static branches and their bridge/island targets; per-vCPU PIC entries;
a suspended fault continuation explicitly admitted for native retry; and a
source already executing under a nonzero code epoch. RSB entries contain only
BlockKeys and are not roots. Compiler jobs, link jobs, native-PC snapshots and
normal-stack fault resolvers own `UnitPin`s; a pin prevents reclamation but does
not make code executable. Every lifecycle transition compares UnitHandle
generation, CodeUnitId, CodeVersion and the operation's admission epoch. A
stale operation may release only its own exact reservation or UnitPin.

## Bounded registries and indexes

The runtime has the following authorities; none is append-only:

- the dispatch index maps BlockKey to generational DispatchSlot;
- the unit and family registries map generational handles to strong CodeUnit
  and HcqFamily references;
- the dependency index maps each physical/mapping page identity to every live
  or retired-but-callable CodeUnit which depends on it;
- the HCQ owner index maps each active or reserved InstructionKey to its exact
  family/build generation;
- the link graph stores incoming roots and outgoing patch records by source and
  target CodeVersion; and
- the native-PC directory derives a segment slot from the fault address and
  acquire-loads that segment generation's immutable sorted fault table.

The native-PC directory has exactly 127 `SegmentPcSlot`s, one for each usable
16 MiB segment. The inaccessible 15 MiB tail has no slot and a PC in it is an
internal fault. All potentially faulting direct-interpreter stubs are pinned
stub CodeUnits inside these segments; arbitrary process-text addresses are not
inserted into a second directory.

At process construction, host page size must equal the memory authority's fixed
4096-byte protection granule; otherwise the native backends are unavailable.
Each SegmentPcSlot contains a checked segment generation, page shift 12 and
exactly 4096 atomic page-table pointers. A
pointer names an immutable `PageFaultTable` sorted by disjoint half-open native
instruction intervals. Each record appears with the same identity and original
global range in every host page it intersects. A recoverable host instruction
is verifier-bounded below 4096 bytes and therefore appears in at most two;
maximal InternalOnly gaps may span and appear in arbitrarily many pages within
their bounded CodeUnit span. Its semantic element and stored bytes use this
exact raw schema:

```text
RawNativePcRecordV1 = repr(C, align(8)), size 64 {
    tag:u8, reserved_zero0:[u8;7],
    native_start:u64, native_end:u64,
    unit:UnitHandle, code_version:u64,
    fault_record_id:u64, immutable_metadata_offset:u32,
    reserved_zero1:u32
}
NativePcRecordIdentityV1 = RawNativePcRecordV1
```

Tag is Recoverable=1 or InternalOnly=2. Both require
`native_start<native_end`, a valid UnitHandle and nonzero CodeVersion.
Recoverable requires nonzero FaultRecordId and a validated metadata offset;
InternalOnly requires both `fault_record_id` and
`immutable_metadata_offset` zero. Every reserved byte is zero. The identity is
the complete canonical raw value, not a pointer/hash/subset. Raw decoding
validates these rules before constructing the corresponding semantic enum.

An immutable table is one FixedSlot header plus two PagedPayload blocks:

```text
PageFaultTableHeaderV1 = repr(C, align(8)), size 400 {
    table_generation:u64, segment_generation:u64,
    page_index:u32, record_count:u16, record_page_count:u16,
    snapshot_pin_count:AtomicU32, reserved_zero0:u32,
    retirement_epoch:AtomicU64,
    records_block:MetadataBlockHandleV1,
    unit_pin_set_block:MetadataBlockHandleV1,
    unit_pin_count:u16, reserved_zero1:u16, reserved_zero2:u32,
    record_page_indices:[u32;67], reserved_zero3:u32
}
```

Record count is 1..=4096. The records block has IndexedSignalPayload type
RawNativePcRecordV1, exact byte_len `record_count*64` and
`record_page_count=ceil_div(record_count,62)` in 1..=67 because every 3,968-byte
payload page holds exactly 62 whole records. The first record_page_count entries
are the exact nonzero pool-page indices of the validated block chain and the
remainder are zero. The unit-pin block contains the sorted-unique UnitHandle for
each table-owned UnitPin, byte_len `unit_pin_count*16`; count is 1..=4096 and
every listed pin is owned until the table's retirement. Current tables have
retirement_epoch zero; removed tables release-store the nonzero retirement epoch
once. All other fields are immutable after pointer publication.

Each SegmentPcSlot pointer is exactly
`AtomicPtr<PageFaultTableHeaderV1>` and points at the payload field of the
FixedSlot wrapper. Canonical Null is a null pointer. After its one Acquire load,
signal lookup validates segment generation/page index/count, computes
`payload_page = record_page_indices[ordinal/62]` and
`payload_offset = 128 + (ordinal%62)*64`, and reads one complete raw record from
that mapped page; it never walks a chain or MetadataPageEntry. Binary search
therefore performs at most 13 header-indexed record loads. Publication pins both
PagedPayloads before the pointer Release store. The pointer/current owner holds
the header allocation's owner pin; its retirement owner holds that pin after
exchange until epoch grace and snapshot-pin zero, then releases the two payload
owner pins and UnitPins before freeing the header.

Cold snapshot acquisition occurs under the JIT-state mutex: validate the
pointer/header/segment/table identity, checked-increment snapshot_pin_count with
AcqRel, then revalidate the same pointer and identities before unlock; mismatch
decrements/wakes and retries or reports stale. Release decrements/wakes. This
pin is independent of the metadata allocation pin and prevents the retired
header/payloads from being repurposed while COW is prepared. No signal handler
takes it because active-code epoch supplies its lifetime.

A guest CodeUnit partitions its complete emitted span into exact recoverable
host-instruction ranges and maximal nonrecoverable InternalOnly gaps; a
BridgeUnit has one InternalOnly range. Recovery is allowed only for Recoverable.
A signal in InternalOnly, an island, padding or an address with no record is
InternalFault; those locations never fabricate MemoryEffectPlan metadata.
Ranges remain global RX half-open addresses when duplicated into each
intersected page; they are not clipped. Because all nonempty records in one
table are globally disjoint and each contains at least one distinct byte of its
4096-byte page, a table contains at most 4096 records regardless of how many
fault sites one guest instruction lowers into. Binary search therefore performs
at most 13 interval comparisons.
PageFaultTables own UnitPins for both variants, which is why BridgeUnit
publication/detachment participates in the same COW directory lifecycle.

The JIT-state mutex is the sole writer of page pointers. Adding or removing a
unit copy-on-writes only pages intersecting that unit. Exact replacement-table
capacity and directory UnitPins are reserved and charged before taking the
mutex. The writer fills and validates every table, then release-exchanges the
affected pointers; no allocation or fallible operation occurs after the first
exchange. Current and retired PageFaultTables own UnitPins for all records they
name. An exchanged-out table is epoch-retired until every executor which could
have loaded it is quiescent. Signal-time lookup is one acquire-load and the
bounded binary search; it does not lock, allocate, increment an Arc or retry a
hazard-pointer loop.

Cold COW preparation never relies on pointer revalidation alone. Under the
JIT-state mutex it acquires a checked explicit snapshot pin and a
[PageFaultTableHandleV1](cohort.md#native-page-cow-ownership) for each current table (or its
canonical typed Null value); the handle carries the generational header block,
not an unowned pointer. Every immutable table
header carries a nonzero checked-u64 `table_generation` allocated by that page
slot; it never wraps. The pin keeps the table and every directory UnitPin it
owns alive while replacement bytes are built outside the mutex. Commit requires
the page pointer and all handle fields still match, or aborts before mutation;
commit/abort releases the snapshot pins only after its last table read. Epoch
retirement still protects signal readers, which never take this pin.

A lookup is valid only for an executor with a nonzero active code epoch and a
matching segment generation. The normal-stack resolver clones an explicit
UnitPin before clearing that epoch. Before segment reuse, every page pointer is
release-exchanged to null, all removed tables pass their grace periods, all
directory UnitPins are released and every old span is reclaimed; only then is
the checked segment generation advanced. An address is never reused while a
current or retired table can classify it as an older CodeVersion. All other
indexes are cold-path structures and are never consulted by ordinary generated
RAM operations or resolved static links.

Registries use reusable generational slabs. A queued compiler/link job owns a
strong reference to every CodeUnit snapshot it reads. Removal from an index
does not free an object until vCPU execution epochs, compiler references, link
jobs and fault dispatchers have all released it. Metadata bytes and occupied
slots count against the same cache budget as their CodeUnit; empty slots are
reused. There is no permanent range partition, process-lifetime node pool or
second executable lookup table.

## Open admission token

Every gate which must forbid a new counted owner while another thread drains
existing owners uses this one generated layout and algorithm:

```text
CountedCloseGateV1 = repr(C, align(4)), size 4 { word: AtomicU32 }
CLOSED = 0x8000_0000
COUNT_MASK = 0x7fff_ffff
```

An open value is `n` and a closed value is `CLOSED|n`; all other
interpretations are invalid. `try_acquire(limit)` acquire-loads the word,
rejects CLOSED, rejects or follows the call site's specified capacity behavior
when `n == limit`, and CASes `n` to `n+1` with AcqRel success/Acquire failure in
a loop. That successful CAS is the acquisition linearization point. Release
uses `fetch_sub(1, AcqRel)`, requires a nonzero low count, and raw-futex-wakes
all waiters on the gate word after every decrement. `close` uses
`fetch_or(CLOSED, AcqRel)` and wakes all; it is idempotent, but only the named
close authority may call it. A drain waiter acquire-loads, returns only for
`CLOSED|0`, otherwise snapshots the complete word, rechecks it and waits on the
same aligned AtomicU32. `reopen` is legal only for the named reopen authority
after an acquire observation of exactly `CLOSED|0`; it release-stores zero and
wakes all. No caller increments a separate count, infers closure from another
atomic, or implements a load/increment/load double check. Acquisition and close
are ordered in the modification order of the same atomic, so a token is either
counted before close or cannot be acquired; this is the required AArch64 as
well as x86-64 proof.

Gateway admission and CodeUnit publication share `open_token_gate`, one such
gate with acquisition limit
`configured_max_vcpus + worker_count + configured_max_address_spaces + 2`
(at most 198). The exhaustive owners are: one per vCPU admission bit across
invoke/synchronous compile/resolve/publish; one per enabled WorkerControlCell;
one per address space serialized by its MappingRequestPermit; one lifecycle
owner serialized by the slot-registration mutex; and one process-wide
code-cache/reclaim publication owner serialized by its owner mutex. An
acquisition site which cannot prove membership in exactly one class is an
invariant error and must not be added. The coordinator also owns one ABA-safe
atomic tagged phase whose `Open`, `Closing` and `Closed` variants embed the
checked `admission_epoch`. The epoch is constant throughout one
Open -> Closing -> Closed transition and increments exactly once immediately
before Closed release-stores the next `Open(next_epoch)` state. There is no
separate admission-epoch atomic whose observation could be mixed with a phase
from another transition.

Token acquisition is exactly:

1. `try_acquire` `open_token_gate` at the bound above;
2. acquire-load and validate the complete phase payload and require `Open(E)`;
3. acquire-load ShutdownRecord and require Idle; and
4. retain a token carrying E if both checks pass, otherwise release the gate
   count and return the observed non-Open/sticky result using TerminalControl
   only.

Closing, while holding the queue mutex, closes `open_token_gate` before it
publishes the matching Open-to-Closing phase CAS. A successful acquisition
ordered before that close is part of E and the closer drains it; one ordered
after cannot succeed. Closed-to-Open first release-publishes `Open(next_epoch)`
and only then reopens the acquire-observed `CLOSED|0` gate; shutdown intent
forbids that reopen. A gate/phase combination other than open-gate with Open or
closed-gate with Open/Closing/Closed during these named finite tails is
corruption. Thus no correctness argument depends on observing two different
atomics in one global order.

A gateway holds its token only until it has release-published a nonzero active
code epoch. It releases the token before loading any DispatchPayload or native
entry. A publisher holds its token through the complete no-fail publication
commit. No token owner waits for maintenance, mutates mappings, invokes a
helper or joins a worker. Open-to-Closing prevents new successful acquisitions
and Closing acquire-waits for the count to become zero before freezing indexes.
Thus a racing gateway either has a visible active code epoch or has not read an
executable pointer; a racing publisher either changes no reachable state or
completes logically before the Closing freeze point and is included by that
transition.

A gateway OpenToken is initially process-only and is acquired while the
TerminalControl-resident ArenaAdmissionToken already protects its slot. While
both are held it validates the arena-resident VcpuHandle/generation, acquires
VcpuUseToken and binds that vCPU identity to the OpenToken. It acquire-
revalidates `Active(g)` after TLS publication and immediately before publishing
the nonzero active epoch; either failure clears TLS, releases VcpuUseToken,
OpenToken and ArenaAdmissionToken in that order and returns Unavailable without
reading a DispatchPayload. A background HCQ publisher OpenToken has no vCPU
owner and uses only the four process-phase checks above. An LCQ publisher or
cold PIC/resume path retains the one already-counted ArenaAdmissionToken and
VcpuUseToken for its vCPU and never acquires a second OpenToken concurrently.
This is the exact mechanism by which
Active-to-Deactivating blocks both a new gateway and a resume; there is no
independent per-vCPU OpenToken flag.
