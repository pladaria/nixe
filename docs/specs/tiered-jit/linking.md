# Static links, indirect dispatch, returns and target protection

[Specification index](README.md) · [Implementation tasks](tasks/README.md) · [Open review items](review-status.md)

Related contracts: [Gateway, state transfer and helper ABI](native-abi.md); [Cohort workspace, arbitration and mutation plans](cohort.md); [Epoch reclamation and terminal teardown](epochs-and-shutdown.md).

## Static and dynamic linking

## Static links

Every external static edge initially branches to a source-local fallback
thunk. The thunk resolves the target BlockKey through the dispatch index or
exits to compile that exact key. Before lookup it uses the source ExitStateMap
to canonicalize dirty state; an already-published target resumes through its
canonical ingress, while a miss leaves through the ordinary compile exit. This
traffic is confined to the unlinked fallback. Once source and target contracts
are compatible, the linker creates the minimal separately allocated
`StaticBridgeUnit`, when a transfer is needed, and marks the edge for a direct
patch. Target-independent source compilation never reserves bytes for a
register-copy veneer whose target contract is not yet known.

A source compiled while its target is already valid attempts the complete
prepublication link protocol before the source becomes executable; if bridge,
island, metadata or root reservation fails, it publishes safely on its
permanent fallback and may enqueue a later InstallLink. Changing already-published code uses the shared
control-poll word to request a process-local JIT maintenance rendezvous. HCQ
publication, replacement and unlink requests may share one pending rendezvous;
they never add another generated-code check. Until the rendezvous, old and
fallback targets remain callable and epoch-live.

Each patch record owns an intrusive pending-state bit and a generational list
handle. The bounded pending set contains at most one handle per live patch
record and has no separately allocated job object. Publication records the
patch and, only for an initially installed edge, its strong backlinks under the
JIT-state mutex before any new DispatchPayload becomes reachable. At
rendezvous, the linker revalidates both
source CodeVersion and target ReachabilityVersion immediately before changing
bytes. A stale record is detached without patching. Invalidation and cache
pressure force completion of every safety-critical unlink before execution
reopens; performance-only new links may be processed in batches of 4096 and
remain on their correct fallback until the next rendezvous.

InstallLink batching is deterministic and introduces no timer or generated-code
check. The process owns a functional, non-exported
`link_batch_poll_sequence: AtomicU64`, initialized to one. Every already-
required cold control poll reserves/increments it once before processing
control; exhaustion permanently disables new InstallLink attempts and cleans
each candidate back to Fallback, without affecting execution. When the first
InstallPending request for a patch is inserted, it stores
`eligible_poll_sequence = current_sequence + 8`, preflighted before changing
Fallback. An InstallLink alone neither asserts shared control nor tries
Open-to-Closing until exactly one of these events occurs: pending install count
reaches 4096; a cold poll acquire-observes sequence greater than or equal to the
minimum eligible sequence; or a safety/cutover request is already closing and
can piggyback it. That event, under the queue mutex, asserts control and makes
the ordinary close attempt. A cohort selects the first at most 4096 eligible
records by request_sequence. Deferred records retain their original eligibility;
handoff immediately schedules another round only if 4096 remain or their
minimum is already due. With no later guest poll and no safety work, an
underfilled batch may remain on its permanent fallback indefinitely, which is
correct because no execution is waiting to use the optimization. Every safety
record asserts/closes immediately and never waits for this batching policy.

Patch shapes are fixed:

- x86-64 uses an eight-byte-aligned eight-byte patch unit containing jmp rel32
  plus padding; and
- AArch64 uses one aligned b imm26.

A compatible in-range edge with an empty transfer bridge is exactly this one
patched branch after the guest branch decision. An edge needing register moves
branches to a `StaticBridgeUnit` and then to the target. That unit contains
the NativeLandingKind-required BTI/ENDBR landing pad when it can be reached
through an indirect island, then only the parallel-copy program derived from
the exact source ExitStateMap and target FastEntryContract followed by an
absolute scratch-register jump. Its immutable identity is exactly the named
[StaticBridgeKey](runtime-state.md#keys-and-dispatch-publication); the PatchRecordHandle supplies both slot and
generation. A directly reached bridge uses
DirectOnly; an island-reached bridge uses IndirectJump. It uses an exact
final-size allocation rounded to 16 bytes and must not exceed 2048 bytes on
either host. The [Task 0](tasks/00-baseline.md) bound proves the worst legal copy fits; an unexpected
overflow is `JitAbiCapacityExceeded`, while ordinary cache/slot pressure simply
leaves the patch on fallback.

AArch64 sources outside direct range branch to their local 16-byte island,
which loads an epoch-managed target and executes `br`; x86-64 uses the
equivalent island only when rel32 cannot reach. For an empty transfer, that
target is the final CodeUnit entry. For a nonempty transfer, it is the
`StaticBridgeUnit`, whether reached directly or through the island. These are
distinct accepted shapes and are not called single-branch links.

Each 16 MiB segment reserves its final 64 KiB for link/helper islands. A
CodeUnit preflights exactly one 16-byte slot for every external static exit plus
one slot for every distinct Leaf-helper target whose direct call range is not
guaranteed. These slots hold only the audited far transfer; register-copy
veneers are never part of an immutable source body. Calls within one segment
share one helper slot for an identical HelperIslandKey, using numeric HelperId,
the SHA-256 of the verified final helper veneer image and NativeLandingKind,
where a
far helper call requires IndirectCall. A helper binary/address change therefore
requires a new process/segment generation and never retargets an old island.
The resolved helper address is relocation input validated against the immutable
process helper table, not a StableEncode key field.
Independently patchable exits may not share. Helper islands
use one of exactly 4096 fixed `IslandSlotRecordV1` entries parallel to the
segment bitmap. A record is `Free=0`,
`StaticExit(PatchRecordHandle,patch_generation)=1` or
`Helper(HelperIslandKey)=2`; unused union bytes are zero. Under the code-cache
writer mutex, helper interning scans records in ascending slot order, compares
the full key, requires at most one match, and otherwise claims the lowest Free
record/bit. Static-exit allocation also claims the lowest Free record but never
shares. This array is the complete intern/ownership table; no hash map or heap
node exists. Its `island_slot_records` layout row is lazy-committed once per
committed segment and charges the generated record size/page rounding as
metadata; `island_bitmaps` has 4096 meaningful bits per segment.
Helper islands are immutable and
are never individually freed or retargeted: their charged slot/record lasts
until the segment has no live spans, all epoch/root grace completes, the whole
segment is decommitted and its checked generation advances. Actual allocation
reserves a helper slot only for each distinct key absent from that segment;
structural fit conservatively counts every distinct helper key used by the
candidate. Static-exit islands instead belong exclusively to their source
PatchRecord and return to the bitmap only after that source is Unlinked,
DirectoryDetached and epoch-quiescent.
StaticBridgeUnits and dynamic BridgeUnits use ordinary executable spans and
never allocate island slots of their own.

Every helper island targets a project-owned `global_asm!` HelperEntryVeneer
selected bijectively by numeric HelperId, never a Rust function's incidental
entry. On x86-64 that veneer begins ENDBR64; on AArch64 it begins `bti c`; it
then uses the declared system ABI and a direct hidden-symbol call/branch to the
typed helper implementation. The helper-image SHA-256 covers the complete
veneer bytes plus its resolved hidden relocation set. AArch64 production links
the main executable with GNU property `GNU_PROPERTY_AARCH64_FEATURE_1_BTI` and
[Task 0](tasks/00-baseline.md)/2 verify the loaded veneer mapping and actual indirect island target;
construction fails before Ready if the property/mapping or landing instruction
is absent. No far island may target past that landing instruction.

Structural fit is tested before consulting live allocator state: compare exact
body bytes with `16 MiB - 64 KiB` and exact distinct island demand with 4096
slots, as if one empty segment were available. [Task 0](tasks/00-baseline.md) records an exhaustive
per-instruction emitted-size/helper bound proving that every one-instruction LCQ
and an at-most-two-exit LCQ island set passes this pure test. If an LCQ attempt
fails it, it discards output and applies the same prefix-halving rule; N=1 is
`JitAbiCapacityExceeded`.

HCQ handles either structural-test failure by removing the highest discovery-ordinal
removable block, dropping blocks no longer reachable from `discovery_root`,
and recomputing selected entries, liveness, external exits, body and island
demand before relowering. Here too, removable means nonmandatory and removable
without disconnecting any mandatory endpoint from `discovery_root`; the
ordinary seed and reshape endpoints are never removed. If the connected
mandatory core cannot fit, it records
`HcqRejected(SegmentCapacity)` for body-span overflow or
`HcqRejected(IslandCapacity)` for island overflow under the exact fingerprint.
Emitted output is never split or patched into separate semantic bodies, and no
unit is published without its full future-target reservation.

Only after the deterministic shape passes does the allocator reserve body and
complete island set atomically from one actual segment generation. It tries
eligible existing segments in ascending slot order and then at most one newly
committed segment; it never keeps a partial reservation. Fragmentation, no free
segment/slot, the 608 MiB performance ceiling or any other current-pressure
failure returns a transient cancellation with no trimming and no
NegativeBuildCache entry. Thus identical snapshots do not acquire different
structural shapes merely because allocator occupancy differs.

Every PatchRecord has invariant fields created with its source: source
UnitHandle/CodeVersion, ExitSiteKey and source state map, target BlockKey, patch
address/shape, optional source-owned island, permanent fallback bytes/target
and record generation. Its state payload is exactly:

```text
Fallback {
    // no target version, bridge or target/bridge backlink
}
InstallPending {
    expected_target_reachability: ReachabilityVersion,
    expected_target_code_version: CodeVersion,
    expected_target_contract: FastEntryContractId,
    prepared_bridge: None | PublishReady UnitHandle,
    // claimed record pins are preparation ownership, not executable backlinks
}
InstallCleanupPending {
    final_state: Fallback | Dead,
    reason: typed LinkCleanupReason,
    cleanup_ticket: LinkCleanupTicket,
    prepared_bridge: None | PublishReady | Aborted UnitHandle,
    // exact claimed pins, unexchanged COW bundle, span and ledger credit
}
Installed | UnlinkPending {
    target_unit: UnitHandle,
    target_reachability: ReachabilityVersion,
    target_code_version: CodeVersion,
    target_contract: FastEntryContractId,
    bridge: None | Published UnitHandle,
    strong_source_target_bridge_backlinks: exact root handles,
}
Dead {
    // source is noncallable; no target version, bridge or backlink
}
```

Thus an external edge whose target has never been compiled is a complete valid
Fallback record. InstallPending owns only explicit preparation pins; the
Installed payload and backlinks become visible before the patch can target
them, and UnlinkPending retains that exact payload until fallback bytes are
globally synchronized. Returning to Fallback/Dead first releases those
backlinks under the lifecycle protocol, so invalidation unlinks without
scanning unrelated code.

Every patch follows exactly:

```text
Fallback -> InstallPending
InstallPending -> Installed(target CodeVersion)
Installed -> UnlinkPending -> Fallback|Dead
Fallback -> Dead                                      [source becomes noncallable]
InstallPending -> InstallCleanupPending(final = Fallback|Dead, reason)
InstallCleanupPending -> Fallback|Dead
```

`InstallPending` and InstallCleanupPending are not target roots. Before enqueue, the linker may allocate,
emit and synchronize a StaticBridgeUnit while it is unreachable. The
generational PatchRecord/registry ownership keeps that unit in `PublishReady`;
the queued request retains only source/target/bridge handles and immutable
identity copies, not a target or source UnitPin. Temporary pins used to copy
source-map/target-contract data are released before enqueue. When ClosingPrepare
claims the record, it acquires and revalidates fresh source, target and optional
bridge UnitPins under the JIT-state mutex into precharged plan storage. For a
nonempty bridge it also acquires PageFaultTableHandles/snapshot pins for the at
most two intersected pages and consumes the InstallLink record's precharged
worst-case credit to build, outside the mutex, one exact insertion replacement
and one distinct same-capacity safety shadow per page. These are growth-capable
current tables, not the old table's same-capacity shadow. Any allocation,
changed handle or capacity failure is performance-only failure before mutation
and leaves the patch on fallback. An empty bridge has no directory insertion. A
deferred record releases those pins before receiving its new sequence and
repeats this acquisition when claimed again. During Closed, installation
revalidates source/target/bridge
versions and state contracts, installs provisional target and bridge backlinks
which pin every future destination, release-exchanges each prepared directory
table before the bridge or patch can become reachable, writes the island and source branch and
performs the complete cache/pipeline synchronization below. Before reopen it
then changes a newly built bridge from PublishReady to Published and the patch
record/root to `Installed`; no executor can observe the patched branch while
the coordinator is Closed. A nonempty InstallPending always names this record's
exact PublishReady bridge; StaticBridgeUnits are source/patch-generation-
specific and are never reused in an already-Published state.
An InstallCleanupPending-to-Fallback/Dead transition changes only its
unreachable StaticBridgeUnit through PublishReady -> Aborted -> Reclaimed,
never through a retirement epoch; it never touches an older installed root.
Installed may enter UnlinkPending only after
the corresponding backlink/root was actually published.
`UnlinkPending` retains the old root. Unlink writes the permanent fallback,
synchronizes it on all cores and only then removes the backlink/root and enters
`Fallback`. Before that patch, a last-root StaticBridgeUnit changes Published
to Invalidating and then UnlinkPending; after synchronized root removal it
changes to Unlinked. The displaced StaticBridgeUnit follows the ordinary
Unlinked/Retired/DirectoryDetached/Reclaimed epoch path; its span or registry
slot is never reused merely because the patch record changed. `Dead` is used
when the source itself is unlinked. Metadata state alone never proves that a
machine-code root has been cut.

Every pre-mutation stale, capacity, sequence-exhaustion, cancellation, shutdown
or FailedBeforeMutation path first changes InstallPending to
InstallCleanupPending with an exact queue-owned `LinkCleanupTicket`; it may not
publish the MaintenanceRecord result or recycle PatchRecord yet. The ticket
releases claimed UnitPins and unexchanged PageFaultTable snapshot pins under the
proper mutex, marks only its exact PublishReady bridge Aborted, then with no
queue/JIT mutex takes the code-cache writer mutex, clears the span's exact body
or island allocation bits and
reconciles ledger credit. The ticket retains a distinct stable cleanup-ownership
reference to the bridge's common unit-registry slot and quota permit. After releasing the cache mutex it
takes only the JIT-state mutex, requires the exact UnitHandle still be Aborted
with zero roots and UnitPins, changes Aborted to Reclaimed, removes the bridge
from the common unit registry and releases its metadata/slot generation and
quota permit; then it unlocks and
consumes the cleanup-ownership reference. It finally reacquires only the queue mutex, changes
the PatchRecord to its recorded Fallback/Dead destination, publishes the normal
result, changes the exact MaintenanceRecord Awaiting state to final, consumes
the counted Awaiting bit and drops the ticket. A cancellation before enqueue is cleaned by the
linker synchronously through the same routine. Terminal shutdown drains all
LinkCleanupTickets before unmapping; no StaleNoOp/Free state can hide a live
PublishReady span.

A source linked while still unreachable follows the same order in staging: its
target backlink exists before source publication and the complete source image
is synchronized before any DispatchPayload names it. New images and live
patches use one audited platform wrapper with this contract:

1. complete all stores through the RW alias;
2. clean modified data-cache lines to the point of unification when required;
3. invalidate the corresponding RX instruction-cache lines;
4. execute the architecture's local barriers;
5. perform process-wide cross-core pipeline synchronization equivalent to the
   pinned `wasmtime_jit_icache_coherence::pipeline_flush_mt` contract;
6. return the affected RW alias pages to `PROT_NONE`; and
7. only then release-publish initial reachability or completed root mutation.

On AArch64, process-JIT construction reads `CTR_EL0` once: D-cache line bytes
are `4 << CTR_EL0[19:16]` and I-cache line bytes are `4 << CTR_EL0[3:0]`.
Each must be a power of two in `4..=4096`; an inaccessible register, invalid
field or checked-address overflow disables the native backend before guest
execution. For modified half-open range `[rw, rw + len)`, the wrapper checked-
rounds the RW range independently to D lines and the offset-identical RX range
to I lines. It executes `dc cvau` for every ascending RW D-line, `dsb ish`, `ic
ivau` for every ascending RX I-line, `dsb ish`, then `isb`. It then calls the
pinned `wasmtime_jit_icache_coherence::pipeline_flush_mt` once and only after
success closes the RW pages. The pinned stock `clear_cache` implementation's
fixed 64-byte I-line loop is not the required proof and is not called. Initial
RX protection is established before the I-cache loop and contains no
reachability publication.

On x86-64 the wrapper performs a release compiler fence after the final RW
store, changes initial RX protection when applicable, calls the same pinned
`pipeline_flush_mt`, and closes RW pages; it emits no fictitious cache
instruction. Initial-publication failure returns the unreachable span. Failure
while patching callable code leaves the RW alias closed, calls
`latch_terminal(CacheCoherenceFailure, CodeSync)`, completes/unlocks the tail and then calls
`drive_or_join_terminal()`; it never reopens native execution. A root
is not reported removed when synchronization failed.

## Indirect dispatch cache

Each vCPU owns a 2048-set, two-way cache in writable nonexecutable memory. A
way does not duplicate the current ExecutionKey. Its exact frozen form is:

```text
PicWayV1 = repr(C, align(8)), size 128 {
    occupied:u8, reserved_zero0:[u8;7],
    source_unit:UnitHandle,
    source_code_version:u64, exit_site_id:u64,
    exit_site_ordinal:u32, edge_kind:EdgeKind, reserved_zero1:[u8;3],
    source_instruction_pc:u64, exit_state_map_id:u64,
    target_guest_pc:u64, target_reachability:u64,
    target_code_version:u64,
    bridge:UnitHandle, bridge_entry_address:u64,
    pic_root:PicRootHandleV1
}
```

The source ExitSiteKey is reconstructed from the stored source fields and the
target BlockKey from the vCPU's bound ExecutionKey plus `target_guest_pc`.
Every occupied way's source and target have that identical ExecutionKey.
`bridge_entry_address` is a canonical 64-bit host address inside the validated
`bridge`; it is never accepted without the way's owned PicRoot. [Task 0](tasks/00-baseline.md) asserts
the displayed 128-byte size. Thus all ways at 128 vCPUs consume exactly
`4096 * 128 * 128 = 67,108,864` bytes; the `pic_ways` row additionally and
exactly charges 256 replacement-bit bytes per configured vCPU, rounded once
with the ways as that row specifies.

Before native entry with an ExecutionKey unequal to the one in
`TierSampleTablesV1`, the vCPU is at epoch zero and calls the one cold
`rebind_execution_caches` routine under the JIT-state mutex. That routine first
publishes every occupied way empty, then releases its PicRoot/backlinks and
marks a newly zero-root bridge for the existing asynchronous unlink path; its
OpenToken/PrequeueProducer ownership makes every cleanup obligation visible
before return. It then clears the sample tables and release-binds the new key.
It performs no COW allocation and never waits for bridge reclamation. Native
entry is forbidden until it completes. Consequently generated code may use the
already-bound key without a header load, while cold lookup always reconstructs
and compares the full identities. Generated BR/BLR:

1. computes the guest target;
2. preserves any lazy guest flags which its scratch-only probe would destroy;
3. probes both ways using the full source-site and target key;
4. jumps to the matching bridge on a valid hit; and
5. canonicalizes dirty state and enters the resolver on a miss.

The cache is private to the vCPU, so hits need no atomic operation, shared
generation load, ExecutionKey-header load or recency write. PIC set index is
canonical FNV-1a-64 of the reconstructed full PicProbeKey StableEncode masked
by 2047. Each set has one
private replacement bit initialized to zero. On a miss, the lowest-numbered
empty/stale way is used; if both are valid, the bit selects the victim. The bit
toggles only after successful installation, and a hit writes no recency state.

Only an in-native PIC hit reads this non-atomic table without a mutex, and its
active epoch protects that read. Every cold read, victim selection, copied-way
snapshot, revalidation, installation or clear holds the JIT-state mutex; this
includes Closed invalidation/deactivation and terminal root clearing (terminal
first drains every use-gate low count). Preparation uses only a bounded copy made in its
initial mutex section and never rereads a private way at epoch zero. Thus the
mutex, not an absent cold-path epoch, excludes resolver/maintenance races.

The cold resolver builds the complete [BridgeKey](runtime-state.md#keys-and-dispatch-publication) and rejects any
platform/ABI/source-map/target-contract inconsistency. The process weak
index contains exactly `1024 * configured_max_vcpus` sets of four ways, yielding
the declared 4096 slots per configured vCPU. Its set is FNV-1a-64 of the full
canonical BridgeKey modulo the set count. Under the JIT-state mutex lookup uses
full equality, removes stale weak handles, uses the lowest empty way, otherwise
replaces the per-set round-robin way and advances that two-bit cursor. It never
keeps a bridge alive.

The cold resolver is canonical, begins with this executor's active epoch zero
and already owns the source UnitPin described above. In one initial JIT-mutex
section it validates the generational source, resolves the exact target, clones
a target UnitPin and copies the immutable source-map/target-contract identities
into bounded owned scratch; a weak hit is usable only when full BridgeKey
equality, UnitHandle generation and lifecycle Published all match, and then
clones the bridge UnitPin. It then
releases the mutex. On a miss it reserves a common generational unit slot plus
one bridge-quota permit,
a charged executable span, emits the pinned/copied
NativeLandingKind::IndirectJump BTI/ENDBR pad when enabled, followed only by the
parallel-copy program plus absolute scratch-register transfer, validates it and
performs new-code synchronization while the BridgeUnit remains unreachable. No
preparation changes the weak index or PIC. Every abort releases source, target
and optional bridge pins only after it has stopped reading their metadata.
Before OpenToken acquisition, both a miss and a weak hit reserve the
precharged PicRoot/backlink records. They prepare a native-PC COW bundle sufficient for
the selected way's exact captured victim-root outcome. While taking the initial
JIT-mutex snapshot, preparation records the victim's checked PIC-root count and
the exact current PageFaultTable pointer/generation for every relevant page and
clones a `PageFaultTablePin` for each nonempty current table before releasing
the mutex. A missing-table sentinel is recorded explicitly. These pins, the
source/target/bridge UnitPins and the sole VcpuUseToken protect every object
read by preparation and commit; the resolver does not impersonate a native
executor by publishing an epoch.
The bundle enumerates
the deduplicated sorted `(segment_generation, RX_page_index)` union touched by
the complete half-open native ranges of a new bridge, when any, plus the victim
only when the captured root count is exactly one. A bridge of at most 2048
bytes can touch at most two 4096-byte pages, so this union has at most four
entries. Each entry owns one replacement of that page's captured current
PageFaultTable containing every retained record, the new bridge insertion iff
this is a miss and the victim removal iff the captured count was one;
coincident insertion/removal
is composed in that result. For every touched page the bundle always reserves two
objects: the replacement current PageFaultTable at the exact resulting count
and a distinct same-capacity safety shadow owned by that new current table.
It never assumes the old current's shadow can grow in place. A missing current
therefore also gets both prepared objects, not a same-capacity fiction. For a
weak hit with no last-root victim this union and bundle are empty and no
directory credit is charged. The at-most-four-page miss bundle/ledger charge
includes all eight worst-case tables.
There is no alternative table selected after preparation: a changed victim
root count or current PageFaultTable pointer makes commit validation fail and
the resolver retries through the canonical path. Failure leaves the way,
replacement bit, weak index
and old root unchanged and completes this transfer through the canonical path.

To commit either case, the resolver acquires an OpenToken for E and then takes
the JIT-state mutex; its active epoch remains zero for the whole cold resolver.
It revalidates Open(E), loads current global code epoch R under that mutex, and
revalidates the exact selected PIC-way
contents/replacement bit, captured victim PIC-root count, every captured
PageFaultTable pointer/generation and source, target and bridge generations.
The source and target must still be root-admissible: an LCQ must be current and
a family must be CurrentStable; CurrentPendingCutover, CutoverOld,
Superseded, Invalidating and Withdrawing reject installation. A weak-hit bridge
must still have the exact full BridgeKey and be Published; a newly built miss
must still be this attempt's exact PublishReady handle and have zero published
roots. Any other bridge state rejects installation. It preflights R+1 iff the
prepared COW bundle is nonempty; an empty weak-hit bundle consumes no code
epoch. Already-precharged root records convert the preparation
source, target and bridge UnitPins into one composite `PicRoot` containing a
strong source-metadata association plus bridge and final-target holds; this
conversion creates the root holds before consuming or releasing any pin. The
commit installs all new source/target/bridge PicRoot backlinks first. Only the
miss branch consumes its already-reserved common unit-registry slot and bridge-
quota permit, inserts the weak handle and changes its newly built BridgeUnit
from PublishReady to Published. The weak-hit branch leaves unit-registry and
quota ownership, weak-index contents and the existing bridge lifecycle
unchanged. It then writes the complete private PIC way and
toggles its replacement bit. The resolver is the sole VcpuUseToken owner for
that PIC, returns canonically and cannot execute the bridge during this locked
interval. If an occupied way was displaced, it removes
only that way's old composite root/backlinks after the new way is visible and
decrements the displaced BridgeUnit's checked PIC-root count. If it becomes
zero, the commit changes that bridge through Invalidating and UnlinkPending to
Unlinked, then explicitly marks it `Retired(R)` and removes its now-stale weak
index entry. It next performs exactly one release-exchange per touched page of
the already-prepared native-PC bundle. A miss table contains the new bridge; a
weak-hit table adds no bridge record. Either omits the victim exactly when its
revalidated count was one. Only
after all exchanges does a zero-root victim become DirectoryDetached. If
another PicRoot remains, the victim stays Published. A miss still consumes its
insertion-only COW bundle; a weak hit has no COW work in that case. No span/
metadata is reclaimed in this commit; ordinary
epoch/UnitPin grace performs Reclaimed later.
Every replaced COW/payload/root snapshot is tagged with retirement epoch R. If
the bundle is nonempty, the commit release-publishes global epoch R+1 after all
exchanges; otherwise it leaves the epoch R unchanged before unlocking.
The commit never writes a value derived from a pre-lock global-epoch load.
After that epoch publication, source/target/bridge preparation pins whose
references were not transferred into the visible composite PicRoot, and every
old PageFaultTablePin, remain live until the resolver has stopped all reads of
their corresponding metadata; it then releases them before the OpenToken.
Those releases cannot fail and no preparation pin survives a successful
resolver call. The OpenToken protects the registry transaction against a
Closing freeze; the strong pins protect the exact objects against reclamation.
An active epoch is neither needed nor permitted because this routine executes
no JIT code. The resolver releases the OpenToken after the complete no-fail
commit and returns the resolved target BlockKey to ordinary canonical
dispatch/gateway ingress. The current miss never jumps through the installed
BridgeUnit: miss canonicalization and the system-ABI resolver may have destroyed
every HostReg named by the source ExitStateMap. Only a later PIC/RSB hit reached
directly from that exact native source site has the required physical contract
and may enter the bridge. A failed
revalidation changes no root. Still under the JIT-state mutex it marks only this
attempt's newly built PublishReady bridge Aborted, when present, and performs no
allocator operation; a weak-hit attempt has no bridge state to change. It then
unlocks, stops reading native metadata, drops source/target/bridge preparation
pins and every PageFaultTablePin, and release-decrements its OpenToken exactly
once (waking Closing if it reaches zero). Only afterward, with no
token or subsystem mutex, a newly built miss takes the code-cache writer mutex,
clears its unreachable span's exact allocation bits and reconciles ledger
credit. It finally
uses a fresh canonical dispatch; no failed exit leaks the token, nests the
allocator lock or reuses the rejected bridge.

PIC ways need no atomic access because only their registered vCPU can install
or execute them, and deactivation/Closed clearing first prevents admission and
waits for that slot's active epoch to become zero. Each occupied PIC way owns
one composite PicRoot; its three internal holds are inseparable and replacing/
clearing the way releases all of them after fallback/no-entry visibility. The
source association keeps the exact ExitSiteKey/map metadata available for
invalidation but is not an independent executable entry into the source.
Rooted bridges therefore
cannot exceed 4096 times active_vcpus. Root removal does not imply immediate
reclamation: executing or retired bridges retain one of exactly
`8192 * configured_max_vcpus` bridge-quota permits plus their common unit slot
until pin/epoch quiescence. If no permit, common unit slot or charged capacity
is available after reclaiming already-quiescent
bridges, the selected PIC way stays empty and the instruction completes via the
ordinary canonical resolver; guest execution does not wait for HCQ, fail or
allocate an untracked bridge. `rooted_bridge_count`, occupied quota permits and
retired bridge bytes are independent checked invariants.

Publication may leave an old cache entry usable only while its source, bridge
and target versions stay callable. The mandatory maintenance rendezvous clears
entries which name a superseded/invalidated source or target before any of
those objects is retired; eviction cannot reclaim them sooner. Dynamic bridges
are ordinary allocator spans and never consume static link-island slots.
Repeated static-target replacement follows the same rule: one patch record has
at most one current StaticBridgeUnit, while displaced bridges enter the
bounded unit/epoch retirement path. If a replacement bridge cannot be reserved,
the source remains on its permanent canonical fallback. Installing a link is a
performance optimization; unlinking a stale root is mandatory safety work.

Guest values are never treated as host pointers.

## Software return stack

Each guest thread owns a 16-entry circular software return stack which follows
that thread across host-vCPU migration and is never executed concurrently.
It is an inline `[Option<BlockKey>; 16]` plus `u8 head` and `u8 depth` field in
the caller-owned GuestThread architectural state, not an arena allocation,
registry entry, JIT root or `jit_charged_bytes` category. GuestThread creation
zeroes every entry/head/depth; ordinary host-vCPU migration moves the owning
GuestThread and does not copy or clear it; guest reset, address-space change or
thread destruction clears it before the old architectural state can be reused.
Nixe creates no per-thread side table, so the number of guest threads cannot
grow JIT-owned storage.
Every entry contains the full continuation BlockKey constructed at the call
from the current unit's complete `ExecutionKey` plus the architecturally
validated continuation guest PC; no field is inherited from an unvalidated
target or reconstructed from host state. The stack stores a
four-bit head and a five-bit depth. BL/BLR pushes after updating architectural
X30; overflow overwrites the oldest entry and keeps depth at sixteen.

RET first computes the architectural target and compares it with the top full
BlockKey. On a match it pops and probes the ordinary per-vCPU PIC using that
RET's ExitSiteKey; a hit reaches the exact return bridge without canonical
state traffic. An unresolved match, mismatch or underflow uses the ordinary
indirect miss resolver; mismatch/underflow also clears the unreliable
prediction chain. The RSB contains no host code pointer and does not keep code
alive. Clearing affected PIC ways at maintenance is sufficient for bridge and
target reclamation.

The structure is only a prediction. It never changes X30 semantics or bypasses
target validation. Guest calls and returns use jumps, not the host return
stack.

## Branch-target protection

Landing bytes are a total function of NativeLandingKind. On AArch64,
`DirectOnly` emits none, `IndirectJump` starts with `bti j`, `IndirectCall`
starts with `bti c`, and `IndirectJumpOrCall` starts with `bti jc`. On x86-64,
DirectOnly emits none and every other kind starts with ENDBR64. Nixe emits these
instructions unconditionally at all such targets; they are compatible hints on
hosts without enforcement. Gateway entry is IndirectCall, every PIC/RSB bridge
entry is IndirectJump, and the final system-ABI helper veneer is IndirectCall.
The backend exports real selected label offsets; there is no ordinal veneer or
common br_table entry dispatcher.

On AArch64/Linux construction reads `getauxval(AT_HWCAP2)` and sets immutable
`use_bti = (value & HWCAP2_BTI) != 0`; the generated Linux constants and this
boolean are host-bound manifest inputs. When true, every RX code-cache page is
made executable with exactly `PROT_READ | PROT_EXEC | PROT_BTI`, where
`PROT_BTI = 0x10`, before any entry becomes reachable, and every later
protection transition preserves that bit. EINVAL or any failure disables the
native backend before a guest starts; it never retries without BTI. When false,
RX uses exactly PROT_READ|PROT_EXEC. On x86-64 Nixe does not enable or disable
process-wide CET policy; unconditional ENDBR64 makes every declared indirect
target valid if the embedding process enabled IBT. [Task 2](tasks/02-lifetime-foundation.md), before its first
native gateway proof, validates these mappings and landing bytes on both hosts;
[Task 4](tasks/04-native-linking.md) only adds the bridge/island shapes.
