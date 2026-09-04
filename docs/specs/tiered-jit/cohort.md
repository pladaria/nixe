# Cohort workspace, arbitration and mutation plans

[Specification index](README.md) · [Implementation tasks](tasks/README.md) · [Open review items](review-status.md)

Related contracts: [Maintenance records, cleanup and mapping requests](maintenance-records.md); [Executable cache, metadata and backend ownership](cache.md); [Request execution, terminal transfer and cohort handoff](coordinator-execution.md).

## Workspace and projection storage

The one coordinator owns a `CohortPlanWorkspaceV1`. Its first page is shared
with the radix histogram and begins with this exact header:

```text
CohortPlanWorkspaceHeaderV1 = repr(C, align(8)), size 80 {
    generation:u64, touched_len:[u32;8], cleanup_len:u32,
    batch_state:AtomicU32, allocator_freeze:CohortAllocatorFreezeV1
}
```

Generation is a checked u64 and each touched length is checked u32. The
batch state is Empty=0, Preparing=1 or Applying=2 and every other value is
corruption. Construction and each completed handoff use Empty. The remaining
workspace has one u32 `touched_indices` array for each of these
domains/capacities:
dispatch 1,048,576; unit 1,048,576; family 262,144; patch 2,097,152;
native-PC page 520,192; PIC way `4096 * configured_max_vcpus`; vCPU 128; and
address space 64. Every array is a construction-time PermanentExtent in the
metadata pool, fully committed at process construction; it is never allocated,
committed, grown or decommitted by a
cohort. At 128 vCPUs their page-rounded charge is exactly 22,011,904 bytes:
4,194,304 + 4,194,304 + 1,048,576 + 8,388,608 + 2,080,768 +
2,097,152 + 4,096 + 4,096 in that domain order. For another configured vCPU
count, only the PIC term changes, to
`ceil_div(4096 * configured_max_vcpus * 4, 4096) * 4096`. Each named
`cohort_touched_*` layout row is logical_capacity_only with backing_row
`metadata_arena_pool_pages`; the pool row owns and charges the actual pages
exactly once. [Task 0](tasks/00-baseline.md) includes their construction-committed pool bytes in
`fixed_charge_at_128_vcpus_worst_case` and the other construction inequalities.

The workspace additionally owns one shared
`radix_tmp:[u32;2_097_152]`; the header/histogram page whose byte offsets are
header `[0,80)`, `count:[u32;256]` `[80,1104)`, `cursor:[u32;256]`
`[1104,2128)` and required zero padding `[2128,4096)`; and
`cleanup_ordinals:[u32;4096]` in a 16,384-byte extent (the length is in the
workspace header and the extent contains only the array).
The cleanup array records each prepared InstallLink cleanup obligation exactly
once and replaces every repeated cutoff scan.

Pressure projection uses these two closed scratch layouts:

```text
MetadataProjectionCellV1 = repr(C, align(8)), size 16 {
    workspace_generation:u64, remove_owner_count:u16,
    expected_live_owner_count:u16, flags:u32
}
SegmentProjectionCellV1 = repr(C, align(8)), size 32 {
    workspace_generation:u64, expected_segment_generation:u64,
    remove_body_spans:u32, expected_live_body_spans:u32,
    remove_static_islands:u16, expected_live_static_islands:u16, flags:u32
}
```

There are exactly 260,086 metadata cells and 127 segment cells. In both types
flags bit zero means initialized/touched, bit one identifies a whole
PagedPayload allocation for a metadata cell or a whole-segment projection for
a segment cell, and every other bit is zero. The first contribution in G copies
the authoritative expected live counters/generation and later contributions
must match them; all additions are checked. A cell from another generation is
logically zero and is initialized before use. The page-rounded array sizes are
4,161,536 and 4,096 bytes. The workspace header also owns
`CohortAllocatorFreezeV1 = repr(C, align(8)), size 32 { state:u8,
reserved_zero:[u8;7], round:u64, workspace_generation:u64,
driver_token:u64 }`; Empty is the all-zero value and Held is state one with all
three identities nonzero.

The segment allocator maintains, in the authoritative CodeSegmentHeader under
the code-cache writer mutex, checked `live_body_span_count:u32` and
`live_static_island_count:u16`. A successful body-span or StaticExit-island
commit increments exactly one; its final post-root/post-pin free decrements
exactly one. Helper islands are deliberately excluded because helper-only
islands are invalidated as part of whole-segment decommit and do not keep a
segment live. These counters are mirrors, not allocators: at debug/[Task 8](tasks/08-lifecycle-stress.md) audit
points they must equal a full L0/IslandSlot owner walk, but production pressure
projection reads them in O(1) and never scans 16,320 bitmap words. A segment is
projected decommittable only when its accepted effects remove all live body
spans and all StaticExit islands; every remaining root, table and pin must belong
to those same accepted owners and is released by their mandatory post-grace
tail. A plan effect identifying an unselected persistent owner is a projection
mismatch, not an assumption that grace will remove it.

For metadata, each effect names one exact allocation owner and its full
MetadataBlockHandle. A FixedSlot contributes once only to its first/only page.
A PagedPayload contributes once to **each** page in its validated chain, but the
allocation owner itself is counted once for equality/pin validation; continuation
pages can remove page charge and never increment the first-page owner count.
On first contribution, the cache-mutex owner copies a FixedSlot page's
`live_slots` or the canonical PagedPayload owner count one into
`expected_live_owner_count`. Projection subtracts a page only when all its live
owners are represented, no Retiring/Staging owner exists, the exact plan pins
are the only pins beyond owner pins, and the complete allocation is in the
accepted no-fail release tail. Permanent, GuardedStack and Control pages never
receive an effect. Any generation/count/storage-kind/pin mismatch rejects
Pressure before mutation. Every allocation/page/span/island effect visit counts
in E below; neither projection cell hides an inner scan.

The page-rounded logical sizes are therefore: touched arrays 22,011,904;
radix temporary 8,388,608; histogram 4,096; cleanup ordinals 16,384; metadata
projection 4,161,536; and segment projection 4,096. At 128 vCPUs the complete
workspace is exactly **34,586,624 committed pool bytes**. The generated layout
has one `logical_capacity_only` PermanentExtent row for each named array, all
backed and charged only by `metadata_arena_pool_pages`; the header and freeze
value share the histogram page and are not a second page. [Task 0](tasks/00-baseline.md) asserts every
offset above. No plan owns a
second sort buffer, projection array, cleanup list, claim list or hash table.

## Object domains and claim identities

`CohortDomain` and each cell action are closed `repr(u8)` domains with these
only discriminants:

```text
CohortDomain = Dispatch=1 | Unit=2 | Family=3 | Patch=4 |
    NativePcPage=5 | PicWay=6 | Vcpu=7 | AddressSpace=8
DispatchPlanAction = PublishUnavailable=1 | PublishRetainedLcq=2
UnitPlanAction = Invalidate=1 | Evict=2 | CutoverPredecessor=3 |
    Unlink=4
FamilyPlanAction = Withdraw=1 | Evict=2 | CutoverPredecessor=3 |
    StabilizeSuccessor=4
PatchPlanAction = RestoreFallback=1 | InstallPreparedTarget=2
NativePagePlanAction = AggregateReplaceRecords=1
PicPlanAction = Clear=1
VcpuPlanAction = Deactivate=1
AddressSpacePlanAction = CommitMapping=1
```

Action zero means unowned only. A nonzero action is decoded using its array's
known CohortDomain; using a discriminant from another domain is corruption.
AggregateReplaceRecords is one precharged COW exchange containing the canonical
union of all accepted removal/addition deltas for that page, not a sequence of
per-plan exchanges. [Task 0](tasks/00-baseline.md)'s enum-discriminant table contains every row above.
The removed Dispatch/Unit successor-publication actions are intentionally
illegal: a replacing publisher already committed those stores before it exposed
CutoverToSuccessor, so closure may only revalidate them and clean/stabilize the
published topology; it must never publish a successor root a second time.

Every committed/live object slot or table entry, not every theoretical registry
capacity, embeds one 32-byte `repr(C, align(8)) CohortObjectPlanCell`:
`{ workspace_generation:u64, owner_record_ordinal:u32, action:u8,
flags:u8, pressure_action:u8, reserved_zero:u8, expected_slot_generation:u64,
expected_semantic_generation:u64 }`. Registry cells are therefore committed
and charged only with their live slab pages. The fixed native-page/PIC/vCPU/
address-space tables include their cells in their own layout rows. Per-object
action payloads live in the owning record's sealed plan or the object's existing
SafetyPayload/COW cell; no alternating 32-byte copy is multiplied by registry
capacity. Flags bit zero is PressureSeen and bits 1..7 are always zero;
pressure_action is nonzero only while that bit is set. An object is unowned for
the current cohort when its cell generation differs or owner/action is zero;
PressureSeen does not itself confer ownership.

The two expected generations have one exhaustive authority table; [Task 0](tasks/00-baseline.md) emits
this table as data and the runtime has no alternate source:

| Domain | expected_slot_generation | expected_semantic_generation | Zero legal | Incrementing writer |
|---|---|---|---|---|
| Dispatch | DispatchSlotHandleV1.slot_generation | new `dispatch_mutation_generation` incremented on every DispatchPayload store | never | JIT-state mutex |
| Unit | UnitHandle.slot_generation | UnitLifecycle.lifecycle_generation | never | JIT-state mutex |
| Family | HcqFamilyHandle.slot_generation | HcqFamily.family_version | never | JIT-state mutex |
| Patch | PatchRecordHandle.slot_generation | PatchRecord.patch_generation | never | maintenance-queue mutex |
| NativePcPage | containing segment's segment_generation | current PageFaultTable.table_generation | semantic zero only for the canonical Null table | JIT-state mutex |
| PicWay | owning VcpuHandle generation | new per-way `pic_mutation_generation` incremented on every cold-path way write/clear | never | JIT-state mutex; native hits never write it |
| Vcpu | VcpuHandle generation | new `vcpu_state_generation` incremented on every slot-state edge | never | slot-registration mutex |
| AddressSpace | AddressSpaceHandle generation | new `mapping_authority_generation` incremented by every committed MappingOperation | never | MemoryTransactionGuard's process memory-transaction mutex |

Every nonzero counter is initialized to one, checked-increments before its
mutation and never wraps. A claim revalidates both listed sources under the one
listed writer mutex immediately before batch commit. NativePcPage Null is
encoded by a null table pointer plus semantic generation zero; a nonnull pointer
with zero, or Null with nonzero, is corruption. No other generation, phase,
epoch, handle field or payload tag may be substituted.

At claim revalidation the driver takes these writer mutexes one at a time in
table order and releases each before taking the next; it never nests them with
the queue mutex. This sequential snapshot remains valid because
`open_token_gate == CLOSED|0` excludes every gateway, publisher, lifecycle and
mapping writer which could advance a listed authority, while Claimed/Applying
records exclude ordinary cleanup of their exact objects. The only cleanup
permitted during Closing is for an unrelated unclaimed object and must first
prove its CohortObjectPlanCell unowned in G. Violation latches terminal. Thus no
claimed authority can change between its revalidation and the queue-locked
batch CAS even though the eight mutexes are not nested.

`OwnerRecordOrdinal` values are
not cohort insertion order: unit safety record slot i is `1+i`; family tier-
cutover slot i starts at 1,048,577; patch record slot i starts at 1,310,721;
VcpuDeactivate record slot i starts at 3,407,873; address-space record i starts
at 3,408,001; and the PressureRecord is 3,408,065. FaultTransitionRecord is
not in this ordinal space because it never owns a RootMutationPlan. The maximum
is below 2^22. A function over the
ordinal returns the exact generational record and variant or fails the cohort.

Each prepared non-Pressure plan owns one immutable PlanClaimSet. Its wire records
are defined below and are read only through the RawT-before-semantic decoder.
`record_kind` is ActionClaim=1 or CommittedPublicationPredicate=2. The latter is
legal only for CutoverToSuccessor and has one of these closed action-byte values:
SuccessorUnitPublished=1, SuccessorFamilyPendingCutover=2,
PredecessorFamilyCutoverOld=3, SuccessorDispatchPublished=4 or
RetainedLcqDispatchPublished=5. It revalidates a store already made by the
replacing publisher and never owns a cell or causes another publication.

For every record ordinal, the generated total function
`semantic_by_record_ordinal(plan_variant, ordinal, decoded_record)` returns one
immutable `ClaimSemanticV1` view. Action claims are sorted into contiguous
domain regions, so `domain_rank = ordinal - generated_domain_start[domain]` is
O(1). The generated `(plan_variant,record_kind,domain,action) -> locator` table
has exactly three locator forms: fields in the sealed plan header; entry
`domain_rank` in that plan's sealed SafetyPayload array; or entry `domain_rank`
in its prepared native-page COW delta. An absent tuple, wrong array type/rank or
count mismatch is an invariant failure. There is no callback, trait object,
hash lookup or live graph traversal behind the accessor.

ClaimSemanticV1's closed variants are DispatchFinal, UnitFinal, FamilyFinal,
PatchFinal, NativePageDelta, PicFinal, VcpuFinal, AddressSpaceFinal and
CommittedPublication. Each StableEncode contains the exact expected and final
semantic values, and NativePageDelta contains its canonical base-table identity
plus sorted delta records. StableEncode excludes allocation addresses,
MetadataBlockHandles, owner pins and padding: two separately allocated COW
blocks which describe the same canonical value compare equal. Their handles,
types, lengths, generations and pins are nevertheless revalidated separately
before access. Fixed-size semantic comparison costs one claim visit; every
variable record read by a comparison, projection or COW composition counts in E
below. [Task 0](tasks/00-baseline.md) generates and freezes the locator table, every ClaimSemanticV1
discriminant/encoder and invalid-tuple tests; runtime code may not infer a
semantic payload merely from the u8 action.

`enumerate_claims` is a total allocation-free ordinal iterator over only
ActionClaim records and emits each required `(domain,index,expected slot and
semantic generations,action,ordinal)` exactly once. A second iterator over
CommittedPublicationPredicate records revalidates every exact already-published
state and its ClaimSemanticV1 but emits no action. The complete array is sorted
unique by `(record_kind,domain,index)`; duplicates, out-of-range indexes, changed
generations, wrong kinds/counts or a non-byte-identical union/Pass-A/Pass-B
stream are invariant failures before mutation.

PressureEvict is the only storage exception. It owns no PlanClaimSet and uses
only its permanently precharged selection bitmaps, victim pins and
SafetyPayloads. Its generated, allocation-free `expand_pressure_claims` walks
selected CurrentStable families and then selected LCQs in their cache-policy
orders and emits the same claim fields plus a canonical ClaimSemanticV1. It may
emit the same `(domain,index)` repeatedly. Its union and rank-2 raw streams must
be byte-identical; this is the sole duplicate exception and E includes every raw
occurrence and semantic/effect record.

## Closure arbitration and commit

Closure arbitration and commit order are exactly the following.

1. The DriverToken holder is the sole reader/writer of the workspace and every
   current-G CohortObjectPlanCell until handoff; those cells are cohort scratch,
   not subsystem state. Under the queue mutex it seals `(round,ticket,cutoff)`
   and makes one increasing-sequence scan of all Q linked records through the
   cutoff. It validates every eligible prepared plan and enumerates each action
   stream once to form the union, without changing a MaintenanceRecord state.
   On first touch it writes G and zeros **owner, action, flags,
   pressure_action, both expected generations and reserved byte** before
   appending the index; later touches append nothing. It releases the mutex.
2. The sole driver radix-sorts every touched slice numerically ascending with
   exactly four stable LSD byte passes, byte 0 through byte 3. Each pass zeroes
   count and cursor, counts the selected byte, computes checked exclusive
   prefixes and scatters stably while alternating touched/radix_tmp; four passes
   leave the result in touched. Adjacent equality is an invariant failure.
   Domains run Dispatch, Unit, Family, Patch, NativePcPage, PicWay, Vcpu and
   AddressSpace. No comparison sort or implementation-selected radix width is
   legal.
3. Plan priority is `(rank,request_sequence)`, lower first, with ranks exactly
   0 InvalidateToUnavailable/MappingChange, 1 VcpuDeactivate, 2 PressureEvict,
   3 CutoverToSuccessor, 4 UnlinkOnly and 5 InstallLink. Sequence is unique.
   Under the queue mutex the driver makes one increasing-sequence scan for each
   of ranks 0 and 1. Pass A enumerates a candidate's whole immutable stream and
   writes nothing: it validates records/semantic views and classifies every
   collision. Invariant dominates recoverable outcomes; otherwise the first
   rejecting claim in canonical `(domain,index)` order chooses the whole-plan
   result. Only an admissible whole plan runs Pass B over the identical stream,
   installing each Free owner/action/generation and composing each mergeable
   aggregate. Rejection therefore leaves no partial claim.
4. Still before lower ranks, the driver expands the unique PressureRecord once.
   At an object's first raw occurrence it sets PressureSeen, stores the two
   expected generations and pressure_action, and counts one distinct required
   claim. A duplicate must have identical expected generations and invokes the
   generated `join_pressure_claim(old_action,old_semantic,new_action,
   new_semantic,object_safety_snapshot)`. It either returns a combined action
   whose canonical semantic value is derived in O(1) by
   `pressure_semantic_for(combined_action,object_safety_snapshot)`, or reports a
   typed incompatibility before ownership. [Task 0](tasks/00-baseline.md) proves this operation total
   over legal inputs, associative, commutative and idempotent, with unavailable/
   cut dominating retained LCQ; no semantic payload is silently discarded.
   After expansion, one sorted-touched scan classifies each PressureSeen object
   exactly once. Free objects become provisional `{owner=PressureOrdinal,
   action=pressure_action}` while retaining PressureSeen for rollback;
   higher-owned objects run the collision rule once against the combined
   semantic value. A mismatch or rejecting predicate rejects Pressure before
   projection, performs step 6's scratch rollback directly and skips the
   allocator gate/projection; duplicates never increment required/satisfied
   counts twice.
5. Only when Pressure passed distinct-object classification, the driver releases the queue, takes only the code-cache writer mutex and
   changes allocator_freeze Empty-to-Held with its exact round/G/DriverToken.
   While Held, an ordinary cache writer must release the mutex and return its
   specified Deferred/Fallback result before any allocation topology or ledger
   mutation; no reservation, credited commit, retirement or decommit proceeds.
   Only the matching DriverGuard, or its exact TerminalDriverGuard transfer, may
   inspect/update projection scratch and eventually clear the gate. It then
   requires Pressure's input ledger sequence equal the stable even sequence,
   walks accepted owner cells and their generated projection effects to fill the
   G projection arrays, and scans metadata pages/segments exactly once. Every
   count/generation/pin is revalidated against the authoritative cache-mutex
   state. `projected_total_bytes` is the checked charged total after all rank
   0/1 and provisional Pressure effects and before the requested allocation;
   success is exactly `projected_total_bytes + requested_total_bytes <= 512
   MiB`. No slab-slot, bitmap-word, dependency or PageFaultTable loop is hidden
   in one projection visit. It releases the cache mutex while retaining the
   linear CohortAllocatorFreezeGuard. With no Pressure candidate, or one
   rejected before this step, allocator_freeze remains Empty and no cache writer
   is delayed.
6. Under the queue mutex it revalidates round/ticket/cutoff/G and all claimed
   record states. On Pressure failure, one touched scan clears owner/action only
   where owner is PressureOrdinal and clears PressureSeen/pressure_action on
   every seen cell, sets Pressure counts to zero and publishes its exact
   StaleNoOp/replan result. If that failure followed projection, it unlocks the
   queue, clears the exact Held freeze under the cache mutex, drops its guard,
   relocks and revalidates the round before lower ranks. On success, action
   already contains the combined pressure_action; a Pressure Pass B composes
   every Free or MergeNativeCow semantic delta (including native-page removal
   bits) and only then clears PressureSeen/pressure_action on every seen cell.
   Checked owned+satisfied equals the distinct required count.
   It processes ranks 3, 4 and 5 with the same whole-plan Pass-A/Pass-B rule.
   UnlinkOnly with every claim satisfied and no Free claim is
   StaleNoOp(CoalescedHigherPrecedence); all-Free or mixed Free+satisfied is
   accepted and owns the Free remainder. InstallLink with any non-Free claim
   owns nothing, appends its ordinal exactly once to cleanup_ordinals and becomes
   AwaitingLinkCleanup(StaleNoOp(Superseded)). At most the 4096 selected links
   can reach this path, so capacity is exact. MandatoryCutoverFollower may own
   remaining Free cleanup claims and completes only at the final barrier.
7. Still under the queue mutex, the driver makes one final sequence scan through
   cutoff, revalidates all plan handles/generations and every Cutover committed-
   publication predicate, and requires all accepted records Claimed. It sets
   `batch_state=Preparing`, then release-stores every accepted record to
   Applying in PlanPriority order. Because Claimed has only the queue-locked
   driver as writer, these stores are infallible after validation; an armed
   BatchApplyingGuard raw-exits 70 on unwind or mismatch rather than exposing a
   partial batch. After the last store it release-sets `batch_state=Applying`;
   that store is the batch commit point. No root/guest-visible mutation occurs
   before it, and after it terminal transfer must complete the entire no-fail
   batch. It releases the queue mutex.
8. Mutation is object-centric, never `for plan { scan objects }` or
   `for object { scan plans }`. Each stage walks its relevant sorted touched
   slice once and applies each matching G aggregate exactly once: (a) under only
   the JIT-state mutex mark unit/family roots inadmissible, publish final
   dispatch payloads and publish R+1 last; (b) without a subsystem mutex wait
   epoch/signal quiescence, scan FaultSlots once in vCPU order and resolve the
   owning unit cell in O(1); (c) with no subsystem mutex, apply each preverified
   static patch byte sequence and its required cache-sync operation in ascending
   patch index, then separately take only JIT-state to clear/install PIC ways in
   ascending index; PatchRecord tags themselves remain queue-owned until the
   final barrier; (d) detach backlinks under their sole writer; (e) under
   only JIT-state exchange one final native-page COW per ascending page; (f) execute
   mapping-authority operations under their sole writer; and (g) perform
   lifecycle/retirement bookkeeping. No stage holds two subsystem mutexes. The
   aggregate cell, never plan iteration order, determines each store.
9. At the final barrier it takes the queue mutex once, scans records through
   cutoff in sequence order and changes every accepted Applying record to its
   variant-specific Applied/follower result. A post-commit failure first latches
   one TerminalCause, completes every structurally required store/cleanup in the
   same stages, and changes every accepted record to
   TerminatedAfterMutation(the same cause); it never rolls back selectively or
   strands Applying. It clears owner/action/flags/pressure_action and both
   expected generations for every current-G touched cell, asserts the reserved
   byte zero, zeroes all touched lengths and sets batch_state Empty. It takes the
   cleanup ordinal prefix, unlocks, executes each ticket once without a
   subsystem lock, relocks once to publish those exact results and zeroes
   cleanup_len. Finally, with no queue lock held, it takes the cache mutex,
   validates and changes the matching Held freeze to Empty when the optional
   CohortAllocatorFreezeGuard exists, or requires it already Empty otherwise;
   it drops the optional guard and only then permits Open handoff. Terminal
   transfer consumes the same optional guard and performs this clear in its
   no-fail tail.

## Collision rules

Free is decided only by `owner==0` and is not a collision-table output. [Task 0](tasks/00-baseline.md)
generates the exhaustive function
`(domain,winner_variant,winner_action,contender_variant,contender_action) ->
CollisionRule`, where CollisionRule is exactly RequireIdentical,
HigherCutsContender, MergeNativeCow, MandatoryCutoverFollower,
AwaitingLinkCleanup(code) or Invariant. The typed evaluator
then receives both expected generations and ClaimSemanticV1 values. It may
return satisfied/mergeable only after the rule's complete predicate passes;
equal action bytes alone never suffice. RequireIdentical compares canonical
semantic values. The generator applies these ordered rows, which are the full
table rather than examples:

1. any non-Free claim whose contender is InstallLink is
   AwaitingLinkCleanup(StaleNoOp(Superseded));
2. NativePcPage/AggregateReplaceRecords with a non-InstallLink winner and
   contender is MergeNativeCow;
3. identical non-InstallLink variant/action tuples are RequireIdentical;
4. an InvalidateToUnavailable or MappingChange winner's root-cut action against
   a later InvalidateToUnavailable, MappingChange, PressureEvict or UnlinkOnly
   claim for the same exact object is HigherCutsContender;
5. a VcpuDeactivate winner's Vcpu Deactivate or PicWay Clear against the exact
   PressureEvict claim is HigherCutsContender;
6. an InvalidateToUnavailable, MappingChange, VcpuDeactivate or PressureEvict
   winner action which makes the exact CutoverToSuccessor object unavailable,
   fallback-only, cleared or unlinked is MandatoryCutoverFollower; and
7. every tuple not selected above is Invariant.

Rows 4--6 still require matching slot/semantic generations and the rule's
generated typed target/subsumption predicate; mismatch is Invariant. After all
claims, UnlinkOnly alone converts “zero Free and all HigherCuts/identical
satisfied” to StaleNoOp(CoalescedHigherPrecedence); this is a whole-plan rule,
not a hidden collision-table output. A follower publishes only after the cohort
barrier; no follower-edge array exists. [Task 0](tasks/00-baseline.md) enumerates the complete Cartesian
product and proves each tuple selects exactly one ordered row and that every
non-Invariant row has both equality and inequality fixtures.

## Native-page COW ownership

The COW handles and delta wire records are exact:

```text
PageFaultTableHandleV1 = repr(C, align(8)), size 64 {
    header_block:MetadataBlockHandleV1, segment_generation:u64,
    page_index:u32, reserved_zero:u32, table_generation:u64
}
RemoveNativePcRecordV1 = repr(C, align(8)), size 72 {
    captured_ordinal:u16, reserved_zero0:u16, reserved_zero1:u32,
    identity:NativePcRecordIdentityV1
}
NativePageDeltaHandleV1 = repr(C, align(8)), size 112 {
    block:MetadataBlockHandleV1, remove_count:u32, reserved_zero:u32,
    base:PageFaultTableHandleV1
}
```

A nonnull table handle has a FixedSlot PageFaultTableHeaderV1 block, nonzero
matching segment/table generations and page_index below 4096. Canonical Null has
an all-zero header block and table generation but retains the expected nonzero
segment generation/page index; no other mixture is legal. A delta block is a
PagedPayload of exactly `remove_count*72` bytes, count 1..=4096, sorted unique by
captured ordinal, with zero reserved/tail bytes and an owner pin held by the
plan. Its base must be nonnull; an InstallLink addition uses its distinct
prebuilt-replacement handle and never this removal type.

Native-page aggregation uses the live table's charged FixedSlot shadow header:

```text
PageFaultTableSafetyShadowV1 = repr(C, align(8)), size 680 {
    workspace_generation:u64,
    captured_current:PageFaultTableHandleV1,
    alternate_table_block:MetadataBlockHandleV1,
    alternate_records_block:MetadataBlockHandleV1,
    result_table_generation:u64,
    captured_count:u16, resulting_count:u16, reserved_zero:u32,
    removed_bitmap:[u64;64]
}
```

The outer MetadataSlotWrapper supplies the allocation header/owner pin. Both
alternate handles are nonzero: table_block is a FixedSlot
PageFaultTableHeaderV1 and records_block is an IndexedSignalPayload of
RawNativePcRecordV1 with byte_len exactly `captured_count*64` at operation time.
Captured count is 1..=4096; a current pointer with zero records is forbidden and
represented by canonical Null. Every handle/chain/generation is pinned and
unused payload/tail bytes are zero. The current PageFaultTable owner holds the
shadow header's owner pin and the shadow holds owner pins for both alternate
blocks until table retirement; there is no bare-pointer ownership. On first use in G the driver zeroes all 64
bitmap words, recording each word visit in E, then copies the captured identity
and count and sets resulting_count=captured_count. Bits use captured record
ordinal, LSB-first; bits at or above captured_count stay zero.
Every safety NativePageDelta is the canonical NativePageDeltaHandleV1 list
above; Pass A validates ordinal, complete NativePcRecordIdentityV1, table
generation and shadow capacity without writing, and Pass B changes a matching
bitmap bit 0-to-1 and decrements resulting_count exactly once. Re-removing an
identical identity is MergeNativeCow/satisfied; a different identity at an
already named ordinal is Invariant. In stage 8(e), one base-table scan copies
every zero-bit record in existing canonical order to the shadow and exchanges
it. InstallLink is rank 5: it may install its prebuilt current+1 replacement only
when NativePcPage was Free; any existing owner sends it to cleanup. Thus no
current+k capacity or addition merge exists. Every RemoveNativePcRecord and
every base record read/written once is counted in E.

For resulting_count greater than zero, stage 8(e) finishes the alternate
PageFaultTableHeaderV1 with result_table_generation, the alternate record block
and its exact UnitPin set, then release-exchanges the page pointer from the
captured header to that alternate header. Pointer ownership thereby moves the
alternate header/record pins to current; the unchanged safety-shadow owner pin
is transferred to the new current in the same no-fail local ownership move.
The captured header/records become the retired alternate and remain immutable
until retirement epoch R, every preexisting snapshot pin and the cohort's own
snapshot pin are gone. Only then are their old UnitPins released, bytes zeroed
and handles installed as the safety shadow's alternate pair. This drain must
finish before Open handoff, so every nonnull current pointer already owns a
same-capacity alternate shadow when readers can re-enter. For resulting_count
zero, the pointer is release-exchanged to canonical Null; captured and alternate
pairs plus the shadow owner retire after the same grace and no zero-record table
is published. InstallLink from Null precharges two same-capacity table/record
pairs and one shadow header before its publication commit and establishes the
same invariant. At no point is a live table without exactly one owned shadow.

## Work bounds and workspace release

Let Q be all linked MaintenanceRecords examined by one sequence scan through
cutoff, including noncandidate states; `Q <= 3,408,065`. Let V be the sum of the
eight touched lengths; at 128 vCPUs `V <= 5,501,120`. Let E be the sum for one
full canonical traversal of every prepared plan of every PlanClaim/raw Pressure
occurrence plus every variable semantic, COW and allocation/page/span/island
effect record consumed by its generated comparison/merge/projection; no
comparator rescans an already-consumed list. Let
`C = V + 260,086 + 127 <= 5,761,333` be the projection's touched-object,
metadata-page and segment visits. Union, six ranks and the final batch scan cost
at most `8*Q`; union/Pass-A/accepted-Pass-B cost at most `3*E`; radix costs at
most `8*V + 24,576`; Pressure classification plus optional rollback costs at
most `2*V`; and projection costs C. All pre-mutation closure work is therefore
bounded by `8*Q + 3*E + 10*V + C + 24,576` logical visits. Including mutation
(at most two visits for Unit/Family and one for other domains), clearing, the
FaultSlot scan and final result scan, the complete ordinary cohort excluding
the separately bounded epoch/futex waits is at most
`9*Q + 3*E + 13*V + C + K + 24,576 + configured_max_vcpus`, where
`K=cleanup_len <= min(4096,Q)`. The 64 bitmap-zero visits for every first-used
native-page shadow and all native record/delta visits are already E items; no
unlisted loop is absorbed into a constant.

Debug/[Task 8](tasks/08-lifecycle-stress.md) increments checked-u64 record, claim/semantic, radix, projection
and mutation/clear counters at the exact loop bodies above and asserts each
category independently; production has no counters and wall-clock jitter is
not a correctness bound. Workspace-generation exhaustion is preflighted with
G/G+1 before first touch and latches terminal without mutation. Closed allocates,
appends and hashes nothing. Handoff requires allocator_freeze Empty,
batch_state Empty, cleanup_len zero, every current-G cell completely zero except
G, and all touched lengths zero; stale elements beyond a length are never read.

## Sealed plan payloads

The seven sealed plan payloads use this exact mixed-record array:

```text
RawPlanRecordV1 = repr(C, align(8)), size 24 {
    record_kind:u8, domain:u8, action_or_predicate:u8, reserved_zero:u8,
    index:u32,
    expected_slot_generation:u64, expected_semantic_generation:u64
}
PlanClaimSetHandleV1 = repr(C, align(8)), size 48 {
    block:MetadataBlockHandleV1, record_count:u32, action_claim_count:u32
}
```

For each non-Pressure variant, the referenced PagedPayload block is exactly
`record_count * 24` bytes, with ActionClaim records followed by
CommittedPublicationPredicate records and each region sorted unique by
`(domain,index)`; `action_claim_count <= record_count <= 5,501,120` and
Pressure uses zero/zero with no block. Every byte is initialized and there is no
trailing record padding. The generated iterators use page-aware
`read_record<RawPlanRecordV1>`, validate raw fields before constructing semantic
enums, and never form a typed slice. The maintenance record owns its MetadataBlockPin
from successful preparation through final result/handoff; it then releases the
pin and decommits/reuses the pages through the normal arena protocol. The
`root_mutation_plan_claim_pages` row is logical_capacity_only with backing_row
`metadata_arena_pool_pages`; its matching PagedPayload metadata_slab row
describes type/capacity, while actual pages charge exactly once through the
pool row, with
max_object_pages=33,273. Pressure contains no `claims` field. Every plan,
including Pressure, contains `required_claim_count:u32`,
`owned_claim_count:u32` and `satisfied_claim_count:u32`; the other six variants
also contain `claims:PlanClaimSetHandleV1`. For a non-Pressure plan, required
equals `action_claim_count`; Cutover's checked
`publication_predicate_count = record_count-action_claim_count`, and every other
variant requires that difference zero. For Pressure, required is the number of distinct combined
claims from its rank-2 expansion. Once accepted, required equals checked
`owned + satisfied`; rejected plans have both latter counts zero. Existing
per-domain count fields are immutable original ActionClaim counts, whose checked
prefix sums are the generated domain starts used by the semantic accessor; they
are never rewritten to owned counts.

- `InvalidateToUnavailable { victim: UnitHandle, code_version,
  reachability, dependency_cause, dispatch_count, unit_count, family_count,
  patch_count, native_page_count, pic_count }` owns those six mark domains;
- `PressureEvict { requested_total_bytes, input_ledger_sequence,
  projected_total_bytes, dispatch_count, unit_count, family_count, patch_count,
  native_page_count, pic_count }` owns those six domains;
- `CutoverToSuccessor { successor: HcqFamilyHandle, successor_version,
  build_fingerprint, predecessor_count, publication_predicate_count,
  dispatch_count, unit_count,
  family_count, patch_count, native_page_count, pic_count }` owns those six
  cleanup/stabilization domains and revalidates, but does not reclaim, its
  committed-publication records;
- `UnlinkOnly { victim: UnitHandle, code_version, unlink_reason, patch_count,
  unit_count, native_page_count, pic_count }` owns unit/patch/native-page/PIC
  marks and is sealable only when ClosingPrepare observes that a prior commit
  already removed all dispatch/family roots or a higher plan in this cohort
  owns their exact cuts. All claims satisfied means StaleNoOp; mixed satisfied
  and Free accepts the Free cleanup remainder as specified above;
- `MappingChange { address_space: AddressSpaceHandle,
  operation: MappingOperationHandleV1,
  expected_mapping_generations:MappingGenerationVectorHandleV1,
  resulting_mapping_generations:MappingGenerationVectorHandleV1,
  dispatch_count, unit_count, family_count,
  patch_count, native_page_count, pic_count }` owns the address-space plus all
  six executable-root domains;
- `VcpuDeactivate { vcpu: VcpuHandle, expected_vcpu_generation,
  expected_fault_word, retry_root_predicate, pic_count }` owns its vCPU/PIC
  marks; and
- `InstallLink { patch_record: PatchRecordHandle, source: UnitHandle,
  target: UnitHandle, bridge: None|UnitHandle, expected_versions,
  native_page_count }` owns one patch mark and its native-page marks.

All count fields are checked u32. Immutable per-domain counts equal the decoded
ActionClaim stream; required/owned/satisfied equal arbitration as defined above.
Pressure's per-domain counts are distinct post-join objects, never raw duplicate
occurrences. Every named handle/generation is revalidated immediately before
Claimed-to-Applying. The fixed two-pass greedy arbitration and Pressure rule
above are the only closure algorithm. There is no replay, loser restoration,
heapsort, fixed-point iteration or per-victim projection. A nonfollower atomic
plan is accepted only when each claim is owned or explicitly satisfied;
MandatoryCutoverFollower uses its stated exception. Closed never applies a
partial or unaccepted plan.
`MappingOperationHandleV1`, dependency cause, unlink reason, expected/resulting
generation records and action payloads are closed `repr(C)` types generated in
[Task 0](tasks/00-baseline.md), with no callback/trait object. It is constructed and revalidated before
the first mutation; the driver never infers either object from reason bits and
never appends roots to it after ClosingFreeze.
`PressureEvict` is the one composite HCQ/optional-LCQ plan defined by the cache
policy in [executable-cache pressure handling](cache.md).

The coordinator, not each record, owns one charged, preallocated
`cohort_fault_snapshot[configured_max_vcpus]`. Each entry can hold only a
VcpuHandle/generation, the acquire-loaded FaultSlot state/sequence, UnitHandle,
CodeVersion and retry-root identity; empty/unregistered slots have an explicit
Empty tag. RootMutationPlans contain only immutable exact affected-unit/version
predicates. This one array is overwritten once per cohort only after all active
epochs are zero, is never retained past handoff and adds O(vCPU) process storage
rather than O(vCPU × record) storage.
