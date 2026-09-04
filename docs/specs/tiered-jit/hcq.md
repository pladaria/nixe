# HCQ admission, region formation and reshape

[Specification index](README.md) · [Implementation tasks](tasks/README.md) · [Open review items](review-status.md)

Related contracts: [Control budget and functional sampling](sampling-and-budget.md); [Code units, versions, registries and admission](units-and-registries.md); [Compilation and publication pipeline](publication.md).

## HCQ selection and region formation

## Admission

A cold poll which elects to attempt HCQ first performs a nonwaiting acquisition
of an exact `Open(E)` OpenToken before claiming an ActiveBuild cell or family/
OwnerCell anchor. Failure leaves no active-build, anchor or queue state and is a
transient admission miss. The attempt retains that token until either its
generational handle is visibly inserted in the pending queue and the queue/cell
own all immutable input, or every partial cell/anchor mutation has been rolled
back. Immediately before the queue push it acquire-revalidates the identical
Open(E); a Closing/Stopping observation pushes nothing and takes the ordinary
rollback path. The OpenToken decrement occurs only after unlocking the queue.
Consequently, a transition which has drained OpenTokens has also closed every
possible ActiveBuild/queue producer; sampling itself may still update the
caller-owned fixed hot table but cannot enqueue work.

A process has a fixed `ActiveBuildTable` of
`configured_max_vcpus + pending_capacity + worker_count` cells, where
`pending_capacity = 8 * worker_count` exactly and is zero when worker_count is
zero. A cell contains
a checked unique BuildToken, complete AdmissionFingerprint, optional finalized
BuildFingerprint, immutable snapshot,
captured admission epoch, anchor DispatchSlot generation, zero, one or two
participating family generations and
an atomic state:

```text
Free(g) -> Initializing(g, BuildToken)
Initializing -> AdmissionReserved|Cancelled(reason)
AdmissionReserved -> Queued|Cancelled(reason)
Queued -> Building|Cancelled(reason)
Building -> Published|Rejected(reason)|Cancelled(reason)
Published|Rejected|Cancelled -> Free(g + 1)
```

The configured-vCPU term covers simultaneous admission attempts; the other
terms cover every queued/building attempt; one vCPU serializes its own cold poll
and can own at most one Initializing cell. A cell is claimed by a Free-to-
Initializing TaggedPayload transition while holding a successfully try-locked
`active_build_writer_mutex`; contention cancels admission without changing the
cell. The winner owns one initialization pin, fills immutable data before a
release transition to AdmissionReserved, and the cell is never
reused until its terminal outcome is reflected in payload/cache/absence, every
exact reservation is detached and both owner pin and `attach_reader_count` are
zero. There is no naked Free-to-Initializing word CAS.

A cold score threshold constructs its AdmissionSnapshot or ReshapeSnapshot in
fixed vCPU scratch, performs a nonblocking exact NegativeBuildCache lookup and
stops with the score saturated on a matching structural result. On a miss it
claims one free active-build cell and then reserves its semantic anchor.

For ordinary admission it first captures the seed slot handle/generation,
payload pointer, ReachabilityVersion, LCQ CodeVersion and every fingerprint
input, then CASes only `hcq_anchor` from zero to its BuildToken. After that CAS
it acquire-reloads the slot generation and payload and retains the token only
if the slot/key, ReachabilityVersion, LCQ CodeVersion and complete
AdmissionFingerprint
inputs still equal the capture. Otherwise it CASes only its exact token back to
zero and cancels. No `(ReachabilityVersion, no_build)` tuple CAS exists and no
128-bit atomic is assumed.

For reshape it uses the JIT-state mutex and exact FamilyLifecycleCell transition from
`CurrentStable(family_version)` to
`ReplacementReserved(family_version, BuildToken)` in ascending HcqFamilyId
order. The lowest-HcqFamilyId member is the serialization anchor; no separate
boundary-source anchor exists. After all CASes it acquire-revalidates both
endpoint InstructionKeys, owner handles, family versions, ExecutionKey and
fingerprint inputs. Failure rolls back only its exact reserved states; an exact
state already changed to Withdrawing is left untouched.

If the first required anchor already carries another BuildToken, admission
try-locks that token's generational active-build cell, validates
`AdmissionReserved|Queued|Building`, generation/token, AdmissionFingerprint and
captured epoch, checked-increments `attach_reader_count`, copies the immutable
snapshot reference, and unlocks. Contention is transient. It attaches only if
the copied admission fingerprint and
captured admission epoch match, and a second acquire-read proves that the same
token still occupies the ordinary slot anchor or every required family state
and that all captured slot/endpoint generations remain equal. For a two-family
reshape, encountering another token after this attempt reserved the first
family first rolls back its own exact state and then applies that complete
attachment check; partially matching tokens never attach. Every attachment
exit first AcqRel-decrements the existing cell's attach-reader count. A
successful attach then try-locks its own unused cell mutex, revalidates exact
Initializing/BuildToken, publishes
`Cancelled(AttachedTo(existing_token))`, releases its initialization pin and,
if both counts are zero, publishes Free(g+1) as a second ordinary
TaggedPayload transition; otherwise it leaves recycle_pending for the common
unlock epilogue. A different, vanished or reused anchor similarly rolls back
exact partial leases, then under its own cell mutex publishes
`Cancelled(AnchorConflict)` and follows the same pin/recycle protocol; its score
becomes seven/three. If a canceller already changed that exact cell, admission
does not overwrite it and only releases its own pin through the common owner
protocol. No direct tag/generation CAS and no hash-only match deduplicates work.

After acquiring the anchor tokens it finishes the cell and release-stores
AdmissionReserved.

It next uses this no-nesting enqueue protocol for the preallocated pending
VecDeque, whose exact capacity is eight cells per worker. If no Free active-
build cell was claimed, the attempt leaves every cell, anchor, score and queue
byte unchanged; the already-saturated score remains saturated and the guest
neither waits nor spins. Otherwise it locks only the cell writer mutex,
requires exact AdmissionReserved and no matching cancel token, publishes
Queued, and unlocks. It then try-locks only the pending queue. Contention or,
after locking, fullness causes it to unlock, reacquire only the cell mutex and
change the still-exact Queued state to `Cancelled(QueueContention)` or
`Cancelled(QueueFull)`; it then rolls back exact anchors in a separate JIT-
mutex stage. If queue capacity exists, while holding only the queue mutex it
acquire-revalidates exact Queued generation/BuildToken and that
`cancel_build_token != BuildToken`; only then does it push the generational
handle with a no-fail operation. A failed revalidation pushes nothing. Thus
Queued may briefly precede its queue handle, but cancellation is visible before
push and a worker sees cells only through handles actually inserted.

After a failed queue-side revalidation, admission unlocks the queue, then under
only the cell mutex changes its still-exact Queued state to
`Cancelled(CancelToken)`; if a canceller already terminalized it, admission
does not write another state. It next rolls back only its anchors under the
JIT-state mutex and releases its exact owner pin. No Queued cell without a queue
handle is left live.

Every cancellation path releases the admitting cell pin. A terminal owner under
the cell mutex publishes the terminal tag and, when attach_reader_count is
nonzero, sets `recycle_pending=true` instead of reusing storage. The last reader
release only decrements, wakes and makes one nonblocking try-lock attempt; on
failure it leaves recycle_pending set. The unlock epilogue of every later cell-
mutex owner revalidates terminal state, zero counts and recycle_pending and
publishes checked next-generation Free. Thus no guest admission or attach path
blocks, spins or dereferences recycled immutable data. Queue contention/fullness leave seed score seven or boundary
score three. A worker pops under the queue mutex, unlocks it, then takes only
the cell mutex and changes exact Queued to Building; a stale or cancelled
handle is discarded.

The queue owns `newest_streak` initialized to zero. Under its lock, dequeue uses
the back and increments the streak while it is below seven; when it equals seven
it uses the front and resets to zero (an empty selected end falls back to the
other end). All workers share this counter, yielding exactly seven newest then
one oldest successful dequeues. On successful
publication the coherent DispatchPayload contains the exact fingerprint before
the build cell becomes Published/Free. On permanent shape rejection the worker
inserts the exact NegativeBuildCache result before Rejected/Free. On stale,
shutdown or ownership cancellation it records Cancelled, releases matching
tokens/reservations and leaves neither a payload nor negative result. Thus the
observable compile state for a fingerprint is derived, in order, from an exact
active cell (`AdmissionReserved|Queued|Building`), matching published payload, matching negative
cache, or `Idle`; this is the sole deduplication protocol. Guest execution stays
in LCQ and no admission path allocates or waits.

The queue handle and then the worker own one attempt-cell pin. Invalidation or
Shutdown may remove a still-Queued handle under the queue mutex and complete it
Cancelled, but for Building it stores that cell's exact BuildToken in
`cancel_build_token`; the worker
alone performs terminal cleanup after dropping all CodeUnit pins. Every worker
checks `cancel_build_token == its BuildToken` after dequeue, after
discovery/reservation, after backend
work and immediately before OpenToken acquisition. A cell returns to Free only
when its queue/worker pin count is zero. An invalidator never reuses a cell or
releases a reservation still consumed by a running worker.

Worker count is selected once from the policy formula. Each worker owns its
decoder scratch, Cranelift context and register allocator. There is no shared
compiler mutex, task-per-thread spawning or general runtime thread pool.
With zero workers HCQ is disabled and no pending container exists.

TerminalControl contains four persistent, cache-line-separated
`WorkerControlCellV1` values; only indices below `worker_count` are enabled.
Each cell is exactly `{ state:AtomicU32, parking_sequence:AtomicU32,
current_build_token:AtomicU64, counted_live:AtomicU32, reserved_zero:[u8;44] }`
and therefore 64-byte aligned/sized. State discriminants are `Unused=0`,
`Running=1`, `StopRequested=2`, `TerminalDriver=3` and `ExitedNoArena=4`.
Enabled cells are Running before Ready and have counted_live one; disabled cells
are Unused/zero. TerminalControl's `worker_live_count` equals the sum of those
bits. A worker release-publishes its BuildToken immediately after changing the
corresponding ActiveBuild cell to Building and clears it only after that token's
pins, anchors, reservations, Awaiting obligation and attempt-cell pin are all
released. It checked-increments and wakes `parking_sequence` after every
state/token change. The word starts at one; ordinary work may publish only
values `1..=u32::MAX-4`. The final four advances are reserved for token clear,
StopRequested or TerminalDriver, ExitedNoArena and one fatal-cleanup wake. An
ordinary advance which would enter that reserve disables new HCQ admission and
latches WorkerInvariant before mutation; no parking word wraps. Every advance
uses release ordering followed by `FUTEX_WAKE_PRIVATE(INT_MAX)`, and every
waiter uses snapshot/recheck, so the word is notification rather than authority.

TerminalControl also contains
`worker_start_gate: AtomicU32` (`Constructing=0`, `Run=1`, `Abort=2`). It has no
second parking-sequence word: a child acquire-loads the gate, rechecks zero and
uses `FUTEX_WAIT_PRIVATE` directly on that aligned AtomicU32 with expected zero;
the parent release-stores Run or Abort exactly once and
`FUTEX_WAKE_PRIVATE(INT_MAX)`. Before spawning, all arena/control state and four
WorkerControlCells are initialized. For each index in ascending order, a
successful `spawn` returns exactly `JoinHandle<()>`; while the gate remains
Constructing the parent, under WorkerJoinSet's mutex, changes that index
Absent-to-Present and stores the handle. It then enters a non-unwinding commit
tail which CASes `counted_live` 0-to-1, checked-increments
`worker_live_count`, changes Unused-to-Running and wakes that cell. A failed
CAS/count or unwind in this tail raw-exits 70; there is no recoverable state in
which a stored handle lacks its live accounting. Only after the complete
prefix and every arena/runtime field a child may read are stored/accounted does
construction arm `ConstructionPublishGuard`, enter a non-unwinding tail,
release-store Run and wake all children, and finally release-publish process
Ready as the last store. It then disarms the guard. Run authorizes children
because all child-visible state already exists; Ready is the external process-
handle publication. An unwind between them raw-exits 70 and cannot expose a
runnable child with partially constructed state.

If any spawn or later setup fails while the gate is still Constructing,
`ConstructionRollbackGuard`
release-stores Abort and wakes all children before taking any handle. Each
started child has read no reclaimable runtime state other than its initialized
control cell; it changes Running-to-ExitedNoArena, consumes counted_live
one-to-zero, decrements/wakes `worker_live_count` and returns `()`. The guard
joins exactly the stored prefix using the normal JoinSet protocol and requires
all its cells ExitedNoArena/count-zero and `worker_live_count == 0`; mismatch or
join panic is raw-fatal. No child can claim terminal or dequeue work while
JoinSet ownership is incomplete. Once Run is stored, rollback is illegal and
the remaining Ready store is the guarded no-fail suffix above.

Stopping changes each other Running cell to StopRequested and wakes it. A
worker which latches or observes Requested first finishes only an already-
entered no-fail publication tail, cancels/cleans its exact BuildToken and calls
`drive_or_join_terminal(origin=Worker(worker_index))`. Winning the terminal
claim returns `Won(TerminalDriverGuard { origin:self })`. The only other
returns are `LostClaim` when its Requested-to-Claimed CAS loses and acquire-
observes Claimed/Applied, `OwnedByNormalDriver` when Closing/Closed carries a
normal DriverToken different from its optional current token, or
`AlreadyStopping` when it owns no TerminalDriverGuard. Any of those three, or
an acquire-observed StopRequested, proves another driver and permits the loser
exit. The shared pre-reserved shutdown DriverToken is never compared to infer
thread identity. It never waits for ShutdownRecord Applied. As its
last access to the reclaimable JIT arenas it proves `current_build_token == 0`,
release-stores ExitedNoArena, atomically consumes counted_live one-to-zero,
release-decrements worker_live_count, wakes on the decrement and returns from
the OS thread entrypoint. If a worker becomes terminal driver, it first performs
the same current-BuildToken cleanup and then must win Requested-to-Claimed plus
the phase-to-Stopping CAS and receive
`TerminalDriverGuard { origin: Worker(worker_index), .. }`. Only then may it CAS
its own still-Running state to TerminalDriver, consume the two live-count bits
exactly once and exclude precisely its own JoinHandle from the terminal join
set. Seeing StopRequested proves another driver won and permits only the loser
path to ExitedNoArena. An external winner receives
`TerminalDriverGuard { origin: External, .. }` and has no WorkerControlCell to
modify. Only a `Worker(i)` winner, after publishing ShutdownRecord Applied,
release-stores its own TerminalDriver cell to ExitedNoArena as its final JIT
access and returns; it does not re-enter the worker loop. The external process owner, never a worker,
owns all JoinHandles. It joins the terminal-driver handle after observing
Applied, before an explicit shutdown call returns and before destroying the
last process handle or TerminalControl page. `ProcessOwner::drop` is the only
destructor path allowed to initiate this final join, and only after it has
itself acquire-observed both Stopped and Applied and constructed the same
AppliedShutdownJoinGuard; it never drops or detaches a raw JoinHandle.
The worker entrypoint returns only `()`. A terminal driver of either origin
retains its arena-aware TerminalDriverGuard through step 8a, which performs all
remaining registry/directory/pin scans and constructs the exact
TerminalUnmapPlanV1 while both arenas are mapped. At the 8a/8b barrier it
consumes every local value whose Drop implementation or borrow can touch either
arena, the pending queue, a registry, a BuildToken, mutex or pin, marks the
external ProcessOwner inert for both External and Worker driver origins, and converts the guard into the
audited non-generic `TerminalUnmapGuard` containing only a TerminalControl
pointer, origin/index and Copy unmap-plan scalars. Step 8b unmaps only the exact
plan ranges; step 9 and a worker driver's final ExitedNoArena store access only
TerminalControl. The tail constructs no arena-backed or fallible Drop value.
[Task 8](tasks/08-lifecycle-stress.md) instruments both External and Worker origins and asserts that every
destructor executed after the barrier touches only process-owner/
TerminalControl memory. A non-unit JoinHandle result or any retained arena-
aware destructor is an architecture violation.
JoinHandle ownership is one `WorkerJoinSet` in the process owner, outside both
JIT arenas: `{ mutex, handles:[Option<JoinHandle<()>>;4],
join_state:[Absent|Present|Joining|Joined;4],
join_sequence:[AtomicU32;4], final_driver_index:Option<u8> }`. Exactly three
guards may operate it. A validated TerminalDriverGuard during Stopping handles
all enabled indices for origin External or all except its own i for origin
Worker(i); before unlocking it records final_driver_index None or Some(i)
respectively. An `AppliedShutdownJoinGuard`, constructed only after one caller
acquire-observes both phase Stopped and ShutdownRecord Applied, handles only
the stored Some(i). A `ConstructionRollbackGuard` exists only before Ready and
handles exactly the successfully stored prefix.

Under the mutex, a permitted guard changes Present to Joining and takes the
Some handle; it unlocks before `join`, then relocks, changes that same index to
Joined, release-increments join_sequence and wakes all waiters. A competing
Applied guard which sees Joining snapshots the sequence, unlocks and futex-
waits/rechecks; Joined succeeds without taking. Only Present with None,
Absent/indices outside the guard's exact set, a changed final_driver_index or a
join error is corruption/raw-fatal. Thus two external callers cannot double-
take or destroy control state while one joins. Bulk construction/terminal
joins use the identical Present-to-Joining-to-Joined transitions. No Drop path
other than the guarded `ProcessOwner::drop` route above consumes a handle; no
JoinHandle Drop is allowed to discard/detach a joinable thread.
`final_driver_index` becomes immutable before Applied.
From the first Present-to-Joining transition until Joined is published/woken,
each of the three guard paths runs under a non-unwinding join guard. Any panic,
join error, poisoned/relock failure or state mismatch in that interval invokes
raw `SYS_exit_group(70)`; even pre-Ready rollback cannot abandon a local
JoinHandle while other observers see Joining.
`WorkerJoinSet`, the four retained compiler contexts and their fixed-capacity
scratch are the only per-JIT process-heap allocations permitted to survive
Ready. They are deliberately outside the JIT ledger and `layout_row` catalog:
`ProcessOwnerControlReportV1` records, for each host, checked
`size_of`/`align_of` for WorkerJoinSet, its exact four-handle capacity, worker
count, and the requested byte/capacity bound of every retained scratch buffer.
[Task 0](tasks/00-baseline.md) freezes those per-host bounds and [Task 2](tasks/02-lifetime-foundation.md) rejects a runtime type/capacity
mismatch. Allocator overhead is not asserted as JIT charge; external PSS counts
it. No executable root, metadata payload or cleanup obligation may be stored in
either excluded allocation. WorkerJoinSet outlives executable/metadata unmap
and is destroyed only after every required successful join.

Every worker entrypoint has a catch-all abort-on-unwind boundary. A panic while
it is an ordinary worker may not masquerade as a clean exit: unless the proven
no-fail cleanup epilogue has already cleared its BuildToken/Awaiting ownership,
consumed counted_live and decremented worker_live_count, it invokes raw
`SYS_exit_group(70)`. A worker holding TerminalDriverGuard wraps the complete
terminal drive, including pre-root-cut steps 1--4, in an unconditional
abort-on-unwind guard; nobody can resume that guard, so any unwind immediately
uses the same raw exit.

After dequeue, the worker's attempt-cell pin protects its immutable snapshot
and reservation state; each demanded code image is protected only by the
explicit UnitPin cloned during the lookup below. There is no additional
compiler epoch, hazard domain or implicit read-side protection. Discovery is
incremental: when the bounded worklist names a BlockKey, the worker briefly
uses the JIT-state mutex to validate the captured admission epoch and clone that
LCQ CodeUnit's UnitPin/immutable reference, then releases the mutex before
decoding it. It scans neither the registry nor all resident
LCQ units. Each worker precharges a 4101-entry FIFO/reference array and an
8192-bucket no-tombstone linear-probe seen set keyed by full CanonicalBlockRef.
The exact maximum is `1 seed + 4 sampled initial refs + 2 * 2048 accepted-block
successors = 4101`; reshape needs at most 4098. Enqueue first inserts into that
set and retains one UnitPin until planning ends. Reaching 4101 before the
derived accepted-block bound is an invariant failure, not a dynamic growth.
Thus a job holds at most 4101 pinned LCQ refs while decoded/selected candidate
contents remain capped at 2048 distinct InstructionKeys. A job never reads a
per-vCPU profile table or a raw registry
pointer. Eviction/invalidation may remove units from indexes but cannot reclaim
their instruction images or metadata until every worker reference is dropped.

## Execution-informed graph

HCQ is a non-speculative region over demand-proven LCQ blocks. A block is
eligible only when:

- its exact LCQ was published and therefore actually demanded;
- its ReachabilityVersion and mapping generations still match;
- it has the snapshot's one ExecutionKey.

The worker explores its immutable snapshots without a shared planner lock. Its
worklist is one FIFO deque plus a set of already-enqueued
named `CanonicalBlockRef { lcq_unit, leader }` values; each `lcq_unit` is
UnitPin-protected while retained;
a ref is inserted in the set when first enqueued, so no later edge changes its
position. These are exactly the fixed 4101-entry/8192-bucket structures above;
there is no heap growth or hidden per-edge list. Every active HcqFamily retains a bounded member-origin map from each
owned InstructionKey to exactly one such LCQ ref. Initial construction chooses
the lowest `MemberOriginKey` by lexicographic StableEncode (therefore including
both UnitHandle slot index and generation) among
identical overlapping origins and reshape reuses that stored choice.

Ordinary initialization enqueues the seed's current LCQ ref and then distinct
refs resolved from its normalized successor list in stored order. The
normalization rule above guarantees that this list contains no call or return;
an `IndirectBranch` target may therefore become a public root, but the dynamic
edge is never rediscovered from live state. Reshape
resolves each directed BoundaryEndpointIdentity through its LCQ identity or
family member-origin map, enqueues the source first and the target second if
distinct, and marks both mandatory; it has no ordinary successor list. Failure
to resolve an exact captured identity is stale. `discovery_root` means the
ordinary seed ref for Ordinary or the directed source ref for Reshape; every
later reachability operation uses this definition. On accepting a block, discovery
assigns the next u32 ordinal and enqueues its eligible direct
CFG targets in this exact order: the one unconditional successor, or
conditional fallthrough followed by conditional taken. Equal-category targets
which are produced as a set rather than by one instruction are sorted by full
BlockKey before enqueue. Calls, returns and dynamic targets do not recursively
enqueue anything. The deque front is always popped; there is no recursion,
hash-table iteration order or priority recomputation.

Unconditional and conditional successors are considered only if their exact
LCQ is already published. Sampled successors come only from the immutable
AdmissionSnapshot; no worker consults live heat state. Direct/indirect calls,
their callees and returns are never initialized or recursively enqueued by the
HCQ graph. A return
continuation is an
independent LCQ/HCQ seed; it is not pulled in merely to compensate for a slow
call ABI. SVC, BRK, FP-mode changes, unsupported instructions and runtime
boundaries terminate the region.

LCQ fragments may overlap when execution first demanded an interior PC. HCQ
does not copy that overlap. It merges matching decoded instructions by
InstructionKey and creates canonical HCQ leaders at every ordinary seed or
reshape endpoint, demanded
entry, observed target and post-terminator PC. An existing sequence is split at
a new leader; conflicting instruction bytes or execution keys make the
snapshot stale. Discovery does not query OwnerCell, ownership reservations or
live families at all; it always discovers the same complete <=2048-key superset
from its immutable snapshots. Foreign ownership is applied only by the single
post-discovery transaction below.

Before accepting a popped item, discovery decodes its complete canonical LCQ
block snapshot and computes the new distinct-InstructionKey total after overlap
deduplication. If that total is at most 2048, it accepts the complete block; if
not, it accepts none of that block, enqueues none of its successors and leaves
the incoming edge external. Failure to fit the discovery_root or a mandatory reshape
endpoint is `HcqRejected(OverInstructionCap)`; failure to fit a nonmandatory
item is not itself a rejection. At total 2048, discovery skips remaining
nonmandatory items but continues popping every initialized mandatory endpoint:
an endpoint whose distinct instructions do not fit rejects the shape, while a
fully overlapping endpoint is accepted. Discovery succeeds only after the
discovery_root and every variant-specific mandatory endpoint are explicitly
marked accepted; it then discards
the residual deque. Canonical leader splitting is performed by ascending
InstructionKey, independent of insertion/hash order. There is no
independent block, page, dependency, component, span or public-entry limit.

Selected public entries are:

- the ordinary seed, or both directed reshape endpoints;
- every included `IndirectBranch` target named by the immutable normalized
  ordinary successor list;
- every target of an internal backedge carrying the mandatory control-budget
  check, so canonical resume after an expired poll has a real dispatch entry;
  and
- every explicitly demanded interior entry named by a reshape endpoint.

All other blocks are internal. Live incoming links are deliberately not a
planner input and there is no final live dispatch/link-index sweep. An outside
link whose target is not in this immutable list remains bound to that target's
current LCQ or permanent canonical fallback; neither an existing nor a racing
link may create an HCQ label or retarget itself to an internal block. A later
real sample may make that target an ordinary seed or reshape endpoint. Thus the
same AdmissionSnapshot and pinned CodeUnit set always select the same public
entries regardless of link-install timing, and no dispatch slot is created for
every instruction.

## Overlap and reshape

Active HCQ membership and in-flight reservation are exclusive by
InstructionKey. The LCQ CodeUnits from which a family was built remain pinned
while that family is active, so every optimized entry has a correct baseline
to restore. Ownership exists only in cold compiler state and does not affect
native lookup or execution.

The owner index admits exactly 1,048,576 live OwnerCells and uses one fixed
2,097,152-bucket Robin-Hood array. Set hash is FNV-1a-64 of the complete
InstructionKey StableEncode masked by 2,097,151; full equality follows the hash,
lookup terminates by probe distance, and deletion backward-shifts so there are
no tombstones or resize. A checked generational OwnerCell slab is committed
lazily; its allocation bitmap, bucket array and committed cell pages appear as
named JitLayoutReport/ledger rows. One reservation transaction receives the
candidate's sorted distinct keys, counts absent cells, verifies the live limit,
preclaims all required cells and bucket positions, and then publishes all
reservations or none under the JIT-state mutex. Capacity failure returns
`HcqTransient(OwnerCapacity)`, changes no cell, creates no negative-cache entry
and never affects LCQ. The owner index has one reusable `OwnerCell` per currently owned/reserved
InstructionKey. Its independently validated fields are:

```text
owner       = None | Owned(HcqFamilyId, family_version)
reservation = None
            | NewOwner(BuildToken, reservation_generation)
            | Reshape(BuildToken, reservation_generation,
                      expected_owner_family_version)
```

`NewOwner` is valid only with no owner. `Reshape` is valid only for the exact
participating owner; that owner remains current during compilation. An empty
cell is removed/reused after its grace period. Owner slots and reservation
generations are precharged; no insertion allocates while holding the JIT-state
mutex.

A normal candidate treats stable foreign ownership as a side-exit boundary, but
never makes discovery depend on another build's timing. After complete
optimistic discovery it takes the JIT-state mutex once, revalidates the
AdmissionFingerprint, captured epochs/versions and exact HcqShapeVersion, and
copies one OwnershipObservation for every discovered InstructionKey in ascending
key order into its fixed 2048-entry array. That complete array and version form
OwnershipSnapshot and finalize the cell's BuildFingerprint before trimming or
negative-cache lookup. If any key has a foreign
non-None `reservation`, or if its seed is no longer unowned, it cancels the
whole attempt as a transient ownership race, changes no OwnerCell and records
no negative result. Only stable `Owned(...)` fields may influence trimming.
The observations contain no reservation because all must be None at successful
snapshot. Otherwise, still under that mutex, it processes every canonical block in
discovery-ordinal order by this exact operation: scan the block's
InstructionKeys in guest order, keep only the nonempty prefix strictly before
the first such key, replace that prefix's original terminator/outgoing edges
with one external side exit to the foreign key, or drop the whole block when
the first key is foreign. Later foreign keys are irrelevant because their
suffix is already gone. It then performs one FIFO reachability pass over the
remaining discovery-ordinal edge lists from the ordinary seed, drops every
unreached block, recomputes external exits and installs `NewOwner` reservations
for resulting keys in ascending InstructionKey order. It does not re-decode,
split at any other point or preserve an original outgoing edge from a cut
block. This bounded pass visits each candidate key/edge once and performs no
allocation or backend work. Reservation/publication revalidate the identical
HcqShapeVersion and every observed owner; an intervening ownership change
cancels rather than producing a different shape under the same BuildFingerprint.

A reshape already owns its ordered
`ReplacementReserved(family_version, BuildToken)` lifecycle states from
admission. The
same transaction accepts keys owned by those exact family versions and marks
them `Reshape`, reserves unowned additions as `NewOwner`, and treats every
other stable owner as foreign. Before trimming, any OwnerCell reservation not
already owned by this exact reshape BuildToken cancels the complete attempt as
a transient ownership race with no negative result; it is never converted into
a side-exit boundary. It then applies the same prefix-cut operation and
FIFO reachability from the directed source endpoint; both mandatory boundary
endpoints must survive and the target must be reachable, otherwise the attempt
is cancelled as an ownership race without a negative result. Optional foreign
and disconnected blocks are trimmed by that same rule.

Only after this transaction does liveness/lowering begin. Publication
revalidates every reservation. For a reshape it atomically marks predecessor
families CutoverOld and their member units Superseded, changes selected keys to
the successor owner, restores
dropped keys to no owner/LCQ and clears exact reservations. Ordinary publication
changes every selected reservation to its new owner. Abort/invalidation clears
only fields whose BuildToken and reservation generation both match, never a
newer reservation or current owner. Thus unrelated workers may compile in
parallel but never begin backend work over the same nonparticipating
InstructionKey.

The cold sampler also observes actual HCQ-to-LCQ and HCQ-to-HCQ boundary edges.
Four samples of the same generation-valid boundary request a reshape:

- exactly one or two distinct CurrentStable families adjacent to that edge
  participate; zero makes the sample stale and routes any LCQ-only seed through
  ordinary admission;
- at most one reshape is in flight for each participating family generation;
- discovery may use members of those families plus eligible unowned blocks;
- the two boundary endpoints are mandatory;
- the same deterministic worklist selects at most 2048 instructions;
- publication marks every participating old family CutoverOld as a whole;
- selected blocks transfer to the successor family; and
- old members not selected fall back to their retained LCQ.

If a useful candidate containing both endpoints cannot be formed within the
instruction ceiling, the owners remain separate and the direct link remains.
This is a valid optimized boundary, not permission for overlapping bodies.
The NegativeBuildCache records `OverInstructionCap` or
`DisconnectedMandatoryEndpoints` under the complete fingerprint. It cannot
retry that same shape until any BoundaryEndpointIdentity component (LCQ
ReachabilityVersion/CodeVersion or HCQ family/member CodeVersion),
HcqShapeVersion, instruction-observation cursor, immutable snapshot or code
dependency changes; BoundaryTable contains no worker result.

Thus the first hot root does not permanently partition the graph. Region shape
can grow, shrink, merge or repartition according to execution, while active
HCQ duplication stays bounded to old/new versions protected during
publication.

## Multi-entry lowering

HCQ lowers one region with internal SSA edges. Each selected entry has:

- one exact guest PC;
- one independently computed live-in contract;
- a real native body label exported by the backend; and
- a canonical ingress adapter which loads only those live-ins.

Internal edges do not pass through adapters, dispatch slots, safepoints or
canonical state. External exits use the fast-chain protocol. The body is
compiled once with opt_level=speed and backtracking allocation, then placed as
one immutable code version. No ordinal, br_table, wrapper function or
per-entry copy of dependency/fault metadata exists.
