# Compilation and publication pipeline

[Specification index](README.md) · [Implementation tasks](tasks/README.md) · [Open review items](review-status.md)

Related contracts: [HCQ admission, region formation and reshape](hcq.md); [Executable cache, metadata and backend ownership](cache.md); [Request execution, terminal transfer and cohort handoff](coordinator-execution.md); [Epoch reclamation and terminal teardown](epochs-and-shutdown.md).

## Compilation and publication pipeline

The exact pipeline is:

~~~text
sample seed
  -> nonblocking bounded admission
  -> optimistic immutable graph discovery
  -> short generation/ownership validation and batch reservation
  -> owned instruction-image capture and dependency freeze
  -> liveness and selected-entry formation
  -> backend lowering into a nonexecutable staging image
  -> executable-cache span allocation and relocation
  -> memory/dependency revalidation and new-code cache synchronization
  -> no-fail metadata-first publication plus registered link cutover
  -> maintenance rendezvous cuts every old incoming root
  -> old versions retired by epoch and strong-reference quiescence
~~~

Workers hold no shared JIT lock during discovery, liveness, lowering,
register allocation or machine compilation. The reservation transaction is
short and occurs before backend work. A stale or losing candidate releases only
its exact reservations.

Instruction bytes are copied through the [versioned executable-content snapshot](memory-authority.md).
Worker lowering never reads live guest memory. A code
or mapping change after capture makes the candidate stale. Backend output
remains in a worker-owned nonexecutable staging buffer until it has a final
size and relocation set.

Publication has a fallible preparation phase followed by an allocation-free,
no-fail commit. Before acquiring an OpenToken, the publisher:

1. completes lowering into a worker-owned nonexecutable byte/relocation buffer
   and computes exact body, bridge, island and metadata sizes;
2. reserves the exact final RX body/island addresses, unit, dependency,
   dispatch, family, link, native-PC-table, payload, safety-shadow and
   retirement capacity; a reshape/replacement also prepares the successor
   family's dedicated tier-cutover [MaintenanceRecord](maintenance-records.md);
3. enables only the reserved noncallable backing pages through the RW alias,
   copies the bytes and applies relocation using the final RX addresses;
4. validates all backend offsets, state maps, patch shapes, relocated bytes,
   island reservations and helper-stack/spill high-water marks;
5. revalidates the complete executable observation, including mapping, content,
   protection and alias generations and an exact byte comparison;
6. constructs COW PageFaultTables and their same-capacity safety shadows from
   pinned current snapshots, retrying preparation if a table changes before
   final validation;
7. performs the complete platform new-code cache-coherence/protection protocol;
   and
8. leaves the RW alias inaccessible and constructs every immutable object the
   commit will store.

Steps 1..8 keep all state exclusively builder-owned. Any failure in those steps
calls the one `abort_unpublished_candidate` routine: it drops snapshot and
compilation UnitPins plus every PageFaultTable snapshot pin, destroys the
builder-owned unpublished current/shadow COW tables, and clears all temporary
immutable objects. Under only the JIT-state mutex it detaches only this
BuildToken's OwnerCell/family anchors and returns this candidate's reserved
unit/family/dispatch/dependency/link/native-PC/payload/safety/retirement slots
and bitmap bits without advancing their never-published semantic identities.
It then takes only the code-cache writer mutex, clears every reserved body's
exact L0 bits and every reserved island bit, decommits any newly empty
segment/metadata page and
reconciles all committed bytes and reservation credit. Each step validates the
full reservation ticket/generation; mismatch latches terminal rather than
freeing another build's resource. It creates no
MaintenanceRecord, result waiter or queue sequence. After step 8, publication
of a replacing candidate's Prepared record, described next, is the final
fallible action before OpenToken acquisition. Initial LCQ/HCQ publication has no
old-family cutover record and proceeds directly to OpenToken acquisition; its
Open/JIT revalidation failure uses the same `abort_unpublished_candidate`
routine.
Failure-injection after every preparation action requires all registry/body/
island allocation bitmaps, registry live counts and four ledger counters to be
byte/equality-identical to their pre-build snapshot, except for monotonic IDs
which were issued and are deliberately never reused.

The mandatory tier-cutover record is not the UnitLifecycle safety-unlink record
or a PatchRecord's performance-install record. Every HcqFamily owns one
precharged tier-cutover record; every UnitLifecycle owns one safety-unlink
record; every PatchRecord owns one performance-install record. As the final
action after successful step 8, a replacing publisher locks only the
maintenance queue and
acquire-revalidates that phase is the exact captured Open(E) and its current
RoundTicket is unsealed. Failure aborts before assigning a sequence or exposing
a record and then uses `abort_unpublished_candidate`. Success assigns a
checked sequence and that ticket, inserts its immutable
`CutoverToSuccessor` RootMutationRequest, and release-stores
`Prepared(g, ReplacingPublisher(BuildToken), sequence, ticket)`. It releases that lock before
OpenToken/JIT locking. Optional new static links remain on fallback if their
later performance record cannot be queued.

It then acquires an OpenToken whose epoch equals the compilation epoch, takes
the JIT-state mutex and revalidates the seed, every ReachabilityVersion,
owner/reservation generation, executable-observation generation, current
PageFaultTable snapshot and exact InstructionKey reservation. Still before the
no-fail tail, it checks that current global code epoch R has successor R+1 and
that every preassigned identity/counter value is current. A failed check
changes no registry/directory pointer or dispatch root. It releases the
JIT-state mutex and OpenToken. An initial publication then invokes
`abort_unpublished_candidate`. A replacing publication invokes the exact
[Awaiting-registration primitive](maintenance-records.md): it either adopts an already-published
matching counted AwaitingBuilder state without incrementing, or registers once
and CASes its exact Prepared record to
`CancelledBeforeCommitAwaitingBuilder(BuildToken, counted=true)`. Its builder
then performs the common staged cleanup, CASes that exact Awaiting state to
`CancelledBeforeCommit`, consumes its counted bit, detaches/acknowledges the
record and wakes the result. No AwaitingBuilder state is acknowledged or woken
before cleanup, and no stage nests those mutexes or owns an OpenToken while
touching a body/island allocation bitmap.

For a replacing publication, the final action before commit step 1 is the
single tag-only CAS defined above from the exact Prepared word to Pending. The
publisher preserves the payload and publication sequence, release-sets the
shared maintenance control hint, increments the record parking sequence and
wakes its waiters. A successful CAS is the proof that the mandatory cutover
record is reachable before any unit mutation. If the CAS instead observes the
coordinator's exact `CancelledBeforeCommitAwaitingBuilder(BuildToken)` state,
the coordinator has already registered the AwaitingBuilder obligation and the
publisher takes the builder-cleanup path below without performing step 1. Any
different word is precommit corruption and latches the terminal cause. Initial
publication has no cutover record and skips this CAS.

A successful commit performs exactly, while global code epoch remains R:

1. install the unit-registry slot as PublishReady;
2. install dependency entries, family ownership, source/target link records and
   every provisional backlink required by already-linked staging code;
3. release-exchange every affected immutable PageFaultTable and epoch-retire
   the old tables at R;
4. mark UnitLifecycle Published and install an HCQ family as CurrentStable or,
   for reshape, CurrentPendingCutover while marking predecessor families
   CutoverOld and their unit lifecycles Superseded;
5. release-store one complete DispatchPayload for each selected BlockKey and
   retire each replaced payload at R;
6. release-store LCQ payloads for reshape members omitted by the successor and
   retire their replaced payloads at R;
7. release-store global code epoch R+1, after every pointer exchange and before
   any reader can announce R+1 and load one of those pointers;
8. detach only the exact non-root logical reservations through precomputed
   atomic/index stores; this step never touches allocator free maps or ledger
   storage; and
9. release the JIT-state mutex and OpenToken, notifying a Closing waiter when
   the atomic token decrement reaches zero.

Every transition record required by later cutover exists in the queue and link
graph before the first DispatchPayload store. If another requester changes
Open to Closing before the publisher obtains its token, token acquisition fails
and an initial publication invokes `abort_unpublished_candidate`. A replacing
publisher invokes the identical Awaiting-registration and common builder-
cleanup protocol just defined; it may adopt the coordinator's matching counted
Awaiting state but can neither increment twice nor acknowledge it before all
builder-owned state is released. A
driver may observe either side of each exact generational cleanup but cannot
claim the cancelled record or clear a newer reservation. The publisher never
relies on a driver whose cutoff may already have been sealed. If it obtained
the token first and won Prepared-to-Pending, a driver may request Closing but
cannot seal its cutoff until commit step 9 releases that OpenToken. After step
9 the publisher enters the ordinary queue-serialized `drive_or_join` protocol:
under the maintenance-queue mutex it revalidates the exact record and ticket;
Pending requests or joins the round which owns that ticket, while Claimed or a
terminal result is joined or observed without another phase transition. It
unlocks before doing driver work. Until cutover is processed, predecessor roots
remain valid and new unresolved sources use permanent fallbacks.

Entering commit step 1 starts the no-fail/no-rollback tail; step 5 is only its
first executable-reachability point. Every step 1..9 is a preallocated,
infallible store or state transition. It performs no allocation, fallible
system call, cache operation, logging, validation, panic-capable code or new
lock acquisition. The tag-only Prepared-to-Pending CAS occurs before this tail,
not within it. Mutex unlock and atomic waiter notification are the only
synchronization epilogue; the token decrement, not successful wake delivery,
is the linearization point and waiters always recheck. An unexpected invariant
failure anywhere in that tail atomically calls
`latch_terminal(JitCommitInvariant, 0)`, completes every remaining structurally
required precomputed store when possible and releases the mutex/OpenToken. An
ownerless worker may then enter `drive_or_join_terminal()` directly. A
publisher attached to a cold executor first uses the fatal cold-exit protocol:
canonicalize its outcome, reach FaultSlot Idle, clear TLS/restore SignalAltStack,
stop per-vCPU reads and release VcpuUseToken then ArenaAdmissionToken from the
external system stack; only then may it drive/join. It never publishes Stopped directly or rolls back a partial
commit. If corruption prevents completion of that bounded tail, the
async-signal-safe fatal escape terminates the process.

Each DispatchPayload store is the execution linearization point for that
BlockKey. Several keys cannot change in one hardware-atomic operation, but a
mixed old/new view is valid only because both versions implement the same
revalidated instruction image and all predecessor roots remain callable until
Closed cutover. Invalidation drains publisher OpenTokens before taking its
index snapshot, so the complete commit is either included by that transition
or made no reachable change. A Prepared record cancelled by another thread
remains in its AwaitingBuilder state until its exact builder has released every
reservation/span/credit and published the final cancellation; only then is it
drained under the queue mutex and allowed to advance the contiguous
acknowledgement prefix. It cannot reacquire a root.

The explicitly requested debug replacement message is emitted once per
published HCQ unit, after the mutex is released. It contains the family/version
and seed, not one line per selected entry and no timing or counters.
