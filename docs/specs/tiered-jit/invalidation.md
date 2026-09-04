# Code and mapping invalidation

[Specification index](README.md) · [Implementation tasks](tasks/README.md) · [Open review items](review-status.md)

Related contracts: [Direct memory and fault authority](memory-authority.md); [Request execution, terminal transfer and cohort handoff](coordinator-execution.md); [Epoch reclamation and terminal teardown](epochs-and-shutdown.md).

## Code and mapping invalidation

Every gateway uses the [OpenToken-to-active-epoch ordering](native-abi.md#gateway-and-fast-mode); every LCQ/HCQ
publication holds a matching OpenToken through its commit. The page indexes
return every intersecting CodeUnit in Published, Superseded, Invalidating or
UnlinkPending whose exact snapshot may still own an executable root. Retired is
never callable because it is reachable only after Unlinked. A MappingChange or
executable-write record is applied in this order:

1. enqueue the complete typed request before changing guest-visible memory;
2. reserve its RoundTicket, change the exact tagged Open phase to Closing,
   assert the shared control request and drain all OpenTokens;
3. in ClosingPrepare, recompute the exact affected unit/family set from
   dependency indexes after that drain, acquire traversal pins, fill the
   precharged unavailable SafetyPayloadCells and native-PC-table shadows, and
   reserve every new ReachabilityVersion plus current R/R+1; if any check fails,
   mark this exact record FailedBeforeMutation, release only its plan/pins,
   leave its requested roots and memory unchanged, and continue processing the
   rest of the sealed cohort; only the common step-12 handoff may reopen;
4. in the no-fail ClosingFreeze, under the JIT-state mutex, revalidate that
   exact set, cancel its HCQ anchors, mark every affected unit Invalidating,
   register all incoming unlinks and release-exchange each affected dispatch
   pointer while global code epoch remains R; then release-store R+1 and release
   the mutex;
5. wait for every native executor to become execution-quiescent and
   compare-exchange the exact Closing tag to Closed;
6. restore affected static patches to their permanent fallbacks, clear every
   PIC root naming an affected source/target/bridge version, and complete the
   required code-cache and cross-core pipeline synchronization; a suspended
   retry naming an affected unit is atomically changed to canonical-only and
   loses its temporary executable root while its FaultSlot retains the UnitPin
   for owner-side canonicalization, and RSB guest predictions are cleared only when
   their execution key is itself invalidated;
7. only after synchronization remove the corresponding backlinks/roots and
   change each fully cut unit through UnlinkPending to Unlinked;
8. retire the replaced payloads and units at already-published epoch R; under
   the JIT-state mutex remove current dispatch, HCQ ownership and outgoing
   associations, release-exchange the prepared COW native-PC shadows and mark
   each unit DirectoryDetached; because admission remains closed and every
   active epoch is zero, those removed tables also use R;
9. release all driver traversal pins and subsystem mutexes; old PageFaultTables
   and dependency metadata retain only their own directory/explicit pins until
   grace, and a synchronous waiter retains only its result cell;
10. with no JIT-state or code-cache lock held, apply the exact mapping,
   protection, backing or write transition under the memory authority;
11. finish every safety record in the sealed cohort and leave later records
    pending without exposing their requested mutation; and
12. reopen with one new admission epoch through the queue-serialized handoff.

An OpenToken admitted before step 2 completes its no-fail publication before
the Closing freeze point and is visible to the step-3 dependency scan. A
publisher which did not obtain that token cannot change an index or payload.
An LCQ compiler, HCQ worker or linker with an older epoch fails later
revalidation and releases only exact generational handles; none can reopen the
process. A synchronous compile therefore cannot race old-address-space code
past the freeze point.

An executable-page write caught by host protection has not executed the guest
store. The writer records the exact prefault/commit-stage continuation and uses
its temporary UnitPin only while copying every reconstruction value and the
typed mapping request into canonical/preallocated storage. It then releases its
native-PC read state and UnitPin, clears its active epoch and only afterward
acquires the MappingRequestPermit or waits/drives the transition. The request
itself acquires its own current strong victim handles during preparation. The
write plan is performed exactly once after roots are cut and synchronized; the
writer never waits for its own epoch/pin and never retries the old native
instruction.

There is no generation test on every static link. The control rendezvous and
unlink ordering make stale direct links unreachable before their targets can be
reclaimed.
