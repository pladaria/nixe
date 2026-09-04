# Epoch reclamation and terminal teardown

[Specification index](README.md) · [Implementation tasks](tasks/README.md) · [Open review items](review-status.md)

Related contracts: [Code units, versions, registries and admission](units-and-registries.md); [Request execution, terminal transfer and cohort handoff](coordinator-execution.md); [Shutdown requests, result ownership and wait protocols](shutdown-and-waits.md).

## Epoch reclamation

The process owns a checked nonzero `global_code_epoch`. Every registered native
executor, including direct-interpreter stub execution, owns one lock-free
`active_code_epoch`; zero means execution-quiescent. Gateway/stub admission
acquires an OpenToken, acquire-loads the global epoch, release-stores it active
and releases the token before reading executable reachability. A canonical exit
first commits all guest/budget/FP state, stops dereferencing native metadata and
releases any fault retry pin, then release-stores zero and wakes a Closing
coordinator.

Every Open root/payload/directory commit and every ClosingFreeze uses one
identical replacement rule. `try_normal_epoch_successor(R)` succeeds exactly
when `1 <= R <= u64::MAX-2` and returns `R+1 <= u64::MAX-1`; when R is
u64::MAX-1 it latches IdentityExhausted before any root/pointer mutation.
u64::MAX is exclusively the terminal marker and can never be returned by this
primitive. Under the JIT-state mutex the writer reads R and calls that primitive
before mutation, release-exchanges every old pointer while the
published epoch is still R, and release-stores R+1 only after the final
exchange. Every replaced object receives retirement epoch R. A reader which
announced R may see old or new and therefore protects the old object; a reader
which announces R+1 acquire-observes every preceding exchange and can see only
the new set. A Closed directory exchange after all readers are zero may use the
R already advanced by its ClosingFreeze, because admission stays closed until
that exchange is complete. No code path increments first and exchanges later.
The execution grace condition is exactly:

```text
for every registered executor:
    active_code_epoch == 0 || active_code_epoch > R
```

Compiler jobs, queued/claimed link work, PageFaultTables and normal-stack fault
resolvers use explicit UnitPins instead of active epochs. An unclaimed
performance-link record holds only generational weak handles and gains target
pins only when claimed/revalidated. Current native-PC records are COW-detached
before DirectoryDetached; removed tables receive an epoch no earlier than their
pointer exchange and retain directory UnitPins until that grace period ends.

Final reclamation order is exactly:

1. cut dispatch, static-link, bridge, PIC and suspended-native-retry roots;
2. mark the unit Unlinked and, in Closed under the memory authority, invalidate/
   release its live executable-observation leases while retaining their
   immutable records; after the last lease on a physical page, recompute every
   reverse alias's protection from the remaining dirty-tracking/permission
   reasons, make it writable only when all such reasons permit, and advance the
   protection and observation generations before reopen;
3. assign the unit retirement epoch and mark it Retired;
4. COW-remove current native-PC records and mark it DirectoryDetached;
5. wait for unit and removed-table epoch grace;
6. wait for every compiler, linker, queue, directory and fault UnitPin to be
   released;
7. detach dependency/link records and dispatch/family/unit slots and free all
   coupled immutable metadata; and
8. clear the executable span's exact allocation bits or decommit its wholly
   free segment.

Steps 5 and 6 are checked again after removing retired PageFaultTables because
a resolver may clone a pin before releasing its active epoch. Addresses and
generational slots are reused only after step 8. Epoch exhaustion first disables
HCQ and then rejects any LCQ publication/lifecycle operation which requires a
new value with a precise checked-capacity failure; no counter wraps.

## Terminal driver teardown

The terminal driver uses this exact order:

1. require all four terminal gates closed by `latch_terminal`, publish Stopping
   through the queue-serialized phase-word protocol, and make every gateway/public JIT API which observes the
   terminal phase return its sticky result using only TerminalControl, without
   dereferencing a vCPU, record or reclaimable arena;
2. close HCQ admission and drain the low counts of `open_token_gate` and
   `prequeue_producer_gate` to zero before closing the producer set.
   Under only the pending-queue mutex remove every still-Queued handle; in
   separate ActiveBuild-cell stages cancel Initializing, AdmissionReserved and
   Queued attempts and publish the exact `cancel_build_token` for every Building
   attempt. In a separate bounded JIT-state-mutex scan of DispatchSlots, change
   every exact `LcqBuildCell::Building(token, reachability, epoch)` to
   `Stale(token, reachability, Shutdown)` and collect its parking word; after
   unlocking, wake every collected LCQ owner/waiter. The exact builder observes
   that tag, drops candidate leases/pins/reservations and releases its vCPU use;
   waiters release their waiter references and take only the sticky terminal
   exit. Change each non-driver Running WorkerControlCell to StopRequested
   and wake it. A worker terminal driver must already have cleaned its own
   BuildToken, published TerminalDriver and consumed its live-count bit; an
   external driver has no worker cell;
3. under only the maintenance-queue mutex, convert every replacing-publisher
   Prepared record to counted `CancelledByStopAwaitingBuilder`, every eligible
   InstallLink state to counted `AwaitingLinkCleanup(CancelledByStop)`, and every
   other non-Applying record to its exact CancelledByStop result. A Prepared
   VcpuDeactivate record is not a builder and goes directly to CancelledByStop.
   Let an Applying no-fail tail reach Applied or
   TerminatedAfterMutation. After the complete bounded record scan, release-
   store `awaiting_registration_closed = true` while still holding the mutex,
   then unlock and wake every worker/result waiter. With no other mutex held,
   take `reclaim_control_mutex`, change an exact Running ReclaimControlCell to
   CancelledByStop with the sticky cause, clear its running-owner bit, unlock
   and broadcast its parking word. The completed OpenToken and prequeue-
   producer drains, plus the terminal phase, prove that no producer can publish
   a new Prepared record or begin another reclaim request after this closure
   point;
4. each non-driver worker finishes its own exact BuildToken/AwaitingBuilder
   cleanup and exits by the WorkerControlCell protocol; it never waits for
   ShutdownRecord. Without a subsystem mutex, join every worker except a worker
   which is this terminal driver. Any `JoinHandle::join` error, missing handle,
   incoherent WorkerControlCell, or returned worker which failed its exact
   counted-live 1-to-0 transition immediately invokes raw
   `SYS_exit_group(70)`, even before any root cut. After all permitted joins,
   acquire-load `worker_live_count` exactly once and require zero; the joins
   synchronize with every possible non-driver decrement, while a worker driver
   consumed its own bit before exclusion. A nonzero value therefore has no
   possible remaining writer and is raw-fatal, not a value to wait on.
   Collect and execute every LinkCleanupTicket outside queue/JIT/cache locks,
   publish their exact final results, and wait for both Awaiting counters to
   reach zero. Acquire-scan every MaintenanceRecord and PatchRecord and require
   no Awaiting state or cleanup ticket. Require the pending queue empty, every
   worker's `current_build_token == 0` and zero queue/worker attempt pins.
   ActiveBuild attach/init readers may still belong to a pre-Stopping cold vCPU;
   this early pass only sets their exact cancel token and `recycle_pending` and
   neither requires them to be zero nor recycles their storage. Now, and not
   earlier, copy every terminal Maintenance/Reclaim result in the
   TerminalDriverGuard's `normal_obligations` bundle to caller-owned scalar
   storage, clear any still-owned Reclaim running-owner bit, decrement every
   requester reference, drop the corresponding NormalResultTokens and assert
   the bundle empty. Wake all other terminal normal-result holders and wait for
   the low count of `normal_result_gate == CLOSED|0`. Acquire-scan
   ReclaimControlCell and require Free or terminal, zero requester references
   and no running-owner bit;
5. make every vCPU slot TerminalDeactivating and every unit root-inadmissible,
   so no new VcpuUseToken, resume token or TLS publication can succeed;
   exchange each dispatch admission to its precharged TerminalUnavailable
   payload while preserving the existing ReachabilityVersion and allocating no
   identity/epoch successor, convert/cut already-
   suspended retry roots to typed terminal outcomes, assert control,
   acquire-wait for `arena_admission_gate == CLOSED|0`, require the complete
   arena-admission bitmap zero, and wait without a subsystem mutex for every
   closed use-gate low count and executor epoch to be zero and every redirected landing
   to reach an Idle FaultSlot. Those waits use only their already-specified
   futex words. Because the owning host thread locally asserts null TLS and
   publishes `tls_clear_generation` before it may release
   VcpuUseToken/ArenaAdmissionToken, the driver then acquire-asserts, without a
   separate wait, that every slot's latest `tls_publication_generation` equals
   its ordinary `tls_clear_generation`. It never reads another thread's ELF
   TLS pointer.
   Static bytes and private
   PIC ways remain untouched during this wait, and the Nixe signal handler
   remains enabled so an already-executing guest access can still land safely.
   The arena gate's same-atomic close/acquire ordering proves that no new bit or
   arena reader can appear after the zero observation;
6. now that native execution is quiescent and can never reopen, first perform a
   definitive second scan of ActiveBuild, PatchRecord, unit/family ownership and
   MaintenanceRecord storage; this catches any cold-executor action which began
   before the step-5 drain. Under each ActiveBuild-cell mutex require no
   Initializing, AdmissionReserved, Queued or Building state, zero initialization/
   owner/queue/worker pins, zero `attach_reader_count`, false
   `recycle_pending` after processing it, and no live family/OwnerCell anchor
   reservation. Require every FamilyPin count zero before releasing any family
   slot. Leave terminal Published/Rejected/Cancelled cells nonreusable with
   zero pins/references in this pass; Stopping never advances an ActiveBuild
   slot generation merely to unmap it. Then
   require every LcqBuildCell to be Idle, Published, Failed or shutdown-Stale
   with no owner, and every DispatchSlot waiter/queue reference count to be zero.
   No Building state is permitted. Then
   clear all PIC,
   bridge and remaining retry roots/backlinks logically and mark their
   lifecycles terminal-unlinked without rewriting executable bytes. A patch
   whose earlier synchronization failed is marked terminal-abandoned in its
   result, not Applied, and shutdown never retries the same fallible cache/
   pipeline operation; the still-mapped bytes remain unreachable until the
   executable reservation is unmapped. No code byte or private PIC root is
   changed before quiescence. Under the one memory-transaction guard and in
   ascending PhysicalPageId order, invalidate and release every remaining
   ExecutableObservationLease, recompute every reverse alias protection from all
   remaining tracking/permission reasons, and advance the protection and
   observation generations before disabling the handler or unmapping metadata.
   A protection or cache-coherence system-call failure in this post-cut terminal
   stage invokes raw `SYS_exit_group(NIXE_RAW_EXIT_INTERNAL)` with
   `NIXE_RAW_EXIT_INTERNAL = 70`; libc `_exit` is forbidden and the path never
   publishes Applied;
7. require the publication/clear equality for every snapshotted vCPU after
   step 5, release-disable this TerminalControl's `handler_enabled`,
   release-publish the pre-reserved
   terminal `handler_generation = Gstop = 2`, and have the TerminalDriverGuard
   release-store Gstop into every registered slot's precharged
   `tls_detached_generation`; this includes its own descriptor, which had to be
   locally null before it could become terminal driver. This field is the
   terminal driver's certification after the step-5 generation-equality proof,
   not a
   host-thread acknowledgement; acquire-scan/assert every slot equals Gstop and
   then wait for this TerminalControl's
   `inflight_signal_handlers == 0`, then acquire-observe
   HostSignalControl's `global_inflight_handlers == 0` once after all TLS
   acknowledgements before reclaiming a vCPU record or TerminalControl. That
   final global grace covers a handler which loaded the old TLS descriptor but
   had not yet incremented the per-process count; handlers entering after the
   acknowledgements can find no generation-valid process registration and
   cannot touch this process. A normal
   pre-Stopping `tls_clear_generation` value never satisfies this terminal
   acknowledgement. The
   TerminalDriverGuard then consumes the process's HostSignal InstallLease,
   takes the HostSignalControl install mutex and CASes this JIT's
   `signal_registered` true-to-false exactly once. If refcount was greater than
   one it checked-decrements it and leaves accepting true; on one-to-zero it
   release-clears accepting before publishing zero. A repeated shutdown/Drop
   observes false and cannot decrement again. This refcount counts JIT process
   registrations, never Arc/process-handle references. The global
   handler/dispositions remain installed while any JIT process exists; a late
   handler sees disabled or cleared TLS and chains through permanent
   HostSignalControl without arena access. A WaitingFaultIdle owner performs
   its Idle-to-Pending protocol before this acknowledgement in a nonterminal
   phase; in Stopping it observes CancelledByStop, does not requeue and then
   acknowledges;
8. perform the following indivisible ownership barrier:
   **8a, arena-aware:** release-exchange every current native-PC directory page
   to null, assign current global epoch R to each newly removed table without
   requesting R+1, wait that grace plus every older retired table's existing
   grace and release their directory-owned UnitPins. Wait for every remaining
   explicit UnitPin count in the complete unit registry (including compiler,
   linker, queue, directory and fault owners), and require every FaultSlot Idle,
   every FaultTransitionRecord Empty, every TLS acknowledgement, zero
   `awaiting_builder_count`, zero `awaiting_link_cleanup_count` and zero low
   count of `prequeue_producer_gate`. Under only the slot-registration mutex,
   change every configured TerminalDeactivating vCPU to TerminalInactive,
   decrementing `active_vcpus` exactly when its payload says
   `was_counted_active=true`; require active_vcpus zero and every configured slot
   TerminalInactive. For each address space in slot order and with no other
   mutex held, take its MappingRequestPermit mutex: Free becomes
   Terminal(cause), while Owned is corruption because producers/results already
   drained. Then under only the address-space registry mutex change Active to
   Destroying and Destroying to TerminalDestroyed, and require every
   MappingChange record terminal with zero result references and every address-
   space pin zero; Free remains Free. Under only the code-cache mutex, unmap
   every per-segment RW alias in ascending segment index, close every segment
   memfd exactly once and require its owner state closed; the RX reservation
   remains mapped but unreachable. Copy the construction-owned RX and metadata
   bases/lengths into Empty TerminalUnmapPlanV1 with exact origin/index and
   release-store Ready. Release all arena mutexes and consume every arena-aware
   handle, reference, iterator and destructor, then release-store
   `process_owner_inert=1` and convert TerminalDriverGuard into
   TerminalUnmapGuard; the conversion type-checks only if no arena lifetime is
   retained.

   **8b, TerminalControl-only:** require the copied plan still Ready and use raw
   `munmap` first on its exact RX range and then on its exact metadata range.
   Any wrong return/raw errno exits 70. After both succeed, release-store plan
   Consumed. This tail may dereference only TerminalControl and the Copy plan;
   neither worker nor external origin runs a destructor capable of inspecting
   either unmapped range;
9. as sole remaining ledger writer, use the odd/even ledger protocol to set
   `jit_charged_bytes` and all three subcounters to zero, release-store the
   reserved `u64::MAX` code-epoch marker, publish Stopped with the final reserved
   phase sequence, acquire-require Claimed edge completion at parking sequence
   three, arm ShutdownWakeGuard for four, CAS the exact cause-preserving
   Claimed word to Applied, and let the guard release-store four and wake every
   waiter before continuing. A failed CAS or sequence mismatch raw-exits 70; no
   plain store or wake substitutes for this edge. A
   worker terminal driver then release-stores its persistent WorkerControlCell
   ExitedNoArena and returns from the thread entrypoint; an external shutdown
   caller which owns the process joins that final handle before returning.

Shutdown uses no successor code epoch: after admission is terminal and all
readers/pins are zero there can be no future observer, so terminal quiescence
itself proves reclamation. The per-JIT TerminalControl page remains until its
final handle destruction, while normalized prior signal dispositions remain in
permanent HostSignalControl until OS process exit; exhaustion of the normal
epoch range cannot block the two reserved terminal phase transitions.
Before ShutdownRecord is Claimed, an unexpected error can still latch its typed
sticky cause and enter this order. After that claim, and also for any
`JoinHandle::join` error even before the claim/root cut, any protection/cache/
unmap syscall failure, impossible tag, counter underflow/overflow or invariant
failure which prevents completing the order calls raw
`SYS_exit_group(NIXE_RAW_EXIT_INTERNAL)` immediately. It never
returns through libc, leaves ShutdownRecord indefinitely Claimed or publishes
Applied after incomplete teardown; an earlier ExplicitShutdown cause cannot
mask this process-fatal escape.
