# Request execution, terminal transfer and cohort handoff

[Specification index](README.md) · [Implementation tasks](tasks/README.md) · [Open review items](review-status.md)

Related contracts: [Coordinator phases, terminal control and signal installation](coordinator.md); [Maintenance records, cleanup and mapping requests](maintenance-records.md); [Cohort workspace, arbitration and mutation plans](cohort.md); [Shutdown requests, result ownership and wait protocols](shutdown-and-waits.md).

## Normal maintenance requests

A nonshutdown requester performs exactly:

1. acquire the [NormalResultToken](shutdown-and-waits.md), then claim the exact Free owner record
   or serialized permit generation;
2. fill its typed record with the RootMutationRequest and stable
   result-cell reference without holding the maintenance-queue mutex. A safety
   request acquires the initiating strong handles required to preserve its
   victim identities; InstallLink retains only generational weak handles and
   acquires its UnitPins if ClosingPrepare claims it;
3. release every JIT-state, code-cache and memory-authority lock;
4. lock the queue, checked-increment `request_sequence`, assign its current or
   next RoundTicket and insert the record once as Pending. Safety/cutover work
   sets its reason hint, release-sets shared control and immediately follows
   the close rule below. InstallLink instead applies the sole batching rule
   in [static linking](linking.md#static-links) and sets control only when that rule says the batch is due. While still
   holding that mutex, when this insertion is required to close and the exact
   validated phase is `Open(E)`, prepare the next phase-payload slot, arm an
   `OpenCloseGuard` whose Drop raw-exits until the phase publication is
   complete, close `open_token_gate`, and CAS the complete phase word to
   `Closing(round, E, ticket)` only while its own record still has that Pending
   sequence. Any impossible CAS failure while the queue-serialized phase still
   names that Open is fatal; if shutdown intent is visible, the gate remains
   closed and the caller follows terminal transfer. The successful CAS creates
   NormalDriverUnwindGuard and only then disarms OpenCloseGuard. Closing/Closed joins the named ticket and
   Stopping/Stopped cancels it by the terminal rule. Then unlock;
5. after unlocking, every asynchronous requester, winner or loser, drops its
   result-cell reference and NormalResultToken because the queue-owned
   reference now preserves the request. The CAS winner retains only its
   DriverGuard and drives the round; and
6. if synchronous completion is required, first become execution-quiescent and
   release every OpenToken/subsystem lock, then wait for that exact record
   generation and sequence.

The requester which wins Open-to-Closing drives the round; another requester
joins it. A faulting/self-modifying vCPU first copies its complete canonical
request, releases directory read state and its UnitPin, clears its active epoch
and only then acquires a MappingRequestPermit or enqueues/waits/drives; it never
makes the transition wait for its own execution root. Checked sequence
exhaustion rejects a new
safety operation before guest-visible state changes and simply drops a
performance-only installation back to Fallback.

The successful Open-to-Closing CAS constructs an armed
`NormalDriverUnwindGuard` in the same non-unwinding function which creates the
normal DriverGuard. It remains armed until exactly one of (a) the matching
Closed-to-Open handoff consumes the DriverGuard or (b) the matching terminal
transfer consumes it into TerminalDriverGuard. Its Drop, including mutex-poison
or panic unwinding at any point in ClosingPrepare, ClosingFreeze, Closed or
handoff, invokes raw `SYS_exit_group(70)` and never returns; no caller may
abandon a Closing/Closed phase and there is no token takeover edge. All normal
driver entrypoints have an outer unwind boundary, but that boundary can report
only panics which occurred before Open-to-Closing; an armed-guard unwind has
already terminated the process. [Task 8](tasks/08-lifecycle-stress.md) injects one panic before ClosingFreeze
and one immediately after the first Applying/root store and verifies the raw
outcome rather than a stranded phase or reopen.

Shutdown needs neither a sequence nor a reopen epoch. Terminal entry is split
into two APIs with nonoverlapping lock requirements.
## Terminal intent and latch

`latch_terminal(cause, detail)` is
lock-free. It first validates the proposed closed cause/detail, release-sets
the wakeup mirror/control request (a spurious hint is permitted), then arms a
`TerminalLatchGuard` whose Drop raw-exits 70 until the complete suffix has
finished. In that non-unwinding suffix it closes, in this order,
`normal_result_gate`, `prequeue_producer_gate`, `open_token_gate` and
`arena_admission_gate`. It then arms ShutdownWakeGuard for value 2 and
AcqRel-CASes shutdown_word from zero to the fully packed Requested(cause,
detail). A winner **again** idempotently closes and wakes `open_token_gate`
after that CAS and only then lets ShutdownWakeGuard release-publish sequence two
and wake; this post-CAS close is part of the Requested edge-completion contract.
A CAS loser disarms its value-two guard, acquire-observes and validates the
already-sticky word, again idempotently closes/wakes `open_token_gate`, disarms
TerminalLatchGuard and immediately returns `LatchedPending(first_cause)`; it
does not futex-wait or claim that edge completion is visible. Concurrent callers therefore race only the
word CAS, but both pre-CAS and post-observation sides close the gate. The winner
disarms TerminalLatchGuard only after the guarded parking publication/wake.
This pairs with handoff's final recheck: even if a latch pauses between its
first close and word CAS while handoff reopens, its post-CAS close occurs before
any Requested-to-Claimed owner can proceed. Acquisition failure on one of the
three terminal-only gates (`normal_result_gate`, `prequeue_producer_gate` or
`arena_admission_gate`) while shutdown_word is temporarily Idle dereferences no
arena and waits/reloads TerminalControl until Requested; the non-unwinding tail
guarantees that interval terminates or the process raw-exits. `open_token_gate`
does not use that rule: CLOSED plus Requested is terminal, CLOSED plus
Closing/Closed joins or retries the normal round, and CLOSED plus Open/Idle is
the finite close/publication or publication/reopen window and returns the typed
TransientClosing result. Per-vCPU use gates are governed solely by their slot
state. `latch_terminal` may
be called from a no-fail tail while holding an OpenToken or subsystem mutex and
never waits, takes the queue mutex or changes phase. After completing every
structurally required store, that caller releases all OpenTokens, active epochs,
UnitPins and subsystem mutexes and invokes `drive_or_join_terminal()`.
A normal shutdown caller invokes the same two APIs in that order.

## Terminal driver acquisition and join

`drive_or_join_terminal(optional_current_DriverToken,
normal_obligations)` requires no OpenToken,
ArenaAdmissionToken, active epoch, UnitPin, VcpuUseToken, SuspendedResumeUseToken,
MappingRequestPermit, directory read-side state, published TLS descriptor or
subsystem mutex. `normal_obligations` is the closed caller-local bundle
`{ maintenance:None|One(MaintenanceRecordHandleV1, requester_ref),
reclaim:None|Requester(ReclaimRequestId, generation,
requester_ref)|RunningOwner(the same fields), normal_result_tokens:u8 }`;
`normal_result_tokens` is exactly the number of present references (0..=2).
Nesting permits at most the Reclaim owner's one current PressureRecord plus its
one ReclaimControl reference; any other combination is corruption. Before
calling it, an executor canonicalizes any owned continuation, reaches the
required FaultSlot state, release-clears TLS, completes its ordinary signal
grace, and stops every per-vCPU read. This is not the later terminal-generation
acknowledgement.

The function first takes the maintenance-queue mutex and classifies the exact
phase and DriverToken. For Open, a caller may compete for terminal ownership.
For Closing or Closed owned by a different DriverToken, it must unlock and join;
it must not CAS ShutdownRecord and cannot steal the normal driver. For a phase
owned by its own normal DriverGuard, it never waits on itself and reaches one of
these exhaustive transfer-ready boundaries: (a) in Closing before the batch
Claimed-to-Applying commit, it cancels the entire accepted batch, releases its
pins/shadows/reservations in the specified nonnested stages, publishes the
cancelled results, and clears all cohort cells and lengths; (b) after that batch
commit, it completes the entire precomputed no-fail freeze and R+1 store,
releases all subsystem mutexes, waits the exact active-code-epoch and signal-
landing grace, CASes its exact Closing phase to Closed, and completes every
Applying mutation/result; or (c) in Closed, it completes the same remaining
Applying mutation/result tail. InstallLink always uses its cleanup ticket. The
boundary requires zero plan-owned pins, clear CohortObjectPlanCells, zero
`touched_len`/`cleanup_len`, and no held subsystem mutex. Case (a) retains the
exact Closing phase as its transfer source; cases (b)/(c) use Closed. No other
transfer boundary exists.

If the queue-locked classifier sees Requested before
`shutdown_parking_sequence` acquire-reaches two, it snapshots that sequence,
unlocks, futex-waits/rechecks edge completion, relocks and restarts the complete
phase/DriverToken classification. It never waits while holding the queue mutex.
Thus the sequence below always begins after the latch winner's post-CAS
`open_token_gate` close.

At a transfer-ready boundary the caller holds the queue mutex and executes this
single non-unwinding sequence; none of its steps are described as simultaneous:

1. require the exact ShutdownRecord state Requested and prepare the complete
   `Stopping(cause, shutdown_DriverToken)` phase payload in the inactive slot;
2. arm `TerminalClaimTransferGuard`, whose armed Drop raw-exits 70;
3. require the Requested edge complete, arm ShutdownWakeGuard for value three,
   and AcqRel-CAS ShutdownRecord Requested-to-Claimed. This CAS is the sole terminal
   winner linearization point. On failure, acquire-observe Claimed/Applied,
   disarm both temporary guards without consuming an optional NormalDriverGuard,
   unlock, wait for the observed edge's completion and follow the loser path;
4. as the winner, CAS the exact previously classified Open, transfer-ready
   Closing, or transfer-ready Closed phase to Stopping using the prepared
   payload. Because the queue mutex excludes every legal phase writer, failure
   or a different source payload is corruption and raw-exits 70; and
5. complete ShutdownWakeGuard's release-store of sequence three and wake,
   consume the optional NormalDriverGuard and move its DriverToken plus the
   entire still-live `normal_obligations` bundle into
   `TerminalDriverGuard(shutdown_DriverToken, normal_obligations)`, then disarm
   TerminalClaimTransferGuard.

Between steps 3 and 5, other callers can only observe Claimed and wait; none can
become a second winner. `Stopping+Requested` and
`Open|Closing|Closed + Claimed` are legal only inside this guarded finite suffix;
outside it either combination is corruption. The winning caller does not settle
its normal references before the claim: [terminal steps 2--4](epochs-and-shutdown.md#terminal-driver-teardown) are the
authority which makes their records terminal.

A caller which loses the claim must run `settle_normal_obligations` before it
waits for the winner's ShutdownRecord. For its MaintenanceRecord it waits only
for that exact terminal result, copies it, decrements its requester reference
and drops the associated NormalResultToken. For ReclaimControl, a RunningOwner
may perform the exact non-Idle-shutdown cancellation under the reclaim mutex;
otherwise it waits only for Applied, Failed or CancelledByStop, copies the
ReclaimAck/error, conditionally clears only a still-owned owner bit, decrements
its requester reference and drops its associated token. Winner steps 2--4 are
obliged to make these record waits progress before waiting for the global token
count. The loser asserts an empty bundle/token count before any ShutdownRecord
wait. Thus only TerminalDriverGuard may temporarily retain normal obligations;
no loser waits for Applied while owning one. Every terminal stage validates
that guard. It enters
Stopping instead of reopening. From the successful Requested-to-Claimed CAS
through publication of Applied, every external or worker driver runs beneath a
non-unwinding terminal guard. There is no guard-takeover edge: any unwind,
poisoned coordination primitive, guard/phase mismatch or inability to continue
at any point in that interval invokes raw `SYS_exit_group(70)`, regardless of
whether the first root cut has occurred.
The only join exception is an HCQ worker which does not own the terminal
DriverToken: after terminalizing and cleaning its own current BuildToken by the
WorkerControlCellV1 protocol, it publishes ExitedNoArena and returns from the
thread entrypoint instead of waiting for ShutdownRecord. The process owner
retains its JoinHandle, so this prevents a driver-to-worker join from cycling
with a worker-to-driver wait without losing a required join. A worker which
owns the terminal DriverToken follows the TerminalDriver branch and never joins
itself. Neither path overwrites Closing/Closed or another driver, and every
later transition treats Stopping/Stopped as sticky. The final terminal order
in [terminal teardown](epochs-and-shutdown.md#terminal-driver-teardown) is the sole authority for producer closure, Awaiting conversion, cleanup,
worker joins and result-token drain.

Shutdown allocates no all-roots snapshot: after admission/workers
are terminal it scans the fixed unit (including all bridge units), family,
patch, vCPU and address-space registries, then the dynamic-bridge weak index and
bridge-permit bitmap, in ascending slot order, using each object's already-precharged
safety cell/shadow plus the one coordinator FaultSlot array. No registry can
gain an entry during those finite passes. A driver which entered as a
synchronous normal requester retains its result-cell reference and
NormalResultToken until [terminal steps 2--4](epochs-and-shutdown.md#terminal-driver-teardown) have terminalized and cleaned that
exact record; consuming either earlier can deadlock on work which only those
steps complete. At the end of step 4 it copies the exact terminal result to
caller-owned stack storage, drops the reference/token, and only then drains all
other NormalResultTokens. A driver's own Deferred record is corruption because
Deferred is asynchronous InstallLink state; WaitingFaultIdle is changed to
CancelledByStop by the step-3 record scan. Immediately before the step-4 count
wait, the driver must own no NormalResultToken, normal result-cell reference,
OpenToken, ArenaAdmissionToken, UnitPin, VcpuUseToken,
SuspendedResumeUseToken, MappingRequestPermit, active epoch or published TLS
descriptor. The driver then completes the remaining shutdown order,
release-stores Stopped and changes ShutdownRecord to Applied. A shutdown caller
returns only after observing both exact states.

## Closing and Closed transition

The transition driver performs exactly:

1. keep the shared control request asserted;
2. acquire-wait until the OpenToken count is zero, establishing the Closing
   freeze point;
3. under only the queue mutex, branch on every still-Prepared record's exact
   PreparedOwner. Change ReplacingPublisher to counted
   CancelledBeforeCommitAwaitingBuilder with its exact BuildToken (leaving all
   builder-owned cleanup and final acknowledgement to that builder), and change
   VcpuDeactivator directly to CancelledBeforeCommit because Active-to-
   Deactivating did not occur and it owns no builder resources. Then seal
   `round_cutoff = request_sequence`, and claim the finite set of Pending
   records at or below that cutoff. In the queue's existing increasing-sequence
   order, keep the first 4096 claimed InstallLink records and immediately change
   every remaining InstallLink from Claimed to
   `Deferred(old_sequence, reserved_sequence, reserved_ticket)`. For each such
   record it first reserves one unique checked request sequence on the already-
   created next RoundTicket. If that reservation cannot be made, the record owns
   no resource, so it goes directly to
   StaleNoOp(PublicationSequenceExhausted) on its old sequence and restores its
   patch to the already-present fallback; it must not create an Awaiting cleanup
   ticket. This all occurs before any deferred record acquires a UnitPin, shadow
   or plan storage. Consequently cleanup_ordinals can contain only the selected
   at-most 4096 prepared InstallLinks;
4. perform `ClosingPrepare`: with publishers frozen but before root mutation,
   make a two-pass exact dependency/root snapshot, reserve all
   ReachabilityVersions and R/R+1, fill each slot's SafetyPayloadCell and each
   affected PageFaultTable's precharged same-capacity shadow, acquire traversal
   pins and build every immutable RootMutationPlan outside subsystem mutexes.
   Every plan that can make a unit noncallable contains the exact affected
   UnitHandle/CodeVersion predicate used against the one coordinator fault
   snapshot; it contains no per-plan FaultSlot array. The coordinator array is
   deliberately not filled yet, so a suspended root published during
   ClosingPrepare cannot be missed. Only safety records and the selected at-most
   4096 InstallLinks are prepared;
   brief JIT-mutex passes count and then revalidate/fill already-sized storage
   and never allocate;
5. if preparation of a non-InstallLink record fails, mark only that record
   FailedBeforeMutation and leave its roots/memory unchanged, but only while
   that variant remains wholly abortable: InvalidateToUnavailable,
   PressureEvict, UnlinkOnly and MappingChange before any memory effect qualify.
   A VcpuDeactivate continuation after Active-to-Deactivating and a
   CutoverToSuccessor after successor/predecessor publication have all plan,
   shadow, sequence and capacity resources precharged by their earlier no-fail
   commit; any unexpected inability to prepare them latches terminal and follows
   their mandatory cleanup, never FailedBeforeMutation or rollback. If preparation of
   InstallLink fails after it owns InstallPending or any claimed pin, span,
   shadow or credit, construct its already-precharged LinkCleanupTicket,
   register AwaitingLinkCleanup exactly once and publish
   `AwaitingLinkCleanup(ticket, StaleNoOp(PreparationFailed))`; only the common
   ticket routine may return its patch/bridge resources and final Fallback/Dead
   state. After every remaining plan is prepared, execute closure-arbitration
   [steps 1--7](cohort.md#closure-arbitration-and-commit), including the allocator freeze and collective
   Claimed-to-Applying commit. Then execute mutation stage 8(a): cancel exact
   HCQ anchors, mark affected roots inadmissible, exchange final dispatch
   payloads to unavailable/retained LCQ while global code epoch is R, and
   release-store R+1 last before unlocking. CutoverToSuccessor only revalidates
   its already-current successor publication and performs cleanup/stabilization;
   it never repeats a successor dispatch store;
6. wait without a subsystem mutex until every native executor has
   `active_code_epoch == 0` and every signal landing has left its async handler;
7. compare-exchange the exact `Closing(round, E, ticket)` to
   `Closed(round, E, ticket)`; if Stopping/Stopped won, enter terminal cleanup
   and never publish Closed/Open;
8. execute [mutation stages 8(b)--8(g)](cohort.md#closure-arbitration-and-commit). In particular, acquire-scan every
   registered FaultSlot exactly once in ascending configured-vCPU order into
   `cohort_fault_snapshot`; because stage 8(a) made affected units
   root-inadmissible and step 6 observed every epoch zero, this closes the sole
   suspended-publication race. The aggregate owning cell, not plan order,
   CASes a matching SuspendedTransition to CanonicalOnly and wakes it. Apply
   static patch/PIC, backlink, one-per-page directory COW, memory-authority and
   lifecycle work in the stated nonnested order. The FaultSlot retains its
   UnitPin for its owner, and removed tables/units use retirement epoch R;
9. release traversal pins after DirectoryDetached and execute closure step 9's
   single result/cleanup/workspace/freeze finalization. Ordinary invalidation,
   mapping, cutover, unlink and deactivation reach Applied/StaleNoOp only after
   their requested aggregate is complete; no record waits in Closed for an
   epoch or requester/fault UnitPin. PressureEvict becomes
   `Applied(PressureRootsCut)` after its
   exact planned roots are cut, even when a snapshotted FaultSlot UnitPin delays
   a projected decommit. Its caller waits for grace/pins and rechecks the ledger
   only after the common Open handoff, then requests a fresh plan if needed.
   Remaining reclamation is asynchronous. A post-commit failure follows the
   [one-cause collective terminal tail](cohort.md#closure-arbitration-and-commit), never a recoverable per-plan
   Failed result. cleanup_ordinals is consumed exactly once; no repeated cutoff
   scan is permitted. A later-ticket InstallLink is not consumed by this round;
   and
10. execute the no-lost-request handoff below.

Every live PageFaultTable owns charged shadow storage large enough for its
maximum current record count. Publishing a larger table reserves both new
current and new shadow capacity before reachability. Thus ClosingPrepare does
not rely on allocation under hard pressure; after executor quiescence the old
table becomes the next shadow. The same alternating rule applies to
SafetyPayloadCell. Counter exhaustion is discovered in step 4, before
ClosingFreeze. A cohort requiring no retirement may omit R/R+1; otherwise it
advances global epoch exactly once.

At most 4096 performance-only link installations are attempted per Closed
cohort. Under the queue mutex, each unprocessed install first becomes
`Deferred(old_sequence, reserved_sequence, reserved_ticket)`; that state closes
its old-ticket participation. Its unique next sequence/ticket was reserved in
transition step 3, before the cohort's LinkCleanup drain, so handoff performs no
fallible arithmetic. Reservation exhaustion moves that performance request
through AwaitingLinkCleanup(StaleNoOp) to Fallback during the same cohort. A
deferred record is not a root and
retains no target UnitPin. Safety
work is never deferred once included in a cohort. A safety request inserted
after the cutoff cannot expose its mapping/code mutation and remains Pending on
the next ticket. A fault always has its preallocated per-vCPU record.

## Closed handoff

For handoff, the driver locks the maintenance queue and, while holding it:

1. advances `acknowledged_sequence` only across a contiguous prefix of terminal
   results or the old-sequence participation of Deferred/WaitingFaultIdle
   records; a synchronous
   result stays generation/sequence-addressable until its waiter consumes it,
   and that waiter holds only the result-cell reference, no victim UnitPin;
2. changes each exact Deferred record to Pending using only its already-
   reserved sequence/ticket. This is an infallible tagged-payload publication;
   no Awaiting state or cleanup ticket can be created during handoff. It does not
   reassign or count WaitingFaultIdle as next-ticket work; only the later owner-
   side [Idle protocol](maintenance-records.md) can do so. It records whether any actual next-ticket
   Pending record exists;
3. if ShutdownRecord is Requested, retain the DriverGuard and keep Closed. If
   its completion sequence is still below two, unlock the queue, wait/recheck
   the Requested edge, relock and restart handoff from step 1; it never waits
   with the mutex. If
   the guard is attached to a cold executor which still owns
   VcpuUseToken/ArenaAdmissionToken, unlock the queue, copy/canonicalize its
   already-completed normal outcome into precharged caller storage, ensure TLS
   null and FaultSlot Idle, change an exact Consumed FaultTransitionRecord to
   Empty when present, stop every per-vCPU read, and release VcpuUseToken
   followed by ArenaAdmissionToken; then relock and require the identical
   Closed(sequence, DriverToken), edge-complete Requested shutdown and zero
   caller-arena obligations. Whether the driver was attached and performed that
   release or was already detached, it executes exactly the five-step guarded
   terminal-claim/phase-transfer sequence defined by
   `drive_or_join_terminal`, with this Closed payload as the transfer source;
   no duplicate CAS ordering exists here. It receives TerminalDriverGuard.
   It unlocks and enters terminal cleanup; it never executes steps 4 or 5. A
   failure after the successful claim is fatal terminal continuation, never
   rollback or Open, and requires no new global/admission epoch;
4. only when ShutdownRecord is not Requested, clear both reason hints and the
   shared control request only when
   no Pending record exists, before making Open visible; and
5. while still holding the queue mutex, only when ShutdownRecord is Idle,
   prepare the next validated phase-payload slot and compare-exchange the exact
   Closed phase word to `Open(ticket.reopen_epoch)` with the next checked phase
   sequence. Acquire-load ShutdownRecord again; if non-Idle, leave the gate
   closed and transfer the just-published Open phase to Stopping. If Idle,
   require `open_token_gate == CLOSED|0`, release-store that gate to zero, wake
   it, and immediately acquire-load ShutdownRecord once more. A non-Idle final
   load closes/wakes the gate again before releasing the queue mutex and
   transfers to Stopping; an Idle final load permits unlock. Thus a lock-free
   latch which closed the gate between the first check and reopen cannot be
   overwritten permanently. A token admitted during the finite reopen/reclose
   interval remains counted and validates ShutdownRecord before use; the
   terminal driver cannot acquire the queue mutex and begin its drain until
   this handoff has reclosed the gate.

If step 5 exposes Open with next-ticket work, control remains asserted and,
after unlocking, this same driver is obligated to loop until it either wins the
exact tagged Open-to-Closing CAS for that next ticket while briefly reacquiring
the queue mutex to prepare/validate its payload slot, observes another driver
owning it, or observes Stopping/Stopped. It never delegates correctness to a
requester which may already have lost a CAS. A racing gateway may acquire a
token in that finite Open interval and is safely drained by the next round.
Thus a request belongs to one sealed finite cohort; continual arrivals cannot
extend that cohort, and no final-handoff wakeup is lost. No unconditional phase
store and no bare request-sequence load is a valid transition.

A synchronous nonshutdown request completes only when its exact result is
terminal and either the resulting Open epoch is visible or its
CancelledByStop/TerminatedAfterMutation result is accompanied by Stopping or
Stopped. It drops its result-cell reference and decrements TerminalControl's
the low count of `normal_result_gate` before returning. acknowledged_sequence is not
a substitute for matching generation/sequence. "Bounded rendezvous" means a
bounded guest path to the next specified checkpoint and finite preregistered
cohort work, not a wall-clock guarantee when the host deschedules a native
thread. The normative phase matrix is:

- Open plus a matching OpenToken may publish already-synchronized immutable
  dispatch/PIC roots and may seal previously unreachable RX pages;
- Closing after token drain may withdraw dispatch roots and mark units
  root-inadmissible, but does not patch callable bytes or mutate guest memory;
- after every active epoch is zero, Closed may patch callable pages, clear
  private PIC/suspended roots, exchange directory tables and mutate mappings;
- the sole pre-quiescence exception is a fault resolver's publication of one
  suspended-retry root before it clears its own active epoch; and
- Stopping/Stopped never admits a token or transitions back to a nonterminal
  phase.

Closed performs no compilation, allocation, logging or unrelated index scan;
all storage and exact root lists used there came from ClosingPrepare or bounded
registries. Unreachable worker buffers/pages may be prepared in Open, but
reachable machine bytes never change outside Closed.
