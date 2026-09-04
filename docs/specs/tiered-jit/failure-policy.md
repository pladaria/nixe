# Failure policy and concurrency rules

[Specification index](README.md) · [Implementation tasks](tasks/README.md) · [Open review items](review-status.md)

Related contracts: [Request execution, terminal transfer and cohort handoff](coordinator-execution.md); [Epoch reclamation and terminal teardown](epochs-and-shutdown.md).

## Failure policy and concurrency rules

- LCQ invalid guest code or lowering failure is a precise CPU failure.
- HCQ work made stale by invalidation, replacement or reservation loss is
  discarded normally and its unpublished allocation is immediately reusable.
- Only a deterministic property of captured input produces HcqRejected:
  disconnected mandatory endpoints or failure of the fixed instruction,
  spill, segment or island ceilings after deterministic trimming. It releases
  exact reservations, keeps LCQ current and inserts the complete fingerprint in
  NegativeBuildCache.
- A missing guest semantic, invalid state map, verifier failure or corrupted
  backend output is an implementation failure delivered at the next control
  boundary; it is not classified as an optimizer rejection.
- Queue/worker contention, fullness, Closing, ownership loss and executable-
  cache/allocator pressure are transient or stale. They never populate the
  negative cache and defer optimization without blocking the guest.
- HCQ capacity pressure closes background admission until reclamation; it does
  not affect installed LCQ.
- LCQ allocation failure before guest execution is reported. HCQ allocation
  pressure defers its optimization without affecting LCQ. Partial worker
  startup is rolled back completely; zero workers is a valid low-core policy.
- Poison recovery is confined to cold coordination and never unwinds through
  an extern boundary.

Locking is intentionally narrow and nonnested:

- A thread holds at most one of the pending-queue mutex, maintenance-queue
  mutex, JIT-state mutex, slot-registration mutex, ActiveBuild-cell writer
  mutex, code-cache writer mutex, process memory-transaction mutex,
  address-space registry mutex, per-address-space MappingRequestPermit mutex or
  reclaim-control mutex or WorkerJoinSet mutex. UnitPins,
  OpenTokens and atomics are not mutexes. The only mutex held with OpenToken is
  (a) JIT-state, after token acquisition, for no-fail CodeUnit publication, PIC
  root commit or suspended-retry root transfer, (b) slot-registration, which
  is acquired first and then performs a nonwaiting Open acquisition solely for
  activation/deactivation, or (c) one at a time of the ActiveBuild-cell,
  JIT-state and pending-queue mutexes during the nonblocking HCQ admission
  transaction above. Case (c) uses try_lock, performs no wait and retains the
  token through queue ownership or complete rollback. These cases are exhaustive and never overlap; an
  owner neither allocates nor waits. Terminal drains OpenTokens before taking
  slot-registration, so the exceptional order has no cycle.
- WorkerJoinSet is never held with a subsystem mutex and is always unlocked
  before `JoinHandle::join`; partial-start rollback and terminal shutdown are
  its only consumers. A MappingRequestPermit owner releases its permit mutex
  before queue access or MemoryTransactionGuard acquisition.
- ReclaimControl is never held with maintenance/JIT/code-cache/memory mutexes
  or during an epoch/ledger/record wait. Its Running owner unlocks before
  issuing or joining PressureRecord and relocks only to publish/revalidate its
  own cell. The process-global static-initialization pthread mutex is never
  nested and is released before HostSignalControl's install mutex. That install
  mutex is used only in construction/terminal registration, never overlaps any
  subsystem, reclaim or WorkerJoinSet mutex, and is released before a signal-
  handler grace wait.
- The transition driver holds no mutex while waiting for OpenTokens, executor
  epochs, signal landings or record completion. It releases the maintenance
  queue before JIT-state work, the JIT-state mutex before code patching, and the
  code-cache mutex before memory-authority work.
- Guest HCQ admission uses `try_lock` on JIT-state for its NegativeBuildCache
  lookup, then the pending queue in a separate stage; contention rolls back
  exact atomic tokens.
  No guest admission blocks or allocates.
- The JIT-state mutex is not held during fetch/copy, decode, liveness, lowering,
  register allocation, executable/metadata reservation, relocation, cache
  synchronization, logging, patching, mapping calls or worker join. Directory
  COW objects are fully allocated/built and revalidated before their no-fail
  pointer exchanges under this mutex.
- A fault resolver obtains its UnitPin, releases directory read state and clears
  its active epoch before any maintenance wait. It enqueues with no JIT/cache/
  memory lock; the async handler uses only atomics and immutable directory data.
- All condition-variable/futex waits match a generational state/sequence and
  recheck it in a loop. Shutdown wakes every LCQ waiter, worker, coordinator and
  suspended fault. No wait is performed from signal context or generated code.
- Publication may fail only before commit step 1; the first DispatchPayload is
  the reachability point, not the fallibility boundary. Abort
  cleanup compares exact BlockKey, admission/ReachabilityVersion, BuildToken,
  reservation generation, UnitHandle generation and CodeUnitId. A poisoned
  coordination mutex or panic crossing an extern/generated boundary latches
  Stopped/JIT-fatal; execution never resumes using possibly partial state.
