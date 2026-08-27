# CPU Runtime Concurrency Contract

Status: normative for runtime, scheduler, memory, fault, and lifetime behavior;
pre-S00 CPU engine topology is superseded.

The [CPU Architecture Simplification Plan](../notes/CPU%20Architecture%20Simplification%20Plan.md)
governs all new CPU architecture work. References below to generic providers,
domains, executors, semantic fallback, compilation flights, eviction,
retirement epochs, or future NCE implementations describe the implementation
being replaced and must not guide new work. The concurrency, canonical-memory,
fault, ordering, invalidation, scheduling-event, and teardown guarantees remain
binding unless that plan replaces their mechanism while preserving their
guest-visible behavior. S10 will rewrite this document after the S09 cutover.

## Ownership boundary

The runtime coordinator alone mutates scheduler lifecycle, leases, waits, the
process registry, address-wait queues, or Horizon service policy. Each
registered process owns its threads' CPU state, objects, and exit records. Each
vCPU worker owns its engine executors. These values are not shared between
owners: a worker receives only an exclusive thread-state lease, a shared
CPU-memory view, an immutable CPU context, an exact timer provider, a cloneable
physical-vCPU event state, and cloneable preemption/invalidation control.

Ownership moves through bounded messages. A worker never retains a reference
to a scheduler entry, process table, handle table, Horizon object table, or GPU
frontend after returning an engine exit. The coordinator never holds a runtime
lock while guest code or a host backend executes.

## Lock hierarchy

Code which needs more than one shared resource acquires locks in this order:

1. execution-domain lifecycle and mapping-mutation gate;
2. process-memory mapping table;
3. canonical allocation transaction;
4. canonical backing-store content-publication transaction, in ascending store
   identity;
5. canonical page state, in ascending canonical page identity;
6. process object state (handle/session/port/event);
7. device timeline or backend state;
8. external-inbox watcher registration.

Scheduler and process registries are coordinator-owned rather than locked and
therefore never appear inside this hierarchy. External callbacks may acquire
only their own device state and the inbox registration lock; they publish a
bounded event and must not call back into the coordinator or Horizon.

No lock is held across host file I/O, backend queue waits, guest execution, or
another subsystem callback. Poisoned synchronization primitives recover their
contained state because guest input must not make teardown impossible.

Provider-private locks do not extend this hierarchy into shared crates. The JIT
cache-state lock is acquired alone: region translation, Cranelift compilation,
executable publication, and executable-arena reclamation all occur without it.
Distinct cache misses acquire one finite domain compilation permit before
frontend/compiler work. Waiting for a permit uses the cache condition variable
without retaining the cache lock; invalidation or domain stop cancels the flight
and wakes the queue. Permit release never nests executable-arena ownership under
cache state.
Dropping retired publications, which may acquire the executable-arena lock,
happens only after releasing cache state. Executor-local lookup, software-TLB,
compiler scratch, native-frame, and exclusive-monitor state has one worker
owner and therefore requires no lock. A virtualization provider likewise may
not hold a framework/VM lock while entering canonical memory or waiting for a
runtime worker.

## Memory mutation and invalidation

Each execution-memory instance publishes a monotonically increasing mapping
epoch. A vCPU execution lease records the epoch at which it starts. Mapping,
permission, and attribute mutation enters the mapping-mutation gate, waits for
all execution leases to reach a safepoint, applies one failure-atomic update,
increments the epoch, and then allows new leases. Content generations remain
canonical-page properties and invalidate exclusive reservations through every
virtual alias.

Every canonical page owned by one execution-memory instance also retains the
same backing-store authority. That authority publishes a monotonically
increasing content-mutation epoch after each successful logical CPU write,
device-write completion, or failure-atomic write batch. Engines read this epoch
with one acquire operation; its cost is independent of the number of resident
pages. Failed transactions do not advance it, and making an already-published
device write CPU-visible does not publish the same mutation a second time.

The store epoch is only an address-space-wide change detector. It neither
replaces nor summarizes per-page content generations: retained ranges, code
dependencies, aliases, and exclusive reservations continue to validate the
generation of each canonical page. In particular, implementations must not use
the maximum page generation as a dirty watermark, because a mutation to a page
below that maximum would be invisible.

The same store also publishes a CPU-write-only epoch and retains a bounded
journal of the canonical page intervals published by CPU writes. Consumers
which care specifically about host writes may capture a fixed canonical byte
dependency, compare the coarse epoch first, and consult the store journal only
after it changes. Lost journal history must invalidate conservatively. Per-page
generations and journals remain authoritative, and consumers may fall back to
them when rebuilding a dependency. Device writes do not advance this
specialized epoch, and an equal epoch is therefore an exact proof that no CPU
write was published anywhere in that store.

A captured byte dependency belongs to the retained derived object which uses
it. Cache lookup compares stable structural identity before creating new
derived state, and successful entries share their dependency observation with
transactional clones. Implementations must not rebuild a dependency by walking
all represented pages before every cache lookup; reconstruction occurs only
when an entry is created or its retained provenance is invalidated.

The mutation intent closes admission before waiting for active leases, so a
stream of new slices cannot starve a pending mapping change. Runtime mapping
entry points publish a mapping-change safepoint request and advance the neutral
mapping epoch after a successful commit. The same failure-atomic mutation
publishes a range record through the process-memory invalidation stream.

The monotonic semantic stream carries mapping ranges, physical executable-page
content changes, and complete instruction-cache invalidations. It never names a
JIT cache object or virtualization framework handle. Each engine domain
consumes the stream through its own cursor and acknowledges it only after stale
derived mappings, translations, links, and code cannot be re-entered. The JIT
clears incoming and outgoing native links, removes affected entries through its
reverse indexes, and reclaims detached payloads and code only after executor
retirement epochs drain. Lost bounded-stream history forces full eviction. The
coordinator does not dispatch the next slice until every required control has
acknowledged the committed cursor. JIT eviction, future NCE reconciliation, and
other provider-private actions never enter the shared record.

Ordinary naturally aligned RAM accesses may execute concurrently through
retained canonical backing. Cross-page accesses, MMIO, and mapping-table slow
paths retain a recoverable semantic transaction because their callbacks and
failure-atomic validation cannot be represented as one host scalar. This lock
is not the Arm memory-order mechanism and no correctness test may rely on it to
serialize vCPUs.

`MemoryOrdering` is the engine-neutral contract for plain, acquire, release,
acquire-release, and sequentially consistent accesses. `BarrierOperation`
separately retains DMB/DSB/ISB, shareability domain, and ordered directions.
The interpreter applies the portable host-fence mapping, Cranelift lowers the
same descriptor, and a future NCE may map it to native ordering or traps without
calling a JIT helper. Relaxed JIT TLB accesses remain relaxed; ordered,
exclusive, volatile, and atomic operations use the precise path unless a
provider proves an equivalent direct lowering.

Atomic read/modify/write and compare/exchange transactions linearize at the
canonical physical backing, not at a virtual mapping or software-TLB entry.
Successful writes advance the canonical page generation and store epochs and
publish executable invalidation exactly once; a failed compare/exchange does
none of those things. Every alias therefore shares one modification order.
Exclusive loads retain physical page, byte offset, width, and generation in the
executor-local monitor. Any canonical write may invalidate that observation;
store-exclusive commits through the same physical writer or fails. A scheduler
migration clears the old vCPU executor's local monitor before the thread may be
dispatched elsewhere. TLB entries need no atomic-specific shootdown because
they retain the same backing and validity authority; mapping changes still use
the acknowledged mapping invalidation protocol above.

## Kernel objects and external events

Handle transfer is one coordinator critical section: destination capacity is
reserved before the source is changed. Sessions, ports, events, shared memory,
and thread identities contain thread-safe shared identities, but table mutation
remains coordinator-owned. Blocking host I/O is not permitted on a vCPU worker.

GPU completion, display, input, timer, IPC, and host-stop producers publish
sequenced events into the bounded external inbox. Publication never acquires a
scheduler or process lock. Device completion establishes canonical CPU
visibility before its guest-visible timeline/event notification is published.

## CPU scheduling events, interrupts, timers, and budgets

The runtime owns one `VcpuEventState` for every emulated physical CPU. Its Arm
event register and normalized pending-interrupt mask persist across guest-thread
and process dispatches on that vCPU. They do not belong to an engine executor,
guest thread, process, native frame, or process-wide control object.
`EngineControl` carries only asynchronous preemption and acknowledged
invalidation; combining those process/executor controls with physical-CPU
events is forbidden.

Engines retire processor hints and return typed engine-neutral scheduler
requests. `YIELD` preempts the current lease. `WFE` atomically consumes a set
local event or registers an event wait. `WFI` continues only while an interrupt
is already pending, otherwise it registers an interrupt wait. `SEVL` sets only
the executing vCPU's event register. `SEV` returns a send-event request which
the coordinator broadcasts to every configured vCPU and uses to wake event
waiters. Runtime interrupt injection names one vCPU, sets its event register,
retains the interrupt mask until an engine observes it, and wakes both WFE and
WFI waiters. The coordinator serializes exit reconciliation, wait registration,
event publication, and readiness changes; engines never mutate scheduler
lifecycle directly.

The timer provider remains runtime-owned and is sampled at the exact guest
system-register read. An engine may not freeze the counter at slice or region
entry. Constant profile-defined system values are shared CPU semantics, while
wall/monotonic clock policy remains outside CPU and engine crates.

Adaptive slice length is coordinator policy shared by deterministic and
parallel modes. The budget grows within a fixed cap only after uninterrupted
budget exhaustion and returns to the machine-profile baseline after a
scheduler, exception, control, or other architectural exit. Exact-budget APIs
exist for replay and deterministic tests; product loops do not implement a
second budget policy. Engines poll at entry and bounded backward-edge or region
safepoints. Atomic event/control publication permits an active worker to stop
without exchanging a host channel message for each short native region.

## Executor and domain teardown

Process removal and coordinator shutdown use the same ordered lifecycle in
deterministic and parallel modes:

1. The coordinator closes new dispatch for the process, publishes preemption
   to every primary and semantic-fallback executor, and calls the domain's
   non-blocking stop request. A JIT closes cache admission and cancels its
   single-flight compilations here; this token is private to the JIT. An NCE
   uses the same neutral phase to stop virtual CPUs.
2. With no scheduler lease in flight, each worker calls executor preparation
   while it still has exclusive ownership. The executor makes local mappings,
   cached code lookups, and TLB entries unreachable, acknowledges the final
   invalidation cursor, and clears its local exclusive monitor. An NCE executor
   may also export vCPU state or reconcile its per-vCPU dirty view here.
3. Runtime verifies required acknowledgements, drops all worker-owned
   executors, and only then invokes idempotent domain shutdown. The domain may
   reconcile remaining domain-wide dirty mappings before releasing its VM. The
   JIT drains retirement epochs and releases link payloads and executable
   publications; no executor may outlive this phase.
4. Canonical process memory, mappings, handles, waits, modules, and address-space
   backing are released only after domain shutdown completes.

These phases never execute guest instructions and therefore cannot fabricate
retired progress. Compilation cancellation is cooperative at bounded frontend
and lowering boundaries; a cancelled or invalidated result cannot publish.
Worker retirement and join operations have a host-time bound and return a typed
failure while ownership remains intact, so a faulty provider cannot make a
public teardown call wait forever or falsely report released resources.

## Capability gate

Parallel execution remains unavailable unless the selected engine reports
concurrent-executor safety, bounded control polling, and acknowledged mapping
invalidation. Safepoint latency is declared as a maximum guest-instruction poll
interval rather than a boolean. Every provider advertising one of these
capabilities must return an out-of-band control path for every executor or
process construction fails. Preemption and invalidation requests are coalesced;
an invalidation is acknowledged separately, after its stale mappings and code
can no longer be re-entered. A provider which cannot meet one of these contracts
reports the missing capability before guest execution. Deterministic serialized
execution remains the default and correctness oracle.
