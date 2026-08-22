# CPU Runtime Concurrency Contract

Status: normative for scheduler architecture phases D and E

## Ownership boundary

The runtime coordinator is the only owner allowed to mutate scheduler state,
process registries, guest thread tables, handle tables, address-wait queues, or
Horizon service policy. These values are intentionally `Send` but are not
shared with vCPU workers. A worker receives only an exclusive thread-state
lease, its worker-owned engine executor, a shared CPU-memory view, an immutable
CPU context, a timer view, and cloneable interrupt/invalidation controls.

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
entry points first publish a TLB safepoint request and, after a successful
commit, publish the new mapping epoch as an invalidation request. Executors
acknowledge that request only after stale translations or code can no longer be
re-entered. The Phase E coordinator must not dispatch the next slice until all
controls acknowledge the committed epoch.

The first parallel implementation deliberately serializes semantic memory
transactions. This is stronger than guest hardware ordering but cannot expose
an architecturally forbidden partial cross-page operation. Atomic and
exclusive operations use the same canonical transaction boundary.

## Kernel objects and external events

Handle transfer is one coordinator critical section: destination capacity is
reserved before the source is changed. Sessions, ports, events, shared memory,
and thread identities contain thread-safe shared identities, but table mutation
remains coordinator-owned. Blocking host I/O is not permitted on a vCPU worker.

GPU completion, display, input, timer, IPC, and host-stop producers publish
sequenced events into the bounded external inbox. Publication never acquires a
scheduler or process lock. Device completion establishes canonical CPU
visibility before its guest-visible timeline/event notification is published.

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
