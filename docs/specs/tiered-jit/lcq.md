# Synchronous LCQ compiler

[Specification index](README.md) · [Implementation tasks](tasks/README.md) · [Open review items](review-status.md)

Related contracts: [Cranelift fork and backend contract](backend.md); [Compilation and publication pipeline](publication.md); [Native fault transport and completion](faults.md).

## LCQ baseline compiler

An LCQ unit is one single-entry straight-line fragment representing a naturally
reached basic block, except that the emergency ceiling may split an unusually
long block:

- it starts at the demanded BlockKey;
- it ends after the first direct branch, conditional branch, call, indirect
  branch, return, SVC, BRK, FP-mode boundary, unsupported instruction or
  architectural exit;
- a straight-line block is cut after 512 instructions and continues through an
  ordinary static link;
- it decodes no successor speculatively; and
- it retains every exact dependency read while decoding that block.

The requesting vCPU compiles LCQ synchronously with opt_level=none and
single_pass register allocation. Each DispatchSlot contains one
`LcqBuildCell` with exactly:

```text
Idle -> Building(token, ReachabilityVersion, admission_epoch)
     -> Published|Failed(error)|Stale
Stale|Published|Failed -> Idle(new ReachabilityVersion or slot reuse)
```

A requester takes the JIT-state mutex, validates the slot/payload and performs
the TaggedPayloadCell exact-state transition Idle to Building, then unlocks
before decoding or compiling. An exact matching competitor canonicalizes, owns
no OpenToken/epoch/lock while waiting. Before unlocking the JIT-state mutex it
checked-increments that DispatchSlot's waiter_refcount and revalidates the exact
cell/slot generation. It then snapshots the cell's parking sequence,
revalidates the full slot generation, BuildToken and ReachabilityVersion, and futex-waits only
on that AtomicU32. Publication, failure and invalidation each take the same
short JIT-state mutex, validate the full payload, publish a terminal tag, then
advance/wake the parking sequence after unlocking. Published/Failed remains
attached to that ReachabilityVersion until it changes or the slot is removed.
Each waiter rechecks the current slot/cell and either observes Published,
reports the same precise failure, or retries. In all cases it first release-
decrements its waiter reference. Published only authorizes a fresh
`cold_reentry`: that path acquires OpenToken, publishes a nonzero active epoch
and only then loads the current DispatchPayload/native entry. A waiter reference
never authorizes an executable-pointer read or jump. Slot removal first
publishes a nonacquirable build state and waits waiter_refcount plus the exact
builder-owner reference to reach zero before generation reuse.

When a commit installs a new DispatchPayload with ReachabilityVersion Rnew,
the JIT-state-mutex owner first changes an old Published or Failed(Rold) cell to
Idle(Rnew), then release-publishes that payload. If the old cell is
Building(token, Rold), invalidation instead publishes Stale(token, Rold) and
the new payload remains canonical/unavailable until that exact builder drops
all staging resources; only that builder may then change exact Stale to
Idle(current Rnew) and wake. Slot removal publishes the same nonacquirable
state, waits both references, increments the slot generation and constructs a
fresh Idle cell for the new key. No reset depends on observing Open without an
OpenToken or on a bare ReachabilityVersion load. Cleanup matches the complete
token, slot generation and old reachability and cannot clear a newer builder.
Unrelated keys compile concurrently from
vCPU-local decoder/Cranelift scratch; there is no process compiler mutex.

LCQ never invokes HCQ discovery, a CFG breadth-first search or a function
scanner. Rare overlap caused by an indirect entry into the middle of an
existing LCQ block is accepted because the units are small, versioned and
reclaimable. Both units are indexed for invalidation.
