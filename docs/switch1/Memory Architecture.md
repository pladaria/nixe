# Nintendo Switch 1 Memory Architecture

Status: current.

This document describes Nixe's process-memory architecture and the invariants
which CPU, GPU, services and loaders may rely on. The CPU-facing execution
boundary is defined by [CPU Architecture](../CPU%20Architecture.md).

## Layers and authority

A Switch process observes virtual addresses, 4 KiB mapping granules,
permissions, mapping purposes and attributes. Multiple virtual mappings may
refer to one physical page. Nixe preserves those semantics with three distinct
layers:

```text
Horizon-visible virtual mapping state
                  |
                  v
ExecutionMemory sparse mappings and physical-page slots
                  |
                  v
Canonical shared-file backing and visibility state
                  |
                  v
Derived checked or LinuxDirect CPU backend
```

`ExecutionMemory` is the only guest mapping authority. Its sparse table records
address-space identity, virtual page, physical slot, mapping generation,
permissions, purpose and attributes. Stable physical slots own RAM or MMIO and
retain the reverse set of every virtual alias. Mapping queries and SVC policy
never infer semantic state from a host mapping.

Canonical RAM pages own their bytes, stable identity, content generation,
CPU-write provenance and CPU/device visibility state. On Linux, materialized
RAM is backed by a shared `memfd`; mapping the same backing offset more than
once produces coherent aliases without copying. Guest or retained objects
never store direct-arena pointers.

The CPU backend is derived acceleration state. It may be rebuilt from the
canonical mappings and backing pages and can never become a second authority.

## Immutable CPU backend selection

Each process address space selects one backend before its first engine binds:

- `Checked` performs all accesses through `CpuMemory` validation and canonical
  backing operations.
- `LinuxDirect` exposes eligible ordinary RAM through a guarded host virtual
  address-space arena and uses checked operations only for documented
  exceptional access classes.

`DirectBackendPolicy::{Disabled, Preferred, Required}` controls construction.
`Preferred` falls back only when host capability detection says the direct
backend is unsupported. `Required` reports the construction failure. Selection
cannot change after publication, so generated code and interpreter handlers do
not negotiate a backend per access.

Opt-in JIT reports record the selected backend and its construction reason.
This makes an intentional checked run, an unsupported-host fallback and a
successful `LinuxDirect` run distinguishable without inferring the choice from
secondary counters.

Direct construction verifies a 4 KiB host protection granule, coherent shared
file aliases, process VMA inventory and configured safety margin, address-size
conversion and reservation overflow. An arena is sized to the actual guest
address-space limit, with inaccessible lower and upper guard pages.

## Direct address-space arena

An eligible guest page is mapped at:

```text
host address = arena base + guest virtual address
```

Every frontend proves unsigned confinement and required alignment before
forming or selecting a usable pointer. Invalid or overflowing addresses select
the upper poison guard. Unmapped pages remain anonymous `PROT_NONE` ranges.

Arena mappings are published eagerly and in coalesced `mmap`/`mprotect` runs.
Adjacent requests are combined only when guest address, backing offset, file
and protection are compatible. Partial operations retain exact 4 KiB
semantics. Removing a mapping replaces the range with fresh anonymous
`PROT_NONE` memory; it never discards canonical bytes.

Every canonical physical page records non-owning registrations for all of its
direct aliases. Revocation enumerates that reverse set rather than scanning the
guest page table. Registrations include the arena, guest page and maximum
semantically representable data protection.

## Effective host protection

One policy derives host protection from canonical state:

```text
effective protection =
    guest data permission
    AND CPU visibility
    AND ordinary-direct eligibility
    AND executable/observer store policy
    AND exact host representability
```

| Canonical state | Direct data protection |
|---|---|
| unmapped, MMIO, invalid, uncached or GPU-newer | `PROT_NONE` |
| ordinary readable RAM | `PROT_READ` |
| writable RAM before first tracked write | `PROT_READ` |
| writable RAM with native stores armed | `PROT_READ | PROT_WRITE` |
| execute-only mapping | `PROT_NONE` |
| physical page currently observed as executable | never writable directly |

The arena's store-control table contains no permission or visibility flags. A
zero entry means only that a native tracked store is not currently eligible.
Host protection is the access authority. The table remains a single pointer
per 4 KiB guest page so generated stores perform one indexed load. It is a
zero-filled anonymous `MAP_NORESERVE` mapping: a 39-bit address space reserves
1 GiB of host virtual address space, but physical control pages become resident
only for guest address ranges whose entries are actually published. This keeps
the native shape flat without recreating the former eager metadata cost.
Each store revalidates the armed epoch after acquiring its physical page's
write sequence. Revocation clears that epoch and drains any already acquired
writer before reducing host protection, closing the race between a checked
observer and a native store which saw the preceding epoch.

## Execution and transition gate

Every interpreter slice and every direct JIT native interval holds a shared
execution lease. JIT discovery and Cranelift compilation deliberately run
outside it; the compiled entry is revalidated after acquiring a fresh lease
and before native entry. Mapping mutations, external device takeovers,
externally initiated visibility changes, observer baselines and arena teardown
acquire the exclusive transition side:

1. close admission to new slices;
2. wait for existing shared holders to reach a safepoint;
3. update canonical state and all derived aliases;
4. commit one transition epoch and reopen admission.

Waiting transitions cannot be overtaken by new slices. CPU fault recovery runs
under the slice's existing shared lease and does not recursively request
exclusive ownership. Transition acquisition and wait time are diagnostic
counters, not per-access checks.

The runtime transfers an acquired lease into the bounded CPU request. Both JIT
and interpreter validate that a `LinuxDirect` request carries a live lease
whose gate identity belongs to the exact borrowed `ExecutionMemory`. The
interpreter retains it. The JIT releases it during synchronous compilation so
GPU work cannot wait for Cranelift, then reacquires and validates an equivalent
proof for the native interval. This prevents public frontend use from racing
arena removal or substituting a different memory owner while retaining old
native pointers.

The runtime installs a cold transition notifier in the gate. When a retained
range or another external owner closes admission while a CPU slice is active,
the notifier requests `Preempt` from those active CPU controls before the
exclusive side waits. The callback is absent from shared admission and memory
accesses. Reports distinguish notifier requests from exclusive wait time.

## Direct reads

The shared baseline is relaxed, naturally aligned scalar accesses of 1, 2, 4
or 8 bytes which remain inside one ordinary CPU-visible RAM page. JIT code
performs confinement, alignment, base addition and the native load; the
interpreter calls one fixed stub for the width. The JIT proof additionally
covers common scalar/SIMD transfers through 16 bytes and scalar/SIMD pairs.
Pair loads delay both destinations and writeback until both accesses succeed,
and their immutable site metadata identifies the faulting element. Neither
frontend loads a software PTE, permission flag or visibility epoch on its
direct path.

Statically identifiable ordered, atomic, exclusive, device-specialized,
unsupported-attribute and complex structure/lane accesses use the checked
interface directly. SIMD pre/post-index forms also remain checked until they
have the same exact whole-instruction recovery proof.
Dynamically unaligned/cross-page or device targets can instead reach that
interface through an attributed fault, keeping the common generated shape
lookup-free. Their counters determine whether a hot site needs specialization.
A protected eligible read likewise reaches the shared fault runtime. Canonical
classification may reconcile device visibility and resume the access, complete
a dynamic checked read once, report an exact guest data fault, or identify a
fatal backend invariant violation. x86-64 resumes at the original native
instruction; AArch64 re-enters the same committed guest checkpoint because
glibc does not restore every volatile register through `setcontext`.

## Direct stores and observers

Eligible stores use the same relaxed, naturally aligned per-access
restrictions. The interpreter exposes the scalar 1/2/4/8-byte subset; the JIT
also emits common SIMD transfers through 16 bytes and scalar/SIMD pair elements
in architectural order. Access permission still comes exclusively from host
protection. A compact immutable control shared by every writable alias contains
addresses of the physical page's publication counters.

The first JIT write after mapping or observer rearm branches to checked
canonical completion before changing bytes; an interpreter stub may reach the
same completion through an attributed tracking fault. Completion records
whole-page dirty provenance, advances the required generations and arms every
semantically writable alias. Multiple simultaneous first writers serialize
through canonical state; each guest store completes once, while the logical
first-write baseline is published once.

Every later native store:

1. acquires the physical page's write sequence;
2. increments the backing store's active CPU-writer count;
3. performs one naturally aligned native store in the arena (atomic at the
   supported 1/2/4/8-byte scalar widths; 16-byte SIMD stores remain protected
   by the page sequence);
4. advances page generation, content epoch and CPU-write epoch;
5. releases the write sequence and active-writer count.

Counter exhaustion rejects the store through the checked/fatal path rather
than wrapping. The sequence also gives checked readers a coherent snapshot and
prevents retained observers from accepting an in-flight native mutation.

Observer contracts are:

| Consumer | Direct-store contract |
|---|---|
| GPU or retained snapshot | exclusive baseline revokes every writable alias; whole-page changed-since-baseline is conservative and sufficient |
| exclusive reservation | every store advances the physical-page generation, including stores through another alias |
| CPU-write dependency | first write records the complete page; later generation checks may conservatively invalidate subranges |
| content/device epochs | every guest store advances the required page/store epochs |
| executable content | native stores stay disabled for all aliases while any mapping or purpose observes the physical page as code |
| checked concurrent reader | page sequence prevents observation of an in-flight native store, including 16-byte SIMD stores |

The page-level dirty fallback may create false-positive observer invalidation;
it cannot miss a write. Exact-range operations, atomics, exclusives and ordered
stores remain checked.

## Executable content

Instruction fetch records both virtual mapping generations and physical page
content generations. A physical slot counts mappings and purposes which expose
the page as executable content. While that count is nonzero, all direct store
controls for the physical page are cleared and writable aliases are reduced to
read-only.

CPU writes through a non-executable alias update canonical bytes but become
visible to an existing translated instruction stream only through guest cache
maintenance. `IC IVAU` invalidates every translation depending on the physical
page; complete instruction invalidation covers the address space. Host and
device writes publish explicit executable-content invalidations because they
do not necessarily execute the guest cache sequence.

## CPU/device visibility and retained ranges

Canonical pages move between `Clean`, `CpuNewer`, `GpuNewer` and invalid or
conflicting states through an injected visibility coordinator. External
takeover occurs under the exclusive gate and revokes every direct alias before
publishing non-CPU ownership. A CPU access to `GpuNewer` content reconciles the
canonical page before making any direct alias readable.

Retained canonical ranges contain page identities, offsets and generations,
not guest mappings or arena pointers. Snapshot and device-baseline creation
wait for quiescence, observe zero active CPU writers and sample CPU-write epochs
twice. Subsequent overlap checks use the bounded exact journal where possible
and page generation as a conservative fallback.

## Fault runtime

`nixe-cpu-direct-memory` owns one process-wide Linux `SIGSEGV`/`SIGBUS` capture
runtime shared by JIT sites and interpreter stubs. Each executing host worker
registers a fixed slot, alternate signal stack and dispatcher stack before
entering native direct code. The interpreter reuses the published arena,
registry and dispatcher snapshot for its whole bounded slice instead of
republishing it per scalar access. Native regions and fixed stubs publish
immutable, non-overlapping PC ranges. Both preallocated worker stacks are
bounded by `PROT_NONE` guard pages, so exhaustion cannot corrupt adjacent host
memory.

The signal handler performs only bounded TID slot lookup, arena and PC
attribution, nested-fault rejection, volatile context copying and redirection
to an assembly landing pad. It does not allocate, lock, format, access lazy TLS
or call emulator policy. Classification runs afterward on the preallocated
normal dispatcher stack. Retry restores the complete captured host context on
x86-64; checked completion escapes to the prepared native invocation frame. On
AArch64, recovery instead escapes and re-enters the exact interpreter stub or
JIT guest PC from its committed checkpoint, avoiding glibc's partial
volatile-register restore. Both choices are exceptional-path-only and add
nothing to generated steady-state accesses.

A signal is accepted only when its address is inside the active arena or poison
guard and its PC is inside the exact registered access site. Null faults,
outside addresses, outside PCs, nested faults and unrelated `SIGBUS` chain to
the previous disposition or are re-raised with the default action. Worker drop
removes its TID before arena or metadata lifetime may end.

The JIT checkpoints dirty architectural state before every faultable direct
instruction, so guest state is exact without post-register-allocation value
maps. The interpreter state is already canonical before entering a fixed stub.

## Failure and teardown

All fallible canonical validation is completed before changing derived host
mappings. If a partially applied fixed mapping or protection operation cannot
be proven restored, the arena is poisoned, the complete data range becomes
`PROT_NONE`, and the process reports a deterministic backend failure. Execution
never continues with an uncertain or overly permissive alias.

Shutdown stops admission, requests CPU safepoints, drops active worker fault
contexts, removes engine bindings and only then releases process memory. Arena
and immutable registry lifetimes therefore cover every native access which can
still execute. JIT and interpreter entry also compare the currently borrowed
memory's arena view with the immutable process binding before touching native
pointers, so their public APIs cannot silently reuse a frontend after replacing
or destroying its original arena. The direct execution lease also borrows the
memory owner at the type level, preventing safe teardown while the lease lives.

## Diagnostics and invariants

Optional reports distinguish semantic guest faults, checked/tracking writes,
visibility retries, MMIO/checked reads, unattributed faults, nested faults and
`SIGBUS`. Arena counters report mapped, protected, replaced and failed pages,
syscall batches, writable alias pages armed and revoked, baseline/peak VMA use
and transition wait time. Successful guest accesses do not update diagnostic
atomics.

Changes to memory code must preserve these invariants:

- canonical mapping and backing state is the sole authority;
- host pointers never escape into guest or retained state;
- no stale accessible alias survives a committed transition;
- every guest store advances all observers which require per-store notice;
- semantic faults modify no canonical bytes and expose exact pre-fault CPU
  state;
- an attributed checked operation executes exactly once;
- unrelated host faults are never converted to guest faults; and
- checked exceptional paths remain explicit access classes, not a hidden
  alternative normal-RAM architecture.
