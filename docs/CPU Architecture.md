# CPU Architecture

This document is the authoritative description of Nixe's current CPU
architecture. Nixe emulates Switch 1 A64 code through two separately selected
backends: a reference interpreter and a direct Cranelift JIT. Neither backend
falls back to the other.

## Shared boundary

`nixe-cpu` owns the platform-selected A64 decoder tables, normalized decoded
instructions, architectural state, pure semantic utilities, execution
requests and exits, and the CPU-facing memory contracts. It does not own a
backend-neutral execution engine or semantic IR.

The selected platform is fixed for one process. Instructions are always 32
bits, the PC is four-byte aligned, and architectural integer registers are
`X0-X30`/`W0-W30` plus SP and PC. Code supplied for the selected platform is
assumed valid; decoding still reports malformed encodings precisely when they
are encountered.

The runtime owns scheduling, thread state, budgets, timers, physical-vCPU
events, process lifecycle and memory mutation. A backend receives an exclusive
architectural state reference and a shared canonical memory interface for one
bounded slice. It returns a typed exit and an exact retired-instruction count.

## Interpreter

`nixe-cpu-interpreter` decodes one A64 instruction and executes its semantics
directly. It is the correctness oracle and has no JIT dependency or
translation cache. On a selected Linux direct-memory backend, eligible relaxed
scalar RAM accesses call fixed native load/store stubs shared with the JIT's
fault runtime. One stable arena/registry/dispatcher snapshot is published for
the bounded interpreter slice, not for every stub call. The stubs contain no
permission or visibility policy. Ordered, atomic, exclusive, cross-page,
device and unsupported accesses continue through the canonical checked
interface. An explicitly selected checked backend uses that interface for
every access.

## Direct JIT

`nixe-cpu-jit` discovers a bounded region from the current PC and lowers each
normalized A64 instruction directly to CLIF. There is no Nixe IR, generic
semantic operation graph, helper-token dispatcher or interpreter fallback.

One process mutex serializes discovery, Cranelift compilation and immutable
publication. A cache miss is compiled synchronously by the requesting guest
thread with this fixed release policy:

- `opt_level=speed`;
- `regalloc_algorithm=backtracking`;
- verifier enabled in debug and tests, disabled in release;
- no tiers, worker pool, hot counters or promotion queue.

Regions may cross direct basic-block edges. Accessed architectural registers
become CLIF SSA values on first use, remain in SSA across internal edges and
are committed only when dirty state becomes externally visible. NZCV is kept
lazy until a condition, architectural read or visible exit needs it.

Every discovered basic-block start is a public entry to the same compiled
function. External entry reads the canonical PC once and dispatches to the
corresponding CLIF block; internal edges bypass that dispatch and retain their
SSA state. Discovery stops when it reaches an entry already published by
another region. This removes overlapping recompilation without splitting
regions at every instruction or introducing a cross-region register ABI.

Pure integer, control, memory and common FP/SIMD operations emit direct CLIF.
Complex FP/SIMD behavior uses small typed functions named for one exact
operation. Runtime calls exist only at real semantic boundaries such as MMIO,
faultable compound memory operations, timers, scheduling hints and cache
maintenance. There is no generic helper ABI.

### Lookup and linking

Published entries are keyed by address space, platform and start PC. Every
basic-block entry maps to its owning native region and function address. Rust
uses a direct-mapped table with a hash-map collision fallback. Static external
edges load stable atomic target slots; a populated slot tail-calls native code
and a null slot returns a compile miss. Indirect edges probe a bounded native
lookup table and return to Rust on miss or collision.

Native code is append-only for the lifetime of `JitProcess`. The arena has a
1 GiB hard limit but commits memory only as code is published. Invalidating a
region removes all of its entries from lookup and clears every incoming and
outgoing link; retired bytes remain allocated and unreachable. There is no
eviction, reclamation, persistent code cache or executor epoch.

### Memory

Canonical memory remains the sole authority for mappings, permissions,
physical aliases, visibility, atomics, exclusive reservations, code
dependencies and faults. Each process address space selects an immutable
`Checked` or `LinuxDirect` backend before an engine binds. On supported Linux
hosts, ordinary RAM is eagerly mapped into a guarded host address-space arena;
host page protections represent guest data permissions and CPU/device
visibility. Eligible JIT reads contain only confinement, required alignment,
base addition and the native load. There is no generated page-table,
permission or visibility lookup.

Eligible relaxed scalar stores use the same arena and a compact physical-page
publication control. The control is tracking metadata, not access authority:
the host mapping alone grants write access. The first write after a visibility
or observer baseline completes once through canonical checked memory and arms
all semantically writable aliases. Subsequent native stores serialize through
one page sequence and advance the content generation, CPU-write epoch and
exclusive-reservation generation for every store. Pages observed as executable
content keep native stores disabled until that observer is removed. Each store
revalidates its armed epoch after acquiring the physical-page sequence;
revocation clears that epoch and drains an acquired writer before reducing host
protection.

The pointer table used to find those controls is a sparse anonymous
`MAP_NORESERVE` mapping. It retains one indexed load in native code while only
committing host pages for guest ranges whose controls are published; its
reserved byte count is reported separately from resident guest mappings.

`nixe-cpu-direct-memory` owns the bounded Linux signal runtime used by both
frontends: process-wide handler installation and chaining, per-worker alternate
stacks and slots, immutable native-PC attribution, assembly landing/retry/escape
pads, and fixed interpreter stubs. The signal handler only captures bounded
state and redirects control. Canonical classification and guest-fault creation
run later on a normal dispatcher stack. Unattributed, nested and out-of-arena
host faults remain fatal. Each frontend compares the arena view supplied by
the currently borrowed memory object with its immutable process binding before
native entry; a stale or replacement arena is rejected before any raw pointer
is used. A public direct request must also supply an `ExecutionMemoryLease`
whose execution-gate identity belongs to that exact borrowed memory; missing
or foreign leases are rejected. The interpreter retains that proof for its
bounded slice. The JIT releases the caller proof while discovering and
compiling, then acquires a fresh lease and reconciles invalidations before each
native interval. If the entry was retired between compilation and acquisition,
it retries compilation outside the lease. The lease type borrows its
`ExecutionMemory`, so safe code cannot destroy the arena owner while a native
proof remains live. Statically identifiable ordered, atomic, exclusive,
device-specialized and complex structure/lane cases use typed checked
operations directly. The JIT direct proof covers common relaxed scalar and
SIMD transfers through 16 bytes plus scalar/SIMD pairs; interpreter stubs cover
the common scalar 1/2/4/8-byte class. Dynamically addressed MMIO,
unaligned/cross-page targets and inaccessible mappings may reach the same
checked classifier through an attributed fault; their independent counters
reveal whether a site needs later specialization.

Before each faultable JIT access, dirty architectural SSA values are committed
to `A64State`. Fault metadata retains the source operation, the pinned integer
register containing its guest address, and only the destination/writeback
contract needed to finish that operation; it does not retain Cranelift
post-register-allocation state maps. Pair loads withhold both destinations and
writeback until both accesses succeed, so a fault on either access observes
the exact pre-instruction state. A resolved visibility fault completes once:
x86-64 retries the original native instruction, while AArch64 re-enters the
same committed guest-PC checkpoint because glibc cannot restore every volatile
register.

Every compiled region retains physical code-page dependencies and the virtual
mapping spans used to fetch it. Mapping changes remove only regions whose
virtual dependencies overlap the changed range. Instruction-cache and
executable-content records remove regions which depend on the affected
physical pages. Lost invalidation history removes all regions for the address
space. Running native code observes the shared invalidation signal at bounded
control points and returns to Rust for reconciliation.

Ordinary A64 stores update canonical bytes and content generations but do not
make those bytes visible to an existing translated instruction stream. Valid
self-modifying code publishes that architectural transition through instruction
cache maintenance; `IC IVAU` invalidates translations through every physical
alias and `IC IALLU` invalidates the complete address space. Host and device
writes remain explicit invalidation producers because they do not necessarily
execute the guest A64 cache-maintenance sequence.

### Budget and control

Budget and pending-control state enter native code once and remain in SSA.
Internal backedges and linked external edges test them before continuing, so a
native loop cannot evade slice boundaries, preemption, shutdown or memory
reconciliation. Fault and architectural exits publish the exact PC and retired
count.

The runtime owns each physical vCPU's event register and interrupt mask.
`YIELD`, `WFE`, `WFI`, `SEV` and `SEVL` return typed scheduling operations;
generated code never changes scheduler lifecycle directly. Architectural
timer reads sample the runtime timer at the guest instruction.

## Diagnostics

The `[cpu.jit]` configuration contains only optional output directories:

- `dump_directory` writes one CLIF file and one native binary per compiled
  region;
- `performance_report_directory` writes one timestamped, title-based TOML
  report at coordinated shutdown.

The aggregate report contains regions and guest blocks discovered, published
entry points, primary and secondary lookup hits, regions compiled, lookup hits
and misses, guest and CLIF instruction counts, native bytes, compile and native
time, Rust exit reasons, slow memory calls and invalidations. Static coverage
counts compiled, unique and overlapping guest instructions, making residual
overlap visible even for indirect targets inside a linear block. Invalidation
details attribute each record to mappings, device, host or cache maintenance
and distinguish relevant records, retired regions, lost history and repeated
compilation of one entry PC. Direct-memory details separately report semantic,
tracking, MMIO, retry, unattributed and nested faults, plus compiled direct
sites by access width; mapping/protection batches, VMA growth and
transition-gate wait time are sampled only when reporting is enabled. Writable
alias pages armed and revoked are reported independently from the number of
host protection syscalls, so batching cannot hide protection churn.
Diagnostics do not select Cranelift policy or change guest semantics.

## Concurrency and teardown

Separate guest vCPUs may execute immutable native regions concurrently.
Compilation and publication remain serialized per process. Atomic link slots,
lookup entries, invalidation signals, control words and canonical memory
metadata are the only state shared with running native code.

Process shutdown publishes the stop word, prevents new compilation, clears
published lookup and link entries, and lets running code return at its next
bounded control point. Runtime drops CPU threads before releasing process
memory. The JIT writes its optional aggregate report during coordinated
process shutdown.

## Change policy

New CPU operations should extend the platform decoder and both concrete
backends directly. A new general semantic IR, fallback path, compilation tier,
asynchronous compiler, native-code reclamation scheme or generic helper layer
requires measured evidence and an explicit architecture amendment before
implementation.
