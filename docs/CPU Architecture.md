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
directly. It is the portable correctness oracle and has no JIT dependency,
translation cache or native-code path.

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
dependencies and faults. Ordinary eligible RAM emits direct host loads and
stores through memory-owned fastmem metadata. MMIO, cross-page operations and
other precise cases call small typed slow functions.

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
compilation of one entry PC. Diagnostics do not select Cranelift policy or
change guest semantics.

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
