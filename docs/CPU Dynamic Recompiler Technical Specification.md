# CPU Dynamic Recompiler Technical Specification

Status: superseded architecture record, 2026-08-27.

The active CPU architecture is the
[CPU Architecture Simplification Plan](../notes/CPU%20Architecture%20Simplification%20Plan.md).
This document describes the pre-S00 private-IR, generic-engine, fallback,
tiered-compilation, bounded-cache, and reclamation design. It remains historical
evidence only and must not guide new CPU implementation work.

Audience: CPU, memory, kernel, scheduler, and GPU implementers

Primary target: Arm A64, A32, and T32 guest code on x86-64 and AArch64 hosts

Applies to: Nintendo Switch and Nintendo Switch 2 research profiles

## 1. Purpose

This document specifies the intended architecture of Nixe CPU execution
engines and their scheduler boundary. It is a decision framework for
implementation rather than a claim that all hardware details of either console
are already known.

The planned production JIT is a portable dynamic binary translator (DBT):

```text
A64 frontend ----+
A32 frontend ----+-> verified Nixe typed IR regions -> Cranelift -> native code
T32 frontend ----+                                      |
                                                          v
                                      bounded cache and native link tables
```

Each frontend fetches, decodes, and lifts its execution state's encodings into
the same host-independent IR. The shared IR and Cranelift lowering do not erase
architectural differences: frontend-specific state access, PC behavior, flags,
conditions, and interworking semantics are made explicit before lowering.

An interpreter using the same architectural definitions is required alongside
the recompiler. It is the independent correctness oracle, debugging engine, and
the explicit `InterpretOne` provider for newly recognized instructions which
have not yet been lowered by the JIT. A production-complete JIT has no such
fallback for instructions already supported by the interpreter.

The interpreter, DBT, and any future native-code-execution (NCE) provider are
implementations of one engine-neutral bounded run-slice protocol. NCE is an
optional family of platform engines backed by a suitable virtualization
facility, not direct execution of untrusted guest code in Nixe's host process.
Availability is discovered and rejected with typed diagnostics when the host,
guest profile, memory model, or requested execution policy is incompatible.

The design prioritizes, in this order:

1. Architectural correctness and precise observable behavior.
2. Explicit, testable boundaries between CPU, memory, kernel, and GPU.
3. Low translation latency and predictable runtime behavior.
4. High steady-state performance on supported x86-64 and AArch64 hosts.
5. Maintainability by a small Rust-focused project.
6. Reuse between consoles only where the behavior is genuinely shared.

## 2. Scope and non-goals

The CPU engine is responsible for:

- A64, A32, and T32 instruction decode and architectural semantics.
- Integer, floating-point, SIMD, atomic, and system-instruction execution.
- Guest virtual memory accesses through a defined memory interface.
- Precise synchronous exceptions and well-defined asynchronous exits.
- Translation, caching, linking, and invalidation of host code.
- Coordination with the scheduler and memory visibility mechanisms.
- Instrumentation hooks required for debugging and validation.

The initial CPU engine is not intended to:

- Be cycle accurate or reproduce a particular microarchitecture's pipeline.
- Compile arbitrary source languages or expose a general compiler framework.
- Translate guest GPU shaders; shader translation is a separate subsystem.
- Call Vulkan, Direct3D, Metal, or a GPU driver directly from generated CPU
  code.
- Depend on confidential or unredistributable platform information.
- Guarantee that the two consoles use identical CPU features, page-table
  formats, cache-maintenance behavior, or memory topology.

Timing-sensitive behavior must be modeled at the scheduler and device level.
The CPU JIT reports retired guest instructions and explicit events; it does not
pretend that host instruction count is guest time.

## 3. Design basis and state of the art

The design draws on techniques demonstrated by several maintained systems:

- Dynarmic demonstrates that a focused Arm DBT with a compact IR can expose a
  clean embedding API while supporting unusual memory layouts.
- QEMU TCG demonstrates mature translation-block lookup, direct block chaining,
  software TLBs, page-based invalidation, multicore invalidation, and recovery
  of precise guest state from host faults.
- Cranelift provides portable x86-64 and AArch64 code generation, bounded
  compilation suitable for a baseline JIT, verifier support, and maintained
  host-ABI handling in Rust.

Nixe therefore retains a specialized typed IR for Arm semantics and emulator
effects, then lowers that IR exclusively through Cranelift. Cranelift does not
own emulator dispatch, cache lifetime, link tables, invalidation, memory
authority, or architectural-state commitment; those facilities remain
JIT-private Nixe integration code around the compiler. LLVM and a second
optimizing compiler tier are outside the production plan. The baseline engine
does use the reference interpreter as a cold execution tier so one-shot code
does not pay Cranelift compilation cost.

## 4. Verified facts, profiles, and assumptions

Nintendo publicly describes the Switch 2 processor only as a custom processor
made by NVIDIA. Publicly verified platform facts are less detailed than many
third-party descriptions. The implementation must not turn a provisional
hardware assumption into a shared architectural invariant.

The CPU engine consequently accepts an immutable `GuestCpuProfile` at process
creation. A profile describes behavior rather than a product name:

```rust,ignore
pub struct GuestCpuProfile {
    pub architecture: ArchitectureRevision,
    pub allowed_execution_states: ExecutionStateSet,
    pub address_space: AddressSpaceProfile,
    pub instruction_features: InstructionFeatures,
    pub floating_point: FloatingPointProfile,
    pub cache_maintenance: CacheMaintenanceProfile,
    pub exception_model: ExceptionProfile,
    pub timer_model: TimerProfile,
}
```

Switch 1 process metadata may select A64 or AArch32 execution. An AArch32
process can execute A32 and T32 encodings and can interwork between those states
where the architecture permits it. Supporting such a process therefore
requires real A32 and T32 decoders, architectural state, and semantics; setting
a profile bit cannot make the A64 frontend decode those instruction sets.

For implementation planning, Switch 2 native processes are treated as A64.
This is a conservative software-profile policy, not a claim that every detail
of the Switch 2 CPU or compatibility mechanism is publicly known. A32/T32
availability, the exact architecture revision, optional instruction features,
and compatibility-mode execution behavior remain unresolved Switch 2 profile
questions. They must not be inferred from the host ISA, Switch 1 behavior, or
unverified SoC descriptions.

Switch 1 and Switch 2 select separate profiles. A profile determines which
execution states, encodings, and features are legal for a process, after its
initial state has been obtained from validated process metadata. It does not
replace any execution state's decoder, state model, or semantics. Frontends may
reuse their declarative decoding framework, semantic primitives, IR, and JIT
lowering while enabling different feature bits and platform callbacks.
Unsupported encodings produce the architecturally appropriate exception; they
must never silently execute according to the host's capabilities.

Profile data must be backed by public documentation, lawful black-box tests, or
other redistributable research. Unverified fields remain explicit open
questions.

### 4.1 Recorded Switch 1 FP/SIMD and crypto decision

The built-in Switch 1 profile enables the architectural `AdvancedSimd` decoder
feature. This is not inferred from games or host capabilities: Arm documents
Advanced SIMD/NEON as mandatory for Armv8-A, and NVIDIA's public Tegra X1
documentation identifies NEON on the Cortex-A57 CPU cores. The provisional
Switch 2 native profile keeps this feature `Unknown` until separately verified
evidence establishes its guest-visible contract.

The Tegra X1 data sheet describes a CPU-complex cryptographic engine, but does
not enumerate the guest-visible AES, SHA-1, SHA-256, or CRC32 instruction
feature fields. Arm defines those as optional, independently discoverable
extensions. The Switch 1 profile therefore keeps `Aes`, `Sha1`, `Sha256`, and
`Crc32` as `Unknown`; their decoder rows are unavailable and the required crypto
subset is empty. A future evidence-backed profile revision must enable the
exact named features and add their shared semantics together. Neither the host
ISA nor the separate Tegra security engine may enable them.

## 5. System architecture

```text
                      Runtime coordinator / scheduler
                                  |       ^
              leases + budget     |       | normalized exit
                                  v       |
                   +------------------------------+
                   | execution domain             |
                   |  `-- vCPU executor.run_slice |
                   | interpreter / JIT / NCE      |
                   +--------------+---------------+
                                  |
                           semantic memory access
                                  |
                  +---------------v----------------+
                  |       Guest memory system        |
                  | VA, page tables, RAM, MMIO,      |
                  | permissions, dirty ownership     |
                  +------+--------------------+------+
                         |                    |
                  device MMIO / queues    guest GPU VA
                         |                    |
                  +------v--------------------v------+
                  |       GPU emulation frontend     |
                  | commands, shaders, sync, caches  |
                  +----------------+-----------------+
                                   |
                             host graphics API
```

The memory system is the shared semantic boundary. No engine owns a separate
guest memory model or needs to know whether a guest physical page is represented
by ordinary host RAM, a host-visible GPU allocation, a device-local mirror, a
sparse allocation, or an MMIO handler. A JIT may receive a checked fast
translation for ordinary RAM and exit to a slow path for everything else; an
NCE engine must bind and reconcile its mappings through the same authority.

## 6. Component boundaries

The target logical crate layout is:

```text
nixe-cpu                    architectural state, profiles, decode, semantics,
                            IR, and translation
nixe-cpu-engine             engine identities, capabilities, bounded run-slice
                            contract, normalized exits, and state commit
nixe-cpu-engine-interpreter complete reference-interpreter engine, including
                            semantic dispatch and executor-local state
nixe-cpu-engine-testkit     dev-only fake NCE provider and reusable
                            engine-boundary acceptance fixtures
nixe-cpu-engine-jit         established provider and private native ABI owning
                            Cranelift; generated code, dispatch, links, cache
                            lifetime, and acceleration state remain here
nixe-scheduler              console-neutral thread/vCPU state machine,
                            topology, ready queues, waits, and decisions
nixe-memory                 mappings, canonical physical identity, aliases,
                            generations, invalidation, and visibility
nixe-runtime                process/thread ownership, coordinator, workers,
                            exception routing, and teardown
nixe-horizon                versioned Horizon ABI and policy adapters
future NCE crates           platform virtualization implementations
```

The neutral CPU boundaries are separate workspace crates and dependency-boundary
tests keep them acyclic. Circular dependencies are forbidden. In particular,
CPU, engine-protocol, and scheduler code must not depend on Horizon, the runtime,
Cranelift, a graphics API, or a platform NCE implementation. A concrete engine
must not call Horizon directly. Product composition owns concrete providers;
`nixe-runtime` accepts the neutral provider protocol and does not depend on the
reference interpreter or concrete JIT in production. The JIT may consume the
reference interpreter's context-free single-step API for cold execution; this
dependency does not expose runtime or scheduler ownership to either engine.

The runtime has no product-name or engine-family execution branch. A provider
supplies a process domain, worker executor, memory synchronization, and
normalized exit/trap behavior; canonical thread state, scheduling, exceptions,
and Horizon policy stay unchanged. Every provider must pass the reusable
conformance suite before production registration. The Cranelift JIT provider is
covered by native-ABI, lowering, publication, lifecycle, capability,
precise-memory, exact FP/SIMD, explicit-boundary, full registry differential,
and neutral provider-conformance tests. The dev-only fake NCE exercises shadow
registers, mirrored bindings, dirty-memory reconciliation, migration, and
teardown without a host virtualization API.

The ordered implementation is the portable Cranelift JIT. HVF, KVM, and other
NCE providers remain future independent engines; the shared contracts preserve
their feasibility without treating them as parallel implementations of this
plan. A future NCE must not duplicate architectural semantics or add platform
branches to runtime, scheduler, or Horizon code.

## 7. Process, thread, and vCPU state

The implementation must keep three different lifetimes and ownership domains
separate:

- `ProcessCpuContext` owns immutable profile selection and address-space
  identity. It constrains legal execution behavior but contains no live general
  registers.
- `ThreadCpuState` owns the canonical architectural register state of one guest
  thread. It has distinct A64 and AArch32 representations; CPSR.T selects A32 or
  T32 within the AArch32 representation. Architectural state never belongs to
  an engine.
- An engine executor owns resources associated with a currently executing
  virtual CPU, such as its dispatch budget, pending-control observation, and
  local exclusive monitor. These resources are not part of a guest thread's
  register file or persistent process state. The scheduler defines how local
  monitor state is handled when a thread migrates.

`VcpuExecutionState` is a runtime worker-owned lease container, not a type in
`nixe-cpu` or a place for engine-specific acceleration. Each engine owns its
executor-local representation; `nixe-cpu` exposes only portable exclusive-
reservation values required by the memory contract.

Conceptually:

```rust,ignore
pub struct ProcessCpuContext {
    pub profile_id: CpuProfileId,
    pub address_space_id: AddressSpaceId,
}

pub enum ThreadCpuState {
    A64(A64State),
    A32(A32State),
}

pub struct A64State {
    pub x: [u64; 31],
    pub sp: u64,
    pub pc: u64,
    pub nzcv: u32,
    pub vector: [u128; 32],
    pub fpcr: u32,
    pub fpsr: u32,
    pub thread_pointer: u64,
}

pub struct A32State {
    pub r: [u32; 15],
    pub pc: u32,
    pub cpsr: u32,
    // VFP/NEON storage, FPSCR, and required user-visible system state.
}

```

These examples illustrate semantic fields, not Rust layout or an ABI. Important
rules are:

- A64 state must not be used as the storage model for A32/T32 state.
- A32 PC reads, CPSR flags, T state, register banking assumptions, and VFP/NEON
  aliases are represented according to AArch32 semantics.
- A32/T32 interworking updates architectural state; it is not a profile change.
- Canonical Rust layout is not a save-state, native-engine interchange, or
  generated-code ABI.
- A native engine imports and exports fields explicitly; generated code
  addresses only its provider-private checked layout.
- State visible to helpers is committed before a helper that can observe it.
- State not observable by an exit may remain in host registers within a region.
- Every faulting IR operation has enough metadata to reconstruct precise guest
  state.
- Host floating-point state is treated as scratch owned by the executor and is
  restored at all host ABI boundaries.
- An engine may cache architectural state while executing, but it commits every
  guest-visible component to `ThreadCpuState` before returning an exit.

The JIT imports canonical state into a private native execution frame once at
slice entry and commits it at every observable exit. Cranelift may keep values
in host registers within and across internal region edges; neither register
assignment nor the native frame layout is part of `ThreadCpuState` or the
engine-neutral ABI.

### 7.1 Engine run-slice boundary

The runtime leases one guest thread and one vCPU to an engine executor for a
bounded slice. The request includes an instruction budget or deadline and a
pending-exit token. The executor returns a normalized result such as budget
completion, safepoint, synchronous exception or SVC, memory slow path,
single-instruction interpreter fallback, asynchronous event, termination, or a
typed engine fault.

Engine-specific traps and host failures are translated inside the concrete
engine adapter. Guest exceptions remain distinct from engine faults. Every
successful return reports precise canonical thread state and all guest-visible
memory effects required at that boundary; neither the scheduler nor Horizon
consumes engine-private exit types.

## 8. Decoders and canonical semantics

Separate A64, A32, and T32 decoders should be generated from declarative
instruction tables or use table-driven decision trees behind a shared decoder
interface. T32 must classify and assemble its 16-bit and 32-bit encodings
correctly; A32 conditional execution and A32/T32 interworking remain explicit.
Each entry specifies:

- Encoding mask and value.
- Required guest feature set.
- Operand extraction.
- Reserved and unallocated constraints.
- Semantic handler.
- Interpreter and IR-lifter coverage identifiers.

Each decoder must distinguish unallocated encodings from implemented but
profile-disabled instructions and from recognized instructions whose semantics
or IR lifting are not implemented yet.

Semantics should be centralized around reusable primitives: add-with-carry,
shift-with-carry, bit masks, saturation, floating-point conversion, vector lane
selection, and memory ordering. The interpreter and lifter must not maintain two
independent handwritten interpretations of difficult rules.

The A32 and T32 tables feed distinct typed normalizers and then share an
encoding-independent AArch32 operand model for data processing, barrel shifts,
ordinary memory, multiple transfer, and vector operations. A32 conditions and
the unconditional encoding space remain A32-owned; T32 instruction width,
ITSTATE, PC rules, and halfword assembly remain T32-owned. Family lifters and
reference-interpreter modules consume normalized instructions and must never
re-extract fields from `InstructionEncoding`.

The Switch 1 frontend includes predicated A32 integer execution, common T32
16-bit forms, selected T32 32-bit forms, ordinary and multiple memory transfers,
the profile-required FP/NEON subset, exceptions, calls, A32/T32 interworking,
and A32 acquire/release and exclusive operations. Predication is explicit in IR
so a false condition cannot perform a memory access, alter exclusive state, or
raise an instruction exception. Exact typed helpers preserve A64 FPCR/FPSR and
AArch32 FPSCR behavior where direct IR is insufficient. The completed
multicore memory model uses the same physical exclusive and ordering contract
for these predicated AArch32 operations.

Generated conformance tests should enumerate boundary encodings and verify that
decoder patterns neither overlap unexpectedly nor leave declared instructions
unreachable.

## 9. Intermediate representation

### 9.1 Form

The IR is typed, SSA-like within one translation unit, and explicitly models
side effects. The production translation unit is a bounded multi-block region,
not a whole guest function: guest code may jump into any aligned instruction
and code pages may change. Region formation internalizes profitable direct and
conditional edges while preserving arbitrary entry points, exact code
dependencies, precise fault locations, execution-state changes, and bounded
safepoints.

Architectural-state read and write operations express guest semantics; they do
not require a canonical-frame load or store at each operation. A guest-visible
region entry imports canonical state, an internal edge passes the current
virtual state directly, and an external exit commits all guest-visible state
and the exact retired-instruction count. This distinction removes block-boundary
state traffic without exposing a native frame layout in Nixe IR.

Required scalar and vector types include:

```text
I1, I8, I16, I32, I64, I128, F16, F32, F64, V64, V128, Address
```

The distinction between integer bits, floating-point values, vector values, and
guest addresses catches lowering mistakes. Guest addresses use dedicated
integer-to-address, address-offset, and address-to-integer operations with an
explicit 32- or 64-bit architectural wrapping width. General bitcasts cannot
enter or leave the address domain. Consequently, frontend IR cannot contain a
host pointer or a loader-storage offset disguised as a guest address.

### 9.2 Operation groups

The first complete IR should cover:

- Integer arithmetic, carry, overflow, shifts, rotates, bit operations.
- Comparisons and condition evaluation.
- Guest register and architectural-state reads/writes.
- Typed loads and stores with direction, size, alignment, byte order, ordering,
  privilege regime, access class, and source PC.
- Acquire, release, barriers, exclusive accesses, and atomic read-modify-write.
- Floating-point arithmetic, conversion, comparison, and status updates.
- Vector lane, arithmetic, permute, widening, narrowing, and saturation ops.
- Direct, conditional, indirect, call, return, and exception exits.
- System-register and cache-maintenance operations.
- Explicit calls to well-defined slow helpers.

An operation that may trap carries a `LocationDescriptor` containing at least
the guest PC and execution context required by the exception model.

Each memory operation represents one complete architectural access. The
frontend does not split an access at a page boundary: JIT lowering may select a
fast path for a proven single-page access or a precise slow path that validates
the whole range before committing visible effects. Pre- and post-indexed base
writeback is emitted after the potentially faulting access so optimization and
lowering preserve exception ordering.

### 9.3 Flags

NZCV must not be represented as an implicit host flags dependency across the
entire IR. Arithmetic produces a lazy flag value, and condition consumers read
only the bits they require. Cranelift may keep a short-lived value in host
flags where profitable, but it must materialize architectural NZCV at exits or
when another operation clobbers the required host flags.

This permits dead-flag elimination and avoids serializing unrelated arithmetic.

### 9.4 Optimization budget

Baseline compilation performs bounded, linear or near-linear passes only:

- Constant folding and algebraic simplification.
- Copy propagation.
- Dead temporary and dead flag elimination.
- Redundant guest-state load/store elimination within the region.
- Address folding.
- Known-bit and zero/sign-extension simplification.
- Local load/store pairing when fault and ordering semantics remain identical.
- Common patterns such as conditional select and rotate recognition.

No baseline pass may have unbounded iteration. Translation latency, generated
code size, and execution speed must be measured separately.

Only bounded local and region-wide transformations needed by the production JIT
are planned. Speculative trace compilation, deoptimization, and a second hot
tier are outside this architecture.

### 9.5 Verification

Every IR block and formed region is verified in debug and test builds for:

- Type correctness.
- Dominance and use-before-definition.
- Exactly one terminator.
- Valid architectural-state accesses.
- Correct effect and exception annotations.
- No reordering of volatile, atomic, or MMIO accesses.

The textual IR printer is a required debugging interface, not optional tooling.

## 10. Translation-region formation

Initial block discovery ends at the earliest of:

- An unconditional or indirect control-flow transfer.
- A conditional branch, whose eligible successors are considered by region
  formation.
- A syscall, exception-generating instruction, or execution-mode change.
- An instruction requiring interpreter fallback.
- A configured instruction or byte limit.
- A guest page boundary in modes where cross-page validation is expensive.

The region key is conceptually:

```rust,ignore
pub struct RegionKey {
    pub address_space_id: u64,
    pub guest_location: LocationDescriptor,
    pub translation_mode: TranslationMode,
    pub root_code_mapping: CodePageDependency,
}
```

Only context that can change translation semantics belongs in the key. Runtime
data such as general registers must not fragment the cache. Resolving a virtual
PC to a physical code-page identity before the main lookup prevents unrelated
mapping changes from invalidating the entire code cache. The root mapping in the
hot key disambiguates aliases and remaps without storing a variable-length key.
A region spanning more than one page records the exact ordered dependency set
in its immutable metadata and reverse invalidation indexes. `translation_mode`
contains only an explicitly selected frontend mode, never arbitrary vCPU state.
Direct branch exits retain the destination guest address and destination
execution state; writable cache-owned link cells resolve those to native
regions after one miss through the JIT-private resolver.

The region builder may internalize bounded direct successors and both sides of
a conditional branch. Indirect edges, required observable exits, and edges that
would exceed instruction, byte, IR, page-dependency, or safepoint limits remain
external. `nixe-cpu` owns region formation for the JIT; the interpreter and
future NCE providers do not depend on the region representation.

Every formed basic-block start is a guest-visible entry. When a newly discovered
target coincides with an instruction boundary inside an existing block, the
block is cut and both paths share the target block; a genuinely distinct
overlapping AArch32 decode remains a separate entry. Each entry is a safepoint,
as is every backward internal edge. Failure to fetch the requested root is a
precise translation failure, while failure to fetch a speculative successor
leaves that edge external for normal dispatch.

## 11. Cranelift lowering and the native ABI

### 11.1 One code generator

`nixe-cpu-engine-jit` pins one compatible Cranelift release and is the only
crate which depends on it. Verified Nixe IR is lowered to Cranelift IR without
re-decoding guest instructions. Both IRs are verified in debug and test builds
before native code is published.

Guest and host capabilities are independent. `GuestCpuProfile` decides whether
an instruction is legal; the JIT's capability probe and guarded lowering decide
how Cranelift implements the same semantics. Pure integer, flag, bitfield,
permutation, lane, narrowing, shift, and bit-preserving FP/SIMD operations must
be emitted as native Cranelift IR; a generic semantic helper is not an allowed
fallback for them. Cranelift owns host instruction
selection, register allocation, calling-convention details, and emission for
supported x86-64 and AArch64 hosts.

Particularly sensitive operations include saturation and narrowing, FP NaNs and
status, vector shifts and table lookups, exclusives and atomics, cache
maintenance, and cross-page accesses. Exact typed slow paths remain permitted
only when execution needs canonical memory, MMIO, shared host state, scheduling,
or FP behavior which cannot yet be proven equivalent to host operations. They
are not a substitute for ordinary native lowering.

### 11.2 Private native execution frame

Generated regions receive one private `repr(C)` execution frame owned by the
JIT. It contains imported architectural state, JIT memory-acceleration data,
control flags, native-chain state, the helper table, and normalized exit
storage. Field offsets are generated and checked. Canonical `ThreadCpuState` is
imported once at slice entry and committed once at every exit that can expose
state; generated code does not cross the Rust ABI at ordinary region
boundaries.

The established frame is versioned and sized explicitly, contains independent
complete A64 and A32/T32 payloads rather than exposing a Rust enum or union,
and uses checked two-limb vector values at its C boundary. Each executor retains
this frame, while domain compilation workers retain independent reusable
Cranelift contexts. The sole domain cache retains
lowered immutable regions and writable link tables. A Cranelift-generated
C-ABI gateway performs the sole transition from Rust to Cranelift's portable
tail convention; linked regions then tail-call one another without host-stack
growth. The frame carries current-region metadata and the cumulative budget
across those calls and is committed only at a normalized engine exit. The
gateway, native links, and single miss resolver are the sole native execution
path; no second semantic executor or runtime-visible region handoff exists.

Cranelift code generation runs in a bounded domain-owned worker pool sized to
`max(1, host logical cores / 2)`. Promotion captures verified Nixe IR on the
vCPU, then interpretation continues while workers compile. Completed code is
adopted only at a dispatch boundary after every captured code dependency is
revalidated. Queued work is bounded and older candidates may be discarded;
running obsolete work is never forcefully terminated and its result is simply
discarded after completion.

The frame, Cranelift IR, software TLB, native entry convention, spill storage,
and exit-record layout never appear in `nixe-cpu-engine`, `nixe-runtime`,
`nixe-memory`, or an NCE crate. Rust panics and unwinding must not cross
generated code. The sole gateway call is contained at a safe Rust boundary and
converts host-side failure to a typed engine fault with precise committed state.

### 11.3 Versioned helper ABI

The helper table is a small, versioned JIT-private ABI generated from typed
declarations. Its memory-read, memory-write, atomic, exclusive, exact-FP, and
system slots have non-unwinding C signatures and are installed only around one
live native chain. The generic named-helper IR is consumed natively for every
pure operation and therefore creates no runtime call or semantic metadata.
Lowering rejects any named operation which is neither translated natively nor
listed explicitly as an exact semantic slow path; an unclassified operation may
not silently expand the helper ABI.
Atomic RMW, CAS, and CASP use the typed atomic slot to reach
the canonical physical-memory transaction; this is a precise baseline lowering,
not a second semantic service. Each helper
declares the architectural state and memory effects it observes, whether it may
fault or schedule, and whether execution may resume inside the region.
Generated code calls no Horizon, scheduler, runtime, GPU, or host graphics API.

Helpers provide the precise path for MMIO, cross-page accesses, uncommon
ordering, exclusives or atomics without an exact native lowering, cache and
system operations, device-authoritative memory, and normalized exits. A helper
which can expose guest state receives committed state at the declared boundary.
Hot recoverable memory helpers may resume at a side entry in the originating
region after updating the execution frame; architectural exits return directly
to the sole gateway boundary.

Helper table indices, signatures, frame offsets, and resume targets are
validated before publication. Adding a helper does not expand the neutral engine
contract and does not create a second semantic memory service.

## 12. Code cache, native linking, and resolution

### 12.1 Lookup

The miss resolver uses a 64-slot direct-mapped per-vCPU lookup followed by the
bounded domain code cache. A local hit compares the complete fixed-size key and
the entry's atomic live state without acquiring the domain lock. The hot key is
the address-space identity, complete guest location, translation mode, and root
physical code mapping; immutable metadata and reverse indexes retain every
physical and virtual-mapping dependency.

An absent key first executes through a bounded 16K-entry direct-mapped cold
frequency table. The first two visits use the reference interpreter's exact
single-step API inside the same `run_slice`; the third visit promotes the key to
baseline compilation. An instruction unsupported by the interpreter is
promoted immediately so the compiled frontend preserves the normal
`InterpretOne` contract. An already-published global-cache entry bypasses the
cold tier.

Every external direct edge probes one link cell and each computed edge probes a
four-way polymorphic link site. Hits load an immutable target record with
acquire ordering, update current-region state in the frame, and tail-call the
published live entry without returning to Rust. One JIT-private resolver handles
every direct, conditional, indirect, call, and return miss, keeps the frame
imported, and publishes an eligible target with release ordering. A full
polymorphic site remains a miss through this resolver; it does not select a
second unlinked executor. Exception, fallback, control, fault, and budget exits
alone cross the neutral engine boundary.

### 12.2 Ownership

The ownership model is:

- Immutable compiled region metadata after publication, plus one atomic live
  bit used to detach executor-local lookups during retirement.
- One domain cache owning translation identity, reverse-dependency indices,
  single-flight state, deterministic retirement, every published result,
  writable link tables, immutable link targets, and incoming-link indexes.
- Read-mostly access by vCPU threads.
- The domain coordinates single-flight compilation and owns all published
  results; compilation scheduling does not change cache ownership.
- One cache-owned cancellation token belongs to each single-flight compile.
  Domain stop closes admission, wakes every waiter, and cooperatively cancels
  frontend/lowering work at bounded boundaries. A result racing cancellation
  is rejected before cache publication; no cancellation type leaves the JIT.
- The JIT provider supplies one reclaimable process-wide executable-memory
  owner retained by live providers, domains, and executors. A domain cache owns
  the identity, metadata, and lifetime of its published results; the
  process owner owns only the bounded OS mapping and page-publication mechanism.
- Link cells point only to immutable cache-owned target payloads. Retirement
  clears incoming and outgoing cells before the target live bit changes.
  Cleared payloads and logically retired regions share an epoch queue and are
  reclaimed only after every executor which could have loaded them is
  quiescent. Executable pages are never rewritten for normal linking.
- Logical retirement first detaches links and shared indexes, then marks the
  entry unavailable. Physical storage returns to the arena after epoch
  quiescence.

A single global lock around dispatch is not acceptable. A coarse lock during
rare cache-segment allocation or retirement can be acceptable if measurements
support it.

### 12.3 Executable memory

The JIT executable-memory owner enforces write xor execute in a 1 GiB virtual
arena with at most 65536 page-isolated publications. Checked accounting rejects an
invalid alignment, arithmetic overflow, byte exhaustion, or segment exhaustion
before publication. A failed platform transition poisons the owner so a
partially transitioned page can never be reused.

Capability probing reserves and seals one internal page containing an
unreachable sentinel. That publication counts against both bounds and exposes
no native entry; every later publication accepts only finalized Cranelift
output.

On non-Apple Unix, newly committed or quiescent recycled pages transition from
read-write to read-execute with `mprotect`. Windows reserves an inaccessible
arena, commits or reopens a recycled segment read-write, seals it read-execute
with `VirtualProtect`, and calls `FlushInstructionCache`. macOS creates one
`MAP_JIT` arena and uses
`pthread_jit_write_protect_np` for thread-local write/execute exclusion plus
`sys_icache_invalidate`; capability probing rejects a missing JIT policy and the
incompatible JIT write-allowlist entitlement. x86-64 relies on its coherent
instruction cache after the compiler publication fence, while AArch64 performs
the required data-cache clean, barriers, and instruction-cache invalidation.

The publisher copies that finalized output during the bounded writable
publication interval. Executing threads see only immutable published regions.
Instruction-cache maintenance is performed according to the host platform on
both x86-64 and AArch64. Executable code and writable link/metadata storage are
separate, and neither an arena pointer nor a native entry address is guest-
visible or part of an engine-neutral API.

### 12.4 Cache pressure

The live domain cache is bounded to 32768 page-isolated publications, 512 MiB of
mapped native storage, and 32 million retained IR operations. Misses for one key
share a condition-variable single flight. Pressure retires the oldest live
region deterministically, marks it unavailable to lock-free local lookups, and
removes all hot keys and reverse-index rows before admitting the new result.

Retired page ranges coalesce in the process arena and are reused before its
high-water mark grows. Reuse is legal only after the owning domain's executor
epochs drain. An invalidation which races compilation advances the cache
revision, cancels and wakes affected flights, and rejects the stale result
rather than publishing it. Domain stop uses the same sole cache ownership with
a terminal cancellation result; there is no second compilation queue or cache.

Persistent on-disk native code is not an initial feature. It creates validation,
relocation, host-feature, executable-version, and security problems. Persisting
decoded metadata or profiles may be considered independently.

## 13. Guest memory architecture

### 13.1 One semantic memory system

All CPU engines and devices use a single semantic memory service. The service
models:

- Guest virtual-to-physical translation.
- Address-space identifiers and mapping generations.
- Read, write, execute, privileged, and device permissions.
- Ordinary RAM, shared memory, aliases, MMIO, and unmapped regions.
- CPU/GPU dirty state and synchronization.
- Executable-page tracking and code invalidation.
- Watchpoints and debugging access.

The loader's validated executable segments feed this service. Loaders do not
create host pointers consumed directly by generated code.

### 13.2 Portable fast path: software TLB

The default fast path is an inline per-vCPU software TLB. Each entry includes:

```rust,ignore
pub struct FastTlbEntry {
    pub guest_page_tag: u64,
    pub host_page_base: usize,
    pub flags: u32,
    pub mapping_epoch: u64,
}
```

The real entry also retains a safe canonical-backing lease, access class,
visibility state, and exact permissions. The host base is valid only for the
lifetime and epoch certified by that lease. `nixe-memory` exposes a neutral
direct-access lease describing canonical identity, lifetime, permissions, and
visibility; it never exposes or stores a JIT software-TLB entry.

For normal RAM, translated code performs tag, permission, access-class,
visibility, width, page-boundary, and epoch checks and then accesses the
retained canonical atomic-word backing. Flags force the slow path for MMIO, watchpoints,
GPU-owned pages, executable or observed writes, unusual alignment or ordering,
and all other special behavior.

Writable pages that can be consumed by a device use a first-write barrier. The
first CPU store in a clean ownership epoch takes a slow path, marks the affected
range `CpuNewer`, updates the TLB entry, and resumes. Further CPU stores may run
directly until a device submission or ownership transition arms the barrier
again. The JIT therefore does not execute a dirty-tracking callback on every
ordinary store.

This design is portable, debuggable, compatible with multiple guest address
spaces, and similar to the proven QEMU SoftMMU strategy.

### 13.3 Cross-page and unaligned access

An access whose bytes can cross a guest page boundary must validate both pages
before committing an architecturally indivisible effect. The fast path may
special-case accesses proven not to cross. The slow path handles splits, MMIO,
endianness, permissions, precise faults, and any atomicity requirement.

Never perform a host load first and attempt to repair an observable partial
effect afterward.

## 14. Code invalidation and cache maintenance

Canonical process memory owns one bounded monotonic semantic invalidation
source. Records describe an affected virtual mapping range, one or two
canonical physical code pages, or a complete guest instruction-cache
invalidation; they never contain a JIT region/link identity or an NCE framework
object. Each engine domain consumes that source through its own acknowledged
cursor. A future NCE can consume the same facts to reconcile VM mappings or
inject traps.

The invalidation and reclamation order is:

1. Canonical memory completes the mapping or content mutation and publishes its
   reserved invalidation cursor.
2. The JIT makes incoming link cells and lookup entries miss.
3. Dependent regions and TLB entries become unavailable for new entries.
4. Every executor acknowledges the event after stale state is unreachable.
5. Native storage is reclaimed only after the retirement epoch has drained.

Arm software normally uses explicit data-cache clean and instruction-cache
invalidate sequences when publishing code. The shared CPU-memory contract
models both operations. In addition, every write to a physical page observed
through an executable or code-purpose alias publishes content invalidation, so
writable aliases and incomplete guest cache sequences cannot expose stale host
code. Once a canonical page has been observed as code, publication of a
device-originated write emits the same physical invalidation before its
writeback can become CPU-visible.

Writes through any virtual alias invalidate regions associated with the same
canonical physical page. Mapping changes invalidate dependencies on the
affected virtual view and only overlapping software-TLB entries. Dispatch polls
shared control and interrupt words directly from generated code. External
invalidation publication raises the shared control word; guest writes to pages
observed as code raise a frame-local invalidation flag in the precise memory
helper. Only a raised word crosses the helper ABI and reads the source cursor.
An invalidation with no reverse-index match and no compilation flight advances
the cache cursor in O(1), without scanning cache slots or changing the
compilation revision. Lost ring history causes a conservative full eviction.

## 15. Precise exceptions and host faults

Every potentially faulting host instruction emitted for a guest operation has a
side-table record:

```text
host PC range
    -> guest PC
    -> guest access description
    -> committed-state map
    -> recovery/slow-path target
```

The table supports binary or page-indexed lookup without allocation. A host
fault handler may only inspect immutable published metadata and write to
preallocated thread-local recovery state. Complex work occurs after control is
transferred to a safe trampoline.

Exceptions are precise with respect to guest instruction order. A faulting
instruction must not expose later register writes. Optimization passes therefore
preserve exception ordering unless they can prove that reordering is
unobservable.

Host arithmetic exceptions are not assumed to match ARM exceptions. Most guest
conditions are checked or generated explicitly.

## 16. Floating point and SIMD

The engine implements architectural FP behavior through one integer-only
provider in `nixe-cpu`, not through the host language's default floating-point
behavior. The interpreter invokes that provider directly and the JIT's typed
slow paths reconstruct a temporary canonical A64 transition from the same
semantic token. AArch32 VFP binary operations reuse its binary32 primitive and
map FPSCR control, cumulative status, and enabled exceptions explicitly. The
provider accounts for:

- FPCR rounding modes.
- Flush-to-zero behavior.
- Default NaN and NaN propagation rules.
- Signaling NaNs.
- Cumulative FPSR exception flags.
- Fused versus unfused operations.
- Min/max variants with different NaN semantics.
- Conversion saturation and invalid-result behavior.

The baseline currently routes every arithmetic FP operation through that exact
provider. Cranelift native FP/vector operations may be added only behind
explicit host-feature, guest-profile, and FPCR guards which prove equivalence;
there is no unguarded host-FP path. Exact helper completion returns the
destination and cumulative FPSR/FPSCR together. An enabled FP exception returns
a precise architectural exit before either is committed, while unrelated FPSR
bits including QC are preserved. The IR retains 64-bit and 128-bit vector
semantics rather than exposing x86-64 or AArch64 register widths.

## 17. Atomics and the memory model

Host memory ordering differs between x86-64 and AArch64 and never defines guest
semantics. The engine-neutral memory contract, and the IR which consumes it,
distinguish:

- Plain memory accesses.
- Acquire and release.
- Acquire-release and sequentially consistent atomics.
- Ordered and unordered device accesses.
- DMB, DSB, and ISB with their scopes and domains.
- Exclusive load/store pairs and explicit exclusive clear.
- Profile-enabled atomic read-modify-write instructions.

Cranelift lowering may implement a guest operation with stronger host ordering
when observable behavior remains correct, but systematic over-serialization is
a performance bug and may conceal missing guest synchronization in tests.

`MemoryOrdering` carries relaxed, acquire, release, acquire-release, and
sequentially consistent requirements. `BarrierOperation` carries DMB, DSB, or
ISB plus shareability domain and read/write scope outside JIT IR. The
interpreter applies its portable host-fence mapping, the Cranelift provider
lowers it independently, and a future NCE may map the same descriptor to native
ordering and traps without importing the JIT helper ABI.

Exclusive accesses require one monitor owned by the active vCPU executor. A
reservation records canonical physical page, exact byte offset and width, and
the generation observed by load-exclusive. Store-exclusive succeeds only when
that identity and generation remain current and the canonical writer commits
the bytes indivisibly. Writes through any virtual alias and CPU/device
visibility publication advance the same authority. Explicit clear resets the
monitor, and scheduler migration clears the old vCPU executor before the guest
thread may run on another vCPU. Implementing `LDXR`/`STXR` as an isolated host
`cmpxchg` without that monitor is forbidden.

Atomic RMW, CAS, and CASP linearize at canonical physical backing. Successful
writes advance page generation, store mutation epochs, CPU-write provenance,
and executable invalidation once; failed CAS observes the previous value but
publishes no write. Software-TLB entries retain the same backing and validity
authority and therefore do not define a separate atomic order. Mapping changes
still revoke their eligibility through the acknowledged mapping epoch and
invalidation stream.

The Switch 1 Cortex-A57 profile is Armv8.0-A, while FEAT_LSE begins in
Armv8.1-A. Switch 1 therefore reports LSE as disabled and uses its complete
load/store-exclusive instruction subset. The LSE decoder, interpreter, IR, and
Cranelift lowering remain capability-gated for independently verified future
profiles; the provisional Switch 2 profile continues to report that capability
as unknown.

All CPU and relevant device writes participate in the generation/ownership
mechanism. Multicore conformance runs contending atomic and exclusive
transactions on simultaneous host threads and release/acquire message passing,
so deterministic scheduler serialization cannot make those tests pass.

## 18. Multicore execution and scheduling

Guest threads are runtime objects scheduled onto emulated vCPUs; they are not
permanently represented by host threads. Parallel execution uses at most one
long-lived host worker per active vCPU, with no more than one leased guest thread
executing on a vCPU at a time. This bounds host resources and keeps priorities,
affinity, migration, suspension, and event delivery under the guest scheduler's
control. Every engine cooperates through the same safepoint and run-slice
protocol.

A region receives a bounded instruction budget. Generated code checks for
exits at entry, bounded safepoints, and backward branches. The check covers:

- Timeslice exhaustion.
- Pending interrupts.
- Normalized preemption, debug-stop, or process-stop requests.
- Global TLB or code invalidation requests.
- Process termination.

Regions poll at bounded safepoints and backward edges. No region-formation or
linking decision may exceed the provider's declared maximum poll interval.
The common native poll is two atomic word loads plus frame-local flag loads and
a predicted branch. It calls the Rust slow path only for a published control
request, pending interrupt, or guest-generated local invalidation.

The runtime owns one persistent event register and pending-interrupt mask per
emulated physical vCPU. That state crosses thread and process dispatches on the
same vCPU and is passed to every engine as a cloneable neutral handle; it is not
stored in the JIT cache, executor control, or guest thread state. A native frame
borrows only the stable address of its pending-interrupt word for the bounded
duration of `run_slice`.
`YIELD`, `WFE`, `WFI`, and `SEV` retire through typed normalized scheduling
requests. `SEVL` sets only the current vCPU event register. The coordinator
alone registers waits, broadcasts `SEV`, injects per-vCPU interrupts, and makes
threads ready, closing publication/wait races before the next lease is chosen.
A future NCE consumes this same contract through injected interrupts or
virtualization exits without importing the JIT helper ABI.

Runtime timer registers are sampled at the exact guest `MRS`, including inside
a linked native chain; a slice-entry timer snapshot is not architectural state
and is not part of the native frame. Profile constants remain shared CPU
semantics. The coordinator owns one capped adaptive budget policy for both
deterministic and parallel execution. Exact caller-supplied budgets are retained
only for replay and deterministic verification, not as a second product policy.

Deterministic execution is a permanent policy, not temporary bring-up code. It
models every configured vCPU while allowing only one host worker to execute one
guest slice at a time, and it remains the oracle for tests, diagnostics, and
replay. Parallel execution may be enabled only after atomics, invalidation, TLB
shootdown, shared runtime state, and device visibility have explicit
concurrency contracts and tests. Deterministic replay records scheduling and
external events, not host timing.

Process teardown is identical under both policies. Runtime first closes
dispatch and calls the engine-neutral non-blocking domain stop request. Every
worker then prepares its exclusively owned primary and semantic-fallback
executors against the final canonical memory binding: stale mappings and TLB
entries become unreachable, the final invalidation cursor is acknowledged, and
the local exclusive monitor is cleared. Runtime verifies acknowledgements,
drops all executors, and only then performs idempotent domain shutdown and
releases process memory.

The JIT stop request atomically closes cache admission, cancels queued and
in-progress single-flight compilation, detaches all links and regions, and
publishes preemption to every executor control. Executor retirement epochs keep
detached code and link payloads alive until native entry is impossible; the
last executor drop triggers another reclamation pass. Translation, Cranelift
compilation, and executable allocation never run while the cache-state lock is
held, and executable-publication destruction never nests the arena lock under
that cache lock. Worker retirement and join have a host-time bound which
returns a typed failure without releasing the still-owned process resources.

The shared lifecycle contains no JIT representation. A future NCE uses domain
stop to interrupt virtual CPUs, executor preparation to export canonical state
and acknowledge mappings, and domain shutdown to reconcile remaining dirty
state and release its VM. Compilation cancellation remains exclusively JIT
private.

## 19. CPU and GPU communication

### 19.1 Separation of responsibilities

The CPU JIT does not submit host graphics work. Guest CPU code communicates with
the emulated GPU as real software does: by writing memory, configuring MMIO or
services, submitting command queues, and waiting or signaling synchronization
objects.

The device layer turns these actions into GPU frontend work. The GPU frontend:

- Resolves guest GPU virtual addresses through the GPU memory manager.
- Reads guest command buffers and descriptors.
- Translates GPU commands and shaders.
- Tracks resource usage, barriers, and completion.
- Reports interrupts, fences, and memory visibility through runtime services.

This boundary allows Vulkan, another host API, or a software GPU backend without
changing generated CPU code.

### 19.2 Shared addressable backing

Guest CPU and GPU mappings may refer to the same guest physical pages even when
the host has physically separate CPU and GPU memory. The memory system therefore
uses a canonical page identity independent of its current host representation:

```text
GuestPageId
    CPU mapping(s)
    GPU mapping(s)
    canonical ownership/version
    optional host RAM backing
    optional host GPU mirror
```

Aliased CPU virtual addresses and GPU virtual addresses must converge on this
identity for dirty tracking and synchronization.

### 19.3 Host unified-memory path

When the host GPU can efficiently access host-visible memory, ordinary guest RAM
may be backed by persistently mapped allocations or imported host memory. This
can avoid copies, but it does not imply automatic synchronization.

The graphics backend must query actual host memory properties. For Vulkan:

- Host-visible memory permits mapping; it does not imply host coherence.
- Non-coherent mappings require range-aligned flushes before device visibility
  and invalidation before host reads of device writes.
- Queue submissions and explicit dependencies establish host/device availability
  and visibility.
- Host-coherent memory removes explicit host cache management requirements, not
  logical CPU/GPU race or ordering requirements.

The coherency manager translates guest synchronization events into the required
host operations.

### 19.4 Host discrete-memory path

On a discrete GPU, keeping all guest RAM in host-visible GPU memory may severely
reduce CPU or GPU performance. The preferred representation is adaptive:

- Canonical host RAM for CPU-heavy and general pages.
- Device-local mirrors for GPU resources.
- Dirty ranges or tiles tracked in both directions.
- Upload before the GPU consumes newer CPU data.
- Download or invalidate before the CPU consumes newer GPU data.
- Deferred writeback while ownership and synchronization prove CPU observation
  impossible.

Copies are batched at guest synchronization boundaries, not issued on every JIT
store. Page faults or write protection may optionally discover CPU access to a
GPU-owned region, but explicit mapping and fence information should be preferred
where available.

### 19.5 Ownership state machine

A useful abstract state per range is:

```text
Clean
  | CPU write                    | GPU write
  v                              v
CpuNewer                       GpuNewer
  | upload                       | download/invalidate
  +-------------> Clean <--------+

CpuNewer + unsynchronized GPU write -> guest-defined race or serialized fallback
GpuNewer + unsynchronized CPU write -> guest-defined race or serialized fallback
```

Real tracking should use subresource-aware GPU ranges where page granularity is
too coarse. The abstract state is independent of whether synchronization is a
copy, cache operation, ownership transfer, or no-op on a coherent host.

### 19.6 Command buffers and JIT visibility

Command-buffer pages remain normal guest memory. A doorbell/MMIO write or kernel
submission causes the GPU frontend to capture or parse commands according to
guest semantics. The JIT must commit earlier stores before the submission helper
observes the queue. Guest release operations and barriers are preserved.

The GPU frontend must not retain raw CPU host pointers across remapping,
invalidation, or backing migration. It retains page identities and versioned
mapping handles.

### 19.7 GPU writeback and CPU reads

If the GPU produces data later read by the CPU, completion alone and visibility
are treated separately:

1. The emulated fence establishes when GPU work completed.
2. Guest synchronization establishes when the CPU may observe it.
3. The coherency manager performs any host API barrier, invalidate, or download.
4. The CPU software-TLB entry becomes readable with the current backing/version.

Until then, a TLB flag routes CPU accesses to the slow path. The slow path may
wait, synchronize, or report the architecturally correct state; generated code
does not contain graphics API logic.

## 20. What is shared between Switch 1 and Switch 2

The following should be shared unless testing disproves the abstraction:

- Shared decoder-table machinery, with distinct A64, A32, and T32 tables.
- Canonical semantic primitives where the Arm execution states genuinely agree.
- Typed IR and verifier.
- Interpreter framework.
- Cranelift lowering in the one portable JIT provider.
- JIT-private code cache, W^X allocator, miss resolver, and link machinery.
- Software TLB structure and memory slow-path ABI.
- Exception metadata and host-fault recovery framework.
- Scheduler safepoint protocol.
- CPU/GPU coherency abstractions and canonical guest-page identity.
- Differential testing, fuzzing, tracing, and profiling tools.

Shared code must be parameterized by behavior. It must not contain scattered
checks such as `if switch2` in instruction lowering.

## 21. What may differ between Switch 1 and Switch 2

Separate profiles or platform implementations may define:

- Allowed process execution states and initial-state metadata interpretation.
- Architecture revision and optional instruction extensions for each state.
- Visible system registers and their values.
- Virtual-address width, page sizes, and translation rules.
- Cache-line and exclusive-reservation granules.
- Cache-maintenance and synchronization behavior visible to software.
- Timer frequency and counter exposure.
- Exception routing and kernel ABI details.
- Number of available guest cores and scheduling topology.
- Memory map, physical memory regions, aliases, and permissions.
- CPU/GPU virtual-memory mapping mechanisms.
- GPU command processor, submission, and coherency details.
- Whether a compatibility mode selects Switch 1 behavior on Switch 2.

ISA and execution-state differences enter through `GuestCpuProfile`. Core
count, topology, priority ranges, and timeslice policy enter through a separate
immutable machine scheduler profile; memory, kernel, and device differences use
their respective platform adapters. Unverified Switch 2 CPU or scheduler facts
remain explicit unknown profile values rather than inheriting Switch 1 values
or a fixed four-core assumption. These differences do not require separate code
generators. Guest semantic differences remain in profiles, normalization, Nixe
IR, and exact helpers.

## 22. Engine-family policy

The engine families are:

1. Reference interpreter: always available, simple, instrumentable, and exact.
2. Cranelift JIT: production-default portable performance engine on supported
   x86-64 and AArch64 hosts when its complete capability probe succeeds.
3. Optional platform NCE engines: selected only when host virtualization and
   the full guest profile, state, memory, trap, and execution-policy contracts
   are supported.

The interpreter is mandatory. JIT and NCE implementations use the same
run-slice and state-commit boundary and may be unavailable without changing
scheduler or Horizon semantics. Engine selection is capability-based; an
explicitly requested incompatible engine fails before guest execution rather
than silently selecting another engine. The JIT interprets cold locations and
promotes repeatedly visited locations to baseline Cranelift code. A speculative
optimizing compiler tier and its deoptimization machinery remain outside the
current architecture.

Product composition registers the JIT before the interpreter. `auto` probes in
that deterministic order and selects the first compatible provider; explicit
`jit` or `interpreter` requests probe only the named provider and fail with its
rejection details when unavailable. A primary provider which advertises
`InterpretOne` is paired with the independently registered interpreter as its
semantic fallback. This policy is expressed entirely through registry order,
stable engine identity, and capabilities, so future NCE providers require no
runtime branch. Process construction validates profile semantics; coordinator
registration validates the actual deterministic or parallel policy for both
the primary and semantic-fallback providers. The runtime never sees a JIT
configuration or compiler type.

Configuration may lower the finite per-domain compiled-region and native-byte
working sets and the maximum number of concurrent compilation flights. These
values are validated both while loading product configuration and while
constructing the JIT provider. Compilation admission is a JIT-private,
teardown-cancellable queue; no compiler flag, IR option, native ABI choice, or
product identity is configurable through this surface.

Platform NCE feasibility, privilege requirements, lifecycle seams, and current
availability are versioned in [Native Code Execution Platform
Feasibility](NCE%20Platform%20Feasibility.md). In particular, Android
virtualization is not currently considered a usable arbitrary-payload NCE
facility.

JIT counters are sampled cheaply; an atomic counter on every region execution
is not acceptable. Deterministic validation uses the same lowering and semantic
boundaries while disabling nondeterministic compilation policy.

One interpreter step receives an immutable execution context containing
`ProcessCpuContext` and a narrow `CpuMemory` view. Architectural register state
remains in `ThreadCpuState`; address-space identity and memory services do not.
This lets scalar loads/stores return structured data faults without making the
interpreter depend on the loader or runtime implementation. Scheduler/event
operations use the engine-neutral physical-vCPU contract. Cache maintenance
already
uses the engine-neutral callback and invalidation source established by
JIT-010. Instructions whose required contract is absent remain explicit
unsupported or future-fallback boundaries rather than approximate no-ops.

The completed frontend covers every A64, A32, and T32 family implemented by the
reference interpreter for the selected Switch 1 profile, including its current
FP/SIMD, acquire/release, and exclusive subsets. It also covers capability-gated
A64 LSE RMW, CAS, and CASP through the neutral atomic contract. The registry
mechanically requires each available family to decode and lift without
`InterpretOne`. Scheduler hints are structured IR operations and execute
through typed neutral requests or local vCPU-event actions; named helper
strings and silent hint fallback are not retained.
The exact profile-required FP/SIMD subset is established, and optional crypto
features remain excluded while their profile status is `Unknown`.

## 23. Fallback policy

Fallback is per instruction or region, not per title. When the lifter encounters
an instruction without JIT support it terminates formation before that
instruction and invokes the interpreter for it. Afterward execution returns to
dispatch.
This mechanism remains part of the architecture for newly recognized
instructions, but the production JIT is not complete until every instruction
already supported by the interpreter runs without `InterpretOne`.

Fallback helpers declare:

- State they read and write.
- Whether they access memory.
- Whether they can raise an exception, schedule, or invalidate code.
- Memory ordering effects.
- Whether execution can resume inside a region.

Unknown instructions do not become no-ops. Profile-disabled or unallocated
encodings take the correct exception path.

Reference-interpreter availability and IR-lifter availability are owned and
tracked independently. `nixe-cpu` owns decoder and lowerer metadata; each
concrete engine owns its semantic coverage. When the frontend cannot lower a
recognized instruction, translation ends immediately before it with an
engine-neutral `InterpretOne` terminator carrying its location, raw encoding,
and stable coverage ID. The selected fallback engine validates those fields and
its own coverage against the live architectural state, executes exactly that
instruction when supported, and resumes at the engine-produced PC. Exceptions,
unsupported semantics, and scheduler exits do not synthesize a normal
fallthrough.

`UnsupportedInstruction` is the normalized engine exit for a recognized
encoding which the selected fallback engine cannot execute. Its diagnostic
contains the raw encoding, deterministic
disassembly, CPU profile through the source location, and the exact guest PC and
execution state. Unallocated, reserved, and profile-disabled encodings instead
leave through the architectural undefined-instruction exception path. No path
may silently skip an instruction or manufacture a successful result.

Tests and validation tools may enable strict fallback policy. Strict mode rejects
every `InterpretOne` dispatch before architectural state is mutated, turning
unexpected fallback coverage into a deterministic test failure.

### 23.1 Coverage discovery

Frontend coverage is generated from the A64, A32, T32-16, and T32-32
declarative decoder registries for a selected `GuestCpuProfile`. Every row
reports decoder availability after execution-state and feature gating plus IR
lowering availability. Reference-semantic coverage is generated and tested by
the interpreter crate instead of being embedded in the neutral frontend. A
decoder entry therefore remains visible when a profile disables it or frontend
lowering is incomplete.

`Lifted` is a frontend completion claim, not merely evidence that a lifter match
arm exists. The generated row may use that state only after it has decoder
classification, IR lowering or an explicit architectural exception, stable
printer output, and a redistributable regression fixture. The
fixture registry is tested by decoding, lowering, verifying, and printing each
completed entry. An instruction added because of a workload report must add its
minimal encoding to that registry and retain a focused semantic test.

One `MissingInstructionTracker` belongs to one process or title scope. It
deduplicates recognized unsupported instructions by stable coverage ID and exact
raw encoding, retains the first PC, opaque runtime-assigned module identity,
execution state, and at most 32 bytes of local instruction context, and counts
total frequency independently from unique occurrences. Runtime integration
feeds `UnsupportedInstruction` terminators into this tracker.

The tracker has one deterministic, bounded export containing the local byte
window required for debugging. It accepts no module paths, title names, host
pointers, or arbitrary caller-provided strings. A report can be reduced to
`MissingInstructionFixture`, which carries only coverage ID, encoding, and
execution state for a regression test.

### 23.2 Diagnostics ownership

Missing-instruction diagnostics are an explicit CPU development tool rather
than an emulation-session policy. Creating a tracker enables collection; code
that does not need a report does not create one. The tracker owns no output path
and performs no I/O, preserving the dependency direction from the CPU frontend
away from runtime, application configuration, graphics APIs, and CLI behavior.
No mutable global diagnostics configuration is permitted.

## 24. Debugging and instrumentation

Required developer features are:

- Single-step interpreter mode.
- Region-level JIT stepping.
- A64, A32, and T32 disassembly plus pre/post-optimization Nixe IR,
  verified Cranelift IR, and host-native disassembly dumps.
- Translation reason and timing traces.
- Register and memory watchpoints.
- Per-op fallback counters.
- Region execution sampling.
- Code cache size, link hit rate, TLB hit rate, and invalidation metrics.
- CPU/GPU upload, download, stall, and dirty-range metrics.
- Deterministic event log sufficient to reproduce scheduler/device ordering.

Instrumentation is inserted through IR or region hooks selected before
translation. Production regions contain no unconditional callback on every
instruction.

The application may opt into provider-private compilation artifacts through
`cpu.jit.dump_directory`. An absent or empty path creates no diagnostic owner,
directory, disassembly, or I/O. The application resolves relative paths from
the configuration directory and passes the result only to
`nixe-cpu-engine-jit`; runtime, interpreter, canonical memory, and the neutral
engine contract do not know the path or artifact format. Each JIT domain writes
guest block bytes, deterministic guest disassembly, verified pre-optimization
Nixe IR, and compilation metadata into a session directory. Artifact slots are
bounded by the configured region-cache capacity and recycled under one
diagnostic lock, so opt-in observation cannot grow without bound during a
long-running title. A completion marker distinguishes complete artifacts from
an interrupted or failed host write.

The frontend exposes a separate opt-in region-report path. Normal translation
does not construct disassembly strings. When requested, the report records each
instruction's guest PC, execution state, raw encoding, deterministic
disassembly, the verified pre-optimization region, each basic block's exact cut
reason, the region entries and exits, and the ordered physical code-page
identities and generations observed by instruction fetch. Page-boundary and
instruction-limit cuts remain distinguishable from guest direct branches even
though all three resume through a direct guest target.

IR dumps carry an explicit `pre-optimization` or `post-optimization` stage. The
frontend currently produces only the former; the latter is an interface
contract for future passes. Root fetch failures produce a structured region
report with a `fetch-fault` reason rather than a partial valid region. Reports
use only guest-domain and stable semantic identities, deterministic ordering,
and no raw host pointers. A raw-byte helper supplies the same bounded-region
report for commands and regression fixtures without bypassing the ordinary
decoder, lifter, region former, or verifier.

## 25. Validation strategy

### 25.1 Unit tests

Each semantic primitive, decoder family, Nixe IR operation, optimization, and
Cranelift lowering receives focused tests. JIT tests verify generated Cranelift
IR, publication metadata, and execution where the test host permits it; they do
not assert bytes from a second Nixe-owned encoder.

### 25.2 Differential execution

For generated instruction sequences:

```text
same initial ProcessCpuContext, ThreadCpuState, and memory snapshot
    -> interpreter
    -> baseline JIT
compare state, memory, exceptions, and retired instruction count
```

Comparison includes undefined or constrained-unpredictable behavior only where
the profile defines a comparison policy.

Where lawful and practical, hardware results or an independent emulator can be
an additional oracle. No single external implementation is assumed correct.

### 25.3 Fuzzing

Fuzz targets include:

- Decoder masks and reserved encodings.
- IR verifier and optimization equivalence.
- JIT versus interpreter scalar and SIMD semantics.
- Cross-page, permission, and alias behavior.
- Fault metadata lookup.
- Concurrent invalidation and link-cell retirement.
- Exclusive monitors and atomic litmus tests.
- CPU/GPU dirty ownership transitions.

### 25.4 Memory-model tests

Litmus tests cover acquire/release, barriers, exclusives, self-modifying code,
and multicore visibility. Tests must run repeatedly with forced yields on both
supported x86-64 and AArch64 hosts; stronger host ordering can otherwise hide
missing barriers.

### 25.5 End-to-end milestones

Small redistributable A64, A32, T32, and mixed A32/T32 test programs should
precede commercial software:

- Integer and branch tests.
- Virtual memory and permission faults.
- Syscalls and thread scheduling.
- Floating-point and NEON suites.
- Atomics and contention.
- Self-modifying code.
- CPU-written GPU command/data followed by fence completion.
- GPU-written buffer read by the CPU after guest synchronization.

## 26. Security and robustness

Guest code is untrusted input even when obtained lawfully. The JIT must:

- Validate every decode and IR block.
- Use checked arithmetic for guest range calculations.
- Never embed an unchecked guest-derived host pointer.
- Enforce W^X and publish immutable code atomically.
- Keep host fault handlers allocation-free and narrowly scoped.
- Bound block length, IR growth, code size, and cache memory.
- Avoid invoking Rust unwinding across generated code or signal frames.
- Validate helper indices, link targets, and resume targets.
- Keep debugging or writable code-cache views inaccessible to guest mappings.

A malformed executable should produce a controlled loader, memory, or guest
exception error, not host undefined behavior.

## 27. Performance policy

Performance work is evidence-driven. Benchmarks separately report:

- Decode/lift time per guest instruction.
- Nixe IR optimization and Cranelift compilation time.
- Generated bytes per guest instruction.
- Cold-start and steady-state execution.
- Dispatcher and indirect-branch miss rates.
- Software-TLB hit/miss cost.
- Helper and interpreter fallback frequency.
- Code invalidation and segment-retirement cost.
- Scheduler safepoint overhead.
- CPU/GPU synchronization bytes and wait time.

The target is low enough translation overhead that code is profitable after few
executions while the single Cranelift JIT remains fast enough for games. Numeric
thresholds should be established from Nixe microbenchmarks on a documented host
matrix rather than copied from unrelated runtimes.

Optimization order should normally be:

1. Remove frequent interpreter/helper fallbacks.
2. Improve memory and dispatch fast paths.
3. Eliminate redundant architectural-state traffic.
4. Improve common integer and SIMD lowering.
5. Improve region formation and linking.
6. Improve bounded Cranelift lowering only after the preceding costs are
   measured.

## 28. Major technical decisions

### D1: Use a specialized typed IR

Decision: accepted.

Justification: it provides a stable boundary between guest semantics and the
one JIT code generator, keeps the interpreter independent, leaves future NCE
engines free of compiler IR, and makes exceptions, ordering, and
instrumentation explicit.

### D2: Use Cranelift as the only JIT code generator

Decision: accepted.

Justification: Cranelift supplies maintained x86-64 and AArch64 instruction
selection, register allocation, ABI handling, verification, and native emission
with compilation latency suitable for the baseline JIT. Nixe retains the
emulator-specific resolver, state frame, helpers, cache, linking, memory fast
path, and invalidation that a general compiler cannot own.

### D3: Keep native-code machinery inside the JIT provider

Decision: accepted.

Justification: Cranelift IR, the execution frame, helper table, software TLB,
link cells, executable memory, cache entries, and retirement epochs are derived
JIT state. Keeping them out of `nixe-cpu-engine`, memory, scheduler, and runtime
preserves the common interpreter/JIT/NCE boundary.

### D4: Implement the software-TLB fast path

Decision: accepted.

Justification: it is portable across the supported host ISAs, supports multiple
address spaces and aliases, keeps faults explicit, and derives safe entries from
canonical memory without making host pointers authoritative. Fault-based
fastmem is not part of the current JIT plan.

### D5: Keep graphics APIs out of generated CPU code

Decision: accepted.

Justification: the semantic boundary is guest memory and device events. This
preserves engine and graphics portability and centralizes CPU/GPU coherence.

### D6: Model one guest memory identity with adaptable host backing

Decision: accepted.

Justification: guest shared/unified memory semantics must work on both integrated
and discrete host GPUs. Canonical page identity plus ownership/version tracking
allows zero-copy, mirroring, or staging without changing the CPU engine.

### D7: Require an interpreter as a first-class engine

Decision: accepted.

Justification: differential testing and incremental instruction coverage are
essential for a maintainable JIT. The interpreter is not temporary scaffolding,
and `InterpretOne` remains available for newly recognized semantics even though
interpreter-parity coverage is a JIT completion gate.

### D8: Preserve NCE as a separate engine family

Decision: accepted.

Future HVF, KVM, or other verified NCE providers consume canonical state,
memory bindings, invalidation, control, and normalized exits. They never depend
on Cranelift or JIT-private state, and NCE never means executing untrusted guest
code directly inside Nixe's host process.

### D9: Bind host workers to active vCPUs, not guest threads

Decision: accepted.

One long-lived host worker per active emulated vCPU is the parallel execution
strategy. A guest thread receives a temporary scheduler lease on a vCPU and its
worker; creating one host thread per guest thread is rejected because it makes
host scheduling and resource consumption accidentally define guest priority,
affinity, migration, and suspension behavior. Permanently serialized execution
is also rejected as the only strategy because it prevents eventual concurrent
vCPU execution, but it remains a permanent deterministic mode over the same
scheduler and topology. Unknown Switch 2 core counts and scheduling details are
supplied by a validated machine scheduler profile and are never encoded as a
four-vCPU invariant.

## 29. Implementation order and completion

The sole ordered implementation and release plan is
[`notes/jit.md`](../notes/jit.md). Its
Cranelift provider, private native ABI, secure executable-memory owner,
frontend parity, bounded region formation, complete lowering, the sole bounded
domain cache, cache-owned native links, the canonical-memory software TLB,
precise invalidation with epoch-safe reclamation, exact shared FP/SIMD
execution, physical atomics, the shared Arm memory-order contract, and
JIT-013's scheduler, physical-vCPU event, timer, system, and adaptive-budget
integration are established. JIT-014's executor ownership, compilation
cancellation, acknowledged teardown, retirement, and bounded worker shutdown
and JIT-015's production registry, semantic fallback, explicit selection, and
bounded JIT resource configuration are also established. JIT-016's complete
registry differential, structural native-path coverage, and provider
conformance are established. The remaining order contains only JIT-017's
release and performance gate.
This specification defines the architecture those tasks must preserve; it does
not maintain a parallel set of phases or accept completion through a
transitional execution path.

A completed slice removes every superseded implementation, adapter, cache,
invalidation mechanism, and test that existed only for the old path. The final
production JIT has one Cranelift lowering path, one bounded domain cache, one
link system, one software TLB with a precise helper slow path, and one
engine-neutral invalidation source.

## 30. Open questions

The following require prototypes or additional lawful research before becoming
decisions:

- Remaining Switch 1 CPU feature details and visible system-register behavior;
  the mandatory Armv8-A Advanced SIMD capability is already recorded.
- Whether verified Switch 2 native process metadata permits any execution state
  other than A64; the provisional native profile remains A64-only meanwhile.
- Exact Switch 2 architecture revision, instruction extensions, visible system
  registers, and feature-disabled encoding behavior.
- Whether Switch 2 compatibility uses native A32/T32 execution, CPU binary
  translation, pretranslated code, or another mechanism; no CPU frontend
  capability is derived from the public compatibility description alone.
- Guest page-table and address-space details exposed to the emulator runtime.
- Required fidelity of cache-maintenance operations for titles and system code.
- Software-TLB shape and replacement policy on supported x86-64 and AArch64
  hosts.
- Code-cache segment sizing and retirement policy under many simultaneously
  active guest threads.
- Granularity of CPU/GPU dirty tracking for buffers, textures, and aliased views.
- Required semantics for CPU/GPU concurrent atomics and accesses to shared
  device-visible memory for each platform profile.
- Feasibility and benefit of imported host memory on each graphics backend.
- Requirements for save states, replay, and debugger integration that affect
  architectural-state versioning.

Open questions must be resolved with a short decision record containing evidence,
alternatives, benchmark/test method, and compatibility impact.

## 31. References

- [Arm: Runtime detection of CPU features on an Armv8-A CPU](https://developer.arm.com/community/arm-community-blogs/b/operating-systems-blog/posts/runtime-detection-of-cpu-features-on-an-armv8-a-cpu)
  — records that Armv8-A makes Advanced SIMD/NEON mandatory for AArch32 and
  AArch64, while AES, SHA, and CRC features require independent runtime
  discovery.
- [NVIDIA Tegra X1 Series SoC Technical Reference Manual](https://forums.developer.nvidia.com/uploads/short-url/4pA0RhQeOC4TEwqPuGNml7uV4Nb.pdf)
  — public SoC documentation identifying NEON and a generic cryptographic
  engine on the Cortex-A57 CPU complex without enumerating the architectural
  crypto feature fields.
- [Dynarmic project overview](https://github.com/azahar-emu/dynarmic) and
  [design documentation](https://github.com/azahar-emu/dynarmic/blob/master/docs/Design.md)
  — focused ARM dynamic recompilation, typed SSA IR, explicit flags, block
  terminals, embedding, and memory-system goals. Its documented accuracy
  limitations are also reasons to retain an independent correctness oracle.
- [QEMU translator internals](https://www.qemu.org/docs/master/devel/tcg.html) — direct
  block chaining, translated-code invalidation, precise exceptions, and MMU
  translation caches.
- [QEMU TCG intermediate representation](https://www.qemu.org/docs/master/devel/tcg-ops.html)
  — typed translation-block IR and CPU-state representation.
- [QEMU multi-threaded TCG](https://www.qemu.org/docs/master/devel/multi-thread-tcg.html)
  — translated-code publication, software-TLB hot paths, cross-vCPU
  invalidation, and memory consistency.
- [Cranelift](https://cranelift.dev/) — fast, maintainable, general-purpose code
  generation and its stated compilation/runtime trade-offs.
- [Armv8-A memory model guide](https://developer.arm.com/-/media/Arm%20Developer%20Community/PDF/Learn%20the%20Architecture/Armv8-A%20memory%20model%20guide.pdf?revision=58b1dd0a-3800-4218-b21a-f95a0332034c)
  — ordering, barriers, and memory types.
- [Armv8 application-level memory model](https://developer.arm.com/-/media/1B6DA269007142C0A160E55EE1D52237.ashx?revision=36e3f097-aa99-46be-89ce-38566e393280)
  — formal application-level ordering background.
- [Vulkan memory model](https://docs.vulkan.org/spec/latest/appendices/memorymodel.html)
  — host/device availability, visibility, and memory-domain operations.
- [Vulkan `vkFlushMappedMemoryRanges`](https://registry.khronos.org/vulkan/specs/latest/man/html/vkFlushMappedMemoryRanges.html)
  — required handling of non-coherent host-visible memory.
- [Nintendo Switch 2 official specifications](https://www.nintendo.com/en-gb/Hardware/Nintendo-Switch-2/Nintendo-Switch-2-Specifications-2785627.html)
  — limits of officially published processor detail.
- [Switchbrew NPDM documentation](https://switchbrew.org/wiki/NPDM) — public
  process metadata research describing the 32/64-bit instruction-mode flag and
  address-space selection used by Switch 1 software.
- [Nintendo Switch 2 developer interview, Chapter 4](https://www.nintendo.com/en-gb/News/2025/April/Ask-the-Developer-Vol-16-Nintendo-Switch-2-Chapter-4-2787954.html)
  — Nintendo's high-level description of its compatibility translation; it
  does not specify a CPU ISA implementation mechanism.
