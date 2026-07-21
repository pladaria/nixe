# CPU Dynamic Recompiler Technical Specification

Status: proposed architecture  
Audience: CPU, memory, kernel, scheduler, and GPU implementers  
Primary target: Arm A64, A32, and T32 guest code on AMD64 hosts
Applies to: Nintendo Switch and Nintendo Switch 2 research profiles

## 1. Purpose

This document specifies the intended architecture of the Swiitx CPU execution
engine. It is a decision framework for implementation rather than a claim that
all hardware details of either console are already known.

The recommended engine is a specialized dynamic binary translator (DBT):

```text
A64 frontend ----+
A32 frontend ----+-> shared typed IR -> AMD64 backend -> executable code cache
T32 frontend ----+
```

Each frontend fetches, decodes, and lifts its execution state's encodings into
the same host-independent IR. The shared IR and backend do not erase
architectural differences: frontend-specific state access, PC behavior, flags,
conditions, and interworking semantics are made explicit before lowering.

An interpreter using the same architectural definitions is required alongside
the recompiler. It is the correctness oracle, debugging engine, and fallback for
instructions which have not yet been lowered by the JIT.

The design prioritizes, in this order:

1. Architectural correctness and precise observable behavior.
2. Explicit, testable boundaries between CPU, memory, kernel, and GPU.
3. Low translation latency and predictable runtime behavior.
4. High steady-state performance on common desktop AMD64 processors.
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

The design combines techniques demonstrated by several maintained systems:

- Dynarmic demonstrates that a focused ARM DBT with a compact IR and dedicated
  x86-64 backend can expose a clean embedding API while supporting unusual
  memory layouts.
- QEMU TCG demonstrates mature translation-block lookup, direct block chaining,
  software TLBs, page-based invalidation, multicore invalidation, and recovery
  of precise guest state from host faults.
- Cranelift demonstrates that fast compilation and verified instruction
  lowering are achievable in a reusable Rust backend. Its general-purpose
  function model is useful as a reference and possible validation backend, but
  it does not by itself solve emulator-specific block patching, fault recovery,
  memory fast paths, or architectural-state synchronization.

The preferred implementation is therefore an emulator-specific IR and AMD64
backend. A general compiler backend is not the primary execution tier. LLVM is
explicitly rejected for the baseline tier because its compilation latency,
optimization scope, and integration complexity are poorly matched to short
translation blocks.

This choice is not an endorsement of unnecessary custom machinery. The custom
backend should implement only the operations required by the supported Arm
frontends and should use small, independently testable components for encoding,
register allocation, patching, and executable-memory management.

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
questions. They must not be inferred from the AMD64 host, Switch 1 behavior, or
unverified SoC descriptions.

Switch 1 and Switch 2 select separate profiles. A profile determines which
execution states, encodings, and features are legal for a process, after its
initial state has been obtained from validated process metadata. It does not
replace any execution state's decoder, state model, or semantics. Frontends may
reuse their declarative decoding framework, semantic primitives, IR, and
backend while enabling different feature bits and platform callbacks.
Unsupported encodings produce the architecturally appropriate exception; they
must never silently execute according to the host's capabilities.

Profile data must be backed by public documentation, lawful black-box tests, or
other redistributable research. Unverified fields remain explicit open
questions.

## 5. System architecture

```text
                         Runtime / kernel HLE
                                  |
                 syscall, exception, scheduling, events
                                  |
        +-------------------------v-------------------------+
        |                  CPU execution engine             |
        |                                                   |
        |  A64 frontend --+                                |
        |  A32 frontend --+-> shared IR -> AMD64 backend   |
        |  T32 frontend --+                    |            |
        |                              code cache           |
        +--------------------+-------------+----------------+
                             |             |
                       memory access   execution events
                             |             |
                  +----------v-------------v---------+
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

The memory system is the shared boundary. The JIT does not know whether a guest
physical page is represented by ordinary host RAM, a host-visible GPU
allocation, a device-local mirror, a sparse allocation, or an MMIO handler.
It receives a fast translation for ordinary RAM and exits to a slow path for
everything else.

## 6. Component boundaries

The logical crate layout should initially be:

```text
crates/cpu/
    state             A64 and AArch32 thread state
    vcpu              non-architectural execution resources
    profile           feature and platform profiles
    decode            generated or table-driven A64/A32/T32 decoders
    semantics         canonical instruction behavior
    interpreter       reference executor
    ir                IR values, operations, blocks, verifier
    translate         A64/A32/T32-to-IR lifting

crates/jit/
    amd64             lowering and machine-code emission
    regalloc          linear-scan register allocation
    code_cache        allocation, lookup, links, invalidation
    dispatch          entry, exit, and indirect-branch dispatch
    executable_memory platform W^X implementation

crates/memory/
    address_space     guest VA mappings and permissions
    fastmem           software TLB and optional fault fastmem
    physical          RAM and device-backed pages
    coherency         CPU/GPU ownership and dirty tracking

crates/runtime/
    process and thread construction
    exception and syscall routing
    scheduler integration
```

These are logical ownership boundaries, not a requirement that every directory
immediately become a separate crate. Circular dependencies are forbidden. In
particular, `cpu` must not depend on `runtime` or a host graphics API.

## 7. Process, thread, and vCPU state

The implementation must keep three different lifetimes and ownership domains
separate:

- `ProcessCpuContext` owns immutable profile selection and address-space
  identity. It constrains legal execution behavior but contains no live general
  registers.
- `ThreadCpuState` owns the canonical architectural register state of one guest
  thread. It has distinct A64 and AArch32 representations; CPSR.T selects A32 or
  T32 within the AArch32 representation.
- `VcpuExecutionState` owns resources associated with a currently executing
  virtual CPU, such as its software TLB, dispatch budget, pending-event state,
  safepoint data, and local exclusive monitor. These resources are not part of
  a guest thread's register file or a persistent process state. The scheduler
  defines how local-monitor state is handled when a thread migrates.

Conceptually:

```rust,ignore
#[repr(C)]
pub struct ProcessCpuContext {
    pub profile_id: CpuProfileId,
    pub address_space_id: AddressSpaceId,
}

pub enum ThreadCpuState {
    A64(A64State),
    A32(A32State),
}

#[repr(C)]
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

#[repr(C)]
pub struct A32State {
    pub r: [u32; 16],
    pub cpsr: u32,
    // VFP/NEON storage, FPSCR, and required user-visible system state.
}

pub struct VcpuExecutionState {
    pub software_tlb: SoftwareTlb,
    pub exclusive: ExclusiveMonitorState,
    pub pending_events: AtomicU32,
    pub dispatch_budget: u64,
}
```

These examples are illustrative; final layouts are decided by generated offset
tests and ABI requirements. Important rules are:

- A64 state must not be used as the storage model for A32/T32 state.
- A32 PC reads, CPSR flags, T state, register banking assumptions, and VFP/NEON
  aliases are represented according to AArch32 semantics.
- A32/T32 interworking updates architectural state; it is not a profile change.
- These layouts are internal and versioned. They are not a save-state format.
- Generated code addresses fields through checked, generated constants.
- State visible to helpers is committed before a helper that can observe it.
- State not observable by an exit may remain in host registers within a block.
- Every faulting IR operation has enough metadata to reconstruct precise guest
  state.
- Host floating-point state is treated as scratch owned by the executor and is
  restored at all host ABI boundaries.

One AMD64 nonvolatile register should normally hold the active
`ThreadCpuState` representation while guest code is running. Another fixed
pointer or a containing execution context may expose `VcpuExecutionState` if
benchmarks show a net benefit across supported host ABIs.

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

Generated conformance tests should enumerate boundary encodings and verify that
decoder patterns neither overlap unexpectedly nor leave declared instructions
unreachable.

## 9. Intermediate representation

### 9.1 Form

The IR is typed, SSA-like within one translation unit, and explicitly models
side effects. A translation unit begins as a basic block and may later become a
small extended block or trace. It is not a whole guest function: guest code may
jump into any aligned instruction and code pages may change.

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
frontend does not split an access at a page boundary: the backend may select a
fast path for a proven single-page access or a precise slow path that validates
the whole range before committing visible effects. Pre- and post-indexed base
writeback is emitted after the potentially faulting access so optimization and
lowering preserve exception ordering.

### 9.3 Flags

NZCV must not be represented as an implicit host flags dependency across the
entire IR. Arithmetic produces a lazy flag value, and condition consumers read
only the bits they require. The backend may keep a short-lived value in host
flags where profitable, but it must materialize architectural NZCV at exits or
when another operation clobbers the required host flags.

This permits dead-flag elimination and avoids serializing unrelated arithmetic.

### 9.4 Optimization budget

Baseline compilation performs bounded, linear or near-linear passes only:

- Constant folding and algebraic simplification.
- Copy propagation.
- Dead temporary and dead flag elimination.
- Redundant guest-state load/store elimination within the unit.
- Address folding.
- Known-bit and zero/sign-extension simplification.
- Local load/store pairing when fault and ordering semantics remain identical.
- Common patterns such as conditional select and rotate recognition.

No baseline pass may have unbounded iteration. Translation latency, generated
code size, and execution speed must be measured separately.

Cross-block optimization is deferred to an optional hot tier. The baseline JIT
must be capable of running complete titles efficiently without it.

### 9.5 Verification

Every IR block is verified in debug and test builds for:

- Type correctness.
- Dominance and use-before-definition.
- Exactly one terminator.
- Valid architectural-state accesses.
- Correct effect and exception annotations.
- No reordering of volatile, atomic, or MMIO accesses.

The textual IR printer is a required debugging interface, not optional tooling.

## 10. Translation unit formation

A baseline block ends at the earliest of:

- An unconditional or indirect control-flow transfer.
- A conditional branch, unless represented as a bounded extended block.
- A syscall, exception-generating instruction, or execution-mode change.
- An instruction requiring interpreter fallback.
- A configured instruction or byte limit.
- A guest page boundary in modes where cross-page validation is expensive.

The block key is conceptually:

```rust,ignore
pub struct BlockKey {
    pub guest_pc: u64,
    pub address_space_id: u64,
    pub code_page_id: u64,
    pub code_generation: u64,
    pub profile_id: u32,
    pub execution_state: ExecutionState,
    pub translation_context: TranslationContext,
}
```

Only context that can change translation semantics belongs in the key. Runtime
data such as general registers must not fragment the cache. Resolving a virtual
PC to a physical code-page identity before the main lookup prevents unrelated
mapping changes from invalidating the entire code cache. A unit spanning more
than one page records every physical page and generation as dependencies in its
metadata rather than expanding the hot lookup key without bound.

`translation_context` contains only additional state that changes decoding or
lifting at block entry, such as T32 IT state when applicable. It is not a bag of
arbitrary vCPU state. Direct branch exits retain the destination guest address
and destination execution state; the dispatcher resolves those to host blocks.

Small extended blocks can include a conditional branch and its fall-through
path. Trace formation and speculative guards are optional future work and must
not complicate precise exceptions in the baseline design.

## 11. AMD64 backend

### 11.1 Host feature tiers

The executable selects a backend feature tier once per process:

- A conservative AMD64 tier for broad compatibility.
- An enhanced SIMD tier using host features such as SSSE3/SSE4.1 when present.
- An AVX2/FMA tier for profitable vector sequences.
- Future tiers only when supported by measurements and tests.

Guest capabilities and host capabilities are independent. A guest instruction
is legal according to `GuestCpuProfile`; its lowering is selected according to
the host tier. Unsupported host operations call a semantically exact helper.

Using AVX requires consistent handling of upper vector state and host ABI
transitions. Mixed SSE/AVX penalties and `vzeroupper` placement must be handled
by backend policy, not individual instruction translators.

### 11.2 Register allocation

A linear-scan allocator is the default. Translation units are short, compile
time matters, and predictable spills are preferable to a complex global
allocator.

Allocator requirements:

- Integer, vector, and flags constraints.
- Fixed-register constraints for shifts, multiply/divide, atomics, and calls.
- Caller/callee-saved knowledge for every supported host ABI.
- Rematerialization of constants and cheap address expressions.
- Spill slots in a dedicated JIT frame, not arbitrary modifications to
  `ThreadCpuState`.
- Parallel move resolution for block exits and helper calls.

The initial implementation should not pin an execution state's entire guest
register file across blocks. Within-block state caching delivers most of the
maintainable benefit. Linked block entry conventions can later carry a small,
measured set of live values.

### 11.3 Instruction selection

Lowering is pattern-based and local. Complex semantics may use helpers until a
native sequence is proven correct. Particularly sensitive areas are:

- Saturating and narrowing SIMD operations.
- Floating-point NaNs, rounding, flush-to-zero, and exception flags.
- Variable vector shifts and table lookups.
- Exclusive monitors and large-system-extension atomics.
- Unaligned accesses crossing a page boundary.

A fast incorrect lowering is worse than a helper. Profiling identifies which
helpers justify native implementations.

### 11.4 Host ABI and generated-code ABI

Generated blocks use a private ABI. Entry and exit trampolines are the only
places that translate between the platform ABI and JIT ABI. Helpers use typed
stubs generated from declarations so that register preservation and stack
alignment cannot diverge between call sites.

Every exit returns an explicit reason:

```rust,ignore
pub enum ExitReason {
    Dispatch { next_pc: u64 },
    Syscall,
    Exception,
    Interrupt,
    TimesliceExpired,
    MemorySlowPath,
    CodeInvalidated,
    DebugTrap,
    Stop,
}
```

Hot memory slow paths should resume inside the originating block when safe
rather than always returning to the outer runtime.

## 12. Code cache and dispatch

### 12.1 Lookup

The dispatcher uses a small per-thread level-zero cache followed by a shared or
per-process block table. An indirect branch may use a polymorphic inline cache
containing a few recent guest-target to host-target pairs.

Direct branches are patched to the destination block after it is compiled.
Links are atomically replaceable so invalidation never exposes a partially
patched instruction stream.

### 12.2 Ownership

The recommended ownership model is:

- Immutable compiled block metadata after publication, except atomic links and
  counters.
- Per-process translation identity and invalidation indices.
- Read-mostly access by vCPU threads.
- Compilation either by the requesting vCPU or a controlled compiler service.
- Reclamation through epochs or stop-the-world cache rotation, never immediate
  free while another vCPU may execute the block.

A single global lock around dispatch is not acceptable. A coarse lock during
rare cache allocation or rotation can be acceptable if measurements support it.

### 12.3 Executable memory

The cache must enforce write xor execute. Preferred implementations use two
views of the same backing allocation, one writable and one executable. Where a
platform cannot provide dual mappings, permission changes are isolated to
sealed batches and synchronized before publication.

The emitter writes only through the writable view. Executing threads see only
fully finalized blocks. Instruction-cache maintenance is performed according to
the host platform even though coherent AMD64 hosts commonly require no explicit
flush.

### 12.4 Cache pressure

Code cache growth is bounded. The first policy should be simple arena rotation:
stop publication, detach links, wait for executing epochs to drain, discard an
arena, and recompile on demand. Fine-grained eviction is deferred until evidence
shows it is required.

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
    pub mapping_epoch: u32,
}
```

For normal RAM, translated code performs tag, permission, and epoch checks and
then accesses `host_page_base + page_offset`. Flags force the slow path for
MMIO, watchpoints, GPU-owned pages, code-write tracking, unusual alignment, or
other special behavior.

Writable pages that can be consumed by a device use a first-write barrier. The
first CPU store in a clean ownership epoch takes a slow path, marks the affected
range `CpuNewer`, updates the TLB entry, and resumes. Further CPU stores may run
directly until a device submission or ownership transition arms the barrier
again. The JIT therefore does not execute a dirty-tracking callback on every
ordinary store.

This design is portable, debuggable, compatible with multiple guest address
spaces, and similar to the proven QEMU SoftMMU strategy.

### 13.3 Optional fault-based fastmem

A later AMD64-specific mode may reserve a large host virtual range and encode
guest addresses directly into host accesses. Host access faults are decoded
using generated-code metadata and redirected to MMIO, permission faults, page
population, or guest exceptions.

Fault fastmem is optional because it introduces substantial costs:

- Signal or structured-exception integration.
- Async-signal-safe lookup constraints.
- Host virtual-address-space requirements.
- Difficult debugger and sanitizer interaction.
- Alias and multiple-address-space complexity.
- Platform-specific recovery code.

The IR and memory API must support both modes without changing AArch64
semantics. Software-TLB fastmem is implemented first and retained as the
portable/reference JIT path.

### 13.4 Cross-page and unaligned access

An access whose bytes can cross a guest page boundary must validate both pages
before committing an architecturally indivisible effect. The fast path may
special-case accesses proven not to cross. The slow path handles splits, MMIO,
endianness, permissions, precise faults, and any atomicity requirement.

Never perform a host load first and attempt to repair an observable partial
effect afterward.

## 14. Code invalidation and cache maintenance

Compiled blocks are indexed by every guest physical code page they cover. Each
page tracks a code generation and a list or compact index of dependent blocks.

When code becomes invalid:

1. The page generation changes.
2. Incoming direct links are atomically redirected to the dispatcher.
3. Blocks are marked unavailable for new entries.
4. Existing executions reach a safe exit before reclamation.

ARM software normally uses explicit data-cache clean and instruction-cache
invalidate sequences when publishing code. The memory subsystem should model
those operations and invalidate at the architecturally visible point. A
conservative write-watch mode must also exist for debugging, incomplete cache
modeling, and mappings where code/data aliases make explicit tracking unsafe.

Writes through any virtual alias must invalidate blocks associated with the same
guest physical page. Indexing solely by guest virtual address is incorrect.

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

The engine must implement architectural FP behavior, not the host language's
default floating-point behavior. The implementation accounts for:

- FPCR rounding modes.
- Flush-to-zero behavior.
- Default NaN and NaN propagation rules.
- Signaling NaNs.
- Cumulative FPSR exception flags.
- Fused versus unfused operations.
- Min/max variants with different NaN semantics.
- Conversion saturation and invalid-result behavior.

Host MXCSR may be specialized for a block when profitable, but all transitions
are explicit and host state is restored before returning to Rust or calling a
normal helper. Rare modes may use software helpers.

NEON is lowered to the best available combination of scalar AMD64, SSE, AVX,
and helpers. The IR retains 64-bit and 128-bit vector semantics rather than
prematurely exposing host register widths. Wider host instructions may combine
independent guest operations only when exceptions and FP status remain correct.

## 17. Atomics and the memory model

AMD64 has a stronger default memory ordering than AArch64, but that does not
remove the need to model AArch64 ordering explicitly. The IR distinguishes:

- Plain memory accesses.
- Acquire and release.
- Acquire-release and sequentially consistent atomics.
- Ordered and unordered device accesses.
- DMB, DSB, and ISB with their scopes and domains.
- Exclusive load/store pairs and explicit exclusive clear.
- Profile-enabled atomic read-modify-write instructions.

The backend may legally implement a guest operation with stronger host ordering
when observable behavior remains correct, but systematic over-serialization is
a performance bug and may conceal missing guest synchronization in tests.

Exclusive accesses require a semantic monitor. A minimal correct model records
the physical granule and a generation observed by the load-exclusive. A
store-exclusive succeeds only when the reservation remains valid and the write
is atomically committed. Interrupts, context changes, conflicting writes, and
explicit clear operations update the reservation according to the selected
profile.

All CPU and relevant device writes participate in the generation/ownership
mechanism. Implementing `LDXR`/`STXR` as an isolated host `cmpxchg` without a
guest monitor is insufficient.

## 18. Multicore execution and scheduling

Each emulated CPU thread may run on a host thread. The scheduler owns guest time,
priorities, affinity, suspension, and event delivery. The JIT cooperates through
safepoints.

A block receives an instruction budget or deadline token. Generated code checks
for exits at bounded intervals and at backward branches. The check covers:

- Timeslice exhaustion.
- Pending interrupts or kernel events.
- Debug stop requests.
- Global TLB or code invalidation requests.
- Process termination.

Checking only at block boundaries is acceptable while blocks are strictly
bounded; trace tiers must insert additional polls.

The first scheduler may be deterministic and conservative. Parallel execution
is enabled only after atomics, invalidation, TLB shootdown, and device visibility
have tests. Deterministic replay should record scheduling and external events,
not host timing.

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
4. The CPU fastmem entry becomes readable with the current backing/version.

Until then, a TLB flag routes CPU accesses to the slow path. The slow path may
wait, synchronize, or report the architecturally correct state; generated code
does not contain graphics API logic.

## 20. What is shared between Switch 1 and Switch 2

The following should be shared unless testing disproves the abstraction:

- Shared decoder-table machinery, with distinct A64, A32, and T32 tables.
- Canonical semantic primitives where the Arm execution states genuinely agree.
- Typed IR and verifier.
- Interpreter framework.
- AMD64 backend, register allocator, and host feature tiers.
- Code cache, W^X allocator, dispatcher, and link machinery.
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

These differences enter through `GuestCpuProfile`, the memory mapper, kernel
callbacks, and device implementations. They do not require separate AMD64
instruction emitters unless guest semantics genuinely differ.

## 22. Interpreter and tiering policy

The execution modes are:

1. Reference interpreter: always available, simple, instrumentable, and exact.
2. Baseline JIT: default execution engine and primary implementation target.
3. Optional optimized hot tier: considered only after profiling real workloads.

The interpreter and baseline JIT are mandatory. A hot tier is not assumed. If
added, it should compile only frequently executed units and use guards with
explicit side exits. Deoptimization metadata must reconstruct the active
canonical `ThreadCpuState` at every side exit.

Tier counters should be sampled or incremented cheaply; an atomic counter on
every block execution is not acceptable. Optimization must be disabled in
deterministic validation modes.

## 23. Fallback policy

Fallback is per instruction or block, not per title. When the lifter encounters
an instruction without JIT support it terminates the block and invokes the
interpreter for that instruction. Afterward execution returns to dispatch.

Fallback helpers declare:

- State they read and write.
- Whether they access memory.
- Whether they can raise an exception, schedule, or invalidate code.
- Memory ordering effects.
- Whether execution can resume inside a block.

Unknown instructions do not become no-ops. Profile-disabled or unallocated
encodings take the correct exception path.

Interpreter availability and IR-lifter availability are tracked independently.
An instruction may therefore be decoded and executed by the reference engine
before it can be lowered to IR. In that case translation ends immediately before
the instruction with an `InterpretOne` terminator carrying its location, raw
encoding, and stable coverage ID. The dispatcher validates those fields against
the live architectural state, executes exactly that instruction, and resumes at
the interpreter-produced PC. Exceptions and scheduler exits do not synthesize a
normal fallthrough.

`UnsupportedInstruction` is reserved for recognized encodings implemented by
neither engine. Its diagnostic contains the raw encoding, deterministic
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
reports decoder availability after execution-state and feature gating, plus
independently maintained reference-interpreter and IR-lifter availability. A
decoder entry therefore remains visible when a profile disables it or either
execution engine is incomplete.

`Lifted` is a completion claim, not merely evidence that a lifter match arm
exists. The generated row may use that state only after it has decoder
classification, reference semantics or an explicit architectural exception, IR
lowering, stable printer output, and a redistributable regression fixture. The
fixture registry is tested by decoding, lowering, verifying, and printing each
completed entry. An instruction added because of a workload report must add its
minimal encoding to that registry and retain a focused semantic test.

One `MissingInstructionTracker` belongs to one process or title scope. It
deduplicates recognized unsupported instructions by stable coverage ID and exact
raw encoding, retains the first PC, opaque runtime-assigned module identity,
execution state, and at most 32 bytes of local instruction context, and counts
total frequency independently from unique occurrences. Runtime integration
feeds `UnsupportedInstruction` terminators into this tracker.

The tracker supports both detailed and sanitized exports. Detailed reports are
the default and include the bounded byte window required for local debugging.
Sanitized reports contain only one raw instruction, guest addresses, opaque
numeric identities, and counters; selecting this policy also prevents the
tracker from retaining surrounding bytes. Neither mode accepts module paths,
title names, or arbitrary caller-provided strings. A report can be reduced to
`MissingInstructionFixture`, which carries only coverage ID, encoding, and
execution state for a regression test.

### 23.2 Diagnostics configuration ownership

Diagnostic detail is a runtime policy, not an intrinsic CPU-profile property
and not a debug-versus-release compile-time choice. Application configuration
loads a user-facing `diagnostics.report_detail` value. The runtime normalizes it
into one immutable `DiagnosticsPolicy` for the emulation session, and
`ProcessBuilder` retains that policy while constructing process resources.
`Detailed` is the default in every build profile; applications may explicitly
select `Sanitized`.

The complete runtime policy may later cover CPU missing-instruction reports, IR
dumps, AMD64 code dumps, GPU command diagnostics, and runtime event logs. It is
never passed wholesale into a subsystem. Instead, the runtime derives narrow
immutable views such as `CpuDiagnosticsConfig`; future backend and GPU crates
must receive equivalent subsystem-specific views. This preserves the dependency
direction: CPU code does not depend on the application configuration crate,
runtime types, graphics APIs, file paths, CLI behavior, or report destinations.

The CPU view currently selects whether missing-instruction collection is
enabled and whether its detail is `Detailed` or `Sanitized`. It also exposes
whether the runtime should fetch surrounding bytes, avoiding unnecessary guest
memory reads in sanitized mode. The tracker owns no output path and performs no
I/O. A later diagnostics sink may route structured reports to console, files, or
developer tools without changing decoder, lifter, interpreter, or backend APIs.
No mutable global diagnostics configuration is permitted.

## 24. Debugging and instrumentation

Required developer features are:

- Single-step interpreter mode.
- Block-level JIT stepping.
- A64, A32, and T32 disassembly plus pre/post-optimization IR and AMD64
  disassembly dumps.
- Translation reason and timing traces.
- Register and memory watchpoints.
- Per-op fallback counters.
- Block execution sampling.
- Code cache size, link hit rate, TLB hit rate, and invalidation metrics.
- CPU/GPU upload, download, stall, and dirty-range metrics.
- Deterministic event log sufficient to reproduce scheduler/device ordering.

Instrumentation is inserted through IR or block hooks selected before
translation. Production blocks contain no unconditional callback on every
instruction.

## 25. Validation strategy

### 25.1 Unit tests

Each semantic primitive, decoder family, IR operation, optimization, register
constraint, and encoder receives focused tests. Backend tests compare emitted
bytes and execute code where the test host permits it.

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
- Concurrent invalidation and direct-link patching.
- Exclusive monitors and atomic litmus tests.
- CPU/GPU dirty ownership transitions.

### 25.4 Memory-model tests

Litmus tests cover acquire/release, barriers, exclusives, self-modifying code,
and multicore visibility. Tests must run repeatedly with forced yields and on
different host architectures when an AArch64 backend exists; AMD64's stronger
ordering can otherwise hide missing barriers.

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
- Validate helper indices and patch targets.
- Keep debugging or writable code-cache views inaccessible to guest mappings.

A malformed executable should produce a controlled loader, memory, or guest
exception error, not host undefined behavior.

## 27. Performance policy

Performance work is evidence-driven. Benchmarks separately report:

- Decode/lift time per guest instruction.
- Optimization and backend time.
- Generated bytes per guest instruction.
- Cold-start and steady-state execution.
- Dispatcher and indirect-branch miss rates.
- Software-TLB hit/miss cost.
- Helper and interpreter fallback frequency.
- Code invalidation and cache rotation cost.
- Scheduler safepoint overhead.
- CPU/GPU synchronization bytes and wait time.

The baseline target is low enough translation overhead that code is profitable
after few executions, while producing code fast enough for games without a hot
tier. Numeric thresholds should be established from Swiitx microbenchmarks on a
documented host matrix rather than copied from unrelated runtimes.

Optimization order should normally be:

1. Remove frequent interpreter/helper fallbacks.
2. Improve memory and dispatch fast paths.
3. Eliminate redundant architectural-state traffic.
4. Improve common integer and SIMD lowering.
5. Improve block formation and linking.
6. Consider a hot tier only after the above are measured.

## 28. Major technical decisions

### D1: Use a specialized typed IR

Decision: accepted.

Justification: it provides a stable boundary between guest semantics and host
backends, supports an interpreter and future AArch64 host backend, and makes
exceptions, ordering, and instrumentation explicit. Direct instruction-to-
instruction translation is initially smaller but becomes difficult to maintain
for flags, SIMD, optimization, and multiple hosts.

### D2: Use a direct AMD64 baseline backend

Decision: accepted.

Justification: short blocks require low compile latency and emulator-specific
control over patching, state, fault sites, and memory fast paths. The backend's
scope is constrained by the IR.

### D3: Keep Cranelift optional

Decision: accepted.

Justification: Cranelift is a credible Rust backend and valuable comparison
point, but adopting it as the primary tier would still require custom machinery
for dispatch, code invalidation, fastmem faults, precise state maps, and block
links. It may later serve as a prototype, validation backend, or hot tier if a
measured experiment justifies it.

### D4: Do not integrate Dynarmic as the architectural core

Decision: provisionally accepted.

Justification: Dynarmic is the closest proven design and an important reference,
but direct integration would place the core CPU engine and substantial C++
infrastructure outside Swiitx's Rust architecture. Reconsider this if early
execution is prioritized over owning and researching the CPU implementation.
License compatibility and maintenance status must be re-evaluated at that time.

### D5: Implement software-TLB fastmem first

Decision: accepted.

Justification: it is portable, supports multiple address spaces and aliases,
and keeps faults debuggable. Fault-based fastmem remains an optional measured
optimization rather than an architectural dependency.

### D6: Keep graphics APIs out of generated CPU code

Decision: accepted.

Justification: the semantic boundary is guest memory and device events. This
preserves backend portability and centralizes CPU/GPU coherence.

### D7: Model one guest memory identity with adaptable host backing

Decision: accepted.

Justification: guest shared/unified memory semantics must work on both integrated
and discrete host GPUs. Canonical page identity plus ownership/version tracking
allows zero-copy, mirroring, or staging without changing the CPU engine.

### D8: Require an interpreter as a first-class engine

Decision: accepted.

Justification: differential testing and incremental instruction coverage are
essential for a maintainable JIT. The interpreter is not temporary scaffolding.

## 29. Implementation phases and exit criteria

### Phase 0: contracts and test harness

- Define `ProcessCpuContext`, `ThreadCpuState`, `VcpuExecutionState`, profiles,
  execution-state selection, memory access results, and exit reasons.
- Implement executable-memory abstraction and a minimal AMD64 call trampoline.
- Establish interpreter/JIT differential harness.

Exit: a generated block can enter and leave Rust safely on supported host ABIs.

### Phase 1: scalar interpreter

- Establish separate A64, A32, and T32 decoder skeletons behind one interface.
- Decode the core A64 integer, branch, load/store, and exception subset first,
  then add A32/T32 subsets according to redistributable tests and observed
  workloads.
- Map executable segments into the new memory service.
- Run small test programs through runtime syscall callbacks.

Exit: deterministic interpreter tests cover control flow and precise memory
faults.

### Phase 2: scalar baseline JIT

- Lift implemented scalar instructions from each enabled execution state to
  verified shared IR.
- Add linear-scan allocation and conservative AMD64 lowering.
- Add block lookup, bounded cache, and interpreter fallback.

Exit: randomized differential tests pass and JIT execution is faster than the
interpreter on representative scalar loops.

### Phase 3: memory and dispatch performance

- Inline software TLB.
- Add direct links and indirect inline caches.
- Add page-based invalidation and side-table exception recovery.

Exit: memory, alias, self-modifying code, and concurrent invalidation suites pass.

### Phase 4: FP, NEON, and atomics

- Expand A32/T32 scalar and interworking coverage required by Switch 1 titles.
- Complete execution-state-specific architectural FP control/status behavior.
- Add vector IR and tiered AMD64 SIMD lowering.
- Implement exclusive monitors, atomics, barriers, and multicore tests.

Exit: differential FP/SIMD suites and memory-model litmus suites pass.

### Phase 5: GPU coherency integration

- Connect guest page identity to CPU and GPU virtual memory managers.
- Implement dirty ownership, uploads/downloads, and fence visibility.
- Measure unified and discrete host strategies.

Exit: CPU-to-GPU and GPU-to-CPU end-to-end tests pass without unconditional
whole-memory copies.

### Phase 6: platform profiles

- Validate and enable Switch 1 profile behavior.
- Add separately sourced Switch 2 profile facts.
- Keep unknown behavior explicit and tested behind profile capabilities.

Exit: no product-name conditional exists inside generic decode, IR, or AMD64
lowering code.

## 30. Open questions

The following require prototypes or additional lawful research before becoming
decisions:

- Exact Switch 1 CPU feature profiles and visible system-register behavior.
- Whether verified Switch 2 native process metadata permits any execution state
  other than A64; the provisional native profile remains A64-only meanwhile.
- Exact Switch 2 architecture revision, instruction extensions, visible system
  registers, and feature-disabled encoding behavior.
- Whether Switch 2 compatibility uses native A32/T32 execution, CPU binary
  translation, pretranslated code, or another mechanism; no CPU frontend
  capability is derived from the public compatibility description alone.
- Guest page-table and address-space details exposed to the emulator runtime.
- Required fidelity of cache-maintenance operations for titles and system code.
- Whether software-TLB fastmem meets performance targets on Windows, Linux, and
  macOS before fault fastmem is justified.
- Best code cache ownership model under many simultaneously active guest threads.
- Host baseline SIMD requirement and whether a non-SSE4.1 tier is worthwhile.
- Granularity of CPU/GPU dirty tracking for buffers, textures, and aliased views.
- Required semantics for CPU/GPU concurrent atomics and accesses to shared
  device-visible memory for each platform profile.
- Feasibility and benefit of imported host memory on each graphics backend.
- Whether an optimized hot tier materially improves games after GPU and memory
  bottlenecks are addressed.
- Requirements for save states, replay, and debugger integration that affect
  architectural-state versioning.

Open questions must be resolved with a short decision record containing evidence,
alternatives, benchmark/test method, and compatibility impact.

## 31. References

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
  — atomic block patching, software-TLB hot paths, cross-vCPU invalidation, and
  memory consistency.
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
