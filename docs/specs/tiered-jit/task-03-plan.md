# Task 3 implementation plan

Status: Task 3 complete; steps 1–7 validated. Demanded
integer/control, register-only SIMD, scalar FP comparisons/rounding and inline
system-register operations execute through the new native ABI and cache in
focused tests. Canonical FP system boundaries, exact comparisons/rounding,
directional/fixed FP-to-integer conversions and runtime system helpers are
connected. Scalar FADD/FSUB, FDIV, FMUL/FNMUL, fused multiply-add, FSQRT and
precision conversions S↔D, SCVTF/UCVTF and FCVTZS/FCVTZU activate native FP
lazily. Advanced SIMD SCVTF/UCVTF (packed and scalar-vector forms) and vector
FDIV and FMUL-by-element also use native activation. Non-memory lowering and
local closure checks are complete; the pinned-fork handoff is deferred by
maintainer decision and does not block step 3 closure.
Tasks 1–2 supply the native
contracts, fork output and publication/lifetime foundation. Production JIT
execution now uses LCQ; `direct` has been removed. External edges return to the
canonical resolver, with no native linking or functional HCQ promotion yet.
Available validation is native x86-64 and
AArch64 QEMU 11.1.1; native Arm hardware validation remains pending.

**Dependency handoff:** the observable FP blocker is fixed in the local
Wasmtime `nixe` branch, with maintainer approval. LCQ sets
`Function::nixe_observable_fp`: arithmetic, comparisons, rounding and
conversions retain ordered FP environment effects even for discarded results.
The egraph cannot fold/merge those operations or look through them from pure
consumers; machine lowering cannot discard or absorb them into value-only
patterns. Integer and bitwise optimizations remain enabled. No per-operation
stores, extra NativeFrame state or forced helpers were added.

Keep the local fork override for subsequent work, as agreed with the
maintainer, in case further Cranelift changes are needed. Updating the pin is
deferred, not a step 3 exit requirement. When retiring the override, publish
the required fork changes, advance Nixe's dependency revision/lockfile together
and rerun validation without it. The current pin does **not** contain this API.
Local tests use `--config /tmp/nixe-observable-fp-local.toml` (temporary Cargo
patches for the six direct fork dependencies); do not commit local paths or
the resulting path-resolved lockfile. Normal pinned builds require the new
fork commit; that handoff does not block further local implementation.

This is a working checklist for
[Task 3](spec.md#task-3-cut-synchronous-lcq-over-as-one-vertical-slice),
not another specification. Follow [CONTRIBUTING.md](../../../CONTRIBUTING.md).
Update progress and concrete decisions here in place, without session logs or
a separate decision register. Agree architectural changes with the maintainer
and update the affected spec section; keep implementation details in code.
This plan may be removed when the task is complete.

## Scope

Connect demanded guest instruction capture and synchronous LCQ compilation to
the real NativeFrame ABI, executable cache, protected dispatch and runtime.
Preserve currently supported guest instruction semantics while replacing their
old execution boundary. Reuse the decoder, normalized instructions, analysis,
typed helper semantics and FP owner; do not introduce another semantic IR.

The completed task has one callable JIT path and no old HCQ promotion/workers.
External edges use source-local canonical fallbacks until Task 4 implements
native links, PICs and the RSB. Functional sampling/background admission remain
disabled until Task 5; new HCQ construction remains Task 6. An LCQ-only interval
is intentional, not a reason to retain the old optimizer as a fallback.

No benchmark runner, homebrew suite, testing framework, runtime tuners or
compatibility executor is part of this task. Test each step as it becomes
executable; do not route homebrews to an incomplete implementation.

## Steps

- [x] **Inspect the production cutover boundaries.** Trace runtime dispatch,
  current instruction coverage, helpers/SVCs, FP-mode changes, exclusive state,
  control requests and progress reporting. Inspect instruction fetch, executable
  snapshots, invalidation notifications and mapping/write synchronization in
  the memory authority. Determine which interfaces already satisfy the required
  ordering and which must change; a cursor check alone is not a coherent byte
  snapshot or a pre-mutation rendezvous. Identify the signal/landing machinery
  reusable without the old JIT fault registry. Confirm supported process-wide
  memory backend selection without adding per-access fallback.

  Record the concrete ownership/API decisions and any real blocker below before
  dependent edits. Use existing semantic tests as the coverage baseline, not a
  new generated inventory or measurement tool.

  **Exit:** the route from demanded PC to canonical runtime return and the
  ordering of capture, mutation and fault retry are explicit. Replacement
  boundaries are identified without changing runtime behavior in this step.

- [x] **Implement demanded capture and exact-key compile ownership.** Capture
  one coherent owned instruction image with exact physical/mapping dependencies
  and its publication cursor. Decode from the demanded BlockKey only, stopping
  at the first specified terminator/architectural boundary or 512 instructions.
  Do not fetch a successor, scan a function or discover a region. Preserve
  precise fetch-failure attribution; an indirect entry into an existing
  fragment may create another independently indexed fragment.

  Implement the memory-side synchronization needed to make that capture
  coherent in this step, including concurrent writers through physical aliases.
  Step 5 connects publication/invalidation to the JIT coordinator; it must not
  be used to defer the snapshot's own correctness.

  Add the specified same-key/current-generation compile claim to the reusable
  cold dispatch state. The winner uses vCPU-local compiler state; same-key
  contenders may wait only after leaving their reader epoch and restoring FP.
  Do not hold process, cache or memory locks during compilation or waiting.
  Closure, failure and shutdown wake waiters; stale completion releases only
  its exact claim. Reserve the emission identity before version-bearing native
  emission and carry its admission epoch through publication. Step 5's tracking
  rendezvous may require a new claim after capture, before identity reservation;
  it never renews already-emitted code.

  **Exit:** tests cover terminators, the emergency cut, page boundaries, overlap,
  stale capture and cancellation. Same-key contention has one winner per claim
  generation, while different-key compilers can make progress concurrently.
  Claims and empty dispatch slots do not accumulate after abandoned work.

- [x] **Lower LCQ into the real native ABI and cache.** Port the existing
  instruction lowering to one demanded fragment, with `opt_level=none` and
  `single_pass`. Use shared effects/liveness and the fork's entry/exit/fault
  boundaries; obtain physical bindings, labels and spill extent from final
  allocation. Keep lazy NZCV as shared SSA recipes until its consumers require
  materialization. Never assume the backend leaves a persistent host flag
  producer available. Polls and address confinement must preserve live flags.

  Emit canonical ingress and source-local exits with exact dirty/live state,
  destination PC, edge kind and the reserved CodeVersion. Consume owned output
  once, relocate at final RX addresses and publish complete CodeUnits through
  Task 2. Supported integer/SIMD operations remain native; architectural helpers
  have typed effects and correct FP suspension/completion. Unsupported behavior
  is an attributed failure, never interpreter or old-JIT fallback. Complete
  memory/fault lowering in step 4 before production cutover.

  **Exit:** non-memory guest fragments execute through protected lookup and the
  gateway with correct state, branches, flags and FP. Final maps agree with the
  emitted exits; stale output never becomes reachable. No JITModule or legacy
  NativeContext is required by these fragments.

- [x] **Connect direct memory and real fault dispatch.** Lower supported scalar,
  SIMD, pair, atomic and exclusive accesses with their existing architectural
  semantics. Normal RAM uses the pinned arena, required confinement and native
  operations, without generated permission/backing lookups or eager canonical
  checkpoints. Account for compound-access ordering, subaccesses and commit
  stages; preserve unaffected state on every fault.

  Connect WorkerFaultContext capture/landing support to the fixed native-PC
  directory and the active Invocation epoch. Keep that protection through
  normal-stack resolution and retry, including flags, FP and physical spills.
  A recoverable RAM fault resumes the identical native instruction; a valid
  non-RAM access performs one typed cold operation; invalid guest access reports
  its precise architectural fault. Unattributed/nested/impossible faults and
  repeated unchanged tracking faults fail precisely. Do not reuse an earlier
  instruction checkpoint or keep the old JIT registry as a second lookup path.

  **Exit:** delivered-fault tests on both available execution hosts cover retry,
  nonretry reconstruction and compound-access side effects. Ordinary RAM has
  the specified native shape. Preserve the interpreter's distinct fixed-stub
  use of shared fault support. The new path must not depend on the legacy JIT
  executor or its registry; production cutover and legacy removal are step 6.

  **Progress:** the local fork now exports `StateMap::fault_bytes`, the exact
  length of each faulting native instruction (including individual accesses
  within compound atomics). x86-64 records the end of each assembler instruction;
  AArch64 records its fixed four-byte extent. Owned output preserves that field,
  and CodeUnit publication rejects semantic intervals that disagree with it.
  This supplies the exact intervals required by the native-PC directory without
  runtime decoding or eager guest-state stores.

  `lcq::invocation::run` now owns mapping-lease acquisition, admission and the
  WorkerFaultContext callback against the live directory. Retry preserves the
  captured image and frame; escape reconstructs state, prepares an owned typed
  completion or precise fault (with captured instruction identity), reconciles
  the captured poll counter and hands off the exclusive monitor before dropping
  the epoch and lease. MMIO completion runs only after return. Three focused
  tests cover misses, tracked-store retry, normal/fault exclusive handoff,
  unmapped faults, MMIO execution after releasing code owners, mapping mutation
  after escape, and pair completion without rereading its native RAM prefix.
  Normal exits now copy GuestExit and its captured instruction before quiescence,
  validating the admitted unit/version and exit-map index. The same directory
  indexes one span per unit, including faultless units, and searches the unit's
  existing sorted fault records for memory attribution. No snapshot ownership,
  extra registry or guest-memory refetch is needed. LCQ is still unlinked;
  canonical exit attribution requires the source version to match the admitted
  entry, and future native linking must identify the actual exiting unit.
  Tests cover faultless BRK/SVC/branch/invalid exits after code teardown and guest
  byte replacement, invalid exit identities, and metadata retention/detachment
  across span reuse.
  `MemoryExit::complete` now consumes typed cold operations and canonical
  exclusive-store exits after protection release. Guest aborts use the existing
  CpuExit::DataFault contract; internal memory errors use CpuFault with captured
  instruction identity, register context and the caller's earned progress.
  Tests cover a reservation passed between two native invocations (successful,
  changed-value, permission-denied and empty-monitor STXR), permission revocation
  before MMIO completion, retained pair reads, and internal diagnostics without
  live instruction refetch or device execution. Rejected Closing/Closed
  admissions also release FP ownership, epochs and leases before waits/mutation.
  Terminal capture failures now report the reason, native PC and fault address
  through a fixed stack buffer and a best-effort write, without allocation,
  stderr locks or metadata lookup. Closed stderr pipes cannot replace the
  original signal with SIGPIPE. Twelve shared subprocess cases retain fatal
  signal/handler-chain behavior, including unchanged retry, nested dispatch,
  missing attribution and caught panic; an additional real LCQ dispatcher case
  rejects an in-arena access from an unpublished PC before consulting memory.
  The full x86 JIT suite passes 394 library plus 4 integration tests; shared
  fault support passes 19 library plus 1 integration test. Both Arm QEMU
  profiles pass eleven ordinary invocation tests; `max` also passes all twelve
  shared fatal scenarios and the real LCQ rejection by direct invocation
  (bypassing old system binfmt), plus 18 ordinary shared tests. The preceding
  directory increment passed all 37 publication/lifetime tests on `max`.
  Clippy is clean on both targets. The step-4 exit audit is complete: native
  accesses, exact fault maps, protected retry/reconstruction, owned cold exits,
  monitor handoff and terminal failure paths are connected and tested.
  Mapping invalidation/rendezvous integration is step 5; production execution,
  slice/control accounting and legacy removal are step 6. The legacy `direct`
  path remains active; native Arm hardware validation is still pending.

  Scalar literal, unsigned, unscaled, pre/post-indexed and register-offset
  loads/stores now lower into LCQ and publish their physical prefault maps and
  semantic records through the existing CodeUnit/native-PC directory. Addressing
  and load-extension semantics are shared with the legacy compiler; its replaced
  scalar implementations have been removed. `Compiler::for_arena` fixes the
  process arena size at compiler creation. An unsigned confinement select sends
  out-of-range addresses to the trailing guard, retaining PRE guest state for
  later reconstruction of the original access. The fork's `nixe_arena_addr`
  adds the offset directly to r13/x19 in one flag-preserving instruction, without
  a context load or base-register copy. Loads and base writeback become dirty
  only after the access succeeds; snapshots retain earlier dirty state, lazy
  NZCV and pending native FP effects. Exits and faults share snapshot allocation.
  A failed lowering discards its unfinished frontend scratch so the vCPU compiler
  remains reusable; successful compilations keep reusing their allocations.

  Single-register SIMD/FP B/H/S/D/Q loads/stores now use the same direct access
  and prefault publication path for unsigned, unscaled, pre/post-indexed and
  register-offset addressing. Q emits one vector128 access; narrower loads zero
  the upper vector bits. These bit transfers do not activate FP or change its
  status. Addressing and value conversions are shared with the legacy compiler,
  whose superseded single-transfer lowering has been removed. Fault snapshots
  preserve earlier vector values and defer destination/base updates until success.

  Scalar W/X/LDPSW and SIMD S/D/Q pairs now lower to two ordered, independently
  confined accesses. Each native fault record identifies its subaccess. Pair
  loads keep PRE destinations/base until both reads succeed, with the first
  result retained as a physical fault-map operand at the second read (not as
  an architectural register update). Cold completion can use those bits without
  replay; a guest fault retains the PRE registers. Stores mark the second access
  with commit stage 1, preserving the already-visible first store. Base writeback
  follows both accesses. Shared pair addressing replaces the legacy copies.
  LDNP/STNP's previously ignored signed offset is fixed in both JITs and the
  interpreter, with a reference-engine test of explicit expected addresses.

  LDAR[B/H]/STLR[B/H] (1/2/4/8 bytes) now use CLIF atomic loads/stores and the
  same direct arena/fault path. RCsc ordering is preserved as native LDAR/STLR
  on AArch64 and MOV loads / MOV+MFENCE stores on x86-64. The fence is outside
  the exact fault interval. The legacy x86 release path now uses atomic_store
  too; its plain store did not enforce STLR-to-LDAR ordering. Shared scalar
  transfer decoding replaces its separate acquire/release lowering.
  Ordered accesses enforce natural alignment before touching RAM on both hosts,
  redirecting misaligned addresses to the guard. Do not rely on host AArch64
  alignment traps (SCTLR.nAA can relax them). The resolver reconstructs
  the original address and reports misalignment before attempting mapping repair;
  this deliberately faulting guard access is not retryable. Original-address
  inspection, alignment-fault construction and connection to memory policy are
  implemented and produce structured memory exits.

  Single-structure LD1-4/ST1-4 lane transfers and LD1R-4R replication now use
  native 1/2/4/8-byte accesses, including immediate/register post-index and
  modulo-32 vector lists. Lane loads preserve all other bits; 64-bit replication
  clears the upper half. Shared lane/replication/writeback helpers replace the
  legacy copies. Each successful element updates its destination before the
  next snapshot; both loads and stores record the completed prefix as their
  commit stage. The base is written only after the complete instruction. These
  transfers neither activate FP nor require an uncommitted pair-read temporary.

  Interleaved LD2/3/4 and ST2/3/4 now use ordered lane-sized native accesses,
  with one prefault map per element and deferred base writeback. Shared value
  helpers replace the legacy interleaved conversions. In 64-bit loads the upper
  half clears on each destination's first successful read, not at instruction
  completion; the interpreter and legacy JIT are corrected accordingly. Shared
  shape decoding rejects the reserved interleaved .1D arrangement.

  The shared WorkerFaultContext now supports registry-free capture. Its
  normal-stack dispatcher can query the new native-PC directory through a
  FaultLookup borrowed from the active Invocation, separately from native frame
  mutation. No legacy fault-site copy or second JIT registry is installed.
  The landing leaf restores the owner's saved caller FP before Rust; retry
  restores the untouched captured guest FP/register state without changing FP
  ownership or committing FPSR. A repeated same-PC/same-byte fault after a
  claimed repair fails before calling policy again. Fixed interpreter stubs
  retain their existing attribution and dispatch path.

  After escape, the worker now lends the captured machine image without allowing
  slot reuse. Cold reconstruction reads architectural host GPR/vector numbers,
  validated frame spills, constants and lazy NZCV recipes into canonical guest
  state. It retains the uncommitted first pair read for cold completion, leaves
  PC at the failing instruction and returns the captured poll balance for the
  execution owner's progress accounting. Captured host FPSR uses the same
  translation as live FP completion and merges after software FPSR writeback.
  Finishing ownership prevents a second merge from the dispatcher's host state.
  The routine rejects mismatched PC/frame/FP ownership and double reconstruction.

  Before repair, read-only inspection decodes the retained instruction image
  and reads its address operands from the physical PRE map or canonical clean
  state. It recovers the original scalar/vector/pair/structure subaccess, checks
  its size/kind/commit stage against published metadata and verifies that the
  Linux fault byte belongs to its confined native extent. It does not refetch
  guest code, add physical map operands, canonicalize state or merge FPSR.
  Ordered misalignment is distinguished before range overflow; both produce
  structured guest faults instead of requesting a retry of the guard.

  LCQ resolution now calls CpuMemory's authority while retaining the memory
  lease and code epoch. ExecutionMemory validates the full original subaccess,
  including second-page permissions, before repairing RAM. Each touched backing
  is reconciled outside the mapping lock; retry requires the resulting host
  protections to be published for every page. Valid MMIO returns Cold without
  calling its handler. Ordered alignment/overflow faults bypass repair, and an
  authority requesting a retry of a confined guard access is rejected.
  The old executor still cannot complete fault-driven MMIO: it now reports that
  missing continuation explicitly instead of calling valid MMIO a guest mapping
  fault. Fixed interpreter stubs retain their page-local eligibility contract.

  Single-access cold completion now handles scalar W/X/literal and SIMD
  B/H/S/D/Q transfers, including LDAR/STLR, addressing and deferred writeback.
  After reconstruction, preparation captures the typed access, store bits,
  destination and writeback from the published instruction/canonical state.
  It retains no code, frame or metadata references. The owner releases its
  invocation and memory lease before consuming this value in ordinary Rust.
  Completion performs one typed operation; only success commits destination,
  base and next PC. It preserves sign/zero extension, XZR, vector upper-bit
  clearing and acquire/release ordering. Device errors are returned without
  retry or CPU-state writeback. Memory permissions are checked again by the
  typed provider, so changes after leaving native execution are respected.
  The same completion owner now supports scalar W/X/LDPSW and SIMD S/D/Q pairs.
  It starts at the published subaccess: a second load consumes the retained
  first result, while a second store never repeats the committed first store.
  Typed reads finish before either destination is committed; earlier stores
  survive a later error. Base writeback and PC advance only on full success.
  This preserves signed/non-temporal/pre/post offsets, SP/XZR/V31 and permitted
  base/destination aliases. Preparation rejects missing retained pair-read bits.
  Single-structure lane/replicate and interleaved LD2/3/4–ST2/3/4 cold completion
  also starts at the named subaccess, preserving the reconstructed prefix.
  Each successful typed load immediately updates its lane or replicated vector;
  64-bit interleaved loads clear the upper half on the first successful element.
  Later failures retain these updates and earlier stores/device effects, but
  suppress base writeback and PC advance. Store sources use canonical vectors;
  no native references or duplicate register-state image is retained.

  Contiguous LD1/ST1 .1D (one to four registers) now uses the same publication,
  fault and cold-completion path, with one 64-bit native access per register.
  Each load replaces its destination and clears the upper half without reading
  the old vector. Shared multiple-structure element indexing preserves register
  repetitions as well as interleaving. Immediate/register post-index, SP and
  wrapping register lists retain the existing contracts.

  Scalar CAS[B/H]/CASA/CASL/CASAL now lowers to a native atomic transaction,
  with natural-alignment confinement, PRE destination/NZCV maps and no eager
  checkpoints. Native ordering is conservatively RCsc; cold completion retains
  the instruction's acquire/release descriptor and performs one typed CAS.
  The resolver validates both read and write permissions before repairing RAM;
  unsupported device atomics never become separate MMIO reads/writes.
  x86 and AArch64 LSE use one fault site. The non-LSE Arm loop has a load and
  two alternative exclusive-store sites, all naming the same uncommitted guest
  operation. A local fork fix writes back the observed value on comparison
  mismatch in Nixe mode, preserving CAS write-permission faults on read-only
  mappings. Its I32 comparison also ignores undefined upper operand bits.

  Scalar LSE RMW (LDADD/LDCLR/LDEOR/LDSET, signed/unsigned min/max and SWP,
  including byte/halfword and store aliases) now uses the same native atomic
  fault/resolution and typed cold-completion path. Address/source aliases are
  read before committing the zero-extended old value; XZR discards only the
  result, not the transaction. Native ordering remains conservatively RCsc;
  cold completion retains release and acquire (suppressed for Rt=XZR).
  The shared CLIF operation mapping also serves the legacy lowering.
  RMW loop load/store sites share one PRE map with no committed subaccess.
  The local fork fixes narrow unsigned min/max comparisons in non-LSE loops
  to ignore undefined upper operand bits and adds the missing LSE SWP rule.

  CASP W now packs both register pairs into one 64-bit native CAS, using the
  existing confinement, PRE maps, permission resolver and typed cold completion.
  Both W results zero-extend only after the transaction completes; faults keep
  both PRE registers and lazy NZCV. SP, WZR and base/source/destination aliases
  are preserved. Pair packing/splitting is shared with the legacy compiler.
  CASP X now uses a native 128-bit CAS: CMPXCHG16B on x86, or the new fork
  lowering to CASPAL / LDAXP-STLXP on Arm. The non-LSE loop validates even
  mismatch by exclusively writing the observed pair back; LDAXP alone cannot
  supply an atomic 128-bit observation. All three loop fault sites share PRE
  state, and both destinations commit together. The same mismatch fix applies
  to canonical memory's Arm 128-bit CAS/load helper, keeping interpreter/cold
  atomics coherent with native execution. Missing x86 CMPXCHG16B is reported
  explicitly when compiling CASP X. No split transaction or RAM helper is
  generated by LCQ.

  Canonical and synthetic memory expose `resolve_exclusive_load` for the
  native-exclusive exit handoff: resolve the physical page/offset from
  the completed load's virtual address while its execution lease still holds,
  retaining its exact observed bits without rereading RAM, repairing visibility
  or calling a device. Do not defer identity resolution until a later STXR,
  when mappings may have changed. Scalar LDXR/LDAXR (B/H/W/X) now use a native
  atomic load, natural-alignment confinement and a NativeFrame pending record
  written only after success, including ZR destinations. No physical lookup or
  RAM helper is generated. `finish_exclusive_load` transfers the last successful
  load to the persistent thread monitor before releasing the memory lease on
  normal/nonretry exits; a retry retains the same frame. A failed load preserves
  the preceding reservation. Cold completion consumes the persistent monitor
  and replaces it only after a successful typed exclusive load. Handoff must
  precede cold/system completion, so CLREX cannot be undone by a delayed record.
  The protected invocation owner connects this handoff; production cutover is
  step 6.
  LDXP/LDAXP W now share the same native 64-bit atomic read and pending record:
  the low/high words commit together, zero-extended, after success. The
  reservation keeps all eight observed bytes, including discarded destinations.
  Identical destination registers are rejected as constrained-unpredictable.
  Cold completion likewise performs one typed exclusive load before committing
  either destination and the persistent monitor; it executes no MMIO callback.
  LDXP/LDAXP X now use two separately atomic 64-bit reads, as Arm permits,
  without requiring a coherent 128-bit snapshot or a CAS. The first site
  enforces 16-byte alignment; each site has its own eight-byte fault record.
  Both destinations stay PRE until success, matching the canonical provider's
  whole-access fault behavior: an aligned pair is in one page, and the execution
  lease prevents mapping/visibility changes between its reads. The second map
  retains the first value; the pending monitor records both halves only after
  success. Cold at the first site uses a typed 16-byte exclusive load; Cold at
  the second site violates this same-page/lease contract and is reported as an
  internal error, never handled by repeating the first read. Normal read-only
  RAM remains native and needs no write permission or CMPXCHG16B.

  Before connecting native exclusive stores, the canonical `store_exclusive`
  path now retains the exact backing whose physical identity matched the
  reservation. It no longer releases the mapping lock and resolves the virtual
  address again for CAS, which could select a replacement page during cold
  execution. Ordinary atomics reuse the same retained-backing transaction;
  synthetic exclusive stores keep their mapping lock through the transaction.
  A deterministic test replaces the mapping between physical selection and CAS
  for all five widths and checks that only the selected original page changes.
  The CPU suite passes 81 library and 17 integration tests; its ten focused
  exclusive tests pass under both Arm QEMU profiles.

  Scalar STXR/STLXR (B/H/W/X) now use native CAS when address and width match
  a successful load in the same invocation. The memory lease protects that
  physical identity; no page lookup or RAM helper is generated. Before CAS,
  the pending record is marked consumed: native retry retains its operands,
  subsequent stores fail without memory access, and normal/nonretry handoff
  clears the persistent monitor. Faults leave the status register PRE; a new
  successful load replaces the consumed record. Incoming reservations, width
  mismatches and different virtual aliases leave through a typed PRE exit:
  handoff resolves the load's physical identity before lease release, then
  completion consumes the persistent reservation and calls `store_exclusive`.
  It neither mistakes virtual aliases for mismatches nor follows a remapped
  address with the old reservation. Same-VA native CAS cannot legitimately
  require Cold after its native RAM load under the same lease; such a provider
  result is an internal error. Overlapping status/source or status/base
  registers are rejected as constrained-unpredictable.

  STXP/STLXP W and X now share that path, using one CAS64 or CAS128 for the
  whole pair. Sources are concatenated little-endian after W truncation;
  identical sources and source/base aliases are valid, but status overlapping
  either source is rejected. X pairs retain both expected halves and require
  16-byte alignment; no split store can leave a partially replaced pair. The
  typed physical exit also retains both source registers. x86 X pairs require
  CMPXCHG16B explicitly; Arm uses the existing LSE or validated LL/SC backend.
  No new fork changes are required. The protected invocation owner performs
  dispatcher/monitor handoff before typed completion.
  The 31 focused exclusive tests pass on x86 and both Arm QEMU profiles,
  including scalar/pair tracking fault/retry, PRE status on write faults,
  comparison of both pair halves, repeated-store failure, replacement by a
  new load, incoming monitors, remapped aliases and MMIO rejection without callbacks.
  The full x86 JIT suite passes 380 library and four integration tests; Clippy
  passes on both targets and formatting is clean. No fork changes were needed
  for this exclusive-store checkpoint; the existing local override is retained.

  Delivered LCQ tests exercise authority-driven RAM repair, guest faults and
  Cold classification as well as manual retry/nonretry reconstruction fixtures.
  The invocation owner connects dispatch, monitor handoff and guest-fault
  reporting; runtime slices/progress accounting are connected in step 6 below.

  Contiguous LD1/ST1 now cover every arrangement and all four register-list
  lengths. One page-offset guard selects a grouped same-page path: one native
  8/16-byte access per register, with no RAM helper or permission lookup.
  Cross-page lists use the shared element loop. Grouped fault records name
  the first element of each register and its PRE destination; reconstruction
  validates the single-page condition. Cold completion always uses the
  original element size and never repeats the committed prefix. Both paths
  join before base writeback. Shared liveness retains old destinations for
  partial loads, including first-element upper-half clearing in 64-bit vectors.
  The legacy JIT's unconditional grouping is not used by this implementation.
  Validation passes: 383 JIT library and four integration tests on x86;
  AArch64 QEMU `max` passes the complete RAM arrangement/addressing matrix,
  mixed RAM/MMIO prefix completion and exact-subaccess retry. Both Arm QEMU
  profiles pass grouped cold completion and physical-map checks; `cortex-a53`
  also passes grouped PRE/partial-vector abort reconstruction. Clippy passes
  for both targets, formatting is clean, and this checkpoint needs no fork changes.

  **Validation:** the preceding fork increment passed 244 codegen and 44 reader
  tests. Extent regressions independently decode scalar/vector loads and
  stores with different displacements, and compound atomic accesses, for both
  encoders and allocators. New LCQ tests compare scalar addressing, signed/unsigned
  extensions, SP/XZR, unaligned and cross-page RAM accesses with the interpreter,
  including lazy flags consumed after memory, using real guarded DirectArena
  mappings and canonical backing pages. SIMD tests cover B/H/S/D/Q, V31, upper-bit
  clearing and unchanged FPCR/FPSR, including NaN bit patterns. Vector map tests
  verify PRE-writeback state, earlier dirty vectors and one native access without
  I128 temporaries or FP activation. Scalar map tests retain earlier FP effects;
  CLIF checks exclude eager checkpoints. Pair tests cover every transfer width,
  addressing mode, signed offset, SP/XZR/V31 and permitted base aliases; maps
  retain the first read and PRE destinations under register/spill pressure and
  distinguish the second store's commit stage. Publication rejects malformed
  extents and retained-read locations/stages. Ordered-access tests compare all
  widths with the interpreter on aligned RAM and verify native LDAR/STLR or
  MOV+MFENCE, exact fault intervals, lazy flags and generated alignment confinement.
  Delivered LCQ alignment-fault tests verify PRE address recovery and structured
  alignment errors on both hosts, reported through structured memory exits.
  Single-structure comparisons cover every element width and register count,
  low/high lanes, both replication widths, SP, wrapping vector lists, post-index
  forms and unaligned/cross-page RAM. Map tests check the completed prefix and
  deferred base writeback without eager checkpoints or I128 temporaries.
  Interleaved comparisons cover all allocated arrangements, post-index forms,
  SP, wrapping lists, base/offset aliases and unaligned/cross-page RAM. Map tests
  cover every element boundary through the maximum 64 accesses. Explicit
  interpreter fault expectations check partial lane values and upper-bit
  clearing; legacy JIT tests check partial loads/stores and suppressed writeback.
  Delivered-fault tests cover scalar loads/stores, Q loads, second pair accesses
  and interleaved completed prefixes. Retry matches no-fault execution, including
  dirty spills, lazy carry, guest rounding and pending native FPSR. The dispatcher
  verifies that it observes the saved caller FP environment, not guest controls.
  Shared-runtime subprocess tests distinguish fatal repeated faults from a
  second policy call. Nonretry tests compare the reconstructed state and partial
  stores with the interpreter, including paired retained reads, deferred base
  writeback, interleaved lanes, 32/64-bit carry/conditional/packed NZCV, spills
  and pending FPSR. The shared capture tests check bounded Arm FPSIMD record
  parsing and image invalidation when starting another batch. These checks
  complement the authority-driven and owned-memory-exit tests below.
  Address inspection tests cover unsigned/signed pre/post/literal/register
  addressing, UXTW/SXTW/LSL/SXTX, SP/XZR, dirty bases, scalar/Q page crossings,
  guard crossings, confined starts, wrapping arithmetic and access overflow.
  Ordered tests cover load/store widths and misalignment on mapped, protected
  and out-of-range addresses. Existing pair/structure retry and escape tests
  also inspect their subaccess before dispatch. Inspection uses the published
  image even when fixture arena bytes at guest PC differ from that image.
  Delivered ExecutionMemory tests cover tracked stores (including two-page
  repair in one dispatch), two-page GPU-newer read reconciliation, unmapped and
  permission-invalid accesses, alignment guards and MMIO read/write classification
  without invoking handlers. A denied second page leaves the first page armed.
  The legacy raw-MMIO regression checks its explicit missing-continuation error.
  Three cold-completion tests deliver real MMIO faults, release all native
  owners, and compare complete state and exact typed device events with the
  interpreter. They cover scalar widths/sign extension, W/X/XZR, SIMD widths/V31,
  literal/register/pre/post addressing, SP, LDAR/STLR, device rejection and
  incorrect result widths. A permission change between preparation and completion
  fails without calling the device. Preparation before reconstruction is rejected.
  Two pair cold-completion tests compare scalar/vector formats and first/second
  device failures against the interpreter. A RAM-first/MMIO-second test changes
  the first RAM bytes after escape: loads retain the captured value, stores do
  not repeat their first write, and a failed second access preserves PRE CPU
  registers. Both native and cold store prefixes remain externally visible.
  Two structure cold-completion tests compare lane, replicate and interleaved
  formats against the interpreter, including every failure stage through access
  64, upper-bit clearing, wrapping register lists and post-index aliases.
  RAM-to-MMIO cases overwrite the native prefix after escape and verify that
  completion never replays it, including when a later cold access fails.
  Contiguous .1D tests cover all four list lengths, addressing forms, unaligned
  and cross-page RAM, physical maps with one 8-byte fault site per register,
  and each cold failure stage. The mixed RAM/MMIO prefix test includes .1D.
  An independent interpreter regression checks explicit byte/register results
  at every abort boundary for all contiguous arrangements and list lengths.
  All four focused JIT structure tests and that interpreter regression pass
  under AArch64 QEMU 11.1.1. Clippy passes for CPU, interpreter and JIT on both
  targets; formatting and diff checks pass.
  CAS tests cover all scalar widths/orderings, success/mismatch, operand aliases,
  SP/XZR, lazy NZCV, alignment confinement, retry and canonical escape. Cold
  tests revalidate RAM remapping/permissions after releasing native owners and
  reject device atomics without calling MMIO handlers. The authority test also
  checks tracked-RAM CAS and comparison mismatch on guest read-only memory.
  Both LSE and non-LSE fault maps are checked; the latter records three native
  sites for one guest transaction. The fork passes 245 codegen tests, including
  independent instruction decoding of all three sites with both allocators.
  Six focused JIT tests pass under QEMU with `-cpu max` (LSE) and
  `-cpu cortex-a53` (LL/SC), including write-permission checks on mismatch.
  RMW tests cover all nine operations, widths/orderings, signed/unsigned edge
  values, SP/XZR and base/source/destination aliases with lazy flags. Exact
  native fault counts are checked on x86 and Arm with/without LSE. Delivered
  faults cover dirty PRE state, natural alignment, retry/escape, tracked-RAM
  repair and write-permission errors even when min/max leaves memory unchanged.
  Cold tests revalidate remapped RAM and permissions and reject device atomics
  without invoking MMIO handlers. The RMW fork increment passes 246 codegen
  tests, including independently decoded narrow comparisons and SWP with both
  allocators.
  Five focused RMW/authority tests pass under QEMU 11.1.1 with both `-cpu max`
  and `-cpu cortex-a53`; JIT Clippy passes on both targets. Formatting and diff
  checks pass. The local override is still required for the fork changes.
  Seven focused CASP/authority tests pass on x86 and under both Arm QEMU
  profiles, including the legacy CASP regressions. They cover full-pair match
  and mismatch, all orderings, aliases/WZR, dirty PRE registers, alignment,
  RAM retry and read-only faults, plus cold remapping/device rejection. Maps
  contain one x86/LSE CAS or three non-LSE sites for the same uncommitted pair;
  CLIF has no helper, canonical store or I128 temporary. No fork changes were
  needed for CASP W; JIT Clippy passes on both targets.
  CASP coverage now includes X pairs, full-width comparisons/results, exact
  16-byte fault records, 16-byte alignment (including an 8-byte-aligned fault),
  dirty PRE retry/escape, tracked stores and failed-comparison write faults.
  Cold tests cover both pair widths with RAM remapping and device rejection.
  The fork adds a 128-bit CAS regression under register pressure with both
  allocators and LSE settings; its ordinary instruction representation stays
  32 bytes. Canonical memory has a concurrent 128-bit load/mismatch regression.
  The fork passes 247 codegen tests. Nine focused JIT tests and two canonical
  atomic tests pass under QEMU with both `-cpu max` and `-cpu cortex-a53`.
  The x86 memory suite passes 48 library plus one integration test; JIT/memory
  Clippy passes on both targets. These CASP X changes require the local fork
  override until its revision is committed, pushed and pinned.
  Exclusive-load handoff tests cover all five widths, physical aliases, other
  physical pages with identical bits, intervening writes, read-only RAM,
  permission/alignment/unmapped faults and device rejection without callbacks.
  The GPU-exclusive regression also verifies that resolution leaves newer
  device data pending. The CPU suite passes 80 library and 17 integration tests
  on x86; six library and three integration exclusive tests pass under Arm
  QEMU. Six further LCQ tests execute scalar exclusive loads, alias/ZR/SP
  operands, lazy flags, successive loads, dirty-address fault reconstruction,
  GPU retry, intervening writes and CLREX across frame lifetimes. Cold cases
  cover RAM remapping and device rejection without callbacks. Both encoders
  have exact read-fault maps and no helper call. These six tests pass on x86
  and under both Arm QEMU profiles; they are complemented by the invocation
  owner's monitor-handoff tests above.
  Three additional tests cover LDXP/LDAXP W results, arbitrary register pairs,
  base/destination aliases, SP/ZR, eight-byte alignment, both dirty PRE
  destinations on faults and a full-width reservation consumed through a
  physical alias. Both encoders emit one eight-byte read-fault record without
  a CAS or helper. GPU-retry and cold-remapping/device tests also cover W pairs.
  The 15 focused exclusive tests pass on x86 and both Arm QEMU profiles.
  Three X-pair tests cover both orderings, read-only RAM, SP/ZR/base aliases,
  all 128 reservation bits, sixteen-byte alignment and PRE-state preservation.
  Both encoders retain two read maps without CAS/helpers; x86 compilation also
  succeeds with CMPXCHG16B disabled. GPU retry and cold remapping/device cases
  include X pairs. The 18 focused exclusive tests pass on x86 and both Arm QEMU
  profiles. Clippy passes on both targets.
  The current x86-64 JIT suite passes 367 library plus 4 integration tests;
  the preceding CPU/interpreter suites passed 78/107 library tests and their integration suites
  (three existing ignored interpreter tests).
  The preceding shared fault suite passed 18 library plus 1 integration test;
  CPU/interpreter suites passed 78/106 library tests and their integration suites
  (three existing ignored interpreter tests).
  All eight delivered-LCQ retry/escape tests and the legacy raw-MMIO regression
  pass under QEMU 11.1.1.
  The preceding seven single/pair/structure cold-completion tests also passed with that
  QEMU (16 focused tests total; the exhaustive run took about ten minutes).
  The CPU and shared runtime pass their 78 and 17 ordinary Arm tests.
  All eight fatal/handler-chain scenarios were also verified
  directly with that QEMU in the preceding increment; the subprocess supervisor
  was bypassed because this machine's binfmt entry still launches QEMU 6.2.
  Clippy passes for both targets; formatting and diff checks pass.
  The preceding increment passed 78 CPU and 106 interpreter library tests and
  their integration suites (three pre-existing ignored tests), plus sixteen
  focused memory/map/legacy-structure JIT tests and the explicit interpreter
  partial-load regression under AArch64 QEMU.
  The preceding pair increment also passed the explicit LDNP/STNP regression
  and legacy pair-fault/publication tests under QEMU.
  The CAS backend fixes require the local fork override until the pin is advanced.
  Current closure evidence and the step-5/6 handoff are summarized above.

- [x] **Integrate guest-memory invalidation with the coordinator.** Register
  exact affected units/compile claims before acknowledging notifications.
  Find all fragments covering changed executable bytes, including overlapping
  roots and physical aliases. Serialize code writes, mapping/permission changes
  and tracking transitions with Closing/Closed: exclude new admission and
  publication, drain active execution/fault dispatch, remove roots, then expose
  the memory mutation and reopen with a fresh epoch. A writing vCPU must leave
  its own epoch before waiting for this rendezvous.

  Revalidate captured memory state on publication and handle invalidation-stream
  overrun explicitly, never by assuming missed records affected nothing.
  Reuse Task 2 retirement, strong references, directory grace periods and
  reclamation; extend exact maintenance records for memory work rather than
  introducing a second coordinator. No cache/memory lock is nested under JIT
  state while applying mutations or waiting.

  **First checkpoint:** the existing lifetime owner now queues exact memory
  invalidations in its bounded unit records. Physical-page lookup includes all
  aliases and mapping generations; virtual ranges intersect every captured
  instruction, including non-root words and overlapping fragments. Address-space
  cache invalidation and a global resynchronization entry point are available.
  Closure cancels old compile claims and prepared publications even with no
  executing reader. A separate sequence in each existing unit record prevents
  duplicate requests or an overlapping eviction/tier cutover from acknowledging
  memory work before unlink. Active HCQ baseline promises are withdrawn first;
  stale in-flight pins must be released before the stop can finish. This does
  not enable HCQ compilation. Directory and span reclamation remain unchanged.

  **Mapping handshake checkpoint:** `ExecutionGate::acquire_mutation` now
  invokes a bound engine before returning memory mutation authority, including
  when there are no execution leases. Read-only capture remains separate;
  instruction capture upgrades to a mutation hold only when tracking needs
  arming, as detailed below. Exclusive
  ownership keeps admission closed without retaining the gate mutex through
  memory work or coordinator callbacks. `ExecutionMemory` routes resize,
  alias map/unmap, permission and attribute changes through this handoff.
  Their known affected ranges are registered before taking the mapping lock;
  mapping-state validation still occurs under that lock after quiescence.
  The JIT's hold uses the existing coordinator, prevents MappingChange batch
  acknowledgement until memory locks and stream reservations are released,
  and preserves unrelated maintenance reasons. Abandoned preflights release
  the stop; unwinding disables JIT admission. Coordination errors retain their
  diagnostic through CPU/Horizon and are not converted into guest result codes.

  **Device visibility checkpoint:** retained-range device preparation (ordinary
  and resident), device-write publication and visibility invalidation now use
  the same mutation handshake. Targets include every retained physical page,
  deduplicated within its backing store, so virtual aliases and captures which
  publish before closure cannot escape invalidation. Even read transfers retire
  affected code: a failed upload can make its source page Invalid. Unrelated
  code remains resident. Device callbacks, visibility changes and invalidation
  stream publication execute while JIT admission stays Closed, with no JIT or
  gate mutex held. The protocol does not claim that GPU work has completed;
  deferred writeback still goes through the existing visibility authority.

  **Canonical batch checkpoint:** staged canonical writes now discover observed
  executable pages with memory capture admission closed, close the JIT before
  waiting for execution readers, and retain the stop through byte/generation
  and invalidation-stream publication. Each log is reserved once in stable
  order, after rendezvous; no log mutex spans an engine callback. Ordinary data
  commits do not close JIT admission or cancel compile claims; the initial
  tracking snapshot has its own stop as described below. Captures made
  after staging are included at commit. `ExecutionMemory::write_bytes` delegates
  log ownership to the batch and revalidates its virtual translation after
  quiescence; a concurrent change retries translation instead of writing the
  retained old mapping. Retained physical batches keep their existing semantics.

  **Retained allocation checkpoint:** `CanonicalAllocation::write` now uses the
  same conditional executable-write handshake and shared log grouping. It
  discovers targets under closed capture admission, drains execution before
  copying, and publishes all affected logs before releasing the engine hold.
  Ordinary unobserved data does not call the engine. Writes still copy only
  the requested bytes, without full-page snapshots. Fallible protection/dirty
  work completes for every affected page before any bytes or generations are
  written. The redundant allocation transaction mutex is removed: the backing
  execution gate already serializes reads, writes and retained-range mutation;
  immutable range construction needs no lock. Visibility and engine callbacks
  no longer inherit that mutex.

  **Trusted host overwrite checkpoint:** `overwrite_mapped_ram` now discovers
  physical executable targets with capture admission closed and enters the
  existing engine rendezvous before copying bytes. Read-only guest mappings
  remain writable by this trusted host API; ordinary data does not cancel JIT
  claims. Device writeback releases both the mapping lock and gate, then retries
  the complete virtual lookup, so a callback remap cannot redirect the write to
  retained old backing. Byte copying remains direct, with no page snapshots.
  The API returns the actual fault instead of a boolean; HID preserves its
  diagnostic instead of reporting every failure as an invalid range. Callers
  must release their execution lease/epoch before invoking the host writer.

  **Instruction-cache checkpoint:** IC invalidation now coordinates with a bound
  code owner before publishing its stream record. VA forms resolve the current
  physical page under closed capture admission and cover all aliases and
  overlapping roots; global forms select the address space and cancel stale
  compilation admission. The mapping/log locks do not survive the rendezvous.
  Native IC completion uses the existing owned system exit after Invocation and
  memory-lease release; failures retain PRE state/PC and the original diagnostic.
  With no bound code owner, IC only notifies the existing stream: it changes no
  bytes/mappings and must not wait for the caller's own memory lease. This also
  preserves the current runtime until cutover. Prefetch does not close admission.

  **Data-cache checkpoint:** DC IVAC/CVAU/CIVAC retain the existing canonical
  visibility semantics. Coherent pages return without closing JIT admission or
  canceling captures. Device writeback runs with no mapping/log lock and then
  revalidates the virtual mapping generation, physical backing and CPU-visible
  state; a remap retries the current destination and an unmap reports its fault.
  DC no longer reserves a duplicate executable-content record around the device
  callback: device publication already invalidated executable aliases before
  exposing GpuNewer state. Downloading that published revision is not another
  guest write. Failure retains the native exit's PRE state/PC and diagnostic.

  **Stream-consumption checkpoint:** the lifetime owner now consumes one bounded,
  coherent invalidation snapshot from canonical mode. It reuses the same memory
  hold and exact retirement drain, with no log lock held while waiting. Explicit
  HistoryLost retires all code and cancels old compilation admission. Only the
  cursor returned by that read (or its HistoryLost error) is acknowledged, after
  successful draining/completion; later records remain pending. Empty reads do
  not close admission. Other source errors and registration/coordinator failures
  retain the old cursor and disable admission with their diagnostic. The runtime
  must own one cursor per process/source, initialized to INITIAL, and wire this
  cold consumer during step 6. Consumption cannot replace a producer's mandatory
  pre-mutation stop or make an already-unsafe write safe retroactively.

  **Initialization checkpoint:** physical RAM initialization now uses the same
  conditional executable-write rendezvous. A mutable memory borrow does not
  exclude owned JIT captures, published units or retained backing readers.
  Initialization retires affected code before copying bytes and keeps the hold
  through generation/HostWrite publication. Device reconciliation runs outside
  the gate and is rechecked after acquiring it. The write still copies only the
  requested bytes; it does not stage full-page snapshots. The API now returns
  errors with their diagnostic instead of a boolean, and setup callers consume
  that result. Empty valid ranges are no-ops; ordinary data initialization does
  not cancel code claims. Native stores and their dirty-epoch protocol are unchanged.

  **Tracking-rearm checkpoint:** explicit CPU-write dependency rearm and the
  dirty/whole/all snapshot APIs now hold the existing engine rendezvous while
  copying bytes and restoring restrictive page protection. The empty mutation
  target set stops execution and cancels old compile admission without retiring
  unchanged native units or publishing a fabricated content invalidation.
  Device reconciliation releases the holds before its callbacks and retries.
  Clean conditional snapshot queries only observe dirty epochs and do not close
  admission. Rearm returns `Result<bool, CanonicalRangeAccessError>`: false means
  the dependency became volatile; coordinator/backing failures retain their
  diagnostic. Snapshot/rearm callers must own no native epoch, memory lease or
  cache lock. The monotonic first-write dirty transition is unchanged.

  **CPU-write audit:** the existing `CpuMemory::maintain_cache` contract makes
  ordinary guest stores/atomics visible to translated instruction execution at
  IC maintenance, not by emitting an invalidation on every store. Preserve that
  distinction from immediate host/device mutation; do not add per-store JIT
  callbacks. Captured instruction images still require dirty-state revalidation
  before publication.

  **Initial dependency checkpoint:** CPU-write dependency capture now closes
  each backing store and its bound engine before arming page protection. Stores
  are acquired in stable identity order and duplicate physical pages are armed
  once. Empty input, coordinator failure and backing failure return an explicit
  error; capture no longer silently returns no dependency. Maxwell resources,
  shader capture, wgpu and presentation preserve that diagnostic. Maxwell's
  obsolete optional-dependency fallback is removed for retained resources and
  descriptor reads. Descriptor tracking is armed before reading bytes, so a
  concurrent CPU write cannot become the clean baseline for an older read.
  GPU interfaces still distinguish resources which genuinely
  do not have a CPU-write dependency. As with rearm, unchanged code stays resident
  while old compile admission is canceled. This does not change the separate
  demanded instruction-image capture path used by the JIT itself.

  **Batch-snapshot checkpoint:** the first staged snapshot of a physical page
  now uses the existing tracking rendezvous before copying/arming protection.
  Only stores with newly captured pages intersecting the requested byte range
  participate, in stable identity order. Editing already-owned staged bytes
  does not touch backing protection or repeat the stop. Staging publishes no
  guest bytes, generation or content notification, preserves the memory mapping
  epoch and retires no native unit,
  but its protection transition cancels old compile admission, including for
  data-only pages. Subsequent private edits and a data-only commit preserve
  fresh compile claims. Reconciliation runs outside the holds and retries;
  errors retain their diagnostic and require discarding the batch. Ordinary
  `read_staged` remains read-only and does not arm tracking.

  **Instruction-capture checkpoint:** already-armed instruction reads retain
  memory exclusion without closing JIT admission or canceling other compilers.
  Encountering an unarmed demanded page discards the partial copy and retries
  under the tracking rendezvous, with no memory lock/lease held during its
  callback. Fault readers drain before protection changes; unchanged code and
  the content cursor are preserved. After capture, a canceled initial claim
  competes for a new exact-key reservation, then revalidates the owned image
  before allocating its emission identity. A competing owner/published entry,
  closed admission or changed image rejects the work. Old publication tokens
  remain stale; native output is never relabelled. Device reconciliation still
  releases the holds and retries; rejection retains its original diagnostic.

  **Checked-atomic checkpoint:** the existing backing-page lock now covers the
  actual checked atomic load/CAS, including the CAS's preceding dirty transition.
  Instruction copying and its observation stamp use that same lock, preventing
  rearm between dirty marking and the write or old bytes carrying a newer stamp.
  Device reconciliation runs outside page locks. Load-exclusive also releases
  the mapping lock before reconciliation and retains the physical page selected
  for its reservation, even if the callback remaps the virtual address. Native
  atomics remain unchanged; they use execution-lease exclusion, not this mutex.
  Checked CAS/RMW/exclusive stores dirty captures without canceling compile
  admission or retiring code on every store; IC still exposes the code change.

  **Checked-RAM checkpoint:** ordinary loads/stores keep the existing page lock
  through byte copying; stores retain it from dirty preparation through commit.
  Cross-page accesses lock distinct physical pages in identity order, handling
  duplicate aliases once and preparing both pages before writing either.
  Device reconciliation releases all page/mapping locks and retries the complete
  virtual translation, including a first page remapped by a second-page callback.
  The unlocked page-level `write_cpu_prepared` API is removed. Native RAM access
  and IC visibility semantics are unchanged. MMIO handlers are not changed here.

  **MMIO checkpoint:** checked reads/writes retain the resolved physical handler
  and release the mapping lock before waiting for handler exclusion or invoking
  the callback. Physical aliases share that exclusion. A remap cannot destroy
  an in-flight handler or replay its side effects; device errors and size checks
  remain precise. A panicking callback poisons only its handler, whose subsequent
  accesses fail explicitly. Handlers must not recursively access themselves;
  callbacks requesting a rendezvous require callers to release their execution
  leases/readers first, as LCQ owned completion does.

  **Publication-race closure:** image validation rejects observed guest writes;
  an ordinary/atomic guest store after its last check may leave pre-IC code
  publishable until IC. IC closes admission under the same state mutex as
  publication: it either cancels the candidate's old epoch or finds and retires
  the published unit before completing. Host/device content publication and
  mapping changes use this handoff immediately, without waiting for IC. The
  final epoch/cursor checks also reject already-prepared stale output. Page
  locks couple checked byte writes with dirty stamps; native writes remain
  protected by execution leases during capture. No memory lock is taken under
  publication state, and no per-store JIT check/version was added.

  Step 5 is closed. Production binding/cutover stays in step 6: runtime callers
  must release their own invocation/lease before rendezvous-capable completion
  and consume notifications from canonical mode. Installing the observer into
  the old executor alone is not a supported cutover. The local Wasmtime override
  is unchanged.

  Validation: nine additional coordinator/memory tests cover real LCQ captures,
  permission visibility with a live fault reader, aliases, resize, stale
  publication, abandoned preflight, competing maintenance and terminal errors.
  Two gate tests check callback lock ordering and rejection; Horizon verifies
  diagnostic propagation. Five device tests additionally cover a live LCQ fault
  reader, physical aliases, publication between device prepare/publish, native
  execution of the new device-produced instruction, failed read transfer,
  resident preparation and rejection without side effects. Five canonical-write
  tests cover later captures, overlapping roots, writable aliases, a live fault
  reader, unaffected data writes, rejected commit validation and permissions
  changed during device writeback. A memory regression covers multi-store
  batches with shared/distinct logs and unlocked callbacks through publication.
  Four allocation regressions cover reader draining, exact touched-page targets,
  log/byte visibility when the hold is released, rejection with its original
  diagnostic, later-page failure without partial bytes, and concurrent read/write
  atomicity without the removed mutex. These allocation tests use an injected
  engine observer; production binding remains part of the runtime cutover.
  Four host-overwrite tests cover real LCQ/fault readers, read-only aliases and
  overlapping roots, unaffected data/compile claims, errors without byte/log
  publication and virtual remapping during device writeback. A HID regression
  verifies preservation of the coordinator failure through the production caller.
  Four IC tests cover physical aliases/live fault readers, address-space scope,
  canceled captures, bad addresses/prefetch and actual native IVAU/IALLU completion
  with success/fault/coordinator-rejection paths. A CPU regression checks IC
  stream notification under an existing lease with no bound code owner.
  Two DC regressions execute all three operation encodings natively and cover
  coherent pages with live compile claims, device writeback, callback remapping,
  unmapping and failure, unrelated resident code, precise completion, and no
  duplicate cache-invalidation record. The callbacks acquire the mapping and
  log locks directly to check that neither is retained by DC completion.
  Four stream regressions cover exact roots with a live native fault reader,
  log publication during draining, empty consumption preserving a compile claim,
  forced ring overrun with unrelated retained records, a later publication after
  the overrun snapshot, and source/registration/coordinator errors without cursor
  advancement. The sources inject notifications, not uncoordinated memory writes.
  Devices are simulated; this is not GPU-hardware validation.
  Four initialization regressions cover owned LCQ captures, a live fault reader,
  overlapping roots, unaffected code/data, empty writes, range/coordinator errors
  and device reconciliation with an unlocked execution gate. A CPU regression
  verifies retained-reader draining, unchanged bytes/generation before quiescence
  and one generation/HostWrite publication afterward.
  Three tracking regressions exercise explicit rearm and all three snapshot
  forms with an active LCQ fault reader, unchanged code retention, compile-claim
  cancellation, clean-query admission, later CPU writes and exact errors. A
  memory regression preserves an invalid-backing error instead of reporting a
  volatile dependency.
  Initial-capture coverage adds a live LCQ reader with duplicate input ranges,
  rejection without a dependency, invalid backing/empty input, and a reversed
  multi-store input whose second owner rejects before any page is armed. Prior
  holds are released. GPU resource and wgpu regressions preserve backing errors.
  Three batch-snapshot regressions cover an active native fault reader before
  staging, unchanged bytes/code on abandonment, private edits without another
  stop, data-only commit preserving fresh claims, and coordinator rejection
  without a staged page. The multi-store log test distinguishes tracking-only
  staging from content publication and checks unlocked logs in both phases.
  Four instruction-capture regressions cover reader draining before rearm,
  already-armed capture preserving other compilers, a code write between capture
  and new admission, and rejection without bytes/protection changes. Two claim
  regressions cover fresh ownership without reviving old tokens, competing
  owners and closed admission.
  Atomic coverage adds concurrent 128-bit CAS/capture stamp checks, a device
  writeback remap during load-exclusive with a retained physical reservation,
  and CAS/RMW/exclusive stores at all five widths under live JIT admission.
  Captures become stale while unchanged code/claims remain resident until IC.
  Three checked-RAM regressions cover cross-page read/write retranslation after
  a device callback remaps the first page, failed second-page reconciliation
  without a partial store, reverse physical lock order and duplicate aliases.
  The concurrent capture test now mixes checked stores and CAS; live-JIT IC
  coverage also exercises ordinary stores at all five widths.
  Two CPU MMIO regressions cover unlocked RAM/mapping access, remapping without
  replay, shared aliases, device/size errors and handler poison isolation. A
  real native LCQ read/write regression retires its own source code and removes
  its device mapping during completion, proving released execution protections,
  retained handler lifetime and exactly-once completion.
  Three final publication regressions place ordinary stores, CAS, RMW and
  exclusive stores before validation and after its last check. They execute
  the old captured breakpoint before IC and the new one afterward, and reject
  IC, host writes and permission changes inserted before publication. Existing
  unit tests additionally cover closure/cursor changes after preparation,
  rejected output reclamation and fault readers retained across closure.
  Complete x86-64 JIT (459 library + four integration),
  CPU, interpreter, memory, Horizon, runtime, Maxwell (470 library tests) and
  wgpu (24 library tests) suites pass; three existing
  interpreter integration tests remain ignored. Two existing runtime tests still
  require caller-owned content/keys and remain ignored. AArch64 QEMU (`max`)
  passes all 135 lifetime, 89 CPU and 59 memory library tests. Clippy passes on x86-64
  (all eight packages) and AArch64 (JIT/CPU/memory); formatting/diff checks pass.

  **Exit:** controlled races cover compilation versus writes/remapping, physical
  aliases, overlapping LCQ roots, executable writes and reader-held fault data.
  No old code runs after executable-content invalidation becomes visible, no
  candidate survives its admission/cursor invalidation, and no operation waits
  for its own active reader. Ordinary guest stores retain the pre-IC semantics
  described above.

- [x] **Switch the runtime, implement control budgets and remove old execution.**
  Make runtime JitProcess/JitThread use the new owner and vCPU-local compiler.
  A miss leaves native mode, compiles only the demanded key, then restarts
  admission; unresolved branches/calls/returns use the canonical resolver.
  Preserve guest call/LR behavior, SVCs, loader return, stop requests and
  scheduler-visible faults through the new boundary.

  Integrate PollBudget with runtime slices: a nonpositive slice never enters
  native code; completed straight-line work is charged at block boundaries and
  required backedges, not through per-instruction progress stores. Honor the
  independent block/backedge maintenance check, reconcile spent work once on
  exit and preserve NZCV/FP ordering. Keep the shared budget representation but
  do not emit functional samples, maintain hotness tables or enqueue HCQ work.
  Update callers/tests that incorrectly require exact instruction stepping.

  Wire cold capacity reclamation and terminal shutdown, including outstanding
  compile claims. Remove the superseded production region discovery/executor,
  PublishedRegion ownership, lookup chains, context-tail gateways, JITModules,
  per-entry promotion and old HCQ workers/configuration that only served them.
  Move reusable semantics instead of retaining adapters to the old context.
  Update exports and boundary tests to the new implementation; do not defer
  callable legacy removal to Task 9. Linked Task 1 proof fixtures remain test-only
  until Task 4 replaces their manual link ownership.

  **Process/vCPU kernel checkpoint:** `cpu-jit/src/engine.rs` now binds one
  retained `ExecutionMemory` to the lifetime owner before constructing workers.
  It rejects checked/unbound memory and a busy/already-owned mutation gate;
  there is no rebinding or background HCQ. Observer binding accepts an Arc-held
  memory owner through the gate's existing idle/single-binding check, without
  a second synchronization mechanism. Memory retains Lifetime, not JitProcess,
  so the ownership graph has no cycle.

  Each new JitThread owns its reader, LCQ compiler and exclusive monitor without
  compiler mutexes. Cold demand uses the existing exact-key claim and reports
  first-word fetch faults without publishing negative entries. Native invocation
  acquires/releases its own memory lease and returns owned exits. Fault-stack
  registration is lazy on the executing OS thread, not the constructing thread;
  the worker-ownership checkpoint below separates it from per-process vCPUs.
  Four regressions exercise native execution/recompilation after bound mutation,
  shared code across vCPUs, construction-to-worker transfer, precise fetch faults,
  rejected binding and ownership release on x86-64 and AArch64 QEMU.
  The complete x86-64 JIT (463 library + four integration), CPU, memory, Horizon
  and runtime suites pass; the two content/key-dependent runtime tests remain
  ignored. Clippy, formatting and diff checks pass.

  These kernel APIs now back the public JIT, as described in the runtime cutover
  below. The old runtime's slice-wide memory lease does not survive into LCQ
  demand or rendezvous-capable completion. No fork changes were needed; the
  local override remains active.

  **Native work-budget checkpoint:** canonical LCQ exits now subtract the
  completed path length from the pinned poll register exactly once. Dispatch
  includes the completed branch/final instruction; architectural/helper and
  unsupported PRE exits exclude their pending instruction. The epilogue uses
  flag-transparent LEA/SUB after writeback, with no per-instruction progress
  store. FP activation does not constitute a guest block exit or reset the count.
  An escaped memory fault has no epilogue, so owned fault handling reconciles its
  completed straight-line prefix from the retained instruction image. Repair/retry
  does not charge separately, and a partially completed access is not counted as
  a completed guest instruction. The owned completion dispatcher below applies
  the successful cold instruction's separate charge.

  The kernel refuses a nonpositive slice before worker registration/native entry.
  Five regressions cover the 512-instruction ceiling, slice/sample overshoot,
  taken/untaken branches with lazy NZCV, PRE exits, escaped faults and a tracked
  store retry through a writable alias. The nine engine tests pass on native
  x86-64 and AArch64 QEMU. This is work accounting, not yet a complete scheduler
  slice/control loop; no functional sampling or HCQ work was enabled.
  Validation also passes all 468 JIT library + four integration tests on x86-64,
  the CPU/memory/runtime suites (two existing content/key-dependent runtime tests
  ignored), 31 native-boundary tests on AArch64 QEMU, Clippy on both targets and
  formatting/diff checks. The local fork and dependency pin are unchanged.

  **Owned completion checkpoint:** `engine/completion.rs` consumes the native
  kernel's owned exits after FP/reader/lease release and routes them to the
  existing typed memory, exclusive-store, system and exact FP providers. Success
  charges one cold instruction, including scheduling and a completion after
  native-prefix budget exhaustion. Data faults, FP traps and unsupported/invalid
  instructions do not earn a completion charge. Branches already committed and
  charged natively are not charged again. Explicit SVC/breakpoint delivery costs
  one instruction and leaves source PC for runtime exception dispatch/return.
  Internal provider/coordinator failures retain source bits, state and diagnostic;
  system host-backing failures are not fabricated guest data aborts.

  Unsupported/reserved/unallocated diagnostics decode the retained instruction,
  never reread possibly replaced memory. Six regressions cover timer/FPSR reads,
  scheduling, self-invalidating IC and invalid IC addresses, exact FP success/trap,
  changed-code diagnostics, explicit exceptions/native dispatch, and successful
  versus failed MMIO without replay. The dispatcher does not itself re-enter
  native code; the slice loop below owns continuation. Production cutover remains
  pending.

  Validation passes all 474 JIT library + four integration tests and the
  CPU/memory/runtime suites on x86-64 (two existing content/key-dependent runtime
  tests ignored), all 15 engine tests on AArch64 QEMU, Clippy on both targets,
  documentation checks and formatting/diff checks. Native Arm hardware remains
  unavailable. No fork changes were needed; the local override remains active
  and the committed dependency pin is unchanged.

  **Canonical slice-loop checkpoint:** `engine/execution.rs` now connects
  protected native invocation, exact-key cold demand and owned completion.
  Compilation, competing-owner waits and stale outputs restart canonical checks;
  no native entry is retained across them. Unlinked calls/branches/returns demand
  only their actual destination. A loader-return sentinel is recognized before
  fetch and reports X0 without compiling the sentinel.

  Zero-budget slices do not register a worker, demand code or consume pending
  notifications. Positive slices check preemption and interrupts at every
  canonical boundary, independently of the poll deadline. Bound-memory mutation
  remains the invalidation authority; CPU notifications only acknowledge a
  canonical vCPU with no retained entry/reader/lease. Completed native work and
  successful cold work contribute once to the report; overshoot prevents the
  next native invocation. The vCPU retains the sample phase between slices but
  generates no functional samples or HCQ work. Budgets above i64::MAX are rejected
  explicitly rather than silently truncated.

  Owned guest stops retain priority over budget/preemption arriving during their
  instruction. After successful completion, the loop checks preemption, pending
  interrupts, loader return and budget, in that order. An admission closed by
  maintenance yields a Safepoint without spinning or acknowledging the owner's
  work; shutdown is Unavailable. This is now the public backend's slice loop;
  the terminal-close and capacity APIs below supply its runtime maintenance.

  Eight regressions cover empty/invalid budgets, pending control/events, demanded
  call/return/system continuations, loader return, backedge overshoot and poll
  phase persistence, completion-time preemption, precise fetch/data faults,
  invalidation notifications and closed/shutdown admission. All 23 engine tests
  pass on native x86-64 and AArch64 QEMU. The complete x86-64 JIT suite passes
  (482 library + four integration tests), as do CPU/memory/runtime suites; two
  existing runtime content/key tests remain ignored. Clippy passes on both
  targets, with formatting/diff checks clean. No fork changes or dependency-pin
  update were required; native Arm validation remains pending.

  **Terminal-close checkpoint:** the new JitProcess now exposes idempotent
  `request_stop` and a cold `try_shutdown` pass through the existing Lifetime
  coordinator. Stop closes admission and wakes exact-key compiler waiters.
  Shutdown returns pending while a reader, another transition owner, a compiler,
  a memory-authority hold or retained code output prevents safe teardown. It
  neither waits for its own worker nor spins on those owners; the runtime must
  join/release them before retrying. Success requires actual code/index release
  and acknowledgement of the terminal batch, not just an unmapping attempt.

  Live compile claims are counted under the existing cold state mutex, including
  canceled claims whose old dispatch slots have already been reclaimed/reused.
  Their drop releases the count even when their slot no longer exists. This
  closes the gap where an unpublished compiler without an installed span could
  outlive a supposedly completed teardown; no generated-code work was added.
  Seven regressions cover waiter cancellation, old/reused compile reservations,
  competing coordinator ownership, unfinished batch acknowledgement, memory
  holds, admitted readers, actual published-code reclamation and stop requested
  from a system completion. Worker fault contexts still belong to their original
  OS thread, now through the separate worker owner below; runtime worker teardown
  and public cutover remain to be wired.

  Validation passes all 489 JIT library + four integration tests and the
  CPU/memory/runtime suites on x86-64 (two existing content/key-dependent runtime
  tests ignored), 139 lifetime and 26 engine tests on AArch64 QEMU, Clippy on both
  targets, documentation and formatting/diff checks. The fork and dependency pin
  are unchanged; the local override remains active.

  **Cold capacity checkpoint:** the slice loop checks the existing soft limit
  only on a cache miss, never on a cache hit or per guest instruction. Soft
  pressure or a typed allocation-capacity failure runs one coalesced Eviction
  rendezvous using `Transition::relieve_pressure`, with no own invocation,
  memory lease or compile claim retained. It reclaims actual spans/indexes and
  uses the existing oldest-HCQ/then-unpinned-LCQ policy. It does not wait for
  compiler snapshots to release capacity or count pending retirement as refunded
  bytes. Concurrent memory holds can complete without a pressure waiter owning
  their transition; shutdown wakes/cancels that waiter.

  Each miss gets at most one pressure pass. The retry rechecks control and
  recaptures/recompiles under fresh admission; failed staged output is never
  relabelled. This pass considers existing charges, not a speculative allocation
  estimate: the real allocator still enforces all charges on retry. If concurrent
  users or retained resources prevent the retry from fitting, it reports
  Unavailable with the capacity reason, PC, state and earned progress rather than
  looping or falling back. Executable-install errors retain their storage type
  and diagnostic instead of being flattened into lowering errors.

  Four regressions cover actual LCQ eviction/re-demand without guest budget
  charges, a hard-capacity stop with unchanged guest state and subsequent cache
  reuse after the external charge is released, staged-install capacity cleanup,
  and reclamation versus a memory-authority hold or shutdown. Runtime cutover
  and retirement of the legacy production path remain pending.

  Validation passes all 493 JIT library + four integration tests and the
  CPU/memory/runtime suites on x86-64 (two existing content/key-dependent runtime
  tests ignored), 140 lifetime and 28 engine tests plus the staged-install
  capacity regression on AArch64 QEMU, Clippy on both targets, documentation and
  formatting/diff checks. No fork or dependency-pin changes were needed; the
  local override remains active and native Arm hardware validation is pending.

  **Host-worker ownership checkpoint:** runtime inspection exposed that a host
  worker can retain vCPU objects for several processes. Giving each vCPU a fault
  registration installs nested alternate signal stacks; retiring processes in
  another order can restore a stack already freed by the earlier retirement.
  Fault-context ownership now resides in `cpu-direct-memory::NativeWorker`,
  constructed inside the runtime's OS-worker loop and shared across all its
  processes and CPU backends. The interpreter, current production JIT and new
  LCQ kernel explicitly borrow it. NativeWorker is neither Send nor Sync;
  registration remains lazy. No per-entry lock, TLS registry, refcount or
  signal-stack installation was introduced. Duplicate registration on the same
  TID is rejected before replacing the existing stack.

  The interpreter retains its process-bound frontend, but borrows the worker
  through a scoped DirectMemorySlice. The slice publishes once, clears its
  snapshot on Drop and performs no per-access registration or allocation. The
  obsolete unbatched frontend path and per-vCPU signal-stack owners are removed.

  Process/vCPU retirement leaves that worker registration alive. Explicit worker
  `finish` uses checked `WorkerFaultContext::unregister`, which reports wrong-TID
  and alternate-stack restoration errors and retains resources on failure.
  Successful unregistration is idempotent; a later Drop cannot clear a reused
  slot. Best-effort Drop retains its previous leak-on-failure safety behavior.
  Runtime worker shutdown propagates restoration failures through the joined
  worker result. Regressions exercise native data faults across independent
  LCQ/interpreter processes, either backend's first use and retirement (including
  coordinator-thread destruction), surviving-process execution, exact stack
  restoration and wrong-thread teardown rejection/retry. A runtime coordinator
  regression alternates the production JIT and interpreter on one OS worker,
  retires either process first and continues native memory accesses in the other.

  **Runtime cutover:** public JitProcess/JitThread now select LCQ. Runtime
  construction binds the owned ExecutionMemory once, before vCPU creation.
  Runtime invalidation synchronization validates that authority and acknowledges
  notifications; the memory observer remains the code-invalidation authority.
  Only the interpreter retains a slice-wide mapping lease. JIT capture/native
  entry acquire their own short leases; cold completion and compilation hold no
  caller lease. Runtime stop closes admission; retirement acknowledges the final
  cursor and clears exclusives before releasing the vCPU. Shutdown reports
  outstanding owners and caches only successful completion, allowing retry.

  The old `direct/` executor, compiler, PublishedRegion/lookup ownership,
  context-tail gateways, slow adapters and HCQ workers are removed.
  `cranelift-jit`/`cranelift-module` remain dev-dependencies for native ABI proof
  fixtures only. Architectural coverage is exercised by LCQ tests and the public
  differential tests; step 7 still owns the final coverage/handoff review.
  The production regression verifies zero budget, an 18-instruction fragment
  completing a budget-1 slice, recompilation after bound-memory code mutation,
  worker retirement and process teardown. This is bounded block-level work
  accounting, not exact stepping or a claim of final linked-JIT performance.

  Post-cutover validation passes all 383 JIT library + four integration tests,
  and CPU (89 library), shared fault-runtime (22), interpreter (107), memory (59)
  and runtime (65) suites on x86-64. The reduced JIT count reflects removal of
  the old executor's tests, not a claim that counting tests proves coverage.
  AArch64 QEMU passes 386 JIT library cases and the runtime suite/integrations;
  the remaining JIT case's subprocess supervisor cannot inherit Cargo's runner
  and fails under system binfmt. Its child passes when explicitly launched with
  QEMU 11.1.1: SIGSEGV and `reason=unattributed-native-pc`, as documented in
  `docs/aarch64-tests.md`. Three existing interpreter and two content/key-dependent
  runtime tests remain ignored. Workspace/all-targets checking, Clippy on both
  targets, documentation and formatting/diff checks pass. No fork or pin changes;
  the local override remains required. Native Arm and real-homebrew validation
  remain pending; step 7 includes the final architectural coverage review.

  **Exit:** configured JIT execution uses only the new LCQ path, with no silent
  fallback or legacy HCQ startup. Zero/small slices and stop requests remain
  responsive with bounded overshoot. Shutdown drains compilation and releases
  real code/metadata; tests assert the new contracts rather than legacy shapes.

- [x] **Validate the vertical slice and close the handoff.** Run the existing
  instruction differential tests against the new JIT, preserving coverage of
  previously supported NormalizedA64 variants. Compare architectural state and
  precise failures at actual observation boundaries, not exact slice counts.
  Consolidate focused capture/deduplication, fault, invalidation, capacity,
  shutdown and emitted-shape evidence from the preceding steps.

  Run affected CPU, memory, direct-memory and runtime tests as well as formatting
  and Clippy. Execute on native x86-64 and AArch64 QEMU; run native Arm when
  available and label that evidence separately. Smoke-test an already available
  legal homebrew through the normal runtime if its required guest services are
  supported; do not create a benchmark suite or conceal unrelated blockers.
  Inspect remaining legacy references and record the concrete Task 4 handoff.

  **Coverage review:** the catalog tests cover normalized integer, control,
  register SIMD, FP, memory and system lowering through the new ABI. Restored
  missing pre-cutover subencoding cases in `lcq/compiler/tests/integer.rs` and
  `simd.rs`, including divide-by-zero/signed overflow in both widths and all
  256 condition/NZCV combinations. The es2gears SXTW regression now verifies
  that only BFM reads the old destination. `shape.rs` verifies bounded code
  size for 1/64/511 NOPs and distinct fault records for all 511 loads in a
  maximum-size fragment on both encoders. The production `engine` test checks
  every recognized-unsupported fixture on both guest platforms, including its
  coverage identity, exact PC, zero progress and unchanged architectural state.

  Existing contract evidence remains in `lcq/tests.rs` (capture/first terminator/
  overlap), `lifetime/compile/tests.rs` (same-key ownership and independent keys),
  `lcq/compiler/tests/memory/` (fault attribution, repair/retry and partial
  completion), `lifetime/memory/` and `lifetime/unit/invalidation/` (mapping/code
  invalidation), and `engine/tests/{budget,execution,capacity,shutdown,worker}.rs`
  (real slice, bounded overshoot, eviction, retirement and OS-worker ownership).
  No new test runner or production instrumentation was added. Corrected obsolete
  Cargo differential aliases to the deleted interpreter crate/test name; the
  aliases now address `nixe-cpu-interpreter`'s existing tests. The documented
  QEMU command no longer skips the deleted `direct` module.

  **Validation commands/results:** all Cargo commands use `--offline --config
  /tmp/nixe-observable-fp-local.toml` in addition to the arguments below.

  - `cargo test -p nixe-cpu-jit -p nixe-cpu -p nixe-cpu-direct-memory
    -p nixe-cpu-interpreter -p nixe-memory -p nixe-runtime --lib --tests --quiet`
    passes on x86-64. After restoring the missing coverage,
    `cargo test -p nixe-cpu-jit --lib --tests --quiet` passes 392 library and
    four integration tests. Other library totals: CPU 89, direct-memory 22,
    interpreter 107, memory 59, runtime 65; their integration suites also pass.
    Three optional interpreter/QEMU tests and two content/key-dependent runtime
    tests remain ignored. The repaired differential aliases resolve their
    actual tests (`cargo test-diff --list`, `cargo test-diff-a64 --list`);
    listing is not counted as executing the optional oracle cases.
  - `cargo check --workspace --all-targets`, `cargo fmt --all -- --check` and
    `git diff --check` pass. Clippy passes for the six affected crates with
    `--all-targets -- -D warnings -A clippy::type-complexity`; the JIT/runtime
    Clippy run also passes with `--target aarch64-unknown-linux-gnu`.
  - AArch64/QEMU: `cargo test -p nixe-cpu-jit -p nixe-runtime --lib --tests
    --quiet` with the target/linker/runner settings and supervisor exclusion in
    [AArch64 tests](../../aarch64-tests.md) passes 390 JIT cases plus integrations
    and 65 runtime cases plus integrations. The five final SIMD/shape/unsupported
    additions pass in a separate run using the same target settings and
    `--lib --quiet -- lcq::compiler::tests::simd:: lcq::compiler::tests::shape::
    recognized_unsupported_catalog_preserves` (395 JIT library cases in total).
    The separately launched fatal child reports `reason=unattributed-native-pc`
    and exits 139 as required; its Cargo subprocess supervisor is not counted
    as passing under the system binfmt runner. Native Arm remains unvalidated.

  **Homebrew:** with the local fork override, the release CLI runs es2gears
  headlessly on the NVIDIA RTX 4070 Ti SUPER, produces frames, and completes
  host-requested shutdown with no rejected SVC kinds or CPU fault. Command:
  `XDG_CACHE_HOME=/tmp/nixe-task3-step7-cache
  LD_LIBRARY_PATH=target/release timeout --signal=INT --kill-after=30s 45s
  target/release/nixe-cli --log-level debug --headless run es2gears`.
  The successful stop reports released resources and a saved pipeline cache.
  A preceding run with a five-second shutdown allowance was forcibly killed;
  that allowance is not evidence of a lifecycle failure. A sandbox run selected
  llvmpipe and could not persist its cache; it is not GPU performance evidence.
  The maintainer also confirms windowed execution, but reports about 10 FPS
  versus 60 before cutover. The subsequent maintainer-run capture in
  `dump/perf-20260911-110035-1KN0ac/` identifies full unit-registry scans in
  retirement selection and pending-work queries, including memory mutations
  with no unit to retire. Those scans are replaced by pending-only intrusive
  LCQ/HCQ lists in the accounted registry slots. Empty retirement checks and
  each drain selection are O(1); reason/sequence queries visit only pending
  units. Duplicate requests share a node, failed unlinks retain it, and
  cancellation/success removes it before slot reuse. HCQ still drains before
  LCQ; memory closure and per-fragment invocation are unchanged.

  **Retirement follow-up validation:** with the local fork override,
  `cargo test -p nixe-cpu-jit -p nixe-runtime --lib --tests --quiet` passes
  396 JIT and 65 runtime library cases plus integrations; the two optional
  real-package tests remain ignored. All 144 `--lib lifetime::` tests also
  pass on AArch64/QEMU. Four new cases cover pending-only membership/reuse,
  sequence queries beyond newer list heads, cancelled cutover requeueing and
  newly queued HCQ priority. Formatting, JIT/runtime all-target checks and
  Clippy pass. Repeat `bash perf-commands.sh` to measure the actual improvement;
  neither these tests nor the missing linking/HCQ implementation quantifies
  the remaining FPS regression.

  **Task 4 handoff:** retain the bound process and per-vCPU owners in `engine`.
  Extend `engine::invoke` / `lcq::invocation::run` from one fragment to a native
  chain under one NativeFrame, execution epoch and mapping lease. Keep demand
  and semantic completion outside those protections. `lcq/compiler.rs` already
  retains final allocated exit states and source-keyed GuestExit records, but
  patches every terminal edge to a canonical exit; connect native bridges,
  PICs and the RSB there using the existing ABI/move emitters. Add owned links
  and the patch rendezvous to lifetime invalidation/retirement before allowing
  targets to be replaced or reclaimed. Preserve budget/control polling and
  existing fault attribution throughout a chain. Production has one JIT route;
  JITModule survives only in test-only ABI fixtures, not as an execution fallback.
  Functional sampling and HCQ remain Tasks 5–6. Keep the local fork override;
  native Arm hardware and a published dependency pin remain explicitly deferred.

  **Exit:** every Task 3 criterion has code/test evidence, supported guest
  coverage is retained, and there is one callable JIT route. Record commands,
  results and actual gaps here; do not claim final linked-JIT performance or
  native Arm conformance from this LCQ/QEMU milestone.

## Starting points and decisions

- New foundation: `cpu-jit/src/lifetime{,/unit}.rs`, `executable.rs`, `abi.rs`
  and `native/`. Task 2's `native/tests/published.rs` and unlinked backend proofs
  demonstrate the real publication/gateway path, not guest decoding or signal
  delivery. Reuse that code, not a parallel runtime.
- Current production: `cpu-jit/src/engine{,/execution,/completion}.rs` and
  `cpu-jit/src/lcq/`; runtime binding is in
  `runtime/src/process/execution.rs`. Shared memory contracts/authority are in
  `cpu/src/memory/`; invalidation publication is in `memory/src/invalidation.rs`;
  signal capture/landing support is in `cpu-direct-memory`.
- Keep the tested Wasmtime revision
  `e2a984d96678207094c0fc50057c8b6bcfd68715`. Inspect
  `/home/pladaria/projects/wasmtime`, branch `nixe`, if integration exposes a
  concrete backend gap. Do not assume a fork change is needed; if one is needed,
  test it locally and pin the agreed committed revision for reproducibility.

### Initial step 1 inspection (historical)

Paths below are relative to `crates/`. This records the pre-cutover replacement
boundaries; the completed-step descriptions above and current code supersede
references here to the removed executor or its caller-owned mapping lease.

- **Ownership and entry.** `runtime/src/process/execution.rs` constructs the
  exported `JitProcess`/`JitThread`, binds an owned `ExecutionMemory`, and passes
  a mapping lease into `run_slice`. Keep those public responsibilities. The new
  process owns Task 2's cache/lifetime and bound memory; each thread owns its
  reader, compiler, NativeFrame storage, control and exclusive state. The old
  `direct/mod.rs::JitThread` already has a vCPU-local compiler: preserve that
  independence, replacing its JITModule output and lookup-node claims, not
  introducing a process compiler mutex.
- **Demanded-PC route.** Start in canonical state; release the caller's memory
  lease before capture/compilation or waiting. Claim the exact BlockKey and
  generation, capture/decode one fragment, reserve its emission identity,
  lower and publish only after capture/claim/admission revalidation. For entry,
  obtain the memory execution lease without a JIT lock, then use protected
  lookup/Invocation to publish the reader epoch before accessing the payload.
  If admission loses to Closing, release the lease and retry from canonical
  mode. Keep the lease and Invocation through native execution and fault retry;
  on an ordinary exit finish state/FP, leave the epoch and release the lease
  before compilation, waiting or a memory transition. Until Task 4, external
  edges return through the canonical resolver, never old lookup chains.
- **Instruction image.** `cpu/src/memory/contracts.rs::InstructionMemory`
  exposes only `code_page_span` and individual `fetch32` reads.
  `ExecutionMemory::fetch` locks mappings for one read; its executable-content
  observer and invalidation cursor do not freeze a whole fragment. Add a
  bounded capture interface owned by the memory authority: consume only demanded
  words, return owned bytes plus exact physical/mapping dependencies and cursor,
  and ensure the complete image corresponds to one coherent memory state.
  Synchronization must cover mapping changes and all writers to the physical
  source, not merely its executable virtual alias. Release capture protection
  before lowering/waiting; do not extend it over compilation. A page crossing
  alone is not an LCQ terminator. Keep unsupported-instruction diagnostics from
  captured bits/identity: `direct/mod.rs::unsupported_exit` currently refetches
  live memory and must not be reused that way.
- **Semantics and runtime exits.** Move the lowering in
  `direct/compiler/a64*.rs` and typed operations in `direct/slow.rs` onto the
  shared state/effects and native helper boundary; remove their NativeContext
  signatures. Reuse `fp_env` ownership, including cumulative FPSR completion
  and host FP restoration; FPCR writes terminate the fragment and select the
  next execution key. Preserve per-thread exclusive reservations and existing
  explicit clear/teardown boundaries, not one reservation per fragment.
  Preserve SVC/BRK attribution, scheduler events, timer reads, guest LR and
  loader-return behavior through canonical exits. `direct::run_slice` currently
  accumulates exit progress without consuming `instruction_budget`; step 6
  must connect the real slice to PollBudget, not copy that loop unchanged.
- **Backend and faults.** Keep the existing process-wide `Checked` versus
  `LinuxDirect` choice (`memory/src/direct.rs`, runtime builder); Checked uses
  explicit typed accesses under the new execution ABI. LinuxDirect does not
  inherit `a64_memory.rs`'s per-access checked branches or eager fault-state
  checkpoints. Preserve required alignment and compound-access semantics using
  native operations or the specified instruction-specific cold handling.
  Reuse `cpu-direct-memory`'s signal capture, landing stack and machine-context
  restoration, but separate activation from the old context-tail invocation.
  Attribute JIT faults through the active Invocation's fixed directory and
  final physical maps. `CapturedFault` currently exposes only an old site and
  address; step 4 must expose the captured native PC/register/FP state needed
  for reconstruction and connect canonical escape to the new gateway without
  skipping Rust resource cleanup. Adapt the interpreter's distinct fixed-stub
  consumer when shared support changes; do not retain an old JIT registry.
  `DirectFaultResolution` currently has Retry/Fault/Fatal only: add the typed
  non-RAM completion route. Ordinary RAM repair retains the epoch and retries
  the same instruction; executable writes or transitions needing Closed first
  reconstruct and exit, then perform the mutation without replaying completed
  subaccesses. Replace PC/address-only retry counting with the specified
  mapping-generation/observation-epoch check.
- **Memory transition order.** `ExecutionGate` already excludes mapping
  mutation from active execution, but its notifier only requests preemption
  when readers exist. It has no affected-page payload, mandatory completion or
  compiler-admission handshake. Host overwrite paths also publish invalidation
  after copying bytes. Keep the engine-neutral memory gate, but add a mandatory
  pre/post mutation handoff to the existing JIT coordinator, including when no
  vCPU is running. Discover affected ranges/physical pages, release memory
  locks, register exact work and close admission; drain execution, remove
  affected roots, then mutate under the memory authority and commit its
  invalidation records before reopening. Revalidate targets if discovery lost
  a mapping race. Callers must not hold a memory/gate lock or their own
  Invocation while waiting; a lease holder must not await admission while
  preventing the transition from draining. Concurrent requests join the same
  coordinator, which cannot reopen until pending mutations finish. Apply this
  to executable-content writes through CPU/host/device aliases, mapping and
  permission changes, and GPU/observation transitions requiring quiescence;
  ordinary RAM stores and monotonic first-write repair do not close the JIT.
  A cursor notification after mutation is not the handshake, and
  a dirty-page observer alone does not intercept every executable write.
- **Validation and readiness.** Reuse `direct/tests.rs`'s existing platform
  catalog tests, integer/FP/SIMD/memory differential cases, exclusive tests and
  `tests/differential.rs`. Port semantic assertions; replace obsolete BFS,
  HCQ-promotion and lookup-shape assertions. Reuse delivered-signal tests in
  `cpu-direct-memory` for host context restoration, then exercise final JIT
  maps with actual signals in step 4. No new coverage manifest or runner is
  needed. The missing snapshot and transition/fault interfaces are work within
  steps 2–5, not external blockers; no fork change has been identified by this
  inspection. Native Arm hardware remains a validation gap, not a prerequisite
  for step 2.

### Step 2 outcome

- `cpu/src/memory/capture.rs` supplies owned `InstructionImage` capture and cold
  revalidation for both memory authorities. Production capture holds the
  existing execution gate exclusively and the mapping lock only for the bounded
  copy/observer arming: active native slices must reach a safepoint, but lowering
  and compile waiting hold neither lock. Canonical page observations cover
  content, visibility and direct-write dirty epochs across physical aliases.
  Device reconciliation runs outside both locks and restarts capture. No
  permission lookup or per-store generation is added to native RAM accesses.
  Step 5 additionally coordinates observer arming with the bound engine;
  already-armed captures remain read-only.
- `cpu-jit/src/lcq.rs` captures only the demanded straight-line fragment, reuses
  decoder output and the shared system-support classifier, and retains a valid
  prefix plus the exact terminal fetch fault. `Compilation` couples that image
  to the winning reader's claim and pre-emission CodeVersion; native lowering
  and compiler-context integration are step 3.
- `lifetime/compile.rs` uses the existing dispatch slots and cold state mutex
  for an atomic compare-and-replace of full claim identities. Same-key waiters
  borrow an inactive reader; different-key owners can work concurrently. Drop,
  closure, failure and shutdown release/wake exact work, including closure
  between slot reservation and claim acquisition. Empty abandoned slots are
  actually reused; stale owners cannot cancel replacement claims.
- Production publication is not enabled by `image_is_current` alone. Step 5
  must join executable writes and observation transitions to JIT Closed and
  connect final memory revalidation to publication before step 6 cutover.
  Wasmtime remains unchanged.

Validation: 17 new capture/ownership tests pass on x86-64 and AArch64 QEMU.
`cargo test --offline -p nixe-memory -p nixe-cpu -p nixe-cpu-direct-memory
-p nixe-cpu-jit --quiet` passed; the final JIT library rerun passes 251 tests.
`cargo test --offline -p nixe-runtime --lib --quiet` passes 63 tests. The
[documented AArch64 command](../../aarch64-tests.md) passes 141 tests with the
legacy `direct::` tests excluded. Formatting and `git diff --check` pass.
Clippy reports only two pre-existing `type_complexity` warnings in
`memory/src/range.rs:260,294`; the affected-crate `--all-targets` run with
`-D warnings -A clippy::type-complexity` passes. Native Arm evidence is still
pending.

### Step 3 progress

- `cpu-jit/src/lcq/compiler.rs` lowers demanded integer/control, register-only
  SIMD, the connected scalar FP operations and inline system-register fragments
  with `opt_level=none` and
  `single_pass`. Canonical ingress and source-local exits
  use final allocated bindings/spills, dirty state, deferred NZCV recipes,
  destination PC, source instruction/edge kind and the reserved CodeVersion.
  Owned output is installed in the real executable cache and published through
  Task 2; execution tests use protected admission and the native gateway.
- Integer and flag semantics moved from `direct/compiler/a64.rs` and the old
  compiler into shared `lowering.rs`; the original copies were removed. Both
  callers use static dispatch, with no legacy context or JITModule dependency
  in the new lowering. Existing legacy semantic tests still pass.
- Register-only SIMD semantics moved from `direct/compiler/a64_fp_simd.rs` into
  shared `simd_lowering.rs`, removing the old copies. All 34 supported variants
  use vector SSA inputs/outputs and final native maps without acquiring guest
  FP ownership. This includes bit transfers, integer lane operations and
  conditional FP selects, not FP arithmetic or SIMD memory. LCQ selects the
  real host ISA capabilities; x86 without SSSE3 emits inline lane permutations
  instead of backend libcalls. No new host CPU requirement is introduced.
- `lcq/compiler/system.rs` lowers NZCV, TPIDR_EL0/TPIDRRO_EL0, FPCR reads,
  profile constants and supported no-op hints. TPIDR writes stay in SSA until
  canonical writeback; NZCV reads consume shared lazy recipes. MSR NZCV masks
  reserved bits immediately through shared lowering, fixing the old compiler's
  MSR-to-MRS case as well. Unsupported system operands retain exact native exit
  attribution rather than becoming an internal missing-lowering error.
- Scalar FCMP/FCMPE (S/D, register or zero) use guarded native comparisons for
  finite normal inputs and zero. These comparisons require no guest FP
  activation: rounding control does not affect their result and they produce
  no FP status. Shared predicates and ordered-result lowering moved out of the
  old compiler into `fp_lowering.rs`, removing the original copies.
  NaNs, infinities and subnormals exit with typed `FpCompareOperation` operands
  and exact PRE-state; FCCMP/FCCMPE use the same boundary under the existing
  exact-lowering policy. `lcq/fp.rs::complete_compare` uses the shared exact
  semantics after gateway FP completion and epoch release. Success commits
  NZCV/FPSR and demands PC+4; a trap preserves PRE-state and PC. A false
  conditional comparison writes its NZCV literal without evaluating FP inputs.
  Production dispatch of these completions remains step 6.
- Scalar FRINTN/P/M/Z/A/X/I (S/D) and the existing exact-only directional and
  fixed-point FP-to-GPR conversions use typed `FpRoundOperation` and
  `FpToIntegerOperation` exits. Capture stops at unconditional exact boundaries
  without fetching their successor. `complete_round` / `complete_to_integer`
  reuse the existing exact primitives after FP completion and epoch release;
  they preserve PRE-state on traps and continue at PC+4 on success. Scalar
  writes clear inactive vector bits, Wn writes zero-extend, and XZR/WZR discards
  only the result, not status or exceptions.
  AArch64 retains guarded native FRINTN/P/M/Z for normal/zero inputs, without
  guest FP activation or inexact-status production; other cases use the typed
  exact exit. x86 keeps all FRINT forms exact under the existing status policy.
  Native round lowering moved from the old compiler into `fp_lowering.rs`;
  the shared policy accepts the selected emission ISA for cross-target checks.
  Ordinary FCVTZS/FCVTZU still await their guarded native port; they are not
  routed through an exact-only substitute.
- Scalar FADD/FSUB (S/D) now run natively after operand and FPCR eligibility
  guards. `lcq/compiler/activation.rs` uses the existing fork's exit/entry maps
  for one internal activation/continuation pair per fragment. Its generated
  leaf installs the guest FP environment only on first use, preserving an
  already active segment and its sticky status. Transfers reuse the fast-link
  parallel-copy engine and final physical allocations, including spilled
  vectors and narrow lazy-carry operands. No Rust call, host-stack adjustment,
  canonical guest checkpoint or epoch transition occurs at activation.
  The shared FP owner supplies the host-control encodings; later exit maps
  declare pending host FPSR, which the existing gateway merges before return.
  Exceptional inputs/modes use typed `FpAddOperation` completion and demand
  PC+4 on success. x86 tiny cancellation under FZ also uses that exact edge:
  x86 FTZ would otherwise add IXC where Arm requires only UFC. The eligibility
  correction is shared with the old compiler, with a regression test there.
- Scalar FSQRT and FCVT S↔D use the same activation/continuation and typed
  `FpUnaryOperation` exact completion. Eligibility predicates and result CLIF
  moved into shared `fp_lowering.rs`, removing the legacy copies. Normal inputs
  use native operations; negative square roots, special inputs and unsupported
  FPCR modes retain precise PRE-state and exact helper semantics. x86 also
  excludes nonzero D→S inputs below the minimum normal single: its tiny-result
  rounding/FTZ status differs from Arm. AArch64 keeps those conversions native.
  Cross-target tests exposed an existing exact-conversion bug: FZ must flush
  before rounding and contribute UFC without IXC. The shared CPU primitive is
  corrected, with explicit expected-result/status regression coverage; both
  compiler drivers and the interpreter consume that correction.
- Scalar FDIV (S/D) uses the existing activation and a typed `FpDivideOperation`
  exact boundary. Nonzero finite normal divisors and normal/zero numerators
  execute natively under supported FPCR modes. On x86 a conservative exponent
  difference guard excludes possible tiny quotients before host status is
  produced; AArch64 keeps tiny quotients native. The domain predicate is shared
  with the old compiler, replacing its divisor-only guard. Exact completion
  uses the existing division primitive, preserves PRE-state on traps and
  continues at PC+4 on success. Its existing FZ ordering bug is also corrected:
  tiny quotients flush before rounding with UFC alone, even if gradual rounding
  would produce a normal result or zero. Scalar/vector exact division share
  that primitive; no new semantic implementation or Wasmtime change is added.
- Scalar FMUL/FNMUL (S/D) share their native result lowering and x86 tiny-product
  guard with the old compiler, removing its original result-lowering copy.
  FNMUL negates the rounded product, preserving directed rounding. Normal/zero
  operands under supported FPCR modes use native execution; x86 conservatively
  excludes possible tiny products by exponent sum, while AArch64 keeps them
  native without that extra guard. Typed `FpMultiplyOperation` completion uses
  the existing exact primitive and the same PRE-state/trap/PC+4 contract.
  The shared exact sum/product packer now applies FZ before rounding with UFC
  alone; the correction also benefits its existing addition/FMA callers.
  Vector multiplication lowering is still pending, not silently routed through
  scalar or exact-only substitutes.
- Scalar FMADD/FMSUB/FNMADD/FNMSUB (S/D) share one result lowering with the old
  compiler: architectural input sign changes precede a single CLIF `fma`.
  On x86 a conservative exact-product/addend exponent bound excludes possible
  tiny cancellation before host status is produced, without evaluating an
  intermediate FP product. AArch64 needs no such guard. `FpFusedOperation`
  provides exact completion for special inputs/modes. The shared exact helper
  now unpacks inputs before NaN selection, prioritizes addend/rn/rm NaNs,
  applies input sign changes to NaNs too, and handles infinity-times-zero with
  a quiet NaN addend using the default NaN and IOC. Native Arm instructions
  independently check those semantics under QEMU.
  LCQ checks the selected ISA's AVX/FMA capabilities; baseline x86 emits only
  the typed exact exit, never a frameless backend libcall or unfused operations.
  The host policy also stops demanded capture there on hosts without AVX/FMA.
  This does not introduce a new host requirement or Wasmtime change.
- Scalar SCVTF/UCVTF W/X→S/D now use shared result lowering and typed
  `IntegerToFpOperation` completion, removing the original legacy conversion
  copy. W sources truncate before signedness is interpreted in both paths;
  XZR/WZR is zero and V31 remains a destination. Supported FPCR modes activate
  native FP once; unsupported modes complete exactly with atomic IXE traps.
  Tests cover rounding, large unsigned X values, poisoned W upper bits,
  dirty sources, lazy carry, exact continuation and pending FPSR. With the local
  observable-FP fork change, the overwritten-result regression also passes on
  x86 and AArch64/QEMU. Optimized constant-input tests retain guest rounding.
- Scalar FCVTZS/FCVTZU S/D→W/X now use shared native range predicates and
  value lowering, replacing the legacy copy. Integer-bit guards admit normal
  in-range values, signed minima and either zero without executing FP before
  activation. Exceptional operands/FPCR modes retain the existing typed
  `FpToIntegerOperation` completion with precise traps. W writes zero-extend;
  discarded and overwritten results still accumulate native FPSR. Negative
  unsigned fractions and just-below-signed-minimum fractions use exact
  completion. No new fork edits are needed beyond observable FP effects.
- Advanced SIMD SCVTF/UCVTF now share value lowering with the legacy compiler,
  whose duplicate conversion code is removed. Packed forms convert only active
  lanes; single-element SIMD forms use scalar CLIF, with higher destination bits
  cleared. `VectorIntegerToFpOperation` retains canonical PRE-state for exact
  completion: IXE in any active lane commits no part of the instruction. Native
  conversions retain FPSR even when their result is overwritten. A focused x86
  regression exposed `-0.0` from Cranelift's unsigned I64X2→F64X2 bias sequence
  under round-down. The local fork now clears the result sign for that sequence
  (and its widened-I32 specialization) in observable-FP functions only; all
  nonzero results are nonnegative, so rounding and status are unchanged. No
  scalarization or runtime helper is added to packed conversions.
- Vector FDIV 2S/4S/2D now shares operand normalization, integer-only guards and
  packed value lowering with the legacy compiler; its old lowering copy is
  removed. Inactive 2S lanes execute exact 0/1, so poisoned upper bits neither
  force exact completion nor contribute status. Lane predicates combine before
  a single reduction; x86 excludes potentially tiny quotients for Arm-compatible
  tininess/FZ status, while AArch64 keeps those normal-input divisions native.
  `VectorFpDivideOperation` supplies atomic exact completion and precise traps.
  Native FPSR survives an overwritten result and a later exact edge. This port
  needs no further fork modification.
- FMUL-by-element 2S/4S/2D shares operand selection, packed value lowering and
  integer domain predicates with the legacy compiler, removing its old copy.
  The selected element comes from the full Rm vector, including S[2]/S[3] for
  2S; unselected elements and inactive Rn lanes cannot affect native eligibility
  or status. x86 guards potential tiny products for Arm-compatible tininess/FZ;
  AArch64 retains native normal-input multiplication. `VectorFpMultiplyElementOperation`
  provides atomic exact completion. Overwritten products still contribute FPSR.
  No additional fork modification is needed.
- FPCR/FPSR writes and FPSR observations cut at an architectural boundary.
  Their typed `FpSystemOperation` exit retains PRE-instruction state and PC;
  the existing gateway merges pending host FPSR and restores the caller first.
  `lcq/system.rs::complete_fp` then reads/replaces canonical state and advances
  PC, before the next demanded-key admission. Tests seed real host FP status
  before native execution to verify this order, including XZR and dirty source
  registers. Production wiring of these completions remains step 6.
- Stateful system operations use typed `RuntimeSystemOperation` exits and
  `lcq/system.rs::complete_runtime`: timer reads, scheduling/event hints,
  barriers, CLREX and IC/DC maintenance. Capture stops at these architectural
  exits without fetching a successor. Completion borrows the existing memory,
  timer, events and persistent exclusive monitor after FP restoration, epoch
  release and memory-lease release. PC advances only on success (including
  scheduling); cache faults retain their original DataAccessFault and source
  PC. No guest-byte refetch or legacy helper context is involved. SEV returns
  the existing scheduler request; process-wide delivery remains the runtime's
  responsibility. Step 5 must still connect memory maintenance to JIT closure,
  and step 6 must call these completions from the production execution loop.
- A valid prefix followed by a fetch fault executes and dispatches to the
  pending PC; the failed fetch is not cached as code. Stale captured output
  is rejected. Unported decoded instruction families fail explicitly and
  cannot silently enter an interpreter or old-JIT fallback.
- **Local closure review:** decoded non-memory catalog fixtures compile using
  a reused compiler on both encoders, with valid final maps and attributed
  exits. Existing execution comparisons cover state, branches, lazy flags,
  native FP and exact completions. Runtime-helper success tests now execute
  the next demanded fragment and consume preserved NZCV; failure tests retain
  PRE-state and fault identity. No additional non-memory lowering gap was found.
- **Step 3 complete:** the functional exit criterion is satisfied with the
  local fork override. The maintainer has deferred the pinned-fork handoff;
  do not commit path-resolved dependencies. Memory/faults,
  concurrent mutation/publication coordination and production runtime cutover
  remain steps 4–6 respectively.

Validation: seventeen new tests cover interpreter comparisons for integers, deferred
flags, conditional branches, calls/returns, final spill maps on both encoders,
unsupported attribution, prefix fetch failure, stale rejection and compiler
reuse. SIMD coverage includes catalog fixtures with varied inputs and preserved
FPCR/FPSR, conservative x86 emission/execution without SSSE3, and mixed integer/
vector pressure with deferred NZCV and partial writes. Vector spills are checked
and executed on baseline x86; both encoders validate final state maps. System
tests cover SSA reads/writes, reserved NZCV bits, lazy flag consumers, unsupported
operands and FP observation/replacement after actual host status accumulation.
Runtime-helper comparisons include pending events/interrupts, exclusive-monitor
preservation, exactly one timer read per operation and restored FP. Cache tests
check one successful invalidation and unchanged architectural state on failure.
Four FP comparison tests cover ordinary and exceptional inputs, signaling NaNs,
signed zero, FZ, enabled exceptions, conditional predicates, dirty vector inputs
and lazy PRE-NZCV. Final maps are checked on both encoders; successful exact
completion resumes through a real demanded PC+4 entry and consumes the new flags.
Four rounding/conversion tests cover all supported scalar FRINT forms, signed/
unsigned directional and fixed conversions, S/D and W/X widths, FZ/DN and
rounding modes, traps, aliased destinations, discarded results and inactive-bit
poisoning. Pending native status survives exact completion; both encoders check
PRE-state maps and the selected ISA's rounding shape.
Six addition/activation tests cover all sixteen supported FPCR encodings,
S/D addition/subtraction, overflow, signed zero and tiny cancellation. They
exercise full register pressure and allocated spills on both encoders, native
to exact to native continuation, preserved lazy carry, pending FPSR, existing
active segments and caller restoration. The production-legacy FP regression
also checks the shared x86 FZ-domain correction.
Three unary FP tests cover FSQRT and S↔D conversion results/status under all
sixteen supported FPCR modes, exception enables, NaNs, subnormals, signed zero,
tiny/overflow rounding boundaries, aliased destinations and inactive-bit
poisoning. They exercise lazy carry and demanded exact continuation, and check
pending FPSR and a single activation across FADD followed by FSQRT on both
encoders. Legacy conversion tests cover the shared tiny-result eligibility.
Three scalar division tests cover S/D, all native FPCR modes, zero divisors,
NaNs, infinities, subnormal operands, overflow, tiny quotients, enabled traps,
aliased destinations and lazy carry across demanded continuation. Both
encoders validate final maps and one activation across consecutive divisions;
native IXC survives a subsequent exact divide-by-zero completion. Explicit CPU
result/status assertions and legacy regression cases cover tiny-result FZ
ordering independently of interpreter/JIT agreement.
Three scalar multiplication tests cover FMUL/FNMUL S/D, directed rounding
before negation, all native FPCR modes, special operands, overflow, tiny
products, traps, aliases, lazy carry and continuation. Both encoders check
final maps and one activation across operations; native IXC survives an exact
invalid-operation completion. Explicit CPU and legacy regressions cover FZ
ordering and FNMUL rounding.
Two fused-operation tests cover all four S/D variants, single-rounding
cancellation, intermediate-product overflow, tiny results, traps, aliases,
pending FPSR and baseline-x86 exact completion. Both encoders check the native
boundary maps. A separate AArch64-only instruction comparison checks the exact
provider's NaN priority/signs and FZ; explicit CPU results and legacy regressions
also cover those corrections.
With the temporary local fork override, the overwritten-result regression is
green on both execution targets. Two additional execution tests exercise dead
FADD/FSUB, FMUL, FDIV, FSQRT and FCVT results, and constant SCVTF inputs with
all four rounding modes, under `none`/`single_pass` and `speed`/`backtracking`.
Two truncating-conversion tests cover S/D→W/X signed/unsigned boundaries,
all sixteen native FPCR encodings, enabled traps, W zero-extension, discarded
results, poisoned inactive bits, lazy carry, exact continuation and pending
FPSR. Both encoders check one activation and precise exact-edge maps. An
AArch64-only test compares the exact provider's result/status against the eight
FCVTZS/FCVTZU instruction forms, including NaNs, infinities, negative unsigned
fractions, saturation and FZ.
Three SIMD integer-conversion tests cover packed/scalar forms, signedness,
all sixteen native FPCR encodings, baseline-ISA emission, inactive lanes,
V31, aliases, atomic IXE traps, lazy carry and demanded exact continuation.
An overwritten result retains IXC at a subsequent exact edge, whose final
maps are checked on both encoders. The unsigned-zero regression exercises
the x86 fork correction. An AArch64-only test compares the shared exact
provider with ten Arm SCVTF/UCVTF instruction forms, including inactive lanes.
Two packed-division tests cover 2S/4S/2D, all sixteen native FPCR encodings,
enabled traps, tiny quotients, overflow, NaNs, subnormals, zero divisors,
aliases/V31, dirty sources and lazy carry across native/exact continuation.
Baseline-ISA execution keeps ordinary inputs native even with poisoned inactive
2S lanes. Both encoders check one activation and precise maps after an
overwritten packed result; pending IXC survives exact divide-by-zero completion.
An AArch64-only test checks all three instruction forms against the shared
exact provider. Legacy regressions cover the shared tiny-result guard and the
updated per-operation CLIF shape without duplicate cold protocol.
Two multiply-element tests cover all ten shape/element selections, including
2S reads of upper Rm elements, all sixteen native FPCR encodings, exceptional
operands, tiny products, overflow, traps, aliases/V31 and lazy carry across
native/exact continuation. Both encoders check precise maps and one activation;
an overwritten product's IXC survives a later exact invalid operation. An
AArch64-only instruction comparison independently checks the shared exact
provider. A legacy regression covers the shared tiny-product guard. The
catalog closure test compiles every decoded non-memory fixture on both
encoders without relying on the list of already-ported FP operations as a filter.
The [documented AArch64 command](../../aarch64-tests.md), with that override,
passes 203 tests with legacy `direct::` tests excluded; the focused legacy
tiny-product/CLIF-shape regressions and the strengthened helper-continuation
test also pass on AArch64. Native x86-64 passes
310 library and four integration tests. The fork's 242 codegen and 44 reader
unit tests pass, including retention/order in both encoders, optimizer opacity,
compare/select pattern isolation, ordinary integer folding, context reuse,
incompatible FP policy rejection and CLIF text round-tripping.
The CPU library passes 78 tests; the interpreter package passes 117 tests
(three pre-existing ignored tests), including the shared FP semantics.
`cargo test --offline -p nixe-runtime --lib --quiet` passes 63 tests. Formatting and
`git diff --check` pass; Clippy with `-D warnings -A clippy::type-complexity`
passes for both compilation targets (the same pre-existing memory warnings
noted above). QEMU evidence does
not replace native Arm validation.

## Specification reading by step

- Steps 1–2: [LCQ](spec.md#lcq-baseline-compiler),
  [keys](spec.md#keys-and-dispatch-publication),
  [direct memory](spec.md#direct-memory-and-fault-authority) and
  [publication](spec.md#compilation-and-publication-pipeline).
- Steps 3–4: [native ABI](spec.md#native-fast-chain-abi),
  [helpers](spec.md#helpers-and-architectural-boundaries) and
  [fault retry](spec.md#native-fault-retry).
- Steps 5–7: [invalidation](spec.md#code-and-mapping-invalidation),
  [coordinator](spec.md#maintenance-coordinator),
  [control budget](spec.md#control-budget-and-functional-sampling) (sampling
  remains disabled), [reclamation](spec.md#epoch-reclamation) and
  [migration](spec.md#migration-map).
