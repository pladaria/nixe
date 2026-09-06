# Task 1 implementation plan

Status: steps 1–4 are complete. Nixe is pinned to the published fork and its
native adapters consume real final register/spill maps. Independently compiled
fragments execute through empty and cyclic bridges and the Nixe gateway, with
exact canonical state, lazy NZCV and FP completion. Integration passes on native
x86-64 and emulated AArch64. Step 5 remains: consolidate the Task 1 exit evidence
and explicitly account for the still-unavailable native AArch64 host. Production
publication/reclamation and the existing JIT's cutover belong to later tasks.

This is a working checklist for [Task 1](spec.md#task-1-freeze-and-prove-the-executable-abi-and-state-contracts),
not another specification. Follow [CONTRIBUTING.md](../../../CONTRIBUTING.md).
The task's scope and exit criteria remain in `spec.md`. Update this file in
place, without session logs. Agree architectural changes with the maintainer
and update the affected specification section; keep local implementation
details in code. This plan may be removed when the task is complete.

## Steps

- [x] **Inspect the existing implementation and backend.** Locate Nixe's
  lowering, architectural state, helpers, FP handling and code-output path.
  Inspect the local Wasmtime fork on branch `nixe`, using the
  [specified release base](spec.md#wasmtime-fork-and-working-branch).
  Identify reusable code and the backend changes needed for both host ISAs.
  Check available native test hosts and the current instruction-coverage
  baseline. Record concrete blockers, not speculative infrastructure needs.

- [x] **Prove the risky backend mechanisms first.** Introduce the minimum
  shared ABI definitions needed to exercise reserved registers,
  NativeFrame-relative spills and prologue-free entries. Generate small
  fragments for both targets and check the emitted instructions and spill
  extent. Also establish how physical entry constraints, final state maps and
  selected labels can be represented before committing to the full integration.
  Execute both external entries of one optimized body, including loops, without
  relying on computations or allocator edits in a skipped analysis root.
  If this requires replacing the allocator or machine backend, stop and discuss
  the architectural change rather than implementing a fallback.

- [x] **Implement the shared state contracts.** Complete the identities,
  counters, NativeFrame, dispatch payload and entry/exit contracts named by
  Task 1. Implement or reuse one use/def, partial-write and liveness analysis
  for both tiers, accounting for helpers, faults and architectural observations.
  Add focused tests for preservation and safe omission of guest values.

- [x] **Complete the backend and native boundary.** Implement the remaining
  fork hooks for physical state maps, selected labels, patchpoints, relocations,
  fault offsets, caller-owned output and required landing instructions. Connect
  the gateway, canonical/fast entries, canonical exit and cycle-safe bridges
  to these real outputs. Preserve caller and guest FP state and lazy flags.
  Keep production epoch/reclamation machinery in Task 2; any temporary support
  for the isolated proof stays test-only.

  Completed: final entry/exit maps now drive canonical loads, generic bridges,
  dirty writeback and lazy NZCV materialization through the actual gateway.
  Tests compile both units independently with every source/target allocator
  combination and execute empty and integer/SIMD-cycle edges. Allocation-chosen
  entry/exit spills also execute under pressure. Instruction-attached prefault
  maps translate to the shared contracts at the actual trap offsets; the fork's
  native x86-64 fault tests cover capture, including compound atomics. Shared FP
  ownership completes pending FPSR after software writeback and restores the
  caller. Both encoders pass; execution is native x86-64 and QEMU AArch64.
  Protected dispatch and epoch publication remain in Task 2; the test-owned
  signal capture is not the production fault/retry path.

- [ ] **Run the native proof and finish integration.** Exercise empty and
  register-cycle bridges, randomized register pressure, the fixed spill
  partition and exact state reconstruction. Inspect both targets' output for
  the Task 1 exit criteria, and execute on available native hosts. Record any
  missing native execution explicitly; encoding or cross-compilation is not
  execution evidence. Use conventional tests, not a new testing framework or
  benchmark runner. Verify the final tested fork pin and lockfile (initial
  integration pin is already in place). Remove superseded task-local scaffolding;
  identify any test-only proof support that must remain until the later production
  cutover.

Test each step as it becomes executable; the final step consolidates evidence,
not the first opportunity to discover integration failures. Full code-cache
management, production linking, LCQ cutover, HCQ and benchmarks are outside this
task.

## Inspection findings

### Nixe: reuse and replacement points

- [CPU frontend](../../../crates/cpu/src/decode/a64/mod.rs): `normalize`,
  `A64Instruction` and `patterns()` are the existing typed instruction contract
  and coverage catalog. The spec's `NormalizedA64` refers to this role; do not
  introduce a second normalized representation. Reuse
  [A64State](../../../crates/cpu/src/state/a64.rs) and the shared
  `crates/cpu/src/semantics/` operations.
- [Compiler](../../../crates/cpu-jit/src/direct/compiler.rs):
  `DirectCompiler::compile`, `create_module`, `compile_gateway` and
  `tail_signature` currently use JITModule ownership and `CallConv::Tail`.
  `CraneliftTranslator` and its `compiler/a64*.rs` emitters are the lowering
  reuse points; `emit_static_exit`, `emit_dynamic_exit` and `commit_state`
  are the old boundary to replace. Instruction effects now live in the shared
  [analysis](../../../crates/cpu-jit/src/analysis.rs); `register_load_blocks`
  consumes its observation-aware liveness. `dirty_states_at_entry` still serves
  the current boundary. Exact post-allocation maps remain a backend-integration task.
- [FP environment](../../../crates/cpu-jit/src/fp_env.rs): the shared
  HostFpState implementation owns `begin`, `ensure`, `suspend`, `resume`, `end`
  and `finish`. NativeFrame uses it directly; `direct/fp_env.rs` now contains
  only the current compiler's typed veneers. Reuse typed operations in
  [slow.rs](../../../crates/cpu-jit/src/direct/slow.rs); replace their old ABI
  veneers, not the instruction semantics.
- [Fault transport](../../../crates/cpu-direct-memory/src/lib.rs):
  `CapturedFault`, `FaultDisposition` and `WorkerFaultContext` are existing
  capture/retry integration points. The JIT's `record_direct_fault_state` and
  `compile_fault_sites` currently record access/source-location metadata,
  not the complete physical prefault state required by the new ABI.

### Fork: concrete intervention points

Paths below are relative to `/home/pladaria/projects/wasmtime/cranelift/codegen/`.

- Register allocation: `src/isa/x64/abi.rs::create_reg_env_systemv` and
  `src/isa/aarch64/abi.rs::create_reg_env`, selected by `get_machine_env`.
  Stock pinning excludes only r15/x21. Nixe also needs r13/r14/r11 and x19/x20
  excluded; x16/x17 are already backend scratch. Audit fixed-register uses
  as well as allocator pools so implicit instructions cannot corrupt pins.
- Frame/spills: `src/machinst/abi.rs::Callee` owns `compute_frame_layout`,
  `gen_prologue`, `gen_epilogue`, `gen_spill` and `gen_reload`. Both ISA
  `abi.rs` files translate `StackAMode` and generate stack-address operations.
  These paths must agree on NativeFrame addressing; removing the prologue
  alone would leave invalid spill, stack-slot and helper-call behavior.
- Entries/state maps: `src/machinst/lower.rs` initializes function arguments
  at one entry; `src/machinst/vcode.rs` presents one entry to regalloc2.
  Fixed operand constraints exist in `src/machinst/reg.rs`, but selected
  external entries still need correct liveness/dominance and physical inputs.
  `compute_value_labels_ranges` is debug information, drops some ranges and,
  without `unwind`, skips spilled values; it is not an exact state-map API.
- Output: `src/context.rs::Context::take_compiled_code` already transfers owned
  output; `CompiledCodeBase` and `MachBufferFinalized` expose bytes, relocations
  and traps. Reuse these. `bb_starts` contains filtered machine-block offsets,
  not stable selected guest-entry identities. Extend label/patch/fault export
  at emission in `src/machinst/vcode.rs` and `buffer.rs` as needed. Landing
  instructions have an existing `MachInst::gen_block_start` hook.

## Current handoff

- Inspected Nixe code revision: `1dc3c343d92fb35e5c3ff6d334f5b7f9ced4b81e`
  (initial inspection). The Wasmtime checkout and published `origin/nixe` are at
  `e2a984d96678207094c0fc50057c8b6bcfd68715`,
  based on release v48.0.1 at `7bac2c2775808aaec5d4aa5627a5e447b51102cf`.
  Nixe's five Cranelift dependencies now use this exact Git `rev` (`0.135.1`),
  with the matching `Cargo.lock`; no local path override or floating branch is
  required. Test-only features enable both target encoders.
- Coverage baseline: the above Nixe revision's `patterns()` plus actual
  family lowering and `region.rs::system_instruction_supported`, not every
  recognized encoding. Existing catalog tests cover scalar/control,
  memory/system and FP/SIMD families; CLREX and exclusive pairs have additional
  tests. ERET/DRPS remain recognized-unsupported, and system operands have
  explicit support restrictions. This inspection does not create a new catalog.
- Checks passed with Rust `1.97.1` on native Linux x86-64:
  `cargo test --offline --locked -p nixe-cpu-jit` (154 unit tests, 2 dependency
  boundary tests and 2 differential tests).
  `cargo check --offline --locked -p nixe-cpu-jit --target aarch64-unknown-linux-gnu`
  and `cargo clippy --offline --locked -p nixe-cpu-jit --lib --no-deps -- -D warnings`
  also passed. Strict Clippy including dependencies hits existing
  `type_complexity` warnings in `crates/memory/src/range.rs`.
  Including `--tests --no-deps` passes with
  `-D warnings -A clippy::field_reassign_with_default` (the latter allows the
  existing ABI/analysis test warnings). These checks cover the current JIT,
  shared contracts, native adapters and actual fork/gateway integration.
- Hosts: local native x86-64 and installed AArch64 Rust target confirmed. No
  native AArch64 host identified in repository configuration; arrange one with
  the maintainer before claiming native AArch64 validation. Encoder tests can
  run locally with Cranelift's test-only `x86` and `arm64` features; production
  uses `host-arch`.
- Implemented in the fork: opt-in `enable_nixe_abi`, required register pools,
  prologue-free leaf fragments, external-frame spills/reloads and both explicit
  and folded stack-slot accesses. `src/nixe.rs` defines the 2 KiB/16 KiB bounds;
  this initial layout places the spill area at NativeFrame offset zero.
  `MachBufferFrameLayout::nixe_frame_size` reports the bounded allocation extent.
  System arguments, calls, returns, dynamic frames and debug value-location
  maps are explicitly unsupported until the real boundary integration exists.
- Backend checks (from the Wasmtime checkout, Rust `1.97.1`):
  `cargo +1.97.1 test --offline --locked -p cranelift-codegen --no-default-features --features std,unwind,all-native-arch,disas --lib --quiet`
  passed 247 tests, with none ignored; the same command with `enable-serde`
  added to the features also passed 247 tests.
  Coverage includes register pools, pressure-generated machine bytes, frame
  limits, unsupported boundaries, marker offsets, allocator constraints and
  the native experiment below. Both Nixe compiler profiles and both target
  encoders are exercised.
  `cargo +1.97.1 test --offline --locked -p cranelift-codegen --no-default-features --features std,x86,arm64,disas --lib nixe --quiet`
  passed 30 tests without `unwind`, with none ignored.
  `cargo +1.97.1 test --offline --locked -p cranelift-reader --lib --quiet`
  passed 43 tests, including CLIF entry constraints and entry/state/exit/fault
  round trips.
  `cargo +1.97.1 clippy --offline --locked -p cranelift-codegen --no-default-features --features std,unwind,all-native-arch,disas --tests --no-deps -- -D warnings`
  also passed, as did strict Clippy for `-p cranelift-reader --lib --no-deps`.
- Representation established by the probes: fixed-def operands can express
  physical entry inputs; `Any` operands and `Output::inst_allocs` expose exact
  register/spill locations without optional debug ranges; zero-byte
  `sequence_point` markers with stable tags retain offsets in hot/cold blocks.
  These are building blocks, not a completed EntryContract/ExitStateMap API.
- Step 4 fork hooks: `nixe_entry(sig, id)` defines all physical inputs together;
  signature returns describe their types, not system-ABI results. Final
  allocation chooses register/spill locations unless
  `Function::nixe_entry_constraints` requests particular integer/SIMD registers.
  Constraints are keyed by entry ID in original result order; `Any` leaves a
  location free. CLIF preserves these as `nixe_inputs` declarations. Validation
  rejects invalid banks, reserved/overlapping registers, wrong arity and missing
  IDs. Unused inputs are explicitly omitted from transfers.
  `nixe_state(id, values)` retains observation operands
  and `nixe_exit(id, values)` terminates in an aligned, initially trapping patch
  unit. `MachBufferFinalized::nixe_states` exports ordered typed locations,
  identities and exact offsets, including entry BTI/CET instructions. A marker
  describes only its own point: later allocator edits invalidate any assumption
  that it also describes a following memory instruction.
  `StateMap::patch_exit` encodes in-range rel32/imm26 branches in owner-supplied
  bytes; it does not allocate islands, synchronize threads or change permissions.
  The x86-64 `enable_nixe_ibt` flag and AArch64 `use_bti` select jump landings.
- Entry-constraint limitation: caller-selected spill offsets and forced-spill
  definitions are not supported. `single_pass`'s `Stack` definition constraint
  generated a post-definition store from an uninitialized register, overwriting
  the supplied input. Native execution caught this despite the allocator checker
  passing; that unsafe option is not exposed. Allocation-chosen entry spills
  remain supported and native-tested under pressure, with exact offsets in the
  final contract. Do not equate this with accepting arbitrary preassigned spill
  contracts or silently add that mode without fixing and proving its semantics.
- Prefault attachment: `nixe_fault_start(id, values)` / `nixe_fault_end(id)`
  bracket ordinary CLIF memory operations in one block. Final memory operands
  retain the old SSA values at the actual fault PC: early uses for precise
  single instructions, late uses for compound operations that can write
  registers before a later fault. `MachBufferFinalized::nixe_faults` exports
  each PC from the existing emitter's trap records, not source/debug markers.
  Malformed spans, `notrap` memory and non-memory traps inside a span are
  rejected; emission asserts if a memory trap lacks allocation-visible state.
  The owner still supplies guest commit semantics and deferred-writeback rules.
- Step 4 regression coverage is in `src/nixe/boundary_tests.rs`: pressure with
  40 integers and 40 vectors, explicit stack slots, aliases, unused inputs,
  independent entries into loops, cold placement, alignment and branch-range
  limits. A native x86-64 adapter initializes fast inputs from the exported map,
  follows the real patched exit and captures registers/spills to verify every
  output. It reuses the existing test-owned invocation adapter; it does not
  prove production gateway, generic cycle bridges, FP ownership or production
  fault retry. Fixed-register constraints are tested alongside 80 live inputs
  and spills, and independently in hot/cold entries sharing a loop. These tests
  enable the allocator checker in addition to inspecting and executing output.
- Cross-fragment regression: `src/nixe/chaining_tests.rs` compiles source and
  destination separately, constraining destination inputs from the real source
  exit map. It patches an empty edge or an explicit three-register cycle in each
  integer/SIMD bank. Both encoders and all source/target allocator combinations
  pass; x86-64 executes 24 cases across three input seeds. Cycle moves use the
  reserved integer scratch and NativeFrame transfer slot, without SP or calls.
  This is a test-owned shuffle, not the generic bridge emitter or canonical
  gateway/exit proof; AArch64 evidence remains encoding-only.
- Native prefault regression: `cranelift/jit/tests/nixe_faults.rs` uses an
  isolated process and real SIGSEGV/ucontext capture, with no production signal
  changes or additional dependencies. Both allocators recover old state in
  72 cases: low/high pressure, aliases, plain loads, both memory instructions
  of an atomic RMW, and load/arithmetic/store sequences. A read-only page faults
  at the atomic write after RAX/temp have changed. The handler returns through
  a test stub; it does not claim to prove Nixe's retry protocol.
  `cargo +1.97.1 test --offline --locked -p cranelift-jit --tests --quiet`
  passed all 10 tests. Strict Clippy for `-p cranelift-jit --test nixe_faults
  --no-deps -- -D warnings` also passed. The existing atomic CLIF filetests
  could not run offline because `block-buffer v0.10.2` is absent from the local
  Cargo cache; no dependency was downloaded or changed.
- Canonical multi-entry support: `nixe::set_entries` declares blocks with no
  parameters and locally defined inputs. Its synthetic selector/root exists
  only for analysis; neither it nor its outgoing critical-edge blocks are
  emitted. LICM cannot move definitions into this non-executable root, but
  still hoists into executable preheaders. IR and machine operand checks reject
  dependencies on omitted definitions. `MachBufferFinalized::nixe_entries`
  exports actual block-label offsets in declaration order, including entry
  allocator edits and any enabled landing instruction, not debug-marker offsets.
- Native regression: `cranelift/codegen/src/nixe/multi_entry.rs` exercises two
  entries into one shared body, 24 integer and 24 vector inputs, spills, loops
  and hot/cold placement. Both compiler profiles pass on x86-64, including the
  original optimized-loop counterexample. One, three and eight independent
  entries also execute correctly; the emitted CFG has no dispatcher/root.
  Both encoders pass; AArch64 still lacks native execution evidence. Tests also
  cover invalid entry dependencies and preservation of valid LICM. The test-only
  SysV adapter replaces terminal traps with returns in a W^X executable copy;
  this is not the final stackless gateway/exit proof.
- Shared contracts: [abi.rs](../../../crates/cpu-jit/src/abi.rs) defines checked
  identities/counters, immutable dispatch payloads, reserved registers,
  entry/exit maps and poll-budget reconciliation. NativeFrame borrows the real
  A64State storage; its uninitialized 16 KiB arena starts at offset zero, with
  2 KiB reserved for transfers. LazyFlags and HostFpState are shared with the
  existing compiler/FP ownership code, not parallel implementations. Exit maps
  distinguish all live physical values (including clean ones) from the dirty
  subset that needs canonical writeback. Atomic publication and reclamation
  remain in Task 2.
- Shared analysis: [analysis.rs](../../../crates/cpu-jit/src/analysis.rs) owns
  the extracted instruction use/def tables and fixed-point CFG liveness.
  Whole X/V registers are tracked with destination reads for preserved partial
  writes; NZCV is tracked per bit. Faults/helpers and other architectural
  observations constrain liveness. Both compiler profiles use this analysis
  for initial register loads; FP lowering and analysis share
  [fp_policy.rs](../../../crates/cpu-jit/src/fp_policy.rs). No second instruction
  IR or old classifier implementation remains. Focused tests cover partial
  writes, flag dependencies, observations, loops, joins and independent entries.
- Nixe transfer implementation: [native.rs](../../../crates/cpu-jit/src/native.rs)
  consumes the shared EntryContract/ExitStateMap and emits x86-64/AArch64 moves.
  Matching inputs emit nothing; closed integer cycles use reserved scratch,
  while SIMD/mixed cycles use bounded transfer slots. Constants, aliases,
  cross-bank moves and partially overlapping spills are covered. Packed NZCV
  participates in the copy graph; matching host-flag contracts are preserved.
  Lazy recipes, canonical NZCV and host flags materialize only the required bits
  for packed targets, before copies can overwrite their operands. Host-to-host
  carry-convention changes flip only carry; identical contracts still emit no
  bytes. Both hosts install packed or materialized values into host flags after
  physical copies. Missing physical values still return explicit errors.
  No publication, final branch, canonical writeback or FP ownership transition
  is hidden in this transfer-only operation.
- Canonical data adapters in `native/canonical.rs` reuse the move encoder and
  NativeFrame's borrowed field pointers, never A64State's Rust layout. Ingress
  loads exactly the target live inputs; writeback stores only dirty live values
  and merges packed NZCV with all 16 dirty-bit masks. Field pointers are reused
  across X/V elements. x86-64 spill transfers borrow RAX with source preservation
  on writeback and delayed RAX initialization on ingress. Lazy/host NZCV is
  materialized before writeback; ingress into host flags works on both hosts.
  Pending host FPSR follows the real
  exit ordering: emit software writeback first, then call NativeFrame::finish_fp
  (suspend_fp before a helper) to collect sticky status and restore the caller
  before general Rust work or epoch quiescence. An unmapped software FPSR stays
  canonical. Collecting before a later software store would lose pending bits.
  The writeback-only API remains usable before helpers. `emit_canonical_exit`
  adds a constant/register/spill destination PC, source version, state-map index
  and reason, then jumps to the invocation's gateway continuation without a
  host call, RET or SP adjustment.
- Shared FP ownership: caller control/status is saved once at invocation entry,
  even for integer-only invocations. Lazy activation and compatible links never
  replace that save or clear pending guest status. Suspend/end/finish restore
  the caller and return sticky guest FPSR; successful resume starts a clean
  segment. Unsupported native FPCR controls are rejected, not silently masked.
  The current compiler and NativeFrame use this single implementation and one
  shared FPCR eligibility policy; host instruction/status logic was moved, not
  duplicated. Four focused owner tests cover real division-by-zero, helper
  suspend/resume, FPCR/FPSR replacement, exact caller restoration and unsupported
  controls. They pass natively on x86-64 and under AArch64 QEMU (replace the
  filter in the command below with `fp_env::tests`). FP completion never clears
  the execution epoch; gateway/lifetime integration still owns that ordering.
- Native boundary tests: thirteen conventional tests, including 128 seeded full-
  register permutations, execute the real emitters with independent assembly
  input/output capture. Canonical tests cover exact live-in loading (unused field
  pointers can be null), selective writeback, every NZCV mask and composition
  of canonical ingress, a nonempty fast transfer and canonical writeback. They
  also execute software FPSR writeback with real pending host division-by-zero
  and shared FP completion, covering register, spill, constant and canonical
  software contributions. They
  check reserved registers and host FP state; transfers also preserve unmentioned
  registers/spills and host flags. Native x86-64 passes. The same tests pass under AArch64 QEMU with:
  `cargo test --offline --locked -p nixe-cpu-jit --lib native::tests --target aarch64-unknown-linux-gnu --config 'target.aarch64-unknown-linux-gnu.linker="aarch64-linux-gnu-gcc"' --config 'target.aarch64-unknown-linux-gnu.runner=["qemu-aarch64", "-L", "/usr/aarch64-linux-gnu"]' --quiet`.
  This is emulated execution, not native AArch64 evidence. Test-only JITModule
  ownership and capture adapters do not implement epochs.
- System-ABI entry/return in `native/gateway.rs` saves all host nonvolatiles once,
  pins frame/arena/poll and establishes an invocation-local exit continuation.
  `enter_protected` requires a caller-saved FP environment and an entry resolved
  under an already published epoch with admission/reachability revalidated.
  It completes software/host FPSR and restores caller FP before result handling,
  reconciles the returned pinned deadline and leaves the epoch active even on
  error. It does not fake Task 2's lifetime coordinator or protected lookup.
  Four additional tests exercise two separately allocated native fragments with
  empty/cyclic transfers, pending FP status, caller restoration, aligned host
  stack, forced/nonforced budget overshoot, dynamic PC and invalid budgets.
  They pass natively on x86-64 and under AArch64 QEMU with the command above.
- Lazy materialization in `native/flags.rs` emits the shared Add/Subtract,
  AddCarry/SubtractCarry, Logical, Packed/Canonical and nested Conditional
  recipes directly, with no helper call or SP use. Borrowed GPRs are saved in
  transfer slots and recipe reads use those saved originals; the packed result
  has a slot excluded from subsequent cycle scratch. One-byte predicates/carry
  consume only the specified byte, including at the end of the spill arena.
  Four tests compare against shared CPU arithmetic for 32/64-bit edges and
  seeded values, exercise all NZCV masks and all 16 host-flag combinations,
  and check carry inversion and preservation through cyclic transfers. The
  two-fragment gateway proof also combines a deferred recipe, cyclic link and
  active guest FP. Native x86-64 and AArch64 QEMU pass; this is not native Arm
  execution evidence.
- Value-to-host ingress uses x16/x17 and MSR NZCV on AArch64. x86-64 uses
  SAHF with a byte addition establishing OF first; RAX is saved/restored in the
  fixed transfer area and r11 is reserved scratch. SAHF's CPUID capability is
  an approved [minimum host requirement](../../host-requirements.md), checked
  by `native::check_host` once in `JitProcess::new`, not on native edges.
  The CPUID check passes on the native host and rejects a QEMU x86-64 CPU with
  `lahf-lm=off`, exercised by `host_requirement_matches_exposed_cpuid`.
  Fast bridges protect the packed value before destructive copies and install it
  afterward; canonical ingress loads it alongside actual live inputs. No helper,
  SP use or allocator-visible register clobber is introduced. Two additional
  tests cover all 16 flag combinations, partial masks, both carry conventions,
  register/SIMD/spill/constant inputs and recipes through cyclic copies. They
  also verify all canonical ingress GPR/SIMD/spill inputs survive installation.
  The native x86-64 and QEMU AArch64 gateway proofs combine deferred-to-host
  conversion, a cyclic link, pending FPSR and canonical return. Tests pass on
  both; native AArch64 execution remains unavailable.
- Final-map integration: `native/backend.rs::AllocatedBoundary` translates the
  fork's ordered, typed allocations into shared bindings and lazy recipe/PC
  operand locations, checking widths, reserved registers and reported spill
  extent at compilation time. It does not infer host flags from backend
  instructions or add any work to guest edges. Eliminated entry operands must
  be omitted explicitly, never fabricated.
  `native/tests/backend.rs` exercises real CLIF entry definitions, selected
  labels/landing instructions, final maps and exit patches through Nixe's
  emitters and gateway. Empty/cyclic links cover all four allocator pairings;
  a separate 31-GPR/31-vector case exercises allocation-chosen spills, and
  instruction-attached fault maps are checked against actual trap offsets.
  Canonical exit verifies the whole architectural state, dirty/clean vectors,
  lazy NZCV, sticky FP status, caller FP restoration and exit identity.
  Both encoders pass; these four integration tests are included in the 27
  native-boundary tests passing on x86-64 and under AArch64 QEMU.
  Test-owned JITModule storage holds caller-assembled bytes patched before RX
  finalization; it is not Task 2's allocator or publication protocol. The
  existing production compiler still uses its old boundary until the planned
  cutover, not as a fallback for these native contracts.
- Next action: step 5's consolidated exit-criterion review and native proof.
  Native AArch64 hardware remains unavailable; QEMU evidence does not close
  that gap. No further fork pin or final-map wiring is pending from step 4.
