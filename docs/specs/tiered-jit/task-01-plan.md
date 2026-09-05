# Task 1 implementation plan

Status: steps 1–3 are complete: inspection, the bounded backend-mechanism proof
and shared state contracts/analysis. The full executable ABI still needs the
backend and native-boundary integration and validation in steps 4–5.

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

- [ ] **Complete the backend and native boundary.** Implement the remaining
  fork hooks for physical state maps, selected labels, patchpoints, relocations,
  fault offsets, caller-owned output and required landing instructions. Connect
  the gateway, canonical/fast entries, canonical exit and cycle-safe bridges
  to these real outputs. Preserve caller and guest FP state and lazy flags.
  Keep production epoch/reclamation machinery in Task 2; any temporary support
  for the isolated proof stays test-only.

- [ ] **Run the native proof and finish integration.** Exercise empty and
  register-cycle bridges, randomized register pressure, the fixed spill
  partition and exact state reconstruction. Inspect both targets' output for
  the Task 1 exit criteria, and execute on available native hosts. Record any
  missing native execution explicitly; encoding or cross-compilation is not
  execution evidence. Use conventional tests, not a new testing framework or
  benchmark runner. Pin the tested fork revision in Cargo and update the lockfile
  as specified. Remove superseded task-local scaffolding; identify any test-only
  proof support that must remain until the later production cutover.

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
- [FP environment](../../../crates/cpu-jit/src/direct/fp_env.rs): retain the
  semantics of `ensure`, `suspend`, `resume` and `finish`, adapting their
  NativeContext coupling. Reuse typed operations in
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
  (initial inspection). The Wasmtime checkout is on branch `nixe`, clean at
  `2f8ccabacf5cd648e3a0f17f00594e86ddda1058`, based on release v48.0.1 at
  `7bac2c2775808aaec5d4aa5627a5e447b51102cf`. Step 3 changes Nixe's shared
  contracts and compiler analysis, not the fork or dependency pins.
- Dependency mismatch: `crates/cpu-jit/Cargo.toml` pins the main Cranelift
  crates to `0.134.3`; the chosen fork is `0.135.1`. Update the compatible
  version requirements together with the Git pins during integration.
- Coverage baseline: the above Nixe revision's `patterns()` plus actual
  family lowering and `region.rs::system_instruction_supported`, not every
  recognized encoding. Existing catalog tests cover scalar/control,
  memory/system and FP/SIMD families; CLREX and exclusive pairs have additional
  tests. ERET/DRPS remain recognized-unsupported, and system operands have
  explicit support restrictions. This inspection does not create a new catalog.
- Checks passed with Rust `1.97.1` on native Linux x86-64:
  `cargo test --offline --locked -p nixe-cpu-jit` (122 unit tests, 2 dependency
  boundary tests and 2 differential tests).
  `cargo check --offline --locked -p nixe-cpu-jit --target aarch64-unknown-linux-gnu`
  and `cargo clippy --offline --locked -p nixe-cpu-jit --lib --no-deps -- -D warnings`
  also passed. Strict Clippy including dependencies hits existing
  `type_complexity` warnings in `crates/memory/src/range.rs`.
  These check the current JIT and shared contracts, not native execution of the
  new ABI or its integration with the fork.
- Hosts: local native x86-64 and installed AArch64 Rust target confirmed. No
  native AArch64 host identified in repository configuration; arrange one with
  the maintainer before claiming native AArch64 validation. Encoder tests can
  run locally with Cranelift's `x86` and `arm64` features; Nixe currently enables
  only `host-arch`.
- Implemented in the fork: opt-in `enable_nixe_abi`, required register pools,
  prologue-free leaf fragments, external-frame spills/reloads and both explicit
  and folded stack-slot accesses. `src/nixe.rs` defines the 2 KiB/16 KiB bounds;
  this initial layout places the spill area at NativeFrame offset zero.
  `MachBufferFrameLayout::nixe_frame_size` reports the bounded allocation extent.
  System arguments, calls, returns, dynamic frames and debug value-location
  maps are explicitly unsupported until the real boundary integration exists.
- Backend checks (from the Wasmtime checkout, Rust `1.97.1`):
  `cargo +1.97.1 test --offline --locked -p cranelift-codegen --no-default-features --features std,unwind,all-native-arch,disas --lib --quiet`
  passed 232 tests, with none ignored; the same command with `enable-serde`
  added to the features also passed 232 tests.
  Coverage includes register pools, pressure-generated machine bytes, frame
  limits, unsupported boundaries, marker offsets, allocator constraints and
  the native experiment below. Both Nixe compiler profiles and both target
  encoders are exercised.
  `cargo +1.97.1 test --offline --locked -p cranelift-codegen --no-default-features --features std,x86,arm64,disas --lib nixe --quiet`
  passed 15 tests without `unwind`, with none ignored.
- Representation established by the probes: fixed-def operands can express
  physical entry inputs; `Any` operands and `Output::inst_allocs` expose exact
  register/spill locations without optional debug ranges; zero-byte
  `sequence_point` markers with stable tags retain offsets in hot/cold blocks.
  These are building blocks, not a completed EntryContract/ExitStateMap API.
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
- Next action: complete the backend and native boundary (step 4). Produce
  physical fast-entry inputs and exact boundary/fault maps from real allocation,
  connect gateway/exits/bridges and preserve lazy flags and FP ownership.
  The shared contracts are implemented but not yet connected to those native
  hooks; the existing JIT still uses its current boundary. Final native proof
  and fork dependency pins remain in step 5.
