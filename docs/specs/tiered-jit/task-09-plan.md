# Task 9 implementation plan

Status: complete (steps 1–6).

Working checklist for [Task 9](spec.md#task-9-remove-the-superseded-architecture).
Follow [CONTRIBUTING.md](../../../CONTRIBUTING.md). Code is authoritative;
historical plans do not justify retaining obsolete implementations.

## Scope

Remove superseded owners, adapters, APIs, inspection-only state and tests of
obsolete contracts. Reuse real publication/admission; do not introduce production
bypasses or test frameworks. Preserve guest semantics, native hot paths, bounded
lifetime/reclamation, indexed invalidation and useful regression coverage.

LCQ/HCQ drivers, host encoders, explicitly selected interpreter, semantic helpers
and shared direct-memory stubs serve distinct supported behavior. They are not
legacy duplicates. General metadata compression, flag-recipe separation, cache
tuning and disk caching remain outside this task.

Use `/tmp/nixe-observable-fp-local.toml` with the local Wasmtime `nixe` branch.
No commits, pushes or dependency-pin changes. No homebrews/perf unless requested.
Source paths below are relative to `crates/cpu-jit/src/`.

## Steps

- [x] **1. Inventory remaining migration code and establish the baseline.**

  Runtime uses JitThread, LCQ/HCQ, shared Translator, one Lifetime/Cache and
  protected gateway. Old BFS/direct executors, lookup/hotness nodes, root-owned
  HCQ publication, context-tail ABI and production JITModule are already absent.
  Remaining candidates are assigned to steps 2–5 below.

  Baseline: Nixe `8e6751988669c59a12398064ba0211d171ce930e`; Wasmtime
  `0380097992d7d337bdf66873510a8c42a8923c0b`. Passed 990 x86-64 JIT unit
  tests, 2 differential and 2 dependency tests, plus production checks.
  Dead-code audits on both targets identified the same 31 warning groups.

- [x] **2. Collapse single-implementation lowering abstractions.**

  Removed IntegerLowering/SimdLowering/FpLowering and the duplicate SSA test
  emitter. Shared semantic modules now implement Translator directly; memory
  helpers and SSA reconciliation consume it. Host capability gates are unchanged.
  Follow-up review removed infallible register/vector writes' Result contracts
  and their propagation through integer, scalar-FP and load writeback helpers.
  Reads, instruction validation and other genuinely fallible operations still
  propagate their errors; no compatibility wrapper replaces the removed traits.
  Follow-up validation: all 978 x86-64 unit tests and 4 integration tests passed,
  along with 216 AArch64/QEMU compiler/SSA tests, strict Clippy, runtime/CLI
  compilation, formatting and whitespace checks.

- [x] **3. Retire prototype executable owners without losing ABI coverage.**

  JITModule and its dev-dependencies are removed. Guest-execution fixtures use
  real publication, emitted identities and reader admission, including the
  compiler polling/entry tests found during the second audit. HCQ fixtures
  first publish their resident LCQ images. LCQ polling uses production self-links.
  No fabricated execution epochs remain in gateway/compiler execution tests.
  Isolated System-ABI flag arithmetic uses the existing W^X cache. The test-only
  register-transfer assembly harness also uses that cache, with a live allocation
  lease, to exercise physical register/spill moves and state preservation directly.
  It publishes no guest entry and adds no production execution path or state.

  Removed the duplicate raw gateway runner and
  `real_gateway_links_independent_units_and_completes_canonical_exit`.
  Coverage remains in the published full-state/FP gateway test, canonical
  adapter register/spill cycles, both backend link tests, randomized final maps
  and LCQ/HCQ deadline tests. PIC/RSB and observation assertions remain.

  Exit: affected tests pass on both hosts; no JITModule/prototype owner or
  production bypass remains. All 108 selected tests passed on x86-64 and
  AArch64/QEMU; strict Clippy passed. No Wasmtime source changes.
  Machine-local override no longer patches the two removed dev-dependencies.

- [x] **4. Remove dead APIs, test-only production state and compatibility branches.**

  Removed all broad JIT dead-code allowances (`hcq`, `sampling`, `executable`
  and `lifetime`) using actual consumer searches and non-test builds on both
  targets. Inspection helpers needed only by tests are behind `cfg(test)`;
  no replacement suppression was added.

  Inspect staging records, native output, registries and sampling for unused
  fields or duplicate ownership kept solely for old assertions. Do not strip
  byte charges, strong references, generation/version identities, state maps
  or evidence used by publication, recovery, eviction or reshape. Remove
  obsolete fallback branches and tests with their owners, not by retaining
  compatibility wrappers to keep those tests green.

  **Completed:** removed Analysis's unused architectural blocks/points and their
  entire duplicate liveness calculation; only native/FP/flag results remain.
  Removed CodeUnit's constant `abi_version`, UnitRecord's unused `published`,
  `Registry::find`, `Samples::invalidate` and `FlagFlow::packs_edge`.
  `Structural::reason` is now a test-only getter of existing production data.
  Removed four tests of the discarded architectural table; native-flow tests
  already cover joins, old fault destinations, partial writes, FP ownership and
  loop state. Remaining tests assert those real contracts directly. Removed
  the unused sampling-invalidation test; version/owner-change reset tests remain.

  Removed the unused stream-consumption adapter, its global-flush path,
  target enum and error variant. Production still invalidates before mutation
  through the bound observer. Migrated the Task 8 history-loss fixture and
  cutover/claim/drain assertions to real instruction-cache invalidation.
  Removed four stream-only tests and the global-flush negative-record test;
  added range-failure coverage at observer entry. Existing tests cover reader
  quiescence, concurrent holds, coordinator failure and negative-index reuse.
  Clarified the spec: unrelated log overflow alone does not flush live sources.
  Test-only inspection helpers now use `cfg(test)`: KeyIndex::is_empty,
  Invocation::fault/frame, Candidate::check, Workers::queue, PreparedBridge::key,
  PreparedLink's contract getters and Lifetime::snapshot. No execution bypass
  or additional production state was introduced.

  Shared fixed-stub fault attribution now uses a sorted immutable table:
  removed incremental publication, atomic page hashing, mutex-owned regions,
  overlap trees, capacity/reservation branches and redundant Arc ownership.
  The ten real stubs remain in one OnceLock; JIT attribution still uses its
  epoch-owned directory. Removed WorkerFaultContext::invoke (test-only callers);
  those tests now compose the real batch setup and active invocation directly.
  Registry/batch internals are private, not exported prototype APIs. Replaced
  publication-race/exhaustion tests with exact immutable lookup/overlap coverage.

  Negative results now retain only their key, invalidation associations and
  storage ownership. Removed unused reason/cursor fields, the duplicate Rejection
  enum and arguments that only carried those fields; positive publication's
  cursor and all negative memory/evidence checks remain. Removed Record::prepare,
  unreserved Index::insert and its compatibility branch; index fixtures reserve
  and install through the worker path, including duplicate/failure release.
  Kept Lifetime's metadata lease, the background cell pin and retired directory
  owners as explicitly named ownership fields; their drop order is unchanged.
  Cold shutdown/promotion diagnostics remain consumers of actual runtime state.
  Removed PreparedTransfer::validate; its tests inspect the same prepared-bridge
  validator consumed by PIC installation, without the forwarding wrapper.

  Removed Ticket: maintenance requests/retirement/invalidation return their
  MaintenanceSequence directly. Production no longer carries the process borrow
  and reason solely for test inspection. Tests check the coordinator's actual
  completed sequence through a cfg(test) getter; no additional state or path.
  Removed Cache::allocate/install forwarding wrappers and segment_for_pc.
  Fixtures use the real island-aware allocation/installation with zero islands
  where appropriate; allocator and admitted-directory bounds coverage remains.

  Removed discard_pending_link and its separate pending-only removal contract.
  Collapsed unlink_link's forwarding wrapper and unlink_registered_link into one
  implementation consumed by production replacement, installation and retirement.
  Pending and installed edges retain their distinct necessary behavior in that
  single path. Link identity remains process-qualified; no new metadata or native
  edge work. Execution, far-branch, fault-race and failed-restoration tests now
  request unit retirement; pending adjacency tests retire sources/targets through
  the coordinator. Internal registry/bridge-accounting and resident-fallback
  fixtures use the same unlink operation as production. No compatibility route.

  Removed Reader::cache_bridge and its quiescent-only installation path.
  NativeSuspension::cache_bridge now owns the implementation directly. PIC,
  weak-reuse and reshape/evidence fixtures admit a real invocation and suspend
  it before installing; no fabricated epoch or runtime test hook. Removed the
  duplicate PicHandle and its test-only process field: the weak index retains
  its existing reader/bridge generations, and weak insertion returns only
  replaced ownership for destruction outside the lock.

  Removed retire_dispatch and OccupiedDispatch. Tests now use unit retirement or
  the real pressure pass for empty reservations; stale publication checks use
  publication validation. Migrating the negative-evidence test exposed a missing
  invalidation in pressure's empty-slot retirement: it now removes associations
  through the existing dispatch-owner index and drops their storage outside the
  lock. Empty reservations still do not invalidate unrelated selection pages.
  Removed two tests of manual slot retirement during active execution, a route
  absent from production. Existing native-closure and second-grace-period tests
  cover actual reader protection; the migrated 70-slot test verifies bounded
  collection, compiler-owner retention and real generational storage reuse.
  Fixed a worker test's completion race: its notification precedes final cleanup,
  so a subsequent nonblocking seed admission can legitimately return Deferred.
  The test now retries only that result within a fixed deadline; production's
  nonblocking admission and error behavior are unchanged.

  **Checkpoint validation:** 980 x86-64 unit tests, 2 differential and 2
  dependency tests passed. AArch64/QEMU: 150 memory/publication/invalidation/
  reclaim/negative/freeze tests passed. The preceding checkpoint also passed
  124 flow/compiler/sampling tests and reran the four gateway tests on both hosts.
  Strict Clippy, x86 JIT/runtime/CLI and AArch64 JIT production checks,
  formatting and whitespace checks passed. No homebrews or perf.
  Commands use the local override: `cargo test -q -p nixe-cpu-jit --lib --tests`;
  Latest Arm library filters (after `--`): `lifetime::memory
  lifetime::unit::invalidation lifetime::unit::reclaim lifetime::unit::reshape::negative
  lifetime::background::work::candidate::freeze hcq::compiler::publication::tests::mutation`,
  with the linker/QEMU runner from [the Arm guide](../../aarch64-tests.md).
  Temporary logs: `/tmp/nixe-task9-step{3,4}-*.log`.

  Fixed-stub cleanup: reran all 980 JIT unit tests and 4 integrations;
  `cargo test -q -p nixe-cpu-direct-memory -p nixe-cpu-interpreter --lib`
  passed 22 + 107 tests on x86-64. Arm/QEMU passed 21 + 107 tests with the
  fatal-signal supervisor excluded; all 12 child scenarios were then launched
  explicitly through QEMU as described in the Arm guide, verifying signal/exit
  and diagnostics. Strict Clippy passed for JIT and direct-memory; production
  runtime/CLI and Arm JIT/direct-memory checks passed. Logs:
  `/tmp/nixe-task9-fault-cleanup-*.log`. No homebrews or perf.

  Negative-result cleanup: all 980 JIT unit tests + 4 integrations passed.
  After moving cursor capture exclusively to positive publication, reran 117
  negative/rejection/HCQ-publication tests on x86 and 163 including reclamation/
  background on Arm/QEMU. Dynamic validation: 23 tests on each host.
  Strict Clippy, runtime/CLI and Arm production checks passed. Commands use
  the same override and runners; logs: `/tmp/nixe-task9-negative-*.log`.

  Maintenance/cache cleanup: 980 x86-64 unit tests + 4 integrations passed;
  560 AArch64/QEMU tests passed (executable, lifetime, native, SSA and affected
  compiler/engine publication, chaining and lifecycle tests). Strict Clippy,
  runtime/CLI and Arm JIT production checks, formatting and whitespace passed.
  Same override/runner; logs: `/tmp/nixe-task9-maintenance-*.log`.

  Link cleanup: all 980 x86-64 unit tests + 4 integrations passed; 92
  AArch64/QEMU tests passed (`lifetime::unit::links`, reclamation, LCQ chaining
  and fault/chaining races, engine fallback/PIC execution). Strict Clippy,
  runtime/CLI and Arm JIT builds, formatting and whitespace passed. Same
  override/runner; logs: `/tmp/nixe-task9-links-*.log`. No homebrews/perf.

  Final step 4 validation: 978 x86-64 unit tests + 4 integrations passed.
  AArch64/QEMU: 459 lifetime/engine-fallback tests, the strengthened pressure
  negative regression and both worker-rejection tests passed. The worker retry
  regression also passed ten consecutive x86 runs. Strict Clippy, x86 runtime/CLI
  and Arm JIT builds, formatting and whitespace passed without dead-code
  suppressions. Same override/runner; logs: `/tmp/nixe-task9-final-cleanup-*.log`,
  `/tmp/nixe-task9-pressure-negative-*.log`, `/tmp/nixe-task9-worker-retry-repeat.log`.
  No homebrews/perf; no Wasmtime changes. All step 4 findings are resolved.

  **Exit:** production has no migration-only API, measurement-only field or
  test-driven alternate route; remaining APIs have current consumers. Error,
  lifetime and native-shape regressions still pass without broad suppressions.
  Record each finding's disposition here (removed, test-only or retained with
  its real consumer/ownership reason); no item is deferred implicitly.

- [x] **5. Finish workspace integration and documentation cleanup.**

  **Completed:** audited runtime/CLI consumers and manifests. Runtime still
  selects concrete JIT/interpreter backends and owns guest-thread return stacks;
  no compatibility adapter was needed. Differential tests use the public engine
  directly; removed their duplicate state/memory setup. Dependency tests no
  longer assert private module/re-export spelling or forbid unrelated emitter
  libraries; they retain CPU/memory isolation, test-only interpreter use and
  reject JITModule dependencies in both production and test sections.

  Removed unused ABI exports: CanonicalEntryContract, DispatchGeneration,
  AdmissionSnapshotSequence, SampleSequence, NATIVE_ABI_VERSION,
  LazyFlags::dirty and the redundant UnsupportedFpControl re-export. Updated
  module comments and the runtime shutdown diagnostic. README/fork docs now
  describe active promotion/reshape and coordinated lifetime management. Fork
  HEAD is clean at 0380097992d7d337bdf66873510a8c42a8923c0b; the earlier portable
  manifest pin is unchanged. Documented the four-crate local override and
  corrected the Arm test command to use it; offline mode remains optional.

  **Approved retained contracts:** the user chose to preserve
  FpSpecialization::Exact, ValueLocation::Constant, GuestValue::Fpsr and
  NzcvLocation::Host and plan their integration. See the separate
  [ABI integration plan](abi-integration-plan.md) for producers, consumers and
  exit criteria; the spec explicitly distinguishes these from current output.
  Four variant-local, non-test dead-code expectations identify this approved
  pending work; they must disappear when production constructs those variants.
  No blanket allowance or artificial producer was added.

  Closed abi/analysis/native modules to external callers. Public engine/error/
  return-stack APIs remain the runtime boundary. Removed unused numeric identity
  accessors; remaining fixture constructors and inspection-only accessors are
  cfg(test). Removed the canonical-writeback re-export; tests import its real
  implementation. NativeFrame::ensure_fp is now a test-only fixture convenience;
  production activation keeps its generated native path. Removed HostFpState::end
  and NativeFrame::end_fp: FP mode replacement already ends the invocation in
  production. Their test now exercises finish/begin rather than an unused partial
  segment API. No new production state or compatibility path was introduced.

  **Validation:** 978 x86-64 JIT unit tests + 4 integration tests passed;
  68 AArch64/QEMU ABI/native/FP-owner tests + the same 4 integration tests passed.
  Strict JIT Clippy, x86 JIT/runtime/CLI and Arm JIT production checks, formatting
  and whitespace checks passed. Commands use the local override and the Arm
  guide's linker/runner. x86: `cargo test -q -p nixe-cpu-jit --lib --tests`;
  Arm library filters: `abi::tests native::tests fp_env::tests`; integration
  targets: `--test dependency_boundaries --test differential`.
  Logs: `/tmp/nixe-task9-step5-{full,arm-unit,arm-integration,clippy,production,arm-production}.log`.
  No homebrews/perf, fork modifications or portable dependency-pin changes.

  Check exports, runtime/CLI consumers, manifests,
  `crates/cpu-jit/tests/dependency_boundaries.rs` and
  `crates/cpu-jit/tests/differential.rs` after the removals.
  Simplify boundaries with a single obsolete adapter, but
  preserve genuinely separate backends and architectural ownership. Remove
  tests of old module names/layout when they no longer assert a valid contract;
  retain dependency isolation and interpreter differential checks.

  Shared invocation/fault code currently under `lcq/` also serves HCQ. A
  directory name alone is not duplicate behavior or a reason to rename it.

  Update current module comments, README and fork documentation where they
  describe removed paths or future integrations already completed, including
  the stale future-task comments in `lib.rs`.
  `docs/cranelift-modifications.md` describes an older local working revision:
  update that independently from the portable dependency pin. Keep the pinned
  fork distinct from the locally tested revision. Update the spec only
  where the implemented contract needs clarification; historical completed
  plans need not be rewritten. Do not add aliases for deleted internal APIs.

  **Exit:** current documentation and consumers describe the surviving system,
  removed dependencies have no users, and JIT/runtime/CLI production builds
  succeed with the supported local override. Any portable pin handoff is explicit.

- [x] **6. Validate the surviving architecture and close Task 9.**

  **Architecture audit:** traced JitProcess/JitThread, both compiler drivers,
  shared Translator, HCQ workers, publication, native links and reclamation.
  Production has one process Lifetime/Cache, one shared semantic lowering path
  and the same identity/admission contracts across LCQ and HCQ. No migration-map
  legacy implementation or additional production test adapter was found. Existing
  native-shape/bridge tests remain in the full suite. Corrected three stale
  comments; this step adds no executable code or runtime abstraction. The four
  ABI contracts retained by explicit approval remain the separate step 5 handoff.

  **Results:** all checks passed. x86-64: 978 JIT unit tests + 4 integration
  tests; CPU 90, memory 60, direct-memory 22, interpreter 107 and runtime 69.
  AArch64/QEMU: 982 JIT unit tests + 4 integration tests, direct-memory 21 and
  interpreter 107, plus the explicit signal children below. Strict Clippy,
  both-target production checks, formatting and whitespace checks passed.
  Unit-count reductions from the baseline are the obsolete contracts removed
  in steps 3–4, not failing tests excluded from validation.

  **Validation commands:** run from the repository root in Bash, using the same
  local override and QEMU 11.1.1 as earlier steps:

  ```bash
  task9_cargo=(--offline --config /tmp/nixe-observable-fp-local.toml -q)
  task9_arm=(--target aarch64-unknown-linux-gnu
    --config 'target.aarch64-unknown-linux-gnu.linker="aarch64-linux-gnu-gcc"'
    --config 'target.aarch64-unknown-linux-gnu.runner=["/usr/local/bin/qemu-aarch64", "-cpu", "max", "-L", "/usr/aarch64-linux-gnu"]')
  cargo test "${task9_cargo[@]}" -p nixe-cpu-jit --lib --tests
  cargo test "${task9_cargo[@]}" -p nixe-cpu-jit --lib --tests "${task9_arm[@]}" -- --skip lcq::invocation::tests::lcq_dispatcher_rejects_an_arena_access_without_native_pc_metadata
  cargo test "${task9_cargo[@]}" -p nixe-cpu -p nixe-memory -p nixe-cpu-direct-memory -p nixe-cpu-interpreter -p nixe-runtime --lib
  cargo test "${task9_cargo[@]}" -p nixe-cpu-direct-memory -p nixe-cpu-interpreter --lib "${task9_arm[@]}" -- --skip tests::unrelated_and_nested_faults_remain_fatal_and_previous_handlers_chain
  cargo clippy "${task9_cargo[@]}" -p nixe-cpu-jit -p nixe-cpu-direct-memory --lib --tests --no-deps -- -D warnings
  cargo check "${task9_cargo[@]}" -p nixe-cpu-jit -p nixe-cpu-direct-memory -p nixe-runtime -p nixe-cli
  cargo check "${task9_cargo[@]}" -p nixe-cpu-jit -p nixe-cpu-direct-memory --target aarch64-unknown-linux-gnu
  cargo fmt --all -- --check
  git diff --check
  ```

  The two excluded Arm supervisors were replaced by explicit QEMU child launches
  as described in the Arm guide, not treated as passing tests. Direct-memory
  cases: `outside_address`, `outside_pc`, `nested`, `retry_livelock`,
  `captured_retry_livelock`, `captured_unattributed`, `captured_panic`,
  `dispatcher_fatal`, `closed_diagnostic_pipe`, `alternate_stack_guard`,
  `unrelated_sigbus`, `chain`; plus the JIT unattributed-fault child. All 13
  produced their expected termination status and diagnostic presence/absence.
  Core dumps were disabled. Logs: `/tmp/nixe-task9-step6-*.log`.

  **Handoff:** no homebrews/perf were run, and no Wasmtime source or portable
  dependency pin was changed. These results use the local override; updating
  the portable pin remains separate. The approved ABI integration plan remains
  pending. Native Arm hardware validation belongs to Task 10; QEMU does not
  establish hardware ordering/cache coherence or commercial-workload performance.

  **Exit:** no callable legacy route, prototype executable owner, redundant
  lowering abstraction or production test scaffolding remains. Useful semantic,
  concurrency, ABI, allocator and lifecycle coverage passes. Task 10 can begin
  native-host conformance; QEMU and cleanup results do not prove Arm hardware
  ordering/cache coherence or commercial-workload performance.
