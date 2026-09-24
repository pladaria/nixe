# Task 8 implementation plan

Status: Task 8 complete; steps 1–8 closed.
Task 7 is closed, including the runtime
reshape-stability and ordinary-reclamation corrections. Native Arm hardware
validation remains under Task 10.

This is a working checklist for
[Task 8](spec.md#task-8-close-lifecycle-fault-and-pressure-races), not another
specification. Follow [CONTRIBUTING.md](../../../CONTRIBUTING.md). Update each
step in place with findings, decisions and validation. The current code is the
starting point; older task notes are not a competing contract. Agree material
architectural changes with the maintainer and update the affected spec.

## Scope and sequencing

Finish the interaction between publication, invalidation, faults, replacement,
reclamation, cache pressure and shutdown. Much of this already exists: inspect
and reuse its tests, add missing coverage and fix demonstrated gaps rather than
reimplementing working mechanisms or building another testing framework.

Preserve native direct links, the shared poll, exact generational identities,
retained LCQ baselines and nonnested locks. Do not add per-link validation,
whole-cache scans on ordinary maintenance, or production instrumentation merely
to make races easier to test. Prefer indexed affected-record operations;
document unavoidable bulk work instead of describing it as O(1).

Task 7 verified ordinary collection below pressure and removed two real reshape
oscillations. It did not establish that larger working sets fit comfortably or
that eviction avoids excessive recompilation. Task 8 must check pressure and
progress, not infer scalability from 60 FPS in small demos. Keep the current
512 MiB soft limit, 480 MiB pressure target, 640 MiB hard limit and 32 MiB LCQ
reserve unless an explicit design change is agreed. These are accounted cache
budgets, not total process RSS limits.

General metadata compression, lazy-flag recipe separation, disk caching and
cache-policy tuning without evidence are outside this task. Accounting errors,
retained garbage and algorithmic bottlenecks found in the lifecycle are in scope.
Wholesale legacy removal remains Task 9; remove superseded paths introduced or
replaced by this task as part of their fixes, not later.

Homebrew execution and perf are not automatic checkpoint requirements. Leave
them to the maintainer unless explicitly authorized for the work in question.
Use conventional unit/integration tests and existing synchronization helpers;
no timing-sensitive sleeps or new benchmark framework.

## Starting points

Paths below are relative to `crates/cpu-jit/src/` unless stated otherwise.

- `lifetime.rs`, `lifetime/maintenance.rs` and `lifetime/memory.rs`: admission,
  transition ownership, request sequences, execution epochs and mutation holds.
- `lifetime/unit.rs`, `lifetime/unit/invalidation.rs`, `lifetime/unit/links/`
  and `lifetime/unit/dynamic/`: publication, incoming roots, exact unlink targets,
  static patches and per-reader dynamic bridges/PICs.
- `lifetime/background/work{,/candidate,/evidence}.rs`,
  `lifetime/background/workers{,/tests/races}.rs` and
  `hcq/compiler/publication{,/tests}/`: immutable inputs, claims, worker
  cancellation, positive/negative publication and real replacement tests.
- `lifetime/unit/reclaim{,/tests}.rs` and `executable{,/tests,/linux,/patch}.rs`:
  pending-retired FIFO, two directory grace periods, pressure recovery, charged
  storage, reusable spans and segment decommit. Ordinary collection must remain
  distinct from exhaustive pressure/shutdown work.
- `lcq/invocation{,/memory}.rs`, `lcq/fault{,/access,/cold}.rs`,
  `engine/completion.rs` and frontend memory lowering: shared native fault
  attribution, reconstructed state, retry and typed cold completion for both tiers.
- `engine/background.rs`, `engine/tests/shutdown.rs`,
  `crates/cpu-direct-memory/src/`, `crates/memory/src/execution_gate.rs` and
  `crates/memory/src/invalidation.rs`: worker lifetime, signal capture, memory
  exclusion and invalidation delivery outside the JIT's own locks.

Use `--config /tmp/nixe-observable-fp-local.toml` for the local Cranelift override;
`--offline` is optional when all dependencies are cached. Inspect the config and
`/home/pladaria/projects/wasmtime`, branch `nixe`, before relying on them. Fork
changes are permitted when needed. Preserve unrelated edits and portable
dependency pins; no commit, push or dependency-pin update unless requested.

## Steps

- [x] **1. Map the remaining lifecycle gaps against production callers.**

  Trace one LCQ demand, HCQ promotion/reshape, memory mutation, fault retry,
  pressure pass and process shutdown through the current owners and call sites.
  Identify where each object becomes reachable, which reference protects it,
  who removes its roots and who eventually releases its allocation. Include
  staged output, retired directory snapshots, negative-result associations and
  the pending-retired FIFO added at Task 7 closure.

  Match the Task 8 exit criteria to existing tests and list only missing cases
  here under their owning step. Check lock acquisition and destruction paths,
  not just explicit lock calls: dropping a lease or the last Arc may take a lock.
  Record baseline validation and fix the sequencing of this plan if code review
  reveals a dependency; do not expand it into a second lifecycle specification.

  **Exit:** each missing invariant has a concrete production consumer and a
  focused validation/fix step. Existing passing behavior is not scheduled for
  redundant reimplementation.

  **Review (2026-09-23):** the production paths are connected. The remaining
  combinations below are coverage targets, not demonstrated defects. No new
  coordinator, reclamation mechanism or test framework is needed.

  - **Demand/publication:** `engine.rs::demand` owns the capture/claim until
    `PreparedUnit::publish`; HCQ's `hcq/compiler/publication.rs` retains frozen
    inputs and staged output through final validation. Publication installs
    directory metadata before dispatch payloads under JIT state. Failed staging
    releases its leases; replaced payloads are destroyed outside the lock.
    Existing `closing_after_preparation_rejects_every_entry_and_returns_exact_span`
    and `real_replacement_mapping_change_joins_pending_cutover_and_invalidates_both_versions`
    cover both basic rejection and a real replacement/mutation crossing.
  - **Memory:** `lifetime/memory.rs::begin_memory_mutation` registers a hold before
    Closing; exact dependencies select affected roots. Execution drains before
    mutation, without waiting for compiler storage pins. Existing memory tests
    cover aliases, interior ranges, history loss, overlapping holds and abandoned
    transition ownership. Guest data writes and instruction-cache publication
    deliberately have different visibility rules; see step 3.
  - **Faults:** `lcq/invocation.rs::run` serves both tiers. The mapping lease and
    announced invocation protect directory/state-map borrows; an escaped exit
    owns its reconstructed completion before those protections end. The direct
    memory dispatcher rejects a repeated unchanged native fault instead of
    retrying indefinitely. Existing HCQ runtime memory tests execute pair,
    structure and exclusive completions; metadata-only checks are not their
    sole coverage.
  - **Retirement:** invalidation/cutover removes dispatch, static and dynamic
    roots before queueing collection. `reclaim_retired` uses the bounded FIFO;
    `reclaim_units` is exhaustive pressure/shutdown work. Code snapshots, baseline
    pins and staged bridges retain storage; directory readers require a second
    grace period after detachment. Existing ordinary-collection tests cover the
    bounded pass and that second grace period. Negative evidence uses weak
    indexed associations, detached under state and destroyed outside it; its
    charge/reuse tests already exist.
  - **Pressure:** `engine/execution.rs::run_slice` drops failed demand ownership,
    calls `recover_capacity`, then retries at most once before a precise capacity
    error. `engine/tests/capacity.rs` tests actual execution and recovery;
    `mixed_root_pressure_reuses_segments_after_snapshot_and_bridge_release`
    proves real reuse. Artificial threshold charges in these tests do not prove
    sustained mixed LCQ/HCQ working-set behavior.
  - **Shutdown/destruction:** `JitProcess::try_shutdown` stops and joins the real
    pool before terminal collection; `try_finish_shutdown` waits for compiler,
    collector and cache-lease owners before clearing foundation storage. Reviewed
    `Work`, `Invocation`, memory-mutation guards, publication and collector drops:
    cache-owning destruction and worker joins occur outside JIT state; invocation
    quiescence precedes mapping-lease release. Existing shutdown tests cover real
    workers, active invocation, failure delivery and idempotence separately.

  **Baseline:** using `cargo test --offline --config
  /tmp/nixe-observable-fp-local.toml -q`, `-p nixe-cpu-jit --lib` passed 967 tests;
  `-p nixe-memory -p nixe-cpu-direct-memory --lib` passed 60 and 22 respectively.
  The override exists and the local Wasmtime `nixe` worktree is clean. No runtime
  code or fork changes, homebrew/perf runs or new cross-target claims in this step.

- [x] **2. Close publication, admission and maintenance races.**

  Exercise LCQ and initial/reshape HCQ publication against Closing, late requests
  during Closed, concurrent publication and cancellation. Include static-link
  installation, PIC insertion and positive/negative worker results. Use controlled
  interleavings around the actual consumers, including the real HCQ publisher.

  Verify coherent payloads and complete metadata before reachability, no lost
  request at reopen, and no new link job after its source is invalidated. Keep
  LCQ's epoch checks distinct from HCQ's exact-version validity across unrelated
  stops. Stale cleanup must release only its own claims/handles; abandoned staged
  output must return its span without changing live dispatch. Mixed old/new
  versions must execute correctly while their roots are being cut over.

  **Exit:** publication never bypasses Closing or exposes partial state; raced
  work either commits under valid authority or cleans up without damaging a
  newer owner. Optional link batching cannot acknowledge unfinished safety work.

  **Completed:** added three tests in
  `hcq/compiler/publication/tests/coordination.rs`, using the real compiler,
  memory authority and existing publication fixtures, without runtime hooks:

  - Promotion and replacement compile through Closing while an invocation keeps
    its old payload. An older LinkPatch batch cannot reopen past a late retirement;
    publication resumes after the stop without losing unaffected inputs/baselines.
  - Two disjoint publishers finish staging against the same segment directory
    before either publishes. Both succeed, retain distinct identities and remain
    attributable; the losing directory snapshot is refreshed without losing the
    winner's metadata. Native ingress executes through both resulting units.
  - A real code write invalidates prepared positive/backend-negative results and
    their dynamic source. Old bridge emission and new preparation from that source
    fail; an already-emitted bridge pins storage until released. Old Work cleanup
    preserves a newer reservation at the same guest root, which subsequently
    compiles and executes the changed instructions through static/PIC/return routes.

  Existing LCQ preparation, PIC insertion, static-link retirement, coherent-payload
  and late/deferred-batch tests remain the complementary coverage. No runtime or
  fork defect was demonstrated by these crossings; no production changes were
  necessary. Step 3 still owns broader alias/history-loss mutation combinations.

  **Validation:** the step-1 Cargo override command with `-p nixe-cpu-jit --lib`
  passed all 970 x86-64 tests. The final coordination filter passed 3 tests;
  `hcq::compiler::publication::tests` passed 39 under AArch64/QEMU using the linker
  and runner from [the guide](../../aarch64-tests.md), plus the local override.
  `cargo fmt --all -- --check` and whitespace checks passed.
  `cargo clippy --offline --config /tmp/nixe-observable-fp-local.toml -q
  -p nixe-cpu-jit --lib --tests --no-deps -- -D warnings` passed. Without
  `--no-deps`, strict Clippy stops on existing `type_complexity` diagnostics in
  `crates/memory/src/range.rs:268` and `:304`; those unrelated APIs were not changed.
  No homebrew/perf execution; QEMU is not native Arm ordering evidence.

- [x] **3. Close executable-write and mapping-change races.**

  Combine replacement, running/staged workers and linked execution with physical
  code writes, virtual remapping, aliases and invalidation-history loss. Include
  changes inside a body rather than only at its entry, retained baselines and
  superseded-but-callable predecessors.

  Distinguish host executable writes/mapping changes from checked guest data
  writes: the latter may leave old translated instructions callable until guest
  instruction-cache invalidation. Preserve the existing
  `guest_write_after_final_validation_can_publish_old_code_only_until_ic`
  contract; do not introduce eager instruction visibility for every data write.

  Verify the real mutation authority keeps coordinated changes invisible until
  admission closes, affected direct/PIC roots are removed and readers reach
  quiescence. A writer faulting inside native code must release its own execution
  protection before waiting; another transition must not reopen through a memory
  hold. Cancel affected compilation/negative evidence without cancelling unrelated
  work or waiting for compiler storage pins before making a safe mutation.

  **Exit:** execution after coordinated mutation or guest instruction-cache
  invalidation observes the new code/mapping, never a
  stale native target. Exact aliases and dependency ranges are invalidated and
  the writing vCPU cannot deadlock on its own epoch or lease.

  **Completed:** added two integration cases in
  `hcq/compiler/publication/tests/mutation.rs`, reusing the real compiler and
  native-ingress fixture:

  - Modify an interior instruction through an executable physical alias after
    HCQ replacement, with cutover either completed or still pending. Before IC,
    the guest write leaves the invalidation cursor unchanged and, when admission
    is open, both translations still execute the old instruction. IC through the
    alias withdraws both HCQ versions, the alias family and affected LCQ entries
    despite retained compiler snapshots. An unrelated frozen worker still
    publishes successfully. Recompiled static/PIC/return ingress and the alias
    execute the changed instruction at the correct virtual PC.
  - Overflow the real memory log with 1,025 coordinated permission changes on
    an unrelated non-code page. Frozen workers survive those exact mutations.
    Then publish a replacement without draining cutover and consume the lost
    history: all entries, old/new HCQ, baselines and unrelated compilation are
    invalidated conservatively. Dropping storage pins permits complete executable
    decommit, followed by successful demand compilation and native execution.

  Existing engine lifecycle tests cover delayed alias visibility while a later
  unit's fault epoch is active and removal of every vCPU's PIC roots. Memory
  tests cover mapping/permission changes, overlapping holds and mutations without
  waiting for compiler pins. `cache_completion_invalidating_its_source_cannot_heat_stale_code`
  covers the production cold completion after releasing its own invocation.
  These checks did not demonstrate a runtime defect; instruction visibility and
  coordinator behavior remain unchanged. Fault retry/compound-access coverage
  belongs to step 4, not a new mutation protocol.

  **Validation:** same override/commands as step 2: all 972 x86-64 JIT tests and
  41 HCQ publication tests under AArch64/QEMU passed. JIT Clippy with `--no-deps
  -- -D warnings`, formatting and whitespace checks passed. No fork changes,
  homebrew/perf execution or native Arm hardware claims.

- [x] **4. Verify fault identity, retry and compound-access commit state.**

  Exercise the native-PC directory during HCQ replacement, invalidation,
  detachment and eventual address/segment reuse. An old fault reader must retain
  the correct CodeVersion and state map throughout resolution; a reused address
  must resolve only to its current metadata after the old grace periods end.

  Check recoverable RAM faults resume the identical native instruction with
  correct registers, deferred NZCV and FP state. Repeated same-generation faults
  without tracking/resolution progress must report a precise error, not spin.
  Preserve the allocation-free, lock-free signal capture/landing contract.

  Extend existing differential cases where needed for pair accesses, SIMD
  structures and atomic/exclusive operations in both tiers. Verify retained
  partial reads, already-committed writes, deferred writeback and precise guest
  faults; typed cold completion must not replay completed device effects.
  Consult the existing Arm references for any changed instruction semantics.

  **Exit:** every reachable fault site has live, precise metadata; retries make
  progress or fail explicitly, and compound-access faults preserve the specified
  architectural commit stage across native and cold execution.

  **Checkpoint:** `hcq/compiler/publication/tests/faults.rs` now tests escaped
  integer pair and SIMD structure loads/stores, with successful and failing MMIO
  suffixes. Unlike the older runtime fixture, these use the real HCQ publisher,
  memory dependencies and coordinator. The LCQ prefix supplies FP status and lazy
  NZCV before entering the faulting HCQ body.

  Keep the owned completion pending while publishing a real merged successor.
  A separately announced reader still attributes all old fault PCs to their old
  unit/version through cutover. After that reader and snapshots are released,
  reclaim the predecessor, restore baselines and promote the identical body:
  assert actual native-address reuse with new unit/version identities and correct
  current fault lookup. Finally remove the replacement's execution permission and
  reclaim it before completing the original access. Differential checks against
  the interpreter verify retained reads, writeback, FP/NZCV, partial stores,
  device order/errors and no replay of the native prefix.

  **Validation:** full x86-64 JIT suite passed 974 tests; the final targeted fault
  cases passed after adding the execution-permission withdrawal check. All 43
  HCQ publication tests passed under AArch64/QEMU. The x86-64 direct-memory suite
  passed 22 tests, including fatal no-progress retry subprocesses; the QEMU
  `aarch64_retry` filter passed 2 register/FP/SIMD and in-place-store retry tests.
  Commands use the step-2 override and documented cross runner. JIT Clippy
  (`--no-deps -- -D warnings`), formatting and whitespace checks passed. No
  production/fork changes or homebrew runs.

  **Closure:** `faults/retry.rs` also crosses real HCQ replacement with a live
  captured RAM fault. An executable fetch arms real write tracking on the data
  page; a postindexed native store faults after FP work and a lazy-NZCV producer.
  The replacement worker finishes executable capture before the invocation takes
  its memory lease: capture excludes native writers and cannot wait on the same
  invocation whose fault dispatcher is waiting for publication. The existing
  test-only memory observer pauses that worker after capture. The normal-stack
  fault dispatcher resumes it, waits for real publication, then verifies that
  cutover/reclamation cannot complete and the captured PC still names the old
  unit/version/state map. The production memory resolver repairs tracking and
  native retry completes without a second fault or architectural reconstruction.
  Compare every architectural register, FP/NZCV, postindex writeback and affected
  memory against the interpreter; release the invocation/snapshots, reclaim the
  predecessor, check the successor's fault directory and decommit at shutdown.

  Existing LCQ memory/atomic/exclusive tests and HCQ observation/runtime tests
  cover precise atomic fault maps, incoming/native reservations and partial
  commits. Reuse those checks rather than add another instruction harness.
  No production/fork change or signal-handler instrumentation was needed.

  **Final validation:** all 975 x86-64 JIT tests passed. The final retry case
  also passed after adding explicit live-FP/noncanonical-NZCV assertions. Under
  AArch64/QEMU, all 99 `hcq::compiler` tests passed, including the 44 publication
  cases and existing memory/exclusive runtime cases. The direct-memory fatal
  no-progress and native-retry checks above remain applicable; their code is
  unchanged. JIT Clippy (`--no-deps -- -D warnings`), formatting and whitespace
  checks passed. No homebrews/perf runs; native Arm hardware remains Task 10.

- [x] **5. Finish reclamation, baseline-pin and metadata-accounting coverage.**

  Race ordinary collection with readers, compiler/linker snapshots, new
  retirements and exhaustive pressure collection. Check FIFO fairness/bounds,
  exclusive collector ownership and both directory grace periods. Releasing the
  last protecting reference must permit collection through production maintenance,
  not only through a direct test call to `reclaim_units`.

  Verify HCQ promises never outlive their retained LCQ baselines. Demonstrate
  reuse of actual executable spans and generational unit/family/dispatch slots,
  stale-handle rejection and safe empty-segment decommit/republication. Check
  aborted and superseded units release dependencies, negative associations and
  charged metadata. Retained registry capacity is not a leak by itself: separate
  reusable capacity from live records, unreclaimed records and unreachable owners.

  **Exit:** coupled code/metadata has one correct lifetime; repeated bounded
  workloads reuse storage after warmup, without live-cache scans on ordinary
  collection or unaccounted retained allocations.

  **Finding and fix:** ordinary maintenance reclaimed CodeUnits but only
  exhaustive collection reclaimed retired dispatch slots. A repeated LCQ/HCQ
  workload returned native spans yet left occupied dispatch owners and their
  payload/worker charges until pressure or shutdown. The regression failed on
  the first completed cycle before the fix.

  `lifetime/dispatch_reclaim.rs` replaces the live-registry dispatch scan with
  an intrusive retired-slot FIFO. Every retirement path enqueues once;
  cancelled compile claims leave queued slots to their collector. Ordinary
  maintenance visits at most 32 retired units and 32 retired dispatch slots,
  including dispatch-only work after all CodeUnits are gone. Epoch- or
  worker-pinned slots rotate without starving the tail. Queue operations are
  O(1); existing reader-grace checks still apply. Destructors run outside JIT
  state. The one extra cold link per slot and queue header are included in
  existing size-based registry/Lifetime charges; no queue allocation, native
  check, budget change or whole-live-cache scan is introduced.

  **Coverage:** `unit/reclaim/tests/ordinary.rs` pauses the real ordinary
  collector at the cache mutex while new retirements and capacity recovery
  arrive. Concurrent ordinary/exhaustive callers cannot steal its work; the
  pinned head does not block the tail. A full metadata budget prevents one
  directory copy while another segment is still reclaimed. Terminal epoch
  exhaustion returns the popped unit to its queue, releases collector ownership
  and preserves its directory/storage and charges.

  Dispatch-only tests retain 70 slots behind an active epoch, then verify exact
  32/32/6 draining, stale-handle rejection and capacity reuse. Cancelled compile
  claims and the existing background-token test cover last-owner release through
  production maintenance. Thirty-two repeated LCQ/HCQ cycles reuse actual code
  addresses and unit/family/dispatch capacity with stable accounted memory;
  live baseline pins prevent premature eviction. Negative-lifecycle cycles also
  abandon prepared HCQ output, release exact negative associations/charges and
  verify no directory snapshots or record owners accumulate. Retained empty
  executable segments and index capacity remain reusable, not leaked; existing
  pressure/decommit/republication tests cover their eventual release.

  **Validation:** all 982 x86-64 JIT tests passed; all 453 `lifetime::` tests
  passed under AArch64/QEMU using the step-2 override/cross commands. JIT Clippy
  (`--no-deps -- -D warnings`), formatting and whitespace checks passed. No fork
  changes, homebrew/perf execution or native Arm hardware claims.

- [x] **6. Validate pressure recovery and execution progress at capacity.**

  Extend current pressure tests through production demand, worker admission and
  maintenance callers. Cover live code, real metadata and staging allocations,
  fragmentation, old/new coexistence and compiler/fault-reader pins. Artificial
  charges may reach a threshold cheaply, but are not evidence of actual span,
  registry or backing-store recovery; test those releases separately.

  Verify soft-limit suppression defers HCQ rather than permanently rejecting a
  valid candidate, LCQ retains its reserve, eligible HCQ is retired before its
  baseline and recovery leaves useful headroom. With reclaimable storage, the
  production path must return below soft; with protected storage, preserve it
  and defer HCQ or report precise LCQ capacity failure, without overcommit or
  deadlock. Background optimization must not make the guest wait for compilation.

  Use bounded executable workloads with a repeated working set and changing code
  phases to check reuse and recompilation after recovery. Distinguish unavoidable
  churn when the working set exceeds capacity from pathological repeated stops
  or recompilation when the needed set fits. Observe through existing test state
  and compiler callbacks, not new production counters or a benchmark framework.
  Fix demonstrated lifecycle/algorithmic defects; agree any eviction-policy
  change rather than silently increasing limits or adding a runtime tuner.

  **Exit:** forced pressure recovers real reusable storage and execution resumes;
  protected references cannot cause unsafe reuse or unbounded waiting. No
  avoidable pressure/recompilation loop is left unexplained in the exercised cases.

  **Coverage:** `engine/tests/capacity/phases.rs` uses the process-owned worker
  pool and real HCQ consumer. Its callback counts compilations only in the test;
  no runtime counters, smaller cache limits or alternative execution path are
  introduced. Charges use the existing accounting API to reach thresholds, but
  all reclamation assertions concern real code, metadata and native execution:

  - Four alternating hot-region phases promote real LCQ to HCQ, then drive
    `run_slice` through a cold miss at the soft limit. HCQ baseline promises
    prevent premature baseline eviction. Recovery retires both tiers, decommits
    their segments and recompiles at the same native address with a fresh
    segment generation. Every slice reports only ADD/BRK progress and preserves
    exact architectural effects. Thirty-two subsequent slices retain code
    identity, compilation count and accounted usage; repeated phases stabilize
    after capacity warmup rather than repeatedly evicting a fitting set.
  - Retain both tier snapshots and real unpublished native-bridge allocations.
    Free/reallocate a middle bridge span to check fragmented reuse. At the hard
    budget a cold demand returns the precise LCQ capacity error with zero guest
    progress, unchanged context and retained executable storage. Releasing only
    one bridge does not release the protected segments. After all references go,
    the next real demand recovers them and executes within the 32 MiB LCQ reserve
    despite the external charge keeping total usage above soft. Repeated native
    samples start no HCQ jobs there; removing the charge permits genuine HCQ
    promotion again, proving pressure did not become a permanent rejection.
  - Keep a real HCQ native-fault directory borrow under its invocation epoch and
    memory lease, without a compiler snapshot. A second thread's cold demand
    requests pressure recovery but cannot finish or decommit the protected
    segments. The old fault record retains exact unit/version/instruction
    identity. Releasing the epoch/lease lets the caller reclaim and execute the
    next region below soft. This tests pressure against a fault-directory reader;
    delivered fault/retry transport itself is covered by step 4.

  Existing mixed static/PIC/return-root pressure tests cover unpublished linker
  ownership. Worker pressure tests cover already-queued/running work, unfinished
  compiler-state reset and reserved reshape families; step 5 covers an ordinary
  collector in flight while retirements and production capacity recovery arrive.
  No additional runtime defect was demonstrated and no eviction policy or budget
  changed. These are bounded correctness/progress checks, not a claim that every
  commercial working set fits or that oversize workloads avoid necessary churn.

  **Validation:** all 985 x86-64 JIT tests and all 88 `engine::tests::` tests on
  AArch64/QEMU passed with the local Cranelift override. JIT Clippy with
  `--lib --tests --no-deps -- -D warnings`, workspace formatting and whitespace
  checks passed. No Wasmtime changes or homebrew runs were needed; native
  AArch64 hardware validation remains outside this checkpoint.

- [x] **7. Close shutdown, failure-delivery and lock-order races.**

  Request stop with queued, running and staged work, pending cutover/collection,
  dynamic links, a native fault resolver and an active memory hold. Include
  background failure, partial worker startup and repeated shutdown requests.
  Use the actual process owner so queue closure, wakeup and joins are exercised,
  not just a synthetic Lifetime teardown.

  Ensure terminal shutdown cannot reopen or publish, waits/joins hold no JIT,
  cache or memory lock, and a caller never waits on its own invocation. Check
  last-reference destruction and failure cleanup for the same lock rules.
  Unsupported semantics and implementation failures remain actionable errors;
  stale optimization remains cancellation, not a fatal error or silent fallback.

  **Exit:** once external protecting references are released, shutdown completes
  idempotently with no workers, reservations, epochs, bridges, executable mappings
  or code-owned storage. Concurrent stop/failure cannot deadlock or revive
  the process.

  **Correction:** process shutdown previously joined workers before checking
  invocation/memory quiescence. A compiler capturing instructions can wait for
  the caller's own execution lease, so that ordering could prevent the caller
  from returning to release it. After terminal closure, `try_shutdown` now
  returns pending while readers or memory mutations remain; it joins only after
  they drain. The check uses existing cold coordinator state, with no native
  overhead, new counters or changed failure classification.

  **Coverage:** five additional tests exercise real process ownership:

  - A compiler blocks in real `ExecutionMemory::capture_instructions` behind an
    admitted caller's lease. The gate notifier establishes the race; shutdown
    must return pending before that caller releases its epoch/lease. The retry
    joins the canceled worker and releases executable storage.
  - An HCQ worker pauses after actual W^X allocation and directory preparation,
    before final publication. Another seed is queued behind it. Repeated stop
    drains queued work without compiling it, respects a memory hold and makes
    the prepared publisher cancel. Concurrent teardown cannot steal the join or
    mistake `Joining` for completion. Final shutdown is terminal and idempotent.
  - An ordinary collector is paused at the cache mutex while an unrelated
    memory hold and terminal stop arrive. JIT state remains available; shutdown
    neither steals collection nor bypasses the hold. Controlled release allows
    collection and final segment disposal.
  - A delivered native load fault requests process shutdown from normal-stack
    dispatch. Its live epoch prevents teardown until escape/reconstruction;
    exact PC, lazy NZCV, completed-prefix budget and caller FP survive, and the
    continuation does not execute. No compiler snapshot substitutes for the
    fault epoch.
  - A real owned worker error or panic races stop. Repeated process APIs retain
    the original diagnostic, all worker captures are released, and final
    process/reference destruction releases failed-state storage. An external
    compiler snapshot keeps only its code/cache alive until its own release.

  Existing mixed static/PIC/return-root shutdown tests cover dynamic unlink;
  worker startup injection covers each partial-spawn failure and checks queue,
  thread, capture and metadata cleanup. The engine constructor binds its memory
  observer only after worker startup succeeds. Helpers and collector inspection
  added for these tests compile only under `cfg(test)`; no runtime fault hooks
  or new worker framework were introduced.

  **Validation:** `cargo test --offline --config
  /tmp/nixe-observable-fp-local.toml -q -p nixe-cpu-jit --lib` passed all 990
  x86-64 tests. The same command with the AArch64 target/linker/QEMU runner from
  step 6 and the `shutdown` filter passed all 43 selected tests, including the
  delivered fault and blocked-capture regression. JIT Clippy with
  `--lib --tests --no-deps -- -D warnings`, workspace formatting and whitespace
  checks passed. No Wasmtime changes, homebrews or perf runs were needed.

- [x] **8. Validate the integrated lifecycle and close Task 8.**

  Run the full x86-64 JIT suite and affected memory/direct-memory tests, then the
  AArch64/QEMU suites appropriate to the changes using
  [the existing guide](../../aarch64-tests.md), including its subprocess caveats.
  Record exact commands, results and any unavailable validation here. Run Clippy,
  formatting and whitespace checks for the changed scope.

  Review production call sites again: tests must not be the only callers that
  trigger collection, invalidation or worker cleanup. Remove temporary tracing
  and obsolete code/tests replaced by this task. Update the spec where behavior
  changed; keep general metadata compactness as a separate pending optimization,
  not a reason to claim commercial-workload scalability from synthetic pressure.
  Do not launch homebrews/perf without explicit authorization.

  **Exit:** every Task 8 criterion has code and test evidence, no known lifecycle,
  fault, pressure-progress or shutdown gap remains, and Task 9 can remove the
  superseded architecture. Native Arm ordering/cache coherence remains explicitly
  unproven until Task 10; QEMU results are not hardware evidence.

  **Final review:** production execution calls `try_service_links`; both that
  maintenance path and the worker completion epilogue call bounded retired
  collection. All three dispatch retirement sites enqueue through the same
  intrusive FIFO; canceled compile cleanup cannot remove an enqueued slot.
  Exhaustive recovery remains the cold demand/pressure/shutdown path, not a
  live-registry scan on native transfers. Actual memory binds the mutation
  observer, and runtime process teardown calls the reviewed stop/join owner.
  Directory publication, protected fault reconstruction, baseline pins and
  last-reference cleanup retain the coverage recorded in steps 2–7.

  No additional defect was demonstrated. Removed an outdated comment describing
  already-implemented maintenance consumers as future work. The superseded
  dispatch scan is gone; test synchronization/inspection is `cfg(test)` only,
  with no temporary production tracing or replacement framework left behind.

  **Validation:** x86-64 passed 990 JIT, 60 memory, 22 direct-memory and
  56 CPU-memory tests. AArch64/QEMU passed 994 JIT, 60 memory, 21 direct-memory
  and 56 CPU-memory tests, plus the 13 explicitly launched fatal/handler-chain
  scenarios below. JIT strict Clippy, production checks for JIT/runtime,
  workspace formatting and tracked/untracked whitespace checks passed.

  Reproduction commands (Bash, from the repository root):

  ```bash
  task8_cargo=(--offline --config /tmp/nixe-observable-fp-local.toml -q)
  task8_a64=(--target aarch64-unknown-linux-gnu
    --config 'target.aarch64-unknown-linux-gnu.linker="aarch64-linux-gnu-gcc"'
    --config 'target.aarch64-unknown-linux-gnu.runner=["/usr/local/bin/qemu-aarch64", "-cpu", "max", "-L", "/usr/aarch64-linux-gnu"]')
  cargo test "${task8_cargo[@]}" -p nixe-cpu-jit -p nixe-memory -p nixe-cpu-direct-memory --lib
  cargo test "${task8_cargo[@]}" -p nixe-cpu --lib memory::
  cargo test "${task8_cargo[@]}" -p nixe-cpu-jit -p nixe-memory -p nixe-cpu-direct-memory --lib "${task8_a64[@]}" -- \
    --skip lcq::invocation::tests::lcq_dispatcher_rejects_an_arena_access_without_native_pc_metadata \
    --skip tests::unrelated_and_nested_faults_remain_fatal_and_previous_handlers_chain
  cargo test "${task8_cargo[@]}" -p nixe-cpu --lib memory:: "${task8_a64[@]}"
  cargo clippy "${task8_cargo[@]}" -p nixe-cpu-jit --lib --tests --no-deps -- -D warnings
  cargo check "${task8_cargo[@]}" -p nixe-cpu-jit -p nixe-runtime
  cargo fmt --all -- --check
  git diff --check
  ```

  The two cross-target subprocess supervisors were excluded from Cargo's run;
  their child scenarios were executed explicitly rather than left untested.
  Following [the guide](../../aarch64-tests.md), launch
  their children directly with QEMU 11.1.1 (`-cpu max -L /usr/aarch64-linux-gnu`),
  `ulimit -c 0`, and binaries reported by Cargo's `--no-run --message-format=json`:

  - Direct-memory: `NIXE_DIRECT_FATAL_CASE=<case>` with `--exact
    tests::fatal_fault_subprocess_entry --nocapture`, for `outside_address`,
    `outside_pc`, `nested`, `retry_livelock`, `captured_retry_livelock`,
    `captured_unattributed`, `captured_panic`, `dispatcher_fatal`,
    `closed_diagnostic_pipe`, `alternate_stack_guard`, `unrelated_sigbus` and
    `chain`. Checked exit 139 (SIGSEGV), except 135 (SIGBUS) and 77 (handler
    chain), plus the expected fatal reasons/native-PC/address diagnostics.
  - JIT: `NIXE_LCQ_UNATTRIBUTED_FAULT=1` with `--exact
    lcq::invocation::tests::unattributed_fault_subprocess_entry --nocapture`;
    checked exit 139 and `reason=unattributed-native-pc`.

  Validation uses the clean local Wasmtime `nixe` checkout at
  `0380097992d7d337bdf66873510a8c42a8923c0b`. No dependency-pin/fork changes,
  commits, homebrew launches or perf runs. Logs are local temporary artifacts
  under `/tmp/nixe-task8-close-*`, not new tracked test infrastructure.

  **Closure:** no known Task 8 blocker remains; Task 9 can proceed. Native
  AArch64 ordering/cache-coherence validation remains Task 10. General metadata
  compression, lazy-flag recipe separation and cache-policy tuning remain
  separate work; these tests do not establish commercial-workload scalability.
