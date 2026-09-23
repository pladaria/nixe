# Retained ABI integration plan

Status: steps 1 and 3 complete; step 2 deferred; step 4 pending.
Approved during the Task 9 export audit.

Keep the four contracts below; their consumers and focused tests already exist.
LCQ/HCQ produce constant maps and subtraction-flag terminal maps; exact-FPCR
specializations and software-FPSR SSA remain pending/deferred as specified below.
Task 9 cleanup is complete. Do not claim
the remaining contracts as production-conformant in Task 10 until their
producer-to-execution tests pass.
No new runtime framework or parallel compiler path is required.

## Work

- [x] **1. Emit constant locations from final allocation.**
  Inspect the fork's rematerialization and final boundary-map representation.
  Extend it where necessary to identify constant operands without guessing from
  pre-allocation SSA. Preserve exact integer/vector bits, including 128-bit
  constants. Translate that representation in `native/backend.rs` to
  `ValueLocation::Constant`, using the existing transfers and reconstruction.
  Replace the test-only constructor with the actual production constructor.
  Verify optimized LCQ/HCQ entry, exit and fault maps and execute their canonical
  and linked paths. Compare native size and boundary cost: do not add a runtime
  map lookup or duplicate resident representation.

  **Implemented:** the local fork carries literal iconst/f32const/f64const/vconst
  bits through boundary lowering/allocation into final Constant locations.
  Other values still use actual register/spill assignments. Entry definitions
  cannot be constants. Nixe consumes these maps in existing reconstruction and
  native transfers; its constant constructor is no longer test-only. Backend
  proof operands grow transiently, within existing accounting, and are discarded
  after publication validation; resident ValueLocation layout is unchanged.
  Optimized small x86 immediates/direct stores and zero/ones SIMD transfers;
  AArch64 repeated-byte vectors use MOVI. All preserve host condition flags.

  **Validation:** full x86-64 JIT suite (981 unit + 4 integration tests), 66
  AArch64/QEMU ABI/link/fault/runtime tests + 4 integrations, 45 fork Nixe tests
  (both targets, single-pass/unoptimized and backtracking/optimized allocation)
  and the fork's native fault test. Strict Clippy, both-target production checks,
  formatting and whitespace checks pass. The fault-pressure regression retains
  its spill/recovery assertions; removed only its incidental x86 allocation-cycle
  requirement, already covered deterministically by native bridge tests.

  **Size comparison:** same LCQ MOVZ sequence and ADD-consuming linked target,
  before/after constant-map emission; bytes include each unit's exit support.
  A temporary probe was removed after measurement. Logs are in
  `/tmp/nixe-abi-constants-*.log`; all Nixe commands use the local Cargo override
  and Arm commands use the documented QEMU runner.

  | Host | Constants | Unit bytes before → after | Bridge bytes before → after |
  | --- | ---: | ---: | ---: |
  | x86-64 | 1 | 528 → 528 | 3 → 6 |
  | x86-64 | 8 | 640 → 576 | 38 → 83 |
  | x86-64 | 24 | 1200 → 736 | 245 → 235 |
  | AArch64 | 1 | 336 → 336 | 4 → 4 |
  | AArch64 | 8 | 416 → 400 | 36 → 64 |
  | AArch64 | 24 | 672 → 528 | 104 → 192 |

  Materialization can move from the body into the bridge: a smaller unit does
  not imply a cheaper boundary or a faster workload. The fork regression also
  verifies that 80 boundary-only literals add no native instructions or spill
  extent versus an empty exit. No cycle/latency benchmark, homebrew or perf run
  was performed; no application speedup is claimed. Fork changes remain local
  on top of 0380097992d7d337bdf66873510a8c42a8923c0b; retain the override until
  publishing/pinning them separately. Native Arm hardware validation is pending.

- [ ] **2. Carry software FPSR in SSA where it is live.**
  Connect `GuestValue::Fpsr` through shared frontend values and HCQ flow/SSA.
  Keep the existing host-FP owner: software status and pending host status are
  distinct, and must merge exactly once at observations. FPCR/FPSR writes still
  terminate the old segment in architectural order. Verify inactive FP, active
  FP, helper success/failure, instruction faults and cross-tier links against
  the interpreter, including FPSR replacement rather than accumulation.
  Do not add unconditional FPSR loads/stores to integer-only paths.

  **Deferred by agreement:** the shared `fp_policy.rs`
  currently has no inline software-status producer. Guarded native operations
  accumulate status in the host; guarded result-only paths leave FPSR unchanged;
  exact completion in `lcq/fp.rs` updates canonical state after native execution
  ends. `frontend/system.rs` also exits before FPSR observations/replacements.
  Consequently, `hcq/flow/native.rs` deliberately excludes FPSR from native SSA,
  while architectural liveness and the invocation still preserve its semantics.
  Merely adding a Values slot and threading FPSR through FP operations would
  transport an unchanged canonical value, without eliminating any observation
  or helper boundary. It could increase register pressure and boundary traffic.

  Outside this integration plan's completion scope: reactivate when a concrete
  helper optimization selects an inline software-status producer. Implement
  that lowering together with its liveness, maps and differential tests, not
  the transport alone. Retain the approved ABI contract; this is deferred, not
  implemented, and has no scheduled task. Existing focused
  x86-64 tests pass (19: FP ownership, HCQ flow/activation and real-gateway FPSR
  observation/replacement), using the local override. No runtime code changed,
  no homebrews ran, and these tests do not establish FPSR SSA integration.

- [x] **3. Produce allocator-backed host-NZCV contracts.**
  First extend the fork's explicit flag ownership, clobber constraints and
  final-map reporting so Nixe can prove which guest bits survive. Never infer
  `NzcvLocation::Host` from the preceding machine instruction. Then select it
  in the shared frontend only for compatible guest operations; retain packed
  and deferred recipes for other cases. Account for x86 carry inversion and
  AArch64 NZCV, polls, arena-address arithmetic, allocator edits, PIC/RSB probes,
  helper suspension and retry. Execute producer-to-consumer chains on both
  targets and retain the existing no-Rust-return hot-link contract. Remove the
  narrow dead-code expectation only once a production producer exists.

  **Implemented:** LCQ/HCQ subtraction recipes now request a fused terminal
  comparison in the local fork. Its two inputs require allocated registers;
  flexible map-only inputs may spill. Each hot/deadline path executes CMP after
  the poll decision, and final maps explicitly export subtraction flags. x86
  borrow is inverted when consuming guest C. The flags-only subtraction result
  is removed from terminal operands; other observations retain their recipes.
  Cold polls preserve explicit host flags, and sampling chooses success/failure
  before restoring them. No Rust call or canonical roundtrip was added to a
  resolved hot link. The production Host variant no longer needs a dead-code
  exemption. This recomputes subtraction flags at the terminal; it does not
  retain ambient flags through the preceding body or introduce host-flag entry
  definitions/fault maps.

  **Producer coverage:** mixed LCQ/HCQ chains consume carry in ADC and observe
  final NZCV with MRS. Both tier orders, W/X subtraction, static links, BR/BLR
  PIC hits and misses, matched RET predictions and sample-only resumption match
  the interpreter. Hits execute with no Rust resolver; misses preserve the
  source prefix on canonical escape. Callback success/failure with active and
  inactive FP is covered separately. Entry, prefault, retry and internal-check
  contracts retain their existing packed/deferred/canonical representations.

  **Cost review:** the first checkpoint unnecessarily converted host flags to
  packed guest NZCV and back around every cold poll/callback. Preserve raw flags
  there instead (LAHF/SETO and ADD/SAHF on x86, MRS/MSR on Arm); pack only at real
  architectural consumers. Packing also uses only reserved scratch or saved
  RAX, rather than saving three mapped GPRs. No resident metadata, frame space,
  runtime switch or helper was added.

  A removed temporary probe compared the previous recipe producer against the
  final implementation, with the same two LCQ units: a CMP/branch source and
  an ADC/branch target.
  Unit sizes include cold support; bridge sizes exclude the final jump:

  | Host | Source unit bytes | Transfer bytes | Source + transfer bytes |
  | --- | ---: | ---: | ---: |
  | x86-64 | 800 → 848 | 310 → 178 | 1110 → 1026 |
  | AArch64 | 568 → 456 | 272 → 92 | 840 → 548 |

  CMP/BRK units shrink from 432 to 272 bytes on x86 and 328 to 152 on Arm.
  x86 indirect source units grow by 80–160 bytes for explicit flag preservation;
  Arm BR/BLR/RET units shrink by 48–96 bytes. Native bridges are accounted
  separately, so this is not a claim that every individual unit gets smaller.
  Nine x86 runs of 10 million two-link loop iterations measured median 8.24 →
  7.50 ns/iteration (ranges 8.21–10.09 → 7.36–7.86); admission/compilation is
  outside the timing, native sampling remains enabled, with no callback. This
  is a focused boundary probe, not an application benchmark or an Arm timing.
  Logs: `/tmp/nixe-abi-hostflags-cost-{baseline-final,final}.log`.

  **Validation:** 985 x86-64 unit + 4 integration tests; 70 AArch64/QEMU native,
  runtime, poll and terminal tests; strict Nixe Clippy. This includes all 16
  flag combinations and masks, both carry conventions, register preservation,
  and the producer-backed paths above. The preceding fork checkpoint passed
  47 Nixe tests across both targets and allocators, including spill pressure
  and dead-terminal metadata removal; no further fork changes were needed.
  Final logs: `/tmp/nixe-abi-hostflags-complete-{x86,arm,clippy}.log`.

  This closes the selected subtraction-terminal integration. Do not extend
  host-flag ownership to more producers or entry/fault maps without a concrete
  lowering benefit and explicit backend proof; unsupported producers keep
  recipes. No homebrews ran, and the local fork override is still required.

- [ ] **4. Select and publish exact-FPCR specializations.**
  Add an explicit HCQ selection policy, based on observed FPCR and a demonstrated
  lowering benefit; do not compile a variant for every observed value by default.
  Freeze the exact mode into the existing work/key/ownership protocol and consume
  it in FP lowering. Canonical admission, static links, PIC/RSB and resumed
  execution must not enter a specialization with a mismatched mode. An FPCR write
  returns to the existing canonical boundary before another dispatch; retain
  dynamic LCQ availability. Bound variants using existing budgets and registries,
  not an additional cache. Test mode alternation, stale work, reshape, invalidation
  and pressure, and compare compilation/memory costs with dynamic code.

Each item changes the existing producer and consumers together, updates the
living spec, and removes its targeted dead-code expectation when applicable.
Use focused unit/differential and native-shape tests, with QEMU for Arm execution
until hardware is available. Run homebrews/perf only when explicitly requested.
Measurements belong in the existing diagnostics workflow, not a new framework.

If a contract cannot deliver a correct, useful production implementation, bring
that finding back for a scope decision rather than adding a synthetic producer,
forcing a slower hot path, or treating its isolated ABI tests as integration.
