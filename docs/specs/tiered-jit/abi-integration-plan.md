# Retained ABI integration plan

Status: planned, not implemented. Approved during the Task 9 export audit.

Keep the four contracts below; their consumers and focused tests already exist,
but current LCQ/HCQ compilation does not produce them. They are explicit pending
work, not evidence of production coverage. Complete Task 9 cleanup first; use
this checklist for the subsequent integration, and do not claim these contracts
as production-conformant in Task 10 until their producer-to-execution tests pass.
No new runtime framework or parallel compiler path is required.

## Work

- [ ] **1. Emit constant locations from final allocation.**
  Inspect the fork's rematerialization and final boundary-map representation.
  Extend it where necessary to identify constant operands without guessing from
  pre-allocation SSA. Preserve exact integer/vector bits, including 128-bit
  constants. Translate that representation in `native/backend.rs` to
  `ValueLocation::Constant`, using the existing transfers and reconstruction.
  Replace the test-only constructor with the actual production constructor.
  Verify optimized LCQ/HCQ entry, exit and fault maps and execute their canonical
  and linked paths. Compare native size and boundary cost: do not add a runtime
  map lookup or duplicate resident representation.

- [ ] **2. Carry software FPSR in SSA where it is live.**
  Connect `GuestValue::Fpsr` through shared frontend values and HCQ flow/SSA.
  Keep the existing host-FP owner: software status and pending host status are
  distinct, and must merge exactly once at observations. FPCR/FPSR writes still
  terminate the old segment in architectural order. Verify inactive FP, active
  FP, helper success/failure, instruction faults and cross-tier links against
  the interpreter, including FPSR replacement rather than accumulation.
  Do not add unconditional FPSR loads/stores to integer-only paths.

- [ ] **3. Produce allocator-backed host-NZCV contracts.**
  First extend the fork's explicit flag ownership, clobber constraints and
  final-map reporting so Nixe can prove which guest bits survive. Never infer
  `NzcvLocation::Host` from the preceding machine instruction. Then select it
  in the shared frontend only for compatible guest operations; retain packed
  and deferred recipes for other cases. Account for x86 carry inversion and
  AArch64 NZCV, polls, arena-address arithmetic, allocator edits, PIC/RSB probes,
  helper suspension and retry. Execute producer-to-consumer chains on both
  targets and retain the existing no-Rust-return hot-link contract. Remove the
  narrow dead-code expectation only once a production producer exists.

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
