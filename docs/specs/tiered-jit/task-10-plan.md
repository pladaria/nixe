# Task 10 implementation plan

Status: step 1 complete; steps 2–8 open. Native Arm host reachable as `ssh pi5`.

Working checklist for [Task 10](spec.md#task-10-prove-cross-target-conformance).
Follow [CONTRIBUTING.md](../../../CONTRIBUTING.md). Code and this active plan
define the work; completed plans are evidence, not competing implementations.
Update each step in place with findings, fixes and actual validation results.

## Scope

Validate the surviving tiered JIT on native Linux x86-64 and AArch64. Reuse
conventional tests and production compilation, publication and admission.
Fix demonstrated correctness/performance problems with focused regressions;
remove superseded code when fixing them. No new testing framework, alternate
executor, production measurement fields or compatibility path for old tests.

ABI integration steps 1 and 3 are implemented: constant maps and explicit
subtraction-flag terminals must execute on hardware. Software-FPSR SSA is
deferred; exact-FPCR specialization is unimplemented and outside this task.
Test current dynamic FP semantics, not hypothetical specialized production
paths. Isolated retained-contract tests do not establish their integration.

Native conformance and the spec's external performance acceptance are distinct
results. Missing workload/reference inputs must not stop useful native testing,
but neither passing tests nor capped homebrew FPS closes the performance gate.
Do not silently waive that gate or change its thresholds. Record any approved
scope change explicitly before claiming Task 10 complete.

This plan does not authorize immediate remote installation, reboot, homebrew
execution or perf collection. Execute the agreed steps when requested; obtain
approval before changing boot configuration/rebooting or starting manual
workloads. No commits, pushes or portable dependency-pin changes unless asked.

## Starting point

Read-only inspection during planning found:

- Nixe HEAD `8c4f4f2905ac2d4fa5bb1f14ea0a56ca8be3ffdb`, with pending ABI
  integration changes, including an untracked runtime regression file. HEAD
  alone does not identify the source to validate.
- Local Wasmtime `nixe` HEAD
  `3a673a02ce35004fbe460d930d12276e2498c88e`, clean at inspection. The
  portable Nixe pin is older; retain the documented four-crate local override.
- Pi 5: Debian 13/Trixie, AArch64, four CPUs, approximately 4 GiB RAM, 2 GiB
  swap and 50 GiB free on SD. Kernel `6.18.50+rpt-rpi-2712`, **16 KiB pages**.
  `cc` and `git` are present; Cargo/Rust/perf were not found on the SSH PATH.
- **Execution prerequisite:** `crates/memory/src/direct.rs::validate_host_limits`
  requires 4096-byte host pages. The installed `kernel8.img` has matching
  `6.18.50+rpt-rpi-v8` configuration with `CONFIG_ARM64_4K_PAGES=y`; the running
  `kernel_2712.img` uses 16 KiB. No boot setting was changed during planning.

Refresh these facts before execution. Keep raw logs, disassembly and profiles
under ignored `dump/`, with a short result summary here. Record exact commands,
source revisions plus dirty/untracked inputs, resolved dependency sources,
toolchain, host/kernel/page size and relevant CPU features. This is evidence
for the tested build, not another baseline-verification tool.

## Steps

- [x] **1. Prepare the two native hosts and identical source inputs.**

  On `pi5`, inspect the current boot configuration, preserve a recoverable copy,
  and agree the boot/reboot operation with the maintainer. Select the installed
  4 KiB AArch64 kernel for this Pi; after reboot verify SSH, `uname -m`, kernel
  version and `getconf PAGESIZE` (must be 4096). Do not weaken the direct-memory
  check, silently enable an interpreter fallback, or undertake a 16 KiB guest
  protection redesign as part of host setup. The separate portability follow-up
  is documented in [next.md](next.md). Document the existing 4 KiB host
  requirement in `docs/host-requirements.md`.

  Install the repository-pinned Rust toolchain via rustup and the actual native
  build/debug dependencies required by the selected crates. Start Cargo builds
  with one job on the 4 GiB Pi; adjust only after checking memory pressure.
  Keep test concurrency separate from build jobs: lifecycle tests must still
  exercise concurrent readers and workers. Do not count swapping/throttling as
  a JIT performance regression; record them during measurements.

  Prepare dedicated Nixe and Wasmtime working directories without overwriting
  unrelated remote files. Transfer all intended source changes, including
  untracked files; do not copy x86 build products or unrelated private data.
  The maintainer explicitly authorized transferring `keys/` and
  `roms/homebrew/` to this Pi; keep key contents out of logs and Git.
  Configure the existing local override with Pi-local paths and confirm Cargo
  resolves the intended fork. Compare source inputs on both machines; a Git
  clone of HEAD alone omits the currently pending Nixe work. Use `--offline`
  only after dependencies have been fetched. No QEMU runner on the Pi.

  **Exit:** both native hosts build the same intended code; the Pi uses AArch64
  userspace and 4 KiB pages. Exact source identity, commands and host facts are
  recorded; setup has not changed production semantics.

  **Completed (2026-09-24):** the maintainer selected `kernel8.img` and
  rebooted the Pi. Verified `aarch64`, kernel `6.18.50+rpt-rpi-v8` and page size
  `4096`. Rust/Cargo 1.97.1, GCC 14.2, CMake 3.31.6 and perf 6.18.50 are
  installed, with the CLI's native build dependencies. Pi CPU features include
  LSE (`atomics`); `vm.max_map_count=1048576`. Host memory requirements are
  documented in `docs/host-requirements.md`.

  Source directories are `/home/pladaria/projects/nixe` and
  `/home/pladaria/projects/wasmtime` on both hosts. `scripts/sync-pi5.sh`
  transfers working sources (not Git history), explicitly authorized homebrews
  and keys, excluding build/runtime caches. Nixe source is the starting HEAD
  above plus the pending ABI integration changes; Wasmtime remains clean at
  the starting revision above. A checksum dry run found no source differences
  in either repository (only the two documentation files being updated here).
  `Cargo.lock` was included in the comparison. The local evidence directory
  `dump/task10-setup-20260924/` records Nixe's diff/status and source-file hashes,
  Wasmtime's empty diff, host/tool versions and the checksum comparison.

  `/tmp/nixe-observable-fp-local.toml` is installed on the Pi with the four
  patches pointing into its local Wasmtime tree. Cargo metadata confirms
  `cranelift-codegen`, `cranelift-frontend`, `cranelift-native` and
  `wasmtime-internal-jit-icache-coherence` resolve to those paths on both hosts.
  Native debug CLI builds succeeded (exit 0) on x86-64 and AArch64:

  ```bash
  # Run in ~/projects/nixe on each host; source ~/.cargo/env on the Pi.
  # The Pi build used CARGO_BUILD_JOBS=1; the x86 build used its default.
  cargo --config /tmp/nixe-observable-fp-local.toml build -q -p nixe-cli
  ```

  Build logs are `build-x86.log` and `build-arm.log` in the evidence directory;
  Cargo metadata is recorded separately for each host. The ARM build took
  approximately 11 minutes, including a period overlapping a maintainer-started
  release build that was subsequently stopped. Monitoring observed no swap
  use or throttling; this is setup evidence, not a performance measurement.
  `cargo --config /tmp/nixe-observable-fp-local.toml cli-dev --help` also
  succeeded on the Pi (`cli-help-arm.log`). Run the development CLI through
  Cargo: it supplies the search path for the build-from-source SDL3 shared
  library, which is not installed system-wide. Standalone binary packaging is
  not established by this check.
  No homebrew or native regression suite was run for this checkpoint. Step 2
  remains responsible for test execution. The override is temporary and must
  be recreated if `/tmp` is cleared; portable dependency pins are unchanged.

- [ ] **2. Establish the native regression baseline and coverage gaps.**

  Run the full JIT library/integration suite and CPU, memory, direct-memory,
  interpreter and runtime library tests on both hosts. Use the command set
  below. On the Pi, run the normal fatal-signal subprocess supervisors too:
  the QEMU-specific exclusions in `docs/aarch64-tests.md` do not apply to native
  binaries. Disable core dumps for intentional fatal cases, not signal checks.

  Review test names and assertions against Task 10, not counts alone. Map
  uncovered cases to steps 3–6 here. Recover the Task 1 accepted-instruction
  baseline from history if available; otherwise explicitly record that the
  historical coverage claim is unresolved rather than substituting today's
  decoder inventory silently. Include currently accepted variants added later.
  Reproduce and attribute failures to source, host setup or the test; do not
  turn a failing case into a skip merely to obtain matching totals.

  **Exit:** each host has an actual baseline result, and each failure or coverage
  gap has an owning step. Architecture-specific test counts may differ; common
  architectural behavior and expected outcomes must agree.

- [ ] **3. Prove LCQ/HCQ semantics and architectural observations.**

  Inspect existing compiler, shared-lowering, HCQ-flow and differential tests.
  Fill only missing coverage for accepted instruction variants: widths, aliases,
  zero/SP registers, partial vector writes, carry/overflow, FP exceptional values
  and modes, FPSR accumulation/replacement and exact helper success/failure.
  Compare full observed state, memory effects, exit reason/PC and completed work
  against the interpreter; use independent expected results for delicate cases
  where sharing a helper could mask an error.

  Exercise real cold LCQ, optimized HCQ, selected HCQ entries and mixed-tier
  chains. Do not require background timing to trigger promotion in a deterministic
  compiler regression; use existing publication fixtures and separately validate
  production promotion in the engine tests. Cover constant final maps and host
  subtraction flags, including spill pressure and callbacks. Test the dynamic
  FPCR behavior through actual FPCR writes; no new specialization or FPSR SSA
  producer is required for this task.

  **Exit:** supported semantics and precise observations agree on both native
  hosts. Every discovered defect has a regression using the surviving path.

- [ ] **4. Verify native ABI, links, polling and emitted hot shapes.**

  Run production-backed gateway/native/compiler/engine link tests on both hosts:
  static jumps, nonempty and empty transfers, mixed tiers, indirect PIC hit/miss,
  guest calls, matched/unmatched returns, stack overflow and resumed execution.
  Hits must reach their destination without a Rust resolver. Check pinned and
  borrowed registers, SP, frame bounds, vector widths and FP ownership.

  Cover zero work/zero-cost checks versus rejected empty slices, small/large
  budgets, loop overshoot bounds, sample-only continuation, callback failure and
  forced control. Use the current accounting contract; do not reinterpret an
  invalid budget as unlimited execution.

  Inspect representative final native bytes with external disassembly: no
  System-ABI call/return or full-state roundtrip on resolved hot edges, no
  repeated entry dispatch, correct landing pads, physical transfers, poll charge
  and terminal flag proof. Separate BTI/CET/IBT instruction-shape evidence from
  enforcement tests: run enforcement where hardware/kernel support it and record
  unavailable features, never claim them from bytes alone.

  **Exit:** representative native paths have correct state/accounting and the
  promised hot shapes on both hosts; missing enforcement evidence is explicit.

- [ ] **5. Validate faults, mutation and publication races on hardware.**

  Reuse direct-memory, LCQ/HCQ fault and lifetime/background/reshape tests.
  Verify recovered accesses retry the identical native instruction, preserve the
  prefault prefix and charge it once; stores/atomics and paired/compound accesses
  must not repeat completed effects. Unattributed/nested/fatal faults must retain
  their expected signal and diagnostic behavior, including subprocess tests.

  Exercise self-modifying code, mapping/protection changes and aliases during
  capture, discovery, compilation, staging, publication and execution. Include
  stale work, negative-evidence invalidation, promotion/reshape cutover, shutdown
  and cancellation with live readers/compiler references. Use existing barriers
  and bounded deadlines rather than sleeps to manufacture races.

  Repeat the affected race/reuse subset across cores on the Pi and x86 host;
  record repetition counts and scheduling configuration. Check Arm dual-alias
  cache maintenance and cross-thread publication by executing changed/reused
  code, not by accepting a successful cache-flush call. Audit ordering when a
  failure is found; repeated green tests alone are not a memory-model proof.

  **Exit:** targeted races and signal/retry scenarios pass natively, with no
  stale instruction execution, stale owner use or lost shutdown progress.

- [ ] **6. Verify W^X, bounded storage and reuse under pressure.**

  Use executable/lifetime pressure, reclamation and metadata tests plus OS
  mapping inspection. Verify no RWX virtual mapping, Closed write windows,
  segment decommit/republication, generational slot/span reuse and removal of
  native-PC attribution after the required grace periods. Exercise actual code
  execution after reuse, including links and retired directory readers.

  Drive deterministic pressure/replacement/invalidation with bounded test-owned
  inputs. Confirm compiler/reference holds prevent early reclamation, releases
  permit progress, and settled pressure returns accounted storage below 512 MiB
  without exceeding the 640 MiB hard bound. Preserve the existing 480 MiB
  pressure target and 32 MiB LCQ reserve. Distinguish committed JIT code plus
  coupled metadata from virtual reservations, total process RSS and build memory.
  Check that ordinary maintenance does not scan the whole cache unnecessarily.

  **Exit:** repeated cycles reclaim real storage/metadata and do not show
  unbounded growth, leaked roots or a liveness failure. Accounted limits are
  demonstrated, not inferred from a small demo or a quiet shutdown alone.

- [ ] **7. Run agreed workload smoke checks and external performance acceptance.**

  Obtain approval and caller-owned inputs before running homebrews/profilers.
  Use hello-world, es2gears and textured_cube for startup/correctness and include
  simplegfx as the existing CPU-heavy regression. Confirm the Pi's display/GPU
  setup before graphical tests; a graphics prerequisite is not a JIT result.
  Use existing shutdown diagnostics and external tools for stability, promotion/
  reshape activity and memory; add no per-entry counters or runtime framework.

  The spec's comparative gate also requires an agreed Ryujinx revision/build and
  at least three legal caller-owned commercial workloads with deterministic
  scenes. These inputs are not available merely because SSH works. Freeze the
  required checked-in external-run manifest before either emulator is measured;
  include revisions/dirty inputs, host configuration, cache states, settings,
  scene setup, warm-up and measurement intervals. Never choose scenes after
  seeing results or drop failing workloads from the aggregate.

  Follow [external performance acceptance](spec.md#external-performance-acceptance)
  unchanged: same-host alternating runs, ten cold starts and ten sustained
  samples per workload, at least 60 seconds per sustained sample, and its stated
  statistics and thresholds. Compare emulators on each host, not Pi FPS against
  the x86 desktop. Detect GPU/vsync limits and thermal/power effects before making
  CPU-throughput claims. Keep lawful workload contents out of Git.

  **Exit:** smoke results and comparative results are separately recorded. If
  the reference, corpus or suitable host configuration is unavailable, leave
  the external gate open and request a scope decision; do not invent evidence
  or broaden emulator/game support implicitly to make this task pass.

- [ ] **8. Consolidate final evidence and decide Task 10 closure.**

  After corrections, rerun the full final native matrix against the same source
  snapshot on both hosts, plus strict lint, formatting, production builds and
  affected fork tests. New fixes invalidate the corresponding earlier result;
  keep the final summary tied to the code actually tested.

  Record outcomes here by host and contract: passed, failed, unavailable or
  explicitly deferred, with commands and log locations. Update the host/Arm
  guides with real native instructions; retain QEMU as a supplementary tool.
  Remove temporary probes and resolved finding notes, not useful regressions.
  Keep the local override/pin handoff explicit and preserve unrelated changes.

  **Exit:** all technical criteria and the external performance gate pass on
  both native hosts, or an approved scope change states precisely what remains
  outside completion. Do not label the entire architecture conformant merely
  because the Pi tests pass; deferred ABI optimizations, unsupported enforcement
  features and unmeasured performance claims remain clearly distinguished.

## Baseline commands

Run from each host's Nixe checkout, with its own validated override file. These
are native commands on both machines: no cross-linker or QEMU runner options.
Use `CARGO_BUILD_JOBS=1` initially on the Pi. Redirect full output to a separate
log per command and inspect failures; do not suppress stderr or pipeline status.

```bash
ulimit -c 0
# Set NIXE_CRANELIFT_OVERRIDE to this host's actual Cargo override file.
task10_cargo=(--config "${NIXE_CRANELIFT_OVERRIDE:?set the local override path}" -q)
cargo test "${task10_cargo[@]}" -p nixe-cpu-jit --lib --tests
cargo test "${task10_cargo[@]}" -p nixe-cpu -p nixe-memory -p nixe-cpu-direct-memory -p nixe-cpu-interpreter -p nixe-runtime --lib
cargo clippy "${task10_cargo[@]}" -p nixe-cpu-jit -p nixe-cpu-direct-memory --lib --tests --no-deps -- -D warnings
cargo check "${task10_cargo[@]}" -p nixe-cpu-jit -p nixe-cpu-direct-memory -p nixe-runtime -p nixe-cli
cargo fmt --all -- --check
git diff --check
```

Run fork Nixe boundary/allocator tests and the native fault integration test
from its checked-out source, using its documented features/toolchain. Final
native-shape and affected JIT regression checks must also exercise a release
build; performance acceptance uses release builds, never debug build timings.
