# CPU performance audit: A2, A5 and A10

This audit implements the corresponding items in
`notes/performance-audit-2026-10-05.md`. The baseline is the working tree at the
start of this task, including earlier uncommitted optimizations; it is not Git
HEAD. Earlier performance gains are not counted again here.

## Changes

### A2: shorten attributed native indirect transfers

The existing native attribution identified LCQ/HCQ dispatch and link code as a
significant generated-code category. This change targets indirect probes,
including the lookup tail used by successful return predictions.

- Compact immutable native PIC records from 56 to 32 bytes.
- Compare target PC, complete source code version and state-map ordinal.
- Remove redundant address-space, profile and FP-context comparisons from each
  native way. Cold preparation already requires the complete target execution
  key to match the source context at its destination PC. A process-local,
  never-reused source code version and map ordinal identify that context.
- Retain complete cold bridge keys, target reachability/code versions, strong
  owners, protected installation and mandatory unlink before reclamation.
- Keep return-stack context checks and prediction behavior unchanged.

This is an optimization of observed adapter work. It does not claim FP-mode
hoisting, new FRINT lowering, cross-call inlining, a larger indirect cache or a
new baseline backend.

### A5: cheaper demand compilation and validated warmup hints

- Decode each exact instruction word once per coherent capture, including
  restart handling when tracking changes bytes or shortens the captured prefix.
- Reuse per-compiler capture scratch and a dense exit-map index, replacing
  repeated allocated-map scans.
- Instrument capture/decode, IR lowering, backend compilation, adapter staging,
  installation/publication, contract validation and bridge construction.
- Add a versioned, content-keyed persistent warmup profile containing module
  offsets and exact guest words. Placement is excluded from identity; context,
  compiler/ABI revision, host ISA and code-generation policy are included.
- Rebuild all executable code, entries, fault maps, dependencies, lifetime
  identities and links from currently executable memory. No old native code or
  process pointers are restored.
- Use one worker in the existing fixed compiler pool, service queued HCQ work
  between batches, skip competing demand claims and cancel maintenance/startup
  waits during shutdown. Publication and backend failures remain visible.
- Bound each profile to 65,536 fragments and 32 MiB of payload; retain eight
  content-keyed profiles with atomic file replacement.

Warmup is **disabled by default**. Explicitly set `warmup_profile = true` in
`[cpu]` to collect and consume it. Measurements below show that proactive
compilation does not improve this workload's startup. The normal demand path
receives the decode/indexing improvements regardless of that setting.

Backend timing includes backend lowering, register allocation and emission.
Those backend subphases are not reported as separately measured exclusive
costs. The specialized-emitter estimate in the historical plan is not an
achieved 2–4x compilation speedup.

### A10: reduce sampling and immutable metadata work

- Use cheap bounded-table placement while retaining complete execution-key and
  version equality on collisions.
- Back off stable seed/boundary admission attempts at 1, 2, 4, 8 and 16 sample
  deadlines. Initial hotness thresholds are unchanged. New successors,
  ownership/reachability changes and deferred requests reset the delay.
- Hash the immutable PC for family ownership placement, retaining full key and
  generational owner checks.
- Share immutable bindings with a dense 68-value architectural index. Charge
  the index and both allocations to the existing metadata budget.
- Accept validated immutable units for bridge emission. Keep dynamic ABI,
  admission and lifetime checks; remove repeated exhaustive contract validation
  and per-target linear binding searches.
- Keep raw-contract validation helpers only in tests; production uses the
  checked-unit path.

## Correctness and architecture audit

All native addresses remain owned and version protected. Neither a placement
hash nor a short fingerprint authorizes execution. Retired targets clear native
ways before their owners are released. Publication still validates complete
contracts and live executable captures.

The persistent profile uses the normal compiler/publication path. Regression
coverage checks content/context mismatch, ASLR relocation, changed live bytes,
execute permissions, malformed/truncated records, bounds, normal invalidation,
eviction, constructor failure and shutdown. A dedicated maintenance test verifies
that warmup cannot acknowledge/reopen a foreign stop and can cancel without
waiting for that owner.

Native transfer tests cover both table ways, collisions, full source versions,
map ordinals, PC mismatch, alignment, flags, aliases and spilled operands. Cold
preparation explicitly rejects address-space, profile, platform and FP mismatch.
The x86 host executes native tests; AArch64 encoding/contract tests do not amount
to execution on an AArch64 host.

The full JIT library suite passed **1,041 tests**. The added foreign-maintenance
regression passed separately. Runtime, CLI and configuration tests passed,
including the opt-in default. Workspace Clippy passed with default features and
with `frame-trace,jit-profile`; formatting and diff-whitespace checks passed.
A concurrent rejection fixture was corrected to retry legitimate nonblocking
`Deferred` admission rather than assuming immediate queue acceptance.

## Measurement method

- Gato Roboto, first save slot, the same confirmed reference gameplay room.
- Frozen release binaries, the same configuration and warm GPU cache, disposable
  save copies. No concurrent builds or tests during counted windows.
- Three default-build runs before and after, each with 60 seconds of gameplay
  after identical navigation and a 20-second settling interval.
- FPS sampled from the window once per second. Frame-normalized CPU counters
  use estimated frames from those readings, not an exact presented-frame count.
- `perf` user-cycle sampling at 99 Hz, plus task-clock, cycles and instructions.
- Separate startup trials identify the title screenshot, with approximately
  two-second polling resolution. Reported startup times exclude profile flushing
  after the title has been confirmed.
- Separate opt-in diagnostic runs provide native attribution and a bounded
  timeline; their FPS is not substituted for the default-build result.

Baseline executable SHA-256:
`f14070e0ec052cc7e7104cb778a449a691917c2740a70b0451ed0399d343a46f`.
Final default executable SHA-256:
`d1b0eb2bdd3e6dc366c526961da3f7042d013cc9e23d56ca1c6c30f2a6713148`.
Final diagnostic executable SHA-256:
`54d0102e45900dc35862431d5c7e5aabec0643bd7a95572515f712ccdf03dbaf`.

## Default-build results

| Metric, mean of three runs | Before | After | Change |
|---|---:|---:|---:|
| Gameplay FPS | 26.062 | 25.972 | -0.35%; no measurable gain |
| User CPU ms / estimated frame | 76.545 | 75.744 | -1.05% |
| Million user cycles / estimated frame | 185.933 | 182.318 | -1.94% |
| Million instructions / estimated frame | 170.711 | 167.701 | -1.76% |
| Title readiness, seconds | 67.690 | 63.885 | -5.62% |

FPS runs: before **25.700, 25.618, 26.868**; after **25.555, 25.770, 26.590**.
The FPS difference is smaller than the variation among runs. This result does
not establish a gameplay framerate improvement.

Title-readiness runs: before **71.129, 66.041, 65.901 s**; final default
**63.865, 62.783, 65.008 s**. Medians are 66.041 and 63.865 seconds, a 3.29%
reduction. Polling resolution and run variation limit precision.

## Optional warmup result

Three matched cold/warm pairs with profile collection enabled:

| Mode | Title-readiness runs, seconds | Mean |
|---|---|---:|
| Empty CPU profile, collection enabled | 64.960, 64.976, 65.916 | 65.284 |
| Learned CPU profile enabled | 67.005, 67.992, 68.106 | 67.701 |

The learned profile contained 65,536 entries and successfully published over
42,000 fragments. Warm readiness was **3.70% slower** than the corresponding
empty-profile mode and effectively unchanged from the original baseline.
Validated proactive compilation is functional, but is not a demonstrated
startup improvement here. It remains an explicit option instead of adding that
cost to the default path.

## Remaining measured costs

The final diagnostic had no trace/export drops or I/O failure. All sampled JIT
candidate cycles mapped to a live, versioned native publication; 5.13% of total
samples remain unresolved host code. Percentages below are sampled process CPU
cycles, not percentages of frame wall time.

| Native/runtime category | Sampled cycles |
|---|---:|
| Generated guest lowering, both tiers | 13.90% |
| Generated dispatch/link code, both tiers | 11.87% |
| Generated entry/exit adapters, both tiers | 6.49% |
| Other generated scaffolding | 5.47% |
| Graphics frontend | 19.71% |
| JIT sampling | 1.04% |
| JIT compilation during settled gameplay | 0.006% |

This explains why reducing cold compilation and a subset of native-probe work
has little settled-gameplay FPS effect. Native boundary traffic still consumes
more sampled cycles than guest lowering. The graphics frontend also
remains substantial. These changes do not remove those larger costs.

The complete eight-second timeline covers 199 frame-production intervals,
averaging **40.011 ms**. Disjoint main-thread attribution assigns **27.783 ms**
to guest execution elapsed time, **2.597 ms** to dispatch queueing and
**5.085 ms** to result collection. Execution spans include host preemption; they
are not pure native-computation time. Normalized to the 200 frontend submissions, frontend spans accumulate
**24.824 ms**, backend-submit spans **9.701 ms**, and device intervals roughly
**1.30 ms** per frontend submission. Those stages overlap and must not be added
to the disjoint frame partition. This scene is not limited by device execution
alone, and the main CPU path remains above a 16.667 ms frame budget.

## Startup compilation diagnostics

Separate instrumented runs captured the first 20 seconds of startup with no
dropped events. The following are sums of elapsed spans over compiler threads,
not additive wall-clock startup savings.

| Final LCQ phase | Completed spans | Accumulated elapsed time |
|---|---:|---:|
| Capture/decode | 93,462 | 0.226 s |
| IR lowering | 93,440 | 0.992 s |
| Backend, including allocation/emission | 93,440 | 4.690 s |
| Adapter staging | 93,440 | 1.447 s |
| Install/publication, excluding nested validation | 93,440 | 2.165 s |

Contract validation accumulated 0.105 s over all units, including HCQ. Bridge
construction accumulated 1.828 s over 226,657 emissions outside and inside
compilation. These nested measurements are not added again to their parents.

The existing whole-LCQ span averaged **99.785 us** before and **101.301 us**
after, over approximately 93,400 fragments in each run. The final diagnostic
adds several timing spans, and this is not an identical-fragment microbenchmark;
it does **not** demonstrate a net compilation-latency improvement. The backend
remains about half the measured LCQ span. A specialized emitter and persistent
native-code restoration were not implemented or benchmarked in this change.

## Artifacts and reproduction

Raw captures are under `/tmp/nixe-a136/a2510-*`; frozen binaries, logs and
`measurements.json` are under `/tmp/nixe-a2510-20261007`.
Local utilities in `dev-tools/performance/a2510/` provide startup confirmation,
phase accounting and the combined report. Gameplay capture uses
`dev-tools/performance/a136/run.py` with the same fixture and timing for both
builds. These local artifacts contain no ROM in the repository.
