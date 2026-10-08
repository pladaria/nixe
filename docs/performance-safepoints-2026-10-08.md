# Safepoint batching and gameplay measurements

## Scope

Implement the four agreed changes: share stops across queued memory operations,
batch deferrable JIT maintenance by time, narrow memory exclusion where it is
unnecessary, and preserve prompt mandatory stops. This report compares only this
change against the working tree at its start; earlier CPU/frontend optimizations
are part of the baseline.

## Implementation and architecture audit

- **Shared memory stops:** the canonical execution gate now assigns FIFO tickets
  to exclusive operations. Admission stays closed until queued operations finish,
  so a CPU cannot resume between them and require another stop. Every operation
  retains its own mutation handshake, failure handling and committed epoch.
  This groups already queued operations; it does not speculate about future
  operations or reorder guest GPU commands.
- **Timed optional link installation:** publishing performance-only static links
  keeps JIT admission open and retains the existing callable branch or canonical
  fallback. The first request establishes a 1 ms batch deadline. Canonical slice
  entry checks the deadline only while optional work exists; generated branches
  perform no host clock reads. Deferred work preserves its sequence and moves to
  the next batch without keeping the native control word asserted. Existing
  mandatory stops can service links sooner. Warmup, positive publication and
  negative publication follow the same optional scheduling rule.
- **Narrow tracking captures:** registering another CPU-write dependency on
  already protected pages locks only those canonical pages, in identity order.
  It neither closes process-wide execution admission nor mutates protections.
  Direct write faults and checked writes must acquire the same page locks before
  disarming tracking. Observer registration and baseline summary sampling finish
  before unlocking, so a racing first write cannot be acknowledged accidentally.
- **Mandatory safety:** unarmed tracking, mapping/protection changes, code
  retirement, visibility changes, eviction and shutdown retain immediate
  coordination. CPU preemption and interrupts are checked before optional
  maintenance. Native control polling remains independent of frame cadence;
  there is no cap of one safepoint per frame or delayed memory invalidation.
- **Attribution:** capture builds distinguish preemption, closed JIT admission,
  demand and optional maintenance, and identify the thread requesting each
  memory gate operation. Already protected captures and link batch stops have
  separate events. These events compile away in the default release build.

The old optimistic tracking inspection/retry path and its unused page query
were removed. There is one memory gate, one JIT maintenance owner and no new
background timer thread. Page locks never survive into a memory/JIT rendezvous.
The memory dependency boundary allows only the platform-neutral trace crate in
its ordinary dependencies.

## Validation

The regression checks include FIFO exclusion across queued transitions, a
capture completing while CPU execution remains admitted, immediate notification
when tracking requires new protection, optional requests retaining tickets
without closing admission, due batches notifying existing native readers,
mandatory requests bypassing optional deadlines, and negative publication not
forcing a premature batch.

A pre-existing parallel runtime test assumed the compute worker could finish
only once before another worker's supervisor call. It now validates each compute
completion independently of host delivery order, preserving its intended check
that one CPU's supervisor call does not reset another CPU's quantum.

Checks passed:

- Memory: 77 unit tests and the dependency-boundary test.
- JIT: the full 1,044-test release suite, followed by both negative-publication
  tests including the additional regression (1,045 distinct tests covered).
- Graphics frontend: 680 passed, four intentionally ignored.
- Runtime, direct CPU memory and tracing: unit and integration checks passed.
- Workspace Clippy with all targets and warnings denied, plus the CLI capture
  and JIT profiling feature combination.
- `cargo fmt --all` and `git diff --check`.

## Measurement protocol

The game is launched from a frozen default release binary, enters the first save
slot using B three times, and reaches the room supplied in the request. Each run
uses a disposable copy of the same save. Screenshots verify the room before
capture. Startup, title and save-selection time are excluded. After 20 seconds
of settling, each of three runs records a 60-second gameplay window with the
same perf settings and window-title FPS sampling once per second.

A separate run on each side records 20 seconds of perf/JIT data and an eight
second frame trace after entering that room. Trace CPU costs and safepoint
counts are normalized per frontend frame submission; traced runs are diagnostic
and are not mixed into the default-build FPS comparison. Total process CPU per
frame is estimated from perf task-clock and sampled mean FPS; it includes
concurrent threads and is not frame wall time. No builds or tests run during
counted measurement windows.

Artifacts: `/tmp/nixe-safepoints-20261007/` contains frozen binaries, source
baseline, check logs and summaries. Raw captures are under
`/tmp/nixe-a136/safepoints-{before,after}-{r1,r2,r3,diag}/`.
`dev-tools/performance/safepoints/analyze.py` summarizes the matched captures.

## Default-build gameplay results

| Run | Before FPS | After FPS | Before CPU ms/frame (estimated) | After CPU ms/frame (estimated) |
| --- | ---: | ---: | ---: | ---: |
| 1 | 27.43 | 29.05 | 68.41 | 64.14 |
| 2 | 27.61 | 28.48 | 66.95 | 66.61 |
| 3 | 27.06 | 28.74 | 69.33 | 65.06 |
| Mean | 27.37 | 28.76 | 68.23 | 65.27 |

The combined change improves sampled gameplay FPS by **5.1%** and reduces total
CPU per estimated frame by **4.3%**. Mean frame wall time inferred from FPS falls
from 36.54 to 34.78 ms. Host instructions per estimated frame fall by only 1.3%
and host cycles by 2.1%. These are repeated measurements of one stationary
in-game room, not a guarantee for other rooms or a formal confidence interval.
There are no isolated measurements attributing a separate FPS gain to each of
the four changes.

Default release binary SHA-256:

- Before: `4df3b700d0d156243edbbb5055afd55d044b749971fc59fd7a042548bbe9b4f1`.
- After: `8b812d9df442f8aed94bae08750e5b70aa13ad36909b3432615efc1f16c9be41`.

## Diagnostic result and remaining limit

Both eight-second trace prefixes contain 220 frontend submissions and no dropped
events. This is one diagnostic capture per side, separate from the default-build
FPS measurements.

| Diagnostic metric | Before | After | Change |
| --- | ---: | ---: | ---: |
| Main-vCPU safepoint exits | 32,610 | 24,790 | -24.0% |
| Main-vCPU safepoints/submission | 148.23 | 112.68 | -24.0% |
| Frontend tracking captures closing memory admission | 69,442 | 928 | -98.7% |
| GPU-owner tracking captures closing memory admission | 35,860 | 87 | -99.8% |
| Main execution CPU ms/submission | 23.64 | 25.17 | +6.5% |
| Frontend CPU ms/submission | 19.90 | 19.19 | -3.6% |
| Main guest instructions/submission | 2,048,374 | 2,050,266 | +0.1% |

The frontend and GPU owner perform 104,440 metadata-only captures without a
process-wide memory stop in the after trace. Removing those closures lets CPUs
run across them; other reads and mutations then encounter active execution more
often. For example, frontend exclusive reads interrupt active execution 10,177
times after the change versus 1,952 before, and GPU-owner mutations do so 10,872
versus 4,867. Therefore the reduction in capture closures does not translate
one-for-one into fewer final CPU exits. Mandatory memory operations remain
process-wide where the current protocol requires execution exclusion.

The diagnostic capture does **not** demonstrate a reduction in main-vCPU CPU
cost: that interval increases by 1.53 ms/submission, and traced frame cadence is
essentially unchanged (36.35 versus 36.28 ms). These figures must not be mixed
with the +5.1% default-build FPS result. Instrumentation, profiling and JIT tier
selection affect the diagnostic execution; one capture per side does not isolate
the cause of this difference.

Timestamped JIT publication attribution maps 100% of sampled JIT candidates on
both sides, with zero dropped profiling events. Across all threads, identified
sampled user cycles are 91.7% before and 94.3% after. In the after capture,
approximately **53.9% of main-vCPU sampled cycles** belong to generated dispatch
links, entry/exit code and scaffolding; **28.7%** belong to guest instruction
lowering. These percentages are normalized to that thread, not to the entire
process. Profiling export consumes about 0.34% of total sampled user cycles.
The remaining runtime and unresolved samples are not relabeled as guest code.

The 25.17 ms main execution CPU interval and 19.19 ms frontend CPU interval still
exceed the 16.67 ms frame budget individually. They overlap and must not be added
as frame wall time. The change removes unnecessary coordination, but does not
remove the much larger recurring generated-code and graphics work. Reaching
60 FPS still requires reducing native transfer/entry/exit overhead and frontend
work, as well as further grouping or narrowing the remaining memory operations.

All benchmark processes exited cleanly with `HostRequested`; original save data
was not used as a write destination. The default release binary was restored.
No superseded tracking retry/query path remains, and the additional audit
fixtures and both Clippy configurations passed after their final updates.
