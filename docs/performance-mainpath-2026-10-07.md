# Main execution path and sustained graphics, 2026-10-07

This work targets gameplay in the first saved room, not title startup. The three
fronts are main-vCPU native transitions, coordinator/worker handoffs, and sustained
graphics frontend preparation. These concurrent costs overlap and must not be
added into a serial frame budget.

## Implemented changes

- Return predictions store only 64-bit guest PCs. A shared context records address
  space, profile, platform and FP specialization; canonical admission clears the
  predictions if any of these changes. Native links already validate the complete
  context. Call/return code therefore omits four repeated context loads/comparisons
  and four stored context fields per entry. The guest-thread allocation decreases
  from 648 to 168 bytes. PIC versions, ownership, flags, precise exits and exactly
  once prediction bookkeeping remain unchanged. Both x86-64 and AArch64 emitters
  use the compact layout. The warmup compiler ABI revision is bumped.
- Realtime adaptive parallel execution learns an instruction budget targeting
  1 ms of worker elapsed execution, with half/double adjustment bounds and a
  four-million-instruction cap, or an explicitly larger configured baseline.
  Only exhausted budgets update the estimate. SVC/event stops preserve it.
  Existing control/interrupt polls remain independent. This target is not a hard
  deadline. Explicit budgets and fixed-clock instruction adaptation retain their
  separate semantics; only the realtime policy reads elapsed-time clocks.
- Optional trace capture reports per-thread CPU time for vCPU execution and GPU
  frontend work, plus stop reasons and retired guest instructions. CPU timing is
  absent from default builds. Linux scheduler counters distinguish aggregate
  runnable delay from blocking; elapsed time alone cannot identify avoidable
  communication.
- MME upload decodes instruction fields once into immutable, copy-on-write
  16-word pages. The interpreter caches its current page, preserving old program
  snapshots, arbitrary captured RAM addresses, dynamic reads, branches, delay
  slots, missing-instruction errors and emitted-method order. Shadow registers
  use the direct paged register file. Named class methods use a generated direct
  index instead of a linear declaration search.
- Resource-domain keys describe exact retained register values, allowing restored
  bindings to reuse an older resolution plan. Fingerprints choose candidates;
  exact COW pages reject collisions and distinguish unset values. Constant-buffer
  keys capture the enabled flag, address and size at the binding command. Mapping,
  content epoch, descriptor, physical-alias, backend generation and staged-write
  validation remain mandatory. The existing bounded caches own retained plans.
- Per-method decoding scratch is cleared before every method. An unrelated method
  can no longer inherit the preceding method's semantic write domains. Scratch
  does not participate in persistent channel-state equality.

## Validation and architecture audit

- CPU JIT: 1,042 unit tests passed, including native return probes, flags, FP,
  exact memory faults, invalidation, code retirement and guest-thread migration.
- Runtime: 84 unit tests and 13 integration tests passed. Fixed-clock, explicit
  quantum, priorities, FIFO leases, interruption, quiescence and teardown retain
  their contracts. Focused tests cover elapsed adaptation bounds and preservation
  across SVCs.
- Graphics frontend: 680 unit tests passed, four existing opt-in tests ignored.
  Added regressions cover decoded program page boundaries and retained upload
  versions, execution-time errors, restored vertex/constant-buffer bindings and
  per-method scratch isolation. Existing atomic failure, aliases, mappings and
  resource ownership tests pass.
- Capture: bounded recording/epoch tests pass, including stale CPU spans.
- Workspace Clippy with all targets and warnings denied passes. CLI Clippy with
  `frame-trace,jit-profile` also passes. Formatting and diff whitespace pass.
- Superseded per-entry RSB contexts, resource revision counters, MME per-step
  bit decoding and instruction-tree lookup are removed. Realtime and fixed-clock
  policies remain separate because they represent different clock contracts.

## Measurement protocol

Three default release captures before and after, 60 seconds each, in the same
verified gameplay room. Each run loads slot one with three B presses, verifies
room pixels from a screenshot, settles for 20 seconds, and only then records FPS,
perf counters and per-thread scheduler totals. Saves are disposable copies.
Compilation and tests do not run during counted windows. FPS comes from the
one-second window title samples; counter/frame figures use their estimated frame
count. Run-to-run variation and counter multiplexing limit precision.

Separate diagnostic builds record 20 seconds of perf/JIT publications and an
eight-second timeline prefix. They are used for attribution, not default-build
FPS claims. Per-thread CPU-time events were added in this change, so they have no
identical pre-change event counterpart. Scheduler totals are available in both.
`context-switches:u` returns zero on this host and is not used; scheduler timeslice
counts are scheduling periods, not exact communication or context-switch counts.

Artifacts: `/tmp/nixe-mainpath-20261007/` and
`/tmp/nixe-a136/mainpath-{before,after}-*`. The interrupted
`mainpath-before-01` capture and failed sandbox diagnostic launch are excluded.
The accepted default baseline labels are `mainpath-before-r1`, `-r2`, `-r3`;
the baseline diagnostic label is `mainpath-before-diag-01`.

## Gameplay results

| Metric (mean of three runs) | Before | After | Change |
| --- | ---: | ---: | ---: |
| FPS | 25.700 | 26.911 | +4.71% |
| Total host CPU ms/frame | 77.156 | 70.874 | -8.14% |
| User cycles/frame (millions) | 184.531 | 174.242 | -5.58% |
| User instructions/frame (millions) | 169.045 | 163.183 | -3.47% |

Run means: before 25.353 / 25.785 / 25.962 FPS; after 27.312 / 26.917 /
26.503 FPS. All three after means exceed all three baseline means. The gain is
modest and does not achieve 60 FPS. Host CPU time sums concurrent threads and is
not the frame's elapsed duration. Cycle/instruction counters cover userspace;
these figures do not isolate one change's contribution.

The static room crop `(270,100)-(1650,720)` is pixel-identical before/after,
excluding the animated character/flowers below it. All measured runs exit with
`HostRequested`, code zero, and no rejected SVC kinds.

Frozen default binary SHA-256:

- Before: `d1b0eb2bdd3e6dc366c526961da3f7042d013cc9e23d56ca1c6c30f2a6713148`
- After: `4df3b700d0d156243edbbb5055afd55d044b749971fc59fd7a042548bbe9b4f1`

`target/release/nixe-cli` is restored to the measured default after binary.

## Diagnostic attribution

The same common eight-second prefix has no dropped events in either trace.
These spans include host scheduling and waits. The frame interval partition is
disjoint; frontend durations overlap it on another thread.

| Diagnostic elapsed metric | Before | After | Change |
| --- | ---: | ---: | ---: |
| Frame-production interval | 39.143 ms | 38.170 ms | -2.49% |
| Main guest execution / frame interval | 28.104 ms | 25.182 ms | -10.40% |
| Main worker dispatch queue / frame interval | 2.651 ms | 1.878 ms | -29.15% |
| Main result collection / frame interval | 5.077 ms | 3.794 ms | -25.26% |
| Frontend / submission | 24.539 ms | 23.449 ms | -4.44% |
| Resource rebuild / submission | 6.989 ms | 6.679 ms | -4.43% |
| Draw rebuild / submission | 4.596 ms | 4.414 ms | -3.95% |
| Worker executions in common prefix | 93,935 | 69,270 | -26.26% |
| Resource rebuilds / submission | 61.245 | 55.033 | -10.14% |
| Indexed whole-resource-plan hits | 0 | 1,260 | new reuse |

About six whole resource plans per submission now reuse an older indexed entry.
The remaining rebuilds are slightly more expensive individually; their reduced
count gives a smaller aggregate improvement. Most draws still replace dynamic
bindings, even though their fixed-state validation is reused. The new identities
do not make dynamic content or physical alias checks disappear.

Host-idle wait in the disjoint partition increases from 1.301 to 4.253 ms/frame;
worker receive wait increases from 0.397 to 1.189 ms/frame. These offset much of
the saved execution/handoff time. It would be incorrect to convert the local
savings into a summed FPS prediction or to remove waits without verifying their
dependencies. The diagnostic frame cadence is also distinct from the default
release result above.

### Coordination and off-CPU time

The baseline vCPU-0 scheduler totals over 20 seconds are 14,994 ms CPU runtime,
73 ms runnable delay and 183,989 scheduling periods; after they are 13,877 ms,
36 ms and 124,048 periods. Coordinator CPU runtime decreases from 4,545 to
3,896 ms and its scheduling periods from 230,174 to 177,357. These periods are
not exact message counts, and the raw totals are not normalized for the differing
frame counts. Runnable delay is small compared with the trace's handoff gaps:
host CPU contention does not explain most of those gaps.

New CPU events report 24.801 ms of actual main-thread execution per frontend
submission versus 25.265 ms elapsed in that same grouping. The frontend uses
20.318 ms CPU versus 23.449 ms elapsed. Their gaps include blocking as well as
preemption; Linux runnable delay is only an aggregate cross-check. No pre-change
CPU events exist, so missing baseline values must not be interpreted as zero.

After the change, the eight-second prefix records 29,630 SVC stops, 39,060
safepoints and only 580 budget stops across all vCPUs. The main guest thread has
2,533 SVC stops, 30,999 safepoints and 580 budget stops. Thus approximately 91%
of its exits are safepoints. Further quantum increases cannot eliminate these.
The current safepoint category covers preemption and closed JIT admission;
separating those origins is a remaining diagnostic task, not proof that every
safepoint can be removed. Necessary service, synchronization and interrupt
boundaries remain intact.

### Native-code attribution

Temporal publication mapping identifies 100% of candidate JIT samples in both
captures; publication export and timeline drops are zero. There are 4,768 / 4,536
perf samples. Roughly 7.5% of all sampled cycles still lack a resolved host
category; this does not mean that generated-code candidates are unidentified.

Using each 20-second diagnostic capture's estimated frame count, the sampled
LCQ dispatch/link group decreases from 22.762 to 16.460 million cycles/frame
(about 28%). MME `step` leaf samples decrease from 3.334 to 1.206 million
cycles/frame (about 64%). These are sampled leaf/group estimates, not exclusive
whole-interpreter timings or isolated causal contributions. Inlining and sample
variation affect attribution. Total frontend elapsed improves only 4.4%.

## Remaining limit

The measured main thread still requires about 24.8 ms of CPU execution per frame,
and the frontend about 20.3 ms per submission, plus blocking/handoffs. Both exceed
the 16.67 ms budget for 60 FPS, even before all surrounding work is considered.
On the main vCPU, generated dispatch/link and entry/exit code still account for
about 38% of its sampled cycles; generated guest instruction lowering is about
29%, with scaffolding and runtime occupying the remainder. Steady-state JIT
compilation is only 0.029% of total sampled user cycles.

The next targeted investigations are the origins of repeated safepoint exits,
remaining native link/adaptation traffic, and frontend allocation/copying plus
canonical resource/alias comparisons. The new result establishes reduced CPU
work and a modest gameplay improvement; it does not establish a path to 60 FPS
from a larger quantum or from MME decoding alone.


