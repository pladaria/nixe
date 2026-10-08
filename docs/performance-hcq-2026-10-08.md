# HCQ admission, stable sampling and native register contracts

## Initial iteration: scope and implementation

The initial iteration implemented the admission and sampling work and partially
addressed native transfers. Removing redundant prediction and shortening PIC
probes left register-contract coordination and preservation incomplete. The
subsequent implementation is documented under **Cross-region contracts follow-up**
below, with a separate working-tree baseline and gameplay measurements.
Earlier CPU, graphics and safepoint changes belong to the baseline and are not
counted as benefits of this implementation.

1. **Gather region discoveries before compiling.** The existing bounded worker
   queue opens a fixed 2 ms discovery window when an empty queue receives work.
   Later arrivals cannot extend that deadline. Further observations coalesce into
   queued seed snapshots through a nonblocking queue lock; running compiler inputs
   stay immutable. Discovery and the existing public-entry sweep therefore see
   demands exposed during the window. There is no new timer thread, registry or
   polling clock in generated guest code. The existing queue capacity, fairness,
   ownership reservations and publication revalidation remain in force.
2. **Require evidence of useful promotion or replacement.** Observation counts
   now accumulate beyond the admission threshold and reset when the versioned
   identity changes. A seed with an internal edge or at least eight instructions
   needs eight samples; a straight fragment of two to seven instructions needs
   32, and a one-instruction leaf needs 128. A one-instruction internal loop still
   qualifies quickly. Replacement policy estimates work from old plus new region
   size and benefit from additional coverage, public entries and merged families;
   the required evidence is bounded between eight and 128 samples. This is a
   cost/benefit heuristic, not measured compilation ROI. Unprofitable candidates
   defer before lowering, retain the current callable version and can qualify
   through subsequent execution. Existing structural and unchanged-result
   suppression stays distinct from a temporary lack of heat.
3. **Stop observing boundaries that cannot improve a stable region.** Optimized
   internal cycles and HCQ call/return terminals resume directly after mandatory
   native polls without calling the Rust sampling observer. Discovery does not
   internalize call/return edges. Other proven stable or structurally suppressed
   transfers use a fixed-size vCPU-local cache with exponential observation
   spacing capped at 16 samples. New source versions and destinations bypass the
   cache; bounded revisits detect changed target ownership. Active candidates
   continue collecting heat. Execution budgets, interrupts, code invalidation,
   memory coordination and control polling are not throttled by this policy.
4. **Shorten native calls, returns and indirect transfers.** The previous guest
   return prediction stack added updates at calls and checks at returns, but a
   prediction hit still performed the full source-keyed PIC lookup. It has been
   removed from generated code, the native ABI, guest-thread ownership and vCPU
   dispatch. RET now uses its architectural destination through the same owned,
   validated PIC as other indirect transfers, including nonlocal returns. On
   x86-64, a register proven unused by the source bindings, destination and lazy
   flag recipe retains the PIC record pointer across key comparisons. This avoids
   repeated NativeFrame record stores/reloads. Register pressure and the other
   supported host ABI retain the existing reserved-register lookup.

Eight observations are sampling events, not eight executions of a guest block.
The normal observation period remains 4,096 emulated instructions per vCPU.
The profitability policy changes what merits compilation after observation; it
is neither a per-frame optimizer nor a promise that every small block is wasteful.

## Architecture and correctness audit

These checks validated the initial iteration only. At that stage, bridges still
adapted independently compiled contracts. The follow-up below adds coordinated
allocation and carried values while retaining bridges for genuinely different
contracts.

Queue locks do not nest with JIT ownership locks. Coalescing changes only owned,
version-matched observations before dequeue; all captured code, predecessor
families and incoming entries undergo the existing freeze/publication checks.
Mandatory stops bypass optimizer scheduling, and unpromoted or newly demanded
entries retain their valid LCQ execution path. Stable observation suppression
never substitutes for executable lifetime checks or memory protection.

Native record hits still validate the destination PC, source code version and
state-map ordinal before following a strongly owned bridge. Removing prediction
state does not change the architectural link register or guest return behavior.
Register selection protects deferred flag operands as well as architectural
bindings. The superseded return-stack modules, native emitters, runtime fields,
API parameters, predictor-only tests and internal-callback preservation path
were removed. The warmup compiler-policy fingerprint was advanced.

Real-game validation exposed an x86 emitter assumption: memory operands had
previously used bases that did not need a SIB byte. The new register-held probe
could select R12. The emitter now handles the mandatory RSP/R12 SIB and the
RBP/R13 zero displacement correctly. The regression executes the PIC with every
allocatable x86 integer register as its record base, in addition to the
register-pressure path, checking complete architectural state, flags, both ways,
collisions, absent tables and register/spill/constant/vector destinations. The
failed preliminary run is excluded from all performance results below.

Checks passed after this correction:

- `cargo test -p nixe-cpu-jit -p nixe-runtime --release`: 1,144 tests passed;
  two pre-existing external-fixture tests remain ignored. This includes the
  1,043-test JIT unit suite, differential and dependency-boundary tests, and
  runtime unit/integration tests.
- Native recursive and nonlocal returns, cross-tier flag transfer, unpublished
  targets, thread migration, retirement, invalidation, shutdown and pressure
  tests retain their architectural checks.
- Regression coverage verifies a fixed discovery deadline, queued successor
  coalescing, eventual profitable replacement, version-local heat reset,
  bounded stable revisits and optimized loops retaining precise control exits.
- Workspace Clippy with all targets and warnings denied, plus CLI checks with
  `frame-trace,jit-profile`; release builds of both configurations.
- Formatting and whitespace checks.

## Measurement protocol

Frozen release binaries run the same first save slot in the supplied gameplay
room. B is pressed for title, slot and play selection; screenshots verify the
room before sampling. Each run uses a disposable copy of the save. Startup and
menu time are excluded from FPS measurements. After settling for 20 seconds,
four runs per binary collect 60 seconds of window-title FPS samples, perf
counters and per-thread scheduler CPU time. Three before runs precede three after
runs; an additional before/after pair checks run-to-run variation. Both sides use
debug logging and the
same perf settings. Builds, tests and expensive profile analysis are excluded
from counted capture windows.

CPU milliseconds per estimated frame use mean sampled FPS to estimate frame
count. Total process CPU includes concurrent threads and must not be read as
frame wall time. Per-thread CPU includes the thread's runtime work; it is not a
measurement of guest instruction lowering alone. HCQ promotion/replacement log
counts describe the whole process, including startup, separately from gameplay.

Separate diagnostic runs collect 20 seconds of perf/JIT publications and an
8-second frame trace with the same startup/navigation/settling protocol. An
additional initial baseline diagnostic had a longer gated settling period; it
is retained as evidence but excluded from the matched diagnostic comparison.

Artifacts and frozen binaries are under `/tmp/nixe-hcq-20261008/`; raw runs are
under `/tmp/nixe-a136/hcq-{before,after}-r{1,2,3,4}/` and the corresponding diagnostic
labels. Local helpers in `dev-tools/performance/hcq/` reproduce the captures and
summarize gameplay separately from whole-process compilation activity.

## Default-release gameplay results

| Run | Before FPS | After FPS | Before main-vCPU CPU ms/frame | After main-vCPU CPU ms/frame |
| --- | ---: | ---: | ---: | ---: |
| 1 | 27.87 | 29.95 | 26.95 | 22.90 |
| 2 | 28.61 | 29.10 | 25.58 | 23.62 |
| 3 | 28.74 | 28.57 | 25.47 | 24.82 |
| Additional before/after pair | 28.96 | 29.20 | 25.53 | 22.94 |
| Mean | 28.55 | 29.20 | 25.88 | 23.57 |

| Mean gameplay metric | Before | After | Change |
| --- | ---: | ---: | ---: |
| Sampled FPS | 28.55 | 29.20 | +2.3% |
| Main-vCPU CPU ms/estimated frame | 25.88 | 23.57 | -8.9% |
| Frontend-thread CPU ms/estimated frame | 19.77 | 19.72 | -0.3% |
| All HCQ workers CPU ms/estimated frame | 0.290 | 0.255 | -12.0% |
| Total process CPU ms/estimated frame | 68.90 | 65.58 | -4.8% |
| Total host cycles/estimated frame | 174.56 million | 164.40 million | -5.8% |
| Total host instructions/estimated frame | 161.07 million | 156.00 million | -3.1% |

The observed FPS increase is modest. Individual run means overlap; the added
before/after pair improves FPS by only 0.8%, while reducing main-vCPU CPU per
estimated frame by 10.2%. These results support reduced CPU work, especially in
the main vCPU, rather than a large or guaranteed FPS increase. They describe one
stationary gameplay room, not a formal confidence interval or other game areas.
The four changes were measured together; there is no isolated FPS attribution
to each individual change.

## Compilation activity, including startup

These are mean whole-process counts through navigation, gameplay capture and
shutdown. They must not be described as gameplay-only events or events per frame.

| Metric | Before | After | Change |
| --- | ---: | ---: | ---: |
| Promotions | 1,130 | 1,095.5 | -3.1% |
| Promotions with at most eight guest instructions | 266 | 179.25 | -32.6% |
| Replacements | 591 | 431 | -27.1% |
| Cumulative published HCQ native bytes | 14.60 MB | 8.92 MB | -38.9% |

The median promoted region increases from 17 instructions in every baseline run
to 19–20 afterwards. Small internal loops and persistently hot leaves can still
qualify; eliminating every small promotion would discard useful optimizations.
Published-byte totals include superseded versions and are not resident cache or
peak memory usage. The decrease combines fewer publications with smaller emitted
adapters/callback paths.

During gameplay, all HCQ workers together save about 0.035 CPU ms per estimated
frame. This is much smaller than the main vCPU's 2.31 ms reduction. High promotion
log volume therefore did not establish background compilation as the dominant
steady-gameplay bottleneck.

## Matched diagnostic trace

Both profiles report zero dropped JIT events and no I/O failure; neither frame
trace drops events. The eight-second trace windows contain 222 submissions before
and 230 after. CPU measurements below are scoped to the traced execution/frontend
intervals and differ from whole-thread scheduler CPU above.

| Trace metric per frontend submission | Before | After | Change |
| --- | ---: | ---: | ---: |
| Main execution CPU ms | 25.62 | 23.75 | -7.3% |
| Frontend CPU ms | 19.15 | 19.24 | +0.5% |
| Main guest instructions | 2,056,694 | 2,048,813 | -0.4% |
| Main safepoint exits | 116.76 | 109.32 | -6.4% |

Mandatory safepoint behavior was preserved. The changed exit count reflects the
observed concurrent workload/timing; it is not a new cap or frame-based stopping
policy. The similar guest instruction counts support comparing the same scene.

The timestamped JIT sidecar resolves 100% of sampled JIT candidates in both
matched profiles; neither perf text export rejects sample lines. Percentages
below are shares of the main vCPU's sampled user cycles, not instruction counts
or independently measured milliseconds.

| Main-vCPU sampled category | Before | After |
| --- | ---: | ---: |
| Generated links, entry/exit adapters and scaffolding | 50.93% | 49.42% |
| Guest instruction lowering | 28.04% | 31.41% |
| Rust JIT sampling runtime | 2.95% | 2.73% |
| LCQ-generated code, all categories | 62.90% | 65.63% |
| HCQ-generated code, all categories | 16.08% | 15.20% |

HCQ generated scaffolding specifically falls from 3.90% to 1.02% of main-vCPU
samples. Shares are renormalized after the change, so an increased share does not
by itself mean greater absolute cost. Unresolved host samples account for 3.74%
before and 2.11% after; they are not silently assigned to JIT categories.

## Remaining limit

This work reduces replacement churn, cumulative generated code and main-vCPU
execution cost. It does not bring the game close to 60 FPS. Approximately half
of main-vCPU sampled cycles still belong to generated connections/adapters and
scaffolding, and the main vCPU still consumes about 23.6 CPU ms per estimated
frame. The graphics frontend consumes about 19.7 ms and was unchanged by this
JIT work. These costs overlap, so they cannot be added as frame wall time, but
each is already above a 16.67 ms frame budget. Removing the remaining background
compiler work alone cannot bridge that gap.

The next substantial performance work must reduce the cost of executing and
connecting hot guest code further and reduce sustained graphics frontend work.
The measured results do not justify claiming that admission thresholds alone,
or fewer promotion log lines, solve the 60 FPS target.

Final default-release SHA-256:

- Before: `8b812d9df442f8aed94bae08750e5b70aa13ad36909b3432615efc1f16c9be41`.
- After: `77fcb1e676912405df86c5702b3309a4e2809cacaf1a16a11bc5966df556b5a0`.

The default `target/release/nixe-cli` was rebuilt after diagnostics and its hash
matches the frozen measured after binary. No ROM or save data is added to the
repository.


## Cross-region contracts follow-up

This follow-up starts from the previous final binary, not from the original
performance-audit baseline. It implements the two previously outstanding items:
coordinate physical contracts and preserve values across separately compiled
regions. It applies to LCQ as well as HCQ because most sampled native execution
still belonged to LCQ.

### Implementation and audit

- Compilation copies a preferred incoming contract from existing target-keyed
  static/PIC adjacency. Executed indirect edges and optimized predecessors take
  priority. In-region predecessors remain ordinary SSA edges. The copied plan
  contains guest identities and physical locations, never executable pointers,
  strong code owners, a new registry or execution-time lookup work.
- A missing indirect destination has no PIC backlink yet. Its actual source map
  is copied while the invocation protects it and passed to demand compilation
  after the invocation and mapping lease have been released. The hint can be
  discarded on cancellation without affecting correctness.
- Both tiers constrain backend `nixe_entry` definitions. Aliased source values
  cannot constrain independent entry definitions to the same register; the
  allocator and ordinary copy scheduler handle the remaining aliases. Register
  and SIMD inputs and already packed NZCV are supported. Arbitrary deferred flag
  expressions remain explicit recipes requiring their normal reconciliation.
- HCQ native liveness incorporates inherited carried state across internal
  branches, public joins, loops, faults and polls. LCQ similarly preserves its
  carried inputs at precise observations. Unchanged bypass values can stay in
  registers until later consumers instead of being committed and reloaded.
  Values overwritten within the region are not forced into the input contract
  merely to preserve their old version: actual semantic reads still determine
  inputs, while required PRE observations can use the committed canonical home.
  This avoids extending unused old versions through the allocator and fault maps.
- Entry contracts also describe values overwritten on **every** path before any
  read or precise observation. Observation-aware architectural liveness proves
  this set separately from physical allocation. Such values are excluded from
  carried inputs and bridge writeback. Faulting writes, PRE helper exits, partial
  writes and paths reaching a cycle poll retain the old value when required.
  Source polls still recover source state before the target can discard it.
- The first experiment exposed allocation-induced ingress shuffles and spills.
  Both tiers now use backtracking register allocation. LCQ still performs no IR
  optimization and captures one straight-line block; HCQ retains region/SSA and
  IR optimization. This removes the production single-pass allocation policy;
  it does not introduce another execution tier or a runtime allocator switch.
- Packed leaves of lazy-NZCV metadata transfer as values. An already matching
  packed contract emits no mask/materialization adapter. Merging missing flag
  bits, preserving non-carried state and genuinely different bindings retain
  their required adapters. Fully compatible links still use the existing direct
  branch / owned PIC target without allocating bridge code.
- Final allocated maps remain the authority for publication and transfer.
  Hints do not become lifetime dependencies. Replacement, code invalidation,
  retirement, thread migration, cache capacity and mandatory control polling
  retain their existing synchronization and ownership rules. No periodic
  register reconciliation or recompilation loop was added. The warmup compiler
  policy identity advances to version 4.

Validation: the final `cargo test -p nixe-cpu-jit -p nixe-runtime --release`
run passed 1,157 tests, with two existing external-fixture tests ignored. The
1,054-test JIT unit suite passed with debug verification before the last
selective-carry refinement; its affected paths were then rechecked with 19
focused debug tests (including the two new pruning regressions). Thirteen new
regressions in total cover alias constraints, both host encoders, carried values
through unrelated regions, empty matching bridges, real first-indirect-demand
execution, canonical entry and budget exits, precise fault recovery, all-path
overwrite proofs and packed flag preservation. Workspace/all-target Clippy with
`frame-trace,jit-profile` and warnings denied, formatting and whitespace checks
passed. The last refinement was followed by the complete release suite again.

### Follow-up measurements

Frozen binaries, logs and comparisons are under
`/tmp/nixe-contracts-20261008/`; raw captures use `/tmp/nixe-a136/contracts-*`.
The baseline is the exact binary at the start of this follow-up. Each normal
series consists of three 60-second windows in the same saved gameplay room,
after navigation and 20 seconds of settling. Startup/menu time is excluded.
Screenshots verify the room, and saves are disposable copies. Builds, tests and
trace analysis do not run during the measured windows. FPS comes from the
existing guest-frame window-title counter; CPU/frame uses scheduler CPU time
divided by estimated frames. Different threads overlap, so their CPU times are
not additive frame latency. Separate diagnostic runs use timestamped JIT maps,
20 seconds of perf sampling and an 8-second trace.

The broad-carry/backtracking version (before the final pruning) measured
30.234 FPS versus 29.678 baseline (+1.87%), and 22.494 versus 23.200 ms of main
vCPU CPU/frame (-3.04%). Its FPS runs were 30.388, 30.888 and 29.425, overlapping
the baseline's 29.143, 30.073 and 29.818. This did not demonstrate a robust large
performance gain. Whole-process HCQ emitted bytes, including startup, grew
from about 8.85 MB to 10.32 MB. In its paired diagnostic, generated entry/exit
share fell from 14.00% to 11.54% of main-vCPU samples, but generated scaffolding
rose from 11.47% to 15.37%. This prompted the additional selective-carry
iteration described above rather than treating the initial result as success.

Final selective-carry results:

| Gameplay metric, mean of three captures | Before | Final | Change |
| --- | ---: | ---: | ---: |
| FPS | 29.678 | 30.331 | +2.20% |
| Main-vCPU CPU ms / estimated frame | 23.200 | 22.339 | -3.71% |
| Graphics-frontend CPU ms / estimated frame | 19.637 | 19.068 | -2.90% |
| Whole-process CPU ms / estimated frame | 64.741 | 62.697 | -3.16% |
| Host retired instructions / estimated frame | 156.872 M | 156.335 M | -0.34% |
| Host cycles / estimated frame | 161.911 M | 157.373 M | -2.80% |
| HCQ-worker CPU ms / estimated frame, all workers | 0.314 | 0.270 | -14.10% |

Final FPS runs were 30.285, 30.598 and 30.108; main-vCPU CPU/frame was 22.524,
22.139 and 22.354 ms. The additional pruning iteration changes the broad-carry
mean from 30.234 to 30.331 FPS (+0.32%), and main-vCPU CPU/frame from 22.494 to
22.339 ms (-0.69%). These small differences are within the scale of run-to-run
variation; they are not evidence of a large or generally assured speedup.
The graphics frontend is unchanged, so its movement must not be presented as a
separate graphics optimization delivered by this work. HCQ-worker savings are
only about 0.044 CPU ms/frame and cannot explain a path to 60 FPS.

The tradeoff remains visible: whole-process HCQ emitted bytes including startup
average 10.078 MB versus 8.851 MB before (+13.87%). The selective-carry version
reduces this from the broad-carry version's approximately 10.321 MB, but carrying
additional live values is not free. This measure is cumulative emitted code,
not the resident hot instruction working set or gameplay-only compilation.

The requested coordination and preservation now exist in the production path,
with compatible-edge transfer elimination covered by native tests. This is not
a globally co-allocated register convention: compilation chooses an existing
predecessor's contract; other predecessors, register pressure and subsequent
replacement can still require adapters. No policy recompiles stable regions
merely to chase matching contracts. The measured gain is modest and does not
establish this mechanism alone as the solution to the 60 FPS target.

Final normal-release SHA-256:

- Before: `77fcb1e676912405df86c5702b3309a4e2809cacaf1a16a11bc5966df556b5a0`.
- Final: `da33f56efc029caa5ab71f0aaf3106f08f5767dc95a25cf229a64110bd13a2a9`.

Diagnostic release SHA-256:

- Before: `82ef7057f852343fbbcf20cdac93936fe7e4cd045c76540579803ee50bd1b912`.
- Final: `6b9fa2b258cc77eb9c2cb1480ee5348e3391c3a6584f9f0fb2a43f076bcaeb2f`.

The separate 8-second diagnostic trace contains 228 frontend submissions before
and 236 after, with no dropped events. Guest work is comparable: 2.056 M versus
2.049 M completed main-vCPU instructions/submission. Main-vCPU CPU time is
23.672 versus 21.544 ms/submission in this diagnostic pair; frontend CPU time is
19.843 versus 20.003 ms/submission. These short, instrumented runs support cost
attribution, not replacement of the three normal FPS captures above. Main-vCPU
safepoint exits change from 99.79 to 82.23/submission without changing the
preemption protocol, so they must not be advertised as a new guaranteed
safepoint-frequency policy.

Timestamped publication mapping resolves 100% of sampled JIT-candidate cycles
in both final diagnostic captures. Shares below are normalized to the main
vCPU, not the whole process:

| Main-vCPU sampled-cycle share | Before | Final |
| --- | ---: | ---: |
| Generated connections / dispatch | 27.43% | 25.97% |
| Generated entry / exit | 14.01% | 13.85% |
| Generated scaffolding | 11.47% | 11.38% |
| Guest instruction lowering | 27.45% | 33.47% |
| LCQ generated code, all categories | 64.93% | 70.64% |
| HCQ generated code, all categories | 15.43% | 14.03% |

Unresolved main-vCPU samples are 2.94% before and 1.50% after. These sampled
shares move with the denominator and workload timing; a larger share does not
alone prove greater absolute cost. The final selective-carry version removes
the broad-carry experiment's increase in scaffolding share, but substantial
boundary work remains. In particular, two hot final static bridges extracted
from the JIT dump are 40 and 232 bytes: the former reloads a spilled value and
commits its canonical home; the latter captures host flags, commits several
spilled/register/vector values, and moves five register inputs before jumping.
Matching one predecessor does not eliminate those obligations for all incoming
edges. Artifacts are in `pruned-native/hot.json` and the corresponding `.bin`
files under the follow-up artifact directory.

This leaves roughly 26% of main-vCPU samples in generated connections/dispatch,
14% in entry/exit, and 11% in scaffolding. Most generated execution still belongs
to LCQ. A larger improvement therefore needs fewer expensive native boundaries
and better allocation across them (including spilled values and incompatible
flag representations), as well as sustained graphics-frontend improvements.
The final normal run still requires 22.34 main-vCPU CPU ms/frame and about 19.07
frontend CPU ms/frame, each above 16.67 ms despite overlapping execution. These
measurements do not support promising 60 FPS from register-contract matching
or background compilation tuning alone.

The normal `target/release/nixe-cli` was restored through Cargo after the trace
build. Its SHA-256 matches the frozen measured final binary. All benchmark game
processes have exited; no original save was modified.

