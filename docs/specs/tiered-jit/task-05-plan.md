# Task 5 implementation plan

Status: Task 5 complete (steps 1–7), within the agreed production activation boundary,
the vCPU-owned sampling tables and nonblocking seed/reshape admission into a
bounded queue, protected incremental LCQ input acquisition and fixed worker
owners with private reusable compiler resources.
Process closure drains/wakes the registered pool, partial startup cleanup is
covered, and original worker failures reach canonical vCPU error reporting.
JitProcess owns and joins its pool before final JIT reclamation. Mapping/version
changes and cache pressure abandon background work without failing the process
or clearing newer reservations. Final validation and the Tasks 6/7 handoff are
recorded in step 7; production activation remains unchanged.
Task 4 supplies production native LCQ chains, static
links, per-vCPU PICs, guest-thread return prediction and coordinated lifetime
management. Resumable native polls, canonical exits, escaped-fault prefixes and
successful cold instruction completions now record LCQ samples. Transfer
observations classify current family boundaries through point lookups. Seed
and reshape admission share the bounded queue but are not activated in production.
Production HCQ worker activation remains pending. Real HCQ region
construction/emission belongs to Task 6 and parallel
membership/reshape publication to Task 7.

This is a working checklist for
[Task 5](spec.md#task-5-add-functional-sampling-and-bounded-background-admission),
not another specification. Follow [CONTRIBUTING.md](../../../CONTRIBUTING.md).
Update each step in place with decisions, remaining work and validation results;
do not append session logs or create a separate decision register. Agree changes
to architectural contracts with the maintainer and update the affected spec
section before implementing dependent work. Keep implementation details in code.

## Scope

Use the existing 4096-instruction cold deadline to identify hot, demand-proven
LCQ seeds and sampled family boundaries. Admit immutable, exactly deduplicated
requests into bounded background storage without blocking guest execution or
adding work to the normal native link path. Implement worker ownership,
selection, cancellation and shutdown using the existing lifetime foundation.

Do not implement HCQ graph formation, optimized native emission, instruction
membership collision trimming or replacement transactions here. Synthetic
families and finite test jobs may exercise the real admission/worker owners;
they are test inputs, not a second production runtime or compiler framework.
Do not add profiling exports, hotness logs, runtime tuners or a benchmark suite.

**Activation boundary (agreed):** Task 5 connects production sampling, but
production admission and worker startup remain disconnected until Task 6
supplies the real HCQ compiler. Implement and test the queue and worker lifecycle
here through their real owners with finite test-only jobs. Do not ship workers
that discard valid requests, pretend to compile them, reject them because the
backend is absent, or continuously requeue them. Reshape admission is connected
with its real consumer in Task 7. This staging is not a new configuration option
or the zero-worker CPU policy; the specified worker-count formula is unchanged.

## Starting points

- `crates/cpu-jit/src/native/poll.rs` emits the cold accounting/control leaf.
  Sample-only deadlines now enter the source-local observer in
  `native/observation.rs` before resuming the source hot continuation.
  `abi.rs::PollBudget` already reconciles overshoot and forced transitions.
- `engine.rs::JitThread` owns the persistent sample phase and vCPU compiler/
  reader. Confirm its runtime ownership in `runtime/src/process/execution.rs`
  before placing the two per-vCPU tables; they must not follow the migrating
  guest-thread RSB or be shared with another execution worker.
- `lifetime/unit.rs::StateRecord` and `TerminalTransfer` retain the actual
  source version, guest exit, dynamic destination location, completed cost,
  poll patch and continuation. `lcq/invocation.rs` protects this metadata for
  the complete native chain and resolves later-unit exits precisely.
- `lifetime/compile.rs` implements synchronous LCQ claims, including a waiting
  path. Background admission must not reuse that waiting path. Dispatch slots,
  generational registries, family owners and compiler references already exist;
  extend their ownership rather than creating a second deduplication registry.
- `JitProcess::{new,request_stop,try_shutdown}`, lifetime maintenance and the
  runtime teardown path are the integration points for worker startup/drain.
  Preserve Task 4's GPU-before-JIT shutdown ordering.
- Continue using `/home/pladaria/projects/wasmtime`, branch `nixe`, and
  `--offline --config /tmp/nixe-observable-fp-local.toml`. Inspect the current
  checkout and pending changes before modifying the fork. Publishing/pinning
  it is a separate maintainer action; do not commit local paths or an
  override-resolved `Cargo.lock` as the portable dependency handoff.

## Step 1 findings and implementation decisions

The audit changes no runtime behavior. The maintainer confirmed the activation
boundary above: production compilation starts with the real consumer in
Task 6, not with a placeholder in Task 5. The affected spec sections now reflect
this boundary and the nonblocking sample-identity rules below.

### Ownership and identities

- `runtime/src/process/execution.rs::create_worker_cpu_thread` creates one
  `JitThread` per process/vCPU; `coordinator/worker.rs` retains it on the worker.
  Put both tables and their sample sequence alongside `sample_remaining` in
  that owner. Guest migration carries architectural state and the RSB, not heat.
- `JitProcess` owns background startup/join handles. Worker closures retain the
  shared background state, memory and Lifetime, not their owning JitProcess;
  avoid an Arc cycle between a process and the threads it must join. Admission
  closes before drain/join, which holds none of the JIT/queue/memory locks.
- A sample names the actual source `UnitHandle`/`CodeUnitId`/`CodeVersion`,
  source instruction and logical block key. For LCQ the seed is the source
  unit's root BlockKey, not its terminal PC or the initial gateway entry.
  Task 6 must retain the logical source block explicitly for multi-block HCQ.
  ReachabilityVersion is a separate dispatch identity; never substitute
  CodeVersion for it or combine observations from an old body with a new slot.
- A real transfer carries its allocated target PC and edge kind. Form a full
  target BlockKey only when valid for the source execution context. Invalid or
  undemanded targets do not create dispatch slots or trigger instruction fetch.
  Non-edge observations have no successor. Calls may heat a seed but do not
  merge a callee into the region; entry/graph selection remains Task 6.
- Classify uncovered LCQ observations as seeds; classify the spec's actual
  HCQ/retained-LCQ ownership boundaries by both InstructionKeys and current
  family versions. Do not scan all families or units to classify a sample.
  Extend cold point-lookup ownership metadata where needed, without doing
  Task 7's candidate membership reservation or replacement transaction here.

### Consume each deadline once

| Current path | Sampling hook and attribution |
| --- | --- |
| Resumable terminal poll | The source-local cold adapter names its map and captures the allocated destination before a host call; reconcile once and resume the already-charged hot patch. |
| Terminal slice exit | Use the same source/target observation, including when sample and slice expire together; do not resample at gateway return. |
| Forced control exit | Charge work and preserve phase but suppress the crossed heat sample. A later poll handles requests arriving after the bounded request check. |
| Canonical PRE/semantic exit | Consume the gateway's `NativeReturn.poll.sample` while the actual source is protected. Attribute prefix work to that LCQ seed, without inventing an outgoing edge. |
| Escaped fault | Consume the result currently discarded by the prefix `reconcile` in `lcq/invocation.rs`, using the published fault source; repair/retry is not a new observation. |
| Successful cold completion | Consume the result currently discarded in `engine/completion.rs` after charging its one instruction. Carry the source key/version identity across the released invocation and revalidate it; failure charges no instruction. |

Reconciliation owns phase advancement even when an observation cannot admit
work. Never replay a missed sample on the next invocation, reconstruct elapsed
heat from an already-rearmed budget, or manufacture one sample per overshot
interval. Normal canonical lookup/semantic work may already require locks;
sampling must not add a blocking acquisition to those paths.

### Resumable observation veneer

Extend source-local cold emission, not the ordinary transfer. Retain only live
caller-clobbered physical values, NZCV/recipe operands and the dynamic target in
ABI-owned scratch; backend spills remain untouched. The metadata supplies the
source without a per-block current-unit store. Do not call `dispatch_link`,
resolve the successor or canonicalize/reload the entire guest merely to sample.
The same invocation epoch and memory lease protect the callback and resume.

`NativeFrame::suspend_fp` is not a drop-in implementation: it merges host status
into canonical FPSR, while `resume_fp` clears native sticky flags. Resuming an
unchanged SSA FPSR could subsequently overwrite that merged contribution.
Add the non-observing pause/resume operation to the existing `fp_env` owner:
save the exact active host control/status, run Rust under the saved caller
environment, and restore the exact guest image on successful continuation.
Do not materialize guest FPSR or modify mapped software FPSR for sampling.
On callback failure, restore saved physical values, canonicalize the source
and merge the saved guest status once while retaining the caller environment;
never unwind through native code or resume guest work after an internal error.
An inactive guest FP segment must remain inactive.

### Nonblocking reservation protocol

`Lifetime::reserve`, `Reader::claim`, `FaultLookup::static_entry` and the current
slot snapshot helpers are not sampling-admission APIs: they lock state, and
reserve may grow storage. Add a cold, existing-key-only `try_lock` lookup which
validates Open/admission epoch, exact source/version and current ownership.
On contention, defer to a later real sample; never add heat under an unverified
identity. Once a verified threshold attempts admission, deferral retains seven
or three as specified. Phase advancement is never undone to retry an observation.

Registry slots reside in a movable Vec. Keep the atomic HCQ reservation cell
at a stable address owned by its dispatch slot (and analogous family owner),
allocated during cold owner creation, not sampling. A job retains that cell
plus generational handles, not a pointer into the registry. It is the owner's
compile state, not a second key-indexed deduplication table. Preserve LCQ demand
claims separately: promotion must not block synchronous baseline demand.

1. Under the short state try-lock, validate identities and retain the existing
   reservation owner. Reserve with an exact checked token, then release state.
2. For reshape, reserve each participating family generation with the same job
   token in stable family-ID order; failure rolls back only this token's claims.
   There are at most two families. Preserve one reshape per family generation.
3. Try-lock the fixed queue. Full/closed/contended admission rolls back by
   exact-token CAS without reacquiring state, allocating or waiting.
4. With a free cell, write the immutable snapshot, release-CAS the reservation
   to Queued and insert before releasing the queue lock. A lost token cannot
   enqueue; a worker cannot dequeue before publication is complete.
5. Workers validate the captured epoch and owner generations before accepting
   the job. Closure in the reservation/enqueue gap can make a job stale, never
   current under a new epoch. Queue closure and draining use that same queue
   lock so no insertion follows the shutdown drain.

Token generations never wrap; exhaustion is an implementation failure, unlike
the local sample-sequence reset. Cancellation/drop may clear only its exact
token. Extend retirement/shutdown accounting so retained reservation owners
pin their actual slots until cleanup finishes, including reserved-but-not-yet-
queued jobs. Ordinary Closed/reopen cycles must not leave a stale token
permanently suppressing promotion of a still-valid LCQ version.

The worker may acquire normal short registry locks after dequeue to clone
strong immutable CodeUnit references. It holds no such lock during discovery,
compilation or waits. No guest-side admission path fetches/decodes instructions
or invokes the code allocator. The seven-newest/one-oldest selection counter
belongs to the one process queue and advances on removal, including stale jobs.

**Validation of this step:** read-through of the runtime vCPU owner, native
poll/exit emission, gateway and fault reconciliation, cold completion, FP owner,
dispatch/compile registries and retirement paths. No runtime/fork changes and
no execution-test claim; implementation and regression tests start in step 2.

## Steps

- [x] **1. Settle sample attribution, admission identities and ownership.**
  Audit complete; see the findings above. Production activation agreed and the
  sampling/admission and Task 5–7 spec sections reconciled. No runtime changes.

  Trace native sample-only, slice, control, semantic and fault exits through
  reconciliation. Identify where each crossed deadline is consumed exactly
  once, before its source metadata loses protection. Specify which observations
  are seed samples, eligible family boundaries or non-edge observations; never
  invent a successor for a fault/semantic exit or infer the source from the
  original gateway entry. Preserve full platform/address-space/FP keys.

  Map the tables to their exclusive vCPU owner and the queue/workers to their
  process owner. Define exact seed and reshape reservation tokens, their slot/
  family pins and linearization points. Resolve admission races with Closing,
  queue insertion, invalidation and slot reuse. Check the whole guest-side path
  for allocation or blocking locks, not just the final queue `try_lock`.
  Settle the Task 6 consumer/activation boundary described above before adding
  production background work; record the agreed decision here.

  **Exit:** each sample, reservation and worker reference has one owner; the
  nonblocking protocol and staged activation are explicit. Required changes to
  the spec are agreed before dependent implementation, without another design
  document.

- [x] **2. Implement the bounded per-vCPU tables and immutable snapshots.**
  Implement the specified 256-set/four-way HotSeedTable and 64-set/two-way
  BoundaryTable, including full-key equality, deterministic replacement,
  saturating scores and checked sample sequences. Seed records retain four
  observed successor slots with their own counts and recency. Copy only the
  required immutable observations into AdmissionSnapshot/ReshapeSnapshot.

  Reset mismatched versions without combining heat from different identities.
  Keep table mutation on the owning vCPU, including invalidation handling;
  another thread must not clear live tables concurrently. At sequence overflow,
  clear both bounded tables and restart at one; identity counters retain their
  existing checked failure rules. With zero workers, scores may saturate but
  cannot admit work and no pending queue exists.

  **Exit:** focused tests cover collisions, all tie-breakers, saturation,
  successor replacement, version reset and sequence overflow. Snapshots remain
  unchanged after subsequent samples. Table storage is allocated outside guest
  sampling; there is no per-sample allocation or unbounded history.

  **Implemented:** `sampling.rs` owns fixed boxed 256×4 seed and 64×2 boundary
  storage, allocated by `JitThread::new`. Observations and snapshots are copied
  values; no worker borrows table storage. Full execution keys select records;
  changed reachability/family identities reset heat in place. Boundary records
  include both endpoint reachability versions and optional owners on both sides,
  matching the spec's zero/one/two-family reshape contract. Non-edge samples
  retain prior successor observations but clear the last edge; an unaligned
  destination is recorded without becoming a successor. Deferral cools only
  the exact still-current snapshot to seven/three. Invalidation and sequence
  reset mutate only the owning vCPU's storage. No production poll/admission
  hooks yet; disabled admission can saturate heat without returning requests.
  Negative reshape-result caching remains part of the real Task 7 consumer.

  **Validation:** 12 focused tests cover thresholds, disabled admission,
  collisions/tie-breakers, successor saturation/replacement, full keys, version
  resets, invalidation, immutable snapshots, stale deferral and overflow of
  both tables. All 551 x86-64 JIT library tests passed; the 12 sampling tests
  also passed on AArch64 under QEMU. `cargo fmt --all -- --check` and package
  Clippy (`--all-targets --no-deps -- -D warnings`) passed. Clippy with workspace
  dependencies reports existing `type_complexity` warnings in
  `crates/memory/src/range.rs`, unrelated to this step. Builds use the local
  Wasmtime override; no fork changes were needed. The missing temporary
  `/tmp/nixe-observable-fp-local.toml` was recreated for these checks.

- [x] **3. Connect functional sampling to native and canonical cold paths.**
  Replace the sample-only rearm placeholder with the actual sampling path.
  Use the protected source map and allocated destination from Task 4. Preserve
  live registers, lazy flags, FP ownership and exclusive state across a
  successful resumable cold sample. General Rust work must run with the caller
  FP environment, restoring the guest segment only before native continuation.

  Consume the same sample phase across cold polls, canonical exits and partial
  fault prefixes. Preserve overshoot, emit at most one sample per crossed
  boundary and suppress heat during forced control transitions. Sampling must
  neither double-charge a branch nor execute/charge a pending semantic operation.
  Resume the already-completed source continuation without repeating its RSB
  update, budget subtraction or guest effects. Keep table/ownership lookups off
  ordinary static, PIC and return hits, including CIVAC's native fast path.

  **Exit:** real linked-loop and later-unit tests attribute samples to the
  correct source/target and preserve architectural state and slice budgets.
  Both host encoders retain the existing hot sub/test/branch shape with no
  hotness load/store/RMW or promotion branch. No sample is logged or exported.

  **Checkpoint — canonical exits:** the production invocation receives the
  owning vCPU's tables and consumes `NativeReturn.poll.sample` after caller-FP
  restoration, before releasing the source epoch. Escaped faults likewise
  consume the prefix reconciliation result exactly once. The native-PC record
  supplies the actual later unit; terminal edges retain the actual destination,
  while PRE/semantic/fault observations have no invented successor. A short
  existing-owner `try_lock` validates the generational unit, current LCQ body,
  reachability and uncovered root. Contention, closure or a stale/replaced body
  drops the observation; poison remains an error. No slot allocation, queue
  operation or normal native-link change was added.

  **Checkpoint validation:** all 559 x86-64 JIT library tests passed; all 20
  sampling tests also passed under AArch64/QEMU. Added production tests cover
  later-unit conditional edges, PRE exits, fault escape/retry, forced control
  and FP sticky status/caller restoration. Owner tests cover contention,
  Closing, replaced/HCQ-covered baselines and poisoned state. Formatting,
  diff whitespace checks and package Clippy with `--no-deps -D warnings` pass.

  **Checkpoint — cold completion:** when the next instruction can cross the
  sample deadline, the owned exit captures a value-only generational unit/root/
  reachability identity before epoch release. Successful semantic or memory
  completion charges one instruction, consumes its deadline and revalidates
  that identity with the same nonblocking lookup. The token pins no code and
  cannot heat a replacement or reused slot. Failure/trap records no completion
  sample; prefix samples are not repeated. Missing/contended/stale identity
  discards heat without undoing the sample phase or instruction charge. No
  source lookup is added when the next instruction cannot cross the deadline.

  **Cold-completion validation:** all 565 x86-64 JIT library tests passed.
  AArch64/QEMU passed all 26 sampling tests and all 13 completion tests.
  Coverage includes a later linked source, prefix/completion deadline
  separation, exhausted slices, successful/failed MMIO, FP traps, source
  replacement and invalidation by the completing cache operation itself.
  Registry tests cover contention, reachability mismatch, actual reclamation
  and a stale token after slot reuse. Formatting, whitespace checks and package
  Clippy (`--all-targets --no-deps -- -D warnings`) passed.

  **Checkpoint — non-observing FP pause:** the existing FP owner can now pause
  an observation without materializing canonical FPSR. It restores the caller
  environment for the observer and resumes the exact guest control/status image,
  including accumulated sticky flags. An inactive segment stays inactive. On
  observer failure, it keeps the caller environment and returns the hardware
  contribution for merging only after source canonical writeback. The pause
  borrows the existing owner, allocates nothing and cannot cross OS threads.
  This primitive is now used by the native poll callback.

  **FP-pause validation:** all 570 x86-64 JIT library tests and all nine FP-owner
  tests on AArch64/QEMU passed. New cases cover all 16 supported native FP modes,
  repeated observations with software FPSR, inactive/suspended segments and
  observer failure without lost or duplicated status. Formatting, whitespace
  checks and package Clippy passed.

  **Checkpoint — physical observation preservation:** source-local emission now
  saves/restores only mapped caller-clobbered registers in the existing transfer
  partition, including clean values, packed/deferred NZCV operands and an
  otherwise unbound dynamic destination. Aliases save once at their largest
  live width. Constants, backend spills and System-ABI nonvolatiles need no
  register save; AArch64 full vectors in v8–v15 do, despite their preserved low
  halves. The destination is captured before the callback. Host-resident NZCV
  can also be preserved, provided the save precedes flag-clobbering arithmetic;
  current LCQ terminal polls instead use packed/deferred flags.

  **Preservation validation:** three focused tests pass on x86-64 and
  AArch64/QEMU. Executable cases repeat a real System-ABI callback which wipes
  volatile GPR/SIMD registers (including v8–v15 upper halves), composed with
  the FP pause/resume operation, without canonical reloads. They compare full
  architectural results against uninterrupted execution for packed, host and
  deferred flags; register/vector/spill/constant destinations; and active or
  inactive FP. Shape tests cover alias deduplication, clean and unbound recipe
  operands, scalar d8 preservation and invalid destination rejection.
  All 576 x86-64 JIT library tests, all 72 engine tests on AArch64/QEMU and the
  observer-failure test on both hosts passed. Linked-loop tests cover actual
  later-source attribution, repeated sample-only resumes, overshoot, coincident
  sample/slice deadlines and FP sticky/software status. The failure test rejects
  a wrong source version, canonicalizes exactly one completed iteration and
  retains its hardware status until after software writeback. Formatting,
  whitespace checks and package Clippy (`--all-targets --no-deps -- -D warnings`)
  passed.

  **Production poll:** the LCQ compiler emits this callback path only on the cold
  sample-only branch; ordinary static/PIC/return hits keep their existing shape.
  The emitted callback arguments identify the actual source native PC, version
  and map without a hot current-unit store. The invocation's vCPU-owned tables,
  epoch and memory lease remain in scope through observation and resumption.
  The callback uses the existing nonblocking identity lookup, performs no
  successor resolution and never reconciles the already-consumed deadline.

  On observer error or caught panic, the adapter restores physical values and
  takes the source canonical control exit instead of resuming guest execution.
  The invocation merges the saved hardware status only after software FPSR
  writeback, reports the error and clears its borrowed callback pointer on all
  returned paths. Slice/control exits bypass the observer; the existing
  canonical sampling hook alone handles a simultaneous sample/slice deadline.

  **Family ownership lookup:** the lifetime owner now indexes every published
  HCQ InstructionKey, including interior instructions without an HCQ dispatch
  entry, by a weak generational family handle. Family publication installs this
  index under the same state lock as dispatch; unlink removes the exact owner
  before reclamation, even when old code remains pinned. Storage growth is
  prepared and charged outside the state lock, capacity is revalidated at
  publication, and empty entries are reusable. The existing HCQ overlap check
  now performs point lookups for candidate instructions instead of scanning all
  families. This is current ownership only, not Task 7's in-flight reservations.

  **Ownership validation:** four new tests cover interior instructions/full
  semantic keys, invisible staged/discarded/stale outputs, unlink with pinned
  old code, stale-owner removal, index growth and capacity reuse. All 580
  x86-64 JIT library tests and all 135 unit/lifetime tests on AArch64/QEMU passed,
  as did formatting, whitespace checks and package Clippy.

  **Boundary classification:** terminal samples now use the actual source
  instruction, logical block's current dispatch reachability and both endpoint
  family identities. Existing demanded targets supply their own reachability;
  instruction ownership alone never creates a dispatch identity. Uncovered
  LCQ observations heat seeds; ownership boundaries heat BoundaryTable instead.
  Same-family HCQ-to-HCQ public-entry edges are ignored, while retained LCQ
  entries into owned interior instructions remain boundaries. Non-edge and
  completion observations cannot heat HCQ-owned interior roots as new seeds.
  Classification uses only existing point lookups under one state try-lock,
  without allocation, guest reads or unit/family scans.

  **Final step validation:** eight new classification tests cover LCQ/HCQ
  boundaries, actual terminal versus root attribution, interior/non-public
  entries, missing dispatch identities, contention/closure, pinned retired
  sources and family replacement with score reset. All 588 x86-64 JIT library
  tests and all 36 sampling tests on AArch64/QEMU passed. Formatting,
  whitespace checks and package Clippy (`--all-targets --no-deps -- -D warnings`)
  passed. These HCQ classification fixtures publish real
  lifetime owners with synthetic code; they do not execute an HCQ compiler.
  Task 6 must supply the explicit logical source block for multi-block HCQ
  observations. Production admission/compilation still waits for Task 6 and
  reshape admission for Task 7. No Wasmtime changes in this checkpoint.

- [x] **4. Implement exact nonblocking admission and the preallocated queue.**
  Reserve the exact seed BlockKey/ReachabilityVersion before trying the queue.
  Implement AdmissionReserved(token) → HcqQueued(token), complete snapshot
  publication before worker visibility, and exact-token rollback on contention,
  fullness or invalidation. Queue failure leaves seed score seven or boundary
  score three so only a subsequent real sample retries. Existing queued/running/
  rejected state prevents duplicate work for the same identity.

  Use one preallocated VecDeque with capacity `8 * worker_count`. Implement
  seven newest dequeues followed by one oldest as one queue-wide selection
  cycle, not independent per-worker counters. Apply equivalent exact admission
  ownership to reshape snapshots keyed by the named instruction/family versions;
  do not implement Task 7's region replacement here. Avoid holding the queue
  lock together with JIT-state, code-cache or memory locks.

  **Exit:** seven seed samples do not enqueue; the eighth can enqueue exactly
  one matching request. Four boundary samples can enqueue one matching reshape.
  Contention/fullness returns promptly without allocation, polling or waiting.
  Controlled races prove that stale rollback cannot clear a newer token and a
  worker cannot observe a partially initialized or not-yet-queued request.

  **Checkpoint — seed reservation and queue:** each dispatch slot now owns a
  stable, charged atomic reservation cell allocated during cold slot creation.
  Existing-key admission revalidates the exact reachability and uncovered LCQ
  root under a state try-lock, reserves a checked process-wide token and releases
  state before trying the queue. Queued/running/rejected states deduplicate that
  identity without interfering with synchronous LCQ claims. Exact-token cleanup
  cannot clear a newer reservation; publication cancels old state, while a new
  admission after maintenance can replace an obsolete epoch's in-flight token.
  Rejection survives ordinary reopen only for the same reachability.

  The process-bound queue preallocates eight cells per selected worker and uses
  one seven-newest/one-oldest removal counter. It copies immutable snapshots and
  retains generational slot/unit identities, not executable addresses or mutable
  sample tables. Fullness/contention rolls back without registry reacquisition
  and cools the exact seed snapshot to seven. Closure and insertion serialize on
  the queue mutex; draining hands off jobs for cleanup outside that mutex.
  Zero workers allocate no queue. Reservation references pin actual dispatch
  storage through cancellation, retirement and shutdown, including the gap
  before enqueue; registry growth cannot move their atomic cell.

  **Checkpoint validation:** 13 new tests cover the eighth-sample threshold,
  immutable snapshots, duplicate/running/rejected reservations, queue and registry
  contention, fixed capacity, shared dequeue fairness, epoch/version replacement,
  stale rollback, slot pinning/reuse, registry growth, concurrent vCPU admission,
  queue closure, shutdown and explicit token exhaustion. All 601 x86-64 JIT
  library tests and the 13 admission tests on AArch64/QEMU passed. Formatting,
  whitespace checks and package Clippy (`--all-targets --no-deps -- -D warnings`)
  passed. No Wasmtime changes were needed.

  **Reshape admission:** the same fixed queue now accepts immutable reshape
  snapshots. Existing-key point lookups revalidate both endpoint reachabilities,
  instruction membership and family generations. Jobs retain the explicit
  logical source block, including when its terminal has no dispatch slot. Zero,
  one or two distinct families are supported; a shared family is reserved once,
  and two families are reserved in family-ID order with one checked job token.
  Busy owners defer with score three. Partial reservation or queued-transition
  failure rolls back only this job's exact tokens.

  A family's reservation cell lives in its unit registry record and survives
  unlink until job cleanup permits actual unit/family slot reclamation. Endpoint
  pins likewise retain actual dispatch slots. Guest-side rollback releases only
  these cells, not Family/CodeUnit objects whose destruction could acquire the
  allocator lock. Zero-family reshapes use a separate source-dispatch claim so
  an ordinary seed rejection neither blocks a different reshape nor gets erased
  by one. This is admission ownership, not candidate membership reservation or
  the negative-result cache owned by Task 7.

  **Final step validation:** 11 additional tests cover zero/one/two families,
  same-family deduplication, stable family ordering, partial reservation and
  partial queued-transition rollback, stale cleanup after replacement, actual
  family/endpoint pins, shared-family vCPU races, queue fullness/contention,
  source-block/endpoint identity validation, independent seed rejection and
  queued/unqueued shutdown. All 612 x86-64 JIT library tests and all 24 admission
  tests on AArch64/QEMU passed. Formatting, whitespace checks and package Clippy
  (`--all-targets --no-deps -- -D warnings`) passed. HCQ family inputs remain
  synthetic published units, not a production HCQ compiler. No Wasmtime changes.

  Worker acceptance/code-reference acquisition remains step 5, and complete
  process cancellation/startup/shutdown wiring remains step 6. Production
  sampling still submits no jobs and starts no workers: seed activation remains
  Task 6 and reshape activation Task 7.

- [x] **5. Implement fixed workers and protected request handoff.**
  Select the logical-CPU count once at process initialization and apply the
  spec formula: zero workers for at most two CPUs, otherwise
  `min(4, max(1, (logical_cpus - 2) / 2))`. Allocate the queue and worker-owned
  resources before enabling admission. Each worker owns its compiler context
  and scratch; no shared compiler mutex, general thread pool or thread per job.

  Workers sleep on an empty queue, claim dequeued tokens exactly, and enter
  compiler read-side protection before resolving registry identities. Supply
  the Task 6 handoff with immutable observations and strong references to named
  demanded LCQ units. Acquire references incrementally by key, not by scanning
  resident units or eagerly copying a process graph. No worker reads per-vCPU
  tables, guest-thread state or unprotected registry pointers.

  **Exit:** finite test jobs exercise the real queue, multiple worker owners,
  newest/oldest fairness and protected source lifetime. Zero-worker initialization
  creates neither threads nor a queue. Production activation follows step 1's
  explicit handoff; validation is not described as real HCQ compilation.

  **Checkpoint — protected worker handoff:** empty-queue consumers now sleep on
  a condition variable and share the existing queue-wide removal cycle. After
  dequeue, acceptance registers compiler protection before resolving captured
  handles and validates the exact admission epoch, owner generations and tokens
  before changing Queued to Running. Stale work releases only its own pins and
  tokens. Seed and reshape observations remain immutable values.

  `Work::lcq` acquires a strong reference to one named, currently demanded LCQ
  input under a short state lock. It checks the execution key and permits only
  uncovered roots or the reshape's participating families; it never creates a
  dispatch slot, fetches guest bytes, returns an HCQ body or copies a graph.
  Candidate construction, interior membership checks and publication validation
  remain Tasks 6/7. No execution epoch or lock spans compilation: ordinary
  Closed maintenance can unlink inputs, whose immutable images survive until
  their last reference drops. Terminal teardown also waits for active compiler
  protection. Phase-boundary checks cancel obsolete work.

  **Checkpoint validation:** eight new tests cover incremental same-context
  inputs, participating-family baselines, stale epochs/versions, foreign-process
  jobs, old-token cleanup, unlink with retained inputs, shutdown, scope unwind
  and concurrent sleeping consumers. All 620 x86-64 JIT library tests and all
  32 background/reshape tests on AArch64/QEMU passed. The final empty-queue
  closure ordering was also rerun in all eight handoff tests on both targets.
  Formatting, whitespace checks and package Clippy
  (`--all-targets --no-deps -- -D warnings`) passed. No Wasmtime changes.

  **Fixed worker owners:** JitProcess now queries available logical CPUs once
  and retains the selected count from the specified formula. Pool startup
  preallocates its queue, decoder buffers and private Cranelift/frontend contexts
  before spawning the fixed threads; each Cranelift context also owns its
  register-allocation scratch. An explicit consumer receives the protected Work
  and that worker's reusable resources. There is no shared compiler mutex,
  per-job thread or default consumer. Task 6 supplies actual HCQ graph building,
  compiler settings and emission; finite tests currently exercise the real pool.
  Zero selected workers create no pool or queue.

  Closing the pool drains queued requests outside the queue mutex and joins
  every started thread without JIT/memory locks. Drop also joins, including
  partial startup. Consumer errors/panics close the queue, wake peers, stop
  lifetime admission and retain their diagnostics for explicit join. Poisoned
  queue cleanup still wakes sleepers and releases queued pins. Full process
  cancellation/pressure integration, control-boundary diagnostic delivery and
  injected partial-startup races remain step 6; production startup/admission
  still waits for Task 6 (Task 7 for reshape).

  **Final step validation:** six new tests cover the CPU-count boundaries,
  zero/invalid worker counts, separate and reused compiler/decoder storage,
  queued-versus-running drain, panic/error propagation and poisoned-queue
  wakeup/join. All 626 x86-64 JIT library tests and all 38 background/reshape
  tests on AArch64/QEMU passed, along with package Clippy, formatting and
  whitespace checks. No Wasmtime changes or production HCQ activation.

- [x] **6. Close cancellation, pressure, startup and shutdown races.**
  Connect mapping invalidation, version replacement, HCQ pressure and process
  closure to background admission. Cancel only the exact queued/reserved/running
  identity; release its slot pins and compiler references once. Workers with
  stale epochs cannot publish or clear newer ownership. Keep installed LCQ
  usable when optimization is deferred.

  Roll back partial startup completely: stop and wake created workers, join
  without JIT/queue/memory locks, and release owned resources before returning
  the failure. Shutdown closes admission before draining/waking/joining workers.
  Respect GPU teardown dependencies and ensure no worker/process ownership cycle
  prevents closure. Deliver worker implementation failures at the next control
  boundary; do not relabel them as optimizer rejection or silently disable HCQ.

  **Exit:** controlled enqueue/invalidate/dequeue/shutdown races leave no orphan
  tokens, references or threads. Capacity deferral and stale work do not block
  guest execution or affect a newer request; failed startup is fully unwound.

  **Checkpoint — process closure and startup ownership:** Lifetime keeps only a
  weak reference to the process's one registered background queue. Registration
  requires Open admission and precedes thread creation, so terminal closure
  cannot miss a starting pool. A duplicate pool cannot replace or close the
  existing registration. Failed construction drops all started threads and
  resources; terminal admission is checked again before returning a started pool.

  Both shutdown request APIs now close/drain the registered queue and wake its
  sleepers after releasing JIT state, including cleanup after a prior process
  failure. A reservation acquired before closure cannot enqueue afterward.
  Running jobs observe the existing invalidated admission epoch and retain
  safe immutable inputs until they finish. Requesting stop does not join or
  wait for compilation; explicit pool join remains outside JIT/memory locks
  and must precede final JIT reclamation after dependent GPU teardown.

  **Checkpoint validation:** six new tests cover failure at each of four thread
  creation positions, duplicate registration, queued/running/late-reservation
  shutdown, startup on a terminal process, closure after failure and the actual
  JitProcess stop API. All 632 x86-64 JIT library tests, 34 background tests and
  four engine shutdown tests on AArch64/QEMU passed. Package Clippy, formatting
  and whitespace checks passed. No Wasmtime changes.

  **Checkpoint — original worker diagnostics:** the first terminal worker
  failure now stores its owned diagnostic under lifetime state before setting
  the existing pending control notification. It closes the registered queue
  outside that lock. Peer failures, join order, repeated shutdown and Drop
  cannot overwrite it; an earlier unrelated lifecycle failure remains primary.
  Join-time failures are recorded too, rather than relying on later Drop.

  The canonical slice error path enriches internal failures with that diagnostic
  after recording exact progress/register context and releasing native/FP
  ownership. Process stop/shutdown and vCPU registration return the same detail.
  No extra lock, flag or lookup is added to successful execution or native links.
  Errors and panics remain failures, never optimization rejections. A failure
  during cold completion does not undo the already-completed instruction or
  execute its successor.

  **Checkpoint validation:** four new tests cover original error/panic delivery
  through real workers to vCPU/process APIs, precise cold-completion progress,
  first-worker failure retention and precedence of earlier lifecycle failures.
  All 636 x86-64 JIT library tests, 36 background tests and six engine shutdown
  tests on AArch64/QEMU passed. Package Clippy, formatting and whitespace checks
  passed. No Wasmtime changes.

  **Checkpoint — process-owned join:** JitProcess now owns the optional pool.
  Its construction-time startup method requires exclusive process ownership
  and an explicit consumer; normal construction still leaves it dormant until
  Task 6. The zero-worker policy creates no pool. Request-stop remains nonjoining;
  final teardown joins owned workers before asking Lifetime to reclaim code.
  A recorded worker error does not bypass join. Existing runtime GPU-before-JIT
  teardown ordering is unchanged.

  The pool is moved out of the process owner mutex before join/drop. A distinct
  Joining state prevents a concurrent teardown from mistaking that temporary
  absence for completed cleanup. Process destruction also closes and joins any
  remaining pool. Worker captures retain Lifetime/shared inputs, not JitProcess;
  no process/thread ownership cycle is introduced. This is cold ownership work,
  not a new check or mutex acquisition on guest execution or sample collection.

  **Checkpoint validation:** four new tests cover owned idle/zero-worker closure,
  join concurrent with an active compiler and another teardown caller, join
  despite a worker failure, and last-owner destruction without reference cycles.
  All 640 x86-64 JIT library tests, 13 runtime process-execution tests and ten
  process-pool/shutdown tests on AArch64/QEMU passed. Package Clippy, formatting
  and whitespace checks passed. No Wasmtime changes.

  **Checkpoint — pressure and invalidated work:** seed/reshape admission now
  checks a nonblocking cache-usage snapshot before taking a reservation. Cache
  contention or soft-limit pressure defers with score seven/three; no cache
  lock spans JIT-state or queue access. Workers check pressure at dequeue and
  input/phase boundaries. These snapshots reserve no storage: Task 6 allocation
  and publication must still enforce capacity and validate captured identities.

  The consumer distinguishes cancellation, capacity deferral and implementation
  failure. Expected abandonment drops exact reservations and strong inputs,
  without permanent rejection or process failure. It also resets any unfinished
  frontend builder before reusing compiler resources. Real errors/panics retain
  the existing fatal diagnostic path. Installed LCQ remains usable under HCQ
  pressure; ordinary native execution gains no new check.

  **Checkpoint validation:** six new tests cover nonblocking cache observation,
  pressured admission/dequeue, running-work deferral and compiler reuse, old-job
  cleanup against a newer running reservation, HCQ family eviction during a
  reshape, and real executable-permission invalidation/republication during
  compilation. All 646 x86-64 JIT library tests and 49 selected background tests
  on AArch64/QEMU passed, along with package Clippy, formatting and whitespace
  checks. No Wasmtime changes. Native Arm hardware validation remains pending.

  Production still starts no workers or admission until Task 6 supplies the
  actual compiler; real HCQ publication validation belongs there. The agreed
  activation boundary remains unchanged.

- [x] **7. Validate Task 5 and hand off to HCQ construction.**
  Run the affected unit/integration tests, host-shape checks and AArch64/QEMU
  cases, including the eight/four thresholds, migration, overflow, queue races,
  worker failure and shutdown. Smoke-test simplegfx, es2gears and textured_cube
  with the existing local override. Use focused perf captures if needed to check
  sampling/admission overhead; keep captures in `dump/`, not a new framework.

  Remove superseded sample-only placeholder code and task-local adapters.
  Update the spec only where agreed contracts changed, and record actual
  production activation, validation and remaining hardware limitations here.
  Describe the exact request/reference handoff to Task 6 and the reshape hooks
  reserved for Task 7. No commit, push or dependency-pin change unless requested.

  **Exit:** every Task 5 criterion has code/test evidence, ordinary native hits
  remain free of heat bookkeeping, and no HCQ emission/performance claim is made
  before the real compiler exists. Any deferred activation is named explicitly.

  **Task 6 handoff:** construct the real consumer through
  `JitProcess::start_background` before sharing the process owner. The pool's
  CPU-count policy and process-owned stop/join path already exist. Enable seed
  admission at the existing cold sampling sites only when this pool exists;
  copy the returned `AdmissionSnapshot`, release the sampling identity lock,
  then call `Lifetime::admit_seed`. Never nest that lock with the cache or queue
  lock. Zero workers still means observations without admission.

  The consumer receives private `Resources` and one protected `Work`.
  `Work::observation` supplies the immutable request; `Work::lcq` acquires only
  named, already-demanded LCQ inputs with their reachability and strong immutable
  unit references. Discovery must respect the 2048-instruction candidate bound
  and never inspect live vCPU tables. Retain required snapshots through emission
  and publication; use `Work::check` between phases and propagate expected
  cancellation/deferral separately from compiler failures. This check is not a
  publication transaction: the real publisher must revalidate all captured
  input versions, dependencies, admission and membership while installing its
  output, and enforce allocation limits. It must release unpublished output
  and exact reservations on failure; dropping `Work` already releases its
  compiler protection and request token.

  **Task 7 hooks:** `sample_transfer` accepts an explicit logical source block
  and instruction for a future multi-block HCQ source. Boundary snapshots,
  `admit_reshape`, the shared queue and generation-specific family reservations
  are implemented, but no production reshape request is admitted until its real
  consumer exists. Instruction membership selection/reservation, overlap
  trimming and replacement publication remain Task 7, not hidden work in this
  queue. Do not activate reshape just because Task 6 can consume seeds.

  **Coverage audit:** sampling tests cover both thresholds, collisions,
  immutable copies, stale deferral and overflow; lifetime tests cover exact
  ownership, queue order/contention, replacement, pressure and shutdown.
  Native observation/FP tests exercise register, lazy-flag and sticky-status
  preservation; compiler shape tests verify the existing terminal budget
  instructions and native link/PIC/return paths on both targets. The added
  guest-migration test exercises `run_slice` across two vCPU owners and proves
  heat and sample phase stay with each vCPU. The former sample-only rearm now
  routes to the real observer; no backend-absent consumer or retry loop remains.

  **Final validation:** all 647 x86-64 JIT library tests, 13 runtime
  process-execution tests and 181 affected AArch64/QEMU tests pass, along with
  package Clippy, formatting and whitespace checks. The Arm selection covers
  engine execution/completion/sampling/shutdown, ABI, FP ownership, sample
  tables/identities, queue/workers/reshape, memory invalidation, native
  observation, compiler shapes and observer failure. It includes the new
  migration test. Release smoke runs of simplegfx, es2gears and textured_cube
  presented frames and exited cleanly on SIGINT with zero rejected SVC kinds; logs are in
  `dump/task5-validation-4o7OgS/`. These concurrent-with-tests runs are not
  performance measurements. A later es2gears repeat reached a displayed 60.2 FPS
  after the broad suite had stopped and also exited cleanly; this is a smoke
  observation, not a comparative benchmark. The explicit AArch64 fatal-fault subprocess exits
  with SIGSEGV and `reason=unattributed-native-pc`, as required. The broad
  AArch64/QEMU suite was interrupted during unrelated exhaustive arithmetic/flag
  cases, without a reported failure; it is not counted as a completed pass.
  Native Arm hardware validation is still outstanding. README and
  backend integration status now distinguish active sampling from dormant HCQ.
  No fork changes, dependency-pin update, commit or push were performed.
