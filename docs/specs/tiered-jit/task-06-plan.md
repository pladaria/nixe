# Task 6 implementation plan

Status: steps 1–2 complete; step 3 in progress (ownership, entry freeze and architectural/native CFG liveness).
Next: lazy-NZCV representation/merge contracts. Production is unchanged.
Task 5 supplies cold-path sampling, bounded admission and a tested worker pool.
Production still executes LCQ only: seed admission and workers are dormant,
and reshape admission remains reserved for its real Task 7 consumer.

This is a working checklist for
[Task 6](spec.md#task-6-add-deterministic-multi-entry-hcq), not another
specification. Follow [CONTRIBUTING.md](../../../CONTRIBUTING.md). Update each
step in place with decisions, remaining work and validation; do not create a
separate changelog or testing framework. Agree architectural changes with the
maintainer and update the affected spec before implementing them.

## Scope and sequencing

Compile deterministic, demand-proven regions with one copy of each included
InstructionKey, internal SSA edges and real selected native entry labels.
Publish them through the existing bounded cache/lifetime machinery, then
connect production seed admission to the real compiler. LCQ remains the
synchronous demand path and retained baseline, not an obsolete backend to delete.

Do not implement trace speculation, deoptimization, another decoder or semantic
IR, an ordinal entry dispatcher, per-entry body copies, runtime tuners or a
benchmark framework. Reuse existing instruction semantics and native ABI
contracts. Reshape discovery/replacement, negative boundary results and the
full collision-trimming policy remain Task 7.

**Agreed sequencing:** implement the minimum exact candidate InstructionKey
batch reservation/revalidation in Task 6, step 3, before liveness or backend
work. Task 5's seed token only deduplicates one root/version; `FamilyOwners`
indexes published membership, not
in-flight candidates. Neither prevents two different roots compiling the same
instructions concurrently.

Until Task 7 adds trimming, a reservation conflict discards/defers the candidate
before backend work, releases only its tokens and does not permanently reject its
seed. Keep trimming, reshape and replacement in Task 7. Do not silently permit
overlapping builds, serialize the whole compiler, change the worker-count
policy or activate production before reservation validation and cleanup are tested.

Tests accompany each step. Homebrew execution and manual performance checks
belong to the maintainer; do not launch homebrews unless explicitly requested.
Do not run them automatically at checkpoints or at final validation.

## Starting points

- `lifetime/background{,/work,/workers}.rs`: immutable observations, exact seed
  tokens, `Work::lcq`, cancellation checks, private `Resources`, failure delivery
  and fixed workers. `Work::check` is not a publication transaction and does not
  freeze every subsequently selected input.
- `engine/background.rs`: process-owned startup/stop/join. The real consumer
  must capture shared memory/Lifetime, never its owning `JitProcess` or pool.
- `sampling.rs` and `lifetime/unit/sampling{,/boundary}.rs`: per-vCPU tables and
  source validation. Current calls intentionally disable admission. Release
  identity lookup locks before attempting queue admission.
- `lcq{,/compiler}.rs`, `lowering.rs`, `fp_lowering.rs`, `simd_lowering.rs` and
  `analysis.rs`: current decoder use, instruction lowering, use/def and
  observation-aware CFG liveness. Several compiler helpers are still embedded
  in the straight-line LCQ driver; extract only what the real HCQ consumer needs.
- `lifetime/unit{,/ownership,/links}.rs` and `executable{,/output}.rs`: unit/family
  metadata, current membership, retained LCQ, staging, bounded W^X storage,
  publication, link cutover and reclamation. Existing synthetic HCQ tests are
  foundation evidence, not an implemented region compiler.
- `lcq/invocation.rs`, `engine/completion.rs`, `native/observation.rs` and
  `native/poll.rs`: actual-source exits, fault reconstruction, sampling and
  resumable polls. Audit assumptions that instructions form one contiguous
  fragment, the first entry is the source root, or an instruction's array index
  is the executed prefix length before allowing real HCQ through these paths.
- `/home/pladaria/projects/wasmtime`, branch `nixe`: `nixe::set_entries`,
  selected label offsets, physical state maps, observable FP and pinned ABI
  support already exist. Inspect current code/pending changes before modifying
  the fork. Use `--offline --config /tmp/nixe-observable-fp-local.toml` for Cargo;
  preserve existing edits and portable lockfile state. No commit, push or
  dependency-pin update unless requested.

## Steps

- [x] **1. Settle the region contract and the Task 7 dependency.**
  Trace one immutable seed request through discovery, entry selection, lowering,
  publication and native execution using current code. Resolve the reservation
  sequencing above with the maintainer, and update the task boundary in the spec
  if agreed. Record the decision here before dependent implementation.

  Identify the minimum region representation: canonical instruction/block
  identities and order, successors, selected entries, strong baseline inputs,
  captured reachabilities/dependencies and executed-path state. Use existing
  decoded instructions and CLIF, not a new semantic IR. Specify deterministic
  leader/worklist ordering and entry-freeze revalidation, including an external
  demand/link arriving after entry selection. Such a late entry must retain its
  correct LCQ path, never enter an unexported HCQ label.

  Audit liveness, dirty-state/NZCV joins, FP activation, exclusive state,
  executed-path budget accounting and fault/completion source attribution.
  Identify the exact frontend/backend seams that need changes; do not assume
  LCQ's linear prefix arithmetic or single-entry helpers generalize unchanged.

  **Exit:** the activation/ownership dependency is resolved, required spec
  changes are agreed, and subsequent steps have one concrete data/publication
  contract without a temporary alternative execution architecture.

  **Closed:** the maintainer approved moving minimum batch reservations into
  Task 6, step 3. The spec's task boundaries now reflect this sequencing;
  Task 7 extends the same mechanism with collision trimming and reshape.
  The source audit establishes the following implementation contract.

  - **Capture and graph (step 2).** `Work::observation()` supplies the immutable
    admission input; `Work::lcq()` returns a demanded BlockKey, captured
    ReachabilityVersion and strong `unit::Snapshot`. Retain distinct input
    units once, with their generational handles, code versions, instruction
    words and dependencies. Keep the per-demand key/version separately: two
    demanded keys are not interchangeable merely because their words overlap.
    Decode these owned words with the existing decoder. The region needs an
    ordered unique instruction table (InstructionKey, bits, decoded instruction),
    canonical blocks referencing that table, and typed internal/external edges.
    These are compiler bookkeeping, not a second semantic IR. Maintain lookup
    indexes separately from deterministic vectors; never emit in hash order.
    Use the spec's seed/sample/direct/fallthrough/taken priorities and full-key
    tie-breaks. Split at leaders before admitting complete blocks against the
    distinct-instruction ceiling. Preserve an external edge when an input is
    absent, foreign-owned or does not fit; conflicting captures cancel the job.

  - **Freeze and entries (step 3).** The short cold-state operation must validate
    every captured demanded key/version and unit generation, not just the seed,
    and acquire the agreed all-key reservation with one exact build identity.
    Allocate storage outside the lock. Derive entries from the seed, captured
    dynamic targets and actual external incoming roots; inspect static sources
    by full target key and registered dynamic/return roots, not just installed
    static bridges. Source membership is determined by its logical instruction,
    not merely by whether its LCQ unit supplied some region words. Order frozen
    entries by canonical block order. Do not manufacture coverage-only slots.
    A new incoming edge after this freeze cannot add an entry to compiled code:
    it continues through that PC's LCQ unless the PC was already exported. The
    selected keys and all baseline inputs are revalidated at publication; late
    unrelated demands alone do not invalidate the body or justify inventing a
    native label. Task 7 may later reshape that boundary.

  - **SSA and semantic state (steps 3–4).** `analysis::liveness` already solves a
    CFG fixed point. LCQ's current singleton setup deliberately strips
    observation sets and tracks inherited dirty inputs separately; do not copy
    that setup into HCQ. Solve the region with its real observation requirements
    and explicit external contracts. At joins, carry needed register values as
    block parameters, including unchanged incoming values on a path where
    another predecessor writes them. Dirty state is a may-dirty union; a clean
    canonical home must not replace a dirty predecessor's SSA value. Selected
    entries define their own inputs before joining the body. Carry compatible
    lazy NZCV recipes through SSA; reconcile incompatible recipes to explicit
    flag values at merges, without storing/reloading canonical NZCV. Preserve
    partial-register writes. FPSR is currently invocation-owned, not an ordinary
    register live-in: retain software status and pending host status at every
    observation, and make FP activation valid on every incoming path. Keep
    exclusive-monitor operations and memory ordering in the shared lowering;
    they are not removable register-only effects. FP-mode changes and calls
    remain external boundaries.

  - **Backend seam (step 4).** Extract reusable instruction emission and state
    capture from `lcq/compiler{,/memory,/fp}.rs` only as HCQ consumes them; keep
    LCQ's compile policy. The fork already implements `nixe::set_entries` and
    exported native offsets. Each declared entry has no CLIF block parameters
    and defines its own inputs; it can then branch to internal parameterized
    blocks. Its analysis-only root is not an executable ordinal dispatcher.
    Use these entries with one optimized body and the existing per-entry ABI
    adapters. Multi-entry optimization, physical maps and fixed-frame extent
    still require executable tests; the API's existence is not that evidence.

  - **Executed-path observations (step 5).** Charge each completed canonical
    block along the taken SSA path, with checks on backedges and external exits.
    Choose cycle-closing checkpoints using a deterministic DFS over all selected
    entries: removing its backedges must leave an acyclic graph, including for
    irreducible control flow. Do not put checks on remaining forward edges.
    Fault/exit metadata must identify the logical source block/instruction and
    its uncharged completed prefix within that block. `lcq/invocation.rs` now
    derives this prefix by enumerating the unit's instruction array; that is
    wrong for a region with joins, loops or alternate entries. Replace that
    assumption with explicit prefix metadata and the mapped path budget. Retry
    charges nothing twice; successful cold completion owns the pending
    instruction exactly once. `sample_lcq` and `completion_sample` also assume
    the first entry is the root (and terminal sampling uses the last instruction).
    Generalize their HCQ callers to actual source identities, using
    `sample_transfer` where applicable. Preserve source-local poll resumption,
    fault commit stages and FP ownership without hot current-unit bookkeeping.

  - **Publish and activate (steps 6–7).** `prepare_unit` currently discovers
    synthetic HCQ baselines by scanning resident units for each instruction.
    Replace that search with the captured input handles and explicit coverage;
    retain each required baseline once, including selected-entry baselines.
    Preparation and final publication must validate those inputs and the exact
    candidate claim, not only selected dispatch payloads or `Work::check()`.
    Freeze the dependency union, validate captured executable images outside
    JIT state, then use the existing admission epoch/cursor checks to close the
    publication race. Preserve pre-IC store semantics. Publish complete maps,
    family ownership and cutover records before exposing selected entries;
    release claims atomically with successful publication or by exact-token
    cleanup on abandonment. Reuse cache accounting, baseline pins, maintenance
    and epoch reclamation. Finally connect `engine/background.rs` to the real
    consumer, retaining Lifetime/memory rather than the owning JitProcess;
    reshape admission stays disabled. Capacity/conflict deferral is not a
    permanent seed rejection or successful compilation.

  Validation for this checkpoint: source audit of the paths named above and
  the local fork's entry API. No executable behavior changed, no test result is
  claimed, and no homebrew was launched. Reservation sequencing is approved
  and documented; no step-1 decision remains open.

- [x] **2. Build deterministic discovery and canonicalize overlapping inputs.**
  From `Work::observation`, explore only named, already-published LCQ through
  `Work::lcq`; retain strong immutable snapshots and their versions. Decode their
  captured words with the existing decoder. Never fetch an undemanded successor
  or inspect live vCPU tables. Treat foreign membership and external calls,
  returns, semantic/runtime and FP-mode boundaries as the spec requires.

  Apply the spec's worklist priority and all tie-breaks explicitly, independent
  of hash iteration or acquisition order. Merge overlap by InstructionKey,
  reject conflicting captured bytes as stale, create/split canonical leaders
  and deduplicate instructions before charging the single 2048-instruction
  budget. Include a complete canonical block only if it fits; otherwise retain
  the incoming external edge. Do not add separate block/page/entry limits.

  **Exit:** identical observations and input versions produce identical
  instruction/block order. Tests cover diamonds, loops, overlapping/interior
  roots, order permutations, absent successors, conflicting snapshots and the
  2048/2049 boundary. Calls do not pull in callees or return continuations.

  **Implemented:** `hcq.rs` and `hcq/discovery.rs` build an owned
  graph from demanded LCQ snapshots, retaining distinct generational unit owners
  once and each demand's reachability separately. Matching words merge by full
  InstructionKey; incompatible bytes/contexts are stale. The existing decoder
  and LCQ boundary classifier supply instruction identity and semantic stops.
  Canonical blocks split at demanded/observed/direct-target/interior leaders,
  gaps and post-terminator PCs. Direct edges resolve only to captured labels;
  calls/returns and semantic boundaries remain external. This is graph formation,
  not public entry selection or optimized emission.

  For a selected input set, words are ordered by PC within the validated execution
  context; blocks are seed-first, then ascending leader PC. Input/unit indexing
  is also independent of acquisition order. This canonical output order does
  not replace the spec's priority order for discovering/selecting inputs.

  `Graph::discover` consumes the immutable seed observation and acquires only
  demanded LCQ through `Work::lcq`. The worklist prioritizes sampled successors
  by count/recency/full key, then direct, conditional fallthrough and taken
  successors; visited keys prevent rediscovering loops. Since samples also
  retain call/return destinations, discovery filters them against the captured
  seed terminator rather than treating every observation as an eligible edge.

  `Work::extent` revalidates each captured demand and performs point lookups for
  its instruction ownership and demanded interior leaders. Its scan is bounded
  by that LCQ image, never the process registry, and its result storage is
  allocated outside JIT state. A foreign-owned instruction ends the prefix.
  Discovery admits complete canonical blocks against the single shared 2048
  instruction ceiling, charging matching overlap once. A block that does not
  fit remains external; smaller pending candidates can still fit. Selection and
  final graph formation share the same builder rather than rebuilding the graph.

  Reservations, final input/entry revalidation and liveness remain step 3;
  reshape discovery remains Task 7. This graph is not yet optimized native code.
  Production admission and workers remain dormant until step 7.

  Validation: 26 tests added across step 2 cover overlap/order permutations,
  interior targets,
  diamonds/backedges, missing destinations, calls/returns, semantic stops,
  malformed/stale captures, foreign membership, priority/budget selection,
  the exact 2048/2049 boundary and real worker snapshots surviving retirement.
  The full host JIT library suite passed: **673/673**
  with `cargo --offline --config /tmp/nixe-observable-fp-local.toml test -p
  nixe-cpu-jit --lib`. JIT-only Clippy passed with `cargo clippy --offline
  --config /tmp/nixe-observable-fp-local.toml -p nixe-cpu-jit --all-targets
  --no-deps -- -D warnings`; including dependency linting stops on existing
  `type_complexity` warnings in `crates/memory/src/range.rs`. Formatting and
  whitespace checks passed. No AArch64 execution or homebrew run was performed
  for this checkpoint; the fork and portable lockfile state were left unchanged.

- [ ] **3. Freeze ownership, selected entries and observation-aware liveness.**
  Implement the agreed minimum candidate reservation before liveness/backend
  work. Revalidate all captured identities in the short batch operation; retain
  exact-token ownership through publication or abandonment. Allocate required
  bookkeeping outside the JIT-state lock and keep it bounded/accounted. Published
  membership alone is not a substitute for the in-flight reservation.

  Acquire the whole batch or none. If any key is now foreign-owned or reserved,
  defer the candidate before backend work without waiting, immediate retry loops
  or permanent seed rejection. Cancellation releases only this build's exact
  tokens; unrelated candidates remain free to compile in parallel. Task 7 adds
  successor-collision trimming to this same mechanism, not a second registry.

  Freeze selected entries from the seed, actual outside incoming links and
  included sampled dynamic targets using existing indexes. Keep the distinction
  between a canonical leader and a public native entry: mere coverage never
  creates a dispatch slot. Freeze owned instruction images and their dependency
  union without duplicating overlapping instructions or per-entry metadata.

  Run shared CFG liveness to a fixed point, with PRE/POST observations and full
  partial-write semantics. Compute each selected entry's true live-ins and
  internal merge contracts, including dirty values, lazy NZCV and FP status.
  Selected entry inputs must not depend on a predecessor that entry bypasses.

  **Exit:** competing roots cannot both reach backend work with overlapping
  reservations; stale cleanup cannot affect a newer build. Selected-entry and
  loop/diamond liveness tests cover registers, flags, faults and partial writes;
  coverage-only instructions have no newly created dispatch slots.

  **Checkpoint — exclusive candidate reservations:** `Work::reserve_candidate`
  owns the immutable graph and revalidates every captured demanded key,
  reachability and generational LCQ owner before atomically claiming all its
  InstructionKeys. A published-family or current candidate conflict returns
  `Deferred` without partial claims or permanent seed rejection. Claims reuse
  the exact seed-work token plus admission epoch; stale cleanup cannot remove
  a newer claim. Independent candidates can remain claimed concurrently.

  The point-lookup index allocates and charges growth outside JIT state, retains
  accounted capacity for reuse, and releases storage at terminal teardown.
  The guard borrows compiler protection and releases its own keys on ordinary
  abandonment or unwind, before dropping strong graph inputs outside the lock.
  `Candidate::check` revalidates the captures and ownership; the eventual HCQ
  publisher must use its locked validation inside the publication transaction
  in step 6. This is not yet connected to production compilation/publication.

  Eight focused tests cover all-or-none conflicts, stale nonseed inputs before
  and after acquisition, old/new epoch cleanup, unwind and capacity reuse,
  index growth with live claims, shutdown protection and concurrent disjoint
  candidates. Entry/dependency freeze and observation-aware CFG liveness remain
  outstanding; this step is not closed.

  **Validation:** all 681 host JIT library tests passed with the local fork
  override; JIT Clippy (`--all-targets --no-deps -- -D warnings`), formatting
  and whitespace checks passed. No AArch64 or homebrew execution was performed.
  The Wasmtime fork and the pre-existing portable lockfile state are unchanged.

  **Checkpoint — selected entries and dependencies:** `Candidate::freeze`
  revalidates the claimed graph and sweeps existing dispatch/link indexes for
  included instructions. It selects the seed, included sampled indirect targets
  and targets with external static/PIC sources (including returns), in canonical
  block order. Mere demanded leaders remain internal; coverage never creates a
  slot. Static discovery includes pending associations, not just installed
  bridges. Both source tests use the logical exit InstructionKey, so including
  a prefix of a baseline does not accidentally classify its excluded terminal
  as internal. All bookkeeping allocation and dependency union work stays
  outside JIT state; there is no resident-unit or reader-table sweep.

  The frozen owner retains the candidate's reservations, unique strong baseline
  captures, block-index entry list and deterministic deduplicated dependency
  union. Mapping identities remain distinct even for the same physical page.
  If an external interior entry appears before freeze without a captured
  baseline/leader, cancel before lowering and release the claims; a subsequent
  sample can rediscover it. Do not fabricate an unpinned entry or retry inline.
  Incoming edges after freeze do not reselect labels: they retain LCQ unless
  their target was exported. Locked publication validation remains for step 6;
  observation-aware CFG liveness and merge contracts are still outstanding.

  Eight focused tests cover internal/coverage-only blocks, pending static
  sources, split-seed indirect samples versus direct-branch hints, registered
  indirect/return roots, partial baseline inclusion, late-demand/edge races,
  input replacement and deterministic dependency deduplication. No production
  HCQ compiler or publisher is activated by this checkpoint.

  **Validation:** all 689 host JIT library tests passed with the local fork
  override, plus JIT Clippy (`--all-targets --no-deps -- -D warnings`), format
  and whitespace checks. No AArch64/homebrew execution; no fork or dependency-pin
  changes. The pre-existing portable lockfile state is preserved.

  **Checkpoint — architectural CFG liveness:** `Frozen::analyze` runs outside
  JIT state on the reserved graph, with cancellation checks before and after.
  `hcq::flow` reuses shared normalized instruction effects, block summaries and
  the CFG liveness fixed point; it retains PRE/POST observation sets rather than
  copying LCQ's stripped singleton analysis. Results are indexed by canonical
  block and unique instruction ordinal. Unsupported/invalid boundaries observe
  PRE state without committing the encoding's nominal writes.

  External exits and deterministic DFS backedges carry full control-exit
  observability. The iterative traversal also covers irreducible cycles and
  disconnected sampled components, without polling every lower-address edge.
  Step 5's emitter must consume these same backedge choices, not add a different
  unmodeled set of polls. Per-instruction records distinguish architectural
  liveness before and after the instruction. The provisional local-write
  worklist has been replaced by the native/inherited-dirty analysis below.

  These are **architectural obligations, not native input bindings**. An
  observation of a clean canonical home must not force an entry load. The native
  layer below supplies materialized live-ins and inherited fast-entry dirty
  state. Lazy NZCV representation remains outstanding; FP activation is covered
  below. Do not use architectural ALL sets as unconditional native entry
  contracts or infer physical flag recipes from bit-level liveness.

  Seven focused tests cover diamonds with independently callable joins,
  prefault destinations and earlier producers, partial integer/vector writes,
  flags and FP PRE/POST state, invalid/unsupported boundaries, closed-loop
  poll observability and an irreducible graph whose checked edges break every
  cycle. The frozen-candidate test also invokes the real analysis API.

  **Validation:** all 696 host JIT library tests passed with the local fork
  override; JIT Clippy (`--all-targets --no-deps -- -D warnings`), formatting
  and whitespace checks passed. No homebrew/AArch64 execution, fork changes or
  dependency-pin update. The pre-existing portable lockfile state is preserved.

  **Checkpoint — native value contracts:** `Analysis::native` computes block
  live-ins/live-outs and per-instruction native liveness/may-dirty sets. It uses
  the same semantic effects, graph edges and observation points as architectural
  liveness. Observation maps require SSA values only for potentially stale
  homes; unmodified canonical state does not become an entry parameter. Shared
  liveness propagates these demands through every predecessor, including the
  unchanged value on a diamond's bypass path and an independently callable
  public join.

  Entry demand and inherited dirty state are solved to a joint fixed point:
  requested writable fast-entry inputs may have stale homes even when this
  region merely reads them. A forward worklist unions inherited/local dirtiness;
  backward liveness retains it at PRE/POST observations until overwritten.
  Additional entry demands feed the next iteration without mixing disconnected
  components. This replaces the provisional local-write-only worklist, rather
  than retaining two dirty-state analyses. Results remain independent of host
  register allocation; a selected label uses its own block's contract.

  FPSR remains invocation-owned and absent from ordinary SSA inputs. FPCR and
  TPIDRRO_EL0 can be read inputs but never inherited-dirty homes, following the
  existing ABI. NZCV demand is bit-precise; choosing compatible lazy recipes or
  explicit-value merges remains the final step-3 work; the FP activation proof
  is described below. No native body or production worker is enabled yet.

  Nine focused tests cover minimal clean-home/fault inputs, read-only inherited
  values until overwrite, public diamond joins, independent prefault entries,
  mutually dependent entry contracts, disconnected components, loop fixed
  points, partial writes/NZCV bits and invocation-owned FPSR/read-only homes.

  **Validation:** all 705 host JIT library tests passed with the local fork
  override, plus JIT Clippy (`--all-targets --no-deps -- -D warnings`), formatting
  and whitespace checks. No homebrew/AArch64 execution, fork changes or pin
  update. The pre-existing portable lockfile state is preserved.

  **Checkpoint — FP activation contracts:** `Analysis::fp` records definite
  activity before/after each instruction and whether its native continuation
  must emit `ensure_fp`. It uses the same `fp_policy` as discovery: guarded
  native operations establish activity; guarded-exact comparisons, bitwise SIMD
  and rejected encodings do not. The activation belongs after the eligibility
  guard, never on its exact-exit path or unconditionally at public ingress.

  Two monotone graph walks compute reachability and paths without a preceding
  native activation in O(blocks + edges). A mixed join remains unknown, while
  all-active predecessors permit eliding another activation. Every selected
  entry contributes an unknown path even if its internal predecessors are
  active; loop backedges cannot hide the first invocation. Unrooted cycles do
  not manufacture a proof. Unknown does not mean inactive: the existing adapter
  must retain an already active invocation and pending status.

  This proof follows successful native continuations. Current exact FP guards
  leave the region, and resumable polls/fault retries preserve or restore FP
  ownership. Step 4 must consume this per-path contract instead of LCQ's single
  translator-wide `fp_activation.is_some()` test; a new exact-helper rejoin
  without restoring ownership would violate it. FPSR remains invocation-owned,
  not a phi/input or a status reset at a block boundary.

  Eight tests cover straight paths, mixed/all-active diamonds, public joins,
  first-visit loops, irreducible flow, comparison/SIMD non-activators,
  disconnected/unrooted cycles and rejected FP encodings. Lazy-NZCV recipes
  are still pending; step 3 is not closed and production remains LCQ-only.

  **Validation:** all 713 host JIT library tests passed with the local fork
  override, plus JIT Clippy (`--all-targets --no-deps -- -D warnings`), formatting
  and whitespace checks. No homebrew/AArch64 execution, fork changes or pin
  update. The pre-existing portable lockfile state is preserved.

- [ ] **4. Emit one optimized multi-entry body with internal SSA edges.**
  Share the existing instruction-family lowerings between LCQ and HCQ without
  copying semantics. Keep LCQ's fast compile policy unchanged. Build HCQ CLIF
  blocks/parameters for the frozen graph, including loop phis and selected-entry
  definitions; preserve values across internal edges instead of storing and
  reloading canonical state. Reconcile differing lazy-flag representations only
  where the merge requires it.

  Use the fork's real multi-entry support and exported final labels, compile
  with `opt_level=speed` and backtracking allocation, and produce one staging
  image with per-entry canonical/fast contracts. Canonical ingress loads only
  that entry's true live-ins. Honor fixed-frame extent and host landing-pad
  requirements; do not grow the host stack or emit a wrapper/ordinal dispatcher.
  Resolve any HCQ-only size/shape limit through the specified rejection/split
  policy, not silent fallback to a different ABI or all-LCQ region emission.

  **Exit:** every selected PC executes through its actual label and agrees with
  the shared semantics/interpreter. Internal forward edges have no bridge,
  canonical round trip, lookup or budget check. Inspect CLIF/native shapes on
  both targets; exercise loop-carried values and entry-specific live-ins.

- [ ] **5. Make region observations, polls and faults precise.**
  Carry the executed canonical-block cost in the existing SSA/pinned budget;
  check at all required backedges and external exits, with no check on ordinary
  internal forward edges. Handle multi-entry/irreducible cycles so no native
  cycle can avoid a bounded control checkpoint. Charge branch work and prefixes
  once, including an exit midway through a block, without replay on resume.

  Generalize the shared invocation/fault/completion metadata and consumers to
  actual region source blocks/instructions. Do not derive completed work from
  a deduplicated instruction-array index. Retain source-local continuations and
  physical state maps for every required observation; preserve lazy flags,
  software/hardware FPSR, FP ownership and exclusive-monitor semantics.

  External static/dynamic/call/return edges use the existing native linking,
  PIC/RSB and canonical fallback paths. Internal region polls resume the real
  continuation without executing the terminator or charging its work twice.
  HCQ observations use explicit source identities, not `sample_lcq`'s root/last
  instruction assumption; production reshape admission stays disabled.

  **Exit:** tests enter different public PCs and exercise internal loops,
  faults/partial commits, semantic completions, forced stops, coincident sample
  and slice deadlines and FP state. LCQ and HCQ use the same lifetime/fault
  authority, with no HCQ-only runtime bypass or loss of source attribution.

- [ ] **6. Publish real HCQ families through the existing cache and cutover.**
  Stage final code/relocations/state/fault/entry metadata, allocate through the
  bounded cache, and revalidate captured executable content/dependencies using
  existing memory authority. Under the publication lock, require Open at the
  captured admission epoch and revalidate every input version and exact claim;
  a prior `Work::check` does not authorize a later unchecked publication.

  Install complete family/membership metadata and retained LCQ pins before
  exposing selected dispatch payloads. Register static/PIC/return root cutover
  through existing maintenance; old code remains callable until safe unlink and
  epoch/reference quiescence. Initial promotion must not become an implicit
  reshape of a foreign family. Coverage-only instructions remain nondispatchable.

  Release unpublished spans, metadata and exact claims on stale work/pressure.
  Keep capacity deferral, version-local optimizer rejection and implementation
  failure distinct. Emit only the spec's one debug replacement message per
  published unit, after releasing state; no heat/performance logging.

  **Exit:** real HCQ executes after LCQ promotion and links in both directions;
  late demand retains a valid baseline. Invalidation/replacement/pressure and
  shutdown races cannot publish stale code, lose LCQ baselines, leak storage or
  expose incomplete fault/entry metadata. Failed builds leave LCQ usable.

- [ ] **7. Activate production seed admission and the real worker consumer.**
  Start the fixed worker group during process construction with the real
  compiler and private reusable resources. Connect immutable seed snapshots
  from the existing cold sampling sites after dropping identity-lookup locks.
  Preserve try-lock admission, exact deduplication, queue fairness and the
  zero-worker policy. A guest waits only for required LCQ demand, never HCQ.

  Use the existing cancellation/failure/stop/join protocol; do not introduce
  another executor, hot promotion checks or process/worker ownership cycles.
  Optimizer-only rejection belongs to the exact validated seed version;
  invalid guest semantics, state maps or backend output remain real failures.
  No consumer may discard reshape jobs as success: production does not enqueue
  them until Task 7 supplies their implementation.

  **Exit:** tests exercising actual production sampling/admission observe the
  threshold, background compilation and native HCQ execution without manually
  publishing a synthetic family. Independent nonoverlapping seeds compile in
  parallel, cold LCQ demand continues, and process closure joins all workers.

- [ ] **8. Validate Task 6 and hand off parallel trimming/reshape to Task 7.**
  Consolidate focused evidence for deterministic graphs, overlap deduplication,
  the instruction ceiling, selected labels/live-ins, internal SSA shapes,
  backedge polling, precise observations and real promotion/cutover. Run the
  affected host tests, both encoders, selected AArch64/QEMU groups, Clippy and
  formatting checks. Record which suites actually completed; do not treat a
  partial QEMU run as a pass or emulation as native Arm hardware validation.

  Remove replaced single-entry assumptions and temporary development adapters
  as their consumers migrate. Update this plan, production-status documentation
  and agreed spec changes. State what is active, remaining hardware limits and
  the exact ownership/entry/publication hooks that Task 7 extends. Do not claim
  measured performance gains from optimized emission alone.

  **Exit:** every Task 6 criterion has code/test evidence, production seeds
  reach real HCQ safely, and no placeholder compiler or second semantic lowering
  remains. Homebrew/performance verification is left to the maintainer unless
  explicitly requested. No commit, push or dependency-pin update is implied.
