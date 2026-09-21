# Task 6 implementation plan

Status: complete (steps 1–8), including the production cancellation-churn fix.
Production samples LCQ seeds and promotes them through the bounded HCQ worker
pool. LCQ remains the demand path and retained baseline; reshape admission
remains reserved for its real Task 7 consumer.

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
  source validation. Seed admission is active; reshape admission remains off.
  Identity lookup locks are released before attempting queue admission.
- `lcq{,/compiler}.rs`, `frontend{,/memory,/fp}.rs`, `lowering.rs`, `fp_lowering.rs`, `simd_lowering.rs` and
  `analysis.rs`: current decoder use, instruction lowering, use/def and
  observation-aware CFG liveness. Native instruction emission is now shared;
  LCQ retains its straight-line compile policy and publisher.
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
    consumes explicit fault-prefix metadata rather than an instruction-array
    ordinal; HCQ still needs to charge earlier executed blocks in the native
    path budget. Retry
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
    JIT state, then validate the exact LCQ source identities/lifecycles and
    claims under current Open authority to close the publication race. Unrelated
    maintenance must not discard finished HCQ output. Preserve pre-IC store semantics. Publish complete maps,
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

- [x] **3. Freeze ownership, selected entries and observation-aware liveness.**
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
  the exact seed-work token independently of maintenance epochs; stale cleanup cannot remove
  a newer claim. Independent candidates can remain claimed concurrently.

  The point-lookup index allocates and charges growth outside JIT state, retains
  accounted capacity for reuse, and releases storage at terminal teardown.
  The guard borrows compiler protection and releases its own keys on ordinary
  abandonment or unwind, before dropping strong graph inputs outside the lock.
  `Candidate::check` revalidates the captures and ownership; the eventual HCQ
  publisher must use its locked validation inside the publication transaction
  in step 6. This is not yet connected to production compilation/publication.

  Eight focused tests cover all-or-none conflicts, stale nonseed inputs before
  and after acquisition, maintenance survival and exact-token cleanup, unwind and capacity reuse,
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
  ownership. HCQ emission consumes this per-path contract instead of treating
  an activation emitted elsewhere as proof for this path; a new exact-helper rejoin
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

  **Closed — lazy-NZCV merge contracts:** `Analysis::flags` fixes each block's
  native input/output recipe shape before emission. It reuses `LazyFlags<()>`,
  not a second flag representation. Equal recipes carry their operands in
  `try_map` order, including captured carry/predicate values; operand register
  identities do not affect compatibility. Operation, width and conditional
  literal differences require a packed SSA merge. The shared integer emitter
  debug-checks its produced shape against the analysis classifier, without
  adding shape work to release LCQ compilation.

  The monotone worklist propagates a shape or packed conflict through loops
  and irreducible joins in O(blocks + edges), after a linear instruction scan.
  Public entries contribute packed live bits independently of internal
  predecessors; definitions kill the incoming representation. Dead flag inputs
  create neither parameters nor packing. Dirty-home and bit-demand authority
  remains `Analysis::native`, separate from recipe shape: Canonical and Packed
  have the same SSA shape but are not a clean-home proof.

  Step 4 must bind recipe operands as block parameters and consume
  `packs_edge` only on edges requiring reconciliation, materializing the target's
  native live-in mask in SSA without canonical stores/reloads. FP comparisons
  and MSR NZCV produce packed values; exact FCCMP exits PRE, and rejected
  encodings cannot replace the previous producer. Local per-instruction recipes
  still supply precise observation maps during emission; this block-level
  analysis is not a replacement for those maps.

  Ten focused tests cover compatible producers with different registers,
  operation/width/carry conflicts, conditional operands/literals, public joins,
  bypass/FP/system sources, loops, dead flags, faults/rejected definitions,
  partial flag demand, disconnected entries and irreducible/unrooted cycles.
  Step 3 is closed; actual HCQ emission, executable multi-entry proofs and
  publication/production activation remain steps 4–7, not claims of this closure.

  **Validation:** all 723 host JIT library tests passed with the local fork
  override, plus JIT Clippy (`--all-targets --no-deps -- -D warnings`), formatting
  and whitespace checks. No homebrew/AArch64 execution, fork changes or pin
  update. The missing `/tmp/nixe-observable-fp-local.toml` was recreated with
  the same local crate paths; the portable lockfile is unchanged.

- [x] **4. Emit one optimized multi-entry body with internal SSA edges.**
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

  **Checkpoint — typed SSA blocks and ingress:** `hcq::ssa::Ssa` declares one
  parameterized block per canonical graph block and one parameter-free native
  ingress per frozen selected entry. Each ingress defines its own `nixe_entry`
  values and branches directly to the common body. Public ingress uses only
  its native live-ins; an entry with no flag demand has no dummy NZCV operand.
  The driver passes these labels, plus resumable FP continuations, to the fork's
  `set_entries` once before compilation.

  Parameters carry guest values and compatible lazy-recipe operands in stable
  order. Typed recipe traversal preserves I8 predicates/carry and I32/I64
  arithmetic operands. Reconciliation uses shared flag lowering and only packs
  the target's required bits. Missing bypass values are compile errors, never
  canonical reloads. Each block rebinds its values from its own parameters,
  rather than inheriting the last block visited by the compiler. Only operand
  lists/flag parameters are retained, not a full register array per block.

  `lowering::values` now owns LCQ's former value storage/layout helpers, used
  by both LCQ and HCQ SSA construction. The old copies have been removed;
  LCQ compile policy and instruction semantics are unchanged. Dirty-state
  authority is separate from parameter binding; HCQ observation emission must
  use native flow masks, not infer full dirtiness from a packed merge recipe.

  Eight focused tests use shared integer lowering to build diamonds, loops,
  selected/public joins, carried recipes and overwritten source registers.
  Both encoders compile these fixtures with `speed` and `backtracking`, checking
  real exported labels, final entry maps and bounded frame extent. Pre-backend
  CLIF has no canonical memory traffic or calls. A host-executed scalar packing
  test checks all nonempty NZCV masks against the interpreter.

  This is the SSA/ingress component, not a complete region compiler. The initial
  test-only body driver has been replaced by the real shared-frontend driver
  below. These tests still do not publish or run HCQ regions. In particular,
  compiling loop fixtures without runtime polling is encoder evidence only,
  not permission to execute or publish them before step 5's accounting.

  **Validation:** all 731 host JIT library tests passed with the local fork
  override, including the eight new SSA/packing tests. JIT Clippy
  (`--all-targets --no-deps -- -D warnings`), formatting and whitespace checks
  passed. Both encoders were exercised, but no AArch64/QEMU execution or
  homebrew run was performed. No fork/pin change; portable lockfile preserved.

  **Checkpoint — shared frontend and HCQ body driver:** `frontend.rs` and its
  `activation/fp/memory/system` modules now own the existing native translator,
  instruction dispatch and pending physical-state records. The former copies
  under `lcq/compiler/` have been moved, not retained as parallel lowerings.
  LCQ uses the same shared instruction entry point with its unchanged compile
  policy. Conditional tests (B.cond/CBZ/TBZ) are shared as well.

  `hcq::compiler::emit` consumes the canonical graph, frozen entry indexes and
  analysis with reusable Cranelift scratch. It emits one register/control body,
  direct internal SSA branches and the shared native exits only at actual
  external boundaries. Rebinding/may-dirty masks come from each block's own
  contract. Calls/returns and architectural exits reuse LCQ's LR/source ordering.
  The SSA tests now use this real driver; their provisional body loop is gone.

  Actual integer/FP/MSR NZCV producers explicitly mark written flags. Snapshots
  and FP activation no longer infer all-bit dirtiness from a Packed recipe,
  which can represent only a subset of inherited live bits. Multi-entry tests
  also exposed an ID collision: HCQ entry maps now use the unit-local range
  starting at `1<<59`, separate from ordinary exits, faults and FP continuations.
  These IDs are backend metadata, not a runtime entry selector.

  Seven new tests cover internal versus external records, all conditional
  branch families, partial SIMD values/public entries, partial NZCV dirtiness,
  calls/returns/architectural exits and reusable scratch after an unconnected
  family is rejected. Both encoders compile the real driver with `speed` and
  `backtracking`; each final exit map is translated through the shared physical
  allocation/state validation path.

  **Remaining:** final adapters/ingress packaging and executable multi-entry
  proofs. Emission returns CLIF and pending records only: no
  publishable output, production native region execution, worker activation or implied
  path-budget correctness. Final adapters/ingress packaging remain step 4;
  executed-path accounting/observations and publication remain steps 5–6.

  **Validation:** all 738 host JIT library tests passed with the local fork
  override, including the seven new driver tests and the existing SSA tests
  migrated to the real driver. JIT Clippy (`--all-targets --no-deps -- -D
  warnings`), formatting and whitespace checks passed. Both encoders were
  exercised; no HCQ native-region execution, AArch64/QEMU run or homebrew was
  performed. No fork/pin change; portable lockfile preserved.

  **Checkpoint — memory and system observations:** HCQ now uses the shared
  scalar/vector memory, pair, atomic, exclusive and system lowerings. The emitter
  accepts the process arena size and retains pending fault records alongside
  exits. Arena validation and the CMPXCHG16B capability check are shared with
  LCQ; instruction semantics are not duplicated. Both encoders resolve the
  final physical fault maps through the existing record conversion, retaining
  actual source PCs, PRE state, subaccesses, partial store stages and uncommitted
  first pair reads across internal edges/public joins.

  A regression exposed an HCQ analysis mismatch: MRS FPSR was treated as a
  native destination write although it exits PRE for cold completion. System
  operations completed outside native execution now have PRE observations and
  no native writes; the incoming destination survives until completion. The
  same rule covers FPCR/FPSR writes and runtime system boundaries, while inline
  system operations and CIVAC probes retain their native effects.

  Eight new tests cover these contracts with noncontiguous blocks, public and
  bypass entries, lazy flags, writeback, pairs, atomics/exclusives and system
  state. This remains staged emission only: executed-path charging, runtime
  fault completion and publication are still steps 5–6. These tests do not
  publish or execute HCQ native regions.

  **Validation:** all 746 host JIT library tests passed with the local fork
  override, including eight new observation tests compiled for both host ISAs.
  JIT Clippy (`--all-targets --no-deps -- -D warnings`), formatting and whitespace
  checks passed. No homebrews, AArch64/QEMU execution, fork changes or pin update;
  the portable lockfile is unchanged. Step 4 remains open.

  **Checkpoint — per-path FP activation:** the shared translator now separates
  its current path's definite FP ownership from its list of activation sites.
  LCQ still needs at most one site on its straight native continuation. HCQ
  initializes each block from `Analysis::fp`, activates only after a successful
  native eligibility guard, and elides activation on already-proven paths.
  Public entries and mixed joins cannot inherit another emitted block's proof;
  all-active joins can. The existing adapter preserves an already active
  segment and accumulated status.

  Every activation has a distinct exit/continuation ID pair and source PC.
  Its continuation is exported as a real backend entry, carrying allocated
  guest values and typed lazy-flag operands without a canonical round trip.
  LCQ and HCQ use the same adapter/map conversion; no parallel FP emitter was
  added. HCQ now admits the shared FP frontend, including guarded comparisons
  and typed exact exits, which do not establish native FP ownership.

  Eight tests compile straight paths, public/disconnected entries, mixed and
  all-active diamonds, loops/irreducible cycles, comparisons, exact/status
  exits and post-activation faults on both targets. Final physical activation
  maps and transfers are validated, including narrow lazy operands and unique
  IDs/exported continuations. These tests validate allocation, not native
  region execution; loop fixtures require step 5's path accounting before execution.

  **Validation:** all 754 host JIT library tests passed with the local fork
  override, including the eight new FP emission tests on both encoders and
  existing LCQ native FP execution tests. JIT Clippy (`--all-targets --no-deps
  -- -D warnings`), formatting and whitespace checks passed. No homebrews,
  AArch64/QEMU execution, fork changes or dependency-pin update; portable
  lockfile preserved. Step 4 remains open for final adapters and staging.

  **Checkpoint — physical public ingress and finite execution:** LCQ and HCQ
  now share final entry-contract conversion and canonical ingress construction
  in `frontend::entry`/`staging`; the former LCQ copies were removed. Contracts
  omit eliminated backend operands, preserve entry-specific NZCV masks, and
  load only the selected entry's allocated inputs. Shared staging also appends
  aligned direct continuation jumps for FP activation.

  HCQ resolves every selected label and its allocated map, builds its own
  contract and appends a canonical adapter ending at that label. Final map and
  label indexes are built once, avoiding a full metadata scan per entry. There
  is no ordinal dispatcher or duplicated region body.

  Three tests check both host encoders' actual labels/landing pads, independent
  contracts and empty inputs. A host-executed test enters every selected label
  of finite integer/FP graphs and compares complete guest state with the
  interpreter, exercising internal branches, public joins and FP continuations.
  It owns one unlinked allocation and uses test-only zero-charge canonical
  leaves: this is an ABI proof, not production exit/link packaging or path-cost
  validation. No loop/fault fixture is executed, and no HCQ worker is activated.

  **Validation:** all 757 host JIT library tests passed with the local fork
  override, including the three new ingress tests and existing LCQ execution
  coverage after the shared-code extraction. JIT Clippy (`--all-targets
  --no-deps -- -D warnings`), formatting and whitespace checks passed. Both
  encoders were exercised; finite native execution was on the host only. No
  homebrews, AArch64/QEMU run, fork changes or dependency-pin update; portable
  lockfile preserved. Final HCQ exit adapters and complete staging remain open.

  **Checkpoint — shared physical exits:** `frontend::exit` now owns the
  physical exit adapter and terminal patch construction previously embedded in
  LCQ. LCQ uses it directly; the old implementation is removed. Static links,
  PIC/RSB operations, canonical fallback and sample/slice/control continuations
  retain the existing contracts. The caller supplies the completed prefix;
  it is never inferred from a region's instruction index or PC distance. A
  supplied cost must match the backend checkpoint, and indirect probes require
  a charged terminal checkpoint.

  Three HCQ tests cover static/dynamic calls, jumps and returns on both encoders,
  including noncontiguous source PCs and rejection of missing/mismatched
  charges. Their single-entry, single-path fixtures supply the known cost of
  three instructions; this is not general multi-entry/loop accounting. The
  finite multi-entry execution test also uses the shared exit builder, retaining
  its explicitly test-only zero-charge canonical leaves. Full HCQ staging and
  production publication are not yet connected; step 4 remains open.

  **Validation:** all 760 host JIT library tests passed with the local fork
  override, including existing linked LCQ execution and the three new HCQ exit
  tests. JIT Clippy (`--all-targets --no-deps -- -D warnings`), formatting and
  whitespace checks passed. No homebrews, AArch64/QEMU execution, fork changes
  or dependency-pin update; portable lockfile preserved.

  **Checkpoint — owned HCQ staging:** `Body::stage` consumes final backend code
  and returns one owned output with selected-entry contracts, exit/FP state
  records and fault records. It resolves physical contracts before consuming
  the backend, then appends shared adapters to one body-byte copy. Exit lookup
  uses one map index; source PCs and fault state indexes remain explicit.
  No executable allocation, family publication or worker activation occurs here.

  The finite native execution tests now use this packager instead of assembling
  their own exits/continuations. Static/PIC/RSB fixtures also go through it.
  Two additional tests check deterministic bytes, complete multi-entry/FP/pair
  fault metadata after clearing backend scratch, and rejection of incomplete
  cost arrays or uncharged static/dynamic dispatches on both encoders. Costs
  remain supplied by the caller: known single-path prefixes and explicitly
  zero-charge PRE leaves in ABI fixtures do not implement general HCQ accounting.

  **Validation:** all 762 host JIT library tests passed with the local fork
  override. JIT Clippy (`--all-targets --no-deps -- -D warnings`), formatting
  and whitespace checks passed. Both encoders were exercised, with native
  execution on the host only. No homebrews, QEMU run, fork changes or pin update;
  portable lockfile preserved.

  **Checkpoint — compiler policy and typed rejection:**
  `hcq::compiler::backend::Compiler` shares immutable native target capabilities
  but borrows each worker's existing Context/FunctionBuilderContext. `emit`
  builds the shared body; `finish` consumes explicit exit costs, compiles with
  `speed`/`backtracking` and produces owned staging. The seam between them is
  where step 5 supplies actual path accounting. Scratch is cleared after
  success, resource rejection and backend/staging failure.

  `frontend::target` shares capabilities, reserved-register ABI and landing-pad
  configuration with LCQ; LCQ retains `none`/`single_pass`. Only the backend's
  typed `ImplLimitExceeded` and `CodeTooLarge` become `Failure::Rejected`.
  Unsupported lowering/calls, verifier failures, register-allocation checker
  failures and invalid physical maps remain implementation failures, with no
  string-based classification. No alternate ABI or hidden LCQ recompilation is
  used. The later worker consumer must release the reservation and record a
  rejection only for the validated captured ReachabilityVersion; this compiler
  does not mutate lifetime/admission state.

  Three tests check both targets' tier policies, typed error classification
  and reuse of the same scratch across successful compilation, a real aggregate
  fixed-frame overflow, an unsupported frontend slot and an invalid physical
  map. Finite native multi-entry execution now also uses this target/compile
  path and still agrees with the interpreter.

  **Step 4 boundary:** body emission, physical packaging and compiler policy
  are implemented. General executed-path costs, backedge polling and runtime
  fault/observation accounting are step 5, not proven by constant-prefix or
  zero-charge ABI fixtures. Loops have SSA/backend coverage but are not yet
  executed as HCQ. Publication, version-local rejection recording and production
  worker activation remain steps 6–7; production continues to execute LCQ only.

  **Closure validation:** all 765 host JIT library tests passed with the local
  fork override, including the three new policy/rejection/reuse tests. JIT
  Clippy (`--all-targets --no-deps -- -D warnings`), formatting and whitespace
  checks passed. Both encoders were exercised; native execution was on the
  host only. No homebrews, QEMU run, fork changes or pin update; portable
  lockfile preserved.

- [x] **5. Make region observations, polls and faults precise.**
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

  **Checkpoint — source-local completed prefixes:** the shared translator now
  receives the current instruction's block-local prefix from the LCQ/HCQ
  driver. Pending exits include the dispatch terminator or exclude the PRE
  instruction as appropriate. LCQ consumes these explicit exit costs instead
  of subtracting PCs from its root. HCQ resets the prefix at each canonical
  block, including public entries and compiler-handled conditional terminals.

  `FaultRecord::completed` retains the uncharged PRE prefix independently of
  subaccess/commit stage. Both accesses of a pair retain the same instruction
  prefix. Native fault escape subtracts this field from the reconstructed poll
  counter, no longer using the instruction's ordinal in the unit. Repair/retry
  still charges nothing and successful cold completion owns the faulting
  instruction once. Publication rejects out-of-bound fault prefixes.

  Two new tests cover noncontiguous/public-entry blocks, pair faults, PRE FP
  exits and external jump/conditional/call/return costs on both encoders.
  Existing memory tests also assert local prefixes. These changes remove the
  linear-unit assumption; they do not yet charge earlier HCQ blocks or emit
  internal polls. HCQ execution/admission remains dormant.

  **Validation:** all 767 host JIT library tests passed with the local fork
  override, including native LCQ fault/retry/cold-completion budget coverage.
  JIT Clippy (`--all-targets --no-deps -- -D warnings`), formatting and whitespace
  checks passed. Both encoders were exercised; no homebrews, QEMU run, fork
  changes or pin update. Portable lockfile preserved. Step 5 remains open.

  **Checkpoint — executed-path block charges:** HCQ now charges completed
  driver-owned blocks in the pinned counter before branching. The fork's
  `nixe_charge` emits one flag-preserving instruction, without a state map,
  memory access, call or deadline check. Forward internal edges remain direct
  SSA branches. Calls/returns and PRE observations retain their uncharged local
  prefixes; external edges after a body charge use a zero-cost terminal poll,
  so neither its hot nor cold adapter repeats the work.

  Compiler completion and staging now consume the pending exit's actual local
  cost. The caller-supplied whole-path cost array and zero-cost fixture policy
  have been removed. Finite native tests compare exact sample/slice balances
  from every selected entry through unequal diamond paths and FP continuations.
  External branch/conditional/call/return tests cover both conditional outcomes,
  coincident deadlines and overshoot from small budgets. Fork tests cover both
  targets/allocators, preserved charge bytes, absent extra state maps, invalid
  costs/ABI/fault-span placement and CLIF round-trip with zero-cost checkpoints.

  Internal cycle checks/resumption and source-aware runtime sampling remain
  pending; cyclic HCQ bodies are not executed and production stays LCQ-only.
  The local fork override remains required; no pin update or homebrew run.

  **Validation:** all 768 host JIT library tests passed, plus 41 fork Nixe
  backend tests (both targets/allocators) and four Nixe CLIF parser tests.
  JIT Clippy (`--all-targets --no-deps -- -D warnings`), formatting and whitespace
  checks passed. Native execution was on the host only; no QEMU/hardware Arm
  run. Portable lockfile preserved. Step 5 remains open.

  **Checkpoint — resumable internal cycle polls:** emission now consumes the
  exact backedges selected by the shared flow analysis, including irreducible
  and multi-entry cycles. Conditional checks live on the selected edge, after
  the source block/branch charge. Ordinary forward edges remain plain SSA.

  The fork's nonterminating `nixe_check` compares the pinned counter and skips
  one cold patch when positive; its continuation includes all subsequent
  allocator edits. No new public ingress, internal link bridge, work charge or
  hot-path canonical writeback is required. Cold checks reuse the same native
  request/budget leaf and sampling callback machinery as external terminals.
  Slice/control exits canonicalize POST state at the actual successor PC.

  Internal sample callbacks preserve all allocatable volatile host registers,
  not only guest bindings: optimization can keep additional temporaries alive
  across the check. This preservation occurs only in the cold callback and
  introduces no regalloc clobbers or hot spills. Lazy recipes, FP ownership,
  optimizer spills and exclusive-monitor state survive resumable samples.

  New tests execute conditional/unconditional loops and irreducible cycles
  from multiple public entries against the interpreter, including sample-only
  resumptions, overshoot, coincident deadlines and all three request sources.
  A callback deliberately clobbers all ABI volatiles while a native FP loop
  resumes; its failure also verifies exact canonical state and FP restoration.
  Backend tests check the cold patch/continuation on both targets/allocators.

  Step 5 remains open for source-aware production observations and region
  fault/completion integration. HCQ execution is still test-only; production
  remains LCQ. No homebrews, dependency-pin update, commit or push.

  **Validation:** all 771 host JIT library tests passed, plus 42 fork Nixe
  backend tests and four Nixe CLIF parser tests. Both target encoders and both
  allocators were exercised; native execution was on the host only. JIT Clippy
  (`--all-targets --no-deps -- -D warnings`), formatting and whitespace checks
  passed. No QEMU/native Arm run. Portable lockfile preserved.

  **Checkpoint — exact runtime source identities:** shared LCQ/HCQ emission
  retains explicit source-block and instruction indices in each guest exit,
  including internal HCQ checks. Canonical exit attribution now indexes the
  captured image directly instead of subtracting PCs from its first word.
  Publication rejects inconsistent/out-of-range source indices.

  The real sampling callback accepts internal HCQ checks and resolves the
  emitted source before calling the existing family-aware transfer observer.
  Canonical transfer samples use that same source. Neither path assumes the
  first public entry or last captured instruction; internal family edges do
  not heat boundaries and reshape admission remains disabled. These lookups
  stay on cold observation paths; native hot edges gain no extra work.

  Four new native tests publish real HCQ code through existing lifetime
  machinery and execute the shared production invocation: internal sample-only
  resumptions and slice exits, external callback/canonical boundary samples,
  noncontiguous PRE exits, and unmapped-load fault escapes from different public
  entries. They check exact source words, family attribution, canonical state
  and path-local work charges. Two metadata tests cover invalid source indices
  and semantic-context mismatches. An older source-mutating fixture now updates
  its instruction index too.

  Step 5 remains open for region cold-completion/partial-commit integration
  coverage. This test publication does not activate production HCQ admission
  or replace the real publication consumer planned in steps 6/7.

  **Validation:** all 777 host JIT library tests passed with the local fork
  override. JIT Clippy (`--all-targets --no-deps -- -D warnings`), formatting
  and whitespace checks passed. Native execution was host-only; no QEMU/native
  Arm run, homebrews or additional fork edits in this checkpoint. Portable
  lockfile preserved; no dependency-pin update, commit or push.

  **Checkpoint — owned region memory completion:** real HCQ invocations now
  have native execution coverage for integer/vector pair loads and stores,
  SIMD structure accesses with native and cold committed prefixes, and
  exclusive stores using native, physical-alias and incoming reservations.
  The three new tests exercise 47 configurations, including different public
  entries, successful completion, first/later MMIO failures and reservation
  invalidation after escape.

  Compound-access tests compare canonical state and ordered device operations
  against the interpreter. They destroy native owners, revoke code permission
  and overwrite the already-accessed RAM prefix before consuming the owned
  completion: loads retain prior values, stores never replay, failure preserves
  the correct partial commits, and base writeback occurs only on success.
  Earlier-block lazy flags, hardware FPSR and caller FP state survive escape.
  Only executed prefixes are charged; a successful cold instruction crosses
  coincident sample/slice deadlines once, while a failed one earns no work.
  HCQ completions do not fabricate LCQ seed samples.

  No new HCQ completion path or production runtime change was needed: these
  tests consume the existing shared invocation, fault and completion owners.
  Production promotion/publication and admission remain steps 6/7; platform
  execution validation beyond the host remains part of step 8.

  **Step 5 closed. Validation:** all 780 host JIT library tests passed with
  the local fork override; the seven real HCQ invocation tests also passed
  separately. JIT Clippy (`--all-targets --no-deps -- -D warnings`), formatting
  and whitespace checks passed. No homebrews, QEMU/native Arm run, additional
  fork edits, dependency-pin update, commit or push in this checkpoint.
  Portable `Cargo.lock` preserved.

- [x] **6. Publish real HCQ families through the existing cache and cutover.**
  Stage final code/relocations/state/fault/entry metadata, allocate through the
  bounded cache, and revalidate captured executable content/dependencies using
  existing memory authority. Under the publication lock, require current Open
  authority and revalidate every input version/lifecycle and exact claim;
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

  **Checkpoint — candidate-bound publication transaction:** `Frozen::prepare`
  binds an owned output to the exact captured words, selected entries and
  dependency union. `PreparedUnit` retains a borrow of that frozen candidate,
  so its instruction claims and compiler/input protection cannot disappear
  before commit or cancellation cleanup.

  Preparation and final publication both validate the running job,
  every captured demand/version/lifecycle and every exact instruction claim under
  the existing publication lock. The final check occurs before any directory,
  family, link or dispatch mutation; an earlier successful phase check is not
  publication authority. The guarded path pins the distinct captured LCQ units
  with registry point lookups instead of scanning the process registry for an
  arbitrary matching image per instruction. Selected-root retention, family
  membership and native-PC/dispatch exposure still use the existing transaction.

  Focused tests reject mismatched output and changes between preparation and
  commit (lost claim, seed/nonentry input replacement, shutdown and actual
  memory invalidation). They verify unchanged LCQ roots, actual
  unpublished span reuse, exact captured-baseline pins, no extra public entry
  for internal/coverage-only PCs and claim release on candidate destruction.
  These tests use synthetic native output to isolate publication authority.

  Step 6 remains open: connect real staged HCQ output and executable-memory
  revalidation to this guarded boundary, then validate mixed-tier execution,
  cutover/cancellation/pressure and the replacement diagnostic. No production
  worker/admission activation is implied; that remains step 7.

  **Validation:** all 783 host JIT library tests passed with the local fork
  override, including three new publication tests. JIT Clippy (`--all-targets
  --no-deps -- -D warnings`), formatting and whitespace checks passed. No
  homebrews, QEMU/native Arm run or fork edits in this checkpoint. Portable
  `Cargo.lock` preserved; no dependency-pin update, commit or push.

  **Checkpoint — real HCQ publication and executable-memory revalidation:**
  the HCQ compiler now accepts a frozen candidate and the process executable
  memory authority, emits optimized native output, installs it in the bounded
  cache and publishes through `Frozen::prepare`. This does not activate workers
  or seed admission; production activation remains step 7.

  Revalidation captures only contiguous runs of included instructions, never
  gaps or additional code. Captured words and physical/mapping dependencies must
  match the frozen LCQ image. Exact image stamps and candidate authority are
  checked before compilation, after staging and after preparation; final locked
  publication checks current Open authority, original input identities/lifecycles
  and claims. Step 8 replaces the overly broad epoch/cursor cancellation rule.
  Cancellation, capacity deferral, optimizer limits and internal failures stay
  distinct.

  Integration tests publish real two-entry HCQ code and execute LCQ-to-HCQ and
  HCQ-to-LCQ paths. They also reject changed captured bytes before compilation
  and coordinated memory mutation after output preparation. The latter exposed
  and fixed an old lifetime bug: baseline pins prohibit eviction, not memory
  invalidation. Invalidation drains affected published families first and may
  unlink their LCQ baselines while cancelled compiler references retain storage.
  It no longer fails the memory producer while waiting for an unpublished HCQ
  family to drop. The focused lifetime regression verifies reopening before the
  prepared output drops, rejected stale publication and deferred reclamation.

  Step 6 remains open for the remaining real-code cutover/late-demand,
  cancellation/pressure/shutdown coverage and the replacement diagnostic.

  **Validation:** all 786 host JIT library tests passed with the local fork
  override, including the three new real-publication tests and the updated
  invalidation regression. JIT Clippy (`--all-targets --no-deps -- -D warnings`),
  formatting and whitespace checks passed. No homebrews, QEMU/native Arm run,
  fork edits or production worker activation. Portable `Cargo.lock` preserved;
  no dependency-pin update, commit or push.

  **Step 6 closed — real-code lifecycle and replacement diagnostic:** focused
  tests now demand an unexported interior PC after promotion, retain its separate
  LCQ entry and execute it before and after HCQ withdrawal. Two readers exercise
  warmed static links, indirect PICs and predicted return paths across promotion
  and retirement; they execute the original baselines after the optimized span
  is actually reclaimed. Coordinated executable-byte mutation removes the HCQ
  family and affected baseline, then the same ingress paths execute newly
  compiled LCQ bytes rather than stale optimized code.

  Capacity exhaustion injected after real backend staging returns deferral,
  preserves the LCQ payload and releases staging charges; the same candidate
  and compiler scratch can publish after capacity returns. Shutdown injected
  after native-output preparation cancels publication and releases all code,
  compiler pins and process metadata once their owners drop. Together with the
  earlier locked-transaction tests, this covers exact-claim/input replacement,
  actual invalidation/shutdown cancellation and unsuccessful publication cleanup.

  Successful HCQ publication emits one debug replacement message with the
  family/version and seed identity, after unlocking state. A focused logger
  test checks lock availability and one message for a multi-entry unit, with
  none for LCQ or rejected publication. This uses the existing workspace `log`
  dependency, without adding a diagnostic callback or hot-path checks.

  **Validation:** all 792 host JIT library tests passed with the local fork
  override. The eight real-publication tests also passed after explicitly
  enabling guest-thread return predictions in the lifecycle fixture. JIT Clippy
  (`--all-targets --no-deps -- -D warnings`), formatting and whitespace checks
  passed. Production worker/admission activation remains step 7; cross-target
  execution validation remains step 8. No homebrews or fork edits. The portable
  lockfile adds only the JIT's `log` dependency; no fork pin update, commit or push.

- [x] **7. Activate production seed admission and the real worker consumer.**
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

  **Checkpoint — real worker consumer and version-local rejection:** the
  existing fixed pool can now call the HCQ seed consumer using its private
  `Context` and `FunctionBuilderContext`. The consumer discovers resident LCQ
  inputs, reserves/finalizes the candidate and invokes the real guarded compiler
  publication path. Its shared captures contain immutable target policy and
  process memory, never the `JitProcess` or worker-pool owner.

  Backend capacity/cancellation remain transient; implementation failures reach
  the existing worker failure protocol. Optimizer limits reject only the exact
  running seed after revalidating all input versions and instruction claims
  under JIT state. Releasing that work preserves its rejected token without
  retaining compiler pins. A stale compilation cannot reject current work;
  replacement reachability can be optimized again. Reshape input is an explicit
  unsupported-consumer failure, never discarded as successful compilation.

  A process-owned real-worker test uses captured native sampling data to enqueue
  a seed, observes real HCQ publication, executes it and closes/joins the pool
  without an ownership cycle. Two additional tests cover persistent rejection
  and stale claim/input/admission changes. This checkpoint deliberately uses
  explicit enqueueing: automatic threshold admission and production startup
  are still pending, along with their zero-worker/parallel-execution tests.
  Step 7 remains open; ordinary `JitProcess::new` is still dormant.

  **Validation:** all 795 host JIT library tests passed with the local fork
  override, including the three new consumer/rejection tests. The real-worker
  test also passed separately after its final test-only cleanup. JIT Clippy
  (`--all-targets --no-deps -- -D warnings`), formatting and whitespace checks
  passed. No homebrews, cross-target run, fork edits or dependency-pin update;
  the existing portable lockfile change for `log` is preserved. No commit/push.

  **Step 7 closed — automatic startup and native sampling admission:** normal
  process construction selects the fixed worker count, creates the real HCQ
  consumer and starts its pool before exposing the process. The memory observer
  is bound last; failure closes/joins the new pool without replacing an existing
  binding. Zero workers create neither target compiler nor queue. The same
  process-owned stop/join protocol handles explicit and last-owner teardown.

  LCQ exit, linked-transfer and successful cold-completion samples now carry
  their verified scalar identities out of the lookup mutex and attempt bounded
  seed admission at the existing threshold. Contention and pressure use the
  existing deferral/backoff, with no new hot-path checks or waits for HCQ.
  Boundary heat remains observational: no reshape jobs enter the seed consumer.

  Native tests observe no promotion below threshold, then automatic publication
  and execution without explicit enqueueing. Two independent seeds occupy two
  private worker contexts while a cold LCQ miss completes; both later promote,
  retrying publication races only through further samples. Other tests exercise
  the zero-worker policy, failed binding cleanup, real-pool shutdown and all
  three sampling sites' exact-threshold admission after releasing JIT state.
  Baseline-specific poll tests explicitly select zero workers so their fragment
  assertions do not race a valid promotion.

  README and backend integration status now describe active HCQ seed promotion.
  No homebrew/performance result or native Arm validation is implied. Step 8
  remains open for consolidated evidence, target execution and the Task 7 handoff.

  **Validation:** all 800 host JIT library tests passed with the local fork
  override. The parallel-seed test passed five additional consecutive runs.
  JIT Clippy (`--all-targets --no-deps -- -D warnings`), formatting and whitespace
  checks passed. No homebrews, QEMU/native Arm execution, fork edits or pin
  updates; the existing portable lockfile change for `log` is preserved. No
  commit or push.

  **Post-activation correction — optimized-away boundaries:** reproduced the
  reported `Nixe checkpoint requires an exit ID and cost in 0..=2048` failure
  with MOVZ/CBZ selecting one of two exits. The cost was valid; optimization
  removed the unreachable exit but left its cost entry behind. The local fork
  now validates input costs before optimization and removes them with dead
  exits. HCQ stages surviving IR boundaries only, including faults, cycle polls
  and FP activation sources; missing machine maps for live boundaries remain
  errors. Independent public entries remain reachable even when another entry
  folds its branch away. No optimizer was disabled, region limit raised or
  compiler failure reclassified as a harmless rejection.

  The CLI now returns process-removal errors alongside the original execution
  diagnostic instead of panicking on an assumed scheduler-lease invariant.
  Failed removal falls back to coordinator-owned Drop cleanup; it does not
  claim a successful explicit teardown report.

  **Validation:** all 806 host JIT library tests and all 30 CLI tests passed,
  plus 23 fork boundary tests (`arm64,disas`, both encoders). Regressions cover
  constant branches, dead fault/poll/FP sources, preserved public entries,
  missing live maps, malformed costs and native execution/accounting. JIT/CLI
  Clippy, formatting and whitespace checks passed. No homebrews or native Arm
  execution; the local fork override is still required. No dependency-pin
  update, commit or push; the portable lockfile retains only its prior `log`
  dependency change.

  **Post-activation correction — distinct memory observations:** reproduced
  `Nixe fault span requires a trapping memory operation` using two guest loads
  from the same address. Alias analysis eliminated the second load while its
  live delimiters remained. Unlike a dead CFG arm, that guest access still
  executes and owns its own memory observation and prefault state. The local
  fork now gives fault-span delimiters compiler-only memory-barrier semantics,
  preventing cross-span load CSE, store forwarding and store elimination.
  No hardware fence is emitted and no empty live fault span is silently
  accepted. Non-guest memory optimization remains enabled outside delimiters.
  Regression tests cover repeated loads/stores on both targets, ordinary
  forwarding outside spans, emitted fault IDs/extents, absence of hardware
  fences and native memory effects/accounting across an internal region edge.

  **Validation:** all 808 host JIT library tests and 44 fork Nixe tests passed
  (`arm64,disas`, both encoders), along with JIT Clippy, formatting and whitespace
  checks. No homebrews or native Arm run. The fix is in the local fork, so the
  existing override remains required. No pin update, commit or push; the
  lockfile keeps only its previous portable `log` dependency addition.

  **Post-activation correction — family invalidation lookup cost:** the
  maintainer's es2gears capture `dump/perf-20260921-192001-SHw1aR` attributed
  99.27% of sampled user cycles to memory-mutation coordination. Sampled
  instruction addresses resolve to the nested baseline-ID search in
  `invalidate_memory`, not background compilation. Family propagation scanned
  all unit records for each baseline while holding JIT state. It now resolves
  both baselines and affected family units through their existing generational
  handles: O(1) per lookup, O(total family pins) for this propagation pass,
  instead of multiplying that work by the unit-registry size. No additional
  index, ownership state, admission-policy change or disabled HCQ path is added.
  Other mapping-range scans are unchanged. Tests preserve exact invalidation
  through a non-entry baseline word/page, unrelated families, repeated requests
  and reused generations in a sparse registry. Follow-up profiling below
  separates this cost from a subsequently reproduced zero-progress stall.

  **Validation:** all 810 host JIT library tests passed with the local override,
  including the two new exact-pin/generation regressions. JIT Clippy, formatting
  and whitespace checks passed. No homebrew run, profiling-binary rebuild, fork
  edit, pin update, commit or push; the portable lockfile retains its prior
  `log` dependency addition only.

  **Post-activation correction — abandoned memory-stop progress:** explicitly
  authorized es2gears captures reproduced a scheduler loop, not sustained HCQ
  compilation. `dump/perf-followup-20260921-193513-hZhFko` shows scheduler/channel
  activity; temporary diagnostics in `dump/perf-followup-20260921-193714-QR3Bzl`
  observe repeated `Safepoint`, zero progress and unchanged PC `0x713d1878`.
  A deterministic regression reproduces a matching coordination defect: the
  last memory hold could drop while another transition owned the stop, leaving
  MappingChange unacknowledged if that owner yielded to the memory work and
  abandoned its transition. The memory authority now acknowledges its own
  drained reason under the state lock even with a separate transition owner.
  It never acknowledges unrelated work or reopens while holds remain. Execution
  also resumes a fully acknowledged but abandoned Closed stop; older batches
  preserve newer completion sequences. No HCQ/admission-policy workaround or
  diagnostic counter remains. All 813 host JIT library tests, JIT Clippy,
  formatting and whitespace checks passed, including three new regressions.

  **Authorized live validation:** after removing temporary instrumentation and
  rebuilding with the local fork, es2gears ran for 90 seconds (30-second warmup,
  60-second perf capture) and stopped cleanly on SIGINT. Results:
  `dump/perf-followup-20260921-194400-dalI2C`. It completed 28,180 SVC calls
  without reproducing the zero-progress scheduler storm. Memory-mutation begin
  accounts for 4.15% of this capture's sampled cycles versus 99.27% in the
  maintainer's original stalled capture; these are profile shares, not an FPS
  ratio. No FPS measurement or guarantee against every concurrency interleaving
  is implied. No fork edit, dependency-pin change, commit or push was needed.

- [x] **8. Validate Task 6 and hand off parallel trimming/reshape to Task 7.**
  **Production finding resolved:** initial HCQ validity no longer depends on
  unrelated maintenance epochs or global memory cursors. Seed jobs and exclusive
  instruction claims retain their exact tokens across Closing/Closed. Real
  source replacement, invalidation, ownership loss, pressure and shutdown still
  cancel work. Cold executable-image checks validate exact memory-owner,
  mapping and content stamps. Final publication validates every captured LCQ
  identity/lifecycle/ReachabilityVersion and claim under current Open authority;
  coordinated memory changes invalidate those sources before becoming visible.
  LCQ's speculative epoch/cursor guard is unchanged.

  Completed native output waits for Open without a guest lease/execution epoch.
  It helps existing link maintenance or sleeps on the coordinator, rechecking
  readiness under the condvar mutex to avoid missed notifications. Shutdown
  wakes and cancels it. A concurrent directory publication or capacity change
  refreshes only cold metadata, retaining the same native span and CodeVersion;
  it does not rerun the backend. No promotion threshold or worker-count change.

  Regression coverage includes reservation survival/deduplication, exact stale
  cleanup, changed seed and nonentry sources, lost claims, real coordinated
  memory mutation after staging, unrelated mutation at all three image checks,
  concurrent same-segment publications, Closed preparation, wait/reopen and
  shutdown. Both executable-memory implementations preserve valid images across
  unrelated changes and reject changed content, owner or mapping permissions.

  **Consolidated evidence:** `hcq/tests` and discovery tests cover deterministic
  overlap/canonicalization, demanded-only successors and the 2048-word ceiling;
  candidate/freeze tests cover exclusive membership and real entry selection;
  flow/SSA/compiler tests cover live-ins, both encoders, native entry labels,
  internal SSA, backedge polls, flags/FP and precise memory/observation exits.
  Publication/lifecycle and engine background tests cover actual mixed-tier
  execution, cutover, retained LCQ, invalidation, capacity and shutdown. LCQ and
  HCQ use the shared frontend semantics, with no production placeholder consumer.

  **Validation completed with the local fork override:** 816 host JIT library
  tests; all 107 CPU tests (90 unit, 14 concurrent-memory, 3 dependency-boundary);
  25 CLI tests. QEMU 11.1.1/AArch64: 123 HCQ tests, 72 lifetime/background tests
  and the exact-image regression. JIT/CPU Clippy (`--all-targets --no-deps --
  -D warnings`), formatting and whitespace checks pass. Native Arm hardware
  cache-coherence/memory-ordering validation remains Task 10, not a QEMU claim.

  **Authorized performance verification:** es2gears ran for 180 seconds, with
  perf in seconds 120–180, and shut down cleanly. There were 473 finished
  emissions and 471 publications, versus 6,256/387 in the earlier 180-second
  diagnostic run. Every emitted seed was distinct; the former 811-attempt seed
  emitted once. Two uncommitted emissions are not attributed by this diagnostic.
  New seeds still promoted late; this does not promise zero future promotions.
  HCQ workers accounted for approximately 0.01% of sampled cycles in the final
  minute versus 11.79% previously. These are instrumented profile shares, not
  an FPS measurement or guarantee. Evidence and caveats are in the local
  `dump/perf-followup-20260921-204144-YzZsIt/analysis.md`. Temporary diagnostics
  were removed and the normal profiling binary rebuilt. The later pre-sleep
  readiness refinement has host/QEMU tests, not another profile here.

  **Task 7 handoff:** extend `Graph::discover` and exact candidate batch claims
  with successor collision trimming; extend `Frozen` entry/dependency selection
  and the existing publication/cutover ownership transaction for zero/one/two
  family reshape and negative boundary results. Its provisional epoch-bound
  family reservations remain dormant until the real consumer is implemented.
  Small regions and large exit adapters were analyzed: inspected boundaries
  fit calls/returns, absent demanded successors and foreign HCQ ownership, not
  a demonstrated general discovery failure. Reshape belongs to Task 7;
  speculative adapter compression is not a hidden Task 6 blocker.

  **Closed:** Task 6 production compilation and all eight steps are complete.
  The local Cranelift override remains necessary. No additional fork edits,
  dependency-pin update, commit or push in this closure; the pre-existing local
  Cargo.lock changes are preserved. Homebrews remain maintainer-run unless
  explicitly authorized, as this profiling verification was.

  **Requested shutdown diagnostic:** debug emits one `JIT shutdown units` and
  one `JIT shutdown cache` snapshot before terminal reclamation. Counts are
  resident units, not historical family IDs or public-entry counts; HCQ Published
  is separated from retired-but-retained storage. Native-span bytes are reported
  by tier; committed segments, charged metadata and total cache bytes are separate
  (not RSS or a peak). No hot-path counter or homebrew run is added.
  Validation: all 330 lifetime tests pass, including empty/populated shutdown,
  multiple HCQ entry labels, exact cache accounting, repeated teardown and
  retired HCQ retained by a compiler snapshot. Logging is checked outside both
  JIT and cache locks. JIT Clippy, formatting and whitespace checks pass.

## Post-closure memory audit

The maintainer authorized es2gears profiling and corrections after observing
about 493 MiB of cache. The 180-second reference reproduced 495.8 MiB with
30,797 LCQ and 469 HCQ resident units. The principal waste was not overlapping
LCQ: 157,954 instruction instances represented 153,088 distinct InstructionKeys
(3.08% extra instances). Almost all LCQ segments exhausted their 4,096 island
slots with only 2.6–3.2 MiB used in each committed 16 MiB segment.

Corrected in the existing production path:

- Coallocate exact aligned island reservations with the owning code span; remove
  the fixed bitmap pool. Preserve checked code/island bounds, ownership, W^X,
  same-segment branch reach and coalescing/decommit behavior.
- Release backend-only labels/maps/relocations after semantic-map validation;
  keep all runtime fault/state/dependency contracts and ABI/frame extent.
- Store register membership as bit masks rather than boolean arrays; StateSet
  is 20 instead of 69 bytes, with direct bitwise union/difference.
- Scope writable-alias permission windows to the owner's host pages. Dense
  segments otherwise expose excessive per-install/per-patch mprotect work.
- Insert into immutable native-PC tables by binary search and neighbor checks,
  using one exact-capacity allocation. This removes duplicate copying/full
  sorting, not the O(n) copy required by the current immutable-table design.

The first three corrections reduced the comparable 180-second cache snapshot
from 495.8 to 250.5 MiB, with essentially identical LCQ coverage. An exposed
startup slowdown was then corrected by the permission/directory changes:
matched 45-second startup captures reached 1,024 guest completions in 20.67
versus 7.55 seconds. This is neither first-frame timing nor FPS.

Temporary emission sizing found 6.94 MB of LCQ backend body, 40.95 MB of exit
support and 1.37 MB of entry support across 30,858 emissions, including
unpublished attempts. Adapter storage remains a real separate cost; this audit
does not introduce a speculative shared-stub architecture or attribute all
expansion to guest-instruction lowering. Detailed local evidence is in
`dump/perf-followup-20260921-232045-Cv0crV/analysis.md`.

Validation: 822 host JIT tests; AArch64/QEMU 11.1.1 storage 27, lifetime 332 and
HCQ 123 tests; JIT Clippy and formatting/whitespace checks. Regression coverage
includes >4,096 colocated island slots, exact extent/padding/last-segment bounds,
span reuse and failure cleanup, page-scoped W^X aliases, released metadata
charges with surviving semantic fault maps, ordered directory insertion and
all architectural register/flag bits. Temporary diagnostic code is removed.
No fork, dependency pin, lockfile, commit or push changes.

Final ordinary-build comparison (180 seconds, perf at 120–180): **495.83 to
240.84 MiB**, saving 254.99 MiB/51.43%. Executable backing is 304 to 80 MiB;
metadata 191.83 to 160.84 MiB. LCQ residents are 30,797 versus 30,793 and HCQ
469 versus 481, with essentially unchanged native-unit byte totals. Both runs
shut down cleanly and have no lost perf samples. First 1,024 completions were
10.65 versus 6.86 seconds (an indicator, not an FPS/first-frame guarantee).
Final evidence: `dump/perf-followup-20260921-233314-VJWojq/analysis.md`.
