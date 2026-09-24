# Task 7 implementation plan

Status: steps 1–8 implemented and validated. The authorized runtime audit exposed
positive reshape oscillation and delayed predecessor collection below cache
pressure; both are corrected and the repeated captures below verify the fixes.
Native Arm hardware validation remains under Task 10.

This is a working checklist for
[Task 7](spec.md#task-7-add-parallel-ownership-and-versioned-reshape), not another
specification. Follow [CONTRIBUTING.md](../../../CONTRIBUTING.md). Update each
step in place with decisions, remaining work and validation; do not create a
separate changelog or testing framework. Agree architectural changes with the
maintainer and update the affected spec before implementing them.

## Scope and sequencing

Allow the initial HCQ partition to grow, shrink, merge and repartition through
versioned replacement. Extend existing exact instruction reservations to trim
successor collisions, and enable production reshape admission only after its
real compiler and replacement consumer are ready.

Task 6 already supplies parallel workers, immutable LCQ snapshots, deterministic
discovery, instruction claims, multi-entry compilation and initial publication.
Reuse them. LCQ remains the synchronous demand path and retained baseline;
reshape compiles from those baseline images, not by decoding generated HCQ code.
Keep the 2048-instruction cap and existing native direct-link ABI. Replacement
must not introduce a Rust transition or a generation check on every native link.

Task 7 includes correctness and lifecycle tests for everything it changes.
Task 8's broader race audit is not a reason to defer replacement safety. Disk
caching, further metadata compression and separating lazy-flag recipes are not
part of this task.

Tests accompany each step. Homebrew execution and manual performance checks
belong to the maintainer; do not launch homebrews unless explicitly requested.
Do not run them automatically at checkpoints or at final validation.

## Starting points

Paths below are relative to `crates/cpu-jit/src/`.

- `sampling.rs` and `lifetime/unit/sampling{,/boundary}.rs`: versioned boundary
  observations and four-sample threshold. Tables belong to their vCPU; workers
  must not mutate them directly. Validated negative outcomes are process-owned
  and consulted during cold admission.
- `lifetime/unit/reshape.rs`: admission for zero, one or two participating
  families, endpoint pins and exact reservation cleanup. Jobs use exact version
  and token validity rather than admission-epoch identity.
- `lifetime/background/work.rs`: incremental retained-LCQ acquisition and
  participant-aware eligibility. Strong references protect storage, not validity.
- `lifetime/background/work/candidate{,/freeze}.rs`: exact batch claims and
  successor public-entry selection accept seeds and reshapes. Frozen candidates
  capture predecessor LCQ fallbacks and identify unchanged partitions;
  replacement publication consumes those identities.
  Successor collisions trim through `hcq/trim.rs`; root/mandatory-endpoint
  collisions and a second race defer without partial claims or retry spins.
- `hcq{,/discovery,/flow,/ssa}.rs`: deterministic canonical graph and shared
  analysis. Extend discovery rather than introduce a second region planner.
- `hcq/worker.rs` and `hcq/compiler{,/backend,/publication}.rs`: real seed/reshape
  consumer, compilation, captured-image validation and publication retries that
  reuse staged code. Production reshape admission uses the cold boundary sampler.
- `lifetime/unit{,/ownership,/links,/patch,/reclaim}.rs`: family membership,
  retained baselines, incoming roots, maintenance cutover and reclamation.
- `engine/background.rs` and the invocation/sampling paths: existing fixed
  worker lifecycle and production admission. Do not add another pool.

Continue using the local Cranelift override with
`--offline --config /tmp/nixe-observable-fp-local.toml`. If a backend change is
necessary, inspect `/home/pladaria/projects/wasmtime`, branch `nixe`, first.
Preserve existing edits and portable lockfile state. No commit, push or
dependency-pin update unless requested.

## Steps

- [x] **1. Define the replacement transaction against the current foundation.**

  Trace one boundary from observation through admission, source acquisition,
  ownership claims, publication and maintenance. Distinguish the logical source
  block, its actual branch InstructionKey and the target; an interior source PC
  need not have its own dispatch slot. Deduplicate endpoints belonging to the
  same family, yielding zero, one or two participants.

  Specify the exact captured identities and validation points for each case:
  endpoint reachability, participant family/version, LCQ sources, instruction
  claims and code dependencies. Define how reshape survives unrelated maintenance
  without accepting invalidated participants, replacing the provisional
  epoch-bound behavior before connecting production.

  Define where bounded negative results live, how the admitting vCPU observes
  them without worker writes to its sample table, and which dependency/version
  changes permit retry. Table eviction must not silently defeat the specified
  suppression guarantee. Define a useful connected candidate containing both
  mandatory endpoints, including how an unchanged partition is handled.

  **Exit:** decisions are recorded here and any changed contract is reflected in
  the spec. The selected entries, dropped entries, ownership transfer and cleanup
  responsibilities are explicit; subsequent steps need no competing publication
  protocol or unspecified cancellation policy.

  **Review:** traced sampling, `ReshapeJob`, worker input/claim validation,
  `Frozen::prepare`, `PreparedUnit::publish_once` and `unlink_next`. The contract
  below is the implementation handoff. The maintainer approved version-valid
  reshape surviving unrelated maintenance and shared negative-result storage
  instead of relying on the evictable per-vCPU BoundaryTable. The spec now
  reflects both contracts. Prefer indexed operations and avoid unjustified
  metadata duplication; the complexity and storage requirements below apply.

  **Endpoint and candidate identity.** Keep `source_block` as the dispatch key
  supplying source reachability; the source InstructionKey identifies the actual
  executed terminal and need not have a dispatch slot. The root is `source_block`,
  not the predecessor family's first entry. Capture both endpoint slot handles,
  preferred unit identities, reachabilities and optional family/version pairs.
  Deduplicate participants and reserve them in family-ID order with one job token;
  zero participants use the existing source-slot reshape owner. This is separate
  from the seed rejection cell. Participant reservations do not replace exact
  candidate InstructionKey claims before analysis/backend work.

  Discovery uses only retained LCQ images. Both endpoint instructions must be in
  the source-root-connected graph through eligible CFG/observed non-call edges;
  queue priority alone does not prove connectivity. Observed indirect edges do
  not authorize converting an unchecked indirect branch to an unconditional SSA
  edge. A candidate is useful if it forms an initial region, merges distinct
  families, changes membership, or changes required public entries. An unchanged
  single-family membership/entry set is a no-op negative, not another compilation.
  Missing/unpublished input, a racing claim or changing eligibility is deferred
  rather than recorded as a stable disconnected result. The mandatory target
  is exported when it is the demanded interior entry that triggered reshape;
  coverage alone does not export other instruction PCs.

  **Approved maintenance contract.** Remove the provisional admission
  epoch from reshape validity and reservation-owner identity. Keep exact
  endpoint/family/source versions and token checks, as for initial HCQ. Check
  them at dequeue, input acquisition, candidate reservation/freeze, and final
  publication; validate captured instruction content through the existing memory
  authority outside JIT state. Invalidation, supersession, eviction or shutdown
  cancels an affected job even if its references still retain the storage.
  Unrelated Closing/Closed intervals do not cancel it. Final publication requires
  current Open authority and uses the existing condition-variable/maintenance
  service, without a guest lease or execution epoch held while waiting. Refresh
  stale directory/capacity preparation, not backend output. Replace epoch-specific
  reshape tests with affected-input cancellation and unrelated-stop survival tests.

  **Approved negative-result contract.** Keep per-vCPU BoundaryTable
  sizes and heat semantics unchanged. Put authoritative negative results in one
  process-owned, cache-accounted index under JIT state. Admission queries it by
  full boundary identity plus logical source key using the existing nonblocking
  cold lookup. Workers publish results; they never write vCPU sample tables.
  Sample sequence numbers are observation order, not negative-result identity.

  Reserve charged result capacity on the worker before discovery, outside state;
  if unavailable, defer before rediscovering the graph. A valid negative is not
  evicted merely to admit another boundary. Store reason and weak generational
  source/dependency evidence, not a graph, emitted code or strong code pins.
  Index its associations to affected endpoints, participants and inspected LCQ
  inputs so their invalidation/retirement removes the exact result without a
  whole-table scan. Relevant code/mapping changes invalidate that evidence through
  the same source lifecycle. The capture cursor is provenance, not an equality
  check against every unrelated global memory event. History loss invalidates
  all affected evidence. Check the evidence again under state when installing
  the result; a stale worker cannot suppress a newer generation.

  This preserves suppression across sample-table eviction and across vCPUs while
  bounding storage through the existing cache budget. Under pressure, normal
  owner/source retirement releases associated results; do not retain code to keep
  a negative alive. Temporary contention, pressure and cancellation are never
  structural negatives. A typed backend rejection is scoped to the validated
  reshape input, not the family's general reservation or an unrelated seed.

  **Complexity and memory budget.** Boundary lookup and exact-result removal
  use expected O(1) hash lookup; removal of each reverse association is O(1)
  through a stable handle/backlink, not a search through its owner's list.
  Invalidating a result with d associations costs O(d); invalidating an owner
  visits only its affected results and their associations. Do not describe
  bulk invalidation as O(1), or scan all boundaries, units or vCPU tables for
  a point operation. Candidate validation still necessarily visits its inputs.
  No added lookup/check belongs on the native direct-link path.

  Deduplicate evidence for the same source/owner; reuse existing generational
  identities and invalidation events. Use a shared index and compact associations,
  not a hash table per unit or a copy of the candidate's instruction/state maps.
  Charge actual allocated capacity, record headers and associations to the cache.
  Reuse released slots; do not preallocate a worst-case dependency array per
  boundary or worker. Any capacity preparation before discovery reserves result
  storage, not an unbounded estimate of a future graph. If additional evidence
  cannot be stored, defer without backend compilation or publishing a negative
  that cannot be invalidated correctly. Step 4 must verify record sizes, charging,
  reuse and cleanup using focused tests, without adding permanent hot counters.

  **Publication and retirement handoff.** Extend the current `PreparedUnit`, not
  a parallel publisher. Before mutation, prepare the successor, exact participant
  set, selected entry payloads, all predecessor public-entry fallbacks, baseline
  pins and incoming-root work. Selected entries preserve their LCQ owner and
  acquire the successor HCQ owner. Every predecessor public entry not exported
  by the successor loses its HCQ owner and retains its LCQ payload, even if its
  instruction remains covered internally. Other covered instructions acquire no
  new dispatch slot. Dropped membership becomes unowned; do not retire its LCQ.

  Under current Open authority and one state transaction, revalidate identities,
  register new metadata and cutover work, transfer selected membership and remove
  dropped membership, publish coherent per-key payloads and mark whole predecessor
  units Superseded with TierCutover retirement. Predecessor records remain callable
  retirement owners, not active discovery owners. Reuse indexed incoming static
  roots and PIC backlinks; outgoing-link preparation must overlay both successor
  payloads and dropped-entry LCQ fallbacks, never attach a new root to a predecessor
  that this transaction is superseding. Do not expose a partially prepared unit.

  Closed maintenance restores/retargets machine-code roots before detaching old
  code. Keep existing code/fault/dependency records, compiler pins and execution
  epoch protection until their normal quiescence conditions hold. New baseline
  pins are acquired before predecessor pins can be released. Successive reshapes
  must not keep uncut predecessor generations accumulating: finish outstanding
  safety cutover before admitting publication of the next generation.

  `unlink_next` currently asserts that removing every old instruction membership
  succeeds. Replace that assumption specifically for transferred/superseded
  ownership: exact old-owner cleanup must leave successor membership untouched,
  while ordinary retirement still detects inconsistent ownership. Likewise clear
  dispatch and link roots only when they still name the old unit/version. Existing
  compare-and-clear job cleanup must never release a newer reservation. Capacity
  failure or stale validation before publication leaves both old families intact.

  **Validation handoff:** steps 2–4 test claims, identities, negative lifetime and
  no-op suppression; step 5 tests mixed-version execution, dropped entries,
  transfer-aware unlink and real reclamation; steps 6–7 test consumer outcomes and
  actual sampling. This checkpoint changes documentation and the approved spec
  contract only; no homebrew or runtime test was run.

- [x] **2. Trim successor ownership collisions before backend work.**

  Extend normal seed reservation beyond whole-candidate deferral. A collision at
  the seed still cancels/defers; a successor collision cuts the eligible prefix
  before the claimed instruction and retains only the root-connected unclaimed
  graph. Recompute canonical blocks, leaders and retained inputs after trimming;
  do not leave edges to removed instructions as internal SSA edges.

  Perform graph rebuilding outside the shared state lock. Revalidate and acquire
  the resulting exact instruction batch under the existing short transaction,
  before liveness/backend work. A further race may defer; do not spin while
  holding state or serialize unrelated discovery/compilation. Cleanup removes
  only claims still carrying this job's token.

  **Exit:** deterministic tests cover seed and interior-successor collisions,
  disconnected tails, and a second race during reservation. Independent workers
  enter backend compilation concurrently, but no unrelated workers enter it
  owning the same InstructionKey. Existing uncontended discovery is unchanged.

  **Implemented:** capture the collision mask with indexed ownership/claim
  lookups under state, then trim outside the lock. Cut each canonical block at
  its first occupied instruction, traverse the root-connected remainder and
  rebuild external/internal targets with the existing canonical builder. Retain
  indirect sampled successors only if their actual seed terminal stays reachable.
  Drop unused input snapshots and shorten remaining captured extents; a successor
  promotion's changed payload does not invalidate an input already removed by
  trimming. Remaining inputs are revalidated before the all-or-none claim batch.

  No new scan of the process cache or persistent metadata was introduced. Work
  visits the bounded candidate graph; instruction ownership queries are expected
  O(1), and ordered input-range queries avoid a words-by-inputs nested scan.
  Rebuilding and releasing unused snapshots happen outside JIT state. A second
  collision or lost capacity defers without retrimming, and token-specific cleanup
  preserves other workers' claims. The uncontended path does not rebuild its graph.

  **Validation:** 836 x86-64 JIT library tests pass, including real publication
  and native execution across the trimmed boundary, and independent compiler
  contexts reaching/completing backend work with disjoint live claims. The 28
  candidate/freeze tests and 125 HCQ tests also pass on AArch64/QEMU. Clippy for
  the JIT library/tests with warnings denied, formatting and whitespace checks
  pass. Homebrews were not
  run; no Wasmtime changes were needed. QEMU is not native Arm hardware evidence.

- [x] **3. Discover and freeze reshape candidates from retained LCQ inputs.**

  Extend the existing graph builder for reshape observations, using the spec's
  mandatory-endpoint ordering and immutable successor observations. Permit
  membership in participating families plus eligible unowned instructions;
  foreign families and unrelated in-flight claims remain boundaries. Reserve
  selected instructions using the same exact-token machinery as seed jobs.

  Preserve deduplication, canonical splitting, execution-key checks, termination
  rules and the 2048-distinct-instruction cap. Calls remain external. Both
  boundary endpoints must be represented in a useful connected candidate;
  adjacency alone does not authorize absorbing a third family.

  Extend the final entry freeze to account for external incoming roots and
  demanded interior entries. Identify selected replacement entries and old
  entries that must return to LCQ. Do not expose every covered instruction as a
  public entry or duplicate bodies/metadata per entry. Revalidate participating
  families and retained inputs before expensive analysis.

  **Exit:** tests cover zero/one/two families, a same-family interior demand,
  growth, shrinkage, merge, changed partition, foreign ownership and late entry
  changes. Each accepted graph contains both endpoints, obeys the cap and has
  exactly the required native entries. No production reshape admission yet.

  **Discovery checkpoint:** the existing worklist now accepts ReshapeSnapshot,
  roots it at the captured logical source and prioritizes mandatory endpoints
  before ordinary successors. Sources still come from indexed retained-LCQ
  lookups; no demand slot is created for an interior source instruction. Discovery
  checks both endpoint versions, rejects invalid transfer kinds, and prunes to
  root-connected instructions with the shared trimming traversal. An indirect
  observation belongs to its actual source InstructionKey, not automatically to
  the root's terminal. If every block survives, reuse the graph without rebuilding
  or decoding it again. Seed inclusion priorities retain their relative order.

  Six focused regressions cover zero/one/two participants with a third-family
  boundary, retained same-family interior demand, calls/returns/runtime stops,
  disconnected mandatory sources, actual-source indirect attribution and the
  shared instruction ceiling. Structural failures currently defer; persistent
  negative outcomes remain step 4 and no production consumer admits reshape yet.

  **Claims checkpoint:** reshapes now use the existing exact InstructionKey
  batch index and their family/source reservation token. Participant-owned
  instructions are eligible; foreign owners and any other candidate's claims
  remain exclusive. Trim optional successors before claiming, but defer the whole
  candidate if trimming loses either mandatory endpoint. Revalidate captured LCQ
  identities, both endpoint versions and participant ownership under the final
  claim lock. Cleanup removes only matching tokens. This adds no work to native
  links, no second claim index and no per-instruction allocation.

  Five additional regressions cover seed/reshape exclusion, overlapping
  zero-family reshapes with no partial claims, optional-successor trimming,
  participant retirement and stale-job cleanup preserving a newer owner token.
  The zero/one/two-family and same-family discovery tests now also reserve and
  validate their candidates. Step 6 replaces the provisional epoch-based tests
  with maintenance survival and exact stale-participant cleanup coverage.

  **Entry-selection checkpoint:** the shared final sweep now freezes reshape
  successor entries before analysis: logical root, mandatory demanded target and
  included instructions with external static/PIC roots. Other demanded leaders
  remain internal; neither coverage nor an old HCQ public label automatically
  exports an entry. PIC lookup checks both indexed LCQ/HCQ owners because an
  optimized target's incoming roots belong to its HCQ unit, and still filters
  by exact target key. No global reader/unit scan is added.

  Eight regressions cover zero/one/two participants, same-family interior demand,
  external static roots, indirect/return roots on either tier, late uncaptured
  entry cancellation, immutable selection after a later PIC insertion, participant
  retirement before analysis and refusal of initial publication for reshape
  (including zero participants). A late external entry cancels before lowering
  if it lacks a captured canonical input. Once frozen, later roots do not create
  new labels; the replacement cutover must restore non-exported entries to LCQ.

  **Predecessor/no-op checkpoint:** frozen candidates retain at most the two
  deduplicated predecessor snapshots and capture each non-exported old public
  entry's exact slot, prior HCQ owner, reachability and LCQ snapshot. This includes
  covered-but-internal labels as well as entries outside the new graph. Selected
  labels reuse their graph input evidence. Capture visits only predecessor entry
  sets with indexed lookups; vector capacity is prepared outside state and no
  instruction/state-map bodies are copied. Snapshot cleanup, including a partial
  capture failure, happens outside the state guard.

  Every frozen validation now checks the fallback evidence as well as graph
  inputs/participants, so storage retained by a snapshot cannot make a retired
  baseline valid. No live membership, payload or link changes during freeze.
  Single-family candidates with identical membership and public-entry sets are
  recognized regardless of root/entry ordering and stop before analysis/backend
  work. Zero-family candidates, merges, changed membership and entry-only changes
  remain useful. Persistent no-op suppression belongs to step 4; until that
  consumer exists, analysis returns Deferred for an unchanged frozen partition.

  Seven additional regressions cover covered/dropped fallback labels, two-family
  merge fallback sets, root-independent no-op detection, same-size membership
  changes, stale/unavailable fallback owners outside the graph, partial-capture
  cleanup and entry-only shrinkage. Earlier cases cover growth and entry-only
  expansion; late PIC insertion cannot alter the frozen fallback set.

  **Validation:** 862 x86-64 JIT library tests pass. The 35 reshape tests and
  62 background-work tests also pass on AArch64/QEMU, covering shared claims,
  trimming, reshape freeze/no-op handling and the existing seed publication path.
  JIT Clippy with warnings denied, formatting and whitespace checks pass.
  No homebrew run or Wasmtime
  change was needed.

  **Handoff:** step 4 installs persistent negatives from validated evidence;
  step 5 consumes frozen predecessor/fallback identities in coordinated
  replacement and incoming-root cutover. Steps 5 and 6 replace the seed-only
  publication/consumer guards and provisional admission-epoch rule before
  production admission in step 7.

- [x] **4. Implement versioned negative reshape outcomes.**

  Record a negative result when the required endpoints cannot form a useful
  candidate because of the cap or disconnected eligibility, or when the selected
  membership/public-entry set is unchanged. Key it by the named
  endpoint/owner versions and dependency cursor required by the spec, using the
  storage and invalidation contract settled in step 1. Validate that evidence
  before recording the result, just as for a positive publication.

  Keep existing families and their direct boundary link. Suppress rediscovery
  while the named inputs remain unchanged, including observations from another
  vCPU. Relevant endpoint reachability, participant version or code-dependency
  changes permit a new attempt. Queue contention, temporary claims, pressure and
  stale jobs are deferrals/cancellations, not permanent structural negatives.

  **Exit:** repeated samples do not rediscover an unchanged rejected boundary;
  each named invalidation cause allows retry. Tests cover cross-vCPU observation,
  stale completion, bounded storage and exact cleanup without suppressing a newer
  boundary generation or allocating on the guest sampling path.

  **Storage checkpoint:** `lifetime/unit/reshape/negative.rs` supplies the shared
  index, owned by the process's Units state and used by validated result
  installation and cold reshape admission. Production reshape remains gated
  until its real consumer is ready. Keys include
  the full boundary identity and logical source, not sample sequence. Records
  store a reason, capture cursor and deduplicated weak unit/dispatch identities;
  they retain no code, graph or baseline pins. Participant family evidence uses
  its immutable unit generation rather than a separate per-family table.

  One owner-head hash table and intrusive per-record backlinks provide expected
  O(1) boundary lookup and O(1) removal of each reverse association. Owner
  invalidation visits only affected records and their evidence. Registry slots
  are reusable and generation-checked. Hash/registry growth is prepared and
  charged outside state, preserves handles, and rejects stale capacity plans
  without mutation. Exact record headers, evidence arrays and allocated index
  capacities are charged; an empty index allocates nothing. Pressure refuses
  additions rather than evicting valid negatives.

  Removed records form an allocation-free garbage chain so invalidation can
  detach under state and release storage/cache charges after unlocking. Chain
  destruction is iterative. Tests cover key identity, duplicate handling,
  evidence deduplication, charging, head/middle/tail detach, capacity failure,
  slot reuse, stale dispatch generations, growth and large code-free batches.

  **Lifecycle checkpoint:** unit retirement/cutover invalidates associated
  evidence immediately; exact memory invalidation includes affected retained
  baselines and their HCQ families. Dispatch publication (including promotion),
  unlink and slot retirement invalidate the exact dispatch identity. These hooks
  use owner backlinks, not a whole-result scan. Unrelated maintenance does not
  clear negatives. History loss and shutdown explicitly clear all evidence with
  one table drain. Detached records drop after unlocking; shutdown also releases
  the index capacities with the other process-owned storage.

  Eight lifecycle regressions inject records through test fixtures and exercise
  the real retirement/publication/memory/shutdown paths: retained snapshots do
  not preserve validity, unrelated owners survive, baseline invalidation reaches
  family evidence, dispatch generations remain isolated, history loss clears
  evidence, and terminal cleanup returns storage. The fixtures do not stand in
  for validated production installation.

  **Worker-capacity checkpoint:** accepting a reshape job now reserves a charged
  record-header budget and a shared registry/key slot before returning Work to
  discovery. Outstanding reservations cannot be consumed by other insertions;
  their count is separate from visible negatives. Capacity checks and reservation
  release use O(1) counts, not registry/free-list scans. Index growth remains an
  amortized operation with charged storage prepared outside state; a racing
  stale capacity plan defers instead of evicting results. No evidence array or
  owner-head table is preallocated for a hypothetical graph. Cancellation releases
  the reservation, and shutdown keeps the index until all compiler owners drain.
  Seeds do not allocate or reserve negative-result storage.

  Six regressions cover slot exclusion/growth, concurrent live worker ownership,
  cancellation/reuse, header/index pressure with retry, stale jobs, seed isolation
  and shutdown. The no-op installer below now consumes that reservation and
  charges the actual evidence; other negative outcomes still need integration.

  **No-op entry evidence checkpoint:** a no-op depends on incoming roots as
  well as unchanged code. `Frozen::validate_unchanged_locked` validates the
  candidate and compares the current required entries with its frozen set under
  the installation guard. It includes newly demanded interior PCs, not only
  captured leaders. The comparison merges existing ordered lists without another
  index. Positive compilation keeps its original frozen-label contract.

  Entry-sensitive negatives use one weak `Entries(UnitHandle)` association for
  the unchanged family, not one association/table per instruction. Existing
  instruction ownership locates that family on static-source insertion/removal,
  PIC insertion/removal and dispatch publication inside its membership. These
  events conservatively invalidate its entry evidence; unrelated families and
  code-only evidence survive. A hit in an already installed private PIC way does
  not invalidate anything. Unit retirement invalidates both code and entry
  evidence. All detached records are released outside state, including private
  PIC reuse, reader destruction and rootless retirement. No native probe or
  direct-link sequence changes.

  Six regressions cover late static/PIC roots, disappearing PIC roots, a newly
  demanded interior entry, family-scoped invalidation, weak PIC reuse/hits,
  reader cleanup and baseline publication. These lifecycle fixtures inject
  records directly; the installer tests below exercise validated Work instead.

  **Validated no-op installation checkpoint:** `Frozen::prepare_unchanged`
  captures deduplicated weak identities for its input units, input dispatch slots,
  predecessor and entry-set evidence. It transfers the worker's prepaid header
  into the record, grows that same metadata lease for the actual evidence, and
  prepares any owner-index growth outside state. No second resident lease, graph,
  code copy or compiler pin is retained in the result. The slot remains reserved
  while preparation owns the header, including failure/abandonment paths.

  Final installation requires Open state and revalidates inputs, claims,
  participant identity and the current required entries under the insertion
  guard. Success consumes exactly this worker's reserved slot. Duplicates retain
  one resident result; stale completion, pressure and abandoned preparation leave
  no partial result and release their resources outside state. Prepared owners
  borrow Frozen/Work, preserving compiler protection until cleanup finishes.

  The HCQ `record_unchanged` path reuses publication's executable-memory capture
  and exact-image validation before preparation and immediately before installing.
  The bound memory mutation coordinator closes the remaining check/install race
  through the captured LCQ owners. The recorded cursor is provenance, not a
  global equality gate. This path allocates no unit identity or executable code.
  Focused installer tests use real accepted reshape jobs. Steps 6–7 connect
  this path to the production consumer and admission.

  A candidate shortened by competing claims/ownership cannot install a no-op:
  the only new members may have disappeared due to that temporary race. Preserve
  that fact once per candidate, not per claim/instruction, and cancel its negative
  outcome; a fresh discovery can retry when the competitor releases its claims.

  Nine regressions cover reserved-slot exclusion/consumption, insufficient
  evidence capacity, duplicate budget cleanup without code retention, late root
  changes, retirement/shutdown, pressure/abandonment with retry, real HCQ memory
  capture, a code mutation during final image validation, and transient trimming
  which must not suppress a later useful larger region.

  **Discovery-evidence checkpoint:** reshape discovery now captures weak
  key/reachability/unit identities for every acquired LCQ input before extent
  selection. The ledger survives connectivity/collision trimming and includes
  inputs excluded by the instruction cap; discarded snapshots are still dropped.
  Its actual vector capacity is charged on transfer from worker scratch to Graph
  and released with the graph. Seed discovery allocates no ledger.

  No-op preparation and installation revalidate this complete acquired-input
  ledger, independently of the retained compilation inputs. Persistent records
  watch its deduplicated unit/dispatch identities plus predecessor/entry evidence.
  No code pins, instruction copies, guest-path probes or whole-registry scans
  are added. Positive candidate validation retains its existing trimmed-input
  contract.

  No-op installation requires every inspected eligible prefix to fit the
  unchanged family. A foreign ownership cut now carries its blocking instruction
  and weak family-unit identity, captured under state and revalidated at install.
  This works both at a successor lookup (including an undemanded interior PC)
  and inside an overlapping LCQ image. Indexed membership lookup is O(1); there
  is no new per-instruction watch table or strong reference to the blocking code.
  The existing unit retirement/cutover association invalidates the result; a
  queued retirement already revokes the proof before ownership is detached.
  Frontier vector capacity is charged outside state and persistent associations
  are deduplicated with other unit evidence.

  Missing exterior inputs and cap truncation still cancel the negative outcome,
  rather than persisting an incomplete discovery proof. An undemanded interior
  label already covered by the graph is allowed: the family entry watch catches
  later demand. Temporary claims are not foreign-family evidence. Tests cover
  retry after missing-input demand/foreign retirement, partial-image cuts,
  foreign interior PCs without fabricated demand, late retirement, weak code
  ownership, exact capacity charging/pressure cleanup, cap-excluded inputs,
  ledger preservation through trimming and seed isolation.

  **Typed discovery outcomes:** `Graph::discover` now distinguishes interrupted
  work (deferred/cancelled/failed) from structural cap/disconnection results.
  Structural results borrow their accepted Work and carry only the charged weak
  inspection/frontier ledger, not a disconnected graph or code snapshots. Input
  generations, blocker liveness and allowed prefix ownership are revalidated on
  capture and can be checked again before consumption. Connectivity trimming
  cannot erase evidence for discarded endpoints or cap-excluded inputs.

  At the exact ceiling, reshape still inspects queued frontier identities: a
  demanded input that cannot fit differs from unavailable input. Missing-input
  discovery remains a deferral, including when the visible component is already
  disconnected. Seed discovery retains its existing ceiling shortcut. Tests
  cover deterministic cap/disconnected classification, exact-cap missing/demanded
  frontiers, discarded-input retirement, charge/refcount cleanup, pressure,
  shutdown and retry. These are discovery results, **not installed negatives**;
  registry revalidation alone is not final memory/selection proof. The seed-only
  production consumer treats an unexpected structural result as an invariant
  failure, never silently discards it or activates reshape early.

  **Selection-page checkpoint (approved):** the existing owner-head index now
  supports deduplicated address-space/guest-page associations. LCQ demand
  publication/unlink and HCQ membership publication/unlink invalidate affected
  records, including membership without a public entry. Empty dispatch
  reservations do not constitute demand. Consecutive membership words on one
  page share one lookup during the existing publication/removal walk; there is
  no second index, per-instruction watch or native-path change. Same-page changes
  may conservatively invalidate extra results, as approved. Four lifecycle tests
  cover scope, deduplication, demand and unpublished-entry ownership. These tests
  inject page associations; the structural installer below now produces them
  from its actual inspected prefixes.

  **Structural memory checkpoint:** rejected discovery results can temporarily
  pin their unique inspected LCQ images under current Work authority, including
  inputs discarded from the graph. The pin vector is charged outside state and
  released before final checks. Publication and rejection share exact word and
  dependency comparison against executable memory. Only memory capture tickets,
  not code pins, survive capture. Tests exercise real disconnected discovery,
  discarded-input mutation during and after validation, and pin-budget refund
  without executable allocation. Memory capture alone is not selection proof;
  the installer below supplies the final state check.

  **Structural installation checkpoint:** discovery captures demanded interior
  leader ordinals in one charged `Vec<u16>`, with ranges per inspected input,
  rather than another full-key list or allocation per input. Structural checks
  compare current demand and eligible ownership across every inspected prefix,
  including cap-excluded inputs, under the same guard as result insertion.
  A newly demanded leader cannot slip through unchanged endpoint versions.

  `record_structural` captures executable memory, prepares the charged record,
  rechecks memory and installs under current Open authority. Cap/disconnected
  results share reserved-header/slot/index handling with no-op results; they add
  exact endpoint/participant watches and deduplicated selection-page associations.
  The resident result retains neither the leader vector, graph nor code pins.
  Missing inputs, temporary claims, pressure and stale proof do not become stable
  negatives. Tests cover real disconnected memory capture/insertion, mutation
  after preparation, cap duplicates and refunds, late leaders, post-install
  demand/ownership changes, unrelated pages, discarded-input retirement,
  pressure and shutdown. Steps 5–7 supply replacement publication and the
  production consumer/admission using these results.

  **Backend-negative checkpoint:** typed implementation/code-size limits can
  install a boundary-scoped result through `record_backend_rejection`. Capture
  checks every inspected LCQ input, including cap-excluded images; preparation
  and insertion revalidate claims, participants, discovery selection and the
  CURRENT public-entry set. Missing inputs and transient claim trimming cancel
  the negative; a capped candidate with completely observed inputs is valid
  evidence. No-op validation still requires every eligible word in its graph.

  Backend results reuse the reserved record/header/index path. An `EntryPage`
  association in the same owner-head index observes static/PIC root changes even
  on LCQ-only pages, independently of structural selection watches. It adds no
  new index, per-instruction association or native-link check; same-page root
  changes conservatively permit retry. Existing PIC hits do not invalidate it.
  Only weak identities/pages remain resident; worker graphs/claims/pins drain
  normally. Seed rejection tokens and unrelated family reservations are untouched.

  Tests exercise a real Cranelift fixed-frame limit, final memory mutation,
  duplicate budget cleanup, zero/one-family candidates, late roots, static/PIC
  insertion/removal, unchanged PIC hits, unrelated pages, missing inputs,
  pressure/retry and cap-excluded dependency retirement. Unsupported/backend
  failures still propagate as failures, not optimization negatives. Production
  remains seed-only; the consumer routing belongs to step 6.

  **Validation:** all 935 x86-64 JIT library tests pass. On AArch64/QEMU,
  all 431 lifetime tests and eight real HCQ negative memory/publication tests pass.
  JIT Clippy with warnings denied, formatting and whitespace checks pass.
  No homebrew execution or Wasmtime change.

  **Admission checkpoint:** after validating the current logical source,
  endpoint/owner identities and Open authority, `reserve_reshape` performs one
  expected-O(1) lookup in the process-owned index. A hit returns `Suppressed`
  before token allocation, family reservation, endpoint pins or queue access.
  It leaves sample heat at four, so the next observation can retry immediately
  after relevant invalidation. Sequence numbers and per-vCPU eviction do not
  affect persistent identity. Contention/pressure keep the existing nonblocking
  deferral and score-three behavior; stale generations remain stale.

  Tests cover independent concurrent vCPU samples, real two-way sample eviction,
  fresh sequence numbers, no JIT cache growth or queued work on a hit, a full
  queue/busy family, another boundary sharing that family, pressure/contention,
  entry-root invalidation with unchanged endpoints and endpoint generation reuse.
  The real-memory no-op/disconnected/backend tests now verify repeat requests
  are suppressed before creating another Work; raw index duplicate insertion
  remains tested separately. No native link, sampling allocation or worker pool
  has been added.

  **Handoff:** step 5 supplies replacement publication; step 6 connects typed
  outcomes to the real reshape consumer; step 7 enables production observation
  admission. The lookup does not bypass those gates.

- [x] **5. Publish replacement families and cut over every incoming root.**

  Extend the current staged publication path for zero/one/two participants.
  Validate sources, participant versions and exact claims under current Open
  authority. Install the new unit and its metadata, transfer selected membership,
  register every required patch/clear, publish selected dispatch payloads and
  restore dropped entries to their retained LCQ payloads. Retire each participating
  predecessor family as a whole, not just its selected members.

  Preserve per-key publication linearization: mixed old/new entries are safe;
  there is no invented atomic whole-family payload. Predecessor machine code,
  fault/dependency records and baseline pins remain alive while incoming roots
  can still call it. Use the existing maintenance rendezvous to cut static and
  dynamic roots before epoch/reference quiescence permits reclamation.

  Capacity/directory changes may require re-preparing metadata, not regenerating
  the backend output. Failures release only this attempt's unpublished resources
  and claims, never a newer owner. Preserve compact resident images and shared
  bindings rather than retaining an expanded discovery graph after publication.

  **Exit:** tests execute old/new mixed chains and LCQ fallbacks, then demonstrate
  real span/metadata reuse after cutover and the last reader/compiler pin. Stale
  publication and cleanup cannot alter newer membership, dispatch or links.

  **Checkpoint — replacement publication core:** the existing prepared-unit
  publisher now accepts frozen zero/one/two-participant candidates. It validates
  claims, participants and fallback baselines before mutation, registers a
  TierCutover for whole predecessors, publishes selected HCQ payloads and restores
  dropped entries to LCQ. Fallback boxes are prepared/accounted outside state;
  replaced boxes are destroyed after unlocking. Shared instruction ownership is
  updated in place with expected-O(1) lookups; only new words need spare index
  capacity. Unselected predecessor words lose ownership immediately. Delayed
  unlink removes only matching old owners, never successor membership.

  Tests cover zero/one/two participants, discarded interior and exterior entries,
  retained compiler snapshots through retirement/reclamation, installed static
  and PIC roots retained before Closed and cut over afterward, unchanged index
  capacity for a merge, directory refresh without re-emission, abandoned output
  and a stale fallback baseline. Selected static roots rebind to the successor;
  dropped-entry sources rebind to LCQ through the existing Closed refresh path.

  **Native/lifecycle closure:** real LCQ/HCQ compilation exercises zero/one/two
  predecessors through static, indirect PIC and return/RSB ingress on two vCPUs.
  A shrinking replacement executes an unchanged outer HCQ family through the
  dropped entry's LCQ fallback into the new HCQ body, after the old body is freed.
  A worker publishes while another OS thread retains an announced invocation;
  its native-PC metadata remains valid and cutover waits for that reader's exit.
  Code changes after staging cancel publication, while a memory change joining
  an already-pending cutover withdraws both versions and their queued roots.

  The lifetime tests additionally borrow predecessor fault metadata after unlink:
  dropping the final compiler pin permits directory detachment, but actual span
  and registry reuse waits for the fault reader's second grace period. Reusing the
  native address attributes faults only to the new unit; stale handles cannot
  retire it or change successor membership. Exhausting both unit/family registry
  reservations alongside a changed directory refreshes metadata without replacing
  the staged CodeUnit or recompiling its bytes.

  Validation: 947 JIT library tests pass on x86-64; 438 lifetime and 24 HCQ
  publication tests pass on AArch64/QEMU. Clippy with warnings denied, formatting
  and whitespace checks pass. No homebrew was run and no Wasmtime change was
  needed.

  **Handoff:** step 6 implements maintenance survival and real worker routing;
  step 7 enables production reshape admission.

- [x] **6. Connect the real reshape worker consumer.**

  Route reshape work through discovery, exact claims, entry freeze, shared
  liveness/lowering, one optimized multi-entry body and replacement publication.
  Reuse private worker resources; no shared lock spans analysis or backend work.
  Carry negative, deferred, cancelled, rejected and failed outcomes distinctly:
  a reshape outcome must not incorrectly poison seed admission or report success
  without producing the specified publication/negative result.

  Apply the agreed maintenance policy to queued, running and staged reshape
  jobs. Test invalidation, pressure and shutdown at their consumption boundaries,
  including wakeup and release of all endpoint/participant/compiler pins.
  Do not let an unrelated stop trigger a compile-discard loop.

  **Exit:** the existing pool completes real reshape requests in tests, unrelated
  backend work remains parallel, and an overlapping family reservation admits
  only the authorized generation. Staged output survives eligible publication
  retries without recompilation. Production admission remains off until step 7.

  **Consumer/maintenance checkpoint:** the existing fixed-pool consumer now
  routes reshapes through discovery, claims, freeze, shared compilation and
  replacement publication. Structural, unchanged and typed backend-limit
  outcomes use the validated negative installers; transient deferrals and
  cancellations remain distinct from failures. Only initial seed backend limits
  reject seed admission. No second pool or backend path is added.

  Removed admission-epoch identity from reshape jobs and reservation owners.
  Exact endpoint/participant versions and reservation tokens remain authoritative
  across unrelated Closing/Closed intervals. A stale job cannot release a newer
  token, and a surviving participant stays reserved until its old job drains.
  Prepared negative results now wait for Open through the existing condition
  variable and revalidate their evidence on wakeup, like positive publication;
  neither wait holds an execution epoch or guest lease.

  Real-pool tests compile a two-family replacement, execute static/PIC/return
  ingress, persist no-op/disconnected negatives without poisoning initial
  promotion, and preserve a running reservation across unrelated maintenance.
  Lifecycle tests cover queued/running/staged survival for zero/one/two
  participants, retaining the same staged CodeUnit, plus prepared-negative
  wakeup and participant withdrawal. Tests relying on epoch rollover to steal
  an active reservation have been replaced with the current contract.

  Validation: 952 x86-64 JIT library tests pass; 440 lifetime and 27 HCQ
  publication tests pass on AArch64/QEMU. JIT Clippy with warnings denied,
  formatting and whitespace checks pass. No homebrew or Wasmtime change.

  **Pool cancellation checkpoint:** two actual workers retain independent
  reshape reservations and distinct Cranelift contexts; one compiles/publishes
  while the other remains paused with its compiler lease. Both resulting
  families execute through native indirect/return ingress. Running real-consumer
  work cancels on code mutation or shutdown and defers on cache pressure.
  Mutation/pressure retries use the same pool and execute the correct bytes,
  demonstrating that neither a permanent negative nor an old reservation leaks.

  With one worker deliberately occupied, a queued reshape invalidated by a code
  write never reaches the consumer; only its readmitted generation compiles.
  Shutdown drains queued reshape pins and cancels the running work before joining.
  A controlled dequeue also covers pressure at the pool's acceptance boundary,
  followed by successful real-worker readmission. No sleeps or production test
  hooks are used.

  Real two-family compilation additionally injects pressure/shutdown after native
  code and replacement metadata are prepared, before publication. Abandonment
  releases unpublished output and scratch; pressure preserves both predecessors
  and allows a successful replacement using the same compiler scratch. Native
  execution stays correct, predecessors are reclaimed after successful cutover,
  and terminal cleanup returns committed executable storage to zero.

  This checkpoint adds only focused tests, not production instrumentation or
  another runtime abstraction. All 957 x86-64 JIT library tests and all 32 HCQ
  publication tests on AArch64/QEMU pass; JIT Clippy with warnings denied,
  formatting and whitespace checks pass. No homebrew or Wasmtime change.

  **Backend rejection closure:** an actual worker now exercises Cranelift's
  fixed-frame implementation limit through the complete consumer, with zero,
  one and two participating families. Its validated negative suppresses only
  the rejected boundary, preserves both dispatch payloads and callable native
  paths, and allocates no executable replacement. A subsequent initial promotion
  succeeds on the same worker and compiler scratch, including the same source
  for the zero-family case. Shutdown releases committed executable storage.

  A separate real-backend Unsupported case verifies that an invalid frontend
  shape remains a visible worker/process failure, not a structural negative or
  successful completion. The fixture adds stack slots before backend compilation
  only under cfg(test), with scoped, one-shot thread-local state; it neither
  fabricates the error nor adds any flag/branch to production compilation.

  Final validation: 959 x86-64 JIT library tests and 34 HCQ publication tests on
  AArch64/QEMU pass. JIT Clippy with warnings denied, the non-test library check,
  formatting and whitespace checks pass. No homebrew or Wasmtime change.
  Step 6 is complete. Production sampling still does not admit reshapes;
  connecting that path and its end-to-end native tests is step 7.

- [x] **7. Enable four-sample production reshape admission.**

  Enable admission where validated native boundary observations are consumed.
  Keep the existing four-sample threshold, score-three deferral, bounded queue,
  nonblocking admission and zero-worker behavior. Consult negative results using
  the agreed cold-path mechanism; add no per-link counter or blocking hot lookup.

  Validate the complete sampled-boundary-to-publication path, including a source
  inside an HCQ body and a demanded interior target. Retain existing publication
  diagnostics, distinguishing initial promotion from replacement and reporting
  the final region sizes without introducing runtime instrumentation.

  **Exit:** native execution tests drive an actual sampled reshape through the
  worker and maintenance consumer, then enter the new entries through native
  links. Ineligible or rejected boundaries remain directly linked. No queued
  reshape is consumed and discarded by a placeholder implementation.

  **Completed:** `sample_transfer` now carries validated boundary identities and
  the existing pool queue across the state-guard release, then applies the four
  samples threshold and nonblocking `admit_reshape`. Admission revalidates current
  versions and consults the process-owned negative index. Busy-family deferral
  leaves score three; the next sample retries. With no pool, heat saturates but
  no job is reserved or enqueued. There is no new per-link counter, native check,
  queue or compilation path.

  Native tests drive both a two-family merge and a source in an existing HCQ
  body's second logical block targeting a late-demanded interior LCQ entry.
  The first three samples do not admit work; subsequent native observations alone
  enqueue the real consumer. After coordinated publication, static ingress
  executes the replacement and the restored LCQ fallback for dropped membership.
  No test-side admission or publication creates these replacements.

  Another native test traverses a rejected return boundary: its validated
  disconnected negative suppresses further compilation from both the original
  and a fresh vCPU sample table, with saturated heat and unchanged callable
  payloads. Existing direct/return paths continue to execute. Focused cold-path
  tests verify exact fourth-sample admission outside the identity mutex and
  score-three retry after a busy family releases its reservation.

  Successful publications log `HCQ replacement` when predecessor families are
  retired, otherwise `HCQ promotion`. Both retain final instruction count, public
  entry count and native-byte size. Logging remains after unlocking publication
  state; a regression checks the replacement label, sizes and lock freedom.

  Validation: 963 x86-64 JIT library tests pass. AArch64/QEMU passes 442 lifetime,
  36 HCQ publication and nine engine background tests. JIT Clippy with warnings
  denied, formatting and whitespace checks pass. No homebrew execution or
  Wasmtime change. Step 8 records the final review/closure of the whole task.

- [x] **8. Validate and close Task 7.**

  Run focused regressions throughout, then the relevant full x86-64 JIT suite
  and AArch64 cross-build/QEMU suites following [the test guide](../../aarch64-tests.md).
  Cover deterministic claim/publication races, zero/one/two-family replacement,
  negative-result invalidation, direct/PIC roots, retained baselines, fault
  identity and shutdown. Use existing tests and targeted synchronization points,
  not another test framework or timing-sensitive sleeps.

  Check repeated replacements reclaim predecessor code and coupled metadata
  after cutover/quiescence; no append-only predecessor, negative-result or
  reservation owner may remain. Check boundary work uses indexed membership and
  affected roots, not full-cache scans. Remove superseded seed-only guards and
  provisional comments only where their replacements are complete.

  Update this plan and the spec to describe the implemented contract and record
  validation and remaining limitations. Native AArch64 hardware validation stays
  explicit under Task 10; QEMU is not hardware evidence. Do not mark Task 7 complete
  with a known ownership, publication or cancellation correctness gap.

  **Exit:** every Task 7 criterion is demonstrated, production reshape is active,
  and Task 8 can audit the complete lifecycle without missing replacement logic.

  Final review confirms exact-token reservation cleanup, version-bound positive
  and negative publication, retained LCQ fallbacks, incoming-root cutover and
  epoch/compiler-pin reclamation. Boundary lookup is indexed; retirement and
  invalidation visit affected memberships, roots and reverse associations, not
  the whole cache. Negative records retain weak evidence, not code or graphs.
  No new native-link checks or runtime instrumentation were added for closure.

  The original grow/no-op/shrink regression has been replaced after the runtime
  audit: 32 alternating interior observations now converge after initial growth
  and entry addition, then yield no-op negatives without shrinking a reachable
  prefix. Registry capacities and charged metadata/committed bytes remain
  constant after warmup; shutdown releases executable storage. Separate tests
  retain discarded-entry fallback and legitimate disconnected-root repartition
  coverage. The late-root
  test now checks candidate validity for both static and PIC insertions instead
  of retaining its obsolete admission-epoch exception. Provisional comments
  and the spec's production-consumer wording have been updated.

  Validation: all 964 x86-64 JIT library tests pass. All 968 AArch64/QEMU library
  tests pass with the subprocess supervisor excluded as prescribed by the test
  guide; its child was run explicitly through QEMU and terminated with the
  expected SIGSEGV and `reason=unattributed-native-pc`. JIT Clippy with warnings
  denied, formatting and whitespace checks pass. No new production correctness
  defect was found in that code/test review. No homebrews were run at that
  checkpoint and no Wasmtime changes were needed. The subsequent authorized
  runtime audit below supersedes its closure; native Arm coherence/ordering
  validation is not claimed.

  **Runtime follow-up (2026-09-23; closure reopened):** all three demos render
  around 60 FPS and stop cleanly, but es2gears produces 1,538 replacements in
  303 seconds and simplegfx 3,349 in 121 seconds. Temporary tracing confirms
  simplegfx repeatedly recompiles the same 51 instructions while alternating
  exported entries; es2gears repeatedly shrinks/grows a 14/7-instruction region.
  Versioned negatives do not suppress these changing positive candidates.
  Stabilize useful entry/member selection across observations without disabling
  legitimate repartition or adding work to native links.

  The audit found production collection ran only on pressure or shutdown:
  simplegfx retains 3,434 HCQ units with only 66 published. Integrate cold
  post-cutover/quiescence reclamation without scanning the whole cache on every
  publication. Add production-path regressions for alternating observations and
  automatic reclamation; the existing reuse test's explicit collector calls do
  not prove that integration. Repeat runtime captures before closing this step.
  Local evidence: `dump/task7-runtime-audit/analysis.md`, raw perf files, timelines
  and logs. Diagnostic instrumentation was removed.

  **Corrections:** discovery first tries the source family's current public root,
  with an exact reachability capture, and falls back once to the logical source
  if the anchor cannot cover the observed boundary within the cap. Freeze retains
  predecessor public entries still inside the candidate; it does not retain
  out-of-body entries or invent dynamic connectivity. These choices keep useful
  prefixes/labels across observations without disabling legitimate repartition.

  Retired units enter an intrusive FIFO by reusing `retirement_next` after
  unlinking. Ordinary cold maintenance visits at most 32 queued records and
  rotates pinned ones; worker completion also collects after releasing Work.
  Neither path scans the live unit/dispatch registry or initiates a new native
  stop solely for collection. Both grace periods and compiler pins still apply.
  Pressure/shutdown retain exhaustive cleanup and empty-segment decommit.
  Tests check below-threshold production collection, bounded passes, pin fairness
  and fault-directory reader safety, without explicit collector calls.

  **Correction validation and closure:** all 967 x86-64 library tests pass;
  final focused reruns pass 150 HCQ and 122 reshape tests. AArch64/QEMU passes
  150 HCQ and 446 lifetime tests after these corrections. JIT Clippy with warnings
  denied, formatting and whitespace checks pass. No Wasmtime changes were needed.

  Ordinary profiling binaries, run sequentially without concurrent compilation
  or QEMU, reproduce approximately 60 FPS and clean exit 0 in all three demos:

  | Demo / duration | Replacements before → after | HCQ resident / published after | JIT cache before → after |
  | --- | ---: | ---: | ---: |
  | simplegfx / 121 s | 3,349 → 26 | 59 / 59 | 109.15 → 41.36 MiB |
  | es2gears / 303 s | 1,538 → 84 | 489 / 489 | 215.16 → 190.87 MiB |
  | textured_cube / 182 s | 144 → 95 | 400 / 400 | 242.82 → 240.05 MiB |

  The formerly oscillating simplegfx body adds its entries at startup without
  toggling them back; sampled HCQ worker cycles fall from 9.36% to about 0.16%.
  Es2gears makes one replacement in the last approximately 56 seconds, versus
  247 before, with no recurring replacements at the two problematic roots.
  Some late first promotions and occasional useful replacements remain; zero
  compiler activity is not required. Its worker cycle share is roughly unchanged
  in the measured window (2.02% → 1.93%), so no additional speedup is claimed.
  Resident/published equality at all three shutdown snapshots demonstrates
  normal predecessor collection below pressure. Process PSS and accounted JIT
  cache are reported separately in `dump/task7-runtime-audit/analysis.md`.
  No execution errors, rejected SVC kinds or lost perf samples were observed.
  These finite captures do not prove universal convergence or long-term leak
  freedom; native Arm validation remains Task 10. Step 8 and Task 7 are closed.
