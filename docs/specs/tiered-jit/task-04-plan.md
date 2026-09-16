# Task 4 implementation plan

Status: steps 1–8 complete (implementation and available-host validation).
Production LCQ publication now
reserves static islands and indexes source sites by their full target key.
Publication automatically registers outgoing links to resident targets (including
self edges) and waiting sources when their destination appears, and schedules
pending/installed replacements when the preferred version changes. Canonical
execution services their link/cutover work. Static fallback hits resolve and
resume canonical ingress within the same invocation. HCQ withdrawal re-registers
links to retained LCQ baselines before reopening admission.
The maintainer-approved state-transfer contract is recorded in the spec.
Step 5 is complete: exact dynamic-transfer preparation, island-free
executable bridges, per-vCPU PIC ownership/retirement, weak deduplication and
stable native-readable PIC tables are implemented. Generated x86-64/AArch64
probes and suspended cold miss resolution are connected to production BR/BLR/RET.
Admission borrows the native table; a cold miss installs for subsequent native
hits and resumes canonical ingress under the existing epoch.
Step 6 is complete: the guest-thread RSB follows scheduler migration and is
exclusively borrowed by native frames. Production BL/BLR and RET update it once
across hot links, cold demand/resolution and sample/budget/control boundaries.
Matched returns use the owning vCPU's PIC without leaving native execution.
Step 7 is complete: mixed-root invalidation, source retirement, stale
preparations, pressure/reuse, protected later-unit fault metadata and memory
holds are covered. Synthetic HCQ families exercise baseline withdrawal and
safety unlink with deferred optional installation. Step 8 is complete;
hardware-validation and fork-pin limitations are recorded in the handoff below.
Task 3 supplies the production LCQ compiler, bounded code cache, fault-aware
gateway and memory/lifetime coordinator. Target-first/source-first static links are connected;
PICs and the guest-thread RSB are connected.

This is a working checklist for
[Task 4](spec.md#task-4-complete-native-chaining-and-safe-cutover), not another
specification. Follow [CONTRIBUTING.md](../../../CONTRIBUTING.md). Update each
step in place with concrete decisions, remaining work and validation results;
do not append session logs or create a separate decision register. Discuss
changes to architectural contracts with the maintainer and update the affected
spec section before implementing dependent work. Keep implementation details
in code. This plan may be removed when the task is complete.

## Scope

Make supported LCQ branches, calls and returns stay in native fast mode on
resolved hits. One native invocation protects an entire chain, not one block.
Implement the link ownership and unlink protocol before making those new roots
reachable in production. Canonical exits remain for real observations, cold
misses, control and errors; they are not a second selectable JIT backend.

Do not add HCQ compilation, functional sampling tables, background workers,
another lowering IR, runtime tuners or a benchmark/testing framework. Synthetic
HCQ units may exercise the existing replacement/family contracts without
implementing Tasks 5–7. Tests accompany each step; the last step consolidates
evidence rather than postponing correctness checks.

## Starting points (before Task 4)

- `cpu-jit/src/engine{,/execution}.rs` and `lcq/invocation.rs` currently enter
  and leave the gateway, reader epoch and mapping lease for every fragment.
  `canonical_exit` assumes that the exiting unit is the initial entry's unit.
- `lcq/compiler.rs` retains allocated entry/exit maps and source-keyed
  `GuestExit` records, but patches terminal transfers to canonical adapters.
  `native.rs::emit_fast_transfer` already emits cycle-safe physical copies;
  it is not a complete inter-unit architectural-state or lifetime protocol.
- `lifetime{,/unit}.rs` owns publication, epochs and retirement;
  `unit/invalidation.rs` and `lifetime/memory.rs` coordinate memory changes.
  Pending retirement lists now avoid full-registry scans for empty work.
  `Reason::LinkPatch` and deferred acknowledgement exist, but do not yet patch
  executable code or own incoming links.
- `executable{,/output,/linux}.rs` supplies bounded W^X storage and instruction
  cache synchronization. Segment island space is reserved; per-source island
  ownership and safe mutation of published patchpoints still need implementing.
- Runtime worker/vCPU owners and scheduled guest-thread state are distinct.
  Inspect `runtime/src/coordinator/worker.rs`, `runtime/src/process/execution.rs`
  and scheduler state before placing the PIC and RSB; the name `JitThread`
  alone does not establish which lifetime it follows.
- Continue using `/home/pladaria/projects/wasmtime`, branch `nixe`, with
  `--offline --config /tmp/nixe-observable-fp-local.toml` on Cargo commands.
  The local fork includes changes absent from the checked-in Git pin. Inspect
  and preserve its pending changes; modify it when required for correct,
  efficient emission. Publishing a new pin is a separate maintainer handoff.
  Do not commit override-induced removal of Git sources from `Cargo.lock`.

## Steps

- [x] **1. Settle chain state, control and ownership boundaries.** Trace the
  current lowering-to-gateway path and inspect both fork backends. Identify the
  exact maps, labels and patchpoint metadata to retain, and the concrete owners
  for static records, incoming backlinks, per-vCPU PICs and guest-thread RSBs.
  Reuse existing identities, state analysis and metadata accounting.

  Resolve architectural-state propagation across independently compiled units:
  a dirty value omitted from a target's live-ins is not necessarily dead, and
  its old host register or spill slot may be overwritten. Specify how values
  inherited from earlier units remain reconstructible at later exits/faults,
  including unchanged-through-block GPRs, vectors, NZCV and FPSR. Check the
  current transfer emitter against that contract; a successful copy plan alone
  is insufficient. Do not solve this by unconditional full-state writeback on
  every edge or by silently making every link incompatible.

  Settle how a running native loop observes coordinator closure, preemption and
  events through the specified deadline/cold-control path. A shared-memory
  request cannot directly change another CPU's pinned budget register. State
  the service bound and reconciliation order explicitly, including requests
  arriving around a cold-poll resume. If the spec's wording requires a change,
  resolve it here rather than adding an undocumented per-edge check.

  **Exit:** record the chosen contracts briefly in this step and nearby code
  comments. The implementation has a concrete solution for inherited dirty
  state, actual exit-source identity, bounded control response and ownership;
  no dependent step assumes that one-fragment behavior automatically chains.

  **Completed inspection and implementation decisions:**

  - **Approved state-transfer contract.** The current `lcq/compiler.rs` clears
    observation sets for local input analysis, initializes `dirty` empty and
    exports only local dirty operands in `snapshot`. Consequently A writing
    X19, followed by B which does not read/write X19, cannot reach a later
    observation through B's map alone: canonical X19 is stale and B is free
    to overwrite A's physical location. Even an input which B reads but never
    writes is absent from B's dirty exit map. Conversely, a target may need a
    clean input absent from the source map. `emit_fast_transfer` checks/copies
    target bindings; it neither repairs those cases nor proves pass-through
    preservation. Task 1's physical-copy proofs do not establish that contract
    for arbitrary independently compiled guest fragments.

    Retain canonical homes for values not carried in a target's physical
    contract. A nonempty bridge commits only source dirty
    values/bits not transferred or proven dead before every target observation,
    directly copies available inputs and loads missing clean inputs from their
    canonical homes. The target treats incoming writable bindings as potentially
    dirty, retaining them in subsequent observation/exit maps until overwritten
    or safely committed. Apply this at NZCV bit granularity, with recipe operands
    kept live until transfer/materialization. No dynamic dirty-mask tests, whole
    register-image copy or extra architectural shadow bank is required.

    This deliberately allows selective canonical memory traffic in nonempty
    static/PIC/RSB bridges. Empty compatible bridges still contain no canonical
    traffic, lookup or PC store. The maintainer approved this adjustment; State
    transfer, static/PIC/RSB linking and the Task 4/conformance criteria now agree.
    Native bridge stores do not return to Rust, release the epoch/mapping lease
    or restore caller FP. Step 2 implements inherited-state maps and exercises
    complete transfer emission with test-owned links; steps 4/5 connect it to
    production owners. The physical-copy primitive alone is insufficient.
    Do not retain obsolete zero-traffic requirements
    for nonempty bridges in their tests.

  - **Control.** Keep `poll_remaining` private in r14/x20. Requesters publish
    coordinator/control/event state; they do not asynchronously write the
    register or another thread's NativeFrame. Each terminal checkpoint charges
    its executed prefix once. The next deadline enters cold control, reconciles
    `PollBudget`, acquire-observes pending work and either exits or rearms/resumes
    the already-completed source transfer. For LCQ the service bound is at most
    4096 + 511 completed guest instructions between checks (less for an earlier
    slice exit), not a wall-clock guarantee. A request just after the final
    check can wait one more interval; the active epoch prevents Closed and
    mutation meanwhile. General Rust cold work suspends guest FP first. Any
    path which waits for maintenance releases its own epoch and mapping lease;
    it never waits for itself. Task 5 consumes the same cold-poll information.

  - **Exit identity and observation.** Add a source-native address emitted only
    at canonical/cold exits, alongside the existing source version/map index.
    Resolve it with the protected native-PC directory and validate version/map;
    do not search by guest destination or assume the initial entry's owner.
    Actual memory faults continue using the captured native PC. Hot links do
    not update a current-unit pointer. Preserve FP ownership and pending host
    FPSR across units even when the last unit contains no FP operation; local
    `fp_activation.is_some()` alone cannot describe inherited host status.
    Keep `PendingExclusiveLoad` invocation-wide and complete its existing
    physical-identity handoff only at a genuine exit. Charge completed block
    checkpoints plus the final partial prefix without double-counting.

  - **Backend boundary.** Both local backends export `nixe_states` with final
    locations and aligned 8/4-byte exit patchpoints; `nixe_entries` exports real
    labels and `nixe_faults` includes exact instruction extents. Entry constraints
    support Any/fixed registers, not caller-selected spill slots. Preserve
    static destination, edge kind, allocated target-PC operand, completed cost,
    patch location and fallback/continuation offsets in owned LCQ output instead
    of discarding them after adapter emission. Add allocation-visible terminal
    checkpoint/continuation support in the fork if needed; do not insert a poll
    which clobbers mapped state after allocation. Bump the native ABI version
    when the accepted frame/contracts change, not during this inspection.

  - **Owners and publication.** Put mutable static-link records/backlinks and
    pending handles in the process `Lifetime` registries; immutable CodeUnits
    retain their semantic exit descriptors. Source retirement detaches outgoing
    records; target retirement restores incoming fallbacks before advancing to
    Unlinked. Link preparations hold strong snapshots and captured admission;
    executable patch writes require Closed authority and cache synchronization.
    Keep new-target publication and later old-root cutover distinct. Optional
    LinkPatch acknowledgement must inspect actual pending records, as retirement
    already does. Safety unlinks cannot inherit the 4096-install batch limit.

  - **PIC and RSB placement.** `worker_main` retains one `CpuThread` per process
    on each vCPU, so its `JitThread`/Reader registration owns stable PIC storage.
    The process tracks those registrations for Closed cache clearing; only the
    owning vCPU probes/updates during native execution, and teardown releases
    all roots before deregistration. Account the weak index against registered
    process/vCPU caches, including inactive caches which still own ways. The RSB
    instead belongs to guest-thread execution state: carry a JIT-owned thread
    sidecar from `GuestThread` through `VcpuExecutionState` and restore it with
    the scheduler lease on every completion/error path. Do not attach it to
    `JitThread` or put JIT prediction fields in architectural A64State. Preserve
    ordinary return target resolution: the loader's mapped SVC stub exits via
    the normal architectural boundary, with recognition owned by the runtime.

  **Validation:** existing host tests for `native::tests::backend`,
  `lcq::compiler::tests::shape` and `abi::tests::poll_budget` pass (9 cases on
  native x86-64 and the same 9 on AArch64/QEMU) with the local fork override.
  They validate current emission/copy/budget primitives,
  not an implementation of the approved chain-state contract. Spec/plan and
  the copy primitive's responsibility comment are updated; no runtime or fork
  behavior changed in step 1.

- [x] **2. Make invocation, polling and reconstruction chain-aware.** Extend
  the native ABI and LCQ boundaries according to step 1. Keep one NativeFrame,
  execution epoch, mapping lease, fault capture and caller-FP save/restore
  across several units. Carry guest FP ownership and exclusive-load state
  across links; reconcile them exactly once when leaving fast mode.

  Emit the required block-cost subtraction/deadline branch before terminal
  transfer, including self-loops and both conditional outcomes. Preserve guest
  flags across infrastructure instructions. The cold path reconciles both
  balances with overshoot, observes control and either resumes the correct
  source continuation or exits canonically. Do not enable functional sampling
  or promotion yet. Preserve loader-return behavior, events, architectural
  helpers, precise partial progress and fault retry without replaying completed
  instructions. Resolve exits and faults to the actual source unit/version,
  not the unit which first entered the chain; retain protected metadata until
  reconstruction and FP completion finish.

  **Exit:** focused tests execute bounded multi-unit chains through the real
  gateway with test-held lifetime protection, including dirty pass-through
  values, spill/register cycles, FP/lazy flags, exclusives and a fault in a
  later unit. An infinite guest loop yields to budget/control correctly.
  Production external edges stay on their existing safe exits until step 3's
  link ownership and mutation protocol is available.

  **Implemented:** native ABI v3 retains `exit_native_pc` (introduced in v2)
  and adds the three borrowed cold-control request pointers. Canonical adapters on
  both hosts publish their own native address using relocation-free LEA/ADR,
  alongside source version/map index. LCQ resolves that address through the
  epoch-protected native-PC directory and checks the version/map; it no longer
  attributes the exit to the initially admitted unit. There is no per-link
  current-owner write or registry scan. Actual fault attribution is unchanged.

  LCQ now seeds potentially dirty state from writable fast inputs (GPRs, SP,
  vectors, TPIDR_EL0 and the required NZCV bits). Snapshots retain those SSA
  values through local uses, PRE-fault/helper exits and FP activation, even
  before their first use. A local flags producer replaces the inherited mask
  with all NZCV bits; otherwise only carried bits are committed. FPCR changes
  exit fast mode and TPIDRRO_EL0 is read-only, so neither needs inherited dirty
  writeback. No all-register ingress, shadow state or runtime dirty-mask walk
  was added.

  `native::emit_chain_transfer` now composes selective canonical writeback,
  cycle-safe physical copies and missing clean-input loads. It commits only
  dirty source values/NZCV bits absent from the target contract; no death proof
  is assumed. Host/deferred flags are captured before clobbering work, absent
  target flag inputs are merged from canonical NZCV, and host-flag ingress is
  installed last. Missing spill loads preserve already transferred RAX on x86.
  Pending host FPSR remains invocation-owned, with no FP transition or status
  read at the link. Compatible empty transfers still emit zero bytes. Public
  maps still cannot reference reserved transfer storage; only emission uses it.
  Staged chain tests use this emitter; production patch ownership is not enabled.

  All LCQ boundary maps permit pending host FPSR, including integer-only units
  and pre-activation boundaries. `host_fpsr_pending` expresses potential state,
  not whether this unit activated FP. The invocation's existing `HostFpState`
  determines whether captured status belongs to the guest. Fault reconstruction
  merges captured guest status only for an active segment; otherwise caller
  status is ignored. Retry preserves the active segment, while escape finishes
  ownership once. Canonical exits already use this same invocation-wide owner.
  No extra FP activation, status SSA operand, status read/store or hot-link
  check is emitted merely because a map permits inherited FPSR.

  Terminal state records now own the final destination location, known static
  BlockKey (when applicable), completed prefix cost, patch width and canonical
  fallback offset. The existing record supplies source version/map, edge kind
  and patch offset. Indirect destinations remain dynamic even for calls;
  PRE helper/unsupported exits exclude their unexecuted instruction. These
  descriptors share CodeUnit ownership and metadata accounting; only terminals
  allocate a payload, not every fault map. Publication validates their physical
  bounds and semantic address space.

  Dispatch terminals now opt into the fork's allocation-visible checkpoint via
  `nixe_exit_costs`. Both targets subtract the completed prefix from the pinned
  counter, then conditionally branch to a separate cold patch for a signed
  nonpositive balance. The hot patch immediately follows the check; resumption
  must start there, not repeat the subtraction. Backend maps export both patch
  locations and cost; LCQ retains the cold offset with its transfer descriptor.
  Mapped SSA values survive the check. Machine condition flags may change:
  LCQ maps hold explicit packed/deferred guest NZCV operands, not implicit host
  flags. No after-allocation byte insertion or hot-path Rust call is used.

  The hot patch still leads to its safe canonical adapter in production. The
  cold patch now enters `native/poll.rs`: acquire-load the process maintenance,
  vCPU control and interrupt words without consuming or acknowledging them.
  Admission binds the process word; the invocation binds the other two to their
  retained runtime owners. Isolated frames without these owners use a static
  quiet word. There is no shared load on the ordinary hot edge.

  A pending request exits with Control; slice exhaustion exits through the
  ordinary adapter. Both leave the old budget and charged counter for the
  gateway to reconcile exactly once. Otherwise the sample-only deadline updates
  both balances, preserves overshoot, rearms their positive minimum and resumes
  at the source hot patch. The backend's <=2048-instruction checkpoint bound
  proves that one +4096 exactly implements the sample-phase wrap on this path.
  No functional sample is enabled; the final NativeReturn reports only its
  final reconciliation, not sample deadlines already consumed while resuming.

  This cold leaf saves/restores only its borrowed GPRs in transfer scratch and
  uses integer instructions/atomic loads, without calls, FP operations, SP
  changes or architectural writeback. The active guest FP segment and pending
  exclusive load remain untouched. General Rust work still requires FP
  suspension first; no such work is performed by this leaf. A request arriving
  after observation waits at most another bounded interval, with the epoch and
  mapping lease still preventing mutation. Actual request handling/waiting stays
  outside native protection. Genuine PRE observations have no checkpoint and
  still charge only their executed prefix on exit. Production links remain
  disconnected. The fork round-trips checkpoint declarations in CLIF and
  includes their compound size in x86 emission bounds.

  **Validation:** the real two-unit gateway fixture verifies the address lies
  in the final unit. A separate production-publication test executes a second
  compiled LCQ unit under the first unit's admission and resolves its precise
  exiting instruction, retaining the epoch and rejecting mismatched versions.
  Missing-address/map tests remain covered.
  Inherited-state regressions enter compiled fast code with deliberately stale
  canonical GPR/SP/vector/TPIDR_EL0/C homes, compare the full result with the
  interpreter, and cover FP activation without losing inherited values or the
  non-carried NZCV bits. PRE-fault maps also retain X19 and C needed only after
  the faulting instruction. Memory-map tests account for inherited inputs as
  well as locally committed writes; existing execution tests still check exact
  pre-writeback values and partial commits.
  The integer-only FPSR regression covers normal return, fault retry and escape,
  with and without an inherited active segment. Distinct nonzero caller status
  must be restored but never imported into guest FPSR; a completed integer
  prefix must not replay. Existing escape checks reject a second reconstruction.

  Terminal tests cover self-loops, both conditional outcomes, direct/indirect
  calls, returns, SVC/BRK/invalid PRE exits and the 512-instruction fragment
  limit. Repatching to the retained fallback reproduces the emitted bytes.
  Execution fixtures check that both budget balances reflect the selected
  terminal's completed cost. Publication rejects a terminal payload on a fault
  observation and its allocated bytes are included in metadata accounting.

  Test-owned compiled loops patch only the hot edge to a physical transfer
  back to fast ingress. The production cold poll resumes across multiple sample
  deadlines and stops the unconditional loop on slice exhaustion or a pending
  request, including zero balance and overshoot; conditional loops also stop
  normally before a deadline. Full state matches the interpreter
  with ADCS/SUBS carrying lazy C across iterations, and both balances charge
  every executed instruction exactly once. The fixture holds the complete
  allocation through one real gateway invocation; it does not install a live
  production link. Cases start at different sample phases, compare reconciliation
  with PollBudget, retain the same epoch/exclusive-load record and carry active
  FP sticky status through integer-only loops. All three request sources force
  Control without consuming the word or reporting a functional sample. Production
  invocation tests verify vCPU/interrupt binding and admission's process binding.
  A two-unit LCQ regression publishes source and target separately, installs
  registered source-to-target and target self links, and requests real
  process maintenance just after the source's first sample-only poll resumes.
  A native release/acquire handshake orders the request without sleeps or
  production hooks. The target executes 4098 further guest instructions before
  its cold poll returns Control; the protected native-PC lookup and version/map
  identify that target, not the initially admitted source. Full state matches
  the interpreter, including lazy C and inherited guest FPSR. Caller FP is
  restored while the invocation's epoch is still active; maintenance completion
  waits for its release. Step 3 also verifies actual unlink and delayed reclaim
  while a compiler snapshot remains, followed by A's restored fallback.
  A second two-unit regression uses production
  `invocation::run`, its mapping lease and delivered fault capture: A performs
  LDXR and passes dirty X0/X2/X3 to B, whose STR either retries write tracking,
  escapes unmapped memory or yields an owned MMIO completion. Full state and
  the exclusive reservation match the interpreter; X2 survives B's fault before
  its first local use. Both balances charge the whole chain exactly once,
  without replaying B's completed prefix. Exit releases the epoch/lease and
  hands off A's actual loaded value. After both native units are released and
  the faulting guest instruction overwritten, the owned exit still reports
  that original instruction; MMIO executes once only on cold completion.
  A separately published LCQ source writes X19/V1/X4 and lazy NZCV, while its
  target receives only C and needs clean X0/V2 missing from the source map.
  The full result matches the interpreter for borrow, zero and overflow cases.
  Native bridge tests force simultaneous GPR/spill cycles, missing spill and
  vector inputs, and reuse of omitted dirty locations. Packed/deferred/host
  NZCV and partial flag masks are exercised on both hosts. They inspect both
  final physical inputs and canonical state: carried homes must remain stale,
  rejecting an accidental full-state writeback.
  The production invocation fixture now also links two high-pressure compiled
  units: both entry/exit maps and the later fault map must contain real spills.
  A leaves inexact guest FPSR active; integer-only B preserves it through a
  tracked-store retry or an unmapped escape. X19/V19 and NZV are selectively
  committed, while the remaining inputs survive until B's post-fault uses.
  Full state, both budget balances, stored RAM and caller FP match their
  expected results without replaying either prefix. Final x86 allocation maps
  are checked to require cycle breaking; AArch64's allocation is acyclic here,
  with deterministic cycles covered by the separate native emitter tests and
  the real-gateway synthetic chain. The latter explicitly rotates two GPRs
  and a spill, and swaps a vector register with a 128-bit spill, on both hosts.
  It retains both code owners through one gateway entry, carries deferred NZCV
  into host flags, and checks full register state, pins, budget overshoot, actual
  exit-unit identity and FP completion before epoch release.
  The staging helper selects the static edge explicitly, not an earlier FP
  guard's canonical exit.
  Fork regressions cover both allocators under spill pressure,
  patch shapes/bounds, invalid costs/IDs, function reuse and CLIF round-tripping.

  The x86-64 JIT library suite passes (407 tests; only the intentional fatal
  subprocess supervisor skipped to avoid core dumps), as does JIT all-targets
  Clippy with the existing `type_complexity` allowance. Thirty focused bridge,
  canonical boundary, gateway, chain, invocation and budget/control tests pass
  on AArch64/QEMU 11.1.1 at closure; see
  [AArch64 tests](../../aarch64-tests.md) for runner limitations.
  The local fork's preceding checkpoint passed 42 Nixe backend regressions
  (with `x86,arm64,disas`) and four Nixe CLIF-reader tests; this native cold-poll
  checkpoint requires no additional fork changes. Formatting and diff whitespace
  checks pass.
  **Scope of completion:** the invocation, transfer, polling and reconstruction
  foundation meets this step's focused exit criteria. These tests are not an
  exhaustive proof of every memory subaccess, exclusive-store or request
  interleaving. Native AArch64 hardware conformance remains Task 10 work; QEMU
  is not a substitute. Production external edges deliberately retain their
  safe exits: owned patch/unlink support is step 3, and live static chaining
  is connected in step 4.

- [x] **3. Implement owned patches, islands and safe unlink.** Add reusable,
  accounted source patch records and target backlinks. Retain exact source
  CodeVersion/ExitSiteKey, target BlockKey/reachability/version, source map,
  target contract and fallback/bridge locations. Queue each pending record
  once using embedded state and generational handles, without separate jobs
  or scans of unrelated units. Prepared link work retains strong snapshots.

  Implement patch shapes for both hosts: aligned eight-byte x86 jmp-rel32 plus
  padding, or aligned AArch64 b-imm26. Reserve each source's worst-case static
  island/helper demand before publication; use the segment's 64 KiB island
  area, with 16-byte far islands as specified. Reuse/release actual slots with
  their owner. Insufficient island capacity tries another segment or splits
  unpublished code; it cannot create an unpatchable published source.

  Extend the existing Closed transition with a protected W^X write window and
  instruction-cache synchronization using RX relocation addresses. Revalidate
  source, target and admission/reachability before installing bytes. Register
  roots before publication makes addresses callable. Replacement, invalidation,
  eviction and shutdown restore incoming fallbacks and detach outgoing roots
  before retirement; a failed safety patch must not reopen execution. Release
  owners outside JIT state and retain epoch/fault protection until actual reuse.
  Drain all safety unlinks; process at most 4096 performance-only installs per
  rendezvous, retaining deferred requests and valid fallbacks.

  **Exit:** tests using published synthetic units prove patch/unpatch, stale
  records, source/target replacement, pending deduplication and deferred batches.
  Force near/far targets and island exhaustion. No RWX mapping, stale backlink,
  early span reuse or lost request remains. Live-code writes cannot occur
  without the coordinator's Closed authority.

  **Implemented checkpoint:** executable storage can reserve a source's full
  island demand together with its code span via `install_with_islands`. Each
  segment's final 64 KiB contains 4096 fixed 16-byte slots. A fixed bitmap in
  the cache's accounted storage tracks ownership; each Allocation retains one
  contiguous slot range without a separately allocated owner/list. Code-span
  selection skips segments lacking a fitting island range, including in its
  best-fit reuse path. Requests larger than a segment fail before publication.
  Outputs without static-link demand reserve zero slots.

  Slots are released with the actual code allocation, not when dispatch stops
  referring to it. Strong snapshots therefore retain both resources. Failed
  installation returns the unpublished code and island ranges; adjacent released
  slots immediately form reusable runs. Tier isolation, segment generation and
  the final 15 MiB segment's reservation bounds apply to islands as well.
  Reservations do not initialize callable islands or permit live-code mutation.

  `native/link.rs` now emits fixed-size final-RX-address branch bytes. Near
  edges reuse the fork's aligned patch encoder; far edges branch to the reserved
  local slot. Its 16 bytes contain MOVABS R11/JMP R11 on x86-64, or LDR X16/BR
  X16 with an inline target literal on AArch64. Neither path changes flags,
  SP or the return continuation. Indirect destinations require an ABI-compatible
  landing. Range/alignment/extent errors are rejected without address wrapping;
  no island is emitted for an in-range edge. The emitted bytes confer no target
  ownership or permission to modify published code.

  `Transition::patch_unit` now writes a registered unit's code span and reserved
  islands only while Closed. It resolves the process-local generational handle
  and retains the actual unit before releasing JIT state. An unforgeable scoped
  permit borrows the transition through RW protection, writes, RX-address cache
  synchronization, RW closure and broadcast pipeline synchronization. Neither
  cache operations nor final owner release run under the JIT-state lock.
  The byte writer is an unsafe linker primitive: it requires valid metadata/ABI
  and already-rooted targets; it does not substitute for link/backlink ownership.

  A partial write, protection/synchronization failure or unwind disables cache
  mutation and prevents process acknowledgement/reopening. The write-window
  guard removes RW on failure; retained RX owners remain intact. Open/Closing,
  foreign-process and unlinked/retired handles cannot obtain permission. This
  adds no generated-code work and no second coordinator or generic job queue.

  Pending static links now use the existing generational registry with charged
  capacity, grown outside JIT state. Each record retains strong source/target
  units, exact map/entry indexes, reserved island and target reachability.
  Immutable unit metadata supplies source version/site, target key/version,
  contracts and patch/fallback addresses without copying binding arrays.
  Preparations retain both owners and captured admission; registration under
  Closed revalidates the owners and preferred dispatch payload. Repeating a
  site does not allocate, enqueue twice or replace its original request sequence.
  Retargeting an uninstalled site first discards its previous pending record.

  Source outgoing lists, target incoming lists and the FIFO pending queue use
  separate links embedded in each record. Detaching a known record is O(1);
  retirement follows only that unit's incoming/outgoing adjacency and releases
  owners outside state. Registration checks only the source's existing exits
  for duplicate sites/island sharing, not unrelated resident units. Replacement,
  eviction, invalidation and shutdown drain those records before retiring the
  unit. The collector's rootless-superseded shortcut also
  schedules coordinated retirement when link adjacency remains, rather than
  bypassing its cleanup. LinkPatch acknowledgement consults the actual queue;
  explicit performance deferral retains the request and valid source fallback.
  Registration itself never redirects native bytes.

  `Transition::install_link` installs static edges through the registered owner
  and Closed writer. Empty transfers remain a direct branch with no bridge
  allocation. Nonempty transfers use the existing architectural transfer emitter
  in a source-owned executable allocation retained by the link record. The
  bridge has an indirect landing, flag-preserving alignment and a final-address
  terminal branch, with its own reserved island for a far target. The cache
  initializes code and any used island before first execution, using the same
  W^X population and synchronization path as other unpublished output.

  Bridge bytes, metadata and boxed owner are charged to the source tier. They
  access only canonical homes/fixed-frame storage, not guest memory; there are
  no guest fault maps or separately published dispatch entries. No compiler or
  cache reference to the bridge escapes installation. Restoring the source
  fallback under Closed and synchronizing therefore ends all possible execution
  of that bridge; removal frees its real span/island outside JIT state. Dynamic
  bridges with independent cache owners remain step 5's separate lifetime work.

  Owner/admission/reachability checks run before preparation and
  immediately before writing. The record becomes conservatively callable before
  mutation; successful synchronization removes only its pending membership,
  retaining both adjacency lists and strong owners. Far edges initialize the
  reserved island before redirecting the patch. There is no guest-edge Rust hop.

  Explicit unlink and unit retirement restore the source-local fallback and
  synchronize before dropping roots/backlinks. Already-installed records cannot
  be discarded as pending work. A failed safety restoration leaves roots intact
  and permanently prevents reopening. Invalidation can join after validation
  while Closed, but its queued safety drain must finish before admission reopens.
  All retirement paths use this same unlink operation, without a safety-unlink
  count limit.

  `Transition::drain_links` drains retirement adjacency before consuming each
  pending FIFO record, including once the installation quota is exhausted. The
  shared coordinator state permits at most 4096 record attempts per stop across
  both individual installations and drain calls. Stale/failed preparations
  count; duplicate already-installed handles do not. Only successful reopening
  resets the counter, so another batch or a replacement Transition owner cannot
  bypass the cap. There is no scan of unrelated units or separate work queue.

  The drain returns true when the pending queue is empty, allowing normal batch
  completion; false requests `complete_with_links_deferred`. Deferred records
  keep FIFO order, strong roots, safe fallbacks and their original LinkPatch
  sequence. A later stop resumes that same request with a fresh quota. Safety
  requests joining after the drain/deferral still block acknowledgement or
  reopening until their actual records are drained. No generated-edge counter,
  timer or extra polling load was added.

  **Validation:** the x86-64 library suite passes (438 tests, with only the
  intentional fatal supervisor skipped). The preceding 19 storage cases passed
  on both hosts; the added executable-island case and three encoding cases
  also pass on x86-64 and AArch64/QEMU. Coverage includes
  exhaustion/another-segment selection, actual address reuse after
  final-owner release, cross-word range reuse, installation failure, tier
  borrowing and last-segment decommit/recommit, exact positive/negative branch
  range limits and invalid addresses. The new execution case initializes an
  unpublished source and its actual reserved island, retains both code owners
  through the call, and repeats after slot reuse with a different target value.
  It forces island routing on both hosts (even where x86 rel32 could reach),
  checking RX/RW separation and the closed write view before execution.
  Five additional mutation regressions pass on both hosts: published leaf
  execution before/after patch and restoration, reserved-island initialization,
  process/lifecycle rejection, actual partial-write errors and caught unwinds.
  They verify permanent failed admission and RW cleanup, including execution
  from another host thread after successful synchronization. OS syscall failure
  injection is not covered here.
  Six pending-link regressions additionally cover duplicate registration and
  charges, registry growth/reuse, FIFO/backlink removal, source/target retirement,
  self-links, deferred acknowledgement, strong preparations, stale admissions,
  target replacement, mapping invalidation, rootless collection and shutdown
  storage release. They also pass on AArch64/QEMU. The
  source's actual fallback still executes after graph cleanup.
  Eight installed-link tests pass on both hosts: actual A-to-B execution and
  restoration, idempotence, replacement/invalidation/source retirement, strong
  compiler retention and real target-address reuse, stale nonempty work,
  nonempty bridge accounting/span/island reuse, failed safety-unlink retention
  of the bridge, and shutdown with an installed self-edge. Nonempty bridges
  also release their real storage after source/target retirement, replacement,
  mapping invalidation and shutdown. A further AArch64/QEMU test uses real
  aligned allocations to put the target beyond 128 MiB, executing both a direct
  far edge and a nonempty bridge with its own far terminal island before
  restoring the near fallback.

  The selective-state LCQ chain test now uses registered live installation,
  rather than a staged test-owned bridge. Through the real gateway it preserves
  absent dirty X19/V1/X4 and partial lazy NZCV, loads clean X0/V2 inputs, accounts
  both blocks and attributes the final exit to B. After unlink it executes only
  A's restored fallback with exact state/budget. This passes on x86-64 and
  AArch64/QEMU. An additional storage test executes a relocated near bridge
  with RX enabled/RW closed and verifies failed tail alignment/extent validation
  returns the actual unpublished span/island and metadata charge on both hosts.
  Two batching regressions exercise 4099 real published sources: shared quota
  across explicit installation/drains and abandoned-owner takeover, pending
  FIFO/fallback execution across reopen, ticket completion only after the later
  stop, a safety request joining after deferral, and restoration of 4098 incoming
  edges without a safety cap. They also check retirement priority and stale
  attempt accounting. Both pass on x86-64 and AArch64/QEMU (17 link tests on
  the latter). No test-only limit or fabricated registry state is used.

  Five integrated LCQ chain tests pass on x86-64 and AArch64/QEMU through the
  actual registered installer. The old staged inter-unit bridge helper is
  removed. Delivered faults cover allocated spill/register cycles, selective
  writeback, lazy flags, inherited FPSR/caller FP, tracking retries without
  replay, precise unmapped exits and exclusive-load/MMIO completion after
  coordinated shutdown has released the native owners.

  Concurrent retirement after A's resumed cold poll waits for the whole
  A->B->B invocation, restores both incoming/self edges and cannot reclaim B
  until its compiler snapshot is released. A separate test pauses at B's fast
  entry, requests B's retirement, then delivers a tracking or unmapped fault
  through `invocation::run` with no extra snapshot retaining B. Closed unlink
  is rejected while the invocation is active; real reclamation succeeds only
  after retry/reconstruction releases its epoch. A then executes its restored
  fallback with exact architectural state and instruction budget. Deterministic
  native entry handshakes are test-only, with no production hooks or sleeps.
  Native hardware conformance is still outstanding.
  Format and Clippy checks pass.

  **Closed:** owned installation and removal meet this step's exit criteria.
  At step 3 closure, production guest-to-guest live patching remained disabled; step 4 connects
  these primitives to demand compilation and real static-edge discovery.

- [x] **4. Connect production static chaining.** Retain real static exit sites
  from LCQ output and build minimal bridges from final allocation maps. Link
  already-valid targets before making a source reachable; later target demand
  or replacement schedules the existing source records for rendezvous. Index
  waiting edges by target key instead of searching all resident code.

  Implement permanent source-local fallback thunks. A fallback canonicalizes
  precisely the source state, resolves the exact BlockKey and uses a published
  target's canonical ingress; a demand miss leaves native protection before
  compilation. Do not compile successors speculatively. Resolved static hits
  use fast ingress, with no Rust call, dispatch lookup, PC store or repeated
  gateway. Cover conditional edges, fallthrough/emergency cuts, B and BL;
  commit architectural X30 exactly once for calls. Use native transfers for
  nonempty bridges and reserved islands only when branch range requires them.

  **Exit:** production LCQ chains execute correctly with target-first and
  source-first publication, loops and replacement. Inspect actual bytes on
  both hosts: after the budget checkpoint, a compatible empty in-range edge
  contains only the guest branch decision and one direct host branch.
  Nonempty/far links satisfy their separate shapes. Ordinary static-hit
  execution no longer visits the per-fragment Rust resolver.

  **Implemented checkpoint:** `Compiler::publish` now reserves one island per
  linkable static exit before publishing the source, in stable state-map order.
  This is the normal `JitThread::demand` publication path. Conditional outcomes
  each reserve a slot, as do B, BL and emergency fragment cuts; dynamic targets
  and observations reserve none. Unit publication rejects insufficient static
  island capacity and static keys attached to non-dispatch edge kinds.

  `TerminalTransfer::static_target` now means a linkable dispatch destination,
  not simply a constant PC. BRK, SVC, unsupported instructions and architectural
  helpers retain their canonical destination/exit metadata but export no static
  link key. This prevents automatic discovery from bypassing their semantic
  completion. The tests' separate publication implementation is removed;
  instrumented LCQ chain fixtures use the same captured-image checks, island
  reservation and owned publication as the production compiler.

  Static source discovery now uses a target-keyed index populated under the
  same JIT-state lock before source dispatch publication. Each source owns an
  accounted array of intrusive list records, retaining its state-map index and
  island ordinal without copying allocated bindings. Each site also names its
  pending/installed generational link directly; duplicate registration no longer
  searches outgoing adjacency, and unlink clears this identity before reuse.
  Lookup visits only that complete BlockKey's sources; removal is O(1) per
  source exit even for a
  high-fan-in target. Capacity grows outside state, is revalidated at publication
  and is reusable. Failed publications create no associations.

  These are weak source identities, not target roots or queued patch jobs.
  Target withdrawal/replacement preserves source discovery for later demand;
  source unlink removes its associations before compiler-held code can be
  reclaimed. Closing/invalidating sources cannot acquire new work through the
  index. Shutdown releases both the source arrays and index storage. This does
  not itself create target roots or change executable bytes.

  Dispatch slots now retain cold generational unit/entry locations for both
  tiers, updated with the payload under JIT state and cleared on withdrawal.
  `prepare_static_link` resolves a source island's full target key to the
  preferred entry and acquires both strong owners in one Closed lock interval.
  Missing/withdrawing targets leave the fallback intact; registration and byte
  installation still revalidate the captured owners/admission/reachability.
  No unit-registry scan or additional generated-code check is needed. LCQ
  replacement retirement and HCQ entry-baseline pinning reuse these locations
  instead of scanning by code identity. Real LCQ chain fixtures now exercise
  this target resolution before registered installation.

  `refresh_static_link` now reconciles a known site with preferred dispatch.
  An unchanged target keeps its original handle/request; a different or missing
  target first restores the source fallback through the Closed unlinker before
  replacing/removing its root. This also supports LCQ-to-HCQ retargeting while
  the old baseline remains alive, and pending-link replacement without duplicate
  queue membership. Graph insertion is separated from admission requests and
  allocation: the existing registration path uses one pre-reserved, nonfailing
  state-locked operation which can be reused by publication before reachability.
  LCQ chain and slice-loop fixtures now use this reconciliation path.

  Canonical execution now consumes pending LinkPatch/TierCutover work through
  `try_service_links`: on closed admission or after completing a native control
  exit, it acquires an available transition, closes deferred admission and yields
  if readers are still active. It never waits for another scheduled vCPU. When
  quiescent, it drains safety retirements, installs up to the existing stop quota,
  and acknowledges or defers through Batch before reopening. Memory mutation,
  capacity and shutdown requests remain with their own authority, including
  requests joining the stop. No own invocation/lease/compile claim survives into
  the service, and no per-edge request load was added. A real `run_slice` test
  now installs a registered LCQ edge and executes it both after closed admission
  and a deferred native control exit, preserving instruction accounting.

  **Automatic outgoing publication:** a new unit resolves its static exits to
  current preferred entries under the publication lock, overlaying its own new
  payloads for self/multi-entry targets. Registry storage is prepared outside
  state; capacity and all required generations/targets are validated before
  mutation. Publication then closes admission and registers each link's strong
  owners, adjacency and pending identity before exposing dispatch. The existing
  canonical maintenance service installs those links under Closed before the
  next normal entry (the established performance quota may defer excess work).
  Missing targets keep their source-local fallback; no speculative compilation
  or second queue is added. Failed preparation/publication cannot expose a
  partially registered graph. This is the normal compiler publication path,
  not a selectable test backend.

  Target-first execution tests now publish both conditional destinations before
  their source and execute B, BL and both CBZ outcomes without explicit link
  registration. They verify patched bytes, X30, precise progress and SVC source
  attribution. A separate publication test checks initial Closed admission,
  exact roots/backlinks, automatic installation and shutdown for ordinary/self
  edges. Manual-patch fixtures explicitly defer registered performance work
  through Batch when they need to inspect the original fallback. Registry batch
  checks cover reusable/held slots and generation exhaustion without mutation.

  **Waiting-source publication:** when a preferred destination is published,
  its full-key index supplies existing unlinked sources. Their roots/backlinks
  and pending membership are registered in that same publication transaction,
  before dispatch exposure. Capacity includes the waiting sites and is
  revalidated before mutation; traversal advances through intrusive cursors,
  with no resident scan, repeated bucket scan or temporary job/worklist. A new
  self edge is attached only once. An existing uninstalled request is replaced
  in that same locked publication: remove its old generational handle, FIFO
  membership and backlinks, then attach the successor. The source stays on its
  fallback throughout; both old units remain registry-owned while dropping the
  extra pending roots. Repeated deferred replacements retain only one request
  per site, and old-target retirement cannot cancel the successor request.
  Each site also records its callable handle: at most one installed edge and
  one pending successor coexist, both in the ordinary ownership/backlink lists.
  Publication replaces only pending work; the callable branch, target and bridge
  remain unchanged for active readers. Under Closed, installation restores and
  synchronizes the fallback before removing the old record and installing the
  successor. This also works when HCQ retains the old LCQ baseline. Cancelling
  a successor restores the site's identity to its still-callable edge; retiring
  the old target removes only that edge, preserving a live source's successor.
  Source retirement/shutdown drains both records. Optional retargeting shares
  the 4096-attempt quota; excess work keeps a valid old edge, or the fallback if
  safety retirement already removed it. No new generated check or job queue.

  A production source-first CBZ test demands only the executed successor and
  checks actual patch bytes after each branch becomes resident, then reuses both
  native links. Additional tests cover 48 waiting sources plus an arriving
  self-link, unrelated keys/FP specializations, stale publication without roots
  or maintenance requests, and fresh generational links after explicit target
  withdrawal and subsequent demand. Safety withdrawal retains its own authority;
  the canonical service does not acknowledge eviction requests.
  Pending replacement tests also cover LCQ replacement, HCQ preference with a
  retained LCQ baseline, repeated deferral and cancellation on source/target
  withdrawal, using normal publication and maintenance without manual refresh.
  Installed replacement tests execute the old branch under an active reader
  after publication, then service and execute its successor for LCQ/HCQ. They
  cover successor cancellation, source retirement, retained-baseline bridge-span
  reuse and real 4097-edge deferral across stops.

  **Retained-baseline cutback:** after HCQ withdrawal rewrites dispatch and
  releases family pins, the same Closed owner reconciles only the full-key
  source buckets for its entries. Eligible LCQ baselines acquire ordinary
  registered roots and LinkPatch work before admission reopens. Sources/targets
  already retiring and shutdown cannot acquire new work. An unchanged baseline
  edge keeps its branch, handle and queue membership while refreshing dispatch
  reachability; no redundant byte patch is required. Tests cover multi-entry
  selection, unrelated/dying sources and 4097 links: every HCQ edge is safely
  removed, excess baseline installations remain on fallback and finish at the
  next stop. No resident scan or separate work queue is introduced.

  **Static fallback resolution:** a source-local canonical adapter now enters
  the invocation's cold System-ABI resolver instead of exiting its gateway.
  It retains the original reader epoch, mapping lease and exclusive-load record.
  Source version/map metadata supplies the full static target key. Lookup uses
  coherent preferred dispatch under the state mutex and refuses Closing/Closed
  admission; it neither waits for maintenance nor compiles while protected.
  Missing destinations, slice exhaustion and control use the ordinary canonical
  exit. Resident hits jump to canonical ingress, without a second gateway or
  epoch. Installed direct links never execute this helper or load its pointers.

  Source writeback precedes FP suspension/status merging; general Rust runs in
  the caller environment, and only successful continuation resumes guest FP.
  Canonical ingress reloads architectural values clobbered by the helper. The
  pinned context/arena/poll registers survive the System ABI, and X30 is not a
  return channel. The callback borrows the invocation's dispatcher and is cleared
  before its stack owner goes away on normal, fault and error paths. Bare native
  gateway users without a dispatch callback still exit canonically.

  Tests deliberately unlink real production sources while leaving destinations
  resident. A single `invoke` then reaches the destination SVC through B/BL,
  both conditional outcomes and a lazy-flags loop. Separate cases cover demand
  miss, exhausted slice, preemption, FP status/caller restoration and a fault in
  the resolved target with correct source-prefix charging and target attribution.
  Protected lookup also tests preferred HCQ, full-key/FP isolation and closure.

  **Loader return:** the approved runtime-owned convention is implemented.
  The existing mapped `SVC #7` stub is executed normally, including through
  native links. Runtime completion recognizes the SVC source against the
  executing thread's loader address and retains X0, source/context and process
  exit cause. The SVC costs one instruction and keeps PC at its source, matching
  ordinary exception dispatch; a budget/control stop before it cannot terminate
  the process. CPU request fields, interpreter/JIT address interception and the
  worker's copied loader address are removed. No per-edge checks or special
  native ABI/fork support are needed. Both backends, including AArch64/QEMU,
  cover split/unsplit return, zero budget, preemption, full-width result and
  nonmatching source/thread/SVC cases. The real installed-chain test now also
  exits through SVC #7. The minimal NRO acceptance case passes through the
  scheduler/runtime with both interpreter and JIT.

  **Closure validation:** all 472 host library tests and 67 runtime tests
  pass, excluding the intentional fatal missing-native-PC supervisor test.
  The five native static-fallback tests and protected-dispatch lookup test pass
  on the host and AArch64/QEMU. Workspace all-target checks pass.
  All 40 AArch64/QEMU link tests pass, including automatic installed retargeting,
  active-reader preservation, cancellation/reuse, retained-baseline bridge reuse
  and the actual 4097-replacement/cutback boundary. The corresponding 39 host link tests
  pass. Clippy and formatting/whitespace checks pass.
  Final AArch64/QEMU checks also pass for all 32 native-boundary tests, three
  direct/far branch byte-shape tests, five chaining tests and twelve production
  execution tests. Minimal-NRO acceptance passes with both CPU backends.
  Twenty-three publication regressions, including automatic target-first native
  B/BL/CBZ execution, self-link ownership, waiting-source registration and pending
  retargeting, pass under AArch64/QEMU. The replacement/mapping-invalidation
  regression also passes there with automatic pending retargeting. Source-first
  conditional demand/patching and full-key/FP
  specialization isolation also pass there. The minimal NRO acceptance case
  passes on the host with both CPU backends; workspace checks and Clippy pass.
  The real slice-loop
  installation test and all four link-service tests pass under AArch64/QEMU,
  including deferral after the actual 4096-installation limit, active readers,
  another transition owner, foreign maintenance and replacement retirement.
  Pending/installed retargeting with a retained LCQ baseline, all five LCQ chain
  tests and the slice-loop service test also pass with the new reconciliation
  path under AArch64/QEMU.
  Link/index, production island-reservation and LCQ chaining tests have also
  passed under AArch64/QEMU with the local Cranelift override. Index coverage
  includes full-key separation, target-first/source-first discovery, replacement,
  multiple exits to one target, growth, head/interior/tail removal, stale
  publication, compiler pins, slot reuse and shutdown. Target resolution also
  covers missing demand, preferred HCQ multi-entry selection, LCQ fallback after
  HCQ withdrawal, replacement and reused dispatch identities. Format,
  diff whitespace and Clippy checks pass (the existing type-complexity allowance
  remains). Native AArch64 hardware validation is still outstanding.

  **Closed:** step-4 implementation and contract review are complete. First-demand
  target-first/source-first/self links are automatically registered and serviced
  in production, as are waiting sources after an explicitly withdrawn target is
  demanded again. Pending and installed links automatically follow the newly
  published preferred version through the existing Closed maintenance service,
  including return to a retained baseline without a new publication. Native
  AArch64 hardware conformance remains outstanding; QEMU is not that evidence.
  PICs, the guest-thread RSB and later cutover/consolidation work remain in
  steps 5–8, not part of this closure.

- [x] **5. Add source-keyed PICs and bounded dynamic bridges.** Allocate the
  specified 2048-set, two-way writable/nonexecutable PIC per active vCPU.
  Probe full ExitSiteKey and target BlockKey from generated BR/BLR/RET paths,
  preserving live guest flags and using only reserved scratch/transfer storage.
  Hits jump to the exact immutable BridgeUnit without Rust/resolver calls,
  atomics, shared-generation loads or recency writes. Its native transfer may
  access only the selective canonical homes required by step 1; empty bridges
  have no canonical traffic. Never interpret a guest address as a host pointer.

  On a miss, canonicalize before the cold resolver; alternate replacement ways
  and build/deduplicate bridges using exact source and target contracts. Each
  occupied way owns a strong generational bridge reference. The process-wide
  weak index uses 4096 slots per active vCPU and round-robin collision
  replacement; weak entries do not keep executable storage alive. Charge both
  metadata and executable spans, and handle vCPU registration/removal without
  leaving roots or exceeding the stated bound. Dynamic bridges do not allocate
  static islands. Clear affected PIC ways while Closed before source/target
  retirement, then retire bridges through the existing epoch/cache lifecycle.

  **Progress:** the cold preparation resolves only BR/BLR/RET source maps to
  a resident preferred entry with the full execution key. Its key includes
  ExitSiteKey, target BlockKey, ReachabilityVersion and target CodeVersion.
  Strong source/target references protect emission outside JIT state; a final
  validation under the insertion lock is required before any PIC exposure.
  Stale preparations cannot become callable, but keep their actual unit storage
  alive until dropped. Static and dynamic bridges share architectural transfer
  emission. Dynamic bridges allocate no static islands: a 16-byte inline tail
  uses a near direct branch or a scratch-only absolute far jump. Empty transfers
  allocate no code and select the target fast ingress with both owners retained.

  The runtime's `create_worker_cpu_thread(vcpu)` creates one `JitThread` per
  process/vCPU, so its Reader registration is the existing lifetime to use for
  the PIC. `NativeWorker` is host-worker fault machinery, not the PIC owner;
  guest-thread migration belongs to the separate RSB step.

  Each Reader now allocates a charged 2048-set/two-way PIC. Cold installation
  revalidates the preparation, retains an immutable bridge owner and alternates
  replacement ways; exact hits leave replacement state untouched. Handles carry
  process, reader and nonreused bridge generations. Occupied ways link into both
  source and target unit lists, so Closed retirement clears only affected ways
  in O(1) per association. Rootless collection cannot bypass that rendezvous.
  vCPU teardown walks its occupied-only list; all removed executable owners and
  PIC backing are destroyed outside JIT state. Shutdown uses the same unlinking.

  Cold installation now takes the prepared key before emission and first tries
  the process-wide weak index. A hit shares the immutable bridge without a new
  code span, metadata owner or generation; a final recheck also shares a winner
  published during concurrent emission. Each active vCPU contributes exactly
  4096 weak slots. A charged compact selector chooses a shard and two-way bucket
  in O(1), with round-robin collision replacement. Entries contain only a PIC
  site and bridge generation, not Weak/Arc allocations. A stale anchor is cleared
  on lookup; changing the active-vCPU set may lose weak hints, never PIC validity.
  Removing a vCPU frees its entire weak shard without rebuilding other shards.

  The native-readable table has two plain pointers per set (32 KiB per vCPU),
  separate from mutable ownership/backlink metadata. Each pointer names an
  immutable scalar-layout record inside the way's strong bridge owner; shared
  bridges share that record too. Full source/target keys have explicit encodings,
  including distinct Dynamic and Exact(0) FP specializations. Installation
  exposes the pointer only after attaching its owner; removal clears it first.
  Stable shared backing with interior-mutable cells keeps registry growth and
  other vCPUs' backlink writes separate from native reads. Table writes still
  require owning-vCPU quiescence/suspension or Closed, not merely the state lock.
  Both backing and records are included in the existing metadata charges.

  Native probes now use the same set selector and compare both ways' complete
  source-site/target keys, with no shared generation lookup, atomics or recency
  writes. Hits jump through the record's bridge address; misses fall through to
  the caller's canonical adapter. All guest locations survive. Live host flags
  are saved/restored around comparisons (including inverted carry); packed/lazy
  operands remain untouched. The x86 LAHF/SAHF sequence briefly saves/restores
  all of RAX in transfer storage; comparisons themselves use reserved scratch
  only. Small key constants use immediate comparisons without redundant pointer
  reloads. A null invocation table takes the same state-preserving miss path.

  Admission now borrows the owning registration's stable table into NativeFrame
  under the same lock as epoch publication; dropping the invocation clears that
  pointer before announcing quiescence, including failed admission. FaultLookup
  carries the invocation's exclusive Reader borrow. Its unsafe `suspend_native`
  boundary permits PIC installation only while native execution is stopped and
  canonical writeback/FP suspension are complete. This thread-bound borrow keeps
  the same epoch announced and reuses the ordinary checked installation path;
  it does not disable the quiescent API's active-reader check. It returns no fast
  address: a cold System-ABI miss must resume canonical ingress, not the source's
  now-clobbered physical register contract.

  Production BR/BLR/RET now execute the probe at their charged terminal patch.
  Their canonical fallback follows the probe; exhausted slice polls jump directly
  to that fallback and control polls use their separate canonical exit. Sample-only
  polls resume the charged hot patch. Static and indirect misses share the
  `dispatch_fallback` gateway and suspended-FP resolver. Published CodeUnit metadata
  retains its write-once generational registration handle, so the protected native-PC
  lookup reaches the actual source's registry slot without scanning resident units.
  Missing/misaligned targets return to canonical demand/fault handling; closure
  or stale preparations yield a canonical Control exit without waiting. A bridge
  capacity failure resumes protected canonical ingress without caching or exceeding
  the hard limit. Table allocation occurs once at vCPU registration.

  Integrated collision/replacement and cross-vCPU sharing checks are complete.
  Mixed-root concurrent invalidation, full pressure/recovery and teardown remain
  step 7's consolidation; native Arm hardware conformance remains outstanding.

  **Validation:** six dynamic-preparation tests pass on x86-64 and AArch64/QEMU:
  exact source maps/versions, multi-entry/preferred targets, full-key rejection,
  empty/nonempty transfers, executable span reuse, stale admission and retirement
  while preparations retain both units. Three `inline_` tests cover near/far
  bytes, execution and failed-extent cleanup on both hosts (Arm via QEMU).
  The x86-64 JIT library suite passes: 508 tests, excluding only the deliberate
  fatal missing-native-PC subprocess. The focused static-link (39 tests) and
  executable (22 tests) suites, runtime (67 tests), Clippy and formatting also pass. Hardware Arm
  conformance remains pending.

  Seven PIC ownership tests additionally pass on x86-64 and AArch64/QEMU:
  two-way collisions without hit recency writes, stale handles, targeted
  multi-vCPU retirement, self edges, teardown/storage reuse, admission/generation
  failure, active-reader protection and 1000 replacements at a fixed metadata
  charge. These exercise cold ownership, not generated PIC hits.

  Six weak-index tests cover nonempty bridge sharing without new allocation,
  stale anchors, full-key collisions, round-robin replacement, active-vCPU
  growth/removal and concurrent preparations sharing the published winner.
  They pass on x86-64 and AArch64/QEMU; no native hardware claim is made.

  Native-layout tests cover field offsets, empty ways and full-key encoding;
  a protected raw-table reader also runs alongside registry growth and another
  vCPU's bridge replacement/backlink writes. Existing ownership tests now check
  that native cells name the exact live record or are null after removal.
  These checks pass on x86-64 and AArch64/QEMU.

  Executed probe tests cover both hit ways, empty/null tables, a first-way
  collision followed by a second-way hit, and single-field full-key mismatches.
  Each host runs 7168 cases combining compact/full-width keys, all 16 NZCV
  patterns, host/packed/deferred flag representations and GPR/spill/constant/SIMD
  target operands. The complete architectural state survives except the explicit
  test exit PC. Invalid target locations are rejected before emission. These
  isolated gateway tests do not claim production miss-resolver integration.
  All 40 native-module tests also pass on AArch64/QEMU with the new frame field.

  Three suspended-installation tests pass on x86-64 and AArch64/QEMU: repeated
  replacements retain the announced epoch/table, closure rejects insertion and
  cannot unlink the old way until exit, and cross-process preparations cannot
  populate the table. They also check pointer cleanup on normal, missing-entry
  and shutdown admission exits. The 16-test PIC filter passes on AArch64/QEMU.

  Seven production integration tests cover BR/BLR/RET miss resolution followed by
  hits with the resolver absent, missing/misaligned destinations, exhausted/control
  poll bypass, sample-only resumption with dirty X5/X19 and lazy flags, target
  retirement/relearning, and 100000-edge indirect loops. Shape tests verify that
  hot indirect patches reach the emitted probe while retained fallbacks stay canonical;
  no dynamic islands are reserved. Three resident destinations separated by 8192
  bytes exercise alternating two-way replacement despite repeated native hits,
  eviction/relearning and unchanged cache usage. Two production vCPUs resolve to
  the same immutable nonempty bridge without additional allocation, execute it
  with no resolver and retain it after one vCPU is destroyed. Shutdown returns
  all executable storage. The twelve static/indirect fallback tests pass on
  x86-64 and AArch64/QEMU. No homebrew FPS improvement has been measured yet.

  **Exit:** tests cover monomorphic hits, two-way collisions, identical targets
  from different source maps/versions, weak-index staleness, replacement,
  vCPU teardown and target churn. Bridge ownership stays within
  4096 × active-vCPU count. Inspect and execute scratch-only probes, emitted
  transfers and all indirect landing pads on both hosts.

- [x] **6. Add the guest-thread return stack.** Store the 16-entry circular RSB
  with the scheduled guest thread, preserving it across vCPU migration while
  selecting the destination vCPU's PIC. Store full continuation BlockKeys only,
  with the specified head/depth and overwrite-oldest overflow policy.
  BL/BLR pushes after updating X30, regardless of whether transfer hits or
  misses; polling/resolver resumption must not push twice.

  RET uses its architectural target, compares the full predicted key, pops on
  a match and probes the ordinary PIC with that RET's source ExitSiteKey.
  Unresolved matches use the miss resolver; mismatch/underflow also clears the
  prediction chain. Preserve loader-return behavior and invalidate predictions
  when their execution key becomes invalid. Guest calls/returns remain jumps,
  never host calls/returns; predictions cannot keep native code alive.

  **Progress:** `ReturnStack` has a scalar native-readable layout for sixteen
  full continuation keys, with explicit Dynamic/Exact FP encoding and separate
  bounded head/depth fields. Runtime-created JIT guest threads own one boxed
  stack each; interpreter threads allocate none. The scheduler moves the same
  allocation alongside architectural state into `VcpuExecutionState` and restores
  it on normal completion, execution errors and aborted worker delivery. Neither
  the process/vCPU `JitThread` nor its PIC owns these predictions. Cloning a guest
  state copies predictions into an independent allocation, never shared mutable
  state or a native-code owner.

  Production `run_slice`/`invoke` now require the scheduled guest's mutable RSB;
  the runtime passes that owner explicitly. `NativeFrame::with_return_stack`
  retains its exclusive Rust borrow for the frame lifetime, including suspended
  miss resolution. Bare ABI fixtures may leave the pointer null, but the
  production JIT path cannot omit its owner. Canonical admission clears the
  prediction chain if its execution identity (address space, profile, platform
  or FP specialization) changes; an ordinary change of PC preserves it.

  Native emission now implements ring insertion (head modulo sixteen, saturated
  depth and overwrite-oldest overflow) and a full-key return check. A matching
  RET pops then uses the ordinary source-keyed PIC; a matched PIC miss preserves
  older predictions. Mismatch/underflow clears the chain and bypasses even a
  populated PIC. A null stack falls through without dereferencing it. The RSB
  comparison and PIC share one saved PC/host-flag image, without duplicate
  capture/restore on a matched hit. Writes and comparisons use reserved scratch
  and transfer storage; x86's flag capture briefly borrows/restores RAX as the
  existing PIC does. No host calls, host-stack adjustment or canonical guest
  state traffic is added by these emitted operations.

  Production placement follows mutually exclusive terminal paths. A linked BL
  pushes in its static bridge before architectural transfer; its unlinked
  fallback pushes before canonical writeback. BLR pushes before its PIC probe,
  and dynamic bridges do not repeat it. RET checks/pops before its PIC probe.
  Exhausted/control polls execute prediction bookkeeping without probing or
  entering a successor; sample-only polls resume the hot patch before that
  bookkeeping. Cold resolver resumption and later demand enter the destination,
  not the completed call/return. Static-call unlink restores the push-bearing
  fallback, whereas indirect fallback metadata points after RSB/PIC operations.

  **Closure:** the production call/return path and guest-thread ownership are
  connected and covered on available hosts. Mixed-root invalidation, pressure
  and teardown consolidation remain step 7; hardware Arm conformance and manual
  homebrew/performance validation remain outstanding, not implied by QEMU tests.

  **Validation:** the native-layout/full-key encoding test passes on x86-64 and
  AArch64/QEMU. Formatting and all-target Clippy for JIT/runtime pass.
  All 69 runtime library tests pass, including ownership transfer through
  vCPU 0→1→0, abort restoration, independent cloned storage and no interpreter
  allocation.
  Three RSB tests verify scalar layout, full-key changes and exclusive frame
  binding. A production invocation test retains sixteen preloaded predictions
  through native NOP/BRK slices on vCPU 0→1→0 and shutdown without retaining code.
  The production path passes all 520 x86-64 JIT library tests
  (excluding the deliberate fatal missing-native-PC subprocess), 53 engine tests
  and all-target JIT/runtime Clippy. RSB binding/key tests and the preservation
  test also pass on AArch64/QEMU, as do the 34 native-boundary tests with the
  extended frame. Native hardware coverage remains pending.
  Isolated native RSB execution now passes on x86-64 and AArch64/QEMU: 3072 push
  combinations per host cover wrap/overflow, narrow/full-width keys and all NZCV
  patterns with host, inverted-carry, packed and lazy flag contracts. Return
  probes cover 24576 combinations per host with register/spill/constant/SIMD
  target operands, both PIC ways, full-key mismatches, empty/absent RSB/PIC and
  a wrong PIC source identity. Tests compare the entire architectural state and
  ring contents, including untouched slots and surviving older predictions.
  Existing native PIC tests still pass after sharing their lookup tail.
  Five production RSB tests cover linked BL/BLR→RET chains with no resolver,
  sample-only resumption, budget/control exits at both call and return, demand
  of an unpublished callee, migration with a live call and a destination-vCPU
  PIC, and warm underflow/mismatch bypass. Nested 20-level calls, recursion to
  depth 24 and nonlocal guest X30 restoration match interpreter state and exact
  instruction accounting. Short warmed recursive chains stay entirely native;
  overflow/nonlocal returns correctly use cold resolution. These tests and the
  emitted-shape checks pass on x86-64 and AArch64/QEMU. The runtime suite includes
  loader-stub execution and termination on both backends.

  **Exit:** nested calls, recursion beyond 16 entries, nonlocal returns, X30
  modification, underflow, miss/resume and guest-thread migration preserve exact
  behavior. Matched RSB/PIC hits stay native, with only required selective
  transfers in nonempty bridges and no canonical-state traffic in empty ones;
  stale predictions cannot bypass PIC validation or prevent reclamation.

- [x] **7. Close linked lifecycle and pressure cases.** Exercise the complete
  production path with every root type present. Replace or invalidate a target
  while another vCPU executes its chain, including a later-unit fault dispatcher.
  Cover source retirement, executable writes through aliases, mapping changes,
  stale compile/link preparations and requests arriving during Closed. Retain
  the memory authority's stop even when no executable unit is affected.

  Verify dispatch/static/PIC roots are cut before reclamation, HCQ baseline
  promises are released in the right order, and safety work cannot be deferred
  as optional linking. Exercise pressure, retained snapshots, bridge/island
  reuse, segment decommit/republication and process shutdown. Fix the actual
  owners rather than adding periodic full-cache flushes or a second collector.

  **Progress:** production BL/BLR callers and RET continuations now have a
  combined lifecycle test with two warmed vCPU PICs. An executable write through
  a writable physical alias, or removal of the callee's execute permission,
  cuts static and dynamic reachability without retiring unaffected callers or
  unrelated code. With the old callee retained by a compiler snapshot, both
  callers on both vCPUs exit before executing it; their RSB push still occurs
  exactly once. Republishing the callee restores links to its new version, and
  resolver-free execution observes the new instruction. Reclamation waits for
  the retained snapshot; shutdown releases executable mappings with the vCPU
  owners still alive.

  A second test warms real faultable callees and holds a caller invocation's
  epoch through a later-unit fault lookup. A channel-coordinated maintenance
  thread cannot finish the stop or expose the alias write until that invocation
  and its memory lease leave. Another OS thread rejects admission during Closing.
  The normal memory authority completes visibility while Closed; a retained
  snapshot delays only storage reclamation, not unlink. After its release,
  native-PC lookup no longer finds the old fault site. Replacement code executes
  through the surviving callers. This controls the fault-dispatch lifetime
  interval; it does not claim a concurrently executing hardware fault test.

  Source retirement is exercised over eight publication/warmup/retirement
  cycles with static and indirect callers on two vCPUs. Eviction requests
  registered after a Closed link batch was captured prevent that older batch
  from reopening admission. Draining removes the caller roots; retained static
  and dynamic preparations delay storage reclamation but cannot be registered
  or emitted afterward. A pending LCQ capture cannot publish across the stop.
  The surviving callee-to-continuation PIC entries still serve pending guest
  returns without Rust, and obsolete source handles remain invalid after reuse.

  Four mixed-root pressure cycles use real cache accounting to require segment
  reclamation, without allocating hundreds of MiB of test data. An unpublished
  bridge and compiler snapshot postpone storage release, not eviction or root
  removal. Dropping the last bridge releases both retained units, decommits the
  segment and returns usage below the soft limit. Republishing reuses the actual
  address with a new segment generation and unit identity; native-PC lookup
  resolves the new owner. After releasing the pressure charge, accounted usage
  returns to the same level each cycle, with both vCPU owners still registered.

  Synthetic HCQ family tests combine a static root and two vCPU PICs selecting
  old/new targets. Withdrawal cuts HCQ roots and releases baseline promises even
  after the optional installation quota is exhausted; the source stays on its
  fallback until deferred LCQ relinking runs. A mapping change affecting only
  the retained LCQ's extra coverage also retires its dependent HCQ, before the
  baseline, and clears both kinds of incoming roots. Republished LCQ receives
  fresh links; retained old snapshots delay shutdown storage release but not
  root removal. These are ownership fixtures, not an HCQ compiler.

  A production-chain test holds memory authority over an unrelated range while
  link work or shutdown joins the stop. Admission and acknowledgement remain
  blocked until that hold ends despite there being no code to invalidate. On
  ordinary completion, both vCPUs' existing static/PIC/return paths still execute
  without a resolver or unnecessary flush; shutdown instead releases the cache.

  **Validation:** 527 x86-64 JIT library tests pass (the deliberate fatal
  missing-native-PC subprocess is excluded). All 58 engine tests and both new
  synthetic mixed-root HCQ tests pass on AArch64/QEMU. JIT all-target Clippy,
  formatting and diff checks pass. No production or fork changes were needed.

  **Closure:** combined coverage and the existing static replacement,
  nonempty/far bridge, safety-failure and epoch tests satisfy this step's
  ownership and reclamation checks. Final production-cutover validation and
  handoff remain step 8; native Arm and BTI/CET hardware enforcement are not
  established by QEMU or these synthetic ownership fixtures.

  **Exit:** repeated churn returns real spans and registry/island slots within
  the configured budgets. No reachable target or fault metadata is reclaimed
  early, and shutdown leaves no link/PIC/bridge owner behind. Concurrency tests
  use controlled interleavings rather than sleeps. This step hardens hooks
  already required in steps 3–6; it does not authorize unsafe interim linking.

- [x] **8. Validate the production cutover and record the handoff.** Run JIT,
  runtime, memory and interpreter regressions affected by the changes, plus
  emitted-shape and native-boundary tests. Run formatting, all-target checks
  and relevant Clippy checks. Use [AArch64 tests](../../aarch64-tests.md) for
  cross-target execution; distinguish QEMU from native hardware and explicitly
  record missing BTI/CET enforcement or native Arm coverage. Validate changed
  fork code with its own focused tests as well as Nixe integration.

  Exercise es2gears and clean shutdown, then repeat the existing manual perf
  capture outside concurrent builds/tests. Compare resolved-edge overhead and
  observed throughput; sampled percentages alone do not establish FPS or final
  performance conformance. Do not create a benchmark framework or make Task 4
  responsible for the later full-architecture performance acceptance.

  **Pressure regression checkpoint:** `textured_cube` exposed repeated full
  registry scans during startup. Segment occupancy and pending-retirement
  accounting now use fixed per-segment counts, including detached records until
  actual reclamation. Collector cursors visit each registry slot once per pass.
  Eviction selects up to 64 oldest eligible units without heap allocation,
  drains their links in HCQ-before-LCQ order, and reclaims once per batch (also
  capped at one segment of selected native bytes). Pressure still triggers at
  512 MiB, but targets 480 MiB when references permit, preserving existing hard
  limits and nonblocking behavior for retained compiler/staging owners.

  Validation: 532 x86-64 JIT library tests pass with the intentional fatal
  missing-native-PC supervisor excluded; 220 lifetime tests pass on AArch64/QEMU.
  All-target JIT Clippy and formatting/diff checks pass. Focused regressions
  cover retained/staged spans, slot reuse, batch order, collector visits and
  headroom. No Cranelift changes were needed; the existing local override remains.
  In a 30-second headless capture, `textured_cube` now submits 641 frames;
  `decommit_unused` has no direct samples, versus 58.75% before. Victim selection
  accounts for 1.59%. These are sampled user cycles, not an FPS comparison, and
  the sandbox capture uses software Vulkan. Results are in local
  `dump/perf-cube-batched-75ehrr/`.

  The follow-up removes eager error construction/destruction from successful
  address alignment. The NVIDIA capture in `dump/perf-cube-align-U5NIhJ/`
  confirms that the per-span destructor call is absent from the emitted loop
  and has no direct samples (previously 5.25%). This does not establish a net
  speedup: the remaining free-span search still costs 13.98% in this capture.
  Keep that search and canonical GPU visibility as remaining profiling targets.

  CLI teardown now stops and joins GPU work before removing the CPU process,
  after guest execution has returned all scheduler leases. Pending GPU memory
  transitions therefore finish before the JIT coordinator closes admission.
  SIGINT exits with status 0 and no teardown errors on NVIDIA and software
  Vulkan, respectively in the capture above and
  `dump/cube-shutdown-writable-hsbyp4/`. Focused validation passes: 23 executable
  tests on each of x86-64 and AArch64/QEMU, 29 CLI tests, 6 GPU-owner tests,
  CLI/JIT all-target Clippy, formatting and diff checks. Native Arm validation
  remains pending; final step-8 evidence follows below.

  **Invocation overhead checkpoint:** `simplegfx` exposed gettid syscalls and
  unconditional condvar wakes on each data-cache-maintenance exit/reentry.
  `WorkerFaultContext` is now statically thread-bound (not Send/Sync), replacing
  its per-entry/exit TID queries; registration, signal dispatch and cold teardown
  retain their OS-TID handling. Released contexts still reject use before
  touching a reusable slot. Invocation destruction notifies only during Closing;
  the last shared memory reader notifies only with a pending transition. Both
  checks retain the existing predicate mutex and epoch-release ordering.

  The same 20-second windowed NVIDIA profile reduces inclusive samples in these
  four paths from 53.3% to 4.5% (`dump/perf-simplegfx-transitions-YvVGzb/`). Separate
  headless smoke runs deliver about 30 frames/s for simplegfx (previously 12),
  59 for es2gears and 59 for textured_cube; all exit 0 on SIGINT. These are guest
  frame-delivery rates, not window FPS measurements. Logs are in
  `dump/jit-transition-smoke-iEJFPC/`. Validation: 533 x86-64 JIT tests, 22 shared
  fault-runtime tests, the thread-affinity compile-fail test, 6 memory-gate tests
  and Clippy pass. AArch64/QEMU passes 220 lifetime tests and 21 fault-runtime
  tests; the 12 fatal subprocess scenarios pass via the explicit QEMU launcher
  instead of the system binfmt supervisor. Native Arm remains unverified.

  The unnecessary syscall/wakeup findings are resolved. Remaining candidate:
  each DC maintenance instruction still takes the full canonical Rust exit and
  readmission path. A cheaper path must preserve per-address faults and canonical
  GPU/CPU visibility, not turn maintenance into a no-op. No cache-instruction
  semantics or Cranelift code changed in this checkpoint.

  Remove superseded one-fragment assumptions and task-local adapters, retain
  genuine canonical misses/observations, and update the README/spec/handoff to
  describe the implemented path. Record tested fork revisions/local changes
  without claiming the old Git pin contains them. Do not commit or push unless
  requested by the maintainer.

  **Final validation and closure:** 533 x86-64 JIT library tests and 538
  AArch64/QEMU library tests pass, including emitted transfer/checkpoint shapes,
  source-keyed PICs, matched returns, inherited state, later-unit faults,
  replacement/unlink and bounded reuse. After removing the unused memory-exit
  result field, the full x86-64 suite and all 10 Arm invocation tests pass again.
  The excluded missing-native-PC supervisor is verified by running its child
  directly on each host: SIGSEGV and `reason=unattributed-native-pc` as required.

  Runtime (69 library tests), memory (60), interpreter (107), their enabled
  integration tests, both JIT differential tests and all 5 homebrew acceptance
  tests pass. Optional interpreter/QEMU oracle tests and caller-owned real-title
  tests remain ignored by their suites; no claim is made for those scenarios.
  Workspace all-target check, affected-crate all-target Clippy, formatting and
  diff checks pass. The local fork passes 42 Nixe codegen tests with
  `--features x86,arm64,disas` and all 45 reader library tests.

  Final NVIDIA capture: `dump/perf-task4-es2gears-headless-HIBUgQ/`, 20 seconds
  after first-frame warmup, without concurrent builds/tests. The run delivers
  about 59.7 guest frames/s and exits 0 on SIGINT. The earlier window capture
  ended prematurely and is explicitly excluded from closure evidence. The
  resolved-edge shape and zero-resolver assertions, not sampling percentages,
  establish the native-hit contract. Recent simplegfx/textured_cube smoke and
  shutdown results are recorded above. This closes Task 4 on the available
  hosts, not the specification's later full performance/hardware conformance.

  **Exit:** every Task 4 criterion has code and available-host test evidence;
  static, PIC and matched-return hits stay native, roots and storage are bounded,
  and there is one production JIT route. Task 5 receives the actual cold-poll
  source/destination/edge information without enabling its sampling policy here.

## Handoff to Task 5

There is one production JIT route: `engine` → LCQ publication/invocation with
static links, per-vCPU PICs and a guest-thread RSB. A native chain shares its
gateway, reader epoch and memory lease. Resolved static/PIC/matched-return
tests reject resolver calls; inherited-state, fault, invalidation and pressure
tests cover the same production owners. Cold semantic exits are intentional.
The unused memory-exit poll-result copy has been removed; completion and tests
consume the reconciled frame/slice budget instead.

Task 5 starts at `native/poll.rs`: sample-only deadlines currently rearm and
resume natively, without recording samples or scheduling compilation. Each
owned terminal retains `GuestExit` source PC/edge kind, the source version and
allocation map, and `TerminalTransfer` destination location/static key,
completed cost, poll patch and hot continuation. Use that exact source and
allocated dynamic target when activating sampling; never infer the source from
the initial invocation entry or destination PC. Keep table/queue policy off the
hot terminal path. No functional sampler, promotion workers or HCQ compiler
has been enabled by Task 4.

The local Wasmtime checkout is branch `nixe`, HEAD
`3dabafe6e5cb04b88265b0d3612f88ffe50b7332` plus uncommitted backend changes.
Continue using `/tmp/nixe-observable-fp-local.toml`; the manifest's older
`e2a984d96678207094c0fc50057c8b6bcfd68715` pin is insufficient. Publishing/pinning
the fork and reconciling `Cargo.lock` remain a separate maintainer action.
No commit, push or fork source edits were performed during this closure review.

Native AArch64 cache ordering and hardware-enforced BTI/CET remain unverified.
The available evidence is x86-64 execution, both encoders' landing/transfer
checks and AArch64/QEMU, not full hardware conformance. Remaining performance
candidates include allocator free-span search and canonical GPU visibility
work recorded in step 8; they are not delegated to HCQ automatically. The
frequent coherent CIVAC exit identified above is resolved by the follow-up below.

## Post-Task 4: coherent CIVAC fast path

Switch 1 CIVAC now uses a confined, trapping byte read through the existing
direct alias, with an allocation-visible PRE state map. Successful probes
continue within the fragment without canonicalization, helper calls or
epoch/lease readmission. The backing owner's existing alias protections prove
CPU-visible RAM; no additional coherence table is introduced. Other maintenance
operations and barriers remain canonical.

A failed probe escapes as an owned cache operation, not an ordinary guest
read. Original-address validation and GPU reconciliation run only after the
native epoch/lease ends, through `maintain_cache`. This retains the current
memory owner's permissions, remapping and error policy without adding MMIO
reads or load-repair retries. No Cranelift changes were needed.

Regression coverage includes exact byte-load emission on both hosts, repeated
unused probes, lazy flags and active FPSR, prefix/completion budgets, unmapped
and out-of-arena addresses, XZR, nonreadable RAM, MMIO rejection, and GPU
download/remap/error handling followed by native reuse of the restored alias.
Validation passes: all 539 x86-64 JIT library tests, the five CIVAC-filtered
tests and three data-cache tests on AArch64/QEMU, affected-crate all-target
Clippy, formatting and diff checks. Native Arm remains unverified.

NVIDIA window profile `dump/perf-simplegfx-civac-PI0J0S/` uses the same 8-second
warmup and 20-second capture as the preceding window profile, with no concurrent
builds/tests. The four previously dominant exit/admission functions fall from
about 32% combined self samples to about 0.24%; generated code now dominates.
The separate headless run `dump/simplegfx-civac-smoke-4EXv01/` delivers 766
frames from VSync 83 to 863, approximately 58.85 frames/s versus the prior
30.06. These are guest frame-delivery rates, not measured window FPS. Both
runs exit 0 on SIGINT. Raw captures and reports remain in `dump/`.
