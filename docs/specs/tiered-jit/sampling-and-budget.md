# Control budget and functional sampling

[Specification index](README.md) · [Implementation tasks](tasks/README.md) · [Open review items](review-status.md)

Related contracts: [Gateway, state transfer and helper ABI](native-abi.md); [Synchronous LCQ compiler](lcq.md); [HCQ admission, region formation and reshape](hcq.md).

## Control budget and functional sampling

Exact instruction observability is not part of the production JIT. Each vCPU
persists an approximate `sample_remaining`; each native invocation adds its
runtime `slice_remaining`. NativeFrame stores both plus `armed_span`, while the
pinned register carries their next positive minimum. Generated code therefore
pays for one counter and one not-taken deadline branch, not two budgets:

All four values use signed i64. After cold-path normalization,
`sample_remaining` is in `1..=4096`; a slice presented to the JIT is in
`0..=i64::MAX-2048` (the runtime splits a larger request), and between
checkpoints `poll_remaining` can undershoot by at most 2048. Arithmetic is
checked in debug builds and these bounds make it nonoverflowing in release.
Zero never enters native code.

```text
gateway:
  if slice_remaining <= 0: canonical BudgetExhausted
  armed_span = min(sample_remaining, slice_remaining)
  poll_remaining = armed_span

generated block checkpoint:
  poll_remaining -= static instructions completed on this path
  if poll_remaining <= 0: cold_poll(source, destination, edge_kind)

cold_poll or any canonical exit:
  spent = armed_span - poll_remaining
  sample_remaining -= spent
  slice_remaining -= spent
  if sample_remaining <= 0:
      emit one functional sample unless this is a forced control transition
      do sample_remaining += 4096 while sample_remaining <= 0
  process control/invalidation
  if slice_remaining <= 0 or control requires exit: canonical scheduler exit
  armed_span = min(sample_remaining, slice_remaining)
  poll_remaining = armed_span
  canonical-dispatch the exact destination continuation
```

Every LCQ fragment performs the subtraction/check immediately before its
terminal transfer. HCQ carries the counter in SSA, subtracts the executed
canonical block cost and checks at every backedge and external exit; an acyclic
forward path is bounded by the 2048-instruction unit ceiling. Negative
`poll_remaining` carries block overshoot into both balances and the repeated
addition preserves it across sample intervals. A forced transition discards a
crossed heat sample but still charges the approximate runtime slice. Any other
canonical exit commits the partial `spent` value before clearing the execution
epoch.

Every helper/control/fault statepoint records `budget_debit_before` plus
`budget_charge_current` (zero or one). The former is the exact completed guest
prefix since the preceding checkpoint; the latter is one for a guest
instruction completed/aborted by this cold path and zero for an asynchronous
control boundary or a terminal checkpoint which already applied its debit.
Before canonical exit the pair is subtracted exactly once and marked applied. A
native retry applies neither field; a multi-subaccess instruction still charges
current once. Forced control, shutdown and internal-fault escapes normalize
each crossed sample interval without emitting a tier sample. This convention is
shared by both tiers and the differential budget oracle.

The cold control path records the actual source BlockKey, destination guest PC
and edge kind, then either resumes or returns to the scheduler according to the
runtime-slice decision above. This reuses a required control boundary; it adds
no dispatch-slot load/store, atomic RMW or promotion branch to the normal link
path. Every target of an HCQ internal backedge checkpoint is a selected public
entry with canonical ingress and a current DispatchPayload. The nonexpired path
stays on its direct SSA edge; an expired path commits the exact destination PC,
leaves the epoch, and a later gateway dispatch either reenters that same current
HCQ label or its current LCQ. There is no unregistered private poll continuation
or accidental loss of the optimized region.

A crossed sample first binds the vCPU sample tables to the current ExecutionKey
as specified below, performs owner-side reconciliation, and is then classified
from the source/target ownership snapshot at that cold poll. Reconciliation
probes any existing HotSeed way for the source guest PC before applying the
ownership classification, so it also runs when that source now executes in
HCQ. With one nonblocking JIT-state acquisition it reconstructs the complete
BlockKey/AdmissionFingerprint and compares the way's exact versions and compact
pending ActiveBuildHandle against ActiveBuildTable,
DispatchPayload.BuildFingerprint.admission and NegativeBuildCache:
AdmissionReserved, Queued or Building keeps it; a preferred payload whose
admission fingerprint matches clears it; a matching
negative keeps it; and a stale version/fingerprint or terminal cancellation
clears/lowers it by the retry rule below. Try-lock failure leaves it unchanged.
Only after reconciliation is HotSeedTable updated for the new sample, and only
when the source is a current,
unowned LCQ seed. An edge with exactly one or two CurrentStable owners and at
least one different-owner/LCQ endpoint updates BoundaryTable. A same-family
CurrentStable HCQ internal edge updates neither table, and a
pending/superseded/invalidating owner creates no admission evidence. Control and
slice accounting still run in every case; HCQ never tries ordinary admission
for an already-owned internal seed.

Each vCPU owns one `TierSampleTablesV1` containing one HotSeedTable of exactly
256 sets by four ways and one BoundaryTable of exactly 64 sets by two ways.
Its header is
`{ execution_key_present:u8, reserved_zero:[u8;7],
execution_key:ExecutionKey, sample_sequence:u64 }`. A cold poll first compares
the complete current ExecutionKey. If the presence bit is zero it zeroes both
tables and binds the key. If the key differs it releases any table-local compact
handles (which never cancel or otherwise mutate the independently owned
ActiveBuild), zeroes both tables and replacement state, writes the new key and
sets `sample_sequence = 0`. Only that vCPU mutates these non-atomic bytes;
process migration cannot corrupt them and may only delay promotion. The
gateway performs this binding before entering code for a different
ExecutionKey, so generated PIC code relies on the same current binding without
a hot-path header comparison.

The frozen compact records are:

```text
HotSeedSuccessorV1 = repr(C, align(8)), size 24 {
    target_guest_pc:u64, last_sample_sequence:u64,
    occupied:u8, edge_kind:EdgeKind, count:u8, reserved_zero:[u8;5]
}
HotSeedWayV1 = repr(C, align(8)), size 192 {
    occupied:u8, score:u8, reserved_zero0:[u8;6],
    source_guest_pc:u64,
    source_dispatch:DispatchSlotHandleV1,
    source_reachability:u64, source_lcq_code_version:u64,
    hcq_shape_version:u64, instruction_observation_cursor:u64,
    last_sample_sequence:u64,
    successors:[HotSeedSuccessorV1;4],
    pending_build:CompactOptionalActiveBuildHandleV1
}
```

`CompactOptionalActiveBuildHandleV1` is all-zero for None; for Some its
`cell_index_plus_one` is the checked ActiveBuildTable index plus one and both
u64 fields are nonzero and exactly match that cell. No HotSeed way contains
BlockKey, ExecutionKey, AdmissionFingerprint, BuildFingerprint or an owning
FingerprintPin. [Task 0](tasks/00-baseline.md) asserts the displayed sizes and
`size_of(TierSampleTablesV1)`, and the `hot_seed_tables` layout row charges all
1024 ways plus the per-vCPU header/replacement bytes exactly.

The full seed BlockKey is reconstructed only as
`header.execution_key + source_guest_pc`; every successor BlockKey is the same
header key plus its `target_guest_pc`. Before using either, cold code validates
the stored DispatchSlotHandle, ReachabilityVersion, LCQ CodeVersion,
HcqShapeVersion and observation cursor against the current authorities. A
mismatch clears the way and treats the current sample as a miss. A target with
a different ExecutionKey is unresolved for these tables and changes only
ordinary control/slice accounting. Set index remains StableEncode FNV-1a-64 of
the reconstructed BlockKey masked by 255 and lookup compares the source PC
after the single header-key equality check. An empty way is used first;
otherwise it
replaces the lowest score, then oldest sequence, then lowest way. A matching
sample increments the score. One eligible cold poll first obtains exactly one
sample sequence S: if the current value is u64::MAX it clears HotSeedTable and
BoundaryTable, resets the counter to zero, then checked-increments to one;
otherwise it checked-increments once. A HotSeed miss selects its way, writes the
source PC, validated dispatch handle and all four authoritative versions,
`score = 1`, `last_sample_sequence = S`, an all-empty successor array and no
pending build. A match sets `score = min(8, score + 1)` and always writes S.
For `Sequential`, `DirectTaken`, `ConditionalTaken`,
`ConditionalFallthrough` or `IndirectBranch` it then updates the matching
successor to `min(255, count + 1)` and S, or chooses the specified successor
victim and initializes `count = 1` and S. `DirectCall`, `IndirectCall` and
`Return` may increment seed heat but never read, occupy or evict a successor
slot. Only
after the whole record is updated does score eight attempt one HCQ admission.
After it has claimed and completely initialized an ActiveBuild cell, admission
writes that cell's compact handle into the local way immediately before the
release publication of AdmissionReserved. Every failure before that publication
clears the compact handle; after publication the ActiveBuild state, not the
table field, owns the build. Queue
contention or fullness keeps score seven so the next real sample retries.
Invalidation never writes another vCPU's non-atomic table; it advances the
authoritative ReachabilityVersion/HcqShapeVersion, and that table's owner lazily
clears the mismatch at its next cold-poll reconciliation.

After successful enqueue the score remains saturated. At a later cold poll the
vCPU derives exact status from ActiveBuildTable, DispatchPayload and
NegativeBuildCache before incorporating that poll's sample. Matching current-
epoch `AdmissionReserved|Queued|Building` preserves the record and sets
`admission_suppressed_for_this_poll = true`; a preferred payload with that
fingerprint clears the local record and sets the same flag; and a matching
negative entry preserves the record and sets the same flag. No match after
stale/cancelled work is `Idle`: reconciliation changes only `score = 8` to
`score = 7`, leaves `admission_suppressed_for_this_poll = false`, and does not
admit yet. The normal sample update then incorporates the current source,
successor and sequence, raises seven back to eight, and the one generic
threshold branch makes exactly one admission attempt from that updated record.
There is no second Idle-specific attempt. An anchor ReachabilityVersion or
fingerprint mismatch instead clears the record before processing the sample,
so the current sample creates a new score-one record and cannot admit. A worker
never writes another vCPU's table, and every terminal outcome therefore has
one explicit bounded re-admission path whose AdmissionSnapshot includes the
current sample.

A successor slot key is reconstructed target BlockKey plus stored edge kind,
covering static and dynamic observations; it also stores a u8 count saturating
at 255 and checked u64 last-sample sequence. Occupied slots are packed from
index zero through the first empty slot and all later slots are zero. A matching
key increments its count; a miss
uses an empty slot or replaces lowest count, then oldest sequence, then lowest
slot. At admission, occupied slots whose kind is `Sequential`, `DirectTaken`,
`ConditionalTaken`, `ConditionalFallthrough` or `IndirectBranch` are normalized
by descending count, then newer sequence, then ascending StableEncode of their
`ObservedSuccessor`; excluded kinds do not consume one of the four output
positions. AdmissionSnapshot
is built under one successful nonblocking JIT-state acquisition and contains
seed BlockKey, ReachabilityVersion, current LCQ CodeVersion, ExecutionKey,
HcqShapeVersion, the executable-content observation cursor acquire-loaded in
that same critical section, and only that ordered list of at most four full
successor keys. Raw counts/sequences are not copied. The full list, not only its
hash, becomes part of BuildFingerprint. A failed try-lock changes no snapshot
and keeps the score at seven. A worker never reads another vCPU's HotSeedTable.

Each vCPU also owns a 64-set, two-way BoundaryTable. Its full key is the named
`BoundaryObservationKey { source, target }`. At the cold observation point
an endpoint is HcqEndpoint only when owned by the exact CurrentStable family;
otherwise it is LcqEndpoint only when the exact BlockKey already has a current
LCQ payload. Endpoint resolution occurs in the same nonblocking JIT-state
critical section as family classification. A missing/stale target payload, a
non-root-admissible owner, unequal ExecutionKeys or any changed generation makes
the observation unresolved: it changes neither table and creates no snapshot.
In particular, an HCQ edge to a first-seen cold target is not misclassified as
ordinary evidence and never fabricates an LCQ version. Canonical fallback may
compile that target; only a later real sample can then name its resolved LCQ
endpoint. The
set of distinct HCQ family generations across both resolved endpoints must contain
exactly one or two members; a pair with neither present is ordinary LCQ seed
evidence and is not inserted here. Its u8 score saturates at four and uses the same
empty/lowest-score/oldest-sequence/lowest-way replacement rule; set index is
FNV-1a-64 of the full key masked by 63. A miss initializes `score = 1` and
`last_sample_sequence = S`, while a
match writes `min(4, score + 1)` and S; threshold evaluation occurs after that
write and uses the same one S allocated for this cold poll. Four samples
copy an immutable ReshapeSnapshot under one successful nonblocking JIT-state
acquisition. It contains that full directed endpoint identity, their common
ExecutionKey, the distinct participating
`ParticipatingFamily` values sorted by HcqFamilyId,
current HcqShapeVersion and the executable-content observation cursor loaded in
the same critical section. Raw score/sequence are excluded; failure keeps score
three. Queue
contention keeps score three. This table covers HCQ-to-LCQ, HCQ-to-HCQ and a
retained-LCQ entry into an HCQ-owned instruction, so a valid reshape names
exactly one or two distinct current families. Zero matching families is stale
and a useful LCQ-only region returns to ordinary seed admission. This is the
only late-entry/reshape counter.

Rejected HCQ results do not live in either per-vCPU table. The process owns a
1024-set, four-way `NegativeBuildCache` under the JIT-state mutex. Its index key
is the complete ordinary-or-reshape AdmissionFingerprint; hash matches are
followed by full-field equality and set index is its FNV hash masked by 1023.
Its value stores the finalized BuildFingerprint plus one of `OverInstructionCap`,
`DisconnectedMandatoryEndpoints`, `BackendSpillCapacity`, `SegmentCapacity` or
`IslandCapacity`. An empty way is used first; otherwise
replacement selects the numerically lowest `instruction_observation_cursor`
(oldest evidence), then oldest checked creation sequence, then lowest way. A
worker may insert only after capturing OwnershipSnapshot and must have
`value.BuildFingerprint.admission == index_key`; a vCPU performs only a
nonblocking exact AdmissionFingerprint lookup during the already-cold admission attempt. Contention
leaves ordinary score seven or boundary score three. A matching negative entry
retains the saturated score and suppresses enqueue; any fingerprint component
change may retry. Temporary queue/cache pressure, stale input and ownership
races never create a negative entry. On negative-cache creation-sequence
exhaustion, its mutex owner clears all 4096 ways and restarts that nonsemantic
sequence at one.

Suppression lasts only while the exact entry remains resident in this bounded
cache. Deterministic replacement may evict it and permit a later real sample to
retry an unchanged shape; the specification does not promise process-lifetime
negative memory.

Sample-sequence overflow has no semantic meaning. At u64::MAX, the owning vCPU
clears both bounded tables at the same cold poll and restarts its sequence at
one. Identity/version counters never use this reset rule. On a host configured
with zero HCQ workers, scores may saturate but admission is disabled and no
queue is allocated.

No timer, log, report, histogram or public API exposes this state.
