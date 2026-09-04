# Direct memory and fault authority

[Specification index](README.md) · [Implementation tasks](tasks/README.md) · [Open review items](review-status.md)

Related contracts: [Native fault transport and completion](faults.md); [Code and mapping invalidation](invalidation.md); [Maintenance records, cleanup and mapping requests](maintenance-records.md).

## Direct memory and fault authority

The supported Linux hosts reserve one flat, guarded virtual arena per guest
address space. Guest mappings and permissions are represented by host mappings
and page protections; that representation is the sole authority on a normal
CPU access. Reverse physical aliases and the dependency index are cold
transition data, not generated-access lookup structures.

The native ABI pins the arena base in `r13` on x86-64 and `x19` on AArch64.
After the architecturally required guest-address calculation, an eligible RAM
operation contains only the minimum confinement needed to prevent an arbitrary
guest value escaping the reserved arena, base addition and the native
load/store/atomic. HCQ may eliminate or hoist redundant confinement when its
proof covers every path. There is no generated page-table, permission,
observer, backing-kind or ownership lookup and no eager architectural-state
checkpoint.

The arena size is a nonzero 4096-byte multiple fixed when the address space is
created, included in its execution identity and guaranteed not to overflow
`arena_base + arena_size`. For a direct access of S bytes, lowering computes
`last = guest_address + (S - 1)` and takes the typed cold path on u64 overflow,
`last >= arena_size`, or
`(guest_address >> 12) != (last >> 12)`. An atomic/exclusive access also takes
the cold path unless `guest_address & (S - 1) == 0`. Only after those conditions
does it form `arena_base + guest_address` and issue the one native instruction.
These comparisons are the required confinement/restartability minimum; neither
tier may omit them without a path-wide proof of the same predicates.

A direct memory operation is eligible only when one restartable host
instruction implements the required size, endian transform, atomicity and
ordering and cannot expose a partial effect before a host fault. An access which
can cross a 4096-byte protection granule is always ineligible for this direct
shape. Every other access uses its instruction-specific typed
cold path; this choice is made during lowering and adds no generated page-table
probe.

Every accepted memory instruction has one immutable `MemoryEffectPlan`, shared
by the interpreter, LCQ, HCQ and fault resolver. It fixes effective-address and
value recipes, ordered subaccesses, sizes/alignment/endian transformations,
load-result and base writebacks, exclusive-monitor effects, atomic/barrier
ordering, architecturally visible commit stages, data-abort state and the exact
successful continuation BlockKey. Each plan is classified as exactly
`SingleRestartable`, `PrefixVisible` or `AllOrNothing`; the detailed lowering
rules are in [native fault transport](faults.md). There is no generic "access
succeeded" writeback rule.

The [Task 0](tasks/00-baseline.md) declarative authority is the following closed schema. Every shown
enum is `repr(u8)` with the listed discriminant; arrays use declaration order,
have a u32 StableEncode length, and contain no implicit callback or Rust
closure:

```text
MemoryEffectClass = SingleRestartable=1 | PrefixVisible=2 | AllOrNothing=3
PlanAccessKind = Load=1 | Store=2 | AtomicRmw=3 |
    ExclusiveLoad=4 | ExclusiveStore=5 | Prefetch=6
PlanEndian = Little=1
PlanOrdering = Plain=1 | Acquire=2 | Release=3 | AcqRel=4 |
    SeqCst=5 | RcpcAcquire=6
AtomicOpV1 = None=0 | Swap=1 | Add=2 | Clear=3 | Xor=4 | Set=5 |
    SignedMax=6 | SignedMin=7 | UnsignedMax=8 | UnsignedMin=9 |
    CompareSwap=10
MemoryWriteCondition = Always=1 | CompareEqual=2 | ExclusiveMonitorPass=3
PlanAlignment = UnalignedAllowed=1 | NaturalRequired=2 |
    ExplicitPowerOfTwo=3
MonitorEffect = Preserve=1 | Set=2 | Clear=3 |
    ConditionalStoreStatus=4
BarrierEffect = None=0 | Load=1 | Store=2 | Full=3 | Instruction=4
StateWriteKind = Replace=1 | MergeLowBitsPreserveHigh=2 |
    MergeHighBitsPreserveLow=3
StateWriteRole = Ordinary=1 | LoadResult=2 | BaseWriteback=3 | Status=4
DataAbortSyndromeClass = Scalar=1 | AcquireRelease=2 | Atomic=3 |
    Exclusive=4 | Vector=5
ContinuationKind = NextPc=1 | ExpressionPc=2

PlanOpV1 =
    ReadState { component:StateComponent }=1 |
    ReadDecodedOperand { operand_ordinal:u8, bit_width:u16 }=2 |
    ReadTemp { temp:u8, bit_width:u16 }=3 |
    Constant { bit_width:u16, little_endian_bits:[u8;16] }=4 |
    AddWrapping { bit_width:u16 }=5 | SubWrapping { bit_width:u16 }=6 |
    And { bit_width:u16 }=7 | Or { bit_width:u16 }=8 |
    Xor { bit_width:u16 }=9 | ShiftLeft { bit_width:u16 }=10 |
    LogicalShiftRight { bit_width:u16 }=11 |
    ArithmeticShiftRight { bit_width:u16 }=12 |
    ZeroExtend { from_bits:u16, to_bits:u16 }=13 |
    SignExtend { from_bits:u16, to_bits:u16 }=14 |
    Truncate { from_bits:u16, to_bits:u16 }=15 |
    Extract { source_bits:u16, low_bit:u16, result_bits:u16 }=16 |
    Concat { low_bits:u16, high_bits:u16 }=17 |
    ByteReverse { bit_width:u16 }=18 |
    Select { bit_width:u16 }=19 |
    EqualZero { bit_width:u16 }=20
PlanExprV1 { result_bits:u16, ops:[PlanOpV1; 1..=64] }
StateWriteV1 {
    destination:StateComponent, kind:StateWriteKind, role:StateWriteRole,
    value:PlanExprV1
}
CommitStageV1 {
    ordinal:u8, after_successful_subaccess_count:u8,
    state_writes:[StateWriteV1; 0..=64],
    monitor_effect:MonitorEffect, barrier_after:BarrierEffect
}
SubaccessV1 {
    ordinal:u8, address:PlanExprV1,
    access:PlanAccessKind, byte_width:u8,
    alignment:PlanAlignment, explicit_alignment:u8,
    endian:PlanEndian, ordering:PlanOrdering,
    write_value:OptionalPlanExprV1,
    atomic_operand:OptionalPlanExprV1,
    compare_value:OptionalPlanExprV1,
    atomic_op:AtomicOpV1, write_condition:MemoryWriteCondition,
    result_temp:OptionalU8, status_temp:OptionalU8
}
DataAbortRecipeV1 {
    fault_address:PlanExprV1, access:PlanAccessKind,
    byte_width:u8, alignment_fault_precedes_permission:bool,
    syndrome_class:DataAbortSyndromeClass
}
FaultRuleV1 {
    faulting_subaccess:u8, committed_stage_count:u8,
    abort:DataAbortRecipeV1
}
ContinuationV1 {
    kind:ContinuationKind, pc:OptionalPlanExprV1
}
MemoryEffectPlanV1 {
    plan_id:MemoryEffectPlanId, class:MemoryEffectClass,
    decoded_operand_count:u8, temp_count:u8,
    base_component:OptionalStateComponent,
    prefault_state:[StateComponent; 0..=64],
    subaccesses:[SubaccessV1; 0..=32],
    commit_stages:[CommitStageV1; 0..=33],
    fault_rules:[FaultRuleV1; 0..=32],
    barrier_before:BarrierEffect,
    success_continuation:ContinuationV1
}
```

`PlanExprV1` is a verified, loop-free RPN stack program over bitvectors of
width 1..=128. Reads push one value; unary/binary/select operations pop their
declared operands and push exactly their declared result; arithmetic is modulo
2^bit_width. For a count greater than or equal to width, left/logical-right
shift yields zero and arithmetic-right yields all sign bits; an instruction
whose A64 count is masked must encode that mask explicitly before the shift.
No operation invokes host-language undefined behavior. Constant bytes above
`ceil(bit_width/8)` and high unused
bits are zero. For every binary operation `rhs = pop()` then `lhs = pop()` and
the result is `lhs op rhs`; shifts use rhs as an unsigned u128 count without
host truncation. Concat pops high then low and produces `high || low`; Extract
pops its source; Select pops false, true and then the one-bit condition.
Declared operand widths must exactly equal the popped widths (apart from the
explicit extend/truncate op) or verification fails. Verification
requires no underflow, maximum stack depth 16 and exactly one final value of
`result_bits`. `prefault_state` is sorted by the manifest StateComponent
numeric encoding, unique, and equals the complete set of ReadState operands
needed before the first potentially faulting subaccess. Decoded operand and
temp ordinals are zero-based and in range.

Subaccess ordinals and commit-stage ordinals start at zero and are contiguous.
Legal byte widths are 1, 2, 4, 8 or 16. An address expression returns
64 bits. Load/ExclusiveLoad requires no write/atomic/compare value and one
result temp. Store requires a same-width write_value, AtomicOp None, Always and
no result/status temp. AtomicRmw requires an old-value result temp, one same-
width atomic_operand, a non-None AtomicOp, and Always except CompareSwap;
CompareSwap additionally requires compare_value and CompareEqual, writes the
operand only on equality and always returns the old value. ExclusiveStore
requires a write_value, AtomicOp None, ExclusiveMonitorPass and a status temp
which becomes exactly zero when memory is written and one when the monitor
rejects it; its memory effect and every success-only stage share that same
condition. ExclusiveLoad sets the declared monitor only after a successful
read. Prefetch has no values/temps, AtomicOp None, Always and cannot commit
architectural data. Every arm-illegal field is None/zero, so the evaluator
never consults an instruction-name-specific rule. Natural alignment means
byte_width; ExplicitPowerOfTwo requires a nonzero power of two at most 16 and
the catalog row proves the precise Arm rule for a pair/multi-subaccess;
UnalignedAllowed requires explicit_alignment zero.
Nintendo Switch profiles use only Little; another endianness is a format-
version change.

Each CommitStage's trigger is monotonic and in 0..=subaccess_count; stages with
the same trigger execute in ordinal order. A stage applies its writes in array
order to a local next-state image and publishes them simultaneously, then
applies its monitor/barrier effect. Memory visibility occurs at the named
Subaccess, never implicitly at a state-write stage. Every faulting subaccess has
exactly one same-ordinal FaultRule. `committed_stage_count` equals the number of
earlier triggered stages and fixes the complete visible prefix; the abort's
remaining FSC/level bits come only from the current memory authority, while
all instruction-derived syndrome fields come from DataAbortRecipeV1.
Continuation NextPc has no pc expression and means checked `current_pc + 4` in
the identical ExecutionKey; ExpressionPc requires a 64-bit expression. The
result must be four-byte aligned and constructs exactly one canonical BlockKey.

SingleRestartable has exactly one subaccess, no stage triggered before one,
and every fault rule has committed_stage_count zero. PrefixVisible has at least
two subaccesses; before faulting subaccess ordinal j it exposes exactly stages
whose `after_successful_subaccess_count <= j`, because exactly j earlier
subaccesses succeeded. AllOrNothing either represents a barrier-only plan with zero
subaccesses, or preflights every address/permission/fault before stage or
subaccess effects; every failure has committed_stage_count zero. Its commit
uses the one memory-transaction mutex when more than one host operation is
needed. Any plan violating these class rules, exceeding a bound, using an
unread state/temp, leaving a temp unwritten before read, or giving two writes to
the same StateComponent in one stage fails [Task 0](tasks/00-baseline.md).

The native statepoint records the plan ID, faulting subaccess ordinal,
committed-stage count and exactly the prefault state/temporaries live at that
point. Interpreter, LCQ, HCQ and resolver call one generated evaluator for
PlanExpr/stage/fault/continuation semantics; target lowering may inline it but
must pass differential vectors against that evaluator. For each catalog row,
`subaccess_count`, `commit_stage_count` and `has_base_writeback` are derived,
not authored twice. `base_component` is None iff no BaseWriteback role exists;
otherwise exactly one StateWrite has that role and writes that component, and
`has_base_writeback` is true. A role/component mismatch is rejected.
`effect_table_hash_sha256 = lowercase_hex(SHA-256(
"nixe-memory-effect-plan-v1\0" || StableEncode(MemoryEffectPlanV1)))`.
[Task 0](tasks/00-baseline.md) stores a golden StableEncode/hash vector for every production plan and
rejects duplicate IDs or a catalog value not derived from that exact plan.

An arena host fault has exactly one typed disposition selected from the native
record plus current memory-authority generations:

- `RetryTrackedRam`: permitted ordinary non-executable RAM whose store fault is
  solely the current `DirtyTrackingArm` or `ResidualReadOnlyArm`; perform the one local monotonic
  repair and retry the identical native PC;
- `CompleteEmulatedAccess`: valid emulated non-RAM; reconstruct the named
  prefault/commit stage, execute the remaining MemoryEffectPlan suffix exactly
  once and continue by canonical dispatch after the guest instruction;
- `CompleteObservedCodeWrite`: permitted RAM with a live executable observation;
  become quiescent, invalidate/unlink dependents, execute the remaining write
  plan exactly once and continue canonically, never at the old native PC;
- `RaiseGuestDataFault`: unmapped, alignment-invalid or permission-invalid guest
  access; reconstruct and deliver the exact Arm data abort without a forbidden
  effect; or
- `InternalFault`: outside-arena address, unattributed/mismatched native PC,
  nested resolver fault, impossible authority state or invalid state map.

"Continue after the instruction" always means a fresh canonical dispatch to
the plan's continuation BlockKey. Only `RetryTrackedRam` restores a captured
native PC. Ordinary valid RAM mappings/protections are installed before Open;
an unexpected missing or stale host mapping is not silently repaired as a
tracking fault.

Every host mapping/PTE protection, backing-kind, guest-permission,
physical-alias and tracking-rearm change uses Closed. There are exactly two
outside-Closed exceptions. First, a resolver may perform the atomic
armed/read-only -> dirty/writable transition when the authority proves the same
mapping generation, valid permission, ordinary RAM, no executable observation
or physical-code dependency, and no mapping/alias/backing field change. It does
not advance mapping generation, and racing repairs converge on the same
writable state. Second, last-lease release may make the metadata-only arm-reason
transition specified below; it does not change any PTE. Every other
reconciliation becomes a typed MappingChange.
Direct-interpreter memory stubs are pinned CodeUnits in the same native-PC
directory and use the same plans/dispositions. A checked backend is a process
selection for unsupported hosts and differential tests, never a per-access or
per-instruction JIT fallback.

Instruction fetch/copy uses an `ExecutableObservationLease` owned by the memory
authority. LCQ may make a provisional versioned copy only to discover the
candidate extent and page set; provisional bytes cannot be lowered or
published. It then requests leases in ascending physical-page identity. HCQ
clones the leases and immutable images of its selected LCQ units rather than
reading live guest memory. Acquiring the complete set does all of the following
as one versioned operation:

1. identifies every physical page and every current virtual alias from which
   instruction bytes will be copied;
2. ensures that every CPU-writable direct alias of those physical pages is in
   the authority's tracked read-only state before any resulting CodeUnit can
   become reachable;
3. after any arming, recopies the exact bytes and returns physical-page
   identities, per-alias mapping/protection generations, page content and
   observation generations, and the global invalidation cursor; and
4. owns one live lease per page from final copy until publication abort or the
   corresponding published CodeUnit becomes Unlinked. Its immutable observation
   record remains in CodeUnitMetadata until reclamation.

Final lease acquisition starts and ends in the same Open admission epoch; it
returns `Transitioning` and no live lease if either phase/epoch check differs.
It does not hold an OpenToken across decoding. Closing invalidates matching
staging leases as well as published dependencies before its memory operation,
so a compiler cannot acquire a new observation between dependency freeze and
an executable write. LCQ retries after the next Open; HCQ completes stale.

If step 2 requires changing host protections, acquisition returns `NeedsArm`
without final bytes. The caller requests MappingChange; Closed arms every
reverse alias, advances the protection/observation generations and restarts the
entire acquisition. If final decode touches an unleased page, LCQ discards that
copy and restarts with the exact superset; it never incrementally lowers mixed
snapshots. The authority invalidates all affected live leases before any later
write. A staging compiler then fails publication validation and releases its
lease; a published dependent also follows mandatory unlink. Protection changes
hold no compiler, JIT-state or code-cache lock. Publication compares every byte
and every returned identity/generation, not merely the global cursor. No
compiler reads live guest memory after accepting the final image.

Dropping a staging lease on abort and dropping a published lease at Unlinked
use the same serialized memory-authority refcount. If the last executable lease
drops outside Closed, the authority does not silently leave an orphan
observation arm or make a mapping writable. Without changing the still-read-only
PTE, it atomically replaces `ExecutableObservationArm` with
`DirtyTrackingArm` when an ordinary dirty-tracking reason remains, otherwise
with `ResidualReadOnlyArm`, and advances protection/observation generations.
The next permitted ordinary-RAM write takes `RetryTrackedRam`; that resolver
atomically consumes the exact arm generation, records dirty state when the arm
was `DirtyTrackingArm`, makes the alias writable and retries. A stale contender
observes the completed writable generation and converges without a second
effect. In Closed, last-lease release instead recomputes every alias protection
immediately from all remaining permission/tracking reasons, so
`ResidualReadOnlyArm` cannot survive reopen. A concurrent new lease is
serialized on the refcount and must revalidate the advanced generations before
final copy; it either replaces either nonexecutable arm with a new
`ExecutableObservationArm` while retaining the read-only PTE, or restarts.

A CPU write fault on an observed page takes
`CompleteObservedCodeWrite` even when dirty tracking also applies. Closed cuts
all dependent roots, invalidates their leases, performs the plan suffix once,
advances content/protection/observation generations and leaves every alias
writable only when no newer observation was installed. A later compilation
rearms it. Interpreter, DMA, GPU, loader, debugger and service writers must call
the same memory-authority API before changing the backing page; it performs the
same close/unlink/generation transaction. Bypassing that API is an internal
failure. No writable alias of valid observed code exists outside this authority.
