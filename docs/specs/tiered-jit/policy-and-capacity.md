# Policy defaults, layouts and safety bounds

[Specification index](README.md) · [Implementation tasks](tasks/README.md) · [Open review items](review-status.md)

Related contracts: [Supported baseline and generated manifest](baseline.md); [Executable cache, metadata and backend ownership](cache.md); [Cohort workspace, arbitration and mutation plans](cohort.md).

## Policy defaults and safety bounds

These are explicit initial production constants. They are policy rather than
semantic invariants, but the implementation contains one value for each and no
runtime tuner or title override. `KiB` means exactly `2^10` bytes and `MiB`
means exactly `2^20` bytes. Every size, count, product, page rounding and byte
offset below uses checked integer arithmetic during process construction;
overflow fails construction before any guest starts.

| Policy | Value |
| --- | ---: |
| LCQ unit | one natural basic block |
| LCQ hard safety ceiling | 512 guest instructions |
| HCQ total instruction ceiling | 2048 unique guest instructions |
| control/tier sample interval | 4096 approximate guest instructions |
| samples required for an HCQ seed | 8 |
| sampled cross-owner edges required for reshape | 4 |
| hot-seed storage | 256 sets x 4 ways per vCPU |
| sampled successors retained per seed | 4 |
| boundary storage | 64 sets x 2 ways per vCPU |
| live dispatch-slot ceiling | 1,048,576 |
| HCQ worker count | 0 when logical_cpus <= 2; otherwise min(4, max(1, (logical_cpus - 2) / 2)) |
| pending HCQ capacity | 8 x worker count |
| pending selection | seven newest jobs, then one oldest job |
| indirect dispatch cache | 2048 sets x 2 ways per vCPU |
| configured address spaces | construction parameter in 1..=64 |
| dynamic-bridge weak-index capacity | 4096 slots x configured maximum vCPU count |
| rooted dynamic bridges | at most 4096 x currently active vCPU count |
| software return-stack size | 16 entries per guest thread |
| executable address envelope | 2047 MiB |
| usable executable segments | 127 x 16 MiB = 2032 MiB |
| executable segment size | 16 MiB |
| link-island reservation per segment | 64 KiB |
| reclamation and HCQ-reopen watermark | 512 MiB |
| HCQ reservation ceiling | 608 MiB |
| JIT charged-total hard limit | 640 MiB |
| LCQ emergency reserve | 32 MiB |
| gateway helper-stack reserve | 4096 bytes per active native invocation |
| signal/fault stacks | per configured vCPU: variable 64..256 KiB SignalAltStack, one 64 KiB landing stack and three guard pages |
| global negative-HCQ cache | 1024 sets x 4 ways per process |
| dynamic-bridge unit quota | 8192 permits x configured maximum vCPU count |
| LCQ compiler policy | opt_level=none, single_pass |
| HCQ compiler policy | opt_level=speed, backtracking |

The executable reservation base is 16 MiB aligned and owns one 2047 MiB
`PROT_NONE` RX envelope. Addresses in `[base, base + 127 * 16 MiB)` map to
segment indices 0..126; the final 15 MiB remain
permanently inaccessible and are neither a partial segment nor allocatable
capacity. This keeps all executable-cache addresses inside one sub-2-GiB
envelope while making segment count and directory size exact. The RW alias is
outside this envelope, is never a branch target and has no native-PC slot.

`configured_max_vcpus` is an immutable process-construction parameter in
`1..=128`; construction fails before starting a guest when the requested value
is zero or above 128. There are always exactly `configured_max_vcpus`
preassigned epoch/PIC/fault record slots. `active_vcpus` is the number of those
slots which still own the checked registration count: `Active`,
`Deactivating`, or `TerminalDeactivating(was_counted_active = true)`. Only
`Active` is schedulable; a Deactivating slot remains counted until its one
Inactive/TerminalInactive commit decrements it. The count is not the number of
allocated records and never exceeds the configured value. Every table
described as "per configured maximum vCPU" reserves its exact maximum storage
at construction, charges that storage as committed pages plus outstanding
fixed reservations, and is never resized. At 128 vCPUs the weak index has
524,288 slots and the bridge-unit quota has 1,048,576 permits. Every bridge is
a normal generational UnitHandle in the common unit registry; this quota is a
precharged permit bitmap/counter, not a fourth object registry.

`configured_max_address_spaces` is a second immutable construction parameter
in `1..=64`. The process reserves a 64-slot generational AddressSpaceRegistry
and makes only indices below that configured value eligible. Each eligible slot
contains one immutable arena descriptor, one MappingRequestPermitCell, one
MappingChange MaintenanceRecord/result cell and the fixed memory-authority
roots needed by terminal scan. `AddressSpaceHandle { slot_index: u32,
slot_generation: u64 }` is validated before obtaining its immutable nonzero-u64
AddressSpaceId. Under the address-space registry mutex, the exact lifecycle is
`Free(g) -> Active(g, id) -> Destroying(g, id) -> Free(g + 1)`; no lock-free
reader dereferences the slot without a NormalResultToken plus OpenToken or a
root/pin acquired while one of those protects it. Normal destruction is a
safety MappingChange: Closed removes every dispatch/code/mapping/vCPU
association first, then the owner waits outside subsystem mutexes for pins and
mapping-result references, commits Free and increments the generation.
Construction beyond the configured count returns
`AddressSpaceCapacityExceeded` before allocating an ID. Shutdown makes every
Active slot Destroying and, after its associations and pins drain, changes each
Destroying slot to `TerminalDestroyed(g, id, cause)` without advancing g; Free
slots remain Free. These are the only terminal slot states. It scans the fixed
64 slots in numeric order. There is no
unbounded address-space map or per-request MappingChange record.

TerminalControl contains a persistent `arena_admission_gate` using the
`CountedCloseGateV1` protocol defined under Open admission, a four-AtomicU32
`arena_admission_bitmap`, one bit for each possible vCPU slot, and immutable
`configured_max_vcpus`. A wire VcpuHandle exposes its slot index and generation
without dereferencing the reclaimable arena. Before an initial caller can touch
that arena it validates the index below the immutable maximum, acquires one
count from `arena_admission_gate` with limit `configured_max_vcpus`, then CASes
its exact bitmap bit zero-to-one. A set bit means and is owned by exactly one
gate count; failure because the bit was already one releases the count and
returns Busy without touching the arena. The resulting
`ArenaAdmissionToken(slot)` then acquires the ordinary process OpenToken; only
while both are held may it validate the arena-resident vCPU generation and
claim VcpuUseToken. At most one caller per configured vCPU can therefore hold
or attempt a gateway OpenToken, preserving the stated OpenToken bound. The
ArenaAdmissionToken remains held through the entire VcpuUseToken/cold
continuation and is released only after VcpuUseToken and the last per-vCPU read;
release first clears and wakes the exact bitmap word and then release-decrements
and wakes `arena_admission_gate`. SuspendedTransition releases it only after its
slot/root owns the continuation. ArenaAdmissionToken carries no admission epoch
and never authorizes publication. Every cold reentry must acquire a fresh
OpenToken for the then-current epoch as specified below. Normal maintenance
does not close this gate, because a cold continuation may be joining that
maintenance round. `latch_terminal` permanently closes it before publishing
Requested; terminal reclamation waits for the gate count to reach zero and then
requires the complete bitmap be zero. No post-close contender can set a bit.

Each vCPU record also has one `CountedCloseGateV1 use_gate` with limit one.
Inactive, Deactivating and all terminal states require CLOSED; Active requires
open except for the finite deactivation-close interval. Before a
gateway, synchronous LCQ compile/build wait, cold poll, PIC resolver or any
other initial ordinary path first touches NativeFrame/PIC/FaultSlot/per-vCPU
scratch, it must already own the ArenaAdmissionToken and process-only OpenToken
acquired solely through TerminalControl. While those tokens prevent arena
reclamation, it acquire-loads `Active(g)`, acquires the gate's sole count,
acquire-reloads the identical `Active(g)`, acquire-loads FaultSlot Idle, then
revalidates Active(g) and the owned count; any mismatch or occupied FaultSlot
releases the count and rejects use before dropping the OpenToken. The gate CAS
and its close RMW totally order admission with deactivation on the same atomic:
an acquisition before close is counted and drained, and one after close cannot
succeed. A cold continuation retains ArenaAdmissionToken and VcpuUseToken
from its gateway, but the gateway has already released OpenToken; every later
publication or native reentry acquires a fresh OpenToken for the then-current
epoch. The one executor
retains that token across native execution and its complete cold continuation,
including waits, and releases it only after it has stopped reading every per-
vCPU object. A signal/landing is nested inside the already-held token and never
increments it. When a resolver publishes SuspendedTransition it releases the
use token only after the occupied FaultSlot/root is visible; the later owner
continuation is protected specially by that slot until it publishes Idle and
does not require Active(g). Asynchronous HCQ work retains no per-vCPU pointer
after its immutable snapshot is enqueued.

The sole exception for returning to native code is a
`SuspendedResumeUseToken` owned by the exact FaultSlot sequence. It acquires the
same use gate's sole count only while newly acquired ArenaAdmissionToken and
process-only OpenToken protect the arena, requires both surrounding acquire-loads to observe the same
`Active(g)` and the intervening slot load to observe that exact
SuspendedTransition, and later revalidates Active(g) under the JIT-state mutex
before a Resuming CAS. It publishes/revalidates TLS and a nonzero active epoch
before dropping that OpenToken, and retains ArenaAdmissionToken with the
converted ordinary VcpuUseToken. Deactivating blocks this acquisition or makes that
revalidation fail, and deactivation's pre-enqueue drain waits for an already-
acquired resume token. A successful Resuming transition converts it into the
ordinary VcpuUseToken retained across restored native execution; every failed,
CanonicalOnly or terminal path releases it after stopping per-vCPU reads.

The complete vCPU-slot state machine is:

```text
Inactive(g) -> Active(g) -> Deactivating(g) -> Inactive(g + 1)
Inactive(g)|Active(g)|Deactivating(g)
    -> TerminalDeactivating(g, cause, was_counted_active)
    -> TerminalInactive(g, cause)
```

This is the `VcpuStateCell` defined below: `g`, `cause` and
`was_counted_active` are full-width payload fields and are never truncated into
the atomic tag word. Every reference to an identical `Active(g)` or
`Deactivating(g)` is a validated word-plus-payload observation.

All activation/deactivation APIs acquire a NormalResultToken through
TerminalControl before dereferencing the slot, then take the one
slot-registration mutex. While holding it they attempt the nonwaiting atomic
OpenToken acquisition for the current E, revalidate the exact slot generation,
perform the transition/`active_vcpus` update, release OpenToken and then unlock.
Failure to acquire Open releases the mutex and returns/joins terminal without
touching the slot. This mutex is never nested with JIT-state,
queue, code-cache or memory-authority mutexes. Activation prepares all fallible
resources first, requires Inactive(g), `use_gate == CLOSED|0`, zero admission
bit, null TLS and Idle FaultSlot, checked-increments active_vcpus,
release-stores Active(g), then release-stores `use_gate = 0` and wakes it as one
no-fail commit tail; failure before that tail rolls back the count/resources
and leaves Inactive(g) with the gate closed. It releases OpenToken and the slot mutex before its
NormalResultToken. Normal
deactivation uses the same OpenToken/mutex order. Its no-fail commit first
closes `use_gate` with the common AcqRel RMW and then changes
Active-to-Deactivating; it releases the OpenToken/mutex and waits for the gate's
low count to reach zero before the pre-enqueue drain below. It
advances g exactly once only at its
Inactive commit. TerminalDriverGuard owns the other row after normal-result
tokens drain: under the same mutex it changes every slot to
TerminalDeactivating, first closing any still-open use gate and recording
whether Active/Deactivating was counted, and
thereafter no API can reactivate it. After per-slot token/TLS/fault/root drain
it changes the slot to TerminalInactive and decrements active_vcpus exactly for
`was_counted_active = true`. Generation arithmetic is preflighted before any
normal mutation and never wraps; terminal states require no successor
generation.

Lifecycle APIs are the deliberate exception to ArenaAdmissionToken: their
NormalResultToken prevents terminal arena reclamation and their OpenToken
excludes a normal Closing freeze, so the scheduler may atomically change
Active-to-Deactivating and assert control while the executor still owns the
admission bit. The bit serializes executor use, not scheduler control writes;
the deactivator acquires no bit until the executor has polled, canonicalized and
released it.

Before taking the slot-registration mutex, deactivation takes only the
maintenance-queue mutex, requires Open(E), claims its precharged
VcpuDeactivate record, reserves one unique request sequence/current-or-next
RoundTicket and publishes
`Prepared(g_record, VcpuDeactivator(VcpuHandle, expected_vcpu_generation:u64,
VcpuDeactivationToken), sequence, ticket)`. `VcpuDeactivationToken` is a
process-allocated checked nonzero-u64 identity, distinct from BuildToken. It
then unlocks. Closing may be requested, but cannot inspect/cancel Prepared
until OpenTokens drain. The scheduler next takes the slot mutex, acquires an
OpenToken for that exact E, revalidates Prepared plus Active(g), and reserves
`g + 1` and every VcpuStateCell publication sequence required through Inactive.
Failure before mutation releases Open/slot and, under only the queue mutex,
changes its exact VcpuDeactivator Prepared record directly to
CancelledBeforeCommit; it never creates or increments AwaitingBuilder. Active
remains unchanged.

The no-fail commit CASes Active to Deactivating, then uses the second tag-only
exception to CAS that exact Prepared record to Pending before releasing the
OpenToken/slot mutex, release-sets the per-vCPU/shared control words and wakes
record waiters. From Active-to-Deactivating onward there is therefore already a
fully sequenced reachable request; no counter/ticket/record claim remains.
No new VcpuUseToken or vCPU-bound gateway OpenToken can complete. Before
driving or joining the round, the initiating scheduler waits without any subsystem mutex for
the low count of `use_gate` and `active_code_epoch` to become zero. If the initiator itself owns
the token, it first finishes/canonicalizes its current continuation and releases
the token, then joins that wait. Consequently a token holder already waiting on
an unrelated maintenance result can pass that round's handoff, finish its cold
continuation and release the use-gate count; no coordinator ever waits for the use token
of a requester in its own sealed cohort. SuspendedTransition has already
released VcpuUseToken and is governed by its FaultSlot.

Generation/publication/request-sequence/RoundTicket exhaustion is therefore a pre-CAS terminal
request and leaves Active unchanged for the terminal driver to drain. After the
Active-to-Deactivating CAS, no ordinary fallible operation remains before the
already-Pending record is processed. An unexpected invariant/platform failure may no longer
restore Active: the initiator latches terminal, and the terminal registry scan
completes that exact Deactivating slot to TerminalInactive after the same
use/TLS/fault/root drain. There is no rollback edge from Deactivating to Active
and no partially deactivated success result.

Only after that use/epoch drain does the scheduler drive or join the already-
queued VcpuDeactivate record. Its plan asserts the closed use gate has low count zero and the active epoch remains zero;
only in Closed does it acquire-inspect its FaultSlot. If it is
SuspendedTransition, the plan
CASes it to CanonicalOnly, cuts its temporary executable root while retaining
the slot UnitPin, wakes that exact fault sequence and changes its already-
Applying VcpuDeactivate record to
`WaitingFaultIdle(old_sequence, retry_root_cut = true)` without waiting. The vCPU
remains Deactivating, retains its PIC roots and cannot be reactivated. Its owner
canonicalizes, releases the pin and publishes FaultSlot Idle even while the
coordinator is Closed. That wake changes WaitingFaultIdle to Pending with a new
sequence on the next ticket; only
that later plan, after revalidating Idle with no retry root, clears each PIC way
before releasing its composite root/backlinks, checked-increments the vCPU-slot
generation, changes the record to Inactive and decrements `active_vcpus`. If the
first inspection already sees Idle, those final steps occur in the first
cohort. No coordinator waits for a fault owner while holding Closed.
If a higher-precedence MappingChange/FaultTransition already changed the exact
snapshot entry to CanonicalOnly and cut its retry root, VcpuDeactivate verifies
that consumed identity/backlink absence and enters the same WaitingFaultIdle
state with `retry_root_cut = true`; losing that CAS does not permit it to mark
Inactive early. With closed-use-gate count/active epoch zero, any state other than matching
SuspendedTransition, matching CanonicalOnly or Idle is an invariant failure.
Reactivation reuses that same record slot only after those steps and publishes
Active with the new generation only after checked-incrementing `active_vcpus`
under the slot-registration authority and requiring the result is at most
`configured_max_vcpus`. Increment failure leaves the slot Inactive; successful
release-publication of Active is the increment's sole commit point. Initial
registration uses the same checked increment, and deactivation decrements
exactly once when it commits Inactive. Reactivation never clears a non-atomic
PIC while the vCPU could execute it.

The VcpuUseToken owner also owns the `tls_native_executor` descriptor on its
current host thread. While the enclosing process-only OpenToken is still held,
and before any native epoch publication, it checked-increments that vCPU's nonzero
`tls_publication_generation`, obtains the unique non-cloneable
`TlsPublicationToken(P)`, fills its one precharged immutable
`TlsDescriptorV1`, then release-stores that descriptor's address into the host
thread's `tls_native_executor: AtomicPtr<TlsDescriptorV1>` with
that value, the vCPU generation and current host/JIT handler generations.
It acquire-revalidates Active(g), the VcpuUseToken and the still-owned
OpenToken after publication; failure clears and drains the descriptor without
entering native code and releases both tokens in safe order. No other
thread may publish a descriptor for that vCPU. On an ordinary canonical exit it
first reaches epoch zero and FaultSlot Idle, release-clears TLS, then waits in
order for the per-JIT handler count and permanent global handler count to reach
zero with acquire observations. Only after that grace does it publish the exact
`tls_publication_generation` in the vCPU slot's ordinary
`tls_clear_generation`.
Native fault authority is then detached, but it retains VcpuUseToken through the
entire ordinary cold continuation and releases/migrates/deactivates it only
after stopping every remaining per-vCPU read. The global grace covers a handler
which read the old TLS descriptor but had not yet incremented the per-JIT count.
On a SuspendedTransition exit, the same sequence is legal after epoch zero once
the occupied slot owns the complete continuation and the landing has returned;
Idle is then deliberately not required. Resumption
checked-increments and publishes a fresh descriptor before converting
SuspendedResumeUseToken and
restoring native context. Migration is clear-and-record on the old host
thread followed by publication on the new one; Inactive has no TLS descriptor.
Publication-generation exhaustion latches terminal before publication and
never wraps. Shutdown snapshots the exact registered vCPU slot/generation and
latest publication generation. `tls_clear_generation` proves ordinary
detach only; it is distinct from the terminal `tls_detached_generation`
acknowledgement described below. ELF TLS belongs to its host thread and is not
remotely addressable: only that thread may load/assert its
`tls_native_executor` pointer. Terminal code observes only the arena-resident
publication/clear generations after the use/admission drain. Shutdown never
waits for or tries to enumerate an unbounded set of historical OS threads.
Successful `sigaltstack` installation first creates a non-cloneable
`AltStackInstallGuard { tid, saved_disabled_stack, vcpu_generation }`. Until TLS
publication, every error path consumes that guard by restoring the exact prior
stack while both signals are blocked; generation exhaustion is included. The
release-store of the descriptor atomically transfers this obligation into
`TlsPublicationToken(P)`. Thus there is no interval in which an installed Nixe
altstack is owned by neither guard, and omitting TLS detach never omits an
altstack restore.
`TlsPublicationToken(P)` is consumed exactly once by the production
`detach_native_tls(P)` routine. That routine first establishes the exit-specific
FaultSlot/epoch condition (Idle plus epoch zero for an ordinary or rollback
exit, the matching occupied SuspendedTransition or CanonicalOnly slot plus
epoch zero for suspension/resume rollback, or the specified fatal
state), then release-stores null to `tls_native_executor`, waits for the per-JIT count and the
global count to become zero in that order, release-stores exactly P to
`tls_clear_generation`, and restores the saved disabled altstack on the same
tid. Every normal, suspend, failed gateway/reentry publication, Stopping and
fatal exit calls it; no caller open-codes a subset. VcpuUseToken and
ArenaAdmissionToken cannot be released while a TlsPublicationToken is live.
This makes a publication which fails before entering an epoch visible to the
same terminal-generation audit as a normal native execution.

`TlsDescriptorV1` is never a TaggedPayloadCell and is never mutated while its
address can be published. It contains exactly `{TerminalControl pointer,
JitProcessHandle, VcpuHandle, host_install_generation,
jit_handler_generation, tls_publication_generation, chain_stack_mode}`;
`chain_stack_mode` is `InterruptedStack` for the required previously-disabled
executor altstack. The descriptor belongs to the vCPU slot and cannot be
rewritten/reused until null publication and both grace waits have completed.

`JitProcessHandle` everywhere means exactly
`JitProcessHandleV1 = repr(C, align(8)), size 16 {
terminal_control_address:u64, process_generation:u64 }`. The address must be a
canonical, nonzero, 4096-aligned round-trip representation of the one
TerminalControl mapping; generation is a checked nonzero-u64 allocated under
HostSignalControl's install mutex and copied into immutable TerminalControl at
construction. It never wraps or changes. A handler may resolve the address only
after incrementing `global_inflight_handlers`, observing HostSignalControl
accepting/installed generation, and loading a nonnull TLS descriptor; it then
requires descriptor pointer, handle address, TerminalControl magic/process
generation, `signal_registered`, host install generation and JIT handler
generation all agree before incrementing the per-JIT in-flight count. Any
mismatch decrements in reverse order and chains without an arena read. The
process owner keeps TerminalControl mapped until TLS null publication, per-JIT
and global signal grace, unregister, ShutdownRecord Applied and all strong
process-handle references drain; only then can that virtual address be reused.
The Copy JitProcessHandle is identity, not a refcount/pin and has no independent
registry. These rules make address reuse ABA-safe without an unbounded global
process map.
The TLS pointer and a separate `tls_altstack_owner: AtomicU32` (`0=None`,
`1=Installing`, `2=Nixe`) are link-time ELF TLS symbols using the initial-exec
TLS model and audited direct per-ISA TLS relocations; `thread_local!`, lazy TLS,
`pthread_getspecific` and allocation from the handler are forbidden. [Task 2](tasks/02-lifetime-foundation.md)
checks generated assembly offsets and a signal at every publication boundary.

[Task 0](tasks/00-baseline.md) records, and [Task 2](tasks/02-lifetime-foundation.md) mechanically rechecks, the byte size, alignment,
page rounding and total fixed charge of every bounded table at 128 vCPUs and
at the configured process value. The manifest's worst-case value uses four HCQ
workers and the full 256-KiB SignalAltStack for every one of 128 slots. The
shared pure `tiered/layout.rs` calculator emits a `JitLayoutReportV1` containing
`configured_max_vcpus`, `configured_max_address_spaces`, `worker_count`,
`at_minsigstksz`,
`signal_alt_stack_bytes`, each `(layout_row_id, object_name, count, size, alignment,
type_layout_sha256, rounded_committed_bytes, reserved_bytes,
worst_case_count, worst_case_rounded_committed_bytes,
worst_case_reserved_bytes, charge_class, backing_row)` row and checked totals.
Layout-row IDs and names are each unique and nonzero; serialization is by
increasing ID and source-file/TOML order has no meaning. Unknown, duplicate or
noncanonical-order rows fail validation. [Task 0](tasks/00-baseline.md) golden-tests
the worst-case report; [Task 2](tasks/02-lifetime-foundation.md) invokes the same calculator with runtime values
and rejects a report whose constants/layout hashes differ from the manifest.
The report is an internal construction proof, not production telemetry.
For both the manifest proof and a runtime report, each row has exactly one
charge interpretation. `fixed_committed` requires
`reserved_bytes == 0` and contributes `rounded_committed_bytes` to fixed charge;
`lazy_committed` requires `reserved_bytes == 0` and contributes only its pages
currently committed; `reserved_credit` requires
`rounded_committed_bytes == 0` and contributes `reserved_bytes` only while its
owning reservation ticket is live; `control_committed_excluded` requires
`reserved_bytes == 0`, contributes its page-rounded bytes only to the report's
separate `excluded_control_committed_bytes` total, and never enters a JIT ledger
subcounter; `virtual_only` requires both values zero and no physical backing;
and `logical_capacity_only` requires current/worst committed/reserved bytes all
zero, `backing_row = "metadata_arena_pool_pages"`, and records only logical
count/type capacity whose actual pages are charged by that one backing row.
Every other class requires an empty backing_row. Checked totals are exactly the sums for
`committed_executable_bytes`, `committed_metadata_bytes` and
`reserved_credit_bytes` defined below, with `jit_charged_bytes` their checked
sum. A row which cannot be assigned bijectively to one subcounter, or a byte
charged by two rows, rejects construction. The
`fixed_charge_at_128_vcpus_worst_case` field is the sum of every
`fixed_committed` row at the manifest's stated maximum stack sizes plus the
construction-time committed portion of each `lazy_committed` row; it contains
no virtual address reserve or transient reservation credit.
Exactly `terminal_control_page` has `control_committed_excluded`; it is one
4096-byte page and includes the four persistent WorkerControlCells described
below. The OS-process-global 4096-byte HostSignalControl page is not a per-JIT
layout row or ledger byte; its separately asserted layout is
`HostSignalControlLayoutV1 { committed_excluded_bytes: 4096 }` and it remains in
process PSS until OS exit.
The `layout_row` names are exactly:
`terminal_control_page`; `address_space_registry_directory`,
`address_space_registry_bitmap`, `address_space_registry_slots`;
`unit_registry_directory`, `unit_registry_bitmap`, `unit_registry_slots`;
`family_registry_directory`, `family_registry_bitmap`,
`family_registry_slots`; `patch_registry_directory`, `patch_registry_bitmap`,
`patch_registry_slots`; `build_fingerprint_registry_directory`,
`build_fingerprint_registry_bitmap`, `build_fingerprint_registry_slots`,
`build_fingerprint_payload_pages`; `dispatch_buckets`,
`dispatch_slot_directory`, `owner_buckets`, `owner_cell_directory`,
`dynamic_bridge_weak_buckets`, `dynamic_bridge_weak_cursors`,
`bridge_quota_bitmap`;
`active_build_cells`, `maintenance_queue_index`, `pressure_request_bitmaps`,
`reclaim_control_cell`;
`cohort_touched_dispatch`, `cohort_touched_unit`, `cohort_touched_family`,
`cohort_touched_patch`, `cohort_touched_native_page`, `cohort_touched_pic`,
`cohort_touched_vcpu`, `cohort_touched_address_space`;
`cohort_radix_tmp`, `cohort_radix_histogram`, `cohort_cleanup_ordinals`,
`cohort_metadata_projection`, `cohort_segment_projection`;
`cohort_fault_snapshot`;
`native_pc_top_directory`, `native_pc_page_slots`,
`page_fault_table_headers`, `page_fault_table_record_pages`,
`page_fault_table_unit_pin_pages`, `page_fault_table_safety_shadows`,
`page_fault_table_shadow_record_pages`; `vcpu_runtime_slots`,
`native_frames`, `fault_slots`, `tls_descriptors`, `epoch_slots`, `pic_ways`,
`hot_seed_tables`, `boundary_tables`, `signal_stack_bundles`,
`signal_alt_stacks`, `landing_stacks`, `signal_guard_pages`;
`pending_hcq_queue`; `code_segment_headers`,
`island_bitmaps`, `island_slot_records`,
`code_body_allocation_bitmap_l0`, `code_body_allocation_bitmap_l1`,
`rx_envelope`, `metadata_arena_virtual_reserve`,
`metadata_arena_header`,
`metadata_arena_page_directory`, `metadata_arena_allocation_bitmaps`,
`metadata_root_directory`, `metadata_arena_pool_pages`;
`mapping_generation_vector_pages`, `mapping_write_buffer_pages`,
`root_mutation_plan_claim_pages`; and
`memory_transaction_guard`.
Exactly `metadata_arena_header`, `metadata_arena_page_directory` and
`metadata_arena_allocation_bitmaps` are fixed-committed physical owners for
pages `[0,2058)`, and exactly `metadata_arena_pool_pages` is the lazy-committed
physical owner for pages `[2058,262144)`. Every logical row whose object resides
in that pool has charge_class logical_capacity_only and backing_row
`metadata_arena_pool_pages`; this includes every PermanentExtent root and every
FixedSlot/PagedPayload object. Its count/type bounds remain auditable but it
contributes no second byte. No pool page may be attributed to a logical row or
metadata_slab in a ledger sum. All other rows have empty backing_row. The
validator rejects any different classification or any committed pool page not
counted once by the pool row.
Nested `SafetyPayloadCell`, `CohortObjectPlanCell`, maintenance records and permits
are fields of the corresponding slot/record row and may not be charged again.
The only identifiers allowed in `count_formula` are decimal constants,
`configured_max_vcpus`, `configured_max_address_spaces`, `worker_count`, and
the fixed registry/segment capacities in this document, combined with checked
`+`, `*` and parenthesized `ceil_div(x,4096)`. [Task 0](tasks/00-baseline.md) emits all and only these
rows; a persistent allocation without exactly one owning row prevents Ready.
All listed `repr(C)` types, including their nested fields, offsets, padding,
size and alignment, are layout-only production schemas created in [Task 0](tasks/00-baseline.md).
Later tasks add methods/ownership behavior but may not change their layout
without reopening [Task 0](tasks/00-baseline.md) and the baseline manifest.
The checked-in [Task 0](tasks/00-baseline.md) baseline for 128 configured vCPUs must have
`fixed_charge_at_128_vcpus_worst_case <= 384 MiB`. Baseline generation fails,
and the specification must be amended rather than shipped, if that inequality
does not hold. At construction the generated report for the requested
configuration must additionally prove that fixed charge plus the 32 MiB LCQ
reserve is at most the 512 MiB watermark and fixed charge plus
`largest_hcq_reservation` is at most 608 MiB. These are three independent
checks: the 384-MiB bound preserves deterministic headroom at the supported
128-vCPU maximum, while the latter two validate the actual host/configuration.
Construction never starts a configuration in which HCQ is structurally unable
to make progress.

Registry policy is fixed: the address-space registry has 64 slots, the unit
registry 1,048,576 slots, the family registry 262,144 slots, the patch
registry 2,097,152 slots and the build-fingerprint registry 266,404 slots
(262,144 family owners, 4,096 NegativeBuildCache ways and the worst-case
128 + 32 + 4 active-build cells). Those summands are disjoint simultaneous
owners; sharing a fingerprint slot between them is forbidden.
Their generated payload types are exactly
`AddressSpaceRegistryPayloadV1`, `UnitRegistryPayloadV1`,
`FamilyRegistryPayloadV1`, `PatchRegistryPayloadV1` and
`BuildFingerprintRegistryPayloadV1`, each `repr(C, align(64))`; there is no
polymorphic `GeneratedReprCSlot`. Storage always uses
`MetadataSlotWrapper<T> = MetadataAllocationHeaderV1 + alignment padding + T`.
For each manifest row, with `payload_alignment=64`, `slot_bytes` equals only
`align_up(align_up(16, payload_alignment) + size_of(T),
max(8,payload_alignment))`; its `metadata_slab` row has that identical
`slot_size` and alignment, and a slab page contains exactly
`floor((4096 - 128) / slot_bytes)` whole slots after its mandatory
`MetadataSlabPageHeaderV1`. It rejects a zero result, records both the header
and unused tail as charged page overhead and never places a slot across pages.
Pages are committed lazily, but every
directory/bitmap is constructed and charged up front. The address-space, unit,
family, patch and fingerprint bitmaps are exactly 8, 131,072, 32,768,
262,144 and 33,301 bytes. Bits beyond `slot_count` in the final byte are zero,
never allocatable and checked by [Task 0](tasks/00-baseline.md)/runtime validation. The manifest and JitLayoutReport contain all five
exact rows; the report also
contains `configured_max_address_spaces` and charges every configured slot's
permit/record storage.

Each row above is exactly `[u8; ceil_div(slot_count,8)]`. Slot i is the
least-significant-bit-first bit `(byte=i/8, bit=i%8)`; one means Free and zero
means reserved, Live, Retiring or invalid tail. Construction sets every valid
slot bit to one and every tail bit to zero. Under that registry's sole writer
mutex, allocation selects the numerically lowest one bit, validates the slot
empty and its checked successor generation available, and clears the bit as the
reservation linearization point. Successful owner publication leaves it zero;
an abort zeroes the complete slot and restores it to one. Retirement first
release-clears the owner, drains its documented pins/epochs, advances the slot
generation and zeroes the payload; only then, under the same mutex, may it set
the bit to one. At every mutex unlock, checked `popcount(bitmap)` equals
`slot_count - reserved_count - live_count - retiring_count`, every one bit maps
to an empty directory slot and every nonempty slot maps to zero. The generated
validator checks those equalities and the zero tail; host byte order never
changes the bit-number formula.

Unit exhaustion
first requests PressureEvict; if no reclaimable unit remains, a required LCQ
fails with `UnitRegistryCapacityExceeded` while HCQ/bridge publication is
transient. Family exhaustion disables only new HCQ admission. Patch exhaustion
emits a permanent canonical fallback for a not-yet-published source and leaves
an existing patch unchanged. Shutdown scans committed registry pages and slots
in ascending numeric order; no hash-table iteration defines that order.

The bounded-index rows are also exact: dispatch is robin_hood with 2,097,152
buckets, a 1,048,576 live limit and set_count/ways zero; owner is robin_hood
with 2,097,152 buckets, a 1,048,576 live limit and set_count/ways zero;
dynamic_bridge_weak is set_associative with
`4096 * configured_max_vcpus` buckets/live entries,
`1024 * configured_max_vcpus` sets and four ways. The separate bridge quota
remains `8192 * configured_max_vcpus` because retired bridges may no longer be
weak-indexed. The exact schemas are:

```text
IndexSlotRefV1 = repr(C, align(8)), size 16 {
    slot_index_plus_one:u32, reserved_zero:u32, slot_generation:u64
}
RobinHoodBucketV1 = repr(C, align(8)), size 24 {
    stable_hash:u64, slot:IndexSlotRefV1
}
DynamicBridgeWeakBucketV1 = repr(C, align(8)), size 24 {
    stable_hash:u64, bridge:IndexSlotRefV1
}
```

An all-zero slot/bridge reference is Empty and requires hash zero; Occupied has
nonzero index-plus-one/generation and the complete FNV hash. Because all three
indexes are accessed only under the JIT-state mutex, fields are ordinary and
no torn lock-free read is permitted. Robin-Hood probe distance is recomputed as
`(current_bucket - (stable_hash & 0x1f_ffff)) & 0x1f_ffff`; it is not a stored
field. Each dynamic-weak set has one separate u8 cursor, initially zero and
always 0..3: insert chooses the lowest empty way, otherwise replaces cursor and
sets cursor `(cursor+1)&3`. Its logical-capacity-only cursor row has
`1024*configured_max_vcpus` bytes rounded to pages. Thus `bucket_bytes` is
exactly 24 for all three rows, and at 128 vCPUs their combined bucket bytes are
`2*2,097,152*24 + 524,288*24 = 113,246,208`; cursors add 131,072 bytes. All bucket
arrays and allocation bitmaps are precharged; lazily committed value slabs are
charged when committed.

`logical_cpus` is sampled once as the number of CPUs in the process's Linux
`sched_getaffinity` mask; an empty mask or syscall failure is a process-JIT
construction error, not a hardware-concurrency guess. Integer division in the
worker formula truncates toward zero. If the formula is nonzero, all workers
and their fixed queue/state must start before guest execution; any partial
thread-start failure stops/joins those already created and fails construction.
Only a formula result of zero is the normal no-HCQ configuration.

The 512 LCQ ceiling is an emergency bound, not a request to build a 512
instruction CFG. A normal LCQ block stops at its first terminator and is
usually much smaller. The 2048 HCQ limit is a total across the complete
candidate; it is never reset per path, entry or continuation. Code-page and
mapping dependencies are collected completely and do not impose independent
discovery limits.
