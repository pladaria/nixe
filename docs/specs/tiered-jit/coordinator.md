# Coordinator phases, terminal control and signal installation

[Specification index](README.md) · [Implementation tasks](tasks/README.md) · [Open review items](review-status.md)

Related contracts: [Tagged state cells, keys and dispatch](runtime-state.md); [Maintenance records, cleanup and mapping requests](maintenance-records.md); [Request execution, terminal transfer and cohort handoff](coordinator-execution.md); [Native fault transport and completion](faults.md).

## Maintenance coordinator

All callable-code, executable-root and guest mapping mutation uses one process
coordinator. It owns:

- one ABA-safe tagged phase
  `Open(admission_epoch)`,
  `Closing(round_id, admission_epoch, RoundTicket, DriverToken)`,
  `Closed(round_id, admission_epoch, RoundTicket, DriverToken)`,
  `Stopping(cause, shutdown_DriverToken)` or terminal `Stopped(cause)`;
- checked-u64 `request_sequence` and `acknowledged_sequence` counters;
- the OpenToken count;
- an intrusive queue of typed `MaintenanceRecord`s; and
- a sticky shutdown/fatal flag and reason bits used only as wakeup hints on the
  existing control/poll word.

The conceptual tagged phase has this mandatory portable representation.
`phase_word: AtomicU64` stores its three-bit tag in bits 0..2 and a checked,
strictly increasing 61-bit `phase_sequence` in bits 3..63; tags 0..4 are Open,
Closing, Closed, Stopping and Stopped and tags 5..7 are invalid. The epoch,
round, RoundTicket handle, checked nonzero `DriverToken` and terminal cause are not packed into that word.
They reside in two fixed `PhasePayloadSlot`s selected by
`phase_sequence & 1`. Each slot contains fixed atomic payload limbs and an
AtomicU64 stamp encoded exactly like TaggedPayloadCell. A writer, while holding
the maintenance-queue mutex, always chooses `next = current + 1`, AcqRel-marks
the opposite-parity slot busy, Release-stores every atomic payload limb,
release-marks it
ready and only then CASes the complete old `phase_word` to the new tag/sequence
with AcqRel success and Acquire failure ordering.
A reader acquire-loads phase_word, Acquire-loads every selected payload limb
only between two equal ready stamps for that phase sequence, uses the same
acquire-fence pattern, and finally
acquire-loads the identical phase_word again; otherwise it retries or fails
admission. Thus every reference to `Open(E)` or an exact Closing/Closed tag
means this validated word-plus-payload observation, and no mixed atomic tuple
or 128-bit CAS is assumed. All normal writers and the transition into Stopping
hold the queue mutex for payload preparation and CAS. The sole exception is
Stopping-to-Stopped: after producer/result/pin drains and metadata-arena unmap,
the unique TerminalDriverGuard prepares the opposite PhasePayloadSlot in
TerminalControl and CASes the exact Stopping word without the now-unmapped
queue mutex. No other writer is then legal, so a lost CAS is raw-fatal
corruption rather than a retry.
Closing-to-Closed, Closed-to-Open and Closed-to-Stopping compare both the
published phase sequence and the identical nonzero DriverToken. `Stopped` has
no live driver and its payload carries a zero DriverToken which is never read
as authority.

Construction initializes unreachable slot 1 with `Open(1)`, ready stamp
`1 << 1`, slot 0 stamp zero and phase_word `(1 << 3) | Open`; process Ready is
the release-publication. `phase_sequence` always increments by exactly one.
Let `M = 2^61 - 1`. A normal round may leave Open(S) only if `S <= M - 5`;
while accepting its first request, the queue authority pre-reserves S+1 for
Closing, S+2 for Closed and S+3 for reopened Open, while S+4/S+5 remain the
terminal reserve. If terminal intent wins from an intermediate Closing/Closed,
it consumes the next two reserved values for Stopping/Stopped and abandons any
unused normal value. If Open has `S > M - 5`, no normal Closing is published:
the requester latches PhaseSequenceExhausted and S+1/S+2 publish
Stopping/Stopped. Terminal entry always publishes Stopping at
`current + 1` and Stopped at `current + 2`, whatever the current parity; it
does not jump to fixed end values or overwrite the active slot. If a normal
request would consume the two-value reserve, it is rejected before mutation
and the coordinator latches terminal intent.
Admission/round/epoch counters are separately checked before their value is
placed in a payload and never wrap. An unsuccessful CAS leaves its prepared
inactive payload unreachable and the queue-serialized writer recomputes from
the newly loaded word; a payload slot is never treated as authority by itself.

The frozen layouts are:

```text
RoundTicketHandleV1 {
  slot:u8, reserved_zero:[u8;7], ticket_generation:u64
}
RoundTicketV1 {
  ticket_generation:u64, round_id:u64,
  closing_admission_epoch:u64, reopen_admission_epoch:u64,
  driver_token:DriverToken,
  first_request_sequence:u64, last_assigned_sequence:u64,
  sealed_cutoff:u64, state:Free=0|Open=1|Sealed=2,
  reserved_zero:[u8;7]
}
PhasePayloadSlotV1 {
  stamp:AtomicU64, admission_epoch:AtomicU64,
  round_id:AtomicU64, ticket_slot_and_zero:AtomicU64,
  ticket_generation:AtomicU64, driver_token:AtomicU64,
  terminal_cause_and_detail:AtomicU64, reserved_zero:[AtomicU64;2]
}
```

There are exactly two precharged RoundTicketV1 slots under the queue mutex.
`ticket_slot_and_zero` stores slot in bits 0..7 and zero above it. Free has all
semantic fields zero except nonzero generation; Open has nonzero round/epochs/
driver/first sequence, last_assigned at least first-1 and sealed_cutoff zero;
Sealed fixes cutoff=last_assigned. The current and precreated next tickets use
different slots. Reopen consumes the sealed ticket, checked-advances its slot
generation only after no record references it, and makes the other ticket
current; no heap ticket/list exists.

Phase payload validity is exact: Open has only admission_epoch nonzero;
Closing/Closed have nonzero admission_epoch/round/ticket/driver and zero
terminal bits; Stopping has only the shutdown DriverToken and packed sticky
cause/detail; Stopped preserves cause/detail and has driver zero. Reserved
limbs are zero. Both PhasePayloadSlotV1 values, phase_word and both RoundTicket
slots are fields of TerminalControl, so the final phase publication does not
dereference the metadata arena.

TerminalControl also reserves this final-tail object:

```text
TerminalUnmapPlanV1 = repr(C, align(8)), size 64 {
  state:AtomicU32, origin:u8, worker_index_plus_one:u8,
  reserved_zero0:u16,
  rx_base:u64, rx_len:u64, metadata_base:u64, metadata_len:u64,
  reserved_zero1:[u64;3]
}
```

State is Empty=0, Ready=1 or Consumed=2. Origin is External=1 or Worker=2;
worker_index_plus_one is zero exactly for External and otherwise 1..=4. The
lengths are the exact nonzero construction reservations (2047 MiB RX and 1024
MiB metadata), bases are page-aligned, checked base+length does not overflow,
and both ranges are disjoint from TerminalControl. Only TerminalDriverGuard
writes Empty-to-Ready in step 8a; only TerminalUnmapGuard changes Ready-to-
Consumed after both exact munmaps. `process_owner_inert: AtomicU32` (0 before
8a, 1 at the barrier) is adjacent TerminalControl state and forbids every
ProcessOwner Drop path from inspecting arena-owned data thereafter.

The phase word/slots, global code epoch, `open_token_gate`, packed shutdown word/
parking sequence, `normal_result_gate`, signal-handler enabled/
generation, `signal_registered`, the noncloneable HostSignal InstallLease
state, `arena_admission_gate`, the four-word arena-admission bitmap,
`awaiting_builder_count`, `awaiting_link_cleanup_count` and the
`awaiting_registration_closed` flag, `prequeue_producer_gate`, and the
per-JIT-process in-flight counter, `worker_live_count`, the four persistent
WorkerControlCells and `worker_start_gate`,
the TerminalUnmapPlan/process-owner-inert fields,
and the five atomic ledger fields (`ledger_sequence`, authoritative
`jit_charged_bytes` and three subcounters) reside in one `repr(C)`
`TerminalControl` whose compile-time size and alignment are asserted to fit one
4096-byte page. Process-JIT construction allocates exactly one anonymous
read/write page for it outside both reclaimable arenas. This page is fixed
process-handle control, not code/version metadata, so it is excluded from
`jit_charged_bytes` and remains visible in total-process PSS. It outlives the
executable and metadata arenas and is unmapped only by the final process-JIT
handle destructor after ShutdownRecord is Applied and all shutdown/result
callers have released their process-handle references, every enabled
WorkerControlCell is ExitedNoArena, and WorkerJoinSet has no Present/Joining
entry and every enabled index is Joined. The external process owner obtains the
AppliedShutdownJoinGuard and completes the omitted worker-driver join before it
may release the last owner reference. That owner/destructor never runs on an
HCQ worker and never attempts a self-join. Thus Stopped, the
terminal zero ledger, signal chaining and waiter wakes never dereference an
arena which shutdown already unmapped. All other persistent JIT objects remain
subject to the charged-arena rule.

Its first 32 bytes are immutable
`TerminalControlPrefixV1 = repr(C, align(8)), size 32 {
magic:[u8;8]=*b"NIXETC01", format_version:u32=1,
terminal_control_size:u32=4096, jit_process_generation:u64,
reserved_zero:u64 }`; all fields listed above follow at Task-0-frozen offsets.
The complete TerminalControl type is `repr(C, align(4096)), size 4096` with zero
tail padding, not merely a type whose size happens to be at most one page.

Per-JIT `handler_generation` has only three values: zero while construction is
disabled, one when the committed HostSignal lease first enables native fault
handling, and the pre-reserved terminal value two. It never increments for an
ordinary TLS attach; those use the separate per-vCPU publication generation.
Shutdown release-publishes two without fallible arithmetic. Values 3..u64::MAX
are invalid in ABI version 1. `signal_registered`, the InstallLease consumed
bit and every field read after arena teardown reside in TerminalControl, never
in reclaimable metadata; its compile-time one-page size assertion includes all
of them.

Construction initializes generation zero and handler_enabled false. Only after
the metadata arena, all vCPU/FaultSlot records, both guarded fault stacks and
TerminalControl are fully initialized and the HostSignal InstallLease is
committed does it release-store generation one and then release-store
handler_enabled true. Publishing process-JIT Ready occurs after both stores.
Failure before Ready leaves false/zero and consumes the lease during rollback;
after Ready only TerminalDriverGuard may store false followed by terminal
generation two.

SIGSEGV/SIGBUS installation is coordinated across multiple simultaneously live
JIT processes by one OS-process-global `HostSignalControl`, not by stacked
per-JIT dispositions. A link-time zero-initialized
`AtomicPtr<HostSignalControl>` and one statically initialized pthread mutex exist
independently of that page. A constructor takes the static mutex and acquire-
loads the pointer. If null, it allocates one page, initializes every atomic and
the page's install mutex, reads/normalizes both prior dispositions, and rejects
SA_RESETHAND before modifying a disposition. On rejection it destroys/unmaps
that still-private page and leaves the pointer null. Otherwise it stores the
frozen dispositions in the page and release-stores the pointer exactly once;
losing constructors use that same pointer, which is never cleared. The
trampolines acquire-load this pointer and cannot run before at least one of them
has been installed. Only after pointer publication and release of the static
mutex may any constructor use the page's own install mutex. That mutex protects
`Uninitialized -> InstallingSegv -> SegvInstalled -> Installed` plus an
`InstallLease`. The
first-ever construction, with `accepting = false`,
changes to InstallingSegv and installs SIGSEGV. Failure returns to
Uninitialized. Success changes to SegvInstalled and is never rolled back or
uninstalled. It then installs SIGBUS; failure leaves SegvInstalled,
`registration_refcount` zero
and accepting false, and a later construction retries only SIGBUS using
the same frozen saved disposition. A handler arriving through the installed
SIGSEGV trampoline while not accepting chains the frozen prior SIGSEGV row.
After SIGBUS succeeds it changes to Installed, sets
`registration_refcount` one, publishes
global install generation one and release-enables accepting. Later zero-to-one
registrations checked-increment that generation. This one-way partial install
eliminates rollback races with signal delivery. The normalized dispositions
are frozen for the rest of the OS process. The Nixe
trampoline and this page are deliberately never uninstalled or unmapped before
OS process exit. A `registration_refcount` transition
0 -> 1 after all JITs were previously closed only checked-advances the global
generation, validates with the libc `sigaction` wrapper that both Nixe dispositions are still
installed and release-enables accepting; it never rereads or rewrites the saved
dispositions. N -> N+1 only checked-increments that checked-u64 field. A
successful noncloneable
`InstallLease { host_install_generation, owns_registration: true }` commits by setting that JIT process's
atomic `signal_registered` false-to-true exactly once. Any later construction
failure releases its committed or uncommitted lease exactly once before
returning. Refcount/generation exhaustion or a replaced disposition rejects JIT
construction before enabling native execution. The embedding contract therefore
grants Nixe exclusive SIGSEGV/SIGBUS disposition ownership after first use;
other components must chain through the frozen prior handlers rather than call
`sigaction` themselves. The page contains the immutable normalized
dispositions, install state, checked-u64 `registration_refcount`, install
generation, global handler-entry count and accepting flag, with all
handler-mutated fields lock-free. Lease consumption changes
`owns_registration` true-to-false exactly once under the install mutex before
the matching decrement; rollback, shutdown and Drop cannot consume it twice. It is not charged to
any JIT cache.

Capture, installation and validation use the pinned `libc::sigaction` wrapper,
not a hand-laid-out `rt_sigaction` argument. The installed mask contains both
SIGSEGV and SIGBUS; flags are exactly `SA_SIGINFO | SA_ONSTACK` plus saved
SA_RESTART and, on AArch64, the kernel-normalized saved `SA_EXPOSE_TAGBITS`.
[Task 0](tasks/00-baseline.md) supplies the pinned Linux constant even if the libc crate does not name
it. Capture reads back the saved disposition through libc, so a retained
SA_EXPOSE_TAGBITS bit is already the kernel's supported value; installation
reads back and requires that bit to remain set before accepting either signal.
Otherwise construction fails and no native execution is enabled, because Nixe
cannot reconstruct address tags which the kernel stripped before handler entry.
On x86-64 that bit must be clear. Neither saved SA_RESTORER nor an arbitrary restorer pointer is
copied. On GNU/Linux the wrapper supplies its ABI-correct restorer, so each
global_asm entry returns with the target C ABI directly to that restorer. The
checked disassembly fixture raises each signal and proves return through
`rt_sigreturn`. Raw handler-side `rt_sigprocmask` uses the Linux kernel sigset
size eight on both supported 64-bit ISAs. The SIG_DFL redelivery shim alone uses
a generated per-ISA kernel-sigaction layout whose size/offset constants are
const-checked against the pinned Linux UAPI fixture; it installs SIG_DFL,
unblocks through that eight-byte mask and calls `tgkill`.

The committed HostSignal InstallLease is stored once in the JIT-process owner;
it is neither Arc-cloned nor accessible to shutdown joiners. Only the
TerminalDriverGuard consumes it after the per-JIT and global signal grace;
construction rollback consumes the same lease, and Drop/joiners never adjust
the refcount independently.

The published `TlsDescriptorV1` contains the generational `{TerminalControl
pointer, JitProcessHandle, VcpuHandle, host_install_generation,
jit_handler_generation, tls_publication_generation, chain_stack_mode}` fields
defined above. Per-JIT shutdown disables and drains
only that TerminalControl and clears all of its TLS descriptors. Explicit JIT
shutdown's sole TerminalDriverGuard, not Rust Drop or a joining caller,
checked-decrements the global refcount under the install mutex after those
acknowledgements; 1 -> 0 release-clears accepting but leaves
the trampoline and frozen dispositions installed. Drop asserts successful
shutdown or performs only the same nonunwinding disable/leak-safe fallback; it
does not promise a fallible return. Any late Nixe handler sees permanent
HostSignalControl and chains. This protocol
permits arbitrary construction/destruction order and forbids treating a
per-JIT TerminalControl as knowable before TLS is read.
