# Native fault transport and completion

[Specification index](README.md) · [Implementation tasks](tasks/README.md) · [Open review items](review-status.md)

Related contracts: [Gateway, state transfer and helper ABI](native-abi.md); [Direct memory and fault authority](memory-authority.md); [Coordinator phases, terminal control and signal installation](coordinator.md); [Epoch reclamation and terminal teardown](epochs-and-shutdown.md).

## Native fault transport and completion

Each recoverable guest-memory host instruction has one immutable
`FaultSiteRecord` containing:

- exact `[native_start, native_end)`, segment generation, UnitHandle,
  CodeUnitId and CodeVersion;
- InstructionKey, exact four guest bytes and MemoryEffectPlanId;
- subaccess and architectural commit-stage ordinals;
- `budget_debit_before` and `budget_charge_current` for the current checkpoint
  interval, with the meanings defined by control-budget accounting;
- access kind, size, ordering and side-effect-free effective-address/value
  recipes;
- the complete PreFaultStateMap and required lazy NZCV/FPSR representation; and
- the successful canonical continuation BlockKey.

The PreFaultStateMap is the complete PhysicalStateMap at the instant immediately
before that host instruction. It has a final location and dirty bit for every
architectural component and no unresolved SSA value. Its recipes use only
captured registers, fixed NativeFrame locations and constants. The backend
statepoint is a scheduling/register-allocation barrier for every memory or
architectural effect whose movement would change the recorded stage.
Infrastructure loads/stores in gateways, NativeFrame spills, bridges, islands,
PIC probes and helpers have no FaultSiteRecord; any host fault at such an
instruction, or at any generated PC absent from the recoverable catalog, is
`InternalFault` rather than guest-memory emulation.

Each configured native executor has one atomic preallocated fault slot and two
distinct charged stacks in one reservation laid out as
`guard | SignalAltStack | guard | 64-KiB landing stack | guard`. Every guard is
one 4096-byte PROT_NONE page. At construction Linux `getauxval(AT_MINSIGSTKSZ)`
must return nonzero; checked arithmetic defines SignalAltStack usable bytes as
`align_up(max(64 KiB, AT_MINSIGSTKSZ + 32 KiB), 4096)` and rejects the host if
the result exceeds 256 KiB. Only the two usable ranges are read/write. This
bound, actual rounded charge and AT_MINSIGSTKSZ value are recorded in the host
layout proof; the kernel signal frame is never placed on NativeFrame's 4096-byte
helper stack or on the landing stack.

On each initial gateway/cold reentry/resume, after the permanent admission
tokens protect the vCPU allocation but before TLS publication, the owner calls
`sigaltstack` on that host thread and requires the prior state have SS_DISABLE
(an embedding-provided enabled altstack makes that thread ineligible for native
JIT execution). It saves that exact disabled `stack_t` plus host tid in the
vCPU slot. The Nixe
SignalAltStack cannot be shared concurrently or migrated. After release-clearing
TLS and completing per-JIT/global handler grace, the same tid restores the saved
`stack_t` before cold work or token release. Installation failure before TLS
returns a typed native-entry error. A restore failure or tid mismatch after
installation takes the raw process-fatal escape: shutdown may never unmap a
range still registered as that thread's alternate stack. The permanent Nixe
dispositions always include SA_SIGINFO and SA_ONSTACK; they do not copy saved
SA_ONSTACK. The chaining shim chooses its call stack by this exhaustive matrix
after reading only the permanent saved disposition and inline TLS owner word:

```text
tls_altstack_owner == Nixe                    -> interrupted ucontext SP
tls_altstack_owner == None, saved SA_ONSTACK -> current handler SP
tls_altstack_owner == None, no SA_ONSTACK    -> interrupted ucontext SP
tls_altstack_owner == Installing             -> raw process-fatal
```

The first row recreates the executor's prior state, whose altstack was disabled
before Nixe installed its own. In the second row Nixe has not changed that
thread's altstack, so its SA_ONSTACK entry already placed the trampoline on the
embedding altstack the saved handler would have used (or left it on the
interrupted stack if disabled). The third row recreates a disposition without
SA_ONSTACK. A selected switch uses target ABI alignment/red-zone rules. Both
signals remain blocked while selecting and changing stacks. Immediately before
calling the saved user handler, the shim uses raw `rt_sigprocmask` to install
exactly `interrupted_ucontext_mask union saved.sa_mask union {signum unless
SA_NODEFER}`; this can deliberately unblock the other fault signal when the
saved disposition allowed it. After a normal return it blocks both signals
before restoring the handler stack, then finally restores the exact interrupted
mask. Every syscall failure takes the balanced raw-fatal path; the JIT helper
stack is never used.

The embedding contract requires every chained saved handler to return normally
to this shim. `siglongjmp`, language exceptions, unwinding or another nonlocal
exit while a Nixe TLS/use/admission guard exists is unsupported embedding
behavior because no cleanup code can regain control. Tests install returning
SA_SIGINFO and non-SA_SIGINFO handlers for every matrix row; Nixe does not
promise cleanup after a nonlocal escape.

The install/restore routine raw-blocks SIGSEGV and SIGBUS on the current thread
around each `sigaltstack` change. Install requires `tls_altstack_owner=None`,
release-stores Installing, performs and records the successful syscall, then
release-stores Nixe before unblocking; only afterward may it publish the TLS
descriptor. Restore occurs after TLS null and both handler graces, changes the
kernel stack first, release-stores None and then restores the old signal mask.
Installing is therefore unobservable to either signal handler; observing it is
raw-fatal corruption. All error paths restore the prior mask or take the raw
escape.

The fault-slot state machine is:

```text
Idle -> Capturing -> SignalCaptured -> Resolving
Resolving -> NativeRetry | SuspendedTransition | CanonicalExit | Fatal
SuspendedTransition -> Resuming | CanonicalOnly
NativeRetry|Resuming|CanonicalOnly|CanonicalExit|Fatal -> Idle
```

This state is one portable `AtomicU64`: bits 0..3 are tags `Idle = 0`,
`Capturing = 1`, `SignalCaptured = 2`, `Resolving = 3`, `NativeRetry = 4`,
`SuspendedTransition = 5`, `Resuming = 6`, `CanonicalOnly = 7`,
`CanonicalExit = 8`, `Fatal = 9` and 10..15 are invalid; bits 4..63 are a
nonzero 60-bit fault sequence. Construction starts at `Idle(0)`. Only the
signal handler's `Idle(s) -> Capturing(s + 1)` CAS allocates a sequence; every
later transition, wake and payload validation compares and preserves that
exact value through the return to `Idle(s + 1)`. If `s + 1` cannot fit, the
handler takes the raw async-safe fatal escape before touching payload storage;
the sequence never wraps. The payload is written before the corresponding
release transition and read only after an acquire observation of the matching
word. NativeFrame's checked fault sequence is a copy for assembly validation,
not a second authority.

The process-global handler covers SIGSEGV and SIGBUS. Its first instruction
sequence, before reading TLS or any reclaimable JIT storage, checked-increments
permanent HostSignalControl's lock-free `global_inflight_handlers`, then
acquire-loads its accepting flag and installed generation. If not accepting it
decrements and chains. A null `siginfo_t*` or `ucontext_t*` decrements and
chains before any dereference. It then admits native classification only for
`SIGSEGV/SEGV_MAPERR(1)`, `SIGSEGV/SEGV_ACCERR(2)`,
`SIGBUS/BUS_ADRALN(1)`, `SIGBUS/BUS_ADRERR(2)` or
`SIGBUS/BUS_OBJERR(3)`. Any `si_code <= 0` (including SI_USER, SI_QUEUE and
SI_TKILL), SI_KERNEL, or another positive code decrements and chains before
reading TLS, even when the interrupted PC happens to lie in Nixe RX. This
allowlist is generated from the pinned Linux ABI constants and [Task 2](tasks/02-lifetime-foundation.md) tests
the decoder with synthetic pinned-layout siginfo fixtures for every admitted
and rejected signum/si_code row. Real mmap PROT_NONE integration produces and
checks representative SEGV_MAPERR/SEGV_ACCERR delivery; raise/tgkill exercise
only SI_USER/SI_TKILL chaining and are not evidence for a positive admitted
code. BUS rows remain synthetic unless [Task 0](tasks/00-baseline.md) documents a deterministic Linux
delivery mechanism in the baseline manifest. Otherwise it reads the generational TLS descriptor. A
null descriptor decrements and chains. For a nonnull descriptor it checked-
increments that TerminalControl's `inflight_signal_handlers`, acquire-reloads
the identical TLS descriptor, and requires its JitProcess/Vcpu generations,
`host_install_generation` against HostSignalControl, and
`jit_handler_generation` against TerminalControl to match and per-process
`handler_enabled` to be true. On success it release-decrements the global count
exactly once and retains the per-process count until capture/chain selection is
complete. On mismatch it release-decrements the per-process count exactly once,
then the global count exactly once, and chains. Explicit handler-local ownership
bits drive one common epilogue for every branch; only a successful validation
may read the vCPU or
arena. Process construction statically and dynamically requires both AtomicU32
counters to be lock-free on the supported hosts; overflow takes the raw fatal
escape.

Both signal entry symbols are handwritten `global_asm` trampolines, not Rust/C
`extern` functions with compiler-generated prologues. Before the global counter
is owned they access only the three kernel argument registers and the permanent
HostSignalControl pointer, adjust no stack, call nothing and preserve the
arguments in audited scratch registers; checked disassembly enforces this exact
prefix on both ISAs. After the counter is owned they may use only the installed
SignalAltStack and fixed integer instructions until validation has selected
chain, fatal or capture. The generated `linux_signal_abi` module supplies
compile-time offsets for the pinned target's `siginfo_t`, `ucontext_t` and
`mcontext_t`; const assertions compare them with libc types and its canonical
layout hash is a host_bound field. No bindgen result or offset discovered on the
runtime host is accepted.

On x86-64 capture reads all Linux REG_R8..REG_RIP/REG_EFL/REG_RSP entries and
requires the ucontext fpregs pointer nonnull and suitably aligned, then copies
MXCSR and the low 128 bits of XMM0..XMM15 from the kernel FXSAVE-compatible
prefix. NixeFast is forbidden to allocate AVX/SVE values wider than those
architectural 128-bit guest registers. On AArch64 it copies regs[0..30], SP, PC
and PSTATE, then walks the 16-byte-aligned `_aarch64_ctx` chain starting in
`uc_mcontext.__reserved`: every record has size at least 16, a multiple of 16
and a checked end within the current SignalAltStack kernel frame; EXTRA_CONTEXT
continuations obey the same bound. It requires exactly one FPSIMD_MAGIC record
and copies its 32 128-bit V registers, FPSR and FPCR. SVE/ZA and unknown
well-formed records remain in the kernel context but are not allocator-visible;
duplicates, a missing FPSIMD record, malformed size/continuation or a required
field outside the bounded frame is nonrepresentable. A nonrepresentable context
chains only when the PC is proven outside Nixe RX; for an in-arena PC it takes
the raw fatal escape after balanced counter release and never publishes a
partial FaultSlot. The handler changes only the kernel PC, SP and three landing
argument registers, preserving ABI alignment, and returns through the installed
kernel restorer/`rt_sigreturn`; the landing restores all captured allocator-
visible state before retry. [Task 2](tasks/02-lifetime-foundation.md)'s checked disassembly and synthetic maximal
x86 xstate/AArch64 context-chain fixtures verify every offset, bound, counter
exit and edited field.

Per-process `handler_enabled`, not phase, is the arena-dereference authority.
While enabled, including during Stopping, the per-process count protects every
subsequent handler read until it has finished capture or chosen a chain. If the
validated executor's active epoch is zero or native PC lies outside that JIT
process's RX reservation, it decrements before chaining the disposition stored
in HostSignalControl with the original siginfo/ucontext; it does not label
unrelated host faults as guest faults. Chaining follows exactly:

- `SA_SIGINFO` with a user function invokes the saved `sa_sigaction(signum,
  siginfo, ucontext)`; a non-`SA_SIGINFO` user function invokes the saved
  `sa_handler(signum)`. The call-mask and restoration sequence is exactly the
  stack-shim protocol above; no second mask rule exists here. Handler installation mirrors
  saved `SA_RESTART` but always adds SA_ONSTACK; the audited stack-switch shim
  above reproduces the saved stack choice. SA_RESETHAND cannot appear here
  because first installation rejects it before changing either disposition.
- `SIG_IGN` returns to the interrupted context only for a signal already proven
  unrelated to Nixe native execution.
- `SIG_DFL` uses raw async-signal-safe Linux
  syscalls to restore the default disposition, unblock this signal and
  `tgkill(getpid(), gettid(), signum)`. If redelivery unexpectedly returns it
  calls raw `SYS_exit_group(128 + signum)`, exactly 139 for SIGSEGV and 135 for
  SIGBUS. Successful redelivery is externally observed as WIFSIGNALED with the
  original WTERMSIG and may produce a core only according to kernel/resource
  configuration. The fallback is instead WIFEXITED with status 139/135 and
  makes no core-file promise; it exists only because execution after a
  successful default redelivery is impossible. No path calls a null function
  pointer.

No chaining row allocates, locks, logs or calls a non-async-signal-safe libc
wrapper. An
active PC inside the RX reservation with no matching current record/generation,
or a matched record with an impossible fault address/state, follows
InternalFault through the TLS executor's fatal landing. One-time handler
installation is serialized by HostSignalControl and preserves the frozen prior
dispositions through the chaining rows above.

Every phrase “raw process-fatal escape”, “raw fatal escape” or “async-signal-
safe fatal escape” in this section means direct
`SYS_exit_group(NIXE_RAW_EXIT_INTERNAL)` with
`NIXE_RAW_EXIT_INTERNAL = 70`; it never calls a libc wrapper, runs destructors
or reports a guest exit status. Only failed default-disposition redelivery uses
128+signum as specified above.

The async handler requires a nonzero active code epoch. After the protected
validation of TLS, that epoch and the coarse RX-envelope PC, its first slot
operation is an
acquire-release CAS from Idle to Capturing. Only the winner acquire-loads the
one SegmentPcSlot page pointer, performs the bounded interval search and writes
the complete integer/vector/flags/FP machine context plus fault address. It
then release-stores SignalCaptured, edits the captured ucontext to redirect to
the landing stack, release-decrements `inflight_signal_handlers` and returns to
the kernel. The active executor epoch and occupied FaultSlot protect the
subsequent landing even though the async handler count is now zero. The landing
stack trampoline acquire-loads that state before changing it to Resolving. It performs
no lock, allocation, Arc increment, mapping query or guest semantic. A signal
which observes any state other than Idle release-sets the fixed nested-fault
diagnostic and invokes the preinstalled async-signal-safe fatal escape; it does
not touch the occupied payload, directory or landing stack. The fatal escape
uses raw `SYS_exit_group(NIXE_RAW_EXIT_INTERNAL)`, where
`NIXE_RAW_EXIT_INTERNAL` is exactly 70, so it cannot wait
on the interrupted resolver.

`Resolving` runs with the originating executor's active epoch still protecting
the loaded PageFaultTable. It validates the complete record identity and
atomically clones a UnitPin from its directory record before releasing the
directory read state or clearing the epoch. A failed validation is
`InternalFault`.

For `RetryTrackedRam`, the resolver changes no A64State field or captured
register/flag/FP/PC. It performs the authority's single monotonic repair and
verifies the same physical page, mapping/content/protection/observation
generations and a still-callable unit. It compares the persistent last-repaired
tuple before storing the new tuple. For an immediate Open retry, it drops the
temporary resolver UnitPin while the active epoch still protects metadata,
release-stores the fault slot Idle, and only then restores the complete captured
context at `native_start`; an immediate refault can therefore win a new
Idle-to-Capturing CAS and reach the identical-tuple livelock check. The active
epoch remains nonzero throughout.

The phase decision after repair is exhaustive: a validated Open(E) takes that
immediate retry path (a later Closing is safe because the active epoch remains
visible); Closing takes the suspended-or-canonical rule below; Stopping copies
the complete prefault state and typed terminal outcome to preallocated
canonical storage, releases directory state/UnitPin/active epoch, stores the
FaultSlot Idle and enters the terminal landing without retrying or creating a
suspended root. Observing Closed with a nonzero executor epoch is
`JitCommitInvariant`, because Closed can be published only after all such
epochs are zero. Stopped is observed only after the handler has been disabled
and cannot enter this resolver.

If Closing is visible, the resolver may suspend only when, under the JIT-state
mutex, the exact unit is still root-admissible (`Published` current LCQ or
CurrentStable family). It registers the UnitPin and a temporary retry backlink
first, then release-stores the complete `{SuspendedTransition, context,
FaultRecordId, tuple, root}` and only afterward release-clears its active epoch.
The Closing driver acquire-observes this publication before it can observe that
epoch as zero. A Superseded, Invalidating, CutoverOld, pending-cutover or
Withdrawing unit cannot gain this new root and instead takes canonical
completion. The resolver never waits while active.

A suspended ordinary-RAM retry stays in that occupied state and its owner waits
only on the slot's checked sequence. On every wake it acquire-loads and validates
the complete word and payload. CanonicalOnly uses the pin-only path below.
SuspendedTransition may attempt native resumption only in this exact order:

1. through TerminalControl, acquire a new ArenaAdmissionToken by observing
   `Open(E)`, setting this slot's admission bit and reobserving the identical
   phase;
2. acquire an ordinary OpenToken for exactly E. Failure clears the admission
   bit and returns to the checked wait without touching arena storage;
3. now that the arena is protected, acquire-load identical `Active(g)` and the
   exact SuspendedTransition sequence, acquire the closed/open-ordered use gate's count as the unique
   `SuspendedResumeUseToken(sequence)`, then acquire-reload both values. Any
   mismatch releases the count, OpenToken and admission bit in that order and
   returns to the wait;
4. on this host thread, install the precharged SignalAltStack while both signals
   are blocked and receive an `AltStackInstallGuard`; checked-increment the TLS
   publication generation, create the non-cloneable `TlsPublicationToken(P)`,
   release-publish the complete TLS descriptor, transfer the altstack guard into
   that token, and revalidate the token, Active(g), slot sequence and Open(E);
5. acquire-load nonzero global code epoch R and release-publish R in this
   vCPU's `active_code_epoch`; and
6. take the JIT-state mutex, revalidate identical Open(E), Active(g),
   SuspendedTransition sequence, callable unit/native interval and every
   dependency generation, then CAS that exact slot to Resuming.

Contention, Closing or Deactivating before step 4 is an ordinary failed attempt.
After TLS publication, an ordinary failed attempt first unlocks, clears any
published active epoch, calls `detach_native_tls(P)` using its permitted
occupied-suspension condition, releases SuspendedResumeUseToken, OpenToken and
ArenaAdmissionToken in that order, and returns to the checked wait with the
temporary root and UnitPin unchanged. A sigaltstack/TLS platform failure latches
the terminal cause and takes the canonical/terminal cleanup path; it is not
silently retried. No branch publishes an active epoch before it owns admission,
Open, use, altstack and TLS authorities.

If CanonicalOnly won before the Resuming CAS, the owner uses the fault-slot
UnitPin to copy every native value required by the remaining cold/terminal plan
into preallocated canonical storage while the pin is live. If callable or
dependency validation fails while phase is still Open, the owner CASes its
exact SuspendedTransition to CanonicalOnly and removes only its own temporary
backlink under the JIT-state mutex, retaining the slot UnitPin; it then performs
the same copy. In both cases it unlocks, stops all native-metadata reads,
release-clears its active epoch, releases the UnitPin, calls
`detach_native_tls(P)` while the matching CanonicalOnly slot remains occupied,
then release-stores FaultSlot Idle as the last per-vCPU state access which can
be observed by shutdown, and finally
releases SuspendedResumeUseToken, OpenToken and ArenaAdmissionToken in that
order. It then completes the copied canonical or terminal outcome. If no TLS
token had yet been published, the `AltStackInstallGuard`, if any, restores the
prior altstack before Idle and the identical order omits TLS detach; a
CanonicalOnly observed before admission needs none of these three tokens and
uses pin, copy, pin release and Idle only.

Invalidation and shutdown race the owner with the same exact
SuspendedTransition-to-CanonicalOnly CAS. A successful Resuming CAS instead
transfers protection from the temporary root/UnitPin to the already-visible
active epoch: under the mutex it removes the backlink, stops using the pin,
releases it, and converts the resume-use token in place to the ordinary
VcpuUseToken without changing the already-owned use-gate count. It then unlocks, releases the
OpenToken, release-stores FaultSlot Idle and restores `native_start`; TLS,
altstack, ArenaAdmissionToken and ordinary VcpuUseToken remain owned across the
restored native/cold continuation. The maintenance/terminal winner which
publishes CanonicalOnly cuts the temporary root but deliberately leaves the
fault-slot UnitPin for this owner and wakes the exact sequence. Shutdown waits
for the pin/Idle/detach cleanup and consumes its typed terminal result; it never
requires reopen. No branch restores the native PC after losing the exact CAS.

Every other nonfatal disposition copies every dirty value from the prefault map
to A64State, applies the MemoryEffectPlan's exact already-committed-prefix and
writeback rule, materializes lazy state, records the typed request and commits
the partial control budget before release-clearing `active_code_epoch`.
`CompleteEmulatedAccess` then executes the current and remaining cold plan
suffix exactly once. `CompleteObservedCodeWrite` enqueues its preallocated
transition record and the Closed memory-authority stage performs that suffix
once after unlink. `RaiseGuestDataFault` applies only the effects permitted at
that exact stage and delivers its recorded exception. All three finish with
canonical dispatch/exit and never jump back into the faulting CodeUnit. Once
the request/result owns every needed canonical value, the resolver releases its
UnitPin, stops reading native metadata, release-clears the active epoch and
release-stores the fault slot Idle before waiting or entering arbitrary cold
code.

`InternalFault`/Fatal first constructs the complete fixed-size local terminal
tuple `(InternalInvariant, NativeFault)` and immediately performs its one
lock-free `latch_terminal(InternalInvariant, NativeFault)` CAS, closing new
admission before teardown. It then releases any directory state,
UnitPin and temporary retry
root, stops reading native metadata, clears the active epoch, release-stores the
fault slot Idle and calls `detach_native_tls(P)`, including its ordinary
clear-generation acknowledgement and altstack restore. Its audited
`fatal_landing_escape` then switches from the landing stack to the saved system
SP/return continuation, and permanently stops using NativeFrame and both fault
stacks. On that external stack it performs the VcpuUseToken release as its final
per-vCPU arena access, then releases ArenaAdmissionToken before calling
`drive_or_join_terminal()`, and returns the
sticky failure to the scheduler. A resolver which observes an already-visible
Stopping phase uses this identical escape and never waits for ShutdownRecord on
a reclaimable stack. If stack restoration, tid, token balance or that fixed
teardown invariant is corrupt, the raw async-safe fatal escape terminates the
OS process. Shutdown never waits for a resolver which still owns its own epoch,
pin, use token, admission bit, TLS descriptor or reclaimable call frame.

The fault slot retains the last repaired tuple
`(CodeUnitId, CodeVersion, native_start, physical_page_id, mapping_generation,
content_generation, protection_generation, observation_generation)`. A second
`RetryTrackedRam` fault with the identical tuple after a successful repair is a
fatal livelock error. A changed generation is a new classification event. Code,
state maps and fault metadata stay pinned for the entire resolution; no eager
per-access checkpoint, duplicate permission lookup or test-only retry route is
permitted.

MemoryEffectPlan lowers each multi-access instruction by its declared class:

- `SingleRestartable` emits one proven restartable direct host instruction and
  has no architectural or memory effect before that instruction;
- `PrefixVisible` emits one non-cross-granule restartable host access per
  ordered subaccess and a prefault statepoint before each. After success it
  updates every resulting guest value and stage in the current physical state
  before starting the next, so a later map contains the permitted completed
  prefix and cold completion starts at, never before, the faulting subaccess;
  and
- `AllOrNothing` uses one host operation whose documented semantics provide the
  required atomicity or takes that instruction's typed cold preflight before
  any effect. The process owns exactly one non-recursive
  `memory_transaction_mutex`; `MemoryTransactionGuard` is the sole RAII owner
  type for that mutex. The canonical cold path has active epoch zero and
  acquires it with no JIT/cache lock held. MappingChange and every interpreter,
  DMA, GPU, loader, debugger or service writer take the same guard before an
  effect which could touch guest backing. After acquisition it revalidates all
  mapping/backing/permission/observation generations immediately before the
  first effect; mismatch releases the guard with no effect and restarts
  classification. It then executes the complete plan and base/result writeback
  before releasing the guard. `AllOrNothing` guarantees no architectural or
  memory effect on a failed preflight/fault; it does not invent stronger
  multi-location visibility than Arm specifies. An instruction requiring
  inter-vCPU atomic visibility uses one suitable host atomic. If it has no
  suitable direct host operation, every cold implementation and every competing
  emulated atomic/exclusive path uses this same process mutex and performs the
  architecturally required atomic accesses while holding it; no hashed or
  implementation-selected stripe exists.
  Ordinary non-atomic observers retain Arm-permitted visibility. This preflight
  is not added to unrelated scalar/vector accesses.

Loads, stores, atomics, barriers, exclusive-monitor changes and base writeback
from distinct commit stages cannot be fused, duplicated or reordered across
their statepoints. The shared plan, not backend convenience, defines permitted
partial effects and exception ordering.
