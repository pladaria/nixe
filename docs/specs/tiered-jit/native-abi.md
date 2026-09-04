# Gateway, state transfer and helper ABI

[Specification index](README.md) · [Implementation tasks](tasks/README.md) · [Open review items](review-status.md)

Related contracts: [Cranelift fork and backend contract](backend.md); [Control budget and functional sampling](sampling-and-budget.md); [Native fault transport and completion](faults.md).

## Gateway and fast mode

The system-ABI gateway is entered once per native invocation. It:

0. decodes the wire VcpuHandle and acquires its ArenaAdmissionToken using only
   TerminalControl; rejection returns the sticky terminal/Unavailable result
   without dereferencing the vCPU;
1. acquires the process OpenToken for admission epoch E;
2. while both tokens protect the arena, validates and acquires the exact
   VcpuUseToken and publishes/revalidates its TLS descriptor;
3. saves the caller host FP environment and every platform-ABI nonvolatile
   register which the allocator or fixed JIT ABI can modify, establishes
   NativeFrame and rejects recursive use of that vCPU's frame;
4. acquire-loads the nonzero global code epoch, revalidates Active(g), and
   release-stores the epoch in the vCPU's `active_code_epoch` slot;
5. releases the OpenToken, then and only then resolves the initial BlockKey and
   acquire-loads its coherent DispatchPayload;
6. validates BlockKey, ReachabilityVersion and entry CodeVersion while protected
   by that execution epoch, taking a canonical unavailable/compile exit if the
   payload does not name a callable current entry;
7. pins NativeFrame, the direct-arena base and current poll deadline while
   retaining both budget balances in NativeFrame;
8. lazily activates the guest FP environment; and
9. jumps to the selected canonical entry.

Every gateway return path reaches epoch zero and the required FaultSlot state,
clears TLS with the two-count grace, stops using NativeFrame/per-vCPU storage
and releases VcpuUseToken and then ArenaAdmissionToken in that order. A
SuspendedTransition follows its
explicit occupied-slot exception instead of returning through ordinary Idle
cleanup.

An ordinary cold continuation does not reenter through the system-ABI gateway:
it still owns ArenaAdmissionToken, VcpuUseToken, the established NativeFrame and
the saved host state, but has epoch zero and null TLS. Its sole `cold_reentry`
primitive acquires a fresh process OpenToken, requires the identical Active(g)
and FaultSlot Idle, checked-increments/publishes a fresh TLS publication
generation, acquire-revalidates those conditions, loads and release-publishes a
nonzero current code epoch, then drops the OpenToken before reading the current
DispatchPayload and jumping to canonical ingress. Every failure clears TLS with
the signal grace, drops OpenToken, finishes the canonical caller result and
releases VcpuUseToken then ArenaAdmissionToken; it never reads an executable
pointer. PIC-miss, control-poll, compile and fallback continuations all use this
one primitive. They never recursively invoke the gateway or overwrite its
HostPreservationImage.

No executable or entry address is read before the execution epoch is active.
Closing may begin after step 4, but it must then wait for this executor and its
asserted control poll. This ordering closes the lookup-to-entry retirement
race; an epoch is cleared only after canonical state is complete and the
executor has stopped dereferencing native metadata.

The initial ABI variants reserve:

| Role | x86-64 | AArch64 |
| --- | --- | --- |
| NativeFrame pointer | r15 | x21 |
| remaining poll budget | r14 | x20 |
| direct guest-arena base | r13 | x19 |
| link scratch | r11 | x16/x17 |

The context, arena and poll registers are pinned and saved/restored once by the
gateway. In addition, x86-64 saves/restores `rbx`, `rbp` and `r12`, and AArch64
saves/restores `x22..x29` plus the ABI-preserved low 64 bits of `v8..v15` if the
allocator may modify them. Together with the pinned registers this is every
allocator-visible SysV/AAPCS64 nonvolatile register; the fork may expose no
additional callee-saved register until this list and HostPreservationImage are
versioned. Link scratch registers are globally unavailable to register
allocation; no EntryContract or ExitStateMap may place a guest value in them.
Generated guest units emit no explicit SP adjustment or host `ret`, and guest
branches/calls/returns never grow the host call chain. The balanced implicit
x86 `call`/helper `ret` or AArch64 `blr`/`ret` of a declared Leaf helper is the
only transient host return frame and restores the identical stable JIT SP.

Each registered vCPU owns one writable, nonexecutable NativeFrame; concurrent
or recursive native use of that frame is forbidden. Its Rust type is
`repr(C, align(64))`, has ABI version 1, total size `0x4400`, and this exact
common layout on both hosts:

| Range/offset | Field |
| --- | --- |
| `0x000` | `frame_magic: u64 = 0x4e49_5845_4a49_5431` |
| `0x008` | `abi_version: u32 = 1` |
| `0x00c` | `target_isa: u16` (`1=x86_64`, `2=aarch64`) |
| `0x00e` | `in_native: u8` (`0=idle`, `1=active`) |
| `0x00f` | `fp_owner: u8` (`0=host`, `1=guest`) |
| `0x010..0x040` | A64State, vCPU-runtime, arena, fault-slot, canonical-landing and fatal-landing pointers, in that order |
| `0x040` | `admission_epoch: u64` |
| `0x048` | `code_epoch: u64` |
| `0x050` | `sample_remaining: i64` |
| `0x058` | `slice_remaining: i64` |
| `0x060` | `armed_span: i64` |
| `0x068` | `saved_poll_remaining: i64` |
| `0x070` | `exit_kind: u32` |
| `0x074` | `exit_flags: u32` |
| `0x078` | `exit_pc: u64` |
| `0x080..0x090` | `exit_payload: [u64; 2]` |
| `0x090` | saved system SP |
| `0x098` | saved system return address/link value |
| `0x0a0` | helper-stack lower-bound pointer |
| `0x0a8` | `helper_stack_size: u32 = 4096` |
| `0x0ac` | reserved zero u32 |
| `0x0b0` | checked fault sequence u64 |
| `0x0b8..0x300` | target-defined HostPreservationImage specified below |
| `0x300..0x380` | shared guest FP/status scratch |
| `0x380..0x400` | reserved zero bytes |
| `0x400..0x4400` | exactly 16 KiB transfer/spill array |

On x86-64 HostPreservationImage stores `rbx`, `rbp`, `r12`, `r13`, `r14` and
`r15` as u64 at offsets `0x0b8`, `0x0c0`, `0x0c8`, `0x0d0`, `0x0d8` and
`0x0e0`; `0x0e8..0x100` is zero; and `0x100..0x300` is the 512-byte,
16-byte-aligned `FXSAVE64` image restored by `FXRSTOR64`. On AArch64 it stores
`x19..x29` as eleven consecutive u64 values at `0x0b8..0x110`, the low 64 bits
of `v8..v15` at `0x110..0x150`, FPCR and FPSR as u64 values at `0x150` and
`0x158`, and zeros `0x160..0x300`. Saving/restoring happens only in the gateway
and canonical landing; fast inter-unit edges never touch this image.

All pointers are native u64 addresses because only 64-bit hosts are supported.
The layout module exposes typed fields plus assembly constants and asserts every
offset, size and alignment at compile time on each target. Reserved bytes are
zeroed on gateway entry and never read. Full signal register capture lives in
the separate preallocated fault slot/landing stack, not in NativeFrame.

Within the 16 KiB array, `0x400..0x0c00` is the first 2 KiB ABI transfer area and
`0x0c00..0x4400` is the 14 KiB backend spill arena. The backend addresses every
spill relative to the pinned frame register, never SP, and reports its maximum
extent before publication. `spill_extent_bytes` is zero when there are no
allocator spills; otherwise it is exactly the maximum, over every compiler-
chosen spill, of `relative_offset + byte_width`, where the absolute
`NixeAllocation::NativeFrame.byte_offset` is `0x0c00 + relative_offset`.
Checked verification requires every such range to be nonempty, naturally
aligned, contained in `[0x0c00, 0x0c00 + spill_extent_bytes)`, and
`spill_extent_bytes <= 0x3800`. A required transfer-area location in
`0x400..0x0c00` is governed by the slot ownership below and does not contribute
to spill extent. No allocation may address any other NativeFrame padding or
field.

The 2 KiB transfer area is 128 sixteen-byte slots with this fixed ownership:

- slots 0..64 are the host-register mirror. Integer register N uses slot N on
  AArch64; vector register N uses slot 32+N. The x86-64 mapping uses slots 0..16
  for GPR encodings and 32..48 for XMM encodings. Slot 31 is the sole flags
  mirror on both hosts: x86 stores/restores the full user-visible RFLAGS value
  with the audited pushfq/popfq sequence, while AArch64 stores/restores the
  value read/written by `mrs/msr NZCV`; unused bits are zero on AArch64. A helper saves a live
  caller-clobbered allocation only in its register's mirror slot;
- slots 64..96 are bridge parallel-copy scratch. The bridge planner uses the
  lowest available slot and releases all such slots before entering its target;
  and
- slots 96..128 are fault/control scratch and may be used only after ordinary
  guest-value allocations have been described by the relevant state map.

Ranges are half-open. A compile-time layout module defines these constants,
uses `repr(C)`/offset assertions for NativeFrame and is shared by Rust, generated
code and gateway assembly. Two uses with overlapping lifetimes may never alias
one transfer slot.

`HostReg { class: Flags }` is permitted only for a RecoverableMemory statepoint
or as a LeafCall `live_across` allocation which its veneer saves in slot 31.
It is forbidden in an entry live-in, exit/patch contract, bridge copy program or
canonical-boundary result. Before any such boundary, lowering materializes
`NzcvHostFlags` to architecture-independent `NzcvBits32` in a GPR or owned
NativeFrame transfer slot; the inverse conversion occurs only inside the
receiving unit. The fork verifier rejects a live host-flags value crossing a
boundary and includes the slot-31 save/restore in helper-stack and copy-program
bounds. Thus helper clobber masks may name RFLAGS/NZCV without an unstated
mirror location.

The gateway reserves exactly 4096 contiguous host-stack bytes and establishes a
stable JIT SP at their midpoint. The lower 2048 bytes contain all callee stack
growth, including the x86 call return word/red zone and the complete transitive
Leaf-helper call depth; the upper 2048 bytes contain outgoing system-ABI stack
arguments. On x86-64 the stable pre-call RSP is 16-byte aligned, so the helper
observes `RSP % 16 == 8`; on AArch64 SP remains 16-byte aligned. The fork reports
`max_outgoing_args` and each descriptor/build verifier reports maximum callee
low-water use. Either value above 2048, a variadic signature or an unbounded/
recursive Leaf call graph is an implementation failure before publication.
Canonical helpers restore/use the ordinary runtime stack after fast-mode exit.
Fast units see the same stable SP at every entry/exit and never adjust it.
Helper veneers save only mapped live caller-clobbered values in transfer mirror
slots, suspend guest FP ownership when required, make the normal SysV/AAPCS64
call, restore successful continuation live-ins and release every slot before
continuing. On AArch64 the gateway saves its caller's x30 once and canonical
exit jumps to a gateway landing address stored in NativeFrame; unit/helper calls
may use x30 normally and guest calls never use it as a host continuation.

Backend spill extent above 14 KiB is handled before publication and never by
growing SP or slicing emitted machine code. LCQ first tries the full natural
block; after each overflow it replaces attempted instruction count N with
`max(1, floor(N / 2))`, discards all output and recompiles the shortened
immutable prefix. The first fitting prefix ends in a `ResourceCut` static edge
to the first omitted guest PC. No monotonic spill-size assumption is made. From
the 512 ceiling there are at most ten total attempts (nine reductions to one);
an overflow at N=1 is the precise `JitAbiCapacityExceeded` implementation
failure. HCQ removes the highest deterministic discovery-ordinal removable
canonical block, drops every block no longer reachable from `discovery_root`,
recomputes entries/liveness/exits/island demand, and recompiles. A block is
removable only when it is not a mandatory endpoint and removing it preserves
reachability from `discovery_root` to every mandatory endpoint. Removed blocks
are never re-added in that BuildToken. If no such block exists and the connected
core still overflows, the exact fingerprint records
`HcqRejected(BackendSpillCapacity)`. A unit never grows an unbounded frame,
silently changes ABI or splits already-emitted code.


## State transfer

Canonical A64State remains the architectural authority outside native
execution. Within a unit, guest values remain in SSA/native registers. Every
control transfer is classified during lowering as exactly one of these three
protocols; a linker may not change the classification:

- an **internal edge** joins blocks in the same HCQ CodeVersion. It stays in SSA,
  has no entry adapter, state map, dispatch lookup, poll or canonical store of
  its own, and is legal only when no architectural/control boundary lies on the
  edge;
- a **fast external edge** stores no GPR, SP, vector, NZCV, FPSR, FPCR or PC to
  A64State. Its immutable source ExitStateMap supplies every target live-in, a
  version-specific bridge performs only the required parallel copies, and the
  target enters through its matching FastEntryContract; or
- a **canonical edge** materializes every dirty architectural component,
  including lazy NZCV/FPSR, writes the `CanonicalBoundaryRecord`'s exact current
  or continuation PC, restores host FP ownership, and reaches a gateway landing
  which clears the execution epoch before invoking cold Rust. A helper outcome
  obtains that PC from its HelperDescriptor; a control, poll, fault or resource
  boundary has its own immutable record emitted by lowering. On that landing
  A64State is an exact architectural snapshot: no dirty guest value remains
  solely in a host register or NativeFrame.

Every external patchpoint retains a complete canonicalizable source map because
its permanent fallback may run after any later unlink. Target liveness may omit
a bridge copy, but it may not make the source value unavailable at that
patchpoint. Values are discarded only on an HCQ internal edge, or after a fast
target has been entered, when the shared analysis proves every path overwrites
them before a guest read, fault, helper, poll, exception, debugger/scheduler
hook or canonical exit. LCQ therefore needs no successor decode: it retains all
dirty values at its terminal patchpoint and a later linker copies only the
actual target live-ins.
A compatible fast edge preserves the active guest FP environment and performs
no entry safepoint. Lazy NZCV remains in its physical producer only when all
intervening infrastructure is flag-transparent; otherwise the source
materializes the exact live flag bits into a mapped transfer location and the
target recreates its declared representation. A fast edge never stores PC.

Use/def, partial-write, live-in and live-out analysis has one implementation
shared by LCQ entry formation, HCQ entry formation, dirty commits and helper
boundaries. A test-only alternate classifier is forbidden.

At every external edge, canonical-helper/control boundary and potentially
faulting native instruction, the backend exports a complete
`PhysicalStateMap` with one entry for each of X0..X30, SP, V0..V31, NZCV, FPCR,
FPSR and PC. A Leaf-helper map instead contains exactly every descriptor input
and value live across its guaranteed-success call. An entry is one of:

```text
CanonicalClean(a64_state_offset, representation)
HostReg(preg, representation)
NativeFrame(byte_offset, width, representation)
Constant(bits, representation)
LazyFlags(recipe_id, inputs)
```

`LazyFlags.inputs` is an ordered array of explicit nonarchitectural
`StateLocation` records, each HostReg, NativeFrame or Constant with its own
representation. It is not a reference to the current architectural X/V entry:
an arithmetic operand may remain needed after that guest register has been
overwritten. These retained operands participate in liveness, NativeFrame
overlap, reserved-register and final-offset verification exactly like mapped
guest values and exist only while their recipe is live.

Every other A64State/system field and the shared exclusive-monitor object stays
canonical at each guest-instruction commit stage and is accessed only by its
typed lowering/helper/MemoryEffectPlan; it is never an undocumented SSA live-
out. Caching another architectural component requires adding it to this map,
StableEncode/ABI versioning and the [Task 0](tasks/00-baseline.md) coverage proof.

`representation` fixes bit width, lane shape and any required extension; a
consumer never infers these from a host register class. `NativeFrame` ranges
must lie wholly in the declared transfer or spill area and respect the value's
alignment. `CanonicalClean` asserts that the named A64State field already has
the current value, so an entry marked dirty may not use it. `LazyFlags` is
permitted only for NZCV/FPSR and names an immutable, verifier-known recipe whose
explicit input records are carried by the map. PC is a compile-time guest PC
for an ordinary boundary and the descriptor-specified continuation for a
completed cold access. The verifier rejects a missing component, overlapping
live NativeFrame ranges, reserved-register use or a recipe cycle.

An `ExitStateMap` is that complete fast-external-edge map plus dirty bits and
the lazy-state recipe. A `CanonicalExitMap` is the same complete map and dirty
information plus exact `exit_kind`, boundary ID and current/continuation PC;
every canonical helper, control, poll, fault and resource edge has one. A
`CanonicalBoundaryRecord` owns that map and any typed payload needed by the
landing. A bridge reads only the ExitStateMap projection demanded by the target
FastEntryContract; a fallback can canonicalize every dirty component. A fault
record retains its complete prefault map. Maps are derived after final register
allocation and verified against final instruction offsets; compiler SSA is
never the sole owner of a required value at a link, helper, control or fault
boundary.

Each published guest PC has exactly two entry contracts:

- canonical ingress reads its true live-ins from current A64State and is used
  by the gateway and a compile/link miss; and
- fast ingress names the required representation and exact physical register
  or fixed NativeFrame slot for each true live-in and is used only by a bridge
  built for that source ExitStateMap, target contract and exact pair of
  CodeVersions. If a required source value is `CanonicalClean`, the bridge
  loads that named A64State field; it is therefore not an empty bridge, but it
  still performs no canonical store.

Static and dynamic link bridges perform cycle-safe parallel copies and use the
fixed transfer slots when source and target physical locations differ. A
compatible static edge whose source locations already satisfy the target fast
contract has an empty bridge followed by one direct branch.

An indirect branch probes with the complete [ExitSiteKey](runtime-state.md#keys-and-dispatch-publication). A PIC hit jumps to the immutable
bridge compiled for that site and target fast-entry contract. It does not
commit A64State, leave the execution epoch or interpret a register-move plan.
Only a PIC miss canonicalizes the dirty state before its cold resolver. Dynamic
BridgeUnits are bounded by their fixed registry-slot permits, deduplicated weakly and
retired through the same epoch/cache lifecycle as normal code.

The PIC-miss canonical landing clones a source UnitPin from the still
epoch-protected source record before clearing its active epoch and carries that
pin plus the immutable ExitStateMap identity into the resolver. Thus leaving
native execution never creates an unprotected source-metadata gap.

The ABI does not promise that all 31 GPRs and 32 vectors remain permanently
assigned to host registers across independently allocated LCQ blocks; HCQ
provides broad cross-block SSA where that matters.

Poll arithmetic, confinement, PIC/RSB comparisons, bridge copies and every
other JIT-infrastructure instruction participate in the same liveness model.
If one would clobber host flags holding a live lazy guest NZCV producer, the
compiler must schedule it after the producer's final consumer, use a
flag-transparent form, or preserve/materialize the exact required guest flags
first. Infrastructure may never silently destroy architectural flags. This is
paid only when such a producer is live, not as an unconditional NZCV store at
each boundary.

## Helpers and architectural boundaries

Every callable helper has one generated, immutable `HelperDescriptor` in the
shared semantic table. It records the exact SysV x86-64 and AAPCS64 signature
(argument order, width, signedness and return representation), `Leaf` or
`Canonical` class, architectural read/write set, host caller-clobber set,
NZCV/FP effects, separate maximum outgoing and transitive callee-low-water bytes,
possible outcomes
and the guest PC associated with each continuation. Lowering and the state-map
verifier consume this same descriptor; handwritten call-site declarations are
forbidden. [Task 0](tasks/00-baseline.md) verifies that every descriptor fits both the 2048-byte
outgoing half and 2048-byte transitive callee half on both hosts.

A `Leaf` helper is allowed only when its descriptor and implementation are
verified nonblocking, nonallocating, nonunwinding, signal-safe with respect to
the JIT, independent of mapping/device/scheduler state, incapable of raising a
guest exception, and unable to compile, publish or invalidate code. It receives
all inputs as explicit scalar/vector ABI arguments and returns only its declared
result. Its veneer saves precisely the live caller-clobbered locations in their
register-mirror slots, suspends guest FP ownership only when the descriptor
requires it, calls by the platform system ABI, restores the declared successful
continuation state and jumps onward without clearing the execution epoch.
Clobber intersection is bit-precise. On AArch64, a live 128-bit allocation in
v8..v15 intersects the declared `.hi64` clobber and the veneer saves/restores
the complete 128-bit mapped value in that register's mirror slot; a live value
confined to the ABI-preserved low 64 bits needs no save. [Task 0](tasks/00-baseline.md)'s descriptor and
machine-code verifier enforces these masks for every Leaf helper.

Every helper which fails any `Leaf` condition is `Canonical`. Generated code
does not call it directly: it takes a canonical edge, commits all dirty state,
sets the descriptor's exact current PC, reaches the gateway landing and clears
the execution epoch before Rust runs. Its result is one of the descriptor's
closed outcomes (`ContinueAt(pc)`, `GuestException(exception_pc, syndrome)`,
`SchedulerExit(reason)` or `InternalFailure(code)`); continuation performs a
fresh canonical dispatch and loads only that entry's true live-ins. SVC, BRK,
FP-mode writes, unsupported instructions, emulated memory, mapping/device
operations and any helper which can block or allocate are always Canonical.
Pure supported integer, branch, RAM and SIMD operations do not use a generic
helper.

Guest FPCR/FPSR and host FP ownership retain one implementation for both tiers:

- the caller FP environment is saved once at gateway entry;
- a compatible guest FP segment remains active across native links;
- sticky host/software status is materialized only when observed, replaced or
  leaving fast mode;
- FPCR/FPSR writes end the current segment with exact ordering; and
- a general Rust helper suspends the guest FP segment and restores it only on a
  successful continuation.
