# Cranelift fork and backend contract

[Specification index](README.md) · [Implementation tasks](tasks/README.md) · [Open review items](review-status.md)

Related contracts: [Supported baseline and generated manifest](baseline.md); [Gateway, state transfer and helper ABI](native-abi.md); [Compilation and publication pipeline](publication.md).

The upstream baseline for the fork is the official stable release
[Wasmtime v48.0.1](https://github.com/bytecodealliance/wasmtime/releases/tag/v48.0.1),
whose tag resolves to commit `7bac2c2775808aaec5d4aa5627a5e447b51102cf` and whose
workspace uses Cranelift `0.135.1`. [Task 0](tasks/00-baseline.md) creates one immutable Nixe fork commit from
that exact base, records both commits in the baseline manifest and changes
every `cranelift-*` dependency used by Nixe to the same git source and exact
`rev`. Mixing the current crates.io `0.134.3` packages, another git revision or
an uncommitted path override is a build failure. Any later upstream or fork
revision change reopens [Task 0](tasks/00-baseline.md) and [Task 1](tasks/01-abi.md) and reruns the complete encoder and
native ABI proofs; branch names and floating tags are forbidden as production
dependency selectors. The development branch itself is defined below.

## Fork workspace and development branch

The development checkout is `/home/pladaria/projects/wasmtime`, with local
branch `nixe` and project-owned remote `origin` pointing to
`https://github.com/pladaria/wasmtime.git` or its SSH equivalent
`git@github.com:pladaria/wasmtime.git`. All Nixe-specific Cranelift and
icache-coherence changes are implemented and tested in that checkout on `nixe`.
The branch is initially created at the exact upstream release commit:

```sh
git -C /home/pladaria/projects/wasmtime switch -c nixe 7bac2c2775808aaec5d4aa5627a5e447b51102cf
```

Before first creation, verify the remote, the release object and a clean
worktree. If local `nixe` already exists, verify that the pinned release commit
is its ancestor and switch to it without resetting or recreating it. If only
`origin/nixe` exists, validate that ancestry and create the local tracking branch
from it. Preserve existing work; a dirty checkout or an unrelated existing
`nixe` branch must be resolved before switching, without automatic discard or
stash. Read the checkout's own contribution and agent instructions before
making fork changes.

The branch advances through ordinary development commits. Once publication is
authorized, publish it as `origin/nixe`; do not rewrite a commit already pinned
by Nixe. For each accepted integration, record the tested full commit `F` from
this branch in the baseline manifest, every affected Cargo `rev` and Cargo.lock,
and run the required fork/ABI gates. `branch = "nixe"`, a local path override,
or the current branch tip without its recorded commit never replaces that pin.
CI may use a detached checkout of `F`; it does not need a local `nixe` branch.
Creating or switching the development branch alone does not complete Task 0
or authorize a remote push.

## Immutable dependency integration

After [Task 0](tasks/00-baseline.md) creates the immutable fork commit `F`, Nixe's workspace manifest
must also contain exactly
`wasmtime-jit-icache-coherence = { package =
"wasmtime-internal-jit-icache-coherence", git =
"https://github.com/pladaria/wasmtime", rev = "F" }`. `F` is the full fork Git
OID recorded in the same baseline manifest, not a literal placeholder, branch
or tag. Every transitive `cranelift-*` package and this package must resolve to
that one Git source/revision in `cargo metadata --locked`; the Rust crate path
is consequently `wasmtime_jit_icache_coherence`. [Task 0](tasks/00-baseline.md) updates and checks the
lockfile immediately after creating `F`.

Stock Cranelift exposes only one pinned register and does not provide
prologue-free multi-entry blocks, link patchpoints or semantic block-label
offsets. The implementation therefore uses a maintained Cargo patch of the
pinned Cranelift revision rather than weakening this ABI. The public fork
boundary is `NIXE_FORK_API_VERSION = 1` and exactly two calls: one pure borrowed
reservation query and one compile call which takes ownership of its request:

```text
Context::nixe_reservation_bound(
    isa: &dyn TargetIsa,
    request: &NixeCompileRequest,
) -> Result<NixeReservationBound, NixeCompileError>

Context::compile_nixe(
    isa: &dyn TargetIsa,
    control_plane: &mut ControlPlane,
    request: NixeCompileRequest,
) -> Result<NixeCompiledCode, NixeCompileError>

NixeCompileRequest {
    api_version: u32,                         // exactly 1
    call_conv: NixeFast,                      // no other value
    target: HostIsa,
    profile: LcqProduction | HcqProduction | Proof,
    entries: Vec<NixeEntryRequest>,
    patchpoints: Vec<NixePatchRequest>,
    statepoints: Vec<NixeStatepointRequest>,
    helper_abis: Vec<NixeHelperAbiDescriptor>, // sorted unique HelperId
    leaf_calls: Vec<NixeLeafCall>,
}
NixeHelperAbiDescriptor {
    helper: HelperId,
    args: Vec<ABITypeToken>,
    result: Void | Direct(ABITypeToken),
    arg_locations: Vec<NixeAbiLocation>,
    result_location: None | Some(NixeAbiLocation),
    gpr_clobber_mask: u64,
    vector_full_clobber_mask: u64,
    vector_hi64_clobber_mask: u64,
    x86_status_clobber_mask: u8,
    aarch64_status_clobber_mask: u8,
    outgoing_args_high_water: u16,
    callee_low_water: u16,
    nzcv_effect: Preserve | Clobber | Produce,
    fp_effect: Preserve | Host | Guest,
}
NixeAbiLocation =
    HostReg { class: Gpr | Vector, encoding: u8 }
  | Stack { byte_offset_from_stable_helper_sp: u16, byte_width: u16 }
NixeEntryRequest {
    id: NonZeroU32,
    block: cranelift_codegen::ir::Block,
    landing_kind: NativeLandingKind,
    live_ins: Vec<NixeLiveIn>,
}
NixeLiveIn {
    component: StateComponent,
    value: cranelift_codegen::ir::Value,
    representation: StateRepresentation,
    required_location: NixeRequiredLocation,
}
NixePatchRequest {
    id: NonZeroU32,
    ir_inst: cranelift_codegen::ir::Inst,
    kind: StaticExternal | DynamicExternal | CanonicalExit,
}
NixeStatepointRequest {
    id: NonZeroU32,
    kind: RecoverableMemory | CanonicalBoundary | PatchBoundary | LeafCall,
    ir_inst: cranelift_codegen::ir::Inst,
    operands: Vec<NixeStateOperand>,
}
NixeStateOperand {
    component: StateComponent | LazyRecipeInput(NonZeroU32),
    value: cranelift_codegen::ir::Value,
    representation: StateRepresentation,
}
NixeLeafCall {
    helper: HelperId,
    ir_inst: cranelift_codegen::ir::Inst,
    args: Vec<NixeLeafArg>,
    result: Void | Direct(Value, ABITypeToken),
    live_across: Vec<NixeStateOperand>,
    statepoint_id: NonZeroU32,
}
NixeLeafArg {
    value: cranelift_codegen::ir::Value,
    abi_type: ABITypeToken,
}

NixeCompiledCode {
    api_version: u32,
    buffer: Vec<u8>,
    alignment: u32,
    relocations: Vec<NixeRelocation>,
    entries: Vec<NixeEntryRecord>,
    patchpoints: Vec<NixePatchRecord>,
    statepoints: Vec<NixeStatepointRecord>,
    traps: Vec<NixeTrapRecord>,
    frame: NixeFrameUsage,
    reservation_bound: NixeReservationBound,
}
NixeFrameUsage {
    spill_extent_bytes: u32,
    explicit_stackslot_bytes: u32,
    outgoing_args_high_water: u32,
    callee_save_bytes: u32,
    prologue_bytes: u32,
    epilogue_bytes: u32,
    dynamic_sp_adjustments: bool,
}
NixeReservationBound {
    image_bytes_upper: u32,                   // includes CLIF text/data/internal islands
    alignment_upper: u32,
    external_island_slot_demand_upper: u16,  // 0..=4096 shared-segment slots
}
NixeRelocation {
    offset: u32,                              // from buffer[0]
    kind: X86CallPCRel4 | Arm64Call,
    target: InternalOffset(u32) | Helper(HelperId),
    addend: i64,
}
NixeEntryRecord {
    id: NonZeroU32,
    entry_offset: u32,                        // from buffer[0]
    landing_range: [u32, u32),
    live_ins: Vec<NixeAllocation>,            // request order
}
NixePatchRecord {
    id: NonZeroU32,
    kind: StaticExternal | DynamicExternal | CanonicalExit,
    patch_range: [u32, u32),                  // from buffer[0], nonempty
    rewrite_unit: X86Rel32FiveBytes | A64BranchFourBytes,
    fallback_bytes: Vec<u8>,                  // length == range length
    statepoint_id: NonZeroU32,
}
NixeStatepointRecord {
    id: NonZeroU32,
    kind: RecoverableMemory | CanonicalBoundary | PatchBoundary | LeafCall,
    native_range: [u32, u32),                 // exact final MachInst(s), nonempty
    trap_code: None | Some(NonZeroU8),
    allocations: Vec<NixeAllocation>,         // request operand order
}
NixeTrapRecord {
    native_offset: u32,                       // from buffer[0]
    trap_code: NonZeroU8,
    statepoint_id: NonZeroU32,
}
NixeAllocation =
    HostReg { class: Gpr | Vector | Flags, encoding: u8,
              representation: StateRepresentation }
  | NativeFrame { byte_offset: u32, byte_width: u16,
                  representation: StateRepresentation }
  | Constant { byte_len: u8, little_endian_bits: [u8; 16],
               representation: StateRepresentation }
NixeRequiredLocation =
    HostReg { class: Gpr | Vector | Flags, encoding: u8 }
  | NativeFrame { byte_offset: u32, byte_width: u16 }
StateComponent = X(0..30) | Sp | V(0..31) | Nzcv | Fpcr | Fpsr | Pc
StateRepresentation =
    Integer { bits: 8 | 16 | 32 | 64,
              extension: Exact | ZeroExtended | SignExtended }
  | Float16 | Float32 | Float64 | Vector64 | Vector128
  | NzcvHostFlags | NzcvBits32 | Fpcr32 | Fpsr32 | GuestPc64
NixeCompileError {
    kind: ApiVersion | WrongTarget | MalformedRequest | Verify | Optimize |
          Lower | Regalloc | RegallocChecker | Frame | Statepoint |
          Relocation | Capacity | InternalInvariant,
    request_id: None | Some(NonZeroU32),
    backend_code: u32,
}
```

HelperCatalogEntryV1 deterministically derives the target-specific
NixeHelperAbiDescriptor. Every leaf call has exactly one equal-ID descriptor,
the call's arg/result tokens must byte-equal it, and unused/duplicate descriptor
rows are MalformedRequest. The fork consults only this owned vector; it has no
callback, global Nixe table or hidden ABI default. These types/fields are part
of ForkApiLayoutV1.
GPR/vector mask bit n is the manifest's target HostReg encoding n.
`vector_full_clobber_mask` means all 128 bits; the hi64 mask is legal only for
AArch64 v8..v15, is disjoint from the full mask, and means bits 64..127 only.
For x86-64, `x86_status_clobber_mask` bits 0, 1 and 2 are respectively rflags,
mxcsr and x87 and the AArch64 mask is zero. For AArch64,
`aarch64_status_clobber_mask` bits 0, 1 and 2 are respectively nzcv, fpcr and
fpsr and the x86 mask is zero. Bits 3..7 are zero. `nzcv_effect` and
`fp_effect` describe guest-state semantics independently of these host ABI
clobbers. Every Stack offset is from the stable helper SP established by
NixeFast, matching the catalog grammar; it is never from the pre-gateway system
SP. The parser rejects a clobber token which cannot be represented exactly by
these fields.

`fork_api_layout_sha256` is
`SHA256("nixe-fork-api-layout-v1\0" || StableEncode(ForkApiLayoutV1))`.
`ForkApiLayoutV1` contains, in UTF-8 type-name order, one row for every public
request/result type with its logical field names, declaration ordinal and
closed type token; every enum variant/discriminant; every fixed `repr(C)`
size/alignment/field offset used across assembly or crate boundaries; the full
per-target HostReg encoding table; relocation kind/addend mapping; and
NIXE_FORK_API_VERSION. Sequence/pointer ownership is represented by schema
tokens (`owned_vec<T>`, `borrowed_ir_handle<T>`) rather than unstable Rust Vec
layout. Rows and tokens use the [StableEncode rules](units-and-registries.md) and the manifest keeps
golden bytes plus the digest. Any field/order/discriminant/register-number or
ownership change increments the API version and changes the hash; a hand-
maintained free-form signature string is forbidden.

Each production compile and both proof encoders construct a fresh default
`ControlPlane` immediately before this call, with chaos/perturbation disabled,
and discard it afterward. It is never shared between jobs and no runtime seed
or environment input may alter it. [Task 0](tasks/00-baseline.md) records the exact fork default in the
fork API hash and has a negative test which enables perturbation only in the
test harness and verifies that production construction rejects that
configuration. Deterministic-output claims never quantify over an arbitrary
caller-supplied ControlPlane.

`Context::nixe_reservation_bound(isa, &request)` is the pure fork call listed
above;
it is pure, does not optimize/lower or mutate Context, and returns the
`NixeReservationBound` derived from [Task 0](tasks/00-baseline.md)'s production encoder bounds and the
request's checked IR counts. The compiler reserves that charged span before
`compile_nixe`; successful output must satisfy `buffer.len() <=
image_bytes_upper`, `alignment <= alignment_upper` and final distinct external
island demand no greater than `external_island_slot_demand_upper <= 4096`.
The query counts one slot per independently patchable external exit and one per
distinct helper-island key; it never charges a private 64-KiB island to a unit.
Allocation reserves those slots from the chosen segment's shared 4096-slot
bitmap and releases any conservative excess before publication. Exceeding any
bound is
`NixeCompileError::InternalInvariant`, aborts the unreachable reservation and
latches the JIT failure; it never truncates output.

Ranges above are half-open, checked within `0..=buffer.len()`, and offsets are
always relative to the first byte of `buffer`; no record uses an RX/RW pointer.
`landing_range` begins at `entry_offset`. Constant bytes beyond `byte_len` are
zero, `byte_len` is `1..=16`, and its representation width equals the used
bits. `HostReg.encoding` uses the fork's generated target register-number table
whose hash is in `fork_api_layout_sha256`; another numbering is not accepted.
An entry's output allocations equal its requested locations and
representations exactly. A PatchBoundary statepoint is referenced by exactly
one patch record and its range contains that patch range; only
RecoverableMemory has `Some(trap_code)`, and every trap record and such
statepoint reference each other bijectively. `fallback_bytes` are the bytes
already present in the returned buffer. A rewrite must use the same fixed-width
unit and is performed only under Closed, so no unstated atomic live patch is
assumed.

All compile-local nonzero-u32 IDs are assigned in deterministic lowering order,
are unique within their own array and occur sorted in both request and output.
The fork rejects unknown IDs, duplicate IDs, an IR object from another Context,
a missing record or an extra output record. There is an exact bijection between
requested entries and `NixeEntryRecord`s, patch requests and
`NixePatchRecord`s, and statepoint IDs and `NixeStatepointRecord`s. A Leaf call
references exactly one `LeafCall` statepoint and no other call may reference it.
Result vectors are sorted by ID; relocations and traps are sorted by `(offset,
kind, target)`. These types and enum discriminants are part of the fork API hash
in the baseline manifest.

Internally the fork still uses the stock optimization/lowering/emission
pipeline. In the pinned source, `Context::compile` returns a borrowed
`&CompiledCode`; `compile_nixe` immediately calls `take_compiled_code()` before
the Context can be cleared or reused and copies/normalizes every referenced
`FunctionParameters::user_named_funcs` entry while those parameters are still
alive. The fork adds one consuming
`MachBufferFinalized<Final>::into_nixe_parts(self) -> NixeMachBufferParts`
method in `cranelift-codegen`. It destructures the private fields inside that
crate, converts the `data`, relocation and trap `SmallVec`s with `into_vec()`,
and moves the alignment and every custom Nixe record into an owned result;
`compile_nixe` consumes `compiled_code.buffer` through exactly that method.
Calling the public borrowing accessors `data()`, `relocs()` or `traps()` and
describing their slices as moved is forbidden; an implementation which elects
to copy instead must amend this contract and its allocation bound. The owned data
contains machine text, padding, jump tables, constant pools and Cranelift's
internal veneers/islands as one position-dependent image; Nixe never slices it
at a text-size estimate or relocates its parts independently. Nixe's separately
reserved per-segment 64-KiB source-link island is outside this buffer and shared
by every unit in that segment. Final fit uses exactly `buffer.len()` body bytes
plus the reported slot demand. The echoed
`reservation_bound` must byte-equal the pure precompile query; a mismatch is a
backend invariant failure.

Existing `value_labels_ranges`, `bb_starts`, `sequence_point` and `MachTrap`
may be backend inputs, but none is accepted as an EntryContract,
PhysicalStateMap or statepoint result. Before optimization, `entries` registers
every public block as an external CFG root. Unreachable-code elimination seeds
all those roots, the dominator structure becomes a forest, and optimization
must treat a requested entry as having a synthetic external predecessor which
defines only its declared live-ins. An internal predecessor may still pass
block arguments, but no value defined solely on that predecessor may dominate
the external entry. Emission places the required landing instruction and
entry-label offset at the actual block ingress; register allocation enforces
the requested physical live-ins for both external and internal predecessors.
Tests include a disconnected selected entry and a selected entry with an
internal predecessor whose incoming values differ from its external contract.

For every `NixeStatepointRequest`, the fork attaches the ID and keepalive uses
to the actual potentially faulting/boundary MachInst before scheduling. It
returns the exact nonempty final native `[start, end)` and, after register
allocation, one allocation per ordered operand: `HostReg`, `NativeFrame` or
`Constant` with the requested representation. A recoverable-memory statepoint
has exactly one trap on that same machine instruction whose value is
`const NIXE_RECOVERABLE_MEMORY_TRAP: TrapCode = TrapCode::unwrap_user(1);` and
exports it as the returned `NonZeroU8` from `.as_raw()`. That value is deliberately
reused: in the pinned fork `TrapCode` is `NonZeroU8`, user values are only
`1..=250`, and `251..=255` are reserved. The unique identity is the
`statepoint_id` plus its final native range/offset, and the required bijection is
between `NixeStatepointRequest`, `NixeStatepointRecord` and `MachTrapRecord`, not
between numeric TrapCode values. The post-optimization and post-RA
verifiers reject folding two statepoints, motion across their ordered effects,
an allocation unavailable throughout the reported instruction range, or a trap
attached to any other instruction.

Every direct recoverable guest-memory subaccess uses that same fixed nonzero
TrapCode and `MemFlags` with `notrap = false`, `readonly = false` and
`can_move = false`. `aligned` is true only when that MemoryEffectPlan proves the
effective address has the host instruction's required alignment. Endianness
and alias region are set explicitly from the plan; the native-default and
generic heap aliases are forbidden. The fork's post-optimization verifier
requires the same statepoint ID, trap record, alias, endianness and effect order and
rejects merging, duplication, deletion or reordering across a commit stage.
Proof tests generate more than 250 recoverable fault sites in one compilation
and require distinct statepoint/range records with the shared trap value.

`NixeLeafCall` is a fork pseudo-operation, not an unconstrained ordinary call.
It consumes the generated HelperDescriptor, uses exactly its system-ABI
argument/result locations, clobber mask, outgoing high-water, mirror saves and
FP suspend/restore rule, and reports them in the LeafCall statepoint. A Leaf
helper returns either `void` or one direct scalar/vector value. Variadic,
multiple-return, implicit/explicit sret and aggregate returns are rejected; any
additional result is written through an explicitly declared `mut_host_ptr64`
argument. Calls to an unlisted ExternalName and implicit libcalls are forbidden
in NixeFast.

For NixeFast, successful output must satisfy exactly:

```text
spill_extent_bytes <= 14 KiB
explicit_stackslot_bytes == 0
outgoing_args_high_water <= 2048
callee_save_bytes == 0
prologue_bytes == 0
epilogue_bytes == 0
dynamic_sp_adjustments == false
```

The fork replaces the pinned stock `expect("register allocation")` and
`expect("register allocation checker")`, plus every NixeFast frame-construction
panic edge, with typed `NixeCompileError::Regalloc`, `RegallocChecker` or
`Frame`. A legal or rejected guest input cannot unwind or abort the process
through this interface. The machine-environment hook removes exactly `r15`,
`r14`, `r13` and `r11` on x86-64 and `x21`, `x20`, `x19`, `x16` and `x17` on
AArch64 from allocator-visible registers. This is verified for both Cranelift
algorithms: `single_pass` maps to regalloc2 Fastalloc and `backtracking` maps to
Ion.

The normalized relocation type has only
`NixeRelocTarget::InternalOffset(u32)` and `Helper(HelperId)`. On x86-64 the
only accepted relocation kind/addend is `X86CallPCRel4/-4`; on AArch64 it is
`Arm64Call/0`. `FinalizedRelocTarget::Func(offset)` is accepted as
InternalOffset only when `offset < buffer.len()`; it is never resolved as an
external helper. `ExternalName::User` is accepted only when the still-owned
FunctionParameters table maps it bijectively to a declared HelperId. `LibCall`,
`KnownSymbol`, `TestCase`, an unmapped User target, absolute, GOT/PLT/TLS and
every other kind/addend are errors. Relocation application checks field
alignment, computes P from the final RX address although bytes are written
through its RW alias, range-checks the displacement and uses the pre-reserved
per-body helper-call island when an otherwise legal helper call is out of
range. There is no late arbitrary veneer allocation. [Task 0](tasks/00-baseline.md) compiles every
accepted CoverageId and HelperDescriptor on both ISAs and fails if any hidden
libcall or non-whitelisted relocation remains.

Nixe declares `cranelift-codegen` with `default-features = false` and production
features exactly `["std", "x86", "arm64"]`; `disas` is enabled only in
dev/proof tests, while `unwind`, `host-arch`, `timing` and incremental-cache
features are absent from production. ISA construction uses
`cranelift_codegen::isa::lookup(Triple)` and the complete manifest setting map,
never `cranelift_native`, so cross-target proof builds do not inherit build-host
features. The baseline contains every shared and selected ISA setting for each
of `LcqProduction`, `HcqProduction` and `Proof`; generation starts from the
pinned commit's defaults, explicitly sets every entry, emits the complete
resulting `Flags::iter()` and ISA-flags map, and CI rejects a missing, extra or
different setting. Both production profiles set `enable_verifier=true`,
`is_pic=false`, `use_colocated_libcalls=false`, `unwind_info=false`,
`preserve_frame_pointers=false`, `enable_probestack=false`, `tls_model=none`,
`stack_switch_model=none`, `enable_pinned_reg=false`,
`regalloc_verbose_logs=false`, `machine_code_cfg_info=true` and
`regalloc_checker=false`; Proof differs only by `regalloc_checker=true` and the
dev-only disassembly feature. LCQ uses `opt_level=none` and
`regalloc_algorithm=single_pass`; HCQ uses `opt_level=speed` and
`regalloc_algorithm=backtracking`. Every x86 `has_*` and AArch64 `has_*` flag,
and AArch64 `use_bti`, maps by name to exactly one frozen `host_feature` bit;
all signing/CSDB flags are false. A host feature not represented in that map is
disabled, not guessed.

If the [Task 0](tasks/00-baseline.md) proof
shows that the required hooks cannot be maintained without replacing
Cranelift's allocator or machine backend, implementation stops and this
specification is amended with the concrete replacement; the implementer does
not choose an undocumented fallback. The patch is limited to:

- reserving the context, arena and poll registers plus every link-scratch
  register;
- redirecting register-allocator spills and stack-slot references to the fixed
  NativeFrame spill arena, with exact maximum-extent reporting;
- emitting prologue-free fast entries and source-local patchpoints;
- exporting selected final block-label offsets;
- accepting physical EntryContracts and exporting final ExitStateMaps at link,
  helper, control and fault boundaries;
- returning the owned final image, relocation and fault/statepoint data to
  Nixe's allocator; and
- selecting jump-appropriate BTI/CET landing pads.

The fork also defines one `NixeFast` internal calling convention on x86-64 and
AArch64. It has no ordinary function parameters or returns, emits no
prologue/epilogue/unwind frame, uses the fixed NativeFrame spill and helper-stack
bases above, ends every unit path in a jump or explicit canonical-exit jump, and
is never used for a Rust/C ABI call. Normal helpers retain the platform system
ABI. The fork rejects a `ret`, dynamic SP adjustment, ordinary stack slot,
unreported outgoing argument, or allocator assignment of any reserved register
in `NixeFast` output.

Bridge and helper veneers use the permanently reserved scratch registers and
the ABI-owned transfer slots. They never borrow an allocator-visible register
without first moving its mapped guest value according to the source state map.

It must not fork instruction semantics, add a second IR or turn every guest
instruction into a custom lowering rule.


### Release provenance and API evidence

The release tag, upstream commit and Cranelift version are distinct manifest
fields. [Task 0](tasks/00-baseline.md) verifies all three against the official upstream tag and the
committed Cargo manifests; Cargo dependencies still pin the full immutable
Nixe fork commit. A stable upstream release is the fork base, not a claim that
the Nixe-specific ABI exists upstream or that the fork has passed its proofs.

Local inspection of the pinned release confirms `Context::take_compiled_code`
in `cranelift/codegen/src/context.rs`, the `enable_pinned_reg` and
`regalloc_algorithm` settings, and `clear_cache`/`pipeline_flush_mt` in
`crates/jit-icache-coherence/src/lib.rs`. The latter package is named
`wasmtime-internal-jit-icache-coherence` and uses workspace version `48.0.1`.
These checks establish source/API provenance only. Every Nixe hook, allocator
constraint, owned-output conversion and numerical bound still requires the
[Task 0](tasks/00-baseline.md) and [Task 1](tasks/01-abi.md) proof on this exact release. Evidence collected from a
development checkout does not close either gate.
