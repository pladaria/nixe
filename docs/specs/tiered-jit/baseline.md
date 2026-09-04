# Supported baseline and generated manifest

[Specification index](README.md) · [Implementation tasks](tasks/README.md) · [Open review items](review-status.md)

Related contracts: [Cranelift fork and backend contract](backend.md); [Policy defaults, layouts and safety bounds](policy-and-capacity.md).

## Supported baseline

The required native hosts are `x86_64-unknown-linux-gnu` using SysV and
`aarch64-unknown-linux-gnu` using AAPCS64, in both cases on a Linux kernel with
4096-byte host pages. A different page size, OS or ABI becomes supported
only after this document is extended with its gateway registers, executable
memory policy, fault transport, instruction-cache synchronization and native
conformance run; it is not accepted through a generic fallback.

[Task 0](tasks/00-baseline.md) creates the Git-tracked
`docs/specs/tiered-jit/tiered-jit-baseline.toml`. Its normative TOML schema is:

```text
format_version: integer = 1
rust_version: string
nixe_migration_commit: 40-lowercase-hex string
wasmtime_upstream_release: string = "v48.0.1"
wasmtime_upstream_commit: 40-lowercase-hex string = "7bac2c2775808aaec5d4aa5627a5e447b51102cf"
cranelift_workspace_version: string = "0.135.1"
fork_url: canonical HTTPS git URL without credentials or query
fork_commit: 40-lowercase-hex string
fork_api_version: integer = 1
fork_api_layout_sha256: 64-lowercase-hex string
native_frame_abi_version: integer
worst_jit_layout_report_sha256: 64-lowercase-hex string

[[migration_registry_package]]
name: string; version: string; cargo_checksum_sha256: 64-lowercase-hex string

[[target_git_package]]
name: string; version: string; source_url: canonical URL; rev: 40-lowercase-hex string

[[host_feature]]
host_isa: integer; bit: integer in 0..127; name: string

[[cranelift_setting]]
profile: "LcqProduction" | "HcqProduction" | "Proof"
host_isa: integer; namespace: "shared" | "x86" | "arm64"
name: string; value: canonical Cranelift setting string

[[instruction]]
coverage_id: integer in 0..2^32-1; variant_path: string
status: "accepted" | "unsupported"
lowering: "native" | "helper" | "boundary"                 # accepted only
semantic_symbol: string                                      # accepted only
arm_doc_url: canonical HTTPS URL on an official Arm domain   # accepted only
helper_id: integer in 1..2^32-1                              # helper only
may_access_memory: boolean                                    # accepted only
memory_plan_id: integer in 1..2^32-1                         # iff may_access_memory
boundary_id: integer in 1..2^32-1                            # boundary only

[[helper_catalog]]
numeric_id: integer in 1..2^32-1; id: string
semantic_symbol: string; class: "Leaf" | "Canonical"
arguments: [ABI type token]; returns: [ABI type token]
sysv_arg_locs: [ABI location token]; sysv_return_locs: [ABI location token]
aapcs64_arg_locs: [ABI location token]; aapcs64_return_locs: [ABI location token]
read_set: [string]; write_set: [string]
sysv_clobbers: [string]; aapcs64_clobbers: [string]
nzcv_effect: "preserve" | "clobber" | "produce"
fp_effect: "preserve" | "host" | "guest"
outcome_rules: [OutcomeRule token]
sysv_outgoing_max: integer; sysv_callee_low_water_max: integer
aapcs64_outgoing_max: integer; aapcs64_callee_low_water_max: integer

[[boundary_catalog]]
numeric_id: integer in 1..2^32-1; id: string; exit_kind: string
read_set: [string]; write_set: [string]; outcome_rules: [OutcomeRule token]

[[memory_plan_catalog]]
numeric_id: integer in 1..2^32-1; id: string; semantic_symbol: string
class: "SingleRestartable" | "PrefixVisible" | "AllOrNothing"
subaccess_count: integer; commit_stage_count: integer
has_base_writeback: boolean; effect_table_hash_sha256: 64-lowercase-hex string

[[host_bound]]
host_isa: integer; lowering_template_max: integer; lcq_fixed_max: integer
island_template_bytes: integer; reserved_registers: [string]
native_frame_layout_sha256: 64-lowercase-hex string
helper_outgoing_max: integer; helper_callee_low_water_max: integer
bridge_parallel_copy_max: integer; static_bridge_bytes_max: integer
dynamic_bridge_bytes_max: integer
linux_signal_abi_layout_sha256: 64-lowercase-hex string
linux_uapi_git_commit: 40-lowercase-hex string
linux_uapi_headers_sha256: 64-lowercase-hex string
signal_alt_stack_min_bytes: integer = 65536
signal_alt_stack_slack_bytes: integer = 32768
signal_alt_stack_max_bytes: integer = 262144
signal_landing_stack_bytes: integer = 65536; signal_guard_bytes: integer = 4096
fixed_charge_at_128_vcpus_worst_case: integer
largest_hcq_reservation: integer

[[registry_capacity]]
name: "address_space" | "unit" | "family" | "patch" | "build_fingerprint"
slot_count: integer; slot_bytes: integer; slots_per_4096_page: integer
allocation_bitmap_bytes: integer

[[bounded_index]]
name: "dispatch" | "owner" | "dynamic_bridge_weak"
kind: "robin_hood" | "set_associative"
bucket_count: integer; live_limit: integer; bucket_bytes: integer
set_count: integer; ways: integer

[[layout_row]]
layout_row_id: integer in 1..2^32-1
name: string; type_name: string; count_formula: string
element_size: integer; element_alignment: integer
type_layout_sha256: 64-lowercase-hex string
worst_case_count: integer
worst_case_rounded_committed_bytes: integer
worst_case_reserved_bytes: integer
charge_class: "fixed_committed" | "lazy_committed" | "reserved_credit" |
              "control_committed_excluded" | "virtual_only" |
              "logical_capacity_only"
backing_row: string

[[metadata_slab]]
type_name: string; type_id: integer in 1..2^32-1
slab_index: integer in 0..255
slot_size: integer; slot_alignment: integer; slots_per_page: integer
storage_kind: "fixed_slot" | "paged_payload" | "permanent_extent" |
              "guarded_stack_extent" | "indexed_signal_payload"
max_object_pages: integer in 1..260086

[[enum_discriminant]]
domain: string; variant: string; value: integer in 0..2^32-1

[[type_domain]]
domain: string; tag: integer in 1..2^32-1

[[golden_vector]]
domain: string; canonical_input_hex: even-length lowercase hex string
stable_encode_hex: even-length lowercase hex string; fnv1a64: 16-lowercase-hex string
```

`HelperId`, `BoundaryId`, `MemoryEffectPlanId` and `CpuProfileId` are distinct
`repr(transparent)` NonZeroU32 newtypes. Every catalog row carries its explicit
`numeric_id`; IDs are unique within their catalog and are never inferred from
TOML order, a Rust enum discriminant or a hash. The three catalogs are encoded
and emitted in increasing numeric ID, with `id` unique canonical ASCII names.
Every instruction reference resolves to exactly one row whose semantic symbol
matches; missing, duplicate or unused production rows fail [Task 0](tasks/00-baseline.md). CpuProfileId
is fixed to Switch1 = 1 and Switch2 = 2; no other value is accepted in this
version. The baseline records these assignments as enum-discriminant rows and
all four IDs use their numeric u32 value in StableEncode and fork requests.

ABI type tokens are exactly `void`, `u8`, `u16`, `u32`, `u64`, `i8`, `i16`,
`i32`, `i64`, `f16`, `f32`, `f64`, `v64`, `v128`, `guest_addr64`,
`const_host_ptr64` or `mut_host_ptr64`; `void` is legal only as the sole return.
ABI location tokens are `gpr:<architectural-name>`,
`vec:<architectural-name>`, `stack:<decimal-byte-offset>:<decimal-byte-width>`
or `indirect:<architectural-name>`. Argument/location and return/location arrays
have equal lengths after removing a sole `void`; stack ranges may not overlap
and are relative to the stable helper SP. A Leaf helper's return list is
exactly `[void]` or one non-void direct scalar/vector token which fits the one
platform return location. It may not use `indirect`, sret, an aggregate,
variadic arguments or more than one return; additional outputs use declared
`mut_host_ptr64` arguments and appear in the write set. Canonical helpers return
only their closed `HelperOutcome` through the canonical Rust boundary, not an
ad-hoc machine signature. Outcome-PC rules are exactly
`current`, `next`, `target_argument:<decimal-index>` or
`fixed:<16-lowercase-hex-PC>`. Catalog parsing, not free-form call-site code,
maps these tokens to typed enums.

Read/write-set tokens are exactly `x0` through `x30`, `sp`, `v0` through
`v31`, `nzcv`, `fpcr`, `fpsr`, `pc`, `exclusive_monitor` or `memory`.
SysV clobber tokens are lowercase x86-64 register names from `rax..r15`,
`xmm0..xmm15`, `rflags`, `mxcsr` and `x87`. AAPCS64 clobber tokens are only
`x0..x18`, `x30`, `v0..v7`, `v8.hi64..v15.hi64`, `v16..v31`, `nzcv`, `fpcr`
and `fpsr`; `vN` means all 128 bits while `vN.hi64` means bits 64..127 and
models AAPCS64's partial preservation exactly. The parser checks each against
the named host, rejects duplicates/overlapping full-and-partial masks and
rejects a descriptor which exposes a JIT-reserved register as an argument/
result location or omits an ABI-permitted clobber not disproved by the checked
helper machine-code verifier. An OutcomeRule token is exactly
`ContinueAt@<pc-rule>`, `GuestException@<pc-rule>`,
`SchedulerExit@current` or `InternalFailure@current`, where `<pc-rule>` uses the
grammar above; one token represents the paired outcome and PC rule, so parallel
arrays cannot disagree. Boundary `exit_kind` is one of `CompileMiss`,
`Unavailable`, `ResourceCut`, `BudgetExhausted`, `GuestException`,
`SchedulerExit` or `InternalFailure` and is assigned a manifest discriminant.

The top-level tables occur in the order shown. Their repeated rows are sorted
exactly as follows: migration_registry_package `(name,version)`;
target_git_package `(name,version)`; host_feature `(host_isa,bit)`;
cranelift_setting `(profile,host_isa,namespace,name)`; instruction
`coverage_id`; helper_catalog, boundary_catalog and memory_plan_catalog each by
numeric_id; host_bound by host_isa; registry_capacity, bounded_index and
layout_row each by name; metadata_slab by numeric type_id; enum_discriminant by
`(domain,value)`; type_domain by `(tag,domain)`; and golden_vector by
`(domain,canonical_input_hex)`. Arrays representing sets are sorted by their
StableEncode byte sequence and contain no duplicate. All strings are valid
UTF-8 and use the exact spelling emitted by the generator; the generator emits
LF line endings, decimal integers, lowercase hex, no inline tables and one
trailing newline. The parser rejects unknown, missing or conditionally illegal
fields. Catalog definition IDs and semantic-definition symbols are unique;
multiple instructions may intentionally reference the same catalog entry.
Every helper, memory-plan and boundary reference resolves exactly once. The
enum-discriminant and type-domain tables contain every enum/type consumed by
StableEncode, with no duplicate value within a domain and no duplicate nonzero
type tag globally.

The manifest records:

- `format_version = 1`, Rust `1.97.1`, and migration source commit
  `33992b03e625ed7843d954426b4513c90b601f60`;
- the crates.io migration-source Cranelift package versions and Cargo checksums,
  separately from the target git packages, for which it records the
  project-owned fork URL, exactly
  `https://github.com/pladaria/wasmtime.git`, and immutable commit used by
  Cargo, upstream Wasmtime stable release `v48.0.1`, base
  `7bac2c2775808aaec5d4aa5627a5e447b51102cf`, and the
  version/source/revision of every resolved `cranelift-*` package; git packages
  do not have or invent a Cargo registry checksum; the fork commit must be a
  descendant of that upstream base and its checked API-layout digest must equal
  the generated public types in the [fork contract](backend.md);
- every decoder `CoverageId` for which the production JIT accepts
  `decode::a64::A64Instruction`, including its nested family variant and whether
  it lowers natively, invokes one named typed helper, or terminates at one named
  architectural boundary; and
- every recognized-but-unsupported `CoverageId`, which must continue to stop
  precisely and is not part of the accepted baseline.

An accepted instruction has exactly one lowering, semantic symbol and official
Arm documentation URL whose scheme/host are exactly
`https://developer.arm.com`. One family URL may be repeated by multiple CoverageIds,
but the shared semantics/lowering source must carry that URL in a nearby source
comment as required by `CONTRIBUTING.md`; CI rejects a missing or non-Arm
reference. A helper
lowering also has exactly one `helper_id`; any accepted instruction whose
shared semantics can touch memory has `may_access_memory = true` and exactly one
`memory_plan_id`, while every other accepted instruction has false and no plan;
a boundary lowering has exactly one `boundary_id`; every
other conditional field is absent. Unsupported entries have none of them. The
catalog tables are normative, data-only declarations in [Task 0](tasks/00-baseline.md): [Task 1](tasks/01-abi.md)
materializes and consumes `HelperDescriptor`, while [Task 3](tasks/03-lcq-cutover.md) materializes and
consumes `MemoryEffectPlan`. This ordering does not require either runtime type
to exist in [Task 0](tasks/00-baseline.md). The effect-table hash covers the canonical declarative
subaccess/commit-stage table from the shared semantic source, not Rust layout or
a function pointer.

The manifest also contains the complete three-profile Cranelift settings map
and fixed registry policy, and, per host ISA, the audited maximum bytes of every
lowering template, maximum LCQ fixed overhead, island template bytes,
reserved-register set, NativeFrame and generated Linux signal-ABI layout
hashes, signal-stack formula constants, helper stack/outgoing-argument
maxima, worst legal parallel-copy program, final StaticBridgeUnit and dynamic
BridgeUnit byte maxima, and the worst-case fixed-charge/worst-HCQ feasibility
bounds.
Both final bridge maxima must be at most 2048 bytes and include landing pads,
copies, transfer-slot accesses and the final transfer instruction. Unknown keys,
duplicate definitions, unsorted entries and a manifest/production-table
mismatch are build-test failures. A generator in `tools/xtask` is the only
writer; CI runs it in `--check` mode and requires a byte-identical file.

ABI/layout hashes are never hashes of debug text, Rust type names alone or raw
memory containing padding. Their canonical inputs are:

```text
AbiFieldLayoutV1 {
  name:utf8, declaration_ordinal:u32, type_token:utf8,
  byte_offset:u32, byte_size:u32, byte_alignment:u32
}
AbiTypeLayoutV1 {
  host_isa:u8, name:utf8, byte_size:u32, byte_alignment:u32,
  fields:[AbiFieldLayoutV1] in declaration order
}
NativeFrameLayoutV1 {
  abi_version:u32, host_isa:u8, stack_alignment:u32,
  frame_bytes:u32, stable_helper_sp_offset:u32,
  types:[AbiTypeLayoutV1] UTF-8-name sorted,
  named_ranges:[{name,start,end,alignment}] UTF-8-name sorted
}
LinuxSignalAbiLayoutV1 {
  host_isa:u8, target_triple:utf8,
  linux_uapi_git_commit:[u8;20], linux_uapi_headers_sha256:[u8;32],
  siginfo:AbiTypeLayoutV1, ucontext:AbiTypeLayoutV1,
  mcontext:AbiTypeLayoutV1, fp_state:AbiTypeLayoutV1,
  signal_numbers:[{name,value}] UTF-8-name sorted,
  si_code_values:[{name,value:i32}] UTF-8-name sorted,
  greg_or_field_mapping:[{architectural_name,type_name,field_name,index}]
      architectural-name sorted
}
```

`type_layout_sha256` is
`SHA256("nixe-type-layout-v1\0" || StableEncode(AbiTypeLayoutV1))`.
`native_frame_layout_sha256` is
`SHA256("nixe-native-frame-layout-v1\0" ||
StableEncode(NativeFrameLayoutV1))`; `linux_signal_abi_layout_sha256` uses the
distinct domain `"nixe-linux-signal-abi-layout-v1\0"` and its complete object.
The pinned UAPI fields in each host_bound identify the exact vendored Linux
headers parsed by the generator; production offset assertions against libc
types must equal that object. `worst_jit_layout_report_sha256` uses domain
`"nixe-jit-layout-report-v1\0"` over the complete worst-case
JitLayoutReportV1. All names/type tokens have a [Task 0](tasks/00-baseline.md) closed vocabulary,
padding ranges are explicitly named `reserved_zero_N`, and the checker zeros
and verifies them. A compensating size/offset change therefore cannot preserve
any digest accidentally.

`A64Instruction` is the normalized instruction type meant by "normalized A64"
throughout this document; no separate `NormalizedA64` type is introduced. The
manifest is generated by walking the production decoder table and is checked
against it in a test, so hand omission is impossible. Both LCQ and HCQ consume
the same accepted entries and the same JIT semantic lowering. The interpreter
may remain an independently implemented differential oracle; architectural
invariant 1 forbids a second semantics implementation between the two JIT
tiers, not an independent interpreter.

The baseline becomes immutable when [Task 0](tasks/00-baseline.md) closes. Later coverage additions
update the manifest only when LCQ and HCQ pass the same semantic, helper and
fault tests on both native hosts. An implementer cannot narrow supported
instructions, relabel a supported operation as unsupported, or use an
interpreter fallback to make an acceptance run pass.
