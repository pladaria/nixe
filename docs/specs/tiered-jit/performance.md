# External performance acceptance

[Specification index](README.md) · [Implementation tasks](tasks/README.md) · [Open review items](review-status.md)

Related contracts: [Supported baseline and generated manifest](baseline.md); [Final conformance gate](conformance.md).

## External performance acceptance

This is a release-acceptance protocol, not product instrumentation. It uses
release builds, external presentation capture, Linux resource accounting and
offline analysis. Production Nixe gains no benchmark mode, timing histogram,
overlap statistic or code-shape counter. The existing private allocation ledger
and the general maintenance control operation named below are the only runtime
observations. Commercial content is never copied into the repository or result
bundle.

## Canonical inputs and identities

`docs/specs/tiered-jit/tiered-jit-protocol.toml` is the sole acceptance input. It is UTF-8
without BOM, NUL or CR bytes, ends in exactly one LF, contains no comments,
date/time or floating-point values, and is parsed with duplicate and unknown
keys rejected. Its exact bytes, not a reserialization, are frozen.

The following lexical types are closed:

- `AsciiId` matches `[a-z][a-z0-9_-]{0,62}`.
- `Hex32` is exactly 64 lowercase hexadecimal digits.
- `UDec` is zero or a nonzero ASCII digit followed by ASCII digits, with no
  leading zero.
- `Utf8NoNul` is a valid UTF-8 scalar sequence without NUL; its exact bytes are
  significant and no normalization is performed.
- `GitOid` has exactly 40 lowercase hexadecimal digits for `sha1` and 64 for
  `sha256`.
- `RelPath` is nonempty printable ASCII, uses `/`, and has no leading `/`,
  backslash, empty component, `.` component or `..` component.
- `EnvName` matches `[A-Z_][A-Z0-9_]*`.

`ProtocolId = lowercase_hex(SHA-256("nixe-tiered-jit-protocol-v1\0" ||
protocol_file_bytes))`. Generated records use `CanonicalJsonV1`: RFC 8785 JCS
bytes followed by exactly one LF as a file; the LF is excluded where a formula
uses `CJ(x)`. Schemas reject duplicate/unknown keys, JSON numbers, `null` and
non-finite values. Every integer is a JSON string containing UDec; a signed
integer is `{negative:bool,magnitude:UDec}`; a rational is
`{num:UDec,den:UDec}` with `den>0` and `gcd(num,den)=1`. Maps are represented
only by arrays sorted by their declared key. JSONL is one JCS object followed
by one LF per record.

Every filesystem reference is
`PathSpec { base, rel:RelPath }`. `base` is exactly `REPOSITORY`, `BUNDLE`,
`RUN_DIR`, `CONTENT_ROOT`, `CACHE_ROOT`, `SNAPSHOT_ROOT`, `CGROUP_ROOT` or
`HOST_ROOT`. `HOST_ROOT` resolves to `/`. A HostSpec supplies HOST_ROOT-relative
RelPaths for the four content/cache/snapshot/cgroup roots; the other bases are
the canonical repository, bundle and current-member directories. Resolution
joins components without symlink traversal and then verifies the opened file by
`openat2` with `RESOLVE_BENEATH|RESOLVE_NO_MAGICLINKS`; HOST_ROOT additionally
uses `RESOLVE_IN_ROOT`. A resolved path may not escape its base. Commands never
receive an unresolved token.

The TOML maps bijectively to this closed schema; array order and uniqueness
requirements shown here are validation rules:

```text
ProtocolV1 {
  format_version: 1,
  bootstrap_seed: Hex32,
  bootstrap_replicates: 100000,
  max_qualification_attempts: u8 in 1..=255,
  max_host_attempts: u8 in 1..=255,
  candidate_build: CandidateBuildSpec,
  reference_builds: [ReferenceBuildSpec] sorted unique host_id,
  tools: [ToolSpec] sorted unique (host_id, tool_id),
  frozen_files: [FrozenFile] sorted unique file_id,
  required_diagnostics: [DiagnosticSpec] sorted unique (host_id, diagnostic_id),
  hosts: [HostSpec; 2] sorted unique host_id,
  workloads: [WorkloadSpec] sorted unique workload_id,
  thresholds: ThresholdSpec
}
FrozenFile {
  file_id: AsciiId, source: PathSpec, sha256: Hex32
}
StdIoPolicy = Empty | Retained | Sha256(Hex32)
ToolSpec {
  host_id: AsciiId, tool_id: AsciiId, executable: PathSpec,
  sha256: Hex32, version_argv: [Utf8NoNul],
  version_stdout: StdIoPolicy, version_stderr: StdIoPolicy
}
PathArg = Literal(Utf8NoNul) | Path(PathSpec)
EnvBinding { name: EnvName, value: Utf8NoNul }
CommandSpec {
  command_id: AsciiId, tool_id: AsciiId, argv: [PathArg],
  cwd: PathSpec, env: [EnvBinding] sorted unique name,
  timeout_ns: u64 > 0,
  stdout_policy: StdIoPolicy, stderr_policy: StdIoPolicy
}
ProbeSpec {
  probe_id: AsciiId, command: CommandSpec,
  stdout_policy: StdIoPolicy, stderr_policy: StdIoPolicy
}
CandidateBuildSpec {
  git_object_format: "sha1" | "sha256",
  expected_remote: Utf8NoNul,
  toolchain_file: FrozenFileId,
  cargo_lock_file: FrozenFileId,
  target_builds: [TargetBuildSpec; 2] sorted unique host_id
}
BuildOutputSpec {
  role: "emulator" | "runner" | "analyzer", relpath: RelPath
}
TargetBuildSpec {
  host_id: AsciiId, target_triple: Utf8NoNul,
  profile: Utf8NoNul, features: [AsciiId] byte-sorted unique,
  build: CommandSpec,
  outputs: [BuildOutputSpec; 3] sorted unique role
}
ReferenceBuildSpec {
  host_id: AsciiId, git_object_format: "sha1" | "sha256",
  git_commit: GitOid, binary: PathSpec, binary_sha256: Hex32,
  toolchain_file: FrozenFileId, cargo_lock_file: FrozenFileId,
  profile: Utf8NoNul, features: [AsciiId] byte-sorted unique
}
RootBindings {
  content_root: RelPath, cache_root: RelPath,
  snapshot_root: RelPath, cgroup_root: RelPath
}
HostSpec {
  host_id: AsciiId, target_triple: Utf8NoNul,
  machine_id: Utf8NoNul, roots: RootBindings, page_bytes: 4096,
  selected_physical_cores: [u16] nonempty sorted unique,
  excluded_smt_siblings: [u16] sorted unique,
  probes: [ProbeSpec] sorted unique probe_id,
  thermal: ThermalSpec, cgroup_parent: RelPath,
  timer_wakeup_lateness_ns: u64,
  pss_period_ns: 50000000,
  pss_start_lateness_ns: u64 < pss_period_ns,
  pss_snapshot_timeout_ns: u64 <= pss_period_ns,
  pss_snapshot_max_tries: 3
}
ThermalSpec {
  sensor: PathSpec, target_millicelsius: i64,
  tolerance_millicelsius: 2000, poll_period_ns: u64 > 0,
  stable_dwell_ns: u64 >= poll_period_ns,
  timeout_ns: u64 >= stable_dwell_ns
}
InvocationSpec {
  argv: [PathArg], cwd: PathSpec,
  env: [EnvBinding] sorted unique name
}
NamedHash { name: AsciiId, sha256: Hex32 }
ResetSpec {
  host_id: AsciiId, emulator: "candidate" | "reference",
  translated_cache_snapshot_sha256: Hex32,
  shader_snapshot_sha256: Hex32,
  pipeline_snapshot_sha256: Hex32,
  filesystem_policy: "primed" | "dropped",
  reset: CommandSpec, restore: CommandSpec, verify: CommandSpec
}
InvocationByHost {
  host_id: AsciiId, emulator: "candidate" | "reference",
  invocation: InvocationSpec
}
ColdSpec { first_present_timeout_ns: u64 > 0 }
SustainedSpec {
  duration_ns: u64 >= 60000000000,
  min_accepted_timestamps: u64 >= 2,
  hang_timeout_ns: u64 > duration_ns,
  maintenance_timeout_ns: u64 > 0
}
QualificationSpec {
  duration_ns: u64 > 0, min_accepted_timestamps: u64 >= 2,
  min_rate_millihz: u64 > 0, hang_timeout_ns: u64 >= duration_ns
}
WorkloadSpec {
  workload_id: AsciiId, class: "commercial" | "smoke",
  roles: ["startup_heavy" | "long_running"] byte-sorted unique,
  content_hashes: [NamedHash] sorted unique name,
  firmware_sha256: Hex32, title_sha256: Hex32,
  update_sha256: Hex32, dlc_hashes: [NamedHash] sorted unique name,
  settings_file: FrozenFileId, correctness_marker_sha256: Hex32,
  capture_tool_id: AsciiId, detector_file: FrozenFileId,
  pre_guest_ready_file: FrozenFileId, setup_warmup_file: FrozenFileId,
  measurement_marker_file: FrozenFileId,
  resets: [ResetSpec] sorted unique (host_id, emulator),
  invocations: [InvocationByHost] sorted unique (host_id, emulator),
  cold: ColdSpec, sustained: SustainedSpec,
  qualification: QualificationSpec
}
DiagnosticSpec {
  host_id: AsciiId, diagnostic_id: AsciiId,
  command: CommandSpec, required_artifacts: [RelPath] sorted unique
}
ThresholdSpec {
  aggregate_sustained_cpu: Rational = 1/1,
  per_workload_sustained_cpu: Rational = 23/20,
  per_workload_p95_interval: Rational = 6/5,
  aggregate_cold: Rational = 11/10,
  per_workload_cold: Rational = 5/4,
  aggregate_p99_interval: Rational = 23/20,
  per_workload_peak_pss_delta: Rational = 5/4
}
```

`FrozenFileId` is an AsciiId which must resolve exactly once. `freeze-protocol`
copies each FrozenFile to `protocol/files/<file_id>`, verifies its hash before
and after the copy and makes later resolution use only that copy. InvocationSpec
deliberately has no binary: the candidate executable comes from
CandidateManifest and the reference executable from ReferenceBuildSpec. The
runner composes exactly one of those verified binaries with InvocationSpec and
rejects any attempted binary/path/hash override. It invokes tools by the
host-matching ToolSpec absolute file descriptor, starts from an empty
environment, adds only declared EnvBindings and uses direct `execveat`; shell
evaluation, PATH lookup, inherited locale/environment and interpolation are
forbidden. Retained stdout/stderr bytes become artifacts; Empty requires zero
bytes and Sha256 requires the declared digest.

The required `probe_id` set is exactly `cpu_topology`, `memory`, `firmware`,
`kernel`, `microcode`, `governor`, `frequency_limits`, `power_limits`,
`throttle_counters`, `driver`, `desktop`, `presentation`, `cgroup_v2`,
`page_size` and `smt_state`. A successful probe has exit status zero and matches
both output policies. The checker additionally requires the two target triples
`x86_64-unknown-linux-gnu` and `aarch64-unknown-linux-gnu`, at least three
commercial workloads, at least one commercial workload with each required role,
all IDs resolved, exactly one reset/invocation for each
`(workload,host,emulator)`, and no duplicate ID.

`CandidateCoreV1` is:

```text
{
  format_version:"1", protocol_id:Hex32,
  git_object_format:"sha1"|"sha256", git_commit:GitOid,
  binaries:[{
    host_id:AsciiId, role:"emulator"|"runner"|"analyzer",
    path:RelPath, sha256:Hex32, size_bytes:UDec
  }] sorted unique (host_id,role)
}
```

The repository must have the protocol-declared remote/commit and empty
`git status --porcelain=v2 --untracked-files=all`. All six binaries must be
outputs of the declared builds. `CandidateId =
SHA-256("nixe-tiered-jit-candidate-v1\0" || CJ(CandidateCoreV1))`.
`CandidateManifestV1={core:CandidateCoreV1,candidate_id:Hex32}` and every
consumer recomputes it.

`AttemptId = SHA-256("nixe-tiered-jit-attempt-v1\0" ||
CJ({protocol_id,candidate_id,host_id,ordinal:UDec}))`. `QualificationId` uses
the identical object with domain
`"nixe-tiered-jit-qualification-v1\0"` and its own contiguous ordinal. IDs are
lowercase hex. No identity includes a wall-clock time, result or artifact hash.

## Qualification and attempt order

The only legal workflow order is:

1. validate every frozen file/tool/reference hash and create the immutable
   ProtocolId;
2. build both native candidate targets from one clean commit and create
   CandidateManifest;
3. pass `check-host` on both hosts;
4. complete one authoritative passing qualification on each host;
5. only after both qualifications pass, start acceptance attempts;
6. analyze each host independently;
7. create the release record; and
8. create checksums.

Qualification never precedes protocol freeze. No acceptance observation may
exist before both qualifications pass. A protocol or candidate change creates
new identities and cannot reuse any host check, qualification or attempt.

Qualification executes every workload, including smoke workloads, in ascending
UTF-8 workload-id bytes, candidate then reference. Each member uses the same
reset, fresh-cgroup, affinity, thermal and host postcheck procedure as an
acceptance member. If accepted timestamps are `t[0..m-1]`, it requires
`m >= min_accepted_timestamps`, `elapsed_ns=t[m-1]-t[0] > 0` and exactly
`(m-1)*1_000_000_000_000 >= min_rate_millihz*elapsed_ns`, plus the frozen
correctness marker and scene. Qualification observations never enter
acceptance/bootstrap analysis. Only InfraInvalid may advance its ordinal up to
`max_qualification_attempts`; the first Pass is authoritative. A correctness or
numeric qualification failure is final for this ProtocolId/CandidateId.
Qualification cannot run after any acceptance-attempt directory exists.

For one host, an acceptance attempt has exactly this total order: phase cold,
then phase sustained; within each phase, commercial workloads in ascending
UTF-8 workload-id bytes; within each workload, pair indices 0..9; within an even
pair, candidate then reference, and within an odd pair, reference then
candidate. Smoke workloads never enter acceptance ratios. Cross-host
wall-clock interleaving is unconstrained and has no semantic effect. No two
members run concurrently on one host. Both members of pair i use
`selected_physical_cores[i % len]`; the process and descendants inherit that
affinity and no excluded SMT sibling may execute them.

Before a member, the runner creates a fresh empty leaf below the frozen cgroup
parent and launches the process into it from birth using
`clone3(CLONE_INTO_CGROUP)` followed by direct exec; post-exec migration,
sub-cgroups, an extra process or a member outside it is InfraInvalid. It then
runs reset/restore/verify, host probes and a throttle snapshot and reaches
thermal stability. Thermal polls occur at absolute
`first_poll + k*poll_period_ns`. The target is the frozen
`target_millicelsius`, never the first sample. The first in-band sample begins a
dwell; an out-of-band value or read failure resets it. Stability requires every
scheduled poll in band until monotonic dwell is at least `stable_dwell_ns`.
Timeout begins at the first poll. Probes/throttle are repeated after the
member; any change or failure invalidates the complete host attempt.

For cold, the launcher reads `CLOCK_MONOTONIC` immediately before `execveat`;
if exec succeeds that timestamp is C0. Cold time is the first accepted frozen-
detector presentation timestamp minus C0. Failure to reach it before the exact
timeout is classified by the outcome rules below. Cold produces no CPU or PSS
acceptance metric.

For sustained, the emulator reaches and remains blocked at the frozen
pre-guest-ready barrier. While held, take coherent `ready_pss` and immediately
after it a Nixe ledger sample when the member is candidate. Release that
barrier, execute setup/warm-up and wait for the measurement marker. Immediately
before releasing the marker gate, read `cpu.stat usage_usec` as U0, set
`S=clock_gettime(CLOCK_MONOTONIC)`, arm a separate absolute CLOCK_MONOTONIC
timer for `E=S+duration_ns` and release the gate. At its expiration, the
runner's first action is U1; wake after `E+timer_wakeup_lateness_ns` is
InfraInvalid. Accepted presentation timestamps are exactly those in closed
`[S,E]`. Let their count be m, require
`m>=min_accepted_timestamps>=2` and form all m-1 adjacent intervals. Checked
CPU ns is `(U1-U0)*1000` and CPU per presentation is that value divided by m.
No event or interval is removed.

PSS uses a dedicated
`timerfd_create(CLOCK_MONOTONIC,TFD_CLOEXEC|TFD_NONBLOCK)` and
`timerfd_settime(...,TFD_TIMER_ABSTIME,...)`. After the ready sample, targets
are `ready_time+k*pss_period_ns`; only targets strictly before E are scheduled,
and one final sample is mandatory immediately after U1. An expiration count
other than one, sample start later than
`target+pss_start_lateness_ns`, inability to finish before the next target or a
final sample exceeding `pss_snapshot_timeout_ns` is InfraInvalid.

A coherent PSS snapshot is retried from scratch at most
`pss_snapshot_max_tries` and must finish within `pss_snapshot_timeout_ns`:
read/numerically sort `cgroup.procs` as M1; for every PID read
`/proc/PID/stat` starttime A, `smaps_rollup` Pss and stat starttime B; then read
sorted membership M2. Accept only if M1=M2, every A=B and every read succeeds.
Sum checked `Pss_kB*1024` over M1; Linux proc kB is exactly 1024 bytes. Record
membership, starttimes, per-PID values/timestamps, retry count and sum.
`peak_pss_delta=max(0,max(all accepted samples including ready and final)-
ready_pss)`. The reference delta must be positive. Immediately after every
accepted candidate PSS sample, including ready/final, read all five ledger words
with the even-sequence protocol and require `jit_charged_bytes<=640 MiB`.

The workload remains alive after E. After final PSS/ledger sampling, issue the
production-general `ReclaimToSoftWatermark` request through the frozen
controller. Its acknowledgement is exactly
`{request_id,requested_after_attempt_id,observed_cutover_sequence,
observed_epoch,reclaimable_count,jit_charged_bytes}`. Before timeout the
matching acknowledgement must have no reclaimable cutover/epoch owner and
`jit_charged_bytes<=512 MiB`; only then may the runner terminate the workload.
This operation belongs to the runtime maintenance API, is not benchmark-only
telemetry and never runs in the timed window.

For sorted intervals `d[0..n-1]`, nearest-rank
`q(p)=d[ceil(p*n)-1]`; n is at least one. Pair ratios are candidate/reference
from the same phase/workload/index. CPU ratio is
`(candidate_cpu_ns*reference_m)/(reference_cpu_ns*candidate_m)`; other ratios
use metric integers directly. Every quantity is a positive integer except
candidate PSS delta may be zero. A zero/missing/nonpositive reference
denominator is InfraInvalid. External profiler/disassembly runs use fresh
post-acceptance cgroups, never interleave with or substitute acceptance
members, and never change a performance result.

## Events, outcomes and retries

`QualificationManifestV1` and `AttemptManifestV1` contain exactly
`{format_version:"1",protocol_id,candidate_id,host_id,ordinal,
qualification_id|attempt_id,started_monotonic_ns,ended_monotonic_ns,outcome,
runs:[RunSummary],artifacts:[ArtifactRef]}`. Runs are in the required schedule;
artifacts are path-sorted. `ArtifactRef={path:RelPath,sha256:Hex32,
size_bytes:UDec,kind:AsciiId}` and never names its containing manifest.
`RunKey={phase:"qualification"|"cold"|"sustained",workload_id,
pair_index:UDec,emulator:"candidate"|"reference"}`; qualification uses
pair_index zero.

Every events.jsonl record is one closed `EventV1` with
`{format_version:"1",seq:UDec,monotonic_ns:UDec,run_key:RunKey,kind,payload}`.
Sequence starts at one and increments exactly once. Payload is exactly one of:

```text
command {
  command_id, stage:"reset"|"restore"|"verify"|"setup"|"postflight",
  start_ns, end_ns, wait_status, stdout:ArtifactRef, stderr:ArtifactRef
}
probe {
  probe_id, stage:"before"|"after", start_ns, end_ns, wait_status,
  stdout:ArtifactRef, stderr:ArtifactRef
}
thermal_sample {
  target_ns, read_ns, millicelsius:SignedDec, in_band:bool,
  dwell_start_ns:None|UDec
}
launch {
  pid:UDec, cgroup:RelPath, physical_core:UDec,
  c0_ns:None|UDec, exec_errno:None|UDec
}
barrier { barrier_id, action:"reached"|"release", success:bool }
marker { marker_id, observed_sha256:Hex32, accepted:bool }
presentation {
  presentation_seq:UDec, timestamp_ns:UDec,
  accepted:bool, detector_code:AsciiId
}
cpu_stat { label:"U0"|"U1", usage_usec:UDec, read_ns:UDec }
pss_snapshot {
  target_ns:None|UDec, start_ns, end_ns, try_count:UDec,
  members:[{pid,starttime,pss_bytes}] PID-sorted, total_pss_bytes:UDec
}
ledger {
  sample_ns, ledger_sequence, committed_executable_bytes,
  committed_metadata_bytes, reserved_credit_bytes, jit_charged_bytes
}
maintenance {
  request_id, request_ns, ack_ns, requested_after_attempt_id,
  observed_cutover_sequence, observed_epoch, reclaimable_count,
  jit_charged_bytes
}
exit { wait_status:UDec, exit_ns:UDec }
```

`None|UDec` is encoded as a closed tagged object, never JSON null. All listed
integer fields are UDec strings. Stdout/stderr ArtifactRefs refer to retained
raw bytes even when empty; policy validation uses those bytes. Wall-clock time
may exist in a separate diagnostics artifact but never in EventV1 or ordering.

Outcome is exactly `Complete`, `InfraInvalid{reason,run_key}`,
`CandidateFailure{reason,run_key}` or, for qualification only,
`QualificationFailure{side,reason,run_key}`. Infra reasons are exactly:
`host_precondition_changed`, `reset_failed`, `cache_restore_failed`,
`thermal_timeout`, `throttle_changed`, `cgroup_contaminated`,
`affinity_changed`, `capture_unavailable`, `timer_late`,
`timer_expiration_lost`, `pid_snapshot_unstable`, `pss_unreadable`,
`reference_exec_failed`, `reference_crash`, `reference_hang`,
`reference_wrong_marker`, `reference_missing_observation`,
`reference_nonpositive_denominator`, `artifact_io` and
`runner_interrupted`. Candidate reasons are exactly:
`candidate_exec_failed`, `candidate_crash`, `candidate_hang`,
`candidate_wrong_marker`, `candidate_missing_observation`,
`ledger_hard_limit`, `maintenance_timeout`,
`maintenance_not_quiescent` and `ledger_soft_limit`. QualificationFailure
uses the corresponding candidate/reference correctness or rate reason.

The runner performs every safe postflight after detecting failure and chooses
outcome precedence CandidateFailure, then InfraInvalid, then Complete.
QualificationFailure is selected before InfraInvalid when its underlying
candidate/reference observation is conclusive. Thus a candidate crash plus a
throttle change cannot be rerun. Complete requires every scheduled run/artifact.

Only InfraInvalid may be retried. The next ordinal is existing maximum plus one;
gaps, reuse or deletion are errors. Retry repeats the complete host schedule and
stops at the protocol maximum. The first Complete attempt is authoritative;
CandidateFailure is immediately final. After Complete, numeric failure is final
and no further run is legal. Every attempt remains in the bundle. A started
directory is `<ordinal>.partial`; interruption never resumes it.
`seal-interrupted` hashes retained raw bytes, writes
InfraInvalid/runner_interrupted, atomically renames to `<ordinal>` and only then
permits the next ordinal. No command supports subset, resume, overwrite or
force.

## Exact analysis

For each host derive
`host_seed=SHA-256("nixe-tiered-jit-bootstrap-host-v1\0" ||
raw_32_byte_bootstrap_seed || UTF8(host_id))` and construct a fresh
Cargo.lock-pinned `ChaCha8Rng::from_seed(host_seed)` at stream/word position
zero. For each replicate 0..99,999 and each commercial workload in ascending
UTF-8 bytes, draw ten sustained then ten cold indices. Each draw calls
`next_u32`, rejects values `>=4_294_967_290` and otherwise uses `value%10`.
Sustained draws are shared by CPU, p95, p99 and PSS for that workload.

Within each resample, sort ten exact rational ratios and take
`(r[4]+r[5])/2`. Aggregate statistics use the equal-workload product. Sort the
100,000 exact replicate values and use zero-based element 94,999 as the one-sided
pointwise 95% upper bound. Aggregate comparison is
`product <= threshold^W`; displayed roots/decimals are non-authoritative. Hosts
are analyzed separately and equality passes. All arithmetic uses reduced
arbitrary-precision rationals.

The seven gates are exactly the ThresholdSpec fields: aggregate sustained
CPU-per-presentation, each workload's sustained CPU-per-presentation, each
workload's p95 interval, aggregate cold first-present, each workload's cold
first-present, aggregate p99 interval and each workload's peak-PSS delta. No
host, workload, pair or metric is pooled, weighted, removed or replaced.

## Required CLI and exits

The package exposes exactly these workflow commands:

```text
tiered-jit-perf-analyze check-protocol --protocol P
tiered-jit-perf-analyze freeze-protocol --protocol P --bundle B
tiered-jit-perf-runner build-candidate --bundle B --host H --repository R
tiered-jit-perf-analyze make-candidate --bundle B
tiered-jit-perf-runner check-host --bundle B --host H
tiered-jit-perf-runner qualify --bundle B --host H --ordinal N
tiered-jit-perf-runner run-batch --bundle B --host H --ordinal N
tiered-jit-perf-runner seal-interrupted --bundle B --host H \
  --kind qualification|attempt --ordinal N
tiered-jit-perf-analyze analyze-host --bundle B --host H
tiered-jit-perf-analyze make-release-record --bundle B
tiered-jit-perf-analyze make-checksums --bundle B
tiered-jit-perf-analyze verify-bundle --bundle B
```

`check-protocol` is read-only and prints exactly `<ProtocolId><LF>`.
`freeze-protocol` requires nonexistent B and copies protocol/frozen files.
Every later command creates only its fixed missing output; an existing output
is an error. `make-candidate` requires both native build manifests, one clean
commit and all six hashes. `qualify` requires current checks for both hosts;
`run-batch` requires authoritative passing qualifications for both.
`make-release-record` requires both authoritative Complete attempts to pass
analysis and every DiagnosticSpec artifact to exist. `make-checksums` is the
last mutation; afterward every mutating command refuses.

Exit codes are exhaustive: 0 success/pass; 2 CLI, canonical schema, identity,
hash, artifact or illegal-order error; 3 host/precondition refusal before an
attempt starts; 4 InfraInvalid, including exhausted infra attempts; 5
QualificationFailure or CandidateFailure; 6 complete data with numeric
acceptance failure; and 70 internal invariant/bug. No other exit code is
emitted.

## Bundle and checksum closure

The fixed tree is:

```text
protocol/{tiered-jit-protocol.toml,protocol-id.txt,files/<file_id>}
candidate/builds/<host>/{emulator,runner,analyzer,build.json}
candidate/candidate.json
hosts/<host>/host-check.json
qualifications/<host>/<ordinal>/{manifest.json,events.jsonl,raw/...}
attempts/<host>/<ordinal>/{manifest.json,events.jsonl,raw/...}
analysis/<host>.json
diagnostics/<host>/...
release-record.json
SHA256SUMS
```

Commercial content is absent. `ReleaseRecordV1` is exactly
`{format_version:"1",protocol_id,candidate_id,hosts:[{host_id,qualification:
ArtifactRef,authoritative_attempt_id,attempt:ArtifactRef,analysis:ArtifactRef}]
host-sorted,diagnostics:[ArtifactRef] path-sorted,overall:"pass"}`. It contains
no checksum-file hash. Create it before SHA256SUMS.

SHA256SUMS is created last and contains every regular file below B except
itself exactly once, sorted by raw ASCII relative-path bytes, as
`<64 lowercase hex><two spaces><path><LF>`. It therefore includes
release-record.json and has no hash cycle. Verification rejects extra/missing
files, symlinks, hardlinks, devices, FIFOs, sockets, case-fold collisions and an
ArtifactRef outside B; it recomputes every hash, canonical record, identity,
ordinal, schedule, outcome, raw metric, bootstrap vector and threshold. No
write after checksums is legal. Any external signature/checksum anchor is
outside both the bundle and acceptance.

Before [Task 10](tasks/10-cross-target-conformance.md) can pass, `tools/tiered-jit-perf` must provide these schemas and
commands, golden canonical/identity/bootstrap vectors and synthetic fixtures
for every outcome and timer/PSS race. Missing either native host, any commercial
workload, qualification, authoritative attempt or required artifact blocks the
gate and never waives it.
