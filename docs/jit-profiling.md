# Native JIT cost attribution

Build with `cargo build --release -p nixe-cli --features jit-profile` and set
`NIXE_JITDUMP` to a local capture directory before starting the program. Normal
builds contain neither export storage nor execution-path instrumentation.

Record flat instruction-pointer samples with the monotonic clock:

```sh
perf record --clockid mono -e cycles:u -F 99 -p "$PID" -o perf.data
perf inject --jit -i perf.data -o perf-jit.data
perf report -i perf-jit.data --stdio --no-children
perf script -G --ns -i perf-jit.data -F comm,pid,tid,time,period,ip,sym,dso > samples.txt
```

The Linux [jitdump specification](https://github.com/torvalds/linux/blob/master/tools/perf/Documentation/jitdump-specification.txt)
is the format authority. The executable marker mapping lets `perf` discover
`jit-PID.dump`, including when attaching after compilation has warmed up.
LOAD records have unique indices. Unit names identify tier, guest process,
address space, unit, version and entry PC. Link bridges, subsequent link patches
and far islands have their own records. Files use `create_new`; captures from
another process or an earlier run are not overwritten.

`jit-PID.regions.csv` retains exact native intervals, captured guest PCs, load
timestamps and allocation reclamation timestamps. The format has no UNLOAD
record; reclamation belongs to this sidecar. A reused address receives a new
LOAD index. Link patch intervals supersede only their actual bytes, and their
allocation's reclamation ends their lifetime too.

The local development utility `dev-tools/performance/incremental/jit_attribution.py`
combines this sidecar with `samples.txt`. It reports flat cycle weights by thread,
tier, guest lowering, entry/exit adapters, dispatch/link code, named host runtime
categories and unresolved samples. Pass `--log game.log` to include final export
status. Guest source labels are virtual PC labels; the CSV supplies their exact
intervals. Guest lowering includes instructions inserted by the compiler within
that source interval. Unmapped generated scaffolding is reported separately.

The generated ABI is frameless and uses tail links. No invented DWARF unwind
rules are exported; sampled IP attribution is valid without assuming a host
call chain through generated code. Machine bytes are exported at publication
and updated link intervals are exported before execution reopens.

One diagnostic worker owns all file I/O. Publication enqueues bounded messages;
a 128 MiB payload budget and 262,144-event queue bound memory independently of JIT
residency. The binary dump has a 4 GiB limit. Overflow or I/O failure is reported
as incomplete diagnostic coverage and never changes emulation semantics. Normal
shutdown flushes the writer outside JIT locks. Abrupt termination can leave an
incomplete capture. Compare default-build FPS with separate opt-in captures;
the diagnostic writer and source maps have a measurable cost.

For long gameplay runs, first attribute raw IP samples with the CSV utility.
It streams publication events in time order and retains only live versions, so
repeated linking/reclamation does not retain an unbounded Python history. The
result includes `sampled_publication_indices`. Use these to prepare a focused
view before asking perf to generate ELF/debug files:

```sh
perf script -G --ns -i perf.data -F comm,pid,tid,time,period,ip,sym,dso > samples.txt
python3 dev-tools/performance/incremental/jit_attribution.py \
  jit-PID.regions.csv samples.txt --log game.log > attribution.json
python3 dev-tools/performance/incremental/focus_jit.py \
  /absolute/path/jit-PID.dump perf.data attribution.json /tmp/jit-focus
perf inject --jit -i /tmp/jit-focus/perf.data -o /tmp/jit-focus/perf-jit.data
```

The destination must be new, with a shorter absolute path than the original
dump. Original captures remain untouched. The tool copies only publications
actually used by samples, preserving indices, timestamps and DEBUG/LOAD order.
It redirects the jitdump marker filename in a copy of perf.data without changing
any event size, sample or native address. Unsupported file formats fail explicitly.
Direct `perf inject` on an entire long capture can create millions of unnecessary
ELF files and use excessive memory; the focused view gives the same sampled
native identities with bounded analysis input.

Chain adapters choose temporary x86 registers at emission time from the actual
source and target contracts. Source values, clean bindings and packed flags stay
protected until physical copies; installed target values stay protected during
missing-input loads. Full-register contracts retain the save/restore adapter.
Immediate canonical stores that fit MOV memory,imm32 need no borrowed register.
No runtime liveness walk, canonical state duplication, epoch transition or host
call is introduced on linked edges.

## Compile phases and immutable contracts

With `frame-trace`, LCQ records `cpu.lcq.capture_decode`, `cpu.lcq.lower`,
`cpu.lcq.backend`, `cpu.lcq.stage_adapters` and `cpu.lcq.install_publish`.
`backend` includes the backend's lowering, register allocation and native
emission; it is not an exclusive register-allocation measurement. Publication
contains nested `cpu.unit.validate` events. Bridge construction has
`cpu.bridge.emit_install`. Subtract nested spans when reporting exclusive cost.

Each compiler owner retains capture and exit-map scratch. Capture memoization
is restricted to the current image's instruction ordinal and exact word: a
tracking restart can replace bytes or shorten the image. The final coherent
image supplies all executable dependencies and precise fetch faults.

Published contracts share immutable bindings with a dense architectural-value
index. Publication checks the whole contract before exposing it; bridge builders
accept published units and retain dynamic admission, ABI and version checks.
They do not revalidate or linearly search immutable binding lists.

A native indirect-cache record stores source code version, state-map ordinal,
target PC and host destination (32 bytes). The process-local, never-reused source
version and ordinal identify the complete source context. Cold preparation
requires the target's complete execution key to match that source at its PC.
The per-vCPU native probe therefore checks source version, ordinal and PC, while
cold cache identity still includes the full target key and target versions.
Retirement clears native ways before releasing their strong owners. This also
shortens the indirect lookup tail used by successful return-stack predictions.
Return-stack predictions contain only 64-bit guest PCs. The complete execution
context (address space, CPU profile, platform and FP specialization) is checked
once at canonical admission and clears predictions when it changes. Static and
dynamic native links preserve that context; native return probes therefore compare
only the PC. The guest-thread stack occupies 168 bytes, including its shared
context, and never owns native code. The warmup compiler ABI revision invalidates
profiles containing the previous return-stack layout.

## Validated CPU warmup profiles

Set `warmup_profile = true` in `[cpu]` to collect and consume hints. It is
disabled by default: proactive compilation can compete with demand work, and
the measured workload has not shown a warm-profile startup benefit. The runtime library opts in
through `ProcessBuilder::with_jit_warmup_directory`. Interpreter execution does
not collect or consume CPU profiles.

Profiles live in `$XDG_CACHE_HOME/nixe/cpu`, or `$HOME/.cache/nixe/cpu`.
Their identity includes all initialized module content, CPU execution context,
compiler/ABI revision, host ABI, host ISA capabilities and code-generation
options. Module placement is excluded; records use module offsets and exact
instruction words. A module update, context or host-policy mismatch is a miss.

Profiles contain no native code, process pointers, old reachability identities
or executable addresses. One existing compiler worker rebuilds likely LCQ
fragments using normal live-memory capture, execute-permission checks,
compilation, fault-map creation, lifetime registration and publication.
Current bytes must match each hint exactly. Already available or currently
claimed fragments are skipped; normal demand compilation remains authoritative.
Pending HCQ work is serviced between small warmup batches. Shutdown cancels
startup/maintenance waits and joins the same fixed worker pool.

A profile is bounded to 65,536 fragments and 32 MiB of payload. Shutdown writes
through an owned temporary file and atomic rename. The dedicated directory
retains at most eight profiles; unrelated files and symlinks are excluded from
eviction. Missing, stale or corrupt files are optional-hint misses. Publication
and backend implementation failures still propagate as emulator failures.

Warmup moves compilation earlier; it does not eliminate compilation or
accelerate already-hot arithmetic. Measure cold and warm startup separately,
and compare total CPU work as well as title-readiness latency and gameplay FPS.
