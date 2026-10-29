# CPU state ownership and interchange contract

Linux x86-64 is the validated host. The model follows stopped-core register access and full scheduler
switch transfers in [Eden's interface](https://git.eden-emu.dev/eden-emu/eden/raw/commit/67bada77f8a43a90da2e94e89b8e7da73c256989/src/core/arm/dynarmic/arm_dynarmic_64.cpp)
and [scheduler](https://git.eden-emu.dev/eden-emu/eden/raw/commit/67bada77f8a43a90da2e94e89b8e7da73c256989/src/core/hle/kernel/k_scheduler.cpp).
Nixe retains coordinator service continuations instead of adopting host fibers.

## Exclusive register authority

Each guest has one movable [`ThreadState`](src/state.rs), exposed to runtime as
`GuestCpuState`. It contains either saved architectural authority, a unique
native owner, or explicit unavailable state. Full state includes X0–X30, SP, PC,
NZCV, V0–V31, FPCR/FPSR and both TLS registers.

Saved state is an [`A64State`](../cpu/src/state/a64.rs). When resident, **all**
registers are authoritative in Dynarmic and bridge TLS storage. The reusable
boxed save area is inaccessible while native/unavailable; its old bytes are not
a current snapshot. No scalar overlay or duplicated authoritative context exists.

`ThreadObject` owns the value between scheduler leases. Dispatch moves it
into `VcpuExecutionState`; its register slot contains `None` until validated reconciliation.
Only the holder may read/write registers. Worker native entry exclusively borrows
that owner synchronously, so callbacks and other threads cannot obtain it while
execution is active. After return, the coordinator can access the stopped native
context through the returned owner. No additional register RPC is needed.

`ThreadState` never exposes a raw native handle. A native owner retains its core,
execution memory and local native monitor storage. Its atomic claim prevents another saved
context from entering that core until the old owner is materialized or discarded.
Cloning creates an independent saved snapshot; it never duplicates native
ownership. Snapshots remain immutable when the current context later changes.

A Dynarmic core belongs to one process on one worker; it can execute many guest
threads. A guest ID, scheduler lease, vCPU, process-local native core and host
`NativeWorker` fault context are distinct identities. The fault context belongs
to its host TID and is still used only by that worker's captured invocations.

## Boundaries and consumers

| Boundary | Required ownership and action |
| --- | --- |
| Creation | Initialize complete saved state, including SP, PC, TLS and entry arguments. |
| Dispatch | Move the sole register owner into the request. Evict the old guest on that process/vCPU before dispatching a different guest. |
| First entry or genuine replacement | Import complete saved state once through public bulk APIs; establish native ownership. |
| Same guest after SVC or event boundary | Keep the complete context native. Re-enter without full import/export. |
| Ordinary SVC | Read only needed registers through the stopped owner; write results directly to it. Runtime validates the SVC source and applies Next/Retry/At exactly once. No eager GPR/SIMD diagnostic snapshot is required. |
| Blocking / wake-up | Install the continuation before suspension. A blocked native context remains readable through its owner until eviction, which saves it completely. Generation-checked completion writes target that current owner, whether saved or native. |
| Guest switch | Save the previous complete context before releasing its claim. Load the next guest on the worker. Every native entry clears local exclusives. |
| Migration | Before either worker accepts a new lease, save the source context. Remove source residency bookkeeping. The destination imports the current saved value and clears local exclusives on entry. |
| Full thread query / debugger | Reject Running, absent or unavailable state. Obtain an owned complete snapshot explicitly; native residency may continue. Horizon also enforces its paused-thread query requirement. |
| Diagnostic consumer | Read an owned snapshot explicitly from the stopped `ThreadState`. Normal returns omit `ExecutionReport.context`; omission does not imply the register owner is unavailable. |
| Unsupported instruction / architectural exception / instruction-fetch failure | Native return is synchronized; correct the source PC and capture a current compact diagnostic context. Full snapshots remain available from the owner. Unsupported instructions stop explicitly. |
| GPU visibility demand | Complete the access inside its native memory callback. Capture the physical range, suspend admission while resolving or waiting, then resume and queue code invalidations before returning the value. Native frames remain alive until Run returns; completed instruction prefixes are never replayed. Translation read-ahead records unavailable code without waiting. |
| Data-cache maintenance | Native translation removes maintenance hints and lowers DC ZVA to ordinary stores to the aligned block. Instruction-cache callbacks queue native invalidations directly and halt for their application after Run returns. |
| MMIO memory instruction | Resolve each individual access synchronously inside its callback using the captured physical range. Waiting may release admission, but never snapshots or reconstructs the instruction. A later access cannot replay an earlier side effect. |
| SVC failure / termination / loader return | Preserve the source and capture diagnostics at the actual consumer. Loader-return reports supply their compact context explicitly. |
| Native C++ or fault-runtime failure | Discard register authority, stop the JIT process and report `CpuFault.context = None`. The old save area cannot be read or resumed, on any subsequent execution request. |
| Lost/panicked worker or stale result | Do not associate unverified state with the expected lease. Discard the result and leave expected table state absent; fault the process. |
| Dispatch rejection before execution | The returned request retains its known owner and can be restored without fabricating state. |
| Concurrent process stop | Drain all leases. Successful late results can restore current authority; unsynchronized failures remain unavailable. |
| Retirement / teardown | Halt and drain execution, synchronize invalidation and clear exclusives on workers, then materialize retained owners before releasing cores/backing. Box cold teardown snapshots so command size is not governed by a full context. |

Asynchronous control requests halt only. It must never read, materialize or write
live registers. The metadata mutex and memory execution gate do not confer
register ownership. Cache/mapping synchronization stays on the worker with the
existing short-lived execution leases. The native callback context pointer is
borrowed for one invocation and cleared on both successful and caught-error exits.
Fatal unattributed native signals retain the existing terminal fault behavior.

Sources: [dispatch](../runtime/src/process/dispatch.rs),
[execution](../runtime/src/process/execution.rs),
[worker](../runtime/src/coordinator/worker.rs),
[exception handling](../runtime/src/process/exception.rs),
[thread queries and migration](../runtime/src/coordinator/thread.rs),
[fault runtime](../cpu-direct-memory/src/lib.rs).

## Native execution and memory policy

Native execution uses the architectural wall clock with cycle counting disabled.
The coordinator waits for CPU completions, host events or actual deadlines on a
single ingress. Scheduler preemption and process stop interrupt native execution;
there are no instruction quotas or synthetic per-block timeslices. User-mode
wait, event and yield hints continue in native code.

The normal memory policy matches the reference Auto configuration with ignored
guest data aborts. Invalid guest data reads return zero and invalid stores have
no effect. The native optimizer does not retain register writes for precise data
abort reconstruction, and there are no per-access halt checks or instruction
completion executor. Instruction-fetch failures, unsupported instructions and
architectural exceptions remain explicit failures. Device/coherency/backend
errors stop the process and discard unsynchronized register authority; they are
not converted to invalid guest access defaults.

## Exclusive monitors

Reservations are not serialized with architectural state. Native execution uses
the reference Auto policy: ignore the global monitor lock and test stores with
the address and expected value captured by native exclusive loads. Each core has
its own native monitor storage for that policy. There are no physical reservation
versions, write epochs, store instrumentation or guest-switch monitor RPCs.
Every native entry clears the local exclusive state, as does the vendored x64
SVC emission. CLREX remains a native operation.

Ordinary writable RAM uses native fastmem and atomic compare-and-exchange.
Observed, device-owned or exceptional mappings use the canonical page table
fallback and memory callbacks. A successful callback CAS publishes observation
changes through the same memory access path as other checked atomics. Pending
coherency accesses suspend admission inside the callback and never replay the
exclusive instruction. The Auto policy deliberately does not detect an
intervening change-and-restore write when the expected value matches again.

## Complete interchange representation

The checked C payload in [native/state.h](native/state.h) is the canonical A64
save-area layout. Rust lends it only during explicit load/save or snapshot calls;
C++ retains no pointer to it. Resident state is accessed exclusively through
Dynarmic's public APIs, never through private object offsets.

| Field | C / Rust representation | Offset | Bytes |
| --- | --- | ---: | ---: |
| X0–X30 | `uint64_t[31]` / `[u64;31]` | 0 | 248 |
| SP | `uint64_t` / `u64` | 248 | 8 |
| PC | `uint64_t` / `u64` | 256 | 8 |
| V0–V31 | `uint64_t[32][2]` / `[[u64;2];32]` | 264 | 512 |
| TPIDR_EL0 | `uint64_t` / `u64` | 776 | 8 |
| TPIDRRO_EL0 | `uint64_t` / `u64` | 784 | 8 |
| NZCV | `uint32_t` / `u32` | 792 | 4 |
| FPCR | `uint32_t` / `u32` | 796 | 4 |
| FPSR | `uint32_t` / `u32` | 800 | 4 |

Total size is 808 bytes, alignment 8, including four bytes of tail padding.
Architectural field payload is 804 bytes. The payload has no pointers, tags,
handles, host flags, native exclusive state or implicit owner. XZR has no storage;
SP is separate. Vector lane 0 is bits 63:0, lane 1 bits 127:64. Use explicit
fixed-width lanes rather than depending on Rust/C++ 128-bit type alignment.
Linux x86-64 is little-endian; raw field storage is not a portable save-state or
guest wire format. Do not serialize its padding or Horizon's thread-context
layout as this C ABI.

NZCV contains only its architectural mask. FPCR/FPSR use guest values and the
supported semantics of Dynarmic's public setters/getters; host MXCSR or host
floating-point control is separate and restored by the invocation gateway.
Dynarmic masks FPCR to its implemented bits and reconstructs FPSR; do not claim
preservation of reserved bits or memcpy its private JitState. TPIDRRO_EL0 remains
runtime-owned but follows the guest context between workers; TPIDR_EL0 may be
modified by guest code. Neither is a host TID or a native core address.

Rust compile-time size/alignment/offset assertions on canonical `A64State` and
C++ static assertions check every field. C++ additionally checks standard layout
and trivial copying. Rust ergonomic
getters/setters must preserve the same bit-level register behavior.

The bridge uses Dynarmic's public `Get/SetRegisters` and `Get/SetVectors`.
Its four typed array values copy the GPR/vector field bytes required by those
public APIs, with compile-time size and vector triviality checks. They are not
a second complete architectural state representation. C arrays and `std::array`
are not reinterpreted as each other's types. Scalar getters/setters retain their
normalization semantics. The bridge does not depend on Dynarmic object offsets,
its vector backing implementation, or a borrowed reference to a public API that
currently returns a value.

