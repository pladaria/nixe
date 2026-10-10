# CPU state ownership and interchange contract

T02/T03 interchange with T04 whole-context residency. Linux x86-64 is the
validated host. The model follows stopped-core register access and full scheduler
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

`GuestThread.state` owns the value between scheduler leases. Dispatch moves it
into `VcpuExecutionState`; the table contains `None` until validated reconciliation.
Only the holder may read/write registers. Worker native entry exclusively borrows
that owner synchronously, so callbacks and other threads cannot obtain it while
execution is active. After return, the coordinator can access the stopped native
context through the returned owner. No additional register RPC is needed.

`ThreadState` never exposes a raw native handle. A native owner retains its core,
process monitor and execution memory. Its atomic claim prevents another saved
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
| Same guest after SVC, budget or event boundary | Keep the complete context native. Re-enter without full import/export. |
| Ordinary SVC | Read only needed registers through the stopped owner; write results directly to it. Runtime validates the SVC source and applies Next/Retry/At exactly once. No eager GPR/SIMD diagnostic snapshot is required. |
| Blocking / wake-up | Install the continuation before suspension. A blocked native context remains readable through its owner until eviction, which saves it completely. Generation-checked completion writes target that current owner, whether saved or native. |
| Guest switch | Save the previous complete context before releasing its claim. Load the next guest on the worker; clear local exclusives for the changed guest ID. |
| Migration | Before either worker accepts a new lease, save the source context and clear its local exclusive state. Remove source residency bookkeeping. The destination imports the current saved value. |
| Full thread query / debugger | Reject Running, absent or unavailable state. Obtain an owned complete snapshot explicitly; native residency may continue. Horizon also enforces its paused-thread query requirement. |
| Diagnostic / replay consumer | Request compact context in `JitRunRequest`. Normal returns may omit `ExecutionReport.context`; omission does not imply the register owner is unavailable. |
| Precise abort / unsupported instruction / architectural exception | Native return is synchronized; correct the source PC and capture a current compact diagnostic context. Full snapshots remain available from the owner. No interpreter fallback. |
| GPU visibility demand | Return at the faulting instruction through MemoryAbort and release the execution lease before resolving the retained page. Loads/atomics retry natively with mapping/permission revalidation. Ordinary stores complete the remaining checked stores and writeback, skipping the native prefix. Translation read-ahead records unavailable code without waiting. |
| Data-cache maintenance | Complete non-zeroing VA operations in the native callback when canonical RAM is CPU-visible. The block still checks halt requests. GPU-owned RAM, mapping faults, set/way operations and cache zeroing return through the existing checked maintenance/trap path after releasing the execution lease. Instruction-cache invalidation still returns to synchronize compiled code. |
| MMIO memory instruction | Exit before the first device side effect. Take a complete snapshot and finish with checked semantics outside the lease, skipping any preceding native RAM stores. This prevents a later visibility demand in a pair access from replaying MMIO. |
| SVC failure / termination / loader return | Preserve the source and capture diagnostics at the actual consumer. Loader-return reports supply their compact context explicitly. |
| Native C++ or fault-runtime failure | Discard register authority, stop the JIT process and report `CpuFault.context = None`. The old save area cannot be read or resumed, even with a zero budget. |
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

## Exclusive monitors

Reservations are not serialized with architectural state. Dynarmic owns the
process-wide monitor and core-local reservation. A changed worker guest ID clears
local exclusives; re-entry of the same resident guest through an ordinary budget
boundary preserves the existing policy. Migration explicitly clears the source.
The vendored x86-64 SVC emission also clears local exclusives. Tests of reservation
retention therefore stop before SVC. These rules never fabricate exclusive success.

The interpreter keeps its independent state/monitor and differential role.
Its checked instruction executor also completes cold MMIO memory instructions
and interrupted ordinary store tails after native return; unsupported Dynarmic
instructions remain errors. The store continuation matches the failed start VA
so vector stores and element-level checked semantics agree. Completed stores
are not revalidated or replayed after mappings or contents change during a wait.
Resident RAM execution and load/atomic visibility retries retain native state.

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

