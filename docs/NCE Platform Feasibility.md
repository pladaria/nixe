# Native Code Execution Platform Feasibility

Status: design record
Revision: 2026-08-11

This note records integration seams, not provider commitments. Nixe must probe
each facility at runtime and expose an unavailable engine with a typed reason.
No frontend may infer availability from the operating-system name or CPU ISA.
Every NCE implementation remains behind `EngineProvider`, `EngineDomain`, and
`EngineExecutor`; raw framework objects, file descriptors, and host pointers
must not cross into runtime or scheduler code. `DomainMemoryBinding` supplies
the common mapping and retained-canonical-backing seam, so runtime never
downcasts a domain or selects a native-specific path.

## Common requirements

An NCE implementation needs an AArch64 host, permission to create a VM and
vCPUs, lossless import/export of all `ThreadCpuState`, explicit guest-memory
map/unmap/protect notification, interrupt injection, and precise normalization
of SVC, instruction/data abort, timer, interrupt, and shutdown exits. It also
needs a small provider-private supervisor because Nixe executes guest user code
while the virtualization facility exposes a virtual machine, not a direct
user-mode process executor.

Canonical `ExecutionMemory` remains semantic authority. An NCE domain may mirror or
map its pages only while it tracks invalidation and dirty generations. Every
bounded `run_slice` must return canonical thread state; mapping changes reach
each executor through `synchronize_address_space`, and executors acknowledge an
invalidation only after stale translations cannot be re-entered. Domain shutdown
runs after its executors are released. Protected VM memory that the host cannot
read cannot satisfy this contract without an explicit shared-memory and
reconciliation design.

NCE providers consume semantic mapping and content invalidation through their
own acknowledged domain cursor. The records contain no JIT region, link-cell,
software-TLB, Cranelift, VM, or vCPU-framework identity. An NCE provider may
derive VM mappings and dirty tracking from canonical memory, but it never
depends on the JIT execution frame or helper ABI and never makes its mirrored
view authoritative.

## Apple silicon Hypervisor.framework

Versioned basis: Apple Developer Documentation retrieved 2026-08-11 for the
[Hypervisor framework](https://developer.apple.com/documentation/hypervisor),
[`hv_vcpu_create`](<https://developer.apple.com/documentation/hypervisor/hv_vcpu_create(_:_:_:)>),
[`hv_vm_map`](https://developer.apple.com/documentation/hypervisor/1441187-hv_vm_map),
and the
[`com.apple.security.hypervisor` entitlement](https://developer.apple.com/documentation/BundleResources/Entitlements/com.apple.security.hypervisor).

- Privilege and probe: the process needs the hypervisor entitlement. The probe
  must check Apple silicon, VM creation, required register access, and the
  maximum supported vCPU count. Entitlement or policy rejection is
  `PrivilegeUnavailable`, not a guest error.
- Lifecycle: create one framework VM per Nixe NCE domain and one
  `hv_vcpu_t` per active virtual CPU. Apple documents vCPU creation for the
  current thread, so the provider's worker owns creation, run, register access, and
  destruction for that vCPU.
- Memory: `hv_vm_map`/unmap/protect operations consume domain mapping
  notifications. Executable mappings must never grant host write and guest
  execute simultaneously unless the canonical memory policy explicitly permits
  it; code mutation is reconciled through canonical generations.
- Exits and interrupts: `hv_vcpu_run` updates the vCPU exit structure. The
  provider translates exit reason and syndrome directly into `EngineExit` and
  consumes interrupts through the common executor control path.
- Supervisor: a minimal EL1 image must establish the EL0 address space, trap
  SVC and faults, virtualize timer state, and return a bounded exit record. Its
  ABI is provider-private and versioned with the provider.
- Feasibility: plausible on entitled Apple-silicon macOS applications, but
  unavailable until register round-trip, alias coherency, timer, and exception
  conformance tests pass. The common contract requires no Apple-specific change.

## Linux AArch64 KVM

Versioned basis: Linux kernel
[`latest` KVM userspace API](https://www.kernel.org/doc/html/latest/virt/kvm/api.html)
retrieved 2026-08-11. A future implementation must additionally record the
minimum tested kernel version and UAPI headers in its crate.

- Privilege and probe: open `/dev/kvm`, verify `KVM_GET_API_VERSION`, required
  capabilities, AArch64 target support, vCPU features, and usable VM creation.
  Missing device access or security-policy permission is a typed host
  rejection.
- Lifecycle: `KVM_CREATE_VM` creates the domain FD, `KVM_CREATE_VCPU` creates
  one vCPU FD per executor, `KVM_RUN` enters it, and closing the owned FDs tears
  the domain down. FDs never leave the provider crate.
- Memory: canonical host mappings become KVM memslots through
  `KVM_SET_USER_MEMORY_REGION`. Map/unmap/protect changes must quiesce affected
  vCPUs and update memslots before acknowledging the generation. Dirty logging
  or an equivalent explicit reconciliation strategy is mandatory for writable
  mirrors and aliases.
- Exits and interrupts: KVM run exits and Arm register ioctls map to canonical
  `ThreadCpuState`, `EngineExit`, and the executor control path. Unsupported or
  lossy register sets reject the profile before execution.
- Executable memory and supervisor: KVM models guest-physical memory rather than
  the runtime's process permissions, so the provider-private EL1 supervisor must
  enforce guest stage-1 permissions and trap EL0 SVC/faults. Host W^X and
  security-module policy remain probe-time constraints.
- Feasibility: plausible on AArch64 Linux with accessible KVM and suitable Arm
  virtualization support. The current contract covers its VM, vCPU, memslot,
  register, interrupt, and exit seams.

## Android virtualization

Versioned basis: Android Open Source Project
[AVF overview](https://source.android.com/docs/core/virtualization) (updated
2026-06-25) and
[AVF architecture](https://source.android.com/docs/core/virtualization/architecture)
(updated 2026-06-17), retrieved 2026-08-11.

- Availability and privilege: AVF is present only on supporting devices; its
  Java APIs are present only there and are optional, and its native surface is
  an NDK subset. A probe must use supported platform APIs and must not assume
  that an ARM64 Android device exposes `/dev/kvm` to an application.
- Lifecycle: AVF's `VirtualizationService` and crosvm own VM/vCPU lifecycle.
  The documented reference VMM uses one POSIX thread per KVM vCPU and enters
  through `KVM_RUN`, but this does not imply that an ordinary application may
  control arbitrary low-level vCPU state.
- Memory and payload policy: protected pVM pages are donated and become
  inaccessible to the host except where explicitly shared. AVF also boots
  firmware and validated VM payloads. Those properties conflict with Nixe's
  current requirement to inspect and reconcile arbitrary canonical process
  pages and to import/export every thread register at instruction boundaries.
- Traps and interrupts: the underlying KVM/crosvm stack has exits and vCPU
  control, but a usable public application API exposing the exact trap,
  register, mapping, and interrupt operations required by the common domain and executor contracts
  has not been established.
- Executable memory and supervisor: any implementation would need a supported
  VM payload containing Nixe's supervisor and a lawful shared-memory protocol;
  it cannot assume arbitrary executable payload or direct memslot control.
- Feasibility: **unsupported NCE profile** for now. Android `auto` may still
  select the portable JIT when the host ISA and executable-memory policy pass
  its independent capability probe, otherwise it selects the interpreter. It
  must not select an NCE until a supported virtualization API satisfies the
  common conformance suite. Direct KVM access on a particular rooted or vendor
  device is not a portable Android NCE capability.

## Contract mapping and decision

| Platform operation                  | Common seam                                      |
| ----------------------------------- | ------------------------------------------------ |
| Host/device/entitlement probe       | `EngineProvider::probe` and rejection report     |
| VM creation and destruction         | `create_domain` / `EngineDomain::shutdown`       |
| vCPU creation, migration, registers | `create_executor` / canonical `RunRequest` state |
| Map, unmap, protect                 | `DomainMemoryBinding` and executor synchronization |
| Dirty pages and invalidation        | canonical generations and executor synchronization |
| SVC, abort, timer, interrupt exit   | normalized `EngineExit`                          |
| Virtual interrupt delivery          | `EngineControl`                                  |
| Stable stop boundary                | canonical state on every `run_slice` return      |

HVF and Linux KVM map to the common contracts without exposing framework handles.
Android currently maps only at the conceptual VM lifecycle level and therefore
remains unavailable. Scheduler work may depend on these common seams, but not on
the existence or behavior of any proposed platform NCE provider.
