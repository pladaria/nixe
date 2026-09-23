# Follow-up: support 16 KiB host pages

Status: deferred design work, not implemented. Native validation in
[Task 10](task-10-plan.md) uses a 4 KiB-page kernel; that does not establish
compatibility with 16 KiB hosts.

## Current limitation

The guest memory granule is 4 KiB. `crates/memory/src/direct.rs` rejects host
page sizes other than 4096, and `crates/cpu-jit/src/engine.rs` requires the
LinuxDirect memory backend. The production JIT does not fall back to checked
memory when that backend is unavailable.

Direct accesses use an arena base plus the guest virtual address. Host mappings
and protections enforce guest mappings and permissions, with native faults
recovered through JIT state maps. On a 16 KiB host, four adjacent guest pages
cannot have arbitrary independent protections. They may also refer to
noncontiguous backing offsets or aliases that a single host mapping cannot
represent. Linux imposes host-page alignment on mappings and protection changes;
see [mmap](https://man7.org/linux/man-pages/man2/mmap.2.html) and
[mprotect](https://man7.org/linux/man-pages/man2/mprotect.2.html).

Changing the guest granule, removing the host-page-size check, or rounding
permissions up to 16 KiB is not a correct solution. Shadow copies would require
additional alias/coherency machinery and must not become a second authority for
guest memory.

## Design work to undertake

Select a memory strategy based on correctness and measured execution cost:

- Software guest-page translation with permissions and a native inline fast
  path, potentially using a translation cache. Ordinary loads/stores must not
  require a Rust call per access.
- Alternatively, a hybrid strategy retaining direct mappings where guest
  mappings are representable and translating incompatible regions. Establish
  whether its performance benefit justifies the additional complexity; do not
  assume incompatible accesses are rare or repeatedly fault through Rust on a
  hot path.

These are alternatives to evaluate, not an approved implementation design.
Retain the existing direct fast path for 4 KiB hosts if it remains the fastest
correct implementation. Select the host backend outside the per-access hot
path; do not add a host-page-size check to every load/store. Distinct supported
host backends are not legacy compatibility paths.

The work primarily affects memory backing/mappings, generated memory accesses,
fault recovery and their synchronization. Reuse LCQ/HCQ, region discovery,
workers, native linking and unaffected instruction lowering. Explicitly cover:

- Independent 4 KiB permissions and arbitrary guest-to-backing mappings.
- Shared physical aliases, including concurrent accesses.
- Cross-page loads/stores and architecturally required partial effects.
- Atomic operations, exclusive reservations and physical identity.
- Concurrent mapping/protection changes and translation invalidation.
- Precise guest faults, retry, state reconstruction and execution accounting.

Inline translation adds instructions and table accesses; translation tables
also consume memory. No numerical overhead estimate is established. Compare
code size, compilation/startup cost, steady-state execution and memory usage
against the current direct backend, and demonstrate that 4 KiB-host performance
has not regressed. Validate on native 16 KiB hardware, not only simulation.

## Sequencing

First validate the existing JIT on native AArch64 with 4 KiB pages in Task 10.
Then scope 16 KiB portability as a separate implementation task with a measured
prototype and focused correctness tests. Do not turn host setup into this
redesign or claim that switching kernels solves product portability.

The inspected Pi 5 currently has both `kernel_2712.img` (16 KiB) and
`kernel8.img` (4 KiB) installed. Selecting the latter permits current testing
without changing emulated memory semantics. Raspberry Pi documents the
[kernel selection setting](https://www.raspberrypi.com/documentation/computers/config_txt.html#kernel)
and [supported kernel images](https://www.raspberrypi.com/documentation/computers/configuration.html).
