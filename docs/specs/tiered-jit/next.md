# Deferred host-memory portability work

Status: not implemented. By maintainer decision, ARM host development and
native validation are deferred. Current work focuses on AMD64. Retain the
existing ARM implementation without claiming native conformance; neither its
removal nor replacement by NCE/hypervisor execution has been decided. Do not
build a custom Pi kernel to unblock this work.

The two problems below are independent: insufficient host virtual address
space with 4 KiB pages, and host pages larger than the guest's 4 KiB granule.
Resume [Task 10's ARM validation](task-10-plan.md) only after its host-memory
prerequisites are resolved. AMD64 validation can proceed separately; it does
not close cross-target conformance.

## 1. Segmented arenas for limited host virtual address space

### Observed blocker

The Pi's installed `kernel8.img`, `6.18.50+rpt-rpi-v8`, has 4 KiB pages but
`CONFIG_ARM64_VA_BITS=39`: a 512 GiB userspace address range. Nixe reserves
512 GiB contiguously for canonical backing (`host_mapped.rs`), then needs
another 512 GiB plus guards for a 39-bit guest's direct arena (`direct.rs`).
Even the first reservation cannot fit alongside the host executable, stack
and libraries. A read-only diagnostic using temporary PROT_NONE mappings
confirmed ENOMEM at 512 GiB and success at 256 GiB. These are virtual address
reservations, not requests to populate that much physical RAM.

The maintainer's `hello-world` launch failed before guest execution. The
interpreter shares the canonical backing and does not bypass this first
failure. Reducing only that reservation would not fix the JIT's guest arena.
Do not shrink the architectural guest address space to make a demo fit.

### Proposed work, pending implementation design and measurements

1. **Prototype guest addressing before migrating the backend.** Compare a
   contiguous arena with an indexed table of large, power-of-two segments on
   the same host. Force either mode for controlled comparisons, not via a
   per-access branch. Measure memory-heavy loops, representative instruction
   mixes, native code size and compilation cost. Choose segment size and
   lookup layout from these results; no size or overhead target is established
   yet. Keep diagnostics external and do not add a benchmark framework.
2. **Segment canonical storage.** Replace the mandatory 512 GiB reservation
   with lazily allocated chunks while retaining stable backing pointers,
   backing identities, shared-file offsets and one authority for guest bytes.
   Define chunk ownership/reclamation and allocation failure handling. This
   change need not add work to JIT accesses through guest aliases.
3. **Select guest-arena mode when creating the process.** Keep the existing
   contiguous mode on capable hosts and use segmented mode where required.
   Define selection and allocation-failure policy before coding; do not infer
   capability merely from the CPU architecture. Freeze the mode for the process
   lifetime and pass it to both LCQ and HCQ. Do not switch beneath published
   code or silently fall back to the interpreter.
4. **Implement segmented guest mappings and native addressing.** Reserve
   segments as needed and emit table lookup plus offset arithmetic directly
   in generated code. Retain hardware-enforced 4 KiB guest permissions on
   this host; segmentation alone does not require per-page software permission
   checks. Keep the contiguous path free of mode checks or lookup overhead.
   Review the reserved arena register, `nixe_arena_addr` contract in the fork,
   fault metadata and native adapters; update only the affected contracts.
5. **Preserve memory semantics and lifetime.** Define absent-segment behavior,
   out-of-range addresses, guard handling and cross-segment accesses. Cover
   atomics/exclusives, aliases, concurrent map/protect/unmap, visibility and
   dirty tracking, host-pointer-to-guest fault attribution and precise retry.
   Coordinate table publication and segment reclamation with execution so
   generated code cannot dereference stale bases. Do not turn hot accesses
   into repeated faults or Rust helper calls. Preserve required partial effects
   and atomicity at boundaries.
6. **Expose actionable failures and validate end to end.** Preserve the failing
   operation, requested size and OS error instead of collapsing all host-mapped
   failures into `CanonicalPageError::ResourceExhausted`. Add focused regression
   coverage for both modes and verify canonical storage independently. Compare
   startup, execution and memory footprint against contiguous mode on the same
   host; record virtual reservations separately from committed memory/RSS.
   Then validate actual execution on the Pi's existing 4 KiB/39-bit kernel.

This affects memory ownership, mappings, generated accesses and fault handling,
not a replacement of LCQ/HCQ, region discovery or background compilation.
Reuse unaffected infrastructure; no parallel legacy implementation for tests.
Keeping two arena modes is justified by distinct host capabilities only if
measurements support the contiguous fast path. Do not claim a performance
improvement or acceptable segmented overhead before measuring it.

## 2. Support 16 KiB host pages

### Current limitation

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

### Design work to undertake

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

### Sequencing

When ARM development resumes, address the limited-address-space blocker above
and validate native execution on the existing 4 KiB Pi kernel. Scope 16 KiB
portability separately, with a measured prototype and focused correctness
tests. Segmented arenas do not by themselves solve subpage permissions or
arbitrary 4 KiB backing mappings on a 16 KiB host.

The inspected Pi 5 currently has both `kernel_2712.img` (16 KiB) and
`kernel8.img` (4 KiB) installed. Selecting the latter fixes the page-granule
mismatch, but its 39-bit virtual address space still prevents current guest
launches. Raspberry Pi documents the
[kernel selection setting](https://www.raspberrypi.com/documentation/computers/config_txt.html#kernel)
and [supported kernel images](https://www.raspberrypi.com/documentation/computers/configuration.html).
