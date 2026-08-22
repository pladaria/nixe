# Nintendo Switch 1 Memory Architecture

This document explains the Nintendo Switch 1 memory model from the point of
view of an application, the Horizon kernel, and an emulator. It then describes
Nixe's current implementation in enough detail to safely inspect or modify the
code.

The public description of Horizon is based on community reverse engineering,
public homebrew interfaces, and the open-source Mesosphere kernel in
Atmosphère. Mesosphere is especially useful because its types and control flow
closely model Horizon, but it is not an official Nintendo specification. When
this document says that Horizon performs an internal operation, the statement
should be read with that provenance in mind.

The most important distinction is:

```text
application-visible virtual memory
                │
                ▼
Horizon mapping state and policy
                │
                ▼
CPU page tables and physical-page ownership
                │
                ▼
DRAM, device memory, and device address spaces
```

These are related layers, not interchangeable names for one table.

## The short version

A process executes with virtual addresses. Horizon assigns that process an
address-space shape, maps code, creates stack and thread-local mappings, and
reserves regions in which heap, aliases, and other mappings may later appear.

Every mapped virtual range has at least:

- a virtual base and size;
- a semantic memory state, such as code, heap, stack, or shared memory;
- read, write, and execute permissions;
- attributes such as uncached or permission-locked;
- one or more physical pages or a device backing; and
- ownership and reference-counting rules.

Two virtual addresses can refer to the same physical page:

```text
virtual code address
        │
        ▼
physical page P
        ▲
        │
writable alias address
```

A write through the alias changes what an instruction fetch through the code
mapping will observe. An emulator must therefore model physical identity, not
only copy bytes into independent virtual ranges.

Horizon maintains both:

1. translation state used by the CPU MMU; and
2. semantic interval metadata used by SVC validation and `QueryMemory`.

A page-table entry alone cannot answer whether a range is heap, stack, IPC
memory, or module code. Conversely, semantic metadata alone cannot translate a
load to physical storage.

Nixe keeps the same conceptual separation, but does not reproduce Horizon's
kernel classes or hardware page-table descriptors literally:

```text
Horizon-visible policy
        │
        ▼
ProcessMemory / CpuMemory contracts
        │
        ▼
ExecutionMemory sparse virtual mappings
        │
        ▼
stable physical-page slots
        │
        ▼
RAM bytes or an MMIO callback
```

## Hardware and architectural foundation

The original Switch is built around NVIDIA's Tegra X1 family and Armv8-A CPU
cores. Ordinary title code normally runs in the non-secure user environment.
It cannot edit translation tables or directly allocate physical DRAM. It asks
Horizon to change its address space through SVCs.

At the CPU architecture level, stage-1 translation converts a process virtual
address into an output physical address and applies permissions and memory
attributes. Arm translation tables are hierarchical: an upper-level entry
either rejects the address, maps a block, or points to another table; a final
entry maps a page. This makes a large sparse virtual address space practical
without allocating a flat descriptor for every possible page. The general
translation-table model is described in Arm's
[Memory management guide](https://developer.arm.com/-/media/Arm%20Developer%20Community/PDF/Learn%20the%20Architecture/LearnTheArchitecture-MemoryManagement-101811_0100_00_en.pdf?revision=1fdc3375-d81c-4457-b786-04fb98557de0).

The page-table descriptor also selects memory properties. Normal cached RAM,
uncached memory, and device registers are not equivalent. Access ordering,
speculation, and cache behavior depend on those properties; see Arm's
[Armv8-A memory model guide](https://developer.arm.com/-/media/Arm%20Developer%20Community/PDF/Learn%20the%20Architecture/Armv8-A%20memory%20model%20guide.pdf?revision=58b1dd0a-3800-4218-b21a-f95a0332034c).

Horizon uses 4 KiB user pages. Mesosphere calls the architecture-specific
page-table implementation from `KPageTableBase` and uses a 2 MiB alignment
when arranging the major virtual regions. The relevant constants and the
32/36/39-bit address-space selection can be inspected in pinned
[`kern_k_page_table_base.hpp`](https://github.com/Atmosphere-NX/Atmosphere/blob/e468f59c9d369b8ebbffa040f4c9fc201b9f75a8/libraries/libmesosphere/include/mesosphere/kern_k_page_table_base.hpp).

The Tegra memory controller, peripheral address map, GPU, and other bus masters
exist below Horizon's process abstraction. A title does not receive arbitrary
access to that physical map. Privileged code decides which physical or device
ranges may be mapped, and GPU access uses a device address space of its own.
See [Graphics and Display Architecture](Graphics%20and%20Display%20Architecture.md)
for the CPU/GPU relationship.

## Process address spaces

### Width is process metadata

The NPDM process metadata selects an address-space type independently of
whether the program executes AArch64 or AArch32 instructions. Public Horizon
types distinguish:

| Process address-space type | Virtual limit used by Nixe |
| -------------------------- | -------------------------- |
| 32-bit                     | `2^32`                     |
| 32-bit without alias       | `2^32`                     |
| deprecated 64-bit          | `2^36`                     |
| modern 64-bit              | `2^39`                     |

The names and flag encodings are present in pinned Atmosphère
[`svc_types_common.hpp`](https://github.com/Atmosphere-NX/Atmosphere/blob/e468f59c9d369b8ebbffa040f4c9fc201b9f75a8/libraries/libvapours/include/vapours/svc/svc_types_common.hpp)
and the process metadata fields are summarized by
[Switchbrew's NPDM documentation](https://switchbrew.org/wiki/NPDM).

A 39-bit address space does not mean that a process owns `2^39` bytes of
physical memory. It means that its virtual addresses are selected from that
domain. Most addresses remain unmapped.

### Reserved regions are placement policy

Horizon derives regions for purposes including:

- the ASLR/map area in which code and other mappings can be placed;
- the heap reservation;
- the alias reservation;
- the stack mapping reservation; and
- kernel-only or TLS/IO placement.

These are virtual windows. Reserving a 128 GiB heap window, for example, does
not allocate 128 GiB of DRAM.

For modern 39-bit processes, large regions are arranged around the process
code with 2 MiB region alignment. ASLR can vary their placement. For older
address-space types, the region geometry is more fixed. Mesosphere represents
the kinds with `KAddressSpaceInfo` and the per-process boundaries in
`KPageTableBase`; see pinned
[`kern_k_address_space_info.hpp`](https://github.com/Atmosphere-NX/Atmosphere/blob/e468f59c9d369b8ebbffa040f4c9fc201b9f75a8/libraries/libmesosphere/include/mesosphere/kern_k_address_space_info.hpp)
and [Switchbrew's memory-layout notes](https://switchbrew.org/wiki/Memory_layout).

A simplified modern layout is:

```text
low virtual addresses
        │
        ▼
ASLR-capable process range
        │
        ├── executable modules
        │
        ├── reserved stack window
        │       └── concrete thread stacks and guard holes
        │
        ├── reserved alias window
        │       └── aliases created by mapping operations
        │
        ├── reserved heap window
        │       └── committed prefix selected by SetHeapSize
        │
        ├── TLS and other mappings
        │
        └── unmapped holes
        │
        ▼
address-space limit
```

The drawing does not imply one universal order. Placement depends on the
address-space type, process code, firmware generation, and ASLR policy.

### ASLR and allocation are separate

ASLR chooses virtual placement. Physical allocation supplies backing pages.
The same physical pages can be mapped at different randomized virtual
addresses without changing their identity.

```text
choose a free virtual range
        │
        ▼
validate region and alignment policy
        │
        ▼
obtain or reference physical pages
        │
        ▼
publish translation entries
        │
        ▼
publish semantic memory-state metadata
```

## What a Horizon memory mapping contains

### Memory state

The low bits reported by `QueryMemory` identify the mapping's semantic type.
Important states include:

| State | Typical origin or role |
| ----- | ---------------------- |
| Free | No mapping covers the range. |
| Io | Physical device or IO mapping. |
| Static | Static mapping established while creating a process. |
| Code | Initial process code, normally executable and immutable. |
| CodeData | Mutable transition of initial process code/data. |
| Normal | Ordinary heap-style memory. |
| Shared | Shared-memory object mapping. |
| AliasCode | Module code mapped through process-code mapping operations. |
| AliasCodeData | Mutable form of aliased module code. |
| Ipc | Temporary IPC buffer mapping. |
| Stack | Stack mapping created through the mapping mechanism. |
| ThreadLocal | Per-thread local region. |
| Transfered | Isolated transfer-memory mapping. |
| SharedTransfered | Shared transfer-memory mapping. |
| SharedCode | Memory from another process. |
| Inaccessible | Reserved but inaccessible range. |
| NonSecureIpc / NonDeviceIpc | IPC mappings with stricter device/security policy. |
| GeneratedCode / CodeOut | Code-memory/JIT-related mappings. |

The complete contemporary enumeration and its capability flags can be read in
pinned Atmosphère
[`kern_k_memory_block.hpp`](https://github.com/Atmosphere-NX/Atmosphere/blob/e468f59c9d369b8ebbffa040f4c9fc201b9f75a8/libraries/libmesosphere/include/mesosphere/kern_k_memory_block.hpp).
Public libnx names are available in pinned
[`svc.h`](https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/kernel/svc.h).

The kernel internally attaches capability bits to a state: whether it may be
reprotected, aliased, transferred, used for IPC, mapped into a device address
space, queried for a physical address, or used as code memory. Therefore a
state is policy, not just a label printed by `QueryMemory`.

### Permissions

User-visible permissions are combinations of:

- read;
- write; and
- execute.

Not every numerical combination is accepted by every SVC or state. Horizon
validates both the requested permission and whether the existing memory state
permits that transition. Normal user mappings are generally kept W^X: writable
and executable mappings are not freely interchangeable.

Changing permissions affects the page-table descriptors, but also the semantic
metadata that future SVCs and `QueryMemory` observe.

### Attributes and reference counts

Important user-visible attributes include:

- locked;
- IPC-locked;
- device-shared;
- uncached; and
- permission-locked.

`MemoryInfo` also reports IPC and device reference counts. These values let the
kernel reject unsafe operations while another subsystem is using a range.
Attributes may affect whether adjacent intervals can be coalesced in a
`QueryMemory` result.

The public structure has:

```text
MemoryInfo
        │
        ├── base address
        ├── size
        ├── memory state/type
        ├── attributes
        ├── permissions
        ├── IPC reference count
        └── device reference count
```

### QueryMemory returns semantic intervals

`svcQueryMemory` does not return one page-table entry. It returns the semantic
memory block containing the requested address. Horizon normally coalesces
compatible neighboring blocks, but internal merge restrictions and reference
history can preserve a boundary even when the public fields on both sides look
the same. The result must therefore come from the kernel's semantic interval
metadata, not be reconstructed only from page-table entries.

For an unmapped address it returns a bounded free interval:

```text
previous mapping end
        │
        ▼
requested unmapped address
        │
        ▼
next mapping start
        │
        ▼
one Free MemoryInfo interval
```

This behavior is part of the process ABI. Guest `rtld`, allocators, and
diagnostic tools use it to discover mappings and holes. See
[Runtime Linker (rtld)](Runtime%20Linker%20(rtld).md).

## Virtual aliases and physical identity

A virtual mapping is a view. A physical page is the storage identity behind
one or more views.

```text
virtual mapping A: read + execute
        │
        ▼
physical page identity P
        ▲
        │
virtual mapping B: read + write
```

The mappings can have different permissions, states, and addresses while
sharing bytes.

This matters for:

- writable aliases of executable code;
- shared memory;
- transfer memory;
- mappings of another process;
- IPC buffer mappings;
- GPU-visible buffers;
- exclusive monitors; and
- physical-page lifetime.

Copying bytes into two independent host allocations is not a correct alias.
Subsequent writes, reference counts, and invalidation would diverge.

## The principal Horizon memory managers

The following names come from Mesosphere. They provide a useful decomposition
of Horizon's responsibilities even though an emulator need not copy their
class hierarchy.

### KMemoryLayout

`KMemoryLayout` describes the kernel's view of physical and linear-mapped
regions. It connects physical address ranges with purposes and pools.

It answers questions such as:

- which physical region contains an address;
- which pool owns it;
- where its kernel linear mapping is; and
- which regions are available for management metadata.

### KMemoryManager

`KMemoryManager` allocates physical pages from logical pools. The pinned
[`kern_k_memory_manager.hpp`](https://github.com/Atmosphere-NX/Atmosphere/blob/e468f59c9d369b8ebbffa040f4c9fc201b9f75a8/libraries/libmesosphere/include/mesosphere/kern_k_memory_manager.hpp)
defines application, applet, system, and non-secure system pools.

Its responsibilities include:

- allocating contiguous or grouped physical pages;
- choosing allocation direction and alignment;
- maintaining page reference counts;
- freeing pages when the last reference closes;
- tracking free space; and
- applying process resource limits.

The pool is a resource/accounting domain. It is not a process virtual region.

### KPageGroup

`KPageGroup` represents a logical group of physical pages as one or more
contiguous physical runs. It can retain and release references without
requiring the pages to form one physically contiguous allocation. See pinned
[`kern_k_page_group.hpp`](https://github.com/Atmosphere-NX/Atmosphere/blob/e468f59c9d369b8ebbffa040f4c9fc201b9f75a8/libraries/libmesosphere/include/mesosphere/kern_k_page_group.hpp).

Conceptually:

```text
logical page group
        │
        ├── physical run: pages 100..107
        ├── physical run: pages 240..243
        └── physical run: page 900
```

The group can then be mapped into a contiguous virtual range.

### KPageTableBase and KPageTableImpl

`KPageTableBase` owns per-address-space policy and coordinates mapping
operations. Its state includes:

- address-space bounds;
- heap, alias, stack, and kernel-map regions;
- current heap end;
- code and alias-code regions;
- locks;
- the architecture-specific page-table implementation;
- the memory-block manager;
- allocation policy and resource limits; and
- mapped-memory accounting.

`KPageTableImpl` performs the architecture-specific descriptor operations.
Keeping these concepts separate lets the kernel validate Horizon state before
editing hardware translation tables.

### KMemoryBlockManager

`KMemoryBlockManager` stores semantic virtual intervals in an intrusive
red-black tree. Each block records a virtual range and its state, permission,
attributes, reference counters, original permission, and merge restrictions.
The manager can split blocks around an update and coalesce compatible
neighbors afterwards. See pinned
[`kern_k_memory_block_manager.hpp`](https://github.com/Atmosphere-NX/Atmosphere/blob/e468f59c9d369b8ebbffa040f4c9fc201b9f75a8/libraries/libmesosphere/include/mesosphere/kern_k_memory_block_manager.hpp).

This is the structure that makes interval-oriented policy possible:

```text
one semantic block
        │
        ▼
update a range in its middle
        │
        ▼
left unchanged block
        │
        ▼
updated middle block
        │
        ▼
right unchanged block
        │
        ▼
coalesce equal neighbors when allowed
```

### System resources and slab managers

Page tables and interval metadata consume kernel memory too. A process may have
a reserved system-resource area used for page-table pages, memory blocks, and
page-group metadata. Exhausting metadata is distinct from exhausting the
process's allowed data pages. This is why mapping operations can fail with an
out-of-resource result even when the requested virtual range is free.

## Typical Horizon operations

### Process creation and executable mappings

At a high level:

```text
loader validates NPDM and executable metadata
        │
        ▼
kernel creates process address-space policy
        │
        ▼
kernel allocates or references code/data pages
        │
        ▼
code and data are mapped with initial permissions
        │
        ▼
semantic blocks are published as Code/CodeData
        │
        ▼
main stack and TLS are prepared
        │
        ▼
initial thread starts at the selected entry point
```

Additional NSO modules are normally placed in the process address space and
exposed with module-code states. Guest `rtld` uses those mappings to relocate
and initialize the title.

### Growing or shrinking the heap

`svcSetHeapSize` changes the committed prefix of a fixed randomized heap
region. It does not move existing heap addresses.

```text
guest requests new heap size
        │
        ▼
validate 2 MiB size alignment and maximum
        │
        ▼
validate heap region and resource limit
        │
        ├── grow
        │       │
        │       ▼
        │   allocate zeroed physical pages
        │       │
        │       ▼
        │   map as read/write Normal memory
        │
        └── shrink
                │
                ▼
            unmap trailing pages
                │
                ▼
            release last physical references
        │
        ▼
return the fixed heap base
```

The public SVC contract and known validation rules are documented by
[Switchbrew](https://switchbrew.org/wiki/SVC#SetHeapSize) and pinned libnx
[`svc.h`](https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/kernel/svc.h).

### Mapping memory as an alias

`svcMapMemory` creates another virtual view of an existing range, historically
in an alias region and on later firmware primarily for stacks. The operation
also changes the source accessibility/state according to Horizon policy. Its
inverse validates the original source/destination relationship.

A correct implementation must retain the same physical pages:

```text
validate source semantic state
        │
        ▼
validate destination is a permitted free range
        │
        ▼
open references to source physical pages
        │
        ▼
publish destination translation
        │
        ▼
update source and destination semantic blocks
        │
        ▼
invalidate affected CPU translations
```

### Shared memory

A shared-memory object owns or references physical backing independently of
one process mapping. Each process maps that backing at a selected virtual
address with permissions constrained by the object.

```text
shared-memory object
        │
        ▼
physical page group
        │
        ├── process A mapping
        │
        └── process B mapping
```

Unmapping one process must not destroy the object or the other mapping while
references remain.

### Transfer memory

Transfer memory temporarily lends a range under a declared permission policy.
The kernel validates the source range, locks or changes its state, creates an
object referring to its physical pages, and later maps that object into another
address space. Ownership and restoration rules differ from ordinary shared
memory.

### IPC mappings

HIPC descriptors can cause a client buffer to be mapped into a server process
for the duration of a request. Horizon selects an IPC memory state based on
descriptor flags and whether device or non-secure access is allowed. It tracks
IPC reference counts and restores the original state when the temporary
mapping closes.

This is not equivalent to copying a message payload. See
[IPC, HIPC and CMIF](IPC,%20HIPC%20and%20CMIF.md).

### Thread stacks and TLS

A stack is ordinary page-backed memory with stack-specific placement and
state. Guard holes around it remain unmapped so overflow faults instead of
silently corrupting a neighboring object.

The thread-local region contains kernel/user thread-local state and the normal
HIPC command buffer. The architectural thread-pointer register points at the
thread's TLS base.

### Device and GPU mappings

CPU virtual mappings and device virtual mappings are different:

```text
CPU virtual address
        │
        ▼
physical backing
        ▲
        │
GPU virtual address
```

The GPU mapping may use the same physical pages while having a separate virtual
address, lifetime, permission, and synchronization mechanism. `Uncached` and
device-sharing state affect whether Horizon permits a mapping, but an emulator
must also model guest-visible synchronization and ownership. A CPU pointer is
not automatically a GPU pointer.

## Atomicity and failure ordering

Mapping changes are multi-structure transactions. A kernel cannot safely
publish half the page-table entries and then discover that it lacks one
metadata block.

A robust operation follows this vertical order:

```text
validate complete request
        │
        ▼
reserve all fallible metadata
        │
        ▼
allocate or open all physical pages
        │
        ▼
prepare page-table update
        │
        ▼
publish translations and semantic metadata
        │
        ▼
perform required TLB/cache maintenance
        │
        ▼
commit accounting
```

Before the publication point, an error releases private resources and leaves
the observable address space unchanged. Cross-page stores require the same
principle at a smaller scale: validate every touched page before writing the
first byte.

## Nixe's memory model

### Layer boundaries

Nixe deliberately keeps generic CPU memory contracts separate from
Switch-specific SVC policy:

```text
loader and prepared modules
        │
        ▼
runtime process builder
        │
        ▼
Horizon process layout and SVC validation
        │
        ▼
ProcessMemory
        │
        ▼
CpuMemory
        │
        ▼
InstructionMemory
        │
        ▼
ExecutionMemory storage
```

The primary source files are:

| Responsibility | Source |
| -------------- | ------ |
| Device-neutral identities and generations | [`crates/memory/src/lib.rs`](../../crates/memory/src/lib.rs) |
| Device-neutral permissions and access declarations | [`crates/memory/src/access.rs`](../../crates/memory/src/access.rs) |
| CPU/device visibility contracts | [`crates/memory/src/visibility.rs`](../../crates/memory/src/visibility.rs) |
| Retained canonical RAM and allocations | [`crates/memory/src/backing.rs`](../../crates/memory/src/backing.rs) |
| Checked pointer-free backing ranges | [`crates/memory/src/range.rs`](../../crates/memory/src/range.rs) |
| Public memory-module facade and re-exports | [`crates/cpu/src/memory/mod.rs`](../../crates/cpu/src/memory/mod.rs) |
| Portable traits, values, and faults | [`crates/cpu/src/memory/contracts.rs`](../../crates/cpu/src/memory/contracts.rs) |
| Small implementation helpers shared by both backends | [`crates/cpu/src/memory/common.rs`](../../crates/cpu/src/memory/common.rs) |
| Deterministic reference backend | [`crates/cpu/src/memory/synthetic.rs`](../../crates/cpu/src/memory/synthetic.rs) |
| Production storage and hot paths | [`crates/cpu/src/memory/execution.rs`](../../crates/cpu/src/memory/execution.rs) |
| Module installation transaction | [`crates/runtime/src/module_memory.rs`](../../crates/runtime/src/module_memory.rs) |
| Address-space regions and initial process mappings | [`crates/runtime/src/process_builder.rs`](../../crates/runtime/src/process_builder.rs) |
| Horizon SVC policy and result translation | [`crates/horizon/src/svc_dispatch.rs`](../../crates/horizon/src/svc_dispatch.rs) |

For a first code review, read them in this order:

```text
memory/mod.rs: public facade and re-exports
        │
        ▼
memory/contracts.rs: public contracts and shared semantic types
        │
        ▼
memory/synthetic.rs: deterministic reference implementation
        │
        ▼
memory/execution.rs: storage invariants and page resolution
        │
        ▼
memory/execution.rs: fetch, read, write, and management operations
        │
        ▼
module_memory.rs: atomic executable installation
        │
        ▼
process_builder.rs: initial virtual layout
        │
        ▼
svc_dispatch.rs: Horizon-visible validation and state numbers
```

This order separates what callers may rely on from how production storage
implements it. The differential tests in `memory/synthetic.rs`, together with
the representation-specific tests in `memory/execution.rs`, form an executable
specification for the deterministic and production backends.

### Portable identity types

The device-neutral `nixe-memory` crate owns domain-specific integer wrappers:

- `GuestVirtualAddress`: an address in a guest virtual space;
- `AddressSpaceId`: the identity of that virtual space;
- `GuestPhysicalPageId`: a store-local physical page shared by aliases;
- `BackingStoreId` and `CanonicalPageId`: an unambiguous cross-device page
  identity;
- `ContentGeneration`: a version of physical page bytes; and
- `MappingGeneration`: a version of one virtual mapping and its access
  metadata.

They are deliberately not host pointers. Checked arithmetic is the default for
guest addresses. CPU frontends and engines import these identities directly
from `nixe-memory`.

### Retained canonical ranges

Production RAM bytes and their content generation live in
`CanonicalBackingPage`. Clones retain a page independently of CPU virtual
mappings, sparse page-table slots, and the `ExecutionMemory` which created it.
Removing the final CPU alias releases process ownership, but an `nvmap`
allocation or in-flight device range which already retained the page remains
valid until that reference is dropped.

`CanonicalRangeTranslator` is the neutral translation contract between CPU
virtual memory and canonical backing. A complete translation validates the
requested CPU virtual range before returning ordered
`CanonicalBackingSegment` values. Each segment contains:

- `CanonicalPageId`;
- page-relative byte offset and size;
- the exact CPU mapping permissions;
- the observed content generation; and
- the observed mapping generation.

The range contains no CPU virtual address as backing identity, physical-slot
index, borrowed byte slice, raw pointer, or host graphics object. Translation
fails at the first unmapped, permission-incompatible, MMIO, overflowing, or
internally inconsistent byte. Its result retains canonical storage rather than
retaining a process or CPU mapping.

`NVMAP_IOC_ALLOC` now uses this adapter before publishing allocation state.
The CPU address remains only the guest ABI view origin needed by the existing
software-framebuffer path; the retained canonical range is the allocation's
cross-device identity and lifetime anchor. A failed translation does not
partially initialize the `nvmap` allocation and is a typed emulator-side stop,
not a fabricated NVIDIA driver result.

Runtime shared-memory objects now also own `CanonicalAllocation` storage and
can expose a retained neutral range. Transfer-memory handles retain the
validated canonical range denoted by their creation request instead of
retaining only a CPU virtual address. The current Horizon shared-memory mapping
path still copies object storage into process pages; replacing that known
coherency limitation with direct canonical aliases is separate kernel mapping
work. The object no longer needs a second architectural storage model in order
to reach that future path.

### Non-CPU access and visibility

`DeviceAccessDeclaration` describes a read, write, or read/write operation
without naming Maxwell, a host graphics API, or a backend resource. Every
declaration identifies the non-CPU consumer and records two distinct ordering
concepts:

- `device_visible_at` is the point before which canonical inputs and
  unaffected bytes must be available to the device; and
- `cpu_visible_at` is required for a write and is the point after which the
  device's output may be reconciled into canonical CPU-visible storage.

`DeviceVisibilityPoint` is an opaque value ordered by the device owner. It does
not itself claim guest-fence, host-queue, or presentation completion. Later
submission and synchronization layers must connect those separate events.

Every `CanonicalBackingPage` owns one visibility authority shared by all CPU
aliases and retained ranges:

```text
Clean
  │
  ├── CPU write ────────────────────────────────► CpuNewer
  │                                                │
  │                          make device visible ──┘
  │
  └── completed declared device write ──────────► GpuNewer
                                                   │
                              CPU read or write ───┘
                                   make CPU visible

incompatible unsynchronized owners ─────────────► Conflicting
failed or malformed transition ─────────────────► Invalid
```

The first implementation deliberately tracks complete canonical pages. A
logical range may start or end inside a page, but its transition includes the
whole page so overlapping aliases cannot silently retain different ownership.
Dirty byte ranges, image subresources, and version vectors are later
optimizations, not changes to this contract.

`VisibilityCoordinator` is the only boundary which performs host residency and
visibility work. Before device access it receives complete canonical bytes.
Before CPU access to `GpuNewer` storage it waits or performs the required host
operations and returns one complete newest page. A discrete backend can
implement these methods with staging uploads and downloads; coherent
unified-memory storage can make data movement a no-op while still enforcing
ordering and lifetime. The interface contains no `wgpu`, Vulkan, Metal,
Direct3D, queue, command-buffer, or host-pointer type.

Production instruction fetches, CPU reads, and CPU writes call canonical page
accessors. `Clean` and `CpuNewer` take the ordinary path. `GpuNewer` invokes the
coordinator before returning or modifying bytes, advances the content
generation when device output becomes canonical, and only then continues the
CPU operation. Consequently host API work remains outside interpreter and
future generated-code semantics. Coordinator failure, incorrect writeback
size, generation exhaustion, a concurrent transition, or conflicting owners
is a typed emulator-side memory failure; stale bytes are never returned.

A CPU read copies bytes while holding the same page-state lock under which it
checks visibility authority. If a device publishes newer ownership after the
slow path returns but before the read reacquires that lock, the read retries
the transition instead of exposing the previous canonical contents.

### Canonical-memory acceptance evidence

The T2 acceptance matrix is intentionally layered rather than concentrated in
one end-to-end test:

| Contract | Permanent evidence |
| -------- | ------------------ |
| CPU aliases share identity and bytes | `data_aliases_share_one_physical_page_identity_and_contents` and `mapping_and_content_generations_change_independently_in_both_backends` in `crates/cpu/src/memory/synthetic.rs` |
| Checked page crossings and exact permissions | `canonical_translation_retains_checked_page_spanning_segments` in `crates/cpu/src/memory/execution.rs` |
| Remaps do not masquerade as byte writes | `mapping_and_content_generations_change_independently_in_both_backends` |
| Content and mapping generations never wrap | `generation_domains_are_distinct_and_never_wrap`, `exhausted_generations_fail_without_publishing_bytes_or_mapping_changes`, and `exhausted_device_writeback_generation_invalidates_without_downloading` |
| Concurrent CPU/device observations cannot accept stale state | `concurrent_cpu_write_rejects_an_in_flight_device_snapshot` |
| CPU unmap, allocation release, and process-memory teardown retain live device ranges | `retained_ranges_outlive_their_allocation_owner` and `retained_translation_survives_cpu_unmap_and_memory_teardown` |
| Neutral GPU code accesses validated canonical ranges without CPU context | `gpu_contract_accesses_a_page_spanning_range_without_cpu_context` in `crates/gpu/tests/canonical_memory_contract.rs` |
| Dependency direction excludes CPU internals and host APIs | the `dependency_boundaries` tests in `nixe-memory`, `nixe-cpu`, `nixe-gpu`, `nixe-runtime`, and `nixe-horizon` |
| Interpreter behavior remains equivalent | `directed_sequences_compare_state_vectors_memory_pc_flags_and_exceptions`, `bounded_generated_a64_sequences_match_the_reference_evaluator`, and the CPU unit suite |
| Existing software-framebuffer behavior remains accepted | `libnx_hello_world_publishes_a_software_frame` in `crates/horizon/tests/homebrew_acceptance.rs` |

The GPU acceptance test starts from a `CanonicalAllocation`, obtains a
page-spanning `CanonicalBackingRange`, and performs a declared device access
using only `nixe-memory` contracts. It has no `ExceptionProcessContext`, CPU
virtual address, borrowed host slice, or raw pointer. The `nixe-gpu`
dependency-boundary test prevents adding CPU, runtime, Horizon, Maxwell, video,
window-system, or concrete host-backend dependencies to make that test pass.

### InstructionMemory

`InstructionMemory` is the read-only frontend used by decode and translation.
It provides:

- the code-page span containing an address;
- aligned 16-bit T32 fetches;
- aligned 32-bit A64/A32 fetches; and
- 32-bit T32 assembly from two halfword fetches.

Every fetch returns canonical little-endian instruction bits plus
`CodeDependencies`. A dependency contains the physical page ID, content
generation, and mapping generation observed while reading.

For a T32 instruction crossing a page:

```text
fetch first halfword
        │
        ▼
record first physical dependency
        │
        ▼
checked address + 2
        │
        ▼
fetch second halfword
        │
        ▼
record or merge second dependency
        │
        ▼
assemble canonical T32 encoding
```

The second halfword can fault independently at its exact address.

### CpuMemory

`CpuMemory` is the interpreter-facing data contract. Each access explicitly
describes:

- byte width from 1 through 16;
- required alignment;
- memory ordering;
- normal, atomic, exclusive, or volatile class; and
- read or write direction.

Results distinguish RAM from device accesses. Faults distinguish unmapped
memory, permissions, alignment, overflow, mixed RAM/device spans, device
errors, and deterministic injected faults.

The contract also exposes local exclusive reservations. Nixe identifies a
reservation by:

```text
physical page ID
        │
        ▼
page-relative byte offset
        │
        ▼
access size
        │
        ▼
content generation
```

An intervening write through any alias changes the generation and causes a
later store-exclusive to fail.

### ProcessMemory

`ProcessMemory` is the runtime mutation boundary. It currently provides:

- atomic resize of a zeroed mapping;
- atomic permission replacement over a complete mapped range; and
- atomic attribute updates.

Horizon policy is intentionally outside this trait. For example,
`set_permissions` knows how to replace permissions, while the Horizon SVC
dispatcher decides which `MemoryMappingPurpose` is eligible for reprotection.

### MemoryMappingPurpose

Nixe currently retains a smaller semantic-purpose enumeration than Horizon's
complete memory-state bitmask:

- normal;
- initial code static or mutable;
- module code static or mutable;
- thread-local;
- heap; and
- shared memory.

The purpose is used by `QueryMemory` translation and SVC policy. It is not the
physical backing type.

The name `MemoryMappingPurpose::Normal` is backend-neutral and must not be
confused with Horizon's `Normal` memory state. The current SVC conversion
reports it as Horizon `Static` (type 2). `MemoryMappingPurpose::Heap` is the
value reported as Horizon `Normal` / libnx `Heap` (type 5).

### SyntheticMemory

`SyntheticMemory` is the deterministic reference backend used by tests. It
uses:

```text
RefCell<SyntheticMemoryInner>
        │
        ├── BTreeMap<(address space, virtual page), mapping>
        ├── BTreeMap<physical page ID, physical page>
        ├── injected instruction faults
        ├── injected data faults
        └── injected installation failure
```

Its purpose is transparent behavior and controllable failures. It can inject a
failure during preflight, allocation, initialization, or publication and prove
that the transaction rolls back. It is not the production hot-path backend.

### ExecutionMemory

`ExecutionMemory` owns independent production storage. It never contains or
delegates to `SyntheticMemory`.

Its representation is:

```text
ExecutionMemory
        │
        ▼
RefCell<ExecutionMemoryInner>
        │
        ├── sparse virtual page table
        │       │
        │       ▼
        │   BTreeMap<(address space, 2 MiB leaf number), leaf>
        │       │
        │       ▼
        │   512 directly indexed virtual-page entries
        │
        ├── Vec<Option<physical page>> stable slots
        ├── free-slot list
        ├── physical ID → slot management index
        └── next physical-page ID
```

One leaf covers:

```text
512 pages × 4096 bytes = 2 MiB
```

Only populated 2 MiB regions allocate leaves. A single mapping near address
zero and another near the top of a 64-bit value allocate two leaves, not a
flat array proportional to the address-space width.

This is a software lookup structure, not an emulation of Arm translation-table
descriptor levels. Its shape was selected for the interpreter's workload:
large sparse guest spaces with many neighboring code/data pages.

### Why ExecutionMemory still has a RefCell

The existing `CpuMemory` contract performs guest writes and MMIO callbacks
through `&self`. Safe Rust therefore requires interior mutability somewhere.
`ExecutionMemory` keeps one `RefCell` around its complete single-threaded state
rather than introducing raw pointers or undocumented `unsafe`.

The production optimization removed the expensive part of the former design:
a fetch no longer performs one associative lookup for the virtual mapping and
a second associative lookup for the physical page. After resolving the sparse
leaf, the mapping contains a directly indexable physical slot.

Changing this ownership model later would require changing the memory traits
and interpreter context to carry exclusive mutable access. Replacing the
`RefCell` with `UnsafeCell` without changing the ownership contract would only
remove a checked invariant and is not an acceptable optimization.

### Virtual entries

Each production virtual-page entry records:

- physical page ID;
- physical slot index;
- mapping generation;
- permissions;
- mapping purpose; and
- attributes.

The physical ID is the observable alias identity. The slot is the direct host
storage index. Repeating both values in multiple virtual entries creates an
alias.

Core invariant:

```text
published virtual entry
        │
        ▼
physical slot is occupied
        │
        ▼
entry physical ID maps back to that slot
```

A free slot appears in neither a mapping nor the ID index.

### Physical pages

A physical slot contains either:

```text
RAM
        │
        └── retained CanonicalBackingPage
                │
                ├── stable CanonicalPageId
                ├── optional 4096-byte allocation
                └── typed content generation

or

MMIO
        │
        └── device callback object
```

The physical slot is only the production CPU lookup strategy. Removing or
reusing it cannot invalidate a retained canonical page.

`None` RAM bytes represent a lazily materialized all-zero page. Reads return
zero without allocating. The first write allocates the 4 KiB backing and then
updates it.

Released slots are kept in a free list and reused. The slot vector therefore
does not grow without bound when a heap repeatedly grows and shrinks.

The physical ID-to-slot `BTreeMap` is used by management operations such as
publishing an alias. It is not consulted by normal instruction or data
accesses, because a resolved virtual entry already carries its slot.

### The instruction-fetch path

The normal A64/A32 `fetch32` path is:

```text
check 4-byte alignment
        │
        ▼
borrow ExecutionMemory state
        │
        ▼
derive virtual page and 2 MiB leaf number
        │
        ▼
one BTreeMap leaf lookup
        │
        ▼
index one of 512 virtual entries
        │
        ▼
check execute permission
        │
        ▼
index the physical slot vector
        │
        ▼
require RAM backing
        │
        ▼
read four little-endian bytes
        │
        ▼
return bits + physical ID + content and mapping generations
```

Aligned 2-byte and 4-byte fetches cannot cross a 4 KiB page because both widths
divide the page size. Cross-page T32 is handled by the two-fetch default
described earlier.

MMIO is never executable in the current model. An executable mapping that
resolves to an MMIO page produces a typed fetch fault.

### The data-read path

For an in-page RAM read:

```text
validate required alignment
        │
        ▼
validate end-address arithmetic
        │
        ▼
resolve virtual entry once
        │
        ▼
check read permission
        │
        ▼
classify directly indexed physical slot as RAM
        │
        ▼
copy requested bytes or synthesize zeros
        │
        ▼
construct typed little-endian MemoryValue
```

For MMIO, the slow path invokes the callback with the page-relative offset and
the complete access descriptor. The callback's returned width is validated.

### The data-write path

For an in-page RAM write:

```text
validate value width
        │
        ▼
validate alignment and address arithmetic
        │
        ▼
resolve virtual entry once
        │
        ▼
check write permission
        │
        ▼
materialize zero page if necessary
        │
        ▼
write little-endian bytes
        │
        ▼
advance content generation
```

The content generation advances for every successful RAM write, whether or not
any alias is executable. It never wraps: exhaustion produces a typed memory
failure before bytes are committed.

### Cross-page data accesses

Architectural data accesses can be unaligned and may touch two pages. Nixe
first resolves every touched page and checks region type and permission.

```text
validate first page
        │
        ▼
validate second page
        │
        ├── fault
        │       │
        │       ▼
        │   commit no bytes
        │
        └── success
                │
                ▼
            copy first fragment
                │
                ▼
            copy second fragment
                │
                ▼
            advance each distinct content generation once
```

An access cannot span RAM and MMIO. Device accesses spanning pages remain on a
precise slow/error path.

### Code invalidation

Fetched or translated code depends on physical pages, not virtual addresses:

```text
translated block
        │
        ▼
CodeDependency(page P, content generation C, mapping generation M)
        │
        ▼
write through any alias of P
        │
        ▼
content generation becomes C + 1
        │
        ▼
old dependency no longer matches
```

Changing mapping permissions or attributes advances the affected mapping
generation without changing the content generation. Code dependencies retain
both values, so a block compiled under an old executable view becomes stale
without falsely recording a byte write.

The translator already attaches these dependencies to IR blocks. The current
reference executor does not yet maintain a native-code cache, so no JIT block
is being evicted today. A future cache must validate both generations and the
virtual mapping from the block's guest location.

### Atomic page installation

The runtime installs a prepared module through `ModuleMemoryBackend`. It first
splits validated executable mappings into exact 4 KiB requests. Production
installation then:

```text
validate all addresses, lengths, collisions, and duplicates
        │
        ▼
choose all physical identities
        │
        ▼
allocate and initialize private RAM pages
        │
        ▼
reserve fallible slot capacity
        │
        ▼
publish every physical slot and virtual entry
        │
        ▼
commit next physical identity
```

No returned error occurs after publication starts. Preflight or resource
failure leaves existing mappings and page counts unchanged.

`SyntheticMemory` additionally injects failures at individual stages to test
the transaction contract.

### Resize and protection

`resize_zeroed_mapping` validates the complete old and new range before
mutation. Growth allocates lazy-zero physical pages. Shrink removes virtual
entries and frees a physical slot only when no alias still references it.

`set_permissions` and `set_attributes` similarly preflight every page before
updating any page. Permission-locked mappings reject incompatible changes.

### QueryMemory in Nixe

`query_memory` coalesces adjacent pages when these observable properties match:

- RAM versus device region;
- permissions;
- mapping purpose; and
- attributes.

For a hole it finds the previous and next mapping boundaries and returns a
free interval. The Horizon SVC layer converts `MemoryMappingPurpose` into the
public numeric state.

This is intentionally an interval/diagnostic slow path. It scans or walks
mapping metadata rather than complicating the instruction-fetch structure with
a second interval index.

## Process construction in Nixe

`ProcessBuilder` currently performs:

```text
read NPDM-derived execution and address-space policy
        │
        ▼
prepare and place executable modules
        │
        ▼
derive Horizon virtual-region geometry
        │
        ▼
create ExecutionMemory
        │
        ▼
atomically install module pages
        │
        ▼
assign initial/module code purposes
        │
        ▼
install main stack and TLS pages
        │
        ▼
install homebrew ABI pages when applicable
        │
        ▼
check physical-memory accounting
        │
        ▼
initialize PC, SP, thread pointer, and ABI registers
```

The address-space types and fixed legacy layouts follow the public Horizon
model. The 39-bit layout uses the known region sizes and deterministic
no-ASLR ordering around the selected code range. Nixe does not yet reproduce
the complete firmware-specific ASLR algorithm.

The main stack is currently installed directly as zeroed read/write pages at
the bottom of the reserved stack region, with explicit unmapped guard gaps
around TLS and homebrew resources. Real Horizon can create and place stacks
through more complete mapping machinery. Because Nixe does not yet have a
dedicated stack purpose, the current main-stack pages retain
`MemoryMappingPurpose::Normal` and are reported by `QueryMemory` as `Static`,
not as Horizon `Stack`.

## Horizon SVC integration in Nixe

Implemented memory-related behavior currently includes:

- `SetHeapSize`;
- `SetMemoryPermission`;
- `SetMemoryAttribute`;
- `QueryMemory`;
- temporary shared-memory map/unmap support; and
- creation of a transfer-memory handle describing an existing range.

The dispatcher validates Horizon-specific rules before calling the portable
memory backend. For example, heap growth checks 2 MiB alignment, heap-region
size, the 4 GiB limit, and process memory capacity.

`QueryMemory` currently maps purposes to the public types:

| Nixe purpose | Horizon type |
| ------------ | ------------ |
| Normal | Static (type 2) |
| CodeStatic | Code |
| CodeMutable | CodeData |
| ModuleCodeStatic | AliasCode |
| ModuleCodeMutable | AliasCodeData |
| ThreadLocal | ThreadLocal |
| Heap | Normal / Heap (type 5) |
| SharedMemory | Shared |

The exact conversion is in
[`query_memory`](../../crates/horizon/src/svc_dispatch.rs).

## Current limitations

The following are deliberate descriptions of current behavior, not promises
that the complete Horizon model is already implemented.

### Semantic state is incomplete

`MemoryMappingPurpose` is smaller than Horizon's `MemoryState`. Nixe does not
yet retain all state capability flags, IPC/device reference counts, original
permissions, merge restrictions, or every memory type.

As a result, extending SVC coverage will require enriching semantic metadata;
it should not encode more Horizon policy into ad-hoc permission checks.

### MapMemory and UnmapMemory are not implemented

The SVC registry names `MapMemory` and `UnmapMemory`, but the current dispatcher
does not implement their semantics. Proper support must create real aliases,
track the source/destination relationship, change source state as Horizon
requires, and restore it atomically on unmap.

### SharedMemory currently copies

The temporary shared-memory path currently creates private zeroed process
pages and copies the object bytes into them. It is not yet a continuously
shared physical-page alias. Updates made through the guest mapping therefore
do not automatically update every other view of the object.

This should eventually be replaced by object-owned physical pages mapped into
each process, not by additional copies.

### TransferMemory is partial

Nixe can validate a source range and create a transfer-memory handle, but does
not yet implement the complete lock, state transition, cross-process mapping,
unmapping, and restoration lifecycle.

### IPC mappings are partial

The HIPC codec validates and copies guest buffers for implemented services. It
does not yet model every temporary IPC mapping state, reference counter, and
device/security variant used by Horizon.

### MMIO exists at the CPU boundary, not the full platform

`ExecutionMemory` can own MMIO callback pages, and the data paths preserve
device access count and order. Normal process construction does not yet install
the complete Switch physical IO map or all privileged mapping SVCs.

### No literal Arm MMU or architectural TLB

The interpreter resolves guest addresses through software structures. It does
not execute a hardware-style page-table walk, maintain guest-visible
architectural TLB entries, or emulate access/dirty bits. This is acceptable
while guest-visible faults, permissions, aliases, attributes, and SVC results
remain correct.

A future host-side software TLB would be an implementation cache, not an
emulated architectural TLB.

### Memory ordering and caches are not a complete hardware model

The access descriptor retains ordering and access class, and exclusive
reservations use physical identity and generation. The current process model
executes one guest thread on one vCPU at a time. This serialized behavior is
retained during scheduler migration as a permanent deterministic policy, which
models all configured vCPUs while permitting only one guest slice to execute at
once. Parallel policy will instead use at most one long-lived host worker per
active vCPU; guest threads remain scheduler-owned runtime objects rather than
dedicated host threads. Canonical backing records conservative CPU/device
visibility authority and can invoke an injected reconciliation slow path. Nixe
does not reproduce the complete multicore Arm memory model, cache hierarchy,
device coherence, or all cache-maintenance effects.

## Planned dynamic-recompiler memory path

This section records a design direction, not implemented behavior or a stable
backend ABI. Here, *dynamic recompiler* means Nixe translating guest Arm code
to host code. It is distinct from Horizon's `GeneratedCode` and `CodeOut`
memory states, which describe code generated by a guest application.

### Semantic authority remains in process memory

A JIT changes how frequent accesses reach RAM; it must not create a second
memory model. The existing guest-visible rules remain authoritative:

- virtual address-space identity and bounds;
- mappings, permissions, purposes, and attributes;
- physical identity shared by aliases;
- RAM versus MMIO behavior;
- precise faults and all-or-nothing cross-page writes;
- exclusive reservations and required ordering; and
- physical code generations and mapping transitions.

Host pointers are derived acceleration data. They must never become Horizon
policy, escape in guest-visible diagnostics, or allow a host mapping to grant
an access that `CpuMemory` would reject.

### Staged implementation

The first correct backend can compile loads and stores as calls to typed
helpers implementing the existing `CpuMemory` semantics:

```text
compiled guest operation
        │
        ▼
JIT helper ABI
        │
        ▼
CpuMemory / ExecutionMemory
        │
        ▼
value, precise fault, or dispatcher exit
```

This isolates instruction lowering, register state, exits, and exception
delivery before memory access is optimized. It is expected to be slower than
an inlined path but provides the reference behavior for differential tests.

The next optimization should be a small software TLB owned by each vCPU. A
conceptual entry contains:

```text
address-space ID + guest-page tag
        │
        ├── host RAM base
        ├── read/write permission bits
        ├── RAM/MMIO and fast-path flags
        └── mapping/code metadata or epochs
```

On a hit, generated code may perform an in-page ordinary RAM access directly
after checking the tag, permissions, width, and page boundary. It must produce
the guest's little-endian result explicitly unless the backend is restricted
to a compatible host. A miss or unsupported entry exits to the semantic slow
path.

The slow path remains responsible for:

- unmapped and permission faults;
- accesses crossing a page boundary;
- MMIO and mixed RAM/device rejection;
- operations needing nontrivial atomic, exclusive, volatile, or ordering
  behavior;
- mapping changes and uncommon attributes; and
- any diagnostic mode that cannot safely use the normal direct path.

The exact split can change after profiling. A semantic case must not migrate
to the fast path until it has equivalent fault, byte-order, alias, and
generation behavior.

### Mapping and code-cache invalidation

Every mapping mutation must invalidate derived lookup state. Mapping,
unmapping, changing permissions, or replacing backing must either evict the
affected software-TLB entries or advance an epoch checked by those entries.
Safepoints provide the boundary at which the runtime can publish such changes
without generated code retaining stale pointers.

A native block cache needs two related checks:

```text
guest block location + address-space identity
        │
        ├── current virtual mapping still resolves to the expected page
        └── every CodeDependency still has the expected content and mapping
            generations
```

Writes through any alias must remain visible to every other alias. A direct
JIT store must also perform the generation update required by code
dependencies and exclusive reservations, or use a proven equivalent
invalidation scheme. Optimizing ordinary stores must not make self-modifying
code or writable code aliases stale.

### Ownership and concurrency

`ExecutionMemory` currently uses a `RefCell` to enforce its single-active-vCPU
ownership contract. Generated code cannot simply hold a Rust borrow while
calling arbitrary runtime helpers, and replacing the `RefCell` with unchecked
interior mutability would not define a safe ABI.

A JIT fast path will therefore need an explicit internal ownership and
lifetime design for stable RAM pointers and metadata. Mapping mutation must be
excluded while compiled code can use an affected pointer, or coordinated
through exits and safepoints. Before Nixe enables its planned parallel policy,
generations, exclusives, TLB invalidation, memory ordering, mapping mutation,
and backing lifetimes require explicit thread-safe synchronization. The
deterministic policy remains supported after that transition and uses the same
canonical backing and vCPU model.

Every interpreter, JIT, or optional platform NCE engine uses this memory model
as semantic authority. An NCE domain may mirror or map canonical pages only if
it observes mapping and invalidation epochs and reconciles dirty state at its
normalized run-slice exits; it cannot introduce an independent memory model.

### Fastmem remains an optional later backend

Fastmem could reserve a large sparse host virtual range and translate an
ordinary guest address with an addition to a host base. A modern Switch
process can expose a 39-bit virtual domain, so reservation need not imply
physical commitment.

This approach is not selected yet. It must first solve, portably:

- multiple guest address spaces;
- two virtual aliases of the same physical backing;
- host protection and signal/exception recovery;
- MMIO and cross-page accesses;
- mapping replacement without stale native blocks; and
- platform-specific virtual-memory APIs.

A software TLB is the preferred initial optimized path because it fits the
current sparse mappings and keeps faults explicit. Fastmem should be adopted
only if measurements justify its additional platform and recovery complexity.

## How to change the implementation safely

### Decide which layer owns the change

Use this decision flow:

```text
is it an Arm load/store/fetch rule?
        │
        ├── yes → CpuMemory or InstructionMemory
        │
        └── no
                │
                ▼
is it Horizon SVC state or result policy?
        │
        ├── yes → horizon svc_dispatch
        │
        └── no
                │
                ▼
is it process placement or initial ABI state?
        │
        ├── yes → runtime process_builder
        │
        └── no
                │
                ▼
is it storage, alias identity, or lookup performance?
        │
        └── yes → memory backend
```

Do not put Horizon state numbers into the generic CPU interpreter, and do not
put host pointers into the Horizon policy layer.

### Preserve these invariants

Any storage change must preserve:

1. every virtual mapping references an occupied physical slot;
2. every mapping's physical ID agrees with the management index;
3. aliases share one slot and generation;
4. a free slot has no mapping or physical-ID entry;
5. a cross-page write validates every page before committing bytes;
6. an atomic installation publishes all pages or none;
7. permission and attribute updates validate the complete range first;
8. instruction dependencies report the exact physical pages read;
9. writes through any alias invalidate physical code dependencies; and
10. errors retain their first failing guest address and typed reason.

### Keep fast and slow paths semantically joined

The in-page RAM path may be optimized, but it must implement the same
permissions, byte order, generation, and fault rules as the slow path.

```text
common validation and resolved mapping
        │
        ├── in-page RAM → direct fast path
        │
        └── MMIO/cross-page/error → precise slow path
```

A cache that is faster only because some mapping mutation forgets to
invalidate it is not a valid optimization.

### Add differential tests

For a new operation, run the same observable sequence against
`SyntheticMemory` and `ExecutionMemory` where both backends support it.
Compare:

- returned values;
- exact faults;
- mapping queries;
- physical IDs and generations;
- alias visibility;
- page counts; and
- rollback after failure.

Existing tests cover RAM, permissions, cross-page accesses, aliases, lazy zero,
MMIO, resize, installation, exclusives, and code writes through aliases.

### Profile after correctness

The production split reduced the interpreter's `fetch32` share from roughly
16.7% to 8.5% in the measured workload and removed `SyntheticMemory` from the
production profile. Throughput rose from approximately 15 million to nearly
20 million guest instructions per second on that host.

Those numbers are workload- and machine-specific. They justify the chosen
lookup structure but are not correctness tests or performance guarantees.

## References

Public Horizon information is based primarily on community reverse engineering
and open-source compatible implementations.

- [Atmosphère/Mesosphere `KPageTableBase`, pinned revision](https://github.com/Atmosphere-NX/Atmosphere/blob/e468f59c9d369b8ebbffa040f4c9fc201b9f75a8/libraries/libmesosphere/include/mesosphere/kern_k_page_table_base.hpp)
- [Atmosphère/Mesosphere `KMemoryManager`, pinned revision](https://github.com/Atmosphere-NX/Atmosphere/blob/e468f59c9d369b8ebbffa040f4c9fc201b9f75a8/libraries/libmesosphere/include/mesosphere/kern_k_memory_manager.hpp)
- [Atmosphère/Mesosphere `KMemoryBlockManager`, pinned revision](https://github.com/Atmosphere-NX/Atmosphere/blob/e468f59c9d369b8ebbffa040f4c9fc201b9f75a8/libraries/libmesosphere/include/mesosphere/kern_k_memory_block_manager.hpp)
- [Atmosphère/Mesosphere `KMemoryBlock` and memory states, pinned revision](https://github.com/Atmosphere-NX/Atmosphere/blob/e468f59c9d369b8ebbffa040f4c9fc201b9f75a8/libraries/libmesosphere/include/mesosphere/kern_k_memory_block.hpp)
- [Atmosphère/Mesosphere `KPageGroup`, pinned revision](https://github.com/Atmosphere-NX/Atmosphere/blob/e468f59c9d369b8ebbffa040f4c9fc201b9f75a8/libraries/libmesosphere/include/mesosphere/kern_k_page_group.hpp)
- [Atmosphère `KAddressSpaceInfo`, pinned revision](https://github.com/Atmosphere-NX/Atmosphere/blob/e468f59c9d369b8ebbffa040f4c9fc201b9f75a8/libraries/libmesosphere/include/mesosphere/kern_k_address_space_info.hpp)
- [Atmosphère public SVC types, pinned revision](https://github.com/Atmosphere-NX/Atmosphere/blob/e468f59c9d369b8ebbffa040f4c9fc201b9f75a8/libraries/libvapours/include/vapours/svc/svc_types_common.hpp)
- [libnx `svc.h`, pinned revision](https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/include/switch/kernel/svc.h)
- [Switchbrew: Memory layout](https://switchbrew.org/wiki/Memory_layout)
- [Switchbrew: SVC](https://switchbrew.org/wiki/SVC)
- [Switchbrew: NPDM](https://switchbrew.org/wiki/NPDM)
- [Arm: Memory management](https://developer.arm.com/-/media/Arm%20Developer%20Community/PDF/Learn%20the%20Architecture/LearnTheArchitecture-MemoryManagement-101811_0100_00_en.pdf?revision=1fdc3375-d81c-4457-b786-04fb98557de0)
- [Arm: Armv8-A memory model](https://developer.arm.com/-/media/Arm%20Developer%20Community/PDF/Learn%20the%20Architecture/Armv8-A%20memory%20model%20guide.pdf?revision=58b1dd0a-3800-4218-b21a-f95a0332034c)
