# Executable cache, metadata and backend ownership

[Specification index](README.md) · [Implementation tasks](tasks/README.md) · [Open review items](review-status.md)

Related contracts: [Policy defaults, layouts and safety bounds](policy-and-capacity.md); [Code units, versions, registries and admission](units-and-registries.md); [Epoch reclamation and terminal teardown](epochs-and-shutdown.md).

## Executable cache and backend ownership

Nixe, not JITModule, owns final native storage. Compilation produces bytes,
relocations, selected labels, state maps and fault sites exactly once.

The allocator reserves the 2047 MiB RX envelope, commits 16 MiB backing
segments on demand, starts reclamation before a reservation would take the
unified charge above 512 MiB and never permits it above 640 MiB, including
old/new cutover versions. LCQ and HCQ have separate active-segment lists but may
borrow completely unused segments. HCQ reservation defers before the
uncommitted balance falls below 32 MiB; that credit is reserved for LCQ code and
coupled metadata and is not a permanently assigned address range.

`jit_charged_bytes` is the authoritative checked-u64 budget value. It is the
sum of:

- the full 16 MiB backing size of every committed executable segment, counted
  once despite RW/RX aliases;
- committed pages in a dedicated decommittable metadata arena, including
  dispatch/unit/family slots, payloads, links, bridges, native-PC tables and
  their retired snapshots, dependency/ownership records, jobs, executable
  observations, epoch records, bounded per-vCPU JIT tables, NativeFrames,
  fault slots and committed SignalAltStack and landing-stack pages. Their three
  PROT_NONE guard pages reserve address space but have no backing charge;
- allocator headers and alignment padding actually committed in either arena;
  and
- outstanding worst-case staging reservations not already represented by the
  committed categories above.

The metadata arena is one 1024-MiB, 2-MiB-aligned process reservation of
exactly 262,144 Linux pages. Construction reserves and trims it with anonymous
private `PROT_NONE|MAP_NORESERVE`; overflow, wrong page size/alignment or a
returned address outside the requested owned range fails before Ready. Page 0
is the fixed `MetadataArenaHeaderV1`. Pages 1..=2048 hold exactly 262,144
32-byte `MetadataPageEntryV1` values, one per arena page. Those 2049 control/
directory pages are committed and charged at construction and never
decommitted before terminal unmap. Pages 2049..=2057 contain the global
allocation bitmap hierarchy and zero padding. The common allocatable pool is
the half-open page range `[2058,262144)`, exactly 260,086 pages. Thus all 2,058
permanent pages charge exactly 8,429,568 bytes. There is deliberately no
per-type virtual-range partition: the hard ledger and concrete-page reservation
protocol below are the simultaneous physical-capacity authority.

The nine bitmap pages contain, in order,
`free_l0:[u64;4064]`, `free_l1:[u64;64]`, `free_l2:u64` and 3,832 zero bytes.
An L0 bit is one iff the corresponding pool page is Free; the ten invalid tail
bits are zero. An L1 bit is one iff its referenced L0 word is nonzero and an L2
bit is one iff its referenced L1 word is nonzero; unused upper bits are zero.
All three levels are changed under the code-cache writer mutex and validated
after every change. Hierarchical trailing-zero selection returns the lowest
Free page. A request for N pages repeats that operation N times and therefore
does not depend on a contiguous virtual hole.

`MetadataArenaHeaderV1` is exactly:

```text
repr(C, align(4096)), size 4096 {
    magic:[u8;8] = *b"NIXEMD01", format_version:u32 = 1,
    page_size:u32 = 4096, page_count:u32 = 262144,
    pool_first_page:u32 = 2058, pool_page_count:u32 = 260086,
    metadata_type_count:u16, reserved_zero0:u16,
    metadata_allocation_generation:UnsafeCell<u64>,
    pool_counts:UnsafeCell<MetadataPoolCountsV1>,
    metadata_root_directory:MetadataBlockHandleV1,
    lowest_nonfull_page:UnsafeCell<[u32;256]>, reserved_zero1:[u8;2976]
}
MetadataPoolCountsV1 = repr(C, align(4)), size 16 {
    occupied_pool_pages:u32, committed_pool_pages:u32,
    reserved_pool_pages:u32, reserved_zero:u32
}
```

`UnsafeCell<T>` is layout-transparent here and these are its only three uses in
the header. The immutable prefix and root handle never change after Ready; the
generation, pool counts and hint array are read or written only by the code-
cache writer mutex owner through their UnsafeCells. The generation begins at one; allocation consumes the current value and
checked-increments the authority before another reservation can use it.
`metadata_type_count` is at most 256 and the hint array is indexed by manifest
metadata-slab row. An unused/no-page hint is `u32::MAX`. Freeing a slot in a formerly full
page writes the minimum hint; filling the hinted page scans directory entries
in ascending page order for the next same-type nonfull page. This is the only
fixed-slot hint and it is always validated before use.

Each permanent directory element is exactly:

```text
MetadataPageEntryV1 = repr(C, align(8)), size 32 {
    type_id:u32, first_page_index:u32,
    page_generation:u64, allocation_generation:u64,
    next_page_index_plus_one:u32,
    state:u8, storage_kind:u8, free_slot_count:u16
}
```

The page-directory mapping is exposed in Rust only as
`UnsafeCell<[MetadataPageEntryV1;262144]>`. Every entry read or mutation,
including diagnostic scans before terminal unmap, occurs while holding the
code-cache writer mutex; no lock-free reader aliases a plain entry. The fields
are intentionally non-atomic. The only publication to another thread is the
owning registry/root/payload location after the complete entry/header is ready,
using the Release ordering specified for that owner; an acquiring reader first
obtains its pin/owner protection and never treats a directory-field store as a
publication.

Page states are Permanent=1, Free=2, Reserved=3, Committed=4 and Retiring=5;
storage kinds are None=0, FixedSlot=1, PagedPayload=2, PermanentExtent=3,
GuardedStackExtent=4, Control=5 and IndexedSignalPayload=6. The last kind is
legal only for immutable RawNativePcRecordV1 arrays reached through a pinned
PageFaultTableHeaderV1 and follows the same Reserved/Committed/Retiring page-
chain and 128-byte-header rules as PagedPayload. For every page in `[0,2058)`, its directory
entry is exactly `(type_id=0, first_page_index=self, page_generation=1,
allocation_generation=0, next=0, state=Permanent, storage_kind=Control,
free_slot_count=0)` and never changes before terminal unmap. None is legal only
for a Free pool page.
Free is exactly `(type_id=0, first_page_index=self,
page_generation=current_nonzero, allocation_generation=0, next=0, state=Free,
storage_kind=None, free_slot_count=0)`. Every pool page begins in that tuple at
generation one. A FixedSlot Reserved entry is exactly
`(type_id=row.type_id, first=self, page_generation=current,
allocation_generation=G, next=0, state=Reserved, kind=FixedSlot,
free_slot_count=slots_per_page-1)` because its first slot is reserved at the
same time. Each PagedPayload or IndexedSignalPayload Reserved entry has the row type, its own current
page generation, common first page and G, strict ascending next link,
Reserved/its declared storage kind and free count zero. Successful decommit restores the
exact Free tuple with the precomputed successor page generation. Reserved and Retiring are
nonacquirable. FixedSlot uses first_page_index equal to itself, zero next and an
exact free-slot count. Every PagedPayload entry names the same first page and
allocation generation; page indices and next links are strictly increasing and
the final next field is zero. Directory state is the authority while a page is
PROT_NONE. At every allocator-mutex unlock and before any owner publication, a
committed in-page header must match it byte for byte where fields overlap; the
mutex-private Reserved-to-Committed initialization may write the header first
and directory second in its non-unwinding suffix.

Every persistent metadata allocation type resolves bijectively to one manifest
`metadata_slab` row; rows have distinct nonzero type_id, distinct `slab_index`,
and slab indexes are the dense range `0..metadata_type_count` after sorting by
type_id. Source order is irrelevant and the header hint array is indexed only
by this generated slab_index. There are at most 256 rows. A FixedSlot row has
power-of-two payload alignment in 1..=64, exactly equal to its generated
`metadata_slab.slot_alignment`,
`slot_size = align_up(align_up(16,payload_alignment) +
size_of(payload), max(8,payload_alignment))`,
`slots_per_page = floor((4096-128)/slot_size)` in 1..=496 and
`max_object_pages=1`. Its first 16 bytes are
`MetadataAllocationHeaderV1 = repr(C, align(8)), size 16 {
allocation_generation:u64, pin_count:AtomicU32, state:u8,
reserved_zero:[u8;3] }`; state is Staging=1,
Live=2 or Retiring=3. The payload starts at
`align_up(16,payload_alignment)`. Registry `slot_bytes` is this complete wrapper
size, not payload size alone.

A PagedPayload row has slot_size/slot_alignment/slots_per_page all zero. It
stores an opaque logical byte sequence across `page_count` independently chosen
pages; every page contributes exactly 3,968 bytes after its own 128-byte header,
so `page_count = ceil_div(byte_len,3968)`. `byte_len` is nonzero and page_count
does not exceed the row's exact `max_object_pages`. No API exposes it as one
contiguous slice: the sole checked accessor splits a logical `(offset,length)`
at 3,968-byte boundaries in chain order. Generated/native code never reads a
PagedPayload directly. IndexedSignalPayload is the sole exception and uses the
fixed 64-byte record/page-index algorithm under the native-PC directory; it is
never exposed as a Rust slice. For ordinary PagedPayload, no code casts payload bytes to `&T` or
`&[T]`: 3,968 is not a multiple of every stored record size. For every semantic
record T, [Task 0](tasks/00-baseline.md) generates a same-size `RawT` containing only integer fields and
byte arrays for which every bit pattern is valid, together with field-by-field
`decode_raw_t` and `encode_raw_t`; Rust enum, bool, NonZero, reference, pointer
and implicit-padding fields are forbidden in RawT. Generated
`read_record<T>(ordinal)` checks `ordinal*size_of(RawT)` and the final bound,
copies one or two page fragments into an aligned local RawT byte buffer,
validates every tag, range and reserved-zero byte, and only then constructs and
returns semantic T. `write_record` first encodes semantic T field by field into
a zero-initialized RawT and copies its bytes; it never serializes Rust padding.
Iteration is ordinal `0..count` through those accessors, including when a record
straddles a page. [Task 0](tasks/00-baseline.md) asserts `size_of::<RawT>()`, alignment and every field
offset and round-trips all legal discriminants plus invalid-tag/reserved-byte
fixtures.

A PermanentExtent row also has zero slot fields. It is constructed only before
Ready, occupies `ceil_div(byte_len,4096)` contiguous pages from the lowest
remaining pool prefix, requires a power-of-two payload alignment in 1..=4096,
exposes its page-aligned and therefore conforming direct base pointer, and
is never resized, moved, individually decommitted or reused. It has no in-page
128-byte header: all 4096 bytes of each page are payload, while its directory
entries carry one allocation generation, contiguous increasing next links and
the first-page identity. Its MetadataBlockHandle has slot zero, exact page
count/byte length and first-page generation. All PermanentExtents are jointly
preflighted and then reserved/committed in increasing layout_row numeric ID
before any dynamic FixedSlot/PagedPayload allocation; consequently contiguous
fit is construction-deterministic and never a runtime fragmentation promise.

`SignalStackBundleExtentV1` is the sole GuardedStackExtent object. Let
`A = signal_alt_stack_bytes / 4096` and `B = A + 19`. It occupies exactly
`configured_max_vcpus * B` contiguous pool pages during the same construction
prefix allocation. Bundle i begins at `first_page + i*B` and its page offsets
are exactly: guard 0; SignalAltStack `[1,1+A)`; guard `1+A`; the 16-page landing
stack `[2+A,18+A)`; and guard `18+A`. All its directory entries are Permanent,
have GuardedStackExtent storage kind, the common first page/allocation
generation and strictly increasing next links. Every one of these pages has a
zero free-L0 bit for the process lifetime. The three guard pages per bundle
retain the original anonymous PROT_NONE|MAP_NORESERVE mapping and are occupied
virtual pages but are neither committed nor ledger-charged. Only the two usable
ranges are replaced by anonymous RW mappings, contributing exactly
`configured_max_vcpus * (A+16) * 4096` committed pool bytes. There is no in-page
slab header. The guarded extent's one root entry and each immutable vCPU stack
descriptor carry its exact handle, B, A and the two bases derived by the
formula above; any independently stored base must equal that derivation.
Its handle has page_count `configured_max_vcpus*B` and byte_len equal to that
page count times 4096 (the virtual extent length, including guards).
GuardedStackExtent is never dynamically allocated, individually freed,
decommitted or reused, and terminal teardown removes it only with the complete
arena.

The committed page header is exactly:

```text
MetadataSlabPageHeaderV1 = repr(C, align(8)), size 128 {
    type_id:u32, first_page_index:u32,
    page_generation:u64, allocation_generation:u64,
    next_page_index_plus_one:u32, logical_page_ordinal:u32,
    page_count:u32, byte_len:u32, pin_count:AtomicU32,
    state:u8, storage_kind:u8, free_slot_count:u16,
    allocation_bits:[u64;8], live_slots:u16, retired_slots:u16,
    reserved_zero:[u8;12]
}
```

For a Committed FixedSlot page, page-header and directory
`allocation_generation` are zero; the page-count/byte-length/page-pin fields in
the page header are also zero. (Its directory may carry the temporary page-
reservation generation only while Reserved.) Each slot's distinct allocation
generation lives in its 16-byte allocation header and handle. Bit i is one
exactly while its slot is Staging/Live/Retiring,
and bits at or above slots_per_page are permanently one. For PagedPayload and
IndexedSignalPayload all
allocation bits/counts are zero; the first page alone stores page_count,
byte_len and the allocation's pin_count, while continuation pages store those
three fields as zero and carry their checked ordinal. All pages carry the same
type/page/allocation/link fields as their directory entry, except for the
explicit FixedSlot Reserved-to-Committed allocation-generation clearing above.

The only generic handle is:

```text
MetadataBlockHandleV1 = repr(C, align(8)), size 40 {
    type_id:u32, first_page:u32, slot_index_plus_one:u32, page_count:u32,
    byte_len:u32, reserved_zero:u32,
    page_generation:u64, allocation_generation:u64
}
```

FixedSlot has slot_index_plus_one in 1..=slots_per_page, page_count one and
byte_len zero. PagedPayload, PermanentExtent and GuardedStackExtent have slot index zero and their
exact nonzero page count and byte length. All fields, the complete page chain and the slot/page
allocation generation are validated before access; page generation alone never
protects fixed-slot reuse. Allocation generations come from the one checked
header authority and never repeat. A `MetadataBlockPin` can be acquired only
under the code-cache writer mutex after validating the typed owning state; it
checked-increments the fixed slot's or first payload page's pin_count and
revalidates before unlock. Release AcqRel-decrements/wakes; Retiring/reuse
requires zero. The only exception to ordinary pin acquisition is the initial
pin created atomically by allocation and returned in this linear value:

```text
MetadataStagingOwnerV1 = repr(C, align(8)), size 48 {
    block:MetadataBlockHandleV1, armed:u8, reserved_zero:[u8;7]
}
```

`armed` is one in the unique live Rust value and zero only after consuming it;
copy, clone and forget are forbidden and its armed Drop raw-exits 70.
Allocation initializes the FixedSlot or first PagedPayload pin_count to one and
returns this owner before releasing the cache mutex. `commit_into` has exactly
two nonnested stages: first it takes only the cache mutex, validates the block,
armed value and pin count one, then unlocks; that linear pin prevents retirement
afterward. It next takes only the target owner mutex, validates the target
preclaim and prequeue/Open protection without rereading MetadataPageEntry,
installs the handle in the typed live owner, release-publishes that owner, then
consumes the staging value without changing pin_count. The live/nonzero owning
location is itself the owner-pin marker; no hidden bool changes a payload
layout. `abort` is called with no subsystem
mutex: it takes the cache mutex, makes the allocation nonacquirable, consumes
the initial pin one-to-zero and runs the normal retire/decommit protocol. Every
branch calls exactly one operation before dropping its PrequeueProducerToken or
OpenToken. A PermanentExtent or GuardedStackExtent is instead pinned by the immutable root
directory until terminal teardown and never has a per-object pin. A bare handle
or pointer provides no lifetime.

Every typed owner, without exception, uses the inverse `detach_owned_block`
operation: under its own mutex it makes the live state/location nonacquirable
and takes the one handle/owner-pin obligation into a linear local; it completes
the type's reader/epoch grace, then with no subsystem mutex acquire-waits on the
allocation pin word until it is exactly one. A value above one is ordinary
reader drainage: release wakes that same word after every decrement. Zero,
overflow, generation/state mismatch or a new acquisition after the owner became
nonacquirable is corruption. The detacher then takes only the cache mutex,
revalidates the exact block, Live/owner state and pin count one,
AcqRel-decrements its owner pin one-to-zero and wakes, and only then changes it
to Retiring/free. Unit, family, patch, dispatch, owner,
fingerprint, mapping-vector, write-buffer and plan-claim owners all call this
same primitive; missing owner location, double detach or a pin count not equal
to the one owner pin after reader drain is fatal corruption.

The bootstrap root is:

```text
MetadataRootEntryV1 = repr(C, align(8)), size 48 {
    layout_row_id:u32, reserved_zero:u32, handle:MetadataBlockHandleV1
}
```

[Task 0](tasks/00-baseline.md) assigns a distinct nonzero layout_row_id and emits at most 256 root
entries, sorted strictly by that ID. Exactly one entry names every
PermanentExtent and GuardedStackExtent, including the root directory itself; no dynamic
FixedSlot/PagedPayload appears there. Construction reserves the exact
`entry_count*48`-byte root directory first at page 2058 (at most three pages),
writes its self entry and header handle, then reserves the remaining extents in
layout-row-ID order and fills their preallocated entries. The directory and
header remain private until all entries validate and Ready is release-published.
After Ready the root directory and immutable header fields are immutable; the
header's three UnsafeCell fields retain their mutex-owned semantics. Every pool-
resident top-level array is reached through this
directory, and TerminalDriverGuard retains it through the final arena scan and
unmaps it only with the entire arena. A missing, duplicate, mutable root
identity/base/page-chain or unreferenced construction extent prevents Ready;
an extent's payload may mutate only under the independently specified contract
of its concrete type (for example AtomicU64 directory entries or the unique
coordinator's cohort workspace).

Let P be the checked sum of pool pages occupied by all ordinary
PermanentExtents, including the root directory, and let A/B have the guarded-
stack meanings above. Immediately before Ready the exact header counts are
`occupied_pool_pages = P + configured_max_vcpus*B`,
`committed_pool_pages = P + configured_max_vcpus*(A+16)` and
`reserved_pool_pages = 0`. The free hierarchy has exactly
`260086 - occupied_pool_pages` L0 bits set and matching L1/L2 summaries, and
the ledger has exactly `(2058 + committed_pool_pages)*4096` committed metadata
bytes and zero construction reservation credit. Construction scans all
262,144 directory entries and validates these equalities before Ready; no
runtime allocation may repair a mismatch.

Every logical reusable registry/index value slab has a fixed
`RegistryDirectoryEntryV1 = repr(C, align(8)), size 8 {
location:AtomicU64 }`. Zero is absent; otherwise
bits 0..31 are `first_page+1`, bits 32..40 are
`slot_index_plus_one` (1..=496), and bits 41..63 are zero. There is exactly one
entry per logical slot: 64 address-space, 1,048,576 unit, 262,144 family,
2,097,152 patch, 266,404 fingerprint, 1,048,576 dispatch and 1,048,576 owner
entries. Each directory is a PermanentExtent; its AtomicU64 elements remain
eight-byte aligned because both its base and element size are multiples of
eight. Registry allocation
initializes the complete typed slot and release-publishes this word; removal
release-clears it before the registry-specific pin/epoch grace and metadata
free. A reader holds the named registry mutex/pin or active epoch while loading,
validates the logical handle generation in the slot, and rechecks location
before releasing that protection. The named `*_registry_directory` rows and
the `dispatch_slot_directory`/`owner_cell_directory` rows are exactly those
arrays; actual wrapper pages and the directories are charged once through
`metadata_arena_pool_pages`, while their logical rows use
logical_capacity_only.

With no other subsystem mutex, the code-cache writer mutex is the sole arena
writer. A FixedSlot allocation uses the validated lowest same-type nonfull page
and lowest zero bit, or reserves the lowest Free page if none exists. A
PagedPayload reservation selects the requested number of lowest Free pages and
links them in ascending order. Before the first directory, bitmap, header,
counter or ledger mutation, every allocation preflights the current nonzero
`metadata_allocation_generation`, its checked successor, the exact selected
page count and all arithmetic. Exhaustion latches IdentityExhausted and leaves
every allocator object unchanged. Each newly selected directory entry then
changes Free-to-Reserved with that one common allocation generation, the
hierarchy is updated, `occupied_pool_pages` and `reserved_pool_pages` each
increase by the selected count, and the reservation owns exactly 4096 ledger-
credit bytes per selected page.

The pool-count invariants are always
`committed_pool_pages + reserved_pool_pages <= occupied_pool_pages <= 260086`.
Occupied includes every Permanent/GuardedStack, Reserved, Committed and
Retiring pool page; committed counts only pages with actual RW backing,
including PermanentExtent payload and usable guarded-stack pages; reserved
counts only Free-to-Reserved dynamic pages awaiting commit. The three guard
pages per vCPU account for the possible strict first inequality. Before page
selection the prospective values must satisfy both inequalities and the
applicable 512/608/640 ledger gate. At every allocator mutex unlock, a directory
and mapping audit would yield those exact counts, and
`committed_metadata_bytes == 2058*4096 + committed_pool_pages*4096`;
`reserved_credit_bytes` separately includes
`reserved_pool_pages*4096` plus the other live reservation tickets specified in
this document. A slot reserved inside an already committed page changes none
of the three page counts and consumes no byte credit, because that page is
already charged, but its Staging bit prevents reuse.

A FixedSlot reservation on a committed same-type page performs exactly these
mutex-owned changes: require the selected usable allocation bit zero and the
slot header all zero; set the bit; decrement both page-header and directory
`free_slot_count`; initialize the slot header with the reserved allocation
generation, staging-owner pin one and Staging; and update the validated nonfull
hint. After the payload is completely initialized, allocator finalization
changes Staging-to-Live, increments `live_slots` and returns the armed
MetadataStagingOwner. Its later `commit_into`, not the allocator mutex section,
release-publishes the owning registry/location word. Retirement first release-clears that owning word and completes its
documented grace; under this mutex it changes the exact Live slot to Retiring,
decrements `live_slots` and increments `retired_slots`. Only at pin zero does
free zero the complete slot header/payload, clear the bit, decrement retired,
increment both free-slot counts and minimize the hint. Underflow, a count/bit/
state mismatch or a publication from Staging is fatal corruption.

For a FixedSlot allocation which selected a Free page, the Reserved directory
entry temporarily carries G. After its mapping succeeds, the no-fail
initialization writes a FixedSlot page header with allocation generation zero,
all invalid tail bits one, exactly the chosen usable bit one, free count
`slots_per_page-1`, live/retired zero, and writes that chosen slot as
Staging(G) with staging-owner pin one; the directory then becomes Committed with allocation generation
zero and the same free count. Its later Live/owner publication is exactly the
common FixedSlot finalization/MetadataStagingOwner transfer above; when its free
count is nonzero, commit also minimizes the exact slab-index hint with this
page. For a PagedPayload, every Reserved entry and
page header retains G; the first header has exact page_count/byte_len and
staging-owner pin one,
continuations have the required zero fields/ordinal, all headers use Committed
state, and the no-fail initializer copies and zero-pads the complete logical
payload across 3,968-byte pieces before returning MetadataStagingOwner. A
PagedPayload handle becomes externally Live solely by `commit_into`'s owning
publication; retirement
clears it before changing every page to Retiring. Mapping syscalls are the only
fallible work after page reservation: source validation, sizes, generations,
headers and complete payload bytes are precomputed or pinned first, and all
post-mapping initialization/counter stores execute under the allocator's
non-unwinding commit guard. The resulting MetadataStagingOwner is the sole
cross-mutex staging token and follows its exact transfer/abort protocol above.

Before the first commit mapping, one `MetadataCommitLedgerGuard` preflights and
reserves exactly one ordinary odd/even ledger-sequence pair and every checked
counter value for the two observable outcomes: full commit or full rollback.
Sequence/arithmetic failure occurs before any syscall. Commit installs each
exact Reserved page in ascending index order with offset-zero anonymous
`MAP_PRIVATE|MAP_ANONYMOUS|MAP_FIXED` RW, validates the returned address, zeroes
it and writes its complete header. Only after every page succeeds does the mutex
owner change all ordinary directory entries to Committed, subtract the page
count from `reserved_pool_pages`, add it to `committed_pool_pages`, and
atomically convert equal reservation credit to committed metadata charge;
Release publication of the owning handle/location follows and is the only
inter-thread publication. MAP_FIXED is legal only for the exact still-Reserved
pages inside this arena.

The first failed or wrong-address commit mapping makes the affected mapping
state unknowable; errno is never used to infer that the old PROT_NONE mapping
survived. With the cache mutex and armed non-unwinding guard still held, rollback
therefore replaces **all N pages** of that reservation, including the failed
index and any not yet attempted, by fresh anonymous
`PROT_NONE|MAP_NORESERVE` mappings at their exact addresses in ascending page
order. Only if every restoration returns its exact requested address does it
return all entries and hierarchy bits to Free, subtract N from occupied and
reserved pool pages, release the complete reservation credit in the one
preflighted ledger publication and report the original allocation failure. It
publishes no handle. If any restoration syscall fails or returns another
address, it invokes raw `SYS_exit_group(70)` immediately without unlocking,
publishing a handle, attempting a ledger audit or claiming to know which pages
remain backed. `MetadataCommitLedgerGuard::drop` has the same raw-exit behavior
unless one complete observable outcome disarms it.

Freeing first makes the typed owner nonacquirable and clears any registry
directory word, then waits its documented epoch/pin/reader grace. Under the
allocator mutex it validates the exact allocation and pin zero. A nonempty
FixedSlot page follows only the slot transitions above. Before an empty
FixedSlot page or PagedPayload chain performs its first Retiring state/bitmap
change or first syscall, the owner computes and retains the checked successor
`page_generation` for every page in ascending chain order. Any overflow leaves
the typed owner nonacquirable but all allocator entries/mappings/counts
unchanged and charged, latches IdentityExhausted and permits only terminal
teardown. It also preflights exactly one ledger odd/even pair and the full-page-
count subtraction before the first state change/syscall. After successful
preflight it marks the complete page set Retiring; FixedSlot requires all usable
bits zero and live/retired zero, while PagedPayload requires its first-page pin
zero. Before a FixedSlot page leaves Committed, if its slab hint equals that
page, the mutex owner scans ascending for the lowest other Committed same-type
page with nonzero free count or writes u32::MAX; a hint may never name Reserved,
Retiring or Free.

Each decommit replaces the page by anonymous `PROT_NONE|MAP_NORESERVE` at the
identical address in ascending order. A failed or wrong-address MAP_FIXED makes
the state unknowable and causes the armed `MetadataDecommitGuard` to invoke raw
`SYS_exit_group(70)` immediately, while still holding the mutex; the live
process never publishes partial counters or attempts typed terminal recovery
from that uncertainty. Only after **every** page succeeds does the mutex owner
use the preflighted ledger pair to subtract the complete page count and byte
charge, change all directory entries to Free with their precomputed page
generations and zero allocation/type/link/count fields, restore all hierarchy
bits, subtract the complete page count from `occupied_pool_pages`, and disarm
the guard. An armed Drop raw-exits 70, so no ordinary continuation can expose
partially replaced mappings with stale ledger counts. No `Vec`, Arc,
free-list node, contiguity search, catch-all header allocation or append-only
retired queue owns production metadata.

[Task 0](tasks/00-baseline.md) fixes these semantic maxima: a MappingGenerationVector has at most
2,097,152 `MappingGenerationEntryV1` values and therefore at most 25,369 pages;
a mapping WriteBuffer has at most 16,777,216 bytes and 4,229 pages; a
BuildFingerprint payload has at most 180,696 bytes and 46 pages; and a
non-Pressure RawPlanRecord array has at most 5,501,120 records/132,026,880 bytes and
33,273 pages. The generator recomputes and requires these exact page ceilings.
The `mapping_generation_vector_pages`, `mapping_write_buffer_pages`,
`build_fingerprint_payload_pages` and `root_mutation_plan_claim_pages` layout
rows are all logical_capacity_only PagedPayload capacities backed and charged
only by `metadata_arena_pool_pages`; none is a second lazy-committed physical
owner.
An ordinary mapping request above the vector limit returns
FailedBeforeMutation(RangeTooLarge). DestroyAddressSpace instead makes the slot
Destroying once, processes ascending at-most-limit page batches in consecutive
rounds while that address space remains inadmissible, and reports completion
only after the final batch. An ExternalWrite above the buffer maximum is split
by its caller into ascending at-most-16-MiB operations with explicit committed-
prefix results; no API promises atomicity across that split. Allocation still
fails Capacity before mutation when concrete pages or ledger credit are
unavailable; theoretical registry capacities are not promised simultaneously
resident.

The uncommitted `PROT_NONE` address envelope, the one TerminalControl page,
WorkerJoinSet/process-owner control, Rust stack, compiler scratch and caller-
owned benchmark data are excluded.
Retained compiler contexts/scratch
are bounded by worker/vCPU count and candidate instruction ceilings but are not
misreported as code-cache metadata. Except for the explicitly bounded
WorkerJoinSet and retained worker compiler contexts/scratch above, every
persistent JIT object is allocated from the charged arena; any other general
heap allocation is worker-local staging and must be freed before publication. Reservation
atomically charges a conservative page-rounded maximum before any staging
growth and reconciles it to exact committed capacity before publication; unused
credit is released and precommit failure rolls back all credit. The ledger has
three independently checked atomic-u64 subcounters:
`committed_executable_bytes`, `committed_metadata_bytes` and
`reserved_credit_bytes`, plus the independently stored authoritative atomic-u64
`jit_charged_bytes`. All allocator writers and all 512/608/640-MiB admission
decisions hold the one code-cache writer mutex and no other subsystem mutex.
Before an update, the writer computes all four prospective values in locals
with checked arithmetic, requires total equals their checked sum and applies
the relevant gate. It then release-stores an odd `ledger_sequence`, stores the
three subcounters and authoritative total, and release-stores the next even
sequence; no operation in that odd interval can fail. Sequence exhaustion calls
`latch_terminal(LedgerInvariant, SequenceExhausted)` before changing a counter, releases
the code-cache writer mutex and then calls `drive_or_join_terminal()`.
`ledger_sequence` starts at zero and ordinary writers advance it by exactly two.
They may publish even values only through `u64::MAX - 3`; the final odd/even
pair `u64::MAX - 2` and `u64::MAX - 1` is reserved exclusively for terminal
zeroing. An ordinary writer which would consume that pair latches terminal
without entering an odd interval. After all ordinary writers drain, the
TerminalDriverGuard may jump from the last ordinary even value to the reserved
odd value, store four zero counters, and publish the reserved even value.
`Stopped` is forbidden before that even store; `u64::MAX` is never a ledger
sequence.

A reader acquire-loads an even `ledger_sequence`, acquire-loads all four
counters, acquire-loads the sequence again, and accepts only equal even values
with `jit_charged_bytes` equal to the checked subcounter sum; otherwise it
retries. The frozen external performance reader uses exactly this protocol.
After all allocator activity is drained, the terminal driver is the sole writer
and uses the same odd/even publication to set all four counters to zero. A debug
arena walk must equal the two committed subcounters, and a walk of the
generational reservation table must equal
`reserved_credit_bytes`; an arena walk alone is not expected to include logical
credit.

HCQ, StaticBridgeUnit, dynamic BridgeUnit, PIC/link-install and every other
performance-only reservation may proceed only when the resulting total is at
most 608 MiB. Only demanded LCQ and already-precharged mandatory
reclamation/COW/safety storage may consume the final 32 MiB through 640 MiB.
Free executable spans remain charged while their
segment is committed. Freed metadata remains charged until all objects on its
arena pages are quiescent and those pages are decommitted. The reported
`jit_charged_bytes` is the quantity used for the 512/608/640 MiB gates and
external acceptance. Process
construction precharges fixed per-vCPU/directory structures and fails before
guest start if they cannot coexist with the 32 MiB LCQ reserve.

The lifecycle applies to allocations, not to an entire segment which may
contain spans at different stages. Protection is defined per 4096-byte alias
page: an RX alias is never writable, and an RW alias is never executable. An RX
page becomes executable before any entry on it is published and stays so while
any span on it is callable. Its RW twin is normally PROT_NONE and may be
temporarily read/write under the code-cache writer lock to populate an exact
byte-disjoint unreachable span, or in Closed to patch callable bytes; the
writer bounds-checks every store to the reserved span/patch and returns the
whole RW page to PROT_NONE before publication/reopen. Thus two virtual aliases
of one backing page may transiently be W and X respectively; this dual-mapping
policy is intentional. No individual virtual mapping is ever RWX, and the RW
alias is never a branch target or exposed to generated code.

Linux creates those aliases from one sealed-size backing object. The allocator
never places two live allocations over the same bytes, and the cache-coherence
wrapper covers complete host cache lines touched through either alias even when
a line also contains unchanged neighboring code. Generated/guest addresses are
derived only from the RX envelope, and failure to create or protect both aliases
disables the native backend before guest start.

The backing protocol is exact and uses one object per committed segment. Under
the code-cache writer mutex, segment creation calls
`memfd_create("nixe-jit-segment", MFD_CLOEXEC|MFD_ALLOW_SEALING)`,
`ftruncate(fd, 16 MiB)`, then adds `F_SEAL_GROW|F_SEAL_SHRINK|F_SEAL_SEAL`;
`F_SEAL_WRITE` is deliberately absent because the private RW alias must remain
writable. It reserves a distinct 16-MiB RW-alias range with anonymous
PROT_NONE. It replaces exactly the already-owned segment slice of the RX
envelope and that RW range with offset-zero `MAP_SHARED|MAP_FIXED` mappings of
the memfd, initially PROT_NONE, and verifies both returned addresses exactly.
MAP_FIXED is legal only for those two pre-reserved ranges while the allocator
mutex proves ownership; it is forbidden elsewhere. Any syscall/seal/address
mismatch closes/unmaps private partial state, restores the RX slice to anonymous
PROT_NONE with MAP_FIXED, and fails construction/allocation before reachability.

To populate pages, the allocator changes only the exact RW alias pages to
PROT_READ|PROT_WRITE, writes/relocates them, performs the platform cache protocol,
returns them to PROT_NONE, and changes the RX pages from PROT_NONE to the exact
executable protection above. Closed patching may temporarily make the RW twin
writable while the RX twin stays executable, but no single mapping is RWX.
A segment is physically decommitted only when it has zero non-island body or
BridgeUnit spans, zero roots/epochs/directory or preparation pins, and all
removed tables passed grace. Helper-island slots and their intern table do not
by themselves keep it live. The allocator makes both aliases PROT_NONE,
unmaps both file mappings, closes the memfd, reinstalls the exact RX-envelope
slice as anonymous PROT_NONE, releases the RW reservation, clears the island
map/bitmap and only then decrements the 16-MiB ledger charge and advances the
checked segment generation. A syscall failure leaves the segment inaccessible
but charged, latches ProtectionFailure and enters terminal cleanup; there is no
unaccounted hole-punch or partial-segment decommit path.

Executable body allocation has no map, node allocation or implementation-
selected coalescing structure. The allocatable body range is exactly
`[segment_base, segment_base + 16 MiB - 64 KiB)`, divided into 1,044,480
sixteen-byte quanta. For each of 127 segments the fixed L0 bitmap contains
16,320 little-endian `u64` words/130,560 bytes; bit q is one iff body quantum q
is allocated. Its fixed L1 bitmap contains 16,320 bits/255 little-endian `u64`
words/2,040 meaningful bytes, followed by zero padding in its layout row; L1
bit w is one iff L0 word w is not `u64::MAX`. The process-global contiguous L0
and L1 arrays are fully committed at construction. Their report formulas are
respectively `ceil_div(127 * 130560,4096) * 4096 = 16,584,704` and
`ceil_div(127 * 2040,4096) * 4096 = 262,144` fixed committed metadata bytes.
Unused tail bytes/bits are permanently zero and never searched.

Body and BridgeUnit spans are rounded to 16 bytes and aligned to
`max(16, backend_alignment)`; backend alignment is a power of two no greater
than 4096. Under the code-cache writer mutex, allocation scans L1/L0 in
increasing quantum order, derives every maximal zero-bit run, and for each run
computes its lowest address satisfying the requested alignment. It selects the
candidate with the fewest quanta in that maximal run, then lowest RX address;
if none fits, that segment cannot satisfy the request. It revalidates all
selected zero bits, sets them, and updates the affected L1 bits before
unlocking. Freeing validates the exact span/segment generation, requires all
bits one, clears them and updates L1; adjacent spans coalesce implicitly as one
zero run. Any mismatched bit is allocator corruption. There is no bump
frontier, free-span node, fragmentation-dependent auxiliary allocation or
alternate search policy.

The final 64 KiB remains the separate fixed island-slot bitmap/record array and
allocates the lowest free slot. A stale unpublished body span was never reachable and is
cleared immediately by its exact reservation owner. A published span is
cleared only after all entry/link/PIC roots are cut, its retirement epoch is
quiescent and all compiler/link/fault strong references are gone. A segment is
decommit-eligible exactly when all meaningful L0 bits are zero, every remaining
set island bit is a Helper record (no StaticExit remains), and it owns no root/
reference named above. Helper-only records do not keep it live: the decommit
transaction invalidates their keys, clears every record/bit and then advances
the segment generation. The segment is then decommitted or reused with that new
generation.

Under soft-limit pressure:

1. reclaim quiescent spans and every decommit-eligible segment;
2. stop new HCQ admission and finish already-pending cutovers;
3. select whole CurrentStable, non-ReplacementReserved HCQ families by lowest
   CodeUnit creation sequence then CodeUnitId; after each selection, simulate
   exact post-grace allocator reference counts and count only executable
   segments which would become wholly decommittable plus metadata pages whose
   last live object would disappear, never the selected unit's logical byte
   size;
4. if that projection is still above 512 MiB with the requested reservation,
   continue the same simulation after removing the selected families' LCQ pins
   and select oldest LCQ units by the same order. An LCQ is selectable only if
   its simulated explicit-pin count is then zero and it is not a pinned
   gateway/stub. Stop when the exact post-grace projection meets the target or
   no eligible unit remains;
5. construct one immutable `PressureEvict` RootMutationRequest whose precharged
   unit/family selection bitmaps contain both selected sets plus the input
   ledger sequence/projection, then request one maintenance rendezvous.
   ClosingPrepare alone revalidates that selection and expands every
   dispatch/patch/native-page/PIC consequence into CohortPlanWorkspace before
   sealing the RootMutationPlan. A key whose HCQ is selected ends at its retained LCQ unless that
   LCQ is also selected, in which case it ends at Unavailable; the transaction
   never briefly publishes that LCQ. ClosingPrepare revalidates the exact pin/
   page projection before any mutation, ClosingFreeze publishes all final
   payloads, and Closed unlinks both sets and releases family-to-LCQ pins. It
   performs no second scan or plan creation in Closed;
6. preserve the 32 MiB LCQ credit and never select gateway/stub pinned units;
   and
7. resume HCQ admission only after `jit_charged_bytes <= 512 MiB` and the next
   HCQ worst-case reservation would remain at or below 608 MiB.

`ReclaimToSoftWatermark` is a production-general cold control operation over
that exact policy, not a benchmark branch. The process owns one fixed,
precharged `ReclaimControlCell` in the metadata arena and one stable
`reclaim_control_mutex`; its complete `TaggedPayloadCell` graph is:

```text
Free(g)
  -> Running(g, ReclaimRequestId)
  -> Applied(g, ReclaimAck) | Failed(g, ReclaimError)
  -> Free(g + 1)                                      [ShutdownRecord Idle]
Running(g, id) -> CancelledByStop(g, id, TerminalCauseCode)
CancelledByStop -> terminal nonreusable                [shutdown]
```

`ReclaimRequestId` is a checked nonzero-u64 allocated by this cell and never
reused. `ReclaimError` is the closed `repr(u8)` domain
`NoEligibleUnit=1 | Stopped=2`; the runtime graph stores `Stopped` only as
`CancelledByStop(cause)`, and the public API maps that state to
`ReclaimError::Stopped`. The cell contains a checked-u32 `requester_refcount`,
one `running_owner` bit and a u32 parking sequence. Every new caller acquires a
NormalResultToken and then a PrequeueProducerToken in that order before taking
the reclaim mutex. While holding that mutex it rechecks both terminal gates and
either changes Free to Running while atomically establishing
`requester_refcount = 1` and the unique owner bit, or checked-increments the
reference count of the exact Running/terminal generation. It unlocks and drops
PrequeueProducerToken only after that publication/reference is complete; a
gate/terminal failure makes no claim and drops both tokens. A caller which
already owns a requester reference needs no second producer token merely to
wait or settle it. Thus every caller, including the owner, holds
exactly one requester reference, and the terminal prequeue drain proves no new
Free-to-Running edge can occur. It unlocks before work or wait,
revalidates after every wake, copies a terminal payload before decrementing its
reference, and releases its NormalResultToken before returning. The last
reference, provided `running_owner == 0` and ShutdownRecord remains Idle,
alone advances Applied/Failed to checked next-generation Free. During shutdown
it leaves the exact terminal tag nonreusable with zero references, so unmap
does not require a successor generation. Each state publication advances and
broadcasts the cell's parking sequence. `requester_refcount` is itself an
aligned AtomicU32 futex; every checked decrement broadcasts on that word, so
MAX-to-MAX-1 capacity and N-to-zero drain waiters progress without consuming
the TaggedPayloadCell's bounded parking-sequence reserve.

The Running owner repeatedly invokes the existing PressureRecord, waits outside
all subsystem mutexes for each cut-root epoch/pin grace, reclaims every eligible
span/page, and recomputes the authoritative ledger. It never starts another
PressureRecord after ShutdownRecord is non-Idle. After the last relevant
PressureRecord is terminal and its Open handoff is acquire-observed, the owner
captures `observed_cutover_sequence` as the coordinator's
`acknowledged_sequence` while holding only the maintenance-queue mutex, unlocks,
acquire-loads `observed_epoch`, and performs one ascending finite scan. For that
scan `reclaimable_count` is exactly the checked sum of: unit slots in Retired or
DirectoryDetached whose existing epoch and UnitPin conditions permit their
next reclamation edge now; FamilyRetired slots whose FamilyPin/member
conditions permit FamilyReclaimed now; retired PageFaultTable/native-directory
snapshots whose recorded grace and reader count permit release now; wholly free
committed executable segments; and committed metadata slab pages with zero live
objects. These categories count allocation owners, are disjoint, and their
documented maxima sum below u32::MAX; a duplicate or overflow is an internal
ledger invariant. The owner reclaims the counted objects and repeats until the
count is zero, or returns `Failed(NoEligibleUnit)` only when the count is zero,
the exact pressure selector has no eligible current unit, and the charged total
still exceeds the watermark.

`ReclaimAck` is exactly `{ request_id:ReclaimRequestId,
observed_cutover_sequence:u64, observed_epoch:u64, reclaimable_count:u32,
jit_charged_bytes:u64 }`, with the ledger value obtained by one even-sequence
read after that final scan; Applied requires both `reclaimable_count == 0` and
`jit_charged_bytes <= 512 MiB`. A caller-supplied timeout and any external
attempt identity are caller metadata and are never stored in the runtime cell
or acknowledgement. After ShutdownRecord is non-Idle, either the exact Running
owner while settling its own obligation or the TerminalDriver in step 3 may,
under only the reclaim mutex, perform the single exact
Running-to-CancelledByStop transition, clear `running_owner` and broadcast.
The loser observes that terminal state and may only release resources; it
cannot overwrite the result or clear the bit twice. This transition completes
before the TerminalDriver waits for NormalResultTokens. Terminal scan requires
a terminal/Free cell, zero
requester references and no owner bit before arena unmap. The operation is safe
at any cold control boundary and never runs inside generated code or a signal
handler.

An LCQ CodeUnit referenced by an unselected active/in-flight HCQ family is not
eligible for eviction. The composite simulation may select it only when every
such pin belongs to an HCQ family in the same plan and all other explicit pins
are zero. Thus no
eviction or reshape can promise a retained baseline which has already been
reclaimed.

The guest never waits for HCQ capacity. An LCQ miss may synchronously reuse an
already-quiescent span. If the 32 MiB reserve is exhausted by live LCQ, the miss
requests the process maintenance rendezvous, retires the oldest eligible units
and waits only for that finite-work unlink/epoch protocol. If no unit can be retired
without violating correctness, execution reports a precise capacity failure;
it never silently falls back to another engine. The 2047 MiB virtual and 640
MiB charged-total bounds are hard. Metadata capacity is derived from and charged to
the same live/retired unit budget; no independent append-only fault or
retired-unit vector exists.
