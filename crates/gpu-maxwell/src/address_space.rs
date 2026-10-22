//! Switch 1 Maxwell GPU virtual address-space semantics.

use std::collections::BTreeMap;
use std::fmt::{Display, Formatter};
use std::sync::Arc;

use nixe_gpu::physical::PhysicalMapping;
use nixe_gpu::{GpuVirtualAddress, GpuVirtualAddressError};
use nixe_memory::{
    AllocationRange, ArenaOffset, Machine, MemoryError, MemoryPermissions, capture_subrange,
};
use std::sync::atomic::{AtomicBool, Ordering};
mod memory;
mod table;
pub use memory::MaxwellGpuMemory;
use nixe_gpu::MappingGeneration;
use nixe_gpu::MappingGenerationExhausted;
use table::PageTable;

use crate::{GpuProfileId, MaxwellGpuProfile};

const ALLOC_SPACE_FIXED: u32 = 1 << 0;
const ALLOC_SPACE_SPARSE: u32 = 1 << 1;
const ALLOC_SPACE_KNOWN_FLAGS: u32 = ALLOC_SPACE_FIXED | ALLOC_SPACE_SPARSE;
const INVALID_PTE_KIND: u8 = 0xff;

/// Hard upper bound for one address-space mapping diagnostic.
pub const MAX_MAPPING_DUMP_ENTRIES: usize = 64;

// The Switch frontend defaults below and the rule which moves the start when a
// different supported big-page size is selected are recorded by this pinned
// public implementation:
// https://github.com/yuzu-emu-mirror/yuzu-mainline/blob/2d2522693e7d453bf10a8246f704350b69e12ebc/src/core/hle/service/nvdrv/devices/nvhost_as_gpu.h#L190-L204
//
// NVIDIA's public Tegra driver independently establishes the one-PDE low hole
// and distinct low-small/high-big allocators:
// https://android.googlesource.com/kernel/tegra.git/+/d28c42ee85e186bf02189e9cdacaff0c3c55f2e9/drivers/gpu/nvgpu/gk20a/mm_gk20a.c#2124
const DEFAULT_VA_START_SHIFT: u32 = 10;
const DEFAULT_VA_SPLIT: u64 = 1_u64 << 34;
const DEFAULT_VA_END: u64 = 1_u64 << 37;

/// Stable identity of one Maxwell GPU virtual address space.
///
/// This is an emulator identity. It is not a Horizon file descriptor, CPU
/// address-space identifier, guest GPU address, or host graphics object.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct MaxwellAddressSpaceId(u64);

impl MaxwellAddressSpaceId {
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl Display for MaxwellAddressSpaceId {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "gpu-address-space=0x{:016x}", self.0)
    }
}

/// Initialization parameters after the Horizon ioctl ABI has been decoded.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MaxwellAddressSpaceInitialization {
    pub big_page_size: u32,
    pub va_range_start: u64,
    pub va_range_end: u64,
    pub va_range_split: u64,
}

/// One page-size-specific allocator region reported to the guest.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellVaRegion {
    offset: GpuVirtualAddress,
    page_size: u32,
    pages: u64,
}

impl MaxwellVaRegion {
    #[must_use]
    pub const fn offset(self) -> GpuVirtualAddress {
        self.offset
    }

    #[must_use]
    pub const fn page_size(self) -> u32 {
        self.page_size
    }

    #[must_use]
    pub const fn pages(self) -> u64 {
        self.pages
    }

    fn end(self) -> Result<u64, MaxwellAddressSpaceError> {
        self.pages
            .checked_mul(u64::from(self.page_size))
            .and_then(|size| self.offset.get().checked_add(size))
            .ok_or(MaxwellAddressSpaceError::ArithmeticOverflow)
    }

    fn contains(self, start: u64, end: u64) -> Result<bool, MaxwellAddressSpaceError> {
        Ok(start >= self.offset.get() && end <= self.end()?)
    }
}

/// Metadata for address space reserved by `NVGPU_AS_IOCTL_ALLOC_SPACE`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellVaReservation {
    offset: GpuVirtualAddress,
    pages: u32,
    page_size: u32,
    sparse: bool,
}

impl MaxwellVaReservation {
    #[must_use]
    pub const fn offset(self) -> GpuVirtualAddress {
        self.offset
    }

    #[must_use]
    pub const fn pages(self) -> u32 {
        self.pages
    }

    #[must_use]
    pub const fn page_size(self) -> u32 {
        self.page_size
    }

    #[must_use]
    pub const fn sparse(self) -> bool {
        self.sparse
    }

    fn end(self) -> Result<u64, MaxwellAddressSpaceError> {
        u64::from(self.pages)
            .checked_mul(u64::from(self.page_size))
            .and_then(|size| self.offset.get().checked_add(size))
            .ok_or(MaxwellAddressSpaceError::ArithmeticOverflow)
    }
}

/// Stable frontend identity of the allocation retained by one GPU mapping.
///
/// Horizon assigns this from the semantic `nvmap` object, never from a
/// guest-visible handle. The Maxwell frontend consequently does not depend on
/// Horizon's handle or exported-ID domains.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct MaxwellAllocationId(u64);

impl MaxwellAllocationId {
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl Display for MaxwellAllocationId {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "allocation=0x{:016x}", self.0)
    }
}

/// Stable identity of one mapping lifetime within an address space.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct MaxwellMappingId(u64);

impl MaxwellMappingId {
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl Display for MaxwellMappingId {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "mapping=0x{:016x}", self.0)
    }
}

/// Fully decoded request owning the accepted physical allocation bytes.
#[derive(Clone, Debug)]
pub struct MaxwellMapRequest {
    pub allocation: MaxwellAllocationId,
    pub backing: Vec<AllocationRange>,
    pub backing_offset: u64,
    pub size: u64,
    pub allocation_alignment: u32,
    pub page_size: u32,
    pub kind: u8,
    pub cacheable: bool,
    pub permissions: MemoryPermissions,
    pub fixed_offset: Option<GpuVirtualAddress>,
}

/// One entry in the big-page sparse-remap ABI after handle resolution.
#[derive(Clone, Debug)]
pub struct MaxwellSparseRemapRequest {
    pub offset: GpuVirtualAddress,
    pub size: u64,
    pub mapping: Option<MaxwellSparseMapping>,
}

/// Accepted physical allocation bytes installed by a sparse-remap entry.
#[derive(Clone, Debug)]
pub struct MaxwellSparseMapping {
    pub allocation: MaxwellAllocationId,
    pub backing: Vec<AllocationRange>,
    pub backing_offset: u64,
    pub kind: u8,
    pub cacheable: bool,
    pub permissions: MemoryPermissions,
}

/// Weak physical interpretation of one mapping lifetime. Live mappings own
/// their exact RAM pins; caches never retain those pins through this descriptor.
#[derive(Clone, Debug)]
pub struct MaxwellGpuMapping {
    owner: std::sync::Weak<memory::TranslationState>,
    id: MaxwellMappingId,
    unmap_offset: GpuVirtualAddress,
    offset: GpuVirtualAddress,
    size: u64,
    allocation: MaxwellAllocationId,
    physical: Arc<PhysicalMapping>,
    current: Arc<AtomicBool>,
    slot: u32,
    backing_offset: u64,
    permissions: MemoryPermissions,
    page_size: u32,
    kind: u8,
    cacheable: bool,
    generation: MappingGeneration,
    fixed: bool,
    sparse: bool,
}

impl PartialEq for MaxwellGpuMapping {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.current, &other.current)
            && self.offset == other.offset
            && self.size == other.size
            && self.generation == other.generation
    }
}
impl Eq for MaxwellGpuMapping {}

struct LiveMapping {
    descriptor: MaxwellGpuMapping,
    pins: Vec<AllocationRange>,
}
impl std::fmt::Debug for LiveMapping {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.descriptor.fmt(f)
    }
}

impl MaxwellGpuMapping {
    #[must_use]
    pub const fn id(&self) -> MaxwellMappingId {
        self.id
    }

    #[must_use]
    pub const fn offset(&self) -> GpuVirtualAddress {
        self.offset
    }

    #[must_use]
    pub const fn size(&self) -> u64 {
        self.size
    }

    #[must_use]
    pub const fn allocation(&self) -> MaxwellAllocationId {
        self.allocation
    }

    #[must_use]
    pub fn physical(&self) -> &Arc<PhysicalMapping> {
        &self.physical
    }

    #[must_use]
    pub const fn backing_offset(&self) -> u64 {
        self.backing_offset
    }

    #[must_use]
    pub const fn permissions(&self) -> MemoryPermissions {
        self.permissions
    }

    #[must_use]
    pub const fn page_size(&self) -> u32 {
        self.page_size
    }

    #[must_use]
    pub const fn kind(&self) -> u8 {
        self.kind
    }

    #[must_use]
    pub const fn cacheable(&self) -> bool {
        self.cacheable
    }

    #[must_use]
    pub const fn generation(&self) -> MappingGeneration {
        self.generation
    }

    #[must_use]
    pub const fn fixed(&self) -> bool {
        self.fixed
    }

    #[must_use]
    pub const fn sparse(&self) -> bool {
        self.sparse
    }

    fn end(&self) -> Result<u64, MaxwellAddressSpaceError> {
        self.offset
            .get()
            .checked_add(self.size)
            .ok_or(MaxwellAddressSpaceError::ArithmeticOverflow)
    }
}

/// Numeric physical translation, valid while the graphics owner keeps its
/// mapping unchanged. Owning operations capture their physical pins separately.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellGpuTranslation {
    pub allocation: nixe_memory::AllocationId,
    pub physical: ArenaOffset,
    pub bytes: u64,
    pub generation: MappingGeneration,
    pub permissions: MemoryPermissions,
    pub kind: u8,
}

/// Maximal contiguous portion of one active mapping beginning at a GPU VA.
///
/// This is weak mapping metadata. A consuming operation captures only its
/// requested physical bytes before delegating work or changing mappings.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaxwellResolvedMapping {
    gpu_offset: GpuVirtualAddress,
    size: u64,
    backing_offset: u64,
    mapping: MaxwellGpuMapping,
}

impl MaxwellResolvedMapping {
    #[must_use]
    pub const fn gpu_offset(&self) -> GpuVirtualAddress {
        self.gpu_offset
    }

    #[must_use]
    pub const fn size(&self) -> u64 {
        self.size
    }

    #[must_use]
    pub const fn backing_offset(&self) -> u64 {
        self.backing_offset
    }

    #[must_use]
    pub const fn mapping(&self) -> &MaxwellGpuMapping {
        &self.mapping
    }
}

/// Checked scatter/gather resolution of one complete GPU virtual range.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaxwellResolvedRange {
    address_space: MaxwellAddressSpaceId,
    offset: GpuVirtualAddress,
    size: u64,
    permissions: MemoryPermissions,
    // Caches share weak mapping metadata. Inline append alone changes topology;
    // accepted native inputs own their physical pins separately.
    segments: Arc<[MaxwellResolvedMapping]>,
}

impl MaxwellResolvedRange {
    /// Extend an ordered inline upload within the same mapping lifetime.
    /// Consumption captures the complete physical range once the run ends.
    pub(crate) fn append_inline_word(&mut self, address: u64) -> Option<()> {
        self.append_inline_bytes(address, 4)
    }

    pub(crate) fn append_inline_bytes(&mut self, address: u64, size: u64) -> Option<()> {
        if self.offset.get().checked_add(self.size)? != address || self.segments.len() != 1 {
            return None;
        }
        if address.checked_add(size)? > self.segments[0].mapping.end().ok()? {
            return None;
        }
        let segment = &mut Arc::make_mut(&mut self.segments)[0];
        segment.size += size;
        self.size += size;
        Some(())
    }

    #[must_use]
    pub const fn address_space(&self) -> MaxwellAddressSpaceId {
        self.address_space
    }

    #[must_use]
    pub const fn offset(&self) -> GpuVirtualAddress {
        self.offset
    }

    #[must_use]
    pub const fn size(&self) -> u64 {
        self.size
    }

    #[must_use]
    pub const fn permissions(&self) -> MemoryPermissions {
        self.permissions
    }

    #[must_use]
    pub fn segments(&self) -> &[MaxwellResolvedMapping] {
        &self.segments
    }
}

/// Pointer-free summary of one active GPU mapping.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellMappingDiagnostic {
    pub offset: GpuVirtualAddress,
    pub size: u64,
    pub allocation: MaxwellAllocationId,
    pub mapping: MaxwellMappingId,
    pub generation: MappingGeneration,
    pub backing_offset: u64,
    pub permissions: MemoryPermissions,
    pub page_size: u32,
    pub kind: u8,
    pub cacheable: bool,
    pub fixed: bool,
    pub sparse: bool,
    pub backing_segments: usize,
    pub first_physical: ArenaOffset,
    pub last_physical: ArenaOffset,
}

impl Display for MaxwellMappingDiagnostic {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{} size=0x{:x} {} {} {} backing-offset=0x{:x} permissions=0x{:x} \
             page-size=0x{:x} kind=0x{:02x} cacheable={} fixed={} sparse={} \
             backing-segments={} first-physical=0x{:x} last-physical=0x{:x}",
            self.offset,
            self.size,
            self.allocation,
            self.mapping,
            self.generation,
            self.backing_offset,
            self.permissions.bits(),
            self.page_size,
            self.kind,
            self.cacheable,
            self.fixed,
            self.sparse,
            self.backing_segments,
            self.first_physical.get(),
            self.last_physical.get(),
        )
    }
}

/// Deterministic, bounded snapshot of one address space's active mappings.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaxwellMappingDump {
    pub address_space: MaxwellAddressSpaceId,
    pub profile: GpuProfileId,
    pub generation: MappingGeneration,
    pub total_mappings: usize,
    pub entries: Box<[MaxwellMappingDiagnostic]>,
}

impl MaxwellMappingDump {
    #[must_use]
    pub fn omitted_mappings(&self) -> usize {
        self.total_mappings.saturating_sub(self.entries.len())
    }
}

impl Display for MaxwellMappingDump {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{} profile={} {} mappings={} shown={}",
            self.address_space,
            self.profile,
            self.generation,
            self.total_mappings,
            self.entries.len(),
        )?;
        for (index, entry) in self.entries.iter().enumerate() {
            write!(formatter, "\nmapping[{index}] {entry}")?;
        }
        if self.omitted_mappings() != 0 {
            write!(
                formatter,
                "\nmappings-truncated={}",
                self.omitted_mappings()
            )?;
        }
        Ok(())
    }
}

/// Persistent Maxwell state created by opening `/dev/nvhost-as-gpu`.
///
/// Each instance retains the exact immutable capability profile selected when
/// it was created. Horizon owns only the descriptor and ioctl wire adapter.
pub struct MaxwellGpuAddressSpace {
    id: MaxwellAddressSpaceId,
    machine: Machine,
    memory: MaxwellGpuMemory,
    free_slots: Vec<u32>,
    profile: MaxwellGpuProfile,
    regions: Option<[MaxwellVaRegion; 2]>,
    reservations: BTreeMap<GpuVirtualAddress, MaxwellVaReservation>,
    mappings: BTreeMap<GpuVirtualAddress, MaxwellGpuMapping>,
    next_mapping_id: u64,
    mapping_generation: MappingGeneration,
}

impl std::fmt::Debug for MaxwellGpuAddressSpace {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MaxwellGpuAddressSpace")
            .field("id", &self.id)
            .field("machine", &self.machine.identity())
            .field("mappings", &self.mappings.len())
            .finish()
    }
}
impl Drop for MaxwellGpuAddressSpace {
    fn drop(&mut self) {
        let mut tables = self
            .memory
            .inner
            .mappings
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.memory
            .inner
            .initialized
            .store(false, Ordering::Release);
        for mapping in self.mappings.values() {
            mapping.current.store(false, Ordering::Release);
            tables.table.unmap(mapping.offset.get(), mapping.size);
        }
        tables.live.clear();
    }
}

impl MaxwellGpuAddressSpace {
    #[must_use]
    pub fn new(machine: Machine, id: MaxwellAddressSpaceId, profile: MaxwellGpuProfile) -> Self {
        Self {
            id,
            machine: machine.clone(),
            memory: MaxwellGpuMemory::new(machine.clone(), id, profile),
            free_slots: Vec::new(),
            profile,
            regions: None,
            reservations: BTreeMap::new(),
            mappings: BTreeMap::new(),
            next_mapping_id: 1,
            mapping_generation: MappingGeneration::INITIAL,
        }
    }

    pub fn memory(&self) -> &MaxwellGpuMemory {
        &self.memory
    }

    #[must_use]
    pub const fn id(&self) -> MaxwellAddressSpaceId {
        self.id
    }

    #[must_use]
    pub const fn profile(&self) -> MaxwellGpuProfile {
        self.profile
    }

    #[must_use]
    pub const fn profile_id(&self) -> GpuProfileId {
        self.profile.id()
    }

    #[must_use]
    pub const fn initialized(&self) -> bool {
        self.regions.is_some()
    }

    /// Constructs an address using this address space's immutable profile
    /// width.
    pub const fn address(&self, value: u64) -> Result<GpuVirtualAddress, GpuVirtualAddressError> {
        GpuVirtualAddress::try_new(value, self.profile.virtual_address().address_bits().bits())
    }

    /// Adds a byte offset using this address space's immutable profile width.
    pub const fn checked_add(
        &self,
        address: GpuVirtualAddress,
        byte_offset: u64,
    ) -> Result<GpuVirtualAddress, GpuVirtualAddressError> {
        address.checked_add(
            byte_offset,
            self.profile.virtual_address().address_bits().bits(),
        )
    }

    /// Initializes the small-page and big-page regions atomically.
    pub fn initialize(
        &mut self,
        request: MaxwellAddressSpaceInitialization,
    ) -> Result<(), MaxwellAddressSpaceError> {
        if self.initialized() {
            return Err(MaxwellAddressSpaceError::AlreadyInitialized);
        }

        let memory = self.profile.memory();
        let big_page_size = if request.big_page_size == 0 {
            memory.big_page_size().raw()
        } else {
            request.big_page_size
        };
        if !big_page_size.is_power_of_two()
            || memory.available_big_page_sizes().raw() & big_page_size == 0
        {
            return Err(MaxwellAddressSpaceError::InvalidBigPageSize {
                page_size: big_page_size,
            });
        }

        let explicit_geometry =
            request.va_range_start != 0 || request.va_range_end != 0 || request.va_range_split != 0;
        let (start, split, end) = if explicit_geometry {
            (
                request.va_range_start,
                request.va_range_split,
                request.va_range_end,
            )
        } else {
            (
                u64::from(big_page_size)
                    .checked_shl(DEFAULT_VA_START_SHIFT)
                    .ok_or(MaxwellAddressSpaceError::ArithmeticOverflow)?,
                DEFAULT_VA_SPLIT,
                DEFAULT_VA_END,
            )
        };
        let small_page_size = memory.small_page_size().raw();
        validate_geometry(
            self.profile,
            start,
            split,
            end,
            small_page_size,
            big_page_size,
        )?;

        let small_offset = self
            .address(start)
            .map_err(MaxwellAddressSpaceError::Address)?;
        let big_offset = self
            .address(split)
            .map_err(MaxwellAddressSpaceError::Address)?;
        let small_pages = (split - start) / u64::from(small_page_size);
        let big_pages = (end - split) / u64::from(big_page_size);
        self.regions = Some([
            MaxwellVaRegion {
                offset: small_offset,
                page_size: small_page_size,
                pages: small_pages,
            },
            MaxwellVaRegion {
                offset: big_offset,
                page_size: big_page_size,
                pages: big_pages,
            },
        ]);
        self.memory.inner.initialized.store(true, Ordering::Release);
        Ok(())
    }

    /// Returns the allocator-owned regions established at initialization.
    pub fn regions(&self) -> Result<[MaxwellVaRegion; 2], MaxwellAddressSpaceError> {
        self.regions.ok_or(MaxwellAddressSpaceError::NotInitialized)
    }

    /// Reserves a fixed or first-fit GPU virtual range.
    pub fn reserve(
        &mut self,
        pages: u32,
        page_size: u32,
        flags: u32,
        align_or_offset: u64,
    ) -> Result<MaxwellVaReservation, MaxwellAddressSpaceError> {
        let regions = self.regions()?;
        if pages == 0 {
            return Err(MaxwellAddressSpaceError::InvalidPageCount);
        }
        if flags & !ALLOC_SPACE_KNOWN_FLAGS != 0 {
            return Err(MaxwellAddressSpaceError::InvalidReservationFlags { flags });
        }
        let region = regions
            .into_iter()
            .find(|region| region.page_size == page_size)
            .ok_or(MaxwellAddressSpaceError::InvalidPageSize { page_size })?;
        let sparse = flags & ALLOC_SPACE_SPARSE != 0;
        if sparse && page_size != regions[1].page_size {
            return Err(MaxwellAddressSpaceError::SparseSmallPagesUnsupported);
        }
        let size = u64::from(pages)
            .checked_mul(u64::from(page_size))
            .ok_or(MaxwellAddressSpaceError::ArithmeticOverflow)?;

        let fixed = flags & ALLOC_SPACE_FIXED != 0;
        let offset = if fixed {
            if align_or_offset & (u64::from(page_size) - 1) != 0 {
                return Err(MaxwellAddressSpaceError::MisalignedAddress {
                    address: align_or_offset,
                    alignment: u64::from(page_size),
                });
            }
            let end = align_or_offset
                .checked_add(size)
                .ok_or(MaxwellAddressSpaceError::ArithmeticOverflow)?;
            if !region.contains(align_or_offset, end)? {
                return Err(MaxwellAddressSpaceError::OutsideVaRegion);
            }
            if self.overlaps_occupied(align_or_offset, end)? {
                return Err(MaxwellAddressSpaceError::OverlappingReservation);
            }
            align_or_offset
        } else {
            let alignment = align_or_offset.max(u64::from(page_size));
            if !alignment.is_power_of_two() || !alignment.is_multiple_of(u64::from(page_size)) {
                return Err(MaxwellAddressSpaceError::InvalidAlignment { alignment });
            }
            self.find_first_fit(region, size, alignment)?
                .ok_or(MaxwellAddressSpaceError::InsufficientAddressSpace)?
        };
        let offset = self
            .address(offset)
            .map_err(MaxwellAddressSpaceError::Address)?;
        let reservation = MaxwellVaReservation {
            offset,
            pages,
            page_size,
            sparse,
        };
        self.reservations.insert(offset, reservation);
        Ok(reservation)
    }

    /// Releases exactly one reservation without accepting partial frees.
    pub fn free(
        &mut self,
        offset: GpuVirtualAddress,
        pages: u32,
        page_size: u32,
    ) -> Result<MaxwellVaReservation, MaxwellAddressSpaceError> {
        let shared = Arc::clone(&self.memory.inner);
        let mut tables = shared
            .mappings
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        self.regions()?;
        let reservation = self
            .reservations
            .get(&offset)
            .copied()
            .ok_or(MaxwellAddressSpaceError::UnknownReservation)?;
        if reservation.pages != pages || reservation.page_size != page_size {
            return Err(MaxwellAddressSpaceError::ReservationShapeMismatch);
        }
        let reservation_end = reservation.end()?;
        let mappings = self
            .mappings
            .iter()
            .filter_map(|(mapping_offset, mapping)| {
                (mapping.offset.get() >= offset.get()
                    && mapping.end().is_ok_and(|end| end <= reservation_end))
                .then_some(*mapping_offset)
            })
            .collect::<Vec<_>>();
        if !mappings.is_empty() {
            self.advance_mapping_generation()?;
            for mapping_offset in mappings {
                self.remove_mapping(&mut tables, mapping_offset);
            }
        }
        self.reservations.remove(&offset);
        Ok(reservation)
    }

    /// Installs physical allocation bytes at a fixed or allocated GPU VA.
    ///
    /// Switch 1 map flags, default-kind selection, and fixed/allocated forms:
    /// https://github.com/switchbrew/libnx/blob/dbcc1beafc6b47b5ffbeb8ba82463a7d45da40bb/nx/source/nvidia/ioctl/nvhost-as-gpu.c#L51-L86
    ///
    /// The public frontend reference also records page-size selection and the
    /// requirement that fixed mappings belong to an allocation:
    /// https://github.com/yuzu-emu-mirror/yuzu-mainline/blob/2d2522693e7d453bf10a8246f704350b69e12ebc/src/core/hle/service/nvdrv/devices/nvhost_as_gpu.cpp
    pub fn map(
        &mut self,
        request: MaxwellMapRequest,
    ) -> Result<MaxwellGpuMapping, MaxwellAddressSpaceError> {
        let shared = Arc::clone(&self.memory.inner);
        let mut tables = shared
            .mappings
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        let regions = self.regions()?;
        validate_backing(
            &self.machine,
            &request.backing,
            request.backing_offset,
            request.size,
            request.permissions,
        )?;
        validate_kind(request.kind)?;
        if request.allocation_alignment == 0 || !request.allocation_alignment.is_power_of_two() {
            return Err(MaxwellAddressSpaceError::InvalidAlignment {
                alignment: u64::from(request.allocation_alignment),
            });
        }

        let page_size = if request.page_size == 0 {
            let big_page_size = regions[1].page_size;
            if request.allocation_alignment >= big_page_size
                && request
                    .backing_offset
                    .is_multiple_of(u64::from(big_page_size))
                && request.size.is_multiple_of(u64::from(big_page_size))
            {
                big_page_size
            } else {
                regions[0].page_size
            }
        } else {
            request.page_size
        };
        let region = regions
            .into_iter()
            .find(|region| region.page_size == page_size)
            .ok_or(MaxwellAddressSpaceError::InvalidPageSize { page_size })?;
        if !request.backing_offset.is_multiple_of(u64::from(page_size))
            || !request.size.is_multiple_of(u64::from(page_size))
        {
            return Err(MaxwellAddressSpaceError::MisalignedBackingRange {
                offset: request.backing_offset,
                size: request.size,
                alignment: u64::from(page_size),
            });
        }

        let fixed = request.fixed_offset.is_some();
        let offset = if let Some(offset) = request.fixed_offset {
            if !offset.get().is_multiple_of(u64::from(page_size)) {
                return Err(MaxwellAddressSpaceError::MisalignedAddress {
                    address: offset.get(),
                    alignment: u64::from(page_size),
                });
            }
            let end = offset
                .get()
                .checked_add(request.size)
                .ok_or(MaxwellAddressSpaceError::ArithmeticOverflow)?;
            let reservation = self
                .reservation_containing(offset.get(), end)?
                .ok_or(MaxwellAddressSpaceError::OutsideReservation)?;
            if self.overlaps_mapping(offset.get(), end)? {
                return Err(MaxwellAddressSpaceError::OverlappingMapping);
            }
            debug_assert!(reservation.offset.get() <= offset.get());
            offset
        } else {
            let offset = self
                .find_first_fit(region, request.size, u64::from(page_size))?
                .ok_or(MaxwellAddressSpaceError::InsufficientAddressSpace)?;
            self.address(offset)
                .map_err(MaxwellAddressSpaceError::Address)?
        };

        let generation = self.next_mapping_generation()?;
        let (id, next_mapping_id) = self.next_mapping_identity()?;
        let sparse = self
            .reservation_containing(
                offset.get(),
                offset
                    .get()
                    .checked_add(request.size)
                    .ok_or(MaxwellAddressSpaceError::ArithmeticOverflow)?,
            )?
            .is_some_and(|reservation| reservation.sparse);
        let physical = PhysicalMapping::new(&request.backing)
            .map_err(|_| MaxwellAddressSpaceError::InvalidBackingRange)?;
        let pins = capture_subrange(&request.backing, request.backing_offset, request.size)
            .map_err(|_| MaxwellAddressSpaceError::InvalidBackingRange)?;
        let mapping = MaxwellGpuMapping {
            owner: Arc::downgrade(&self.memory.inner),
            id,
            unmap_offset: offset,
            offset,
            size: request.size,
            allocation: request.allocation,
            physical,
            current: Arc::new(AtomicBool::new(true)),
            slot: 0,
            backing_offset: request.backing_offset,
            permissions: request.permissions,
            page_size,
            kind: request.kind,
            cacheable: request.cacheable,
            generation,
            fixed,
            sparse,
        };
        let mapping = self.install_mapping(&mut tables, mapping, pins);
        self.next_mapping_id = next_mapping_id;
        self.mapping_generation = generation;
        self.memory
            .inner
            .generation
            .store(generation.get(), Ordering::Release);
        Ok(mapping)
    }

    /// Removes every active segment belonging to the mapping at `offset`.
    ///
    /// Returned descriptors do not retain RAM. Accepted physical operations
    /// keep their own exact pins. Unknown offsets are a guest-visible no-op.
    pub fn unmap(
        &mut self,
        offset: GpuVirtualAddress,
    ) -> Result<Vec<MaxwellGpuMapping>, MaxwellAddressSpaceError> {
        let shared = Arc::clone(&self.memory.inner);
        let mut tables = shared
            .mappings
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        self.regions()?;
        let Some(id) = self
            .mappings
            .values()
            .find(|mapping| mapping.unmap_offset == offset)
            .map(|mapping| mapping.id)
        else {
            return Ok(Vec::new());
        };
        let removed = self
            .mappings
            .iter()
            .filter_map(|(mapping_offset, mapping)| (mapping.id == id).then_some(*mapping_offset))
            .collect::<Vec<_>>();
        self.advance_mapping_generation()?;
        Ok(removed
            .into_iter()
            .filter_map(|mapping_offset| self.remove_mapping(&mut tables, mapping_offset))
            .collect())
    }

    /// Atomically replaces big-page subranges of sparse reservations.
    ///
    /// The 20-byte Switch 1 remap entry and sparse-reservation restriction are
    /// versioned here:
    /// https://switchbrew.org/w/index.php?title=NV_services&oldid=14790#NVGPU_AS_IOCTL_REMAP
    pub fn remap_sparse(
        &mut self,
        requests: Vec<MaxwellSparseRemapRequest>,
    ) -> Result<(), MaxwellAddressSpaceError> {
        let shared = Arc::clone(&self.memory.inner);
        let mut tables = shared
            .mappings
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        let regions = self.regions()?;
        let page_size = regions[1].page_size;
        if requests.is_empty() {
            return Err(MaxwellAddressSpaceError::EmptyRemap);
        }

        let mut spans = Vec::with_capacity(requests.len());
        for request in &requests {
            let end = validate_sparse_remap_request(self, request, page_size)?;
            if spans
                .iter()
                .any(|(start, prior_end)| request.offset.get() < *prior_end && *start < end)
            {
                return Err(MaxwellAddressSpaceError::OverlappingRemapEntries);
            }
            spans.push((request.offset.get(), end));
        }

        let generation = self.next_mapping_generation()?;
        let new_mapping_count = requests
            .iter()
            .filter(|request| request.mapping.is_some())
            .count();
        let next_mapping_id = self
            .next_mapping_id
            .checked_add(
                u64::try_from(new_mapping_count)
                    .map_err(|_| MaxwellAddressSpaceError::MappingIdentityExhausted)?,
            )
            .ok_or(MaxwellAddressSpaceError::MappingIdentityExhausted)?;
        let mut mapping_id = self.next_mapping_id;
        // Validate all guest entries before changing the table. Only affected
        // mappings are split; unrelated address-space state is never cloned.
        for request in requests {
            let end = request.offset.get() + request.size;
            self.replace_mapping_range(&mut tables, request.offset.get(), end, generation)?;
            if let Some(source) = request.mapping {
                let physical = PhysicalMapping::new(&source.backing)
                    .map_err(|_| MaxwellAddressSpaceError::InvalidBackingRange)?;
                let pins = capture_subrange(&source.backing, source.backing_offset, request.size)
                    .map_err(|_| MaxwellAddressSpaceError::InvalidBackingRange)?;
                let mapping = MaxwellGpuMapping {
                    owner: Arc::downgrade(&self.memory.inner),
                    id: MaxwellMappingId(mapping_id),
                    unmap_offset: request.offset,
                    offset: request.offset,
                    size: request.size,
                    allocation: source.allocation,
                    physical,
                    current: Arc::new(AtomicBool::new(true)),
                    slot: 0,
                    backing_offset: source.backing_offset,
                    permissions: source.permissions,
                    page_size,
                    kind: source.kind,
                    cacheable: source.cacheable,
                    generation,
                    fixed: true,
                    sparse: true,
                };
                mapping_id += 1;
                self.install_mapping(&mut tables, mapping, pins);
            }
        }
        self.next_mapping_id = next_mapping_id;
        self.mapping_generation = generation;
        self.memory
            .inner
            .generation
            .store(generation.get(), Ordering::Release);
        Ok(())
    }

    /// Changes the kind/cacheability of a checked subrange of one mapping.
    pub fn modify_mapping(
        &mut self,
        unmap_offset: GpuVirtualAddress,
        within_mapping: u64,
        size: u64,
        kind: u8,
        cacheable: bool,
    ) -> Result<(), MaxwellAddressSpaceError> {
        let shared = Arc::clone(&self.memory.inner);
        let mut tables = shared
            .mappings
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        self.regions()?;
        validate_kind(kind)?;
        if size == 0 {
            return Err(MaxwellAddressSpaceError::EmptyMapping);
        }
        let Some(id) = self
            .mappings
            .values()
            .find(|mapping| mapping.unmap_offset == unmap_offset)
            .map(|mapping| mapping.id)
        else {
            return Err(MaxwellAddressSpaceError::UnknownMapping);
        };
        let start = unmap_offset
            .get()
            .checked_add(within_mapping)
            .ok_or(MaxwellAddressSpaceError::ArithmeticOverflow)?;
        let end = start
            .checked_add(size)
            .ok_or(MaxwellAddressSpaceError::ArithmeticOverflow)?;
        let group = self
            .mappings
            .values()
            .filter(|mapping| mapping.id == id)
            .collect::<Vec<_>>();
        let page_size = group
            .first()
            .map(|mapping| mapping.page_size)
            .ok_or(MaxwellAddressSpaceError::UnknownMapping)?;
        if !start.is_multiple_of(u64::from(page_size)) || !size.is_multiple_of(u64::from(page_size))
        {
            return Err(MaxwellAddressSpaceError::MisalignedAddress {
                address: start,
                alignment: u64::from(page_size),
            });
        }
        let mut covered = start;
        for mapping in &group {
            let mapping_end = mapping.end()?;
            if mapping_end <= covered || mapping.offset.get() >= end {
                continue;
            }
            if mapping.offset.get() > covered {
                return Err(MaxwellAddressSpaceError::PartialMapping);
            }
            covered = mapping_end.min(end);
            if covered == end {
                break;
            }
        }
        if covered != end {
            return Err(MaxwellAddressSpaceError::PartialMapping);
        }

        let generation = self.next_mapping_generation()?;
        let address_bits = self.profile.virtual_address().address_bits().bits();
        let affected = self
            .mappings
            .iter()
            .filter_map(|(offset, mapping)| {
                mapping
                    .end()
                    .ok()
                    .filter(|mapping_end| {
                        mapping.id == id && start < *mapping_end && mapping.offset.get() < end
                    })
                    .map(|_| *offset)
            })
            .collect::<Vec<_>>();
        for offset in affected {
            let mapping = self
                .mappings
                .get(&offset)
                .cloned()
                .ok_or(MaxwellAddressSpaceError::UnknownMapping)?;
            let pins = mapping
                .physical
                .pin(mapping.backing_offset, mapping.size)
                .map_err(|_| MaxwellAddressSpaceError::InvalidBackingRange)?;
            self.remove_mapping(&mut tables, offset);
            let _pins = pins;
            let mapping_start = mapping.offset.get();
            let mapping_end = mapping.end()?;
            let changed_start = mapping_start.max(start);
            let changed_end = mapping_end.min(end);
            if mapping_start < changed_start {
                let mut left = mapping.clone();
                left.size = changed_start - mapping_start;
                left.generation = generation;
                self.install_split(&mut tables, left)?;
            }
            let mut changed = mapping.clone();
            changed.offset = GpuVirtualAddress::try_new(changed_start, address_bits)
                .map_err(MaxwellAddressSpaceError::Address)?;
            changed.backing_offset = changed
                .backing_offset
                .checked_add(changed_start - mapping_start)
                .ok_or(MaxwellAddressSpaceError::ArithmeticOverflow)?;
            changed.size = changed_end - changed_start;
            changed.kind = kind;
            changed.cacheable = cacheable;
            changed.generation = generation;
            self.install_split(&mut tables, changed)?;
            if changed_end < mapping_end {
                let mut right = mapping;
                right.offset = GpuVirtualAddress::try_new(changed_end, address_bits)
                    .map_err(MaxwellAddressSpaceError::Address)?;
                right.backing_offset = right
                    .backing_offset
                    .checked_add(changed_end - mapping_start)
                    .ok_or(MaxwellAddressSpaceError::ArithmeticOverflow)?;
                right.size = mapping_end - changed_end;
                right.generation = generation;
                self.install_split(&mut tables, right)?;
            }
        }
        self.mapping_generation = generation;
        self.memory
            .inner
            .generation
            .store(generation.get(), Ordering::Release);
        Ok(())
    }

    #[must_use]
    pub fn mapping(&self, offset: GpuVirtualAddress) -> Option<MaxwellGpuMapping> {
        self.mappings.get(&offset).cloned()
    }

    pub fn machine(&self) -> &Machine {
        &self.machine
    }

    /// Captures active mappings in GPU-VA order with a hard size bound.
    #[must_use]
    pub fn mapping_dump(&self, requested_entries: usize) -> MaxwellMappingDump {
        let limit = requested_entries.min(MAX_MAPPING_DUMP_ENTRIES);
        let entries = self
            .mappings
            .values()
            .take(limit)
            .map(|mapping| {
                let physical = mapping
                    .physical
                    .slice(mapping.backing_offset, mapping.size)
                    .expect("live mapping has valid physical bounds");
                let spans = physical.spans();
                let first = spans.first().expect("live mapping is nonempty");
                let last = spans.last().unwrap();
                MaxwellMappingDiagnostic {
                    offset: mapping.offset,
                    size: mapping.size,
                    allocation: mapping.allocation,
                    mapping: mapping.id,
                    generation: mapping.generation,
                    backing_offset: mapping.backing_offset,
                    permissions: mapping.permissions,
                    page_size: mapping.page_size,
                    kind: mapping.kind,
                    cacheable: mapping.cacheable,
                    fixed: mapping.fixed,
                    sparse: mapping.sparse,
                    backing_segments: spans.len(),
                    first_physical: first.start,
                    last_physical: ArenaOffset::new(last.start.get() + last.bytes - 1),
                }
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        MaxwellMappingDump {
            address_space: self.id,
            profile: self.profile_id(),
            generation: self.mapping_generation,
            total_mappings: self.mappings.len(),
            entries,
        }
    }

    #[must_use]
    pub fn mapping_count(&self) -> usize {
        self.mappings.len()
    }

    #[must_use]
    pub const fn mapping_generation(&self) -> MappingGeneration {
        self.mapping_generation
    }

    #[must_use]
    pub fn reservation(&self, offset: GpuVirtualAddress) -> Option<MaxwellVaReservation> {
        self.reservations.get(&offset).copied()
    }

    #[must_use]
    pub fn reservation_count(&self) -> usize {
        self.reservations.len()
    }

    fn overlaps_reservation(&self, start: u64, end: u64) -> Result<bool, MaxwellAddressSpaceError> {
        let end_address = self
            .address(end - 1)
            .map_err(MaxwellAddressSpaceError::Address)?;
        self.reservations
            .range(..=end_address)
            .next_back()
            .map_or(Ok(false), |(_, reservation)| Ok(reservation.end()? > start))
    }

    fn overlaps_mapping(&self, start: u64, end: u64) -> Result<bool, MaxwellAddressSpaceError> {
        let end_address = self
            .address(end - 1)
            .map_err(MaxwellAddressSpaceError::Address)?;
        self.mappings
            .range(..=end_address)
            .next_back()
            .map_or(Ok(false), |(_, mapping)| Ok(mapping.end()? > start))
    }

    fn overlaps_occupied(&self, start: u64, end: u64) -> Result<bool, MaxwellAddressSpaceError> {
        Ok(self.overlaps_reservation(start, end)? || self.overlaps_mapping(start, end)?)
    }

    fn reservation_containing(
        &self,
        start: u64,
        end: u64,
    ) -> Result<Option<MaxwellVaReservation>, MaxwellAddressSpaceError> {
        let address = self
            .address(start)
            .map_err(MaxwellAddressSpaceError::Address)?;
        match self.reservations.range(..=address).next_back() {
            Some((_, reservation)) if end <= reservation.end()? => Ok(Some(*reservation)),
            _ => Ok(None),
        }
    }

    fn find_first_fit(
        &self,
        region: MaxwellVaRegion,
        size: u64,
        alignment: u64,
    ) -> Result<Option<u64>, MaxwellAddressSpaceError> {
        let region_end = region.end()?;
        let mut candidate = align_up(region.offset.get(), alignment)?;
        let mut reservations = self.reservations.values().peekable();
        let mut mappings = self.mappings.values().peekable();
        loop {
            let candidate_end = candidate
                .checked_add(size)
                .ok_or(MaxwellAddressSpaceError::ArithmeticOverflow)?;
            if candidate_end > region_end {
                return Ok(None);
            }
            let next_reservation = match (reservations.peek(), mappings.peek()) {
                (Some(reservation), Some(mapping)) => reservation.offset <= mapping.offset,
                (Some(_), None) => true,
                (None, Some(_)) => false,
                (None, None) => return Ok(Some(candidate)),
            };
            let (start, end) = if next_reservation {
                let reservation = reservations.next().unwrap();
                (reservation.offset.get(), reservation.end()?)
            } else {
                let mapping = mappings.next().unwrap();
                (mapping.offset.get(), mapping.end()?)
            };
            if end <= candidate {
                continue;
            }
            if start >= candidate_end {
                return Ok(Some(candidate));
            }
            candidate = align_up(end, alignment)?;
        }
    }

    fn install_mapping(
        &mut self,
        tables: &mut memory::Mappings,
        mut descriptor: MaxwellGpuMapping,
        pins: Vec<AllocationRange>,
    ) -> MaxwellGpuMapping {
        let live_slots = &mut tables.live;
        let slot = self.free_slots.pop().unwrap_or_else(|| {
            let slot =
                u32::try_from(live_slots.len()).expect("GPU mappings exceed addressable RAM pages");
            live_slots.push(None);
            slot
        });
        descriptor.slot = slot;
        let mut address = descriptor.offset.get();
        for pin in &pins {
            for extent in pin.physical_extents() {
                tables.table.map(
                    address,
                    extent.bytes,
                    table::Entry {
                        allocation: pin.allocation().identity(),
                        physical_page: (extent.start.get() >> 12) as u32,
                        slot,
                    },
                );
                address += extent.bytes;
            }
        }
        self.mappings.insert(descriptor.offset, descriptor.clone());
        live_slots[slot as usize] = Some(LiveMapping {
            descriptor: descriptor.clone(),
            pins,
        });
        descriptor
    }

    fn remove_mapping(
        &mut self,
        tables: &mut memory::Mappings,
        offset: GpuVirtualAddress,
    ) -> Option<MaxwellGpuMapping> {
        let mapping = self.mappings.remove(&offset)?;
        mapping.current.store(false, Ordering::Release);
        tables.table.unmap(mapping.offset.get(), mapping.size);
        let live = tables.live[mapping.slot as usize]
            .take()
            .expect("active table slot");
        debug_assert!(!live.pins.is_empty());
        drop(live);
        self.free_slots.push(mapping.slot);
        Some(mapping)
    }

    fn install_split(
        &mut self,
        tables: &mut memory::Mappings,
        mut mapping: MaxwellGpuMapping,
    ) -> Result<(), MaxwellAddressSpaceError> {
        let pins = mapping
            .physical
            .pin(mapping.backing_offset, mapping.size)
            .map_err(|_| MaxwellAddressSpaceError::InvalidBackingRange)?;
        mapping.current = Arc::new(AtomicBool::new(true));
        self.install_mapping(tables, mapping, pins);
        Ok(())
    }

    fn replace_mapping_range(
        &mut self,
        tables: &mut memory::Mappings,
        start: u64,
        end: u64,
        generation: MappingGeneration,
    ) -> Result<(), MaxwellAddressSpaceError> {
        let overlapping: Vec<_> = self
            .mappings
            .values()
            .filter(|mapping| {
                mapping.offset.get() < end && mapping.offset.get() + mapping.size > start
            })
            .cloned()
            .collect();
        for mapping in overlapping {
            let pins = mapping
                .physical
                .pin(mapping.backing_offset, mapping.size)
                .map_err(|_| MaxwellAddressSpaceError::InvalidBackingRange)?;
            let mapping_end = mapping.offset.get() + mapping.size;
            self.remove_mapping(tables, mapping.offset);
            if mapping.offset.get() < start {
                let mut left = mapping.clone();
                left.size = start - mapping.offset.get();
                left.generation = generation;
                self.install_split(tables, left)?;
            }
            if mapping_end > end {
                let mut right = mapping.clone();
                right.offset = self
                    .address(end)
                    .map_err(MaxwellAddressSpaceError::Address)?;
                right.size = mapping_end - end;
                right.backing_offset += end - mapping.offset.get();
                right.generation = generation;
                self.install_split(tables, right)?;
            }
            drop(pins);
        }
        Ok(())
    }

    fn next_mapping_generation(&self) -> Result<MappingGeneration, MaxwellAddressSpaceError> {
        self.mapping_generation
            .next()
            .map_err(MaxwellAddressSpaceError::GenerationExhausted)
    }

    fn advance_mapping_generation(&mut self) -> Result<(), MaxwellAddressSpaceError> {
        self.mapping_generation = self.next_mapping_generation()?;
        self.memory
            .inner
            .generation
            .store(self.mapping_generation.get(), Ordering::Release);
        Ok(())
    }

    fn next_mapping_identity(&self) -> Result<(MaxwellMappingId, u64), MaxwellAddressSpaceError> {
        let next = self
            .next_mapping_id
            .checked_add(1)
            .ok_or(MaxwellAddressSpaceError::MappingIdentityExhausted)?;
        Ok((MaxwellMappingId(self.next_mapping_id), next))
    }
}

/// A verified invalid state or argument at the Maxwell semantic boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellAddressSpaceError {
    AlreadyInitialized,
    NotInitialized,
    InvalidBigPageSize {
        page_size: u32,
    },
    InvalidGeometry,
    InvalidPageCount,
    InvalidPageSize {
        page_size: u32,
    },
    InvalidReservationFlags {
        flags: u32,
    },
    InvalidAlignment {
        alignment: u64,
    },
    MisalignedAddress {
        address: u64,
        alignment: u64,
    },
    OutsideVaRegion,
    OutsideReservation,
    OverlappingReservation,
    OverlappingMapping,
    OverlappingRemapEntries,
    UnknownReservation,
    ReservationShapeMismatch,
    SparseSmallPagesUnsupported,
    NonSparseReservation,
    EmptyRemap,
    EmptyMapping,
    UnknownMapping,
    PartialMapping,
    InvalidBackingRange,
    InvalidMappingPermissions,
    InvalidKind {
        kind: u8,
    },
    MisalignedBackingRange {
        offset: u64,
        size: u64,
        alignment: u64,
    },
    InsufficientAddressSpace,
    MappingIdentityExhausted,
    ArithmeticOverflow,
    Address(GpuVirtualAddressError),
    GenerationExhausted(MappingGenerationExhausted),
}

impl Display for MaxwellAddressSpaceError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for MaxwellAddressSpaceError {}

/// Failure to resolve or access bytes through the GPU virtual address space.
#[derive(Clone, Debug)]
pub enum MaxwellGpuAccessError {
    NotInitialized,
    EmptyRange,
    InvalidPermissions,
    ArithmeticOverflow,
    WrongAddressSpace {
        expected: MaxwellAddressSpaceId,
        actual: MaxwellAddressSpaceId,
    },
    Address(GpuVirtualAddressError),
    UnmappedAddress {
        address: GpuVirtualAddress,
    },
    PermissionDenied {
        address: GpuVirtualAddress,
        required: MemoryPermissions,
        available: MemoryPermissions,
    },
    OutputSizeMismatch {
        expected: u64,
        actual: u64,
    },
    StaleMapping {
        mapping: MaxwellMappingId,
        generation: MappingGeneration,
    },
    Backing(Arc<MemoryError>),
}

impl Display for MaxwellGpuAccessError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotInitialized => formatter.write_str("GPU address space is not initialized"),
            Self::EmptyRange => formatter.write_str("GPU virtual range is empty"),
            Self::InvalidPermissions => {
                formatter.write_str("GPU virtual range requires no permissions")
            }
            Self::ArithmeticOverflow => {
                formatter.write_str("GPU virtual range arithmetic overflow")
            }
            Self::WrongAddressSpace { expected, actual } => write!(
                formatter,
                "resolved GPU range belongs to {expected}, not {actual}"
            ),
            Self::Address(error) => write!(formatter, "invalid GPU virtual address: {error}"),
            Self::UnmappedAddress { address } => {
                write!(formatter, "GPU virtual address is unmapped: {address}")
            }
            Self::PermissionDenied {
                address,
                required,
                available,
            } => write!(
                formatter,
                "GPU virtual permission denied at {address}: required=0x{:x} available=0x{:x}",
                required.bits(),
                available.bits(),
            ),
            Self::OutputSizeMismatch { expected, actual } => write!(
                formatter,
                "resolved GPU range output size mismatch: expected=0x{expected:x} actual=0x{actual:x}"
            ),
            Self::StaleMapping {
                mapping,
                generation,
            } => write!(
                formatter,
                "resolved GPU range is stale: {mapping} {generation}"
            ),
            Self::Backing(error) => write!(formatter, "physical memory access failed: {error}"),
        }
    }
}

impl std::error::Error for MaxwellGpuAccessError {}

fn validate_backing(
    machine: &Machine,
    backing: &[AllocationRange],
    offset: u64,
    size: u64,
    permissions: MemoryPermissions,
) -> Result<(), MaxwellAddressSpaceError> {
    if size == 0 {
        return Err(MaxwellAddressSpaceError::EmptyMapping);
    }
    let end = offset
        .checked_add(size)
        .ok_or(MaxwellAddressSpaceError::ArithmeticOverflow)?;
    let bytes = backing.iter().try_fold(0u64, |bytes, range| {
        if range.allocation().machine().identity() != machine.identity()
            || (range.offset() | range.bytes()) & 0xfff != 0
        {
            return Err(MaxwellAddressSpaceError::InvalidBackingRange);
        }
        bytes
            .checked_add(range.bytes())
            .ok_or(MaxwellAddressSpaceError::ArithmeticOverflow)
    })?;
    if end > bytes {
        return Err(MaxwellAddressSpaceError::InvalidBackingRange);
    }
    if permissions == MemoryPermissions::NONE {
        return Err(MaxwellAddressSpaceError::InvalidMappingPermissions);
    }
    Ok(())
}

fn validate_kind(kind: u8) -> Result<(), MaxwellAddressSpaceError> {
    if kind == INVALID_PTE_KIND {
        Err(MaxwellAddressSpaceError::InvalidKind { kind })
    } else {
        Ok(())
    }
}

fn validate_sparse_remap_request(
    address_space: &MaxwellGpuAddressSpace,
    request: &MaxwellSparseRemapRequest,
    page_size: u32,
) -> Result<u64, MaxwellAddressSpaceError> {
    if request.size == 0 {
        return Err(MaxwellAddressSpaceError::EmptyMapping);
    }
    if !request.offset.get().is_multiple_of(u64::from(page_size))
        || !request.size.is_multiple_of(u64::from(page_size))
    {
        return Err(MaxwellAddressSpaceError::MisalignedAddress {
            address: request.offset.get(),
            alignment: u64::from(page_size),
        });
    }
    let end = request
        .offset
        .get()
        .checked_add(request.size)
        .ok_or(MaxwellAddressSpaceError::ArithmeticOverflow)?;
    let reservation = address_space
        .reservation_containing(request.offset.get(), end)?
        .ok_or(MaxwellAddressSpaceError::OutsideReservation)?;
    if !reservation.sparse {
        return Err(MaxwellAddressSpaceError::NonSparseReservation);
    }
    if let Some(mapping) = &request.mapping {
        validate_kind(mapping.kind)?;
        validate_backing(
            &address_space.machine,
            &mapping.backing,
            mapping.backing_offset,
            request.size,
            mapping.permissions,
        )?;
        if !mapping.backing_offset.is_multiple_of(u64::from(page_size)) {
            return Err(MaxwellAddressSpaceError::MisalignedBackingRange {
                offset: mapping.backing_offset,
                size: request.size,
                alignment: u64::from(page_size),
            });
        }
    }
    Ok(end)
}

fn validate_geometry(
    profile: MaxwellGpuProfile,
    start: u64,
    split: u64,
    end: u64,
    small_page_size: u32,
    big_page_size: u32,
) -> Result<(), MaxwellAddressSpaceError> {
    if start >= split
        || split >= end
        || !start.is_multiple_of(u64::from(small_page_size))
        || !split.is_multiple_of(u64::from(small_page_size))
        || !split.is_multiple_of(u64::from(big_page_size))
        || !end.is_multiple_of(u64::from(big_page_size))
    {
        return Err(MaxwellAddressSpaceError::InvalidGeometry);
    }
    let address_bits = profile.virtual_address().address_bits().bits();
    let address_limit = 1_u64.checked_shl(u32::from(address_bits));
    if address_limit.is_some_and(|limit| end > limit) {
        return Err(MaxwellAddressSpaceError::InvalidGeometry);
    }
    Ok(())
}

fn align_up(value: u64, alignment: u64) -> Result<u64, MaxwellAddressSpaceError> {
    value
        .checked_add(alignment - 1)
        .map(|aligned| aligned & !(alignment - 1))
        .ok_or(MaxwellAddressSpaceError::ArithmeticOverflow)
}

#[cfg(test)]
mod tests;
