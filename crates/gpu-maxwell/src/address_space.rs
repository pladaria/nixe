//! Switch 1 Maxwell GPU virtual address-space semantics.

use std::collections::BTreeMap;
use std::fmt::{Display, Formatter};

use nixe_gpu::{GpuVirtualAddress, GpuVirtualAddressError};
use nixe_memory::{
    CanonicalBackingRange, CanonicalPageId, CanonicalRangeAccessError, GenerationExhausted,
    MappingGeneration, MemoryPermissions,
};

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

/// Fully decoded request to map retained canonical allocation bytes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaxwellMapRequest {
    pub allocation: MaxwellAllocationId,
    pub backing: CanonicalBackingRange,
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
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaxwellSparseRemapRequest {
    pub offset: GpuVirtualAddress,
    pub size: u64,
    pub mapping: Option<MaxwellSparseMapping>,
}

/// Canonical allocation bytes installed by a sparse-remap entry.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaxwellSparseMapping {
    pub allocation: MaxwellAllocationId,
    pub backing: CanonicalBackingRange,
    pub backing_offset: u64,
    pub kind: u8,
    pub cacheable: bool,
    pub permissions: MemoryPermissions,
}

/// One active GPU-virtual interpretation of retained canonical bytes.
///
/// Clones retain canonical pages. An in-flight consumer may therefore keep
/// this value after the active mapping is unmapped without retaining a CPU
/// pointer or observing freed host storage. New lookups use the address-space
/// table and no longer see the removed generation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaxwellGpuMapping {
    id: MaxwellMappingId,
    unmap_offset: GpuVirtualAddress,
    offset: GpuVirtualAddress,
    size: u64,
    allocation: MaxwellAllocationId,
    backing: CanonicalBackingRange,
    backing_offset: u64,
    permissions: MemoryPermissions,
    page_size: u32,
    kind: u8,
    cacheable: bool,
    generation: MappingGeneration,
    fixed: bool,
    sparse: bool,
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
    pub const fn backing(&self) -> &CanonicalBackingRange {
        &self.backing
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

/// Maximal contiguous portion of one active mapping beginning at a GPU VA.
///
/// The retained mapping contains canonical backing rather than a host pointer.
/// Consumers must validate the snapshot against its address space immediately
/// before access so an unmap or remap cannot authorize stale work.
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
    segments: Box<[MaxwellResolvedMapping]>,
}

impl MaxwellResolvedRange {
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
    pub first_page: CanonicalPageId,
    pub last_page: CanonicalPageId,
}

impl Display for MaxwellMappingDiagnostic {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{} size=0x{:x} {} {} {} backing-offset=0x{:x} permissions=0x{:x} \
             page-size=0x{:x} kind=0x{:02x} cacheable={} fixed={} sparse={} \
             backing-segments={} first-page=[{}] last-page=[{}]",
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
            self.first_page,
            self.last_page,
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
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaxwellGpuAddressSpace {
    id: MaxwellAddressSpaceId,
    profile: MaxwellGpuProfile,
    regions: Option<[MaxwellVaRegion; 2]>,
    reservations: BTreeMap<GpuVirtualAddress, MaxwellVaReservation>,
    mappings: BTreeMap<GpuVirtualAddress, MaxwellGpuMapping>,
    next_mapping_id: u64,
    mapping_generation: MappingGeneration,
}

impl MaxwellGpuAddressSpace {
    #[must_use]
    pub const fn new(id: MaxwellAddressSpaceId, profile: MaxwellGpuProfile) -> Self {
        Self {
            id,
            profile,
            regions: None,
            reservations: BTreeMap::new(),
            mappings: BTreeMap::new(),
            next_mapping_id: 1,
            mapping_generation: MappingGeneration::INITIAL,
        }
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
                self.mappings.remove(&mapping_offset);
            }
        }
        self.reservations.remove(&offset);
        Ok(reservation)
    }

    /// Maps retained canonical allocation bytes at a fixed or allocated GPU VA.
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
        let regions = self.regions()?;
        validate_backing(
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
        let mapping = MaxwellGpuMapping {
            id,
            unmap_offset: offset,
            offset,
            size: request.size,
            allocation: request.allocation,
            backing: request.backing,
            backing_offset: request.backing_offset,
            permissions: request.permissions,
            page_size,
            kind: request.kind,
            cacheable: request.cacheable,
            generation,
            fixed,
            sparse,
        };
        self.mappings.insert(offset, mapping.clone());
        self.next_mapping_id = next_mapping_id;
        self.mapping_generation = generation;
        Ok(mapping)
    }

    /// Removes every active segment belonging to the mapping at `offset`.
    ///
    /// The returned clones keep canonical pages alive for already retained
    /// work. Unknown offsets are a verified no-op in the public Switch
    /// frontend behavior.
    pub fn unmap(
        &mut self,
        offset: GpuVirtualAddress,
    ) -> Result<Vec<MaxwellGpuMapping>, MaxwellAddressSpaceError> {
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
            .filter_map(|mapping_offset| self.mappings.remove(&mapping_offset))
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
        let mut mappings = self.mappings.clone();
        for request in requests {
            let end = request
                .offset
                .get()
                .checked_add(request.size)
                .ok_or(MaxwellAddressSpaceError::ArithmeticOverflow)?;
            replace_mapping_range(
                &mut mappings,
                request.offset.get(),
                end,
                generation,
                self.profile.virtual_address().address_bits().bits(),
            )?;
            if let Some(source) = request.mapping {
                let id = MaxwellMappingId(mapping_id);
                mapping_id = mapping_id
                    .checked_add(1)
                    .ok_or(MaxwellAddressSpaceError::MappingIdentityExhausted)?;
                mappings.insert(
                    request.offset,
                    MaxwellGpuMapping {
                        id,
                        unmap_offset: request.offset,
                        offset: request.offset,
                        size: request.size,
                        allocation: source.allocation,
                        backing: source.backing,
                        backing_offset: source.backing_offset,
                        permissions: source.permissions,
                        page_size,
                        kind: source.kind,
                        cacheable: source.cacheable,
                        generation,
                        fixed: true,
                        sparse: true,
                    },
                );
            }
        }
        self.mappings = mappings;
        self.next_mapping_id = next_mapping_id;
        self.mapping_generation = generation;
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
        let mut mappings = self.mappings.clone();
        let affected = mappings
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
            let mapping = mappings
                .remove(&offset)
                .ok_or(MaxwellAddressSpaceError::UnknownMapping)?;
            let mapping_start = mapping.offset.get();
            let mapping_end = mapping.end()?;
            let changed_start = mapping_start.max(start);
            let changed_end = mapping_end.min(end);
            if mapping_start < changed_start {
                let mut left = mapping.clone();
                left.size = changed_start - mapping_start;
                left.generation = generation;
                mappings.insert(left.offset, left);
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
            mappings.insert(changed.offset, changed);
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
                mappings.insert(right.offset, right);
            }
        }
        self.mappings = mappings;
        self.mapping_generation = generation;
        Ok(())
    }

    #[must_use]
    pub fn mapping(&self, offset: GpuVirtualAddress) -> Option<MaxwellGpuMapping> {
        self.mappings.get(&offset).cloned()
    }

    /// Resolves the maximal active mapping suffix beginning at `offset`.
    pub fn resolve_maximal(
        &self,
        offset: GpuVirtualAddress,
        permissions: MemoryPermissions,
    ) -> Result<MaxwellResolvedMapping, MaxwellGpuAccessError> {
        self.validate_resolution_request(offset, permissions)?;
        let mapping = self
            .mappings
            .range(..=offset)
            .next_back()
            .map(|(_, mapping)| mapping)
            .filter(|mapping| mapping.end().is_ok_and(|end| offset.get() < end))
            .ok_or(MaxwellGpuAccessError::UnmappedAddress { address: offset })?;
        if !mapping.permissions.contains(permissions) {
            return Err(MaxwellGpuAccessError::PermissionDenied {
                address: offset,
                required: permissions,
                available: mapping.permissions,
            });
        }
        let within_mapping = offset
            .get()
            .checked_sub(mapping.offset.get())
            .ok_or(MaxwellGpuAccessError::ArithmeticOverflow)?;
        let size = mapping
            .size
            .checked_sub(within_mapping)
            .ok_or(MaxwellGpuAccessError::ArithmeticOverflow)?;
        let backing_offset = mapping
            .backing_offset
            .checked_add(within_mapping)
            .ok_or(MaxwellGpuAccessError::ArithmeticOverflow)?;
        Ok(MaxwellResolvedMapping {
            gpu_offset: offset,
            size,
            backing_offset,
            mapping: mapping.clone(),
        })
    }

    /// Resolves a complete GPU range into ordered retained mapping segments.
    ///
    /// Resolution is atomic: a hole, permission failure, or arithmetic error
    /// returns no partially usable range.
    pub fn resolve_range(
        &self,
        offset: GpuVirtualAddress,
        size: u64,
        permissions: MemoryPermissions,
    ) -> Result<MaxwellResolvedRange, MaxwellGpuAccessError> {
        self.validate_resolution_request(offset, permissions)?;
        if size == 0 {
            return Err(MaxwellGpuAccessError::EmptyRange);
        }
        self.checked_add(offset, size - 1)
            .map_err(MaxwellGpuAccessError::Address)?;

        let mut remaining = size;
        let mut cursor = offset;
        let mut segments = Vec::new();
        while remaining != 0 {
            let mut segment = self.resolve_maximal(cursor, permissions)?;
            segment.size = segment.size.min(remaining);
            remaining -= segment.size;
            if remaining != 0 {
                cursor = self
                    .checked_add(cursor, segment.size)
                    .map_err(MaxwellGpuAccessError::Address)?;
            }
            segments.push(segment);
        }
        Ok(MaxwellResolvedRange {
            address_space: self.id,
            offset,
            size,
            permissions,
            segments: segments.into_boxed_slice(),
        })
    }

    /// Copies a previously resolved range from canonical bytes.
    ///
    /// Every mapping snapshot is checked before the first backing access. This
    /// synchronous helper is suitable for deterministic validators and tests;
    /// it does not claim host GPU completion or guest-fence progress.
    pub fn read_resolved(
        &self,
        resolved: &MaxwellResolvedRange,
        output: &mut [u8],
    ) -> Result<(), MaxwellGpuAccessError> {
        let output_size =
            u64::try_from(output.len()).map_err(|_| MaxwellGpuAccessError::ArithmeticOverflow)?;
        if resolved.address_space != self.id {
            return Err(MaxwellGpuAccessError::WrongAddressSpace {
                expected: resolved.address_space,
                actual: self.id,
            });
        }
        if !resolved.permissions.contains(MemoryPermissions::READ) {
            return Err(MaxwellGpuAccessError::PermissionDenied {
                address: resolved.offset,
                required: MemoryPermissions::READ,
                available: resolved.permissions,
            });
        }
        if output_size != resolved.size {
            return Err(MaxwellGpuAccessError::OutputSizeMismatch {
                expected: resolved.size,
                actual: output_size,
            });
        }
        for segment in &resolved.segments {
            if !self.retained_mapping_is_current(&segment.mapping) {
                return Err(MaxwellGpuAccessError::StaleMapping {
                    mapping: segment.mapping.id,
                    generation: segment.mapping.generation,
                });
            }
        }

        let mut copied = 0_usize;
        for segment in &resolved.segments {
            let segment_size = usize::try_from(segment.size)
                .map_err(|_| MaxwellGpuAccessError::ArithmeticOverflow)?;
            let copied_end = copied
                .checked_add(segment_size)
                .ok_or(MaxwellGpuAccessError::ArithmeticOverflow)?;
            segment
                .mapping
                .backing
                .read(segment.backing_offset, &mut output[copied..copied_end])
                .map_err(MaxwellGpuAccessError::Backing)?;
            copied = copied_end;
        }
        Ok(())
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
                let (backing_segments, first_page, last_page) = mapping_backing_summary(mapping);
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
                    backing_segments,
                    first_page,
                    last_page,
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

    /// Reports whether a retained mapping still names the active generation.
    #[must_use]
    pub fn retained_mapping_is_current(&self, retained: &MaxwellGpuMapping) -> bool {
        self.mappings.values().any(|mapping| {
            mapping.id == retained.id
                && mapping.generation == retained.generation
                && mapping.offset == retained.offset
                && mapping.size == retained.size
        })
    }

    #[must_use]
    pub fn reservation(&self, offset: GpuVirtualAddress) -> Option<MaxwellVaReservation> {
        self.reservations.get(&offset).copied()
    }

    #[must_use]
    pub fn reservation_count(&self) -> usize {
        self.reservations.len()
    }

    fn validate_resolution_request(
        &self,
        offset: GpuVirtualAddress,
        permissions: MemoryPermissions,
    ) -> Result<(), MaxwellGpuAccessError> {
        if !self.initialized() {
            return Err(MaxwellGpuAccessError::NotInitialized);
        }
        self.address(offset.get())
            .map_err(MaxwellGpuAccessError::Address)?;
        if permissions == MemoryPermissions::NONE {
            return Err(MaxwellGpuAccessError::InvalidPermissions);
        }
        Ok(())
    }

    fn overlaps_reservation(&self, start: u64, end: u64) -> Result<bool, MaxwellAddressSpaceError> {
        for reservation in self.reservations.values().copied() {
            if start < reservation.end()? && reservation.offset.get() < end {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn overlaps_mapping(&self, start: u64, end: u64) -> Result<bool, MaxwellAddressSpaceError> {
        for mapping in self.mappings.values() {
            if start < mapping.end()? && mapping.offset.get() < end {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn overlaps_occupied(&self, start: u64, end: u64) -> Result<bool, MaxwellAddressSpaceError> {
        Ok(self.overlaps_reservation(start, end)? || self.overlaps_mapping(start, end)?)
    }

    fn reservation_containing(
        &self,
        start: u64,
        end: u64,
    ) -> Result<Option<MaxwellVaReservation>, MaxwellAddressSpaceError> {
        for reservation in self.reservations.values().copied() {
            if start >= reservation.offset.get() && end <= reservation.end()? {
                return Ok(Some(reservation));
            }
        }
        Ok(None)
    }

    fn find_first_fit(
        &self,
        region: MaxwellVaRegion,
        size: u64,
        alignment: u64,
    ) -> Result<Option<u64>, MaxwellAddressSpaceError> {
        let region_end = region.end()?;
        let mut candidate = align_up(region.offset.get(), alignment)?;
        loop {
            let candidate_end = candidate
                .checked_add(size)
                .ok_or(MaxwellAddressSpaceError::ArithmeticOverflow)?;
            if candidate_end > region_end {
                return Ok(None);
            }
            let mut overlapping_end = None;
            for (occupied_start, occupied_end) in self.occupied_ranges()? {
                if candidate < occupied_end && occupied_start < candidate_end {
                    overlapping_end = Some(
                        overlapping_end.map_or(occupied_end, |end: u64| end.max(occupied_end)),
                    );
                }
            }
            let Some(overlapping_end) = overlapping_end else {
                return Ok(Some(candidate));
            };
            candidate = align_up(overlapping_end, alignment)?;
        }
    }

    fn occupied_ranges(&self) -> Result<Vec<(u64, u64)>, MaxwellAddressSpaceError> {
        let mut ranges = Vec::with_capacity(self.reservations.len() + self.mappings.len());
        for reservation in self.reservations.values().copied() {
            ranges.push((reservation.offset.get(), reservation.end()?));
        }
        for mapping in self.mappings.values() {
            ranges.push((mapping.offset.get(), mapping.end()?));
        }
        Ok(ranges)
    }

    fn next_mapping_generation(&self) -> Result<MappingGeneration, MaxwellAddressSpaceError> {
        self.mapping_generation
            .next()
            .map_err(MaxwellAddressSpaceError::GenerationExhausted)
    }

    fn advance_mapping_generation(&mut self) -> Result<(), MaxwellAddressSpaceError> {
        self.mapping_generation = self.next_mapping_generation()?;
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
    GenerationExhausted(GenerationExhausted),
}

impl Display for MaxwellAddressSpaceError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for MaxwellAddressSpaceError {}

/// Failure to resolve or access bytes through the GPU virtual address space.
#[derive(Clone, Debug, Eq, PartialEq)]
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
    Backing(CanonicalRangeAccessError),
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
            Self::Backing(error) => write!(formatter, "canonical backing access failed: {error}"),
        }
    }
}

impl std::error::Error for MaxwellGpuAccessError {}

fn mapping_backing_summary(
    mapping: &MaxwellGpuMapping,
) -> (usize, CanonicalPageId, CanonicalPageId) {
    let mapped_end = mapping
        .backing_offset
        .checked_add(mapping.size)
        .expect("validated mapping backing range cannot overflow");
    let mut logical_start = 0_u64;
    let mut first_page = None;
    let mut last_page = None;
    let mut count = 0_usize;
    for segment in mapping.backing.segments() {
        let logical_end = logical_start
            .checked_add(segment.size())
            .expect("validated canonical backing range cannot overflow");
        if mapping.backing_offset < logical_end && logical_start < mapped_end {
            first_page.get_or_insert_with(|| segment.page());
            last_page = Some(segment.page());
            count += 1;
        }
        logical_start = logical_end;
        if logical_start >= mapped_end {
            break;
        }
    }
    (
        count,
        first_page.expect("validated mapping must touch canonical backing"),
        last_page.expect("validated mapping must touch canonical backing"),
    )
}

fn validate_backing(
    backing: &CanonicalBackingRange,
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
    if end > backing.size() {
        return Err(MaxwellAddressSpaceError::InvalidBackingRange);
    }
    if permissions == MemoryPermissions::NONE
        || backing
            .segments()
            .iter()
            .any(|segment| !segment.permissions().contains(permissions))
    {
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

fn replace_mapping_range(
    mappings: &mut BTreeMap<GpuVirtualAddress, MaxwellGpuMapping>,
    start: u64,
    end: u64,
    generation: MappingGeneration,
    address_bits: u8,
) -> Result<(), MaxwellAddressSpaceError> {
    let overlapping = mappings
        .iter()
        .filter_map(|(offset, mapping)| {
            mapping
                .end()
                .ok()
                .filter(|mapping_end| start < *mapping_end && mapping.offset.get() < end)
                .map(|_| *offset)
        })
        .collect::<Vec<_>>();
    for offset in overlapping {
        let mapping = mappings
            .remove(&offset)
            .ok_or(MaxwellAddressSpaceError::ArithmeticOverflow)?;
        let mapping_end = mapping.end()?;
        if mapping.offset.get() < start {
            let mut left = mapping.clone();
            left.size = start - mapping.offset.get();
            left.generation = generation;
            mappings.insert(left.offset, left);
        }
        if mapping_end > end {
            let mut right = mapping;
            let consumed = end - right.offset.get();
            right.offset = GpuVirtualAddress::try_new(end, address_bits)
                .map_err(MaxwellAddressSpaceError::Address)?;
            right.size = mapping_end - end;
            right.backing_offset = right
                .backing_offset
                .checked_add(consumed)
                .ok_or(MaxwellAddressSpaceError::ArithmeticOverflow)?;
            right.generation = generation;
            mappings.insert(right.offset, right);
        }
    }
    Ok(())
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
mod tests {
    use super::*;
    use crate::SWITCH_1_GM20B_PROFILE;
    use nixe_memory::CanonicalAllocation;

    fn address_space() -> MaxwellGpuAddressSpace {
        MaxwellGpuAddressSpace::new(MaxwellAddressSpaceId::new(1), SWITCH_1_GM20B_PROFILE)
    }

    fn backing(size: usize) -> (CanonicalAllocation, CanonicalBackingRange) {
        let allocation = CanonicalAllocation::zeroed(size, 0x1000).unwrap();
        let backing = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        (allocation, backing)
    }

    fn map_request(
        backing: CanonicalBackingRange,
        allocation: u64,
        size: u64,
    ) -> MaxwellMapRequest {
        MaxwellMapRequest {
            allocation: MaxwellAllocationId::new(allocation),
            backing,
            backing_offset: 0,
            size,
            allocation_alignment: 0x1000,
            page_size: 0,
            kind: 0,
            cacheable: false,
            permissions: MemoryPermissions::READ_WRITE,
            fixed_offset: None,
        }
    }

    #[test]
    fn address_space_retains_identity_and_immutable_profile() {
        let address_space = address_space();

        assert_eq!(address_space.id(), MaxwellAddressSpaceId::new(1));
        assert_eq!(address_space.profile(), SWITCH_1_GM20B_PROFILE);
        assert_eq!(address_space.profile_id(), SWITCH_1_GM20B_PROFILE.id());
        assert_eq!(
            address_space.id().to_string(),
            "gpu-address-space=0x0000000000000001"
        );
    }

    #[test]
    fn address_arithmetic_uses_the_bound_profile_width() {
        let address_space = address_space();
        let last_page = address_space.address(0x00ff_ffff_f000).unwrap();

        assert_eq!(
            address_space.checked_add(last_page, 0x1000),
            Err(GpuVirtualAddressError::AddressOutOfRange {
                value: 0x0100_0000_0000,
                bits: 40,
            })
        );
    }

    #[test]
    fn initialization_derives_two_regions_from_profile_and_state() {
        let mut address_space = address_space();
        address_space
            .initialize(MaxwellAddressSpaceInitialization::default())
            .unwrap();

        assert_eq!(
            address_space.regions().unwrap(),
            [
                MaxwellVaRegion {
                    offset: address_space.address(0x0800_0000).unwrap(),
                    page_size: 0x1000,
                    pages: 0x3f_8000,
                },
                MaxwellVaRegion {
                    offset: address_space.address(0x4_0000_0000).unwrap(),
                    page_size: 0x2_0000,
                    pages: 0xe_0000,
                },
            ]
        );
        assert_eq!(
            address_space.initialize(MaxwellAddressSpaceInitialization::default()),
            Err(MaxwellAddressSpaceError::AlreadyInitialized)
        );
    }

    #[test]
    fn alternate_big_page_size_changes_default_start_and_regions() {
        let mut address_space = address_space();
        address_space
            .initialize(MaxwellAddressSpaceInitialization {
                big_page_size: 0x1_0000,
                ..Default::default()
            })
            .unwrap();
        let regions = address_space.regions().unwrap();

        assert_eq!(regions[0].offset().get(), 0x0400_0000);
        assert_eq!(regions[1].page_size(), 0x1_0000);
        assert_eq!(regions[1].pages(), 0x1c_0000);
    }

    #[test]
    fn reservations_use_aligned_first_fit_and_exact_atomic_free() {
        let mut address_space = address_space();
        address_space
            .initialize(MaxwellAddressSpaceInitialization::default())
            .unwrap();
        let first = address_space.reserve(3, 0x1000, 0, 0x4000).unwrap();
        let fixed_offset = address_space.address(0x0801_0000).unwrap();
        let fixed = address_space
            .reserve(2, 0x1000, ALLOC_SPACE_FIXED, fixed_offset.get())
            .unwrap();
        let second = address_space.reserve(1, 0x1000, 0, 0x1000).unwrap();

        assert_eq!(first.offset().get(), 0x0800_0000);
        assert_eq!(second.offset().get(), 0x0800_3000);
        assert_eq!(fixed.offset(), fixed_offset);
        assert_eq!(address_space.reservation_count(), 3);
        assert_eq!(
            address_space.free(first.offset(), 2, first.page_size()),
            Err(MaxwellAddressSpaceError::ReservationShapeMismatch)
        );
        assert_eq!(address_space.reservation(first.offset()), Some(first));
        assert_eq!(
            address_space
                .free(first.offset(), first.pages(), first.page_size())
                .unwrap(),
            first
        );
        assert_eq!(address_space.reservation(first.offset()), None);
    }

    #[test]
    fn reservation_validation_rejects_partial_or_invalid_state() {
        let mut address_space = address_space();
        assert_eq!(
            address_space.reserve(1, 0x1000, 0, 0x1000),
            Err(MaxwellAddressSpaceError::NotInitialized)
        );
        address_space
            .initialize(MaxwellAddressSpaceInitialization::default())
            .unwrap();

        assert_eq!(
            address_space.reserve(0, 0x1000, 0, 0x1000),
            Err(MaxwellAddressSpaceError::InvalidPageCount)
        );
        assert_eq!(
            address_space.reserve(1, 0x1000, ALLOC_SPACE_SPARSE, 0x1000),
            Err(MaxwellAddressSpaceError::SparseSmallPagesUnsupported)
        );
        assert_eq!(
            address_space.reserve(1, 0x2000, 0, 0x2000),
            Err(MaxwellAddressSpaceError::InvalidPageSize { page_size: 0x2000 })
        );
        assert_eq!(
            address_space.reserve(1, 0x1000, ALLOC_SPACE_FIXED, 0x0800_0001),
            Err(MaxwellAddressSpaceError::MisalignedAddress {
                address: 0x0800_0001,
                alignment: 0x1000,
            })
        );
    }

    #[test]
    fn allocated_and_fixed_maps_retain_typed_canonical_state() {
        let mut address_space = address_space();
        address_space
            .initialize(MaxwellAddressSpaceInitialization::default())
            .unwrap();
        let (allocation, backing) = backing(0x4000);
        allocation.write(0, &[1, 2, 3, 4]).unwrap();

        let first = address_space
            .map(map_request(backing.clone(), 7, 0x2000))
            .unwrap();
        let reservation = address_space
            .reserve(2, 0x1000, ALLOC_SPACE_FIXED, 0x0801_0000)
            .unwrap();
        let mut fixed_request = map_request(backing.clone(), 7, 0x2000);
        fixed_request.fixed_offset = Some(reservation.offset());
        fixed_request.backing_offset = 0x2000;
        let second = address_space.map(fixed_request).unwrap();

        assert_ne!(first.offset(), second.offset());
        assert_eq!(first.allocation(), second.allocation());
        assert_eq!(
            first.backing().segments()[0].page(),
            backing.segments()[0].page()
        );
        assert_eq!(second.backing_offset(), 0x2000);
        assert_eq!(second.permissions(), MemoryPermissions::READ_WRITE);
        assert_eq!(second.page_size(), 0x1000);
        assert!(second.fixed());
        assert!(!second.sparse());
        assert!(second.generation() > first.generation());
    }

    #[test]
    fn aliases_share_backing_identity_and_unmap_retains_in_flight_bytes() {
        let mut address_space = address_space();
        address_space
            .initialize(MaxwellAddressSpaceInitialization::default())
            .unwrap();
        let allocation = CanonicalAllocation::zeroed(0x2000, 0x1000).unwrap();
        allocation.write(0, &[0xaa, 0xbb]).unwrap();
        let backing = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let retained = address_space
            .map(map_request(backing.clone(), 9, 0x2000))
            .unwrap();
        let alias = address_space.map(map_request(backing, 9, 0x2000)).unwrap();

        assert_ne!(retained.offset(), alias.offset());
        assert_eq!(
            retained.backing().segments()[0].page(),
            alias.backing().segments()[0].page()
        );
        assert!(address_space.retained_mapping_is_current(&retained));
        drop(allocation);

        let removed = address_space.unmap(retained.offset()).unwrap();
        assert_eq!(removed, vec![retained.clone()]);
        assert!(!address_space.retained_mapping_is_current(&retained));
        let mut bytes = [0; 2];
        retained.backing().read(0, &mut bytes).unwrap();
        assert_eq!(bytes, [0xaa, 0xbb]);
        assert!(address_space.retained_mapping_is_current(&alias));
    }

    #[test]
    fn invalid_maps_and_free_are_atomic() {
        let mut address_space = address_space();
        address_space
            .initialize(MaxwellAddressSpaceInitialization::default())
            .unwrap();
        let (_, backing) = backing(0x4000);
        let reservation = address_space
            .reserve(4, 0x1000, ALLOC_SPACE_FIXED, 0x0801_0000)
            .unwrap();
        let mut request = map_request(backing.clone(), 1, 0x2000);
        request.fixed_offset = Some(reservation.offset());
        let mapping = address_space.map(request.clone()).unwrap();
        let generation = address_space.mapping_generation();

        assert_eq!(
            address_space.map(request),
            Err(MaxwellAddressSpaceError::OverlappingMapping)
        );
        assert_eq!(address_space.mapping_generation(), generation);
        assert_eq!(address_space.mapping_count(), 1);
        assert_eq!(
            address_space.free(reservation.offset(), 3, 0x1000),
            Err(MaxwellAddressSpaceError::ReservationShapeMismatch)
        );
        assert_eq!(address_space.mapping(mapping.offset()), Some(mapping));
        address_space.free(reservation.offset(), 4, 0x1000).unwrap();
        assert_eq!(address_space.mapping_count(), 0);
    }

    #[test]
    fn sparse_remap_is_atomic_and_splits_replaced_ranges() {
        let mut address_space = address_space();
        address_space
            .initialize(MaxwellAddressSpaceInitialization::default())
            .unwrap();
        let (_, backing) = backing(0x80_000);
        let reservation = address_space
            .reserve(
                4,
                0x2_0000,
                ALLOC_SPACE_FIXED | ALLOC_SPACE_SPARSE,
                0x4_0000_0000,
            )
            .unwrap();
        let source = MaxwellSparseMapping {
            allocation: MaxwellAllocationId::new(3),
            backing: backing.clone(),
            backing_offset: 0,
            kind: 0,
            cacheable: true,
            permissions: MemoryPermissions::READ_WRITE,
        };
        let invalid = vec![
            MaxwellSparseRemapRequest {
                offset: reservation.offset(),
                size: 0x2_0000,
                mapping: Some(source.clone()),
            },
            MaxwellSparseRemapRequest {
                offset: address_space.address(0x5_0000_0000).unwrap(),
                size: 0x2_0000,
                mapping: Some(source.clone()),
            },
        ];
        assert_eq!(
            address_space.remap_sparse(invalid),
            Err(MaxwellAddressSpaceError::OutsideReservation)
        );
        assert_eq!(address_space.mapping_count(), 0);
        assert_eq!(
            address_space.mapping_generation(),
            MappingGeneration::INITIAL
        );

        address_space
            .remap_sparse(vec![MaxwellSparseRemapRequest {
                offset: reservation.offset(),
                size: 0x4_0000,
                mapping: Some(source),
            }])
            .unwrap();
        let retained = address_space.mapping(reservation.offset()).unwrap();
        let second_page = address_space
            .checked_add(reservation.offset(), 0x2_0000)
            .unwrap();
        address_space
            .remap_sparse(vec![MaxwellSparseRemapRequest {
                offset: second_page,
                size: 0x2_0000,
                mapping: None,
            }])
            .unwrap();

        assert_eq!(address_space.mapping_count(), 1);
        assert_eq!(
            address_space.mapping(reservation.offset()).unwrap().size(),
            0x2_0000
        );
        assert!(!address_space.retained_mapping_is_current(&retained));
    }

    #[test]
    fn mapping_modify_splits_kind_state_without_changing_backing_identity() {
        let mut address_space = address_space();
        address_space
            .initialize(MaxwellAddressSpaceInitialization::default())
            .unwrap();
        let (_, backing) = backing(0x3000);
        let mapping = address_space.map(map_request(backing, 4, 0x3000)).unwrap();

        address_space
            .modify_mapping(mapping.offset(), 0x1000, 0x1000, 0xfe, true)
            .unwrap();

        assert_eq!(address_space.mapping_count(), 3);
        let middle = address_space
            .mapping(address_space.checked_add(mapping.offset(), 0x1000).unwrap())
            .unwrap();
        assert_eq!(middle.kind(), 0xfe);
        assert!(middle.cacheable());
        assert_eq!(middle.allocation(), mapping.allocation());
        assert!(!address_space.retained_mapping_is_current(&mapping));
        assert_eq!(address_space.unmap(mapping.offset()).unwrap().len(), 3);
        assert_eq!(address_space.mapping_count(), 0);
    }

    #[test]
    fn internal_mapping_lifetime_exhaustion_is_atomic_and_typed() {
        let mut address_space = address_space();
        address_space
            .initialize(MaxwellAddressSpaceInitialization::default())
            .unwrap();
        let (_, backing) = backing(0x1000);
        address_space.mapping_generation = MappingGeneration::MAX;
        assert!(matches!(
            address_space.map(map_request(backing.clone(), 1, 0x1000)),
            Err(MaxwellAddressSpaceError::GenerationExhausted(_))
        ));
        assert_eq!(address_space.mapping_count(), 0);

        address_space.mapping_generation = MappingGeneration::INITIAL;
        address_space.next_mapping_id = u64::MAX;
        assert_eq!(
            address_space.map(map_request(backing, 1, 0x1000)),
            Err(MaxwellAddressSpaceError::MappingIdentityExhausted)
        );
        assert_eq!(address_space.mapping_count(), 0);
        assert_eq!(
            address_space.mapping_generation(),
            MappingGeneration::INITIAL
        );
    }

    #[test]
    fn maximal_and_scatter_gather_resolution_preserve_boundaries_and_bytes() {
        let mut address_space = address_space();
        address_space
            .initialize(MaxwellAddressSpaceInitialization::default())
            .unwrap();
        let (first_allocation, first_backing) = backing(0x1000);
        let (second_allocation, second_backing) = backing(0x1000);
        first_allocation.write(0xffe, &[0x11, 0x22]).unwrap();
        second_allocation.write(0, &[0x33, 0x44]).unwrap();
        let first = address_space
            .map(map_request(first_backing, 1, 0x1000))
            .unwrap();
        let second = address_space
            .map(map_request(second_backing, 2, 0x1000))
            .unwrap();
        assert_eq!(second.offset().get(), first.offset().get() + 0x1000);

        let start = address_space.checked_add(first.offset(), 0xffe).unwrap();
        let maximal = address_space
            .resolve_maximal(start, MemoryPermissions::READ)
            .unwrap();
        assert_eq!(maximal.gpu_offset(), start);
        assert_eq!(maximal.size(), 2);
        assert_eq!(maximal.backing_offset(), 0xffe);

        let resolved = address_space
            .resolve_range(start, 4, MemoryPermissions::READ)
            .unwrap();
        assert_eq!(resolved.segments().len(), 2);
        assert_eq!(resolved.segments()[0].size(), 2);
        assert_eq!(resolved.segments()[1].size(), 2);
        let mut bytes = [0; 4];
        address_space.read_resolved(&resolved, &mut bytes).unwrap();
        assert_eq!(bytes, [0x11, 0x22, 0x33, 0x44]);

        address_space.unmap(second.offset()).unwrap();
        assert_eq!(
            address_space.resolve_range(start, 4, MemoryPermissions::READ),
            Err(MaxwellGpuAccessError::UnmappedAddress {
                address: second.offset(),
            })
        );
    }

    #[test]
    fn resolution_crosses_canonical_pages_and_rejects_boundary_failures() {
        let mut address_space = address_space();
        address_space
            .initialize(MaxwellAddressSpaceInitialization::default())
            .unwrap();
        let (allocation, read_backing) = backing(0x2000);
        allocation.write(0xfff, &[0xaa, 0xbb]).unwrap();
        let mut request = map_request(read_backing, 1, 0x2000);
        request.permissions = MemoryPermissions::READ;
        let mapping = address_space.map(request).unwrap();
        let last = address_space.checked_add(mapping.offset(), 0x1fff).unwrap();
        let end = address_space.checked_add(mapping.offset(), 0x2000).unwrap();

        let cross_page = address_space
            .resolve_range(
                address_space.checked_add(mapping.offset(), 0xfff).unwrap(),
                2,
                MemoryPermissions::READ,
            )
            .unwrap();
        let mut bytes = [0; 2];
        address_space
            .read_resolved(&cross_page, &mut bytes)
            .unwrap();
        assert_eq!(bytes, [0xaa, 0xbb]);
        assert!(
            address_space
                .resolve_maximal(last, MemoryPermissions::READ)
                .is_ok()
        );
        assert_eq!(
            address_space.resolve_maximal(end, MemoryPermissions::READ),
            Err(MaxwellGpuAccessError::UnmappedAddress { address: end })
        );
        assert_eq!(
            address_space.resolve_maximal(mapping.offset(), MemoryPermissions::WRITE),
            Err(MaxwellGpuAccessError::PermissionDenied {
                address: mapping.offset(),
                required: MemoryPermissions::WRITE,
                available: MemoryPermissions::READ,
            })
        );
        let (_, write_backing) = backing(0x1000);
        let mut write_request = map_request(write_backing, 2, 0x1000);
        write_request.permissions = MemoryPermissions::WRITE;
        let write_mapping = address_space.map(write_request).unwrap();
        let write_resolution = address_space
            .resolve_range(write_mapping.offset(), 1, MemoryPermissions::WRITE)
            .unwrap();
        assert_eq!(
            address_space.read_resolved(&write_resolution, &mut [0]),
            Err(MaxwellGpuAccessError::PermissionDenied {
                address: write_mapping.offset(),
                required: MemoryPermissions::READ,
                available: MemoryPermissions::WRITE,
            })
        );
        assert_eq!(
            address_space.resolve_range(mapping.offset(), 0, MemoryPermissions::READ),
            Err(MaxwellGpuAccessError::EmptyRange)
        );
        let mut other_address_space =
            MaxwellGpuAddressSpace::new(MaxwellAddressSpaceId::new(2), SWITCH_1_GM20B_PROFILE);
        other_address_space
            .initialize(MaxwellAddressSpaceInitialization::default())
            .unwrap();
        assert_eq!(
            other_address_space.read_resolved(&cross_page, &mut bytes),
            Err(MaxwellGpuAccessError::WrongAddressSpace {
                expected: address_space.id(),
                actual: other_address_space.id(),
            })
        );
        let last_profile_address = address_space.address(0x00ff_ffff_ffff).unwrap();
        assert!(matches!(
            address_space.resolve_range(last_profile_address, 2, MemoryPermissions::READ),
            Err(MaxwellGpuAccessError::Address(_))
        ));
    }

    #[test]
    fn remap_and_unmap_make_resolutions_stale_before_backing_access() {
        let mut address_space = address_space();
        address_space
            .initialize(MaxwellAddressSpaceInitialization::default())
            .unwrap();
        let (first_allocation, first_backing) = backing(0x2_0000);
        let (second_allocation, second_backing) = backing(0x2_0000);
        first_allocation.write(0, &[0x11]).unwrap();
        second_allocation.write(0, &[0x22]).unwrap();
        let reservation = address_space
            .reserve(
                1,
                0x2_0000,
                ALLOC_SPACE_FIXED | ALLOC_SPACE_SPARSE,
                0x4_0000_0000,
            )
            .unwrap();
        let sparse_mapping = |allocation, backing| MaxwellSparseMapping {
            allocation: MaxwellAllocationId::new(allocation),
            backing,
            backing_offset: 0,
            kind: 0,
            cacheable: false,
            permissions: MemoryPermissions::READ_WRITE,
        };
        address_space
            .remap_sparse(vec![MaxwellSparseRemapRequest {
                offset: reservation.offset(),
                size: 0x2_0000,
                mapping: Some(sparse_mapping(1, first_backing.clone())),
            }])
            .unwrap();
        let stale = address_space
            .resolve_range(reservation.offset(), 1, MemoryPermissions::READ)
            .unwrap();

        address_space
            .remap_sparse(vec![MaxwellSparseRemapRequest {
                offset: reservation.offset(),
                size: 0x2_0000,
                mapping: Some(sparse_mapping(2, second_backing)),
            }])
            .unwrap();
        first_backing.invalidate_visibility().unwrap();
        assert!(matches!(
            address_space.read_resolved(&stale, &mut [0]),
            Err(MaxwellGpuAccessError::StaleMapping { .. })
        ));

        drop(first_allocation);
        drop(second_allocation);
        let current = address_space
            .resolve_range(reservation.offset(), 1, MemoryPermissions::READ)
            .unwrap();
        let mut byte = [0];
        address_space.read_resolved(&current, &mut byte).unwrap();
        assert_eq!(byte, [0x22]);
        address_space.unmap(reservation.offset()).unwrap();
        assert!(matches!(
            address_space.read_resolved(&current, &mut byte),
            Err(MaxwellGpuAccessError::StaleMapping { .. })
        ));
    }

    #[test]
    fn resolved_aliases_retain_identical_canonical_page_identities() {
        let mut address_space = address_space();
        address_space
            .initialize(MaxwellAddressSpaceInitialization::default())
            .unwrap();
        let (allocation, backing) = backing(0x1000);
        allocation.write(0, &[0x5a]).unwrap();
        let first = address_space
            .map(map_request(backing.clone(), 7, 0x1000))
            .unwrap();
        let second = address_space.map(map_request(backing, 7, 0x1000)).unwrap();
        let first_resolution = address_space
            .resolve_range(first.offset(), 1, MemoryPermissions::READ)
            .unwrap();
        let second_resolution = address_space
            .resolve_range(second.offset(), 1, MemoryPermissions::READ)
            .unwrap();

        assert_eq!(
            first_resolution.segments()[0]
                .mapping()
                .backing()
                .segments()[0]
                .page(),
            second_resolution.segments()[0]
                .mapping()
                .backing()
                .segments()[0]
                .page()
        );
        drop(allocation);
        let mut byte = [0];
        address_space
            .read_resolved(&second_resolution, &mut byte)
            .unwrap();
        assert_eq!(byte, [0x5a]);
    }

    #[test]
    fn mapping_dumps_are_ordered_pointer_free_and_hard_bounded() {
        let mut address_space = address_space();
        address_space
            .initialize(MaxwellAddressSpaceInitialization::default())
            .unwrap();
        let (_, backing) = backing(0x1000);
        for allocation in 0..=MAX_MAPPING_DUMP_ENTRIES {
            address_space
                .map(map_request(
                    backing.clone(),
                    u64::try_from(allocation).unwrap() + 1,
                    0x1000,
                ))
                .unwrap();
        }

        let dump = address_space.mapping_dump(usize::MAX);
        assert_eq!(dump.total_mappings, MAX_MAPPING_DUMP_ENTRIES + 1);
        assert_eq!(dump.entries.len(), MAX_MAPPING_DUMP_ENTRIES);
        assert_eq!(dump.omitted_mappings(), 1);
        assert!(
            dump.entries
                .windows(2)
                .all(|entries| entries[0].offset < entries[1].offset)
        );
        let formatted = dump.to_string();
        assert!(formatted.starts_with(
            "gpu-address-space=0x0000000000000001 profile=switch1-gm20b \
             mapping-generation=65 mappings=65 shown=64"
        ));
        assert!(formatted.contains("permissions=0x3"));
        assert!(formatted.contains("backing-segments=1"));
        assert!(formatted.ends_with("mappings-truncated=1"));
    }
}
