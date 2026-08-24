//! Shared contracts and value types for CPU memory backends.

use std::fmt::{Display, Formatter};

use nixe_memory::{
    AddressSpaceId, ContentGeneration, GuestPhysicalPageId, GuestVirtualAddress, MappingGeneration,
};

use crate::error::{InstructionFetchFault, InstructionFetchFaultReason};

pub use nixe_memory::MemoryPermissions;

/// Page size used by the synthetic and production memory backends.
pub const SYNTHETIC_PAGE_SIZE: usize = 4096;

/// Stage of an atomic synthetic RAM installation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SyntheticInstallStage {
    /// Request validation before resources are created.
    Preflight,
    /// Private physical-page allocation.
    Allocation,
    /// Private physical-page initialization.
    Initialization,
    /// Atomic virtual-mapping publication.
    Publication,
}

/// One ephemeral page request for [`SyntheticMemory::install_ram_pages_atomic`].
#[derive(Clone, Copy, Debug)]
pub struct SyntheticRamPage<'a> {
    /// Page-aligned guest virtual address.
    pub virtual_address: GuestVirtualAddress,
    /// Exact initialized contents of one synthetic page.
    pub bytes: &'a [u8],
    /// Final guest-visible permissions.
    pub permissions: MemoryPermissions,
}

/// Failure of an atomic synthetic RAM installation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SyntheticInstallError {
    /// Stage which rejected the request.
    pub stage: SyntheticInstallStage,
    /// Guest page associated with the failure, when available.
    pub address: Option<GuestVirtualAddress>,
    /// Backend-specific diagnostic.
    pub reason: Box<str>,
}

impl Display for SyntheticInstallError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "synthetic RAM installation failed")?;
        if let Some(address) = self.address {
            write!(formatter, " at {address}")?;
        }
        write!(formatter, " during {:?}: {}", self.stage, self.reason)
    }
}

impl std::error::Error for SyntheticInstallError {}

/// Observable identity and permissions of one synthetic virtual mapping.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SyntheticMappingInfo {
    /// Runtime-owned physical-page identity.
    pub physical_page: GuestPhysicalPageId,
    /// Version of this virtual mapping and its access metadata.
    pub mapping_generation: MappingGeneration,
    /// Exact guest-visible mapping permissions.
    pub permissions: MemoryPermissions,
    /// Runtime-visible mapping attributes.
    pub attributes: MemoryAttributes,
    /// Runtime-assigned semantic state of the mapping.
    pub purpose: MemoryMappingPurpose,
}

/// Identity and content version of one physical code page.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CodePageDependency {
    /// Stable physical-page identity, shared by virtual aliases.
    pub page: GuestPhysicalPageId,
    /// Monotonic content generation observed during the fetch.
    pub generation: ContentGeneration,
    /// Generation of the virtual mapping used for the fetch.
    pub mapping_generation: MappingGeneration,
}

/// The one or two physical pages on which fetched instruction bytes depend.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CodeDependencies {
    first: CodePageDependency,
    second: Option<CodePageDependency>,
}

impl CodeDependencies {
    /// Creates a dependency set for bytes contained in one page.
    #[must_use]
    pub const fn one(first: CodePageDependency) -> Self {
        Self {
            first,
            second: None,
        }
    }

    /// Creates an ordered dependency set for bytes spanning two pages.
    ///
    /// Equal dependencies are canonicalized to a one-page set.
    #[must_use]
    pub fn two(first: CodePageDependency, second: CodePageDependency) -> Self {
        Self::one(first).merge(Self::one(second))
    }

    /// Returns dependencies in address order, without duplicate aliases.
    pub fn iter(self) -> impl Iterator<Item = CodePageDependency> {
        [Some(self.first), self.second].into_iter().flatten()
    }

    pub(super) fn merge(self, other: Self) -> Self {
        let mut merged = self;
        for dependency in other.iter() {
            if !merged.iter().any(|present| present == dependency) {
                debug_assert!(merged.second.is_none());
                merged.second = Some(dependency);
            }
        }
        merged
    }
}

/// Canonical instruction bits accompanied by code-cache dependencies.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct FetchedCode<T> {
    /// Host-endian integer holding the canonical architectural bit pattern.
    pub bits: T,
    /// Physical pages and generations from which the bytes were read.
    pub dependencies: CodeDependencies,
}

/// Contiguous virtual extent of one code page.
///
/// The extent belongs to the memory backend rather than a CPU profile: real
/// process mappings may use different page sizes, and frontend block formation
/// must not assume the synthetic backend's 4 KiB granule.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CodePageSpan {
    /// First byte covered by the page.
    pub start: GuestVirtualAddress,
    /// First byte after the page. `None` represents a page ending at 2^64.
    pub end_exclusive: Option<GuestVirtualAddress>,
}

impl CodePageSpan {
    /// Creates a validated non-empty span containing `address`.
    #[must_use]
    pub const fn containing(
        start: GuestVirtualAddress,
        end_exclusive: Option<GuestVirtualAddress>,
        address: GuestVirtualAddress,
    ) -> Option<Self> {
        let after_start = address.get() >= start.get();
        let before_end = match end_exclusive {
            Some(end) => start.get() < end.get() && address.get() < end.get(),
            None => true,
        };
        if after_start && before_end {
            Some(Self {
                start,
                end_exclusive,
            })
        } else {
            None
        }
    }

    /// Returns whether `address` lies in this span.
    #[must_use]
    pub const fn contains(self, address: GuestVirtualAddress) -> bool {
        address.get() >= self.start.get()
            && match self.end_exclusive {
                Some(end) => address.get() < end.get(),
                None => true,
            }
    }
}

/// Read-only instruction view of a final process address space.
///
/// Implementations enforce execute permission and the alignment implied by the
/// operation. Returned integers are canonical bit patterns; implementations
/// must decode guest bytes explicitly and never rely on host endianness.
pub trait InstructionMemory: Send + Sync {
    /// Returns the virtual code-page extent containing `address`.
    ///
    /// Translators use this only as a block-cut boundary. Fetch methods remain
    /// authoritative for mapping, permission, byte, and generation checks.
    fn code_page_span(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
    ) -> Result<CodePageSpan, InstructionFetchFault>;

    /// Fetches a 16-bit T32 halfword at a two-byte-aligned address.
    fn fetch16(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
    ) -> Result<FetchedCode<u16>, InstructionFetchFault>;

    /// Fetches one A64 or A32 word at a four-byte-aligned address.
    fn fetch32(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
    ) -> Result<FetchedCode<u32>, InstructionFetchFault>;

    /// Fetches a 32-bit T32 encoding as two architectural halfwords.
    ///
    /// The first halfword occupies bits 31:16 of the canonical encoding. This
    /// default deliberately performs two fetches so a page-boundary instruction
    /// records both dependencies and faults precisely on its second halfword.
    fn fetch_t32_32(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
    ) -> Result<FetchedCode<u32>, InstructionFetchFault> {
        let first = self.fetch16(address_space, address)?;
        let second_address = address.checked_add(2).ok_or_else(|| {
            InstructionFetchFault::new(
                address_space,
                address,
                InstructionFetchFaultReason::AddressOverflow,
            )
        })?;
        let second = self.fetch16(address_space, second_address)?;
        Ok(FetchedCode {
            bits: (u32::from(first.bits) << 16) | u32::from(second.bits),
            dependencies: first.dependencies.merge(second.dependencies),
        })
    }
}

/// Width of one architectural data access.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u8)]
pub enum MemoryAccessSize {
    /// One byte.
    Byte = 1,
    /// Two bytes.
    Halfword = 2,
    /// Four bytes.
    Word = 4,
    /// Eight bytes.
    Doubleword = 8,
    /// Sixteen bytes.
    Quadword = 16,
}

impl MemoryAccessSize {
    /// Returns the access width in bytes.
    #[must_use]
    pub const fn bytes(self) -> usize {
        self as usize
    }
}

/// Required alignment independently of the access width.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MemoryAlignment {
    /// The architecture permits an unaligned access.
    Unaligned,
    /// Alignment equals [`MemoryAccessSize`].
    Natural,
    /// Explicit two-byte alignment.
    Bytes2,
    /// Explicit four-byte alignment.
    Bytes4,
    /// Explicit eight-byte alignment.
    Bytes8,
    /// Explicit sixteen-byte alignment.
    Bytes16,
}

impl MemoryAlignment {
    pub(super) const fn bytes(self, size: MemoryAccessSize) -> u8 {
        match self {
            Self::Unaligned => 1,
            Self::Natural => size as u8,
            Self::Bytes2 => 2,
            Self::Bytes4 => 4,
            Self::Bytes8 => 8,
            Self::Bytes16 => 16,
        }
    }
}

/// Ordering required by the architectural operation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MemoryOrdering {
    /// No ordering beyond the access itself.
    Relaxed,
    /// Acquire ordering.
    Acquire,
    /// Release ordering.
    Release,
    /// Acquire and release ordering.
    AcquireRelease,
    /// Sequentially consistent ordering.
    SequentiallyConsistent,
}

/// Semantic class used to select ordinary, atomic, exclusive, or volatile paths.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MemoryAccessClass {
    /// Ordinary architectural load or store.
    Normal,
    /// Atomic read/modify/write component.
    Atomic,
    /// Load-exclusive or store-exclusive component.
    Exclusive,
    /// Access whose externally observable count and order must be preserved.
    Volatile,
}

/// Complete portable description of one data access.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MemoryAccess {
    /// Transfer width.
    pub size: MemoryAccessSize,
    /// Architectural alignment requirement.
    pub alignment: MemoryAlignment,
    /// Architectural ordering requirement.
    pub ordering: MemoryOrdering,
    /// Semantic access class.
    pub class: MemoryAccessClass,
}

impl MemoryAccess {
    /// Creates an access description.
    #[must_use]
    pub const fn new(
        size: MemoryAccessSize,
        alignment: MemoryAlignment,
        ordering: MemoryOrdering,
        class: MemoryAccessClass,
    ) -> Self {
        Self {
            size,
            alignment,
            ordering,
            class,
        }
    }

    /// Creates a naturally aligned ordinary relaxed access.
    #[must_use]
    pub const fn normal(size: MemoryAccessSize) -> Self {
        Self::new(
            size,
            MemoryAlignment::Natural,
            MemoryOrdering::Relaxed,
            MemoryAccessClass::Normal,
        )
    }
}

/// Typed scalar/vector bit pattern transferred by [`CpuMemory`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MemoryValue {
    /// 8-bit bits.
    U8(u8),
    /// 16-bit bits.
    U16(u16),
    /// 32-bit bits.
    U32(u32),
    /// 64-bit bits.
    U64(u64),
    /// 128-bit bits.
    U128(u128),
}

impl MemoryValue {
    /// Returns the represented width.
    #[must_use]
    pub const fn size(self) -> MemoryAccessSize {
        match self {
            Self::U8(_) => MemoryAccessSize::Byte,
            Self::U16(_) => MemoryAccessSize::Halfword,
            Self::U32(_) => MemoryAccessSize::Word,
            Self::U64(_) => MemoryAccessSize::Doubleword,
            Self::U128(_) => MemoryAccessSize::Quadword,
        }
    }

    pub(super) fn from_le_slice(size: MemoryAccessSize, bytes: &[u8]) -> Self {
        let mut value = [0_u8; 16];
        value[..bytes.len()].copy_from_slice(bytes);
        let bits = u128::from_le_bytes(value);
        match size {
            MemoryAccessSize::Byte => Self::U8(bits as u8),
            MemoryAccessSize::Halfword => Self::U16(bits as u16),
            MemoryAccessSize::Word => Self::U32(bits as u32),
            MemoryAccessSize::Doubleword => Self::U64(bits as u64),
            MemoryAccessSize::Quadword => Self::U128(bits),
        }
    }

    pub(super) fn copy_le_bytes(self, destination: &mut [u8]) {
        let bits = match self {
            Self::U8(value) => u128::from(value),
            Self::U16(value) => u128::from(value),
            Self::U32(value) => u128::from(value),
            Self::U64(value) => u128::from(value),
            Self::U128(value) => value,
        };
        destination.copy_from_slice(&bits.to_le_bytes()[..destination.len()]);
    }
}

/// Whether a completed access touched ordinary memory or a device handler.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MemoryRegionKind {
    /// Ordinary page-backed RAM.
    Ram,
    /// Observable MMIO/device access.
    Device,
}

/// One contiguous virtual-memory query result.
///
/// The CPU contract deliberately exposes only mapping facts needed by generic
/// runtimes. Platform layers remain responsible for assigning OS-specific
/// memory-state values.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MemoryQueryResult {
    pub base: GuestVirtualAddress,
    pub size: u64,
    pub region: Option<MemoryRegionKind>,
    pub permissions: MemoryPermissions,
    pub attributes: MemoryAttributes,
    pub purpose: MemoryMappingPurpose,
}

/// Runtime-assigned purpose of a virtual mapping.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum MemoryMappingPurpose {
    #[default]
    Normal,
    CodeStatic,
    CodeMutable,
    ModuleCodeStatic,
    ModuleCodeMutable,
    ThreadLocal,
    Heap,
    SharedMemory,
}

impl MemoryMappingPurpose {
    /// Returns whether the mapping state permits SVC-style reprotection.
    #[must_use]
    pub const fn allows_reprotect(self) -> bool {
        matches!(
            self,
            Self::CodeMutable | Self::ModuleCodeMutable | Self::Heap
        )
    }

    /// Returns whether the mapping state permits cache-attribute changes.
    #[must_use]
    pub const fn allows_attribute_change(self) -> bool {
        matches!(
            self,
            Self::CodeMutable | Self::ModuleCodeMutable | Self::Heap
        )
    }
}

/// Generic guest-visible attributes retained independently of OS result codes.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct MemoryAttributes(u32);

impl MemoryAttributes {
    pub const NONE: Self = Self(0);
    pub const UNCACHED: Self = Self(1 << 3);
    pub const PERMISSION_LOCKED: Self = Self(1 << 4);
    pub const KNOWN: Self = Self(Self::UNCACHED.0 | Self::PERMISSION_LOCKED.0);

    #[must_use]
    pub const fn from_bits(bits: u32) -> Option<Self> {
        if bits & !Self::KNOWN.0 == 0 {
            Some(Self(bits))
        } else {
            None
        }
    }

    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }

    #[must_use]
    pub const fn contains(self, attributes: Self) -> bool {
        self.0 & attributes.0 == attributes.0
    }
}

/// Successful data-read result.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DataReadResult {
    /// Returned architectural bits.
    pub value: MemoryValue,
    /// Kind of backing that serviced the operation.
    pub region: MemoryRegionKind,
}

/// Successful data-write result.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DataWriteResult {
    /// Kind of backing that serviced the operation.
    pub region: MemoryRegionKind,
}

/// Kind of failed data operation.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum DataAccessKind {
    /// Load/read.
    Read,
    /// Store/write.
    Write,
}

/// Precise reason for a data-access failure.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum DataAccessFaultReason {
    /// No virtual mapping covers the first failing byte.
    Unmapped,
    /// Read permission is absent.
    ReadPermissionDenied,
    /// Write permission is absent.
    WritePermissionDenied,
    /// Address violates the access description.
    Misaligned { required_alignment: u8 },
    /// Address calculation overflowed.
    AddressOverflow,
    /// Value width did not equal the access width.
    ValueSizeMismatch,
    /// An access cannot span distinct RAM/device regions.
    MixedRegions,
    /// The backing content version cannot advance without observable reuse.
    ContentGenerationExhausted,
    /// Canonical host backing could not complete an emulator-side operation.
    HostBacking(Box<str>),
    /// Device handler rejected the operation.
    Device(Box<str>),
    /// Synthetic fault requested by a test.
    Injected(Box<str>),
}

/// Precise failure of an interpreter-visible data access.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct DataAccessFault {
    /// Process address-space identity.
    pub address_space: AddressSpaceId,
    /// First failing virtual byte, or the operation address for whole-access faults.
    pub address: GuestVirtualAddress,
    /// Read or write operation.
    pub kind: DataAccessKind,
    /// Structured reason.
    pub reason: DataAccessFaultReason,
}

impl DataAccessFault {
    /// Creates a structured data-access fault for a memory implementation.
    #[must_use]
    pub const fn new(
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
        kind: DataAccessKind,
        reason: DataAccessFaultReason,
    ) -> Self {
        Self {
            address_space,
            address,
            kind,
            reason,
        }
    }
}

/// Interpreter-facing semantic memory contract.
pub trait CpuMemory: InstructionMemory {
    /// Performs one complete architectural read.
    fn read(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
        access: MemoryAccess,
    ) -> Result<DataReadResult, DataAccessFault>;

    /// Performs one complete architectural write.
    fn write(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
        access: MemoryAccess,
        value: MemoryValue,
    ) -> Result<DataWriteResult, DataAccessFault>;

    /// Queries the maximal contiguous mapping state containing `address`.
    ///
    /// `end_exclusive` is supplied by the process address-space policy and
    /// bounds both mapped and unmapped results.
    fn query_memory(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
        end_exclusive: GuestVirtualAddress,
    ) -> Option<MemoryQueryResult>;

    /// Loads a value and returns the backend identity required by a local
    /// exclusive monitor.
    fn load_exclusive(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
        access: MemoryAccess,
    ) -> Result<(DataReadResult, crate::exclusive::ExclusiveReservation), DataAccessFault>;

    /// Conditionally stores if the supplied physical reservation is current.
    fn store_exclusive(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
        access: MemoryAccess,
        value: MemoryValue,
        reservation: crate::exclusive::ExclusiveReservation,
    ) -> Result<(DataWriteResult, bool), DataAccessFault>;
}

/// Runtime-facing mutation contract for a process address space.
///
/// Execution engines consume [`CpuMemory`]. Kernel policy receives this
/// narrower extension only while applying validated mapping operations, which
/// keeps Horizon concepts out of the CPU crate.
pub trait ProcessMemory: CpuMemory {
    /// Copies a checked RAM range into host storage while resolving each
    /// spanned mapping under one bounded memory transaction.
    fn read_bytes(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
        output: &mut [u8],
    ) -> Result<(), DataAccessFault>;

    /// Atomically copies host bytes into a completely validated writable RAM
    /// range and publishes one content mutation per affected physical page.
    fn write_bytes(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
        bytes: &[u8],
    ) -> Result<(), DataAccessFault>;

    /// Atomically resizes a zero-initialized mapping from its fixed base.
    fn resize_zeroed_mapping(
        &self,
        address_space: AddressSpaceId,
        start: GuestVirtualAddress,
        old_size: u64,
        new_size: u64,
        permissions: MemoryPermissions,
        purpose: MemoryMappingPurpose,
    ) -> Result<(), MemoryMappingError>;

    /// Atomically replaces permissions on a complete page-aligned mapped range.
    fn set_permissions(
        &self,
        address_space: AddressSpaceId,
        start: GuestVirtualAddress,
        size: u64,
        permissions: MemoryPermissions,
    ) -> Result<(), MemoryProtectionError>;

    /// Atomically updates selected attributes on a mapped page-aligned range.
    fn set_attributes(
        &self,
        address_space: AddressSpaceId,
        start: GuestVirtualAddress,
        size: u64,
        mask: MemoryAttributes,
        value: MemoryAttributes,
    ) -> Result<(), MemoryProtectionError>;
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MemoryMappingErrorReason {
    InvalidRange,
    AlreadyMapped,
    MappingStateMismatch,
    WritableExecutable,
    ResourceExhausted,
    GenerationExhausted,
}

/// Pointer-free reason a runtime mapping resize was rejected.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MemoryMappingError {
    pub address_space: AddressSpaceId,
    pub address: GuestVirtualAddress,
    pub reason: MemoryMappingErrorReason,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MemoryProtectionErrorReason {
    InvalidRange,
    Unmapped,
    WritableExecutable,
    PermissionLocked,
    GenerationExhausted,
}

/// Pointer-free reason a runtime mapping-protection operation was rejected.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MemoryProtectionError {
    pub address_space: AddressSpaceId,
    pub address: GuestVirtualAddress,
    pub reason: MemoryProtectionErrorReason,
}

/// Callback interface used by synthetic MMIO pages.
pub trait SyntheticMmio: Send {
    /// Reads a value at a page-relative byte offset.
    fn read(&mut self, offset: u64, access: MemoryAccess) -> Result<MemoryValue, Box<str>>;

    /// Writes a value at a page-relative byte offset.
    fn write(
        &mut self,
        offset: u64,
        access: MemoryAccess,
        value: MemoryValue,
    ) -> Result<(), Box<str>>;
}
