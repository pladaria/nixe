//! Portable CPU memory contracts and a synthetic implementation for tests.
//!
//! Frontends fetch from the final process address space through these traits.
//! They never consume loader images, file storage, or mutable host pointers.

use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
    fmt::{Display, Formatter},
};

use crate::{
    address::{AddressSpaceId, CodeGeneration, GuestPhysicalPageId, GuestVirtualAddress},
    error::{InstructionFetchFault, InstructionFetchFaultReason},
};

/// Page size used by [`SyntheticMemory`].
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
    /// Exact guest-visible mapping permissions.
    pub permissions: MemoryPermissions,
}

/// Identity and content version of one physical code page.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CodePageDependency {
    /// Stable physical-page identity, shared by virtual aliases.
    pub page: GuestPhysicalPageId,
    /// Monotonic content generation observed during the fetch.
    pub generation: CodeGeneration,
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

    fn merge(self, other: Self) -> Self {
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

/// Read-only instruction view of a final process address space.
///
/// Implementations enforce execute permission and the alignment implied by the
/// operation. Returned integers are canonical bit patterns; implementations
/// must decode guest bytes explicitly and never rely on host endianness.
pub trait InstructionMemory {
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
    const fn bytes(self, size: MemoryAccessSize) -> u8 {
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

    fn from_le_slice(size: MemoryAccessSize, bytes: &[u8]) -> Self {
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

    fn copy_le_bytes(self, destination: &mut [u8]) {
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
}

/// Read, write, and execute permissions on a synthetic virtual mapping.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct MemoryPermissions(u8);

impl MemoryPermissions {
    /// No access.
    pub const NONE: Self = Self(0);
    /// Read-only data.
    pub const READ: Self = Self(1 << 0);
    /// Write-only data.
    pub const WRITE: Self = Self(1 << 1);
    /// Instruction execution.
    pub const EXECUTE: Self = Self(1 << 2);
    /// Read/write data.
    pub const READ_WRITE: Self = Self(Self::READ.0 | Self::WRITE.0);
    /// Readable executable code.
    pub const READ_EXECUTE: Self = Self(Self::READ.0 | Self::EXECUTE.0);
    /// Writable executable memory, useful for coherency tests.
    pub const READ_WRITE_EXECUTE: Self = Self(Self::READ.0 | Self::WRITE.0 | Self::EXECUTE.0);

    const fn contains(self, permission: Self) -> bool {
        self.0 & permission.0 == permission.0
    }
}

/// Callback interface used by synthetic MMIO pages.
pub trait SyntheticMmio {
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

#[derive(Clone, Copy)]
struct Mapping {
    physical_page: GuestPhysicalPageId,
    permissions: MemoryPermissions,
}

enum PhysicalPage {
    Ram {
        bytes: Box<[u8; SYNTHETIC_PAGE_SIZE]>,
        generation: u64,
    },
    Mmio(Box<dyn SyntheticMmio>),
}

struct SyntheticMemoryInner {
    mappings: BTreeMap<(AddressSpaceId, u64), Mapping>,
    pages: BTreeMap<GuestPhysicalPageId, PhysicalPage>,
    instruction_faults: BTreeMap<(AddressSpaceId, GuestVirtualAddress), Box<str>>,
    data_faults: BTreeMap<(AddressSpaceId, GuestVirtualAddress, DataAccessKind), Box<str>>,
    next_page_id: u64,
    install_failure: Option<(SyntheticInstallStage, usize, Box<str>)>,
}

impl Default for SyntheticMemoryInner {
    fn default() -> Self {
        Self {
            mappings: BTreeMap::new(),
            pages: BTreeMap::new(),
            instruction_faults: BTreeMap::new(),
            data_faults: BTreeMap::new(),
            next_page_id: 1,
            install_failure: None,
        }
    }
}

/// Small deterministic process-memory implementation for frontend tests.
///
/// Its APIs expose copies, identities, and callbacks only; no raw mutable host
/// pointer crosses the CPU/memory boundary.
#[derive(Default)]
pub struct SyntheticMemory {
    inner: RefCell<SyntheticMemoryInner>,
}

impl SyntheticMemory {
    /// Creates empty synthetic memory.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Atomically creates, initializes, and publishes ordinary RAM pages.
    ///
    /// Physical pages are owned by this memory object after success. A failed
    /// request changes neither existing mappings nor physical pages.
    pub fn install_ram_pages_atomic(
        &mut self,
        address_space: AddressSpaceId,
        requests: &[SyntheticRamPage<'_>],
    ) -> Result<(), SyntheticInstallError> {
        let inner = self.inner.get_mut();
        let mut virtual_pages = Vec::with_capacity(requests.len());
        let mut unique_virtual_pages = BTreeSet::new();
        for (index, request) in requests.iter().enumerate() {
            fail_install_if_requested(
                inner,
                SyntheticInstallStage::Preflight,
                index,
                request.virtual_address,
            )?;
            if !request
                .virtual_address
                .is_aligned_to(SYNTHETIC_PAGE_SIZE as u64)
            {
                return Err(install_error(
                    SyntheticInstallStage::Preflight,
                    Some(request.virtual_address),
                    "virtual address is not page aligned",
                ));
            }
            if request.bytes.len() != SYNTHETIC_PAGE_SIZE {
                return Err(install_error(
                    SyntheticInstallStage::Preflight,
                    Some(request.virtual_address),
                    "page contents do not match the synthetic page size",
                ));
            }
            if request
                .virtual_address
                .checked_add((SYNTHETIC_PAGE_SIZE - 1) as u64)
                .is_none()
            {
                return Err(install_error(
                    SyntheticInstallStage::Preflight,
                    Some(request.virtual_address),
                    "virtual page range overflows",
                ));
            }
            let virtual_page = request.virtual_address.get() / SYNTHETIC_PAGE_SIZE as u64;
            if !unique_virtual_pages.insert(virtual_page) {
                return Err(install_error(
                    SyntheticInstallStage::Preflight,
                    Some(request.virtual_address),
                    "request contains a duplicate virtual page",
                ));
            }
            if inner.mappings.contains_key(&(address_space, virtual_page)) {
                return Err(install_error(
                    SyntheticInstallStage::Preflight,
                    Some(request.virtual_address),
                    "virtual page is already mapped",
                ));
            }
            virtual_pages.push(virtual_page);
        }

        let mut next_page_id = inner.next_page_id;
        let mut pending = Vec::with_capacity(requests.len());
        for (index, request) in requests.iter().enumerate() {
            fail_install_if_requested(
                inner,
                SyntheticInstallStage::Allocation,
                index,
                request.virtual_address,
            )?;
            while inner
                .pages
                .contains_key(&GuestPhysicalPageId::new(next_page_id))
            {
                next_page_id = next_page_id.checked_add(1).ok_or_else(|| {
                    install_error(
                        SyntheticInstallStage::Allocation,
                        Some(request.virtual_address),
                        "physical-page identities are exhausted",
                    )
                })?;
            }
            let physical_page = GuestPhysicalPageId::new(next_page_id);
            next_page_id = next_page_id.checked_add(1).ok_or_else(|| {
                install_error(
                    SyntheticInstallStage::Allocation,
                    Some(request.virtual_address),
                    "physical-page identities are exhausted",
                )
            })?;
            fail_install_if_requested(
                inner,
                SyntheticInstallStage::Initialization,
                index,
                request.virtual_address,
            )?;
            let mut contents = Box::new([0; SYNTHETIC_PAGE_SIZE]);
            contents.copy_from_slice(request.bytes);
            pending.push((
                virtual_pages[index],
                Mapping {
                    physical_page,
                    permissions: request.permissions,
                },
                PhysicalPage::Ram {
                    bytes: contents,
                    generation: 1,
                },
            ));
        }
        for (index, request) in requests.iter().enumerate() {
            fail_install_if_requested(
                inner,
                SyntheticInstallStage::Publication,
                index,
                request.virtual_address,
            )?;
        }

        for (virtual_page, mapping, page) in pending {
            let previous_page = inner.pages.insert(mapping.physical_page, page);
            let previous_mapping = inner
                .mappings
                .insert((address_space, virtual_page), mapping);
            debug_assert!(previous_page.is_none());
            debug_assert!(previous_mapping.is_none());
        }
        inner.next_page_id = next_page_id;
        Ok(())
    }

    /// Injects a deterministic failure into a future atomic installation.
    pub fn inject_install_failure(
        &mut self,
        stage: SyntheticInstallStage,
        request_index: usize,
        reason: impl Into<Box<str>>,
    ) {
        self.inner.get_mut().install_failure = Some((stage, request_index, reason.into()));
    }

    /// Returns mapping identity and permissions for a page containing `address`.
    pub fn mapping_info(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
    ) -> Option<SyntheticMappingInfo> {
        mapping_at(&self.inner.borrow(), address_space, address).map(|mapping| {
            SyntheticMappingInfo {
                physical_page: mapping.physical_page,
                permissions: mapping.permissions,
            }
        })
    }

    /// Returns the number of physical pages currently owned by this backend.
    pub fn physical_page_count(&self) -> usize {
        self.inner.borrow().pages.len()
    }

    /// Creates a zero-filled ordinary physical page.
    pub fn add_ram_page(&mut self, page: GuestPhysicalPageId) -> bool {
        self.inner
            .get_mut()
            .pages
            .insert(
                page,
                PhysicalPage::Ram {
                    bytes: Box::new([0; SYNTHETIC_PAGE_SIZE]),
                    generation: 0,
                },
            )
            .is_none()
    }

    /// Creates a device-backed physical page.
    pub fn add_mmio_page(
        &mut self,
        page: GuestPhysicalPageId,
        handler: impl SyntheticMmio + 'static,
    ) -> bool {
        self.inner
            .get_mut()
            .pages
            .insert(page, PhysicalPage::Mmio(Box::new(handler)))
            .is_none()
    }

    /// Maps one page-aligned virtual page; aliases map the same physical ID again.
    pub fn map_page(
        &mut self,
        address_space: AddressSpaceId,
        virtual_address: GuestVirtualAddress,
        physical_page: GuestPhysicalPageId,
        permissions: MemoryPermissions,
    ) -> bool {
        if !virtual_address.is_aligned_to(SYNTHETIC_PAGE_SIZE as u64)
            || !self.inner.get_mut().pages.contains_key(&physical_page)
        {
            return false;
        }
        self.inner
            .get_mut()
            .mappings
            .insert(
                (
                    address_space,
                    virtual_address.get() / SYNTHETIC_PAGE_SIZE as u64,
                ),
                Mapping {
                    physical_page,
                    permissions,
                },
            )
            .is_none()
    }

    /// Copies fixture bytes directly into a RAM page and advances its generation.
    pub fn initialize_ram(
        &mut self,
        page: GuestPhysicalPageId,
        offset: usize,
        bytes: &[u8],
    ) -> bool {
        let Some(PhysicalPage::Ram {
            bytes: contents,
            generation,
        }) = self.inner.get_mut().pages.get_mut(&page)
        else {
            return false;
        };
        let Some(end) = offset.checked_add(bytes.len()) else {
            return false;
        };
        let Some(destination) = contents.get_mut(offset..end) else {
            return false;
        };
        destination.copy_from_slice(bytes);
        *generation = generation.wrapping_add(1);
        true
    }

    /// Injects a deterministic fetch failure at an exact virtual address.
    pub fn inject_instruction_fault(
        &mut self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
        reason: impl Into<Box<str>>,
    ) {
        self.inner
            .get_mut()
            .instruction_faults
            .insert((address_space, address), reason.into());
    }

    /// Injects a deterministic data failure at an exact operation address.
    pub fn inject_data_fault(
        &mut self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
        kind: DataAccessKind,
        reason: impl Into<Box<str>>,
    ) {
        self.inner
            .get_mut()
            .data_faults
            .insert((address_space, address, kind), reason.into());
    }

    fn fetch<const N: usize>(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
        alignment: u8,
    ) -> Result<([u8; N], CodeDependencies), InstructionFetchFault> {
        if !address.is_aligned_to(u64::from(alignment)) {
            return Err(InstructionFetchFault::new(
                address_space,
                address,
                InstructionFetchFaultReason::Misaligned {
                    required_alignment: alignment,
                },
            ));
        }
        let inner = self.inner.borrow();
        let mut bytes = [0; N];
        let mut dependencies: Option<CodeDependencies> = None;
        for (index, destination) in bytes.iter_mut().enumerate() {
            let Some(current) = address.checked_add(index as u64) else {
                return Err(InstructionFetchFault::new(
                    address_space,
                    address,
                    InstructionFetchFaultReason::AddressOverflow,
                ));
            };
            if let Some(reason) = inner.instruction_faults.get(&(address_space, current)) {
                return Err(InstructionFetchFault::new(
                    address_space,
                    current,
                    InstructionFetchFaultReason::Memory(reason.clone()),
                ));
            }
            let mapping = mapping_at(&inner, address_space, current).ok_or_else(|| {
                InstructionFetchFault::new(
                    address_space,
                    current,
                    InstructionFetchFaultReason::Unmapped,
                )
            })?;
            if !mapping.permissions.contains(MemoryPermissions::EXECUTE) {
                return Err(InstructionFetchFault::new(
                    address_space,
                    current,
                    InstructionFetchFaultReason::ExecutePermissionDenied,
                ));
            }
            let Some(PhysicalPage::Ram {
                bytes: contents,
                generation,
            }) = inner.pages.get(&mapping.physical_page)
            else {
                return Err(InstructionFetchFault::new(
                    address_space,
                    current,
                    InstructionFetchFaultReason::Memory("executable mapping is not RAM".into()),
                ));
            };
            *destination = contents[page_offset(current)];
            let dependency = CodePageDependency {
                page: mapping.physical_page,
                generation: CodeGeneration::new(*generation),
            };
            dependencies = Some(match dependencies {
                None => CodeDependencies::one(dependency),
                Some(current_dependencies) => {
                    current_dependencies.merge(CodeDependencies::one(dependency))
                }
            });
        }
        Ok((
            bytes,
            dependencies.expect("non-empty fetch has a dependency"),
        ))
    }
}

fn install_error(
    stage: SyntheticInstallStage,
    address: Option<GuestVirtualAddress>,
    reason: impl Into<Box<str>>,
) -> SyntheticInstallError {
    SyntheticInstallError {
        stage,
        address,
        reason: reason.into(),
    }
}

fn fail_install_if_requested(
    inner: &SyntheticMemoryInner,
    stage: SyntheticInstallStage,
    index: usize,
    address: GuestVirtualAddress,
) -> Result<(), SyntheticInstallError> {
    if let Some((requested_stage, requested_index, reason)) = &inner.install_failure
        && *requested_stage == stage
        && *requested_index == index
    {
        return Err(install_error(stage, Some(address), reason.clone()));
    }
    Ok(())
}

impl InstructionMemory for SyntheticMemory {
    fn fetch16(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
    ) -> Result<FetchedCode<u16>, InstructionFetchFault> {
        let (bytes, dependencies) = self.fetch::<2>(address_space, address, 2)?;
        Ok(FetchedCode {
            bits: u16::from_le_bytes(bytes),
            dependencies,
        })
    }

    fn fetch32(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
    ) -> Result<FetchedCode<u32>, InstructionFetchFault> {
        let (bytes, dependencies) = self.fetch::<4>(address_space, address, 4)?;
        Ok(FetchedCode {
            bits: u32::from_le_bytes(bytes),
            dependencies,
        })
    }
}

impl CpuMemory for SyntheticMemory {
    fn read(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
        access: MemoryAccess,
    ) -> Result<DataReadResult, DataAccessFault> {
        validate_data_access(self, address_space, address, access, DataAccessKind::Read)?;
        let mut inner = self.inner.borrow_mut();
        if let Some(reason) = inner
            .data_faults
            .get(&(address_space, address, DataAccessKind::Read))
        {
            return Err(DataAccessFault::new(
                address_space,
                address,
                DataAccessKind::Read,
                DataAccessFaultReason::Injected(reason.clone()),
            ));
        }
        let mapping = mapping_at(&inner, address_space, address).expect("validated mapping");
        match inner
            .pages
            .get_mut(&mapping.physical_page)
            .expect("mapping references a page")
        {
            PhysicalPage::Mmio(handler) => {
                if page_offset(address) + access.size.bytes() > SYNTHETIC_PAGE_SIZE {
                    return Err(DataAccessFault::new(
                        address_space,
                        address,
                        DataAccessKind::Read,
                        DataAccessFaultReason::MixedRegions,
                    ));
                }
                let value =
                    handler
                        .read(page_offset(address) as u64, access)
                        .map_err(|reason| {
                            DataAccessFault::new(
                                address_space,
                                address,
                                DataAccessKind::Read,
                                DataAccessFaultReason::Device(reason),
                            )
                        })?;
                if value.size() != access.size {
                    return Err(DataAccessFault::new(
                        address_space,
                        address,
                        DataAccessKind::Read,
                        DataAccessFaultReason::ValueSizeMismatch,
                    ));
                }
                Ok(DataReadResult {
                    value,
                    region: MemoryRegionKind::Device,
                })
            }
            PhysicalPage::Ram { .. } => {
                let mut bytes = [0_u8; 16];
                for (index, byte) in bytes[..access.size.bytes()].iter_mut().enumerate() {
                    let current = address.checked_add(index as u64).expect("validated range");
                    let mapping =
                        mapping_at(&inner, address_space, current).expect("validated mapping");
                    let PhysicalPage::Ram {
                        bytes: contents, ..
                    } = inner
                        .pages
                        .get(&mapping.physical_page)
                        .expect("validated RAM region")
                    else {
                        unreachable!()
                    };
                    *byte = contents[page_offset(current)];
                }
                Ok(DataReadResult {
                    value: MemoryValue::from_le_slice(access.size, &bytes[..access.size.bytes()]),
                    region: MemoryRegionKind::Ram,
                })
            }
        }
    }

    fn write(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
        access: MemoryAccess,
        value: MemoryValue,
    ) -> Result<DataWriteResult, DataAccessFault> {
        if value.size() != access.size {
            return Err(DataAccessFault::new(
                address_space,
                address,
                DataAccessKind::Write,
                DataAccessFaultReason::ValueSizeMismatch,
            ));
        }
        validate_data_access(self, address_space, address, access, DataAccessKind::Write)?;
        let mut inner = self.inner.borrow_mut();
        if let Some(reason) =
            inner
                .data_faults
                .get(&(address_space, address, DataAccessKind::Write))
        {
            return Err(DataAccessFault::new(
                address_space,
                address,
                DataAccessKind::Write,
                DataAccessFaultReason::Injected(reason.clone()),
            ));
        }
        let mapping = mapping_at(&inner, address_space, address).expect("validated mapping");
        if matches!(
            inner.pages.get(&mapping.physical_page),
            Some(PhysicalPage::Mmio(_))
        ) {
            if page_offset(address) + access.size.bytes() > SYNTHETIC_PAGE_SIZE {
                return Err(DataAccessFault::new(
                    address_space,
                    address,
                    DataAccessKind::Write,
                    DataAccessFaultReason::MixedRegions,
                ));
            }
            let PhysicalPage::Mmio(handler) = inner
                .pages
                .get_mut(&mapping.physical_page)
                .expect("validated MMIO page")
            else {
                unreachable!()
            };
            handler
                .write(page_offset(address) as u64, access, value)
                .map_err(|reason| {
                    DataAccessFault::new(
                        address_space,
                        address,
                        DataAccessKind::Write,
                        DataAccessFaultReason::Device(reason),
                    )
                })?;
            return Ok(DataWriteResult {
                region: MemoryRegionKind::Device,
            });
        }
        let mut bytes = [0_u8; 16];
        value.copy_le_bytes(&mut bytes[..access.size.bytes()]);
        let mut touched_pages = Vec::with_capacity(2);
        for (index, byte) in bytes[..access.size.bytes()].iter().copied().enumerate() {
            let current = address.checked_add(index as u64).expect("validated range");
            let mapping = mapping_at(&inner, address_space, current).expect("validated mapping");
            let PhysicalPage::Ram {
                bytes: contents, ..
            } = inner
                .pages
                .get_mut(&mapping.physical_page)
                .expect("validated RAM region")
            else {
                unreachable!()
            };
            contents[page_offset(current)] = byte;
            if !touched_pages.contains(&mapping.physical_page) {
                touched_pages.push(mapping.physical_page);
            }
        }
        for page in touched_pages {
            let Some(PhysicalPage::Ram { generation, .. }) = inner.pages.get_mut(&page) else {
                unreachable!()
            };
            *generation = generation.wrapping_add(1);
        }
        Ok(DataWriteResult {
            region: MemoryRegionKind::Ram,
        })
    }
}

fn mapping_at(
    inner: &SyntheticMemoryInner,
    address_space: AddressSpaceId,
    address: GuestVirtualAddress,
) -> Option<Mapping> {
    inner
        .mappings
        .get(&(address_space, address.get() / SYNTHETIC_PAGE_SIZE as u64))
        .copied()
}

fn page_offset(address: GuestVirtualAddress) -> usize {
    address.get() as usize % SYNTHETIC_PAGE_SIZE
}

fn validate_data_access(
    memory: &SyntheticMemory,
    address_space: AddressSpaceId,
    address: GuestVirtualAddress,
    access: MemoryAccess,
    kind: DataAccessKind,
) -> Result<(), DataAccessFault> {
    let required_alignment = access.alignment.bytes(access.size);
    if !address.is_aligned_to(u64::from(required_alignment)) {
        return Err(DataAccessFault::new(
            address_space,
            address,
            kind,
            DataAccessFaultReason::Misaligned { required_alignment },
        ));
    }
    let inner = memory.inner.borrow();
    let mut region = None;
    for index in 0..access.size.bytes() {
        let Some(current) = address.checked_add(index as u64) else {
            return Err(DataAccessFault::new(
                address_space,
                address,
                kind,
                DataAccessFaultReason::AddressOverflow,
            ));
        };
        let mapping = mapping_at(&inner, address_space, current).ok_or_else(|| {
            DataAccessFault::new(
                address_space,
                current,
                kind,
                DataAccessFaultReason::Unmapped,
            )
        })?;
        let required = match kind {
            DataAccessKind::Read => MemoryPermissions::READ,
            DataAccessKind::Write => MemoryPermissions::WRITE,
        };
        if !mapping.permissions.contains(required) {
            let reason = match kind {
                DataAccessKind::Read => DataAccessFaultReason::ReadPermissionDenied,
                DataAccessKind::Write => DataAccessFaultReason::WritePermissionDenied,
            };
            return Err(DataAccessFault::new(address_space, current, kind, reason));
        }
        let current_region = match inner.pages.get(&mapping.physical_page) {
            Some(PhysicalPage::Ram { .. }) => MemoryRegionKind::Ram,
            Some(PhysicalPage::Mmio(_)) => MemoryRegionKind::Device,
            None => {
                return Err(DataAccessFault::new(
                    address_space,
                    current,
                    kind,
                    DataAccessFaultReason::Unmapped,
                ));
            }
        };
        if region.is_some_and(|first| first != current_region) {
            return Err(DataAccessFault::new(
                address_space,
                current,
                kind,
                DataAccessFaultReason::MixedRegions,
            ));
        }
        region = Some(current_region);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use super::*;

    const SPACE: AddressSpaceId = AddressSpaceId::new(7);
    const CODE: GuestVirtualAddress = GuestVirtualAddress::new(0x1000);
    const ALIAS: GuestVirtualAddress = GuestVirtualAddress::new(0x5000);
    const PAGE_1: GuestPhysicalPageId = GuestPhysicalPageId::new(11);
    const PAGE_2: GuestPhysicalPageId = GuestPhysicalPageId::new(12);

    fn code_memory() -> SyntheticMemory {
        let mut memory = SyntheticMemory::new();
        assert!(memory.add_ram_page(PAGE_1));
        assert!(memory.initialize_ram(PAGE_1, 0, &[0x1f, 0x20, 0x03, 0xd5]));
        assert!(memory.map_page(SPACE, CODE, PAGE_1, MemoryPermissions::READ_EXECUTE));
        memory
    }

    #[test]
    fn a64_and_a32_words_use_explicit_little_endian_canonicalization() {
        let memory = code_memory();

        let a64_or_a32 = memory.fetch32(SPACE, CODE).unwrap();

        assert_eq!(a64_or_a32.bits, 0xd503_201f);
        assert_eq!(
            a64_or_a32.dependencies.iter().collect::<Vec<_>>(),
            vec![CodePageDependency {
                page: PAGE_1,
                generation: CodeGeneration::new(1)
            }]
        );
    }

    #[test]
    fn fetch_requires_architectural_alignment_and_execute_permission() {
        let mut memory = code_memory();
        assert!(memory.map_page(SPACE, ALIAS, PAGE_1, MemoryPermissions::READ_WRITE));

        let misaligned = memory
            .fetch32(SPACE, CODE.checked_add(2).unwrap())
            .unwrap_err();
        let denied = memory.fetch16(SPACE, ALIAS).unwrap_err();

        assert_eq!(
            misaligned.reason,
            InstructionFetchFaultReason::Misaligned {
                required_alignment: 4
            }
        );
        assert_eq!(
            denied.reason,
            InstructionFetchFaultReason::ExecutePermissionDenied
        );
    }

    #[test]
    fn aliases_report_physical_identity_and_observe_generation_changes() {
        let mut memory = code_memory();
        assert!(memory.map_page(SPACE, ALIAS, PAGE_1, MemoryPermissions::READ_WRITE_EXECUTE));
        let before = memory.fetch32(SPACE, CODE).unwrap();

        memory
            .write(
                SPACE,
                ALIAS,
                MemoryAccess::normal(MemoryAccessSize::Word),
                MemoryValue::U32(0x1122_3344),
            )
            .unwrap();
        let after = memory.fetch32(SPACE, CODE).unwrap();

        assert_eq!(after.bits, 0x1122_3344);
        assert_eq!(before.dependencies.iter().next().unwrap().page, PAGE_1);
        assert_eq!(after.dependencies.iter().next().unwrap().page, PAGE_1);
        assert_ne!(before.dependencies, after.dependencies);
    }

    #[test]
    fn t32_cross_page_fetch_records_both_pages_in_address_order() {
        let mut memory = SyntheticMemory::new();
        assert!(memory.add_ram_page(PAGE_1));
        assert!(memory.add_ram_page(PAGE_2));
        assert!(memory.initialize_ram(PAGE_1, SYNTHETIC_PAGE_SIZE - 2, &[0x00, 0xf0]));
        assert!(memory.initialize_ram(PAGE_2, 0, &[0x01, 0xf8]));
        assert!(memory.map_page(SPACE, CODE, PAGE_1, MemoryPermissions::READ_EXECUTE));
        assert!(memory.map_page(
            SPACE,
            GuestVirtualAddress::new(0x2000),
            PAGE_2,
            MemoryPermissions::READ_EXECUTE
        ));

        let fetched = memory
            .fetch_t32_32(SPACE, GuestVirtualAddress::new(0x1ffe))
            .unwrap();

        assert_eq!(fetched.bits, 0xf000_f801);
        assert_eq!(
            fetched
                .dependencies
                .iter()
                .map(|dependency| dependency.page)
                .collect::<Vec<_>>(),
            vec![PAGE_1, PAGE_2]
        );
    }

    #[test]
    fn t32_second_halfword_fault_identifies_the_unavailable_address() {
        let mut memory = SyntheticMemory::new();
        assert!(memory.add_ram_page(PAGE_1));
        assert!(memory.initialize_ram(PAGE_1, SYNTHETIC_PAGE_SIZE - 2, &[0x00, 0xf0]));
        assert!(memory.map_page(SPACE, CODE, PAGE_1, MemoryPermissions::READ_EXECUTE));

        let fault = memory
            .fetch_t32_32(SPACE, GuestVirtualAddress::new(0x1ffe))
            .unwrap_err();

        assert_eq!(fault.address, GuestVirtualAddress::new(0x2000));
        assert_eq!(fault.reason, InstructionFetchFaultReason::Unmapped);
    }

    #[test]
    fn data_accesses_enforce_permissions_alignment_and_fault_injection() {
        let mut memory = code_memory();
        assert!(memory.map_page(SPACE, ALIAS, PAGE_1, MemoryPermissions::READ_WRITE));
        let misaligned_access = MemoryAccess::normal(MemoryAccessSize::Word);
        let misaligned = memory
            .read(SPACE, ALIAS.checked_add(2).unwrap(), misaligned_access)
            .unwrap_err();
        assert_eq!(
            misaligned.reason,
            DataAccessFaultReason::Misaligned {
                required_alignment: 4
            }
        );

        memory.inject_data_fault(SPACE, ALIAS, DataAccessKind::Read, "test bus error");
        let injected = memory.read(SPACE, ALIAS, misaligned_access).unwrap_err();
        assert_eq!(
            injected.reason,
            DataAccessFaultReason::Injected("test bus error".into())
        );

        let denied = memory
            .write(
                SPACE,
                CODE,
                MemoryAccess::normal(MemoryAccessSize::Byte),
                MemoryValue::U8(1),
            )
            .unwrap_err();
        assert_eq!(denied.reason, DataAccessFaultReason::WritePermissionDenied);
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    enum MmioEvent {
        Read(u64, MemoryAccess),
        Write(u64, MemoryAccess, MemoryValue),
    }

    struct RecordingMmio {
        events: Rc<RefCell<Vec<MmioEvent>>>,
    }

    impl SyntheticMmio for RecordingMmio {
        fn read(&mut self, offset: u64, access: MemoryAccess) -> Result<MemoryValue, Box<str>> {
            self.events
                .borrow_mut()
                .push(MmioEvent::Read(offset, access));
            Ok(MemoryValue::U32(0xaabb_ccdd))
        }

        fn write(
            &mut self,
            offset: u64,
            access: MemoryAccess,
            value: MemoryValue,
        ) -> Result<(), Box<str>> {
            self.events
                .borrow_mut()
                .push(MmioEvent::Write(offset, access, value));
            Ok(())
        }
    }

    #[test]
    fn mmio_results_and_callbacks_remain_observable() {
        let device_page = GuestPhysicalPageId::new(99);
        let device_address = GuestVirtualAddress::new(0x9000);
        let events = Rc::new(RefCell::new(Vec::new()));
        let mut memory = SyntheticMemory::new();
        assert!(memory.add_mmio_page(
            device_page,
            RecordingMmio {
                events: Rc::clone(&events)
            }
        ));
        assert!(memory.map_page(
            SPACE,
            device_address,
            device_page,
            MemoryPermissions::READ_WRITE
        ));
        let access = MemoryAccess::new(
            MemoryAccessSize::Word,
            MemoryAlignment::Natural,
            MemoryOrdering::AcquireRelease,
            MemoryAccessClass::Volatile,
        );

        let read = memory.read(SPACE, device_address, access).unwrap();
        let write = memory
            .write(SPACE, device_address, access, MemoryValue::U32(5))
            .unwrap();

        assert_eq!(
            read,
            DataReadResult {
                value: MemoryValue::U32(0xaabb_ccdd),
                region: MemoryRegionKind::Device
            }
        );
        assert_eq!(write.region, MemoryRegionKind::Device);
        assert_eq!(
            *events.borrow(),
            vec![
                MmioEvent::Read(0, access),
                MmioEvent::Write(0, access, MemoryValue::U32(5))
            ]
        );
    }

    #[test]
    fn injected_fetch_fault_never_synthesizes_zero_bytes() {
        let mut memory = code_memory();
        memory.inject_instruction_fault(SPACE, CODE, "synthetic instruction abort");

        let fault = memory.fetch32(SPACE, CODE).unwrap_err();

        assert_eq!(
            fault.reason,
            InstructionFetchFaultReason::Memory("synthetic instruction abort".into())
        );
    }

    #[test]
    fn atomic_page_install_rejects_identity_exhaustion_without_changes() {
        let mut memory = SyntheticMemory::new();
        memory.inner.get_mut().next_page_id = u64::MAX;
        let bytes = [0x5a; SYNTHETIC_PAGE_SIZE];
        let request = SyntheticRamPage {
            virtual_address: CODE,
            bytes: &bytes,
            permissions: MemoryPermissions::READ_EXECUTE,
        };

        let error = memory
            .install_ram_pages_atomic(SPACE, &[request])
            .unwrap_err();

        assert_eq!(error.stage, SyntheticInstallStage::Allocation);
        assert_eq!(memory.physical_page_count(), 0);
        assert!(memory.mapping_info(SPACE, CODE).is_none());
    }

    #[test]
    fn atomic_page_install_rejects_malformed_and_duplicate_requests() {
        let bytes = [0x5a; SYNTHETIC_PAGE_SIZE];
        let valid = SyntheticRamPage {
            virtual_address: CODE,
            bytes: &bytes,
            permissions: MemoryPermissions::READ,
        };
        let malformed = [
            SyntheticRamPage {
                virtual_address: CODE.checked_add(1).unwrap(),
                ..valid
            },
            SyntheticRamPage {
                bytes: &bytes[..SYNTHETIC_PAGE_SIZE - 1],
                ..valid
            },
        ];
        for request in malformed {
            let mut memory = SyntheticMemory::new();
            let error = memory
                .install_ram_pages_atomic(SPACE, &[request])
                .unwrap_err();
            assert_eq!(error.stage, SyntheticInstallStage::Preflight);
            assert_eq!(memory.physical_page_count(), 0);
        }

        let mut memory = SyntheticMemory::new();
        let error = memory
            .install_ram_pages_atomic(SPACE, &[valid, valid])
            .unwrap_err();
        assert_eq!(error.stage, SyntheticInstallStage::Preflight);
        assert_eq!(memory.physical_page_count(), 0);
        assert!(memory.mapping_info(SPACE, CODE).is_none());
    }
}
