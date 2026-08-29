//! Shared contracts and value types for CPU memory backends.

use std::fmt::{Display, Formatter};

use nixe_memory::{
    AddressSpaceId, ContentGeneration, ContentMutationEpoch, GuestPhysicalPageId,
    GuestVirtualAddress, MappingGeneration,
};

use crate::error::InstructionFetchFault;

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
/// process mappings may use different page sizes, and frontend region formation
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
    /// Returns the latest completely published canonical-content mutation.
    ///
    /// This address-space-wide value is an O(1) rejection filter. Exact code
    /// dependencies remain authoritative when a consumer observes a change.
    fn content_mutation_epoch(&self) -> ContentMutationEpoch;

    /// Returns the virtual code-page extent containing `address`.
    ///
    /// Translators use this only as a block-cut boundary. Fetch methods remain
    /// authoritative for mapping, permission, byte, and generation checks.
    fn code_page_span(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
    ) -> Result<CodePageSpan, InstructionFetchFault>;

    /// Fetches one A64 word at a four-byte-aligned address.
    fn fetch32(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
    ) -> Result<FetchedCode<u32>, InstructionFetchFault>;
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

impl MemoryOrdering {
    /// Returns whether operations before this access must be published before
    /// its write component.
    #[must_use]
    pub const fn has_release(self) -> bool {
        matches!(
            self,
            Self::Release | Self::AcquireRelease | Self::SequentiallyConsistent
        )
    }

    /// Returns whether operations after this access must observe its read
    /// component before proceeding.
    #[must_use]
    pub const fn has_acquire(self) -> bool {
        matches!(
            self,
            Self::Acquire | Self::AcquireRelease | Self::SequentiallyConsistent
        )
    }
}

pub(crate) fn begin_ordered_write(ordering: MemoryOrdering) {
    use core::sync::atomic::{Ordering, fence};

    match ordering {
        MemoryOrdering::SequentiallyConsistent => fence(Ordering::SeqCst),
        ordering if ordering.has_release() => fence(Ordering::Release),
        _ => {}
    }
}

pub(crate) fn complete_ordered_read(ordering: MemoryOrdering) {
    use core::sync::atomic::{Ordering, fence};

    match ordering {
        MemoryOrdering::SequentiallyConsistent => fence(Ordering::SeqCst),
        ordering if ordering.has_acquire() => fence(Ordering::Acquire),
        _ => {}
    }
}

/// Shareability domain encoded by an architectural memory barrier.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum BarrierDomain {
    NonShareable,
    InnerShareable,
    OuterShareable,
    FullSystem,
}

/// Access directions ordered by an architectural memory barrier.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum BarrierAccess {
    Reads,
    Writes,
    ReadsAndWrites,
}

/// Engine-neutral barrier semantics shared by interpreters, JITs, and future
/// native-code execution providers.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum BarrierOperation {
    DataMemory {
        domain: BarrierDomain,
        access: BarrierAccess,
    },
    DataSynchronization {
        domain: BarrierDomain,
        access: BarrierAccess,
    },
    InstructionSynchronization,
}

/// Applies the portable host-fence baseline for an architectural barrier.
/// Engines with a native guest execution context may map the same neutral
/// descriptor directly instead.
pub fn apply_host_memory_barrier(barrier: BarrierOperation) {
    use core::sync::atomic::{Ordering, compiler_fence, fence};

    match barrier {
        BarrierOperation::DataMemory { access, .. }
        | BarrierOperation::DataSynchronization { access, .. } => fence(match access {
            BarrierAccess::Reads => Ordering::Acquire,
            BarrierAccess::Writes => Ordering::Release,
            BarrierAccess::ReadsAndWrites => Ordering::AcqRel,
        }),
        BarrierOperation::InstructionSynchronization => compiler_fence(Ordering::SeqCst),
    }
}

/// Atomic read/modify/write function, independent of any engine lowering.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum AtomicRmwKind {
    Add,
    Clear,
    Xor,
    Set,
    SignedMaximum,
    SignedMinimum,
    UnsignedMaximum,
    UnsignedMinimum,
    Swap,
}

impl AtomicRmwKind {
    /// Applies the architectural fixed-width transform. Width mismatch is
    /// rejected by the memory authority before any write can occur.
    #[must_use]
    pub fn apply(self, current: MemoryValue, operand: MemoryValue) -> Option<MemoryValue> {
        if current.size() != operand.size() {
            return None;
        }
        let size = current.size();
        let bits = size.bytes() * 8;
        let mask = if bits == 128 {
            u128::MAX
        } else {
            (1_u128 << bits) - 1
        };
        let current_bits = current.bits();
        let operand_bits = operand.bits();
        let signed_key = |value: u128| value ^ (1_u128 << (bits - 1));
        let result = match self {
            Self::Add => current_bits.wrapping_add(operand_bits) & mask,
            Self::Clear => current_bits & !operand_bits & mask,
            Self::Xor => current_bits ^ operand_bits,
            Self::Set => current_bits | operand_bits,
            Self::SignedMaximum => {
                if signed_key(current_bits) >= signed_key(operand_bits) {
                    current_bits
                } else {
                    operand_bits
                }
            }
            Self::SignedMinimum => {
                if signed_key(current_bits) <= signed_key(operand_bits) {
                    current_bits
                } else {
                    operand_bits
                }
            }
            Self::UnsignedMaximum => current_bits.max(operand_bits),
            Self::UnsignedMinimum => current_bits.min(operand_bits),
            Self::Swap => operand_bits,
        };
        Some(MemoryValue::from_bits(size, result))
    }
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

    /// Returns the zero-extended bit pattern.
    #[must_use]
    pub const fn bits(self) -> u128 {
        match self {
            Self::U8(value) => value as u128,
            Self::U16(value) => value as u128,
            Self::U32(value) => value as u128,
            Self::U64(value) => value as u128,
            Self::U128(value) => value,
        }
    }

    /// Constructs a value of `size`, truncating high bits architecturally.
    #[must_use]
    pub const fn from_bits(size: MemoryAccessSize, bits: u128) -> Self {
        match size {
            MemoryAccessSize::Byte => Self::U8(bits as u8),
            MemoryAccessSize::Halfword => Self::U16(bits as u16),
            MemoryAccessSize::Word => Self::U32(bits as u32),
            MemoryAccessSize::Doubleword => Self::U64(bits as u64),
            MemoryAccessSize::Quadword => Self::U128(bits),
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
    Stack,
    Heap,
    SharedMemory,
}

impl MemoryMappingPurpose {
    /// Returns whether the runtime classifies the mapping as guest code.
    #[must_use]
    pub const fn is_code(self) -> bool {
        matches!(
            self,
            Self::CodeStatic | Self::CodeMutable | Self::ModuleCodeStatic | Self::ModuleCodeMutable
        )
    }

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

/// Complete guest-visible metadata attached to one ordinary virtual mapping.
///
/// Mapping transactions compare these properties before mutating anything so
/// platform policy cannot accidentally apply a stale address-space decision.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MemoryMappingProperties {
    pub permissions: MemoryPermissions,
    pub purpose: MemoryMappingPurpose,
    pub attributes: MemoryAttributes,
}

impl MemoryMappingProperties {
    #[must_use]
    pub const fn new(
        permissions: MemoryPermissions,
        purpose: MemoryMappingPurpose,
        attributes: MemoryAttributes,
    ) -> Self {
        Self {
            permissions,
            purpose,
            attributes,
        }
    }
}

/// One fully specified atomic alias transition.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MemoryAliasRequest {
    pub address_space: AddressSpaceId,
    pub destination: GuestVirtualAddress,
    pub source: GuestVirtualAddress,
    pub size: u64,
    pub source_before: MemoryMappingProperties,
    pub source_after: MemoryMappingProperties,
    pub destination_properties: MemoryMappingProperties,
}

impl MemoryAliasRequest {
    #[must_use]
    pub const fn new(
        address_space: AddressSpaceId,
        destination: GuestVirtualAddress,
        source: GuestVirtualAddress,
        size: u64,
        source_before: MemoryMappingProperties,
        source_after: MemoryMappingProperties,
        destination_properties: MemoryMappingProperties,
    ) -> Self {
        Self {
            address_space,
            destination,
            source,
            size,
            source_before,
            source_after,
            destination_properties,
        }
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

/// Result of one indivisible atomic memory transaction.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct AtomicMemoryResult {
    /// Value observed before the optional write.
    pub previous: MemoryValue,
    /// Whether the transaction committed a write. RMW operations always do;
    /// compare/exchange reports false when comparison failed.
    pub stored: bool,
    /// Kind of backing that serviced the transaction.
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
    /// Atomic operations require the atomic class and natural alignment.
    InvalidAtomicAccess,
    /// Atomicity is not defined for a cross-page or device transaction.
    AtomicRegionUnsupported,
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

/// Engine-neutral cache-maintenance operation issued by a guest CPU.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CacheMaintenanceKind {
    InstructionInvalidate,
    DataInvalidate,
    DataClean,
    DataCleanAndInvalidate,
    InstructionPrefetch,
}

/// Engine-facing semantic memory contract shared by every CPU provider.
pub trait CpuMemory: InstructionMemory + nixe_memory::MemoryInvalidationSource {
    /// Returns the Linux fastmem arena used by generated native code.
    fn fastmem_view(&self, _address_space: AddressSpaceId) -> Option<nixe_memory::FastmemView> {
        None
    }

    /// Arms one ordinary CPU-visible RAM page in the native fastmem arena.
    fn arm_fastmem_page(
        &self,
        _address_space: AddressSpaceId,
        _page: GuestVirtualAddress,
        _kind: DataAccessKind,
    ) -> bool {
        false
    }

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

    /// Atomically reads, transforms, and writes one naturally aligned RAM
    /// scalar. The transaction linearizes by canonical physical identity, so
    /// every virtual alias and every CPU engine observes one modification
    /// order. Device atomics require a future explicit device contract.
    fn atomic_read_modify_write(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
        access: MemoryAccess,
        kind: AtomicRmwKind,
        operand: MemoryValue,
    ) -> Result<AtomicMemoryResult, DataAccessFault>;

    /// Atomically compares the complete scalar bit pattern and conditionally
    /// replaces it. The returned `previous` value is the architectural result.
    fn atomic_compare_exchange(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
        access: MemoryAccess,
        expected: MemoryValue,
        replacement: MemoryValue,
    ) -> Result<AtomicMemoryResult, DataAccessFault>;

    /// Applies the host-side ordering required by an architectural barrier.
    /// Domain and access remain explicit so both CPU backends implement the
    /// same architectural operation without sharing backend-private code.
    fn memory_barrier(&self, barrier: BarrierOperation) {
        apply_host_memory_barrier(barrier);
    }

    /// Applies one architecturally visible cache-maintenance operation.
    ///
    /// Canonical RAM is coherent, so data maintenance reconciles ownership
    /// rather than modelling a host cache. Instruction invalidation publishes
    /// through the neutral memory-invalidation stream consumed by every engine.
    /// Ordinary guest stores deliberately do not invalidate translated code:
    /// valid A64 self-modifying code makes new instructions visible with an IC
    /// maintenance operation.
    fn maintain_cache(
        &self,
        address_space: AddressSpaceId,
        kind: CacheMaintenanceKind,
        address: Option<GuestVirtualAddress>,
    ) -> Result<(), DataAccessFault>;

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

    /// Atomically aliases existing RAM pages at an unmapped destination while
    /// replacing the source mapping metadata.
    fn map_alias(&self, request: MemoryAliasRequest) -> Result<(), MemoryAliasError>;

    /// Atomically removes an alias after verifying its physical identity and
    /// restores the source mapping metadata.
    fn unmap_alias(&self, request: MemoryAliasRequest) -> Result<(), MemoryAliasError>;

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
pub enum MemoryAliasErrorReason {
    InvalidRange,
    SourceStateMismatch,
    DestinationStateMismatch,
    PhysicalIdentityMismatch,
    ResourceExhausted,
    GenerationExhausted,
}

/// Pointer-free reason an atomic virtual alias transition was rejected.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MemoryAliasError {
    pub address_space: AddressSpaceId,
    pub address: GuestVirtualAddress,
    pub reason: MemoryAliasErrorReason,
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

#[cfg(test)]
mod tests {
    use super::{AtomicRmwKind, MemoryValue};

    #[test]
    fn every_atomic_rmw_transform_uses_the_exact_fixed_width_bit_pattern() {
        let apply = |kind: AtomicRmwKind, current: u8, operand: u8| {
            kind.apply(MemoryValue::U8(current), MemoryValue::U8(operand))
                .expect("equal-width atomic operands")
        };

        assert_eq!(apply(AtomicRmwKind::Add, 250, 10), MemoryValue::U8(4));
        assert_eq!(
            apply(AtomicRmwKind::Clear, 0b1111, 0b0101),
            MemoryValue::U8(0b1010)
        );
        assert_eq!(
            apply(AtomicRmwKind::Xor, 0b1100, 0b1010),
            MemoryValue::U8(0b0110)
        );
        assert_eq!(
            apply(AtomicRmwKind::Set, 0b1100, 0b0011),
            MemoryValue::U8(0b1111)
        );
        assert_eq!(
            apply(AtomicRmwKind::SignedMaximum, 0x80, 0x7f),
            MemoryValue::U8(0x7f)
        );
        assert_eq!(
            apply(AtomicRmwKind::SignedMinimum, 0x80, 0x7f),
            MemoryValue::U8(0x80)
        );
        assert_eq!(
            apply(AtomicRmwKind::UnsignedMaximum, 1, 2),
            MemoryValue::U8(2)
        );
        assert_eq!(
            apply(AtomicRmwKind::UnsignedMinimum, 1, 2),
            MemoryValue::U8(1)
        );
        assert_eq!(
            apply(AtomicRmwKind::Swap, 0xaa, 0x55),
            MemoryValue::U8(0x55)
        );
        assert_eq!(
            AtomicRmwKind::Add.apply(MemoryValue::U8(1), MemoryValue::U16(1)),
            None
        );
    }
}
