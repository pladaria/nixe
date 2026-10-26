//! Architectural memory values and instruction-access contract.
pub use nixe_memory::MemoryPermissions;
use nixe_memory::{AddressSpaceId, GuestVirtualAddress};

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

/// Architectural barrier semantics shared by native execution and future
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

/// Typed scalar/vector bit pattern transferred by instruction memory operations.
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

impl From<&nixe_memory::MemoryError> for DataAccessFaultReason {
    fn from(error: &nixe_memory::MemoryError) -> Self {
        match error {
            nixe_memory::MemoryError::Unmapped(_) => Self::Unmapped,
            nixe_memory::MemoryError::PermissionDenied { required, .. } => {
                if required.contains(nixe_memory::MemoryPermissions::WRITE) {
                    Self::WritePermissionDenied
                } else {
                    Self::ReadPermissionDenied
                }
            }
            nixe_memory::MemoryError::AddressRange { .. } => Self::AddressOverflow,
            error => Self::HostBacking(error.to_string().into()),
        }
    }
}

/// Precise failure of a guest data access.
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

/// Architectural data accesses used by the shared instruction semantics.
/// Mapping policy and native admission belong to their concrete owners.
pub trait InstructionMemoryAccess {
    type Reservation: Copy;
    fn read(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
        access: MemoryAccess,
    ) -> Result<DataReadResult, DataAccessFault>;

    fn write(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
        access: MemoryAccess,
        value: MemoryValue,
    ) -> Result<DataWriteResult, DataAccessFault>;

    fn atomic_read_modify_write(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
        access: MemoryAccess,
        kind: AtomicRmwKind,
        operand: MemoryValue,
    ) -> Result<AtomicMemoryResult, DataAccessFault>;

    fn atomic_compare_exchange(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
        access: MemoryAccess,
        expected: MemoryValue,
        replacement: MemoryValue,
    ) -> Result<AtomicMemoryResult, DataAccessFault>;

    fn memory_barrier(&self, barrier: BarrierOperation) {
        apply_host_memory_barrier(barrier);
    }

    fn maintain_cache(
        &self,
        address_space: AddressSpaceId,
        kind: CacheMaintenanceKind,
        address: Option<GuestVirtualAddress>,
    ) -> Result<(), DataAccessFault>;

    fn load_exclusive(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
        access: MemoryAccess,
    ) -> Result<(DataReadResult, Self::Reservation), DataAccessFault>;

    fn store_exclusive(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
        access: MemoryAccess,
        value: MemoryValue,
        reservation: Self::Reservation,
    ) -> Result<(DataWriteResult, bool), DataAccessFault>;
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum MemoryAliasErrorReason {
    InvalidRange,
    SourceStateMismatch,
    DestinationStateMismatch,
    PhysicalIdentityMismatch,
    ResourceExhausted,
    GenerationExhausted,
}

/// Pointer-free reason an atomic virtual alias transition was rejected.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct MemoryAliasError {
    pub address_space: AddressSpaceId,
    pub address: GuestVirtualAddress,
    pub reason: MemoryAliasErrorReason,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum MemoryMappingErrorReason {
    InvalidRange,
    AlreadyMapped,
    MappingStateMismatch,
    WritableExecutable,
    ResourceExhausted,
    GenerationExhausted,
}

/// Pointer-free reason a runtime mapping resize was rejected.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct MemoryMappingError {
    pub address_space: AddressSpaceId,
    pub address: GuestVirtualAddress,
    pub reason: MemoryMappingErrorReason,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum MemoryProtectionErrorReason {
    InvalidRange,
    Unmapped,
    WritableExecutable,
    UnsupportedPermissions,
    PermissionLocked,
    GenerationExhausted,
}

/// Pointer-free reason a runtime mapping-protection operation was rejected.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct MemoryProtectionError {
    pub address_space: AddressSpaceId,
    pub address: GuestVirtualAddress,
    pub reason: MemoryProtectionErrorReason,
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
