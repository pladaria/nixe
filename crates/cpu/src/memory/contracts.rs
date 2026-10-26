//! Architectural memory descriptors used by decoding and diagnostics.
pub use nixe_memory::MemoryPermissions;

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

/// Engine-neutral cache-maintenance operation issued by a guest CPU.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CacheMaintenanceKind {
    InstructionInvalidate,
    DataInvalidate,
    DataClean,
    DataCleanAndInvalidate,
    InstructionPrefetch,
}
