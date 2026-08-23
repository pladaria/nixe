//! Device-neutral guest-memory identities and version contracts.
//!
//! This crate defines pointer-free values shared by CPU, runtime, and device
//! code. It deliberately contains no CPU execution, Horizon, console GPU, or
//! host graphics API behavior.

use core::fmt;
use core::sync::atomic::{AtomicU64, Ordering};

mod access;
mod backing;
mod range;
mod visibility;

pub use access::{
    DeviceAccessDeclaration, DeviceAccessDeclarationError, DeviceAccessKind, DeviceVisibilityPoint,
    MemoryPermissions, NonCpuDeviceId,
};
pub use backing::{
    CanonicalAllocation, CanonicalAllocationError, CanonicalBackingPage, CanonicalBackingStore,
    CanonicalCpuWriteOverlap, CanonicalCpuWriteRange, CanonicalPageError, CanonicalWriteBatch,
    CanonicalWriteBatchError,
};
pub use range::{
    CanonicalBackingRange, CanonicalBackingSegment, CanonicalCpuWriteDependency,
    CanonicalRangeAccessError, CanonicalRangeError, CanonicalRangeTranslationError,
    CanonicalRangeTranslationErrorReason, CanonicalRangeTranslator,
};
pub use visibility::{
    CpuVisibilityRequest, DeviceVisibilityRequest, VisibilityCoordinator,
    VisibilityCoordinatorError, VisibilityError, VisibilityState,
};

macro_rules! opaque_u64 {
    ($(#[$meta:meta])* $name:ident, $display_prefix:literal) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
        #[repr(transparent)]
        pub struct $name(u64);

        impl $name {
            /// Creates a value from its owner-assigned numeric representation.
            #[must_use]
            pub const fn new(value: u64) -> Self {
                Self(value)
            }

            /// Returns the numeric representation without changing domains.
            #[must_use]
            pub const fn get(self) -> u64 {
                self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(
                    formatter,
                    concat!($display_prefix, "0x{:016x}"),
                    self.0
                )
            }
        }
    };
}

/// A virtual address in a guest process.
///
/// It deliberately has no `Add` or `Sub` implementation: callers must choose
/// checked arithmetic or explicitly request architectural wrapping.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct GuestVirtualAddress(u64);

impl GuestVirtualAddress {
    /// The lowest guest virtual address.
    pub const MIN: Self = Self(u64::MIN);
    /// The highest address representable by this domain type.
    pub const MAX: Self = Self(u64::MAX);

    /// Creates a guest virtual address from its architectural bit pattern.
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the architectural bit pattern.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }

    /// Adds an unsigned byte offset, returning `None` on overflow.
    #[must_use]
    pub const fn checked_add(self, byte_offset: u64) -> Option<Self> {
        match self.0.checked_add(byte_offset) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }

    /// Subtracts an unsigned byte offset, returning `None` on underflow.
    #[must_use]
    pub const fn checked_sub(self, byte_offset: u64) -> Option<Self> {
        match self.0.checked_sub(byte_offset) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }

    /// Adds a signed byte displacement, returning `None` outside the domain.
    #[must_use]
    pub const fn checked_offset(self, byte_displacement: i64) -> Option<Self> {
        match self.0.checked_add_signed(byte_displacement) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }

    /// Adds an unsigned byte offset with explicit architectural wrapping.
    ///
    /// Use this only when the applicable architecture requires modulo-2^64
    /// arithmetic. Normal memory traversal must use [`Self::checked_add`].
    #[must_use]
    pub const fn wrapping_add(self, byte_offset: u64) -> Self {
        Self(self.0.wrapping_add(byte_offset))
    }

    /// Adds a signed displacement with explicit architectural wrapping.
    #[must_use]
    pub const fn wrapping_offset(self, byte_displacement: i64) -> Self {
        Self(self.0.wrapping_add_signed(byte_displacement))
    }

    /// Returns whether the address satisfies a non-zero power-of-two alignment.
    #[must_use]
    pub const fn is_aligned_to(self, alignment: u64) -> bool {
        alignment.is_power_of_two() && self.0 & (alignment - 1) == 0
    }
}

impl fmt::Display for GuestVirtualAddress {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "0x{:016x}", self.0)
    }
}

opaque_u64!(
    /// Runtime-assigned identity of a guest CPU virtual address space.
    AddressSpaceId,
    "address-space="
);
opaque_u64!(
    /// Store-local identity of a physical page containing guest bytes.
    ///
    /// This value is stable across aliases but is not sufficient as a
    /// cross-device identity without its [`BackingStoreId`].
    GuestPhysicalPageId,
    "page="
);
opaque_u64!(
    /// Identity of one ownership domain for canonical guest backing.
    BackingStoreId,
    "backing-store="
);

static NEXT_BACKING_STORE_ID: AtomicU64 = AtomicU64::new(1);

impl BackingStoreId {
    /// Allocates a process-wide unique store identity without wraparound.
    pub fn allocate() -> Result<Self, BackingIdentityExhausted> {
        NEXT_BACKING_STORE_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1)
            })
            .map(Self)
            .map_err(|_| BackingIdentityExhausted)
    }
}

/// Failure to allocate another globally unambiguous backing-store identity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BackingIdentityExhausted;

impl fmt::Display for BackingIdentityExhausted {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("canonical backing-store identities are exhausted")
    }
}

impl std::error::Error for BackingIdentityExhausted {}

/// Cross-device identity of one canonical physical page.
///
/// The composite identity remains stable and unambiguous even when distinct
/// stores assign the same local [`GuestPhysicalPageId`].
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CanonicalPageId {
    store: BackingStoreId,
    page: GuestPhysicalPageId,
}

impl CanonicalPageId {
    /// Combines a store identity and its store-local physical page.
    #[must_use]
    pub const fn new(store: BackingStoreId, page: GuestPhysicalPageId) -> Self {
        Self { store, page }
    }

    /// Returns the backing store which owns the page.
    #[must_use]
    pub const fn store(self) -> BackingStoreId {
        self.store
    }

    /// Returns the page identity local to the backing store.
    #[must_use]
    pub const fn page(self) -> GuestPhysicalPageId {
        self.page
    }
}

impl fmt::Display for CanonicalPageId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} {}", self.store, self.page)
    }
}

/// Kind of observable generation whose numeric domain was exhausted.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GenerationKind {
    /// Version of bytes in canonical backing.
    Content,
    /// Store-wide publication epoch for canonical content mutations.
    ContentMutation,
    /// Store-wide publication epoch for CPU-originated writes.
    CpuWrite,
    /// Version of a virtual-to-backing mapping and its access metadata.
    Mapping,
}

impl fmt::Display for GenerationKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Content => "content",
            Self::ContentMutation => "content mutation",
            Self::CpuWrite => "CPU write",
            Self::Mapping => "mapping",
        })
    }
}

/// Failure to allocate a generation newer than an observable value.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GenerationExhausted {
    kind: GenerationKind,
}

impl GenerationExhausted {
    /// Returns the exhausted generation domain.
    #[must_use]
    pub const fn kind(self) -> GenerationKind {
        self.kind
    }
}

impl fmt::Display for GenerationExhausted {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} generation is exhausted", self.kind)
    }
}

impl std::error::Error for GenerationExhausted {}

macro_rules! generation {
    ($(#[$meta:meta])* $name:ident, $kind:expr, $display_prefix:literal) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
        #[repr(transparent)]
        pub struct $name(u64);

        impl $name {
            /// Initial generation for state which has not changed.
            pub const INITIAL: Self = Self(0);
            /// Largest observable generation.
            pub const MAX: Self = Self(u64::MAX);

            /// Creates a generation from a previously validated value.
            #[must_use]
            pub const fn new(value: u64) -> Self {
                Self(value)
            }

            /// Returns the numeric generation.
            #[must_use]
            pub const fn get(self) -> u64 {
                self.0
            }

            /// Returns the next generation without observable wraparound.
            pub const fn next(self) -> Result<Self, GenerationExhausted> {
                match self.0.checked_add(1) {
                    Some(value) => Ok(Self(value)),
                    None => Err(GenerationExhausted { kind: $kind }),
                }
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(formatter, concat!($display_prefix, "0x{:016x}"), self.0)
            }
        }
    };
}

generation!(
    /// Version of canonical backing bytes.
    ContentGeneration,
    GenerationKind::Content,
    "generation="
);
generation!(
    /// Store-wide epoch advanced whenever canonical content is published.
    ContentMutationEpoch,
    GenerationKind::ContentMutation,
    "content-mutation-epoch="
);
generation!(
    /// Store-wide epoch advanced only by CPU-originated canonical writes.
    CpuWriteEpoch,
    GenerationKind::CpuWrite,
    "cpu-write-epoch="
);
/// Version of a virtual mapping and its access metadata.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct MappingGeneration(u64);

impl MappingGeneration {
    /// Initial generation for state which has not changed.
    pub const INITIAL: Self = Self(0);
    /// Largest observable generation.
    pub const MAX: Self = Self(u64::MAX);

    /// Creates a generation from a previously validated value.
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the numeric generation.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }

    /// Returns the next generation without observable wraparound.
    pub const fn next(self) -> Result<Self, GenerationExhausted> {
        match self.0.checked_add(1) {
            Some(value) => Ok(Self(value)),
            None => Err(GenerationExhausted {
                kind: GenerationKind::Mapping,
            }),
        }
    }
}

impl fmt::Display for MappingGeneration {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "mapping-generation={}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guest_address_arithmetic_is_checked_by_default() {
        let address = GuestVirtualAddress::new(u64::MAX - 1);

        assert_eq!(address.checked_add(1), Some(GuestVirtualAddress::MAX));
        assert_eq!(address.checked_add(2), None);
        assert_eq!(GuestVirtualAddress::MIN.checked_sub(1), None);
        assert_eq!(GuestVirtualAddress::MIN.checked_offset(-1), None);
    }

    #[test]
    fn wrapping_address_arithmetic_is_explicit() {
        assert_eq!(
            GuestVirtualAddress::MAX.wrapping_add(1),
            GuestVirtualAddress::MIN
        );
        assert_eq!(
            GuestVirtualAddress::MIN.wrapping_offset(-1),
            GuestVirtualAddress::MAX
        );
    }

    #[test]
    fn canonical_page_identity_includes_the_owning_store() {
        let page = GuestPhysicalPageId::new(7);
        let first = CanonicalPageId::new(BackingStoreId::new(1), page);
        let second = CanonicalPageId::new(BackingStoreId::new(2), page);

        assert_ne!(first, second);
        assert_eq!(first.page(), second.page());
        assert_ne!(first.store(), second.store());
    }

    #[test]
    fn generation_domains_are_distinct_and_never_wrap() {
        assert_eq!(
            ContentGeneration::INITIAL.next(),
            Ok(ContentGeneration::new(1))
        );
        assert_eq!(
            MappingGeneration::INITIAL.next(),
            Ok(MappingGeneration::new(1))
        );
        assert_eq!(
            ContentMutationEpoch::INITIAL.next(),
            Ok(ContentMutationEpoch::new(1))
        );
        assert_eq!(CpuWriteEpoch::INITIAL.next(), Ok(CpuWriteEpoch::new(1)));
        assert_eq!(
            ContentGeneration::MAX.next(),
            Err(GenerationExhausted {
                kind: GenerationKind::Content
            })
        );
        assert_eq!(
            MappingGeneration::MAX.next(),
            Err(GenerationExhausted {
                kind: GenerationKind::Mapping
            })
        );
        assert_eq!(
            ContentMutationEpoch::MAX.next(),
            Err(GenerationExhausted {
                kind: GenerationKind::ContentMutation
            })
        );
        assert_eq!(
            CpuWriteEpoch::MAX.next(),
            Err(GenerationExhausted {
                kind: GenerationKind::CpuWrite
            })
        );
    }

    #[test]
    fn identity_formats_are_pointer_free_and_domain_specific() {
        assert_eq!(
            GuestVirtualAddress::new(0x1234).to_string(),
            "0x0000000000001234"
        );
        assert_eq!(
            CanonicalPageId::new(BackingStoreId::new(2), GuestPhysicalPageId::new(3)).to_string(),
            "backing-store=0x0000000000000002 page=0x0000000000000003"
        );
        assert_eq!(
            ContentGeneration::new(4).to_string(),
            "generation=0x0000000000000004"
        );
        assert_eq!(
            MappingGeneration::new(5).to_string(),
            "mapping-generation=5"
        );
    }
}
