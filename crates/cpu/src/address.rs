//! Guest address-domain and identity types.

use core::fmt;

macro_rules! opaque_u64 {
    ($(#[$meta:meta])* $name:ident, $display_prefix:literal) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
        #[repr(transparent)]
        pub struct $name(u64);

        impl $name {
            /// Creates an identity from its runtime-owned numeric value.
            #[must_use]
            pub const fn new(value: u64) -> Self {
                Self(value)
            }

            /// Returns the numeric value without changing its domain.
            #[must_use]
            pub const fn get(self) -> u64 {
                self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, concat!($display_prefix, "0x{:016x}"), self.0)
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
    /// Use this only when the applicable Arm rule requires modulo-2^64
    /// arithmetic. Normal frontend address traversal must use `checked_add`.
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
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0x{:016x}", self.0)
    }
}

opaque_u64!(
    /// Stable identity of a physical page containing guest code or data.
    GuestPhysicalPageId,
    "page="
);
opaque_u64!(
    /// Runtime-assigned identity of a process address space.
    AddressSpaceId,
    "address-space="
);
opaque_u64!(
    /// Monotonic code-content generation associated with a mapped page.
    CodeGeneration,
    "generation="
);

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
    fn alignment_rejects_invalid_alignment_values() {
        let address = GuestVirtualAddress::new(0x1004);

        assert!(address.is_aligned_to(4));
        assert!(!address.is_aligned_to(8));
        assert!(!address.is_aligned_to(0));
        assert!(!address.is_aligned_to(3));
    }

    #[test]
    fn domains_have_unambiguous_diagnostic_formats() {
        assert_eq!(
            GuestVirtualAddress::new(0x1234).to_string(),
            "0x0000000000001234"
        );
        assert_eq!(
            GuestPhysicalPageId::new(1).to_string(),
            "page=0x0000000000000001"
        );
        assert_eq!(
            AddressSpaceId::new(2).to_string(),
            "address-space=0x0000000000000002"
        );
        assert_eq!(
            CodeGeneration::new(3).to_string(),
            "generation=0x0000000000000003"
        );
    }
}
