//! Compatibility facade for device-neutral guest-memory domain types.
//!
//! New cross-device code should import these values from `nixe-memory`
//! directly. Existing CPU callers retain this module path while the underlying
//! identities no longer belong to the CPU crate.

pub use nixe_memory::{
    AddressSpaceId, BackingStoreId, CanonicalPageId, ContentGeneration, GuestPhysicalPageId,
    GuestVirtualAddress, MappingGeneration,
};

/// Compatibility name for the content generation used by CPU code caches and
/// exclusive reservations.
pub type CodeGeneration = ContentGeneration;

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
