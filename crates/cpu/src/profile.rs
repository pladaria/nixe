//! Immutable guest CPU behavior profiles.

use core::fmt;

/// Stable identity of an immutable guest CPU profile.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct CpuProfileId(u64);

impl CpuProfileId {
    /// Creates a profile identity from its registry value.
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the registry value.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl fmt::Display for CpuProfileId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "profile=0x{:016x}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_identity_has_an_unambiguous_format() {
        assert_eq!(
            CpuProfileId::new(7).to_string(),
            "profile=0x0000000000000007"
        );
    }
}
