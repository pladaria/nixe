//! Stable identities used to measure decoder and semantic coverage.

use core::fmt;

/// Stable, explicitly assigned identity for one architectural instruction.
///
/// Values are grouped by execution state and must not be renumbered when table
/// entries move. They are suitable for counters, profiles, and test reports.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct CoverageId(u32);

impl CoverageId {
    /// Creates an ID assigned by an instruction table.
    #[must_use]
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    /// Returns the stable numeric value.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

impl fmt::Display for CoverageId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "insn-{:08x}", self.0)
    }
}
