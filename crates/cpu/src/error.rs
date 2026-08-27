//! Precise A64 decode and instruction-fetch diagnostics.

use core::fmt;

use nixe_memory::{AddressSpaceId, GuestVirtualAddress};

use crate::location::{InstructionEncoding, LocationDescriptor};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct InstructionDiagnostic {
    pub location: LocationDescriptor,
    pub encoding: InstructionEncoding,
}

impl InstructionDiagnostic {
    #[must_use]
    pub const fn new(location: LocationDescriptor, encoding: InstructionEncoding) -> Self {
        Self { location, encoding }
    }
}

impl fmt::Display for InstructionDiagnostic {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} encoding={}", self.location, self.encoding)
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum InstructionFetchFaultReason {
    Unmapped,
    ExecutePermissionDenied,
    Misaligned,
    IncompleteCrossPageFetch,
    AddressOverflow,
    Memory(Box<str>),
}

impl fmt::Display for InstructionFetchFaultReason {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unmapped => formatter.write_str("unmapped address"),
            Self::ExecutePermissionDenied => formatter.write_str("execute permission denied"),
            Self::Misaligned => formatter.write_str("misaligned A64 instruction address"),
            Self::IncompleteCrossPageFetch => formatter.write_str("incomplete cross-page fetch"),
            Self::AddressOverflow => formatter.write_str("instruction address overflow"),
            Self::Memory(reason) => formatter.write_str(reason),
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct InstructionFetchFault {
    pub address_space: AddressSpaceId,
    pub address: GuestVirtualAddress,
    pub reason: InstructionFetchFaultReason,
}

impl InstructionFetchFault {
    #[must_use]
    pub const fn new(
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
        reason: InstructionFetchFaultReason,
    ) -> Self {
        Self {
            address_space,
            address,
            reason,
        }
    }
}

impl fmt::Display for InstructionFetchFault {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "A64 instruction fetch fault: pc={} {} reason={}",
            self.address, self.address_space, self.reason
        )
    }
}

impl std::error::Error for InstructionFetchFault {}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct UnallocatedEncoding {
    pub instruction: InstructionDiagnostic,
    pub reason: Box<str>,
}

impl UnallocatedEncoding {
    #[must_use]
    pub fn new(instruction: InstructionDiagnostic, reason: impl Into<Box<str>>) -> Self {
        Self {
            instruction,
            reason: reason.into(),
        }
    }
}

impl fmt::Display for UnallocatedEncoding {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "unallocated encoding: {} reason={}",
            self.instruction, self.reason
        )
    }
}

impl std::error::Error for UnallocatedEncoding {}
