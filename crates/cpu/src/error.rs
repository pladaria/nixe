//! Structured CPU frontend diagnostics.

use core::fmt;

use crate::{
    address::{AddressSpaceId, GuestVirtualAddress},
    location::{InstructionEncoding, LocationDescriptor},
};

/// Reproduction context shared by instruction decode diagnostics.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct InstructionDiagnostic {
    /// Architectural instruction location.
    pub location: LocationDescriptor,
    /// Raw encoding consumed by the decoder.
    pub encoding: InstructionEncoding,
}

impl InstructionDiagnostic {
    /// Creates complete instruction diagnostic context.
    #[must_use]
    pub const fn new(location: LocationDescriptor, encoding: InstructionEncoding) -> Self {
        Self { location, encoding }
    }
}

impl fmt::Display for InstructionDiagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} encoding={}", self.location, self.encoding)
    }
}

/// Architectural reason why instruction bytes could not be fetched.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum InstructionFetchFaultReason {
    /// No mapping covers the requested bytes.
    Unmapped,
    /// The mapping does not grant instruction execution permission.
    ExecutePermissionDenied,
    /// The PC violates the current execution state's alignment.
    Misaligned {
        /// Required byte alignment.
        required_alignment: u8,
    },
    /// Fetch crossed into a page which could not supply the remaining bytes.
    IncompleteCrossPageFetch,
    /// The memory implementation reported a platform-specific failure.
    Memory(Box<str>),
}

impl fmt::Display for InstructionFetchFaultReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unmapped => f.write_str("unmapped address"),
            Self::ExecutePermissionDenied => f.write_str("execute permission denied"),
            Self::Misaligned { required_alignment } => write!(
                f,
                "misaligned instruction address (requires {required_alignment}-byte alignment)"
            ),
            Self::IncompleteCrossPageFetch => f.write_str("incomplete cross-page fetch"),
            Self::Memory(reason) => f.write_str(reason),
        }
    }
}

/// Precise failure to fetch an instruction from a process address space.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct InstructionFetchFault {
    /// Address space in which the fetch was attempted.
    pub address_space: AddressSpaceId,
    /// Address of the first unavailable or invalid byte.
    pub address: GuestVirtualAddress,
    /// Architectural or memory-system reason for the fault.
    pub reason: InstructionFetchFaultReason,
}

impl InstructionFetchFault {
    /// Creates a structured instruction-fetch fault.
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
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "instruction fetch fault: pc={} state=<unavailable> profile=<unavailable> \
             encoding=<unavailable> {} reason={}",
            self.address, self.address_space, self.reason
        )
    }
}

impl std::error::Error for InstructionFetchFault {}

/// Classification of a failure after an encoding reached a decoder.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum DecodeFailureKind {
    /// The instruction is recognized but its semantic implementation is absent.
    RecognizedButNotImplemented,
    /// Operand or reserved-bit constraints made an otherwise matched encoding invalid.
    InvalidOperands,
    /// No decoder rule could classify the input due to incomplete frontend coverage.
    NoMatchingRule,
}

impl fmt::Display for DecodeFailureKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::RecognizedButNotImplemented => "recognized but not implemented",
            Self::InvalidOperands => "invalid operands or reserved bits",
            Self::NoMatchingRule => "no matching decoder rule",
        })
    }
}

/// Decoder failure distinct from profile rejection and unallocated encodings.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct DecodeFailure {
    /// PC, state, profile, and raw bits required to reproduce the failure.
    pub instruction: InstructionDiagnostic,
    /// Stable high-level classification.
    pub kind: DecodeFailureKind,
    /// Decoder-specific detail.
    pub reason: Box<str>,
}

impl DecodeFailure {
    /// Creates a structured decode failure.
    #[must_use]
    pub fn new(
        instruction: InstructionDiagnostic,
        kind: DecodeFailureKind,
        reason: impl Into<Box<str>>,
    ) -> Self {
        Self {
            instruction,
            kind,
            reason: reason.into(),
        }
    }
}

impl fmt::Display for DecodeFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "decode failure: {} reason={} ({})",
            self.instruction, self.kind, self.reason
        )
    }
}

impl std::error::Error for DecodeFailure {}

/// A known instruction whose required capability is disabled by the profile.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ProfileDisabledInstruction {
    /// PC, state, profile, and raw bits required to reproduce the rejection.
    pub instruction: InstructionDiagnostic,
    /// Stable name of the missing profile capability.
    pub required_feature: Box<str>,
    /// Optional detail about the profile restriction.
    pub reason: Box<str>,
}

impl ProfileDisabledInstruction {
    /// Creates a profile-disabled instruction diagnostic.
    #[must_use]
    pub fn new(
        instruction: InstructionDiagnostic,
        required_feature: impl Into<Box<str>>,
        reason: impl Into<Box<str>>,
    ) -> Self {
        Self {
            instruction,
            required_feature: required_feature.into(),
            reason: reason.into(),
        }
    }
}

impl fmt::Display for ProfileDisabledInstruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "profile-disabled instruction: {} feature={} reason={}",
            self.instruction, self.required_feature, self.reason
        )
    }
}

impl std::error::Error for ProfileDisabledInstruction {}

/// An encoding architecturally classified as unallocated.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct UnallocatedEncoding {
    /// PC, state, profile, and raw bits required to reproduce the rejection.
    pub instruction: InstructionDiagnostic,
    /// Constraint or encoding-space reason for the classification.
    pub reason: Box<str>,
}

impl UnallocatedEncoding {
    /// Creates an unallocated-encoding diagnostic.
    #[must_use]
    pub fn new(instruction: InstructionDiagnostic, reason: impl Into<Box<str>>) -> Self {
        Self {
            instruction,
            reason: reason.into(),
        }
    }
}

impl fmt::Display for UnallocatedEncoding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "unallocated encoding: {} reason={}",
            self.instruction, self.reason
        )
    }
}

impl std::error::Error for UnallocatedEncoding {}

/// IR verification failure with optional source-instruction context.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct InvalidIr {
    /// Source instruction when the invalid construct belongs to one operation.
    pub instruction: Option<InstructionDiagnostic>,
    /// Deterministic verifier reason.
    pub reason: Box<str>,
}

impl InvalidIr {
    /// Creates an IR verification failure.
    #[must_use]
    pub fn new(instruction: Option<InstructionDiagnostic>, reason: impl Into<Box<str>>) -> Self {
        Self {
            instruction,
            reason: reason.into(),
        }
    }
}

impl fmt::Display for InvalidIr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("invalid IR: ")?;
        format_optional_instruction(f, self.instruction)?;
        write!(f, " reason={}", self.reason)
    }
}

impl std::error::Error for InvalidIr {}

/// Frontend invariant failure which should be treated as an implementation bug.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct FrontendInternalError {
    /// Source instruction when known.
    pub instruction: Option<InstructionDiagnostic>,
    /// Deterministic description of the violated invariant.
    pub reason: Box<str>,
}

impl FrontendInternalError {
    /// Creates an internal frontend error.
    #[must_use]
    pub fn new(instruction: Option<InstructionDiagnostic>, reason: impl Into<Box<str>>) -> Self {
        Self {
            instruction,
            reason: reason.into(),
        }
    }
}

impl fmt::Display for FrontendInternalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("frontend internal error: ")?;
        format_optional_instruction(f, self.instruction)?;
        write!(f, " reason={}", self.reason)
    }
}

impl std::error::Error for FrontendInternalError {}

fn format_optional_instruction(
    f: &mut fmt::Formatter<'_>,
    instruction: Option<InstructionDiagnostic>,
) -> fmt::Result {
    match instruction {
        Some(instruction) => write!(f, "{instruction}"),
        None => f.write_str(
            "pc=<unavailable> state=<unavailable> profile=<unavailable> encoding=<unavailable>",
        ),
    }
}

/// Error surface returned by frontend translation entry points.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum FrontendError {
    /// Instruction memory could not supply executable bytes.
    InstructionFetch(InstructionFetchFault),
    /// The decoder could not produce semantics for an encoding.
    Decode(DecodeFailure),
    /// The profile rejected a recognized instruction.
    ProfileDisabled(ProfileDisabledInstruction),
    /// The architecture classifies the encoding as unallocated.
    Unallocated(UnallocatedEncoding),
    /// Produced or supplied IR failed verification.
    InvalidIr(InvalidIr),
    /// A frontend implementation invariant was violated.
    Internal(FrontendInternalError),
}

impl fmt::Display for FrontendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InstructionFetch(error) => error.fmt(f),
            Self::Decode(error) => error.fmt(f),
            Self::ProfileDisabled(error) => error.fmt(f),
            Self::Unallocated(error) => error.fmt(f),
            Self::InvalidIr(error) => error.fmt(f),
            Self::Internal(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for FrontendError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(match self {
            Self::InstructionFetch(error) => error,
            Self::Decode(error) => error,
            Self::ProfileDisabled(error) => error,
            Self::Unallocated(error) => error,
            Self::InvalidIr(error) => error,
            Self::Internal(error) => error,
        })
    }
}

macro_rules! impl_frontend_error_from {
    ($source:ty, $variant:ident) => {
        impl From<$source> for FrontendError {
            fn from(error: $source) -> Self {
                Self::$variant(error)
            }
        }
    };
}

impl_frontend_error_from!(InstructionFetchFault, InstructionFetch);
impl_frontend_error_from!(DecodeFailure, Decode);
impl_frontend_error_from!(ProfileDisabledInstruction, ProfileDisabled);
impl_frontend_error_from!(UnallocatedEncoding, Unallocated);
impl_frontend_error_from!(InvalidIr, InvalidIr);
impl_frontend_error_from!(FrontendInternalError, Internal);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{location::ExecutionState, profile::CpuProfileId};

    fn instruction() -> InstructionDiagnostic {
        InstructionDiagnostic::new(
            LocationDescriptor::new(
                GuestVirtualAddress::new(0x0071_0000_1000),
                ExecutionState::A64,
                CpuProfileId::new(2),
            ),
            InstructionEncoding::from_u32(0xd503_201f),
        )
    }

    #[test]
    fn decode_diagnostic_contains_complete_reproduction_context() {
        let error = DecodeFailure::new(
            instruction(),
            DecodeFailureKind::RecognizedButNotImplemented,
            "semantic handler is pending",
        );

        assert_eq!(
            error.to_string(),
            "decode failure: pc=0x0000007100001000 state=A64 \
             profile=0x0000000000000002 encoding=0xd503201f \
             reason=recognized but not implemented (semantic handler is pending)"
        );
    }

    #[test]
    fn distinct_error_classes_remain_machine_matchable() {
        let disabled = FrontendError::from(ProfileDisabledInstruction::new(
            instruction(),
            "FEAT_X",
            "disabled by selected profile",
        ));
        let unallocated = FrontendError::from(UnallocatedEncoding::new(
            instruction(),
            "reserved encoding field",
        ));

        assert!(matches!(disabled, FrontendError::ProfileDisabled(_)));
        assert!(matches!(unallocated, FrontendError::Unallocated(_)));
    }

    #[test]
    fn fetch_fault_reports_unavailable_decode_context_explicitly() {
        let error = InstructionFetchFault::new(
            AddressSpaceId::new(9),
            GuestVirtualAddress::new(0x1002),
            InstructionFetchFaultReason::Misaligned {
                required_alignment: 4,
            },
        );

        assert_eq!(
            error.to_string(),
            "instruction fetch fault: pc=0x0000000000001002 state=<unavailable> \
             profile=<unavailable> encoding=<unavailable> \
             address-space=0x0000000000000009 reason=misaligned instruction address \
             (requires 4-byte alignment)"
        );
    }

    #[test]
    fn block_level_ir_errors_mark_missing_instruction_context() {
        let error = InvalidIr::new(None, "block has no terminator");

        assert_eq!(
            error.to_string(),
            "invalid IR: pc=<unavailable> state=<unavailable> profile=<unavailable> \
             encoding=<unavailable> reason=block has no terminator"
        );
    }
}
