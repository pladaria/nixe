//! Reference instruction interpretation and translation fallback.
//!
//! The interpreter consumes decoded architectural instructions directly. It
//! deliberately does not execute frontend IR, making it useful as an
//! independent oracle for differential tests.

mod a32;
mod a64;
mod t32;

use core::fmt;

use crate::{
    address::GuestVirtualAddress,
    coverage::CoverageId,
    decode::{self, DecodeResult, DecodedOpcode},
    ir::terminator::{ExceptionKind, Terminator},
    location::{DecodedInstruction, ExecutionState, InstructionEncoding, LocationDescriptor},
    profile::GuestCpuProfile,
    state::ThreadCpuState,
};

/// Coverage state of one recognized architectural instruction.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum InstructionSupport {
    /// The frontend can lower the instruction to IR.
    Lifted,
    /// Only the reference interpreter currently implements the instruction.
    InterpreterOnly,
    /// The encoding is known but neither execution engine implements it.
    RecognizedUnsupported,
    /// The selected CPU profile disables a required feature.
    ProfileDisabled,
    /// The architecture classifies the encoding as unallocated or reserved.
    Unallocated,
}

/// Policy applied when dispatch reaches an `InterpretOne` terminator.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct InterpreterPolicy {
    /// Converts every JIT-to-interpreter fallback into a deterministic error.
    pub strict_fallback: bool,
}

/// Successful result of executing exactly one interpreted instruction.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum InterpreterOutcome {
    /// One instruction retired and dispatch must continue at this location.
    Resume(LocationDescriptor),
    /// The instruction raised a precise synchronous architectural exception.
    Exception {
        source: LocationDescriptor,
        kind: ExceptionKind,
        syndrome: Option<u64>,
    },
    /// The instruction handed control to the scheduler without retiring into
    /// the ordinary dispatcher path.
    Scheduled { source: LocationDescriptor },
}

/// Deterministic interpreter/fallback failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum InterpreterError {
    /// Dispatch supplied a terminator which is not an interpreter fallback.
    NotInterpreterFallback,
    /// Strict test policy rejected an otherwise valid fallback.
    StrictFallback {
        source: LocationDescriptor,
        coverage_id: CoverageId,
    },
    /// Terminator, decoded instruction, and live architectural state disagree.
    ContextMismatch {
        source: LocationDescriptor,
        reason: Box<str>,
    },
    /// No reference semantics exist for a recognized instruction.
    UnsupportedInstruction {
        source: LocationDescriptor,
        encoding: InstructionEncoding,
        disassembly: Box<str>,
        coverage_id: CoverageId,
    },
}

impl fmt::Display for InterpreterError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotInterpreterFallback => formatter.write_str("terminator is not InterpretOne"),
            Self::StrictFallback {
                source,
                coverage_id,
            } => write!(
                formatter,
                "strict interpreter fallback rejected: {source} coverage={coverage_id}"
            ),
            Self::ContextMismatch { source, reason } => {
                write!(
                    formatter,
                    "interpreter context mismatch: {source} reason={reason}"
                )
            }
            Self::UnsupportedInstruction {
                source,
                encoding,
                disassembly,
                coverage_id,
            } => write!(
                formatter,
                "unsupported instruction: {source} encoding={encoding} disassembly={disassembly} coverage={coverage_id}"
            ),
        }
    }
}

impl std::error::Error for InterpreterError {}

/// Returns the independently tracked engine coverage for a decoded opcode.
#[must_use]
pub fn instruction_support(decoded: &DecodedInstruction<DecodedOpcode>) -> InstructionSupport {
    let interpreter = crate::coverage::interpreter_coverage(
        decoded.location.execution_state,
        decoded.instruction.coverage_id(),
    );
    let lifter = crate::coverage::lifter_coverage(
        decoded.location.execution_state,
        decoded.instruction.coverage_id(),
    );
    match (interpreter, lifter) {
        (
            crate::coverage::EngineCoverage::Implemented,
            crate::coverage::EngineCoverage::Implemented,
        ) => InstructionSupport::Lifted,
        (
            crate::coverage::EngineCoverage::Implemented,
            crate::coverage::EngineCoverage::Missing,
        ) => InstructionSupport::InterpreterOnly,
        (crate::coverage::EngineCoverage::Missing, _) => InstructionSupport::RecognizedUnsupported,
    }
}

/// Returns whether the reference engine has executable semantics for this ID.
#[must_use]
pub fn has_semantics(decoded: &DecodedInstruction<DecodedOpcode>) -> bool {
    crate::coverage::interpreter_coverage(
        decoded.location.execution_state,
        decoded.instruction.coverage_id(),
    ) == crate::coverage::EngineCoverage::Implemented
}

/// Executes the instruction represented by one JIT fallback terminator.
///
/// A successful fallback always interprets exactly one instruction. It never
/// re-enters the current translated block: the returned location is consumed
/// by the outer dispatcher.
pub fn execute_fallback(
    policy: InterpreterPolicy,
    profile: &GuestCpuProfile,
    state: &mut ThreadCpuState,
    terminator: &Terminator,
) -> Result<InterpreterOutcome, InterpreterError> {
    let Terminator::InterpretOne {
        source,
        encoding,
        coverage_id,
    } = terminator
    else {
        return Err(InterpreterError::NotInterpreterFallback);
    };
    let coverage_id = CoverageId::new(*coverage_id);
    if policy.strict_fallback {
        return Err(InterpreterError::StrictFallback {
            source: *source,
            coverage_id,
        });
    }
    validate_context(profile, state, *source)?;
    let decoded = match decode::decode(profile, *source, *encoding) {
        DecodeResult::Decoded(decoded) | DecodeResult::RecognizedUnimplemented(decoded) => decoded,
        DecodeResult::Unallocated { .. } | DecodeResult::Reserved { .. } => {
            return Ok(InterpreterOutcome::Exception {
                source: *source,
                kind: ExceptionKind::UndefinedInstruction,
                syndrome: None,
            });
        }
        DecodeResult::ProfileDisabled { .. } => {
            return Ok(InterpreterOutcome::Exception {
                source: *source,
                kind: ExceptionKind::UndefinedInstruction,
                syndrome: None,
            });
        }
    };
    if decoded.instruction.coverage_id() != coverage_id {
        return Err(InterpreterError::ContextMismatch {
            source: *source,
            reason: "terminator coverage ID does not match decoded instruction".into(),
        });
    }
    execute_decoded(state, &decoded)
}

/// Executes one already-fetched instruction as a reference-engine step.
pub fn execute_one(
    profile: &GuestCpuProfile,
    state: &mut ThreadCpuState,
    encoding: InstructionEncoding,
) -> Result<InterpreterOutcome, InterpreterError> {
    let source = current_location(profile, state);
    validate_context(profile, state, source)?;
    match decode::decode(profile, source, encoding) {
        DecodeResult::Decoded(decoded) | DecodeResult::RecognizedUnimplemented(decoded) => {
            execute_decoded(state, &decoded)
        }
        DecodeResult::Unallocated { .. } | DecodeResult::Reserved { .. } => {
            Ok(InterpreterOutcome::Exception {
                source,
                kind: ExceptionKind::UndefinedInstruction,
                syndrome: None,
            })
        }
        DecodeResult::ProfileDisabled { .. } => Ok(InterpreterOutcome::Exception {
            source,
            kind: ExceptionKind::UndefinedInstruction,
            syndrome: None,
        }),
    }
}

fn execute_decoded(
    state: &mut ThreadCpuState,
    decoded: &DecodedInstruction<DecodedOpcode>,
) -> Result<InterpreterOutcome, InterpreterError> {
    match (state, decoded.location.execution_state) {
        (ThreadCpuState::A64(state), ExecutionState::A64) => a64::execute(state, decoded),
        (ThreadCpuState::A32(state), ExecutionState::A32) => a32::execute(state, decoded),
        (ThreadCpuState::A32(state), ExecutionState::T32) => t32::execute(state, decoded),
        _ => Err(InterpreterError::ContextMismatch {
            source: decoded.location,
            reason: "architectural state representation does not match execution state".into(),
        }),
    }
}

fn current_location(profile: &GuestCpuProfile, state: &ThreadCpuState) -> LocationDescriptor {
    let (pc, execution_state) = match state {
        ThreadCpuState::A64(state) => (state.pc(), ExecutionState::A64),
        ThreadCpuState::A32(state) => (
            u64::from(state.instruction_address()),
            state.execution_state(),
        ),
    };
    LocationDescriptor::new(GuestVirtualAddress::new(pc), execution_state, profile.id())
}

fn validate_context(
    profile: &GuestCpuProfile,
    state: &ThreadCpuState,
    source: LocationDescriptor,
) -> Result<(), InterpreterError> {
    if source.profile_id != profile.id() {
        return Err(InterpreterError::ContextMismatch {
            source,
            reason: "source profile does not match selected profile".into(),
        });
    }
    let current = current_location(profile, state);
    if current.pc != source.pc || current.execution_state != source.execution_state {
        return Err(InterpreterError::ContextMismatch {
            source,
            reason: "source PC or execution state does not match live state".into(),
        });
    }
    Ok(())
}

fn unsupported(decoded: &DecodedInstruction<DecodedOpcode>) -> InterpreterError {
    InterpreterError::UnsupportedInstruction {
        source: decoded.location,
        encoding: decoded.encoding,
        disassembly: decode::disassemble(&decoded.instruction).to_string().into(),
        coverage_id: decoded.instruction.coverage_id(),
    }
}

#[cfg(test)]
mod tests;
