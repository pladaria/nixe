//! Explicit control-flow terminators for one translation unit.

use crate::{
    address::GuestVirtualAddress,
    location::{ExecutionState, InstructionEncoding, LocationDescriptor},
};

use super::value::Operand;

/// Direct or computed guest control target.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ControlTarget {
    /// Statically known guest PC and execution state.
    Direct {
        pc: GuestVirtualAddress,
        execution_state: ExecutionState,
    },
    /// Guest address computed by the block; state is explicit for interworking.
    Indirect {
        address: Operand,
        execution_state: ExecutionState,
    },
    /// A32 `BX`/`BLX`-style target whose bit zero selects T32 versus A32 at
    /// runtime. The backend masks the address according to the selected state.
    A32Interworking { address: Operand },
}

/// Architectural exception exit classification.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ExceptionKind {
    SupervisorCall,
    Breakpoint,
    UndefinedInstruction,
    InstructionAbort,
    DataAbort,
    AlignmentFault,
    FloatingPoint,
    SystemRegisterTrap,
}

/// Reason execution stops without choosing another guest block.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum StopReason {
    DispatchBudgetExhausted,
    PendingEvent,
    DebugRequest,
    ProcessExit,
    TranslationLimit,
}

/// Exactly one control-flow exit ending an IR block.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum Terminator {
    /// Unconditional direct branch.
    Direct { target: ControlTarget },
    /// Conditional direct branch with an explicit fallthrough.
    Conditional {
        condition: Operand,
        taken: ControlTarget,
        fallthrough: ControlTarget,
    },
    /// Unconditional computed branch.
    Indirect { target: ControlTarget },
    /// Call after the frontend has represented the architectural link update.
    Call {
        target: ControlTarget,
        return_address: GuestVirtualAddress,
    },
    /// Architectural return through a computed or direct target.
    Return { target: ControlTarget },
    /// Precise synchronous architectural exception.
    Exception {
        source: LocationDescriptor,
        kind: ExceptionKind,
        syndrome: Option<u64>,
    },
    /// Execute exactly one instruction using the reference interpreter.
    InterpretOne {
        source: LocationDescriptor,
        encoding: InstructionEncoding,
        coverage_id: u32,
    },
    /// Deterministic report for an instruction implemented by neither engine.
    UnsupportedInstruction {
        source: LocationDescriptor,
        encoding: InstructionEncoding,
        /// Deterministic decoder-produced disassembly.
        disassembly: Box<str>,
        /// Why neither execution engine can handle the instruction.
        reason: Box<str>,
    },
    /// Non-architectural dispatch/runtime stop.
    Stop {
        source: LocationDescriptor,
        reason: StopReason,
    },
}

impl Terminator {
    /// Returns the precise source for exits which are tied to one instruction.
    #[must_use]
    pub const fn source(&self) -> Option<LocationDescriptor> {
        match self {
            Self::Exception { source, .. }
            | Self::InterpretOne { source, .. }
            | Self::UnsupportedInstruction { source, .. }
            | Self::Stop { source, .. } => Some(*source),
            Self::Direct { .. }
            | Self::Conditional { .. }
            | Self::Indirect { .. }
            | Self::Call { .. }
            | Self::Return { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{address::GuestVirtualAddress, profile::CpuProfileId};

    fn location() -> LocationDescriptor {
        LocationDescriptor::new(
            GuestVirtualAddress::new(0x1000),
            ExecutionState::T32,
            CpuProfileId::new(1),
        )
    }

    #[test]
    fn exception_and_fallback_terminators_preserve_source_context() {
        let exception = Terminator::Exception {
            source: location(),
            kind: ExceptionKind::SupervisorCall,
            syndrome: Some(4),
        };
        let fallback = Terminator::InterpretOne {
            source: location(),
            encoding: InstructionEncoding::from_u16(0xbf00),
            coverage_id: 7,
        };
        let unsupported = Terminator::UnsupportedInstruction {
            source: location(),
            encoding: InstructionEncoding::from_u16(0xffff),
            disassembly: "udf #255".into(),
            reason: "missing semantics".into(),
        };

        assert_eq!(exception.source(), Some(location()));
        assert_eq!(fallback.source(), Some(location()));
        assert_eq!(unsupported.source(), Some(location()));
    }

    #[test]
    fn direct_conditional_indirect_call_and_return_are_distinct() {
        let direct = ControlTarget::Direct {
            pc: GuestVirtualAddress::new(0x2000),
            execution_state: ExecutionState::A64,
        };
        let indirect = ControlTarget::Indirect {
            address: super::super::value::Immediate::Address(GuestVirtualAddress::new(0x3000))
                .into(),
            execution_state: ExecutionState::A64,
        };
        let interworking = ControlTarget::A32Interworking {
            address: super::super::value::Immediate::Address(GuestVirtualAddress::new(0x3001))
                .into(),
        };
        let exits = [
            Terminator::Direct { target: direct },
            Terminator::Indirect { target: indirect },
            Terminator::Call {
                target: direct,
                return_address: GuestVirtualAddress::new(0x1004),
            },
            Terminator::Return { target: indirect },
        ];

        assert_eq!(exits.len(), 4);
        assert!(matches!(exits[0], Terminator::Direct { .. }));
        assert!(matches!(exits[1], Terminator::Indirect { .. }));
        assert!(matches!(exits[2], Terminator::Call { .. }));
        assert!(matches!(exits[3], Terminator::Return { .. }));
        assert!(matches!(
            interworking,
            ControlTarget::A32Interworking { .. }
        ));
    }
}
