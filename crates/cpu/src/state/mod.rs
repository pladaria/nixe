//! Guest architectural thread state.
//!
//! The `repr(C)` layouts in this module are an internal code-generation ABI.
//! They are deliberately not a save-state or interchange format. Persistent
//! state must use an explicitly versioned, field-by-field representation so a
//! backend layout change cannot silently change serialized data.

pub mod a32;
pub mod a64;

use crate::{location::ExecutionState, profile::ThreadCpuConfiguration};

pub use a32::A32State;
pub use a64::A64State;

/// Canonical architectural register state owned by one guest thread.
///
/// AArch32 has its own representation. Its CPSR T bit chooses A32 or T32 and
/// may change through interworking without replacing this enum variant.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ThreadCpuState {
    A64(Box<A64State>),
    A32(Box<A32State>),
}

impl ThreadCpuState {
    /// Creates zeroed architectural state from validated immutable metadata.
    #[must_use]
    pub fn new(configuration: ThreadCpuConfiguration) -> Self {
        match configuration.initial_execution_state() {
            ExecutionState::A64 => Self::A64(Box::default()),
            ExecutionState::A32 => Self::A32(Box::default()),
            ExecutionState::T32 => Self::A32(Box::new(A32State::t32())),
        }
    }

    /// Returns the instruction set currently selected by architectural state.
    #[must_use]
    pub const fn execution_state(&self) -> ExecutionState {
        match self {
            Self::A64(_) => ExecutionState::A64,
            Self::A32(state) => state.execution_state(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{address::AddressSpaceId, profile::GuestCpuProfile};

    use super::*;

    #[test]
    fn construction_uses_distinct_architectural_representations() {
        let process = crate::profile::ProcessCpuContext::new(
            GuestCpuProfile::switch_1(),
            AddressSpaceId::new(1),
        );

        let a64 = ThreadCpuState::new(process.thread_configuration(ExecutionState::A64).unwrap());
        let t32 = ThreadCpuState::new(process.thread_configuration(ExecutionState::T32).unwrap());

        assert!(matches!(a64, ThreadCpuState::A64(_)));
        assert!(matches!(t32, ThreadCpuState::A32(_)));
        assert_eq!(t32.execution_state(), ExecutionState::T32);
    }
}
