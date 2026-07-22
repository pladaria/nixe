//! Guest architectural thread state.
//!
//! The `repr(C)` layouts in this module are an internal code-generation ABI.
//! They are deliberately not a save-state or interchange format. Persistent
//! state must use an explicitly versioned, field-by-field representation so a
//! backend layout change cannot silently change serialized data.

pub mod a32;
pub mod a64;

use core::fmt;

use crate::address::GuestVirtualAddress;
use crate::{location::ExecutionState, profile::ThreadCpuConfiguration};

pub use a32::{A32State, Cpsr};
pub use a64::{A64State, Nzcv};

/// Bounded, pointer-free A64 context suitable for runtime diagnostics.
///
/// General registers remain plain architectural values because they are not
/// necessarily addresses. PC is typed explicitly in the guest address domain.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct A64RegisterContext {
    pub x: [u64; a64::GENERAL_REGISTER_COUNT],
    pub sp: u64,
    pub pc: GuestVirtualAddress,
    pub nzcv: Nzcv,
}

/// Bounded, pointer-free A32/T32 context suitable for runtime diagnostics.
///
/// R15 is represented by the stored instruction address rather than the
/// pipeline-adjusted operand view. CPSR retains the execution-state, IT, and
/// condition flags needed to understand either A32 or T32 execution.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct A32RegisterContext {
    pub r: [u32; a32::GENERAL_REGISTER_COUNT],
    pub pc: GuestVirtualAddress,
    pub cpsr: Cpsr,
}

/// Architectural register and flag context captured at an execution stop.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum RegisterContext {
    A64(A64RegisterContext),
    A32(A32RegisterContext),
}

impl RegisterContext {
    /// Returns the execution state represented by this snapshot.
    #[must_use]
    pub const fn execution_state(&self) -> ExecutionState {
        match self {
            Self::A64(_) => ExecutionState::A64,
            Self::A32(context) => context.cpsr.execution_state(),
        }
    }
}

impl fmt::Display for RegisterContext {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::A64(context) => {
                formatter.write_str("state=A64")?;
                for (index, value) in context.x.iter().enumerate() {
                    write!(formatter, " x{index}=0x{value:016x}")?;
                }
                write!(
                    formatter,
                    " sp=0x{:016x} pc={} nzcv=0x{:08x} flags=N{}Z{}C{}V{}",
                    context.sp,
                    context.pc,
                    context.nzcv.bits(),
                    u8::from(context.nzcv.negative()),
                    u8::from(context.nzcv.zero()),
                    u8::from(context.nzcv.carry()),
                    u8::from(context.nzcv.overflow()),
                )
            }
            Self::A32(context) => {
                write!(formatter, "state={}", context.cpsr.execution_state())?;
                for (index, value) in context.r.iter().enumerate() {
                    write!(formatter, " r{index}=0x{value:08x}")?;
                }
                write!(
                    formatter,
                    " pc={} cpsr=0x{:08x} flags=N{}Z{}C{}V{}Q{} T{} IT=0x{:02x}",
                    context.pc,
                    context.cpsr.bits(),
                    u8::from(context.cpsr.negative()),
                    u8::from(context.cpsr.zero()),
                    u8::from(context.cpsr.carry()),
                    u8::from(context.cpsr.overflow()),
                    u8::from(context.cpsr.saturation()),
                    u8::from(context.cpsr.execution_state() == ExecutionState::T32),
                    context.cpsr.it_state().bits(),
                )
            }
        }
    }
}

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

    /// Captures bounded diagnostic state without retaining a reference to the
    /// live thread or exposing its backend-oriented in-memory layout.
    #[must_use]
    pub fn register_context(&self) -> RegisterContext {
        match self {
            Self::A64(state) => RegisterContext::A64(state.register_context()),
            Self::A32(state) => RegisterContext::A32(state.register_context()),
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

    #[test]
    fn diagnostic_context_is_execution_state_specific_and_deterministic() {
        use crate::state::a32::A32GeneralRegister;

        let mut state = A32State::t32();
        state.write_r(A32GeneralRegister::new(0).unwrap(), 0x1234_5678);
        state.set_cpsr(Cpsr::from_bits(
            Cpsr::USER_MODE | Cpsr::T | Cpsr::N | Cpsr::C | Cpsr::Q,
        ));
        let context = ThreadCpuState::A32(Box::new(state)).register_context();

        assert_eq!(context.execution_state(), ExecutionState::T32);
        assert_eq!(context.to_string(), context.to_string());
        assert!(context.to_string().contains("state=T32 r0=0x12345678"));
        assert!(
            context
                .to_string()
                .contains("cpsr=0xa8000030 flags=N1Z0C1V0Q1 T1 IT=0x00")
        );
    }

    #[test]
    fn a64_diagnostic_context_contains_gprs_stack_pc_and_nzcv() {
        use crate::state::a64::{A64GeneralRegister, A64Register};

        let mut state = A64State::default();
        state.write_x(
            A64Register::General(A64GeneralRegister::new(30).unwrap()),
            0x1234_5678_9abc_def0,
        );
        state.write_x(A64Register::StackPointer, 0x7100_0080_0000);
        state.set_pc(0x7100_0000_1000);
        state.set_nzcv(Nzcv::from_bits(Nzcv::Z | Nzcv::V));
        let context = ThreadCpuState::A64(Box::new(state)).register_context();
        let printed = context.to_string();

        assert_eq!(context.execution_state(), ExecutionState::A64);
        assert!(printed.contains("x30=0x123456789abcdef0"));
        assert!(printed.contains("sp=0x0000710000800000"));
        assert!(printed.contains("pc=0x0000710000001000"));
        assert!(printed.contains("nzcv=0x50000000 flags=N0Z1C0V1"));
    }
}
