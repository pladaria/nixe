//! Reference instruction interpretation.
//!
//! The interpreter consumes decoded architectural instructions directly. It
//! deliberately does not execute frontend IR, making it useful as an
//! independent oracle for differential tests.

mod a32;
mod a64;
mod aarch32;
mod t32;

use core::{cell::RefCell, fmt};

use nixe_cpu::execution::{ArchitecturalTimer, CpuExit, TimerSnapshot, VcpuEventState};
use nixe_cpu::{
    coverage::CoverageId,
    decode::{self, DecodeResult, DecodedOpcode},
    location::{
        DecodedInstruction, ExecutionState, InstructionEncoding, LocationDescriptor,
        current_location,
    },
    memory::{CpuMemory, SyntheticMemory},
    profile::{GuestCpuProfile, ProcessCpuContext},
    state::ThreadCpuState,
};
use nixe_memory::AddressSpaceId;

/// Result of one decoded instruction in the direct interpreter loop.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum InstructionStep {
    Continue,
    Exit(CpuExit),
}

impl InstructionStep {
    pub(crate) const fn architectural_exception(
        source: LocationDescriptor,
        kind: nixe_cpu::exception::ExceptionKind,
        syndrome: Option<u64>,
    ) -> Self {
        Self::Exit(CpuExit::ArchitecturalException {
            source,
            kind,
            syndrome,
        })
    }

    pub(crate) const fn supervisor_call(source: LocationDescriptor, immediate: u32) -> Self {
        Self::Exit(CpuExit::SupervisorCall { source, immediate })
    }

    pub(crate) const fn data_fault(
        source: LocationDescriptor,
        fault: nixe_cpu::memory::DataAccessFault,
    ) -> Self {
        Self::Exit(CpuExit::DataFault { source, fault })
    }

    pub(crate) const fn scheduled(
        source: LocationDescriptor,
        request: nixe_cpu::execution::SchedulerRequest,
    ) -> Self {
        Self::Exit(CpuExit::Scheduled { source, request })
    }
}

/// Concrete services available to every production interpreter instruction.
#[derive(Clone, Copy)]
pub struct InterpreterContext<'a> {
    process: ProcessCpuContext,
    memory: &'a dyn CpuMemory,
    exclusive_monitor: &'a RefCell<nixe_cpu::exclusive::ExclusiveMonitorState>,
    architectural_timer: &'a dyn ArchitecturalTimer,
    events: &'a VcpuEventState,
}

impl<'a> InterpreterContext<'a> {
    #[must_use]
    pub const fn new(
        process: ProcessCpuContext,
        memory: &'a dyn CpuMemory,
        exclusive_monitor: &'a RefCell<nixe_cpu::exclusive::ExclusiveMonitorState>,
        architectural_timer: &'a dyn ArchitecturalTimer,
        events: &'a VcpuEventState,
    ) -> Self {
        Self {
            process,
            memory,
            exclusive_monitor,
            architectural_timer,
            events,
        }
    }

    #[must_use]
    pub fn architectural_timer(self) -> TimerSnapshot {
        self.architectural_timer.snapshot()
    }

    #[must_use]
    pub const fn vcpu_events(self) -> &'a VcpuEventState {
        self.events
    }

    #[must_use]
    pub const fn exclusive_monitor(
        self,
    ) -> &'a RefCell<nixe_cpu::exclusive::ExclusiveMonitorState> {
        self.exclusive_monitor
    }

    #[must_use]
    pub const fn process(self) -> ProcessCpuContext {
        self.process
    }

    #[must_use]
    pub const fn memory(self) -> &'a dyn CpuMemory {
        self.memory
    }
}

struct ZeroTimer;

impl ArchitecturalTimer for ZeroTimer {
    fn snapshot(&self) -> TimerSnapshot {
        TimerSnapshot {
            counter: 0,
            frequency: 0,
        }
    }
}

/// Deterministic interpreter failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum InterpreterError {
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
    InvalidInstructionStream {
        source: LocationDescriptor,
        encoding: InstructionEncoding,
        reason: Box<str>,
    },
}

impl fmt::Display for InterpreterError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
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
            Self::InvalidInstructionStream {
                source,
                encoding,
                reason,
            } => write!(
                formatter,
                "invalid instruction stream: {source} encoding={encoding} reason={reason}"
            ),
        }
    }
}

impl std::error::Error for InterpreterError {}

/// Executes one already-fetched instruction as a reference-engine step.
pub fn execute_one(
    profile: &GuestCpuProfile,
    state: &mut ThreadCpuState,
    encoding: InstructionEncoding,
) -> Result<InstructionStep, InterpreterError> {
    let memory = SyntheticMemory::new();
    let monitor = RefCell::new(Default::default());
    let timer = ZeroTimer;
    let events = VcpuEventState::default();
    let context = InterpreterContext::new(
        ProcessCpuContext::new(*profile, AddressSpaceId::new(0)),
        &memory,
        &monitor,
        &timer,
        &events,
    );
    execute_one_with_context(context, state, encoding)
}

/// Executes one instruction with process address-space and memory services.
pub fn execute_one_with_context(
    context: InterpreterContext<'_>,
    state: &mut ThreadCpuState,
    encoding: InstructionEncoding,
) -> Result<InstructionStep, InterpreterError> {
    let process = context.process();
    let source = current_location(process, state);
    match decode::decode(process.decoder(), source, encoding) {
        DecodeResult::Decoded(decoded) | DecodeResult::RecognizedUnimplemented(decoded) => {
            execute_decoded(context, state, &decoded)
        }
        DecodeResult::Unallocated {
            instruction,
            reason,
        } => Err(InterpreterError::InvalidInstructionStream {
            source: instruction.location,
            encoding,
            reason: reason.into(),
        }),
        DecodeResult::Reserved {
            instruction,
            name,
            reason,
        } => Err(InterpreterError::InvalidInstructionStream {
            source: instruction.location,
            encoding,
            reason: format!("{name}: reserved: {reason}").into(),
        }),
    }
}

pub(crate) fn execute_decoded(
    context: InterpreterContext<'_>,
    state: &mut ThreadCpuState,
    decoded: &DecodedInstruction<DecodedOpcode>,
) -> Result<InstructionStep, InterpreterError> {
    match (state, decoded.location.execution_state) {
        (ThreadCpuState::A64(state), ExecutionState::A64) => a64::execute(context, state, decoded),
        (ThreadCpuState::A32(state), ExecutionState::A32) => a32::execute(context, state, decoded),
        (ThreadCpuState::A32(state), ExecutionState::T32) => t32::execute(context, state, decoded),
        _ => unreachable!("current location must match canonical architectural state"),
    }
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
