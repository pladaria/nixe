//! Reference instruction interpretation.
//!
//! The interpreter consumes decoded architectural instructions directly. It
//! deliberately does not execute frontend IR, making it useful as an
//! independent oracle for differential tests.

mod a64;

use core::{cell::RefCell, fmt};

use nixe_cpu::execution::{ArchitecturalTimer, CpuExit, TimerSnapshot, VcpuEventState};
use nixe_cpu::{
    coverage::CoverageId,
    decode::{self, DecodeResult, DecodedOpcode},
    location::{DecodedInstruction, InstructionEncoding, LocationDescriptor},
    memory::{CpuMemory, SyntheticMemory},
    platform::TargetPlatform,
    profile::ProcessCpuContext,
    state::a64::A64State,
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
    direct_memory: Option<&'a RefCell<nixe_cpu_direct_memory::DirectScalarFrontend>>,
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
            direct_memory: None,
        }
    }

    pub(crate) const fn with_direct_memory(
        mut self,
        direct_memory: Option<&'a RefCell<nixe_cpu_direct_memory::DirectScalarFrontend>>,
    ) -> Self {
        self.direct_memory = direct_memory;
        self
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

    #[must_use]
    pub(crate) const fn direct_memory(
        self,
    ) -> Option<&'a RefCell<nixe_cpu_direct_memory::DirectScalarFrontend>> {
        self.direct_memory
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
    DirectMemory {
        source: LocationDescriptor,
        detail: Box<str>,
    },
}

impl fmt::Display for InterpreterError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
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
            Self::DirectMemory { source, detail } => {
                write!(
                    formatter,
                    "direct interpreter memory failed at {source}: {detail}"
                )
            }
        }
    }
}

impl std::error::Error for InterpreterError {}

/// Executes one already-fetched instruction as a reference-engine step.
pub fn execute_one(
    platform: &TargetPlatform,
    state: &mut A64State,
    encoding: u32,
) -> Result<InstructionStep, InterpreterError> {
    let memory = SyntheticMemory::new();
    let monitor = RefCell::new(Default::default());
    let timer = ZeroTimer;
    let events = VcpuEventState::default();
    let context = InterpreterContext::new(
        ProcessCpuContext::new(*platform, AddressSpaceId::new(0)),
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
    state: &mut A64State,
    encoding: u32,
) -> Result<InstructionStep, InterpreterError> {
    let process = context.process();
    let source = LocationDescriptor::new(
        nixe_memory::GuestVirtualAddress::new(state.pc()),
        process.profile_id(),
    );
    let encoding = InstructionEncoding::from_u32(encoding);
    match decode::decode(process.decoder(), source, encoding) {
        DecodeResult::Decoded(decoded) => execute_decoded(context, state, &decoded),
        DecodeResult::RecognizedUnimplemented(decoded) => Err(unsupported(&decoded)),
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
    state: &mut A64State,
    decoded: &DecodedInstruction<DecodedOpcode>,
) -> Result<InstructionStep, InterpreterError> {
    a64::execute(context, state, decoded)
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
