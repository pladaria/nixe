//! Portable reference execution lifecycle for a constructed process.

use std::collections::VecDeque;
use std::error::Error;
use std::fmt::{Display, Formatter};

use swiitx_cpu::decode::{DecodeResult, decode, disassemble};
use swiitx_cpu::error::InstructionFetchFault;
use swiitx_cpu::error::{ProfileDisabledInstruction, UnallocatedEncoding};
use swiitx_cpu::interpreter::{
    InterpreterContext, InterpreterError, InterpreterOutcome, execute_one_with_context,
};
use swiitx_cpu::ir::terminator::ExceptionKind;
use swiitx_cpu::location::{ExecutionState, InstructionEncoding, LocationDescriptor};
use swiitx_cpu::memory::{InstructionMemory, SyntheticMemory};
use swiitx_cpu::profile::ProcessCpuContext;
use swiitx_cpu::state::{RegisterContext, ThreadCpuState};
use swiitx_cpu::vcpu::VcpuExecutionState;
use swiitx_cpu::{coverage::CoverageId, memory::DataAccessFault};

use crate::{DiagnosticsPolicy, ReportDetail};

/// Maximum number of guest instructions retained by an enabled trace.
pub const MAX_INSTRUCTION_TRACE_ENTRIES: usize = 64;
/// Maximum UTF-8 byte length retained for one detailed disassembly.
pub const MAX_TRACE_DISASSEMBLY_BYTES: usize = 96;
/// Maximum number of UTF-8 bytes emitted by an instruction-trace export.
pub const MAX_INSTRUCTION_TRACE_EXPORT_BYTES: usize = 16 * 1024;

/// One pointer-free instruction observation in execution order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InstructionTraceEntry {
    pub sequence: u64,
    pub source: LocationDescriptor,
    pub encoding: InstructionEncoding,
    /// Present only when the project-wide report detail is `Detailed`.
    pub disassembly: Option<Box<str>>,
}

impl Display for InstructionTraceEntry {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "#{} source=[{}] encoding={}",
            self.sequence, self.source, self.encoding
        )?;
        if let Some(disassembly) = &self.disassembly {
            write!(formatter, " disassembly={disassembly}")?;
        }
        Ok(())
    }
}

/// Bounded snapshot of the most recently executed guest instructions.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InstructionTrace {
    enabled: bool,
    entries: Box<[InstructionTraceEntry]>,
    discarded: u64,
}

impl InstructionTrace {
    /// Returns whether trace capture was enabled for this process.
    #[must_use]
    pub const fn enabled(&self) -> bool {
        self.enabled
    }

    /// Returns retained entries from oldest to newest.
    #[must_use]
    pub const fn entries(&self) -> &[InstructionTraceEntry] {
        &self.entries
    }

    /// Returns the number of older observations evicted from the bounded trace.
    #[must_use]
    pub const fn discarded(&self) -> u64 {
        self.discarded
    }
}

impl Display for InstructionTrace {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        if !self.enabled {
            return formatter.write_str("disabled");
        }
        let mut output = format!(
            "retained={} discarded={}",
            self.entries.len(),
            self.discarded
        );
        for entry in &self.entries {
            let line = format!("\n{entry}");
            if output.len().saturating_add(line.len()) > MAX_INSTRUCTION_TRACE_EXPORT_BYTES {
                const MARKER: &str = "\n<trace-export-truncated>";
                if output.len().saturating_add(MARKER.len()) <= MAX_INSTRUCTION_TRACE_EXPORT_BYTES {
                    output.push_str(MARKER);
                }
                break;
            }
            output.push_str(&line);
        }
        formatter.write_str(&output)
    }
}

struct InstructionTraceRecorder {
    enabled: bool,
    detailed: bool,
    entries: VecDeque<InstructionTraceEntry>,
    next_sequence: u64,
    discarded: u64,
}

impl InstructionTraceRecorder {
    fn new(policy: DiagnosticsPolicy) -> Self {
        Self {
            enabled: policy.instruction_trace,
            detailed: policy.report_detail == ReportDetail::Detailed,
            entries: VecDeque::new(),
            next_sequence: 0,
            discarded: 0,
        }
    }

    fn record(
        &mut self,
        cpu: ProcessCpuContext,
        source: LocationDescriptor,
        encoding: InstructionEncoding,
    ) {
        if !self.enabled {
            return;
        }
        if self.entries.len() == MAX_INSTRUCTION_TRACE_ENTRIES {
            self.entries.pop_front();
            self.discarded = self.discarded.saturating_add(1);
        }
        let disassembly = self
            .detailed
            .then(|| instruction_description(cpu, source, encoding));
        self.entries.push_back(InstructionTraceEntry {
            sequence: self.next_sequence,
            source,
            encoding,
            disassembly,
        });
        self.next_sequence = self.next_sequence.saturating_add(1);
    }

    fn snapshot(&self) -> InstructionTrace {
        InstructionTrace {
            enabled: self.enabled,
            entries: self.entries.iter().cloned().collect(),
            discarded: self.discarded,
        }
    }
}

/// Host-side lifecycle state of one emulated process.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ProcessExecutionStatus {
    Ready,
    Running,
    Suspended,
    Exited,
    Faulted,
}

/// Reason a bounded reference-execution call returned to the runtime.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExecutionStop {
    UnsupportedSemantics {
        source: LocationDescriptor,
        encoding: InstructionEncoding,
        disassembly: Box<str>,
        coverage_id: CoverageId,
    },
    ProfileDisabled {
        error: ProfileDisabledInstruction,
    },
    UnallocatedEncoding {
        error: UnallocatedEncoding,
    },
    FetchFault {
        fault: InstructionFetchFault,
    },
    BudgetExhausted,
    Safepoint,
    PendingEvent {
        mask: u32,
    },
    Scheduled {
        source: LocationDescriptor,
    },
    ArchitecturalException {
        source: LocationDescriptor,
        kind: ExceptionKind,
        syndrome: Option<u64>,
    },
    SupervisorCall {
        source: LocationDescriptor,
        immediate: Option<u64>,
    },
    DataFault {
        source: LocationDescriptor,
        fault: DataAccessFault,
    },
}

impl Display for ExecutionStop {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSemantics {
                source,
                encoding,
                disassembly,
                coverage_id,
            } => write!(
                formatter,
                "unsupported-semantics source=[{source}] encoding={encoding} disassembly={disassembly} coverage={coverage_id}"
            ),
            Self::ProfileDisabled { error } => write!(formatter, "profile-disabled {error}"),
            Self::UnallocatedEncoding { error } => {
                write!(formatter, "unallocated-encoding {error}")
            }
            Self::FetchFault { fault } => write!(formatter, "fetch-fault {fault}"),
            Self::BudgetExhausted => formatter.write_str("budget-exhausted"),
            Self::Safepoint => formatter.write_str("safepoint"),
            Self::PendingEvent { mask } => write!(formatter, "pending-event mask=0x{mask:08x}"),
            Self::Scheduled { source } => write!(formatter, "scheduled source=[{source}]"),
            Self::ArchitecturalException {
                source,
                kind,
                syndrome,
            } => write!(
                formatter,
                "architectural-exception source=[{source}] kind={kind:?} syndrome={syndrome:?}"
            ),
            Self::SupervisorCall { source, immediate } => write!(
                formatter,
                "supervisor-call source=[{source}] immediate={immediate:?}"
            ),
            Self::DataFault { source, fault } => {
                write!(formatter, "data-fault source=[{source}] fault={fault:?}")
            }
        }
    }
}

/// Result of one bounded reference-execution slice.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionReport {
    pub instructions_executed: u64,
    pub stop: ExecutionStop,
    /// Pointer-free architectural state at the exact stop boundary.
    pub context: RegisterContext,
    /// Opt-in bounded history, ordered from oldest to newest.
    pub trace: InstructionTrace,
}

impl Display for ExecutionReport {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "instructions={} stop=[{}] registers=[{}] trace=[{}]",
            self.instructions_executed, self.stop, self.context, self.trace
        )
    }
}

/// Structured runtime failure which prevented an execution slice from ending normally.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProcessExecutionError {
    NotRunnable {
        status: ProcessExecutionStatus,
        context: Box<RegisterContext>,
    },
    Interpreter {
        instructions_executed: u64,
        error: InterpreterError,
        context: Box<RegisterContext>,
    },
}

impl Display for ProcessExecutionError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotRunnable { status, context } => {
                write!(
                    formatter,
                    "process is not runnable while {status:?}: registers=[{context}]"
                )
            }
            Self::Interpreter {
                instructions_executed,
                error,
                context,
            } => write!(
                formatter,
                "reference execution failed after {instructions_executed} instructions: {error} registers=[{context}]"
            ),
        }
    }
}

impl Error for ProcessExecutionError {}

/// Summary returned while deterministically consuming process-owned resources.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ProcessTeardownReport {
    pub previous_status: ProcessExecutionStatus,
    pub threads_released: usize,
    pub modules_released: usize,
    pub mappings_released: usize,
    pub physical_pages_released: usize,
    pub mounts_released: usize,
    pub handles_released: usize,
}

pub(crate) struct ProcessExecutionControl {
    status: ProcessExecutionStatus,
    vcpu: VcpuExecutionState<()>,
    trace: InstructionTraceRecorder,
}

impl ProcessExecutionControl {
    pub(crate) fn new(diagnostics: DiagnosticsPolicy) -> Self {
        Self {
            status: ProcessExecutionStatus::Ready,
            vcpu: VcpuExecutionState::new((), 0),
            trace: InstructionTraceRecorder::new(diagnostics),
        }
    }

    pub(crate) const fn status(&self) -> ProcessExecutionStatus {
        self.status
    }

    pub(crate) fn request_safepoint(&mut self) {
        self.vcpu.dispatch_mut().request_safepoint();
    }

    pub(crate) fn post_event(&self, mask: u32) {
        self.vcpu.post_interrupts(mask);
    }

    pub(crate) fn resume(&mut self) -> bool {
        if self.status != ProcessExecutionStatus::Suspended {
            return false;
        }
        self.status = ProcessExecutionStatus::Ready;
        true
    }

    pub(crate) fn terminate(&mut self) -> bool {
        if matches!(
            self.status,
            ProcessExecutionStatus::Exited | ProcessExecutionStatus::Faulted
        ) {
            return false;
        }
        self.status = ProcessExecutionStatus::Exited;
        true
    }
}

pub(crate) fn run_reference(
    control: &mut ProcessExecutionControl,
    cpu: ProcessCpuContext,
    memory: &SyntheticMemory,
    state: &mut ThreadCpuState,
    instruction_budget: u64,
) -> Result<ExecutionReport, ProcessExecutionError> {
    if control.status != ProcessExecutionStatus::Ready {
        return Err(ProcessExecutionError::NotRunnable {
            status: control.status,
            context: Box::new(state.register_context()),
        });
    }
    control.status = ProcessExecutionStatus::Running;
    control.vcpu.dispatch_mut().set_budget(instruction_budget);
    let mut executed = 0_u64;

    loop {
        if control.vcpu.dispatch().safepoint_requested() {
            control.vcpu.dispatch_mut().clear_safepoint();
            control.status = ProcessExecutionStatus::Ready;
            return Ok(ExecutionReport {
                instructions_executed: executed,
                stop: ExecutionStop::Safepoint,
                context: state.register_context(),
                trace: control.trace.snapshot(),
            });
        }
        let events = control.vcpu.take_pending_interrupts();
        if events != 0 {
            control.status = ProcessExecutionStatus::Ready;
            return Ok(ExecutionReport {
                instructions_executed: executed,
                stop: ExecutionStop::PendingEvent { mask: events },
                context: state.register_context(),
                trace: control.trace.snapshot(),
            });
        }
        if control.vcpu.dispatch().budget() == 0 {
            control.status = ProcessExecutionStatus::Ready;
            return Ok(ExecutionReport {
                instructions_executed: executed,
                stop: ExecutionStop::BudgetExhausted,
                context: state.register_context(),
                trace: control.trace.snapshot(),
            });
        }

        let encoding = match fetch_current(memory, cpu, state) {
            Ok(encoding) => encoding,
            Err(fault) => {
                control.status = ProcessExecutionStatus::Faulted;
                return Ok(ExecutionReport {
                    instructions_executed: executed,
                    stop: ExecutionStop::FetchFault { fault },
                    context: state.register_context(),
                    trace: control.trace.snapshot(),
                });
            }
        };
        let source = current_location(cpu, state);
        let context = InterpreterContext::new(cpu).with_memory(memory);
        let outcome = match execute_one_with_context(context, state, encoding) {
            Ok(outcome) => outcome,
            Err(InterpreterError::UnsupportedInstruction {
                source,
                encoding,
                disassembly,
                coverage_id,
            }) => {
                control.status = ProcessExecutionStatus::Faulted;
                return Ok(ExecutionReport {
                    instructions_executed: executed,
                    stop: ExecutionStop::UnsupportedSemantics {
                        source,
                        encoding,
                        disassembly,
                        coverage_id,
                    },
                    context: state.register_context(),
                    trace: control.trace.snapshot(),
                });
            }
            Err(error) => {
                control.status = ProcessExecutionStatus::Faulted;
                return Err(ProcessExecutionError::Interpreter {
                    instructions_executed: executed,
                    error,
                    context: Box::new(state.register_context()),
                });
            }
        };
        control.trace.record(cpu, source, encoding);
        executed += 1;
        let remaining = control.vcpu.dispatch().budget() - 1;
        control.vcpu.dispatch_mut().set_budget(remaining);

        match outcome {
            InterpreterOutcome::Resume(_) => {}
            InterpreterOutcome::Exception {
                source,
                kind: ExceptionKind::SupervisorCall,
                syndrome,
            } => {
                control.status = ProcessExecutionStatus::Suspended;
                return Ok(ExecutionReport {
                    instructions_executed: executed,
                    stop: ExecutionStop::SupervisorCall {
                        source,
                        immediate: syndrome,
                    },
                    context: state.register_context(),
                    trace: control.trace.snapshot(),
                });
            }
            InterpreterOutcome::Exception {
                source,
                kind,
                syndrome,
            } => {
                control.status = ProcessExecutionStatus::Suspended;
                return Ok(ExecutionReport {
                    instructions_executed: executed,
                    stop: ExecutionStop::ArchitecturalException {
                        source,
                        kind,
                        syndrome,
                    },
                    context: state.register_context(),
                    trace: control.trace.snapshot(),
                });
            }
            InterpreterOutcome::Scheduled { source } => {
                control.status = ProcessExecutionStatus::Suspended;
                return Ok(ExecutionReport {
                    instructions_executed: executed,
                    stop: ExecutionStop::Scheduled { source },
                    context: state.register_context(),
                    trace: control.trace.snapshot(),
                });
            }
            InterpreterOutcome::DataAbort { source, fault } => {
                control.status = ProcessExecutionStatus::Suspended;
                return Ok(ExecutionReport {
                    instructions_executed: executed,
                    stop: ExecutionStop::DataFault { source, fault },
                    context: state.register_context(),
                    trace: control.trace.snapshot(),
                });
            }
            InterpreterOutcome::ProfileDisabled(error) => {
                control.status = ProcessExecutionStatus::Suspended;
                return Ok(ExecutionReport {
                    instructions_executed: executed,
                    stop: ExecutionStop::ProfileDisabled { error },
                    context: state.register_context(),
                    trace: control.trace.snapshot(),
                });
            }
            InterpreterOutcome::Unallocated(error) => {
                control.status = ProcessExecutionStatus::Suspended;
                return Ok(ExecutionReport {
                    instructions_executed: executed,
                    stop: ExecutionStop::UnallocatedEncoding { error },
                    context: state.register_context(),
                    trace: control.trace.snapshot(),
                });
            }
        }
    }
}

fn current_location(cpu: ProcessCpuContext, state: &ThreadCpuState) -> LocationDescriptor {
    let (pc, execution_state) = match state {
        ThreadCpuState::A64(state) => (state.pc(), ExecutionState::A64),
        ThreadCpuState::A32(state) => (
            u64::from(state.instruction_address()),
            state.execution_state(),
        ),
    };
    LocationDescriptor::new(
        swiitx_cpu::address::GuestVirtualAddress::new(pc),
        execution_state,
        cpu.profile().id(),
    )
}

fn instruction_description(
    cpu: ProcessCpuContext,
    source: LocationDescriptor,
    encoding: InstructionEncoding,
) -> Box<str> {
    let description = match decode(&cpu.profile(), source, encoding) {
        DecodeResult::Decoded(decoded) | DecodeResult::RecognizedUnimplemented(decoded) => {
            disassemble(&decoded.instruction).to_string()
        }
        DecodeResult::Unallocated { reason, .. } => format!("<unallocated: {reason}>"),
        DecodeResult::Reserved { name, reason, .. } => {
            format!("<{name}: reserved: {reason}>")
        }
        DecodeResult::ProfileDisabled {
            name, rejection, ..
        } => format!("<{name}: profile-disabled: {rejection}>"),
    };
    truncate_utf8(description, MAX_TRACE_DISASSEMBLY_BYTES).into()
}

fn truncate_utf8(mut value: String, maximum_bytes: usize) -> String {
    if value.len() <= maximum_bytes {
        return value;
    }
    let mut boundary = maximum_bytes;
    while !value.is_char_boundary(boundary) {
        boundary -= 1;
    }
    value.truncate(boundary);
    value
}

fn fetch_current(
    memory: &impl InstructionMemory,
    cpu: ProcessCpuContext,
    state: &ThreadCpuState,
) -> Result<InstructionEncoding, InstructionFetchFault> {
    let (pc, execution_state) = match state {
        ThreadCpuState::A64(state) => (state.pc(), ExecutionState::A64),
        ThreadCpuState::A32(state) => (
            u64::from(state.instruction_address()),
            state.execution_state(),
        ),
    };
    let address = swiitx_cpu::address::GuestVirtualAddress::new(pc);
    let address_space = cpu.address_space_id();
    match execution_state {
        ExecutionState::A64 | ExecutionState::A32 => memory
            .fetch32(address_space, address)
            .map(|fetched| InstructionEncoding::from_u32(fetched.bits)),
        ExecutionState::T32 => {
            let first = memory.fetch16(address_space, address)?;
            if execution_state.instruction_size(first.bits)
                == swiitx_cpu::location::InstructionSize::Bits16
            {
                Ok(InstructionEncoding::from_u16(first.bits))
            } else {
                memory
                    .fetch_t32_32(address_space, address)
                    .map(|fetched| InstructionEncoding::from_u32(fetched.bits))
            }
        }
    }
}
