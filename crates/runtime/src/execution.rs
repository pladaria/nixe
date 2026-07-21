//! Portable reference execution lifecycle for a constructed process.

use std::error::Error;
use std::fmt::{Display, Formatter};

use swiitx_cpu::error::InstructionFetchFault;
use swiitx_cpu::interpreter::{
    InterpreterContext, InterpreterError, InterpreterOutcome, execute_one_with_context,
};
use swiitx_cpu::ir::terminator::ExceptionKind;
use swiitx_cpu::location::{ExecutionState, InstructionEncoding, LocationDescriptor};
use swiitx_cpu::memory::{InstructionMemory, SyntheticMemory};
use swiitx_cpu::profile::ProcessCpuContext;
use swiitx_cpu::state::ThreadCpuState;
use swiitx_cpu::vcpu::VcpuExecutionState;

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
    BudgetExhausted,
    Safepoint,
    PendingEvent {
        mask: u32,
    },
    Scheduled {
        source: LocationDescriptor,
    },
    Exception {
        source: LocationDescriptor,
        kind: ExceptionKind,
        syndrome: Option<u64>,
    },
    DataAbort {
        source: LocationDescriptor,
        fault: swiitx_cpu::memory::DataAccessFault,
    },
}

/// Result of one bounded reference-execution slice.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionReport {
    pub instructions_executed: u64,
    pub stop: ExecutionStop,
}

/// Structured runtime failure which prevented an execution slice from ending normally.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProcessExecutionError {
    NotRunnable {
        status: ProcessExecutionStatus,
    },
    InstructionFetch {
        instructions_executed: u64,
        fault: InstructionFetchFault,
    },
    Interpreter {
        instructions_executed: u64,
        error: InterpreterError,
    },
}

impl Display for ProcessExecutionError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotRunnable { status } => {
                write!(formatter, "process is not runnable while {status:?}")
            }
            Self::InstructionFetch {
                instructions_executed,
                fault,
            } => write!(
                formatter,
                "reference execution failed after {instructions_executed} instructions: {fault}"
            ),
            Self::Interpreter {
                instructions_executed,
                error,
            } => write!(
                formatter,
                "reference execution failed after {instructions_executed} instructions: {error}"
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
}

impl Default for ProcessExecutionControl {
    fn default() -> Self {
        Self {
            status: ProcessExecutionStatus::Ready,
            vcpu: VcpuExecutionState::new((), 0),
        }
    }
}

impl ProcessExecutionControl {
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
            });
        }
        let events = control.vcpu.take_pending_interrupts();
        if events != 0 {
            control.status = ProcessExecutionStatus::Ready;
            return Ok(ExecutionReport {
                instructions_executed: executed,
                stop: ExecutionStop::PendingEvent { mask: events },
            });
        }
        if control.vcpu.dispatch().budget() == 0 {
            control.status = ProcessExecutionStatus::Ready;
            return Ok(ExecutionReport {
                instructions_executed: executed,
                stop: ExecutionStop::BudgetExhausted,
            });
        }

        let encoding = fetch_current(memory, cpu, state).map_err(|fault| {
            control.status = ProcessExecutionStatus::Faulted;
            ProcessExecutionError::InstructionFetch {
                instructions_executed: executed,
                fault,
            }
        })?;
        let context = InterpreterContext::new(cpu).with_memory(memory);
        let outcome = execute_one_with_context(context, state, encoding).map_err(|error| {
            control.status = ProcessExecutionStatus::Faulted;
            ProcessExecutionError::Interpreter {
                instructions_executed: executed,
                error,
            }
        })?;
        executed += 1;
        let remaining = control.vcpu.dispatch().budget() - 1;
        control.vcpu.dispatch_mut().set_budget(remaining);

        match outcome {
            InterpreterOutcome::Resume(_) => {}
            InterpreterOutcome::Exception {
                source,
                kind,
                syndrome,
            } => {
                control.status = ProcessExecutionStatus::Suspended;
                return Ok(ExecutionReport {
                    instructions_executed: executed,
                    stop: ExecutionStop::Exception {
                        source,
                        kind,
                        syndrome,
                    },
                });
            }
            InterpreterOutcome::Scheduled { source } => {
                control.status = ProcessExecutionStatus::Suspended;
                return Ok(ExecutionReport {
                    instructions_executed: executed,
                    stop: ExecutionStop::Scheduled { source },
                });
            }
            InterpreterOutcome::DataAbort { source, fault } => {
                control.status = ProcessExecutionStatus::Suspended;
                return Ok(ExecutionReport {
                    instructions_executed: executed,
                    stop: ExecutionStop::DataAbort { source, fault },
                });
            }
        }
    }
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
