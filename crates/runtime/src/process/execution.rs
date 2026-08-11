//! Process-local execution domain and per-vCPU executor ownership.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{Display, Formatter};

use nixe_cpu::location::LocationDescriptor;
use nixe_cpu::memory::ExecutionMemory;
use nixe_cpu::profile::ProcessCpuContext;
use nixe_cpu::state::{RegisterContext, ThreadCpuState};
use nixe_cpu_engine::{
    DomainRequest, EngineDomain, EngineDomainId, EngineExecutor, EngineExecutorId, EngineFault,
    EngineProvider, EngineTimer, ExecutorRequest, RunRequest, SafepointReason, TimerSnapshot,
    TracePolicy,
};
use nixe_memory::GuestVirtualAddress;

use crate::{DiagnosticsPolicy, ExceptionTerminationScope, ReportDetail, VirtualClock};

pub use nixe_cpu_engine::{
    EngineExit as ExecutionStop, ExecutionReport, InstructionTrace, InstructionTraceEntry,
    MAX_INSTRUCTION_TRACE_ENTRIES, MAX_INSTRUCTION_TRACE_EXPORT_BYTES, MAX_TRACE_DISASSEMBLY_BYTES,
};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ProcessExecutionStatus {
    Ready,
    Running,
    Suspended,
    Exited,
    Faulted,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ProcessExitCause {
    HostRequested,
    ProcessRequested,
    LastThreadExited,
    LoaderReturned,
    GuestBreak { reason: u64, info: u64, size: u64 },
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ProcessExit {
    pub cause: ProcessExitCause,
    pub exit_code: u64,
    pub source: Option<LocationDescriptor>,
    pub thread_id: u64,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ThreadExit {
    pub requested_scope: ExceptionTerminationScope,
    pub exit_code: u64,
    pub source: Option<LocationDescriptor>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProcessExecutionError {
    UnknownThread(nixe_scheduler::GuestThreadId),
    NotRunnable {
        status: ProcessExecutionStatus,
        context: Box<RegisterContext>,
    },
    Engine {
        fault: EngineFault,
    },
}

impl Display for ProcessExecutionError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownThread(thread) => {
                write!(formatter, "guest thread {thread} does not exist")
            }
            Self::NotRunnable { status, context } => write!(
                formatter,
                "process is not runnable while {status:?}: registers=[{context}]"
            ),
            Self::Engine { fault } => write!(formatter, "CPU engine failed: {fault}"),
        }
    }
}
impl Error for ProcessExecutionError {}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ProcessTeardownReport {
    pub previous_status: ProcessExecutionStatus,
    pub exit: Option<ProcessExit>,
    pub threads_released: usize,
    pub modules_released: usize,
    pub mappings_released: usize,
    pub physical_pages_released: usize,
    pub mounts_released: usize,
    pub handles_released: usize,
    pub address_waiters_released: usize,
}

pub(crate) struct ProcessExecutionControl {
    domain: Box<dyn EngineDomain>,
    executors: BTreeMap<nixe_scheduler::VirtualCpuId, Box<dyn EngineExecutor>>,
    trace_policy: TracePolicy,
    virtual_clock: VirtualClock,
    architectural_timer_frequency: u64,
}

impl ProcessExecutionControl {
    pub(crate) fn with_provider(
        diagnostics: DiagnosticsPolicy,
        virtual_clock: VirtualClock,
        timer_frequency: u64,
        cpu: ProcessCpuContext,
        provider: &dyn EngineProvider,
    ) -> Result<Self, EngineFault> {
        let trace = TracePolicy {
            enabled: diagnostics.instruction_trace,
            detailed: diagnostics.report_detail == ReportDetail::Detailed,
        };
        let mut domain = provider.create_domain(DomainRequest {
            domain: EngineDomainId::new(1),
            cpu,
        })?;
        let executor = domain.create_executor(ExecutorRequest {
            executor: EngineExecutorId::new(1),
            trace,
        })?;
        let mut executors = BTreeMap::new();
        executors.insert(nixe_scheduler::VirtualCpuId::new(0), executor);
        Ok(Self {
            domain,
            executors,
            trace_policy: trace,
            virtual_clock,
            architectural_timer_frequency: timer_frequency,
        })
    }

    pub(crate) fn engine_descriptor(&self) -> nixe_cpu_engine::EngineDescriptor {
        self.domain.descriptor()
    }
    pub(crate) fn request_safepoint(&mut self) {
        for executor in self.executors.values_mut() {
            executor.request_safepoint(SafepointReason::Requested);
        }
    }
    pub(crate) fn post_event(&self, mask: u32) {
        for executor in self.executors.values() {
            executor.post_event(mask);
        }
    }
    fn executor_id(vcpu: nixe_scheduler::VirtualCpuId) -> EngineExecutorId {
        EngineExecutorId::new(u64::from(vcpu.get()) + 1)
    }
    fn executor_for(
        &mut self,
        vcpu: nixe_scheduler::VirtualCpuId,
    ) -> Result<&mut Box<dyn EngineExecutor>, EngineFault> {
        if !self.executors.contains_key(&vcpu) {
            let executor = self.domain.create_executor(ExecutorRequest {
                executor: Self::executor_id(vcpu),
                trace: self.trace_policy,
            })?;
            self.executors.insert(vcpu, executor);
        }
        Ok(self
            .executors
            .get_mut(&vcpu)
            .expect("executor was installed"))
    }
    pub(crate) fn clear_local_exclusive_reservation(&mut self, vcpu: nixe_scheduler::VirtualCpuId) {
        if let Some(executor) = self.executors.get_mut(&vcpu) {
            executor.clear_local_exclusive_reservation();
        }
    }
    pub(crate) fn switch_provider(
        &mut self,
        cpu: ProcessCpuContext,
        provider: &dyn EngineProvider,
    ) -> Result<nixe_cpu_engine::StateCommitBarrier, nixe_cpu_engine::HandoffFailure> {
        let mut replacement = provider
            .create_domain(DomainRequest {
                domain: EngineDomainId::new(self.domain.domain_id().get().saturating_add(1)),
                cpu,
            })
            .map_err(|fault| nixe_cpu_engine::HandoffFailure {
                stage: nixe_cpu_engine::HandoffFailureStage::Import,
                fault,
            })?;
        let mut replacement_executors = BTreeMap::new();
        for vcpu in self.executors.keys().copied() {
            let executor = replacement
                .create_executor(ExecutorRequest {
                    executor: Self::executor_id(vcpu),
                    trace: self.trace_policy,
                })
                .map_err(|fault| nixe_cpu_engine::HandoffFailure {
                    stage: nixe_cpu_engine::HandoffFailureStage::Import,
                    fault,
                })?;
            replacement_executors.insert(vcpu, executor);
        }
        let memory = nixe_cpu_engine::MemorySynchronizationRecord {
            address_space: cpu.address_space_id(),
            invalidation_generation: 0,
            dirty_generation: 0,
        };
        let (replacement, barrier) =
            nixe_cpu_engine::prepare_handoff(self.domain.as_mut(), replacement, memory)?;
        self.domain = replacement;
        self.executors = replacement_executors;
        Ok(barrier)
    }
}

struct RuntimeTimer<'a> {
    clock: &'a VirtualClock,
    frequency: u64,
}
impl EngineTimer for RuntimeTimer<'_> {
    fn snapshot(&self) -> TimerSnapshot {
        let ticks = self
            .clock
            .elapsed()
            .as_nanos()
            .saturating_mul(u128::from(self.frequency))
            / 1_000_000_000;
        TimerSnapshot {
            counter: u64::try_from(ticks).unwrap_or(u64::MAX),
            frequency: self.frequency,
        }
    }
}

pub(crate) fn run_engine(
    control: &mut ProcessExecutionControl,
    vcpu: nixe_scheduler::VirtualCpuId,
    cpu: ProcessCpuContext,
    memory: &ExecutionMemory,
    state: &mut ThreadCpuState,
    instruction_budget: u64,
    loader_return: Option<GuestVirtualAddress>,
) -> Result<ExecutionReport, ProcessExecutionError> {
    let virtual_clock = control.virtual_clock.clone();
    let frequency = control.architectural_timer_frequency;
    let timer = RuntimeTimer {
        clock: &virtual_clock,
        frequency,
    };
    let executor = control
        .executor_for(vcpu)
        .map_err(|fault| ProcessExecutionError::Engine { fault })?;
    let result = executor.run_slice(RunRequest {
        cpu,
        memory,
        state,
        instruction_budget,
        loader_return,
        timer: &timer,
    });
    match result {
        Ok(report) => {
            if report.state_commit != nixe_cpu_engine::StateCommitStatus::Canonical {
                let fault = EngineFault {
                    engine: executor.descriptor().id,
                    kind: nixe_cpu_engine::EngineFaultKind::StateExport,
                    instructions_executed: report.instructions_executed,
                    message: "engine returned before committing canonical thread state".into(),
                    context: Box::new(report.context),
                };
                return Err(ProcessExecutionError::Engine { fault });
            }
            Ok(report)
        }
        Err(fault) => Err(ProcessExecutionError::Engine { fault }),
    }
}

pub(crate) fn current_location(
    cpu: ProcessCpuContext,
    state: &ThreadCpuState,
) -> LocationDescriptor {
    nixe_cpu::location::current_location(cpu, state)
}
