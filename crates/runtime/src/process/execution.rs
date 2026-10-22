//! Runtime ownership of native CPU processes and per-core register contexts.
use crate::{ExceptionTerminationScope, GuestBreakPayload, GuestCpuState, VirtualClock};
use nixe_cpu::{
    execution::{
        ArchitecturalTimer, ControlRequest, CpuControl, CpuFault, CpuFaultKind, CpuProcessId,
        TimerSnapshot, VcpuEventState,
    },
    location::LocationDescriptor,
    profile::ProcessCpuContext,
    state::RegisterContext,
};
use nixe_cpu_jit::{JitProcess, JitRunRequest, JitThread};
use nixe_memory::{AddressSpace, GuestVirtualAddress};
use std::{
    collections::BTreeMap,
    error::Error,
    fmt::{Display, Formatter},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

static NEXT_CPU_PROCESS_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
pub(super) fn allocate_cpu_process_id() -> Option<CpuProcessId> {
    NEXT_CPU_PROCESS_ID
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
        .ok()
        .map(CpuProcessId::new)
}
pub use nixe_cpu::execution::{CpuExit as ExecutionStop, ExecutionReport};
#[derive(Clone, Debug, Default)]
pub enum CpuBackendConfig {
    /// The standalone interpreter is excluded from the machine-memory runtime.
    Interpreter,
    #[default]
    Jit,
}
pub(crate) type CpuThread = Box<JitThread>;
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ProcessExitCause {
    HostRequested,
    ProcessRequested,
    LastThreadExited,
    LoaderReturned,
    GuestBreak {
        reason: u64,
        info: u64,
        size: u64,
        payload: Option<GuestBreakPayload>,
    },
}

/// One A64 frame-pointer record retained for post-teardown diagnostics.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GuestStackFrame {
    /// Address of the frame record in the guest stack.
    pub frame_pointer: u64,
    /// Saved link register belonging to the caller.
    pub return_address: u64,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ProcessExit {
    pub cause: ProcessExitCause,
    pub exit_code: u64,
    pub source: Option<LocationDescriptor>,
    pub thread_id: u64,
    /// Architectural state captured at the terminating instruction, when the
    /// runtime owns a resident CPU state for the exiting thread.
    pub context: Option<Box<RegisterContext>>,
    /// Bounded A64 frame-pointer walk captured before guest memory is released.
    pub frames: Box<[GuestStackFrame]>,
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
    Cpu {
        fault: CpuFault,
    },
    ConcurrentProcessStop {
        lifecycle: nixe_scheduler::ProcessLifecycle,
        context: Box<RegisterContext>,
    },
    BackendUnavailable {
        process: CpuProcessId,
    },
}

impl Display for ProcessExecutionError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownThread(thread) => {
                write!(formatter, "guest thread {thread} does not exist")
            }
            Self::Cpu { fault } => write!(formatter, "CPU backend failed: {fault}"),
            Self::ConcurrentProcessStop { lifecycle, context } => write!(
                formatter,
                "another vCPU stopped the process while this slice was in flight: lifecycle={lifecycle:?}, registers=[{context}]"
            ),
            Self::BackendUnavailable { process } => write!(
                formatter,
                "CPU thread for process {} is unavailable",
                process.get()
            ),
        }
    }
}

impl Error for ProcessExecutionError {}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ProcessTeardownReport {
    pub previous_lifecycle: nixe_scheduler::ProcessLifecycle,
    pub exit: Option<ProcessExit>,
    pub threads_released: usize,
    pub modules_released: usize,
    pub mappings_released: usize,
    pub physical_pages_released: usize,
    pub mounts_released: usize,
    pub handles_released: usize,
    pub address_waiters_released: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProcessTeardownFailure {
    pub report: Box<ProcessTeardownReport>,
    pub fault: Box<CpuFault>,
}

impl Display for ProcessTeardownFailure {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "CPU backend failed to shut down during teardown: {}",
            self.fault
        )
    }
}

impl Error for ProcessTeardownFailure {}

pub(crate) struct ProcessExecutionControl {
    process: Arc<JitProcess>,
    process_id: CpuProcessId,
    controls: BTreeMap<nixe_scheduler::VirtualCpuId, CpuControl>,
    pub(super) residents: BTreeMap<nixe_scheduler::VirtualCpuId, nixe_scheduler::GuestThreadId>,
    virtual_clock: VirtualClock,
    architectural_timer_frequency: u64,
    shutdown_complete: bool,
    stopping: bool,
    pending_safepoint: AtomicBool,
}
pub(crate) struct ProcessExecutionConfiguration {
    pub(crate) virtual_clock: VirtualClock,
    pub(crate) timer_frequency: u64,
    pub(crate) cpu: ProcessCpuContext,
}
impl ProcessExecutionControl {
    pub(crate) fn new(
        configuration: ProcessExecutionConfiguration,
        space: Arc<AddressSpace>,
        process_id: CpuProcessId,
        selection: &CpuBackendConfig,
    ) -> Result<Self, CpuFault> {
        if matches!(selection, CpuBackendConfig::Interpreter) {
            return Err(jit_boundary_fault(
                "the legacy interpreter is excluded from this runtime",
            ));
        }
        let process = JitProcess::new(configuration.cpu, space, configuration.timer_frequency)
            .map_err(jit_boundary_fault)?;
        Ok(Self {
            process,
            process_id,
            controls: BTreeMap::new(),
            residents: BTreeMap::new(),
            virtual_clock: configuration.virtual_clock,
            architectural_timer_frequency: configuration.timer_frequency,
            shutdown_complete: false,
            stopping: false,
            pending_safepoint: AtomicBool::new(false),
        })
    }
    pub(crate) const fn backend_name(&self) -> &'static str {
        "jit"
    }
    pub(crate) const fn process_id(&self) -> CpuProcessId {
        self.process_id
    }
    pub(crate) fn request_vcpu_safepoint(&self, vcpu: nixe_scheduler::VirtualCpuId) {
        if let Some(control) = self.controls.get(&vcpu) {
            control.request(ControlRequest::Preempt);
        }
    }
    pub(crate) fn request_safepoint(&self) {
        if self.controls.is_empty() {
            self.pending_safepoint.store(true, Ordering::Release);
        }
        for control in self.controls.values() {
            control.request(ControlRequest::Preempt);
        }
    }
    pub(crate) fn request_stop(&mut self) -> Result<(), CpuFault> {
        self.process.request_stop();
        self.stopping = true;
        Ok(())
    }
    pub(crate) fn shutdown(&mut self) -> Result<(), CpuFault> {
        if self.shutdown_complete {
            return Ok(());
        }
        self.request_stop()?;
        if !self.process.try_shutdown().map_err(jit_boundary_fault)? {
            return Err(jit_boundary_fault("native execution has not stopped"));
        }
        self.shutdown_complete = true;
        Ok(())
    }
    pub(crate) fn create_worker_cpu_thread(
        &mut self,
        vcpu: nixe_scheduler::VirtualCpuId,
    ) -> Result<CpuThread, CpuFault> {
        if self.stopping {
            return Err(CpuFault {
                backend: "jit",
                kind: CpuFaultKind::Unavailable,
                progress: 0,
                message: "CPU process is stopping".into(),
                context: None,
            });
        }
        let thread = Box::new(self.process.create_thread().map_err(jit_boundary_fault)?);
        let control = thread.control();
        if self.pending_safepoint.swap(false, Ordering::AcqRel) {
            control.request(ControlRequest::Preempt);
        }
        self.controls.insert(vcpu, control);
        Ok(thread)
    }
    pub(crate) fn complete_cpu_thread_retirement(&mut self) {
        self.controls.clear();
    }
    pub(crate) fn execution_environment(&self) -> (VirtualClock, u64) {
        (
            self.virtual_clock.clone(),
            self.architectural_timer_frequency,
        )
    }
}
impl Drop for ProcessExecutionControl {
    fn drop(&mut self) {
        if !self.shutdown_complete {
            let _ = self.shutdown();
        }
    }
}
fn jit_boundary_fault(message: impl ToString) -> CpuFault {
    CpuFault {
        backend: "jit",
        kind: CpuFaultKind::Internal,
        progress: 0,
        message: message.to_string().into_boxed_str(),
        context: None,
    }
}
struct RuntimeTimer<'a> {
    clock: &'a VirtualClock,
    frequency: u64,
}
pub(crate) struct VcpuExecutionState {
    pub(crate) thread: GuestCpuState,
    pub(crate) virtual_clock: VirtualClock,
    pub(crate) architectural_timer_frequency: u64,
    pub(crate) instruction_budget: u64,
    pub(crate) capture_context: bool,
    pub(crate) events: VcpuEventState,
}
impl VcpuExecutionState {
    pub(crate) fn run(
        &mut self,
        worker: &mut nixe_cpu_direct_memory::NativeWorker,
        thread: &mut CpuThread,
    ) -> Result<ExecutionReport, ProcessExecutionError> {
        let timer = RuntimeTimer {
            clock: &self.virtual_clock,
            frequency: self.architectural_timer_frequency,
        };
        thread
            .run_slice(
                worker,
                JitRunRequest {
                    state: &mut self.thread,
                    instruction_budget: self.instruction_budget,
                    timer: &timer,
                    events: &self.events,
                    capture_context: self.capture_context,
                },
            )
            .map_err(|fault| ProcessExecutionError::Cpu { fault })
    }
}
impl ArchitecturalTimer for RuntimeTimer<'_> {
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
pub(crate) fn current_location(
    cpu: ProcessCpuContext,
    state: &GuestCpuState,
) -> LocationDescriptor {
    LocationDescriptor::new(GuestVirtualAddress::new(state.pc()), cpu.profile_id())
}
