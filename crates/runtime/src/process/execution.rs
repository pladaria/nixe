//! Runtime-owned concrete CPU backend selection and per-vCPU state.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use nixe_cpu::execution::{
    ArchitecturalTimer, ControlRequest, CpuControl, CpuFault, CpuFaultKind, CpuProcessId,
    CpuThreadId, MemoryBinding, RunRequest, TimerSnapshot, VcpuEventState,
};
use nixe_cpu::location::LocationDescriptor;
use nixe_cpu::memory::ExecutionMemory;
use nixe_cpu::profile::ProcessCpuContext;
use nixe_cpu::state::{RegisterContext, ThreadCpuState};
use nixe_cpu_interpreter::{InterpreterProcess, InterpreterRunRequest, InterpreterThread};
use nixe_cpu_jit::{JitConfiguration, JitProcess, JitThread};
use nixe_memory::{GuestVirtualAddress, MemoryInvalidationSource};

use crate::{ExceptionTerminationScope, GuestBreakPayload, VirtualClock};

static NEXT_CPU_PROCESS_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

pub(super) fn allocate_cpu_process_id() -> Option<CpuProcessId> {
    NEXT_CPU_PROCESS_ID
        .fetch_update(
            std::sync::atomic::Ordering::Relaxed,
            std::sync::atomic::Ordering::Relaxed,
            |id| id.checked_add(1),
        )
        .ok()
        .map(CpuProcessId::new)
}

pub use nixe_cpu::execution::{CpuExit as ExecutionStop, ExecutionReport};

#[derive(Clone, Debug)]
pub enum CpuBackendConfig {
    Interpreter,
    Jit(JitConfiguration),
}

impl Default for CpuBackendConfig {
    fn default() -> Self {
        Self::Jit(JitConfiguration::default())
    }
}

enum CpuBackend {
    Interpreter(InterpreterProcess),
    Jit(Arc<JitProcess>),
}

impl CpuBackend {
    fn new(
        selection: &CpuBackendConfig,
        _id: CpuProcessId,
        cpu: ProcessCpuContext,
    ) -> Result<Self, CpuFault> {
        match selection {
            CpuBackendConfig::Interpreter => Ok(Self::Interpreter(InterpreterProcess::new(cpu))),
            CpuBackendConfig::Jit(configuration) => {
                JitProcess::with_configuration(cpu, configuration.clone())
                    .map(Arc::new)
                    .map(Self::Jit)
                    .map_err(|error| {
                        runtime_fault(cpu, CpuFaultKind::Unavailable, error.to_string())
                    })
            }
        }
    }

    fn bind_memory(&mut self, binding: MemoryBinding<'_>) -> Result<(), CpuFault> {
        match self {
            Self::Interpreter(process) => process.bind_memory(binding),
            Self::Jit(process) => process.bind_memory(binding).map_err(|error| CpuFault {
                backend: "jit",
                kind: CpuFaultKind::Unavailable,
                instructions_executed: 0,
                message: error.to_string().into_boxed_str(),
                context: Box::new(ThreadCpuState::default().register_context()),
            }),
        }
    }

    fn create_thread(&mut self, id: CpuThreadId) -> Result<CpuThread, CpuFault> {
        match self {
            Self::Interpreter(process) => process.create_thread(id).map(CpuThread::Interpreter),
            Self::Jit(process) => Ok(CpuThread::Jit {
                process: Arc::clone(process),
                thread: JitThread::new(),
            }),
        }
    }

    fn request_stop(&mut self) -> Result<(), CpuFault> {
        match self {
            Self::Interpreter(process) => process.request_stop(),
            Self::Jit(_) => Ok(()),
        }
    }

    fn shutdown(&mut self) -> Result<(), CpuFault> {
        match self {
            Self::Interpreter(process) => process.shutdown(),
            Self::Jit(process) => {
                process.shutdown();
                Ok(())
            }
        }
    }

    const fn name(&self) -> &'static str {
        match self {
            Self::Interpreter(_) => "interpreter",
            Self::Jit(_) => "jit",
        }
    }
}

pub(crate) enum CpuThread {
    Interpreter(InterpreterThread),
    Jit {
        process: Arc<JitProcess>,
        thread: JitThread,
    },
}

impl CpuThread {
    pub(crate) fn run_slice(
        &mut self,
        request: RunRequest<'_>,
    ) -> Result<ExecutionReport, CpuFault> {
        match self {
            Self::Interpreter(thread) => {
                let RunRequest {
                    memory,
                    memory_lease,
                    state,
                    instruction_budget,
                    loader_return,
                    timer,
                    events,
                    ..
                } = request;
                thread.run_slice(InterpreterRunRequest {
                    memory,
                    memory_lease,
                    state,
                    instruction_budget,
                    loader_return,
                    timer,
                    events,
                })
            }
            Self::Jit { process, thread } => thread.run_slice(process, request),
        }
    }

    pub(crate) fn synchronize_address_space(
        &mut self,
        binding: MemoryBinding<'_>,
        _state: &ThreadCpuState,
    ) -> Result<(), CpuFault> {
        match self {
            Self::Interpreter(thread) => thread.synchronize_address_space(binding),
            Self::Jit { process, thread } => thread.synchronize_address_space(process, binding),
        }
    }

    #[must_use]
    pub(crate) fn control(&self) -> CpuControl {
        match self {
            Self::Interpreter(thread) => thread.control(),
            Self::Jit { thread, .. } => thread.control(),
        }
    }

    pub(crate) fn prepare_shutdown(
        &mut self,
        binding: MemoryBinding<'_>,
        _state: &ThreadCpuState,
    ) -> Result<(), CpuFault> {
        match self {
            Self::Interpreter(thread) => thread.prepare_shutdown(binding),
            Self::Jit { process, thread } => thread.prepare_shutdown(process, binding),
        }
    }

    pub(crate) fn clear_local_exclusive_reservation(&mut self) {
        match self {
            Self::Interpreter(thread) => thread.clear_local_exclusive_reservation(),
            Self::Jit { thread, .. } => thread.clear_local_exclusive_reservation(),
        }
    }
}

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
    backend: CpuBackend,
    process_id: CpuProcessId,
    controls: BTreeMap<nixe_scheduler::VirtualCpuId, CpuControl>,
    cpu: ProcessCpuContext,
    virtual_clock: VirtualClock,
    architectural_timer_frequency: u64,
    address_space_end: GuestVirtualAddress,
    shutdown_result: Option<Result<(), CpuFault>>,
    stopping: bool,
    pending_safepoint: AtomicBool,
    transition_safepoints: Arc<TransitionSafepoints>,
}

#[derive(Default)]
struct TransitionSafepoints {
    controls: Mutex<BTreeMap<nixe_scheduler::VirtualCpuId, CpuControl>>,
}

impl TransitionSafepoints {
    fn register(&self, vcpu: nixe_scheduler::VirtualCpuId, control: CpuControl) {
        self.lock_controls().insert(vcpu, control);
    }

    fn clear(&self) {
        self.lock_controls().clear();
    }

    fn request(&self) {
        for control in self
            .lock_controls()
            .values()
            .filter(|control| control.execution_active())
        {
            control.request(ControlRequest::Preempt);
        }
    }

    fn lock_controls(
        &self,
    ) -> std::sync::MutexGuard<'_, BTreeMap<nixe_scheduler::VirtualCpuId, CpuControl>> {
        self.controls.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

pub(crate) struct CpuThreadTeardownState {
    memory: Arc<ExecutionMemory>,
    cpu: ProcessCpuContext,
    address_space_end: GuestVirtualAddress,
    state: ThreadCpuState,
}

impl CpuThreadTeardownState {
    pub(crate) fn prepare(&self, thread: &mut CpuThread) -> Result<(), CpuFault> {
        thread.prepare_shutdown(self.binding(), &self.state)
    }

    fn binding(&self) -> MemoryBinding<'_> {
        MemoryBinding {
            address_space: self.cpu.address_space_id(),
            end_exclusive: self.address_space_end,
            memory: self.memory.as_ref(),
            mapping_epoch: self.memory.mapping_epoch().get(),
            invalidation_cursor: self.memory.invalidation_cursor(),
        }
    }
}

pub(crate) struct ProcessExecutionConfiguration {
    pub(crate) virtual_clock: VirtualClock,
    pub(crate) timer_frequency: u64,
    pub(crate) cpu: ProcessCpuContext,
    pub(crate) address_space_end: GuestVirtualAddress,
}

impl ProcessExecutionControl {
    pub(crate) fn new(
        configuration: ProcessExecutionConfiguration,
        memory: &ExecutionMemory,
        process_id: CpuProcessId,
        selection: &CpuBackendConfig,
    ) -> Result<Self, CpuFault> {
        let ProcessExecutionConfiguration {
            virtual_clock,
            timer_frequency,
            cpu,
            address_space_end,
        } = configuration;
        let mut backend = CpuBackend::new(selection, process_id, cpu)?;
        if let Err(fault) = backend.bind_memory(MemoryBinding {
            address_space: cpu.address_space_id(),
            end_exclusive: address_space_end,
            memory,
            mapping_epoch: memory.mapping_epoch().get(),
            invalidation_cursor: memory.invalidation_cursor(),
        }) {
            let _ = backend.shutdown();
            return Err(fault);
        }
        let transition_safepoints = Arc::new(TransitionSafepoints::default());
        let weak_safepoints = Arc::downgrade(&transition_safepoints);
        memory.set_transition_notifier(Some(Arc::new(move || {
            if let Some(safepoints) = weak_safepoints.upgrade() {
                safepoints.request();
            }
        })));
        Ok(Self {
            backend,
            process_id,
            controls: BTreeMap::new(),
            cpu,
            virtual_clock,
            architectural_timer_frequency: timer_frequency,
            address_space_end,
            shutdown_result: None,
            stopping: false,
            pending_safepoint: AtomicBool::new(false),
            transition_safepoints,
        })
    }

    pub(crate) const fn backend_name(&self) -> &'static str {
        self.backend.name()
    }
    pub(crate) const fn process_id(&self) -> CpuProcessId {
        self.process_id
    }

    pub(crate) fn request_safepoint(&self) {
        if self.controls.is_empty() {
            self.pending_safepoint.store(true, Ordering::Release);
        } else {
            for control in self.controls.values() {
                control.request(ControlRequest::Preempt);
            }
        }
    }

    pub(crate) fn request_stop(&mut self) -> Result<(), CpuFault> {
        if !self.stopping {
            self.request_safepoint();
            self.backend.request_stop()?;
            self.stopping = true;
        }
        Ok(())
    }

    pub(crate) fn shutdown(&mut self) -> Result<(), CpuFault> {
        if let Some(result) = &self.shutdown_result {
            return result.clone();
        }
        self.request_stop()?;
        let result = self.backend.shutdown();
        self.shutdown_result = Some(result.clone());
        result
    }

    pub(crate) fn request_mapping_safepoint(&self) {
        for control in self
            .controls
            .values()
            .filter(|control| control.execution_active())
        {
            control.request(ControlRequest::Preempt);
        }
    }

    pub(crate) fn publish_memory_invalidation(
        &self,
        cursor: nixe_memory::MemoryInvalidationCursor,
    ) {
        for control in self.controls.values() {
            control.request_invalidation(cursor.get());
        }
    }

    pub(crate) fn memory_invalidation_acknowledged(
        &self,
        cursor: nixe_memory::MemoryInvalidationCursor,
    ) -> bool {
        let mut controls = self.controls.values();
        controls.next().is_some_and(|first| {
            first.acknowledged_invalidation(cursor.get())
                && controls.all(|control| control.acknowledged_invalidation(cursor.get()))
        })
    }

    pub(crate) fn create_worker_cpu_thread(
        &mut self,
        vcpu: nixe_scheduler::VirtualCpuId,
    ) -> Result<CpuThread, CpuFault> {
        if self.stopping {
            return Err(runtime_fault(
                self.cpu,
                CpuFaultKind::Unavailable,
                "CPU backend is stopping",
            ));
        }
        let thread = self
            .backend
            .create_thread(CpuThreadId::new(u64::from(vcpu.get()) + 1))?;
        let control = thread.control();
        if self.pending_safepoint.swap(false, Ordering::AcqRel) {
            control.request(ControlRequest::Preempt);
        }
        self.transition_safepoints.register(vcpu, control.clone());
        self.controls.insert(vcpu, control);
        Ok(thread)
    }

    pub(crate) fn cpu_thread_teardown_state(
        &self,
        memory: Arc<ExecutionMemory>,
        state: ThreadCpuState,
    ) -> CpuThreadTeardownState {
        CpuThreadTeardownState {
            memory,
            cpu: self.cpu,
            address_space_end: self.address_space_end,
            state,
        }
    }

    pub(crate) fn complete_cpu_thread_retirement(
        &mut self,
        cursor: nixe_memory::MemoryInvalidationCursor,
    ) -> Result<(), CpuFault> {
        if self
            .controls
            .values()
            .any(|control| !control.acknowledged_invalidation(cursor.get()))
        {
            return Err(runtime_fault(
                self.cpu,
                CpuFaultKind::Internal,
                "CPU thread retirement did not acknowledge the final invalidation cursor",
            ));
        }
        self.controls.clear();
        self.transition_safepoints.clear();
        Ok(())
    }

    pub(crate) fn execution_environment(
        &self,
    ) -> (VirtualClock, u64, ProcessCpuContext, GuestVirtualAddress) {
        (
            self.virtual_clock.clone(),
            self.architectural_timer_frequency,
            self.cpu,
            self.address_space_end,
        )
    }
}

impl Drop for ProcessExecutionControl {
    fn drop(&mut self) {
        if self.shutdown_result.is_none() {
            let _ = self.shutdown();
        }
    }
}

impl crate::exception_dispatch::MemoryMutationControl for ProcessExecutionControl {
    fn request_mapping_safepoint(&self) {
        Self::request_mapping_safepoint(self);
    }
    fn publish_memory_invalidation(&self, cursor: nixe_memory::MemoryInvalidationCursor) {
        Self::publish_memory_invalidation(self, cursor);
    }
}

fn runtime_fault(
    _cpu: ProcessCpuContext,
    kind: CpuFaultKind,
    message: impl Into<Box<str>>,
) -> CpuFault {
    CpuFault {
        backend: "runtime",
        kind,
        instructions_executed: 0,
        message: message.into(),
        context: Box::new(ThreadCpuState::default().register_context()),
    }
}

struct RuntimeTimer<'a> {
    clock: &'a VirtualClock,
    frequency: u64,
}

pub(crate) struct VcpuExecutionState {
    pub(crate) thread: ThreadCpuState,
    pub(crate) cpu: ProcessCpuContext,
    pub(crate) memory: Arc<ExecutionMemory>,
    pub(crate) virtual_clock: VirtualClock,
    pub(crate) architectural_timer_frequency: u64,
    pub(crate) address_space_end: GuestVirtualAddress,
    pub(crate) instruction_budget: u64,
    pub(crate) loader_return: Option<GuestVirtualAddress>,
    pub(crate) events: VcpuEventState,
}

impl VcpuExecutionState {
    pub(crate) fn run(
        &mut self,
        thread: &mut CpuThread,
    ) -> Result<ExecutionReport, ProcessExecutionError> {
        let memory_lease = self.memory.acquire_execution_lease();
        if let Some(error) = self.memory.direct_backend_failure() {
            return Err(ProcessExecutionError::Cpu {
                fault: runtime_fault(self.cpu, CpuFaultKind::Internal, error),
            });
        }
        thread
            .synchronize_address_space(
                MemoryBinding {
                    address_space: self.cpu.address_space_id(),
                    end_exclusive: self.address_space_end,
                    memory: self.memory.as_ref(),
                    mapping_epoch: memory_lease.epoch().get(),
                    invalidation_cursor: self.memory.invalidation_cursor(),
                },
                &self.thread,
            )
            .map_err(|fault| ProcessExecutionError::Cpu { fault })?;
        let timer = RuntimeTimer {
            clock: &self.virtual_clock,
            frequency: self.architectural_timer_frequency,
        };
        let control = thread.control();
        let _execution = control.enter_execution();
        thread
            .run_slice(RunRequest {
                cpu: self.cpu,
                memory: self.memory.as_ref(),
                memory_lease: Some(memory_lease),
                state: &mut self.thread,
                instruction_budget: self.instruction_budget,
                loader_return: self.loader_return,
                timer: &timer,
                events: self.events.clone(),
            })
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
    state: &ThreadCpuState,
) -> LocationDescriptor {
    nixe_cpu::location::current_location(cpu, state)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn external_transition_requests_preemption_only_from_active_cpu_slices() {
        let safepoints = TransitionSafepoints::default();
        let control = CpuControl::default();
        safepoints.register(nixe_scheduler::VirtualCpuId::new(0), control.clone());

        safepoints.request();
        assert!(control.take_pending().is_none());

        let execution = control.enter_execution();
        safepoints.request();
        assert!(
            control
                .take_pending()
                .is_some_and(|pending| pending.contains(ControlRequest::Preempt))
        );
        drop(execution);
    }
}
