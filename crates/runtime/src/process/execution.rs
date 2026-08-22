//! Process-local execution domain and per-vCPU executor ownership.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use nixe_cpu::location::{ExecutionState, LocationDescriptor};
use nixe_cpu::memory::ExecutionMemory;
use nixe_cpu::profile::ProcessCpuContext;
use nixe_cpu::state::{RegisterContext, ThreadCpuState};
use nixe_cpu_engine::{
    DomainRequest, EngineDomain, EngineDomainId, EngineExecutor, EngineExecutorId, EngineFault,
    EngineProvider, EngineTimer, ExecutorRequest, RunRequest, TimerSnapshot, TracePolicy,
};
use nixe_memory::GuestVirtualAddress;

use crate::{
    DiagnosticsPolicy, ExceptionTerminationScope, GuestBreakPayload, ReportDetail, VirtualClock,
};

static NEXT_ENGINE_DOMAIN_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

pub(super) fn allocate_engine_domain_id() -> Option<EngineDomainId> {
    NEXT_ENGINE_DOMAIN_ID
        .fetch_update(
            std::sync::atomic::Ordering::Relaxed,
            std::sync::atomic::Ordering::Relaxed,
            |id| id.checked_add(1),
        )
        .ok()
        .map(EngineDomainId::new)
}

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
    GuestBreak {
        reason: u64,
        info: u64,
        size: u64,
        payload: Option<GuestBreakPayload>,
    },
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
    ConcurrentProcessStop {
        status: ProcessExecutionStatus,
        context: Box<RegisterContext>,
    },
    ExecutorUnavailable {
        engine: EngineDomainId,
    },
    FallbackUnavailable {
        engine: nixe_cpu_engine::EngineId,
    },
    InvalidFallbackBoundary {
        engine: nixe_cpu_engine::EngineId,
        expected: LocationDescriptor,
        requested: LocationDescriptor,
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
            Self::ConcurrentProcessStop { status, context } => write!(
                formatter,
                "another vCPU stopped the process while this slice was in flight: status={status:?}, registers=[{context}]"
            ),
            Self::ExecutorUnavailable { engine } => {
                write!(
                    formatter,
                    "CPU executor for engine domain {} is unavailable",
                    engine.get()
                )
            }
            Self::FallbackUnavailable { engine } => {
                write!(
                    formatter,
                    "CPU engine {} requested an unavailable semantic fallback",
                    engine.get()
                )
            }
            Self::InvalidFallbackBoundary {
                engine,
                expected,
                requested,
            } => write!(
                formatter,
                "CPU engine {} requested semantic fallback at {requested:?}, but canonical execution is at {expected:?}",
                engine.get()
            ),
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProcessTeardownFailure {
    pub report: Box<ProcessTeardownReport>,
    pub fault: Box<EngineFault>,
}

impl Display for ProcessTeardownFailure {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "CPU engine domain failed to shut down during teardown: {}",
            self.fault
        )
    }
}

impl Error for ProcessTeardownFailure {}

pub(crate) struct ProcessExecutionControl {
    domain: Box<dyn EngineDomain>,
    executors: BTreeMap<nixe_scheduler::VirtualCpuId, Box<dyn EngineExecutor>>,
    fallback: Option<FallbackExecutionControl>,
    controls: BTreeMap<nixe_scheduler::VirtualCpuId, nixe_cpu_engine::EngineControl>,
    cpu: ProcessCpuContext,
    trace_policy: TracePolicy,
    virtual_clock: VirtualClock,
    architectural_timer_frequency: u64,
    address_space_end: nixe_memory::GuestVirtualAddress,
    shutdown_result: Option<Result<(), EngineFault>>,
    pending_safepoint: AtomicBool,
    pending_events: AtomicU32,
}

struct FallbackExecutionControl {
    domain: Box<dyn EngineDomain>,
    executors: BTreeMap<nixe_scheduler::VirtualCpuId, Box<dyn EngineExecutor>>,
}

pub(crate) struct ProcessExecutionConfiguration {
    pub(crate) diagnostics: DiagnosticsPolicy,
    pub(crate) virtual_clock: VirtualClock,
    pub(crate) timer_frequency: u64,
    pub(crate) cpu: ProcessCpuContext,
    pub(crate) address_space_end: nixe_memory::GuestVirtualAddress,
}

impl ProcessExecutionControl {
    pub(crate) fn with_provider(
        configuration: ProcessExecutionConfiguration,
        memory: &nixe_cpu::memory::ExecutionMemory,
        domain_id: EngineDomainId,
        provider: &dyn EngineProvider,
        fallback: Option<(EngineDomainId, &dyn EngineProvider)>,
    ) -> Result<Self, EngineFault> {
        let ProcessExecutionConfiguration {
            diagnostics,
            virtual_clock,
            timer_frequency,
            cpu,
            address_space_end,
        } = configuration;
        let trace = TracePolicy {
            enabled: diagnostics.instruction_trace,
            detailed: diagnostics.report_detail == ReportDetail::Detailed,
        };
        let mut domain = provider.create_domain(DomainRequest {
            domain: domain_id,
            cpu,
        })?;
        let binding = nixe_cpu_engine::DomainMemoryBinding {
            address_space: cpu.address_space_id(),
            end_exclusive: address_space_end,
            memory,
            invalidation_generation: memory.mapping_epoch().get(),
        };
        if let Err(fault) = domain.bind_memory(binding) {
            let _ = domain.shutdown();
            return Err(fault);
        }
        let executor = match domain.create_executor(ExecutorRequest {
            executor: EngineExecutorId::new(1),
            trace,
        }) {
            Ok(executor) => executor,
            Err(fault) => {
                let _ = domain.shutdown();
                return Err(fault);
            }
        };
        let descriptor = domain.descriptor();
        if executor.control().is_none() && descriptor.capabilities.requires_control_path() {
            let fault = missing_control_path_fault(cpu, descriptor.id);
            let _ = domain.shutdown();
            return Err(fault);
        }
        drop(executor);
        let fallback = match fallback
            .map(|(domain_id, provider)| create_fallback_domain(cpu, binding, domain_id, provider))
            .transpose()
        {
            Ok(fallback) => fallback,
            Err(fault) => {
                let _ = domain.shutdown();
                return Err(fault);
            }
        };
        if fallback.as_ref().is_some_and(|fallback| {
            fallback
                .domain
                .descriptor()
                .capabilities
                .interpret_one_fallback
        }) {
            let fault = engine_fault(
                cpu,
                descriptor.id,
                nixe_cpu_engine::EngineFaultKind::InvalidRequest,
                "semantic fallback engines cannot recursively request InterpretOne",
            );
            let _ = domain.shutdown();
            if let Some(mut fallback) = fallback {
                let _ = fallback.domain.shutdown();
            }
            return Err(fault);
        }
        if descriptor.capabilities.interpret_one_fallback && fallback.is_none() {
            let _ = domain.shutdown();
            return Err(engine_fault(
                cpu,
                descriptor.id,
                nixe_cpu_engine::EngineFaultKind::InvalidRequest,
                "engine advertises InterpretOne exits without a configured fallback provider",
            ));
        }
        Ok(Self {
            domain,
            executors: BTreeMap::new(),
            fallback,
            controls: BTreeMap::new(),
            cpu,
            trace_policy: trace,
            virtual_clock,
            architectural_timer_frequency: timer_frequency,
            address_space_end,
            shutdown_result: None,
            pending_safepoint: AtomicBool::new(false),
            pending_events: AtomicU32::new(0),
        })
    }

    pub(crate) fn engine_descriptor(&self) -> nixe_cpu_engine::EngineDescriptor {
        self.domain.descriptor()
    }

    pub(crate) const fn instruction_trace_enabled(&self) -> bool {
        self.trace_policy.enabled
    }

    pub(crate) fn domain_id(&self) -> EngineDomainId {
        self.domain.domain_id()
    }

    pub(crate) fn fallback_domain_id(&self) -> Option<EngineDomainId> {
        self.fallback
            .as_ref()
            .map(|fallback| fallback.domain.domain_id())
    }
    pub(crate) fn request_safepoint(&self) {
        if self.controls.is_empty() {
            self.pending_safepoint.store(true, Ordering::Release);
            return;
        }
        for control in self.controls.values() {
            control.request(nixe_cpu_engine::CrossVcpuRequest::Preempt);
        }
    }

    pub(crate) fn shutdown(&mut self) -> Result<(), EngineFault> {
        if let Some(result) = &self.shutdown_result {
            return result.clone();
        }
        self.request_safepoint();
        self.executors.clear();
        if let Some(fallback) = &mut self.fallback {
            fallback.executors.clear();
        }
        let primary_result = self.domain.shutdown();
        let fallback_result = self
            .fallback
            .as_mut()
            .map_or(Ok(()), |fallback| fallback.domain.shutdown());
        let result = primary_result.and(fallback_result);
        self.shutdown_result = Some(result.clone());
        result
    }
    pub(crate) fn request_mapping_safepoint(&self) {
        for control in self.controls.values() {
            if control.execution_active() {
                control.request(nixe_cpu_engine::CrossVcpuRequest::Preempt);
            }
        }
    }
    pub(crate) fn publish_mapping_invalidation(&self, epoch: nixe_cpu::memory::MappingEpoch) {
        for control in self.controls.values() {
            control.request_invalidation(epoch.get());
        }
    }
    pub(crate) fn mapping_invalidation_acknowledged(
        &self,
        epoch: nixe_cpu::memory::MappingEpoch,
    ) -> bool {
        self.controls.values().next().is_some_and(|_| {
            self.controls
                .values()
                .all(|control| control.acknowledged_invalidation(epoch.get()))
        })
    }
    #[cfg(test)]
    pub(crate) fn post_event(&self, mask: u32) {
        if self.controls.is_empty() {
            self.pending_events.fetch_or(mask, Ordering::Release);
            return;
        }
        for control in self.controls.values() {
            control.post_event(mask);
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
            if let Some(control) = executor.control() {
                self.publish_pending_control(&control);
                self.controls.insert(vcpu, control);
            } else if self
                .domain
                .descriptor()
                .capabilities
                .requires_control_path()
            {
                return Err(missing_control_path_fault(
                    self.cpu,
                    self.domain.descriptor().id,
                ));
            }
            self.executors.insert(vcpu, executor);
        }
        Ok(self
            .executors
            .get_mut(&vcpu)
            .expect("executor was installed"))
    }

    pub(crate) fn lease_executor(
        &mut self,
        vcpu: nixe_scheduler::VirtualCpuId,
    ) -> Result<Box<dyn EngineExecutor>, EngineFault> {
        self.executor_for(vcpu)?;
        Ok(self
            .executors
            .remove(&vcpu)
            .expect("a prepared executor can be leased exactly once"))
    }

    pub(crate) fn restore_executor(
        &mut self,
        vcpu: nixe_scheduler::VirtualCpuId,
        executor: Box<dyn EngineExecutor>,
    ) {
        debug_assert!(!self.executors.contains_key(&vcpu));
        self.executors.insert(vcpu, executor);
    }

    pub(crate) fn lease_fallback_executor(
        &mut self,
        vcpu: nixe_scheduler::VirtualCpuId,
    ) -> Result<Option<Box<dyn EngineExecutor>>, EngineFault> {
        let Some(fallback) = &mut self.fallback else {
            return Ok(None);
        };
        if !fallback.executors.contains_key(&vcpu) {
            let executor = fallback.domain.create_executor(ExecutorRequest {
                executor: Self::executor_id(vcpu),
                trace: self.trace_policy,
            })?;
            fallback.executors.insert(vcpu, executor);
        }
        Ok(fallback.executors.remove(&vcpu))
    }

    pub(crate) fn restore_fallback_executor(
        &mut self,
        vcpu: nixe_scheduler::VirtualCpuId,
        executor: Box<dyn EngineExecutor>,
    ) {
        let fallback = self
            .fallback
            .as_mut()
            .expect("a leased fallback executor retains its domain");
        debug_assert!(!fallback.executors.contains_key(&vcpu));
        fallback.executors.insert(vcpu, executor);
    }

    fn publish_pending_control(&self, control: &nixe_cpu_engine::EngineControl) {
        if self.pending_safepoint.swap(false, Ordering::AcqRel) {
            control.request(nixe_cpu_engine::CrossVcpuRequest::Preempt);
        }
        let events = self.pending_events.swap(0, Ordering::AcqRel);
        if events != 0 {
            control.post_event(events);
        }
    }

    pub(crate) fn execution_environment(
        &self,
    ) -> (
        VirtualClock,
        u64,
        ProcessCpuContext,
        nixe_memory::GuestVirtualAddress,
    ) {
        (
            self.virtual_clock.clone(),
            self.architectural_timer_frequency,
            self.cpu,
            self.address_space_end,
        )
    }
}

fn create_fallback_domain(
    cpu: ProcessCpuContext,
    binding: nixe_cpu_engine::DomainMemoryBinding<'_>,
    domain_id: EngineDomainId,
    provider: &dyn EngineProvider,
) -> Result<FallbackExecutionControl, EngineFault> {
    let mut domain = provider.create_domain(DomainRequest {
        domain: domain_id,
        cpu,
    })?;
    let result = domain.bind_memory(binding).and_then(|()| {
        domain
            .create_executor(ExecutorRequest {
                executor: EngineExecutorId::new(1),
                trace: TracePolicy {
                    enabled: false,
                    detailed: false,
                },
            })
            .map(drop)
    });
    if let Err(fault) = result {
        let _ = domain.shutdown();
        return Err(fault);
    }
    Ok(FallbackExecutionControl {
        domain,
        executors: BTreeMap::new(),
    })
}

impl Drop for ProcessExecutionControl {
    fn drop(&mut self) {
        if self.shutdown_result.is_none() {
            let _ = self.shutdown();
        }
    }
}

impl crate::exception_dispatch::MappingInvalidationControl for ProcessExecutionControl {
    fn request_mapping_safepoint(&self) {
        Self::request_mapping_safepoint(self);
    }

    fn publish_mapping_invalidation(&self, epoch: nixe_cpu::memory::MappingEpoch) {
        Self::publish_mapping_invalidation(self, epoch);
    }
}

fn missing_control_path_fault(
    cpu: ProcessCpuContext,
    engine: nixe_cpu_engine::EngineId,
) -> EngineFault {
    engine_fault(
        cpu,
        engine,
        nixe_cpu_engine::EngineFaultKind::InvalidRequest,
        "engine advertises parallel control capabilities without an out-of-band control path",
    )
}

fn engine_fault(
    cpu: ProcessCpuContext,
    engine: nixe_cpu_engine::EngineId,
    kind: nixe_cpu_engine::EngineFaultKind,
    message: &'static str,
) -> EngineFault {
    let configuration = cpu
        .thread_configuration(ExecutionState::A64)
        .expect("supported process profiles include A64");
    let context = ThreadCpuState::new(configuration).register_context();
    EngineFault {
        engine,
        kind,
        instructions_executed: 0,
        message: message.into(),
        context: Box::new(context),
    }
}

struct RuntimeTimer<'a> {
    clock: &'a VirtualClock,
    frequency: u64,
}

/// All mutable state which crosses the coordinator/worker boundary for one
/// bounded slice. No borrowed process state or raw host pointer can escape in
/// a worker message.
pub(crate) struct VcpuExecutionState {
    pub(crate) thread: ThreadCpuState,
    pub(crate) cpu: ProcessCpuContext,
    pub(crate) memory: Arc<ExecutionMemory>,
    pub(crate) virtual_clock: VirtualClock,
    pub(crate) architectural_timer_frequency: u64,
    pub(crate) address_space_end: GuestVirtualAddress,
    pub(crate) instruction_budget: u64,
    pub(crate) loader_return: Option<GuestVirtualAddress>,
}

impl VcpuExecutionState {
    pub(crate) fn run(
        &mut self,
        executor: &mut dyn EngineExecutor,
    ) -> Result<ExecutionReport, ProcessExecutionError> {
        self.run_with_budget(executor, self.instruction_budget)
    }

    pub(crate) fn run_with_budget(
        &mut self,
        executor: &mut dyn EngineExecutor,
        instruction_budget: u64,
    ) -> Result<ExecutionReport, ProcessExecutionError> {
        let memory_lease = self.memory.acquire_execution_lease();
        executor
            .synchronize_address_space(
                nixe_cpu_engine::DomainMemoryBinding {
                    address_space: self.cpu.address_space_id(),
                    end_exclusive: self.address_space_end,
                    memory: self.memory.as_ref(),
                    invalidation_generation: memory_lease.epoch().get(),
                },
                &self.thread,
            )
            .map_err(|fault| ProcessExecutionError::Engine { fault })?;
        let timer = RuntimeTimer {
            clock: &self.virtual_clock,
            frequency: self.architectural_timer_frequency,
        };
        let _execution = executor.control().map(|control| control.enter_execution());
        let result = executor.run_slice(RunRequest {
            cpu: self.cpu,
            memory: self.memory.as_ref(),
            state: &mut self.thread,
            instruction_budget,
            loader_return: self.loader_return,
            timer: &timer,
        });
        result.map_err(|fault| ProcessExecutionError::Engine { fault })
    }
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

pub(crate) fn current_location(
    cpu: ProcessCpuContext,
    state: &ThreadCpuState,
) -> LocationDescriptor {
    nixe_cpu::location::current_location(cpu, state)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use nixe_cpu_engine::{
        CapabilityReport, EngineCapabilities, EngineDescriptor, EngineId, EngineKind,
    };
    use nixe_memory::AddressSpaceId;

    const ENGINE_ID: EngineId = EngineId::new(0xf001);

    fn descriptor() -> EngineDescriptor {
        EngineDescriptor {
            id: ENGINE_ID,
            name: "shutdown-failure-test".into(),
            kind: EngineKind::Test,
            capabilities: EngineCapabilities {
                a64: true,
                ..Default::default()
            },
        }
    }

    struct FailingProvider(Arc<AtomicUsize>);

    impl EngineProvider for FailingProvider {
        fn descriptor(&self) -> EngineDescriptor {
            descriptor()
        }

        fn probe(
            &self,
            _profile: nixe_cpu::profile::GuestCpuProfile,
            _required: EngineCapabilities,
        ) -> CapabilityReport {
            CapabilityReport {
                descriptor: descriptor(),
                available: true,
                rejections: Box::new([]),
            }
        }

        fn create_domain(
            &self,
            request: DomainRequest,
        ) -> Result<Box<dyn EngineDomain>, EngineFault> {
            Ok(Box::new(FailingDomain {
                id: request.domain,
                attempts: Arc::clone(&self.0),
                cpu: request.cpu,
            }))
        }
    }

    struct FailingDomain {
        id: EngineDomainId,
        attempts: Arc<AtomicUsize>,
        cpu: ProcessCpuContext,
    }

    impl EngineDomain for FailingDomain {
        fn descriptor(&self) -> EngineDescriptor {
            descriptor()
        }

        fn domain_id(&self) -> EngineDomainId {
            self.id
        }

        fn create_executor(
            &mut self,
            request: ExecutorRequest,
        ) -> Result<Box<dyn EngineExecutor>, EngineFault> {
            Ok(Box::new(InertExecutor(request.executor)))
        }

        fn shutdown(&mut self) -> Result<(), EngineFault> {
            self.attempts.fetch_add(1, Ordering::AcqRel);
            Err(engine_fault(
                self.cpu,
                ENGINE_ID,
                nixe_cpu_engine::EngineFaultKind::Internal,
                "injected shutdown failure",
            ))
        }
    }

    struct InertExecutor(EngineExecutorId);

    impl EngineExecutor for InertExecutor {
        fn descriptor(&self) -> EngineDescriptor {
            descriptor()
        }

        fn executor_id(&self) -> EngineExecutorId {
            self.0
        }

        fn run_slice(&mut self, _request: RunRequest<'_>) -> Result<ExecutionReport, EngineFault> {
            unreachable!("the shutdown test never enters guest execution")
        }

        fn clear_local_exclusive_reservation(&mut self) {}
    }

    #[test]
    fn failed_shutdown_is_reported_once_and_drop_does_not_retry() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let cpu = ProcessCpuContext::new(
            crate::ProcessBuildConfig::default().cpu_profile,
            AddressSpaceId::new(1),
        );
        {
            let provider = FailingProvider(Arc::clone(&attempts));
            let memory = nixe_cpu::memory::ExecutionMemory::new();
            let mut execution = ProcessExecutionControl::with_provider(
                ProcessExecutionConfiguration {
                    diagnostics: DiagnosticsPolicy::default(),
                    virtual_clock: VirtualClock::default(),
                    timer_frequency: 19_200_000,
                    cpu,
                    address_space_end: nixe_memory::GuestVirtualAddress::new(1_u64 << 39),
                },
                &memory,
                allocate_engine_domain_id().unwrap(),
                &provider,
                None,
            )
            .unwrap();
            assert!(execution.shutdown().is_err());
            assert!(execution.shutdown().is_err());
        }
        assert_eq!(attempts.load(Ordering::Acquire), 1);
    }
}
