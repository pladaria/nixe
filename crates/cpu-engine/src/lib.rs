//! Engine-neutral CPU execution contracts.
//!
//! This crate owns the boundary between runtime orchestration and CPU execution
//! implementations. It deliberately has no dependency on runtime, Horizon,
//! scheduler, graphics, application configuration, or host backend APIs.

use core::fmt::{self, Display, Formatter};
use std::sync::Arc;

use nixe_cpu::coverage::CoverageId;
use nixe_cpu::error::{InstructionFetchFault, ProfileDisabledInstruction, UnallocatedEncoding};
use nixe_cpu::exception::ExceptionKind;
use nixe_cpu::location::{InstructionEncoding, LocationDescriptor};
use nixe_cpu::memory::{CpuMemory, DataAccessFault};
use nixe_cpu::profile::{GuestCpuProfile, ProcessCpuContext};
use nixe_cpu::state::{RegisterContext, ThreadCpuState};
use nixe_memory::GuestVirtualAddress;

mod capability;
mod conformance;
mod control;
mod handoff;
pub use capability::{
    CapabilityRejection, CapabilityRejectionReason, CapabilityReport, EngineCapabilities,
    EngineDescriptor, EngineKind, HostArchitecture, HostCapabilities,
};
pub use conformance::{
    CONFORMANCE_FALLBACK_ENCODING, ConformanceCase, ConformanceFailure, ConformanceReport,
    run_provider_conformance,
};
pub use control::{ControlSnapshot, CrossVcpuRequest, EngineControl, EngineExecutionGuard};
pub use handoff::{
    DomainMemory, DomainMemoryBinding, DomainQuiescenceToken, HandoffFailure, HandoffFailureStage,
    MemorySynchronizationRecord, StateCommitBarrier, prepare_handoff,
};

pub const MAX_INSTRUCTION_TRACE_ENTRIES: usize = 64;
pub const MAX_TRACE_DISASSEMBLY_BYTES: usize = 96;
pub const MAX_INSTRUCTION_TRACE_EXPORT_BYTES: usize = 16 * 1024;

macro_rules! identity {
    ($name:ident) => {
        #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        pub struct $name(u64);

        impl $name {
            #[must_use]
            pub const fn new(value: u64) -> Self {
                Self(value)
            }

            #[must_use]
            pub const fn get(self) -> u64 {
                self.0
            }
        }
    };
}

identity!(EngineId);
identity!(EngineDomainId);
identity!(EngineExecutorId);
identity!(EngineGeneration);
identity!(ControlEpoch);

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SafepointReason {
    Requested,
    PendingEvent { mask: u32 },
    Timer,
    MappingChanged,
    EngineHandoff,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum StateCommitStatus {
    Canonical,
    BackendPrivate,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct TimerSnapshot {
    pub counter: u64,
    pub frequency: u64,
}

pub trait EngineTimer {
    fn snapshot(&self) -> TimerSnapshot;
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct TracePolicy {
    pub enabled: bool,
    pub detailed: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InstructionTraceEntry {
    pub sequence: u64,
    pub source: LocationDescriptor,
    pub encoding: InstructionEncoding,
    pub disassembly: Option<Box<str>>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InstructionTrace {
    pub enabled: bool,
    pub entries: Box<[InstructionTraceEntry]>,
    pub discarded: u64,
}

impl InstructionTrace {
    #[must_use]
    pub const fn enabled(&self) -> bool {
        self.enabled
    }
    #[must_use]
    pub const fn entries(&self) -> &[InstructionTraceEntry] {
        &self.entries
    }
    #[must_use]
    pub const fn discarded(&self) -> u64 {
        self.discarded
    }
}

impl Display for InstructionTraceEntry {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
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

impl Display for InstructionTrace {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
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

/// One bounded request. Borrowed resources cannot escape the call.
pub struct RunRequest<'a> {
    pub cpu: ProcessCpuContext,
    pub memory: &'a dyn CpuMemory,
    pub state: &'a mut ThreadCpuState,
    pub instruction_budget: u64,
    pub loader_return: Option<GuestVirtualAddress>,
    pub timer: &'a dyn EngineTimer,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EngineExit {
    /// Requests one precise instruction through the configured semantic
    /// fallback engine. State and memory are canonical at `source`.
    InterpretOne {
        source: LocationDescriptor,
    },
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
        immediate: u32,
    },
    DataFault {
        source: LocationDescriptor,
        fault: DataAccessFault,
    },
    LoaderReturn {
        source: LocationDescriptor,
        result_code: u64,
    },
}

/// Precise synchronous exception presented to runtime policy.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ExceptionDispatchRequest {
    source: LocationDescriptor,
    kind: ExceptionKind,
    syndrome: Option<u64>,
}

impl ExceptionDispatchRequest {
    #[must_use]
    pub const fn new(
        source: LocationDescriptor,
        kind: ExceptionKind,
        syndrome: Option<u64>,
    ) -> Self {
        Self {
            source,
            kind,
            syndrome,
        }
    }
    #[must_use]
    pub const fn source(self) -> LocationDescriptor {
        self.source
    }
    #[must_use]
    pub const fn kind(self) -> ExceptionKind {
        self.kind
    }
    #[must_use]
    pub const fn syndrome(self) -> Option<u64> {
        self.syndrome
    }
}

impl EngineExit {
    #[must_use]
    pub fn exception_dispatch_request(&self) -> Option<ExceptionDispatchRequest> {
        let (source, kind, syndrome) = match self {
            Self::ArchitecturalException {
                source,
                kind,
                syndrome,
            } => (*source, *kind, *syndrome),
            Self::SupervisorCall { source, immediate } => (
                *source,
                ExceptionKind::SupervisorCall,
                Some(u64::from(*immediate)),
            ),
            Self::DataFault { source, .. } => (*source, ExceptionKind::DataAbort, None),
            Self::ProfileDisabled { error } => (
                error.instruction.location,
                ExceptionKind::UndefinedInstruction,
                None,
            ),
            Self::UnallocatedEncoding { error } => (
                error.instruction.location,
                ExceptionKind::UndefinedInstruction,
                None,
            ),
            _ => return None,
        };
        Some(ExceptionDispatchRequest::new(source, kind, syndrome))
    }
}

impl Display for EngineExit {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::InterpretOne { source } => write!(formatter, "interpret-one source=[{source}]"),
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
            Self::LoaderReturn {
                source,
                result_code,
            } => write!(
                formatter,
                "loader-return source=[{source}] result=0x{result_code:016x}"
            ),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionReport {
    pub instructions_executed: u64,
    pub stop: EngineExit,
    pub context: RegisterContext,
    pub trace: InstructionTrace,
    pub state_commit: StateCommitStatus,
}

impl Display for ExecutionReport {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "instructions={} stop=[{}] registers=[{}] trace=[{}] commit={:?}",
            self.instructions_executed, self.stop, self.context, self.trace, self.state_commit
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum EngineFaultKind {
    InvalidRequest,
    Internal,
    StateImport,
    StateExport,
    Synchronization,
    Unavailable,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EngineFault {
    pub engine: EngineId,
    pub kind: EngineFaultKind,
    pub instructions_executed: u64,
    pub message: Box<str>,
    pub context: Box<RegisterContext>,
}

impl Display for EngineFault {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "engine={} kind={:?} after={} message={} registers=[{}]",
            self.engine.get(),
            self.kind,
            self.instructions_executed,
            self.message,
            self.context
        )
    }
}

impl std::error::Error for EngineFault {}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DomainRequest {
    pub domain: EngineDomainId,
    pub cpu: ProcessCpuContext,
}

/// Construction parameters for one worker-owned vCPU executor.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ExecutorRequest {
    pub executor: EngineExecutorId,
    pub trace: TracePolicy,
}

pub trait EngineProvider: Send + Sync {
    fn descriptor(&self) -> EngineDescriptor;
    /// Probes the complete guest profile rather than a registry identity. This
    /// lets providers conservatively reject unknown architectural features.
    fn probe(&self, profile: GuestCpuProfile, required: EngineCapabilities) -> CapabilityReport;
    fn create_domain(&self, request: DomainRequest) -> Result<Box<dyn EngineDomain>, EngineFault>;
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum EnginePreference {
    #[default]
    Auto,
    Explicit(EngineId),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EngineSelectionError {
    pub requested: EnginePreference,
    pub reports: Box<[CapabilityReport]>,
}

impl Display for EngineSelectionError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(formatter, "no CPU engine satisfies {:?}", self.requested)
    }
}
impl std::error::Error for EngineSelectionError {}

/// Deterministically ordered provider registry. `Auto` selects the first
/// available provider, so applications control policy through registration
/// order rather than host-dependent iteration order.
#[derive(Default)]
pub struct EngineRegistry {
    providers: Vec<Arc<dyn EngineProvider>>,
}

impl EngineRegistry {
    #[must_use]
    pub fn new(providers: impl IntoIterator<Item = Arc<dyn EngineProvider>>) -> Self {
        Self {
            providers: providers.into_iter().collect(),
        }
    }

    pub fn select(
        &self,
        profile: GuestCpuProfile,
        required: EngineCapabilities,
        preference: EnginePreference,
    ) -> Result<Arc<dyn EngineProvider>, EngineSelectionError> {
        let reports: Vec<_> = self
            .providers
            .iter()
            .map(|provider| {
                let declared = provider.descriptor();
                let mut report = provider.probe(profile, required);
                if report.descriptor != declared
                    || !declared.capabilities.is_coherent()
                    || !declared.capabilities.contains(required)
                    || !declared.capabilities.supports_profile(profile, required)
                {
                    report.available = false;
                    let mut rejections = Vec::from(report.rejections);
                    rejections.push(CapabilityRejection {
                        engine: declared.id,
                        reason: CapabilityRejectionReason::MissingCapabilities,
                        detail: "provider report disagrees with its descriptor or guest profile"
                            .into(),
                    });
                    report.rejections = rejections.into_boxed_slice();
                }
                report
            })
            .collect();
        let selected = match preference {
            EnginePreference::Auto => reports.iter().position(|report| report.available),
            EnginePreference::Explicit(id) => reports
                .iter()
                .position(|report| report.available && report.descriptor.id == id),
        };
        selected
            .map(|index| Arc::clone(&self.providers[index]))
            .ok_or_else(|| EngineSelectionError {
                requested: preference,
                reports: reports.into_boxed_slice(),
            })
    }
}

pub trait EngineDomain: Send {
    fn descriptor(&self) -> EngineDescriptor;
    fn domain_id(&self) -> EngineDomainId;
    /// Creates executor-local state for an already-bound domain. Transactional
    /// replacement may create executors before [`Self::activate`]; they cannot
    /// enter guest execution until activation succeeds.
    fn create_executor(
        &mut self,
        request: ExecutorRequest,
    ) -> Result<Box<dyn EngineExecutor>, EngineFault>;
    /// Binds the complete canonical address-space view before executors are
    /// created. Implementations must copy or retain safe backing objects; they
    /// must not retain the borrowed trait object.
    fn bind_memory(&mut self, _binding: DomainMemoryBinding<'_>) -> Result<(), EngineFault> {
        Ok(())
    }
    /// Flushes backend-private writes and acknowledges the current mapping
    /// generation before an engine handoff or teardown.
    fn synchronize_memory(
        &mut self,
        binding: DomainMemoryBinding<'_>,
    ) -> Result<MemorySynchronizationRecord, EngineFault> {
        Ok(binding.synchronization_record())
    }
    /// Imports the synchronization point exported by the previous domain.
    fn import_memory(&mut self, _record: MemorySynchronizationRecord) -> Result<(), EngineFault> {
        Ok(())
    }
    /// Makes a bound, synchronized domain available for executor entry.
    fn activate(&mut self) -> Result<(), EngineFault> {
        Ok(())
    }
    fn quiesce(&mut self) -> Result<DomainQuiescenceToken, EngineFault>;
    /// Permanently releases domain resources after every executor was dropped.
    /// Implementations with external resources must make this idempotent.
    fn shutdown(&mut self) -> Result<(), EngineFault> {
        self.quiesce().map(|_| ())
    }
}

/// Mutable execution resources leased exclusively to one host worker.
pub trait EngineExecutor: Send {
    fn descriptor(&self) -> EngineDescriptor;
    fn executor_id(&self) -> EngineExecutorId;
    fn run_slice(&mut self, request: RunRequest<'_>) -> Result<ExecutionReport, EngineFault>;
    /// Applies mapping/code invalidation before guest re-entry. Providers which
    /// advertise acknowledged invalidation must override this method.
    fn synchronize_invalidation(
        &mut self,
        epoch: u64,
        state: &ThreadCpuState,
        memory: &dyn CpuMemory,
    ) -> Result<(), EngineFault> {
        let _ = memory;
        if !self.descriptor().capabilities.acknowledged_invalidation {
            return Ok(());
        }
        Err(EngineFault {
            engine: self.descriptor().id,
            kind: EngineFaultKind::InvalidRequest,
            instructions_executed: 0,
            message: format!(
                "engine advertises invalidation acknowledgement but cannot synchronize epoch {epoch}"
            )
            .into_boxed_str(),
            context: Box::new(state.register_context()),
        })
    }
    /// Reconciles mappings and canonical backing before guest re-entry. Native
    /// engines override this hook; semantic engines inherit epoch handling.
    fn synchronize_address_space(
        &mut self,
        binding: DomainMemoryBinding<'_>,
        state: &ThreadCpuState,
    ) -> Result<(), EngineFault> {
        self.synchronize_invalidation(binding.invalidation_generation, state, binding.memory)
    }
    /// Returns a cloneable control path which remains usable while a worker
    /// exclusively owns this executor.
    fn control(&self) -> Option<EngineControl> {
        None
    }
    /// Clears executor-local exclusive-reservation state at an explicit
    /// scheduler migration or context-switch boundary.
    fn clear_local_exclusive_reservation(&mut self);
}

/// Executes the canonical `InterpretOne` fallback contract. The helper forces
/// an exact one-instruction upper bound; architectural exits are returned
/// unchanged and therefore never acquire a synthesized fallthrough target.
pub fn run_interpret_one_fallback(
    executor: &mut dyn EngineExecutor,
    request: RunRequest<'_>,
) -> Result<ExecutionReport, EngineFault> {
    executor.run_slice(RunRequest {
        instruction_budget: 1,
        ..request
    })
}
