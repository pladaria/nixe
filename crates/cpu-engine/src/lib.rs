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
use nixe_cpu::memory::{CpuMemory, DataAccessFault, MemoryPermissions};
use nixe_cpu::profile::{CpuProfileId, ProcessCpuContext};
use nixe_cpu::state::{RegisterContext, ThreadCpuState};
use nixe_memory::{AddressSpaceId, GuestVirtualAddress};

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

/// Stable implementation family used for selection and diagnostics.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum EngineKind {
    Interpreter,
    NativeCodeExecution,
    Test,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum HostArchitecture {
    Aarch64,
    X86_64,
    Other,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct HostCapabilities {
    pub architecture: HostArchitecture,
    pub logical_parallelism: Option<usize>,
}

impl HostCapabilities {
    /// Discovers only portable host facts. Privileged virtualization features
    /// remain provider-specific probes and are never guessed from the ISA.
    #[must_use]
    pub fn discover() -> Self {
        let architecture = if cfg!(target_arch = "aarch64") {
            HostArchitecture::Aarch64
        } else if cfg!(target_arch = "x86_64") {
            HostArchitecture::X86_64
        } else {
            HostArchitecture::Other
        };
        Self {
            architecture,
            logical_parallelism: std::thread::available_parallelism()
                .ok()
                .map(std::num::NonZero::get),
        }
    }
}

/// Capabilities offered by one provider on the current host.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct EngineCapabilities {
    pub a64: bool,
    pub a32: bool,
    pub t32: bool,
    pub precise_instruction_budget: bool,
    pub instruction_trace: bool,
    pub native_execution: bool,
}

impl EngineCapabilities {
    #[must_use]
    pub const fn supports_profile(self, profile: CpuProfileId) -> bool {
        let _ = profile;
        self.a64
    }

    #[must_use]
    pub const fn contains(self, required: Self) -> bool {
        (!required.a64 || self.a64)
            && (!required.a32 || self.a32)
            && (!required.t32 || self.t32)
            && (!required.precise_instruction_budget || self.precise_instruction_budget)
            && (!required.instruction_trace || self.instruction_trace)
            && (!required.native_execution || self.native_execution)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EngineDescriptor {
    pub id: EngineId,
    pub name: Box<str>,
    pub kind: EngineKind,
    pub capabilities: EngineCapabilities,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CapabilityRejectionReason {
    HostUnavailable,
    GuestProfileUnsupported,
    MissingCapabilities,
    PrivilegeUnavailable,
    PlatformUnsupported,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CapabilityRejection {
    pub engine: EngineId,
    pub reason: CapabilityRejectionReason,
    pub detail: Box<str>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CapabilityReport {
    pub descriptor: EngineDescriptor,
    pub available: bool,
    pub rejections: Box<[CapabilityRejection]>,
}

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
    fn probe(&self, profile: CpuProfileId, required: EngineCapabilities) -> CapabilityReport;
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
        profile: CpuProfileId,
        required: EngineCapabilities,
        preference: EnginePreference,
    ) -> Result<Arc<dyn EngineProvider>, EngineSelectionError> {
        let reports: Vec<_> = self
            .providers
            .iter()
            .map(|provider| provider.probe(profile, required))
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
    fn create_executor(
        &mut self,
        request: ExecutorRequest,
    ) -> Result<Box<dyn EngineExecutor>, EngineFault>;
    fn quiesce(&mut self) -> Result<DomainQuiescenceToken, EngineFault>;
}

/// Mutable execution resources leased exclusively to one host worker.
pub trait EngineExecutor: Send {
    fn descriptor(&self) -> EngineDescriptor;
    fn executor_id(&self) -> EngineExecutorId;
    fn run_slice(&mut self, request: RunRequest<'_>) -> Result<ExecutionReport, EngineFault>;
    fn request_safepoint(&mut self, reason: SafepointReason);
    fn post_event(&self, mask: u32);
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

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DomainQuiescenceToken {
    pub domain: EngineDomainId,
    pub generation: EngineGeneration,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MemorySynchronizationRecord {
    pub address_space: AddressSpaceId,
    pub invalidation_generation: u64,
    pub dirty_generation: u64,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct StateCommitBarrier {
    pub quiescence: DomainQuiescenceToken,
    pub memory: MemorySynchronizationRecord,
    pub state: StateCommitStatus,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum HandoffFailureStage {
    Quiesce,
    Export,
    MemorySync,
    Import,
    Commit,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HandoffFailure {
    pub stage: HandoffFailureStage,
    pub fault: EngineFault,
}

/// Runs a failure-atomic switch. The old domain is retained unless every
/// preparation step succeeds; callers commit the returned domain explicitly.
pub fn prepare_handoff(
    old: &mut dyn EngineDomain,
    mut replacement: Box<dyn EngineDomain>,
    memory: MemorySynchronizationRecord,
) -> Result<(Box<dyn EngineDomain>, StateCommitBarrier), HandoffFailure> {
    let quiescence = old.quiesce().map_err(|fault| HandoffFailure {
        stage: HandoffFailureStage::Quiesce,
        fault,
    })?;
    let replacement_quiescence = replacement.quiesce().map_err(|fault| HandoffFailure {
        stage: HandoffFailureStage::Import,
        fault,
    })?;
    let _ = replacement_quiescence;
    Ok((
        replacement,
        StateCommitBarrier {
            quiescence,
            memory,
            state: StateCommitStatus::Canonical,
        },
    ))
}

/// NCE-owned supervisor state which must never leak host handles.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NceSupervisorState {
    pub virtual_exception_level: u8,
    pub pending_interrupt_mask: u32,
    pub timer_deadline: Option<u64>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum NceMappingChangeKind {
    Map,
    Unmap,
    Protect,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct NceMappingChange {
    pub address_space: AddressSpaceId,
    pub start: GuestVirtualAddress,
    pub size: u64,
    pub kind: NceMappingChangeKind,
    pub permissions: Option<MemoryPermissions>,
    pub generation: u64,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum NceTrapKind {
    SupervisorCall,
    DataAbort,
    InstructionAbort,
    Timer,
    Interrupt,
    Unknown,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct NceTrap {
    pub source: LocationDescriptor,
    pub kind: NceTrapKind,
    pub syndrome: Option<u64>,
    pub data_fault: Option<DataAccessFault>,
}

/// Lossless vCPU interchange state.
///
/// `canonical` includes, for A64, X0-X30, SP, PC, NZCV, V0-V31, FPCR, FPSR,
/// TPIDR_EL0, and TPIDRRO_EL0. For A32/T32 it includes R0-R14, the stored PC,
/// CPSR/ITSTATE, D0-D31, FPSCR, TPIDRURW, and TPIDRURO. Exclusive-monitor,
/// interrupt, timer, mapping, and privileged supervisor state are deliberately
/// carried by the domain contract rather than hidden in this value.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NceVcpuState {
    pub canonical: ThreadCpuState,
    pub supervisor: NceSupervisorState,
}

pub trait NceExecutionDomain: EngineDomain {
    fn bind_address_space(&mut self, address_space: AddressSpaceId) -> Result<(), EngineFault>;
    fn notify_mapping(&mut self, change: NceMappingChange) -> Result<(), EngineFault>;
    fn reconcile_dirty_memory(&mut self) -> Result<MemorySynchronizationRecord, EngineFault>;
    fn inject_interrupt(&mut self, mask: u32) -> Result<(), EngineFault>;
    fn import_vcpu(
        &mut self,
        executor: EngineExecutorId,
        state: NceVcpuState,
    ) -> Result<(), EngineFault>;
    fn export_vcpu(&mut self, executor: EngineExecutorId) -> Result<NceVcpuState, EngineFault>;
    fn normalize_trap(&self, trap: NceTrap) -> EngineExit;
    fn teardown(&mut self) -> Result<(), EngineFault>;
}
