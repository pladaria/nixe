//! Engine provider, process domain, and executor-local native state.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, OnceLock};

use cranelift_codegen::isa::OwnedTargetIsa;
use cranelift_codegen::settings::{self, Configurable};
use nixe_cpu::coverage::CoverageId;
use nixe_cpu::error::FrontendError;
use nixe_cpu::exclusive::ExclusiveMonitorState;
use nixe_cpu::location::{ExecutionState, LocationDescriptor};
use nixe_cpu::profile::{GuestCpuProfile, ProcessCpuContext};
use nixe_cpu::state::ThreadCpuState;
use nixe_cpu::translate::translate_region;
use nixe_cpu_engine::{
    CapabilityRejection, CapabilityRejectionReason, CapabilityReport, CrossVcpuRequest,
    DomainMemoryBinding, DomainRequest, EngineCapabilities, EngineControl, EngineDescriptor,
    EngineDomain, EngineDomainId, EngineExecutor, EngineExecutorId, EngineFault, EngineFaultKind,
    EngineId, EngineKind, EngineProvider, ExecutionReport, RunRequest, SchedulerRequest,
};
use nixe_memory::GuestVirtualAddress;

use crate::abi::{
    EXIT_ARCHITECTURAL, EXIT_BUDGET_EXHAUSTED, EXIT_DATA_FAULT, EXIT_DISPATCH, EXIT_INTERPRET_ONE,
    EXIT_LOADER_RETURN, EXIT_NONE, EXIT_PENDING_EVENT, EXIT_SAFEPOINT, EXIT_SCHEDULED,
    EXIT_UNSUPPORTED, ExecutionFrame, FrameError, NativeExit, NativeGateway, SCHEDULE_SEND_EVENT,
    SCHEDULE_WAIT_FOR_EVENT, SCHEDULE_WAIT_FOR_INTERRUPT, SCHEDULE_YIELD,
};
use crate::cache::{
    CacheError, DomainCodeCache, ExecutorEpoch, LocalLookupCache, PendingRegion, RegionKey,
    TranslationMode, root_code_mapping,
};
use crate::compiler::{CompiledRegionMetadata, CompilerContext, SideExit, compile_gateway};
use crate::configuration::JitConfiguration;
use crate::executable_memory::{
    ExecutableMemoryError, PublishedCode, SharedExecutableMemory, process_executable_memory,
};
use crate::helpers::{HELPER_TABLE, NativeContext};
use crate::tlb::SoftwareTlb;

pub const JIT_ENGINE_ID: EngineId = EngineId::new(2);

const CONTROL_PREEMPT: u32 = 1 << 0;
const CONTROL_CODE_INVALIDATION: u32 = 1 << 1;

enum HostSupport {
    Available(OwnedTargetIsa),
    Unavailable {
        reason: CapabilityRejectionReason,
        detail: Box<str>,
    },
}

/// Cranelift JIT provider and owner of the process-wide executable arena.
pub struct JitProvider {
    host: OnceLock<HostSupport>,
    executable_memory: OnceLock<Result<SharedExecutableMemory, ExecutableMemoryError>>,
    configuration: JitConfiguration,
}

impl JitProvider {
    #[must_use]
    pub fn new() -> Self {
        Self::with_configuration(JitConfiguration::default())
    }

    /// Creates a provider with validated domain-local resource budgets.
    #[must_use]
    pub fn with_configuration(configuration: JitConfiguration) -> Self {
        Self {
            host: OnceLock::new(),
            executable_memory: OnceLock::new(),
            configuration,
        }
    }

    #[cfg(test)]
    fn with_executable_error(error: ExecutableMemoryError) -> Self {
        let provider = Self {
            host: OnceLock::new(),
            executable_memory: OnceLock::new(),
            configuration: JitConfiguration::default(),
        };
        assert!(
            provider.executable_memory.set(Err(error)).is_ok(),
            "test provider executable-memory state is initially empty"
        );
        provider
    }

    fn executable_memory(&self) -> &Result<SharedExecutableMemory, ExecutableMemoryError> {
        self.executable_memory
            .get_or_init(process_executable_memory)
    }

    fn host(&self) -> &HostSupport {
        self.host.get_or_init(probe_host)
    }

    fn availability_rejections(
        &self,
        profile: GuestCpuProfile,
        required: EngineCapabilities,
    ) -> Vec<CapabilityRejection> {
        let capabilities = capabilities();
        let mut rejections = Vec::new();
        if profile.id() != GuestCpuProfile::SWITCH_1_ID
            || !capabilities.supports_profile(profile, required)
        {
            rejections.push(rejection(
                CapabilityRejectionReason::GuestProfileUnsupported,
                "JIT supports only the verified Switch 1 CPU profile",
            ));
        }
        if !capabilities.contains(required) {
            rejections.push(rejection(
                CapabilityRejectionReason::MissingCapabilities,
                "required capability set is unavailable",
            ));
        }
        if let HostSupport::Unavailable { reason, detail } = self.host() {
            rejections.push(CapabilityRejection {
                engine: JIT_ENGINE_ID,
                reason: *reason,
                detail: detail.clone(),
            });
        }
        if rejections.is_empty()
            && let Err(error) = self.executable_memory()
        {
            rejections.push(CapabilityRejection {
                engine: JIT_ENGINE_ID,
                reason: error.rejection_reason(),
                detail: error.detail().into(),
            });
        }
        rejections
    }

    fn available_resources(
        &self,
        cpu: ProcessCpuContext,
    ) -> Result<(OwnedTargetIsa, SharedExecutableMemory), EngineFault> {
        let rejections = self.availability_rejections(cpu.profile(), EngineCapabilities::default());
        if !rejections.is_empty() {
            return Err(fault(
                EngineFaultKind::Unavailable,
                0,
                format!(
                    "JIT domain creation rejected: {}",
                    rejections
                        .iter()
                        .map(|rejection| rejection.detail.as_ref())
                        .collect::<Vec<_>>()
                        .join("; ")
                ),
                &ThreadCpuState::new(
                    cpu.thread_configuration(first_execution_state(cpu.profile()))
                        .expect("the selected state belongs to the profile"),
                ),
            ));
        }
        let isa = match self.host() {
            HostSupport::Available(isa) => Arc::clone(isa),
            HostSupport::Unavailable { .. } => unreachable!("rejections handled host failure"),
        };
        let executable_memory = self
            .executable_memory()
            .as_ref()
            .expect("rejections handled executable-memory failure")
            .clone();
        Ok((isa, executable_memory))
    }
}

impl Default for JitProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl EngineProvider for JitProvider {
    fn descriptor(&self) -> EngineDescriptor {
        descriptor()
    }

    fn probe(&self, profile: GuestCpuProfile, required: EngineCapabilities) -> CapabilityReport {
        let rejections = self.availability_rejections(profile, required);
        CapabilityReport {
            descriptor: descriptor(),
            available: rejections.is_empty(),
            rejections: rejections.into_boxed_slice(),
        }
    }

    fn create_domain(&self, request: DomainRequest) -> Result<Box<dyn EngineDomain>, EngineFault> {
        let (isa, executable_memory) = self.available_resources(request.cpu)?;
        let (gateway, gateway_code) =
            compile_gateway(&isa, &executable_memory).map_err(|error| {
                domain_fault(
                    request.cpu,
                    EngineFaultKind::Internal,
                    format!("JIT native gateway failed: {}", error.detail()),
                )
            })?;
        Ok(Box::new(JitDomain {
            id: request.domain,
            cpu: request.cpu,
            isa,
            executable_memory: Some(executable_memory),
            gateway: Some(gateway),
            gateway_code: Some(gateway_code),
            code_cache: Some(Arc::new(DomainCodeCache::new(self.configuration))),
            controls: Vec::new(),
            binding: None,
            stopping: false,
            shutdown: false,
        }))
    }
}

#[derive(Clone, Copy)]
struct BoundMemory {
    end_exclusive: GuestVirtualAddress,
    mapping_epoch: u64,
    invalidation_cursor: nixe_memory::MemoryInvalidationCursor,
}

struct JitDomain {
    id: EngineDomainId,
    cpu: ProcessCpuContext,
    isa: OwnedTargetIsa,
    // The provider owns the process-wide arena; each live domain retains the
    // owner so code can never outlive its OS mapping.
    executable_memory: Option<SharedExecutableMemory>,
    gateway: Option<NativeGateway>,
    gateway_code: Option<PublishedCode>,
    code_cache: Option<Arc<DomainCodeCache>>,
    controls: Vec<EngineControl>,
    binding: Option<BoundMemory>,
    stopping: bool,
    shutdown: bool,
}

impl EngineDomain for JitDomain {
    fn descriptor(&self) -> EngineDescriptor {
        descriptor()
    }

    fn domain_id(&self) -> EngineDomainId {
        self.id
    }

    fn create_executor(
        &mut self,
        executor: EngineExecutorId,
    ) -> Result<Box<dyn EngineExecutor>, EngineFault> {
        if self.stopping || self.shutdown {
            return Err(domain_fault(
                self.cpu,
                EngineFaultKind::Unavailable,
                "JIT domain is shut down",
            ));
        }
        if self.executable_memory.is_none() {
            return Err(domain_fault(
                self.cpu,
                EngineFaultKind::Unavailable,
                "JIT executable-memory owner is unavailable",
            ));
        }
        let Some(binding) = self.binding else {
            return Err(domain_fault(
                self.cpu,
                EngineFaultKind::InvalidRequest,
                "canonical memory must be bound before creating a JIT executor",
            ));
        };
        let code_cache = self
            .code_cache
            .as_ref()
            .expect("live domain retains its code cache")
            .clone();
        let executor_epoch = code_cache.register_executor();
        let control = EngineControl::default();
        self.controls.push(control.clone());
        Ok(Box::new(JitExecutor {
            id: executor,
            cpu: self.cpu,
            address_space_end: binding.end_exclusive,
            frame: ExecutionFrame::default(),
            compiler: CompilerContext::new(Arc::clone(&self.isa)),
            executable_memory: self
                .executable_memory
                .as_ref()
                .expect("live domain retains executable memory")
                .clone(),
            gateway: self
                .gateway
                .expect("live domain retains its native gateway"),
            _gateway_code: self
                .gateway_code
                .as_ref()
                .expect("live domain retains its native gateway publication")
                .clone(),
            code_cache,
            executor_epoch,
            local_lookup: LocalLookupCache::new(),
            control,
            exclusive_monitor: ExclusiveMonitorState::default(),
            tlb: SoftwareTlb::new(),
            mapping_epoch: binding.mapping_epoch,
            invalidation_cursor: binding.invalidation_cursor,
        }))
    }

    fn bind_memory(&mut self, binding: DomainMemoryBinding<'_>) -> Result<(), EngineFault> {
        if self.stopping || self.shutdown {
            return Err(domain_fault(
                self.cpu,
                EngineFaultKind::Unavailable,
                "cannot bind memory after JIT domain stop was requested",
            ));
        }
        if binding.address_space != self.cpu.address_space_id() {
            return Err(domain_fault(
                self.cpu,
                EngineFaultKind::InvalidRequest,
                "canonical memory binding belongs to a different address space",
            ));
        }
        self.binding = Some(BoundMemory {
            end_exclusive: binding.end_exclusive,
            mapping_epoch: binding.mapping_epoch,
            invalidation_cursor: binding.invalidation_cursor,
        });
        Ok(())
    }

    fn request_stop(&mut self) -> Result<(), EngineFault> {
        if self.stopping || self.shutdown {
            return Ok(());
        }
        if let Some(cache) = &self.code_cache {
            cache.begin_shutdown().map_err(|error| {
                domain_fault(
                    self.cpu,
                    EngineFaultKind::Unavailable,
                    cache_error_detail(error),
                )
            })?;
        }
        for control in &self.controls {
            control.request(CrossVcpuRequest::Preempt);
        }
        self.stopping = true;
        Ok(())
    }

    fn shutdown(&mut self) -> Result<(), EngineFault> {
        self.request_stop()?;
        if self
            .code_cache
            .as_ref()
            .is_some_and(|cache| cache.live_executor_count() != 0)
        {
            return Err(domain_fault(
                self.cpu,
                EngineFaultKind::Unavailable,
                "JIT domain shutdown requires every executor to be prepared and dropped",
            ));
        }
        self.shutdown = true;
        self.binding = None;
        self.executable_memory = None;
        self.gateway = None;
        self.gateway_code = None;
        self.code_cache = None;
        self.controls.clear();
        Ok(())
    }
}

struct JitExecutor {
    id: EngineExecutorId,
    cpu: ProcessCpuContext,
    address_space_end: GuestVirtualAddress,
    frame: ExecutionFrame,
    compiler: CompilerContext,
    // Native execution retains the process arena for the executor's complete
    // lifetime; runtime drops every executor before releasing its domain.
    executable_memory: SharedExecutableMemory,
    gateway: NativeGateway,
    _gateway_code: PublishedCode,
    code_cache: Arc<DomainCodeCache>,
    executor_epoch: ExecutorEpoch,
    local_lookup: LocalLookupCache,
    control: EngineControl,
    exclusive_monitor: ExclusiveMonitorState,
    tlb: SoftwareTlb,
    mapping_epoch: u64,
    invalidation_cursor: nixe_memory::MemoryInvalidationCursor,
}

impl EngineExecutor for JitExecutor {
    fn descriptor(&self) -> EngineDescriptor {
        descriptor()
    }

    fn executor_id(&self) -> EngineExecutorId {
        self.id
    }

    fn run_slice(&mut self, request: RunRequest<'_>) -> Result<ExecutionReport, EngineFault> {
        if request.cpu != self.cpu {
            return Err(fault(
                EngineFaultKind::InvalidRequest,
                0,
                "run request CPU context differs from the JIT domain",
                request.state,
            ));
        }
        let current_pc = current_pc(request.state);
        if current_pc >= self.address_space_end.get() {
            return Err(fault(
                EngineFaultKind::InvalidRequest,
                0,
                "canonical PC lies outside the bound address space",
                request.state,
            ));
        }

        self.frame.import_state(request.state);
        self.frame.memory = self
            .tlb
            .acceleration(request.cpu.address_space_id(), self.mapping_epoch);
        self.frame.control.instruction_budget = request.instruction_budget;
        self.frame.control.invalidation_epoch = self.invalidation_cursor.get();
        self.frame.control.loader_return_valid = u32::from(request.loader_return.is_some());
        self.frame.control.loader_return =
            request.loader_return.map_or(0, GuestVirtualAddress::get);

        let pending = self.control.take_pending();
        self.frame.control.event_mask = request.events.take_pending_interrupts();
        if let Some(snapshot) = pending {
            self.frame.control.request_flags =
                (u32::from(snapshot.contains(CrossVcpuRequest::Preempt)) * CONTROL_PREEMPT)
                    | (u32::from(snapshot.contains(CrossVcpuRequest::CodeInvalidation))
                        * CONTROL_CODE_INVALIDATION);
            self.frame.control.invalidation_epoch = snapshot.invalidation_epoch;
        } else {
            self.frame.control.request_flags = 0;
        }

        enum DispatchResult {
            Native {
                exit: NativeExit,
                region: Option<Arc<crate::cache::CachedRegion>>,
                data_fault: Option<nixe_cpu::memory::DataAccessFault>,
                pending: Option<nixe_cpu_engine::ControlSnapshot>,
            },
            Frontend(FrontendError),
            Fault(EngineFaultKind, Box<str>),
            Panicked,
        }

        let translation_mode = TranslationMode::Baseline;
        let mut instructions_executed = 0_u64;
        let mut pending_link = None;
        let dispatch = {
            loop {
                if let Err(error) = self.consume_invalidations(
                    request.memory,
                    request.memory.invalidation_cursor(),
                    request.state,
                ) {
                    break DispatchResult::Fault(error.kind, error.message);
                }
                self.frame.control.instruction_budget = request
                    .instruction_budget
                    .checked_sub(instructions_executed)
                    .expect("link resolver never exceeds the requested instruction budget");
                if let Some(mut exit) = self.pre_entry_exit() {
                    exit.instructions_executed = instructions_executed;
                    break DispatchResult::Native {
                        exit,
                        region: None,
                        data_fault: None,
                        pending: None,
                    };
                }

                let location = LocationDescriptor::new(
                    GuestVirtualAddress::new(self.frame.current_pc()),
                    self.frame
                        .execution_state()
                        .expect("imported native frame has a valid execution state"),
                    request.cpu.profile().id(),
                );
                let root = match root_code_mapping(
                    request.memory,
                    request.cpu.address_space_id(),
                    location,
                ) {
                    Ok(root) => root,
                    Err(error) => break DispatchResult::Frontend(error),
                };
                let key = RegionKey::new(
                    request.cpu.address_space_id(),
                    location,
                    translation_mode,
                    root,
                );
                let cached = if let Some(region) = self.local_lookup.lookup(key) {
                    Ok(region)
                } else {
                    let cache = Arc::clone(&self.code_cache);
                    let compile_cursor = request.memory.invalidation_cursor();
                    let result = cache.resolve(key, |cancellation| {
                        cancellation.check()?;
                        let region = translate_region(
                            translation_mode.config(),
                            &request.cpu.profile(),
                            request.cpu.address_space_id(),
                            location,
                            request.memory,
                        )?;
                        cancellation.check()?;
                        if request.memory.invalidation_cursor() != compile_cursor {
                            return Err(CacheError::Stale);
                        }
                        let compiled = self.compiler.compile(
                            &region,
                            &self.executable_memory,
                            cancellation,
                        )?;
                        cancellation.check()?;
                        if request.memory.invalidation_cursor() != compile_cursor {
                            return Err(CacheError::Stale);
                        }
                        PendingRegion::new(
                            request.cpu.address_space_id(),
                            translation_mode,
                            &region,
                            compiled,
                        )
                    });
                    if let Ok(region) = &result {
                        self.local_lookup.insert(key, Arc::clone(region));
                    }
                    result
                };
                let cached = match cached {
                    Ok(cached) => cached,
                    Err(CacheError::Stale) => continue,
                    Err(CacheError::Cancelled) => {
                        break DispatchResult::Fault(
                            EngineFaultKind::Unavailable,
                            "JIT domain stopped while compilation was in progress".into(),
                        );
                    }
                    Err(CacheError::Frontend(error)) => break DispatchResult::Frontend(error),
                    Err(CacheError::Compiler(error)) => {
                        break DispatchResult::Fault(
                            EngineFaultKind::Internal,
                            format!("JIT lowering failed: {}", error.detail()).into_boxed_str(),
                        );
                    }
                    Err(CacheError::Capacity(detail)) => {
                        break DispatchResult::Fault(EngineFaultKind::Unavailable, detail);
                    }
                    Err(CacheError::Internal(detail)) => {
                        break DispatchResult::Fault(EngineFaultKind::Internal, detail);
                    }
                };

                if let Some((source, site, linked_location)) = pending_link.take() {
                    debug_assert_eq!(linked_location, location);
                    match self.code_cache.link(source, site, location, &cached) {
                        Ok(()) => {}
                        Err(CacheError::Stale) => continue,
                        Err(error) => {
                            break DispatchResult::Fault(
                                EngineFaultKind::Internal,
                                cache_error_detail(error),
                            );
                        }
                    }
                }

                let _native_epoch =
                    match self.code_cache.begin_native(&cached, &self.executor_epoch) {
                        Ok(guard) => guard,
                        Err(CacheError::Stale) => continue,
                        Err(error) => {
                            break DispatchResult::Fault(
                                EngineFaultKind::Internal,
                                cache_error_detail(error),
                            );
                        }
                    };
                let compiled = cached.compiled();
                let mut native_context = NativeContext::new(
                    request.memory,
                    &mut self.exclusive_monitor,
                    &mut self.tlb,
                    &self.control,
                    request.cpu,
                    request.timer,
                    &request.events,
                );
                cached.install_dispatch(&mut self.frame);
                self.frame.install_host_context(
                    &HELPER_TABLE,
                    std::ptr::from_mut(&mut native_context).cast(),
                );
                let native_exit = contain_rust_boundary(|| {
                    // SAFETY: the live cached region and imported frame remain
                    // valid for this complete non-unwinding native call.
                    unsafe { compiled.execute(self.gateway, &raw mut self.frame) };
                    self.frame.exit
                });
                self.frame.clear_host_context();
                let data_fault = native_context.data_fault.take();
                let native_pending = native_context.control_snapshot.take();
                drop(native_context);
                let mut native_exit = match native_exit {
                    Ok(exit) => exit,
                    Err(()) => break DispatchResult::Panicked,
                };
                instructions_executed =
                    match instructions_executed.checked_add(native_exit.instructions_executed) {
                        Some(total) if total <= request.instruction_budget => total,
                        _ => {
                            break DispatchResult::Fault(
                                EngineFaultKind::Internal,
                                "native region exceeded the slice instruction budget".into(),
                            );
                        }
                    };
                native_exit.instructions_executed = instructions_executed;
                if native_exit.kind == EXIT_DISPATCH {
                    if data_fault.is_some() || native_pending.is_some() {
                        break DispatchResult::Fault(
                            EngineFaultKind::Internal,
                            "ordinary region dispatch retained incompatible slow-path state".into(),
                        );
                    }
                    let location = LocationDescriptor::new(
                        GuestVirtualAddress::new(self.frame.current_pc()),
                        self.frame
                            .execution_state()
                            .expect("linked native state has a valid execution state"),
                        request.cpu.profile().id(),
                    );
                    pending_link =
                        Some((self.frame.dispatch.region_id, native_exit.detail, location));
                    continue;
                }
                break DispatchResult::Native {
                    exit: native_exit,
                    region: self
                        .code_cache
                        .region_for_exit(self.frame.dispatch.region_id),
                    data_fault,
                    pending: native_pending,
                };
            }
        };

        self.commit_or_fault(request.state, instructions_executed)?;
        self.acknowledge_snapshot(pending);
        match dispatch {
            DispatchResult::Native {
                exit,
                region,
                data_fault,
                pending,
            } => {
                self.acknowledge_snapshot(pending);
                Ok(ExecutionReport {
                    instructions_executed: exit.instructions_executed,
                    stop: normalize_exit(
                        exit,
                        region.as_ref().map(|region| &region.compiled().metadata),
                        data_fault,
                        request.cpu,
                        request.state,
                    )?,
                    context: request.state.register_context(),
                })
            }
            DispatchResult::Frontend(error) => Ok(ExecutionReport {
                instructions_executed,
                stop: normalize_frontend_error(error, instructions_executed, request.state)?,
                context: request.state.register_context(),
            }),
            DispatchResult::Fault(kind, message) => {
                Err(fault(kind, instructions_executed, message, request.state))
            }
            DispatchResult::Panicked => Err(fault(
                EngineFaultKind::Internal,
                instructions_executed,
                "panic was contained at the JIT native-entry boundary",
                request.state,
            )),
        }
    }

    fn synchronize_invalidation(
        &mut self,
        cursor: nixe_memory::MemoryInvalidationCursor,
        state: &ThreadCpuState,
        memory: &dyn nixe_cpu::memory::CpuMemory,
    ) -> Result<(), EngineFault> {
        self.consume_invalidations(memory, cursor, state)
    }

    fn synchronize_address_space(
        &mut self,
        binding: DomainMemoryBinding<'_>,
        state: &ThreadCpuState,
    ) -> Result<(), EngineFault> {
        if binding.address_space != self.cpu.address_space_id() {
            return Err(fault(
                EngineFaultKind::InvalidRequest,
                0,
                "address-space synchronization belongs to a different domain",
                state,
            ));
        }
        self.address_space_end = binding.end_exclusive;
        let mapping_changed = self.mapping_epoch != binding.mapping_epoch;
        self.synchronize_invalidation(binding.invalidation_cursor, state, binding.memory)?;
        if mapping_changed {
            self.tlb
                .advance_mapping_epoch(binding.address_space, binding.mapping_epoch);
            self.mapping_epoch = binding.mapping_epoch;
            self.frame.memory.mapping_epoch = self.mapping_epoch;
        }
        Ok(())
    }

    fn control(&self) -> Option<EngineControl> {
        Some(self.control.clone())
    }

    fn prepare_shutdown(
        &mut self,
        binding: DomainMemoryBinding<'_>,
        state: &ThreadCpuState,
    ) -> Result<(), EngineFault> {
        if binding.address_space != self.cpu.address_space_id() {
            return Err(fault(
                EngineFaultKind::InvalidRequest,
                0,
                "JIT teardown binding belongs to a different address space",
                state,
            ));
        }
        self.local_lookup.clear();
        self.tlb.clear();
        self.exclusive_monitor.clear();
        self.invalidation_cursor = binding.invalidation_cursor;
        self.mapping_epoch = binding.mapping_epoch;
        self.frame.memory.mapping_epoch = binding.mapping_epoch;
        self.frame.control.invalidation_epoch = binding.invalidation_cursor.get();
        self.control
            .acknowledge_invalidation(binding.invalidation_cursor.get());
        Ok(())
    }

    fn clear_local_exclusive_reservation(&mut self) {
        self.exclusive_monitor.clear();
    }
}

impl JitExecutor {
    fn consume_invalidations(
        &mut self,
        memory: &dyn nixe_cpu::memory::CpuMemory,
        requested: nixe_memory::MemoryInvalidationCursor,
        state: &ThreadCpuState,
    ) -> Result<(), EngineFault> {
        if requested <= self.invalidation_cursor {
            self.control.acknowledge_invalidation(requested.get());
            return Ok(());
        }
        let mut records = Vec::new();
        let (through, history_lost) =
            match memory.read_invalidations_since(self.invalidation_cursor, &mut records) {
                Ok(through) => (through, false),
                Err(nixe_memory::MemoryInvalidationError::HistoryLost { latest, .. }) => {
                    records.clear();
                    (latest, true)
                }
                Err(error) => {
                    return Err(fault(
                        EngineFaultKind::Unavailable,
                        0,
                        error.to_string(),
                        state,
                    ));
                }
            };
        self.code_cache
            .apply_invalidations(&records, through, history_lost)
            .map_err(|error| {
                fault(
                    EngineFaultKind::Unavailable,
                    0,
                    cache_error_detail(error),
                    state,
                )
            })?;
        if history_lost {
            self.local_lookup.clear();
            self.tlb.clear();
        } else {
            self.tlb.apply_invalidations(&records);
        }
        self.invalidation_cursor = through;
        self.frame.control.invalidation_epoch = through.get();
        self.control.acknowledge_invalidation(through.get());
        Ok(())
    }

    fn acknowledge_snapshot(&mut self, snapshot: Option<nixe_cpu_engine::ControlSnapshot>) {
        let Some(snapshot) = snapshot else {
            return;
        };
        if snapshot.contains(CrossVcpuRequest::CodeInvalidation) {
            self.invalidation_cursor =
                self.invalidation_cursor
                    .max(nixe_memory::MemoryInvalidationCursor::new(
                        snapshot.invalidation_epoch,
                    ));
            self.frame.control.invalidation_epoch = self.invalidation_cursor.get();
        }
        self.control.acknowledge(snapshot);
    }

    fn pre_entry_exit(&mut self) -> Option<NativeExit> {
        self.frame
            .execution_state()
            .expect("imported native frame has a valid execution state");
        let source_pc = self.frame.current_pc();
        if self.frame.control.event_mask != 0 {
            return Some(NativeExit {
                kind: EXIT_PENDING_EVENT,
                detail: self.frame.control.event_mask,
                source_pc,
                ..NativeExit::default()
            });
        }
        if self.frame.control.request_flags & (CONTROL_PREEMPT | CONTROL_CODE_INVALIDATION) != 0 {
            return Some(NativeExit {
                kind: EXIT_SAFEPOINT,
                source_pc,
                ..NativeExit::default()
            });
        }
        if self.frame.control.instruction_budget == 0 {
            return Some(NativeExit {
                kind: EXIT_BUDGET_EXHAUSTED,
                source_pc,
                ..NativeExit::default()
            });
        }
        if self.frame.control.loader_return_valid != 0
            && source_pc == self.frame.control.loader_return
            && let Some(result_code) = self.frame.a64_result_code()
        {
            return Some(NativeExit {
                kind: EXIT_LOADER_RETURN,
                source_pc,
                payload0: result_code,
                ..NativeExit::default()
            });
        }
        None
    }

    fn commit_or_fault(
        &self,
        state: &mut ThreadCpuState,
        instructions_executed: u64,
    ) -> Result<(), EngineFault> {
        self.frame.commit_state(state).map_err(|error| {
            fault(
                EngineFaultKind::Internal,
                instructions_executed,
                match error {
                    FrameError::StateKindChanged => {
                        "native frame attempted to change the canonical state representation"
                    }
                    FrameError::InconsistentA32ExecutionState => {
                        "native frame AArch32 state disagrees with CPSR.T"
                    }
                    FrameError::InvalidA32InstructionAddress => {
                        "native frame contained an invalid AArch32 instruction address"
                    }
                },
                state,
            )
        })
    }
}

fn normalize_exit(
    exit: NativeExit,
    metadata: Option<&CompiledRegionMetadata>,
    data_fault: Option<nixe_cpu::memory::DataAccessFault>,
    cpu: ProcessCpuContext,
    state: &ThreadCpuState,
) -> Result<nixe_cpu_engine::EngineExit, EngineFault> {
    let execution_state = state.execution_state();
    let source = LocationDescriptor::new(
        GuestVirtualAddress::new(exit.source_pc),
        execution_state,
        cpu.profile().id(),
    );
    match exit.kind {
        EXIT_INTERPRET_ONE => Ok(nixe_cpu_engine::EngineExit::InterpretOne { source }),
        EXIT_DISPATCH => Err(fault(
            EngineFaultKind::Internal,
            exit.instructions_executed,
            "native link miss escaped the JIT resolver",
            state,
        )),
        EXIT_ARCHITECTURAL => {
            let (metadata, side_exit) =
                side_exit(metadata, exit.detail, state, exit.instructions_executed)?;
            match side_exit {
                SideExit::Architectural {
                    source,
                    kind,
                    syndrome,
                } => {
                    let source =
                        compact_source(metadata, *source, state, exit.instructions_executed)?;
                    if *kind == nixe_cpu::exception::ExceptionKind::SupervisorCall {
                        let immediate = syndrome
                            .and_then(|value| u32::try_from(value).ok())
                            .ok_or_else(|| {
                                fault(
                                    EngineFaultKind::Internal,
                                    exit.instructions_executed,
                                    "supervisor-call exit has an invalid immediate",
                                    state,
                                )
                            })?;
                        Ok(nixe_cpu_engine::EngineExit::SupervisorCall { source, immediate })
                    } else {
                        Ok(nixe_cpu_engine::EngineExit::ArchitecturalException {
                            source,
                            kind: *kind,
                            syndrome: *syndrome,
                        })
                    }
                }
                _ => Err(fault(
                    EngineFaultKind::Internal,
                    exit.instructions_executed,
                    "architectural exit references incompatible side metadata",
                    state,
                )),
            }
        }
        EXIT_UNSUPPORTED => {
            let (metadata, side_exit) =
                side_exit(metadata, exit.detail, state, exit.instructions_executed)?;
            match side_exit {
                SideExit::Unsupported {
                    source,
                    encoding,
                    coverage_id,
                    disassembly,
                    ..
                } => Ok(nixe_cpu_engine::EngineExit::UnsupportedSemantics {
                    source: compact_source(metadata, *source, state, exit.instructions_executed)?,
                    encoding: *encoding,
                    disassembly: disassembly.clone(),
                    coverage_id: CoverageId::new(*coverage_id),
                }),
                _ => Err(fault(
                    EngineFaultKind::Internal,
                    exit.instructions_executed,
                    "unsupported exit references incompatible side metadata",
                    state,
                )),
            }
        }
        EXIT_DATA_FAULT => data_fault
            .map(|fault| nixe_cpu_engine::EngineExit::DataFault { source, fault })
            .ok_or_else(|| {
                fault(
                    EngineFaultKind::Internal,
                    exit.instructions_executed,
                    "memory helper exited without a precise data fault",
                    state,
                )
            }),
        EXIT_SCHEDULED => {
            let request = match exit.detail {
                SCHEDULE_YIELD => SchedulerRequest::Yield,
                SCHEDULE_WAIT_FOR_EVENT => SchedulerRequest::WaitForEvent,
                SCHEDULE_WAIT_FOR_INTERRUPT => SchedulerRequest::WaitForInterrupt,
                SCHEDULE_SEND_EVENT => SchedulerRequest::SendEvent,
                _ => {
                    return Err(fault(
                        EngineFaultKind::Internal,
                        exit.instructions_executed,
                        "native scheduling exit has an unknown request",
                        state,
                    ));
                }
            };
            Ok(nixe_cpu_engine::EngineExit::Scheduled { source, request })
        }
        EXIT_BUDGET_EXHAUSTED => Ok(nixe_cpu_engine::EngineExit::BudgetExhausted),
        EXIT_SAFEPOINT => Ok(nixe_cpu_engine::EngineExit::Safepoint),
        EXIT_PENDING_EVENT => Ok(nixe_cpu_engine::EngineExit::PendingEvent { mask: exit.detail }),
        EXIT_LOADER_RETURN => Ok(nixe_cpu_engine::EngineExit::LoaderReturn {
            source,
            result_code: exit.payload0,
        }),
        EXIT_NONE => Err(fault(
            EngineFaultKind::Internal,
            exit.instructions_executed,
            "native frame returned without a normalized exit",
            state,
        )),
        crate::abi::EXIT_INTERNAL => Err(fault(
            EngineFaultKind::Internal,
            exit.instructions_executed,
            "native helper reported an internal lowering failure",
            state,
        )),
        _ => Err(fault(
            EngineFaultKind::Internal,
            exit.instructions_executed,
            format!("native frame returned unknown exit kind {}", exit.kind),
            state,
        )),
    }
}

fn side_exit<'a>(
    metadata: Option<&'a CompiledRegionMetadata>,
    index: u32,
    state: &ThreadCpuState,
    instructions_executed: u64,
) -> Result<(&'a CompiledRegionMetadata, &'a SideExit), EngineFault> {
    metadata
        .and_then(|metadata| {
            metadata
                .side_exits
                .get(index as usize)
                .map(|exit| (metadata, exit))
        })
        .ok_or_else(|| {
            fault(
                EngineFaultKind::Internal,
                instructions_executed,
                "native exit references missing side metadata",
                state,
            )
        })
}

fn compact_source(
    metadata: &CompiledRegionMetadata,
    index: u32,
    state: &ThreadCpuState,
    instructions_executed: u64,
) -> Result<LocationDescriptor, EngineFault> {
    metadata
        .sources
        .get(index as usize)
        .copied()
        .ok_or_else(|| {
            fault(
                EngineFaultKind::Internal,
                instructions_executed,
                "native exit references missing compact source metadata",
                state,
            )
        })
}

fn normalize_frontend_error(
    error: FrontendError,
    instructions_executed: u64,
    state: &ThreadCpuState,
) -> Result<nixe_cpu_engine::EngineExit, EngineFault> {
    match error {
        FrontendError::InstructionFetch(fault) => {
            Ok(nixe_cpu_engine::EngineExit::FetchFault { fault })
        }
        FrontendError::ProfileDisabled(error) => {
            Ok(nixe_cpu_engine::EngineExit::ProfileDisabled { error })
        }
        FrontendError::Unallocated(error) => {
            Ok(nixe_cpu_engine::EngineExit::UnallocatedEncoding { error })
        }
        FrontendError::Decode(error) => Ok(nixe_cpu_engine::EngineExit::InterpretOne {
            source: error.instruction.location,
        }),
        FrontendError::InvalidIr(error) => Err(fault(
            EngineFaultKind::Internal,
            instructions_executed,
            format!("JIT frontend produced invalid IR: {error}"),
            state,
        )),
        FrontendError::Internal(error) => Err(fault(
            EngineFaultKind::Internal,
            instructions_executed,
            format!("JIT frontend failed internally: {error}"),
            state,
        )),
        _ => Err(fault(
            EngineFaultKind::Internal,
            instructions_executed,
            "JIT frontend returned an unknown failure",
            state,
        )),
    }
}

fn descriptor() -> EngineDescriptor {
    EngineDescriptor {
        id: JIT_ENGINE_ID,
        name: "cranelift-jit".into(),
        kind: EngineKind::Jit,
        capabilities: capabilities(),
    }
}

/// Contains every Rust-side operation adjacent to native entry. Published
/// functions and helper slots use `extern "C"`, so unwind is not an ABI outcome;
/// helpers must perform the same containment before returning to native code.
fn contain_rust_boundary<T>(operation: impl FnOnce() -> T) -> Result<T, ()> {
    catch_unwind(AssertUnwindSafe(operation)).map_err(|_| ())
}

const fn capabilities() -> EngineCapabilities {
    EngineCapabilities {
        a64: true,
        a32: true,
        t32: true,
        interpret_one_fallback: true,
        concurrent_executors: true,
        // Entry polls and explicit backward-edge polls bound a default region
        // to at most this many instructions before returning to the runtime.
        max_safepoint_instructions: std::num::NonZeroU64::new(
            nixe_cpu::translate::DEFAULT_MAX_GUEST_INSTRUCTIONS_PER_REGION.get() as u64,
        ),
        acknowledged_invalidation: true,
        deterministic_execution: true,
        canonical_memory_binding: true,
        max_concurrent_executors: None,
    }
}

fn probe_host() -> HostSupport {
    if !cfg!(any(target_arch = "x86_64", target_arch = "aarch64")) {
        return HostSupport::Unavailable {
            reason: CapabilityRejectionReason::PlatformUnsupported,
            detail: format!(
                "Cranelift JIT supports only x86-64 and AArch64 hosts, not {}",
                std::env::consts::ARCH
            )
            .into_boxed_str(),
        };
    }
    let builder = match cranelift_native::builder() {
        Ok(builder) => builder,
        Err(detail) => {
            return HostSupport::Unavailable {
                reason: CapabilityRejectionReason::HostUnavailable,
                detail: format!("Cranelift rejected the native host ISA: {detail}")
                    .into_boxed_str(),
            };
        }
    };
    let mut flag_builder = settings::builder();
    if let Err(error) = flag_builder.set("preserve_frame_pointers", "true") {
        return HostSupport::Unavailable {
            reason: CapabilityRejectionReason::HostUnavailable,
            detail: format!("Cranelift tail-call configuration failed: {error}").into_boxed_str(),
        };
    }
    let flags = settings::Flags::new(flag_builder);
    match builder.finish(flags) {
        Ok(isa) => HostSupport::Available(isa),
        Err(error) => HostSupport::Unavailable {
            reason: CapabilityRejectionReason::HostUnavailable,
            detail: format!("Cranelift native ISA configuration failed: {error}").into_boxed_str(),
        },
    }
}

fn rejection(reason: CapabilityRejectionReason, detail: &'static str) -> CapabilityRejection {
    CapabilityRejection {
        engine: JIT_ENGINE_ID,
        reason,
        detail: detail.into(),
    }
}

fn first_execution_state(profile: GuestCpuProfile) -> ExecutionState {
    [
        ExecutionState::A64,
        ExecutionState::A32,
        ExecutionState::T32,
    ]
    .into_iter()
    .find(|state| profile.allowed_execution_states().contains(*state))
    .expect("a guest CPU profile has at least one execution state")
}

fn current_pc(state: &ThreadCpuState) -> u64 {
    match state {
        ThreadCpuState::A64(state) => state.pc(),
        ThreadCpuState::A32(state) => u64::from(state.instruction_address()),
    }
}

fn domain_fault(
    cpu: ProcessCpuContext,
    kind: EngineFaultKind,
    message: impl Into<Box<str>>,
) -> EngineFault {
    let state = ThreadCpuState::new(
        cpu.thread_configuration(first_execution_state(cpu.profile()))
            .expect("the selected state belongs to the profile"),
    );
    fault(kind, 0, message, &state)
}

fn cache_error_detail(error: CacheError) -> Box<str> {
    match error {
        CacheError::Frontend(error) => format!("JIT frontend failed: {error}").into_boxed_str(),
        CacheError::Compiler(error) => {
            format!("JIT lowering failed: {}", error.detail()).into_boxed_str()
        }
        CacheError::Capacity(detail) | CacheError::Internal(detail) => detail,
        CacheError::Stale => "JIT cache observation became stale".into(),
        CacheError::Cancelled => "JIT domain no longer accepts compilation work".into(),
    }
}

fn fault(
    kind: EngineFaultKind,
    instructions_executed: u64,
    message: impl Into<Box<str>>,
    state: &ThreadCpuState,
) -> EngineFault {
    EngineFault {
        engine: JIT_ENGINE_ID,
        kind,
        instructions_executed,
        message: message.into(),
        context: Box::new(state.register_context()),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use nixe_cpu::memory::{
        CpuMemory, ExecutionMemory, MemoryAccess, MemoryAccessClass, MemoryAccessSize,
        MemoryAlignment, MemoryOrdering, MemoryPermissions, MemoryValue, SyntheticMemory,
        SyntheticMmio, SyntheticRamPage,
    };
    use nixe_cpu::state::a32::{A32GeneralRegister as A32Register, A32State};
    use nixe_cpu::state::a64::{A64GeneralRegister, A64Register, A64State, Nzcv};
    use nixe_cpu_engine::{EngineProvider, EngineTimer, TimerSnapshot};
    use nixe_memory::{
        AddressSpaceId, CanonicalRangeTranslator, GuestPhysicalPageId, MemoryInvalidationSource,
    };

    use super::*;

    const SPACE: AddressSpaceId = AddressSpaceId::new(7);

    struct FixedTimer;

    impl EngineTimer for FixedTimer {
        fn snapshot(&self) -> TimerSnapshot {
            TimerSnapshot {
                counter: 123,
                frequency: 19_200_000,
            }
        }
    }

    fn cpu() -> ProcessCpuContext {
        ProcessCpuContext::new(GuestCpuProfile::switch_1(), SPACE)
    }

    fn lse_cpu() -> ProcessCpuContext {
        ProcessCpuContext::new(
            GuestCpuProfile::switch_1().with_instruction_feature(
                nixe_cpu::profile::InstructionFeature::LargeSystemExtensions,
                nixe_cpu::profile::CapabilityStatus::Enabled,
            ),
            SPACE,
        )
    }

    fn bound_executor() -> (Box<dyn EngineDomain>, Box<dyn EngineExecutor>) {
        let provider = JitProvider::new();
        let mut domain = provider
            .create_domain(DomainRequest {
                domain: EngineDomainId::new(9),
                cpu: cpu(),
            })
            .unwrap();
        let memory = ExecutionMemory::new();
        domain
            .bind_memory(DomainMemoryBinding {
                address_space: SPACE,
                end_exclusive: GuestVirtualAddress::new(1 << 39),
                memory: &memory,
                mapping_epoch: 3,
                invalidation_cursor: nixe_memory::MemoryInvalidationCursor::INITIAL,
            })
            .unwrap();
        let executor = domain.create_executor(EngineExecutorId::new(11)).unwrap();
        (domain, executor)
    }

    fn executor_for_execution_memory(
        memory: &ExecutionMemory,
        domain_id: u64,
        executor_id: u64,
    ) -> (Box<dyn EngineDomain>, Box<dyn EngineExecutor>) {
        let provider = JitProvider::new();
        let mut domain = provider
            .create_domain(DomainRequest {
                domain: EngineDomainId::new(domain_id),
                cpu: cpu(),
            })
            .unwrap();
        domain
            .bind_memory(DomainMemoryBinding {
                address_space: SPACE,
                end_exclusive: GuestVirtualAddress::new(1 << 39),
                memory,
                mapping_epoch: memory.mapping_epoch().get(),
                invalidation_cursor: memory.invalidation_cursor(),
            })
            .unwrap();
        let executor = domain
            .create_executor(EngineExecutorId::new(executor_id))
            .unwrap();
        (domain, executor)
    }

    #[test]
    fn domain_shutdown_refuses_to_release_resources_while_an_executor_is_live() {
        let (mut domain, executor) = bound_executor();
        domain.request_stop().unwrap();
        let error = domain.shutdown().unwrap_err();
        assert_eq!(error.kind, EngineFaultKind::Unavailable);
        assert!(error.message.contains("every executor"));
        drop(executor);
        domain.shutdown().unwrap();
        domain.shutdown().unwrap();
    }

    fn execute_a64_program(
        encodings: &[u32],
        budget: u64,
        events: nixe_cpu_engine::VcpuEventState,
        timer: &dyn EngineTimer,
    ) -> (ExecutionReport, ThreadCpuState) {
        let (_domain, mut executor) = bound_executor();
        let mut memory = SyntheticMemory::new();
        let code_page = GuestPhysicalPageId::new(101);
        assert!(memory.add_ram_page(code_page));
        for (index, encoding) in encodings.iter().enumerate() {
            assert!(memory.initialize_ram(code_page, index * 4, &encoding.to_le_bytes()));
        }
        assert!(memory.map_page(
            SPACE,
            GuestVirtualAddress::new(0x1000),
            code_page,
            MemoryPermissions::READ_EXECUTE,
        ));
        let mut state = ThreadCpuState::A64(Box::new({
            let mut state = A64State::default();
            state.set_pc(0x1000);
            state
        }));
        let report = executor
            .run_slice(RunRequest {
                cpu: cpu(),
                memory: &memory,
                state: &mut state,
                instruction_budget: budget,
                loader_return: None,
                timer,
                events,
            })
            .unwrap();
        (report, state)
    }

    #[test]
    fn processor_hints_lower_to_typed_scheduler_and_vcpu_event_actions() {
        const YIELD: u32 = 0xd503_203f;
        const WFE: u32 = 0xd503_205f;
        const WFI: u32 = 0xd503_207f;
        const SEV: u32 = 0xd503_209f;
        const SEVL: u32 = 0xd503_20bf;

        for (encoding, expected) in [
            (YIELD, SchedulerRequest::Yield),
            (WFE, SchedulerRequest::WaitForEvent),
            (WFI, SchedulerRequest::WaitForInterrupt),
            (SEV, SchedulerRequest::SendEvent),
        ] {
            let (report, state) = execute_a64_program(
                &[encoding],
                4,
                nixe_cpu_engine::VcpuEventState::default(),
                &FixedTimer,
            );
            assert_eq!(report.instructions_executed, 1);
            assert!(matches!(
                report.stop,
                nixe_cpu_engine::EngineExit::Scheduled { request, .. } if request == expected
            ));
            let ThreadCpuState::A64(state) = state else {
                unreachable!()
            };
            assert_eq!(state.pc(), 0x1004);
        }

        let events = nixe_cpu_engine::VcpuEventState::default();
        events.signal_event();
        let (report, _) = execute_a64_program(&[WFE, YIELD], 4, events, &FixedTimer);
        assert_eq!(report.instructions_executed, 2);
        assert!(matches!(
            report.stop,
            nixe_cpu_engine::EngineExit::Scheduled {
                request: SchedulerRequest::Yield,
                ..
            }
        ));

        let (report, _) = execute_a64_program(
            &[SEVL, WFE, YIELD],
            4,
            nixe_cpu_engine::VcpuEventState::default(),
            &FixedTimer,
        );
        assert_eq!(report.instructions_executed, 3);
        assert!(matches!(
            report.stop,
            nixe_cpu_engine::EngineExit::Scheduled {
                request: SchedulerRequest::Yield,
                ..
            }
        ));
    }

    #[test]
    fn native_timer_reads_sample_the_runtime_provider_at_each_instruction() {
        struct AdvancingTimer(std::sync::atomic::AtomicU64);

        impl EngineTimer for AdvancingTimer {
            fn snapshot(&self) -> TimerSnapshot {
                TimerSnapshot {
                    counter: self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
                    frequency: 19_200_000,
                }
            }
        }

        let timer = AdvancingTimer(std::sync::atomic::AtomicU64::new(100));
        let (report, state) = execute_a64_program(
            &[0xd53b_e020, 0xd53b_e021, 0xd503_203f],
            4,
            nixe_cpu_engine::VcpuEventState::default(),
            &timer,
        );
        assert_eq!(report.instructions_executed, 3);
        let ThreadCpuState::A64(state) = state else {
            unreachable!()
        };
        assert_eq!(
            state.read_x(A64Register::General(A64GeneralRegister::new(0).unwrap())),
            100
        );
        assert_eq!(
            state.read_x(A64Register::General(A64GeneralRegister::new(1).unwrap())),
            101
        );
    }

    #[test]
    fn native_host_and_executable_memory_are_capability_gates() {
        let permitted = JitProvider::new();
        let report = permitted.probe(GuestCpuProfile::switch_1(), EngineCapabilities::default());
        if cfg!(all(
            any(target_arch = "x86_64", target_arch = "aarch64"),
            any(
                all(unix, not(target_vendor = "apple")),
                target_os = "macos",
                windows
            )
        )) {
            assert!(report.available, "{:?}", report.rejections);
        } else {
            assert!(!report.available);
            assert!(report.rejections.iter().any(|rejection| {
                rejection.reason == CapabilityRejectionReason::PlatformUnsupported
            }));
        }

        let denied = JitProvider::with_executable_error(
            ExecutableMemoryError::privilege_denied_for_test("sandbox forbids JIT"),
        );
        let report = denied.probe(GuestCpuProfile::switch_1(), EngineCapabilities::default());
        assert!(!report.available);
        if cfg!(all(
            any(target_arch = "x86_64", target_arch = "aarch64"),
            any(
                all(unix, not(target_vendor = "apple")),
                target_os = "macos",
                windows
            )
        )) {
            assert!(report.rejections.iter().any(|rejection| {
                rejection.reason == CapabilityRejectionReason::PrivilegeUnavailable
                    && rejection.detail.as_ref() == "sandbox forbids JIT"
            }));
        } else {
            assert!(report.rejections.iter().any(|rejection| {
                rejection.reason == CapabilityRejectionReason::PlatformUnsupported
            }));
        }
        let error = match denied.create_domain(DomainRequest {
            domain: EngineDomainId::new(1),
            cpu: cpu(),
        }) {
            Ok(_) => panic!("denied executable-code policy created a domain"),
            Err(error) => error,
        };
        assert_eq!(error.kind, EngineFaultKind::Unavailable);
    }

    #[test]
    fn provider_construction_defers_executable_memory_until_capability_probe() {
        let provider = JitProvider::new();
        assert!(provider.host.get().is_none());
        assert!(provider.executable_memory.get().is_none());
        assert_eq!(provider.descriptor().id, JIT_ENGINE_ID);
        assert!(provider.host.get().is_none());
        assert!(provider.executable_memory.get().is_none());
    }

    #[cfg(all(
        any(target_arch = "x86_64", target_arch = "aarch64"),
        any(all(unix, not(target_vendor = "apple")), target_os = "macos", windows)
    ))]
    #[test]
    fn providers_share_the_single_process_executable_memory_owner() {
        let first = JitProvider::new();
        let second = JitProvider::new();
        let first = first.executable_memory().as_ref().unwrap();
        let second = second.executable_memory().as_ref().unwrap();
        assert!(Arc::ptr_eq(first, second));
    }

    #[test]
    fn unverified_guest_profile_is_rejected() {
        let provider = JitProvider::new();
        let report = provider.probe(
            GuestCpuProfile::switch_2_native(),
            EngineCapabilities::default(),
        );
        assert!(!report.available);
        assert!(report.rejections.iter().any(|rejection| {
            rejection.reason == CapabilityRejectionReason::GuestProfileUnsupported
        }));
    }

    #[test]
    fn domain_requires_the_matching_canonical_memory_binding() {
        let provider = JitProvider::new();
        let mut domain = provider
            .create_domain(DomainRequest {
                domain: EngineDomainId::new(1),
                cpu: cpu(),
            })
            .unwrap();
        assert!(domain.create_executor(EngineExecutorId::new(1)).is_err());

        let memory = ExecutionMemory::new();
        let error = domain
            .bind_memory(DomainMemoryBinding {
                address_space: AddressSpaceId::new(99),
                end_exclusive: GuestVirtualAddress::new(1 << 39),
                memory: &memory,
                mapping_epoch: 0,
                invalidation_cursor: nixe_memory::MemoryInvalidationCursor::INITIAL,
            })
            .unwrap_err();
        assert_eq!(error.kind, EngineFaultKind::InvalidRequest);
    }

    #[test]
    fn executor_lowers_publishes_and_executes_verified_integer_region() {
        let (_domain, mut executor) = bound_executor();
        let mut memory = SyntheticMemory::new();
        let code_page = GuestPhysicalPageId::new(77);
        assert!(memory.add_ram_page(code_page));
        // ADD X0,X0,#1; B 0x2000. Arm ARM DDI 0602 ADD (immediate)
        // and B: https://developer.arm.com/documentation/ddi0602/latest/
        assert!(memory.initialize_ram(code_page, 0, &0x9100_0400_u32.to_le_bytes()));
        assert!(memory.initialize_ram(code_page, 4, &0x1400_03ff_u32.to_le_bytes()));
        assert!(memory.map_page(
            SPACE,
            GuestVirtualAddress::new(0x1000),
            code_page,
            MemoryPermissions::READ_EXECUTE,
        ));
        let mut a64 = A64State::default();
        a64.set_pc(0x1000);
        let x0 = A64Register::General(A64GeneralRegister::new(0).unwrap());
        a64.write_x(x0, 41);
        let mut state = ThreadCpuState::A64(Box::new(a64));

        let report = executor
            .run_slice(RunRequest {
                cpu: cpu(),
                memory: &memory,
                state: &mut state,
                instruction_budget: 10,
                loader_return: None,
                timer: &FixedTimer,
                events: nixe_cpu_engine::VcpuEventState::default(),
            })
            .unwrap();

        let ThreadCpuState::A64(state) = state else {
            unreachable!()
        };
        assert_eq!(state.read_x(x0), 42);
        assert_eq!(state.pc(), 0x2000);
        assert_eq!(report.instructions_executed, 2);
        assert!(matches!(
            report.stop,
            nixe_cpu_engine::EngineExit::FetchFault { fault }
                if fault.address == GuestVirtualAddress::new(0x2000)
        ));
    }

    #[test]
    fn exact_a64_fp_slow_path_obeys_guest_rounding_and_updates_fpsr() {
        let (_domain, mut executor) = bound_executor();
        let mut memory = SyntheticMemory::new();
        let code_page = GuestPhysicalPageId::new(103);
        assert!(memory.add_ram_page(code_page));
        // FADD D0,D1,D2; B 0x2000.
        assert!(memory.initialize_ram(code_page, 0, &0x1e62_2820_u32.to_le_bytes()));
        assert!(memory.initialize_ram(code_page, 4, &0x1400_03ff_u32.to_le_bytes()));
        assert!(memory.map_page(
            SPACE,
            GuestVirtualAddress::new(0x1000),
            code_page,
            MemoryPermissions::READ_EXECUTE,
        ));
        let halfway = 970_u64 << 52; // 2^-53, half an ulp at 1.0.
        let mut a64 = A64State::default();
        a64.set_pc(0x1000);
        a64.set_fpcr(1 << 22); // Round toward positive infinity.
        a64.set_fpsr(1 << 27); // Preserve cumulative QC across the helper exit.
        assert!(a64.set_vector(1, u128::from(1.0_f64.to_bits())));
        assert!(a64.set_vector(2, u128::from(halfway)));
        let mut state = ThreadCpuState::A64(Box::new(a64));

        let report = executor
            .run_slice(RunRequest {
                cpu: cpu(),
                memory: &memory,
                state: &mut state,
                instruction_budget: 10,
                loader_return: None,
                timer: &FixedTimer,
                events: nixe_cpu_engine::VcpuEventState::default(),
            })
            .unwrap();

        let ThreadCpuState::A64(state) = state else {
            unreachable!()
        };
        assert_eq!(state.vector(0), Some(u128::from(1.0_f64.to_bits() + 1)));
        assert_eq!(state.fpsr(), (1 << 27) | (1 << 4)); // QC | IXC
        assert_eq!(report.instructions_executed, 2);
        assert!(matches!(
            report.stop,
            nixe_cpu_engine::EngineExit::FetchFault { fault }
                if fault.address == GuestVirtualAddress::new(0x2000)
        ));
    }

    #[test]
    fn exact_a64_fp_trap_is_precise_and_atomic() {
        let (_domain, mut executor) = bound_executor();
        let mut memory = SyntheticMemory::new();
        let code_page = GuestPhysicalPageId::new(104);
        assert!(memory.add_ram_page(code_page));
        assert!(memory.initialize_ram(code_page, 0, &0x1e62_2820_u32.to_le_bytes()));
        assert!(memory.map_page(
            SPACE,
            GuestVirtualAddress::new(0x1000),
            code_page,
            MemoryPermissions::READ_EXECUTE,
        ));
        let mut a64 = A64State::default();
        a64.set_pc(0x1000);
        a64.set_fpcr(1 << 8); // Invalid-operation exception enable.
        a64.set_fpsr(0x80);
        assert!(a64.set_vector(0, u128::MAX));
        assert!(a64.set_vector(1, u128::from(f64::INFINITY.to_bits())));
        assert!(a64.set_vector(2, u128::from(f64::NEG_INFINITY.to_bits())));
        let mut state = ThreadCpuState::A64(Box::new(a64));

        let report = executor
            .run_slice(RunRequest {
                cpu: cpu(),
                memory: &memory,
                state: &mut state,
                instruction_budget: 10,
                loader_return: None,
                timer: &FixedTimer,
                events: nixe_cpu_engine::VcpuEventState::default(),
            })
            .unwrap();

        let ThreadCpuState::A64(state) = state else {
            unreachable!()
        };
        assert_eq!(state.vector(0), Some(u128::MAX));
        assert_eq!(state.fpsr(), 0x80);
        assert_eq!(state.pc(), 0x1000);
        assert_eq!(report.instructions_executed, 1);
        assert!(matches!(
            report.stop,
            nixe_cpu_engine::EngineExit::ArchitecturalException {
                kind: nixe_cpu::exception::ExceptionKind::FloatingPoint,
                syndrome: None,
                ..
            }
        ));
    }

    #[test]
    fn exact_a64_simd_slow_path_preserves_lane_and_permutation_rules() {
        let (_domain, mut executor) = bound_executor();
        let mut memory = SyntheticMemory::new();
        let code_page = GuestPhysicalPageId::new(105);
        assert!(memory.add_ram_page(code_page));
        // ZIP1 V7.8B,V1.8B,V2.8B; B 0x2000.
        assert!(memory.initialize_ram(code_page, 0, &0x0e02_3827_u32.to_le_bytes()));
        assert!(memory.initialize_ram(code_page, 4, &0x1400_03ff_u32.to_le_bytes()));
        assert!(memory.map_page(
            SPACE,
            GuestVirtualAddress::new(0x1000),
            code_page,
            MemoryPermissions::READ_EXECUTE,
        ));
        let mut a64 = A64State::default();
        a64.set_pc(0x1000);
        assert!(a64.set_vector(
            1,
            u128::from_le_bytes([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        ));
        assert!(a64.set_vector(
            2,
            u128::from_le_bytes([
                0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x8b, 0x8c, 0x8d,
                0x8e, 0x8f,
            ]),
        ));
        let mut state = ThreadCpuState::A64(Box::new(a64));

        let report = executor
            .run_slice(RunRequest {
                cpu: cpu(),
                memory: &memory,
                state: &mut state,
                instruction_budget: 10,
                loader_return: None,
                timer: &FixedTimer,
                events: nixe_cpu_engine::VcpuEventState::default(),
            })
            .unwrap();

        let ThreadCpuState::A64(state) = state else {
            unreachable!()
        };
        assert_eq!(
            state.vector(7),
            Some(u128::from_le_bytes([
                0, 0x80, 1, 0x81, 2, 0x82, 3, 0x83, 0, 0, 0, 0, 0, 0, 0, 0,
            ]))
        );
        assert_eq!(report.instructions_executed, 2);
    }

    #[test]
    fn exact_a64_conversion_saturates_nan_and_infinity_and_accumulates_status() {
        let (_domain, mut executor) = bound_executor();
        let mut memory = SyntheticMemory::new();
        let code_page = GuestPhysicalPageId::new(107);
        assert!(memory.add_ram_page(code_page));
        // FCVTZU X0,D1; FCVTZS X2,D3; B 0x2000.
        assert!(memory.initialize_ram(code_page, 0, &0x9e79_0020_u32.to_le_bytes()));
        assert!(memory.initialize_ram(code_page, 4, &0x9e78_0062_u32.to_le_bytes()));
        assert!(memory.initialize_ram(code_page, 8, &0x1400_03fe_u32.to_le_bytes()));
        assert!(memory.map_page(
            SPACE,
            GuestVirtualAddress::new(0x1000),
            code_page,
            MemoryPermissions::READ_EXECUTE,
        ));
        let mut a64 = A64State::default();
        a64.set_pc(0x1000);
        assert!(a64.set_vector(1, u128::from(f64::INFINITY.to_bits())));
        assert!(a64.set_vector(3, u128::from(f64::NAN.to_bits())));
        let mut state = ThreadCpuState::A64(Box::new(a64));

        let report = executor
            .run_slice(RunRequest {
                cpu: cpu(),
                memory: &memory,
                state: &mut state,
                instruction_budget: 10,
                loader_return: None,
                timer: &FixedTimer,
                events: nixe_cpu_engine::VcpuEventState::default(),
            })
            .unwrap();

        let ThreadCpuState::A64(state) = state else {
            unreachable!()
        };
        assert_eq!(
            state.read_x(A64Register::General(A64GeneralRegister::new(0).unwrap())),
            u64::MAX
        );
        assert_eq!(
            state.read_x(A64Register::General(A64GeneralRegister::new(2).unwrap())),
            0
        );
        assert_eq!(state.fpsr(), 1); // IOC
        assert_eq!(report.instructions_executed, 3);
    }

    #[test]
    fn precise_memory_helper_uses_canonical_memory_and_precise_state() {
        let (_domain, mut executor) = bound_executor();
        let mut memory = SyntheticMemory::new();
        let code_page = GuestPhysicalPageId::new(78);
        let data_page = GuestPhysicalPageId::new(79);
        assert!(memory.add_ram_page(code_page));
        assert!(memory.add_ram_page(data_page));
        // LDR X0,[X1]; B 0x3000. Arm ARM DDI 0602 LDR (immediate).
        assert!(memory.initialize_ram(code_page, 0, &0xf940_0020_u32.to_le_bytes()));
        assert!(memory.initialize_ram(code_page, 4, &0x1400_07ff_u32.to_le_bytes()));
        assert!(memory.initialize_ram(data_page, 0, &0x0123_4567_89ab_cdef_u64.to_le_bytes(),));
        assert!(memory.map_page(
            SPACE,
            GuestVirtualAddress::new(0x1000),
            code_page,
            MemoryPermissions::READ_EXECUTE,
        ));
        assert!(memory.map_page(
            SPACE,
            GuestVirtualAddress::new(0x2000),
            data_page,
            MemoryPermissions::READ_WRITE,
        ));
        let mut a64 = A64State::default();
        a64.set_pc(0x1000);
        let x0 = A64Register::General(A64GeneralRegister::new(0).unwrap());
        let x1 = A64Register::General(A64GeneralRegister::new(1).unwrap());
        a64.write_x(x1, 0x2000);
        let mut state = ThreadCpuState::A64(Box::new(a64));

        let report = executor
            .run_slice(RunRequest {
                cpu: cpu(),
                memory: &memory,
                state: &mut state,
                instruction_budget: 10,
                loader_return: None,
                timer: &FixedTimer,
                events: nixe_cpu_engine::VcpuEventState::default(),
            })
            .unwrap();

        let ThreadCpuState::A64(state) = state else {
            unreachable!()
        };
        assert_eq!(state.read_x(x0), 0x0123_4567_89ab_cdef);
        assert_eq!(state.pc(), 0x3000);
        assert_eq!(report.instructions_executed, 2);
        assert!(matches!(
            report.stop,
            nixe_cpu_engine::EngineExit::FetchFault { fault }
                if fault.address == GuestVirtualAddress::new(0x3000)
        ));
    }

    #[test]
    fn ordinary_execution_memory_uses_the_canonical_tlb_fast_path() {
        let mut memory = ExecutionMemory::new();
        let mut code = [0_u8; nixe_cpu::memory::SYNTHETIC_PAGE_SIZE];
        let instructions = [
            0xf940_0020_u32, // LDR X0,[X1] -- installs the read lease.
            0xf940_0022_u32, // LDR X2,[X1] -- direct read.
            0x9100_0400_u32, // ADD X0,X0,#1.
            0xf900_0020_u32, // STR X0,[X1] -- first-write guard.
            0x9100_0400_u32, // ADD X0,X0,#1.
            0xf900_0020_u32, // STR X0,[X1] -- direct store.
            0xf940_0022_u32, // LDR X2,[X1] -- observes the direct store.
            0x1400_07f9_u32, // B 0x3000.
        ];
        for (index, instruction) in instructions.into_iter().enumerate() {
            code[index * 4..index * 4 + 4].copy_from_slice(&instruction.to_le_bytes());
        }
        let mut data = [0_u8; nixe_cpu::memory::SYNTHETIC_PAGE_SIZE];
        data[..8].copy_from_slice(&40_u64.to_le_bytes());
        memory
            .install_ram_pages_atomic(
                SPACE,
                &[
                    SyntheticRamPage {
                        virtual_address: GuestVirtualAddress::new(0x1000),
                        bytes: &code,
                        permissions: MemoryPermissions::READ_EXECUTE,
                    },
                    SyntheticRamPage {
                        virtual_address: GuestVirtualAddress::new(0x2000),
                        bytes: &data,
                        permissions: MemoryPermissions::READ_WRITE,
                    },
                ],
            )
            .unwrap();

        let provider = JitProvider::new();
        let mut domain = provider
            .create_domain(DomainRequest {
                domain: EngineDomainId::new(19),
                cpu: cpu(),
            })
            .unwrap();
        domain
            .bind_memory(DomainMemoryBinding {
                address_space: SPACE,
                end_exclusive: GuestVirtualAddress::new(1 << 39),
                mapping_epoch: memory.mapping_epoch().get(),
                invalidation_cursor: memory.invalidation_cursor(),
                memory: &memory,
            })
            .unwrap();
        let mut executor = domain.create_executor(EngineExecutorId::new(20)).unwrap();
        let mut a64 = A64State::default();
        a64.set_pc(0x1000);
        let x1 = A64Register::General(A64GeneralRegister::new(1).unwrap());
        a64.write_x(x1, 0x2000);
        let mut state = ThreadCpuState::A64(Box::new(a64));

        let report = executor
            .run_slice(RunRequest {
                cpu: cpu(),
                memory: &memory,
                state: &mut state,
                instruction_budget: 16,
                loader_return: None,
                timer: &FixedTimer,
                events: nixe_cpu_engine::VcpuEventState::default(),
            })
            .unwrap();
        assert_eq!(report.instructions_executed, 8);
        let ThreadCpuState::A64(state) = state else {
            unreachable!()
        };
        let x2 = A64Register::General(A64GeneralRegister::new(2).unwrap());
        assert_eq!(state.read_x(x2), 42);
        let read = memory
            .read(
                SPACE,
                GuestVirtualAddress::new(0x2000),
                MemoryAccess::new(
                    MemoryAccessSize::Doubleword,
                    MemoryAlignment::Natural,
                    MemoryOrdering::Relaxed,
                    MemoryAccessClass::Normal,
                ),
            )
            .unwrap();
        assert_eq!(read.value, MemoryValue::U64(42));
        let range = memory
            .translate_canonical_range(
                SPACE,
                GuestVirtualAddress::new(0x2000),
                8,
                MemoryPermissions::READ,
            )
            .unwrap();
        assert_eq!(range.segments()[0].current_content_generation().get(), 3);
        assert_eq!(memory.content_mutation_epoch().get(), 2);
    }

    #[test]
    fn canonical_aliases_share_one_backing_through_native_tlb_entries() {
        let mut memory = ExecutionMemory::new();
        let mut code = [0_u8; nixe_cpu::memory::SYNTHETIC_PAGE_SIZE];
        for (index, instruction) in [
            0xf940_0020_u32, // LDR X0,[X1].
            0x9100_0400_u32, // ADD X0,X0,#1.
            0xf900_0040_u32, // STR X0,[X2].
            0xf940_0023_u32, // LDR X3,[X1].
            0x1400_07fc_u32, // B 0x3000 (the alias is not executable).
        ]
        .into_iter()
        .enumerate()
        {
            code[index * 4..index * 4 + 4].copy_from_slice(&instruction.to_le_bytes());
        }
        let mut data = [0_u8; nixe_cpu::memory::SYNTHETIC_PAGE_SIZE];
        data[..8].copy_from_slice(&41_u64.to_le_bytes());
        memory
            .install_ram_pages_atomic(
                SPACE,
                &[
                    SyntheticRamPage {
                        virtual_address: GuestVirtualAddress::new(0x1000),
                        bytes: &code,
                        permissions: MemoryPermissions::READ_EXECUTE,
                    },
                    SyntheticRamPage {
                        virtual_address: GuestVirtualAddress::new(0x2000),
                        bytes: &data,
                        permissions: MemoryPermissions::READ_WRITE,
                    },
                ],
            )
            .unwrap();
        let primary = memory
            .translate_canonical_range(
                SPACE,
                GuestVirtualAddress::new(0x2000),
                1,
                MemoryPermissions::READ,
            )
            .unwrap();
        let physical_page = primary.segments()[0].page().page();
        drop(primary);
        assert!(memory.map_page(
            SPACE,
            GuestVirtualAddress::new(0x3000),
            physical_page,
            MemoryPermissions::READ_WRITE,
        ));

        let (_domain, mut executor) = executor_for_execution_memory(&memory, 41, 42);
        let mut a64 = A64State::default();
        a64.set_pc(0x1000);
        a64.write_x(
            A64Register::General(A64GeneralRegister::new(1).unwrap()),
            0x2000,
        );
        a64.write_x(
            A64Register::General(A64GeneralRegister::new(2).unwrap()),
            0x3000,
        );
        let mut state = ThreadCpuState::A64(Box::new(a64));
        let report = executor
            .run_slice(RunRequest {
                cpu: cpu(),
                memory: &memory,
                state: &mut state,
                instruction_budget: 8,
                loader_return: None,
                timer: &FixedTimer,
                events: nixe_cpu_engine::VcpuEventState::default(),
            })
            .unwrap();

        assert_eq!(report.instructions_executed, 5);
        assert!(matches!(
            report.stop,
            nixe_cpu_engine::EngineExit::FetchFault { fault }
                if fault.address == GuestVirtualAddress::new(0x3000)
        ));
        let ThreadCpuState::A64(state) = state else {
            unreachable!()
        };
        assert_eq!(
            state.read_x(A64Register::General(A64GeneralRegister::new(3).unwrap())),
            42
        );
        assert_eq!(
            memory
                .read(
                    SPACE,
                    GuestVirtualAddress::new(0x2000),
                    MemoryAccess::normal(MemoryAccessSize::Doubleword),
                )
                .unwrap()
                .value,
            MemoryValue::U64(42)
        );
    }

    #[test]
    fn cross_page_native_access_uses_the_precise_memory_boundary() {
        let mut memory = ExecutionMemory::new();
        let mut code = [0_u8; nixe_cpu::memory::SYNTHETIC_PAGE_SIZE];
        code[..4].copy_from_slice(&0xf940_0020_u32.to_le_bytes()); // LDR X0,[X1].
        code[4..8].copy_from_slice(&0x1400_07ff_u32.to_le_bytes()); // B 0x3000.
        let mut first = [0_u8; nixe_cpu::memory::SYNTHETIC_PAGE_SIZE];
        let mut second = [0_u8; nixe_cpu::memory::SYNTHETIC_PAGE_SIZE];
        first[nixe_cpu::memory::SYNTHETIC_PAGE_SIZE - 4..]
            .copy_from_slice(&[0x10, 0x32, 0x54, 0x76]);
        second[..4].copy_from_slice(&[0x98, 0xba, 0xdc, 0xfe]);
        memory
            .install_ram_pages_atomic(
                SPACE,
                &[
                    SyntheticRamPage {
                        virtual_address: GuestVirtualAddress::new(0x1000),
                        bytes: &code,
                        permissions: MemoryPermissions::READ_EXECUTE,
                    },
                    SyntheticRamPage {
                        virtual_address: GuestVirtualAddress::new(0x4000),
                        bytes: &first,
                        permissions: MemoryPermissions::READ_WRITE,
                    },
                    SyntheticRamPage {
                        virtual_address: GuestVirtualAddress::new(0x5000),
                        bytes: &second,
                        permissions: MemoryPermissions::READ_WRITE,
                    },
                ],
            )
            .unwrap();

        let (_domain, mut executor) = executor_for_execution_memory(&memory, 43, 44);
        let mut a64 = A64State::default();
        a64.set_pc(0x1000);
        a64.write_x(
            A64Register::General(A64GeneralRegister::new(1).unwrap()),
            0x4ffc,
        );
        let mut state = ThreadCpuState::A64(Box::new(a64));
        let report = executor
            .run_slice(RunRequest {
                cpu: cpu(),
                memory: &memory,
                state: &mut state,
                instruction_budget: 4,
                loader_return: None,
                timer: &FixedTimer,
                events: nixe_cpu_engine::VcpuEventState::default(),
            })
            .unwrap();

        assert_eq!(report.instructions_executed, 2);
        assert!(matches!(
            report.stop,
            nixe_cpu_engine::EngineExit::FetchFault { fault }
                if fault.address == GuestVirtualAddress::new(0x3000)
        ));
        let ThreadCpuState::A64(state) = state else {
            unreachable!()
        };
        assert_eq!(
            state.read_x(A64Register::General(A64GeneralRegister::new(0).unwrap())),
            0xfedc_ba98_7654_3210
        );
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    enum JitMmioEvent {
        Read(u64, MemoryAccess),
        Write(u64, MemoryAccess, MemoryValue),
    }

    struct RecordingJitMmio {
        events: Arc<Mutex<Vec<JitMmioEvent>>>,
    }

    impl SyntheticMmio for RecordingJitMmio {
        fn read(&mut self, offset: u64, access: MemoryAccess) -> Result<MemoryValue, Box<str>> {
            self.events
                .lock()
                .unwrap()
                .push(JitMmioEvent::Read(offset, access));
            Ok(MemoryValue::U32(0xaabb_ccdd))
        }

        fn write(
            &mut self,
            offset: u64,
            access: MemoryAccess,
            value: MemoryValue,
        ) -> Result<(), Box<str>> {
            self.events
                .lock()
                .unwrap()
                .push(JitMmioEvent::Write(offset, access, value));
            Ok(())
        }
    }

    #[test]
    fn mmio_never_enters_the_native_ram_fast_path() {
        let mut memory = ExecutionMemory::new();
        let code_page = GuestPhysicalPageId::new(120);
        let device_page = GuestPhysicalPageId::new(121);
        let events = Arc::new(Mutex::new(Vec::new()));
        assert!(memory.add_ram_page(code_page));
        assert!(memory.initialize_ram(code_page, 0, &0xb940_0020_u32.to_le_bytes()));
        assert!(memory.initialize_ram(code_page, 4, &0xb900_0420_u32.to_le_bytes()));
        assert!(memory.initialize_ram(code_page, 8, &0x1400_07fe_u32.to_le_bytes()));
        assert!(memory.add_mmio_page(
            device_page,
            RecordingJitMmio {
                events: Arc::clone(&events),
            },
        ));
        assert!(memory.map_page(
            SPACE,
            GuestVirtualAddress::new(0x1000),
            code_page,
            MemoryPermissions::READ_EXECUTE,
        ));
        assert!(memory.map_page(
            SPACE,
            GuestVirtualAddress::new(0x8000),
            device_page,
            MemoryPermissions::READ_WRITE,
        ));

        let (_domain, mut executor) = executor_for_execution_memory(&memory, 45, 46);
        let mut a64 = A64State::default();
        a64.set_pc(0x1000);
        a64.write_x(
            A64Register::General(A64GeneralRegister::new(1).unwrap()),
            0x8000,
        );
        let mut state = ThreadCpuState::A64(Box::new(a64));
        let report = executor
            .run_slice(RunRequest {
                cpu: cpu(),
                memory: &memory,
                state: &mut state,
                instruction_budget: 5,
                loader_return: None,
                timer: &FixedTimer,
                events: nixe_cpu_engine::VcpuEventState::default(),
            })
            .unwrap();

        assert_eq!(report.instructions_executed, 3);
        assert!(matches!(
            report.stop,
            nixe_cpu_engine::EngineExit::FetchFault { fault }
                if fault.address == GuestVirtualAddress::new(0x3000)
        ));
        assert_eq!(
            *events.lock().unwrap(),
            [
                JitMmioEvent::Read(
                    0,
                    MemoryAccess::new(
                        MemoryAccessSize::Word,
                        MemoryAlignment::Unaligned,
                        MemoryOrdering::Relaxed,
                        MemoryAccessClass::Normal,
                    ),
                ),
                JitMmioEvent::Write(
                    4,
                    MemoryAccess::new(
                        MemoryAccessSize::Word,
                        MemoryAlignment::Unaligned,
                        MemoryOrdering::Relaxed,
                        MemoryAccessClass::Normal,
                    ),
                    MemoryValue::U32(0xaabb_ccdd),
                ),
            ]
        );
    }

    #[test]
    fn misaligned_computed_call_faults_before_committing_the_link_register() {
        let (_domain, mut executor) = bound_executor();
        let mut memory = SyntheticMemory::new();
        let code_page = GuestPhysicalPageId::new(122);
        assert!(memory.add_ram_page(code_page));
        assert!(memory.initialize_ram(code_page, 0, &0xd63f_0000_u32.to_le_bytes())); // BLR X0.
        assert!(memory.map_page(
            SPACE,
            GuestVirtualAddress::new(0x1000),
            code_page,
            MemoryPermissions::READ_EXECUTE,
        ));
        let mut a64 = A64State::default();
        a64.set_pc(0x1000);
        a64.write_x(
            A64Register::General(A64GeneralRegister::new(0).unwrap()),
            0x2002,
        );
        let link = A64Register::General(A64GeneralRegister::new(30).unwrap());
        a64.write_x(link, 0xfeed_face_cafe_beef);
        let mut state = ThreadCpuState::A64(Box::new(a64));
        let report = executor
            .run_slice(RunRequest {
                cpu: cpu(),
                memory: &memory,
                state: &mut state,
                instruction_budget: 2,
                loader_return: None,
                timer: &FixedTimer,
                events: nixe_cpu_engine::VcpuEventState::default(),
            })
            .unwrap();

        assert_eq!(report.instructions_executed, 1);
        assert!(matches!(
            report.stop,
            nixe_cpu_engine::EngineExit::ArchitecturalException {
                source,
                kind: nixe_cpu::exception::ExceptionKind::AlignmentFault,
                syndrome: None,
            } if source.pc == GuestVirtualAddress::new(0x1000)
        ));
        let ThreadCpuState::A64(state) = state else {
            unreachable!()
        };
        assert_eq!(state.read_x(link), 0xfeed_face_cafe_beef);
        assert_eq!(state.pc(), 0x1000);
    }

    #[test]
    fn lse_atomic_ir_uses_the_typed_atomic_helper_and_canonical_memory_order() {
        let mut memory = ExecutionMemory::new();
        let mut code = [0_u8; nixe_cpu::memory::SYNTHETIC_PAGE_SIZE];
        for (index, instruction) in [
            0x88e0_fc41_u32, // CASAL W0,W1,[X2].
            0xb8e3_0044_u32, // LDADDAL W3,W4,[X2].
            0x4866_fd48_u32, // CASPAL X6,X7,X8,X9,[X10].
            0x1400_07fd_u32, // B 0x3000.
        ]
        .into_iter()
        .enumerate()
        {
            code[index * 4..index * 4 + 4].copy_from_slice(&instruction.to_le_bytes());
        }
        let mut data = [0_u8; nixe_cpu::memory::SYNTHETIC_PAGE_SIZE];
        data[..4].copy_from_slice(&5_u32.to_le_bytes());
        let pair = 0x1111_2222_3333_4444_u128 | (0x5555_6666_7777_8888_u128 << 64);
        data[16..32].copy_from_slice(&pair.to_le_bytes());
        memory
            .install_ram_pages_atomic(
                SPACE,
                &[
                    SyntheticRamPage {
                        virtual_address: GuestVirtualAddress::new(0x1000),
                        bytes: &code,
                        permissions: MemoryPermissions::READ_EXECUTE,
                    },
                    SyntheticRamPage {
                        virtual_address: GuestVirtualAddress::new(0x2000),
                        bytes: &data,
                        permissions: MemoryPermissions::READ_WRITE,
                    },
                ],
            )
            .unwrap();
        let provider = JitProvider::new();
        let mut domain = provider
            .create_domain(DomainRequest {
                domain: EngineDomainId::new(29),
                cpu: lse_cpu(),
            })
            .unwrap();
        domain
            .bind_memory(DomainMemoryBinding {
                address_space: SPACE,
                end_exclusive: GuestVirtualAddress::new(1 << 39),
                mapping_epoch: memory.mapping_epoch().get(),
                invalidation_cursor: memory.invalidation_cursor(),
                memory: &memory,
            })
            .unwrap();
        let mut executor = domain.create_executor(EngineExecutorId::new(30)).unwrap();
        let mut a64 = A64State::default();
        a64.set_pc(0x1000);
        for (index, value) in [
            (0, 5),
            (1, 9),
            (2, 0x2000),
            (3, 2),
            (6, pair as u64),
            (7, (pair >> 64) as u64),
            (8, 0xaaaa_bbbb_cccc_dddd),
            (9, 0xeeee_ffff_0000_1111),
            (10, 0x2010),
        ] {
            a64.write_x(
                A64Register::General(A64GeneralRegister::new(index).unwrap()),
                value,
            );
        }
        let mut state = ThreadCpuState::A64(Box::new(a64));
        let report = executor
            .run_slice(RunRequest {
                cpu: lse_cpu(),
                memory: &memory,
                state: &mut state,
                instruction_budget: 8,
                loader_return: None,
                timer: &FixedTimer,
                events: nixe_cpu_engine::VcpuEventState::default(),
            })
            .unwrap();
        assert_eq!(report.instructions_executed, 4);
        let ThreadCpuState::A64(state) = state else {
            unreachable!()
        };
        assert_eq!(
            state.read_x(A64Register::General(A64GeneralRegister::new(0).unwrap())),
            5
        );
        assert_eq!(
            state.read_x(A64Register::General(A64GeneralRegister::new(4).unwrap())),
            9
        );
        assert_eq!(
            memory
                .read(
                    SPACE,
                    GuestVirtualAddress::new(0x2000),
                    MemoryAccess::normal(MemoryAccessSize::Word),
                )
                .unwrap()
                .value,
            MemoryValue::U32(11)
        );
        assert_eq!(
            memory
                .read(
                    SPACE,
                    GuestVirtualAddress::new(0x2010),
                    MemoryAccess::normal(MemoryAccessSize::Quadword),
                )
                .unwrap()
                .value,
            MemoryValue::U128(0xaaaa_bbbb_cccc_dddd_u128 | (0xeeee_ffff_0000_1111_u128 << 64))
        );
    }

    #[test]
    fn native_memory_fault_counts_the_faulting_instruction_once() {
        let (_domain, mut executor) = bound_executor();
        let mut memory = SyntheticMemory::new();
        let code_page = GuestPhysicalPageId::new(80);
        assert!(memory.add_ram_page(code_page));
        assert!(memory.initialize_ram(code_page, 0, &0xf940_0020_u32.to_le_bytes()));
        assert!(memory.initialize_ram(code_page, 4, &0x1400_07ff_u32.to_le_bytes()));
        assert!(memory.map_page(
            SPACE,
            GuestVirtualAddress::new(0x1000),
            code_page,
            MemoryPermissions::READ_EXECUTE,
        ));
        let mut a64 = A64State::default();
        a64.set_pc(0x1000);
        let x1 = A64Register::General(A64GeneralRegister::new(1).unwrap());
        a64.write_x(x1, 0x4000);
        let mut state = ThreadCpuState::A64(Box::new(a64));

        let report = executor
            .run_slice(RunRequest {
                cpu: cpu(),
                memory: &memory,
                state: &mut state,
                instruction_budget: 10,
                loader_return: None,
                timer: &FixedTimer,
                events: nixe_cpu_engine::VcpuEventState::default(),
            })
            .unwrap();

        assert_eq!(report.instructions_executed, 1);
        assert!(matches!(
            report.stop,
            nixe_cpu_engine::EngineExit::DataFault { source, fault }
                if source.pc == GuestVirtualAddress::new(0x1000)
                    && fault.address == GuestVirtualAddress::new(0x4000)
        ));
        let ThreadCpuState::A64(state) = state else {
            unreachable!()
        };
        assert_eq!(state.pc(), 0x1000);
    }

    #[test]
    fn backward_native_edge_polls_and_obeys_the_exact_budget() {
        let (_domain, mut executor) = bound_executor();
        let mut memory = SyntheticMemory::new();
        let code_page = GuestPhysicalPageId::new(81);
        assert!(memory.add_ram_page(code_page));
        // B . is an architectural backward edge and therefore a JIT safepoint.
        assert!(memory.initialize_ram(code_page, 0, &0x1400_0000_u32.to_le_bytes()));
        assert!(memory.map_page(
            SPACE,
            GuestVirtualAddress::new(0x1000),
            code_page,
            MemoryPermissions::READ_EXECUTE,
        ));
        let mut a64 = A64State::default();
        a64.set_pc(0x1000);
        let mut state = ThreadCpuState::A64(Box::new(a64));

        let report = executor
            .run_slice(RunRequest {
                cpu: cpu(),
                memory: &memory,
                state: &mut state,
                instruction_budget: 5,
                loader_return: None,
                timer: &FixedTimer,
                events: nixe_cpu_engine::VcpuEventState::default(),
            })
            .unwrap();

        assert_eq!(report.instructions_executed, 5);
        assert_eq!(report.stop, nixe_cpu_engine::EngineExit::BudgetExhausted);
        let ThreadCpuState::A64(state) = state else {
            unreachable!()
        };
        assert_eq!(state.pc(), 0x1000);
    }

    #[test]
    fn aarch32_typed_semantic_helper_preserves_native_region_state() {
        let (_domain, mut executor) = bound_executor();
        let mut memory = SyntheticMemory::new();
        let code_page = GuestPhysicalPageId::new(82);
        assert!(memory.add_ram_page(code_page));
        // ADD R0,R0,#1; B 0x2000. Arm ARM DDI 0602 A32 ADD (immediate).
        assert!(memory.initialize_ram(code_page, 0, &0xe280_0001_u32.to_le_bytes()));
        assert!(memory.initialize_ram(code_page, 4, &0xea00_03fd_u32.to_le_bytes()));
        assert!(memory.map_page(
            SPACE,
            GuestVirtualAddress::new(0x1000),
            code_page,
            MemoryPermissions::READ_EXECUTE,
        ));
        let mut a32 = A32State::a32();
        a32.set_instruction_address(0x1000).unwrap();
        let r0 = A32Register::new(0).unwrap();
        a32.write_r(r0, 41);
        let mut state = ThreadCpuState::A32(Box::new(a32));

        let report = executor
            .run_slice(RunRequest {
                cpu: cpu(),
                memory: &memory,
                state: &mut state,
                instruction_budget: 10,
                loader_return: None,
                timer: &FixedTimer,
                events: nixe_cpu_engine::VcpuEventState::default(),
            })
            .unwrap();

        let ThreadCpuState::A32(state) = state else {
            unreachable!()
        };
        assert_eq!(state.read_r(r0), 42);
        assert_eq!(state.instruction_address(), 0x2000);
        assert_eq!(report.instructions_executed, 2);
        assert!(matches!(
            report.stop,
            nixe_cpu_engine::EngineExit::FetchFault { fault }
                if fault.address == GuestVirtualAddress::new(0x2000)
        ));
    }

    #[test]
    fn exact_aarch32_vfp_slow_path_uses_shared_binary32_semantics() {
        let (_domain, mut executor) = bound_executor();
        let mut memory = SyntheticMemory::new();
        let code_page = GuestPhysicalPageId::new(106);
        assert!(memory.add_ram_page(code_page));
        // VADD.F32 D0,D0,D0; B 0x2000.
        assert!(memory.initialize_ram(code_page, 0, &0xee00_0a00_u32.to_le_bytes()));
        assert!(memory.initialize_ram(code_page, 4, &0xea00_03fd_u32.to_le_bytes()));
        assert!(memory.map_page(
            SPACE,
            GuestVirtualAddress::new(0x1000),
            code_page,
            MemoryPermissions::READ_EXECUTE,
        ));
        let mut a32 = A32State::a32();
        a32.set_instruction_address(0x1000).unwrap();
        assert!(a32.write_d(
            0,
            u64::from(1.5_f32.to_bits()) | (u64::from((-0.0_f32).to_bits()) << 32),
        ));
        let mut state = ThreadCpuState::A32(Box::new(a32));

        let report = executor
            .run_slice(RunRequest {
                cpu: cpu(),
                memory: &memory,
                state: &mut state,
                instruction_budget: 10,
                loader_return: None,
                timer: &FixedTimer,
                events: nixe_cpu_engine::VcpuEventState::default(),
            })
            .unwrap();

        let ThreadCpuState::A32(state) = state else {
            unreachable!()
        };
        assert_eq!(
            state.read_d(0),
            Some(u64::from(3.0_f32.to_bits()) | (u64::from((-0.0_f32).to_bits()) << 32))
        );
        assert_eq!(state.fpscr(), 0);
        assert_eq!(state.instruction_address(), 0x2000);
        assert_eq!(report.instructions_executed, 2);
    }

    #[test]
    fn exact_aarch32_vfp_enabled_exception_is_precise_and_atomic() {
        let (_domain, mut executor) = bound_executor();
        let mut memory = SyntheticMemory::new();
        let code_page = GuestPhysicalPageId::new(108);
        assert!(memory.add_ram_page(code_page));
        // VADD.F32 D0,D1,D2. Both lanes raise invalid for +inf + -inf.
        assert!(memory.initialize_ram(code_page, 0, &0xee01_0a02_u32.to_le_bytes()));
        assert!(memory.map_page(
            SPACE,
            GuestVirtualAddress::new(0x1000),
            code_page,
            MemoryPermissions::READ_EXECUTE,
        ));
        let mut a32 = A32State::a32();
        a32.set_instruction_address(0x1000).unwrap();
        a32.set_fpscr((1 << 8) | 0x80); // IOE plus preserved cumulative state.
        assert!(a32.write_d(0, u64::MAX));
        assert!(a32.write_d(
            1,
            u64::from(f32::INFINITY.to_bits()) | (u64::from(f32::INFINITY.to_bits()) << 32),
        ));
        assert!(a32.write_d(
            2,
            u64::from(f32::NEG_INFINITY.to_bits()) | (u64::from(f32::NEG_INFINITY.to_bits()) << 32),
        ));
        let mut state = ThreadCpuState::A32(Box::new(a32));

        let report = executor
            .run_slice(RunRequest {
                cpu: cpu(),
                memory: &memory,
                state: &mut state,
                instruction_budget: 10,
                loader_return: None,
                timer: &FixedTimer,
                events: nixe_cpu_engine::VcpuEventState::default(),
            })
            .unwrap();

        let ThreadCpuState::A32(state) = state else {
            unreachable!()
        };
        assert_eq!(state.read_d(0), Some(u64::MAX));
        assert_eq!(state.fpscr(), (1 << 8) | 0x80);
        assert_eq!(state.instruction_address(), 0x1000);
        assert_eq!(report.instructions_executed, 1);
        assert!(matches!(
            report.stop,
            nixe_cpu_engine::EngineExit::ArchitecturalException {
                kind: nixe_cpu::exception::ExceptionKind::FloatingPoint,
                syndrome: None,
                ..
            }
        ));
    }

    #[test]
    fn indirect_region_exit_uses_the_native_link_resolver() {
        let (_domain, mut executor) = bound_executor();
        let mut memory = SyntheticMemory::new();
        let first_page = GuestPhysicalPageId::new(83);
        let second_page = GuestPhysicalPageId::new(84);
        assert!(memory.add_ram_page(first_page));
        assert!(memory.add_ram_page(second_page));
        // ADD X0,X0,#1; BR X1. The computed edge cannot be internalized by
        // region formation and therefore exercises the domain link resolver.
        assert!(memory.initialize_ram(first_page, 0, &0x9100_0400_u32.to_le_bytes()));
        assert!(memory.initialize_ram(first_page, 4, &0xd61f_0020_u32.to_le_bytes()));
        // ADD X0,X0,#1; B .
        assert!(memory.initialize_ram(second_page, 0, &0x9100_0400_u32.to_le_bytes()));
        assert!(memory.initialize_ram(second_page, 4, &0x1400_0000_u32.to_le_bytes()));
        assert!(memory.map_page(
            SPACE,
            GuestVirtualAddress::new(0x1000),
            first_page,
            MemoryPermissions::READ_EXECUTE,
        ));
        assert!(memory.map_page(
            SPACE,
            GuestVirtualAddress::new(0x2000),
            second_page,
            MemoryPermissions::READ_EXECUTE,
        ));
        let mut a64 = A64State::default();
        a64.set_pc(0x1000);
        let x0 = A64Register::General(A64GeneralRegister::new(0).unwrap());
        let x1 = A64Register::General(A64GeneralRegister::new(1).unwrap());
        a64.write_x(x1, 0x2000);
        let mut state = ThreadCpuState::A64(Box::new(a64));

        let report = executor
            .run_slice(RunRequest {
                cpu: cpu(),
                memory: &memory,
                state: &mut state,
                instruction_budget: 5,
                loader_return: None,
                timer: &FixedTimer,
                events: nixe_cpu_engine::VcpuEventState::default(),
            })
            .unwrap();

        assert_eq!(report.instructions_executed, 5);
        assert_eq!(report.stop, nixe_cpu_engine::EngineExit::BudgetExhausted);
        let ThreadCpuState::A64(state) = state else {
            unreachable!()
        };
        assert_eq!(state.read_x(x0), 2);
        assert_eq!(state.pc(), 0x2004);
    }

    #[test]
    fn conditional_external_exit_is_resolved_by_the_native_linker() {
        let (_domain, mut executor) = bound_executor();
        let mut memory = SyntheticMemory::new();
        let code_page = GuestPhysicalPageId::new(86);
        assert!(memory.add_ram_page(code_page));
        // B.EQ 0x2000; B . . The taken target is deliberately unmapped, so
        // region formation leaves that conditional edge external.
        assert!(memory.initialize_ram(code_page, 0, &0x5400_8000_u32.to_le_bytes()));
        assert!(memory.initialize_ram(code_page, 4, &0x1400_0000_u32.to_le_bytes()));
        assert!(memory.map_page(
            SPACE,
            GuestVirtualAddress::new(0x1000),
            code_page,
            MemoryPermissions::READ_EXECUTE,
        ));
        let mut a64 = A64State::default();
        a64.set_pc(0x1000);
        a64.set_nzcv(Nzcv::from_bits(Nzcv::Z));
        let mut state = ThreadCpuState::A64(Box::new(a64));

        let report = executor
            .run_slice(RunRequest {
                cpu: cpu(),
                memory: &memory,
                state: &mut state,
                instruction_budget: 5,
                loader_return: None,
                timer: &FixedTimer,
                events: nixe_cpu_engine::VcpuEventState::default(),
            })
            .unwrap();

        assert_eq!(report.instructions_executed, 1);
        assert!(matches!(
            report.stop,
            nixe_cpu_engine::EngineExit::FetchFault { fault }
                if fault.address == GuestVirtualAddress::new(0x2000)
        ));
    }

    #[test]
    fn call_and_return_edges_share_the_same_native_link_system() {
        let (_domain, mut executor) = bound_executor();
        let mut memory = SyntheticMemory::new();
        let caller_page = GuestPhysicalPageId::new(87);
        let callee_page = GuestPhysicalPageId::new(88);
        assert!(memory.add_ram_page(caller_page));
        assert!(memory.add_ram_page(callee_page));
        // BL 0x2000; B .
        assert!(memory.initialize_ram(caller_page, 0, &0x9400_0400_u32.to_le_bytes()));
        assert!(memory.initialize_ram(caller_page, 4, &0x1400_0000_u32.to_le_bytes()));
        // RET
        assert!(memory.initialize_ram(callee_page, 0, &0xd65f_03c0_u32.to_le_bytes()));
        assert!(memory.map_page(
            SPACE,
            GuestVirtualAddress::new(0x1000),
            caller_page,
            MemoryPermissions::READ_EXECUTE,
        ));
        assert!(memory.map_page(
            SPACE,
            GuestVirtualAddress::new(0x2000),
            callee_page,
            MemoryPermissions::READ_EXECUTE,
        ));
        let mut state = ThreadCpuState::A64(Box::new({
            let mut state = A64State::default();
            state.set_pc(0x1000);
            state
        }));

        let report = executor
            .run_slice(RunRequest {
                cpu: cpu(),
                memory: &memory,
                state: &mut state,
                instruction_budget: 4,
                loader_return: None,
                timer: &FixedTimer,
                events: nixe_cpu_engine::VcpuEventState::default(),
            })
            .unwrap();

        assert_eq!(report.instructions_executed, 4);
        assert_eq!(report.stop, nixe_cpu_engine::EngineExit::BudgetExhausted);
        let ThreadCpuState::A64(state) = state else {
            unreachable!()
        };
        let lr = A64Register::General(A64GeneralRegister::new(30).unwrap());
        assert_eq!(state.read_x(lr), 0x1004);
        assert_eq!(state.pc(), 0x1004);
    }

    #[test]
    fn linked_indirect_cycle_carries_one_budget_across_tail_calls() {
        let (_domain, mut executor) = bound_executor();
        let mut memory = SyntheticMemory::new();
        let first_page = GuestPhysicalPageId::new(97);
        let second_page = GuestPhysicalPageId::new(98);
        assert!(memory.add_ram_page(first_page));
        assert!(memory.add_ram_page(second_page));
        // Both computed transfers remain external region edges. After their
        // first misses, the two PICs form a native tail-call cycle.
        // ADD X0,X0,#1; BR X1.
        assert!(memory.initialize_ram(first_page, 0, &0x9100_0400_u32.to_le_bytes()));
        assert!(memory.initialize_ram(first_page, 4, &0xd61f_0020_u32.to_le_bytes()));
        // ADD X0,X0,#1; BR X2.
        assert!(memory.initialize_ram(second_page, 0, &0x9100_0400_u32.to_le_bytes()));
        assert!(memory.initialize_ram(second_page, 4, &0xd61f_0040_u32.to_le_bytes()));
        assert!(memory.map_page(
            SPACE,
            GuestVirtualAddress::new(0x1000),
            first_page,
            MemoryPermissions::READ_EXECUTE,
        ));
        assert!(memory.map_page(
            SPACE,
            GuestVirtualAddress::new(0x2000),
            second_page,
            MemoryPermissions::READ_EXECUTE,
        ));
        let mut a64 = A64State::default();
        a64.set_pc(0x1000);
        let x0 = A64Register::General(A64GeneralRegister::new(0).unwrap());
        let x1 = A64Register::General(A64GeneralRegister::new(1).unwrap());
        let x2 = A64Register::General(A64GeneralRegister::new(2).unwrap());
        a64.write_x(x1, 0x2000);
        a64.write_x(x2, 0x1000);
        let mut state = ThreadCpuState::A64(Box::new(a64));

        let report = executor
            .run_slice(RunRequest {
                cpu: cpu(),
                memory: &memory,
                state: &mut state,
                instruction_budget: 8,
                loader_return: None,
                timer: &FixedTimer,
                events: nixe_cpu_engine::VcpuEventState::default(),
            })
            .unwrap();

        assert_eq!(report.instructions_executed, 8);
        assert_eq!(report.stop, nixe_cpu_engine::EngineExit::BudgetExhausted);
        let ThreadCpuState::A64(state) = state else {
            unreachable!()
        };
        assert_eq!(state.read_x(x0), 4);
        assert_eq!(state.pc(), 0x1000);
    }

    #[test]
    fn exception_and_interpreter_fallback_leave_only_through_normalized_exits() {
        let (_domain, mut executor) = bound_executor();
        let mut a64_memory = SyntheticMemory::new();
        let a64_page = GuestPhysicalPageId::new(89);
        assert!(a64_memory.add_ram_page(a64_page));
        assert!(a64_memory.initialize_ram(a64_page, 0, &0xd400_00e1_u32.to_le_bytes()));
        assert!(a64_memory.map_page(
            SPACE,
            GuestVirtualAddress::new(0x1000),
            a64_page,
            MemoryPermissions::READ_EXECUTE,
        ));
        let mut a64 = ThreadCpuState::A64(Box::new({
            let mut state = A64State::default();
            state.set_pc(0x1000);
            state
        }));
        let report = executor
            .run_slice(RunRequest {
                cpu: cpu(),
                memory: &a64_memory,
                state: &mut a64,
                instruction_budget: 5,
                loader_return: None,
                timer: &FixedTimer,
                events: nixe_cpu_engine::VcpuEventState::default(),
            })
            .unwrap();
        assert_eq!(report.instructions_executed, 1);
        assert!(matches!(
            report.stop,
            nixe_cpu_engine::EngineExit::SupervisorCall { immediate: 7, .. }
        ));

        let mut t32_memory = SyntheticMemory::new();
        let t32_page = GuestPhysicalPageId::new(90);
        assert!(t32_memory.add_ram_page(t32_page));
        // YIELD is intentionally delegated to the exact one-instruction
        // interpreter boundary by the current T32 frontend.
        assert!(t32_memory.initialize_ram(t32_page, 0, &0xbf10_u16.to_le_bytes()));
        assert!(t32_memory.map_page(
            SPACE,
            GuestVirtualAddress::new(0x1000),
            t32_page,
            MemoryPermissions::READ_EXECUTE,
        ));
        let mut t32 = A32State::t32();
        t32.set_instruction_address(0x1000).unwrap();
        let mut t32 = ThreadCpuState::A32(Box::new(t32));
        let report = executor
            .run_slice(RunRequest {
                cpu: cpu(),
                memory: &t32_memory,
                state: &mut t32,
                instruction_budget: 5,
                loader_return: None,
                timer: &FixedTimer,
                events: nixe_cpu_engine::VcpuEventState::default(),
            })
            .unwrap();
        assert_eq!(report.instructions_executed, 0);
        assert!(matches!(
            report.stop,
            nixe_cpu_engine::EngineExit::InterpretOne { source }
                if source.pc == GuestVirtualAddress::new(0x1000)
                    && source.execution_state == ExecutionState::T32
        ));
    }

    #[test]
    fn physical_content_record_retires_cached_native_code_before_reentry() {
        let (_domain, mut executor) = bound_executor();
        let mut memory = SyntheticMemory::new();
        let code_page = GuestPhysicalPageId::new(85);
        assert!(memory.add_ram_page(code_page));
        // ADD X0,X0,#1; B .
        assert!(memory.initialize_ram(code_page, 0, &0x9100_0400_u32.to_le_bytes()));
        assert!(memory.initialize_ram(code_page, 4, &0x1400_0000_u32.to_le_bytes()));
        assert!(memory.map_page(
            SPACE,
            GuestVirtualAddress::new(0x1000),
            code_page,
            MemoryPermissions::READ_EXECUTE,
        ));
        let x0 = A64Register::General(A64GeneralRegister::new(0).unwrap());
        let mut state = ThreadCpuState::A64(Box::new({
            let mut state = A64State::default();
            state.set_pc(0x1000);
            state
        }));
        let run = |executor: &mut dyn EngineExecutor,
                   memory: &SyntheticMemory,
                   state: &mut ThreadCpuState,
                   budget| {
            executor.run_slice(RunRequest {
                cpu: cpu(),
                memory,
                state,
                instruction_budget: budget,
                loader_return: None,
                timer: &FixedTimer,
                events: nixe_cpu_engine::VcpuEventState::default(),
            })
        };

        assert_eq!(
            run(executor.as_mut(), &memory, &mut state, 2).unwrap().stop,
            nixe_cpu_engine::EngineExit::BudgetExhausted
        );
        // Replace ADD #1 with ADD #2 after the first native region is cached.
        assert!(memory.initialize_ram(code_page, 0, &0x9100_0800_u32.to_le_bytes()));
        let ThreadCpuState::A64(a64) = &mut state else {
            unreachable!()
        };
        a64.set_pc(0x1000);
        assert_eq!(
            run(executor.as_mut(), &memory, &mut state, 1).unwrap().stop,
            nixe_cpu_engine::EngineExit::BudgetExhausted
        );
        let ThreadCpuState::A64(a64) = state else {
            unreachable!()
        };
        assert_eq!(a64.read_x(x0), 3);
    }

    #[test]
    fn zero_budget_and_control_requests_are_normalized_without_fallback() {
        let (_domain, mut executor) = bound_executor();
        let memory = ExecutionMemory::new();
        let mut state = ThreadCpuState::A64(Box::new({
            let mut state = A64State::default();
            state.set_pc(0x1000);
            state
        }));
        let events = nixe_cpu_engine::VcpuEventState::default();
        let run = |executor: &mut dyn EngineExecutor, state: &mut ThreadCpuState, budget| {
            executor.run_slice(RunRequest {
                cpu: cpu(),
                memory: &memory,
                state,
                instruction_budget: budget,
                loader_return: None,
                timer: &FixedTimer,
                events: events.clone(),
            })
        };

        assert_eq!(
            run(executor.as_mut(), &mut state, 0).unwrap().stop,
            nixe_cpu_engine::EngineExit::BudgetExhausted
        );
        let control = executor.control().unwrap();
        events.post_interrupts(0x20);
        assert_eq!(
            run(executor.as_mut(), &mut state, 10).unwrap().stop,
            nixe_cpu_engine::EngineExit::PendingEvent { mask: 0x20 }
        );
        control.request(CrossVcpuRequest::Preempt);
        assert_eq!(
            run(executor.as_mut(), &mut state, 10).unwrap().stop,
            nixe_cpu_engine::EngineExit::Safepoint
        );
        control.request_invalidation(9);
        assert_eq!(
            run(executor.as_mut(), &mut state, 10).unwrap().stop,
            nixe_cpu_engine::EngineExit::Safepoint
        );
        assert!(control.acknowledged_invalidation(9));
    }

    #[test]
    fn native_boundary_contains_panics() {
        let result = contain_rust_boundary(|| -> NativeExit {
            panic!("synthetic generated-code boundary failure")
        });
        assert!(result.is_err());
    }

    #[test]
    fn provider_is_usable_behind_the_neutral_trait() {
        let provider: Arc<dyn EngineProvider> = Arc::new(JitProvider::new());
        let descriptor = provider.descriptor();
        assert_eq!(descriptor.kind, EngineKind::Jit);
        assert_eq!(
            descriptor.capabilities.max_safepoint_instructions,
            std::num::NonZeroU64::new(u64::from(
                nixe_cpu::translate::DEFAULT_MAX_GUEST_INSTRUCTIONS_PER_REGION.get()
            ))
        );
    }
}
