//! Concrete JIT process and thread state.

use std::cell::RefCell;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use cranelift_codegen::isa::OwnedTargetIsa;
use cranelift_codegen::settings::{self, Configurable};
use nixe_cpu::coverage::CoverageId;
use nixe_cpu::error::FrontendError;
use nixe_cpu::exclusive::ExclusiveMonitorState;
use nixe_cpu::execution::{
    ControlRequest, CpuControl, CpuFault, CpuFaultKind, CpuProcessId, CpuThreadId, ExecutionReport,
    MemoryBinding, RunRequest, SchedulerRequest,
};
use nixe_cpu::location::{ExecutionState, LocationDescriptor};
use nixe_cpu::profile::ProcessCpuContext;
use nixe_cpu::state::ThreadCpuState;
use nixe_cpu::translate::translate_region_with_decoder;
use nixe_memory::GuestVirtualAddress;

use crate::abi::{
    EXIT_ARCHITECTURAL, EXIT_BUDGET_EXHAUSTED, EXIT_DATA_FAULT, EXIT_DISPATCH, EXIT_INTERNAL,
    EXIT_LOADER_RETURN, EXIT_NONE, EXIT_PENDING_EVENT, EXIT_SAFEPOINT, EXIT_SCHEDULED,
    EXIT_UNSUPPORTED, ExecutionFrame, FrameError, NO_LOADER_RETURN, NativeExit, NativeGateway,
    SCHEDULE_SEND_EVENT, SCHEDULE_WAIT_FOR_EVENT, SCHEDULE_WAIT_FOR_INTERRUPT, SCHEDULE_YIELD,
};
use crate::cache::{
    CacheError, CodeTier, LocalLookupCache, PendingRegion, ProcessCodeCache, PromotionCell,
    RegionKey, ThreadEpoch, TranslationMode, root_code_mapping,
};
use crate::compilation_pool::{CompilationPool, CompilationPoolHandle, host_compilation_workers};
use crate::compiler::{
    CompilationMetrics, CompiledRegionMetadata, CompilerContext, SideExit, compile_gateway,
};
use crate::configuration::JitConfiguration;
use crate::diagnostics::JitDiagnostics;
use crate::executable_memory::{
    ExecutableMemoryError, PublishedCode, SharedExecutableMemory, process_executable_memory,
};
use crate::helpers::{HELPER_TABLE, NativeContext};
use crate::performance::{JitPerformanceReport, ProcessPerformance, ThreadPerformance};

const CONTROL_PREEMPT: u32 = 1 << 0;
const CONTROL_CODE_INVALIDATION: u32 = 1 << 1;

enum HostSupport {
    Available {
        light: OwnedTargetIsa,
        optimized: OwnedTargetIsa,
    },
    Unavailable {
        detail: Box<str>,
    },
}

struct JitResources {
    host: OnceLock<HostSupport>,
    executable_memory: OnceLock<Result<SharedExecutableMemory, ExecutableMemoryError>>,
    performance_report: OnceLock<Result<Arc<JitPerformanceReport>, Box<str>>>,
    configuration: JitConfiguration,
}

impl JitResources {
    fn new(configuration: JitConfiguration) -> Self {
        Self {
            host: OnceLock::new(),
            executable_memory: OnceLock::new(),
            performance_report: OnceLock::new(),
            configuration,
        }
    }

    fn executable_memory(&self) -> &Result<SharedExecutableMemory, ExecutableMemoryError> {
        self.executable_memory
            .get_or_init(process_executable_memory)
    }

    fn host(&self) -> &HostSupport {
        self.host.get_or_init(probe_host)
    }

    fn configured_performance_report(&self) -> Result<Option<Arc<JitPerformanceReport>>, Box<str>> {
        let Some(directory) = self.configuration.performance_report_directory() else {
            return Ok(None);
        };
        match self.performance_report.get_or_init(|| {
            JitPerformanceReport::new(directory, self.configuration.performance_report_title())
                .map(Arc::new)
        }) {
            Ok(report) => Ok(Some(Arc::clone(report))),
            Err(detail) => Err(detail.clone()),
        }
    }

    fn available_resources(
        &self,
        cpu: ProcessCpuContext,
    ) -> Result<(OwnedTargetIsa, OwnedTargetIsa, SharedExecutableMemory), CpuFault> {
        if let HostSupport::Unavailable { detail } = self.host() {
            return Err(fault(
                CpuFaultKind::Unavailable,
                0,
                format!("JIT process creation failed: {detail}"),
                &ThreadCpuState::new(cpu.thread_configuration(ExecutionState::A64)),
            ));
        }
        let (light, optimized) = match self.host() {
            HostSupport::Available { light, optimized } => {
                (Arc::clone(light), Arc::clone(optimized))
            }
            HostSupport::Unavailable { .. } => unreachable!("host failure handled above"),
        };
        let executable_memory = self
            .executable_memory()
            .as_ref()
            .map_err(|error| {
                fault(
                    CpuFaultKind::Unavailable,
                    0,
                    format!("JIT executable memory unavailable: {}", error.detail()),
                    &ThreadCpuState::new(cpu.thread_configuration(ExecutionState::A64)),
                )
            })?
            .clone();
        Ok((light, optimized, executable_memory))
    }
}

impl JitResources {
    fn create_process(
        &self,
        id: CpuProcessId,
        cpu: ProcessCpuContext,
    ) -> Result<JitProcess, CpuFault> {
        let (light_isa, optimized_isa, executable_memory) = self.available_resources(cpu)?;
        let performance = self
            .configured_performance_report()
            .map(|report| report.map(|report| Arc::new(ProcessPerformance::new(report, id))))
            .map_err(|detail| process_fault(cpu, CpuFaultKind::Unavailable, detail))?;
        if let Some(performance) = &performance {
            log::info!(
                "JIT performance report enabled: path={}",
                performance.report_path().display()
            );
        }
        let diagnostics = self
            .configuration
            .dump_directory()
            .map(|directory| {
                JitDiagnostics::new(directory, id, self.configuration.max_cached_regions())
                    .map(Arc::new)
            })
            .transpose()
            .map_err(|detail| process_fault(cpu, CpuFaultKind::Unavailable, detail))?;
        if let Some(diagnostics) = &diagnostics {
            log::info!(
                "JIT compilation dumps enabled: directory={}",
                diagnostics.directory().display()
            );
        }
        let (gateway, gateway_code) =
            compile_gateway(&light_isa, &executable_memory).map_err(|error| {
                process_fault(
                    cpu,
                    CpuFaultKind::Internal,
                    format!("JIT native gateway failed: {}", error.detail()),
                )
            })?;
        let code_cache = Arc::new(ProcessCodeCache::new(self.configuration.clone()));
        let compilation_workers = host_compilation_workers();
        let compilation_pool = CompilationPool::new(
            compilation_workers,
            optimized_isa,
            executable_memory.clone(),
            Arc::clone(&code_cache),
            diagnostics.clone(),
            cpu.profile(),
            performance.is_some(),
        )
        .map_err(|detail| process_fault(cpu, CpuFaultKind::Unavailable, detail))?;
        log::info!("JIT asynchronous compilation pool enabled: workers={compilation_workers}");
        Ok(JitProcess {
            id,
            cpu,
            light_isa,
            executable_memory: Some(executable_memory),
            gateway: Some(gateway),
            gateway_code: Some(gateway_code),
            code_cache: Some(code_cache),
            compilation_pool: Some(compilation_pool),
            diagnostics,
            performance,
            controls: Vec::new(),
            binding: None,
            stopping: false,
            shutdown: false,
        })
    }
}

#[derive(Clone, Copy)]
struct BoundMemory {
    end_exclusive: GuestVirtualAddress,
    mapping_epoch: u64,
    invalidation_cursor: nixe_memory::MemoryInvalidationCursor,
}

pub struct JitProcess {
    id: CpuProcessId,
    cpu: ProcessCpuContext,
    light_isa: OwnedTargetIsa,
    // The process owns the arena so code can never outlive its OS mapping.
    executable_memory: Option<SharedExecutableMemory>,
    gateway: Option<NativeGateway>,
    gateway_code: Option<PublishedCode>,
    code_cache: Option<Arc<ProcessCodeCache>>,
    compilation_pool: Option<CompilationPool>,
    diagnostics: Option<Arc<JitDiagnostics>>,
    performance: Option<Arc<ProcessPerformance>>,
    controls: Vec<CpuControl>,
    binding: Option<BoundMemory>,
    stopping: bool,
    shutdown: bool,
}

impl JitProcess {
    pub fn new(
        id: CpuProcessId,
        cpu: ProcessCpuContext,
        configuration: JitConfiguration,
    ) -> Result<Self, CpuFault> {
        JitResources::new(configuration).create_process(id, cpu)
    }

    #[must_use]
    pub const fn id(&self) -> CpuProcessId {
        self.id
    }

    pub fn create_thread(&mut self, thread_id: CpuThreadId) -> Result<JitThread, CpuFault> {
        if self.stopping || self.shutdown {
            return Err(process_fault(
                self.cpu,
                CpuFaultKind::Unavailable,
                "JIT process is shut down",
            ));
        }
        if self.executable_memory.is_none() {
            return Err(process_fault(
                self.cpu,
                CpuFaultKind::Unavailable,
                "JIT executable-memory owner is unavailable",
            ));
        }
        let Some(binding) = self.binding else {
            return Err(process_fault(
                self.cpu,
                CpuFaultKind::InvalidRequest,
                "canonical memory must be bound before creating a JIT thread",
            ));
        };
        let code_cache = self
            .code_cache
            .as_ref()
            .expect("live process retains its code cache")
            .clone();
        let thread_epoch = code_cache.register_thread();
        let control = CpuControl::default();
        self.controls.push(control.clone());
        Ok(JitThread {
            id: thread_id,
            cpu: self.cpu,
            address_space_end: binding.end_exclusive,
            frame: ExecutionFrame::default(),
            light_compiler: CompilerContext::new(
                Arc::clone(&self.light_isa),
                self.performance.is_some(),
            ),
            compilation_pool: self
                .compilation_pool
                .as_ref()
                .expect("live process retains its compilation pool")
                .handle(),
            _executable_memory: self
                .executable_memory
                .as_ref()
                .expect("live process retains executable memory")
                .clone(),
            gateway: self
                .gateway
                .expect("live process retains its native gateway"),
            _gateway_code: self
                .gateway_code
                .as_ref()
                .expect("live process retains its native gateway publication")
                .clone(),
            code_cache,
            diagnostics: self.diagnostics.clone(),
            process_performance: self.performance.clone(),
            performance: self
                .performance
                .as_ref()
                .map(|_| ThreadPerformance::default()),
            thread_epoch,
            local_lookup: LocalLookupCache::new(),
            control,
            exclusive_monitor: RefCell::new(ExclusiveMonitorState::default()),
            mapping_epoch: binding.mapping_epoch,
            invalidation_cursor: binding.invalidation_cursor,
        })
    }

    pub fn bind_memory(&mut self, binding: MemoryBinding<'_>) -> Result<(), CpuFault> {
        if self.stopping || self.shutdown {
            return Err(process_fault(
                self.cpu,
                CpuFaultKind::Unavailable,
                "cannot bind memory after JIT process stop was requested",
            ));
        }
        if binding.address_space != self.cpu.address_space_id() {
            return Err(process_fault(
                self.cpu,
                CpuFaultKind::InvalidRequest,
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

    pub fn request_stop(&mut self) -> Result<(), CpuFault> {
        if self.stopping || self.shutdown {
            return Ok(());
        }
        if let Some(pool) = &self.compilation_pool {
            pool.begin_shutdown();
        }
        if let Some(cache) = &self.code_cache {
            cache.begin_shutdown().map_err(|error| {
                process_fault(
                    self.cpu,
                    CpuFaultKind::Unavailable,
                    cache_error_detail(error),
                )
            })?;
        }
        for control in &self.controls {
            control.request(ControlRequest::Preempt);
        }
        self.stopping = true;
        Ok(())
    }

    pub fn shutdown(&mut self) -> Result<(), CpuFault> {
        self.request_stop()?;
        if self
            .code_cache
            .as_ref()
            .is_some_and(|cache| cache.live_thread_count() != 0)
        {
            return Err(process_fault(
                self.cpu,
                CpuFaultKind::Unavailable,
                "JIT process shutdown requires every thread to be prepared and dropped",
            ));
        }
        if let Some(pool) = &mut self.compilation_pool {
            pool.shutdown();
        }
        let compilation_pool_performance = self
            .compilation_pool
            .as_ref()
            .map(CompilationPool::snapshot);
        self.shutdown = true;
        if let Some(performance) = self.performance.take() {
            let cache_performance = self
                .code_cache
                .as_ref()
                .and_then(|cache| cache.performance_snapshot());
            match performance.write(
                cache_performance.as_ref(),
                compilation_pool_performance.as_ref(),
            ) {
                Ok(()) => log::info!(
                    "JIT performance report written: path={}",
                    performance.report_path().display()
                ),
                Err(detail) => log::error!("JIT performance report failed: {detail}"),
            }
        }
        self.binding = None;
        self.executable_memory = None;
        self.gateway = None;
        self.gateway_code = None;
        self.code_cache = None;
        self.compilation_pool = None;
        self.diagnostics = None;
        self.controls.clear();
        Ok(())
    }
}

pub struct JitThread {
    id: CpuThreadId,
    cpu: ProcessCpuContext,
    address_space_end: GuestVirtualAddress,
    frame: ExecutionFrame,
    light_compiler: CompilerContext,
    compilation_pool: CompilationPoolHandle,
    // Native execution retains the process arena for the thread's complete
    // lifetime; runtime drops every thread before releasing its process.
    _executable_memory: SharedExecutableMemory,
    gateway: NativeGateway,
    _gateway_code: PublishedCode,
    code_cache: Arc<ProcessCodeCache>,
    diagnostics: Option<Arc<JitDiagnostics>>,
    process_performance: Option<Arc<ProcessPerformance>>,
    performance: Option<ThreadPerformance>,
    thread_epoch: ThreadEpoch,
    local_lookup: LocalLookupCache,
    control: CpuControl,
    exclusive_monitor: RefCell<ExclusiveMonitorState>,
    mapping_epoch: u64,
    invalidation_cursor: nixe_memory::MemoryInvalidationCursor,
}

impl JitThread {
    #[must_use]
    pub const fn id(&self) -> CpuThreadId {
        self.id
    }

    pub fn run_slice(&mut self, request: RunRequest<'_>) -> Result<ExecutionReport, CpuFault> {
        if let Some(performance) = &mut self.performance {
            performance.run_slices = performance.run_slices.saturating_add(1);
        }
        if request.cpu != self.cpu {
            return Err(fault(
                CpuFaultKind::InvalidRequest,
                0,
                "run request CPU context differs from the JIT process",
                request.state,
            ));
        }
        let current_pc = current_pc(request.state);
        if current_pc >= self.address_space_end.get() {
            return Err(fault(
                CpuFaultKind::InvalidRequest,
                0,
                "canonical PC lies outside the bound address space",
                request.state,
            ));
        }
        let trace_jit = log::log_enabled!(log::Level::Trace);

        self.frame.import_state(request.state);
        let fastmem = request.memory.fastmem_view(request.cpu.address_space_id());
        self.frame.memory = crate::abi::MemoryAcceleration {
            address_space: request.cpu.address_space_id().get(),
            mapping_epoch: self.mapping_epoch,
            fastmem_base: fastmem.map_or(0, |view| view.base),
            fastmem_entries: fastmem.map_or(0, |view| view.entries),
            fastmem_size: fastmem.map_or(0, |view| view.address_space_size),
        };
        self.frame.control.instruction_budget = request.instruction_budget;
        self.frame.control.invalidation_epoch = self.invalidation_cursor.get();
        self.frame.control.loader_return = request
            .loader_return
            .map_or(NO_LOADER_RETURN, GuestVirtualAddress::get);
        self.frame.control.control_pending_address = self.control.pending_word_address();
        self.frame.control.interrupt_pending_address = request.events.pending_interrupts_address();

        let mut pending = self.control.take_pending();
        self.frame.control.event_mask = request.events.take_pending_interrupts();
        if let Some(snapshot) = pending {
            self.frame.control.request_flags =
                (u32::from(snapshot.contains(ControlRequest::Preempt)) * CONTROL_PREEMPT)
                    | (u32::from(snapshot.contains(ControlRequest::CodeInvalidation))
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
                pending: Option<nixe_cpu::execution::ControlSnapshot>,
            },
            Frontend(FrontendError),
            Fault(CpuFaultKind, Box<str>),
            Panicked,
        }

        let translation_mode = TranslationMode::Baseline;
        let mut instructions_executed = 0_u64;
        let mut pending_link = None;
        let dispatch = {
            loop {
                if let Some(error) = self.code_cache.background_failure() {
                    break DispatchResult::Fault(
                        CpuFaultKind::Internal,
                        format!(
                            "asynchronous optimized JIT compilation failed: {}",
                            cache_error_detail(error)
                        )
                        .into_boxed_str(),
                    );
                }
                if let Some(snapshot) = self.control.take_pending() {
                    self.frame.control.request_flags |=
                        (u32::from(snapshot.contains(ControlRequest::Preempt)) * CONTROL_PREEMPT)
                            | (u32::from(snapshot.contains(ControlRequest::CodeInvalidation))
                                * CONTROL_CODE_INVALIDATION);
                    self.frame.control.invalidation_epoch = self
                        .frame
                        .control
                        .invalidation_epoch
                        .max(snapshot.invalidation_epoch);
                    pending = Some(match pending {
                        Some(previous) => nixe_cpu::execution::ControlSnapshot {
                            requests: previous.requests | snapshot.requests,
                            invalidation_epoch: previous
                                .invalidation_epoch
                                .max(snapshot.invalidation_epoch),
                        },
                        None => snapshot,
                    });
                }
                if let Err(error) = self.consume_invalidations(
                    request.memory,
                    request.memory.invalidation_cursor(),
                    request.state,
                ) {
                    break DispatchResult::Fault(error.kind, error.message);
                }
                if pending.is_some_and(|snapshot| {
                    snapshot.contains(ControlRequest::CodeInvalidation)
                        && snapshot.invalidation_epoch <= self.invalidation_cursor.get()
                }) {
                    self.frame.control.request_flags &= !CONTROL_CODE_INVALIDATION;
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
                    if let Some(performance) = &mut self.performance {
                        performance.local_cache_hits =
                            performance.local_cache_hits.saturating_add(1);
                    }
                    Ok(region)
                } else {
                    if let Some(performance) = &mut self.performance {
                        performance.local_cache_misses =
                            performance.local_cache_misses.saturating_add(1);
                    }
                    if let Some(region) = self.code_cache.lookup_ready(key) {
                        self.local_lookup.insert(key, Arc::clone(&region));
                        Ok(region)
                    } else {
                        if let Some(performance) = &mut self.performance {
                            performance.cache_miss_resolution_calls =
                                performance.cache_miss_resolution_calls.saturating_add(1);
                        }
                        let resolution_started = self.performance.as_ref().map(|_| Instant::now());
                        let code_cache = Arc::clone(&self.code_cache);
                        let executable_memory = self._executable_memory.clone();
                        let diagnostics = self.diagnostics.clone();
                        let profile = request.cpu.profile();
                        let decoder = request.cpu.decoder();
                        let address_space = request.cpu.address_space_id();
                        let memory = request.memory;
                        let collect_performance = self.performance.is_some();
                        let compiler = &mut self.light_compiler;
                        let performance = &mut self.performance;
                        let result = code_cache.resolve(key, |cancellation| {
                            if let Some(performance) = performance.as_mut() {
                                performance.compilation_attempts =
                                    performance.compilation_attempts.saturating_add(1);
                            }
                            let frontend_started = collect_performance.then(Instant::now);
                            let region = translate_region_with_decoder(
                                translation_mode.config(),
                                decoder,
                                &profile,
                                address_space,
                                location,
                                memory,
                            )?;
                            if let (Some(performance), Some(started)) =
                                (performance.as_mut(), frontend_started)
                            {
                                ThreadPerformance::add_duration(
                                    &mut performance.frontend_ns,
                                    started.elapsed(),
                                );
                            }
                            let region = Arc::new(region);
                            let promotion = Arc::new(PromotionCell::new());
                            let mut metrics = CompilationMetrics::default();
                            let compiled = compiler.compile(
                                &region,
                                &executable_memory,
                                cancellation,
                                Some(Arc::as_ptr(&promotion).addr()),
                                &mut metrics,
                            )?;
                            if let Some(diagnostics) = &diagnostics {
                                diagnostics
                                    .dump_region(&profile, &region, &compiled)
                                    .map_err(CacheError::Internal)?;
                            }
                            if let Some(performance) = performance.as_mut() {
                                performance.record_light_compilation(&region, &compiled, metrics);
                            }
                            PendingRegion::new(
                                address_space,
                                translation_mode,
                                region,
                                compiled,
                                CodeTier::Light,
                                Some(promotion),
                            )
                        });
                        if let (Some(performance), Some(started)) =
                            (&mut self.performance, resolution_started)
                        {
                            ThreadPerformance::add_duration(
                                &mut performance.cache_miss_resolution_ns,
                                started.elapsed(),
                            );
                        }
                        if let Ok(region) = &result {
                            self.local_lookup.insert(key, Arc::clone(region));
                        }
                        result
                    }
                };
                let cached = match cached {
                    Ok(cached) => cached,
                    Err(CacheError::Stale) => continue,
                    Err(CacheError::Cancelled) => {
                        break DispatchResult::Fault(
                            CpuFaultKind::Unavailable,
                            "JIT process stopped while compilation was in progress".into(),
                        );
                    }
                    Err(CacheError::Frontend(error)) => break DispatchResult::Frontend(error),
                    Err(CacheError::Compiler(error)) => {
                        break DispatchResult::Fault(
                            CpuFaultKind::Internal,
                            format!("JIT lowering failed: {}", error.detail()).into_boxed_str(),
                        );
                    }
                    Err(CacheError::Capacity(detail)) => {
                        break DispatchResult::Fault(CpuFaultKind::Unavailable, detail);
                    }
                    Err(CacheError::Internal(detail)) => {
                        break DispatchResult::Fault(CpuFaultKind::Internal, detail);
                    }
                };

                if let Some((source, site, linked_location)) = pending_link.take() {
                    debug_assert_eq!(linked_location, location);
                    match self.code_cache.link(source, site, location, &cached) {
                        Ok((kind, outcome)) => {
                            if let Some(performance) = &mut self.performance {
                                performance.record_link(kind, outcome);
                            }
                        }
                        Err(CacheError::Stale) => {
                            if let Some(performance) = &mut self.performance {
                                performance.link_stale = performance.link_stale.saturating_add(1);
                            }
                            continue;
                        }
                        Err(error) => {
                            break DispatchResult::Fault(
                                CpuFaultKind::Internal,
                                cache_error_detail(error),
                            );
                        }
                    }
                }

                let _native_epoch = match self.code_cache.begin_native(&cached, &self.thread_epoch)
                {
                    Ok(guard) => guard,
                    Err(CacheError::Stale) => continue,
                    Err(error) => {
                        break DispatchResult::Fault(
                            CpuFaultKind::Internal,
                            cache_error_detail(error),
                        );
                    }
                };
                let compiled = cached.compiled();
                let mut native_context = NativeContext::new(
                    request.memory,
                    self.exclusive_monitor.get_mut(),
                    &self.control,
                    request.cpu,
                    request.timer,
                    &request.events,
                )
                .with_invalidations(&self.code_cache, &mut self.invalidation_cursor)
                .with_compilation_pool(&self.compilation_pool)
                .with_performance(self.performance.as_mut());
                cached.install_dispatch(&mut self.frame);
                self.frame.install_host_context(
                    &HELPER_TABLE,
                    std::ptr::from_mut(&mut native_context).cast(),
                );
                let entry_region = cached.id();
                if trace_jit {
                    log::trace!(
                        "JIT native region execution started: thread={:?} region={} start=[{}] entry_pc={:#018x} budget={} native_bytes={}",
                        self.id,
                        entry_region,
                        compiled.metadata.start,
                        self.frame.current_pc(),
                        self.frame.control.instruction_budget,
                        compiled.mapped_len(),
                    );
                }
                if let Some(performance) = native_context.performance.as_deref_mut() {
                    performance.native_entries = performance.native_entries.saturating_add(1);
                }
                let native_started = native_context.performance.as_ref().map(|_| Instant::now());
                let mut fastmem_fault = None;
                let native_exit = contain_rust_boundary(|| {
                    // SAFETY: the live cached region and imported frame remain
                    // valid for this complete non-unwinding native call. The
                    // Linux boundary converts arena SIGSEGV/SIGBUS faults into
                    // an explicit result before control returns to Rust.
                    fastmem_fault = unsafe {
                        crate::fastmem_fault::execute(
                            self.gateway,
                            &raw mut self.frame,
                            compiled.entry_address(),
                            self.frame.memory.fastmem_base,
                            self.frame.memory.fastmem_size,
                        )
                    };
                    self.frame.exit
                });
                let native_elapsed = native_started.map(|started| started.elapsed());
                if trace_jit {
                    match &native_exit {
                        Ok(exit) => log::trace!(
                            "JIT native region execution completed: thread={:?} entry_region={} final_region={} instructions={} exit={}({}) detail={} source_pc={:#018x} next_pc={:#018x}",
                            self.id,
                            entry_region,
                            self.frame.dispatch.region_id,
                            exit.instructions_executed,
                            native_exit_kind_name(exit.kind),
                            exit.kind,
                            exit.detail,
                            exit.source_pc,
                            self.frame.current_pc(),
                        ),
                        Err(()) => log::trace!(
                            "JIT native region execution completed: thread={:?} entry_region={} outcome=panic next_pc={:#018x}",
                            self.id,
                            entry_region,
                            self.frame.current_pc(),
                        ),
                    }
                }
                self.frame.clear_host_context();
                let data_fault = native_context.data_fault.take();
                let native_pending = native_context.control_snapshot.take();
                drop(native_context);
                if let (Some(performance), Some(elapsed)) = (&mut self.performance, native_elapsed)
                {
                    ThreadPerformance::add_duration(&mut performance.native_ns, elapsed);
                    if let Ok(exit) = &native_exit {
                        performance.record_native_exit(exit.kind, exit.instructions_executed);
                    }
                }
                if let Some(address) = fastmem_fault {
                    break DispatchResult::Fault(
                        CpuFaultKind::Internal,
                        format!("native fastmem fault at host address {address:#018x}")
                            .into_boxed_str(),
                    );
                }
                let mut native_exit = match native_exit {
                    Ok(exit) => exit,
                    Err(()) => break DispatchResult::Panicked,
                };
                instructions_executed =
                    match instructions_executed.checked_add(native_exit.instructions_executed) {
                        Some(total) if total <= request.instruction_budget => total,
                        _ => {
                            break DispatchResult::Fault(
                                CpuFaultKind::Internal,
                                "native region exceeded the slice instruction budget".into(),
                            );
                        }
                    };
                native_exit.instructions_executed = instructions_executed;
                if native_exit.kind == EXIT_DISPATCH {
                    if data_fault.is_some() || native_pending.is_some() {
                        break DispatchResult::Fault(
                            CpuFaultKind::Internal,
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

        if let Some(performance) = &mut self.performance {
            performance.instructions_reported = performance
                .instructions_reported
                .saturating_add(instructions_executed);
        }
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
                CpuFaultKind::Internal,
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
    ) -> Result<(), CpuFault> {
        self.consume_invalidations(memory, cursor, state)
    }

    pub fn synchronize_address_space(
        &mut self,
        binding: MemoryBinding<'_>,
        state: &ThreadCpuState,
    ) -> Result<(), CpuFault> {
        if binding.address_space != self.cpu.address_space_id() {
            return Err(fault(
                CpuFaultKind::InvalidRequest,
                0,
                "address-space synchronization belongs to a different process",
                state,
            ));
        }
        self.address_space_end = binding.end_exclusive;
        let mapping_changed = self.mapping_epoch != binding.mapping_epoch;
        self.synchronize_invalidation(binding.invalidation_cursor, state, binding.memory)?;
        if mapping_changed {
            self.mapping_epoch = binding.mapping_epoch;
            self.frame.memory.mapping_epoch = self.mapping_epoch;
        }
        Ok(())
    }

    #[must_use]
    pub fn control(&self) -> CpuControl {
        self.control.clone()
    }

    pub fn prepare_shutdown(
        &mut self,
        binding: MemoryBinding<'_>,
        state: &ThreadCpuState,
    ) -> Result<(), CpuFault> {
        if binding.address_space != self.cpu.address_space_id() {
            return Err(fault(
                CpuFaultKind::InvalidRequest,
                0,
                "JIT teardown binding belongs to a different address space",
                state,
            ));
        }
        self.local_lookup.clear();
        self.exclusive_monitor.get_mut().clear();
        self.invalidation_cursor = binding.invalidation_cursor;
        self.mapping_epoch = binding.mapping_epoch;
        self.frame.memory.mapping_epoch = binding.mapping_epoch;
        self.frame.control.invalidation_epoch = binding.invalidation_cursor.get();
        self.control
            .acknowledge_invalidation(binding.invalidation_cursor.get());
        Ok(())
    }

    pub fn clear_local_exclusive_reservation(&mut self) {
        self.exclusive_monitor.get_mut().clear();
    }
}

impl Drop for JitThread {
    fn drop(&mut self) {
        if let (Some(process), Some(performance)) =
            (&self.process_performance, self.performance.take())
        {
            process.record_thread(self.id, performance);
        }
    }
}

const fn native_exit_kind_name(kind: u32) -> &'static str {
    match kind {
        EXIT_NONE => "none",
        EXIT_BUDGET_EXHAUSTED => "budget-exhausted",
        EXIT_SAFEPOINT => "safepoint",
        EXIT_PENDING_EVENT => "pending-event",
        EXIT_LOADER_RETURN => "loader-return",
        EXIT_DISPATCH => "dispatch",
        EXIT_ARCHITECTURAL => "architectural",
        EXIT_UNSUPPORTED => "unsupported",
        EXIT_DATA_FAULT => "data-fault",
        EXIT_SCHEDULED => "scheduled",
        EXIT_INTERNAL => "internal",
        _ => "unknown",
    }
}

impl JitThread {
    fn consume_invalidations(
        &mut self,
        memory: &dyn nixe_cpu::memory::CpuMemory,
        requested: nixe_memory::MemoryInvalidationCursor,
        state: &ThreadCpuState,
    ) -> Result<(), CpuFault> {
        if requested <= self.invalidation_cursor {
            self.control.acknowledge_invalidation(requested.get());
            if self.frame.control.invalidation_epoch <= self.invalidation_cursor.get() {
                self.frame.control.request_flags &= !CONTROL_CODE_INVALIDATION;
            }
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
                        CpuFaultKind::Unavailable,
                        0,
                        error.to_string(),
                        state,
                    ));
                }
            };
        if let Some(performance) = &mut self.performance {
            performance.invalidation_batches = performance.invalidation_batches.saturating_add(1);
            performance.invalidation_records = performance
                .invalidation_records
                .saturating_add(records.len() as u64);
            performance.invalidation_history_lost = performance
                .invalidation_history_lost
                .saturating_add(u64::from(history_lost));
        }
        let _effect = self
            .code_cache
            .apply_invalidations(&records, through, history_lost)
            .map_err(|error| {
                fault(
                    CpuFaultKind::Unavailable,
                    0,
                    cache_error_detail(error),
                    state,
                )
            })?;
        if history_lost {
            self.local_lookup.clear();
        }
        self.invalidation_cursor = through;
        self.frame.control.invalidation_epoch = through.get();
        self.control.acknowledge_invalidation(through.get());
        self.frame.control.request_flags &= !CONTROL_CODE_INVALIDATION;
        Ok(())
    }

    fn acknowledge_snapshot(&mut self, snapshot: Option<nixe_cpu::execution::ControlSnapshot>) {
        let Some(snapshot) = snapshot else {
            return;
        };
        if snapshot.contains(ControlRequest::CodeInvalidation) {
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
        if source_pc == self.frame.control.loader_return
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
    ) -> Result<(), CpuFault> {
        self.frame.commit_state(state).map_err(|error| {
            fault(
                CpuFaultKind::Internal,
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
) -> Result<nixe_cpu::execution::CpuExit, CpuFault> {
    let execution_state = state.execution_state();
    let source = LocationDescriptor::new(
        GuestVirtualAddress::new(exit.source_pc),
        execution_state,
        cpu.profile().id(),
    );
    match exit.kind {
        EXIT_DISPATCH => Err(fault(
            CpuFaultKind::Internal,
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
                                    CpuFaultKind::Internal,
                                    exit.instructions_executed,
                                    "supervisor-call exit has an invalid immediate",
                                    state,
                                )
                            })?;
                        Ok(nixe_cpu::execution::CpuExit::SupervisorCall { source, immediate })
                    } else {
                        Ok(nixe_cpu::execution::CpuExit::ArchitecturalException {
                            source,
                            kind: *kind,
                            syndrome: *syndrome,
                        })
                    }
                }
                _ => Err(fault(
                    CpuFaultKind::Internal,
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
                } => {
                    let source =
                        compact_source(metadata, *source, state, exit.instructions_executed)?;
                    Err(fault(
                        CpuFaultKind::Internal,
                        exit.instructions_executed,
                        format!(
                            "JIT does not support guest instruction: source={source} \
                             encoding={encoding:?} coverage_id={} disassembly={disassembly}",
                            CoverageId::new(*coverage_id)
                        ),
                        state,
                    ))
                }
                _ => Err(fault(
                    CpuFaultKind::Internal,
                    exit.instructions_executed,
                    "unsupported exit references incompatible side metadata",
                    state,
                )),
            }
        }
        EXIT_DATA_FAULT => data_fault
            .map(|fault| nixe_cpu::execution::CpuExit::DataFault { source, fault })
            .ok_or_else(|| {
                fault(
                    CpuFaultKind::Internal,
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
                        CpuFaultKind::Internal,
                        exit.instructions_executed,
                        "native scheduling exit has an unknown request",
                        state,
                    ));
                }
            };
            Ok(nixe_cpu::execution::CpuExit::Scheduled { source, request })
        }
        EXIT_BUDGET_EXHAUSTED => Ok(nixe_cpu::execution::CpuExit::BudgetExhausted),
        EXIT_SAFEPOINT => Ok(nixe_cpu::execution::CpuExit::Safepoint),
        EXIT_PENDING_EVENT => Ok(nixe_cpu::execution::CpuExit::PendingEvent { mask: exit.detail }),
        EXIT_LOADER_RETURN => Ok(nixe_cpu::execution::CpuExit::LoaderReturn {
            source,
            result_code: exit.payload0,
        }),
        EXIT_NONE => Err(fault(
            CpuFaultKind::Internal,
            exit.instructions_executed,
            "native frame returned without a normalized exit",
            state,
        )),
        crate::abi::EXIT_INTERNAL => Err(fault(
            CpuFaultKind::Internal,
            exit.instructions_executed,
            "native helper reported an internal lowering failure",
            state,
        )),
        _ => Err(fault(
            CpuFaultKind::Internal,
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
) -> Result<(&'a CompiledRegionMetadata, &'a SideExit), CpuFault> {
    metadata
        .and_then(|metadata| {
            metadata
                .side_exits
                .get(index as usize)
                .map(|exit| (metadata, exit))
        })
        .ok_or_else(|| {
            fault(
                CpuFaultKind::Internal,
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
) -> Result<LocationDescriptor, CpuFault> {
    metadata
        .sources
        .get(index as usize)
        .copied()
        .ok_or_else(|| {
            fault(
                CpuFaultKind::Internal,
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
) -> Result<nixe_cpu::execution::CpuExit, CpuFault> {
    match error {
        FrontendError::InstructionFetch(fault) => {
            Ok(nixe_cpu::execution::CpuExit::FetchFault { fault })
        }
        FrontendError::Unallocated(error) => {
            Ok(nixe_cpu::execution::CpuExit::UnallocatedEncoding { error })
        }
        FrontendError::Decode(error) => Err(fault(
            CpuFaultKind::Internal,
            instructions_executed,
            format!("JIT cannot decode guest instruction: {error}"),
            state,
        )),
        FrontendError::InvalidIr(error) => Err(fault(
            CpuFaultKind::Internal,
            instructions_executed,
            format!("JIT frontend produced invalid IR: {error}"),
            state,
        )),
        FrontendError::Internal(error) => Err(fault(
            CpuFaultKind::Internal,
            instructions_executed,
            format!("JIT frontend failed internally: {error}"),
            state,
        )),
        _ => Err(fault(
            CpuFaultKind::Internal,
            instructions_executed,
            "JIT frontend returned an unknown failure",
            state,
        )),
    }
}

/// Contains every Rust-side operation adjacent to native entry. Published
/// functions and helper slots use `extern "C"`, so unwind is not an ABI outcome;
/// helpers must perform the same containment before returning to native code.
fn contain_rust_boundary<T>(operation: impl FnOnce() -> T) -> Result<T, ()> {
    catch_unwind(AssertUnwindSafe(operation)).map_err(|_| ())
}

fn probe_host() -> HostSupport {
    if !cfg!(any(target_arch = "x86_64", target_arch = "aarch64")) {
        return HostSupport::Unavailable {
            detail: format!(
                "Cranelift JIT supports only x86-64 and AArch64 hosts, not {}",
                std::env::consts::ARCH
            )
            .into_boxed_str(),
        };
    }
    let build = |opt_level: &str,
                 regalloc_algorithm: &str,
                 enable_verifier: bool|
     -> Result<OwnedTargetIsa, Box<str>> {
        let builder = cranelift_native::builder().map_err(|detail| {
            format!("Cranelift rejected the native host ISA: {detail}").into_boxed_str()
        })?;
        let mut flag_builder = settings::builder();
        flag_builder
            .set("preserve_frame_pointers", "true")
            .map_err(|error| {
                format!("Cranelift tail-call configuration failed: {error}").into_boxed_str()
            })?;
        flag_builder.set("opt_level", opt_level).map_err(|error| {
            format!("Cranelift optimization configuration failed: {error}").into_boxed_str()
        })?;
        flag_builder
            .set("regalloc_algorithm", regalloc_algorithm)
            .map_err(|error| {
                format!("Cranelift register-allocation configuration failed: {error}")
                    .into_boxed_str()
            })?;
        flag_builder
            .set(
                "enable_verifier",
                if enable_verifier { "true" } else { "false" },
            )
            .map_err(|error| {
                format!("Cranelift verifier configuration failed: {error}").into_boxed_str()
            })?;
        builder
            .finish(settings::Flags::new(flag_builder))
            .map_err(|error| {
                format!("Cranelift native ISA configuration failed: {error}").into_boxed_str()
            })
    };
    match (
        build("none", "single_pass", cfg!(debug_assertions)),
        build("speed", "backtracking", true),
    ) {
        (Ok(light), Ok(optimized)) => HostSupport::Available { light, optimized },
        (Err(detail), _) | (_, Err(detail)) => HostSupport::Unavailable { detail },
    }
}

fn current_pc(state: &ThreadCpuState) -> u64 {
    match state {
        ThreadCpuState::A64(state) => state.pc(),
        ThreadCpuState::A32(state) => u64::from(state.instruction_address()),
    }
}

fn process_fault(
    cpu: ProcessCpuContext,
    kind: CpuFaultKind,
    message: impl Into<Box<str>>,
) -> CpuFault {
    let state = ThreadCpuState::new(cpu.thread_configuration(ExecutionState::A64));
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
        CacheError::Cancelled => "JIT process no longer accepts compilation work".into(),
    }
}

fn fault(
    kind: CpuFaultKind,
    instructions_executed: u64,
    message: impl Into<Box<str>>,
    state: &ThreadCpuState,
) -> CpuFault {
    CpuFault {
        backend: "jit",
        kind,
        instructions_executed,
        message: message.into(),
        context: Box::new(state.register_context()),
    }
}
