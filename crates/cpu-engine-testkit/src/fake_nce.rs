use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use nixe_cpu::exception::ExceptionKind;
use nixe_cpu::location::current_location;
use nixe_cpu::memory::{CpuMemory, MemoryRegionKind};
use nixe_cpu::profile::{GuestCpuProfile, ProcessCpuContext};
use nixe_cpu::state::ThreadCpuState;
use nixe_cpu_engine::{
    CapabilityRejection, CapabilityRejectionReason, CapabilityReport, DomainMemoryBinding,
    DomainRequest, EngineCapabilities, EngineControl, EngineDescriptor, EngineDomain,
    EngineDomainId, EngineExecutor, EngineExecutorId, EngineFault, EngineFaultKind, EngineId,
    EngineKind, EngineProvider, ExecutionReport, ExecutorRequest, InstructionTrace, RunRequest,
};
use nixe_cpu_engine_interpreter::InterpreterDomain;
use nixe_memory::{AddressSpaceId, GuestVirtualAddress, MemoryPermissions};

pub const FAKE_NCE_ENGINE_ID: EngineId = EngineId::new(0xf200);

#[derive(Default)]
pub struct FakeNceMetrics {
    mapping_notifications: AtomicU64,
    invalidation_syncs: AtomicU64,
    teardowns: AtomicU64,
    normalized_traps: AtomicU64,
}

impl FakeNceMetrics {
    #[must_use]
    pub fn mapping_notifications(&self) -> u64 {
        self.mapping_notifications.load(Ordering::Acquire)
    }

    #[must_use]
    pub fn invalidation_syncs(&self) -> u64 {
        self.invalidation_syncs.load(Ordering::Acquire)
    }

    #[must_use]
    pub fn teardowns(&self) -> u64 {
        self.teardowns.load(Ordering::Acquire)
    }

    #[must_use]
    pub fn normalized_traps(&self) -> u64 {
        self.normalized_traps.load(Ordering::Acquire)
    }
}

#[derive(Clone, Default)]
pub struct FakeNceProvider {
    metrics: Arc<FakeNceMetrics>,
}

impl FakeNceProvider {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn metrics(&self) -> Arc<FakeNceMetrics> {
        Arc::clone(&self.metrics)
    }

    #[must_use]
    pub fn create_nce_domain(&self, request: DomainRequest) -> FakeNceDomain {
        FakeNceDomain::new(request, Arc::clone(&self.metrics))
    }
}

impl EngineProvider for FakeNceProvider {
    fn descriptor(&self) -> EngineDescriptor {
        descriptor()
    }

    fn probe(&self, profile: GuestCpuProfile, required: EngineCapabilities) -> CapabilityReport {
        let descriptor = descriptor();
        let available = descriptor.capabilities.supports_profile(profile, required)
            && descriptor.capabilities.contains(required);
        CapabilityReport {
            descriptor,
            available,
            rejections: if available {
                Box::new([])
            } else {
                Box::new([CapabilityRejection {
                    engine: FAKE_NCE_ENGINE_ID,
                    reason: CapabilityRejectionReason::MissingCapabilities,
                    detail: "fake NCE does not satisfy the requested profile and capabilities"
                        .into(),
                }])
            },
        }
    }

    fn create_domain(&self, request: DomainRequest) -> Result<Box<dyn EngineDomain>, EngineFault> {
        Ok(Box::new(self.create_nce_domain(request)))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MirroredMapping {
    base: GuestVirtualAddress,
    size: u64,
    permissions: MemoryPermissions,
}

fn snapshot_mappings(binding: DomainMemoryBinding<'_>) -> Result<Vec<MirroredMapping>, Box<str>> {
    let mut next = Vec::new();
    let mut cursor = GuestVirtualAddress::new(0);
    while cursor.get() < binding.end_exclusive.get() {
        let region = binding
            .memory
            .query_memory(binding.address_space, cursor, binding.end_exclusive)
            .ok_or_else(|| Box::<str>::from("canonical memory query did not advance"))?;
        let end = region
            .base
            .checked_add(region.size)
            .ok_or_else(|| Box::<str>::from("canonical memory region overflowed"))?;
        if end.get() <= cursor.get() {
            return Err("canonical memory query returned an empty region".into());
        }
        if region.region == Some(MemoryRegionKind::Ram) {
            binding
                .memory
                .translate_canonical_range(
                    binding.address_space,
                    region.base,
                    region.size,
                    MemoryPermissions::NONE,
                )
                .map_err(|error| Box::<str>::from(error.to_string()))?;
            next.push(MirroredMapping {
                base: region.base,
                size: region.size,
                permissions: region.permissions,
            });
        }
        cursor = end;
    }
    Ok(next)
}

fn publish_mappings(
    mappings: &Mutex<Vec<MirroredMapping>>,
    metrics: &FakeNceMetrics,
    next: Vec<MirroredMapping>,
) {
    let mut mappings = mappings
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let changes = next
        .iter()
        .filter(|mapping| !mappings.contains(mapping))
        .count()
        + mappings
            .iter()
            .filter(|mapping| !next.contains(mapping))
            .count();
    metrics
        .mapping_notifications
        .fetch_add(changes as u64, Ordering::AcqRel);
    *mappings = next;
}

pub struct FakeNceDomain {
    id: EngineDomainId,
    cpu: ProcessCpuContext,
    oracle: InterpreterDomain,
    metrics: Arc<FakeNceMetrics>,
    shadow: Arc<Mutex<BTreeMap<EngineExecutorId, ThreadCpuState>>>,
    mappings: Arc<Mutex<Vec<MirroredMapping>>>,
    address_space: Option<AddressSpaceId>,
    mapping_generation: u64,
    torn_down: bool,
}

impl FakeNceDomain {
    fn new(request: DomainRequest, metrics: Arc<FakeNceMetrics>) -> Self {
        Self {
            id: request.domain,
            cpu: request.cpu,
            oracle: InterpreterDomain::new(request.domain),
            metrics,
            shadow: Arc::new(Mutex::new(BTreeMap::new())),
            mappings: Arc::new(Mutex::new(Vec::new())),
            address_space: None,
            mapping_generation: 0,
            torn_down: false,
        }
    }

    #[must_use]
    pub fn mirrored_binding_count(&self) -> usize {
        self.mappings
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }

    fn fault(&self, message: impl Into<Box<str>>) -> EngineFault {
        let state = ThreadCpuState::new(
            self.cpu
                .thread_configuration(nixe_cpu::location::ExecutionState::A64)
                .expect("the fake NCE supports A64"),
        );
        EngineFault {
            engine: FAKE_NCE_ENGINE_ID,
            kind: EngineFaultKind::InvalidRequest,
            instructions_executed: 0,
            message: message.into(),
            context: Box::new(state.register_context()),
        }
    }

    fn mirror(&mut self, binding: DomainMemoryBinding<'_>) -> Result<(), EngineFault> {
        let next = snapshot_mappings(binding).map_err(|message| self.fault(message))?;
        publish_mappings(&self.mappings, &self.metrics, next);
        self.address_space = Some(binding.address_space);
        self.mapping_generation = binding.invalidation_generation;
        Ok(())
    }
}

impl EngineDomain for FakeNceDomain {
    fn descriptor(&self) -> EngineDescriptor {
        descriptor()
    }

    fn domain_id(&self) -> EngineDomainId {
        self.id
    }

    fn bind_memory(&mut self, binding: DomainMemoryBinding<'_>) -> Result<(), EngineFault> {
        if self.torn_down {
            return Err(self.fault("fake NCE domain has been shut down"));
        }
        self.mirror(binding)
    }

    fn create_executor(
        &mut self,
        request: ExecutorRequest,
    ) -> Result<Box<dyn EngineExecutor>, EngineFault> {
        if self.torn_down || self.address_space.is_none() {
            return Err(self.fault("fake NCE domain is not bound"));
        }
        let oracle = self.oracle.create_executor(request)?;
        Ok(Box::new(FakeNceExecutor {
            id: request.executor,
            oracle,
            shadow: Arc::clone(&self.shadow),
            metrics: Arc::clone(&self.metrics),
            mappings: Arc::clone(&self.mappings),
            acknowledged_epoch: self.mapping_generation,
        }))
    }

    fn shutdown(&mut self) -> Result<(), EngineFault> {
        if !self.torn_down {
            self.torn_down = true;
            self.mappings
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clear();
            self.shadow
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clear();
            self.metrics.teardowns.fetch_add(1, Ordering::AcqRel);
        }
        Ok(())
    }
}

impl Drop for FakeNceDomain {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

struct FakeNceExecutor {
    id: EngineExecutorId,
    oracle: Box<dyn EngineExecutor>,
    shadow: Arc<Mutex<BTreeMap<EngineExecutorId, ThreadCpuState>>>,
    metrics: Arc<FakeNceMetrics>,
    mappings: Arc<Mutex<Vec<MirroredMapping>>>,
    acknowledged_epoch: u64,
}

impl EngineExecutor for FakeNceExecutor {
    fn descriptor(&self) -> EngineDescriptor {
        descriptor()
    }

    fn executor_id(&self) -> EngineExecutorId {
        self.id
    }

    fn run_slice(&mut self, request: RunRequest<'_>) -> Result<ExecutionReport, EngineFault> {
        let RunRequest {
            cpu,
            memory,
            state,
            instruction_budget,
            loader_return,
            timer,
        } = request;
        let mut shadow = self
            .shadow
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&self.id)
            .unwrap_or_else(|| state.clone());
        shadow.clone_from(state);
        let source = current_location(cpu, &shadow);
        let encoding = memory.fetch32(cpu.address_space_id(), source.pc).ok();
        let result = match encoding.map(|fetched| fetched.bits) {
            Some(bits) if bits & 0xffe0_001f == 0xd400_0001 => {
                self.metrics.normalized_traps.fetch_add(1, Ordering::AcqRel);
                Ok(ExecutionReport {
                    instructions_executed: 1,
                    stop: nixe_cpu_engine::EngineExit::SupervisorCall {
                        source,
                        immediate: (bits >> 5) & 0xffff,
                    },
                    context: shadow.register_context(),
                    trace: InstructionTrace {
                        enabled: false,
                        entries: Box::new([]),
                        discarded: 0,
                    },
                })
            }
            Some(bits) if bits & 0xffe0_001f == 0xd420_0000 => {
                self.metrics.normalized_traps.fetch_add(1, Ordering::AcqRel);
                Ok(ExecutionReport {
                    instructions_executed: 1,
                    stop: nixe_cpu_engine::EngineExit::ArchitecturalException {
                        source,
                        kind: ExceptionKind::Breakpoint,
                        syndrome: Some(u64::from((bits >> 5) & 0xffff)),
                    },
                    context: shadow.register_context(),
                    trace: InstructionTrace {
                        enabled: false,
                        entries: Box::new([]),
                        discarded: 0,
                    },
                })
            }
            _ => self.oracle.run_slice(RunRequest {
                cpu,
                memory,
                state: &mut shadow,
                instruction_budget,
                loader_return,
                timer,
            }),
        };
        if result.is_ok() {
            state.clone_from(&shadow);
        }
        self.shadow
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(self.id, shadow);
        result
    }

    fn synchronize_invalidation(
        &mut self,
        epoch: u64,
        state: &ThreadCpuState,
        memory: &dyn CpuMemory,
    ) -> Result<(), EngineFault> {
        if epoch > self.acknowledged_epoch {
            self.acknowledged_epoch = epoch;
            self.metrics
                .invalidation_syncs
                .fetch_add(1, Ordering::AcqRel);
        }
        self.oracle.synchronize_invalidation(epoch, state, memory)
    }

    fn synchronize_address_space(
        &mut self,
        binding: DomainMemoryBinding<'_>,
        state: &ThreadCpuState,
    ) -> Result<(), EngineFault> {
        let next = snapshot_mappings(binding).map_err(|message| executor_fault(state, message))?;
        publish_mappings(&self.mappings, &self.metrics, next);
        self.synchronize_invalidation(binding.invalidation_generation, state, binding.memory)
    }

    fn control(&self) -> Option<EngineControl> {
        self.oracle.control()
    }

    fn clear_local_exclusive_reservation(&mut self) {
        self.oracle.clear_local_exclusive_reservation();
    }
}

fn executor_fault(state: &ThreadCpuState, message: impl Into<Box<str>>) -> EngineFault {
    EngineFault {
        engine: FAKE_NCE_ENGINE_ID,
        kind: EngineFaultKind::InvalidRequest,
        instructions_executed: 0,
        message: message.into(),
        context: Box::new(state.register_context()),
    }
}

fn descriptor() -> EngineDescriptor {
    EngineDescriptor {
        id: FAKE_NCE_ENGINE_ID,
        name: "synthetic-virtualized-nce".into(),
        kind: EngineKind::NativeCodeExecution,
        capabilities: EngineCapabilities {
            a64: true,
            a32: false,
            t32: false,
            instruction_trace: false,
            interpret_one_fallback: false,
            concurrent_executors: true,
            max_safepoint_instructions: std::num::NonZeroU64::new(1),
            acknowledged_invalidation: true,
            deterministic_execution: true,
            canonical_memory_binding: true,
            max_concurrent_executors: None,
        },
    }
}
