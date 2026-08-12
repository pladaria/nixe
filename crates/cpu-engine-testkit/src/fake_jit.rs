use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use nixe_cpu::location::{ExecutionState, LocationDescriptor};
use nixe_cpu::memory::CodeDependencies;
use nixe_cpu::profile::{CpuProfileId, GuestCpuProfile, ProcessCpuContext};
use nixe_cpu::state::{ThreadCpuState, a64::A64GeneralRegister, a64::A64Register};
use nixe_cpu_engine::{
    CONFORMANCE_FALLBACK_ENCODING, CapabilityRejection, CapabilityRejectionReason,
    CapabilityReport, CrossVcpuRequest, DomainQuiescenceToken, DomainRequest, EngineCapabilities,
    EngineControl, EngineDescriptor, EngineDomain, EngineDomainId, EngineExecutor,
    EngineExecutorId, EngineFault, EngineGeneration, EngineId, EngineKind, EngineProvider,
    ExecutionReport, ExecutorRequest, InstructionTrace, RunRequest, StateCommitStatus,
};
use nixe_cpu_engine_interpreter::InterpreterDomain;
use nixe_memory::{AddressSpaceId, GuestVirtualAddress};

pub const FAKE_JIT_ENGINE_ID: EngineId = EngineId::new(0xf100);

#[derive(Default)]
pub struct FakeJitMetrics {
    compiled_blocks: AtomicU64,
    cache_hits: AtomicU64,
    invalidations: AtomicU64,
}

impl FakeJitMetrics {
    #[must_use]
    pub fn compiled_blocks(&self) -> u64 {
        self.compiled_blocks.load(Ordering::Acquire)
    }

    #[must_use]
    pub fn cache_hits(&self) -> u64 {
        self.cache_hits.load(Ordering::Acquire)
    }

    #[must_use]
    pub fn invalidations(&self) -> u64 {
        self.invalidations.load(Ordering::Acquire)
    }
}

#[derive(Clone, Default)]
pub struct FakeJitProvider {
    metrics: Arc<FakeJitMetrics>,
}

impl FakeJitProvider {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn metrics(&self) -> Arc<FakeJitMetrics> {
        Arc::clone(&self.metrics)
    }
}

impl EngineProvider for FakeJitProvider {
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
                    engine: FAKE_JIT_ENGINE_ID,
                    reason: CapabilityRejectionReason::MissingCapabilities,
                    detail: "fake JIT does not satisfy the requested capability set".into(),
                }])
            },
        }
    }

    fn create_domain(&self, request: DomainRequest) -> Result<Box<dyn EngineDomain>, EngineFault> {
        Ok(Box::new(FakeJitDomain {
            id: request.domain,
            oracle: InterpreterDomain::new(request.domain),
            metrics: Arc::clone(&self.metrics),
            generation: EngineGeneration::new(0),
        }))
    }
}

struct FakeJitDomain {
    id: EngineDomainId,
    oracle: InterpreterDomain,
    metrics: Arc<FakeJitMetrics>,
    generation: EngineGeneration,
}

impl EngineDomain for FakeJitDomain {
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
        let oracle = self.oracle.create_executor(request)?;
        Ok(Box::new(FakeJitExecutor {
            id: request.executor,
            oracle,
            cache: HashMap::new(),
            metrics: Arc::clone(&self.metrics),
            acknowledged_epoch: 0,
        }))
    }

    fn quiesce(&mut self) -> Result<DomainQuiescenceToken, EngineFault> {
        let _ = self.oracle.quiesce()?;
        let token = DomainQuiescenceToken {
            domain: self.id,
            generation: self.generation,
        };
        self.generation = EngineGeneration::new(self.generation.get().saturating_add(1));
        Ok(token)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct BlockKey {
    address_space: AddressSpaceId,
    profile: CpuProfileId,
    pc: GuestVirtualAddress,
    dependencies: CodeDependencies,
}

struct FakeJitExecutor {
    id: EngineExecutorId,
    oracle: Box<dyn EngineExecutor>,
    cache: HashMap<BlockKey, u32>,
    metrics: Arc<FakeJitMetrics>,
    acknowledged_epoch: u64,
}

impl EngineExecutor for FakeJitExecutor {
    fn descriptor(&self) -> EngineDescriptor {
        descriptor()
    }

    fn executor_id(&self) -> EngineExecutorId {
        self.id
    }

    fn run_slice(&mut self, request: RunRequest<'_>) -> Result<ExecutionReport, EngineFault> {
        let source = current_location(request.cpu, request.state);
        let control = self
            .oracle
            .control()
            .expect("the fake JIT oracle provides asynchronous control");
        if let Some(snapshot) = control.take_pending() {
            if snapshot.contains(CrossVcpuRequest::CodeInvalidation) {
                self.cache.clear();
                self.acknowledged_epoch = self.acknowledged_epoch.max(snapshot.invalidation_epoch);
            }
            control.acknowledge(snapshot);
            if snapshot.event_mask != 0 {
                return Ok(empty_report(
                    request.state,
                    nixe_cpu_engine::EngineExit::PendingEvent {
                        mask: snapshot.event_mask,
                    },
                ));
            }
            if [
                CrossVcpuRequest::Preempt,
                CrossVcpuRequest::ProcessStop,
                CrossVcpuRequest::DebuggerStop,
                CrossVcpuRequest::TlbShootdown,
                CrossVcpuRequest::EngineHandoff,
            ]
            .into_iter()
            .any(|request| snapshot.contains(request))
            {
                return Ok(empty_report(
                    request.state,
                    nixe_cpu_engine::EngineExit::Safepoint,
                ));
            }
        }
        if source.execution_state != ExecutionState::A64 {
            return self.oracle.run_slice(request);
        }
        let fetched = request
            .memory
            .fetch32(request.cpu.address_space_id(), source.pc)
            .map_err(|fault| EngineFault {
                engine: FAKE_JIT_ENGINE_ID,
                kind: nixe_cpu_engine::EngineFaultKind::StateImport,
                instructions_executed: 0,
                message: format!("fake JIT block fetch failed: {fault}").into_boxed_str(),
                context: Box::new(request.state.register_context()),
            })?;
        let key = BlockKey {
            address_space: request.cpu.address_space_id(),
            profile: request.cpu.profile().id(),
            pc: source.pc,
            dependencies: fetched.dependencies,
        };
        let cached = match self.cache.entry(key) {
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(fetched.bits);
                self.metrics.compiled_blocks.fetch_add(1, Ordering::AcqRel);
                fetched.bits
            }
            std::collections::hash_map::Entry::Occupied(entry) => {
                self.metrics.cache_hits.fetch_add(1, Ordering::AcqRel);
                *entry.get()
            }
        };
        if cached == CONFORMANCE_FALLBACK_ENCODING {
            return Ok(ExecutionReport {
                instructions_executed: 0,
                stop: nixe_cpu_engine::EngineExit::InterpretOne { source },
                context: request.state.register_context(),
                trace: InstructionTrace {
                    enabled: false,
                    entries: Box::new([]),
                    discarded: 0,
                },
                state_commit: StateCommitStatus::Canonical,
            });
        }
        if matches!(cached, 0xd503_201f | 0x9100_0400) {
            let ThreadCpuState::A64(state) = request.state else {
                unreachable!("the synthetic block path is A64-only");
            };
            if cached == 0x9100_0400 {
                let x0 =
                    A64Register::General(A64GeneralRegister::new(0).expect("X0 is architectural"));
                state.write_x(x0, state.read_x(x0).wrapping_add(1));
            }
            state.set_pc(state.pc().wrapping_add(4));
            return Ok(ExecutionReport {
                instructions_executed: 1,
                stop: nixe_cpu_engine::EngineExit::BudgetExhausted,
                context: request.state.register_context(),
                trace: InstructionTrace {
                    enabled: false,
                    entries: Box::new([]),
                    discarded: 0,
                },
                state_commit: StateCommitStatus::Canonical,
            });
        }
        self.oracle.run_slice(request)
    }

    fn synchronize_invalidation(
        &mut self,
        epoch: u64,
        state: &ThreadCpuState,
        memory: &dyn nixe_cpu::memory::CpuMemory,
    ) -> Result<(), EngineFault> {
        if epoch > self.acknowledged_epoch {
            self.cache.clear();
            self.acknowledged_epoch = epoch;
            self.metrics.invalidations.fetch_add(1, Ordering::AcqRel);
        }
        self.oracle.synchronize_invalidation(epoch, state, memory)
    }

    fn control(&self) -> Option<EngineControl> {
        self.oracle.control()
    }

    fn clear_local_exclusive_reservation(&mut self) {
        self.oracle.clear_local_exclusive_reservation();
    }
}

fn descriptor() -> EngineDescriptor {
    EngineDescriptor {
        id: FAKE_JIT_ENGINE_ID,
        name: "synthetic-block-jit".into(),
        kind: EngineKind::BlockJit,
        capabilities: EngineCapabilities {
            a64: true,
            a32: true,
            t32: true,
            precise_instruction_budget: true,
            instruction_trace: false,
            interpret_one_fallback: true,
            native_execution: false,
            concurrent_executors: true,
            max_safepoint_instructions: std::num::NonZeroU64::new(1),
            acknowledged_invalidation: true,
            canonical_state_version: 1,
            deterministic_execution: true,
            precise_exceptions: true,
            engine_handoff: true,
            canonical_memory_binding: false,
            max_concurrent_executors: None,
        },
    }
}

fn current_location(cpu: ProcessCpuContext, state: &ThreadCpuState) -> LocationDescriptor {
    nixe_cpu::location::current_location(cpu, state)
}

fn empty_report(state: &ThreadCpuState, stop: nixe_cpu_engine::EngineExit) -> ExecutionReport {
    ExecutionReport {
        instructions_executed: 0,
        stop,
        context: state.register_context(),
        trace: InstructionTrace {
            enabled: false,
            entries: Box::new([]),
            discarded: 0,
        },
        state_commit: StateCommitStatus::Canonical,
    }
}
