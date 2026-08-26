//! Provider, domain, executor, bounded dispatch, and trace implementation.

use std::cell::RefCell;

use nixe_cpu::error::InstructionFetchFault;
use nixe_cpu::exception::ExceptionKind;
use nixe_cpu::location::{
    ExecutionState, InstructionEncoding, LocationDescriptor, current_location,
};
use nixe_cpu::memory::InstructionMemory;
use nixe_cpu::profile::ProcessCpuContext;
use nixe_cpu::state::ThreadCpuState;
use nixe_cpu_engine::{
    CapabilityRejection, CapabilityRejectionReason, CapabilityReport, DomainRequest,
    EngineCapabilities, EngineDescriptor, EngineDomain, EngineDomainId, EngineExecutor,
    EngineExecutorId, EngineFault, EngineFaultKind, EngineId, EngineKind, EngineProvider,
    ExecutionReport, RunRequest,
};
use nixe_memory::GuestVirtualAddress;

use crate::interpreter::{
    InterpreterContext, InterpreterError, InterpreterOutcome, execute_one_with_context,
};

pub const INTERPRETER_ENGINE_ID: EngineId = EngineId::new(1);

#[derive(Clone, Copy, Debug, Default)]
pub struct InterpreterProvider;

impl EngineProvider for InterpreterProvider {
    fn descriptor(&self) -> EngineDescriptor {
        descriptor()
    }

    fn probe(
        &self,
        profile: nixe_cpu::profile::GuestCpuProfile,
        required: EngineCapabilities,
    ) -> CapabilityReport {
        let capabilities = capabilities();
        let mut rejections = Vec::new();
        if !capabilities.supports_profile(profile, required) {
            rejections.push(CapabilityRejection {
                engine: INTERPRETER_ENGINE_ID,
                reason: CapabilityRejectionReason::GuestProfileUnsupported,
                detail: "guest profile is unsupported".into(),
            });
        }
        if !capabilities.contains(required) {
            rejections.push(CapabilityRejection {
                engine: INTERPRETER_ENGINE_ID,
                reason: CapabilityRejectionReason::MissingCapabilities,
                detail: "required capability set is unavailable".into(),
            });
        }
        CapabilityReport {
            descriptor: descriptor(),
            available: rejections.is_empty(),
            rejections: rejections.into_boxed_slice(),
        }
    }

    fn create_domain(&self, request: DomainRequest) -> Result<Box<dyn EngineDomain>, EngineFault> {
        let _ = request.cpu;
        Ok(Box::new(InterpreterDomain::new(request.domain)))
    }
}

fn descriptor() -> EngineDescriptor {
    EngineDescriptor {
        id: INTERPRETER_ENGINE_ID,
        name: "reference-interpreter".into(),
        kind: EngineKind::Interpreter,
        capabilities: capabilities(),
    }
}

const fn capabilities() -> EngineCapabilities {
    EngineCapabilities {
        a64: true,
        a32: true,
        t32: true,
        interpret_one_fallback: false,
        concurrent_executors: true,
        max_safepoint_instructions: std::num::NonZeroU64::new(1),
        // The interpreter retains neither translated code nor TLB entries.
        acknowledged_invalidation: true,
        deterministic_execution: true,
        canonical_memory_binding: false,
        max_concurrent_executors: None,
    }
}

pub struct InterpreterDomain {
    id: EngineDomainId,
}

impl InterpreterDomain {
    #[must_use]
    pub const fn new(id: EngineDomainId) -> Self {
        Self { id }
    }
}

impl EngineDomain for InterpreterDomain {
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
        Ok(Box::new(InterpreterExecutor::new(executor)))
    }
}

pub struct InterpreterExecutor {
    id: EngineExecutorId,
    exclusive_monitor: RefCell<nixe_cpu::exclusive::ExclusiveMonitorState>,
    control: nixe_cpu_engine::EngineControl,
}

impl InterpreterExecutor {
    fn new(id: EngineExecutorId) -> Self {
        Self {
            id,
            exclusive_monitor: RefCell::new(Default::default()),
            control: nixe_cpu_engine::EngineControl::default(),
        }
    }
}

impl EngineExecutor for InterpreterExecutor {
    fn descriptor(&self) -> EngineDescriptor {
        descriptor()
    }

    fn executor_id(&self) -> EngineExecutorId {
        self.id
    }

    fn run_slice(&mut self, request: RunRequest<'_>) -> Result<ExecutionReport, EngineFault> {
        let mut remaining = request.instruction_budget;
        let mut executed = 0_u64;
        let context = InterpreterContext::new(request.cpu)
            .with_memory(request.memory)
            .with_exclusive_monitor(&self.exclusive_monitor)
            .with_architectural_timer_provider(request.timer)
            .with_vcpu_events(&request.events);
        loop {
            if let Some((source, result_code)) =
                loader_return_observation(request.cpu, request.state, request.loader_return)
            {
                return Ok(self.report(
                    executed,
                    nixe_cpu_engine::EngineExit::LoaderReturn {
                        source,
                        result_code,
                    },
                    request.state,
                ));
            }
            let pending_interrupts = request.events.take_pending_interrupts();
            if pending_interrupts != 0 {
                return Ok(self.report(
                    executed,
                    nixe_cpu_engine::EngineExit::PendingEvent {
                        mask: pending_interrupts,
                    },
                    request.state,
                ));
            }
            if let Some(control) = self.control.take_pending() {
                // The interpreter retains no code cache or TLB. Observing the
                // request at this instruction boundary completes every effect.
                self.control.acknowledge(control);
                if control.contains(nixe_cpu_engine::CrossVcpuRequest::Preempt) {
                    return Ok(self.report(
                        executed,
                        nixe_cpu_engine::EngineExit::Safepoint,
                        request.state,
                    ));
                }
                continue;
            }
            if remaining == 0 {
                return Ok(self.report(
                    executed,
                    nixe_cpu_engine::EngineExit::BudgetExhausted,
                    request.state,
                ));
            }
            let encoding =
                match fetch_current_instruction(request.memory, request.cpu, request.state) {
                    Ok(encoding) => encoding,
                    Err(fault) => {
                        return Ok(self.report(
                            executed,
                            nixe_cpu_engine::EngineExit::FetchFault { fault },
                            request.state,
                        ));
                    }
                };
            let outcome = match execute_one_with_context(context, request.state, encoding) {
                Ok(outcome) => outcome,
                Err(InterpreterError::UnsupportedInstruction {
                    source,
                    encoding,
                    disassembly,
                    coverage_id,
                }) => {
                    return Ok(self.report(
                        executed,
                        nixe_cpu_engine::EngineExit::UnsupportedSemantics {
                            source,
                            encoding,
                            disassembly,
                            coverage_id,
                        },
                        request.state,
                    ));
                }
                Err(error) => {
                    return Err(EngineFault {
                        engine: INTERPRETER_ENGINE_ID,
                        kind: EngineFaultKind::Internal,
                        instructions_executed: executed,
                        message: error.to_string().into(),
                        context: Box::new(request.state.register_context()),
                    });
                }
            };
            executed += 1;
            remaining -= 1;
            let stop = match outcome {
                InterpreterOutcome::Resume(_) => continue,
                InterpreterOutcome::Exception {
                    source,
                    kind: ExceptionKind::SupervisorCall,
                    syndrome: Some(syndrome),
                } if let Ok(immediate) = u32::try_from(syndrome) => {
                    nixe_cpu_engine::EngineExit::SupervisorCall { source, immediate }
                }
                InterpreterOutcome::Exception {
                    source,
                    kind,
                    syndrome,
                } => nixe_cpu_engine::EngineExit::ArchitecturalException {
                    source,
                    kind,
                    syndrome,
                },
                InterpreterOutcome::Scheduled { source, request } => {
                    nixe_cpu_engine::EngineExit::Scheduled { source, request }
                }
                InterpreterOutcome::DataAbort { source, fault } => {
                    nixe_cpu_engine::EngineExit::DataFault { source, fault }
                }
                InterpreterOutcome::ProfileDisabled(error) => {
                    nixe_cpu_engine::EngineExit::ProfileDisabled { error }
                }
                InterpreterOutcome::Unallocated(error) => {
                    nixe_cpu_engine::EngineExit::UnallocatedEncoding { error }
                }
            };
            return Ok(self.report(executed, stop, request.state));
        }
    }

    fn clear_local_exclusive_reservation(&mut self) {
        *self.exclusive_monitor.get_mut() = Default::default();
    }

    fn synchronize_invalidation(
        &mut self,
        cursor: nixe_memory::MemoryInvalidationCursor,
        _state: &ThreadCpuState,
        _memory: &dyn nixe_cpu::memory::CpuMemory,
    ) -> Result<(), EngineFault> {
        // The interpreter retains no translation cache or TLB.
        self.control.acknowledge_invalidation(cursor.get());
        Ok(())
    }
    fn control(&self) -> Option<nixe_cpu_engine::EngineControl> {
        Some(self.control.clone())
    }
}

impl InterpreterExecutor {
    fn report(
        &self,
        instructions_executed: u64,
        stop: nixe_cpu_engine::EngineExit,
        state: &ThreadCpuState,
    ) -> ExecutionReport {
        ExecutionReport {
            instructions_executed,
            stop,
            context: state.register_context(),
        }
    }
}

fn loader_return_observation(
    cpu: ProcessCpuContext,
    state: &ThreadCpuState,
    loader_return: Option<GuestVirtualAddress>,
) -> Option<(LocationDescriptor, u64)> {
    let return_address = loader_return?;
    let ThreadCpuState::A64(state) = state else {
        return None;
    };
    (state.pc() == return_address.get()).then(|| {
        let source =
            LocationDescriptor::new(return_address, ExecutionState::A64, cpu.profile().id());
        let result_code = state.read_x(nixe_cpu::state::a64::A64Register::General(
            nixe_cpu::state::a64::A64GeneralRegister::new(0).expect("valid result register"),
        ));
        (source, result_code)
    })
}

pub fn fetch_current_instruction(
    memory: &dyn InstructionMemory,
    cpu: ProcessCpuContext,
    state: &ThreadCpuState,
) -> Result<InstructionEncoding, InstructionFetchFault> {
    let location = current_location(cpu, state);
    let address = location.pc;
    let address_space = cpu.address_space_id();
    match location.execution_state {
        ExecutionState::A64 | ExecutionState::A32 => memory
            .fetch32(address_space, address)
            .map(|fetched| InstructionEncoding::from_u32(fetched.bits)),
        ExecutionState::T32 => {
            let first = memory.fetch16(address_space, address)?;
            if location.execution_state.instruction_size(first.bits)
                == nixe_cpu::location::InstructionSize::Bits16
            {
                Ok(InstructionEncoding::from_u16(first.bits))
            } else {
                memory
                    .fetch_t32_32(address_space, address)
                    .map(|fetched| InstructionEncoding::from_u32(fetched.bits))
            }
        }
    }
}
