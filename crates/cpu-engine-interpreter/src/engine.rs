//! Provider, domain, executor, bounded dispatch, and trace implementation.

use std::cell::RefCell;
use std::collections::VecDeque;

use nixe_cpu::decode::{DecodeResult, decode, disassemble};
use nixe_cpu::error::InstructionFetchFault;
use nixe_cpu::exception::ExceptionKind;
use nixe_cpu::location::{
    ExecutionState, InstructionEncoding, LocationDescriptor, current_location,
};
use nixe_cpu::memory::InstructionMemory;
use nixe_cpu::profile::ProcessCpuContext;
use nixe_cpu::state::ThreadCpuState;
use nixe_cpu_engine::{
    CapabilityRejection, CapabilityRejectionReason, CapabilityReport, DomainQuiescenceToken,
    DomainRequest, EngineCapabilities, EngineDescriptor, EngineDomain, EngineDomainId,
    EngineExecutor, EngineExecutorId, EngineFault, EngineFaultKind, EngineGeneration, EngineId,
    EngineKind, EngineProvider, ExecutionReport, ExecutorRequest, InstructionTrace,
    InstructionTraceEntry, MAX_INSTRUCTION_TRACE_ENTRIES, MAX_TRACE_DISASSEMBLY_BYTES, RunRequest,
    SafepointReason, StateCommitStatus, TimerSnapshot, TracePolicy,
};
use nixe_memory::GuestVirtualAddress;

use crate::interpreter::{
    ArchitecturalTimer, ArchitecturalTimerSnapshot, InterpreterContext, InterpreterError,
    InterpreterOutcome, execute_one_with_context,
};

pub const INTERPRETER_ENGINE_ID: EngineId = EngineId::new(1);

#[derive(Clone, Copy, Debug, Default)]
struct DispatchState {
    budget: u64,
}

impl DispatchState {
    const fn budget(self) -> u64 {
        self.budget
    }
    const fn set_budget(&mut self, budget: u64) {
        self.budget = budget;
    }
}

/// Transient state owned by one interpreter executor, never by a guest thread.
struct InterpreterExecutionState {
    exclusive_monitor: RefCell<nixe_cpu::exclusive::ExclusiveMonitorState>,
    control: nixe_cpu_engine::EngineControl,
    dispatch: DispatchState,
}

impl InterpreterExecutionState {
    fn new() -> Self {
        Self {
            exclusive_monitor: RefCell::new(nixe_cpu::exclusive::ExclusiveMonitorState::default()),
            control: nixe_cpu_engine::EngineControl::default(),
            dispatch: DispatchState { budget: 0 },
        }
    }
    const fn dispatch(&self) -> &DispatchState {
        &self.dispatch
    }
    const fn dispatch_mut(&mut self) -> &mut DispatchState {
        &mut self.dispatch
    }
    const fn exclusive_monitor_cell(&self) -> &RefCell<nixe_cpu::exclusive::ExclusiveMonitorState> {
        &self.exclusive_monitor
    }
    fn clear_local_exclusive_reservation(&mut self) {
        *self.exclusive_monitor.get_mut() = nixe_cpu::exclusive::ExclusiveMonitorState::default();
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct InterpreterProvider;

impl EngineProvider for InterpreterProvider {
    fn descriptor(&self) -> EngineDescriptor {
        descriptor()
    }

    fn probe(
        &self,
        profile: nixe_cpu::profile::CpuProfileId,
        required: EngineCapabilities,
    ) -> CapabilityReport {
        let capabilities = capabilities();
        let mut rejections = Vec::new();
        if !capabilities.supports_profile(profile) {
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
        precise_instruction_budget: true,
        instruction_trace: true,
        native_execution: false,
        concurrent_executors: true,
        max_safepoint_instructions: std::num::NonZeroU64::new(1),
        // The interpreter retains neither translated code nor TLB entries.
        acknowledged_invalidation: true,
    }
}

pub struct InterpreterDomain {
    id: EngineDomainId,
    generation: EngineGeneration,
}

impl InterpreterDomain {
    #[must_use]
    pub const fn new(id: EngineDomainId) -> Self {
        Self {
            id,
            generation: EngineGeneration::new(0),
        }
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
        request: ExecutorRequest,
    ) -> Result<Box<dyn EngineExecutor>, EngineFault> {
        Ok(Box::new(InterpreterExecutor::new(
            request.executor,
            request.trace,
        )))
    }

    fn quiesce(&mut self) -> Result<DomainQuiescenceToken, EngineFault> {
        let token = DomainQuiescenceToken {
            domain: self.id,
            generation: self.generation,
        };
        self.generation = EngineGeneration::new(self.generation.get().saturating_add(1));
        Ok(token)
    }
}

pub struct InterpreterExecutor {
    id: EngineExecutorId,
    execution: InterpreterExecutionState,
    trace: TraceRecorder,
}

impl InterpreterExecutor {
    fn new(id: EngineExecutorId, trace: TracePolicy) -> Self {
        Self {
            id,
            execution: InterpreterExecutionState::new(),
            trace: TraceRecorder::new(trace),
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
        self.execution
            .dispatch_mut()
            .set_budget(request.instruction_budget);
        let mut executed = 0_u64;
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
            if let Some(control) = self.execution.control.take_pending() {
                // The interpreter retains no code cache or TLB. Observing the
                // request at this instruction boundary completes every effect.
                self.execution.control.acknowledge(control);
                if control.event_mask != 0 {
                    return Ok(self.report(
                        executed,
                        nixe_cpu_engine::EngineExit::PendingEvent {
                            mask: control.event_mask,
                        },
                        request.state,
                    ));
                }
                let must_stop = [
                    nixe_cpu_engine::CrossVcpuRequest::Preempt,
                    nixe_cpu_engine::CrossVcpuRequest::ProcessStop,
                    nixe_cpu_engine::CrossVcpuRequest::DebuggerStop,
                    nixe_cpu_engine::CrossVcpuRequest::EngineHandoff,
                ]
                .into_iter()
                .any(|request| control.contains(request));
                if must_stop {
                    return Ok(self.report(
                        executed,
                        nixe_cpu_engine::EngineExit::Safepoint,
                        request.state,
                    ));
                }
                continue;
            }
            if self.execution.dispatch().budget() == 0 {
                return Ok(self.report(
                    executed,
                    nixe_cpu_engine::EngineExit::BudgetExhausted,
                    request.state,
                ));
            }
            let encoding = match fetch_current(request.memory, request.cpu, request.state) {
                Ok(encoding) => encoding,
                Err(fault) => {
                    return Ok(self.report(
                        executed,
                        nixe_cpu_engine::EngineExit::FetchFault { fault },
                        request.state,
                    ));
                }
            };
            let source = current_location(request.cpu, request.state);
            let timer = InterpreterTimer(request.timer);
            let context = InterpreterContext::new(request.cpu)
                .with_memory(request.memory)
                .with_exclusive_monitor(self.execution.exclusive_monitor_cell())
                .with_architectural_timer_provider(&timer);
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
            self.trace.record(request.cpu, source, encoding);
            executed += 1;
            let remaining = self.execution.dispatch().budget() - 1;
            self.execution.dispatch_mut().set_budget(remaining);
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
                InterpreterOutcome::Scheduled { source } => {
                    nixe_cpu_engine::EngineExit::Scheduled { source }
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
        self.execution.clear_local_exclusive_reservation();
    }

    fn request_safepoint(&mut self, reason: SafepointReason) {
        let request = match reason {
            SafepointReason::Requested | SafepointReason::Timer => {
                nixe_cpu_engine::CrossVcpuRequest::Preempt
            }
            SafepointReason::PendingEvent { mask } => {
                self.execution.control.post_event(mask);
                return;
            }
            SafepointReason::MappingChanged => nixe_cpu_engine::CrossVcpuRequest::TlbShootdown,
            SafepointReason::EngineHandoff => nixe_cpu_engine::CrossVcpuRequest::EngineHandoff,
        };
        let _ = self.execution.control.request(request);
    }
    fn post_event(&self, mask: u32) {
        self.execution.control.post_event(mask);
    }

    fn synchronize_invalidation(
        &mut self,
        epoch: u64,
        _state: &ThreadCpuState,
    ) -> Result<(), EngineFault> {
        // The interpreter retains no translation cache or TLB.
        self.execution.control.acknowledge_invalidation(epoch);
        Ok(())
    }
    fn control(&self) -> Option<nixe_cpu_engine::EngineControl> {
        Some(self.execution.control.clone())
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
            trace: self.trace.snapshot(),
            state_commit: StateCommitStatus::Canonical,
        }
    }
}

struct InterpreterTimer<'a>(&'a dyn nixe_cpu_engine::EngineTimer);
impl ArchitecturalTimer for InterpreterTimer<'_> {
    fn snapshot(&self) -> ArchitecturalTimerSnapshot {
        let TimerSnapshot { counter, frequency } = self.0.snapshot();
        ArchitecturalTimerSnapshot { counter, frequency }
    }
}

struct TraceRecorder {
    policy: TracePolicy,
    entries: VecDeque<InstructionTraceEntry>,
    next_sequence: u64,
    discarded: u64,
}
impl TraceRecorder {
    fn new(policy: TracePolicy) -> Self {
        Self {
            policy,
            entries: VecDeque::new(),
            next_sequence: 0,
            discarded: 0,
        }
    }
    fn record(
        &mut self,
        cpu: ProcessCpuContext,
        source: LocationDescriptor,
        encoding: InstructionEncoding,
    ) {
        if !self.policy.enabled {
            return;
        }
        if self.entries.len() == MAX_INSTRUCTION_TRACE_ENTRIES {
            self.entries.pop_front();
            self.discarded = self.discarded.saturating_add(1);
        }
        let disassembly = self
            .policy
            .detailed
            .then(|| instruction_description(cpu, source, encoding));
        self.entries.push_back(InstructionTraceEntry {
            sequence: self.next_sequence,
            source,
            encoding,
            disassembly,
        });
        self.next_sequence = self.next_sequence.saturating_add(1);
    }
    fn snapshot(&self) -> InstructionTrace {
        InstructionTrace {
            enabled: self.policy.enabled,
            entries: self.entries.iter().cloned().collect(),
            discarded: self.discarded,
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

fn fetch_current(
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

fn instruction_description(
    cpu: ProcessCpuContext,
    source: LocationDescriptor,
    encoding: InstructionEncoding,
) -> Box<str> {
    let description = match decode(&cpu.profile(), source, encoding) {
        DecodeResult::Decoded(decoded) | DecodeResult::RecognizedUnimplemented(decoded) => {
            disassemble(&decoded.instruction).to_string()
        }
        DecodeResult::Unallocated { reason, .. } => format!("<unallocated: {reason}>"),
        DecodeResult::Reserved { name, reason, .. } => format!("<{name}: reserved: {reason}>"),
        DecodeResult::ProfileDisabled {
            name, rejection, ..
        } => format!("<{name}: profile-disabled: {rejection}>"),
    };
    truncate_utf8(description, MAX_TRACE_DISASSEMBLY_BYTES).into()
}

fn truncate_utf8(mut value: String, maximum_bytes: usize) -> String {
    if value.len() <= maximum_bytes {
        return value;
    }
    let mut boundary = maximum_bytes;
    while !value.is_char_boundary(boundary) {
        boundary -= 1;
    }
    value.truncate(boundary);
    value
}
