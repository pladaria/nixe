//! Reusable black-box conformance checks for CPU engine providers.

use std::fmt::{Display, Formatter};
use std::sync::Arc;

use nixe_cpu::location::ExecutionState;
use nixe_cpu::memory::{
    CpuMemory, ExecutionMemory, MemoryAccess, MemoryAccessSize, MemoryPermissions, MemoryValue,
    SyntheticMemory,
};
use nixe_cpu::profile::{GuestCpuProfile, ProcessCpuContext};
use nixe_cpu::state::{ThreadCpuState, a64::A64GeneralRegister, a64::A64Register, a64::A64State};
use nixe_memory::GuestPhysicalPageId;
use nixe_memory::{AddressSpaceId, GuestVirtualAddress};

use crate::{
    CrossVcpuRequest, DomainRequest, EngineDomain, EngineExecutor, EngineExecutorId, EngineExit,
    EngineProvider, EngineTimer, RunRequest, TimerSnapshot,
};

const SPACE: AddressSpaceId = AddressSpaceId::new(1);
const CODE: GuestVirtualAddress = GuestVirtualAddress::new(0x1000);
const DATA: GuestVirtualAddress = GuestVirtualAddress::new(0x2000);
const CODE_PAGE: GuestPhysicalPageId = GuestPhysicalPageId::new(1);
const DATA_PAGE: GuestPhysicalPageId = GuestPhysicalPageId::new(2);
pub const CONFORMANCE_FALLBACK_ENCODING: u32 = 0xd420_0000;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ConformanceCase {
    CapabilityTruthfulness,
    CanonicalState,
    PreciseException,
    InterpretOneFallback,
    Timer,
    Interrupt,
    Invalidation,
    SelfModifyingCode,
    Atomics,
    Migration,
    Cancellation,
    Teardown,
    DeterministicExecution,
    ConcurrentOwnership,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConformanceFailure {
    pub case: ConformanceCase,
    pub detail: Box<str>,
}

impl Display for ConformanceFailure {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "engine conformance {:?} failed: {}",
            self.case, self.detail
        )
    }
}

impl std::error::Error for ConformanceFailure {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConformanceReport {
    pub engine: crate::EngineId,
    pub passed: Box<[ConformanceCase]>,
    pub skipped: Box<[ConformanceCase]>,
}

/// Runs the portable provider/domain/executor contract against an A64 fixture.
/// Capability-dependent cases are skipped only when the descriptor declines
/// that capability; an advertised capability must pass its corresponding case.
pub fn run_provider_conformance(
    provider: Arc<dyn EngineProvider>,
) -> Result<ConformanceReport, ConformanceFailure> {
    let descriptor = provider.descriptor();
    let mut suite = Suite {
        provider,
        cpu: ProcessCpuContext::new(GuestCpuProfile::switch_1(), SPACE),
        binding_memory: ExecutionMemory::new(),
        next_domain: 1,
        next_executor: 1,
        passed: Vec::new(),
        skipped: Vec::new(),
    };
    suite.capability_truthfulness()?;
    suite.canonical_state()?;
    suite.precise_exception()?;
    suite.interpret_one_fallback()?;
    suite.timer()?;
    suite.interrupt()?;
    suite.invalidation()?;
    suite.self_modifying_code()?;
    suite.atomics()?;
    suite.migration()?;
    suite.cancellation()?;
    suite.teardown()?;
    suite.deterministic_execution()?;
    suite.concurrent_ownership()?;
    Ok(ConformanceReport {
        engine: descriptor.id,
        passed: suite.passed.into_boxed_slice(),
        skipped: suite.skipped.into_boxed_slice(),
    })
}

struct Suite {
    provider: Arc<dyn EngineProvider>,
    cpu: ProcessCpuContext,
    binding_memory: ExecutionMemory,
    next_domain: u64,
    next_executor: u64,
    passed: Vec<ConformanceCase>,
    skipped: Vec<ConformanceCase>,
}

type FreshExecutor = (Box<dyn EngineDomain>, Box<dyn EngineExecutor>);

impl Suite {
    fn fail(case: ConformanceCase, detail: impl Into<Box<str>>) -> ConformanceFailure {
        ConformanceFailure {
            case,
            detail: detail.into(),
        }
    }

    fn domain(
        &mut self,
        case: ConformanceCase,
    ) -> Result<Box<dyn EngineDomain>, ConformanceFailure> {
        let id = crate::EngineDomainId::new(self.next_domain);
        self.next_domain = self.next_domain.saturating_add(1);
        let mut domain = self
            .provider
            .create_domain(DomainRequest {
                domain: id,
                cpu: self.cpu,
            })
            .map_err(|error| Self::fail(case, error.to_string()))?;
        let binding = crate::DomainMemoryBinding {
            address_space: self.cpu.address_space_id(),
            end_exclusive: GuestVirtualAddress::new(1_u64 << 39),
            memory: &self.binding_memory,
            invalidation_generation: self.binding_memory.mapping_epoch().get(),
        };
        domain
            .bind_memory(binding)
            .map_err(|error| Self::fail(case, error.to_string()))?;
        if domain.domain_id() != id {
            return Err(Self::fail(
                case,
                "domain did not preserve its requested identity",
            ));
        }
        if domain.descriptor() != self.provider.descriptor() {
            return Err(Self::fail(case, "provider and domain descriptors differ"));
        }
        Ok(domain)
    }

    fn executor(
        &mut self,
        domain: &mut dyn EngineDomain,
        case: ConformanceCase,
    ) -> Result<Box<dyn EngineExecutor>, ConformanceFailure> {
        let id = EngineExecutorId::new(self.next_executor);
        self.next_executor = self.next_executor.saturating_add(1);
        let executor = domain
            .create_executor(id)
            .map_err(|error| Self::fail(case, error.to_string()))?;
        if executor.executor_id() != id {
            return Err(Self::fail(
                case,
                "executor did not preserve its requested identity",
            ));
        }
        if executor.descriptor() != self.provider.descriptor() {
            return Err(Self::fail(case, "provider and executor descriptors differ"));
        }
        Ok(executor)
    }

    fn capability_truthfulness(&mut self) -> Result<(), ConformanceFailure> {
        let case = ConformanceCase::CapabilityTruthfulness;
        let descriptor = self.provider.descriptor();
        if !descriptor.capabilities.is_coherent() {
            return Err(Self::fail(
                case,
                "descriptor contains an internally inconsistent capability set",
            ));
        }
        let report = self
            .provider
            .probe(self.cpu.profile(), descriptor.capabilities);
        if !report.available || report.descriptor != descriptor || !report.rejections.is_empty() {
            return Err(Self::fail(
                case,
                "provider rejected or changed its advertised capability set",
            ));
        }
        for (advertised, required) in [
            (
                descriptor.capabilities.a64,
                crate::EngineCapabilities {
                    a64: true,
                    ..Default::default()
                },
            ),
            (
                descriptor.capabilities.a32,
                crate::EngineCapabilities {
                    a32: true,
                    ..Default::default()
                },
            ),
            (
                descriptor.capabilities.t32,
                crate::EngineCapabilities {
                    t32: true,
                    ..Default::default()
                },
            ),
        ] {
            let available = self.provider.probe(self.cpu.profile(), required).available;
            if available != advertised {
                return Err(Self::fail(
                    case,
                    "execution-state probe disagrees with the descriptor",
                ));
            }
        }
        let mut domain = self.domain(case)?;
        let executor = self.executor(domain.as_mut(), case)?;
        if descriptor.capabilities.requires_control_path() && executor.control().is_none() {
            return Err(Self::fail(
                case,
                "advertised asynchronous controls have no control path",
            ));
        }
        drop(executor);
        domain
            .shutdown()
            .map_err(|error| Self::fail(case, error.to_string()))?;
        self.passed.push(case);
        Ok(())
    }

    fn canonical_state(&mut self) -> Result<(), ConformanceFailure> {
        let case = ConformanceCase::CanonicalState;
        let (mut domain, mut executor) = self.fresh_executor(case)?;
        let memory = fixture_memory(&[0xd503_201f], 7);
        let mut state = state_at(CODE);
        let report = run(self.cpu, executor.as_mut(), &memory, &mut state, 1)
            .map_err(|error| Self::fail(case, error))?;
        if report.instructions_executed != 1
            || report.stop != EngineExit::BudgetExhausted
            || report.context != state.register_context()
            || instruction_address(&state) != CODE.get() + 4
        {
            return Err(Self::fail(
                case,
                "one-instruction state commit was not exact",
            ));
        }
        drop(executor);
        domain
            .shutdown()
            .map_err(|error| Self::fail(case, error.to_string()))?;
        self.passed.push(case);
        Ok(())
    }

    fn precise_exception(&mut self) -> Result<(), ConformanceFailure> {
        let case = ConformanceCase::PreciseException;
        let (mut domain, mut executor) = self.fresh_executor(case)?;
        let memory = fixture_memory(&[0xd400_0841], 0);
        let mut state = state_at(CODE);
        let report = run(self.cpu, executor.as_mut(), &memory, &mut state, 1)
            .map_err(|error| Self::fail(case, error))?;
        if !matches!(report.stop, EngineExit::SupervisorCall { source, immediate: 0x42 } if source.pc == CODE)
            || report.instructions_executed != 1
            || instruction_address(&state) != CODE.get()
        {
            return Err(Self::fail(
                case,
                "SVC exit was not precise at its source instruction",
            ));
        }
        drop(executor);
        domain
            .shutdown()
            .map_err(|error| Self::fail(case, error.to_string()))?;
        self.passed.push(case);
        Ok(())
    }

    fn interpret_one_fallback(&mut self) -> Result<(), ConformanceFailure> {
        let case = ConformanceCase::InterpretOneFallback;
        if !self
            .provider
            .descriptor()
            .capabilities
            .interpret_one_fallback
        {
            self.skipped.push(case);
            return Ok(());
        }
        let (mut domain, mut executor) = self.fresh_executor(case)?;
        let memory = fixture_memory(&[CONFORMANCE_FALLBACK_ENCODING], 0);
        let mut state = state_at(CODE);
        let report = run(self.cpu, executor.as_mut(), &memory, &mut state, 1)
            .map_err(|error| Self::fail(case, error))?;
        if report.stop
            != (EngineExit::InterpretOne {
                source: location(self.cpu),
            })
            || report.instructions_executed != 0
            || instruction_address(&state) != CODE.get()
        {
            return Err(Self::fail(
                case,
                "fallback exit did not preserve its canonical boundary",
            ));
        }
        drop(executor);
        domain
            .shutdown()
            .map_err(|error| Self::fail(case, error.to_string()))?;
        self.passed.push(case);
        Ok(())
    }

    fn timer(&mut self) -> Result<(), ConformanceFailure> {
        let case = ConformanceCase::Timer;
        let (mut domain, mut executor) = self.fresh_executor(case)?;
        let memory = fixture_memory(&[0xd53b_e020], 0);
        let mut state = state_at(CODE);
        run(self.cpu, executor.as_mut(), &memory, &mut state, 1)
            .map_err(|error| Self::fail(case, error))?;
        let ThreadCpuState::A64(state) = state else {
            unreachable!();
        };
        if state.read_x(A64Register::General(register(0))) != FixedTimer.snapshot().counter {
            return Err(Self::fail(
                case,
                "architectural timer did not use the supplied timer",
            ));
        }
        drop(executor);
        domain
            .shutdown()
            .map_err(|error| Self::fail(case, error.to_string()))?;
        self.passed.push(case);
        Ok(())
    }

    fn interrupt(&mut self) -> Result<(), ConformanceFailure> {
        let case = ConformanceCase::Interrupt;
        let (mut domain, mut executor) = self.fresh_executor(case)?;
        let Some(control) = executor.control() else {
            self.skipped.push(case);
            return Ok(());
        };
        control.post_event(0x20);
        let memory = fixture_memory(&[0xd503_201f], 0);
        let mut state = state_at(CODE);
        let report = run(self.cpu, executor.as_mut(), &memory, &mut state, 1)
            .map_err(|error| Self::fail(case, error))?;
        if report.stop != (EngineExit::PendingEvent { mask: 0x20 })
            || report.instructions_executed != 0
        {
            return Err(Self::fail(
                case,
                "pending event was not observed before execution",
            ));
        }
        drop(executor);
        domain
            .shutdown()
            .map_err(|error| Self::fail(case, error.to_string()))?;
        self.passed.push(case);
        Ok(())
    }

    fn invalidation(&mut self) -> Result<(), ConformanceFailure> {
        let case = ConformanceCase::Invalidation;
        if !self
            .provider
            .descriptor()
            .capabilities
            .acknowledged_invalidation
        {
            self.skipped.push(case);
            return Ok(());
        }
        let (mut domain, mut executor) = self.fresh_executor(case)?;
        let control = executor
            .control()
            .ok_or_else(|| Self::fail(case, "invalidation capability has no control path"))?;
        let mut state = state_at(CODE);
        control.request_invalidation(9);
        executor
            .synchronize_invalidation(9, &state, &fixture_memory(&[0xd503_201f], 0))
            .map_err(|error| Self::fail(case, error.to_string()))?;
        if !control.acknowledged_invalidation(9) {
            return Err(Self::fail(case, "synchronized epoch was not acknowledged"));
        }
        let memory = fixture_memory(&[0xd503_201f], 0);
        run(self.cpu, executor.as_mut(), &memory, &mut state, 1)
            .map_err(|error| Self::fail(case, error))?;
        drop(executor);
        domain
            .shutdown()
            .map_err(|error| Self::fail(case, error.to_string()))?;
        self.passed.push(case);
        Ok(())
    }

    fn self_modifying_code(&mut self) -> Result<(), ConformanceFailure> {
        let case = ConformanceCase::SelfModifyingCode;
        if !self
            .provider
            .descriptor()
            .capabilities
            .acknowledged_invalidation
        {
            self.skipped.push(case);
            return Ok(());
        }
        let (mut domain, mut executor) = self.fresh_executor(case)?;
        let mut memory = fixture_memory(&[0xd503_201f], 0);
        let mut state = state_at(CODE);
        run(self.cpu, executor.as_mut(), &memory, &mut state, 1)
            .map_err(|error| Self::fail(case, error))?;
        assert!(memory.initialize_ram(CODE_PAGE, 0, &0x9100_0400_u32.to_le_bytes()));
        set_instruction_address(&mut state, CODE);
        executor
            .synchronize_invalidation(10, &state, &memory)
            .map_err(|error| Self::fail(case, error.to_string()))?;
        run(self.cpu, executor.as_mut(), &memory, &mut state, 1)
            .map_err(|error| Self::fail(case, error))?;
        let ThreadCpuState::A64(state) = state else {
            unreachable!();
        };
        if state.read_x(A64Register::General(register(0))) != 1 {
            return Err(Self::fail(
                case,
                "executor reused stale code after invalidation",
            ));
        }
        drop(executor);
        domain
            .shutdown()
            .map_err(|error| Self::fail(case, error.to_string()))?;
        self.passed.push(case);
        Ok(())
    }

    fn atomics(&mut self) -> Result<(), ConformanceFailure> {
        let case = ConformanceCase::Atomics;
        let (mut domain, mut executor) = self.fresh_executor(case)?;
        let memory = fixture_memory(&[0x885f_fc60, 0x8801_fc60], 11);
        let mut state = state_at(CODE);
        let ThreadCpuState::A64(a64) = &mut state else {
            unreachable!();
        };
        a64.write_x(A64Register::General(register(3)), DATA.get());
        run(self.cpu, executor.as_mut(), &memory, &mut state, 1)
            .map_err(|error| Self::fail(case, error))?;
        run(self.cpu, executor.as_mut(), &memory, &mut state, 1)
            .map_err(|error| Self::fail(case, error))?;
        let ThreadCpuState::A64(a64) = state else {
            unreachable!();
        };
        let value = memory
            .read(SPACE, DATA, MemoryAccess::normal(MemoryAccessSize::Word))
            .map_err(|error| Self::fail(case, format!("{error:?}")))?;
        if a64.read_w(A64Register::General(register(1))) != 0 || value.value != MemoryValue::U32(11)
        {
            return Err(Self::fail(
                case,
                "exclusive load/store did not commit atomically",
            ));
        }
        drop(executor);
        domain
            .shutdown()
            .map_err(|error| Self::fail(case, error.to_string()))?;
        self.passed.push(case);
        Ok(())
    }

    fn migration(&mut self) -> Result<(), ConformanceFailure> {
        let case = ConformanceCase::Migration;
        let mut domain = self.domain(case)?;
        let mut first = self.executor(domain.as_mut(), case)?;
        let mut second = self.executor(domain.as_mut(), case)?;
        let memory = fixture_memory(&[0x9100_0400, 0x9100_0400], 0);
        let mut state = state_at(CODE);
        run(self.cpu, first.as_mut(), &memory, &mut state, 1)
            .map_err(|error| Self::fail(case, error))?;
        first.clear_local_exclusive_reservation();
        run(self.cpu, second.as_mut(), &memory, &mut state, 1)
            .map_err(|error| Self::fail(case, error))?;
        let ThreadCpuState::A64(state) = state else {
            unreachable!();
        };
        if state.read_x(A64Register::General(register(0))) != 2 {
            return Err(Self::fail(
                case,
                "canonical state did not migrate between executors",
            ));
        }
        drop((first, second));
        domain
            .shutdown()
            .map_err(|error| Self::fail(case, error.to_string()))?;
        self.passed.push(case);
        Ok(())
    }

    fn cancellation(&mut self) -> Result<(), ConformanceFailure> {
        let case = ConformanceCase::Cancellation;
        let (mut domain, mut executor) = self.fresh_executor(case)?;
        let Some(control) = executor.control() else {
            self.skipped.push(case);
            return Ok(());
        };
        control.request(CrossVcpuRequest::Preempt);
        let memory = fixture_memory(&[0xd503_201f], 0);
        let mut state = state_at(CODE);
        let report = run(self.cpu, executor.as_mut(), &memory, &mut state, 1)
            .map_err(|error| Self::fail(case, error))?;
        if report.stop != EngineExit::Safepoint || report.instructions_executed != 0 {
            return Err(Self::fail(
                case,
                "preemption did not stop at the next bounded poll",
            ));
        }
        drop(executor);
        domain
            .shutdown()
            .map_err(|error| Self::fail(case, error.to_string()))?;
        self.passed.push(case);
        Ok(())
    }

    fn teardown(&mut self) -> Result<(), ConformanceFailure> {
        let case = ConformanceCase::Teardown;
        let mut domain = self.domain(case)?;
        let executor = self.executor(domain.as_mut(), case)?;
        drop(executor);
        domain
            .shutdown()
            .and_then(|()| domain.shutdown())
            .map_err(|error| Self::fail(case, error.to_string()))?;
        self.passed.push(case);
        Ok(())
    }

    fn deterministic_execution(&mut self) -> Result<(), ConformanceFailure> {
        let case = ConformanceCase::DeterministicExecution;
        let mut observations = Vec::new();
        for _ in 0..2 {
            let (mut domain, mut executor) = self.fresh_executor(case)?;
            let memory = fixture_memory(&[0x9100_0400], 0);
            let mut state = state_at(CODE);
            let report = run(self.cpu, executor.as_mut(), &memory, &mut state, 1)
                .map_err(|error| Self::fail(case, error))?;
            observations.push((state, report));
            drop(executor);
            domain
                .shutdown()
                .map_err(|error| Self::fail(case, error.to_string()))?;
        }
        if observations[0] != observations[1] {
            return Err(Self::fail(case, "identical deterministic runs diverged"));
        }
        self.passed.push(case);
        Ok(())
    }

    fn concurrent_ownership(&mut self) -> Result<(), ConformanceFailure> {
        let case = ConformanceCase::ConcurrentOwnership;
        if !self.provider.descriptor().capabilities.concurrent_executors {
            self.skipped.push(case);
            return Ok(());
        }
        let mut domain = self.domain(case)?;
        let executors = [
            self.executor(domain.as_mut(), case)?,
            self.executor(domain.as_mut(), case)?,
        ];
        let cpu = self.cpu;
        let workers: Vec<_> = executors
            .into_iter()
            .map(|mut executor| {
                std::thread::spawn(move || {
                    let memory = fixture_memory(&[0x9100_0400], 0);
                    let mut state = state_at(CODE);
                    run(cpu, executor.as_mut(), &memory, &mut state, 1).map(|_| state)
                })
            })
            .collect();
        for worker in workers {
            let state = worker
                .join()
                .map_err(|_| Self::fail(case, "executor worker panicked"))?
                .map_err(|error| Self::fail(case, error))?;
            let ThreadCpuState::A64(state) = state else {
                unreachable!();
            };
            if state.read_x(A64Register::General(register(0))) != 1 {
                return Err(Self::fail(case, "concurrent executor lost private state"));
            }
        }
        domain
            .shutdown()
            .map_err(|error| Self::fail(case, error.to_string()))?;
        self.passed.push(case);
        Ok(())
    }

    fn fresh_executor(
        &mut self,
        case: ConformanceCase,
    ) -> Result<FreshExecutor, ConformanceFailure> {
        let mut domain = self.domain(case)?;
        let executor = self.executor(domain.as_mut(), case)?;
        Ok((domain, executor))
    }
}

struct FixedTimer;

impl EngineTimer for FixedTimer {
    fn snapshot(&self) -> TimerSnapshot {
        TimerSnapshot {
            counter: 123,
            frequency: 19_200_000,
        }
    }
}

fn run(
    cpu: ProcessCpuContext,
    executor: &mut dyn EngineExecutor,
    memory: &SyntheticMemory,
    state: &mut ThreadCpuState,
    budget: u64,
) -> Result<crate::ExecutionReport, Box<str>> {
    executor
        .run_slice(RunRequest {
            cpu,
            memory,
            state,
            instruction_budget: budget,
            loader_return: None,
            timer: &FixedTimer,
        })
        .map_err(|error| error.to_string().into_boxed_str())
}

fn fixture_memory(code: &[u32], data: u32) -> SyntheticMemory {
    let mut memory = SyntheticMemory::new();
    assert!(memory.add_ram_page(CODE_PAGE));
    assert!(memory.add_ram_page(DATA_PAGE));
    for (index, encoding) in code.iter().enumerate() {
        assert!(memory.initialize_ram(CODE_PAGE, index * 4, &encoding.to_le_bytes()));
    }
    assert!(memory.initialize_ram(DATA_PAGE, 0, &data.to_le_bytes()));
    assert!(memory.map_page(
        SPACE,
        CODE,
        CODE_PAGE,
        MemoryPermissions::READ_WRITE_EXECUTE
    ));
    assert!(memory.map_page(SPACE, DATA, DATA_PAGE, MemoryPermissions::READ_WRITE));
    memory
}

fn state_at(address: GuestVirtualAddress) -> ThreadCpuState {
    let mut state = A64State::default();
    state.set_pc(address.get());
    ThreadCpuState::A64(Box::new(state))
}

fn instruction_address(state: &ThreadCpuState) -> u64 {
    let ThreadCpuState::A64(state) = state else {
        unreachable!();
    };
    state.pc()
}

fn set_instruction_address(state: &mut ThreadCpuState, address: GuestVirtualAddress) {
    let ThreadCpuState::A64(state) = state else {
        unreachable!();
    };
    state.set_pc(address.get());
}

fn register(index: u8) -> A64GeneralRegister {
    A64GeneralRegister::new(index).expect("conformance registers are architectural")
}

fn location(cpu: ProcessCpuContext) -> nixe_cpu::location::LocationDescriptor {
    nixe_cpu::location::LocationDescriptor::new(CODE, ExecutionState::A64, cpu.profile().id())
}
