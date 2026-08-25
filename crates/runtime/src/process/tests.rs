use super::*;

use std::fs;

use nixe_cpu::exception::ExceptionKind;
use nixe_cpu::ir::terminator::{ControlTarget, Terminator};
use nixe_cpu::location::InstructionEncoding;
use nixe_cpu::memory::{
    CpuMemory, InstructionMemory, MemoryAccess, MemoryAccessSize, MemoryPermissions, MemoryValue,
    SYNTHETIC_PAGE_SIZE,
};

use crate::{Launcher, LauncherInput};

fn reference_process_builder() -> ProcessBuilder {
    ProcessBuilder::default()
        .with_engine_provider(Arc::new(nixe_cpu_engine_interpreter::InterpreterProvider))
}

#[derive(Default)]
struct RecordingSupervisorCallDispatcher {
    expected_encoding: Option<InstructionEncoding>,
    observed: Option<(
        crate::ExceptionDispatchRequest,
        AddressSpaceId,
        nixe_scheduler::GuestThreadId,
        nixe_scheduler::VirtualCpuId,
        u64,
        u32,
    )>,
}

impl crate::ExceptionDispatcher for RecordingSupervisorCallDispatcher {
    type Fault = &'static str;

    fn dispatch(
        &mut self,
        context: &mut crate::ExceptionDispatchContext<'_>,
        request: crate::ExceptionDispatchRequest,
    ) -> crate::ExceptionDispatchOutcome<Self::Fault> {
        let address_space = context.process().cpu().address_space_id();
        let encoding = match request.source().execution_state {
            ExecutionState::A64 | ExecutionState::A32 => context
                .process()
                .memory()
                .fetch32(address_space, request.source().pc)
                .map(|value| InstructionEncoding::from_u32(value.bits))
                .unwrap(),
            ExecutionState::T32 => context
                .process()
                .memory()
                .fetch16(address_space, request.source().pc)
                .map(|value| InstructionEncoding::from_u16(value.bits))
                .unwrap(),
        };
        assert_eq!(Some(encoding), self.expected_encoding);
        assert!(
            context
                .process()
                .handles()
                .get_as::<crate::ThreadObject>(context.thread().handle())
                .is_some()
        );

        let thread_id = context.thread().object().thread_id();
        let handle = context.thread().handle();
        assert_eq!(
            context.thread().state().execution_state(),
            request.source().execution_state
        );
        match context.thread_mut().state_mut() {
            ThreadCpuState::A64(state) => state.write_x(
                nixe_cpu::state::a64::A64Register::General(a64_register(0)),
                0xfeed_face,
            ),
            ThreadCpuState::A32(state) => state.write_r(a32_register(0), 0xfeed_face),
        }
        self.observed = Some((
            request,
            address_space,
            context.thread().id(),
            context.thread().vcpu(),
            thread_id,
            handle,
        ));
        crate::ExceptionDispatchOutcome::Suspend(crate::ExceptionResume::Retry)
    }
}

struct FixedSupervisorCallDispatcher<F> {
    outcome: Option<crate::ExceptionDispatchOutcome<F>>,
}

impl<F> crate::ExceptionDispatcher for FixedSupervisorCallDispatcher<F> {
    type Fault = F;

    fn dispatch(
        &mut self,
        _context: &mut crate::ExceptionDispatchContext<'_>,
        _request: crate::ExceptionDispatchRequest,
    ) -> crate::ExceptionDispatchOutcome<Self::Fault> {
        self.outcome.take().expect("dispatcher is called once")
    }
}

struct PcMutatingSupervisorCallDispatcher<F> {
    outcome: Option<crate::ExceptionDispatchOutcome<F>>,
}

struct RuntimeFakeProvider;
impl nixe_cpu_engine::EngineProvider for RuntimeFakeProvider {
    fn descriptor(&self) -> nixe_cpu_engine::EngineDescriptor {
        nixe_cpu_engine::EngineDescriptor {
            id: nixe_cpu_engine::EngineId::new(99),
            name: "runtime-fake".into(),
            kind: nixe_cpu_engine::EngineKind::Test,
            capabilities: nixe_cpu_engine::EngineCapabilities {
                a64: true,
                deterministic_execution: true,
                ..Default::default()
            },
        }
    }
    fn probe(
        &self,
        _profile: nixe_cpu::profile::GuestCpuProfile,
        _required: nixe_cpu_engine::EngineCapabilities,
    ) -> nixe_cpu_engine::CapabilityReport {
        nixe_cpu_engine::CapabilityReport {
            descriptor: self.descriptor(),
            available: true,
            rejections: Box::new([]),
        }
    }
    fn create_domain(
        &self,
        request: nixe_cpu_engine::DomainRequest,
    ) -> Result<Box<dyn nixe_cpu_engine::EngineDomain>, nixe_cpu_engine::EngineFault> {
        Ok(Box::new(RuntimeFakeDomain { id: request.domain }))
    }
}

struct ControlLyingProvider;

fn control_lying_descriptor() -> nixe_cpu_engine::EngineDescriptor {
    nixe_cpu_engine::EngineDescriptor {
        id: nixe_cpu_engine::EngineId::new(100),
        name: "control-lying-fake".into(),
        kind: nixe_cpu_engine::EngineKind::Test,
        capabilities: nixe_cpu_engine::EngineCapabilities {
            a64: true,
            concurrent_executors: true,
            max_safepoint_instructions: std::num::NonZeroU64::new(1),
            acknowledged_invalidation: true,
            deterministic_execution: true,
            ..Default::default()
        },
    }
}

impl nixe_cpu_engine::EngineProvider for ControlLyingProvider {
    fn descriptor(&self) -> nixe_cpu_engine::EngineDescriptor {
        control_lying_descriptor()
    }

    fn probe(
        &self,
        _profile: nixe_cpu::profile::GuestCpuProfile,
        _required: nixe_cpu_engine::EngineCapabilities,
    ) -> nixe_cpu_engine::CapabilityReport {
        nixe_cpu_engine::CapabilityReport {
            descriptor: self.descriptor(),
            available: true,
            rejections: Box::new([]),
        }
    }

    fn create_domain(
        &self,
        request: nixe_cpu_engine::DomainRequest,
    ) -> Result<Box<dyn nixe_cpu_engine::EngineDomain>, nixe_cpu_engine::EngineFault> {
        Ok(Box::new(ControlLyingDomain { id: request.domain }))
    }
}

struct ControlLyingDomain {
    id: nixe_cpu_engine::EngineDomainId,
}

impl nixe_cpu_engine::EngineDomain for ControlLyingDomain {
    fn descriptor(&self) -> nixe_cpu_engine::EngineDescriptor {
        control_lying_descriptor()
    }

    fn domain_id(&self) -> nixe_cpu_engine::EngineDomainId {
        self.id
    }

    fn create_executor(
        &mut self,
        executor: nixe_cpu_engine::EngineExecutorId,
    ) -> Result<Box<dyn nixe_cpu_engine::EngineExecutor>, nixe_cpu_engine::EngineFault> {
        Ok(Box::new(RuntimeFakeExecutor { id: executor }))
    }
}

struct RuntimeFakeDomain {
    id: nixe_cpu_engine::EngineDomainId,
}
impl nixe_cpu_engine::EngineDomain for RuntimeFakeDomain {
    fn descriptor(&self) -> nixe_cpu_engine::EngineDescriptor {
        <RuntimeFakeProvider as nixe_cpu_engine::EngineProvider>::descriptor(&RuntimeFakeProvider)
    }
    fn domain_id(&self) -> nixe_cpu_engine::EngineDomainId {
        self.id
    }

    fn create_executor(
        &mut self,
        executor: nixe_cpu_engine::EngineExecutorId,
    ) -> Result<Box<dyn nixe_cpu_engine::EngineExecutor>, nixe_cpu_engine::EngineFault> {
        Ok(Box::new(RuntimeFakeExecutor { id: executor }))
    }
}

struct RuntimeFakeExecutor {
    id: nixe_cpu_engine::EngineExecutorId,
}

impl nixe_cpu_engine::EngineExecutor for RuntimeFakeExecutor {
    fn descriptor(&self) -> nixe_cpu_engine::EngineDescriptor {
        <RuntimeFakeProvider as nixe_cpu_engine::EngineProvider>::descriptor(&RuntimeFakeProvider)
    }

    fn executor_id(&self) -> nixe_cpu_engine::EngineExecutorId {
        self.id
    }

    fn run_slice(
        &mut self,
        request: nixe_cpu_engine::RunRequest<'_>,
    ) -> Result<nixe_cpu_engine::ExecutionReport, nixe_cpu_engine::EngineFault> {
        Ok(nixe_cpu_engine::ExecutionReport {
            instructions_executed: 0,
            stop: nixe_cpu_engine::EngineExit::PendingEvent { mask: 0x80 },
            context: request.state.register_context(),
        })
    }
    fn clear_local_exclusive_reservation(&mut self) {}
}

impl<F> crate::ExceptionDispatcher for PcMutatingSupervisorCallDispatcher<F> {
    type Fault = F;

    fn dispatch(
        &mut self,
        context: &mut crate::ExceptionDispatchContext<'_>,
        _request: crate::ExceptionDispatchRequest,
    ) -> crate::ExceptionDispatchOutcome<Self::Fault> {
        match context.thread_mut().state_mut() {
            ThreadCpuState::A64(state) => state.set_pc(0x1000),
            ThreadCpuState::A32(state) => state.set_instruction_address(0x1000).unwrap(),
        }
        self.outcome.take().expect("dispatcher is called once")
    }
}

fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn replace_entry_instructions(process: &mut RunnableProcess, encodings: &[u32]) {
    let address_space = process.cpu.address_space_id();
    let entry = GuestVirtualAddress::new(process.entry_module().entry_address());
    let mapping = process.memory.mapping_info(address_space, entry).unwrap();
    let alias_page = GuestVirtualAddress::new(0x6000_0000);
    assert!(
        std::sync::Arc::get_mut(&mut process.memory)
            .unwrap()
            .map_page(
                address_space,
                alias_page,
                mapping.physical_page,
                MemoryPermissions::READ_WRITE,
            )
    );
    let page_offset = entry.get() % SYNTHETIC_PAGE_SIZE as u64;
    for (index, encoding) in encodings.iter().copied().enumerate() {
        let instruction_offset = u64::try_from(index).unwrap() * 4;
        process
            .memory
            .write(
                address_space,
                GuestVirtualAddress::new(alias_page.get() + page_offset + instruction_offset),
                MemoryAccess::normal(MemoryAccessSize::Word),
                MemoryValue::U32(encoding),
            )
            .unwrap();
    }
}

fn replace_entry_instruction(process: &mut RunnableProcess, encoding: u32) {
    replace_entry_instructions(process, &[encoding]);
}

#[test]
fn runtime_orchestration_accepts_an_engine_neutral_fake_domain() {
    let (_directory, plan) = plan();
    let mut process = reference_process_builder()
        .with_engine_provider(Arc::new(RuntimeFakeProvider))
        .build(&plan)
        .unwrap();
    let before = process.main_thread().state().register_context();
    let report = process.run(50).unwrap();
    assert_eq!(report.instructions_executed, 0);
    assert_eq!(
        report.stop,
        crate::ExecutionStop::PendingEvent { mask: 0x80 }
    );
    assert_eq!(report.context, before);
}

#[test]
fn worker_slice_moves_thread_state_out_of_the_process_until_reconciliation() {
    let mut process = synthetic_process_for_coordinator(1);
    let thread = process.main_thread_id();
    let vcpu = nixe_scheduler::VirtualCpuId::new(3);

    let execution = process
        .begin_thread_execution(thread, vcpu, 1, nixe_cpu_engine::VcpuEventState::default())
        .unwrap();
    assert!(process.main_thread().state.is_none());

    process.abort_thread_execution(thread, vcpu, execution);
    assert!(process.main_thread().state.is_some());
    assert_eq!(
        process.lifecycle(),
        nixe_scheduler::ProcessLifecycle::Faulted
    );
}

#[test]
fn application_must_inject_a_cpu_engine_provider() {
    let (_directory, plan) = plan();
    let Err(error) = ProcessBuilder::default().build(&plan) else {
        panic!("process construction must reject a missing CPU engine provider");
    };
    assert_eq!(error.stage(), ProcessBuildStage::EngineInitialization);
}

#[test]
fn runtime_mapping_mutation_requires_engine_acknowledgement_before_reentry() {
    let (_directory, plan) = plan();
    let mut process = reference_process_builder().build(&plan).unwrap();
    let heap = process.memory_layout().heap();
    process
        .resize_memory_mapping(
            heap.base(),
            0,
            SYNTHETIC_PAGE_SIZE as u64,
            MemoryPermissions::READ_WRITE,
            MemoryMappingPurpose::Heap,
        )
        .unwrap();
    let cursor = process.memory.invalidation_cursor();
    assert!(!process.memory_invalidation_acknowledged(cursor));
    assert_eq!(
        process.run(0).unwrap().stop,
        crate::ExecutionStop::BudgetExhausted
    );
    assert!(process.memory_invalidation_acknowledged(cursor));
}

#[test]
fn provider_cannot_advertise_parallel_controls_without_returning_one() {
    let (_directory, plan) = plan();
    let Err(error) = ProcessBuilder::default()
        .with_engine_provider(Arc::new(ControlLyingProvider))
        .build(&plan)
    else {
        panic!("a lying control capability must be rejected");
    };
    assert_eq!(error.stage(), ProcessBuildStage::EngineInitialization);
    assert!(error.to_string().contains("out-of-band control path"));
}

fn process_stopped_at_svc(
    execution_state: ExecutionState,
) -> (RunnableProcess, crate::ExecutionReport, u64) {
    let encoding = match execution_state {
        ExecutionState::A64 => 0xd400_4681,
        ExecutionState::A32 => 0xef12_3456,
        ExecutionState::T32 => 0xbf00_df7b,
    };
    let (_directory, plan) = plan();
    let mut process = reference_process_builder().build(&plan).unwrap();
    replace_entry_instruction(&mut process, encoding);
    let entry = process.entry_module().entry_address();
    if execution_state != ExecutionState::A64 {
        let mut state = match execution_state {
            ExecutionState::A32 => nixe_cpu::state::A32State::a32(),
            ExecutionState::T32 => nixe_cpu::state::A32State::t32(),
            ExecutionState::A64 => unreachable!(),
        };
        state
            .set_instruction_address(u32::try_from(entry).unwrap())
            .unwrap();
        *process.main_thread_mut().state_mut() = ThreadCpuState::A32(Box::new(state));
    }
    let report = process.run(1).unwrap();
    assert!(matches!(
        report.stop,
        crate::ExecutionStop::SupervisorCall { .. }
    ));
    (process, report, entry)
}

fn instruction_address(state: &ThreadCpuState) -> u64 {
    match state {
        ThreadCpuState::A64(state) => state.pc(),
        ThreadCpuState::A32(state) => u64::from(state.instruction_address()),
    }
}

fn synthetic_nro() -> Vec<u8> {
    let mut bytes = vec![0; 0x2800];
    bytes[..4].copy_from_slice(&0x1400_0020_u32.to_le_bytes()); // B entry + 0x80
    bytes[0x10..0x14].copy_from_slice(b"NRO0");
    put_u32(&mut bytes, 0x18, 0x2800);
    put_u32(&mut bytes, 0x20, 0);
    put_u32(&mut bytes, 0x24, 0x1000);
    put_u32(&mut bytes, 0x28, 0x1000);
    put_u32(&mut bytes, 0x2c, 0x1000);
    put_u32(&mut bytes, 0x30, 0x2000);
    put_u32(&mut bytes, 0x34, 0x800);
    put_u32(&mut bytes, 0x38, 0x800);
    bytes[0x40..0x60].fill(0x5a);
    bytes
}

fn plan() -> (tempfile::TempDir, LaunchPlan) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("synthetic.nro");
    fs::write(&path, synthetic_nro()).unwrap();
    let plan = Launcher::build(LauncherInput::new(&path)).unwrap();
    (directory, plan)
}

pub(crate) fn synthetic_process_for_coordinator(process_id: u64) -> RunnableProcess {
    let (_directory, plan) = plan();
    reference_process_builder()
        .with_config(ProcessBuildConfig {
            process_id,
            address_space_id: AddressSpaceId::new(process_id),
            ..ProcessBuildConfig::default()
        })
        .build(&plan)
        .unwrap()
}

pub(crate) fn synthetic_instruction_process_for_coordinator(
    process_id: u64,
    encodings: &[u32],
) -> RunnableProcess {
    let mut process = synthetic_process_for_coordinator(process_id);
    replace_entry_instructions(&mut process, encodings);
    process
}

pub(crate) fn synthetic_svc_process_for_coordinator(process_id: u64) -> RunnableProcess {
    let mut process = synthetic_process_for_coordinator(process_id);
    replace_entry_instruction(&mut process, 0xd400_4681);
    process
}

mod construction;
mod exception;
mod run;
