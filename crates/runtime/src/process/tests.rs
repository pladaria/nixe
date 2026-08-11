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
                precise_instruction_budget: true,
                ..Default::default()
            },
        }
    }
    fn probe(
        &self,
        _profile: nixe_cpu::profile::CpuProfileId,
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
        request: nixe_cpu_engine::ExecutorRequest,
    ) -> Result<Box<dyn nixe_cpu_engine::EngineExecutor>, nixe_cpu_engine::EngineFault> {
        Ok(Box::new(RuntimeFakeExecutor {
            id: request.executor,
        }))
    }

    fn quiesce(
        &mut self,
    ) -> Result<nixe_cpu_engine::DomainQuiescenceToken, nixe_cpu_engine::EngineFault> {
        Ok(nixe_cpu_engine::DomainQuiescenceToken {
            domain: self.id,
            generation: nixe_cpu_engine::EngineGeneration::new(0),
        })
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
            trace: nixe_cpu_engine::InstructionTrace {
                enabled: false,
                entries: Box::new([]),
                discarded: 0,
            },
            state_commit: nixe_cpu_engine::StateCommitStatus::Canonical,
        })
    }
    fn request_safepoint(&mut self, _reason: nixe_cpu_engine::SafepointReason) {}
    fn post_event(&self, _mask: u32) {}
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
    assert!(process.memory.map_page(
        address_space,
        alias_page,
        mapping.physical_page,
        MemoryPermissions::READ_WRITE,
    ));
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
    let before = process.main_thread().state.register_context();
    let report = process.run_reference(50).unwrap();
    assert_eq!(report.instructions_executed, 0);
    assert_eq!(
        report.stop,
        crate::ExecutionStop::PendingEvent { mask: 0x80 }
    );
    assert_eq!(report.context, before);
    assert_eq!(
        process.execution_status(),
        crate::ProcessExecutionStatus::Ready
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
        process.main_thread_mut().state = ThreadCpuState::A32(Box::new(state));
    }
    let report = process.run_reference(1).unwrap();
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

pub(crate) fn synthetic_svc_process_for_coordinator(process_id: u64) -> RunnableProcess {
    let mut process = synthetic_process_for_coordinator(process_id);
    replace_entry_instruction(&mut process, 0xd400_4681);
    process
}

#[test]
fn builder_propagates_runtime_diagnostics_to_cpu_resources() {
    let builder = reference_process_builder();
    assert_eq!(
        builder.cpu_diagnostics().report_detail,
        nixe_cpu::coverage::MissingInstructionReportDetail::Detailed
    );
}

#[test]
fn npdm_address_space_values_keep_distinct_runtime_meanings() {
    assert_eq!(
        ProcessAddressSpace::from_npdm(AddressSpaceType::AddressSpace32Bit),
        ProcessAddressSpace::Bit32
    );
    assert_eq!(
        ProcessAddressSpace::from_npdm(AddressSpaceType::AddressSpace32BitNoReserved),
        ProcessAddressSpace::Bit32NoReserved
    );
    assert_eq!(
        ProcessAddressSpace::from_npdm(AddressSpaceType::AddressSpace64BitOld),
        ProcessAddressSpace::Bit64Old
    );
    assert_eq!(
        ProcessAddressSpace::from_npdm(AddressSpaceType::AddressSpace64Bit),
        ProcessAddressSpace::Bit64
    );
    assert!(validate_range(ProcessAddressSpace::Bit32, u64::from(u32::MAX), 2).is_err());
}

#[test]
fn horizon_layout_profiles_keep_allocation_windows_and_resource_limits_distinct() {
    let code_start = 0x7100_0000;
    let code_end = 0x7100_4000;
    let limit = 0x1234_0000;
    let layout = ProcessMemoryLayout::for_address_space(
        ProcessMemoryLayoutProfile::Horizon2Plus,
        ProcessAddressSpace::Bit64,
        code_start,
        code_end,
        limit,
    )
    .unwrap();
    assert_eq!(layout.aslr().base().get(), 0x0800_0000);
    assert_eq!(layout.aslr().end(), 1_u64 << 39);
    assert!(layout.aslr().base().get() <= layout.stack().base().get());
    assert!(layout.stack().end() <= layout.aslr().end());
    assert!(layout.alias().end() <= layout.aslr().end());
    assert!(layout.heap().end() <= layout.aslr().end());
    assert!(layout.stack().base().get() >= 0x7120_0000);
    assert_eq!(layout.alias().size(), 0x10_0000_0000);
    assert_eq!(layout.heap().size(), 0x2_0000_0000);
    assert_eq!(layout.stack().size(), 0x8000_0000);
    assert_eq!(layout.memory_capacity(), limit);

    let high_code_start = 0x64_0000_0000;
    let high_layout = ProcessMemoryLayout::for_address_space(
        ProcessMemoryLayoutProfile::Horizon2Plus,
        ProcessAddressSpace::Bit64,
        high_code_start,
        high_code_start + HORIZON_REGION_ALIGNMENT,
        limit,
    )
    .unwrap();
    assert!(high_layout.heap().end() <= high_code_start);

    let without_alias = ProcessMemoryLayout::for_address_space(
        ProcessMemoryLayoutProfile::Horizon2Plus,
        ProcessAddressSpace::Bit32NoReserved,
        0x0020_0000,
        0x0040_0000,
        limit,
    )
    .unwrap();
    assert_eq!(without_alias.alias().size(), 0);
    assert_eq!(without_alias.heap().base().get(), 0x4000_0000);
    assert_eq!(without_alias.heap().size(), 0x8000_0000);

    let deprecated = ProcessMemoryLayout::for_address_space(
        ProcessMemoryLayoutProfile::Horizon1,
        ProcessAddressSpace::Bit64Old,
        0x0800_0000,
        0x0820_0000,
        limit,
    )
    .unwrap();
    assert_eq!(deprecated.aslr().end(), 1_u64 << 36);
    assert_eq!(deprecated.alias().size(), 0x1_8000_0000);
    assert_eq!(deprecated.heap().size(), 0x2_0000_0000);
    assert!(
        ProcessMemoryLayout::for_address_space(
            ProcessMemoryLayoutProfile::Horizon1,
            ProcessAddressSpace::Bit64,
            code_start,
            code_end,
            limit,
        )
        .is_err()
    );
}

#[test]
fn a32_thread_initialization_uses_32_bit_pc_stack_and_tls() {
    let cpu = ProcessCpuContext::new(GuestCpuProfile::switch_1(), AddressSpaceId::new(7));
    let configuration = cpu.thread_configuration(ExecutionState::A32).unwrap();
    let mut state = ThreadCpuState::new(configuration);
    initialize_thread(
        &mut state,
        GuestVirtualAddress::new(0x0020_0000),
        GuestVirtualAddress::new(0x0080_0000),
        GuestVirtualAddress::new(0x0090_0000),
        1,
        None,
        None,
    )
    .unwrap();
    let ThreadCpuState::A32(state) = state else {
        panic!("A32 metadata must create AArch32 state");
    };
    assert_eq!(state.instruction_address(), 0x0020_0000);
    assert_eq!(state.read_r(a32_register(13)), 0x0080_0000);
    assert_eq!(state.tpidrurw(), 0x0090_0000);
    assert_eq!(state.tpidruro(), 0x0090_0000);
    assert_eq!(state.read_r(a32_register(1)), 1);
}

#[test]
fn a32_created_thread_initialization_uses_create_thread_abi() {
    let cpu = ProcessCpuContext::new(GuestCpuProfile::switch_1(), AddressSpaceId::new(7));
    let configuration = cpu.thread_configuration(ExecutionState::A32).unwrap();
    let mut state = ThreadCpuState::new(configuration);
    initialize_created_thread(
        &mut state,
        &ThreadCreateRequest {
            entry: GuestVirtualAddress::new(0x0020_0100),
            argument: 0x1234_5678,
            stack_top: GuestVirtualAddress::new(0x0080_0000),
            priority: 20,
            ideal_vcpu: Some(nixe_scheduler::VirtualCpuId::new(0)),
            affinity: nixe_scheduler::MachineSchedulerProfile::new(
                vec![nixe_scheduler::VirtualCpuDescriptor::new(
                    nixe_scheduler::VirtualCpuId::new(0),
                    0,
                )],
                nixe_scheduler::PriorityRange::new(0, 63).unwrap(),
                1,
            )
            .unwrap()
            .all_cores(),
        },
        GuestVirtualAddress::new(0x0090_0000),
    )
    .unwrap();
    let ThreadCpuState::A32(state) = state else {
        panic!("A32 configuration must create AArch32 state");
    };
    assert_eq!(state.instruction_address(), 0x0020_0100);
    assert_eq!(state.read_r(a32_register(0)), 0x1234_5678);
    assert_eq!(state.read_r(a32_register(13)), 0x0080_0000);
    assert_eq!(state.tpidrurw(), 0x0090_0000);
    assert_eq!(state.tpidruro(), 0x0090_0000);
}

#[test]
fn synthetic_launch_translates_entry_only_through_process_memory() {
    let (_directory, plan) = plan();
    let process = reference_process_builder().build(&plan).unwrap();
    let entry = GuestVirtualAddress::new(process.entry_module().entry_address());
    assert_eq!(
        process
            .memory()
            .fetch32(process.cpu_context().address_space_id(), entry)
            .unwrap()
            .bits,
        0x1400_0020
    );
    let dump = process.print_entry_ir().unwrap();
    assert!(dump.contains(" A64 "));
    assert!(dump.contains("raw=0x14000020"));
    assert!(dump.contains("guest=\"b imm=#128\""));
    let report = process.print_entry_report();
    assert!(report.starts_with("nixe-frontend-block-report-v1\n"));
    assert!(report.contains("outcome=translated end=direct-branch"));
    assert!(report.contains("ir-dump stage=pre-optimization"));
    assert!(report.contains("dependency page="));
    assert_eq!(
        process.main_thread().state.execution_state(),
        ExecutionState::A64
    );
    let ThreadCpuState::A64(state) = &process.main_thread().state else {
        panic!("homebrew fixture must initialize A64");
    };
    assert_eq!(
        process
            .handles()
            .get_as::<crate::ThreadObject>(process.main_thread().handle)
            .map(crate::ThreadObject::thread_id),
        Some(1)
    );
    assert!(process.mounts().primary().is_none());
    assert!(process.mounts().add_ons().is_empty());
    assert_eq!(state.pc(), entry.get());
    assert_eq!(
        state.read_x(A64Register::StackPointer),
        process.main_thread().stack_top.get()
    );
    assert_eq!(state.tpidr_el0(), process.main_thread().tls_base.get());
    let context = process.main_thread().abi_context.unwrap();
    assert_eq!(
        state.read_x(A64Register::General(a64_register(0))),
        context.get()
    );
    assert_eq!(
        state.read_x(A64Register::General(a64_register(1))),
        u64::MAX
    );
    let loader_return = process.main_thread().loader_return.unwrap();
    assert_eq!(
        state.read_x(A64Register::General(a64_register(30))),
        loader_return.get()
    );
    assert_eq!(
        process
            .memory()
            .mapping_info(process.cpu_context().address_space_id(), loader_return)
            .unwrap()
            .permissions,
        MemoryPermissions::READ_EXECUTE
    );
    assert_eq!(
        process
            .memory()
            .fetch32(process.cpu_context().address_space_id(), loader_return)
            .unwrap()
            .bits,
        HOME_BREW_EXIT_PROCESS_INSTRUCTION
    );
    assert_eq!(
        process
            .memory()
            .read(
                process.cpu_context().address_space_id(),
                context,
                MemoryAccess::normal(MemoryAccessSize::Word),
            )
            .unwrap()
            .value,
        MemoryValue::U32(HOME_BREW_MAIN_THREAD_HANDLE_KEY)
    );
    assert_eq!(
        process
            .memory()
            .read(
                process.cpu_context().address_space_id(),
                context.checked_add(8).unwrap(),
                MemoryAccess::normal(MemoryAccessSize::Doubleword),
            )
            .unwrap()
            .value,
        MemoryValue::U64(u64::from(process.main_thread().handle))
    );
    assert_eq!(
        process
            .memory()
            .read(
                process.cpu_context().address_space_id(),
                context
                    .checked_add(HOME_BREW_CONFIG_ENTRY_SIZE as u64)
                    .unwrap(),
                MemoryAccess::normal(MemoryAccessSize::Word),
            )
            .unwrap()
            .value,
        MemoryValue::U32(0)
    );
}

#[test]
fn nro_loader_return_preserves_x0_and_exits_without_executing_the_gateway() {
    let (_directory, plan) = plan();
    let mut process = reference_process_builder().build(&plan).unwrap();
    replace_entry_instruction(&mut process, 0xd65f_03c0); // RET X30
    let loader_return = process.main_thread().loader_return.unwrap();
    let ThreadCpuState::A64(state) = &mut process.main_thread_mut().state else {
        panic!("homebrew fixture must initialize A64");
    };
    state.write_x(A64Register::General(a64_register(0)), 0x1234_5678);

    let report = process.run_reference(1).unwrap();

    assert_eq!(report.instructions_executed, 1);
    assert_eq!(
        report.stop,
        crate::ExecutionStop::LoaderReturn {
            source: LocationDescriptor::new(
                loader_return,
                ExecutionState::A64,
                process.cpu_context().profile().id(),
            ),
            result_code: 0x1234_5678,
        }
    );
    assert_eq!(process.execution_status(), ProcessExecutionStatus::Exited);
    assert_eq!(
        process.exit(),
        Some(ProcessExit {
            cause: ProcessExitCause::LoaderReturned,
            exit_code: 0x1234_5678,
            source: Some(LocationDescriptor::new(
                loader_return,
                ExecutionState::A64,
                process.cpu_context().profile().id(),
            )),
            thread_id: 1,
        })
    );
    assert_eq!(
        process.main_thread().exit(),
        Some(ThreadExit {
            requested_scope: ExceptionTerminationScope::Process,
            exit_code: 0x1234_5678,
            source: Some(LocationDescriptor::new(
                loader_return,
                ExecutionState::A64,
                process.cpu_context().profile().id(),
            )),
        })
    );
    assert!(matches!(
        process.run_reference(1),
        Err(ProcessExecutionError::NotRunnable {
            status: ProcessExecutionStatus::Exited,
            ..
        })
    ));
    let teardown = process.teardown();
    assert_eq!(teardown.exit.unwrap().exit_code, 0x1234_5678);
}

#[test]
fn image_base_is_relocatable_without_changing_pc_relative_translation() {
    let (_directory, plan) = plan();
    let first = reference_process_builder()
        .with_config(ProcessBuildConfig {
            image_base: GuestVirtualAddress::new(0x7100_0000),
            ..ProcessBuildConfig::default()
        })
        .build(&plan)
        .unwrap();
    let second = reference_process_builder()
        .with_config(ProcessBuildConfig {
            image_base: GuestVirtualAddress::new(0x7200_0000),
            ..ProcessBuildConfig::default()
        })
        .build(&plan)
        .unwrap();
    assert_eq!(
        second.entry_module().entry_address() - first.entry_module().entry_address(),
        0x0100_0000
    );
    let first_block = first.translate_entry().unwrap();
    let second_block = second.translate_entry().unwrap();
    let direct_target = |block: &IrBlock| match block.terminator {
        Terminator::Direct {
            target: ControlTarget::Direct { pc, .. },
        } => pc.get(),
        ref terminator => panic!("unexpected terminator {terminator:?}"),
    };
    assert_eq!(
        direct_target(&second_block) - direct_target(&first_block),
        0x0100_0000
    );
    assert_eq!(
        second.modules()[0].mappings()[0].guest_address()
            - first.modules()[0].mappings()[0].guest_address(),
        0x0100_0000
    );
}

#[test]
fn writable_code_alias_updates_the_fetched_generation() {
    let (_directory, plan) = plan();
    let mut process = reference_process_builder().build(&plan).unwrap();
    let space = process.cpu.address_space_id();
    let entry = GuestVirtualAddress::new(process.entry_module().entry_address());
    let before = process.memory.fetch32(space, entry).unwrap().dependencies;
    let mapping = process.memory.mapping_info(space, entry).unwrap();
    let alias = GuestVirtualAddress::new(0x7000_0000);
    assert!(process.memory.map_page(
        space,
        alias,
        mapping.physical_page,
        MemoryPermissions::READ_WRITE
    ));
    process
        .memory
        .write(
            space,
            alias,
            MemoryAccess::normal(MemoryAccessSize::Word),
            MemoryValue::U32(0xd503_201f),
        )
        .unwrap();
    let after = process.memory.fetch32(space, entry).unwrap().dependencies;
    assert_ne!(before, after);
}

#[test]
fn reference_execution_honors_budget_and_preserves_dispatch_pc() {
    let (_directory, plan) = plan();
    let mut process = reference_process_builder().build(&plan).unwrap();
    let entry = process.entry_module().entry_address();

    let report = process.run_reference(1).unwrap();
    assert_eq!(report.instructions_executed, 1);
    assert_eq!(report.stop, crate::ExecutionStop::BudgetExhausted);
    assert!(report.stop.exception_dispatch_request().is_none());
    assert!(!report.trace.enabled());
    assert!(report.trace.entries().is_empty());
    assert_eq!(
        process.execution_status(),
        crate::ProcessExecutionStatus::Ready
    );
    let nixe_cpu::state::RegisterContext::A64(context) = &report.context else {
        panic!("homebrew fixture must report A64 context");
    };
    assert_eq!(context.pc.get(), entry + 0x80);
    assert!(report.to_string().contains("flags=N0Z0C0V0"));
    let ThreadCpuState::A64(state) = &process.main_thread().state else {
        panic!("homebrew fixture must initialize A64");
    };
    assert_eq!(state.pc(), entry + 0x80);
}

#[test]
fn reference_slices_preserve_instruction_and_supervisor_call_boundaries() {
    let (_directory, plan) = plan();
    let mut process = reference_process_builder()
        .with_diagnostics(crate::DiagnosticsPolicy {
            instruction_trace: true,
            ..crate::DiagnosticsPolicy::default()
        })
        .build(&plan)
        .unwrap();
    replace_entry_instructions(
        &mut process,
        &[
            0x9100_0400, // ADD X0,X0,#1
            0x9100_0800, // ADD X0,X0,#2
            0xd400_0841, // SVC #0x42
            0x9100_1000, // ADD X0,X0,#4
        ],
    );
    let entry = process.entry_module().entry_address();
    let ThreadCpuState::A64(state) = &mut process.main_thread_mut().state else {
        panic!("homebrew fixture must initialize A64");
    };
    state.write_x(A64Register::General(a64_register(0)), 0);

    let first = process.run_reference(1).unwrap();
    assert_eq!(first.instructions_executed, 1);
    assert_eq!(first.stop, crate::ExecutionStop::BudgetExhausted);
    let ThreadCpuState::A64(state) = &process.main_thread().state else {
        unreachable!();
    };
    assert_eq!(state.read_x(A64Register::General(a64_register(0))), 1);
    assert_eq!(state.pc(), entry + 4);

    let second = process.run_reference(1).unwrap();
    assert_eq!(second.instructions_executed, 1);
    assert_eq!(second.stop, crate::ExecutionStop::BudgetExhausted);
    let ThreadCpuState::A64(state) = &process.main_thread().state else {
        unreachable!();
    };
    assert_eq!(state.read_x(A64Register::General(a64_register(0))), 3);
    assert_eq!(state.pc(), entry + 8);

    let svc = process.run_reference(1).unwrap();
    assert_eq!(svc.instructions_executed, 1);
    assert!(matches!(
        svc.stop,
        crate::ExecutionStop::SupervisorCall {
            source,
            immediate: 0x42,
        } if source.pc.get() == entry + 8
    ));
    let sources = svc
        .trace
        .entries()
        .iter()
        .map(|entry| entry.source.pc.get())
        .collect::<Vec<_>>();
    assert_eq!(sources, [entry, entry + 4, entry + 8]);

    let mut dispatcher = FixedSupervisorCallDispatcher {
        outcome: Some(crate::ExceptionDispatchOutcome::<&'static str>::Resume(
            crate::ExceptionResume::Next,
        )),
    };
    assert_eq!(
        process
            .route_supervisor_call(&svc.stop, &mut dispatcher)
            .unwrap(),
        crate::ExceptionHandlingResult::Resumed
    );
    let resumed = process.run_reference(1).unwrap();
    assert_eq!(resumed.instructions_executed, 1);
    assert_eq!(resumed.stop, crate::ExecutionStop::BudgetExhausted);
    let ThreadCpuState::A64(state) = &process.main_thread().state else {
        unreachable!();
    };
    assert_eq!(state.read_x(A64Register::General(a64_register(0))), 7);
    assert_eq!(state.pc(), entry + 16);
}

#[test]
fn fixed_virtual_timer_is_stable_across_reference_slices() {
    let (_directory, plan) = plan();
    let frequency = 24_000_000;
    let mut process = reference_process_builder()
        .with_config(ProcessBuildConfig {
            architectural_timer_frequency: frequency,
            ..ProcessBuildConfig::default()
        })
        .with_virtual_clock(crate::VirtualClock::new(crate::VirtualClockMode::Fixed {
            unix_seconds: 1_700_000_000,
        }))
        .build(&plan)
        .unwrap();
    replace_entry_instructions(
        &mut process,
        &[
            0xd53b_e001, // MRS X1,CNTFRQ_EL0
            0xd53b_e022, // MRS X2,CNTVCT_EL0
            0xd53b_e023, // MRS X3,CNTVCT_EL0
        ],
    );

    let first = process.run_reference(2).unwrap();
    assert_eq!(first.instructions_executed, 2);
    assert_eq!(first.stop, crate::ExecutionStop::BudgetExhausted);
    let second = process.run_reference(1).unwrap();
    assert_eq!(second.instructions_executed, 1);
    assert_eq!(second.stop, crate::ExecutionStop::BudgetExhausted);

    let ThreadCpuState::A64(state) = &process.main_thread().state else {
        unreachable!();
    };
    assert_eq!(
        state.read_x(A64Register::General(a64_register(1))),
        frequency
    );
    assert_eq!(state.read_x(A64Register::General(a64_register(2))), 0);
    assert_eq!(state.read_x(A64Register::General(a64_register(3))), 0);
}

#[test]
fn exclusive_monitor_persists_and_observes_generation_changes_across_slices() {
    let (_directory, plan) = plan();
    let mut process = reference_process_builder().build(&plan).unwrap();
    replace_entry_instructions(
        &mut process,
        &[
            0x885f_fc60, // LDAXR W0,[X3]
            0x8801_fc60, // STLXR W1,W0,[X3]
        ],
    );
    let entry = process.entry_module().entry_address();
    let data = {
        let ThreadCpuState::A64(state) = &mut process.main_thread_mut().state else {
            panic!("homebrew fixture must initialize A64");
        };
        let address = GuestVirtualAddress::new(
            state
                .read_x(A64Register::StackPointer)
                .checked_sub(8)
                .unwrap(),
        );
        state.write_x(A64Register::General(a64_register(3)), address.get());
        address
    };
    let address_space = process.cpu_context().address_space_id();
    process
        .memory()
        .write(
            address_space,
            data,
            MemoryAccess::normal(MemoryAccessSize::Word),
            MemoryValue::U32(7),
        )
        .unwrap();

    assert_eq!(
        process.run_reference(1).unwrap().stop,
        crate::ExecutionStop::BudgetExhausted
    );
    let ThreadCpuState::A64(state) = &mut process.main_thread_mut().state else {
        unreachable!();
    };
    assert_eq!(state.read_w(A64Register::General(a64_register(0))), 7);
    state.write_x(A64Register::General(a64_register(0)), 9);
    assert_eq!(
        process.run_reference(1).unwrap().stop,
        crate::ExecutionStop::BudgetExhausted
    );
    let ThreadCpuState::A64(state) = &process.main_thread().state else {
        unreachable!();
    };
    assert_eq!(state.read_w(A64Register::General(a64_register(1))), 0);
    assert_eq!(
        process
            .memory()
            .read(
                address_space,
                data,
                MemoryAccess::normal(MemoryAccessSize::Word),
            )
            .unwrap()
            .value,
        MemoryValue::U32(9)
    );

    let ThreadCpuState::A64(state) = &mut process.main_thread_mut().state else {
        unreachable!();
    };
    state.set_pc(entry);
    assert_eq!(
        process.run_reference(1).unwrap().stop,
        crate::ExecutionStop::BudgetExhausted
    );
    process
        .memory()
        .write(
            address_space,
            data,
            MemoryAccess::normal(MemoryAccessSize::Word),
            MemoryValue::U32(11),
        )
        .unwrap();
    let ThreadCpuState::A64(state) = &mut process.main_thread_mut().state else {
        unreachable!();
    };
    state.write_x(A64Register::General(a64_register(0)), 13);
    assert_eq!(
        process.run_reference(1).unwrap().stop,
        crate::ExecutionStop::BudgetExhausted
    );
    let ThreadCpuState::A64(state) = &process.main_thread().state else {
        unreachable!();
    };
    assert_eq!(state.read_w(A64Register::General(a64_register(1))), 1);
    assert_eq!(
        process
            .memory()
            .read(
                address_space,
                data,
                MemoryAccess::normal(MemoryAccessSize::Word),
            )
            .unwrap()
            .value,
        MemoryValue::U32(11)
    );
}

#[test]
fn reference_execution_observes_safepoints_before_fetch() {
    let (_directory, plan) = plan();
    let mut process = reference_process_builder().build(&plan).unwrap();
    let entry = process.entry_module().entry_address();
    process.request_safepoint();

    let report = process.run_reference(10).unwrap();
    assert_eq!(report.instructions_executed, 0);
    assert_eq!(report.stop, crate::ExecutionStop::Safepoint);
    let ThreadCpuState::A64(state) = &process.main_thread().state else {
        panic!("homebrew fixture must initialize A64");
    };
    assert_eq!(state.pc(), entry);
}

#[test]
fn reference_execution_observes_pending_events_before_fetch() {
    let (_directory, plan) = plan();
    let mut process = reference_process_builder().build(&plan).unwrap();
    let entry = process.entry_module().entry_address();
    process.post_event(0b0001);
    process.post_event(0b0100);

    let report = process.run_reference(10).unwrap();
    assert_eq!(report.instructions_executed, 0);
    assert_eq!(
        report.stop,
        crate::ExecutionStop::PendingEvent { mask: 0b0101 }
    );
    let next = process.run_reference(1).unwrap();
    assert_eq!(next.instructions_executed, 1);
    assert_eq!(next.stop, crate::ExecutionStop::BudgetExhausted);
    let ThreadCpuState::A64(state) = &process.main_thread().state else {
        unreachable!();
    };
    assert_eq!(state.pc(), entry + 0x80);
}

#[test]
fn reference_execution_reports_instruction_fetch_faults_as_a_distinct_stop() {
    let (_directory, plan) = plan();
    let mut process = reference_process_builder().build(&plan).unwrap();
    let ThreadCpuState::A64(state) = &mut process.main_thread_mut().state else {
        panic!("homebrew fixture must initialize A64");
    };
    state.set_pc(0x1000);

    let report = process.run_reference(1).unwrap();
    assert_eq!(report.instructions_executed, 0);
    assert!(matches!(
        report.stop,
        crate::ExecutionStop::FetchFault { .. }
    ));
    let nixe_cpu::state::RegisterContext::A64(context) = &report.context else {
        panic!("homebrew fixture must report A64 context");
    };
    assert_eq!(context.pc.get(), 0x1000);
    assert!(report.to_string().contains("fetch-fault"));
    assert_eq!(
        process.execution_status(),
        crate::ProcessExecutionStatus::Faulted
    );
}

#[test]
fn unallocated_encoding_suspends_until_runtime_resumes_thread() {
    let (_directory, plan) = plan();
    let mut process = reference_process_builder().build(&plan).unwrap();

    let report = process.run_reference(2).unwrap();
    assert!(matches!(
        report.stop,
        crate::ExecutionStop::UnallocatedEncoding { .. }
    ));
    assert_eq!(
        process.execution_status(),
        crate::ProcessExecutionStatus::Suspended
    );
    assert!(matches!(
        process.run_reference(1),
        Err(crate::ProcessExecutionError::NotRunnable {
            status: crate::ProcessExecutionStatus::Suspended,
            ..
        })
    ));
    assert!(process.resume());
    assert_eq!(
        process.execution_status(),
        crate::ProcessExecutionStatus::Ready
    );
}

#[test]
fn reference_execution_distinguishes_unsupported_profile_and_unallocated_code() {
    let (_directory, plan) = plan();

    let mut unsupported = reference_process_builder().build(&plan).unwrap();
    replace_entry_instruction(&mut unsupported, 0xd503_205f); // WFE
    let report = unsupported.run_reference(1).unwrap();
    assert!(matches!(
        report.stop,
        crate::ExecutionStop::UnsupportedSemantics { .. }
    ));
    assert_eq!(
        unsupported.execution_status(),
        ProcessExecutionStatus::Faulted
    );
    assert!(report.to_string().contains("unsupported-semantics"));

    let mut profile_disabled = reference_process_builder()
        .with_config(ProcessBuildConfig {
            cpu_profile: GuestCpuProfile::switch_2_native(),
            ..ProcessBuildConfig::default()
        })
        .build(&plan)
        .unwrap();
    replace_entry_instruction(&mut profile_disabled, 0x4e22_1c20);
    let report = profile_disabled.run_reference(1).unwrap();
    assert_eq!(
        report.stop.exception_dispatch_request().unwrap().kind(),
        nixe_cpu::exception::ExceptionKind::UndefinedInstruction
    );
    assert!(matches!(
        report.stop,
        crate::ExecutionStop::ProfileDisabled { .. }
    ));
    assert!(report.to_string().contains("profile-disabled"));

    let mut unallocated = reference_process_builder().build(&plan).unwrap();
    replace_entry_instruction(&mut unallocated, 0);
    let report = unallocated.run_reference(1).unwrap();
    assert_eq!(
        report.stop.exception_dispatch_request().unwrap().kind(),
        nixe_cpu::exception::ExceptionKind::UndefinedInstruction
    );
    assert!(matches!(
        report.stop,
        crate::ExecutionStop::UnallocatedEncoding { .. }
    ));
    assert!(report.to_string().contains("unallocated-encoding"));
}

#[test]
fn reference_execution_distinguishes_svc_architectural_and_data_fault_stops() {
    let (_directory, plan) = plan();

    let mut svc = reference_process_builder().build(&plan).unwrap();
    replace_entry_instruction(&mut svc, 0xd400_0841); // SVC #0x42
    let report = svc.run_reference(1).unwrap();
    let dispatch = report.stop.exception_dispatch_request().unwrap();
    assert_eq!(
        dispatch.kind(),
        nixe_cpu::exception::ExceptionKind::SupervisorCall
    );
    assert_eq!(dispatch.syndrome(), Some(0x42));
    assert!(matches!(
        report.stop,
        crate::ExecutionStop::SupervisorCall {
            immediate: 0x42,
            ..
        }
    ));
    assert!(report.to_string().contains("supervisor-call"));

    let mut breakpoint = reference_process_builder().build(&plan).unwrap();
    replace_entry_instruction(&mut breakpoint, 0xd420_2460); // BRK #0x123
    let report = breakpoint.run_reference(1).unwrap();
    let dispatch = report.stop.exception_dispatch_request().unwrap();
    assert_eq!(
        dispatch.kind(),
        nixe_cpu::exception::ExceptionKind::Breakpoint
    );
    assert_eq!(dispatch.syndrome(), Some(0x123));
    assert!(matches!(
        report.stop,
        crate::ExecutionStop::ArchitecturalException {
            kind: nixe_cpu::exception::ExceptionKind::Breakpoint,
            syndrome: Some(0x123),
            ..
        }
    ));
    assert!(report.to_string().contains("architectural-exception"));

    let mut data_fault = reference_process_builder().build(&plan).unwrap();
    replace_entry_instruction(&mut data_fault, 0xf940_0020); // LDR X0,[X1]
    let ThreadCpuState::A64(state) = &mut data_fault.main_thread_mut().state else {
        panic!("homebrew fixture must initialize A64");
    };
    state.write_x(
        nixe_cpu::state::a64::A64Register::General(a64_register(1)),
        0x1000,
    );
    let report = data_fault.run_reference(1).unwrap();
    assert_eq!(
        report.stop.exception_dispatch_request().unwrap().kind(),
        nixe_cpu::exception::ExceptionKind::DataAbort
    );
    assert!(matches!(
        report.stop,
        crate::ExecutionStop::DataFault { .. }
    ));
    assert!(report.to_string().contains("data-fault"));
}

#[test]
fn supervisor_calls_route_a64_a32_and_t32_with_current_runtime_context() {
    let cases = [
        (ExecutionState::A64, 0xd400_4681, 0x234),
        (ExecutionState::A32, 0xef12_3456, 0x12_3456),
        (ExecutionState::T32, 0xbf00_df7b, 0x7b),
    ];

    for (execution_state, encoding, immediate) in cases {
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
            process.main_thread_mut().state = ThreadCpuState::A32(Box::new(state));
        }

        let report = process.run_reference(1).unwrap();
        let expected_encoding = match execution_state {
            ExecutionState::T32 => InstructionEncoding::from_u16(encoding as u16),
            ExecutionState::A64 | ExecutionState::A32 => InstructionEncoding::from_u32(encoding),
        };
        let mut dispatcher = RecordingSupervisorCallDispatcher {
            expected_encoding: Some(expected_encoding),
            observed: None,
        };
        let outcome = process
            .route_supervisor_call(&report.stop, &mut dispatcher)
            .unwrap();

        assert_eq!(outcome, crate::ExceptionHandlingResult::Suspended);
        let (request, address_space, typed_thread, vcpu, thread_id, handle) =
            dispatcher.observed.unwrap();
        assert_eq!(request.kind(), ExceptionKind::SupervisorCall);
        assert_eq!(request.syndrome(), Some(immediate));
        assert_eq!(request.source().pc.get(), entry);
        assert_eq!(request.source().execution_state, execution_state);
        assert_eq!(address_space, process.cpu_context().address_space_id());
        assert_eq!(thread_id, 1);
        assert_eq!(typed_thread, nixe_scheduler::GuestThreadId::new(1));
        assert_eq!(vcpu, nixe_scheduler::VirtualCpuId::new(0));
        assert_eq!(handle, process.main_thread().handle);
        match &process.main_thread().state {
            ThreadCpuState::A64(state) => assert_eq!(
                state.read_x(nixe_cpu::state::a64::A64Register::General(a64_register(0))),
                0xfeed_face
            ),
            ThreadCpuState::A32(state) => {
                assert_eq!(state.read_r(a32_register(0)), 0xfeed_face)
            }
        }
    }
}

struct IdentityOutcomeDispatcher {
    expected_thread: nixe_scheduler::GuestThreadId,
    expected_vcpu: nixe_scheduler::VirtualCpuId,
    outcome: Option<crate::ExceptionDispatchOutcome<&'static str>>,
}

impl crate::ExceptionDispatcher for IdentityOutcomeDispatcher {
    type Fault = &'static str;

    fn dispatch(
        &mut self,
        context: &mut crate::ExceptionDispatchContext<'_>,
        _request: crate::ExceptionDispatchRequest,
    ) -> crate::ExceptionDispatchOutcome<Self::Fault> {
        assert_eq!(context.thread().id(), self.expected_thread);
        assert_eq!(context.thread().vcpu(), self.expected_vcpu);
        self.outcome.take().unwrap()
    }
}

#[test]
fn every_supervisor_outcome_targets_an_explicit_second_thread() {
    let cases = [
        (
            crate::ExceptionDispatchOutcome::Resume(crate::ExceptionResume::Next),
            nixe_scheduler::ThreadLifecycle::Ready,
        ),
        (
            crate::ExceptionDispatchOutcome::Suspend(crate::ExceptionResume::Retry),
            nixe_scheduler::ThreadLifecycle::Waiting,
        ),
        (
            crate::ExceptionDispatchOutcome::Reject {
                diagnostic: "guest",
            },
            nixe_scheduler::ThreadLifecycle::Ready,
        ),
        (
            crate::ExceptionDispatchOutcome::Terminate {
                scope: crate::ExceptionTerminationScope::CurrentThread,
                exit_code: 9,
                reason: crate::ExceptionTerminationReason::Requested,
            },
            nixe_scheduler::ThreadLifecycle::Exited,
        ),
        (
            crate::ExceptionDispatchOutcome::Fault("host"),
            nixe_scheduler::ThreadLifecycle::Faulted,
        ),
    ];
    for (outcome, expected_lifecycle) in cases {
        let (mut process, report, _) = process_stopped_at_svc(ExecutionState::A64);
        let second_id = nixe_scheduler::GuestThreadId::new(2);
        let mut second = process.main_thread().clone();
        second.id = second_id;
        second.object = crate::ThreadObject::new(second_id.get());
        second.lifecycle = nixe_scheduler::ThreadLifecycle::Waiting;
        process.threads.insert(second).unwrap();
        let vcpu = nixe_scheduler::VirtualCpuId::new(3);
        let mut dispatcher = IdentityOutcomeDispatcher {
            expected_thread: second_id,
            expected_vcpu: vcpu,
            outcome: Some(outcome),
        };
        process
            .route_supervisor_call_for(second_id, vcpu, &report.stop, &mut dispatcher)
            .unwrap();
        assert_eq!(
            process.thread(second_id).unwrap().lifecycle(),
            expected_lifecycle
        );
        if matches!(
            expected_lifecycle,
            nixe_scheduler::ThreadLifecycle::Exited | nixe_scheduler::ThreadLifecycle::Faulted
        ) {
            assert_eq!(
                process.lifecycle(),
                nixe_scheduler::ProcessLifecycle::Running
            );
        }
    }
}

#[test]
fn handled_supervisor_calls_advance_once_in_a64_a32_and_t32() {
    let cases = [
        (ExecutionState::A64, 4_u64),
        (ExecutionState::A32, 4_u64),
        (ExecutionState::T32, 2_u64),
    ];

    for (execution_state, width) in cases {
        let (mut process, report, entry) = process_stopped_at_svc(execution_state);
        let mut dispatcher = FixedSupervisorCallDispatcher {
            outcome: Some(crate::ExceptionDispatchOutcome::<&'static str>::Resume(
                crate::ExceptionResume::Next,
            )),
        };

        let result = process
            .route_supervisor_call(&report.stop, &mut dispatcher)
            .unwrap();

        assert_eq!(result, crate::ExceptionHandlingResult::Resumed);
        assert_eq!(process.execution_status(), ProcessExecutionStatus::Ready);
        assert_eq!(
            instruction_address(&process.main_thread_mut().state),
            entry + width
        );
        let next = process.run_reference(1).unwrap();
        assert!(!matches!(
            next.stop,
            crate::ExecutionStop::SupervisorCall { source, .. } if source.pc.get() == entry
        ));
    }
}

#[test]
fn supervisor_call_retry_is_explicit_and_reexecutes_the_source() {
    for execution_state in [
        ExecutionState::A64,
        ExecutionState::A32,
        ExecutionState::T32,
    ] {
        let (mut process, report, entry) = process_stopped_at_svc(execution_state);
        let mut dispatcher = PcMutatingSupervisorCallDispatcher {
            outcome: Some(crate::ExceptionDispatchOutcome::<&'static str>::Resume(
                crate::ExceptionResume::Retry,
            )),
        };

        assert_eq!(
            process
                .route_supervisor_call(&report.stop, &mut dispatcher)
                .unwrap(),
            crate::ExceptionHandlingResult::Resumed
        );
        assert_eq!(instruction_address(&process.main_thread_mut().state), entry);
        let retried = process.run_reference(1).unwrap();
        assert!(matches!(
            retried.stop,
            crate::ExecutionStop::SupervisorCall { source, .. } if source.pc.get() == entry
        ));
    }
}

#[test]
fn suspended_supervisor_call_installs_continuation_without_becoming_runnable() {
    let (mut process, report, entry) = process_stopped_at_svc(ExecutionState::A64);
    let mut dispatcher = FixedSupervisorCallDispatcher {
        outcome: Some(crate::ExceptionDispatchOutcome::<&'static str>::Suspend(
            crate::ExceptionResume::Next,
        )),
    };

    assert_eq!(
        process
            .route_supervisor_call(&report.stop, &mut dispatcher)
            .unwrap(),
        crate::ExceptionHandlingResult::Suspended
    );
    assert_eq!(
        instruction_address(&process.main_thread_mut().state),
        entry + 4
    );
    assert_eq!(
        process.execution_status(),
        ProcessExecutionStatus::Suspended
    );
    assert!(matches!(
        process.run_reference(1),
        Err(crate::ProcessExecutionError::NotRunnable {
            status: ProcessExecutionStatus::Suspended,
            ..
        })
    ));
    assert!(process.resume());
    assert_eq!(
        instruction_address(&process.main_thread_mut().state),
        entry + 4
    );
}

#[test]
fn faulted_supervisor_call_retains_source_and_cannot_run() {
    let (mut process, report, entry) = process_stopped_at_svc(ExecutionState::A64);
    let mut dispatcher = PcMutatingSupervisorCallDispatcher {
        outcome: Some(crate::ExceptionDispatchOutcome::Fault(
            "svc dispatch failed",
        )),
    };

    assert_eq!(
        process
            .route_supervisor_call(&report.stop, &mut dispatcher)
            .unwrap(),
        crate::ExceptionHandlingResult::Fault("svc dispatch failed")
    );
    assert_eq!(instruction_address(&process.main_thread_mut().state), entry);
    assert_eq!(process.execution_status(), ProcessExecutionStatus::Faulted);
    assert!(matches!(
        process.run_reference(1),
        Err(crate::ProcessExecutionError::NotRunnable {
            status: ProcessExecutionStatus::Faulted,
            ..
        })
    ));
    assert!(!process.resume());
}

#[test]
fn supervisor_call_termination_scope_is_preserved_through_teardown() {
    let cases = [
        (
            crate::ExceptionTerminationScope::CurrentThread,
            crate::ProcessExitCause::LastThreadExited,
        ),
        (
            crate::ExceptionTerminationScope::Process,
            crate::ProcessExitCause::ProcessRequested,
        ),
    ];

    for (scope, expected_cause) in cases {
        let (mut process, report, entry) = process_stopped_at_svc(ExecutionState::A64);
        let mut dispatcher = FixedSupervisorCallDispatcher {
            outcome: Some(crate::ExceptionDispatchOutcome::<&'static str>::Terminate {
                scope,
                exit_code: 0x55,
                reason: crate::ExceptionTerminationReason::Requested,
            }),
        };

        assert_eq!(
            process
                .route_supervisor_call(&report.stop, &mut dispatcher)
                .unwrap(),
            crate::ExceptionHandlingResult::Terminated {
                scope,
                exit_code: 0x55,
                reason: crate::ExceptionTerminationReason::Requested,
            }
        );
        assert_eq!(process.execution_status(), ProcessExecutionStatus::Exited);
        assert_eq!(process.exit().unwrap().cause, expected_cause);
        assert_eq!(process.exit().unwrap().source.unwrap().pc.get(), entry);
        assert_eq!(process.main_thread().exit().unwrap().requested_scope, scope);
        assert!(matches!(
            process.run_reference(1),
            Err(crate::ProcessExecutionError::NotRunnable {
                status: ProcessExecutionStatus::Exited,
                ..
            })
        ));

        let teardown = process.teardown();
        assert_eq!(teardown.previous_status, ProcessExecutionStatus::Exited);
        assert_eq!(teardown.exit.unwrap().cause, expected_cause);
        assert_eq!(teardown.threads_released, 1);
    }
}

#[test]
fn detailed_instruction_trace_is_opt_in_bounded_and_persistent_across_slices() {
    let (_directory, plan) = plan();
    let mut process = reference_process_builder()
        .with_diagnostics(crate::DiagnosticsPolicy {
            instruction_trace: true,
            ..crate::DiagnosticsPolicy::default()
        })
        .build(&plan)
        .unwrap();
    replace_entry_instruction(&mut process, 0x1400_0000); // B #0

    let first = process
        .run_reference(crate::MAX_INSTRUCTION_TRACE_ENTRIES as u64 + 3)
        .unwrap();
    assert!(first.trace.enabled());
    assert_eq!(
        first.trace.entries().len(),
        crate::MAX_INSTRUCTION_TRACE_ENTRIES
    );
    assert_eq!(first.trace.discarded(), 3);
    assert_eq!(first.trace.entries()[0].sequence, 3);
    assert_eq!(
        first.trace.entries().last().unwrap().sequence,
        crate::MAX_INSTRUCTION_TRACE_ENTRIES as u64 + 2
    );
    assert!(
        first
            .trace
            .entries()
            .iter()
            .all(|entry| entry.disassembly.as_deref() == Some("b imm=#0"))
    );
    assert!(first.trace.to_string().len() <= crate::MAX_INSTRUCTION_TRACE_EXPORT_BYTES);

    let second = process.run_reference(1).unwrap();
    assert_eq!(second.trace.discarded(), 4);
    assert_eq!(second.trace.entries()[0].sequence, 4);
    assert_eq!(
        second.trace.entries().last().unwrap().sequence,
        crate::MAX_INSTRUCTION_TRACE_ENTRIES as u64 + 3
    );
}

#[test]
fn sanitized_instruction_trace_omits_detailed_disassembly() {
    let (_directory, plan) = plan();
    let mut process = reference_process_builder()
        .with_diagnostics(crate::DiagnosticsPolicy {
            report_detail: crate::ReportDetail::Sanitized,
            instruction_trace: true,
            ..crate::DiagnosticsPolicy::default()
        })
        .build(&plan)
        .unwrap();

    let report = process.run_reference(1).unwrap();
    assert_eq!(report.trace.entries().len(), 1);
    assert!(report.trace.entries()[0].disassembly.is_none());
    assert!(!report.trace.to_string().contains("disassembly="));
}

#[test]
fn teardown_reports_resources_owned_by_the_process() {
    let (_directory, plan) = plan();
    let mut process = reference_process_builder().build(&plan).unwrap();
    assert!(process.terminate());
    assert_eq!(
        process.exit().unwrap().cause,
        crate::ProcessExitCause::HostRequested
    );
    assert_eq!(
        process.main_thread().exit().unwrap().requested_scope,
        crate::ExceptionTerminationScope::Process
    );

    let report = process.teardown();
    assert_eq!(
        report.previous_status,
        crate::ProcessExecutionStatus::Exited
    );
    assert_eq!(
        report.exit.unwrap().cause,
        crate::ProcessExitCause::HostRequested
    );
    assert_eq!(report.threads_released, 1);
    assert_eq!(report.modules_released, 1);
    assert!(report.mappings_released > 0);
    assert!(report.physical_pages_released > 0);
    assert_eq!(report.mounts_released, 0);
    assert_eq!(report.handles_released, 1);
}
