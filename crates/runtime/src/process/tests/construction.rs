use super::*;

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
        process.main_thread().state().execution_state(),
        ExecutionState::A64
    );
    let ThreadCpuState::A64(state) = process.main_thread().state() else {
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
        MemoryValue::U32(HOME_BREW_ARGV_KEY)
    );
    let argv_address = context.checked_add(HOME_BREW_ARGV_OFFSET as u64).unwrap();
    assert_eq!(
        process
            .memory()
            .read(
                process.cpu_context().address_space_id(),
                context
                    .checked_add((HOME_BREW_CONFIG_ENTRY_SIZE + 16) as u64)
                    .unwrap(),
                MemoryAccess::normal(MemoryAccessSize::Doubleword),
            )
            .unwrap()
            .value,
        MemoryValue::U64(argv_address.get())
    );
    let argv = crate::HOME_BREW_EXECUTABLE_ARGV0
        .bytes()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();
    let actual = (0..argv.len())
        .map(|offset| {
            let value = process
                .memory()
                .read(
                    process.cpu_context().address_space_id(),
                    argv_address.checked_add(offset as u64).unwrap(),
                    MemoryAccess::normal(MemoryAccessSize::Byte),
                )
                .unwrap()
                .value;
            let MemoryValue::U8(value) = value else {
                panic!("byte read must return a byte value");
            };
            value
        })
        .collect::<Vec<_>>();
    assert_eq!(actual, argv);
    assert_eq!(
        process
            .memory()
            .read(
                process.cpu_context().address_space_id(),
                context
                    .checked_add((HOME_BREW_CONFIG_ENTRY_SIZE * 2) as u64)
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
    let ThreadCpuState::A64(state) = process.main_thread_mut().state_mut() else {
        panic!("homebrew fixture must initialize A64");
    };
    state.write_x(A64Register::General(a64_register(0)), 0x1234_5678);

    let report = process.run(1).unwrap();

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
    assert_eq!(
        process.lifecycle(),
        nixe_scheduler::ProcessLifecycle::Exited
    );
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
    let teardown = process.try_teardown().unwrap();
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
    assert!(
        std::sync::Arc::get_mut(&mut process.memory)
            .unwrap()
            .map_page(
                space,
                alias,
                mapping.physical_page,
                MemoryPermissions::READ_WRITE
            )
    );
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
