use std::cell::RefCell;
use std::collections::BTreeSet;
use std::sync::{Arc, Barrier, Mutex, mpsc};
use std::time::{Duration, Instant};

use nixe_cpu::decode::{self, DecodeResult, DecodeSupport};
use nixe_cpu::execution::MemoryBinding;
use nixe_cpu::location::LocationDescriptor;
use nixe_cpu::memory::{
    CacheMaintenanceKind, CpuMemory, ExecutionMemory, MemoryAccess, MemoryAccessClass,
    MemoryAccessSize, MemoryAlignment, MemoryAttributes, MemoryOrdering, MemoryPermissions,
    MemoryValue, ProcessMemory, SyntheticMemory, SyntheticMmio,
};
use nixe_cpu::platform::TargetPlatform;
use nixe_cpu::profile::ProcessCpuContext;
use nixe_cpu::state::a64::{A64GeneralRegister, A64Register, A64State, Nzcv};
use nixe_cpu_interpreter::{
    InstructionStep, InterpreterContext, execute_one, execute_one_with_context,
};
use nixe_memory::{
    AddressSpaceId, CanonicalRangeTranslator, CpuVisibilityRequest, DeviceAccessDeclaration,
    DeviceVisibilityPoint, DeviceVisibilityRequest, DirectBackendPolicy, GuestPhysicalPageId,
    GuestVirtualAddress, MemoryInvalidationKind, MemoryInvalidationSource, NonCpuDeviceId,
    VisibilityCoordinator, VisibilityCoordinatorError,
};

use super::lookup::{RegionLookup, index_for_pc, lookup_salt};
use super::region::{RegionKey, RegionLimits, discover_region};
use super::*;

const CODE: u64 = 0x1000;
const DATA: u64 = 0x8000;
const SPACE: AddressSpaceId = AddressSpaceId::new(1);

struct DeviceWriteback {
    bytes: Box<[u8]>,
}

impl VisibilityCoordinator for DeviceWriteback {
    fn make_device_visible(
        &self,
        _request: DeviceVisibilityRequest,
        _canonical_bytes: &[u8],
    ) -> Result<(), VisibilityCoordinatorError> {
        Ok(())
    }

    fn make_cpu_visible(
        &self,
        _request: CpuVisibilityRequest,
    ) -> Result<Box<[u8]>, VisibilityCoordinatorError> {
        Ok(self.bytes.clone())
    }
}

#[test]
fn configured_diagnostics_dump_direct_clif_native_code_and_one_compact_report() {
    let root = tempfile::tempdir().unwrap();
    let dumps = root.path().join("dumps");
    let reports = root.path().join("reports");
    let process = JitProcess::with_configuration(
        cpu(),
        JitConfiguration::default()
            .with_dump_directory(Some(dumps.clone()))
            .with_performance_report_directory(Some(reports.clone()))
            .with_performance_report_title("Test Title"),
    )
    .unwrap();
    let memory = memory(&[(0, 0xd503_201f), (4, breakpoint(7))]);
    let mut state = state(CODE, 0);

    assert!(matches!(
        JitThread::new()
            .run(&process, &memory, &mut state, 1)
            .unwrap(),
        DirectExit::Budget { .. }
    ));
    process.shutdown();
    process.shutdown();

    let session = std::fs::read_dir(&dumps)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let dump_names: BTreeSet<_> = std::fs::read_dir(session)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert!(dump_names.iter().any(|name| name.ends_with(".clif")));
    assert!(dump_names.iter().any(|name| name.ends_with(".bin")));
    assert!(!dump_names.iter().any(|name| name.contains("nixe-ir")));

    let report_paths: Vec<_> = std::fs::read_dir(reports)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect();
    assert_eq!(report_paths.len(), 1);
    let report_path = &report_paths[0];
    assert!(
        report_path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .starts_with("test-title-")
    );
    let report: toml::Value =
        toml::from_str(&std::fs::read_to_string(report_path).unwrap()).unwrap();
    let table = report.as_table().unwrap();
    assert_eq!(table["version"].as_integer(), Some(7));
    let expected: BTreeSet<_> = [
        "version",
        "memory_backend",
        "memory_backend_reason",
        "regions_discovered",
        "guest_blocks_discovered",
        "regions_compiled",
        "region_entry_points",
        "secondary_entry_hits",
        "compiled_guest_instructions",
        "unique_guest_instructions",
        "overlapping_guest_instructions",
        "lookup_hits",
        "lookup_misses",
        "guest_instructions",
        "clif_instructions",
        "native_bytes",
        "compile_time_ns",
        "native_time_ns",
        "slow_memory_calls",
        "direct_faults",
        "compiled_direct_accesses",
        "direct_memory",
        "invalidations",
        "invalidation_details",
        "exit_reasons",
    ]
    .into_iter()
    .collect();
    assert_eq!(
        table.keys().map(String::as_str).collect::<BTreeSet<_>>(),
        expected
    );
    assert_eq!(table["regions_compiled"].as_integer(), Some(1));
    assert_eq!(table["memory_backend"].as_str(), Some("unbound"));
    assert_eq!(table["region_entry_points"].as_integer(), Some(1));
    assert_eq!(table["compiled_guest_instructions"].as_integer(), Some(2));
    assert_eq!(table["unique_guest_instructions"].as_integer(), Some(2));
    assert_eq!(
        table["overlapping_guest_instructions"].as_integer(),
        Some(0)
    );
    assert_eq!(table["lookup_misses"].as_integer(), Some(1));
    assert_eq!(table["guest_instructions"].as_integer(), Some(1));
    assert!(table["clif_instructions"].as_integer().unwrap() > 0);
    assert!(table["native_bytes"].as_integer().unwrap() > 0);
    assert_eq!(
        table["direct_memory"]["writable_alias_pages_armed"].as_integer(),
        Some(0)
    );
    assert_eq!(
        table["direct_memory"]["writable_alias_pages_revoked"].as_integer(),
        Some(0)
    );
    assert_eq!(
        table["direct_memory"]["transition_safepoint_notifications"].as_integer(),
        Some(0)
    );
    assert_eq!(
        table["compiled_direct_accesses"]["read_8"].as_integer(),
        Some(0)
    );
    assert_eq!(
        table["invalidation_details"]["regions_retired"].as_integer(),
        Some(0)
    );
    assert_eq!(
        table["invalidation_details"]["mapping"].as_integer(),
        Some(1)
    );
    assert_eq!(
        table["invalidation_details"]["irrelevant"].as_integer(),
        Some(1)
    );
    assert_eq!(table["exit_reasons"]["budget"].as_integer(), Some(1));
}

#[test]
fn direct_diagnostics_count_compiled_access_sites_by_kind_and_width() {
    let root = tempfile::tempdir().unwrap();
    let reports = root.path().join("reports");
    let mut memory = ExecutionMemory::new();
    let code_page = GuestPhysicalPageId::new(1);
    let data_page = GuestPhysicalPageId::new(2);
    assert!(memory.add_ram_page(code_page));
    assert!(memory.add_ram_page(data_page));
    for (offset, encoding) in [
        (0, 0x3940_0020_u32),  // LDRB W0,[X1]
        (4, 0x3900_0020_u32),  // STRB W0,[X1]
        (8, 0x7940_0020_u32),  // LDRH W0,[X1]
        (12, 0x7900_0020_u32), // STRH W0,[X1]
        (16, 0xb940_0020_u32), // LDR W0,[X1]
        (20, 0xb900_0020_u32), // STR W0,[X1]
        (24, 0xf940_0020_u32), // LDR X0,[X1]
        // Register-offset lowering can duplicate the faultable guard while
        // arranging native control flow. It is still one logical guest site.
        (28, 0xf822_7820_u32), // STR X0,[X1,X2,LSL#3]
        (32, 0x3dc0_0020_u32), // LDR Q0,[X1]
        (36, 0x3d80_0020_u32), // STR Q0,[X1]
        (40, breakpoint(0)),
    ] {
        assert!(memory.initialize_ram(code_page, offset, &encoding.to_le_bytes()));
    }
    assert!(memory.map_page(
        SPACE,
        GuestVirtualAddress::new(CODE),
        code_page,
        MemoryPermissions::READ_EXECUTE,
    ));
    assert!(memory.map_page(
        SPACE,
        GuestVirtualAddress::new(DATA),
        data_page,
        MemoryPermissions::READ_WRITE,
    ));
    memory
        .bind_cpu_memory_backend(SPACE, 0x1_0000, DirectBackendPolicy::Required)
        .unwrap();

    let process = JitProcess::with_configuration(
        cpu(),
        JitConfiguration::default()
            .with_performance_report_directory(Some(reports.clone()))
            .with_performance_report_title("Direct access counts"),
    )
    .unwrap();
    let binding = MemoryBinding {
        address_space: SPACE,
        end_exclusive: GuestVirtualAddress::new(0x1_0000),
        memory: &memory,
        mapping_epoch: memory.mapping_epoch().get(),
        invalidation_cursor: memory.invalidation_cursor(),
    };
    process.bind_memory(binding).unwrap();
    let mut thread = JitThread::new();
    thread.synchronize_address_space(&process, binding).unwrap();
    let mut state = memory_state();
    thread.run(&process, &memory, &mut state, 8).unwrap();
    process.shutdown();

    let report_path = std::fs::read_dir(reports)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let report: toml::Value =
        toml::from_str(&std::fs::read_to_string(report_path).unwrap()).unwrap();
    assert_eq!(report["memory_backend"].as_str(), Some("linux_direct"));
    assert!(
        report["memory_backend_reason"]
            .as_str()
            .is_some_and(|reason| reason.contains("host capability validation"))
    );
    let accesses = &report["compiled_direct_accesses"];
    for width in [1, 2, 4, 8] {
        assert_eq!(
            accesses[format!("read_{width}")].as_integer(),
            Some(1),
            "unexpected direct read sites for width {width}: {accesses}"
        );
        assert_eq!(
            accesses[format!("write_{width}")].as_integer(),
            Some(1),
            "unexpected direct write sites for width {width}: {accesses}"
        );
    }
    assert_eq!(accesses["read_16"].as_integer(), Some(1));
    assert_eq!(accesses["write_16"].as_integer(), Some(1));
    assert_eq!(
        report["direct_faults"]["tracking_writes"].as_integer(),
        Some(0),
        "statically detected first writes complete without a host signal"
    );
    assert!(
        report["direct_memory"]["writable_alias_pages_armed"]
            .as_integer()
            .is_some_and(|value| value >= 1)
    );
    assert!(
        report["direct_memory"]["vma_samples"]
            .as_integer()
            .is_some_and(|value| value >= 1)
    );
}

#[test]
fn normalized_multiblock_region_executes_without_nixe_ir() {
    let memory = memory(&[
        (0x00, 0xd503_201f), // NOP
        (0x04, branch(CODE + 4, CODE + 12)),
        (0x08, add_x0(7)),
        (0x0c, add_x0(1)),
        (0x10, breakpoint(0x31)),
    ]);
    let cpu = cpu();
    let region = discover_region(
        cpu,
        &memory,
        location(CODE),
        RegionLimits::default(),
        |_| false,
    )
    .unwrap();
    assert_eq!(region.blocks.len(), 2);
    assert_eq!(region.instruction_count, 4);
    assert_eq!(region.external_exits.len(), 1);

    let process = JitProcess::new(cpu).unwrap();
    let thread = JitThread::new();
    let mut state = state(CODE, 10);
    let exit = thread.run(&process, &memory, &mut state, 10).unwrap();
    assert_eq!(
        exit,
        DirectExit::Architectural {
            pc: GuestVirtualAddress::new(CODE + 0x10),
            detail: (2 << 24) | 0x31,
            instructions: 4,
        }
    );
    assert_eq!(read_x0(&state), 11);
}

#[test]
fn multiblock_region_publishes_one_native_function_for_every_block_entry() {
    let memory = memory(&[
        (0x00, 0xd503_201f),
        (0x04, branch(CODE + 4, CODE + 12)),
        (0x0c, add_x0(1)),
        (0x10, breakpoint(0x32)),
    ]);
    let process = JitProcess::new(cpu()).unwrap();
    let primary = process.entry_for(&memory, location(CODE)).unwrap().1;
    let secondary = process.entry_for(&memory, location(CODE + 0x0c)).unwrap().1;
    assert_eq!(secondary, primary);
    {
        let state = process.state.lock().unwrap();
        assert_eq!(state.compiled_regions, 1);
        let secondary_key = RegionKey::new(cpu(), location(CODE + 0x0c));
        assert_eq!(
            state.lookup.get(secondary_key).unwrap().owner.start.get(),
            CODE
        );
    }

    let mut state = state(CODE + 0x0c, 10);
    assert_eq!(
        JitThread::new()
            .run(&process, &memory, &mut state, 4)
            .unwrap(),
        DirectExit::Architectural {
            pc: GuestVirtualAddress::new(CODE + 0x10),
            detail: (2 << 24) | 0x32,
            instructions: 2,
        }
    );
    assert_eq!(read_x0(&state), 11);
}

#[test]
fn discovery_stops_at_an_existing_secondary_region_entry() {
    let target = CODE + 0x100;
    let secondary = target + 0x0c;
    let memory = memory(&[
        (0x00, branch(CODE, secondary)),
        (0x100, 0xd503_201f),
        (0x104, branch(target + 4, secondary)),
        (0x10c, breakpoint(4)),
    ]);
    let process = JitProcess::new(cpu()).unwrap();
    let target_entry = process.entry_for(&memory, location(target)).unwrap().1;
    assert_eq!(
        process.entry_for(&memory, location(secondary)).unwrap().1,
        target_entry
    );
    process.entry_for(&memory, location(CODE)).unwrap();

    let state = process.state.lock().unwrap();
    assert_eq!(state.compiled_regions, 2);
    let source = state
        .region_for(RegionKey::new(cpu(), location(CODE)))
        .unwrap();
    assert_eq!(source.guest_blocks, 1);
    assert_eq!(source.entry_keys.len(), 1);
    assert_eq!(source.links[0].slot.load(Ordering::Acquire), target_entry);
}

#[test]
fn conditional_branch_consumes_lazy_flags_in_native_cfg() {
    let memory = memory(&[
        (0x00, 0xf100_0400),                                // SUBS X0,X0,#1
        (0x04, conditional_branch(CODE + 4, CODE + 12, 1)), // B.NE
        (0x08, breakpoint(1)),
        (0x0c, breakpoint(2)),
    ]);
    let process = JitProcess::new(cpu()).unwrap();
    let thread = JitThread::new();

    let mut zero = state(CODE, 1);
    assert_eq!(
        thread.run(&process, &memory, &mut zero, 8).unwrap(),
        DirectExit::Architectural {
            pc: GuestVirtualAddress::new(CODE + 8),
            detail: (2 << 24) | 1,
            instructions: 3,
        }
    );
    assert!(zero.nzcv().zero());

    let mut nonzero = state(CODE, 2);
    assert_eq!(
        thread.run(&process, &memory, &mut nonzero, 8).unwrap(),
        DirectExit::Architectural {
            pc: GuestVirtualAddress::new(CODE + 12),
            detail: (2 << 24) | 2,
            instructions: 3,
        }
    );
    assert!(!nonzero.nzcv().zero());
}

#[test]
fn cold_unsupported_system_instruction_does_not_reject_the_region() {
    let memory = memory(&[
        (0x00, 0xd53b_00e5),                                     // MRS X5,DCZID_EL0
        (0x04, 0x9240_10a5),                                     // AND X5,X5,#0x1f
        (0x08, 0xf100_10bf),                                     // CMP X5,#4
        (0x0c, conditional_branch(CODE + 0x0c, CODE + 0x14, 1)), // B.NE
        (0x10, 0xd50b_7423), // DC ZVA,X3 (prohibited by Switch 1 DCZID_EL0)
        (0x14, 0xd503_201f), // NOP
        (0x18, branch(CODE + 0x18, CODE + 0x18)),
    ]);
    let process = JitProcess::new(cpu()).unwrap();
    let mut state = state(CODE, 0);

    assert_eq!(
        JitThread::new()
            .run(&process, &memory, &mut state, 5)
            .unwrap(),
        DirectExit::Budget {
            pc: GuestVirtualAddress::new(CODE + 0x18),
            instructions: 5,
        }
    );
    assert_eq!(read_register(&state, 5), 0x14);
}

#[test]
fn direct_backedge_keeps_register_ssa_until_precise_budget_exit() {
    let memory = memory(&[(0x00, add_x0(1)), (0x04, branch(CODE + 4, CODE))]);
    let process = JitProcess::new(cpu()).unwrap();
    let thread = JitThread::new();
    let mut state = state(CODE, 0);
    assert_eq!(
        thread.run(&process, &memory, &mut state, 5).unwrap(),
        DirectExit::Budget {
            pc: GuestVirtualAddress::new(CODE + 4),
            instructions: 5,
        }
    );
    assert_eq!(read_x0(&state), 3);

    let key = RegionKey::new(cpu(), location(CODE));
    let process_state = process.state.lock().unwrap();
    let compiled = process_state.region_for(key).unwrap();
    assert_eq!(compiled.guest_blocks, 1);
    assert!(compiled.native_bytes > 0);
    assert_eq!(compiled.dependencies.len(), 1);
    assert_eq!(compiled.register_loads, 1);
    assert_eq!(compiled.register_stores, 1);
}

#[test]
fn call_and_return_use_canonical_state_only_at_region_boundaries() {
    let memory = memory(&[
        (0x00, branch_link(CODE, CODE + 0x10)),
        (0x04, breakpoint(3)),
        (0x10, add_x0(1)),
        (0x14, 0xd65f_03c0), // RET X30
    ]);
    let process = JitProcess::new(cpu()).unwrap();
    let thread = JitThread::new();
    let mut state = state(CODE, 4);
    assert_eq!(
        thread.run(&process, &memory, &mut state, 10).unwrap(),
        DirectExit::Architectural {
            pc: GuestVirtualAddress::new(CODE + 4),
            detail: (2 << 24) | 3,
            instructions: 4,
        }
    );
    assert_eq!(read_x0(&state), 5);
    assert_eq!(
        state.read_x(A64Register::General(A64GeneralRegister::new(30).unwrap())),
        CODE + 4
    );

    let process_state = process.state.lock().unwrap();
    let source = process_state
        .region_for(RegionKey::new(cpu(), location(CODE)))
        .unwrap();
    let target = process_state
        .lookup
        .get(RegionKey::new(cpu(), location(CODE + 0x10)))
        .unwrap();
    assert_eq!(source.links.len(), 1);
    assert_eq!(source.links[0].slot.load(Ordering::Acquire), target.entry);
}

#[test]
fn indirect_branch_and_return_chain_through_the_native_lookup() {
    let source_key = RegionKey::new(cpu(), location(CODE));
    let salt = lookup_salt(source_key);
    let source_slot = index_for_pc(CODE, salt);
    let target = (CODE + 4..)
        .step_by(4)
        .find(|pc| index_for_pc(*pc, salt) == source_slot)
        .unwrap();
    let return_address = CODE + 0x80;
    let mut memory = memory(&[(0x00, 0xd61f_0000), (0x80, breakpoint(7))]); // BR X0
    let target_page = GuestPhysicalPageId::new(2);
    let target_base = target & !0xfff;
    let target_offset = usize::try_from(target - target_base).unwrap();
    assert!(memory.add_ram_page(target_page));
    assert!(memory.initialize_ram(target_page, target_offset, &add_x0(1).to_le_bytes()));
    assert!(memory.initialize_ram(
        target_page,
        target_offset + 4,
        &0xd65f_03c0_u32.to_le_bytes(),
    ));
    assert!(memory.map_page(
        SPACE,
        GuestVirtualAddress::new(target_base),
        target_page,
        MemoryPermissions::READ_EXECUTE,
    ));
    let process = JitProcess::new(cpu()).unwrap();
    for pc in [target, return_address, CODE] {
        process.entry_for(&memory, location(pc)).unwrap();
    }

    let thread = JitThread::new();
    let mut state = state(CODE, target);
    write_register(&mut state, 30, return_address);
    assert_eq!(
        thread.run(&process, &memory, &mut state, 8).unwrap(),
        DirectExit::Architectural {
            pc: GuestVirtualAddress::new(return_address),
            detail: (2 << 24) | 7,
            instructions: 4,
        }
    );
    assert_eq!(read_x0(&state), target + 1);
    assert_eq!(thread.rust_dispatches.load(Ordering::Relaxed), 0);
}

#[test]
fn budget_and_control_exit_before_the_next_guest_instruction() {
    let memory = memory(&[(0x00, 0xd503_201f), (0x04, breakpoint(0))]);
    let process = JitProcess::new(cpu()).unwrap();

    let thread = JitThread::new();
    let mut budget_state = state(CODE, 0);
    assert_eq!(
        thread.run(&process, &memory, &mut budget_state, 1).unwrap(),
        DirectExit::Budget {
            pc: GuestVirtualAddress::new(CODE + 4),
            instructions: 1,
        }
    );

    let preempted = JitThread::new();
    preempted.request_preempt();
    let mut control_state = state(CODE, 0);
    assert_eq!(
        preempted
            .run(&process, &memory, &mut control_state, 10)
            .unwrap(),
        DirectExit::Control {
            pc: GuestVirtualAddress::new(CODE),
            instructions: 0,
        }
    );
}

#[test]
fn one_process_lock_produces_one_synchronous_compilation_flight() {
    let memory = Arc::new(memory(&[(0x00, breakpoint(0))]));
    let process = Arc::new(JitProcess::new(cpu()).unwrap());
    let mut joins = Vec::new();
    for _ in 0..4 {
        let memory = Arc::clone(&memory);
        let process = Arc::clone(&process);
        joins.push(std::thread::spawn(move || {
            process
                .entry_for(memory.as_ref(), location(CODE))
                .unwrap()
                .1
        }));
    }
    let entries: Vec<_> = joins.into_iter().map(|join| join.join().unwrap()).collect();
    assert!(entries.windows(2).all(|pair| pair[0] == pair[1]));
    assert_eq!(process.state.lock().unwrap().compiled_regions, 1);
}

#[test]
fn direct_lookup_uses_collision_fallback_without_replacing_primary_slot() {
    let first = RegionKey::new(cpu(), location(CODE));
    let salt = lookup_salt(first);
    let first_slot = index_for_pc(first.start.get(), salt);
    let second_pc = (CODE + 4..)
        .step_by(4)
        .find(|pc| index_for_pc(*pc, salt) == first_slot)
        .unwrap();
    let second = RegionKey::new(cpu(), location(second_pc));
    let mut lookup = RegionLookup::new();
    lookup.insert(first, first, 1);
    lookup.insert(second, second, 2);
    assert_eq!(lookup.get(first).unwrap().entry, 1);
    assert_eq!(lookup.get(second).unwrap().entry, 2);
    assert_eq!(lookup.collision_count(), 1);
    assert_eq!(lookup.native_entry(first), 1);
    lookup.remove(first).unwrap();
    assert_eq!(lookup.native_entry(first), 0);
    assert_eq!(lookup.get(second).unwrap().entry, 2);
}

#[test]
fn executable_invalidation_unlinks_the_target_and_keeps_native_code_allocated() {
    let target = CODE + 0x1000;
    let mut memory = SyntheticMemory::new();
    let source_page = GuestPhysicalPageId::new(1);
    let target_page = GuestPhysicalPageId::new(2);
    assert!(memory.add_ram_page(source_page));
    assert!(memory.add_ram_page(target_page));
    assert!(memory.initialize_ram(source_page, 0, &branch_link(CODE, target).to_le_bytes(),));
    assert!(memory.initialize_ram(target_page, 0, &breakpoint(1).to_le_bytes()));
    assert!(memory.map_page(
        SPACE,
        GuestVirtualAddress::new(CODE),
        source_page,
        MemoryPermissions::READ_EXECUTE,
    ));
    assert!(memory.map_page(
        SPACE,
        GuestVirtualAddress::new(target),
        target_page,
        MemoryPermissions::READ_EXECUTE,
    ));
    assert!(memory.map_page(
        SPACE,
        GuestVirtualAddress::new(DATA),
        target_page,
        MemoryPermissions::READ_WRITE,
    ));

    let process = JitProcess::new(cpu()).unwrap();
    process.entry_for(&memory, location(CODE)).unwrap();
    process.entry_for(&memory, location(target)).unwrap();
    let source_key = RegionKey::new(cpu(), location(CODE));
    let target_key = RegionKey::new(cpu(), location(target));
    let (old_target, source_link) = {
        let state = process.state.lock().unwrap();
        let source = state.region_for(source_key).unwrap();
        let target = state.region_for(target_key).unwrap();
        assert_eq!(source.links[0].slot.load(Ordering::Acquire), target.entry);
        (Arc::clone(target), Arc::clone(&source.links[0].slot))
    };

    memory
        .write(
            SPACE,
            GuestVirtualAddress::new(DATA),
            MemoryAccess::normal(MemoryAccessSize::Word),
            MemoryValue::U32(breakpoint(2)),
        )
        .unwrap();
    process.reconcile(&memory).unwrap();
    {
        let state = process.state.lock().unwrap();
        assert!(state.lookup.get(source_key).is_some());
        assert!(state.lookup.get(target_key).is_some());
        assert!(state.retired.is_empty());
    }
    memory
        .maintain_cache(
            SPACE,
            CacheMaintenanceKind::InstructionInvalidate,
            Some(GuestVirtualAddress::new(DATA)),
        )
        .unwrap();
    process.reconcile(&memory).unwrap();
    {
        let state = process.state.lock().unwrap();
        assert!(state.lookup.get(source_key).is_some());
        assert!(state.lookup.get(target_key).is_none());
        assert_eq!(source_link.load(Ordering::Acquire), 0);
        assert!(
            state
                .retired
                .iter()
                .any(|region| Arc::ptr_eq(region, &old_target))
        );
    }

    let new_entry = process.entry_for(&memory, location(target)).unwrap().1;
    assert_ne!(new_entry, old_target.entry);
    assert_eq!(source_link.load(Ordering::Acquire), new_entry);
}

#[test]
fn host_write_to_an_executable_alias_invalidates_immediately() {
    let mut memory = memory(&[(0, breakpoint(1))]);
    assert!(memory.map_page(
        SPACE,
        GuestVirtualAddress::new(DATA),
        GuestPhysicalPageId::new(1),
        MemoryPermissions::READ_WRITE,
    ));
    let process = JitProcess::new(cpu()).unwrap();
    let key = RegionKey::new(cpu(), location(CODE));
    process.entry_for(&memory, location(CODE)).unwrap();

    memory
        .write_bytes(
            SPACE,
            GuestVirtualAddress::new(DATA),
            &breakpoint(2).to_le_bytes(),
        )
        .unwrap();
    process.reconcile(&memory).unwrap();

    let state = process.state.lock().unwrap();
    assert!(state.lookup.get(key).is_none());
    assert_eq!(state.retired.len(), 1);
}

#[test]
fn invalidation_unpublishes_every_entry_of_one_native_region() {
    let mut memory = memory(&[(0x00, branch(CODE, CODE + 0x0c)), (0x0c, breakpoint(1))]);
    assert!(memory.map_page(
        SPACE,
        GuestVirtualAddress::new(DATA),
        GuestPhysicalPageId::new(1),
        MemoryPermissions::READ_WRITE,
    ));
    let process = JitProcess::new(cpu()).unwrap();
    let primary = RegionKey::new(cpu(), location(CODE));
    let secondary = RegionKey::new(cpu(), location(CODE + 0x0c));
    process.entry_for(&memory, location(CODE)).unwrap();
    {
        let state = process.state.lock().unwrap();
        assert!(state.lookup.get(primary).is_some());
        assert!(state.lookup.get(secondary).is_some());
    }

    memory
        .write_bytes(
            SPACE,
            GuestVirtualAddress::new(DATA),
            &breakpoint(2).to_le_bytes(),
        )
        .unwrap();
    process.reconcile(&memory).unwrap();

    let state = process.state.lock().unwrap();
    assert!(state.lookup.get(primary).is_none());
    assert!(state.lookup.get(secondary).is_none());
    assert!(state.regions.is_empty());
    assert_eq!(state.retired.len(), 1);
}

#[test]
fn mapping_invalidation_retires_only_overlapping_code() {
    let mut memory = memory(&[(0, breakpoint(0))]);
    let process = JitProcess::new(cpu()).unwrap();
    process.entry_for(&memory, location(CODE)).unwrap();
    let key = RegionKey::new(cpu(), location(CODE));
    assert!(memory.add_ram_page(GuestPhysicalPageId::new(2)));
    assert!(memory.map_page(
        SPACE,
        GuestVirtualAddress::new(DATA),
        GuestPhysicalPageId::new(2),
        MemoryPermissions::READ_WRITE,
    ));
    process.reconcile(&memory).unwrap();
    {
        let state = process.state.lock().unwrap();
        assert!(state.lookup.get(key).is_some());
        assert!(state.retired.is_empty());
    }
    memory
        .set_permissions(
            SPACE,
            GuestVirtualAddress::new(CODE),
            nixe_cpu::memory::SYNTHETIC_PAGE_SIZE as u64,
            MemoryPermissions::READ,
        )
        .unwrap();
    process.reconcile(&memory).unwrap();
    let state = process.state.lock().unwrap();
    assert!(state.lookup.get(key).is_none());
    assert_eq!(state.retired.len(), 1);
}

#[test]
fn mapping_overlap_is_half_open_and_safe_at_the_top_of_address_space() {
    let ordinary = nixe_cpu::memory::CodePageSpan::containing(
        GuestVirtualAddress::new(0x1000),
        Some(GuestVirtualAddress::new(0x2000)),
        GuestVirtualAddress::new(0x1000),
    )
    .unwrap();
    assert!(!mapping_range_overlaps(
        ordinary,
        GuestVirtualAddress::new(0x0000),
        0x1000,
    ));
    assert!(mapping_range_overlaps(
        ordinary,
        GuestVirtualAddress::new(0x1fff),
        1,
    ));
    assert!(!mapping_range_overlaps(
        ordinary,
        GuestVirtualAddress::new(0x2000),
        0x1000,
    ));

    let top = nixe_cpu::memory::CodePageSpan::containing(
        GuestVirtualAddress::new(u64::MAX - 0xfff),
        None,
        GuestVirtualAddress::new(u64::MAX),
    )
    .unwrap();
    assert!(mapping_range_overlaps(
        top,
        GuestVirtualAddress::new(u64::MAX),
        1,
    ));
}

struct BlockingTimer {
    entered: Arc<Barrier>,
    release: Arc<Barrier>,
}

impl nixe_cpu::execution::ArchitecturalTimer for BlockingTimer {
    fn snapshot(&self) -> nixe_cpu::execution::TimerSnapshot {
        self.entered.wait();
        self.release.wait();
        nixe_cpu::execution::TimerSnapshot {
            counter: 0,
            frequency: 19_200_000,
        }
    }
}

fn blocking_native_loop() -> (SyntheticMemory, BlockingTimer) {
    let entered = Arc::new(Barrier::new(2));
    let release = Arc::new(Barrier::new(2));
    let mut memory = memory(&[
        (0, 0xd53b_e020), // MRS X0,CNTVCT_EL0
        (4, branch(CODE + 4, CODE)),
    ]);
    assert!(memory.map_page(
        SPACE,
        GuestVirtualAddress::new(DATA),
        GuestPhysicalPageId::new(1),
        MemoryPermissions::READ_WRITE,
    ));
    (memory, BlockingTimer { entered, release })
}

#[test]
fn running_native_backedge_observes_concurrent_executable_invalidation() {
    let (memory, timer) = blocking_native_loop();
    let entered = Arc::clone(&timer.entered);
    let release = Arc::clone(&timer.release);
    let timer = Arc::new(timer);
    let memory = Arc::new(memory);
    let process = Arc::new(JitProcess::new(cpu()).unwrap());
    process.entry_for(memory.as_ref(), location(CODE)).unwrap();
    let thread = Arc::new(JitThread::new());
    let worker_memory = Arc::clone(&memory);
    let worker_process = Arc::clone(&process);
    let worker_thread = Arc::clone(&thread);
    let worker_timer = Arc::clone(&timer);
    let worker = std::thread::spawn(move || {
        let mut state = state(CODE, 0);
        worker_thread.run_with_runtime(
            worker_process.as_ref(),
            NativeRunRequest {
                memory: worker_memory.as_ref(),
                state: &mut state,
                instruction_budget: u64::MAX,
                loader_return: None,
                timer: worker_timer.as_ref(),
                events: &worker_thread.events,
            },
        )
    });
    entered.wait();
    memory
        .write(
            SPACE,
            GuestVirtualAddress::new(DATA),
            MemoryAccess::normal(MemoryAccessSize::Word),
            MemoryValue::U32(breakpoint(9)),
        )
        .unwrap();
    memory
        .maintain_cache(
            SPACE,
            CacheMaintenanceKind::InstructionInvalidate,
            Some(GuestVirtualAddress::new(DATA)),
        )
        .unwrap();
    release.wait();
    assert_eq!(
        worker.join().unwrap().unwrap(),
        DirectExit::Architectural {
            pc: GuestVirtualAddress::new(CODE),
            detail: (2 << 24) | 9,
            instructions: 3,
        }
    );
    let state = process.state.lock().unwrap();
    assert_eq!(state.retired.len(), 1);
    assert_eq!(state.compiled_regions, 2);
}

#[test]
fn shutdown_unlinks_code_and_stops_a_running_native_backedge() {
    let (memory, timer) = blocking_native_loop();
    let entered = Arc::clone(&timer.entered);
    let release = Arc::clone(&timer.release);
    let timer = Arc::new(timer);
    let memory = Arc::new(memory);
    let process = Arc::new(JitProcess::new(cpu()).unwrap());
    process.entry_for(memory.as_ref(), location(CODE)).unwrap();
    let thread = Arc::new(JitThread::new());
    let worker_memory = Arc::clone(&memory);
    let worker_process = Arc::clone(&process);
    let worker_thread = Arc::clone(&thread);
    let worker_timer = Arc::clone(&timer);
    let worker = std::thread::spawn(move || {
        let mut state = state(CODE, 0);
        worker_thread.run_with_runtime(
            worker_process.as_ref(),
            NativeRunRequest {
                memory: worker_memory.as_ref(),
                state: &mut state,
                instruction_budget: u64::MAX,
                loader_return: None,
                timer: worker_timer.as_ref(),
                events: &worker_thread.events,
            },
        )
    });
    entered.wait();
    process.shutdown();
    release.wait();
    let error = worker.join().unwrap().unwrap_err();
    assert_eq!(error.kind, DirectJitErrorKind::Shutdown);
    process.shutdown();
    let state = process.state.lock().unwrap();
    assert!(state.lookup.keys().next().is_none());
    assert_eq!(state.retired.len(), 1);
}

#[test]
fn every_platform_scalar_control_catalog_entry_compiles_directly() {
    const INTEGER_FIXTURES: [u32; 17] = [
        0x9100_0400,
        0xd280_0020,
        0x8b01_0000,
        0x8b21_4000,
        0x9a01_0000,
        0x9240_0000,
        0xaa01_0000,
        0xd340_fc00,
        0x93c1_0400,
        0x9ac1_2000,
        0xfa41_0000,
        0xfa41_0800,
        0x9a81_0000,
        0x9b01_0800,
        0xdac0_1000,
        0x1000_0000,
        0x9000_0000,
    ];
    const CONTROL_FIXTURES: [u32; 11] = [
        0xd503_201f,
        0x1400_0000,
        0x9400_0000,
        0xd61f_0000,
        0xd63f_0000,
        0xd65f_03c0,
        0x5400_0000,
        0xb400_0000,
        0x3600_0000,
        0xd400_0001,
        0xd420_0000,
    ];
    const EXPECTED_IDS: [u32; 28] = [
        0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x10, 0x11, 0x12, 0x13, 0x14,
        0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x20, 0x21, 0x44, 0x45,
    ];

    for platform in [TargetPlatform::Switch1, TargetPlatform::Switch2] {
        let cpu = ProcessCpuContext::for_platform(platform, SPACE);
        let mut covered = BTreeSet::new();
        for encoding in INTEGER_FIXTURES.into_iter().chain(CONTROL_FIXTURES) {
            let memory = memory(&[(0, encoding), (4, breakpoint(0))]);
            let location = location_for(cpu, CODE);
            let decoded = match decode::decode(cpu.decoder(), location, encoding.into()) {
                DecodeResult::Decoded(decoded) => decoded,
                result => panic!("{platform:?} rejected {encoding:#010x}: {result:?}"),
            };
            covered.insert(decoded.instruction.coverage_id().get());
            JitProcess::new(cpu)
                .unwrap()
                .entry_for(&memory, location)
                .unwrap_or_else(|error| {
                    panic!("{platform:?} failed direct compilation of {encoding:#010x}: {error}")
                });
        }
        let catalog: BTreeSet<_> = nixe_cpu::decode::a64::patterns()
            .iter()
            .filter(|pattern| {
                pattern.decoder == DecodeSupport::Ready
                    && scalar_control_catalog_id(pattern.coverage_id.get())
            })
            .map(|pattern| pattern.coverage_id.get())
            .collect();
        assert_eq!(catalog, EXPECTED_IDS.into_iter().collect());
        assert_eq!(covered, catalog);
    }
}

fn scalar_control_catalog_id(id: u32) -> bool {
    matches!(
        id,
        0x01 | 0x02 | 0x03 | 0x04..=0x0a | 0x10..=0x1d | 0x20 | 0x21 | 0x44 | 0x45
    )
}

#[test]
fn every_platform_memory_system_catalog_entry_compiles_directly() {
    for platform in [TargetPlatform::Switch1, TargetPlatform::Switch2] {
        let cpu = ProcessCpuContext::for_platform(platform, SPACE);
        let catalog: Vec<_> = nixe_cpu::decode::a64::patterns()
            .iter()
            .filter(|pattern| {
                pattern.decoder == DecodeSupport::Ready
                    && matches!(pattern.coverage_id.get(), 0x0b..=0x0f | 0x22..=0x2f)
            })
            .collect();
        let mut covered = BTreeSet::new();
        for pattern in catalog.iter().copied() {
            let encoding = pattern
                .regression_fixture
                .expect("implemented catalog entry has a regression fixture")
                .encoding
                .bits();
            let memory = memory(&[(0, encoding), (4, breakpoint(0))]);
            let location = location_for(cpu, CODE);
            let decoded = match decode::decode(cpu.decoder(), location, encoding.into()) {
                DecodeResult::Decoded(decoded) => decoded,
                result => panic!("{platform:?} rejected {encoding:#010x}: {result:?}"),
            };
            covered.insert(decoded.instruction.coverage_id().get());
            JitProcess::new(cpu)
                .unwrap()
                .entry_for(&memory, location)
                .unwrap_or_else(|error| {
                    panic!("{platform:?} failed direct compilation of {encoding:#010x}: {error}")
                });
        }
        assert_eq!(
            covered,
            catalog
                .into_iter()
                .map(|pattern| pattern.coverage_id.get())
                .collect()
        );
    }
}

#[test]
fn every_platform_fp_simd_catalog_entry_has_a_direct_or_exact_typed_lowering() {
    for platform in [TargetPlatform::Switch1, TargetPlatform::Switch2] {
        let cpu = ProcessCpuContext::for_platform(platform, SPACE);
        let catalog: Vec<_> = nixe_cpu::decode::a64::patterns()
            .iter()
            .filter(|pattern| {
                pattern.decoder == DecodeSupport::Ready
                    && matches!(
                        pattern.coverage_id.get(),
                        0x30..=0x43 | 0x48..=0x5d | 0x60..=0xa0
                    )
            })
            .collect();
        let mut covered = BTreeSet::new();
        for pattern in catalog.iter().copied() {
            let encoding = pattern
                .regression_fixture
                .expect("implemented FP/SIMD entry has a regression fixture")
                .encoding
                .bits();
            let memory = memory(&[(0, encoding), (4, breakpoint(0))]);
            let location = location_for(cpu, CODE);
            let decoded = match decode::decode(cpu.decoder(), location, encoding.into()) {
                DecodeResult::Decoded(decoded) => decoded,
                result => panic!("{platform:?} rejected {encoding:#010x}: {result:?}"),
            };
            covered.insert(decoded.instruction.coverage_id().get());
            JitProcess::new(cpu)
                .unwrap()
                .entry_for(&memory, location)
                .unwrap_or_else(|error| {
                    panic!("{platform:?} failed direct FP/SIMD compilation of {encoding:#010x}: {error}")
                });
        }
        assert_eq!(
            covered,
            catalog
                .into_iter()
                .map(|pattern| pattern.coverage_id.get())
                .collect()
        );
    }
}

#[test]
fn every_recognized_unsupported_catalog_entry_exits_with_its_exact_identity() {
    for platform in [TargetPlatform::Switch1, TargetPlatform::Switch2] {
        let cpu = ProcessCpuContext::for_platform(platform, SPACE);
        for pattern in nixe_cpu::decode::a64::patterns()
            .iter()
            .filter(|pattern| pattern.decoder == DecodeSupport::RecognizedUnimplemented)
        {
            let encoding = pattern
                .regression_fixture
                .expect("every unsupported catalog entry has a fixture")
                .encoding
                .bits();
            let memory = memory(&[(0, encoding)]);
            let process = JitProcess::new(cpu).unwrap();
            let mut state = state(CODE, 0);
            let exit = JitThread::new()
                .run(&process, &memory, &mut state, 1)
                .unwrap();
            let DirectExit::Unsupported { pc, instructions } = exit else {
                panic!("{} did not take the unsupported exit", pattern.name);
            };
            assert_eq!(pc, GuestVirtualAddress::new(CODE));
            assert_eq!(instructions, 0);
            assert!(matches!(
                unsupported_exit(&process, &memory, pc, instructions, &state).unwrap(),
                CpuExit::UnsupportedSemantics { coverage_id, .. }
                    if coverage_id == pattern.coverage_id
            ));
        }
    }
}

#[test]
fn direct_simd_integer_moves_permutations_and_shifts_match_the_interpreter() {
    let cases = [
        0x4e01_0c20_u32, // DUP V0.16B,W1
        0x4e22_1c20,     // AND V0.16B,V1.16B,V2.16B
        0x4e22_8420,     // ADD V0.16B,V1.16B,V2.16B
        0x4e22_bc20,     // ADDP V0.16B,V1.16B,V2.16B
        0x4e21_34a3,     // CMGT V3.16B,V5.16B,V1.16B
        0x4e02_1823,     // UZP1 V3.16B,V1.16B,V2.16B
        0x6e02_4023,     // EXT V3.16B,V1.16B,V2.16B,#8
        0x0f0c_8400,     // SHRN V0.8B,V0.8H,#4
        0x2f0f_0420,     // USHR V0.8B,V1.8B,#1
        0x0e22_4420,     // SSHL V0.8B,V1.8B,V2.8B
        0x4e20_5862,     // CNT V2.16B,V3.16B
        0x4e31_b862,     // ADDV B2,V3.16B
    ];
    for encoding in cases {
        let mut initial = rich_state();
        for register in 0_u8..32 {
            let byte = register.wrapping_mul(7).wrapping_add(3);
            let value = u128::from_le_bytes([byte; 16]) ^ 0x0123_4567_89ab_cdef_fedc_ba98_7654_3210;
            assert!(initial.set_vector(register, value));
        }
        assert_matches_interpreter(encoding, initial);
    }
}

#[test]
fn compact_simd_emitters_match_the_interpreter_across_shapes_and_operations() {
    let cases = [
        // Bitwise operations, including destination-as-mask forms.
        0x4e22_1c20_u32,
        0x4e62_1c20,
        0x4ea2_1c20,
        0x4ee2_1c20,
        0x6e22_1c20,
        0x6e62_1c20,
        0x6ea2_1c20,
        0x6ee2_1c20,
        // Pairwise integer operations.
        0x4e22_bc20,
        0x4e22_a420,
        0x4e22_ac20,
        0x6e22_a420,
        0x6e22_ac20,
        0x0e62_bc20,
        0x6ea2_a420,
        0x4ee2_bc20,
        // Element-wise min/max and comparisons.
        0x0ebf_6fdf,
        0x4e21_34a3,
        0x6e21_34a3,
        0x4e21_3ca3,
        0x6e21_3ca3,
        0x4e21_8ca3,
        0x6e21_8ca3,
        0x4e20_8823,
        0x6e20_8823,
        0x4e20_9823,
        0x4ee1_34a3,
        0x2e21_3ca3,
        // Permutations and byte extracts.
        0x4e02_1823,
        0x4e02_5824,
        0x4e02_2825,
        0x4e02_6826,
        0x0e02_3827,
        0x4e42_6828,
        0x4e82_5829,
        0x4ec2_782a,
        0x6e02_4023,
        0x2e0b_3949,
        0x6e1f_43ff,
        // Narrowing and immediate shifts.
        0x0f0c_8400,
        0x4f0c_8420,
        0x0e21_2820,
        0x0e61_2862,
        0x0ea1_28a4,
        0x4e21_28e6,
        0x4e61_2928,
        0x4ea1_296a,
        0x2f0f_0420,
        0x4f0f_0630,
        0x2f1f_04a4,
        0x6f10_04e6,
        0x2f3f_0528,
        0x6f20_056a,
        0x5f60_57de,
        0x0f20_a7fe,
        // Signed and unsigned variable shifts across lane widths and aliases.
        0x0e22_4420,
        0x2e24_4462,
        0x0ebd_47fd,
        0x0e62_4420,
        0x0ea2_4420,
        0x4ee2_4420,
        // Across-vector reductions.
        0x0e31_bbde,
        0x4e31_b862,
        0x0e71_b8a4,
        0x4e71_b8e6,
        0x4eb1_b928,
    ];
    for encoding in cases {
        let mut initial = rich_state();
        for register in 0_u8..32 {
            let low = u64::from(register)
                .wrapping_mul(0x0102_0304_0506_0708)
                .wrapping_add(0x807f_01ff_55aa_cc33);
            let high = low.rotate_left(u32::from(register));
            assert!(initial.set_vector(register, u128::from(low) | (u128::from(high) << 64)));
        }
        assert_matches_interpreter(encoding, initial);
    }
}

#[test]
fn direct_simd_quadword_and_pair_memory_match_the_interpreter() {
    let words = [
        (0, 0x3d80_0080_u32),  // STR Q0,[X4]
        (4, 0x3dc0_0082_u32),  // LDR Q2,[X4]
        (8, 0xad01_0480_u32),  // STP Q0,Q1,[X4,#32]
        (12, 0xad41_0c82_u32), // LDP Q2,Q3,[X4,#32]
        (16, breakpoint(0)),
    ];
    let expected_memory = memory_with_words_and_data(&words);
    let actual_memory = memory_with_words_and_data(&words);
    let mut initial = rich_state();
    write_register(&mut initial, 4, DATA);
    assert!(initial.set_vector(0, 0x1122_3344_5566_7788_99aa_bbcc_ddee_ff00));
    assert!(initial.set_vector(1, 0xffee_ddcc_bbaa_9988_7766_5544_3322_1100));

    let mut expected = initial.clone();
    let monitor = RefCell::new(nixe_cpu::exclusive::ExclusiveMonitorState::default());
    let events = nixe_cpu::execution::VcpuEventState::default();
    let context = InterpreterContext::new(cpu(), &expected_memory, &monitor, &ZeroTimer, &events);
    for (_, encoding) in words[..4].iter().copied() {
        assert_eq!(
            execute_one_with_context(context, &mut expected, encoding).unwrap(),
            InstructionStep::Continue,
            "{encoding:#010x}"
        );
    }

    let mut actual = initial;
    assert_eq!(
        JitThread::new()
            .run(
                &JitProcess::new(cpu()).unwrap(),
                &actual_memory,
                &mut actual,
                4,
            )
            .unwrap(),
        DirectExit::Budget {
            pc: GuestVirtualAddress::new(CODE + 16),
            instructions: 4,
        }
    );
    assert_eq!(actual, expected);
    assert_memory_prefix_equal(&actual_memory, &expected_memory, 64);
}

#[test]
fn direct_exact_fp_matches_normal_special_rounding_saturation_and_status_cases() {
    let mut normal = rich_state();
    assert!(normal.set_vector(1, u128::from(1.25_f64.to_bits())));
    assert!(normal.set_vector(2, u128::from(2.5_f64.to_bits())));
    assert_matches_interpreter(0x1e62_2820, normal); // FADD D0,D1,D2

    let halfway = (970_u64) << 52; // 2^-53, half an ulp at 1.0.
    for mode in 0_u32..=3 {
        let mut rounding = rich_state();
        rounding.set_fpcr(mode << 22);
        assert!(rounding.set_vector(1, u128::from(1.0_f64.to_bits())));
        assert!(rounding.set_vector(2, u128::from(halfway)));
        assert_matches_interpreter(0x1e62_2820, rounding); // FADD D0,D1,D2
    }

    let mut nan = rich_state();
    nan.set_fpsr(0x80);
    assert!(nan.set_vector(0, u128::from(0x7ff8_0000_0000_0001_u64)));
    assert!(nan.set_vector(1, u128::from(1.0_f64.to_bits())));
    assert_matches_interpreter(0x1e61_2000, nan); // FCMP D0,D1

    let mut infinity = rich_state();
    infinity.set_fpsr(0x80);
    assert!(infinity.set_vector(1, u128::from(1.0_f32.to_bits())));
    assert!(infinity.set_vector(2, 0));
    assert_matches_interpreter(0x1e22_1820, infinity); // FDIV S0,S1,S2

    let mut denormal = rich_state();
    denormal.set_fpcr(1 << 24); // FZ
    assert!(denormal.set_vector(1, 1));
    assert!(denormal.set_vector(2, u128::from(1.0_f64.to_bits())));
    assert_matches_interpreter(0x1e62_0820, denormal); // FMUL D0,D1,D2

    for value in [f64::INFINITY, f64::NAN, -0.5] {
        let mut saturation = rich_state();
        assert!(saturation.set_vector(1, u128::from(value.to_bits())));
        assert_matches_interpreter(0x9e79_0020, saturation); // FCVTZU X0,D1
    }
}

#[test]
fn direct_exact_fp_enabled_exception_is_atomic() {
    let encoding = 0x1e62_2820_u32; // FADD D0,D1,D2
    let mut initial = rich_state();
    initial.set_fpcr(1 << 8); // IOE
    initial.set_fpsr(0x80);
    assert!(initial.set_vector(0, u128::MAX));
    assert!(initial.set_vector(1, u128::from(f64::INFINITY.to_bits())));
    assert!(initial.set_vector(2, u128::from(f64::NEG_INFINITY.to_bits())));

    let mut expected = initial.clone();
    assert!(matches!(
        execute_one(&cpu().platform(), &mut expected, encoding).unwrap(),
        InstructionStep::Exit(nixe_cpu::execution::CpuExit::ArchitecturalException {
            kind: nixe_cpu::exception::ExceptionKind::FloatingPoint,
            ..
        })
    ));

    let mut actual = initial;
    assert_eq!(
        JitThread::new()
            .run(
                &JitProcess::new(cpu()).unwrap(),
                &memory(&[(0, encoding), (4, breakpoint(0))]),
                &mut actual,
                1,
            )
            .unwrap(),
        DirectExit::Architectural {
            pc: GuestVirtualAddress::new(CODE),
            detail: 6 << 24,
            instructions: 1,
        }
    );
    assert_eq!(actual, expected);
}

#[test]
fn scalar_memory_addressing_and_transfer_forms_match_the_interpreter() {
    let cases = [
        0xf940_0020, // LDR X0,[X1]
        0xf900_0023, // STR X3,[X1]
        0xb940_0020, // LDR W0,[X1]
        0xb900_0023, // STR W3,[X1]
        0x3940_0020, // LDRB W0,[X1]
        0x3900_0023, // STRB W3,[X1]
        0x3980_0020, // LDRSB X0,[X1]
        0x39c0_0020, // LDRSB W0,[X1]
        0xb980_0020, // LDRSW X0,[X1]
        0xf840_0020, // LDUR X0,[X1]
        0xf840_8420, // LDR X0,[X1],#8
        0xf840_8c20, // LDR X0,[X1,#8]!
        0xf862_6820, // LDR X0,[X1,X2]
        0xa940_0c20, // LDP X0,X3,[X1]
        0xa900_0c20, // STP X0,X3,[X1]
        0x6940_0c20, // LDPSW X0,X3,[X1]
        0x5800_0040, // LDR X0,[PC,#8]
        0xc8df_fc20, // LDAR X0,[X1]
        0xc89f_fc23, // STLR X3,[X1]
    ];
    for encoding in cases {
        assert_memory_matches_interpreter(encoding);
    }
}

#[test]
fn linux_direct_clean_scalar_reads_are_eager_and_never_enter_rust() {
    for encoding in [
        0x3940_0020, // LDRB W0,[X1]
        0x7940_0020, // LDRH W0,[X1]
        0xb940_0020, // LDR W0,[X1]
        0xf940_0020, // LDR X0,[X1]
    ] {
        let mut memory = execution_memory_with_data(encoding);
        assert_eq!(
            memory
                .bind_cpu_memory_backend(SPACE, 0x1_0000, DirectBackendPolicy::Required)
                .unwrap(),
            nixe_memory::CpuMemoryBackend::LinuxDirect
        );
        let process = JitProcess::new(cpu()).unwrap();
        let binding = MemoryBinding {
            address_space: SPACE,
            end_exclusive: GuestVirtualAddress::new(0x1_0000),
            memory: &memory,
            mapping_epoch: memory.mapping_epoch().get(),
            invalidation_cursor: memory.invalidation_cursor(),
        };
        process.bind_memory(binding).unwrap();
        let mut thread = JitThread::new();
        thread.synchronize_address_space(&process, binding).unwrap();
        let mut state = memory_state();
        thread.run(&process, &memory, &mut state, 1).unwrap();
        assert_eq!(
            process.slow_memory_calls.load(Ordering::Relaxed),
            0,
            "{encoding:#010x} did not use its eager direct alias"
        );

        state.set_pc(CODE);
        write_register(&mut state, 0, 0);
        thread.run(&process, &memory, &mut state, 1).unwrap();
        assert_eq!(
            process.slow_memory_calls.load(Ordering::Relaxed),
            0,
            "{encoding:#010x} returned to Rust on a repeated direct read"
        );
    }
}

#[test]
fn linux_direct_jit_rejects_a_replacement_arena_before_native_entry() {
    let mut first = execution_memory_with_data(0xf940_0020);
    first
        .bind_cpu_memory_backend(SPACE, 0x1_0000, DirectBackendPolicy::Required)
        .unwrap();
    let mut replacement = execution_memory_with_data(0xf940_0020);
    replacement
        .bind_cpu_memory_backend(SPACE, 0x1_0000, DirectBackendPolicy::Required)
        .unwrap();
    let process = JitProcess::new(cpu()).unwrap();
    let binding = MemoryBinding {
        address_space: SPACE,
        end_exclusive: GuestVirtualAddress::new(0x1_0000),
        memory: &first,
        mapping_epoch: first.mapping_epoch().get(),
        invalidation_cursor: first.invalidation_cursor(),
    };
    process.bind_memory(binding).unwrap();
    let mut thread = JitThread::new();
    thread.synchronize_address_space(&process, binding).unwrap();
    drop(first);
    let mut state = memory_state();

    let error = thread
        .run(&process, &replacement, &mut state, 1)
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("differs from its immutable process binding")
    );
}

#[test]
fn linux_direct_jit_requires_a_live_mapping_lease() {
    let mut memory = execution_memory_with_data(0xf940_0020);
    memory
        .bind_cpu_memory_backend(SPACE, 0x1_0000, DirectBackendPolicy::Required)
        .unwrap();
    let process = JitProcess::new(cpu()).unwrap();
    let binding = MemoryBinding {
        address_space: SPACE,
        end_exclusive: GuestVirtualAddress::new(0x1_0000),
        memory: &memory,
        mapping_epoch: memory.mapping_epoch().get(),
        invalidation_cursor: memory.invalidation_cursor(),
    };
    process.bind_memory(binding).unwrap();
    let mut thread = JitThread::new();
    thread.synchronize_address_space(&process, binding).unwrap();
    let mut state = memory_state();

    let error = thread
        .run_slice(
            &process,
            RunRequest {
                cpu: cpu(),
                memory: &memory,
                memory_lease: None,
                state: &mut state,
                instruction_budget: 1,
                loader_return: None,
                timer: &ZeroTimer,
                events: VcpuEventState::default(),
            },
        )
        .unwrap_err();
    assert!(error.message.contains("requires its live mapping lease"));
}

#[test]
fn linux_direct_jit_compiles_outside_the_native_execution_lease() {
    let mut memory = execution_memory_with_data(0xd53b_e020); // MRS X0,CNTVCT_EL0
    memory
        .bind_cpu_memory_backend(SPACE, 0x1_0000, DirectBackendPolicy::Required)
        .unwrap();
    let memory = Arc::new(memory);
    let process = Arc::new(JitProcess::new(cpu()).unwrap());
    process
        .bind_memory(MemoryBinding {
            address_space: SPACE,
            end_exclusive: GuestVirtualAddress::new(0x1_0000),
            memory: memory.as_ref(),
            mapping_epoch: memory.mapping_epoch().get(),
            invalidation_cursor: memory.invalidation_cursor(),
        })
        .unwrap();

    let timer = Arc::new(BlockingTimer {
        entered: Arc::new(Barrier::new(2)),
        release: Arc::new(Barrier::new(2)),
    });
    let (ready_tx, ready_rx) = mpsc::channel();
    let (run_tx, run_rx) = mpsc::channel();
    let worker_memory = Arc::clone(&memory);
    let worker_process = Arc::clone(&process);
    let worker_timer = Arc::clone(&timer);
    let worker = std::thread::spawn(move || {
        let binding = MemoryBinding {
            address_space: SPACE,
            end_exclusive: GuestVirtualAddress::new(0x1_0000),
            memory: worker_memory.as_ref(),
            mapping_epoch: worker_memory.mapping_epoch().get(),
            invalidation_cursor: worker_memory.invalidation_cursor(),
        };
        let mut thread = JitThread::new();
        thread
            .synchronize_address_space(worker_process.as_ref(), binding)
            .unwrap();
        let lease = worker_memory.acquire_execution_lease();
        ready_tx.send(()).unwrap();
        run_rx.recv().unwrap();
        let mut state = state(CODE, 0);
        thread.run_slice(
            worker_process.as_ref(),
            RunRequest {
                cpu: cpu(),
                memory: worker_memory.as_ref(),
                memory_lease: Some(lease),
                state: &mut state,
                instruction_budget: 1,
                loader_return: None,
                timer: worker_timer.as_ref(),
                events: VcpuEventState::default(),
            },
        )
    });
    ready_rx.recv_timeout(Duration::from_secs(1)).unwrap();

    let (transition_tx, transition_rx) = mpsc::channel();
    let transition_memory = Arc::clone(&memory);
    let transition = std::thread::spawn(move || {
        let result = transition_memory.set_attributes(
            SPACE,
            GuestVirtualAddress::new(DATA),
            0x1000,
            MemoryAttributes::UNCACHED,
            MemoryAttributes::UNCACHED,
        );
        transition_tx.send(result).unwrap();
    });
    let wait_started = Instant::now();
    while !memory.mapping_mutation_pending() {
        assert!(
            wait_started.elapsed() < Duration::from_secs(1),
            "mapping transition did not close execution admission"
        );
        std::thread::yield_now();
    }
    run_tx.send(()).unwrap();

    timer.entered.wait();
    let transition_result = transition_rx.recv_timeout(Duration::from_secs(1));
    timer.release.wait();
    let report = worker.join().unwrap().unwrap();
    transition.join().unwrap();

    transition_result
        .expect("mapping transition must complete before native entry")
        .unwrap();
    assert_eq!(report.instructions_executed, 1);
    assert_eq!(report.stop, CpuExit::BudgetExhausted);
}

#[test]
fn linux_direct_gpu_newer_read_reconciles_and_retries_the_native_load_once() {
    let mut memory = execution_memory_with_data(0xf940_0020); // LDR X0,[X1]
    memory
        .bind_cpu_memory_backend(SPACE, 0x1_0000, DirectBackendPolicy::Required)
        .unwrap();
    let retained = memory
        .translate_canonical_range(
            SPACE,
            GuestVirtualAddress::new(DATA),
            0x1000,
            MemoryPermissions::READ_WRITE,
        )
        .unwrap();
    let mut bytes = vec![0_u8; 0x1000];
    bytes[..8].copy_from_slice(&0xa5c3_9678_1234_fedc_u64.to_le_bytes());
    let coordinator: Arc<dyn VisibilityCoordinator> = Arc::new(DeviceWriteback {
        bytes: bytes.into_boxed_slice(),
    });
    let declaration = DeviceAccessDeclaration::write(
        NonCpuDeviceId::new(19),
        DeviceVisibilityPoint::new(1),
        DeviceVisibilityPoint::new(2),
    )
    .unwrap();
    retained
        .prepare_device_access(declaration, Arc::clone(&coordinator))
        .unwrap();
    retained
        .publish_device_write(declaration, coordinator)
        .unwrap();

    let process = JitProcess::new(cpu()).unwrap();
    let binding = MemoryBinding {
        address_space: SPACE,
        end_exclusive: GuestVirtualAddress::new(0x1_0000),
        memory: &memory,
        mapping_epoch: memory.mapping_epoch().get(),
        invalidation_cursor: memory.invalidation_cursor(),
    };
    process.bind_memory(binding).unwrap();
    let mut thread = JitThread::new();
    thread.synchronize_address_space(&process, binding).unwrap();
    let mut state = memory_state();

    assert!(matches!(
        thread.run(&process, &memory, &mut state, 1).unwrap(),
        DirectExit::Budget { .. }
    ));
    assert_eq!(read_register(&state, 0), 0xa5c3_9678_1234_fedc);
    assert_eq!(process.slow_memory_calls.load(Ordering::Relaxed), 0);
}

#[test]
fn linux_direct_scalar_stores_fault_once_then_publish_natively() {
    for (encoding, size, first, second) in [
        (0x3900_0020_u32, MemoryAccessSize::Byte, 0xa5_u64, 0x5a_u64),
        (0x7900_0020, MemoryAccessSize::Halfword, 0xa5c3, 0x5a3c),
        (
            0xb900_0020,
            MemoryAccessSize::Word,
            0xa5c3_9678,
            0x5a3c_6987,
        ),
        (
            0xf900_0020,
            MemoryAccessSize::Doubleword,
            0xa5c3_9678_1234_fedc,
            0x5a3c_6987_edcb_0123,
        ),
    ] {
        let mut memory = ExecutionMemory::new();
        let code_page = GuestPhysicalPageId::new(1);
        let data_page = GuestPhysicalPageId::new(2);
        assert!(memory.add_ram_page(code_page));
        assert!(memory.add_ram_page(data_page));
        assert!(memory.initialize_ram(code_page, 0, &encoding.to_le_bytes()));
        assert!(memory.initialize_ram(code_page, 4, &breakpoint(0).to_le_bytes()));
        assert!(memory.map_page(
            SPACE,
            GuestVirtualAddress::new(CODE),
            code_page,
            MemoryPermissions::READ_EXECUTE,
        ));
        assert!(memory.map_page(
            SPACE,
            GuestVirtualAddress::new(DATA),
            data_page,
            MemoryPermissions::READ_WRITE,
        ));
        memory
            .bind_cpu_memory_backend(SPACE, 0x1_0000, DirectBackendPolicy::Required)
            .unwrap();
        let process = JitProcess::new(cpu()).unwrap();
        let binding = MemoryBinding {
            address_space: SPACE,
            end_exclusive: GuestVirtualAddress::new(0x1_0000),
            memory: &memory,
            mapping_epoch: memory.mapping_epoch().get(),
            invalidation_cursor: memory.invalidation_cursor(),
        };
        process.bind_memory(binding).unwrap();
        let mut thread = JitThread::new();
        thread.synchronize_address_space(&process, binding).unwrap();
        let mut state = memory_state();
        write_register(&mut state, 0, first);
        write_register(&mut state, 1, DATA);
        assert!(matches!(
            thread.run(&process, &memory, &mut state, 1).unwrap(),
            DirectExit::Budget { .. }
        ));
        assert_eq!(process.slow_memory_calls.load(Ordering::Relaxed), 1);
        assert_eq!(
            memory
                .read(
                    SPACE,
                    GuestVirtualAddress::new(DATA),
                    MemoryAccess::normal(size),
                )
                .unwrap()
                .value,
            MemoryValue::from_bits(size, u128::from(first)),
        );

        state.set_pc(CODE);
        write_register(&mut state, 0, second);
        assert!(matches!(
            thread.run(&process, &memory, &mut state, 1).unwrap(),
            DirectExit::Budget { .. }
        ));
        assert_eq!(
            process.slow_memory_calls.load(Ordering::Relaxed),
            1,
            "{encoding:#010x} returned to Rust after direct-store publication",
        );
        assert_eq!(
            memory
                .read(
                    SPACE,
                    GuestVirtualAddress::new(DATA),
                    MemoryAccess::normal(size),
                )
                .unwrap()
                .value,
            MemoryValue::from_bits(size, u128::from(second)),
        );
    }
}

#[test]
fn linux_direct_scalar_pair_loads_remain_native() {
    let mut memory = execution_memory_with_data(0xa940_0c20); // LDP X0,X3,[X1]
    memory
        .bind_cpu_memory_backend(SPACE, 0x1_0000, DirectBackendPolicy::Required)
        .unwrap();
    let process = JitProcess::new(cpu()).unwrap();
    let binding = MemoryBinding {
        address_space: SPACE,
        end_exclusive: GuestVirtualAddress::new(0x1_0000),
        memory: &memory,
        mapping_epoch: memory.mapping_epoch().get(),
        invalidation_cursor: memory.invalidation_cursor(),
    };
    process.bind_memory(binding).unwrap();
    let mut thread = JitThread::new();
    thread.synchronize_address_space(&process, binding).unwrap();
    let mut state = memory_state();

    assert!(matches!(
        thread.run(&process, &memory, &mut state, 1).unwrap(),
        DirectExit::Budget { .. }
    ));
    assert_eq!(read_register(&state, 0), 0x0706_0504_0302_0100);
    assert_eq!(read_register(&state, 3), 0x0f0e_0d0c_0b0a_0908);
    assert_eq!(process.slow_memory_calls.load(Ordering::Relaxed), 0);
}

#[test]
fn linux_direct_simd_quadword_loads_and_stores_use_the_shared_arena() {
    let mut memory = ExecutionMemory::new();
    let code_page = GuestPhysicalPageId::new(1);
    let data_page = GuestPhysicalPageId::new(2);
    assert!(memory.add_ram_page(code_page));
    assert!(memory.add_ram_page(data_page));
    assert!(memory.initialize_ram(code_page, 0, &0x3d80_0020_u32.to_le_bytes()));
    assert!(memory.initialize_ram(code_page, 4, &0x3dc0_0022_u32.to_le_bytes()));
    assert!(memory.initialize_ram(code_page, 8, &breakpoint(0).to_le_bytes()));
    assert!(memory.map_page(
        SPACE,
        GuestVirtualAddress::new(CODE),
        code_page,
        MemoryPermissions::READ_EXECUTE,
    ));
    assert!(memory.map_page(
        SPACE,
        GuestVirtualAddress::new(DATA),
        data_page,
        MemoryPermissions::READ_WRITE,
    ));
    memory
        .bind_cpu_memory_backend(SPACE, 0x1_0000, DirectBackendPolicy::Required)
        .unwrap();
    let process = JitProcess::new(cpu()).unwrap();
    let binding = MemoryBinding {
        address_space: SPACE,
        end_exclusive: GuestVirtualAddress::new(0x1_0000),
        memory: &memory,
        mapping_epoch: memory.mapping_epoch().get(),
        invalidation_cursor: memory.invalidation_cursor(),
    };
    process.bind_memory(binding).unwrap();
    let mut thread = JitThread::new();
    thread.synchronize_address_space(&process, binding).unwrap();
    let mut state = memory_state();
    let first = 0x1122_3344_5566_7788_99aa_bbcc_ddee_ff00;
    assert!(state.set_vector(0, first));

    assert!(matches!(
        thread.run(&process, &memory, &mut state, 2).unwrap(),
        DirectExit::Budget { .. }
    ));
    assert_eq!(state.vector(2), Some(first));
    assert_eq!(process.slow_memory_calls.load(Ordering::Relaxed), 1);

    let second = 0xffee_ddcc_bbaa_9988_7766_5544_3322_1100;
    state.set_pc(CODE);
    assert!(state.set_vector(0, second));
    assert!(matches!(
        thread.run(&process, &memory, &mut state, 1).unwrap(),
        DirectExit::Budget { .. }
    ));
    assert_eq!(
        process.slow_memory_calls.load(Ordering::Relaxed),
        1,
        "an armed 16-byte SIMD store returned to Rust",
    );
    assert_eq!(
        memory
            .read(
                SPACE,
                GuestVirtualAddress::new(DATA),
                MemoryAccess::normal(MemoryAccessSize::Quadword),
            )
            .unwrap()
            .value,
        MemoryValue::U128(second),
    );
}

#[test]
fn linux_direct_unaligned_scalar_accesses_complete_through_checked_memory() {
    let mut memory = ExecutionMemory::new();
    let code_page = GuestPhysicalPageId::new(1);
    let data_page = GuestPhysicalPageId::new(2);
    assert!(memory.add_ram_page(code_page));
    assert!(memory.add_ram_page(data_page));
    assert!(memory.initialize_ram(code_page, 0, &0xf900_0020_u32.to_le_bytes()));
    assert!(memory.initialize_ram(code_page, 4, &0xf940_0022_u32.to_le_bytes()));
    assert!(memory.initialize_ram(code_page, 8, &breakpoint(0).to_le_bytes()));
    assert!(memory.map_page(
        SPACE,
        GuestVirtualAddress::new(CODE),
        code_page,
        MemoryPermissions::READ_EXECUTE,
    ));
    assert!(memory.map_page(
        SPACE,
        GuestVirtualAddress::new(DATA),
        data_page,
        MemoryPermissions::READ_WRITE,
    ));
    memory
        .bind_cpu_memory_backend(SPACE, 0x1_0000, DirectBackendPolicy::Required)
        .unwrap();
    let process = JitProcess::new(cpu()).unwrap();
    let binding = MemoryBinding {
        address_space: SPACE,
        end_exclusive: GuestVirtualAddress::new(0x1_0000),
        memory: &memory,
        mapping_epoch: memory.mapping_epoch().get(),
        invalidation_cursor: memory.invalidation_cursor(),
    };
    process.bind_memory(binding).unwrap();
    let mut thread = JitThread::new();
    thread.synchronize_address_space(&process, binding).unwrap();
    let mut state = memory_state();
    let address = GuestVirtualAddress::new(DATA + 4);
    let value = 0xa5c3_9678_1234_fedc;
    write_register(&mut state, 0, value);
    write_register(&mut state, 1, address.get());

    assert!(matches!(
        thread.run(&process, &memory, &mut state, 2).unwrap(),
        DirectExit::Budget { .. }
    ));
    assert_eq!(read_register(&state, 2), value);
    assert_eq!(process.slow_memory_calls.load(Ordering::Relaxed), 2);
    let access = MemoryAccess::new(
        MemoryAccessSize::Doubleword,
        MemoryAlignment::Unaligned,
        MemoryOrdering::Relaxed,
        MemoryAccessClass::Normal,
    );
    assert_eq!(
        memory.read(SPACE, address, access).unwrap().value,
        MemoryValue::U64(value),
    );
}

#[test]
fn unmapped_memory_fault_matches_the_interpreter_at_the_faulting_pc() {
    let encoding = 0xf940_0020; // LDR X0,[X1]
    let expected_memory = memory(&[(0, encoding), (4, breakpoint(0))]);
    let actual_memory = memory(&[(0, encoding), (4, breakpoint(0))]);
    let mut initial = memory_state();
    write_register(&mut initial, 1, DATA);
    let mut expected = initial.clone();
    let monitor = RefCell::new(nixe_cpu::exclusive::ExclusiveMonitorState::default());
    let events = nixe_cpu::execution::VcpuEventState::default();
    let context = InterpreterContext::new(cpu(), &expected_memory, &monitor, &ZeroTimer, &events);
    let InstructionStep::Exit(nixe_cpu::execution::CpuExit::DataFault {
        source,
        fault: expected_fault,
    }) = execute_one_with_context(context, &mut expected, encoding).unwrap()
    else {
        panic!("reference interpreter did not report a data fault")
    };

    let mut actual = initial;
    let DirectExit::DataFault {
        pc,
        fault,
        instructions,
    } = JitThread::new()
        .run(
            &JitProcess::new(cpu()).unwrap(),
            &actual_memory,
            &mut actual,
            1,
        )
        .unwrap()
    else {
        panic!("direct JIT did not report a data fault")
    };
    assert_eq!(pc, source.pc);
    assert_eq!(fault, expected_fault);
    assert_eq!(instructions, 1);
    assert_eq!(actual.pc(), CODE);
    assert_eq!(actual, expected);
}

#[test]
fn linux_direct_unmapped_read_reconstructs_exact_prefault_state() {
    let mut memory = ExecutionMemory::new();
    let code_page = GuestPhysicalPageId::new(1);
    assert!(memory.add_ram_page(code_page));
    assert!(memory.initialize_ram(code_page, 0, &0xf940_0020_u32.to_le_bytes())); // LDR X0,[X1]
    assert!(memory.initialize_ram(code_page, 4, &breakpoint(0).to_le_bytes()));
    assert!(memory.map_page(
        SPACE,
        GuestVirtualAddress::new(CODE),
        code_page,
        MemoryPermissions::READ_EXECUTE,
    ));
    assert_eq!(
        memory
            .bind_cpu_memory_backend(SPACE, 0x1_0000, DirectBackendPolicy::Required)
            .unwrap(),
        nixe_memory::CpuMemoryBackend::LinuxDirect
    );
    let process = JitProcess::new(cpu()).unwrap();
    let binding = MemoryBinding {
        address_space: SPACE,
        end_exclusive: GuestVirtualAddress::new(0x1_0000),
        memory: &memory,
        mapping_epoch: memory.mapping_epoch().get(),
        invalidation_cursor: memory.invalidation_cursor(),
    };
    process.bind_memory(binding).unwrap();
    let mut thread = JitThread::new();
    thread.synchronize_address_space(&process, binding).unwrap();
    let mut state = rich_state();
    state.set_pc(CODE);
    write_register(&mut state, 1, DATA);
    let before = state.clone();
    let exit = thread.run(&process, &memory, &mut state, 1).unwrap();
    assert!(matches!(
        exit,
        DirectExit::DataFault {
            pc,
            instructions: 1,
            ..
        } if pc == GuestVirtualAddress::new(CODE)
    ));
    assert_eq!(state, before);
}

#[test]
fn linux_direct_discarded_load_retains_its_architectural_fault() {
    let mut memory = ExecutionMemory::new();
    let code_page = GuestPhysicalPageId::new(1);
    assert!(memory.add_ram_page(code_page));
    assert!(memory.initialize_ram(code_page, 0, &0xf940_003f_u32.to_le_bytes())); // LDR XZR,[X1]
    assert!(memory.initialize_ram(code_page, 4, &breakpoint(0).to_le_bytes()));
    assert!(memory.map_page(
        SPACE,
        GuestVirtualAddress::new(CODE),
        code_page,
        MemoryPermissions::READ_EXECUTE,
    ));
    memory
        .bind_cpu_memory_backend(SPACE, 0x1_0000, DirectBackendPolicy::Required)
        .unwrap();
    let process = JitProcess::new(cpu()).unwrap();
    let binding = MemoryBinding {
        address_space: SPACE,
        end_exclusive: GuestVirtualAddress::new(0x1_0000),
        memory: &memory,
        mapping_epoch: memory.mapping_epoch().get(),
        invalidation_cursor: memory.invalidation_cursor(),
    };
    process.bind_memory(binding).unwrap();
    let mut thread = JitThread::new();
    thread.synchronize_address_space(&process, binding).unwrap();
    let mut state = rich_state();
    state.set_pc(CODE);
    write_register(&mut state, 1, DATA);
    let before = state.clone();

    assert!(matches!(
        thread.run(&process, &memory, &mut state, 1).unwrap(),
        DirectExit::DataFault {
            pc,
            instructions: 1,
            ..
        } if pc == GuestVirtualAddress::new(CODE)
    ));
    assert_eq!(state, before);
}

#[test]
fn linux_direct_overwritten_load_retains_its_architectural_fault() {
    let mut memory = ExecutionMemory::new();
    let code_page = GuestPhysicalPageId::new(1);
    assert!(memory.add_ram_page(code_page));
    for (offset, encoding) in [
        (0, 0xf940_0108_u32), // LDR X8,[X8]
        (4, 0xaa02_03e8_u32), // MOV X8,X2
        (8, breakpoint(0)),
    ] {
        assert!(memory.initialize_ram(code_page, offset, &encoding.to_le_bytes()));
    }
    assert!(memory.map_page(
        SPACE,
        GuestVirtualAddress::new(CODE),
        code_page,
        MemoryPermissions::READ_EXECUTE,
    ));
    memory
        .bind_cpu_memory_backend(SPACE, 0x1_0000, DirectBackendPolicy::Required)
        .unwrap();
    let process = JitProcess::new(cpu()).unwrap();
    let binding = MemoryBinding {
        address_space: SPACE,
        end_exclusive: GuestVirtualAddress::new(0x1_0000),
        memory: &memory,
        mapping_epoch: memory.mapping_epoch().get(),
        invalidation_cursor: memory.invalidation_cursor(),
    };
    process.bind_memory(binding).unwrap();
    let mut thread = JitThread::new();
    thread.synchronize_address_space(&process, binding).unwrap();
    let mut state = rich_state();
    state.set_pc(CODE);
    write_register(&mut state, 8, DATA);
    let before = state.clone();

    assert!(matches!(
        thread.run(&process, &memory, &mut state, 2).unwrap(),
        DirectExit::DataFault {
            pc,
            instructions: 1,
            ..
        } if pc == GuestVirtualAddress::new(CODE)
    ));
    assert_eq!(state, before);
}

#[test]
fn linux_direct_poison_guard_preserves_the_unconfined_guest_address() {
    let mut memory = ExecutionMemory::new();
    let code_page = GuestPhysicalPageId::new(1);
    assert!(memory.add_ram_page(code_page));
    assert!(memory.initialize_ram(code_page, 0, &0xf940_0020_u32.to_le_bytes()));
    assert!(memory.initialize_ram(code_page, 4, &breakpoint(0).to_le_bytes()));
    assert!(memory.map_page(
        SPACE,
        GuestVirtualAddress::new(CODE),
        code_page,
        MemoryPermissions::READ_EXECUTE,
    ));
    assert_eq!(
        memory
            .bind_cpu_memory_backend(SPACE, 0x1_0000, DirectBackendPolicy::Required)
            .unwrap(),
        nixe_memory::CpuMemoryBackend::LinuxDirect
    );
    let process = JitProcess::new(cpu()).unwrap();
    let binding = MemoryBinding {
        address_space: SPACE,
        end_exclusive: GuestVirtualAddress::new(0x1_0000),
        memory: &memory,
        mapping_epoch: memory.mapping_epoch().get(),
        invalidation_cursor: memory.invalidation_cursor(),
    };
    process.bind_memory(binding).unwrap();
    let mut thread = JitThread::new();
    thread.synchronize_address_space(&process, binding).unwrap();
    let mut state = rich_state();
    write_register(&mut state, 1, u64::MAX);
    let DirectExit::DataFault { fault, .. } = thread.run(&process, &memory, &mut state, 1).unwrap()
    else {
        panic!("unconfined direct read did not produce a guest data fault");
    };
    assert_eq!(fault.address, GuestVirtualAddress::new(u64::MAX));
}

#[test]
fn linux_direct_fault_recovers_every_dirty_register_class_and_spills() {
    let mut encodings = vec![
        0xd51b_4405_u32, // MSR FPCR,X5
        0x1e62_1824_u32, // FDIV D4,D1,D2: 0/0 publishes IOC in FPSR
    ];
    encodings.extend((0_u32..31).map(|register| {
        0x9100_0400 | (register << 5) | register // ADD Xn,Xn,#1
    }));
    encodings.extend((0_u32..32).map(|register| {
        0x4e01_0c20 | register // DUP Vn.16B,W1
    }));
    encodings.extend([
        0x9100_43ff, // ADD SP,SP,#16
        0xb100_0442, // ADDS X2,X2,#1
        0xf940_03a0, // LDR X0,[X29]
    ]);
    let mut memory = ExecutionMemory::new();
    let code_page = GuestPhysicalPageId::new(1);
    assert!(memory.add_ram_page(code_page));
    for (index, encoding) in encodings.iter().copied().enumerate() {
        assert!(memory.initialize_ram(code_page, index * 4, &encoding.to_le_bytes()));
    }
    assert!(memory.initialize_ram(code_page, encodings.len() * 4, &breakpoint(0).to_le_bytes()));
    assert!(memory.map_page(
        SPACE,
        GuestVirtualAddress::new(CODE),
        code_page,
        MemoryPermissions::READ_EXECUTE,
    ));
    assert_eq!(
        memory
            .bind_cpu_memory_backend(SPACE, 0x1_0000, DirectBackendPolicy::Required)
            .unwrap(),
        nixe_memory::CpuMemoryBackend::LinuxDirect
    );

    let mut initial = rich_state();
    write_register(&mut initial, 29, DATA - 1);
    write_register(&mut initial, 5, 2 << 22);
    initial.set_fpcr(1 << 22);
    initial.set_fpsr(0x80);
    let mut expected = initial.clone();
    for encoding in &encodings[..encodings.len() - 1] {
        assert_eq!(
            execute_one(&cpu().platform(), &mut expected, *encoding).unwrap(),
            InstructionStep::Continue
        );
    }
    let monitor = RefCell::new(nixe_cpu::exclusive::ExclusiveMonitorState::default());
    let events = nixe_cpu::execution::VcpuEventState::default();
    let context = InterpreterContext::new(cpu(), &memory, &monitor, &ZeroTimer, &events);
    let InstructionStep::Exit(nixe_cpu::execution::CpuExit::DataFault {
        source,
        fault: expected_fault,
    }) = execute_one_with_context(context, &mut expected, encodings[encodings.len() - 1]).unwrap()
    else {
        panic!("reference interpreter did not fault");
    };

    let process = JitProcess::new(cpu()).unwrap();
    let binding = MemoryBinding {
        address_space: SPACE,
        end_exclusive: GuestVirtualAddress::new(0x1_0000),
        memory: &memory,
        mapping_epoch: memory.mapping_epoch().get(),
        invalidation_cursor: memory.invalidation_cursor(),
    };
    process.bind_memory(binding).unwrap();
    let mut thread = JitThread::new();
    thread.synchronize_address_space(&process, binding).unwrap();
    let mut actual = initial;
    let DirectExit::DataFault {
        pc,
        fault,
        instructions,
    } = thread.run(&process, &memory, &mut actual, 128).unwrap()
    else {
        panic!("direct JIT did not fault");
    };
    assert_eq!(pc, source.pc);
    assert_eq!(fault, expected_fault);
    assert_eq!(instructions, encodings.len() as u64);
    assert_eq!(actual, expected);
}

#[test]
fn pair_second_access_fault_preserves_load_destinations_and_first_store() {
    for encoding in [0x2940_0c20_u32, 0x2900_0c20] {
        let expected_memory = memory_with_data(encoding);
        let actual_memory = memory_with_data(encoding);
        let mut initial = memory_state();
        write_register(&mut initial, 1, DATA + 4092);
        let mut expected = initial.clone();
        let monitor = RefCell::new(nixe_cpu::exclusive::ExclusiveMonitorState::default());
        let events = nixe_cpu::execution::VcpuEventState::default();
        let context =
            InterpreterContext::new(cpu(), &expected_memory, &monitor, &ZeroTimer, &events);
        assert!(matches!(
            execute_one_with_context(context, &mut expected, encoding).unwrap(),
            InstructionStep::Exit(nixe_cpu::execution::CpuExit::DataFault { .. })
        ));
        let mut actual = initial;
        assert!(matches!(
            JitThread::new()
                .run(
                    &JitProcess::new(cpu()).unwrap(),
                    &actual_memory,
                    &mut actual,
                    1,
                )
                .unwrap(),
            DirectExit::DataFault { .. }
        ));
        assert_eq!(actual, expected, "{encoding:#010x}");
        assert_memory_prefix_equal(&actual_memory, &expected_memory, 4096);
    }
}

#[test]
fn linux_direct_pair_second_access_fault_matches_checked_partial_semantics() {
    for encoding in [0x2940_0c20_u32, 0x2900_0c20] {
        let expected_memory = memory_with_data(encoding);
        let mut actual_memory = execution_memory_with_data(encoding);
        actual_memory
            .bind_cpu_memory_backend(SPACE, 0x1_0000, DirectBackendPolicy::Required)
            .unwrap();
        let mut initial = memory_state();
        write_register(&mut initial, 1, DATA + 4092);

        let mut expected = initial.clone();
        let monitor = RefCell::new(nixe_cpu::exclusive::ExclusiveMonitorState::default());
        let events = nixe_cpu::execution::VcpuEventState::default();
        let context =
            InterpreterContext::new(cpu(), &expected_memory, &monitor, &ZeroTimer, &events);
        assert!(matches!(
            execute_one_with_context(context, &mut expected, encoding).unwrap(),
            InstructionStep::Exit(nixe_cpu::execution::CpuExit::DataFault { .. })
        ));

        let process = JitProcess::new(cpu()).unwrap();
        let binding = MemoryBinding {
            address_space: SPACE,
            end_exclusive: GuestVirtualAddress::new(0x1_0000),
            memory: &actual_memory,
            mapping_epoch: actual_memory.mapping_epoch().get(),
            invalidation_cursor: actual_memory.invalidation_cursor(),
        };
        process.bind_memory(binding).unwrap();
        let mut thread = JitThread::new();
        thread.synchronize_address_space(&process, binding).unwrap();
        let mut actual = initial;
        assert!(matches!(
            thread
                .run(&process, &actual_memory, &mut actual, 1)
                .unwrap(),
            DirectExit::DataFault { .. }
        ));
        assert_eq!(actual, expected, "{encoding:#010x}");

        let access = MemoryAccess::normal(MemoryAccessSize::Word);
        assert_eq!(
            actual_memory
                .read(SPACE, GuestVirtualAddress::new(DATA + 4092), access,)
                .unwrap()
                .value,
            expected_memory
                .read(SPACE, GuestVirtualAddress::new(DATA + 4092), access,)
                .unwrap()
                .value,
            "{encoding:#010x}",
        );
    }
}

#[test]
fn linux_direct_simd_pair_fault_preserves_both_load_destinations() {
    let encoding = 0xad40_0c22_u32; // LDP Q2,Q3,[X1]
    let expected_memory = memory_with_data(encoding);
    let mut actual_memory = execution_memory_with_data(encoding);
    actual_memory
        .bind_cpu_memory_backend(SPACE, 0x1_0000, DirectBackendPolicy::Required)
        .unwrap();
    let mut initial = memory_state();
    write_register(&mut initial, 1, DATA + 4080);
    assert!(initial.set_vector(2, 0x1122_3344_5566_7788_99aa_bbcc_ddee_ff00));
    assert!(initial.set_vector(3, 0xffee_ddcc_bbaa_9988_7766_5544_3322_1100));

    let mut expected = initial.clone();
    let monitor = RefCell::new(nixe_cpu::exclusive::ExclusiveMonitorState::default());
    let events = nixe_cpu::execution::VcpuEventState::default();
    let context = InterpreterContext::new(cpu(), &expected_memory, &monitor, &ZeroTimer, &events);
    assert!(matches!(
        execute_one_with_context(context, &mut expected, encoding).unwrap(),
        InstructionStep::Exit(nixe_cpu::execution::CpuExit::DataFault { .. })
    ));

    let process = JitProcess::new(cpu()).unwrap();
    let binding = MemoryBinding {
        address_space: SPACE,
        end_exclusive: GuestVirtualAddress::new(0x1_0000),
        memory: &actual_memory,
        mapping_epoch: actual_memory.mapping_epoch().get(),
        invalidation_cursor: actual_memory.invalidation_cursor(),
    };
    process.bind_memory(binding).unwrap();
    let mut thread = JitThread::new();
    thread.synchronize_address_space(&process, binding).unwrap();
    let mut actual = initial;
    assert!(matches!(
        thread
            .run(&process, &actual_memory, &mut actual, 1)
            .unwrap(),
        DirectExit::DataFault { .. }
    ));
    assert_eq!(actual, expected);
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum MmioEvent {
    Read(u64, MemoryAccess),
    Write(u64, MemoryAccess, MemoryValue),
}

struct RecordingMmio {
    events: Arc<Mutex<Vec<MmioEvent>>>,
}

impl SyntheticMmio for RecordingMmio {
    fn read(&mut self, offset: u64, access: MemoryAccess) -> Result<MemoryValue, Box<str>> {
        self.events
            .lock()
            .unwrap()
            .push(MmioEvent::Read(offset, access));
        Ok(MemoryValue::U64(0x0123_4567_89ab_cdef))
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
            .push(MmioEvent::Write(offset, access, value));
        Ok(())
    }
}

#[test]
fn special_page_accesses_use_precise_typed_slow_calls() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let mut memory = memory(&[
        (0, 0xf940_0020), // LDR X0,[X1]
        (4, 0xf900_0023), // STR X3,[X1]
        (8, breakpoint(0)),
    ]);
    let page = GuestPhysicalPageId::new(2);
    assert!(memory.add_mmio_page(
        page,
        RecordingMmio {
            events: Arc::clone(&events),
        },
    ));
    assert!(memory.map_page(
        SPACE,
        GuestVirtualAddress::new(DATA),
        page,
        MemoryPermissions::READ_WRITE,
    ));
    let mut state = memory_state();
    let exit = JitThread::new()
        .run(&JitProcess::new(cpu()).unwrap(), &memory, &mut state, 2)
        .unwrap();
    assert!(matches!(
        exit,
        DirectExit::Budget {
            instructions: 2,
            ..
        }
    ));
    assert_eq!(read_x0(&state), 0x0123_4567_89ab_cdef);
    let events = events.lock().unwrap();
    assert_eq!(events.len(), 2);
    assert!(matches!(events[0], MmioEvent::Read(0, _)));
    assert!(matches!(
        events[1],
        MmioEvent::Write(0, _, MemoryValue::U64(0x8877_6655_4433_2211))
    ));
}

#[test]
fn linux_direct_dynamic_mmio_faults_complete_each_operation_exactly_once() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let mut memory = ExecutionMemory::new();
    let code_page = GuestPhysicalPageId::new(1);
    assert!(memory.add_ram_page(code_page));
    assert!(memory.initialize_ram(code_page, 0, &0xf940_0020_u32.to_le_bytes()));
    assert!(memory.initialize_ram(code_page, 4, &0xf900_0023_u32.to_le_bytes()));
    assert!(memory.initialize_ram(code_page, 8, &breakpoint(0).to_le_bytes()));
    assert!(memory.map_page(
        SPACE,
        GuestVirtualAddress::new(CODE),
        code_page,
        MemoryPermissions::READ_EXECUTE,
    ));
    let mmio_page = GuestPhysicalPageId::new(2);
    assert!(memory.add_mmio_page(
        mmio_page,
        RecordingMmio {
            events: Arc::clone(&events),
        },
    ));
    assert!(memory.map_page(
        SPACE,
        GuestVirtualAddress::new(DATA),
        mmio_page,
        MemoryPermissions::READ_WRITE,
    ));
    memory
        .bind_cpu_memory_backend(SPACE, 0x1_0000, DirectBackendPolicy::Required)
        .unwrap();
    let process = JitProcess::new(cpu()).unwrap();
    let binding = MemoryBinding {
        address_space: SPACE,
        end_exclusive: GuestVirtualAddress::new(0x1_0000),
        memory: &memory,
        mapping_epoch: memory.mapping_epoch().get(),
        invalidation_cursor: memory.invalidation_cursor(),
    };
    process.bind_memory(binding).unwrap();
    let mut thread = JitThread::new();
    thread.synchronize_address_space(&process, binding).unwrap();
    let mut state = memory_state();

    let exit = thread.run(&process, &memory, &mut state, 2).unwrap();
    assert!(matches!(
        exit,
        DirectExit::Budget {
            instructions: 2,
            ..
        }
    ));
    assert_eq!(read_x0(&state), 0x0123_4567_89ab_cdef);
    assert_eq!(process.slow_memory_calls.load(Ordering::Relaxed), 2);
    let events = events.lock().unwrap();
    assert_eq!(events.len(), 2);
    assert!(matches!(events[0], MmioEvent::Read(0, _)));
    assert!(matches!(events[1], MmioEvent::Write(0, _, _)));
}

#[test]
fn exclusive_load_store_round_trip_matches_the_interpreter() {
    let words = [
        (0, 0xc85f_7c20), // LDXR X0,[X1]
        (4, 0xc800_7c23), // STXR W0,X3,[X1]
        (8, breakpoint(0)),
    ];
    let expected_memory = memory_with_words_and_data(&words);
    let actual_memory = memory_with_words_and_data(&words);
    let mut expected = memory_state();
    let monitor = RefCell::new(nixe_cpu::exclusive::ExclusiveMonitorState::default());
    let events = nixe_cpu::execution::VcpuEventState::default();
    let context = InterpreterContext::new(cpu(), &expected_memory, &monitor, &ZeroTimer, &events);
    for encoding in [0xc85f_7c20_u32, 0xc800_7c23] {
        assert_eq!(
            execute_one_with_context(context, &mut expected, encoding).unwrap(),
            InstructionStep::Continue
        );
    }
    let mut actual = memory_state();
    JitThread::new()
        .run(
            &JitProcess::new(cpu()).unwrap(),
            &actual_memory,
            &mut actual,
            2,
        )
        .unwrap();
    assert_eq!(actual, expected);
    assert_memory_prefix_equal(&actual_memory, &expected_memory, 16);
}

#[test]
fn exclusive_atomic_updates_are_indivisible_across_vcpus() {
    const VCPUS: usize = 4;
    const ITERATIONS: usize = 100;
    let cpu = cpu();
    let memory = Arc::new(memory_with_words_and_data(&[
        (0x00, 0x885f_7c40),                                // LDXR W0,[X2]
        (0x04, 0x1100_0400),                                // ADD W0,W0,#1
        (0x08, 0x8801_7c40),                                // STXR W1,W0,[X2]
        (0x0c, compare_branch(CODE + 0x0c, CODE, 1, true)), // CBNZ W1,loop
        (0x10, breakpoint(0)),
    ]));
    let process = Arc::new(JitProcess::new(cpu).unwrap());
    let mut workers = Vec::new();
    for _ in 0..VCPUS {
        let memory = Arc::clone(&memory);
        let process = Arc::clone(&process);
        workers.push(std::thread::spawn(move || {
            let thread = JitThread::new();
            let mut state = A64State::default();
            write_register(&mut state, 2, DATA);
            for _ in 0..ITERATIONS {
                state.set_pc(CODE);
                let exit = thread
                    .run(&process, memory.as_ref(), &mut state, 1_000)
                    .unwrap();
                assert!(matches!(exit, DirectExit::Architectural { .. }));
            }
        }));
    }
    for worker in workers {
        worker.join().unwrap();
    }
    let result = memory
        .read(
            SPACE,
            GuestVirtualAddress::new(DATA),
            MemoryAccess::normal(MemoryAccessSize::Word),
        )
        .unwrap()
        .value;
    let initial = u32::from_le_bytes([0, 1, 2, 3]);
    assert_eq!(
        result,
        MemoryValue::U32(initial + (VCPUS * ITERATIONS) as u32)
    );
}

#[test]
fn executable_alias_writes_wait_for_instruction_cache_maintenance() {
    let mut memory = memory(&[(0, 0xf900_0023), (4, breakpoint(0))]); // STR X3,[X1]
    assert!(memory.map_page(
        SPACE,
        GuestVirtualAddress::new(DATA),
        GuestPhysicalPageId::new(1),
        MemoryPermissions::READ_WRITE,
    ));
    let before = memory.invalidation_cursor();
    let mut state = memory_state();
    JitThread::new()
        .run(&JitProcess::new(cpu()).unwrap(), &memory, &mut state, 1)
        .unwrap();
    let mut invalidations = Vec::new();
    let after_write = memory
        .read_invalidations_since(before, &mut invalidations)
        .unwrap();
    assert_eq!(after_write, before);
    assert!(invalidations.is_empty());

    memory
        .maintain_cache(
            SPACE,
            CacheMaintenanceKind::InstructionInvalidate,
            Some(GuestVirtualAddress::new(DATA)),
        )
        .unwrap();
    memory
        .read_invalidations_since(after_write, &mut invalidations)
        .unwrap();
    assert!(
        invalidations
            .iter()
            .any(|invalidation| match invalidation.kind {
                MemoryInvalidationKind::ExecutableContent { first, second } => {
                    first == GuestPhysicalPageId::new(1) && second.is_none()
                }
                _ => false,
            })
    );
}

struct FixedTimer;

impl nixe_cpu::execution::ArchitecturalTimer for FixedTimer {
    fn snapshot(&self) -> nixe_cpu::execution::TimerSnapshot {
        nixe_cpu::execution::TimerSnapshot {
            counter: 0x1234_5678_9abc_def0,
            frequency: 19_200_000,
        }
    }
}

#[test]
fn system_register_timer_barrier_and_cache_paths_match_the_interpreter() {
    let encodings = [
        0xd51b_4200, // MSR NZCV,X0
        0xd53b_4204, // MRS X4,NZCV
        0xd51b_d043, // MSR TPIDR_EL0,X3
        0xd53b_d045, // MRS X5,TPIDR_EL0
        0xd53b_0026, // MRS X6,CTR_EL0
        0xd53b_00e7, // MRS X7,DCZID_EL0
        0xd53b_e008, // MRS X8,CNTFRQ_EL0
        0xd53b_e029, // MRS X9,CNTVCT_EL0
        0xd503_3bbf, // DMB ISH
        0xd50b_7521, // IC IVAU,X1
    ];
    let mut words: Vec<_> = encodings
        .iter()
        .enumerate()
        .map(|(index, encoding)| ((index * 4) as u64, *encoding))
        .collect();
    words.push(((encodings.len() * 4) as u64, breakpoint(0)));
    let expected_memory = memory_with_words_and_data(&words);
    let actual_memory = memory_with_words_and_data(&words);
    let mut initial = memory_state();
    write_register(&mut initial, 0, 0xa000_0000);
    write_register(&mut initial, 3, 0xfeed_face_cafe_beef);
    let mut expected = initial.clone();
    let monitor = RefCell::new(nixe_cpu::exclusive::ExclusiveMonitorState::default());
    let events = nixe_cpu::execution::VcpuEventState::default();
    let context = InterpreterContext::new(cpu(), &expected_memory, &monitor, &FixedTimer, &events);
    for encoding in encodings {
        assert_eq!(
            execute_one_with_context(context, &mut expected, encoding).unwrap(),
            InstructionStep::Continue,
            "{encoding:#010x}"
        );
    }

    let mut actual = initial;
    let actual_events = nixe_cpu::execution::VcpuEventState::default();
    let exit = JitThread::new()
        .run_with_runtime(
            &JitProcess::new(cpu()).unwrap(),
            NativeRunRequest {
                memory: &actual_memory,
                state: &mut actual,
                instruction_budget: encodings.len() as u64,
                loader_return: None,
                timer: &FixedTimer,
                events: &actual_events,
            },
        )
        .unwrap();
    assert!(matches!(
        exit,
        DirectExit::Budget {
            instructions: 10,
            ..
        }
    ));
    assert_eq!(actual, expected);
}

#[test]
fn switch_1_pointer_authentication_hint_family_matches_the_interpreter() {
    for encoding in [
        0xd503_20ff_u32, // XPACLRI
        0xd503_211f,     // PACIA1716
        0xd503_215f,     // PACIB1716
        0xd503_219f,     // AUTIA1716
        0xd503_21df,     // AUTIB1716
        0xd503_231f,     // PACIAZ
        0xd503_233f,     // PACIASP
        0xd503_235f,     // PACIBZ
        0xd503_237f,     // PACIBSP
        0xd503_239f,     // AUTIAZ
        0xd503_23bf,     // AUTIASP
        0xd503_23df,     // AUTIBZ
        0xd503_23ff,     // AUTIBSP
    ] {
        let mut initial = rich_state();
        write_register(&mut initial, 30, 0xabcd_0000_7518_7c14);
        assert_matches_interpreter(encoding, initial);
    }
}

#[test]
fn scheduling_hints_publish_the_next_pc_and_exact_request() {
    let yield_memory = memory(&[(0, 0xd503_203f), (4, breakpoint(0))]);
    let mut yield_state = state(CODE, 0);
    let exit = JitThread::new()
        .run(
            &JitProcess::new(cpu()).unwrap(),
            &yield_memory,
            &mut yield_state,
            1,
        )
        .unwrap();
    assert_eq!(
        exit,
        DirectExit::Scheduled {
            pc: GuestVirtualAddress::new(CODE),
            request: nixe_cpu::execution::SchedulerRequest::Yield,
            instructions: 1,
        }
    );
    assert_eq!(yield_state.pc(), CODE + 4);

    let wfe_memory = memory(&[(0, 0xd503_205f), (4, breakpoint(0))]);
    let process = JitProcess::new(cpu()).unwrap();
    let thread = JitThread::new();
    let mut waiting = state(CODE, 0);
    assert_eq!(
        thread.run(&process, &wfe_memory, &mut waiting, 1).unwrap(),
        DirectExit::Scheduled {
            pc: GuestVirtualAddress::new(CODE),
            request: nixe_cpu::execution::SchedulerRequest::WaitForEvent,
            instructions: 1,
        }
    );
    thread.events.signal_event();
    let mut resumed = state(CODE, 0);
    assert!(matches!(
        thread.run(&process, &wfe_memory, &mut resumed, 1).unwrap(),
        DirectExit::Budget {
            instructions: 1,
            ..
        }
    ));
    assert_eq!(resumed.pc(), CODE + 4);
}

fn assert_memory_matches_interpreter(encoding: u32) {
    let expected_memory = memory_with_data(encoding);
    let actual_memory = memory_with_data(encoding);
    let mut expected_state = memory_state();
    let monitor = RefCell::new(nixe_cpu::exclusive::ExclusiveMonitorState::default());
    let events = nixe_cpu::execution::VcpuEventState::default();
    let context = InterpreterContext::new(cpu(), &expected_memory, &monitor, &ZeroTimer, &events);
    assert_eq!(
        execute_one_with_context(context, &mut expected_state, encoding).unwrap(),
        InstructionStep::Continue,
        "{encoding:#010x}"
    );

    let mut actual_state = memory_state();
    let exit = JitThread::new()
        .run(
            &JitProcess::new(cpu()).unwrap(),
            &actual_memory,
            &mut actual_state,
            1,
        )
        .unwrap_or_else(|error| panic!("{encoding:#010x}: {error}"));
    assert!(matches!(
        exit,
        DirectExit::Budget {
            instructions: 1,
            ..
        }
    ));
    assert_eq!(actual_state, expected_state, "{encoding:#010x}");
    let mut expected_bytes = [0_u8; 32];
    let mut actual_bytes = [0_u8; 32];
    expected_memory
        .read_bytes(SPACE, GuestVirtualAddress::new(DATA), &mut expected_bytes)
        .unwrap();
    actual_memory
        .read_bytes(SPACE, GuestVirtualAddress::new(DATA), &mut actual_bytes)
        .unwrap();
    assert_eq!(actual_bytes, expected_bytes, "{encoding:#010x}");
}

fn memory_state() -> A64State {
    let mut state = rich_state();
    write_register(&mut state, 1, DATA);
    write_register(&mut state, 2, 8);
    write_register(&mut state, 3, 0x8877_6655_4433_2211);
    state
}

fn memory_with_data(encoding: u32) -> SyntheticMemory {
    let mut memory = memory(&[(0, encoding), (4, breakpoint(0))]);
    let code_page = GuestPhysicalPageId::new(1);
    assert!(memory.initialize_ram(code_page, 8, &0x1020_3040_5060_7080_u64.to_le_bytes(),));
    let data_page = GuestPhysicalPageId::new(2);
    assert!(memory.add_ram_page(data_page));
    let bytes: Vec<_> = (0_u8..=255).collect();
    assert!(memory.initialize_ram(data_page, 0, &bytes));
    assert!(memory.map_page(
        SPACE,
        GuestVirtualAddress::new(DATA),
        data_page,
        MemoryPermissions::READ_WRITE,
    ));
    memory
}

fn memory_with_words_and_data(words: &[(u64, u32)]) -> SyntheticMemory {
    let mut memory = memory(words);
    let data_page = GuestPhysicalPageId::new(2);
    assert!(memory.add_ram_page(data_page));
    let bytes: Vec<_> = (0_u8..=255).collect();
    assert!(memory.initialize_ram(data_page, 0, &bytes));
    assert!(memory.map_page(
        SPACE,
        GuestVirtualAddress::new(DATA),
        data_page,
        MemoryPermissions::READ_WRITE,
    ));
    memory
}

fn assert_memory_prefix_equal(actual: &SyntheticMemory, expected: &SyntheticMemory, length: usize) {
    let mut actual_bytes = vec![0; length];
    let mut expected_bytes = vec![0; length];
    actual
        .read_bytes(SPACE, GuestVirtualAddress::new(DATA), &mut actual_bytes)
        .unwrap();
    expected
        .read_bytes(SPACE, GuestVirtualAddress::new(DATA), &mut expected_bytes)
        .unwrap();
    assert_eq!(actual_bytes, expected_bytes);
}

fn execution_memory_with_data(encoding: u32) -> ExecutionMemory {
    let mut memory = ExecutionMemory::new();
    let code_page = GuestPhysicalPageId::new(1);
    assert!(memory.add_ram_page(code_page));
    assert!(memory.initialize_ram(code_page, 0, &encoding.to_le_bytes()));
    assert!(memory.initialize_ram(code_page, 4, &breakpoint(0).to_le_bytes()));
    assert!(memory.map_page(
        SPACE,
        GuestVirtualAddress::new(CODE),
        code_page,
        MemoryPermissions::READ_EXECUTE,
    ));
    let data_page = GuestPhysicalPageId::new(2);
    assert!(memory.add_ram_page(data_page));
    let bytes: Vec<_> = (0_u8..=255).collect();
    assert!(memory.initialize_ram(data_page, 0, &bytes));
    assert!(memory.map_page(
        SPACE,
        GuestVirtualAddress::new(DATA),
        data_page,
        MemoryPermissions::READ_WRITE,
    ));
    memory
}

#[test]
fn every_scalar_integer_family_matches_the_reference_interpreter() {
    let encodings = [
        0x9100_0400, // ADD X0,X0,#1
        0xd280_0020, // MOVZ X0,#1
        0x8b01_0000, // ADD X0,X0,X1
        0x8b21_4000, // ADD X0,X0,W1,UXTW
        0x9a01_0000, // ADC X0,X0,X1
        0x9240_0000, // AND X0,X0,#1
        0xaa01_0000, // ORR X0,X0,X1
        0xd340_fc00, // UBFM X0,X0,#0,#63
        0x93c1_0400, // EXTR X0,X0,X1,#1
        0x9ac1_2000, // LSLV X0,X0,X1
        0xfa41_0000, // CCMP X0,X1,#0,EQ
        0xfa41_0800, // CCMP X0,#1,#0,EQ
        0x9a81_0000, // CSEL X0,X0,X1,EQ
        0x9b01_0800, // MADD X0,X0,X1,X2
        0xdac0_1000, // CLZ X0,X0
        0x1000_0000, // ADR X0,#0
        0x9000_0000, // ADRP X0,#0
    ];
    for encoding in encodings {
        assert_matches_interpreter(encoding, rich_state());
    }
}

#[test]
fn scalar_edge_cases_match_the_reference_interpreter() {
    let representative = [
        0xf2d5_79a0, // MOVK X0,#0xabcd,LSL#32
        0xab02_0c20, // ADDS X0,X1,X2,LSL#3
        0x6b82_1c20, // SUBS W0,W1,W2,ASR#7
        0x8b21_cbe0, // ADD X0,SP,W1,SXTW#2
        0xba02_0020, // ADCS X0,X1,X2
        0x7a02_0020, // SBCS W0,W1,W2
        0x9208_9c20, // AND X0,X1,#0xff00ff00ff00ff00
        0x6ae2_1420, // BICS W0,W1,W2,ROR#5
        0x331b_0c20, // BFI W0,W1,#5,#4
        0x3300_1020, // BFXIL W0,W1,#0,#5
        0x93c2_3420, // EXTR X0,X1,X2,#13
        0x9ac2_0820, // UDIV X0,X1,X2
        0x9ac2_0c20, // SDIV X0,X1,X2
        0x9ac2_2420, // LSRV X0,X1,X2
        0x9ac2_2820, // ASRV X0,X1,X2
        0x9ac2_2c20, // RORV X0,X1,X2
        0xfa42_102a, // CCMP X1,X2,#10,NE
        0xfa47_0825, // CCMP X1,#7,#5,EQ
        0x9a82_b420, // CSINC X0,X1,X2,LT
        0xda82_a020, // CSINV X0,X1,X2,GE
        0xda82_8420, // CSNEG X0,X1,X2,HI
        0x9b02_0c20, // MADD X0,X1,X2,X3
        0x1b02_8c20, // MSUB W0,W1,W2,W3
        0x9b22_0c20, // SMADDL X0,W1,W2,X3
        0x9b22_8c20, // SMSUBL X0,W1,W2,X3
        0x9b42_7c20, // SMULH X0,X1,X2
        0x9ba2_0c20, // UMADDL X0,W1,W2,X3
        0x9ba2_8c20, // UMSUBL X0,W1,W2,X3
        0x9bc2_7c20, // UMULH X0,X1,X2
        0xdac0_0020, // RBIT X0,X1
        0xdac0_0420, // REV16 X0,X1
        0xdac0_0820, // REV32 X0,X1
        0xdac0_0c20, // REV X0,X1
        0xdac0_1020, // CLZ X0,X1
        0xdac0_1420, // CLS X0,X1
    ];
    for encoding in representative {
        assert_matches_interpreter(encoding, rich_state());
    }

    let mut divide_by_zero = rich_state();
    write_register(&mut divide_by_zero, 2, 0);
    assert_matches_interpreter(0x9ac2_0820, divide_by_zero.clone());
    assert_matches_interpreter(0x9ac2_0c20, divide_by_zero);

    let mut signed_overflow = rich_state();
    write_register(&mut signed_overflow, 1, i64::MIN as u64);
    write_register(&mut signed_overflow, 2, u64::MAX);
    assert_matches_interpreter(0x9ac2_0c20, signed_overflow);
}

#[test]
fn scalar_control_edges_match_the_reference_interpreter() {
    let mut initial = rich_state();
    write_register(&mut initial, 0, CODE);
    for encoding in [
        0xd503_201f, // NOP
        0x1400_0000, // B .
        0x9400_0000, // BL .
        0xd61f_0000, // BR X0
        0xd63f_0000, // BLR X0
        0xd65f_03c0, // RET X30
        0x5400_0000, // B.EQ .
        0xb500_0000, // CBNZ X0,.
        0xb628_0000, // TBZ X0,#37,.
    ] {
        assert_control_matches_interpreter(encoding, initial.clone());
    }
}

#[test]
fn a64_register_invariants_are_explicit_in_the_direct_jit() {
    let words = [
        (0x00, 0x1100_0400), // ADD W0,W0,#1
        (0x04, 0x9100_23e2), // ADD X2,SP,#8
        (0x08, 0x9100_43ff), // ADD SP,SP,#16
        (0x0c, 0xaa01_03e3), // ORR X3,XZR,X1
        (0x10, 0xaa02_003f), // ORR XZR,X1,X2
        (0x14, breakpoint(0)),
    ];
    let memory = memory(&words);
    let mut expected = rich_state();
    let initial_sp = expected.read_x(A64Register::StackPointer);
    for (_, encoding) in words[..5].iter().copied() {
        assert_eq!(
            execute_one(&cpu().platform(), &mut expected, encoding).unwrap(),
            InstructionStep::Continue
        );
    }

    let mut actual = rich_state();
    assert_eq!(
        JitThread::new()
            .run(&JitProcess::new(cpu()).unwrap(), &memory, &mut actual, 5)
            .unwrap(),
        DirectExit::Budget {
            pc: GuestVirtualAddress::new(CODE + 20),
            instructions: 5,
        }
    );
    assert_eq!(actual, expected);
    assert_eq!(read_register(&actual, 0), 2);
    assert_eq!(read_register(&actual, 2), initial_sp + 8);
    assert_eq!(actual.read_x(A64Register::StackPointer), initial_sp + 16);
    assert_eq!(read_register(&actual, 3), read_register(&actual, 1));
}

#[test]
fn every_a64_condition_matches_for_every_nzcv_combination() {
    for condition in 0_u32..16 {
        let encoding = 0x9a82_0020 | (condition << 12); // CSEL X0,X1,X2,cond
        let memory = memory(&[(0, encoding), (4, breakpoint(0))]);
        let process = JitProcess::new(cpu()).unwrap();
        let thread = JitThread::new();
        for packed in 0_u32..16 {
            let mut actual = rich_state();
            actual.set_nzcv(Nzcv::from_bits(packed << 28));
            let mut expected = actual.clone();
            assert_eq!(
                execute_one(&cpu().platform(), &mut expected, encoding).unwrap(),
                InstructionStep::Continue
            );
            thread.run(&process, &memory, &mut actual, 1).unwrap();
            assert_eq!(
                actual, expected,
                "condition={condition:#x} nzcv={packed:#x}"
            );
        }
    }
}

#[test]
fn lazy_flags_merge_at_internal_join_without_canonical_reload() {
    let memory = memory(&[
        (0x00, conditional_branch(CODE, CODE + 0x0c, 0)), // B.EQ
        (0x04, 0xb100_0421),                              // ADDS X1,X1,#1
        (0x08, branch(CODE + 8, CODE + 0x14)),
        (0x0c, 0xf100_0421), // SUBS X1,X1,#1
        (0x10, branch(CODE + 0x10, CODE + 0x14)),
        (0x14, 0x9a82_0022), // CSEL X2,X1,X2,EQ
        (0x18, breakpoint(9)),
    ]);
    let process = JitProcess::new(cpu()).unwrap();
    let thread = JitThread::new();
    for initial_flags in [Nzcv::from_bits(Nzcv::Z), Nzcv::from_bits(0)] {
        let mut state = rich_state();
        state.set_nzcv(initial_flags);
        let mut expected = state.clone();
        for offset in if initial_flags.zero() {
            [0x00, 0x0c, 0x10, 0x14]
        } else {
            [0x00, 0x04, 0x08, 0x14]
        } {
            let encoding = memory_word(&memory, offset);
            execute_one(&cpu().platform(), &mut expected, encoding).unwrap();
        }
        thread.run(&process, &memory, &mut state, 4).unwrap();
        assert_eq!(state, expected);
    }

    let mut side_entry = rich_state();
    side_entry.set_pc(CODE + 0x14);
    side_entry.set_nzcv(Nzcv::from_bits(Nzcv::Z));
    let mut expected = side_entry.clone();
    for offset in [0x14, 0x18] {
        execute_one(
            &cpu().platform(),
            &mut expected,
            memory_word(&memory, offset),
        )
        .unwrap();
    }
    thread.run(&process, &memory, &mut side_entry, 2).unwrap();
    assert_eq!(side_entry, expected);
    assert_eq!(process.state.lock().unwrap().compiled_regions, 1);
}

#[test]
fn simple_scalar_shapes_remain_bounded() {
    for encoding in [0x9100_0400, 0xaa01_0000, 0x9a81_0000] {
        let memory = memory(&[(0, encoding), (4, breakpoint(0))]);
        let process = JitProcess::new(cpu()).unwrap();
        process.entry_for(&memory, location(CODE)).unwrap();
        let state = process.state.lock().unwrap();
        let region = state
            .region_for(RegionKey::new(cpu(), location(CODE)))
            .unwrap();
        assert!(
            region.clif_instructions <= 104,
            "{encoding:#010x}: {} CLIF instructions",
            region.clif_instructions
        );
        assert!(
            region.native_bytes <= 768,
            "{encoding:#010x}: {} native bytes",
            region.native_bytes
        );
    }
}

#[test]
fn memory_and_system_shapes_remain_bounded() {
    for encoding in [
        0xf940_0020, // LDR X0,[X1]
        0xf900_0023, // STR X3,[X1]
        0xc8df_fc20, // LDAR X0,[X1]
        0xc85f_7c20, // LDXR X0,[X1]
        0xd53b_e020, // MRS X0,CNTVCT_EL0
        0xd503_3bbf, // DMB ISH
    ] {
        let memory = memory(&[(0, encoding), (4, breakpoint(0))]);
        let process = JitProcess::new(cpu()).unwrap();
        process.entry_for(&memory, location(CODE)).unwrap();
        let state = process.state.lock().unwrap();
        let region = state
            .region_for(RegionKey::new(cpu(), location(CODE)))
            .unwrap();
        assert!(region.clif_instructions <= 512, "{encoding:#010x}");
        assert!(region.native_bytes <= 4096, "{encoding:#010x}");
    }
}

#[test]
fn linux_direct_scalar_memory_shapes_are_compact() {
    for (name, encoding, clif_limit, native_limit) in [
        ("load", 0xf940_0020_u32, 144, 896),
        ("store", 0xf900_0020_u32, 256, 1536),
    ] {
        let mut memory = execution_memory_with_data(encoding);
        memory
            .bind_cpu_memory_backend(SPACE, 0x1_0000, DirectBackendPolicy::Required)
            .unwrap();
        let process = JitProcess::new(cpu()).unwrap();
        process
            .bind_memory(MemoryBinding {
                address_space: SPACE,
                end_exclusive: GuestVirtualAddress::new(0x1_0000),
                memory: &memory,
                mapping_epoch: memory.mapping_epoch().get(),
                invalidation_cursor: memory.invalidation_cursor(),
            })
            .unwrap();
        process.entry_for(&memory, location(CODE)).unwrap();
        let state = process.state.lock().unwrap();
        let region = state
            .region_for(RegionKey::new(cpu(), location(CODE)))
            .unwrap();
        println!(
            "direct_shape={name} clif_instructions={} native_bytes={}",
            region.clif_instructions, region.native_bytes
        );
        assert!(
            region.clif_instructions <= clif_limit,
            "{name}: {} CLIF instructions exceed {clif_limit}",
            region.clif_instructions,
        );
        assert!(
            region.native_bytes <= native_limit,
            "{name}: {} native bytes exceed {native_limit}",
            region.native_bytes,
        );
    }
}

#[test]
#[ignore = "manual release microbenchmark"]
fn direct_jit_scalar_loops_materially_beat_checked_memory_by_width() {
    for (width, load, store) in [
        (1, 0x3940_0020_u32, 0x3900_0020_u32),
        (2, 0x7940_0020_u32, 0x7900_0020_u32),
        (4, 0xb940_0020_u32, 0xb900_0020_u32),
        (8, 0xf940_0020_u32, 0xf900_0020_u32),
    ] {
        let direct_read = jit_scalar_loop_ns(load, DirectBackendPolicy::Required);
        let checked_read = jit_scalar_loop_ns(load, DirectBackendPolicy::Disabled);
        let direct_store = jit_scalar_loop_ns(store, DirectBackendPolicy::Required);
        let checked_store = jit_scalar_loop_ns(store, DirectBackendPolicy::Disabled);
        println!(
            "width={width} direct_jit_ns_per_read={direct_read} checked_jit_ns_per_read={checked_read} direct_jit_ns_per_store={direct_store} checked_jit_ns_per_store={checked_store}"
        );
        assert!(
            direct_read * 10 < checked_read * 9,
            "width {width} direct JIT read did not improve checked memory by 10%"
        );
        assert!(
            direct_store * 10 < checked_store * 9,
            "width {width} direct JIT store did not improve checked memory by 10%"
        );
    }
}

fn jit_scalar_loop_ns(encoding: u32, policy: DirectBackendPolicy) -> u128 {
    const WARMUP_ITERATIONS: u64 = 10_000;
    const MEASURED_ITERATIONS: u64 = 1_000_000;
    let mut memory = ExecutionMemory::new();
    let code_page = GuestPhysicalPageId::new(1);
    let data_page = GuestPhysicalPageId::new(2);
    assert!(memory.add_ram_page(code_page));
    assert!(memory.add_ram_page(data_page));
    for (offset, word) in [
        (0, encoding),
        (4, 0xf100_0442),                           // SUBS X2,X2,#1
        (8, conditional_branch(CODE + 8, CODE, 1)), // B.NE
        (12, breakpoint(0)),
    ] {
        assert!(memory.initialize_ram(code_page, offset, &word.to_le_bytes()));
    }
    assert!(memory.map_page(
        SPACE,
        GuestVirtualAddress::new(CODE),
        code_page,
        MemoryPermissions::READ_EXECUTE,
    ));
    assert!(memory.map_page(
        SPACE,
        GuestVirtualAddress::new(DATA),
        data_page,
        MemoryPermissions::READ_WRITE,
    ));
    memory
        .bind_cpu_memory_backend(SPACE, 0x1_0000, policy)
        .unwrap();

    let process = JitProcess::new(cpu()).unwrap();
    let binding = MemoryBinding {
        address_space: SPACE,
        end_exclusive: GuestVirtualAddress::new(0x1_0000),
        memory: &memory,
        mapping_epoch: memory.mapping_epoch().get(),
        invalidation_cursor: memory.invalidation_cursor(),
    };
    process.bind_memory(binding).unwrap();
    let mut thread = JitThread::new();
    thread.synchronize_address_space(&process, binding).unwrap();

    let run = |iterations| {
        let mut state = memory_state();
        write_register(&mut state, 0, 0xa5c3_9678_1234_fedc);
        write_register(&mut state, 1, DATA);
        write_register(&mut state, 2, iterations);
        assert!(matches!(
            thread
                .run(&process, &memory, &mut state, iterations * 3)
                .unwrap(),
            DirectExit::Budget { .. }
        ));
    };
    run(WARMUP_ITERATIONS);
    let started = std::time::Instant::now();
    run(MEASURED_ITERATIONS);
    started.elapsed().as_nanos() / u128::from(MEASURED_ITERATIONS)
}

#[test]
fn switch_1_pointer_authentication_hints_add_no_clif_over_nop() {
    let metrics = |encoding| {
        let memory = memory(&[(0, encoding), (4, breakpoint(0))]);
        let process = JitProcess::new(cpu()).unwrap();
        process.entry_for(&memory, location(CODE)).unwrap();
        let state = process.state.lock().unwrap();
        let region = state
            .region_for(RegionKey::new(cpu(), location(CODE)))
            .unwrap();
        (region.clif_instructions, region.native_bytes)
    };
    let nop = metrics(0xd503_201f);

    for encoding in [
        0xd503_20ff_u32,
        0xd503_211f,
        0xd503_215f,
        0xd503_219f,
        0xd503_21df,
        0xd503_231f,
        0xd503_233f,
        0xd503_235f,
        0xd503_237f,
        0xd503_239f,
        0xd503_23bf,
        0xd503_23df,
        0xd503_23ff,
    ] {
        assert_eq!(metrics(encoding), nop, "{encoding:#010x}");
    }
}

#[test]
fn fp_simd_shapes_remain_bounded_by_family() {
    let cases = [
        ("integer-add", 0x4e22_8420_u32, 104, 768),
        ("bitwise-select", 0x6e62_1c20, 104, 768),
        ("compare", 0x4e21_34a3, 104, 768),
        ("min-max", 0x0ebf_6fdf, 104, 768),
        ("pairwise", 0x4e22_a420, 104, 768),
        ("permute", 0x4e02_1823, 104, 768),
        ("extract", 0x6e02_4023, 104, 768),
        ("narrow", 0x0f0c_8400, 104, 768),
        ("immediate-shift", 0x2f0f_0420, 104, 768),
        ("shift-left-long", 0x0f20_a7fe, 104, 768),
        ("register-shift", 0x0e22_4420, 144, 1280),
        ("reduction", 0x4e31_b862, 104, 768),
        ("exact-fp", 0x1e62_2820, 160, 1024),
        ("quadword-memory", 0x3dc0_0020, 208, 1280),
    ];
    for (family, encoding, clif_limit, native_limit) in cases {
        let memory = memory(&[(0, encoding), (4, breakpoint(0))]);
        let process = JitProcess::new(cpu()).unwrap();
        process.entry_for(&memory, location(CODE)).unwrap();
        let state = process.state.lock().unwrap();
        let region = state
            .region_for(RegionKey::new(cpu(), location(CODE)))
            .unwrap();
        assert!(
            region.clif_instructions <= clif_limit,
            "{family}: {} CLIF instructions exceed {clif_limit}",
            region.clif_instructions
        );
        assert!(
            region.native_bytes <= native_limit,
            "{family}: {} native bytes exceed {native_limit}",
            region.native_bytes
        );
    }
}

fn cpu() -> ProcessCpuContext {
    ProcessCpuContext::for_platform(TargetPlatform::Switch1, SPACE)
}

fn location(pc: u64) -> LocationDescriptor {
    location_for(cpu(), pc)
}

fn location_for(cpu: ProcessCpuContext, pc: u64) -> LocationDescriptor {
    LocationDescriptor::new(GuestVirtualAddress::new(pc), cpu.profile_id())
}

fn rich_state() -> A64State {
    let mut state = state(CODE, 0x8000_0000_0000_0001);
    write_register(&mut state, 1, 0xfedc_ba98_7654_3210);
    write_register(&mut state, 2, 3);
    write_register(&mut state, 3, 0x1234_5678_9abc_def0);
    write_register(&mut state, 30, CODE);
    state.write_x(A64Register::StackPointer, 0x7fff_0000);
    state.set_nzcv(Nzcv::from_bits(Nzcv::Z | Nzcv::C));
    state
}

fn write_register(state: &mut A64State, index: u8, value: u64) {
    state.write_x(
        A64Register::General(A64GeneralRegister::new(index).unwrap()),
        value,
    );
}

fn read_register(state: &A64State, index: u8) -> u64 {
    state.read_x(A64Register::General(
        A64GeneralRegister::new(index).unwrap(),
    ))
}

fn assert_matches_interpreter(encoding: u32, initial: A64State) {
    let mut expected = initial.clone();
    assert_eq!(
        execute_one(&cpu().platform(), &mut expected, encoding).unwrap(),
        InstructionStep::Continue,
        "{encoding:#010x}"
    );

    let memory = memory(&[(0, encoding), (4, breakpoint(0))]);
    let process = JitProcess::new(cpu()).unwrap();
    let mut actual = initial;
    JitThread::new()
        .run(&process, &memory, &mut actual, 1)
        .unwrap_or_else(|error| panic!("{encoding:#010x}: {error}"));
    assert_eq!(actual, expected, "{encoding:#010x}");
}

fn assert_control_matches_interpreter(encoding: u32, initial: A64State) {
    let mut expected = initial.clone();
    assert_eq!(
        execute_one(&cpu().platform(), &mut expected, encoding).unwrap(),
        InstructionStep::Continue,
        "{encoding:#010x}"
    );
    let memory = memory(&[(0, encoding), (4, breakpoint(0))]);
    let process = JitProcess::new(cpu()).unwrap();
    let mut actual = initial;
    JitThread::new()
        .run(&process, &memory, &mut actual, 1)
        .unwrap_or_else(|error| panic!("{encoding:#010x}: {error}"));
    assert_eq!(actual, expected, "{encoding:#010x}");
}

fn memory_word(memory: &SyntheticMemory, offset: u64) -> u32 {
    let fetched = nixe_cpu::memory::InstructionMemory::fetch32(
        memory,
        SPACE,
        GuestVirtualAddress::new(CODE + offset),
    )
    .unwrap();
    fetched.bits
}

fn state(pc: u64, x0: u64) -> A64State {
    let mut state = A64State::default();
    state.set_pc(pc);
    state.write_x(
        A64Register::General(A64GeneralRegister::new(0).unwrap()),
        x0,
    );
    state
}

fn read_x0(state: &A64State) -> u64 {
    state.read_x(A64Register::General(A64GeneralRegister::new(0).unwrap()))
}

fn memory(words: &[(u64, u32)]) -> SyntheticMemory {
    let mut memory = SyntheticMemory::new();
    let page = GuestPhysicalPageId::new(1);
    assert!(memory.add_ram_page(page));
    for &(offset, encoding) in words {
        assert!(memory.initialize_ram(page, offset as usize, &encoding.to_le_bytes()));
    }
    assert!(memory.map_page(
        SPACE,
        GuestVirtualAddress::new(CODE),
        page,
        MemoryPermissions::READ_EXECUTE,
    ));
    memory
}

fn add_x0(immediate: u16) -> u32 {
    0x9100_0000 | (u32::from(immediate) << 10)
}

fn branch(source: u64, target: u64) -> u32 {
    direct_branch(0x1400_0000, source, target)
}

fn branch_link(source: u64, target: u64) -> u32 {
    direct_branch(0x9400_0000, source, target)
}

fn direct_branch(base: u32, source: u64, target: u64) -> u32 {
    let displacement = target.wrapping_sub(source) as i64;
    base | (((displacement >> 2) as u32) & 0x03ff_ffff)
}

fn conditional_branch(source: u64, target: u64, condition: u8) -> u32 {
    let displacement = target.wrapping_sub(source) as i64;
    0x5400_0000 | ((((displacement >> 2) as u32) & 0x7ffff) << 5) | u32::from(condition)
}

fn compare_branch(source: u64, target: u64, register: u8, nonzero: bool) -> u32 {
    let displacement = target.wrapping_sub(source) as i64;
    0x3400_0000
        | (u32::from(nonzero) << 24)
        | ((((displacement >> 2) as u32) & 0x7ffff) << 5)
        | u32::from(register)
}

fn breakpoint(immediate: u16) -> u32 {
    0xd420_0000 | (u32::from(immediate) << 5)
}
