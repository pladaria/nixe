use std::cell::RefCell;
use std::collections::BTreeSet;
use std::mem::size_of;
use std::sync::{Arc, Barrier, Mutex, mpsc};
use std::time::{Duration, Instant};

use nixe_cpu::decode::{self, DecodeResult, DecodeSupport};
use nixe_cpu::execution::MemoryBinding;
use nixe_cpu::location::LocationDescriptor;
use nixe_cpu::memory::{
    CacheMaintenanceKind, CpuMemory, DataAccessFaultReason, ExecutionMemory, MemoryAccess,
    MemoryAccessClass, MemoryAccessSize, MemoryAlignment, MemoryAttributes, MemoryOrdering,
    MemoryPermissions, MemoryValue, ProcessMemory, SyntheticMemory, SyntheticMmio,
};
use nixe_cpu::platform::TargetPlatform;
use nixe_cpu::profile::ProcessCpuContext;
use nixe_cpu::state::a64::{A64GeneralRegister, A64Register, A64State, Nzcv};
use nixe_cpu_interpreter::{
    InstructionStep, InterpreterContext, execute_one, execute_one_with_context,
};
use nixe_memory::{
    AddressSpaceId, CanonicalRangeTranslator, CpuVisibilityRequest, DIRECT_PAGE_SIZE,
    DeviceAccessDeclaration, DeviceVisibilityPoint, DeviceVisibilityRequest, DirectBackendPolicy,
    GuestPhysicalPageId, GuestVirtualAddress, MemoryInvalidationKind, MemoryInvalidationSource,
    NonCpuDeviceId, VisibilityCoordinator, VisibilityCoordinatorError,
};

use super::lookup::{RegionLookup, index_for_pc, lookup_salt};
use super::region::{RegionKey, RegionLimits, discover_region};
use super::*;

const CODE: u64 = 0x1000;
const DATA: u64 = 0x8000;
const SPACE: AddressSpaceId = AddressSpaceId::new(1);

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct HostFpSnapshot {
    control: u64,
    status: u64,
}

#[cfg(target_arch = "x86_64")]
fn read_host_fp() -> HostFpSnapshot {
    use core::arch::asm;

    let mut mxcsr = 0_u32;
    unsafe { asm!("stmxcsr [{value}]", value = in(reg) &mut mxcsr, options(nostack)) };
    HostFpSnapshot {
        control: u64::from(mxcsr & !0x3f),
        status: u64::from(mxcsr & 0x3f),
    }
}

#[cfg(target_arch = "x86_64")]
fn write_host_fp(snapshot: HostFpSnapshot) {
    use core::arch::asm;

    let mxcsr = (snapshot.control | snapshot.status) as u32;
    unsafe { asm!("ldmxcsr [{value}]", value = in(reg) &mxcsr, options(nostack)) };
}

#[cfg(target_arch = "aarch64")]
fn read_host_fp() -> HostFpSnapshot {
    use core::arch::asm;

    let control: u64;
    let status: u64;
    unsafe {
        asm!("mrs {value}, fpcr", value = out(reg) control, options(nomem, nostack));
        asm!("mrs {value}, fpsr", value = out(reg) status, options(nomem, nostack));
    }
    HostFpSnapshot { control, status }
}

#[cfg(target_arch = "aarch64")]
fn write_host_fp(snapshot: HostFpSnapshot) {
    use core::arch::asm;

    unsafe {
        asm!("msr fpcr, {value}", value = in(reg) snapshot.control, options(nomem, nostack));
        asm!("msr fpsr, {value}", value = in(reg) snapshot.status, options(nomem, nostack));
    }
}

#[cfg(target_arch = "x86_64")]
fn distinct_host_fp(original: HostFpSnapshot, index: u32) -> HostFpSnapshot {
    HostFpSnapshot {
        control: (original.control & !(3 << 13)) | (u64::from(index & 3) << 13),
        status: 1 << (index % 6),
    }
}

#[cfg(target_arch = "aarch64")]
fn distinct_host_fp(original: HostFpSnapshot, index: u32) -> HostFpSnapshot {
    HostFpSnapshot {
        control: (original.control & !(3 << 22)) | (u64::from(index & 3) << 22),
        status: (original.status & !0x9f) | (1 << (index % 5)),
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
struct RestoreHostFp(HostFpSnapshot);

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
impl Drop for RestoreHostFp {
    fn drop(&mut self) {
        write_host_fp(self.0);
    }
}

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

struct BlockingDeviceWriteback {
    bytes: Box<[u8]>,
    entered: Arc<Barrier>,
    release: Arc<Barrier>,
}

impl VisibilityCoordinator for BlockingDeviceWriteback {
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
        self.entered.wait();
        self.release.wait();
        Ok(self.bytes.clone())
    }
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
    let exit = thread.run(&process, &memory, &mut state).unwrap();
    assert_eq!(
        exit,
        DirectExit::Architectural {
            pc: GuestVirtualAddress::new(CODE + 0x10),
            detail: (2 << 24) | 0x31,
            progress: 1,
        }
    );
    assert_eq!(read_x0(&state), 11);
}

#[test]
fn secondary_entry_is_published_as_a_separate_region() {
    let memory = memory(&[
        (0x00, 0xd503_201f),
        (0x04, branch(CODE + 4, CODE + 12)),
        (0x0c, add_x0(1)),
        (0x10, breakpoint(0x32)),
    ]);
    let process = JitProcess::new(cpu()).unwrap();
    let primary = process.entry_for(&memory, location(CODE)).unwrap().1;
    let secondary = process.entry_for(&memory, location(CODE + 0x0c)).unwrap().1;
    assert_ne!(secondary, primary);
    {
        let state = process.state.lock().unwrap();
        assert_eq!(state.regions.len(), 2);
        let secondary_key = RegionKey::new(cpu(), location(CODE + 0x0c));
        assert_eq!(
            state.lookup.get(secondary_key).unwrap().owner,
            secondary_key
        );
    }

    let mut state = state(CODE + 0x0c, 10);
    assert_eq!(
        JitThread::new().run(&process, &memory, &mut state).unwrap(),
        DirectExit::Architectural {
            pc: GuestVirtualAddress::new(CODE + 0x10),
            detail: (2 << 24) | 0x32,
            progress: 1,
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
    let secondary_entry = process.entry_for(&memory, location(secondary)).unwrap().1;
    assert_ne!(secondary_entry, target_entry);
    process.entry_for(&memory, location(CODE)).unwrap();

    let state = process.state.lock().unwrap();
    assert_eq!(state.regions.len(), 3);
    let source = state
        .region_for(RegionKey::new(cpu(), location(CODE)))
        .unwrap();
    assert_eq!(source.entry_keys.len(), 1);
    assert_eq!(
        source.links[0].slot.load(Ordering::Acquire),
        secondary_entry
    );
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
        thread.run(&process, &memory, &mut zero).unwrap(),
        DirectExit::Architectural {
            pc: GuestVirtualAddress::new(CODE + 8),
            detail: (2 << 24) | 1,
            progress: 1,
        }
    );
    assert!(zero.nzcv().zero());

    let mut nonzero = state(CODE, 2);
    assert_eq!(
        thread.run(&process, &memory, &mut nonzero).unwrap(),
        DirectExit::Architectural {
            pc: GuestVirtualAddress::new(CODE + 12),
            detail: (2 << 24) | 2,
            progress: 1,
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
        (0x18, breakpoint(0)),
    ]);
    let process = JitProcess::new(cpu()).unwrap();
    let mut state = state(CODE, 0);

    assert_eq!(
        JitThread::new().run(&process, &memory, &mut state).unwrap(),
        DirectExit::Architectural {
            pc: GuestVirtualAddress::new(CODE + 0x18),
            detail: 2 << 24,
            progress: 1,
        }
    );
    assert_eq!(read_register(&state, 5), 0x14);
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
        thread.run(&process, &memory, &mut state).unwrap(),
        DirectExit::Architectural {
            pc: GuestVirtualAddress::new(CODE + 4),
            detail: (2 << 24) | 3,
            progress: 1,
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
        thread.run(&process, &memory, &mut state).unwrap(),
        DirectExit::Architectural {
            pc: GuestVirtualAddress::new(return_address),
            detail: (2 << 24) | 7,
            progress: 1,
        }
    );
    assert_eq!(read_x0(&state), target + 1);
}

#[test]
fn normal_jit_does_not_single_step_while_entry_control_is_immediate() {
    let memory = memory(&[(0x00, 0xd503_201f), (0x04, breakpoint(0))]);
    let process = JitProcess::new(cpu()).unwrap();

    let thread = JitThread::new();
    let mut budget_state = state(CODE, 0);
    assert_eq!(
        thread.run(&process, &memory, &mut budget_state).unwrap(),
        DirectExit::Architectural {
            pc: GuestVirtualAddress::new(CODE + 4),
            detail: 2 << 24,
            progress: 1,
        }
    );

    let preempted = JitThread::new();
    preempted.request_preempt();
    let mut control_state = state(CODE, 0);
    assert_eq!(
        preempted
            .run(&process, &memory, &mut control_state)
            .unwrap(),
        DirectExit::Control {
            pc: GuestVirtualAddress::new(CODE),
            progress: 0,
        }
    );
}

#[test]
fn native_backedge_reaches_a_periodic_scheduler_safepoint() {
    let memory = memory(&[(0x00, branch(CODE, CODE))]);
    let process = JitProcess::new(cpu()).unwrap();
    let thread = JitThread::new();
    let mut state = state(CODE, 0);

    assert_eq!(
        thread.run(&process, &memory, &mut state).unwrap(),
        DirectExit::Control {
            pc: GuestVirtualAddress::new(CODE),
            progress: 0,
        }
    );
}

#[test]
fn periodic_native_backedge_exit_yields_to_the_runtime() {
    let memory = memory(&[(0x00, branch(CODE, CODE))]);
    let process = JitProcess::new(cpu()).unwrap();
    let mut thread = JitThread::new();
    let mut state = state(CODE, 0);

    let report = thread
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
        .unwrap();
    assert_eq!(report.stop, CpuExit::Safepoint);
    assert_eq!(report.progress, 0);
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
    assert_eq!(process.state.lock().unwrap().regions.len(), 1);
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
    process.entry_for(&memory, location(CODE + 0x0c)).unwrap();
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
    assert_eq!(state.retired.len(), 2);
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
        (4, branch(CODE + 4, CODE + 8)),
        (8, 0xd503_201f), // NOP
        (12, branch(CODE + 12, CODE + 8)),
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
            GuestVirtualAddress::new(DATA + 8),
            MemoryAccess::normal(MemoryAccessSize::Word),
            MemoryValue::U32(breakpoint(9)),
        )
        .unwrap();
    memory
        .maintain_cache(
            SPACE,
            CacheMaintenanceKind::InstructionInvalidate,
            Some(GuestVirtualAddress::new(DATA + 8)),
        )
        .unwrap();
    release.wait();
    assert_eq!(
        worker.join().unwrap().unwrap(),
        DirectExit::Architectural {
            pc: GuestVirtualAddress::new(CODE + 8),
            detail: (2 << 24) | 9,
            progress: 1,
        }
    );
    let state = process.state.lock().unwrap();
    assert_eq!(state.retired.len(), 1);
    assert_eq!(state.regions.len(), 1);
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
                        0x30..=0x43 | 0x48..=0x5d | 0x60..=0xa1
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
            let exit = JitThread::new().run(&process, &memory, &mut state).unwrap();
            let DirectExit::Unsupported { pc, progress } = exit else {
                panic!("{} did not take the unsupported exit", pattern.name);
            };
            assert_eq!(pc, GuestVirtualAddress::new(CODE));
            assert_eq!(progress, 0);
            assert!(matches!(
                unsupported_exit(&process, &memory, pc, progress, &state).unwrap(),
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
                &mut actual
            )
            .unwrap(),
        DirectExit::Architectural {
            pc: GuestVirtualAddress::new(CODE + 16),
            detail: 2 << 24,
            progress: 1,
        }
    );
    assert_eq!(actual, expected);
    assert_memory_prefix_equal(&actual_memory, &expected_memory, 64);
}

#[test]
fn direct_fp_matches_normal_special_rounding_saturation_and_status_cases() {
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

    let single_lanes = [1.5_f32, -2.0, 4.0, 8.0]
        .into_iter()
        .enumerate()
        .fold(0_u128, |bits, (lane, value)| {
            bits | (u128::from(value.to_bits()) << (lane * 32))
        });
    let double_lanes = u128::from(1.5_f64.to_bits()) | (u128::from((-2.0_f64).to_bits()) << 64);
    for (encoding, rn, rm, lanes) in [
        (0x0f82_9020, 1, 2, single_lanes),   // FMUL V0.2S,V1.2S,V2.S[0]
        (0x0fa5_9883, 4, 5, single_lanes),   // FMUL V3.2S,V4.2S,V5.S[3]
        (0x4fa8_90e6, 7, 8, single_lanes),   // FMUL V6.4S,V7.4S,V8.S[1]
        (0x4f8b_9949, 10, 11, single_lanes), // FMUL V9.4S,V10.4S,V11.S[2]
        (0x4fce_91ac, 13, 14, double_lanes), // FMUL V12.2D,V13.2D,V14.D[0]
        (0x4fd1_9a0f, 16, 17, double_lanes), // FMUL V15.2D,V16.2D,V17.D[1]
        (0x4fa2_9821, 1, 2, single_lanes),   // FMUL V1.4S,V1.4S,V2.S[3]
        (0x4fc4_9863, 3, 4, double_lanes),   // FMUL V3.2D,V3.2D,V4.D[1]
        (0x4f96_93fb, 31, 22, single_lanes), // observed FMUL V27.4S,V31.4S,V22.S[0]
    ] {
        let mut by_element = rich_state();
        assert!(by_element.set_vector(rn, lanes));
        assert!(by_element.set_vector(rm, lanes));
        assert_matches_interpreter(encoding, by_element);
    }

    let mut to_integer = rich_state();
    assert!(to_integer.set_vector(1, u128::from((-3.75_f64).to_bits())));
    assert_matches_interpreter(0x9e78_0020, to_integer); // FCVTZS X0,D1

    let mut from_integer = rich_state();
    write_register(&mut from_integer, 2, (1_u64 << 53) + 1);
    assert_matches_interpreter(0x9e62_0043, from_integer); // SCVTF D3,X2

    let mut precision = rich_state();
    assert!(precision.set_vector(5, u128::from(1.25_f32.to_bits())));
    assert_matches_interpreter(0x1e22_c0a4, precision); // FCVT D4,S5
    let mut demote = rich_state();
    assert!(demote.set_vector(7, u128::from((1.0_f64 + 2.0_f64.powi(-24)).to_bits())));
    assert_matches_interpreter(0x1e62_40e6, demote); // FCVT S6,D7

    let mut round = rich_state();
    assert!(round.set_vector(9, u128::from(2.5_f64.to_bits())));
    assert_matches_interpreter(0x1e64_4128, round); // FRINTN D8,D9

    for encoding in [
        0x4e21_dbfc_u32, // SCVTF V28.4S,V31.4S
        0x2e21_d8e6,     // UCVTF V6.2S,V7.2S
        0x6e61_d96a,     // UCVTF V10.2D,V11.2D
        0x7e21_d9ad,     // UCVTF S13,S13
        0x6e3e_ff9c,     // FDIV V28.4S,V28.4S,V30.4S
    ] {
        assert_matches_interpreter(encoding, rich_state());
    }

    let mut arithmetic = rich_state();
    assert!(arithmetic.set_vector(1, u128::from(5.5_f32.to_bits())));
    assert!(arithmetic.set_vector(2, u128::from(2.0_f32.to_bits())));
    assert_matches_interpreter(0x1e22_3820, arithmetic); // FSUB S0,S1,S2

    let mut multiply = rich_state();
    assert!(multiply.set_vector(29, u128::from(1.5_f64.to_bits())));
    assert!(multiply.set_vector(28, u128::from(2.0_f64.to_bits())));
    assert_matches_interpreter(0x1e7c_0bbc, multiply); // FMUL D28,D29,D28

    let mut negated_multiply = rich_state();
    assert!(negated_multiply.set_vector(7, u128::from(2.0_f32.to_bits())));
    assert!(negated_multiply.set_vector(8, u128::from((-3.0_f32).to_bits())));
    assert_matches_interpreter(0x1e28_88e6, negated_multiply); // FNMUL S6,S7,S8

    for encoding in [0x1f02_0c20, 0x1f02_8c20, 0x1f22_0c20, 0x1f22_8c20] {
        let mut fused = rich_state();
        assert!(fused.set_vector(1, u128::from(2.0_f32.to_bits())));
        assert!(fused.set_vector(2, u128::from(3.0_f32.to_bits())));
        assert!(fused.set_vector(3, u128::from(4.0_f32.to_bits())));
        assert_matches_interpreter(encoding, fused);
    }

    let mut square_root = rich_state();
    assert!(square_root.set_vector(1, u128::from(2.0_f64.to_bits())));
    assert_matches_interpreter(0x1e61_c020, square_root); // FSQRT D0,D1

    let mut signaling_compare = rich_state();
    assert!(signaling_compare.set_vector(0, u128::from(0x7ff8_0000_0000_0001_u64)));
    assert!(signaling_compare.set_vector(1, u128::from(1.0_f64.to_bits())));
    assert_matches_interpreter(0x1e61_2010, signaling_compare); // FCMPE D0,D1

    let mut conditional_compare = rich_state();
    assert!(conditional_compare.set_vector(31, u128::from(1.0_f64.to_bits())));
    assert!(conditional_compare.set_vector(30, u128::from(2.0_f64.to_bits())));
    conditional_compare.set_nzcv(Nzcv::from_bits(0)); // NE holds.
    assert_matches_interpreter(0x1e7e_17e4, conditional_compare); // FCCMP D31,D30,#4,NE

    let mut signed_zero = rich_state();
    signed_zero.set_fpcr(2 << 22); // round toward negative infinity
    assert!(signed_zero.set_vector(1, u128::from(1.0_f64.to_bits())));
    assert!(signed_zero.set_vector(2, u128::from((-1.0_f64).to_bits())));
    assert_matches_interpreter(0x1e62_2820, signed_zero); // FADD D0,D1,D2
}

#[test]
fn direct_fpcr_dn_fz_widths_and_vector_arrangements_match_the_interpreter() {
    for (encoding, nan, one) in [
        (
            0x1e22_2820_u32, // FADD S0,S1,S2
            u64::from(0x7fc1_2345_u32),
            u64::from(1.0_f32.to_bits()),
        ),
        (
            0x1e62_2820, // FADD D0,D1,D2
            0x7ff8_0000_0000_1234,
            1.0_f64.to_bits(),
        ),
    ] {
        let mut state = rich_state();
        state.set_fpcr(1 << 25); // DN
        assert!(state.set_vector(1, u128::from(nan)));
        assert!(state.set_vector(2, u128::from(one)));
        assert_matches_interpreter(encoding, state);
    }

    for (encoding, denormal, one) in [
        (
            0x1e22_0820_u32, // FMUL S0,S1,S2
            1_u64,
            u64::from(1.0_f32.to_bits()),
        ),
        (
            0x1e62_0820, // FMUL D0,D1,D2
            1_u64,
            1.0_f64.to_bits(),
        ),
    ] {
        let mut state = rich_state();
        state.set_fpcr(1 << 24); // FZ
        assert!(state.set_vector(1, u128::from(denormal)));
        assert!(state.set_vector(2, u128::from(one)));
        assert_matches_interpreter(encoding, state);
    }

    let v2s_numerator = u128::from(1.0_f32.to_bits()) | (u128::from((-1.0_f32).to_bits()) << 32);
    let v2s_denominator = u128::from(3.0_f32.to_bits()) | (u128::from(3.0_f32.to_bits()) << 32);
    let exact = nixe_cpu::semantics::a64_fp_simd::exact_vector_float_divide(
        v2s_numerator,
        v2s_denominator,
        32,
        64,
        0,
    );
    assert_eq!(
        nixe_cpu::semantics::a64_fp_simd::fp_status_bits(exact.status),
        1 << 4
    );
    for rounding in 0_u32..=3 {
        let mut state = rich_state();
        state.set_fpcr(rounding << 22);
        assert!(state.set_vector(1, v2s_numerator));
        assert!(state.set_vector(2, v2s_denominator));
        assert_matches_interpreter(0x2e22_fc20, state); // FDIV V0.2S,V1.2S,V2.2S
    }

    let mut v4s = rich_state();
    let v4s_numerator = [1.0_f32, -1.0, 5.0, -5.0]
        .into_iter()
        .enumerate()
        .fold(0_u128, |bits, (lane, value)| {
            bits | (u128::from(value.to_bits()) << (lane * 32))
        });
    let v4s_denominator = [3.0_f32, 3.0, 2.0, 2.0]
        .into_iter()
        .enumerate()
        .fold(0_u128, |bits, (lane, value)| {
            bits | (u128::from(value.to_bits()) << (lane * 32))
        });
    assert!(v4s.set_vector(28, v4s_numerator));
    assert!(v4s.set_vector(30, v4s_denominator));
    assert_matches_interpreter(0x6e3e_ff9c, v4s); // FDIV V28.4S,V28.4S,V30.4S

    let mut v2d = rich_state();
    assert!(v2d.set_vector(
        7,
        u128::from(1.0_f64.to_bits()) | (u128::from((-6.0_f64).to_bits()) << 64)
    ));
    assert!(v2d.set_vector(
        8,
        u128::from(3.0_f64.to_bits()) | (u128::from(2.0_f64.to_bits()) << 64)
    ));
    assert_matches_interpreter(0x6e68_fce6, v2d); // FDIV V6.2D,V7.2D,V8.2D
}

#[test]
fn direct_to_exact_fp_boundary_preserves_cumulative_status_and_traps() {
    let cumulative_words = [
        (0, 0x1e62_2820), // FADD D0,D1,D2: IXC
        (4, 0x1e67_c083), // FRINTI D3,D4: IOC for signaling NaN
        (8, 0xd53b_4425), // MRS X5,FPSR
        (12, breakpoint(0)),
    ];
    let cumulative_memory = memory(&cumulative_words);
    let mut expected = rich_state();
    expected.set_fpsr(0x80);
    assert!(expected.set_vector(1, u128::from(1.0_f64.to_bits())));
    assert!(expected.set_vector(2, u128::from((2.0_f64.powi(-53)).to_bits())));
    assert!(expected.set_vector(4, u128::from(0x7ff0_0000_0000_0001_u64)));
    for (_, encoding) in cumulative_words[..3].iter().copied() {
        assert_eq!(
            execute_one(&cpu().platform(), &mut expected, encoding).unwrap(),
            InstructionStep::Continue
        );
    }
    let mut actual = rich_state();
    actual.set_fpsr(0x80);
    assert!(actual.set_vector(1, u128::from(1.0_f64.to_bits())));
    assert!(actual.set_vector(2, u128::from((2.0_f64.powi(-53)).to_bits())));
    assert!(actual.set_vector(4, u128::from(0x7ff0_0000_0000_0001_u64)));
    JitThread::new()
        .run(
            &JitProcess::new(cpu()).unwrap(),
            &cumulative_memory,
            &mut actual,
        )
        .unwrap();
    assert_eq!(actual, expected);
    assert_eq!(read_register(&actual, 5) & 0x91, 0x91);

    let trap_words = [
        (0, 0x1e62_2820), // FADD D0,D1,D2: IXC
        (4, 0xd51b_4406), // MSR FPCR,X6: enable IOE and end the segment
        (8, 0x1e67_c083), // FRINTI D3,D4: synchronous invalid-operation trap
        (12, breakpoint(0)),
    ];
    let trap_memory = memory(&trap_words);
    let mut expected = rich_state();
    expected.set_fpsr(0x80);
    write_register(&mut expected, 6, 1 << 8);
    assert!(expected.set_vector(1, u128::from(1.0_f64.to_bits())));
    assert!(expected.set_vector(2, u128::from((2.0_f64.powi(-53)).to_bits())));
    assert!(expected.set_vector(3, u128::from(0x0123_4567_89ab_cdef_u64)));
    assert!(expected.set_vector(4, u128::from(0x7ff0_0000_0000_0001_u64)));
    for (_, encoding) in trap_words[..2].iter().copied() {
        assert_eq!(
            execute_one(&cpu().platform(), &mut expected, encoding).unwrap(),
            InstructionStep::Continue
        );
    }
    assert!(matches!(
        execute_one(&cpu().platform(), &mut expected, trap_words[2].1).unwrap(),
        InstructionStep::Exit(nixe_cpu::execution::CpuExit::ArchitecturalException {
            kind: nixe_cpu::exception::ExceptionKind::FloatingPoint,
            ..
        })
    ));

    let mut actual = rich_state();
    actual.set_fpsr(0x80);
    write_register(&mut actual, 6, 1 << 8);
    assert!(actual.set_vector(1, u128::from(1.0_f64.to_bits())));
    assert!(actual.set_vector(2, u128::from((2.0_f64.powi(-53)).to_bits())));
    assert!(actual.set_vector(3, u128::from(0x0123_4567_89ab_cdef_u64)));
    assert!(actual.set_vector(4, u128::from(0x7ff0_0000_0000_0001_u64)));
    assert!(matches!(
        JitThread::new().run(&JitProcess::new(cpu()).unwrap(), &trap_memory, &mut actual),
        Ok(DirectExit::Architectural {
            pc,
            detail,
            ..
        }) if pc == GuestVirtualAddress::new(CODE + 8) && detail == 6 << 24
    ));
    assert_eq!(actual, expected);
    assert_eq!(actual.fpsr() & 0x90, 0x90);
}

#[test]
fn repeated_direct_fp_operations_do_not_duplicate_the_cold_protocol() {
    let shape = |encoding, repetitions: usize| {
        let destination = encoding & 0x1f;
        // Give every guarded failure site the same dirty-destination publication
        // obligation so the delta isolates the repeated FP lowering itself.
        let mut words = vec![(0, 0x4f00_0400 | destination)]; // MOVI Vd.4S,#0
        words.extend((0..repetitions).map(|index| (((index + 1) * 4) as u64, encoding)));
        words.push((((repetitions + 1) * 4) as u64, breakpoint(0)));
        let memory = memory(&words);
        let process = JitProcess::new(cpu()).unwrap();
        process.entry_for(&memory, location(CODE)).unwrap();
        let state = process.state.lock().unwrap();
        let region = state
            .region_for(RegionKey::new(cpu(), location(CODE)))
            .unwrap();
        (region.clif_instructions, region.native_bytes)
    };
    for (name, encoding, expected_clif_delta, x86_native_limit, a64_native_limit) in [
        ("FADD D", 0x1e62_2820, 75, 544, 336),
        ("FDIV V.4S", 0x6e3e_ff9c, 95, 704, 400),
        ("FMUL V.4S element", 0x4f96_93fb, 78, 544, 352),
    ] {
        let one = shape(encoding, 1);
        let two = shape(encoding, 2);
        let three = shape(encoding, 3);
        let clif_deltas = [two.0 - one.0, three.0 - two.0];
        let native_deltas = [two.1 - one.1, three.1 - two.1];
        assert_eq!(
            clif_deltas, [expected_clif_delta; 2],
            "{name} changed its per-operation CLIF shape: one={one:?} two={two:?} three={three:?}"
        );
        let native_limit = if cfg!(target_arch = "x86_64") {
            x86_native_limit
        } else {
            a64_native_limit
        };
        assert!(
            native_deltas.into_iter().all(|delta| delta <= native_limit),
            "{name} duplicated native boundary work: one={one:?} two={two:?} three={three:?}"
        );
        assert!(native_deltas[1] <= native_deltas[0]);
        assert!(clif_deltas[0] < one.0 && native_deltas[0] < one.1);
    }
}

#[test]
fn repeated_exact_fp_operations_share_the_cold_boundary_protocol() {
    let shape = |words: &[(u64, u32)]| {
        let memory = memory(words);
        let process = JitProcess::new(cpu()).unwrap();
        process.entry_for(&memory, location(CODE)).unwrap();
        let state = process.state.lock().unwrap();
        let region = state
            .region_for(RegionKey::new(cpu(), location(CODE)))
            .unwrap();
        (region.clif_instructions, region.native_bytes)
    };
    let one = shape(&[
        // Equalize the minimum state publication required if either typed call
        // fails; the measured delta then excludes an unrelated first dirty V0.
        (0, 0x4f00_0400), // MOVI V0.4S,#0
        (4, 0x1e67_c020), // FRINTI D0,D1
        (8, breakpoint(0)),
    ]);
    let two = shape(&[
        (0, 0x4f00_0400), // MOVI V0.4S,#0
        (4, 0x1e67_c020), // FRINTI D0,D1
        (8, 0x1e67_c020), // FRINTI D0,D1
        (12, breakpoint(0)),
    ]);
    let three = shape(&[
        (0, 0x4f00_0400),  // MOVI V0.4S,#0
        (4, 0x1e67_c020),  // FRINTI D0,D1
        (8, 0x1e67_c020),  // FRINTI D0,D1
        (12, 0x1e67_c020), // FRINTI D0,D1
        (16, breakpoint(0)),
    ]);
    let clif_deltas = [two.0 - one.0, three.0 - two.0];
    let native_deltas = [two.1 - one.1, three.1 - two.1];
    assert_eq!(
        clif_deltas, [32; 2],
        "exact FP changed its per-operation CLIF shape: one={one:?} two={two:?} three={three:?}"
    );
    let native_limit = if cfg!(target_arch = "x86_64") {
        256
    } else {
        192
    };
    assert!(
        native_deltas.into_iter().all(|delta| delta <= native_limit),
        "repeated exact FP duplicated native boundary work: one={one:?} two={two:?} three={three:?}"
    );
    assert!(native_deltas[1] <= native_deltas[0]);
    assert!(clif_deltas[0] < one.0 && native_deltas[0] < one.1);
}

#[test]
fn direct_fp_rounding_and_conversion_matrix_matches_the_interpreter() {
    for encoding in [
        0x1e64_4020_u32, // FRINTN D0,D1
        0x1e64_c020,     // FRINTP D0,D1
        0x1e65_4020,     // FRINTM D0,D1
        0x1e65_c020,     // FRINTZ D0,D1
        0x1e66_4020,     // FRINTA D0,D1
    ] {
        let mut state = rich_state();
        assert!(state.set_vector(1, u128::from(1.5_f64.to_bits())));
        assert_matches_interpreter(encoding, state);
    }

    for fpcr in [0, 1 << 22, 2 << 22, 3 << 22] {
        for encoding in [
            0x1e67_4020_u32, // FRINTX D0,D1
            0x1e67_c020,     // FRINTI D0,D1
        ] {
            let mut state = rich_state();
            state.set_fpcr(fpcr);
            assert!(state.set_vector(1, u128::from((-1.75_f64).to_bits())));
            assert_matches_interpreter(encoding, state);
        }
    }

    for encoding in [
        0x1e20_0020_u32, // FCVTNS W0,S1
        0x1e24_0020,     // FCVTAS W0,S1
        0x1e28_0020,     // FCVTPS W0,S1
        0x1e30_0020,     // FCVTMS W0,S1
        0x1e38_0020,     // FCVTZS W0,S1
        0x1e19_e020,     // FCVTZU W0,S1,#8
        0x1e58_f820,     // FCVTZS W0,D1,#2
        0x9e59_c020,     // FCVTZU X0,D1,#16
    ] {
        let mut state = rich_state();
        assert!(state.set_vector(1, u128::from((-3.75_f64).to_bits())));
        assert_matches_interpreter(encoding, state);
    }

    for (encoding, value) in [
        (0x1e22_0020_u32, u64::from(u32::MAX)), // SCVTF S0,W1
        (0x1e23_0020, u64::from(u32::MAX)),     // UCVTF S0,W1
        (0x9e62_0020, (1_u64 << 53) + 1),       // SCVTF D0,X1
        (0x9e63_0020, u64::MAX),                // UCVTF D0,X1
    ] {
        let mut state = rich_state();
        write_register(&mut state, 1, value);
        assert_matches_interpreter(encoding, state);
    }
}

#[test]
fn direct_native_fp_status_materializes_across_rust_and_mrs_boundaries() {
    let words = [
        (0, 0x1e62_2820), // FADD D0,D1,D2
        (4, 0xd53b_e008), // MRS X8,CNTFRQ_EL0 (typed Rust boundary)
        (8, 0xd53b_4424), // MRS X4,FPSR
        (12, breakpoint(0)),
    ];
    let expected_memory = memory(&words);
    let actual_memory = memory(&words);
    let mut expected = rich_state();
    assert!(expected.set_vector(1, u128::from(1.0_f64.to_bits())));
    assert!(expected.set_vector(2, u128::from((2.0_f64.powi(-53)).to_bits())));
    let monitor = RefCell::new(nixe_cpu::exclusive::ExclusiveMonitorState::default());
    let events = nixe_cpu::execution::VcpuEventState::default();
    let context = InterpreterContext::new(cpu(), &expected_memory, &monitor, &ZeroTimer, &events);
    for (_, encoding) in words[..3].iter().copied() {
        assert_eq!(
            execute_one_with_context(context, &mut expected, encoding).unwrap(),
            InstructionStep::Continue
        );
    }

    let mut actual = rich_state();
    assert!(actual.set_vector(1, u128::from(1.0_f64.to_bits())));
    assert!(actual.set_vector(2, u128::from((2.0_f64.powi(-53)).to_bits())));
    JitThread::new()
        .run(
            &JitProcess::new(cpu()).unwrap(),
            &actual_memory,
            &mut actual,
        )
        .unwrap();
    assert_eq!(actual, expected);
    assert_ne!(read_register(&actual, 4) & (1 << 4), 0);
}

#[test]
fn linked_exact_region_preserves_inherited_native_fp_status() {
    const EXACT: u64 = CODE + 0x100;
    let encodings = [
        0x1e62_2820, // FADD D0,D1,D2: IXC
        branch(CODE + 4, EXACT),
        0x1e67_c083, // FRINTI D3,D4
        0xd53b_4425, // MRS X5,FPSR
    ];
    let memory = memory(&[
        (0, encodings[0]),
        (4, encodings[1]),
        (0x100, encodings[2]),
        (0x104, encodings[3]),
        (0x108, breakpoint(0)),
    ]);
    let mut expected = rich_state();
    assert!(expected.set_vector(1, u128::from(1.0_f64.to_bits())));
    assert!(expected.set_vector(2, u128::from((2.0_f64.powi(-53)).to_bits())));
    assert!(expected.set_vector(4, u128::from(1.75_f64.to_bits())));
    for encoding in encodings {
        assert_eq!(
            execute_one(&cpu().platform(), &mut expected, encoding).unwrap(),
            InstructionStep::Continue
        );
    }

    let process = JitProcess::new(cpu()).unwrap();
    process.entry_for(&memory, location(EXACT)).unwrap();
    let mut actual = rich_state();
    assert!(actual.set_vector(1, u128::from(1.0_f64.to_bits())));
    assert!(actual.set_vector(2, u128::from((2.0_f64.powi(-53)).to_bits())));
    assert!(actual.set_vector(4, u128::from(1.75_f64.to_bits())));
    JitThread::new()
        .run(&process, &memory, &mut actual)
        .unwrap();
    assert_eq!(actual, expected);
    assert_ne!(read_register(&actual, 5) & (1 << 4), 0);
}

#[test]
fn direct_native_fp_status_materializes_at_replacement_and_scheduler_boundaries() {
    let words = [
        (0, 0x1e62_2820),  // FADD D0,D1,D2: IXC
        (4, 0x1e65_0883),  // FMUL D3,D4,D5: OFC | IXC
        (8, 0xd53b_4426),  // MRS X6,FPSR
        (12, 0xd51b_4427), // MSR FPSR,X7
        (16, 0xd53b_4428), // MRS X8,FPSR
        (20, breakpoint(0)),
    ];
    let actual_memory = memory(&words);
    let mut expected = rich_state();
    assert!(expected.set_vector(1, u128::from(1.0_f64.to_bits())));
    assert!(expected.set_vector(2, u128::from((2.0_f64.powi(-53)).to_bits())));
    assert!(expected.set_vector(4, u128::from(f64::MAX.to_bits())));
    assert!(expected.set_vector(5, u128::from(2.0_f64.to_bits())));
    write_register(&mut expected, 7, 0x82);
    for (_, encoding) in words[..5].iter().copied() {
        assert_eq!(
            execute_one(&cpu().platform(), &mut expected, encoding).unwrap(),
            InstructionStep::Continue
        );
    }

    let mut actual = rich_state();
    assert!(actual.set_vector(1, u128::from(1.0_f64.to_bits())));
    assert!(actual.set_vector(2, u128::from((2.0_f64.powi(-53)).to_bits())));
    assert!(actual.set_vector(4, u128::from(f64::MAX.to_bits())));
    assert!(actual.set_vector(5, u128::from(2.0_f64.to_bits())));
    write_register(&mut actual, 7, 0x82);
    JitThread::new()
        .run(
            &JitProcess::new(cpu()).unwrap(),
            &actual_memory,
            &mut actual,
        )
        .unwrap();
    assert_eq!(actual, expected);
    assert_eq!(read_register(&actual, 6) & 0x1f, (1 << 2) | (1 << 4));
    assert_eq!(read_register(&actual, 8), 0x82);

    let schedule_memory = memory(&[
        (0, 0x1e62_2820), // FADD
        (4, 0xd503_205f), // WFE
        (8, breakpoint(0)),
    ]);
    let mut scheduled = rich_state();
    assert!(scheduled.set_vector(1, u128::from(1.0_f64.to_bits())));
    assert!(scheduled.set_vector(2, u128::from((2.0_f64.powi(-53)).to_bits())));
    assert_eq!(
        JitThread::new()
            .run(
                &JitProcess::new(cpu()).unwrap(),
                &schedule_memory,
                &mut scheduled,
            )
            .unwrap(),
        DirectExit::Scheduled {
            pc: GuestVirtualAddress::new(CODE + 4),
            request: nixe_cpu::execution::SchedulerRequest::WaitForEvent,
            progress: 1,
        }
    );
    assert_eq!(scheduled.pc(), CODE + 8);
    assert_ne!(scheduled.fpsr() & (1 << 4), 0);
}

#[test]
fn direct_fpcr_write_ends_the_native_mode_segment() {
    let words = [
        (0, 0x1e62_2820),  // FADD D0,D1,D2
        (4, 0xd51b_4405),  // MSR FPCR,X5
        (8, 0x1e62_2823),  // FADD D3,D1,D2
        (12, 0xd53b_4424), // MRS X4,FPSR
        (16, breakpoint(0)),
    ];
    let actual_memory = memory(&words);
    let mut expected = rich_state();
    write_register(&mut expected, 5, 1 << 22); // round toward +infinity
    assert!(expected.set_vector(1, u128::from(1.0_f64.to_bits())));
    assert!(expected.set_vector(2, u128::from((2.0_f64.powi(-53)).to_bits())));
    for (_, encoding) in words[..4].iter().copied() {
        assert_eq!(
            execute_one(&cpu().platform(), &mut expected, encoding).unwrap(),
            InstructionStep::Continue
        );
    }

    let mut actual = rich_state();
    write_register(&mut actual, 5, 1 << 22);
    assert!(actual.set_vector(1, u128::from(1.0_f64.to_bits())));
    assert!(actual.set_vector(2, u128::from((2.0_f64.powi(-53)).to_bits())));
    JitThread::new()
        .run(
            &JitProcess::new(cpu()).unwrap(),
            &actual_memory,
            &mut actual,
        )
        .unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn direct_fpcr_write_updates_the_exact_boundary_mode_cache() {
    let words = [
        (0, 0xd51b_4405), // MSR FPCR,X5
        (4, 0x1e67_c020), // FRINTI D0,D1
        (8, breakpoint(0)),
    ];
    let actual_memory = memory(&words);
    let mut expected = rich_state();
    write_register(&mut expected, 5, 1 << 22); // round toward +infinity
    assert!(expected.set_vector(1, u128::from((-1.75_f64).to_bits())));
    for (_, encoding) in words[..2].iter().copied() {
        assert_eq!(
            execute_one(&cpu().platform(), &mut expected, encoding).unwrap(),
            InstructionStep::Continue
        );
    }

    let mut actual = rich_state();
    write_register(&mut actual, 5, 1 << 22);
    assert!(actual.set_vector(1, u128::from((-1.75_f64).to_bits())));
    JitThread::new()
        .run(
            &JitProcess::new(cpu()).unwrap(),
            &actual_memory,
            &mut actual,
        )
        .unwrap();
    assert_eq!(actual, expected);
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[test]
fn direct_native_fp_environment_restores_the_calling_thread() {
    let original = read_host_fp();
    let _restore = RestoreHostFp(original);
    let caller = distinct_host_fp(original, 1);
    write_host_fp(caller);
    let mut state = rich_state();
    assert!(state.set_vector(1, u128::from(1.25_f64.to_bits())));
    assert!(state.set_vector(2, u128::from(2.5_f64.to_bits())));
    JitThread::new()
        .run(
            &JitProcess::new(cpu()).unwrap(),
            &memory(&[(0, 0x1e62_2820), (4, breakpoint(0))]),
            &mut state,
        )
        .unwrap();
    assert_eq!(read_host_fp(), caller);

    #[cfg(target_arch = "x86_64")]
    {
        // The x86 lowering for CLIF rounding can raise MXCSR.PE even though Arm's
        // non-exact FRINT forms do not update FPSR. The JIT must therefore keep
        // this operation off the native x86 status path and preserve a clean
        // caller status word.
        let caller = HostFpSnapshot {
            control: original.control & !(3 << 13),
            status: 0,
        };
        write_host_fp(caller);
        let mut state = rich_state();
        assert!(state.set_vector(9, u128::from(2.5_f64.to_bits())));
        JitThread::new()
            .run(
                &JitProcess::new(cpu()).unwrap(),
                &memory(&[(0, 0x1e64_4128), (4, breakpoint(0))]), // FRINTN D8,D9
                &mut state,
            )
            .unwrap();
        assert_eq!(read_host_fp(), caller);
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[test]
fn direct_native_fp_state_is_isolated_between_concurrent_vcpus() {
    let memory = Arc::new(memory(&[
        (0, 0x1e62_2820), // FADD D0,D1,D2
        (4, 0xd53b_4424), // MRS X4,FPSR
        (8, breakpoint(0)),
    ]));
    let process = Arc::new(JitProcess::new(cpu()).unwrap());
    let ready = Arc::new(Barrier::new(2));
    let mut workers = Vec::new();

    for index in 0_u32..2 {
        let memory = Arc::clone(&memory);
        let process = Arc::clone(&process);
        let ready = Arc::clone(&ready);
        workers.push(std::thread::spawn(move || {
            let original = read_host_fp();
            let restore = RestoreHostFp(original);
            let caller = distinct_host_fp(original, index + 1);
            write_host_fp(caller);

            let mut state = rich_state();
            let initial_fpsr = if index == 0 { 0 } else { 1 << 1 };
            state.set_fpsr(initial_fpsr);
            assert!(state.set_vector(1, u128::from(1.0_f64.to_bits())));
            let rhs = if index == 0 { 2.0_f64.powi(-53) } else { 1.0 };
            assert!(state.set_vector(2, u128::from(rhs.to_bits())));

            ready.wait();
            let exit = JitThread::new().run(process.as_ref(), memory.as_ref(), &mut state);
            let observed_host = read_host_fp();
            drop(restore);
            (exit, state, caller, observed_host, initial_fpsr)
        }));
    }

    for worker in workers {
        let (exit, state, caller, observed_host, initial_fpsr) = worker.join().unwrap();
        assert!(matches!(exit.unwrap(), DirectExit::Architectural { .. }));
        assert_eq!(observed_host, caller);
        let expected_fpsr = if initial_fpsr == 0 {
            1 << 4
        } else {
            initial_fpsr
        };
        assert_eq!(state.fpsr(), expected_fpsr);
        assert_eq!(read_register(&state, 4), u64::from(expected_fpsr));
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[test]
fn direct_native_fp_state_survives_vcpu_host_thread_migration() {
    let memory = Arc::new(memory(&[
        (0, 0x1e62_2820),  // FADD D0,D1,D2
        (4, 0xd503_205f),  // WFE
        (8, 0x1e62_2823),  // FADD D3,D1,D2
        (12, 0xd53b_4424), // MRS X4,FPSR
        (16, breakpoint(0)),
    ]));
    let process = Arc::new(JitProcess::new(cpu()).unwrap());
    let mut initial = rich_state();
    assert!(initial.set_vector(1, u128::from(1.0_f64.to_bits())));
    assert!(initial.set_vector(2, u128::from((2.0_f64.powi(-53)).to_bits())));

    let first_memory = Arc::clone(&memory);
    let first_process = Arc::clone(&process);
    let first = std::thread::spawn(move || {
        let original = read_host_fp();
        let restore = RestoreHostFp(original);
        let caller = distinct_host_fp(original, 1);
        write_host_fp(caller);
        let mut state = initial;
        let exit = JitThread::new().run(first_process.as_ref(), first_memory.as_ref(), &mut state);
        let observed_host = read_host_fp();
        drop(restore);
        (exit, state, caller, observed_host)
    });
    let (first_exit, scheduled, first_caller, first_observed) = first.join().unwrap();
    assert_eq!(
        first_exit.unwrap(),
        DirectExit::Scheduled {
            pc: GuestVirtualAddress::new(CODE + 4),
            request: nixe_cpu::execution::SchedulerRequest::WaitForEvent,
            progress: 1,
        }
    );
    assert_eq!(first_observed, first_caller);
    assert_eq!(scheduled.pc(), CODE + 8);
    assert_eq!(scheduled.fpsr() & (1 << 4), 1 << 4);

    let second_memory = Arc::clone(&memory);
    let second_process = Arc::clone(&process);
    let second = std::thread::spawn(move || {
        let original = read_host_fp();
        let restore = RestoreHostFp(original);
        let caller = distinct_host_fp(original, 2);
        write_host_fp(caller);
        let mut state = scheduled;
        let exit =
            JitThread::new().run(second_process.as_ref(), second_memory.as_ref(), &mut state);
        let observed_host = read_host_fp();
        drop(restore);
        (exit, state, caller, observed_host)
    });
    let (second_exit, completed, second_caller, second_observed) = second.join().unwrap();
    assert!(matches!(
        second_exit.unwrap(),
        DirectExit::Architectural { .. }
    ));
    assert_eq!(second_observed, second_caller);
    assert_eq!(completed.fpsr() & (1 << 4), 1 << 4);
    assert_eq!(read_register(&completed, 4) & (1 << 4), 1 << 4);
}

#[test]
fn direct_every_enabled_fp_exception_is_precise_and_atomic() {
    let mut invalid = rich_state();
    invalid.set_fpcr(1 << 8); // IOE
    assert!(invalid.set_vector(1, u128::from(f64::INFINITY.to_bits())));
    assert!(invalid.set_vector(2, u128::from(f64::NEG_INFINITY.to_bits())));
    assert_fp_exception_matches_interpreter(0x1e62_2820, invalid); // FADD D0,D1,D2

    let mut divide_by_zero = rich_state();
    divide_by_zero.set_fpcr(1 << 9); // DZE
    assert!(divide_by_zero.set_vector(1, u128::from(1.0_f64.to_bits())));
    assert!(divide_by_zero.set_vector(2, 0));
    assert_fp_exception_matches_interpreter(0x1e62_1820, divide_by_zero); // FDIV D0,D1,D2

    let mut overflow = rich_state();
    overflow.set_fpcr(1 << 10); // OFE
    assert!(overflow.set_vector(1, u128::from(f64::MAX.to_bits())));
    assert!(overflow.set_vector(2, u128::from(2.0_f64.to_bits())));
    assert_fp_exception_matches_interpreter(0x1e62_0820, overflow); // FMUL D0,D1,D2

    let mut underflow = rich_state();
    underflow.set_fpcr(1 << 11); // UFE
    assert!(underflow.set_vector(1, u128::from(0x0010_0000_0000_0000_u64)));
    assert!(underflow.set_vector(2, u128::from(0x0010_0000_0000_0000_u64)));
    assert_fp_exception_matches_interpreter(0x1e62_0820, underflow); // FMUL D0,D1,D2

    let mut inexact = rich_state();
    inexact.set_fpcr(1 << 12); // IXE
    assert!(inexact.set_vector(1, u128::from(1.0_f64.to_bits())));
    assert!(inexact.set_vector(2, u128::from((2.0_f64.powi(-53)).to_bits())));
    assert_fp_exception_matches_interpreter(0x1e62_2820, inexact); // FADD D0,D1,D2

    let mut input_denormal = rich_state();
    input_denormal.set_fpcr((1 << 15) | (1 << 24)); // IDE | FZ
    assert!(input_denormal.set_vector(1, 1));
    assert!(input_denormal.set_vector(2, u128::from(1.0_f64.to_bits())));
    assert_fp_exception_matches_interpreter(0x1e62_0820, input_denormal); // FMUL D0,D1,D2
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
        0xf800_0023, // STUR X3,[X1]
        0xf840_8420, // LDR X0,[X1],#8
        0xf800_8423, // STR X3,[X1],#8
        0xf840_8c20, // LDR X0,[X1,#8]!
        0xf800_8c23, // STR X3,[X1,#8]!
        0xf862_6820, // LDR X0,[X1,X2]
        0xf822_6823, // STR X3,[X1,X2]
        0xa940_0c20, // LDP X0,X3,[X1]
        0xa900_0c20, // STP X0,X3,[X1]
        0xa8c1_0c20, // LDP X0,X3,[X1],#16
        0xa881_0c20, // STP X0,X3,[X1],#16
        0xa9c1_0c20, // LDP X0,X3,[X1,#16]!
        0xa981_0c20, // STP X0,X3,[X1,#16]!
        0x6940_0c20, // LDPSW X0,X3,[X1]
        0x5800_0040, // LDR X0,[PC,#8]
        0x08df_fc20, // LDARB W0,[X1]
        0x48df_fc20, // LDARH W0,[X1]
        0x88df_fc20, // LDAR W0,[X1]
        0xc8df_fc20, // LDAR X0,[X1]
        0x089f_fc23, // STLRB W3,[X1]
        0x489f_fc23, // STLRH W3,[X1]
        0x889f_fc23, // STLR W3,[X1]
        0xc89f_fc23, // STLR X3,[X1]
    ];
    for encoding in cases {
        assert_memory_matches_interpreter(encoding);
    }
}

#[test]
fn linux_direct_ordered_scalar_accesses_use_the_shared_arena() {
    for (size, load, store, value) in [
        (
            MemoryAccessSize::Byte,
            0x08df_fc20_u32,
            0x089f_fc23_u32,
            0xa5_u64,
        ),
        (MemoryAccessSize::Halfword, 0x48df_fc20, 0x489f_fc23, 0xa5c3),
        (
            MemoryAccessSize::Word,
            0x88df_fc20,
            0x889f_fc23,
            0xa5c3_9678,
        ),
        (
            MemoryAccessSize::Doubleword,
            0xc8df_fc20,
            0xc89f_fc23,
            0xa5c3_9678_1234_fedc,
        ),
    ] {
        let mut memory = execution_memory_with_data(store);
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
        write_register(&mut state, 3, value);
        assert!(matches!(
            thread.run(&process, &memory, &mut state).unwrap(),
            DirectExit::Architectural { .. }
        ));
        assert_eq!(
            memory
                .read(
                    SPACE,
                    GuestVirtualAddress::new(DATA),
                    MemoryAccess::normal(size)
                )
                .unwrap()
                .value,
            MemoryValue::from_bits(size, u128::from(value)),
        );

        let mut memory = execution_memory_with_data(load);
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
            thread.run(&process, &memory, &mut state).unwrap(),
            DirectExit::Architectural { .. }
        ));
        let expected = u64::from_le_bytes([0, 1, 2, 3, 4, 5, 6, 7])
            & if size == MemoryAccessSize::Doubleword {
                u64::MAX
            } else {
                (1_u64 << (size.bytes() * 8)) - 1
            };
        assert_eq!(read_register(&state, 0), expected);
    }
}

#[test]
fn linux_direct_ordered_access_rejects_misalignment_before_native_memory() {
    let mut memory = execution_memory_with_data(0xc8df_fc20); // LDAR X0,[X1]
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
    write_register(&mut state, 1, DATA + 1);

    let DirectExit::DataFault { fault, .. } = thread.run(&process, &memory, &mut state).unwrap()
    else {
        panic!("misaligned ordered load did not report a data fault");
    };
    assert_eq!(
        fault.reason,
        DataAccessFaultReason::Misaligned {
            required_alignment: 8,
        }
    );
}

#[test]
fn simd_structure_memory_shapes_match_the_interpreter() {
    for encoding in [
        0x4c40_7020, // LD1 {V0.16B},[X1]
        0x0c40_7020, // LD1 {V0.8B},[X1]
        0x0c40_7420, // LD1 {V0.4H},[X1]
        0x0c40_7820, // LD1 {V0.2S},[X1]
        0x0c40_7c20, // LD1 {V0.1D},[X1]
        0x4c00_7020, // ST1 {V0.16B},[X1]
        0x4c40_a020, // LD1 {V0.16B,V1.16B},[X1]
        0x4c40_6020, // LD1 {V0.16B-V2.16B},[X1]
        0x4c40_2020, // LD1 {V0.16B-V3.16B},[X1]
        0x4c40_8020, // LD2 {V0.16B,V1.16B},[X1]
        0x4c40_4020, // LD3 {V0.16B-V2.16B},[X1]
        0x4c40_0020, // LD4 {V0.16B-V3.16B},[X1]
        0x4c00_8020, // ST2 {V0.16B,V1.16B},[X1]
        0x4c00_4020, // ST3 {V0.16B-V2.16B},[X1]
        0x4c00_0020, // ST4 {V0.16B-V3.16B},[X1]
        0x0d40_0c20, // LD1 {V0.B}[3],[X1]
        0x0d60_0c20, // LD2 {V0.B,V1.B}[3],[X1]
        0x0d40_2c20, // LD3 {V0.B-V2.B}[3],[X1]
        0x0d60_2c20, // LD4 {V0.B-V3.B}[3],[X1]
        0x0d00_0c20, // ST1 {V0.B}[3],[X1]
        0x0d20_0c20, // ST2 {V0.B,V1.B}[3],[X1]
        0x0d00_2c20, // ST3 {V0.B-V2.B}[3],[X1]
        0x0d20_2c20, // ST4 {V0.B-V3.B}[3],[X1]
        0x4d40_c020, // LD1R {V0.16B},[X1]
        0x0d40_c020, // LD1R {V0.8B},[X1]
        0x0d40_c420, // LD1R {V0.4H},[X1]
        0x0d40_c820, // LD1R {V0.2S},[X1]
        0x0d40_cc20, // LD1R {V0.1D},[X1]
        0x4d60_c020, // LD2R {V0.16B,V1.16B},[X1]
        0x4d40_e020, // LD3R {V0.16B-V2.16B},[X1]
        0x4d60_e020, // LD4R {V0.16B-V3.16B},[X1]
        0x4cdf_883e, // LD2 {V30.4S,V31.4S},[X1],#32
        0x4c82_443f, // ST3 {V31.8H-V1.8H},[X1],X2
        0x0dff_783e, // LD4 {V30.H-V1.H}[3],[X1],#8
        0x0d40_8020, // LD1 {V0.S}[0],[X1]
        0x0d40_8420, // LD1 {V0.D}[0],[X1]
        0x4da2_843f, // ST2 {V31.D,V0.D}[1],[X1],X2
        0x4ddf_e83f, // LD3R {V31.4S-V1.4S},[X1],#12
        0x4de2_ec3e, // LD4R {V30.2D-V1.2D},[X1],X2
    ] {
        assert_memory_matches_interpreter(encoding);
    }
}

#[test]
fn linux_direct_scalar_indexed_faults_suppress_writeback() {
    for (encoding, base) in [
        (0xf840_8420_u32, DATA + DIRECT_PAGE_SIZE as u64), // LDR X0,[X1],#8
        (
            0xf840_8c20,
            DATA + DIRECT_PAGE_SIZE as u64 - size_of::<u64>() as u64,
        ), // LDR X0,[X1,#8]!
    ] {
        let expected_memory = memory_with_data(encoding);
        let mut expected = memory_state();
        write_register(&mut expected, 1, base);
        let monitor = RefCell::new(nixe_cpu::exclusive::ExclusiveMonitorState::default());
        let events = nixe_cpu::execution::VcpuEventState::default();
        let context =
            InterpreterContext::new(cpu(), &expected_memory, &monitor, &ZeroTimer, &events);
        assert!(matches!(
            execute_one_with_context(context, &mut expected, encoding).unwrap(),
            InstructionStep::Exit(nixe_cpu::execution::CpuExit::DataFault { .. })
        ));

        let mut actual_memory = execution_memory_with_data(encoding);
        actual_memory
            .bind_cpu_memory_backend(SPACE, 0x1_0000, DirectBackendPolicy::Required)
            .unwrap();
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
        let mut actual = memory_state();
        write_register(&mut actual, 1, base);
        assert!(matches!(
            thread.run(&process, &actual_memory, &mut actual).unwrap(),
            DirectExit::DataFault { .. }
        ));
        assert_eq!(actual, expected, "{encoding:#010x}");
        assert_eq!(read_register(&actual, 1), base);
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
        thread.run(&process, &memory, &mut state).unwrap();

        state.set_pc(CODE);
        write_register(&mut state, 0, 0);
        thread.run(&process, &memory, &mut state).unwrap();
    }
}

#[test]
fn linux_direct_uncached_ram_uses_the_same_native_alias() {
    let mut memory = execution_memory_with_data(0xf940_0020); // LDR X0,[X1]
    memory
        .bind_cpu_memory_backend(SPACE, 0x1_0000, DirectBackendPolicy::Required)
        .unwrap();
    memory
        .set_attributes(
            SPACE,
            GuestVirtualAddress::new(DATA),
            4096,
            MemoryAttributes::UNCACHED,
            MemoryAttributes::UNCACHED,
        )
        .unwrap();
    assert_eq!(
        memory.direct_protection_at(SPACE, GuestVirtualAddress::new(DATA)),
        Some(nixe_memory::DirectProtection::ReadWrite)
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

    assert!(matches!(
        thread.run(&process, &memory, &mut state).unwrap(),
        DirectExit::Architectural { .. }
    ));
    assert_eq!(
        read_x0(&state),
        u64::from_le_bytes([0, 1, 2, 3, 4, 5, 6, 7])
    );
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

    let error = thread.run(&process, &replacement, &mut state).unwrap_err();
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
    assert_eq!(report.progress, 1);
    assert!(matches!(
        report.stop,
        CpuExit::ArchitecturalException { .. }
    ));
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
        thread.run(&process, &memory, &mut state).unwrap(),
        DirectExit::Architectural { .. }
    ));
    assert_eq!(read_register(&state, 0), 0xa5c3_9678_1234_fedc);
}

#[test]
fn linux_direct_fault_in_flight_survives_mapping_transition_and_jit_shutdown() {
    let mut memory = execution_memory_with_data(0xf940_0020); // LDR X0,[X1]
    memory
        .bind_cpu_memory_backend(SPACE, 0x1_0000, DirectBackendPolicy::Required)
        .unwrap();
    let retained = memory
        .translate_canonical_range(
            SPACE,
            GuestVirtualAddress::new(DATA),
            DIRECT_PAGE_SIZE as u64,
            MemoryPermissions::READ_WRITE,
        )
        .unwrap();
    let entered = Arc::new(Barrier::new(2));
    let release = Arc::new(Barrier::new(2));
    let expected = 0xa5c3_9678_1234_fedc_u64;
    let mut bytes = vec![0_u8; DIRECT_PAGE_SIZE];
    bytes[..size_of::<u64>()].copy_from_slice(&expected.to_le_bytes());
    let coordinator: Arc<dyn VisibilityCoordinator> = Arc::new(BlockingDeviceWriteback {
        bytes: bytes.into_boxed_slice(),
        entered: Arc::clone(&entered),
        release: Arc::clone(&release),
    });
    let declaration = DeviceAccessDeclaration::write(
        NonCpuDeviceId::new(21),
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

    let memory = Arc::new(memory);
    let process = Arc::new(JitProcess::new(cpu()).unwrap());
    let binding = MemoryBinding {
        address_space: SPACE,
        end_exclusive: GuestVirtualAddress::new(0x1_0000),
        memory: memory.as_ref(),
        mapping_epoch: memory.mapping_epoch().get(),
        invalidation_cursor: memory.invalidation_cursor(),
    };
    process.bind_memory(binding).unwrap();
    let worker_memory = Arc::clone(&memory);
    let worker_process = Arc::clone(&process);
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
        let mut state = memory_state();
        let exit = thread.run(worker_process.as_ref(), worker_memory.as_ref(), &mut state);
        (exit, read_register(&state, 0))
    });

    entered.wait();
    let transition_memory = Arc::clone(&memory);
    let transition = std::thread::spawn(move || {
        transition_memory.set_attributes(
            SPACE,
            GuestVirtualAddress::new(DATA),
            DIRECT_PAGE_SIZE as u64,
            MemoryAttributes::UNCACHED,
            MemoryAttributes::UNCACHED,
        )
    });
    let wait_started = Instant::now();
    while !memory.mapping_mutation_pending() {
        assert!(
            wait_started.elapsed() < Duration::from_secs(1),
            "mapping transition did not wait for the faulting native slice"
        );
        std::thread::yield_now();
    }
    process.shutdown();
    release.wait();

    let (exit, observed) = worker.join().unwrap();
    assert!(matches!(exit.unwrap(), DirectExit::Architectural { .. }));
    assert_eq!(observed, expected);
    transition.join().unwrap().unwrap();
    assert_eq!(
        memory.direct_protection_at(SPACE, GuestVirtualAddress::new(DATA)),
        Some(nixe_memory::DirectProtection::Read),
    );
}

#[test]
fn linux_direct_first_write_retries_then_stores_remain_native() {
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
            thread.run(&process, &memory, &mut state).unwrap(),
            DirectExit::Architectural { .. }
        ));
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
            thread.run(&process, &memory, &mut state).unwrap(),
            DirectExit::Architectural { .. }
        ));
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
fn linux_direct_fault_retry_preserves_pending_native_fpsr() {
    let mut memory = execution_memory_with_words_and_data(&[
        (0, 0x1e62_2820), // FADD D0,D1,D2: IXC remains in host status
        (4, 0xf900_0023), // STR X3,[X1]: first write faults and retries
        (8, 0xd53b_4424), // MRS X4,FPSR
        (12, breakpoint(0)),
    ]);
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

    let stored = 0xa5c3_9678_1234_fedc_u64;
    let mut state = memory_state();
    write_register(&mut state, 1, DATA);
    write_register(&mut state, 3, stored);
    assert!(state.set_vector(1, u128::from(1.0_f64.to_bits())));
    assert!(state.set_vector(2, u128::from((2.0_f64.powi(-53)).to_bits())));

    assert!(matches!(
        thread.run(&process, &memory, &mut state).unwrap(),
        DirectExit::Architectural { .. }
    ));
    assert_eq!(state.fpsr() & (1 << 4), 1 << 4);
    assert_eq!(read_register(&state, 4) & (1 << 4), 1 << 4);
    assert_eq!(
        memory
            .read(
                SPACE,
                GuestVirtualAddress::new(DATA),
                MemoryAccess::normal(MemoryAccessSize::Doubleword),
            )
            .unwrap()
            .value,
        MemoryValue::from_bits(MemoryAccessSize::Doubleword, u128::from(stored)),
    );
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
        thread.run(&process, &memory, &mut state).unwrap(),
        DirectExit::Architectural { .. }
    ));
    assert_eq!(read_register(&state, 0), 0x0706_0504_0302_0100);
    assert_eq!(read_register(&state, 3), 0x0f0e_0d0c_0b0a_0908);
}

#[test]
fn linux_direct_pair_load_retries_its_second_cross_page_element() {
    let mut memory = ExecutionMemory::new();
    let code_page = GuestPhysicalPageId::new(1);
    let first_data_page = GuestPhysicalPageId::new(2);
    let second_data_page = GuestPhysicalPageId::new(3);
    for page in [code_page, first_data_page, second_data_page] {
        assert!(memory.add_ram_page(page));
    }
    assert!(memory.initialize_ram(code_page, 0, &0xa940_0c20_u32.to_le_bytes()));
    assert!(memory.initialize_ram(code_page, 4, &breakpoint(0).to_le_bytes()));
    let first = 0x8877_6655_4433_2211_u64;
    let second = 0x1020_3040_5060_7080_u64;
    assert!(memory.initialize_ram(
        first_data_page,
        DIRECT_PAGE_SIZE - size_of::<u64>(),
        &first.to_le_bytes(),
    ));
    assert!(memory.map_page(
        SPACE,
        GuestVirtualAddress::new(CODE),
        code_page,
        MemoryPermissions::READ_EXECUTE,
    ));
    assert!(memory.map_page(
        SPACE,
        GuestVirtualAddress::new(DATA),
        first_data_page,
        MemoryPermissions::READ_WRITE,
    ));
    assert!(memory.map_page(
        SPACE,
        GuestVirtualAddress::new(DATA + DIRECT_PAGE_SIZE as u64),
        second_data_page,
        MemoryPermissions::READ_WRITE,
    ));
    memory
        .bind_cpu_memory_backend(SPACE, 0x1_0000, DirectBackendPolicy::Required)
        .unwrap();

    let retained = memory
        .translate_canonical_range(
            SPACE,
            GuestVirtualAddress::new(DATA + DIRECT_PAGE_SIZE as u64),
            DIRECT_PAGE_SIZE as u64,
            MemoryPermissions::READ_WRITE,
        )
        .unwrap();
    let mut bytes = vec![0_u8; DIRECT_PAGE_SIZE];
    bytes[..size_of::<u64>()].copy_from_slice(&second.to_le_bytes());
    let coordinator: Arc<dyn VisibilityCoordinator> = Arc::new(DeviceWriteback {
        bytes: bytes.into_boxed_slice(),
    });
    let declaration = DeviceAccessDeclaration::write(
        NonCpuDeviceId::new(20),
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
    write_register(
        &mut state,
        1,
        DATA + DIRECT_PAGE_SIZE as u64 - size_of::<u64>() as u64,
    );

    assert!(matches!(
        thread.run(&process, &memory, &mut state).unwrap(),
        DirectExit::Architectural { .. }
    ));
    assert_eq!(read_register(&state, 0), first);
    assert_eq!(read_register(&state, 3), second);
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
        thread.run(&process, &memory, &mut state).unwrap(),
        DirectExit::Architectural { .. }
    ));
    assert_eq!(state.vector(2), Some(first));

    let second = 0xffee_ddcc_bbaa_9988_7766_5544_3322_1100;
    state.set_pc(CODE);
    assert!(state.set_vector(0, second));
    assert!(matches!(
        thread.run(&process, &memory, &mut state).unwrap(),
        DirectExit::Architectural { .. }
    ));
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
fn linux_direct_interleaved_simd_structure_load_uses_the_shared_arena() {
    let mut memory = execution_memory_with_data(0x4c40_8020); // LD2 {V0.16B,V1.16B},[X1]
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
        thread.run(&process, &memory, &mut state).unwrap(),
        DirectExit::Architectural { .. }
    ));
    assert_eq!(
        state.vector(0),
        Some(u128::from_le_bytes(std::array::from_fn(|lane| {
            (lane * 2) as u8
        })))
    );
    assert_eq!(
        state.vector(1),
        Some(u128::from_le_bytes(std::array::from_fn(|lane| {
            (lane * 2 + 1) as u8
        })))
    );
}

#[test]
fn linux_direct_same_page_unaligned_scalar_accesses_are_native() {
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
        thread.run(&process, &memory, &mut state).unwrap(),
        DirectExit::Architectural { .. }
    ));
    assert_eq!(read_register(&state, 2), value);
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
fn linux_direct_cross_page_store_is_all_or_nothing() {
    let mut memory = ExecutionMemory::new();
    for page in 1..=3 {
        assert!(memory.add_ram_page(GuestPhysicalPageId::new(page)));
    }
    assert!(memory.initialize_ram(
        GuestPhysicalPageId::new(1),
        0,
        &0xf900_0020_u32.to_le_bytes(),
    ));
    assert!(memory.initialize_ram(GuestPhysicalPageId::new(1), 4, &breakpoint(0).to_le_bytes(),));
    assert!(memory.map_page(
        SPACE,
        GuestVirtualAddress::new(CODE),
        GuestPhysicalPageId::new(1),
        MemoryPermissions::READ_EXECUTE,
    ));
    assert!(memory.map_page(
        SPACE,
        GuestVirtualAddress::new(DATA),
        GuestPhysicalPageId::new(2),
        MemoryPermissions::READ_WRITE,
    ));
    assert!(memory.map_page(
        SPACE,
        GuestVirtualAddress::new(DATA + 0x1000),
        GuestPhysicalPageId::new(3),
        MemoryPermissions::READ,
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
    let address = GuestVirtualAddress::new(DATA + 0xffc);
    let mut state = memory_state();
    write_register(&mut state, 0, 0xa5c3_9678_1234_fedc);
    write_register(&mut state, 1, address.get());

    let DirectExit::DataFault { fault, .. } = thread.run(&process, &memory, &mut state).unwrap()
    else {
        panic!("cross-page store did not report the second-page permission fault");
    };
    assert_eq!(fault.reason, DataAccessFaultReason::WritePermissionDenied);
    assert_eq!(
        memory
            .read(SPACE, address, MemoryAccess::normal(MemoryAccessSize::Word))
            .unwrap()
            .value,
        MemoryValue::U32(0),
        "the writable first-page fragment must remain untouched",
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
        progress,
    } = JitThread::new()
        .run(
            &JitProcess::new(cpu()).unwrap(),
            &actual_memory,
            &mut actual,
        )
        .unwrap()
    else {
        panic!("direct JIT did not report a data fault")
    };
    assert_eq!(pc, source.pc);
    assert_eq!(fault, expected_fault);
    assert_eq!(progress, 1);
    assert_eq!(actual.pc(), CODE);
    assert_eq!(actual, expected);
}

#[test]
fn linux_direct_confinement_preserves_the_unconfined_guest_address() {
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
    let DirectExit::DataFault { fault, .. } = thread.run(&process, &memory, &mut state).unwrap()
    else {
        panic!("unconfined direct read did not produce a guest data fault");
    };
    assert_eq!(fault.address, GuestVirtualAddress::new(u64::MAX));
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
                    &mut actual
                )
                .unwrap(),
            DirectExit::DataFault { .. }
        ));
        assert_eq!(actual, expected, "{encoding:#010x}");
        assert_memory_prefix_equal(&actual_memory, &expected_memory, 4096);
    }
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
        .run(&JitProcess::new(cpu()).unwrap(), &memory, &mut state)
        .unwrap();
    assert!(matches!(
        exit,
        DirectExit::Architectural { progress: 1, .. }
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
fn linux_direct_raw_mmio_fault_does_not_invoke_the_device_handler() {
    let events = Arc::new(Mutex::new(Vec::new()));
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

    let report = thread
        .run_slice(
            &process,
            RunRequest {
                cpu: cpu(),
                memory: &memory,
                memory_lease: Some(memory.acquire_execution_lease()),
                state: &mut state,
                instruction_budget: 1,
                loader_return: None,
                timer: &ZeroTimer,
                events: VcpuEventState::default(),
            },
        )
        .unwrap();
    let CpuExit::DataFault { ref fault, .. } = report.stop else {
        panic!("raw direct MMIO access did not produce a data fault");
    };
    assert!(matches!(
        fault.reason,
        nixe_cpu::memory::DataAccessFaultReason::Device(_)
    ));
    assert!(events.lock().unwrap().is_empty());
    assert!(report.context.is_none());
    assert!(report.to_string().contains("registers=[unavailable]"));
}

#[test]
fn scalar_and_pair_exclusive_forms_match_the_interpreter() {
    let mut encodings = Vec::new();
    for size in 0..=3 {
        for acquire in [false, true] {
            for release in [false, true] {
                encodings.push(load_exclusive(size, acquire));
                encodings.push(store_exclusive(size, release));
            }
        }
    }
    for size in 2..=3 {
        for acquire in [false, true] {
            for release in [false, true] {
                encodings.push(load_exclusive_pair(size, acquire));
                encodings.push(store_exclusive_pair(size, release));
            }
        }
    }
    let mut words: Vec<_> = encodings
        .iter()
        .enumerate()
        .map(|(index, encoding)| ((index * 4) as u64, *encoding))
        .collect();
    words.push(((encodings.len() * 4) as u64, breakpoint(0)));
    let expected_memory = memory_with_words_and_data(&words);
    let actual_memory = memory_with_words_and_data(&words);
    let mut expected = memory_state();
    let monitor = RefCell::new(nixe_cpu::exclusive::ExclusiveMonitorState::default());
    let events = nixe_cpu::execution::VcpuEventState::default();
    let context = InterpreterContext::new(cpu(), &expected_memory, &monitor, &ZeroTimer, &events);
    for encoding in encodings {
        assert_eq!(
            execute_one_with_context(context, &mut expected, encoding).unwrap(),
            InstructionStep::Continue,
            "{encoding:#010x}"
        );
    }
    let mut actual = memory_state();
    JitThread::new()
        .run(
            &JitProcess::new(cpu()).unwrap(),
            &actual_memory,
            &mut actual,
        )
        .unwrap();
    assert_eq!(actual, expected);
    assert_eq!(read_register(&actual, 2), 0);
    assert_memory_prefix_equal(&actual_memory, &expected_memory, 16);
}

#[test]
fn clear_exclusive_forces_the_next_store_exclusive_to_fail() {
    let words = [
        (0, load_exclusive(2, true)),
        (4, 0xd503_3f5f), // CLREX
        (8, store_exclusive(2, true)),
        (12, breakpoint(0)),
    ];
    let expected_memory = memory_with_words_and_data(&words);
    let actual_memory = memory_with_words_and_data(&words);
    let mut expected = memory_state();
    let monitor = RefCell::new(nixe_cpu::exclusive::ExclusiveMonitorState::default());
    let events = nixe_cpu::execution::VcpuEventState::default();
    let context = InterpreterContext::new(cpu(), &expected_memory, &monitor, &ZeroTimer, &events);
    for encoding in [words[0].1, words[1].1, words[2].1] {
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
        )
        .unwrap();
    assert_eq!(actual, expected);
    assert_eq!(read_register(&actual, 2), 1);
    assert_memory_prefix_equal(&actual_memory, &expected_memory, 16);
}

#[test]
fn pair_exclusive_reservation_uses_physical_alias_identity() {
    let words = [
        (0, load_exclusive_pair(3, true)),
        (4, 0x9140_0421), // ADD X1,X1,#1,LSL #12
        (8, store_exclusive_pair(3, true)),
        (12, breakpoint(0)),
    ];
    let mut expected_memory = memory_with_words_and_data(&words);
    let mut actual_memory = memory_with_words_and_data(&words);
    for memory in [&mut expected_memory, &mut actual_memory] {
        assert!(memory.map_page(
            SPACE,
            GuestVirtualAddress::new(DATA + DIRECT_PAGE_SIZE as u64),
            GuestPhysicalPageId::new(2),
            MemoryPermissions::READ_WRITE,
        ));
    }

    let mut expected = memory_state();
    let monitor = RefCell::new(nixe_cpu::exclusive::ExclusiveMonitorState::default());
    let events = nixe_cpu::execution::VcpuEventState::default();
    let context = InterpreterContext::new(cpu(), &expected_memory, &monitor, &ZeroTimer, &events);
    for encoding in [words[0].1, words[1].1, words[2].1] {
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
        )
        .unwrap();
    assert_eq!(actual, expected);
    assert_eq!(read_register(&actual, 2), 0);
    assert_memory_prefix_equal(&actual_memory, &expected_memory, 16);
}

#[test]
fn linux_direct_lse_cas_and_casp_forms_match_the_interpreter() {
    let mut encodings = Vec::new();
    for size in 0..=3 {
        for acquire in [false, true] {
            for release in [false, true] {
                for operation in 0..=8 {
                    encodings.push(lse_rmw(size, acquire, release, operation));
                }
                encodings.push(compare_and_swap(size, acquire, release));
            }
        }
    }
    for size in 0..=1 {
        for acquire in [false, true] {
            for release in [false, true] {
                encodings.push(compare_and_swap_pair(size, acquire, release));
            }
        }
    }

    let mut words: Vec<_> = encodings
        .iter()
        .enumerate()
        .map(|(index, encoding)| ((index * 4) as u64, *encoding))
        .collect();
    words.push(((encodings.len() * 4) as u64, breakpoint(0)));
    let expected_memory = memory_with_words_and_data(&words);
    let mut actual_memory = execution_memory_with_words_and_data(&words);
    actual_memory
        .bind_cpu_memory_backend(SPACE, 0x1_0000, DirectBackendPolicy::Required)
        .unwrap();

    let mut expected = memory_state();
    let monitor = RefCell::new(nixe_cpu::exclusive::ExclusiveMonitorState::default());
    let events = nixe_cpu::execution::VcpuEventState::default();
    let context = InterpreterContext::new(cpu(), &expected_memory, &monitor, &ZeroTimer, &events);
    for encoding in encodings {
        assert_eq!(
            execute_one_with_context(context, &mut expected, encoding).unwrap(),
            InstructionStep::Continue,
            "{encoding:#010x}"
        );
    }

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
    let mut actual = memory_state();
    assert!(matches!(
        thread.run(&process, &actual_memory, &mut actual).unwrap(),
        DirectExit::Architectural { .. }
    ));

    assert_eq!(actual, expected);
    let mut expected_bytes = [0; 32];
    let mut actual_bytes = [0; 32];
    expected_memory
        .read_bytes(SPACE, GuestVirtualAddress::new(DATA), &mut expected_bytes)
        .unwrap();
    actual_memory
        .read_bytes(SPACE, GuestVirtualAddress::new(DATA), &mut actual_bytes)
        .unwrap();
    assert_eq!(actual_bytes, expected_bytes);
}

#[test]
fn linux_direct_cas_and_casp_successfully_replace_every_width() {
    let cases = [
        (compare_and_swap(0, true, true), 0, 0),
        (compare_and_swap(1, true, true), 0x0100, 0),
        (compare_and_swap(2, true, true), 0x0302_0100, 0),
        (compare_and_swap(3, true, true), 0x0706_0504_0302_0100, 0),
        (
            compare_and_swap_pair(0, true, true),
            0x0302_0100,
            0x0706_0504,
        ),
        (
            compare_and_swap_pair(1, true, true),
            0x0706_0504_0302_0100,
            0x0f0e_0d0c_0b0a_0908,
        ),
    ];
    for (encoding, expected_low, expected_high) in cases {
        let expected_memory = memory_with_data(encoding);
        let mut actual_memory = execution_memory_with_data(encoding);
        actual_memory
            .bind_cpu_memory_backend(SPACE, 0x1_0000, DirectBackendPolicy::Required)
            .unwrap();
        let mut initial = memory_state();
        write_register(&mut initial, 2, expected_low);
        write_register(&mut initial, 3, expected_high);
        write_register(&mut initial, 4, 0x1122_3344_5566_7788);
        write_register(&mut initial, 5, 0x99aa_bbcc_ddee_ff00);

        let mut expected = initial.clone();
        let monitor = RefCell::new(nixe_cpu::exclusive::ExclusiveMonitorState::default());
        let events = nixe_cpu::execution::VcpuEventState::default();
        let context =
            InterpreterContext::new(cpu(), &expected_memory, &monitor, &ZeroTimer, &events);
        assert_eq!(
            execute_one_with_context(context, &mut expected, encoding).unwrap(),
            InstructionStep::Continue
        );

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
        thread.run(&process, &actual_memory, &mut actual).unwrap();
        assert_eq!(actual, expected, "{encoding:#010x}");

        let mut expected_bytes = [0; 16];
        let mut actual_bytes = [0; 16];
        expected_memory
            .read_bytes(SPACE, GuestVirtualAddress::new(DATA), &mut expected_bytes)
            .unwrap();
        actual_memory
            .read_bytes(SPACE, GuestVirtualAddress::new(DATA), &mut actual_bytes)
            .unwrap();
        assert_eq!(actual_bytes, expected_bytes, "{encoding:#010x}");
    }
}

#[test]
fn linux_direct_atomic_fault_retry_commits_once() {
    let encoding = lse_rmw(2, true, true, 0); // LDADDAL W2,W3,[X1]
    let mut memory = execution_memory_with_data(encoding);
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
    write_register(&mut state, 2, 1);

    let initial = u32::from_le_bytes([0, 1, 2, 3]);
    for increment in 1..=2 {
        state.set_pc(CODE);
        thread.run(&process, &memory, &mut state).unwrap();
        assert_eq!(
            memory
                .read(
                    SPACE,
                    GuestVirtualAddress::new(DATA),
                    MemoryAccess::normal(MemoryAccessSize::Word),
                )
                .unwrap()
                .value,
            MemoryValue::U32(initial + increment),
        );
    }
}

#[test]
fn linux_direct_lse_updates_are_atomic_through_physical_aliases() {
    const VCPUS: usize = 4;
    const ITERATIONS: usize = 100;
    let encoding = lse_rmw(2, true, true, 0); // LDADDAL W2,W3,[X1]
    let mut memory = execution_memory_with_data(encoding);
    assert!(memory.map_page(
        SPACE,
        GuestVirtualAddress::new(DATA + DIRECT_PAGE_SIZE as u64),
        GuestPhysicalPageId::new(2),
        MemoryPermissions::READ_WRITE,
    ));
    memory
        .bind_cpu_memory_backend(SPACE, 0x1_0000, DirectBackendPolicy::Required)
        .unwrap();
    let memory = Arc::new(memory);
    let process = Arc::new(JitProcess::new(cpu()).unwrap());
    let binding = MemoryBinding {
        address_space: SPACE,
        end_exclusive: GuestVirtualAddress::new(0x1_0000),
        memory: memory.as_ref(),
        mapping_epoch: memory.mapping_epoch().get(),
        invalidation_cursor: memory.invalidation_cursor(),
    };
    process.bind_memory(binding).unwrap();

    let mut workers = Vec::new();
    for vcpu in 0..VCPUS {
        let memory = Arc::clone(&memory);
        let process = Arc::clone(&process);
        workers.push(std::thread::spawn(move || {
            let binding = MemoryBinding {
                address_space: SPACE,
                end_exclusive: GuestVirtualAddress::new(0x1_0000),
                memory: memory.as_ref(),
                mapping_epoch: memory.mapping_epoch().get(),
                invalidation_cursor: memory.invalidation_cursor(),
            };
            let mut thread = JitThread::new();
            thread
                .synchronize_address_space(process.as_ref(), binding)
                .unwrap();
            let mut state = memory_state();
            write_register(
                &mut state,
                1,
                DATA + (vcpu % 2) as u64 * DIRECT_PAGE_SIZE as u64,
            );
            write_register(&mut state, 2, 1);
            for _ in 0..ITERATIONS {
                state.set_pc(CODE);
                thread
                    .run(process.as_ref(), memory.as_ref(), &mut state)
                    .unwrap();
            }
        }));
    }
    for worker in workers {
        worker.join().unwrap();
    }

    let initial = u32::from_le_bytes([0, 1, 2, 3]);
    for address in [DATA, DATA + DIRECT_PAGE_SIZE as u64] {
        assert_eq!(
            memory
                .read(
                    SPACE,
                    GuestVirtualAddress::new(address),
                    MemoryAccess::normal(MemoryAccessSize::Word),
                )
                .unwrap()
                .value,
            MemoryValue::U32(initial + (VCPUS * ITERATIONS) as u32),
        );
    }
}

#[test]
fn linux_direct_atomic_access_rejects_misalignment_before_native_memory() {
    let encoding = compare_and_swap_pair(1, true, true); // CASPAL X2,X3,X4,X5,[X1]
    let mut memory = execution_memory_with_data(encoding);
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
    write_register(&mut state, 1, DATA + 8);

    let DirectExit::DataFault { fault, .. } = thread.run(&process, &memory, &mut state).unwrap()
    else {
        panic!("misaligned CASP did not report a data fault");
    };
    assert_eq!(
        fault.reason,
        DataAccessFaultReason::Misaligned {
            required_alignment: 16,
        }
    );
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
                let exit = thread.run(&process, memory.as_ref(), &mut state).unwrap();
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
fn pair_exclusive_updates_are_indivisible_across_vcpus() {
    const VCPUS: usize = 4;
    const ITERATIONS: usize = 100;
    let cpu = cpu();
    let memory = Arc::new(memory_with_words_and_data(&[
        (0x00, load_exclusive_pair(3, true)),
        (0x04, 0x9100_0400), // ADD X0,X0,#1
        (0x08, store_exclusive_pair(3, true)),
        (0x0c, compare_branch(CODE + 0x0c, CODE, 2, true)),
        (0x10, breakpoint(0)),
    ]));
    let process = Arc::new(JitProcess::new(cpu).unwrap());
    let mut workers = Vec::new();
    for _ in 0..VCPUS {
        let memory = Arc::clone(&memory);
        let process = Arc::clone(&process);
        workers.push(std::thread::spawn(move || {
            let thread = JitThread::new();
            let mut state = memory_state();
            for _ in 0..ITERATIONS {
                state.set_pc(CODE);
                let exit = thread.run(&process, memory.as_ref(), &mut state).unwrap();
                assert!(matches!(exit, DirectExit::Architectural { .. }));
            }
        }));
    }
    for worker in workers {
        worker.join().unwrap();
    }

    let mut bytes = [0; 16];
    memory
        .read_bytes(SPACE, GuestVirtualAddress::new(DATA), &mut bytes)
        .unwrap();
    let low = u64::from_le_bytes(bytes[..8].try_into().unwrap());
    let high = u64::from_le_bytes(bytes[8..].try_into().unwrap());
    assert_eq!(low, 0x0706_0504_0302_0100 + (VCPUS * ITERATIONS) as u64);
    assert_eq!(high, 0x0f0e_0d0c_0b0a_0908);
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
        .run(&JitProcess::new(cpu()).unwrap(), &memory, &mut state)
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
                loader_return: None,
                timer: &FixedTimer,
                events: &actual_events,
            },
        )
        .unwrap();
    assert!(matches!(
        exit,
        DirectExit::Architectural { progress: 1, .. }
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
        )
        .unwrap();
    assert_eq!(
        exit,
        DirectExit::Scheduled {
            pc: GuestVirtualAddress::new(CODE),
            request: nixe_cpu::execution::SchedulerRequest::Yield,
            progress: 1,
        }
    );
    assert_eq!(yield_state.pc(), CODE + 4);

    let wfe_memory = memory(&[(0, 0xd503_205f), (4, breakpoint(0))]);
    let process = JitProcess::new(cpu()).unwrap();
    let thread = JitThread::new();
    let mut waiting = state(CODE, 0);
    assert_eq!(
        thread.run(&process, &wfe_memory, &mut waiting).unwrap(),
        DirectExit::Scheduled {
            pc: GuestVirtualAddress::new(CODE),
            request: nixe_cpu::execution::SchedulerRequest::WaitForEvent,
            progress: 1,
        }
    );
    thread.events.signal_event();
    let mut resumed = state(CODE, 0);
    assert!(matches!(
        thread.run(&process, &wfe_memory, &mut resumed).unwrap(),
        DirectExit::Architectural { progress: 1, .. }
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
        )
        .unwrap_or_else(|error| panic!("{encoding:#010x}: {error}"));
    assert!(matches!(
        exit,
        DirectExit::Architectural { progress: 1, .. }
    ));
    assert_eq!(actual_state, expected_state, "{encoding:#010x}");
    let mut expected_bytes = [0_u8; 64];
    let mut actual_bytes = [0_u8; 64];
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

fn execution_memory_with_words_and_data(words: &[(u64, u32)]) -> ExecutionMemory {
    let mut memory = ExecutionMemory::new();
    let code_page = GuestPhysicalPageId::new(1);
    assert!(memory.add_ram_page(code_page));
    for &(offset, encoding) in words {
        assert!(memory.initialize_ram(code_page, offset as usize, &encoding.to_le_bytes()));
    }
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
    write_register(&mut initial, 0, CODE + 8);
    write_register(&mut initial, 30, CODE + 8);
    for encoding in [
        0xd503_201f, // NOP
        0x1400_0002, // B +8
        0x9400_0002, // BL +8
        0xd61f_0000, // BR X0
        0xd63f_0000, // BLR X0
        0xd65f_03c0, // RET X30
        0x5400_0040, // B.EQ +8
        0xb500_0040, // CBNZ X0,+8
        0xb628_0040, // TBZ X0,#37,+8
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
            .run(&JitProcess::new(cpu()).unwrap(), &memory, &mut actual)
            .unwrap(),
        DirectExit::Architectural {
            pc: GuestVirtualAddress::new(CODE + 20),
            detail: 2 << 24,
            progress: 1,
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
            thread.run(&process, &memory, &mut actual).unwrap();
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
        thread.run(&process, &memory, &mut state).unwrap();
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
    thread.run(&process, &memory, &mut side_entry).unwrap();
    assert_eq!(side_entry, expected);
    assert_eq!(process.state.lock().unwrap().regions.len(), 2);
}

#[test]
fn straight_line_nops_add_no_per_instruction_scaffolding() {
    let compile = |nop_count: usize| {
        let mut words: Vec<_> = (0..nop_count)
            .map(|index| ((index * 4) as u64, 0xd503_201f))
            .collect();
        words.push(((nop_count * 4) as u64, breakpoint(0)));
        let memory = memory(&words);
        let process = JitProcess::new(cpu()).unwrap();
        process.entry_for(&memory, location(CODE)).unwrap();
        let state = process.state.lock().unwrap();
        let region = state
            .region_for(RegionKey::new(cpu(), location(CODE)))
            .unwrap();
        (region.clif_instructions, region.native_bytes)
    };
    let short = compile(1);
    let long = compile(64);
    assert!(long.0 <= short.0 + 2, "short={short:?} long={long:?}");
    assert!(long.1 <= short.1 + 16, "short={short:?} long={long:?}");
}

#[test]
fn direct_fault_metadata_has_no_trap_code_site_cap() {
    const LOAD_COUNT: usize = 251;

    let mut words: Vec<_> = (0..LOAD_COUNT)
        .map(|index| ((index * 4) as u64, 0xf940_0020)) // LDR X0,[X1]
        .collect();
    words.push(((LOAD_COUNT * 4) as u64, breakpoint(0)));
    let memory = memory(&words);
    let region = discover_region(
        cpu(),
        &memory,
        location(CODE),
        RegionLimits::default(),
        |_| false,
    )
    .unwrap();
    let mut compiler = super::compiler::DirectCompiler::new().unwrap();
    compiler
        .bind_memory_backend(nixe_memory::CpuMemoryBackend::LinuxDirect)
        .unwrap();
    let compiled = compiler.compile(&region, &[]).unwrap();

    assert_eq!(compiled.fault_sites.len(), LOAD_COUNT);
}

#[test]
fn branch_local_read_is_loaded_after_the_primary_branch() {
    let memory = memory(&[
        (0x00, compare_branch(CODE, CODE + 0x0c, 0, false)), // CBZ W0,skip
        (0x04, 0xaa1f_0062),                                 // ORR X2,X3,XZR
        (0x08, branch(CODE + 0x08, CODE + 0x10)),
        (0x0c, 0xd503_201f),
        (0x10, breakpoint(0)),
    ]);
    let process = JitProcess::new(cpu()).unwrap();
    process.entry_for(&memory, location(CODE)).unwrap();
    let state = process.state.lock().unwrap();
    let region = state
        .region_for(RegionKey::new(cpu(), location(CODE)))
        .unwrap();
    assert_eq!(region.deferred_register_loads, 1);
}

#[test]
fn loop_carried_read_modify_write_is_loaded_before_the_loop() {
    let memory = memory(&[
        (0x00, branch(CODE, CODE + 4)),
        (0x04, add_x0(1)),
        (0x08, 0xf100_081f),                                // CMP X0,#2
        (0x0c, conditional_branch(CODE + 12, CODE + 4, 1)), // B.NE loop
        (0x10, breakpoint(0)),
    ]);
    let process = JitProcess::new(cpu()).unwrap();
    let thread = JitThread::new();
    let mut state = state(CODE, 0);

    assert!(matches!(
        thread.run(&process, &memory, &mut state).unwrap(),
        DirectExit::Architectural { .. }
    ));
    assert_eq!(read_x0(&state), 2);
    let compiled = process.state.lock().unwrap();
    let region = compiled
        .region_for(RegionKey::new(cpu(), location(CODE)))
        .unwrap();
    assert_eq!(region.deferred_register_loads, 0);
}

#[test]
fn branch_join_preserves_a_register_not_written_on_the_taken_path() {
    let memory = memory(&[
        (0x00, compare_branch(CODE, CODE + 0x0c, 0, false)), // CBZ W0,skip
        (0x04, 0xd280_0022),                                 // MOVZ X2,#1
        (0x08, branch(CODE + 0x08, CODE + 0x10)),
        (0x0c, 0xd503_201f), // skip: NOP
        (0x10, breakpoint(0)),
    ]);
    let process = JitProcess::new(cpu()).unwrap();
    let thread = JitThread::new();

    let mut skipped = state(CODE, 0);
    write_register(&mut skipped, 2, 0x1234_5678_9abc_def0);
    thread.run(&process, &memory, &mut skipped).unwrap();
    assert_eq!(read_register(&skipped, 2), 0x1234_5678_9abc_def0);

    let mut written = state(CODE, 1);
    write_register(&mut written, 2, 0x1234_5678_9abc_def0);
    thread.run(&process, &memory, &mut written).unwrap();
    assert_eq!(read_register(&written, 2), 1);
}

#[test]
fn equivalent_exits_share_one_cold_epilogue() {
    let memory = memory(&[
        (0x00, compare_branch(CODE, CODE + 0x08, 0, false)),
        (0x04, breakpoint(7)),
        (0x08, breakpoint(7)),
    ]);
    let process = JitProcess::new(cpu()).unwrap();
    process.entry_for(&memory, location(CODE)).unwrap();
    let state = process.state.lock().unwrap();
    let region = state
        .region_for(RegionKey::new(cpu(), location(CODE)))
        .unwrap();
    // Reconciliation, control, and the shared architectural exit.
    assert_eq!(region.exit_tail_count, 3);
}

#[test]
fn switch_1_pointer_authentication_hints_add_no_clif_over_nop() {
    let shape = |encoding| {
        let memory = memory(&[(0, encoding), (4, breakpoint(0))]);
        let process = JitProcess::new(cpu()).unwrap();
        process.entry_for(&memory, location(CODE)).unwrap();
        let state = process.state.lock().unwrap();
        let region = state
            .region_for(RegionKey::new(cpu(), location(CODE)))
            .unwrap();
        (region.clif_instructions, region.native_bytes)
    };
    let nop = shape(0xd503_201f);

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
        assert_eq!(shape(encoding), nop, "{encoding:#010x}");
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
        .run(&process, &memory, &mut actual)
        .unwrap_or_else(|error| panic!("{encoding:#010x}: {error}"));
    assert_eq!(actual, expected, "{encoding:#010x}");
}

fn assert_fp_exception_matches_interpreter(encoding: u32, initial: A64State) {
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
            )
            .unwrap(),
        DirectExit::Architectural {
            pc: GuestVirtualAddress::new(CODE),
            detail: 6 << 24,
            progress: 1,
        },
        "{encoding:#010x}"
    );
    assert_eq!(actual, expected, "{encoding:#010x}");
}

fn assert_control_matches_interpreter(encoding: u32, initial: A64State) {
    let mut expected = initial.clone();
    assert_eq!(
        execute_one(&cpu().platform(), &mut expected, encoding).unwrap(),
        InstructionStep::Continue,
        "{encoding:#010x}"
    );
    let memory = memory(&[(0, encoding), (4, breakpoint(0)), (8, breakpoint(0))]);
    let process = JitProcess::new(cpu()).unwrap();
    let mut actual = initial;
    JitThread::new()
        .run(&process, &memory, &mut actual)
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

fn lse_rmw(size: u8, acquire: bool, release: bool, operation: u8) -> u32 {
    0x3820_0000
        | (u32::from(size) << 30)
        | (u32::from(acquire) << 23)
        | (u32::from(release) << 22)
        | (u32::from(2_u8) << 16)
        | (u32::from(operation) << 12)
        | (u32::from(1_u8) << 5)
        | u32::from(3_u8)
}

fn compare_and_swap(size: u8, acquire: bool, release: bool) -> u32 {
    0x08a0_7c00
        | (u32::from(size) << 30)
        | (u32::from(acquire) << 22)
        | (u32::from(release) << 15)
        | (u32::from(2_u8) << 16)
        | (u32::from(1_u8) << 5)
        | u32::from(3_u8)
}

fn compare_and_swap_pair(size: u8, acquire: bool, release: bool) -> u32 {
    0x0820_7c00
        | (u32::from(size) << 30)
        | (u32::from(acquire) << 22)
        | (u32::from(release) << 15)
        | (u32::from(2_u8) << 16)
        | (u32::from(1_u8) << 5)
        | u32::from(4_u8)
}

fn load_exclusive(size: u8, acquire: bool) -> u32 {
    0x085f_7c00 | (u32::from(size) << 30) | (u32::from(acquire) << 15) | (u32::from(1_u8) << 5)
}

fn store_exclusive(size: u8, release: bool) -> u32 {
    0x0800_7c00
        | (u32::from(size) << 30)
        | (u32::from(release) << 15)
        | (u32::from(2_u8) << 16)
        | (u32::from(1_u8) << 5)
}

fn load_exclusive_pair(size: u8, acquire: bool) -> u32 {
    0x087f_0000
        | (u32::from(size) << 30)
        | (u32::from(acquire) << 15)
        | (u32::from(3_u8) << 10)
        | (u32::from(1_u8) << 5)
}

fn store_exclusive_pair(size: u8, release: bool) -> u32 {
    0x0820_0000
        | (u32::from(size) << 30)
        | (u32::from(release) << 15)
        | (u32::from(2_u8) << 16)
        | (u32::from(3_u8) << 10)
        | (u32::from(1_u8) << 5)
}

fn breakpoint(immediate: u16) -> u32 {
    0xd420_0000 | (u32::from(immediate) << 5)
}
