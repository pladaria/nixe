//! Code invalidation and mapping lifetime on the production physical arena.
use nixe_cpu::{
    execution::{ArchitecturalTimer, ControlRequest, CpuExit, TimerSnapshot, VcpuEventState},
    platform::TargetPlatform,
    profile::ProcessCpuContext,
    state::a64::{A64GeneralRegister, A64Register},
};
use nixe_cpu_jit::{JitProcess, JitRunRequest, JitThread, ThreadState};
use nixe_memory::{
    Access, AddressSpace, AddressWidth, AllocationRange, GuestVirtualAddress, Machine,
    MemoryPermissions, PAGE_SIZE,
};
use std::sync::Arc;

const CODE: GuestVirtualAddress = GuestVirtualAddress::new(0x1000);
const DATA: GuestVirtualAddress = GuestVirtualAddress::new(0x2000);
const ALIAS: GuestVirtualAddress = GuestVirtualAddress::new(0x4000);
struct Timer;
impl ArchitecturalTimer for Timer {
    fn snapshot(&self) -> TimerSnapshot {
        TimerSnapshot {
            counter: 0,
            frequency: 19_200_000,
        }
    }
}
fn fixture(words: &[u32]) -> (Arc<AddressSpace>, Arc<JitProcess>) {
    let machine = Machine::new(4 * PAGE_SIZE).unwrap();
    let account = machine.account(4 * PAGE_SIZE).unwrap();
    let code = account.allocate(PAGE_SIZE).unwrap();
    let data = account.allocate(PAGE_SIZE).unwrap();
    let bytes: Vec<_> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    machine
        .write_resolved(
            &[AllocationRange::new(code.pin(), 0, bytes.len() as u64).unwrap()],
            &bytes,
        )
        .unwrap();
    let space = AddressSpace::new(machine, AddressWidth::Bits32).unwrap();
    for (address, allocation, permissions) in [
        (CODE, code.clone(), MemoryPermissions::READ_EXECUTE),
        (ALIAS, code, MemoryPermissions::READ_WRITE),
        (DATA, data, MemoryPermissions::READ_WRITE),
    ] {
        space
            .map(address, allocation, 0, PAGE_SIZE, permissions)
            .unwrap();
    }
    let process = JitProcess::new(
        ProcessCpuContext::for_platform(TargetPlatform::Switch1, space.identity()),
        space.clone(),
        19_200_000,
    )
    .unwrap();
    (space, process)
}
fn x(index: u8) -> A64Register {
    A64Register::General(A64GeneralRegister::new(index).unwrap())
}
fn run(jit: &mut JitThread, state: &mut ThreadState) -> nixe_cpu::execution::ExecutionReport {
    jit.run_slice(
        &mut nixe_cpu_direct_memory::NativeWorker::default(),
        JitRunRequest {
            state,
            instruction_budget: 1000,
            timer: &Timer,
            events: &VcpuEventState::default(),
            capture_context: true,
        },
    )
    .unwrap()
}
fn value_at(jit: &mut JitThread, address: GuestVirtualAddress) -> u64 {
    let mut state = ThreadState::default();
    state.set_pc(address.get());
    for _ in 0..16 {
        match run(jit, &mut state).stop {
            CpuExit::SupervisorCall { .. } => return state.read_x(x(0)),
            CpuExit::Safepoint => {}
            stop => panic!("unexpected code execution boundary: {stop:?}"),
        }
    }
    panic!("code invalidation failed to make progress")
}
fn write(space: &AddressSpace, address: GuestVirtualAddress, bytes: &[u8]) {
    assert!(matches!(
        space.write(address, bytes).unwrap(),
        Access::Ready(())
    ));
}

#[test]
fn host_code_overwrite_through_a_nonexecutable_alias_invalidates_native_blocks() {
    let (space, process) = fixture(&[0xd2800540, 0xd4000001]); // MOV X0,#42; SVC
    let mut jit = process.create_thread().unwrap();
    assert_eq!(value_at(&mut jit, CODE), 42);
    write(&space, ALIAS, &0xd2800560_u32.to_le_bytes());
    assert_eq!(value_at(&mut jit, CODE), 43);
}

#[test]
fn preemption_cannot_discard_a_pending_code_invalidation() {
    let (space, process) = fixture(&[0xd2800540, 0xd4000001]);
    let mut jit = process.create_thread().unwrap();
    assert_eq!(value_at(&mut jit, CODE), 42);
    write(&space, ALIAS, &0xd2800560_u32.to_le_bytes());
    jit.control().request(ControlRequest::Preempt);
    let mut state = ThreadState::default();
    state.set_pc(CODE.get());
    assert_eq!(run(&mut jit, &mut state).stop, CpuExit::Safepoint);
    assert!(matches!(
        run(&mut jit, &mut state).stop,
        CpuExit::SupervisorCall { .. }
    ));
    assert_eq!(state.read_x(x(0)), 43);
}

#[test]
fn remapped_code_and_surviving_physical_aliases_reach_both_idle_cores() {
    let (space, process) = fixture(&[0xd2800540, 0xd4000001]);
    let mut cores = [
        process.create_thread().unwrap(),
        process.create_thread().unwrap(),
    ];
    let old_alias = GuestVirtualAddress::new(0x6000);
    space
        .map_alias(
            CODE,
            old_alias,
            PAGE_SIZE,
            MemoryPermissions::READ_EXECUTE,
            MemoryPermissions::READ_EXECUTE,
        )
        .unwrap();
    let replacement = space
        .machine()
        .account(PAGE_SIZE)
        .unwrap()
        .allocate(PAGE_SIZE)
        .unwrap();
    let bytes = [0xd2800560_u32.to_le_bytes(), 0xd4000001_u32.to_le_bytes()].concat();
    space
        .machine()
        .write_resolved(
            &[AllocationRange::new(replacement.pin(), 0, bytes.len() as u64).unwrap()],
            &bytes,
        )
        .unwrap();
    for core in &mut cores {
        assert_eq!(value_at(core, CODE), 42);
        assert_eq!(value_at(core, old_alias), 42);
    }
    write(&space, ALIAS, &0xd2800580_u32.to_le_bytes());
    space.unmap(CODE, PAGE_SIZE).unwrap();
    space
        .map(
            CODE,
            replacement.clone(),
            0,
            PAGE_SIZE,
            MemoryPermissions::READ_EXECUTE,
        )
        .unwrap();
    space
        .machine()
        .write_resolved(
            &[AllocationRange::new(replacement.pin(), 0, 4).unwrap()],
            &0xd28005a0_u32.to_le_bytes(),
        )
        .unwrap();
    for core in &mut cores {
        assert_eq!(value_at(core, CODE), 45);
        assert_eq!(value_at(core, old_alias), 44);
    }
    assert!(process.try_shutdown().unwrap());
}

#[test]
fn host_alias_writes_reach_concurrent_native_cores() {
    let (space, process) = fixture(&[0xd2800540, 0xd4000001]);
    let (results, completed) = std::sync::mpsc::channel();
    let mut commands = Vec::new();
    let mut workers = Vec::new();
    for _ in 0..2 {
        let mut jit = process.create_thread().unwrap();
        let (send, receive) = std::sync::mpsc::channel();
        commands.push(send);
        let results = results.clone();
        workers.push(std::thread::spawn(move || {
            for expected in receive {
                assert_eq!(value_at(&mut jit, CODE), expected);
                results.send(()).unwrap();
            }
        }));
    }
    drop(results);
    for value in 42..50u32 {
        write(
            &space,
            ALIAS,
            &(0xd2800000_u32 | (value << 5)).to_le_bytes(),
        );
        for command in &commands {
            command.send(u64::from(value)).unwrap();
        }
        for _ in 0..2 {
            completed
                .recv_timeout(std::time::Duration::from_secs(5))
                .unwrap();
        }
    }
    drop(commands);
    for worker in workers {
        worker.join().unwrap();
    }
    assert!(process.try_shutdown().unwrap());
}

#[test]
fn execute_permission_loss_retires_blocks_and_restoration_refetches() {
    use nixe_cpu::error::InstructionFetchFaultReason;
    let (space, process) = fixture(&[0xd2800540, 0xd4000001]);
    let mut jit = process.create_thread().unwrap();
    assert_eq!(value_at(&mut jit, CODE), 42);
    space
        .protect(CODE, PAGE_SIZE, MemoryPermissions::READ)
        .unwrap();
    let mut state = ThreadState::default();
    state.set_pc(CODE.get());
    assert!(
        matches!(run(&mut jit, &mut state).stop, CpuExit::FetchFault { fault } if fault.reason == InstructionFetchFaultReason::ExecutePermissionDenied)
    );
    assert_eq!(state.pc(), CODE.get());
    assert_eq!(state.read_x(x(0)), 0);
    write(&space, ALIAS, &0xd2800560_u32.to_le_bytes());
    space
        .protect(CODE, PAGE_SIZE, MemoryPermissions::EXECUTE)
        .unwrap();
    state.leave_core();
    assert_eq!(value_at(&mut jit, CODE), 43);
}

#[test]
fn speculative_fetch_at_execute_only_page_boundary_keeps_precise_faults() {
    use nixe_cpu::error::InstructionFetchFaultReason;
    let (space, process) = fixture(&[0xd503201f]);
    write(
        &space,
        GuestVirtualAddress::new(ALIAS.get() + PAGE_SIZE - 4),
        &0xd4000001_u32.to_le_bytes(),
    );
    space
        .protect(CODE, PAGE_SIZE, MemoryPermissions::EXECUTE)
        .unwrap();
    space.unmap(DATA, PAGE_SIZE).unwrap();
    let mut jit = process.create_thread().unwrap();
    let end = GuestVirtualAddress::new(CODE.get() + PAGE_SIZE - 4);
    assert_eq!(value_at(&mut jit, end), 0);
    write(
        &space,
        GuestVirtualAddress::new(ALIAS.get() + PAGE_SIZE - 4),
        &0x91000400_u32.to_le_bytes(),
    );
    let mut state = ThreadState::default();
    state.set_pc(end.get());
    assert!(
        matches!(run(&mut jit, &mut state).stop, CpuExit::FetchFault { fault } if fault.address == DATA && fault.reason == InstructionFetchFaultReason::Unmapped)
    );
    assert_eq!(state.pc(), DATA.get());
    assert_eq!(state.read_x(x(0)), 1);
    state.set_pc(end.get() + 1);
    assert!(
        matches!(run(&mut jit, &mut state).stop, CpuExit::FetchFault { fault } if fault.reason == InstructionFetchFaultReason::Misaligned)
    );
}

#[test]
fn invalidation_retires_links_to_a_different_code_page() {
    let (space, process) = fixture(&[0xd2800540, 0x140003ff]); // MOV #42; B 0x2000
    write(
        &space,
        DATA,
        &[0xd2800560_u32.to_le_bytes(), 0xd4000001_u32.to_le_bytes()].concat(),
    );
    space
        .protect(DATA, PAGE_SIZE, MemoryPermissions::READ_EXECUTE)
        .unwrap();
    let mut jit = process.create_thread().unwrap();
    for _ in 0..3 {
        assert_eq!(value_at(&mut jit, CODE), 43);
    }
    let target = space.capture(DATA, 4, MemoryPermissions::READ).unwrap();
    space
        .machine()
        .write_resolved(&target, &0xd2800580_u32.to_le_bytes())
        .unwrap();
    assert_eq!(value_at(&mut jit, CODE), 44);
}

#[test]
fn switch_1_pointer_authentication_hints_preserve_registers() {
    let hints = [
        0xd50320ff, 0xd503211f, 0xd503215f, 0xd503219f, 0xd50321df, 0xd503231f, 0xd503233f,
        0xd503235f, 0xd503237f, 0xd503239f, 0xd50323bf, 0xd50323df, 0xd50323ff,
    ];
    let mut words = hints.to_vec();
    words.push(0xd4200000);
    let (_space, process) = fixture(&words);
    let mut jit = process.create_thread().unwrap();
    let mut state = ThreadState::default();
    state.set_pc(CODE.get());
    for index in 0..31 {
        state.write_x(x(index), 0xabcd000075187c14 + u64::from(index));
    }
    let mut expected = state.clone();
    expected.set_pc(CODE.get() + hints.len() as u64 * 4);
    assert!(matches!(
        run(&mut jit, &mut state).stop,
        CpuExit::ArchitecturalException { .. }
    ));
    assert_eq!(state, expected);
}

#[test]
fn prohibited_cache_zero_traps_without_writing_ram() {
    let (space, process) = fixture(&[0xd50b7421, 0xd4000001]); // DC ZVA,X1; SVC
    write(&space, DATA, &[0x55; 64]);
    let mut jit = process.create_thread().unwrap();
    let mut state = ThreadState::default();
    state.set_pc(CODE.get());
    state.write_x(x(1), DATA.get());
    assert!(matches!(
        run(&mut jit, &mut state).stop,
        CpuExit::ArchitecturalException {
            kind: nixe_cpu::exception::ExceptionKind::SystemRegisterTrap,
            ..
        }
    ));
    assert_eq!(state.pc(), CODE.get());
    assert!(matches!(
        space.read_value(DATA, 8).unwrap(),
        Access::Ready(0x5555555555555555)
    ));
}
