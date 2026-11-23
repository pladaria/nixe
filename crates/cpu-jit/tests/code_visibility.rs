//! Code invalidation and mapping lifetime on the production physical arena.
use nixe_cpu::{
    execution::{ArchitecturalTimer, CpuExit, TimerSnapshot},
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
        None,
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
            timer: &Timer,
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
fn native_cache_zero_clears_the_aligned_guest_cache_line() {
    let (space, process) = fixture(&[0xd50b7421, 0xd4000001]); // DC ZVA,X1; SVC.
    write(&space, DATA, &[0x55; 128]);
    let mut jit = process.create_thread().unwrap();
    let mut state = ThreadState::default();
    state.set_pc(CODE.get());
    state.write_x(x(1), DATA.get() + 31);
    assert!(matches!(
        run(&mut jit, &mut state).stop,
        CpuExit::SupervisorCall { .. }
    ));
    assert_eq!(space.read_value(DATA, 16).unwrap().into_ready().unwrap(), 0);
    assert_eq!(
        space
            .read_value(GuestVirtualAddress::new(DATA.get() + 48), 16)
            .unwrap()
            .into_ready()
            .unwrap(),
        0
    );
    assert_eq!(
        space
            .read_value(GuestVirtualAddress::new(DATA.get() + 64), 16)
            .unwrap()
            .into_ready()
            .unwrap(),
        u128::from_le_bytes([0x55; 16])
    );
}
