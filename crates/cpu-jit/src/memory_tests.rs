//! Memory behavior through the same process/context entry used by runtime.
use crate::{JitProcess, JitRunRequest, ThreadState};
use nixe_cpu::{
    execution::{ArchitecturalTimer, CpuExit, TimerSnapshot},
    platform::TargetPlatform,
    profile::ProcessCpuContext,
    state::a64::{A64GeneralRegister, A64Register},
};
use nixe_cpu_direct_memory::NativeWorker;
use nixe_memory::{
    AddressSpace, AddressWidth, GuestVirtualAddress, Machine, MemoryPermissions, PAGE_SIZE,
};
use std::sync::Arc;

const CODE: u64 = 0x1000;
const DATA: u64 = 0x3000;
const ALIAS: u64 = 0x6000;
struct Timer;
impl ArchitecturalTimer for Timer {
    fn snapshot(&self) -> TimerSnapshot {
        TimerSnapshot {
            counter: 0,
            frequency: 19_200_000,
        }
    }
}
fn x(index: u8) -> A64Register {
    A64Register::General(A64GeneralRegister::new(index).unwrap())
}
fn fixture(words: &[u32]) -> (Arc<AddressSpace>, Arc<JitProcess>, ThreadState) {
    let machine = Machine::new(8 * PAGE_SIZE).unwrap();
    let account = machine.account(8 * PAGE_SIZE).unwrap();
    let code = account.allocate(PAGE_SIZE).unwrap();
    machine
        .write_resolved(
            &[AllocationRange::new(code.pin(), 0, words.len() as u64 * 4).unwrap()],
            &words
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let data = account.allocate(2 * PAGE_SIZE).unwrap();
    let space = AddressSpace::new(machine, AddressWidth::Bits32).unwrap();
    space
        .map(
            GuestVirtualAddress::new(CODE),
            code,
            0,
            PAGE_SIZE,
            MemoryPermissions::READ_EXECUTE,
        )
        .unwrap();
    for address in [DATA, ALIAS] {
        space
            .map(
                GuestVirtualAddress::new(address),
                data.clone(),
                0,
                2 * PAGE_SIZE,
                MemoryPermissions::READ_WRITE,
            )
            .unwrap();
    }
    let process = JitProcess::new(
        ProcessCpuContext::new(TargetPlatform::Switch1, space.identity()),
        Arc::clone(&space),
        19_200_000,
    )
    .unwrap();
    let mut state = ThreadState::default();
    state.set_pc(CODE);
    state.write_x(x(0), 7);
    state.write_x(x(1), DATA);
    state.write_x(x(4), ALIAS);
    state.write_x(x(5), 9);
    (space, process, state)
}
fn run(process: &Arc<JitProcess>, state: &mut ThreadState) -> CpuExit {
    let mut thread = process.create_thread().unwrap();
    let mut worker = NativeWorker::default();
    let report = thread
        .run_slice(
            &mut worker,
            JitRunRequest {
                state,
                timer: &Timer,
            },
        )
        .unwrap();
    state.leave_core();
    drop(thread);
    worker.finish().unwrap();
    report.stop
}
fn read(space: &AddressSpace, address: u64, bytes: usize) -> u128 {
    space
        .read_value(GuestVirtualAddress::new(address), bytes)
        .unwrap()
        .into_ready()
        .unwrap()
}

#[test]
fn write_only_guest_mapping_uses_the_checked_native_store() {
    let (space, process, mut state) = fixture(&[0xf9000020, 0xd4000001]);
    space
        .protect(
            GuestVirtualAddress::new(DATA),
            2 * PAGE_SIZE,
            MemoryPermissions::WRITE,
        )
        .unwrap();
    assert!(matches!(
        run(&process, &mut state),
        CpuExit::SupervisorCall { .. }
    ));
    assert_eq!(read(&space, ALIAS, 8), 7);
}

#[test]
fn invalid_guest_writes_are_ignored_without_ending_native_execution() {
    let (space, process, mut guest) = fixture(&[0xf9000020, 0xd2800020, 0xd4000001]);
    guest.write_x(x(1), 0xdead0000);
    assert!(matches!(
        run(&process, &mut guest),
        CpuExit::SupervisorCall { .. }
    ));
    assert_eq!(guest.read_x(x(0)), 1);
    assert_eq!(read(&space, DATA, 8), 0);
}
