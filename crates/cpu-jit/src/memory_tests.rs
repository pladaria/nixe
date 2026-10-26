//! Memory behavior through the same process/context entry used by runtime.
use crate::{JitProcess, JitRunRequest, ThreadState};
use nixe_cpu::{
    execution::{ArchitecturalTimer, CpuExit, TimerSnapshot, VcpuEventState},
    platform::TargetPlatform,
    profile::ProcessCpuContext,
    state::a64::{A64GeneralRegister, A64Register},
};
use nixe_cpu_direct_memory::NativeWorker;
use nixe_memory::{
    AccessOperation, AddressSpace, AddressWidth, AllocationRange, DeviceAccessEndpoint,
    DeviceAccessRequest, GuestVirtualAddress, Machine, MemoryPermissions, PAGE_SIZE,
};
use std::{
    sync::{Arc, mpsc},
    time::Duration,
};

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
                events: &VcpuEventState::default(),
            },
        )
        .unwrap();
    worker.finish().unwrap();
    report.stop
}
struct Endpoint(mpsc::Sender<DeviceAccessRequest>);
impl DeviceAccessEndpoint for Endpoint {
    fn submit(&self, request: DeviceAccessRequest) {
        self.0.send(request).unwrap();
    }
    fn cancel(&self, _: nixe_memory::AccessRequestId) {}
    fn allocation_released(&self, _: nixe_memory::MachineId, _: nixe_memory::AllocationId) {}
    fn physical_retired(
        &self,
        _: nixe_memory::MachineId,
        _: nixe_memory::AllocationId,
        _: Box<[nixe_memory::Extent]>,
    ) {
    }
    fn allocation_pressure(&self, _: nixe_memory::MachineId, _: u64) {}
}
fn endpoint(
    space: &AddressSpace,
) -> (
    Arc<dyn DeviceAccessEndpoint>,
    mpsc::Receiver<DeviceAccessRequest>,
) {
    let (send, receive) = mpsc::channel();
    let endpoint: Arc<dyn DeviceAccessEndpoint> = Arc::new(Endpoint(send));
    space.machine().set_device_endpoint(&endpoint);
    (endpoint, receive)
}
fn capture(space: &AddressSpace, address: u64, bytes: u64) -> Vec<AllocationRange> {
    space
        .capture(
            GuestVirtualAddress::new(address),
            bytes,
            MemoryPermissions::READ_WRITE,
        )
        .unwrap()
}
fn read(space: &AddressSpace, address: u64, bytes: usize) -> u128 {
    space
        .read_value(GuestVirtualAddress::new(address), bytes)
        .unwrap()
        .into_ready()
        .unwrap()
}

#[test]
fn observed_cross_page_simd_store_invalidates_observation_and_preserves_registers() {
    // STR Q0,[X1]; LDR Q1,[X1]; SVC.
    for address in [DATA + 56, DATA + PAGE_SIZE - 8] {
        let (space, process, mut state) = fixture(&[0x3d800020, 0x3dc00021, 0xd4000001]);
        state.write_x(x(1), address);
        let value = 0x112233445566778899aabbccddeeff00;
        state.set_vector(0, value);
        let observation = nixe_memory::Observation::default();
        space
            .machine()
            .access_captured(
                &capture(&space, address, 16),
                AccessOperation::ReadObserved(observation.clone()),
            )
            .unwrap()
            .into_ready()
            .unwrap();
        assert!(!observation.is_invalid());
        assert!(matches!(
            run(&process, &mut state),
            CpuExit::SupervisorCall { .. }
        ));
        assert!(observation.is_invalid());
        assert_eq!(state.vector(0), Some(value));
        assert_eq!(state.vector(1), Some(value));
        assert_eq!(read(&space, address, 16), value);
    }
}

#[test]
fn device_simd_read_completes_within_the_original_native_instruction() {
    // LDR Q0,[X1]; ADD X0,X0,#1; SVC.
    let (space, process, mut state) = fixture(&[0x3dc00020, 0x91000400, 0xd4000001]);
    let address = DATA + PAGE_SIZE - 8;
    let value = 0x22222222222222221111111111111111;
    space
        .write_value(GuestVirtualAddress::new(address), 16, value)
        .unwrap()
        .into_ready()
        .unwrap();
    let ranges = capture(&space, DATA + PAGE_SIZE, 8);
    let (_endpoint, requests) = endpoint(&space);
    space
        .machine()
        .publish_device_write(&ranges, || Ok(()))
        .unwrap();
    state.write_x(x(1), address);
    let execution = std::thread::spawn(move || {
        assert!(matches!(
            run(&process, &mut state),
            CpuExit::SupervisorCall { .. }
        ));
        state
    });
    let request = requests.recv_timeout(Duration::from_secs(5)).unwrap();
    assert_eq!(request.bytes(), 16);
    let result = space
        .machine()
        .apply_resolved_access(request.ranges(), request.operation())
        .unwrap();
    request.finish(Ok(result));
    let state = execution.join().unwrap();
    assert_eq!(state.vector(0), Some(value));
    assert_eq!(state.read_x(x(0)), 8);
    assert!(requests.try_recv().is_err());
}

#[test]
fn pending_callback_allows_remapping_and_commits_its_original_physical_access_once() {
    let (space, process, mut state) = fixture(&[0xb9400020, 0xd4000001]);
    space
        .write_value(GuestVirtualAddress::new(DATA), 4, 17)
        .unwrap()
        .into_ready()
        .unwrap();
    let old = capture(&space, DATA, 4);
    let (_endpoint, requests) = endpoint(&space);
    space
        .machine()
        .publish_device_write(&old, || Ok(()))
        .unwrap();
    let execution = std::thread::spawn(move || {
        assert!(matches!(
            run(&process, &mut state),
            CpuExit::SupervisorCall { .. }
        ));
        state
    });
    let request = requests.recv_timeout(Duration::from_secs(5)).unwrap();
    space
        .unmap(GuestVirtualAddress::new(DATA), 2 * PAGE_SIZE)
        .unwrap();
    let replacement = space
        .machine()
        .account(2 * PAGE_SIZE)
        .unwrap()
        .allocate(2 * PAGE_SIZE)
        .unwrap();
    space
        .map(
            GuestVirtualAddress::new(DATA),
            replacement,
            0,
            2 * PAGE_SIZE,
            MemoryPermissions::READ_WRITE,
        )
        .unwrap();
    space
        .write_value(GuestVirtualAddress::new(DATA), 4, 23)
        .unwrap()
        .into_ready()
        .unwrap();
    let result = space
        .machine()
        .apply_resolved_access(request.ranges(), request.operation())
        .unwrap();
    request.finish(Ok(result));
    let state = execution.join().unwrap();
    assert_eq!(state.read_x(x(0)), 17);
    assert_eq!(read(&space, DATA, 4), 23);
    assert!(requests.try_recv().is_err());
}

#[test]
fn pair_store_never_replays_its_prefix_while_a_later_callback_waits() {
    // STP X0,X5,[X1],#16; SVC. Only the second scalar store needs the device.
    let (space, process, mut state) = fixture(&[0xa8811420, 0xd4000001]);
    let first = DATA + PAGE_SIZE - 8;
    state.write_x(x(1), first);
    let (_endpoint, requests) = endpoint(&space);
    space
        .machine()
        .publish_device_write(&capture(&space, DATA + PAGE_SIZE, 8), || Ok(()))
        .unwrap();
    let execution = std::thread::spawn(move || {
        assert!(matches!(
            run(&process, &mut state),
            CpuExit::SupervisorCall { .. }
        ));
        state
    });
    let request = requests.recv_timeout(Duration::from_secs(5)).unwrap();
    assert_eq!(request.bytes(), 8);
    assert!(matches!(request.operation(), AccessOperation::Write(_)));
    assert_eq!(read(&space, ALIAS + PAGE_SIZE - 8, 8), 7);
    // Both the old physical prefix and its new mapping change during the wait.
    space
        .write_value(GuestVirtualAddress::new(ALIAS + PAGE_SIZE - 8), 8, 17)
        .unwrap()
        .into_ready()
        .unwrap();
    space
        .unmap(GuestVirtualAddress::new(DATA), PAGE_SIZE)
        .unwrap();
    let replacement = space
        .machine()
        .account(PAGE_SIZE)
        .unwrap()
        .allocate(PAGE_SIZE)
        .unwrap();
    space
        .map(
            GuestVirtualAddress::new(DATA),
            replacement,
            0,
            PAGE_SIZE,
            MemoryPermissions::READ_WRITE,
        )
        .unwrap();
    space
        .write_value(GuestVirtualAddress::new(first), 8, 23)
        .unwrap()
        .into_ready()
        .unwrap();
    let result = space
        .machine()
        .apply_resolved_access(request.ranges(), request.operation())
        .unwrap();
    request.finish(Ok(result));
    let state = execution.join().unwrap();
    assert_eq!(state.read_x(x(1)), first + 16);
    assert_eq!(read(&space, first, 8), 23);
    assert_eq!(read(&space, ALIAS + PAGE_SIZE - 8, 8), 17);
    assert_eq!(
        space
            .machine()
            .apply_resolved_access(
                &capture(&space, DATA + PAGE_SIZE, 8),
                &AccessOperation::Read,
            )
            .unwrap(),
        nixe_memory::AccessResult::Read(Box::new(9u64.to_le_bytes()))
    );
    assert!(requests.try_recv().is_err());
}

#[test]
fn mixed_page_repeated_reads_reuse_coherency_without_caching_values() {
    // LDR W0,[X1]; ADD X3,X3,X0; SUBS X2,X2,#1; B.NE; SVC.
    let (space, process, mut state) =
        fixture(&[0xb9400020, 0x8b000063, 0xf1000442, 0x54ffffa1, 0xd4000001]);
    let data = capture(&space, DATA, PAGE_SIZE);
    let field = data[0].slice(128, 4).unwrap();
    let reusable = data[0].slice(64, 448).unwrap();
    space
        .machine()
        .write_resolved(std::slice::from_ref(&field), &7u32.to_le_bytes())
        .unwrap();
    let (_endpoint, requests) = endpoint(&space);
    space
        .machine()
        .publish_device_write(&[data[0].slice(0, 64).unwrap()], || Ok(()))
        .unwrap();
    state.write_x(x(1), DATA + 128);
    state.write_x(x(2), 1000);
    let writing_space = Arc::clone(&space);
    let execution = std::thread::spawn(move || {
        let mut thread = process.create_thread().unwrap();
        let mut worker = NativeWorker::default();
        for value in [7, 9] {
            state.set_pc(CODE);
            state.write_x(x(2), 1000);
            state.write_x(x(3), 0);
            let report = thread
                .run_slice(
                    &mut worker,
                    JitRunRequest {
                        state: &mut state,
                        timer: &Timer,
                        events: &VcpuEventState::default(),
                    },
                )
                .unwrap();
            assert!(matches!(report.stop, CpuExit::SupervisorCall { .. }));
            assert_eq!(state.read_x(x(3)), value * 1000);
            writing_space
                .machine()
                .write_resolved(std::slice::from_ref(&field), &9u32.to_le_bytes())
                .unwrap();
        }
        worker.finish().unwrap();
    });
    let request = requests.recv_timeout(Duration::from_secs(5)).unwrap();
    assert!(matches!(request.operation(), AccessOperation::ReadNative));
    let result = space
        .machine()
        .read_native_resolved(request.ranges(), Some(&reusable))
        .unwrap();
    request.finish(Ok(result));
    execution.join().unwrap();
    assert!(requests.try_recv().is_err());
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
