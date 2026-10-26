use super::*;
use crate::ThreadState;
use nixe_cpu::{
    execution::{ArchitecturalTimer, TimerSnapshot, VcpuEventState},
    platform::TargetPlatform,
    state::a64::{A64GeneralRegister, A64Register},
};
use nixe_memory::{Access, AddressWidth, AllocationRange, MemoryPermissions, PAGE_SIZE};
use std::{sync::mpsc, time::Duration};
const CODE: u64 = 0x1000;
const DATA: u64 = 0x3000;
struct Timer;
impl ArchitecturalTimer for Timer {
    fn snapshot(&self) -> TimerSnapshot {
        TimerSnapshot {
            counter: 0,
            frequency: 19_200_000,
        }
    }
}
fn register(index: u8) -> A64Register {
    A64Register::General(A64GeneralRegister::new(index).unwrap())
}
fn fixture(words: &[u32]) -> Arc<Process> {
    let machine = nixe_memory::Machine::new(4 * PAGE_SIZE).unwrap();
    let account = machine.account(4 * PAGE_SIZE).unwrap();
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
    space
        .map(
            GuestVirtualAddress::new(DATA),
            account.allocate(PAGE_SIZE).unwrap(),
            0,
            PAGE_SIZE,
            MemoryPermissions::READ_WRITE,
        )
        .unwrap();
    Process::new(
        ProcessCpuContext::new(TargetPlatform::Switch1, space.identity()),
        space,
        19_200_000,
    )
    .unwrap()
}
fn state() -> ThreadState {
    let mut state = ThreadState::default();
    state.set_pc(CODE);
    state.write_x(register(1), DATA);
    state.write_x(register(0), 7);
    state.write_x(register(5), 9);
    state
}
fn run(
    thread: &mut Thread,
    worker: &mut NativeWorker,
    state: &mut ThreadState,
) -> Result<ExecutionReport, CpuFault> {
    thread.run_slice(
        worker,
        JitRunRequest {
            state,
            timer: &Timer,
            events: &VcpuEventState::default(),
        },
    )
}
#[test]
fn exclusive_wait_loop_progresses_after_a_store_without_an_explicit_sev() {
    let process = fixture(&[
        0xb9000825, // STR W5,[X1,#8]: consumer entered native code.
        0xd50320bf, // SEVL
        0xd503205f, // loop: WFE
        0x885ffc20, // LDAXR W0,[X1]
        0x35ffffc0, // CBNZ W0,loop
        0xd4000001, // SVC
        0x889ffc3f, // STLR WZR,[X1]
        0xd4000001, // SVC
    ]);
    process
        .space
        .write_value(GuestVirtualAddress::new(DATA), 4, 1)
        .unwrap()
        .into_ready()
        .unwrap();
    let mut consumer = process.create_thread().unwrap();
    let (done, completed) = mpsc::channel();
    let executing = std::thread::spawn(move || {
        let mut worker = NativeWorker::default();
        let mut guest = state();
        let result = run(&mut consumer, &mut worker, &mut guest);
        done.send((result, guest)).unwrap();
        worker.finish().unwrap();
    });
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while process
        .space
        .read_value(GuestVirtualAddress::new(DATA + 8), 4)
        .unwrap()
        .into_ready()
        .unwrap()
        != 9
    {
        if std::time::Instant::now() >= deadline {
            process.request_stop();
            executing.join().unwrap();
            panic!("consumer did not enter native execution");
        }
        std::thread::yield_now();
    }
    let mut producer = process.create_thread().unwrap();
    let mut guest = state();
    guest.set_pc(CODE + 24);
    let mut worker = NativeWorker::default();
    assert!(matches!(
        run(&mut producer, &mut worker, &mut guest).unwrap().stop,
        CpuExit::SupervisorCall { .. }
    ));
    let completed = completed.recv_timeout(Duration::from_secs(5));
    if completed.is_err() {
        process.request_stop();
    }
    executing.join().unwrap();
    let (result, guest) = completed.unwrap();
    assert!(matches!(
        result.unwrap().stop,
        CpuExit::SupervisorCall { .. }
    ));
    assert_eq!(guest.read_x(register(0)), 0);
    worker.finish().unwrap();
}

#[test]
fn canonical_process_prepares_code_and_retains_registers_across_svc_and_migration() {
    let process = fixture(&[0x91000400, 0xd4000001, 0x17fffffe]); // ADD X0,#1; SVC; B start.
    let mut first = process.create_thread().unwrap();
    let mut second = process.create_thread().unwrap();
    let mut guest = state();
    let mut worker = NativeWorker::default();
    for expected in 8..11 {
        let report = run(&mut first, &mut worker, &mut guest).unwrap();
        assert!(matches!(
            report.stop,
            CpuExit::SupervisorCall { immediate: 0, .. }
        ));
        assert!(report.context.is_none());
        assert!(guest.is_resident());
        assert_eq!(guest.read_x(register(0)), expected);
        guest.set_pc(guest.pc() + 4);
    }
    guest.leave_core();
    assert!(!guest.is_resident());
    let mut other = state();
    other.write_x(register(0), 20);
    assert!(matches!(
        run(&mut first, &mut worker, &mut other).unwrap().stop,
        CpuExit::SupervisorCall { .. }
    ));
    assert!(matches!(
        run(&mut second, &mut worker, &mut guest).unwrap().stop,
        CpuExit::SupervisorCall { .. }
    ));
    assert_eq!(guest.read_x(register(0)), 11);
    assert_eq!(other.read_x(register(0)), 21);
    assert!(process.try_shutdown().unwrap());
    assert_eq!(process.space.machine().statistics().allocated_bytes, 0);
    worker.finish().unwrap();
}

struct Endpoint(mpsc::Sender<nixe_memory::DeviceAccessRequest>);
impl nixe_memory::DeviceAccessEndpoint for Endpoint {
    fn submit(&self, request: nixe_memory::DeviceAccessRequest) {
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
#[test]
fn stopping_a_native_dependency_cancels_the_unexecuted_suffix_and_releases_ram() {
    let process = fixture(&[0xa9001420, 0xd4000001]); // STP X0,X5,[X1].
    let machine = process.space.machine().clone();
    let (send, requests) = mpsc::channel();
    let endpoint: Arc<dyn nixe_memory::DeviceAccessEndpoint> = Arc::new(Endpoint(send));
    machine.set_device_endpoint(&endpoint);
    process
        .space
        .map(
            GuestVirtualAddress::new(DATA + PAGE_SIZE),
            machine
                .account(PAGE_SIZE)
                .unwrap()
                .allocate(PAGE_SIZE)
                .unwrap(),
            0,
            PAGE_SIZE,
            MemoryPermissions::READ_WRITE,
        )
        .unwrap();
    let data = process
        .space
        .capture(
            GuestVirtualAddress::new(DATA + PAGE_SIZE - 8),
            16,
            MemoryPermissions::WRITE,
        )
        .unwrap();
    machine.publish_device_write(&data, || Ok(())).unwrap();
    drop(data);
    let mut cpu = process.create_thread().unwrap();
    let (done, completed) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        let mut worker = NativeWorker::default();
        let mut guest = state();
        guest.write_x(register(1), DATA + PAGE_SIZE - 8);
        let result = run(&mut cpu, &mut worker, &mut guest);
        done.send((result, guest.is_available())).unwrap();
        worker.finish().unwrap();
    });
    let request = requests.recv_timeout(Duration::from_secs(5)).unwrap();
    assert_eq!(
        request.bytes(),
        8,
        "the page boundary preserves a separate unexecuted suffix"
    );
    process.request_stop();
    let (result, available) = completed.recv_timeout(Duration::from_secs(5)).unwrap();
    assert!(result.is_err());
    assert!(!available);
    assert!(request.is_cancelled());
    assert!(
        requests.try_recv().is_err(),
        "the cancelled instruction must not start its suffix"
    );
    worker.join().unwrap();
    assert!(process.try_shutdown().unwrap());
    assert_eq!(machine.statistics().guest_owned_bytes, 0);
    assert_eq!(machine.statistics().retained_bytes, PAGE_SIZE);
    drop(request);
    assert_eq!(machine.statistics().allocated_bytes, 0);
}

#[test]
fn a_concurrent_stop_preserves_the_original_completed_device_failure() {
    let process = fixture(&[0xb9400020, 0xd4000001]); // LDR W0,[X1].
    let (send, requests) = mpsc::channel();
    let endpoint: Arc<dyn nixe_memory::DeviceAccessEndpoint> = Arc::new(Endpoint(send));
    let machine = process.space.machine().clone();
    machine.set_device_endpoint(&endpoint);
    let data = process
        .space
        .capture(GuestVirtualAddress::new(DATA), 4, MemoryPermissions::WRITE)
        .unwrap();
    machine.publish_device_write(&data, || Ok(())).unwrap();
    drop(data);
    let mut cpu = process.create_thread().unwrap();
    let (done, completed) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        let mut worker = NativeWorker::default();
        let result = run(&mut cpu, &mut worker, &mut state());
        done.send(result).unwrap();
        worker.finish().unwrap();
    });
    let request = requests.recv_timeout(Duration::from_secs(5)).unwrap();
    let original = Arc::new(MemoryError::Invalid("injected original device failure"));
    request.finish(Err(Arc::clone(&original)));
    process.request_stop();
    let fault = completed
        .recv_timeout(Duration::from_secs(5))
        .unwrap()
        .unwrap_err();
    assert!(fault.message.contains("injected original device failure"));
    assert!(Arc::ptr_eq(process.failure.get().unwrap(), &original));
    worker.join().unwrap();
    assert!(process.try_shutdown().unwrap());
    assert_eq!(machine.statistics().allocated_bytes, 0);
}

#[test]
fn canonical_process_reports_fetch_faults_and_relaxes_invalid_guest_data() {
    for fetch in [true, false] {
        let process = fixture(&[0xb9400020, 0xd4000001]);
        let mut cpu = process.create_thread().unwrap();
        let mut guest = state();
        if fetch {
            guest.set_pc(0x6000);
        } else {
            guest.write_x(register(1), 0x6000);
        }
        let mut worker = NativeWorker::default();
        let report = run(&mut cpu, &mut worker, &mut guest).unwrap();
        if fetch {
            assert!(report.context.is_some());
            assert!(
                matches!(report.stop, CpuExit::FetchFault { fault } if fault.address.get() == 0x6000)
            );
        } else {
            assert!(matches!(report.stop, CpuExit::SupervisorCall { .. }));
            assert!(report.context.is_none());
            assert_eq!(guest.read_x(register(0)), 0);
        }
        assert!(matches!(
            process
                .space
                .read_value(GuestVirtualAddress::new(DATA), 4)
                .unwrap(),
            Access::Ready(0)
        ));
        assert!(process.try_shutdown().unwrap());
        worker.finish().unwrap();
    }
}
