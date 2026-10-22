//! Native ABI coverage on the canonical arena and physical reservation metadata.
//! https://developer.arm.com/documentation/ddi0602/latest/Base-Instructions/LDXR--Load-Exclusive-Register-
//! https://developer.arm.com/documentation/ddi0602/latest/Base-Instructions/STXR--Store-Exclusive-Register-
use crate::execution::{Configuration, Core};
use crate::instruction::InstructionContinuation;
use nixe_cpu::{
    execution::{ArchitecturalTimer, TimerSnapshot},
    state::a64::{A64GeneralRegister, A64Register, A64State},
};
use nixe_cpu_direct_memory::NativeWorker;
use nixe_memory::{
    Access, AddressSpace, AddressWidth, AllocationRange, DeviceWriteEpoch, GuestVirtualAddress,
    Machine, MemoryPermissions, PAGE_SIZE,
};
use std::sync::Arc;

const CODE: u64 = 0x1000;
const DATA: u64 = 0x3000;
const ALIAS: u64 = 0x5000;
struct Timer;
impl ArchitecturalTimer for Timer {
    fn snapshot(&self) -> TimerSnapshot {
        TimerSnapshot {
            counter: 0,
            frequency: 19_200_000,
        }
    }
}
struct Runner {
    core: Core,
    space: Arc<AddressSpace>,
}
impl Runner {
    fn run(&mut self) {
        for _ in 0..16 {
            if let Some(pending) = self.invoke(1000, true) {
                self.complete_ram(pending);
            } else {
                return;
            }
        }
        panic!("native memory completion did not reach SVC");
    }
    fn complete_ram(&mut self, mut pending: InstructionContinuation) {
        let completed = pending
            .poll(self.space.machine(), &Timer)
            .expect("RAM completion cannot park")
            .unwrap();
        assert!(matches!(
            self.core.complete(completed).unwrap().0,
            nixe_cpu_interpreter::InstructionStep::Continue
        ));
    }
    fn load_exclusive(&mut self) {
        assert!(self.invoke(1, false).is_none());
        assert_eq!(self.register(32), CODE + 12);
    }
    fn invoke(&mut self, budget: u64, svc: bool) -> Option<InstructionContinuation> {
        let mut worker = NativeWorker::default();
        let boundary = self
            .core
            .run(&mut worker, budget, &Timer, &Default::default())
            .unwrap();
        assert!(boundary.code.is_empty());
        if boundary.memory.is_some() {
            assert!(matches!(boundary.event.kind, 0 | 1));
        } else {
            assert_eq!(boundary.event.kind, if svc { 2 } else { 0 });
        }
        if svc && boundary.memory.is_none() {
            self.core.write_register(32, boundary.event.pc + 4);
        }
        worker.finish().unwrap();
        boundary.memory
    }
    fn register(&self, index: u32) -> u64 {
        self.core.read_register(index)
    }
}
fn fixture(words: &[u32]) -> (Runner, DeviceWriteEpoch) {
    fixture_with_tracking(words, true)
}
fn fixture_with_tracking(words: &[u32], track_code: bool) -> (Runner, DeviceWriteEpoch) {
    let machine = Machine::new(3 * PAGE_SIZE).unwrap();
    let account = machine.account(2 * PAGE_SIZE).unwrap();
    let code_allocation = account.allocate(PAGE_SIZE).unwrap();
    let code_range =
        AllocationRange::new(code_allocation.pin(), 0, words.len() as u64 * 4).unwrap();
    machine
        .write_resolved(
            std::slice::from_ref(&code_range),
            &words
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    if track_code {
        machine
            .access_code_captured(vec![code_range])
            .unwrap()
            .0
            .into_ready()
            .unwrap();
    }
    let data = account.allocate(PAGE_SIZE).unwrap();
    let epoch = data.pin().device_write_epoch();
    machine
        .prepare_reservations(&[AllocationRange::new(data.pin(), 0, 16).unwrap()])
        .unwrap();
    let space = AddressSpace::new(machine, AddressWidth::Bits32).unwrap();
    space
        .map(
            GuestVirtualAddress::new(CODE),
            code_allocation,
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
                PAGE_SIZE,
                MemoryPermissions::READ_WRITE,
            )
            .unwrap();
    }
    let mut core = Core::new(
        Arc::clone(&space),
        Configuration {
            cpu: nixe_cpu::profile::ProcessCpuContext::new(
                nixe_cpu::platform::TargetPlatform::Switch1,
                space.identity(),
            ),
            timer_frequency: 19_200_000,
        },
    )
    .unwrap();
    let mut state = A64State::default();
    state.set_pc(CODE);
    for (index, value) in [(0, 7), (1, DATA), (4, ALIAS), (5, 9)] {
        state.write_x(
            A64Register::General(A64GeneralRegister::new(index).unwrap()),
            value,
        );
    }
    core.load(&state);
    (Runner { core, space }, epoch)
}

#[test]
fn data_capture_stops_only_the_conflicting_native_store_and_replays_it_once() {
    // STR X0, [X1]; ADD X0, X0, #1; B -8.
    let (runner, _) = fixture(&[0xf9000020, 0x91000400, 0x17fffffe]);
    let machine = runner.space.machine().clone();
    let range = runner
        .space
        .capture(GuestVirtualAddress::new(DATA), 8, MemoryPermissions::READ)
        .unwrap();
    let control = runner.core.control();
    let mut runner = runner;
    for _ in 0..32 {
        let executing = std::thread::spawn(move || {
            let mut runner = runner;
            let mut worker = NativeWorker::default();
            let boundary = runner
                .core
                .run(&mut worker, 100_000_000, &Timer, &Default::default())
                .unwrap();
            worker.finish().unwrap();
            (runner, boundary)
        });
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        let mut stored = [0; 8];
        while std::time::Instant::now() < deadline {
            machine.capture_stream(&range, &mut stored).unwrap();
            if control.execution_active() && u64::from_le_bytes(stored) != 0 {
                break;
            }
            std::thread::yield_now();
        }
        assert_ne!(
            u64::from_le_bytes(stored),
            0,
            "native store loop did not start"
        );
        let observation = nixe_memory::Observation::default();
        let nixe_memory::AccessResult::Read(observation_bytes) = machine
            .access_captured(
                &range,
                nixe_memory::AccessOperation::ReadObserved(observation.clone()),
            )
            .unwrap()
            .into_ready()
            .unwrap()
        else {
            panic!("expected captured bytes")
        };
        let (next_runner, boundary) = executing.join().unwrap();
        runner = next_runner;
        assert!(boundary.control.is_none());
        assert!(boundary.fault.is_none());
        let pending = boundary
            .memory
            .expect("conflicting store must leave Run before invalidation");
        assert!(!observation.is_invalid());
        let before = u64::from_le_bytes(observation_bytes.as_ref().try_into().unwrap());
        let source = runner.register(0);
        assert_eq!(source, before + 1);
        runner.complete_ram(pending);
        assert!(observation.is_invalid());
        for address in [DATA, ALIAS] {
            assert_eq!(
                runner
                    .space
                    .read_value(GuestVirtualAddress::new(address), 8)
                    .unwrap()
                    .into_ready()
                    .unwrap(),
                u128::from(source)
            );
        }
    }
}

#[test]
fn device_dependent_stores_collect_inside_native_execution_without_instruction_replay() {
    // STR Q0, [X1]; STR Q0, [X4]; SVC #0. Both addresses name physical aliases.
    let (mut runner, _) = fixture(&[0x3d800020, 0x3d800080, 0xd4000001]);
    let machine = runner.space.machine().clone();
    let range = runner
        .space
        .capture(GuestVirtualAddress::new(DATA), 16, MemoryPermissions::WRITE)
        .unwrap();
    machine.publish_device_write(&range, || Ok(())).unwrap();
    machine.enable_cpu_write_collection().unwrap();
    let mut state = A64State::default();
    runner.core.save(&mut state);
    let value = 0x102030405060708090a0b0c0d0e0f000;
    assert!(state.set_vector(0, value));
    runner.core.load(&state);
    assert!(runner.invoke(100, true).is_none());
    let dirty = machine.collect_cpu_writes().unwrap();
    assert_eq!(dirty.len(), 1);
    assert_eq!(dirty[0].bytes(), 16);
    let mut ram = [0; 16];
    machine.capture_stream(&range, &mut ram).unwrap();
    assert_eq!(u128::from_le_bytes(ram), value);
    machine.disable_cpu_write_collection();
}

#[test]
fn ordinary_cross_page_and_cross_granule_simd_stores_complete_inside_native_run() {
    // STR Q0, [X1]; SVC #0.
    // https://developer.arm.com/documentation/ddi0602/latest/SIMD-FP-Instructions/STR--immediate---SIMD-FP---Store-SIMD-FP-register--immediate--
    for address in [DATA + 56, DATA + PAGE_SIZE - 8] {
        let (mut runner, _) = fixture(&[0x3d800020, 0xd4000001]);
        let machine = runner.space.machine().clone();
        let second = machine
            .account(PAGE_SIZE)
            .unwrap()
            .allocate(PAGE_SIZE)
            .unwrap();
        runner
            .space
            .map(
                GuestVirtualAddress::new(DATA + PAGE_SIZE),
                second,
                0,
                PAGE_SIZE,
                MemoryPermissions::READ_WRITE,
            )
            .unwrap();
        let mut state = A64State::default();
        runner.core.save(&mut state);
        state.write_x(
            A64Register::General(A64GeneralRegister::new(1).unwrap()),
            address,
        );
        let value = 0x112233445566778899aabbccddeeff00;
        assert!(state.set_vector(0, value));
        runner.core.load(&state);
        assert!(runner.invoke(1000, true).is_none());
        assert!(
            matches!(runner.space.read_value(GuestVirtualAddress::new(address), 16).unwrap(), Access::Ready(actual) if actual == value)
        );
    }
}

#[test]
fn plain_cross_page_simd_store_faults_without_publishing_a_writable_prefix() {
    // STR Q0, [X1] must not commit the first eight bytes when the following
    // page is absent or read-only. Exercise the real host fault and callback.
    for readonly in [false, true] {
        let (mut runner, _) = fixture(&[0x3d800020, 0xd4000001]);
        if readonly {
            let next = runner
                .space
                .machine()
                .account(PAGE_SIZE)
                .unwrap()
                .allocate(PAGE_SIZE)
                .unwrap();
            runner
                .space
                .map(
                    GuestVirtualAddress::new(DATA + PAGE_SIZE),
                    next,
                    0,
                    PAGE_SIZE,
                    MemoryPermissions::READ,
                )
                .unwrap();
        }
        let address = GuestVirtualAddress::new(DATA + PAGE_SIZE - 8);
        runner
            .space
            .write_value(address, 8, 0xaaaa_bbbb_cccc_ddddu128)
            .unwrap()
            .into_ready()
            .unwrap();
        let mut state = A64State::default();
        runner.core.save(&mut state);
        state.write_x(
            A64Register::General(A64GeneralRegister::new(1).unwrap()),
            address.get(),
        );
        assert!(state.set_vector(0, 0x112233445566778899aabbccddeeff00));
        runner.core.load(&state);
        let mut worker = NativeWorker::default();
        let boundary = runner
            .core
            .run(&mut worker, 1000, &Timer, &Default::default())
            .unwrap();
        assert!(boundary.memory.is_none());
        let fault = boundary
            .fault
            .expect("cross-page write must report its denied suffix");
        assert_eq!(fault.pc.get(), CODE);
        assert_eq!(fault.kind, nixe_cpu::memory::DataAccessKind::Write);
        assert_eq!(
            runner
                .space
                .read_value(address, 8)
                .unwrap()
                .into_ready()
                .unwrap(),
            0xaaaa_bbbb_cccc_dddd
        );
        if readonly {
            assert_eq!(
                runner
                    .space
                    .read_value(GuestVirtualAddress::new(DATA + PAGE_SIZE), 8)
                    .unwrap()
                    .into_ready()
                    .unwrap(),
                0
            );
        }
        worker.finish().unwrap();
    }
}

#[test]
fn canonical_page_entries_and_monitor_offsets_execute_every_exclusive_width() {
    for (load, store, width) in [
        (0x085f7c22, 0x08037c80, 1),
        (0x485f7c22, 0x48037c80, 2),
        (0x885f7c22, 0x88037c80, 4),
        (0xc85ffc22, 0xc803fc80, 8),
        (0xc87f0c22, 0xc8241420, 16),
    ] {
        let (mut runner, _) = fixture(&[load, 0x14000002, 0xd4200000, store, 0xd4000001]);
        runner.load_exclusive();
        runner.run();
        assert_eq!(
            runner.register(if width == 16 { 4 } else { 3 }),
            0,
            "width={width}"
        );
        assert!(
            matches!(runner.space.read_value(GuestVirtualAddress::new(DATA), width).unwrap(), Access::Ready(value) if value == if width == 16 { (9u128 << 64) | 7 } else { 7 })
        );
    }
}

#[test]
fn plain_alias_stores_and_device_epochs_use_atomic_exclusive_commit() {
    for mode in 0..3 {
        let words: &[u32] = if mode == 2 {
            &[
                0x885f7c22, 0x14000002, 0xd4200000, 0xb9000085, 0x88037c80, 0xd4000001,
            ]
        } else {
            &[0x885f7c22, 0x14000002, 0xd4200000, 0x88037c80, 0xd4000001]
        };
        let (mut runner, epoch) = fixture(words);
        runner.load_exclusive();
        if mode == 0 {
            for value in [1, 0] {
                assert!(matches!(
                    runner
                        .space
                        .write_value(GuestVirtualAddress::new(ALIAS), 4, value)
                        .unwrap(),
                    Access::Ready(())
                ));
            }
        } else if mode == 1 {
            epoch.mark_device_modified();
        }
        runner.run();
        assert_eq!(runner.register(3), u64::from(mode != 0));
        assert!(
            matches!(runner.space.read_value(GuestVirtualAddress::new(DATA), 4).unwrap(), Access::Ready(value) if value == if mode == 0 { 7 } else if mode == 2 { 9 } else { 0 })
        );
    }
}

#[test]
fn cold_exclusive_expiration_completes_without_a_device_download() {
    for (load, store, width, status) in [
        (0x885f7c22, 0x88037c80, 4, 3),
        (0xc87f0c22, 0xc8241420, 16, 4),
    ] {
        let (mut runner, _) = fixture(&[load, 0x14000002, 0xd4200000, store, 0xd4000001]);
        runner.load_exclusive();
        let destination = runner
            .space
            .capture(
                GuestVirtualAddress::new(DATA),
                width,
                MemoryPermissions::WRITE,
            )
            .unwrap();
        runner
            .space
            .machine()
            .publish_device_write(&destination, || Ok(()))
            .unwrap();
        let pending = runner
            .invoke(1000, false)
            .expect("device STXR leaves native execution");
        runner.complete_ram(pending);
        assert_eq!(runner.register(status), 1);
        assert_eq!(runner.register(32), CODE + 16);
        runner
            .space
            .machine()
            .retire_device_pages(&destination)
            .unwrap();
        runner.run();
        assert!(matches!(
            runner
                .space
                .read_value(GuestVirtualAddress::new(DATA), width as usize)
                .unwrap(),
            Access::Ready(0)
        ));
    }
}

#[test]
fn cold_exclusive_rejects_another_byte_address_or_width_before_writing() {
    for mismatch_width in [false, true] {
        let store = if mismatch_width {
            0xc8037c80
        } else {
            0x88037c80
        };
        let (mut runner, _) = fixture(&[0x885f7c22, 0x14000002, 0xd4200000, store, 0xd4000001]);
        runner.load_exclusive();
        let destination = runner
            .space
            .capture(GuestVirtualAddress::new(DATA), 16, MemoryPermissions::WRITE)
            .unwrap();
        runner
            .space
            .machine()
            .publish_device_write(&destination, || Ok(()))
            .unwrap();
        if !mismatch_width {
            runner.core.write_register(4, ALIAS + 8);
        }
        runner.run();
        assert_eq!(runner.register(3), 1);
        runner
            .space
            .machine()
            .retire_device_pages(&destination)
            .unwrap();
        assert!(matches!(
            runner
                .space
                .read_value(GuestVirtualAddress::new(DATA), 16)
                .unwrap(),
            Access::Ready(0)
        ));
    }
}

struct Endpoint(std::sync::mpsc::Sender<nixe_memory::DeviceAccessRequest>);
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
fn acquire_loads_from_readonly_ram_finish_in_one_native_invocation() {
    // LDAR W2,[X1]; ADD X3,X3,X2; SUBS X0,X0,#1; B.NE loop; SVC.
    // https://developer.arm.com/documentation/ddi0602/latest/Base-Instructions/LDAR--Load-Acquire-Register-
    let (mut runner, _) = fixture(&[0x88dffc22, 0x8b020063, 0xf1000400, 0x54ffffa1, 0xd4000001]);
    assert!(matches!(
        runner
            .space
            .write_value(GuestVirtualAddress::new(DATA), 4, 17)
            .unwrap(),
        Access::Ready(())
    ));
    runner
        .space
        .protect(
            GuestVirtualAddress::new(DATA),
            PAGE_SIZE,
            MemoryPermissions::READ,
        )
        .unwrap();
    runner.core.write_register(0, 16);
    runner.core.write_register(3, 0);
    assert!(runner.invoke(1000, true).is_none());
    assert_eq!(runner.register(0), 0);
    assert_eq!(runner.register(2), 17);
    assert_eq!(runner.register(3), 16 * 17);
}

#[test]
fn data_cache_clean_does_not_download_device_data_before_an_actual_load() {
    // DC CVAC,X1; SVC; LDR X2,[X1]; SVC.
    let (mut runner, _) = fixture(&[0xd50b7a21, 0xd4000001, 0xf9400022, 0xd4000001]);
    let machine = runner.space.machine().clone();
    let ranges = runner
        .space
        .capture(GuestVirtualAddress::new(DATA), 8, MemoryPermissions::WRITE)
        .unwrap();
    machine.write_resolved(&ranges, &[0x22; 8]).unwrap();
    machine.publish_device_write(&ranges, || Ok(())).unwrap();
    let (send, receive) = std::sync::mpsc::channel();
    let endpoint: Arc<dyn nixe_memory::DeviceAccessEndpoint> = Arc::new(Endpoint(send));
    machine.set_device_endpoint(&endpoint);
    assert!(runner.invoke(1000, true).is_none());
    assert!(receive.try_recv().is_err());
    let mut pending = runner.invoke(1000, true).unwrap();
    assert!(pending.poll(&machine, &Timer).is_none());
    let request = receive
        .recv_timeout(std::time::Duration::from_secs(5))
        .unwrap();
    assert_eq!(request.bytes(), 8);
    let result = machine
        .apply_resolved_access(request.ranges(), request.operation())
        .unwrap();
    request.finish(Ok(result));
    runner
        .core
        .complete(pending.poll(&machine, &Timer).unwrap().unwrap())
        .unwrap();
    assert_eq!(runner.register(2), 0x2222222222222222);
    assert!(runner.invoke(1000, true).is_none());
    assert!(receive.try_recv().is_err());
}

#[test]
fn data_cache_clean_reports_an_unmapped_operand_at_the_original_instruction() {
    let (mut runner, _) = fixture(&[0xd50b7a21, 0xd2800540, 0xd4000001]);
    runner.core.write_register(0, 17);
    runner.core.write_register(1, 0x6000);
    let boundary = runner
        .core
        .run(
            &mut NativeWorker::default(),
            1000,
            &Timer,
            &Default::default(),
        )
        .unwrap();
    let fault = boundary.fault.unwrap();
    assert_eq!(fault.pc.get(), CODE);
    assert_eq!(fault.address.get(), 0x6000);
    assert!(matches!(fault.cause, nixe_memory::MemoryError::Unmapped(_)));
    assert_eq!(runner.register(0), 17);
}

#[test]
fn unaligned_simd_load_waits_for_device_bytes_on_its_second_page() {
    let (mut runner, _) = fixture(&[0x3dc00020, 0xd4000001]); // LDR Q0,[X1]; SVC
    let machine = runner.space.machine().clone();
    let second = machine
        .account(PAGE_SIZE)
        .unwrap()
        .allocate(PAGE_SIZE)
        .unwrap();
    runner
        .space
        .map(
            GuestVirtualAddress::new(DATA + PAGE_SIZE),
            second,
            0,
            PAGE_SIZE,
            MemoryPermissions::READ_WRITE,
        )
        .unwrap();
    assert!(matches!(
        runner
            .space
            .write_value(
                GuestVirtualAddress::new(DATA + PAGE_SIZE - 8),
                8,
                0x1111111111111111
            )
            .unwrap(),
        Access::Ready(())
    ));
    let ranges = runner
        .space
        .capture(
            GuestVirtualAddress::new(DATA + PAGE_SIZE),
            8,
            MemoryPermissions::WRITE,
        )
        .unwrap();
    machine.write_resolved(&ranges, &[0x22; 8]).unwrap();
    machine.publish_device_write(&ranges, || Ok(())).unwrap();
    let (send, receive) = std::sync::mpsc::channel();
    let endpoint: Arc<dyn nixe_memory::DeviceAccessEndpoint> = Arc::new(Endpoint(send));
    machine.set_device_endpoint(&endpoint);
    runner.core.write_register(1, DATA + PAGE_SIZE - 8);
    let mut pending = runner.invoke(1000, true).unwrap();
    assert!(pending.poll(&machine, &Timer).is_none());
    let request = receive
        .recv_timeout(std::time::Duration::from_secs(5))
        .unwrap();
    assert_eq!(request.bytes(), 16);
    let result = machine
        .apply_resolved_access(request.ranges(), request.operation())
        .unwrap();
    request.finish(Ok(result));
    runner
        .core
        .complete(pending.poll(&machine, &Timer).unwrap().unwrap())
        .unwrap();
    let mut state = A64State::default();
    runner.core.save(&mut state);
    assert_eq!(state.vector(0), Some(0x22222222222222221111111111111111));
    assert!(runner.invoke(1000, true).is_none());
    assert!(receive.try_recv().is_err());
}

#[test]
fn device_exclusive_load_and_store_complete_once_outside_cpp() {
    for (load, store, width, status) in [
        (0x885f7c22, 0x88037c80, 4, 3),
        (0xc87f0c22, 0xc8241420, 16, 4),
    ] {
        let (mut runner, _) = fixture(&[load, 0x14000002, 0xd4200000, store, 0xd4000001]);
        let machine = runner.space.machine().clone();
        let (sender, receiver) = std::sync::mpsc::channel();
        let endpoint: Arc<dyn nixe_memory::DeviceAccessEndpoint> = Arc::new(Endpoint(sender));
        machine.set_device_endpoint(&endpoint);
        let destination = runner
            .space
            .capture(
                GuestVirtualAddress::new(DATA),
                width,
                MemoryPermissions::WRITE,
            )
            .unwrap();
        machine
            .publish_device_write(&destination, || Ok(()))
            .unwrap();
        for load in [true, false] {
            let mut pending = runner
                .invoke(1000, false)
                .expect("exclusive operation captures before leaving native execution");
            assert!(pending.poll(&machine, &Timer).is_none());
            let request = receiver
                .recv_timeout(std::time::Duration::from_secs(5))
                .unwrap();
            assert_eq!(request.bytes() as u64, width);
            assert!(
                matches!(request.operation(), nixe_memory::AccessOperation::Atomic(operation)
                if matches!(operation.operation, nixe_memory::AtomicOperation::LoadExclusive) == load)
            );
            let result = machine
                .apply_resolved_access(request.ranges(), request.operation())
                .unwrap();
            request.finish(Ok(result));
            let completed = pending.poll(&machine, &Timer).unwrap().unwrap();
            assert!(matches!(
                runner.core.complete(completed).unwrap().0,
                nixe_cpu_interpreter::InstructionStep::Continue
            ));
            if load {
                assert_eq!(runner.register(2), 0);
                assert!(runner.invoke(1, false).is_none());
                assert_eq!(runner.register(32), CODE + 12);
            }
        }
        assert_eq!(runner.register(status), 0);
        machine.retire_device_pages(&destination).unwrap();
        runner.run();
        assert!(
            matches!(runner.space.read_value(GuestVirtualAddress::new(DATA), width as usize).unwrap(), Access::Ready(value) if value == if width == 16 { (9u128 << 64) | 7 } else { 7 })
        );
        assert!(receiver.try_recv().is_err());
    }
}

fn finish_ram_code(core: &mut Core, worker: &mut NativeWorker) -> crate::execution::Boundary {
    for _ in 0..8 {
        let boundary = core.run(worker, 1000, &Timer, &Default::default()).unwrap();
        if boundary.code.is_empty() {
            return boundary;
        }
        assert!(boundary.memory.is_none());
        for capture in boundary.code {
            let mut read = capture.start(core.space()).unwrap();
            let resolved = read
                .poll()
                .expect("RAM code completes without parking")
                .unwrap();
            core.install_code(resolved).unwrap();
        }
    }
    panic!("code preparation failed to reach native execution");
}

#[test]
fn cold_code_arms_tracking_outside_cpp_and_invalidates_failed_fetch_terminals() {
    // MOV X0, #11; SVC #0.
    let (mut runner, _) = fixture_with_tracking(&[0xd2800160, 0xd4000001], false);
    let mut worker = NativeWorker::default();
    assert_eq!(finish_ram_code(&mut runner.core, &mut worker).event.kind, 2);
    assert_eq!(runner.register(0), 11);
    assert_eq!(
        runner.space.mapping_observation_count(),
        0,
        "ordinary RAM code must not accumulate owned-word cache entries"
    );
    let code = runner
        .space
        .capture(
            GuestVirtualAddress::new(CODE),
            4,
            MemoryPermissions::EXECUTE,
        )
        .unwrap();
    runner
        .space
        .machine()
        .write_resolved(&code, &0xd28001a0u32.to_le_bytes())
        .unwrap(); // MOV X0, #13.
    runner.core.write_register(32, CODE);
    assert_eq!(finish_ram_code(&mut runner.core, &mut worker).event.kind, 2);
    assert_eq!(runner.register(0), 13);
    worker.finish().unwrap();
}

#[test]
fn pending_code_uses_original_physical_input_and_cannot_populate_a_remapped_va() {
    let (mut runner, _) = fixture_with_tracking(&[0xd2800160, 0xd4000001], false);
    let machine = runner.space.machine().clone();
    let (sender, receiver) = std::sync::mpsc::channel();
    let endpoint: Arc<dyn nixe_memory::DeviceAccessEndpoint> = Arc::new(Endpoint(sender));
    machine.set_device_endpoint(&endpoint);
    let old = runner
        .space
        .capture(
            GuestVirtualAddress::new(CODE),
            PAGE_SIZE,
            MemoryPermissions::EXECUTE,
        )
        .unwrap();
    machine.publish_device_write(&old, || Ok(())).unwrap();
    let mut worker = NativeWorker::default();
    let boundary = runner
        .core
        .run(&mut worker, 1000, &Timer, &Default::default())
        .unwrap();
    assert_eq!(boundary.code.len(), 1);
    let mut read = boundary
        .code
        .into_iter()
        .next()
        .unwrap()
        .start(&runner.space)
        .unwrap();
    assert!(read.poll().is_none());
    let request = receiver.recv().unwrap();
    let captured_allocation = request.ranges()[0].allocation().identity();
    runner
        .space
        .unmap(GuestVirtualAddress::new(CODE), PAGE_SIZE)
        .unwrap();
    // The data allocation is still guest owned; the old code's pending input
    // retains physical RAM but no guest charge.
    assert_eq!(machine.statistics().retained_bytes, PAGE_SIZE);
    drop(old);
    let allocation = machine
        .account(PAGE_SIZE)
        .unwrap()
        .allocate(PAGE_SIZE)
        .unwrap();
    assert_ne!(allocation.identity(), captured_allocation);
    machine
        .write_resolved(
            &[AllocationRange::new(allocation.pin(), 0, 8).unwrap()],
            &[0xd28001a0u32.to_le_bytes(), 0xd4000001u32.to_le_bytes()].concat(),
        )
        .unwrap();
    runner
        .space
        .map(
            GuestVirtualAddress::new(CODE),
            allocation,
            0,
            PAGE_SIZE,
            MemoryPermissions::READ_EXECUTE,
        )
        .unwrap();
    let result = machine
        .apply_resolved_access(request.ranges(), request.operation())
        .unwrap();
    request.finish(Ok(result));
    runner
        .core
        .install_code(read.poll().unwrap().unwrap())
        .unwrap();
    assert_eq!(finish_ram_code(&mut runner.core, &mut worker).event.kind, 2);
    assert_eq!(runner.register(0), 13);
    assert_eq!(machine.statistics().retained_bytes, 0);
    assert!(receiver.try_recv().is_err());
    worker.finish().unwrap();
}

#[test]
fn a_captured_store_retains_its_original_opcode_and_physical_target() {
    let (mut runner, _) = fixture(&[0xb9000020, 0xd4000001]); // STR W0, [X1].
    let physical = runner
        .space
        .capture(
            GuestVirtualAddress::new(DATA),
            PAGE_SIZE,
            MemoryPermissions::WRITE,
        )
        .unwrap();
    let observed = nixe_memory::Observation::default();
    runner
        .space
        .machine()
        .access_captured(
            &physical,
            nixe_memory::AccessOperation::ReadObserved(observed.clone()),
        )
        .unwrap()
        .into_ready()
        .unwrap();
    let captured = runner.invoke(1000, false).unwrap();
    let code = runner
        .space
        .capture(
            GuestVirtualAddress::new(CODE),
            4,
            MemoryPermissions::EXECUTE,
        )
        .unwrap();
    runner
        .space
        .machine()
        .write_resolved(&code, &0xb9000025u32.to_le_bytes())
        .unwrap(); // STR W5, [X1].
    let old_identity = physical[0].allocation().identity();
    runner
        .space
        .unmap(GuestVirtualAddress::new(DATA), PAGE_SIZE)
        .unwrap();
    runner
        .space
        .map(
            GuestVirtualAddress::new(DATA),
            runner
                .space
                .machine()
                .account(PAGE_SIZE)
                .unwrap()
                .allocate(PAGE_SIZE)
                .unwrap(),
            0,
            PAGE_SIZE,
            MemoryPermissions::READ_WRITE,
        )
        .unwrap();
    assert_ne!(
        runner
            .space
            .translate(GuestVirtualAddress::new(DATA), MemoryPermissions::READ)
            .unwrap()
            .allocation,
        old_identity
    );
    runner.complete_ram(captured);
    assert!(observed.is_invalid());
    assert!(matches!(
        runner
            .space
            .read_value(GuestVirtualAddress::new(ALIAS), 4)
            .unwrap(),
        Access::Ready(7)
    ));
    assert!(matches!(
        runner
            .space
            .read_value(GuestVirtualAddress::new(DATA), 4)
            .unwrap(),
        Access::Ready(0)
    ));
}

#[test]
fn pair_and_simd_suffixes_survive_remapping_without_replaying_committed_stores() {
    for (word, element, total) in [(0xa9001420u32, 8u64, 16u64), (0x4c9fa020, 16, 32)] {
        let (mut runner, _) = fixture(&[word, 0xd4000001]);
        let machine = runner.space.machine().clone();
        let second = machine
            .account(PAGE_SIZE)
            .unwrap()
            .allocate(PAGE_SIZE)
            .unwrap();
        let second_address = DATA + PAGE_SIZE;
        runner
            .space
            .map(
                GuestVirtualAddress::new(second_address),
                second.clone(),
                0,
                PAGE_SIZE,
                MemoryPermissions::READ_WRITE,
            )
            .unwrap();
        let tail = runner
            .space
            .capture(
                GuestVirtualAddress::new(second_address),
                element,
                MemoryPermissions::WRITE,
            )
            .unwrap();
        machine
            .access_captured(
                &tail,
                nixe_memory::AccessOperation::ReadObserved(nixe_memory::Observation::default()),
            )
            .unwrap()
            .into_ready()
            .unwrap();
        let start = DATA + PAGE_SIZE - element;
        let mut state = A64State::default();
        runner.core.save(&mut state);
        state.write_x(
            A64Register::General(A64GeneralRegister::new(1).unwrap()),
            start,
        );
        if element == 16 {
            assert!(state.set_vector(0, 7));
            assert!(state.set_vector(1, 9));
        }
        runner.core.load(&state);
        let mut pending = runner.invoke(1000, false).unwrap();
        assert!(matches!(
            runner
                .space
                .read_value(
                    GuestVirtualAddress::new(ALIAS + PAGE_SIZE - element),
                    element as usize
                )
                .unwrap(),
            Access::Ready(7)
        ));
        // Replace both VAs. The already committed first store may be overwritten
        // independently; the pending suffix must still target the old second allocation.
        runner
            .space
            .unmap(GuestVirtualAddress::new(DATA), PAGE_SIZE)
            .unwrap();
        runner
            .space
            .map(
                GuestVirtualAddress::new(DATA),
                second,
                0,
                PAGE_SIZE,
                MemoryPermissions::READ,
            )
            .unwrap();
        runner
            .space
            .unmap(GuestVirtualAddress::new(second_address), PAGE_SIZE)
            .unwrap();
        runner
            .space
            .map_alias(
                GuestVirtualAddress::new(ALIAS),
                GuestVirtualAddress::new(second_address),
                PAGE_SIZE,
                MemoryPermissions::READ_WRITE,
                MemoryPermissions::READ_WRITE,
            )
            .unwrap();
        assert!(matches!(
            runner
                .space
                .write_value(
                    GuestVirtualAddress::new(ALIAS + PAGE_SIZE - element),
                    element as usize,
                    99
                )
                .unwrap(),
            Access::Ready(())
        ));
        let completed = pending.poll(&machine, &Timer).unwrap().unwrap();
        runner.core.complete(completed).unwrap();
        assert!(matches!(
            runner
                .space
                .read_value(
                    GuestVirtualAddress::new(ALIAS + PAGE_SIZE - element),
                    element as usize
                )
                .unwrap(),
            Access::Ready(99)
        ));
        assert!(matches!(
            runner
                .space
                .read_value(GuestVirtualAddress::new(DATA), element as usize)
                .unwrap(),
            Access::Ready(9)
        ));
        assert!(matches!(
            runner
                .space
                .read_value(GuestVirtualAddress::new(second_address), element as usize)
                .unwrap(),
            Access::Ready(0)
        ));
        assert_eq!(
            runner.register(1),
            if element == 16 { start + total } else { start }
        );
        runner.run();
    }
}

#[test]
fn device_rmw_is_one_architectural_operation_and_resumes_without_native_retry() {
    // LDADD W0, W2, [X1]; SVC #0.
    let (mut runner, _) = fixture(&[0xb8200022, 0xd4000001]);
    let machine = runner.space.machine().clone();
    assert!(matches!(
        runner
            .space
            .write_value(GuestVirtualAddress::new(DATA), 4, 10)
            .unwrap(),
        Access::Ready(())
    ));
    let destination = runner
        .space
        .capture(
            GuestVirtualAddress::new(DATA),
            4,
            MemoryPermissions::READ_WRITE,
        )
        .unwrap();
    let (sender, receiver) = std::sync::mpsc::channel();
    let endpoint: Arc<dyn nixe_memory::DeviceAccessEndpoint> = Arc::new(Endpoint(sender));
    machine.set_device_endpoint(&endpoint);
    machine
        .publish_device_write(&destination, || Ok(()))
        .unwrap();
    let mut pending = runner.invoke(1000, false).unwrap();
    assert!(pending.poll(&machine, &Timer).is_none());
    let request = receiver
        .recv_timeout(std::time::Duration::from_secs(5))
        .unwrap();
    assert!(matches!(
        request.operation(),
        nixe_memory::AccessOperation::Atomic(nixe_memory::AtomicAccess {
            operation: nixe_memory::AtomicOperation::ReadModifyWrite {
                transform: nixe_memory::AtomicTransform::Add,
                operand: 7
            },
            ..
        })
    ));
    let result = machine
        .apply_resolved_access(request.ranges(), request.operation())
        .unwrap();
    request.finish(Ok(result));
    runner
        .core
        .complete(pending.poll(&machine, &Timer).unwrap().unwrap())
        .unwrap();
    assert_eq!(runner.register(2), 10);
    machine.retire_device_pages(&destination).unwrap();
    runner.run();
    assert!(matches!(
        runner
            .space
            .read_value(GuestVirtualAddress::new(DATA), 4)
            .unwrap(),
        Access::Ready(17)
    ));
    assert!(receiver.try_recv().is_err());
}

#[test]
fn preemption_before_native_entry_is_consumed_without_running_guest_code() {
    let (mut runner, _) = fixture(&[0xd2800160, 0xd4000001]);
    runner
        .core
        .control()
        .request(nixe_cpu::execution::ControlRequest::Preempt);
    let mut worker = NativeWorker::default();
    let boundary = runner
        .core
        .run(&mut worker, 1000, &Timer, &Default::default())
        .unwrap();
    assert!(
        boundary
            .control
            .unwrap()
            .contains(nixe_cpu::execution::ControlRequest::Preempt)
    );
    assert_eq!(boundary.ticks, 0);
    assert_eq!(runner.register(32), CODE);
    assert_eq!(runner.register(0), 7);
    worker.finish().unwrap();
    runner.run();
    assert_eq!(runner.register(0), 11);
}

#[test]
fn tracking_faults_restore_direct_ram_access_after_completion() {
    let (mut runner, _) = fixture(&[0xb9000020, 0xd4000001]);
    let physical = runner
        .space
        .capture(GuestVirtualAddress::new(DATA), 4, MemoryPermissions::WRITE)
        .unwrap();
    let observation = nixe_memory::Observation::default();
    runner
        .space
        .machine()
        .access_captured(
            &physical,
            nixe_memory::AccessOperation::ReadObserved(observation.clone()),
        )
        .unwrap()
        .into_ready()
        .unwrap();
    let pending = runner.invoke(1000, false).unwrap();
    runner.complete_ram(pending);
    assert!(observation.is_invalid());
    runner.core.write_register(32, CODE);
    runner.core.write_register(0, 19);
    let mut worker = NativeWorker::default();
    let boundary = runner
        .core
        .run(&mut worker, 1000, &Timer, &Default::default())
        .unwrap();
    assert_eq!(boundary.event.kind, 2);
    assert!(
        boundary.memory.is_none(),
        "a consumed observation cannot permanently force checked completion"
    );
    assert!(boundary.fault.is_none());
    assert!(matches!(
        runner
            .space
            .read_value(GuestVirtualAddress::new(DATA), 4)
            .unwrap(),
        Access::Ready(19)
    ));
    worker.finish().unwrap();
}

#[test]
fn native_guest_data_fault_preserves_location_and_does_not_poison_memory() {
    let (mut runner, _) = fixture(&[0xb9400020, 0xd4000001]); // LDR W0, [X1].
    runner.core.write_register(1, 0x6000);
    let mut worker = NativeWorker::default();
    let boundary = runner
        .core
        .run(&mut worker, 1000, &Timer, &Default::default())
        .unwrap();
    assert!(boundary.memory.is_none());
    let fault = boundary
        .fault
        .expect("unmapped load must retain its typed fault");
    assert_eq!(fault.pc.get(), CODE);
    assert_eq!(fault.address.get(), 0x6000);
    assert_eq!(fault.kind, nixe_cpu::memory::DataAccessKind::Read);
    runner
        .space
        .map_alias(
            GuestVirtualAddress::new(DATA),
            GuestVirtualAddress::new(0x6000),
            PAGE_SIZE,
            MemoryPermissions::READ_WRITE,
            MemoryPermissions::READ_WRITE,
        )
        .unwrap();
    assert!(matches!(
        runner
            .space
            .write_value(GuestVirtualAddress::new(DATA), 4, 77)
            .unwrap(),
        Access::Ready(())
    ));
    let boundary = runner
        .core
        .run(&mut worker, 1000, &Timer, &Default::default())
        .unwrap();
    assert_eq!(boundary.event.kind, 2);
    assert!(boundary.memory.is_none());
    assert!(boundary.fault.is_none());
    assert_eq!(runner.register(0), 77);
    worker.finish().unwrap();
}

#[test]
fn contiguous_simd_lanes_use_one_owned_operation_for_the_instruction() {
    let bytes: Vec<u8> = (0..32).collect();
    for (word, load) in [(0x4c9fa020u32, false), (0x4cdfa020, true)] {
        let (mut runner, _) = fixture(&[word, 0xd4000001]);
        let machine = runner.space.machine().clone();
        let target = runner
            .space
            .capture(
                GuestVirtualAddress::new(DATA),
                32,
                MemoryPermissions::READ_WRITE,
            )
            .unwrap();
        if load {
            machine.write_resolved(&target, &bytes).unwrap();
        } else {
            let mut state = A64State::default();
            runner.core.save(&mut state);
            assert!(state.set_vector(0, u128::from_le_bytes(bytes[..16].try_into().unwrap())));
            assert!(state.set_vector(1, u128::from_le_bytes(bytes[16..].try_into().unwrap())));
            runner.core.load(&state);
        }
        let (sender, receiver) = std::sync::mpsc::channel();
        let endpoint: Arc<dyn nixe_memory::DeviceAccessEndpoint> = Arc::new(Endpoint(sender));
        machine.set_device_endpoint(&endpoint);
        machine.publish_device_write(&target, || Ok(())).unwrap();
        let mut pending = runner.invoke(1000, false).unwrap();
        assert!(pending.poll(&machine, &Timer).is_none());
        let request = receiver
            .recv_timeout(std::time::Duration::from_secs(5))
            .unwrap();
        assert_eq!(request.bytes(), 32);
        assert!(
            matches!(
                request.operation(),
                nixe_memory::AccessOperation::ReadNative
            ) == load
        );
        let result = machine
            .apply_resolved_access(request.ranges(), request.operation())
            .unwrap();
        request.finish(Ok(result));
        runner
            .core
            .complete(pending.poll(&machine, &Timer).unwrap().unwrap())
            .unwrap();
        assert_eq!(runner.register(1), DATA + 32);
        if load {
            let mut state = A64State::default();
            runner.core.save(&mut state);
            assert_eq!(
                state.vector(0),
                Some(u128::from_le_bytes(bytes[..16].try_into().unwrap()))
            );
            assert_eq!(
                state.vector(1),
                Some(u128::from_le_bytes(bytes[16..].try_into().unwrap()))
            );
        }
        let mut actual = vec![0; 32];
        machine.capture_stream(&target, &mut actual).unwrap();
        assert_eq!(actual, bytes);
        assert!(
            receiver.try_recv().is_err(),
            "lane completion must not issue scalar owner requests"
        );
        machine.retire_device_pages(&target).unwrap();
        runner.run();
    }
}

#[test]
fn mixed_page_reads_reuse_coherent_ram_without_repeated_instruction_continuations() {
    // LDR W0,[X1]; ADD X3,X3,X0; SUBS X2,X2,#1; B.NE loop; SVC.
    let (mut runner, _) = fixture(&[0xb9400020, 0x8b000063, 0xf1000442, 0x54ffffa1, 0xd4000001]);
    let machine = runner.space.machine().clone();
    let data = runner
        .space
        .capture(
            GuestVirtualAddress::new(DATA),
            PAGE_SIZE,
            MemoryPermissions::READ_WRITE,
        )
        .unwrap();
    let reusable = data[0].slice(64, 448).unwrap();
    let field = data[0].slice(128, 4).unwrap();
    machine
        .write_resolved(std::slice::from_ref(&field), &7u32.to_le_bytes())
        .unwrap();
    let (sender, receiver) = std::sync::mpsc::channel();
    let endpoint: Arc<dyn nixe_memory::DeviceAccessEndpoint> = Arc::new(Endpoint(sender));
    machine.set_device_endpoint(&endpoint);
    machine
        .publish_device_write(&[data[0].slice(0, 64).unwrap()], || Ok(()))
        .unwrap();
    runner.core.write_register(1, DATA + 128);
    for value in [7u64, 9] {
        if value == 9 {
            machine
                .write_resolved(std::slice::from_ref(&field), &9u32.to_le_bytes())
                .unwrap();
        }
        runner.core.write_register(32, CODE);
        runner.core.write_register(2, 1000);
        runner.core.write_register(3, 0);
        if value == 7 {
            let mut pending = runner
                .invoke(100_000, false)
                .expect("cold read needs its owner");
            assert!(pending.poll(&machine, &Timer).is_none());
            let request = receiver
                .recv_timeout(std::time::Duration::from_secs(2))
                .unwrap();
            assert!(matches!(
                request.operation(),
                nixe_memory::AccessOperation::ReadNative
            ));
            let result = machine
                .read_native_resolved(request.ranges(), Some(&reusable))
                .unwrap();
            request.finish(Ok(result));
            runner
                .core
                .complete(pending.poll(&machine, &Timer).unwrap().unwrap())
                .unwrap();
        }
        let mut worker = NativeWorker::default();
        let boundary = runner
            .core
            .run(&mut worker, 100_000, &Timer, &Default::default())
            .unwrap();
        assert!(
            boundary.memory.is_none(),
            "warm reads must stay inside native execution"
        );
        assert!(boundary.fault.is_none());
        assert_eq!(boundary.event.kind, 2);
        assert_eq!(runner.register(3), value * 1000);
        assert!(receiver.try_recv().is_err());
        worker.finish().unwrap();
    }
    runner
        .space
        .map_alias(
            GuestVirtualAddress::new(DATA),
            GuestVirtualAddress::new(0x9000),
            PAGE_SIZE,
            MemoryPermissions::READ_WRITE,
            MemoryPermissions::READ_WRITE,
        )
        .unwrap();
    runner.core.write_register(32, CODE);
    runner.core.write_register(1, 0x9000 + 128);
    runner.core.write_register(2, 1000);
    runner.core.write_register(3, 0);
    let mut worker = NativeWorker::default();
    let boundary = runner
        .core
        .run(&mut worker, 100_000, &Timer, &Default::default())
        .unwrap();
    assert!(
        boundary.memory.is_none(),
        "physical read proof also covers a current alias"
    );
    assert_eq!(boundary.event.kind, 2);
    assert_eq!(runner.register(3), 9000);
    assert!(receiver.try_recv().is_err());
    worker.finish().unwrap();
}

#[test]
fn repeated_device_stores_fault_once_and_reopened_ram_stays_direct() {
    use crate::metrics::{Counter, test_snapshot};
    // STR W0,[X1]; ADD X0,X0,#1; SUBS X2,X2,#1; B.NE -12; SVC #0.
    let (mut runner, _) = fixture(&[0xb9000020, 0x91000400, 0xf1000442, 0x54ffffa1, 0xd4000001]);
    let machine = runner.space.machine().clone();
    let ranges = runner
        .space
        .capture(GuestVirtualAddress::new(DATA), 4, MemoryPermissions::WRITE)
        .unwrap();
    machine.publish_device_write(&ranges, || Ok(())).unwrap();
    machine.enable_cpu_write_collection().unwrap();
    for (iteration, value) in [(0, 1000), (1, 2000), (2, 3000)] {
        if iteration == 2 {
            machine.collect_cpu_writes().unwrap();
            machine.retire_device_pages(&ranges).unwrap();
        }
        runner.core.write_register(0, value);
        runner.core.write_register(2, 128);
        runner.core.write_register(32, CODE);
        let before = test_snapshot();
        assert!(runner.invoke(2000, true).is_none());
        let after = test_snapshot();
        let faults =
            after[Counter::FaultDispatches as usize] - before[Counter::FaultDispatches as usize];
        assert!(
            faults <= u64::from(iteration == 0),
            "repeated protection faults: {faults}"
        );
        if iteration == 2 {
            assert_eq!(
                after[Counter::WriteCallbacks as usize] - before[Counter::WriteCallbacks as usize],
                0,
                "a previously faulted store must return to direct RAM instructions"
            );
        }
        let mut bytes = [0; 4];
        machine.capture_stream(&ranges, &mut bytes).unwrap();
        assert_eq!(u32::from_le_bytes(bytes), value as u32 + 127);
    }
    machine.disable_cpu_write_collection();
}

#[test]
fn faulted_store_sites_preserve_write_only_guest_permissions() {
    use crate::metrics::{Counter, test_snapshot};
    let (mut runner, _) = fixture(&[0xb9000020, 0xd4000001]);
    runner
        .space
        .protect(
            GuestVirtualAddress::new(DATA),
            PAGE_SIZE,
            MemoryPermissions::WRITE,
        )
        .unwrap();
    for (iteration, value) in [(0, 37), (1, 42)] {
        runner.core.write_register(0, value);
        runner.core.write_register(32, CODE);
        let before = test_snapshot();
        assert!(runner.invoke(100, true).is_none());
        let after = test_snapshot();
        let faults =
            after[Counter::FaultDispatches as usize] - before[Counter::FaultDispatches as usize];
        assert!(
            faults <= u64::from(iteration == 0),
            "write-only fallback repeated a host fault: {faults}"
        );
        assert_eq!(
            runner
                .space
                .read_value(GuestVirtualAddress::new(ALIAS), 4)
                .unwrap()
                .into_ready()
                .unwrap(),
            value as u128
        );
    }
}
