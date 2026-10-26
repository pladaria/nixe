//! Native process execution and complete architectural context ownership.
use nixe_cpu::execution::{ArchitecturalTimer, CpuExit, TimerSnapshot, VcpuEventState};
use nixe_cpu::platform::TargetPlatform;
use nixe_cpu::profile::ProcessCpuContext;
use nixe_cpu::state::a64::{A64GeneralRegister, A64Register, A64State};
use nixe_cpu_jit::{JitProcess, JitRunRequest, JitThread, ThreadState};
use nixe_memory::{
    AddressSpace, AddressWidth, AllocationRange, GuestVirtualAddress, Machine, MemoryPermissions,
    PAGE_SIZE,
};
use std::sync::Arc;
const CODE: GuestVirtualAddress = GuestVirtualAddress::new(0x1000);
struct FixedTimer;
impl ArchitecturalTimer for FixedTimer {
    fn snapshot(&self) -> TimerSnapshot {
        TimerSnapshot {
            counter: 0,
            frequency: 19_200_000,
        }
    }
}
fn fixture(code: &[u32]) -> (Arc<AddressSpace>, JitThread) {
    let machine = Machine::new(2 * PAGE_SIZE).unwrap();
    let account = machine.account(2 * PAGE_SIZE).unwrap();
    let executable = account.allocate(PAGE_SIZE).unwrap();
    let data = account.allocate(PAGE_SIZE).unwrap();
    let bytes: Vec<_> = code.iter().flat_map(|word| word.to_le_bytes()).collect();
    machine
        .write_resolved(
            &[AllocationRange::new(executable.pin(), 0, bytes.len() as u64).unwrap()],
            &bytes,
        )
        .unwrap();
    let memory = AddressSpace::new(machine, AddressWidth::Bits32).unwrap();
    memory
        .map(
            CODE,
            executable,
            0,
            PAGE_SIZE,
            MemoryPermissions::READ_EXECUTE,
        )
        .unwrap();
    memory
        .map(
            GuestVirtualAddress::new(0x2000),
            data,
            0,
            PAGE_SIZE,
            MemoryPermissions::READ_WRITE,
        )
        .unwrap();
    let process = JitProcess::new(
        ProcessCpuContext::for_platform(TargetPlatform::Switch1, memory.identity()),
        memory.clone(),
        19_200_000,
    )
    .unwrap();
    (memory, process.create_thread().unwrap())
}
fn x(index: u8) -> A64Register {
    A64Register::General(A64GeneralRegister::new(index).unwrap())
}
fn a64_state() -> A64State {
    let mut state = A64State::default();
    state.set_pc(CODE.get());
    state
}

fn pattern(seed: u64) -> A64State {
    let mut state = a64_state();
    for i in 0..31 {
        state.write_x(
            x(i),
            seed.wrapping_mul(0x123456789abcdef)
                .rotate_left(u32::from(i)),
        );
    }
    state.write_x(A64Register::StackPointer, 0x8000 + seed * 16);
    for i in 0..32 {
        state.set_vector(
            i,
            (u128::from(seed + u64::from(i)) << 96) | u128::from(0xfedcba9876543210_u64 ^ seed),
        );
    }
    state.set_tpidr_el0(0xabc000 + seed);
    state.set_tpidrro_el0_from_runtime(0xdef000 + seed);
    state.set_nzcv(nixe_cpu::state::a64::Nzcv::from_bits((seed as u32) << 28));
    state.set_fpcr(((seed as u32) & 3) << 22);
    state.set_fpsr(0x08000000 | ((seed as u32) & 0x1f));
    state
}

fn run(jit: &mut JitThread, state: &mut ThreadState) -> nixe_cpu::execution::ExecutionReport {
    jit.run_slice(
        &mut nixe_cpu_direct_memory::NativeWorker::default(),
        JitRunRequest {
            state,
            timer: &FixedTimer,
            events: &VcpuEventState::default(),
        },
    )
    .unwrap()
}

#[test]
fn direct_memory_exclusives_simd_fp_tls_and_system_registers() {
    // STR/LDR, LDXR/ADD/STXR, FP and SIMD, TLS/cache/timer registers, SVC #0x42.
    let code = [
        0xf9000020, 0xf9400022, 0xc85f7c23, 0x91000463, 0xc8047c23, 0xf9400025, 0x1e6e1000,
        0x1e602801, 0x4f02e442, 0xd53bd046, 0xd53bd067, 0xd53be008, 0xd53be029, 0xd53b002a,
        0xd53b00eb, 0xd4000841,
    ];
    let (_memory, mut jit) = fixture(&code);
    let mut state: ThreadState = a64_state().into();
    state.write_x(x(0), 41);
    state.write_x(x(1), 0x2000);
    state.set_tpidr_el0(0xabcdef);
    state.set_tpidrro_el0_from_runtime(0x123456);
    let report = run(&mut jit, &mut state);
    assert!(matches!(
        report.stop,
        CpuExit::SupervisorCall {
            immediate: 0x42,
            ..
        }
    ));
    assert_eq!(state.pc(), CODE.get() + (code.len() as u64 - 1) * 4);
    assert_eq!(state.read_x(x(2)), 41);
    assert_eq!(state.read_x(x(4)), 0);
    assert_eq!(state.read_x(x(5)), 42);
    assert_eq!(state.vector(1), Some(u128::from(2.0_f64.to_bits())));
    assert_eq!(state.vector(2), Some(u128::from_le_bytes([0x42; 16])));
    assert_eq!(state.read_x(x(6)), 0xabcdef);
    assert_eq!(state.read_x(x(7)), 0x123456);
    assert_eq!(state.read_x(x(8)), 19_200_000);
    assert_eq!(state.read_x(x(9)), 0);
    assert_eq!(state.read_x(x(10)), 0x8444c004);
    assert_eq!(state.read_x(x(11)), 4);
}

#[test]
fn invalid_data_reads_return_zero_and_continue_with_native_register_optimization() {
    let (_memory, mut jit) = fixture(&[0xd2800020, 0xf9400022, 0xd2800040, 0xd4000001]);
    let mut state: ThreadState = pattern(5).into();
    state.write_x(x(1), 0x3000);
    state.write_x(x(2), 0xdead);
    let mut expected = state.clone();
    expected.write_x(x(0), 2);
    expected.write_x(x(2), 0);
    expected.set_pc(CODE.get() + 12);
    let report = run(&mut jit, &mut state);
    assert!(matches!(report.stop, CpuExit::SupervisorCall { .. }));
    assert!(report.context.is_none());
    assert_eq!(state, expected);
}

#[test]
fn data_observation_keeps_native_execution_running_until_preemption() {
    use std::sync::mpsc;
    use std::time::{Duration, Instant};
    // ADD X0,#1; STR X0,[X1]; B back to ADD. The guest publishes readiness.
    let (memory, mut jit) = fixture(&[0x91000400, 0xf9000020, 0x17fffffe]);
    let mut state: ThreadState = a64_state().into();
    state.write_x(x(1), 0x2008);
    let ranges = memory
        .capture(GuestVirtualAddress::new(0x2000), 4, MemoryPermissions::READ)
        .unwrap();
    let control = jit.control();
    let (send, completed) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        let mut worker = nixe_cpu_direct_memory::NativeWorker::default();
        let report = jit.run_slice(
            &mut worker,
            JitRunRequest {
                state: &mut state,
                timer: &FixedTimer,
                events: &VcpuEventState::default(),
            },
        );
        send.send((report, state)).unwrap();
        worker.finish().unwrap();
    });
    let deadline = Instant::now() + Duration::from_secs(5);
    while memory
        .read_value(GuestVirtualAddress::new(0x2008), 8)
        .unwrap()
        .into_ready()
        .unwrap()
        == 0
        && Instant::now() < deadline
    {
        std::thread::yield_now();
    }
    let running = memory
        .read_value(GuestVirtualAddress::new(0x2008), 8)
        .unwrap()
        .into_ready()
        .unwrap()
        != 0;
    let observation = nixe_memory::Observation::default();
    let captured = memory.machine().access_captured(
        &ranges,
        nixe_memory::AccessOperation::ReadObserved(observation.clone()),
    );
    let premature = completed.recv_timeout(Duration::from_millis(50));
    control.request_preemption();
    let (returned_for_memory, (report, state)) = match premature {
        Ok(result) => (true, result),
        Err(mpsc::RecvTimeoutError::Timeout) => (
            false,
            completed.recv_timeout(Duration::from_secs(5)).unwrap(),
        ),
        Err(error) => panic!("native worker disconnected: {error}"),
    };
    worker.join().unwrap();
    assert!(running);
    assert!(matches!(
        captured,
        Ok(nixe_memory::Access::Ready(nixe_memory::AccessResult::Read(
            _
        )))
    ));
    assert!(
        !returned_for_memory,
        "memory quiescence is not a scheduler exit"
    );
    assert!(!observation.is_invalid());
    assert_eq!(report.unwrap().stop, CpuExit::Safepoint);
    assert!(state.read_x(x(0)) > 0);
}

#[test]
fn scheduler_interrupt_exits_direct_and_indirect_native_loops() {
    for branch in [0x17fffffe, 0xd61f0040] {
        // B -8 / BR X2.
        let (memory, mut jit) = fixture(&[0x91000400, 0xf9000020, branch]);
        let control = jit.control();
        let (tx, rx) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            let mut state: ThreadState = pattern(5).into();
            state.write_x(x(1), 0x2000);
            state.write_x(x(2), CODE.get());
            let expected = state.clone();
            let mut worker = nixe_cpu_direct_memory::NativeWorker::default();
            let report = jit
                .run_slice(
                    &mut worker,
                    JitRunRequest {
                        state: &mut state,
                        timer: &FixedTimer,
                        events: &VcpuEventState::default(),
                    },
                )
                .unwrap();
            tx.send((report, state, expected)).unwrap();
            worker.finish().unwrap();
        });
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while memory
            .read_value(GuestVirtualAddress::new(0x2000), 8)
            .unwrap()
            .into_ready()
            .unwrap()
            == 0
            && std::time::Instant::now() < deadline
        {
            std::thread::yield_now();
        }
        let running = memory
            .read_value(GuestVirtualAddress::new(0x2000), 8)
            .unwrap()
            .into_ready()
            .unwrap()
            != 0;
        control.request_preemption();
        let (report, state, mut expected) =
            rx.recv_timeout(std::time::Duration::from_secs(5)).unwrap();
        worker.join().unwrap();
        assert!(running);
        assert_eq!(report.stop, CpuExit::Safepoint);
        expected.write_x(x(0), state.read_x(x(0)));
        expected.set_pc(state.pc());
        assert_eq!(state, expected);
    }
}

#[test]
fn complete_context_survives_alternating_guests_and_migrating_between_cores() {
    let (memory, initial) = fixture(&[0xd4000001]);
    drop(initial);
    // One process/monitor, three host-owned native cores; none is a guest thread.
    let process = JitProcess::new(
        ProcessCpuContext::for_platform(TargetPlatform::Switch1, memory.identity()),
        memory,
        19_200_000,
    )
    .unwrap();
    let mut first = process.create_thread().unwrap();
    let mut second = process.create_thread().unwrap();
    let mut third = process.create_thread().unwrap();
    let mut states = [
        ThreadState::from(pattern(5)),
        ThreadState::from(pattern(10)),
    ];
    let expected = states.clone();
    let mut worker = nixe_cpu_direct_memory::NativeWorker::default();
    for turn in 0..24 {
        let guest = turn % 2;
        states[1 - guest].materialize();
        let core = match (turn / 2) % 3 {
            0 => &mut first,
            1 => &mut second,
            _ => &mut third,
        };
        let report = core
            .run_slice(
                &mut worker,
                JitRunRequest {
                    state: &mut states[guest],
                    timer: &FixedTimer,
                    events: &VcpuEventState::default(),
                },
            )
            .unwrap();
        assert!(matches!(report.stop, CpuExit::SupervisorCall { .. }));
        assert_eq!(states[guest], expected[guest]);
        assert!(report.context.is_none());
    }
}

#[test]
fn failed_native_invocation_has_no_fabricated_context_and_cannot_resume() {
    let (_memory, mut jit) = fixture(&[0xd4000001]);
    let mut state: ThreadState = a64_state().into();
    let mut worker = nixe_cpu_direct_memory::NativeWorker::default();
    jit.run_slice(
        &mut worker,
        JitRunRequest {
            state: &mut state,
            timer: &FixedTimer,
            events: &VcpuEventState::default(),
        },
    )
    .unwrap();
    // Registering a second native fault context on this TID is a real error.
    let failure = jit
        .run_slice(
            &mut nixe_cpu_direct_memory::NativeWorker::default(),
            JitRunRequest {
                state: &mut state,
                timer: &FixedTimer,
                events: &VcpuEventState::default(),
            },
        )
        .unwrap_err();
    assert!(failure.message.contains("already registered"));
    assert!(failure.context.is_none());
    assert!(failure.to_string().contains("registers=[unavailable]"));
    let failure = jit
        .run_slice(
            &mut worker,
            JitRunRequest {
                state: &mut state,
                timer: &FixedTimer,
                events: &VcpuEventState::default(),
            },
        )
        .unwrap_err();
    assert!(failure.message.contains("unavailable"));
    assert!(!state.is_available());
    assert!(failure.context.is_none());
}

#[test]
fn ordinary_svc_keeps_complete_context_native_and_snapshots_are_owned() {
    let (_memory, mut jit) = fixture(&[0x91000400, 0xd4000001]); // ADD X0,#1; SVC
    let mut state: ThreadState = pattern(5).into();
    let mut worker = nixe_cpu_direct_memory::NativeWorker::default();
    let events = VcpuEventState::default();
    let mut retained = None;
    for turn in 0..24 {
        state.set_pc(CODE.get());
        state.write_x(x(0), turn);
        state.set_tpidr_el0(0xabc000 + turn);
        let report = jit
            .run_slice(
                &mut worker,
                JitRunRequest {
                    state: &mut state,
                    timer: &FixedTimer,
                    events: &events,
                },
            )
            .unwrap();
        assert!(matches!(report.stop, CpuExit::SupervisorCall { .. }));
        assert!(report.context.is_none());
        assert!(state.is_resident());
        assert_eq!(state.read_x(x(0)), turn + 1);
        assert_eq!(state.tpidr_el0(), 0xabc000 + turn);
        if turn == 0 {
            retained = Some(state.snapshot());
        }
    }
    let snapshot = retained.unwrap();
    assert_eq!(snapshot.read_x(x(0)), 1);
    assert_eq!(snapshot.tpidr_el0(), 0xabc000);
    for index in 0..32 {
        assert_eq!(state.vector(index), snapshot.vector(index));
    }
    state.materialize();
    assert!(!state.is_resident());
    assert_eq!(state.read_x(x(0)), 24);
}

#[test]
fn replacing_an_unexported_owner_is_rejected_without_corrupting_either_guest() {
    let (_memory, mut jit) = fixture(&[0xd4000001]);
    let mut first: ThreadState = pattern(5).into();
    let mut second: ThreadState = pattern(10).into();
    let mut worker = nixe_cpu_direct_memory::NativeWorker::default();
    let events = VcpuEventState::default();
    let mut run = |state: &mut ThreadState| {
        jit.run_slice(
            &mut worker,
            JitRunRequest {
                state,
                timer: &FixedTimer,
                events: &events,
            },
        )
    };
    run(&mut first).unwrap();
    let expected_first = first.snapshot();
    let expected_second = second.snapshot();
    let failure = run(&mut second).unwrap_err();
    assert!(failure.message.contains("another guest context"));
    assert_eq!(first.snapshot(), expected_first);
    assert_eq!(second.snapshot(), expected_second);
    first.leave_core();
    run(&mut second).unwrap();
    assert_eq!(second.snapshot(), expected_second);
}

#[test]
fn stopped_native_owner_retains_core_monitor_and_backing_until_materialized() {
    let (memory, mut jit) = fixture(&[0xd4000001]);
    let weak_memory = Arc::downgrade(&memory);
    let mut state: ThreadState = pattern(5).into();
    let expected = state.snapshot();
    jit.run_slice(
        &mut nixe_cpu_direct_memory::NativeWorker::default(),
        JitRunRequest {
            state: &mut state,
            timer: &FixedTimer,
            events: &VcpuEventState::default(),
        },
    )
    .unwrap();
    drop(jit);
    drop(memory);
    assert!(weak_memory.upgrade().is_some());
    assert_eq!(state.snapshot(), expected);
    state.leave_core();
    assert!(weak_memory.upgrade().is_none());
    assert_eq!(state.snapshot(), expected);
}
