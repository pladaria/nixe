//! Native exclusive accesses and the expected-value CAS policy.
//! https://developer.arm.com/documentation/ddi0602/latest/Base-Instructions/LDXR--Load-Exclusive-Register-
//! https://developer.arm.com/documentation/ddi0602/latest/Base-Instructions/STXR--Store-Exclusive-Register-

use nixe_cpu::execution::{ArchitecturalTimer, CpuExit, TimerSnapshot, VcpuEventState};
use nixe_cpu::memory::MemoryAccessSize;
use nixe_cpu::platform::TargetPlatform;
use nixe_cpu::profile::ProcessCpuContext;
use nixe_cpu::state::a64::{A64GeneralRegister, A64Register};
use nixe_cpu_jit::{JitProcess, JitRunRequest, JitThread, ThreadState};
use nixe_memory::{
    AddressSpace, AddressWidth, AllocationRange, GuestVirtualAddress, Machine, MemoryPermissions,
    PAGE_SIZE,
};
use std::sync::Arc;

const DATA: GuestVirtualAddress = GuestVirtualAddress::new(0x3000);
const ALIAS: GuestVirtualAddress = GuestVirtualAddress::new(0x5000);
struct Timer;
impl ArchitecturalTimer for Timer {
    fn snapshot(&self) -> TimerSnapshot {
        TimerSnapshot {
            counter: 0,
            frequency: 19_200_000,
        }
    }
}
fn x(i: u8) -> A64Register {
    A64Register::General(A64GeneralRegister::new(i).unwrap())
}
fn fixture(code: &[u32]) -> (Arc<AddressSpace>, Arc<JitProcess>) {
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
            GuestVirtualAddress::new(0x1000),
            executable,
            0,
            PAGE_SIZE,
            MemoryPermissions::READ_EXECUTE,
        )
        .unwrap();
    for address in [DATA, ALIAS] {
        memory
            .map(
                address,
                data.clone(),
                0,
                PAGE_SIZE,
                MemoryPermissions::READ_WRITE,
            )
            .unwrap();
    }
    let process = JitProcess::new(
        ProcessCpuContext::for_platform(TargetPlatform::Switch1, memory.identity()),
        memory.clone(),
        19_200_000,
    )
    .unwrap();
    (memory, process)
}
fn state() -> ThreadState {
    let mut state = ThreadState::default();
    state.set_pc(0x1000);
    state.write_x(x(0), 7);
    state.write_x(x(1), DATA.get());
    state.write_x(x(4), ALIAS.get());
    state.write_x(x(5), 9);
    state
}
fn run(jit: &mut JitThread, state: &mut ThreadState) -> CpuExit {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        let stop = jit
            .run_slice(
                &mut nixe_cpu_direct_memory::NativeWorker::default(),
                JitRunRequest {
                    state,
                    timer: &Timer,
                    events: &VcpuEventState::default(),
                },
            )
            .unwrap()
            .stop;
        if !matches!(stop, CpuExit::Safepoint) || std::time::Instant::now() >= deadline {
            return stop;
        }
    }
}

#[test]
fn exclusive_store_with_overlapping_zero_registers_preserves_memory_semantics() {
    // LDAXR X2,[X8]; optionally CLREX; STLXR WZR,XZR,[X8]; SVC.
    // The overlapping status/data registers select the same defined behavior
    // as Eden, including the exact store encoding from the reported failure.
    // https://developer.arm.com/documentation/ddi0602/latest/Base-Instructions/STLXR--Store-Release-Exclusive-Register-
    for reserve in [false, true] {
        for clear in [false, true] {
            for observed in [false, true] {
                let mut code = Vec::new();
                if reserve {
                    code.push(0xc85ffd02);
                }
                if clear {
                    code.push(0xd5033f5f);
                }
                code.extend([0xc81ffd1f, 0xd4000001]);
                let (memory, process) = fixture(&code);
                memory
                    .write_value(DATA, 8, 7)
                    .unwrap()
                    .into_ready()
                    .unwrap();
                let observation = nixe_memory::Observation::default();
                if observed {
                    let range = memory.capture(DATA, 8, MemoryPermissions::READ).unwrap();
                    memory
                        .machine()
                        .access_captured(
                            &range,
                            nixe_memory::AccessOperation::ReadObserved(observation.clone()),
                        )
                        .unwrap()
                        .into_ready()
                        .unwrap();
                }
                let mut state = state();
                state.write_x(x(8), DATA.get());
                state.write_x(A64Register::StackPointer, 0x7000);
                let stop = run(&mut process.create_thread().unwrap(), &mut state);
                assert!(matches!(stop, CpuExit::SupervisorCall { .. }), "{stop:?}");
                assert_eq!(state.read_x(A64Register::StackPointer), 0x7000);
                let committed = reserve && !clear;
                assert_eq!(
                    memory.read_value(DATA, 8).unwrap().into_ready().unwrap(),
                    if committed { 0 } else { 7 }
                );
                if observed {
                    assert_eq!(observation.is_invalid(), committed);
                }
            }
        }
    }
}

#[test]
fn every_scalar_width_and_acquire_release_commit_through_native_exclusives() {
    for (load, store, size) in [
        (0x085f7c22, 0x08037c20, MemoryAccessSize::Byte),
        (0x485f7c22, 0x48037c20, MemoryAccessSize::Halfword),
        (0x885f7c22, 0x88037c20, MemoryAccessSize::Word),
        (0xc85f7c22, 0xc8037c20, MemoryAccessSize::Doubleword),
        (0xc85ffc22, 0xc803fc20, MemoryAccessSize::Doubleword),
    ] {
        let (memory, process) = fixture(&[load, store, 0xd4000001]);
        let mut state = state();
        assert!(matches!(
            run(&mut process.create_thread().unwrap(), &mut state),
            CpuExit::SupervisorCall { .. }
        ));
        assert_eq!(state.read_x(x(3)), 0);
        assert_eq!(
            memory
                .read_value(DATA, size.bytes())
                .unwrap()
                .into_ready()
                .unwrap(),
            7
        );
    }
}

#[test]
fn paired_exclusives_preserve_both_lanes() {
    for (load, store, size, expected) in [
        (
            0x887f0c22,
            0x88241420,
            MemoryAccessSize::Doubleword,
            (9_u128 << 32) | 7,
        ),
        (
            0xc87f0c22,
            0xc8241420,
            MemoryAccessSize::Quadword,
            (9_u128 << 64) | 7,
        ),
    ] {
        let (memory, process) = fixture(&[load, store, 0xd4000001]);
        let mut state = state();
        assert!(matches!(
            run(&mut process.create_thread().unwrap(), &mut state),
            CpuExit::SupervisorCall { .. }
        ));
        assert_eq!(state.read_x(x(4)), 0);
        assert_eq!(
            memory
                .read_value(DATA, size.bytes())
                .unwrap()
                .into_ready()
                .unwrap(),
            expected
        );
    }
}

#[test]
fn paired_exclusive_read_preserves_readonly_observation() {
    let (memory, process) = fixture(&[0xc87f0c22, 0xd4000001]);
    memory
        .write_value(DATA, MemoryAccessSize::Quadword.bytes(), (9_u128 << 64) | 7)
        .unwrap()
        .into_ready()
        .unwrap();
    memory
        .protect(DATA, PAGE_SIZE, MemoryPermissions::READ)
        .unwrap();
    let range = memory.capture(DATA, 16, MemoryPermissions::READ).unwrap();
    let dependency = nixe_memory::Observation::default();
    memory
        .machine()
        .access_captured(
            &range,
            nixe_memory::AccessOperation::ReadObserved(dependency.clone()),
        )
        .unwrap()
        .into_ready()
        .unwrap();
    let mut state = state();
    assert!(matches!(
        run(&mut process.create_thread().unwrap(), &mut state),
        CpuExit::SupervisorCall { .. }
    ));
    assert_eq!(state.read_x(x(2)), 7);
    assert_eq!(state.read_x(x(3)), 9);
    assert!(!dependency.is_invalid());
}

#[test]
fn ordinary_alias_writes_use_expected_value_cas_without_store_instrumentation() {
    // Eden's normal stores do not update the exclusive monitor. A changed
    // observed value must fail, while same-value/ABA may commit by host CAS.
    for (middle, status, final_value) in [
        (vec![0xf9000085], 1, 9),
        (vec![0xf9000085, 0xf9000080], 0, 11),
        (vec![0xf9000080], 0, 11),
    ] {
        let mut code = vec![0xc85f7c22];
        code.extend(middle);
        code.extend([0xc8037c26, 0xd4000001]); // STXR W3, X6, [X1]
        let (memory, process) = fixture(&code);
        memory
            .write_value(DATA, 8, 7)
            .unwrap()
            .into_ready()
            .unwrap();
        let mut state = state();
        state.write_x(x(6), 11);
        assert!(matches!(
            run(&mut process.create_thread().unwrap(), &mut state),
            CpuExit::SupervisorCall { .. }
        ));
        assert_eq!(state.read_x(x(3)), status);
        assert_eq!(
            memory.read_value(DATA, 8).unwrap().into_ready().unwrap(),
            final_value
        );
    }
}

#[test]
fn every_native_exclusive_width_rejects_a_changed_value_from_a_plain_alias_store() {
    for (load, ordinary, store, status) in [
        (0x085f7c22, 0x39000085, 0x08037c20, 3),
        (0x485f7c22, 0x79000085, 0x48037c20, 3),
        (0x885f7c22, 0xb9000085, 0x88037c20, 3),
        (0xc85f7c22, 0xf9000085, 0xc8037c20, 3),
        (0xc87f0c22, 0xf9000085, 0xc8241420, 4),
    ] {
        let (memory, process) = fixture(&[load, ordinary, store, 0xd4000001]);
        let mut state = state();
        assert!(matches!(
            run(&mut process.create_thread().unwrap(), &mut state),
            CpuExit::SupervisorCall { .. }
        ));
        assert_eq!(state.read_x(x(status)), 1);
        assert_eq!(memory.read_value(DATA, 8).unwrap().into_ready().unwrap(), 9);
    }
}

#[test]
fn clrex_clears_the_local_reservation_while_hints_continue_natively() {
    for (middle, cleared) in [(0xd5033f5f, true), (0xd503203f, false)] {
        let (memory, process) = fixture(&[0xc85f7c22, middle, 0xc8037c20, 0xd4000001]);
        let mut guest = state();
        assert!(matches!(
            run(&mut process.create_thread().unwrap(), &mut guest),
            CpuExit::SupervisorCall { .. }
        ));
        assert_eq!(guest.read_x(x(3)), u64::from(cleared));
        assert_eq!(
            memory.read_value(DATA, 8).unwrap().into_ready().unwrap(),
            if cleared { 0 } else { 7 }
        );
    }
}

#[test]
fn native_release_acquire_publishes_the_prior_payload() {
    // Producer: STR payload; LDAXR/STLXR flag; retry; SVC.
    // Consumer at +24: LDAXR flag; retry zero; LDR payload; CLREX; SVC.
    let (memory, process) = fixture(&[
        0xf9000825, 0xc85ffc22, 0xc803fc20, 0x35ffffc3, 0xd4000001, 0xd503201f, 0xc85ffc22,
        0xb4ffffe2, 0xf9400826, 0xd5033f5f, 0xd4000001,
    ]);
    for iteration in 1..=16 {
        for address in [DATA, DATA.checked_add(16).unwrap()] {
            memory
                .write_value(address, MemoryAccessSize::Doubleword.bytes(), 0)
                .unwrap()
                .into_ready()
                .unwrap();
        }
        let barrier = Arc::new(std::sync::Barrier::new(3));
        let mut workers = Vec::new();
        for consumer in [false, true] {
            let barrier = barrier.clone();
            let mut jit = process.create_thread().unwrap();
            workers.push(std::thread::spawn(move || {
                let mut state = state();
                state.write_x(x(0), 1);
                state.write_x(x(5), iteration);
                if consumer {
                    state.set_pc(0x1018);
                }
                barrier.wait();
                let stop = run(&mut jit, &mut state);
                assert!(matches!(stop, CpuExit::SupervisorCall { .. }), "{stop:?}");
                if consumer {
                    assert_eq!(state.read_x(x(6)), iteration);
                }
            }));
        }
        barrier.wait();
        for worker in workers {
            worker.join().unwrap();
        }
    }
}

#[test]
fn observed_first_write_retries_without_losing_the_observer() {
    // LDXR/STXR or LDXP/STXP; retry the exclusive load after a watched write.
    for (load, store, retry, width, status, expected) in [
        (0xc85f7c22, 0xc8037c20, 0x35ffffc3, 8, 3, 7),
        (0xc87f0c22, 0xc8241420, 0x35ffffc4, 16, 4, (9u128 << 64) | 7),
    ] {
        let (memory, process) = fixture(&[load, store, retry, 0xd4000001]);
        let range = memory.capture(DATA, 16, MemoryPermissions::READ).unwrap();
        let dependency = nixe_memory::Observation::default();
        memory
            .machine()
            .access_captured(
                &range,
                nixe_memory::AccessOperation::ReadObserved(dependency.clone()),
            )
            .unwrap()
            .into_ready()
            .unwrap();
        assert!(!dependency.is_invalid());
        let mut state = state();
        assert!(matches!(
            run(&mut process.create_thread().unwrap(), &mut state),
            CpuExit::SupervisorCall { .. }
        ));
        assert_eq!(state.read_x(x(status)), 0);
        assert!(dependency.is_invalid());
        assert_eq!(
            memory
                .read_value(DATA, width)
                .unwrap()
                .into_ready()
                .unwrap(),
            expected
        );
    }
}

#[test]
fn competing_native_cores_make_progress_without_lost_updates() {
    // LDAXR X2,[X1]; ADD; STLXR W3,X2,[X1]; CBNZ; SUBS; B.NE; SVC.
    let (memory, process) = fixture(&[
        0xc85ffc22, 0x91000442, 0xc803fc22, 0x35ffffa3, 0xf1000400, 0x54ffff61, 0xd4000001,
    ]);
    let mut workers = Vec::new();
    for _ in 0..2 {
        let mut jit = process.create_thread().unwrap();
        workers.push(std::thread::spawn(move || {
            let mut state = state();
            state.write_x(x(0), 10_000);
            assert!(matches!(
                run(&mut jit, &mut state),
                CpuExit::SupervisorCall { .. }
            ));
        }));
    }
    for worker in workers {
        worker.join().unwrap();
    }
    assert_eq!(
        memory
            .read_value(DATA, MemoryAccessSize::Doubleword.bytes())
            .unwrap()
            .into_ready()
            .unwrap(),
        20_000
    );
}

#[test]
fn native_exclusives_preserve_live_floating_point_registers_and_flags() {
    // FADD D2,D0,D1; CMP X0,X0; LDXR X8,[X1]; ADD X8,X8,#1;
    // STXR W9,X8,[X1]; CSET W10,EQ; FADD D3,D2,D1; SVC.
    let (memory, process) = fixture(&[
        0x1e612802, 0xeb00001f, 0xc85f7c28, 0x91000508, 0xc8097c28, 0x1a9f17ea, 0x1e612843,
        0xd4000001,
    ]);
    for iteration in 0..4u64 {
        let mut state = state();
        for index in 0..32 {
            state.set_vector(
                index,
                0x1234_5678_9abc_def0_1122_3344_5566_7700 + u128::from(index),
            );
        }
        state.set_vector(0, u128::from(1.25f64.to_bits()));
        state.set_vector(1, u128::from(2.5f64.to_bits()));
        assert!(matches!(
            run(&mut process.create_thread().unwrap(), &mut state),
            CpuExit::SupervisorCall { .. }
        ));
        assert_eq!(state.read_x(x(9)), 0);
        assert_eq!(state.read_x(x(10)), 1);
        assert_eq!(state.nzcv().bits(), 0x6000_0000);
        assert_eq!(state.vector(0), Some(u128::from(1.25f64.to_bits())));
        assert_eq!(state.vector(1), Some(u128::from(2.5f64.to_bits())));
        assert_eq!(state.vector(2), Some(u128::from(3.75f64.to_bits())));
        assert_eq!(state.vector(3), Some(u128::from(6.25f64.to_bits())));
        for index in 4..32 {
            assert_eq!(
                state.vector(index),
                Some(0x1234_5678_9abc_def0_1122_3344_5566_7700 + u128::from(index))
            );
        }
        assert_eq!(
            memory.read_value(DATA, 8).unwrap().into_ready().unwrap(),
            u128::from(iteration + 1)
        );
    }
}
