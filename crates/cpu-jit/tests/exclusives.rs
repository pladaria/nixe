//! ARM exclusive semantics, independently of the former expected-value model.
//! https://developer.arm.com/documentation/ddi0602/latest/Base-Instructions/LDXR--Load-Exclusive-Register-
//! https://developer.arm.com/documentation/ddi0602/latest/Base-Instructions/STXR--Store-Exclusive-Register-

use nixe_cpu::execution::{ArchitecturalTimer, CpuExit, TimerSnapshot, VcpuEventState};
use nixe_cpu::memory::{
    CpuMemory, ExecutionMemory, MemoryAccess, MemoryAccessSize, MemoryPermissions, MemoryValue,
    ProcessMemory,
};
use nixe_cpu::platform::TargetPlatform;
use nixe_cpu::profile::ProcessCpuContext;
use nixe_cpu::state::a64::{A64GeneralRegister, A64Register};
use nixe_cpu_jit::{JitProcess, JitRunRequest, JitThread, ThreadState};
use nixe_memory::{
    AddressSpaceId, CanonicalCpuWriteDependency, CanonicalRangeTranslator, DirectBackendPolicy,
    GuestPhysicalPageId, GuestVirtualAddress,
};
use std::sync::Arc;

const SPACE: AddressSpaceId = AddressSpaceId::new(4);
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
fn fixture(code: &[u32]) -> (Arc<ExecutionMemory>, Arc<JitProcess>) {
    let mut memory = ExecutionMemory::new();
    for id in [1, 2] {
        assert!(memory.add_ram_page(GuestPhysicalPageId::new(id)));
    }
    memory
        .initialize_ram(
            GuestPhysicalPageId::new(1),
            0,
            &code
                .iter()
                .copied()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    for (address, id, permissions) in [
        (0x1000, 1, MemoryPermissions::READ_EXECUTE),
        (DATA.get(), 2, MemoryPermissions::READ_WRITE),
        (ALIAS.get(), 2, MemoryPermissions::READ_WRITE),
    ] {
        assert!(memory.map_page(
            SPACE,
            GuestVirtualAddress::new(address),
            GuestPhysicalPageId::new(id),
            permissions
        ));
    }
    memory
        .bind_cpu_memory_backend(SPACE, 0x10000, DirectBackendPolicy::Required)
        .unwrap();
    let memory = Arc::new(memory);
    let process = Arc::new(
        JitProcess::new(
            ProcessCpuContext::for_platform(TargetPlatform::Switch1, SPACE),
            memory.clone(),
            19_200_000,
        )
        .unwrap(),
    );
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
                    instruction_budget: 10_000_000,
                    timer: &Timer,
                    events: &VcpuEventState::default(),
                    capture_context: true,
                },
            )
            .unwrap()
            .stop;
        if !matches!(stop, CpuExit::Safepoint | CpuExit::BudgetExhausted)
            || std::time::Instant::now() >= deadline
        {
            return stop;
        }
    }
}

#[test]
fn every_scalar_width_and_acquire_release_use_physical_alias_reservations() {
    for (load, store, size) in [
        (0x085f7c22, 0x08037c80, MemoryAccessSize::Byte),
        (0x485f7c22, 0x48037c80, MemoryAccessSize::Halfword),
        (0x885f7c22, 0x88037c80, MemoryAccessSize::Word),
        (0xc85f7c22, 0xc8037c80, MemoryAccessSize::Doubleword),
        (0xc85ffc22, 0xc803fc80, MemoryAccessSize::Doubleword),
    ] {
        let (memory, process) = fixture(&[load, store, 0xd4000001]);
        let mut state = state();
        assert!(matches!(
            run(&mut JitThread::new(process).unwrap(), &mut state),
            CpuExit::SupervisorCall { .. }
        ));
        assert_eq!(state.read_x(x(3)), 0);
        assert_eq!(
            memory
                .read(SPACE, DATA, MemoryAccess::normal(size))
                .unwrap()
                .value
                .bits(),
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
            run(&mut JitThread::new(process).unwrap(), &mut state),
            CpuExit::SupervisorCall { .. }
        ));
        assert_eq!(state.read_x(x(4)), 0);
        assert_eq!(
            memory
                .read(SPACE, DATA, MemoryAccess::normal(size))
                .unwrap()
                .value
                .bits(),
            expected
        );
    }
}

#[test]
fn paired_exclusive_read_preserves_readonly_observation() {
    let (memory, process) = fixture(&[0xc87f0c22, 0xd4000001]);
    memory
        .write(
            SPACE,
            DATA,
            MemoryAccess::normal(MemoryAccessSize::Quadword),
            MemoryValue::U128((9_u128 << 64) | 7),
        )
        .unwrap();
    memory
        .set_permissions(SPACE, DATA, 4096, MemoryPermissions::READ)
        .unwrap();
    let range = memory
        .translate_canonical_range(SPACE, DATA, 16, MemoryPermissions::READ)
        .unwrap();
    let dependency = CanonicalCpuWriteDependency::capture(&range).unwrap();
    let mut state = state();
    assert!(matches!(
        run(&mut JitThread::new(process).unwrap(), &mut state),
        CpuExit::SupervisorCall { .. }
    ));
    assert_eq!(state.read_x(x(2)), 7);
    assert_eq!(state.read_x(x(3)), 9);
    assert!(dependency.remains_current());
}

#[test]
fn generated_ordinary_alias_writes_invalidate_aba_and_same_value_reservations() {
    for middle in [vec![0xf9000085, 0xf9000080], vec![0xf9000080]] {
        let mut code = vec![0xc85f7c22];
        code.extend(middle);
        code.extend([0xc8037c20, 0xd4000001]);
        let (memory, process) = fixture(&code);
        memory
            .write(
                SPACE,
                DATA,
                MemoryAccess::normal(MemoryAccessSize::Doubleword),
                MemoryValue::U64(7),
            )
            .unwrap();
        let mut state = state();
        assert!(matches!(
            run(&mut JitThread::new(process).unwrap(), &mut state),
            CpuExit::SupervisorCall { .. }
        ));
        assert_eq!(state.read_x(x(3)), 1);
    }
}

#[test]
fn clrex_and_context_switch_clear_the_local_reservation() {
    for (middle, scheduled, clear) in [
        (0xd5033f5f, false, true),
        (0xd503203f, true, true),
        (0xd503203f, true, false),
    ] {
        let (memory, process) = fixture(&[0xc85f7c22, middle, 0xc8037c20, 0xd4000001]);
        let mut jit = JitThread::new(process).unwrap();
        let range = memory
            .translate_canonical_range(SPACE, DATA, 16, MemoryPermissions::READ)
            .unwrap();
        let dependency = CanonicalCpuWriteDependency::capture(&range).unwrap();
        let mut state = state();
        let stop = run(&mut jit, &mut state);
        assert!(if scheduled {
            matches!(stop, CpuExit::Scheduled { .. })
        } else {
            matches!(stop, CpuExit::SupervisorCall { .. })
        });
        if scheduled {
            if clear {
                jit.clear_local_exclusive_reservation();
            }
            state.set_pc(0x1008);
            assert!(matches!(
                run(&mut jit, &mut state),
                CpuExit::SupervisorCall { .. }
            ));
        }
        assert_eq!(state.read_x(x(3)), u64::from(clear));
        assert_eq!(
            dependency.remains_current(),
            clear,
            "only committed STXR may invalidate observation"
        );
        assert_eq!(
            memory
                .read(
                    SPACE,
                    DATA,
                    MemoryAccess::normal(MemoryAccessSize::Doubleword)
                )
                .unwrap()
                .value,
            MemoryValue::U64(if clear { 0 } else { 7 })
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
                .write(
                    SPACE,
                    address,
                    MemoryAccess::normal(MemoryAccessSize::Doubleword),
                    MemoryValue::U64(0),
                )
                .unwrap();
        }
        let barrier = Arc::new(std::sync::Barrier::new(3));
        let mut workers = Vec::new();
        for consumer in [false, true] {
            let barrier = barrier.clone();
            let mut jit = JitThread::new(process.clone()).unwrap();
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
fn misaligned_exclusive_faults_without_partial_effects() {
    let (memory, process) = fixture(&[0xc85f7c22, 0xc8037c20, 0xd4000001]);
    let mut state = state();
    state.write_x(x(1), DATA.get() + 4092);
    state.write_x(x(2), 0x1234);
    let stop = run(&mut JitThread::new(process).unwrap(), &mut state);
    assert!(matches!(stop, CpuExit::DataFault { .. }));
    assert_eq!(state.pc(), 0x1000);
    assert_eq!(state.read_x(x(2)), 0x1234);
    assert_eq!(
        memory
            .read(
                SPACE,
                DATA,
                MemoryAccess::normal(MemoryAccessSize::Doubleword)
            )
            .unwrap()
            .value,
        MemoryValue::U64(0)
    );
}

#[test]
fn observed_first_write_retries_without_losing_the_observer() {
    // LDXR; STXR; CBNZ status,LDXR; SVC.
    let (memory, process) = fixture(&[0xc85f7c22, 0xc8037c20, 0x35ffffc3, 0xd4000001]);
    let range = memory
        .translate_canonical_range(SPACE, DATA, 16, MemoryPermissions::READ)
        .unwrap();
    let dependency = CanonicalCpuWriteDependency::capture(&range).unwrap();
    assert!(dependency.remains_current());
    let mut state = state();
    assert!(matches!(
        run(&mut JitThread::new(process).unwrap(), &mut state),
        CpuExit::SupervisorCall { .. }
    ));
    assert_eq!(state.read_x(x(3)), 0);
    assert!(!dependency.remains_current());
    assert_eq!(
        memory
            .read(
                SPACE,
                DATA,
                MemoryAccess::normal(MemoryAccessSize::Doubleword)
            )
            .unwrap()
            .value,
        MemoryValue::U64(7)
    );
}

#[test]
fn competing_native_cores_make_progress_without_lost_updates() {
    // LDAXR X2,[X1]; ADD; STLXR W3,X2,[X1]; CBNZ; SUBS; B.NE; SVC.
    let (memory, process) = fixture(&[
        0xc85ffc22, 0x91000442, 0xc803fc22, 0x35ffffa3, 0xf1000400, 0x54ffff61, 0xd4000001,
    ]);
    let mut workers = Vec::new();
    for _ in 0..2 {
        let mut jit = JitThread::new(process.clone()).unwrap();
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
            .read(
                SPACE,
                DATA,
                MemoryAccess::normal(MemoryAccessSize::Doubleword)
            )
            .unwrap()
            .value,
        MemoryValue::U64(20_000)
    );
}
