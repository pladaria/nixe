use nixe_cpu::execution::{
    ArchitecturalTimer, CpuExit, CpuThreadId, MemoryBinding, TimerSnapshot, VcpuEventState,
};
use nixe_cpu::memory::{ExecutionMemory, MemoryPermissions};
use nixe_cpu::platform::TargetPlatform;
use nixe_cpu::profile::ProcessCpuContext;
use nixe_cpu::state::a64::{A64GeneralRegister, A64Register, A64State};
use nixe_cpu_interpreter::{InterpreterProcess, InterpreterRunRequest};
use nixe_cpu_jit::{JitProcess, JitRunRequest, JitThread, ThreadState};
use nixe_memory::{
    AddressSpaceId, GuestPhysicalPageId, GuestVirtualAddress, MemoryInvalidationSource,
};
use std::sync::Arc;

const SPACE: AddressSpaceId = AddressSpaceId::new(7);
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

#[test]
fn concrete_interpreter_and_jit_match_at_an_architectural_boundary() {
    let cpu = ProcessCpuContext::for_platform(TargetPlatform::Switch1, SPACE);
    let code = [0xd503_201f_u32, 0xd420_0000];
    let mut memory = executable_memory(&code);
    memory
        .bind_cpu_memory_backend(SPACE, 0x10000, nixe_memory::DirectBackendPolicy::Required)
        .unwrap();
    let memory = Arc::new(memory);
    let binding = MemoryBinding {
        address_space: SPACE,
        end_exclusive: GuestVirtualAddress::new(1_u64 << 39),
        memory: memory.as_ref(),
        mapping_epoch: memory.mapping_epoch().get(),
        invalidation_cursor: memory.invalidation_cursor(),
    };

    let mut interpreter_process = InterpreterProcess::new(cpu);
    interpreter_process.bind_memory(binding).unwrap();
    let mut interpreter = interpreter_process
        .create_thread(CpuThreadId::new(1))
        .unwrap();
    let jit_process = Arc::new(JitProcess::new(cpu, memory.clone(), 19_200_000).unwrap());
    let mut jit = JitThread::new(jit_process.clone()).unwrap();

    let mut interpreter_state = a64_state();
    let mut jit_state: ThreadState = a64_state().into();
    let interpreter_report = interpreter
        .run_slice(
            &mut nixe_cpu_direct_memory::NativeWorker::default(),
            interpreter_request(&memory, &mut interpreter_state, 2),
        )
        .unwrap();
    let jit_report = jit
        .run_slice(
            &mut nixe_cpu_direct_memory::NativeWorker::default(),
            JitRunRequest {
                state: &mut jit_state,
                instruction_budget: 2,
                timer: &FixedTimer,
                events: &VcpuEventState::default(),
                capture_context: true,
            },
        )
        .unwrap();

    assert_eq!(interpreter_state, jit_state.snapshot());
    assert!(matches!(
        interpreter_report.stop,
        CpuExit::ArchitecturalException { .. }
    ));
    assert_eq!(interpreter_report.stop, jit_report.stop);

    drop(jit);
    assert!(jit_process.try_shutdown().unwrap());
}

#[test]
fn switch_1_pointer_authentication_hint_family_is_differentially_nop() {
    let cpu = ProcessCpuContext::for_platform(TargetPlatform::Switch1, SPACE);
    let hints = [
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
    ];
    let mut code = hints.to_vec();
    code.push(0xd420_0000);
    let mut memory = executable_memory(&code);
    memory
        .bind_cpu_memory_backend(SPACE, 0x10000, nixe_memory::DirectBackendPolicy::Required)
        .unwrap();
    let memory = Arc::new(memory);
    let binding = MemoryBinding {
        address_space: SPACE,
        end_exclusive: GuestVirtualAddress::new(1_u64 << 39),
        memory: memory.as_ref(),
        mapping_epoch: memory.mapping_epoch().get(),
        invalidation_cursor: memory.invalidation_cursor(),
    };

    let mut interpreter_process = InterpreterProcess::new(cpu);
    interpreter_process.bind_memory(binding).unwrap();
    let mut interpreter = interpreter_process
        .create_thread(CpuThreadId::new(1))
        .unwrap();
    let jit_process = Arc::new(JitProcess::new(cpu, memory.clone(), 19_200_000).unwrap());
    let mut jit = JitThread::new(jit_process.clone()).unwrap();

    let link_register = A64Register::General(A64GeneralRegister::new(30).unwrap());
    let signed_pointer = 0xabcd_0000_7518_7c14;
    let mut interpreter_state = a64_state();
    interpreter_state.write_x(link_register, signed_pointer);
    let mut jit_state: ThreadState = interpreter_state.clone().into();
    let interpreter_budget = hints.len() as u64 + 1;
    let interpreter_report = interpreter
        .run_slice(
            &mut nixe_cpu_direct_memory::NativeWorker::default(),
            interpreter_request(&memory, &mut interpreter_state, interpreter_budget),
        )
        .unwrap();
    let jit_report = jit
        .run_slice(
            &mut nixe_cpu_direct_memory::NativeWorker::default(),
            JitRunRequest {
                state: &mut jit_state,
                instruction_budget: interpreter_budget,
                timer: &FixedTimer,
                events: &VcpuEventState::default(),
                capture_context: true,
            },
        )
        .unwrap();

    assert_eq!(interpreter_state, jit_state.snapshot());
    assert_eq!(interpreter_state.read_x(link_register), signed_pointer);
    assert!(matches!(
        interpreter_report.stop,
        CpuExit::ArchitecturalException { .. }
    ));
    assert_eq!(interpreter_report.stop, jit_report.stop);

    drop(jit);
    assert!(jit_process.try_shutdown().unwrap());
}

fn executable_memory(code: &[u32]) -> ExecutionMemory {
    let mut memory = ExecutionMemory::new();
    let page = GuestPhysicalPageId::new(1);
    assert!(memory.add_ram_page(page));
    let bytes: Vec<_> = code.iter().copied().flat_map(u32::to_le_bytes).collect();
    memory.initialize_ram(page, 0, &bytes).unwrap();
    assert!(memory.map_page(SPACE, CODE, page, MemoryPermissions::READ_EXECUTE));
    memory
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

fn interpreter_request<'a>(
    memory: &'a ExecutionMemory,
    state: &'a mut A64State,
    instruction_budget: u64,
) -> InterpreterRunRequest<'a> {
    InterpreterRunRequest {
        memory,
        memory_lease: Some(memory.acquire_execution_lease()),
        state,
        instruction_budget,
        timer: &FixedTimer,
        events: VcpuEventState::default(),
    }
}

fn fixture(code: &[u32]) -> (Arc<ExecutionMemory>, JitThread) {
    let mut memory = executable_memory(code);
    let data = GuestPhysicalPageId::new(2);
    assert!(memory.add_ram_page(data));
    assert!(memory.map_page(
        SPACE,
        GuestVirtualAddress::new(0x2000),
        data,
        MemoryPermissions::READ_WRITE
    ));
    memory
        .bind_cpu_memory_backend(SPACE, 0x10000, nixe_memory::DirectBackendPolicy::Required)
        .unwrap();
    let memory = Arc::new(memory);
    let cpu = ProcessCpuContext::for_platform(TargetPlatform::Switch1, SPACE);
    let jit = JitThread::new(Arc::new(
        JitProcess::new(cpu, memory.clone(), 19_200_000).unwrap(),
    ))
    .unwrap();
    (memory, jit)
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
            timer: &FixedTimer,
            events: &VcpuEventState::default(),
            capture_context: true,
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
    assert_eq!(state.read_x(x(10)), 0x40004);
    assert_eq!(state.read_x(x(11)), 0x14);
}

#[test]
fn unmapped_data_fault_stops_at_the_load_without_committing_destination() {
    let (_memory, mut jit) = fixture(&[0xd2800020, 0xf9400022, 0xd2800040, 0xd4000001]);
    let mut state: ThreadState = pattern(5).into();
    state.write_x(x(1), 0x3000);
    state.write_x(x(2), 0xdead);
    let mut expected = state.clone();
    expected.write_x(x(0), 1);
    expected.set_pc(CODE.get() + 4);
    let report = run(&mut jit, &mut state);
    assert!(
        matches!(report.stop, CpuExit::DataFault { ref source, ref fault } if source.pc.get() == CODE.get() + 4 && fault.address.get() == 0x3000)
    );
    assert_eq!(state.pc(), CODE.get() + 4);
    assert_eq!(state.read_x(x(0)), 1);
    assert_eq!(state.read_x(x(2)), 0xdead);
    assert_eq!(state, expected);
}

#[test]
fn host_code_overwrite_invalidates_native_blocks() {
    let (memory, mut jit) = fixture(&[0xd2800540, 0xd4000001]); // MOV X0,#42; SVC
    let mut state: ThreadState = a64_state().into();
    run(&mut jit, &mut state);
    assert_eq!(state.read_x(x(0)), 42);
    memory
        .overwrite_mapped_ram(SPACE, CODE, &0xd2800560_u32.to_le_bytes())
        .unwrap();
    state.set_pc(CODE.get());
    run(&mut jit, &mut state);
    assert_eq!(state.read_x(x(0)), 43);
}

#[test]
fn scheduler_interrupt_exits_a_linked_infinite_loop() {
    use nixe_cpu::execution::ControlRequest;
    let (_memory, mut jit) = fixture(&[0x91000400, 0x17ffffff]); // ADD X0,#1; B -4
    let control = jit.control();
    let (tx, rx) = std::sync::mpsc::channel();
    let worker = std::thread::spawn(move || {
        let mut state: ThreadState = pattern(5).into();
        let expected = state.clone();
        let report = jit
            .run_slice(
                &mut nixe_cpu_direct_memory::NativeWorker::default(),
                JitRunRequest {
                    state: &mut state,
                    instruction_budget: i64::MAX as u64,
                    timer: &FixedTimer,
                    events: &VcpuEventState::default(),
                    capture_context: true,
                },
            )
            .unwrap();
        tx.send((report, state, expected)).unwrap();
    });
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while !control.execution_active() && std::time::Instant::now() < deadline {
        std::thread::yield_now();
    }
    assert!(control.execution_active());
    control.request(ControlRequest::Preempt);
    let (report, state, mut expected) = rx.recv_timeout(std::time::Duration::from_secs(5)).unwrap();
    assert_eq!(report.stop, CpuExit::Safepoint);
    expected.write_x(x(0), state.read_x(x(0)));
    expected.set_pc(state.pc());
    assert_eq!(state, expected);
    assert_eq!(report.context, Some(state.register_context()));
    worker.join().unwrap();
}

#[test]
fn guest_code_stores_become_visible_after_instruction_cache_maintenance() {
    use nixe_cpu::memory::{
        CacheMaintenanceKind, CpuMemory, MemoryAccess, MemoryAccessSize, MemoryValue,
    };
    let code = [0xd2800540, 0xd4000001];
    let mut memory = executable_memory(&code);
    let alias = GuestVirtualAddress::new(0x4000);
    assert!(memory.map_page(
        SPACE,
        alias,
        GuestPhysicalPageId::new(1),
        MemoryPermissions::READ_WRITE
    ));
    memory
        .bind_cpu_memory_backend(SPACE, 0x10000, nixe_memory::DirectBackendPolicy::Required)
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
    let mut first = JitThread::new(process.clone()).unwrap();
    let mut second = JitThread::new(process).unwrap();
    for jit in [&mut first, &mut second] {
        let mut state: ThreadState = a64_state().into();
        run(jit, &mut state);
        assert_eq!(state.read_x(x(0)), 42);
    }
    memory
        .write(
            SPACE,
            alias,
            MemoryAccess::normal(MemoryAccessSize::Word),
            MemoryValue::U32(0xd2800560),
        )
        .unwrap();
    let mut state: ThreadState = a64_state().into();
    run(&mut first, &mut state);
    assert_eq!(state.read_x(x(0)), 42);
    memory
        .maintain_cache(
            SPACE,
            CacheMaintenanceKind::InstructionInvalidate,
            Some(CODE),
        )
        .unwrap();
    state.leave_core();
    for jit in [&mut first, &mut second] {
        let mut state: ThreadState = a64_state().into();
        run(jit, &mut state);
        assert_eq!(state.read_x(x(0)), 43);
    }
}

#[test]
fn lost_invalidation_history_clears_the_native_cache() {
    let (memory, mut jit) = fixture(&[0xd2800540, 0xd4000001]);
    let mut state: ThreadState = a64_state().into();
    run(&mut jit, &mut state);
    for _ in 0..1100 {
        memory
            .overwrite_mapped_ram(SPACE, CODE, &0xd2800560_u32.to_le_bytes())
            .unwrap();
    }
    state.set_pc(CODE.get());
    run(&mut jit, &mut state);
    assert_eq!(state.read_x(x(0)), 43);
}

#[test]
fn indirect_branches_observe_the_block_budget() {
    let (_memory, mut jit) = fixture(&[0x91000400, 0xd61f0020]); // ADD X0,#1; BR X1
    let mut state: ThreadState = pattern(5).into();
    state.write_x(x(0), 0);
    state.write_x(x(1), CODE.get());
    let mut expected = state.clone();
    expected.write_x(x(0), 10);
    let report = jit
        .run_slice(
            &mut nixe_cpu_direct_memory::NativeWorker::default(),
            JitRunRequest {
                state: &mut state,
                instruction_budget: 20,
                timer: &FixedTimer,
                events: &VcpuEventState::default(),
                capture_context: true,
            },
        )
        .unwrap();
    assert_eq!(report.stop, CpuExit::BudgetExhausted);
    assert_eq!(state.read_x(x(0)), 10);
    assert_eq!(state, expected);
}

#[test]
fn prohibited_cache_zero_traps_without_writing_memory() {
    use nixe_cpu::memory::{CpuMemory, MemoryAccess, MemoryAccessSize};
    let (memory, mut jit) = fixture(&[0xd50b7421, 0xd4000001]); // DC ZVA,X1; SVC
    let mut state: ThreadState = a64_state().into();
    state.write_x(x(1), 0x2000);
    memory
        .overwrite_mapped_ram(SPACE, GuestVirtualAddress::new(0x2000), &[0x55; 64])
        .unwrap();
    let report = run(&mut jit, &mut state);
    assert!(matches!(
        report.stop,
        CpuExit::ArchitecturalException {
            kind: nixe_cpu::exception::ExceptionKind::SystemRegisterTrap,
            ..
        }
    ));
    assert_eq!(state.pc(), CODE.get());
    assert_eq!(
        memory
            .read(
                SPACE,
                GuestVirtualAddress::new(0x2000),
                MemoryAccess::normal(MemoryAccessSize::Doubleword)
            )
            .unwrap()
            .value
            .bits(),
        0x5555555555555555
    );
}

#[test]
fn complete_context_survives_alternating_guests_and_migrating_between_cores() {
    let (memory, initial) = fixture(&[0xd4000001]);
    drop(initial);
    // One process/monitor, three host-owned native cores; none is a guest thread.
    let process = Arc::new(
        JitProcess::new(
            ProcessCpuContext::for_platform(TargetPlatform::Switch1, SPACE),
            memory,
            19_200_000,
        )
        .unwrap(),
    );
    let mut first = JitThread::new(process.clone()).unwrap();
    let mut second = JitThread::new(process.clone()).unwrap();
    let mut third = JitThread::new(process).unwrap();
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
                    instruction_budget: 1,
                    timer: &FixedTimer,
                    events: &VcpuEventState::default(),
                    capture_context: true,
                },
            )
            .unwrap();
        assert!(matches!(report.stop, CpuExit::SupervisorCall { .. }));
        assert_eq!(states[guest], expected[guest]);
        assert_eq!(report.context, Some(expected[guest].register_context()));
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
            instruction_budget: 1,
            timer: &FixedTimer,
            events: &VcpuEventState::default(),
            capture_context: true,
        },
    )
    .unwrap();
    // Registering a second native fault context on this TID is a real error.
    let failure = jit
        .run_slice(
            &mut nixe_cpu_direct_memory::NativeWorker::default(),
            JitRunRequest {
                state: &mut state,
                instruction_budget: 1,
                timer: &FixedTimer,
                events: &VcpuEventState::default(),
                capture_context: true,
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
                instruction_budget: 1,
                timer: &FixedTimer,
                events: &VcpuEventState::default(),
                capture_context: true,
            },
        )
        .unwrap_err();
    assert!(failure.message.contains("unavailable"));
    assert!(!state.is_available());
    assert!(failure.context.is_none());
}

#[test]
fn full_state_matches_interpreter_after_fp_simd_and_repeated_svc_entries() {
    // Encodings already covered by instruction tests: FMOV, FADD, MOVI,
    // MRS TPIDR_EL0/TPIDRRO_EL0, ADDS, SVC.
    let code = [
        0x1e6e1000, 0x1e602801, 0x4f02e442, 0xd53bd046, 0xd53bd067, 0xb1000400, 0xd4000841,
    ];
    let (memory, mut jit) = fixture(&code);
    let mut interpreter_process = InterpreterProcess::new(ProcessCpuContext::for_platform(
        TargetPlatform::Switch1,
        SPACE,
    ));
    interpreter_process
        .bind_memory(MemoryBinding {
            address_space: SPACE,
            end_exclusive: GuestVirtualAddress::new(0x10000),
            memory: memory.as_ref(),
            mapping_epoch: memory.mapping_epoch().get(),
            invalidation_cursor: memory.invalidation_cursor(),
        })
        .unwrap();
    let mut interpreter = interpreter_process
        .create_thread(CpuThreadId::new(1))
        .unwrap();
    let mut states = [ThreadState::from(pattern(5)), ThreadState::from(pattern(5))];
    let mut worker = nixe_cpu_direct_memory::NativeWorker::default();
    for turn in 0..24 {
        for state in &mut states {
            state.set_pc(CODE.get());
            // Model runtime writes between native entries, including both lanes
            // of a vector not touched by the guest program.
            state.set_tpidr_el0(0xabcdef + turn);
            state.set_tpidrro_el0_from_runtime(0x123456 + turn);
            state.set_vector(31, (u128::from(turn) << 96) | 0xfedcba9876543210);
            state.set_fpcr(((turn as u32) & 3) << 22);
        }
        let interpreted = interpreter
            .run_slice(
                &mut worker,
                interpreter_request(&memory, states[0].saved_mut(), 1000),
            )
            .unwrap();
        let recompiled = jit
            .run_slice(
                &mut worker,
                JitRunRequest {
                    state: &mut states[1],
                    instruction_budget: 1000,
                    timer: &FixedTimer,
                    events: &VcpuEventState::default(),
                    capture_context: true,
                },
            )
            .unwrap();
        assert!(matches!(interpreted.stop, CpuExit::SupervisorCall { .. }));
        assert_eq!(recompiled.stop, interpreted.stop);
        assert_eq!(states[1], states[0]);
        assert_eq!(recompiled.context, interpreted.context);
    }
}

#[test]
fn full_state_matches_interpreter_after_precise_memory_abort() {
    let (memory, mut jit) = fixture(&[0xd2800020, 0xf9400022, 0xd4000001]);
    let mut interpreter_process = InterpreterProcess::new(ProcessCpuContext::for_platform(
        TargetPlatform::Switch1,
        SPACE,
    ));
    interpreter_process
        .bind_memory(MemoryBinding {
            address_space: SPACE,
            end_exclusive: GuestVirtualAddress::new(0x10000),
            memory: memory.as_ref(),
            mapping_epoch: memory.mapping_epoch().get(),
            invalidation_cursor: memory.invalidation_cursor(),
        })
        .unwrap();
    let mut interpreter = interpreter_process
        .create_thread(CpuThreadId::new(1))
        .unwrap();
    let mut initial = pattern(5);
    initial.write_x(x(1), 0x3000);
    let mut interpreted_state = initial.clone();
    let mut recompiled_state: ThreadState = initial.into();
    let mut worker = nixe_cpu_direct_memory::NativeWorker::default();
    let interpreted = interpreter
        .run_slice(
            &mut worker,
            interpreter_request(&memory, &mut interpreted_state, 1000),
        )
        .unwrap();
    let recompiled = jit
        .run_slice(
            &mut worker,
            JitRunRequest {
                state: &mut recompiled_state,
                instruction_budget: 1000,
                timer: &FixedTimer,
                events: &VcpuEventState::default(),
                capture_context: true,
            },
        )
        .unwrap();
    assert!(matches!(interpreted.stop, CpuExit::DataFault { .. }));
    assert_eq!(recompiled.stop, interpreted.stop);
    assert_eq!(recompiled_state.snapshot(), interpreted_state);
    assert_eq!(recompiled.context, interpreted.context);
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
                    instruction_budget: 2,
                    timer: &FixedTimer,
                    events: &events,
                    capture_context: false,
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
                instruction_budget: 1,
                timer: &FixedTimer,
                events: &events,
                capture_context: false,
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
            instruction_budget: 1,
            timer: &FixedTimer,
            events: &VcpuEventState::default(),
            capture_context: false,
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

#[test]
fn preemption_does_not_acknowledge_unconsumed_invalidations() {
    use nixe_cpu::execution::ControlRequest;
    let (memory, mut jit) = fixture(&[0xd2800540, 0xd4000001]);
    let mut state: ThreadState = a64_state().into();
    run(&mut jit, &mut state);
    memory
        .overwrite_mapped_ram(SPACE, CODE, &0xd2800560_u32.to_le_bytes())
        .unwrap();
    let cursor = memory.invalidation_cursor();
    let control = jit.control();
    control.request_invalidation(cursor.get());
    control.request(ControlRequest::Preempt);
    state.set_pc(CODE.get());
    assert_eq!(run(&mut jit, &mut state).stop, CpuExit::Safepoint);
    assert!(!control.acknowledged_invalidation(cursor.get()));
    jit.synchronize_address_space(MemoryBinding {
        memory: memory.as_ref(),
        address_space: SPACE,
        end_exclusive: GuestVirtualAddress::new(0x10000),
        mapping_epoch: memory.mapping_epoch().get(),
        invalidation_cursor: memory.invalidation_cursor(),
    })
    .unwrap();
    assert!(control.acknowledged_invalidation(cursor.get()));
    assert!(matches!(
        run(&mut jit, &mut state).stop,
        CpuExit::SupervisorCall { .. }
    ));
    assert_eq!(state.read_x(x(0)), 43);
}

fn value_at(jit: &mut JitThread, address: GuestVirtualAddress) -> u64 {
    let mut state = ThreadState::default();
    state.set_pc(address.get());
    assert!(matches!(
        run(jit, &mut state).stop,
        CpuExit::SupervisorCall { .. }
    ));
    state.read_x(x(0))
}

#[test]
fn remapped_physical_pages_invalidate_old_and_current_aliases_on_idle_cores() {
    use nixe_cpu::memory::ProcessMemory;
    let mut memory = ExecutionMemory::new();
    memory
        .bind_cpu_memory_backend(SPACE, 0x10000, nixe_memory::DirectBackendPolicy::Required)
        .unwrap();
    let memory = Arc::new(memory);
    let a = memory.allocate_shared_backing(4096).unwrap();
    let b = memory.allocate_shared_backing(4096).unwrap();
    for (backing, word) in [(&a, 0xd2800540_u32), (&b, 0xd2800560)] {
        backing
            .write(
                0,
                &[word.to_le_bytes(), 0xd4000001_u32.to_le_bytes()].concat(),
            )
            .unwrap();
    }
    let old_alias = GuestVirtualAddress::new(0x4000);
    let new_alias = GuestVirtualAddress::new(0x6000);
    for (address, backing) in [(CODE, &a), (old_alias, &a), (new_alias, &b)] {
        memory
            .map_shared_backing(SPACE, address, backing, MemoryPermissions::READ_WRITE)
            .unwrap();
        memory
            .set_permissions(SPACE, address, 4096, MemoryPermissions::READ_EXECUTE)
            .unwrap();
    }
    let process = Arc::new(
        JitProcess::new(
            ProcessCpuContext::for_platform(TargetPlatform::Switch1, SPACE),
            memory.clone(),
            19_200_000,
        )
        .unwrap(),
    );
    let mut cores = [
        JitThread::new(process.clone()).unwrap(),
        JitThread::new(process.clone()).unwrap(),
    ];
    for core in &mut cores {
        assert_eq!(value_at(core, CODE), 42);
        assert_eq!(value_at(core, old_alias), 42);
        assert_eq!(value_at(core, new_alias), 43);
    }
    // Content A precedes remap, content B follows it, all while cores are idle.
    a.write(0, &0xd2800580_u32.to_le_bytes()).unwrap();
    memory.unmap_shared_backing(SPACE, CODE, &a).unwrap();
    memory
        .map_shared_backing(SPACE, CODE, &b, MemoryPermissions::READ)
        .unwrap();
    memory
        .set_permissions(SPACE, CODE, 4096, MemoryPermissions::READ_EXECUTE)
        .unwrap();
    b.write(0, &0xd28005a0_u32.to_le_bytes()).unwrap();
    let cursor = memory.invalidation_cursor();
    for core in &mut cores {
        let control = core.control();
        assert!(!control.acknowledged_invalidation(cursor.get()));
        core.synchronize_address_space(MemoryBinding {
            memory: memory.as_ref(),
            address_space: SPACE,
            end_exclusive: GuestVirtualAddress::new(0x10000),
            mapping_epoch: memory.mapping_epoch().get(),
            invalidation_cursor: memory.invalidation_cursor(),
        })
        .unwrap();
        assert!(control.acknowledged_invalidation(cursor.get()));
        assert_eq!(value_at(core, CODE), 45);
        assert_eq!(value_at(core, old_alias), 44);
        assert_eq!(value_at(core, new_alias), 45);
    }
    // Shutdown accepts pending invalidations without requiring guest execution.
    b.write(0, &0xd28005c0_u32.to_le_bytes()).unwrap();
    let shutdown_cursor = memory.invalidation_cursor();
    assert!(process.try_shutdown().unwrap());
    for core in &mut cores {
        core.synchronize_address_space(MemoryBinding {
            memory: memory.as_ref(),
            address_space: SPACE,
            end_exclusive: GuestVirtualAddress::new(0x10000),
            mapping_epoch: memory.mapping_epoch().get(),
            invalidation_cursor: memory.invalidation_cursor(),
        })
        .unwrap();
        assert!(
            core.control()
                .acknowledged_invalidation(shutdown_cursor.get())
        );
    }
}

#[test]
fn host_writes_through_nonexecutable_aliases_reach_concurrent_native_cores() {
    let mut memory = executable_memory(&[0xd2800540, 0xd4000001]);
    let alias = GuestVirtualAddress::new(0x4000);
    assert!(memory.map_page(
        SPACE,
        alias,
        GuestPhysicalPageId::new(1),
        MemoryPermissions::READ_WRITE
    ));
    memory
        .bind_cpu_memory_backend(SPACE, 0x10000, nixe_memory::DirectBackendPolicy::Required)
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
    let (results, completed) = std::sync::mpsc::channel();
    let mut commands = Vec::new();
    let mut workers = Vec::new();
    for _ in 0..2 {
        let mut core = JitThread::new(process.clone()).unwrap();
        let (send, receive) = std::sync::mpsc::channel();
        commands.push(send);
        let results = results.clone();
        workers.push(std::thread::spawn(move || {
            for expected in receive {
                assert_eq!(value_at(&mut core, CODE), expected);
                results.send(()).unwrap();
            }
        }));
    }
    drop(results);
    for value in 42..50 {
        let word = 0xd2800000_u32 | (value << 5);
        // Many records and concurrent consumers, but no guest accesses overlap
        // the trusted host mutation's exclusive lease.
        for _ in 0..16 {
            memory
                .overwrite_mapped_ram(SPACE, alias, &word.to_le_bytes())
                .unwrap();
        }
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
fn execute_permission_loss_retires_compiled_blocks_and_restoration_refetches() {
    use nixe_cpu::error::InstructionFetchFaultReason;
    use nixe_cpu::memory::ProcessMemory;
    let (memory, mut jit) = fixture(&[0xd2800540, 0xd4000001]);
    assert_eq!(value_at(&mut jit, CODE), 42);
    memory
        .set_permissions(SPACE, CODE, 4096, MemoryPermissions::READ)
        .unwrap();
    let mut state: ThreadState = a64_state().into();
    let result = run(&mut jit, &mut state);
    assert!(
        matches!(result.stop, CpuExit::FetchFault { fault } if fault.reason == InstructionFetchFaultReason::ExecutePermissionDenied)
    );
    assert_eq!(state.pc(), CODE.get());
    assert_eq!(state.read_x(x(0)), 0);
    memory
        .overwrite_mapped_ram(SPACE, CODE, &0xd2800560_u32.to_le_bytes())
        .unwrap();
    memory
        .set_permissions(SPACE, CODE, 4096, MemoryPermissions::EXECUTE)
        .unwrap();
    state.leave_core();
    assert_eq!(value_at(&mut jit, CODE), 43);
}

#[test]
fn speculative_fetch_at_execute_only_page_boundary_preserves_precise_faults() {
    use nixe_cpu::error::InstructionFetchFaultReason;
    let mut memory = ExecutionMemory::new();
    assert!(memory.add_ram_page(GuestPhysicalPageId::new(1)));
    memory
        .initialize_ram(
            GuestPhysicalPageId::new(1),
            4092,
            &0xd4000001_u32.to_le_bytes(),
        )
        .unwrap();
    assert!(memory.map_page(
        SPACE,
        CODE,
        GuestPhysicalPageId::new(1),
        MemoryPermissions::EXECUTE
    ));
    memory
        .bind_cpu_memory_backend(SPACE, 0x10000, nixe_memory::DirectBackendPolicy::Required)
        .unwrap();
    let memory = Arc::new(memory);
    let mut jit = JitThread::new(Arc::new(
        JitProcess::new(
            ProcessCpuContext::for_platform(TargetPlatform::Switch1, SPACE),
            memory.clone(),
            19_200_000,
        )
        .unwrap(),
    ))
    .unwrap();
    let end = GuestVirtualAddress::new(0x1ffc);
    assert_eq!(value_at(&mut jit, end), 0);
    memory
        .overwrite_mapped_ram(SPACE, end, &0x91000400_u32.to_le_bytes())
        .unwrap();
    let mut state = ThreadState::default();
    state.set_pc(end.get());
    let result = run(&mut jit, &mut state);
    assert!(
        matches!(result.stop, CpuExit::FetchFault { fault } if fault.address.get() == 0x2000 && fault.reason == InstructionFetchFaultReason::Unmapped)
    );
    assert_eq!(state.pc(), 0x2000);
    assert_eq!(state.read_x(x(0)), 1);
    state.set_pc(0x1ffd);
    assert!(
        matches!(run(&mut jit, &mut state).stop, CpuExit::FetchFault { fault } if fault.reason == InstructionFetchFaultReason::Misaligned)
    );
}

#[test]
fn invalidation_retires_native_links_to_a_different_code_page() {
    let mut memory = executable_memory(&[0xd2800540, 0x140007ff]); // MOV #42; B 0x3000
    let target = GuestVirtualAddress::new(0x3000);
    let page = GuestPhysicalPageId::new(2);
    assert!(memory.add_ram_page(page));
    memory
        .initialize_ram(
            page,
            0,
            &[0xd2800560_u32.to_le_bytes(), 0xd4000001_u32.to_le_bytes()].concat(),
        )
        .unwrap();
    assert!(memory.map_page(SPACE, target, page, MemoryPermissions::READ_EXECUTE));
    memory
        .bind_cpu_memory_backend(SPACE, 0x10000, nixe_memory::DirectBackendPolicy::Required)
        .unwrap();
    let memory = Arc::new(memory);
    let mut jit = JitThread::new(Arc::new(
        JitProcess::new(
            ProcessCpuContext::for_platform(TargetPlatform::Switch1, SPACE),
            memory.clone(),
            19_200_000,
        )
        .unwrap(),
    ))
    .unwrap();
    for _ in 0..3 {
        assert_eq!(value_at(&mut jit, CODE), 43);
    }
    memory
        .overwrite_mapped_ram(SPACE, target, &0xd2800580_u32.to_le_bytes())
        .unwrap();
    assert_eq!(value_at(&mut jit, CODE), 44);
}
