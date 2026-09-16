use super::*;
use nixe_cpu::execution::{ArchitecturalTimer, CpuExit, TimerSnapshot, VcpuEventState};

struct Timer;
impl ArchitecturalTimer for Timer {
    fn snapshot(&self) -> TimerSnapshot {
        TimerSnapshot {
            counter: 0,
            frequency: 19_200_000,
        }
    }
}

pub(super) fn signal_stack() -> (*mut libc::c_void, usize) {
    let mut stack = std::mem::MaybeUninit::<libc::stack_t>::uninit();
    assert_eq!(
        unsafe { libc::sigaltstack(std::ptr::null(), stack.as_mut_ptr()) },
        0
    );
    let stack = unsafe { stack.assume_init() };
    (stack.ss_sp, stack.ss_size)
}

fn data_fault(thread: &mut JitThread, worker: &mut NativeWorker) {
    let mut state = A64State::default();
    state.set_pc(PC.get());
    state.general_register_storage_mut()[1] = 0x5000;
    let report = thread
        .run_slice(
            &mut crate::ReturnStack::default(),
            worker,
            &mut state,
            10,
            &Timer,
            &VcpuEventState::default(),
        )
        .unwrap();
    assert!(
        matches!(report.stop, CpuExit::DataFault { source, fault } if source.pc.get() == PC.get() + 4 && fault.address.get() == 0x5000)
    );
    assert_eq!(report.progress, 1);
    assert_eq!(report.context, Some(state.register_context()));
}

#[test]
fn one_os_worker_serves_multiple_processes_and_survives_either_retirement_order() {
    for retire_first in [true, false] {
        let previous = signal_stack();
        let mut worker = NativeWorker::default();
        let mut first = budget::setup(&[0xd503201f, 0xf9400020], false); // NOP; LDR X0,[X1].
        let mut second = budget::setup(&[0xd503201f, 0xf9400020], false);
        data_fault(&mut first, &mut worker);
        let installed = signal_stack();
        assert_ne!(installed, previous);
        data_fault(&mut second, &mut worker);
        assert_eq!(signal_stack(), installed);
        let (retired, mut survivor) = if retire_first {
            (first, second)
        } else {
            (second, first)
        };
        // Retiring a process on the coordinator's thread cannot free or restore
        // the executing worker's signal stacks, even after native execution.
        std::thread::spawn(move || {
            assert!(retired.process.try_shutdown().unwrap());
            drop(retired);
        })
        .join()
        .unwrap();
        assert_eq!(signal_stack(), installed);
        data_fault(&mut survivor, &mut worker);
        assert!(survivor.process.try_shutdown().unwrap());
        drop(survivor);
        assert_eq!(signal_stack(), installed);
        worker.finish().unwrap();
        worker.finish().unwrap();
        assert_eq!(signal_stack(), previous);
    }
}

#[test]
fn interpreter_and_lcq_share_one_worker_and_survive_either_backend_retirement() {
    use nixe_cpu::execution::{CpuThreadId, MemoryBinding};
    use nixe_cpu_interpreter::{InterpreterProcess, InterpreterRunRequest};
    use nixe_memory::MemoryInvalidationSource;

    for interpreter_first in [true, false] {
        for retire_interpreter in [true, false] {
            let previous = signal_stack();
            let mut worker = NativeWorker::default();
            let mut jit = budget::setup(&[0xd503201f, 0xf9400020], false);
            let memory = memory(DirectBackendPolicy::Required);
            memory
                .overwrite_mapped_ram(SPACE, PC, &0xf9400020_u32.to_le_bytes())
                .unwrap();
            let mut process = InterpreterProcess::new(cpu());
            process
                .bind_memory(MemoryBinding {
                    address_space: SPACE,
                    end_exclusive: GuestVirtualAddress::new(0x10000),
                    memory: memory.as_ref(),
                    mapping_epoch: memory.mapping_epoch().get(),
                    invalidation_cursor: memory.invalidation_cursor(),
                })
                .unwrap();
            let mut interpreter = process.create_thread(CpuThreadId::new(1)).unwrap();
            let run_interpreter = |thread: &mut nixe_cpu_interpreter::InterpreterThread,
                                   worker: &mut NativeWorker| {
                let mut state = A64State::default();
                state.set_pc(PC.get());
                state.general_register_storage_mut()[1] = 0x5000;
                let report = thread
                    .run_slice(
                        worker,
                        InterpreterRunRequest {
                            memory: memory.as_ref(),
                            memory_lease: Some(memory.acquire_execution_lease()),
                            state: &mut state,
                            instruction_budget: 1,
                            timer: &Timer,
                            events: VcpuEventState::default(),
                        },
                    )
                    .unwrap();
                assert!(
                    matches!(report.stop, CpuExit::DataFault { fault, .. } if fault.address.get() == 0x5000)
                );
                assert_eq!(state.pc(), PC.get());
            };
            assert_eq!(signal_stack(), previous);
            if interpreter_first {
                run_interpreter(&mut interpreter, &mut worker);
            } else {
                data_fault(&mut jit, &mut worker);
            }
            let installed = signal_stack();
            assert_ne!(installed, previous);
            run_interpreter(&mut interpreter, &mut worker);
            data_fault(&mut jit, &mut worker);
            assert_eq!(signal_stack(), installed);

            if retire_interpreter {
                std::thread::spawn(move || drop((interpreter, process)))
                    .join()
                    .unwrap();
                data_fault(&mut jit, &mut worker);
                assert!(jit.process.try_shutdown().unwrap());
                drop(jit);
            } else {
                std::thread::spawn(move || {
                    assert!(jit.process.try_shutdown().unwrap());
                    drop(jit);
                })
                .join()
                .unwrap();
                run_interpreter(&mut interpreter, &mut worker);
                drop((interpreter, process));
            }
            assert_eq!(signal_stack(), installed);
            worker.finish().unwrap();
            assert_eq!(signal_stack(), previous);
        }
    }
}
