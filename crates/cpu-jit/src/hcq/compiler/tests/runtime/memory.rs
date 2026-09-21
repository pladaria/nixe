use super::*;
use nixe_cpu::{
    execution::{ArchitecturalTimer, CpuExit, TimerSnapshot, VcpuEventState},
    memory::{
        CpuMemory, MemoryAccess, MemoryAccessSize, MemoryValue, ProcessMemory, SyntheticMmio,
    },
    profile::ProcessCpuContext,
};
use nixe_cpu_interpreter::{InstructionStep, InterpreterContext, execute_one_with_context};
use std::{cell::RefCell, sync::Mutex};

#[derive(Debug, PartialEq)]
struct Event(u64, MemoryAccess, Option<MemoryValue>);

struct Device {
    events: Arc<Mutex<Vec<Event>>>,
    fail_at: Option<usize>,
}

impl SyntheticMmio for Device {
    fn read(&mut self, offset: u64, access: MemoryAccess) -> Result<MemoryValue, Box<str>> {
        let mut events = self.events.lock().unwrap();
        events.push(Event(offset, access, None));
        if self.fail_at == Some(events.len()) {
            return Err("device rejected read".into());
        }
        Ok(MemoryValue::from_bits(
            access.size,
            0xfedc_ba98_7654_3210_8000_0000_8000_8080,
        ))
    }

    fn write(
        &mut self,
        offset: u64,
        access: MemoryAccess,
        value: MemoryValue,
    ) -> Result<(), Box<str>> {
        let mut events = self.events.lock().unwrap();
        events.push(Event(offset, access, Some(value)));
        if self.fail_at == Some(events.len()) {
            return Err("device rejected write".into());
        }
        Ok(())
    }
}

struct Timer;
impl ArchitecturalTimer for Timer {
    fn snapshot(&self) -> TimerSnapshot {
        TimerSnapshot {
            counter: 0,
            frequency: 1,
        }
    }
}

fn data_memory(events: Arc<Mutex<Vec<Event>>>, fail_at: Option<usize>) -> ExecutionMemory {
    let mut memory = ExecutionMemory::new();
    let ram = GuestPhysicalPageId::new(2);
    let mmio = GuestPhysicalPageId::new(3);
    assert!(memory.add_ram_page(ram));
    memory.initialize_ram(ram, 0, &[0x92; 4096]).unwrap();
    assert!(memory.add_mmio_page(mmio, Device { events, fail_at }));
    for (pc, page) in [(0x2000, ram), (0x3000, mmio)] {
        assert!(memory.map_page(
            AddressSpaceId::new(1),
            GuestVirtualAddress::new(pc),
            page,
            MemoryPermissions::READ_WRITE
        ));
    }
    memory
}

#[test]
fn hcq_runtime_repeated_memory_accesses_across_internal_edge_keep_effects_and_cost() {
    let graph = graph(&[
        (0x1000, &[0xf9400020, 0x14000001]), // LDR X0,[X1]; B 0x1008
        (0x1008, &[0x91000400, 0xf9000020, 0xf9400022, 0xd4200000]), // ADD; STR; LDR X2; BRK
    ]);
    let (mut reader, memory) = fixture_with_memory(
        &graph,
        &[0],
        &[],
        data_memory(Arc::new(Mutex::new(Vec::new())), None),
    );
    let mut state = A64State::default();
    state.set_pc(0x1000);
    state.general_register_storage_mut()[1] = 0x2000;
    let mut frame = NativeFrame::new(&mut state, PollBudget::new(100, 100).unwrap());
    let mut worker = WorkerFaultContext::register().unwrap();
    let exit = unsafe {
        invocation::run(
            &mut Samples::new(),
            &mut reader,
            &mut frame,
            &memory,
            &mut worker,
            &mut ExclusiveMonitorState::default(),
            key(0x1000),
        )
    }
    .unwrap()
    .unwrap();
    let invocation::Exit::Native {
        returned, guest, ..
    } = exit
    else {
        panic!()
    };
    assert_eq!(returned.reason, NativeExitReason::Architectural);
    assert_eq!(guest.pc.get(), 0x1014);
    assert_eq!(frame.budget.slice_remaining, 95);
    assert_eq!(frame.execution_epoch, 0);
    assert_eq!(
        state.general_register_storage_mut()[0],
        0x9292_9292_9292_9293
    );
    assert_eq!(
        state.general_register_storage_mut()[2],
        0x9292_9292_9292_9293
    );
    let mut bytes = [0; 8];
    memory
        .read_bytes(
            AddressSpaceId::new(1),
            GuestVirtualAddress::new(0x2000),
            &mut bytes,
        )
        .unwrap();
    assert_eq!(u64::from_le_bytes(bytes), 0x9292_9292_9292_9293);
}

// Exercise the production escape/owned-completion boundary, with an actual
// native prefix in RAM and a cold suffix in MMIO. The interpreter supplies
// architectural partial-commit, writeback and device-order expectations.
fn compound(word: u32, prefix_bytes: usize, fail_at: Option<usize>, entry_pc: u64) {
    let _restore = crate::fp_env::tests::RestoreHost::new();
    let caller = crate::fp_env::tests::distinct_caller();
    let graph = graph(&[
        (0x1000, &[0x1e622820, 0xf10004a5, 0x1400000e]), // FADD D0,D1,D2; SUBS X5,X5,#1; B 0x1040
        (0x1040, &[0x910004c6, word, 0xd4200000]),       // ADD X6,X6,#1; access; BRK
    ]);
    let entries = [0, block(&graph, 0x1040)];
    let events = Arc::new(Mutex::new(Vec::new()));
    let oracle_events = Arc::new(Mutex::new(Vec::new()));
    let (mut reader, memory) =
        fixture_with_memory(&graph, &entries, &[], data_memory(events.clone(), fail_at));
    let oracle = data_memory(oracle_events.clone(), fail_at);
    let mut state = A64State::default();
    state.set_pc(entry_pc);
    state.set_fpsr(1 << 27);
    state.general_register_storage_mut()[0] = 0x1234_5678_9abc_def0;
    state.general_register_storage_mut()[1] = 0x3000 - prefix_bytes as u64;
    state.general_register_storage_mut()[2] = 0xfedc_ba98_7654_3210;
    state.general_register_storage_mut()[5] = 1;
    for register in 0..32 {
        state.set_vector(register, u128::MAX - u128::from(register));
    }
    state.set_vector(1, u128::from(1.0f64.to_bits()));
    state.set_vector(2, u128::from(2.0f64.powi(-53).to_bits()));
    let mut expected = state.clone();
    let monitor = RefCell::new(ExclusiveMonitorState::default());
    let event_state = VcpuEventState::default();
    let expected_result = loop {
        let word = graph
            .instructions
            .iter()
            .find(|word| word.instruction.key.block_key().pc.get() == expected.pc())
            .unwrap();
        let result = execute_one_with_context(
            InterpreterContext::new(
                ProcessCpuContext::new(key(entry_pc).platform, AddressSpaceId::new(1)),
                &oracle,
                &monitor,
                &Timer,
                &event_state,
            ),
            &mut expected,
            word.instruction.bits,
        )
        .unwrap();
        if word.instruction.key.block_key().pc.get() == 0x1044 {
            break result;
        }
        assert!(matches!(result, InstructionStep::Continue));
    };
    let completed = if entry_pc == 0x1000 { 4 } else { 1 };
    let mut frame = NativeFrame::new(
        &mut state,
        PollBudget::new(completed + 1, completed + 1).unwrap(),
    );
    let mut worker = WorkerFaultContext::register().unwrap();
    let mut monitor = ExclusiveMonitorState::default();
    let mut samples = Samples::new();
    let exit = unsafe {
        invocation::run(
            &mut samples,
            &mut reader,
            &mut frame,
            &memory,
            &mut worker,
            &mut monitor,
            key(entry_pc),
        )
    }
    .unwrap()
    .unwrap();
    let invocation::Exit::Memory {
        instruction,
        outcome,
        completion_sample,
    } = exit
    else {
        panic!("compound access must escape")
    };
    assert!(matches!(outcome, invocation::MemoryExit::Cold(_)));
    assert!(
        completion_sample.is_none(),
        "HCQ completion must not invent LCQ seed heat"
    );
    assert_eq!(instruction.key.block_key(), key(0x1044));
    assert_eq!(instruction.bits, word);
    assert_eq!(frame.execution_epoch, 0);
    assert_eq!(frame.host_fp.saved, 0);
    assert_eq!(frame.budget.slice_remaining, 1);
    assert_eq!(frame.budget.sample_remaining, 1);
    assert!(
        events.lock().unwrap().is_empty(),
        "preparation cannot touch the device"
    );
    let mut budget = frame.budget;
    assert_eq!(state.pc(), 0x1044);
    assert_eq!(
        state.general_register_storage_mut()[1],
        0x3000 - prefix_bytes as u64
    );
    // Destruction and mapping changes must be legal now: Completion owns its
    // instruction semantics and retained first read, not a CodeUnit or lease.
    drop(worker);
    drop(reader);
    memory
        .set_permissions(
            AddressSpaceId::new(1),
            GuestVirtualAddress::new(0x1000),
            4096,
            MemoryPermissions::NONE,
        )
        .unwrap();
    let base = GuestVirtualAddress::new(0x3000 - prefix_bytes as u64);
    let mut observed = vec![0; prefix_bytes];
    let mut expected_bytes = vec![0; prefix_bytes];
    memory
        .read_bytes(AddressSpaceId::new(1), base, &mut observed)
        .unwrap();
    oracle
        .read_bytes(AddressSpaceId::new(1), base, &mut expected_bytes)
        .unwrap();
    assert_eq!(
        observed, expected_bytes,
        "native stores commit before escape"
    );
    memory
        .write_bytes(AddressSpaceId::new(1), base, &vec![0x77; prefix_bytes])
        .unwrap();
    let actual = outcome
        .complete(
            instruction,
            &mut state,
            &memory,
            &mut monitor,
            completed as u64,
        )
        .unwrap();
    match expected_result {
        InstructionStep::Continue => {
            assert!(actual.is_none());
            // The canonical caller charges a successfully completed guest
            // instruction once, independent of the number of subaccesses.
            let poll = budget.reconcile(budget.armed_span - 1, false).unwrap();
            assert!(poll.sample && poll.exhausted);
            assert_eq!(budget.slice_remaining, 0);
        }
        InstructionStep::Exit(CpuExit::DataFault { source, fault }) => {
            let Some(CpuExit::DataFault {
                source: actual_source,
                fault: actual_fault,
            }) = actual
            else {
                panic!("missing data fault")
            };
            assert_eq!(actual_source, source);
            assert_eq!(actual_fault, fault);
            assert_eq!(
                budget.slice_remaining, 1,
                "failed instruction earns no work"
            );
        }
        other => panic!("unexpected interpreter result: {other:?}"),
    }
    assert_eq!(
        state, expected,
        "word={word:08x}, entry={entry_pc:x}, fail={fail_at:?}"
    );
    assert_eq!(*events.lock().unwrap(), *oracle_events.lock().unwrap());
    memory
        .read_bytes(AddressSpaceId::new(1), base, &mut observed)
        .unwrap();
    assert_eq!(
        observed,
        vec![0x77; prefix_bytes],
        "completion must not replay the native prefix"
    );
    let mut host = crate::abi::HostFpState::default();
    unsafe {
        host.begin();
        host.finish();
    }
    assert_eq!((host.saved_control, host.saved_status), caller);
}

#[test]
fn hcq_runtime_pair_completion_preserves_retained_reads_and_native_store_prefix() {
    for vector in [false, true] {
        for load in [false, true] {
            // Post-indexed integer X0/X2 or vector Q0/Q2 pair via X1.
            let word = (if vector { 0xac00_0000 } else { 0xa800_0000 })
                | (u32::from(load) << 22)
                | (1 << 23)
                | (2 << 15)
                | (2 << 10)
                | (1 << 5);
            for fail_at in [None, Some(1)] {
                for entry in [0x1000, 0x1040] {
                    compound(word, if vector { 16 } else { 8 }, fail_at, entry);
                }
            }
        }
    }
}

#[test]
fn hcq_runtime_structure_completion_preserves_native_and_cold_partial_commits() {
    for (full, size, prefix) in [(true, 3, 3), (false, 0, 1)] {
        for load in [false, true] {
            // LD4/ST4 V31..V2, post-indexed by immediate. Include 64-bit
            // arrangements whose successful loads clear upper vector lanes.
            let word = 0x0c00_0000
                | (u32::from(full) << 30)
                | (u32::from(load) << 22)
                | (size << 10)
                | (1 << 23)
                | (31 << 16)
                | (1 << 5)
                | 31;
            for fail_at in [None, Some(1), Some(2)] {
                for entry in [0x1000, 0x1040] {
                    compound(word, prefix << size, fail_at, entry);
                }
            }
        }
    }
}

#[test]
fn hcq_runtime_exclusive_store_preserves_native_and_incoming_reservations() {
    for alias in [false, true] {
        let graph = graph(&[
            (0x1000, &[0xc85f7c22, 0x1400000f]), // LDXR X2,[X1]; B 0x1040
            (
                0x1040,
                &[
                    if alias { 0x91400821 } else { 0xd503201f }, // ADD X1,X1,#2,LSL #12 or NOP
                    0xc8037c20,                                  // STXR W3,X0,[X1]
                    0xd4200000,
                ],
            ),
        ]);
        let entries = [0, block(&graph, 0x1040)];
        for entry_pc in [0x1000, 0x1040] {
            for invalidate in [false, true] {
                // A same-VA store in the same invocation never escapes, so
                // there is no cold completion at which to invalidate it.
                if !alias && entry_pc == 0x1000 && invalidate {
                    continue;
                }
                let mut data = data_memory(Arc::new(Mutex::new(Vec::new())), None);
                assert!(data.map_page(
                    AddressSpaceId::new(1),
                    GuestVirtualAddress::new(0x4000),
                    GuestPhysicalPageId::new(2),
                    MemoryPermissions::READ_WRITE
                ));
                let (mut reader, memory) = fixture_with_memory(&graph, &entries, &[], data);
                let mut monitor = ExclusiveMonitorState::default();
                if entry_pc == 0x1040 {
                    let (_, reservation) = memory
                        .load_exclusive(
                            AddressSpaceId::new(1),
                            GuestVirtualAddress::new(0x2000),
                            MemoryAccess::normal(MemoryAccessSize::Doubleword),
                        )
                        .unwrap();
                    monitor.reserve(reservation);
                }
                let mut state = A64State::default();
                state.set_pc(entry_pc);
                state.general_register_storage_mut()[0] = 19;
                state.general_register_storage_mut()[1] = 0x2000;
                state.general_register_storage_mut()[3] = 99;
                let mut worker = WorkerFaultContext::register().unwrap();
                let mut frame = NativeFrame::new(&mut state, PollBudget::new(4096, 100).unwrap());
                let exit = unsafe {
                    invocation::run(
                        &mut Samples::new(),
                        &mut reader,
                        &mut frame,
                        &memory,
                        &mut worker,
                        &mut monitor,
                        key(entry_pc),
                    )
                }
                .unwrap()
                .unwrap();
                assert_eq!(frame.execution_epoch, 0);
                assert_eq!(frame.exclusive_load.bytes, 0);
                if !alias && entry_pc == 0x1000 {
                    let invocation::Exit::Native { guest, .. } = exit else {
                        panic!("same-invocation same-VA store should stay native")
                    };
                    assert_eq!(guest.pc.get(), 0x1048);
                    assert_eq!(frame.budget.slice_remaining, 96);
                    assert_eq!(state.general_register_storage_mut()[3], 0);
                } else {
                    let invocation::Exit::Memory {
                        instruction,
                        outcome,
                        ..
                    } = exit
                    else {
                        panic!("physical reservation must escape")
                    };
                    assert!(matches!(outcome, invocation::MemoryExit::ExclusiveStore(_)));
                    assert_eq!(instruction.key.block_key(), key(0x1044));
                    let completed = if entry_pc == 0x1000 { 3 } else { 1 };
                    assert_eq!(frame.budget.slice_remaining, 100 - completed);
                    assert_eq!(
                        monitor.reservation().unwrap().page,
                        GuestPhysicalPageId::new(2)
                    );
                    assert_eq!(state.general_register_storage_mut()[3], 99);
                    drop(worker);
                    drop(reader);
                    if invalidate {
                        memory
                            .write_bytes(
                                AddressSpaceId::new(1),
                                GuestVirtualAddress::new(0x2000),
                                &71u64.to_le_bytes(),
                            )
                            .unwrap();
                    }
                    assert!(
                        outcome
                            .complete(
                                instruction,
                                &mut state,
                                &memory,
                                &mut monitor,
                                completed as u64
                            )
                            .unwrap()
                            .is_none()
                    );
                    assert_eq!(state.pc(), 0x1048);
                    assert_eq!(
                        state.general_register_storage_mut()[3],
                        u64::from(invalidate)
                    );
                }
                assert!(monitor.reservation().is_none());
                let mut bytes = [0; 8];
                memory
                    .read_bytes(
                        AddressSpaceId::new(1),
                        GuestVirtualAddress::new(0x2000),
                        &mut bytes,
                    )
                    .unwrap();
                assert_eq!(u64::from_le_bytes(bytes), if invalidate { 71 } else { 19 });
            }
        }
    }
}
