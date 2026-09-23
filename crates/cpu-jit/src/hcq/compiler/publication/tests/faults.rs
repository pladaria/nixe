use super::lifecycle::{demand, promote_at, retire};
use super::negative::reshape;
use super::*;
use crate::lifetime::Error;
use nixe_cpu::{
    execution::{ArchitecturalTimer, CpuExit, TimerSnapshot, VcpuEventState},
    memory::{ProcessMemory, SyntheticMmio},
    profile::ProcessCpuContext,
};
use nixe_cpu_interpreter::{InstructionStep, InterpreterContext, execute_one_with_context};
use std::cell::RefCell;

mod retry;

#[derive(Debug, PartialEq)]
struct Event(u64, MemoryAccess, Option<MemoryValue>);

struct Device {
    events: Arc<Mutex<Vec<Event>>>,
    fail: bool,
}
impl SyntheticMmio for Device {
    fn read(&mut self, offset: u64, access: MemoryAccess) -> Result<MemoryValue, Box<str>> {
        self.events
            .lock()
            .unwrap()
            .push(Event(offset, access, None));
        if self.fail {
            return Err("device rejected read".into());
        }
        Ok(MemoryValue::from_bits(access.size, 0x9876_5432_1234_5678))
    }
    fn write(
        &mut self,
        offset: u64,
        access: MemoryAccess,
        value: MemoryValue,
    ) -> Result<(), Box<str>> {
        self.events
            .lock()
            .unwrap()
            .push(Event(offset, access, Some(value)));
        if self.fail {
            return Err("device rejected write".into());
        }
        Ok(())
    }
}

fn data(memory: &mut ExecutionMemory, events: Arc<Mutex<Vec<Event>>>, fail: bool) {
    assert!(memory.add_ram_page(GuestPhysicalPageId::new(8)));
    memory
        .initialize_ram(GuestPhysicalPageId::new(8), 0, &[0x92; 4096])
        .unwrap();
    assert!(memory.add_mmio_page(GuestPhysicalPageId::new(9), Device { events, fail }));
    for page in [8, 9] {
        assert!(memory.map_page(
            AddressSpaceId::new(1),
            GuestVirtualAddress::new(page * 4096),
            GuestPhysicalPageId::new(page),
            MemoryPermissions::READ_WRITE,
        ));
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

fn completion_after_reuse(word: u32, prefix: usize, fail: bool) {
    let _restore = crate::fp_env::tests::RestoreHost::new();
    let caller = crate::fp_env::tests::distinct_caller();
    let (process, mut memory, mut reader) = setup();
    let events = Arc::new(Mutex::new(Vec::new()));
    data(&mut memory, events.clone(), fail);
    let oracle_events = Arc::new(Mutex::new(Vec::new()));
    let mut oracle = ExecutionMemory::new();
    data(&mut oracle, oracle_events.clone(), fail);
    // Reuse the FP/lazy-NZCV prefix and compound instructions already covered
    // by compiler/tests/runtime/memory.rs, now through real publication.
    let prefix_words = [0x1e622820u32, 0xf10004a5, 0x140003fe]; // FADD; SUBS; B 0x2000.
    for (pc, words) in [
        (0x1000, &prefix_words[..]),
        (0x2000, &[word, 0xd4200000][..]),
    ] {
        let bytes: Vec<_> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
        memory
            .overwrite_mapped_ram(AddressSpaceId::new(1), GuestVirtualAddress::new(pc), &bytes)
            .unwrap();
        demand(&process, &memory, &mut reader, pc);
    }
    let predecessor = promote_at(&process, &memory, &mut reader, 0x2000);
    let old = process.snapshot(predecessor).unwrap();
    let address = old.code.allocation.address();
    let old_id = old.id;
    let old_version = old.version;
    let fault_pcs: Vec<_> = old
        .faults
        .iter()
        .map(|fault| address + fault.native_start as usize)
        .collect();
    assert!(!fault_pcs.is_empty());
    let mut state = A64State::default();
    state.set_pc(0x1000);
    state.set_fpsr(1 << 27);
    state.general_register_storage_mut()[0] = 0x1234_5678;
    state.general_register_storage_mut()[1] = 0x9000 - prefix as u64;
    state.general_register_storage_mut()[2] = 0xfeed_abcd;
    state.general_register_storage_mut()[5] = 1;
    for index in 0..32 {
        state.set_vector(index, u128::MAX - u128::from(index));
    }
    state.set_vector(1, u128::from(1.0f64.to_bits()));
    state.set_vector(2, u128::from(2.0f64.powi(-53).to_bits()));
    let mut expected = state.clone();
    let monitor = RefCell::new(ExclusiveMonitorState::default());
    let event_state = VcpuEventState::default();
    let mut expected_result = None;
    for word in prefix_words.into_iter().chain([word]) {
        expected_result = Some(
            execute_one_with_context(
                InterpreterContext::new(
                    ProcessCpuContext::new(key(0x1000).platform, AddressSpaceId::new(1)),
                    &oracle,
                    &monitor,
                    &Timer,
                    &event_state,
                ),
                &mut expected,
                word,
            )
            .unwrap(),
        );
    }
    let mut frame = NativeFrame::new(&mut state, PollBudget::new(4096, 100).unwrap());
    let mut worker = WorkerFaultContext::register().unwrap();
    let mut monitor = ExclusiveMonitorState::default();
    let exit = unsafe {
        invocation::run(
            &mut Samples::new(),
            &mut reader,
            &mut frame,
            &memory,
            &mut worker,
            &mut monitor,
            key(0x1000),
        )
    }
    .unwrap()
    .unwrap();
    let invocation::Exit::Memory {
        instruction,
        outcome,
        ..
    } = exit
    else {
        panic!("expected compound escape")
    };
    assert!(matches!(outcome, invocation::MemoryExit::Cold(_)));
    assert_eq!(instruction.bits, word);
    assert_eq!(instruction.key.block_key(), key(0x2000));
    assert_eq!(frame.execution_epoch, 0);
    assert_eq!(frame.host_fp.saved, 0);
    assert_eq!(frame.budget.slice_remaining, 97);
    assert!(events.lock().unwrap().is_empty());
    drop(worker);

    let work = reshape(&process, &mut reader, 0x1000, 0x1008, 0x2000);
    let frozen = work
        .reserve_candidate(Graph::discover(&work).unwrap())
        .unwrap()
        .freeze()
        .unwrap();
    let mut observing_state = A64State::default();
    let mut observing_frame =
        NativeFrame::new(&mut observing_state, PollBudget::new(4096, 100).unwrap());
    let observing = unsafe { reader.admit(&mut observing_frame, key(0x2000)) }
        .unwrap()
        .unwrap();
    let successor = std::thread::scope(|scope| {
        scope
            .spawn(|| {
                Compiler::new(host(), 0x10000)
                    .unwrap()
                    .publish(
                        &mut Context::new(),
                        &mut FunctionBuilderContext::new(),
                        &frozen,
                        &memory,
                    )
                    .unwrap()
            })
            .join()
            .unwrap()
    });
    assert!(!process.try_service_links().unwrap());
    for pc in &fault_pcs {
        let found = observing.fault(*pc).unwrap();
        assert_eq!(found.unit.id, old_id);
        assert_eq!(found.unit.version, old_version);
        assert_eq!(found.instruction().bits, word);
    }
    drop(observing);
    process.try_service_links().unwrap();
    drop(frozen);
    drop(work);
    drop(old);
    assert_eq!(process.reclaim_units().unwrap(), 1);
    assert!(matches!(
        process.snapshot(predecessor),
        Err(Error::StaleUnit)
    ));
    // Restore baselines, then compile the identical small body into the freed
    // span. The outstanding cold completion must own no native metadata.
    retire(&process, successor);
    process.reclaim_units().unwrap();
    let current_handle = promote_at(&process, &memory, &mut reader, 0x2000);
    let current = process.snapshot(current_handle).unwrap();
    assert_eq!(current.code.allocation.address(), address);
    assert_ne!(current.id, old_id);
    assert_ne!(current.version, old_version);
    let observing = unsafe { reader.admit(&mut observing_frame, key(0x2000)) }
        .unwrap()
        .unwrap();
    for pc in &fault_pcs {
        let found = observing.fault(*pc).unwrap();
        assert_eq!(found.unit.id, current.id);
        assert_eq!(found.unit.version, current.version);
    }
    drop(observing);
    drop(current);
    // Remove even the new source's executable authority. Finishing the already
    // started access must not re-fetch its instruction or borrow replacement
    // metadata just because its old native address happened to be reused.
    memory
        .set_permissions(
            AddressSpaceId::new(1),
            GuestVirtualAddress::new(0x2000),
            4096,
            MemoryPermissions::NONE,
        )
        .unwrap();
    process.try_service_links().unwrap();
    process.reclaim_units().unwrap();
    assert!(matches!(
        process.snapshot(current_handle),
        Err(Error::StaleUnit)
    ));

    let base = GuestVirtualAddress::new(0x9000 - prefix as u64);
    let mut actual_bytes = vec![0; prefix];
    let mut expected_bytes = vec![0; prefix];
    memory
        .read_bytes(AddressSpaceId::new(1), base, &mut actual_bytes)
        .unwrap();
    oracle
        .read_bytes(AddressSpaceId::new(1), base, &mut expected_bytes)
        .unwrap();
    assert_eq!(actual_bytes, expected_bytes); // Native stores already committed.
    memory
        .write_bytes(AddressSpaceId::new(1), base, &vec![0x77; prefix])
        .unwrap();
    let actual = outcome
        .complete(instruction, &mut state, &memory, &mut monitor, 3)
        .unwrap();
    match expected_result.unwrap() {
        InstructionStep::Continue => assert!(actual.is_none()),
        InstructionStep::Exit(CpuExit::DataFault { source, fault }) => {
            let Some(CpuExit::DataFault {
                source: actual_source,
                fault: actual_fault,
            }) = actual
            else {
                panic!("expected device fault")
            };
            assert_eq!(actual_source, source);
            assert_eq!(actual_fault, fault);
        }
        other => panic!("unexpected oracle result: {other:?}"),
    }
    assert_eq!(state, expected);
    assert_eq!(*events.lock().unwrap(), *oracle_events.lock().unwrap());
    memory
        .read_bytes(AddressSpaceId::new(1), base, &mut actual_bytes)
        .unwrap();
    assert_eq!(actual_bytes, vec![0x77; prefix]); // No native-prefix replay.
    let mut host = crate::abi::HostFpState::default();
    unsafe {
        host.begin();
        host.finish();
    }
    assert_eq!((host.saved_control, host.saved_status), caller);
    assert!(process.try_shutdown().unwrap());
    assert_eq!(process.executable_cache().usage().unwrap().committed, 0);
}

#[test]
fn real_hcq_pair_completion_survives_replacement_and_exact_fault_span_reuse() {
    for load in [false, true] {
        let word = 0xa8800000 | (u32::from(load) << 22) | (2 << 15) | (2 << 10) | (1 << 5);
        for fail in [false, true] {
            completion_after_reuse(word, 8, fail);
        }
    }
}

#[test]
fn real_hcq_structure_completion_survives_replacement_and_exact_fault_span_reuse() {
    for load in [false, true] {
        let word = 0x4c000000
            | (u32::from(load) << 22)
            | (3 << 10)
            | (1 << 23)
            | (31 << 16)
            | (1 << 5)
            | 31;
        for fail in [false, true] {
            completion_after_reuse(word, 24, fail);
        }
    }
}
