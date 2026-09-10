use super::*;
use nixe_cpu::memory::{
    ExecutionMemory, MemoryAccess, MemoryAccessSize, MemoryValue, SyntheticMmio,
};
use nixe_memory::DirectBackendPolicy;
use std::sync::Mutex;

#[derive(Debug, PartialEq)]
struct Event {
    offset: u64,
    access: MemoryAccess,
    value: Option<MemoryValue>,
}
#[derive(Clone)]
struct Device {
    events: Arc<Mutex<Vec<Event>>>,
    fail_at: Option<usize>,
    wrong_size: bool,
}
impl SyntheticMmio for Device {
    fn read(&mut self, offset: u64, access: MemoryAccess) -> Result<MemoryValue, Box<str>> {
        self.events.lock().unwrap().push(Event {
            offset,
            access,
            value: None,
        });
        if self.fail_at == Some(self.events.lock().unwrap().len()) {
            return Err("device rejected read".into());
        }
        Ok(MemoryValue::from_bits(
            if self.wrong_size {
                MemoryAccessSize::Byte
            } else {
                access.size
            },
            0xfedc_ba98_7654_3210_8000_0000_8000_8080,
        ))
    }
    fn write(
        &mut self,
        offset: u64,
        access: MemoryAccess,
        value: MemoryValue,
    ) -> Result<(), Box<str>> {
        self.events.lock().unwrap().push(Event {
            offset,
            access,
            value: Some(value),
        });
        if self.fail_at == Some(self.events.lock().unwrap().len()) {
            Err("device rejected write".into())
        } else {
            Ok(())
        }
    }
}

fn setup(words: &[u32], device: Device) -> ExecutionMemory {
    let mut memory = ExecutionMemory::new();
    let code = GuestPhysicalPageId::new(1);
    let mmio = GuestPhysicalPageId::new(2);
    assert!(memory.add_ram_page(code));
    let bytes: Vec<_> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    memory.initialize_ram(code, 0, &bytes).unwrap();
    assert!(memory.add_mmio_page(mmio, device));
    assert!(memory.map_page(
        SPACE,
        GuestVirtualAddress::new(PC),
        code,
        MemoryPermissions::READ_EXECUTE
    ));
    assert!(memory.map_page(
        SPACE,
        GuestVirtualAddress::new(0x3000),
        mmio,
        MemoryPermissions::READ_WRITE
    ));
    memory
        .bind_cpu_memory_backend(SPACE, ARENA as u64, DirectBackendPolicy::Required)
        .unwrap();
    memory
}

fn check(word: u32, fail: bool, wrong_size: bool) {
    check_at(word, fail.then_some(1), wrong_size);
}

fn check_at(word: u32, fail_at: Option<usize>, wrong_size: bool) {
    let words = [0xf100_04a5, 0x9100_0421, word, 0xd420_0000]; // dirty NZCV and X1
    let events = Arc::new(Mutex::new(Vec::new()));
    let memory = setup(
        &words,
        Device {
            events: events.clone(),
            fail_at,
            wrong_size,
        },
    );
    let expected_events = Arc::new(Mutex::new(Vec::new()));
    let oracle = setup(
        &words,
        Device {
            events: expected_events.clone(),
            fail_at,
            wrong_size,
        },
    );
    let mut state = A64State::default();
    state.set_pc(PC);
    state.set_fpsr(0x0800_009f);
    state.set_fpcr(1 << 22);
    state.general_register_storage_mut()[0] = 0xfeed_face_dead_beef;
    state.general_register_storage_mut()[1] = 0x301f;
    state.general_register_storage_mut()[2] = 1;
    state.general_register_storage_mut()[5] = 1;
    *state.stack_pointer_storage_mut() = 0x3020;
    for index in 0..32 {
        state.set_vector(index, u128::MAX - u128::from(index));
    }
    let mut expected = state.clone();
    let monitor = RefCell::new(ExclusiveMonitorState::default());
    let event_state = VcpuEventState::default();
    let mut result = InstructionStep::Continue;
    for &word in &words[..3] {
        result = execute_one_with_context(
            InterpreterContext::new(
                ProcessCpuContext::new(TargetPlatform::Switch1, SPACE),
                &oracle,
                &monitor,
                &Timer,
                &event_state,
            ),
            &mut expected,
            word,
        )
        .unwrap();
    }
    let completion = authority::prepare_cold(&memory, &mut state);
    assert!(
        events.lock().unwrap().is_empty(),
        "classification/preparation executes no device operation"
    );
    let result_actual =
        completion.complete(&mut state, &memory, &mut ExclusiveMonitorState::default());
    match result {
        InstructionStep::Continue => result_actual.unwrap(),
        InstructionStep::Exit(nixe_cpu::execution::CpuExit::DataFault { fault, .. }) => {
            let Err(crate::lcq::fault::cold::Error::Data(actual)) = result_actual else {
                panic!("{result_actual:?}");
            };
            assert_eq!(actual, fault);
        }
        other => panic!("unexpected interpreter exit {other:?}"),
    }
    assert_eq!(
        state, expected,
        "word={word:08x} fail_at={fail_at:?} wrong_size={wrong_size}"
    );
    assert_eq!(*events.lock().unwrap(), *expected_events.lock().unwrap());
    assert!(!events.lock().unwrap().is_empty());
}

#[test]
fn delivered_lcq_single_cold_completion_matches_interpreter() {
    for size in 0..4 {
        for opc in 0..4 {
            if nixe_cpu::semantics::a64::scalar_transfer(
                opc,
                nixe_cpu::semantics::a64::memory_size(size),
            )
            .is_none()
            {
                continue;
            }
            for rt in [0, 31] {
                check(
                    0x3900_0000 | (u32::from(size) << 30) | (u32::from(opc) << 22) | (1 << 5) | rt,
                    false,
                    false,
                );
            }
        }
        for form in [0x08df_fc00, 0x089f_fc00] {
            // natural alignment and acquire/release
            check(form | (u32::from(size) << 30) | (1 << 5), false, false);
        }
    }
    for (size, opc) in [(0, 0), (1, 0), (2, 0), (3, 0), (0, 2)] {
        for load in [0, 1] {
            check(
                0x3d00_0000 | (size << 30) | ((opc + load) << 22) | (1 << 5) | 31,
                false,
                false,
            );
        }
    }
    for word in [
        0xf840_8420,
        0xf800_8420, // scalar post-index
        0xf840_8c20,
        0xf800_8c20, // scalar pre-index
        0x3cc1_0420,
        0x3c81_0420, // Q post-index
        0x3cc1_0c20,
        0x3c81_0c20, // Q pre-index
        0xf840_87e0,
        0xf800_87e0, // SP writeback
        0xf862_7820,
        0x3ce2_d820, // register offsets
        0x5801_0000, // literal: PC+0x2000
    ] {
        check(word, false, false);
    }
}

#[test]
fn delivered_lcq_cold_device_errors_preserve_pre_state_without_retry() {
    for word in [
        0xf840_8420,
        0xf800_8c20,
        0x3cc1_0420,
        0x3c81_0c20,
        0xf840_87e0,
    ] {
        check(word, true, false);
    }
    check(0xf840_8420, false, true); // bad device result width must not commit load/base/PC
}

#[test]
fn delivered_lcq_pair_cold_completion_matches_interpreter_and_failure_stages() {
    for vector in [false, true] {
        for size in 0..3 {
            for load in [0, 1] {
                if !vector && size == 1 && load == 0 {
                    continue;
                }
                for mode in 0..4 {
                    if !vector && size == 1 && mode == 0 {
                        continue;
                    }
                    for immediate in [2, 0x7e] {
                        let form = (if vector { 0x2c00_0000 } else { 0x2800_0000 })
                            | (size << 30)
                            | (load << 22)
                            | (mode << 23)
                            | (immediate << 15);
                        for (rn, rt, rt2) in [(1, 0, 3), (31, 31, 0)] {
                            check_at(form | (rn << 5) | rt | (rt2 << 10), None, false);
                        }
                    }
                }
            }
        }
    }
    // No writeback: a load destination may alias the PRE address base.
    check_at(0xa940_0021, None, false);
    for word in [0xa8c1_0c20, 0xa881_0c20, 0xacc1_0c20, 0xac81_0c20] {
        for stage in [1, 2] {
            check_at(word, Some(stage), false);
        }
    }
}

#[test]
fn delivered_lcq_pair_cold_completion_does_not_replay_the_native_prefix() {
    for vector in [false, true] {
        for size in 0..3 {
            for load in [false, true] {
                if !vector && size == 1 && !load {
                    continue;
                }
                for fail in [false, true] {
                    let bytes = if vector {
                        4 << size
                    } else if size == 2 {
                        8
                    } else {
                        4
                    };
                    let word = (if vector { 0x2c00_0000 } else { 0x2800_0000 })
                        | (size << 30)
                        | (u32::from(load) << 22)
                        | (1 << 23)
                        | (2 << 15)
                        | (3 << 10)
                        | (1 << 5);
                    let events = Arc::new(Mutex::new(Vec::new()));
                    let mut memory = setup(
                        &[word, 0xd420_0000],
                        Device {
                            events: events.clone(),
                            fail_at: fail.then_some(1),
                            wrong_size: false,
                        },
                    );
                    let ram = GuestPhysicalPageId::new(3);
                    assert!(memory.add_ram_page(ram));
                    memory.initialize_ram(ram, 0, &[0x92; 4096]).unwrap();
                    assert!(memory.map_page(
                        SPACE,
                        GuestVirtualAddress::new(0x2000),
                        ram,
                        MemoryPermissions::READ_WRITE
                    ));
                    let base = 0x3000 - bytes;
                    let mut state = A64State::default();
                    state.set_pc(PC);
                    state.general_register_storage_mut()[1] = base;
                    state.general_register_storage_mut()[0] = 0x0123_4567_89ab_cdef;
                    state.general_register_storage_mut()[3] = 0xfedc_ba98_7654_3210;
                    state.set_vector(0, u128::MAX);
                    state.set_vector(3, 0xfedc_ba98_7654_3210_0123_4567_89ab_cdef);
                    let mut expected = state.clone();
                    let completion = authority::prepare_cold(&memory, &mut state);
                    assert_eq!(
                        state, expected,
                        "pair destinations/base stay PRE after escape"
                    );
                    let mut observed = vec![0; bytes as usize];
                    memory
                        .read_bytes(SPACE, GuestVirtualAddress::new(base), &mut observed)
                        .unwrap();
                    let initial = if load {
                        [0x92; 16]
                    } else if vector {
                        [0xff; 16]
                    } else {
                        0x0123_4567_89ab_cdefu128.to_le_bytes()
                    };
                    assert_eq!(observed, initial[..bytes as usize]);
                    // After escaping, change the earlier RAM bytes. A load must
                    // consume retained bits; a store must never rewrite them.
                    memory
                        .write_bytes(
                            SPACE,
                            GuestVirtualAddress::new(base),
                            &vec![0x77; bytes as usize],
                        )
                        .unwrap();
                    let result = completion.complete(
                        &mut state,
                        &memory,
                        &mut ExclusiveMonitorState::default(),
                    );
                    if fail {
                        assert!(matches!(
                            result,
                            Err(crate::lcq::fault::cold::Error::Data(_))
                        ));
                    } else {
                        result.unwrap();
                        if load {
                            let first = u128::from_le_bytes([0x92; 16]) >> (128 - bytes * 8);
                            let second = 0xfedc_ba98_7654_3210_8000_0000_8000_8080u128;
                            let mask = u128::MAX >> (128 - bytes * 8);
                            if vector {
                                expected.set_vector(0, first);
                                expected.set_vector(3, second & mask);
                            } else {
                                expected.general_register_storage_mut()[0] = if size == 1 {
                                    0xffff_ffff_9292_9292
                                } else {
                                    first as u64
                                };
                                expected.general_register_storage_mut()[3] = if size == 1 {
                                    0xffff_ffff_8000_8080
                                } else {
                                    (second & mask) as u64
                                };
                            }
                        }
                        expected.general_register_storage_mut()[1] = base + 2 * bytes;
                        expected.set_pc(PC + 4);
                    }
                    assert_eq!(
                        state, expected,
                        "vector={vector} size={size} load={load} fail={fail}"
                    );
                    let mut first_bytes = vec![0; bytes as usize];
                    memory
                        .read_bytes(SPACE, GuestVirtualAddress::new(base), &mut first_bytes)
                        .unwrap();
                    assert_eq!(first_bytes, vec![0x77; bytes as usize]);
                    let events = events.lock().unwrap();
                    assert_eq!(events.len(), 1);
                    assert_eq!(events[0].offset, 0);
                    assert_eq!(events[0].access.size.bytes(), bytes as usize);
                    if !load {
                        assert_eq!(
                            events[0].value.unwrap().bits(),
                            if vector {
                                0xfedc_ba98_7654_3210_0123_4567_89ab_cdef
                                    & (u128::MAX >> (128 - bytes * 8))
                            } else {
                                0xfedc_ba98_7654_3210u128 & (u128::MAX >> (128 - bytes * 8))
                            }
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn delivered_lcq_structure_cold_completion_matches_interpreter_and_failure_stages() {
    for count in 1..=4 {
        for size in 0..4 {
            for (post, rm, rn, rt) in [
                (false, 0, 1, 31),
                (true, 31, 31, 30),
                (true, 2, 1, 0),
                (true, 1, 1, 31),
            ] {
                let address = (u32::from(post) << 23) | (rm << 16) | (rn << 5) | rt;
                for lane in [0, (16 >> size) - 1] {
                    for load in [false, true] {
                        check(
                            single_structure_word(count, size, Some(lane), load, false) | address,
                            false,
                            false,
                        );
                    }
                }
                for full in [false, true] {
                    check(
                        single_structure_word(count, size, None, true, full) | address,
                        false,
                        false,
                    );
                }
            }
            for lane in [Some((16 >> size) - 1), None] {
                for load in [false, true] {
                    if lane.is_none() && !load {
                        continue;
                    }
                    let word = single_structure_word(count, size, lane, load, false)
                        | (1 << 23)
                        | (31 << 16)
                        | (1 << 5)
                        | 31;
                    for failure in 1..=count as usize {
                        check_at(word, Some(failure), false);
                    }
                    if load && size > 0 {
                        check_at(word, None, true);
                    }
                }
            }
        }
    }
    for (opcode, count) in [(8, 2), (4, 3), (0, 4)] {
        for full in [false, true] {
            for size in 0..4 {
                if !full && size == 3 {
                    continue;
                }
                for load in [false, true] {
                    for (post, rm, rn, rt) in [
                        (false, 0, 1, 31),
                        (true, 31, 31, 30),
                        (true, 2, 1, 0),
                        (true, 1, 1, 31),
                    ] {
                        let word = 0x0c00_0000
                            | (u32::from(full) << 30)
                            | (u32::from(post) << 23)
                            | (u32::from(load) << 22)
                            | (rm << 16)
                            | (opcode << 12)
                            | (size << 10)
                            | (rn << 5)
                            | rt;
                        check_at(word, None, false);
                        if post && rm == 31 {
                            // Every failure stage, including upper clearing on
                            // the first lane and the maximum 64th access.
                            for failure in 1..=count * ((if full { 16 } else { 8 }) >> size) {
                                check_at(word, Some(failure), false);
                            }
                            if load && size > 0 {
                                check_at(word, None, true);
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn delivered_scalar_rmw_cold_completion_revalidates_ram_and_rejects_device_atomics() {
    use nixe_cpu::memory::{DataAccessFaultReason, DataAccessKind};
    for size in 0..4 {
        for opcode in 0..9 {
            for case in 0..4 {
                let word = super::atomic::rmw(size, opcode, 3, 1, 2, 2);
                let events = Arc::new(Mutex::new(Vec::new()));
                let mut memory = setup(
                    &[word, 0xd420_0000],
                    Device {
                        events: events.clone(),
                        fail_at: None,
                        wrong_size: false,
                    },
                );
                let mut state = A64State::default();
                state.set_pc(PC);
                state.general_register_storage_mut()[1] = 0x3000;
                state.general_register_storage_mut()[2] = 0x0123_4567_89ab_cdef;
                let mut expected = state.clone();
                let completion = authority::prepare_cold(&memory, &mut state);
                if case != 0 {
                    memory
                        .resize_zeroed_mapping(
                            SPACE,
                            GuestVirtualAddress::new(0x3000),
                            4096,
                            0,
                            MemoryPermissions::READ_WRITE,
                            nixe_cpu::memory::MemoryMappingPurpose::Normal,
                        )
                        .unwrap();
                    let page = GuestPhysicalPageId::new(3);
                    assert!(memory.add_ram_page(page));
                    memory.initialize_ram(page, 0, &[0x92; 4096]).unwrap();
                    assert!(memory.map_page(
                        SPACE,
                        GuestVirtualAddress::new(0x3000),
                        page,
                        match case {
                            1 => MemoryPermissions::READ_WRITE,
                            2 => MemoryPermissions::READ,
                            _ => MemoryPermissions::NONE,
                        }
                    ));
                }
                let result =
                    completion.complete(&mut state, &memory, &mut ExclusiveMonitorState::default());
                if case == 1 {
                    result.unwrap();
                    let bytes = 1 << size;
                    let width = nixe_cpu::semantics::a64::memory_size(size as u8);
                    let previous = MemoryValue::from_bits(width, 0x9292_9292_9292_9292);
                    let operand = MemoryValue::from_bits(width, 0x0123_4567_89ab_cdef);
                    let new = nixe_cpu::semantics::a64::atomic_rmw_kind(opcode as u8)
                        .unwrap()
                        .apply(previous, operand)
                        .unwrap();
                    expected.general_register_storage_mut()[2] = previous.bits() as u64;
                    expected.set_pc(PC + 4);
                    let mut observed = [0; 8];
                    memory
                        .read_bytes(SPACE, GuestVirtualAddress::new(0x3000), &mut observed)
                        .unwrap();
                    let mut wanted = [0x92; 8];
                    wanted[..bytes].copy_from_slice(&new.bits().to_le_bytes()[..bytes]);
                    assert_eq!(observed, wanted);
                } else {
                    let Err(crate::lcq::fault::cold::Error::Data(fault)) = result else {
                        panic!("{result:?}");
                    };
                    assert_eq!(
                        fault.reason,
                        match case {
                            0 => DataAccessFaultReason::AtomicRegionUnsupported,
                            2 => DataAccessFaultReason::WritePermissionDenied,
                            _ => DataAccessFaultReason::ReadPermissionDenied,
                        }
                    );
                    assert_eq!(
                        fault.kind,
                        if case == 3 {
                            DataAccessKind::Read
                        } else {
                            DataAccessKind::Write
                        }
                    );
                }
                assert_eq!(state, expected);
                assert!(
                    events.lock().unwrap().is_empty(),
                    "RMW must not become separate device calls"
                );
            }
        }
    }
}

#[test]
fn delivered_cas_casp_cold_completion_revalidates_ram_and_rejects_device_atomics() {
    use nixe_cpu::memory::{DataAccessFaultReason, DataAccessKind};
    for (size, pair) in [
        (0, false),
        (1, false),
        (2, false),
        (3, false),
        (3, true),
        (4, true),
    ] {
        for ordering in 0..4 {
            for case in 0..5 {
                let word = if pair {
                    super::casp::word(ordering, 1, 2, 4) | ((size - 3) << 30)
                } else {
                    0x08a2_7c23 | (size << 30) | ((ordering & 1) << 22) | ((ordering >> 1) << 15)
                };
                let events = Arc::new(Mutex::new(Vec::new()));
                let mut memory = setup(
                    &[word, 0xd420_0000],
                    Device {
                        events: events.clone(),
                        fail_at: None,
                        wrong_size: false,
                    },
                );
                let mut state = A64State::default();
                state.set_pc(PC);
                state.general_register_storage_mut()[1] = 0x3000;
                state.general_register_storage_mut()[2] =
                    if case == 2 { 0 } else { 0x9292_9292_9292_9292 };
                state.general_register_storage_mut()[3] = 0x0123_4567_89ab_cdef;
                if pair {
                    state.general_register_storage_mut()[3] = 0xffff_ffff_9292_9292;
                    state.general_register_storage_mut()[4] = 0xeeee_eeee_89ab_cdef;
                    state.general_register_storage_mut()[5] = 0xdddd_dddd_0123_4567;
                    if size == 4 {
                        state.general_register_storage_mut()[3] = 0x9292_9292_9292_9292;
                    }
                }
                let mut expected = state.clone();
                let completion = authority::prepare_cold(&memory, &mut state);
                if case != 0 {
                    memory
                        .resize_zeroed_mapping(
                            SPACE,
                            GuestVirtualAddress::new(0x3000),
                            4096,
                            0,
                            MemoryPermissions::READ_WRITE,
                            nixe_cpu::memory::MemoryMappingPurpose::Normal,
                        )
                        .unwrap();
                    let page = GuestPhysicalPageId::new(3);
                    assert!(memory.add_ram_page(page));
                    memory.initialize_ram(page, 0, &[0x92; 4096]).unwrap();
                    assert!(memory.map_page(
                        SPACE,
                        GuestVirtualAddress::new(0x3000),
                        page,
                        match case {
                            3 => MemoryPermissions::READ,
                            4 => MemoryPermissions::NONE,
                            _ => MemoryPermissions::READ_WRITE,
                        }
                    ));
                }
                let result =
                    completion.complete(&mut state, &memory, &mut ExclusiveMonitorState::default());
                if matches!(case, 1 | 2) {
                    result.unwrap();
                    let bytes = 1 << size;
                    expected.general_register_storage_mut()[2] =
                        0x9292_9292_9292_9292u64 >> (64 - bytes.min(8) * 8);
                    if pair {
                        let old = if size == 4 {
                            0x9292_9292_9292_9292
                        } else {
                            0x9292_9292
                        };
                        expected.general_register_storage_mut()[2] = old;
                        expected.general_register_storage_mut()[3] = old;
                    }
                    expected.set_pc(PC + 4);
                    let mut observed = [0; 16];
                    memory
                        .read_bytes(SPACE, GuestVirtualAddress::new(0x3000), &mut observed)
                        .unwrap();
                    let mut wanted = [0x92; 16];
                    if case == 1 {
                        let replacement: u128 = if size == 4 {
                            0xdddd_dddd_0123_4567_eeee_eeee_89ab_cdef
                        } else {
                            0x0123_4567_89ab_cdef
                        };
                        wanted[..bytes].copy_from_slice(&replacement.to_le_bytes()[..bytes]);
                    }
                    assert_eq!(observed, wanted);
                } else {
                    let Err(crate::lcq::fault::cold::Error::Data(fault)) = result else {
                        panic!("{result:?}");
                    };
                    assert_eq!(
                        fault.reason,
                        match case {
                            0 => DataAccessFaultReason::AtomicRegionUnsupported,
                            3 => DataAccessFaultReason::WritePermissionDenied,
                            _ => DataAccessFaultReason::ReadPermissionDenied,
                        }
                    );
                    assert_eq!(
                        fault.kind,
                        if case == 4 {
                            DataAccessKind::Read
                        } else {
                            DataAccessKind::Write
                        }
                    );
                }
                assert_eq!(state, expected);
                assert!(
                    events.lock().unwrap().is_empty(),
                    "CAS never becomes separate MMIO read/write calls"
                );
            }
        }
    }
}

#[test]
fn delivered_lcq_contiguous_grouped_cold_uses_element_descriptors_and_partial_commits() {
    for (opcode, count) in [(7, 1), (10, 2), (6, 3), (2, 4)] {
        for full in [false, true] {
            for size in 0..4 {
                if !full && size == 3 {
                    continue;
                }
                for load in [false, true] {
                    let word = 0x0c9f_003f
                        | (u32::from(full) << 30)
                        | (u32::from(load) << 22)
                        | (opcode << 12)
                        | (size << 10);
                    let elements = count * ((if full { 16 } else { 8 }) >> size);
                    check_at(word, None, false);
                    for failure in [1, elements / 2, elements] {
                        check_at(word, Some(failure), false);
                    }
                }
            }
        }
    }
}

#[test]
fn delivered_lcq_contiguous_single_d_cold_completion() {
    for (opcode, count) in [(7, 1), (10, 2), (6, 3), (2, 4)] {
        for load in [false, true] {
            for (post, rm, rn, rt) in [
                (false, 0, 1, 31),
                (true, 31, 31, 30),
                (true, 2, 1, 0),
                (true, 1, 1, 31),
            ] {
                let word = 0x0c00_0c00
                    | (u32::from(post) << 23)
                    | (u32::from(load) << 22)
                    | (rm << 16)
                    | (opcode << 12)
                    | (rn << 5)
                    | rt;
                check_at(word, None, false);
                for failure in 1..=count {
                    check_at(word, Some(failure), false);
                }
            }
        }
    }
}

#[test]
fn delivered_lcq_structure_cold_completion_preserves_native_and_cold_prefixes() {
    for size in 0..4 {
        for load in [false, true] {
            let address = (1 << 23) | (31 << 16) | (1 << 5) | 31;
            let mut cases = vec![
                (
                    single_structure_word(4, size, Some((16 >> size) - 1), load, false) | address,
                    3,
                ),
                (
                    0x4c00_0000 | (u32::from(load) << 22) | (size << 10) | address,
                    5,
                ),
            ];
            if load {
                cases.push((
                    single_structure_word(4, size, None, true, false) | address,
                    2,
                ));
            }
            if size < 3 {
                cases.push((
                    0x0c00_0000 | (u32::from(load) << 22) | (size << 10) | address,
                    1,
                ));
            }
            if size == 0 {
                cases.push((0x4c00_0000 | (u32::from(load) << 22) | address, 63));
            }
            if size == 3 {
                cases.push((0x0c00_2c00 | (u32::from(load) << 22) | address, 3));
            }
            // Contiguous lists cross into MMIO after a partial first vector,
            // or after a full vector and a partial second one.
            for full in [false, true] {
                if !full && size == 3 {
                    continue;
                }
                let word = 0x0c00_2000
                    | (u32::from(full) << 30)
                    | (u32::from(load) << 22)
                    | (size << 10)
                    | address;
                cases.push((word, 1));
                cases.push((word, ((if full { 16 } else { 8 }) >> size) + 1));
            }
            for (word, prefix) in cases {
                for fail_at in [None, Some(1), Some(2)] {
                    let make_memory = |events| {
                        let mut memory = setup(
                            &[word, 0xd420_0000],
                            Device {
                                events,
                                fail_at,
                                wrong_size: false,
                            },
                        );
                        let ram = GuestPhysicalPageId::new(3);
                        assert!(memory.add_ram_page(ram));
                        memory.initialize_ram(ram, 0, &[0x92; 4096]).unwrap();
                        assert!(memory.map_page(
                            SPACE,
                            GuestVirtualAddress::new(0x2000),
                            ram,
                            MemoryPermissions::READ_WRITE,
                        ));
                        memory
                    };
                    let events = Arc::new(Mutex::new(Vec::new()));
                    let expected_events = Arc::new(Mutex::new(Vec::new()));
                    let memory = make_memory(events.clone());
                    let oracle = make_memory(expected_events.clone());
                    let prefix_bytes = prefix << size;
                    let base = 0x3000 - prefix_bytes as u64;
                    let mut state = A64State::default();
                    state.set_pc(PC);
                    state.general_register_storage_mut()[1] = base;
                    for register in 0..32 {
                        state.set_vector(register, u128::MAX - u128::from(register));
                    }
                    let mut expected = state.clone();
                    let monitor = RefCell::new(ExclusiveMonitorState::default());
                    let event_state = VcpuEventState::default();
                    let result = execute_one_with_context(
                        InterpreterContext::new(
                            ProcessCpuContext::new(TargetPlatform::Switch1, SPACE),
                            &oracle,
                            &monitor,
                            &Timer,
                            &event_state,
                        ),
                        &mut expected,
                        word,
                    )
                    .unwrap();
                    let completion = authority::prepare_cold(&memory, &mut state);
                    assert!(events.lock().unwrap().is_empty());
                    assert_eq!(state.pc(), PC);
                    assert_eq!(state.general_register_storage_mut()[1], base);
                    // Native stores must already be visible before we overwrite
                    // the prefix; native loads must retain its original bits.
                    let mut observed = vec![0; prefix_bytes];
                    let mut expected_bytes = vec![0; prefix_bytes];
                    memory
                        .read_bytes(SPACE, GuestVirtualAddress::new(base), &mut observed)
                        .unwrap();
                    oracle
                        .read_bytes(SPACE, GuestVirtualAddress::new(base), &mut expected_bytes)
                        .unwrap();
                    assert_eq!(observed, expected_bytes);
                    memory
                        .write_bytes(
                            SPACE,
                            GuestVirtualAddress::new(base),
                            &vec![0x77; prefix_bytes],
                        )
                        .unwrap();
                    let actual = completion.complete(
                        &mut state,
                        &memory,
                        &mut ExclusiveMonitorState::default(),
                    );
                    match result {
                        InstructionStep::Continue => actual.unwrap(),
                        InstructionStep::Exit(nixe_cpu::execution::CpuExit::DataFault {
                            fault,
                            ..
                        }) => {
                            let Err(crate::lcq::fault::cold::Error::Data(actual)) = actual else {
                                panic!("{actual:?}");
                            };
                            assert_eq!(actual, fault);
                        }
                        other => panic!("unexpected interpreter exit {other:?}"),
                    }
                    assert_eq!(
                        state, expected,
                        "word={word:08x} prefix={prefix} fail_at={fail_at:?}"
                    );
                    assert_eq!(*events.lock().unwrap(), *expected_events.lock().unwrap());
                    memory
                        .read_bytes(SPACE, GuestVirtualAddress::new(base), &mut observed)
                        .unwrap();
                    assert_eq!(
                        observed,
                        vec![0x77; prefix_bytes],
                        "native prefix must not be replayed"
                    );
                }
            }
        }
    }
}

#[test]
fn cold_completion_revalidates_mapping_after_native_owners_are_gone() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let memory = setup(
        &[0xf940_0020, 0xd420_0000],
        Device {
            events: events.clone(),
            fail_at: None,
            wrong_size: false,
        },
    );
    let mut state = A64State::default();
    state.set_pc(PC);
    state.general_register_storage_mut()[1] = 0x3000;
    let completion = authority::prepare_cold(&memory, &mut state);
    let expected = state.clone();
    memory
        .set_permissions(
            SPACE,
            GuestVirtualAddress::new(0x3000),
            4096,
            MemoryPermissions::NONE,
        )
        .unwrap();
    let Err(crate::lcq::fault::cold::Error::Data(fault)) =
        completion.complete(&mut state, &memory, &mut ExclusiveMonitorState::default())
    else {
        panic!()
    };
    assert_eq!(
        fault.reason,
        nixe_cpu::memory::DataAccessFaultReason::ReadPermissionDenied
    );
    assert_eq!(state, expected);
    assert!(events.lock().unwrap().is_empty());
}
