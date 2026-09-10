use super::*;
use crate::lcq::invocation::{self, Exit, MemoryExit};
use nixe_cpu::execution::{CpuExit, CpuFaultKind};
use nixe_cpu::memory::{DataAccessFaultReason, ExecutionMemory, MemoryValue};
use nixe_cpu_direct_memory::WorkerFaultContext;
use nixe_memory::DirectBackendPolicy;
use std::sync::atomic::{AtomicUsize, Ordering};

fn fixture(words: &[u32], device: Option<Arc<AtomicUsize>>) -> ExecutionMemory {
    let mut memory = ExecutionMemory::new();
    for page in 1..=2 {
        let id = GuestPhysicalPageId::new(page);
        assert!(memory.add_ram_page(id));
        if page == 1 {
            let bytes: Vec<_> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
            memory.initialize_ram(id, 0, &bytes).unwrap();
        }
        assert!(memory.map_page(
            SPACE,
            GuestVirtualAddress::new(page * 4096),
            id,
            if page == 1 {
                MemoryPermissions::READ_EXECUTE
            } else {
                MemoryPermissions::READ_WRITE
            }
        ));
    }
    if let Some(device) = device {
        let id = GuestPhysicalPageId::new(3);
        assert!(memory.add_mmio_page(id, authority::Device(device)));
        assert!(memory.map_page(
            SPACE,
            GuestVirtualAddress::new(0x3000),
            id,
            MemoryPermissions::READ_WRITE
        ));
    }
    memory
        .bind_cpu_memory_backend(SPACE, ARENA as u64, DirectBackendPolicy::Required)
        .unwrap();
    memory
}

#[test]
fn lcq_invocation_owns_normal_exit_identity_without_memory_fault_sites() {
    for (word, kind, reason, target) in [
        (
            0xd420_00e0,
            EdgeKind::Breakpoint(7),
            NativeExitReason::Architectural,
            PC + 4,
        ),
        (
            0xd400_0121,
            EdgeKind::SupervisorCall(9),
            NativeExitReason::Architectural,
            PC + 4,
        ),
        (
            0x1400_0008,
            EdgeKind::Static,
            NativeExitReason::Dispatch,
            PC + 36,
        ),
        (
            0,
            EdgeKind::InvalidInstruction,
            NativeExitReason::Unsupported,
            PC + 4,
        ),
    ] {
        let memory = fixture(&[0x9100_0400, word], None); // ADD X0, X0, #1
        let cache = Cache::new().unwrap();
        let process = Arc::new(Lifetime::new(cache.clone()).unwrap());
        let mut reader = process.register().unwrap();
        let Request::Owner(claim) = reader.claim(key()).unwrap() else {
            panic!()
        };
        let handle = Compiler::for_arena(native_abi(), ARENA)
            .unwrap()
            .publish(
                Compilation::capture(claim, &memory).unwrap(),
                &process,
                &cache,
                &memory,
            )
            .unwrap();
        assert!(process.snapshot(handle).unwrap().faults.is_empty());
        let mut worker = WorkerFaultContext::register().unwrap();
        let mut monitor = ExclusiveMonitorState::default();
        let mut state = A64State::default();
        state.set_pc(PC);
        let mut frame = NativeFrame::new(&mut state, PollBudget::new(4096, 1000).unwrap());
        let result = unsafe {
            invocation::run(
                &mut reader,
                &mut frame,
                &memory,
                &mut worker,
                &mut monitor,
                key(),
            )
        }
        .unwrap()
        .unwrap();
        assert_eq!(frame.execution_epoch, 0);
        drop(reader);
        drop(process);
        drop(cache);
        // Overwrite the exiting instruction after all native owners are gone.
        memory
            .set_permissions(
                SPACE,
                GuestVirtualAddress::new(PC),
                4096,
                MemoryPermissions::READ_WRITE,
            )
            .unwrap();
        memory
            .write_bytes(
                SPACE,
                GuestVirtualAddress::new(PC + 4),
                &0xd503_201fu32.to_le_bytes(),
            )
            .unwrap();
        let Exit::Native {
            returned,
            guest,
            instruction,
        } = result
        else {
            panic!("unexpected memory exit")
        };
        assert_eq!(returned.reason, reason);
        assert_eq!(guest.kind, kind);
        assert_eq!(guest.pc.get(), PC + 4);
        assert_eq!(instruction.key.block_key().pc.get(), PC + 4);
        assert_eq!(instruction.bits, word);
        assert_eq!(state.pc(), target);
        assert_eq!(state.general_register_storage_mut()[0], 1);
    }
}

#[test]
fn lcq_invocation_owns_admission_retry_and_exclusive_handoff() {
    // A tracked STR retries, then LDXR publishes the value observed natively.
    let memory = fixture(&[0xf900_0020, 0xc85f_7c22, 0xd420_0000], None);
    let cache = Cache::new().unwrap();
    let process = Arc::new(Lifetime::new(cache.clone()).unwrap());
    let mut reader = process.register().unwrap();
    let mut worker = WorkerFaultContext::register().unwrap();
    let mut monitor = ExclusiveMonitorState::default();
    let mut state = A64State::default();
    state.set_pc(PC);
    state.general_register_storage_mut()[0] = 0x1234_5678;
    state.general_register_storage_mut()[1] = DATA as u64;
    let mut frame = NativeFrame::new(&mut state, PollBudget::new(4096, 1000).unwrap());
    assert!(
        unsafe {
            invocation::run(
                &mut reader,
                &mut frame,
                &memory,
                &mut worker,
                &mut monitor,
                key(),
            )
        }
        .unwrap()
        .is_none()
    );
    assert_eq!(frame.execution_epoch, 0);
    assert_eq!(frame.host_fp.saved, 0);
    let Request::Owner(claim) = reader.claim(key()).unwrap() else {
        panic!()
    };
    Compiler::for_arena(native_abi(), ARENA)
        .unwrap()
        .publish(
            Compilation::capture(claim, &memory).unwrap(),
            &process,
            &cache,
            &memory,
        )
        .unwrap();
    assert_eq!(
        memory.direct_protection_at(SPACE, GuestVirtualAddress::new(DATA as u64)),
        Some(DirectProtection::Read)
    );
    let Some(Exit::Native {
        returned,
        guest,
        instruction,
    }) = (unsafe {
        invocation::run(
            &mut reader,
            &mut frame,
            &memory,
            &mut worker,
            &mut monitor,
            key(),
        )
    })
    .unwrap()
    else {
        panic!("tracked store did not retry to the canonical exit")
    };
    assert_eq!(returned.reason, NativeExitReason::Architectural);
    assert_eq!(guest.pc.get(), PC + 8);
    assert_eq!(guest.kind, EdgeKind::Breakpoint(0));
    assert_eq!(instruction.bits, 0xd420_0000);
    assert_eq!(frame.execution_epoch, 0);
    assert_eq!(frame.admission_epoch, 0);
    assert_eq!(frame.host_fp.saved, 0);
    assert_eq!(frame.exclusive_load.bytes, 0);
    assert_eq!(
        monitor.reservation().unwrap().expected,
        MemoryValue::U64(0x1234_5678)
    );
    assert_eq!(state.pc(), PC + 8);
    assert_eq!(state.general_register_storage_mut()[2], 0x1234_5678);
    assert_eq!(
        memory.direct_protection_at(SPACE, GuestVirtualAddress::new(DATA as u64)),
        Some(DirectProtection::ReadWrite)
    );
}

#[test]
fn lcq_invocation_escapes_with_owned_cold_or_precise_fault_and_releases_owners() {
    for (device, revoke) in [(false, false), (true, false), (true, true)] {
        let calls = Arc::new(AtomicUsize::new(0));
        // LDXR succeeds before a dirty address calculation and faulting LDR.
        let words = [0xc85f_7c22, 0x9100_0463, 0xf940_0060, 0xd420_0000];
        let memory = fixture(&words, device.then(|| calls.clone()));
        let cache = Cache::new().unwrap();
        let process = Arc::new(Lifetime::new(cache.clone()).unwrap());
        let mut reader = process.register().unwrap();
        let Request::Owner(claim) = reader.claim(key()).unwrap() else {
            panic!()
        };
        Compiler::for_arena(native_abi(), ARENA)
            .unwrap()
            .publish(
                Compilation::capture(claim, &memory).unwrap(),
                &process,
                &cache,
                &memory,
            )
            .unwrap();
        let mut worker = WorkerFaultContext::register().unwrap();
        let mut monitor = ExclusiveMonitorState::default();
        let mut state = A64State::default();
        state.set_pc(PC);
        state.general_register_storage_mut()[0] = 99;
        state.general_register_storage_mut()[1] = DATA as u64;
        state.general_register_storage_mut()[3] = 0x2fff;
        let mut frame = NativeFrame::new(&mut state, PollBudget::new(4096, 1000).unwrap());
        let Some(Exit::Memory {
            instruction,
            poll,
            outcome,
        }) = (unsafe {
            invocation::run(
                &mut reader,
                &mut frame,
                &memory,
                &mut worker,
                &mut monitor,
                key(),
            )
        })
        .unwrap()
        else {
            panic!("non-RAM access did not escape")
        };
        assert_eq!(instruction.bits, words[2]);
        assert_eq!(
            instruction.key.block_key(),
            key().at(GuestVirtualAddress::new(PC + 8)).unwrap()
        );
        assert!(!poll.exhausted);
        assert!(!poll.sample);
        assert_eq!(frame.execution_epoch, 0);
        assert_eq!(frame.admission_epoch, 0);
        assert_eq!(frame.host_fp.saved, 0);
        assert_eq!(frame.exclusive_load.bytes, 0);
        assert_eq!(
            monitor.reservation().unwrap().page,
            GuestPhysicalPageId::new(2)
        );
        assert_eq!(state.pc(), PC + 8);
        assert_eq!(state.general_register_storage_mut()[0], 99);
        assert_eq!(state.general_register_storage_mut()[3], 0x3000);
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        drop(worker);
        drop(reader);
        drop(process);
        drop(cache);
        // A mapping mutation on this same thread would deadlock if run retained
        // its lease. Revoke guest execution before consuming the owned result.
        memory
            .set_permissions(
                SPACE,
                GuestVirtualAddress::new(PC),
                4096,
                MemoryPermissions::READ,
            )
            .unwrap();
        assert_eq!(matches!(outcome, MemoryExit::Cold(_)), device);
        if revoke {
            memory
                .set_permissions(
                    SPACE,
                    GuestVirtualAddress::new(0x3000),
                    4096,
                    MemoryPermissions::NONE,
                )
                .unwrap();
        }
        match outcome
            .complete(instruction, &mut state, &memory, &mut monitor, 7)
            .unwrap()
        {
            None if device && !revoke => {
                assert_eq!(state.pc(), PC + 12);
                assert_eq!(state.general_register_storage_mut()[0], 19);
                assert_eq!(calls.load(Ordering::Relaxed), 1);
                assert!(monitor.reservation().is_some());
            }
            Some(CpuExit::DataFault { source, fault }) if !device || revoke => {
                assert_eq!(
                    source,
                    nixe_cpu::location::LocationDescriptor::new(
                        GuestVirtualAddress::new(PC + 8),
                        key().profile
                    )
                );
                assert_eq!(fault.address.get(), 0x3000);
                assert_eq!(
                    fault.reason,
                    if revoke {
                        DataAccessFaultReason::ReadPermissionDenied
                    } else {
                        DataAccessFaultReason::Unmapped
                    }
                );
                assert_eq!(state.pc(), PC + 8);
                assert_eq!(state.general_register_storage_mut()[0], 99);
                assert_eq!(calls.load(Ordering::Relaxed), 0);
            }
            _ => panic!("incorrect memory exit classification"),
        }
    }
}

#[test]
fn lcq_invocation_pair_cold_exit_retains_the_first_native_read() {
    let calls = Arc::new(AtomicUsize::new(0));
    // LDP X0, X2, [X1], #16: first read is RAM, second read is MMIO.
    let memory = fixture(&[0xa8c1_0820, 0xd420_0000], Some(calls.clone()));
    let cache = Cache::new().unwrap();
    let process = Arc::new(Lifetime::new(cache.clone()).unwrap());
    let mut reader = process.register().unwrap();
    let Request::Owner(claim) = reader.claim(key()).unwrap() else {
        panic!()
    };
    Compiler::for_arena(native_abi(), ARENA)
        .unwrap()
        .publish(
            Compilation::capture(claim, &memory).unwrap(),
            &process,
            &cache,
            &memory,
        )
        .unwrap();
    let mut worker = WorkerFaultContext::register().unwrap();
    let mut monitor = ExclusiveMonitorState::default();
    let mut state = A64State::default();
    state.set_pc(PC);
    state.general_register_storage_mut()[0] = 99;
    state.general_register_storage_mut()[1] = 0x2ff8;
    state.general_register_storage_mut()[2] = 98;
    let mut frame = NativeFrame::new(&mut state, PollBudget::new(4096, 1000).unwrap());
    let Some(Exit::Memory {
        instruction,
        outcome,
        ..
    }) = (unsafe {
        invocation::run(
            &mut reader,
            &mut frame,
            &memory,
            &mut worker,
            &mut monitor,
            key(),
        )
    })
    .unwrap()
    else {
        panic!("pair did not produce an owned completion")
    };
    assert_eq!(state.pc(), PC);
    assert_eq!(state.general_register_storage_mut()[0], 99);
    assert_eq!(state.general_register_storage_mut()[1], 0x2ff8);
    assert_eq!(state.general_register_storage_mut()[2], 98);
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    drop(worker);
    drop(reader);
    drop(process);
    drop(cache);
    use nixe_cpu::memory::{CpuMemory, MemoryAccess, MemoryAccessSize};
    memory
        .write(
            SPACE,
            GuestVirtualAddress::new(0x2ff8),
            MemoryAccess::normal(MemoryAccessSize::Doubleword),
            MemoryValue::U64(71),
        )
        .unwrap();
    assert!(
        outcome
            .complete(instruction, &mut state, &memory, &mut monitor, 7)
            .unwrap()
            .is_none()
    );
    assert_eq!(state.pc(), PC + 4);
    assert_eq!(
        state.general_register_storage_mut()[0],
        0,
        "must not reread changed RAM"
    );
    assert_eq!(state.general_register_storage_mut()[1], 0x3008);
    assert_eq!(state.general_register_storage_mut()[2], 19);
    assert_eq!(calls.load(Ordering::Relaxed), 1);
}

#[test]
fn lcq_invocation_completes_exclusive_store_from_a_previous_invocation() {
    use nixe_cpu::memory::{CpuMemory, MemoryAccess, MemoryAccessSize};
    // LDXR X2, [X1]; B next; STXR W3, X0, [X1]; BRK.
    let words = [0xc85f_7c22, 0x1400_0001, 0xc803_7c20, 0xd420_0000];
    for case in 0..4 {
        let memory = fixture(&words, None);
        let cache = Cache::new().unwrap();
        let process = Arc::new(Lifetime::new(cache.clone()).unwrap());
        let mut reader = process.register().unwrap();
        let mut worker = WorkerFaultContext::register().unwrap();
        let mut monitor = ExclusiveMonitorState::default();
        let mut state = A64State::default();
        state.set_pc(PC);
        state.general_register_storage_mut()[0] = 19;
        state.general_register_storage_mut()[1] = DATA as u64;
        state.general_register_storage_mut()[3] = 99;
        let mut pending = None;
        for pc in [PC, PC + 8] {
            let demanded = key().at(GuestVirtualAddress::new(pc)).unwrap();
            let Request::Owner(claim) = reader.claim(demanded).unwrap() else {
                panic!()
            };
            Compiler::for_arena(native_abi(), ARENA)
                .unwrap()
                .publish(
                    Compilation::capture(claim, &memory).unwrap(),
                    &process,
                    &cache,
                    &memory,
                )
                .unwrap();
            let mut frame = NativeFrame::new(&mut state, PollBudget::new(4096, 1000).unwrap());
            let exit = unsafe {
                invocation::run(
                    &mut reader,
                    &mut frame,
                    &memory,
                    &mut worker,
                    &mut monitor,
                    demanded,
                )
            }
            .unwrap()
            .unwrap();
            assert_eq!(frame.execution_epoch, 0);
            assert_eq!(frame.exclusive_load.bytes, 0);
            if pc == PC {
                assert!(matches!(exit, Exit::Native { .. }));
                assert_eq!(state.pc(), PC + 8);
                assert_eq!(monitor.reservation().unwrap().expected, MemoryValue::U64(0));
                if case == 3 {
                    monitor.clear();
                    state.general_register_storage_mut()[1] = u64::MAX;
                }
            } else {
                let Exit::Memory {
                    instruction,
                    outcome,
                    ..
                } = exit
                else {
                    panic!()
                };
                assert!(matches!(outcome, MemoryExit::ExclusiveStore(_)));
                assert_eq!(instruction.bits, words[2]);
                assert_eq!(state.general_register_storage_mut()[3], 99);
                assert_eq!(state.pc(), PC + 8);
                pending = Some((instruction, outcome));
            }
        }
        drop(worker);
        drop(reader);
        drop(process);
        drop(cache);
        match case {
            1 => memory
                .write_bytes(
                    SPACE,
                    GuestVirtualAddress::new(DATA as u64),
                    &71u64.to_le_bytes(),
                )
                .unwrap(),
            2 => memory
                .set_permissions(
                    SPACE,
                    GuestVirtualAddress::new(DATA as u64),
                    4096,
                    MemoryPermissions::READ,
                )
                .unwrap(),
            _ => {}
        }
        let (instruction, outcome) = pending.unwrap();
        let stop = outcome
            .complete(instruction, &mut state, &memory, &mut monitor, 1)
            .unwrap();
        assert!(monitor.reservation().is_none());
        if case == 2 {
            let Some(CpuExit::DataFault { source, fault }) = stop else {
                panic!()
            };
            assert_eq!(
                source,
                nixe_cpu::location::LocationDescriptor::new(
                    GuestVirtualAddress::new(PC + 8),
                    key().profile
                )
            );
            assert_eq!(fault.reason, DataAccessFaultReason::WritePermissionDenied);
            assert_eq!(state.general_register_storage_mut()[3], 99);
            assert_eq!(state.pc(), PC + 8);
        } else {
            assert!(stop.is_none());
            assert_eq!(
                state.general_register_storage_mut()[3],
                u64::from(case != 0)
            );
            assert_eq!(state.pc(), PC + 12);
        }
        assert_eq!(
            memory
                .read(
                    SPACE,
                    GuestVirtualAddress::new(DATA as u64),
                    MemoryAccess::normal(MemoryAccessSize::Doubleword)
                )
                .unwrap()
                .value,
            MemoryValue::U64(match case {
                0 => 19,
                1 => 71,
                _ => 0,
            })
        );
    }
}

#[test]
fn lcq_invocation_reports_internal_memory_errors_with_captured_identity_and_state() {
    use crate::abi::InstructionKey;
    use crate::lifetime::unit::Instruction;
    for fatal in [false, true] {
        let calls = Arc::new(AtomicUsize::new(0));
        let word = 0xf940_0020;
        let memory = fixture(&[word, 0xd420_0000], Some(calls.clone()));
        let instruction = Instruction {
            key: InstructionKey::new(key()).unwrap(),
            bits: word,
        };
        let mut state = A64State::default();
        state.set_pc(PC);
        state.general_register_storage_mut()[1] = 0x3000;
        let outcome = if fatal {
            MemoryExit::Fatal("published RAM mapping contradicts memory policy".into())
        } else {
            let completion = authority::prepare_cold(&memory, &mut state);
            state.set_pc(PC + 4); // An invalid continuation must not touch MMIO.
            MemoryExit::Cold(Box::new(completion))
        };
        memory
            .set_permissions(
                SPACE,
                GuestVirtualAddress::new(PC),
                4096,
                MemoryPermissions::READ_WRITE,
            )
            .unwrap();
        memory
            .write_bytes(
                SPACE,
                GuestVirtualAddress::new(PC),
                &0xd503_201fu32.to_le_bytes(),
            )
            .unwrap();
        let context = state.register_context();
        let error = outcome
            .complete(
                instruction,
                &mut state,
                &memory,
                &mut ExclusiveMonitorState::default(),
                7,
            )
            .unwrap_err();
        assert_eq!(error.kind, CpuFaultKind::Internal);
        assert_eq!(error.backend, "jit");
        assert_eq!(error.progress, 7);
        assert_eq!(*error.context, context);
        assert!(error.message.contains("0xf9400020"));
        assert!(
            error
                .message
                .contains(&GuestVirtualAddress::new(PC).to_string())
        );
        assert!(error.message.contains(if fatal {
            "contradicts memory policy"
        } else {
            "PC no longer matches"
        }));
        assert_eq!(state.register_context(), context);
        assert_eq!(calls.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn lcq_invocation_rejected_admission_releases_fp_epoch_and_mapping_lease() {
    use crate::lifetime::{Error as LifetimeError, Reason};
    let memory = fixture(&[0xd420_0000], None);
    let cache = Cache::new().unwrap();
    let process = Arc::new(Lifetime::new(cache).unwrap());
    let mut reader = process.register().unwrap();
    let mut worker = WorkerFaultContext::register().unwrap();
    let mut monitor = ExclusiveMonitorState::default();
    let mut state = A64State::default();
    state.set_pc(PC);
    let mut frame = NativeFrame::new(&mut state, PollBudget::new(4096, 1000).unwrap());
    process.request(Reason::MappingChange).unwrap();
    let mut transition = process.try_transition().unwrap().unwrap();
    for closed in [false, true] {
        if closed {
            // This same-thread wait would deadlock if rejected admission left
            // its epoch announced. No native entry has been published here.
            transition.wait_closed().unwrap();
        }
        let result = unsafe {
            invocation::run(
                &mut reader,
                &mut frame,
                &memory,
                &mut worker,
                &mut monitor,
                key(),
            )
        };
        assert!(matches!(
            result,
            Err(invocation::Error::Lifetime(LifetimeError::Closed))
        ));
        assert_eq!(frame.execution_epoch, 0);
        assert_eq!(frame.admission_epoch, 0);
        assert_eq!(frame.host_fp.saved, 0);
        // Likewise, a leaked shared memory lease would block this mutation.
        memory
            .set_permissions(
                SPACE,
                GuestVirtualAddress::new(DATA as u64),
                4096,
                if closed {
                    MemoryPermissions::READ_WRITE
                } else {
                    MemoryPermissions::READ
                },
            )
            .unwrap();
    }
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
    assert!(
        unsafe {
            invocation::run(
                &mut reader,
                &mut frame,
                &memory,
                &mut worker,
                &mut monitor,
                key(),
            )
        }
        .unwrap()
        .is_none()
    );
    assert_eq!(frame.host_fp.saved, 0);
    assert_eq!(state.pc(), PC);
}
