//! Delivered later-unit faults and exclusive-load handoff through the production
//! invocation owner and registered Closed linker, including real bridge/root
//! teardown before consuming an owned cold result.
use super::*;
use crate::lcq::compiler::tests::chaining::{install, publish_staged};
use nixe_cpu::memory::{CpuMemory, MemoryAccess, MemoryAccessSize};

#[test]
fn retirement_waits_for_a_later_unit_fault_retry_or_escape_before_unlink_and_reclaim() {
    use crate::lcq::compiler::tests::chaining::pause_fast_entry;
    use std::sync::{atomic::AtomicU32, mpsc};
    crate::native::check_host().unwrap();
    let words = [
        0x9100_0400,
        0x1400_0001, // A: ADD X0,X0,#1; B B
        0x9100_0800,
        0xf900_0020,
        0xd420_0000,
    ]; // B: ADD X0,X0,#2; STR X0,[X1]; BRK
    for escape in [false, true] {
        let memory = fixture(&words, None);
        let cache = Cache::new().unwrap();
        let process = Arc::new(Lifetime::new(cache.clone()).unwrap());
        let mut reader = process.register().unwrap();
        let mut compiler = Compiler::for_arena(native_abi(), ARENA).unwrap();
        let target_key = key().at(GuestVirtualAddress::new(PC + 8)).unwrap();
        let Request::Owner(claim) = reader.claim(target_key).unwrap() else {
            panic!()
        };
        let compilation = Compilation::capture(claim, &memory).unwrap();
        let mut lowered = compiler
            .lower(&compilation.fragment, compilation.identity.version())
            .unwrap();
        pause_fast_entry(&mut lowered);
        let target = publish_staged(compilation, lowered, &process, &cache, &memory);
        let Request::Owner(claim) = reader.claim(key()).unwrap() else {
            panic!()
        };
        let compilation = Compilation::capture(claim, &memory).unwrap();
        let lowered = compiler
            .lower(&compilation.fragment, compilation.identity.version())
            .unwrap();
        let source = publish_staged(compilation, lowered, &process, &cache, &memory);
        let link = install(&process, source, target);
        // No compiler snapshot artificially keeps B alive during the fault.
        let rendezvous = [AtomicU32::new(0), AtomicU32::new(0)];
        let mut state = A64State::default();
        state.set_pc(PC);
        state.set_nzcv(Nzcv::from_bits(0xa000_0000));
        state.general_register_storage_mut()[0] = 41;
        state.general_register_storage_mut()[1] = if escape { 0x3000 } else { DATA as u64 };
        let mut worker = WorkerFaultContext::register().unwrap();
        let mut monitor = ExclusiveMonitorState::default();
        let mut frame = NativeFrame::new(&mut state, PollBudget::new(4096, 1000).unwrap());
        frame.runtime = rendezvous.as_ptr().cast_mut().cast();
        let (closed_tx, closed_rx) = mpsc::channel();
        let exit = std::thread::scope(|scope| {
            scope.spawn(|| {
                while rendezvous[0].load(Ordering::Acquire) == 0 {
                    std::thread::yield_now();
                }
                let ticket = process.retire_unit(target);
                // Even failure releases the native test waiter.
                if ticket.is_err() {
                    rendezvous[1].store(1, Ordering::Release);
                }
                let ticket = ticket.unwrap();
                let mut transition = process.try_transition().unwrap().unwrap();
                let premature = transition.unlink_link(link);
                let reclaimed = process.reclaim_units();
                rendezvous[1].store(1, Ordering::Release);
                assert_eq!(premature, Err(crate::lifetime::Error::Closed));
                assert_eq!(reclaimed.unwrap(), 0);
                transition.wait_closed().unwrap();
                assert!(transition.drain_links().unwrap());
                assert_eq!(
                    transition.unlink_link(link),
                    Err(crate::lifetime::Error::StaleUnit)
                );
                assert_eq!(process.reclaim_units().unwrap(), 1);
                transition.batch().unwrap().complete().unwrap();
                assert!(transition.try_reopen().unwrap());
                assert!(ticket.is_complete().unwrap());
                closed_tx.send(()).unwrap();
            });
            let exit = unsafe {
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
            closed_rx.recv().unwrap();
            exit
        });
        // Reconstruction/retry finished before B's actual directory/bytes were
        // reclaimed. The escaped result owns its instruction, not a map borrow.
        assert_eq!((frame.execution_epoch, frame.admission_epoch), (0, 0));
        assert_eq!(frame.budget.slice_remaining, if escape { 997 } else { 996 });
        assert_eq!(state.general_register_storage_mut()[0], 44);
        assert_eq!(state.nzcv().bits(), 0xa000_0000);
        match exit {
            Exit::Memory {
                instruction,
                outcome: MemoryExit::Fault(fault),
                ..
            } if escape => {
                assert_eq!(instruction.bits, words[3]);
                assert_eq!(instruction.key.block_key().pc.get(), PC + 12);
                assert_eq!(fault.reason, DataAccessFaultReason::Unmapped);
                assert_eq!(state.pc(), PC + 12);
            }
            Exit::Native {
                returned,
                instruction,
                ..
            } if !escape => {
                assert_eq!(returned.reason, NativeExitReason::Architectural);
                assert_eq!(instruction.bits, words[4]);
                assert_eq!(
                    memory
                        .read(
                            SPACE,
                            GuestVirtualAddress::new(DATA as u64),
                            MemoryAccess::normal(MemoryAccessSize::Doubleword)
                        )
                        .unwrap()
                        .value,
                    MemoryValue::U64(44)
                );
            }
            _ => panic!("wrong retired-target fault outcome, escape={escape}"),
        }
        // With B gone, A must execute its restored fallback instead of a stale
        // jump. Its state and completed-instruction charge remain exact.
        state.set_pc(PC);
        let mut frame = NativeFrame::new(&mut state, PollBudget::new(4096, 1000).unwrap());
        let exit = unsafe {
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
        assert!(
            matches!(exit, Exit::Native { returned, .. } if returned.reason == NativeExitReason::Dispatch)
        );
        assert_eq!(frame.budget.slice_remaining, 998);
        assert_eq!(state.general_register_storage_mut()[0], 45);
        assert_eq!(state.pc(), PC + 8);
        assert!(process.try_shutdown().unwrap());
        memory
            .set_permissions(
                SPACE,
                GuestVirtualAddress::new(PC),
                4096,
                MemoryPermissions::READ,
            )
            .unwrap();
    }
}

#[test]
fn selective_chain_faults_preserve_spills_lazy_flags_and_inherited_fp() {
    use crate::abi::ValueLocation;
    crate::native::check_host().unwrap();
    let mut words = Vec::new();
    for register in 2..31 {
        words.push(0x9100_0400 | (register << 5) | register); // ADD Xn,Xn,#1
    }
    for register in 0..32 {
        words.push(0x4e20_8400 | (register << 16) | (register << 5) | register); // ADD Vn.16B,Vn,Vn
    }
    words.extend([
        0x9e67_0121, // FMOV D1,X9
        0x9e67_0142, // FMOV D2,X10
        0xf100_04a5, // SUBS X5,X5,#1
        0x1e62_2820, // FADD D0,D1,D2: inexact under guest round toward +infinity
        0x1400_0001, // B B
    ]);
    let target_index = words.len();
    words.extend([
        0x9100_0400, // B: ADD X0,X0,#1 (clean input absent from A)
        0xf900_0023, // STR X3,[X1]: tracking retry or precise unmapped escape
    ]);
    // Keep nearly the complete register image live beyond the fault. X19/V19
    // are deliberately absent: the bridge must commit them before B can reuse
    // A's physical locations. B does not itself activate floating-point state.
    for register in (2..31).filter(|register| *register != 19) {
        words.push(0x9100_0400 | (register << 5) | register);
    }
    for register in (0..32).filter(|register| *register != 19) {
        words.push(0x4e20_8400 | (register << 16) | (register << 5) | register);
    }
    words.extend([0x9a1f_00c6, 0xd420_0000]); // ADC X6,X6,XZR; BRK
    for escape in [false, true] {
        let memory = fixture(&words, None);
        let reference = fixture(&words, None);
        let cache = Cache::new().unwrap();
        let process = Arc::new(Lifetime::new(cache.clone()).unwrap());
        let mut reader = process.register().unwrap();
        let mut compiler = Compiler::for_arena(native_abi(), ARENA).unwrap();
        let target_key = key()
            .at(GuestVirtualAddress::new(PC + target_index as u64 * 4))
            .unwrap();
        let Request::Owner(claim) = reader.claim(target_key).unwrap() else {
            panic!()
        };
        let target = compiler
            .publish(
                Compilation::capture(claim, &memory).unwrap(),
                &process,
                &cache,
                &memory,
            )
            .unwrap();
        let second = process.snapshot(target).unwrap();
        let entry = &second.entries[0].contract;
        assert!(!entry.live_in.integer.x[19] && !entry.live_in.vector[19]);
        assert_eq!(entry.live_in.nzcv, crate::analysis::C);
        assert!(
            entry
                .bindings
                .iter()
                .any(|b| matches!(b.location, ValueLocation::Spill { .. }))
        );
        let fault = &second.states[second.faults[0].state_map as usize].state;
        assert!(fault.host_fpsr_pending);
        assert!(
            fault
                .bindings
                .iter()
                .any(|b| matches!(b.location, ValueLocation::Spill { .. }))
        );
        let Request::Owner(claim) = reader.claim(key()).unwrap() else {
            panic!()
        };
        let compilation = Compilation::capture(claim, &memory).unwrap();
        let lowered = compiler
            .lower(&compilation.fragment, compilation.identity.version())
            .unwrap();
        let source = &lowered
            .states
            .iter()
            .find(|s| s.exit.is_some_and(|exit| exit.kind == EdgeKind::Static))
            .unwrap()
            .state;
        assert!(source.dirty_live.integer.x[19] && source.dirty_live.vector[19]);
        assert!(
            source
                .bindings
                .iter()
                .any(|b| matches!(b.location, ValueLocation::Spill { .. }))
        );
        // Inspect the final allocator maps, not an assumed register ordering:
        // after removing ready writes, a remaining dependency proves that this
        // x86 bridge really exercises cycle breaking in the shared arena.
        // AArch64 allocates an acyclic transfer here; its deterministic cycles
        // are covered separately by native::tests::bridge.
        let mut copies: Vec<_> = entry
            .bindings
            .iter()
            .filter_map(|target| {
                source
                    .bindings
                    .iter()
                    .find(|b| b.value == target.value)
                    .filter(|b| b.location != target.location)
                    .map(|b| (b.location, target.location))
            })
            .collect();
        while let Some(ready) = copies.iter().position(|(_, destination)| {
            copies
                .iter()
                .all(|(source, _)| !crate::abi::locations_overlap(*destination, *source))
        }) {
            copies.remove(ready);
        }
        if native_abi() == HostAbi::X86_64 {
            assert!(
                !copies.is_empty(),
                "x86 fixture must require an allocated transfer cycle"
            );
        }
        let handle = publish_staged(compilation, lowered, &process, &cache, &memory);
        let first = process.snapshot(handle).unwrap();
        install(&process, handle, target);
        let mut state = crate::lcq::compiler::tests::integer::initial_state();
        for register in 0..32 {
            state.set_vector(
                register,
                (u128::from(register) + 1) * 0x0102_0304_0506_0708_0102_0304_0506_0708,
            );
        }
        state.set_fpcr(1 << 22);
        state.set_fpsr(1 << 27);
        state.general_register_storage_mut()[1] = if escape { 0x3000 } else { DATA as u64 };
        state.general_register_storage_mut()[9] = 1.0f64.to_bits() - 1;
        state.general_register_storage_mut()[10] = 0x3ca0_0000_0000_0000 - 1;
        let mut expected = state.clone();
        let expected_monitor = RefCell::new(ExclusiveMonitorState::default());
        let events = VcpuEventState::default();
        let context = InterpreterContext::new(
            ProcessCpuContext::new(TargetPlatform::Switch1, SPACE),
            &reference,
            &expected_monitor,
            &Timer,
            &events,
        );
        let completed = if escape {
            target_index + 1
        } else {
            words.len() - 1
        };
        for &word in &words[..completed] {
            assert_eq!(
                execute_one_with_context(context, &mut expected, word).unwrap(),
                InstructionStep::Continue
            );
        }
        assert_ne!(expected.fpsr() & (1 << 4), 0); // Inexact survives into integer B.
        let _restore = crate::fp_env::tests::RestoreHost::new();
        let caller = crate::fp_env::tests::distinct_caller();
        let mut worker = WorkerFaultContext::register().unwrap();
        let mut monitor = ExclusiveMonitorState::default();
        let mut frame = NativeFrame::new(&mut state, PollBudget::new(4096, 1000).unwrap());
        assert_eq!(
            memory.direct_protection_at(SPACE, GuestVirtualAddress::new(DATA as u64)),
            Some(DirectProtection::Read)
        );
        let exit = unsafe {
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
        assert_eq!(frame.budget.slice_remaining, 1000 - completed as i64);
        assert_eq!(frame.budget.sample_remaining, 4096 - completed as i64);
        assert_eq!((frame.execution_epoch, frame.admission_epoch), (0, 0));
        assert_eq!((frame.host_fp.active, frame.host_fp.saved), (0, 0));
        let mut probe = crate::abi::HostFpState::default();
        unsafe {
            probe.begin();
            probe.finish();
        }
        assert_eq!((probe.saved_control, probe.saved_status), caller);
        assert_eq!(state, expected, "escape={escape}");
        match exit {
            Exit::Memory {
                instruction,
                outcome: MemoryExit::Fault(fault),
            } if escape => {
                assert_eq!(instruction.bits, words[target_index + 1]);
                assert_eq!(
                    instruction.key.block_key().pc.get(),
                    PC + (target_index as u64 + 1) * 4
                );
                assert_eq!(fault.reason, DataAccessFaultReason::Unmapped);
                assert_eq!(fault.address.get(), 0x3000);
            }
            Exit::Native {
                returned,
                instruction,
                guest,
            } if !escape => {
                assert_eq!(returned.reason, NativeExitReason::Architectural);
                assert_eq!(instruction.bits, 0xd420_0000);
                assert_eq!(guest.pc.get(), PC + completed as u64 * 4);
                assert!(!returned.poll.sample && !returned.poll.exhausted);
                assert_eq!(
                    memory.direct_protection_at(SPACE, GuestVirtualAddress::new(DATA as u64)),
                    Some(DirectProtection::ReadWrite)
                );
                let address = GuestVirtualAddress::new(DATA as u64);
                let access = MemoryAccess::normal(MemoryAccessSize::Doubleword);
                assert_eq!(
                    memory.read(SPACE, address, access).unwrap().value,
                    reference.read(SPACE, address, access).unwrap().value
                );
            }
            _ => panic!("wrong pressure-chain exit, escape={escape}"),
        }
        drop(first);
        drop(second);
        drop(worker);
        drop(reader);
        assert!(process.try_shutdown().unwrap());
        drop(process);
        drop(cache);
        // A leaked mapping lease would deadlock this same-thread mutation.
        memory
            .set_permissions(
                SPACE,
                GuestVirtualAddress::new(PC),
                4096,
                MemoryPermissions::READ,
            )
            .unwrap();
    }
}

#[test]
fn later_unit_retry_and_escape_preserve_prefix_and_exclusive_load() {
    crate::native::check_host().unwrap();
    // All source writes (X0, X2, X3) are inputs of B. X2 is used only AFTER
    // the fault, so B's PRE map must retain it even before its first local use.
    // Neither unit writes NZCV; no absent dirty flags need bridge writeback.
    let words = [
        0x9100_0400, // A: ADD X0,X0,#1
        0x9100_4063, // ADD X3,X3,#16
        0xc85f_7c22, // LDXR X2,[X1]
        0x1400_0001, // B B
        0x9100_0800, // B: ADD X0,X0,#2 (must never replay on retry)
        0xf900_0060, // STR X0,[X3]
        0x9100_0442, // ADD X2,X2,#1
        0xd420_0000, // BRK
    ];
    // Read a nonzero value from immutable code backing; keep the store on a
    // different page so retry and cold completion do not invalidate this load.
    let exclusive_value = u64::from(words[0]) | (u64::from(words[1]) << 32);
    for case in 0..3 {
        let calls = Arc::new(AtomicUsize::new(0));
        let memory = fixture(&words, (case == 2).then(|| calls.clone()));
        let reference = fixture(&words, None);
        let target_address = if case == 0 { DATA as u64 + 128 } else { 0x3000 };
        let cache = Cache::new().unwrap();
        let process = Arc::new(Lifetime::new(cache.clone()).unwrap());
        let mut reader = process.register().unwrap();
        let mut compiler = Compiler::for_arena(native_abi(), ARENA).unwrap();
        let second_key = key().at(GuestVirtualAddress::new(PC + 16)).unwrap();
        let Request::Owner(claim) = reader.claim(second_key).unwrap() else {
            panic!()
        };
        let target = compiler
            .publish(
                Compilation::capture(claim, &memory).unwrap(),
                &process,
                &cache,
                &memory,
            )
            .unwrap();
        let second = process.snapshot(target).unwrap();
        let Request::Owner(claim) = reader.claim(key()).unwrap() else {
            panic!()
        };
        let compilation = Compilation::capture(claim, &memory).unwrap();
        let lowered = compiler
            .lower(&compilation.fragment, compilation.identity.version())
            .unwrap();
        let handle = publish_staged(compilation, lowered, &process, &cache, &memory);
        let first = process.snapshot(handle).unwrap();
        install(&process, handle, target);
        assert_ne!(first.version, second.version);
        let mut state = A64State::default();
        state.set_pc(PC);
        state.set_nzcv(Nzcv::from_bits(0xa000_0000));
        state.general_register_storage_mut()[0] = 41;
        state.general_register_storage_mut()[1] = PC;
        state.general_register_storage_mut()[2] = 99;
        state.general_register_storage_mut()[3] = target_address - 16;
        let mut expected = state.clone();
        let expected_monitor = RefCell::new(ExclusiveMonitorState::default());
        let events = VcpuEventState::default();
        let context = InterpreterContext::new(
            ProcessCpuContext::new(TargetPlatform::Switch1, SPACE),
            &reference,
            &expected_monitor,
            &Timer,
            &events,
        );
        let completed = if case == 0 { 7 } else { 5 };
        for &word in &words[..completed] {
            assert_eq!(
                execute_one_with_context(context, &mut expected, word).unwrap(),
                InstructionStep::Continue
            );
        }
        let mut worker = WorkerFaultContext::register().unwrap();
        let mut monitor = ExclusiveMonitorState::default();
        let mut frame = NativeFrame::new(&mut state, PollBudget::new(4096, 1000).unwrap());
        assert_eq!(
            memory.direct_protection_at(SPACE, GuestVirtualAddress::new(DATA as u64)),
            Some(DirectProtection::Read)
        );
        let exit = unsafe {
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
        assert_eq!(frame.budget.slice_remaining, 1000 - completed as i64);
        assert_eq!(frame.budget.sample_remaining, 4096 - completed as i64);
        assert_eq!((frame.execution_epoch, frame.admission_epoch), (0, 0));
        assert_eq!((frame.host_fp.active, frame.host_fp.saved), (0, 0));
        assert_eq!(frame.exclusive_load.bytes, 0);
        if case == 0 {
            assert_eq!(frame.exit_source_version, second.version.get());
        }
        assert_eq!(state, expected, "case={case}");
        assert_eq!(monitor, expected_monitor.into_inner(), "case={case}");
        assert_eq!(
            monitor.reservation().unwrap().expected,
            MemoryValue::U64(exclusive_value)
        );
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        // The result owns its precise instruction even after both native units
        // disappear. A same-thread mutation also detects a leaked mapping lease.
        drop(first);
        drop(second);
        drop(worker);
        drop(reader);
        assert!(process.try_shutdown().unwrap());
        drop(process);
        drop(cache);
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
                GuestVirtualAddress::new(PC + 20),
                &0xd503_201fu32.to_le_bytes(),
            )
            .unwrap();
        match exit {
            Exit::Native {
                returned,
                guest,
                instruction,
            } if case == 0 => {
                assert_eq!(returned.reason, NativeExitReason::Architectural);
                assert!(!returned.poll.exhausted && !returned.poll.sample);
                assert_eq!(guest.pc.get(), PC + 28);
                assert_eq!(guest.kind, EdgeKind::Breakpoint(0));
                assert_eq!(instruction.bits, words[7]);
                assert_eq!(
                    memory.direct_protection_at(SPACE, GuestVirtualAddress::new(DATA as u64)),
                    Some(DirectProtection::ReadWrite)
                );
                assert_eq!(
                    memory
                        .read(
                            SPACE,
                            GuestVirtualAddress::new(target_address),
                            MemoryAccess::normal(MemoryAccessSize::Doubleword)
                        )
                        .unwrap()
                        .value,
                    MemoryValue::U64(44)
                );
            }
            Exit::Memory {
                instruction,
                outcome,
            } if case != 0 => {
                assert_eq!(instruction.bits, words[5]);
                assert_eq!(instruction.key.block_key().pc.get(), PC + 20);
                assert_eq!(matches!(outcome, MemoryExit::Cold(_)), case == 2);
                let result = outcome
                    .complete(
                        instruction,
                        &mut state,
                        &memory,
                        &mut monitor,
                        completed as u64,
                    )
                    .unwrap();
                if case == 1 {
                    let Some(CpuExit::DataFault { source, fault }) = result else {
                        panic!()
                    };
                    assert_eq!(source.pc.get(), PC + 20);
                    assert_eq!(fault.address.get(), target_address);
                    assert_eq!(fault.reason, DataAccessFaultReason::Unmapped);
                    assert_eq!(state, expected);
                } else {
                    assert!(result.is_none());
                    expected.set_pc(PC + 24);
                    assert_eq!(state, expected);
                    assert_eq!(calls.load(Ordering::Relaxed), 1);
                }
                assert_eq!(
                    monitor.reservation().unwrap().expected,
                    MemoryValue::U64(exclusive_value)
                );
            }
            _ => panic!("wrong chained exit for case {case}"),
        }
    }
}
