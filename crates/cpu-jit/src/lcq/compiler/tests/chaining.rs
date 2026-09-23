//! Independently published LCQ chains using the registered Closed linker.
use super::*;
use crate::lifetime::Reason;
use std::mem::offset_of;
use std::sync::{
    atomic::{AtomicU32, Ordering},
    mpsc,
};

pub(super) fn publish_staged(
    compilation: Compilation<'_>,
    lowered: Lowered,
    process: &Lifetime,
    cache: &Arc<Cache>,
    memory: &(impl ExecutableMemory + nixe_memory::MemoryInvalidationSource),
) -> UnitHandle {
    let handle = Compiler::publish_lowered(compilation, lowered, process, cache, memory).unwrap();
    // Keep these instrumented fixtures on their fallbacks until `install`
    // deliberately redirects them; use normal optional-work deferral.
    if let Some(mut transition) = process.try_transition().unwrap() {
        transition.wait_closed().unwrap();
        transition
            .batch()
            .unwrap()
            .complete_with_links_deferred()
            .unwrap();
        assert!(transition.try_reopen().unwrap());
    }
    handle
}

pub(super) fn install(
    process: &Lifetime,
    source: UnitHandle,
    target: UnitHandle,
) -> crate::lifetime::unit::links::LinkHandle {
    let from = process.snapshot(source).unwrap();
    let to = process.snapshot(target).unwrap();
    let (island, _) = from
        .states
        .iter()
        .filter(|state| {
            state
                .transfer
                .as_ref()
                .is_some_and(|transfer| transfer.static_target.is_some())
        })
        .enumerate()
        .find(|(_, state)| {
            state.transfer.as_ref().unwrap().static_target == Some(to.entries[0].key)
        })
        .unwrap();
    process.request(Reason::LinkPatch).unwrap();
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    let handle = transition
        .refresh_static_link(source, island)
        .unwrap()
        .unwrap();
    assert!(transition.drain_links().unwrap());
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
    handle
}

/// Instrument only this fixture's fast entry before publication. Real source
/// patchpoints/bridges remain untouched until the Closed linker installs them.
/// The wait's CMP is safe for this x86 fixture's packed (not host) NZCV inputs.
pub(super) fn pause_fast_entry(lowered: &mut Lowered) {
    let abi = lowered.entry.abi;
    if abi == HostAbi::X86_64 {
        assert!(!matches!(
            lowered.entry.nzcv,
            crate::abi::NzcvLocation::Host { .. }
        ));
    }
    let (wrapper, tail) = crate::native::link::bridge(abi, &handoff(abi));
    let mut bytes = lowered.output.bytes.to_vec();
    let start = append(&mut bytes, &wrapper);
    let branch =
        crate::native::link::emit(abi, (start + tail) as u64, u64::from(lowered.fast), 0).unwrap();
    assert!(branch.island.is_none());
    bytes[start + tail..start + tail + branch.patch().len()].copy_from_slice(branch.patch());
    let label = lowered
        .output
        .metadata
        .entries
        .iter_mut()
        .find(|(_, offset)| *offset == lowered.fast)
        .unwrap();
    label.1 = start as u32;
    lowered.fast = start as u32;
    lowered.output.bytes = bytes.into_boxed_slice();
}

/// Test-only native rendezvous at the first hot continuation. Signal ready
/// with release, then acquire-wait for the requester. No Rust call or mapped
/// operand clobber. Runtime points to two live adjacent AtomicU32 words.
fn handoff(abi: HostAbi) -> Vec<u8> {
    let offset = offset_of!(NativeFrame<'static>, runtime) as u32;
    match abi {
        HostAbi::X86_64 => {
            let mut bytes = vec![0x4d, 0x8b, 0x9f]; // MOV R11,[R15+runtime]
            bytes.extend(offset.to_le_bytes());
            bytes.extend([0x41, 0xc7, 0x03, 1, 0, 0, 0]); // MOV [R11],1
            bytes.extend([0x41, 0x83, 0x7b, 4, 0, 0x74, 0xf9]); // wait [R11+4]
            bytes
        }
        HostAbi::Aarch64 => [
            0xf94002b0 | ((offset / 8) << 10), // LDR X16,[X21+runtime]
            0x52800031,                        // MOV W17,#1
            0x889ffe11,                        // STLR W17,[X16]
            0x91001210,                        // ADD X16,X16,#4
            0x88dffe11,                        // LDAR W17,[X16]
            0x34fffff1,                        // CBZ W17,previous load
        ]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect(),
    }
}

#[test]
fn selective_bridge_keeps_absent_dirty_state_and_loads_clean_inputs() {
    crate::native::check_host().unwrap();
    // A's X19/V1/X4 and NZV are not B inputs. B needs clean X0/V2 absent
    // from A's physical map, plus A's lazy C. All homes must agree at BRK.
    let words = [
        0x9100_0673, // ADD X19,X19,#1
        0x4ea3_1c61, // ORR V1.16B,V3.16B,V3.16B
        0xf100_0484, // SUBS X4,X4,#1
        0x1400_0001, // B B
        0x9a1f_0000, // ADC X0,X0,XZR
        0x4ea2_1c40, // ORR V0.16B,V2.16B,V2.16B
        0xd420_0000,
    ];
    let memory = memory(&words);
    let cache = Cache::new().unwrap();
    let process = Arc::new(Lifetime::new(cache.clone()).unwrap());
    let mut reader = process.register().unwrap();
    let mut compiler = Compiler::new(native_abi()).unwrap();
    let target_key = key().at(GuestVirtualAddress::new(PC + 16)).unwrap();
    let Request::Owner(claim) = reader.claim(target_key).unwrap() else {
        panic!()
    };
    let handle = compiler
        .publish(
            Compilation::capture(claim, &memory).unwrap(),
            &process,
            &cache,
            &memory,
        )
        .unwrap();
    let second = process.snapshot(handle).unwrap();
    assert_eq!(second.entries[0].contract.live_in.nzcv, crate::analysis::C);
    let Request::Owner(claim) = reader.claim(key()).unwrap() else {
        panic!()
    };
    let compilation = Compilation::capture(claim, &memory).unwrap();
    let lowered = compiler
        .lower(&compilation.fragment, compilation.identity.version())
        .unwrap();
    let site = lowered
        .states
        .iter()
        .position(|state| {
            state
                .transfer
                .as_ref()
                .is_some_and(|transfer| transfer.static_target == Some(target_key))
        })
        .unwrap() as u32;
    let src = publish_staged(compilation, lowered, &process, &cache, &memory);
    process.request(Reason::LinkPatch).unwrap();
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    let prepared = transition.prepare_link(src, site, handle, 0, 0).unwrap();
    assert!(
        !crate::native::emit_chain_transfer(prepared.source_state(), prepared.target_contract())
            .unwrap()
            .is_empty()
    );
    let installed = transition.register_link(prepared).unwrap();
    assert!(transition.install_link(installed).unwrap());
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
    drop(transition);
    // Only the registered graph owns the bridge; the native invocation spans
    // A -> owned bridge -> B and the real canonical exit from B.
    for value in [0, 1, 0x8000_0000_0000_0000] {
        let mut state = integer::initial_state();
        state.general_register_storage_mut()[4] = value;
        let mut expected = state.clone();
        for word in &words[..6] {
            nixe_cpu_interpreter::execute_one(&TargetPlatform::Switch1, &mut expected, *word)
                .unwrap();
        }
        let mut frame = NativeFrame::new(&mut state, PollBudget::new(4096, 1000).unwrap());
        let mut invocation = unsafe { reader.admit(&mut frame, key()) }.unwrap().unwrap();
        let entry = invocation.payload().preferred().unwrap();
        let returned = unsafe {
            crate::native::enter_protected(
                invocation.frame(),
                std::ptr::null_mut(),
                entry.canonical.get() as *const u8,
            )
        }
        .unwrap();
        assert_eq!(returned.reason, NativeExitReason::Architectural);
        assert_eq!(invocation.frame().exit_source_version, second.version.get());
        assert_eq!(invocation.frame().budget.slice_remaining, 994);
        drop(invocation);
        assert_eq!(state, expected);
    }
    process.request(Reason::LinkPatch).unwrap();
    let mut transition = process.try_transition().unwrap().unwrap();
    transition.wait_closed().unwrap();
    process.retire_unit(handle).unwrap();
    assert_eq!(transition.drain_retirements().unwrap(), 1);
    transition.batch().unwrap().complete().unwrap();
    assert!(transition.try_reopen().unwrap());
    let mut state = integer::initial_state();
    let mut expected = state.clone();
    for word in &words[..4] {
        nixe_cpu_interpreter::execute_one(&TargetPlatform::Switch1, &mut expected, *word).unwrap();
    }
    let mut frame = NativeFrame::new(&mut state, PollBudget::new(4096, 1000).unwrap());
    let mut invocation = unsafe { reader.admit(&mut frame, key()) }.unwrap().unwrap();
    let entry = invocation.payload().preferred().unwrap();
    let returned = unsafe {
        crate::native::enter_protected(
            invocation.frame(),
            std::ptr::null_mut(),
            entry.canonical.get() as *const u8,
        )
    }
    .unwrap();
    assert_eq!(returned.reason, NativeExitReason::Dispatch);
    assert_eq!(invocation.frame().budget.slice_remaining, 996);
    drop(invocation);
    assert_eq!(state, expected);
}

#[test]
fn request_after_poll_resume_exits_later_unit_before_maintenance_can_close() {
    crate::native::check_host().unwrap();
    let words = [
        0xba1f0021, 0xf1000400, 0x14000001, // A: ADCS; SUBS; B B
        0xba1f0021, 0xf1000400, 0x17fffffe, // B: ADCS; SUBS; B B
    ];
    let memory = memory(&words);
    let cache = Cache::new().unwrap();
    let process = Arc::new(Lifetime::new(cache.clone()).unwrap());
    let mut reader = process.register().unwrap();
    let mut compiler = Compiler::new(native_abi()).unwrap();
    let second_key = key().at(GuestVirtualAddress::new(PC + 12)).unwrap();
    let Request::Owner(claim) = reader.claim(second_key).unwrap() else {
        panic!()
    };
    let compilation = Compilation::capture(claim, &memory).unwrap();
    let mut lowered = compiler
        .lower(&compilation.fragment, compilation.identity.version())
        .unwrap();
    pause_fast_entry(&mut lowered);
    let target = publish_staged(compilation, lowered, &process, &cache, &memory);
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
    install(&process, target, target);
    let incoming = install(&process, handle, target);
    assert_ne!(first.version, second.version);
    let rendezvous = [AtomicU32::new(0), AtomicU32::new(0)];
    let mut cpu = integer::initial_state();
    let mut expected = cpu.clone();
    // A's 3 instructions trigger the first sample-only poll. After rearm the
    // requester arrives; B then completes ceil(4096/3) loops before observing it.
    const COMPLETED: i64 = 3 + 4096u64.div_ceil(3) as i64 * 3;
    for word in words[..3]
        .iter()
        .chain(words[3..].iter().cycle().take((COMPLETED - 3) as usize))
    {
        nixe_cpu_interpreter::execute_one(&TargetPlatform::Switch1, &mut expected, *word).unwrap();
    }
    expected.set_fpsr(expected.fpsr() | 2);
    let _restore = crate::fp_env::tests::RestoreHost::new();
    let caller = crate::fp_env::tests::distinct_caller();
    let mut frame = NativeFrame::new(&mut cpu, PollBudget::new(3, 10000).unwrap());
    frame.runtime = rendezvous.as_ptr().cast_mut().cast();
    let mut invocation = unsafe { reader.admit(&mut frame, key()) }.unwrap().unwrap();
    let entry = invocation.payload().preferred().unwrap();
    let (closed_tx, closed_rx) = mpsc::channel();
    let (waiting_tx, waiting_rx) = mpsc::channel();
    std::thread::scope(|scope| {
        scope.spawn(|| {
            while rendezvous[0].load(Ordering::Acquire) == 0 {
                std::thread::yield_now();
            }
            let ticket = process.retire_unit(target);
            // Always release the native waiter, even if request creation failed.
            rendezvous[1].store(1, Ordering::Release);
            let ticket = ticket.unwrap();
            let mut transition = process.try_transition().unwrap().unwrap();
            waiting_tx.send(()).unwrap();
            transition.wait_closed().unwrap();
            assert!(transition.drain_links().unwrap());
            assert_eq!(
                transition.install_link(incoming),
                Err(crate::lifetime::Error::StaleUnit)
            );
            // Both incoming A->B and B's self edge are detached before the
            // target can retire. Existing compiler snapshots still protect B.
            assert_eq!(process.reclaim_units().unwrap(), 0);
            closed_tx.send(()).unwrap();
            transition.batch().unwrap().complete().unwrap();
            assert!(transition.try_reopen().unwrap());
            assert!(
                process
                    .maintenance_complete(crate::lifetime::Reason::Eviction, ticket)
                    .unwrap()
            );
        });
        let (frame, lookup) = invocation.frame_and_faults();
        let epoch = frame.execution_epoch;
        let returned = unsafe {
            frame.ensure_fp().unwrap();
            crate::fp_env::tests::divide_by_zero();
            crate::native::enter_protected(
                frame,
                std::ptr::null_mut(),
                entry.canonical.get() as *const u8,
            )
        }
        .unwrap();
        assert_eq!(returned.reason, NativeExitReason::Control);
        assert!(!returned.poll.sample && !returned.poll.exhausted);
        assert_eq!(frame.budget.slice_remaining, 10000 - COMPLETED);
        assert_eq!(frame.budget.sample_remaining, 4094);
        assert_eq!(frame.exit_source_version, second.version.get());
        assert_eq!(lookup.unit(frame.exit_native_pc).unwrap().id, second.id);
        assert_eq!(frame.exit_state_map, 0);
        assert_eq!((frame.host_fp.active, frame.host_fp.saved), (0, 0));
        let mut probe = crate::abi::HostFpState::default();
        unsafe {
            probe.begin();
            probe.finish();
        }
        assert_eq!((probe.saved_control, probe.saved_status), caller);
        assert_eq!(frame.execution_epoch, epoch);
        assert_ne!(epoch, 0);
        waiting_rx.recv().unwrap();
        assert_eq!(closed_rx.try_recv(), Err(mpsc::TryRecvError::Empty));
        drop(invocation);
        closed_rx.recv().unwrap();
    });
    assert_eq!(frame.execution_epoch, 0);
    assert_eq!(cpu, expected);
    assert_eq!(process.control_word().load(Ordering::Acquire), 0);
    drop(second);
    assert_eq!(process.reclaim_units().unwrap(), 1);
    // A remains published and reaches its real restored canonical fallback.
    let mut cpu = integer::initial_state();
    let mut expected = cpu.clone();
    for word in &words[..3] {
        nixe_cpu_interpreter::execute_one(&TargetPlatform::Switch1, &mut expected, *word).unwrap();
    }
    let mut frame = NativeFrame::new(&mut cpu, PollBudget::new(4096, 1000).unwrap());
    let mut invocation = unsafe { reader.admit(&mut frame, key()) }.unwrap().unwrap();
    let entry = invocation.payload().preferred().unwrap();
    let returned = unsafe {
        crate::native::enter_protected(
            invocation.frame(),
            std::ptr::null_mut(),
            entry.canonical.get() as *const u8,
        )
    }
    .unwrap();
    assert_eq!(returned.reason, NativeExitReason::Dispatch);
    assert_eq!(invocation.frame().budget.slice_remaining, 997);
    drop(invocation);
    assert_eq!(cpu, expected);
}
