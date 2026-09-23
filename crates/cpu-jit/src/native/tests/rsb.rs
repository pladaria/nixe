use super::*;
use crate::native::pic::{Record, Table, set_index};
use crate::native::rsb::{emit_push, emit_return_probe};
use crate::rsb::{CAPACITY, Continuation, ReturnStack};
use nixe_cpu::{
    platform::TargetPlatform,
    profile::CpuProfileId,
    state::a64::{A64State, Nzcv},
};
use nixe_memory::{AddressSpaceId, GuestVirtualAddress};

fn key(wide: bool) -> BlockKey {
    BlockKey {
        address_space: AddressSpaceId::new(if wide { 0xfedc_ba98_7654_3210 } else { 1 }),
        pc: GuestVirtualAddress::new(if wide { 0x1234_5678_9abc_0000 } else { 0x1000 }),
        profile: CpuProfileId::new(if wide { 0x8765_4321_fedc_ba98 } else { 1 }),
        platform: if wide {
            TargetPlatform::Switch2
        } else {
            TargetPlatform::Switch1
        },
        fp: if wide {
            FpSpecialization::Exact(u32::MAX)
        } else {
            FpSpecialization::Dynamic
        },
    }
}

#[test]
fn shared_cold_exits_update_the_return_stack_exactly_once() {
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let (mut source, entry) = contracts(abi, &[]);
        let target = key(true);
        let pc = ValueLocation::constant(target.pc.get().into());
        for returning in [false, true] {
            let operation = if returning {
                crate::native::rsb::emit_return_update(&source, target, pc).unwrap()
            } else {
                emit_push(&source, target).unwrap()
            };
            for indirect in [false, true] {
                let (_, [slice, control]) =
                    super::super::canonical::emit_polled_exit(&source, pc, &operation, indirect)
                        .unwrap();
                if !canonical::native(abi) {
                    continue;
                }
                for (path, offset) in [0, slice, control].into_iter().enumerate() {
                    let owner = published::Published::new(|process, cache| {
                        let identity = process.begin_unit(crate::executable::Tier::Lcq).unwrap();
                        source.site = ExitSiteKey {
                            source: identity.version(),
                            state_map: 0,
                        };
                        let (shared, _) = super::super::canonical::emit_polled_exit(
                            &source, pc, &operation, indirect,
                        )
                        .unwrap();
                        let mut bytes = gateway::landing(abi);
                        if indirect && path == 0 {
                            // The live probe performs the operation before falling
                            // through on a cache miss; the fallback must not repeat it.
                            bytes.extend_from_slice(&operation);
                        }
                        if abi == HostAbi::X86_64 {
                            bytes.push(0xe9);
                            bytes.extend((offset as i32).to_le_bytes());
                        } else {
                            bytes.extend((0x14000000 | ((offset as u32 + 4) / 4)).to_le_bytes());
                        }
                        let exit_offset = bytes.len() as u32;
                        bytes.extend_from_slice(&shared);
                        published::encoded(
                            identity,
                            cache,
                            bytes,
                            crate::lifetime::unit::Entry {
                                key: published::key(),
                                canonical_offset: 0,
                                fast_offset: 0,
                                contract: entry.clone(),
                            },
                            vec![crate::lifetime::unit::StateRecord {
                                exit: None,
                                transfer: None,
                                native_offset: exit_offset,
                                state: source.clone(),
                            }],
                            Box::new([]),
                        )
                    });
                    let mut reader = owner.process.register().unwrap();
                    let mut stack = ReturnStack {
                        entries: [Continuation::from(target); CAPACITY],
                        head: 2,
                        depth: 2,
                    };
                    let mut state = A64State::default();
                    {
                        let mut frame =
                            NativeFrame::new(&mut state, PollBudget::new(77, 1000).unwrap())
                                .with_return_stack(&mut stack);
                        let mut invocation = unsafe { reader.admit(&mut frame, published::key()) }
                            .unwrap()
                            .unwrap();
                        let address = invocation.payload().preferred().unwrap().canonical.get();
                        let frame = invocation.frame();
                        let result = unsafe {
                            enter_protected(frame, std::ptr::null_mut(), address as *const u8)
                        }
                        .unwrap();
                        assert_eq!(
                            result.reason,
                            if path == 2 {
                                NativeExitReason::Control
                            } else {
                                NativeExitReason::Dispatch
                            }
                        );
                        assert_eq!(frame.budget.slice_remaining, 1000);
                    }
                    assert_eq!(stack.depth, if returning { 1 } else { 3 });
                    assert_eq!(stack.head, stack.depth);
                    assert_eq!(state.pc(), target.pc.get());
                    drop(reader);
                    owner.shutdown();
                }
            }
        }
    }
}

#[test]
fn native_rsb_return_checks_full_keys_pops_and_uses_only_matched_pics() {
    let _restore = crate::fp_env::tests::RestoreHost::new();
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        for flags in 0..4 {
            let (mut source, mut entry) = canonical::complete(abi);
            if flags < 2 {
                source.nzcv = NzcvLocation::Host {
                    carry_inverted: flags == 1,
                };
                entry.nzcv = source.nzcv.clone();
            } else if flags == 3 {
                source.nzcv = NzcvLocation::Deferred(LazyFlags::Packed(spill(3240, 4)));
            }
            for wide in [false, true] {
                let target = key(wide);
                for operand in 0..4 {
                    let pc = match operand {
                        0 => {
                            source
                                .bindings
                                .iter()
                                .find(|b| b.value == GuestValue::General(0))
                                .unwrap()
                                .location
                        }
                        1 => spill(3400, 8),
                        2 => ValueLocation::constant(target.pc.get().into()),
                        _ => vector(0),
                    };
                    emit_return_probe(&source, target, pc).unwrap();
                    if !canonical::native(abi) {
                        continue;
                    }
                    let mut hit_offset = 0;
                    let owner = published::Published::new(|process, cache| {
                        let identity = process.begin_unit(crate::executable::Tier::Lcq).unwrap();
                        source.site = ExitSiteKey {
                            source: identity.version(),
                            state_map: 0,
                        };
                        let mut bytes = gateway::landing(abi);
                        bytes.extend(emit_canonical_entry(&entry).unwrap());
                        bytes.extend(emit_return_probe(&source, target, pc).unwrap());
                        let miss_offset = bytes.len() as u32;
                        bytes.extend(
                            emit_canonical_exit(
                                &source,
                                ValueLocation::constant(0x8888),
                                NativeExitReason::Control,
                                0,
                            )
                            .unwrap(),
                        );
                        hit_offset = bytes.len();
                        bytes.extend(gateway::landing(abi));
                        let mut hit_state = source.clone();
                        hit_state.site.state_map = 1;
                        bytes.extend(
                            emit_canonical_exit(
                                &hit_state,
                                ValueLocation::constant(0x4444),
                                NativeExitReason::Dispatch,
                                0,
                            )
                            .unwrap(),
                        );
                        published::encoded(
                            identity,
                            cache,
                            bytes,
                            crate::lifetime::unit::Entry {
                                key: published::key(),
                                canonical_offset: 0,
                                fast_offset: 0,
                                contract: entry.clone(),
                            },
                            vec![
                                crate::lifetime::unit::StateRecord {
                                    exit: None,
                                    transfer: None,
                                    native_offset: miss_offset,
                                    state: source.clone(),
                                },
                                crate::lifetime::unit::StateRecord {
                                    exit: None,
                                    transfer: None,
                                    native_offset: hit_offset as u32,
                                    state: hit_state,
                                },
                            ],
                            Box::new([]),
                        )
                    });
                    let mut reader = owner.process.register().unwrap();
                    let table = Table::new();
                    let slot = set_index(source.site, target) * 2;
                    let mut record = Box::new(Record::new(source.site, target, 0));
                    for case in 0..12 {
                        // 0/1 hit either way; 2 no cached bridge; 3..7 wrong RSB
                        // field; 8 empty RSB; 9 absent RSB; 10 absent PIC;
                        // 11 wrong PIC source, even though the RSB matches.
                        record.source =
                            source.site.source.get() ^ if case == 11 { 1 << 40 } else { 0 };
                        unsafe {
                            table.set(slot, std::ptr::null());
                            table.set(slot + 1, std::ptr::null());
                            if case != 2 {
                                table.set(slot + usize::from(case == 1), &*record);
                            }
                        }
                        for head in [0_u32, 15] {
                            for depth in [1, 16] {
                                for nibble in 0..16 {
                                    let mut stack = ReturnStack {
                                        entries: [Continuation::from(target); CAPACITY],
                                        head,
                                        depth: if case == 8 { 0 } else { depth },
                                    };
                                    let top = ((head.wrapping_sub(1)) & 15) as usize;
                                    match case {
                                        3 => stack.entries[top].pc ^= 1, // Misaligned architectural target must not match.
                                        4 => stack.entries[top].address_space ^= 1 << 40,
                                        5 => stack.entries[top].profile ^= 1 << 40,
                                        6 => stack.entries[top].platform ^= 1,
                                        7 => stack.entries[top].fp ^= 1 << 32, // Dynamic vs Exact(0).
                                        _ => {}
                                    }
                                    let mut expected_stack = stack.clone();
                                    if (3..=8).contains(&case) {
                                        expected_stack.clear();
                                    } else if case != 9 {
                                        expected_stack.head = top as u32;
                                        expected_stack.depth -= 1;
                                    }
                                    let hit = case < 2;
                                    let (mut state, _) = canonical::pattern(&entry);
                                    state.set_fpcr(0);
                                    state.set_nzcv(Nzcv::from_bits(nibble << 28));
                                    state.general_register_storage_mut()[0] = target.pc.get();
                                    if operand == 3 {
                                        state.set_vector(
                                            0,
                                            u128::from(target.pc.get()) | (0xabcd_u128 << 64),
                                        );
                                    }
                                    let mut expected_state = state.clone();
                                    expected_state.set_pc(if hit { 0x4444 } else { 0x8888 });
                                    {
                                        let mut frame = NativeFrame::new(
                                            &mut state,
                                            PollBudget::new(7, 11).unwrap(),
                                        );
                                        if case != 9 {
                                            frame = frame.with_return_stack(&mut stack);
                                        }
                                        for (i, byte) in
                                            target.pc.get().to_le_bytes().into_iter().enumerate()
                                        {
                                            frame.spill[3400 + i] = MaybeUninit::new(byte);
                                        }
                                        let mut invocation =
                                            unsafe { reader.admit(&mut frame, published::key()) }
                                                .unwrap()
                                                .unwrap();
                                        let address = invocation
                                            .payload()
                                            .preferred()
                                            .unwrap()
                                            .canonical
                                            .get();
                                        record.address = address + hit_offset;
                                        let frame = invocation.frame();
                                        frame.indirect_pic = if case == 10 {
                                            std::ptr::null()
                                        } else {
                                            table.as_ptr()
                                        };
                                        let returned = unsafe {
                                            enter_protected(
                                                frame,
                                                std::ptr::dangling_mut(),
                                                address as *const u8,
                                            )
                                        }
                                        .unwrap();
                                        assert_eq!(
                                            returned.reason,
                                            if hit {
                                                NativeExitReason::Dispatch
                                            } else {
                                                NativeExitReason::Control
                                            },
                                            "{abi:?}, case={case}"
                                        );
                                    }
                                    assert_eq!(
                                        state, expected_state,
                                        "{abi:?}, case={case}, flags={flags}, nzcv={nibble}, operand={operand}"
                                    );
                                    assert_eq!(
                                        stack, expected_stack,
                                        "{abi:?}, case={case}, head={head}, depth={depth}"
                                    );
                                }
                            }
                        }
                    }
                    unsafe {
                        table.set(slot, std::ptr::null());
                        table.set(slot + 1, std::ptr::null());
                    }
                    drop(reader);
                    owner.shutdown();
                }
            }
        }
    }
}

#[test]
fn native_rsb_push_rejects_misaligned_continuations() {
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let (source, _) = canonical::complete(abi);
        let mut target = key(false);
        target.pc = GuestVirtualAddress::new(0x1001);
        assert_eq!(
            emit_push(&source, target),
            Err(TransferError::InvalidContract("unaligned RSB continuation"))
        );
    }
}

#[test]
fn native_rsb_push_wraps_and_saturates_without_changing_guest_state() {
    let _restore = crate::fp_env::tests::RestoreHost::new();
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        for flags in 0..4 {
            let (mut source, mut entry) = canonical::complete(abi);
            if flags < 2 {
                source.nzcv = NzcvLocation::Host {
                    carry_inverted: flags == 1,
                };
                entry.nzcv = source.nzcv.clone();
            } else if flags == 3 {
                source.nzcv = NzcvLocation::Deferred(LazyFlags::Packed(spill(3240, 4)));
            }
            for wide in [false, true] {
                for count in [1, 20] {
                    let first = key(wide);
                    let continuations: Vec<_> = (0..count)
                        .map(|index| first.at(first.pc.checked_add(index * 4).unwrap()).unwrap())
                        .collect();
                    let mut bytes = Vec::new();
                    for continuation in &continuations {
                        bytes.extend(emit_push(&source, *continuation).unwrap());
                    }
                    if !canonical::native(abi) {
                        continue;
                    }
                    let owner = published::Published::new(|process, cache| {
                        published::synthetic(
                            process.begin_unit(crate::executable::Tier::Lcq).unwrap(),
                            cache,
                            entry.clone(),
                            source.clone(),
                            (bytes, None),
                            ValueLocation::constant(0x4444),
                            NativeExitReason::Dispatch,
                        )
                    });
                    let mut reader = owner.process.register().unwrap();
                    for head in [0, 1, 15] {
                        for depth in [0, 1, 15, 16] {
                            for nibble in 0..16 {
                                let (mut state, _) = canonical::pattern(&entry);
                                state.set_fpcr(0);
                                state.set_nzcv(Nzcv::from_bits(nibble << 28));
                                let mut expected_state = state.clone();
                                expected_state.set_pc(0x4444);
                                let mut stack = ReturnStack {
                                    entries: [Continuation::default(); CAPACITY],
                                    head,
                                    depth,
                                };
                                // Distinct old entries detect writes to the wrong slot, including
                                // unoccupied cells which the push has no reason to touch.
                                for (slot, value) in stack.entries.iter_mut().enumerate() {
                                    *value = Continuation::from(
                                        first
                                            .at(GuestVirtualAddress::new(0x8000 + slot as u64 * 4))
                                            .unwrap(),
                                    );
                                }
                                let mut expected = stack.clone();
                                for continuation in &continuations {
                                    expected.entries[expected.head as usize] =
                                        Continuation::from(*continuation);
                                    expected.head = (expected.head + 1) % 16;
                                    expected.depth = (expected.depth + 1).min(16);
                                }
                                {
                                    let mut frame = NativeFrame::new(
                                        &mut state,
                                        PollBudget::new(7, 11).unwrap(),
                                    )
                                    .with_return_stack(&mut stack);
                                    let mut invocation =
                                        unsafe { reader.admit(&mut frame, published::key()) }
                                            .unwrap()
                                            .unwrap();
                                    let address =
                                        invocation.payload().preferred().unwrap().canonical.get();
                                    let frame = invocation.frame();
                                    let returned = unsafe {
                                        enter_protected(
                                            frame,
                                            std::ptr::dangling_mut(),
                                            address as *const u8,
                                        )
                                    }
                                    .unwrap();
                                    assert_eq!(returned.reason, NativeExitReason::Dispatch);
                                }
                                assert_eq!(
                                    state, expected_state,
                                    "{abi:?}, flags={flags}, nzcv={nibble}"
                                );
                                assert_eq!(
                                    stack, expected,
                                    "{abi:?}, head={head}, depth={depth}, count={count}"
                                );
                            }
                        }
                    }
                    // An isolated frame may omit prediction storage. Native guest
                    // register/flag semantics must remain identical in that case.
                    let (mut state, _) = canonical::pattern(&entry);
                    state.set_fpcr(0);
                    let mut expected = state.clone();
                    expected.set_pc(0x4444);
                    {
                        let mut frame =
                            NativeFrame::new(&mut state, PollBudget::new(7, 11).unwrap());
                        let mut invocation = unsafe { reader.admit(&mut frame, published::key()) }
                            .unwrap()
                            .unwrap();
                        let address = invocation.payload().preferred().unwrap().canonical.get();
                        let frame = invocation.frame();
                        unsafe {
                            enter_protected(frame, std::ptr::dangling_mut(), address as *const u8)
                        }
                        .unwrap();
                    }
                    assert_eq!(state, expected);
                    drop(reader);
                    owner.shutdown();
                }
            }
        }
    }
}
