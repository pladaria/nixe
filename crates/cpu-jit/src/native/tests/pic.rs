use super::*;
use crate::native::pic::{Record, Table, probe, set_index};
use nixe_cpu::{platform::TargetPlatform, profile::CpuProfileId, state::a64::Nzcv};
use nixe_memory::{AddressSpaceId, GuestVirtualAddress};

#[test]
fn native_pic_rejects_targets_in_reserved_registers_or_transfer_storage() {
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let (source, _) = canonical::complete(abi);
        let target = BlockKey {
            address_space: AddressSpaceId::new(1),
            pc: GuestVirtualAddress::new(0),
            profile: CpuProfileId::new(1),
            platform: TargetPlatform::Switch1,
            fp: FpSpecialization::Dynamic,
        };
        for pc in [
            integer(abi.reserved().link_scratch[0]),
            spill(0, 8),
            ValueLocation::constant(1_u128 << 64),
        ] {
            assert_eq!(
                probe::emit(&source, target, pc),
                Err(TransferError::InvalidContract(
                    "invalid PIC target location"
                ))
            );
        }
    }
}

#[test]
fn native_pic_executes_both_ways_and_misses_without_changing_guest_state() {
    let _restore = crate::fp_env::tests::RestoreHost::new();
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        for wide in [false, true] {
            for flags in 0..4 {
                for operand in 0..4 {
                    let (mut source, mut entry) = canonical::complete(abi);
                    if flags < 2 {
                        source.nzcv = NzcvLocation::Host {
                            carry_inverted: flags == 1,
                        };
                        entry.nzcv = source.nzcv.clone();
                    } else if flags == 3 {
                        source.nzcv = NzcvLocation::Deferred(LazyFlags::Packed(spill(3240, 4)));
                    }
                    let target = BlockKey {
                        address_space: AddressSpaceId::new(if wide {
                            0x1234_5678_9abc_def0
                        } else {
                            1
                        }),
                        pc: GuestVirtualAddress::new(if wide {
                            0x1234_5678_9abc_def0
                        } else {
                            0x1000
                        }),
                        profile: CpuProfileId::new(if wide { 0xfedc_ba98_7654_3210 } else { 1 }),
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
                    };
                    // RAX/X0, allocator spill, constant and vector-held PC. The
                    // probe must not destroy its own operand while saving flags.
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
                    probe::emit(&source, target, pc).unwrap();
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
                        bytes.extend(probe::emit(&source, target, pc).unwrap());
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
                    // Each mutation is made between completed native invocations.
                    let mut record = Box::new(Record::new(source.site, target, 0));
                    let mut collision = Box::new(Record::new(source.site, target, 0));
                    collision.source ^= 1 << 40;
                    for case in 0..14 {
                        *record = Record::new(source.site, target, 0);
                        match case {
                            4 => record.source ^= 1 << 40,
                            5 => record.state_map ^= 1,
                            6 => record.platform ^= 1,
                            7 => record.pc ^= 1 << 40,
                            8 => record.address_space ^= 1 << 40,
                            9 => record.profile ^= 1 << 40,
                            10 => record.fp ^= 1 << 32,
                            11 => record.fp ^= 1,
                            12 => record.pc ^= 1, // misaligned, same set selector
                            _ => {}
                        }
                        unsafe {
                            table.set(slot, std::ptr::null());
                            table.set(slot + 1, std::ptr::null());
                            if case != 2 {
                                table.set(slot + usize::from(case == 1), &*record);
                            }
                            if case == 13 {
                                table.set(slot, &*collision);
                                table.set(slot + 1, &*record);
                            }
                        }
                        for nibble in 0..16 {
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
                            let hit = case < 2 || case == 13;
                            let mut before = state.clone();
                            before.set_pc(if hit { 0x4444 } else { 0x8888 });
                            {
                                let mut frame =
                                    NativeFrame::new(&mut state, PollBudget::new(7, 11).unwrap());
                                for (i, byte) in
                                    target.pc.get().to_le_bytes().into_iter().enumerate()
                                {
                                    frame.spill[3400 + i] = MaybeUninit::new(byte);
                                }
                                let mut invocation =
                                    unsafe { reader.admit(&mut frame, published::key()) }
                                        .unwrap()
                                        .unwrap();
                                let address =
                                    invocation.payload().preferred().unwrap().canonical.get();
                                record.address = address + hit_offset;
                                collision.address = record.address;
                                let frame = invocation.frame();
                                frame.indirect_pic = if case == 3 {
                                    std::ptr::null()
                                } else {
                                    table.as_ptr()
                                };
                                let outcome = unsafe {
                                    enter_protected(
                                        frame,
                                        std::ptr::dangling_mut(),
                                        address as *const u8,
                                    )
                                }
                                .unwrap();
                                assert_eq!(
                                    outcome.reason,
                                    if hit {
                                        NativeExitReason::Dispatch
                                    } else {
                                        NativeExitReason::Control
                                    },
                                    "{abi:?}, flags={flags}, operand={operand}, case={case}, nzcv={nibble}"
                                );
                                assert_eq!(frame.exit_pc, if hit { 0x4444 } else { 0x8888 });
                            }
                            assert_eq!(
                                state, before,
                                "guest state changed: {abi:?}, flags={flags}, operand={operand}, case={case}, nzcv={nibble}"
                            );
                        }
                    }
                    // Remove the last raw pointer before its record/executable owner.
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
