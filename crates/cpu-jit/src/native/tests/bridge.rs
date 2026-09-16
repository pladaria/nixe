use super::flags::captured_host_nzcv;
use super::*;
use nixe_cpu::state::a64::{A64State, Nzcv};

#[test]
fn selective_bridge_preserves_cycles_missing_inputs_and_partial_flags() {
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        for source_bits in [0, crate::analysis::C, NZCV] {
            for target_bits in [0, crate::analysis::C, NZCV] {
                for kind in 0..4 {
                    for host_target in [false, true] {
                        let (mut source, _) = contracts(
                            abi,
                            &[
                                (GuestValue::General(0), integer(0), integer(0)),
                                (GuestValue::General(1), integer(1), integer(1)),
                                (GuestValue::General(2), spill(2048, 8), spill(2048, 8)),
                                (GuestValue::General(3), spill(2056, 8), spill(2056, 8)),
                                (GuestValue::General(19), integer(2), integer(2)),
                                (GuestValue::Vector(1), vector(0), vector(0)),
                                (GuestValue::Sp, spill(2200, 8), spill(2200, 8)),
                            ],
                        );
                        let (_, mut target) = contracts(
                            abi,
                            &[
                                (GuestValue::General(0), integer(1), integer(1)),
                                (GuestValue::General(1), integer(0), integer(0)),
                                (GuestValue::General(2), spill(2056, 8), spill(2056, 8)),
                                (GuestValue::General(3), spill(2048, 8), spill(2048, 8)),
                                (GuestValue::General(4), spill(2064, 8), spill(2064, 8)),
                                (GuestValue::General(5), integer(2), integer(2)),
                                (GuestValue::Vector(2), vector(0), vector(0)),
                            ],
                        );
                        source.live.nzcv = source_bits;
                        source.dirty_live.nzcv = source_bits;
                        target.live_in.nzcv = target_bits;
                        let bits = 0x6000_0000u32;
                        source.nzcv = match kind {
                            0 => NzcvLocation::Packed(ValueLocation::Constant(bits.into())),
                            1 => NzcvLocation::Deferred(LazyFlags::Packed(
                                ValueLocation::Constant(bits.into()),
                            )),
                            2 => NzcvLocation::Host {
                                carry_inverted: true,
                            },
                            _ => NzcvLocation::Packed(integer(0)),
                        };
                        target.nzcv = if host_target {
                            NzcvLocation::Host {
                                carry_inverted: false,
                            }
                        } else {
                            NzcvLocation::Packed(spill(2080, 4))
                        };
                        let mut code = Vec::new();
                        if kind == 2 {
                            let (mut seed, mut entry) = contracts(abi, &[]);
                            seed.live.nzcv = source_bits;
                            seed.nzcv = NzcvLocation::Packed(ValueLocation::Constant(bits.into()));
                            entry.live_in.nzcv = source_bits;
                            entry.nzcv = source.nzcv.clone();
                            code.extend(emit_fast_transfer(&seed, &entry).unwrap());
                        }
                        code.extend(emit_chain_transfer(&source, &target).unwrap());
                        if !canonical::native(abi) {
                            continue;
                        }
                        let mut state = A64State::default();
                        state.set_nzcv(Nzcv::from_bits(0x9000_0000));
                        state
                            .general_register_storage_mut()
                            .iter_mut()
                            .enumerate()
                            .for_each(|(i, x)| *x = 100 + i as u64);
                        state.set_vector(2, 0x1234_5678_9abc_def0_fedc_ba98_7654_3210);
                        let initial = state.clone();
                        let mut frame =
                            NativeFrame::new(&mut state, PollBudget::new(77, 1000).unwrap());
                        let mut seed = 97;
                        for byte in &mut frame.spill {
                            *byte = MaybeUninit::new((next(&mut seed) >> 32) as u8);
                        }
                        if kind == 3 {
                            // Packed flags alias a data input inside the GPR
                            // cycle, including a 32-bit view of its 64-bit value.
                            for (i, byte) in u64::from(bits).to_le_bytes().into_iter().enumerate() {
                                frame.spill[4096 + i] = MaybeUninit::new(byte);
                            }
                        }
                        let before: Vec<u8> = frame
                            .spill
                            .iter()
                            .map(|b| unsafe { b.assume_init() })
                            .collect();
                        invoke(abi, code, &mut frame);
                        let after: Vec<u8> = frame
                            .spill
                            .iter()
                            .map(|b| unsafe { b.assume_init() })
                            .collect();
                        for binding in &target.bindings {
                            let expected = if let Some(input) =
                                source.bindings.iter().find(|b| b.value == binding.value)
                            {
                                read(&before, input.location, binding.value.bytes(), false)
                            } else {
                                match binding.value {
                                    GuestValue::General(i) => {
                                        (100 + u64::from(i)).to_le_bytes().to_vec()
                                    }
                                    GuestValue::Vector(2) => {
                                        initial.vector(2).unwrap().to_le_bytes().to_vec()
                                    }
                                    _ => unreachable!(),
                                }
                            };
                            assert_eq!(
                                read(&after, binding.location, binding.value.bytes(), true),
                                expected,
                                "{abi:?} {kind} {:?}",
                                binding.value
                            );
                        }
                        let expected_flags = (bits & (u32::from(source_bits) << 28))
                            | (initial.nzcv().bits() & !(u32::from(source_bits) << 28));
                        let actual_flags = if host_target {
                            captured_host_nzcv(abi, &after)
                        } else {
                            u32::from_le_bytes(
                                read(&after, spill(2080, 4), 4, true).try_into().unwrap(),
                            )
                        };
                        assert_eq!(
                            actual_flags & (u32::from(target_bits) << 28),
                            expected_flags & (u32::from(target_bits) << 28)
                        );
                        let mut expected = initial;
                        expected.general_register_storage_mut()[19] = u64::from_le_bytes(
                            read(&before, integer(2), 8, false).try_into().unwrap(),
                        );
                        expected.set_vector(
                            1,
                            u128::from_le_bytes(
                                read(&before, vector(0), 16, false).try_into().unwrap(),
                            ),
                        );
                        *expected.stack_pointer_storage_mut() = u64::from_le_bytes(
                            read(&before, spill(2200, 8), 8, false).try_into().unwrap(),
                        );
                        let mask = u32::from(source_bits & !target_bits) << 28;
                        expected.set_nzcv(Nzcv::from_bits(
                            (expected.nzcv().bits() & !mask) | (bits & mask),
                        ));
                        // Carried values and bits must remain stale in canonical
                        // homes: this rejects a hidden full-state writeback.
                        assert_eq!(state, expected);
                    }
                }
            }
        }
    }
}
