use super::super::observation::{DESTINATION, emit};
use super::*;
use canonical::{complete, native, pattern};

#[test]
fn sampling_callback_emits_one_shared_register_restore() {
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let (source, _) = complete(abi);
        let destination = integer(0);
        let restore = emit(&source, destination).unwrap().restore;
        let (code, continuations) =
            super::super::observation::emit_callback(&source, destination, false).unwrap();
        assert!(!restore.is_empty());
        assert_eq!(
            code.windows(restore.len())
                .filter(|bytes| *bytes == restore.as_slice())
                .count(),
            1
        );
        assert!(continuations[0] < continuations[1]);
    }
}

#[test]
fn observation_saves_aliases_once_and_only_volatile_live_registers() {
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let nonvolatile = if abi == HostAbi::X86_64 { 3 } else { 22 };
        let (source, _) = contracts(
            abi,
            &[
                (GuestValue::General(0), integer(0), integer(0)),
                (GuestValue::General(1), integer(0), integer(0)),
                (
                    GuestValue::General(2),
                    integer(nonvolatile),
                    integer(nonvolatile),
                ),
                (GuestValue::General(3), vector(8), vector(8)),
                (GuestValue::Vector(0), vector(8), vector(8)),
                (GuestValue::General(4), spill(2048, 8), spill(2048, 8)),
                (
                    GuestValue::General(5),
                    ValueLocation::constant(17),
                    integer(1),
                ),
            ],
        );
        let preservation = emit(&source, integer(2)).unwrap();
        let mut expected = Emitter::new(abi);
        expected.memory(true, RegisterClass::Integer, 0, 0, 8);
        expected.memory(true, RegisterClass::Integer, 2, 8, 8);
        expected.memory(true, RegisterClass::Vector, 8, 16, 16);
        assert_eq!(preservation.restore, expected.finish());
        // The scalar half of d8 is preserved by AAPCS64, but not by SysV.
        let (scalar, _) = contracts(abi, &[(GuestValue::General(0), vector(8), vector(8))]);
        let scalar = emit(&scalar, ValueLocation::constant(0)).unwrap();
        assert_eq!(scalar.restore.is_empty(), abi == HostAbi::Aarch64);
    }
}

#[test]
fn observation_keeps_unbound_lazy_operands_and_rejects_reserved_destinations() {
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let (mut source, _) = contracts(abi, &[]);
        source.live.nzcv = NZCV;
        source.dirty_live.nzcv = NZCV;
        source.nzcv = NzcvLocation::Deferred(LazyFlags::Conditional {
            predicate: integer(0),
            when_true: Box::new(LazyFlags::AddCarry {
                lhs: integer(1),
                rhs: integer(2),
                carry: vector(0),
                result: integer(6),
                width: 64,
            }),
            when_false: 0,
        });
        let preservation = emit(&source, integer(7)).unwrap();
        let mut expected = Emitter::new(abi);
        for (slot, register) in [0, 1, 2, 6, 7].into_iter().enumerate() {
            expected.memory(true, RegisterClass::Integer, register, slot as u32 * 8, 8);
        }
        expected.memory(true, RegisterClass::Vector, 0, 40, 8);
        assert_eq!(preservation.restore, expected.finish());
        assert!(emit(&source, integer(abi.reserved().link_scratch[0])).is_err());
        assert!(emit(&source, spill(0, 8)).is_err());
    }
}

#[test]
fn observation_survives_real_system_call_without_canonical_roundtrip() {
    let _restore = crate::fp_env::tests::RestoreHost::new();
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        for flags in 0..4 {
            let (mut source, mut entry) = complete(abi);
            match flags {
                1 => {
                    source.nzcv = NzcvLocation::Host {
                        carry_inverted: true,
                    };
                    entry.nzcv = source.nzcv.clone();
                }
                2 => {
                    // Leave the incoming packed NZCV alone: the output uses a
                    // live recipe, including operands not dirty in this unit.
                    source.nzcv = NzcvLocation::Deferred(LazyFlags::Conditional {
                        predicate: integer(0),
                        when_true: Box::new(LazyFlags::AddCarry {
                            lhs: integer(1),
                            rhs: integer(2),
                            carry: integer(6),
                            result: integer(7),
                            width: 64,
                        }),
                        when_false: 5,
                    });
                }
                3 => {
                    std::sync::Arc::make_mut(&mut source.bindings)[0].location = spill(3304, 8);
                    std::sync::Arc::make_mut(&mut entry.bindings)[0].location = spill(3304, 8);
                    source.nzcv = NzcvLocation::Packed(integer(0));
                    entry.nzcv = source.nzcv.clone();
                }
                _ => {}
            }
            let destinations = [
                integer(1),
                vector(8),
                spill(3200, 8),
                ValueLocation::constant(0xabc),
            ];
            for destination in destinations {
                // Save clean values too; the test continuation will consume all
                // mapped values, whereas this observation dirties none of them.
                let mut observation = source.clone();
                observation.dirty_live = StateSet::default();
                let preservation = emit(&observation, destination).unwrap();
                if !native(abi) {
                    continue;
                }
                for fp in [false, true] {
                    let mut results = Vec::new();
                    for observe in [false, true] {
                        let mut code = Vec::new();
                        if observe {
                            // Repeat to prove the restored image remains usable
                            // without reloading any canonical architectural home.
                            for _ in 0..3 {
                                code.extend_from_slice(&preservation.save);
                                let mut call = Emitter::new(abi);
                                call.copy(Copy {
                                    source: integer(abi.reserved().frame),
                                    destination: integer(if abi == HostAbi::X86_64 {
                                        7
                                    } else {
                                        0
                                    }),
                                    bytes: 8,
                                });
                                let scratch = abi.reserved().link_scratch[0];
                                call.constant(scratch, observer as *const () as u64, 8);
                                if abi == HostAbi::X86_64 {
                                    call.code.extend([0x41, 0xff, 0xd3]); // CALL R11
                                } else {
                                    call.word(0xd63f0200); // BLR X16
                                }
                                code.extend(call.finish());
                                code.extend_from_slice(&preservation.restore);
                            }
                        }
                        let owner = published::Published::new(|process, cache| {
                            published::synthetic(
                                process.begin_unit(crate::executable::Tier::Lcq).unwrap(),
                                cache,
                                entry.clone(),
                                source.clone(),
                                (code, None),
                                destination,
                                NativeExitReason::Dispatch,
                            )
                        });
                        let mut reader = owner.process.register().unwrap();
                        let (mut state, _) = pattern(&entry);
                        state.set_fpcr(0);
                        state.set_fpsr(1 << 27);
                        let mut observed = [0u64; 2];
                        {
                            let mut frame =
                                NativeFrame::new(&mut state, PollBudget::new(77, 1000).unwrap());
                            frame.runtime = observed.as_mut_ptr().cast();
                            let mut invocation =
                                unsafe { reader.admit(&mut frame, published::key()) }
                                    .unwrap()
                                    .unwrap();
                            let address = invocation.payload().preferred().unwrap().canonical.get()
                                as *const u8;
                            let epoch = invocation.frame().execution_epoch;
                            let frame = invocation.frame();
                            unsafe {
                                if fp {
                                    frame.ensure_fp().unwrap();
                                    crate::fp_env::tests::divide_by_zero();
                                }
                                enter_protected(frame, std::ptr::dangling_mut(), address).unwrap();
                            }
                            assert_eq!(frame.execution_epoch, epoch);
                            assert_eq!(frame.budget.sample_remaining, 77);
                            assert_eq!(frame.budget.slice_remaining, 1000);
                        }
                        assert_eq!(observed[1], if observe { 3 } else { 0 });
                        if observe {
                            assert_eq!(observed[0], state.pc());
                        }
                        assert_eq!(state.fpsr(), (1 << 27) | if fp { 2 } else { 0 });
                        results.push(state);
                        drop(reader);
                        owner.shutdown();
                    }
                    let mut actual = results.pop().unwrap();
                    let mut expected = results.pop().unwrap();
                    assert_eq!(
                        actual.general_register_storage_mut(),
                        expected.general_register_storage_mut()
                    );
                    assert_eq!(
                        actual.vector_register_storage_mut(),
                        expected.vector_register_storage_mut()
                    );
                    assert_eq!(
                        actual.stack_pointer_storage_mut(),
                        expected.stack_pointer_storage_mut()
                    );
                    assert_eq!(actual.nzcv(), expected.nzcv());
                    assert_eq!(actual.pc(), expected.pc());
                    assert_eq!(actual.fpcr(), expected.fpcr());
                    assert_eq!(actual.fpsr(), expected.fpsr());
                }
            }
        }
    }
}

unsafe extern "C" fn observer(frame: *mut NativeFrame<'_>) {
    let frame = unsafe { &mut *frame };
    let pause = unsafe { frame.host_fp.pause_observation() };
    let target = unsafe {
        frame
            .spill
            .as_ptr()
            .byte_add(DESTINATION as usize)
            .cast::<u64>()
            .read()
    };
    let observed = frame.runtime.cast::<u64>();
    unsafe {
        observed.write(target);
        *observed.add(1) += 1;
        // Only the caller environment is active in this System-ABI callback.
        clobber();
        pause.resume();
    }
}

#[cfg(target_arch = "x86_64")]
#[unsafe(naked)]
unsafe extern "C" fn clobber() {
    core::arch::naked_asm!(
        ".irp reg,rax,rcx,rdx,rsi,rdi,r8,r9,r10,r11",
        "xor \\reg, \\reg",
        ".endr",
        ".irp n,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15",
        "pxor xmm\\n, xmm\\n",
        ".endr",
        "ret",
    );
}

#[cfg(target_arch = "aarch64")]
#[unsafe(naked)]
unsafe extern "C" fn clobber() {
    core::arch::naked_asm!(
        ".irp n,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17",
        "mov x\\n, xzr",
        ".endr",
        ".irp n,0,1,2,3,4,5,6,7,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31",
        r"movi v\n\().16b, #0",
        ".endr",
        ".irp n,8,9,10,11,12,13,14,15",
        r"mov v\n\().d[1], xzr",
        ".endr",
        "cmp x0, x0",
        "ret",
    );
}
