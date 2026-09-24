use super::*;
use crate::{ReturnStack, rsb::Continuation};

#[test]
fn subtraction_flags_cross_tiers_static_pic_and_return_links() {
    crate::native::check_host().unwrap();
    // Each producer is compiled once as LCQ and once as HCQ. ADC consumes C at
    // both destinations, and MRS observes every final flag after the second link.
    for hcq_source in [false, true] {
        for wide in [false, true] {
            for branch in [0x1400_0007, 0xd61f_00e0, 0xd63f_00e0, 0xd65f_03c0] {
                // B 0x1020 / BR X7 / BLR X7 / RET X30.
                let source = [if wide { 0xeb01_001f } else { 0x6b01_001f }, branch];
                let middle = [
                    0x9a1f_0042,                                  // ADC X2,X2,XZR
                    if wide { 0xeb04_007f } else { 0x6b04_007f }, // CMP X3/W3,X4/W4
                    0x1400_0006,                                  // B 0x1040
                ];
                let target = [0x9a1f_03e5, 0xd53b_4206, 0xd420_0000]; // ADC; MRS NZCV; BRK
                let (hcq_pc, hcq_words, lcq_pc, lcq_words) = if hcq_source {
                    (0x1000, source.as_slice(), 0x1020, middle.as_slice())
                } else {
                    (0x1020, middle.as_slice(), 0x1000, source.as_slice())
                };
                let graph = graph(&[(hcq_pc, hcq_words)]);
                let (mut reader, memory) =
                    fixture(&graph, &[0], &[(lcq_pc, lcq_words), (0x1040, &target)]);
                let mut worker = WorkerFaultContext::register().unwrap();
                let mut prediction = ReturnStack::default();
                if branch == 0xd65f_03c0 {
                    prediction.entries[0] = Continuation::from(key(0x1020));
                    prediction.head = 1;
                    prediction.depth = 1;
                }
                for (lhs, rhs) in [
                    (0, 1),
                    (42, 42),
                    (1_u64 << 63, 1),
                    (0xfeed_beef_8000_0000, 1),
                ] {
                    let mut initial = A64State::default();
                    initial.set_pc(0x1000);
                    let regs = initial.general_register_storage_mut();
                    regs[0] = lhs;
                    regs[1] = rhs;
                    regs[2] = 100;
                    regs[3] = rhs;
                    regs[4] = lhs;
                    regs[7] = 0x1020;
                    regs[30] = 0x1020;
                    let mut expected = initial.clone();
                    if branch != 0x1400_0007 && lhs == 0 {
                        // The first PIC probe cannot resolve without a callback.
                        // Its canonical escape must retain the producer's flags.
                        let mut missing = initial.clone();
                        let mut prefix = initial.clone();
                        for word in source {
                            nixe_cpu_interpreter::execute_one(
                                &key(0x1000).platform,
                                &mut prefix,
                                word,
                            )
                            .unwrap();
                        }
                        let mut returns = prediction.clone();
                        let mut frame =
                            NativeFrame::new(&mut missing, PollBudget::new(4096, 100).unwrap())
                                .with_return_stack(&mut returns);
                        let mut admitted = unsafe { reader.admit(&mut frame, key(0x1000)) }
                            .unwrap()
                            .unwrap();
                        let address = admitted.payload().preferred().unwrap().canonical.get();
                        let returned = unsafe {
                            crate::native::enter_protected(
                                admitted.frame(),
                                std::ptr::null_mut(),
                                address as *const u8,
                            )
                        }
                        .unwrap();
                        assert_eq!(returned.reason, NativeExitReason::Dispatch);
                        assert_eq!(admitted.frame().budget.slice_remaining, 98);
                        drop(admitted);
                        assert_eq!(missing, prefix);
                        assert_eq!(returns.depth, u32::from(branch == 0xd63f_00e0));
                    }
                    for word in source
                        .into_iter()
                        .chain(middle)
                        .chain(target[..2].iter().copied())
                    {
                        nixe_cpu_interpreter::execute_one(
                            &key(0x1000).platform,
                            &mut expected,
                            word,
                        )
                        .unwrap();
                    }
                    // The first indirect invocation resolves a real miss and
                    // installs the PIC. Check its state too, not just the hit.
                    let mut cold = initial.clone();
                    let mut returns = prediction.clone();
                    let mut frame =
                        NativeFrame::new(&mut cold, PollBudget::new(4096, 100).unwrap())
                            .with_return_stack(&mut returns);
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
                    let invocation::Exit::Native { returned, .. } = exit else {
                        panic!()
                    };
                    assert_eq!(returned.reason, NativeExitReason::Architectural);
                    assert_eq!(frame.budget.slice_remaining, 93);
                    assert_eq!(cold, expected);
                    assert_eq!(returns.depth, u32::from(branch == 0xd63f_00e0));

                    for sample in [4096, 1] {
                        let mut state = initial.clone();
                        let mut returns = prediction.clone();
                        let mut frame =
                            NativeFrame::new(&mut state, PollBudget::new(sample, 100).unwrap())
                                .with_return_stack(&mut returns);
                        let mut admitted = unsafe { reader.admit(&mut frame, key(0x1000)) }
                            .unwrap()
                            .unwrap();
                        let address = admitted.payload().preferred().unwrap().canonical.get();
                        // No resolver, no faultable memory, no Rust dispatch:
                        // a broken static/PIC/RSB link cannot hide behind fallback.
                        let returned = unsafe {
                            crate::native::enter_protected(
                                admitted.frame(),
                                std::ptr::null_mut(),
                                address as *const u8,
                            )
                        }
                        .unwrap();
                        assert_eq!(returned.reason, NativeExitReason::Architectural);
                        assert_eq!(admitted.frame().budget.slice_remaining, 93);
                        drop(admitted);
                        assert_eq!(
                            state, expected,
                            "HCQ source={hcq_source}, wide={wide}, branch={branch:x}, sample={sample}"
                        );
                        assert_eq!(returns.depth, u32::from(branch == 0xd63f_00e0));
                    }
                }
            }
        }
    }
}
