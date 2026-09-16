//! Test-owned links isolate terminal checkpoint behavior from live patching.
//! The complete executable allocation stays owned through the gateway.
use super::*;
use std::sync::atomic::{AtomicU32, Ordering};

#[test]
fn native_loops_charge_once_and_preserve_lazy_flags_at_the_deadline() {
    crate::native::check_host().unwrap();
    for conditional in [false, true] {
        let words = [
            0xba1f_0021, // ADCS X1,X1,XZR: consume the preceding iteration's C.
            0xf100_0400, // SUBS X0,X0,#1.
            if conditional {
                0x54ff_ffc1
            } else {
                0x17ff_fffe
            }, // B.NE/B PC.
        ];
        let memory = memory(&words);
        let fragment = Fragment::capture(&memory, key()).unwrap();
        let abi = native_abi();
        let lowered = Compiler::new(abi)
            .unwrap()
            .lower(&fragment, CodeVersion::new(1).unwrap())
            .unwrap();
        let mut bytes = lowered.output.bytes.to_vec();
        let source = lowered
            .states
            .iter()
            .find(|record| {
                record.exit.unwrap().kind
                    == if conditional {
                        EdgeKind::Taken
                    } else {
                        EdgeKind::Static
                    }
            })
            .unwrap();
        let map = lowered
            .output
            .metadata
            .states
            .iter()
            .find(|map| !map.entry && map.offset == source.native_offset)
            .unwrap();
        // This particular loop carries every writable input. Other dirty flags
        // are overwritten before observation; this is not a general bridge.
        let bridge = crate::native::emit_fast_transfer(&source.state, &lowered.entry).unwrap();
        let bridge_start = append(&mut bytes, &bridge);
        let jump = bytes.len().next_multiple_of(8);
        while bytes.len() < jump {
            bytes.extend(nop(abi));
        }
        bytes.resize(jump + usize::from(map.patch_bytes), 0);
        let mut back = map.clone();
        back.offset = jump as u32;
        back.poll = None;
        back.patch_exit(&mut bytes, 0, u64::from(lowered.fast))
            .unwrap();
        map.patch_exit(&mut bytes, 0, bridge_start as u64).unwrap();
        // Leave the production cold poll intact. Sample deadlines resume this
        // hot edge, while slice exhaustion/control must leave the native loop.
        let output = Output {
            bytes: bytes.into_boxed_slice(),
            ..lowered.output
        };
        let cache = Cache::new().unwrap();
        let owner = cache.install(output, Tier::Lcq, |_| None).unwrap();
        let cases = [1, 2, 3, 4, 32, 33, 34, 4096, 8192]
            .into_iter()
            .map(|slice| (4096, slice, None))
            .chain([(1, 8192, None), (2, 8192, None), (7, 8192, None)])
            .chain((0..3).map(|request| (3, 8192, Some(request))));
        for (sample, slice, request) in cases {
            let active_fp = sample < 4096;
            let budget = PollBudget::new(sample, slice).unwrap();
            let stop = if request.is_some() {
                budget.armed_span
            } else {
                slice
            };
            let iterations = (stop as u64)
                .div_ceil(3)
                .min(if conditional { 11 } else { u64::MAX });
            let completed = (iterations * 3) as i64;
            let mut initial = integer::initial_state();
            initial.general_register_storage_mut()[0] = 11;
            initial.general_register_storage_mut()[1] = u64::MAX - 2;
            initial.set_nzcv(Nzcv::from_bits(0xb000_0000));
            let mut expected = initial.clone();
            let mut expected_budget = budget;
            let mut remaining = budget.armed_span;
            for _ in 0..iterations {
                for word in words {
                    nixe_cpu_interpreter::execute_one(
                        &TargetPlatform::Switch1,
                        &mut expected,
                        word,
                    )
                    .unwrap();
                }
                remaining -= 3;
                if remaining <= 0
                    && request.is_none()
                    && expected_budget.slice_remaining - (expected_budget.armed_span - remaining)
                        > 0
                {
                    assert!(expected_budget.reconcile(remaining, false).unwrap().sample);
                    remaining = expected_budget.armed_span;
                }
            }
            let expected_poll = expected_budget
                .reconcile(remaining, request.is_some())
                .unwrap();
            if active_fp {
                expected.set_fpsr(expected.fpsr() | 2);
            }
            let mut actual = initial;
            let pending = AtomicU32::new(0);
            if request.is_some() {
                pending.store(1, Ordering::Release);
            }
            {
                let mut frame = NativeFrame::new(&mut actual, budget);
                if let Some(index) = request {
                    frame.poll_requests[index] = &pending;
                }
                frame.exclusive_load.address = 0x4321;
                frame.exclusive_load.value = [123, 456];
                frame.exclusive_load.bytes = 16;
                frame.execution_epoch = 1;
                let returned = unsafe {
                    frame.begin_fp();
                    if active_fp {
                        frame.ensure_fp().unwrap();
                        crate::fp_env::tests::divide_by_zero();
                    }
                    crate::native::enter_protected(
                        &mut frame,
                        std::ptr::null_mut(),
                        (owner.allocation.address() + lowered.canonical as usize) as *const u8,
                    )
                }
                .unwrap();
                assert_eq!(
                    returned.reason,
                    if request.is_some() {
                        NativeExitReason::Control
                    } else {
                        NativeExitReason::Dispatch
                    }
                );
                assert_eq!(returned.poll.exhausted, completed >= slice);
                assert_eq!(returned.poll, expected_poll);
                assert_eq!(frame.budget.slice_remaining, slice - completed);
                assert_eq!(
                    frame.budget.sample_remaining,
                    expected_budget.sample_remaining
                );
                assert_eq!(frame.budget.armed_span, expected_budget.armed_span);
                assert_eq!(frame.exit_source_version, 1);
                assert_eq!(frame.host_fp.saved, 0);
                assert_eq!(frame.execution_epoch, 1);
                assert_eq!(frame.exclusive_load.address, 0x4321);
                assert_eq!(frame.exclusive_load.value, [123, 456]);
                assert_eq!(frame.exclusive_load.bytes, 16);
                frame.execution_epoch = 0;
            }
            assert_eq!(
                pending.load(Ordering::Acquire),
                u32::from(request.is_some())
            );
            assert_eq!(actual, expected, "conditional={conditional}, slice={slice}");
        }
    }
}
