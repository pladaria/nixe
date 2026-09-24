//! Real published self-links exercise terminal checkpoints and lazy state.
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
        let cache = Cache::new().unwrap();
        let process = Arc::new(Lifetime::new(cache.clone()).unwrap());
        let mut reader = process.register().unwrap();
        let Request::Owner(claim) = reader.claim(key()).unwrap() else {
            panic!()
        };
        let compilation = Compilation::capture(claim, &memory).unwrap();
        Compiler::new(native_abi())
            .unwrap()
            .publish(compilation, &process, &cache, &memory)
            .unwrap();
        process.try_service_links().unwrap();
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
                let mut invocation = unsafe { reader.admit(&mut frame, key()) }.unwrap().unwrap();
                let address = invocation.payload().preferred().unwrap().canonical.get();
                let epoch = invocation.frame().execution_epoch;
                let frame = invocation.frame();
                if let Some(index) = request {
                    frame.poll_requests[index] = &pending;
                }
                frame.exclusive_load.address = 0x4321;
                frame.exclusive_load.value = [123, 456];
                frame.exclusive_load.bytes = 16;
                let returned = unsafe {
                    if active_fp {
                        frame.ensure_fp().unwrap();
                        crate::fp_env::tests::divide_by_zero();
                    }
                    crate::native::enter_protected(
                        frame,
                        std::ptr::null_mut(),
                        address as *const u8,
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
                assert_eq!(frame.execution_epoch, epoch);
                assert_eq!(frame.exclusive_load.address, 0x4321);
                assert_eq!(frame.exclusive_load.value, [123, 456]);
                assert_eq!(frame.exclusive_load.bytes, 16);
                drop(invocation);
            }
            assert_eq!(
                pending.load(Ordering::Acquire),
                u32::from(request.is_some())
            );
            assert_eq!(actual, expected, "conditional={conditional}, slice={slice}");
        }
    }
}
