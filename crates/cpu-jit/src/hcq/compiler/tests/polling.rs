use super::*;
use crate::abi::{NativeFrame, PollBudget};
use crate::executable::{Cache, Tier};
use nixe_cpu::state::a64::{A64State, Nzcv};
use std::sync::atomic::AtomicU32;

fn host() -> HostAbi {
    if cfg!(target_arch = "x86_64") {
        HostAbi::X86_64
    } else {
        HostAbi::Aarch64
    }
}

fn staged(graph: &Graph, entries: &[usize], abi: HostAbi) -> stage::Staged {
    let compiler = backend::Compiler::new(abi, 0x10000).unwrap();
    let mut context = Context::new();
    let analysis = Analysis::build(graph, entries);
    let body = compiler
        .emit(
            &mut context,
            &mut FunctionBuilderContext::new(),
            graph,
            &analysis,
            entries,
        )
        .unwrap();
    assert_eq!(
        body.polls.len(),
        analysis.backedges.iter().flatten().filter(|&&v| v).count()
    );
    compiler
        .finish(&mut context, body, graph, CodeVersion::new(1).unwrap())
        .unwrap()
}

#[test]
fn hcq_internal_cycles_resume_samples_and_exit_with_exact_post_state() {
    crate::native::check_host().unwrap();
    for conditional in [false, true] {
        let words = [
            0xba1f0021,
            0xf1000400,
            if conditional { 0x54ffffc1 } else { 0x17fffffe },
        ];
        let graph = if conditional {
            graph(&[(0, &words), (12, &[0xd4200000])])
        } else {
            graph(&[(0, &words)])
        };
        let image = staged(&graph, &[0], host());
        let entry = image.entries[0].canonical_offset;
        let owner = Cache::new()
            .unwrap()
            .install(image.output, Tier::Hcq, |_| None)
            .unwrap();
        for (sample, slice, request) in [
            (4096, 1, None),
            (4096, 32, None),
            (1, 8192, None),
            (2, 8192, None),
            (3, 3, None),
            (3, 8192, Some(0)),
            (3, 8192, Some(1)),
            (3, 8192, Some(2)),
        ] {
            let budget = PollBudget::new(sample, slice).unwrap();
            let steps = (if request.is_some() {
                budget.armed_span
            } else {
                slice
            } + 2)
                / 3
                * 3;
            let mut state = A64State::default();
            state.general_register_storage_mut()[0] = 10000;
            state.general_register_storage_mut()[1] = u64::MAX - 2;
            state.set_nzcv(Nzcv::from_bits(0xb0000000));
            let mut expected = state.clone();
            for _ in 0..steps / 3 {
                for word in words {
                    nixe_cpu_interpreter::execute_one(
                        &graph.blocks[0].key.platform,
                        &mut expected,
                        word,
                    )
                    .unwrap();
                }
            }
            let mut expected_budget = budget;
            expected_budget
                .reconcile(budget.armed_span - steps, request.is_some())
                .unwrap();
            let stop = AtomicU32::new(1);
            let mut frame = NativeFrame::new(&mut state, budget);
            if let Some(index) = request {
                frame.poll_requests[index] = &stop;
            }
            frame.execution_epoch = 1;
            frame.exclusive_load.address = 0x4321;
            frame.exclusive_load.value = [123, 456];
            frame.exclusive_load.bytes = 16;
            let result = unsafe {
                frame.begin_fp();
                crate::native::enter_protected(
                    &mut frame,
                    std::ptr::null_mut(),
                    (owner.allocation.address() + entry as usize) as *const u8,
                )
            }
            .unwrap();
            assert_eq!(
                result.reason,
                if request.is_some() {
                    NativeExitReason::Control
                } else {
                    NativeExitReason::BudgetExhausted
                }
            );
            assert_eq!(
                frame.budget.slice_remaining,
                expected_budget.slice_remaining
            );
            assert_eq!(
                frame.budget.sample_remaining,
                expected_budget.sample_remaining
            );
            assert_eq!(frame.exclusive_load.address, 0x4321);
            assert_eq!(frame.exclusive_load.value, [123, 456]);
            assert_eq!(frame.exclusive_load.bytes, 16);
            frame.execution_epoch = 0;
            assert_eq!(state, expected);
        }
    }
}

#[test]
fn hcq_irreducible_and_multi_entry_checks_match_the_shared_flow_proof() {
    let graph = graph(&[
        (0, &[0x54000080]),
        (4, &[ADDS, 0x14000002]),
        (16, &[0x54000020]),
        (20, &[SUBS, 0x17fffffb]),
    ]);
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        for entries in [vec![0], vec![block(&graph, 4), block(&graph, 16)]] {
            let image = staged(&graph, &entries, abi);
            let count = Analysis::build(&graph, &entries)
                .backedges
                .iter()
                .flatten()
                .filter(|&&b| b)
                .count();
            assert!(count > 0);
            assert_eq!(
                image
                    .states
                    .iter()
                    .filter(|s| s.exit.is_some() && s.transfer.is_none())
                    .count(),
                count
            );
            if abi != host() {
                continue;
            }
            let owner = Cache::new()
                .unwrap()
                .install(image.output, Tier::Hcq, |_| None)
                .unwrap();
            let analysis = Analysis::build(&graph, &entries);
            for entry in image.entries.iter() {
                for slice in [31, 8192] {
                    let mut actual = A64State::default();
                    actual.set_pc(entry.key.pc.get());
                    actual.general_register_storage_mut()[1] = 5;
                    let mut expected = actual.clone();
                    let mut completed = 0;
                    loop {
                        let index = block(&graph, expected.pc());
                        let source = &graph.blocks[index];
                        for ordinal in source.instructions.clone() {
                            nixe_cpu_interpreter::execute_one(
                                &source.key.platform,
                                &mut expected,
                                graph.instructions[ordinal].instruction.bits,
                            )
                            .unwrap();
                            completed += 1;
                        }
                        let targets = match source.exit {
                            Exit::Jump(t) | Exit::Fallthrough(t) => [Some(t), None],
                            Exit::Conditional { fallthrough, taken } => {
                                [Some(fallthrough), Some(taken)]
                            }
                            _ => unreachable!(),
                        };
                        let checked = targets.into_iter().zip(analysis.backedges[index]).any(|(t, check)| {
                            check && matches!(t, Some(Target::Internal(i)) if graph.blocks[i].key.pc.get() == expected.pc())
                        });
                        if checked && completed >= slice {
                            break;
                        }
                        assert!(completed < slice + graph.instructions.len() as i64);
                    }
                    {
                        let mut frame =
                            NativeFrame::new(&mut actual, PollBudget::new(1, slice).unwrap());
                        frame.execution_epoch = 1;
                        let result = unsafe {
                            frame.begin_fp();
                            crate::native::enter_protected(
                                &mut frame,
                                std::ptr::null_mut(),
                                (owner.allocation.address() + entry.canonical_offset as usize)
                                    as *const u8,
                            )
                        }
                        .unwrap();
                        assert_eq!(result.reason, NativeExitReason::BudgetExhausted);
                        assert_eq!(frame.budget.slice_remaining, slice - completed);
                        frame.execution_epoch = 0;
                    }
                    assert_eq!(actual, expected);
                }
            }
        }
    }
}

#[derive(Default)]
struct Observation {
    calls: usize,
    source: usize,
    version: u64,
    map: u32,
    destination: u64,
    fail: bool,
}

unsafe extern "C" fn observe(
    context: *mut libc::c_void,
    frame: *mut libc::c_void,
    source: usize,
    version: u64,
    map: u32,
) -> u32 {
    let frame = unsafe { &mut *frame.cast::<NativeFrame<'_>>() };
    let pause = unsafe { frame.host_fp.pause_observation() };
    let observation = unsafe { &mut *context.cast::<Observation>() };
    observation.calls += 1;
    observation.source = source;
    observation.version = version;
    observation.map = map;
    observation.destination = unsafe {
        frame
            .spill
            .as_ptr()
            .byte_add(crate::native::observation::DESTINATION as usize)
            .cast::<u64>()
            .read()
    };
    // Destroy every System-ABI volatile to exercise physical preservation,
    // including vectors with no guest write on the incoming edge.
    #[cfg(target_arch = "x86_64")]
    unsafe {
        core::arch::asm!(
            ".irp reg,rax,rcx,rdx,rsi,rdi,r8,r9,r10,r11",
            "xor \\reg, \\reg",
            ".endr",
            ".irp n,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15",
            "pxor xmm\\n, xmm\\n",
            ".endr",
            clobber_abi("C"),
            options(nostack),
        );
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        core::arch::asm!(
            ".irp n,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17",
            "mov x\\n, xzr",
            ".endr",
            ".irp n,0,1,2,3,4,5,6,7,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31",
            r"movi v\n\().16b, #0",
            ".endr",
            ".irp n,8,9,10,11,12,13,14,15",
            r"mov v\n\().d[1], xzr",
            ".endr",
            "cmp x0,x0",
            clobber_abi("C"),
            options(nostack),
        );
    }
    if observation.fail {
        return 0;
    }
    unsafe {
        pause.resume();
    }
    1
}

#[test]
fn hcq_internal_sample_callback_preserves_live_ssa_fp_and_precise_failure_state() {
    crate::native::check_host().unwrap();
    let words = [0x1e222800, 0x8b040063, 0xf1000400, 0x54ffffa1]; // FADD; ADD; SUBS; B.NE 0
    let graph = graph(&[(0, &words), (16, &[0xd4200000])]);
    let image = staged(&graph, &[0], host());
    let entry = image.entries[0].canonical_offset;
    let poll_index = image
        .states
        .iter()
        .position(|s| s.exit.is_some() && s.transfer.is_none())
        .unwrap();
    let owner = Cache::new()
        .unwrap()
        .install(image.output, Tier::Hcq, |_| None)
        .unwrap();
    for fail in [false, true] {
        let mut observation = Observation {
            fail,
            ..Default::default()
        };
        let mut actual = A64State::default();
        actual.general_register_storage_mut()[0] = 2000;
        actual.general_register_storage_mut()[4] = 13;
        actual.set_vector(2, u128::from(0.25f32.to_bits()));
        actual.set_fpsr(1 << 27);
        let mut expected = actual.clone();
        let iterations = if fail { 1 } else { 2000 };
        for _ in 0..iterations {
            for word in words {
                nixe_cpu_interpreter::execute_one(
                    &graph.blocks[0].key.platform,
                    &mut expected,
                    word,
                )
                .unwrap();
            }
        }
        {
            let mut frame = NativeFrame::new(&mut actual, PollBudget::new(1, 10000).unwrap());
            frame.execution_epoch = 1;
            frame.dispatch_context = (&mut observation as *mut Observation).cast();
            frame.sample_observer = Some(observe);
            let result = unsafe {
                frame.begin_fp();
                crate::native::enter_protected(
                    &mut frame,
                    std::ptr::null_mut(),
                    (owner.allocation.address() + entry as usize) as *const u8,
                )
            }
            .unwrap();
            assert_eq!(
                result.reason,
                if fail {
                    NativeExitReason::Control
                } else {
                    NativeExitReason::Architectural
                }
            );
            assert_eq!(frame.budget.slice_remaining, 10000 - iterations * 4);
            assert_eq!(frame.host_fp.active, 0);
            assert_eq!(frame.host_fp.saved, 0);
            frame.execution_epoch = 0;
        }
        assert_eq!(actual, expected);
        assert_eq!(observation.calls, if fail { 1 } else { 2 });
        assert_eq!(observation.version, 1);
        assert_eq!(observation.map as usize, poll_index);
        assert_eq!(observation.destination, 0);
        assert!(observation.source >= owner.allocation.address());
        assert!(observation.source < owner.allocation.address() + owner.allocation.len());
    }
}
