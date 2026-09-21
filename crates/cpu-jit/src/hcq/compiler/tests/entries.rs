use super::*;
use crate::abi::{NativeFrame, NzcvLocation, PollBudget};
use crate::executable::{Cache, Tier, output::Output};
use crate::frontend::staging;
use crate::lifetime::unit::Entry;
use nixe_cpu::state::a64::{A64State, Nzcv};

#[test]
fn hcq_public_ingress_resolves_own_contracts_and_real_labels_on_both_targets() {
    let graph = graph(&[(0, &[ADDS, 0x1400000f]), (64, &[0x9a020020, RET])]);
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let (context, body) = emitted(&graph, &[0, block(&graph, 64)], abi);
        let code = context.compiled_code().unwrap();
        let mut bytes = code.code_buffer().to_vec();
        let entries: Vec<_> = body
            .prepare_entries(abi, code, &graph)
            .unwrap()
            .into_iter()
            .map(|entry| entry.append(&mut bytes).unwrap())
            .collect();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].key.pc.get(), 0);
        assert_eq!(entries[1].key.pc.get(), 64);
        assert_eq!(entries[0].contract.live_in.nzcv, 0);
        assert_eq!(entries[1].contract.live_in.nzcv, crate::analysis::NZCV);
        for (entry, selected) in entries.iter().zip(&body.ssa.entries) {
            entry.contract.validate().unwrap();
            assert!(
                code.buffer
                    .nixe_entries
                    .contains(&(selected.label, entry.fast_offset))
            );
            let canonical = entry.canonical_offset as usize;
            assert!(canonical >= code.code_buffer().len());
            assert_eq!(&bytes[canonical..canonical + 4], staging::landing(abi));
            let fast = entry.fast_offset as usize;
            assert_eq!(&bytes[fast..fast + 4], staging::landing(abi));
        }
        assert_ne!(entries[0].fast_offset, entries[1].fast_offset);
        assert_ne!(entries[0].canonical_offset, entries[1].canonical_offset);
    }
}

#[test]
fn hcq_entry_without_inputs_has_no_fabricated_flag_binding_or_canonical_loads() {
    let graph = graph(&[(0, &[0xd2800020, 0xd4200000])]);
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let (context, body) = emitted(&graph, &[0], abi);
        let code = context.compiled_code().unwrap();
        let mut bytes = code.code_buffer().to_vec();
        let entries: Vec<_> = body
            .prepare_entries(abi, code, &graph)
            .unwrap()
            .into_iter()
            .map(|entry| entry.append(&mut bytes).unwrap())
            .collect();
        let entry = &entries[0];
        assert!(entry.contract.live_in.is_empty());
        assert!(entry.contract.bindings.is_empty());
        assert_eq!(entry.contract.nzcv, NzcvLocation::Canonical);
        // Landing pad, alignment NOPs, direct branch slot; no home traffic.
        assert_eq!(bytes.len() - entry.canonical_offset as usize, 16);
    }
}

// Finite-body ABI and executed-path accounting through the real staging path.
// No loops, faults, publication or dispatch linking are exercised here.
fn finite_image(graph: &Graph, abi: HostAbi) -> (Output, Box<[Entry]>) {
    let selected: Vec<_> = (0..graph.blocks.len()).collect();
    assert!(
        !Analysis::build(graph, &selected)
            .backedges
            .iter()
            .flatten()
            .any(|&edge| edge)
    );
    let compiler = backend::Compiler::new(abi, 0x10000).unwrap();
    let mut context = Context::new();
    let body = compiler
        .emit(
            &mut context,
            &mut FunctionBuilderContext::new(),
            graph,
            &Analysis::build(graph, &selected),
            &selected,
        )
        .unwrap();
    assert!(body.faults.is_empty());
    let version = CodeVersion::new(1).unwrap();
    let staged = compiler.finish(&mut context, body, graph, version).unwrap();
    (staged.output, staged.entries)
}

#[test]
fn hcq_selected_labels_execute_shared_diamonds_and_fp_continuations_like_interpreter() {
    let abi = if cfg!(target_arch = "x86_64") {
        HostAbi::X86_64
    } else {
        HostAbi::Aarch64
    };
    crate::native::check_host().unwrap();
    let graphs = [
        graph(&[
            (0, &[0x54000080]),
            (4, &[ADDS, 0x14000005]),
            (16, &[SUBS, 0xd503201f, 0x14000001]),
            (28, &[0x9a020020, 0xd4200000]),
        ]),
        graph(&[
            (0, &[0x1e222820, ADDS, 0x1400000e]),
            (64, &[0x1e222820, 0x9a020020, 0xd4200000]),
        ]),
    ];
    for graph in graphs {
        let (output, entries) = finite_image(&graph, abi);
        let owner = Cache::new()
            .unwrap()
            .install(output, Tier::Hcq, |_| None)
            .unwrap();
        for entry in entries.iter() {
            for nzcv in [0, Nzcv::Z | Nzcv::C, Nzcv::N | Nzcv::V] {
                let mut initial = A64State::default();
                initial.set_pc(entry.key.pc.get());
                initial.general_register_storage_mut()[0] = 0xfeed;
                initial.general_register_storage_mut()[1] = u64::MAX;
                initial.general_register_storage_mut()[2] = 4;
                initial.set_vector(1, u128::from(1.0f32.to_bits()));
                initial.set_vector(2, u128::from(2.0f32.to_bits()));
                initial.set_nzcv(Nzcv::from_bits(nzcv));
                initial.set_fpsr(1 << 27);
                let mut expected = initial.clone();
                let mut steps = 0;
                loop {
                    let word = graph
                        .instructions
                        .iter()
                        .find(|word| word.instruction.key.block_key().pc.get() == expected.pc())
                        .unwrap();
                    if word.instruction.bits == 0xd4200000 {
                        break;
                    }
                    nixe_cpu_interpreter::execute_one(
                        &entry.key.platform,
                        &mut expected,
                        word.instruction.bits,
                    )
                    .unwrap();
                    steps += 1;
                    assert!(steps <= graph.instructions.len());
                }
                let mut actual = initial;
                {
                    let mut frame =
                        NativeFrame::new(&mut actual, PollBudget::new(4096, 1000).unwrap());
                    // The test exclusively owns the complete unlinked allocation.
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
                    assert_eq!(result.reason, NativeExitReason::Architectural);
                    assert_eq!(frame.budget.sample_remaining, 4096 - steps as i64);
                    assert_eq!(frame.budget.slice_remaining, 1000 - steps as i64);
                    assert_eq!(frame.host_fp.active, 0);
                    assert_eq!(frame.host_fp.saved, 0);
                    frame.execution_epoch = 0;
                }
                assert_eq!(actual, expected, "entry {:?}", entry.key);
            }
        }
    }
}
