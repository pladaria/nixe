use super::*;
use crate::hcq::flow::tests::graph;
use cranelift_codegen::settings::{OptLevel, RegallocAlgorithm};

#[test]
fn hcq_repeated_guest_accesses_keep_distinct_fault_sites() {
    for words in [
        [0xf9400020, 0xf9400022, 0xd65f03c0], // LDR X0,[X1]; LDR X2,[X1]
        [0xf9000020, 0xf9400022, 0xd65f03c0], // STR X0,[X1]; LDR X2,[X1]
        [0xf9000020, 0xf9000020, 0xd65f03c0], // repeated STR X0,[X1]
        [0xf9000020, 0xf9000022, 0xd65f03c0], // overwritten STR X0,[X1]
    ] {
        let graph = graph(&[(0, &words)]);
        let analysis = Analysis::build(&graph, &[0]);
        for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
            let compiler = Compiler::new(abi, 0x10000).unwrap();
            let mut context = Context::new();
            let body = compiler
                .emit(
                    &mut context,
                    &mut FunctionBuilderContext::new(),
                    &graph,
                    &analysis,
                    &[0],
                )
                .unwrap();
            let staged = compiler
                .finish(&mut context, body, &graph, CodeVersion::new(1).unwrap())
                .unwrap();
            assert_eq!(staged.faults.len(), 2, "{abi:?}: {words:x?}");
            for (index, fault) in staged.faults.iter().enumerate() {
                assert_eq!(fault.instruction.block_key().pc.get(), index as u64 * 4);
                assert_eq!(fault.completed, index as u16);
            }
        }
    }
}

#[test]
fn hcq_constant_branch_eliminates_only_the_unreachable_exit() {
    // MOVZ X0,#0; CBZ X0,16. Both arms are discovered, but only RET is live.
    let graph = graph(&[
        (0, &[0xd2800000, 0xb4000060]),
        (8, &[0xd61f0020]),
        (16, &[0xd65f03c0]),
    ]);
    let analysis = Analysis::build(&graph, &[0]);
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let compiler = Compiler::new(abi, 0x10000).unwrap();
        let mut context = Context::new();
        let body = compiler
            .emit(
                &mut context,
                &mut FunctionBuilderContext::new(),
                &graph,
                &analysis,
                &[0],
            )
            .unwrap();
        let staged = compiler
            .finish(&mut context, body, &graph, CodeVersion::new(1).unwrap())
            .unwrap();
        assert_eq!(staged.states.len(), 1);
        assert_eq!(staged.states[0].exit.unwrap().pc.get(), 16);
    }
}

#[test]
fn hcq_dead_arm_drops_fault_and_cycle_maps_but_not_public_entries() {
    for dead in [&[0xf9400021, 0xd61f0020][..], &[0x14000000][..]] {
        // MOVZ; CBZ selects RET. The other arm contains a fault or cycle poll.
        let graph = graph(&[
            (0, &[0xd2800000, 0xb4000060]),
            (8, dead),
            (16, &[0xd65f03c0]),
        ]);
        let other = crate::hcq::flow::tests::block(&graph, 8);
        for entries in [&[0][..], &[0, other][..]] {
            let analysis = Analysis::build(&graph, entries);
            for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
                let compiler = Compiler::new(abi, 0x10000).unwrap();
                let mut context = Context::new();
                let body = compiler
                    .emit(
                        &mut context,
                        &mut FunctionBuilderContext::new(),
                        &graph,
                        &analysis,
                        entries,
                    )
                    .unwrap();
                let staged = compiler
                    .finish(&mut context, body, &graph, CodeVersion::new(1).unwrap())
                    .unwrap();
                assert_eq!(staged.entries.len(), entries.len());
                if entries.len() == 1 {
                    assert_eq!(staged.states.len(), 1);
                    assert!(staged.faults.is_empty());
                } else {
                    assert!(staged.states.len() > 1);
                    assert_eq!(staged.faults.len(), usize::from(dead[0] == 0xf9400021));
                }
            }
        }
    }
}

#[test]
fn hcq_live_exit_without_a_machine_map_remains_an_error() {
    let graph = graph(&[(0, &[0xd65f03c0])]);
    let analysis = Analysis::build(&graph, &[0]);
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let compiler = Compiler::new(abi, 0x10000).unwrap();
        let mut context = Context::new();
        let body = compiler
            .emit(
                &mut context,
                &mut FunctionBuilderContext::new(),
                &graph,
                &analysis,
                &[0],
            )
            .unwrap();
        context.func.nixe_exit_costs.insert(1, 1);
        context
            .compile(&*compiler.isa, &mut ControlPlane::default())
            .unwrap();
        let mut code = context.take_compiled_code().unwrap();
        code.buffer.nixe_states.retain(|map| map.entry);
        let error = body
            .stage(
                abi,
                code,
                &context.func,
                &graph,
                CodeVersion::new(1).unwrap(),
            )
            .err()
            .unwrap();
        assert!(error.to_string().contains("HCQ exit map missing"));
    }
}

#[test]
fn hcq_dead_fp_activation_needs_no_source_map() {
    let graph = graph(&[
        (0, &[0xd2800000, 0xb4000060]),
        (8, &[0x1e222820, 0xd65f03c0]), // dead FADD; RET (continuation is an external label)
        (16, &[0xd65f03c0]),
    ]);
    let analysis = Analysis::build(&graph, &[0]);
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let compiler = Compiler::new(abi, 0x10000).unwrap();
        let mut context = Context::new();
        let body = compiler
            .emit(
                &mut context,
                &mut FunctionBuilderContext::new(),
                &graph,
                &analysis,
                &[0],
            )
            .unwrap();
        assert_eq!(body.fp_activations.len(), 1);
        let staged = compiler
            .finish(&mut context, body, &graph, CodeVersion::new(1).unwrap())
            .unwrap();
        assert_eq!(staged.entries.len(), 1);
        assert!(staged.states.iter().all(|state| state.exit.is_some()));
    }
}

#[test]
fn hcq_live_fault_without_a_machine_map_remains_an_error() {
    let graph = graph(&[(0, &[0xf9400021, 0xd65f03c0])]);
    let analysis = Analysis::build(&graph, &[0]);
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let compiler = Compiler::new(abi, 0x10000).unwrap();
        let mut context = Context::new();
        let body = compiler
            .emit(
                &mut context,
                &mut FunctionBuilderContext::new(),
                &graph,
                &analysis,
                &[0],
            )
            .unwrap();
        context.func.nixe_exit_costs.insert(1, 2);
        context
            .compile(&*compiler.isa, &mut ControlPlane::default())
            .unwrap();
        let mut code = context.take_compiled_code().unwrap();
        assert_eq!(code.buffer.nixe_faults.len(), 1);
        code.buffer.nixe_faults.clear();
        let error = body
            .stage(
                abi,
                code,
                &context.func,
                &graph,
                CodeVersion::new(1).unwrap(),
            )
            .err()
            .unwrap();
        assert!(
            error
                .to_string()
                .contains("unexpected number of faulting instructions")
        );
    }
}

#[test]
fn hcq_policy_preserves_native_capabilities_and_lcq_compile_policy() {
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let compiler = Compiler::new(abi, 0x10000).unwrap();
        let hcq = &compiler.isa;
        let lcq = target::build(abi, Policy::Lcq).unwrap();
        assert_eq!(hcq.flags().opt_level(), OptLevel::Speed);
        assert_eq!(
            hcq.flags().regalloc_algorithm(),
            RegallocAlgorithm::Backtracking
        );
        assert_eq!(lcq.flags().opt_level(), OptLevel::None);
        assert_eq!(
            lcq.flags().regalloc_algorithm(),
            RegallocAlgorithm::SinglePass
        );
        let capabilities = |isa: &dyn TargetIsa| {
            isa.isa_flags()
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
        };
        assert_eq!(capabilities(&*lcq), capabilities(&**hcq));
        for isa in [&lcq, hcq] {
            assert!(isa.flags().enable_pinned_reg());
            assert!(isa.flags().enable_nixe_abi());
            assert!(isa.flags().machine_code_cfg_info());
            if abi == HostAbi::X86_64 {
                assert!(isa.flags().enable_nixe_ibt());
            } else {
                assert!(
                    isa.isa_flags()
                        .iter()
                        .any(|flag| flag.name == "use_bti" && flag.as_bool() == Some(true))
                );
            }
        }
    }
}

#[test]
fn hcq_backend_rejects_only_typed_resource_limits() {
    assert!(matches!(
        Failure::backend(CodegenError::CodeTooLarge),
        Failure::Rejected(Limit::CodeSize)
    ));
    assert!(matches!(
        Failure::backend(CodegenError::ImplLimitExceeded),
        Failure::Rejected(Limit::Implementation)
    ));
    for error in [
        CodegenError::Unsupported("missing native lowering".into()),
        CodegenError::Verifier(Default::default()),
    ] {
        let Failure::Failed(error) = Failure::backend(error) else {
            panic!("compiler failures must not be hidden as optimizer rejections");
        };
        assert_eq!(error.kind, crate::jit_error::Kind::Internal);
        assert!(error.to_string().contains("HCQ Cranelift"));
    }
}

#[test]
fn hcq_backend_reuses_scratch_after_success_frame_rejection_and_backend_failure() {
    let graph = graph(&[(0, &[0xd2800020, 0xd65f03c0])]); // MOVZ; RET
    let analysis = Analysis::build(&graph, &[0]);
    for abi in [HostAbi::X86_64, HostAbi::Aarch64] {
        let compiler = Compiler::new(abi, 0x10000).unwrap();
        let mut context = Context::new();
        let mut frontend = FunctionBuilderContext::new();
        // Good jobs before/after every rejection/failure use the SAME scratch.
        for mode in [0, 1, 0, 2, 0, 3, 0] {
            let mut body = compiler
                .emit(&mut context, &mut frontend, &graph, &analysis, &[0])
                .unwrap();
            if mode == 1 {
                // Each legal slot fits individually, but their aggregate frame
                // exceeds the fixed arena. Exercise the real backend limit.
                for _ in 0..2 {
                    context.func.create_sized_stack_slot(ir::StackSlotData::new(
                        ir::StackSlotKind::ExplicitSlot,
                        8192,
                        4,
                    ));
                }
            } else if mode == 2 {
                // An illegal frontend slot shape is Unsupported, not permission
                // to permanently suppress this seed as an optimizer limit.
                context.func.create_sized_stack_slot(ir::StackSlotData::new(
                    ir::StackSlotKind::ExplicitSlot,
                    cranelift_codegen::nixe::FRAME_BYTES + 1,
                    4,
                ));
            } else if mode == 3 {
                body.exits[0].pc_operand = usize::MAX; // invalid physical map request
            }
            let result = compiler.finish(&mut context, body, &graph, CodeVersion::new(1).unwrap());
            match mode {
                0 => {
                    let staged = result.unwrap();
                    assert_eq!(staged.entries.len(), 1);
                    let transfer = staged.states[0].transfer.as_ref().unwrap();
                    assert_eq!(transfer.completed, 2);
                    assert!(transfer.poll_offset.is_some());
                }
                1 => assert!(matches!(
                    result,
                    Err(Failure::Rejected(Limit::Implementation))
                )),
                _ => assert!(matches!(result, Err(Failure::Failed(_)))),
            }
            assert!(context.compiled_code().is_none());
            assert!(context.func.layout.blocks().next().is_none());
            assert!(context.func.nixe_exit_costs.is_empty());
        }
    }
}
