use super::*;
use crate::hcq::flow::tests::{block, graph};
use cranelift_codegen::{
    Context,
    control::ControlPlane,
    isa,
    settings::{self, Configurable},
};
use cranelift_frontend::FunctionBuilderContext;
use nixe_cpu::decode::{
    DecodeResult,
    a64::{self, A64Instruction},
};

const ADDS: u32 = 0xb1000420;
const SUBS: u32 = 0xf1000420;
const RET: u32 = 0xd65f03c0;

fn body(graph: &Graph, entries: &[usize]) -> (ir::Function, Ssa) {
    let analysis = Analysis::build(graph, entries);
    let mut context = Context::new();
    let mut frontend = FunctionBuilderContext::new();
    let isa = target("x86_64-unknown-linux-gnu");
    let body = crate::hcq::compiler::emit(
        crate::abi::HostAbi::X86_64,
        &*isa,
        None,
        &mut context,
        &mut frontend,
        graph,
        &analysis,
        entries,
    )
    .unwrap();
    (context.func, body.ssa)
}

pub(in crate::hcq) fn target(triple: &str) -> std::sync::Arc<dyn isa::TargetIsa> {
    let mut flags = settings::builder();
    for (name, value) in [
        ("enable_pinned_reg", "true"),
        ("enable_nixe_abi", "true"),
        ("opt_level", "speed"),
        ("regalloc_algorithm", "backtracking"),
        ("regalloc_checker", "true"),
    ] {
        flags.set(name, value).unwrap();
    }
    let mut isa = isa::lookup(triple.parse().unwrap()).unwrap();
    if triple.starts_with("x86_64") {
        flags.set("enable_nixe_ibt", "true").unwrap();
    } else {
        isa.set("use_bti", "true").unwrap();
    }
    isa.finish(settings::Flags::new(flags)).unwrap()
}

fn compile_both(function: ir::Function, ssa: &Ssa) {
    // The declared root is analysis-only. Each actual ingress has one nixe_entry
    // and the shared body has no ABI calls or canonical home accesses.
    for block in function.layout.blocks() {
        for inst in function.layout.block_insts(block) {
            let opcode = function.dfg.insts[inst].opcode();
            assert!(!opcode.can_load() && !opcode.can_store() && !opcode.is_call());
        }
    }
    let entries: Vec<_> = ssa.entries.iter().map(|entry| entry.label).collect();
    for &entry in &entries {
        assert!(function.dfg.block_params(entry).is_empty());
    }
    assert_eq!(function.nixe_entries, entries);
    for triple in ["x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"] {
        let isa = target(triple);
        let mut context = Context::for_function(function.clone());
        let code = context
            .compile(&*isa, &mut ControlPlane::default())
            .unwrap();
        assert_eq!(code.buffer.nixe_entries.len(), entries.len());
        for entry in &ssa.entries {
            assert!(
                code.buffer
                    .nixe_entries
                    .iter()
                    .any(|(block, _)| *block == entry.label)
            );
            let map = code
                .buffer
                .nixe_states
                .iter()
                .find(|map| map.entry && map.id == entry.id)
                .unwrap();
            assert_eq!(
                map.values.len(),
                ssa.blocks[entry.target].operands.len()
                    + usize::from(ssa.blocks[entry.target].flags.is_some())
            );
        }
        assert!(
            code.buffer.frame_layout().unwrap().nixe_frame_size.unwrap()
                <= cranelift_codegen::nixe::FRAME_BYTES
        );
    }
}

#[test]
fn hcq_ssa_public_labels_define_own_minimal_inputs_and_share_one_body() {
    let graph = graph(&[(0, &[ADDS, 0x14000001]), (8, &[RET])]);
    let join = block(&graph, 8);
    let (function, ssa) = body(&graph, &[0, join]);
    assert_eq!(ssa.entries.len(), 2);
    assert_eq!(
        ssa.blocks[0].operands,
        vec![GuestValue::General(1), GuestValue::General(30)]
    );
    assert!(ssa.blocks[0].flags.is_none());
    assert_eq!(
        ssa.blocks[join].operands,
        vec![
            GuestValue::General(0),
            GuestValue::General(1),
            GuestValue::General(30)
        ]
    );
    assert!(matches!(ssa.blocks[join].flags, Some(LazyFlags::Packed(_))));
    assert_eq!(function.layout.blocks().count(), graph.blocks.len() + 3);
    compile_both(function, &ssa);
}

#[test]
fn hcq_ssa_diamond_carries_lazy_values_or_reconciles_different_recipes() {
    for other in [ADDS, SUBS, 0x31000420, 0xba020020, 0xfa42102a] {
        let graph = graph(&[
            (0, &[0x54000080]),
            (4, &[ADDS, 0x14000004]),
            (16, &[other, 0x14000001]),
            (24, &[RET]),
        ]);
        let (function, ssa) = body(&graph, &[0]);
        let join = &ssa.blocks[block(&graph, 24)];
        if other == ADDS {
            assert!(matches!(join.flags, Some(LazyFlags::Add { width: 64, .. })));
        } else {
            assert!(matches!(join.flags, Some(LazyFlags::Packed(_))));
        }
        compile_both(function, &ssa);
    }
}

#[test]
fn hcq_ssa_loop_parameters_preserve_first_entry_and_backedge_values() {
    for public_header in [false, true] {
        let graph = graph(&[
            (0, &[ADDS, 0x14000001]),
            (8, &[0x54000040]),
            (12, &[RET]),
            (16, &[SUBS, 0x17fffffd]),
        ]);
        let entries = if public_header {
            vec![0, block(&graph, 8)]
        } else {
            vec![0]
        };
        let (function, ssa) = body(&graph, &entries);
        compile_both(function, &ssa);
    }
}

#[test]
fn hcq_ssa_typed_carry_and_conditional_operands_survive_register_overwrites() {
    for word in [0xba020020, 0x7a020020, 0xfa42102a] {
        let graph = graph(&[(0, &[word, 0xd2800001, 0x14000001]), (12, &[RET])]);
        let (function, ssa) = body(&graph, &[0]);
        let join = &ssa.blocks[block(&graph, 12)];
        let flags = join.flags.as_ref().unwrap();
        flags
            .try_map_with_bits(&mut |value, bits| {
                assert_eq!(function.dfg.value_type(*value).bits(), u32::from(bits));
                Ok::<_, ()>(())
            })
            .unwrap();
        let params = function.dfg.block_params(join.label);
        assert_eq!(params.len(), join.operands.len() + 4);
        compile_both(function, &ssa);
    }
}

#[test]
fn hcq_ssa_missing_bypass_value_is_an_error_not_a_canonical_load() {
    let graph = graph(&[(0, &[ADDS, 0x14000001]), (8, &[RET])]);
    let (_, ssa) = body(&graph, &[0]);
    assert!(
        ssa.blocks[block(&graph, 8)]
            .arguments(&Values::default(), None)
            .is_err()
    );
}

#[test]
fn hcq_ssa_entry_without_live_inputs_has_no_dummy_flag_parameter() {
    let graph = graph(&[(0, &[0xd280003e, RET])]); // MOVZ X30,#1; RET
    let (function, ssa) = body(&graph, &[0]);
    assert!(ssa.blocks[0].operands.is_empty());
    assert!(ssa.blocks[0].flags.is_none());
    compile_both(function, &ssa);
}

#[test]
fn hcq_ssa_packed_join_retains_the_bit_mask_instead_of_demanding_all_flags() {
    let graph = graph(&[
        (0, &[0x54000080]),
        (4, &[ADDS, 0x14000004]),
        (16, &[SUBS, 0x14000001]),
        (24, &[0x9a820020, ADDS, RET]),
    ]);
    let (function, ssa) = body(&graph, &[0]);
    assert_eq!(ssa.blocks[block(&graph, 24)].flag_mask, crate::analysis::Z);
    compile_both(function, &ssa);
}

#[test]
fn hcq_ssa_partial_packing_executes_only_the_required_flag_contract() {
    use crate::executable::{
        Cache, Tier,
        output::{Metadata, Output},
    };
    use nixe_cpu::{
        platform::TargetPlatform,
        state::a64::{A64State, Nzcv},
    };
    // This isolated flag calculation uses the host ABI, not a guest entry.
    // Cache owns the immutable code; no dispatch payload is published.
    let isa = cranelift_native::builder()
        .unwrap()
        .finish(settings::Flags::new(settings::builder()))
        .unwrap();
    let abi = if cfg!(target_arch = "x86_64") {
        crate::abi::HostAbi::X86_64
    } else {
        crate::abi::HostAbi::Aarch64
    };
    let cache = Cache::new().unwrap();
    for word in [
        0xab020020, 0x6b020020, 0xba020020, 0xfa020020, 0xea020020, 0xfa42102a,
    ] {
        let graph = graph(&[(0, &[word, RET])]);
        let DecodeResult::Decoded(decoded) = &graph.instructions[0].decoded else {
            unreachable!()
        };
        let A64Instruction::Integer(instruction) =
            a64::normalize(&decoded.instruction, decoded.encoding)
        else {
            unreachable!()
        };
        for mask in 1..=15 {
            let mut context = Context::new();
            context.func.signature.call_conv = isa.default_call_conv();
            context.func.signature.params = vec![
                AbiParam::new(types::I64),
                AbiParam::new(types::I64),
                AbiParam::new(types::I32),
            ];
            context.func.signature.returns = vec![AbiParam::new(types::I32)];
            let mut frontend = FunctionBuilderContext::new();
            let mut builder = FunctionBuilder::new(&mut context.func, &mut frontend);
            let entry = builder.create_block();
            builder.append_block_params_for_function_params(entry);
            builder.switch_to_block(entry);
            let args = builder.block_params(entry).to_vec();
            let mut values = Values::default();
            values.bind(GuestValue::General(1), args[0]);
            values.bind(GuestValue::General(2), args[1]);
            let mut emitter = Translator::new(
                builder,
                abi,
                &*isa,
                None,
                crate::analysis::StateSet::default(),
            );
            emitter.values = values;
            let flags = emitter
                .emit_integer(
                    graph.blocks[0].key.pc,
                    instruction,
                    &LazyFlags::Canonical(args[2]),
                )
                .unwrap()
                .unwrap();
            let packed = emitter.packed_flag_subset(&flags, mask);
            emitter.builder.ins().return_(&[packed]);
            emitter.builder.seal_all_blocks();
            emitter.builder.finalize(isa.frontend_config());
            let compiled = context
                .compile(&*isa, &mut ControlPlane::default())
                .unwrap();
            assert!(compiled.buffer.relocs().is_empty());
            assert!(compiled.buffer.traps().is_empty());
            let code = cache
                .install_with_islands(
                    Output {
                        bytes: compiled.buffer.data().into(),
                        alignment: compiled.buffer.alignment as usize,
                        metadata: Metadata {
                            abi,
                            // No NativeFrame or published Nixe state maps in this host-ABI fixture.
                            frame_extent: 0,
                            entries: Box::new([]),
                            states: Box::new([]),
                            faults: Box::new([]),
                            traps: Box::new([]),
                            relocations: Box::new([]),
                        },
                    },
                    Tier::Lcq,
                    0,
                    |_| None,
                )
                .unwrap();
            // SAFETY: host-compatible compiled signature, no external relocations;
            // the allocation lease stays alive through every call below.
            let function: unsafe extern "C" fn(u64, u64, u32) -> u32 =
                unsafe { std::mem::transmute(code.allocation.address()) };
            for (lhs, rhs) in [
                (0, 0),
                (u64::MAX, 1),
                (1 << 63, 1),
                (0x7fff_ffff, 1),
                (5, 7),
            ] {
                for input in 0..16 {
                    let mut state = A64State::default();
                    state.general_register_storage_mut()[1] = lhs;
                    state.general_register_storage_mut()[2] = rhs;
                    state.set_nzcv(Nzcv::from_bits(input << 28));
                    nixe_cpu_interpreter::execute_one(&TargetPlatform::Switch1, &mut state, word)
                        .unwrap();
                    let result = unsafe { function(lhs, rhs, input << 28) };
                    let mask = u32::from(mask) << 28;
                    assert_eq!(
                        result & mask,
                        state.nzcv().bits() & mask,
                        "{word:x} mask {mask:x}"
                    );
                }
            }
        }
    }
}
