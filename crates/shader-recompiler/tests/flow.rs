use nixe_shader_recompiler::{
    diagnostics::FailureCategory,
    environment::*,
    ir::{Condition, Reg},
    maxwell::{
        bits::insert,
        decode::Opcode,
        flow::{EndClass, FlowGraph, Token, TokenStack},
        location::{FlowLocation, Location},
    },
    types::{ByteOffset, CbufIndex, ProgramOffset, Stage},
};

fn pc(offset: u32) -> Location {
    Location::new(ProgramOffset(offset)).unwrap()
}
fn word(op: Opcode) -> u64 {
    op.encoding().value | (7 << 16)
}
fn branch(op: Opcode, from: u32, target: u32) -> u64 {
    insert::<20, 24>(
        word(op) | 15,
        target.wrapping_sub(from).wrapping_sub(8) as u64,
    )
}
fn environment(code: &[(u32, u64)]) -> SnapshotEnvironment {
    SnapshotEnvironment::new(
        EnvironmentMetadata {
            stage: Stage::Compute,
            ..Default::default()
        },
        Observations {
            instructions: code
                .iter()
                .map(|&(address, word)| (ProgramOffset(address), word))
                .collect(),
            ..Default::default()
        },
    )
}
fn block_at(graph: &FlowGraph, address: u32) -> nixe_shader_recompiler::arena::FlowBlockId {
    graph.functions[0].blocks[&FlowLocation::Instruction(pc(address))]
}

#[test]
fn token_pop_discards_newer_entries_and_nested_transfers_follow_the_stack() {
    let mut stack = TokenStack::default();
    stack.push(Token::Ssy, pc(104));
    stack.push(Token::Pbk, pc(80));
    stack.push(Token::Pcnt, pc(56));
    let (target, remaining) = stack.pop(Token::Pbk).unwrap();
    assert_eq!(target, pc(80));
    assert_eq!(remaining.peek(Token::Ssy), Some(pc(104)));
    assert_eq!(remaining.peek(Token::Pcnt), None);
    assert!(remaining.pop(Token::Pbk).is_err());
    let mut env = environment(&[
        (8, branch(Opcode::SSY, 8, 104)),
        (16, branch(Opcode::PBK, 16, 80)),
        (24, branch(Opcode::PCNT, 24, 56)),
        (40, word(Opcode::CONT) | 15),
        (56, word(Opcode::BRK) | 15),
        (80, word(Opcode::SYNC) | 15),
        (104, word(Opcode::EXIT) | 15),
    ]);
    let graph = FlowGraph::analyze(&mut env, pc(8), false).unwrap();
    for (from, to) in [(8, 56), (56, 80), (80, 104)] {
        assert_eq!(
            graph.block(block_at(&graph, from)).unwrap().branch_true,
            Some(block_at(&graph, to))
        );
    }
    assert_eq!(
        graph.block(block_at(&graph, 104)).unwrap().end_class,
        EndClass::Exit
    );
}

#[test]
fn branch_to_middle_splits_blocks_and_preserves_loop_end() {
    let mut env = environment(&[
        (8, word(Opcode::NOP)),
        (16, word(Opcode::NOP)),
        (24, branch(Opcode::BRA, 24, 16)),
    ]);
    let graph = FlowGraph::analyze(&mut env, pc(8), false).unwrap();
    let first = block_at(&graph, 8);
    let tail = block_at(&graph, 16);
    assert_eq!(
        graph.block(first).unwrap().end,
        FlowLocation::Instruction(pc(16))
    );
    assert_eq!(graph.block(first).unwrap().branch_true, Some(tail));
    assert_eq!(
        graph.block(tail).unwrap().end,
        FlowLocation::Instruction(pc(24))
    );
    assert_eq!(graph.block(tail).unwrap().branch_true, Some(tail));
    assert_eq!(graph.functions[0].blocks.len(), 2);
}

#[test]
fn incoming_back_edge_targets_predicate_test_before_instruction() {
    let predicated = insert::<16, 4>(word(Opcode::ISETP_reg), 0);
    let mut env = environment(&[
        (8, word(Opcode::NOP)),
        (16, predicated),
        (24, branch(Opcode::BRA, 24, 16)),
    ]);
    let graph = FlowGraph::analyze(&mut env, pc(8), false).unwrap();
    let test = graph.functions[0].blocks[&FlowLocation::PredicateBefore(pc(16))];
    let execution = block_at(&graph, 16);
    let back = block_at(&graph, 24);
    assert_eq!(graph.block(test).unwrap().branch_true, Some(execution));
    assert_eq!(graph.block(test).unwrap().branch_false, Some(back));
    assert_eq!(graph.block(execution).unwrap().branch_true, Some(back));
    assert_eq!(graph.block(back).unwrap().branch_true, Some(test));
    assert_eq!(graph.block(test).unwrap().condition.predicate.index(), 0);
    assert_eq!(
        graph.block(execution).unwrap().condition,
        Condition::always(true)
    );
}

#[test]
fn calls_and_demotions_keep_their_continuations() {
    let mut env = environment(&[
        (8, branch(Opcode::CAL, 8, 40)),
        (16, word(Opcode::EXIT) | 15),
        (40, word(Opcode::RET) | 15),
    ]);
    let graph = FlowGraph::analyze(&mut env, pc(8), false).unwrap();
    assert_eq!(graph.functions.len(), 2);
    let call = graph.block(block_at(&graph, 8)).unwrap();
    assert_eq!(call.end_class, EndClass::Call);
    assert_eq!(call.function_call.unwrap().0, 1);
    assert_eq!(call.return_block, Some(block_at(&graph, 16)));
    let returned = *graph.functions[1].blocks.values().next().unwrap();
    assert_eq!(graph.block(returned).unwrap().end_class, EndClass::Return);
    let mut env = environment(&[(8, word(Opcode::KIL) | 15), (16, word(Opcode::EXIT) | 15)]);
    let graph = FlowGraph::analyze(&mut env, pc(8), false).unwrap();
    let kill = graph.block(block_at(&graph, 8)).unwrap();
    assert_eq!(kill.end_class, EndClass::Kill);
    assert_eq!(kill.branch_true, Some(block_at(&graph, 16)));
}

#[test]
fn vertex_a_exits_share_a_dispatch_continuation() {
    let mut env = environment(&[
        (8, insert::<16, 4>(word(Opcode::EXIT) | 15, 0)),
        (16, word(Opcode::EXIT) | 15),
    ]);
    let graph = FlowGraph::analyze(&mut env, pc(8), true).unwrap();
    let dispatcher = graph.dispatcher.unwrap();
    assert_eq!(
        graph.block(block_at(&graph, 8)).unwrap().branch_true,
        Some(dispatcher)
    );
    assert_eq!(
        graph.block(block_at(&graph, 16)).unwrap().branch_true,
        Some(dispatcher)
    );
    assert_eq!(graph.block(dispatcher).unwrap().end_class, EndClass::Exit);
    assert_eq!(
        graph.block(dispatcher).unwrap().begin,
        FlowLocation::Instruction(pc(24))
    );
}

fn indirect_environment(conditional: bool, negative: bool) -> SnapshotEnvironment {
    let imnmx = word(Opcode::IMNMX_imm) | 2 | (1 << 8) | (2 << 20) | ((negative as u64) << 56);
    let shl = word(Opcode::SHL_imm) | 3 | (2 << 8) | (2 << 20);
    let ldc = insert::<48, 3>(
        word(Opcode::LDC) | 4 | (3 << 8) | (0x100 << 20) | (2 << 36),
        4,
    );
    let brx = insert::<16, 4>(
        word(Opcode::BRX) | 15 | (4 << 8),
        if conditional { 0 } else { 7 },
    );
    let code = [
        (8, imnmx),
        (16, shl),
        (24, ldc),
        (40, brx),
        (56, word(Opcode::EXIT) | 15),
        (80, word(Opcode::EXIT) | 15),
    ];
    SnapshotEnvironment::new(
        EnvironmentMetadata {
            stage: Stage::Compute,
            ..Default::default()
        },
        Observations {
            instructions: code
                .into_iter()
                .map(|(pc, word)| (ProgramOffset(pc), word))
                .collect(),
            cbuf_words: [(0x100, 8), (0x104, 32), (0x108, 8)]
                .into_iter()
                .map(|(offset, value)| {
                    (
                        CbufWordKey {
                            index: CbufIndex(2),
                            offset: ByteOffset(offset),
                        },
                        value,
                    )
                })
                .collect(),
            ..Default::default()
        },
    )
}

#[test]
fn indirect_branch_targets_are_finite_sorted_and_record_every_cbuf_word() {
    let mut source = indirect_environment(false, false);
    let mut recording = RecordingEnvironment::new(&mut source);
    let graph = FlowGraph::analyze(&mut recording, pc(8), false).unwrap();
    assert_eq!(recording.observations().cbuf_words.len(), 3);
    let entry = graph.block(block_at(&graph, 8)).unwrap();
    assert_eq!(entry.end_class, EndClass::IndirectBranch);
    assert_eq!(entry.branch_reg, Reg(4));
    assert_eq!(entry.branch_offset, 48);
    assert_eq!(
        entry
            .indirect_branches
            .iter()
            .map(|entry| entry.address.0)
            .collect::<Vec<_>>(),
        [56, 80]
    );
    for (conditional, negative, reason) in [
        (true, false, "Conditional indirect"),
        (false, true, "Failed to track"),
    ] {
        let mut env = indirect_environment(conditional, negative);
        let error = FlowGraph::analyze(&mut env, pc(8), false).err().unwrap();
        assert_eq!(error.category, FailureCategory::Unsupported);
        assert_eq!(error.pc, Some(ProgramOffset(40)));
        assert!(error.message.contains(reason));
    }
}

#[test]
fn unsupported_control_forms_fail_with_source_context() {
    for (code, reason) in [
        (vec![(8, word(Opcode::PRET))], "PRET"),
        (
            vec![(8, branch(Opcode::BRA, 8, 40) | (1 << 5))],
            "constant buffer offset",
        ),
        (
            vec![
                (8, branch(Opcode::PEXIT, 8, 40)),
                (16, insert::<16, 4>(word(Opcode::EXIT) | 15, 0)),
            ],
            "Conditional EXIT with PEXIT",
        ),
        (vec![(8, word(Opcode::BRX) | 15)], "Failed to track"),
    ] {
        let mut env = environment(&code);
        let error = FlowGraph::analyze(&mut env, pc(8), false).err().unwrap();
        assert_eq!(error.category, FailureCategory::Unsupported);
        assert_eq!(error.stage, Some(Stage::Compute));
        assert!(error.pc.is_some());
        assert!(error.instruction.is_some());
        assert!(error.message.contains(reason), "{}", error.message);
    }
}
