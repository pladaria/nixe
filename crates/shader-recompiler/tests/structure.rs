use nixe_shader_recompiler::{
    Environment, HostTranslateInfo, Result,
    arena::{BlockId, StatementId},
    diagnostics::FailureCategory,
    environment::{EnvironmentMetadata, Observations, SnapshotEnvironment},
    ir::{Condition, Emitter, Ir, Opcode, Pred, Program, Reg, SyntaxNode, Value},
    maxwell::{
        asl::{CodeTranslator, build_asl},
        bits::insert,
        decode::Opcode as GuestOpcode,
        flow::FlowGraph,
        location::Location,
        structure::{StatementKind as S, StatementTree},
    },
    types::{ProgramOffset, Stage},
};

fn children(tree: &StatementTree, parent: StatementId) -> Vec<StatementId> {
    let mut nodes = Vec::new();
    let mut next = tree.node(parent).unwrap().first;
    while let Some(id) = next {
        nodes.push(id);
        next = tree.node(id).unwrap().next;
    }
    nodes
}

#[test]
fn forward_and_backward_movement_preserve_node_identity() {
    for backward in [false, true] {
        let mut tree = StatementTree::new().unwrap();
        let root = tree.root;
        let condition = tree.allocate(S::Identity(Condition::always(true))).unwrap();
        let label = tree.allocate(S::Label(0)).unwrap();
        let jump = tree.allocate(S::Goto { condition, label }).unwrap();
        tree.insert(root, None, if backward { label } else { jump })
            .unwrap();
        let code = tree.insert_new(root, None, S::Code(None)).unwrap();
        tree.insert(root, None, if backward { jump } else { label })
            .unwrap();
        tree.remove_goto(jump).unwrap();
        tree.verify().unwrap();
        let container = tree.node(root).unwrap().first.unwrap();
        assert!(matches!(
            tree.node(container).unwrap().kind,
            S::Loop(_) | S::If(_)
        ));
        assert_eq!(tree.node(code).unwrap().parent, Some(container));
        assert_eq!(tree.node(jump).unwrap().parent, None);
        assert_eq!(
            tree.node(label).unwrap().parent,
            Some(if backward { container } else { root })
        );
    }
}

#[test]
fn nested_entry_and_lifting_keep_labels_and_update_selection_conditions() {
    for backward in [false, true] {
        let mut tree = StatementTree::new().unwrap();
        let root = tree.root;
        let condition = tree.allocate(S::Identity(Condition::always(true))).unwrap();
        let label = tree.allocate(S::Label(3)).unwrap();
        let jump = tree.allocate(S::Goto { condition, label }).unwrap();
        if !backward {
            tree.insert(root, None, jump).unwrap();
        }
        let selection = tree.insert_new(root, None, S::If(condition)).unwrap();
        tree.insert_new(selection, None, S::Code(None)).unwrap();
        tree.insert(selection, None, label).unwrap();
        tree.insert_new(selection, None, S::Code(None)).unwrap();
        if backward {
            tree.insert(root, None, jump).unwrap();
        }
        tree.remove_goto(jump).unwrap();
        tree.verify().unwrap();
        let S::If(expr) = tree.node(selection).unwrap().kind else {
            panic!()
        };
        assert!(matches!(tree.node(expr).unwrap().kind, S::Or(_, _)));
        assert_eq!(tree.node(label).unwrap().parent, Some(selection));
        if backward {
            let outer = tree.node(selection).unwrap().parent.unwrap();
            assert!(matches!(tree.node(outer).unwrap().kind, S::Loop(_)));
        }
    }
}

#[test]
fn outward_loop_transfers_become_distinct_conditional_breaks() {
    let mut tree = StatementTree::new().unwrap();
    let root = tree.root;
    let condition = tree.allocate(S::Identity(Condition::always(true))).unwrap();
    let outer = tree.insert_new(root, None, S::Loop(condition)).unwrap();
    tree.insert_new(outer, None, S::Code(None)).unwrap();
    let a = tree.insert_new(root, None, S::Label(0)).unwrap();
    tree.insert_new(root, None, S::Code(None)).unwrap();
    let b = tree.insert_new(root, None, S::Label(1)).unwrap();
    let ja = tree
        .insert_new(
            outer,
            None,
            S::Goto {
                condition,
                label: a,
            },
        )
        .unwrap();
    let jb = tree
        .insert_new(
            outer,
            None,
            S::Goto {
                condition,
                label: b,
            },
        )
        .unwrap();
    tree.remove_goto(jb).unwrap();
    tree.remove_goto(ja).unwrap();
    tree.verify().unwrap();
    assert_eq!(
        children(&tree, outer)
            .iter()
            .filter(|&&id| matches!(tree.node(id).unwrap().kind, S::Break(_)))
            .count(),
        2
    );
}

#[test]
fn outward_selection_retains_exact_suffix_boundaries_and_reports_empty_range() {
    for empty in [false, true] {
        let mut tree = StatementTree::new().unwrap();
        let root = tree.root;
        let condition = tree.allocate(S::Identity(Condition::always(true))).unwrap();
        let selection = tree.insert_new(root, None, S::If(condition)).unwrap();
        let label = tree.insert_new(root, None, S::Label(0)).unwrap();
        let jump = tree
            .insert_new(selection, None, S::Goto { condition, label })
            .unwrap();
        let suffix = if empty {
            None
        } else {
            let dropped = tree.insert_new(selection, None, S::Label(1)).unwrap();
            let kept = tree.insert_new(selection, None, S::Code(None)).unwrap();
            Some((dropped, kept))
        };
        let result = tree.remove_goto(jump);
        if let Some((dropped, kept)) = suffix {
            result.unwrap();
            tree.verify().unwrap();
            assert_eq!(tree.node(dropped).unwrap().parent, None);
            assert_ne!(tree.node(kept).unwrap().parent, Some(selection));
        } else {
            assert_eq!(
                result.unwrap_err().category,
                FailureCategory::UnsupportedReferenceCase
            );
        }
    }
}

#[test]
fn lifting_rejects_capture_of_breaks() {
    let mut tree = StatementTree::new().unwrap();
    let root = tree.root;
    let condition = tree.allocate(S::Identity(Condition::always(true))).unwrap();
    let label = tree.insert_new(root, None, S::Label(0)).unwrap();
    tree.insert_new(root, None, S::Break(condition)).unwrap();
    let jump = tree
        .insert_new(root, None, S::Goto { condition, label })
        .unwrap();
    let error = tree.remove_goto(jump).unwrap_err();
    assert_eq!(error.category, FailureCategory::Unsupported);
    assert!(error.message.contains("break nodes"));
}

fn word(op: GuestOpcode) -> u64 {
    op.encoding().value | (7 << 16)
}
fn branch(op: GuestOpcode, from: u32, to: u32) -> u64 {
    insert::<20, 24>(word(op) | 15, to.wrapping_sub(from).wrapping_sub(8) as u64)
}
fn pc(address: u32) -> Location {
    Location::new(ProgramOffset(address)).unwrap()
}

#[derive(Default)]
struct Ranges(Vec<(u32, u32)>);
impl<E: Environment> CodeTranslator<E> for Ranges {
    fn translate(
        &mut self,
        _: &mut E,
        ir: &mut Ir,
        block: BlockId,
        begin: Location,
        end: Location,
    ) -> Result<()> {
        self.0.push((begin.address().0, end.address().0));
        let mut emit = Emitter::new(ir, block)?;
        emit.set_reg(Reg(0), Value::U32(begin.address().0))?;
        emit.set_pred(Pred::new(0)?, Value::U1(false))?;
        Ok(())
    }
}

fn compile(code: &[(u32, u64)], reorder: bool) -> (Ir, Program, Ranges) {
    let mut env = SnapshotEnvironment::new(
        EnvironmentMetadata {
            stage: Stage::Compute,
            ..Default::default()
        },
        Observations {
            instructions: code
                .iter()
                .map(|&(pc, word)| (ProgramOffset(pc), word))
                .collect(),
            ..Default::default()
        },
    );
    let graph = FlowGraph::analyze(&mut env, pc(8), false).unwrap();
    let mut tree = StatementTree::from_flow(&graph).unwrap();
    tree.verify().unwrap();
    let mut ir = Ir::new().unwrap();
    let mut translator = Ranges::default();
    let mut program = Program::new(Stage::Compute);
    program.syntax = build_asl(
        &mut env,
        &mut ir,
        &graph,
        &mut tree,
        &HostTranslateInfo {
            needs_demote_reorder: reorder,
            ..Default::default()
        },
        &mut translator,
    )
    .unwrap();
    program.blocks = program
        .syntax
        .iter()
        .filter_map(|node| {
            if let SyntaxNode::Block(id) = node {
                Some(*id)
            } else {
                None
            }
        })
        .collect();
    if !reorder {
        ir.verify(&program).unwrap();
    }
    (ir, program, translator)
}

#[test]
fn predicate_test_precedes_the_instruction_that_overwrites_it() {
    let (ir, program, ranges) = compile(
        &[
            (8, insert::<16, 4>(word(GuestOpcode::ISETP_reg), 0)),
            (16, word(GuestOpcode::EXIT) | 15),
        ],
        false,
    );
    assert_eq!(ranges.0, [(8, 16), (16, 24)]);
    let SyntaxNode::Block(header) = program.syntax[0] else {
        panic!()
    };
    assert_eq!(
        ir.inst(ir.block(header).unwrap().instructions()[0])
            .unwrap()
            .opcode(),
        Opcode::Prologue
    );
    let (condition, body) = program
        .syntax
        .iter()
        .find_map(|node| {
            if let SyntaxNode::If {
                condition, body, ..
            } = node
            {
                Some((*condition, *body))
            } else {
                None
            }
        })
        .unwrap();
    let Value::Inst(condition) = condition else {
        panic!()
    };
    assert!(
        ir.block(header)
            .unwrap()
            .instructions()
            .contains(&condition)
    );
    assert!(
        ir.block(body)
            .unwrap()
            .instructions()
            .iter()
            .any(|&id| ir.inst(id).unwrap().opcode() == Opcode::SetPred)
    );
}

#[test]
fn calls_expand_at_each_site_and_loops_have_separate_continue_and_merge() {
    let (_, program, ranges) = compile(
        &[
            (8, branch(GuestOpcode::CAL, 8, 56)),
            (16, branch(GuestOpcode::CAL, 16, 56)),
            (24, word(GuestOpcode::EXIT) | 15),
            (56, word(GuestOpcode::NOP)),
            (72, word(GuestOpcode::RET) | 15),
        ],
        false,
    );
    assert_eq!(ranges.0, [(56, 72), (56, 72), (24, 40)]);
    assert_eq!(
        program
            .syntax
            .iter()
            .filter(|node| matches!(node, SyntaxNode::Return))
            .count(),
        1
    );
    let (ir, program, _) = compile(
        &[
            (8, word(GuestOpcode::NOP)),
            (16, insert::<16, 4>(branch(GuestOpcode::BRA, 16, 8), 0)),
            (24, word(GuestOpcode::EXIT) | 15),
        ],
        false,
    );
    let (body, continuation, merge) = program
        .syntax
        .iter()
        .find_map(|node| {
            if let SyntaxNode::Loop {
                body,
                continue_block,
                merge,
            } = node
            {
                Some((*body, *continue_block, *merge))
            } else {
                None
            }
        })
        .unwrap();
    assert_ne!(body, continuation);
    assert_ne!(continuation, merge);
    assert!(
        ir.block(continuation)
            .unwrap()
            .successors()
            .contains(&merge)
    );
}

#[test]
fn conditional_returns_and_demote_reordering_keep_epilogues() {
    let (_, program, _) = compile(
        &[
            (8, insert::<16, 4>(word(GuestOpcode::EXIT) | 15, 0)),
            (16, word(GuestOpcode::EXIT) | 15),
        ],
        false,
    );
    assert_eq!(
        program
            .syntax
            .iter()
            .filter(|node| matches!(node, SyntaxNode::Return))
            .count(),
        2
    );
    for reorder in [false, true] {
        let (ir, program, _) = compile(
            &[
                (8, insert::<16, 4>(word(GuestOpcode::KIL) | 15, 0)),
                (16, word(GuestOpcode::EXIT) | 15),
            ],
            reorder,
        );
        assert_eq!(
            program
                .syntax
                .iter()
                .filter(|node| matches!(node, SyntaxNode::Return))
                .count(),
            1
        );
        let demotes = program
            .blocks
            .iter()
            .flat_map(|&id| ir.block(id).unwrap().instructions())
            .filter(|&&id| ir.inst(id).unwrap().opcode() == Opcode::DemoteToHelperInvocation)
            .count();
        assert_eq!(demotes, 1);
    }
}

#[test]
fn demote_reordering_preserves_nested_and_multiple_return_guards() {
    for nested in [false, true] {
        let mut env = SnapshotEnvironment::new(
            EnvironmentMetadata {
                stage: Stage::Compute,
                ..Default::default()
            },
            Observations {
                instructions: [(ProgramOffset(8), word(GuestOpcode::EXIT) | 15)].into(),
                ..Default::default()
            },
        );
        let graph = FlowGraph::analyze(&mut env, pc(8), false).unwrap();
        let mut tree = StatementTree::new().unwrap();
        let root = tree.root;
        let cond = tree.allocate(S::Identity(Condition::always(true))).unwrap();
        let outer = tree.insert_new(root, None, S::If(cond)).unwrap();
        let inner = if nested {
            tree.insert_new(outer, None, S::If(cond)).unwrap()
        } else {
            outer
        };
        tree.insert_new(inner, None, S::Kill).unwrap();
        if !nested {
            tree.insert_new(inner, None, S::Return).unwrap();
        }
        tree.insert_new(root, None, S::Return).unwrap();
        let mut ir = Ir::new().unwrap();
        let syntax = build_asl(
            &mut env,
            &mut ir,
            &graph,
            &mut tree,
            &HostTranslateInfo {
                needs_demote_reorder: true,
                ..Default::default()
            },
            &mut Ranges::default(),
        )
        .unwrap();
        let condition = syntax
            .iter()
            .find_map(|node| {
                if let SyntaxNode::If { condition, .. } = node {
                    Some(*condition)
                } else {
                    None
                }
            })
            .unwrap();
        let Value::Inst(condition) = condition else {
            panic!()
        };
        assert_eq!(ir.inst(condition).unwrap().opcode(), Opcode::ConditionRef);
        assert_eq!(
            syntax
                .iter()
                .filter(|node| matches!(node, SyntaxNode::If { .. }))
                .count(),
            if nested { 2 } else { 1 }
        );
    }
}
