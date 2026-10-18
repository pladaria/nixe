use super::{
    flow::FlowGraph,
    location::Location,
    structure::{StatementKind, StatementTree},
};
use crate::{
    CompileError, Environment, HostTranslateInfo, Result,
    arena::{BlockId, StatementId},
    diagnostics::FailureCategory,
    ir::{Emitter, Ir, Opcode, SyntaxNode, Value},
};

fn internal(message: impl Into<String>) -> CompileError {
    CompileError::new(FailureCategory::Internal, message)
}
fn undefined(message: impl Into<String>) -> CompileError {
    CompileError::new(FailureCategory::UnsupportedReferenceCase, message)
}

/// A statically dispatched visitor translates one discovered range into its
/// structured block. No callbacks are stored in programs or emitted artifacts.
pub trait CodeTranslator<E: Environment> {
    fn translate(
        &mut self,
        environment: &mut E,
        ir: &mut Ir,
        block: BlockId,
        begin: Location,
        end: Location,
    ) -> Result<()>;
}

fn emit_expression(
    tree: &StatementTree,
    emit: &mut Emitter<'_>,
    expression: StatementId,
) -> Result<Value> {
    match tree.node(expression)?.kind {
        StatementKind::Identity(condition) => emit.condition(condition),
        StatementKind::Not(expression) => {
            let value = emit_expression(tree, emit, expression)?;
            emit.logical_not(value)
        }
        StatementKind::Or(a, b) => {
            let a = emit_expression(tree, emit, a)?;
            let b = emit_expression(tree, emit, b)?;
            emit.logical_or(a, b)
        }
        StatementKind::Variable(id) => emit.get_goto_variable(id),
        StatementKind::IndirectCondition(address) => {
            let target = emit.get_indirect_branch_variable()?;
            emit.i_equal(target, Value::U32(address))
        }
        _ => Err(internal("Statement is not a boolean expression")),
    }
}

struct Builder<'a, E, T> {
    environment: &'a mut E,
    ir: &'a mut Ir,
    graph: &'a FlowGraph,
    tree: &'a mut StatementTree,
    translator: &'a mut T,
    syntax: Vec<SyntaxNode>,
    uses_demote: bool,
}

impl<E: Environment, T: CodeTranslator<E>> Builder<'_, E, T> {
    fn ensure_block(&mut self, current: &mut Option<BlockId>) -> Result<BlockId> {
        if let Some(block) = *current {
            return Ok(block);
        }
        let block = self.ir.new_block()?;
        self.syntax.push(SyntaxNode::Block(block));
        *current = Some(block);
        Ok(block)
    }
    fn merge_block(&mut self, statement: StatementId) -> Result<BlockId> {
        let mut next = self.tree.node(statement)?.next;
        let mut found = false;
        while let Some(node) = next {
            let node = self.tree.node(node)?;
            if node.kind.has_children() {
                break;
            }
            if matches!(node.kind, StatementKind::Code(_)) {
                found = true;
                break;
            }
            next = node.next;
        }
        if !found {
            let parent = self
                .tree
                .node(statement)?
                .parent
                .ok_or_else(|| internal("Merge statement has no parent"))?;
            self.tree.insert_new(
                parent,
                self.tree.node(statement)?.next,
                StatementKind::Code(None),
            )?;
        }
        self.ir.new_block()
    }
    fn condition(&mut self, block: BlockId, expression: StatementId) -> Result<Value> {
        let mut emit = Emitter::new(self.ir, block)?;
        let value = emit_expression(self.tree, &mut emit, expression)?;
        emit.condition_ref(value)
    }
    fn block_at(&self, index: usize) -> Result<BlockId> {
        match self.syntax.get(index) {
            Some(SyntaxNode::Block(block)) => Ok(*block),
            _ => Err(undefined("Structured body does not begin with an IR block")),
        }
    }
    fn visit(
        &mut self,
        parent: StatementId,
        break_block: Option<BlockId>,
        fallthrough: Option<BlockId>,
    ) -> Result<()> {
        let mut current = None;
        let mut next = self.tree.node(parent)?.first;
        while let Some(statement) = next {
            match self.tree.node(statement)?.kind {
                StatementKind::Label(_) => (),
                StatementKind::Code(flow) => {
                    let block = self.ensure_block(&mut current)?;
                    if let Some(flow) = flow {
                        let flow = self.graph.block(flow)?;
                        if flow.begin != flow.end {
                            self.translator.translate(
                                self.environment,
                                self.ir,
                                block,
                                flow.begin.instruction()?,
                                flow.end.instruction()?,
                            )?;
                        }
                    }
                }
                StatementKind::SetVariable { id, value } => {
                    let block = self.ensure_block(&mut current)?;
                    let mut emit = Emitter::new(self.ir, block)?;
                    let value = emit_expression(self.tree, &mut emit, value)?;
                    emit.set_goto_variable(id, value)?;
                }
                StatementKind::SetIndirectVariable { reg, offset } => {
                    let block = self.ensure_block(&mut current)?;
                    let mut emit = Emitter::new(self.ir, block)?;
                    let reg = emit.get_reg(reg)?;
                    let address = emit.i_add(reg, Value::U32(offset as u32))?;
                    emit.set_indirect_branch_variable(address)?;
                }
                StatementKind::If(expression) => {
                    let header = self.ensure_block(&mut current)?;
                    let merge = self.merge_block(statement)?;
                    let condition = self.condition(header, expression)?;
                    let index = self.syntax.len();
                    self.syntax.push(SyntaxNode::Unreachable);
                    let body_index = self.syntax.len();
                    self.visit(statement, break_block, Some(merge))?;
                    let body = self.block_at(body_index)?;
                    self.ir.add_branch(header, body)?;
                    self.ir.add_branch(header, merge)?;
                    current = Some(merge);
                    self.syntax[index] = SyntaxNode::If {
                        condition,
                        body,
                        merge,
                    };
                    self.syntax.push(SyntaxNode::EndIf { merge });
                    self.syntax.push(SyntaxNode::Block(merge));
                }
                StatementKind::Loop(expression) => {
                    let header = self.ir.new_block()?;
                    if let Some(current) = current {
                        self.ir.add_branch(current, header)?;
                    }
                    self.syntax.push(SyntaxNode::Block(header));
                    let continue_block = self.ir.new_block()?;
                    let merge = self.merge_block(statement)?;
                    let index = self.syntax.len();
                    self.syntax.push(SyntaxNode::Unreachable);
                    let body_index = self.syntax.len();
                    self.visit(statement, Some(merge), Some(continue_block))?;
                    let condition = self.condition(continue_block, expression)?;
                    let body = self.block_at(body_index)?;
                    self.ir.add_branch(header, body)?;
                    self.ir.add_branch(continue_block, header)?;
                    self.ir.add_branch(continue_block, merge)?;
                    current = Some(merge);
                    self.syntax[index] = SyntaxNode::Loop {
                        body,
                        continue_block,
                        merge,
                    };
                    self.syntax.push(SyntaxNode::Block(continue_block));
                    self.syntax.push(SyntaxNode::Repeat {
                        condition,
                        loop_header: header,
                        merge,
                    });
                    self.syntax.push(SyntaxNode::Block(merge));
                }
                StatementKind::Break(expression) => {
                    let block = self.ensure_block(&mut current)?;
                    let merge = break_block.ok_or_else(|| internal("Break outside loop"))?;
                    let skip = self.merge_block(statement)?;
                    let condition = self.condition(block, expression)?;
                    self.ir.add_branch(block, merge)?;
                    self.ir.add_branch(block, skip)?;
                    current = Some(skip);
                    self.syntax.push(SyntaxNode::Break {
                        condition,
                        merge,
                        skip,
                    });
                    self.syntax.push(SyntaxNode::Block(skip));
                }
                StatementKind::Return => {
                    let block = self.ensure_block(&mut current)?;
                    let epilogue = self.ir.new_block()?;
                    Emitter::new(self.ir, epilogue)?.epilogue()?;
                    self.ir.add_branch(block, epilogue)?;
                    self.syntax.push(SyntaxNode::Block(epilogue));
                    self.syntax.push(SyntaxNode::Return);
                    current = None;
                }
                StatementKind::Kill => {
                    let block = self.ensure_block(&mut current)?;
                    let merge = self.merge_block(statement)?;
                    Emitter::new(self.ir, block)?.demote_to_helper_invocation()?;
                    self.ir.add_branch(block, merge)?;
                    current = Some(merge);
                    self.syntax.push(SyntaxNode::Block(merge));
                    self.uses_demote = true;
                }
                StatementKind::Unreachable => {
                    self.ensure_block(&mut current)?;
                    current = None;
                    self.syntax.push(SyntaxNode::Unreachable);
                }
                _ => return Err(internal("Unexpected statement during structured emission")),
            }
            next = self.tree.node(statement)?.next;
        }
        if let Some(current) = current {
            if let Some(fallthrough) = fallthrough {
                self.ir.add_branch(current, fallthrough)?;
            } else {
                self.syntax.push(SyntaxNode::Unreachable);
            }
        }
        Ok(())
    }

    fn combine_demotes(&mut self) -> Result<()> {
        let mut demote_blocks = Vec::new();
        let mut epilogues = 0;
        let mut depth = 0u32;
        for node in &self.syntax {
            match node {
                SyntaxNode::If { .. } => depth += 1,
                SyntaxNode::EndIf { .. } => {
                    depth = depth
                        .checked_sub(1)
                        .ok_or_else(|| internal("Unbalanced structured selection"))?;
                }
                _ => (),
            }
            let SyntaxNode::Block(block) = node else {
                continue;
            };
            if depth > 1 {
                continue;
            }
            for &instruction in self.ir.block(*block)?.instructions() {
                match self.ir.inst(instruction)?.opcode() {
                    Opcode::DemoteToHelperInvocation => {
                        demote_blocks.push(*block);
                        break;
                    }
                    Opcode::Epilogue => epilogues += 1,
                    _ => (),
                }
            }
        }
        if demote_blocks.is_empty() || epilogues > 1 {
            return Ok(());
        }
        let mut conditions = Vec::new();
        let mut search_start = 0;
        for &block in &demote_blocks {
            let Some(index) =
                self.syntax
                    .iter()
                    .enumerate()
                    .skip(search_start)
                    .find_map(|(index, node)| {
                        matches!(node, SyntaxNode::If {body,..} if *body == block).then_some(index)
                    })
            else {
                return Ok(());
            };
            let SyntaxNode::If {
                condition, merge, ..
            } = self.syntax[index]
            else {
                unreachable!()
            };
            conditions.push(condition);
            search_start = index;
            self.syntax.remove(index);
            let body = self
                .syntax
                .iter()
                .enumerate()
                .skip(index)
                .find_map(|(index, node)| {
                    matches!(node, SyntaxNode::Block(id) if *id == block).then_some(index)
                })
                .ok_or_else(|| undefined("Demote selection has no body node"))?;
            self.syntax.remove(body);
            let end = self
                .syntax
                .iter()
                .enumerate()
                .skip(body)
                .find_map(|(index, node)| {
                    matches!(node, SyntaxNode::EndIf {merge:target} if *target == merge)
                        .then_some(index)
                })
                .ok_or_else(|| undefined("Demote selection has no end node"))?;
            self.syntax.remove(end);
        }
        let mut epilogue = None;
        for (index, node) in self.syntax.iter().enumerate().rev() {
            let SyntaxNode::Block(block) = node else {
                continue;
            };
            if self.ir.block(*block)?.instructions().iter().any(|&id| {
                self.ir
                    .inst(id)
                    .is_ok_and(|inst| inst.opcode() == Opcode::Epilogue)
            }) {
                epilogue = Some((index, *block));
                break;
            }
        }
        let (index, merge) =
            epilogue.ok_or_else(|| undefined("Demote reordering requires an epilogue"))?;
        let previous = index
            .checked_sub(1)
            .ok_or_else(|| undefined("Demote reordering requires a preceding block"))?;
        let block = self.block_at(previous)?;
        let mut emit = Emitter::new(self.ir, block)?;
        let mut condition = Value::U1(false);
        for value in conditions {
            condition = emit.logical_or(condition, value)?;
        }
        let Value::Inst(id) = condition else {
            return Err(internal("Empty demote condition"));
        };
        self.ir.add_external_use(id)?;
        self.syntax.splice(
            index..index,
            [
                SyntaxNode::If {
                    condition,
                    body: demote_blocks[0],
                    merge,
                },
                SyntaxNode::Block(demote_blocks[0]),
                SyntaxNode::EndIf { merge },
            ],
        );
        Ok(())
    }
}

pub fn build_asl<E: Environment, T: CodeTranslator<E>>(
    environment: &mut E,
    ir: &mut Ir,
    graph: &FlowGraph,
    tree: &mut StatementTree,
    host: &HostTranslateInfo,
    translator: &mut T,
) -> Result<Vec<SyntaxNode>> {
    let root = tree.root;
    let mut builder = Builder {
        environment,
        ir,
        graph,
        tree,
        translator,
        syntax: Vec::new(),
        uses_demote: false,
    };
    builder.visit(root, None, None)?;
    let first = builder.block_at(0)?;
    builder.ir.insert(first, 0, Opcode::Prologue, &[], 0)?;
    if builder.uses_demote && host.needs_demote_reorder {
        builder.combine_demotes()?;
    }
    Ok(builder.syntax)
}
