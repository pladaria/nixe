use super::flow::{EndClass, FlowGraph, FunctionId};
use crate::{
    CompileError, Result,
    arena::{Arena, FlowBlockId, StatementId},
    diagnostics::FailureCategory,
    ir::{Condition, Reg},
    session::SessionId,
};
use std::collections::HashMap;

fn internal(message: impl Into<String>) -> CompileError {
    CompileError::new(FailureCategory::Internal, message)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StatementKind {
    Code(Option<FlowBlockId>),
    Goto {
        condition: StatementId,
        label: StatementId,
    },
    Label(u32),
    If(StatementId),
    Loop(StatementId),
    Break(StatementId),
    Return,
    Kill,
    Unreachable,
    Function,
    Identity(Condition),
    Not(StatementId),
    Or(StatementId, StatementId),
    SetVariable {
        id: u32,
        value: StatementId,
    },
    SetIndirectVariable {
        reg: Reg,
        offset: i32,
    },
    Variable(u32),
    IndirectCondition(u32),
}

impl StatementKind {
    pub fn has_children(self) -> bool {
        matches!(self, Self::If(_) | Self::Loop(_) | Self::Function)
    }
}

#[derive(Debug)]
pub struct Statement {
    pub kind: StatementKind,
    pub parent: Option<StatementId>,
    pub previous: Option<StatementId>,
    pub next: Option<StatementId>,
    pub first: Option<StatementId>,
    pub last: Option<StatementId>,
}

/// Intrusive lists indexed into reusable storage preserve node identity across
/// range movement. Allocation order is independent of statement order.
pub struct StatementTree {
    nodes: Arena<Statement, StatementId>,
    pub root: StatementId,
}

impl StatementTree {
    pub fn new() -> Result<Self> {
        let mut nodes = Arena::new(SessionId::fresh()?);
        let root = nodes.allocate(Statement {
            kind: StatementKind::Function,
            parent: None,
            previous: None,
            next: None,
            first: None,
            last: None,
        })?;
        Ok(Self { nodes, root })
    }
    pub fn node(&self, id: StatementId) -> Result<&Statement> {
        self.nodes.get(id)
    }
    pub fn node_mut(&mut self, id: StatementId) -> Result<&mut Statement> {
        self.nodes.get_mut(id)
    }
    pub fn allocate(&mut self, kind: StatementKind) -> Result<StatementId> {
        self.nodes.allocate(Statement {
            kind,
            parent: None,
            previous: None,
            next: None,
            first: None,
            last: None,
        })
    }
    pub fn insert(
        &mut self,
        parent: StatementId,
        before: Option<StatementId>,
        node: StatementId,
    ) -> Result<()> {
        if !self.node(parent)?.kind.has_children() {
            return Err(internal("Parent cannot contain statements"));
        }
        if self.node(node)?.parent.is_some() {
            return Err(internal("Statement is already linked"));
        }
        let previous = if let Some(before) = before {
            if self.node(before)?.parent != Some(parent) {
                return Err(internal("Insertion point is in another list"));
            }
            self.node(before)?.previous
        } else {
            self.node(parent)?.last
        };
        let item = self.node_mut(node)?;
        item.parent = Some(parent);
        item.previous = previous;
        item.next = before;
        if let Some(previous) = previous {
            self.node_mut(previous)?.next = Some(node);
        } else {
            self.node_mut(parent)?.first = Some(node);
        }
        if let Some(before) = before {
            self.node_mut(before)?.previous = Some(node);
        } else {
            self.node_mut(parent)?.last = Some(node);
        }
        Ok(())
    }
    pub fn insert_new(
        &mut self,
        parent: StatementId,
        before: Option<StatementId>,
        kind: StatementKind,
    ) -> Result<StatementId> {
        let node = self.allocate(kind)?;
        self.insert(parent, before, node)?;
        Ok(node)
    }
    pub fn detach(&mut self, id: StatementId) -> Result<()> {
        let node = self.node(id)?;
        let parent = node
            .parent
            .ok_or_else(|| internal("Statement is not linked"))?;
        let (previous, next) = (node.previous, node.next);
        if let Some(previous) = previous {
            self.node_mut(previous)?.next = next;
        } else {
            self.node_mut(parent)?.first = next;
        }
        if let Some(next) = next {
            self.node_mut(next)?.previous = previous;
        } else {
            self.node_mut(parent)?.last = previous;
        }
        let node = self.node_mut(id)?;
        node.parent = None;
        node.previous = None;
        node.next = None;
        Ok(())
    }
    fn move_range(
        &mut self,
        mut first: Option<StatementId>,
        end: Option<StatementId>,
        destination: StatementId,
    ) -> Result<()> {
        while first != end {
            let node = first.ok_or_else(|| internal("Invalid statement range"))?;
            first = self.node(node)?.next;
            self.detach(node)?;
            self.insert(destination, None, node)?;
        }
        Ok(())
    }
    fn parent(&self, node: StatementId) -> Result<StatementId> {
        self.node(node)?
            .parent
            .ok_or_else(|| internal("Statement has no parent"))
    }
    fn label_id(&self, node: StatementId) -> Result<u32> {
        match self.node(node)?.kind {
            StatementKind::Label(id) => Ok(id),
            _ => Err(internal("Expected label")),
        }
    }
    fn goto(&self, node: StatementId) -> Result<(StatementId, StatementId)> {
        match self.node(node)?.kind {
            StatementKind::Goto { condition, label } => Ok((condition, label)),
            _ => Err(internal("Expected goto")),
        }
    }
    fn level(&self, node: StatementId) -> Result<usize> {
        let mut level = 0;
        let mut parent = self.node(node)?.parent;
        while let Some(id) = parent {
            level += 1;
            parent = self.node(id)?.parent;
        }
        Ok(level)
    }
    fn directly_related(&self, mut a: StatementId, mut b: StatementId) -> Result<bool> {
        let (mut al, mut bl) = (self.level(a)?, self.level(b)?);
        while al > bl {
            a = self.parent(a)?;
            al -= 1;
        }
        while bl > al {
            b = self.parent(b)?;
            bl -= 1;
        }
        Ok(self.node(a)?.parent == self.node(b)?.parent)
    }
    fn sibling_from_nephew(
        &self,
        uncle: StatementId,
        mut nephew: StatementId,
    ) -> Result<StatementId> {
        let parent = self.parent(uncle)?;
        while self.node(nephew)?.parent != Some(parent) {
            nephew = self.parent(nephew)?;
        }
        Ok(nephew)
    }
    fn ordered(&self, left: StatementId, right: StatementId) -> Result<bool> {
        let mut current = Some(right);
        while let Some(node) = current {
            if node == left {
                return Ok(false);
            }
            current = self.node(node)?.next;
        }
        Ok(true)
    }

    pub fn from_flow(graph: &FlowGraph) -> Result<Self> {
        let mut tree = Self::new()?;
        let mut label_id = 0;
        let mut gotos = Vec::new();
        tree.build_function(
            graph,
            FunctionId(0),
            &mut label_id,
            &mut gotos,
            None,
            None,
            &mut Vec::new(),
        )?;
        for node in gotos.into_iter().rev() {
            tree.remove_goto(node)?;
        }
        Ok(tree)
    }

    #[allow(clippy::too_many_arguments)]
    fn build_function(
        &mut self,
        graph: &FlowGraph,
        function: FunctionId,
        label_id: &mut u32,
        gotos: &mut Vec<StatementId>,
        before: Option<StatementId>,
        return_label: Option<StatementId>,
        calls: &mut Vec<FunctionId>,
    ) -> Result<()> {
        if calls.contains(&function) {
            return Err(CompileError::new(
                FailureCategory::Unsupported,
                "Recursive shader calls",
            ));
        }
        calls.push(function);
        let false_expr = self.allocate(StatementKind::Identity(Condition::always(false)))?;
        let mut labels = HashMap::new();
        for &block in graph.functions[function.0].blocks.values() {
            let label = self.insert_new(self.root, before, StatementKind::Label(*label_id))?;
            *label_id = label_id
                .checked_add(1)
                .ok_or_else(|| internal("Too many statement labels"))?;
            labels.insert(block, label);
        }
        for &block_id in graph.functions[function.0].blocks.values() {
            let block = graph.block(block_id)?;
            let label = labels[&block_id];
            let ip = self.node(label)?.next;
            let variable_id = self.label_id(label)?;
            self.insert_new(
                self.root,
                self.node(self.root)?.first,
                StatementKind::SetVariable {
                    id: variable_id,
                    value: false_expr,
                },
            )?;
            self.insert_new(
                self.root,
                ip,
                StatementKind::SetVariable {
                    id: variable_id,
                    value: false_expr,
                },
            )?;
            self.insert_new(self.root, ip, StatementKind::Code(Some(block_id)))?;
            let lookup = |target: Option<FlowBlockId>| -> Result<StatementId> {
                target
                    .and_then(|id| labels.get(&id).copied())
                    .ok_or_else(|| internal("Flow target has no local label"))
            };
            match block.end_class {
                EndClass::Branch => {
                    let always = self.allocate(StatementKind::Identity(Condition::always(true)))?;
                    if block.condition == Condition::always(true) {
                        gotos.push(self.insert_new(
                            self.root,
                            ip,
                            StatementKind::Goto {
                                condition: always,
                                label: lookup(block.branch_true)?,
                            },
                        )?);
                    } else if block.condition == Condition::always(false) {
                        gotos.push(self.insert_new(
                            self.root,
                            ip,
                            StatementKind::Goto {
                                condition: always,
                                label: lookup(block.branch_false)?,
                            },
                        )?);
                    } else {
                        let condition = self.allocate(StatementKind::Identity(block.condition))?;
                        gotos.push(self.insert_new(
                            self.root,
                            ip,
                            StatementKind::Goto {
                                condition,
                                label: lookup(block.branch_true)?,
                            },
                        )?);
                        gotos.push(self.insert_new(
                            self.root,
                            ip,
                            StatementKind::Goto {
                                condition: always,
                                label: lookup(block.branch_false)?,
                            },
                        )?);
                    }
                }
                EndClass::IndirectBranch => {
                    self.insert_new(
                        self.root,
                        ip,
                        StatementKind::SetIndirectVariable {
                            reg: block.branch_reg,
                            offset: block.branch_offset,
                        },
                    )?;
                    for branch in &block.indirect_branches {
                        let condition =
                            self.allocate(StatementKind::IndirectCondition(branch.address.0))?;
                        gotos.push(self.insert_new(
                            self.root,
                            ip,
                            StatementKind::Goto {
                                condition,
                                label: lookup(Some(branch.block))?,
                            },
                        )?);
                    }
                    self.insert_new(self.root, ip, StatementKind::Unreachable)?;
                }
                EndClass::Call => {
                    let callee = block
                        .function_call
                        .ok_or_else(|| internal("Call lacks callee"))?;
                    self.build_function(
                        graph,
                        callee,
                        label_id,
                        gotos,
                        ip,
                        Some(lookup(block.return_block)?),
                        calls,
                    )?;
                }
                EndClass::Exit => {
                    self.insert_new(self.root, ip, StatementKind::Return)?;
                }
                EndClass::Return => {
                    let condition = self.allocate(StatementKind::Identity(block.condition))?;
                    let label = return_label
                        .ok_or_else(|| internal("Return outside a called shader function"))?;
                    gotos.push(self.insert_new(
                        self.root,
                        ip,
                        StatementKind::Goto { condition, label },
                    )?);
                }
                EndClass::Kill => {
                    self.insert_new(self.root, ip, StatementKind::Kill)?;
                }
            }
        }
        calls.pop();
        Ok(())
    }

    pub fn remove_goto(&mut self, mut goto: StatementId) -> Result<()> {
        let (_, label) = self.goto(goto)?;
        while !self.directly_related(goto, label)? {
            goto = self.move_outward(goto)?;
        }
        let label_level = self.level(label)?;
        let mut goto_level = self.level(goto)?;
        if goto_level > label_level {
            while goto_level > label_level {
                goto = self.move_outward(goto)?;
                goto_level -= 1;
            }
        } else {
            let sibling = self.sibling_from_nephew(goto, label)?;
            if self.ordered(sibling, goto)? {
                goto = self.lift(goto)?;
            }
            while goto_level < label_level {
                goto = self.move_inward(goto)?;
                goto_level += 1;
            }
        }
        if self.node(goto)?.next == Some(label) {
            return self.detach(goto);
        }
        let parent = self.parent(goto)?;
        let (condition, _) = self.goto(goto)?;
        if self.ordered(goto, label)? {
            let not = self.allocate(StatementKind::Not(condition))?;
            let selection = self.allocate(StatementKind::If(not))?;
            self.move_range(self.node(goto)?.next, Some(label), selection)?;
            self.insert(parent, Some(goto), selection)?;
        } else {
            let loop_node = self.allocate(StatementKind::Loop(condition))?;
            self.move_range(Some(label), Some(goto), loop_node)?;
            self.insert(parent, Some(goto), loop_node)?;
        }
        self.detach(goto)
    }

    fn move_outward(&mut self, goto: StatementId) -> Result<StatementId> {
        let parent = self.parent(goto)?;
        let grandparent = self.parent(parent)?;
        let (condition, label) = self.goto(goto)?;
        let variable_id = self.label_id(label)?;
        self.insert_new(
            parent,
            Some(goto),
            StatementKind::SetVariable {
                id: variable_id,
                value: condition,
            },
        )?;
        let variable = self.allocate(StatementKind::Variable(variable_id))?;
        match self.node(parent)?.kind {
            StatementKind::If(_) => {
                let not = self.allocate(StatementKind::Not(variable))?;
                let selection = self.allocate(StatementKind::If(not))?;
                self.move_range(self.node(goto)?.next, None, selection)?;
                let first = self.node(selection)?.first.ok_or_else(|| {
                    CompileError::new(
                        FailureCategory::UnsupportedReferenceCase,
                        "Outward-if movement would remove the first node of an empty range",
                    )
                })?;
                self.detach(first)?;
                self.insert(parent, Some(goto), selection)?;
            }
            StatementKind::Loop(_) => {
                self.insert_new(parent, Some(goto), StatementKind::Break(variable))?;
            }
            _ => return Err(internal("Invalid outward movement")),
        }
        self.detach(goto)?;
        let condition = self.allocate(StatementKind::Variable(variable_id))?;
        self.insert_new(
            grandparent,
            self.node(parent)?.next,
            StatementKind::Goto { condition, label },
        )
    }

    fn move_inward(&mut self, goto: StatementId) -> Result<StatementId> {
        let parent = self.parent(goto)?;
        let (condition, label) = self.goto(goto)?;
        let nested = self.sibling_from_nephew(goto, label)?;
        let variable_id = self.label_id(label)?;
        self.insert_new(
            parent,
            Some(goto),
            StatementKind::SetVariable {
                id: variable_id,
                value: condition,
            },
        )?;
        let variable = self.allocate(StatementKind::Variable(variable_id))?;
        let not = self.allocate(StatementKind::Not(variable))?;
        let selection = self.allocate(StatementKind::If(not))?;
        self.move_range(self.node(goto)?.next, Some(nested), selection)?;
        if self.node(selection)?.first.is_some() {
            self.insert(parent, Some(goto), selection)?;
        }
        self.detach(goto)?;
        match self.node(nested)?.kind {
            StatementKind::If(old_condition) => {
                let condition = self.allocate(StatementKind::Or(variable, old_condition))?;
                self.node_mut(nested)?.kind = StatementKind::If(condition);
            }
            StatementKind::Loop(_) => (),
            _ => return Err(internal("Invalid inward movement")),
        }
        self.insert_new(
            nested,
            self.node(nested)?.first,
            StatementKind::Goto {
                condition: variable,
                label,
            },
        )
    }

    fn lift(&mut self, goto: StatementId) -> Result<StatementId> {
        let parent = self.parent(goto)?;
        let (condition, label) = self.goto(goto)?;
        let nested = self.sibling_from_nephew(goto, label)?;
        let variable_id = self.label_id(label)?;
        let variable = self.allocate(StatementKind::Variable(variable_id))?;
        let loop_node = self.allocate(StatementKind::Loop(variable))?;
        self.move_range(Some(nested), Some(goto), loop_node)?;
        let mut child = self.node(loop_node)?.first;
        while let Some(node) = child {
            if matches!(self.node(node)?.kind, StatementKind::Break(_)) {
                return Err(CompileError::new(
                    FailureCategory::Unsupported,
                    "Capturing statement with break nodes",
                ));
            }
            child = self.node(node)?.next;
        }
        self.insert(parent, Some(goto), loop_node)?;
        let new_goto = self.insert_new(
            loop_node,
            self.node(loop_node)?.first,
            StatementKind::Goto {
                condition: variable,
                label,
            },
        )?;
        self.insert_new(
            loop_node,
            None,
            StatementKind::SetVariable {
                id: variable_id,
                value: condition,
            },
        )?;
        self.detach(goto)?;
        Ok(new_goto)
    }

    pub fn verify(&self) -> Result<()> {
        let mut parents = vec![self.root];
        let mut visited = std::collections::HashSet::new();
        while let Some(parent) = parents.pop() {
            let mut child = self.node(parent)?.first;
            let mut previous = None;
            while let Some(id) = child {
                if !visited.insert(id) {
                    return Err(internal("Duplicate or cyclic statement list"));
                }
                let node = self.node(id)?;
                if node.parent != Some(parent) || node.previous != previous {
                    return Err(internal("Inconsistent statement links"));
                }
                if matches!(node.kind, StatementKind::Goto { .. }) {
                    return Err(internal("Residual goto after structuring"));
                }
                if node.kind.has_children() {
                    parents.push(id);
                }
                previous = Some(id);
                child = node.next;
            }
            if previous != self.node(parent)?.last {
                return Err(internal("Inconsistent statement list tail"));
            }
        }
        Ok(())
    }
}
