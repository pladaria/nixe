use super::{NUM_REGS, Opcode, Type, Value};
use crate::{
    CompileError, Result,
    arena::{Arena, AssociatedId, BlockId, InstId},
    artifact::{ExecutionMetadata, ShaderInfo},
    diagnostics::FailureCategory,
    session::SessionId,
    types::Stage,
};
use std::collections::{HashSet, VecDeque};

fn invalid(message: impl Into<String>) -> CompileError {
    CompileError::new(FailureCategory::InvalidArgument, message)
}
fn internal(message: impl Into<String>) -> CompileError {
    CompileError::new(FailureCategory::Internal, message)
}

#[derive(Debug)]
enum Operands {
    Fixed([Value; 5]),
    Phi(Vec<(BlockId, Value)>),
}

#[derive(Debug)]
pub struct Instruction {
    opcode: Opcode,
    pub flags: u32,
    use_count: i32,
    pub definition: u32,
    operands: Operands,
    associated: Option<AssociatedId>,
}

impl Instruction {
    pub fn opcode(&self) -> Opcode {
        self.opcode
    }
    pub fn use_count(&self) -> i32 {
        self.use_count
    }
    pub fn has_uses(&self) -> bool {
        self.use_count > 0
    }
    pub fn has_associated_pseudo_operation(&self) -> bool {
        self.associated.is_some()
    }
    pub fn may_have_side_effects(&self) -> bool {
        self.opcode.signature().side_effects
    }
    pub fn is_pseudo(&self) -> bool {
        self.opcode.pseudo_slot().is_some()
    }
    pub fn ty(&self) -> Type {
        if self.opcode == Opcode::Phi {
            Type(self.flags)
        } else {
            self.opcode.signature().result
        }
    }
    pub fn num_args(&self) -> usize {
        match &self.operands {
            Operands::Fixed(_) => self.opcode.signature().arguments.len(),
            Operands::Phi(args) => args.len(),
        }
    }
    pub fn arg(&self, index: usize) -> Result<Value> {
        if index >= self.num_args() {
            return Err(invalid("Instruction operand index out of range"));
        }
        Ok(match &self.operands {
            Operands::Fixed(args) => args[index],
            Operands::Phi(args) => args[index].1,
        })
    }
    pub fn phi_block(&self, index: usize) -> Result<BlockId> {
        match &self.operands {
            Operands::Phi(args) => args
                .get(index)
                .map(|arg| arg.0)
                .ok_or_else(|| invalid("Phi operand index out of range")),
            _ => Err(invalid("Instruction is not a phi")),
        }
    }
}

#[derive(Debug)]
struct AssociatedInsts([Option<InstId>; 4]);

#[derive(Debug)]
pub struct Block {
    instructions: Vec<InstId>,
    predecessors: Vec<BlockId>,
    successors: Vec<BlockId>,
    pub ssa_reg_values: [Value; NUM_REGS],
    pub is_ssa_sealed: bool,
    pub definition: u32,
    pub order: u32,
}

impl Default for Block {
    fn default() -> Self {
        Self {
            instructions: Vec::new(),
            predecessors: Vec::new(),
            successors: Vec::new(),
            ssa_reg_values: [Value::None; NUM_REGS],
            is_ssa_sealed: false,
            definition: 0,
            order: 0,
        }
    }
}

impl Block {
    pub(crate) fn instructions_mut(&mut self) -> &mut Vec<InstId> {
        &mut self.instructions
    }
    pub fn instructions(&self) -> &[InstId] {
        &self.instructions
    }
    pub fn predecessors(&self) -> &[BlockId] {
        &self.predecessors
    }
    pub fn successors(&self) -> &[BlockId] {
        &self.successors
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SyntaxNode {
    Block(BlockId),
    If {
        condition: Value,
        body: BlockId,
        merge: BlockId,
    },
    EndIf {
        merge: BlockId,
    },
    Loop {
        body: BlockId,
        continue_block: BlockId,
        merge: BlockId,
    },
    Repeat {
        condition: Value,
        loop_header: BlockId,
        merge: BlockId,
    },
    Break {
        condition: Value,
        merge: BlockId,
        skip: BlockId,
    },
    Return,
    Unreachable,
}

#[derive(Debug)]
pub struct Program {
    pub syntax: Vec<SyntaxNode>,
    pub blocks: Vec<BlockId>,
    pub post_order: Vec<BlockId>,
    pub info: ShaderInfo,
    pub stage: Stage,
    pub execution: ExecutionMetadata,
}

impl Program {
    pub fn new(stage: Stage) -> Self {
        Self {
            syntax: Vec::new(),
            blocks: Vec::new(),
            post_order: Vec::new(),
            info: ShaderInfo::default(),
            stage,
            execution: ExecutionMetadata::default(),
        }
    }
}

pub struct Ir {
    instructions: Arena<Instruction, InstId>,
    blocks: Arena<Block, BlockId>,
    associated: Arena<AssociatedInsts, AssociatedId>,
}

impl Ir {
    pub fn new() -> Result<Self> {
        let session = SessionId::fresh()?;
        Ok(Self {
            instructions: Arena::new(session),
            blocks: Arena::new(session),
            associated: Arena::new(session),
        })
    }

    pub fn reset(&mut self) -> Result<()> {
        let session = SessionId::fresh()?;
        self.instructions.reset(session);
        self.blocks.reset(session);
        self.associated.reset(session);
        Ok(())
    }

    pub fn inst(&self, id: InstId) -> Result<&Instruction> {
        self.instructions.get(id)
    }
    pub fn inst_mut(&mut self, id: InstId) -> Result<&mut Instruction> {
        self.instructions.get_mut(id)
    }

    pub fn add_external_use(&mut self, id: InstId) -> Result<()> {
        self.inst_mut(id)?.use_count += 1;
        Ok(())
    }
    pub fn block(&self, id: BlockId) -> Result<&Block> {
        self.blocks.get(id)
    }
    pub fn block_mut(&mut self, id: BlockId) -> Result<&mut Block> {
        self.blocks.get_mut(id)
    }
    pub fn new_block(&mut self) -> Result<BlockId> {
        self.blocks.allocate(Block::default())
    }

    /// Resolve lazy replacement chains without recursion on the host stack.
    pub fn resolve(&self, mut value: Value) -> Result<Value> {
        let mut remaining = self.instructions.len();
        while let Value::Inst(id) = value {
            let inst = self.inst(id)?;
            if inst.opcode != Opcode::Identity {
                break;
            }
            if remaining == 0 {
                return Err(internal("Cyclic identity chain"));
            }
            remaining -= 1;
            value = inst.arg(0)?;
        }
        Ok(value)
    }

    pub fn value_type(&self, value: Value) -> Result<Type> {
        Ok(match self.resolve(value)? {
            Value::Inst(id) => self.inst(id)?.ty(),
            immediate => immediate.raw_type(),
        })
    }

    pub fn is_immediate(&self, value: Value) -> Result<bool> {
        Ok(!matches!(self.resolve(value)?, Value::Inst(_)))
    }

    pub fn all_args_immediate(&self, id: InstId) -> Result<bool> {
        let inst = self.inst(id)?;
        if inst.opcode == Opcode::Phi {
            return Err(internal("Cannot test immediate arguments of a phi"));
        }
        for index in 0..inst.num_args() {
            if !self.is_immediate(inst.arg(index)?)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn use_value(&mut self, user: InstId, value: Value) -> Result<()> {
        if self.is_immediate(value)? {
            return Ok(());
        }
        let Value::Inst(producer) = value else {
            return Err(internal("Non-immediate without instruction"));
        };
        self.inst_mut(producer)?.use_count += 1;
        if let Some(slot) = self.inst(user)?.opcode.pseudo_slot() {
            let associated = if let Some(id) = self.inst(producer)?.associated {
                id
            } else {
                let id = self.associated.allocate(AssociatedInsts([None; 4]))?;
                self.inst_mut(producer)?.associated = Some(id);
                id
            };
            let destination = &mut self.associated.get_mut(associated)?.0[slot];
            if destination.is_some() {
                return Err(internal(
                    "Only one pseudo-result per association slot is allowed",
                ));
            }
            *destination = Some(user);
        }
        Ok(())
    }

    fn undo_use(&mut self, user: InstId, value: Value) -> Result<()> {
        if self.is_immediate(value)? {
            return Ok(());
        }
        let Value::Inst(producer) = value else {
            return Err(internal("Non-immediate without instruction"));
        };
        self.inst_mut(producer)?.use_count -= 1;
        if let Some(slot) = self.inst(user)?.opcode.pseudo_slot() {
            let associated = self
                .inst(producer)?
                .associated
                .ok_or_else(|| internal("Missing pseudo-result association"))?;
            let attached = self.associated.get(associated)?.0[slot]
                .ok_or_else(|| internal("Missing pseudo-result"))?;
            if self.inst(attached)?.opcode != self.inst(user)?.opcode {
                return Err(internal("Removing mismatched pseudo-result"));
            }
            self.associated.get_mut(associated)?.0[slot] = None;
        }
        Ok(())
    }

    pub fn associated_pseudo(&self, id: InstId, opcode: Opcode) -> Result<Option<InstId>> {
        let Some(associated) = self.inst(id)?.associated else {
            return Ok(None);
        };
        let slot = opcode
            .pseudo_slot()
            .ok_or_else(|| invalid("Operation is not a pseudo-result"))?;
        let value = self.associated.get(associated)?.0[slot];
        if let Some(value) = value
            && self.inst(value)?.opcode != opcode
        {
            return Err(internal("Mismatched pseudo-result association"));
        }
        Ok(value)
    }

    pub fn set_arg(&mut self, id: InstId, index: usize, value: Value) -> Result<()> {
        let previous = self.inst(id)?.arg(index)?;
        self.undo_use(id, previous)?;
        self.use_value(id, value)?;
        match &mut self.inst_mut(id)?.operands {
            Operands::Fixed(args) => args[index] = value,
            Operands::Phi(args) => args[index].1 = value,
        }
        Ok(())
    }

    /// This checks the operation's declared signature; optional texture operands
    /// use Opaque and can carry None. SSA phis are constructed separately.
    pub fn append(
        &mut self,
        block: BlockId,
        opcode: Opcode,
        args: &[Value],
        flags: u32,
    ) -> Result<InstId> {
        let position = self.block(block)?.instructions.len();
        self.insert(block, position, opcode, args, flags)
    }

    pub fn insert(
        &mut self,
        block: BlockId,
        position: usize,
        opcode: Opcode,
        args: &[Value],
        flags: u32,
    ) -> Result<InstId> {
        if position > self.block(block)?.instructions.len() {
            return Err(invalid("IR insertion position out of range"));
        }
        if opcode == Opcode::Phi {
            return Err(invalid("Use new_phi to construct phi operands"));
        }
        let signature = opcode.signature();
        if args.len() != signature.arguments.len() {
            return Err(invalid(format!("Invalid operand count for {opcode:?}")));
        }
        for (&arg, &expected) in args.iter().zip(signature.arguments) {
            let actual = self.value_type(arg)?;
            if actual != expected && actual != Type::Opaque && expected != Type::Opaque {
                return Err(invalid(format!(
                    "Invalid operand type {actual:?} for {opcode:?}: expected {expected:?}"
                )));
            }
        }
        let id = self.instructions.allocate(Instruction {
            opcode,
            flags,
            use_count: 0,
            definition: 0,
            operands: Operands::Fixed([Value::None; 5]),
            associated: None,
        })?;
        self.block_mut(block)?.instructions.insert(position, id);
        for (index, &arg) in args.iter().enumerate() {
            self.set_arg(id, index, arg)?;
        }
        Ok(id)
    }

    pub fn copy_instruction(
        &mut self,
        block: BlockId,
        position: usize,
        source: InstId,
    ) -> Result<InstId> {
        let source = self.inst(source)?;
        let Operands::Fixed(args) = source.operands else {
            return Err(CompileError::new(
                FailureCategory::Unsupported,
                "Copying a phi instruction is not supported",
            ));
        };
        self.insert(
            block,
            position,
            source.opcode,
            &args[..source.num_args()],
            source.flags,
        )
    }

    pub fn new_phi(&mut self, block: BlockId, ty: Type) -> Result<InstId> {
        self.block(block)?;
        let id = self.instructions.allocate(Instruction {
            opcode: Opcode::Phi,
            flags: ty.0,
            use_count: 0,
            definition: 0,
            operands: Operands::Phi(Vec::new()),
            associated: None,
        })?;
        self.block_mut(block)?.instructions.insert(0, id);
        Ok(id)
    }

    pub fn add_phi_operand(
        &mut self,
        id: InstId,
        predecessor: BlockId,
        value: Value,
    ) -> Result<()> {
        self.block(predecessor)?;
        if self.inst(id)?.opcode != Opcode::Phi {
            return Err(invalid("Instruction is not a phi"));
        }
        self.use_value(id, value)?;
        let Operands::Phi(args) = &mut self.inst_mut(id)?.operands else {
            unreachable!()
        };
        args.push((predecessor, value));
        Ok(())
    }

    pub fn order_phi_args(&mut self, id: InstId) -> Result<()> {
        let (blocks, instructions) = (&self.blocks, &mut self.instructions);
        let Operands::Phi(args) = &mut instructions.get_mut(id)?.operands else {
            return Err(invalid("Instruction is not a phi"));
        };
        args.sort_by_key(|&(block, _)| blocks.get(block).expect("Validated phi predecessor").order);
        Ok(())
    }

    pub fn clear_args(&mut self, id: InstId) -> Result<()> {
        let count = match &self.inst(id)?.operands {
            Operands::Fixed(_) => 5,
            Operands::Phi(args) => args.len(),
        };
        for index in 0..count {
            let value = match &self.inst(id)?.operands {
                Operands::Fixed(args) => args[index],
                Operands::Phi(args) => args[index].1,
            };
            self.undo_use(id, value)?;
        }
        match &mut self.inst_mut(id)?.operands {
            Operands::Fixed(args) => args.fill(Value::None),
            Operands::Phi(args) => args.clear(),
        }
        Ok(())
    }

    pub fn replace_opcode(&mut self, id: InstId, opcode: Opcode) -> Result<()> {
        if self.inst(id)?.opcode == opcode {
            return Ok(());
        }
        if opcode == Opcode::Phi {
            return Err(internal(
                "Cannot transition an existing instruction into a phi",
            ));
        }
        let inst = self.inst_mut(id)?;
        if inst.opcode == Opcode::Phi {
            inst.operands = Operands::Fixed([Value::None; 5]);
        }
        inst.opcode = opcode;
        Ok(())
    }

    pub fn invalidate(&mut self, id: InstId) -> Result<()> {
        self.clear_args(id)?;
        self.replace_opcode(id, Opcode::Void)
    }

    pub fn replace_uses_with(&mut self, id: InstId, value: Value) -> Result<()> {
        if self.resolve(value)? == Value::Inst(id) {
            return Err(internal("Identity cannot refer to itself"));
        }
        self.invalidate(id)?;
        self.replace_opcode(id, Opcode::Identity)?;
        self.set_arg(id, 0, value)
    }

    /// Removing from execution order does not invalidate storage or references.
    pub fn detach(&mut self, block: BlockId, id: InstId) -> Result<()> {
        let list = &mut self.block_mut(block)?.instructions;
        let index = list
            .iter()
            .position(|&item| item == id)
            .ok_or_else(|| invalid("Instruction is not in block"))?;
        list.remove(index);
        Ok(())
    }

    pub fn attach(&mut self, block: BlockId, position: usize, id: InstId) -> Result<()> {
        self.inst(id)?;
        let list = &mut self.block_mut(block)?.instructions;
        if position > list.len() {
            return Err(invalid("IR insertion position out of range"));
        }
        list.insert(position, id);
        Ok(())
    }

    pub fn add_branch(&mut self, from: BlockId, to: BlockId) -> Result<()> {
        if self.block(from)?.successors.contains(&to)
            || self.block(to)?.predecessors.contains(&from)
        {
            return Err(internal("Control-flow edge already inserted"));
        }
        self.block_mut(from)?.successors.push(to);
        self.block_mut(to)?.predecessors.push(from);
        Ok(())
    }

    pub fn post_order(&self, root: &SyntaxNode) -> Result<Vec<BlockId>> {
        let SyntaxNode::Block(first) = root else {
            return Err(internal("First syntax node is not a block"));
        };
        let mut visited = HashSet::from([*first]);
        let mut stack = vec![*first];
        let mut order = Vec::new();
        while let Some(block) = stack.pop() {
            if let Some(&successor) = self
                .block(block)?
                .successors
                .iter()
                .find(|&&successor| visited.insert(successor))
            {
                stack.push(block);
                stack.push(successor);
            } else {
                order.push(block);
            }
        }
        Ok(order)
    }

    pub fn breadth_first_search<T>(
        &self,
        value: Value,
        mut predicate: impl FnMut(InstId, &Instruction) -> Option<T>,
    ) -> Result<Option<T>> {
        let Value::Inst(root) = self.resolve(value)? else {
            return Ok(None);
        };
        let mut visited = Vec::new();
        let mut queue = VecDeque::from([root]);
        while let Some(id) = queue.pop_front() {
            let inst = self.inst(id)?;
            if let Some(value) = predicate(id, inst) {
                return Ok(Some(value));
            }
            for index in (0..inst.num_args()).rev() {
                let Value::Inst(arg) = self.resolve(inst.arg(index)?)? else {
                    continue;
                };
                if !visited.contains(&arg) {
                    visited.push(arg);
                    queue.push_back(arg);
                }
            }
        }
        Ok(None)
    }

    pub fn dump_program(&self, program: &Program) -> Result<String> {
        use std::fmt::Write;
        let mut output = format!("{:?}\n", program.stage);
        for &id in &program.blocks {
            let block = self.block(id)?;
            writeln!(
                output,
                "block {id:?} pred={:?} succ={:?}",
                block.predecessors, block.successors
            )
            .unwrap();
            for &id in &block.instructions {
                let inst = self.inst(id)?;
                write!(
                    output,
                    "  {id:?} = {} flags={:#x}",
                    inst.opcode.name(),
                    inst.flags
                )
                .unwrap();
                for index in 0..inst.num_args() {
                    if inst.opcode == Opcode::Phi {
                        write!(output, " [{:?}]", inst.phi_block(index)?).unwrap();
                    }
                    write!(output, " {:?}", inst.arg(index)?).unwrap();
                }
                writeln!(output, " (uses={})", inst.use_count).unwrap();
            }
        }
        for node in &program.syntax {
            writeln!(output, "{node:?}").unwrap();
        }
        Ok(output)
    }

    /// Expensive structural checks for debug tools and tests, never the draw path.
    pub fn verify(&self, program: &Program) -> Result<()> {
        let blocks: HashSet<_> = program.blocks.iter().copied().collect();
        if blocks.len() != program.blocks.len() {
            return Err(internal("Duplicate program block"));
        }
        let mut instructions = HashSet::new();
        for &id in &program.blocks {
            let block = self.block(id)?;
            for &successor in &block.successors {
                if !blocks.contains(&successor)
                    || !self.block(successor)?.predecessors.contains(&id)
                {
                    return Err(internal("Successor/predecessor mismatch"));
                }
            }
            for &predecessor in &block.predecessors {
                if !blocks.contains(&predecessor)
                    || !self.block(predecessor)?.successors.contains(&id)
                {
                    return Err(internal("Predecessor/successor mismatch"));
                }
            }
            let mut non_phi = false;
            for &id in &block.instructions {
                if !instructions.insert(id) {
                    return Err(internal("Instruction occurs in multiple list positions"));
                }
                let inst = self.inst(id)?;
                if inst.opcode == Opcode::Phi {
                    if non_phi {
                        return Err(internal("Phi after non-phi instruction"));
                    }
                    for index in 0..inst.num_args() {
                        if !block.predecessors.contains(&inst.phi_block(index)?) {
                            return Err(internal("Phi input is not an immediate predecessor"));
                        }
                    }
                } else {
                    non_phi = true;
                }
                for index in 0..inst.num_args() {
                    let value = inst.arg(index)?;
                    let actual = self.value_type(value)?;
                    let expected = if inst.opcode == Opcode::Phi {
                        inst.ty()
                    } else {
                        inst.opcode.signature().arguments[index]
                    };
                    if actual != expected && actual != Type::Opaque && expected != Type::Opaque {
                        return Err(internal(format!(
                            "Operand type mismatch in {:?}",
                            inst.opcode
                        )));
                    }
                }
                if inst.use_count < 0 {
                    return Err(internal("Negative instruction use count"));
                }
                if let Some(associated) = inst.associated {
                    for pseudo in self.associated.get(associated)?.0.iter().flatten() {
                        let pseudo = self.inst(*pseudo)?;
                        if !pseudo.is_pseudo() || pseudo.arg(0)? != Value::Inst(id) {
                            return Err(internal("Invalid pseudo-result association"));
                        }
                    }
                }
            }
        }
        for node in &program.syntax {
            let (targets, condition): (Vec<_>, _) = match *node {
                SyntaxNode::Block(block) => (vec![block], None),
                SyntaxNode::If {
                    condition,
                    body,
                    merge,
                } => (vec![body, merge], Some(condition)),
                SyntaxNode::EndIf { merge } => (vec![merge], None),
                SyntaxNode::Loop {
                    body,
                    continue_block,
                    merge,
                } => (vec![body, continue_block, merge], None),
                SyntaxNode::Repeat {
                    condition,
                    loop_header,
                    merge,
                } => (vec![loop_header, merge], Some(condition)),
                SyntaxNode::Break {
                    condition,
                    merge,
                    skip,
                } => (vec![merge, skip], Some(condition)),
                SyntaxNode::Return | SyntaxNode::Unreachable => (vec![], None),
            };
            if targets.iter().any(|id| !blocks.contains(id)) {
                return Err(internal("Syntax target is not a program block"));
            }
            if let Some(condition) = condition
                && self.value_type(condition)? != Type::U1
            {
                return Err(internal("Syntax condition is not boolean"));
            }
        }
        Ok(())
    }
}
