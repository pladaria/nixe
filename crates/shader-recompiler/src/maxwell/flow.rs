use super::{
    decode::{Opcode, fetch},
    indirect::track_indirect_table,
    instruction::Instruction,
    location::{FlowLocation, Location},
};
use crate::{
    CompileError, Environment, Result,
    arena::{Arena, FlowBlockId},
    diagnostics::FailureCategory,
    ir::{Condition, FlowTest, Pred, Reg},
    session::SessionId,
    types::{ByteOffset, ProgramOffset},
};
use std::{
    collections::BTreeMap,
    ops::Bound::{Excluded, Unbounded},
};

fn unsupported(message: impl Into<String>) -> CompileError {
    CompileError::new(FailureCategory::Unsupported, message)
}
fn internal(message: impl Into<String>) -> CompileError {
    CompileError::new(FailureCategory::Internal, message)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Token {
    Ssy,
    Pbk,
    Pexit,
    Pret,
    Pcnt,
    Plongjmp,
}

impl Token {
    fn from_opcode(opcode: Opcode) -> Result<Self> {
        Ok(match opcode {
            Opcode::SSY | Opcode::SYNC => Self::Ssy,
            Opcode::PBK | Opcode::BRK => Self::Pbk,
            Opcode::PEXIT | Opcode::EXIT => Self::Pexit,
            Opcode::PRET | Opcode::RET | Opcode::CAL => Self::Pret,
            Opcode::PCNT | Opcode::CONT => Self::Pcnt,
            Opcode::PLONGJMP | Opcode::LONGJMP => Self::Plongjmp,
            _ => return Err(internal("Opcode does not identify a flow token")),
        })
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TokenStack(Vec<(Token, Location)>);

impl TokenStack {
    pub fn push(&mut self, token: Token, target: Location) {
        self.0.push((token, target));
    }
    pub fn peek(&self, token: Token) -> Option<Location> {
        self.0
            .iter()
            .rev()
            .find(|entry| entry.0 == token)
            .map(|entry| entry.1)
    }
    /// Popping a token also discards the more recent entries above it.
    pub fn pop(&self, token: Token) -> Result<(Location, Self)> {
        let position = self
            .0
            .iter()
            .rposition(|entry| entry.0 == token)
            .ok_or_else(|| internal(format!("Flow token {token:?} could not be found")))?;
        Ok((self.0[position].1, Self(self.0[..position].to_vec())))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EndClass {
    Branch,
    IndirectBranch,
    Call,
    Exit,
    Return,
    Kill,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FunctionId(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndirectBranch {
    pub block: FlowBlockId,
    pub address: ProgramOffset,
}

#[derive(Clone, Debug)]
pub struct FlowBlock {
    pub begin: FlowLocation,
    pub end: FlowLocation,
    pub end_class: EndClass,
    pub condition: Condition,
    pub stack: TokenStack,
    pub branch_true: Option<FlowBlockId>,
    pub branch_false: Option<FlowBlockId>,
    pub function_call: Option<FunctionId>,
    pub return_block: Option<FlowBlockId>,
    pub branch_reg: Reg,
    pub branch_offset: i32,
    pub indirect_branches: Vec<IndirectBranch>,
}

impl FlowBlock {
    fn at(location: Location, stack: TokenStack) -> Self {
        Self {
            begin: FlowLocation::Instruction(location),
            end: FlowLocation::Instruction(location),
            end_class: EndClass::Branch,
            condition: Condition::always(true),
            stack,
            branch_true: None,
            branch_false: None,
            function_call: None,
            return_block: None,
            branch_reg: Reg(0),
            branch_offset: 0,
            indirect_branches: Vec::new(),
        }
    }
    pub fn contains(&self, location: Location) -> bool {
        let location = FlowLocation::Instruction(location);
        self.begin <= location && location < self.end
    }
}

#[derive(Clone, Copy, Debug)]
struct Label {
    address: Location,
    block: FlowBlockId,
}

#[derive(Debug)]
pub struct Function {
    pub entrypoint: Location,
    pub blocks: BTreeMap<FlowLocation, FlowBlockId>,
    labels: Vec<Label>,
}

pub struct FlowGraph {
    blocks: Arena<FlowBlock, FlowBlockId>,
    pub functions: Vec<Function>,
    pub program_start: Location,
    pub dispatcher: Option<FlowBlockId>,
}

impl FlowGraph {
    pub fn analyze<E: Environment>(
        environment: &mut E,
        program_start: Location,
        exits_to_dispatcher: bool,
    ) -> Result<Self> {
        let mut graph = Self {
            blocks: Arena::new(SessionId::fresh()?),
            functions: Vec::new(),
            program_start,
            dispatcher: None,
        };
        if exits_to_dispatcher {
            // The dispatcher acquires its ordered location after the main function
            // is discovered. It is never fetched as guest code.
            let mut block = FlowBlock::at(program_start, TokenStack::default());
            block.end_class = EndClass::Exit;
            graph.dispatcher = Some(graph.blocks.allocate(block)?);
        }
        graph.add_function(program_start)?;
        let mut function = 0;
        while function < graph.functions.len() {
            while let Some(label) = graph.functions[function].labels.pop() {
                graph.analyze_label(environment, FunctionId(function), label)?;
            }
            function += 1;
        }
        if let Some(dispatcher) = graph.dispatcher {
            let (_, &last) = graph.functions[0]
                .blocks
                .last_key_value()
                .ok_or_else(|| internal("Empty main function"))?;
            let end = graph.block(last)?.end.instruction()?.next();
            graph.block_mut(dispatcher)?.begin = FlowLocation::Instruction(end);
            graph.block_mut(dispatcher)?.end = FlowLocation::Instruction(end);
            graph.functions[0]
                .blocks
                .insert(FlowLocation::Instruction(end), dispatcher);
        }
        Ok(graph)
    }

    pub fn block(&self, id: FlowBlockId) -> Result<&FlowBlock> {
        self.blocks.get(id)
    }
    fn block_mut(&mut self, id: FlowBlockId) -> Result<&mut FlowBlock> {
        self.blocks.get_mut(id)
    }

    fn add_function(&mut self, entrypoint: Location) -> Result<FunctionId> {
        let block = self
            .blocks
            .allocate(FlowBlock::at(entrypoint, TokenStack::default()))?;
        let id = FunctionId(self.functions.len());
        self.functions.push(Function {
            entrypoint,
            blocks: BTreeMap::new(),
            labels: vec![Label {
                address: entrypoint,
                block,
            }],
        });
        Ok(id)
    }

    fn add_label(
        &mut self,
        from: FlowBlockId,
        stack: TokenStack,
        pc: Location,
        function: FunctionId,
    ) -> Result<FlowBlockId> {
        let real = FlowLocation::Instruction(pc);
        if self.block(from)?.begin == real {
            return Ok(from);
        }
        if let Some(&existing) = self.functions[function.0].blocks.get(&real) {
            return Ok(self.functions[function.0]
                .blocks
                .get(&FlowLocation::PredicateBefore(pc))
                .copied()
                .unwrap_or(existing));
        }
        if let Some(label) = self.functions[function.0]
            .labels
            .iter()
            .find(|label| label.address == pc)
        {
            return Ok(label.block);
        }
        let block = self.blocks.allocate(FlowBlock::at(pc, stack))?;
        self.functions[function.0]
            .labels
            .push(Label { address: pc, block });
        Ok(block)
    }

    fn split_visited(&mut self, function: FunctionId, label: Label) -> Result<bool> {
        let Some((_, &old_id)) = self.functions[function.0]
            .blocks
            .range(..=FlowLocation::Instruction(label.address))
            .next_back()
        else {
            return Ok(false);
        };
        if !self.block(old_id)?.contains(label.address) {
            return Ok(false);
        }
        if self.block(old_id)?.begin == FlowLocation::Instruction(label.address) {
            return Err(internal("Dangling flow block"));
        }
        let old_begin = self.block(old_id)?.begin.instruction()?;
        let stack = self.block(old_id)?.stack.clone();
        let mut replacement = FlowBlock::at(old_begin, stack);
        replacement.end = FlowLocation::Instruction(label.address);
        replacement.branch_true = Some(label.block);
        let mut tail = std::mem::replace(self.block_mut(old_id)?, replacement);
        tail.begin = FlowLocation::Instruction(label.address);
        *self.block_mut(label.block)? = tail;
        self.functions[function.0]
            .blocks
            .insert(FlowLocation::Instruction(label.address), label.block);
        Ok(true)
    }

    fn analyze_label<E: Environment>(
        &mut self,
        environment: &mut E,
        function: FunctionId,
        label: Label,
    ) -> Result<()> {
        if self.split_visited(function, label)? {
            return Ok(());
        }
        let next = self.functions[function.0]
            .blocks
            .range((
                Excluded(FlowLocation::Instruction(label.address)),
                Unbounded,
            ))
            .next()
            .map(|(_, &block)| block);
        let mut pc = label.address;
        let mut branched = false;
        while next.is_none() || FlowLocation::Instruction(pc) < self.block(next.unwrap())?.begin {
            let decoded = fetch(environment, pc)?;
            branched = self
                .analyze_instruction(
                    environment,
                    label.block,
                    function,
                    pc,
                    decoded.instruction,
                    decoded.opcode,
                )
                .map_err(|error| {
                    error.at(
                        environment.metadata().stage,
                        pc.address(),
                        Some(decoded.instruction.0),
                    )
                })?;
            if branched {
                break;
            }
            pc = pc.next();
        }
        if !branched {
            let block = self.block_mut(label.block)?;
            block.end = FlowLocation::Instruction(pc);
            block.condition = Condition::always(true);
            block.branch_true = next;
            block.branch_false = None;
        }
        let begin = self.block(label.block)?.begin;
        self.functions[function.0].blocks.insert(begin, label.block);
        Ok(())
    }

    fn conditional_instruction(
        &mut self,
        id: FlowBlockId,
        function: FunctionId,
        pc: Location,
        end_class: EndClass,
        condition: Condition,
    ) -> Result<()> {
        let stack = self.block(id)?.stack.clone();
        if self.block(id)?.begin != FlowLocation::Instruction(pc) {
            let next = self.add_label(id, stack, pc, function)?;
            let block = self.block_mut(id)?;
            block.end = FlowLocation::Instruction(pc);
            block.condition = Condition::always(true);
            block.branch_true = Some(next);
            block.branch_false = None;
            return Ok(());
        }
        let mut virtual_block = FlowBlock::at(pc, stack.clone());
        virtual_block.begin = FlowLocation::PredicateBefore(pc);
        virtual_block.end = virtual_block.begin;
        virtual_block.condition = condition;
        let mut executing = std::mem::replace(self.block_mut(id)?, virtual_block);
        executing.end = FlowLocation::Instruction(pc.next());
        executing.end_class = end_class;
        let conditional = self.blocks.allocate(executing)?;
        self.block_mut(id)?.branch_true = Some(conditional);
        let endif = self.add_label(conditional, stack, pc.next(), function)?;
        self.block_mut(id)?.branch_false = Some(endif);
        if matches!(end_class, EndClass::Branch | EndClass::Kill) {
            let executing = self.block_mut(conditional)?;
            executing.condition = Condition::always(true);
            executing.branch_true = Some(endif);
            executing.branch_false = None;
        }
        self.functions[function.0]
            .blocks
            .insert(FlowLocation::Instruction(pc), conditional);
        Ok(())
    }

    fn analyze_branch(
        &mut self,
        id: FlowBlockId,
        function: FunctionId,
        pc: Location,
        inst: Instruction,
    ) -> Result<bool> {
        if inst.branch_is_cbuf() {
            return Err(unsupported("Branch with constant buffer offset"));
        }
        let (predicate, negated) = inst.predicate();
        if predicate == Pred::PT && negated {
            return Ok(false);
        }
        let condition = inst.condition();
        if condition != Condition::always(true) {
            let next = self.add_label(id, self.block(id)?.stack.clone(), pc.next(), function)?;
            self.block_mut(id)?.condition = condition;
            self.block_mut(id)?.branch_false = Some(next);
        } else {
            self.block_mut(id)?.condition = Condition::always(true);
        }
        Ok(true)
    }

    fn analyze_instruction<E: Environment>(
        &mut self,
        environment: &mut E,
        id: FlowBlockId,
        function: FunctionId,
        pc: Location,
        inst: Instruction,
        opcode: Opcode,
    ) -> Result<bool> {
        match opcode {
            Opcode::BRA | Opcode::JMP | Opcode::RET => {
                if !self.analyze_branch(id, function, pc, inst)? {
                    return Ok(false);
                }
                if opcode == Opcode::RET {
                    self.block_mut(id)?.end_class = EndClass::Return;
                } else {
                    let target = if opcode == Opcode::JMP {
                        Location::new(ProgramOffset(inst.branch_absolute()))?
                    } else {
                        pc.branch_relative(inst.branch_offset())?
                    };
                    let target =
                        self.add_label(id, self.block(id)?.stack.clone(), target, function)?;
                    self.block_mut(id)?.branch_true = Some(target);
                }
                self.block_mut(id)?.end = FlowLocation::Instruction(pc);
                return Ok(true);
            }
            Opcode::BRK | Opcode::CONT | Opcode::LONGJMP | Opcode::SYNC => {
                if !self.analyze_branch(id, function, pc, inst)? {
                    return Ok(false);
                }
                let (target, stack) = self.block(id)?.stack.pop(Token::from_opcode(opcode)?)?;
                let target = self.add_label(id, stack, target, function)?;
                self.block_mut(id)?.branch_true = Some(target);
                self.block_mut(id)?.end = FlowLocation::Instruction(pc);
                return Ok(true);
            }
            Opcode::KIL => {
                self.conditional_instruction(id, function, pc, EndClass::Kill, inst.condition())?;
                return Ok(true);
            }
            Opcode::PBK | Opcode::PCNT | Opcode::PEXIT | Opcode::PLONGJMP | Opcode::SSY => {
                self.block_mut(id)?.stack.push(
                    Token::from_opcode(opcode)?,
                    pc.branch_relative(inst.branch_offset())?,
                );
                return Ok(false);
            }
            Opcode::BRX | Opcode::JMX => {
                self.analyze_indirect(environment, id, function, pc, inst, opcode == Opcode::JMX)?;
                return Ok(true);
            }
            Opcode::EXIT => return self.analyze_exit(id, function, pc, inst),
            Opcode::PRET => return Err(unsupported("PRET flow analysis")),
            Opcode::CAL | Opcode::JCAL => {
                let target = if opcode == Opcode::JCAL {
                    Location::new(ProgramOffset(inst.branch_absolute()))?
                } else {
                    pc.branch_relative(inst.branch_offset())?
                };
                let callee = if let Some(index) =
                    self.functions.iter().position(|f| f.entrypoint == target)
                {
                    FunctionId(index)
                } else {
                    self.add_function(target)?
                };
                let next =
                    self.add_label(id, self.block(id)?.stack.clone(), pc.next(), function)?;
                let block = self.block_mut(id)?;
                block.end_class = EndClass::Call;
                block.function_call = Some(callee);
                block.return_block = Some(next);
                block.end = FlowLocation::Instruction(pc);
                return Ok(true);
            }
            _ => (),
        }
        let (predicate, negated) = inst.predicate();
        if predicate == Pred::PT {
            return Ok(false);
        }
        self.conditional_instruction(
            id,
            function,
            pc,
            EndClass::Branch,
            Condition {
                flow_test: FlowTest::T,
                predicate,
                negated,
            },
        )?;
        Ok(true)
    }

    fn analyze_indirect<E: Environment>(
        &mut self,
        environment: &mut E,
        id: FlowBlockId,
        function: FunctionId,
        pc: Location,
        inst: Instruction,
        absolute: bool,
    ) -> Result<()> {
        let table = track_indirect_table(environment, pc, self.program_start)?
            .ok_or_else(|| unsupported("Failed to track finite indirect branch table"))?;
        if inst.condition() != Condition::always(true) {
            return Err(unsupported("Conditional indirect branch"));
        }
        let base = if absolute { 0 } else { pc.address().0 };
        let mut targets = Vec::with_capacity(table.num_entries as usize);
        for index in 0..table.num_entries {
            let target = environment.read_cbuf_value(
                table.cbuf_index,
                ByteOffset(table.cbuf_offset.0.wrapping_add(index * 4)),
            )?;
            targets.push(
                target
                    .wrapping_add(base)
                    .wrapping_add_signed(table.branch_offset)
                    .wrapping_add(8),
            );
        }
        targets.sort_unstable();
        targets.dedup();
        for address in targets {
            let target = self.add_label(
                id,
                self.block(id)?.stack.clone(),
                Location::new(ProgramOffset(address))?,
                function,
            )?;
            self.block_mut(id)?.indirect_branches.push(IndirectBranch {
                block: target,
                address: ProgramOffset(address),
            });
        }
        let block = self.block_mut(id)?;
        block.condition = Condition::always(true);
        block.end = FlowLocation::Instruction(pc.next());
        block.end_class = EndClass::IndirectBranch;
        block.branch_reg = table.branch_reg;
        block.branch_offset = table
            .branch_offset
            .wrapping_add(8)
            .wrapping_add(base as i32);
        Ok(())
    }

    fn analyze_exit(
        &mut self,
        id: FlowBlockId,
        function: FunctionId,
        pc: Location,
        inst: Instruction,
    ) -> Result<bool> {
        let (predicate, negated) = inst.predicate();
        if (predicate == Pred::PT && negated) || inst.flow_test() == FlowTest::F {
            return Ok(false);
        }
        if self.dispatcher.is_some() && function.0 != 0 {
            return Err(unsupported("Dispatch EXIT on external function"));
        }
        let condition = inst.condition();
        if condition != Condition::always(true) {
            if self.block(id)?.stack.peek(Token::Pexit).is_some() {
                return Err(unsupported("Conditional EXIT with PEXIT token"));
            }
            if let Some(dispatcher) = self.dispatcher {
                let next =
                    self.add_label(id, self.block(id)?.stack.clone(), pc.next(), function)?;
                let block = self.block_mut(id)?;
                block.end = FlowLocation::Instruction(pc);
                block.end_class = EndClass::Branch;
                block.condition = condition;
                block.branch_true = Some(dispatcher);
                block.branch_false = Some(next);
            } else {
                self.conditional_instruction(id, function, pc, EndClass::Exit, condition)?;
            }
            return Ok(true);
        }
        if self.block(id)?.stack.peek(Token::Pexit).is_some() {
            let (target, stack) = self.block(id)?.stack.pop(Token::Pexit)?;
            let target = self.add_label(id, stack, target, function)?;
            let block = self.block_mut(id)?;
            block.condition = Condition::always(true);
            block.branch_true = Some(target);
            block.branch_false = None;
            // This path leaves the existing block end unchanged.
            return Ok(true);
        }
        if let Some(dispatcher) = self.dispatcher {
            let block = self.block_mut(id)?;
            block.condition = Condition::always(true);
            block.end = FlowLocation::Instruction(pc);
            block.end_class = EndClass::Branch;
            block.branch_true = Some(dispatcher);
            block.branch_false = None;
        } else {
            let block = self.block_mut(id)?;
            block.end = FlowLocation::Instruction(pc.next());
            block.end_class = EndClass::Exit;
        }
        Ok(true)
    }
}
