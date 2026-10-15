use super::*;
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct VariableId(u32);
impl VariableId {
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// Explicit frontend contract for a live-in mutable variable. No shared pass
/// assumes that guest registers/predicates start at zero or true.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InitialValue {
    Undefined,
    Constant(Constant),
    Parameter(u32),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Operand {
    Variable(VariableId),
    Value(ValueId),
}
impl From<ValueId> for Operand {
    fn from(value: ValueId) -> Self {
        Self::Value(value)
    }
}
impl From<VariableId> for Operand {
    fn from(value: VariableId) -> Self {
        Self::Variable(value)
    }
}

#[derive(Clone, Debug)]
enum Statement {
    Instruction {
        op: Op,
        arguments: Vec<Operand>,
        results: Box<[ValueId]>,
        source: ShaderSourceLocation,
    },
    Assign {
        variable: VariableId,
        value: Operand,
        source: ShaderSourceLocation,
    },
}
#[derive(Clone, Debug)]
struct RawBlock {
    source: ShaderSourceLocation,
    statements: Vec<Statement>,
    terminator: Option<(Terminator<Operand>, ShaderSourceLocation)>,
}

/// Temporary mutable names exist only during frontend compilation. Pruned phi
/// insertion and dominator-tree renaming remove them in one shared algorithm.
pub struct FunctionBuilder {
    stage: ShaderStage,
    blocks: Vec<RawBlock>,
    variables: Vec<(Type, Option<InitialValue>)>,
    values: Vec<Type>,
    effect: VariableId,
}

impl FunctionBuilder {
    #[must_use]
    pub fn new(stage: ShaderStage) -> Self {
        Self {
            stage,
            blocks: vec![RawBlock {
                source: ShaderSourceLocation::new(0),
                statements: vec![],
                terminator: None,
            }],
            variables: vec![(Type::Effect, None)],
            values: vec![],
            effect: VariableId(0),
        }
    }
    #[must_use]
    pub const fn entry(&self) -> BlockId {
        BlockId(0)
    }
    pub fn block(&mut self, source: ShaderSourceLocation) -> BlockId {
        let id = BlockId(self.blocks.len() as u32);
        self.blocks.push(RawBlock {
            source,
            statements: vec![],
            terminator: None,
        });
        id
    }
    pub fn variable(&mut self, ty: Type, initial: InitialValue) -> VariableId {
        assert_ne!(
            ty,
            Type::Effect,
            "effect variables are owned by the builder"
        );
        let id = VariableId(self.variables.len() as u32);
        self.variables.push((ty, Some(initial)));
        id
    }
    pub fn emit(
        &mut self,
        block: BlockId,
        source: ShaderSourceLocation,
        op: Op,
        mut arguments: Vec<Operand>,
    ) -> Vec<ValueId> {
        assert!(
            self.blocks[block.index()].terminator.is_none(),
            "instruction after terminator"
        );
        if op.ordered() {
            arguments.push(self.effect.into());
        }
        let (_, types) = op.signature();
        let results: Vec<_> = types
            .iter()
            .map(|&ty| {
                let id = ValueId(self.values.len() as u32);
                self.values.push(ty);
                id
            })
            .collect();
        self.blocks[block.index()]
            .statements
            .push(Statement::Instruction {
                op: op.clone(),
                arguments,
                results: results.clone().into_boxed_slice(),
                source,
            });
        if op.ordered() {
            self.assign(
                block,
                source,
                self.effect,
                (*results.last().unwrap()).into(),
            );
            results[..results.len() - 1].to_vec()
        } else {
            results
        }
    }
    pub fn assign(
        &mut self,
        block: BlockId,
        source: ShaderSourceLocation,
        variable: VariableId,
        value: Operand,
    ) {
        assert!(
            self.blocks[block.index()].terminator.is_none(),
            "assignment after terminator"
        );
        self.blocks[block.index()]
            .statements
            .push(Statement::Assign {
                variable,
                value,
                source,
            });
    }
    pub fn terminate(
        &mut self,
        block: BlockId,
        source: ShaderSourceLocation,
        terminator: Terminator<Operand>,
    ) {
        assert!(
            self.blocks[block.index()].terminator.is_none(),
            "block already terminated"
        );
        self.blocks[block.index()].terminator = Some((terminator, source));
    }

    pub fn finish(mut self) -> Result<Function, Error> {
        // All variables have explicit initial values, but unused initial values
        // are removed by DCE. Initial definitions dominate every real entry.
        let mut initial = Vec::new();
        for (index, &(ty, value)) in self.variables.iter().enumerate() {
            let op = match value {
                None => Op::EffectStart,
                Some(InitialValue::Undefined) => Op::Undefined(ty),
                Some(InitialValue::Parameter(index)) => Op::Parameter { index, ty },
                Some(InitialValue::Constant(constant)) => {
                    if constant.ty != ty {
                        return Err(Error::at(
                            ShaderSourceLocation::new(0),
                            "initial value type mismatch",
                        ));
                    }
                    Op::Constant(constant)
                }
            };
            let result = ValueId(self.values.len() as u32);
            self.values.push(ty);
            initial.push(Statement::Instruction {
                op,
                arguments: vec![],
                results: vec![result].into_boxed_slice(),
                source: ShaderSourceLocation::new(0),
            });
            initial.push(Statement::Assign {
                variable: VariableId(index as u32),
                value: result.into(),
                source: ShaderSourceLocation::new(0),
            });
        }
        initial.append(&mut self.blocks[0].statements);
        self.blocks[0].statements = initial;
        let edges: Vec<_> = self
            .blocks
            .iter()
            .map(|b| {
                b.terminator
                    .as_ref()
                    .map_or_else(Vec::new, |(t, _)| t.successors())
            })
            .collect();
        let (_, order) = super::dominance::traverse(&edges)?;
        let reachable: BTreeSet<_> = order.into_iter().collect();
        let mut remap = vec![None; self.blocks.len()];
        let mut blocks = Vec::new();
        for (index, block) in self.blocks.into_iter().enumerate() {
            if reachable.contains(&BlockId(index as u32)) {
                if block.terminator.is_none() {
                    return Err(Error::at(block.source, "reachable block has no terminator"));
                }
                remap[index] = Some(BlockId(blocks.len() as u32));
                blocks.push(block);
            }
        }
        for block in &mut blocks {
            remap_targets(&mut block.terminator.as_mut().unwrap().0, |b| {
                remap[b.index()].unwrap()
            });
        }
        let dominance = Dominance::from_edges(
            &blocks
                .iter()
                .map(|b| b.terminator.as_ref().unwrap().0.successors())
                .collect::<Vec<_>>(),
        )?;
        construct(
            self.stage,
            blocks,
            self.variables.iter().map(|&(ty, _)| ty).collect(),
            self.values,
            dominance,
        )
    }
}

pub(crate) fn remap_targets<A>(
    terminator: &mut Terminator<A>,
    mut map: impl FnMut(BlockId) -> BlockId,
) {
    match terminator {
        Terminator::Jump(b) => *b = map(*b),
        Terminator::Branch {
            when_true,
            when_false,
            ..
        } => {
            *when_true = map(*when_true);
            *when_false = map(*when_false);
        }
        Terminator::Switch { cases, default, .. } => {
            for (_, b) in cases {
                *b = map(*b);
            }
            *default = map(*default);
        }
        _ => {}
    }
}

fn operands<A: Copy>(terminator: &Terminator<A>) -> Vec<A> {
    match terminator {
        Terminator::Branch { condition, .. } => vec![*condition],
        Terminator::Switch { selector, .. } => vec![*selector],
        _ => vec![],
    }
}
pub(crate) fn map_terminator<A: Copy, B>(
    term: &Terminator<A>,
    mut map: impl FnMut(A) -> B,
) -> Terminator<B> {
    match term {
        Terminator::Jump(b) => Terminator::Jump(*b),
        Terminator::Branch {
            condition,
            when_true,
            when_false,
        } => Terminator::Branch {
            condition: map(*condition),
            when_true: *when_true,
            when_false: *when_false,
        },
        Terminator::Switch {
            selector,
            cases,
            default,
        } => Terminator::Switch {
            selector: map(*selector),
            cases: cases.clone(),
            default: *default,
        },
        Terminator::Return => Terminator::Return,
        Terminator::Discard => Terminator::Discard,
        Terminator::Unreachable => Terminator::Unreachable,
    }
}

fn construct(
    stage: ShaderStage,
    raw: Vec<RawBlock>,
    types: Vec<Type>,
    value_types: Vec<Type>,
    dom: Dominance,
) -> Result<Function, Error> {
    let mut uses = vec![BTreeSet::new(); raw.len()];
    let mut defs = vec![BTreeSet::new(); raw.len()];
    for (index, block) in raw.iter().enumerate() {
        // Assignments are mutable definitions; instruction results are already
        // unique temporaries and need no phi insertion.
        for statement in &block.statements {
            match statement {
                Statement::Instruction { arguments, .. } => {
                    for &a in arguments {
                        if let Operand::Variable(v) = a
                            && !defs[index].contains(&v)
                        {
                            uses[index].insert(v);
                        }
                    }
                }
                Statement::Assign {
                    variable, value, ..
                } => {
                    if let Operand::Variable(v) = *value
                        && !defs[index].contains(&v)
                    {
                        uses[index].insert(v);
                    }
                    defs[index].insert(*variable);
                }
            }
        }
        for a in operands(&block.terminator.as_ref().unwrap().0)
            .into_iter()
            .chain([Operand::Variable(VariableId(0))])
        {
            if let Operand::Variable(v) = a
                && !defs[index].contains(&v)
            {
                uses[index].insert(v);
            }
        }
    }
    let mut live = uses.clone();
    loop {
        let mut changed = false;
        for &b in dom.reverse_postorder.iter().rev() {
            let mut next = uses[b.index()].clone();
            for successor in raw[b.index()].terminator.as_ref().unwrap().0.successors() {
                next.extend(
                    live[successor.index()]
                        .difference(&defs[b.index()])
                        .copied(),
                );
            }
            changed |= next != live[b.index()];
            live[b.index()] = next;
        }
        if !changed {
            break;
        }
    }
    let mut definition_blocks = vec![Vec::new(); types.len()];
    for (index, definitions) in defs.iter().enumerate() {
        for variable in definitions {
            definition_blocks[variable.index()].push(BlockId(index as u32));
        }
    }
    let mut phi_variables = vec![BTreeSet::new(); raw.len()];
    for variable in 0..types.len() {
        let variable = VariableId(variable as u32);
        let mut work = definition_blocks[variable.index()].clone();
        while let Some(block) = work.pop() {
            for &frontier in &dom.frontiers[block.index()] {
                if live[frontier.index()].contains(&variable)
                    && phi_variables[frontier.index()].insert(variable)
                {
                    work.push(frontier);
                }
            }
        }
    }
    let placeholder = Value {
        ty: Type::U32,
        definition: Definition::Phi { block: BlockId(0) },
    };
    let mut function = Function {
        stage,
        inputs: Vec::new(),
        outputs: Vec::new(),
        resources: Vec::new(),
        workgroup_size: None,
        tessellation_control_points: None,
        values: value_types
            .iter()
            .map(|&ty| Value { ty, ..placeholder })
            .collect(),
        blocks: Vec::new(),
    };
    for (index, block) in raw.iter().enumerate() {
        let id = BlockId(index as u32);
        let phis = phi_variables[index]
            .iter()
            .map(|v| {
                let result = ValueId(function.values.len() as u32);
                function.values.push(Value {
                    ty: types[v.index()],
                    definition: Definition::Phi { block: id },
                });
                Phi {
                    result,
                    incoming: Vec::new(),
                    source: block.source,
                }
            })
            .collect();
        function.blocks.push(Block {
            source: block.source,
            phis,
            instructions: vec![],
            terminator: Terminator::Unreachable,
            terminator_source: block.terminator.as_ref().unwrap().1,
            effect: ValueId(0),
        });
    }
    let mut stacks: Vec<Vec<ValueId>> = vec![Vec::new(); types.len()];
    let mut saved = vec![Vec::<VariableId>::new(); raw.len()];
    let mut work = vec![(BlockId(0), false)];
    while let Some((block, leaving)) = work.pop() {
        if leaving {
            for v in saved[block.index()].iter().rev() {
                stacks[v.index()].pop();
            }
            continue;
        }
        for (variable, phi) in phi_variables[block.index()]
            .iter()
            .zip(&function.blocks[block.index()].phis)
        {
            stacks[variable.index()].push(phi.result);
            saved[block.index()].push(*variable);
        }
        for statement in &raw[block.index()].statements {
            let resolve = |a: Operand| -> Result<ValueId, Error> {
                match a {
                    Operand::Value(v) => Ok(v),
                    Operand::Variable(v) => stacks
                        .get(v.index())
                        .and_then(|s| s.last())
                        .copied()
                        .ok_or_else(|| {
                            Error::at(
                                raw[block.index()].source,
                                "variable has no dominating definition",
                            )
                        }),
                }
            };
            match statement {
                Statement::Assign {
                    variable,
                    value,
                    source,
                } => {
                    let value = resolve(*value)?;
                    if value.index() >= function.values.len()
                        || variable.index() >= types.len()
                        || function.values[value.index()].ty != types[variable.index()]
                    {
                        return Err(Error::at(*source, "assignment type or ID mismatch"));
                    }
                    stacks[variable.index()].push(value);
                    saved[block.index()].push(*variable);
                }
                Statement::Instruction {
                    op,
                    arguments,
                    results,
                    source,
                } => {
                    let arguments = arguments
                        .iter()
                        .map(|&a| resolve(a))
                        .collect::<Result<Vec<_>, _>>()?;
                    let index = function.blocks[block.index()].instructions.len() as u32;
                    for (result, &value) in results.iter().enumerate() {
                        function.values[value.index()].definition = Definition::Instruction {
                            block,
                            index,
                            result: result as u32,
                        };
                    }
                    function.blocks[block.index()]
                        .instructions
                        .push(Instruction {
                            op: op.clone(),
                            arguments: arguments.into_boxed_slice(),
                            results: results.clone(),
                            source: *source,
                        });
                }
            }
        }
        let resolve = |a| match a {
            Operand::Value(v) => v,
            Operand::Variable(v) => *stacks[v.index()]
                .last()
                .expect("explicit initial definition"),
        };
        function.blocks[block.index()].terminator =
            map_terminator(&raw[block.index()].terminator.as_ref().unwrap().0, resolve);
        function.blocks[block.index()].effect = resolve(Operand::Variable(VariableId(0)));
        for successor in raw[block.index()]
            .terminator
            .as_ref()
            .unwrap()
            .0
            .successors()
        {
            for (variable, phi) in phi_variables[successor.index()]
                .iter()
                .zip(&mut function.blocks[successor.index()].phis)
            {
                phi.incoming.push((block, resolve((*variable).into())));
            }
        }
        work.push((block, true));
        work.extend(
            dom.children[block.index()]
                .iter()
                .rev()
                .map(|&b| (b, false)),
        );
    }
    for block in &mut function.blocks {
        for phi in &mut block.phis {
            phi.incoming.sort_unstable_by_key(|(b, _)| *b);
        }
    }
    // Remove unreachable temporaries and rebuild dense definition indices even
    // before optimization. This is construction, not a release verifier.
    super::optimize::compact(&mut function);
    #[cfg(debug_assertions)]
    function.verify()?;
    Ok(function)
}
