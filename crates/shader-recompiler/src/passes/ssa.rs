use crate::{
    CompileError, Result,
    arena::{BlockId, InstId},
    diagnostics::FailureCategory,
    ir::{Ir, Opcode as O, Program, Type, Value},
};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Variable {
    Register(u8),
    Predicate(u8),
    Zero,
    Sign,
    Carry,
    Overflow,
    Goto(u32),
    Indirect,
}
impl Variable {
    fn ty(self) -> Type {
        if matches!(self, Self::Register(_) | Self::Indirect) {
            Type::U32
        } else {
            Type::U1
        }
    }
    fn undef(self) -> O {
        if self.ty() == Type::U32 {
            O::UndefU32
        } else {
            O::UndefU1
        }
    }
}
#[derive(Clone, Copy)]
enum Status {
    Start,
    SetValue,
    PushPhi,
}
struct ReadState {
    block: BlockId,
    status: Status,
    result: Value,
    phi: Option<InstId>,
    predecessor: usize,
}
impl ReadState {
    fn new(block: BlockId) -> Self {
        Self {
            block,
            status: Status::Start,
            result: Value::None,
            phi: None,
            predecessor: 0,
        }
    }
}
#[derive(Default)]
struct Pass {
    definitions: HashMap<(Variable, BlockId), Value>,
    incomplete: HashMap<BlockId, BTreeMap<Variable, InstId>>,
    read_stack: Vec<ReadState>,
}
impl Pass {
    fn definition(&self, ir: &Ir, block: BlockId, variable: Variable) -> Result<Value> {
        Ok(if let Variable::Register(reg) = variable {
            ir.block(block)?.ssa_reg_values[reg as usize]
        } else {
            *self
                .definitions
                .get(&(variable, block))
                .unwrap_or(&Value::None)
        })
    }
    fn write(
        &mut self,
        ir: &mut Ir,
        block: BlockId,
        variable: Variable,
        value: Value,
    ) -> Result<()> {
        if let Variable::Register(reg) = variable {
            ir.block_mut(block)?.ssa_reg_values[reg as usize] = value;
        } else {
            self.definitions.insert((variable, block), value);
        }
        Ok(())
    }
    fn trivial_phi(
        &mut self,
        ir: &mut Ir,
        block: BlockId,
        phi: InstId,
        variable: Variable,
    ) -> Result<Value> {
        let mut same = Value::None;
        for index in 0..ir.inst(phi)?.num_args() {
            let value = ir.inst(phi)?.arg(index)?;
            if ir.resolve(value)? == ir.resolve(same)? || value == Value::Inst(phi) {
                continue;
            }
            if same != Value::None {
                return Ok(Value::Inst(phi));
            }
            same = value;
        }
        ir.detach(block, phi)?;
        let mut position = ir
            .block(block)?
            .instructions()
            .iter()
            .position(|&id| ir.inst(id).expect("Live instruction").opcode() != O::Phi)
            .unwrap_or(ir.block(block)?.instructions().len());
        if same == Value::None {
            same = Value::Inst(ir.insert(block, position, variable.undef(), &[], 0)?);
            position += 1;
        }
        ir.attach(block, position, phi)?;
        ir.replace_uses_with(phi, same)?;
        Ok(same)
    }
    fn read(&mut self, ir: &mut Ir, root: BlockId, variable: Variable) -> Result<Value> {
        self.read_stack.clear();
        self.read_stack.push(ReadState::new(root));
        loop {
            let top = self.read_stack.len() - 1;
            let block = self.read_stack[top].block;
            let result = match self.read_stack[top].status {
                Status::Start => {
                    let defined = self.definition(ir, block, variable)?;
                    if defined != Value::None {
                        defined
                    } else if !ir.block(block)?.is_ssa_sealed {
                        let phi = ir.new_phi(block, variable.ty())?;
                        self.incomplete
                            .entry(block)
                            .or_default()
                            .insert(variable, phi);
                        Value::Inst(phi)
                    } else {
                        let predecessors = ir.block(block)?.predecessors();
                        if predecessors.len() == 1 {
                            let predecessor = predecessors[0];
                            self.read_stack[top].status = Status::SetValue;
                            self.read_stack.push(ReadState::new(predecessor));
                            continue;
                        }
                        let first = predecessors.first().copied();
                        let phi = ir.new_phi(block, variable.ty())?;
                        self.write(ir, block, variable, Value::Inst(phi))?;
                        self.read_stack[top].phi = Some(phi);
                        if let Some(first) = first {
                            self.read_stack[top].status = Status::PushPhi;
                            self.read_stack.push(ReadState::new(first));
                            continue;
                        }
                        self.trivial_phi(ir, block, phi, variable)?
                    }
                }
                Status::SetValue => self.read_stack[top].result,
                Status::PushPhi => {
                    let state = &mut self.read_stack[top];
                    let phi = state.phi.expect("Phi read frame");
                    let predecessor = ir.block(block)?.predecessors()[state.predecessor];
                    ir.add_phi_operand(phi, predecessor, state.result)?;
                    state.predecessor += 1;
                    if let Some(&next) = ir.block(block)?.predecessors().get(state.predecessor) {
                        self.read_stack.push(ReadState::new(next));
                        continue;
                    }
                    self.trivial_phi(ir, block, phi, variable)?
                }
            };
            self.write(ir, block, variable, result)?;
            self.read_stack.pop();
            if let Some(parent) = self.read_stack.last_mut() {
                parent.result = result;
            } else {
                return Ok(result);
            }
        }
    }
    fn seal(&mut self, ir: &mut Ir, block: BlockId) -> Result<()> {
        if let Some(phis) = self.incomplete.remove(&block) {
            for (variable, phi) in phis {
                let mut index = 0;
                while let Some(&predecessor) = ir.block(block)?.predecessors().get(index) {
                    let value = self.read(ir, predecessor, variable)?;
                    ir.add_phi_operand(phi, predecessor, value)?;
                    index += 1;
                }
                self.trivial_phi(ir, block, phi, variable)?;
            }
        }
        ir.block_mut(block)?.is_ssa_sealed = true;
        Ok(())
    }
    fn instruction(&mut self, ir: &mut Ir, block: BlockId, id: InstId) -> Result<()> {
        let inst = ir.inst(id)?;
        let op = inst.opcode();
        let (variable, write, value) = match op {
            O::GetRegister | O::SetRegister => {
                let Value::Reg(reg) = inst.arg(0)? else {
                    return Err(malformed());
                };
                if reg.0 == 255 {
                    return Ok(());
                }
                (
                    Variable::Register(reg.0),
                    op == O::SetRegister,
                    if op == O::SetRegister {
                        inst.arg(1)?
                    } else {
                        Value::None
                    },
                )
            }
            O::GetPred | O::SetPred => {
                let Value::Pred(pred) = inst.arg(0)? else {
                    return Err(malformed());
                };
                if pred.index() == 7 {
                    return Ok(());
                }
                (
                    Variable::Predicate(pred.index() as u8),
                    op == O::SetPred,
                    if op == O::SetPred {
                        inst.arg(1)?
                    } else {
                        Value::None
                    },
                )
            }
            O::GetGotoVariable | O::SetGotoVariable => {
                let Value::U32(index) = inst.arg(0)? else {
                    return Err(malformed());
                };
                (
                    Variable::Goto(index),
                    op == O::SetGotoVariable,
                    if op == O::SetGotoVariable {
                        inst.arg(1)?
                    } else {
                        Value::None
                    },
                )
            }
            O::GetZFlag => (Variable::Zero, false, Value::None),
            O::SetZFlag => (Variable::Zero, true, inst.arg(0)?),
            O::GetSFlag => (Variable::Sign, false, Value::None),
            O::SetSFlag => (Variable::Sign, true, inst.arg(0)?),
            O::GetCFlag => (Variable::Carry, false, Value::None),
            O::SetCFlag => (Variable::Carry, true, inst.arg(0)?),
            O::GetOFlag => (Variable::Overflow, false, Value::None),
            O::SetOFlag => (Variable::Overflow, true, inst.arg(0)?),
            O::GetIndirectBranchVariable => (Variable::Indirect, false, Value::None),
            O::SetIndirectBranchVariable => (Variable::Indirect, true, inst.arg(0)?),
            _ => return Ok(()),
        };
        if write {
            self.write(ir, block, variable, value)
        } else {
            let value = self.read(ir, block, variable)?;
            ir.replace_uses_with(id, value)
        }
    }
}
fn malformed() -> CompileError {
    CompileError::new(
        FailureCategory::Internal,
        "Malformed mutable-context operand during SSA construction",
    )
}
fn concrete_type(ir: &Ir, id: InstId) -> Result<Type> {
    let mut queue = VecDeque::from([id]);
    let mut seen = HashSet::new();
    while let Some(id) = queue.pop_front() {
        if !seen.insert(id) {
            continue;
        }
        let inst = ir.inst(id)?;
        for i in 0..inst.num_args() {
            let value = inst.arg(i)?;
            let ty = ir.value_type(value)?;
            if ty != Type::Opaque {
                return Ok(ty);
            }
            if let Value::Inst(id) = ir.resolve(value)? {
                queue.push_back(id);
            }
        }
    }
    Ok(Type::Opaque)
}
pub fn rewrite_ssa(ir: &mut Ir, program: &Program) -> Result<()> {
    let mut pass = Pass::default();
    for &block in program.post_order.iter().rev() {
        let mut index = 0;
        while let Some(&id) = ir.block(block)?.instructions().get(index) {
            let before = ir.block(block)?.instructions().len();
            pass.instruction(ir, block, id)?;
            index += 1 + ir.block(block)?.instructions().len() - before;
        }
        pass.seal(ir, block)?;
    }
    for &block in program.post_order.iter().rev() {
        let mut index = 0;
        while let Some(&id) = ir.block(block)?.instructions().get(index) {
            if ir.inst(id)?.opcode() == O::Phi {
                if ir.inst(id)?.ty() == Type::Opaque {
                    ir.inst_mut(id)?.flags = concrete_type(ir, id)?.0;
                }
                ir.order_phi_args(id)?;
            }
            index += 1;
        }
    }
    Ok(())
}
