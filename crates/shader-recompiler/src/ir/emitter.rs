use super::*;
use crate::{
    CompileError, Result,
    arena::{BlockId, InstId},
    diagnostics::FailureCategory,
};

pub struct Emitter<'a> {
    pub ir: &'a mut Ir,
    pub block: BlockId,
    position: usize,
}

impl<'a> Emitter<'a> {
    pub fn new(ir: &'a mut Ir, block: BlockId) -> Result<Self> {
        let position = ir.block(block)?.instructions().len();
        Ok(Self {
            ir,
            block,
            position,
        })
    }
    pub fn before(ir: &'a mut Ir, block: BlockId, instruction: InstId) -> Result<Self> {
        let position = ir
            .block(block)?
            .instructions()
            .iter()
            .position(|&id| id == instruction)
            .ok_or_else(|| Self::invalid("Insertion instruction is not in block"))?;
        Ok(Self {
            ir,
            block,
            position,
        })
    }
    pub fn at_position(ir: &'a mut Ir, block: BlockId, position: usize) -> Result<Self> {
        if position > ir.block(block)?.instructions().len() {
            return Err(Self::invalid("IR insertion position out of range"));
        }
        Ok(Self {
            ir,
            block,
            position,
        })
    }
    pub fn emit(&mut self, opcode: Opcode, args: &[Value], flags: u32) -> Result<Value> {
        let id = self
            .ir
            .insert(self.block, self.position, opcode, args, flags)?;
        self.position += 1;
        Ok(Value::Inst(id))
    }
    pub(super) fn invalid(message: impl Into<String>) -> CompileError {
        CompileError::new(FailureCategory::InvalidArgument, message)
    }
    pub(super) fn same_type(&self, values: &[Value]) -> Result<Type> {
        let first = values
            .first()
            .ok_or_else(|| Self::invalid("Expected at least one operand"))?;
        let ty = self.ir.value_type(*first)?;
        for &value in &values[1..] {
            if self.ir.value_type(value)? != ty {
                return Err(Self::invalid("Mismatching operand types"));
            }
        }
        Ok(ty)
    }

    pub(super) fn require_type(&self, value: Value, expected: Type) -> Result<()> {
        if self.ir.value_type(value)? != expected {
            return Err(Self::invalid(format!("Expected operand type {expected:?}")));
        }
        Ok(())
    }
    pub fn get_pred(&mut self, pred: Pred, negated: bool) -> Result<Value> {
        if pred == Pred::PT {
            return Ok(Value::U1(!negated));
        }
        let value = self.emit(Opcode::GetPred, &[Value::Pred(pred)], 0)?;
        if negated {
            self.emit(Opcode::LogicalNot, &[value], 0)
        } else {
            Ok(value)
        }
    }
    pub fn set_pred(&mut self, pred: Pred, value: Value) -> Result<()> {
        if pred != Pred::PT {
            self.emit(Opcode::SetPred, &[Value::Pred(pred), value], 0)?;
        }
        Ok(())
    }
    pub fn phi_move(&mut self, phi: InstId, value: Value) -> Result<Value> {
        self.emit(Opcode::PhiMove, &[Value::Inst(phi), value], 0)
    }
    pub fn make_lod_clamp_pair(&mut self, bias: Value, clamp: Value) -> Result<Value> {
        match (bias.is_empty(), clamp.is_empty()) {
            (false, false) => self.composite_construct(&[bias, clamp]),
            (false, true) => Ok(bias),
            (true, false) => Ok(clamp),
            (true, true) => Ok(Value::None),
        }
    }
    pub fn fp_abs_neg(&mut self, value: Value, abs: bool, neg: bool) -> Result<Value> {
        let value = if abs { self.fp_abs(value)? } else { value };
        if neg { self.fp_neg(value) } else { Ok(value) }
    }
    pub fn fp_ordered(&mut self, lhs: Value, rhs: Value) -> Result<Value> {
        self.same_type(&[lhs, rhs])?;
        let lhs_nan = self.fp_is_nan(lhs)?;
        let rhs_nan = self.fp_is_nan(rhs)?;
        let lhs_ordered = self.logical_not(lhs_nan)?;
        let rhs_ordered = self.logical_not(rhs_nan)?;
        self.logical_and(lhs_ordered, rhs_ordered)
    }
    pub fn fp_unordered(&mut self, lhs: Value, rhs: Value) -> Result<Value> {
        self.same_type(&[lhs, rhs])?;
        let lhs_nan = self.fp_is_nan(lhs)?;
        let rhs_nan = self.fp_is_nan(rhs)?;
        self.logical_or(lhs_nan, rhs_nan)
    }
    pub fn i_equal(&mut self, lhs: Value, rhs: Value) -> Result<Value> {
        match self.same_type(&[lhs, rhs])? {
            Type::U32 => self.emit(Opcode::IEqual, &[lhs, rhs], 0),
            Type::U64 => {
                let lhs = self.unpack_uint2x32(lhs)?;
                let rhs = self.unpack_uint2x32(rhs)?;
                let lhs_low = self.composite_extract(lhs, 0)?;
                let rhs_low = self.composite_extract(rhs, 0)?;
                let lhs_high = self.composite_extract(lhs, 1)?;
                let rhs_high = self.composite_extract(rhs, 1)?;
                let low_equal = self.i_equal(lhs_low, rhs_low)?;
                let high_equal = self.i_equal(lhs_high, rhs_high)?;
                self.logical_and(low_equal, high_equal)
            }
            _ => Err(Self::invalid(
                "Integer equality requires matching U32 or U64 operands",
            )),
        }
    }
}
