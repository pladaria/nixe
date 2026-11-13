//! Compile draw-uniform handle expressions, including indexed handle tables.
//! https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/shader_recompiler/ir_opt/texture_pass.cpp#L643
use super::handle_control_flow::conditional_phi;
use crate::{
    Result,
    arena::{BlockId, InstId},
    artifact::{HandleOp as H, HandleProgram},
    ir::{Ir, Opcode as O, Value},
};
use std::collections::HashMap;

pub(super) fn compile(
    ir: &Ir,
    value: Value,
    block: BlockId,
    allow_index: bool,
) -> Result<Option<(HandleProgram, Option<InstId>)>> {
    let mut b = Builder {
        ir,
        block,
        code: Vec::new(),
        slots: HashMap::new(),
        next_slot: 0,
        budget: 4096,
        allow_index,
        index: None,
        load_depth: 0,
        scaled: 0,
    };
    if !b.build(value)? {
        return Ok(None);
    }
    Ok(HandleProgram::new(b.code).map(|p| (p, b.index)))
}
struct Builder<'a> {
    ir: &'a Ir,
    block: BlockId,
    code: Vec<H>,
    // None marks an expression currently being built, detecting loop-carried values.
    slots: HashMap<InstId, Option<u8>>,
    next_slot: u16,
    budget: u32,
    allow_index: bool,
    index: Option<InstId>,
    load_depth: u32,
    scaled: u32,
}
fn undefined(op: O) -> bool {
    matches!(
        op,
        O::UndefU1 | O::UndefU8 | O::UndefU16 | O::UndefU32 | O::UndefU64
    )
}
impl Builder<'_> {
    fn build(&mut self, value: Value) -> Result<bool> {
        let value = self.ir.resolve(value)?;
        let Value::Inst(id) = value else {
            return self.node(value);
        };
        if let Some(slot) = self.slots.get(&id) {
            if let Some(slot) = slot {
                self.code.push(H::Load(*slot));
                return Ok(true);
            }
            return Ok(false);
        }
        self.slots.insert(id, None);
        let (start, first_slot, index) = (self.code.len(), self.next_slot, self.index);
        let built = self.node(value)?;
        self.slots.remove(&id);
        if !built {
            self.code.truncate(start);
            self.slots
                .retain(|_, slot| slot.is_none_or(|s| u16::from(s) < first_slot));
            self.next_slot = first_slot;
            self.index = index;
        } else if self.ir.inst(id)?.use_count() > 1 && self.next_slot < 256 {
            let slot = self.next_slot as u8;
            self.next_slot += 1;
            self.slots.insert(id, Some(slot));
            self.code.push(H::Tee(slot));
        }
        Ok(built)
    }
    fn arg(&mut self, id: InstId, arg: usize) -> Result<bool> {
        self.build(self.ir.inst(id)?.arg(arg)?)
    }
    fn binary(&mut self, id: InstId, op: H) -> Result<bool> {
        if !self.arg(id, 0)? || !self.arg(id, 1)? {
            return Ok(false);
        }
        self.code.push(op);
        Ok(true)
    }
    fn unary(&mut self, id: InstId, op: H) -> Result<bool> {
        if !self.arg(id, 0)? {
            return Ok(false);
        }
        self.code.push(op);
        Ok(true)
    }
    fn address(&mut self, value: Value) -> Result<bool> {
        let (allowed, scaled) = (self.allow_index, self.scaled);
        self.allow_index &= self.load_depth == 0;
        self.scaled = 0;
        self.load_depth += 1;
        let built = self.build(value);
        self.load_depth -= 1;
        self.scaled = scaled;
        self.allow_index = allowed;
        built
    }
    fn load(&mut self, id: InstId, source: InstId, extra: u32) -> Result<bool> {
        let inst = self.ir.inst(source)?;
        let cbuf = matches!(inst.opcode(), O::GetCbufU32 | O::GetCbufU32x2);
        if !cbuf
            && !matches!(
                inst.opcode(),
                O::LoadStorage32 | O::LoadStorage64 | O::LoadStorage128
            )
        {
            return Ok(false);
        }
        let Value::U32(binding) = self.ir.resolve(inst.arg(0)?)? else {
            return Ok(false);
        };
        let offset = self.ir.resolve(inst.arg(1)?)?;
        if cbuf && let Value::U32(offset) = offset {
            self.code.push(H::Cbuf(binding, offset.wrapping_add(extra)));
            return Ok(true);
        }
        if !self.address(offset)? {
            if !self.allow_index
                || self.load_depth == 0
                || self.scaled == 0
                || self.index.is_some_and(|i| i != id)
            {
                return Ok(false);
            }
            self.index = Some(id);
            self.code.push(H::Index);
            return Ok(true);
        }
        if extra != 0 {
            self.code.extend([H::Immediate(extra), H::Add]);
        }
        self.code.push(if cbuf {
            H::CbufDynamic(binding)
        } else {
            H::Storage(binding)
        });
        Ok(true)
    }
    fn mask(&mut self, id: InstId, bits: usize, offset: Option<usize>) -> Result<bool> {
        self.code.push(H::Immediate(1));
        if !self.arg(id, bits)? {
            return Ok(false);
        }
        self.code.extend([H::Shl, H::Immediate(1), H::Sub]);
        if let Some(offset) = offset {
            if !self.arg(id, offset)? {
                return Ok(false);
            }
            self.code.push(H::Shl);
        }
        Ok(true)
    }
    fn node(&mut self, value: Value) -> Result<bool> {
        let Some(budget) = self.budget.checked_sub(1) else {
            return Ok(false);
        };
        self.budget = budget;
        let id = match value {
            Value::U32(v) => {
                self.code.push(H::Immediate(v));
                return Ok(true);
            }
            Value::U1(v) => {
                self.code.push(H::Immediate(u32::from(v)));
                return Ok(true);
            }
            Value::Inst(id) => id,
            _ => return Ok(false),
        };
        let inst = self.ir.inst(id)?;
        match inst.opcode() {
            O::Identity
            | O::ConditionRef
            | O::ShuffleIndex
            | O::VoteAny
            | O::VoteAll
            | O::BitCastU32F32
            | O::BitCastF32U32 => self.arg(id, 0),
            O::GetCbufU32 | O::LoadStorage32 => self.load(id, id, 0),
            O::CompositeExtractU32x2 | O::CompositeExtractU32x3 | O::CompositeExtractU32x4 => {
                let (Value::Inst(source), Value::U32(element)) = (
                    self.ir.resolve(inst.arg(0)?)?,
                    self.ir.resolve(inst.arg(1)?)?,
                ) else {
                    return Ok(false);
                };
                self.load(id, source, element.wrapping_mul(4))
            }
            O::Phi => {
                let resolved = self
                    .ir
                    .resolve(conditional_phi(self.ir, value, self.block)?)?;
                if resolved != value {
                    return self.build(resolved);
                }
                let mut candidates = Vec::new();
                let mut chosen = None;
                for arg in 0..inst.num_args() {
                    let incoming = self.ir.resolve(inst.arg(arg)?)?;
                    if let Value::Inst(input) = incoming
                        && undefined(self.ir.inst(input)?.opcode())
                    {
                        continue;
                    }
                    candidates.push(arg);
                    let shuffled = matches!(incoming, Value::Inst(input) if self.ir.inst(input)?.opcode() == O::ShuffleIndex);
                    if chosen.is_none() || shuffled {
                        chosen = Some(arg);
                    }
                }
                if let Some(arg) = chosen
                    && self.arg(id, arg)?
                {
                    return Ok(true);
                }
                for arg in candidates {
                    if Some(arg) != chosen && self.arg(id, arg)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            O::IAdd32 => self.binary(id, H::Add),
            O::ISub32 => self.binary(id, H::Sub),
            O::IMul32 | O::ShiftLeftLogical32 => {
                let shift = inst.opcode() == O::ShiftLeftLogical32;
                let scaled = self.ir.is_immediate(inst.arg(1)?)?
                    || (!shift && self.ir.is_immediate(inst.arg(0)?)?);
                self.scaled += u32::from(scaled);
                let built = self.binary(id, if shift { H::Shl } else { H::Mul });
                self.scaled -= u32::from(scaled);
                built
            }
            O::BitwiseAnd32 => self.binary(id, H::And),
            O::BitwiseOr32 => self.binary(id, H::Or),
            O::BitwiseXor32 => self.binary(id, H::Xor),
            O::ShiftRightLogical32 => self.binary(id, H::Shr),
            O::ShiftRightArithmetic32 => self.binary(id, H::Sar),
            O::INeg32 => self.unary(id, H::Neg),
            O::BitwiseNot32 => self.unary(id, H::Not),
            O::UMin32 => self.binary(id, H::UMin),
            O::UMax32 => self.binary(id, H::UMax),
            O::SMin32 => self.binary(id, H::SMin),
            O::SMax32 => self.binary(id, H::SMax),
            O::SLessThan => self.binary(id, H::SLt),
            O::SLessThanEqual => self.binary(id, H::SLe),
            O::SGreaterThan => self.binary(id, H::SGt),
            O::SGreaterThanEqual => self.binary(id, H::SGe),
            O::ULessThan => self.binary(id, H::ULt),
            O::ULessThanEqual => self.binary(id, H::ULe),
            O::UGreaterThan => self.binary(id, H::UGt),
            O::UGreaterThanEqual => self.binary(id, H::UGe),
            O::IEqual => self.binary(id, H::Eq),
            O::INotEqual => self.binary(id, H::Ne),
            O::LogicalAnd => self.binary(id, H::LAnd),
            O::LogicalOr => self.binary(id, H::LOr),
            O::LogicalXor => self.binary(id, H::LXor),
            O::LogicalNot => self.unary(id, H::LNot),
            O::GetZeroFromOp | O::GetSignFromOp => {
                let zero = inst.opcode() == O::GetZeroFromOp;
                if !self.arg(id, 0)? {
                    return Ok(false);
                }
                self.code
                    .extend([H::Immediate(0), if zero { H::Eq } else { H::SLt }]);
                Ok(true)
            }
            O::GetCarryFromOp => {
                let Value::Inst(add) = self.ir.resolve(inst.arg(0)?)? else {
                    return Ok(false);
                };
                if self.ir.inst(add)?.opcode() != O::IAdd32
                    || !self.build(Value::Inst(add))?
                    || !self.arg(add, 0)?
                {
                    return Ok(false);
                }
                self.code.push(H::ULt);
                Ok(true)
            }
            O::UndefU1 | O::UndefU32 => {
                self.code.push(H::Immediate(0));
                Ok(true)
            }
            O::BitFieldUExtract => {
                if !self.arg(id, 0)? || !self.arg(id, 1)? {
                    return Ok(false);
                }
                self.code.push(H::Shr);
                if !self.mask(id, 2, None)? {
                    return Ok(false);
                }
                self.code.push(H::And);
                Ok(true)
            }
            O::BitFieldSExtract => {
                if !self.arg(id, 0)? {
                    return Ok(false);
                }
                self.code.push(H::Immediate(32));
                if !self.arg(id, 1)? {
                    return Ok(false);
                }
                self.code.push(H::Sub);
                if !self.arg(id, 2)? {
                    return Ok(false);
                }
                self.code.extend([H::Sub, H::Shl, H::Immediate(32)]);
                if !self.arg(id, 2)? {
                    return Ok(false);
                }
                self.code.extend([H::Sub, H::Sar]);
                Ok(true)
            }
            O::BitFieldInsert => {
                if !self.arg(id, 0)? || !self.mask(id, 3, Some(2))? {
                    return Ok(false);
                }
                self.code.extend([H::Not, H::And]);
                if !self.arg(id, 1)? || !self.arg(id, 2)? {
                    return Ok(false);
                }
                self.code.push(H::Shl);
                if !self.mask(id, 3, Some(2))? {
                    return Ok(false);
                }
                self.code.extend([H::And, H::Or]);
                Ok(true)
            }
            O::UClamp32 | O::SClamp32 => {
                let signed = inst.opcode() == O::SClamp32;
                if !self.binary(id, if signed { H::SMax } else { H::UMax })? || !self.arg(id, 2)? {
                    return Ok(false);
                }
                self.code.push(if signed { H::SMin } else { H::UMin });
                Ok(true)
            }
            O::SelectU1 | O::SelectU32 => {
                if !self.arg(id, 0)? || !self.arg(id, 1)? || !self.arg(id, 2)? {
                    return Ok(false);
                }
                self.code.push(H::Select);
                Ok(true)
            }
            _ => Ok(false),
        }
    }
}
