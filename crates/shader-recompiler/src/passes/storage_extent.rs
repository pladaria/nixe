//! Prove store extents before replacing global addresses with storage offsets.
//! Unknown, loop-carried and memory-loaded offsets keep the complete binding.
//! https://github.com/zortos293/Lichii/blob/646b5d17135a2b2701ccc7a959afd0bbbf2f6160/src/shader_recompiler/ir_opt/global_memory_to_storage_buffer_pass.cpp
use super::{
    fold::arg,
    storage::{cbuf_word, low_address},
};
use crate::{
    arena::InstId,
    artifact::StorageWriteExtent,
    ir::{Ir, Opcode as O, Value},
};
use std::collections::{HashMap, HashSet};

/// Exact expressions are modulo 2^32. Inexact expressions are nonnegative
/// upper bounds: subtraction or multiplication by a wrapping value is unsafe.
#[derive(Clone, Copy)]
struct Bound {
    c: i64,
    coef: [u64; 6],
    base: u32,
    exact: bool,
}
impl Bound {
    fn constant(c: i64) -> Self {
        Self {
            c,
            coef: [0; 6],
            base: 0,
            exact: true,
        }
    }
    fn is_constant(self) -> bool {
        self.base == 0 && self.coef == [0; 6]
    }
    fn unsigned(self) -> bool {
        self.base == 0 && self.c >= 0
    }
    fn inexact(mut self) -> Self {
        self.exact = false;
        self
    }
    fn checked(self) -> Option<Self> {
        (self.c.unsigned_abs() <= 1 << 40
            && self.base <= 1
            && self.coef.iter().all(|&c| c < 1 << 32))
        .then_some(self)
    }
    fn sum(self, rhs: Self) -> Option<Self> {
        if (!self.exact && rhs.c < 0) || (!rhs.exact && self.c < 0) {
            return None;
        }
        Self {
            c: self.c + rhs.c,
            base: self.base + rhs.base,
            exact: self.exact && rhs.exact,
            coef: std::array::from_fn(|i| self.coef[i] + rhs.coef[i]),
        }
        .checked()
    }
    fn max(self, rhs: Self) -> Option<Self> {
        if self.base != rhs.base || self.c < 0 || rhs.c < 0 {
            return None;
        }
        Some(Self {
            c: self.c.max(rhs.c),
            base: self.base,
            exact: false,
            coef: std::array::from_fn(|i| self.coef[i].max(rhs.coef[i])),
        })
    }
    fn scale(self, factor: u64, exact: bool) -> Option<Self> {
        if self.base != 0 || factor >= 1 << 32 || (!exact && self.c < 0) {
            return None;
        }
        let mut result = Self::constant(self.c.checked_mul(factor as i64)?);
        result.exact = self.exact && exact;
        for (out, coef) in result.coef.iter_mut().zip(self.coef) {
            *out = coef.checked_mul(factor)?;
        }
        result.checked()
    }
    fn shift_down(self, shift: u32) -> Option<Self> {
        if !self.unsigned() || shift >= 32 {
            return None;
        }
        Some(Self {
            c: (self.c as u64).div_ceil(1 << shift) as i64,
            coef: self.coef.map(|c| c.div_ceil(1 << shift)),
            base: 0,
            exact: false,
        })
    }
}

pub(super) struct StoreExtentAnalysis<'a> {
    ir: &'a Ir,
    pointer: (u32, u32),
    workgroup: [u32; 3],
    memo: HashMap<InstId, Option<Bound>>,
    active: HashSet<InstId>,
    guard: Option<Bound>,
    guard_failed: bool,
}
impl<'a> StoreExtentAnalysis<'a> {
    pub(super) fn new(ir: &'a Ir, pointer: (u32, u32), workgroup: [u32; 3]) -> Self {
        Self {
            ir,
            pointer,
            workgroup,
            memo: HashMap::new(),
            active: HashSet::new(),
            guard: None,
            guard_failed: false,
        }
    }
    pub(super) fn store(&mut self, id: InstId) -> Option<StorageWriteExtent> {
        let (low, _, offset) = low_address(self.ir, id).ok()??;
        let low = self.evaluate(low, 0)?;
        if low.base != 1 || self.guard_failed {
            return None;
        }
        let start = low.c + i64::from(offset as i32);
        let end = start + i64::from(store_size(self.ir.inst(id).ok()?.opcode()));
        if start < 0 || end > i64::from(u32::MAX) {
            return None;
        }
        let mut extent = StorageWriteExtent {
            bounded: true,
            const_end: end as u32,
            ..Default::default()
        };
        for i in 0..3 {
            extent.local_stride[i] = u32::try_from(low.coef[i]).ok()?;
            extent.group_stride[i] = u32::try_from(low.coef[i + 3]).ok()?;
        }
        if let Some(guard) = self.guard {
            extent.guarded = true;
            extent.guard_const = u32::from(u16::try_from(guard.c).ok()?);
            for i in 0..3 {
                extent.guard_local_stride[i] = u32::from(u16::try_from(guard.coef[i]).ok()?);
                extent.guard_group_stride[i] = u32::from(u16::try_from(guard.coef[i + 3]).ok()?);
            }
        }
        Some(extent)
    }
    fn static_max(&self, value: Bound) -> Option<u64> {
        if !value.unsigned() || value.coef[3..] != [0; 3] {
            return None;
        }
        let mut total = value.c as u64;
        for i in 0..3 {
            total = total.checked_add(
                value.coef[i].checked_mul(u64::from(self.workgroup[i].checked_sub(1)?))?,
            )?;
        }
        Some(total)
    }
    fn guard(&mut self, value: Bound) {
        if !value.unsigned() {
            self.guard_failed = true;
            return;
        }
        self.guard = match self.guard {
            None => Some(value),
            Some(prior) => prior.max(value),
        };
        self.guard_failed |= self.guard.is_none();
    }
    fn immediate(&self, id: InstId, index: usize) -> Option<u32> {
        match arg(self.ir, id, index).ok()? {
            Value::U32(value) => Some(value),
            _ => None,
        }
    }
    fn operand(&mut self, id: InstId, index: usize, depth: u32) -> Option<Bound> {
        self.evaluate(arg(self.ir, id, index).ok()?, depth + 1)
    }
    fn extract(&mut self, id: InstId, depth: u32) -> Option<Bound> {
        let offset = self.immediate(id, 1)?;
        let count = self.immediate(id, 2)?;
        if offset >= 32 || count == 0 || count > 32 {
            return None;
        }
        if let Some(value) = self.immediate(id, 0) {
            return Some(Bound::constant(
                ((u64::from(value) >> offset) & ((1u64 << count) - 1)) as i64,
            ));
        }
        let source = self.operand(id, 0, depth);
        let field_max = (count <= 24).then(|| Bound::constant((1i64 << count) - 1).inexact());
        let Some(source) = source.filter(|s| s.unsigned()) else {
            return field_max;
        };
        if self
            .static_max(source)
            .is_some_and(|max| max >> offset == 0)
        {
            return Some(Bound::constant(0));
        }
        if offset == 0 {
            return Some(if count == 32 {
                source
            } else {
                source.inexact()
            });
        }
        if offset >= 16 {
            self.guard(source);
            return Some(Bound::constant(0));
        }
        source.shift_down(offset)
    }
    fn evaluate(&mut self, value: Value, depth: u32) -> Option<Bound> {
        let id = match self.ir.resolve(value).ok()? {
            Value::U32(value) => return Some(Bound::constant(i64::from(value as i32))),
            Value::Inst(id) if depth <= 256 => id,
            _ => return None,
        };
        if let Some(result) = self.memo.get(&id) {
            return *result;
        }
        if !self.active.insert(id) {
            return None;
        }
        let result = self.instruction(id, depth);
        self.active.remove(&id);
        self.memo.insert(id, result);
        result
    }
    fn instruction(&mut self, id: InstId, depth: u32) -> Option<Bound> {
        if let Some(word) = cbuf_word(self.ir, id).ok()? {
            return (word == self.pointer).then_some(Bound {
                base: 1,
                ..Bound::constant(0)
            });
        }
        let inst = self.ir.inst(id).ok()?;
        match inst.opcode() {
            O::CompositeExtractU32x3 => {
                let Value::Inst(vector) = arg(self.ir, id, 0).ok()? else {
                    return None;
                };
                let element = self.immediate(id, 1)? as usize;
                if element > 2 {
                    return None;
                }
                let index = match self.ir.inst(vector).ok()?.opcode() {
                    O::LocalInvocationId => element,
                    O::WorkgroupId => 3 + element,
                    _ => return None,
                };
                let mut result = Bound::constant(0);
                result.coef[index] = 1;
                Some(result)
            }
            O::IAdd32 => self.operand(id, 0, depth)?.sum(self.operand(id, 1, depth)?),
            O::ISub32 => {
                let lhs = self.operand(id, 0, depth)?;
                let rhs = self.operand(id, 1, depth)?;
                if !rhs.is_constant() || !rhs.exact {
                    return None;
                }
                lhs.sum(Bound::constant(-rhs.c))
            }
            O::IMul32 => {
                let lhs = self.operand(id, 0, depth)?;
                let rhs = self.operand(id, 1, depth)?;
                if rhs.is_constant() && rhs.c >= 0 {
                    lhs.scale(rhs.c as u64, rhs.exact)
                } else if lhs.is_constant() && lhs.c >= 0 {
                    rhs.scale(lhs.c as u64, lhs.exact)
                } else {
                    None
                }
            }
            O::ShiftLeftLogical32 => {
                let shift = self.immediate(id, 1)?;
                if shift >= 32 {
                    return None;
                }
                self.operand(id, 0, depth)?.scale(1 << shift, true)
            }
            O::ShiftRightLogical32 => {
                let shift = self.immediate(id, 1)?;
                self.operand(id, 0, depth)?.shift_down(shift)
            }
            O::BitwiseAnd32 => {
                let lhs = self.operand(id, 0, depth);
                let rhs = self.operand(id, 1, depth);
                [rhs, lhs]
                    .into_iter()
                    .flatten()
                    .find(|b| b.unsigned() && b.is_constant() && b.c <= 0xffffff)
                    .or_else(|| [lhs, rhs].into_iter().flatten().find(|b| b.unsigned()))
                    .map(Bound::inexact)
            }
            O::BitwiseOr32 | O::BitwiseXor32 | O::UMax32 => {
                let lhs = self.operand(id, 0, depth)?;
                let rhs = self.operand(id, 1, depth)?;
                if !lhs.unsigned() || !rhs.unsigned() {
                    return None;
                }
                lhs.sum(rhs).map(Bound::inexact)
            }
            O::UMin32 => {
                let lhs = self.operand(id, 0, depth);
                let rhs = self.operand(id, 1, depth);
                [rhs, lhs]
                    .into_iter()
                    .flatten()
                    .find(|b| b.unsigned())
                    .map(Bound::inexact)
            }
            O::BitFieldUExtract => self.extract(id, depth),
            O::BitFieldInsert => {
                let offset = self.immediate(id, 2)?;
                self.immediate(id, 3)?;
                if offset >= 32 {
                    return None;
                }
                let base = self.operand(id, 0, depth)?;
                let insert = self.operand(id, 1, depth)?;
                if !base.unsigned() || !insert.unsigned() {
                    return None;
                }
                base.sum(insert.scale(1 << offset, true)?)
                    .map(Bound::inexact)
            }
            O::SelectU32 => self.operand(id, 1, depth)?.max(self.operand(id, 2, depth)?),
            O::Phi => {
                let mut result: Option<Bound> = None;
                for index in 0..inst.num_args() {
                    let incoming = self.operand(id, index, depth)?;
                    result = Some(match result {
                        Some(prior) => prior.max(incoming)?,
                        None => incoming,
                    });
                }
                result
            }
            _ => None,
        }
    }
}
fn store_size(op: O) -> u32 {
    match op {
        O::WriteGlobalS8 | O::WriteGlobalU8 => 1,
        O::WriteGlobalS16 | O::WriteGlobalU16 => 2,
        O::WriteGlobal128 => 16,
        O::WriteGlobal64
        | O::GlobalAtomicIAdd64
        | O::GlobalAtomicSMin64
        | O::GlobalAtomicUMin64
        | O::GlobalAtomicSMax64
        | O::GlobalAtomicUMax64
        | O::GlobalAtomicAnd64
        | O::GlobalAtomicOr64
        | O::GlobalAtomicXor64
        | O::GlobalAtomicExchange64
        | O::GlobalAtomicIAdd32x2
        | O::GlobalAtomicSMin32x2
        | O::GlobalAtomicUMin32x2
        | O::GlobalAtomicSMax32x2
        | O::GlobalAtomicUMax32x2
        | O::GlobalAtomicAnd32x2
        | O::GlobalAtomicOr32x2
        | O::GlobalAtomicXor32x2
        | O::GlobalAtomicExchange32x2
        | O::GlobalAtomicAddF32x2
        | O::GlobalAtomicMinF32x2
        | O::GlobalAtomicMaxF32x2 => 8,
        _ => 4,
    }
}
