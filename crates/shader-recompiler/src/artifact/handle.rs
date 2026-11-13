//! Draw-time handle programs over constant and storage buffer words.
//! https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/shader_recompiler/shader_info.h#L424

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum HandleOp {
    Immediate(u32),
    Cbuf(u32, u32),
    CbufDynamic(u32),
    Storage(u32),
    Add,
    Sub,
    Mul,
    And,
    Or,
    Xor,
    Shl,
    Shr,
    Sar,
    Neg,
    Not,
    UMin,
    UMax,
    SMin,
    SMax,
    SLt,
    SLe,
    SGt,
    SGe,
    ULt,
    ULe,
    UGt,
    UGe,
    Eq,
    Ne,
    LAnd,
    LOr,
    LXor,
    LNot,
    Select,
    Tee(u8),
    Load(u8),
    Index,
}

pub trait HandleMemory {
    type Error;
    fn cbuf(&mut self, index: u32, offset: u32) -> Result<u32, Self::Error>;
    fn storage(&mut self, binding: u32, offset: u32) -> Result<u32, Self::Error>;
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HandleProgram {
    code: Vec<HandleOp>,
}
impl HandleProgram {
    pub(crate) fn new(mut code: Vec<HandleOp>) -> Option<Self> {
        while matches!(code.last(), Some(HandleOp::Tee(_))) {
            code.pop();
        }
        let mut depth = 0usize;
        let mut initialized = [false; 256];
        for &op in &code {
            use HandleOp::*;
            let (take, put) = match op {
                Immediate(_) | Cbuf(..) | Index => (0, 1),
                Load(slot) if initialized[usize::from(slot)] => (0, 1),
                Load(_) => return None,
                Tee(slot) => {
                    initialized[usize::from(slot)] = true;
                    (1, 1)
                }
                Neg | Not | LNot | CbufDynamic(_) | Storage(_) => (1, 1),
                Select => (3, 1),
                _ => (2, 1),
            };
            depth = depth.checked_sub(take)? + put;
            if depth > 64 {
                return None;
            }
        }
        (depth == 1).then_some(Self { code })
    }
    pub(crate) fn remap_storage(&mut self, bindings: &[u32]) {
        for op in &mut self.code {
            if let HandleOp::Storage(binding) = op {
                *binding = bindings[*binding as usize];
            }
        }
    }
    pub fn evaluate<M: HandleMemory>(&self, memory: &mut M, element: u32) -> Result<u32, M::Error> {
        let mut stack = [0u32; 64];
        let mut slots = [0u32; 256];
        let mut depth = 0usize;
        for &op in &self.code {
            use HandleOp::*;
            let value = match op {
                Immediate(value) => value,
                Cbuf(index, offset) => memory.cbuf(index, offset)?,
                Load(slot) => slots[usize::from(slot)],
                Index => element,
                Tee(slot) => {
                    slots[usize::from(slot)] = stack[depth - 1];
                    continue;
                }
                Neg | Not | LNot | CbufDynamic(_) | Storage(_) => {
                    depth -= 1;
                    let a = stack[depth];
                    match op {
                        Neg => a.wrapping_neg(),
                        Not => !a,
                        LNot => u32::from(a == 0),
                        CbufDynamic(index) => memory.cbuf(index, a)?,
                        Storage(binding) => memory.storage(binding, a)?,
                        _ => unreachable!(),
                    }
                }
                Select => {
                    depth -= 3;
                    if stack[depth] != 0 {
                        stack[depth + 1]
                    } else {
                        stack[depth + 2]
                    }
                }
                _ => {
                    depth -= 2;
                    let a = stack[depth];
                    let b = stack[depth + 1];
                    let sa = a as i32;
                    let sb = b as i32;
                    match op {
                        Add => a.wrapping_add(b),
                        Sub => a.wrapping_sub(b),
                        Mul => a.wrapping_mul(b),
                        And => a & b,
                        Or => a | b,
                        Xor => a ^ b,
                        Shl => a.checked_shl(b).unwrap_or(0),
                        Shr => a.checked_shr(b).unwrap_or(0),
                        Sar => (sa >> b.min(31)) as u32,
                        UMin => a.min(b),
                        UMax => a.max(b),
                        SMin => sa.min(sb) as u32,
                        SMax => sa.max(sb) as u32,
                        SLt => u32::from(sa < sb),
                        SLe => u32::from(sa <= sb),
                        SGt => u32::from(sa > sb),
                        SGe => u32::from(sa >= sb),
                        ULt => u32::from(a < b),
                        ULe => u32::from(a <= b),
                        UGt => u32::from(a > b),
                        UGe => u32::from(a >= b),
                        Eq => u32::from(a == b),
                        Ne => u32::from(a != b),
                        LAnd => u32::from(a != 0 && b != 0),
                        LOr => u32::from(a != 0 || b != 0),
                        LXor => u32::from((a != 0) != (b != 0)),
                        _ => unreachable!(),
                    }
                }
            };
            stack[depth] = value;
            depth += 1;
        }
        Ok(stack[0])
    }
}
