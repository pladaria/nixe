//! Proven binary quantum of finite FP32 values: every finite value is an integer
//! multiple of 2^q. Unknown FP32 has q=-149; unpacked FP16 has q=-24. Addition
//! takes the minimum quantum, multiplication adds them, and FMA combines both.
//! If operands and result have q>=-126, none can be a nonzero FP32 subnormal.
//! NaNs, infinities and signed zero still use the ordinary host guarantees.
use super::*;
use std::collections::VecDeque;

const UNKNOWN: i16 = -149;
const MAXIMUM: i16 = 127;

#[derive(Clone, Copy)]
enum Source<'a> {
    Phi(&'a Phi),
    Instruction(&'a Instruction),
}

pub(super) struct FloatFacts {
    quanta: Vec<i16>,
}
impl FloatFacts {
    pub(super) fn analyze(function: &Function) -> Self {
        let n = function.values.len();
        let mut sources = vec![None; n];
        let mut users = vec![Vec::new(); n];
        for block in &function.blocks {
            for phi in &block.phis {
                sources[phi.result.index()] = Some(Source::Phi(phi));
                for &(_, value) in &phi.incoming {
                    users[value.index()].push(phi.result.index());
                }
            }
            for inst in &block.instructions {
                for &result in &inst.results {
                    sources[result.index()] = Some(Source::Instruction(inst));
                    for arg in &inst.arguments {
                        users[arg.index()].push(result.index());
                    }
                }
            }
        }
        let mut result = Self {
            quanta: vec![UNKNOWN; n],
        };
        let mut queue: VecDeque<_> = (0..n).collect();
        let mut queued = vec![true; n];
        // Monotone, bounded lattice; revisit users only when a proof improves.
        // A loop starts conservatively, never by assuming its backedge's range.
        while let Some(value) = queue.pop_front() {
            queued[value] = false;
            let q = match sources[value] {
                Some(Source::Phi(phi)) => phi
                    .incoming
                    .iter()
                    .map(|&(_, v)| result.quanta[v.index()])
                    .min()
                    .unwrap_or(UNKNOWN),
                Some(Source::Instruction(inst))
                    if inst.results.first().is_some_and(|v| v.index() == value) =>
                {
                    result.instruction(inst)
                }
                _ => UNKNOWN,
            };
            if q > result.quanta[value] {
                result.quanta[value] = q;
                for &user in &users[value] {
                    if !queued[user] {
                        queued[user] = true;
                        queue.push_back(user);
                    }
                }
            }
        }
        result
    }
    fn instruction(&self, inst: &Instruction) -> i16 {
        let q = |i: usize| self.quanta[inst.arguments[i].index()];
        let controlled = |q: i16, c: ShaderFloatControl| {
            if c.saturate { q.min(0) } else { q }.clamp(UNKNOWN, MAXIMUM)
        };
        match inst.op {
            Op::UnpackHalf
            | Op::HalfFma { .. }
            | Op::Convert {
                from: Type::F16, ..
            }
            | Op::Convert { to: Type::F16, .. } => -24,
            Op::Constant(c) if matches!(c.ty, Type::F32 | Type::U32 | Type::I32) => {
                let bits = c.bits as u32;
                let exponent = (bits >> 23) & 255;
                let fraction = bits & 0x7fffff;
                if exponent == 255 || bits & 0x7fffffff == 0 {
                    MAXIMUM
                } else if exponent == 0 {
                    UNKNOWN + fraction.trailing_zeros() as i16
                } else {
                    exponent as i16 - 150 + (fraction | 0x800000).trailing_zeros() as i16
                }
            }
            Op::Copy(_) => q(0),
            Op::Bitcast { from, to } if from.bits() == Some(32) && to.bits() == Some(32) => q(0),
            Op::Unary {
                op: Unary::Absolute | Unary::Negate,
                ty: Type::F32,
            } => q(0),
            Op::Select(_) => q(1).min(q(2)),
            Op::RoundIntegral { .. } => 0,
            Op::Convert {
                from: Type::U32 | Type::I32 | Type::U64 | Type::I64,
                to: Type::F32,
                ..
            } => 0,
            Op::Binary {
                op: Binary::Add,
                ty: Type::F32,
                float_control,
            } => controlled(q(0).min(q(1)), float_control),
            Op::Binary {
                op: Binary::Multiply,
                ty: Type::F32,
                float_control,
            } => controlled(q(0) + q(1), float_control),
            Op::Fma {
                ty: Type::F32,
                float_control,
            } => controlled((q(0) + q(1)).min(q(2)), float_control),
            _ => UNKNOWN,
        }
    }
    pub(super) fn normal(&self, inst: &Instruction) -> bool {
        inst.arguments
            .iter()
            .chain(inst.results.first())
            .all(|v| self.quanta[v.index()] >= -126)
    }
}
