//! A reduced SFU encoding may be preserved by packed writes and overwritten
//! before any bit is observed. Prove that case; never approximate its encoding.
use super::*;
use nixe_gpu::ssa::{Definition, ValueId};
use std::collections::VecDeque;

fn mask(ty: Type) -> u64 {
    match ty {
        Type::Bool => 1,
        Type::F16 => 0xffff,
        Type::U32 | Type::I32 | Type::F32 => u64::from(u32::MAX),
        Type::U64 | Type::I64 | Type::F64 => u64::MAX,
        Type::Effect => 0,
    }
}
mod control;
struct Demand {
    bits: Vec<[u64; 2]>,
    queued: Vec<[bool; 2]>,
    pending: VecDeque<(ValueId, usize)>,
}
impl Demand {
    fn add(&mut self, function: &Function, states: &[u8], value: ValueId, bits: u64, contexts: u8) {
        let block = match function.values[value.index()].definition {
            Definition::Phi { block } | Definition::Instruction { block, .. } => block,
        };
        let contexts = contexts & states[block.index()];
        let bits = bits & mask(function.values[value.index()].ty);
        for context in 0..2 {
            if contexts & (1 << context) != 0 && bits & !self.bits[value.index()][context] != 0 {
                self.bits[value.index()][context] |= bits;
                if !std::mem::replace(&mut self.queued[value.index()][context], true) {
                    self.pending.push_back((value, context));
                }
            }
        }
    }
}

pub(super) fn resolve_range_bits(
    binary: &MaxwellShaderBinary,
    function: &mut Function,
) -> Result<(), MaxwellShaderTranslationError> {
    let ranges: BTreeMap<_, _> = binary
        .bundles()
        .iter()
        .flat_map(|b| {
            b.instructions
                .iter()
                .enumerate()
                .filter(|&(_, &encoding)| super::super::special::is_range_reduction(encoding))
                .map(move |(slot, &encoding)| (b.offset + 8 + slot as u32 * 8, encoding))
        })
        .collect();
    let mut opaque = Vec::new();
    for (block_index, block) in function.blocks.iter().enumerate() {
        for (index, inst) in block.instructions.iter().enumerate() {
            if matches!(inst.op, Op::Undefined(Type::U32))
                && ranges.contains_key(&inst.source.byte_offset())
            {
                opaque.push((block_index, index, inst.results[0]));
            }
        }
    }
    if opaque.is_empty() {
        return Ok(());
    }
    let control = control::Control::new(function);
    let mut groups = BTreeMap::<Option<ValueId>, Vec<_>>::new();
    for marker in opaque {
        groups
            .entry(control.guard(function, marker.0))
            .or_default()
            .push(marker);
    }
    let mut erase = Vec::new();
    for (base, markers) in groups {
        let states = control.states(function, base);
        let bits = live_bits(function, &control, &states, base);
        for (block, index, value) in markers {
            if bits[value.index()] != [0; 2] {
                let inst = &function.blocks[block].instructions[index];
                return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
                    stage: binary.stage(),
                    instruction_offset: inst.source.byte_offset(),
                    encoding: ranges[&inst.source.byte_offset()],
                    detail: "RRO reduced representation has observable bits outside a compatible MUFU",
                });
            }
            erase.push((block, index));
        }
    }
    for (block, index) in erase {
        // The opaque value has no observable bits in any feasible guard context.
        function.blocks[block].instructions[index].op = Op::Constant(Constant::u32(0));
    }
    function.optimize()?;
    Ok(())
}

fn live_bits(
    function: &Function,
    control: &control::Control,
    states: &[u8],
    base: Option<ValueId>,
) -> Vec<[u64; 2]> {
    let count = function.values.len();
    let mut demand = Demand {
        bits: vec![[0; 2]; count],
        queued: vec![[false; 2]; count],
        pending: VecDeque::new(),
    };
    let mut phis = vec![None; count];
    let mut constants = vec![None; count];
    for (block_index, block) in function.blocks.iter().enumerate() {
        let contexts = states[block_index];
        for phi in &block.phis {
            phis[phi.result.index()] = Some(phi);
        }
        for inst in &block.instructions {
            if let Op::Constant(c) = inst.op {
                constants[inst.results[0].index()] = Some(c.bits);
            }
            if inst.op.ordered() {
                for &arg in &inst.arguments {
                    demand.add(function, states, arg, u64::MAX, contexts);
                }
            }
        }
        match block.terminator {
            Terminator::Branch { condition, .. } => {
                demand.add(function, states, condition, 1, contexts)
            }
            Terminator::Switch { selector, .. } => {
                demand.add(function, states, selector, u64::MAX, contexts)
            }
            _ => {}
        }
    }
    while let Some((value, context)) = demand.pending.pop_front() {
        demand.queued[value.index()][context] = false;
        let bits = demand.bits[value.index()][context];
        let contexts = 1 << context;
        let Definition::Instruction { block, index, .. } =
            function.values[value.index()].definition
        else {
            let Definition::Phi { block } = function.values[value.index()].definition else {
                unreachable!()
            };
            for &(previous, incoming) in
                &phis[value.index()].expect("dense phi definition").incoming
            {
                let feasible = control.edge(function, states, base, previous, block.index());
                demand.add(function, states, incoming, bits, contexts & feasible);
            }
            continue;
        };
        let inst = &function.blocks[block.index()].instructions[index as usize];
        let args = &inst.arguments;
        let get = |i: usize| constants[args[i].index()];
        let mut add = |i, b, c| demand.add(function, states, args[i], b, c);
        match inst.op {
            Op::Copy(_) | Op::Bitcast { .. } | Op::Unary { op: Unary::Not, .. } => {
                add(0, bits, contexts)
            }
            Op::Select(_) => {
                add(0, 1, contexts);
                let (condition, inverted) = control::condition(function, args[0]);
                if base == Some(condition) {
                    add(1, bits, contexts & if inverted { 1 } else { 2 });
                    add(2, bits, contexts & if inverted { 2 } else { 1 });
                } else {
                    add(1, bits, contexts);
                    add(2, bits, contexts);
                }
            }
            Op::UnpackHalf => add(0, 0xffff, contexts),
            Op::Binary { op, ty, .. }
                if matches!(
                    ty,
                    Type::U32 | Type::I32 | Type::U64 | Type::I64 | Type::Bool
                ) =>
            {
                match op {
                    Binary::And => {
                        add(0, bits & get(1).unwrap_or(u64::MAX), contexts);
                        add(1, bits & get(0).unwrap_or(u64::MAX), contexts);
                    }
                    Binary::Or => {
                        add(0, bits & !get(1).unwrap_or(0), contexts);
                        add(1, bits & !get(0).unwrap_or(0), contexts);
                    }
                    Binary::Xor => {
                        add(0, bits, contexts);
                        add(1, bits, contexts);
                    }
                    Binary::ShiftLeft
                        if get(1).is_some_and(|n| n < u64::from(mask(ty).count_ones())) =>
                    {
                        add(0, bits >> get(1).unwrap(), contexts)
                    }
                    Binary::ShiftRight
                        if get(1).is_some_and(|n| n < u64::from(mask(ty).count_ones())) =>
                    {
                        let shift = get(1).unwrap();
                        let mut input = bits << shift;
                        if matches!(ty, Type::I32 | Type::I64) {
                            let sign = if ty == Type::I32 { 31 } else { 63 };
                            if shift > sign || bits >> (sign - shift) != 0 {
                                input |= 1 << sign;
                            }
                        }
                        add(0, input, contexts);
                    }
                    Binary::Add | Binary::Multiply => {
                        let prefix = u64::MAX >> bits.leading_zeros();
                        add(0, prefix, contexts);
                        add(1, prefix, contexts);
                    }
                    _ => {
                        for i in 0..args.len() {
                            add(i, u64::MAX, contexts);
                        }
                    }
                }
            }
            _ => {
                for i in 0..args.len() {
                    add(i, u64::MAX, contexts);
                }
            }
        }
    }
    demand.bits
}
