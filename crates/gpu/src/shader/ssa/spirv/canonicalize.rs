//! Exact cleanup after numerical lowering. Generated repair code participates
//! too; no floating algebra, contraction, rounding or execution mode is changed.
use rspirv::{
    dr::{Module, Operand},
    spirv::Op,
};
use std::collections::HashMap;

fn resolve(aliases: &mut [u32], id: u32) -> u32 {
    let mut root = id;
    while aliases[root as usize] != root {
        root = aliases[root as usize];
    }
    let mut current = id;
    while aliases[current as usize] != root {
        let next = aliases[current as usize];
        aliases[current as usize] = root;
        current = next;
    }
    root
}

pub(super) fn run(module: &mut Module) {
    let bound = module.header.as_ref().expect("emitted module header").bound as usize;
    let mut aliases: Vec<_> = (0..bound as u32).collect();
    let mut types = vec![0; bound];
    let mut casts = vec![None; bound];
    let mut decorated = vec![false; bound];
    for inst in &module.annotations {
        for id in inst.operands.iter().filter_map(Operand::id_ref_any) {
            decorated[id as usize] = true;
        }
    }
    for inst in module.all_inst_iter() {
        if let (Some(id), Some(ty)) = (inst.result_id, inst.result_type) {
            types[id as usize] = ty;
            if inst.class.opcode == Op::Bitcast {
                casts[id as usize] = inst.operands[0].id_ref_any();
            }
        }
    }
    // Decorated values retain their identity: in particular, nonuniform and
    // relaxed-precision metadata must not migrate onto unrelated producers.
    // https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpBitcast
    let mut roots = vec![0u32; bound];
    let mut path = Vec::new();
    for id in 1..bound {
        if casts[id].is_none() || decorated[id] || roots[id] != 0 {
            continue;
        }
        path.clear();
        let mut source = id;
        while roots[source] == 0 && !decorated[source] {
            let Some(parent) = casts[source] else { break };
            path.push(source);
            source = parent as usize;
        }
        let root = if roots[source] != 0 {
            roots[source]
        } else {
            source as u32
        };
        for &cast in &path {
            roots[cast] = root;
        }
    }
    for id in 1..bound {
        if roots[id] != 0 && types[roots[id] as usize] == types[id] {
            aliases[id] = roots[id];
        }
    }
    let mut constants = HashMap::new();
    for inst in &mut module.types_global_values {
        if matches!(
            inst.class.opcode,
            Op::Constant
                | Op::ConstantTrue
                | Op::ConstantFalse
                | Op::ConstantNull
                | Op::ConstantComposite
        ) {
            let id = inst.result_id.expect("constant result");
            if decorated[id as usize] {
                continue;
            }
            for operand in &mut inst.operands {
                if let Some(id) = operand.id_ref_any_mut() {
                    *id = resolve(&mut aliases, *id);
                }
            }
            // Literal operands retain their exact bit patterns and type,
            // including signed zero and distinct NaN payloads.
            let key = (inst.class.opcode, inst.result_type, inst.operands.clone());
            if let Some(&existing) = constants.get(&key) {
                aliases[id as usize] = existing;
            } else {
                constants.insert(key, id);
            }
        }
    }
    for id in 1..bound {
        resolve(&mut aliases, id as u32);
    }
    module
        .types_global_values
        .retain(|i| i.result_id.is_none_or(|id| aliases[id as usize] == id));
    for function in &mut module.functions {
        for block in &mut function.blocks {
            block
                .instructions
                .retain(|i| i.result_id.is_none_or(|id| aliases[id as usize] == id));
        }
    }
    for inst in module.all_inst_iter_mut() {
        if inst.class.opcode == Op::Bitcast
            && let Some(id) = inst.result_id
            && roots[id as usize] != 0
        {
            inst.operands[0] = Operand::IdRef(roots[id as usize]);
        }
        for operand in &mut inst.operands {
            if let Some(id) = operand.id_ref_any_mut() {
                *id = aliases[*id as usize];
            }
        }
    }
    // Remove now-unused casts by use counts. Each edge is visited once, without
    // repeatedly scanning the module as longer inverse chains disappear.
    let mut uses = vec![0usize; bound];
    casts.fill(None);
    for inst in module.all_inst_iter() {
        for id in inst.operands.iter().filter_map(Operand::id_ref_any) {
            uses[id as usize] += 1;
        }
        if inst.class.opcode == Op::Bitcast {
            casts[inst.result_id.unwrap() as usize] = inst.operands[0].id_ref_any();
        }
    }
    let mut pending: Vec<_> = casts
        .iter()
        .enumerate()
        .filter_map(|(id, source)| {
            (source.is_some() && uses[id] == 0 && !decorated[id]).then_some(id)
        })
        .collect();
    let mut dead = vec![false; bound];
    while let Some(id) = pending.pop() {
        dead[id] = true;
        let source = casts[id].expect("dead cast source") as usize;
        uses[source] -= 1;
        if uses[source] == 0 && casts[source].is_some() && !decorated[source] {
            pending.push(source);
        }
    }
    for function in &mut module.functions {
        for block in &mut function.blocks {
            block
                .instructions
                .retain(|i| i.result_id.is_none_or(|id| !dead[id as usize]));
        }
    }
    // Dense IDs avoid leaving the driver an unnecessarily large ID bound after
    // cleanup. All operand ID classes, including scope/semantics, participate.
    let mut ids = vec![0u32; bound];
    let mut next = 1;
    for inst in module.all_inst_iter() {
        if let Some(id) = inst.result_id {
            ids[id as usize] = next;
            next += 1;
        }
    }
    for inst in module.all_inst_iter_mut() {
        if let Some(id) = &mut inst.result_type {
            *id = ids[*id as usize];
        }
        if let Some(id) = &mut inst.result_id {
            *id = ids[*id as usize];
        }
        for operand in &mut inst.operands {
            if let Some(id) = operand.id_ref_any_mut() {
                *id = ids[*id as usize];
            }
        }
    }
    module.header.as_mut().unwrap().bound = next;
}

#[cfg(test)]
mod tests {
    use super::*;
    use rspirv::{dr::Builder, spirv as spv};

    #[test]
    fn exact_constants_and_inverse_casts_preserve_payloads_types_and_id_operands() {
        let mut b = Builder::new();
        let uint = b.type_int(32, 0);
        let int = b.type_int(32, 1);
        let float = b.type_float(32, None);
        let half = b.type_float(16, None);
        for bits in [0, 0, 0x80000000, 0x7fc00001, 0x7fc00001, 0x7fc00002] {
            b.constant_bit32(float, bits);
        }
        b.constant_bit32(half, 0);
        let scope = b.constant_bit32(uint, 2);
        let duplicate_scope = b.constant_bit32(uint, 2);
        let semantics = b.constant_bit32(uint, 0);
        let ty = b.type_function(uint, [uint]);
        b.begin_function(uint, None, spv::FunctionControl::NONE, ty)
            .unwrap();
        let input = b.function_parameter(uint).unwrap();
        b.begin_block(None).unwrap();
        let a = b.bitcast(float, None, input).unwrap();
        let a = b.bitcast(int, None, a).unwrap();
        let a = b.bitcast(float, None, a).unwrap();
        let a = b.bitcast(uint, None, a).unwrap();
        b.control_barrier(scope, duplicate_scope, semantics)
            .unwrap();
        b.ret_value(a).unwrap();
        b.end_function().unwrap();
        let mut m = b.module();
        run(&mut m);
        assert!(!m.all_inst_iter().any(|i| i.class.opcode == Op::Bitcast));
        let f32_type = m
            .types_global_values
            .iter()
            .find(|i| i.class.opcode == Op::TypeFloat && i.operands == [Operand::LiteralBit32(32)])
            .unwrap()
            .result_id;
        let literals: Vec<_> = m
            .types_global_values
            .iter()
            .filter(|i| i.class.opcode == Op::Constant && i.result_type == f32_type)
            .map(|i| i.operands[0].unwrap_literal_bit32())
            .collect();
        assert_eq!(literals, [0, 0x80000000, 0x7fc00001, 0x7fc00002]);
        let barrier = m
            .all_inst_iter()
            .find(|i| i.class.opcode == Op::ControlBarrier)
            .unwrap();
        assert_eq!(
            barrier.operands[0].id_ref_any(),
            barrier.operands[1].id_ref_any()
        );
        let ids: Vec<_> = m.all_inst_iter().filter_map(|i| i.result_id).collect();
        assert_eq!(
            ids,
            (1..m.header.as_ref().unwrap().bound).collect::<Vec<_>>()
        );
        let ret = m
            .all_inst_iter()
            .find(|i| i.class.opcode == Op::ReturnValue)
            .unwrap();
        assert_eq!(
            ret.operands[0].id_ref_any(),
            m.functions[0].parameters[0].result_id
        );
    }

    #[test]
    fn decorated_casts_and_precision_metadata_keep_their_identity() {
        let mut b = Builder::new();
        let uint = b.type_int(32, 0);
        let float = b.type_float(32, None);
        let ty = b.type_function(uint, [uint]);
        b.begin_function(uint, None, spv::FunctionControl::NONE, ty)
            .unwrap();
        let input = b.function_parameter(uint).unwrap();
        b.begin_block(None).unwrap();
        let a = b.bitcast(float, None, input).unwrap();
        b.decorate(a, spv::Decoration::RelaxedPrecision, []);
        let value = b.bitcast(uint, None, a).unwrap();
        b.ret_value(value).unwrap();
        b.end_function().unwrap();
        let mut m = b.module();
        run(&mut m);
        assert_eq!(
            m.all_inst_iter()
                .filter(|i| i.class.opcode == Op::Bitcast)
                .count(),
            2
        );
        let target = m.annotations[0].operands[0].id_ref_any().unwrap();
        assert!(
            m.all_inst_iter()
                .any(|i| i.result_id == Some(target) && i.class.opcode == Op::Bitcast)
        );
    }
}
