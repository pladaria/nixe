use nixe_shader_recompiler::{
    arena::{BlockId, InstId},
    ir::{Emitter, Ir, Opcode as O, Pred, Program, Reg, Type, Value},
    passes::rewrite_ssa,
    types::Stage,
};
fn program(ir: &Ir, blocks: Vec<BlockId>) -> Program {
    let mut p = Program::new(Stage::Compute);
    p.post_order = ir
        .post_order(&nixe_shader_recompiler::ir::SyntaxNode::Block(blocks[0]))
        .unwrap();
    p.blocks = blocks;
    p
}
fn phis(ir: &Ir, b: BlockId) -> Vec<InstId> {
    ir.block(b)
        .unwrap()
        .instructions()
        .iter()
        .copied()
        .filter(|&id| ir.inst(id).unwrap().opcode() == O::Phi)
        .collect()
}
#[test]
fn diamond_reads_merge_distinct_values_and_phi_order_uses_block_order() {
    let mut ir = Ir::new().unwrap();
    let a = ir.new_block().unwrap();
    let b = ir.new_block().unwrap();
    let c = ir.new_block().unwrap();
    let d = ir.new_block().unwrap();
    ir.add_branch(a, b).unwrap();
    ir.add_branch(a, c).unwrap();
    ir.add_branch(b, d).unwrap();
    ir.add_branch(c, d).unwrap();
    ir.block_mut(b).unwrap().order = 2;
    ir.block_mut(c).unwrap().order = 1;
    Emitter::new(&mut ir, b)
        .unwrap()
        .set_reg(Reg(2), Value::U32(10))
        .unwrap();
    Emitter::new(&mut ir, c)
        .unwrap()
        .set_reg(Reg(2), Value::U32(20))
        .unwrap();
    let result = Emitter::new(&mut ir, d).unwrap().get_reg(Reg(2)).unwrap();
    let p = program(&ir, vec![a, b, c, d]);
    rewrite_ssa(&mut ir, &p).unwrap();
    ir.verify(&p).unwrap();
    let phi = phis(&ir, d)[0];
    assert_eq!(ir.resolve(result).unwrap(), Value::Inst(phi));
    assert_eq!(ir.inst(phi).unwrap().ty(), Type::U32);
    assert_eq!(ir.inst(phi).unwrap().phi_block(0).unwrap(), c);
    assert_eq!(
        ir.resolve(ir.inst(phi).unwrap().arg(0).unwrap()).unwrap(),
        Value::U32(20)
    );
    assert_eq!(ir.inst(phi).unwrap().phi_block(1).unwrap(), b);
}
#[test]
fn loop_header_seal_sequence_keeps_placeholder_back_edge() {
    let mut ir = Ir::new().unwrap();
    let entry = ir.new_block().unwrap();
    let header = ir.new_block().unwrap();
    let body = ir.new_block().unwrap();
    let exit = ir.new_block().unwrap();
    ir.add_branch(entry, header).unwrap();
    ir.add_branch(header, body).unwrap();
    ir.add_branch(header, exit).unwrap();
    ir.add_branch(body, header).unwrap();
    Emitter::new(&mut ir, entry)
        .unwrap()
        .set_reg(Reg(3), Value::U32(0))
        .unwrap();
    let counter = Emitter::new(&mut ir, header)
        .unwrap()
        .get_reg(Reg(3))
        .unwrap();
    let mut e = Emitter::new(&mut ir, body).unwrap();
    let old = e.get_reg(Reg(3)).unwrap();
    let next = e.i_add(old, Value::U32(1)).unwrap();
    e.set_reg(Reg(3), next).unwrap();
    let result = Emitter::new(&mut ir, exit)
        .unwrap()
        .get_reg(Reg(3))
        .unwrap();
    let p = program(&ir, vec![entry, header, body, exit]);
    rewrite_ssa(&mut ir, &p).unwrap();
    ir.verify(&p).unwrap();
    assert_eq!(ir.resolve(result).unwrap(), ir.resolve(counter).unwrap());
    let Value::Inst(phi) = ir.resolve(counter).unwrap() else {
        panic!()
    };
    let inst = ir.inst(phi).unwrap();
    assert_eq!(inst.opcode(), O::Phi);
    assert_eq!(inst.num_args(), 2);
    let args = (0..2)
        .map(|i| ir.resolve(inst.arg(i).unwrap()).unwrap())
        .collect::<Vec<_>>();
    assert!(
        args.contains(&Value::U32(0)) && args.contains(&Value::Inst(phi)) && !args.contains(&next),
        "{}",
        ir.dump_program(&p).unwrap()
    );
}
#[test]
fn predicates_flags_gotos_indirect_values_and_undefined_entry_values_keep_types() {
    let mut ir = Ir::new().unwrap();
    let b = ir.new_block().unwrap();
    let mut e = Emitter::new(&mut ir, b).unwrap();
    let undefined = e.get_reg(Reg(4)).unwrap();
    let undefined_pred = e.get_pred(Pred::new(1).unwrap(), false).unwrap();
    let pairs = [
        (O::SetZFlag, O::GetZFlag),
        (O::SetSFlag, O::GetSFlag),
        (O::SetCFlag, O::GetCFlag),
        (O::SetOFlag, O::GetOFlag),
    ];
    let mut outputs = Vec::new();
    for (set, get) in pairs {
        e.emit(set, &[Value::U1(true)], 0).unwrap();
        outputs.push(e.emit(get, &[], 0).unwrap());
    }
    e.set_pred(Pred::new(2).unwrap(), Value::U1(true)).unwrap();
    outputs.push(e.get_pred(Pred::new(2).unwrap(), false).unwrap());
    e.emit(O::SetGotoVariable, &[Value::U32(1000), Value::U1(true)], 0)
        .unwrap();
    outputs.push(e.emit(O::GetGotoVariable, &[Value::U32(1000)], 0).unwrap());
    e.emit(O::SetIndirectBranchVariable, &[Value::U32(64)], 0)
        .unwrap();
    let indirect = e.emit(O::GetIndirectBranchVariable, &[], 0).unwrap();
    e.set_reg(Reg::RZ, Value::U32(100)).unwrap();
    let rz = e.get_reg(Reg::RZ).unwrap();
    let pt = e.get_pred(Pred::PT, false).unwrap();
    let p = program(&ir, vec![b]);
    rewrite_ssa(&mut ir, &p).unwrap();
    ir.verify(&p).unwrap();
    for value in outputs {
        assert_eq!(ir.resolve(value).unwrap(), Value::U1(true));
    }
    assert_eq!(ir.resolve(indirect).unwrap(), Value::U32(64));
    assert_eq!(pt, Value::U1(true));
    assert_eq!(ir.resolve(rz).unwrap(), rz); // RZ folding is performed by constant propagation.
    for (value, opcode) in [(undefined, O::UndefU32), (undefined_pred, O::UndefU1)] {
        let Value::Inst(id) = ir.resolve(value).unwrap() else {
            panic!()
        };
        assert_eq!(ir.inst(id).unwrap().opcode(), opcode);
    }
}
#[test]
fn unchanged_loop_value_eliminates_self_phi_and_reinserts_identity_after_phis() {
    let mut ir = Ir::new().unwrap();
    let entry = ir.new_block().unwrap();
    let loop_block = ir.new_block().unwrap();
    ir.add_branch(entry, loop_block).unwrap();
    ir.add_branch(loop_block, loop_block).unwrap();
    Emitter::new(&mut ir, entry)
        .unwrap()
        .set_reg(Reg(5), Value::U32(42))
        .unwrap();
    let value = Emitter::new(&mut ir, loop_block)
        .unwrap()
        .get_reg(Reg(5))
        .unwrap();
    let p = program(&ir, vec![entry, loop_block]);
    rewrite_ssa(&mut ir, &p).unwrap();
    ir.verify(&p).unwrap();
    assert_eq!(ir.resolve(value).unwrap(), Value::U32(42));
    assert!(phis(&ir, loop_block).is_empty());
}
#[test]
fn long_predecessor_chain_uses_explicit_read_stack() {
    let mut ir = Ir::new().unwrap();
    let mut blocks = Vec::new();
    for _ in 0..12000 {
        let b = ir.new_block().unwrap();
        if let Some(&prev) = blocks.last() {
            ir.add_branch(prev, b).unwrap();
        }
        blocks.push(b);
    }
    Emitter::new(&mut ir, blocks[0])
        .unwrap()
        .set_reg(Reg(7), Value::U32(99))
        .unwrap();
    let value = Emitter::new(&mut ir, *blocks.last().unwrap())
        .unwrap()
        .get_reg(Reg(7))
        .unwrap();
    let p = program(&ir, blocks);
    rewrite_ssa(&mut ir, &p).unwrap();
    assert_eq!(ir.resolve(value).unwrap(), Value::U32(99));
    ir.verify(&p).unwrap();
}
