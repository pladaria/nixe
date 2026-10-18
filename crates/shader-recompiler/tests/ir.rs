use nixe_shader_recompiler::{arena::InstId, diagnostics::FailureCategory, ir::*, types::Stage};

fn id(value: Value) -> InstId {
    let Value::Inst(id) = value else {
        panic!("Expected instruction")
    };
    id
}

#[test]
fn identities_preserve_users_and_invalidation_releases_operands() {
    let mut ir = Ir::new().unwrap();
    let block = ir.new_block().unwrap();
    let mut emit = Emitter::new(&mut ir, block).unwrap();
    let a = emit.get_reg(Reg(0)).unwrap();
    let b = emit.get_reg(Reg(1)).unwrap();
    let sum = emit.i_add(a, b).unwrap();
    let result = emit.i_mul(sum, Value::U32(5)).unwrap();
    assert_eq!(ir.inst(id(a)).unwrap().use_count(), 1);
    ir.replace_uses_with(id(sum), b).unwrap();
    assert_eq!(ir.inst(id(a)).unwrap().use_count(), 0);
    assert_eq!(ir.inst(id(b)).unwrap().use_count(), 1);
    assert_eq!(ir.resolve(sum).unwrap(), b);
    assert_eq!(ir.inst(id(result)).unwrap().arg(0).unwrap(), sum);
    ir.invalidate(id(result)).unwrap();
    assert_eq!(ir.inst(id(sum)).unwrap().use_count(), 0);
    ir.invalidate(id(sum)).unwrap();
    assert_eq!(ir.inst(id(b)).unwrap().use_count(), 0);
}

#[test]
fn pseudo_result_keeps_primary_alive_and_releases_association() {
    let mut ir = Ir::new().unwrap();
    let block = ir.new_block().unwrap();
    let mut emit = Emitter::new(&mut ir, block).unwrap();
    let sum = emit.i_add(Value::U32(u32::MAX), Value::U32(1)).unwrap();
    let carry = emit.get_carry_from_op(sum).unwrap();
    let only_consumer = emit.condition_ref(carry).unwrap();
    assert_eq!(ir.inst(id(sum)).unwrap().use_count(), 1);
    assert_eq!(
        ir.associated_pseudo(id(sum), Opcode::GetCarryFromOp)
            .unwrap(),
        Some(id(carry))
    );
    ir.invalidate(id(only_consumer)).unwrap();
    ir.invalidate(id(carry)).unwrap();
    assert_eq!(ir.inst(id(sum)).unwrap().use_count(), 0);
    assert_eq!(
        ir.associated_pseudo(id(sum), Opcode::GetCarryFromOp)
            .unwrap(),
        None
    );
    assert!(ir.inst(id(sum)).unwrap().has_associated_pseudo_operation());
    let mut emit = Emitter::new(&mut ir, block).unwrap();
    let zero = emit.get_zero_from_op(sum).unwrap();
    assert!(
        emit.ir
            .associated_pseudo(id(sum), Opcode::GetSparseFromOp)
            .is_err()
    );
    emit.ir.invalidate(id(zero)).unwrap();
    let sparse = emit.get_sparse_from_op(sum).unwrap();
    assert_eq!(
        emit.ir
            .associated_pseudo(id(sum), Opcode::GetSparseFromOp)
            .unwrap(),
        Some(id(sparse))
    );
}

#[test]
fn phi_order_and_graph_traversal_handle_back_edges() {
    let mut ir = Ir::new().unwrap();
    let entry = ir.new_block().unwrap();
    let body = ir.new_block().unwrap();
    let exit = ir.new_block().unwrap();
    ir.add_branch(entry, body).unwrap();
    ir.add_branch(body, body).unwrap();
    ir.add_branch(body, exit).unwrap();
    assert!(ir.add_branch(entry, body).is_err());
    assert_eq!(
        ir.post_order(&SyntaxNode::Block(entry)).unwrap(),
        [exit, body, entry]
    );
    assert_eq!(ir.block(body).unwrap().predecessors(), [entry, body]);
    ir.block_mut(entry).unwrap().order = 0;
    ir.block_mut(body).unwrap().order = 1;
    let phi = ir.new_phi(body, Type::U32).unwrap();
    ir.add_phi_operand(phi, body, Value::Inst(phi)).unwrap();
    ir.add_phi_operand(phi, entry, Value::U32(42)).unwrap();
    ir.order_phi_args(phi).unwrap();
    assert_eq!(ir.inst(phi).unwrap().phi_block(0).unwrap(), entry);
    assert_eq!(ir.inst(phi).unwrap().arg(0).unwrap(), Value::U32(42));
    assert_eq!(ir.value_type(Value::Inst(phi)).unwrap(), Type::U32);
    assert!(ir.all_args_immediate(phi).is_err());
    let mut emit = Emitter::new(&mut ir, body).unwrap();
    let reg = emit.get_reg(Reg(4)).unwrap();
    let sum = emit.i_add(Value::Inst(phi), reg).unwrap();
    let found = ir
        .breadth_first_search(sum, |id, inst| {
            (inst.opcode() == Opcode::GetRegister).then_some(id)
        })
        .unwrap();
    assert_eq!(found, Some(id(reg)));
    let found = ir
        .breadth_first_search(Value::Inst(phi), |_, _| None::<()>)
        .unwrap();
    assert_eq!(found, None);
    ir.detach(body, id(reg)).unwrap();
    ir.attach(entry, 0, id(reg)).unwrap();
    assert_eq!(ir.block(body).unwrap().instructions(), [phi, id(sum)]);
    assert_eq!(ir.block(entry).unwrap().instructions(), [id(reg)]);
    let mut program = Program::new(Stage::Compute);
    program.blocks = vec![entry, body, exit];
    ir.verify(&program).unwrap();
    ir.replace_uses_with(phi, Value::U32(42)).unwrap();
    assert_eq!(ir.resolve(Value::Inst(phi)).unwrap(), Value::U32(42));
    assert!(ir.dump_program(&program).unwrap().contains("IAdd32"));
}

#[test]
fn clearing_operands_releases_slots_hidden_by_an_opcode_change() {
    let mut ir = Ir::new().unwrap();
    let block = ir.new_block().unwrap();
    let mut emit = Emitter::new(&mut ir, block).unwrap();
    let input = emit.get_reg(Reg(0)).unwrap();
    let sum = emit.i_add(input, input).unwrap();
    assert_eq!(ir.inst(id(input)).unwrap().use_count(), 2);
    ir.replace_opcode(id(sum), Opcode::INeg32).unwrap();
    ir.invalidate(id(sum)).unwrap();
    assert_eq!(ir.inst(id(input)).unwrap().use_count(), 0);
}

#[test]
fn debug_verifier_rejects_invalid_list_and_syntax_relationships() {
    let mut ir = Ir::new().unwrap();
    let block = ir.new_block().unwrap();
    let instruction = ir.append(block, Opcode::Void, &[], 0).unwrap();
    let mut program = Program::new(Stage::Fragment);
    program.blocks = vec![block];
    ir.verify(&program).unwrap();
    ir.attach(block, 0, instruction).unwrap();
    assert!(ir.verify(&program).is_err());
    ir.detach(block, instruction).unwrap();
    program.syntax.push(SyntaxNode::If {
        condition: Value::U32(0),
        body: block,
        merge: block,
    });
    assert!(ir.verify(&program).is_err());
}

#[test]
fn growth_and_reset_do_not_recycle_node_identities() {
    let mut ir = Ir::new().unwrap();
    let first = ir.new_block().unwrap();
    let reg = ir
        .append(first, Opcode::GetRegister, &[Value::Reg(Reg(1))], 0)
        .unwrap();
    for _ in 0..10_000 {
        ir.append(first, Opcode::Void, &[], 0).unwrap();
    }
    assert_eq!(ir.inst(reg).unwrap().arg(0).unwrap(), Value::Reg(Reg(1)));
    ir.reset().unwrap();
    let second = ir.new_block().unwrap();
    let newer = ir
        .append(second, Opcode::GetRegister, &[Value::Reg(Reg(2))], 0)
        .unwrap();
    assert_ne!(first, second);
    assert_ne!(reg, newer);
    assert!(ir.inst(reg).is_err());
    assert!(ir.block(first).is_err());
    assert!(std::mem::size_of::<Instruction>() <= 128);
    assert_eq!(std::mem::size_of::<Value>(), 16);
    assert_eq!(std::mem::size_of::<Option<InstId>>(), 8);
}

#[test]
fn emitter_enforces_types_and_expands_integer_equality_and_texture_pairs() {
    let mut ir = Ir::new().unwrap();
    let block = ir.new_block().unwrap();
    let mut emit = Emitter::new(&mut ir, block).unwrap();
    assert!(emit.i_add(Value::U32(1), Value::U64(2)).is_err());
    assert!(
        emit.composite_construct(&[Value::f32(1.0), Value::U32(1)])
            .is_err()
    );
    let equal = emit
        .i_equal(Value::U64(0x1_0000_0000), Value::U64(0))
        .unwrap();
    assert_eq!(
        emit.ir.inst(id(equal)).unwrap().opcode(),
        Opcode::LogicalAnd
    );
    assert_eq!(emit.ir.block(block).unwrap().instructions().len(), 9);
    let pair = emit
        .make_lod_clamp_pair(Value::f32(2.0), Value::f32(3.0))
        .unwrap();
    assert_eq!(emit.ir.value_type(pair).unwrap(), Type::F32x2);
    assert!(emit.composite_extract(pair, 2).is_err());
    assert!(
        emit.select(Value::U32(1), Value::U32(2), Value::U32(3))
            .is_err()
    );
    let attrs = emit.get_attribute(Attribute::Generic0X).unwrap();
    assert_eq!(
        emit.ir.inst(id(attrs)).unwrap().arg(1).unwrap(),
        Value::U32(0)
    );
    assert!(
        emit.fp_convert(64, emit.ir.resolve(pair).unwrap(), FpControl::default())
            .is_err()
    );
    assert!(emit.u_convert(16, Value::U64(3)).is_err());
    assert!(emit.s_convert(32, Value::U32(3)).is_err());
    let shared = emit.load_shared(8, true, Value::U32(4)).unwrap();
    assert_eq!(
        emit.ir.inst(id(shared)).unwrap().opcode(),
        Opcode::LoadSharedS8
    );
    assert_eq!(emit.get_pred(Pred::PT, true).unwrap(), Value::U1(false));
    let count = emit.ir.block(block).unwrap().instructions().len();
    emit.set_pred(Pred::PT, Value::U1(false)).unwrap();
    assert_eq!(emit.ir.block(block).unwrap().instructions().len(), count);
    assert_eq!(Reg::RZ.offset(1).unwrap(), Reg::RZ);
    assert!(Reg::RZ.is_aligned(2));
    assert!(Reg(254).offset(1).is_err());
    assert!(Reg(0).offset(-1).is_err());
}

#[test]
fn flags_and_immediates_preserve_exact_bits() {
    assert_ne!(Value::f32(-0.0), Value::f32(0.0));
    let nan = Value::F32(0x7fc0_1234);
    assert_eq!(nan, nan);
    let flags = FpControl {
        no_contraction: true,
        rounding: FpRounding::RM,
        fmz_mode: FmzMode::FMZ,
    };
    assert_eq!(FpControl::from_bits(flags.bits()).unwrap(), flags);
    let mut texture = TextureInstInfo(u32::MAX);
    texture.set_descriptor_index(17);
    texture.set_has_bias(false);
    assert_eq!(texture.descriptor_index(), 17);
    assert!(!texture.has_bias());
    assert!(texture.is_integer());
    assert!(texture.has_lod_clamp());
    assert_eq!(texture.gather_component(), 3);
}

#[test]
fn operation_vocabulary_and_effects_are_complete() {
    let types = [
        Type::Void,
        Type::Opaque,
        Type::Reg,
        Type::Pred,
        Type::Attribute,
        Type::Patch,
        Type::U1,
        Type::U8,
        Type::U16,
        Type::U32,
        Type::U64,
        Type::F16,
        Type::F32,
        Type::F64,
        Type::U32x2,
        Type::U32x3,
        Type::U32x4,
        Type::F16x2,
        Type::F16x3,
        Type::F16x4,
        Type::F32x2,
        Type::F32x3,
        Type::F32x4,
        Type::F64x2,
        Type::F64x3,
        Type::F64x4,
    ];
    let names = [
        "Void",
        "Opaque",
        "Reg",
        "Pred",
        "Attribute",
        "Patch",
        "U1",
        "U8",
        "U16",
        "U32",
        "U64",
        "F16",
        "F32",
        "F64",
        "U32x2",
        "U32x3",
        "U32x4",
        "F16x2",
        "F16x3",
        "F16x4",
        "F32x2",
        "F32x3",
        "F32x4",
        "F64x2",
        "F64x3",
        "F64x4",
    ];
    let lookup = |name| types[names.iter().position(|&n| n == name).unwrap()];
    let rows: Vec<_> = include_str!("data/ir-signatures.txt").lines().collect();
    assert_eq!(rows.len(), 534);
    assert_eq!(Opcode::ALL.len(), 534);
    for (opcode, row) in Opcode::ALL.iter().zip(rows) {
        let words: Vec<_> = row.split_whitespace().collect();
        assert_eq!(opcode.name(), words[0]);
        assert_eq!(opcode.signature().result, lookup(words[1]));
        assert_eq!(
            opcode.signature().arguments,
            words[2..]
                .iter()
                .take_while(|&&name| name != "Void")
                .map(|&name| lookup(name))
                .collect::<Vec<_>>()
        );
    }
    assert!(Opcode::StorageAtomicIAdd64.signature().side_effects);
    assert!(Opcode::ConditionRef.signature().side_effects);
    assert!(Opcode::BoundImageWrite.signature().side_effects);
    assert!(!Opcode::IAdd32.signature().side_effects);
    assert!(!Opcode::GetRegister.signature().side_effects);
}

fn eval_bool(ir: &Ir, value: Value, z: bool, s: bool, c: bool, o: bool) -> bool {
    match ir.resolve(value).unwrap() {
        Value::U1(value) => value,
        Value::Inst(id) => {
            let inst = ir.inst(id).unwrap();
            let arg = |index| eval_bool(ir, inst.arg(index).unwrap(), z, s, c, o);
            match inst.opcode() {
                Opcode::GetZFlag => z,
                Opcode::GetSFlag => s,
                Opcode::GetCFlag => c,
                Opcode::GetOFlag => o,
                Opcode::LogicalAnd => arg(0) && arg(1),
                Opcode::LogicalOr => arg(0) || arg(1),
                Opcode::LogicalXor => arg(0) ^ arg(1),
                Opcode::LogicalNot => !arg(0),
                _ => panic!("Not a boolean flow-test operation"),
            }
        }
        _ => panic!("Not a boolean"),
    }
}

#[test]
fn flow_tests_cover_all_flag_combinations_and_explicit_rejections() {
    for bits in 0..16 {
        let (z, s, c, o) = (bits & 1 != 0, bits & 2 != 0, bits & 4 != 0, bits & 8 != 0);
        let expected = [
            false,
            (s && !z) ^ o,
            !s && z,
            s ^ (z || o),
            (!s ^ o) && !z,
            !z,
            !(s ^ o),
            !s || !z,
            s && z,
            s ^ o,
            z,
            (s ^ o) || z,
            !s ^ (z || o),
            s || !z,
            (!s || z) ^ o,
            true,
            !o,
            !c,
            !s,
            z || !c,
            c && !z,
            s,
            c,
            o,
        ];
        let mut ir = Ir::new().unwrap();
        let block = ir.new_block().unwrap();
        for (index, expected) in expected.into_iter().enumerate() {
            let value = Emitter::new(&mut ir, block)
                .unwrap()
                .get_flow_test_result(FlowTest::try_from(index as u8).unwrap())
                .unwrap();
            assert_eq!(
                eval_bool(&ir, value, z, s, c, o),
                expected,
                "test {index}, flags {bits}"
            );
        }
        for index in [24, 25, 26, 27, 29] {
            let error = Emitter::new(&mut ir, block)
                .unwrap()
                .get_flow_test_result(FlowTest::try_from(index).unwrap())
                .unwrap_err();
            assert_eq!(error.category, FailureCategory::Unsupported);
        }
        for (test, expected) in [
            (FlowTest::FCSM_TR, false),
            (FlowTest::RLE, s || z),
            (FlowTest::RGT, !s && !z),
        ] {
            let value = Emitter::new(&mut ir, block)
                .unwrap()
                .get_flow_test_result(test)
                .unwrap();
            assert_eq!(eval_bool(&ir, value, z, s, c, o), expected);
        }
    }
}
