use nixe_shader_recompiler::{
    arena::{BlockId, InstId},
    environment::{EnvironmentMetadata, Observations, SnapshotEnvironment},
    ir::{Emitter, Ir, Opcode as O, Program, TextureInstInfo, Value},
    passes::{constant_propagation, remove_identities, verify_program},
    types::{Stage, TextureType},
};
fn program() -> (Ir, Program, BlockId) {
    let mut ir = Ir::new().unwrap();
    let block = ir.new_block().unwrap();
    let mut p = Program::new(Stage::Compute);
    p.blocks.push(block);
    p.post_order.push(block);
    (ir, p, block)
}
fn id(value: Value) -> InstId {
    let Value::Inst(id) = value else { panic!() };
    id
}
fn fold(ir: &mut Ir, p: &Program) {
    let mut env = SnapshotEnvironment::new(EnvironmentMetadata::default(), Observations::default());
    constant_propagation(&mut env, ir, p).unwrap();
}
fn bfe(e: &mut Emitter<'_>, v: Value, offset: u32, width: u32) -> Value {
    e.emit(
        O::BitFieldUExtract,
        &[v, Value::U32(offset), Value::U32(width)],
        0,
    )
    .unwrap()
}
fn shift(e: &mut Emitter<'_>, v: Value) -> Value {
    e.emit(O::ShiftLeftLogical32, &[v, Value::U32(16)], 0)
        .unwrap()
}
#[test]
fn xmad_multiply_requires_matching_factors_and_halfword_widths() {
    for width in [16, 15] {
        let (mut ir, p, block) = program();
        let mut e = Emitter::new(&mut ir, block).unwrap();
        let a = e.emit(O::UndefU32, &[], 0).unwrap();
        let b = e.emit(O::UndefU32, &[], 0).unwrap();
        let hi = bfe(&mut e, a, 16, width);
        let lo = bfe(&mut e, a, 0, 16);
        let hi = e.i_mul(hi, b).unwrap();
        let hi = shift(&mut e, hi);
        let lo = e.i_mul(lo, b).unwrap();
        let result = e.i_add(hi, lo).unwrap();
        fold(&mut ir, &p);
        let inst = ir.inst(id(ir.resolve(result).unwrap())).unwrap();
        assert_eq!(
            inst.opcode(),
            if width == 16 { O::IMul32 } else { O::IAdd32 }
        );
        if width == 16 {
            assert_eq!(inst.arg(0).unwrap(), a);
            assert_eq!(inst.arg(1).unwrap(), b);
        }
        remove_identities(&mut ir, &p).unwrap();
        verify_program(&ir, &p).unwrap();
    }
}
#[test]
fn xmad_multiply_add_recovers_full_inputs_and_rejects_wrong_insert_width() {
    for width in [16, 15] {
        let (mut ir, p, block) = program();
        let mut e = Emitter::new(&mut ir, block).unwrap();
        let a = e.emit(O::UndefU32, &[], 0).unwrap();
        let b = e.emit(O::UndefU32, &[], 0).unwrap();
        let c = e.emit(O::UndefU32, &[], 0).unwrap();
        let a_hi = bfe(&mut e, a, 16, 16);
        let a_lo = bfe(&mut e, a, 0, 16);
        let b_hi = bfe(&mut e, b, 16, 16);
        let b_lo = bfe(&mut e, b, 0, 16);
        let partial = e.i_mul(b_lo, a_hi).unwrap();
        let packed = e
            .emit(
                O::BitFieldInsert,
                &[partial, a_lo, Value::U32(16), Value::U32(width)],
                0,
            )
            .unwrap();
        let packed_hi = bfe(&mut e, packed, 16, 16);
        let hi = e.i_mul(b_hi, packed_hi).unwrap();
        let hi = shift(&mut e, hi);
        let packed_shift = shift(&mut e, packed);
        let low = e.i_mul(b_lo, a_lo).unwrap();
        let low = e.i_add(low, c).unwrap();
        let low = e.i_add(packed_shift, low).unwrap();
        let result = e.i_add(hi, low).unwrap();
        fold(&mut ir, &p);
        let result_id = id(ir.resolve(result).unwrap());
        let inst = ir.inst(result_id).unwrap();
        assert_eq!(inst.opcode(), O::IAdd32);
        if width == 16 {
            assert_eq!(inst.arg(1).unwrap(), c);
            let mul = ir.inst(id(inst.arg(0).unwrap())).unwrap();
            assert_eq!(mul.opcode(), O::IMul32);
            assert_eq!((mul.arg(0).unwrap(), mul.arg(1).unwrap()), (a, b));
        } else {
            assert_eq!(result_id, id(result));
        }
        remove_identities(&mut ir, &p).unwrap();
        verify_program(&ir, &p).unwrap();
    }
}
#[test]
fn derivative_folding_checks_shuffle_clamp_segment_and_swizzle() {
    for (mask, lane, clamp, segment, expected) in [
        (0x99, 1, 3, 28, O::DPdxFine),
        (0xa5, 2, 3, 28, O::DPdyFine),
        (0x99, 1, 7, 28, O::FSwizzleAdd),
        (0x99, 1, 3, 27, O::FSwizzleAdd),
        (0x98, 1, 3, 28, O::FSwizzleAdd),
    ] {
        let (mut ir, p, block) = program();
        let mut e = Emitter::new(&mut ir, block).unwrap();
        let input = e.emit(O::UndefU32, &[], 0).unwrap();
        let shuffle = e
            .emit(
                O::ShuffleButterfly,
                &[
                    input,
                    Value::U32(lane),
                    Value::U32(clamp),
                    Value::U32(segment),
                ],
                0,
            )
            .unwrap();
        let a = e.emit(O::BitCastF32U32, &[shuffle], 0).unwrap();
        let b = e.emit(O::BitCastF32U32, &[input], 0).unwrap();
        let result = e
            .emit(O::FSwizzleAdd, &[a, b, Value::U32(mask)], 0)
            .unwrap();
        fold(&mut ir, &p);
        assert_eq!(
            ir.inst(id(ir.resolve(result).unwrap())).unwrap().opcode(),
            expected
        );
        remove_identities(&mut ir, &p).unwrap();
        verify_program(&ir, &p).unwrap();
    }
}
#[test]
fn ndv_gradient_recovers_three_coordinate_rows_and_rejects_repeated_zero_masks() {
    for outer_mask in [0xf0, 0xcc] {
        let (mut ir, p, block) = program();
        let mut e = Emitter::new(&mut ir, block).unwrap();
        let mut rows = Vec::new();
        let mut inputs = Vec::new();
        for _ in 0..3 {
            let mut parts = Vec::new();
            for _ in 0..3 {
                let input = e.emit(O::UndefU32, &[], 0).unwrap();
                inputs.push(input);
                let broadcast = e
                    .emit(O::QuadBroadcast, &[input, Value::U32(3)], 0)
                    .unwrap();
                parts.push(e.emit(O::BitCastF32U32, &[broadcast], 0).unwrap());
            }
            let inner = e
                .emit(O::FSwizzleAdd, &[parts[1], parts[2], Value::U32(0xcc)], 0)
                .unwrap();
            rows.push(
                e.emit(
                    O::FSwizzleAdd,
                    &[inner, parts[0], Value::U32(outer_mask)],
                    0,
                )
                .unwrap(),
            );
        }
        let coords = e.emit(O::CompositeConstructF32x3, &rows, 0).unwrap();
        let mut flags = TextureInstInfo::default();
        flags.set_texture_type(TextureType::Color3D);
        flags.set_ndv_is_active(true);
        let result = e
            .emit(
                O::BoundImageSampleImplicitLod,
                &[Value::U32(8), coords, Value::None, Value::None],
                flags.0,
            )
            .unwrap();
        fold(&mut ir, &p);
        let inst = ir.inst(id(ir.resolve(result).unwrap())).unwrap();
        if outer_mask == 0xf0 {
            assert_eq!(inst.opcode(), O::BoundImageGradient);
            assert_eq!(TextureInstInfo(inst.flags).num_derivatives(), 3);
            let coordinates = ir.inst(id(inst.arg(1).unwrap())).unwrap();
            for i in 0..3 {
                let cast = ir.inst(id(coordinates.arg(i).unwrap())).unwrap();
                assert_eq!(cast.opcode(), O::BitCastF32U32);
                assert_eq!(cast.arg(0).unwrap(), inputs[3 * i]);
            }
        } else {
            assert_eq!(inst.opcode(), O::BoundImageSampleImplicitLod);
        }
        remove_identities(&mut ir, &p).unwrap();
        verify_program(&ir, &p).unwrap();
    }
}
#[test]
fn arithmetic_folding_keeps_carry_producers_and_simplifies_composites() {
    let (mut ir, p, block) = program();
    let mut e = Emitter::new(&mut ir, block).unwrap();
    let add = e.i_add(Value::U32(u32::MAX), Value::U32(1)).unwrap();
    let carry = e.emit(O::GetCarryFromOp, &[add], 0).unwrap();
    let wrapped = e.i_add(Value::U32(u32::MAX), Value::U32(1)).unwrap();
    let composite = e
        .emit(
            O::CompositeConstructU32x3,
            &[Value::U32(7), Value::U32(8), Value::U32(9)],
            0,
        )
        .unwrap();
    let inserted = e
        .emit(
            O::CompositeInsertU32x3,
            &[composite, Value::U32(20), Value::U32(1)],
            0,
        )
        .unwrap();
    let first = e
        .emit(O::CompositeExtractU32x3, &[inserted, Value::U32(0)], 0)
        .unwrap();
    let second = e
        .emit(O::CompositeExtractU32x3, &[inserted, Value::U32(1)], 0)
        .unwrap();
    fold(&mut ir, &p);
    assert_eq!(ir.inst(id(add)).unwrap().opcode(), O::IAdd32);
    assert_eq!(ir.inst(id(carry)).unwrap().arg(0).unwrap(), add);
    assert_eq!(ir.resolve(wrapped).unwrap(), Value::U32(0));
    assert_eq!(ir.resolve(first).unwrap(), Value::U32(7));
    assert_eq!(ir.resolve(second).unwrap(), Value::U32(20));
}

#[test]
fn driver_constants_and_hle_replacements_record_consumed_dependencies() {
    use nixe_shader_recompiler::{
        environment::{CbufWordKey, RecordingEnvironment},
        ir::Attribute,
        types::{ByteOffset, CbufIndex, ReplaceConstant},
    };
    for hle in [false, true] {
        let (mut ir, p, block) = program();
        let mut e = Emitter::new(&mut ir, block).unwrap();
        let value = e
            .get_cbuf(Value::U32(1), Value::U32(20), 32, false)
            .unwrap();
        let key = CbufWordKey {
            index: CbufIndex(1),
            offset: ByteOffset(20),
        };
        let mut obs = Observations::default();
        obs.cbuf_words.insert(key, 1234);
        obs.replacements
            .insert(key, Some(ReplaceConstant::BaseVertex));
        let mut source = SnapshotEnvironment::new(
            EnvironmentMetadata {
                has_hle_macro_state: hle,
                is_proprietary_driver: !hle,
                ..Default::default()
            },
            obs,
        );
        let mut recorder = RecordingEnvironment::new(&mut source);
        constant_propagation(&mut recorder, &mut ir, &p).unwrap();
        if hle {
            let inst = ir.inst(id(ir.resolve(value).unwrap())).unwrap();
            assert_eq!(inst.opcode(), O::GetAttributeU32);
            assert_eq!(
                inst.arg(0).unwrap(),
                Value::Attribute(Attribute::BaseVertex)
            );
            assert_eq!(recorder.observations().replacements.len(), 1);
            assert!(recorder.observations().cbuf_words.is_empty());
        } else {
            assert_eq!(ir.resolve(value).unwrap(), Value::U32(1234));
            assert_eq!(recorder.observations().cbuf_words.len(), 1);
            assert!(recorder.observations().replacements.is_empty());
        }
    }
}
