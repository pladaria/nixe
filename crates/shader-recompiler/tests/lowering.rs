use nixe_shader_recompiler::{
    HostTranslateInfo,
    arena::{BlockId, InstId},
    ir::{Emitter, FpControl, Ir, Opcode as O, Program, SyntaxNode, Type, Value},
    passes::{conditional_barriers, early_passes, lower_fp16, lower_fp64, lower_int64},
    types::Stage,
};
use std::collections::HashMap;
#[derive(Clone, Copy, Debug, PartialEq)]
enum Evaluated {
    Scalar(u64),
    Pair(u32, u32),
}
fn scalar(value: Evaluated) -> u64 {
    match value {
        Evaluated::Scalar(x) => x,
        _ => panic!(),
    }
}
fn evaluate(ir: &Ir, block: BlockId, result: Value) -> Evaluated {
    let mut values = HashMap::<InstId, Evaluated>::new();
    let mut carries = HashMap::new();
    fn get(ir: &Ir, values: &HashMap<InstId, Evaluated>, value: Value) -> Evaluated {
        match ir.resolve(value).unwrap() {
            Value::Inst(id) => values[&id],
            Value::U32(x) | Value::F32(x) => Evaluated::Scalar(x as u64),
            Value::U64(x) => Evaluated::Scalar(x),
            Value::U1(x) => Evaluated::Scalar(u64::from(x)),
            _ => panic!(),
        }
    }
    for &id in ir.block(block).unwrap().instructions() {
        let inst = ir.inst(id).unwrap();
        let mut args = [Evaluated::Scalar(0); 4];
        for (i, arg) in args.iter_mut().enumerate().take(inst.num_args()) {
            *arg = get(ir, &values, inst.arg(i).unwrap());
        }
        let a = || scalar(args[0]) as u32;
        let b = || scalar(args[1]) as u32;
        let value = match inst.opcode() {
            O::Identity => args[0],
            O::CompositeConstructU32x2 => Evaluated::Pair(a(), b()),
            O::CompositeExtractU32x2 => {
                let Evaluated::Pair(x, y) = args[0] else {
                    panic!()
                };
                Evaluated::Scalar(u64::from(if b() == 0 { x } else { y }))
            }
            O::GetCarryFromOp => {
                let Value::Inst(producer) = inst.arg(0).unwrap() else {
                    panic!()
                };
                Evaluated::Scalar(u64::from(carries[&producer]))
            }
            O::IAdd32 => {
                let (value, carry) = a().overflowing_add(b());
                carries.insert(id, carry);
                Evaluated::Scalar(value as u64)
            }
            O::ISub32 => Evaluated::Scalar(a().wrapping_sub(b()) as u64),
            O::BitwiseNot32 => Evaluated::Scalar(!a() as u64),
            O::BitwiseAnd32 => Evaluated::Scalar((a() & b()) as u64),
            O::BitwiseOr32 => Evaluated::Scalar((a() | b()) as u64),
            O::ShiftLeftLogical32 => Evaluated::Scalar(a().wrapping_shl(b()) as u64),
            O::ShiftRightLogical32 => Evaluated::Scalar(a().wrapping_shr(b()) as u64),
            O::ShiftRightArithmetic32 => {
                Evaluated::Scalar((a() as i32).wrapping_shr(b()) as u32 as u64)
            }
            O::IEqual => Evaluated::Scalar(u64::from(a() == b())),
            O::INotEqual => Evaluated::Scalar(u64::from(a() != b())),
            O::UGreaterThan => Evaluated::Scalar(u64::from(a() > b())),
            O::SGreaterThanEqual => Evaluated::Scalar(u64::from(a() as i32 >= b() as i32)),
            O::SelectU32 => {
                if a() != 0 {
                    args[1]
                } else {
                    args[2]
                }
            }
            O::BitCastF32U32 | O::BitCastU32F32 => args[0],
            O::BitFieldUExtract => {
                let offset = b();
                let width = scalar(args[2]) as u32;
                let x = if offset >= 32 || width == 0 {
                    0
                } else {
                    a() >> offset & (u32::MAX >> (32 - width.min(32)))
                };
                Evaluated::Scalar(x as u64)
            }
            O::BitFieldInsert => {
                let offset = scalar(args[2]) as u32;
                let width = scalar(args[3]) as u32;
                let mask = if offset >= 32 || width == 0 {
                    0
                } else {
                    (u32::MAX >> (32 - width.min(32))).wrapping_shl(offset)
                };
                Evaluated::Scalar(((a() & !mask) | (b().wrapping_shl(offset) & mask)) as u64)
            }
            other => panic!("Unexpected evaluator operation {other:?}"),
        };
        values.insert(id, value);
    }
    get(ir, &values, result)
}
fn setup() -> (Ir, BlockId, Program) {
    let mut ir = Ir::new().unwrap();
    let b = ir.new_block().unwrap();
    let mut p = Program::new(Stage::Compute);
    p.blocks.push(b);
    p.post_order.push(b);
    (ir, b, p)
}
fn lowered_integer(op: O, args: &[Value]) -> Evaluated {
    let (mut ir, b, p) = setup();
    let value = Emitter::new(&mut ir, b).unwrap().emit(op, args, 0).unwrap();
    lower_int64(&mut ir, &p).unwrap();
    ir.verify(&p).unwrap();
    evaluate(&ir, b, value)
}
fn pair(x: u64) -> Evaluated {
    Evaluated::Pair(x as u32, (x >> 32) as u32)
}
#[test]
fn split_integer_arithmetic_preserves_carry_borrow_sign_and_shift_boundaries() {
    let inputs = [
        0,
        1,
        0xffff_ffff,
        0x1_0000_0000,
        0x8000_0000_0000_0000,
        0xabcdef01_23456789,
        u64::MAX,
    ];
    for a in inputs {
        for b in inputs {
            assert_eq!(
                lowered_integer(O::IAdd64, &[Value::U64(a), Value::U64(b)]),
                pair(a.wrapping_add(b))
            );
            assert_eq!(
                lowered_integer(O::ISub64, &[Value::U64(a), Value::U64(b)]),
                pair(a.wrapping_sub(b))
            );
        }
        assert_eq!(
            lowered_integer(O::INeg64, &[Value::U64(a)]),
            pair(a.wrapping_neg())
        );
        assert_eq!(
            lowered_integer(O::IAbs64, &[Value::U64(a)]),
            pair((a as i64).wrapping_abs() as u64)
        );
        for shift in [0, 1, 16, 31, 32, 33, 63] {
            assert_eq!(
                lowered_integer(O::ShiftLeftLogical64, &[Value::U64(a), Value::U32(shift)]),
                pair(a << shift)
            );
            assert_eq!(
                lowered_integer(O::ShiftRightLogical64, &[Value::U64(a), Value::U32(shift)]),
                pair(a >> shift)
            );
            assert_eq!(
                lowered_integer(
                    O::ShiftRightArithmetic64,
                    &[Value::U64(a), Value::U32(shift)]
                ),
                pair(((a as i64) >> shift) as u64)
            );
        }
    }
    assert_eq!(
        lowered_integer(
            O::SelectU64,
            &[Value::U1(false), Value::U64(1), Value::U64(9)]
        ),
        pair(9)
    );
    assert_eq!(lowered_integer(O::UndefU64, &[]), pair(0));
}
#[test]
fn integer_conversion_pairs_and_atomic_tables_do_not_blanket_rewrite_other_widths() {
    let (mut ir, b, p) = setup();
    let mut e = Emitter::new(&mut ir, b).unwrap();
    let value = e.u_convert(64, Value::U32(0xdeadbeef)).unwrap();
    let low = e.u_convert(32, value).unwrap();
    lower_int64(&mut ir, &p).unwrap();
    ir.verify(&p).unwrap();
    assert_eq!(evaluate(&ir, b, low), Evaluated::Scalar(0xdeadbeef));
    let (mut ir, b, p) = setup();
    let mut e = Emitter::new(&mut ir, b).unwrap();
    let vector = e
        .composite_construct(&[Value::U32(1), Value::U32(2)])
        .unwrap();
    let wide = e.pack_uint2x32(vector).unwrap();
    let atom = e.global_atomic_i_add(wide, wide).unwrap();
    let Value::Inst(atom) = atom else { panic!() };
    lower_int64(&mut ir, &p).unwrap();
    ir.verify(&p).unwrap();
    assert_eq!(ir.inst(atom).unwrap().opcode(), O::GlobalAtomicIAdd32x2);
    let (mut ir, b, p) = setup();
    let mut e = Emitter::new(&mut ir, b).unwrap();
    let product = e
        .fp_add(Value::f64(3.0), Value::f64(4.0), FpControl::default())
        .unwrap();
    lower_int64(&mut ir, &p).unwrap();
    let Value::Inst(product) = product else {
        panic!()
    };
    assert_eq!(ir.inst(product).unwrap().opcode(), O::FPAdd64);
}
#[test]
fn fp64_pack_and_unpack_preserve_literal_exponent_transform() {
    for raw in [
        0u64,
        1,
        0x3ff0_0000_0000_0000,
        0x7ff0_0000_0000_0000,
        0x7ff8_1234_5678_9abc,
        0x8000_0000_0000_0000,
    ] {
        let (mut ir, b, p) = setup();
        let mut e = Emitter::new(&mut ir, b).unwrap();
        let pair = e
            .composite_construct(&[Value::U32(raw as u32), Value::U32((raw >> 32) as u32)])
            .unwrap();
        let result = e.pack_double2x32(pair).unwrap();
        lower_fp64(&mut ir, &p).unwrap();
        ir.verify(&p).unwrap();
        let exp = ((raw >> 52) & 2047) as u32;
        let exp = if exp == 0 {
            0
        } else if exp == 2047 {
            255
        } else {
            exp + 896
        };
        let expected =
            ((raw >> 32) as u32 & 0x80000000) | (exp << 23) | ((raw >> 29) as u32 & 0x7fffff);
        assert_eq!(evaluate(&ir, b, result), Evaluated::Scalar(expected as u64));
    }
    for raw in [
        0u32,
        1,
        1.0f32.to_bits(),
        f32::INFINITY.to_bits(),
        f32::NAN.to_bits(),
        0x80000000,
    ] {
        let (mut ir, b, p) = setup();
        // The producer is narrowed before the unpack visitor consumes it.
        let source = ir
            .append(b, O::ConvertF64F32, &[Value::F32(raw)], 0)
            .unwrap();
        let result = Emitter::new(&mut ir, b)
            .unwrap()
            .unpack_double2x32(Value::Inst(source))
            .unwrap();
        lower_fp64(&mut ir, &p).unwrap();
        ir.verify(&p).unwrap();
        let exp = (raw >> 23) & 255;
        let exp = if exp == 0 {
            0
        } else if exp == 255 {
            2047
        } else {
            exp.wrapping_sub(896)
        };
        let lo = (raw & 7) << 29;
        let hi = (raw & 0x80000000) | (exp << 20) | ((raw & 0x7fffff) >> 3);
        assert_eq!(evaluate(&ir, b, result), Evaluated::Pair(lo, hi));
    }
}
#[test]
fn half_lowering_keeps_controls_and_only_rewrites_its_explicit_operations() {
    let (mut ir, b, p) = setup();
    let mut e = Emitter::new(&mut ir, b).unwrap();
    let pair = e.unpack_float2x16(Value::U32(0x3c003c00)).unwrap();
    let x = e.composite_extract(pair, 0).unwrap();
    let control = FpControl {
        no_contraction: true,
        ..Default::default()
    };
    let sum = e.fp_add(x, x, control).unwrap();
    let result = e.fp_convert(32, sum, FpControl::default()).unwrap();
    lower_fp16(&mut ir, &p).unwrap();
    ir.verify(&p).unwrap();
    assert_eq!(ir.value_type(result).unwrap(), Type::F32);
    let Value::Inst(sum) = sum else { panic!() };
    assert_eq!(ir.inst(sum).unwrap().opcode(), O::FPAdd32);
    assert_eq!(ir.inst(sum).unwrap().flags, control.bits());
    let (mut ir, b, p) = setup();
    let zero = Value::Inst(ir.append(b, O::BitCastF16U16, &[Value::U16(0)], 0).unwrap());
    let id = ir
        .append(b, O::SelectF16, &[Value::U1(true), zero, zero], 0)
        .unwrap();
    lower_fp16(&mut ir, &p).unwrap();
    assert_eq!(ir.inst(id).unwrap().opcode(), O::SelectF16);
}
#[test]
fn conversion_families_and_pseudo_guard_are_explicit() {
    for op in [
        O::ConvertF16S64,
        O::ConvertF32S64,
        O::ConvertF64S64,
        O::ConvertF16U64,
        O::ConvertF32U64,
        O::ConvertF64U64,
    ] {
        let (mut ir, b, p) = setup();
        let id = ir.append(b, op, &[Value::U64(123)], 0).unwrap();
        lower_int64(&mut ir, &p).unwrap();
        ir.verify(&p).unwrap();
        assert_eq!(ir.inst(id).unwrap().opcode(), O::Identity);
    }
    for (op, value) in [
        (O::ConvertS64F16, Value::U16(0x3c00)),
        (O::ConvertS64F32, Value::F32(1f32.to_bits())),
        (O::ConvertS64F64, Value::F64(1f64.to_bits())),
        (O::ConvertU64F16, Value::U16(0x3c00)),
        (O::ConvertU64F32, Value::F32(1f32.to_bits())),
        (O::ConvertU64F64, Value::F64(1f64.to_bits())),
    ] {
        let (mut ir, b, p) = setup();
        let value = if let Value::U16(_) = value {
            Value::Inst(ir.append(b, O::BitCastF16U16, &[value], 0).unwrap())
        } else {
            value
        };
        let id = ir.append(b, op, &[value], 0).unwrap();
        lower_int64(&mut ir, &p).unwrap();
        ir.verify(&p).unwrap();
        assert_eq!(ir.value_type(Value::Inst(id)).unwrap(), Type::U32x2);
    }
    let (mut ir, b, p) = setup();
    let mut e = Emitter::new(&mut ir, b).unwrap();
    let sum = e.i_add(Value::U64(1), Value::U64(2)).unwrap();
    e.get_carry_from_op(sum).unwrap();
    assert!(lower_int64(&mut ir, &p).is_err());
}
#[test]
fn conditional_return_keeps_barrier_suppression_after_merge_and_host_gate() {
    let (mut ir, a, mut p) = setup();
    let b = ir.new_block().unwrap();
    let c = ir.new_block().unwrap();
    p.blocks.extend([b, c]);
    let first = ir.append(a, O::Barrier, &[], 0).unwrap();
    let inner = ir.append(b, O::Barrier, &[], 0).unwrap();
    let later = ir.append(c, O::Barrier, &[], 0).unwrap();
    p.syntax = vec![
        SyntaxNode::Block(a),
        SyntaxNode::If {
            condition: Value::U1(true),
            body: b,
            merge: c,
        },
        SyntaxNode::Block(b),
        SyntaxNode::Return,
        SyntaxNode::EndIf { merge: c },
        SyntaxNode::Block(c),
    ];
    let host = HostTranslateInfo {
        support_float64: true,
        support_float16: true,
        support_int64: true,
        support_conditional_barrier: true,
        ..Default::default()
    };
    early_passes(&mut ir, &p, &host).unwrap();
    assert_eq!(ir.inst(inner).unwrap().opcode(), O::Barrier);
    conditional_barriers(&mut ir, &p).unwrap();
    assert_eq!(ir.inst(first).unwrap().opcode(), O::Barrier);
    for id in [inner, later] {
        assert_eq!(ir.inst(id).unwrap().opcode(), O::Identity);
        assert_eq!(ir.inst(id).unwrap().arg(0).unwrap(), Value::None);
    }
    ir.verify(&p).unwrap();
}
