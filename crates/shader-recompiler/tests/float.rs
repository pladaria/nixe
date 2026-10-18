use nixe_shader_recompiler::{
    arena::{BlockId, InstId},
    diagnostics::FailureCategory,
    ir::{FmzMode, FpControl, FpRounding, Ir, Opcode as I, Program, Value},
    maxwell::{
        bits::insert,
        decode::{DecodedInstruction, Opcode},
        instruction::Instruction,
        location::Location,
        translate::translate_instruction,
    },
    types::{ProgramOffset, Stage},
};
use std::collections::HashMap;

fn word(op: Opcode) -> u64 {
    op.encoding().value | 7 << 16 | 2 << 8 | 4 << 20
}
fn compile(opcode: Opcode, word: u64) -> nixe_shader_recompiler::Result<(Ir, BlockId)> {
    let mut ir = Ir::new()?;
    let block = ir.new_block()?;
    translate_instruction(
        &nixe_shader_recompiler::environment::EnvironmentMetadata::default(),
        &mut ir,
        block,
        DecodedInstruction {
            location: Location::new(ProgramOffset(8))?,
            instruction: Instruction(word),
            opcode,
        },
    )?;
    let mut program = Program::new(Stage::Compute);
    program.blocks.push(block);
    ir.verify(&program)?;
    Ok((ir, block))
}
fn count(ir: &Ir, block: BlockId, opcode: I) -> usize {
    ir.block(block)
        .unwrap()
        .instructions()
        .iter()
        .filter(|&&id| ir.inst(id).unwrap().opcode() == opcode)
        .count()
}
fn operations(ir: &Ir, block: BlockId, opcode: I) -> Vec<InstId> {
    ir.block(block)
        .unwrap()
        .instructions()
        .iter()
        .copied()
        .filter(|&id| ir.inst(id).unwrap().opcode() == opcode)
        .collect()
}

fn execute(op: Opcode, word: u64, inputs: &[(usize, u32)]) -> [u32; 256] {
    let (ir, block) = compile(op, word).unwrap();
    let mut registers = [0u32; 256];
    for &(index, value) in inputs {
        registers[index] = value;
    }
    let mut values = HashMap::new();
    for &id in ir.block(block).unwrap().instructions() {
        let inst = ir.inst(id).unwrap();
        let mut args = [0u64; 5];
        for (index, arg) in args.iter_mut().enumerate().take(inst.num_args()) {
            *arg = match inst.arg(index).unwrap() {
                Value::Inst(id) => values[&id],
                Value::F32(x) | Value::U32(x) => x as u64,
                Value::F64(x) | Value::U64(x) => x,
                Value::U1(x) => x as u64,
                Value::Reg(r) => r.0 as u64,
                Value::Pred(p) => p.index() as u64,
                _ => panic!(),
            };
        }
        let [x, y, z, _, _] = args;
        let (fx, fy, fz) = (
            f32::from_bits(x as u32),
            f32::from_bits(y as u32),
            f32::from_bits(z as u32),
        );
        let float = |x: f32| x.to_bits() as u64;
        let result = match inst.opcode() {
            I::GetRegister => registers[x as usize] as u64,
            I::SetRegister => {
                registers[x as usize] = y as u32;
                0
            }
            I::GetPred => {
                assert_eq!(x, 7);
                1
            }
            I::BitCastF32U32 | I::BitCastU32F32 => x,
            I::FPAdd32 => float(fx + fy),
            I::FPMul32 => float(fx * fy),
            I::FPFma32 => float(fx.mul_add(fy, fz)),
            I::FPAbs32 => x & 0x7fffffff,
            I::FPNeg32 => x ^ 0x80000000,
            I::FPSaturate32 => float(if fx.is_nan() { 0.0 } else { fx.clamp(0.0, 1.0) }),
            I::FPClamp32 => float(fx.max(fy).min(fz)),
            I::FPRoundEven32 => float(fx.round_ties_even()),
            I::FPFloor32 => float(fx.floor()),
            I::FPCeil32 => float(fx.ceil()),
            I::FPTrunc32 => float(fx.trunc()),
            I::FPIsNan32 => fx.is_nan() as u64,
            I::FPOrdEqual32 => (fx == fy) as u64,
            I::FPUnordEqual32 => (fx.is_nan() || fy.is_nan() || fx == fy) as u64,
            I::FPOrdNotEqual32 => (!fx.is_nan() && !fy.is_nan() && fx != fy) as u64,
            I::FPUnordNotEqual32 => (fx != fy) as u64,
            I::FPOrdLessThan32 => (fx < fy) as u64,
            I::FPUnordLessThan32 => (fx.is_nan() || fy.is_nan() || fx < fy) as u64,
            I::FPOrdGreaterThan32 => (fx > fy) as u64,
            I::FPUnordGreaterThan32 => (fx.is_nan() || fy.is_nan() || fx > fy) as u64,
            I::FPOrdLessThanEqual32 => (fx <= fy) as u64,
            I::FPUnordLessThanEqual32 => (fx.is_nan() || fy.is_nan() || fx <= fy) as u64,
            I::FPOrdGreaterThanEqual32 => (fx >= fy) as u64,
            I::FPUnordGreaterThanEqual32 => (fx.is_nan() || fy.is_nan() || fx >= fy) as u64,
            I::LogicalAnd => x & y,
            I::LogicalOr => x | y,
            I::LogicalNot => (x == 0) as u64,
            I::SelectF32 | I::SelectU32 | I::SelectU64 | I::SelectU1 => {
                if x != 0 {
                    y
                } else {
                    z
                }
            }
            I::ConvertS32F32 => fx as i32 as u32 as u64,
            I::ConvertU32F32 => fx as u32 as u64,
            I::ConvertS64F32 => fx as i64 as u64,
            I::ConvertU64F32 => fx as u64,
            I::ConvertF32S32 => float(x as u32 as i32 as f32),
            I::ConvertF32U32 => float(x as u32 as f32),
            I::ConvertF32S64 => float(x as i64 as f32),
            I::ConvertF32U64 => float(x as f32),
            I::BitFieldSExtract => {
                let n = z as u32;
                let bits = (x as u32).wrapping_shr(y as u32);
                ((bits << (32 - n)) as i32 >> (32 - n)) as u32 as u64
            }
            I::BitFieldUExtract => (x >> y) & (u32::MAX >> (32 - z)) as u64,
            I::IEqual => (x == y) as u64,
            I::IAdd32 => (x as u32).wrapping_add(y as u32) as u64,
            I::BitwiseXor32 => x ^ y,
            I::ShiftRightArithmetic32 => ((x as u32 as i32) >> (y as u32)) as u32 as u64,
            I::IAbs32 => (x as u32 as i32).wrapping_abs() as u32 as u64,
            I::CompositeConstructU32x2 => x | (y << 32),
            I::CompositeExtractU32x2 => (x >> (32 * y)) & 0xffffffff,
            I::PackUint2x32 | I::UnpackUint2x32 => x,
            _ => panic!("Unexpected test operation {:?}", inst.opcode()),
        };
        values.insert(id, result);
    }
    registers
}

#[test]
fn abs_precedes_negation_and_rounding_and_denormal_flags_survive() {
    for round in 0..4 {
        for flush in [false, true] {
            let encoded =
                insert::<39, 2>(insert::<44, 1>(word(Opcode::FADD_reg), flush as u64), round)
                    | 1 << 46
                    | 1 << 48
                    | 1 << 49;
            let (ir, block) = compile(Opcode::FADD_reg, encoded).unwrap();
            let add = operations(&ir, block, I::FPAdd32)[0];
            assert_eq!(
                FpControl::from_bits(ir.inst(add).unwrap().flags).unwrap(),
                FpControl {
                    no_contraction: true,
                    rounding: [
                        FpRounding::RN,
                        FpRounding::RM,
                        FpRounding::RP,
                        FpRounding::RZ
                    ][round as usize],
                    fmz_mode: if flush { FmzMode::FTZ } else { FmzMode::None }
                }
            );
            let result = execute(
                Opcode::FADD_reg,
                encoded,
                &[(2, (-2.0f32).to_bits()), (4, (-0.5f32).to_bits())],
            );
            assert_eq!(f32::from_bits(result[0]), -1.5);
        }
    }
}

#[test]
fn fmz_zero_product_overrides_nan_and_infinity_and_sat_keeps_its_own_path() {
    for other in [f32::NAN, f32::INFINITY] {
        for a in [0.0f32, -0.0] {
            let encoded = insert::<44, 2>(word(Opcode::FMUL_reg), 2);
            assert_eq!(
                execute(
                    Opcode::FMUL_reg,
                    encoded,
                    &[(2, a.to_bits()), (4, other.to_bits())]
                )[0],
                0
            );
            let encoded = insert::<53, 2>(word(Opcode::FFMA_reg) | 6 << 39, 2);
            assert_eq!(
                execute(
                    Opcode::FFMA_reg,
                    encoded,
                    &[
                        (2, a.to_bits()),
                        (4, other.to_bits()),
                        (6, 3.0f32.to_bits())
                    ]
                )[0],
                3.0f32.to_bits()
            );
        }
    }
    let encoded = insert::<44, 2>(word(Opcode::FMUL_reg), 2) | 1 << 50;
    let (ir, block) = compile(Opcode::FMUL_reg, encoded).unwrap();
    assert_eq!(count(&ir, block, I::SelectF32), 0);
    assert_eq!(count(&ir, block, I::FPSaturate32), 1);
    assert_eq!(
        execute(
            Opcode::FMUL_reg,
            encoded,
            &[(2, 2.0f32.to_bits()), (4, 3.0f32.to_bits())]
        )[0],
        1.0f32.to_bits()
    );
}

#[test]
fn full_immediate_fma_reads_prior_destination_and_compare_immediate_keeps_float_bits() {
    let encoded = insert::<20, 32>(word(Opcode::FFMA32I), 2.5f32.to_bits() as u64);
    assert_eq!(
        execute(
            Opcode::FFMA32I,
            encoded,
            &[(0, 7.0f32.to_bits()), (2, 4.0f32.to_bits())]
        )[0],
        17.0f32.to_bits()
    );
    let encoded = insert::<48, 4>(
        insert::<20, 19>(
            word(Opcode::FCMP_imm) | 6 << 39,
            (1.5f32.to_bits() >> 12) as u64,
        ),
        0,
    );
    assert_eq!(execute(Opcode::FCMP_imm, encoded, &[])[0], 1.5f32.to_bits());
}

#[test]
fn comparisons_keep_nan_ordering_for_every_condition() {
    for op in 0..16 {
        let encoded = insert::<39, 3>(insert::<48, 4>(word(Opcode::FSET_reg), op), 7);
        let result = execute(
            Opcode::FSET_reg,
            encoded,
            &[(2, f32::NAN.to_bits()), (4, 1.0f32.to_bits())],
        );
        assert_eq!(
            result[0],
            if op >= 8 { u32::MAX } else { 0 },
            "condition {op}"
        );
    }
}

#[test]
fn conversion_rounding_nans_and_small_signed_minima_are_explicit() {
    for (mode, expected) in [(0, 2), (1, 2), (2, 3), (3, 2)] {
        let encoded =
            insert::<8, 4>(insert::<39, 2>(word(Opcode::F2I_reg), mode), 2 | 2 << 2) | 1 << 12;
        assert_eq!(
            execute(Opcode::F2I_reg, encoded, &[(4, 2.5f32.to_bits())])[0],
            expected
        );
        assert_eq!(
            execute(Opcode::F2I_reg, encoded, &[(4, f32::NAN.to_bits())])[0],
            0
        );
    }
    let encoded = insert::<8, 4>(word(Opcode::F2I_reg), 3 | 2 << 2) | 1 << 12;
    let result = execute(Opcode::F2I_reg, encoded, &[(4, f32::NAN.to_bits())]);
    assert_eq!([result[0], result[1]], [0, 0x80000000]);
    for format in [0, 1] {
        for absolute in [false, true] {
            let encoded = insert::<8, 4>(word(Opcode::I2F_reg), 2 | format << 2)
                | 1 << 13
                | 1 << 45
                | u64::from(absolute) << 49;
            let bits = 8 << format;
            let min = 1u32 << (bits - 1);
            let result = execute(Opcode::I2F_reg, encoded, &[(4, min)]);
            assert_eq!(
                f32::from_bits(result[0]),
                if absolute { min as f32 } else { -(min as f32) }
            );
        }
    }
}

#[test]
fn double_routes_enforce_pairs_but_conversion_sources_allow_odd_registers() {
    let (ir, block) = compile(Opcode::DADD_reg, word(Opcode::DADD_reg)).unwrap();
    assert_eq!(count(&ir, block, I::FPAdd64), 1);
    assert_eq!(count(&ir, block, I::SetRegister), 2);
    assert!(compile(Opcode::DADD_reg, insert::<20, 8>(word(Opcode::DADD_reg), 3)).is_err());
    let encoded = insert::<8, 4>(insert::<20, 8>(word(Opcode::F2I_reg), 3), 2 | 3 << 2);
    assert!(compile(Opcode::F2I_reg, encoded).is_ok());
    let encoded = insert::<8, 4>(insert::<20, 8>(word(Opcode::I2F_reg), 3), 2 | 3 << 2);
    assert!(compile(Opcode::I2F_reg, encoded).is_ok());
    let encoded = insert::<8, 4>(insert::<20, 14>(word(Opcode::F2I_cbuf), 2), 2 | 3 << 2);
    let (ir, block) = compile(Opcode::F2I_cbuf, encoded).unwrap();
    let read = operations(&ir, block, I::GetCbufU32)[0];
    assert_eq!(ir.inst(read).unwrap().arg(1).unwrap(), Value::U32(12));
    assert_eq!(count(&ir, block, I::GetCbufU32), 1);
}

#[test]
fn half_swizzles_promote_mixed_inputs_and_merge_reads_only_the_needed_destination() {
    for merge in 0..4 {
        let encoded = insert::<49, 2>(insert::<28, 2>(word(Opcode::HADD2_reg), 1), merge);
        let (ir, block) = compile(Opcode::HADD2_reg, encoded).unwrap();
        assert_eq!(count(&ir, block, I::FPAdd32), 2);
        assert_eq!(
            count(&ir, block, I::ConvertF32F16),
            if merge == 1 { 3 } else { 2 }
        );
        assert_eq!(count(&ir, block, I::ConvertF16F32), 2);
        let reads = operations(&ir, block, I::GetRegister);
        let reads_destination = reads.iter().any(|&id| {
            ir.inst(id).unwrap().arg(0).unwrap() == Value::Reg(nixe_shader_recompiler::ir::Reg(0))
        });
        assert_eq!(reads_destination, merge >= 2);
    }
    let encoded = insert::<47, 2>(insert::<28, 2>(word(Opcode::HADD2_reg), 1), 1);
    assert_eq!(
        compile(Opcode::HADD2_reg, encoded).err().unwrap().category,
        FailureCategory::UnsupportedReferenceCase
    );
    for merge in 1..4 {
        assert!(compile(Opcode::HADD2_reg, insert::<49, 2>(encoded, merge)).is_ok());
    }
    let encoded = insert::<39, 2>(word(Opcode::HMUL2_reg), 2);
    assert_eq!(
        compile(Opcode::HMUL2_reg, encoded).err().unwrap().category,
        FailureCategory::InvalidArgument
    );
    assert!(compile(Opcode::HMUL2_reg, encoded | 1 << 32).is_ok());
}

#[test]
fn half_immediate_signs_predicate_pairing_and_inherited_reduction_are_visible() {
    let encoded =
        insert::<20, 9>(insert::<30, 9>(word(Opcode::HADD2_imm), 0x10), 0x20) | 1 << 29 | 1 << 56;
    let (ir, block) = compile(Opcode::HADD2_imm, encoded).unwrap();
    assert!(
        operations(&ir, block, I::UnpackFloat2x16)
            .iter()
            .any(|&id| ir.inst(id).unwrap().arg(0).unwrap() == Value::U32(0x84008800))
    );
    for combine in [false, true] {
        let encoded = insert::<35, 4>(word(Opcode::HSETP2_reg), 2) | u64::from(combine) << 49;
        let (ir, block) = compile(Opcode::HSETP2_reg, encoded).unwrap();
        assert_eq!(count(&ir, block, I::FPOrdEqual16), 2);
        assert_eq!(count(&ir, block, I::SetPred), 2);
        assert_eq!(count(&ir, block, I::LogicalNot), usize::from(combine));
    }
    for mode in [0, 1] {
        let encoded = insert::<39, 1>(word(Opcode::RRO_reg), mode) | 1 << 45 | 1 << 49;
        assert_eq!(
            execute(Opcode::RRO_reg, encoded, &[(4, (-2.0f32).to_bits())])[0],
            (-2.0f32).to_bits()
        );
    }
}

#[test]
fn f2f_same_width_rounding_and_precision_changes_preserve_nan_and_control_paths() {
    for (round, expected) in [
        (0, 2.5f32),
        (3, 2.5),
        (8, 2.0),
        (9, 2.0),
        (10, 3.0),
        (11, 2.0),
    ] {
        let encoded = insert::<8, 4>(insert::<39, 4>(word(Opcode::F2F_reg), round), 2 | 2 << 2);
        assert_eq!(
            execute(Opcode::F2F_reg, encoded, &[(4, 2.5f32.to_bits())])[0],
            expected.to_bits()
        );
        assert!(
            f32::from_bits(execute(Opcode::F2F_reg, encoded, &[(4, f32::NAN.to_bits())])[0])
                .is_nan()
        );
    }
    let encoded =
        insert::<8, 4>(insert::<39, 2>(word(Opcode::F2F_reg), 2), 3 | 2 << 2) | 1 << 44 | 1 << 50;
    let (ir, block) = compile(Opcode::F2F_reg, encoded).unwrap();
    let conversion = operations(&ir, block, I::ConvertF64F32)[0];
    let control = FpControl::from_bits(ir.inst(conversion).unwrap().flags).unwrap();
    assert_eq!(control.rounding, FpRounding::RP);
    assert_eq!(control.fmz_mode, FmzMode::None);
    assert_eq!(count(&ir, block, I::FPSaturate64), 0);
    assert!(
        compile(
            Opcode::F2F_reg,
            insert::<8, 4>(word(Opcode::F2F_reg), 1 | 3 << 2)
        )
        .is_err()
    );
    assert!(
        compile(
            Opcode::F2F_imm,
            insert::<8, 4>(word(Opcode::F2F_imm) | 1 << 56, 1 | 1 << 2)
        )
        .is_err()
    );
}

#[test]
fn numerical_guards_reject_unsupported_modes_without_substituting_operations() {
    for (opcode, encoded) in [
        (Opcode::FMUL_reg, insert::<41, 3>(word(Opcode::FMUL_reg), 1)),
        (Opcode::FMUL_reg, insert::<44, 2>(word(Opcode::FMUL_reg), 3)),
        (Opcode::MUFU, insert::<20, 4>(word(Opcode::MUFU), 6)),
        (Opcode::DADD_reg, word(Opcode::DADD_reg) | 1 << 47),
        (Opcode::FSWZADD, word(Opcode::FSWZADD) | 1 << 47),
        (
            Opcode::F2I_cbuf,
            insert::<8, 4>(insert::<20, 14>(word(Opcode::F2I_cbuf), 1), 2 | 3 << 2),
        ),
    ] {
        assert_eq!(
            compile(opcode, encoded).err().unwrap().category,
            FailureCategory::Unsupported
        );
    }
}
