use nixe_shader_recompiler::{
    arena::BlockId,
    environment::EnvironmentMetadata,
    ir::{
        Instruction as IrInstruction, Ir, Opcode as I, Program, Reg, TextureInstInfo, Type, Value,
    },
    maxwell::{
        bits::insert,
        decode::{DecodedInstruction, Opcode},
        instruction::Instruction,
        location::Location,
        translate::translate_instruction,
    },
    types::{ImageFormat, ProgramOffset, Stage, TextureType},
};
fn word(op: Opcode) -> u64 {
    op.encoding().value | 7 << 16 | 2 << 8 | 8 << 20
}
fn compile(opcode: Opcode, word: u64) -> nixe_shader_recompiler::Result<(Ir, BlockId)> {
    let mut ir = Ir::new()?;
    let block = ir.new_block()?;
    translate_instruction(
        &EnvironmentMetadata::default(),
        &mut ir,
        block,
        DecodedInstruction {
            location: Location::new(ProgramOffset(8))?,
            instruction: Instruction(word),
            opcode,
        },
    )?;
    let mut p = Program::new(Stage::Compute);
    p.blocks.push(block);
    ir.verify(&p)?;
    Ok((ir, block))
}
fn ops(ir: &Ir, block: BlockId, opcode: I) -> Vec<&IrInstruction> {
    ir.block(block)
        .unwrap()
        .instructions()
        .iter()
        .filter_map(|&id| {
            let i = ir.inst(id).unwrap();
            (i.opcode() == opcode).then_some(i)
        })
        .collect()
}
fn reads(ir: &Ir, block: BlockId) -> Vec<u8> {
    ops(ir, block, I::GetRegister)
        .iter()
        .map(|i| match i.arg(0).unwrap() {
            Value::Reg(r) => r.0,
            _ => panic!(),
        })
        .collect()
}
fn standard(op: Opcode, kind: u64) -> u64 {
    insert::<51, 3>(insert::<31, 4>(insert::<28, 3>(word(op), kind), 15), 7)
}
#[test]
fn texture_coordinate_shapes_layers_handles_and_compacted_masks() {
    for (kind, ty, shape, order) in [
        (0, TextureType::Color1D, Type::F32, vec![2]),
        (1, TextureType::ColorArray1D, Type::F32x2, vec![3, 2]),
        (2, TextureType::Color2D, Type::F32x2, vec![2, 3]),
        (3, TextureType::ColorArray2D, Type::F32x3, vec![3, 4, 2]),
        (4, TextureType::Color3D, Type::F32x3, vec![2, 3, 4]),
        (6, TextureType::ColorCube, Type::F32x3, vec![2, 3, 4]),
        (
            7,
            TextureType::ColorArrayCube,
            Type::F32x4,
            vec![3, 4, 5, 2],
        ),
    ] {
        for bindless in [false, true] {
            let op = if bindless { Opcode::TEX_b } else { Opcode::TEX };
            let mut w = standard(op, kind);
            w = if bindless {
                insert::<37, 3>(w, 1)
            } else {
                insert::<55, 3>(w, 1)
            };
            w = insert::<31, 4>(w, 10);
            let (ir, b) = compile(op, w).unwrap();
            let sample = ops(
                &ir,
                b,
                if bindless {
                    I::BindlessImageSampleExplicitLod
                } else {
                    I::BoundImageSampleExplicitLod
                },
            )[0];
            assert_eq!(TextureInstInfo(sample.flags).texture_type(), ty);
            assert_eq!(ir.value_type(sample.arg(1).unwrap()).unwrap(), shape);
            assert_eq!(sample.arg(2).unwrap(), Value::F32(0));
            let mut expected = order.clone();
            if bindless {
                expected.push(8);
            }
            assert_eq!(reads(&ir, b), expected);
            assert_eq!(ops(&ir, b, I::SetRegister).len(), 2);
        }
    }
    assert!(compile(Opcode::TEX, standard(Opcode::TEX, 5)).is_err());
}
#[test]
fn lod_depth_offsets_and_sparse_result_keep_separate_dependencies() {
    let w = insert::<51, 3>(insert::<37, 3>(standard(Opcode::TEX_b, 3), 2), 1) | 1 << 36 | 1 << 50;
    let (ir, b) = compile(Opcode::TEX_b, w).unwrap();
    assert_eq!(reads(&ir, b), [3, 4, 2, 8, 9, 10, 11]);
    let sample = ops(&ir, b, I::BindlessImageSampleDrefImplicitLod)[0];
    let info = TextureInstInfo(sample.flags);
    assert!(info.is_depth() && info.has_bias());
    assert_eq!(ops(&ir, b, I::BitFieldSExtract).len(), 2);
    assert_eq!(ops(&ir, b, I::GetSparseFromOp).len(), 1);
    let writes = ops(&ir, b, I::SetRegister);
    assert_eq!(writes.len(), 4);
    assert_eq!(ops(&ir, b, I::LogicalNot).len(), 1);
    assert!(compile(Opcode::TEX_b, w | 1 << 40).is_err());
}
#[test]
fn fetch_and_gradient_keep_integer_coords_derivatives_and_metadata_order() {
    let w = standard(Opcode::TLD_b, 3) | 1 << 55 | 1 << 35 | 1 << 50;
    let (ir, b) = compile(Opcode::TLD_b, w).unwrap();
    assert_eq!(reads(&ir, b), [3, 4, 2, 8, 9, 10, 11]);
    let sample = ops(&ir, b, I::BindlessImageFetch)[0];
    assert_eq!(ir.value_type(sample.arg(1).unwrap()).unwrap(), Type::U32x3);
    let w = standard(Opcode::TXD_b, 3) | 1 << 35;
    let (ir, b) = compile(Opcode::TXD_b, w).unwrap();
    assert_eq!(reads(&ir, b), [2, 3, 4, 5, 8, 9, 10, 11, 5]);
    let sample = ops(&ir, b, I::BindlessImageGradient)[0];
    assert_eq!(TextureInstInfo(sample.flags).num_derivatives(), 2);
    assert_eq!(ir.value_type(sample.arg(2).unwrap()).unwrap(), Type::F32x4);
    assert_eq!(
        ops(&ir, b, I::BitFieldSExtract)[0].arg(1).unwrap(),
        Value::U32(16)
    );
    assert!(compile(Opcode::TXD_b, w | 1 << 50).is_err());
}
#[test]
fn gathers_keep_ptp_pairs_and_queries_omit_array_layer_and_rescale_lod() {
    let w = insert::<36, 2>(standard(Opcode::TLD4_b, 2), 2) | 1 << 50;
    let (ir, b) = compile(Opcode::TLD4_b, w).unwrap();
    assert_eq!(reads(&ir, b), [2, 3, 8, 9, 10, 11]);
    let sample = ops(&ir, b, I::BindlessImageGatherDref)[0];
    assert_eq!(ir.value_type(sample.arg(2).unwrap()).unwrap(), Type::U32x4);
    assert_eq!(ir.value_type(sample.arg(3).unwrap()).unwrap(), Type::U32x4);
    assert_eq!(ops(&ir, b, I::BitFieldSExtract).len(), 8);
    let w = insert::<31, 4>(standard(Opcode::TMML, 3), 3);
    let (ir, b) = compile(Opcode::TMML, w).unwrap();
    assert_eq!(reads(&ir, b), [3, 4]);
    assert_eq!(ops(&ir, b, I::ShiftLeftLogical32).len(), 2);
    let w = insert::<22, 3>(insert::<31, 4>(word(Opcode::TXQ_b), 9), 1);
    let (ir, b) = compile(Opcode::TXQ_b, w).unwrap();
    assert_eq!(reads(&ir, b), [2, 3]);
    assert_eq!(
        ops(&ir, b, I::BindlessImageQueryDimensions)[0]
            .arg(2)
            .unwrap(),
        Value::U1(false)
    );
}
#[test]
fn all_swizzled_sample_and_fetch_encodings_route_and_pack() {
    for op in [Opcode::TEXS, Opcode::TLDS] {
        for encoding in 0..16 {
            let valid = if op == Opcode::TEXS {
                encoding <= 13
            } else {
                matches!(encoding, 0 | 1 | 2 | 4 | 5 | 6 | 7 | 8 | 12)
            };
            for half in [false, true] {
                let w = insert::<59, 1>(
                    insert::<53, 4>(insert::<50, 3>(insert::<28, 8>(word(op), 16), 4), encoding),
                    u64::from(!half),
                );
                let result = compile(op, w);
                if !valid {
                    assert!(result.is_err());
                    continue;
                }
                let (ir, b) = result.unwrap();
                assert_eq!(ops(&ir, b, I::SetRegister).len(), if half { 2 } else { 4 });
                assert_eq!(ops(&ir, b, I::PackHalf2x16).len(), if half { 2 } else { 0 });
                let destinations: Vec<_> = ops(&ir, b, I::SetRegister)
                    .iter()
                    .map(|i| i.arg(0).unwrap())
                    .collect();
                assert_eq!(
                    destinations,
                    if half {
                        vec![Value::Reg(Reg(0)), Value::Reg(Reg(16))]
                    } else {
                        [0, 1, 16, 17].map(|r| Value::Reg(Reg(r))).to_vec()
                    }
                );
            }
        }
    }
}
#[test]
fn swizzle_tables_pack_single_components_and_pad_odd_counts_with_zero() {
    for second in [255, 16] {
        for swizzle in 0..8 {
            let w = insert::<53, 4>(
                insert::<50, 3>(insert::<28, 8>(word(Opcode::TEXS), second), swizzle),
                2,
            );
            let result = compile(Opcode::TEXS, w & !(1 << 59));
            if second != 255 && swizzle > 4 {
                assert!(result.is_err());
                continue;
            }
            let (ir, b) = result.unwrap();
            let count = if second == 255 {
                if swizzle < 4 { 1 } else { 2 }
            } else if swizzle == 4 {
                4
            } else {
                3
            };
            let packs = ops(&ir, b, I::PackHalf2x16);
            assert_eq!(packs.len(), if count > 2 { 2 } else { 1 });
            if count % 2 == 1 {
                let Value::Inst(pair) = packs.last().unwrap().arg(0).unwrap() else {
                    panic!()
                };
                assert_eq!(ir.inst(pair).unwrap().arg(1).unwrap(), Value::F32(0));
            }
        }
    }
    for depth in [false, true] {
        for offset in [false, true] {
            for half in [false, true] {
                let w = insert::<55, 1>(
                    insert::<51, 1>(
                        insert::<50, 1>(insert::<28, 8>(word(Opcode::TLD4S), 16), u64::from(depth)),
                        u64::from(offset),
                    ),
                    u64::from(half),
                );
                let (ir, b) = compile(Opcode::TLD4S, w).unwrap();
                assert_eq!(ops(&ir, b, I::SetRegister).len(), if half { 2 } else { 4 });
            }
        }
    }
}
#[test]
fn surfaces_keep_storage_formats_dimensions_masks_and_reduction_effects() {
    for kind in 0..6 {
        for size in 0..7 {
            for store in [false, true] {
                let op = if store { Opcode::SUST } else { Opcode::SULD };
                let w = insert::<33, 3>(insert::<20, 4>(word(op), size), kind) | 1 << 51 | 1 << 52;
                let (ir, b) = compile(op, w).unwrap();
                let access = ops(
                    &ir,
                    b,
                    if store {
                        I::BoundImageWrite
                    } else {
                        I::BoundImageRead
                    },
                )[0];
                let info = TextureInstInfo(access.flags);
                assert_eq!(
                    info.image_format(),
                    ImageFormat::try_from(size as u32 + 1).unwrap()
                );
                if !store {
                    assert_eq!(
                        ops(&ir, b, I::SetRegister).len(),
                        if size < 5 {
                            1
                        } else if size == 5 {
                            2
                        } else {
                            4
                        }
                    );
                }
            }
        }
    }
    let w = insert::<20, 4>(word(Opcode::SULD), 10) | 1 << 51;
    let (ir, b) = compile(Opcode::SULD, w).unwrap();
    let extracts = ops(&ir, b, I::CompositeExtractU32x4);
    assert_eq!(extracts[0].arg(1).unwrap(), Value::U32(1));
    assert_eq!(extracts[1].arg(1).unwrap(), Value::U32(3));
    assert!(compile(Opcode::SUST, w).is_err());
    for op in 0..8 {
        let w = insert::<24, 3>(insert::<20, 3>(word(Opcode::SURED), 0), op) | 1 << 51;
        let (ir, b) = compile(Opcode::SURED, w).unwrap();
        assert!(ops(&ir, b, I::SetRegister).is_empty());
        assert!(
            ir.block(b)
                .unwrap()
                .instructions()
                .iter()
                .any(|&id| ir.inst(id).unwrap().may_have_side_effects())
        );
    }
}
