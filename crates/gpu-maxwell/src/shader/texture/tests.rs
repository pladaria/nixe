use super::super::error::MaxwellShaderTranslationError;
use super::super::link::finalize_shader_ir;
use super::super::test_support::{
    translated_fixture, translated_fixture_with_register_count, validate_wgsl,
};
use super::*;
use crate::MaxwellShaderStage;
use nixe_gpu::{
    ShaderOperation, ShaderRegister, ShaderResourceKind, ShaderTextureSampleOutput,
    lower_shader_ir_to_wgsl,
};
use std::collections::BTreeMap;

#[test]
fn txq_mip_count_keeps_the_established_image_type_and_binding() {
    use nixe_gpu::{
        ShaderInstruction, ShaderInterfaceElement, ShaderIoLocation, ShaderIr, ShaderPredicate,
        ShaderResourceAccess, ShaderSourceLocation, ShaderStage, VerifiedShaderIr,
    };
    for image_kind in [
        ShaderResourceKind::SampledImage,
        ShaderResourceKind::SampledImage2DArray,
        ShaderResourceKind::SampledDepthImage,
        ShaderResourceKind::SampledDepthImage2DArray,
    ] {
        let query =
            decode_texture_query(MaxwellShaderStage::Pixel, 8, 0xdf48016400470a0a, 32).unwrap();
        assert_eq!(query.destination.index(), 10);
        assert_eq!(query.descriptor, 22);
        let mut ops = vec![ShaderOperation::QueryTextureLevels {
            destination: query.destination,
            image_binding: 32,
        }];
        ops.push(ShaderOperation::StoreOutput {
            sources: vec![ShaderRegister::new(10)].into(),
            location: ShaderIoLocation::Color(0),
            first_component: 0,
            scalar_type: ShaderScalarType::Unsigned32,
        });
        ops.push(ShaderOperation::Exit);
        let ir = VerifiedShaderIr::verify(ShaderIr::new(
            ShaderStage::Fragment,
            vec![],
            vec![
                ShaderInterfaceElement::new(
                    ShaderIoLocation::Color(0),
                    0,
                    ShaderScalarType::Unsigned32,
                    None,
                )
                .unwrap(),
            ],
            vec![ShaderResourceAccess::new(32, image_kind, true, false).unwrap()],
            ops.into_iter()
                .enumerate()
                .map(|(i, op)| {
                    ShaderInstruction::new(
                        ShaderSourceLocation::new(i as u32 * 8),
                        ShaderPredicate::Always,
                        op,
                    )
                })
                .collect(),
        ))
        .unwrap();
        let wgsl = nixe_gpu::lower_shader_ir_to_wgsl(&ir).unwrap();
        super::super::test_support::validate_wgsl(&wgsl);
        assert!(wgsl.source().contains("textureNumLevels(sampled_image_32)"));
    }
}

#[test]
fn txq_translation_and_stage_linking_preserve_texture_identity() {
    let mut header = [0; 20];
    header[0] = 0x0002_5462;
    header[18] = 15;
    let sample = (0xc03a_0083_2ff7_0404 & !(0x1fff << 36)) | (22 << 36);
    let translated = translated_fixture_with_register_count(
        MaxwellShaderStage::Pixel,
        header,
        &[0, 0xdf48016400470a0a, sample, 0xe30000000007000f],
        32,
    );
    let remapped = finalize_shader_ir(
        translated.ir().clone(),
        MaxwellShaderStage::Pixel,
        &BTreeMap::from([(32, 2), (33, 3)]),
        &[],
    )
    .unwrap();
    assert!(remapped.ir().instructions().iter().any(|i| matches!(i.operation(),
        ShaderOperation::QueryTextureLevels { destination, image_binding: 2 } if destination.index() == 10)));
    let wgsl = lower_shader_ir_to_wgsl(&remapped).unwrap();
    validate_wgsl(&wgsl);
    assert!(wgsl.source().contains("textureNumLevels(sampled_image_2)"));
}

#[test]
fn texs_cube_lod_preserves_packed_directions_mips_outputs_and_query_identity() {
    let word = 0xd9a2018270470608;
    let mut bindings = BTreeMap::new();
    let ops = decode_texture_access_simplified(
        MaxwellShaderStage::Pixel,
        8,
        word,
        48,
        &mut bindings,
        &mut 48,
    )
    .unwrap();
    assert!(
        matches!(&ops[0], ShaderOperation::SampleTextureCube { coordinates, lod: Some(lod), outputs, image_binding: 32, sampler_binding: 33 }
        if coordinates.map(|r| r.index()) == [6,7,4] && lod.index() == 5 && outputs.iter().map(|o| o.destination().index()).collect::<Vec<_>>() == [8,9,39])
    );
    assert_eq!(
        bindings[&24].image_kind,
        ShaderResourceKind::SampledImageCube
    );
    let mut header = [0; 20];
    header[0] = 0x00025462;
    header[18] = 15;
    let query = (0xdf48016400470a0a & !(0x1fff << 36)) | (24 << 36);
    let translated = translated_fixture_with_register_count(
        MaxwellShaderStage::Pixel,
        header,
        &[0, query, word, 0xe30000000007000f],
        48,
    );
    let remapped = finalize_shader_ir(
        translated.ir().clone(),
        MaxwellShaderStage::Pixel,
        &BTreeMap::from([(32, 2), (33, 3)]),
        &[],
    )
    .unwrap();
    let module = lower_shader_ir_to_wgsl(&remapped).unwrap();
    validate_wgsl(&module);
    assert!(module.source().contains("texture_cube<f32>"));
    assert!(
        module
            .source()
            .contains("textureSampleLevel(sampled_image_2")
    );
    assert!(
        module
            .source()
            .contains("textureNumLevels(sampled_image_2)")
    );
    for invalid in [word ^ (1 << 8), word ^ (1 << 20)] {
        assert!(
            decode_texture_access_simplified(
                MaxwellShaderStage::Pixel,
                8,
                invalid,
                48,
                &mut BTreeMap::new(),
                &mut 48
            )
            .is_err()
        );
    }
}

#[test]
fn tlds_level_zero_preserves_coordinates_channels_and_samplerless_binding() {
    let mut bindings = BTreeMap::new();
    let captured = 0xda50_1a40_2077_0600;
    let operation = decode_texture_access_simplified(
        MaxwellShaderStage::Pixel,
        0x30,
        captured,
        8,
        &mut bindings,
        &mut 32,
    )
    .unwrap();
    assert_eq!(operation.len(), 1);
    assert_eq!(
        operation[0],
        ShaderOperation::LoadTexture2D {
            outputs: (0..4)
                .map(|component| ShaderTextureSampleOutput::new(
                    ShaderRegister::new(u16::from(component)),
                    component
                )
                .unwrap())
                .collect(),
            coordinates: [ShaderRegister::new(6), ShaderRegister::new(7)],
            image_binding: 32,
            mip_level: 0,
        }
    );
    assert_eq!(bindings[&420].constant_buffer_byte_offset, 1680);
    assert_eq!(bindings[&420].sampler_binding, None);
    // The same descriptor can later be filtered: reserve its sampler once.
    let sample = (captured & !(0x1f << 53)) | (1 << 53);
    decode_texture_access_simplified(
        MaxwellShaderStage::Pixel,
        0x38,
        sample,
        8,
        &mut bindings,
        &mut 32,
    )
    .unwrap();
    assert_eq!(bindings.len(), 1);
    assert_eq!(bindings[&420].sampler_binding, Some(33));
    decode_texture_access_simplified(
        MaxwellShaderStage::Pixel,
        0x40,
        captured,
        8,
        &mut bindings,
        &mut 32,
    )
    .unwrap();
    assert_eq!(bindings[&420].sampler_binding, Some(33));
    for word in [
        captured & !(1 << 59),
        captured ^ (1 << 53),
        captured | (1 << 55),
    ] {
        assert!(matches!(
            decode_texture_access_simplified(
                MaxwellShaderStage::Pixel,
                0x30,
                word,
                8,
                &mut BTreeMap::new(),
                &mut 32
            ),
            Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail { .. })
        ));
    }
}

#[test]
fn tlds_reaches_wgsl_without_declaring_a_sampler() {
    let mut header = [0_u32; 20];
    header[0] = 0x0002_5462;
    header[18] = 0xf;
    let translated = translated_fixture_with_register_count(
        MaxwellShaderStage::Pixel,
        header,
        &[
            0,
            0x0100_0000_0037_f006,
            0x0100_0000_0057_f007,
            0xda50_1a40_2077_0600,
            0,
            0xe300_0000_0007_000f,
        ],
        8,
    );
    assert_eq!(translated.ir().resources().len(), 1);
    assert_eq!(
        translated.ir().resources()[0].kind(),
        ShaderResourceKind::SampledImage
    );
    let module = lower_shader_ir_to_wgsl(&translated).unwrap();
    assert!(module.source().contains("textureLoad(sampled_image_32"));
    assert!(!module.source().contains("textureSampleBias("));
    assert!(!module.source().contains("var sampler_"));
    validate_wgsl(&module);
    let mapping = BTreeMap::from([(32, 4)]);
    let remapped = finalize_shader_ir(
        translated.ir().clone(),
        MaxwellShaderStage::Pixel,
        &mapping,
        &[],
    )
    .unwrap();
    assert!(
        remapped
            .ir()
            .instructions()
            .iter()
            .any(|instruction| matches!(
                instruction.operation(),
                ShaderOperation::LoadTexture2D {
                    image_binding: 4,
                    ..
                }
            ))
    );
}

#[test]
fn texs_2d_implicit_lod_decodes_captured_split_rgba_operands() {
    let encoding = 0xd830_0080_2007_0100;
    let mut bindings = BTreeMap::new();
    let operation = decode_texture_access_simplified(
        MaxwellShaderStage::Pixel,
        0x2a8,
        encoding,
        4,
        &mut bindings,
        &mut 32,
    )
    .unwrap();
    assert_eq!(operation.len(), 1);

    assert_eq!(
        operation[0],
        ShaderOperation::SampleTexture2D {
            offset: None,
            outputs: (0..4)
                .map(|component| {
                    ShaderTextureSampleOutput::new(
                        ShaderRegister::new(u16::from(component)),
                        component,
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            coordinates: [ShaderRegister::new(1), ShaderRegister::new(0)],
            image_binding: 32,
            lod: None,
            sampler_binding: 33,
        }
    );
    assert_eq!(
        bindings.get(&8),
        Some(&MaxwellTextureResourceBinding {
            constant_buffer_byte_offset: 32,
            image_binding: 32,
            sampler_binding: Some(33),
            image_kind: ShaderResourceKind::SampledImage,
        })
    );
}

#[test]
fn texs_2d_array_implicit_lod_decodes_packed_layer_and_coordinates() {
    let encoding = 0xd8e0_1a4f_f027_0003;
    let mut bindings = BTreeMap::new();
    let operation = decode_texture_access_simplified(
        MaxwellShaderStage::Pixel,
        0x30,
        encoding,
        4,
        &mut bindings,
        &mut 32,
    )
    .unwrap();
    assert_eq!(operation.len(), 1);

    assert_eq!(
        operation[0],
        ShaderOperation::SampleTexture2DArray {
            offset: None,
            outputs: vec![ShaderTextureSampleOutput::new(ShaderRegister::new(3), 0).unwrap()]
                .into_boxed_slice(),
            coordinates: [ShaderRegister::new(1), ShaderRegister::new(2)],
            array_index: ShaderRegister::new(0),
            image_binding: 32,
            lod: None,
            sampler_binding: 33,
        }
    );
    assert_eq!(
        bindings.get(&420),
        Some(&MaxwellTextureResourceBinding {
            constant_buffer_byte_offset: 1680,
            image_binding: 32,
            sampler_binding: Some(33),
            image_kind: ShaderResourceKind::SampledImage2DArray,
        })
    );
}

#[test]
fn texs_2d_array_lowers_to_an_array_texture_and_unsigned_u16_layer() {
    let mut header = [0_u32; 20];
    header[0] = 0x0002_5462;
    header[18] = 0x0000_000f;
    let translated = translated_fixture(
        MaxwellShaderStage::Pixel,
        header,
        &[
            0,
            0x0100_0000_0007_f000_u64 | (7_u64 << 20),
            0x0100_0000_0007_f001_u64 | (u64::from(0.25_f32.to_bits()) << 20),
            0x0100_0000_0007_f002_u64 | (u64::from(0.75_f32.to_bits()) << 20),
            0,
            0xd8e0_1a4f_f027_0003,
            0xe300_0000_0007_000f,
            0,
        ],
    );

    assert!(translated.ir().resources().iter().any(|resource| {
        resource.binding() == 32 && resource.kind() == ShaderResourceKind::SampledImage2DArray
    }));
    let module = lower_shader_ir_to_wgsl(&translated).unwrap();
    assert!(module.source().contains("texture_2d_array<f32>"));
    assert!(module.source().contains("i32(registers[0] & 0xffffu)"));
    validate_wgsl(&module);
}

#[test]
fn texs_2d_implicit_lod_translates_to_verified_sample_resources_and_wgsl() {
    let mut header = [0_u32; 20];
    header[0] = 0x0002_5462;
    header[18] = 0x0000_000f;
    let move_x = 0x0100_0000_0007_f000_u64 | (u64::from(0.25_f32.to_bits()) << 20);
    let move_y = 0x0100_0000_0007_f001_u64 | (u64::from(0.75_f32.to_bits()) << 20);
    let translated = translated_fixture(
        MaxwellShaderStage::Pixel,
        header,
        &[
            0,
            move_x,
            move_y,
            0xd830_0080_2007_0100,
            0,
            0xe300_0000_0007_000f,
            0,
            0,
        ],
    );

    assert!(translated.ir().resources().iter().any(|resource| {
        resource.binding() == 32 && resource.kind() == ShaderResourceKind::SampledImage
    }));
    assert!(translated.ir().resources().iter().any(|resource| {
        resource.binding() == 33 && resource.kind() == ShaderResourceKind::Sampler
    }));
    let module = lower_shader_ir_to_wgsl(&translated).unwrap();
    assert!(module.source().contains("textureSample"));
    validate_wgsl(&module);
}

#[test]
fn texs_half_results_pack_pairs_without_aliasing_sample_outputs() {
    use nixe_gpu::{
        ShaderInstruction, ShaderInterfaceElement, ShaderIoLocation, ShaderIr, ShaderPredicate,
        ShaderScalarType, ShaderSourceLocation, ShaderStage, VerifiedShaderIr,
    };
    let encoding = 0xd030_0080_1007_0100;
    let operations = decode_texture_access_simplified(
        MaxwellShaderStage::Pixel,
        0x30,
        encoding,
        2,
        &mut BTreeMap::new(),
        &mut 2,
    )
    .unwrap();
    let ShaderOperation::SampleTexture2D {
        outputs,
        coordinates,
        ..
    } = &operations[0]
    else {
        panic!("sample expected")
    };
    assert_eq!(
        *coordinates,
        [ShaderRegister::new(1), ShaderRegister::new(0)]
    );
    assert_eq!(
        outputs
            .iter()
            .map(|o| (o.destination().index(), o.component()))
            .collect::<Vec<_>>(),
        [(2, 0), (3, 1), (4, 2), (5, 3)]
    );
    // Isolate the packing arithmetic with deterministic sampled values. The
    // sample uses four temporaries; only two actual result registers are needed.
    let mut code = outputs
        .iter()
        .zip([1.0_f32, -2.0, 0.75, 0.5])
        .map(|(out, value)| ShaderOperation::MoveImmediate32 {
            destination: out.destination(),
            bits: value.to_bits(),
            scalar_type: ShaderScalarType::Float32,
        })
        .collect::<Vec<_>>();
    code.extend_from_slice(&operations[1..]);
    code.push(ShaderOperation::StoreOutput {
        sources: vec![ShaderRegister::new(0), ShaderRegister::new(1)].into(),
        location: ShaderIoLocation::Color(0),
        first_component: 0,
        scalar_type: ShaderScalarType::Unsigned32,
    });
    code.push(ShaderOperation::Exit);
    let ir = VerifiedShaderIr::verify(ShaderIr::new(
        ShaderStage::Fragment,
        Vec::new(),
        (0..2)
            .map(|c| {
                ShaderInterfaceElement::new(
                    ShaderIoLocation::Color(0),
                    c,
                    ShaderScalarType::Unsigned32,
                    None,
                )
                .unwrap()
            })
            .collect(),
        Vec::new(),
        code.into_iter()
            .enumerate()
            .map(|(i, op)| {
                ShaderInstruction::new(
                    ShaderSourceLocation::new(i as u32 * 8),
                    ShaderPredicate::Always,
                    op,
                )
            })
            .collect(),
    ))
    .unwrap();
    let result =
        nixe_gpu::evaluate_shader_ir(&ir, &nixe_gpu::ShaderEvaluationInputs::default(), 64)
            .unwrap();
    assert_eq!(
        result.output_bits(ShaderIoLocation::Color(0), 0),
        Some(0xc000_3c00)
    );
    assert_eq!(
        result.output_bits(ShaderIoLocation::Color(0), 1),
        Some(0x3800_3a00)
    );
    validate_wgsl(&lower_shader_ir_to_wgsl(&ir).unwrap());
    let mut header = [0; 20];
    header[0] = 0x0002_5462;
    header[18] = 3;
    let shader = translated_fixture_with_register_count(
        MaxwellShaderStage::Pixel,
        header,
        &[
            0,
            0x0100_0000_0007_f000,
            0x0100_0000_0007_f001,
            encoding,
            0,
            0xe300_0000_0007_000f,
            0,
            0,
        ],
        2,
    );
    validate_wgsl(&lower_shader_ir_to_wgsl(&shader).unwrap());
}

#[test]
fn texs_explicit_zero_lod_preserves_sampling_semantics() {
    for (selector, encoding) in [
        (2_u64, 0xd830_0080_2007_0100_u64),
        (8, 0xd8e0_1a4f_f027_0003),
    ] {
        let encoding = (encoding & !(15 << 53)) | (selector << 53);
        let operations = decode_texture_access_simplified(
            MaxwellShaderStage::Pixel,
            8,
            encoding,
            4,
            &mut BTreeMap::new(),
            &mut 32,
        )
        .unwrap();
        let ShaderOperation::MoveImmediate32 {
            destination,
            bits: 0,
            scalar_type: nixe_gpu::ShaderScalarType::Float32,
        } = operations[0]
        else {
            panic!("explicit LOD zero required")
        };
        match &operations[1] {
            ShaderOperation::SampleTexture2D { lod, .. }
            | ShaderOperation::SampleTexture2DArray { lod, .. } => {
                assert_eq!(*lod, Some(destination))
            }
            other => panic!("unexpected sample {other:?}"),
        }
    }
}

#[test]
fn tex_packs_sparse_channels_and_shares_the_texs_descriptor() {
    let mut bindings = BTreeMap::new();
    let encoding = 0xc03a_0083_2ff7_0404;
    let operations = decode_texture_sample(
        MaxwellShaderStage::Pixel,
        0x78,
        encoding,
        8,
        &mut bindings,
        &mut 32,
        &[],
    )
    .unwrap();
    assert_eq!(
        operations,
        vec![ShaderOperation::SampleTexture2D {
            offset: None,
            outputs: vec![
                ShaderTextureSampleOutput::new(ShaderRegister::new(4), 1).unwrap(),
                ShaderTextureSampleOutput::new(ShaderRegister::new(5), 2).unwrap(),
            ]
            .into_boxed_slice(),
            coordinates: [ShaderRegister::new(4), ShaderRegister::new(5)],
            image_binding: 32,
            sampler_binding: 33,
            lod: None,
        }]
    );
    decode_texture_access_simplified(
        MaxwellShaderStage::Pixel,
        0x80,
        0xd830_0080_2007_0100,
        8,
        &mut bindings,
        &mut 32,
    )
    .unwrap();
    assert_eq!(bindings.len(), 1);
    assert_eq!(bindings[&8].constant_buffer_byte_offset, 32);
    assert_eq!(bindings[&8].sampler_binding, Some(33));
}

#[test]
fn tex_array_explicit_lod_reads_metadata_and_preserves_the_layer() {
    let encoding = (0xc03a_0083_2ff7_0404 & !((7 << 28) | (7 << 55) | (255 << 20)))
        | (3 << 28)
        | (3 << 55)
        | (7 << 20);
    let operations = decode_texture_sample(
        MaxwellShaderStage::Pixel,
        0x78,
        encoding,
        8,
        &mut BTreeMap::new(),
        &mut 32,
        &[],
    )
    .unwrap();
    assert!(
        matches!(&operations[0], ShaderOperation::SampleTexture2DArray {
        coordinates, array_index, lod: Some(level), ..
    } if *coordinates == [ShaderRegister::new(5), ShaderRegister::new(6)]
        && *array_index == ShaderRegister::new(4) && *level == ShaderRegister::new(7))
    );
}

#[test]
fn tex_modifiers_fail_explicitly_and_lod_zero_does_not_read_rz_metadata() {
    let encoding = 0xc03a_0083_2ff7_0404;
    for word in [
        encoding | (1 << 35),
        encoding | (1 << 58),
        encoding ^ (1 << 51),
        encoding | (2 << 55),
    ] {
        assert!(matches!(
            decode_texture_sample(
                MaxwellShaderStage::Pixel,
                0x78,
                word,
                8,
                &mut BTreeMap::new(),
                &mut 32,
                &[]
            ),
            Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail { .. })
        ));
    }
    let operations = decode_texture_sample(
        MaxwellShaderStage::Pixel,
        0x78,
        encoding | (1 << 55),
        8,
        &mut BTreeMap::new(),
        &mut 32,
        &[],
    )
    .unwrap();
    assert!(matches!(
        operations[0],
        ShaderOperation::MoveImmediate32 {
            bits: 0,
            scalar_type: ShaderScalarType::Float32,
            ..
        }
    ));
    assert!(matches!(operations[1], ShaderOperation::SampleTexture2D {
        lod: Some(level), ..
    } if level == ShaderRegister::new(32)));
    assert!(
        decode_texture_sample(
            MaxwellShaderStage::Pixel,
            0x78,
            encoding,
            5,
            &mut BTreeMap::new(),
            &mut 32,
            &[]
        )
        .is_err()
    );
}

#[test]
fn tex_translation_preserves_predicate_and_samples_before_aliased_writes() {
    let mut header = [0_u32; 20];
    header[0] = 0x0002_5462;
    header[18] = 0xf;
    let encoding = 0xc03a_0083_2ff7_0404;
    let translated = translated_fixture_with_register_count(
        MaxwellShaderStage::Pixel,
        header,
        &[
            0,
            0x0100_0000_0007_f004 | (u64::from(0.25_f32.to_bits()) << 20),
            0x0100_0000_0007_f005 | (u64::from(0.75_f32.to_bits()) << 20),
            0x5bb1_8380_0007_ff07 | (4 << 20),
            0,
            encoding & !(7 << 16),
            0xe300_0000_0007_000f,
        ],
        8,
    );
    let sample = translated
        .ir()
        .instructions()
        .iter()
        .find(|instruction| {
            matches!(
                instruction.operation(),
                ShaderOperation::SampleTexture2D { .. }
            )
        })
        .unwrap();
    assert_eq!(
        sample.predicate(),
        nixe_gpu::ShaderPredicate::Register {
            register: 0,
            inverted: false
        }
    );
    let module = lower_shader_ir_to_wgsl(&translated).unwrap();
    validate_wgsl(&module);
    let source = module.source();
    let sample_position = source.find("textureSampleBias(").unwrap();
    assert!(source[sample_position..].contains("registers[4] ="));
    assert!(source[sample_position..].contains("registers[5] ="));
}

#[test]
fn tex_depth_array_comparison_preserves_reference_layer_channels_and_sampler_type() {
    let word = 0xc03e0120b1770c0d;
    let ops = decode_texture_sample(
        MaxwellShaderStage::Pixel,
        0x1f8,
        word,
        24,
        &mut BTreeMap::new(),
        &mut 32,
        &[],
    )
    .unwrap();
    assert!(matches!(&ops[0],ShaderOperation::SampleDepthCompare2D {
            offset: None, coordinates,array_index:Some(layer),reference,image_binding:32,sampler_binding:33,level_zero:false,outputs}
        if coordinates[0].index()==13 && coordinates[1].index()==14 && layer.index()==12 && reference.index()==23 && outputs[0].destination().index()==13));
    let mut header = [0; 20];
    header[0] = 0x00025462;
    header[18] = 15;
    let ir = translated_fixture_with_register_count(
        MaxwellShaderStage::Pixel,
        header,
        &[0, word & !0xff, 0xe30000000007000f, 0],
        24,
    );
    assert!(
        ir.ir()
            .resources()
            .iter()
            .any(|r| r.kind() == ShaderResourceKind::ComparisonSampler)
    );
    let module = lower_shader_ir_to_wgsl(&ir).unwrap();
    validate_wgsl(&module);
    assert!(
        module.source().contains("texture_depth_2d_array")
            && module.source().contains("textureSampleCompare(")
    );
    let remapped = finalize_shader_ir(
        ir.ir().clone(),
        MaxwellShaderStage::Pixel,
        &BTreeMap::from([(32, 4), (33, 5)]),
        &[],
    )
    .unwrap();
    assert!(remapped.ir().instructions().iter().any(|i| matches!(
        i.operation(),
        ShaderOperation::SampleDepthCompare2D {
            image_binding: 4,
            sampler_binding: 5,
            ..
        }
    )));
}

#[test]
fn tex_aoffi_signed_nibbles_preserve_native_offsets_and_metadata_routing() {
    use nixe_gpu::{ShaderInstruction, ShaderPredicate, ShaderSourceLocation};
    let captured = 0xc0fa_0087_a037_0400;
    for packed in 0..256_u32 {
        let previous = [ShaderInstruction::new(
            ShaderSourceLocation::new(0x18),
            ShaderPredicate::Always,
            ShaderOperation::MoveImmediate32 {
                destination: ShaderRegister::new(3),
                bits: packed,
                scalar_type: ShaderScalarType::Unsigned32,
            },
        )];
        let ops = decode_texture_sample(
            MaxwellShaderStage::Pixel,
            0x30,
            captured,
            8,
            &mut BTreeMap::new(),
            &mut 32,
            &previous,
        )
        .unwrap();
        let expected = [
            ((packed & 15) as i32) << 28 >> 28,
            (((packed >> 4) & 15) as i32) << 28 >> 28,
        ];
        assert!(
            matches!(ops.last().unwrap(),ShaderOperation::SampleTexture2D { offset:Some(offset), lod:Some(_),.. } if *offset==expected)
        );
        let dc = decode_texture_sample(
            MaxwellShaderStage::Pixel,
            0x30,
            captured | (1 << 50),
            8,
            &mut BTreeMap::new(),
            &mut 32,
            &previous,
        )
        .unwrap();
        assert!(
            matches!(dc.last().unwrap(),ShaderOperation::SampleDepthCompare2D { offset:Some(offset),reference,level_zero:true,.. } if *offset==expected && reference.index()==4)
        );
    }
    let mut header = [0; 20];
    header[0] = 0x0002_5462;
    header[18] = 15;
    let shader = translated_fixture_with_register_count(
        MaxwellShaderStage::Pixel,
        header,
        &[
            0,
            0x0100_0000_0887_f003,
            0x0100_0000_0007_f004,
            0x0100_0000_0007_f005,
            0,
            captured,
            0xe300_0000_0007_000f,
        ],
        8,
    );
    let module = lower_shader_ir_to_wgsl(&shader).unwrap();
    assert!(module.source().contains("vec2<i32>(-8, -8)"));
    validate_wgsl(&module);
    let dc_shader = translated_fixture_with_register_count(
        MaxwellShaderStage::Pixel,
        header,
        &[
            0,
            0x0100_0000_0887_f003,
            0x0100_0000_0007_f004,
            0x0100_0000_0007_f005,
            0,
            captured | (1 << 50),
            0xe300_0000_0007_000f,
        ],
        8,
    );
    validate_wgsl(&lower_shader_ir_to_wgsl(&dc_shader).unwrap());
}

#[test]
fn tex_aoffi_never_freezes_uniforms_conditional_writes_or_branch_join_values() {
    use nixe_gpu::{ShaderInstruction, ShaderPredicate, ShaderSourceLocation};
    let r = ShaderRegister::new;
    let literal = ShaderInstruction::new(
        ShaderSourceLocation::new(8),
        ShaderPredicate::Always,
        ShaderOperation::MoveImmediate32 {
            destination: r(3),
            bits: 17,
            scalar_type: ShaderScalarType::Unsigned32,
        },
    );
    let conditional = ShaderInstruction::new(
        ShaderSourceLocation::new(16),
        ShaderPredicate::Register {
            register: 0,
            inverted: false,
        },
        ShaderOperation::MoveImmediate32 {
            destination: r(3),
            bits: 0,
            scalar_type: ShaderScalarType::Unsigned32,
        },
    );
    let branch = ShaderInstruction::new(
        ShaderSourceLocation::new(0),
        ShaderPredicate::Always,
        ShaderOperation::Branch {
            target: ShaderSourceLocation::new(24),
        },
    );
    let uniform = ShaderInstruction::new(
        ShaderSourceLocation::new(8),
        ShaderPredicate::Always,
        ShaderOperation::LoadConstantBuffer32 {
            destination: r(3),
            binding: 0,
            byte_offset: 0,
            scalar_type: ShaderScalarType::Unsigned32,
        },
    );
    for previous in [
        vec![literal.clone(), conditional],
        vec![branch, literal],
        vec![uniform],
        vec![],
    ] {
        assert!(matches!(
            decode_texture_sample(
                MaxwellShaderStage::Pixel,
                0x30,
                0xc0fa_0087_a037_0400,
                8,
                &mut BTreeMap::new(),
                &mut 32,
                &previous
            ),
            Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
                detail: "TEX AOFFI operand is not a proven shader constant",
                ..
            })
        ));
    }
}
