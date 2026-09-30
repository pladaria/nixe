use super::*;
use crate::TessellationControl;

fn shaders() -> (VerifiedShaderIr, VerifiedShaderIr) {
    let components = vec![
        interface(ShaderIoLocation::Position, 2, ShaderScalarType::Float32),
        interface(
            ShaderIoLocation::Generic(7),
            3,
            ShaderScalarType::Unsigned32,
        ),
        interface(ShaderIoLocation::Generic(9), 1, ShaderScalarType::Signed32),
    ];
    let vertex = ShaderIr::new(
        ShaderStage::Vertex,
        vec![],
        components.clone(),
        vec![],
        instructions(vec![ShaderOperation::Exit]),
    );
    let evaluation = ShaderIr::new(
        ShaderStage::TessellationEvaluation,
        components,
        vec![],
        vec![],
        instructions(vec![ShaderOperation::Exit]),
    );
    (
        VerifiedShaderIr::verify(vertex).unwrap(),
        VerifiedShaderIr::verify(evaluation).unwrap(),
    )
}

fn options(domain: TessellationDomain, points: u8) -> SpirvDefaultControlOptions {
    SpirvDefaultControlOptions {
        input_control_points: points,
        domain,
        push_constant_offset: 16,
    }
}

#[test]
fn default_control_has_dynamic_raw_levels_and_only_forwards_required_components() {
    let (vertex, evaluation) = shaders();
    for (domain, mask) in [
        (TessellationDomain::Triangles, 0b01_0111),
        (TessellationDomain::Quads, 0b11_1111),
        (TessellationDomain::Isolines, 0b00_0011),
    ] {
        let shader =
            lower_default_tessellation_control_to_spirv(&vertex, &evaluation, options(domain, 5))
                .unwrap();
        assert_eq!(shader.required_levels(), mask);
        let before = shader.module().clone();
        let words = [
            0x8000_0000,
            0x7fc0_1234,
            1,
            0x7f80_0000,
            0xff80_0000,
            0x3f80_0000,
        ];
        assert_eq!(
            shader
                .parameters(TessellationControl::DefaultLevels {
                    outer: words[..4].try_into().unwrap(),
                    inner: words[4..].try_into().unwrap(),
                    defined: mask
                })
                .unwrap(),
            words
        );
        assert_eq!(
            shader.module(),
            &before,
            "dynamic parameters cannot modify compiled code"
        );
        assert!(matches!(
            shader.parameters(TessellationControl::DefaultLevels {
                outer: [0; 4],
                inner: [0; 2],
                defined: mask & !1
            }),
            Err(SpirvShaderError::DefaultTessellationLevels { .. })
        ));
        assert!(shader.parameters(TessellationControl::Shader).is_err());
        let module = rspirv::dr::load_words(shader.module().words()).unwrap();
        let text = module.disassemble();
        assert!(text.contains("OutputVertices 5"));
        assert!(text.contains("PushConstant"));
        assert!(!text.contains("DescriptorSet"));
        assert!(!text.contains("OpControlBarrier"));
        assert!(!text.contains("Float64"));
        assert!(!text.contains("OpFAdd"));
        assert_eq!(
            ops(&module, spv::Op::Store).len(),
            3 + mask.count_ones() as usize
        );
    }
}

#[test]
fn default_control_rejects_missing_or_mistyped_links_and_undefined_patch_values() {
    let (vertex, evaluation) = shaders();
    let opts = options(TessellationDomain::Triangles, 4);
    for location in [ShaderIoLocation::Patch(0), ShaderIoLocation::Generic(3)] {
        let mut ir = evaluation.ir().clone();
        ir.inputs = ir
            .inputs
            .iter()
            .copied()
            .chain([interface(location, 0, ShaderScalarType::Float32)])
            .collect();
        let ir = VerifiedShaderIr::verify(ir).unwrap();
        assert!(
            matches!(lower_default_tessellation_control_to_spirv(&vertex, &ir, opts), Err(SpirvShaderError::Interface(l)) if l == location)
        );
    }
    let mut ir = evaluation.ir().clone();
    ir.inputs[1].scalar_type = ShaderScalarType::Float32;
    let ir = VerifiedShaderIr::verify(ir).unwrap();
    assert!(matches!(
        lower_default_tessellation_control_to_spirv(&vertex, &ir, opts),
        Err(SpirvShaderError::Interface(_))
    ));
    for opts in [
        SpirvDefaultControlOptions {
            input_control_points: 0,
            ..opts
        },
        SpirvDefaultControlOptions {
            push_constant_offset: 3,
            ..opts
        },
        SpirvDefaultControlOptions {
            push_constant_offset: u32::MAX - 3,
            ..opts
        },
    ] {
        assert!(matches!(
            lower_default_tessellation_control_to_spirv(&vertex, &evaluation, opts),
            Err(SpirvShaderError::Options(_))
        ));
    }
}

#[test]
fn evaluation_level_reads_extend_the_fixed_function_level_mask() {
    let (vertex, evaluation) = shaders();
    let mut ir = evaluation.ir().clone();
    ir.inputs = ir
        .inputs
        .iter()
        .copied()
        .chain([
            interface(
                ShaderIoLocation::TessLevelOuter,
                3,
                ShaderScalarType::Float32,
            ),
            interface(
                ShaderIoLocation::TessLevelInner,
                1,
                ShaderScalarType::Float32,
            ),
        ])
        .collect();
    let evaluation = VerifiedShaderIr::verify(ir).unwrap();
    let shader = lower_default_tessellation_control_to_spirv(
        &vertex,
        &evaluation,
        options(TessellationDomain::Isolines, 1),
    )
    .unwrap();
    assert_eq!(shader.required_levels(), 0b10_1011);
}

pub(super) fn validation_fixtures() -> Vec<(String, SpirvShaderModule)> {
    let (vertex, evaluation) = shaders();
    [
        TessellationDomain::Triangles,
        TessellationDomain::Quads,
        TessellationDomain::Isolines,
    ]
    .into_iter()
    .flat_map(|domain| {
        [1, 4, 5, 32].map(|points| {
            let shader = lower_default_tessellation_control_to_spirv(
                &vertex,
                &evaluation,
                options(domain, points),
            )
            .unwrap();
            (
                format!("default-control-{domain:?}-{points}"),
                shader.module().clone(),
            )
        })
    })
    .collect()
}
