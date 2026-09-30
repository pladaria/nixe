use super::*;
use nixe_gpu::{ShaderEvaluationInputs, evaluate_shader_ir};

fn decoded(
    kind: u8,
    ftz: bool,
    dnz: bool,
) -> Result<Vec<ShaderOperation>, MaxwellShaderTranslationError> {
    let stage = MaxwellThreeDShaderStage::Vertex;
    let mut temporary = 4;
    match kind {
        0 => decode_float_add(
            stage,
            8,
            0x5c58_0000_0017_0003 | (u64::from(ftz) << 44),
            4,
            &mut temporary,
        )
        .map(|d| d.operations),
        1 => decode_float_multiply(
            stage,
            8,
            0x5c68_0000_0017_0003 | (u64::from(ftz) << 44) | (u64::from(dnz) << 45),
            4,
            &mut temporary,
        )
        .map(|d| d.operations),
        _ => decode_float_fused_multiply_add(
            stage,
            8,
            0x5980_0100_0017_0003 | (u64::from(ftz) << 53) | (u64::from(dnz) << 54),
            4,
            &mut temporary,
        )
        .map(|d| d.operations),
    }
}

fn evaluate(kind: u8, ftz: bool, values: [u32; 3]) -> u32 {
    let mut code: Vec<_> = values
        .into_iter()
        .enumerate()
        .map(|(i, bits)| ShaderOperation::MoveImmediate32 {
            destination: ShaderRegister::new(i as u16),
            bits,
            scalar_type: ShaderScalarType::Float32,
        })
        .collect();
    code.extend(decoded(kind, ftz, false).unwrap());
    code.push(ShaderOperation::StoreOutput {
        sources: vec![ShaderRegister::new(3)].into(),
        location: ShaderIoLocation::Generic(0),
        first_component: 0,
        scalar_type: ShaderScalarType::Float32,
    });
    code.push(ShaderOperation::Exit);
    let ir = ShaderIr::new(
        ShaderStage::Vertex,
        vec![],
        vec![
            ShaderInterfaceElement::new(
                ShaderIoLocation::Generic(0),
                0,
                ShaderScalarType::Float32,
                None,
            )
            .unwrap(),
        ],
        vec![],
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
    );
    let verified = VerifiedShaderIr::verify(ir).unwrap();
    evaluate_shader_ir(&verified, &ShaderEvaluationInputs::default(), 32)
        .unwrap()
        .output_bits(ShaderIoLocation::Generic(0), 0)
        .unwrap()
}

#[test]
fn ftz_is_not_an_independent_output_only_modifier() {
    for kind in 0..3 {
        for ftz in [false, true] {
            let operations = decoded(kind, ftz, false).unwrap();
            let control = match operations.last().unwrap() {
                ShaderOperation::Add32 { float_control, .. }
                | ShaderOperation::Multiply32 { float_control, .. }
                | ShaderOperation::FusedMultiplyAdd32 { float_control, .. } => float_control,
                _ => panic!("unexpected arithmetic lowering"),
            };
            assert_eq!(control.flush_denormals_to_zero(), ftz);
            assert_eq!(control.denormals_are_zero(), ftz);
        }
    }
    // A tiny input can affect a NORMAL result; output-only FTZ loses semantics.
    assert_eq!(evaluate(0, false, [1, 0x0080_0000, 0]), 0x0080_0001);
    assert_eq!(evaluate(0, true, [1, 0x0080_0000, 0]), 0x0080_0000);
    assert_eq!(
        evaluate(1, false, [1, 0x7e80_0000, 0]),
        (2_f32.powi(-23)).to_bits()
    );
    assert_eq!(evaluate(1, true, [1, 0x7e80_0000, 0]), 0);
    assert_eq!(
        evaluate(1, true, [0x8000_0001, 0x7e80_0000, 0]),
        0x8000_0000
    );
    assert_eq!(
        evaluate(2, false, [1, 0x7e80_0000, 0]),
        (2_f32.powi(-23)).to_bits()
    );
    assert_eq!(evaluate(2, true, [1, 0x7e80_0000, 0]), 0);
    // FMA must flush the addend too, while preserving single-rounding semantics.
    assert!(evaluate(2, false, [0x0080_0000, 0x3f00_0000, 0x007f_ffff]) >= 0x0080_0000);
    assert_eq!(
        evaluate(2, true, [0x0080_0000, 0x3f00_0000, 0x007f_ffff]),
        0
    );
}

#[test]
fn dnz_is_rejected_instead_of_misrepresented_as_daz() {
    for kind in [1, 2] {
        for ftz in [false, true] {
            assert!(matches!(decoded(kind,ftz,true),
                Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
                    stage: MaxwellThreeDShaderStage::Vertex, instruction_offset: 8, detail, ..
                }) if detail.contains("DNZ zero-multiply")));
        }
    }
}
