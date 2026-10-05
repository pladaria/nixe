//! Packed half multiply with explicit operand swizzles and result conversion.
use super::decode::{allocate_shader_temporary, validate_register_range};
use super::error::MaxwellShaderTranslationError;
use super::float::{apply_float_source_modifiers, prepare_float_register_source};
use crate::MaxwellShaderStage;
use nixe_gpu::{
    ShaderBitwiseOperation, ShaderFloatControl, ShaderOperation, ShaderRegister, ShaderScalarType,
};

pub(super) const fn is_half_multiply(encoding: u64) -> bool {
    encoding & 0xfff8_0000_0000_0000 == 0x5d08_0000_0000_0000
}

pub(super) fn decode_half_multiply(
    stage: MaxwellShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
    next_temporary: &mut u16,
) -> Result<Vec<ShaderOperation>, MaxwellShaderTranslationError> {
    // HMUL2 register encoding and float/half swizzle/result modes:
    // https://github.com/envytools/envytools/blob/master/envydis/gm107.c#L1133-L1162
    // https://github.com/eden-emulator/mirror/blob/d16735f5b618942136d6ab53466e3be0a382c30a/src/shader_recompiler/frontend/maxwell/translate/impl/half_floating_point_multiply.cpp
    // https://github.com/eden-emulator/mirror/blob/d16735f5b618942136d6ab53466e3be0a382c30a/src/shader_recompiler/frontend/maxwell/translate/impl/half_floating_point_helper.cpp
    let precision = (encoding >> 39) & 3;
    if precision != 0 || encoding & (1 << 32) != 0 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "HMUL2 FTZ/FMZ or saturation mode is not implemented",
        });
    }
    let raw_destination = encoding as u8;
    validate_register_range(stage, offset, encoding, raw_destination, 1, register_count)?;
    let destination = ShaderRegister::new(u16::from(raw_destination));
    let mut operations = Vec::new();
    let left = prepare_float_register_source(
        stage,
        offset,
        encoding,
        (encoding >> 8) as u8,
        false,
        false,
        register_count,
        next_temporary,
        &mut operations,
    )?;
    let right = prepare_float_register_source(
        stage,
        offset,
        encoding,
        (encoding >> 20) as u8,
        false,
        false,
        register_count,
        next_temporary,
        &mut operations,
    )?;
    let temporary = |next: &mut u16| {
        allocate_shader_temporary(
            stage,
            offset,
            encoding,
            "HMUL2 temporary register overflow",
            next,
        )
    };
    let merge = (encoding >> 49) & 3;
    let lanes: &[u8] = match merge {
        0 => &[0, 1],
        3 => &[1],
        _ => &[0],
    };
    let mut results = [ShaderRegister::new(0); 2];
    for (result_index, &lane) in lanes.iter().enumerate() {
        let mut operands = [ShaderRegister::new(0); 2];
        for (operand_index, (source, swizzle, absolute, negate)) in [
            (left, (encoding >> 47) & 3, encoding & (1 << 44) != 0, false),
            (
                right,
                (encoding >> 28) & 3,
                encoding & (1 << 30) != 0,
                encoding & (1 << 31) != 0,
            ),
        ]
        .into_iter()
        .enumerate()
        {
            let value = if swizzle == 1 {
                source
            } else {
                let unpacked = temporary(next_temporary)?;
                operations.push(ShaderOperation::UnpackHalf32 {
                    destination: unpacked,
                    source,
                    high: match swizzle {
                        0 => lane == 1,
                        2 => false,
                        3 => true,
                        _ => unreachable!(),
                    },
                });
                unpacked
            };
            operands[operand_index] = apply_float_source_modifiers(
                stage,
                offset,
                encoding,
                value,
                absolute,
                negate,
                next_temporary,
                &mut operations,
            )?;
        }
        let product = temporary(next_temporary)?;
        operations.push(ShaderOperation::Multiply32 {
            destination: product,
            left: operands[0],
            right: operands[1],
            scalar_type: ShaderScalarType::Float32,
            float_control: ShaderFloatControl::PRECISE,
        });
        let rounded = temporary(next_temporary)?;
        // Two binary16 significands have an exact binary32 product; mixed F32
        // input uses F32 multiplication before the same half result rounding.
        operations.push(ShaderOperation::PackHalf32 {
            destination: rounded,
            source: product,
        });
        results[result_index] = rounded;
    }
    if merge == 1 {
        operations.push(ShaderOperation::UnpackHalf32 {
            destination,
            source: results[0],
            high: false,
        });
    } else {
        let shift = temporary(next_temporary)?;
        operations.push(ShaderOperation::MoveImmediate32 {
            destination: shift,
            bits: 16,
            scalar_type: ShaderScalarType::Unsigned32,
        });
        let high = temporary(next_temporary)?;
        if merge == 0 || merge == 3 {
            operations.push(ShaderOperation::ShiftLeft32 {
                destination: high,
                value: results[lanes.len() - 1],
                amount: shift,
                wrap: false,
            });
        }
        let (low, high) = if merge == 0 {
            (results[0], high)
        } else {
            let mask = temporary(next_temporary)?;
            let preserved = temporary(next_temporary)?;
            operations.push(ShaderOperation::MoveImmediate32 {
                destination: mask,
                bits: if merge == 2 { 0xffff_0000 } else { 0xffff },
                scalar_type: ShaderScalarType::Unsigned32,
            });
            operations.push(ShaderOperation::Bitwise32 {
                destination: preserved,
                left: destination,
                right: mask,
                operation: ShaderBitwiseOperation::And,
            });
            if merge == 2 {
                (results[0], preserved)
            } else {
                (preserved, high)
            }
        };
        operations.push(ShaderOperation::Bitwise32 {
            destination,
            left: low,
            right: high,
            operation: ShaderBitwiseOperation::Or,
        });
    }
    Ok(operations)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nixe_gpu::{
        ShaderInstruction, ShaderInterfaceElement, ShaderIoLocation, ShaderIr, ShaderPredicate,
        ShaderSourceLocation, ShaderStage, VerifiedShaderIr,
    };
    #[test]
    fn hmul2_float_output_retains_half_result_rounding_and_operand_aliasing() {
        let mut code = vec![
            ShaderOperation::MoveImmediate32 {
                destination: ShaderRegister::new(1),
                bits: 1.23456_f32.to_bits(),
                scalar_type: ShaderScalarType::Float32,
            },
            ShaderOperation::MoveImmediate32 {
                destination: ShaderRegister::new(4),
                bits: 0xbe00_3a00,
                scalar_type: ShaderScalarType::Unsigned32,
            },
        ];
        // F32 left, broadcast low half right, F32 result; destination aliases left.
        code.extend(
            decode_half_multiply(
                MaxwellShaderStage::Pixel,
                8,
                0x5d0a_8000_2047_0101,
                5,
                &mut 5,
            )
            .unwrap(),
        );
        code.push(ShaderOperation::StoreOutput {
            sources: vec![ShaderRegister::new(1)].into(),
            location: ShaderIoLocation::Color(0),
            first_component: 0,
            scalar_type: ShaderScalarType::Float32,
        });
        code.push(ShaderOperation::Exit);
        let ir = VerifiedShaderIr::verify(ShaderIr::new(
            ShaderStage::Fragment,
            Vec::new(),
            vec![
                ShaderInterfaceElement::new(
                    ShaderIoLocation::Color(0),
                    0,
                    ShaderScalarType::Float32,
                    None,
                )
                .unwrap(),
            ],
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
            Some(0x3f6d_0000)
        );
        let wgsl = nixe_gpu::lower_shader_ir_to_wgsl(&ir).unwrap();
        super::super::test_support::validate_wgsl(&wgsl);
        assert!(
            decode_half_multiply(
                MaxwellShaderStage::Pixel,
                8,
                0x5d0a_8080_2047_0101,
                5,
                &mut 5
            )
            .is_err()
        );
    }
}
