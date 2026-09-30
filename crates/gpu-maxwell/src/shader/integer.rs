//! Maxwell integer comparison decoding. ISA fields follow the public GM107
//! emitter, not instruction samples:
//! https://github.com/devkitPro/uam/blob/master/mesa-imported/codegen/nv50_ir_emit_gm107.cpp#L2084-L2130
use super::*;
use nixe_gpu::ShaderIntegerComparison;

pub(super) const fn is_set_predicate(encoding: u64) -> bool {
    let opcode = (encoding >> 48) as u16;
    opcode & 0xfff0 == 0x5b60 || opcode & 0xfff0 == 0x4b60 || opcode & 0xfef0 == 0x3660
}

pub(super) struct DecodedIntegerPredicate {
    pub operations: Vec<ShaderOperation>,
    pub constant_buffer_binding: Option<u8>,
}

pub(super) fn decode_set_predicate(
    stage: MaxwellThreeDShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
    next_temporary: &mut u16,
) -> Result<DecodedIntegerPredicate, MaxwellShaderTranslationError> {
    if encoding & (1 << 43) != 0 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "ISETP.X condition-code input",
        });
    }
    let opcode = (encoding >> 48) as u16;
    let register = opcode & 0xfff0 == 0x5b60;
    let constant = opcode & 0xfff0 == 0x4b60;
    let operand_mask = if register {
        0xff_u64 << 20
    } else {
        0x7ffff_u64 << 20
    };
    let allowed = 0xffff_0000_000f_ff3f_u64 | operand_mask | (0xf << 39) | (3 << 45);
    if encoding & !allowed != 0 {
        return Err(malformed(
            stage,
            offset,
            encoding,
            "ISETP reserved field is nonzero",
        ));
    }
    let destinations = [((encoding >> 3) & 7) as u8, (encoding & 7) as u8]
        .map(|value| (value != 7).then_some(value));
    if destinations[0].is_some() && destinations[0] == destinations[1] {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "ISETP aliased predicate destinations",
        });
    }
    let set_operation = match (encoding >> 45) & 3 {
        0 => ShaderPredicateSetOperation::And,
        1 => ShaderPredicateSetOperation::Or,
        2 => ShaderPredicateSetOperation::Xor,
        _ => {
            return Err(malformed(
                stage,
                offset,
                encoding,
                "ISETP reserved boolean operation",
            ));
        }
    };
    let comparison = match (encoding >> 49) & 7 {
        0 => ShaderIntegerComparison::False,
        1 => ShaderIntegerComparison::Less,
        2 => ShaderIntegerComparison::Equal,
        3 => ShaderIntegerComparison::LessOrEqual,
        4 => ShaderIntegerComparison::Greater,
        5 => ShaderIntegerComparison::NotEqual,
        6 => ShaderIntegerComparison::GreaterOrEqual,
        7 => ShaderIntegerComparison::True,
        _ => unreachable!(),
    };
    let mut operations = Vec::with_capacity(3);
    let mut source_register = |raw: u8,
                               operations: &mut Vec<ShaderOperation>|
     -> Result<ShaderRegister, MaxwellShaderTranslationError> {
        if raw == 0xff {
            let destination = allocate_shader_temporary(
                stage,
                offset,
                encoding,
                "ISETP zero operand temporary overflow",
                next_temporary,
            )?;
            operations.push(ShaderOperation::MoveImmediate32 {
                destination,
                bits: 0,
                scalar_type: ShaderScalarType::Unsigned32,
            });
            Ok(destination)
        } else {
            validate_register_range(stage, offset, encoding, raw, 1, register_count)?;
            Ok(ShaderRegister::new(u16::from(raw)))
        }
    };
    let left = source_register((encoding >> 8) as u8, &mut operations)?;
    let mut constant_buffer_binding = None;
    let right = if register {
        source_register((encoding >> 20) as u8, &mut operations)?
    } else {
        let destination = allocate_shader_temporary(
            stage,
            offset,
            encoding,
            "ISETP right operand temporary overflow",
            next_temporary,
        )?;
        if constant {
            let binding = ((encoding >> 34) & 0x1f) as u8;
            constant_buffer_binding = Some(binding);
            operations.push(ShaderOperation::LoadConstantBuffer32 {
                destination,
                binding,
                byte_offset: (((encoding >> 20) & 0x3fff) as u32) * 4,
                scalar_type: ShaderScalarType::Unsigned32,
            });
        } else {
            let bits = ((encoding >> 20) & 0x7ffff) as u32
                | if encoding & (1 << 56) != 0 {
                    0xfff8_0000
                } else {
                    0
                };
            operations.push(ShaderOperation::MoveImmediate32 {
                destination,
                bits,
                scalar_type: ShaderScalarType::Unsigned32,
            });
        }
        destination
    };
    operations.push(ShaderOperation::SetPredicateInteger32 {
        destinations,
        left,
        right,
        signed: encoding & (1 << 48) != 0,
        comparison,
        accumulator: decode_predicate_fields(encoding, 39, 42),
        set_operation,
    });
    Ok(DecodedIntegerPredicate {
        operations,
        constant_buffer_binding,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const STAGE: MaxwellThreeDShaderStage = MaxwellThreeDShaderStage::TessellationInit;

    #[test]
    fn isetp_register_immediate_and_constant_forms() {
        let captured = 0x5b64_0380_0ff7_0007;
        let decoded = decode_set_predicate(STAGE, 0x10, captured, 5, &mut 5).unwrap();
        assert!(matches!(
            decoded.operations.as_slice(),
            [
                ShaderOperation::MoveImmediate32 { bits: 0, .. },
                ShaderOperation::SetPredicateInteger32 {
                    destinations: [Some(0), None],
                    signed: false,
                    comparison: ShaderIntegerComparison::Equal,
                    accumulator: ShaderPredicate::Always,
                    set_operation: ShaderPredicateSetOperation::And,
                    ..
                }
            ]
        ));
        for (encoding, bits) in [
            (0x3664_0380_0017_0007, 1),
            (0x3764_03ff_fff7_0007, u32::MAX),
            (0x3764_0380_0007_0007, 0xfff8_0000),
        ] {
            assert!(is_set_predicate(encoding));
            let decoded = decode_set_predicate(STAGE, 8, encoding, 5, &mut 5).unwrap();
            assert!(
                matches!(decoded.operations[0], ShaderOperation::MoveImmediate32 { bits: actual, .. } if actual == bits)
            );
        }
        let encoding = 0x4b65_0380_0037_0007 | (2 << 34);
        let decoded = decode_set_predicate(STAGE, 8, encoding, 5, &mut 5).unwrap();
        assert_eq!(decoded.constant_buffer_binding, Some(2));
        assert!(matches!(
            decoded.operations[0],
            ShaderOperation::LoadConstantBuffer32 {
                binding: 2,
                byte_offset: 12,
                ..
            }
        ));
        assert!(matches!(
            decoded.operations[1],
            ShaderOperation::SetPredicateInteger32 { signed: true, .. }
        ));
        for bits in [1 << 6, 1 << 7, 1 << 28, 1 << 38, 1 << 44, 1 << 47, 3 << 45] {
            assert!(
                decode_set_predicate(STAGE, 8, captured | bits, 5, &mut 5).is_err(),
                "reserved bits {bits:x}"
            );
        }
        assert!(matches!(
            decode_set_predicate(STAGE, 8, captured | (1 << 43), 5, &mut 5),
            Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
                detail: "ISETP.X condition-code input",
                ..
            })
        ));
    }
}
