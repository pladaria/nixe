//! Maxwell integer instruction decoding. ISA fields follow the public GM107
//! emitter, not instruction samples:
//! https://github.com/devkitPro/uam/blob/master/mesa-imported/codegen/nv50_ir_emit_gm107.cpp#L2084-L2130
use super::*;
use nixe_gpu::{ShaderBitwiseOperation, ShaderIntegerComparison};

pub(super) const fn is_bitwise(encoding: u64) -> bool {
    let opcode = (encoding >> 48) as u16;
    opcode & 0xfff8 == 0x5c40 || opcode & 0xfff8 == 0x4c40 || opcode & 0xfef8 == 0x3840
}

pub(super) struct DecodedIntegerOperation {
    pub operations: Vec<ShaderOperation>,
    pub constant_buffer_binding: Option<u8>,
}

pub(super) fn decode_bitwise(
    stage: MaxwellThreeDShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
    next_temporary: &mut u16,
) -> Result<DecodedIntegerOperation, MaxwellShaderTranslationError> {
    // LOP register/cbuf/signed-20-bit immediate forms and operand complements.
    // https://github.com/devkitPro/uam/blob/master/mesa-imported/codegen/nv50_ir_emit_gm107.cpp#L1568-L1645
    let error = |detail| MaxwellShaderTranslationError::UnsupportedSemanticDetail {
        stage,
        instruction_offset: offset,
        encoding,
        detail,
    };
    if encoding & (1 << 47) != 0 {
        return Err(error("LOP condition-code write"));
    }
    if encoding & (1 << 43) != 0 {
        return Err(error("LOP extended condition-code input"));
    }
    if (encoding >> 48) & 7 != 7 || encoding & (3 << 44) != 0 {
        return Err(error("LOP predicate output"));
    }
    let opcode = (encoding >> 48) as u16;
    let register = opcode & 0xfff8 == 0x5c40;
    let constant = opcode & 0xfff8 == 0x4c40;
    let operand_mask = if register {
        0xff_u64 << 20
    } else {
        0x7ffff_u64 << 20
    };
    let allowed = 0xffff_0000_000f_ffff | operand_mask | (0xf << 39);
    if encoding & !allowed != 0 {
        return Err(malformed(
            stage,
            offset,
            encoding,
            "LOP reserved field is nonzero",
        ));
    }
    let mut operations = Vec::with_capacity(6);
    let mut temporary = || {
        allocate_shader_temporary(
            stage,
            offset,
            encoding,
            "LOP temporary register overflow",
            next_temporary,
        )
    };
    let destination = encoding as u8;
    let destination = if destination == 0xff {
        temporary()?
    } else {
        validate_register_range(stage, offset, encoding, destination, 1, register_count)?;
        ShaderRegister::new(u16::from(destination))
    };
    let mut operand = |raw: u8,
                       operations: &mut Vec<ShaderOperation>|
     -> Result<_, MaxwellShaderTranslationError> {
        if raw == 0xff {
            let destination = temporary()?;
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
    let selector = (encoding >> 41) & 3;
    // PASS_B does not read the first operand.
    let left = if selector == 3 {
        None
    } else {
        Some(operand((encoding >> 8) as u8, &mut operations)?)
    };
    let mut constant_buffer_binding = None;
    let right = if register {
        operand((encoding >> 20) as u8, &mut operations)?
    } else {
        let destination = temporary()?;
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
    let mut complement = |value, inverted: bool| -> Result<_, MaxwellShaderTranslationError> {
        if !inverted {
            return Ok(value);
        }
        let mask = temporary()?;
        let result = temporary()?;
        operations.push(ShaderOperation::MoveImmediate32 {
            destination: mask,
            bits: u32::MAX,
            scalar_type: ShaderScalarType::Unsigned32,
        });
        operations.push(ShaderOperation::Bitwise32 {
            destination: result,
            left: value,
            right: mask,
            operation: ShaderBitwiseOperation::Xor,
        });
        Ok(result)
    };
    let left = left
        .map(|left| complement(left, encoding & (1 << 39) != 0))
        .transpose()?;
    let right = complement(right, encoding & (1 << 40) != 0)?;
    operations.push(if let Some(left) = left {
        ShaderOperation::Bitwise32 {
            destination,
            left,
            right,
            operation: match selector {
                0 => ShaderBitwiseOperation::And,
                1 => ShaderBitwiseOperation::Or,
                2 => ShaderBitwiseOperation::Xor,
                _ => unreachable!(),
            },
        }
    } else {
        ShaderOperation::Move32 {
            destination,
            source: right,
            scalar_type: ShaderScalarType::Unsigned32,
        }
    });
    Ok(DecodedIntegerOperation {
        operations,
        constant_buffer_binding,
    })
}

pub(super) const fn is_set_predicate(encoding: u64) -> bool {
    let opcode = (encoding >> 48) as u16;
    opcode & 0xfff0 == 0x5b60 || opcode & 0xfff0 == 0x4b60 || opcode & 0xfef0 == 0x3660
}

pub(super) const fn is_add(encoding: u64) -> bool {
    let opcode = (encoding >> 48) as u16;
    opcode & 0xfff8 == 0x5c10 || opcode & 0xfff8 == 0x4c10 || opcode & 0xfef8 == 0x3810
}

pub(super) fn decode_add(
    stage: MaxwellThreeDShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
    next_temporary: &mut u16,
) -> Result<DecodedIntegerOperation, MaxwellShaderTranslationError> {
    // IADD uses modulo-2^32 addition. Operand forms, sign extension and flags:
    // https://github.com/devkitPro/uam/blob/master/mesa-imported/codegen/nv50_ir_emit_gm107.cpp#L1648-L1685
    for (mask, detail) in [
        (1 << 47, "IADD condition-code write"),
        (1 << 43, "IADD extended carry input"),
        (1 << 50, "IADD signed saturation"),
        (3 << 48, "IADD negated operands"),
    ] {
        if encoding & mask != 0 {
            return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
                stage,
                instruction_offset: offset,
                encoding,
                detail,
            });
        }
    }
    let opcode = (encoding >> 48) as u16;
    let register = opcode == 0x5c10;
    let constant = opcode == 0x4c10;
    let operand_mask = if register {
        0xff_u64 << 20
    } else {
        0x7ffff_u64 << 20
    };
    if encoding & !(0xffff_0000_000f_ffff | operand_mask) != 0 {
        return Err(malformed(
            stage,
            offset,
            encoding,
            "IADD reserved field is nonzero",
        ));
    }
    let mut operations = Vec::with_capacity(3);
    let mut temporary = || {
        allocate_shader_temporary(
            stage,
            offset,
            encoding,
            "IADD temporary register overflow",
            next_temporary,
        )
    };
    let destination = encoding as u8;
    let destination = if destination == 0xff {
        temporary()?
    } else {
        validate_register_range(stage, offset, encoding, destination, 1, register_count)?;
        ShaderRegister::new(u16::from(destination))
    };
    let mut operand = |raw: u8,
                       operations: &mut Vec<ShaderOperation>|
     -> Result<_, MaxwellShaderTranslationError> {
        if raw == 0xff {
            let destination = temporary()?;
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
    let left = operand((encoding >> 8) as u8, &mut operations)?;
    let mut constant_buffer_binding = None;
    let right = if register {
        operand((encoding >> 20) as u8, &mut operations)?
    } else {
        let destination = temporary()?;
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
    operations.push(ShaderOperation::Add32 {
        destination,
        left,
        right,
        scalar_type: ShaderScalarType::Unsigned32,
        float_control: ShaderFloatControl::new(
            ShaderRoundingMode::NearestEven,
            ShaderNanMode::Propagate,
            false,
            false,
            false,
        ),
    });
    Ok(DecodedIntegerOperation {
        operations,
        constant_buffer_binding,
    })
}

pub(super) fn decode_set_predicate(
    stage: MaxwellThreeDShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
    next_temporary: &mut u16,
) -> Result<DecodedIntegerOperation, MaxwellShaderTranslationError> {
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
    Ok(DecodedIntegerOperation {
        operations,
        constant_buffer_binding,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const STAGE: MaxwellThreeDShaderStage = MaxwellThreeDShaderStage::TessellationInit;

    fn evaluate_lop(encoding: u64, left: u32, right: u32) -> u32 {
        let decoded = decode_bitwise(STAGE, 8, encoding, 4, &mut 4).unwrap();
        evaluate_integer(decoded, left, right)
    }

    fn evaluate_iadd(encoding: u64, left: u32, right: u32) -> u32 {
        assert!(is_add(encoding));
        let decoded = decode_add(STAGE, 8, encoding, 4, &mut 4).unwrap();
        evaluate_integer(decoded, left, right)
    }

    fn evaluate_integer(decoded: DecodedIntegerOperation, left: u32, right: u32) -> u32 {
        let resources = decoded
            .constant_buffer_binding
            .into_iter()
            .map(|binding| {
                ShaderResourceAccess::new(binding, ShaderResourceKind::ConstantBuffer, true, false)
                    .unwrap()
            })
            .collect();
        let mut operations = vec![
            ShaderOperation::MoveImmediate32 {
                destination: ShaderRegister::new(0),
                bits: left,
                scalar_type: ShaderScalarType::Unsigned32,
            },
            ShaderOperation::MoveImmediate32 {
                destination: ShaderRegister::new(1),
                bits: right,
                scalar_type: ShaderScalarType::Unsigned32,
            },
        ];
        operations.extend(decoded.operations);
        operations.extend([
            ShaderOperation::StoreOutput {
                sources: vec![ShaderRegister::new(0)].into(),
                location: ShaderIoLocation::Color(0),
                first_component: 0,
                scalar_type: ShaderScalarType::Unsigned32,
            },
            ShaderOperation::Exit,
        ]);
        let shader = VerifiedShaderIr::verify(ShaderIr::new(
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
            resources,
            operations
                .into_iter()
                .enumerate()
                .map(|(i, operation)| {
                    ShaderInstruction::new(
                        ShaderSourceLocation::new(i as u32 * 8),
                        ShaderPredicate::Always,
                        operation,
                    )
                })
                .collect::<Vec<_>>(),
        ))
        .unwrap();
        nixe_gpu::evaluate_shader_ir(
            &shader,
            &nixe_gpu::ShaderEvaluationInputs::default().with_constant_buffer_bits(2, 12, right),
            32,
        )
        .unwrap()
        .output_bits(ShaderIoLocation::Color(0), 0)
        .unwrap()
    }

    #[test]
    fn iadd_forms_preserve_wrapping_addition_and_signed_immediates() {
        for left in [0_u32, 1, 0x7fff_ffff, 0x8000_0000, u32::MAX] {
            for right in [0_u32, 1, 0x7fff_ffff, 0x8000_0000, u32::MAX] {
                for word in [0x5c10_0000_0017_0000, 0x4c10_0000_0037_0000 | (2 << 34)] {
                    assert_eq!(evaluate_iadd(word, left, right), left.wrapping_add(right));
                }
            }
            for (word, right) in [
                (0x3810_0000_0017_0000, 1),
                (0x3810_007f_fff7_0000, 0x0007_ffff),
                (0x3910_007f_fff7_0000, u32::MAX),
                (0x3910_0000_0007_0000, 0xfff8_0000),
            ] {
                assert_eq!(evaluate_iadd(word, left, 0), left.wrapping_add(right));
            }
        }
        assert_eq!(evaluate_iadd(0x3810_0000_0017_ff00, 42, 0), 1);
        assert_eq!(evaluate_iadd(0x5c10_0000_0ff7_0000, 42, 0), 42);
        assert_eq!(evaluate_iadd(0x3810_0000_0017_00ff, 42, 0), 42);
    }

    #[test]
    fn iadd_rejects_unrepresented_modifiers_and_invalid_registers() {
        let captured = 0x3810_0000_0017_0101_u64;
        for (modifier, detail) in [
            (1 << 47, "IADD condition-code write"),
            (1 << 43, "IADD extended carry input"),
            (1 << 50, "IADD signed saturation"),
            (1 << 48, "IADD negated operands"),
            (1 << 49, "IADD negated operands"),
        ] {
            assert!(is_add(captured | modifier));
            assert!(
                matches!(decode_add(STAGE, 0x48, captured | modifier, 4, &mut 4),
                Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail { detail: actual, instruction_offset: 0x48, .. }) if actual == detail)
            );
        }
        for word in [
            captured | (1 << 39),
            captured | (1 << 46),
            captured | 4,
            captured | (4 << 8),
            0x5c10_0000_0047_0000,
            0x5c10_0000_1017_0000,
        ] {
            assert!(
                decode_add(STAGE, 8, word, 4, &mut 4).is_err(),
                "{word:016x}"
            );
        }
        assert!(!is_add(0x5c18_0000_0017_0000)); // ISCADD is a distinct instruction.
    }

    #[test]
    fn lop_forms_complements_and_pass_b_match_integer_semantics() {
        let left = 0xa5a5_1234_u32;
        for (encoding, right) in [
            (0x5c47_0000_0017_0000_u64, 0x8000_2468_u32),
            (0x4c47_0000_0037_0000 | (2 << 34), 0x8000_2468),
            (0x3847_0000_0027_0000, 2),
            (0x3947_007f_fff7_0000, u32::MAX),
            (0x3947_0000_0007_0000, 0xfff8_0000),
        ] {
            assert!(is_bitwise(encoding));
            for selector in 0..4 {
                for inversion in 0..4 {
                    let a = if inversion & 1 != 0 { !left } else { left };
                    let b = if inversion & 2 != 0 { !right } else { right };
                    let expected = match selector {
                        0 => a & b,
                        1 => a | b,
                        2 => a ^ b,
                        _ => b,
                    };
                    let word = encoding | (selector << 41) | (inversion << 39);
                    assert_eq!(evaluate_lop(word, left, right), expected, "{word:016x}");
                }
            }
        }
        // RZ is zero on reads and discards writes; PASS_B does not require A.
        assert_eq!(evaluate_lop(0x3847_0000_0027_ff00, left, 0), 0);
        assert_eq!(evaluate_lop(0x5c47_0000_0ff7_0000, left, 0), 0);
        assert_eq!(evaluate_lop(0x3847_0000_0027_00ff, left, 0), left);
        assert_eq!(evaluate_lop(0x3847_0600_0027_fe00, left, 0), 2);
    }

    #[test]
    fn lop_rejects_flags_predicate_outputs_reserved_bits_and_bad_registers() {
        let captured = 0x3847_0000_0027_0000_u64;
        for (word, detail) in [
            (captured | (1 << 47), "LOP condition-code write"),
            (captured | (1 << 43), "LOP extended condition-code input"),
            (captured & !(7 << 48), "LOP predicate output"),
            (captured | (1 << 44), "LOP predicate output"),
        ] {
            assert!(matches!(decode_bitwise(STAGE, 8, word, 4, &mut 4),
                Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail { detail: actual, .. }) if actual == detail));
        }
        for word in [
            captured | (1 << 46),
            0x5c47_0000_1017_0000,
            captured | 4,
            captured | (4 << 8),
        ] {
            assert!(
                decode_bitwise(STAGE, 8, word, 4, &mut 4).is_err(),
                "{word:016x}"
            );
        }
    }

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
