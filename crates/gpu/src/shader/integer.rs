//! Integer operations shared by shader output and the test oracle.
use super::*;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ShaderBitwiseOperation {
    And,
    Or,
    Xor,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ShaderIntegerComparison {
    False,
    Less,
    Equal,
    LessOrEqual,
    Greater,
    NotEqual,
    GreaterOrEqual,
    True,
}

pub(super) fn compare(
    left: u32,
    right: u32,
    signed: bool,
    comparison: ShaderIntegerComparison,
) -> bool {
    let order = if signed {
        (left as i32).cmp(&(right as i32))
    } else {
        left.cmp(&right)
    };
    use ShaderIntegerComparison::*;
    match comparison {
        False => false,
        Less => order.is_lt(),
        Equal => order.is_eq(),
        LessOrEqual => !order.is_gt(),
        Greater => order.is_gt(),
        NotEqual => !order.is_eq(),
        GreaterOrEqual => !order.is_lt(),
        True => true,
    }
}

pub(super) fn combine(compared: bool, accumulated: bool, op: ShaderPredicateSetOperation) -> bool {
    match op {
        ShaderPredicateSetOperation::And => compared && accumulated,
        ShaderPredicateSetOperation::Or => compared || accumulated,
        ShaderPredicateSetOperation::Xor => compared ^ accumulated,
    }
}

pub(super) fn emit_wgsl(source: &mut String, operation: &ShaderOperation) {
    let ShaderOperation::SetPredicateInteger32 {
        destinations,
        left,
        right,
        signed,
        comparison,
        accumulator,
        set_operation,
    } = operation
    else {
        unreachable!()
    };
    let scalar = if *signed { "i32" } else { "u32" };
    let left = format!("bitcast<{scalar}>(registers[{}])", left.index());
    let right = format!("bitcast<{scalar}>(registers[{}])", right.index());
    use ShaderIntegerComparison::*;
    let compared = match comparison {
        False => "false".to_owned(),
        True => "true".to_owned(),
        comparison => {
            let op = match comparison {
                Less => "<",
                Equal => "==",
                LessOrEqual => "<=",
                Greater => ">",
                NotEqual => "!=",
                GreaterOrEqual => ">=",
                _ => unreachable!(),
            };
            format!("{left} {op} {right}")
        }
    };
    // Scope each expansion and snapshot both sources before any predicate write.
    source.push_str(&format!(
        "  {{\n    let compared = {compared};\n    let accumulated = {};\n",
        wgsl_predicate_expression(*accumulator)
    ));
    let op = match set_operation {
        ShaderPredicateSetOperation::And => "&&",
        ShaderPredicateSetOperation::Or => "||",
        ShaderPredicateSetOperation::Xor => "!=",
    };
    for (index, destination) in destinations.iter().enumerate() {
        if let Some(destination) = destination {
            let invert = if index == 0 { "" } else { "!" };
            source.push_str(&format!(
                "    predicates[{destination}] = ({invert}compared) {op} accumulated;\n"
            ));
        }
    }
    source.push_str("  }\n");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logical_right_shift_is_unsigned_alias_safe_and_checks_unwrapped_amounts() {
        for value in [0, 1, 0x8000_0000, u32::MAX] {
            for amount in [0, 1, 31, 32, 33, u32::MAX] {
                for wrap in [false, true] {
                    let shader = program(vec![
                        instruction(ShaderPredicate::Always, immediate(0, value)),
                        instruction(ShaderPredicate::Always, immediate(1, amount)),
                        instruction(
                            ShaderPredicate::Always,
                            ShaderOperation::ShiftRightLogical32 {
                                destination: ShaderRegister::new(0),
                                value: ShaderRegister::new(0),
                                amount: ShaderRegister::new(1),
                                wrap,
                            },
                        ),
                        instruction(ShaderPredicate::Always, store(0, 0)),
                        instruction(ShaderPredicate::Always, ShaderOperation::Exit),
                    ])
                    .unwrap();
                    let result =
                        evaluate_shader_ir(&shader, &ShaderEvaluationInputs::default(), 16)
                            .unwrap();
                    let expected = if wrap {
                        value >> (amount & 31)
                    } else {
                        value.checked_shr(amount).unwrap_or(0)
                    };
                    assert_eq!(
                        result.output_bits(ShaderIoLocation::Position, 0),
                        Some(expected)
                    );
                    let wgsl = lower_shader_ir_to_wgsl(&shader).unwrap();
                    let module = naga::front::wgsl::parse_str(wgsl.source()).unwrap();
                    naga::valid::Validator::new(
                        naga::valid::ValidationFlags::all(),
                        naga::valid::Capabilities::all(),
                    )
                    .validate(&module)
                    .unwrap();
                }
            }
        }
    }

    #[test]
    fn add_carry_handles_two_overflows_aliasing_and_false_predicates() {
        let r = ShaderRegister::new;
        let always = ShaderPredicate::Always;
        for (left, right, carry) in [
            (0_u32, 0_u32, 0_u32),
            (u32::MAX, 1, 0),
            (u32::MAX, 0, 1),
            (u32::MAX, u32::MAX, 1),
            (0x8000_0000, 0x8000_0000, 0),
            (17, 25, 3),
        ] {
            for enabled in [false, true] {
                let shader = program(vec![
                    instruction(always, immediate(0, left)),
                    instruction(always, immediate(1, right)),
                    instruction(always, immediate(2, carry)),
                    instruction(
                        always,
                        set(if enabled {
                            ShaderIntegerComparison::True
                        } else {
                            ShaderIntegerComparison::False
                        }),
                    ),
                    instruction(
                        predicate(false),
                        ShaderOperation::AddCarry32 {
                            destination: r(0),
                            carry_out: r(2),
                            left: r(0),
                            right: r(1),
                            carry_in: Some(r(2)),
                        },
                    ),
                    instruction(always, store(0, 0)),
                    instruction(always, store(2, 1)),
                    instruction(always, ShaderOperation::Exit),
                ])
                .unwrap();
                let result =
                    evaluate_shader_ir(&shader, &ShaderEvaluationInputs::default(), 32).unwrap();
                let total = u64::from(left) + u64::from(right) + u64::from(carry & 1);
                assert_eq!(
                    result.output_bits(ShaderIoLocation::Position, 0),
                    Some(if enabled { total as u32 } else { left })
                );
                assert_eq!(
                    result.output_bits(ShaderIoLocation::Position, 1),
                    Some(if enabled { (total >> 32) as u32 } else { carry })
                );
                let wgsl = lower_shader_ir_to_wgsl(&shader).unwrap();
                let module = naga::front::wgsl::parse_str(wgsl.source()).unwrap();
                naga::valid::Validator::new(
                    naga::valid::ValidationFlags::all(),
                    naga::valid::Capabilities::all(),
                )
                .validate(&module)
                .unwrap();
            }
        }
    }

    #[test]
    fn add_carry_requires_distinct_outputs_and_a_defined_carry_input() {
        let r = ShaderRegister::new;
        for (carry_out, carry_in) in [(r(0), None), (r(2), Some(r(2)))] {
            assert!(
                program(vec![
                    instruction(ShaderPredicate::Always, immediate(0, 0)),
                    instruction(
                        ShaderPredicate::Always,
                        ShaderOperation::AddCarry32 {
                            destination: r(0),
                            carry_out,
                            left: r(0),
                            right: r(0),
                            carry_in,
                        }
                    ),
                    instruction(ShaderPredicate::Always, ShaderOperation::Exit),
                ])
                .is_err()
            );
        }
    }

    fn instruction(predicate: ShaderPredicate, operation: ShaderOperation) -> ShaderInstruction {
        ShaderInstruction::new(ShaderSourceLocation::new(8), predicate, operation)
    }

    #[test]
    fn bitwise_operations_preserve_all_bits_aliasing_and_predication() {
        let values = [
            0,
            1,
            2,
            3,
            0x8000_0000,
            0xffff_ffff,
            0xaaaa_5555,
            0x5555_aaaa,
        ];
        for operation in [
            ShaderBitwiseOperation::And,
            ShaderBitwiseOperation::Or,
            ShaderBitwiseOperation::Xor,
        ] {
            for left in values {
                for right in values {
                    for enabled in [false, true] {
                        let always = ShaderPredicate::Always;
                        let shader = program(vec![
                            instruction(always, immediate(0, left)),
                            instruction(always, immediate(1, right)),
                            instruction(
                                always,
                                set(if enabled {
                                    ShaderIntegerComparison::True
                                } else {
                                    ShaderIntegerComparison::False
                                }),
                            ),
                            instruction(
                                predicate(false),
                                ShaderOperation::Bitwise32 {
                                    destination: ShaderRegister::new(0),
                                    left: ShaderRegister::new(0),
                                    right: ShaderRegister::new(1),
                                    operation,
                                },
                            ),
                            instruction(always, store(0, 0)),
                            instruction(always, ShaderOperation::Exit),
                        ])
                        .unwrap();
                        let expected = if enabled {
                            match operation {
                                ShaderBitwiseOperation::And => left & right,
                                ShaderBitwiseOperation::Or => left | right,
                                ShaderBitwiseOperation::Xor => left ^ right,
                            }
                        } else {
                            left
                        };
                        assert_eq!(
                            evaluate_shader_ir(&shader, &ShaderEvaluationInputs::default(), 16)
                                .unwrap()
                                .output_bits(ShaderIoLocation::Position, 0),
                            Some(expected)
                        );
                        let wgsl = lower_shader_ir_to_wgsl(&shader).unwrap();
                        let module = naga::front::wgsl::parse_str(wgsl.source()).unwrap();
                        naga::valid::Validator::new(
                            naga::valid::ValidationFlags::all(),
                            naga::valid::Capabilities::all(),
                        )
                        .validate(&module)
                        .unwrap();
                    }
                }
            }
        }
    }

    fn immediate(register: u16, bits: u32) -> ShaderOperation {
        ShaderOperation::MoveImmediate32 {
            destination: ShaderRegister::new(register),
            bits,
            scalar_type: ShaderScalarType::Unsigned32,
        }
    }

    fn predicate(inverted: bool) -> ShaderPredicate {
        ShaderPredicate::Register {
            register: 0,
            inverted,
        }
    }

    fn set(comparison: ShaderIntegerComparison) -> ShaderOperation {
        ShaderOperation::SetPredicateInteger32 {
            destinations: [Some(0), None],
            left: ShaderRegister::new(0),
            right: ShaderRegister::new(1),
            signed: false,
            comparison,
            accumulator: ShaderPredicate::Always,
            set_operation: ShaderPredicateSetOperation::And,
        }
    }

    fn program(
        mut instructions: Vec<ShaderInstruction>,
    ) -> Result<VerifiedShaderIr, ShaderVerificationError> {
        for (index, instruction) in instructions.iter_mut().enumerate() {
            instruction.source = ShaderSourceLocation::new((index as u32 + 1) * 8);
        }
        VerifiedShaderIr::verify(ShaderIr::new(
            ShaderStage::Vertex,
            vec![],
            (0..2)
                .map(|component| {
                    ShaderInterfaceElement::new(
                        ShaderIoLocation::Position,
                        component,
                        ShaderScalarType::Float32,
                        None,
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>(),
            vec![],
            instructions,
        ))
    }

    fn store(register: u16, component: u8) -> ShaderOperation {
        ShaderOperation::StoreOutput {
            sources: vec![ShaderRegister::new(register)].into_boxed_slice(),
            location: ShaderIoLocation::Position,
            first_component: component,
            scalar_type: ShaderScalarType::Float32,
        }
    }

    #[test]
    fn integer_predicates_cover_signed_edges_and_atomic_accumulator_aliases() {
        use ShaderIntegerComparison::*;
        for (left, right) in [
            (0, 0),
            (0, 1),
            (1, 0),
            (u32::MAX, 0),
            (0x8000_0000, 0x7fff_ffff),
            (u32::MAX, 0x8000_0000),
        ] {
            for signed in [false, true] {
                let (a, b) = if signed {
                    (i64::from(left as i32), i64::from(right as i32))
                } else {
                    (i64::from(left), i64::from(right))
                };
                for (comparison, expected) in [
                    (False, false),
                    (Less, a < b),
                    (Equal, a == b),
                    (LessOrEqual, a <= b),
                    (Greater, a > b),
                    (NotEqual, a != b),
                    (GreaterOrEqual, a >= b),
                    (True, true),
                ] {
                    for accumulated in [false, true] {
                        for op in [
                            ShaderPredicateSetOperation::And,
                            ShaderPredicateSetOperation::Or,
                            ShaderPredicateSetOperation::Xor,
                        ] {
                            let mut instructions = vec![
                                instruction(ShaderPredicate::Always, immediate(0, left)),
                                instruction(ShaderPredicate::Always, immediate(1, right)),
                                instruction(
                                    ShaderPredicate::Always,
                                    set(if accumulated { True } else { False }),
                                ),
                            ];
                            instructions.push(instruction(
                                ShaderPredicate::Always,
                                ShaderOperation::SetPredicateInteger32 {
                                    destinations: [Some(0), Some(1)],
                                    left: ShaderRegister::new(0),
                                    right: ShaderRegister::new(1),
                                    signed,
                                    comparison,
                                    accumulator: predicate(false),
                                    set_operation: op,
                                },
                            ));
                            for (register, predicate) in [(2, 0), (3, 1)] {
                                instructions.push(instruction(
                                    ShaderPredicate::Always,
                                    immediate(register, 0),
                                ));
                                instructions.push(instruction(
                                    ShaderPredicate::Register {
                                        register: predicate,
                                        inverted: false,
                                    },
                                    immediate(register, 1),
                                ));
                                instructions.push(instruction(
                                    ShaderPredicate::Always,
                                    store(register, predicate),
                                ));
                            }
                            instructions
                                .push(instruction(ShaderPredicate::Always, ShaderOperation::Exit));
                            let ir = program(instructions).unwrap();
                            let result =
                                evaluate_shader_ir(&ir, &ShaderEvaluationInputs::default(), 64)
                                    .unwrap();
                            for (component, comparison_result) in [(0, expected), (1, !expected)] {
                                let expected = match op {
                                    ShaderPredicateSetOperation::And => {
                                        comparison_result && accumulated
                                    }
                                    ShaderPredicateSetOperation::Or => {
                                        comparison_result || accumulated
                                    }
                                    ShaderPredicateSetOperation::Xor => {
                                        comparison_result != accumulated
                                    }
                                };
                                assert_eq!(
                                    result.output_bits(ShaderIoLocation::Position, component),
                                    Some(u32::from(expected))
                                );
                            }
                            let wgsl = lower_shader_ir_to_wgsl(&ir).unwrap();
                            let module = naga::front::wgsl::parse_str(wgsl.source()).unwrap();
                            naga::valid::Validator::new(
                                naga::valid::ValidationFlags::all(),
                                naga::valid::Capabilities::empty(),
                            )
                            .validate(&module)
                            .unwrap();
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn guarded_definitions_require_the_same_unchanged_predicate() {
        let prefix = vec![
            instruction(ShaderPredicate::Always, immediate(0, 0)),
            instruction(ShaderPredicate::Always, immediate(1, 1)),
            instruction(ShaderPredicate::Always, set(ShaderIntegerComparison::Less)),
            instruction(predicate(false), immediate(2, 42)),
        ];
        for guard in [predicate(false), predicate(true), ShaderPredicate::Always] {
            let mut instructions = prefix.clone();
            instructions.push(instruction(guard, store(2, 0)));
            instructions.push(instruction(ShaderPredicate::Always, ShaderOperation::Exit));
            assert_eq!(program(instructions).is_ok(), guard == predicate(false));
        }
        let mut instructions = prefix;
        instructions.push(instruction(
            ShaderPredicate::Always,
            set(ShaderIntegerComparison::Greater),
        ));
        instructions.push(instruction(predicate(false), store(2, 0)));
        instructions.push(instruction(ShaderPredicate::Always, ShaderOperation::Exit));
        assert!(matches!(
            program(instructions),
            Err(ShaderVerificationError::UndefinedRegister { .. })
        ));
    }

    #[test]
    fn guarded_facts_at_joins_are_intersections_not_unions() {
        for other_path in [predicate(false), predicate(true), ShaderPredicate::Always] {
            let mut selector = set(ShaderIntegerComparison::True);
            if let ShaderOperation::SetPredicateInteger32 { destinations, .. } = &mut selector {
                *destinations = [Some(1), None];
            }
            let instructions = vec![
                instruction(ShaderPredicate::Always, immediate(0, 0)),
                instruction(ShaderPredicate::Always, immediate(1, 1)),
                instruction(ShaderPredicate::Always, set(ShaderIntegerComparison::Less)),
                instruction(ShaderPredicate::Always, selector),
                instruction(
                    ShaderPredicate::Register {
                        register: 1,
                        inverted: false,
                    },
                    ShaderOperation::Branch {
                        target: ShaderSourceLocation::new(64),
                    },
                ),
                instruction(predicate(false), immediate(2, 42)),
                instruction(
                    ShaderPredicate::Always,
                    ShaderOperation::Branch {
                        target: ShaderSourceLocation::new(72),
                    },
                ),
                instruction(other_path, immediate(2, 43)),
                instruction(predicate(false), store(2, 0)),
                instruction(ShaderPredicate::Always, ShaderOperation::Exit),
            ];
            assert_eq!(program(instructions).is_ok(), other_path != predicate(true));
        }
    }

    #[test]
    fn predicate_alias_and_discard_contract() {
        for destinations in [
            [Some(0), Some(0)],
            [Some(7), None],
            [None, None],
            [None, Some(0)],
        ] {
            let mut operation = set(ShaderIntegerComparison::True);
            if let ShaderOperation::SetPredicateInteger32 {
                destinations: target,
                ..
            } = &mut operation
            {
                *target = destinations;
            }
            let ir = program(vec![
                instruction(ShaderPredicate::Always, immediate(0, 0)),
                instruction(ShaderPredicate::Always, immediate(1, 0)),
                instruction(ShaderPredicate::Always, operation),
                instruction(ShaderPredicate::Always, ShaderOperation::Exit),
            ]);
            assert_eq!(ir.is_ok(), destinations[0].is_none());
        }
    }
}
