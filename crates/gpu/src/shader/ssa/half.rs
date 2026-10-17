//! Preserve half rounding boundaries while removing FP32 transport. A widening
//! of a quiet binary16 is exact, including payload/sign; an arbitrary F32 value
//! must still round. Never use the inverse identity for F32 -> F16 -> F32.
//! https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpFConvert
use super::*;

pub(super) fn simplify(function: &mut Function) -> bool {
    if !function.values.iter().any(|v| v.ty == Type::F16) {
        return false;
    }
    let mut defs = vec![None; function.values.len()];
    let mut quiet = vec![false; function.values.len()];
    for block in &function.blocks {
        for inst in &block.instructions {
            if inst.results.len() == 1 {
                defs[inst.results[0].index()] = Some(inst);
            }
        }
    }
    // Visit only users of a newly proven value. A loop phi never proves its
    // own quieting; no repeated full-function scan for long transport chains.
    let mut users = vec![Vec::new(); defs.len()];
    for (id, def) in defs.iter().enumerate() {
        if function.values[id].ty == Type::F16
            && let Some(inst) = def
        {
            for arg in &inst.arguments {
                users[arg.index()].push(id);
            }
        }
    }
    let mut pending: std::collections::VecDeque<_> = (0..defs.len())
        .filter(|&id| function.values[id].ty == Type::F16)
        .collect();
    while let Some(id) = pending.pop_front() {
        if quiet[id] {
            continue;
        }
        let Some(inst) = defs[id] else {
            continue;
        };
        let proven = match inst.op {
            Op::UnpackHalf
            | Op::HalfFma { .. }
            | Op::Convert { to: Type::F16, .. }
            | Op::Binary { ty: Type::F16, .. } => true,
            Op::Constant(c) => c.bits & 0x7fff <= 0x7c00 || c.bits & 0x200 != 0,
            Op::Copy(_) | Op::Unary { .. } => quiet[inst.arguments[0].index()],
            Op::Select(_) => quiet[inst.arguments[1].index()] && quiet[inst.arguments[2].index()],
            _ => false,
        };
        if proven {
            quiet[id] = true;
            pending.extend(users[id].iter().copied());
        }
    }
    let widened = |value: ValueId| -> Option<ValueId> {
        let inst = defs[value.index()]?;
        if matches!(
            inst.op,
            Op::Convert {
                from: Type::F16,
                to: Type::F32,
                flush_denormals: false,
                ..
            }
        ) && quiet[inst.arguments[0].index()]
        {
            Some(inst.arguments[0])
        } else {
            None
        }
    };
    let mut edits = Vec::new();
    for (block_id, block) in function.blocks.iter().enumerate() {
        for (index, inst) in block.instructions.iter().enumerate() {
            let replacement = match inst.op {
                Op::UnpackHalf => defs[inst.arguments[0].index()]
                    .filter(|d| d.op == Op::PackHalf && quiet[d.arguments[0].index()])
                    .map(|d| (Op::Copy(Type::F16), vec![d.arguments[0]], false)),
                Op::Unary {
                    op: op @ (Unary::Negate | Unary::Absolute),
                    ty: Type::F32,
                } => widened(inst.arguments[0])
                    .map(|v| (Op::Unary { op, ty: Type::F16 }, vec![v], true)),
                Op::Convert {
                    from: Type::F32,
                    to: Type::F16,
                    rounding: ShaderRoundingMode::NearestEven,
                    flush_denormals: false,
                } => {
                    if let Some(value) = widened(inst.arguments[0]) {
                        Some((Op::Copy(Type::F16), vec![value], false))
                    } else if let Some(source) = defs[inst.arguments[0].index()] {
                        if let Op::Binary {
                            op: op @ (Binary::Add | Binary::Multiply),
                            ty: Type::F32,
                            float_control,
                        } = source.op
                        {
                            if float_control.rounding == ShaderRoundingMode::NearestEven
                                && !float_control.denormals_are_zero
                                && !float_control.flush_denormals_to_zero
                                && float_control.nan_mode == crate::ShaderNanMode::Propagate
                            {
                                widened(source.arguments[0])
                                    .zip(widened(source.arguments[1]))
                                    .map(|(a, b)| {
                                        (
                                            Op::Binary {
                                                op,
                                                ty: Type::F16,
                                                float_control,
                                            },
                                            vec![a, b],
                                            false,
                                        )
                                    })
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                }
                _ => None,
            };
            if let Some((op, args, widen)) = replacement {
                edits.push((block_id, index, op, args, widen));
            }
        }
    }
    drop(defs);
    let changed = !edits.is_empty();
    // Reverse order permits insertion without invalidating the recorded indices.
    for (block, index, op, args, widen) in edits.into_iter().rev() {
        let old = &mut function.blocks[block].instructions[index];
        if widen {
            let result = ValueId(function.values.len() as u32);
            function.values.push(Value {
                ty: Type::F16,
                definition: Definition::Instruction {
                    block: BlockId(block as u32),
                    index: index as u32,
                    result: 0,
                },
            });
            let narrow = Instruction {
                op,
                arguments: args.into_boxed_slice(),
                results: vec![result].into_boxed_slice(),
                source: old.source,
            };
            old.op = Op::Convert {
                from: Type::F16,
                to: Type::F32,
                rounding: ShaderRoundingMode::NearestEven,
                flush_denormals: false,
            };
            old.arguments = vec![result].into_boxed_slice();
            function.blocks[block].instructions.insert(index, narrow);
        } else {
            old.op = op;
            old.arguments = args.into_boxed_slice();
        }
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;
    fn convert(
        b: &mut FunctionBuilder,
        block: BlockId,
        v: Operand,
        from: Type,
        to: Type,
    ) -> Operand {
        b.emit(
            block,
            ShaderSourceLocation::new(8),
            Op::Convert {
                from,
                to,
                rounding: ShaderRoundingMode::NearestEven,
                flush_denormals: false,
            },
            vec![v],
        )[0]
        .into()
    }
    fn finish(b: &mut FunctionBuilder, block: BlockId, v: Operand) {
        b.emit(
            block,
            ShaderSourceLocation::new(24),
            Op::Output {
                location: ShaderIoLocation::Color(0),
                component: 0,
                ty: Type::U32,
            },
            vec![v],
        );
        b.terminate(block, ShaderSourceLocation::new(32), Terminator::Return);
    }
    #[test]
    fn half_fma_and_binary_keep_native_values_across_word_transport_and_sign_modifiers() {
        let mut b = FunctionBuilder::new(ShaderStage::Fragment);
        let block = b.entry();
        let pc = ShaderSourceLocation::new(8);
        let word = b.emit(
            block,
            pc,
            Op::Parameter {
                index: 0,
                ty: Type::U32,
            },
            vec![],
        )[0];
        let half = b.emit(block, pc, Op::UnpackHalf, vec![word.into()])[0];
        let wide = convert(&mut b, block, half.into(), Type::F16, Type::F32);
        let neg = b.emit(
            block,
            pc,
            Op::Unary {
                op: Unary::Negate,
                ty: Type::F32,
            },
            vec![wide],
        )[0];
        let narrow = convert(&mut b, block, neg.into(), Type::F32, Type::F16);
        let fused = b.emit(block, pc, Op::HalfFma { saturate: false }, vec![narrow; 3])[0];
        let word = b.emit(block, pc, Op::PackHalf, vec![fused.into()])[0];
        let half = b.emit(block, pc, Op::UnpackHalf, vec![word.into()])[0];
        let wide = convert(&mut b, block, half.into(), Type::F16, Type::F32);
        let sum = b.emit(
            block,
            pc,
            Op::Binary {
                op: Binary::Add,
                ty: Type::F32,
                float_control: ShaderFloatControl::PRECISE,
            },
            vec![wide, wide],
        )[0];
        let narrow = convert(&mut b, block, sum.into(), Type::F32, Type::F16);
        let word = b.emit(block, pc, Op::PackHalf, vec![narrow])[0];
        finish(&mut b, block, word.into());
        let mut f = b.finish().unwrap();
        f.optimize().unwrap();
        f.verify().unwrap();
        let ops: Vec<_> = f
            .blocks
            .iter()
            .flat_map(|b| &b.instructions)
            .map(|i| &i.op)
            .collect();
        assert!(!ops.iter().any(|op| matches!(
            op,
            Op::Convert { .. } | Op::Unary { ty: Type::F32, .. } | Op::Binary { ty: Type::F32, .. }
        )));
        assert!(ops.contains(&&Op::Unary {
            op: Unary::Negate,
            ty: Type::F16
        }));
        assert!(ops.contains(&&Op::Binary {
            op: Binary::Add,
            ty: Type::F16,
            float_control: ShaderFloatControl::PRECISE
        }));
        assert_eq!(
            ops.iter().filter(|op| matches!(op, Op::UnpackHalf)).count(),
            1
        );
        assert_eq!(
            ops.iter().filter(|op| matches!(op, Op::PackHalf)).count(),
            1
        );
    }
    #[test]
    fn narrowing_arbitrary_fp32_and_quieting_signaling_half_are_not_inverse_identities() {
        for signaling in [false, true] {
            let mut b = FunctionBuilder::new(ShaderStage::Fragment);
            let block = b.entry();
            let pc = ShaderSourceLocation::new(8);
            let (input, first, second) = if signaling {
                (
                    b.emit(
                        block,
                        pc,
                        Op::Constant(Constant {
                            ty: Type::F16,
                            bits: 0x7c01,
                        }),
                        vec![],
                    )[0],
                    Type::F16,
                    Type::F32,
                )
            } else {
                (
                    b.emit(
                        block,
                        pc,
                        Op::Parameter {
                            index: 0,
                            ty: Type::F32,
                        },
                        vec![],
                    )[0],
                    Type::F32,
                    Type::F16,
                )
            };
            let v = convert(&mut b, block, input.into(), first, second);
            let v = convert(&mut b, block, v, second, first);
            let word = if signaling {
                b.emit(block, pc, Op::PackHalf, vec![v])[0]
            } else {
                b.emit(
                    block,
                    pc,
                    Op::Bitcast {
                        from: Type::F32,
                        to: Type::U32,
                    },
                    vec![v],
                )[0]
            };
            finish(&mut b, block, word.into());
            let mut f = b.finish().unwrap();
            f.optimize().unwrap();
            assert_eq!(
                f.blocks
                    .iter()
                    .flat_map(|b| &b.instructions)
                    .filter(|i| matches!(i.op, Op::Convert { .. }))
                    .count(),
                2,
                "signaling={signaling}"
            );
        }
    }
}
