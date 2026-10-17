//! Choose the shader environment from semantic operations, never GPU identity.
//! Vulkan FTZ guarantees result flushing, but operand DAZ is optional and tiny
//! results may flush before rounding. Keep explicit DAZ and exact RTE multiply/
//! FMA underflow repair even when the entry point enables native FTZ.
//! https://docs.vulkan.org/spec/latest/appendices/spirvenv.html#spirvenv-precision
use super::*;
use crate::shader::spirv::math::Denorm32;

pub(super) fn configure(math: &mut Math, function: &Function) {
    let caps = math.options.float32;
    let mut flush = math.options.vertex_position_transform.is_none()
        && !math.options.depth_clip_negative_one_to_one;
    let mut used = false;
    let mut rtz_multiply_only = true;
    for inst in function.blocks.iter().flat_map(|b| &b.instructions) {
        match inst.op {
            Op::Binary {
                ty: Type::F32,
                float_control,
                ..
            }
            | Op::Fma {
                ty: Type::F32,
                float_control,
            } => {
                used = true;
                flush &= float_control.denormals_are_zero && float_control.flush_denormals_to_zero;
                rtz_multiply_only &= matches!(
                    inst.op,
                    Op::Binary {
                        op: Binary::Multiply,
                        ..
                    }
                ) && float_control.rounding == ShaderRoundingMode::TowardZero;
                // Non-RTE operations otherwise use their existing exact bit
                // lowering, so a mixed shader retains the RTE environment.
            }
            Op::Math { .. }
            | Op::FloatMinMax { .. }
            | Op::QuadSwizzleAdd { .. }
            | Op::Interpolate { .. }
            | Op::Quad(_)
            | Op::TextureSample { .. } => {
                flush = false;
                rtz_multiply_only = false;
            }
            Op::FloatCompare {
                flush_denormals, ..
            } => {
                flush &= flush_denormals;
                rtz_multiply_only = false;
            }
            Op::Convert {
                from: Type::F16,
                to: Type::F32,
                ..
            }
            | Op::Convert {
                from: Type::F32,
                to: Type::F16,
                ..
            } => {
                rtz_multiply_only = false;
            }
            Op::Convert { .. } | Op::HalfFma { .. } | Op::Binary { ty: Type::F16, .. } => {
                flush = false;
                rtz_multiply_only = false;
            }
            _ => {}
        }
    }
    if !used || !flush {
        return;
    }
    // Preserve32 avoids the repair code when available. On hosts lacking it,
    // native FTZ can remove result masks and cancellation repair for addition.
    if !caps.denorm_preserve && caps.denorm_flush_to_zero && caps.independent_denormals {
        math.denorm32 = Denorm32::Flush;
    }
    if rtz_multiply_only
        && caps.rounding_mode_rtz
        && caps.independent_rounding
        && (caps.denorm_preserve || math.denorm32 == Denorm32::Flush)
    {
        math.rounding32 = ShaderRoundingMode::TowardZero;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn function(rounding: ShaderRoundingMode, mixed: bool) -> Function {
        let mut b = FunctionBuilder::new(ShaderStage::Fragment);
        let block = b.entry();
        let pc = ShaderSourceLocation::new(0);
        let x = b.emit(
            block,
            pc,
            Op::Parameter {
                index: 0,
                ty: Type::F32,
            },
            vec![],
        )[0];
        let value = b.emit(
            block,
            pc,
            Op::Binary {
                op: Binary::Multiply,
                ty: Type::F32,
                float_control: ShaderFloatControl::new(
                    rounding,
                    crate::ShaderNanMode::Propagate,
                    true,
                    true,
                    false,
                ),
            },
            vec![x.into(), x.into()],
        )[0];
        let value = if mixed {
            b.emit(
                block,
                pc,
                Op::Binary {
                    op: Binary::Add,
                    ty: Type::F32,
                    float_control: ShaderFloatControl::PRECISE,
                },
                vec![value.into(), x.into()],
            )[0]
        } else {
            value
        };
        b.emit(
            block,
            pc,
            Op::Output {
                location: ShaderIoLocation::Color(0),
                component: 0,
                ty: Type::F32,
            },
            vec![value.into()],
        );
        b.terminate(block, pc, Terminator::Return);
        let mut f = b.finish().unwrap();
        f.outputs.push(
            ShaderInterfaceElement::new(
                ShaderIoLocation::Color(0),
                0,
                crate::ShaderScalarType::Float32,
                None,
            )
            .unwrap(),
        );
        f
    }
    #[test]
    fn modes_require_whole_shader_agreement_and_enabled_independent_guarantees() {
        let mut opts = super::super::tests::options();
        opts.shader.float32.denorm_preserve = false;
        opts.shader.float32.denorm_flush_to_zero = true;
        opts.shader.float32.rounding_mode_rtz = true;
        for independent in [false, true] {
            opts.shader.float32.independent_denormals = independent;
            opts.shader.float32.independent_rounding = independent;
            for mixed in [false, true] {
                let f = function(ShaderRoundingMode::TowardZero, mixed);
                let mut math = Math::new(opts.shader);
                configure(&mut math, &f);
                assert_eq!(
                    math.denorm32,
                    if independent && !mixed {
                        Denorm32::Flush
                    } else {
                        Denorm32::Unspecified
                    }
                );
                assert_eq!(
                    math.rounding32,
                    if independent && !mixed {
                        ShaderRoundingMode::TowardZero
                    } else {
                        ShaderRoundingMode::NearestEven
                    }
                );
                let module = emit(&f, opts).unwrap();
                let parsed = rspirv::dr::load_words(module.words).unwrap();
                let modes: Vec<_> = parsed
                    .execution_modes
                    .iter()
                    .map(|i| &i.operands[1])
                    .collect();
                assert_eq!(
                    modes.contains(&&SpvOperand::ExecutionMode(
                        spv::ExecutionMode::RoundingModeRTZ
                    )),
                    independent && !mixed
                );
            }
        }
        opts.shader.float32.denorm_preserve = true;
        let mut math = Math::new(opts.shader);
        configure(&mut math, &function(ShaderRoundingMode::NearestEven, false));
        assert_eq!(
            math.denorm32,
            Denorm32::Preserve,
            "native preservation avoids exact FTZ underflow repair"
        );
    }
    #[test]
    fn half_only_arithmetic_does_not_declare_unused_float32_execution_modes() {
        let mut b = FunctionBuilder::new(ShaderStage::Fragment);
        let block = b.entry();
        let pc = ShaderSourceLocation::new(0);
        let word = b.emit(
            block,
            pc,
            Op::Parameter {
                index: 0,
                ty: Type::U32,
            },
            vec![],
        )[0];
        let h = b.emit(block, pc, Op::UnpackHalf, vec![word.into()])[0];
        let h = b.emit(
            block,
            pc,
            Op::HalfFma { saturate: false },
            vec![h.into(); 3],
        )[0];
        let word = b.emit(block, pc, Op::PackHalf, vec![h.into()])[0];
        b.emit(
            block,
            pc,
            Op::Output {
                location: ShaderIoLocation::Color(0),
                component: 0,
                ty: Type::U32,
            },
            vec![word.into()],
        );
        b.terminate(block, pc, Terminator::Return);
        let mut opts = super::super::tests::options();
        opts.shader.native_half_conversions = true;
        opts.shader.native_half_fma = true;
        let mut f = b.finish().unwrap();
        f.outputs.push(
            ShaderInterfaceElement::new(
                ShaderIoLocation::Color(0),
                0,
                crate::ShaderScalarType::Unsigned32,
                None,
            )
            .unwrap(),
        );
        let module = emit(&f, opts).unwrap();
        let parsed = rspirv::dr::load_words(module.words).unwrap();
        assert!(
            parsed
                .execution_modes
                .iter()
                .all(|i| i.operands.last() != Some(&SpvOperand::LiteralBit32(32)))
        );
        assert!(
            parsed
                .execution_modes
                .iter()
                .any(|i| i.operands.last() == Some(&SpvOperand::LiteralBit32(16)))
        );
    }
}
