use super::*;

impl Emitter<'_> {
    fn bits(&mut self, value: u32) -> Result<u32, Error> {
        self.math
            .b
            .bitcast(self.math.uint, None, value)
            .map_err(|e| self.error(e.to_string()))
    }
    fn float(&mut self, bits: u32) -> Result<u32, Error> {
        self.math
            .b
            .bitcast(self.math.float, None, bits)
            .map_err(|e| self.error(e.to_string()))
    }
    pub(super) fn numeric(&mut self, op: &Op, args: &[u32]) -> Result<u32, Error> {
        match *op {
            Op::UnpackHalf => self.half_unpack(args[0]),
            Op::PackHalf => self.half_pack(args[0]),
            Op::HalfFma { saturate } => self.half_fma([args[0], args[1], args[2]], saturate),
            Op::RoundIntegral {
                rounding,
                flush_denormals,
            } => {
                let bits = self.bits(args[0])?;
                let bits = self
                    .math
                    .round_integral_bits(bits, rounding, flush_denormals)
                    .map_err(|e| self.error(e.to_string()))?;
                self.float(bits)
            }
            Op::FloatToInteger {
                to,
                bits,
                rounding,
                flush_denormals,
            } => {
                if !matches!(to, Type::U32 | Type::I32) {
                    return Err(self.error("float-to-integer result must be a 32-bit scalar"));
                }
                let input = self.bits(args[0])?;
                let bits = self
                    .math
                    .float_to_integer_bits(input, to == Type::I32, bits, rounding, flush_denormals)
                    .map_err(|e| self.error(e.to_string()))?;
                if to == Type::U32 {
                    Ok(bits)
                } else {
                    self.math
                        .b
                        .bitcast(self.math.int, None, bits)
                        .map_err(|e| self.error(e.to_string()))
                }
            }
            Op::Convert {
                from,
                to,
                rounding,
                flush_denormals,
            } => {
                if from == to {
                    return Ok(args[0]);
                }
                match (from, to) {
                    (Type::F16, Type::F32) if !flush_denormals => self.half_widen(args[0]),
                    (Type::F32, Type::F16)
                        if rounding == ShaderRoundingMode::NearestEven && !flush_denormals =>
                    {
                        self.half_narrow(args[0])
                    }
                    (Type::U32 | Type::I32, Type::F32) => {
                        let bits = if from == Type::U32 {
                            args[0]
                        } else {
                            self.bits(args[0])?
                        };
                        let scalar = if from == Type::U32 {
                            crate::ShaderScalarType::Unsigned32
                        } else {
                            crate::ShaderScalarType::Signed32
                        };
                        let value = self
                            .math
                            .directed_integer_to_float(bits, scalar, rounding)
                            .map_err(|e| self.error(e.to_string()))?;
                        self.float(value)
                    }
                    (Type::F32, Type::U32 | Type::I32) => self.numeric(
                        &Op::FloatToInteger {
                            to,
                            bits: 32,
                            rounding,
                            flush_denormals,
                        },
                        args,
                    ),
                    (Type::U32, Type::I32) | (Type::I32, Type::U32) => {
                        let ty = self.ty(to)?;
                        self.math
                            .b
                            .bitcast(ty, None, args[0])
                            .map_err(|e| self.error(e.to_string()))
                    }
                    _ => {
                        Err(self.error("conversion width requires an explicit numerical lowering"))
                    }
                }
            }
            Op::FloatCompare {
                ty: Type::F32,
                comparison,
                flush_denormals,
            } => {
                let a = self.bits(args[0])?;
                let b = self.bits(args[1])?;
                self.math
                    .compare_float_bits(a, b, comparison, flush_denormals)
                    .map_err(|e| self.error(e.to_string()))
            }

            Op::FloatCompare { .. } => {
                Err(self.error("wide comparison requires explicit denormal guarantees"))
            }
            Op::FloatMinMax { float_control } => {
                let a = self.bits(args[0])?;
                let b = self.bits(args[1])?;
                let bits = self
                    .math
                    .minmax_bits(a, b, args[2], float_control)
                    .map_err(|e| self.error(e.to_string()))?;
                self.float(bits)
            }
            Op::Math {
                function,
                accuracy,
                float_control,
            } => {
                if accuracy != ShaderMathAccuracy::Approximate {
                    return Err(self.error("special math requires a proven exact host lowering"));
                }
                self.math
                    .require_repaired_float(float_control)
                    .map_err(|e| self.error(e.to_string()))?;
                let bits = self.bits(args[0])?;
                let input = self
                    .math
                    .float_operand(bits, float_control)
                    .map_err(|e| self.error(e.to_string()))?;
                let value = if function == MathFunction::Reciprocal {
                    let one = self
                        .math
                        .b
                        .constant_bit32(self.math.float, 1.0f32.to_bits());
                    self.math
                        .b
                        .f_div(self.math.float, None, one, input)
                        .map_err(|e| self.error(e.to_string()))?
                } else {
                    let set = *self
                        .math
                        .glsl_import
                        .get_or_insert_with(|| self.math.b.ext_inst_import("GLSL.std.450"));
                    use ShaderSpecialFunction as F;
                    use spv::GlslStd450Op as G;
                    let operation = match function {
                        MathFunction::ReciprocalSqrt => G::InverseSqrt,
                        MathFunction::Special(F::Cosine) => G::Cos,
                        MathFunction::Special(F::Sine) => G::Sin,
                        MathFunction::Special(F::Exp2) => G::Exp2,
                        MathFunction::Special(F::Log2) => G::Log2,
                        MathFunction::Special(F::SquareRoot) => G::Sqrt,
                        MathFunction::Reciprocal => unreachable!(),
                    };
                    self.math
                        .b
                        .ext_inst(
                            self.math.float,
                            None,
                            set,
                            operation as u32,
                            [SpvOperand::IdRef(input)],
                        )
                        .map_err(|e| self.error(e.to_string()))?
                };
                self.math
                    .b
                    .decorate(value, spv::Decoration::NoContraction, []);
                let value = if !self.math.preserves_float32()
                    && !(float_control.denormals_are_zero && float_control.flush_denormals_to_zero)
                    && matches!(
                        function,
                        MathFunction::Reciprocal | MathFunction::ReciprocalSqrt
                    ) {
                    self.math
                        .repair_inverse(value, input, function == MathFunction::ReciprocalSqrt)
                        .map_err(|e| self.error(e.to_string()))?
                } else {
                    value
                };
                let value = if !self.math.preserves_float32()
                    && !(float_control.denormals_are_zero && float_control.flush_denormals_to_zero)
                {
                    use spv::GlslStd450Op as G;
                    let operation = match function {
                        MathFunction::Special(ShaderSpecialFunction::Cosine) => Some(G::Cos),
                        MathFunction::Special(ShaderSpecialFunction::Sine) => Some(G::Sin),
                        MathFunction::Special(ShaderSpecialFunction::Exp2) => Some(G::Exp2),
                        MathFunction::Special(ShaderSpecialFunction::Log2) => Some(G::Log2),
                        MathFunction::Special(ShaderSpecialFunction::SquareRoot) => Some(G::Sqrt),
                        _ => None,
                    };
                    if let Some(operation) = operation {
                        self.math
                            .repair_sfu(value, input, operation)
                            .map_err(|e| self.error(e.to_string()))?
                    } else {
                        value
                    }
                } else {
                    value
                };
                let bits = self
                    .math
                    .float_result(value, float_control)
                    .map_err(|e| self.error(e.to_string()))?;
                self.float(bits)
            }
            _ => unreachable!("non-numerical operation passed to scalar numerical lowering"),
        }
    }
}
