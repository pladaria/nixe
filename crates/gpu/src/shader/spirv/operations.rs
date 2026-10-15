use super::*;

impl Emitter {
    pub(super) fn operation(&mut self, operation: &ShaderOperation) -> Result<()> {
        use ShaderOperation::*;
        match operation {
            Nop => {}
            QuadShuffle32 { .. } | FloatQuadSwizzleAdd32 { .. } => {
                return Err(self
                    .math
                    .unsupported("fragment quad exchange requires the host quad backend"));
            }
            QueryTextureLevels { .. } => {
                return Err(self
                    .math
                    .unsupported("image queries require the WGSL texture backend"));
            }
            FusedMultiplyAddHalf32 { .. } => {
                return Err(self
                    .math
                    .unsupported("direct half FMA is not implemented by native SPIR-V lowering"));
            }
            LoadComputeBuiltin32 { .. } | StoreStorageBuffer32 { .. } => {
                return Err(self
                    .math
                    .unsupported("compute operations require the WGSL compute backend"));
            }
            Undefined32 { destination } => {
                self.write(*destination, self.math.undefined_uint);
            }
            MoveImmediate32 {
                destination,
                bits,
                scalar_type,
            } => {
                self.math.require_scalar32(*scalar_type)?;
                let value = self.math.constant(*bits);
                self.write(*destination, value);
            }
            Move32 {
                destination,
                source,
                scalar_type,
            } => {
                self.math.require_scalar32(*scalar_type)?;
                self.write(*destination, self.read(*source)?);
            }
            FloatAbsolute32 {
                destination,
                source,
            }
            | FloatNegate32 {
                destination,
                source,
            } => {
                let value = self.read(*source)?;
                let mask = self
                    .math
                    .constant(if matches!(operation, FloatAbsolute32 { .. }) {
                        0x7fff_ffff
                    } else {
                        0x8000_0000
                    });
                let result = if matches!(operation, FloatAbsolute32 { .. }) {
                    self.math.b.bitwise_and(self.math.uint, None, value, mask)?
                } else {
                    self.math.b.bitwise_xor(self.math.uint, None, value, mask)?
                };
                self.write(*destination, result);
            }
            UnpackHalf32 {
                destination,
                source,
                high,
            } => {
                let packed = self.read(*source)?;
                let value = self.math.unpack_half(packed, *high)?;
                self.write(*destination, value);
            }
            PackHalf32 {
                destination,
                source,
            } => {
                let value = self.read(*source)?;
                let packed = self.math.pack_half(value)?;
                self.write(*destination, packed);
            }
            ConvertIntegerToFloat32 {
                destination,
                source,
                source_type,
                rounding,
            } => {
                if *rounding != ShaderRoundingMode::NearestEven {
                    let source = self.read(*source)?;
                    let bits =
                        self.math
                            .directed_integer_to_float(source, *source_type, *rounding)?;
                    self.write(*destination, bits);
                    return Ok(());
                }
                // Integer conversions cannot produce subnormals. Request RTE
                // without requiring denormal preservation or emitting repairs.
                // https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpConvertSToF
                self.math.require_float(ShaderFloatControl::new(
                    ShaderRoundingMode::NearestEven,
                    ShaderNanMode::Propagate,
                    true,
                    true,
                    false,
                ))?;
                let source = self.read(*source)?;
                let result = match source_type {
                    ShaderScalarType::Unsigned32 => {
                        self.math.b.convert_u_to_f(self.math.float, None, source)?
                    }
                    ShaderScalarType::Signed32 => {
                        let source = self.math.b.bitcast(self.math.int, None, source)?;
                        self.math.b.convert_s_to_f(self.math.float, None, source)?
                    }
                    _ => {
                        return Err(self
                            .math
                            .unsupported("native integer conversion requires a 32-bit integer"));
                    }
                };
                let bits = self.math.b.bitcast(self.math.uint, None, result)?;
                self.write(*destination, bits);
            }
            LoadInput {
                destinations,
                location,
                first_component,
                ..
            } => {
                for (index, destination) in destinations.iter().enumerate() {
                    let value =
                        self.load_interface(true, *location, first_component + index as u8, None)?;
                    self.write(*destination, value);
                }
            }
            StoreOutput {
                sources,
                location,
                first_component,
                ..
            } => {
                for (index, source) in sources.iter().enumerate() {
                    self.store_interface(
                        *location,
                        first_component + index as u8,
                        None,
                        self.read(*source)?,
                    )?;
                }
            }
            InterpolateInput {
                destination,
                location,
                component,
                interpolation,
            } => {
                // Interpolation occurs at the fragment stage interface, not as
                // a second multiplication by the reciprocal of clip W.
                if !self
                    .io
                    .elements
                    .get(&(true, *location, *component))
                    .is_some_and(|element| element.interpolation == Some(*interpolation))
                {
                    return Err(self.math.unsupported(
                        "interpolation operation disagrees with its native interface",
                    ));
                }
                let value = self.load_interface(true, *location, *component, None)?;
                self.write(*destination, value);
            }
            LoadStorageBuffer32 {
                destination,
                binding,
                word_index,
            } => {
                let word = self.read(*word_index)?;
                let value = self.load_buffer_word(*binding, word)?;
                self.write(*destination, value);
            }
            LoadConstantBuffer32 {
                destination,
                binding,
                byte_offset,
                scalar_type,
            } => {
                self.math.require_scalar32(*scalar_type)?;
                if byte_offset & 3 != 0 {
                    return Err(self.math.unsupported("unaligned constant-buffer word load"));
                }
                let word = self.math.constant(byte_offset / 4);
                let value = self.load_buffer_word(*binding, word)?;
                self.write(*destination, value);
            }
            LoadConstantBufferIndexed32 {
                destination,
                binding,
                base_byte_offset,
                dynamic_byte_offset,
                scalar_type,
            } => {
                self.math.require_scalar32(*scalar_type)?;
                let dynamic = self.read(*dynamic_byte_offset)?;
                let base = self.math.constant(*base_byte_offset as u32);
                let address = self.math.b.i_add(self.math.uint, None, dynamic, base)?;
                let shift = self.math.constant(2);
                let word = self
                    .math
                    .b
                    .shift_right_logical(self.math.uint, None, address, shift)?;
                let value = self.load_buffer_word(*binding, word)?;
                self.write(*destination, value);
            }
            LoadControlPoint {
                destination,
                vertex,
                output,
                location,
                component,
            } => {
                let value =
                    self.load_interface(!output, *location, *component, Some(self.read(*vertex)?))?;
                self.write(*destination, value);
            }
            StoreControlPoint {
                source,
                vertex,
                location,
                component,
            } => {
                self.store_interface(
                    *location,
                    *component,
                    Some(self.read(*vertex)?),
                    self.read(*source)?,
                )?;
            }
            LoadPatchOutput {
                destination,
                location,
                component,
            } => {
                let value = self.load_interface(false, *location, *component, None)?;
                self.write(*destination, value);
            }
            PatchBarrier => {
                // GLSL450 TCS Output storage: execution Workgroup, memory
                // Invocation, semantics None. OpControlBarrier itself makes
                // output writes visible to the other invocations in the patch.
                // https://docs.vulkan.org/spec/latest/appendices/memorymodel.html#memory-model-tessellation-output-ordering
                let execution = self.math.constant(spv::Scope::Workgroup as u32);
                let memory = self.math.constant(spv::Scope::Invocation as u32);
                let semantics = self.math.constant(spv::MemorySemantics::empty().bits());
                self.math.b.control_barrier(execution, memory, semantics)?;
            }
            ShiftLeft32 {
                destination,
                value,
                amount,
                wrap,
            }
            | ShiftRight32 {
                destination,
                value,
                amount,
                wrap,
                ..
            } => {
                let mut base = self.read(*value)?;
                let raw_count = self.read(*amount)?;
                let (right, signed, reverse) = match operation {
                    ShiftRight32 {
                        signed, reverse, ..
                    } => (true, *signed, *reverse),
                    _ => (false, false, false),
                };
                if reverse {
                    base = self.math.b.bit_reverse(self.math.uint, None, base)?;
                }
                let mask = self.math.constant(31);
                let count = self
                    .math
                    .b
                    .bitwise_and(self.math.uint, None, raw_count, mask)?;
                let zero = self.math.constant(0);
                let (mut bits, clamped) = if signed {
                    let base = self.math.b.bitcast(self.math.int, None, base)?;
                    let shifted =
                        self.math
                            .b
                            .shift_right_arithmetic(self.math.int, None, base, count)?;
                    let sign =
                        self.math
                            .b
                            .shift_right_arithmetic(self.math.int, None, base, mask)?;
                    (
                        self.math.b.bitcast(self.math.uint, None, shifted)?,
                        self.math.b.bitcast(self.math.uint, None, sign)?,
                    )
                } else if right {
                    (
                        self.math
                            .b
                            .shift_right_logical(self.math.uint, None, base, count)?,
                        zero,
                    )
                } else {
                    (
                        self.math
                            .b
                            .shift_left_logical(self.math.uint, None, base, count)?,
                        zero,
                    )
                };
                if !wrap {
                    let limit = self.math.constant(32);
                    let inside =
                        self.math
                            .b
                            .u_less_than(self.math.boolean, None, raw_count, limit)?;
                    bits = self
                        .math
                        .b
                        .select(self.math.uint, None, inside, bits, clamped)?;
                }
                self.write(*destination, bits);
            }
            ExtractBitfield32 {
                destination,
                value,
                offset,
                count,
                signed,
                reverse,
            } => {
                let mut base = self.read(*value)?;
                if *reverse {
                    base = self.math.b.bit_reverse(self.math.uint, None, base)?;
                }
                let position = self.read(*offset)?;
                let raw_count = self.read(*count)?;
                let zero = self.math.constant(0);
                let limit = self.math.constant(32);
                let inside = self
                    .math
                    .b
                    .u_less_than(self.math.boolean, None, position, limit)?;
                let position = self
                    .math
                    .b
                    .select(self.math.uint, None, inside, position, limit)?;
                let available = self.math.b.i_sub(self.math.uint, None, limit, position)?;
                let smaller =
                    self.math
                        .b
                        .u_less_than(self.math.boolean, None, raw_count, available)?;
                let size =
                    self.math
                        .b
                        .select(self.math.uint, None, smaller, raw_count, available)?;
                let extracted = if *signed {
                    let base = self.math.b.bitcast(self.math.int, None, base)?;
                    let field = self.math.b.bit_field_s_extract(
                        self.math.int,
                        None,
                        base,
                        position,
                        size,
                    )?;
                    let shift = self.math.constant(31);
                    let sign =
                        self.math
                            .b
                            .shift_right_arithmetic(self.math.int, None, base, shift)?;
                    let value = self
                        .math
                        .b
                        .select(self.math.int, None, inside, field, sign)?;
                    self.math.b.bitcast(self.math.uint, None, value)?
                } else {
                    self.math
                        .b
                        .bit_field_u_extract(self.math.uint, None, base, position, size)?
                };
                let empty = self
                    .math
                    .b
                    .i_equal(self.math.boolean, None, raw_count, zero)?;
                let value = self
                    .math
                    .b
                    .select(self.math.uint, None, empty, zero, extracted)?;
                self.write(*destination, value);
            }
            Bitwise32 {
                destination,
                left,
                right,
                operation,
            } => {
                let left = self.read(*left)?;
                let right = self.read(*right)?;
                let value = match operation {
                    ShaderBitwiseOperation::And => {
                        self.math.b.bitwise_and(self.math.uint, None, left, right)?
                    }
                    ShaderBitwiseOperation::Or => {
                        self.math.b.bitwise_or(self.math.uint, None, left, right)?
                    }
                    ShaderBitwiseOperation::Xor => {
                        self.math.b.bitwise_xor(self.math.uint, None, left, right)?
                    }
                };
                self.write(*destination, value);
            }
            AddCarry32 {
                destination,
                carry_out,
                left,
                right,
                carry_in,
            } => {
                let left = self.read(*left)?;
                let right = self.read(*right)?;
                let zero = self.math.constant(0);
                let one = self.math.constant(1);
                let carry = if let Some(register) = carry_in {
                    let value = self.read(*register)?;
                    self.math.b.bitwise_and(self.math.uint, None, value, one)?
                } else {
                    zero
                };
                let sum = self.math.b.i_add(self.math.uint, None, left, right)?;
                let total = self.math.b.i_add(self.math.uint, None, sum, carry)?;
                let first = self
                    .math
                    .b
                    .u_less_than(self.math.boolean, None, sum, left)?;
                let second = self
                    .math
                    .b
                    .u_less_than(self.math.boolean, None, total, sum)?;
                let overflow = self
                    .math
                    .b
                    .logical_or(self.math.boolean, None, first, second)?;
                let carry = self
                    .math
                    .b
                    .select(self.math.uint, None, overflow, one, zero)?;
                self.write(*destination, total);
                self.write(*carry_out, carry);
            }
            Add32 {
                destination,
                left,
                right,
                scalar_type,
                float_control,
            }
            | Multiply32 {
                destination,
                left,
                right,
                scalar_type,
                float_control,
            } => {
                let multiply = matches!(operation, Multiply32 { .. });
                let left = self.read(*left)?;
                let right = self.read(*right)?;
                let value = match scalar_type {
                    ShaderScalarType::Unsigned32 | ShaderScalarType::Signed32 => {
                        if multiply {
                            self.math.b.i_mul(self.math.uint, None, left, right)?
                        } else {
                            self.math.b.i_add(self.math.uint, None, left, right)?
                        }
                    }
                    ShaderScalarType::Float32 => {
                        self.math
                            .float_binary(left, right, *float_control, multiply)?
                    }
                    _ => {
                        return Err(self
                            .math
                            .unsupported("64-bit arithmetic is not implemented"));
                    }
                };
                self.write(*destination, value);
            }
            FloatMultiplyZero32 {
                destination,
                left,
                right,
                float_control,
            } => {
                let mut left = self.read(*left)?;
                let mut right = self.read(*right)?;
                // Test integer magnitudes, including subnormals when DAZ is
                // enabled. Substitute +0 for BOTH operands before floating
                // arithmetic so 0*Inf/NaN never reaches the host multiply.
                let mask = self.math.constant(0x7fff_ffff);
                let threshold = self.math.constant(if float_control.denormals_are_zero() {
                    0x007f_ffff
                } else {
                    0
                });
                let a = self.math.b.bitwise_and(self.math.uint, None, left, mask)?;
                let b = self.math.b.bitwise_and(self.math.uint, None, right, mask)?;
                let a_zero =
                    self.math
                        .b
                        .u_less_than_equal(self.math.boolean, None, a, threshold)?;
                let b_zero =
                    self.math
                        .b
                        .u_less_than_equal(self.math.boolean, None, b, threshold)?;
                let absorbing = self
                    .math
                    .b
                    .logical_or(self.math.boolean, None, a_zero, b_zero)?;
                let zero = self.math.constant(0);
                left = self
                    .math
                    .b
                    .select(self.math.uint, None, absorbing, zero, left)?;
                right = self
                    .math
                    .b
                    .select(self.math.uint, None, absorbing, zero, right)?;
                let value = self.math.float_binary(left, right, *float_control, true)?;
                self.write(*destination, value);
            }
            FloatMinMax32 {
                destination,
                left,
                right,
                minimum,
                float_control,
            } => {
                self.math.require_float(*float_control)?;
                let left = self.math.float_operand(self.read(*left)?, *float_control)?;
                let right = self
                    .math
                    .float_operand(self.read(*right)?, *float_control)?;
                let less = self
                    .math
                    .b
                    .f_ord_less_than(self.math.boolean, None, left, right)?;
                let greater =
                    self.math
                        .b
                        .f_ord_greater_than(self.math.boolean, None, left, right)?;
                let minimum = self.predicate(*minimum)?;
                let select_left =
                    self.math
                        .b
                        .select(self.math.boolean, None, minimum, less, greater)?;
                let selected =
                    self.math
                        .b
                        .select(self.math.float, None, select_left, left, right)?;
                // min/max return the numeric operand when just one is NaN.
                let right_nan = self.math.b.is_nan(self.math.boolean, None, right)?;
                let selected =
                    self.math
                        .b
                        .select(self.math.float, None, right_nan, left, selected)?;
                let value = self.math.float_result(selected, *float_control)?;
                self.write(*destination, value);
            }
            FusedMultiplyAdd32 {
                destination,
                left,
                right,
                addend,
                float_control,
            } => {
                let left = self.read(*left)?;
                let right = self.read(*right)?;
                let addend = self.read(*addend)?;
                let value = self.math.fma(left, right, addend, *float_control)?;
                self.write(*destination, value);
            }
            Select32 {
                destination,
                condition,
                when_true,
                when_false,
            } => {
                let bits = self.read(*condition)?;
                let zero = self.math.constant(0);
                let condition = self
                    .math
                    .b
                    .i_not_equal(self.math.boolean, None, bits, zero)?;
                let yes = self.read(*when_true)?;
                let no = self.read(*when_false)?;
                let result = self
                    .math
                    .b
                    .select(self.math.uint, None, condition, yes, no)?;
                self.write(*destination, result);
            }
            SetPredicatesFromBits {
                source,
                first_bit,
                write_mask,
            } => {
                let bits = self.read(*source)?;
                let zero = self.math.constant(0);
                for index in 0..7_u8 {
                    if write_mask & (1 << index) != 0 {
                        let mask = self.math.constant(1 << (first_bit + index));
                        let bit = self.math.b.bitwise_and(self.math.uint, None, bits, mask)?;
                        let value = self
                            .math
                            .b
                            .i_not_equal(self.math.boolean, None, bit, zero)?;
                        self.predicates[usize::from(index)] = Some(value);
                    }
                }
            }
            SetPredicateBoolean {
                destinations,
                inputs,
                operations,
            } => {
                let a = self.predicate(inputs[0])?;
                let b = self.predicate(inputs[1])?;
                let c = self.predicate(inputs[2])?;
                for (index, destination) in destinations.iter().enumerate() {
                    if let Some(destination) = destination {
                        let first = if index == 0 {
                            a
                        } else {
                            self.math.b.logical_not(self.math.boolean, None, a)?
                        };
                        let mut value = first;
                        for (operation, operand) in operations.iter().zip([b, c]) {
                            value = match operation {
                                ShaderPredicateSetOperation::And => self.math.b.logical_and(
                                    self.math.boolean,
                                    None,
                                    value,
                                    operand,
                                )?,
                                ShaderPredicateSetOperation::Or => self.math.b.logical_or(
                                    self.math.boolean,
                                    None,
                                    value,
                                    operand,
                                )?,
                                ShaderPredicateSetOperation::Xor => self.math.b.logical_not_equal(
                                    self.math.boolean,
                                    None,
                                    value,
                                    operand,
                                )?,
                            };
                        }
                        self.predicates[usize::from(*destination)] = Some(value);
                    }
                }
            }
            SetPredicateInteger32 {
                destinations,
                left,
                right,
                signed,
                comparison,
                accumulator,
                set_operation,
            } => {
                // Snapshot both operands and accumulator before either result.
                let accumulator = self.predicate(*accumulator)?;
                let mut left = self.read(*left)?;
                let mut right = self.read(*right)?;
                if *signed {
                    left = self.math.b.bitcast(self.math.int, None, left)?;
                    right = self.math.b.bitcast(self.math.int, None, right)?;
                }
                use ShaderIntegerComparison::*;
                let result = match comparison {
                    False => self.math.false_value,
                    True => self.math.true_value,
                    Equal => self.math.b.i_equal(self.math.boolean, None, left, right)?,
                    NotEqual => self
                        .math
                        .b
                        .i_not_equal(self.math.boolean, None, left, right)?,
                    Less if *signed => {
                        self.math
                            .b
                            .s_less_than(self.math.boolean, None, left, right)?
                    }
                    Less => self
                        .math
                        .b
                        .u_less_than(self.math.boolean, None, left, right)?,
                    LessOrEqual if *signed => {
                        self.math
                            .b
                            .s_less_than_equal(self.math.boolean, None, left, right)?
                    }
                    LessOrEqual => {
                        self.math
                            .b
                            .u_less_than_equal(self.math.boolean, None, left, right)?
                    }
                    Greater if *signed => {
                        self.math
                            .b
                            .s_greater_than(self.math.boolean, None, left, right)?
                    }
                    Greater => self
                        .math
                        .b
                        .u_greater_than(self.math.boolean, None, left, right)?,
                    GreaterOrEqual if *signed => {
                        self.math
                            .b
                            .s_greater_than_equal(self.math.boolean, None, left, right)?
                    }
                    GreaterOrEqual => {
                        self.math
                            .b
                            .u_greater_than_equal(self.math.boolean, None, left, right)?
                    }
                };
                for (index, destination) in destinations.iter().enumerate() {
                    if let Some(destination) = destination {
                        let compared = if index == 0 {
                            result
                        } else {
                            self.math.b.logical_not(self.math.boolean, None, result)?
                        };
                        let value = match set_operation {
                            ShaderPredicateSetOperation::And => self.math.b.logical_and(
                                self.math.boolean,
                                None,
                                compared,
                                accumulator,
                            )?,
                            ShaderPredicateSetOperation::Or => self.math.b.logical_or(
                                self.math.boolean,
                                None,
                                compared,
                                accumulator,
                            )?,
                            ShaderPredicateSetOperation::Xor => self.math.b.logical_not_equal(
                                self.math.boolean,
                                None,
                                compared,
                                accumulator,
                            )?,
                        };
                        self.predicates[usize::from(*destination)] = Some(value);
                    }
                }
            }
            Branch { .. } | Exit => {
                return Err(self
                    .math
                    .unsupported("guest control flow requires CFG structurization"));
            }
            _ => {
                return Err(self
                    .math
                    .unsupported("operation has no native SPIR-V lowering"));
            }
        }
        Ok(())
    }
}
