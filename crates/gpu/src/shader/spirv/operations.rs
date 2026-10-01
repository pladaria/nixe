use super::*;

impl Emitter {
    pub(super) fn operation(&mut self, operation: &ShaderOperation) -> Result<()> {
        use ShaderOperation::*;
        match operation {
            LoadComputeBuiltin32 { .. }
            | LoadStorageBuffer32 { .. }
            | StoreStorageBuffer32 { .. } => {
                return Err(self.unsupported("compute operations require the WGSL compute backend"));
            }
            Undefined32 { destination } => {
                self.write(*destination, self.undefined_uint);
            }
            MoveImmediate32 {
                destination,
                bits,
                scalar_type,
            } => {
                self.require_scalar32(*scalar_type)?;
                let value = self.constant(*bits);
                self.write(*destination, value);
            }
            Move32 {
                destination,
                source,
                scalar_type,
            } => {
                self.require_scalar32(*scalar_type)?;
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
                let mask = self.constant(if matches!(operation, FloatAbsolute32 { .. }) {
                    0x7fff_ffff
                } else {
                    0x8000_0000
                });
                let result = if matches!(operation, FloatAbsolute32 { .. }) {
                    self.b.bitwise_and(self.uint, None, value, mask)?
                } else {
                    self.b.bitwise_xor(self.uint, None, value, mask)?
                };
                self.write(*destination, result);
            }
            ConvertIntegerToFloat32 {
                destination,
                source,
                source_type,
            } => {
                // Integer conversions cannot produce subnormals. Request RTE
                // without requiring denormal preservation or emitting repairs.
                // https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpConvertSToF
                self.require_float(ShaderFloatControl::new(
                    ShaderRoundingMode::NearestEven,
                    ShaderNanMode::Propagate,
                    true,
                    true,
                    false,
                ))?;
                let source = self.read(*source)?;
                let result = match source_type {
                    ShaderScalarType::Unsigned32 => {
                        self.b.convert_u_to_f(self.float, None, source)?
                    }
                    ShaderScalarType::Signed32 => {
                        let source = self.b.bitcast(self.int, None, source)?;
                        self.b.convert_s_to_f(self.float, None, source)?
                    }
                    _ => {
                        return Err(
                            self.unsupported("native integer conversion requires a 32-bit integer")
                        );
                    }
                };
                let bits = self.b.bitcast(self.uint, None, result)?;
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
                    .interfaces
                    .get(&(true, *location, *component))
                    .is_some_and(|element| element.interpolation == Some(*interpolation))
                {
                    return Err(self.unsupported(
                        "interpolation operation disagrees with its native interface",
                    ));
                }
                let value = self.load_interface(true, *location, *component, None)?;
                self.write(*destination, value);
            }
            LoadConstantBuffer32 {
                destination,
                binding,
                byte_offset,
                scalar_type,
            } => {
                self.require_scalar32(*scalar_type)?;
                if byte_offset & 3 != 0 {
                    return Err(self.unsupported("unaligned constant-buffer word load"));
                }
                let word = self.constant(byte_offset / 4);
                let value = self.load_constant_buffer(*binding, word)?;
                self.write(*destination, value);
            }
            LoadConstantBufferIndexed32 {
                destination,
                binding,
                base_byte_offset,
                dynamic_byte_offset,
                scalar_type,
            } => {
                self.require_scalar32(*scalar_type)?;
                let dynamic = self.read(*dynamic_byte_offset)?;
                let base = self.constant(*base_byte_offset as u32);
                let address = self.b.i_add(self.uint, None, dynamic, base)?;
                let shift = self.constant(2);
                let word = self
                    .b
                    .shift_right_logical(self.uint, None, address, shift)?;
                let value = self.load_constant_buffer(*binding, word)?;
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
                let execution = self.constant(spv::Scope::Workgroup as u32);
                let memory = self.constant(spv::Scope::Invocation as u32);
                let semantics = self.constant(spv::MemorySemantics::empty().bits());
                self.b.control_barrier(execution, memory, semantics)?;
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
                        self.b.bitwise_and(self.uint, None, left, right)?
                    }
                    ShaderBitwiseOperation::Or => {
                        self.b.bitwise_or(self.uint, None, left, right)?
                    }
                    ShaderBitwiseOperation::Xor => {
                        self.b.bitwise_xor(self.uint, None, left, right)?
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
                let zero = self.constant(0);
                let one = self.constant(1);
                let carry = if let Some(register) = carry_in {
                    let value = self.read(*register)?;
                    self.b.bitwise_and(self.uint, None, value, one)?
                } else {
                    zero
                };
                let sum = self.b.i_add(self.uint, None, left, right)?;
                let total = self.b.i_add(self.uint, None, sum, carry)?;
                let first = self.b.u_less_than(self.boolean, None, sum, left)?;
                let second = self.b.u_less_than(self.boolean, None, total, sum)?;
                let overflow = self.b.logical_or(self.boolean, None, first, second)?;
                let carry = self.b.select(self.uint, None, overflow, one, zero)?;
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
                            self.b.i_mul(self.uint, None, left, right)?
                        } else {
                            self.b.i_add(self.uint, None, left, right)?
                        }
                    }
                    ShaderScalarType::Float32 => {
                        self.float_binary(left, right, *float_control, multiply)?
                    }
                    _ => return Err(self.unsupported("64-bit arithmetic is not implemented")),
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
                let mask = self.constant(0x7fff_ffff);
                let threshold = self.constant(if float_control.denormals_are_zero() {
                    0x007f_ffff
                } else {
                    0
                });
                let a = self.b.bitwise_and(self.uint, None, left, mask)?;
                let b = self.b.bitwise_and(self.uint, None, right, mask)?;
                let a_zero = self.b.u_less_than_equal(self.boolean, None, a, threshold)?;
                let b_zero = self.b.u_less_than_equal(self.boolean, None, b, threshold)?;
                let absorbing = self.b.logical_or(self.boolean, None, a_zero, b_zero)?;
                let zero = self.constant(0);
                left = self.b.select(self.uint, None, absorbing, zero, left)?;
                right = self.b.select(self.uint, None, absorbing, zero, right)?;
                let value = self.float_binary(left, right, *float_control, true)?;
                self.write(*destination, value);
            }
            FusedMultiplyAdd32 {
                destination,
                left,
                right,
                addend,
                float_control,
            } => {
                self.require_float(*float_control)?;
                if !self.options.float32.fused_multiply_add {
                    return Err(self.unsupported("correctly rounded float32 FMA requires VK_KHR_shader_fma / shaderFmaFloat32"));
                }
                // GLSL.std.450 Fma is explicitly allowed to be unfused in Vulkan.
                // Use the correctly rounded operation; never silently split it.
                // https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_shader_fma.html
                if !self.fma_extension {
                    self.b.extension("SPV_KHR_fma");
                    self.b.capability(spv::Capability::FMAKHR);
                    self.fma_extension = true;
                }
                let left = self.float_operand(self.read(*left)?, *float_control)?;
                let right = self.float_operand(self.read(*right)?, *float_control)?;
                let addend = self.float_operand(self.read(*addend)?, *float_control)?;
                let value = self.b.fma_khr(self.float, None, left, right, addend)?;
                self.b.decorate(value, spv::Decoration::NoContraction, []);
                let value = if self.options.float32.denorm_preserve {
                    value
                } else {
                    self.repair_product_underflow(value, left, right, Some(addend))?
                };
                let value = self.float_result(value, *float_control)?;
                self.write(*destination, value);
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
                    left = self.b.bitcast(self.int, None, left)?;
                    right = self.b.bitcast(self.int, None, right)?;
                }
                use ShaderIntegerComparison::*;
                let result = match comparison {
                    False => self.false_value,
                    True => self.true_value,
                    Equal => self.b.i_equal(self.boolean, None, left, right)?,
                    NotEqual => self.b.i_not_equal(self.boolean, None, left, right)?,
                    Less if *signed => self.b.s_less_than(self.boolean, None, left, right)?,
                    Less => self.b.u_less_than(self.boolean, None, left, right)?,
                    LessOrEqual if *signed => {
                        self.b.s_less_than_equal(self.boolean, None, left, right)?
                    }
                    LessOrEqual => self.b.u_less_than_equal(self.boolean, None, left, right)?,
                    Greater if *signed => self.b.s_greater_than(self.boolean, None, left, right)?,
                    Greater => self.b.u_greater_than(self.boolean, None, left, right)?,
                    GreaterOrEqual if *signed => {
                        self.b
                            .s_greater_than_equal(self.boolean, None, left, right)?
                    }
                    GreaterOrEqual => {
                        self.b
                            .u_greater_than_equal(self.boolean, None, left, right)?
                    }
                };
                for (index, destination) in destinations.iter().enumerate() {
                    if let Some(destination) = destination {
                        let compared = if index == 0 {
                            result
                        } else {
                            self.b.logical_not(self.boolean, None, result)?
                        };
                        let value = match set_operation {
                            ShaderPredicateSetOperation::And => {
                                self.b
                                    .logical_and(self.boolean, None, compared, accumulator)?
                            }
                            ShaderPredicateSetOperation::Or => {
                                self.b
                                    .logical_or(self.boolean, None, compared, accumulator)?
                            }
                            ShaderPredicateSetOperation::Xor => self.b.logical_not_equal(
                                self.boolean,
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
                return Err(self.unsupported("guest control flow requires CFG structurization"));
            }
            _ => return Err(self.unsupported("operation has no native SPIR-V lowering")),
        }
        Ok(())
    }

    fn require_float(&mut self, control: ShaderFloatControl) -> Result<()> {
        if control.rounding != ShaderRoundingMode::NearestEven || control.saturate {
            return Err(self.unsupported(
                "native float arithmetic requires nearest-even rounding without saturation",
            ));
        }
        let caps = self.options.float32;
        if !caps.rounding_mode_rte || !caps.signed_zero_inf_nan_preserve {
            return Err(self.unsupported(
                "float32 requires RoundingModeRTE and SignedZeroInfNanPreserve host guarantees",
            ));
        }
        if !caps.denorm_preserve && !(control.denormals_are_zero && control.flush_denormals_to_zero)
        {
            return Err(self.unsupported("float32 without DenormPreserve requires both DAZ and FTZ; preserving or mixed denormal controls are unsupported"));
        }
        if !self.float_modes {
            self.b.extension("SPV_KHR_float_controls");
            for (capability, mode) in [
                (
                    spv::Capability::RoundingModeRTE,
                    spv::ExecutionMode::RoundingModeRTE,
                ),
                (
                    spv::Capability::SignedZeroInfNanPreserve,
                    spv::ExecutionMode::SignedZeroInfNanPreserve,
                ),
            ] {
                self.b.capability(capability);
                self.b.execution_mode(self.entry, mode, [32]);
            }
            if caps.denorm_preserve {
                self.b.capability(spv::Capability::DenormPreserve);
                self.b
                    .execution_mode(self.entry, spv::ExecutionMode::DenormPreserve, [32]);
            }
            self.float_modes = true;
        }
        Ok(())
    }

    fn require_scalar32(&self, ty: ShaderScalarType) -> Result<()> {
        if matches!(
            ty,
            ShaderScalarType::Unsigned32 | ShaderScalarType::Signed32 | ShaderScalarType::Float32
        ) {
            Ok(())
        } else {
            Err(self.unsupported("64-bit scalar moves are not implemented"))
        }
    }

    fn float_binary(
        &mut self,
        left: u32,
        right: u32,
        control: ShaderFloatControl,
        multiply: bool,
    ) -> Result<u32> {
        self.require_float(control)?;
        let left = self.float_operand(left, control)?;
        let right = self.float_operand(right, control)?;
        let value = if multiply {
            self.b.f_mul(self.float, None, left, right)?
        } else {
            self.b.f_add(self.float, None, left, right)?
        };
        self.b.decorate(value, spv::Decoration::NoContraction, []);
        let value = if self.options.float32.denorm_preserve {
            value
        } else {
            self.repair_binary_underflow(value, left, right, multiply)?
        };
        self.float_result(value, control)
    }

    fn flush_denormal(&mut self, bits: u32) -> Result<u32> {
        // Integer classification preserves the sign of zero and leaves NaNs and
        // infinities untouched. Per-operation DAZ/FTZ can coexist in one shader.
        let exponent_mask = self.constant(0x7f80_0000);
        let exponent = self.b.bitwise_and(self.uint, None, bits, exponent_mask)?;
        let zero = self.constant(0);
        let denormal_or_zero = self.b.i_equal(self.boolean, None, exponent, zero)?;
        let sign_mask = self.constant(0x8000_0000);
        let sign = self.b.bitwise_and(self.uint, None, bits, sign_mask)?;
        Ok(self
            .b
            .select(self.uint, None, denormal_or_zero, sign, bits)?)
    }

    fn float_operand(&mut self, mut bits: u32, control: ShaderFloatControl) -> Result<u32> {
        if control.denormals_are_zero {
            bits = self.flush_denormal(bits)?;
        }
        Ok(self.b.bitcast(self.float, None, bits)?)
    }

    fn float_result(&mut self, value: u32, control: ShaderFloatControl) -> Result<u32> {
        let mut bits = self.b.bitcast(self.uint, None, value)?;
        if control.nan_mode == ShaderNanMode::Canonicalize {
            let is_nan = self.b.is_nan(self.boolean, None, value)?;
            let nan = self.constant(f32::NAN.to_bits());
            bits = self.b.select(self.uint, None, is_nan, nan, bits)?;
        }
        if control.flush_denormals_to_zero {
            bits = self.flush_denormal(bits)?;
        }
        Ok(bits)
    }
}
