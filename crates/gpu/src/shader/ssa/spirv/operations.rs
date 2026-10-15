use super::*;

impl Emitter<'_> {
    pub(super) fn operation(&mut self, inst: &Instruction) -> Result<(), Error> {
        let args: Vec<_> = inst
            .arguments
            .iter()
            .filter(|v| self.function.values[v.index()].ty != Type::Effect)
            .map(|&v| self.id(v))
            .collect();
        let result = inst
            .results
            .first()
            .copied()
            .filter(|v| self.function.values[v.index()].ty != Type::Effect);
        let result_type = result.map(|v| self.function.values[v.index()].ty);
        let ty = if let Some(ty) = result_type {
            self.ty(ty)?
        } else {
            0
        };
        let reserved = result.map(|v| self.id(v));
        let native = match inst.op {
            Op::EffectStart => return Ok(()),
            Op::Constant(c) => Some(match c.ty {
                Type::Bool => {
                    if c.bits != 0 {
                        self.math.true_value
                    } else {
                        self.math.false_value
                    }
                }
                Type::U32 => self.math.constant(c.bits as u32),
                Type::I32 | Type::F32 => self.math.b.constant_bit32(ty, c.bits as u32),
                Type::U64 | Type::I64 | Type::F64 => self.math.b.constant_bit64(ty, c.bits),
                Type::Effect => return Err(self.error("effect constant is invalid")),
            }),
            Op::Undefined(_) => Some(self.math.b.undef(ty, None)),
            Op::Parameter { index, ty: scalar } => {
                let variable = self
                    .parameters
                    .ok_or_else(|| self.error("parameter ABI was not reflected"))?;
                let ptr =
                    self.math
                        .b
                        .type_pointer(None, spv::StorageClass::PushConstant, self.math.uint);
                let zero = self.math.constant(0);
                let index = self.math.constant(index);
                let address = self
                    .math
                    .b
                    .access_chain(ptr, None, variable, [zero, index])
                    .map_err(|e| self.error(e.to_string()))?;
                let word = self
                    .math
                    .b
                    .load(self.math.uint, None, address, None, [])
                    .map_err(|e| self.error(e.to_string()))?;
                Some(if scalar == Type::U32 {
                    word
                } else if scalar == Type::Bool {
                    self.math
                        .b
                        .i_not_equal(self.math.boolean, None, word, zero)
                        .map_err(|e| self.error(e.to_string()))?
                } else {
                    self.math
                        .b
                        .bitcast(ty, None, word)
                        .map_err(|e| self.error(e.to_string()))?
                })
            }
            Op::Copy(_) => Some(args[0]),
            Op::Bitcast { from, to } => Some(if from == to {
                args[0]
            } else {
                self.math
                    .b
                    .bitcast(ty, reserved, args[0])
                    .map_err(|e| self.error(e.to_string()))?
            }),
            Op::Unary { op, ty: scalar } => Some(self.unary(op, scalar, args[0])?),
            Op::Binary {
                op,
                ty: scalar,
                float_control,
            } => {
                if scalar == Type::F32 && matches!(op, Binary::Add | Binary::Multiply) {
                    let a = self
                        .math
                        .b
                        .bitcast(self.math.uint, None, args[0])
                        .map_err(|e| self.error(e.to_string()))?;
                    let b = self
                        .math
                        .b
                        .bitcast(self.math.uint, None, args[1])
                        .map_err(|e| self.error(e.to_string()))?;
                    let bits = self
                        .math
                        .float_binary(a, b, float_control, op == Binary::Multiply)
                        .map_err(|e| self.error(e.to_string()))?;
                    Some(
                        self.math
                            .b
                            .bitcast(ty, None, bits)
                            .map_err(|e| self.error(e.to_string()))?,
                    )
                } else {
                    Some(
                        match op {
                            Binary::Add if scalar == Type::F64 => return Err(self.error(
                                "wide floating arithmetic requires explicit numerical guarantees",
                            )),
                            Binary::Multiply if scalar == Type::F64 => return Err(self.error(
                                "wide floating arithmetic requires explicit numerical guarantees",
                            )),
                            Binary::Add => self.math.b.i_add(ty, reserved, args[0], args[1]),
                            Binary::Multiply => self.math.b.i_mul(ty, reserved, args[0], args[1]),
                            Binary::And if scalar == Type::Bool => {
                                self.math.b.logical_and(ty, reserved, args[0], args[1])
                            }
                            Binary::Or if scalar == Type::Bool => {
                                self.math.b.logical_or(ty, reserved, args[0], args[1])
                            }
                            Binary::Xor if scalar == Type::Bool => self
                                .math
                                .b
                                .logical_not_equal(ty, reserved, args[0], args[1]),
                            Binary::And => self.math.b.bitwise_and(ty, reserved, args[0], args[1]),
                            Binary::Or => self.math.b.bitwise_or(ty, reserved, args[0], args[1]),
                            Binary::Xor => self.math.b.bitwise_xor(ty, reserved, args[0], args[1]),
                            Binary::ShiftLeft => self
                                .math
                                .b
                                .shift_left_logical(ty, reserved, args[0], args[1]),
                            Binary::ShiftRight if matches!(scalar, Type::I32 | Type::I64) => self
                                .math
                                .b
                                .shift_right_arithmetic(ty, reserved, args[0], args[1]),
                            Binary::ShiftRight => self
                                .math
                                .b
                                .shift_right_logical(ty, reserved, args[0], args[1]),
                        }
                        .map_err(|e| self.error(e.to_string()))?,
                    )
                }
            }
            Op::Fma {
                ty: Type::F32,
                float_control,
            } => {
                let mut bits = Vec::new();
                for arg in args {
                    bits.push(
                        self.math
                            .b
                            .bitcast(self.math.uint, None, arg)
                            .map_err(|e| self.error(e.to_string()))?,
                    );
                }
                let value = self
                    .math
                    .fma(bits[0], bits[1], bits[2], float_control)
                    .map_err(|e| self.error(e.to_string()))?;
                Some(
                    self.math
                        .b
                        .bitcast(ty, None, value)
                        .map_err(|e| self.error(e.to_string()))?,
                )
            }
            Op::Fma { .. } => {
                return Err(self.error("wide FMA requires an explicit host numerical lowering"));
            }
            Op::IntegerCompare {
                ty: scalar,
                comparison,
            } => {
                let signed = matches!(scalar, Type::I32 | Type::I64);
                use ShaderIntegerComparison::*;
                Some(match comparison {
                    False => self.math.false_value,
                    True => self.math.true_value,
                    Equal => self
                        .math
                        .b
                        .i_equal(ty, reserved, args[0], args[1])
                        .map_err(|e| self.error(e.to_string()))?,
                    NotEqual => self
                        .math
                        .b
                        .i_not_equal(ty, reserved, args[0], args[1])
                        .map_err(|e| self.error(e.to_string()))?,
                    c => match (c, signed) {
                        (Less, false) => self.math.b.u_less_than(ty, reserved, args[0], args[1]),
                        (Less, true) => self.math.b.s_less_than(ty, reserved, args[0], args[1]),
                        (LessOrEqual, false) => self
                            .math
                            .b
                            .u_less_than_equal(ty, reserved, args[0], args[1]),
                        (LessOrEqual, true) => self
                            .math
                            .b
                            .s_less_than_equal(ty, reserved, args[0], args[1]),
                        (Greater, false) => {
                            self.math.b.u_greater_than(ty, reserved, args[0], args[1])
                        }
                        (Greater, true) => {
                            self.math.b.s_greater_than(ty, reserved, args[0], args[1])
                        }
                        (GreaterOrEqual, false) => self
                            .math
                            .b
                            .u_greater_than_equal(ty, reserved, args[0], args[1]),
                        (GreaterOrEqual, true) => self
                            .math
                            .b
                            .s_greater_than_equal(ty, reserved, args[0], args[1]),
                        _ => unreachable!(),
                    }
                    .map_err(|e| self.error(e.to_string()))?,
                })
            }
            Op::Select(_) => Some(
                self.math
                    .b
                    .select(ty, reserved, args[0], args[1], args[2])
                    .map_err(|e| self.error(e.to_string()))?,
            ),
            Op::Builtin { builtin, component } => Some(self.builtin(builtin, component)?),
            Op::Input {
                location,
                component,
                ty: scalar,
            } => {
                let bits = self
                    .io
                    .load_interface(&mut self.math, true, location, component as u8, None)
                    .map_err(|e| self.error(e.to_string()))?;
                Some(if scalar == Type::U32 {
                    bits
                } else {
                    self.math
                        .b
                        .bitcast(ty, None, bits)
                        .map_err(|e| self.error(e.to_string()))?
                })
            }
            Op::Output {
                location,
                component,
                ty: scalar,
            } => {
                let bits = if scalar == Type::U32 {
                    args[0]
                } else {
                    self.math
                        .b
                        .bitcast(self.math.uint, None, args[0])
                        .map_err(|e| self.error(e.to_string()))?
                };
                self.io
                    .store_interface(&mut self.math, location, component as u8, None, bits)
                    .map_err(|e| self.error(e.to_string()))?;
                None
            }
            Op::Load {
                space,
                resource,
                ty: scalar,
                address_type,
            } => {
                if !matches!(scalar, Type::U32 | Type::I32 | Type::F32) {
                    return Err(self.error("memory operation must be legalized to 32-bit words"));
                }
                let pointer = self.memory_pointer(space, resource, args[0], address_type)?;
                let bits = self
                    .math
                    .b
                    .load(self.math.uint, None, pointer, None, [])
                    .map_err(|e| self.error(e.to_string()))?;
                Some(if scalar == Type::U32 {
                    bits
                } else {
                    self.math
                        .b
                        .bitcast(ty, None, bits)
                        .map_err(|e| self.error(e.to_string()))?
                })
            }
            Op::Store {
                space,
                resource,
                ty: scalar,
                address_type,
            } => {
                if !matches!(scalar, Type::U32 | Type::I32 | Type::F32) {
                    return Err(self.error("memory operation must be legalized to 32-bit words"));
                }
                let pointer = self.memory_pointer(space, resource, args[0], address_type)?;
                let bits = if scalar == Type::U32 {
                    args[1]
                } else {
                    self.math
                        .b
                        .bitcast(self.math.uint, None, args[1])
                        .map_err(|e| self.error(e.to_string()))?
                };
                self.math
                    .b
                    .store(pointer, bits, None, [])
                    .map_err(|e| self.error(e.to_string()))?;
                None
            }
            Op::ControlPointLoad {
                location,
                component,
                output,
            } => Some(
                self.io
                    .load_interface(
                        &mut self.math,
                        !output,
                        location,
                        component as u8,
                        Some(args[0]),
                    )
                    .map_err(|e| self.error(e.to_string()))?,
            ),
            Op::ControlPointStore {
                location,
                component,
            } => {
                self.io
                    .store_interface(
                        &mut self.math,
                        location,
                        component as u8,
                        Some(args[0]),
                        args[1],
                    )
                    .map_err(|e| self.error(e.to_string()))?;
                None
            }
            Op::PatchOutputLoad {
                location,
                component,
            } => Some(
                self.io
                    .load_interface(&mut self.math, false, location, component as u8, None)
                    .map_err(|e| self.error(e.to_string()))?,
            ),
            Op::Interpolate {
                location,
                component,
                interpolation,
            } => {
                if self.function.stage != ShaderStage::Fragment
                    || !self
                        .io
                        .elements
                        .get(&(true, location, component as u8))
                        .is_some_and(|e| e.interpolation == Some(interpolation))
                {
                    return Err(self.error("interpolation disagrees with the fragment interface"));
                }
                let bits = self
                    .io
                    .load_interface(&mut self.math, true, location, component as u8, None)
                    .map_err(|e| self.error(e.to_string()))?;
                Some(
                    self.math
                        .b
                        .bitcast(self.math.float, None, bits)
                        .map_err(|e| self.error(e.to_string()))?,
                )
            }
            Op::Barrier {
                execution,
                memory,
                spaces,
            } => {
                if execution == BarrierScope::Patch {
                    if self.function.stage != ShaderStage::TessellationControl
                        || memory != BarrierScope::Invocation
                        || spaces != BarrierMemory::OUTPUTS
                    {
                        return Err(self.error("patch barrier requires TCS output ordering"));
                    }
                    // GLSL450 TCS output ordering is provided by the execution
                    // barrier itself; this is not device-wide image/buffer sync.
                    // https://docs.vulkan.org/spec/latest/appendices/memorymodel.html#memory-model-tessellation-output-ordering
                    let execution = self.math.constant(spv::Scope::Workgroup as u32);
                    let memory = self.math.constant(spv::Scope::Invocation as u32);
                    let semantics = self.math.constant(0);
                    self.math
                        .b
                        .control_barrier(execution, memory, semantics)
                        .map_err(|e| self.error(e.to_string()))?;
                    return Ok(());
                }
                if spaces.contains(BarrierMemory::OUTPUTS)
                    || memory == BarrierScope::Patch
                    || execution == BarrierScope::Device
                {
                    return Err(
                        self.error("barrier requires a supported shader execution/memory scope")
                    );
                }
                if execution == BarrierScope::Workgroup
                    && self.function.stage != ShaderStage::Compute
                {
                    return Err(self.error("workgroup rendezvous requires compute stage"));
                }
                let scope = |s| match s {
                    BarrierScope::Invocation => spv::Scope::Invocation,
                    BarrierScope::Workgroup => spv::Scope::Workgroup,
                    BarrierScope::Device => spv::Scope::Device,
                    BarrierScope::Patch => unreachable!(),
                } as u32;
                let memory_id = self.math.constant(scope(memory));
                let mut flags = if spaces == BarrierMemory::NONE {
                    spv::MemorySemantics::empty()
                } else {
                    spv::MemorySemantics::ACQUIRE_RELEASE
                };
                if spaces.contains(BarrierMemory::SHARED) {
                    flags |= spv::MemorySemantics::WORKGROUP_MEMORY;
                }
                if spaces.contains(BarrierMemory::STORAGE) {
                    flags |= spv::MemorySemantics::UNIFORM_MEMORY;
                }
                if spaces.contains(BarrierMemory::IMAGES) {
                    flags |= spv::MemorySemantics::IMAGE_MEMORY;
                }
                let semantics = self.math.constant(flags.bits());
                if execution == BarrierScope::Invocation {
                    self.math
                        .b
                        .memory_barrier(memory_id, semantics)
                        .map_err(|e| self.error(e.to_string()))?;
                } else {
                    let execution_id = self.math.constant(scope(execution));
                    self.math
                        .b
                        .control_barrier(execution_id, memory_id, semantics)
                        .map_err(|e| self.error(e.to_string()))?;
                }
                None
            }
            Op::AddCarry => {
                let pair = self.math.b.type_struct([self.math.uint, self.math.uint]);
                let first = self
                    .math
                    .b
                    .i_add_carry(pair, None, args[0], args[1])
                    .map_err(|e| self.error(e.to_string()))?;
                let sum = self
                    .math
                    .b
                    .composite_extract(self.math.uint, None, first, [0])
                    .map_err(|e| self.error(e.to_string()))?;
                let c1 = self
                    .math
                    .b
                    .composite_extract(self.math.uint, None, first, [1])
                    .map_err(|e| self.error(e.to_string()))?;
                let one = self.math.constant(1);
                let carry_in = self
                    .math
                    .b
                    .bitwise_and(self.math.uint, None, args[2], one)
                    .map_err(|e| self.error(e.to_string()))?;
                let second = self
                    .math
                    .b
                    .i_add_carry(pair, None, sum, carry_in)
                    .map_err(|e| self.error(e.to_string()))?;
                let sum = self
                    .math
                    .b
                    .composite_extract(self.math.uint, None, second, [0])
                    .map_err(|e| self.error(e.to_string()))?;
                let c2 = self
                    .math
                    .b
                    .composite_extract(self.math.uint, None, second, [1])
                    .map_err(|e| self.error(e.to_string()))?;
                let carry = self
                    .math
                    .b
                    .bitwise_or(self.math.uint, None, c1, c2)
                    .map_err(|e| self.error(e.to_string()))?;
                self.result(inst.results[1], carry);
                Some(sum)
            }
            Op::FloatCompare { .. }
            | Op::Convert { .. }
            | Op::UnpackHalf
            | Op::PackHalf
            | Op::HalfFma { .. }
            | Op::FloatMinMax { .. }
            | Op::Math { .. }
            | Op::RoundIntegral { .. }
            | Op::FloatToInteger { .. } => Some(self.numeric(&inst.op, &args)?),
            Op::TextureSample { .. } | Op::TextureLevels { .. } => {
                self.texture(inst, &args)?;
                None
            }
            Op::Quad(_) | Op::QuadSwizzleAdd { .. } => Some(self.quad(&inst.op, &args)?),
        };
        if let (Some(result), Some(native)) = (result, native) {
            self.result(result, native);
        }
        Ok(())
    }
    fn builtin(&mut self, builtin: ShaderComputeBuiltin, component: u32) -> Result<u32, Error> {
        if self.function.stage != ShaderStage::Compute {
            return Err(self.error("compute builtin outside compute stage"));
        }
        let scalar = builtin == ShaderComputeBuiltin::LocalInvocationIndex;
        if component >= if scalar { 1 } else { 3 } {
            return Err(self.error("invalid builtin component"));
        }
        let variable = *self
            .builtins
            .get(&builtin)
            .ok_or_else(|| self.error("builtin was not reflected before function emission"))?;
        let pointer = if scalar {
            variable
        } else {
            let ptr = self
                .math
                .b
                .type_pointer(None, spv::StorageClass::Input, self.math.uint);
            let index = self.math.constant(component);
            self.math
                .b
                .access_chain(ptr, None, variable, [index])
                .map_err(|e| self.error(e.to_string()))?
        };
        self.math
            .b
            .load(self.math.uint, None, pointer, None, [])
            .map_err(|e| self.error(e.to_string()))
    }
    pub(super) fn declare_builtin(&mut self, builtin: ShaderComputeBuiltin) -> Result<(), Error> {
        let scalar = builtin == ShaderComputeBuiltin::LocalInvocationIndex;
        let variable = if let Some(&variable) = self.builtins.get(&builtin) {
            variable
        } else {
            let ty = if scalar {
                self.math.uint
            } else {
                self.math.b.type_vector(self.math.uint, 3)
            };
            let ptr = self.math.b.type_pointer(None, spv::StorageClass::Input, ty);
            let variable = self
                .math
                .b
                .variable(ptr, None, spv::StorageClass::Input, None);
            let kind = match builtin {
                ShaderComputeBuiltin::GlobalInvocationId => spv::BuiltIn::GlobalInvocationId,
                ShaderComputeBuiltin::LocalInvocationId => spv::BuiltIn::LocalInvocationId,
                ShaderComputeBuiltin::WorkgroupId => spv::BuiltIn::WorkgroupId,
                ShaderComputeBuiltin::NumWorkgroups => spv::BuiltIn::NumWorkgroups,
                ShaderComputeBuiltin::LocalInvocationIndex => spv::BuiltIn::LocalInvocationIndex,
            };
            self.math.b.decorate(
                variable,
                spv::Decoration::BuiltIn,
                [SpvOperand::BuiltIn(kind)],
            );
            self.io.variables.push(variable);
            self.builtins.insert(builtin, variable);
            variable
        };
        let _ = variable;
        Ok(())
    }
    pub(super) fn declare_memory(&mut self, space: MemorySpace) -> Result<(), Error> {
        let (storage, bytes) = match space {
            MemorySpace::Shared => (
                spv::StorageClass::Workgroup,
                self.options.shared_memory_bytes,
            ),
            MemorySpace::Local => (spv::StorageClass::Private, self.options.local_memory_bytes),
            _ => return Err(self.error("descriptor-backed memory has no resource")),
        };
        if bytes == 0 || !bytes.is_multiple_of(4) {
            return Err(self.error(
                "shared/local word allocation needs a declared nonzero word-aligned byte size",
            ));
        }
        if (if space == MemorySpace::Shared {
            self.shared
        } else {
            self.local
        })
        .is_some()
        {
            return Ok(());
        }
        let count = self.math.constant(bytes / 4);
        let array = self.math.b.type_array(self.math.uint, count);
        let ptr = self.math.b.type_pointer(None, storage, array);
        let variable = self.math.b.variable(ptr, None, storage, None);
        self.io.variables.push(variable);
        if space == MemorySpace::Shared {
            self.shared = Some(variable);
        } else {
            self.local = Some(variable);
        }
        Ok(())
    }
    fn memory_pointer(
        &mut self,
        space: MemorySpace,
        resource: Option<ResourceId>,
        address: u32,
        address_type: Type,
    ) -> Result<u32, Error> {
        let address_ty = self.ty(address_type)?;
        let two = if address_type == Type::U64 {
            self.math.b.constant_bit64(address_ty, 2)
        } else {
            self.math.constant(2)
        };
        let index = self
            .math
            .b
            .shift_right_logical(address_ty, None, address, two)
            .map_err(|e| self.error(e.to_string()))?;
        if let Some(resource) = resource {
            let buffer = self
                .buffers
                .get(&resource)
                .ok_or_else(|| self.error("missing live buffer binding"))?;
            let zero = self.math.constant(0);
            return self
                .math
                .b
                .access_chain(buffer.pointer, None, buffer.variable, [zero, index])
                .map_err(|e| self.error(e.to_string()));
        }
        let storage = if space == MemorySpace::Shared {
            spv::StorageClass::Workgroup
        } else {
            spv::StorageClass::Private
        };
        let variable = (if space == MemorySpace::Shared {
            self.shared
        } else {
            self.local
        })
        .ok_or_else(|| self.error("shared/local allocation was not reflected before emission"))?;
        let ptr = self.math.b.type_pointer(None, storage, self.math.uint);
        self.math
            .b
            .access_chain(ptr, None, variable, [index])
            .map_err(|e| self.error(e.to_string()))
    }
}
