//! Relocate proven constant-buffer-based 64-bit pointers into storage buffers.
//! Numeric address arithmetic remains in IR when otherwise observable. Only
//! global accesses with a proven base, aligned offset and dispatch bound can
//! become a host buffer access. No shader fingerprints or fixed ABI slots.
//!
//! STG fields: public UAM GM107 emitSTG/emitLDSTs/emitLDSTc.
//! https://github.com/devkitPro/uam/blob/master/mesa-imported/codegen/nv50_ir_emit_gm107.cpp
use super::*;

#[cfg(test)]
mod tests;

pub(super) const fn is_store(encoding: u64) -> bool {
    (encoding >> 48) & 0xfff8 == 0xeed8
}

#[derive(Clone, Debug)]
enum Bound {
    Constant(u32),
    Local(u8),
    Group(u8),
    Add(usize, usize),
    Shift(usize, u32),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Offset {
    node: usize,
    alignment: u32,
    constant: Option<u32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Pointer {
    constant_buffer: u8,
    byte_offset: u32,
    offset: Offset,
    frozen: ShaderRegister,
}

#[derive(Clone, Copy)]
enum Value {
    Offset(Offset),
    BaseWord(u8, u32),
    Low(Pointer),
    High(Pointer),
    Carry(Pointer),
}

#[derive(Clone, Debug)]
pub(crate) struct GlobalBufferBinding {
    pub constant_buffer: u8,
    pub byte_offset: u32,
    pub binding: u8,
    accesses: Vec<(usize, u32)>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct GlobalBufferBindings {
    pub buffers: Vec<GlobalBufferBinding>,
    bounds: Vec<Bound>,
}

impl GlobalBufferBindings {
    /// Linear-time evaluation of the shared bound DAG, once per dispatch.
    /// A wrapping operation conservatively expands to the entire u32 domain;
    /// it never truncates a possible access interval.
    pub(crate) fn byte_extents(&self, groups: [u32; 3], size: [u32; 3]) -> Vec<u64> {
        let mut values = Vec::<u32>::with_capacity(self.bounds.len());
        for bound in &self.bounds {
            let value = match *bound {
                Bound::Constant(value) => value,
                Bound::Local(component) => size[usize::from(component)].saturating_sub(1),
                Bound::Group(component) => groups[usize::from(component)].saturating_sub(1),
                Bound::Add(a, b) => values[a].saturating_add(values[b]),
                Bound::Shift(value, amount) => {
                    u32::try_from(u64::from(values[value]) << amount).unwrap_or(u32::MAX)
                }
            };
            values.push(value);
        }
        self.buffers
            .iter()
            .map(|buffer| {
                buffer
                    .accesses
                    .iter()
                    .map(|&(node, offset)| u64::from(values[node]) + u64::from(offset) + 4)
                    .max()
                    .expect("binding is introduced by a store")
            })
            .collect()
    }
}

#[derive(Default)]
pub(super) struct GlobalMemory {
    values: BTreeMap<ShaderRegister, Value>,
    pub bindings: GlobalBufferBindings,
    control_flow: bool,
}

impl GlobalMemory {
    pub(super) fn observe_control_flow(
        &mut self,
        offset: u32,
        encoding: u64,
    ) -> Result<(), MaxwellShaderTranslationError> {
        // A later backward edge can invalidate an earlier store's inferred
        // bounds. Reject both instruction orders until pointer joins exist.
        if !self.bindings.buffers.is_empty() {
            return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
                stage: MaxwellShaderStage::Compute,
                instruction_offset: offset,
                encoding,
                detail: "global stores across control flow require pointer merge analysis",
            });
        }
        self.control_flow = true;
        Ok(())
    }

    fn offset(&mut self, bound: Bound, alignment: u32, constant: Option<u32>) -> Value {
        let node = self.bindings.bounds.len();
        self.bindings.bounds.push(bound);
        Value::Offset(Offset {
            node,
            alignment,
            constant,
        })
    }

    /// Observe the freshly emitted scalar operations, snapshotting relative
    /// offsets before an aliased address destination overwrites them.
    pub(super) fn observe(
        &mut self,
        instructions: &mut Vec<ShaderInstruction>,
        start: usize,
        encoding: u64,
        next_temporary: &mut u16,
    ) -> Result<(), MaxwellShaderTranslationError> {
        let mut index = start;
        while index < instructions.len() {
            let instruction = &instructions[index];
            let op = instruction.operation();
            let get = |r: &ShaderRegister| self.values.get(r).copied();
            let mut carry_value = None;
            let mut freeze = None;
            let value = if instruction.predicate() != ShaderPredicate::Always {
                None
            } else {
                match op {
                    ShaderOperation::MoveImmediate32 { bits, .. } => Some(self.offset(
                        Bound::Constant(*bits),
                        bits.trailing_zeros(),
                        Some(*bits),
                    )),
                    ShaderOperation::Move32 { source, .. } => get(source),
                    ShaderOperation::LoadConstantBuffer32 {
                        binding,
                        byte_offset,
                        ..
                    } => Some(Value::BaseWord(*binding, *byte_offset)),
                    ShaderOperation::LoadComputeBuiltin32 {
                        builtin, component, ..
                    } => match builtin {
                        nixe_gpu::ShaderComputeBuiltin::LocalInvocationId => {
                            Some(self.offset(Bound::Local(*component), 0, None))
                        }
                        nixe_gpu::ShaderComputeBuiltin::WorkgroupId => {
                            Some(self.offset(Bound::Group(*component), 0, None))
                        }
                        _ => None,
                    },
                    ShaderOperation::Add32 {
                        left,
                        right,
                        scalar_type: ShaderScalarType::Unsigned32 | ShaderScalarType::Signed32,
                        ..
                    } => match (get(left), get(right)) {
                        (Some(Value::Offset(a)), Some(Value::Offset(b))) => Some(self.offset(
                            Bound::Add(a.node, b.node),
                            a.alignment.min(b.alignment),
                            a.constant.zip(b.constant).map(|(a, b)| a.wrapping_add(b)),
                        )),
                        _ => None,
                    },
                    ShaderOperation::ShiftLeft32 {
                        value,
                        amount,
                        wrap,
                        ..
                    } => match (get(value), get(amount)) {
                        (
                            Some(Value::Offset(a)),
                            Some(Value::Offset(Offset {
                                constant: Some(mut amount),
                                ..
                            })),
                        ) => {
                            if *wrap {
                                amount &= 31;
                            }
                            if amount >= 32 {
                                Some(self.offset(Bound::Constant(0), 32, Some(0)))
                            } else {
                                Some(self.offset(
                                    Bound::Shift(a.node, amount),
                                    (a.alignment + amount).min(32),
                                    a.constant.map(|v| v.wrapping_shl(amount)),
                                ))
                            }
                        }
                        _ => None,
                    },
                    ShaderOperation::AddCarry32 {
                        left,
                        right,
                        carry_in,
                        ..
                    } => {
                        let pair = (get(left), get(right));
                        if carry_in.is_none() {
                            let base = match pair {
                                (Some(Value::BaseWord(b, o)), Some(Value::Offset(v))) => {
                                    Some((b, o, v, *right))
                                }
                                (Some(Value::Offset(v)), Some(Value::BaseWord(b, o))) => {
                                    Some((b, o, v, *left))
                                }
                                _ => None,
                            };
                            if let Some((constant_buffer, byte_offset, offset, source)) = base {
                                let frozen = allocate_shader_temporary(
                                    MaxwellShaderStage::Compute,
                                    instruction.source().byte_offset(),
                                    encoding,
                                    "global pointer offset temporary overflow",
                                    next_temporary,
                                )?;
                                let pointer = Pointer {
                                    constant_buffer,
                                    byte_offset,
                                    offset,
                                    frozen,
                                };
                                freeze = Some((frozen, source));
                                carry_value = Some(Value::Carry(pointer));
                                Some(Value::Low(pointer))
                            } else {
                                None
                            }
                        } else if let Some(Value::Carry(pointer)) =
                            carry_in.and_then(|r| self.values.get(&r).copied())
                        {
                            let high = match pair {
                                (
                                    Some(Value::BaseWord(b, o)),
                                    Some(Value::Offset(Offset {
                                        constant: Some(0), ..
                                    })),
                                )
                                | (
                                    Some(Value::Offset(Offset {
                                        constant: Some(0), ..
                                    })),
                                    Some(Value::BaseWord(b, o)),
                                ) => Some((b, o)),
                                _ => None,
                            };
                            if high
                                == pointer
                                    .byte_offset
                                    .checked_add(4)
                                    .map(|o| (pointer.constant_buffer, o))
                            {
                                Some(Value::High(pointer))
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    }
                    _ => None,
                }
            };
            // Any write, including a conditional write, invalidates provenance
            // unless this transfer function has proved its replacement value.
            op.visit_destination_registers(|r| {
                self.values.remove(&r);
            });
            if let Some(value) = value {
                let mut first = true;
                op.visit_destination_registers(|r| {
                    if first {
                        self.values.insert(r, value);
                        first = false;
                    }
                });
            }
            if let ShaderOperation::AddCarry32 { carry_out, .. } = op
                && let Some(value) = carry_value
            {
                self.values.insert(*carry_out, value);
            }
            if let Some((destination, source)) = freeze {
                instructions.insert(
                    index,
                    ShaderInstruction::new(
                        instruction.source(),
                        instruction.predicate(),
                        ShaderOperation::Move32 {
                            destination,
                            source,
                            scalar_type: ShaderScalarType::Unsigned32,
                        },
                    ),
                );
                index += 1;
            }
            index += 1;
        }
        Ok(())
    }

    pub(super) fn store(
        &mut self,
        offset: u32,
        encoding: u64,
        register_count: u8,
        next: &mut u16,
    ) -> Result<Vec<ShaderOperation>, MaxwellShaderTranslationError> {
        let stage = MaxwellShaderStage::Compute;
        let error = |detail| MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail,
        };
        if self.control_flow || decode_predicate(encoding) != ShaderPredicate::Always {
            return Err(error(
                "global stores across control flow require pointer merge analysis",
            ));
        }
        if (encoding >> 48) & 7 != 4 || encoding & (1 << 45) == 0 {
            return Err(error(
                "STG requires scalar 32-bit data and a 64-bit address",
            ));
        }
        if encoding & ((3 << 46) | (1 << 44)) != 0 {
            return Err(error(
                "STG cache modifier or reserved field is not supported",
            ));
        }
        let displacement = ((encoding >> 20) & 0xff_ffff) as u32;
        if displacement & 0x80_0003 != 0 {
            return Err(error(
                "STG requires a nonnegative word-aligned displacement",
            ));
        }
        let address = (encoding >> 8) as u8;
        validate_register_range(stage, offset, encoding, address, 2, register_count)?;
        let (Some(Value::Low(pointer)), Some(Value::High(high))) = (
            self.values.get(&ShaderRegister::new(u16::from(address))),
            self.values
                .get(&ShaderRegister::new(u16::from(address) + 1)),
        ) else {
            return Err(error(
                "STG address has no proven constant-buffer base and dispatch-bounded offset",
            ));
        };
        if pointer != high || pointer.offset.alignment < 2 {
            return Err(error(
                "STG address halves or relative alignment are not proven compatible",
            ));
        }
        let pointer = *pointer;
        let index = if let Some(index) = self.bindings.buffers.iter().position(|b| {
            b.constant_buffer == pointer.constant_buffer && b.byte_offset == pointer.byte_offset
        }) {
            index
        } else {
            let index = self.bindings.buffers.len();
            let binding = u8::try_from(index + 32)
                .map_err(|_| MaxwellShaderTranslationError::ResourceBindingExhausted)?;
            self.bindings.buffers.push(GlobalBufferBinding {
                constant_buffer: pointer.constant_buffer,
                byte_offset: pointer.byte_offset,
                binding,
                accesses: Vec::new(),
            });
            index
        };
        let buffer = &mut self.bindings.buffers[index];
        buffer.accesses.push((pointer.offset.node, displacement));
        let binding = buffer.binding;
        let mut temporary =
            || allocate_shader_temporary(stage, offset, encoding, "STG temporary overflow", next);
        let shift = temporary()?;
        let word_index = temporary()?;
        let mut ops = vec![
            ShaderOperation::MoveImmediate32 {
                destination: shift,
                bits: 2,
                scalar_type: ShaderScalarType::Unsigned32,
            },
            ShaderOperation::ShiftRightLogical32 {
                destination: word_index,
                value: pointer.frozen,
                amount: shift,
                wrap: false,
            },
        ];
        if displacement != 0 {
            ops.push(ShaderOperation::MoveImmediate32 {
                destination: shift,
                bits: displacement / 4,
                scalar_type: ShaderScalarType::Unsigned32,
            });
            // Dividing first preserves an address carry above bit 31. The sum
            // of these word offsets cannot overflow u32.
            ops.push(ShaderOperation::Add32 {
                destination: word_index,
                left: word_index,
                right: shift,
                scalar_type: ShaderScalarType::Unsigned32,
                float_control: ShaderFloatControl::PRECISE,
            });
        }
        let source = if encoding as u8 == 0xff {
            let source = temporary()?;
            ops.push(ShaderOperation::MoveImmediate32 {
                destination: source,
                bits: 0,
                scalar_type: ShaderScalarType::Unsigned32,
            });
            source
        } else {
            validate_register_range(stage, offset, encoding, encoding as u8, 1, register_count)?;
            ShaderRegister::new(u16::from(encoding as u8))
        };
        ops.push(ShaderOperation::StoreStorageBuffer32 {
            source,
            binding,
            word_index,
        });
        Ok(ops)
    }
}
