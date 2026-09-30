//! Relocate proven intra-patch ISBE accesses without exposing a fabricated
//! invocation-info value or a host subgroup lane ID to the guest program.
//!
//! The producer constructs ISBE addresses as primitive-slot * input-count +
//! vertex-index. Both slot and count occupy eight bits in invocation-info.
//! We retain that expression symbolically until ISBERD consumes it; only its
//! patch-local index reaches neutral IR. This is instruction semantics, not a
//! match against a shader, address, or fixed instruction sequence.
//! https://github.com/devkitPro/uam/blob/master/mesa-imported/codegen/nv50_ir_lowering_gm107.cpp#L236-L254
//! https://github.com/devkitPro/uam/blob/master/mesa-imported/codegen/nv50_ir_emit_gm107.cpp#L2487-L2515
//! https://envytools.readthedocs.io/en/latest/hw/graph/maxwell/cuda/int.html#multiply-add-xmad
use super::*;

#[cfg(test)]
mod tests;

pub(super) const fn is_supported_family(encoding: u64) -> bool {
    let opcode = (encoding >> 48) as u16;
    opcode == 0xefd0
        || opcode & 0xfffe == 0x3846
        || opcode & 0xfffe == 0x3800
        || opcode & 0xffc0 == 0x5b00
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Index {
    Constant(u32),
    Register(ShaderRegister),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Value {
    Constant(u32),
    Register(ShaderRegister),
    InvocationInfo,
    PrimitiveSlot,
    InputCount,
    PatchBase,
    PatchAddress(Index),
    UpperSlot,
    Handle(ShaderRegister),
    EvaluationLane,
}

#[derive(Default)]
pub(super) struct PatchAddresses {
    values: Option<Box<[Option<Value>; 256]>>,
    live: u16,
    control_flow: bool,
}

impl PatchAddresses {
    fn value(&self, register: ShaderRegister) -> Option<Value> {
        self.values
            .as_ref()
            .and_then(|values| values.get(usize::from(register.index())))
            .copied()
            .flatten()
    }

    fn set(&mut self, register: u8, value: Value) {
        let values = self.values.get_or_insert_with(|| Box::new([None; 256]));
        if values[usize::from(register)].replace(value).is_none() {
            self.live += 1;
        }
    }

    fn remove(&mut self, register: ShaderRegister) {
        if let Some(values) = &mut self.values
            && let Some(value) = values.get_mut(usize::from(register.index()))
            && value.take().is_some()
        {
            self.live -= 1;
        }
    }

    fn get(&self, register: u8) -> Value {
        if register == 0xff {
            return Value::Constant(0);
        }
        let register = ShaderRegister::new(u16::from(register));
        self.value(register).unwrap_or(Value::Register(register))
    }

    fn abstracted(&self, register: u8) -> bool {
        !matches!(self.get(register), Value::Register(_)) && register != 0xff
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn lower(
        &mut self,
        stage: MaxwellThreeDShaderStage,
        offset: u32,
        encoding: u64,
        register_count: u8,
        next_temporary: &mut u16,
        inputs: &mut Vec<ShaderInterfaceElement>,
    ) -> Result<Option<Vec<ShaderOperation>>, MaxwellShaderTranslationError> {
        if !matches!(
            stage,
            MaxwellThreeDShaderStage::TessellationInit | MaxwellThreeDShaderStage::Tessellation
        ) {
            return Ok(None);
        }
        let error = |detail| MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail,
        };
        let opcode = (encoding >> 48) as u16;
        let dst = encoding as u8;
        let a = (encoding >> 8) as u8;
        let b = (encoding >> 20) as u8;
        let c = (encoding >> 39) as u8;
        let system = tessellation::is_system_register_read(encoding);
        let selector = ((encoding >> 20) & 0xff) as u8;
        let isberd = opcode == 0xefd0;
        let ald = is_attribute_load(encoding);
        let lop = matches!(opcode & 0xfffe, 0x3846);
        let bfe = opcode & 0xfffe == 0x3800;
        let xmad = opcode & 0xffc0 == 0x5b00;
        let intercepted = (system
            && (selector == 0x1d
                || (selector == 0 && stage == MaxwellThreeDShaderStage::Tessellation)))
            || isberd
            || ald
            || (lop && self.abstracted(a))
            || (bfe && self.abstracted(a))
            || (xmad && [a, b, c].into_iter().any(|r| self.abstracted(r)));
        if !intercepted {
            return Ok(None);
        }
        if self.control_flow {
            return Err(error(
                "patch address across control-flow requires address merge analysis",
            ));
        }
        if decode_predicate(encoding) != ShaderPredicate::Always {
            return Err(error(
                "conditional patch-address operation requires address control-flow analysis",
            ));
        }
        let count = if ald {
            (((encoding >> 47) & 3) + 1) as u8
        } else {
            1
        };
        validate_register_range(stage, offset, encoding, dst, count, register_count)?;
        let source_count = if xmad {
            3
        } else {
            usize::from(lop || bfe || isberd)
        };
        for register in [a, b, c].into_iter().take(source_count) {
            if register != 0xff {
                validate_register_range(stage, offset, encoding, register, 1, register_count)?;
            }
        }
        let mut operations = Vec::new();
        let mut temporary = || {
            allocate_shader_temporary(
                stage,
                offset,
                encoding,
                "patch-address temporary exceeds register bank",
                next_temporary,
            )
        };
        let mut freeze = |index: Index| -> Result<ShaderRegister, MaxwellShaderTranslationError> {
            if let Index::Register(register) = index
                && register.index() >= u16::from(register_count)
            {
                return Ok(register);
            }
            let destination = temporary()?;
            operations.push(match index {
                Index::Constant(bits) => ShaderOperation::MoveImmediate32 {
                    destination,
                    bits,
                    scalar_type: ShaderScalarType::Unsigned32,
                },
                Index::Register(source) => ShaderOperation::Move32 {
                    destination,
                    source,
                    scalar_type: ShaderScalarType::Unsigned32,
                },
            });
            Ok(destination)
        };
        let result = if system {
            if encoding & !0xffff_0000_0fff_00ff != 0 {
                return Err(malformed(
                    stage,
                    offset,
                    encoding,
                    "S2R reserved field is nonzero",
                ));
            }
            Some(if selector == 0x1d {
                Value::InvocationInfo
            } else {
                Value::EvaluationLane
            })
        } else if lop {
            // LOP.AND immediate without predicate/condition-code side effects.
            if opcode != 0x3847 || encoding & 0x0000_ff80_0000_0000 != 0 {
                return Err(error("patch address LOP requires plain immediate AND"));
            }
            match (self.get(a), (encoding >> 20) & 0x7ffff) {
                (Value::InvocationInfo, 0xff) => Some(Value::PrimitiveSlot),
                _ => {
                    return Err(error(
                        "patch address LOP does not isolate the primitive-slot field",
                    ));
                }
            }
        } else if bfe {
            if opcode != 0x3800 || encoding & 0x0000_ff80_0000_0000 != 0 {
                return Err(error(
                    "patch address BFE requires unsigned immediate extraction without flags",
                ));
            }
            match (self.get(a), (encoding >> 20) & 0x7ffff) {
                (Value::InvocationInfo, 0x810) => Some(Value::InputCount),
                (Value::InvocationInfo, 0x800) => Some(Value::PrimitiveSlot),
                _ => {
                    return Err(error(
                        "patch address BFE uses an unrepresented invocation-info field",
                    ));
                }
            }
        } else if xmad {
            // Register XMAD: half selection, product shift, merge and CBCC.
            // Signed half-products, carry flags and other C modes stay explicit.
            let allowed = 0xfffc_ffff_0fff_ffff_u64;
            if encoding & !allowed != 0
                || encoding & ((3 << 48) | (1 << 47) | (1 << 38) | (7 << 32)) != 0
            {
                return Err(error(
                    "patch address XMAD uses signed halves, carry or reserved fields",
                ));
            }
            let lhs = half(self.get(a), encoding & (1 << 53) != 0);
            let rhs = half(self.get(b), encoding & (1 << 35) != 0);
            let product = match (lhs, rhs) {
                (Some(Value::Constant(0)), _) | (_, Some(Value::Constant(0))) => {
                    Some(Value::Constant(0))
                }
                (Some(Value::InputCount), Some(Value::PrimitiveSlot))
                | (Some(Value::PrimitiveSlot), Some(Value::InputCount)) => Some(Value::PatchBase),
                (Some(Value::Constant(a)), Some(Value::Constant(b))) => {
                    Some(Value::Constant(a.wrapping_mul(b)))
                }
                _ => None,
            }
            .ok_or_else(|| error("patch address XMAD product cannot be proven patch-relative"))?;
            let product = if encoding & (1 << 36) != 0 {
                shift16(product)
            } else {
                Some(product)
            }
            .ok_or_else(|| error("patch address XMAD shifted product is unrepresented"))?;
            let addend = match (encoding >> 50) & 7 {
                0 => Some(self.get(c)),
                4 => shift16(self.get(b)).and_then(|carry| add(self.get(c), carry)),
                _ => None,
            }
            .ok_or_else(|| error("patch address XMAD addend mode is unrepresented"))?;
            let result = add(product, addend).ok_or_else(|| {
                error("patch address XMAD result cannot be proven patch-relative")
            })?;
            let result = if encoding & (1 << 37) != 0 {
                half(result, false)
                    .and_then(|low| shift16(self.get(b)).and_then(|high| merge(low, high)))
                    .ok_or_else(|| error("patch address XMAD merge is unrepresented"))?
            } else {
                result
            };
            Some(match result {
                Value::PatchAddress(Index::Register(register)) => {
                    Value::PatchAddress(Index::Register(freeze(Index::Register(register))?))
                }
                result => result,
            })
        } else if isberd {
            if encoding & !0xffff_0000_000f_ffff != 0 {
                return Err(error("ISBERD addressing mode is not represented"));
            }
            let index = match self.get(a) {
                Value::PatchBase => Index::Constant(0),
                Value::PatchAddress(index) => index,
                _ => return Err(error("ISBERD source is not a proven current-patch address")),
            };
            Some(Value::Handle(freeze(index)?))
        } else {
            // ALD uses an ISBE handle, not the numeric contents of the register.
            if encoding & ((1 << 30) | (0x3f << 33)) != 0 {
                return Err(malformed(
                    stage,
                    offset,
                    encoding,
                    "ALD reserved field is nonzero",
                ));
            }
            if a != 0xff {
                return Err(error("indexed ALD attribute address is not represented"));
            }
            let output = encoding & (1 << 32) != 0;
            let patch = encoding & (1 << 31) != 0;
            if patch
                && !(output && stage == MaxwellThreeDShaderStage::TessellationInit && c == 0xff)
            {
                return Err(error(
                    "patch ALD requires a direct control-shader output address",
                ));
            }
            let address = ((encoding >> 20) & 0x3ff) as u16;
            let handle = self.get(c);
            let mut loaded = Vec::with_capacity(usize::from(count));
            for lane in 0..count {
                let destination = ShaderRegister::new(u16::from(dst + lane));
                let address = address + u16::from(lane) * 4;
                let operation = if patch {
                    let (location, component) =
                        tessellation::patch_location(address).ok_or_else(|| {
                            malformed(
                                stage,
                                offset,
                                encoding,
                                "ALD.O.P reserved or misaligned patch attribute",
                            )
                        })?;
                    ShaderOperation::LoadPatchOutput {
                        destination,
                        location,
                        component,
                    }
                } else {
                    match handle {
                        Value::EvaluationLane
                            if output
                                && stage == MaxwellThreeDShaderStage::Tessellation
                                && matches!(address, 0x2f0 | 0x2f4) =>
                        {
                            let component = ((address - 0x2f0) / 4) as u8;
                            if !inputs.iter().any(|input| {
                                input.location() == ShaderIoLocation::TessCoord
                                    && input.component() == component
                            }) {
                                inputs.push(interface_element(
                                    ShaderIoLocation::TessCoord,
                                    component,
                                    None,
                                ));
                            }
                            ShaderOperation::LoadInput {
                                destinations: vec![destination].into_boxed_slice(),
                                location: ShaderIoLocation::TessCoord,
                                first_component: component,
                                scalar_type: ShaderScalarType::Float32,
                            }
                        }
                        Value::Handle(vertex) if !output => {
                            let (location, component) =
                                attribute_location(stage, offset, encoding, address)?;
                            ShaderOperation::LoadControlPoint {
                                destination,
                                vertex,
                                output: false,
                                location,
                                component,
                            }
                        }
                        _ => {
                            return Err(error(
                                "ALD handle does not select represented patch input or current domain coordinates",
                            ));
                        }
                    }
                };
                loaded.push(operation);
            }
            for lane in 0..count {
                self.remove(ShaderRegister::new(u16::from(dst + lane)));
            }
            operations.extend(loaded);
            None
        };
        if let Some(result) = result {
            self.set(dst, result);
        }
        Ok(Some(operations))
    }

    /// Ordinary instructions may overwrite dead internal handles, but cannot
    /// observe their numeric representation. Operands are checked before writes.
    pub(super) fn ordinary(
        &mut self,
        stage: MaxwellThreeDShaderStage,
        encoding: u64,
        instructions: &[ShaderInstruction],
    ) -> Result<(), MaxwellShaderTranslationError> {
        if matches!(
            stage,
            MaxwellThreeDShaderStage::TessellationInit | MaxwellThreeDShaderStage::Tessellation
        ) && instructions
            .iter()
            .any(|instruction| matches!(instruction.operation(), ShaderOperation::Branch { .. }))
        {
            self.control_flow = true;
        }
        if self.live == 0 {
            return Ok(());
        }
        for instruction in instructions {
            for source in instruction.operation().source_registers() {
                if self.value(source).is_some() {
                    return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
                        stage,
                        instruction_offset: instruction.source().byte_offset(),
                        encoding,
                        detail: "internal patch address escapes into ordinary shader arithmetic or output",
                    });
                }
            }
            let mut conditional_overwrite = false;
            instruction
                .operation()
                .visit_destination_registers(|destination| {
                    if instruction.predicate() == ShaderPredicate::Always {
                        self.remove(destination);
                    } else if self.value(destination).is_some() {
                        conditional_overwrite = true;
                    }
                });
            if conditional_overwrite {
                return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
                    stage,
                    instruction_offset: instruction.source().byte_offset(),
                    encoding,
                    detail: "conditional overwrite of an internal patch address",
                });
            }
            if matches!(instruction.operation(), ShaderOperation::Branch { .. }) && self.live != 0 {
                return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
                    stage,
                    instruction_offset: instruction.source().byte_offset(),
                    encoding,
                    detail: "patch address across control-flow requires address merge analysis",
                });
            }
        }
        Ok(())
    }
}

fn half(value: Value, high: bool) -> Option<Value> {
    match value {
        Value::Constant(value) => Some(Value::Constant(if high {
            value >> 16
        } else {
            value & 0xffff
        })),
        Value::InputCount | Value::PrimitiveSlot | Value::PatchBase => {
            Some(if high { Value::Constant(0) } else { value })
        }
        Value::UpperSlot => Some(if high {
            Value::PrimitiveSlot
        } else {
            Value::Constant(0)
        }),
        _ => None,
    }
}

fn shift16(value: Value) -> Option<Value> {
    match value {
        Value::Constant(value) => Some(Value::Constant(value << 16)),
        Value::PrimitiveSlot => Some(Value::UpperSlot),
        Value::UpperSlot => Some(Value::Constant(0)),
        _ => None,
    }
}

fn add(a: Value, b: Value) -> Option<Value> {
    match (a, b) {
        (Value::Constant(0), value) | (value, Value::Constant(0)) => Some(value),
        (Value::Constant(a), Value::Constant(b)) => Some(Value::Constant(a.wrapping_add(b))),
        (Value::PatchBase, Value::Register(r)) | (Value::Register(r), Value::PatchBase) => {
            Some(Value::PatchAddress(Index::Register(r)))
        }
        (Value::PatchBase, Value::Constant(c)) | (Value::Constant(c), Value::PatchBase) => {
            Some(Value::PatchAddress(Index::Constant(c)))
        }
        _ => None,
    }
}

fn merge(low: Value, high: Value) -> Option<Value> {
    match (low, high) {
        (Value::Constant(0), value) | (value, Value::Constant(0)) => Some(value),
        (Value::Constant(low), Value::Constant(high)) => Some(Value::Constant(low | high)),
        _ => None,
    }
}
