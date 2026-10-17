//! Relocate proven constant-buffer-based 64-bit pointers into storage buffers.
//! Numeric address arithmetic remains in IR when otherwise observable. Only
//! global accesses with a proven base become a host buffer access. Generic
//! pointers use inferred dispatch bounds; NVN descriptors provide runtime size.
//!
//! LDG/STG fields: public UAM GM107 emitSTG/emitLDSTs/emitLDSTc.
//! https://github.com/devkitPro/uam/blob/master/mesa-imported/codegen/nv50_ir_emit_gm107.cpp
use super::decode::{allocate_shader_temporary, decode_predicate, validate_register_range};
use super::error::MaxwellShaderTranslationError;
use crate::MaxwellShaderStage;
use crate::shader::decoded::Register as ShaderRegister;
use crate::shader::decoded::{Instruction as ShaderInstruction, Operation as ShaderOperation};
use nixe_gpu::{ShaderFloatControl, ShaderPredicate, ShaderScalarType};
use std::collections::BTreeMap;

mod facts;
mod flow;
use facts::Facts;
pub(super) use flow::lower;

#[cfg(test)]
mod tests;

pub(super) const fn is_load(encoding: u64) -> bool {
    (encoding >> 48) & 0xfff8 == 0xeed0
}

pub(super) const fn is_store(encoding: u64) -> bool {
    (encoding >> 48) & 0xfff8 == 0xeed8
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum Bound {
    Unknown,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Value {
    Offset(Offset),
    BaseWord(u8, u32),
    Low(Pointer),
    High(Pointer),
    Carry(Pointer),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct GlobalBufferBinding {
    pub constant_buffer: u8,
    pub byte_offset: u32,
    pub binding: u32,
    pub read: bool,
    pub write: bool,
    pub runtime_extent: bool,
    accesses: Vec<(usize, u32)>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct GlobalBufferBindings {
    pub buffers: Vec<GlobalBufferBinding>,
    bounds: Vec<Bound>,
}

impl GlobalBufferBinding {
    /// Only the retiring renderer calls this after checked oracle export.
    pub(crate) fn oracle_binding(&self) -> u8 {
        u8::try_from(self.binding).expect("oracle resource width was checked before export")
    }
}
impl GlobalBufferBindings {
    /// Linear-time evaluation of the shared bound DAG, once per dispatch.
    /// A wrapping operation conservatively expands to the entire u32 domain;
    /// it never truncates a possible access interval.
    pub(crate) fn byte_extents(&self, groups: [u32; 3], size: [u32; 3]) -> Vec<u64> {
        let mut values = Vec::<u32>::with_capacity(self.bounds.len());
        for bound in &self.bounds {
            let value = match *bound {
                Bound::Unknown => u32::MAX,
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
                    .expect("binding is introduced by a global access")
            })
            .collect()
    }
}

pub(super) struct GlobalMemory {
    stage: MaxwellShaderStage,
    values: Facts,
    pub bindings: GlobalBufferBindings,
    nodes: BTreeMap<Bound, usize>,
    frozen: BTreeMap<(u32, usize), ShaderRegister>,
}

impl Default for GlobalMemory {
    fn default() -> Self {
        Self::new(MaxwellShaderStage::Compute)
    }
}

pub(super) fn decode_access(
    stage: MaxwellShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
) -> Result<ShaderOperation, MaxwellShaderTranslationError> {
    let load = is_load(encoding);
    let error = |detail| MaxwellShaderTranslationError::UnsupportedSemanticDetail {
        stage,
        instruction_offset: offset,
        encoding,
        detail,
    };
    if !load && stage != MaxwellShaderStage::Compute {
        return Err(error("graphics global stores are not implemented"));
    }
    let width = (encoding >> 48) & 7;
    let words: u8 = match width {
        4 => 1,
        5 => 2,
        6 => 4,
        7 if load => 4,
        _ => return Err(error("global access element width is not implemented")),
    };
    if encoding & (1 << 45) == 0 {
        return Err(error("global access requires a 64-bit address"));
    }
    // LDG cache policies alter caching, not the value or resource identity.
    // Host storage loads execute under the dispatch's visibility barriers.
    if encoding & (1 << 44) != 0 || (!load && encoding & (3 << 46) != 0) {
        return Err(error(
            "global access cache modifier or reserved field is not supported",
        ));
    }
    let displacement = ((encoding >> 20) & 0xff_ffff) as u32;
    if displacement & 0x80_0003 != 0 {
        return Err(error(
            "global access requires a nonnegative word-aligned displacement",
        ));
    }
    let address = (encoding >> 8) as u8;
    validate_register_range(stage, offset, encoding, address, 2, register_count)?;
    if !address.is_multiple_of(2) {
        return Err(error(
            "64-bit global address requires an even register pair",
        ));
    }
    let raw = encoding as u8;
    let data = if raw == 255 {
        Box::default()
    } else {
        validate_register_range(stage, offset, encoding, raw, words, register_count)?;
        if !raw.is_multiple_of(words) {
            return Err(error(
                "global access data registers are not naturally aligned",
            ));
        }
        (0..words)
            .map(|word| ShaderRegister::new(u32::from(raw) + u32::from(word)))
            .collect()
    };
    Ok(ShaderOperation::GlobalMemory {
        encoding,
        address: [
            ShaderRegister::new(u32::from(address)),
            ShaderRegister::new(u32::from(address) + 1),
        ],
        data,
        load,
    })
}

impl GlobalMemory {
    pub(super) fn new(stage: MaxwellShaderStage) -> Self {
        Self {
            stage,
            values: Facts::default(),
            nodes: BTreeMap::from([(Bound::Unknown, 0)]),
            frozen: BTreeMap::new(),
            bindings: GlobalBufferBindings {
                buffers: Vec::new(),
                bounds: vec![Bound::Unknown],
            },
        }
    }
}

impl GlobalMemory {
    pub(super) fn forget(&mut self, register: ShaderRegister) {
        self.values.forget(register);
    }

    fn offset(&mut self, bound: Bound, alignment: u32, constant: Option<u32>) -> Value {
        let node = *self.nodes.entry(bound.clone()).or_insert_with(|| {
            let node = self.bindings.bounds.len();
            self.bindings.bounds.push(bound);
            node
        });
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
        next_temporary: &mut u32,
    ) -> Result<(), MaxwellShaderTranslationError> {
        let mut index = start;
        let mut ordinal = 0;
        while index < instructions.len() {
            let instruction = &instructions[index];
            let op = instruction.operation();
            let predicate = instruction.predicate();
            if predicate == ShaderPredicate::Never {
                index += 1;
                ordinal += 1;
                continue;
            }
            let get = |r: &ShaderRegister| {
                self.values
                    .get(*r, predicate)
                    .or(Some(Value::Offset(Offset {
                        node: 0,
                        alignment: 0,
                        constant: None,
                    })))
            };
            let mut carry_value = None;
            let mut freeze = None;
            let value = {
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
                    } => Some(Value::BaseWord(
                        u8::try_from(*binding).expect("decoded constant-buffer ISA slot"),
                        *byte_offset,
                    )),
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
                                let key = (instruction.source().byte_offset(), ordinal);
                                let frozen = if let Some(&frozen) = self.frozen.get(&key) {
                                    frozen
                                } else {
                                    let frozen = allocate_shader_temporary(
                                        self.stage,
                                        key.0,
                                        encoding,
                                        "global pointer offset temporary overflow",
                                        next_temporary,
                                    )?;
                                    self.frozen.insert(key, frozen);
                                    frozen
                                };
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
                            carry_in.and_then(|r| self.values.get(r, predicate))
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
            let mut first = true;
            op.visit_destination_registers(|r| {
                let fact = if first { value } else { None };
                self.values.write(r, predicate, fact);
                first = false;
            });
            if let ShaderOperation::AddCarry32 { carry_out, .. } = op {
                self.values.write(*carry_out, predicate, carry_value);
            }
            self.values.invalidate_predicate_writes(op);
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
            ordinal += 1;
        }
        Ok(())
    }

    pub(super) fn load(
        &mut self,
        offset: u32,
        encoding: u64,
        register_count: u8,
        next: &mut u32,
    ) -> Result<Vec<ShaderOperation>, MaxwellShaderTranslationError> {
        self.access(offset, encoding, register_count, next, true)
    }

    pub(super) fn store(
        &mut self,
        offset: u32,
        encoding: u64,
        register_count: u8,
        next: &mut u32,
    ) -> Result<Vec<ShaderOperation>, MaxwellShaderTranslationError> {
        self.access(offset, encoding, register_count, next, false)
    }

    fn access(
        &mut self,
        offset: u32,
        encoding: u64,
        register_count: u8,
        next: &mut u32,
        load: bool,
    ) -> Result<Vec<ShaderOperation>, MaxwellShaderTranslationError> {
        let stage = self.stage;
        decode_access(stage, offset, encoding, register_count)?;
        let error = |detail| MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail,
        };
        let words = match (encoding >> 48) & 7 {
            4 => 1,
            5 => 2,
            _ => 4,
        };
        let displacement = ((encoding >> 20) & 0xff_ffff) as u32;
        let address = (encoding >> 8) as u8;
        let predicate = decode_predicate(encoding);
        let halves = (
            self.values
                .get(ShaderRegister::new(u32::from(address)), predicate),
            self.values
                .get(ShaderRegister::new(u32::from(address) + 1), predicate),
        );
        let mut prefix = Vec::new();
        let pointer = match halves {
            (Some(Value::Low(pointer)), Some(Value::High(high)))
                if pointer == high
                    && (pointer.offset.alignment >= 2
                        || (pointer.constant_buffer == 0
                            && (0x100..0x700).contains(&pointer.byte_offset)
                            && pointer.byte_offset.is_multiple_of(16))) =>
            {
                pointer
            }
            (
                Some(Value::BaseWord(cb, byte_offset)),
                Some(Value::BaseWord(high_cb, high_offset)),
            ) if cb == high_cb && byte_offset.checked_add(4) == Some(high_offset) => {
                // A direct pair of consecutive constant-buffer words is a base
                // pointer with zero relative offset, including aliased LDG Rd,[Rd].
                let Value::Offset(relative) = self.offset(Bound::Constant(0), 32, Some(0)) else {
                    unreachable!();
                };
                let frozen = allocate_shader_temporary(
                    stage,
                    offset,
                    encoding,
                    "global pointer temporary overflow",
                    next,
                )?;
                prefix.push(ShaderOperation::MoveImmediate32 {
                    destination: frozen,
                    bits: 0,
                    scalar_type: ShaderScalarType::Unsigned32,
                });
                Pointer {
                    constant_buffer: cb,
                    byte_offset,
                    offset: relative,
                    frozen,
                }
            }
            _ => {
                return Err(error(
                    "global address has no proven constant-buffer base and aligned bounded offset",
                ));
            }
        };
        let index = if let Some(index) = self.bindings.buffers.iter().position(|b| {
            b.constant_buffer == pointer.constant_buffer && b.byte_offset == pointer.byte_offset
        }) {
            index
        } else {
            let index = self.bindings.buffers.len();
            let binding = index as u32 + 32;
            self.bindings.buffers.push(GlobalBufferBinding {
                constant_buffer: pointer.constant_buffer,
                byte_offset: pointer.byte_offset,
                binding,
                read: false,
                write: false,
                runtime_extent: false,
                accesses: Vec::new(),
            });
            index
        };
        let mut nodes = vec![pointer.offset.node];
        let mut runtime_extent = false;
        while let Some(node) = nodes.pop() {
            match self.bindings.bounds[node] {
                Bound::Unknown => {
                    runtime_extent = true;
                    break;
                }
                Bound::Add(a, b) => nodes.extend([a, b]),
                Bound::Shift(value, _) => nodes.push(value),
                _ => {}
            }
        }
        if runtime_extent
            && !(pointer.constant_buffer == 0
                && (0x100..0x700).contains(&pointer.byte_offset)
                && pointer.byte_offset.is_multiple_of(16))
        {
            return Err(error(
                "global access has no inferred bound or NVN size descriptor",
            ));
        }
        let buffer = &mut self.bindings.buffers[index];
        buffer.runtime_extent |= runtime_extent;
        buffer.read |= load;
        buffer.write |= !load;
        buffer
            .accesses
            .push((pointer.offset.node, displacement + 4 * u32::from(words - 1)));
        let binding = buffer.binding;
        let mut temporary = || {
            allocate_shader_temporary(
                stage,
                offset,
                encoding,
                "global access temporary overflow",
                next,
            )
        };
        let shift = temporary()?;
        let word_index = temporary()?;
        let mut ops = prefix;
        ops.extend([
            ShaderOperation::MoveImmediate32 {
                destination: shift,
                bits: 2,
                scalar_type: ShaderScalarType::Unsigned32,
            },
            ShaderOperation::ShiftRight32 {
                destination: word_index,
                value: pointer.frozen,
                amount: shift,
                wrap: false,
                signed: false,
                reverse: false,
            },
        ]);
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
        let raw = encoding as u8;
        if raw != 255 {
            validate_register_range(stage, offset, encoding, raw, words, register_count)?;
            if !raw.is_multiple_of(words) {
                return Err(error(
                    "global access data registers are not naturally aligned",
                ));
            }
        }
        if words > 1 {
            ops.push(ShaderOperation::MoveImmediate32 {
                destination: shift,
                bits: 1,
                scalar_type: ShaderScalarType::Unsigned32,
            });
        }
        for word in 0..words {
            let register = if raw == 255 {
                let register = temporary()?;
                if !load {
                    ops.push(ShaderOperation::MoveImmediate32 {
                        destination: register,
                        bits: 0,
                        scalar_type: ShaderScalarType::Unsigned32,
                    });
                }
                register
            } else {
                ShaderRegister::new(u32::from(raw) + u32::from(u16::from(word)))
            };
            ops.push(if load {
                ShaderOperation::LoadStorageBuffer32 {
                    destination: register,
                    binding,
                    word_index,
                }
            } else {
                ShaderOperation::StoreStorageBuffer32 {
                    source: register,
                    binding,
                    word_index,
                }
            });
            if word + 1 < words {
                ops.push(ShaderOperation::Add32 {
                    destination: word_index,
                    left: word_index,
                    right: shift,
                    scalar_type: ShaderScalarType::Unsigned32,
                    float_control: ShaderFloatControl::PRECISE,
                });
            }
        }
        Ok(ops)
    }
}
