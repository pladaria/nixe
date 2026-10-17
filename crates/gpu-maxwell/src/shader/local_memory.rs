//! Promote statically addressed per-invocation local words to SSA variables.
//! Overlapping scalar/vector spills share the same word identities; normal SSA
//! construction supplies joins, loops, predication and undefined initial values.
//! An indexed access makes the entire frame private memory, preserving aliasing.
//! https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/cla097sph.h
//! https://github.com/eden-emulator/mirror/blob/master/src/shader_recompiler/frontend/maxwell/translate/impl/load_store_local_shared.cpp
use super::{
    binary::{MaxwellShaderBinary, MaxwellShaderMetadata},
    decode::{allocate_shader_temporary, validate_register_range},
    decoded::{Operation, Register},
    error::MaxwellShaderTranslationError,
};
use nixe_gpu::ShaderScalarType;
use std::collections::BTreeMap;

pub(super) enum LocalMemory {
    Promoted(BTreeMap<u32, Register>),
    Private(u32),
}
impl Default for LocalMemory {
    fn default() -> Self {
        Self::Promoted(BTreeMap::new())
    }
}

pub(super) const fn is_access(encoding: u64) -> bool {
    matches!((encoding >> 48) & 0xfff8, 0xef40 | 0xef50)
}

impl LocalMemory {
    pub(super) fn new(
        binary: &MaxwellShaderBinary,
        flow: &super::control_flow::ShaderControlFlow,
    ) -> Self {
        let dynamic = binary.bundles().iter().any(|bundle| {
            bundle
                .instructions
                .iter()
                .enumerate()
                .any(|(slot, &encoding)| {
                    flow.reachable
                        .contains(&(bundle.offset + 8 + slot as u32 * 8))
                        && is_access(encoding)
                        && (encoding >> 8) & 255 != 255
                })
        });
        if dynamic {
            Self::Private(0)
        } else {
            Self::default()
        }
    }
    pub(super) fn private_bytes(&self) -> u32 {
        match self {
            Self::Private(bytes) => *bytes,
            Self::Promoted(_) => 0,
        }
    }

    pub(super) fn decode(
        &mut self,
        binary: &MaxwellShaderBinary,
        pc: u32,
        encoding: u64,
        registers: u8,
        next: &mut u32,
    ) -> Result<Vec<Operation>, MaxwellShaderTranslationError> {
        let fail = |detail| MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage: binary.stage(),
            instruction_offset: pc,
            encoding,
            detail,
        };
        let base = ((encoding >> 8) & 255) as u8;
        if base != 255 && matches!(self, Self::Promoted(_)) {
            return Err(fail("local memory address is not a static spill offset"));
        }
        if (encoding >> 44) & 15 != 0 {
            return Err(fail("local memory access qualifier is not represented"));
        }
        let count = match (encoding >> 48) & 7 {
            4 => 1,
            5 => 2,
            6 => 4,
            _ => {
                return Err(fail(
                    "local subword or reserved access width is not represented",
                ));
            }
        };
        let offset = ((encoding >> 20) & 0xff_ffff) as u32;
        if base == 255 && offset & 3 != 0 {
            return Err(fail("local spill address is not word aligned"));
        }
        let MaxwellShaderMetadata::Graphics(header) = binary.metadata else {
            return Err(fail(
                "compute local memory requires its QMD allocation size",
            ));
        };
        let size = header.local_memory_bytes();
        if size == 0 || (base == 255 && u64::from(offset) + u64::from(count) * 4 > size) {
            return Err(fail(
                "local access exceeds the declared SPH memory allocation",
            ));
        }
        let reg = encoding as u8;
        if !reg.is_multiple_of(count) {
            return Err(fail("local vector data register is not naturally aligned"));
        }
        let load = (encoding >> 52) & 15 == 4;
        if reg == 255 && load {
            return Ok(vec![Operation::Nop]);
        }
        if reg != 255 {
            validate_register_range(binary.stage(), pc, encoding, reg, count, registers)?;
        }
        if let Self::Private(bytes) = self {
            *bytes = u32::try_from(size.next_multiple_of(4)).map_err(|_| {
                fail("private local allocation exceeds the 32-bit byte address space")
            })?;
            return private_access(
                binary, pc, encoding, registers, next, base, reg, count, offset, load,
            );
        }
        let Self::Promoted(words) = self else {
            unreachable!()
        };
        let mut operations = Vec::with_capacity(usize::from(count));
        for word in 0..count {
            let address = offset + u32::from(word) * 4;
            let slot = if let Some(&slot) = words.get(&address) {
                slot
            } else {
                let slot = allocate_shader_temporary(
                    binary.stage(),
                    pc,
                    encoding,
                    "local spill temporary namespace overflow",
                    next,
                )?;
                words.insert(address, slot);
                slot
            };
            let data = Register::new(u32::from(reg) + u32::from(word));
            operations.push(if reg == 255 {
                Operation::MoveImmediate32 {
                    destination: slot,
                    bits: 0,
                    scalar_type: ShaderScalarType::Unsigned32,
                }
            } else {
                Operation::Move32 {
                    destination: if load { data } else { slot },
                    source: if load { slot } else { data },
                    scalar_type: ShaderScalarType::Unsigned32,
                }
            });
        }
        Ok(operations)
    }
}

#[allow(clippy::too_many_arguments)]
fn private_access(
    binary: &MaxwellShaderBinary,
    pc: u32,
    encoding: u64,
    registers: u8,
    next: &mut u32,
    base: u8,
    reg: u8,
    count: u8,
    offset: u32,
    load: bool,
) -> Result<Vec<Operation>, MaxwellShaderTranslationError> {
    if base != 255 {
        validate_register_range(binary.stage(), pc, encoding, base, 1, registers)?;
    }
    let mut temporary = || {
        allocate_shader_temporary(
            binary.stage(),
            pc,
            encoding,
            "private memory address temporary overflow",
            next,
        )
    };
    let mut ops = Vec::new();
    let address = temporary()?;
    if base == 255 {
        ops.push(Operation::MoveImmediate32 {
            destination: address,
            bits: offset,
            scalar_type: ShaderScalarType::Unsigned32,
        });
    } else {
        let displacement = temporary()?;
        let signed = ((offset << 8) as i32 >> 8) as u32;
        ops.push(Operation::MoveImmediate32 {
            destination: displacement,
            bits: signed,
            scalar_type: ShaderScalarType::Unsigned32,
        });
        ops.push(Operation::Add32 {
            destination: address,
            left: Register::new(u32::from(base)),
            right: displacement,
            scalar_type: ShaderScalarType::Unsigned32,
            float_control: nixe_gpu::ShaderFloatControl::PRECISE,
        });
    }
    let zero = if reg == 255 {
        let zero = temporary()?;
        ops.push(Operation::MoveImmediate32 {
            destination: zero,
            bits: 0,
            scalar_type: ShaderScalarType::Unsigned32,
        });
        Some(zero)
    } else {
        None
    };
    for word in 0..count {
        let element = if word == 0 {
            address
        } else {
            let delta = temporary()?;
            let element = temporary()?;
            ops.push(Operation::MoveImmediate32 {
                destination: delta,
                bits: u32::from(word) * 4,
                scalar_type: ShaderScalarType::Unsigned32,
            });
            ops.push(Operation::Add32 {
                destination: element,
                left: address,
                right: delta,
                scalar_type: ShaderScalarType::Unsigned32,
                float_control: nixe_gpu::ShaderFloatControl::PRECISE,
            });
            element
        };
        let data = zero.unwrap_or(Register::new(u32::from(reg) + u32::from(word)));
        ops.push(if load {
            Operation::LoadLocal32 {
                destination: data,
                address: element,
            }
        } else {
            Operation::StoreLocal32 {
                source: data,
                address: element,
            }
        });
    }
    Ok(ops)
}
