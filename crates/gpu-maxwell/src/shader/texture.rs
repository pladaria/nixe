//! Maxwell texture instructions and their local resource bindings.

use super::decode::{allocate_shader_temporary, validate_register_range};
use super::error::{MaxwellShaderTranslationError, malformed};
use crate::MaxwellShaderStage;
use nixe_gpu::{
    ShaderBitwiseOperation, ShaderInstruction, ShaderOperation, ShaderPredicate, ShaderRegister,
    ShaderResourceKind, ShaderScalarType, ShaderTextureSampleOutput,
};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct MaxwellTextureResourceBinding {
    pub(super) constant_buffer_byte_offset: u32,
    pub(super) image_binding: u8,
    pub(super) sampler_binding: Option<u8>,
    pub(super) image_kind: ShaderResourceKind,
}

impl MaxwellTextureResourceBinding {
    pub(crate) const fn constant_buffer_byte_offset(self) -> u32 {
        self.constant_buffer_byte_offset
    }
    pub(crate) const fn image_binding(self) -> u8 {
        self.image_binding
    }
    pub(crate) const fn sampler_binding(self) -> Option<u8> {
        self.sampler_binding
    }
    pub(crate) const fn image_kind(self) -> ShaderResourceKind {
        self.image_kind
    }
}

fn texture_binding(
    stage: MaxwellShaderStage,
    offset: u32,
    encoding: u64,
    bindings: &mut BTreeMap<u16, MaxwellTextureResourceBinding>,
    constant_buffer_dword_offset: u16,
    image_kind: ShaderResourceKind,
    sampled: bool,
) -> Result<MaxwellTextureResourceBinding, MaxwellShaderTranslationError> {
    if let Some(binding) = bindings.get_mut(&constant_buffer_dword_offset) {
        if binding.image_kind != image_kind {
            return Err(malformed(
                stage,
                offset,
                encoding,
                "texture instructions reuse one descriptor with contradictory image dimensions",
            ));
        }
        if sampled && binding.sampler_binding.is_none() {
            binding.sampler_binding = Some(binding.image_binding + 1);
        }
        Ok(*binding)
    } else {
        let next_pair = u8::try_from(32 + bindings.len() * 2).map_err(|_| {
            malformed(
                stage,
                offset,
                encoding,
                "texture instructions neutral resource binding space is exhausted",
            )
        })?;
        let binding = MaxwellTextureResourceBinding {
            constant_buffer_byte_offset: u32::from(constant_buffer_dword_offset) * 4,
            image_binding: next_pair,
            sampler_binding: (sampled).then_some(next_pair.checked_add(1).ok_or_else(|| {
                malformed(
                    stage,
                    offset,
                    encoding,
                    "texture instructions neutral resource binding space is exhausted",
                )
            })?),
            image_kind,
        };
        bindings.insert(constant_buffer_dword_offset, binding);
        Ok(binding)
    }
}

pub(super) const fn is_texture_sample(encoding: u64) -> bool {
    encoding & 0xfc38_0000_0000_0000 == 0xc038_0000_0000_0000
}

pub(super) const fn is_texture_query(encoding: u64) -> bool {
    encoding & 0xfff8_0000_0000_0000 == 0xdf48_0000_0000_0000
}

pub(super) struct DecodedTextureQuery {
    pub(super) destination: ShaderRegister,
    pub(super) descriptor: u16,
}

pub(super) fn decode_texture_query(
    stage: MaxwellShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
) -> Result<DecodedTextureQuery, MaxwellShaderTranslationError> {
    // TXQ.DIMENSION selects width/height/depth/mip count with the RGBA mask.
    // Mip count is independent of the input LOD.
    // https://github.com/eden-emulator/mirror/blob/d16735f5b618942136d6ab53466e3be0a382c30a/src/shader_recompiler/frontend/maxwell/translate/impl/texture_query.cpp
    if (encoding >> 22) & 7 != 1 || (encoding >> 31) & 15 != 8 {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "TXQ mode or dimension outputs other than mip count are not implemented",
        });
    }
    let destination = encoding as u8;
    validate_register_range(stage, offset, encoding, destination, 1, register_count)?;
    let descriptor = ((encoding >> 36) & 0x1fff) as u16;
    Ok(DecodedTextureQuery {
        destination: ShaderRegister::new(u16::from(destination)),
        descriptor,
    })
}

pub(super) fn decode_texture_sample(
    stage: MaxwellShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
    bindings: &mut BTreeMap<u16, MaxwellTextureResourceBinding>,
    next_temporary: &mut u16,
    previous: &[ShaderInstruction],
) -> Result<Vec<ShaderOperation>, MaxwellShaderTranslationError> {
    // Full TEX uses consecutive coordinates and packs selected RGBA channels into
    // consecutive destinations, unlike TEXS's separate coordinates/destinations.
    // Its immediate descriptor is the same constant-buffer dword offset as TEXS.
    // https://github.com/envytools/envytools/blob/f102b82381f3f11cee113d16374c87091db039d9/envydis/gm107.c
    // https://github.com/eden-emulator/mirror/blob/d16735f5b618942136d6ab53466e3be0a382c30a/src/shader_recompiler/frontend/maxwell/translate/impl/texture_fetch.cpp
    let texture_type = (encoding >> 28) & 7;
    let lod_mode = (encoding >> 55) & 7;
    let depth_compare = encoding & (1 << 50) != 0;
    if stage != MaxwellShaderStage::Pixel || !matches!(texture_type, 2 | 3) {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "TEX requires fragment 2D or 2D-array sampling",
        });
    }
    if !matches!(lod_mode, 0 | 1 | 3)
        || (depth_compare && lod_mode == 3)
        || encoding & ((1 << 35) | (1 << 58)) != 0
        || (encoding >> 51) & 7 != 7
    {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "TEX bias, nonzero comparison LOD, sparse result, NDV or LOD clamp is not implemented",
        });
    }
    let texel_offset = if encoding & (1 << 54) != 0 {
        let metadata = ((encoding >> 20) & 255) as u16 + u16::from(lod_mode == 3);
        let bits = if metadata == 255 {
            Some(0)
        } else {
            literal_register(previous, ShaderRegister::new(metadata))
        };
        let bits = bits.ok_or(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "TEX AOFFI operand is not a proven shader constant",
        })?;
        Some([
            ((bits & 15) as i32) << 28 >> 28,
            (((bits >> 4) & 15) as i32) << 28 >> 28,
        ])
    } else {
        None
    };
    // NODEP only changes the Maxwell scoreboard dependency, not sampled values;
    // the neutral IR already orders register dependencies and texture operations.
    let coordinate = ((encoding >> 8) & 0xff) as u8;
    let destination = (encoding & 0xff) as u8;
    let mask = ((encoding >> 31) & 15) as u8;
    if mask == 0 {
        return Err(malformed(
            stage,
            offset,
            encoding,
            "TEX has no selected output channels",
        ));
    }
    validate_register_range(
        stage,
        offset,
        encoding,
        coordinate,
        if texture_type == 3 { 3 } else { 2 },
        register_count,
    )?;
    validate_register_range(
        stage,
        offset,
        encoding,
        destination,
        mask.count_ones() as u8,
        register_count,
    )?;
    let mut operations = Vec::new();
    let lod = match lod_mode {
        0 => None,
        1 if !depth_compare => {
            let zero = allocate_shader_temporary(
                stage,
                offset,
                encoding,
                "TEX LOD temporary overflow",
                next_temporary,
            )?;
            operations.push(ShaderOperation::MoveImmediate32 {
                destination: zero,
                bits: 0,
                scalar_type: ShaderScalarType::Float32,
            });
            Some(zero)
        }
        1 => None,
        3 => {
            let source = ((encoding >> 20) & 0xff) as u8;
            validate_register_range(stage, offset, encoding, source, 1, register_count)?;
            Some(ShaderRegister::new(u16::from(source)))
        }
        _ => unreachable!(),
    };
    let image_kind = match (texture_type == 3, depth_compare) {
        (true, true) => ShaderResourceKind::SampledDepthImage2DArray,
        (false, true) => ShaderResourceKind::SampledDepthImage,
        (true, false) => ShaderResourceKind::SampledImage2DArray,
        (false, false) => ShaderResourceKind::SampledImage,
    };
    let binding = texture_binding(
        stage,
        offset,
        encoding,
        bindings,
        ((encoding >> 36) & 0x1fff) as u16,
        image_kind,
        true,
    )?;
    let outputs = (0..4_u8)
        .filter(|component| mask & (1 << component) != 0)
        .enumerate()
        .map(|(index, component)| {
            ShaderTextureSampleOutput::new(
                ShaderRegister::new(u16::from(destination) + index as u16),
                component,
            )
            .expect("selected channel is RGBA")
        })
        .collect();
    let sampler_binding = binding.sampler_binding.expect("TEX reserves a sampler");
    operations.push(if depth_compare {
        // Explicit LZ does not consume a metadata GPR; implicit comparison uses
        // meta_reg directly as depth reference. Nonzero comparison LOD is rejected.
        // https://github.com/eden-emulator/mirror/blob/master/src/shader_recompiler/frontend/maxwell/translate/impl/texture_fetch.cpp
        let reference = ((encoding >> 20) & 255) as u8;
        let reference = reference
            .checked_add(u8::from(texel_offset.is_some()))
            .ok_or_else(|| malformed(stage, offset, encoding, "TEX metadata extends beyond RZ"))?;
        validate_register_range(stage, offset, encoding, reference, 1, register_count)?;
        let base = u16::from(coordinate) + u16::from(texture_type == 3);
        ShaderOperation::SampleDepthCompare2D {
            offset: texel_offset,
            outputs,
            coordinates: [ShaderRegister::new(base), ShaderRegister::new(base + 1)],
            array_index: (texture_type == 3).then_some(ShaderRegister::new(u16::from(coordinate))),
            reference: ShaderRegister::new(u16::from(reference)),
            image_binding: binding.image_binding,
            sampler_binding,
            level_zero: lod_mode == 1,
        }
    } else if texture_type == 3 {
        ShaderOperation::SampleTexture2DArray {
            offset: texel_offset,
            outputs,
            coordinates: [
                ShaderRegister::new(u16::from(coordinate) + 1),
                ShaderRegister::new(u16::from(coordinate) + 2),
            ],
            array_index: ShaderRegister::new(u16::from(coordinate)),
            image_binding: binding.image_binding,
            sampler_binding,
            lod,
        }
    } else {
        ShaderOperation::SampleTexture2D {
            offset: texel_offset,
            outputs,
            coordinates: [
                ShaderRegister::new(u16::from(coordinate)),
                ShaderRegister::new(u16::from(coordinate) + 1),
            ],
            image_binding: binding.image_binding,
            sampler_binding,
            lod,
        }
    });
    Ok(operations)
}

pub(super) const fn is_texture_access_simplified(encoding: u64) -> bool {
    matches!(
        encoding & 0xf600_0000_0000_0000,
        0xd000_0000_0000_0000 | 0xd200_0000_0000_0000
    )
}

pub(super) fn decode_texture_access_simplified(
    stage: MaxwellShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
    bindings: &mut BTreeMap<u16, MaxwellTextureResourceBinding>,
    next_temporary: &mut u16,
) -> Result<Vec<ShaderOperation>, MaxwellShaderTranslationError> {
    // TEXS/TLDS operand fields, dimensionality/LOD selectors, and split destination
    // channel masks follow envytools' pinned public GM107 ISA table:
    // https://github.com/envytools/envytools/blob/f102b82381f3f11cee113d16374c87091db039d9/envydis/gm107.c
    let selector = ((encoding >> 53) & 0xf) as u8;
    let fetch = encoding & 0xf600_0000_0000_0000 == 0xd200_0000_0000_0000;
    // TLDS.LZ.2D has separate integer X/Y source registers and the same
    // split RGBA destinations as TEXS. F16, offsets and other modes need
    // distinct semantics; never reinterpret them as a filtered sample.
    // https://github.com/devkitPro/uam/blob/master/mesa-imported/codegen/nv50_ir_emit_gm107.cpp#L2658-L2764
    if fetch && (selector != 2 || encoding & (1 << 59) == 0) {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "TLDS mode other than F32 2D level-zero without offsets",
        });
    }
    if !fetch
        && (stage != MaxwellShaderStage::Pixel || !matches!(selector, 1 | 2 | 3 | 7 | 8 | 12 | 13))
    {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "TEXS dimensionality or sampling mode is not implemented",
        });
    }
    let primary_destination = (encoding & 0xff) as u8;
    let x_coordinate = ((encoding >> 8) & 0xff) as u8;
    let y_coordinate = ((encoding >> 20) & 0xff) as u8;
    let secondary_destination = ((encoding >> 28) & 0xff) as u8;
    // TEXS/TLDS store a dword offset into SET_BINDLESS_TEXTURE_CONSTANT_BUFFER_SLOT,
    // not a TIC index. The u32 fetched there is the raw TIC/TSC handle. This
    // distinction is visible in yuzu's pinned Maxwell translator:
    // https://github.com/yuzu-emu/yuzu/blob/55bf3dbf5ddaa3f7c1c3efade5553b07499fe289/src/shader_recompiler/frontend/maxwell/translate/impl/texture_fetch_swizzled.cpp#L28-L72
    let constant_buffer_dword_offset = ((encoding >> 36) & 0x1fff) as u16;
    if fetch || matches!(selector, 1 | 2) {
        validate_register_range(stage, offset, encoding, x_coordinate, 1, register_count)?;
        validate_register_range(stage, offset, encoding, y_coordinate, 1, register_count)?;
    } else {
        if !x_coordinate.is_multiple_of(2) {
            return Err(malformed(
                stage,
                offset,
                encoding,
                "TEXS packed coordinate register is misaligned",
            ));
        }
        validate_register_range(stage, offset, encoding, x_coordinate, 2, register_count)?;
        if selector == 13 && !y_coordinate.is_multiple_of(2) {
            return Err(malformed(
                stage,
                offset,
                encoding,
                "TEXS cube direction/LOD register pair is misaligned",
            ));
        }
        validate_register_range(
            stage,
            offset,
            encoding,
            y_coordinate,
            if selector == 13 { 2 } else { 1 },
            register_count,
        )?;
    }

    let channel_selector = ((encoding >> 50) & 0x7) as usize;
    let channels: &[u8] = if secondary_destination == u8::MAX {
        match channel_selector {
            0 => &[0],
            1 => &[1],
            2 => &[2],
            3 => &[3],
            4 => &[0, 1],
            5 => &[0, 3],
            6 => &[1, 3],
            7 => &[2, 3],
            _ => unreachable!(),
        }
    } else {
        match channel_selector {
            0 => &[0, 1, 2],
            1 => &[0, 1, 3],
            2 => &[0, 2, 3],
            3 => &[1, 2, 3],
            4 => &[0, 1, 2, 3],
            _ => {
                return Err(malformed(
                    stage,
                    offset,
                    encoding,
                    "TEXS/TLDS split-destination channel selector is reserved",
                ));
            }
        }
    };
    let half_output = !fetch && encoding & (1 << 59) == 0;
    let primary_count = channels.len().min(2);
    validate_register_range(
        stage,
        offset,
        encoding,
        primary_destination,
        if half_output { 1 } else { primary_count as u8 },
        register_count,
    )?;
    if channels.len() > primary_count {
        validate_register_range(
            stage,
            offset,
            encoding,
            secondary_destination,
            if half_output {
                1
            } else {
                (channels.len() - primary_count) as u8
            },
            register_count,
        )?;
    }

    let image_kind = if matches!(selector, 12 | 13) {
        ShaderResourceKind::SampledImageCube
    } else if matches!(selector, 7 | 8) {
        ShaderResourceKind::SampledImage2DArray
    } else {
        ShaderResourceKind::SampledImage
    };
    let binding = texture_binding(
        stage,
        offset,
        encoding,
        bindings,
        constant_buffer_dword_offset,
        image_kind,
        !fetch,
    )?;
    let mut outputs = Vec::new();
    for (index, &component) in channels.iter().enumerate() {
        let register = if half_output {
            allocate_shader_temporary(
                stage,
                offset,
                encoding,
                "TEXS half output temporary overflow",
                next_temporary,
            )?
        } else {
            ShaderRegister::new(u16::from(if index < primary_count {
                primary_destination + index as u8
            } else {
                secondary_destination + (index - primary_count) as u8
            }))
        };
        outputs.push(
            ShaderTextureSampleOutput::new(register, component).expect("decoded component is RGBA"),
        );
    }
    let sampled_registers = if half_output {
        outputs.iter().map(|o| o.destination()).collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    let outputs = outputs.into_boxed_slice();
    let mut operations = Vec::new();
    let lod = if !fetch && matches!(selector, 2 | 8) {
        let destination = allocate_shader_temporary(
            stage,
            offset,
            encoding,
            "TEXS LOD temporary overflow",
            next_temporary,
        )?;
        operations.push(ShaderOperation::MoveImmediate32 {
            destination,
            bits: 0,
            scalar_type: ShaderScalarType::Float32,
        });
        Some(destination)
    } else if !fetch && matches!(selector, 3 | 13) {
        Some(ShaderRegister::new(
            u16::from(y_coordinate) + u16::from(selector == 13),
        ))
    } else {
        None
    };
    let sample = if fetch {
        ShaderOperation::LoadTexture2D {
            outputs,
            coordinates: [
                ShaderRegister::new(u16::from(x_coordinate)),
                ShaderRegister::new(u16::from(y_coordinate)),
            ],
            image_binding: binding.image_binding,
            mip_level: 0,
        }
    } else if matches!(selector, 12 | 13) {
        ShaderOperation::SampleTextureCube {
            outputs,
            coordinates: [
                ShaderRegister::new(u16::from(x_coordinate)),
                ShaderRegister::new(u16::from(x_coordinate) + 1),
                ShaderRegister::new(u16::from(y_coordinate)),
            ],
            image_binding: binding.image_binding,
            lod,
            sampler_binding: binding.sampler_binding.expect("TEXS reserves a sampler"),
        }
    } else if matches!(selector, 1..=3) {
        ShaderOperation::SampleTexture2D {
            offset: None,
            outputs,
            coordinates: [
                ShaderRegister::new(u16::from(x_coordinate)),
                ShaderRegister::new(if selector == 3 {
                    u16::from(x_coordinate) + 1
                } else {
                    u16::from(y_coordinate)
                }),
            ],
            image_binding: binding.image_binding,
            lod,
            sampler_binding: binding.sampler_binding.expect("TEXS reserves a sampler"),
        }
    } else {
        ShaderOperation::SampleTexture2DArray {
            offset: None,
            outputs,
            coordinates: [
                ShaderRegister::new(u16::from(x_coordinate + 1)),
                ShaderRegister::new(u16::from(y_coordinate)),
            ],
            array_index: ShaderRegister::new(u16::from(x_coordinate)),
            image_binding: binding.image_binding,
            lod,
            sampler_binding: binding.sampler_binding.expect("TEXS reserves a sampler"),
        }
    };
    operations.push(sample);
    if half_output {
        // TEXS.F16 packs selected components in pairs, not consecutive dword
        // outputs. Sample into temporaries first: coordinates/destinations may
        // alias, and split destinations that overlap in F32 are valid in F16.
        // https://github.com/eden-emulator/mirror/blob/d16735f5b618942136d6ab53466e3be0a382c30a/src/shader_recompiler/frontend/maxwell/translate/impl/texture_fetch_swizzled.cpp#L218-L251
        let temporary = |next: &mut u16| {
            allocate_shader_temporary(
                stage,
                offset,
                encoding,
                "TEXS half packing temporary overflow",
                next,
            )
        };
        for (pair, registers) in sampled_registers.chunks(2).enumerate() {
            let destination = ShaderRegister::new(u16::from(if pair == 0 {
                primary_destination
            } else {
                secondary_destination
            }));
            let low = temporary(next_temporary)?;
            operations.push(ShaderOperation::PackHalf32 {
                destination: low,
                source: registers[0],
            });
            if registers.len() == 1 {
                operations.push(ShaderOperation::Move32 {
                    destination,
                    source: low,
                    scalar_type: ShaderScalarType::Unsigned32,
                });
            } else {
                let high = temporary(next_temporary)?;
                let shift = temporary(next_temporary)?;
                operations.push(ShaderOperation::PackHalf32 {
                    destination: high,
                    source: registers[1],
                });
                operations.push(ShaderOperation::MoveImmediate32 {
                    destination: shift,
                    bits: 16,
                    scalar_type: ShaderScalarType::Unsigned32,
                });
                operations.push(ShaderOperation::ShiftLeft32 {
                    destination: high,
                    value: high,
                    amount: shift,
                    wrap: false,
                });
                operations.push(ShaderOperation::Bitwise32 {
                    destination,
                    left: low,
                    right: high,
                    operation: ShaderBitwiseOperation::Or,
                });
            }
        }
    }
    Ok(operations)
}

#[cfg(test)]
mod tests;

/// AOFFI texel offsets are often shader literals. Stop at control-flow or a
/// conditional redefinition; never freeze a uniform or an invocation input.
fn literal_register(
    mut instructions: &[ShaderInstruction],
    mut register: ShaderRegister,
) -> Option<u32> {
    if instructions.iter().any(|instruction| {
        instruction.predicate() != ShaderPredicate::Never
            && matches!(
                instruction.operation(),
                ShaderOperation::Branch { .. } | ShaderOperation::BranchTable { .. }
            )
    }) {
        return None;
    }
    loop {
        let mut copied = None;
        for (index, instruction) in instructions.iter().enumerate().rev() {
            if instruction.predicate() == ShaderPredicate::Never {
                continue;
            }
            if matches!(instruction.operation(), ShaderOperation::Exit) {
                return None;
            }
            let mut writes = false;
            instruction
                .operation()
                .visit_destination_registers(|destination| writes |= destination == register);
            if !writes {
                continue;
            }
            if instruction.predicate() != ShaderPredicate::Always {
                return None;
            }
            match instruction.operation() {
                ShaderOperation::MoveImmediate32 { bits, .. } => return Some(*bits),
                ShaderOperation::Move32 { source, .. } => {
                    copied = Some((index, *source));
                    break;
                }
                _ => return None,
            }
        }
        let (index, source) = copied?;
        instructions = &instructions[..index];
        register = source;
    }
}
