//! Maxwell simplified texture instructions and their local resource bindings.

use super::decode::validate_register_range;
use super::error::{MaxwellShaderTranslationError, malformed};
use crate::MaxwellShaderStage;
use nixe_gpu::{ShaderOperation, ShaderRegister, ShaderResourceKind, ShaderTextureSampleOutput};
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
) -> Result<ShaderOperation, MaxwellShaderTranslationError> {
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
    if !fetch && (stage != MaxwellShaderStage::Pixel || !matches!(selector, 1 | 7)) {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "TEXS mode other than fragment 2D/2D-array implicit LOD",
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
    let constant_buffer_byte_offset = u32::from(constant_buffer_dword_offset) * 4;
    if fetch || selector == 1 {
        validate_register_range(stage, offset, encoding, x_coordinate, 1, register_count)?;
        validate_register_range(stage, offset, encoding, y_coordinate, 1, register_count)?;
    } else {
        if !x_coordinate.is_multiple_of(2) {
            return Err(malformed(
                stage,
                offset,
                encoding,
                "TEXS 2D-array packed layer/first-coordinate register is misaligned",
            ));
        }
        validate_register_range(stage, offset, encoding, x_coordinate, 2, register_count)?;
        validate_register_range(stage, offset, encoding, y_coordinate, 1, register_count)?;
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
    let primary_count = channels.len().min(2);
    validate_register_range(
        stage,
        offset,
        encoding,
        primary_destination,
        primary_count as u8,
        register_count,
    )?;
    if channels.len() > primary_count {
        validate_register_range(
            stage,
            offset,
            encoding,
            secondary_destination,
            (channels.len() - primary_count) as u8,
            register_count,
        )?;
    }

    let image_kind = if fetch || selector == 1 {
        ShaderResourceKind::SampledImage
    } else {
        ShaderResourceKind::SampledImage2DArray
    };
    let binding = if let Some(binding) = bindings.get_mut(&constant_buffer_dword_offset) {
        if binding.image_kind != image_kind {
            return Err(malformed(
                stage,
                offset,
                encoding,
                "TEXS/TLDS reuse one descriptor with contradictory image dimensions",
            ));
        }
        if !fetch && binding.sampler_binding.is_none() {
            binding.sampler_binding = Some(binding.image_binding + 1);
        }
        *binding
    } else {
        let next_pair = u8::try_from(32 + bindings.len() * 2).map_err(|_| {
            malformed(
                stage,
                offset,
                encoding,
                "TEXS/TLDS neutral resource binding space is exhausted",
            )
        })?;
        let binding = MaxwellTextureResourceBinding {
            constant_buffer_byte_offset,
            image_binding: next_pair,
            sampler_binding: (!fetch).then_some(next_pair.checked_add(1).ok_or_else(|| {
                malformed(
                    stage,
                    offset,
                    encoding,
                    "TEXS/TLDS neutral resource binding space is exhausted",
                )
            })?),
            image_kind,
        };
        bindings.insert(constant_buffer_dword_offset, binding);
        binding
    };
    let outputs = channels
        .iter()
        .enumerate()
        .map(|(index, component)| {
            let register = if index < primary_count {
                primary_destination + index as u8
            } else {
                secondary_destination + (index - primary_count) as u8
            };
            ShaderTextureSampleOutput::new(ShaderRegister::new(u16::from(register)), *component)
                .expect("decoded texture component is in RGBA range")
        })
        .collect::<Vec<_>>()
        .into_boxed_slice();
    if fetch {
        Ok(ShaderOperation::LoadTexture2D {
            outputs,
            coordinates: [
                ShaderRegister::new(u16::from(x_coordinate)),
                ShaderRegister::new(u16::from(y_coordinate)),
            ],
            image_binding: binding.image_binding,
            mip_level: 0,
        })
    } else if selector == 1 {
        Ok(ShaderOperation::SampleTexture2D {
            outputs,
            coordinates: [
                ShaderRegister::new(u16::from(x_coordinate)),
                ShaderRegister::new(u16::from(y_coordinate)),
            ],
            image_binding: binding.image_binding,
            sampler_binding: binding.sampler_binding.expect("TEXS reserves a sampler"),
        })
    } else {
        Ok(ShaderOperation::SampleTexture2DArray {
            outputs,
            coordinates: [
                ShaderRegister::new(u16::from(x_coordinate + 1)),
                ShaderRegister::new(u16::from(y_coordinate)),
            ],
            array_index: ShaderRegister::new(u16::from(x_coordinate)),
            image_binding: binding.image_binding,
            sampler_binding: binding.sampler_binding.expect("TEXS reserves a sampler"),
        })
    }
}

#[cfg(test)]
mod tests;
