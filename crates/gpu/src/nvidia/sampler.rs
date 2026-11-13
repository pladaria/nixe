use super::*;
pub fn decode_sampler(
    descriptor: u32,
    bytes: [u8; 32],
) -> Result<SamplerDescription, DescriptorError> {
    // Maxwell TSC field placement and public enums follow deko3d's pinned
    // descriptor definition and generator:
    // https://github.com/devkitPro/deko3d/blob/350f2b00a3e76ecd4f00191f8c5d6544ffbcb9db/source/maxwell/texture_sampler_control_block.h
    // https://github.com/devkitPro/deko3d/blob/350f2b00a3e76ecd4f00191f8c5d6544ffbcb9db/source/maxwell/tsc_generate.cpp
    let words = descriptor_words(bytes);
    // Eden TSCEntry::reduction_filter, word 1 bits 10..11.
    // https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/textures/texture.h
    let reduction = match (words[1] >> 10) & 3 {
        0 => crate::SamplerReduction::WeightedAverage,
        1 => crate::SamplerReduction::Min,
        2 => crate::SamplerReduction::Max,
        value => {
            return Err(DescriptorError {
                descriptor,
                field: "reduction filter",
                value,
            });
        }
    };
    let compare = (words[0] & (1 << 9) != 0).then(|| {
        use crate::DepthCompareOperation as C;
        [
            C::Never,
            C::Less,
            C::Equal,
            C::LessEqual,
            C::Greater,
            C::NotEqual,
            C::GreaterEqual,
            C::Always,
        ][((words[0] >> 10) & 7) as usize]
    });
    let address_mode = |raw: u32| match raw {
        0 => Ok(AddressMode::Repeat),
        1 => Ok(AddressMode::MirroredRepeat),
        2 => Ok(AddressMode::ClampToEdge),
        3 => Ok(AddressMode::ClampToBorder),
        4 => Ok(AddressMode::Clamp),
        _ => Err(DescriptorError {
            descriptor,
            field: "address mode",
            value: raw,
        }),
    };
    let filter = |raw: u32, field| match raw {
        1 => Ok(FilterMode::Nearest),
        2 => Ok(FilterMode::Linear),
        _ => Err(DescriptorError {
            descriptor,
            field,
            value: raw,
        }),
    };
    let mip_filter = match (words[1] >> 6) & 3 {
        1 | 2 => FilterMode::Nearest,
        3 => FilterMode::Linear,
        value => {
            return Err(DescriptorError {
                descriptor,
                field: "mip filter",
                value,
            });
        }
    };
    let bias = ((words[1] >> 12) & 0x1fff) as i32;
    let signed_bias = (bias << 19) >> 19;
    let lod_min_fixed = (words[2] & 0xfff) as u16;
    let lod_max_fixed = ((words[2] >> 12) & 0xfff) as u16;
    if lod_min_fixed > lod_max_fixed {
        return Err(DescriptorError {
            descriptor,
            field: "LOD clamp",
            value: words[2] & 0x00ff_ffff,
        });
    }
    let max_anisotropy = [1u8, 2, 4, 6, 8, 10, 12, 16][((words[0] >> 20) & 7) as usize];
    let min_filter = filter((words[1] >> 4) & 3, "minification filter")?;
    let mag_filter = filter(words[1] & 3, "magnification filter")?;
    let address_modes = [
        address_mode(words[0] & 7)?,
        address_mode((words[0] >> 3) & 7)?,
        address_mode((words[0] >> 6) & 7)?,
    ];
    let samples_border = address_modes.contains(&AddressMode::ClampToBorder)
        || (mag_filter == FilterMode::Linear && address_modes.contains(&AddressMode::Clamp));
    let border_color = samples_border.then_some([words[4], words[5], words[6], words[7]]);
    if let Some(color) = border_color
        && words[0] & (1 << 13) != 0
    {
        let srgb = [
            words[2] >> 24,
            (words[3] >> 12) & 255,
            (words[3] >> 20) & 255,
        ];
        for (bits, encoded) in color[..3].iter().copied().zip(srgb) {
            if (bits == 0 && encoded != 0) || (bits == 1.0_f32.to_bits() && encoded != 255) {
                return Err(DescriptorError {
                    descriptor,
                    field: "distinct sRGB border color",
                    value: words[2] ^ words[3],
                });
            }
        }
    }
    Ok(SamplerDescription {
        compare,
        reduction,
        min_filter,
        mag_filter,
        mip_filter,
        address_modes,
        border_color,
        lod_bias: signed_bias as f32 / 256.0,
        lod_min: f32::from(lod_min_fixed) / 256.0,
        lod_max: f32::from(lod_max_fixed) / 256.0,
        max_anisotropy: f32::from(max_anisotropy),
    })
}
