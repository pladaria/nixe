use super::invalid;
use nixe_shader_recompiler::{
    Result,
    types::{TexturePixelFormat as F, TextureType as T},
};

// Maxwell TIC layout:
// https://github.com/devkitPro/deko3d/blob/350f2b00a3e76ecd4f00191f8c5d6544ffbcb9db/source/maxwell/texture_image_control_block.h
pub(super) fn texture_type(words: [u32; 8]) -> Result<T> {
    match (words[4] >> 23) & 15 {
        0 => Ok(T::Color1D),
        1 | 7 => Ok(if words[5] & (1 << 31) != 0 {
            T::Color2D
        } else {
            T::Color2DRect
        }),
        2 => Ok(T::Color3D),
        3 => Ok(T::ColorCube),
        4 => Ok(T::ColorArray1D),
        5 => Ok(T::ColorArray2D),
        6 => Ok(T::Buffer),
        8 => Ok(T::ColorArrayCube),
        _ => Ok(T::Color2D),
    }
}

// Storage format classification does not depend on swizzles, image addresses,
// dimensions or sampler state. Only the consumed format enters shader identity.
pub(super) fn pixel_format(words: [u32; 8]) -> Result<F> {
    let format = words[0] & 0x7f;
    let component = (words[0] >> 7) & 7;
    let srgb = words[4] & (1 << 22) != 0;
    let components = [7, 10, 13, 16].map(|shift| (words[0] >> shift) & 7);
    let active = match format {
        0x01 | 0x03 | 0x08 => 4,
        0x02 => 3,
        0x04 | 0x0c | 0x18 => 2,
        _ => 1,
    };
    if components[..active].iter().any(|&value| value != component) {
        return Err(invalid(format!(
            "Mixed component types in TIC color format {format:#x}"
        )));
    }
    let family = match format {
        0x01 => [
            Some(F::R32G32B32A32_SINT),
            Some(F::R32G32B32A32_UINT),
            Some(F::R32G32B32A32_FLOAT),
            None,
            None,
        ],
        0x03 => [
            Some(F::R16G16B16A16_SINT),
            Some(F::R16G16B16A16_UINT),
            Some(F::R16G16B16A16_FLOAT),
            Some(F::R16G16B16A16_SNORM),
            Some(F::R16G16B16A16_UNORM),
        ],
        0x04 => [
            Some(F::R32G32_SINT),
            Some(F::R32G32_UINT),
            Some(F::R32G32_FLOAT),
            None,
            None,
        ],
        0x08 => [
            Some(F::A8B8G8R8_SINT),
            Some(F::A8B8G8R8_UINT),
            None,
            Some(F::A8B8G8R8_SNORM),
            Some(if srgb {
                F::A8B8G8R8_SRGB
            } else {
                F::A8B8G8R8_UNORM
            }),
        ],
        0x0c => [
            Some(F::R16G16_SINT),
            Some(F::R16G16_UINT),
            Some(F::R16G16_FLOAT),
            Some(F::R16G16_SNORM),
            Some(F::R16G16_UNORM),
        ],
        0x0f => [
            Some(F::R32_SINT),
            Some(F::R32_UINT),
            Some(F::R32_FLOAT),
            None,
            None,
        ],
        0x18 => [
            Some(F::R8G8_SINT),
            Some(F::R8G8_UINT),
            None,
            Some(F::R8G8_SNORM),
            Some(F::R8G8_UNORM),
        ],
        0x1b => [
            Some(F::R16_SINT),
            Some(F::R16_UINT),
            Some(F::R16_FLOAT),
            Some(F::R16_SNORM),
            Some(F::R16_UNORM),
        ],
        0x1d => [
            Some(F::R8_SINT),
            Some(F::R8_UINT),
            None,
            Some(F::R8_SNORM),
            Some(F::R8_UNORM),
        ],
        _ => [None; 5],
    };
    let slot = match component {
        3 => Some(0),
        4 => Some(1),
        7 => Some(2),
        1 => Some(3),
        2 => Some(4),
        _ => None,
    };
    if let Some(value) = slot.and_then(|i| family[i]) {
        return Ok(value);
    }
    let compressed = match (format, component, srgb) {
        (0x40, 2, false) => Some(F::ASTC_2D_4X4_UNORM),
        (0x40, 2, true) => Some(F::ASTC_2D_4X4_SRGB),
        (0x41, 2, false) => Some(F::ASTC_2D_5X5_UNORM),
        (0x41, 2, true) => Some(F::ASTC_2D_5X5_SRGB),
        (0x42, 2, false) => Some(F::ASTC_2D_6X6_UNORM),
        (0x42, 2, true) => Some(F::ASTC_2D_6X6_SRGB),
        (0x44, 2, false) => Some(F::ASTC_2D_8X8_UNORM),
        (0x44, 2, true) => Some(F::ASTC_2D_8X8_SRGB),
        (0x45, 2, false) => Some(F::ASTC_2D_10X10_UNORM),
        (0x45, 2, true) => Some(F::ASTC_2D_10X10_SRGB),
        (0x46, 2, false) => Some(F::ASTC_2D_12X12_UNORM),
        (0x46, 2, true) => Some(F::ASTC_2D_12X12_SRGB),
        (0x50, 2, false) => Some(F::ASTC_2D_5X4_UNORM),
        (0x50, 2, true) => Some(F::ASTC_2D_5X4_SRGB),
        (0x51, 2, false) => Some(F::ASTC_2D_6X5_UNORM),
        (0x51, 2, true) => Some(F::ASTC_2D_6X5_SRGB),
        (0x52, 2, false) => Some(F::ASTC_2D_8X6_UNORM),
        (0x52, 2, true) => Some(F::ASTC_2D_8X6_SRGB),
        (0x53, 2, false) => Some(F::ASTC_2D_10X8_UNORM),
        (0x53, 2, true) => Some(F::ASTC_2D_10X8_SRGB),
        (0x54, 2, false) => Some(F::ASTC_2D_12X10_UNORM),
        (0x54, 2, true) => Some(F::ASTC_2D_12X10_SRGB),
        (0x55, 2, false) => Some(F::ASTC_2D_8X5_UNORM),
        (0x55, 2, true) => Some(F::ASTC_2D_8X5_SRGB),
        (0x56, 2, false) => Some(F::ASTC_2D_10X5_UNORM),
        (0x56, 2, true) => Some(F::ASTC_2D_10X5_SRGB),
        (0x57, 2, false) => Some(F::ASTC_2D_10X6_UNORM),
        (0x57, 2, true) => Some(F::ASTC_2D_10X6_SRGB),
        (0x02, 7, false) => Some(F::R32G32B32_FLOAT),
        (0x09, 2, false) => Some(F::A2B10G10R10_UNORM),
        (0x09, 4, false) => Some(F::A2B10G10R10_UINT),
        (0x10, 7, false) => Some(F::BC6H_SFLOAT),
        (0x11, 7, false) => Some(F::BC6H_UFLOAT),
        (0x12, 2, false) => Some(F::A4B4G4R4_UNORM),
        (0x13, 2, false) => Some(F::A5B5G5R1_UNORM),
        (0x14, 2, false) => Some(F::A1B5G5R5_UNORM),
        (0x1e, 2, false) => Some(F::G4R4_UNORM),
        (0x20, 7, false) => Some(F::E5B9G9R9_FLOAT),
        (0x29 | 0x0e, 4, false) if components[1] == 2 => Some(F::S8_UINT_D24_UNORM),
        (0x2b, 2, false) if components[1] == 4 => Some(F::D24_UNORM_S8_UINT),
        (0x30 | 0x05, 7, false) if components[1] == 4 => Some(F::D32_FLOAT_S8_UINT),
        (0x2a, 2, false) => Some(F::X8_D24_UNORM),
        (0x3a, 2, false) => Some(F::D16_UNORM),
        (0x15, 2, false) => Some(F::B5G6R5_UNORM),
        (0x21, 7, false) => Some(F::B10G11R11_FLOAT),
        (0x24, 2, false) => Some(F::BC1_RGBA_UNORM),
        (0x24, 2, true) => Some(F::BC1_RGBA_SRGB),
        (0x25, 2, false) => Some(F::BC2_UNORM),
        (0x25, 2, true) => Some(F::BC2_SRGB),
        (0x26, 2, false) => Some(F::BC3_UNORM),
        (0x26, 2, true) => Some(F::BC3_SRGB),
        (0x27, 2, false) => Some(F::BC4_UNORM),
        (0x27, 1, false) => Some(F::BC4_SNORM),
        (0x28, 2, false) => Some(F::BC5_UNORM),
        (0x28, 1, false) => Some(F::BC5_SNORM),
        (0x17, 2, false) => Some(F::BC7_UNORM),
        (0x17, 2, true) => Some(F::BC7_SRGB),
        (0x2f, 7, false) => Some(F::D32_FLOAT),
        _ => None,
    };
    compressed.ok_or_else(|| {
        invalid(format!(
            "Unsupported TIC pixel format {format:#x}, component {component}, sRGB {srgb}"
        ))
    })
}

pub(super) fn is_integer(format: F) -> bool {
    matches!(
        format,
        F::A8B8G8R8_SINT
            | F::A8B8G8R8_UINT
            | F::A2B10G10R10_UINT
            | F::R8_SINT
            | F::R8_UINT
            | F::R16G16B16A16_SINT
            | F::R16G16B16A16_UINT
            | F::R32G32B32A32_UINT
            | F::R32G32B32A32_SINT
            | F::R32G32_SINT
            | F::R16_UINT
            | F::R16_SINT
            | F::R16G16_UINT
            | F::R16G16_SINT
            | F::R8G8_SINT
            | F::R8G8_UINT
            | F::R32G32_UINT
            | F::R32_UINT
            | F::R32_SINT
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn depth_stencil_and_integer_color_have_different_shader_classes() {
        let words = |format, components: [u32; 4]| {
            let mut words = [0; 8];
            words[0] = format;
            for (shift, value) in [7, 10, 13, 16].into_iter().zip(components) {
                words[0] |= value << shift;
            }
            words
        };
        let depth = pixel_format(words(0x29, [4, 2, 2, 2])).unwrap();
        assert_eq!(depth, F::S8_UINT_D24_UNORM);
        assert!(!is_integer(depth));
        assert!(is_integer(pixel_format(words(0x08, [4; 4])).unwrap()));
        assert!(!is_integer(pixel_format(words(0x08, [1; 4])).unwrap()));
        assert!(pixel_format(words(0x08, [4, 7, 4, 4])).is_err());
        let mut astc = words(0x40, [2; 4]);
        astc[4] |= 1 << 22;
        assert_eq!(pixel_format(astc).unwrap(), F::ASTC_2D_4X4_SRGB);
    }
}
