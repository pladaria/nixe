use crate::ImageFormat;
/// Decodes verified Maxwell TIC format/component/swizzle tuples into their
/// neutral storage and sampling semantics. Numeric component and swizzle values are
/// defined by deko3d's pinned public Maxwell tables:
/// https://github.com/devkitPro/deko3d/blob/350f2b00a3e76ecd4f00191f8c5d6544ffbcb9db/source/maxwell/image_formats.h
pub(super) fn decode_sampled_texture_format(
    image_format: u8,
    components: [u8; 4],
    swizzle: [u8; 4],
    srgb: bool,
    reserved: u32,
) -> Option<ImageFormat> {
    if reserved != 0 || !swizzle.iter().all(|c| matches!(c, 0 | 2..=7)) {
        return None;
    }
    if components == [2; 4] {
        let astc = match (image_format, srgb) {
            (0x40, false) => Some(ImageFormat::Astc4x4Unorm),
            (0x40, true) => Some(ImageFormat::Astc4x4Srgb),
            (0x50, false) => Some(ImageFormat::Astc5x4Unorm),
            (0x50, true) => Some(ImageFormat::Astc5x4Srgb),
            (0x41, false) => Some(ImageFormat::Astc5x5Unorm),
            (0x41, true) => Some(ImageFormat::Astc5x5Srgb),
            (0x51, false) => Some(ImageFormat::Astc6x5Unorm),
            (0x51, true) => Some(ImageFormat::Astc6x5Srgb),
            (0x42, false) => Some(ImageFormat::Astc6x6Unorm),
            (0x42, true) => Some(ImageFormat::Astc6x6Srgb),
            (0x55, false) => Some(ImageFormat::Astc8x5Unorm),
            (0x55, true) => Some(ImageFormat::Astc8x5Srgb),
            (0x52, false) => Some(ImageFormat::Astc8x6Unorm),
            (0x52, true) => Some(ImageFormat::Astc8x6Srgb),
            (0x44, false) => Some(ImageFormat::Astc8x8Unorm),
            (0x44, true) => Some(ImageFormat::Astc8x8Srgb),
            (0x56, false) => Some(ImageFormat::Astc10x5Unorm),
            (0x56, true) => Some(ImageFormat::Astc10x5Srgb),
            (0x57, false) => Some(ImageFormat::Astc10x6Unorm),
            (0x57, true) => Some(ImageFormat::Astc10x6Srgb),
            (0x53, false) => Some(ImageFormat::Astc10x8Unorm),
            (0x53, true) => Some(ImageFormat::Astc10x8Srgb),
            (0x45, false) => Some(ImageFormat::Astc10x10Unorm),
            (0x45, true) => Some(ImageFormat::Astc10x10Srgb),
            (0x54, false) => Some(ImageFormat::Astc12x10Unorm),
            (0x54, true) => Some(ImageFormat::Astc12x10Srgb),
            (0x46, false) => Some(ImageFormat::Astc12x12Unorm),
            (0x46, true) => Some(ImageFormat::Astc12x12Srgb),
            _ => None,
        };
        if astc.is_some() {
            return astc;
        }
    }
    match (image_format, components, swizzle, srgb) {
        // Component type 4 is UINT, type 3 is SINT; selectors do not
        // change the integer storage or its shader-visible component type.
        // https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/texture_cache/format_lookup_table.cpp
        (0x1d, [4, 4, 4, 4], _, false) => Some(ImageFormat::R8Uint),
        (0x1d, [3, 3, 3, 3], _, false) => Some(ImageFormat::R8Sint),
        (0x18, [4, 4, 4, 4], _, false) => Some(ImageFormat::Rg8Uint),
        (0x18, [3, 3, 3, 3], _, false) => Some(ImageFormat::Rg8Sint),
        (0x08, [4, 4, 4, 4], _, false) => Some(ImageFormat::Rgba8Uint),
        (0x08, [3, 3, 3, 3], _, false) => Some(ImageFormat::Rgba8Sint),
        (0x1b, [4, 4, 4, 4], _, false) => Some(ImageFormat::R16Uint),
        (0x1b, [3, 3, 3, 3], _, false) => Some(ImageFormat::R16Sint),
        (0x0c, [4, 4, 4, 4], _, false) => Some(ImageFormat::Rg16Uint),
        (0x0c, [3, 3, 3, 3], _, false) => Some(ImageFormat::Rg16Sint),
        (0x03, [4, 4, 4, 4], _, false) => Some(ImageFormat::Rgba16Uint),
        (0x03, [3, 3, 3, 3], _, false) => Some(ImageFormat::Rgba16Sint),
        (0x0f, [4, 4, 4, 4], _, false) => Some(ImageFormat::R32Uint),
        (0x0f, [3, 3, 3, 3], _, false) => Some(ImageFormat::R32Sint),
        (0x04, [4, 4, 4, 4], _, false) => Some(ImageFormat::Rg32Uint),
        (0x04, [3, 3, 3, 3], _, false) => Some(ImageFormat::Rg32Sint),
        (0x01, [4, 4, 4, 4], _, false) => Some(ImageFormat::Rgba32Uint),
        (0x01, [3, 3, 3, 3], _, false) => Some(ImageFormat::Rgba32Sint),
        (0x1d, [1, 1, 1, 1], _, false) => Some(ImageFormat::R8Snorm),
        (0x18, [1, 1, 1, 1], _, false) => Some(ImageFormat::Rg8Snorm),
        (0x08, [1, 1, 1, 1], _, false) => Some(ImageFormat::Rgba8Snorm),
        (0x1b, [2, 2, 2, 2], _, false) => Some(ImageFormat::R16Unorm),
        (0x1b, [1, 1, 1, 1], _, false) => Some(ImageFormat::R16Snorm),
        (0x0c, [2, 2, 2, 2], _, false) => Some(ImageFormat::Rg16Unorm),
        (0x0c, [1, 1, 1, 1], _, false) => Some(ImageFormat::Rg16Snorm),
        (0x03, [2, 2, 2, 2], _, false) => Some(ImageFormat::Rgba16Unorm),
        (0x03, [1, 1, 1, 1], _, false) => Some(ImageFormat::Rgba16Snorm),
        (0x15, [2, 2, 2, 2], _, false) => Some(ImageFormat::Rgb565Unorm),
        // Z24S8 uses unsigned stencil R and normalized depth G. Selecting G
        // for RGB samples the depth aspect, with alpha forced to one.
        (0x29, [4, 2, 2, 2], [3, 3, 3, 7], false) => Some(ImageFormat::Depth24UnormStencil8Uint),
        // https://github.com/eden-emulator/mirror/blob/master/src/video_core/texture_cache/format_lookup_table.cpp
        (0x3a, [2, 2, 2, 2] | [2, 4, 4, 4], _, false) => Some(ImageFormat::Depth16Unorm),
        // ZF32 uses a float depth channel, replicated to RGB by its TIC mapping.
        // https://github.com/devkitPro/deko3d/blob/master/source/maxwell/image_formats.h
        (0x2f, [7, 7, 7, 7] | [7, 4, 4, 4] | [7, 2, 2, 2], _, false) => {
            Some(ImageFormat::Depth32Float)
        }
        // BC6H floating-point component types follow Maxwell's TIC table.
        // https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/texture_cache/format_lookup_table.cpp#L190-L193
        (0x10, [7, 7, 7, 7], _, false) => Some(ImageFormat::Bc6hRgbSfloat),
        (0x11, [7, 7, 7, 7], _, false) => Some(ImageFormat::Bc6hRgbUfloat),
        (0x27, [2, 2, 2, 2], _, false) => Some(ImageFormat::Bc4RUnorm),
        (0x28, [2, 2, 2, 2], _, false) => Some(ImageFormat::Bc5RgUnorm),
        // Component type 1 is SNORM; signed RGTC endpoints are not UNORM.
        // https://github.com/devkitPro/deko3d/blob/350f2b00a3e76ecd4f00191f8c5d6544ffbcb9db/source/maxwell/image_formats.h
        (0x27, [1, 1, 1, 1], _, false) => Some(ImageFormat::Bc4RSnorm),
        (0x28, [1, 1, 1, 1], _, false) => Some(ImageFormat::Bc5RgSnorm),
        (0x17, [2, 2, 2, 2], _, srgb) => Some(if srgb {
            ImageFormat::Bc7RgbaSrgb
        } else {
            ImageFormat::Bc7RgbaUnorm
        }),
        (0x26, [2, 2, 2, 2], _, srgb) => Some(if srgb {
            ImageFormat::Bc3RgbaSrgb
        } else {
            ImageFormat::Bc3RgbaUnorm
        }),
        // https://registry.khronos.org/OpenGL/extensions/EXT/EXT_texture_compression_s3tc.txt
        // DXT1 storage and TIC selectors are independent. The opaque RGB
        // variant is equivalent only when no output observes the stored alpha.
        (0x24, [2, 2, 2, 2], mapping, srgb) => Some(match (mapping.contains(&5), srgb) {
            (false, false) => ImageFormat::Bc1RgbUnorm,
            (false, true) => ImageFormat::Bc1RgbSrgb,
            (true, false) => ImageFormat::Bc1RgbaUnorm,
            (true, true) => ImageFormat::Bc1RgbaSrgb,
        }),
        // Single-channel masks can independently route R or constants to
        // every sampled component, including white RGB with red as alpha.
        (0x1d, [2, 2, 2, 2], _, false) => Some(ImageFormat::R8Unorm),
        // Storage and view selectors are independent; luminance-alpha uses
        // R,R,R,G without changing the two-byte RG8 storage format.
        // https://github.com/eden-emulator/mirror/blob/master/src/video_core/texture_cache/format_lookup_table.cpp
        (0x18, [2, 2, 2, 2], _, false) => Some(ImageFormat::Rg8Unorm),
        (0x08, [2, 2, 2, 2], [4, 3, 2, 5], false) => Some(ImageFormat::Bgra8Unorm),
        (0x08, [2, 2, 2, 2], [4, 3, 2, 5], true) => Some(ImageFormat::Bgra8Srgb),
        (0x08, [2, 2, 2, 2], _, false) => Some(ImageFormat::Rgba8Unorm),
        (0x08, [2, 2, 2, 2], _, true) => Some(ImageFormat::Rgba8Srgb),
        (0x1b, [7, 7, 7, 7], _, false) => Some(ImageFormat::R16Float),
        (0x0c, [7, 7, 7, 7], _, false) => Some(ImageFormat::Rg16Float),
        (0x03, [7, 7, 7, 7], _, false) => Some(ImageFormat::Rgba16Float),
        (0x21, [7, 7, 7, 7], _, false) => Some(ImageFormat::Rg11b10Ufloat),
        (0x09, [2, 2, 2, 2], _, false) => Some(ImageFormat::Rgb10a2Unorm),
        (0x0f, [7, 7, 7, 7], _, false) => Some(ImageFormat::R32Float),
        (0x04, [7, 7, 7, 7], _, false) => Some(ImageFormat::Rg32Float),
        (0x01, [7, 7, 7, 7], _, false) => Some(ImageFormat::Rgba32Float),
        _ => None,
    }
}
