//! Raw TIC interpretation for the owning texture cache. Mutable descriptor
//! admission and native resource selection belong to that cache, not this parser.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/texture_cache/image_info.cpp
use super::*;

/// Raw TIC addresses have sixteen high bits; the GPU VA space uses forty.
/// https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/texture_cache/util.cpp#L830-L846
pub fn texture_address(words: [u32; 8]) -> u64 {
    u64::from(words[1]) | (u64::from(words[2] & 0xffff) << 32)
}

pub fn sampled_image_info(
    required_dimension: TextureDimension,
    descriptor_index: u32,
    bytes: [u8; 32],
    tiling: TextureTiling,
) -> Result<ImageInfo, DescriptorError> {
    // Maxwell 2 TIC field placement follows deko3d's pinned public
    // descriptor definition, itself cross-referenced to envytools:
    // https://github.com/devkitPro/deko3d/blob/350f2b00a3e76ecd4f00191f8c5d6544ffbcb9db/source/maxwell/texture_image_control_block.h
    let words = descriptor_words(bytes);
    let format_word = words[0];
    let image_format = (format_word & 0x7f) as u8;
    let components = [7, 10, 13, 16].map(|shift| ((format_word >> shift) & 7) as u8);
    let swizzle = [19, 22, 25, 28].map(|shift| ((format_word >> shift) & 7) as u8);
    let srgb = words[4] & (1 << 22) != 0;
    let format =
        decode_sampled_texture_format(image_format, components, swizzle, srgb, format_word >> 31)
            .ok_or(DescriptorError {
            descriptor: descriptor_index,
            field: "format/component/swizzle",
            value: format_word,
        })?;
    let sample_swizzle = if !matches!(format, ImageFormat::Bgra8Unorm | ImageFormat::Bgra8Srgb) {
        Swizzle(swizzle.map(|component| match component {
            0 => ComponentSwizzle::Zero,
            // TIC B5G6R5 stores red in the low bits; the R5G6B5 storage
            // format puts it in the high bits. Compose that channel swap
            // with every selector, including repeated channels and constants.
            // https://github.com/devkitPro/deko3d/blob/350f2b00a3e76ecd4f00191f8c5d6544ffbcb9db/source/maxwell/format_traits.inc
            2 if format == ImageFormat::Rgb565Unorm => ComponentSwizzle::Blue,
            4 if format == ImageFormat::Rgb565Unorm => ComponentSwizzle::Red,
            2 => ComponentSwizzle::Red,
            3 => ComponentSwizzle::Green,
            4 => ComponentSwizzle::Blue,
            5 => ComponentSwizzle::Alpha,
            6 | 7 => ComponentSwizzle::One,
            _ => unreachable!("validated TIC component mapping"),
        }))
    } else {
        // These packed color formats already include their native mapping.
        Swizzle::IDENTITY
    };
    let header_version = ((words[2] >> 21) & 7) as u8;
    let texture_type = ((words[4] >> 23) & 0xf) as u8;
    let width = (words[4] & 0xffff) + 1;
    let height = (words[5] & 0xffff) + 1;
    let depth = ((words[5] >> 16) & 0x3fff) + 1;
    let block_width_log2 = (words[3] & 7) as u8;
    let block_height_log2 = ((words[3] >> 3) & 7) as u8;
    let block_depth_log2 = ((words[3] >> 6) & 7) as u8;
    let mip_max = ((words[3] >> 28) & 0xf) as u8;
    let view_mip_min = (words[7] & 0xf) as u8;
    let view_mip_max = ((words[7] >> 4) & 0xf) as u8;
    let msaa = ((words[7] >> 8) & 0xf) as u8;
    // Sparse residency is represented by actual GPU map views, not a different texel layout.
    let normalized = words[5] & (1 << 31) != 0;
    let layer_base = match tiling {
        TextureTiling::Maxwell => {
            ((words[4] >> 16) & 7)
                | (((words[2] >> 16) & 0x1f) << 3)
                | (((words[2] >> 29) & 7) << 8)
        }
        // @SPECULATIVE: native descriptors address the selected view directly;
        // the legacy layer-base fields do not describe an additional prefix.
        // https://github.com/zortos293/Lichii/blob/21d62e43caa178aeb04e74585b3aad622b278b84/src/video_core/texture_cache/image_info.h#L16-L26
        TextureTiling::Ampere => 0,
    };
    let (required_texture_type, layers) = match required_dimension {
        TextureDimension::One => (0, 1),
        TextureDimension::OneArray => (
            4,
            u16::try_from(depth).map_err(|_| DescriptorError {
                descriptor: descriptor_index,
                field: "1D-array layer count",
                value: depth,
            })?,
        ),
        TextureDimension::Two => (1, 1_u16),
        TextureDimension::Three => (2, 1),
        // A single TIC cubemap has depth one and six independent faces.
        // https://github.com/eden-emulator/mirror/blob/d16735f5b618942136d6ab53466e3be0a382c30a/src/video_core/texture_cache/image_info.cpp
        TextureDimension::Cube => (3, 6),
        TextureDimension::CubeArray => (
            8,
            u16::try_from(depth * 6).map_err(|_| DescriptorError {
                descriptor: descriptor_index,
                field: "cube-array face count",
                value: depth,
            })?,
        ),
        TextureDimension::TwoArray => (
            5,
            u16::try_from(depth).map_err(|_| DescriptorError {
                descriptor: descriptor_index,
                field: "2D-array layer count",
                value: depth,
            })?,
        ),
    };
    let pitch_linear = header_version == 2;
    let cube = matches!(
        required_dimension,
        TextureDimension::Cube | TextureDimension::CubeArray
    );
    let texture_type_matches = texture_type == required_texture_type
        || (texture_type == 7
            && matches!(
                required_dimension,
                TextureDimension::Two | TextureDimension::TwoArray
            ))
        || (required_dimension == TextureDimension::CubeArray && texture_type == 3)
        || (required_dimension == TextureDimension::OneArray && texture_type == 0)
        || (required_dimension == TextureDimension::TwoArray && texture_type == 1);
    // TEX's array coordinate does not require multiple storage layers. A
    // non-array TIC can be sampled through a one-layer array view, retaining
    // the instruction's coordinate layout and host layer clamping.
    // Eden likewise preserves TEX dimensionality when it differs from TIC:
    // https://github.com/eden-emulator/mirror/blob/master/src/shader_recompiler/ir_opt/texture_pass.cpp
    if !matches!(header_version, 2 | 3)
        || !texture_type_matches
        || (cube && (pitch_linear || width != height))
        || (pitch_linear && depth != 1)
        || (!matches!(texture_type, 2 | 4 | 5 | 8) && depth != 1)
        || (matches!(
            required_dimension,
            TextureDimension::One | TextureDimension::OneArray
        ) && (height != 1 || pitch_linear))
        || (required_dimension == TextureDimension::Two && depth != 1)
        || (!pitch_linear
            && (block_width_log2 != 0 || block_depth_log2 > 5 || block_height_log2 > 5))
        || (pitch_linear && mip_max != 0)
        || view_mip_min != 0
        || view_mip_max != mip_max
        || msaa != 0
        // Only non-array 2D TICs select rectangle-coordinate semantics with
        // this bit. Eden leaves 1D, array, cube and 3D view types unchanged.
        || (!normalized && matches!(texture_type, 1 | 7))
        || layer_base != 0
    {
        return Err(DescriptorError {
            descriptor: descriptor_index,
            field: "sampled texture shape",
            value: words[2] ^ words[3] ^ words[4] ^ words[5] ^ words[7],
        });
    }
    // Width spacing changes Tegra GOB row alignment; the packed mip
    // layout below currently represents spacing zero only.
    if !pitch_linear && (words[3] >> 10) & 7 != 0 {
        return Err(DescriptorError {
            descriptor: descriptor_index,
            field: "tile-width spacing",
            value: words[3],
        });
    }
    let description = ImageDescription::new(
        if matches!(
            required_dimension,
            TextureDimension::One | TextureDimension::OneArray
        ) {
            ImageDimension::One
        } else if cube {
            ImageDimension::Cube
        } else if required_dimension == TextureDimension::Three {
            ImageDimension::Three
        } else {
            ImageDimension::Two
        },
        ImageExtent {
            width,
            height,
            depth: if required_dimension == TextureDimension::Three {
                depth
            } else {
                1
            },
        },
        format,
        if format.is_depth_stencil() {
            ImageKind::DepthStencil
        } else {
            ImageKind::Color
        },
        mip_max + 1,
        layers,
        SampleCount::One,
    )
    .map_err(|_| DescriptorError {
        descriptor: descriptor_index,
        field: "image geometry",
        value: 0,
    })?;
    let address = texture_address(words);
    let [block_width, block_height] = format.block_extent();
    let bytes_per_block = format
        .plane_bytes_per_block(0)
        .expect("decoded sampled image has one plane");
    let row_bytes = u64::from(width.div_ceil(block_width)) * u64::from(bytes_per_block);
    let SampledMipChain {
        layouts: mip_layouts,
        layer_stride,
        layer_size,
    } = if pitch_linear {
        // Pitch occupies bits 5:20 of the byte stride; the low five bits
        // are implicitly zero. Pitched storage has a single 2D layer.
        let row_pitch = u64::from(words[3] & 0xffff) << 5;
        if row_pitch < row_bytes {
            return Err(DescriptorError {
                descriptor: descriptor_index,
                field: "pitch smaller than texel row",
                value: row_pitch as u32,
            });
        }
        let layer_stride = row_pitch
            .checked_mul(u64::from(height.div_ceil(block_height)))
            .ok_or(DescriptorError {
                descriptor: descriptor_index,
                field: "storage overflow",
                value: 0,
            })?;
        SampledMipChain {
            layouts: vec![(
                0,
                ImageMemoryLayout::PitchLinear {
                    row_pitch,
                    layer_stride,
                },
            )],
            layer_stride,
            layer_size: layer_stride,
        }
    } else {
        sampled_mip_layouts(
            description.extent(),
            format,
            mip_max + 1,
            [block_height_log2, block_depth_log2],
            descriptor_index,
            tiling,
        )?
    };
    let layout = match mip_layouts[0].1 {
        ImageMemoryLayout::BlockLinear(mut blocks)
        | ImageMemoryLayout::BlockLinearAmpere(mut blocks) => {
            blocks.layer_stride = layer_stride;
            tiling.layout(blocks)
        }
        layout => layout,
    };
    let size = if layers == 1 {
        layer_size
    } else {
        layer_stride
            .checked_mul(u64::from(layers))
            .ok_or(DescriptorError {
                descriptor: descriptor_index,
                field: "storage overflow",
                value: 0,
            })?
    };
    Ok(ImageInfo {
        address,
        description,
        swizzle: sample_swizzle,
        layout,
        mip_layouts: mip_layouts.into(),
        layer_stride,
        maybe_unaligned_layer_stride: layer_size,
        size,
        packing: if format == ImageFormat::Depth24UnormStencil8Uint {
            ImagePacking::Depth24High
        } else {
            ImagePacking::Native
        },
        sample_grid: match description.samples() {
            SampleCount::One => [1, 1],
            SampleCount::Four => [2, 2],
            _ => {
                return Err(DescriptorError {
                    descriptor: descriptor_index,
                    field: "sample layout",
                    value: 0,
                });
            }
        },
    })
}

/// Tegra 2D mip chains: each level has a reduced GOB height, and array
/// layers align the complete chain rather than every level to the base block.
/// https://github.com/eden-emulator/mirror/blob/master/src/video_core/texture_cache/util.cpp
fn sampled_mip_layouts(
    extent: ImageExtent,
    format: ImageFormat,
    levels: u8,
    block: [u8; 2],
    descriptor_index: u32,
    tiling: TextureTiling,
) -> Result<SampledMipChain, DescriptorError> {
    let ImageExtent {
        width,
        height,
        depth,
    } = extent;
    let [block_height_log2, block_depth_log2] = block;
    let [bw, bh] = format.block_extent();
    let bytes = u64::from(format.plane_bytes_per_block(0).unwrap());
    let mut layouts = Vec::with_capacity(usize::from(levels));
    let mut offset = 0_u64;
    for level in 0..levels {
        let columns = (width >> level).max(1).div_ceil(bw);
        let rows = (height >> level).max(1).div_ceil(bh);
        let slices = (depth >> level).max(1);
        let mut gob_depth = block_depth_log2;
        let mut gob_height = block_height_log2;
        if levels > 1 {
            while gob_height > 0 && u64::from(rows) <= (8_u64 << (gob_height - 1)) {
                gob_height -= 1;
            }
        }
        if levels > 1 {
            while gob_depth > 0 && u64::from(slices) <= (1u64 << (gob_depth - 1)) {
                gob_depth -= 1;
            }
        }
        let padded_slices = align_up(u64::from(slices), 1u64 << gob_depth, descriptor_index)?;
        let size = align_up(u64::from(columns) * bytes, 64, descriptor_index)?
            .checked_mul(align_up(
                u64::from(rows),
                8_u64 << gob_height,
                descriptor_index,
            )?)
            .and_then(|size| size.checked_mul(padded_slices))
            .ok_or(DescriptorError {
                descriptor: descriptor_index,
                field: "storage overflow",
                value: 0,
            })?;
        layouts.push((
            offset,
            tiling.layout(BlockLinearLayout {
                block_width_log2: 0,
                block_height_log2: gob_height,
                block_depth_log2: gob_depth,
                layer_stride: size,
            }),
        ));
        offset = offset.checked_add(size).ok_or(DescriptorError {
            descriptor: descriptor_index,
            field: "storage overflow",
            value: 0,
        })?;
    }
    // Match Tegra's layer alignment, including compressed block-height rounding.
    let mut gob_height = block_height_log2;
    let aligned_height = u64::from(height.div_ceil(bh)) * u64::from(bh);
    while gob_height > 0 && aligned_height <= (8_u64 << (gob_height - 1)) {
        gob_height -= 1;
    }
    let mut gob_depth = block_depth_log2;
    while gob_depth > 0 && u64::from(depth) <= (1u64 << (gob_depth - 1)) {
        gob_depth -= 1;
    }
    let stride = align_up(
        offset,
        512_u64 << (gob_height + gob_depth),
        descriptor_index,
    )?;
    Ok(SampledMipChain {
        layouts,
        layer_stride: stride,
        layer_size: offset,
    })
}

struct SampledMipChain {
    layouts: Vec<(u64, ImageMemoryLayout)>,
    layer_stride: u64,
    layer_size: u64,
}
fn align_up(value: u64, alignment: u64, descriptor: u32) -> Result<u64, DescriptorError> {
    value
        .checked_add(alignment - 1)
        .map(|v| v & !(alignment - 1))
        .ok_or(DescriptorError {
            descriptor,
            field: "storage overflow",
            value: 0,
        })
}
