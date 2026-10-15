//! Exact ASTC LDR and BC decoding for hosts lacking the encoded format.
//! Linear ASTC retains FP16 texels; sRGB retains the specified UNORM8 decode.
//! https://registry.khronos.org/OpenGL/extensions/KHR/KHR_texture_compression_astc_ldr.txt
//! https://github.com/ARM-software/astc-encoder
use crate::ImageFormat;
use std::collections::HashMap;
#[derive(Debug)]
pub struct TextureDecodeError(String);
impl TextureDecodeError {
    fn failure(detail: String) -> Self {
        Self(detail)
    }
}
impl std::fmt::Display for TextureDecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for TextureDecodeError {}
fn unsupported(detail: &str) -> TextureDecodeError {
    TextureDecodeError(detail.into())
}
fn usize_from_u64(size: u64, label: &str) -> Result<usize, TextureDecodeError> {
    usize::try_from(size).map_err(|_| unsupported(label))
}
use ctt_astcenc::{Context, Flags, Preset, Profile, Swizzle, bindings, config_init};

#[derive(Default)]
pub struct BlockDecoder {
    contexts: HashMap<ImageFormat, Context>,
    // u16 alignment is required for the native half-float output.
    pixels: Vec<u16>,
}
impl BlockDecoder {
    pub fn decode(
        &mut self,
        format: ImageFormat,
        width: u32,
        height: u32,
        blocks: &[u8],
    ) -> Result<&[u8], TextureDecodeError> {
        if format.astc_profile().is_none() {
            return self.decode_bc(format, width, height, blocks);
        }
        let srgb = format
            .astc_profile()
            .ok_or_else(|| unsupported("ASTC decode format"))?;
        let [bw, bh] = format.block_extent();
        let expected = u64::from(width.div_ceil(bw))
            .checked_mul(u64::from(height.div_ceil(bh)))
            .and_then(|n| n.checked_mul(16))
            .ok_or_else(|| unsupported("ASTC block size overflow"))?;
        if width == 0 || height == 0 || blocks.len() as u64 != expected {
            return Err(unsupported("ASTC compressed block extent mismatch"));
        }
        let pixel_bytes = if srgb { 4_u64 } else { 8 };
        let size = u64::from(width)
            .checked_mul(u64::from(height))
            .and_then(|n| n.checked_mul(pixel_bytes))
            .ok_or_else(|| unsupported("ASTC output size overflow"))?;
        let size = usize_from_u64(size, "ASTC output size")?;
        self.pixels.resize(size / 2, 0);
        if let std::collections::hash_map::Entry::Vacant(entry) = self.contexts.entry(format) {
            let config = config_init(
                if srgb { Profile::LdrSrgb } else { Profile::Ldr },
                bw,
                bh,
                1,
                Preset::Fastest,
                Flags::DECOMPRESS_ONLY,
            )
            .map_err(|e| {
                TextureDecodeError::failure(format!("ASTC decoder configuration failed: {e}"))
            })?;
            entry.insert(Context::new(&config).map_err(|e| {
                TextureDecodeError::failure(format!("ASTC decoder allocation failed: {e}"))
            })?);
        }
        let context = self.contexts.get_mut(&format).unwrap();
        let mut slice = self.pixels.as_mut_ptr().cast::<std::ffi::c_void>();
        let mut output = bindings::astcenc_image {
            dim_x: width,
            dim_y: height,
            dim_z: 1,
            data_type: if srgb {
                bindings::astcenc_type_ASTCENC_TYPE_U8
            } else {
                bindings::astcenc_type_ASTCENC_TYPE_F16
            },
            data: &mut slice,
        };
        // The wrapper consumes raw pointers synchronously. The aligned allocation
        // contains exactly width * height * 4 components, with one live z slice.
        context
            .decompress(blocks, &mut output, Swizzle::IDENTITY)
            .map_err(|e| TextureDecodeError::failure(format!("ASTC decompression failed: {e}")))?;
        context
            .decompress_reset()
            .map_err(|e| TextureDecodeError::failure(format!("ASTC decoder reset failed: {e}")))?;
        // SAFETY: u16 has no invalid bit patterns. The FFI has finished writing;
        // the returned byte view borrows the same initialized allocation.
        Ok(unsafe { std::slice::from_raw_parts(self.pixels.as_ptr().cast::<u8>(), size) })
    }
}

impl BlockDecoder {
    fn decode_bc(
        &mut self,
        format: ImageFormat,
        width: u32,
        height: u32,
        blocks: &[u8],
    ) -> Result<&[u8], TextureDecodeError> {
        let [bw, bh] = format.block_extent();
        let block_bytes = usize::from(
            format
                .plane_bytes_per_block(0)
                .ok_or_else(|| unsupported("BC storage block size"))?,
        );
        let columns = width.div_ceil(bw);
        let expected = u64::from(columns)
            .checked_mul(u64::from(height.div_ceil(bh)))
            .and_then(|n| n.checked_mul(block_bytes as u64))
            .ok_or_else(|| unsupported("BC block size overflow"))?;
        if width == 0 || height == 0 || blocks.len() as u64 != expected {
            return Err(unsupported("BC compressed block extent mismatch"));
        }
        let channels = match format {
            ImageFormat::Bc4RUnorm | ImageFormat::Bc4RSnorm => 1,
            ImageFormat::Bc5RgUnorm | ImageFormat::Bc5RgSnorm => 2,
            ImageFormat::Bc1RgbUnorm
            | ImageFormat::Bc1RgbSrgb
            | ImageFormat::Bc1RgbaUnorm
            | ImageFormat::Bc1RgbaSrgb
            | ImageFormat::Bc3RgbaUnorm
            | ImageFormat::Bc3RgbaSrgb
            | ImageFormat::Bc7RgbaUnorm
            | ImageFormat::Bc7RgbaSrgb => 4,
            _ => return Err(unsupported("BC software import format")),
        };
        let words_per_pixel = if channels == 4 { 2 } else { channels };
        let size = u64::from(width)
            .checked_mul(u64::from(height))
            .and_then(|n| n.checked_mul(words_per_pixel as u64))
            .ok_or_else(|| unsupported("BC output size overflow"))?;
        self.pixels
            .resize(usize_from_u64(size, "BC output size")?, 0);
        for (index, encoded) in blocks.chunks_exact(block_bytes).enumerate() {
            let x = (index % columns as usize) as u32 * bw;
            let y = (index / columns as usize) as u32 * bh;
            let mut normalized = [0.0; 32];
            let mut rgba = [0; 64];
            match format {
                ImageFormat::Bc4RUnorm | ImageFormat::Bc4RSnorm => bcdec_rs::bc4_float(
                    encoded,
                    &mut normalized,
                    4,
                    format == ImageFormat::Bc4RSnorm,
                ),
                ImageFormat::Bc5RgUnorm | ImageFormat::Bc5RgSnorm => bcdec_rs::bc5_float(
                    encoded,
                    &mut normalized,
                    8,
                    format == ImageFormat::Bc5RgSnorm,
                ),
                ImageFormat::Bc7RgbaUnorm | ImageFormat::Bc7RgbaSrgb => {
                    bcdec_rs::bc7(encoded, &mut rgba, 16);
                }
                ImageFormat::Bc3RgbaUnorm | ImageFormat::Bc3RgbaSrgb => {
                    bcdec_rs::bc3(encoded, &mut rgba, 16)
                }
                _ => bcdec_rs::bc1(encoded, &mut rgba, 16),
            }
            if matches!(format, ImageFormat::Bc1RgbUnorm | ImageFormat::Bc1RgbSrgb) {
                for pixel in rgba.chunks_exact_mut(4) {
                    pixel[3] = 255;
                }
            }
            for row in 0..bh.min(height - y) {
                for column in 0..bw.min(width - x) {
                    let source = (row * bw + column) as usize * channels;
                    let destination = ((y + row) as usize * width as usize + (x + column) as usize)
                        * words_per_pixel;
                    if channels == 4 {
                        self.pixels[destination] =
                            u16::from_ne_bytes([rgba[source], rgba[source + 1]]);
                        self.pixels[destination + 1] =
                            u16::from_ne_bytes([rgba[source + 2], rgba[source + 3]]);
                    } else {
                        for channel in 0..channels {
                            self.pixels[destination + channel] =
                                half::f16::from_f32(normalized[source + channel]).to_bits();
                        }
                    }
                }
            }
        }
        // SAFETY: the complete initialized u16 allocation has no invalid bit
        // patterns, and its byte view stays borrowed until staging completes.
        Ok(unsafe {
            std::slice::from_raw_parts(self.pixels.as_ptr().cast::<u8>(), self.pixels.len() * 2)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(rgba: [u16; 4]) -> [u8; 16] {
        let mut block = [0xff; 16];
        block[0] = 0xfc;
        block[1] = 0xfd;
        for (component, value) in rgba.into_iter().enumerate() {
            block[8 + component * 2..10 + component * 2].copy_from_slice(&value.to_le_bytes());
        }
        block
    }

    #[test]
    fn clipped_astc_blocks_preserve_fp16_texels_and_reuse_decoder() {
        let mut decoder = BlockDecoder::default();
        let blocks = [
            [65535, 0, 0, 65535],
            [0, 65535, 0, 65535],
            [0, 0, 65535, 65535],
            [65535; 4],
        ]
        .map(solid)
        .concat();
        let expected = [
            [0x3c00_u16, 0, 0, 0x3c00],
            [0, 0x3c00, 0, 0x3c00],
            [0, 0, 0x3c00, 0x3c00],
            [0x3c00; 4],
        ];
        for _ in 0..2 {
            let pixels = decoder
                .decode(ImageFormat::Astc8x8Unorm, 13, 9, &blocks)
                .unwrap();
            assert_eq!(pixels.len(), 13 * 9 * 8);
            for (index, pixel) in pixels.chunks_exact(8).enumerate() {
                let block = usize::from(index % 13 >= 8) + 2 * usize::from(index / 13 >= 8);
                let actual: Vec<_> = pixel
                    .chunks_exact(2)
                    .map(|v| u16::from_ne_bytes([v[0], v[1]]))
                    .collect();
                assert_eq!(actual, expected[block]);
            }
        }
        assert_eq!(decoder.contexts.len(), 1);
        assert!(
            decoder
                .decode(ImageFormat::Astc8x8Unorm, 13, 9, &blocks[..16])
                .is_err()
        );
    }

    #[test]
    fn clipped_bc3_preserves_independent_alpha_and_four_colors_when_endpoints_are_reversed() {
        let block = [
            200, 60, 0x92, 0x24, 0x49, 0x92, 0x24, 0x49, 0, 0, 255, 255, 255, 255, 255, 255,
        ];
        let blocks = [block, [0; 16]].concat();
        for format in [ImageFormat::Bc3RgbaUnorm, ImageFormat::Bc3RgbaSrgb] {
            let mut decoder = BlockDecoder::default();
            let pixels = decoder.decode(format, 5, 3, &blocks).unwrap();
            assert_eq!(pixels.len(), 5 * 3 * 4);
            for (index, pixel) in pixels.chunks_exact(4).enumerate() {
                assert_eq!(
                    pixel,
                    if index % 5 < 4 {
                        &[170, 170, 170, 180]
                    } else {
                        &[0, 0, 0, 0]
                    }
                );
            }
        }
    }

    #[test]
    fn clipped_bc5_import_preserves_fractional_interpolation_without_unorm8_quantization() {
        let mut decoder = BlockDecoder::default();
        // Selector 2 uses the interpolated endpoint palette, not a stored byte.
        let block = [
            201, 60, 0x92, 0x24, 0x49, 0x92, 0x24, 0x49, 20, 220, 0, 0, 0, 0, 0, 0,
        ];
        let blocks = [block, [0; 16]].concat();
        let pixels = decoder
            .decode(ImageFormat::Bc5RgUnorm, 5, 3, &blocks)
            .unwrap();
        assert_eq!(pixels.len(), 5 * 3 * 4);
        for (index, pixel) in pixels.chunks_exact(4).enumerate() {
            let channels = pixel
                .chunks_exact(2)
                .map(|v| half::f16::from_bits(u16::from_ne_bytes([v[0], v[1]])).to_f32())
                .collect::<Vec<_>>();
            let expected = if index % 5 < 4 {
                [(6.0 * 201.0 + 60.0) / (7.0 * 255.0), 20.0 / 255.0]
            } else {
                [0.0; 2]
            };
            for (actual, expected) in channels.into_iter().zip(expected) {
                assert!((actual - expected).abs() < 0.0003);
            }
        }
    }

    #[test]
    fn clipped_bc7_import_preserves_rgba_endpoints_and_srgb_storage() {
        let block = [64, 32, 16, 4, 130, 64, 254, 127, 0, 0, 0, 0, 0, 0, 0, 0];
        for format in [ImageFormat::Bc7RgbaUnorm, ImageFormat::Bc7RgbaSrgb] {
            let mut decoder = BlockDecoder::default();
            let pixels = decoder
                .decode(format, 5, 3, &[block, block].concat())
                .unwrap();
            assert_eq!(pixels.len(), 60);
            assert!(pixels.chunks_exact(4).all(|p| p == [128, 64, 32, 254]));
        }
    }

    #[test]
    fn clipped_signed_rgtc_import_preserves_negative_endpoints_and_interpolation() {
        for format in [ImageFormat::Bc4RSnorm, ImageFormat::Bc5RgSnorm] {
            let block = [
                128, 127, 0x92, 0x24, 0x49, 0x92, 0x24, 0x49, 129, 127, 255, 255, 255, 255, 255,
                255,
            ];
            let channels = if format == ImageFormat::Bc4RSnorm {
                1
            } else {
                2
            };
            let bytes = channels * 8;
            let blocks = [block[..bytes].to_vec(), vec![0; bytes]].concat();
            let mut decoder = BlockDecoder::default();
            let pixels = decoder.decode(format, 5, 3, &blocks).unwrap();
            for (index, pixel) in pixels.chunks_exact(channels * 2).enumerate() {
                for (channel, word) in pixel.chunks_exact(2).enumerate() {
                    let actual =
                        half::f16::from_bits(u16::from_ne_bytes([word[0], word[1]])).to_f32();
                    let expected = if index % 5 == 4 {
                        0.0
                    } else if channel == 0 {
                        -0.6
                    } else {
                        1.0
                    };
                    assert!(
                        (actual - expected).abs() < 0.0003,
                        "{format:?} pixel={index} channel={channel}"
                    );
                }
            }
        }
    }

    #[test]
    fn clipped_bc1_import_preserves_transparency_and_srgb_bytes() {
        let mut decoder = BlockDecoder::default();
        let blocks = [
            [0, 0, 255, 255, 255, 255, 255, 255],
            [0x10, 0x84, 0, 0, 0, 0, 0, 0],
        ]
        .concat();
        let pixels = decoder
            .decode(ImageFormat::Bc1RgbaSrgb, 5, 3, &blocks)
            .unwrap();
        for (index, pixel) in pixels.chunks_exact(4).enumerate() {
            assert_eq!(
                pixel,
                if index % 5 < 4 {
                    &[0, 0, 0, 0]
                } else {
                    &[132, 130, 132, 255]
                }
            );
        }
    }

    #[test]
    fn srgb_astc_decode_preserves_unorm8_before_gpu_transfer_function() {
        let mut decoder = BlockDecoder::default();
        let block = solid([128 * 257, 64 * 257, 32 * 257, 65535]);
        let pixels = decoder
            .decode(ImageFormat::Astc5x4Srgb, 3, 2, &block)
            .unwrap();
        assert_eq!(pixels, [128, 64, 32, 255].repeat(6));
    }
}
