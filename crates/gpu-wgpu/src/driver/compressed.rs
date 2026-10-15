//! Cached imports for ASTC LDR and BC textures with non-aligned base extents.
//! WebGPU requires BC base dimensions to be block-aligned; expanding texels
//! instead preserves the guest dimensions, addressing and filtering.
//! https://www.w3.org/TR/webgpu/#dom-gputexturedescriptor-size
//! BC decoding uses https://docs.rs/bcdec_rs/0.2.0/bcdec_rs/
//! Decode only dirty blocks, retain decoder contexts, and filter the resulting
//! texture on the GPU. Linear ASTC uses fp16 texels; sRGB uses the mandated
//! 8-bit decode before the host sampler applies the sRGB transfer function.
//! https://registry.khronos.org/OpenGL/extensions/KHR/KHR_texture_compression_astc_ldr.txt
//! https://github.com/ARM-software/astc-encoder
use super::*;
pub(super) use nixe_gpu::texture_decode::BlockDecoder;

pub(super) fn decoded_texture_format(description: ImageDescription) -> Option<TextureFormat> {
    let format = description.format();
    if format.astc_profile().is_some() {
        return texture_format(format);
    }
    let [bw, bh] = format.block_extent();
    if description.extent().width.is_multiple_of(bw)
        && description.extent().height.is_multiple_of(bh)
    {
        return None;
    }
    Some(match format {
        ImageFormat::Bc4RUnorm | ImageFormat::Bc4RSnorm => TextureFormat::R16Float,
        ImageFormat::Bc5RgUnorm | ImageFormat::Bc5RgSnorm => TextureFormat::Rg16Float,
        ImageFormat::Bc1RgbSrgb
        | ImageFormat::Bc1RgbaSrgb
        | ImageFormat::Bc3RgbaSrgb
        | ImageFormat::Bc7RgbaSrgb => TextureFormat::Rgba8UnormSrgb,
        ImageFormat::Bc1RgbUnorm
        | ImageFormat::Bc1RgbaUnorm
        | ImageFormat::Bc3RgbaUnorm
        | ImageFormat::Bc7RgbaUnorm => TextureFormat::Rgba8Unorm,
        _ => return None,
    })
}

impl WgpuBackendDriver {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn upload_decoded_region(
        &mut self,
        texture: &Texture,
        description: ImageDescription,
        domain: &ImageContentDomain,
        canonical: &nixe_memory::CanonicalByteSnapshots,
        region: ImageUploadRegion,
        columns: u32,
        encoder: &mut CommandEncoder,
    ) -> Result<(), BackendDriverError> {
        let format = description.format();
        let [bw, bh] = format.block_extent();
        let extent = description
            .mip_extent(domain.subresources.mip_level)
            .ok_or_else(|| unsupported("decoded texture upload mip"))?;
        let origin = Origin3d {
            x: region.x * bw,
            y: region.y * bh,
            z: u32::from(domain.subresources.base_layer),
        };
        let width = (region.width * bw).min(extent.width - origin.x);
        let height = (region.height * bh).min(extent.height - origin.y);
        let bytes_per_block = u32::from(
            format
                .plane_bytes_per_block(0)
                .ok_or_else(|| unsupported("decoded texture storage block size"))?,
        );
        let mut blocks = std::mem::take(&mut self.upload_canonical);
        let encoded_pitch = region
            .width
            .checked_mul(bytes_per_block)
            .ok_or_else(|| unsupported("decoded texture block row size overflow"))?;
        blocks.resize(
            usize_from_u64(
                u64::from(encoded_pitch) * u64::from(region.height),
                "encoded texture block size",
            )?,
            0,
        );
        linearize_snapshot_region(
            canonical,
            blocks.as_mut_slice().into(),
            domain.layout,
            region,
            columns,
            bytes_per_block as usize,
            encoded_pitch,
        )?;
        let mut decoder = std::mem::take(&mut self.block_decoder);
        let result = (|| {
            let decoded = decoder
                .decode(format, width, height, &blocks)
                .map_err(|e| BackendDriverError::failure(e.to_string()))?;
            let texel_bytes = decoded_texture_format(description)
                .and_then(|format| format.block_copy_size(None))
                .ok_or_else(|| unsupported("decoded texture host texel size"))?;
            let row_bytes = width
                .checked_mul(texel_bytes)
                .ok_or_else(|| unsupported("ASTC decoded row size overflow"))?;
            let row_pitch = align_u32(row_bytes, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)?;
            nixe_gpu::metrics::record(
                nixe_gpu::metrics::Counter::ImageLinearizedBytes,
                blocks.len() as u64,
            );
            nixe_gpu::metrics::record(
                nixe_gpu::metrics::Counter::ImageUploadedBytes,
                decoded.len() as u64,
            );
            self.stage_texture_upload(
                encoder,
                TexelCopyTextureInfo {
                    texture,
                    mip_level: u32::from(domain.subresources.mip_level),
                    origin,
                    aspect: TextureAspect::All,
                },
                row_pitch as usize * height as usize,
                TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row_pitch),
                    rows_per_image: Some(height),
                },
                Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                |mut output| {
                    for (y, row) in decoded.chunks_exact(row_bytes as usize).enumerate() {
                        output
                            .slice(y * row_pitch as usize..y * row_pitch as usize + row.len())
                            .copy_from_slice(row);
                    }
                    Ok(())
                },
            )
        })();
        self.block_decoder = decoder;
        self.upload_canonical = blocks;
        result
    }
}
