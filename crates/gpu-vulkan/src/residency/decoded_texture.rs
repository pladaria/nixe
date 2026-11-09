//! Encoded bytes stay canonical. A decoded, read-only sampled representation
//! observes those bytes but never claims to contain re-encodable GPU producers.
use super::*;
use nixe_gpu::ImageFormat;
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct DecodedImage {
    pub format: vk::Format,
    pub planes: Vec<ImagePlane>,
}
fn codec(format: vk::Format) -> Option<ImageFormat> {
    use ImageFormat::*;
    Some(match format {
        vk::Format::BC1_RGB_UNORM_BLOCK => Bc1RgbUnorm,
        vk::Format::BC1_RGB_SRGB_BLOCK => Bc1RgbSrgb,
        vk::Format::BC1_RGBA_UNORM_BLOCK => Bc1RgbaUnorm,
        vk::Format::BC1_RGBA_SRGB_BLOCK => Bc1RgbaSrgb,
        vk::Format::BC3_UNORM_BLOCK => Bc3RgbaUnorm,
        vk::Format::BC3_SRGB_BLOCK => Bc3RgbaSrgb,
        vk::Format::BC4_UNORM_BLOCK => Bc4RUnorm,
        vk::Format::BC4_SNORM_BLOCK => Bc4RSnorm,
        vk::Format::BC5_UNORM_BLOCK => Bc5RgUnorm,
        vk::Format::BC5_SNORM_BLOCK => Bc5RgSnorm,
        vk::Format::BC7_UNORM_BLOCK => Bc7RgbaUnorm,
        vk::Format::BC7_SRGB_BLOCK => Bc7RgbaSrgb,
        vk::Format::BC6H_UFLOAT_BLOCK => Bc6hRgbUfloat,
        vk::Format::BC6H_SFLOAT_BLOCK => Bc6hRgbSfloat,
        vk::Format::ASTC_4X4_UNORM_BLOCK => Astc4x4Unorm,
        vk::Format::ASTC_4X4_SRGB_BLOCK => Astc4x4Srgb,
        vk::Format::ASTC_5X4_UNORM_BLOCK => Astc5x4Unorm,
        vk::Format::ASTC_5X4_SRGB_BLOCK => Astc5x4Srgb,
        vk::Format::ASTC_5X5_UNORM_BLOCK => Astc5x5Unorm,
        vk::Format::ASTC_5X5_SRGB_BLOCK => Astc5x5Srgb,
        vk::Format::ASTC_6X5_UNORM_BLOCK => Astc6x5Unorm,
        vk::Format::ASTC_6X5_SRGB_BLOCK => Astc6x5Srgb,
        vk::Format::ASTC_6X6_UNORM_BLOCK => Astc6x6Unorm,
        vk::Format::ASTC_6X6_SRGB_BLOCK => Astc6x6Srgb,
        vk::Format::ASTC_8X5_UNORM_BLOCK => Astc8x5Unorm,
        vk::Format::ASTC_8X5_SRGB_BLOCK => Astc8x5Srgb,
        vk::Format::ASTC_8X6_UNORM_BLOCK => Astc8x6Unorm,
        vk::Format::ASTC_8X6_SRGB_BLOCK => Astc8x6Srgb,
        vk::Format::ASTC_8X8_UNORM_BLOCK => Astc8x8Unorm,
        vk::Format::ASTC_8X8_SRGB_BLOCK => Astc8x8Srgb,
        vk::Format::ASTC_10X5_UNORM_BLOCK => Astc10x5Unorm,
        vk::Format::ASTC_10X5_SRGB_BLOCK => Astc10x5Srgb,
        vk::Format::ASTC_10X6_UNORM_BLOCK => Astc10x6Unorm,
        vk::Format::ASTC_10X6_SRGB_BLOCK => Astc10x6Srgb,
        vk::Format::ASTC_10X8_UNORM_BLOCK => Astc10x8Unorm,
        vk::Format::ASTC_10X8_SRGB_BLOCK => Astc10x8Srgb,
        vk::Format::ASTC_10X10_UNORM_BLOCK => Astc10x10Unorm,
        vk::Format::ASTC_10X10_SRGB_BLOCK => Astc10x10Srgb,
        vk::Format::ASTC_12X10_UNORM_BLOCK => Astc12x10Unorm,
        vk::Format::ASTC_12X10_SRGB_BLOCK => Astc12x10Srgb,
        vk::Format::ASTC_12X12_UNORM_BLOCK => Astc12x12Unorm,
        vk::Format::ASTC_12X12_SRGB_BLOCK => Astc12x12Srgb,
        _ => return None,
    })
}
fn decoded_format(format: ImageFormat) -> (vk::Format, u32) {
    if let Some(srgb) = format.astc_profile() {
        return if srgb {
            (vk::Format::R8G8B8A8_SRGB, 4)
        } else {
            (vk::Format::R16G16B16A16_SFLOAT, 8)
        };
    }
    match format {
        ImageFormat::Bc6hRgbUfloat | ImageFormat::Bc6hRgbSfloat => {
            (vk::Format::R16G16B16A16_SFLOAT, 8)
        }
        ImageFormat::Bc4RUnorm | ImageFormat::Bc4RSnorm => (vk::Format::R16_SFLOAT, 2),
        ImageFormat::Bc5RgUnorm | ImageFormat::Bc5RgSnorm => (vk::Format::R16G16_SFLOAT, 4),
        ImageFormat::Bc1RgbSrgb
        | ImageFormat::Bc1RgbaSrgb
        | ImageFormat::Bc3RgbaSrgb
        | ImageFormat::Bc7RgbaSrgb => (vk::Format::R8G8B8A8_SRGB, 4),
        _ => (vk::Format::R8G8B8A8_UNORM, 4),
    }
}
impl Residency<'_> {
    /// Native BC handles non-block-aligned dimensions directly. Decode only when
    /// the host really lacks a format, rather than inheriting WebGPU size rules.
    pub fn sampled_image_for(
        &mut self,
        geometry: Vec<Arc<Footprint>>,
        description: NativeImageDescription,
        planes: Vec<ImagePlane>,
        guest_source: Option<texture_cache::ImageRegistration>,
    ) -> Result<ImageId, VulkanError> {
        let support = unsafe {
            self.runtime
                .device
                .owner
                .instance
                .instance
                .get_physical_device_image_format_properties(
                    self.runtime.device.owner.physical,
                    description.format,
                    description.image_type,
                    vk::ImageTiling::OPTIMAL,
                    description.usage,
                    description.flags,
                )
        };
        let enabled = if codec(description.format).is_some_and(|f| f.astc_profile().is_some()) {
            self.runtime
                .device
                .profile()
                .core
                .texture_compression_astc_ldr
                != 0
        } else if matches!(description.format.as_raw(), 131..=146) {
            self.runtime.device.profile().core.texture_compression_bc != 0
        } else {
            true
        };
        match support {
            Ok(_) if enabled => return self.image_for(geometry, description, planes, guest_source),
            Err(e) if e != vk::Result::ERROR_FORMAT_NOT_SUPPORTED => {
                return Err(VulkanError::vk("sampled image format query", e));
            }
            _ => {}
        }
        let format = codec(description.format).ok_or(VulkanError::InvalidResource(
            "host sampled format unavailable and no exact decoder exists",
        ))?;
        if description.usage.intersects(
            vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                | vk::ImageUsageFlags::STORAGE,
        ) {
            return Err(VulkanError::InvalidResource(
                "compressed decode supports sampled imports, not writable encoded images",
            ));
        }
        let (native, bytes) = decoded_format(format);
        let mut target = description;
        target.format = native;
        target.usage = vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::TRANSFER_SRC;
        let native_planes = planes
            .iter()
            .map(|p| ImagePlane {
                offset: 0,
                row_pitch: u64::from(p.extent.width) * u64::from(bytes),
                block_extent: vk::Extent2D {
                    width: 1,
                    height: 1,
                },
                bytes_per_block: bytes,
                ..*p
            })
            .collect();
        let id = self.create_image_representation(
            geometry,
            target,
            native_planes,
            Some(DecodedImage {
                format: description.format,
                planes,
            }),
            [1, 1],
            guest_source,
        )?;
        Ok(id)
    }
    pub(super) fn refresh_decoded_image(&mut self, id: ImageId) -> Result<(), VulkanError> {
        let image = self.texture_cache.get(id);
        let geometry = image.contents.geometry.clone();
        let encoded = image.decoded.as_ref().unwrap().clone();
        let native = image.planes.clone();
        let bytes = encoded
            .planes
            .iter()
            .map(|p| p.end().expect("encoded plane"))
            .max()
            .expect("encoded planes");
        let mut inputs = Vec::new();
        for footprint in &geometry {
            footprint.observed_pages(footprint.mapping(), &mut inputs)?;
        }
        let inputs = nixe_gpu::physical::pin_selected_pages(inputs)?;
        let mut captured = vec![0; bytes as usize];
        self.runtime.memory.read_resources(&inputs, |reader| {
            for footprint in &geometry {
                footprint.capture_resource(reader, 0, &mut captured)?;
            }
            Ok(())
        })?;
        let mut decoder = std::mem::take(&mut self.texture_cache.decoder);
        let result = (|| {
            let format = codec(encoded.format).unwrap();
            // Decode and upload each plane as one request; staging can grow
            // instead of splitting the image to fit a command-slot arena.
            let mut pixels = Vec::new();
            for (source, target) in encoded.planes.iter().zip(native.iter().copied()) {
                let (bw, bh) = (source.block_extent.width, source.block_extent.height);
                let pixel_bytes = target.bytes_per_block as usize;
                let pitch = source.extent.width as usize * pixel_bytes;
                pixels.resize(pitch * source.extent.height as usize, 0);
                for row in 0..source.rows() {
                    for column in 0..source.extent.width.div_ceil(bw) {
                        let offset = source.offset
                            + u64::from(row) * source.row_pitch
                            + u64::from(column) * u64::from(source.bytes_per_block);
                        let block = &captured
                            [offset as usize..offset as usize + source.bytes_per_block as usize];
                        let width = bw.min(source.extent.width - column * bw);
                        let height = bh.min(source.extent.height - row * bh);
                        let decoded = decoder
                            .decode(format, width, height, block)
                            .map_err(|e| VulkanError::TextureDecode(e.to_string().into()))?;
                        let row_bytes = width as usize * pixel_bytes;
                        for y in 0..height as usize {
                            let to = (row * bh) as usize * pitch
                                + y * pitch
                                + (column * bw) as usize * pixel_bytes;
                            pixels[to..to + row_bytes]
                                .copy_from_slice(&decoded[y * row_bytes..(y + 1) * row_bytes]);
                        }
                    }
                }
                self.upload_decoded_rect(
                    id,
                    target,
                    vk::Rect2D {
                        offset: vk::Offset2D::default(),
                        extent: vk::Extent2D {
                            width: source.extent.width,
                            height: source.extent.height,
                        },
                    },
                    &pixels,
                )?;
            }
            let image = &mut self.texture_cache.images[id];
            image.synchronized = true;
            Ok(())
        })();
        self.texture_cache.decoder = decoder;
        result
    }
    fn upload_decoded_rect(
        &mut self,
        id: ImageId,
        p: ImagePlane,
        rect: vk::Rect2D,
        pixels: &[u8],
    ) -> Result<(), VulkanError> {
        let slice = self
            .runtime
            .device
            .submission
            .reserve_upload_aligned(pixels, 4)?;
        let (image, layout) = self.transfer_image(id, commands::plane_range(p))?;
        let region = vk::BufferImageCopy::default()
            .buffer_offset(slice.offset)
            .image_subresource(p.layers())
            .image_offset(vk::Offset3D {
                x: rect.offset.x,
                y: rect.offset.y,
                z: p.depth_slice as i32,
            })
            .image_extent(vk::Extent3D {
                width: rect.extent.width,
                height: rect.extent.height,
                depth: 1,
            });
        self.runtime.barriers.emit(&mut self.runtime.device)?;
        unsafe {
            self.runtime.device.dispatch.copy_buffer_to_image(
                self.runtime.device.submission.staging_buffer(&slice).handle,
                image,
                layout,
                &[region],
            );
        }
        self.runtime.statistics.cpu_uploaded_bytes += pixels.len() as u64;
        self.runtime.statistics.decoded_upload_commands += 1;
        Ok(())
    }
}
