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
        ImageFormat::Bc4RUnorm | ImageFormat::Bc4RSnorm => (vk::Format::R16_SFLOAT, 2),
        ImageFormat::Bc5RgUnorm | ImageFormat::Bc5RgSnorm => (vk::Format::R16G16_SFLOAT, 4),
        ImageFormat::Bc1RgbSrgb
        | ImageFormat::Bc1RgbaSrgb
        | ImageFormat::Bc3RgbaSrgb
        | ImageFormat::Bc7RgbaSrgb => (vk::Format::R8G8B8A8_SRGB, 4),
        _ => (vk::Format::R8G8B8A8_UNORM, 4),
    }
}
impl VulkanResidency {
    /// Native BC handles non-block-aligned dimensions directly. Decode only when
    /// the host really lacks a format, rather than inheriting WebGPU size rules.
    pub fn sampled_image_for(
        &mut self,
        geometry: Vec<Arc<Footprint>>,
        description: NativeImageDescription,
        planes: Vec<ImagePlane>,
    ) -> Result<ImageId, VulkanError> {
        let support = unsafe {
            self.device
                .owner
                .instance
                .instance
                .get_physical_device_image_format_properties(
                    self.device.owner.physical,
                    description.format,
                    description.image_type,
                    vk::ImageTiling::OPTIMAL,
                    description.usage,
                    description.flags,
                )
        };
        let enabled = if codec(description.format).is_some_and(|f| f.astc_profile().is_some()) {
            self.device.profile().core.texture_compression_astc_ldr != 0
        } else if matches!(description.format.as_raw(), 131..=146) {
            self.device.profile().core.texture_compression_bc != 0
        } else {
            true
        };
        match support {
            Ok(_) if enabled => return self.image_for(geometry, description, planes),
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
            true,
            Some(DecodedImage {
                format: description.format,
                planes,
            }),
            [1, 1],
        )?;
        Ok(id)
    }
    pub(super) fn refresh_decoded_image(&mut self, id: ImageId) -> Result<(), VulkanError> {
        let image = self.image(id)?;
        let geometry = image.contents.geometry.clone();
        let encoded = image.decoded.as_ref().unwrap().clone();
        let native = image.planes.clone();
        let bytes = encoded
            .planes
            .iter()
            .map(|p| p.end().expect("encoded plane"))
            .max()
            .expect("encoded planes");
        let mut parts = Vec::new();
        for footprint in &geometry {
            parts.extend(self.authority.resolve_footprint(footprint)?);
        }
        let device = vec![
            self.authority
                .observe_native_resource(Resource::Image(id))?,
        ];
        let captured = if parts.iter().all(|p| p.producer.is_none()) {
            let mut inputs = Vec::new();
            for footprint in &geometry {
                footprint.mapping().observed_pages(footprint, &mut inputs)?;
            }
            let inputs = nixe_gpu::physical::pin_selected_pages(inputs)?;
            let copies: Vec<_> = parts
                .iter()
                .map(|p| nixe_memory::PhysicalCapture {
                    source: nixe_memory::Extent {
                        start: p.start,
                        bytes: p.bytes,
                    },
                    destination: p.logical as usize,
                })
                .collect();
            let mut bytes = vec![0; bytes as usize];
            self.memory
                .capture_resource_scattered(&inputs, &copies, &mut bytes)?;
            bytes
        } else {
            // Software decoding is an actual CPU consumer of the encoded bytes.
            // Materialize only those inputs through the versioned readback path;
            // it does not create a cached linear image representation.
            self.track_image_cpu(id)?;
            let mut inputs = Vec::new();
            for part in &parts {
                let mut input = None;
                for footprint in &geometry {
                    if let Some(range) =
                        footprint
                            .mapping()
                            .pin_physical(part.allocation, part.start, part.bytes)?
                    {
                        input = Some(range);
                        break;
                    }
                }
                inputs.push(
                    input.ok_or(VulkanError::InvalidResource("encoded image input retired"))?,
                );
            }
            let nixe_memory::AccessResult::Read(data) =
                self.access_memory(inputs, nixe_memory::AccessOperation::Read)?
            else {
                return Err(VulkanError::InvalidResource(
                    "encoded image read returned no bytes",
                ));
            };
            let mut captured = vec![0; bytes as usize];
            let mut offset = 0;
            for part in &parts {
                let length = part.bytes as usize;
                captured[part.logical as usize..part.logical as usize + length]
                    .copy_from_slice(&data[offset..offset + length]);
                offset += length;
            }
            captured
        };
        let mut decoder = std::mem::take(&mut self.decoder);
        let result = (|| {
            let format = codec(encoded.format).unwrap();
            // Decode into bounded upload rectangles, as Eden's conversion path
            // does for staging copies, rather than issuing a copy per block.
            // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/texture_cache/texture_cache.h
            let limit = self.device.submission.configuration.arena_bytes;
            let mut pixels = Vec::new();
            for (source, target) in encoded.planes.iter().zip(native.iter().copied()) {
                let (bw, bh) = (source.block_extent.width, source.block_extent.height);
                let pixel_bytes = u64::from(target.bytes_per_block);
                let block_bytes = u64::from(bw) * u64::from(bh) * pixel_bytes;
                if limit < block_bytes {
                    return Err(VulkanError::InvalidResource(
                        "upload arena cannot hold a decoded block",
                    ));
                }
                let columns = source.extent.width.div_ceil(bw);
                let tile_columns = columns.min((limit / block_bytes) as u32);
                let tile_width = (tile_columns * bw).min(source.extent.width);
                let tile_rows = (limit / (u64::from(tile_width) * pixel_bytes * u64::from(bh)))
                    .max(1)
                    .min(u64::from(source.rows())) as u32;
                for first_row in (0..source.rows()).step_by(tile_rows as usize) {
                    let rows = tile_rows.min(source.rows() - first_row);
                    let height = (rows * bh).min(source.extent.height - first_row * bh);
                    for first_column in (0..columns).step_by(tile_columns as usize) {
                        let count = tile_columns.min(columns - first_column);
                        let width = (count * bw).min(source.extent.width - first_column * bw);
                        let pitch = width as usize * pixel_bytes as usize;
                        pixels.resize(pitch * height as usize, 0);
                        for row in first_row..first_row + rows {
                            for column in first_column..first_column + count {
                                let offset = source.offset
                                    + u64::from(row) * source.row_pitch
                                    + u64::from(column) * u64::from(source.bytes_per_block);
                                let block = &captured[offset as usize
                                    ..offset as usize + source.bytes_per_block as usize];
                                let w = bw.min(source.extent.width - column * bw);
                                let h = bh.min(source.extent.height - row * bh);
                                let decoded = decoder.decode(format, w, h, block).map_err(|e| {
                                    VulkanError::TextureDecode(e.to_string().into())
                                })?;
                                let row_bytes = w as usize * pixel_bytes as usize;
                                for y in 0..h as usize {
                                    let to = ((row - first_row) * bh) as usize * pitch
                                        + y * pitch
                                        + ((column - first_column) * bw) as usize
                                            * pixel_bytes as usize;
                                    pixels[to..to + row_bytes].copy_from_slice(
                                        &decoded[y * row_bytes..(y + 1) * row_bytes],
                                    );
                                }
                            }
                        }
                        self.upload_decoded_rect(
                            id,
                            target,
                            vk::Rect2D {
                                offset: vk::Offset2D {
                                    x: (first_column * bw) as i32,
                                    y: (first_row * bh) as i32,
                                },
                                extent: vk::Extent2D { width, height },
                            },
                            &pixels,
                        )?;
                    }
                }
            }
            let work = self.device.work_ticket()?;
            let image = self.images[id.slot].entry.as_mut().unwrap();
            image.contents.device = device;
            image.synchronized = true;
            image.discarded.clear();
            image.producer = Some(work);
            image.version = image
                .version
                .checked_add(1)
                .ok_or(VulkanError::InvalidResource(
                    "decoded image version exhausted",
                ))?;
            Ok(())
        })();
        self.decoder = decoder;
        result
    }
    fn upload_decoded_rect(
        &mut self,
        id: ImageId,
        p: ImagePlane,
        rect: vk::Rect2D,
        pixels: &[u8],
    ) -> Result<(), VulkanError> {
        let (slot, slice) = self.reserve_upload(pixels, 4)?;
        let (image, layout) = self.image_access_subresources(
            id,
            commands::plane_range(p),
            AccessScope::new(
                vk::PipelineStageFlags2::COPY,
                vk::AccessFlags2::TRANSFER_WRITE,
            ),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        )?;
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
        self.barriers.emit(&mut self.device);
        unsafe {
            self.device.dispatch.cmd_copy_buffer_to_image(
                self.device.submission.slots[slot]
                    .upload_buffer(slice)
                    .handle,
                image,
                layout,
                &[region],
            );
        }
        self.define_image_rect(id, p, rect);
        self.statistics.cpu_uploaded_bytes += pixels.len() as u64;
        self.statistics.decoded_upload_commands += 1;
        Ok(())
    }
}
