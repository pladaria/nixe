//! Video writes use ordinary image hazards, canonical authority and completion.
use super::*;
use crate::residency::commands::plane_range;

#[cfg(target_os = "linux")]
mod external;
#[cfg(target_os = "linux")]
pub(crate) use external::{VideoTransfers, device_uuid};
#[cfg(not(target_os = "linux"))]
#[derive(Default)]
pub(crate) struct VideoTransfers;
#[cfg(not(target_os = "linux"))]
pub(crate) fn device_uuid(_: &VulkanDevice) -> Option<[u8; 16]> {
    None
}
#[cfg(not(target_os = "linux"))]
impl VideoTransfers {
    pub(crate) fn import(
        &mut self,
        _: &mut VulkanResidency,
        _: [ImageId; 2],
        _: &nixe_gpu::DeviceImageFrame,
    ) -> Result<(), VulkanError> {
        Err(VulkanError::ExternalVideo(
            "external decoded video requires supported OS memory and semaphore handles".into(),
        ))
    }
}

impl VulkanResidency {
    /// The caller retains and synchronizes the external source through completion.
    pub(crate) unsafe fn video_plane_from_buffer(
        &mut self,
        image: ImageId,
        source: vk::Buffer,
        offset: u64,
        pitch: u64,
        rectangle: vk::Rect2D,
    ) -> Result<WorkTicket, VulkanError> {
        let plane = self.checked_image_rect(image, 0, rectangle)?;
        if plane.block_extent
            != (vk::Extent2D {
                width: 1,
                height: 1,
            })
            || pitch < u64::from(rectangle.extent.width) * u64::from(plane.bytes_per_block)
            || !pitch.is_multiple_of(u64::from(plane.bytes_per_block))
        {
            return Err(VulkanError::InvalidResource(
                "invalid decoded video plane pitch/format",
            ));
        }
        self.prepare_image_rect(image, plane, rectangle)?;
        let (target, layout) = self.image_access_subresources(
            image,
            plane_range(plane),
            AccessScope::new(
                vk::PipelineStageFlags2::COPY,
                vk::AccessFlags2::TRANSFER_WRITE,
            ),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        )?;
        self.barriers.emit(&mut self.device);
        let slot = self.device.begin_recording()?;
        let region = vk::BufferImageCopy::default()
            .buffer_offset(offset)
            .buffer_row_length((pitch / u64::from(plane.bytes_per_block)) as u32)
            .image_subresource(plane.layers())
            .image_offset(vk::Offset3D {
                x: rectangle.offset.x,
                y: rectangle.offset.y,
                z: 0,
            })
            .image_extent(vk::Extent3D {
                width: rectangle.extent.width,
                height: rectangle.extent.height,
                depth: 1,
            });
        unsafe {
            self.device.owner.device.cmd_copy_buffer_to_image(
                self.device.submission.slots[slot].command,
                source,
                target,
                layout,
                &[region],
            );
        }
        self.finish_image_rect(image, plane, rectangle)?;
        self.statistics.gpu_copied_bytes += u64::from(rectangle.extent.width)
            * u64::from(rectangle.extent.height)
            * u64::from(plane.bytes_per_block);
        self.device.work_ticket()
    }

    pub(crate) fn upload_video_plane(
        &mut self,
        image: ImageId,
        width: u32,
        bytes: &[u8],
    ) -> Result<(), VulkanError> {
        let plane = *self
            .image(image)?
            .planes
            .first()
            .ok_or(VulkanError::InvalidResource(
                "decoded video image has no plane",
            ))?;
        let pitch = u64::from(width) * u64::from(plane.bytes_per_block);
        if pitch == 0 || bytes.len() as u64 != pitch * u64::from(plane.extent.height) {
            return Err(VulkanError::InvalidResource(
                "decoded video bytes differ from visible dimensions",
            ));
        }
        let chunk_bytes = (self.device.submission.configuration.arena_bytes / pitch) * pitch;
        if chunk_bytes == 0 {
            return Err(VulkanError::InvalidResource(
                "video row exceeds upload arena",
            ));
        }
        for (chunk, bytes) in bytes.chunks(chunk_bytes as usize).enumerate() {
            let (slot, slice) = self.reserve_upload(bytes, 4)?;
            let source = self.device.submission.slots[slot]
                .upload_buffer(slice)
                .handle;
            let rectangle = vk::Rect2D {
                offset: vk::Offset2D {
                    x: 0,
                    y: ((chunk as u64 * chunk_bytes) / pitch) as i32,
                },
                extent: vk::Extent2D {
                    width,
                    height: (bytes.len() as u64 / pitch) as u32,
                },
            };
            unsafe {
                self.video_plane_from_buffer(image, source, slice.offset, pitch, rectangle)?;
            }
            self.statistics.cpu_uploaded_bytes += bytes.len() as u64;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::residency::tests::{backing_allocation, cache, cpu_read};

    #[test]
    fn decoded_planes_preserve_pitch_padding_and_publish_only_visible_texels() {
        let Some(mut cache) = cache(crate::ApiPreference::Auto, 8 * 1024 * 1024) else {
            return;
        };
        let validation = cache.device.validation_monitor();
        cache.device.submission = crate::queue::QueueState::new(
            Arc::clone(&cache.device.owner),
            cache.device.queues().graphics_compute_family,
            crate::QueueConfiguration {
                command_slots: 2,
                arena_bytes: 512,
                overflow_bytes: 0,
            },
        )
        .unwrap();
        // Odd visible dimensions and a padded destination exercise both chroma
        // rounding and partial-plane producer coverage. A tiny arena splits rows.
        for (format, bpp, width, height) in [
            (vk::Format::R8_UNORM, 1, 67, 37),
            (vk::Format::R8G8_UNORM, 2, 34, 19),
        ] {
            let (_, backing) = backing_allocation(8192, 4096, 0xa5);
            let extent = vk::Extent3D {
                width: 128 / bpp,
                height,
                depth: 1,
            };
            let plane = ImagePlane {
                offset: 32,
                row_pitch: 128,
                mip_level: 0,
                array_layer: 0,
                aspect: vk::ImageAspectFlags::COLOR,
                extent,
                block_extent: vk::Extent2D {
                    width: 1,
                    height: 1,
                },
                bytes_per_block: bpp,
            };
            let image = cache
                .image_for_deferred_import(
                    backing.clone(),
                    NativeImageDescription {
                        image_type: vk::ImageType::TYPE_2D,
                        format,
                        extent,
                        mip_levels: 1,
                        array_layers: 1,
                        samples: vk::SampleCountFlags::TYPE_1,
                        usage: vk::ImageUsageFlags::SAMPLED,
                        flags: vk::ImageCreateFlags::empty(),
                    },
                    vec![plane],
                )
                .unwrap();
            let data: Vec<_> = (0..width * height * bpp)
                .map(|i| (i * 17 + 3) as u8)
                .collect();
            let before = cache.statistics();
            cache.upload_video_plane(image, width, &data).unwrap();
            assert_eq!(
                cache.statistics().cpu_uploaded_bytes - before.cpu_uploaded_bytes,
                data.len() as u64
            );
            assert_eq!(
                cache.statistics().native_readback_bytes,
                before.native_readback_bytes
            );
            assert_eq!(
                cache.statistics().canonical_readback_bytes,
                before.canonical_readback_bytes
            );
            cache.flush().unwrap();
            let output = cpu_read(&mut cache, &backing);
            let mut expected = vec![0xa5; 8192];
            for row in 0..height as usize {
                let pitch = (width * bpp) as usize;
                expected[32 + row * 128..32 + row * 128 + pitch]
                    .copy_from_slice(&data[row * pitch..(row + 1) * pitch]);
            }
            assert_eq!(output, expected);
            assert!(
                cache
                    .upload_video_plane(image, width, &data[..data.len() - 1])
                    .is_err()
            );
        }
        drop(cache);
        validation.check().unwrap();
    }
}
