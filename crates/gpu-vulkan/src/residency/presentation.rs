//! Native host consumers share guest residency, exact authority and hazards.
use super::*;
impl VulkanResidency {
    /// A scanout snapshot consumes encoded bytes, rather than the producer's
    /// sampling interpretation. Size-compatible color copies preserve those
    /// bytes exactly; opaque storage must already have a resident producer.
    /// https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdCopyImage.html
    pub(crate) fn resident_frame_source(
        &self,
        footprint: &Arc<Footprint>,
        description: NativeImageDescription,
        planes: &[ImagePlane],
    ) -> Option<ImageId> {
        let mut resolved = None;
        self.authority
            .resource_candidates(footprint)
            .into_iter()
            .find_map(|resource| {
                let Resource::Image(id) = resource else {
                    return None;
                };
                let entry = self.image(id).ok()?;
                let producer = entry.description;
                let plane = entry.planes.first()?;
                let requested = planes[0];
                let compatible = NativeImageDescription {
                    format: producer.format,
                    usage: producer.usage,
                    flags: producer.flags,
                    extent: producer.extent,
                    ..description
                };
                // A taller resident attachment can contain the exact visible scanout
                // prefix. Verify native geometry and the physical byte projection;
                // padding is never imported, sampled, or reported initialized.
                if entry.decoded.is_some()
                    || producer != compatible
                    || producer.extent.width != description.extent.width
                    || producer.extent.height < description.extent.height
                    || entry.planes.len() != 1
                    || plane.offset != 0
                    || requested.offset != 0
                    || plane.row_pitch != requested.row_pitch
                    || plane.bytes_per_block != requested.bytes_per_block
                    || plane.block_extent != requested.block_extent
                    || plane.aspect != requested.aspect
                    || format::copy_block(producer.format, vk::ImageAspectFlags::COLOR)
                        != format::copy_block(description.format, vk::ImageAspectFlags::COLOR)
                    || self
                        .require_resident_frame(
                            id,
                            [description.extent.width, description.extent.height],
                        )
                        .is_err()
                {
                    return None;
                }
                if entry.contents.geometry.as_slice() == std::slice::from_ref(footprint) {
                    return Some(id);
                }
                // A taller attachment or another mapping can expose the same
                // visible bytes. Prove each requested physical byte's current
                // native source; an overlap candidate alone is insufficient.
                let parts = resolved
                    .get_or_insert_with(|| self.authority.resolve_footprint(footprint).ok());
                parts
                    .as_ref()?
                    .iter()
                    .all(|part| {
                        part.producer.is_some_and(|source| {
                            source.resource == Resource::Image(id) && source.native == part.logical
                        })
                    })
                    .then_some(id)
            })
    }
    pub(crate) fn require_resident_frame(
        &self,
        image: ImageId,
        extent: [u32; 2],
    ) -> Result<(), VulkanError> {
        self.require_frame_source(image, extent, false)
    }
    /// CPU writes refresh an existing import; GPU alias invalidation must
    /// reselect its current producer instead of rebuilding an obsolete mirror.
    pub(crate) fn require_frame_source(
        &self,
        image: ImageId,
        extent: [u32; 2],
        allow_cpu_refresh: bool,
    ) -> Result<(), VulkanError> {
        let entry = self.image(image)?;
        let plane = entry.planes[0];
        let bytes = u64::from(extent[1]) * plane.row_pitch;
        if entry.producer.is_none()
            || extent[0] > plane.extent.width
            || extent[1] > plane.extent.height
            || entry
                .contents
                .invalid
                .iter()
                .any(|&(offset, length)| offset < bytes && length != 0)
            || entry
                .contents
                .device
                .iter()
                .any(ResourceObservation::is_invalid)
            || (!allow_cpu_refresh && entry.contents.cpu.iter().any(Observation::is_invalid))
        {
            return Err(VulkanError::InvalidResource(
                "opaque framebuffer has no current resident producer; canonical import is forbidden",
            ));
        }
        self.require_image_initialized(
            image,
            entry.planes[0],
            vk::Rect2D {
                offset: Default::default(),
                extent: vk::Extent2D {
                    width: extent[0],
                    height: extent[1],
                },
            },
        )
    }
    pub(crate) fn allocate_host_image(
        &mut self,
        extent: [u32; 2],
        format: vk::Format,
    ) -> Result<Arc<AllocatedImage>, VulkanError> {
        let description = NativeImageDescription {
            image_type: vk::ImageType::TYPE_2D,
            format,
            extent: vk::Extent3D {
                width: extent[0],
                height: extent[1],
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            usage: vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            flags: vk::ImageCreateFlags::empty(),
        };
        let owner = Arc::clone(&self.device.owner);
        Ok(Arc::new(allocate_image_charged(
            &owner,
            description,
            AllocationMode::Managed,
            "host presentation image",
            |bytes| self.reserve(bytes),
        )?))
    }
    pub(crate) fn host_source(
        &mut self,
        image: ImageId,
        extent: [u32; 2],
    ) -> Result<(Arc<AllocatedImage>, vk::ImageLayout), VulkanError> {
        if self.require_resident_frame(image, extent).is_err() {
            self.ensure_image(image)?;
        }
        let plane = self.image(image)?.planes[0];
        self.require_image_initialized(
            image,
            plane,
            vk::Rect2D {
                offset: Default::default(),
                extent: vk::Extent2D {
                    width: extent[0],
                    height: extent[1],
                },
            },
        )?;
        let (_, layout) = self.image_access_subresources(
            image,
            commands::plane_range(plane),
            AccessScope::new(
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_READ,
            ),
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        )?;
        self.barriers.emit(&mut self.device);
        let native = Arc::clone(&self.image(image)?.native);
        let work = self.device.work_ticket()?;
        self.device.retire_image(Arc::clone(&native), work)?;
        Ok((native, layout))
    }
}
