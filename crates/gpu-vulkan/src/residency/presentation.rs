//! Native host consumers share guest residency, exact authority and hazards.
use super::*;
impl VulkanResidency {
    pub(crate) fn presentation_source(
        &mut self,
        source: &nixe_gpu::native::NativePresentationImage,
        format: vk::Format,
    ) -> Result<ImageId, VulkanError> {
        self.invalidate_cpu_resources(&[])?;
        let [plane] = source.planes.as_ref() else {
            return Err(VulkanError::InvalidResource(
                "framebuffer requires one color plane",
            ));
        };
        let description = NativeImageDescription {
            image_type: vk::ImageType::TYPE_2D,
            format,
            extent: vk::Extent3D {
                width: source.extent[0],
                height: source.extent[1],
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            usage: vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::SAMPLED,
            flags: vk::ImageCreateFlags::empty(),
        };
        let native_plane = ImagePlane {
            offset: 0,
            row_pitch: plane.row_pitch,
            mip_level: 0,
            array_layer: 0,
            extent: description.extent,
            aspect: vk::ImageAspectFlags::COLOR,
            block_extent: vk::Extent2D {
                width: 1,
                height: 1,
            },
            bytes_per_block: plane.bytes_per_block,
        };
        if let Some(id) = self.resident_frame_source(&plane.footprint, description, &[native_plane])
        {
            return Ok(id);
        }
        if !source.allow_canonical_import {
            return Err(VulkanError::InvalidResource(
                "opaque framebuffer requires a matching native resident representation",
            ));
        }
        self.image_for(
            vec![Arc::clone(&plane.footprint)],
            description,
            vec![native_plane],
        )
    }
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
        let entry = self.image(image)?;
        let plane = entry.planes[0];
        if entry.producer.is_none()
            || extent[0] > plane.extent.width
            || extent[1] > plane.extent.height
        {
            return Err(VulkanError::InvalidResource(
                "opaque framebuffer has no current resident producer; canonical import is forbidden",
            ));
        }
        if !entry.current() {
            // CPU dirtiness is page-granular; scanout authority is byte-exact.
            // An unrelated store in the same page must not invalidate opaque
            // pixels whose complete visible coverage still belongs here.
            let footprint = &entry.contents.geometry[0];
            let nixe_gpu::physical::WriteDomain::Image(mut domain) = footprint.domain() else {
                unreachable!("image footprint")
            };
            domain.rect.width = extent[0].div_ceil(plane.block_extent.width);
            domain.rect.height = extent[1].div_ceil(plane.block_extent.height);
            let visible = Footprint::new(
                Arc::clone(footprint.mapping()),
                nixe_gpu::physical::WriteDomain::Image(domain),
            )?;
            if self.authority.covering_resource(&visible)? != Some(Resource::Image(image)) {
                return Err(VulkanError::InvalidResource(
                    "framebuffer visible bytes have a newer CPU or alias producer",
                ));
            }
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
