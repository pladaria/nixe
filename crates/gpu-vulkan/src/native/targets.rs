use ash::vk;
// Shared physical image selection and attachment-view construction.
impl super::NativeExecution<'_> {
    pub(in crate::native) fn find_or_insert_image(
        &mut self,
        info: &nixe_gpu::native::ImageInfo,
        space: &(impl nixe_gpu::native::ImageAddressSpace + ?Sized),
    ) -> Result<(crate::ImageId, [u32; 2]), crate::VulkanError> {
        if let Some(found) = self.residency.find_image(info, space)? {
            return Ok(found);
        }
        let image = self.residency.insert_guest_image(info, space)?;
        Ok((image, [0, 0]))
    }
    pub(super) fn image_target_view(
        &mut self,
        info: &nixe_gpu::native::ImageInfo,
        space: &(impl nixe_gpu::native::ImageAddressSpace + ?Sized),
    ) -> Result<crate::ImageViewId, crate::VulkanError> {
        let (image, [base_mip, base_layer]) = self.find_or_insert_image(info, space)?;
        let aspects = self
            .residency
            .texture_cache
            .get(image)
            .planes
            .iter()
            .fold(vk::ImageAspectFlags::empty(), |a, p| a | p.aspect);
        self.residency.image_view(
            image,
            crate::ImageViewDescription {
                size: {
                    let e = info.description.extent();
                    vk::Extent3D {
                        width: e.width,
                        height: e.height,
                        depth: e.depth,
                    }
                },
                format: super::images::format(info.description.format()),
                // Attachment views expose array layers or volume slices;
                // the clear/draw operation selects the actual layer.
                view_type: if info.description.dimension() == nixe_gpu::ImageDimension::Three
                    || info.description.array_layers() > 1
                {
                    vk::ImageViewType::TYPE_2D_ARRAY
                } else {
                    vk::ImageViewType::TYPE_2D
                },
                aspects,
                base_mip,
                mip_count: 1,
                base_layer,
                layer_count: if info.description.dimension() == nixe_gpu::ImageDimension::Three {
                    info.description.extent().depth
                } else {
                    u32::from(info.description.array_layers())
                },
                swizzle: [vk::ComponentSwizzle::IDENTITY; 4],
            },
        )
    }
}
