//! Native image representations and persistent compatible views.
use super::*;
mod aliases;

#[derive(Clone)]
pub(super) struct PlaneCopies {
    source: ImagePlane,
    destination: ImagePlane,
    copies: Vec<vk::ImageCopy>,
}
pub(super) struct ImageAlias {
    pub image: ImageId,
    pub copies: Arc<[PlaneCopies]>,
}
impl Residency<'_> {
    /// Register canonical storage; actual LOAD or shader consumption imports it.
    pub fn image_for(
        &mut self,
        geometry: Vec<Arc<Footprint>>,
        description: NativeImageDescription,
        planes: Vec<ImagePlane>,
        guest_source: Option<texture_cache::ImageRegistration>,
    ) -> Result<ImageId, VulkanError> {
        self.create_image_representation(geometry, description, planes, None, [1, 1], guest_source)
    }
    /// Device-authored samples share canonical ownership with ordinary images.
    /// Their sample grid is supplied by the frontend, independently of host layout.
    pub fn multisample_image_for(
        &mut self,
        geometry: Vec<Arc<Footprint>>,
        mut description: NativeImageDescription,
        planes: Vec<ImagePlane>,
        sample_grid: [u32; 2],
        guest_source: Option<texture_cache::ImageRegistration>,
    ) -> Result<ImageId, VulkanError> {
        description.usage |= vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
        // CPU sample visibility reads encoded bits through a compatible UNORM
        // view, avoiding an approximate inverse of the sampler's sRGB decode.
        if matches!(
            description.format,
            vk::Format::R8G8B8A8_SRGB | vk::Format::B8G8R8A8_SRGB
        ) {
            description.flags |= vk::ImageCreateFlags::MUTABLE_FORMAT;
        }
        self.create_image_representation(
            geometry,
            description,
            planes,
            None,
            sample_grid,
            guest_source,
        )
    }
    pub(super) fn create_image_representation(
        &mut self,
        geometry: Vec<Arc<Footprint>>,
        mut description: NativeImageDescription,
        planes: Vec<ImagePlane>,
        decoded: Option<super::decoded_texture::DecodedImage>,
        sample_grid: [u32; 2],
        guest_source: Option<texture_cache::ImageRegistration>,
    ) -> Result<ImageId, VulkanError> {
        description.usage |= vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
        if self.runtime.device.profile().attachment_feedback_loop
            && description.usage.contains(vk::ImageUsageFlags::SAMPLED)
            && !description.usage.contains(vk::ImageUsageFlags::STORAGE)
            && description.usage.intersects(
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            )
        {
            description.usage |= vk::ImageUsageFlags::ATTACHMENT_FEEDBACK_LOOP_EXT;
        }
        if let Some(decoded) = &decoded {
            let mut encoded = description;
            encoded.format = decoded.format;
            validate_planes(&geometry, encoded, &decoded.planes, [1, 1])?;
        } else {
            validate_planes(&geometry, description, &planes, sample_grid)?;
        }
        if geometry
            .iter()
            .any(|footprint| footprint.mapping().machine() != self.runtime.memory.identity())
        {
            return Err(VulkanError::InvalidResource(
                "image belongs to another physical machine",
            ));
        }
        let mut maps = RangeSet::default();
        let cached = if let Some(source) = guest_source {
            for (address, bytes) in source.maps {
                maps.insert(address, bytes);
            }
            Some(texture_cache::CachedImage::new(
                source.info,
                source.address_space,
                source.cpu_address,
            ))
        } else {
            for footprint in &geometry {
                let bounds = footprint.bounds();
                if let Some(address) = self
                    .runtime
                    .device_memory
                    .find_address(footprint.mapping(), bounds.clone())?
                {
                    maps.insert(address, bounds.end - bounds.start);
                }
            }
            None
        };
        let candidate = if cached.is_none() {
            let mut candidates = smallvec::SmallVec::<[ImageId; 16]>::new();
            for (address, size) in maps.iter() {
                for page in address >> 20..=(address + size - 1) >> 20 {
                    if let Some(ids) = self.texture_cache.image_pages.get(&page) {
                        for &id in ids {
                            if !candidates.contains(&id) {
                                candidates.push(id);
                            }
                        }
                    }
                }
            }
            candidates.into_iter().find(|&id| {
                let entry = self.texture_cache.get(id);
                let compatible = NativeImageDescription {
                    usage: entry.description.usage,
                    flags: entry.description.flags,
                    ..description
                };
                entry.cached.is_none()
                    && entry.description == compatible
                    && entry.description.usage.contains(description.usage)
                    && entry.description.flags.contains(description.flags)
                    && entry.decoded == decoded
                    && entry.sample_grid == sample_grid
                    && entry.planes.as_ref() == planes
                    && entry.contents.geometry == geometry
            })
        } else {
            None
        };
        if let Some(id) = candidate {
            self.touch(Resource::Image(id));
            self.runtime.statistics.cache_hits += 1;
            return Ok(id);
        }
        let owner = Arc::clone(&self.runtime.device.owner);
        let native = allocate_image_charged(
            &owner,
            description,
            AllocationMode::Managed,
            "resident image",
            |bytes| self.charge(bytes),
        )?;
        let id = self.texture_cache.images.insert(ImageEntry {
            synchronized: false,
            aliases: Vec::new(),
            cpu_tracked: false,
            maps: maps.iter().collect(),
            cached,
            modification_tick: 0,
            gpu_modified: false,
            forced_flushed: false,
            dma_downloaded: false,
            contents: Contents {
                lru: None,
                geometry: geometry.clone(),
                last_use: None,
                touched: 0,
            },
            native: Arc::new(native),
            description,
            planes: planes.clone().into(),
            sample_grid,
            views: Vec::new(),
            initialized: false,
            decoded,
        });
        self.texture_cache.register_image_mappings(id);
        self.register_image_aliases(id)?;
        self.runtime.statistics.image_allocations += 1;
        self.runtime.statistics.image_layout_plans +=
            self.texture_cache.get(id).contents.geometry.len() as u64;
        self.touch(Resource::Image(id));
        Ok(id)
    }
    /// Refresh CPU contents and aliases before consumption, or discard them
    /// for a full clear. Modification belongs to the image, not its rectangles.
    /// https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/texture_cache/texture_cache.h#L2523-L2553
    pub fn prepare_image(
        &mut self,
        id: ImageId,
        is_modification: bool,
        invalidate: bool,
    ) -> Result<(), VulkanError> {
        self.runtime.constructing.push(Resource::Image(id));
        let result = (|| {
            if invalidate {
                let entry = &mut self.texture_cache.images[id];
                entry.synchronized = true;
                entry.gpu_modified = false;
                self.texture_cache.track_image_cpu(id)?;
            } else {
                if !self.texture_cache.get(id).synchronized {
                    self.texture_cache.track_image_cpu(id)?;
                    self.refresh_image(id)?;
                }
                if !self.texture_cache.get(id).aliases.is_empty() {
                    self.synchronize_image_aliases(id)?;
                }
            }
            if is_modification {
                self.mark_image_modified(id)?;
            }
            self.touch(Resource::Image(id));
            Ok(())
        })();
        self.unpin(Resource::Image(id));
        result
    }
    pub(super) fn is_full_image_clear(
        &self,
        id: ImageId,
        plane: ImagePlane,
        rect: vk::Rect2D,
        aspects: vk::ImageAspectFlags,
    ) -> bool {
        let entry = self.texture_cache.get(id);
        if entry.description.mip_levels != 1
            || entry.description.array_layers != 1
            || entry.description.extent.depth != 1
        {
            return false;
        }
        entry.planes.iter().all(|p| {
            p.mip_level == plane.mip_level
                && p.array_layer == plane.array_layer
                && p.depth_slice == plane.depth_slice
                && aspects.contains(p.aspect)
                && rect == super::transfers::full_rect(*p)
        })
    }
    fn refresh_image(&mut self, id: ImageId) -> Result<(), VulkanError> {
        // Eden RefreshContents uploads the full CPU-modified image. Aliases are
        // synchronized separately by their image modification ticks.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/texture_cache/texture_cache.h#L1115-L1178
        if self.texture_cache.get(id).decoded.is_some() {
            return self.refresh_decoded_image(id);
        }
        let entry = self.texture_cache.get(id);
        if entry.description.samples != vk::SampleCountFlags::TYPE_1 {
            return Err(VulkanError::InvalidResource(
                "CPU-modified multisample image upload requires sample conversion",
            ));
        }
        let geometry = entry.contents.geometry.clone();
        let planes = entry.planes.clone();
        let mut inputs = Vec::new();
        for footprint in &geometry {
            footprint.observed_pages(footprint.mapping(), &mut inputs)?;
        }
        let inputs = nixe_gpu::physical::pin_selected_pages(inputs)?;
        self.upload_cpu_image(id, &geometry, &planes, &inputs)?;
        let entry = &mut self.texture_cache.images[id];
        entry.synchronized = true;
        Ok(())
    }
    /// Deswizzle directly into one aligned mapped reservation, as Eden's
    /// UploadImageContents/UnswizzleImage. Physical inputs remain admitted for
    /// the complete capture; recording receives only native staging references.
    /// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/texture_cache/texture_cache.h
    fn upload_cpu_image(
        &mut self,
        id: ImageId,
        geometry: &[Arc<Footprint>],
        planes: &[ImagePlane],
        inputs: &[nixe_memory::AllocationRange],
    ) -> Result<(), VulkanError> {
        let mut regions = smallvec::SmallVec::<[(u64, u64); 16]>::new();
        let mut packed = 0u64;
        let alignment = planes
            .iter()
            .map(|plane| u64::from(plane.bytes_per_block).max(4))
            .max()
            .unwrap();
        for plane in planes {
            let align = u64::from(plane.bytes_per_block).max(4);
            packed = packed.next_multiple_of(align);
            let bytes = plane.end().expect("validated image plane") - plane.offset;
            regions.push((packed, bytes));
            packed = packed
                .checked_add(bytes)
                .ok_or(VulkanError::InvalidResource("image upload size overflows"))?;
        }
        let slice = self.runtime.device.submission.reserve_slice(
            packed,
            crate::queue::staging::Usage::Upload,
            alignment,
        )?;
        // SAFETY: one fresh reservation owns every written byte and its atom
        // padding. Capture and flush finish before any native copy is recorded.
        unsafe {
            self.runtime
                .device
                .submission
                .staging
                .write_with(&slice, |destination| {
                    destination.fill(0);
                    self.runtime
                        .memory
                        .read_resources(inputs, |reader| {
                            for ((footprint, plane), &(offset, bytes)) in
                                geometry.iter().zip(planes).zip(&regions)
                            {
                                footprint.capture_resource(
                                    reader,
                                    plane.offset,
                                    &mut destination[offset as usize..(offset + bytes) as usize],
                                )?;
                            }
                            Ok(())
                        })
                        .map_err(memory_error)
                })?;
        }
        for (&plane, &(offset, bytes)) in planes.iter().zip(&regions) {
            self.record_cpu_image_plane(id, plane, &slice, offset..offset + bytes)?;
        }
        Ok(())
    }
    fn record_cpu_image_plane(
        &mut self,
        id: ImageId,
        plane: ImagePlane,
        slice: &crate::queue::Slice,
        source: std::ops::Range<u64>,
    ) -> Result<(), VulkanError> {
        let bytes = source.end - source.start;
        debug_assert_eq!(
            bytes,
            plane.end().expect("validated image plane") - plane.offset
        );
        let height = plane.extent.height;
        let (image, layout) = self.transfer_image(id, commands::plane_range(plane))?;
        let region = vk::BufferImageCopy::default()
            .buffer_offset(slice.offset + source.start)
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(plane.aspect)
                    .mip_level(plane.mip_level)
                    .base_array_layer(plane.array_layer)
                    .layer_count(1),
            )
            .image_offset(vk::Offset3D {
                x: 0,
                y: 0,
                z: plane.depth_slice as i32,
            })
            .image_extent(vk::Extent3D {
                width: plane.extent.width,
                height,
                depth: 1,
            });
        self.runtime.barriers.emit(&mut self.runtime.device)?;
        unsafe {
            self.runtime.device.dispatch.copy_buffer_to_image(
                self.runtime.device.submission.staging_buffer(slice).handle,
                image,
                layout,
                &[region],
            );
        }
        self.runtime.statistics.cpu_uploaded_bytes += bytes;
        Ok(())
    }
    pub fn image_view(
        &mut self,
        image: ImageId,
        description: ImageViewDescription,
    ) -> Result<ImageViewId, VulkanError> {
        let entry = self.texture_cache.get(image);
        if let Some(&view) = entry
            .views
            .iter()
            .find(|&&view| self.texture_cache.views[view].description == description)
        {
            return Ok(view);
        }
        let d = description;
        if d.size.width == 0 || d.size.height == 0 || d.size.depth == 0 {
            return Err(VulkanError::InvalidResource(
                "logical image-view size is empty",
            ));
        }
        let same_format = d.format == entry.description.format;
        if !same_format
            && (!format::view_compatible(entry.description.format, d.format)
                || !entry
                    .description
                    .flags
                    .contains(vk::ImageCreateFlags::MUTABLE_FORMAT))
        {
            return Err(VulkanError::InvalidResource(
                "image view format requires a supported mutable-format compatibility class",
            ));
        }
        if d.mip_count == 0
            || d.layer_count == 0
            || d.aspects.is_empty()
            || d.base_mip
                .checked_add(d.mip_count)
                .is_none_or(|end| end > entry.description.mip_levels)
            || d.base_layer.checked_add(d.layer_count).is_none_or(|end| {
                end > if entry.description.image_type == vk::ImageType::TYPE_3D
                    && d.view_type != vk::ImageViewType::TYPE_3D
                {
                    (entry.description.extent.depth >> d.base_mip).max(1)
                } else {
                    entry.description.array_layers
                }
            })
        {
            return Err(VulkanError::InvalidResource(
                "image view exceeds resident subresources",
            ));
        }
        let aspects = entry
            .planes
            .iter()
            .fold(vk::ImageAspectFlags::empty(), |aspects, plane| {
                aspects | plane.aspect
            });
        if !aspects.contains(d.aspects) {
            return Err(VulkanError::InvalidResource(
                "image view aspect has no native plane",
            ));
        }
        for level in d.base_mip..d.base_mip + d.mip_count {
            for layer in d.base_layer..d.base_layer + d.layer_count {
                for aspect in [
                    vk::ImageAspectFlags::COLOR,
                    vk::ImageAspectFlags::DEPTH,
                    vk::ImageAspectFlags::STENCIL,
                ] {
                    if d.aspects.contains(aspect)
                        && !entry.planes.iter().any(|plane| {
                            plane.mip_level == level
                                && (if entry.description.image_type == vk::ImageType::TYPE_3D
                                    && d.view_type != vk::ImageViewType::TYPE_3D
                                {
                                    plane.depth_slice
                                } else {
                                    plane.array_layer
                                }) == layer
                                && plane.aspect == aspect
                        })
                    {
                        return Err(VulkanError::InvalidResource(
                            "image view references an uninitialized native subresource",
                        ));
                    }
                }
            }
        }
        match d.view_type {
            vk::ImageViewType::TYPE_1D
                if entry.description.image_type == vk::ImageType::TYPE_1D && d.layer_count == 1 => {
            }
            vk::ImageViewType::TYPE_1D_ARRAY
                if entry.description.image_type == vk::ImageType::TYPE_1D => {}
            vk::ImageViewType::TYPE_2D if d.layer_count == 1 => {}
            vk::ImageViewType::TYPE_2D_ARRAY => {}
            vk::ImageViewType::TYPE_3D
                if entry.description.image_type == vk::ImageType::TYPE_3D
                    && d.base_layer == 0
                    && d.layer_count == 1 => {}
            vk::ImageViewType::CUBE | vk::ImageViewType::CUBE_ARRAY
                if entry
                    .description
                    .flags
                    .contains(vk::ImageCreateFlags::CUBE_COMPATIBLE)
                    && d.layer_count.is_multiple_of(6)
                    && (d.view_type != vk::ImageViewType::CUBE || d.layer_count == 6) => {}
            _ => {
                return Err(VulkanError::InvalidResource(
                    "image view type is incompatible with resident image/layers",
                ));
            }
        }
        if self
            .runtime
            .device
            .profile()
            .portability
            .as_ref()
            .is_some_and(|p| !p.image_view_format_swizzle)
            && d.swizzle != [vk::ComponentSwizzle::IDENTITY; 4]
        {
            return Err(VulkanError::InvalidResource(
                "component swizzle requires imageViewFormatSwizzle on this portability device",
            ));
        }
        let info = vk::ImageViewCreateInfo::default()
            .image(entry.native.handle)
            .view_type(d.view_type)
            .format(d.format)
            .components(vk::ComponentMapping {
                r: d.swizzle[0],
                g: d.swizzle[1],
                b: d.swizzle[2],
                a: d.swizzle[3],
            })
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(d.aspects)
                    .base_mip_level(d.base_mip)
                    .level_count(d.mip_count)
                    .base_array_layer(d.base_layer)
                    .layer_count(d.layer_count),
            );
        // Alternate views must support the inherited usage on this device.
        // https://docs.vulkan.org/refpages/latest/refpages/source/VkImageViewCreateInfo.html
        let _supported = unsafe {
            self.runtime
                .device
                .owner
                .instance
                .instance
                .get_physical_device_image_format_properties(
                    self.runtime.device.owner.physical,
                    d.format,
                    entry.description.image_type,
                    vk::ImageTiling::OPTIMAL,
                    entry.description.usage,
                    entry.description.flags,
                )
        }
        .map_err(|e| VulkanError::vk("image view format usage", e))?;
        let handle = unsafe {
            self.runtime
                .device
                .owner
                .device
                .create_image_view(&info, None)
        }
        .map_err(|e| VulkanError::vk("vkCreateImageView", e))?;
        let entry = self.texture_cache.get(image);
        let native = Arc::new(crate::AllocatedImageView::from_handle(
            Arc::clone(&entry.native),
            handle,
            info.subresource_range,
        ));
        let downloaded = entry.forced_flushed;
        // Eden records format capabilities once in ImageView, then selects a
        // persistent sampler variant when that view is consumed.
        // https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/renderer_vulkan/vk_texture_cache.cpp
        let supports_minmax_filter = self
            .runtime
            .device
            .format_properties(description.format)
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_MINMAX);
        let view = self.texture_cache.views.insert(ImageView {
            image,
            description,
            native,
            supports_minmax_filter,
            downloaded,
        });
        self.texture_cache.images[image].views.push(view);
        self.runtime.statistics.image_view_allocations += 1;
        Ok(view)
    }
}

fn validate_planes(
    geometry: &[Arc<Footprint>],
    description: NativeImageDescription,
    planes: &[ImagePlane],
    sample_grid: [u32; 2],
) -> Result<(), VulkanError> {
    if planes.is_empty() || geometry.len() != planes.len() {
        return Err(VulkanError::InvalidResource(
            "image residency needs native copy planes",
        ));
    }
    if !matches!(
        description.image_type,
        vk::ImageType::TYPE_1D | vk::ImageType::TYPE_2D | vk::ImageType::TYPE_3D
    ) || sample_grid[0] == 0
        || sample_grid[1] == 0
        || sample_grid[0].checked_mul(sample_grid[1]) != Some(description.samples.as_raw())
        || description.samples.as_raw().count_ones() != 1
        || description.samples != vk::SampleCountFlags::TYPE_1 && description.mip_levels != 1
    {
        return Err(VulkanError::InvalidResource(
            "image residency requires valid dimensions and a frontend sample grid",
        ));
    }
    let mut prior = 0;
    let mut subresources = std::collections::HashSet::new();
    for (plane, footprint) in planes.iter().zip(geometry) {
        let nixe_gpu::physical::WriteDomain::Image(physical) = footprint.domain() else {
            return Err(VulkanError::InvalidResource(
                "native image requires image footprint geometry",
            ));
        };
        if !subresources.insert((
            plane.mip_level,
            plane.array_layer,
            plane.depth_slice,
            plane.aspect.as_raw(),
        )) {
            return Err(VulkanError::InvalidResource(
                "duplicate native image subresource plane",
            ));
        }
        if plane.extent.width == 0
            || plane.extent.height == 0
            || plane.extent.depth != 1
            || plane.block_extent.width == 0
            || plane.block_extent.height == 0
            || plane.bytes_per_block == 0
            || plane.aspect.as_raw().count_ones() != 1
            || plane.mip_level >= description.mip_levels
            || plane.array_layer >= description.array_layers
            || plane.depth_slice >= (description.extent.depth >> plane.mip_level).max(1)
            || (description.image_type != vk::ImageType::TYPE_3D && plane.depth_slice != 0)
        {
            return Err(VulkanError::InvalidResource(
                "invalid native image copy plane",
            ));
        }
        if format::copy_block(description.format, plane.aspect)
            != Some((plane.block_extent, plane.bytes_per_block))
        {
            return Err(VulkanError::InvalidResource(
                "native copy plane geometry does not match format/aspect; packed or unsupported encodings require conversion recording",
            ));
        }
        let width = description
            .extent
            .width
            .checked_shr(plane.mip_level)
            .unwrap_or(0)
            .max(1);
        let height = description
            .extent
            .height
            .checked_shr(plane.mip_level)
            .unwrap_or(0)
            .max(1);
        if plane.extent.width != width
            || plane.extent.height != height
            || plane.mip_level >= u32::BITS
        {
            return Err(VulkanError::InvalidResource(
                "image copy plane extent does not match native mip extent",
            ));
        }
        let storage = ImagePlane {
            extent: vk::Extent3D {
                width: width
                    .checked_mul(sample_grid[0])
                    .ok_or(VulkanError::InvalidResource("sample width overflow"))?,
                height: height
                    .checked_mul(sample_grid[1])
                    .ok_or(VulkanError::InvalidResource("sample height overflow"))?,
                depth: 1,
            },
            ..*plane
        };
        if plane.row_pitch < storage.row_bytes()
            || plane.offset < prior
            || storage.end().is_none()
            || physical.depth_slice != plane.depth_slice
            || physical.native_base != plane.offset
            || physical.native_row_pitch != plane.row_pitch
            || physical.native_element_bytes != plane.bytes_per_block
            || physical.columns != storage.extent.width.div_ceil(plane.block_extent.width)
            || physical.rows != storage.extent.height.div_ceil(plane.block_extent.height)
            || physical.rect.x != 0
            || physical.rect.y != 0
            || physical.rect.width != physical.columns
            || physical.rect.height != physical.rows
        {
            return Err(VulkanError::InvalidResource(
                "overlapping or out-of-bounds image sample planes",
            ));
        }
        prior = storage.end().expect("checked sample extent");
    }
    Ok(())
}

impl Residency<'_> {
    pub fn image_view_lease(
        &self,
        id: ImageViewId,
    ) -> Result<Arc<crate::AllocatedImageView>, VulkanError> {
        Ok(Arc::clone(&self.texture_cache.views[id].native))
    }
}
