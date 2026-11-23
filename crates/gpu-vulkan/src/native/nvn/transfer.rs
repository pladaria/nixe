//! NVN image commands capture texture views and regions at recording time.
//! https://github.com/seedds/nxnative/blob/a366365/dm/gfx/nvn/hle_nvn.c#L1709-L1721
use super::*;
use nixe_gpu::ImageFormat;

fn slice(mut t: Texture, z: u32) -> Result<Texture, VulkanError> {
    if z >= t.layer_count() {
        return Err(error("NVN image region exceeds selected layers"));
    }
    t.first_layer = (u32::from(t.first_layer) + z).try_into().map_err(error)?;
    t.view_layers = 1;
    t.view_levels = 1;
    Ok(t)
}
fn rectangle([x, y, _, width, height, _]: [u32; 6]) -> nixe_gpu::ScissorRect {
    nixe_gpu::ScissorRect {
        x,
        y,
        width,
        height,
    }
}
impl NativeExecution<'_> {
    pub(super) fn copy_nvn_buffer_image(
        &mut self,
        buffer: u64,
        texture: Texture,
        mut region: [u32; 6],
        strides: nixe_nvn::CopyStrides,
        flags: u32,
        upload: bool,
    ) -> Result<(), VulkanError> {
        if flags & !(if upload { 3 } else { 0x1f }) != 0 {
            return Err(error(format!(
                "unsupported NVN buffer/image copy flags {flags:#x}"
            )));
        }
        if texture.target == 3 {
            // NVN locates layers of a 1D array on the Y axis of the copy region.
            region = [region[0], 0, region[1], region[3], 1, region[4]];
        }
        if region[3..].contains(&0) {
            return Ok(());
        }
        if region[2]
            .checked_add(region[5])
            .is_none_or(|end| end > texture.layer_count())
        {
            return Err(error("NVN buffer/image copy exceeds selected layers"));
        }
        if texture.target == 2 {
            region[2] += u32::from(texture.first_layer);
        }
        let info = resources::info(texture, false)?;
        let (id, base) = self.find_or_insert_image(&info, &resources::PoolSpace)?;
        if upload {
            self.residency
                .copy_buffer_image_region::<true>(id, base, buffer, region, strides, [false; 2])
        } else {
            // The SDK reverses the destination pitch for Y and Z in this command.
            self.residency.copy_buffer_image_region::<false>(
                id,
                base,
                buffer,
                region,
                strides,
                [flags & 8 != 0, flags & 16 != 0],
            )
        }
    }
    pub(super) fn clear_nvn_image(
        &mut self,
        t: Texture,
        region: [u32; 6],
        color: [f32; 4],
        mask: u32,
    ) -> Result<(), VulkanError> {
        if mask == 0 || region[3..].contains(&0) {
            return Ok(());
        }
        if mask != 15 {
            return Err(error(format!(
                "unsupported NVN texture clear mask {mask:#x}"
            )));
        }
        let rect = rectangle(region);
        let end = region[2]
            .checked_add(region[5])
            .ok_or_else(|| error("NVN clear layer range overflow"))?;
        for z in region[2]..end {
            let view = resources::target_view(self, slice(t, z)?)?;
            self.clear_native(view, rect, nixe_gpu::ClearValue::Color(color))?;
        }
        Ok(())
    }
    pub(super) fn copy_nvn_images(
        &mut self,
        textures: [Texture; 2],
        regions: [[u32; 6]; 2],
        flags: u32,
    ) -> Result<(), VulkanError> {
        if flags & !0x1f != 0 {
            return Err(error(format!(
                "unsupported NVN texture copy flags {flags:#x}"
            )));
        }
        let [src, dst] = regions;
        // Copy flags reverse the source region independently on each axis.
        // https://github.com/seedds/nxnative/blob/a366365/dm/gfx/nvn/nvn_mtl.m#L16581-L16583
        let mirror = [flags & 4 != 0, flags & 8 != 0];
        if src[5] != dst[5] {
            return Err(error("NVN texture copy requires equal layer counts"));
        }
        if src[3..].contains(&0) || dst[3..].contains(&0) {
            return Ok(());
        }
        let source_format = resources::format(textures[0].format)?;
        let core = &self.residency.device().profile().core;
        let encoded_copy = source_format.block_extent() != [1, 1]
            && if source_format.astc_profile().is_some() {
                core.texture_compression_astc_ldr == 0
            } else {
                core.texture_compression_bc == 0
            };
        if encoded_copy
            && (textures[0].format != textures[1].format
                || src[3..5] != dst[3..5]
                || mirror != [false; 2])
        {
            return Err(error(
                "encoded NVN copy requires matching formats and block regions",
            ));
        }
        for z in 0..src[5] {
            let get = |i: usize| {
                let slice = if i == 0 && flags & 16 != 0 {
                    src[5] - 1 - z
                } else {
                    z
                };
                regions[i][2]
                    .checked_add(slice)
                    .ok_or_else(|| error("NVN copy layer range overflow"))
            };
            if encoded_copy {
                let source = encoded_plane(slice(textures[0], get(0)?)?, src, source_format)?;
                let destination = encoded_plane(slice(textures[1], get(1)?)?, dst, source_format)?;
                self.residency.copy_encoded_planes([source, destination])?;
                continue;
            }
            let retained = self.residency.runtime.constructing.len();
            let result = (|| {
                let mut views = [ImageViewId::default(); 2];
                for (i, view) in views.iter_mut().enumerate() {
                    *view = resources::target_view(self, slice(textures[i], get(i)?)?)?;
                    let image = self.residency.texture_cache.views[*view].image;
                    self.residency
                        .runtime
                        .constructing
                        .push(crate::residency::Resource::Image(image));
                }
                if mirror == [false; 2]
                    && textures[0].format == textures[1].format
                    && textures[0].samples.max(1) == textures[1].samples.max(1)
                    && src[3..5] == dst[3..5]
                {
                    let aspects = self.residency.image_view_description(views[0])?.aspects;
                    let mut planes = smallvec::SmallVec::<[(usize, usize); 2]>::new();
                    for aspect in [
                        vk::ImageAspectFlags::COLOR,
                        vk::ImageAspectFlags::DEPTH,
                        vk::ImageAspectFlags::STENCIL,
                    ] {
                        if aspects.contains(aspect) {
                            planes.push((
                                self.residency.image_view_plane(views[0], aspect)?,
                                self.residency.image_view_plane(views[1], aspect)?,
                            ));
                        }
                    }
                    let [from, to] = views.map(|v| self.residency.texture_cache.views[v].image);
                    self.residency.copy_image_regions(
                        from,
                        to,
                        &planes,
                        images::rect(rectangle(src)),
                        images::rect(rectangle(dst)).offset,
                    )?;
                    return Ok(());
                }
                self.blit_native(
                    views[0],
                    views[1],
                    rectangle(src),
                    rectangle(dst),
                    flags & 1 != 0,
                    textures[0].samples > 1 && textures[1].samples <= 1,
                    None,
                    mirror,
                )
            })();
            self.residency.runtime.constructing.truncate(retained);
            result?;
        }
        Ok(())
    }
}

fn encoded_plane(
    texture: Texture,
    [x, y, _, w, h, _]: [u32; 6],
    format: ImageFormat,
) -> Result<(u64, u64, nixe_gpu::physical::ImageFootprint), VulkanError> {
    use nixe_gpu::physical::ByteRect;
    let [bw, bh] = format.block_extent();
    let [width, height] = texture.view_extent();
    let valid = |start: u32, count: u32, end: u32, block: u32| {
        start.is_multiple_of(block)
            && start
                .checked_add(count)
                .is_some_and(|stop| stop <= end && (count.is_multiple_of(block) || stop == end))
    };
    if !valid(x, w, width, bw) || !valid(y, h, height, bh) {
        return Err(error("encoded NVN copy rectangle is not block aligned"));
    }
    let info = resources::info(texture, false)?;
    let bytes = u32::from(format.plane_bytes_per_block(0).unwrap());
    let mut plane = crate::image_layout::plane_layout(
        info.layout,
        [
            0,
            if texture.target == 2 {
                u32::from(texture.first_layer)
            } else {
                0
            },
        ],
        [width.div_ceil(bw), height.div_ceil(bh)],
        bytes,
        bytes,
        (0, bytes),
    )?;
    plane.rect = ByteRect {
        x: x / bw,
        y: y / bh,
        width: w.div_ceil(bw),
        height: h.div_ceil(bh),
    };
    Ok((info.address, info.size, plane))
}
