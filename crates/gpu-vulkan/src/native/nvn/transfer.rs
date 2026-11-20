//! NVN image commands capture texture views and regions at recording time.
//! https://github.com/seedds/nxnative/blob/a366365/dm/gfx/nvn/hle_nvn.c#L1709-L1721
use super::*;

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
