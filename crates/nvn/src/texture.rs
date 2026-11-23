use super::*;
pub fn register_texture(c: &Call) -> Result<u64> {
    let t = texels::view(c, c.args[2], c.args[3])?;
    let id = c.session.descriptors.textures.lock().unwrap().intern(t);
    c.write(pool::slot(c, c.args[0], c.args[1] as u32)?, &id)?;
    Ok(0)
}
/// NvRmSurface (88 bytes): dimensions, color format, layout, pitch, NvMap
/// handle and offset, block-linear kind/height and plane size.
pub fn nv_surface(c: &Call) -> Result<u64> {
    let t: Texture = c.read(c.args[0])?;
    let [_, offset, pool]: [u64; 3] = c.read(c.args[0] + size_of::<Texture>() as u64)?;
    let l = t.layout()?;
    // NVN 53 exports the original NvColorFormat codes. The client library
    // uses the numeric code to select its VIC input/output surface format.
    let legacy = c.session.sdk_api_major() == Some(53);
    let color: u64 = match (legacy, t.format) {
        (true, 1) => 0x000e09200408,
        (true, 0x0d) => 0x000e08440710,
        (true, 0x25 | 0x38) => 0x000100532120,
        (true, 0x3c) => 0x00010a881210,
        (true, 0x9a | 0x9c) => 0x000108332120,
        (true, 0x6d) => 0x00010a532120,
        (false, 1) => 0x000f12400408,
        (false, 0x0d) => 0x000f10880710,
        (false, 0x25 | 0x38) => 0x000200a62120,
        (false, 0x3c) => 0x000215101210,
        (false, 0x9a | 0x9c) => 0x000201a22120,
        (false, 0x6d) => 0x000214a62120,
        _ => 0,
    };
    let linear = t.flags & 0x50 != 0;
    let mut words = [0u32; 22];
    words[0] = t.width;
    words[1] = t.height;
    words[2] = color as u32;
    words[3] = (color >> 32) as u32;
    words[4] = if linear { 1 } else { 3 };
    words[5] = l.pitch;
    let handle = pool::surface_handle(c, pool)?;
    words[6] = handle;
    words[7] = u32::try_from(offset).map_err(|_| "NvRmSurface offset overflow")?;
    if !linear {
        words[8] = 0xfe;
        words[9] = l.block_height.into()
    }
    words[14] = l.bytes as u32;
    words[15] = (l.bytes >> 32) as u32;
    c.write(c.args[2], &words)?;
    Ok(u64::from(handle))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_linear_texture_respects_application_pitch() {
        let t = Texture {
            width: 424,
            height: 240,
            levels: 1,
            target: 1,
            flags: 0x10,
            stride: 1024,
            format: 0x95,
            ..Default::default()
        };
        let l = t.layout().unwrap();
        assert_eq!((l.pitch, l.bytes), (1024, 1024 * 240));
        assert!(Texture { stride: 640, ..t }.layout().is_err());
    }

    #[test]
    fn volume_view_selects_slices_without_shifting_the_mip_storage() {
        let volume = Texture {
            width: 8,
            height: 8,
            depth: 8,
            target: 2,
            levels: 2,
            format: 0x25,
            ..Default::default()
        };
        let view = Texture {
            first_layer: 1,
            first_level: 1,
            ..volume
        };
        assert_eq!(view.layers(), 1);
        assert_eq!(view.viewable_layers(1), 4);
        assert_eq!(view.layer_count(), 3);
        assert_eq!(
            view.layout().unwrap().mips[1].offset,
            volume.layout().unwrap().mips[1].offset
        );
    }

    #[test]
    fn tiled_rgba_framebuffer_has_complete_gob_rows() {
        let t = Texture {
            width: 1920,
            height: 1080,
            levels: 1,
            target: 1,
            format: 0x25,
            ..Default::default()
        };
        let l = t.layout().unwrap();
        assert_eq!((l.block_height, l.pitch, l.bytes), (4, 7680, 0x870000));
    }
}
