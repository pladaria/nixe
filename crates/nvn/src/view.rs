use super::*;
pub(crate) fn apply(
    mut t: Texture,
    view: Option<View>,
) -> core::result::Result<Texture, &'static str> {
    if let Some(v) = view {
        if v.first >= t.levels
            || v.levels > t.levels - v.first
            || u32::from(v.layer) >= t.viewable_layers(v.first)
            || u32::from(v.layers) > t.viewable_layers(v.first) - u32::from(v.layer)
        {
            return Err("texture view exceeds mip or layer range");
        }
        if v.format != 0 {
            let orig = t.block()?;
            t.format = v.format;
            let view = t.block()?;
            if orig != view {
                // Raw block views address one uncompressed texel per compressed block.
                // Keep the original mip and layer storage geometry when reinterpreting it.
                // https://github.com/seedds/nxnative/blob/a366365/docs/reference/rendering.md
                if orig.0 != view.0 || (view.1, view.2) != (1, 1) {
                    return Err("incompatible texture view format");
                }
                t.block_view = ((orig.1 - 1) | ((orig.2 - 1) << 4)) as u8;
            }
        }
        if v.swizzle[0] != 255 {
            t.swizzle = v.swizzle
        }
        if v.mode != 255 {
            t.stencil = v.mode.into()
        }
        t.first_level = v.first;
        t.view_levels = v.levels;
        t.first_layer = v.layer;
        t.view_layers = v.layers;
        if v.target != 255 {
            t.view_target = v.target + 1
        }
    }
    Ok(t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_block_view_preserves_storage_at_nonzero_mip_and_layer() {
        let texture = Texture {
            width: 37,
            height: 19,
            depth: 3,
            levels: 4,
            target: 4,
            format: 0x49, // BC4, eight bytes per 4x4 block
            ..Default::default()
        };
        let original = texture.layout().unwrap();
        let view = apply(
            texture,
            Some(View {
                format: 0x2c, // RGBA16UI, eight bytes per texel
                first: 1,
                levels: 1,
                layer: 2,
                layers: 1,
                ..Default::default()
            }),
        )
        .unwrap();
        let layout = view.layout().unwrap();
        assert_eq!(view.view_extent(), [5, 3]);
        assert_eq!(layout.layer_stride, original.layer_stride);
        assert_eq!(layout.mips[1].offset, original.mips[1].offset);
        assert_eq!(layout.mips[1].pitch, original.mips[1].pitch);
        assert_eq!(view.layer_count(), 1);
        assert!(
            apply(
                texture,
                Some(View {
                    format: 0x25,
                    ..Default::default()
                })
            )
            .is_err()
        );
    }
}
