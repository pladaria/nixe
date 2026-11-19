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
            if t.block()? != orig {
                return Err("incompatible texture view format");
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
