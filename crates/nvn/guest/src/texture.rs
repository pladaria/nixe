use super::*;

pub fn set_device(c: &Call) -> Result<u64> {
    c.write(c.args[0], &c.args[1])?;
    Ok(0)
}
pub fn defaults(c: &Call) -> Result<u64> {
    c.write(
        c.args[0] + 8,
        &Texture {
            target: 1,
            depth: 1,
            width: 1,
            height: 1,
            swizzle: [2, 3, 4, 5],
            levels: 1,
            ..Default::default()
        },
    )?;
    Ok(0)
}
macro_rules! setter {
    ($name:ident,$field:ident,$ty:ty) => {
        pub fn $name(c: &Call) -> Result<u64> {
            c.write(
                c.args[0] + 8 + core::mem::offset_of!(Texture, $field) as u64,
                &(c.args[1] as $ty),
            )?;
            Ok(0)
        }
    };
}
setter!(flags, flags, u32);
setter!(target, target, u8);
setter!(levels, levels, u8);
setter!(format, format, u16);
setter!(samples, samples, u32);
setter!(stride, stride, u32);
pub fn size(c: &Call) -> Result<u64> {
    c.write(
        c.args[0] + 8 + core::mem::offset_of!(Texture, width) as u64,
        &[c.args[1] as u32, c.args[2] as u32],
    )?;
    Ok(0)
}
pub fn storage(c: &Call) -> Result<u64> {
    let p: Pool = c.read(c.args[1])?;
    c.write(
        c.args[0] + 8 + size_of::<Texture>() as u64,
        &[p.cpu + c.args[2], c.args[2], c.args[1]],
    )?;
    c.write(
        c.args[0] + 8,
        &p.gpu
            .checked_add(c.args[2])
            .ok_or("texture address overflows")?,
    )?;
    Ok(0)
}
pub fn alignment(c: &Call) -> Result<u64> {
    Ok(u64::from(
        c.read::<Texture>(c.args[0] + 8)?.layout()?.alignment,
    ))
}
pub fn storage_size(c: &Call) -> Result<u64> {
    Ok(c.read::<Texture>(c.args[0] + 8)?.layout()?.bytes)
}
pub fn initialize(c: &Call) -> Result<u64> {
    let t: Texture = c.read(c.args[1] + 8)?;
    t.layout()?;
    c.write(c.args[0], &t)?;
    c.write(
        c.args[0] + size_of::<Texture>() as u64,
        &c.read::<[u64; 3]>(c.args[1] + 8 + size_of::<Texture>() as u64)?,
    )?;
    Ok(1)
}

setter!(depth, depth, u32);
setter!(set_width, width, u32);
setter!(set_height, height, u32);
setter!(stencil, stencil, u32);
pub fn get_flags(c: &Call) -> Result<u64> {
    Ok(c.read::<Texture>(c.args[0])?.flags.into())
}
pub fn packaged(c: &Call) -> Result<u64> {
    let value = if c.args[1] == 0 {
        0
    } else {
        u32::from(c.read::<u16>(c.args[1])?) + 1
    };
    c.write(
        c.args[0] + 8 + core::mem::offset_of!(Texture, packaged) as u64,
        &value,
    )?;
    Ok(0)
}
pub fn swizzle(c: &Call) -> Result<u64> {
    c.write(
        c.args[0] + 8 + core::mem::offset_of!(Texture, swizzle) as u64,
        &[
            c.args[1] as u8,
            c.args[2] as u8,
            c.args[3] as u8,
            c.args[4] as u8,
        ],
    )?;
    Ok(0)
}
pub fn storage_class(_: &Call) -> Result<u64> {
    // @SIMPLIFICATION The SMMU maps the same physical pages for every storage class;
    // resource layout and depth compression are selected separately by Vulkan.
    Ok(0)
}
// @SIMPLIFICATION Host rendering owns its hierarchical depth state. Return a nonzero,
// tile-sized save area for guest allocators without storing hardware ZCull metadata.
pub fn zcull(c: &Call) -> Result<u64> {
    let t: Texture = c.read(c.args[0] + 8)?;
    Ok((u64::from(t.width.div_ceil(32))
        * u64::from(t.height.div_ceil(32))
        * u64::from(t.depth.max(1))
        * 8)
    .next_multiple_of(256)
    .max(256))
}
pub fn width(c: &Call) -> Result<u64> {
    Ok(c.read::<Texture>(c.args[0])?.width.into())
}
pub fn height(c: &Call) -> Result<u64> {
    Ok(c.read::<Texture>(c.args[0])?.height.into())
}
pub fn get_format(c: &Call) -> Result<u64> {
    Ok(c.read::<Texture>(c.args[0])?.format.into())
}
pub fn get_target(c: &Call) -> Result<u64> {
    Ok(c.read::<Texture>(c.args[0])?.target.into())
}
pub fn finalize(c: &Call) -> Result<u64> {
    c.write(c.args[0], &Texture::default())?;
    Ok(0)
}

pub fn get_depth(c: &Call) -> Result<u64> {
    Ok(c.read::<Texture>(c.args[0])?.depth.into())
}
