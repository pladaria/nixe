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
            swizzle: [2, 3, 4, 5],
            levels: 1,
            ..Default::default()
        },
    )?;
    c.write(c.args[0] + 8 + size_of::<Texture>() as u64, &[0u64; 2])?;
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
    dimensions(c, c.args[2] as u32, 1)
}

pub fn size_1d(c: &Call) -> Result<u64> {
    dimensions(c, 1, 1)
}

pub fn size_3d(c: &Call) -> Result<u64> {
    dimensions(c, c.args[2] as u32, c.args[3] as u32)
}

fn dimensions(c: &Call, height: u32, depth: u32) -> Result<u64> {
    c.write(
        c.args[0] + 8 + core::mem::offset_of!(Texture, width) as u64,
        &[c.args[1] as u32, height],
    )?;
    c.write(
        c.args[0] + 8 + core::mem::offset_of!(Texture, depth) as u64,
        &depth,
    )?;
    Ok(0)
}
pub fn storage(c: &Call) -> Result<u64> {
    c.write(
        c.args[0] + 8 + size_of::<Texture>() as u64,
        &[c.args[2], c.args[1]],
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
    initialize_texture::<false>(c)
}

pub fn initialize_ampere(c: &Call) -> Result<u64> {
    initialize_texture::<true>(c)
}

fn initialize_texture<const AMPERE: bool>(c: &Call) -> Result<u64> {
    let builder = c.args[1];
    let [offset, pool]: [u64; 2] = c.read(builder + 8 + size_of::<Texture>() as u64)?;
    if pool == 0 {
        return Ok(0);
    }
    let p: Pool = c.read(pool)?;
    if p.flags & 0x100 != 0 {
        return Ok(0);
    }
    let mut t: Texture = c.read(builder + 8)?;
    let device: u64 = c.read(builder)?;
    // The device can request the X1 order for its textures. The default NVN2
    // order exchanges address bits 5 and 6 within each 512-byte GOB.
    // https://github.com/seedds/nxnative/blob/a366365/docs/reference/nvn-objects.md#the-switch-2s-order-inside-a-gob
    if AMPERE && c.read::<u32>(device)? & 0x10000 == 0 {
        t.use_ampere_gob_order();
    }
    t.validate_layout()?;
    t.address = p
        .gpu
        .checked_add(offset)
        .ok_or("texture address overflows")?;
    c.write(c.args[0], &t)?;
    c.write(
        c.args[0] + size_of::<Texture>() as u64,
        &[p.cpu + offset, offset, pool, device],
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
fn zcull_size(t: Texture) -> u64 {
    if !(matches!(t.format, 0x32..=0x36) || t.format == 0x31 && t.flags & 0x400 != 0) {
        return 0;
    }
    // Maxwell has one 32-byte region entry per 32x32 pixels and layer,
    // plus the region and subregion headers advertised by nvhost-ctrl-gpu.
    // @SIMPLIFICATION Use ordinary ZCull storage for adaptive targets. Host
    // hierarchical depth stays on the device and never spills adaptive data here.
    let tiles = u64::from(t.width.div_ceil(32)) * u64::from(t.height.div_ceil(32));
    0x20 * tiles * u64::from(t.depth.max(1)) + 0x20 + 0xc0
}
pub fn builder_zcull(c: &Call) -> Result<u64> {
    Ok(zcull_size(c.read(c.args[0] + 8)?))
}
pub fn zcull(c: &Call) -> Result<u64> {
    Ok(zcull_size(c.read(c.args[0])?))
}
pub fn equal(c: &Call) -> Result<u64> {
    let metadata = size_of::<Texture>() as u64 + 8;
    // Equal addresses can come from different pools. Object identity includes the
    // offset, pool and device in addition to the texture description and swizzle.
    Ok(
        (c.read::<[u64; 3]>(c.args[0] + metadata)? == c.read::<[u64; 3]>(c.args[1] + metadata)?
            && c.read::<Texture>(c.args[0])? == c.read::<Texture>(c.args[1])?) as u64,
    )
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

pub fn get_depth(c: &Call) -> Result<u64> {
    Ok(c.read::<Texture>(c.args[0])?.depth.into())
}

pub fn get_levels(c: &Call) -> Result<u64> {
    Ok(c.read::<Texture>(c.args[0])?.levels.into())
}

pub fn get_samples(c: &Call) -> Result<u64> {
    Ok(c.read::<Texture>(c.args[0])?.samples.into())
}

pub fn get_storage_size(c: &Call) -> Result<u64> {
    Ok(c.read::<Texture>(c.args[0])?.layout()?.bytes)
}

pub fn get_view_offset(c: &Call) -> Result<u64> {
    if c.args[1] == 0 {
        return Ok(0);
    }
    let t: Texture = c.read(c.args[0])?;
    let v: View = c.read(c.args[1])?;
    if v.first == 0 && v.layer == 0 {
        return Ok(0);
    }
    let layout = t.layout()?;
    let offset = layout
        .mips
        .get(v.first as usize)
        .ok_or("invalid view mip")?
        .offset;
    if t.target == 2 && v.layer != 0 {
        return Err("storage offset of a 3D slice is unsupported");
    }
    Ok(offset + u64::from(v.layer) * layout.layer_stride)
}
