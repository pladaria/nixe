//! Source-preserving 2D surfaces and the pixels-from-memory launch boundary.
use super::{
    CLASS, MaxwellTwoDClipEnable, MaxwellTwoDColorKeyEnable, MaxwellTwoDOperation,
    MaxwellTwoDRegister, MaxwellTwoDRenderEnableMode, MaxwellTwoDState,
};
use crate::engines::{
    AppliedMethod, MaxwellEngineDispatchError, MaxwellEngineMethodMetadata, PendingEngineOperation,
};
use crate::engines::{MaxwellMemoryCopyLayout, MaxwellMemoryCopyOperation};
use crate::{MaxwellMethodDispatch, MaxwellMethodSource};
use nixe_gpu::GpuMethodId;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MaxwellTwoDBlitState {
    src: [MaxwellTwoDRegister<u32>; 10],
    pub(super) dst: [MaxwellTwoDRegister<u32>; 10],
    parameters: [MaxwellTwoDRegister<u32>; 12],
    sample_mode: MaxwellTwoDRegister<u32>,
    pub(super) compression: MaxwellTwoDRegister<u32>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct BlitSurface {
    pub address: u64,
    pub width: u32,
    pub height: u32,
    pub format: u8,
    pub layout: BlitSurfaceLayout,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum BlitSurfaceLayout {
    Pitch(u32),
    BlockLinear(u8),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BlitKind {
    Resolve,
    Copy {
        origins: [nixe_gpu::ImageOrigin; 2],
        extent: nixe_gpu::ImageExtent,
    },
}

/// A 2:1 bilinear 2D operation eligible for a four-sample resolve. Memory-kind
/// and resident-source validation must still establish that it really is MSAA.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellTwoDBlitOperation {
    pub(crate) source: MaxwellMethodSource,
    pub(crate) images: [BlitSurface; 2],
    pub(crate) destination_compression: bool,
    pub(crate) kind: BlitKind,
}

impl MaxwellTwoDBlitOperation {
    pub const fn source(&self) -> MaxwellMethodSource {
        self.source
    }
}

fn error(source: MaxwellMethodSource, reason: &'static str) -> MaxwellEngineDispatchError {
    MaxwellEngineDispatchError::InvalidTwoDMethodEncoding {
        source,
        method_name: "PIXELS_FROM_MEMORY_SRC_Y0_INT",
        reason,
    }
}

impl MaxwellTwoDBlitState {
    fn launch(
        &self,
        state: &MaxwellTwoDState,
        source: MaxwellMethodSource,
    ) -> Result<PendingEngineOperation, MaxwellEngineDispatchError> {
        if state.operation().value() != Some(&MaxwellTwoDOperation::SourceCopy) {
            return Err(error(
                source,
                "2D pixels-from-memory requires SET_OPERATION=SRCCOPY",
            ));
        }
        if state.clip_enable().value() != Some(&MaxwellTwoDClipEnable::Disabled) {
            return Err(error(
                source,
                "2D pixels-from-memory requires SET_CLIP_ENABLE=FALSE",
            ));
        }
        if state.color_key_enable().value() != Some(&MaxwellTwoDColorKeyEnable::Disabled) {
            return Err(error(
                source,
                "2D pixels-from-memory requires SET_COLOR_KEY_ENABLE=FALSE",
            ));
        }
        if state.render_enable().mode().value() != Some(&MaxwellTwoDRenderEnableMode::Enabled) {
            return Err(error(
                source,
                "2D pixels-from-memory requires SET_RENDER_ENABLE_C=TRUE",
            ));
        }
        if self.sample_mode.value() == Some(&0) {
            let copy = self.copy(source)?;
            if self.compression.value() != Some(&1) && self.src[1].value() == Some(&1) {
                return Ok(PendingEngineOperation::MemoryCopy(copy));
            }
            // Compression is image state, not a byte-copy optimization. Preserve
            // the destination's resident representation across partial uploads.
            // https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/twod/cl902d.h#L578-L581
            let surface = |r: &[MaxwellTwoDRegister<u32>; 10], address, layout| {
                let (width, height, origin, layout) = match layout {
                    MaxwellMemoryCopyLayout::Pitch { pitch } => (
                        copy.width / 4,
                        copy.height,
                        nixe_gpu::ImageOrigin { x: 0, y: 0, z: 0 },
                        BlitSurfaceLayout::Pitch(pitch),
                    ),
                    MaxwellMemoryCopyLayout::BlockLinear {
                        surface_width,
                        surface_height,
                        x,
                        y,
                        block_height_log2,
                    } => (
                        surface_width / 4,
                        surface_height,
                        nixe_gpu::ImageOrigin { x: x / 4, y, z: 0 },
                        BlitSurfaceLayout::BlockLinear(block_height_log2),
                    ),
                };
                (
                    BlitSurface {
                        address,
                        width,
                        height,
                        format: *r[0].value().expect("copy validated the surface format") as u8,
                        layout,
                    },
                    origin,
                )
            };
            let (src, src_origin) = surface(&self.src, copy.source_address, copy.source_layout);
            let (dst, dst_origin) =
                surface(&self.dst, copy.destination_address, copy.destination_layout);
            return Ok(PendingEngineOperation::TwoDBlit(MaxwellTwoDBlitOperation {
                source,
                images: [src, dst],
                destination_compression: self.compression.value() == Some(&1),
                kind: BlitKind::Copy {
                    origins: [src_origin, dst_origin],
                    extent: nixe_gpu::ImageExtent {
                        width: copy.width / 4,
                        height: copy.height,
                        depth: 1,
                    },
                },
            }));
        }
        let get = |register: &MaxwellTwoDRegister<u32>| {
            register
                .value()
                .copied()
                .ok_or_else(|| error(source, "incomplete 2D surface or pixels-from-memory state"))
        };
        let surface = |r: &[MaxwellTwoDRegister<u32>; 10],
                       dst: bool|
         -> Result<BlitSurface, MaxwellEngineDispatchError> {
            let format = get(&r[0])?;
            let block = get(&r[2])?;
            if !matches!(format, 0xcf | 0xd0 | 0xd5 | 0xd6)
                || get(&r[1])? != 0
                || block & !0x70 != 0
                || ((block >> 4) & 7) > 5
                || get(&r[3])? != 1
                || (dst && get(&r[4])? != 0)
            {
                return Err(error(
                    source,
                    "2D resolve requires RGBA8/BGRA8 UNORM/sRGB block-linear 2D single-layer surfaces",
                ));
            }
            Ok(BlitSurface {
                address: (u64::from(get(&r[8])?) << 32) | u64::from(get(&r[9])?),
                width: get(&r[6])?,
                height: get(&r[7])?,
                format: format as u8,
                layout: BlitSurfaceLayout::BlockLinear((block >> 4) as u8),
            })
        };
        let images = [surface(&self.src, false)?, surface(&self.dst, true)?];
        let [src, dst] = images;
        if src.format != dst.format
            || dst.width == 0
            || dst.height == 0
            || dst.width.checked_mul(2) != Some(src.width)
            || dst.height.checked_mul(2) != Some(src.height)
        {
            return Err(error(
                source,
                "2D resolve requires matching formats and 2x2 sample dimensions",
            ));
        }
        // deko3d averages each 2x2 sample block at its center. Other affine
        // transforms are blits, not hardware resolves, and must not be coerced.
        // https://github.com/devkitPro/deko3d/blob/master/source/dk_image.cpp#L666-L695
        if get(&self.sample_mode)? != 0x10 {
            return Err(error(
                source,
                "2D resolve requires center-origin bilinear filtering",
            ));
        }
        for (register, expected) in self.parameters.iter().zip([
            0,
            0,
            dst.width,
            dst.height,
            0,
            2,
            0,
            2,
            0x8000_0000,
            0,
            0x8000_0000,
            0,
        ]) {
            if get(register)? != expected {
                return Err(error(
                    source,
                    "2D blit is not a full, equally weighted four-sample resolve",
                ));
            }
        }
        Ok(PendingEngineOperation::TwoDBlit(MaxwellTwoDBlitOperation {
            source,
            images,
            destination_compression: get(&self.compression)? != 0,
            kind: BlitKind::Resolve,
        }))
    }
    // Integral center-origin point sampling with unit scaling is a byte copy.
    // Pitch surfaces do not consume block size/depth/layer registers.
    // https://github.com/NVIDIA/open-gpu-doc/blob/master/classes/twod/cl902d.h#L939-L978
    fn copy(
        &self,
        source: MaxwellMethodSource,
    ) -> Result<MaxwellMemoryCopyOperation, MaxwellEngineDispatchError> {
        let get = |r: &MaxwellTwoDRegister<u32>| {
            r.value()
                .copied()
                .ok_or_else(|| error(source, "incomplete 2D copy state"))
        };
        if get(&self.src[0])? != get(&self.dst[0])?
            || !matches!(get(&self.src[0])?, 0xcf | 0xd0 | 0xd5 | 0xd6)
        {
            return Err(error(
                source,
                "2D copy requires matching RGBA8/BGRA8 formats",
            ));
        }
        if [
            get(&self.parameters[4])?,
            get(&self.parameters[5])?,
            get(&self.parameters[6])?,
            get(&self.parameters[7])?,
            get(&self.parameters[8])?,
            get(&self.parameters[10])?,
        ] != [0, 1, 0, 1, 0, 0]
        {
            return Err(error(
                source,
                "2D point copy requires integral coordinates and unit scaling",
            ));
        }
        let width = get(&self.parameters[2])?;
        let height = get(&self.parameters[3])?;
        if width == 0 || height == 0 {
            return Err(error(source, "2D copy dimensions must be nonzero"));
        }
        let surface =
            |r: &[MaxwellTwoDRegister<u32>; 10],
             dst: bool,
             x: u32,
             y: u32|
             -> Result<(u64, MaxwellMemoryCopyLayout), MaxwellEngineDispatchError> {
                let sw = get(&r[6])?;
                let sh = get(&r[7])?;
                if x.checked_add(width).is_none_or(|end| end > sw)
                    || y.checked_add(height).is_none_or(|end| end > sh)
                {
                    return Err(error(source, "2D copy rectangle exceeds its surface"));
                }
                let base = (u64::from(get(&r[8])?) << 32) | u64::from(get(&r[9])?);
                let byte = |v: u32| {
                    v.checked_mul(4)
                        .ok_or_else(|| error(source, "2D copy byte dimensions overflow"))
                };
                if get(&r[1])? == 1 {
                    let pitch = get(&r[5])?;
                    if pitch == 0 {
                        return Err(error(source, "2D copy pitch must be nonzero"));
                    }
                    let address = base
                        .checked_add(u64::from(y) * u64::from(pitch))
                        .and_then(|v| v.checked_add(u64::from(x) * 4))
                        .ok_or_else(|| error(source, "2D pitch address overflows"))?;
                    Ok((address, MaxwellMemoryCopyLayout::Pitch { pitch }))
                } else {
                    let block = get(&r[2])?;
                    if block & !0x70 != 0
                        || ((block >> 4) & 7) > 5
                        || get(&r[3])? != 1
                        || (dst && get(&r[4])? != 0)
                    {
                        return Err(error(
                            source,
                            "2D copy requires single-layer 2D block-linear surfaces",
                        ));
                    }
                    Ok((
                        base,
                        MaxwellMemoryCopyLayout::BlockLinear {
                            surface_width: byte(sw)?,
                            surface_height: sh,
                            x: byte(x)?,
                            y,
                            block_height_log2: (block >> 4) as u8,
                        },
                    ))
                }
            };
        let (source_address, source_layout) = surface(
            &self.src,
            false,
            get(&self.parameters[9])?,
            get(&self.parameters[11])?,
        )?;
        let (destination_address, destination_layout) = surface(
            &self.dst,
            true,
            get(&self.parameters[0])?,
            get(&self.parameters[1])?,
        )?;
        let byte_width = width
            .checked_mul(4)
            .ok_or_else(|| error(source, "2D copy byte width overflows"))?;
        MaxwellMemoryCopyOperation::byte_copy(
            source_address,
            destination_address,
            source_layout,
            destination_layout,
            byte_width,
            height,
            source,
        )
        .map_err(|_| error(source, "2D copy address range overflows"))
    }
}

/// O(1) decoding of the bounded public surface/launch register ranges.
pub(super) fn apply(
    method: MaxwellMethodDispatch,
    state: &mut MaxwellTwoDState,
) -> Option<Result<AppliedMethod, MaxwellEngineDispatchError>> {
    let source = method.source();
    let address = source.method().0;
    // https://github.com/NVIDIA/open-gpu-doc/blob/master/classes/twod/cl902d.h
    let (register, name, mask) = match address {
        0x0200..=0x0224 | 0x0230..=0x0254 if address & 3 == 0 && address != 0x0240 => {
            let dst = address < 0x0230;
            let slot = ((address - if dst { 0x0200 } else { 0x0230 }) / 4) as usize;
            let name = if dst {
                [
                    "SET_DST_FORMAT",
                    "SET_DST_MEMORY_LAYOUT",
                    "SET_DST_BLOCK_SIZE",
                    "SET_DST_DEPTH",
                    "SET_DST_LAYER",
                    "SET_DST_PITCH",
                    "SET_DST_WIDTH",
                    "SET_DST_HEIGHT",
                    "SET_DST_OFFSET_UPPER",
                    "SET_DST_OFFSET_LOWER",
                ][slot]
            } else {
                [
                    "SET_SRC_FORMAT",
                    "SET_SRC_MEMORY_LAYOUT",
                    "SET_SRC_BLOCK_SIZE",
                    "SET_SRC_DEPTH",
                    "",
                    "SET_SRC_PITCH",
                    "SET_SRC_WIDTH",
                    "SET_SRC_HEIGHT",
                    "SET_SRC_OFFSET_UPPER",
                    "SET_SRC_OFFSET_LOWER",
                ][slot]
            };
            let mask = match slot {
                0 | 8 => 0xff,
                1 => 1,
                2 => 0x770,
                _ => u32::MAX,
            };
            let register = if dst {
                &mut state.blit.dst[slot]
            } else {
                &mut state.blit.src[slot]
            };
            (register, name, mask)
        }
        0x02d4 => (&mut state.blit.compression, "SET_DST_COMPRESSION", 1),
        0x088c => (
            &mut state.blit.sample_mode,
            "SET_PIXELS_FROM_MEMORY_SAMPLE_MODE",
            0x11,
        ),
        0x08b0..=0x08dc if address & 3 == 0 => {
            let slot = ((address - 0x08b0) / 4) as usize;
            let name = [
                "SET_PIXELS_FROM_MEMORY_DST_X0",
                "SET_PIXELS_FROM_MEMORY_DST_Y0",
                "SET_PIXELS_FROM_MEMORY_DST_WIDTH",
                "SET_PIXELS_FROM_MEMORY_DST_HEIGHT",
                "SET_PIXELS_FROM_MEMORY_DU_DX_FRAC",
                "SET_PIXELS_FROM_MEMORY_DU_DX_INT",
                "SET_PIXELS_FROM_MEMORY_DV_DY_FRAC",
                "SET_PIXELS_FROM_MEMORY_DV_DY_INT",
                "SET_PIXELS_FROM_MEMORY_SRC_X0_FRAC",
                "SET_PIXELS_FROM_MEMORY_SRC_X0_INT",
                "SET_PIXELS_FROM_MEMORY_SRC_Y0_FRAC",
                "PIXELS_FROM_MEMORY_SRC_Y0_INT",
            ][slot];
            (&mut state.blit.parameters[slot], name, u32::MAX)
        }
        _ => return None,
    };
    if source.argument() & !mask != 0 {
        return Some(Err(MaxwellEngineDispatchError::InvalidTwoDMethodEncoding {
            source,
            method_name: name,
            reason: "reserved bits are set",
        }));
    }
    *register = MaxwellTwoDRegister::programmed(source.argument(), source.argument(), source);
    let operation = if address == 0x08dc {
        match state.blit.launch(state, source) {
            Ok(operation) => Some(operation),
            Err(error) => return Some(Err(error)),
        }
    } else {
        None
    };
    Some(Ok(AppliedMethod::new(
        method,
        MaxwellEngineMethodMetadata::new(CLASS, "FERMI_TWOD_A", GpuMethodId(address), name),
        operation,
    )))
}
