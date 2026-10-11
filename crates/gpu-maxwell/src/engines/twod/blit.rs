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
    Scale {
        filter: nixe_gpu::BlitFilter,
        first: [i64; 2],
        step: [i64; 2],
        destination_origin: nixe_gpu::ImageOrigin,
        extent: nixe_gpu::ImageExtent,
    },
    Downsample2x {
        filter: nixe_gpu::BlitFilter,
    },
    Convert {
        origins: [nixe_gpu::ImageOrigin; 2],
        extent: nixe_gpu::ImageExtent,
    },
    Copy {
        origins: [nixe_gpu::ImageOrigin; 2],
        extent: nixe_gpu::ImageExtent,
    },
}

/// Validated 2D copy, color conversion or full affine resampling.
/// The source PTE kind distinguishes multisample resolve from texel filtering.
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

// Public color format enums; all supported formats have one texel per block.
// https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/twod/cl902d.h
fn pixel_bytes(format: u8) -> Option<u32> {
    match format {
        0xca => Some(8),
        0xf3 => Some(1),
        0xcf | 0xd0 | 0xd5 | 0xd6 | 0xe0 => Some(4),
        _ => None,
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
        if self
            .sample_mode
            .value()
            .is_some_and(|mode| mode & 0x10 == 0)
            && self.parameters[4..8]
                .iter()
                .map(|r| r.value().copied())
                .eq([Some(0), Some(1), Some(0), Some(1)])
        {
            let (images, origins, extent) = self.point_surfaces(source)?;
            let converting = images[0].format != images[1].format;
            if !converting
                && self.compression.value() != Some(&1)
                && matches!(images[0].layout, BlitSurfaceLayout::Pitch(_))
            {
                let bytes = pixel_bytes(images[0].format).expect("validated point format");
                let surface =
                    |image: BlitSurface,
                     origin: nixe_gpu::ImageOrigin|
                     -> Result<_, MaxwellEngineDispatchError> {
                        match image.layout {
                            BlitSurfaceLayout::Pitch(pitch) => Ok((
                                image
                                    .address
                                    .checked_add(u64::from(origin.y) * u64::from(pitch))
                                    .and_then(|address| {
                                        address.checked_add(u64::from(origin.x) * u64::from(bytes))
                                    })
                                    .ok_or_else(|| error(source, "2D pitch address overflows"))?,
                                MaxwellMemoryCopyLayout::Pitch { pitch },
                            )),
                            BlitSurfaceLayout::BlockLinear(block_height_log2) => Ok((
                                image.address,
                                MaxwellMemoryCopyLayout::BlockLinear {
                                    surface_width: image.width.checked_mul(bytes).ok_or_else(
                                        || error(source, "2D byte dimensions overflow"),
                                    )?,
                                    surface_height: image.height,
                                    x: origin.x.checked_mul(bytes).ok_or_else(|| {
                                        error(source, "2D byte dimensions overflow")
                                    })?,
                                    y: origin.y,
                                    block_height_log2,
                                },
                            )),
                        }
                    };
                let (src, src_layout) = surface(images[0], origins[0])?;
                let (dst, dst_layout) = surface(images[1], origins[1])?;
                let copy = MaxwellMemoryCopyOperation::byte_copy(
                    src,
                    dst,
                    src_layout,
                    dst_layout,
                    extent
                        .width
                        .checked_mul(bytes)
                        .ok_or_else(|| error(source, "2D byte dimensions overflow"))?,
                    extent.height,
                    source,
                )
                .map_err(|_| error(source, "2D copy address range overflows"))?;
                return Ok(PendingEngineOperation::MemoryCopy(copy));
            }
            return Ok(PendingEngineOperation::TwoDBlit(MaxwellTwoDBlitOperation {
                source,
                images,
                destination_compression: self.compression.value() == Some(&1),
                kind: if converting {
                    BlitKind::Convert { origins, extent }
                } else {
                    BlitKind::Copy { origins, extent }
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
            if pixel_bytes(format as u8).is_none()
                || get(&r[1])? != 0
                || block & !0x70 != 0
                || ((block >> 4) & 7) > 5
                || get(&r[3])? != 1
                || (dst && get(&r[4])? != 0)
            {
                return Err(error(
                    source,
                    "2D downsample requires supported block-linear 2D single-layer color surfaces",
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
        if src.width == 0 || src.height == 0 || dst.width == 0 || dst.height == 0 {
            return Err(error(source, "2D sampling surfaces must be nonempty"));
        }
        // deko3d averages each 2x2 sample block at its center. Other affine
        // transforms are blits, not hardware resolves, and must not be coerced.
        // https://github.com/devkitPro/deko3d/blob/master/source/dk_image.cpp#L666-L695
        let mode = get(&self.sample_mode)?;
        if mode & !0x11 != 0 {
            return Err(error(source, "2D downsample sample mode has reserved bits"));
        }
        let (fraction, integer) = if mode & 1 == 0 {
            (0x8000_0000, 0)
        } else {
            (0, 1)
        };
        let downsample = dst.width.checked_mul(2) == Some(src.width)
            && dst.height.checked_mul(2) == Some(src.height)
            && self.parameters.iter().map(|r| r.value().copied()).eq([
                0, 0, dst.width, dst.height, 0, 2, 0, 2, fraction, integer, fraction, integer,
            ]
            .map(Some));
        let filter = if mode & 0x10 == 0 {
            nixe_gpu::BlitFilter::Point
        } else {
            nixe_gpu::BlitFilter::Linear
        };
        let kind = if downsample {
            BlitKind::Downsample2x { filter }
        } else {
            // NVIDIA's increments and initial coordinates are 32.32 fixed
            // point. CENTER coordinates put texel centers at integers; convert
            // to the corner-coordinate convention used by neutral blits.
            // https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/twod/cl902d.h
            let destination_origin = nixe_gpu::ImageOrigin {
                x: get(&self.parameters[0])?,
                y: get(&self.parameters[1])?,
                z: 0,
            };
            let extent = nixe_gpu::ImageExtent {
                width: get(&self.parameters[2])?,
                height: get(&self.parameters[3])?,
                depth: 1,
            };
            if extent.width == 0
                || extent.height == 0
                || destination_origin
                    .x
                    .checked_add(extent.width)
                    .is_none_or(|end| end > dst.width)
                || destination_origin
                    .y
                    .checked_add(extent.height)
                    .is_none_or(|end| end > dst.height)
            {
                return Err(error(
                    source,
                    "scaled 2D destination rectangle is empty or exceeds its surface",
                ));
            }
            let fixed = |slot: usize| -> Result<i64, MaxwellEngineDispatchError> {
                Ok(((u64::from(get(&self.parameters[slot + 1])?) << 32)
                    | u64::from(get(&self.parameters[slot])?)) as i64)
            };
            let step = [fixed(4)?, fixed(6)?];
            let adjustment = if mode & 1 == 0 { 1_i64 << 31 } else { 0 };
            let first = [
                fixed(8)?.checked_add(adjustment),
                fixed(10)?.checked_add(adjustment),
            ];
            let [Some(first_x), Some(first_y)] = first else {
                return Err(error(source, "scaled 2D source coordinate overflows"));
            };
            let first = [first_x, first_y];
            for axis in 0..2 {
                let count = [extent.width, extent.height][axis];
                let limit = i128::from([src.width, src.height][axis]) << 32;
                let last = i128::from(first[axis]) + i128::from(step[axis]) * i128::from(count - 1);
                if step[axis] == 0
                    || i128::from(first[axis]) < 0
                    || i128::from(first[axis]) >= limit
                    || last < 0
                    || last >= limit
                {
                    return Err(error(source, "scaled 2D samples exceed the source surface"));
                }
            }
            BlitKind::Scale {
                filter,
                first,
                step,
                destination_origin,
                extent,
            }
        };
        Ok(PendingEngineOperation::TwoDBlit(MaxwellTwoDBlitOperation {
            source,
            images,
            destination_compression: get(&self.compression)? != 0,
            kind,
        }))
    }
    // Unit point sampling is an exact texel mapping for both origins. deko3d
    // places the corner sample at a half texel and the center sample at zero.
    // https://github.com/devkitPro/deko3d/blob/master/source/dk_image.cpp#L774-L776
    fn point_surfaces(
        &self,
        source: MaxwellMethodSource,
    ) -> Result<
        (
            [BlitSurface; 2],
            [nixe_gpu::ImageOrigin; 2],
            nixe_gpu::ImageExtent,
        ),
        MaxwellEngineDispatchError,
    > {
        let get = |r: &MaxwellTwoDRegister<u32>| {
            r.value()
                .copied()
                .ok_or_else(|| error(source, "incomplete 2D point state"))
        };
        let fraction = if get(&self.sample_mode)? & 1 == 0 {
            0
        } else {
            0x8000_0000
        };
        if [
            get(&self.parameters[4])?,
            get(&self.parameters[5])?,
            get(&self.parameters[6])?,
            get(&self.parameters[7])?,
            get(&self.parameters[8])?,
            get(&self.parameters[10])?,
        ] != [0, 1, 0, 1, fraction, fraction]
        {
            return Err(error(
                source,
                "2D point blit requires texel-aligned coordinates and unit scaling",
            ));
        }
        let extent = nixe_gpu::ImageExtent {
            width: get(&self.parameters[2])?,
            height: get(&self.parameters[3])?,
            depth: 1,
        };
        if extent.width == 0 || extent.height == 0 {
            return Err(error(source, "2D point dimensions must be nonzero"));
        }
        let origins = [
            nixe_gpu::ImageOrigin {
                x: get(&self.parameters[9])?,
                y: get(&self.parameters[11])?,
                z: 0,
            },
            nixe_gpu::ImageOrigin {
                x: get(&self.parameters[0])?,
                y: get(&self.parameters[1])?,
                z: 0,
            },
        ];
        let surface = |r: &[MaxwellTwoDRegister<u32>; 10],
                       origin: nixe_gpu::ImageOrigin,
                       destination: bool|
         -> Result<BlitSurface, MaxwellEngineDispatchError> {
            let format = get(&r[0])? as u8;
            if pixel_bytes(format).is_none() {
                return Err(error(source, "2D point surface format is not implemented"));
            }
            let width = get(&r[6])?;
            let height = get(&r[7])?;
            if origin
                .x
                .checked_add(extent.width)
                .is_none_or(|end| end > width)
                || origin
                    .y
                    .checked_add(extent.height)
                    .is_none_or(|end| end > height)
            {
                return Err(error(source, "2D point rectangle exceeds its surface"));
            }
            let layout = if get(&r[1])? == 1 {
                let pitch = get(&r[5])?;
                if pitch == 0 {
                    return Err(error(source, "2D point pitch must be nonzero"));
                }
                BlitSurfaceLayout::Pitch(pitch)
            } else {
                let block = get(&r[2])?;
                if block & !0x70 != 0
                    || ((block >> 4) & 7) > 5
                    || get(&r[3])? != 1
                    || (destination && get(&r[4])? != 0)
                {
                    return Err(error(
                        source,
                        "2D point surfaces require single-layer block-linear 2D layout",
                    ));
                }
                BlitSurfaceLayout::BlockLinear((block >> 4) as u8)
            };
            Ok(BlitSurface {
                address: (u64::from(get(&r[8])?) << 32) | u64::from(get(&r[9])?),
                width,
                height,
                format,
                layout,
            })
        };
        Ok((
            [
                surface(&self.src, origins[0], false)?,
                surface(&self.dst, origins[1], true)?,
            ],
            origins,
            extent,
        ))
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
            Err(error) => {
                log::debug!("untranslated 2D blit state: {:?}", state.blit);
                return Some(Err(error));
            }
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
