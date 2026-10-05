//! Solid 2D rectangles. The second point's Y method launches a rectangle.
//!
//! NVIDIA defines the registers and formats; Nouveau submits two (x,y) points
//! for an exclusive-upper-bound rectangle, including repeated rectangle draws.
//! https://github.com/NVIDIA/open-gpu-doc/blob/9e6d83fe0770bc8644850a0b1bf5ddb1519905ba/classes/twod/cl902d.h#L699-L807
//! https://github.com/X11Libre/xf86-video-nouveau/blob/master/src/nv50_exa.c#L196-L241
use super::{
    blit::{BlitSurface, BlitSurfaceLayout},
    *,
};
use crate::engines::{AppliedMethod, MaxwellEngineMethodMetadata, PendingEngineOperation};
use crate::{MaxwellMethodDispatch, MaxwellMethodSource};
use nixe_gpu::GpuMethodId;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(super) struct SolidState {
    zeta: MaxwellTwoDRegister<u32>,
    pattern_format: MaxwellTwoDRegister<u32>,
    mode: MaxwellTwoDRegister<u32>,
    format: MaxwellTwoDRegister<u32>,
    color: [MaxwellTwoDRegister<u32>; 4],
    points: [MaxwellTwoDRegister<u32>; 4],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellTwoDSolidOperation {
    pub(crate) source: MaxwellMethodSource,
    pub(crate) destination: BlitSurface,
    pub(crate) compression: bool,
    pub(crate) zeta: bool,
    pub(crate) color: u32,
    pub(crate) rectangle: [u32; 4],
}

fn error(source: MaxwellMethodSource, reason: &'static str) -> MaxwellEngineDispatchError {
    MaxwellEngineDispatchError::InvalidTwoDMethodEncoding {
        source,
        method_name: "RENDER_SOLID_PRIM_POINT_Y",
        reason,
    }
}

fn launch(
    state: &MaxwellTwoDState,
    source: MaxwellMethodSource,
) -> Result<MaxwellTwoDSolidOperation, MaxwellEngineDispatchError> {
    let get = |r: &MaxwellTwoDRegister<u32>| {
        r.value()
            .copied()
            .ok_or_else(|| error(source, "incomplete solid rectangle state"))
    };
    if get(&state.solid.mode)? != 4 {
        return Err(error(source, "only solid RECT primitives are implemented"));
    }
    if state.operation().value() != Some(&MaxwellTwoDOperation::SourceCopy)
        || state.clip_enable().value() != Some(&MaxwellTwoDClipEnable::Disabled)
        || state.color_key_enable().value() != Some(&MaxwellTwoDColorKeyEnable::Disabled)
        || state.render_enable().mode().value() != Some(&MaxwellTwoDRenderEnableMode::Enabled)
    {
        return Err(error(
            source,
            "solid rectangles require SRCCOPY, disabled clipping/keying and unconditional rendering",
        ));
    }
    let dst = &state.blit.dst;
    let format = get(&dst[0])?;
    if !matches!(format, 0xcf | 0xd5) || get(&state.solid.format)? != format {
        return Err(error(
            source,
            "solid rectangles require matching BGRA8/RGBA8 UNORM formats",
        ));
    }
    let width = get(&dst[6])?;
    let height = get(&dst[7])?;
    let layout = if get(&dst[1])? == 1 {
        let pitch = get(&dst[5])?;
        if width.checked_mul(4).is_none_or(|row| row > pitch) {
            return Err(error(
                source,
                "solid rectangle pitch is smaller than the row",
            ));
        }
        BlitSurfaceLayout::Pitch(pitch)
    } else {
        let block = get(&dst[2])?;
        if block & !0x70 != 0 || (block >> 4) > 5 || get(&dst[3])? != 1 || get(&dst[4])? != 0 {
            return Err(error(
                source,
                "solid rectangles require single-layer 2D surfaces",
            ));
        }
        BlitSurfaceLayout::BlockLinear((block >> 4) as u8)
    };
    let mut rectangle = [0; 4];
    for (out, coordinate) in rectangle.iter_mut().zip(&state.solid.points) {
        *out = (get(coordinate)? as i32).max(0) as u32;
    }
    rectangle[0] = rectangle[0].min(width);
    rectangle[2] = rectangle[2].min(width);
    rectangle[1] = rectangle[1].min(height);
    rectangle[3] = rectangle[3].min(height);
    Ok(MaxwellTwoDSolidOperation {
        source,
        destination: BlitSurface {
            address: (u64::from(get(&dst[8])?) << 32) | u64::from(get(&dst[9])?),
            width,
            height,
            format: format as u8,
            layout,
        },
        compression: state.blit.compression.value() == Some(&1),
        zeta: get(&state.solid.zeta)? != 0,
        color: get(&state.solid.color[0])?,
        rectangle,
    })
}

pub(super) fn apply(
    method: MaxwellMethodDispatch,
    state: &mut MaxwellTwoDState,
) -> Option<Result<AppliedMethod, MaxwellEngineDispatchError>> {
    let source = method.source();
    let address = source.method().0;
    if matches!(address, 0x0604 | 0x060c) && state.solid.mode.value() != Some(&4) {
        return Some(Err(error(
            source,
            "only solid RECT primitives are implemented",
        )));
    }
    let (register, name, mask) = match address {
        0x02b8 => (
            &mut state.solid.zeta,
            "SET_DST_COLOR_RENDER_TO_ZETA_SURFACE",
            1,
        ),
        0x02e8 => (
            &mut state.solid.pattern_format,
            "SET_MONOCHROME_PATTERN_COLOR_FORMAT",
            0xff,
        ),
        0x0540..=0x054c if address & 3 == 0 => (
            &mut state.solid.color[((address - 0x0540) / 4) as usize],
            "SET_RENDER_SOLID_PRIM_COLOR",
            u32::MAX,
        ),
        0x0588 => (
            &mut state.solid.color[0],
            "SET_RENDER_SOLID_PRIM_COLOR",
            u32::MAX,
        ),
        0x0580 => (&mut state.solid.mode, "RENDER_SOLID_PRIM_MODE", 7),
        0x0584 => (
            &mut state.solid.format,
            "SET_RENDER_SOLID_PRIM_COLOR_FORMAT",
            0xff,
        ),
        0x0600..=0x060c if address & 3 == 0 => (
            &mut state.solid.points[((address - 0x0600) / 4) as usize],
            if address & 4 == 0 {
                "RENDER_SOLID_PRIM_POINT_SET_X"
            } else {
                "RENDER_SOLID_PRIM_POINT_Y"
            },
            u32::MAX,
        ),
        _ => return None,
    };
    if source.argument() & !mask != 0 {
        return Some(Err(error(source, "reserved bits are set")));
    }
    *register = MaxwellTwoDRegister::programmed(source.argument(), source.argument(), source);
    let operation = if address == 0x060c {
        match launch(state, source) {
            Ok(rect) => Some(PendingEngineOperation::TwoDSolid(rect)),
            Err(e) => return Some(Err(e)),
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
