//! Source-preserving 2D surfaces and the pixels-from-memory launch boundary.
use super::{
    CLASS, MaxwellTwoDClipEnable, MaxwellTwoDColorKeyEnable, MaxwellTwoDOperation,
    MaxwellTwoDRegister, MaxwellTwoDRenderEnableMode, MaxwellTwoDState,
};
use crate::engines::{
    AppliedMethod, MaxwellEngineDispatchError, MaxwellEngineMethodMetadata, PendingEngineOperation,
};
use crate::{MaxwellMethodDispatch, MaxwellMethodSource};
use nixe_gpu::GpuMethodId;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MaxwellTwoDBlitState {
    src: [MaxwellTwoDRegister<u32>; 10],
    dst: [MaxwellTwoDRegister<u32>; 10],
    parameters: [MaxwellTwoDRegister<u32>; 12],
    sample_mode: MaxwellTwoDRegister<u32>,
    compression: MaxwellTwoDRegister<u32>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct ResolveSurface {
    pub address: u64,
    pub width: u32,
    pub height: u32,
    pub format: u8,
    pub block_height_log2: u8,
}

/// A 2:1 bilinear 2D operation eligible for a four-sample resolve. Memory-kind
/// and resident-source validation must still establish that it really is MSAA.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellTwoDResolveOperation {
    pub(crate) source: MaxwellMethodSource,
    pub(crate) images: [ResolveSurface; 2],
    pub(crate) destination_compression: bool,
}

impl MaxwellTwoDResolveOperation {
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
    ) -> Result<MaxwellTwoDResolveOperation, MaxwellEngineDispatchError> {
        if state.operation().value() != Some(&MaxwellTwoDOperation::SourceCopy) {
            return Err(error(source, "2D resolve requires SET_OPERATION=SRCCOPY"));
        }
        if state.clip_enable().value() != Some(&MaxwellTwoDClipEnable::Disabled) {
            return Err(error(source, "2D resolve requires SET_CLIP_ENABLE=FALSE"));
        }
        if state.color_key_enable().value() != Some(&MaxwellTwoDColorKeyEnable::Disabled) {
            return Err(error(
                source,
                "2D resolve requires SET_COLOR_KEY_ENABLE=FALSE",
            ));
        }
        if state.render_enable().mode().value() != Some(&MaxwellTwoDRenderEnableMode::Enabled) {
            return Err(error(
                source,
                "2D resolve requires SET_RENDER_ENABLE_C=TRUE",
            ));
        }
        let get = |register: &MaxwellTwoDRegister<u32>| {
            register
                .value()
                .copied()
                .ok_or_else(|| error(source, "incomplete 2D surface or pixels-from-memory state"))
        };
        let surface = |r: &[MaxwellTwoDRegister<u32>; 10],
                       dst: bool|
         -> Result<ResolveSurface, MaxwellEngineDispatchError> {
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
            Ok(ResolveSurface {
                address: (u64::from(get(&r[8])?) << 32) | u64::from(get(&r[9])?),
                width: get(&r[6])?,
                height: get(&r[7])?,
                format: format as u8,
                block_height_log2: (block >> 4) as u8,
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
        Ok(MaxwellTwoDResolveOperation {
            source,
            images,
            destination_compression: get(&self.compression)? != 0,
        })
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
            Ok(resolve) => Some(PendingEngineOperation::TwoDResolve(resolve)),
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
