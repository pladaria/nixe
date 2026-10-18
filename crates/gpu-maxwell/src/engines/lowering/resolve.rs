use super::*;
pub(super) fn solid_clear_value(
    packed: u32,
    color_format: u8,
    guest_format: super::super::threed::MaxwellThreeDGuestImageFormat,
) -> Result<ClearValue, MaxwellLoweringError> {
    use super::super::threed::{
        MaxwellThreeDDepthStencilFormat as Depth, MaxwellThreeDGuestImageFormat as Guest,
    };
    match guest_format {
        // NVIDIA names are in MSB-to-LSB order: S8Z24 is Mesa Z24_UNORM_S8_UINT,
        // while Z24S8 is S8_UINT_Z24_UNORM. Preserve all packed bits in a zeta write.
        // https://github.com/chaotic-cx/mesa-mirror/blob/main/src/gallium/drivers/nouveau/nv50/nv50_formats.c#L138-L140
        Guest::DepthStencil(Depth::Stencil8Z24) => Ok(ClearValue::DepthStencil {
            depth: (packed & 0xffffff) as f32 / 16777215.0,
            stencil: (packed >> 24) as u8,
        }),
        Guest::DepthStencil(Depth::Z24Stencil8) => Ok(ClearValue::DepthStencil {
            depth: (packed >> 8) as f32 / 16777215.0,
            stencil: packed as u8,
        }),
        Guest::Color(_) => {
            let bytes = packed.to_le_bytes();
            let rgba = if color_format == 0xcf {
                [bytes[2], bytes[1], bytes[0], bytes[3]]
            } else {
                bytes
            };
            Ok(ClearValue::Color(rgba.map(|v| f32::from(v) / 255.0)))
        }
        _ => Err(MaxwellLoweringError::IncompleteClear(
            "unsupported 2D solid pixel encoding",
        )),
    }
}
