//! Native polygon-line state. Width is dynamic, not a pipeline specialization.
use super::*;

pub(super) fn key(
    state: TriangleRasterization,
) -> (std::mem::Discriminant<TriangleRasterization>, bool) {
    (
        std::mem::discriminant(&state),
        matches!(state, TriangleRasterization::Wireframe { smooth: true, .. }),
    )
}

pub(super) fn width_bits(state: TriangleRasterization) -> Option<u32> {
    match state {
        TriangleRasterization::Wireframe { width_bits, .. } => Some(width_bits),
        _ => None,
    }
}

pub(super) fn validate(
    state: TriangleRasterization,
    caps: crate::VulkanRasterCapabilities,
) -> Result<(), BackendDriverError> {
    let TriangleRasterization::Wireframe { width_bits, smooth } = state else {
        return if state == TriangleRasterization::Fill {
            Ok(())
        } else {
            Err(unsupported(
                "native tessellation fill-rectangle rasterization",
            ))
        };
    };
    if !caps.wireframe {
        return Err(unsupported("native wireframe requires fillModeNonSolid"));
    }
    validate_line(nixe_gpu::LineRasterization { width_bits, smooth }, caps)
}

pub(super) fn validate_line(
    line: nixe_gpu::LineRasterization,
    caps: crate::VulkanRasterCapabilities,
) -> Result<(), BackendDriverError> {
    let nixe_gpu::LineRasterization { width_bits, smooth } = line;
    if (smooth && !caps.smooth_lines) || (!smooth && !caps.rectangular_lines) {
        return Err(unsupported(if smooth {
            "native smooth lines require line-rasterization smoothLines"
        } else {
            "native rectangular lines require line-rasterization rectangularLines"
        }));
    }
    let width = f32::from_bits(width_bits);
    let [min, max] = caps.line_width_range_bits.map(f32::from_bits);
    // Do not clamp or replace a consumed guest width to satisfy the host.
    // https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdSetLineWidth.html
    if !width.is_finite() || width <= 0.0 || width < min || width > max {
        return Err(error(format!(
            "line width {width:?} (0x{width_bits:08x}) is outside the supported positive finite range [{min}, {max}]"
        )));
    }
    if width != 1.0 && !caps.wide_lines {
        return Err(unsupported(
            "native line width other than one requires wideLines",
        ));
    }
    Ok(())
}
