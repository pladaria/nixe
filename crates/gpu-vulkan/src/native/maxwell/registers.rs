//! Native state is consumed directly from Maxwell3D's canonical register bank.
//! Constant address reads replace neutral raster/depth/stencil/viewport objects.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/fixed_pipeline_state.cpp
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/vk_rasterizer.cpp
use crate::{DeviceProfile, VulkanError};
use ash::vk;
use nixe_gpu_maxwell::engines::threed::state::GeneratedPrimitive;
pub(super) type Regs = [u32; 0xe00];
/// Render-pass formats decode the hardware words at native consumption, as
/// Eden MakeRenderPassKey/MaxwellToVK. Image layout/footprint belongs to ImageInfo.
/// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/vk_graphics_pipeline.cpp
pub(super) fn color_attachment_format(raw: u8, index: usize) -> Result<vk::Format, VulkanError> {
    Ok(match raw {
        0xc6 => vk::Format::R16G16B16A16_UNORM,
        0xc7 => vk::Format::R16G16B16A16_SNORM,
        0xd7 => vk::Format::R8G8B8A8_SNORM,
        0xda => vk::Format::R16G16_UNORM,
        0xdb => vk::Format::R16G16_SNORM,
        0xdf => vk::Format::A2R10G10B10_UNORM_PACK32,
        0xe9 | 0xf8 => vk::Format::A1R5G5B5_UNORM_PACK16,
        0xeb => vk::Format::R8G8_SNORM,
        0xee => vk::Format::R16_UNORM,
        0xef => vk::Format::R16_SNORM,
        0xf4 => vk::Format::R8_SNORM,
        0 => vk::Format::UNDEFINED,
        0xc0 | 0xc3 => vk::Format::R32G32B32A32_SFLOAT,
        0xca | 0xce => vk::Format::R16G16B16A16_SFLOAT,
        0xcb => vk::Format::R32G32_SFLOAT,
        0xcf | 0xe6 => vk::Format::B8G8R8A8_UNORM,
        0xd0 | 0xe7 => vk::Format::B8G8R8A8_SRGB,
        0xd1 => vk::Format::A2B10G10R10_UNORM_PACK32,
        0xd5 | 0xf9 => vk::Format::R8G8B8A8_UNORM,
        0xd6 | 0xfa => vk::Format::R8G8B8A8_SRGB,
        0xde => vk::Format::R16G16_SFLOAT,
        0xe0 => vk::Format::B10G11R11_UFLOAT_PACK32,
        0xe5 => vk::Format::R32_SFLOAT,
        0xe8 => vk::Format::R5G6B5_UNORM_PACK16,
        0xea => vk::Format::R8G8_UNORM,
        0xf2 => vk::Format::R16_SFLOAT,
        0xf3 => vk::Format::R8_UNORM,
        value => {
            return Err(unsupported(
                0x0810 + index as u32 * 0x40,
                u32::from(value),
                "unsupported color attachment format",
            ));
        }
    })
}
pub(super) fn depth_attachment_format(raw: u32) -> Result<vk::Format, VulkanError> {
    Ok(match raw {
        0x0a => vk::Format::D32_SFLOAT,
        0x13 => vk::Format::D16_UNORM,
        0x14 | 0x16 => vk::Format::D24_UNORM_S8_UINT,
        0x19 => vk::Format::D32_SFLOAT_S8_UINT,
        value => {
            return Err(unsupported(
                0x0fe8,
                value,
                "unsupported depth attachment format",
            ));
        }
    })
}
pub(super) fn attachment_samples(raw: u32) -> Result<vk::SampleCountFlags, VulkanError> {
    Ok(match raw {
        0 => vk::SampleCountFlags::TYPE_1,
        2 => vk::SampleCountFlags::TYPE_4,
        value => {
            return Err(unsupported(
                0x15d0,
                value,
                "unsupported attachment sample mode",
            ));
        }
    })
}
pub(in crate::native) fn surface_extent(r: &Regs) -> [u32; 2] {
    [r[0x0ff4 / 4] >> 16, r[0x0ff8 / 4] >> 16]
}
pub(super) fn topology(value: u32) -> Result<vk::PrimitiveTopology, VulkanError> {
    Ok(match value {
        0 => vk::PrimitiveTopology::POINT_LIST,
        1 => vk::PrimitiveTopology::LINE_LIST,
        3 => vk::PrimitiveTopology::LINE_STRIP,
        4 | 7 | 8 => vk::PrimitiveTopology::TRIANGLE_LIST,
        5 => vk::PrimitiveTopology::TRIANGLE_STRIP,
        6 => vk::PrimitiveTopology::TRIANGLE_FAN,
        14 => vk::PrimitiveTopology::PATCH_LIST,
        _ => {
            return Err(unsupported(
                0x1618,
                value,
                "unsupported current draw topology",
            ));
        }
    })
}
/// Make native draw parameters from the channel's actual DrawManager state.
/// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/vk_rasterizer.cpp
pub(in crate::native) fn draw_arguments(
    state: &super::MaxwellThreeDState,
    draw: &super::MaxwellThreeDDrawState,
) -> Result<crate::Draw, VulkanError> {
    let vertex_id_base = state.register_words()[0x1118 / 4];
    if draw.indexed {
        if draw.base_vertex != vertex_id_base {
            return Err(unsupported(
                0x1118,
                vertex_id_base,
                "distinct vertex-fetch and shader-ID bases are unsupported",
            ));
        }
        Ok(crate::Draw::Indexed {
            indices: draw.index.count,
            instances: draw.instance_count,
            first_index: draw.index.first,
            vertex_offset: draw.base_vertex as i32,
            first_instance: draw.base_instance,
        })
    } else {
        if draw.base_vertex != 0 || vertex_id_base != 0 {
            return Err(unsupported(
                0x1118,
                vertex_id_base,
                "non-indexed shader-ID base adjustment is unsupported",
            ));
        }
        Ok(crate::Draw::Direct {
            vertices: draw.vertex_count,
            instances: draw.instance_count,
            first_vertex: draw.vertex_first,
            first_instance: draw.base_instance,
        })
    }
}
pub(super) fn unsupported(method: u32, value: u32, reason: &str) -> VulkanError {
    VulkanError::Compilation(format!("MAXWELL_B {method:#06x}={value:#x}: {reason}").into())
}
pub(super) fn pack_comparison(method: u32, raw: u32) -> Result<u32, VulkanError> {
    match raw {
        0x200..=0x207 => Ok(raw - 0x200),
        1..=8 => Ok(raw - 1),
        _ => Err(unsupported(method, raw, "unsupported comparison operation")),
    }
}
pub(super) fn compare(method: u32, raw: u32) -> Result<vk::CompareOp, VulkanError> {
    Ok(vk::CompareOp::from_raw(pack_comparison(method, raw)? as i32))
}
pub(super) fn boolean(r: &Regs, method: usize) -> Result<bool, VulkanError> {
    match r[method / 4] {
        0 => Ok(false),
        1 => Ok(true),
        value => Err(unsupported(method as u32, value, "invalid boolean state")),
    }
}
pub(super) fn pack_front_face(r: &Regs) -> Result<u32, VulkanError> {
    if r[0x13ac / 4] & !0x11 != 0 {
        return Err(unsupported(
            0x13ac,
            r[0x13ac / 4],
            "unsupported window origin",
        ));
    }
    let face = match r[0x191c / 4] {
        0x900 => 0,
        0x901 => 1,
        value => return Err(unsupported(0x191c, value, "unsupported front face")),
    };
    Ok(face ^ ((r[0x13ac / 4] >> 4) & 1))
}
pub(super) fn unpack_front_face(packed: u32) -> vk::FrontFace {
    // Maxwell GL clockwise packs to zero; Vulkan counter-clockwise is zero.
    // FixedPipelineState::UnpackFrontFace followed by MaxwellToVK::FrontFace.
    if packed == 0 {
        vk::FrontFace::CLOCKWISE
    } else {
        vk::FrontFace::COUNTER_CLOCKWISE
    }
}
pub(super) fn pack_cull_face(r: &Regs) -> Result<u32, VulkanError> {
    if !boolean(r, 0x1918)? {
        return Ok(0);
    }
    match r[0x1920 / 4] {
        0x404 => Ok(0),
        0x405 => Ok(1),
        0x408 => Ok(2),
        value => Err(unsupported(0x1920, value, "unsupported cull face")),
    }
}
pub(super) fn pack_polygon(r: &Regs) -> Result<u32, VulkanError> {
    if r[0x113c / 4] == 2 {
        return Ok(3);
    }
    match r[0x0dac / 4] {
        0x1b00..=0x1b02 => Ok(r[0x0dac / 4] - 0x1b00),
        value => Err(unsupported(0x0dac, value, "unsupported polygon mode")),
    }
}
pub(super) fn depth_bias_enabled(r: &Regs, topology: u32) -> Result<bool, VulkanError> {
    let method = match topology {
        0 => 0x0dc0,
        1..=3 | 10..=11 => 0x0dc4,
        _ => 0x0dc8,
    };
    boolean(r, method)
}
fn stencil_op(method: u32, raw: u32) -> Result<u16, VulkanError> {
    Ok(match raw {
        0x1e00 | 1 => 0,
        0 | 2 => 1,
        0x1e01 | 3 => 2,
        0x1e02 | 4 => 3,
        0x1e03 | 5 => 4,
        0x150a | 6 => 5,
        0x8507 | 7 => 6,
        0x8508 | 8 => 7,
        _ => return Err(unsupported(method, raw, "unsupported stencil operation")),
    })
}
pub(super) fn stencil_enabled(r: &Regs) -> bool {
    r[0x1538 / 4] != 0 && r[0x1380 / 4] == 1
}
pub(super) fn stencil_faces(r: &Regs) -> Result<[u16; 2], VulkanError> {
    let face = |base: usize| -> Result<u16, VulkanError> {
        Ok(stencil_op(base as u32, r[base / 4])?
            | (stencil_op(base as u32 + 4, r[base / 4 + 1])? << 3)
            | (stencil_op(base as u32 + 8, r[base / 4 + 2])? << 6)
            | ((pack_comparison(base as u32 + 12, r[base / 4 + 3])? as u16) << 9))
    };
    let front = face(0x1384)?;
    let back = match r[0x1594 / 4] {
        0 => front,
        1 => face(0x1598)?,
        value => {
            return Err(unsupported(
                0x1594,
                value,
                "invalid two-sided stencil enable",
            ));
        }
    };
    Ok([front, back])
}
pub(super) fn stencil_face(raw: u16) -> crate::cache::StencilState {
    crate::cache::StencilState {
        fail: vk::StencilOp::from_raw(i32::from(raw & 7)),
        depth_fail: vk::StencilOp::from_raw(i32::from((raw >> 3) & 7)),
        pass: vk::StencilOp::from_raw(i32::from((raw >> 6) & 7)),
        compare: vk::CompareOp::from_raw(i32::from((raw >> 9) & 7)),
    }
}
/// Validate unsupported raster effects when their canonical inputs change.
/// Native enums are produced by the static pipeline or individual dynamic setter.
fn validate_raster(
    r: &Regs,
    primitive: Option<GeneratedPrimitive>,
    topology: u32,
) -> Result<(), VulkanError> {
    boolean(r, 0x037c)?;
    if r[0x13ac / 4] & !0x11 != 0 {
        return Err(unsupported(
            0x13ac,
            r[0x13ac / 4],
            "unsupported window origin",
        ));
    }
    if primitive == Some(GeneratedPrimitive::Triangles) {
        if r[0x0dac / 4] != r[0x0db0 / 4] {
            return Err(unsupported(
                0x0db0,
                r[0x0db0 / 4],
                "independent front/back polygon modes are unsupported",
            ));
        }
        match r[0x0dac / 4] {
            0x1b02 => {
                if r[0x0db4 / 4] != 0 || r[0x168c / 4] != 0 {
                    return Err(unsupported(
                        0x0db4,
                        r[0x0db4 / 4],
                        "polygon smoothing/stipple is unsupported",
                    ));
                }
            }
            0x1b01 => {
                if r[0x113c / 4] != 0
                    || r[0x1570 / 4] != 1
                    || r[0x166c / 4] != 0
                    || r[0x0f8c / 4] != 0
                    || r[0x15e4 / 4] != 1
                {
                    return Err(unsupported(
                        0x0dac,
                        r[0x0dac / 4],
                        "unsupported polygon-line coverage, stipple or generated-edge state",
                    ));
                }
            }
            value => return Err(unsupported(0x0dac, value, "unsupported polygon mode")),
        }
    }
    if primitive == Some(GeneratedPrimitive::Lines) {
        if r[0x166c / 4] != 0 {
            return Err(unsupported(
                0x166c,
                r[0x166c / 4],
                "line stipple is unsupported",
            ));
        }
        if r[0x1570 / 4] != 1 && r[0x020c / 4] != 0 {
            return Err(unsupported(
                0x020c,
                r[0x020c / 4],
                "aliased line width selection is unsupported",
            ));
        }
    }
    depth_bias_enabled(r, topology)?;
    Ok(())
}
pub(super) fn viewport(r: &Regs, profile: &DeviceProfile) -> Result<vk::Viewport, VulkanError> {
    if r[0x0a1c / 4] != 0 {
        return Err(unsupported(
            0x0a1c,
            r[0x0a1c / 4],
            "viewport snap grid precision is unsupported",
        ));
    }
    let enabled = r[0x192c / 4] == 1;
    let reflect_y = match r[0x0a18 / 4] {
        0x6420 => false,
        0x6430 if enabled => true,
        value => {
            return Err(unsupported(
                0x0a18,
                value,
                "unsupported viewport coordinate swizzle",
            ));
        }
    };
    let viewport = if !enabled {
        if r[0x192c / 4] != 0 {
            return Err(unsupported(
                0x192c,
                r[0x192c / 4],
                "invalid viewport enable",
            ));
        }
        let [width, height] = surface_extent(r).map(|v| v.max(1));
        let x = (r[0x0ff4 / 4] & 0xffff) as f32;
        let mut y = (r[0x0ff8 / 4] & 0xffff) as f32;
        let mut height = height as f32;
        if r[0x13ac / 4] & 1 != 0 {
            y += height;
            height = -height;
        }
        vk::Viewport {
            x,
            y,
            width: width as f32,
            height,
            min_depth: 0.,
            max_depth: 1.,
        }
    } else {
        let sx = f32::from_bits(r[0x0a00 / 4]);
        let mut sy = f32::from_bits(r[0x0a04 / 4]);
        let ox = f32::from_bits(r[0x0a0c / 4]);
        let mut oy = f32::from_bits(r[0x0a10 / 4]);
        if reflect_y {
            sy = -sy;
        }
        if r[0x13ac / 4] & 1 != 0 {
            sy = -sy;
            oy = (r[0x0ff8 / 4] >> 16) as f32 - oy;
        }
        let scale = f32::from_bits(r[0x0a08 / 4]);
        let offset = f32::from_bits(r[0x0a14 / 4]);
        if r[0x0d7c / 4] > 1 {
            return Err(unsupported(
                0x0d7c,
                r[0x0d7c / 4],
                "invalid viewport depth mode",
            ));
        }
        let mut depth = [
            offset - if r[0x0d7c / 4] == 0 { scale } else { 0. },
            offset + scale,
        ];
        if !profile.depth_range_unrestricted {
            depth = depth.map(|value| value.clamp(0., 1.));
        }
        vk::Viewport {
            x: ox - sx,
            y: oy - sy,
            width: sx * 2.,
            height: sy * 2.,
            min_depth: depth[0],
            max_depth: depth[1],
        }
    };
    let v = viewport;
    let limits = &profile.limits;
    if [v.x, v.y, v.width, v.height, v.min_depth, v.max_depth]
        .iter()
        .any(|value| !value.is_finite())
        || v.width <= 0.
        || v.height == 0.
        || v.width > limits.max_viewport_dimensions[0] as f32
        || v.height.abs() > limits.max_viewport_dimensions[1] as f32
        || v.x < limits.viewport_bounds_range[0]
        || v.x + v.width > limits.viewport_bounds_range[1]
        || v.y.min(v.y + v.height) < limits.viewport_bounds_range[0]
        || v.y.max(v.y + v.height) > limits.viewport_bounds_range[1]
        || (!profile.depth_range_unrestricted
            && (!(0. ..=1.).contains(&v.min_depth) || !(0. ..=1.).contains(&v.max_depth)))
    {
        return Err(unsupported(
            0x0a00,
            r[0x0a00 / 4],
            "viewport exceeds enabled native bounds",
        ));
    }
    Ok(viewport)
}
/// Scissors follow the current viewport control and their own register bounds;
/// the framebuffer clamps SurfaceClip separately, as Eden GetScissorState.
/// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/vk_rasterizer.cpp
pub(super) fn scissor(r: &Regs) -> vk::Rect2D {
    if r[0x192c / 4] == 0 {
        let [width, height] = surface_extent(r).map(|v| v.max(1));
        let x = r[0x0ff4 / 4] & 0xffff;
        let mut y = (r[0x0ff8 / 4] & 0xffff) as i32;
        if r[0x13ac / 4] & 1 != 0 {
            y = (r[0x0ff8 / 4] >> 16) as i32 - y - height as i32;
        }
        return vk::Rect2D {
            offset: vk::Offset2D {
                x: x as i32,
                y: y.max(0),
            },
            extent: vk::Extent2D { width, height },
        };
    }
    scissor_state(r)
}
fn scissor_state(r: &Regs) -> vk::Rect2D {
    if r[0x0e00 / 4] == 0 {
        return vk::Rect2D {
            offset: Default::default(),
            extent: vk::Extent2D {
                width: i32::MAX as u32,
                height: i32::MAX as u32,
            },
        };
    }
    let sx = r[0x0e04 / 4];
    let sy = r[0x0e08 / 4];
    let x = sx & 0xffff;
    let end_x = sx >> 16;
    let (y, end_y) = if r[0x13ac / 4] & 1 != 0 {
        let height = (r[0x0ff8 / 4] >> 16) as i32;
        (
            (height - (sy >> 16) as i32).max(0) as u32,
            (height - (sy & 0xffff) as i32).max(0) as u32,
        )
    } else {
        (sy & 0xffff, sy >> 16)
    };
    vk::Rect2D {
        offset: vk::Offset2D {
            x: x as i32,
            y: y as i32,
        },
        extent: vk::Extent2D {
            width: end_x.saturating_sub(x),
            height: end_y.saturating_sub(y),
        },
    }
}

/// Clear uses GetScissorState directly, independently of viewport enable, and
/// clamps it to the current framebuffer's actual render area, as Eden Clear.
/// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/vk_rasterizer.cpp#L400-L500
pub(super) fn clear_rect(r: &Regs, area: vk::Rect2D) -> Result<vk::Rect2D, VulkanError> {
    let control = r[0x10f8 / 4];
    if control & !0x1111 != 0 {
        return Err(unsupported(0x10f8, control, "invalid clear control"));
    }
    let mut rect = if control & 0x100 != 0 {
        scissor_state(r)
    } else {
        area
    };
    let clamp = |offset: &mut i32, extent: &mut u32, limit: u32| {
        if *offset < 0 {
            *extent = extent.saturating_sub(offset.unsigned_abs());
            *offset = 0;
        }
        *offset = (*offset).min(limit as i32);
        *extent = (*extent).min(limit.saturating_sub(*offset as u32));
    };
    clamp(
        &mut rect.offset.x,
        &mut rect.extent.width,
        area.extent.width,
    );
    clamp(
        &mut rect.offset.y,
        &mut rect.extent.height,
        area.extent.height,
    );
    Ok(rect)
}

/// Unsupported effects are checked at consumption; ordinary register admission
/// stays a raw store. These checks do not build another draw description.
pub(super) fn validate_draw(state: &super::MaxwellThreeDState) -> Result<(), VulkanError> {
    let r = state.register_words();
    if r[0x1618 / 4] & ((1 << 24) | (3 << 29)) != 0 {
        return Err(unsupported(
            0x1618,
            r[0x1618 / 4],
            "primitive-ID continuation and split modes are unsupported",
        ));
    }
    Ok(())
}

pub(super) fn validate_graphics(
    state: &super::MaxwellThreeDState,
    draw: &super::MaxwellThreeDDrawState,
) -> Result<(), VulkanError> {
    let r = state.register_words();
    if r[0x121c / 4] & 15 > 8 {
        return Err(unsupported(
            0x121c,
            r[0x121c / 4],
            "color target count exceeds eight slots",
        ));
    }
    if r[0x1538 / 4] > 1 {
        return Err(unsupported(0x1538, r[0x1538 / 4], "invalid Zeta enable"));
    }
    if r[0x1948 / 4] > 1 {
        return Err(unsupported(
            0x1948,
            r[0x1948 / 4],
            "unsupported topology control",
        ));
    }
    if r[0x121c / 4] & 15 != 0 {
        if r[0x0dd0 / 4] & 3 != 0 {
            return Err(unsupported(
                0x0dd0,
                r[0x0dd0 / 4],
                "iterated blending is unsupported",
            ));
        }
        if r[0x19c4 / 4] == 1 {
            return Err(unsupported(
                0x19c8,
                r[0x19c8 / 4],
                "enabled color logic operations are unsupported",
            ));
        }
    }
    for (method, mask, reason) in [
        (
            0x030c,
            1,
            "conditional constant-buffer loading is unsupported",
        ),
        (0x02e4, 1, "stream output is unsupported"),
        (0x1688, 1, "two-sided lighting is unsupported"),
        (0x13a8, 0x11111111, "pixel output saturation is unsupported"),
        (
            0x07a0,
            u32::MAX,
            "shader local-memory allocation is unsupported",
        ),
        (0x0d9c, 1, "color reduction is unsupported"),
        (0x0f1c, 1, "post-Z pixel shader imask is unsupported"),
        (0x15b4, 1, "CSAA is unsupported"),
        (
            0x153c,
            0x11,
            "alpha-to-coverage/alpha-to-one is unsupported",
        ),
        (0x11f8, 1, "coverage-to-color is unsupported"),
        (
            0x16b4,
            u32::MAX,
            "alpha-to-coverage override is unsupported",
        ),
        (0x0fb4, 1, "target independent rasterization is unsupported"),
        (0x194c, 1, "window clip is unsupported"),
        (0x197c, 1, "clip-ID testing is unsupported"),
        (0x1924, 1, "integer viewport pixel centers are unsupported"),
        (0x1148, 1, "conservative rasterization is unsupported"),
        (0x19bc, 1, "depth-bounds testing is unsupported"),
    ] {
        if r[method / 4] & mask != 0 {
            return Err(unsupported(method as u32, r[method / 4], reason));
        }
    }
    if r[0x12d4 / 4] == 0x1d00 {
        return Err(unsupported(
            0x12d4,
            r[0x12d4 / 4],
            "fixed-function flat shading is unsupported",
        ));
    }
    if r[0x1138 / 4] == 1 && r[0x12ec / 4] == 1 {
        return Err(unsupported(
            0x1138,
            1,
            "post-PS initial coverage with alpha test is unsupported",
        ));
    }
    if r[0x113c / 4] == 1 {
        return Err(unsupported(
            0x113c,
            1,
            "fill-all via triangle is unsupported",
        ));
    }
    if r[0x0754 / 4] != 1 {
        return Err(unsupported(
            0x0754,
            r[0x0754 / 4],
            "hybrid antialiasing requires single-pass per-fragment coverage",
        ));
    }
    if r[0x1224 / 4] != 0
        && state
            .shader_execution()
            .pixel_shader_interlock_control()
            .value()
            .is_some_and(|value| value.conflict_detection_enabled())
    {
        return Err(unsupported(
            0x1224,
            r[0x1224 / 4],
            "pixel shader interlock is unsupported",
        ));
    }
    boolean(r, 0x12ec)?;
    attachment_samples(r[0x15d0 / 4])?;
    if r[0x1538 / 4] != 0 && r[0x0fe8 / 4] > 31 {
        return Err(unsupported(
            0x0fe8,
            r[0x0fe8 / 4],
            "invalid depth format word",
        ));
    }
    validate_raster(r, state.generated_primitive(draw.topology), draw.topology)?;
    let four = r[0x15d0 / 4] == 2;
    for group in 0..4 {
        let method = 0x11e0 + group * 4;
        let expected = if four { 0xeaa26e26 } else { 0x88888888 };
        if r[method / 4] != expected {
            return Err(unsupported(
                method as u32,
                r[method / 4],
                "nonstandard sample locations are unsupported",
            ));
        }
    }
    if four {
        if r[0x1534 / 4] != 1 || r[0x0fa4 / 4] & 0x10 != 0 {
            return Err(unsupported(
                0x0fa4,
                r[0x0fa4 / 4],
                "four-sample rasterization requires enabled AA and no target masking",
            ));
        }
        if r[0x0fa4 / 4] & 1 != 0 {
            for method in [0x0fbc, 0x0fc0, 0x0fc4, 0x0fc8] {
                if r[method / 4] & 15 != 15 {
                    return Err(unsupported(
                        method as u32,
                        r[method / 4],
                        "non-full four-sample quadrant mask is unsupported",
                    ));
                }
            }
        }
    }
    Ok(())
}
