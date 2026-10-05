//! Consumed polygon facing, mode and line state. Host-independent framebuffer
//! convention; do not compensate for the viewport Y sign again in the backend.
use super::super::threed::{MaxwellThreeDCullFace, MaxwellThreeDFrontFace};
use super::*;
use MaxwellThreeDFixedFunctionRegister as R;
use MaxwellThreeDFixedFunctionValue as V;
use nixe_gpu::{CullMode, FrontFace};

#[derive(Clone, Copy)]
pub(super) struct DrawRasterState {
    pub front_face: FrontFace,
    pub cull_mode: CullMode,
    pub triangles: TriangleRasterization,
}

pub(super) fn draw_state(
    state: &MaxwellThreeDState,
) -> Result<DrawRasterState, MaxwellLoweringError> {
    if state.fixed_function().register(R::RasterEnable).value() == Some(&V::Boolean(false)) {
        return Err(MaxwellLoweringError::UnsupportedPolygonRasterization(
            "rasterizer discard",
        ));
    }
    let (front_face, cull_mode) = face_state(state)?;
    let triangles = if state.generated_primitive()
        == Some(super::super::threed::state::GeneratedPrimitive::Triangles)
        && cull_mode != CullMode::FrontAndBack
    {
        let mode = |register, name| match state.fixed_function().register(register).value() {
            Some(V::PolygonMode(mode)) => Ok(*mode),
            None => Err(MaxwellLoweringError::IncompleteDraw(name)),
            _ => Err(wrong_type()),
        };
        let selected = match cull_mode {
            CullMode::Back => mode(R::FrontPolygonMode, "SET_FRONT_POLYGON_MODE")?,
            CullMode::Front => mode(R::BackPolygonMode, "SET_BACK_POLYGON_MODE")?,
            CullMode::None => {
                let front = mode(R::FrontPolygonMode, "SET_FRONT_POLYGON_MODE")?;
                let back = mode(R::BackPolygonMode, "SET_BACK_POLYGON_MODE")?;
                if front != back {
                    return Err(MaxwellLoweringError::UnsupportedPolygonRasterization(
                        "different front/back polygon modes without face culling",
                    ));
                }
                front
            }
            CullMode::FrontAndBack => unreachable!(),
        };
        let offset = match selected {
            MaxwellThreeDPolygonMode::Fill => R::PolygonOffsetFillEnable,
            MaxwellThreeDPolygonMode::Line => R::PolygonOffsetLineEnable,
            MaxwellThreeDPolygonMode::Point => {
                return Err(MaxwellLoweringError::UnsupportedPolygonRasterization(
                    "point polygon mode",
                ));
            }
        };
        if state.fixed_function().register(offset).value() == Some(&V::Boolean(true)) {
            return Err(MaxwellLoweringError::UnsupportedPolygonRasterization(
                "depth bias for the consumed polygon mode",
            ));
        }
        match selected {
            MaxwellThreeDPolygonMode::Fill => {
                // Point/line/fill smoothing and depth bias are independent.
                // https://github.com/devkitPro/deko3d/blob/master/source/maxwell/gpu_3d_state.cpp#L61-L66
                if state.raster().polygon_smooth_enable().value() == Some(&true) {
                    return Err(MaxwellLoweringError::UnsupportedPolygonSmoothSemantics);
                }
                if state.raster().polygon_stipple_enable().value() == Some(&true) {
                    return Err(MaxwellLoweringError::UnsupportedPolygonStippleSemantics);
                }
                TriangleRasterization::Fill
            }
            MaxwellThreeDPolygonMode::Line => smooth_wireframe(state)?,
            MaxwellThreeDPolygonMode::Point => unreachable!(),
        }
    } else {
        // No polygon fragments; keep the draw and pre-raster shader execution.
        TriangleRasterization::Fill
    };
    Ok(DrawRasterState {
        front_face,
        cull_mode,
        triangles,
    })
}

fn smooth_wireframe(
    state: &MaxwellThreeDState,
) -> Result<TriangleRasterization, MaxwellLoweringError> {
    if state
        .raster()
        .fill_via_triangle()
        .value()
        .is_some_and(|mode| *mode != MaxwellThreeDFillViaTriangleMode::Disabled)
    {
        return Err(MaxwellLoweringError::UnsupportedPolygonRasterization(
            "non-fill polygon mode with fill-via-triangle",
        ));
    }
    match state.line().anti_aliased_line_enable().value() {
        Some(MaxwellThreeDAntiAliasedLineEnable::Enabled) => {}
        Some(MaxwellThreeDAntiAliasedLineEnable::Disabled) => {
            return Err(MaxwellLoweringError::UnsupportedPolygonRasterization(
                "aliased polygon-line coverage",
            ));
        }
        None => {
            return Err(MaxwellLoweringError::IncompleteDraw(
                "SET_ANTI_ALIASED_LINE",
            ));
        }
    }
    if state
        .line()
        .stipple_enable()
        .value()
        .copied()
        .ok_or(MaxwellLoweringError::IncompleteDraw("SET_LINE_STIPPLE"))?
    {
        return Err(MaxwellLoweringError::UnsupportedPolygonRasterization(
            "stippled polygon lines",
        ));
    }
    match state.line().polygon_clip_generated_edge().value() {
        Some(MaxwellThreeDPolygonClipGeneratedEdge::DrawLine) => {}
        Some(MaxwellThreeDPolygonClipGeneratedEdge::DoNotDrawLine) => {
            return Err(MaxwellLoweringError::UnsupportedPolygonClipGeneratedEdgeSemantics);
        }
        None => {
            return Err(MaxwellLoweringError::IncompleteDraw(
                "SET_POLYGON_CLIP_GENERATED_EDGE",
            ));
        }
    }
    match state.raster().edge_flag().value() {
        Some(MaxwellThreeDEdgeFlag::Enabled) => {}
        Some(MaxwellThreeDEdgeFlag::Disabled) => {
            return Err(MaxwellLoweringError::UnsupportedEdgeFlagSemantics(
                MaxwellThreeDEdgeFlag::Disabled,
            ));
        }
        None => return Err(MaxwellLoweringError::IncompleteDraw("SET_EDGE_FLAG")),
    }
    let line = smooth_line(state)?;
    Ok(TriangleRasterization::Wireframe {
        width_bits: line.width_bits,
        smooth: line.smooth,
    })
}

pub(super) fn smooth_line(
    state: &MaxwellThreeDState,
) -> Result<nixe_gpu::LineRasterization, MaxwellLoweringError> {
    if state.line().stipple_enable().value() == Some(&true) {
        return Err(MaxwellLoweringError::UnsupportedPolygonRasterization(
            "stippled smooth lines",
        ));
    }
    // Smooth lines consume the smooth width regardless of the aliased-width
    // selector. Do not approximate aliased coverage with rectangular lines.
    // https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h
    // https://docs.vulkan.org/spec/latest/chapters/primsrast.html#primsrast-lines-smooth
    let Some(V::FloatBits(width_bits)) = state
        .fixed_function()
        .register(R::LineWidth)
        .value()
        .copied()
    else {
        return Err(MaxwellLoweringError::IncompleteDraw("SET_LINE_WIDTH_FLOAT"));
    };
    Ok(nixe_gpu::LineRasterization {
        width_bits: width_bits.get(),
        smooth: true,
    })
}

pub(super) fn face_state(
    state: &MaxwellThreeDState,
) -> Result<(FrontFace, CullMode), MaxwellLoweringError> {
    if state.generated_primitive()
        != Some(super::super::threed::state::GeneratedPrimitive::Triangles)
    {
        return Ok((FrontFace::CounterClockwise, CullMode::None));
    }
    let required = |register, name| {
        state
            .fixed_function()
            .register(register)
            .value()
            .copied()
            .ok_or(MaxwellLoweringError::IncompleteDraw(name))
    };
    // FLIP_Y reverses polygon facing; it does not negate viewport coordinates.
    // Lower-left origin changes window coordinates in viewport/scissor
    // lowering. Keep it distinct from the facing-only FLIP_Y bit.
    // Encodings: https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L2599-L2605
    // https://github.com/yuzu-emu-mirror/yuzu-mainline/blob/310c1f50beb77fc5c6f9075029973161d4e51a4a/src/video_core/renderer_vulkan/vk_rasterizer.cpp#L1338-L1349
    let V::Mask(origin) = required(R::WindowOrigin, "SET_WINDOW_ORIGIN")? else {
        return Err(wrong_type());
    };
    if origin & !0x11 != 0 {
        return Err(MaxwellLoweringError::UnsupportedWindowOrigin(origin));
    }
    let V::FrontFace(face) = required(R::FrontFace, "SET_FRONT_FACE")? else {
        return Err(wrong_type());
    };
    let face = match (face, origin & 0x10 != 0) {
        (MaxwellThreeDFrontFace::Clockwise, false)
        | (MaxwellThreeDFrontFace::CounterClockwise, true) => FrontFace::Clockwise,
        (MaxwellThreeDFrontFace::CounterClockwise, false)
        | (MaxwellThreeDFrontFace::Clockwise, true) => FrontFace::CounterClockwise,
    };
    let V::Boolean(enabled) = required(R::CullEnable, "SET_CULL_FACE_ENABLE")? else {
        return Err(wrong_type());
    };
    let mode = if enabled {
        let V::CullFace(mode) = required(R::CullFace, "SET_CULL_FACE")? else {
            return Err(wrong_type());
        };
        match mode {
            MaxwellThreeDCullFace::Front => CullMode::Front,
            MaxwellThreeDCullFace::Back => CullMode::Back,
            MaxwellThreeDCullFace::FrontAndBack => CullMode::FrontAndBack,
        }
    } else {
        CullMode::None
    };
    Ok((face, mode))
}

fn wrong_type() -> MaxwellLoweringError {
    MaxwellLoweringError::ContradictoryState {
        reason: "polygon raster register has an inconsistent typed value",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::tests::{program_three_d, three_d_channel};

    #[test]
    fn smooth_direct_lines_preserve_width_without_polygon_edge_state() {
        let mut channel = three_d_channel();
        // Direct lines do not consume polygon clip-edge, edge-flag or mode state.
        program_three_d(&mut channel, 0x1618, 3);
        program_three_d(&mut channel, 0x1570, 1);
        program_three_d(&mut channel, 0x166c, 0);
        for width in [1.0_f32, 4.0, 16.0] {
            program_three_d(&mut channel, 0x13b0, width.to_bits());
            let line = super::smooth_line(channel.three_d()).unwrap();
            assert_eq!(line.width_bits, width.to_bits());
            assert!(line.smooth);
        }
        program_three_d(&mut channel, 0x166c, 1);
        assert!(super::smooth_line(channel.three_d()).is_err());
    }

    fn wireframe_channel() -> crate::MaxwellGpuChannel {
        let mut channel = three_d_channel();
        for (method, value) in [
            (0x1618, 14),
            (0x320, 0x201),
            (0x1918, 1),
            (0x191c, 0x901),
            (0x1920, 0x405),
            (0x0dac, 0x1b01),
            (0x0db0, 0x1b01),
            (0x1570, 1),
            (0x15e4, 1),
            (0x13b0, 4_f32.to_bits()),
            (0x13b4, 99_f32.to_bits()),
            (0x0db4, 1),
            (0x1658, 1),
            (0x0dc4, 0),
        ] {
            program_three_d(&mut channel, method, value);
        }
        channel
    }

    #[test]
    fn depth_bias_parameters_are_consumed_only_for_the_enabled_polygon_mode() {
        let mut channel = wireframe_channel();
        for method in [0x156c, 0x15bc, 0x187c] {
            program_three_d(&mut channel, method, 2_f32.to_bits());
        }
        assert!(draw_state(channel.three_d()).is_ok());
        program_three_d(&mut channel, 0x0dc8, 1);
        assert!(draw_state(channel.three_d()).is_ok());
        program_three_d(&mut channel, 0x0dc4, 1);
        assert!(matches!(
            draw_state(channel.three_d()),
            Err(MaxwellLoweringError::UnsupportedPolygonRasterization(
                "depth bias for the consumed polygon mode"
            ))
        ));
        program_three_d(&mut channel, 0x0dc4, 0);
        assert!(draw_state(channel.three_d()).is_ok());
        program_three_d(&mut channel, 0x0dac, 0x1b02);
        assert!(matches!(
            draw_state(channel.three_d()),
            Err(MaxwellLoweringError::UnsupportedPolygonRasterization(
                "depth bias for the consumed polygon mode"
            ))
        ));
    }

    #[test]
    fn patch_wireframe_consumes_line_smoothing_and_smooth_width_only() {
        let mut channel = wireframe_channel();
        for selector in [0, 1] {
            program_three_d(&mut channel, 0x020c, selector);
            let raster = draw_state(channel.three_d()).unwrap();
            assert_eq!(raster.cull_mode, CullMode::Back);
            assert_eq!(raster.front_face, FrontFace::CounterClockwise);
            assert_eq!(
                raster.triangles,
                TriangleRasterization::Wireframe {
                    width_bits: 4_f32.to_bits(),
                    smooth: true
                }
            );
        }
        let shader_identity = channel.three_d().shader_state_identity();
        let draw_identity = channel.three_d().draw_state_identity();
        program_three_d(&mut channel, 0x13b0, 2_f32.to_bits());
        assert!(shader_identity.matches(channel.three_d()));
        assert!(!draw_identity.matches(channel.three_d()));
        assert_eq!(
            draw_state(channel.three_d()).unwrap().triangles,
            TriangleRasterization::Wireframe {
                width_bits: 2_f32.to_bits(),
                smooth: true
            }
        );
        // A stale smooth-polygon bit must become effective again in fill mode.
        program_three_d(&mut channel, 0x0dac, 0x1b02);
        assert!(matches!(
            draw_state(channel.three_d()),
            Err(MaxwellLoweringError::UnsupportedPolygonSmoothSemantics)
        ));
        program_three_d(&mut channel, 0x0db4, 0);
        assert_eq!(
            draw_state(channel.three_d()).unwrap().triangles,
            TriangleRasterization::Fill
        );
    }

    #[test]
    fn culled_face_modes_are_inactive_but_mixed_visible_modes_fail() {
        let mut channel = wireframe_channel();
        program_three_d(&mut channel, 0x0db0, 0x1b00);
        assert!(draw_state(channel.three_d()).is_ok());
        program_three_d(&mut channel, 0x1918, 0);
        assert!(matches!(
            draw_state(channel.three_d()),
            Err(MaxwellLoweringError::UnsupportedPolygonRasterization(_))
        ));
        program_three_d(&mut channel, 0x1918, 1);
        program_three_d(&mut channel, 0x1920, 0x404);
        assert!(matches!(
            draw_state(channel.three_d()),
            Err(MaxwellLoweringError::UnsupportedPolygonRasterization(
                "point polygon mode"
            ))
        ));
        program_three_d(&mut channel, 0x1920, 0x408);
        let raster = draw_state(channel.three_d()).unwrap();
        assert_eq!(raster.cull_mode, CullMode::FrontAndBack);
        assert_eq!(raster.triangles, TriangleRasterization::Fill);
    }

    #[test]
    fn active_unsupported_line_effects_are_not_silently_dropped() {
        for (method, value) in [
            (0x1570, 0),
            (0x166c, 1),
            (0x0f8c, 1),
            (0x15e4, 0),
            (0x0dc4, 1),
        ] {
            let mut channel = wireframe_channel();
            program_three_d(&mut channel, method, value);
            assert!(
                draw_state(channel.three_d()).is_err(),
                "method=0x{method:x}"
            );
        }
        let mut channel = wireframe_channel();
        // Inactive fill/point bias is not line bias.
        for method in [0x0dc0, 0x0dc8] {
            program_three_d(&mut channel, method, 1);
        }
        assert!(draw_state(channel.three_d()).is_ok());
    }

    #[test]
    fn line_reset_values_survive_shadow_replay_and_channel_recreation() {
        for method in [0x0f8c, 0x166c] {
            let mut channel = wireframe_channel();
            program_three_d(&mut channel, 0x124, 3);
            program_three_d(&mut channel, method, u32::MAX);
            let line = channel.three_d().line();
            assert_eq!(
                line.polygon_clip_generated_edge().value(),
                Some(&MaxwellThreeDPolygonClipGeneratedEdge::DrawLine)
            );
            assert_eq!(line.stipple_enable().value(), Some(&false));
            assert!(draw_state(channel.three_d()).is_ok());
        }
    }

    #[test]
    fn consumed_faces_use_initialized_context_and_preserve_all_cull_modes() {
        let mut channel = three_d_channel();
        program_three_d(&mut channel, 0x1618, 4);
        assert_eq!(
            face_state(channel.three_d()).unwrap(),
            (FrontFace::Clockwise, CullMode::None)
        );
        program_three_d(&mut channel, 0x191c, 0x901);
        assert_eq!(
            face_state(channel.three_d()).unwrap(),
            (FrontFace::CounterClockwise, CullMode::None)
        );
        program_three_d(&mut channel, 0x1918, 1);
        assert_eq!(
            face_state(channel.three_d()).unwrap(),
            (FrontFace::CounterClockwise, CullMode::Back)
        );
        for (raw, face) in [
            (0x900, FrontFace::Clockwise),
            (0x901, FrontFace::CounterClockwise),
        ] {
            program_three_d(&mut channel, 0x191c, raw);
            for (raw, mode) in [
                (0x404, CullMode::Front),
                (0x405, CullMode::Back),
                (0x408, CullMode::FrontAndBack),
            ] {
                program_three_d(&mut channel, 0x1920, raw);
                assert_eq!(face_state(channel.three_d()).unwrap(), (face, mode));
            }
        }
        program_three_d(&mut channel, 0x13ac, 0x10);
        assert_eq!(
            face_state(channel.three_d()).unwrap(),
            (FrontFace::Clockwise, CullMode::FrontAndBack)
        );
        program_three_d(&mut channel, 0x191c, 0x900);
        assert_eq!(
            face_state(channel.three_d()).unwrap(),
            (FrontFace::CounterClockwise, CullMode::FrontAndBack)
        );
        program_three_d(&mut channel, 0x13ac, 0x1);
        assert_eq!(
            face_state(channel.three_d()).unwrap(),
            (FrontFace::Clockwise, CullMode::FrontAndBack)
        );
        // Direct lines/points do not consume polygon facing, including stale
        // unsupported polygon state left by the previous draw.
        for topology in [0, 1, 3] {
            program_three_d(&mut channel, 0x1618, topology);
            assert_eq!(
                face_state(channel.three_d()).unwrap(),
                (FrontFace::CounterClockwise, CullMode::None)
            );
        }
        // Patches consume the generated primitive, not their input topology.
        program_three_d(&mut channel, 0x13ac, 0);
        program_three_d(&mut channel, 0x191c, 0x901);
        program_three_d(&mut channel, 0x1618, 14);
        program_three_d(&mut channel, 0x320, 0x201);
        assert_eq!(
            face_state(channel.three_d()).unwrap(),
            (FrontFace::CounterClockwise, CullMode::FrontAndBack)
        );
    }
}
