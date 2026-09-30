//! Maxwell's domain-space tessellation register, not host front-face state.
use nixe_gpu::{
    TessellationDomain as Domain, TessellationMode, TessellationOutput as Output,
    TessellationSpacing as Spacing, TessellationWinding as Winding,
};

/// Retain every written bit and diagnose unsupported/malformed combinations at
/// the consuming patch draw. This also preserves MME shadow replay semantics.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct MaxwellThreeDTessellationMode(u32);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaxwellTessellationModeError {
    ReservedBits,
    ReservedDomain,
    ReservedSpacing,
    UnsupportedIsolineConnectivity,
}

impl MaxwellThreeDTessellationMode {
    pub const fn new(raw: u32) -> Self {
        Self(raw)
    }
    pub const fn raw(self) -> u32 {
        self.0
    }

    pub fn lower(self) -> Result<TessellationMode, MaxwellTessellationModeError> {
        // deko3d's producer/compiler ABI distinguishes connectedness and winding.
        // In particular 0x201 means connected CCW triangles, NOT the CW label in
        // clb197's dump enum. Do not apply a host viewport flip at this boundary.
        // https://github.com/devkitPro/deko3d/blob/master/source/maxwell/engine_3d.def#L41-L59
        // Confirmed by the compiler producing param_c8 from GLSL winding:
        // https://github.com/devkitPro/uam/blob/master/source/compiler_iface.cpp#L491-L528
        if self.0 & !0x333 != 0 {
            return Err(MaxwellTessellationModeError::ReservedBits);
        }
        let domain = match self.0 & 3 {
            0 => Domain::Isolines,
            1 => Domain::Triangles,
            2 => Domain::Quads,
            _ => return Err(MaxwellTessellationModeError::ReservedDomain),
        };
        let spacing = match (self.0 >> 4) & 3 {
            0 => Spacing::Equal,
            1 => Spacing::FractionalOdd,
            2 => Spacing::FractionalEven,
            _ => return Err(MaxwellTessellationModeError::ReservedSpacing),
        };
        let flags = (self.0 >> 8) & 3;
        let output = match domain {
            Domain::Isolines => match flags {
                0 => Output::Points,
                1 => Output::Lines,
                // The encoding fits the field, but no verified isoline meaning
                // is available for the triangle-connected flag.
                _ => return Err(MaxwellTessellationModeError::UnsupportedIsolineConnectivity),
            },
            Domain::Triangles | Domain::Quads if flags & 2 == 0 => Output::Points,
            _ => Output::Triangles(if flags & 1 == 0 {
                Winding::CounterClockwise
            } else {
                Winding::Clockwise
            }),
        };
        Ok(TessellationMode {
            domain,
            spacing,
            output,
        })
    }
}

pub(in crate::engines) fn draw_state(
    state: &super::MaxwellThreeDState,
) -> Result<Option<nixe_gpu::TessellationState>, super::MaxwellThreeDLoweringError> {
    use super::{
        MaxwellThreeDLoweringError as Error, MaxwellThreeDShaderStage as Stage,
        MaxwellThreeDTessellationLod,
    };
    use nixe_gpu::{TessellationControl, TessellationState};
    let bindings = state.shader_bindings();
    let enabled = |stage| {
        bindings
            .pipeline()
            .iter()
            .any(|p| p.enabled().value() == Some(&true) && p.stage().value() == Some(&stage))
    };
    let patches = state
        .vertex_input()
        .primitive()
        .active_begin()
        .is_some_and(|b| b.topology() == 14);
    if !patches {
        if enabled(Stage::TessellationInit) || enabled(Stage::Tessellation) {
            return Err(Error::TessellationStageTopology);
        }
        return Ok(None);
    }
    let size = state
        .vertex_input()
        .primitive()
        .patch_size()
        .value()
        .copied()
        .ok_or(Error::IncompleteDraw("SET_PATCH"))?;
    if size.control_points() == 0 {
        return Err(Error::InvalidPatchSize(size));
    }
    let register = bindings.tessellation_mode();
    let value = register
        .value()
        .copied()
        .ok_or(Error::IncompleteDraw("SET_TESSELLATION_PARAMETERS"))?;
    let mode = value.lower().map_err(|reason| Error::TessellationMode {
        value,
        source: register.source(),
        reason,
    })?;
    if !enabled(Stage::Tessellation) {
        return Err(Error::IncompleteDraw(
            "patch draw requires tessellation evaluation shader",
        ));
    }
    if enabled(Stage::Geometry) {
        return Err(Error::UnsupportedShaderStage(Stage::Geometry));
    }
    if !enabled(Stage::Vertex) {
        return Err(Error::IncompleteDraw("patch draw requires a vertex shader"));
    }
    let control = if enabled(Stage::TessellationInit) {
        TessellationControl::Shader
    } else {
        // Only domain-consumed levels require programming. Unused array lanes are
        // canonical padding, not substituted guest levels. Retain all programmed
        // bits, including negative zero, infinities and NaN payloads.
        let used = match mode.domain {
            Domain::Isolines => [true, true, false, false, false, false],
            Domain::Triangles => [true, true, true, false, true, false],
            Domain::Quads => [true; 6],
        };
        let mut levels = [0; 6];
        let mut defined = 0;
        for (i, lane) in levels.iter_mut().enumerate() {
            match bindings
                .tessellation_lod(MaxwellThreeDTessellationLod::from_index(i as u8))
                .value()
            {
                Some(bits) => {
                    *lane = *bits;
                    defined |= 1 << i;
                }
                None if used[i] => {
                    return Err(Error::IncompleteDraw(
                        "default tessellation level consumed without a control shader",
                    ));
                }
                None => {}
            }
        }
        TessellationControl::DefaultLevels {
            outer: levels[..4].try_into().unwrap(),
            inner: levels[4..].try_into().unwrap(),
            defined,
        }
    };
    Ok(Some(TessellationState {
        mode,
        input_control_points: size.control_points(),
        control,
    }))
}

pub(in crate::engines) fn validate_default_level_inputs(
    control: nixe_gpu::TessellationControl,
    ir: &nixe_gpu::ShaderIr,
) -> Result<(), super::MaxwellThreeDLoweringError> {
    let nixe_gpu::TessellationControl::DefaultLevels { defined, .. } = control else {
        return Ok(());
    };
    if ir.stage() != nixe_gpu::ShaderStage::TessellationEvaluation {
        return Ok(());
    }
    for input in ir.inputs() {
        let bit = match input.location() {
            nixe_gpu::ShaderIoLocation::TessLevelOuter => input.component(),
            nixe_gpu::ShaderIoLocation::TessLevelInner => 4 + input.component(),
            _ => continue,
        };
        if defined & (1 << bit) == 0 {
            return Err(super::MaxwellThreeDLoweringError::IncompleteDraw(
                "default tessellation level consumed by evaluation shader",
            ));
        }
    }
    Ok(())
}
