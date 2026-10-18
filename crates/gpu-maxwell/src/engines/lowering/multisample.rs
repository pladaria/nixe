use super::*;
pub(super) fn validate(state: &MaxwellThreeDState) -> Result<(), MaxwellLoweringError> {
    use MaxwellThreeDFixedFunctionRegister as R;
    use MaxwellThreeDFixedFunctionValue as V;
    let four = state.fixed_function().register(R::SampleMode).value()
        == Some(&V::SampleMode(
            super::super::threed::MaxwellThreeDSampleMode::Samples2x2,
        ));
    for (group, register) in state.coverage().sample_locations().iter().enumerate() {
        let Some(value) = register.value().copied() else {
            if four {
                return Err(MaxwellLoweringError::IncompleteDraw(
                    "SET_ANTI_ALIAS_SAMPLE_POSITIONS",
                ));
            }
            continue;
        };
        // deko3d's default 4x pattern is the standard Vulkan four-sample pattern:
        // (6,2), (14,6), (2,10), (10,14), in sixteenths of a pixel.
        // https://github.com/devkitPro/deko3d/blob/master/source/maxwell/gpu_3d_ms.cpp
        // https://docs.vulkan.org/spec/latest/chapters/primsrast.html#primsrast-standard-sample-locations
        if if four {
            value.raw() != 0xeaa2_6e26
        } else {
            !value.is_centered()
        } {
            return Err(MaxwellLoweringError::UnsupportedSampleLocationsSemantics {
                group: group as u8,
                value,
            });
        }
    }
    if !four {
        return Ok(());
    }
    if state.fixed_function().register(R::AntiAliasEnable).value() != Some(&V::Boolean(true)) {
        return Err(MaxwellLoweringError::UnsupportedMultisampleState(
            "four-sample targets require multisample rasterization enabled",
        ));
    }
    let control = state
        .fixed_function()
        .register(R::SampleMaskControl)
        .value();
    if matches!(control, Some(V::Mask(value)) if value & 0x10 != 0) {
        return Err(MaxwellLoweringError::UnsupportedMultisampleState(
            "color-target sample masking",
        ));
    }
    // Full raster masks are state-independent whether masking is enabled or
    // disabled; no reset value for SET_SAMPLE_MASK is inferred here.
    if !matches!(control, Some(V::Mask(value)) if value & 1 == 0) {
        for register in [
            R::SampleMask0,
            R::SampleMask1,
            R::SampleMask2,
            R::SampleMask3,
        ] {
            if !matches!(state.fixed_function().register(register).value(), Some(V::Mask(mask)) if mask & 0xf == 0xf)
            {
                return Err(MaxwellLoweringError::UnsupportedMultisampleState(
                    "missing or non-full per-quadrant four-sample mask",
                ));
            }
        }
    }
    Ok(())
}
