//! Effective coverage state for the currently supported sample patterns.
use super::*;

pub(super) fn validate_shader_mask(
    state: &MaxwellThreeDState,
    shaders: &MaxwellThreeDTranslatedShaders,
    cache: &MaxwellLoweringCache,
) -> Result<(), MaxwellLoweringError> {
    let pre_ps_initial_coverage =
        state.coverage().post_ps_initial_coverage().value() == Some(&true);
    if state.ps_output_sample_mask_effective() != Some(true) && !pre_ps_initial_coverage {
        return Ok(());
    }
    let fragment = shaders
        .shaders()
        .iter()
        .find(|shader| shader.stage() == ShaderStage::Fragment)
        .ok_or(MaxwellLoweringError::InvalidTranslatedShaders)?;
    let record = cache
        .shader_translations
        .get(fragment.cache_fingerprint)
        .ok_or(MaxwellLoweringError::InvalidTranslatedShaders)?;
    // SET_PS_OUTPUT_SAMPLE_MASK_USAGE enables consumption, not shader export.
    // The SPH OMAP_SAMPLE_MASK bit independently declares that export and is
    // retained in the translated interface. A color-only shader has no mask
    // to apply, even when usage is enabled (as in deko3d's MSAA configuration).
    // https://github.com/NVIDIA/open-gpu-doc/blob/master/classes/3d/clb197.h
    // https://download.nvidia.com/open-gpu-doc/Shader-Program-Header/1/Shader-Program-Header.html
    if record
        .module
        .ir()
        .ir()
        .outputs()
        .iter()
        .any(|output| output.location() == nixe_gpu::ShaderIoLocation::SampleMask)
    {
        return Err(if pre_ps_initial_coverage {
            MaxwellLoweringError::UnsupportedPostPsInitialCoverageSemantics
        } else {
            MaxwellLoweringError::UnsupportedPsOutputSampleMaskSemantics
        });
    }
    // Without a shader mask export or pixel-kill, pre/post-PS coverage is
    // identical. Pixel-kill remains an explicit SPH translation boundary;
    // alpha-test discard is validated separately before this call.
    // https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L1817-L1820
    // https://github.com/eden-emulator/mirror/blob/d16735f5b618942136d6ab53466e3be0a382c30a/src/video_core/engines/maxwell_3d.h#L2793
    Ok(())
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::tests::{program_three_d, three_d_channel};

    #[test]
    fn shader_mask_usage_requires_an_actual_fragment_export() {
        use nixe_gpu::{
            ShaderBackendModule, ShaderInstruction, ShaderInterfaceElement, ShaderIoLocation,
            ShaderIr, ShaderOperation, ShaderPredicate, ShaderRegister, ShaderScalarType,
            ShaderSourceLocation, VerifiedShaderIr,
        };
        let mut channel = three_d_channel();
        let shaders = MaxwellThreeDTranslatedShaders::new(
            vec![MaxwellThreeDTranslatedShader::new(
                ShaderStage::Fragment,
                ShaderId::new(2),
                2,
                None,
                0,
            )],
            vec![],
        )
        .unwrap();
        let mut cache = MaxwellLoweringCache::default();
        cache.seed_test_shader_translations(&shaders);
        program_three_d(&mut channel, 0x1534, 1);
        program_three_d(&mut channel, 0x0300, 3);
        validate_shader_mask(channel.three_d(), &shaders, &cache).unwrap();
        program_three_d(&mut channel, 0x1138, 1);
        validate_shader_mask(channel.three_d(), &shaders, &cache).unwrap();
        program_three_d(&mut channel, 0x1138, 0);

        let scalar_type = ShaderScalarType::Unsigned32;
        let ir = ShaderIr::new(
            ShaderStage::Fragment,
            vec![],
            vec![
                ShaderInterfaceElement::new(ShaderIoLocation::SampleMask, 0, scalar_type, None)
                    .unwrap(),
            ],
            vec![],
            [
                ShaderOperation::MoveImmediate32 {
                    destination: ShaderRegister::new(0),
                    bits: 1,
                    scalar_type,
                },
                ShaderOperation::StoreOutput {
                    sources: vec![ShaderRegister::new(0)].into(),
                    location: ShaderIoLocation::SampleMask,
                    first_component: 0,
                    scalar_type,
                },
                ShaderOperation::Exit,
            ]
            .into_iter()
            .map(|op| {
                ShaderInstruction::new(ShaderSourceLocation::new(0), ShaderPredicate::Always, op)
            })
            .collect(),
        );
        cache.shader_translations.get_mut(2).unwrap().module =
            ShaderBackendModule::new(VerifiedShaderIr::verify(ir).unwrap());
        assert_eq!(
            validate_shader_mask(channel.three_d(), &shaders, &cache),
            Err(MaxwellLoweringError::UnsupportedPsOutputSampleMaskSemantics)
        );
        program_three_d(&mut channel, 0x1534, 0);
        validate_shader_mask(channel.three_d(), &shaders, &cache).unwrap();
        program_three_d(&mut channel, 0x0300, 1);
        assert_eq!(
            validate_shader_mask(channel.three_d(), &shaders, &cache),
            Err(MaxwellLoweringError::UnsupportedPsOutputSampleMaskSemantics)
        );
        program_three_d(&mut channel, 0x0300, 0);
        program_three_d(&mut channel, 0x1138, 1);
        assert_eq!(
            validate_shader_mask(channel.three_d(), &shaders, &cache),
            Err(MaxwellLoweringError::UnsupportedPostPsInitialCoverageSemantics)
        );
    }

    #[test]
    fn standard_four_sample_coverage_requires_matching_effective_state() {
        let mut channel = three_d_channel();
        program_three_d(&mut channel, 0x15d0, 2);
        assert!(validate(channel.three_d()).is_err());
        for group in 0..4 {
            program_three_d(&mut channel, 0x11e0 + group * 4, 0xeaa2_6e26);
            program_three_d(&mut channel, 0x0fbc + group * 4, 0xffff);
        }
        program_three_d(&mut channel, 0x1534, 1);
        validate(channel.three_d()).unwrap();
        for (method, bad, good) in [
            (0x1534, 0, 1),
            (0x11e8, 0x8888_8888, 0xeaa2_6e26),
            (0x0fc4, 0xfffe, 0xffff),
            (0x0fa4, 0x11, 1),
        ] {
            program_three_d(&mut channel, method, bad);
            assert!(validate(channel.three_d()).is_err(), "method={method:x}");
            program_three_d(&mut channel, method, good);
            validate(channel.three_d()).unwrap();
        }
        program_three_d(&mut channel, 0x0fa4, 0);
        program_three_d(&mut channel, 0x0fc4, 0);
        validate(channel.three_d()).unwrap(); // Explicitly disabled raster mask.
    }
}
