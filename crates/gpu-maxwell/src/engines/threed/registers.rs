//! Register admission and sparse derived-state decoding.
//!
//! Ordinary writes update one authoritative raw word and dirty metadata.
//! Execution methods retain ordered effects even when the argument is equal.
//! Reference mechanisms (independently implemented):
//! https://github.com/eden-emulator/mirror/blob/d16735f5b618942136d6ab53466e3be0a382c30a/src/video_core/engines/maxwell_3d.cpp
//! https://github.com/yangjieshao/Ryujinx/blob/2505a1abcd5f31e4364bfb7f9313546f1840e5e1/src/Ryujinx.Graphics.Gpu/Engine/Threed/ThreedClass.cs

use super::*;

const fn output_state(method: u32) -> bool {
    match method {
        0x0800..=0x09fc => (method - 0x0800) % 0x40 <= 0x20,
        0x0a00..=0x0d3c
        | 0x1360..=0x137c
        | 0x1a00..=0x1a1c
        | 0x1700..=0x177c
        | 0x0d80..=0x0d8c
        | 0x19e0..=0x19fc
        | 0x0fbc..=0x0fc8
        | 0x131c..=0x1328 => true,
        0x0e00..=0x0efc => (method - 0x0e00) % 0x10 <= 8,
        0x1e00..=0x1efc => (method - 0x1e00) % 0x20 <= 0x18,
        0x0ff4 | 0x0ff8 => true,
        0x037c | 0x07a4 | 0x07a8 | 0x0d6c | 0x0d70 | 0x0d90 | 0x0da0 | 0x0dac | 0x0db0 | 0x0db4
        | 0x0dc0 | 0x0dc4 | 0x0dc8 | 0x0df8 | 0x0dfc | 0x0f54 | 0x0f58 | 0x0f5c | 0x0f8c
        | 0x0f90 | 0x0f9c | 0x0fa0 | 0x0fa4 | 0x0fac | 0x0fb8 | 0x0fdc | 0x0fe0 | 0x0fe4
        | 0x0fe8 | 0x0fec | 0x0ff0 | 0x10f8 | 0x113c | 0x1140 | 0x1148 | 0x11f0 | 0x1208
        | 0x121c | 0x1220 | 0x1228 | 0x122c | 0x1230 | 0x12cc | 0x12d4 | 0x12e4 | 0x12e8
        | 0x12ec | 0x130c | 0x1310 | 0x1314 | 0x133c | 0x1340 | 0x1344 | 0x1348 | 0x134c
        | 0x1350 | 0x1358 | 0x135c | 0x1380 | 0x1384 | 0x1388 | 0x138c | 0x1390 | 0x1394
        | 0x1398 | 0x139c | 0x13a8 | 0x13ac | 0x13b0 | 0x13b4 | 0x1510 | 0x1520 | 0x1534
        | 0x1538 | 0x153c | 0x156c | 0x1570 | 0x1594 | 0x1598 | 0x159c | 0x15a0 | 0x15a4
        | 0x15b8 | 0x15bc | 0x15cc | 0x15d0 | 0x1658 | 0x166c | 0x1680 | 0x1684 | 0x1688
        | 0x168c | 0x179c | 0x187c | 0x1910 | 0x1918 | 0x191c | 0x1920 | 0x1924 | 0x192c
        | 0x193c | 0x1940 | 0x194c | 0x1950 | 0x197c | 0x19bc | 0x19c0 | 0x19c4 | 0x19c8
        | 0x19cc | 0x2600 => true,
        _ => false,
    }
}

pub(super) const fn passive_action(action: MethodAction) -> bool {
    // Exhaustive: every new action explicitly chooses execution or passive state.
    match action {
        MethodAction::NotifyAddress
        | MethodAction::Delay
        | MethodAction::VpcPerfKnob
        | MethodAction::InstrumentationHeader
        | MethodAction::InstrumentationData
        | MethodAction::IteratedBlend
        | MethodAction::IteratedBlendPassCount
        | MethodAction::SubtilingPerfKnobA
        | MethodAction::SubtilingPerfKnobB
        | MethodAction::TiledCacheEnable
        | MethodAction::TiledCacheTileSize
        | MethodAction::TiledCacheUnknownConfig(..)
        | MethodAction::ShaderWatermarks(..)
        | MethodAction::L1Configuration
        | MethodAction::ColorReductionThresholdsEnable
        | MethodAction::ColorReductionThresholdsUnorm8
        | MethodAction::ColorReductionThresholdsUnorm10
        | MethodAction::ColorReductionThresholdsUnorm16
        | MethodAction::ColorReductionThresholdsFp16
        | MethodAction::ColorReductionThresholdsSrgb8
        | MethodAction::OpportunisticEarlyZHysteresis
        | MethodAction::ApiMandatedEarlyZ
        | MethodAction::PostPsInitialCoverage
        | MethodAction::PostZPixelShaderImask
        | MethodAction::PixelShaderInterlockControl
        | MethodAction::ConstantColorRenderingEnable
        | MethodAction::ConstantColorRenderingComponent(..)
        | MethodAction::AlphaFraction
        | MethodAction::TirMode
        | MethodAction::TirControl
        | MethodAction::TirModulation
        | MethodAction::TirModulationFunction
        | MethodAction::CoverageToColor
        | MethodAction::AlphaToCoverageOverride
        | MethodAction::AlphaToCoverageDither
        | MethodAction::HybridAntiAliasControl
        | MethodAction::SampleLocations(..)
        | MethodAction::RasterBoundingBox
        | MethodAction::VafL2CacheControl
        | MethodAction::RopL2CacheControl(..)
        | MethodAction::ReportSemaphoreAddressUpper
        | MethodAction::ReportSemaphoreAddressLower
        | MethodAction::ReportSemaphorePayload
        | MethodAction::PointSize
        | MethodAction::PointSpriteSelect
        | MethodAction::PointCenterMode
        | MethodAction::EdgeFlag
        | MethodAction::ViewportZClip
        | MethodAction::RenderEnableControl
        | MethodAction::RenderEnableMode
        | MethodAction::RenderEnableOverride
        | MethodAction::PsOutputSampleMaskUsage
        | MethodAction::VisibleCallLimit
        | MethodAction::SmTimeoutCounterBit
        | MethodAction::ShaderExceptionsEnable
        | MethodAction::ShaderLocalMemoryWindowBaseAddress
        | MethodAction::ShaderLocalMemoryAddressUpper
        | MethodAction::ShaderLocalMemoryAddressLower
        | MethodAction::ShaderLocalMemorySizeUpper
        | MethodAction::ShaderLocalMemorySizeLower
        | MethodAction::ShaderLocalMemoryDefaultSizePerWarp
        | MethodAction::CsaaEnable
        | MethodAction::AliasedLineWidthEnable
        | MethodAction::ActiveZCullRegion
        | MethodAction::ZCullStorage(..)
        | MethodAction::ZCullSubregionAllocation
        | MethodAction::ZCullSubregionAlgorithm
        | MethodAction::ZCullReportSelection
        | MethodAction::ZCullReportType
        | MethodAction::ZCullRegionLocation
        | MethodAction::ZCullRegionAliquots
        | MethodAction::ZCullRegionFormat
        | MethodAction::ZCullRegionSize(..)
        | MethodAction::ZCullRegionPixelOffset(..)
        | MethodAction::ZCullSubregion
        | MethodAction::ZCullDirectionFormat
        | MethodAction::ZCullStatsEnable
        | MethodAction::ZPassPixelCountEnable
        | MethodAction::StatisticsCounterEnable
        | MethodAction::StreamOutputEnable
        | MethodAction::ZCullCriterion
        | MethodAction::ZCullEnable
        | MethodAction::ZCullBounds => true,
        MethodAction::Notify
        | MethodAction::NoOperation
        | MethodAction::DecompressSurface
        | MethodAction::WaitForIdle
        | MethodAction::PixelShaderBarrier
        | MethodAction::TiledCacheBarrier
        | MethodAction::DiscardRenderTarget
        | MethodAction::MmeShadowRamControl
        | MethodAction::MutableMethodControl
        | MethodAction::FalconFirmwareCall4
        | MethodAction::TiledCacheFlush
        | MethodAction::CheckSphVersion
        | MethodAction::CheckAamVersion
        | MethodAction::ReportSemaphoreTrigger
        | MethodAction::InvalidateShaderCaches
        | MethodAction::InvalidateShaderCachesNoWfi
        | MethodAction::InvalidateTextureCacheNoWfi(..)
        | MethodAction::InvalidateTextureCache(..)
        | MethodAction::FlushPendingWrites
        | MethodAction::IncrementSyncpoint
        | MethodAction::ZCullMaintenance
        | MethodAction::ClearReportValue
        | MethodAction::DrawVertexArray
        | MethodAction::DrawIndexBuffer
        | MethodAction::Missing(..) => false,
    }
}

const PASSIVE_REGISTERS: [bool; 4096] = {
    let mut table = [false; 4096];
    let mut index = 0;
    while index < table.len() {
        let method = index as u32 * 4;
        table[index] = if output_state(method) {
            true
        } else if matches!(REGISTER_FAMILIES[index], RegisterFamily::VertexAndBinding) {
            !matches!(method, 0x1614 | 0x1618) && !(method >= 0x2400 && method < 0x2500)
        } else if method >= 0x180 && method <= 0x1ac {
            true
        } else if let Some(declaration) = METHODS_BY_DWORD[index] {
            passive_action(declaration.action)
        } else {
            false
        };
        index += 1;
    }
    table
};

pub(super) fn passive(method: GpuMethodId) -> bool {
    method.0 & 3 == 0
        && PASSIVE_REGISTERS
            .get((method.0 / 4) as usize)
            .copied()
            .unwrap_or(false)
}

/// Validate the encoding consumed by the register write, independently of
/// relational/rendering validation at a later draw. Pure decoders build no
/// state mirror; repeated equal, already admitted values need no validation.
#[inline]
pub(super) fn validate_encoding(
    source: crate::MaxwellMethodSource,
    candidate: &MaxwellThreeDFrontendState,
) -> Result<(), MaxwellEngineDispatchError> {
    match register_family(source.method().0) {
        RegisterFamily::VertexAndBinding => {
            if preflight_vertex_and_binding_state(source, candidate)?.is_none() {
                return Err(MaxwellEngineDispatchError::UnknownMethod {
                    source,
                    class_name: CLASS_NAME,
                });
            }
        }
        RegisterFamily::InlineToMemory => {
            inline_to_memory::preflight(source, candidate.operation_state().inline_to_memory())?;
        }
        _ => {
            if let Some(declaration) = METHODS_BY_DWORD[(source.method().0 / 4) as usize].as_ref() {
                decode_declared_state(source, declaration)?;
            } else if preflight_output_state(source)?.is_none() {
                return Err(MaxwellEngineDispatchError::UnknownMethod {
                    source,
                    class_name: CLASS_NAME,
                });
            }
        }
    }
    Ok(())
}

pub(super) fn materialize(
    profile: &MaxwellGpuProfile,
    candidate: &mut MaxwellThreeDFrontendState,
    groups: u64,
) -> Result<(), MaxwellEngineDispatchError> {
    while let Some(source) = candidate.pending_register(groups) {
        if candidate.pending_returns_to_original(source) {
            candidate.decoded_register(source.method());
            continue;
        }
        let decoded = preflight_register(profile, source, candidate, true)?;
        debug_assert!(decoded.operation.is_none() && decoded.writes_state);
        candidate.decoded_register(source.method());
    }
    Ok(())
}

const fn dependency_groups(method: u32) -> u64 {
    match method {
        0x2390..=0x23fc | 0x2410..=0x24f0 => (1 << 0x23) | (1 << 0x15),
        0x1b0 | 0x1b4 => 1 << 1,
        // BEGIN/END operate the primitive sequence, not immutable draw state.
        0x238c | 0x1614 | 0x1618 | 0x114 | 0x118 | 0x11c | 0x120 | 0x124 | 0x3400..=0x37fc => 0,
        _ => u64::MAX,
    }
}

static DEPENDENCY_GROUPS: [u64; 4096] = {
    let mut groups = [0; 4096];
    let mut index = 0;
    while index < groups.len() {
        groups[index] = dependency_groups(index as u32 * 4);
        index += 1;
    }
    groups
};

pub(super) fn dependencies(method: GpuMethodId) -> u64 {
    DEPENDENCY_GROUPS
        .get((method.0 / 4) as usize)
        .copied()
        .unwrap_or(u64::MAX)
}

/// Complete the valid prefix of a contiguous CB data packet in one transition.
/// The borrowed packet remains the payload; no per-word effect/source vector is built.
pub(in crate::engines) fn constant_buffer_words(
    profile: &MaxwellGpuProfile,
    candidate: &mut MaxwellThreeDFrontendState,
    subchannel: crate::MaxwellPushbufferSubchannel,
    words: crate::MaxwellMethodSpan<'_>,
) -> Result<(MaxwellThreeDInlineConstantBufferUpload, usize), MaxwellEngineDispatchError> {
    materialize(
        profile,
        candidate,
        dependencies(words.first().unwrap().method()),
    )?;
    let first = crate::MaxwellMethodSource::decoded(words.first().unwrap(), subchannel);
    let (_, _, upload) =
        preflight_constant_buffer_load(first, candidate)?.expect("CB data aperture");
    let upload = upload.expect("CB data produces an upload");
    let offset = upload.offset();
    let size = *candidate
        .operation_state()
        .shader_bindings()
        .selector()
        .size()
        .value()
        .unwrap();
    let available =
        u64::from(size - offset).min((1_u64 << 40) - upload.address().get() - u64::from(offset));
    let count = words.len().min((available / 4) as usize);
    debug_assert!(count != 0);
    let last = crate::MaxwellMethodSource::decoded(words.get(count - 1).unwrap(), subchannel);
    candidate.apply(
        MaxwellThreeDStateWrite::ShaderBinding(
            MaxwellThreeDShaderBindingWrite::ConstantBufferLoadData {
                value: last.argument(),
                next_offset: offset + count as u32 * 4,
                source: last,
            },
        ),
        true,
    );
    let control = candidate.mme().shadow_ram_control().value().copied();
    for word in words.slice(..count).last_writes() {
        let source = crate::MaxwellMethodSource::decoded(word, subchannel);
        candidate.record_raw_register(source);
        candidate.mme_mut().track_shadow_register(control, source);
    }
    Ok((upload, count))
}

pub(in crate::engines) fn embedded_words(
    profile: &MaxwellGpuProfile,
    candidate: &mut MaxwellThreeDFrontendState,
    subchannel: crate::MaxwellPushbufferSubchannel,
    words: crate::MaxwellMethodSpan<'_>,
) -> Result<(MaxwellInlineToMemoryUpload, usize), MaxwellEngineDispatchError> {
    materialize(profile, candidate, 1 << 1)?;
    let count = words.len().min(
        candidate
            .operation_state()
            .inline_to_memory()
            .pending_words(),
    );
    let first = crate::MaxwellMethodSource::decoded(words.first().unwrap(), subchannel);
    let (_, _, upload) =
        inline_to_memory::preflight(first, candidate.operation_state().inline_to_memory())?
            .expect("embedded inline data");
    let upload = upload.expect("embedded inline data produces an upload");
    debug_assert!(count != 0);
    let last = crate::MaxwellMethodSource::decoded(words.get(count - 1).unwrap(), subchannel);
    candidate.apply(
        MaxwellThreeDStateWrite::InlineToMemory(MaxwellThreeDInlineToMemoryStateWrite::Data {
            value: last.argument(),
            next_offset: upload.offset() + count as u32 * 4,
            source: last,
        }),
        true,
    );
    candidate.record_raw_register(last);
    let control = candidate.mme().shadow_ram_control().value().copied();
    candidate.mme_mut().track_shadow_register(control, last);
    Ok((upload, count))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deferred_values_are_visible_to_mme_before_decode_and_unchanged_groups_keep_identities() {
        let mut channel = crate::engines::tests::three_d_channel();
        crate::engines::tests::program_three_d(&mut channel, 0x1518, 0);
        let source = channel.three_d().raster().point_size().source().unwrap();
        let (profile, state) = channel.three_d_with_profile_mut();
        let before = state.operation_state().draw_state_identity();
        for value in 1..=100 {
            let mut method = MaxwellMethodDispatch::emitted_by_mme(
                source.emitted_by_mme(GpuMethodId(0x1518), value),
                CLASS,
            );
            assert!(write_plain_method(&mut method, state).unwrap());
        }
        assert_eq!(state.raw_register_value(GpuMethodId(0x1518)), Some(100));
        assert_eq!(state.operation_state().raster().point_size().raw(), Some(0));
        assert!(state.pending_register(u64::MAX).is_some());
        materialize(profile, state, u64::MAX).unwrap();
        assert!(!before.matches_parameters(state.operation_state()));
        let current = state.operation_state().draw_state_identity();
        for _ in 0..100 {
            let mut method = MaxwellMethodDispatch::emitted_by_mme(
                source.emitted_by_mme(GpuMethodId(0x1518), 100),
                CLASS,
            );
            assert!(write_plain_method(&mut method, state).unwrap());
        }
        assert!(state.pending_register(u64::MAX).is_none());
        // An intermediate changed value restored before consumption also keeps
        // the already decoded group and its prepared identity.
        for value in [101, 102, 100] {
            let mut method = MaxwellMethodDispatch::emitted_by_mme(
                source.emitted_by_mme(GpuMethodId(0x1518), value),
                CLASS,
            );
            write_plain_method(&mut method, state).unwrap();
        }
        materialize(profile, state, u64::MAX).unwrap();
        assert!(
            current.matches(state.operation_state())
                && current.matches_parameters(state.operation_state())
        );
    }

    #[test]
    fn passive_table_never_coalesces_an_execution_method_or_unknown_address() {
        let mut channel = crate::engines::tests::three_d_channel();
        crate::engines::tests::program_three_d(&mut channel, 0x0d74, 0);
        let source = channel
            .three_d()
            .vertex_input()
            .primitive()
            .vertex_array_start()
            .source()
            .unwrap();
        let (profile, state) = channel.three_d_with_profile_mut();
        for method in (0..0x4000).step_by(4) {
            if passive(GpuMethodId(method)) {
                let mut recognized = false;
                for argument in [0, 1, u32::MAX] {
                    match preflight_register(
                        profile,
                        source.emitted_by_mme(GpuMethodId(method), argument),
                        state,
                        true,
                    ) {
                        Ok(decoded) => {
                            recognized = true;
                            assert!(
                                decoded.operation.is_none() && decoded.writes_state,
                                "{method:#x}"
                            );
                        }
                        Err(MaxwellEngineDispatchError::UnknownMethod { .. }) => {}
                        Err(_) => {
                            recognized = true;
                        }
                    }
                }
                assert!(recognized, "unknown passive register {method:#x}");
            }
        }
        for method in [
            0x110, 0x118, 0x120, 0x124, 0x1300, 0x15e8, 0x15ec, 0x1614, 0x1618, 0x19d0, 0x238c,
            0x2390, 0x2410, 0x3800,
        ] {
            assert!(
                !passive(GpuMethodId(method)),
                "execution method {method:#x}"
            );
        }
    }
}
