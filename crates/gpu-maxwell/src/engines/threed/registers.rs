//! Canonical register writes and renderer dirty domains.
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
        | MethodAction::TiledCacheUnknownConfig
        | MethodAction::ShaderWatermarks
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
        | MethodAction::ConstantColorRenderingComponent
        | MethodAction::AlphaFraction
        | MethodAction::TirMode
        | MethodAction::TirControl
        | MethodAction::TirModulation
        | MethodAction::TirModulationFunction
        | MethodAction::CoverageToColor
        | MethodAction::AlphaToCoverageOverride
        | MethodAction::AlphaToCoverageDither
        | MethodAction::HybridAntiAliasControl
        | MethodAction::SampleLocations
        | MethodAction::RasterBoundingBox
        | MethodAction::VafL2CacheControl
        | MethodAction::RopL2CacheControl
        | MethodAction::ReportSemaphoreAddressUpper
        | MethodAction::ReportSemaphoreAddressLower
        | MethodAction::ReportSemaphorePayload
        | MethodAction::PointSize
        | MethodAction::PointSpriteSelect
        | MethodAction::PointCenterMode
        | MethodAction::EdgeFlag
        | MethodAction::ViewportZClip
        | MethodAction::RenderEnableControl
        | MethodAction::RenderEnableAddressUpper
        | MethodAction::RenderEnableAddressLower
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
        | MethodAction::ZCullStorage
        | MethodAction::ZCullSubregionAllocation
        | MethodAction::ZCullSubregionAlgorithm
        | MethodAction::ZCullReportSelection
        | MethodAction::ZCullReportType
        | MethodAction::ZCullRegionLocation
        | MethodAction::ZCullRegionAliquots
        | MethodAction::ZCullRegionFormat
        | MethodAction::ZCullRegionSize
        | MethodAction::ZCullRegionPixelOffset
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
        | MethodAction::RenderEnableMode
        | MethodAction::RenderEnableOverride
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

/// Complete the valid prefix of a contiguous CB data packet in one transition.
/// The borrowed packet remains the payload; no per-word effect/source vector is built.
pub(in crate::engines) fn constant_buffer_words(
    candidate: &mut MaxwellThreeDFrontendState,
    subchannel: crate::MaxwellPushbufferSubchannel,
    words: crate::MaxwellMethodSpan<'_>,
) -> Result<(MaxwellThreeDInlineConstantBufferUpload, usize), MaxwellEngineDispatchError> {
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
    candidate.apply(MaxwellThreeDStateWrite::ShaderBinding(
        MaxwellThreeDShaderBindingWrite::ConstantBufferLoadData {
            next_offset: offset + count as u32 * 4,
            source: last,
        },
    ));
    let control = candidate.mme().shadow_ram_control().value().copied();
    for word in words.slice(..count).last_writes() {
        let source = crate::MaxwellMethodSource::decoded(word, subchannel);
        candidate.record_raw_register(source);
        candidate
            .mme_mut()
            .track_shadow_register(control, source.method(), source.argument());
    }
    Ok((upload, count))
}

pub(in crate::engines) fn embedded_words(
    candidate: &mut MaxwellThreeDFrontendState,
    subchannel: crate::MaxwellPushbufferSubchannel,
    words: crate::MaxwellMethodSpan<'_>,
) -> Result<(MaxwellInlineToMemoryUpload, usize), MaxwellEngineDispatchError> {
    let count = words.len().min(
        candidate
            .operation_state()
            .inline_to_memory()
            .pending_words(),
    );
    let first = crate::MaxwellMethodSource::decoded(words.first().unwrap(), subchannel);
    let (_, _, upload) =
        inline_to_memory::preflight(first, &candidate.operation_state().inline_to_memory())?
            .expect("embedded inline data");
    let upload = upload.expect("embedded inline data produces an upload");
    debug_assert!(count != 0);
    let last = crate::MaxwellMethodSource::decoded(words.get(count - 1).unwrap(), subchannel);
    candidate.apply(MaxwellThreeDStateWrite::InlineToMemory(
        MaxwellThreeDInlineToMemoryStateWrite::Data {
            value: last.argument(),
            next_offset: upload.offset() + upload.remaining_bytes().min(count as u32 * 4),
            source: last,
        },
    ));
    candidate.record_raw_register(last);
    let control = candidate.mme().shadow_ram_control().value().copied();
    candidate
        .mme_mut()
        .track_shadow_register(control, last.method(), last.argument());
    Ok((upload, count))
}

// Renderer domains consume the canonical register bank. These flags select
// cache keys to rebuild; they never enqueue another interpretation of methods.
// https://github.com/eden-emulator/mirror/blob/d16735f5b618942136d6ab53466e3be0a382c30a/src/video_core/engines/maxwell_3d.cpp
pub(super) const DRAW: u16 = 1;
pub(super) const PARAMETERS: u16 = 2;
pub(super) const SHADERS: u16 = 4;
pub(super) const VERTEX: u16 = 8;
pub(super) const INDEX: u16 = 16;
pub(super) const COLOR: u16 = 32;
pub(super) const DEPTH: u16 = 64;
pub(super) const HEADERS: u16 = 128;
pub(super) const SAMPLERS: u16 = 256;
pub(super) const SAMPLE_MODE: u16 = 512;

const fn dirty_domains(method: u32) -> u16 {
    let mut flags = 0;
    if output_state(method)
        && !matches!(
            method,
            0x07a4 | 0x07a8 | 0x0d6c | 0x0d70 | 0x0d80..=0x0d90 | 0x0da0 | 0x10f8
        )
    {
        flags |= DRAW | PARAMETERS;
    }
    // Vertex fetch configuration and primitive sequence state.
    if matches!(method, 0x1120..=0x112c | 0x1160..=0x11dc | 0x1880..=0x18fc | 0x1c00..=0x1dfc | 0x1f00..=0x1ffc
        | 0x0374 | 0x0f84 | 0x0f88 | 0x02d0 | 0x0dcc | 0x0de8 | 0x1118 | 0x1434 | 0x1438 | 0x1610
        | 0x1618 | 0x1644 | 0x1648 | 0x164c | 0x17c8..=0x17e0 | 0x1948 | 0x1970 | 0x0320..=0x0338)
    {
        flags |= DRAW | PARAMETERS;
    }
    if matches!(method, 0x1c00..=0x1dff) {
        if (method - 0x1c00) % 16 <= 8 {
            flags |= VERTEX;
        }
        if matches!((method - 0x1c00) % 16, 4 | 8) {
            flags &= !PARAMETERS;
        }
    }
    if matches!(method, 0x1f00..=0x1ffc) {
        flags = DRAW | VERTEX;
    }
    if matches!(method, 0x17c8..=0x17e0) {
        flags |= INDEX;
        if method != 0x17d8 {
            flags &= !PARAMETERS;
        }
    }
    if matches!(method, 0x1160..=0x11dc | 0x13ac | 0x192c | 0x1608 | 0x160c | 0x2000..=0x217c | 0x2608 | 0x1240..=0x125c)
    {
        flags |= SHADERS;
    }
    if matches!(method, 0x0800..=0x09fc) && (method - 0x0800) % 0x40 <= 0x20
        || matches!(method, 0x19e0..=0x19fc | 0x15b8)
    {
        flags |= COLOR;
    }
    if matches!(method, 0x0fe0..=0x0ff0 | 0x1208 | 0x1228..=0x1230 | 0x179c | 0x19cc) {
        flags |= DEPTH;
    }
    if method == 0x15d0 {
        flags |= SAMPLE_MODE;
    }
    if matches!(method, 0x1574 | 0x1578 | 0x157c | 0x0f10 | 0x1234) {
        flags |= HEADERS;
    }
    if matches!(method, 0x155c | 0x1560 | 0x1564 | 0x1234 | 0x157c) {
        flags |= SAMPLERS;
    }
    if let Some(declaration) = METHODS_BY_DWORD[(method / 4) as usize]
        && matches!(
            declaration.action,
            MethodAction::IteratedBlend
                | MethodAction::IteratedBlendPassCount
                | MethodAction::SubtilingPerfKnobA
                | MethodAction::SubtilingPerfKnobB
                | MethodAction::ShaderWatermarks
                | MethodAction::L1Configuration
                | MethodAction::ColorReductionThresholdsEnable
                | MethodAction::ColorReductionThresholdsUnorm8
                | MethodAction::ColorReductionThresholdsUnorm10
                | MethodAction::ColorReductionThresholdsUnorm16
                | MethodAction::ColorReductionThresholdsFp16
                | MethodAction::ColorReductionThresholdsSrgb8
                | MethodAction::ApiMandatedEarlyZ
                | MethodAction::PostPsInitialCoverage
                | MethodAction::PostZPixelShaderImask
                | MethodAction::PixelShaderInterlockControl
                | MethodAction::AlphaFraction
                | MethodAction::RasterBoundingBox
                | MethodAction::PointSize
                | MethodAction::PointSpriteSelect
                | MethodAction::PointCenterMode
                | MethodAction::EdgeFlag
                | MethodAction::ViewportZClip
                | MethodAction::PsOutputSampleMaskUsage
                | MethodAction::HybridAntiAliasControl
                | MethodAction::TirMode
                | MethodAction::TirControl
                | MethodAction::TirModulation
                | MethodAction::TirModulationFunction
                | MethodAction::CoverageToColor
                | MethodAction::AlphaToCoverageOverride
                | MethodAction::AlphaToCoverageDither
                | MethodAction::SampleLocations
                | MethodAction::RenderEnableControl
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
                | MethodAction::StreamOutputEnable
        )
    {
        flags |= DRAW | PARAMETERS;
    }
    flags
}
static DIRTY_DOMAINS: [u16; 4096] = {
    let mut domains = [0; 4096];
    let mut index = 0;
    while index < domains.len() {
        domains[index] = dirty_domains(index as u32 * 4);
        index += 1;
    }
    domains
};
pub(super) fn domains(method: GpuMethodId) -> u16 {
    DIRTY_DOMAINS[(method.0 / 4) as usize]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::tests::{program_three_d, three_d_channel};

    #[test]
    fn passive_configuration_is_immediately_visible_without_replaying_methods() {
        let mut channel = three_d_channel();
        for raw in [0, 1_f32.to_bits(), 2_f32.to_bits()] {
            program_three_d(&mut channel, 0x1518, raw);
            assert_eq!(channel.three_d().raster().point_size().raw(), Some(raw));
            assert_eq!(
                channel
                    .three_d_mut()
                    .raw_register_value(GpuMethodId(0x1518)),
                Some(raw)
            );
        }
    }

    #[test]
    fn noop_and_wait_do_not_decode_intermediate_configuration() {
        let mut channel = three_d_channel();
        program_three_d(&mut channel, 0x0a1c, 0xffff_ffff);
        program_three_d(&mut channel, 0x20c0, 0xffff_ffff);
        let before = channel.three_d().draw_state_identity();
        for method in [0x100, 0x110] {
            program_three_d(&mut channel, method, 0);
            assert!(before.matches(channel.three_d()));
        }
        // A later complete configuration replaces intermediate bits directly.
        program_three_d(&mut channel, 0x0a1c, 0x0302);
        program_three_d(&mut channel, 0x20c0, 0x31);
        assert_eq!(
            channel
                .three_d()
                .fixed_function()
                .viewport(0)
                .snap_grid_precision()
                .value(),
            Some(&[2, 3])
        );
        assert_eq!(
            channel
                .three_d()
                .shader_bindings()
                .pipeline(3)
                .stage()
                .value(),
            Some(&MaxwellShaderStage::Tessellation)
        );
    }

    #[test]
    fn graphics_domains_change_at_draw_consumption_and_ignore_memory_selectors() {
        let mut channel = three_d_channel();
        program_three_d(&mut channel, 0x0d78, 3);
        let before = channel.three_d().draw_state_identity();
        program_three_d(&mut channel, 0x2380, 64);
        program_three_d(&mut channel, 0x0d78, 3);
        assert!(before.matches(channel.three_d()));
        program_three_d(&mut channel, 0x1518, 2_f32.to_bits());
        program_three_d(&mut channel, 0x0d78, 3);
        assert!(!before.matches_parameters(channel.three_d()));
    }
}
