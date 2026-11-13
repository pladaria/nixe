//! Canonical register writes and renderer dirty domains.
//!
//! Ordinary writes update one authoritative raw word and dirty metadata.
//! Execution methods retain ordered effects even when the argument is equal.
//! Reference mechanisms (independently implemented):
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/engines/maxwell_3d.cpp

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
        | MethodAction::VertexCount
        | MethodAction::IndexCount
        | MethodAction::Missing(..) => false,
    }
}

// Like Eden's execution mask and dirty tables, method routing is fixed at build time.
// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/engines/maxwell_3d.cpp
// Each word maps directly to its canonical consumer flags. There is no
// delayed domain/revision conversion.
pub(super) const VERTEX_SHIFT: u32 = Flags::VERTEX_SHIFT;
pub(super) const COLOR_SHIFT: u32 = Flags::COLOR_SHIFT;
use nixe_gpu::native::GraphicsDirty as Flags;
const VIEWPORT: u128 = Flags::VIEWPORT;
const SCISSOR: u128 = Flags::SCISSOR;
const BLEND_CONSTANTS: u128 = Flags::BLEND_CONSTANTS;
const ALPHA_TEST: u128 = Flags::ALPHA_TEST;
const CONFIGURATION: u128 = Flags::PIPELINE_STATE;

// Dirty tables distinguish consumers, not historical register combinations.
// Coupled state (window origin, surface clip, alpha-test activation) marks
// every consumer, matching Eden's multi-table dirty routing.
// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/vk_state_tracker.cpp
// Raster validation consumes enabled shader stages, program-region completeness
// and viewport mode. Code offsets, CB-group/handle selectors and geometry
// passthrough masks belong to code/resources, not fixed-function parameters.
// Inputs to unsupported-state checks which do not already have a dedicated
// consumed raster/attachment flag. Keep raw admission and precise errors at
// consumption; changed unsupported words must not hide behind a cached key.
const fn validation_input(method: u32) -> bool {
    matches!(method,
        0x030c | 0x02e4 | 0x1688 | 0x13a8 | 0x07a0 | 0x0d9c | 0x0f1c
        | 0x15b4 | 0x153c | 0x11f8 | 0x16b4 | 0x0fb4 | 0x194c | 0x197c
        | 0x1924 | 0x1148 | 0x19bc | 0x12d4 | 0x1138 | 0x113c | 0x0754
        | 0x1224 | 0x0dd0 | 0x19c4 | 0x19c8 | 0x1534 | 0x0fa4
        | 0x0fbc..=0x0fc8 | 0x11e0..=0x11ec | 0x15e4 | 0x1948)
}
const fn parameter_dirty(method: u32, domains: u16) -> u128 {
    if method >= 0x0a18 && method <= 0x0bf8 && (method - 0x0a18).is_multiple_of(0x20) {
        return Flags::VIEWPORT_SWIZZLES | if method == 0x0a18 { VIEWPORT } else { 0 };
    }
    if domains & (COLOR | DEPTH | TARGET_CONTROL | SAMPLE_MODE) != 0 {
        return 0;
    }
    let configuration = if validation_input(method) {
        CONFIGURATION
    } else {
        0
    };
    if matches!(method, 0x1160..=0x11dc) {
        return Flags::VERTEX_ATTRIBUTES | configuration;
    }
    if matches!(method, 0x1880..=0x18fc)
        || (method >= 0x1c00 && method <= 0x1dfc && (method - 0x1c00) % 16 == 12)
    {
        return Flags::VERTEX_INSTANCING | configuration;
    }
    // Stride belongs to VertexBuffers, not VertexInput. Static-stride hosts
    // read it while refreshing their fixed pipeline state, as Eden does.
    if method >= 0x1c00 && method <= 0x1dfc && (method - 0x1c00).is_multiple_of(16) {
        return configuration;
    }
    configuration
        | match method {
            0x0f90 | 0x1a00..=0x1a1c => Flags::COLOR_MASK,
            0x135c | 0x1360..=0x137c => Flags::BLEND_ENABLE,
            0x12e4 | 0x133c..=0x1358 | 0x1e00..=0x1efc => Flags::BLEND_EQUATIONS,
            0x0a00..=0x0a1c | 0x0c00..=0x0c0c | 0x0d7c => VIEWPORT,
            0x193c => VIEWPORT | Flags::DEPTH_CLAMP_ENABLE,
            0x192c => VIEWPORT,
            0x0e00..=0x0e08 => SCISSOR,
            0x0ff4 | 0x0ff8 => VIEWPORT,
            0x13ac => VIEWPORT | Flags::FRONT_FACE,
            0x12cc => Flags::DEPTH_TEST | VIEWPORT,
            0x12e8 => Flags::DEPTH_WRITE,
            0x130c => Flags::DEPTH_COMPARE,
            0x19bc => Flags::DEPTH_BOUNDS_ENABLE,
            0x0f9c | 0x0fa0 => Flags::DEPTH_BOUNDS,
            0x1918 | 0x1920 => Flags::CULL_MODE,
            0x191c => Flags::FRONT_FACE,
            0x037c => Flags::RASTERIZER_DISCARD,
            0x1644 => Flags::PRIMITIVE_RESTART,
            0x0dc0 | 0x0dc4 | 0x0dc8 => Flags::DEPTH_BIAS_ENABLE,
            0x15bc | 0x187c | 0x156c => Flags::DEPTH_BIAS,
            0x13b0 | 0x13b4 | 0x1570 | 0x020c => Flags::LINE_WIDTH,
            0x1380 => Flags::STENCIL_TEST,
            0x1384..=0x1390 | 0x1598..=0x15a4 => Flags::STENCIL_OP,
            0x1594 => Flags::STENCIL_PROPERTIES | Flags::STENCIL_OP,
            0x1394 | 0x0f54 => Flags::STENCIL_PROPERTIES | Flags::STENCIL_REFERENCE,
            0x139c | 0x0f58 => Flags::STENCIL_PROPERTIES | Flags::STENCIL_WRITE_MASK,
            0x1398 | 0x0f5c => Flags::STENCIL_PROPERTIES | Flags::STENCIL_COMPARE,
            0x131c..=0x1328 => BLEND_CONSTANTS,
            0x12ec => CONFIGURATION | ALPHA_TEST,
            0x1310 | 0x1314 => ALPHA_TEST,
            _ => {
                if domains & PARAMETERS != 0 {
                    CONFIGURATION
                } else {
                    0
                }
            }
        }
}

// ProcessDirtyRegisters uses two byte indices into the canonical flag bank.
// No family or descriptive metadata is read for ordinary register writes.
// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/engines/maxwell_3d.h#L3150-L3157
pub(super) static DIRTY_TABLES: [[u8; 0xe00]; 2] = {
    let mut tables = [[Flags::NULL_INDEX; 0xe00]; 2];
    let mut index = 0;
    while index < 0xe00 {
        let method = index as u32 * 4;
        let domains = dirty_domains(method);
        let vertices = if domains & VERTEX != 0 {
            let slot = if method >= 0x1f00 {
                (method - 0x1f00) / 8
            } else {
                (method - 0x1c00) / 16
            };
            1_u32 << slot
        } else {
            0
        };
        let colors = if domains & COLOR != 0 {
            if method == 0x15b8 {
                0xff
            } else if method >= 0x19e0 {
                1_u8 << ((method - 0x19e0) / 4)
            } else {
                1_u8 << ((method - 0x0800) / 0x40)
            }
        } else {
            0
        };
        let mut flags = parameter_dirty(method, domains)
            | ((vertices as u128) << VERTEX_SHIFT)
            | ((colors as u128) << COLOR_SHIFT);
        if domains & VERTEX != 0 {
            flags |= if method >= 0x1c00 && method <= 0x1df0 && (method - 0x1c00).is_multiple_of(16)
            {
                Flags::VERTEX_STRIDES
            } else {
                Flags::VERTEX_BUFFER_ADDRESSES
            };
        }
        if domains & INDEX != 0 {
            flags |= Flags::INDEX;
        }
        if domains & SHADERS != 0 {
            flags |= Flags::SHADERS;
        }
        // Attachment addresses/layouts dirty TextureCache slots. Only format,
        // sample mode and Zeta enable alter native attachment parameters.
        if matches!(method, 0x0fe8 | 0x1538 | 0x15d0)
            || (method >= 0x0810 && method < 0x0a00 && (method - 0x0810).is_multiple_of(0x40))
        {
            flags |= Flags::TARGET_FORMAT;
        }
        if domains & DEPTH != 0 {
            flags |= Flags::DEPTH;
        }
        if method == 0x1538 {
            flags |= Flags::DEPTH;
        }
        if method == 0x15b8 {
            flags = Flags::RENDER_TARGET_CONTROL;
        }
        if domains & TARGET_CONTROL != 0 && method != 0x1538 {
            flags = Flags::RENDER_TARGET_CONTROL | Flags::FRAMEBUFFER;
        }
        if domains & SAMPLE_MODE != 0 {
            flags = Flags::RENDER_TARGET_CONTROL | Flags::TARGET_FORMAT;
        }
        if domains & FRAMEBUFFER != 0 {
            flags |= Flags::FRAMEBUFFER;
        }
        // Multi-consumer state uses the same two-index routing as Eden.
        // Viewport consumption also invalidates scissors; RT control forces
        // current target lookups in their owning texture cache.
        assert!(flags.count_ones() <= 2);
        if flags != 0 {
            tables[0][index] = flags.trailing_zeros() as u8;
            flags &= flags - 1;
        }
        if flags != 0 {
            tables[1][index] = flags.trailing_zeros() as u8;
        }
        index += 1;
    }
    tables
};
// DmaPusher consults one execution bit before touching dirty-register routing.
// The immutable class mask is shared by channels; no per-channel rebuild or
// second per-method execution flag is retained.
// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/engines/engine_interface.h#L46
// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/dma_pusher.cpp#L175-L191
static EXECUTION_MASK: [u64; 4096 / 64] = {
    let mut mask = [0; 4096 / 64];
    let mut index = 0;
    while index < 4096 {
        if !passive_method(index) {
            mask[index / 64] |= 1 << (index % 64);
        }
        index += 1;
    }
    mask
};
const fn passive_method(index: usize) -> bool {
    let method = index as u32 * 4;
    let family = classify_family(method);
    if output_state(method) {
        true
    } else if matches!(family, RegisterFamily::VertexAndBinding) {
        !matches!(
            method,
            0x0d74 | 0x1118 | 0x1434 | 0x1438 | 0x1614 | 0x1618 | 0x17dc | 0x1970
        ) && !(method >= 0x2400 && method < 0x2500)
    } else if (method >= 0x3400
        && method < 0x3400 + MAXWELL_THREE_D_MME_SHADOW_SCRATCH_COUNT as u32 * 4)
        || (method >= 0x180 && method <= 0x1ac)
    {
        true
    } else if let Some(declaration) = METHODS_BY_DWORD[index] {
        passive_action(declaration.action)
    } else {
        false
    }
}
#[inline]
pub(super) fn executable(method: GpuMethodId) -> bool {
    let index = (method.0 / 4) as usize;
    method.0 & 3 != 0
        || EXECUTION_MASK
            .get(index / 64)
            .is_none_or(|word| word & (1 << (index % 64)) != 0)
}

pub(super) fn completes_notification(method: GpuMethodId) -> bool {
    METHODS_BY_DWORD
        .get((method.0 / 4) as usize)
        .and_then(|declaration| *declaration)
        .is_some_and(|declaration| matches!(declaration.action, MethodAction::NoOperation))
}

pub(in crate::engines) fn embedded_words(
    candidate: &mut MaxwellThreeDEngineState,
    method: GpuMethodId,
    subchannel: crate::MaxwellPushbufferSubchannel,
    words: crate::MaxwellDmaArguments<'_>,
) -> Result<(MaxwellInlineToMemoryUpload, usize), MaxwellEngineDispatchError> {
    let count = words.len().min(
        candidate
            .operation_state()
            .inline_to_memory()
            .pending_words(),
    );
    let first = words.source(0, method, subchannel);
    let (_, upload) =
        inline_to_memory::preflight(first, &candidate.operation_state().inline_to_memory())?
            .expect("embedded inline data");
    let upload = upload.expect("embedded inline data produces an upload");
    debug_assert!(count != 0);
    let last = words.source(count - 1, method, subchannel);
    candidate.apply_inline(MaxwellThreeDInlineToMemoryStateWrite::Data {
        value: last.argument(),
        next_offset: upload.offset() + upload.remaining_bytes().min(count as u32 * 4),
        source: last,
    });
    candidate.record_raw_register(last);
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
pub(super) const TARGET_CONTROL: u16 = 128;
pub(super) const FRAMEBUFFER: u16 = 256;
pub(super) const SAMPLE_MODE: u16 = 512;

const fn dirty_domains(method: u32) -> u16 {
    let mut flags = 0;
    if matches!(method, 0x0ff4 | 0x0ff8) {
        flags |= FRAMEBUFFER;
    }
    if matches!(method, 0x121c | 0x1538) {
        flags |= TARGET_CONTROL;
    }
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
        | 0x1644 | 0x1648 | 0x164c | 0x17c8..=0x17e0 | 0x1948 | 0x1970 | 0x0320..=0x0338)
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
    // Base vertex/instance are draw arguments, not shader specialization.
    if matches!(method, 0x1434 | 0x1438) {
        flags &= !PARAMETERS;
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
    // Code capture consumes program selection and environment observations.
    // Vertex formats and window origin specialize the native pipeline instead.
    // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/fixed_pipeline_state.cpp
    if matches!(method, 0x192c | 0x1608 | 0x160c | 0x2608 | 0x1240..=0x125c)
        || (method >= 0x2000 && method < 0x2180 && matches!((method - 0x2000) % 0x40, 0 | 4 | 16))
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
    // Physical attachment addresses select views, not rasterization semantics.
    if matches!(method, 0x0fe0 | 0x0fe4)
        || (method >= 0x0800 && method < 0x0a00 && (method - 0x0800) % 0x40 <= 4)
    {
        flags &= !PARAMETERS;
    }
    // The current raster path consumes viewport/scissor zero. Window clipping
    // is rejected by its enable control; its rectangle words are not consumed.
    // Keep other words canonical for MME and future consumers without treating
    // their programming as a change to the current draw's raster parameters.
    if matches!(method, 0x0a20..=0x0bfc | 0x0c10..=0x0cfc | 0x0e10..=0x0efc | 0x1700..=0x177c) {
        flags &= !(DRAW | PARAMETERS);
    }
    flags
}
