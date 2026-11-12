//! GM20B `MAXWELL_B` 3D class methods and register transitions.

mod bindings;
mod color_reduction;
mod coverage;
pub(crate) mod draw;
pub(crate) mod front;
pub use draw::{MaxwellThreeDDrawIndexBuffer, MaxwellThreeDDrawState};
mod inline_to_memory;
mod line;
mod mme;
mod operations;
mod output;
pub(in crate::engines) mod registers;
mod render_enable;
mod render_targets;
mod resource;
pub use resource::texture::texture_address;
mod shader_execution;
pub mod state;
pub(super) use registers::embedded_words;
pub mod tessellation;
mod vertex;
pub use tessellation::{MaxwellTessellationModeError, MaxwellThreeDTessellationMode};

pub use super::lowering::MaxwellLoweringError;
pub use bindings::{
    MAXWELL_BIND_GROUP_COUNT, MAXWELL_CONSTANT_BUFFER_SLOT_COUNT, MAXWELL_PIPELINE_SHADER_COUNT,
    MAXWELL_TESSELLATION_LOD_COUNT, MaxwellShaderStage, MaxwellThreeDBindGroupState,
    MaxwellThreeDConstantBufferBinding, MaxwellThreeDConstantBufferSelectorState,
    MaxwellThreeDDescriptorPoolState, MaxwellThreeDPipelineBindingState,
    MaxwellThreeDProgramRegionState, MaxwellThreeDSamplerBindingMode,
    MaxwellThreeDShaderBindingState, MaxwellThreeDTessellationLod,
};
pub use color_reduction::{
    MaxwellThreeDColorReductionFp16Threshold, MaxwellThreeDColorReductionSrgb8Threshold,
    MaxwellThreeDColorReductionState, MaxwellThreeDColorReductionThresholdsEnable,
    MaxwellThreeDColorReductionThresholdsFp16, MaxwellThreeDColorReductionThresholdsSrgb8,
    MaxwellThreeDColorReductionThresholdsUnorm8, MaxwellThreeDColorReductionThresholdsUnorm10,
    MaxwellThreeDColorReductionThresholdsUnorm16, MaxwellThreeDUnorm8,
};
pub use coverage::{
    MAXWELL_SAMPLE_LOCATION_GROUP_COUNT, MAXWELL_SAMPLE_LOCATIONS_PER_GROUP,
    MaxwellThreeDAlphaToCoverageDither, MaxwellThreeDAlphaToCoverageOverride,
    MaxwellThreeDCoverageState, MaxwellThreeDCoverageToColor, MaxwellThreeDCsaaEnable,
    MaxwellThreeDHybridAntiAliasCentroid, MaxwellThreeDHybridAntiAliasControl,
    MaxwellThreeDPostZPixelShaderImask, MaxwellThreeDPsOutputSampleMaskUsage,
    MaxwellThreeDSampleLocation, MaxwellThreeDSampleLocationGroup, MaxwellThreeDTirControl,
    MaxwellThreeDTirMode, MaxwellThreeDTirModulationComponentSelect,
    MaxwellThreeDTirModulationFunction,
};
pub use inline_to_memory::{
    MaxwellThreeDInlineToMemoryCompletion, MaxwellThreeDInlineToMemoryLaunch,
    MaxwellThreeDInlineToMemoryLayout, MaxwellThreeDInlineToMemoryState,
    MaxwellThreeDInlineToMemoryStateWrite,
};
pub use line::{
    MaxwellThreeDAliasedLineWidthEnable, MaxwellThreeDAntiAliasedLineEnable,
    MaxwellThreeDLineState, MaxwellThreeDLineStippleParameters,
    MaxwellThreeDPolygonClipGeneratedEdge,
};
pub use mme::{
    MAXWELL_THREE_D_MME_EMITTED_METHOD_LIMIT, MAXWELL_THREE_D_MME_EXECUTION_INSTRUCTION_LIMIT,
    MAXWELL_THREE_D_MME_SHADOW_SCRATCH_COUNT, MaxwellThreeDMmeExecutionError,
    MaxwellThreeDMmeInstruction, MaxwellThreeDMmeLoadError, MaxwellThreeDMmeRam,
    MaxwellThreeDMmeRamAddress, MaxwellThreeDMmeShadowRamControl, MaxwellThreeDMmeState,
    MaxwellThreeDMmeStateWrite, MaxwellThreeDMutableMethodControl,
};
pub use operations::{
    MaxwellThreeDDecompressSurface, MaxwellThreeDFlushPendingWrites,
    MaxwellThreeDReportSemaphoreControl, MaxwellThreeDReportSemaphoreOperation,
    MaxwellThreeDReportSemaphorePipelineLocation, MaxwellThreeDReportSemaphoreState,
    MaxwellThreeDReportSemaphoreStructureSize, MaxwellThreeDReportSemaphoreWrite,
    MaxwellThreeDShaderCacheInvalidation, MaxwellThreeDSynchronizationError,
    MaxwellThreeDSynchronizationPlan, MaxwellThreeDSynchronizationTrigger,
    MaxwellThreeDSyncpointCondition, MaxwellThreeDSyncpointIncrement,
    MaxwellThreeDTextureCacheInvalidation, MaxwellThreeDTextureCacheLines,
    MaxwellThreeDTextureCacheTarget, MaxwellThreeDTiledCacheFlushMode,
    lower_maxwell_three_d_synchronization,
};

pub use output::{
    MAXWELL_SCISSOR_COUNT, MAXWELL_VIEWPORT_COUNT, MAXWELL_WINDOW_CLIP_COUNT,
    MaxwellThreeDBlendControlState, MaxwellThreeDBlendFloatPixelKillEnable,
    MaxwellThreeDBlendPerFormatEnable, MaxwellThreeDBlendZeroTimesAnythingIsZero,
    MaxwellThreeDClipAxis, MaxwellThreeDClipIdTestEnable, MaxwellThreeDCompareOp,
    MaxwellThreeDCullFace, MaxwellThreeDFixedFunctionRegister, MaxwellThreeDFixedFunctionState,
    MaxwellThreeDFixedFunctionValue, MaxwellThreeDFrontFace, MaxwellThreeDIteratedBlend,
    MaxwellThreeDIteratedBlendPassCount, MaxwellThreeDLogicOp, MaxwellThreeDPixelShaderClampRange,
    MaxwellThreeDPixelShaderSaturate, MaxwellThreeDPolygonMode, MaxwellThreeDProvokingVertex,
    MaxwellThreeDSampleMode, MaxwellThreeDScissorState, MaxwellThreeDShadeMode,
    MaxwellThreeDStencilOp, MaxwellThreeDViewportClipControl,
    MaxwellThreeDViewportCoordinateSwizzle, MaxwellThreeDViewportScaleOffsetEnable,
    MaxwellThreeDViewportSwizzleComponent, MaxwellThreeDViewportTransformState,
    MaxwellThreeDWindowClipState, MaxwellThreeDWindowClipType,
};
pub use render_enable::{
    MaxwellThreeDConditionalLoadConstantBuffer, MaxwellThreeDRenderEnableMode,
    MaxwellThreeDRenderEnableOverride, MaxwellThreeDRenderEnableState,
};
pub use render_targets::{
    MAXWELL_COLOR_TARGET_COUNT, MaxwellThreeDAttachmentReadiness, MaxwellThreeDClearState,
    MaxwellThreeDClearSurface, MaxwellThreeDColorCompressionMode, MaxwellThreeDColorTargetFormat,
    MaxwellThreeDColorTargetState, MaxwellThreeDCompressionThreshold,
    MaxwellThreeDDepthArrayControl, MaxwellThreeDDepthStencilFormat,
    MaxwellThreeDDepthStencilTargetState, MaxwellThreeDDepthTargetCount, MaxwellThreeDImageKind,
    MaxwellThreeDImageLayout, MaxwellThreeDRawValue, MaxwellThreeDRectangle,
    MaxwellThreeDRenderTargetIndexOffset, MaxwellThreeDRenderTargetLayer,
    MaxwellThreeDRenderTargetLayerControl, MaxwellThreeDRenderTargetState,
    MaxwellThreeDSeparateFragmentData, MaxwellThreeDZCompressionMode,
};
pub use resource::texture::sampled_image_info;
pub use resource::{
    AttachmentInfo, ImageInfo, attachment_description, blit_image_infos, decode_sampler,
    render_target_info, solid_image_info,
};
pub use resource::{
    MaxwellThreeDGuestImageFormat, MaxwellThreeDResolvedSampler, MaxwellThreeDResourceError,
    MaxwellThreeDResourceRole, MaxwellThreeDTextureDimension, MaxwellThreeDTextureReference,
};
pub use shader_execution::{
    MAXWELL_THREE_D_SHADER_LOCAL_MEMORY_PER_WARP_SIZE_MAX,
    MAXWELL_THREE_D_SM_TIMEOUT_COUNTER_BIT_MAX, MaxwellThreeDApiMandatedEarlyZ,
    MaxwellThreeDDirectlyAddressableMemory, MaxwellThreeDPixelShaderInterlockControl,
    MaxwellThreeDPixelShaderInterlockFragmentOrder, MaxwellThreeDPixelShaderInterlockMode,
    MaxwellThreeDPixelShaderInterlockTileSize, MaxwellThreeDShaderExceptionsEnable,
    MaxwellThreeDShaderExecutionState, MaxwellThreeDShaderLocalMemoryPerWarpSize,
    MaxwellThreeDShaderLocalMemoryState, MaxwellThreeDShaderWatermarkRange,
    MaxwellThreeDShaderWatermarkTarget, MaxwellThreeDSmTimeoutCounterBit,
    MaxwellThreeDSubtilingPerfKnobA, MaxwellThreeDSubtilingPerfKnobB,
    MaxwellThreeDVisibleCallLimit,
};
pub use state::MaxwellThreeDEngineState;
pub use state::{
    MAXWELL_POLYGON_STIPPLE_PATTERN_WORD_COUNT, MaxwellThreeDAlphaFraction,
    MaxwellThreeDAttributePointSize, MaxwellThreeDConservativeRasterEnable, MaxwellThreeDEdgeFlag,
    MaxwellThreeDFillViaTriangleMode, MaxwellThreeDPointCenterMode, MaxwellThreeDPointSize,
    MaxwellThreeDPointSpriteOrigin, MaxwellThreeDPointSpriteRMode, MaxwellThreeDPointSpriteSelect,
    MaxwellThreeDRasterBoundingBox, MaxwellThreeDRasterBoundingBoxMode, MaxwellThreeDRasterState,
    MaxwellThreeDRegister, MaxwellThreeDState, MaxwellThreeDViewportPixelCenter,
    MaxwellThreeDViewportState, MaxwellThreeDViewportZClipRange,
};
pub use vertex::{
    MAXWELL_THREE_D_PRIMITIVE_AREA_MAX, MAXWELL_VERTEX_ATTRIBUTE_COUNT,
    MAXWELL_VERTEX_STREAM_COUNT, MaxwellThreeDAttributeDefaultVector,
    MaxwellThreeDAttributeDefaults, MaxwellThreeDBalancedPrimitiveWorkload, MaxwellThreeDBegin,
    MaxwellThreeDBeginInstance, MaxwellThreeDPatchSize,
    MaxwellThreeDPrimitiveCircularBufferThrottle, MaxwellThreeDPrimitiveState,
    MaxwellThreeDPrimitiveTopology, MaxwellThreeDUnresolvedAddress,
    MaxwellThreeDVertexArrayPrimitiveRestartEnable, MaxwellThreeDVertexAssemblyState,
    MaxwellThreeDVertexAttributeFormat, MaxwellThreeDVertexComponentWidths,
    MaxwellThreeDVertexIdUsesArrayStart, MaxwellThreeDVertexInputState,
    MaxwellThreeDVertexNumericalType, MaxwellThreeDVertexStreamFormat,
    MaxwellThreeDVertexStreamState, MaxwellThreeDVertexStreamSubstituteState,
};

use nixe_gpu::{GpuClassId, GpuMethodId};

use super::{
    MaxwellEngineCapability, MaxwellEngineDispatchError, MaxwellEngineMethodMetadata,
    MaxwellEngineStreamError, MaxwellInlineToMemoryUpload, MaxwellShaderCacheInvalidation,
};
use crate::{
    MaxwellAamVersion, MaxwellAamVersionRange, MaxwellGpuProfile,
    MaxwellShaderProgramHeaderVersion, MaxwellShaderProgramHeaderVersionRange,
};

pub(super) const CLASS: GpuClassId = GpuClassId(0xb197);
const CLASS_NAME: &str = "MAXWELL_B";

#[derive(Clone, Copy)]
enum MethodAction {
    NotifyAddress,
    Notify,
    NoOperation,
    Delay,
    VpcPerfKnob,
    DecompressSurface,
    WaitForIdle,
    PixelShaderBarrier,
    TiledCacheBarrier,
    DiscardRenderTarget,
    MmeShadowRamControl,
    MutableMethodControl,
    FalconFirmwareCall4,
    InstrumentationHeader,
    InstrumentationData,
    IteratedBlend,
    IteratedBlendPassCount,
    SubtilingPerfKnobA,
    SubtilingPerfKnobB,
    TiledCacheEnable,
    TiledCacheTileSize,
    TiledCacheUnknownConfig,
    TiledCacheFlush,
    ShaderWatermarks,
    L1Configuration,
    ColorReductionThresholdsEnable,
    ColorReductionThresholdsUnorm8,
    ColorReductionThresholdsUnorm10,
    ColorReductionThresholdsUnorm16,
    ColorReductionThresholdsFp16,
    ColorReductionThresholdsSrgb8,
    OpportunisticEarlyZHysteresis,
    ApiMandatedEarlyZ,
    PostPsInitialCoverage,
    PostZPixelShaderImask,
    PixelShaderInterlockControl,
    ConstantColorRenderingEnable,
    ConstantColorRenderingComponent,
    AlphaFraction,
    TirMode,
    TirControl,
    TirModulation,
    TirModulationFunction,
    CoverageToColor,
    AlphaToCoverageOverride,
    AlphaToCoverageDither,
    HybridAntiAliasControl,
    SampleLocations,
    RasterBoundingBox,
    CheckSphVersion,
    CheckAamVersion,
    VafL2CacheControl,
    RopL2CacheControl,
    ReportSemaphoreAddressUpper,
    ReportSemaphoreAddressLower,
    ReportSemaphorePayload,
    ReportSemaphoreTrigger,
    PointSize,
    PointSpriteSelect,
    PointCenterMode,
    EdgeFlag,
    InvalidateShaderCaches,
    InvalidateShaderCachesNoWfi,
    InvalidateTextureCacheNoWfi(MaxwellThreeDTextureCacheTarget),
    InvalidateTextureCache(MaxwellThreeDTextureCacheTarget),
    FlushPendingWrites,
    IncrementSyncpoint,
    ViewportZClip,
    RenderEnableControl,
    RenderEnableAddressUpper,
    RenderEnableAddressLower,
    RenderEnableMode,
    RenderEnableOverride,
    PsOutputSampleMaskUsage,
    VisibleCallLimit,
    SmTimeoutCounterBit,
    ShaderExceptionsEnable,
    ShaderLocalMemoryWindowBaseAddress,
    ShaderLocalMemoryAddressUpper,
    ShaderLocalMemoryAddressLower,
    ShaderLocalMemorySizeUpper,
    ShaderLocalMemorySizeLower,
    ShaderLocalMemoryDefaultSizePerWarp,
    CsaaEnable,
    AliasedLineWidthEnable,
    ActiveZCullRegion,
    ZCullStorage,
    ZCullSubregionAllocation,
    ZCullSubregionAlgorithm,
    ZCullReportSelection,
    ZCullReportType,
    ZCullRegionLocation,
    ZCullRegionAliquots,
    ZCullRegionFormat,
    ZCullRegionSize,
    ZCullRegionPixelOffset,
    ZCullSubregion,
    ZCullDirectionFormat,
    ZCullMaintenance,
    ClearReportValue,
    ZCullStatsEnable,
    ZPassPixelCountEnable,
    StatisticsCounterEnable,
    StreamOutputEnable,
    ZCullCriterion,
    ZCullEnable,
    ZCullBounds,
    VertexCount,
    IndexCount,
    Missing(MaxwellEngineCapability),
}

#[derive(Clone, Copy)]
struct MethodDeclaration {
    metadata: &'static MaxwellEngineMethodMetadata,
    defined_mask: u32,
    action: MethodAction,
}

macro_rules! methods {
    ($($identifier:ident => ($method:literal, $name:literal, $mask:expr, $action:expr)),+ $(,)?) => {
        $(const $identifier: MaxwellEngineMethodMetadata = MaxwellEngineMethodMetadata::new(
            CLASS,
            CLASS_NAME,
            GpuMethodId($method),
            $name,
        );)+
        const METHODS: &[MethodDeclaration] = &[
            $(MethodDeclaration {
                metadata: &$identifier,
                defined_mask: $mask,
                action: $action,
            }),+
        ];
        static METHODS_BY_DWORD: [Option<MethodDeclaration>; 4096] = {
            let mut table = [None; 4096];
            let mut index = 0;
            while index < METHODS.len() {
                let declaration = METHODS[index];
                let method = declaration.metadata.method().0 as usize;
                assert!(method % 4 == 0 && method / 4 < table.len());
                assert!(table[method / 4].is_none());
                table[method / 4] = Some(declaration);
                index += 1;
            }
            table
        };
    };
}

// Class, method, field, and enum values are pinned to NVIDIA's generated
// public MAXWELL_B header. Register-bank initialization is defined in state.rs.
// https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h
methods!(
    NO_OPERATION => (0x0100, "NO_OPERATION", u32::MAX, MethodAction::NoOperation),
    DECOMPRESS_SURFACE => (
        0x02e0,
        "DECOMPRESS_SURFACE",
        0x000f_fff7,
        MethodAction::DecompressSurface
    ),
    PIPE_NOP => (0x1a2c, "PIPE_NOP", u32::MAX, MethodAction::NoOperation),
    DELAY => (0x1a24, "DELAY", u32::MAX, MethodAction::Delay),
    SET_VPC_PERF_KNOB => (0x0f14, "SET_VPC_PERF_KNOB", u32::MAX, MethodAction::VpcPerfKnob),
    SET_INSTRUMENTATION_METHOD_HEADER => (
        0x0150,
        "SET_INSTRUMENTATION_METHOD_HEADER",
        u32::MAX,
        MethodAction::InstrumentationHeader
    ),
    SET_INSTRUMENTATION_METHOD_DATA => (
        0x0154,
        "SET_INSTRUMENTATION_METHOD_DATA",
        u32::MAX,
        MethodAction::InstrumentationData
    ),
    SET_SUBTILING_PERF_KNOB_A => (
        0x0360,
        "SET_SUBTILING_PERF_KNOB_A",
        u32::MAX,
        MethodAction::SubtilingPerfKnobA
    ),
    SET_SUBTILING_PERF_KNOB_B => (
        0x0364,
        "SET_SUBTILING_PERF_KNOB_B",
        0x0000_00ff,
        MethodAction::SubtilingPerfKnobB
    ),
    SET_CONSTANT_COLOR_RENDERING => (
        0x0f40,
        "SET_CONSTANT_COLOR_RENDERING",
        0x0000_0001,
        MethodAction::ConstantColorRenderingEnable
    ),
    SET_CONSTANT_COLOR_RENDERING_RED => (
        0x0f44,
        "SET_CONSTANT_COLOR_RENDERING_RED",
        u32::MAX,
        MethodAction::ConstantColorRenderingComponent
    ),
    SET_CONSTANT_COLOR_RENDERING_GREEN => (
        0x0f48,
        "SET_CONSTANT_COLOR_RENDERING_GREEN",
        u32::MAX,
        MethodAction::ConstantColorRenderingComponent
    ),
    SET_CONSTANT_COLOR_RENDERING_BLUE => (
        0x0f4c,
        "SET_CONSTANT_COLOR_RENDERING_BLUE",
        u32::MAX,
        MethodAction::ConstantColorRenderingComponent
    ),
    SET_CONSTANT_COLOR_RENDERING_ALPHA => (
        0x0f50,
        "SET_CONSTANT_COLOR_RENDERING_ALPHA",
        u32::MAX,
        MethodAction::ConstantColorRenderingComponent
    ),
    SET_OPPORTUNISTIC_EARLY_Z_HYSTERESIS => (0x0204, "SET_OPPORTUNISTIC_EARLY_Z_HYSTERESIS", 0x1f, MethodAction::OpportunisticEarlyZHysteresis),
    SET_API_MANDATED_EARLY_Z => (
        0x0210,
        "SET_API_MANDATED_EARLY_Z",
        0x0000_0001,
        MethodAction::ApiMandatedEarlyZ
    ),
    SET_POST_PS_INITIAL_COVERAGE => (0x1138, "SET_POST_PS_INITIAL_COVERAGE", 1, MethodAction::PostPsInitialCoverage),
    SET_POST_Z_PS_IMASK => (
        0x0f1c,
        "SET_POST_Z_PS_IMASK",
        0x0000_0001,
        MethodAction::PostZPixelShaderImask
    ),
    SET_PIXEL_SHADER_INTERLOCK_CONTROL => (
        0x1224,
        "SET_PIXEL_SHADER_INTERLOCK_CONTROL",
        0x0000_000f,
        MethodAction::PixelShaderInterlockControl
    ),
    SET_TILED_CACHE_ENABLE => (
        0x0f60,
        "SET_TILED_CACHE_ENABLE",
        0x0000_0001,
        MethodAction::TiledCacheEnable
    ),
    SET_TILED_CACHE_TILE_SIZE => (
        0x0f64,
        "SET_TILED_CACHE_TILE_SIZE",
        u32::MAX,
        MethodAction::TiledCacheTileSize
    ),
    SET_TILED_CACHE_UNKNOWN_CONFIG_0 => (
        0x0f68,
        "SET_TILED_CACHE_UNKNOWN_CONFIG_0",
        u32::MAX,
        MethodAction::TiledCacheUnknownConfig
    ),
    SET_TILED_CACHE_UNKNOWN_CONFIG_1 => (
        0x0f6c,
        "SET_TILED_CACHE_UNKNOWN_CONFIG_1",
        u32::MAX,
        MethodAction::TiledCacheUnknownConfig
    ),
    SET_TILED_CACHE_UNKNOWN_CONFIG_2 => (
        0x1108,
        "SET_TILED_CACHE_UNKNOWN_CONFIG_2",
        u32::MAX,
        MethodAction::TiledCacheUnknownConfig
    ),
    SET_TILED_CACHE_UNKNOWN_CONFIG_3 => (
        0x0f70,
        "SET_TILED_CACHE_UNKNOWN_CONFIG_3",
        u32::MAX,
        MethodAction::TiledCacheUnknownConfig
    ),
    // NVIDIA's public class header omits this trigger; the pinned deko3d
    // definition documents method 0x0f80 and both allocated values.
    // https://github.com/devkitPro/deko3d/blob/350f2b00a3e76ecd4f00191f8c5d6544ffbcb9db/source/maxwell/engine_3d.def#L255-L258
    TILED_CACHE_FLUSH => (
        0x0f80,
        "TILED_CACHE_FLUSH",
        0x0000_0001,
        MethodAction::TiledCacheFlush
    ),
    SET_MUTABLE_METHOD_CONTROL => (
        0x1134,
        "SET_MUTABLE_METHOD_CONTROL",
        0x0000_0001,
        MethodAction::MutableMethodControl
    ),
    SET_VTG_WARP_WATERMARKS => (
        0x0f98,
        "SET_VTG_WARP_WATERMARKS",
        u32::MAX,
        MethodAction::ShaderWatermarks
    ),
    SET_PS_WARP_WATERMARKS => (
        0x1450,
        "SET_PS_WARP_WATERMARKS",
        u32::MAX,
        MethodAction::ShaderWatermarks
    ),
    SET_PS_REGISTER_WATERMARKS => (
        0x1454,
        "SET_PS_REGISTER_WATERMARKS",
        u32::MAX,
        MethodAction::ShaderWatermarks
    ),
    SET_L2_CACHE_CONTROL_FOR_VAF_REQUESTS => (
        0x1000,
        "SET_L2_CACHE_CONTROL_FOR_VAF_REQUESTS",
        0x0000_0031,
        MethodAction::VafL2CacheControl
    ),
    INCREMENT_SYNC_POINT => (
        0x02c8,
        "INCREMENT_SYNC_POINT",
        0x0011_0fff,
        MethodAction::IncrementSyncpoint
    ),
    SET_L1_CONFIGURATION => (
        0x0308,
        "SET_L1_CONFIGURATION",
        0x0000_0007,
        MethodAction::L1Configuration
    ),
    SET_RENDER_ENABLE_CONTROL => (
        0x030c,
        "SET_RENDER_ENABLE_CONTROL",
        0x0000_0001,
        MethodAction::RenderEnableControl
    ),
    SET_PS_OUTPUT_SAMPLE_MASK_USAGE => (
        0x0300,
        "SET_PS_OUTPUT_SAMPLE_MASK_USAGE",
        0x0000_0003,
        MethodAction::PsOutputSampleMaskUsage
    ),
    SET_REDUCE_COLOR_THRESHOLDS_ENABLE => (
        0x0d9c,
        "SET_REDUCE_COLOR_THRESHOLDS_ENABLE",
        0x0000_0001,
        MethodAction::ColorReductionThresholdsEnable
    ),
    INVALIDATE_SHADER_CACHES_NO_WFI => (
        0x0da4,
        "INVALIDATE_SHADER_CACHES_NO_WFI",
        0x0000_1011,
        MethodAction::InvalidateShaderCachesNoWfi
    ),
    INVALIDATE_SHADER_CACHES => (
        0x021c,
        "INVALIDATE_SHADER_CACHES",
        0x0000_1017,
        MethodAction::InvalidateShaderCaches
    ),
    SET_REDUCE_COLOR_THRESHOLDS_UNORM8 => (
        0x10cc,
        "SET_REDUCE_COLOR_THRESHOLDS_UNORM8",
        0x00ff_00ff,
        MethodAction::ColorReductionThresholdsUnorm8
    ),
    SET_REDUCE_COLOR_THRESHOLDS_UNORM10 => (
        0x10e0,
        "SET_REDUCE_COLOR_THRESHOLDS_UNORM10",
        0x00ff_00ff,
        MethodAction::ColorReductionThresholdsUnorm10
    ),
    SET_REDUCE_COLOR_THRESHOLDS_UNORM16 => (
        0x10e4,
        "SET_REDUCE_COLOR_THRESHOLDS_UNORM16",
        0x00ff_00ff,
        MethodAction::ColorReductionThresholdsUnorm16
    ),
    SET_REDUCE_COLOR_THRESHOLDS_FP16 => (
        0x10ec,
        "SET_REDUCE_COLOR_THRESHOLDS_FP16",
        0x00ff_00ff,
        MethodAction::ColorReductionThresholdsFp16
    ),
    SET_REDUCE_COLOR_THRESHOLDS_SRGB8 => (
        0x10f0,
        "SET_REDUCE_COLOR_THRESHOLDS_SRGB8",
        0x00ff_00ff,
        MethodAction::ColorReductionThresholdsSrgb8
    ),
    SET_ALPHA_FRACTION => (
        0x074c,
        "SET_ALPHA_FRACTION",
        0x0000_00ff,
        MethodAction::AlphaFraction
    ),
    SET_HYBRID_ANTI_ALIAS_CONTROL => (
        0x0754,
        "SET_HYBRID_ANTI_ALIAS_CONTROL",
        0x0000_003f,
        MethodAction::HybridAntiAliasControl
    ),
    SET_TIR => (
        0x0fb4,
        "SET_TIR",
        0x0000_0003,
        MethodAction::TirMode
    ),
    SET_TIR_MODULATION => (
        0x0fd4,
        "SET_TIR_MODULATION",
        0x0000_0003,
        MethodAction::TirModulation
    ),
    SET_TIR_MODULATION_FUNCTION => (
        0x0fd8,
        "SET_TIR_MODULATION_FUNCTION",
        0x0000_0001,
        MethodAction::TirModulationFunction
    ),
    SET_COVERAGE_TO_COLOR => (
        0x11f8,
        "SET_COVERAGE_TO_COLOR",
        0x0000_0071,
        MethodAction::CoverageToColor
    ),
    SET_ALPHA_TO_COVERAGE_OVERRIDE => (
        0x16b4,
        "SET_ALPHA_TO_COVERAGE_OVERRIDE",
        0x0000_0003,
        MethodAction::AlphaToCoverageOverride
    ),
    SET_ALPHA_TO_COVERAGE_DITHER_CONTROL => (
        0x12e0,
        "SET_ALPHA_TO_COVERAGE_DITHER_CONTROL",
        0x0000_000f,
        MethodAction::AlphaToCoverageDither
    ),
    SET_TIR_CONTROL => (
        0x1130,
        "SET_TIR_CONTROL",
        0x0000_0013,
        MethodAction::TirControl
    ),
    SAMPLE_LOCATIONS_0 => (
        0x11e0,
        "SAMPLE_LOCATIONS(0)",
        u32::MAX,
        MethodAction::SampleLocations
    ),
    SAMPLE_LOCATIONS_1 => (
        0x11e4,
        "SAMPLE_LOCATIONS(1)",
        u32::MAX,
        MethodAction::SampleLocations
    ),
    SAMPLE_LOCATIONS_2 => (
        0x11e8,
        "SAMPLE_LOCATIONS(2)",
        u32::MAX,
        MethodAction::SampleLocations
    ),
    SAMPLE_LOCATIONS_3 => (
        0x11ec,
        "SAMPLE_LOCATIONS(3)",
        u32::MAX,
        MethodAction::SampleLocations
    ),
    SET_RASTER_BOUNDING_BOX => (
        0x02ec,
        "SET_RASTER_BOUNDING_BOX",
        0x0000_0ff1,
        MethodAction::RasterBoundingBox
    ),
    SET_SHADER_LOCAL_MEMORY_WINDOW => (
        0x077c,
        "SET_SHADER_LOCAL_MEMORY_WINDOW",
        u32::MAX,
        MethodAction::ShaderLocalMemoryWindowBaseAddress
    ),
    SET_SHADER_LOCAL_MEMORY_A => (
        0x0790,
        "SET_SHADER_LOCAL_MEMORY_A",
        0x0000_00ff,
        MethodAction::ShaderLocalMemoryAddressUpper
    ),
    SET_SHADER_LOCAL_MEMORY_B => (
        0x0794,
        "SET_SHADER_LOCAL_MEMORY_B",
        u32::MAX,
        MethodAction::ShaderLocalMemoryAddressLower
    ),
    SET_SHADER_LOCAL_MEMORY_C => (
        0x0798,
        "SET_SHADER_LOCAL_MEMORY_C",
        0x0000_003f,
        MethodAction::ShaderLocalMemorySizeUpper
    ),
    SET_SHADER_LOCAL_MEMORY_D => (
        0x079c,
        "SET_SHADER_LOCAL_MEMORY_D",
        u32::MAX,
        MethodAction::ShaderLocalMemorySizeLower
    ),
    SET_SHADER_LOCAL_MEMORY_E => (
        0x07a0,
        "SET_SHADER_LOCAL_MEMORY_E",
        MAXWELL_THREE_D_SHADER_LOCAL_MEMORY_PER_WARP_SIZE_MAX,
        MethodAction::ShaderLocalMemoryDefaultSizePerWarp
    ),
    SET_SHADER_EXCEPTIONS => (
        0x1528,
        "SET_SHADER_EXCEPTIONS",
        0x0000_0001,
        MethodAction::ShaderExceptionsEnable
    ),
    CHECK_SPH_VERSION => (
        0x16a8,
        "CHECK_SPH_VERSION",
        u32::MAX,
        MethodAction::CheckSphVersion
    ),
    CHECK_AAM_VERSION => (
        0x1794,
        "CHECK_AAM_VERSION",
        u32::MAX,
        MethodAction::CheckAamVersion
    ),
    SET_L2_CACHE_CONTROL_FOR_ROP_PREFETCH_READ_REQUESTS => (
        0x0218,
        "SET_L2_CACHE_CONTROL_FOR_ROP_PREFETCH_READ_REQUESTS",
        0x0000_0030,
        MethodAction::RopL2CacheControl
    ),
    SET_L2_CACHE_CONTROL_FOR_ROP_NONINTERLOCKED_READ_REQUESTS => (
        0x10fc,
        "SET_L2_CACHE_CONTROL_FOR_ROP_NONINTERLOCKED_READ_REQUESTS",
        0x0000_0030,
        MethodAction::RopL2CacheControl
    ),
    SET_L2_CACHE_CONTROL_FOR_ROP_INTERLOCKED_READ_REQUESTS => (
        0x1290,
        "SET_L2_CACHE_CONTROL_FOR_ROP_INTERLOCKED_READ_REQUESTS",
        0x0000_0030,
        MethodAction::RopL2CacheControl
    ),
    INVALIDATE_TEXTURE_DATA_CACHE_NO_WFI => (
        0x1288,
        "INVALIDATE_TEXTURE_DATA_CACHE_NO_WFI",
        0x03ff_fff1,
        MethodAction::InvalidateTextureCacheNoWfi(MaxwellThreeDTextureCacheTarget::Data)
    ),
    INVALIDATE_SAMPLER_CACHE_NO_WFI => (
        0x1424,
        "INVALIDATE_SAMPLER_CACHE_NO_WFI",
        0x03ff_fff1,
        MethodAction::InvalidateTextureCacheNoWfi(MaxwellThreeDTextureCacheTarget::Sampler)
    ),
    INVALIDATE_TEXTURE_HEADER_CACHE_NO_WFI => (
        0x1428,
        "INVALIDATE_TEXTURE_HEADER_CACHE_NO_WFI",
        0x03ff_fff1,
        MethodAction::InvalidateTextureCacheNoWfi(MaxwellThreeDTextureCacheTarget::Header)
    ),
    INVALIDATE_SAMPLER_CACHE => (
        0x1330,
        "INVALIDATE_SAMPLER_CACHE",
        0x03ff_fff1,
        MethodAction::InvalidateTextureCache(MaxwellThreeDTextureCacheTarget::Sampler)
    ),
    INVALIDATE_TEXTURE_HEADER_CACHE => (
        0x1334,
        "INVALIDATE_TEXTURE_HEADER_CACHE",
        0x03ff_fff1,
        MethodAction::InvalidateTextureCache(MaxwellThreeDTextureCacheTarget::Header)
    ),
    INVALIDATE_TEXTURE_DATA_CACHE => (
        0x1338,
        "INVALIDATE_TEXTURE_DATA_CACHE",
        0x03ff_fff1,
        MethodAction::InvalidateTextureCache(MaxwellThreeDTextureCacheTarget::Data)
    ),
    // deko3d's waiting data-cache invalidation, used after tiled/fragment barriers.
    // https://github.com/devkitPro/deko3d/blob/350f2b00a3e76ecd4f00191f8c5d6544ffbcb9db/source/maxwell/engine_3d.def#L242-L249
    INVALIDATE_TEXTURE_DATA_CACHE_TILED => (
        0x0f74,
        "INVALIDATE_TEXTURE_DATA_CACHE_TILED",
        0x03ff_fff1,
        MethodAction::InvalidateTextureCache(MaxwellThreeDTextureCacheTarget::Data)
    ),
    SET_L2_CACHE_CONTROL_FOR_ROP_NONINTERLOCKED_WRITE_REQUESTS => (
        0x12d8,
        "SET_L2_CACHE_CONTROL_FOR_ROP_NONINTERLOCKED_WRITE_REQUESTS",
        0x0000_0030,
        MethodAction::RopL2CacheControl
    ),
    SET_L2_CACHE_CONTROL_FOR_ROP_INTERLOCKED_WRITE_REQUESTS => (
        0x12dc,
        "SET_L2_CACHE_CONTROL_FOR_ROP_INTERLOCKED_WRITE_REQUESTS",
        0x0000_0030,
        MethodAction::RopL2CacheControl
    ),
    SET_REPORT_SEMAPHORE_A => (
        0x1b00,
        "SET_REPORT_SEMAPHORE_A",
        0x0000_00ff,
        MethodAction::ReportSemaphoreAddressUpper
    ),
    SET_REPORT_SEMAPHORE_B => (
        0x1b04,
        "SET_REPORT_SEMAPHORE_B",
        u32::MAX,
        MethodAction::ReportSemaphoreAddressLower
    ),
    SET_REPORT_SEMAPHORE_C => (
        0x1b08,
        "SET_REPORT_SEMAPHORE_C",
        u32::MAX,
        MethodAction::ReportSemaphorePayload
    ),
    SET_REPORT_SEMAPHORE_D => (
        0x1b0c,
        "SET_REPORT_SEMAPHORE_D",
        0x1fb7_ffff,
        MethodAction::ReportSemaphoreTrigger
    ),
    SET_ALIASED_LINE_WIDTH_ENABLE => (
        0x020c,
        "SET_ALIASED_LINE_WIDTH_ENABLE",
        0x0000_0001,
        MethodAction::AliasedLineWidthEnable
    ),
    SET_NOTIFY_A => (
        0x0104,
        "SET_NOTIFY_A",
        0x0000_00ff,
        MethodAction::NotifyAddress
    ),
    SET_NOTIFY_B => (0x0108, "SET_NOTIFY_B", u32::MAX, MethodAction::NotifyAddress),
    NOTIFY => (0x010c, "NOTIFY", u32::MAX, MethodAction::Notify),
    WAIT_FOR_IDLE => (
        0x0110,
        "WAIT_FOR_IDLE",
        u32::MAX,
        MethodAction::WaitForIdle
    ),
    PIXEL_SHADER_BARRIER => (
        0x0de0,
        "PIXEL_SHADER_BARRIER",
        0x0000_0001,
        MethodAction::PixelShaderBarrier
    ),
    // Only the zero payload is verified by deko3d's DkBarrier_Tiles emission.
    // https://github.com/devkitPro/deko3d/blob/350f2b00a3e76ecd4f00191f8c5d6544ffbcb9db/source/maxwell/gpu_base.cpp#L49-L85
    TILED_CACHE_BARRIER => (
        0x0f7c,
        "TILED_CACHE_BARRIER",
        0,
        MethodAction::TiledCacheBarrier
    ),
    // Bit 0 selects depth/stencil; bits 4..6 select a color target.
    // https://github.com/devkitPro/deko3d/blob/350f2b00a3e76ecd4f00191f8c5d6544ffbcb9db/source/maxwell/engine_3d.def#L250-L253
    DISCARD_RENDER_TARGET => (
        0x0f78,
        "DISCARD_RENDER_TARGET",
        0x0000_0071,
        MethodAction::DiscardRenderTarget
    ),
    SET_FALCON04 => (
        0x2310,
        "SET_FALCON04",
        u32::MAX,
        MethodAction::FalconFirmwareCall4
    ),
    SET_MME_SHADOW_RAM_CONTROL => (
        0x0124,
        "SET_MME_SHADOW_RAM_CONTROL",
        0x0000_0003,
        MethodAction::MmeShadowRamControl
    ),
    DRAW_ZERO_INDEX => (
        0x0304,
        "DRAW_ZERO_INDEX",
        u32::MAX,
        MethodAction::Missing(MaxwellEngineCapability::HostBackend)
    ),
    DRAW_VERTEX_ARRAY => (
        0x0d78,
        "DRAW_VERTEX_ARRAY",
        u32::MAX,
        MethodAction::VertexCount
    ),
    // https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L2830-L2831
    DRAW_INDEX_BUFFER => (
        0x17e0,
        "DRAW_INDEX_BUFFER",
        u32::MAX,
        MethodAction::IndexCount
    ),
    SET_API_VISIBLE_CALL_LIMIT => (
        0x0d64,
        "SET_API_VISIBLE_CALL_LIMIT",
        0x0000_000f,
        MethodAction::VisibleCallLimit
    ),
    SET_SM_TIMEOUT_INTERVAL => (
        0x0de4,
        "SET_SM_TIMEOUT_INTERVAL",
        MAXWELL_THREE_D_SM_TIMEOUT_COUNTER_BIT_MAX,
        MethodAction::SmTimeoutCounterBit
    ),
    SET_VIEWPORT_Z_CLIP => (
        0x0d7c,
        "SET_VIEWPORT_Z_CLIP",
        0x0000_0001,
        MethodAction::ViewportZClip
    ),
    SET_POINT_SIZE => (
        0x1518,
        "SET_POINT_SIZE",
        u32::MAX,
        MethodAction::PointSize
    ),
    SET_POINT_SPRITE_SELECT => (
        0x1604,
        "SET_POINT_SPRITE_SELECT",
        0x0000_1fff,
        MethodAction::PointSpriteSelect
    ),
    SET_POINT_CENTER_MODE => (
        0x165c,
        "SET_POINT_CENTER_MODE",
        0x0000_0001,
        MethodAction::PointCenterMode
    ),
    SET_EDGE_FLAG => (
        0x15e4,
        "SET_EDGE_FLAG",
        0x0000_0001,
        MethodAction::EdgeFlag
    ),
    FLUSH_PENDING_WRITES => (
        0x1144,
        "FLUSH_PENDING_WRITES",
        0x0000_0001,
        MethodAction::FlushPendingWrites
    ),
    SET_ZCULL_STATS => (
        0x151c,
        "SET_ZCULL_STATS",
        0x0000_0001,
        MethodAction::ZCullStatsEnable
    ),
    SET_ZCULL_REGION_LOCATION => (
        0x07e0, "SET_ZCULL_REGION_LOCATION", u32::MAX, MethodAction::ZCullRegionLocation
    ),
    // Hierarchical depth cache backing, separate from the depth attachment.
    // https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L639-L649
    SET_ZCULL_STORAGE_A => (0x07e8, "SET_ZCULL_STORAGE_A", 0xff, MethodAction::ZCullStorage),
    SET_ZCULL_STORAGE_B => (0x07ec, "SET_ZCULL_STORAGE_B", u32::MAX, MethodAction::ZCullStorage),
    SET_ZCULL_STORAGE_C => (0x07f0, "SET_ZCULL_STORAGE_C", 0xff, MethodAction::ZCullStorage),
    SET_ZCULL_STORAGE_D => (0x07f4, "SET_ZCULL_STORAGE_D", u32::MAX, MethodAction::ZCullStorage),
    SET_ZCULL_REGION_ALIQUOTS => (
        0x07e4, "SET_ZCULL_REGION_ALIQUOTS", 0xffff, MethodAction::ZCullRegionAliquots
    ),
    SET_ZCULL_REGION_FORMAT => (
        0x15c8, "SET_ZCULL_REGION_FORMAT", 0xf, MethodAction::ZCullRegionFormat
    ),
    SET_ZCULL_REGION_SIZE_A => (
        0x07c0, "SET_ZCULL_REGION_SIZE_A", 0xffff,
        MethodAction::ZCullRegionSize
    ),
    SET_ZCULL_REGION_SIZE_B => (
        0x07c4, "SET_ZCULL_REGION_SIZE_B", 0xffff,
        MethodAction::ZCullRegionSize
    ),
    SET_ZCULL_REGION_SIZE_C => (
        0x07c8, "SET_ZCULL_REGION_SIZE_C", 0xffff,
        MethodAction::ZCullRegionSize
    ),
    SET_ZCULL_REGION_PIXEL_OFFSET_A => (
        0x15fc, "SET_ZCULL_REGION_PIXEL_OFFSET_A", 0xffff,
        MethodAction::ZCullRegionPixelOffset
    ),
    SET_ZCULL_REGION_PIXEL_OFFSET_B => (
        0x1600, "SET_ZCULL_REGION_PIXEL_OFFSET_B", 0xffff,
        MethodAction::ZCullRegionPixelOffset
    ),
    SET_ZCULL_REGION_PIXEL_OFFSET_C => (
        0x07cc, "SET_ZCULL_REGION_PIXEL_OFFSET_C", 0xffff,
        MethodAction::ZCullRegionPixelOffset
    ),
    // Allocation layout and policy are cache optimization state, not attachments.
    // https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L351-L378
    SET_ZCULL_SUBREGION_TO_REPORT => (0x036c, "SET_ZCULL_SUBREGION_TO_REPORT", 0x0ff1, MethodAction::ZCullReportSelection),
    SET_ZCULL_SUBREGION_REPORT_TYPE => (0x0370, "SET_ZCULL_SUBREGION_REPORT_TYPE", 0x71, MethodAction::ZCullReportType),
    SET_ZCULL_SUBREGION_ALLOCATION => (0x02f8, "SET_ZCULL_SUBREGION_ALLOCATION", 0x0fff_ffff, MethodAction::ZCullSubregionAllocation),
    ASSIGN_ZCULL_SUBREGIONS => (0x02fc, "ASSIGN_ZCULL_SUBREGIONS", 3, MethodAction::ZCullSubregionAlgorithm),
    SET_ZCULL_SUBREGION => (
        0x02e8, "SET_ZCULL_SUBREGION", 0x0fff_fff1, MethodAction::ZCullSubregion
    ),
    SET_ZCULL_DIR_FORMAT => (
        0x0dbc, "SET_ZCULL_DIR_FORMAT", u32::MAX, MethodAction::ZCullDirectionFormat
    ),
    CLEAR_REPORT_VALUE => (0x1530, "CLEAR_REPORT_VALUE", 0x1f, MethodAction::ClearReportValue),
    CLEAR_ZCULL_REGION => (
        0x12c8, "CLEAR_ZCULL_REGION", 0x001f_ffff, MethodAction::ZCullMaintenance
    ),
    // Save/restore only the optional hierarchical culling cache. The neutral
    // renderer keeps ordinary depth testing authoritative and has no Maxwell
    // Z-cull cache to serialize; do not fabricate a hardware cache encoding.
    // Existing emulator semantics also leave these outside executable methods:
    // https://github.com/eden-emulator/mirror/blob/d16735f5b618942136d6ab53466e3be0a382c30a/src/video_core/engines/maxwell_3d.cpp
    // https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L2647-L2659
    STORE_ZCULL => (0x1464, "STORE_ZCULL", 1, MethodAction::ZCullMaintenance),
    LOAD_ZCULL => (0x1500, "LOAD_ZCULL", 1, MethodAction::ZCullMaintenance),
    INVALIDATE_ZCULL => (
        0x1958, "INVALIDATE_ZCULL", 0, MethodAction::ZCullMaintenance
    ),
    SET_STREAM_OUTPUT => (
        0x0744,
        "SET_STREAM_OUTPUT",
        0x0000_0001,
        MethodAction::StreamOutputEnable
    ),
    SET_STATISTICS_COUNTER => (
        0x0d68,
        "SET_STATISTICS_COUNTER",
        0x0000_ffff,
        MethodAction::StatisticsCounterEnable
    ),
    SET_ZPASS_PIXEL_COUNT => (
        0x1514,
        "SET_ZPASS_PIXEL_COUNT",
        0x0000_0001,
        MethodAction::ZPassPixelCountEnable
    ),
    SET_ZCULL_CRITERION => (
        0x0dd8,
        "SET_ZCULL_CRITERION",
        0xffff_03ff,
        MethodAction::ZCullCriterion
    ),
    SET_ITERATED_BLEND => (
        0x0dd0,
        "SET_ITERATED_BLEND",
        0x0000_0003,
        MethodAction::IteratedBlend
    ),
    SET_ITERATED_BLEND_PASS => (
        0x0dd4,
        "SET_ITERATED_BLEND_PASS",
        0x0000_00ff,
        MethodAction::IteratedBlendPassCount
    ),
    SET_ZCULL => (
        0x1968,
        "SET_ZCULL",
        0x0000_0011,
        MethodAction::ZCullEnable
    ),
    SET_ZCULL_BOUNDS => (
        0x196c,
        "SET_ZCULL_BOUNDS",
        0x0000_0011,
        MethodAction::ZCullBounds
    ),
    SET_RENDER_ENABLE_OVERRIDE => (
        0x1944,
        "SET_RENDER_ENABLE_OVERRIDE",
        0x0000_0003,
        MethodAction::RenderEnableOverride
    ),
    SET_RENDER_ENABLE_A => (0x1550, "SET_RENDER_ENABLE_A", 0xff, MethodAction::RenderEnableAddressUpper),
    SET_RENDER_ENABLE_B => (0x1554, "SET_RENDER_ENABLE_B", u32::MAX, MethodAction::RenderEnableAddressLower),
    SET_RENDER_ENABLE_C => (
        0x1558,
        "SET_RENDER_ENABLE_C",
        0x0000_0007,
        MethodAction::RenderEnableMode
    ),
    SET_ACTIVE_ZCULL_REGION => (
        0x1590,
        "SET_ACTIVE_ZCULL_REGION",
        0x0000_003f,
        MethodAction::ActiveZCullRegion
    ),
    CSAA_ENABLE => (
        0x15b4,
        "CSAA_ENABLE",
        0x0000_0001,
        MethodAction::CsaaEnable
    ),
);

pub(crate) const fn is_mme_aperture(method: GpuMethodId) -> bool {
    // Maxwell3D::CallMethod treats every method above MacroRegistersStart as
    // a macro pair. Program selection aliases pairs onto macro_positions.
    // https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/engines/maxwell_3d.cpp#L375-L410
    method.0 >= 0x3800 && method.0 & 3 == 0
}

/// Consume an admitted effect against the preparation owner's canonical bank.
/// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/engines/maxwell_3d.cpp#L388-L413
#[inline]
pub(super) fn call_method<L: crate::MaxwellFrontend, H: crate::MaxwellCommandHost>(
    profile: &MaxwellGpuProfile,
    method: GpuMethodId,
    argument: u32,
    candidate: &mut MaxwellThreeDEngineState,
    execution: &mut crate::execution::ChannelExecution<'_, L, H>,
    origin: impl Fn() -> crate::MaxwellMethodSource,
) -> Result<(), MaxwellEngineStreamError> {
    if method.0 & 3 != 0 || method.0 >= 0x3800 {
        return Err((MaxwellEngineDispatchError::UnknownMethod {
            source: origin(),
            class_name: CLASS_NAME,
        })
        .into());
    }
    if registers::executable(method) || candidate.raw_register_value(method) != Some(argument) {
        draw::before_write(candidate, method, execution)?;
    }
    candidate.write_register(method, argument);
    if !registers::executable(method) {
        if candidate.pending_notification.is_some()
            && let Some(declaration) = METHODS_BY_DWORD[(method.0 / 4) as usize]
        {
            return Err((invalid_encoding(
                origin().emitted_by_mme(method, argument),
                declaration.metadata.method_name(),
                "pending notification requires completion of a method other than NO_OPERATION",
            ))
            .into());
        }
        return Ok(());
    }
    // DrawManager consumes these as raw state, with no execution or provenance
    // record. COUNT still latches indexed drawing on every repeated write.
    match method.0 {
        0x0d74 | 0x17dc | 0x1970 => return Ok(()),
        0x1118 | 0x1434 | 0x1438 if candidate.pending_notification.is_none() => return Ok(()),
        0x0d78 if candidate.pending_notification.is_none() => return Ok(()),
        0x17e0 if candidate.pending_notification.is_none() => {
            draw::index_count(candidate);
            return Ok(());
        }
        _ => {}
    }
    // DrawManager and CBBind consume raw arguments. Construct diagnostic
    // provenance only for an error or an actually deferred/unavailable draw;
    // it does not travel through the rasterizer's synchronous Draw/Clear calls.
    // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/engines/draw_manager.cpp#L16-L85
    let consumed_origin = || origin().emitted_by_mme(method, argument);
    match method.0 {
        0x1618 => {
            if candidate.has_inline_indices() {
                return Err(invalid_encoding(
                    consumed_origin(),
                    "BEGIN",
                    "a pending inline primitive requires END",
                )
                .into());
            }
            let value = MaxwellThreeDBegin::parse(argument).ok_or_else(|| {
                invalid_encoding(
                    consumed_origin(),
                    "BEGIN",
                    "invalid topology or begin modifier",
                )
            })?;
            if value.instance() == MaxwellThreeDBeginInstance::First {
                draw::flush(candidate, execution, false)?;
            }
            draw::begin(candidate, argument, value, consumed_origin);
            return Ok(());
        }
        0x1614 => return draw::end(candidate, argument, execution, consumed_origin),
        0x19d0 => return draw::clear(candidate, argument, execution, consumed_origin),
        0x2400..0x2500 if (method.0 - 0x2400) % 0x20 == 0x10 => {
            bind_constant_buffer(candidate, method, argument, execution, consumed_origin)?;
            return Ok(());
        }
        _ => {}
    }
    let family = classify_family(method.0);
    if family == RegisterFamily::ConstantBuffer {
        let bytes = argument.to_le_bytes();
        process_constant_buffer_data(
            candidate,
            nixe_memory::WriteSource::Bytes(&bytes),
            execution,
            || origin().emitted_by_mme(method, argument),
        )?;
        return Ok(());
    }
    execute_register(profile, origin(), candidate, family, execution)
}

/// Coalesce adjacent scalar sends only when every word has the same valid
/// page/selector boundary. Invalid or clipped writes retain per-word effects
/// and the original failing method. Shadow RAM was already applied on admission.
pub(crate) fn call_constant_buffer_words<
    L: crate::MaxwellFrontend,
    H: crate::MaxwellCommandHost,
>(
    channel: &mut crate::MaxwellGpuChannel,
    source: crate::MaxwellMethodSource,
    methods: &[GpuMethodId],
    arguments: &[u32],
    execution: &mut crate::execution::ChannelExecution<'_, L, H>,
) -> Result<(), MaxwellEngineStreamError> {
    let (profile, state) = channel.three_d_with_profile_mut();
    draw::before_write(state, source.method(), execution)?;
    let r = state.operation_state().register_words();
    let address = (u64::from(r[0x2384 / 4]) << 32) | u64::from(r[0x2388 / 4]);
    let size = r[0x2380 / 4];
    let offset = r[0x238c / 4];
    let bytes = arguments.len() as u64 * 4;
    if arguments.len() == 1
        || r[0x030c / 4] != 0
        || r[0x2384 / 4] > 0xff
        || address == 0
        || offset > size
        || bytes > u64::from(size.saturating_sub(offset))
        || address + u64::from(offset) + bytes > 1_u64 << 40
        || bytes > 4096 - ((address + u64::from(offset)) & 4095)
        || state.pending_notification.is_some()
    {
        for (&method, &argument) in methods.iter().zip(arguments) {
            call_method(profile, method, argument, state, execution, || {
                source.emitted_by_mme(method, argument)
            })?;
        }
        return Ok(());
    }
    let mut data = [0u8; 256];
    for (index, (&method, &argument)) in methods.iter().zip(arguments).enumerate() {
        state.write_register(method, argument);
        data[index * 4..index * 4 + 4].copy_from_slice(&argument.to_le_bytes());
    }
    process_constant_buffer_data(
        state,
        nixe_memory::WriteSource::Bytes(&data[..bytes as usize]),
        execution,
        || source,
    )?;
    Ok(())
}

/// Packet index ports normalize packed lanes into the native UINT32 stream.
/// No state operation, source vector or reserve call is produced per lane.
pub(super) fn inline_index_words(
    candidate: &mut MaxwellThreeDEngineState,
    method: GpuMethodId,
    subchannel: crate::MaxwellPushbufferSubchannel,
    words: crate::MaxwellDmaArguments<'_>,
) -> Result<(), MaxwellEngineDispatchError> {
    candidate.append_index_values(words.source(0, method, subchannel), words.arguments())?;
    let last = words.source(words.len() - 1, method, subchannel);
    candidate.record_raw_register(last);
    Ok(())
}

// A byte-sized routing table avoids probing every specialized register family
// on every write, including methods emitted by MME. Value validation remains at
// the selected consumer. Shadow replay and repeated trigger writes still run.
// NVIDIA's register aperture: https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
enum RegisterFamily {
    Other,
    InlineToMemory,
    ConstantBuffer,
    VertexAndBinding,
}

const fn classify_family(method: u32) -> RegisterFamily {
    match method {
        0x180..=0x1b4 => RegisterFamily::InlineToMemory,
        0x2390..=0x23fc => RegisterFamily::ConstantBuffer,
        0x1120..=0x112c
        | 0x1160..=0x11dc
        | 0x1880..=0x18fc
        | 0x1c00..=0x1dfc
        | 0x1f00..=0x1ffc
        | 0x0320..=0x0338
        | 0x0374
        | 0x0f84
        | 0x0f88
        | 0x02d0
        | 0x0dcc
        | 0x0de8
        | 0x0d74
        | 0x1118
        | 0x1434
        | 0x1438
        | 0x1610
        | 0x1614
        | 0x1618
        | 0x1644
        | 0x1648
        | 0x164c
        | 0x17c8
        | 0x17cc
        | 0x17d0
        | 0x17d4
        | 0x17d8
        | 0x17dc
        | 0x1948
        | 0x1970
        | 0x0310
        | 0x1608
        | 0x160c
        | 0x0f10
        | 0x1234
        | 0x155c
        | 0x1560
        | 0x1564
        | 0x1574
        | 0x1578
        | 0x157c
        | 0x2380
        | 0x2384
        | 0x2388
        | 0x238c
        | 0x2608 => RegisterFamily::VertexAndBinding,
        0x2000..=0x217c if matches!((method - 0x2000) % 0x40, 0 | 4 | 12 | 16) => {
            RegisterFamily::VertexAndBinding
        }
        0x1240..=0x125c => RegisterFamily::VertexAndBinding,
        0x2400..=0x24fc if (method - 0x2400) % 0x20 == 0x10 => RegisterFamily::VertexAndBinding,
        _ => RegisterFamily::Other,
    }
}

fn execute_register<L: crate::MaxwellFrontend, H: crate::MaxwellCommandHost>(
    profile: &MaxwellGpuProfile,
    source: crate::MaxwellMethodSource,
    candidate: &mut MaxwellThreeDEngineState,
    family: RegisterFamily,
    execution: &mut crate::execution::ChannelExecution<'_, L, H>,
) -> Result<(), MaxwellEngineStreamError> {
    // NVIDIA's packed inline ports append low lanes first. END consumes the
    // complete stream, as in the independently reviewed public draw managers:
    // https://github.com/eden-emulator/mirror/blob/d16735f5b618942136d6ab53466e3be0a382c30a/src/video_core/engines/draw_manager.cpp
    if matches!(source.method().0, 0x1300 | 0x15e8 | 0x15ec) {
        candidate.append_inline_indices(source)?;
        return Ok(());
    }
    if family == RegisterFamily::InlineToMemory
        && let Some((write, upload)) =
            inline_to_memory::preflight(source, &candidate.operation_state().inline_to_memory())?
    {
        candidate.apply_inline(write);
        if let Some(upload) = upload {
            execution.inline_to_memory(upload, None)?;
        }
        return Ok(());
    }
    let Some(declaration) = (source.method().0 & 3 == 0)
        .then(|| METHODS_BY_DWORD.get((source.method().0 / 4) as usize))
        .flatten()
        .and_then(Option::as_ref)
    else {
        return Err((MaxwellEngineDispatchError::UnknownMethod {
            source,
            class_name: CLASS_NAME,
        })
        .into());
    };
    if source.argument() & !declaration.defined_mask != 0 {
        return Err((MaxwellEngineDispatchError::InvalidMethodValue {
            source,
            metadata: declaration.metadata,
            defined_mask: declaration.defined_mask,
        })
        .into());
    }
    // The documented notification request completes after the following method.
    // NO_OPERATION is the explicit way to request notification without other work.
    // Other completion-producing methods and the interrupt variant stay unsupported.
    // https://envytools.readthedocs.io/en/latest/hw/graph/intro.html#notify-method
    // https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L33-L40
    if candidate.pending_notification.is_some()
        && !matches!(declaration.action, MethodAction::NoOperation)
    {
        return Err((invalid_encoding(
            source,
            declaration.metadata.method_name(),
            "pending notification requires completion of a method other than NO_OPERATION",
        ))
        .into());
    }
    match declaration.action {
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
        | MethodAction::ConstantColorRenderingEnable
        | MethodAction::ConstantColorRenderingComponent
        | MethodAction::OpportunisticEarlyZHysteresis
        | MethodAction::ApiMandatedEarlyZ
        | MethodAction::PostPsInitialCoverage
        | MethodAction::PostZPixelShaderImask
        | MethodAction::PixelShaderInterlockControl
        | MethodAction::AlphaFraction
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
        | MethodAction::RenderEnableAddressUpper
        | MethodAction::RenderEnableAddressLower
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
        | MethodAction::ActiveZCullRegion
        | MethodAction::ZCullReportSelection
        | MethodAction::ZCullReportType
        | MethodAction::ZCullSubregionAllocation
        | MethodAction::ZCullSubregionAlgorithm
        | MethodAction::ZCullStorage
        | MethodAction::ZCullRegionLocation
        | MethodAction::ZCullRegionAliquots
        | MethodAction::ZCullRegionSize
        | MethodAction::ZCullRegionPixelOffset
        | MethodAction::ZCullRegionFormat
        | MethodAction::ZCullSubregion
        | MethodAction::ZCullDirectionFormat
        | MethodAction::ZCullStatsEnable
        | MethodAction::StreamOutputEnable
        | MethodAction::StatisticsCounterEnable
        | MethodAction::ZPassPixelCountEnable
        | MethodAction::ZCullCriterion
        | MethodAction::ZCullEnable
        | MethodAction::ZCullBounds
        | MethodAction::RenderEnableOverride => {
            unreachable!("passive method entered execution dispatch")
        }
        MethodAction::RenderEnableMode => {
            if MaxwellThreeDRenderEnableMode::parse(source.argument()).is_none() {
                return Err((MaxwellEngineDispatchError::InvalidMethodValue {
                    source,
                    metadata: declaration.metadata,
                    defined_mask: declaration.defined_mask,
                })
                .into());
            }
            {
                execution.render_condition(candidate.operation_state(), source)?;
            }
        }
        MethodAction::Notify => {
            if source.argument() != 0 {
                return Err((MaxwellEngineDispatchError::UnsupportedMethod {
                    source,
                    metadata: declaration.metadata,
                })
                .into());
            }
            let high = candidate.raw_register_value(GpuMethodId(0x0104));
            let low = candidate.raw_register_value(GpuMethodId(0x0108));
            let (Some(high), Some(low)) = (high, low) else {
                return Err((invalid_encoding(
                    source,
                    declaration.metadata.method_name(),
                    "notification address has not been programmed",
                ))
                .into());
            };
            if high > 0xff {
                return Err((invalid_encoding(
                    source,
                    declaration.metadata.method_name(),
                    "notification address exceeds its 40-bit field",
                ))
                .into());
            }
            candidate.pending_notification =
                Some(((u64::from(high) << 32) | u64::from(low), source));
        }
        MethodAction::NoOperation => {
            if let Some((address, source)) = candidate.pending_notification.take() {
                {
                    execution.notification(address, source)?;
                }
            }
        }
        MethodAction::DiscardRenderTarget => {
            // A discard relinquishes preservation of the selected target's
            // contents; it is not a clear, unbind, deallocation, or barrier.
            // Retaining those contents is valid. Our resident-image path
            // keeps them without adding a host pass, readback, resource
            // lookup, or pipeline invalidation. Do not mark an uninitialized
            // compressed image as materialized merely because it was discarded.
            // The caller supplies any required fragment ordering separately.
            // https://github.com/devkitPro/deko3d/blob/350f2b00a3e76ecd4f00191f8c5d6544ffbcb9db/source/maxwell/gpu_3d_base.cpp#L514-L530
            // Discard/invalidation permits retained contents, not a required
            // replacement value: https://registry.khronos.org/OpenGL/specs/gl/glspec43.core.pdf section 17.4.4.
        }
        MethodAction::ClearReportValue => {
            // Eden ProcessCounterReset dispatches Z-pass reset to the query
            // cache. Hierarchical Z-cull statistics have no host counterpart.
            // https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/engines/maxwell_3d.cpp#L569-L585
            match source.argument() {
                1 => execution.reset_samples_passed()?,
                2 => (),
                _ => {
                    return Err((MaxwellEngineDispatchError::InvalidMethodEncoding {
                        source,
                        method_name: "CLEAR_REPORT_VALUE",
                        reason: "resetting this performance counter is not implemented",
                    })
                    .into());
                }
            }
        }

        MethodAction::ZCullMaintenance => {
            // These commands maintain only Maxwell's hierarchical Z-cull
            // metadata, never the depth/stencil attachment. We retain no
            // such cache: neutral draws perform their own depth/stencil
            // tests. All documented selectors therefore require no host
            // work, memory writes, or pipeline invalidation. In particular,
            // deko3d clears conservatively (0x19) after changing targets.
            // https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L2091-L2108
            // https://github.com/devkitPro/deko3d/blob/350f2b00a3e76ecd4f00191f8c5d6544ffbcb9db/source/maxwell/clear_buffers.mme#L26-L37
        }
        MethodAction::DecompressSurface => {
            let request = MaxwellThreeDDecompressSurface::parse(source.argument()).ok_or(
                MaxwellEngineDispatchError::InvalidMethodValue {
                    source,
                    metadata: declaration.metadata,
                    defined_mask: declaration.defined_mask,
                },
            )?;
            {
                execution.three_d_synchronization(
                    candidate.operation_state(),
                    MaxwellThreeDSynchronizationTrigger::DecompressSurface { request, source },
                )?;
            }
        }
        MethodAction::WaitForIdle => {
            execution.three_d_synchronization(
                candidate.operation_state(),
                MaxwellThreeDSynchronizationTrigger::WaitForIdle {
                    value: source.argument(),
                    source,
                },
            )?;
        }
        MethodAction::PixelShaderBarrier => {
            execution.three_d_synchronization(
                candidate.operation_state(),
                MaxwellThreeDSynchronizationTrigger::PixelShaderBarrier {
                    system_memory_barrier: source.argument() & 1 != 0,
                    source,
                },
            )?;
        }
        MethodAction::TiledCacheBarrier => {
            execution.three_d_synchronization(
                candidate.operation_state(),
                MaxwellThreeDSynchronizationTrigger::TiledCacheBarrier { source },
            )?;
        }
        MethodAction::MmeShadowRamControl | MethodAction::MutableMethodControl => {
            unreachable!("macro control belongs to the command processor")
        }
        MethodAction::FalconFirmwareCall4 => {
            // These internal hardware writes do not configure the emulated
            // rasterizer. The guest-visible result is scratch[0] completion.
            // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/engines/maxwell_3d.cpp
            candidate.write_register(GpuMethodId(0x3400), 1);
        }
        MethodAction::TiledCacheFlush => {
            let mode = MaxwellThreeDTiledCacheFlushMode::parse(source.argument()).ok_or(
                MaxwellEngineDispatchError::InvalidMethodValue {
                    source,
                    metadata: declaration.metadata,
                    defined_mask: declaration.defined_mask,
                },
            )?;
            {
                execution.three_d_synchronization(
                    candidate.operation_state(),
                    MaxwellThreeDSynchronizationTrigger::TiledCacheFlush { mode, source },
                )?;
            }
        }
        MethodAction::CheckSphVersion => {
            let requested = MaxwellShaderProgramHeaderVersionRange::new(
                MaxwellShaderProgramHeaderVersion::new(source.argument() as u16),
                MaxwellShaderProgramHeaderVersion::new((source.argument() >> 16) as u16),
            );
            if !requested.is_well_ordered() {
                return Err((MaxwellEngineDispatchError::InvalidMethodEncoding {
                    source,
                    method_name: declaration.metadata.method_name(),
                    reason: "current SPH version precedes oldest supported SPH version",
                })
                .into());
            }
            let supported = profile.shader().sph_versions();
            if !supported.overlaps(requested) {
                return Err(
                    (MaxwellEngineDispatchError::IncompatibleShaderProgramHeaderVersion {
                        source,
                        requested,
                        supported,
                    })
                    .into(),
                );
            }
        }
        MethodAction::CheckAamVersion => {
            let requested = MaxwellAamVersionRange::new(
                MaxwellAamVersion::new(source.argument() as u16),
                MaxwellAamVersion::new((source.argument() >> 16) as u16),
            );
            if !requested.is_well_ordered() {
                return Err((MaxwellEngineDispatchError::InvalidMethodEncoding {
                    source,
                    method_name: declaration.metadata.method_name(),
                    reason: "current AAM version precedes oldest supported AAM version",
                })
                .into());
            }
            let supported = profile.aam_versions();
            if !supported.overlaps(requested) {
                return Err((MaxwellEngineDispatchError::IncompatibleAamVersion {
                    source,
                    requested,
                    supported,
                })
                .into());
            }
        }
        MethodAction::ReportSemaphoreTrigger => {
            let control = MaxwellThreeDReportSemaphoreControl::parse(source.argument())
                .ok_or_else(|| {
                    invalid_encoding(
                        source,
                        "SET_REPORT_SEMAPHORE_D",
                        "control contains an unallocated enum value",
                    )
                })?;
            {
                execution.three_d_synchronization(
                    candidate.operation_state(),
                    MaxwellThreeDSynchronizationTrigger::ReportSemaphore { control, source },
                )?;
            }
        }
        MethodAction::InvalidateShaderCachesNoWfi => {
            let caches = MaxwellShaderCacheInvalidation::new(
                source.argument() & 1 != 0,
                source.argument() & (1 << 4) != 0,
                source.argument() & (1 << 12) != 0,
            );
            {
                execution.three_d_synchronization(
                    candidate.operation_state(),
                    MaxwellThreeDSynchronizationTrigger::InvalidateShaderCachesNoWfi {
                        caches,
                        source,
                    },
                )?;
            }
        }
        MethodAction::InvalidateShaderCaches => {
            let caches = MaxwellShaderCacheInvalidation::new(
                source.argument() & 1 != 0,
                source.argument() & (1 << 4) != 0,
                source.argument() & (1 << 12) != 0,
            );
            let request = MaxwellThreeDShaderCacheInvalidation::new(
                caches,
                source.argument() & (1 << 1) != 0,
                source.argument() & (1 << 2) != 0,
            );
            {
                execution.three_d_synchronization(
                    candidate.operation_state(),
                    MaxwellThreeDSynchronizationTrigger::InvalidateShaderCaches { request, source },
                )?;
            }
        }
        MethodAction::InvalidateTextureCacheNoWfi(target) => {
            let request = MaxwellThreeDTextureCacheInvalidation::new(
                target,
                if source.argument() & 1 == 0 {
                    MaxwellThreeDTextureCacheLines::All
                } else {
                    MaxwellThreeDTextureCacheLines::One
                },
                (source.argument() >> 4) & 0x003f_ffff,
            );
            {
                execution.three_d_synchronization(
                    candidate.operation_state(),
                    MaxwellThreeDSynchronizationTrigger::InvalidateTextureCacheNoWfi {
                        request,
                        source,
                    },
                )?;
            }
        }
        MethodAction::InvalidateTextureCache(target) => {
            let request = MaxwellThreeDTextureCacheInvalidation::new(
                target,
                if source.argument() & 1 == 0 {
                    MaxwellThreeDTextureCacheLines::All
                } else {
                    MaxwellThreeDTextureCacheLines::One
                },
                (source.argument() >> 4) & 0x003f_ffff,
            );
            {
                execution.three_d_synchronization(
                    candidate.operation_state(),
                    MaxwellThreeDSynchronizationTrigger::InvalidateTextureCache { request, source },
                )?;
            }
        }
        MethodAction::FlushPendingWrites => {
            let request = MaxwellThreeDFlushPendingWrites::new(source.argument() != 0);
            {
                execution.three_d_synchronization(
                    candidate.operation_state(),
                    MaxwellThreeDSynchronizationTrigger::FlushPendingWrites { request, source },
                )?;
            }
        }
        MethodAction::IncrementSyncpoint => {
            let request = MaxwellThreeDSyncpointIncrement::new(
                nixe_gpu::GuestSyncpointId::new(source.argument() & 0x0fff),
                source.argument() & (1 << 16) != 0,
                if source.argument() & (1 << 20) == 0 {
                    MaxwellThreeDSyncpointCondition::StreamOutWritesDone
                } else {
                    MaxwellThreeDSyncpointCondition::RopWritesDone
                },
            );
            {
                execution.three_d_synchronization(
                    candidate.operation_state(),
                    MaxwellThreeDSynchronizationTrigger::IncrementSyncpoint { request, source },
                )?;
            }
        }
        MethodAction::VertexCount => (),
        MethodAction::IndexCount => {
            draw::index_count(candidate);
        }
        MethodAction::Missing(capability) => {
            return Err((MaxwellEngineDispatchError::MissingCapability {
                source,
                metadata: declaration.metadata,
                capability,
            })
            .into());
        }
    };
    Ok(())
}

/// ProcessCBData and ProcessCBMultiData write the canonical selector directly
/// through MemoryManager::WriteBlockCached, then advance its offset. DMA packets
/// and single MME words share this synchronous path without upload records.
/// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/engines/maxwell_3d.cpp#L611-L632
pub(in crate::engines) fn process_constant_buffer_data<
    L: crate::MaxwellFrontend,
    H: crate::MaxwellCommandHost,
>(
    candidate: &mut MaxwellThreeDEngineState,
    data: nixe_memory::WriteSource<'_>,
    execution: &mut crate::execution::ChannelExecution<'_, L, H>,
    origin: impl FnOnce() -> crate::MaxwellMethodSource,
) -> Result<usize, MaxwellEngineStreamError> {
    if data.is_empty() {
        return Ok(0);
    }
    let r = candidate.operation_state().register_words();
    if r[0x030c / 4] == MaxwellThreeDConditionalLoadConstantBuffer::Enabled.raw() {
        return Err(
            MaxwellEngineDispatchError::UnsupportedConditionalConstantBufferLoad {
                source: origin(),
            }
            .into(),
        );
    }
    let upper = r[0x2384 / 4];
    if upper > 0xff {
        return Err(invalid_encoding(
            origin(),
            "LOAD_CONSTANT_BUFFER",
            "data load requires a complete selector address",
        )
        .into());
    }
    let address = (u64::from(upper) << 32) | u64::from(r[0x2388 / 4]);
    let size = r[0x2380 / 4];
    let offset = r[0x238c / 4];
    let Some(next_offset) = offset.checked_add(4) else {
        return Err(invalid_encoding(
            origin(),
            "LOAD_CONSTANT_BUFFER",
            "inline upload offset overflows",
        )
        .into());
    };
    if next_offset > size {
        return Err(invalid_encoding(
            origin(),
            "LOAD_CONSTANT_BUFFER",
            "inline upload exceeds the selected constant-buffer size",
        )
        .into());
    }
    if address + u64::from(next_offset) > 1_u64 << 40 {
        return Err(invalid_encoding(
            origin(),
            "LOAD_CONSTANT_BUFFER",
            "inline upload GPU range overflows",
        )
        .into());
    }
    let available = u64::from(size - offset).min((1_u64 << 40) - address - u64::from(offset));
    let count = (data.len() / 4).min((available / 4) as usize);
    let bytes = count * 4;
    let mut memory = execution.address_space.borrow_mut();
    let address = match memory.address(address + u64::from(offset)) {
        Ok(address) => address,
        Err(error) => {
            return Err(crate::MaxwellSubmissionExecutionError::InlineAddress {
                source: origin(),
                error: crate::MaxwellGpuAccessError::Address(error),
            }
            .into());
        }
    };
    if let Err(error) = memory.write_cached(address, data.slice(0..bytes)) {
        return Err(crate::MaxwellSubmissionExecutionError::InlineAddress {
            source: origin(),
            error,
        }
        .into());
    }
    drop(memory);
    candidate.write_register(GpuMethodId(0x238c), offset + bytes as u32);
    Ok(count)
}

fn bind_constant_buffer<L: crate::MaxwellFrontend, H: crate::MaxwellCommandHost>(
    candidate: &mut MaxwellThreeDEngineState,
    method: GpuMethodId,
    raw: u32,
    execution: &mut crate::execution::ChannelExecution<'_, L, H>,
    origin: impl Fn() -> crate::MaxwellMethodSource,
) -> Result<(), MaxwellEngineStreamError> {
    let group = ((method.0 - 0x2400) / 0x20) as u8;
    if raw & !0x1f1 != 0 {
        return Err(invalid_encoding(
            origin(),
            "BIND_GROUP_CONSTANT_BUFFER",
            "undefined constant-buffer binding bits",
        )
        .into());
    }
    let enabled = raw & 1 != 0;
    let slot = ((raw >> 4) & 0x1f) as u8;
    let selector = candidate.operation_state().shader_bindings().selector();
    let binding = if enabled {
        let address = selector.address().ok_or_else(|| {
            invalid_encoding(
                origin(),
                "BIND_GROUP_CONSTANT_BUFFER",
                "enabled binding requires a complete selector address",
            )
        })?;
        let size = *selector.size().value().ok_or_else(|| {
            invalid_encoding(
                origin(),
                "BIND_GROUP_CONSTANT_BUFFER",
                "enabled binding requires selector size",
            )
        })?;
        if size == 0
            || address
                .get()
                .checked_add(u64::from(size))
                .is_none_or(|end| end > (1_u64 << 40))
        {
            return Err(invalid_encoding(
                origin(),
                "BIND_GROUP_CONSTANT_BUFFER",
                "constant-buffer range is empty or overflows",
            )
            .into());
        }
        Some(MaxwellThreeDConstantBufferBinding {
            address,
            size: std::num::NonZeroU32::new(size).expect("validated binding size"),
        })
    } else {
        None
    };
    candidate.bind_constant_buffer(group, slot, binding);
    execution.bind_constant_buffer(candidate.operation_state(), group, slot, origin)?;
    Ok(())
}

fn invalid_encoding(
    source: crate::MaxwellMethodSource,
    method_name: &'static str,
    reason: &'static str,
) -> MaxwellEngineDispatchError {
    MaxwellEngineDispatchError::InvalidMethodEncoding {
        source,
        method_name,
        reason,
    }
}

fn rectangle(raw: u32) -> Option<MaxwellThreeDRectangle> {
    let value = MaxwellThreeDRectangle {
        min: raw as u16,
        max: (raw >> 16) as u16,
    };
    (value.min <= value.max).then_some(value)
}
