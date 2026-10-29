//! GM20B `MAXWELL_B` 3D class methods and register transitions.

mod bindings;
mod color_reduction;
mod coverage;
pub(super) mod draw;
mod falcon;
mod inline_to_memory;
mod line;
mod mme;
mod operations;
mod output;
pub(in crate::engines) mod registers;
mod render_enable;
mod render_targets;
mod resource;
mod shader_execution;
pub(super) mod state;
pub(super) use registers::{constant_buffer_words, embedded_words};
pub(super) mod tessellation;
mod vertex;
pub use tessellation::{MaxwellTessellationModeError, MaxwellThreeDTessellationMode};

pub use super::lowering::{MaxwellLoweringError, MaxwellThreeDOperationTrigger};
pub use bindings::{
    MAXWELL_BIND_GROUP_COUNT, MAXWELL_CONSTANT_BUFFER_SLOT_COUNT, MAXWELL_PIPELINE_SHADER_COUNT,
    MAXWELL_TESSELLATION_LOD_COUNT, MaxwellShaderStage, MaxwellThreeDBindGroupState,
    MaxwellThreeDConstantBufferBinding, MaxwellThreeDConstantBufferSelectorState,
    MaxwellThreeDDescriptorPoolState, MaxwellThreeDInlineConstantBufferUpload,
    MaxwellThreeDPipelineBindingState, MaxwellThreeDProgramRegionState,
    MaxwellThreeDSamplerBindingMode, MaxwellThreeDShaderBindingState,
    MaxwellThreeDShaderBindingWrite, MaxwellThreeDTessellationLod,
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
pub use falcon::{
    MaxwellThreeDFalconError, MaxwellThreeDFalconMaskedRegisterWrite, MaxwellThreeDFalconRegister,
    MaxwellThreeDFalconRegisterAddress, MaxwellThreeDFalconState,
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
    MAXWELL_THREE_D_MME_CAPTURED_INSTRUCTION_WORDS, MAXWELL_THREE_D_MME_CAPTURED_START_ADDRESSES,
    MAXWELL_THREE_D_MME_EMITTED_METHOD_LIMIT, MAXWELL_THREE_D_MME_EXECUTION_INSTRUCTION_LIMIT,
    MAXWELL_THREE_D_MME_SHADOW_SCRATCH_COUNT, MaxwellThreeDMmeExecutionError,
    MaxwellThreeDMmeInstruction, MaxwellThreeDMmeLoadError, MaxwellThreeDMmeRam,
    MaxwellThreeDMmeRamAddress, MaxwellThreeDMmeShadowRamControl, MaxwellThreeDMmeShadowRamError,
    MaxwellThreeDMmeShadowScratchIndex, MaxwellThreeDMmeState, MaxwellThreeDMmeStateWrite,
    MaxwellThreeDMutableMethodControl,
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
    MaxwellThreeDBlendControlState, MaxwellThreeDBlendFactor,
    MaxwellThreeDBlendFloatPixelKillEnable, MaxwellThreeDBlendOp,
    MaxwellThreeDBlendPerFormatEnable, MaxwellThreeDBlendZeroTimesAnythingIsZero,
    MaxwellThreeDClipAxis, MaxwellThreeDClipIdTestEnable, MaxwellThreeDColorMask,
    MaxwellThreeDCompareOp, MaxwellThreeDCullFace, MaxwellThreeDFixedFunctionRegister,
    MaxwellThreeDFixedFunctionState, MaxwellThreeDFixedFunctionValue, MaxwellThreeDFrontFace,
    MaxwellThreeDIteratedBlend, MaxwellThreeDIteratedBlendPassCount, MaxwellThreeDLogicOp,
    MaxwellThreeDPixelShaderClampRange, MaxwellThreeDPixelShaderSaturate, MaxwellThreeDPolygonMode,
    MaxwellThreeDProvokingVertex, MaxwellThreeDSampleMode, MaxwellThreeDScissorState,
    MaxwellThreeDShadeMode, MaxwellThreeDSingleRopControl, MaxwellThreeDStencilOp,
    MaxwellThreeDViewportClipControl, MaxwellThreeDViewportCoordinateSwizzle,
    MaxwellThreeDViewportScaleOffsetEnable, MaxwellThreeDViewportSwizzleComponent,
    MaxwellThreeDViewportTransformState, MaxwellThreeDWindowClipState, MaxwellThreeDWindowClipType,
};
pub use render_enable::{
    MaxwellThreeDConditionalLoadConstantBuffer, MaxwellThreeDRenderEnableMode,
    MaxwellThreeDRenderEnableOverride, MaxwellThreeDRenderEnableState,
};
pub use render_targets::{
    MAXWELL_COLOR_TARGET_COUNT, MaxwellThreeDAttachmentReadiness, MaxwellThreeDClearState,
    MaxwellThreeDClearSurface, MaxwellThreeDClearSurfaceControl, MaxwellThreeDColorCompressionMode,
    MaxwellThreeDColorTargetFormat, MaxwellThreeDColorTargetSelection,
    MaxwellThreeDColorTargetState, MaxwellThreeDCompressionThreshold,
    MaxwellThreeDDepthArrayControl, MaxwellThreeDDepthStencilFormat,
    MaxwellThreeDDepthStencilTargetState, MaxwellThreeDDepthTargetCount, MaxwellThreeDImageKind,
    MaxwellThreeDImageLayout, MaxwellThreeDRawValue, MaxwellThreeDRectangle,
    MaxwellThreeDRenderTargetIndexOffset, MaxwellThreeDRenderTargetLayer,
    MaxwellThreeDRenderTargetLayerControl, MaxwellThreeDRenderTargetState,
    MaxwellThreeDSeparateFragmentData, MaxwellThreeDZCompressionMode,
};
pub(crate) use resource::{ComputeTextures, MaxwellThreeDResolvedResourceCache};
pub use resource::{
    MaxwellThreeDGuestImageFormat, MaxwellThreeDMappingReference,
    MaxwellThreeDPreservedImageLayout, MaxwellThreeDResolvedBuffer, MaxwellThreeDResolvedImage,
    MaxwellThreeDResolvedResource, MaxwellThreeDResolvedResources, MaxwellThreeDResolvedSampler,
    MaxwellThreeDResourceAccess, MaxwellThreeDResourceError, MaxwellThreeDResourceRole,
    MaxwellThreeDTextureDimension, MaxwellThreeDTextureReference,
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
pub use state::{
    MAXWELL_POLYGON_STIPPLE_PATTERN_WORD_COUNT, MaxwellThreeDAlphaFraction,
    MaxwellThreeDAttributePointSize, MaxwellThreeDConservativeRasterEnable, MaxwellThreeDEdgeFlag,
    MaxwellThreeDFillViaTriangleMode, MaxwellThreeDPointCenterMode, MaxwellThreeDPointSize,
    MaxwellThreeDPointSpriteOrigin, MaxwellThreeDPointSpriteRMode, MaxwellThreeDPointSpriteSelect,
    MaxwellThreeDRasterBoundingBox, MaxwellThreeDRasterBoundingBoxMode, MaxwellThreeDRasterState,
    MaxwellThreeDRegister, MaxwellThreeDState, MaxwellThreeDStateWrite,
    MaxwellThreeDViewportPixelCenter, MaxwellThreeDViewportState, MaxwellThreeDViewportZClipRange,
};
pub(crate) use state::{MaxwellThreeDFrontendState, MaxwellThreeDResourceStateIdentity};
pub use vertex::{
    MAXWELL_THREE_D_PRIMITIVE_AREA_MAX, MAXWELL_VERTEX_ATTRIBUTE_COUNT,
    MAXWELL_VERTEX_STREAM_COUNT, MaxwellThreeDAttributeDefaultVector,
    MaxwellThreeDAttributeDefaults, MaxwellThreeDBalancedPrimitiveWorkload, MaxwellThreeDBegin,
    MaxwellThreeDBeginInstance, MaxwellThreeDIndexBufferState, MaxwellThreeDIndexElementSize,
    MaxwellThreeDPatchSize, MaxwellThreeDPrimitiveCircularBufferThrottle,
    MaxwellThreeDPrimitiveState, MaxwellThreeDPrimitiveTopology, MaxwellThreeDUnresolvedAddress,
    MaxwellThreeDVertexArrayPrimitiveRestartEnable, MaxwellThreeDVertexAssemblyState,
    MaxwellThreeDVertexAttributeFormat, MaxwellThreeDVertexComponentWidths,
    MaxwellThreeDVertexIdUsesArrayStart, MaxwellThreeDVertexInputState,
    MaxwellThreeDVertexInputWrite, MaxwellThreeDVertexNumericalType,
    MaxwellThreeDVertexStreamFormat, MaxwellThreeDVertexStreamState,
    MaxwellThreeDVertexStreamSubstituteState,
};

use nixe_gpu::{GpuClassId, GpuMethodId};

use mme::{MaxwellThreeDMmeHost, MaxwellThreeDMmeRunError};

use super::{
    AppliedMethod, MaxwellEngineCapability, MaxwellEngineDispatchError,
    MaxwellEngineMethodMetadata, MaxwellInlineToMemoryUpload, MaxwellShaderCacheInvalidation,
    PendingEngineOperation,
};
use crate::{
    MaxwellAamVersion, MaxwellAamVersionRange, MaxwellGpuProfile, MaxwellMethodDispatch,
    MaxwellShaderProgramHeaderVersion, MaxwellShaderProgramHeaderVersionRange,
};

pub(super) const CLASS: GpuClassId = GpuClassId(0xb197);
const CLASS_NAME: &str = "MAXWELL_B";

#[derive(Clone, Copy)]
enum PendingOperation {
    Notification {
        address: u64,
        source: crate::MaxwellMethodSource,
    },
    ThreeD(MaxwellThreeDOperationTrigger),
    RenderCondition(crate::MaxwellMethodSource),
    Synchronization(MaxwellThreeDSynchronizationTrigger),
    InlineToMemory(MaxwellInlineToMemoryUpload),
    InlineConstantBuffer(MaxwellThreeDInlineConstantBufferUpload),
}

struct PreparedMethod {
    metadata: MaxwellEngineMethodMetadata,
    operation: Option<PendingOperation>,
    writes_state: bool,
}

impl PreparedMethod {
    const fn new(
        metadata: MaxwellEngineMethodMetadata,
        operation: Option<PendingOperation>,
        writes_state: bool,
    ) -> Self {
        Self {
            metadata,
            operation,
            writes_state,
        }
    }
}

const fn no_operation() -> (Option<PendingOperation>, bool) {
    (None, false)
}

const fn state_write() -> (Option<PendingOperation>, bool) {
    (None, true)
}

const fn three_d_operation(
    trigger: MaxwellThreeDOperationTrigger,
) -> (Option<PendingOperation>, bool) {
    (Some(PendingOperation::ThreeD(trigger)), false)
}

const fn state_operation(
    trigger: MaxwellThreeDOperationTrigger,
) -> (Option<PendingOperation>, bool) {
    (Some(PendingOperation::ThreeD(trigger)), true)
}

const fn synchronization_operation(
    trigger: MaxwellThreeDSynchronizationTrigger,
) -> (Option<PendingOperation>, bool) {
    (Some(PendingOperation::Synchronization(trigger)), false)
}

const fn state_inline_to_memory(
    upload: MaxwellInlineToMemoryUpload,
) -> (Option<PendingOperation>, bool) {
    (Some(PendingOperation::InlineToMemory(upload)), true)
}

const fn state_inline_constant_buffer(
    upload: MaxwellThreeDInlineConstantBufferUpload,
) -> (Option<PendingOperation>, bool) {
    (Some(PendingOperation::InlineConstantBuffer(upload)), true)
}

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
    DrawVertexArray,
    DrawIndexBuffer,
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
// public MAXWELL_B header. That header does not publish register reset values,
// so state begins explicitly unset rather than assuming zero.
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
        MethodAction::DrawVertexArray
    ),
    // https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L2830-L2831
    DRAW_INDEX_BUFFER => (
        0x17e0,
        "DRAW_INDEX_BUFFER",
        u32::MAX,
        MethodAction::DrawIndexBuffer
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

pub(super) const fn is_mme_aperture(method: GpuMethodId) -> bool {
    method.0 >= 0x3800 && method.0 <= 0x3ffc && method.0 & 3 == 0
}

pub(super) enum MaxwellThreeDMmePreflightError<E> {
    Dispatch(Box<MaxwellEngineDispatchError>),
    Consumer(E),
}

impl<E> MaxwellThreeDMmePreflightError<E> {
    fn dispatch(error: MaxwellEngineDispatchError) -> Self {
        Self::Dispatch(Box::new(error))
    }
}

pub(super) async fn preflight_mme_call<E>(
    profile: &MaxwellGpuProfile,
    source: crate::MaxwellMethodSource,
    parameters: &[u32],
    candidate: &mut MaxwellThreeDFrontendState,
    consume: &mut impl AsyncFnMut(
        PendingEngineOperation,
        &MaxwellThreeDState,
        Option<(u32, u32)>,
    ) -> Result<(), E>,
) -> Result<(), MaxwellThreeDMmePreflightError<E>> {
    let macro_index = ((source.method().0 - 0x3800) / 8) as u8;
    let program = candidate.mme().program();

    {
        let mut host = MmeDispatchHost {
            profile,
            source,
            candidate,
            consume,
            _error: std::marker::PhantomData,
        };
        match program.execute(macro_index, parameters, &mut host).await {
            Ok(()) => {}
            Err(MaxwellThreeDMmeRunError::Execution(error)) => {
                return Err(MaxwellThreeDMmePreflightError::dispatch(
                    MaxwellEngineDispatchError::MmeExecution { source, error },
                ));
            }
            Err(MaxwellThreeDMmeRunError::Host(error)) => return Err(error),
        };
    }
    // A macro is an explicit draw group, including its deferred END.
    // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/engines/maxwell_3d.cpp
    draw::flush(candidate, consume)
        .await
        .map_err(MaxwellThreeDMmePreflightError::Consumer)
}

struct MmeDispatchHost<'a, E, F> {
    profile: &'a MaxwellGpuProfile,
    source: crate::MaxwellMethodSource,
    candidate: &'a mut MaxwellThreeDFrontendState,
    consume: &'a mut F,
    _error: std::marker::PhantomData<E>,
}

impl<E, F> MaxwellThreeDMmeHost for MmeDispatchHost<'_, E, F>
where
    F: AsyncFnMut(PendingEngineOperation, &MaxwellThreeDState, Option<(u32, u32)>) -> Result<(), E>,
{
    type Error = MaxwellThreeDMmePreflightError<E>;

    fn read_register(&self, method_dword: u16) -> Result<u32, Self::Error> {
        let method = GpuMethodId(u32::from(method_dword) * 4);
        self.candidate
            .raw_register_value(method)
            .or_else(|| state::verified_raw_register_reset(method))
            .ok_or_else(|| {
                MaxwellThreeDMmePreflightError::dispatch(MaxwellEngineDispatchError::MmeExecution {
                    source: self.source,
                    error: MaxwellThreeDMmeExecutionError::RegisterReadUnavailable { method_dword },
                })
            })
    }

    fn emit_register(&mut self, method_dword: u16, argument: u32) -> Result<bool, Self::Error> {
        if draw::is_pending(self.candidate) {
            return Ok(false);
        }
        let method = GpuMethodId(u32::from(method_dword) * 4);
        if !is_plain_method(method) {
            return Ok(false);
        }
        let source = self.source.emitted_by_mme(method, argument);
        write_plain_source(source, self.candidate)
            .map_err(MaxwellThreeDMmePreflightError::dispatch)?;
        Ok(true)
    }

    async fn emit_method(&mut self, method_dword: u16, argument: u32) -> Result<(), Self::Error> {
        let method = GpuMethodId(u32::from(method_dword) * 4);
        if is_mme_aperture(method) {
            return Err(MaxwellThreeDMmePreflightError::dispatch(
                MaxwellEngineDispatchError::MmeExecution {
                    source: self.source,
                    error: MaxwellThreeDMmeExecutionError::RecursiveMacroCall { method_dword },
                },
            ));
        }
        let source = self.source.emitted_by_mme(method, argument);
        draw::before_method(self.candidate, source, self.consume)
            .await
            .map_err(MaxwellThreeDMmePreflightError::Consumer)?;
        if is_plain_method(method) {
            write_plain_source(source, self.candidate)
                .map_err(MaxwellThreeDMmePreflightError::dispatch)?;
            return Ok(());
        }
        let mut dispatch = MaxwellMethodDispatch::class_method(source, CLASS);
        // MME sends use the same method path, including shadow tracking/replay,
        // as pushbuffer writes. The interpreter and JIT both call CallMethod:
        // https://github.com/eden-emulator/mirror/blob/67bada77f8a43a90da2e94e89b8e7da73c256989/src/video_core/macro.cpp#L664-L666
        let prepared = preflight_with_shadow(self.profile, &mut dispatch, self.candidate)
            .map_err(MaxwellThreeDMmePreflightError::dispatch)?;
        if let Some(operation) = lower_pending_operation(prepared.operation, self.candidate) {
            draw::operation(self.candidate, operation, self.consume)
                .await
                .map_err(MaxwellThreeDMmePreflightError::Consumer)?;
            self.candidate.complete_inline_draw();
        }
        Ok(())
    }
}

#[inline]
pub(super) fn preflight(
    profile: &MaxwellGpuProfile,
    mut method: MaxwellMethodDispatch,
    candidate: &mut MaxwellThreeDFrontendState,
) -> Result<AppliedMethod, MaxwellEngineDispatchError> {
    let prepared = preflight_with_shadow(profile, &mut method, candidate)?;
    let operation = lower_pending_operation(prepared.operation, candidate);
    Ok(AppliedMethod::new(method, prepared.metadata, operation))
}

fn lower_pending_operation(
    operation: Option<PendingOperation>,
    _state: &MaxwellThreeDFrontendState,
) -> Option<PendingEngineOperation> {
    operation.map(|operation| match operation {
        PendingOperation::Notification { address, source } => {
            PendingEngineOperation::Notification { address, source }
        }
        PendingOperation::ThreeD(trigger) => PendingEngineOperation::ThreeD(trigger),
        PendingOperation::RenderCondition(source) => {
            PendingEngineOperation::ThreeDRenderCondition(source)
        }
        PendingOperation::Synchronization(trigger) => {
            PendingEngineOperation::ThreeDSynchronization(trigger)
        }
        PendingOperation::InlineToMemory(upload) => PendingEngineOperation::InlineToMemory(upload),
        PendingOperation::InlineConstantBuffer(upload) => {
            PendingEngineOperation::ThreeDInlineConstantBuffer(upload)
        }
    })
}

#[inline]
fn preflight_with_shadow(
    profile: &MaxwellGpuProfile,
    method: &mut MaxwellMethodDispatch,
    candidate: &mut MaxwellThreeDFrontendState,
) -> Result<PreparedMethod, MaxwellEngineDispatchError> {
    if write_plain_method(method, candidate)? {
        return Ok(PreparedMethod::new(
            MaxwellEngineMethodMetadata::new(
                CLASS,
                CLASS_NAME,
                method.source().method(),
                "STATE_REGISTER",
            ),
            None,
            true,
        ));
    }
    let source = method.source();
    if is_mme_aperture(source.method()) {
        return Err(MaxwellEngineDispatchError::MmeExecution {
            source,
            error: MaxwellThreeDMmeExecutionError::DataWithoutCall,
        });
    }
    let shadow_control = candidate.mme().shadow_ram_control().value().copied();
    let effective_argument = candidate
        .mme()
        .resolve_shadow_argument(source.method(), source.argument())
        .map_err(|error| MaxwellEngineDispatchError::MmeShadowRam { source, error })?;

    method.set_effective_argument(effective_argument);
    let effective_source = method.source();
    let prepared = preflight_register(profile, effective_source, candidate)?;
    if prepared.writes_state {
        candidate.record_raw_register(effective_source);
    }
    candidate.mme_mut().track_shadow_register(
        shadow_control,
        effective_source.method(),
        effective_source.argument(),
    );
    if matches!(prepared.operation, Some(PendingOperation::ThreeD(_))) {
        candidate.flush_dirty();
    }
    Ok(prepared)
}

/// Plain state admission builds no typed write, metadata or operation adapter.
/// The same path handles PFIFO and MME sends, including shadow replay.
#[inline]
pub(super) fn write_plain_method(
    method: &mut MaxwellMethodDispatch,
    candidate: &mut MaxwellThreeDFrontendState,
) -> Result<bool, MaxwellEngineDispatchError> {
    if !registers::passive(method.source().method()) {
        return Ok(false);
    }
    let argument = write_plain_source(method.source(), candidate)?;
    method.set_effective_argument(argument);
    Ok(true)
}

#[inline]
pub(super) fn write_plain_source(
    source: crate::MaxwellMethodSource,
    candidate: &mut MaxwellThreeDFrontendState,
) -> Result<u32, MaxwellEngineDispatchError> {
    debug_assert!(registers::passive(source.method()));
    let control = candidate.mme().shadow_ram_control().value().copied();
    let argument = candidate
        .mme()
        .resolve_shadow_argument(source.method(), source.argument())
        .map_err(|error| MaxwellEngineDispatchError::MmeShadowRam { source, error })?;
    let effective_source = source.emitted_by_mme(source.method(), argument);
    if candidate.pending_notification.is_some()
        && let Some(declaration) = METHODS_BY_DWORD[(source.method().0 / 4) as usize]
    {
        return Err(invalid_encoding(
            effective_source,
            declaration.metadata.method_name(),
            "pending notification requires completion of a method other than NO_OPERATION",
        ));
    }
    candidate.write_register(source.method(), argument);
    candidate.mme_mut().track_shadow_register(
        control,
        effective_source.method(),
        effective_source.argument(),
    );
    Ok(argument)
}

pub(super) fn write_plain_span(
    subchannel: crate::MaxwellPushbufferSubchannel,
    words: crate::MaxwellMethodSpan<'_>,
    candidate: &mut MaxwellThreeDFrontendState,
    #[cfg(test)] mut observe: impl FnMut(crate::MaxwellMethodSource, u32),
) -> Result<usize, MaxwellEngineDispatchError> {
    match candidate.mme().shadow_ram_control().value().copied() {
        Some(MaxwellThreeDMmeShadowRamControl::MethodReplay) => write_plain_words::<true, false>(
            subchannel,
            words,
            candidate,
            #[cfg(test)]
            &mut observe,
        ),
        Some(
            MaxwellThreeDMmeShadowRamControl::MethodTrack
            | MaxwellThreeDMmeShadowRamControl::MethodTrackWithFilter,
        ) => write_plain_words::<false, true>(
            subchannel,
            words,
            candidate,
            #[cfg(test)]
            &mut observe,
        ),
        _ => write_plain_words::<false, false>(
            subchannel,
            words,
            candidate,
            #[cfg(test)]
            &mut observe,
        ),
    }
}

fn write_plain_words<const REPLAY: bool, const TRACK: bool>(
    subchannel: crate::MaxwellPushbufferSubchannel,
    words: crate::MaxwellMethodSpan<'_>,
    candidate: &mut MaxwellThreeDFrontendState,
    #[cfg(test)] observe: &mut impl FnMut(crate::MaxwellMethodSource, u32),
) -> Result<usize, MaxwellEngineDispatchError> {
    let mut consumed = 0;
    for index in 0..words.len() {
        let method = words.method_at(index);
        if !registers::passive(method) {
            break;
        }
        let argument = if REPLAY {
            candidate
                .mme()
                .shadow_value(method)
                .or_else(|| state::verified_raw_register_reset(method))
                .ok_or_else(|| MaxwellEngineDispatchError::MmeShadowRam {
                    source: crate::MaxwellMethodSource::decoded(
                        words.get(index).unwrap(),
                        subchannel,
                    ),
                    error: MaxwellThreeDMmeShadowRamError::ReplayRegisterUnavailable {
                        method_dword: (method.0 / 4) as u16,
                    },
                })?
        } else {
            words.argument_at(index)
        };
        candidate.write_register(method, argument);
        if TRACK {
            candidate.mme_mut().write_shadow_register(method, argument);
        }
        #[cfg(test)]
        observe(
            crate::MaxwellMethodSource::decoded(words.get(index).unwrap(), subchannel),
            argument,
        );
        consumed += 1;
    }
    Ok(consumed)
}

pub(super) fn is_plain_method(method: GpuMethodId) -> bool {
    registers::passive(method)
}

/// Packet index ports normalize packed lanes into the native UINT32 stream.
/// No state operation, source vector or reserve call is produced per lane.
pub(super) fn inline_index_words(
    candidate: &mut MaxwellThreeDFrontendState,
    subchannel: crate::MaxwellPushbufferSubchannel,
    words: crate::MaxwellMethodSpan<'_>,
) -> Result<(), MaxwellEngineDispatchError> {
    candidate.append_index_values(
        crate::MaxwellMethodSource::decoded(words.first().unwrap(), subchannel),
        words.arguments(),
    )?;
    let last = crate::MaxwellMethodSource::decoded(words.last().unwrap(), subchannel);
    candidate.record_raw_register(last);
    let control = candidate.mme().shadow_ram_control().value().copied();
    candidate
        .mme_mut()
        .track_shadow_register(control, last.method(), last.argument());
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
    Mme,
    ConstantBuffer,
    VertexAndBinding,
}

const REGISTER_FAMILIES: [RegisterFamily; 4096] = {
    let mut families = [RegisterFamily::Other; 4096];
    let mut index = 0;
    while index < families.len() {
        let method = index as u32 * 4;
        families[index] = match method {
            0x180..=0x1b4 => RegisterFamily::InlineToMemory,
            0x114 | 0x118 | 0x11c | 0x120 | 0x3400..=0x37fc => RegisterFamily::Mme,
            0x238c..=0x23fc => RegisterFamily::ConstantBuffer,
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
            | 0x2608 => RegisterFamily::VertexAndBinding,
            0x2000..=0x217c if matches!((method - 0x2000) % 0x40, 0 | 4 | 12 | 16) => {
                RegisterFamily::VertexAndBinding
            }
            0x1240..=0x125c => RegisterFamily::VertexAndBinding,
            0x2400..=0x24fc if (method - 0x2400) % 0x20 == 0x10 => RegisterFamily::VertexAndBinding,
            _ => RegisterFamily::Other,
        };
        index += 1;
    }
    families
};

#[inline]
fn register_family(method: u32) -> RegisterFamily {
    if method & 3 != 0 {
        return RegisterFamily::Other;
    }
    REGISTER_FAMILIES
        .get((method / 4) as usize)
        .copied()
        .unwrap_or(RegisterFamily::Other)
}

fn preflight_register(
    profile: &MaxwellGpuProfile,
    source: crate::MaxwellMethodSource,
    candidate: &mut MaxwellThreeDFrontendState,
) -> Result<PreparedMethod, MaxwellEngineDispatchError> {
    // NVIDIA's packed inline ports append low lanes first. END consumes the
    // complete stream, as in the independently reviewed public draw managers:
    // https://github.com/eden-emulator/mirror/blob/d16735f5b618942136d6ab53466e3be0a382c30a/src/video_core/engines/draw_manager.cpp
    if matches!(source.method().0, 0x1300 | 0x15e8 | 0x15ec) {
        candidate.append_inline_indices(source)?;
        return Ok(PreparedMethod::new(
            MaxwellEngineMethodMetadata::new(CLASS, CLASS_NAME, source.method(), "INLINE_INDEX"),
            None,
            true,
        ));
    }
    if source.method().0 == 0x1618 && candidate.has_inline_indices() {
        return Err(invalid_encoding(
            source,
            "BEGIN",
            "a pending inline primitive requires END",
        ));
    }
    if matches!(source.method().0, 0x0d78 | 0x17e0) && candidate.has_inline_indices() {
        return Err(invalid_encoding(
            source,
            "DRAW",
            "a pending inline primitive must be consumed by END",
        ));
    }
    if source.method().0 == 0x1614 && candidate.has_inline_indices() {
        checked_bool(source, "END")?;
        let index_count = candidate.publish_inline_indices(source)?;
        return Ok(PreparedMethod::new(
            MaxwellEngineMethodMetadata::new(CLASS, CLASS_NAME, source.method(), "END"),
            Some(PendingOperation::ThreeD(
                MaxwellThreeDOperationTrigger::DrawIndexBuffer {
                    source,
                    index_count,
                },
            )),
            true,
        ));
    }
    let family = register_family(source.method().0);
    if family == RegisterFamily::InlineToMemory
        && let Some((write, method_name, upload)) =
            inline_to_memory::preflight(source, &candidate.operation_state().inline_to_memory())?
    {
        candidate.apply(write);
        let metadata =
            MaxwellEngineMethodMetadata::new(CLASS, CLASS_NAME, source.method(), method_name);
        let (operation, writes_state) = upload.map_or(state_write(), state_inline_to_memory);
        return Ok(PreparedMethod::new(metadata, operation, writes_state));
    }
    if family == RegisterFamily::Mme
        && let Some((write, method_name)) = preflight_mme_state(source, candidate)?
    {
        candidate.apply(write);
        let metadata =
            MaxwellEngineMethodMetadata::new(CLASS, CLASS_NAME, source.method(), method_name);
        return Ok(PreparedMethod::new(metadata, None, true));
    }
    if family == RegisterFamily::ConstantBuffer
        && let Some((write, method_name, upload)) =
            preflight_constant_buffer_load(source, candidate)?
    {
        candidate.apply(write);
        let metadata =
            MaxwellEngineMethodMetadata::new(CLASS, CLASS_NAME, source.method(), method_name);
        let (operation, writes_state) = upload.map_or(state_write(), state_inline_constant_buffer);
        return Ok(PreparedMethod::new(metadata, operation, writes_state));
    }
    if family == RegisterFamily::VertexAndBinding
        && let Some((write, method_name)) = preflight_vertex_and_binding_state(source, candidate)?
    {
        candidate.apply(write);
        let metadata =
            MaxwellEngineMethodMetadata::new(CLASS, CLASS_NAME, source.method(), method_name);
        return Ok(PreparedMethod::new(metadata, None, true));
    }
    if let Some(prepared) = preflight_clear(source)? {
        return Ok(prepared);
    }
    let Some(declaration) = (source.method().0 & 3 == 0)
        .then(|| METHODS_BY_DWORD.get((source.method().0 / 4) as usize))
        .flatten()
        .and_then(Option::as_ref)
    else {
        return Err(MaxwellEngineDispatchError::UnknownMethod {
            source,
            class_name: CLASS_NAME,
        });
    };
    if source.argument() & !declaration.defined_mask != 0 {
        return Err(MaxwellEngineDispatchError::InvalidMethodValue {
            source,
            metadata: declaration.metadata,
            defined_mask: declaration.defined_mask,
        });
    }
    // The documented notification request completes after the following method.
    // NO_OPERATION is the explicit way to request notification without other work.
    // Other completion-producing methods and the interrupt variant stay unsupported.
    // https://envytools.readthedocs.io/en/latest/hw/graph/intro.html#notify-method
    // https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L33-L40
    if candidate.pending_notification.is_some()
        && !matches!(declaration.action, MethodAction::NoOperation)
    {
        return Err(invalid_encoding(
            source,
            declaration.metadata.method_name(),
            "pending notification requires completion of a method other than NO_OPERATION",
        ));
    }
    let (operation, writes_state) = match declaration.action {
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
        | MethodAction::ZCullBounds => unreachable!("passive method entered execution dispatch"),
        MethodAction::RenderEnableMode | MethodAction::RenderEnableOverride => {
            let valid = match declaration.action {
                MethodAction::RenderEnableMode => {
                    MaxwellThreeDRenderEnableMode::parse(source.argument()).is_some()
                }
                MethodAction::RenderEnableOverride => {
                    MaxwellThreeDRenderEnableOverride::parse(source.argument()).is_some()
                }
                _ => unreachable!(),
            };
            if !valid {
                return Err(MaxwellEngineDispatchError::InvalidMethodValue {
                    source,
                    metadata: declaration.metadata,
                    defined_mask: declaration.defined_mask,
                });
            }
            (Some(PendingOperation::RenderCondition(source)), true)
        }
        MethodAction::Notify => {
            if source.argument() != 0 {
                return Err(MaxwellEngineDispatchError::UnsupportedMethod {
                    source,
                    metadata: declaration.metadata,
                });
            }
            let high = candidate.raw_register_value(GpuMethodId(0x0104));
            let low = candidate.raw_register_value(GpuMethodId(0x0108));
            let (Some(high), Some(low)) = (high, low) else {
                return Err(invalid_encoding(
                    source,
                    declaration.metadata.method_name(),
                    "notification address has not been programmed",
                ));
            };
            if high > 0xff {
                return Err(invalid_encoding(
                    source,
                    declaration.metadata.method_name(),
                    "notification address exceeds its 40-bit field",
                ));
            }
            candidate.pending_notification =
                Some(((u64::from(high) << 32) | u64::from(low), source));
            state_write()
        }
        MethodAction::NoOperation => {
            if let Some((address, source)) = candidate.pending_notification.take() {
                (
                    Some(PendingOperation::Notification { address, source }),
                    false,
                )
            } else {
                no_operation()
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
            no_operation()
        }
        MethodAction::ClearReportValue => {
            // Type 2 resets the hierarchical Z-cull statistics. We do not
            // run that guest cache, so there is no accumulated cache work
            // to reset. Counter reporting remains explicitly unsupported;
            // resetting this cache must not clear depth or other counters.
            // https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L2722-L2741
            if source.argument() != 2 {
                return Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                    source,
                    method_name: "CLEAR_REPORT_VALUE",
                    reason: "resetting non-Z-cull performance counters is not implemented",
                });
            }
            no_operation()
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
            no_operation()
        }
        MethodAction::DecompressSurface => {
            let request = MaxwellThreeDDecompressSurface::parse(source.argument()).ok_or(
                MaxwellEngineDispatchError::InvalidMethodValue {
                    source,
                    metadata: declaration.metadata,
                    defined_mask: declaration.defined_mask,
                },
            )?;
            synchronization_operation(MaxwellThreeDSynchronizationTrigger::DecompressSurface {
                request,
                source,
            })
        }
        MethodAction::WaitForIdle => {
            synchronization_operation(MaxwellThreeDSynchronizationTrigger::WaitForIdle {
                value: source.argument(),
                source,
            })
        }
        MethodAction::PixelShaderBarrier => {
            synchronization_operation(MaxwellThreeDSynchronizationTrigger::PixelShaderBarrier {
                system_memory_barrier: source.argument() & 1 != 0,
                source,
            })
        }
        MethodAction::TiledCacheBarrier => {
            synchronization_operation(MaxwellThreeDSynchronizationTrigger::TiledCacheBarrier {
                source,
            })
        }
        MethodAction::MmeShadowRamControl => {
            let value = MaxwellThreeDMmeShadowRamControl::parse(source.argument()).ok_or(
                MaxwellEngineDispatchError::InvalidMethodValue {
                    source,
                    metadata: declaration.metadata,
                    defined_mask: declaration.defined_mask,
                },
            )?;
            let write =
                MaxwellThreeDStateWrite::Mme(MaxwellThreeDMmeStateWrite::ShadowRamControl {
                    value,
                    source,
                });
            candidate.apply(write);
            state_write()
        }
        MethodAction::MutableMethodControl => {
            let value = MaxwellThreeDMutableMethodControl::parse(source.argument()).ok_or(
                MaxwellEngineDispatchError::InvalidMethodValue {
                    source,
                    metadata: declaration.metadata,
                    defined_mask: declaration.defined_mask,
                },
            )?;
            let write =
                MaxwellThreeDStateWrite::Mme(MaxwellThreeDMmeStateWrite::MutableMethodControl {
                    value,
                    source,
                });
            candidate.apply(write);
            state_write()
        }
        MethodAction::FalconFirmwareCall4 => {
            let address = MaxwellThreeDFalconRegisterAddress::try_new(source.argument()).ok_or(
                MaxwellEngineDispatchError::FalconFirmware {
                    source,
                    error: MaxwellThreeDFalconError::UnalignedRegisterAddress {
                        address: source.argument(),
                    },
                },
            )?;
            let firmware_argument = |index| {
                candidate
                    .mme()
                    .shadow_scratch(MaxwellThreeDMmeShadowScratchIndex::new(index))
                    .and_then(MaxwellThreeDRegister::raw)
                    .ok_or(MaxwellEngineDispatchError::FalconFirmware {
                        source,
                        error: MaxwellThreeDFalconError::MissingFirmwareArgument { index },
                    })
            };
            let value = firmware_argument(1)?;
            let mask = firmware_argument(2)?;
            let write = MaxwellThreeDStateWrite::FalconMaskedRegister(
                MaxwellThreeDFalconMaskedRegisterWrite::new(address, value, mask, source),
            );
            candidate.apply(write);
            state_write()
        }
        MethodAction::TiledCacheFlush => {
            let mode = MaxwellThreeDTiledCacheFlushMode::parse(source.argument()).ok_or(
                MaxwellEngineDispatchError::InvalidMethodValue {
                    source,
                    metadata: declaration.metadata,
                    defined_mask: declaration.defined_mask,
                },
            )?;
            synchronization_operation(MaxwellThreeDSynchronizationTrigger::TiledCacheFlush {
                mode,
                source,
            })
        }
        MethodAction::CheckSphVersion => {
            let requested = MaxwellShaderProgramHeaderVersionRange::new(
                MaxwellShaderProgramHeaderVersion::new(source.argument() as u16),
                MaxwellShaderProgramHeaderVersion::new((source.argument() >> 16) as u16),
            );
            if !requested.is_well_ordered() {
                return Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                    source,
                    method_name: declaration.metadata.method_name(),
                    reason: "current SPH version precedes oldest supported SPH version",
                });
            }
            let supported = profile.shader().sph_versions();
            if !supported.overlaps(requested) {
                return Err(
                    MaxwellEngineDispatchError::IncompatibleShaderProgramHeaderVersion {
                        source,
                        requested,
                        supported,
                    },
                );
            }
            no_operation()
        }
        MethodAction::CheckAamVersion => {
            let requested = MaxwellAamVersionRange::new(
                MaxwellAamVersion::new(source.argument() as u16),
                MaxwellAamVersion::new((source.argument() >> 16) as u16),
            );
            if !requested.is_well_ordered() {
                return Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
                    source,
                    method_name: declaration.metadata.method_name(),
                    reason: "current AAM version precedes oldest supported AAM version",
                });
            }
            let supported = profile.aam_versions();
            if !supported.overlaps(requested) {
                return Err(MaxwellEngineDispatchError::IncompatibleAamVersion {
                    source,
                    requested,
                    supported,
                });
            }
            no_operation()
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
            synchronization_operation(MaxwellThreeDSynchronizationTrigger::ReportSemaphore {
                control,
                source,
            })
        }
        MethodAction::InvalidateShaderCachesNoWfi => {
            let caches = MaxwellShaderCacheInvalidation::new(
                source.argument() & 1 != 0,
                source.argument() & (1 << 4) != 0,
                source.argument() & (1 << 12) != 0,
            );
            synchronization_operation(
                MaxwellThreeDSynchronizationTrigger::InvalidateShaderCachesNoWfi { caches, source },
            )
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
            synchronization_operation(
                MaxwellThreeDSynchronizationTrigger::InvalidateShaderCaches { request, source },
            )
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
            synchronization_operation(
                MaxwellThreeDSynchronizationTrigger::InvalidateTextureCacheNoWfi {
                    request,
                    source,
                },
            )
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
            synchronization_operation(
                MaxwellThreeDSynchronizationTrigger::InvalidateTextureCache { request, source },
            )
        }
        MethodAction::FlushPendingWrites => {
            let request = MaxwellThreeDFlushPendingWrites::new(source.argument() != 0);
            synchronization_operation(MaxwellThreeDSynchronizationTrigger::FlushPendingWrites {
                request,
                source,
            })
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
            synchronization_operation(MaxwellThreeDSynchronizationTrigger::IncrementSyncpoint {
                request,
                source,
            })
        }
        MethodAction::DrawVertexArray => {
            if source.argument() == 0 {
                return Err(invalid_encoding(
                    source,
                    "DRAW_VERTEX_ARRAY",
                    "vertex count is zero",
                ));
            }
            three_d_operation(MaxwellThreeDOperationTrigger::DrawVertexArray {
                source,
                vertex_count: source.argument(),
            })
        }
        MethodAction::DrawIndexBuffer => {
            if source.argument() == 0 {
                return Err(invalid_encoding(
                    source,
                    "DRAW_INDEX_BUFFER",
                    "index count is zero",
                ));
            }
            state_operation(MaxwellThreeDOperationTrigger::DrawIndexBuffer {
                source,
                index_count: source.argument(),
            })
        }
        MethodAction::Missing(capability) => {
            return Err(MaxwellEngineDispatchError::MissingCapability {
                source,
                metadata: declaration.metadata,
                capability,
            });
        }
    };
    Ok(PreparedMethod::new(
        *declaration.metadata,
        operation,
        writes_state,
    ))
}

fn preflight_mme_state(
    source: crate::MaxwellMethodSource,
    candidate: &MaxwellThreeDFrontendState,
) -> Result<Option<(MaxwellThreeDStateWrite, &'static str)>, MaxwellEngineDispatchError> {
    let mme = candidate.mme();
    if (0x3400..0x3800).contains(&source.method().0) {
        let index = ((source.method().0 - 0x3400) / 4) as u8;
        return Ok(Some((
            MaxwellThreeDStateWrite::Mme(MaxwellThreeDMmeStateWrite::ShadowScratch {
                index: MaxwellThreeDMmeShadowScratchIndex::new(index),
                value: source.argument(),
                source,
            }),
            "SET_MME_SHADOW_SCRATCH",
        )));
    }
    let (write, method_name) = match source.method().0 {
        0x0114 => (
            MaxwellThreeDMmeStateWrite::InstructionPointer {
                value: MaxwellThreeDMmeRamAddress::new(source.argument()),
                source,
            },
            "LOAD_MME_INSTRUCTION_RAM_POINTER",
        ),
        0x0118 => {
            let address =
                mme.next_instruction_address()
                    .ok_or(MaxwellEngineDispatchError::MmeRamLoad {
                        source,
                        ram: MaxwellThreeDMmeRam::Instruction,
                        error: MaxwellThreeDMmeLoadError::PointerUnset,
                    })?;
            if address.raw() == u32::MAX {
                return Err(MaxwellEngineDispatchError::MmeRamLoad {
                    source,
                    ram: MaxwellThreeDMmeRam::Instruction,
                    error: MaxwellThreeDMmeLoadError::PointerOverflow,
                });
            }
            if mme.instruction(address).is_none()
                && mme.instruction_count() >= MAXWELL_THREE_D_MME_CAPTURED_INSTRUCTION_WORDS
            {
                return Err(MaxwellEngineDispatchError::MmeRamLoad {
                    source,
                    ram: MaxwellThreeDMmeRam::Instruction,
                    error: MaxwellThreeDMmeLoadError::StorageLimitExceeded {
                        limit: MAXWELL_THREE_D_MME_CAPTURED_INSTRUCTION_WORDS,
                    },
                });
            }
            (
                MaxwellThreeDMmeStateWrite::Instruction {
                    address,
                    value: MaxwellThreeDMmeInstruction::new(source.argument()),
                    source,
                },
                "LOAD_MME_INSTRUCTION_RAM",
            )
        }
        0x011c => (
            MaxwellThreeDMmeStateWrite::StartAddressPointer {
                value: MaxwellThreeDMmeRamAddress::new(source.argument()),
                source,
            },
            "LOAD_MME_START_ADDRESS_RAM_POINTER",
        ),
        0x0120 => {
            let index =
                mme.next_start_address_index()
                    .ok_or(MaxwellEngineDispatchError::MmeRamLoad {
                        source,
                        ram: MaxwellThreeDMmeRam::StartAddress,
                        error: MaxwellThreeDMmeLoadError::PointerUnset,
                    })?;
            if index.raw() == u32::MAX {
                return Err(MaxwellEngineDispatchError::MmeRamLoad {
                    source,
                    ram: MaxwellThreeDMmeRam::StartAddress,
                    error: MaxwellThreeDMmeLoadError::PointerOverflow,
                });
            }
            if mme.start_address(index).is_none()
                && mme.start_address_count() >= MAXWELL_THREE_D_MME_CAPTURED_START_ADDRESSES
            {
                return Err(MaxwellEngineDispatchError::MmeRamLoad {
                    source,
                    ram: MaxwellThreeDMmeRam::StartAddress,
                    error: MaxwellThreeDMmeLoadError::StorageLimitExceeded {
                        limit: MAXWELL_THREE_D_MME_CAPTURED_START_ADDRESSES,
                    },
                });
            }
            (
                MaxwellThreeDMmeStateWrite::StartAddress {
                    index,
                    address: MaxwellThreeDMmeRamAddress::new(source.argument()),
                    source,
                },
                "LOAD_MME_START_ADDRESS_RAM",
            )
        }
        _ => return Ok(None),
    };
    Ok(Some((MaxwellThreeDStateWrite::Mme(write), method_name)))
}

type PreflightConstantBufferLoad = (
    MaxwellThreeDStateWrite,
    &'static str,
    Option<MaxwellThreeDInlineConstantBufferUpload>,
);

fn preflight_constant_buffer_load(
    source: crate::MaxwellMethodSource,
    candidate: &MaxwellThreeDFrontendState,
) -> Result<Option<PreflightConstantBufferLoad>, MaxwellEngineDispatchError> {
    use MaxwellThreeDShaderBindingWrite as B;

    let raw = source.argument();
    match source.method().0 {
        0x238c if raw <= u32::from(u16::MAX) => Ok(Some((
            MaxwellThreeDStateWrite::ShaderBinding(B::ConstantBufferLoadOffset {
                value: raw as u16,
                source,
            }),
            "LOAD_CONSTANT_BUFFER_OFFSET",
            None,
        ))),
        0x238c => Err(invalid_encoding(
            source,
            "LOAD_CONSTANT_BUFFER_OFFSET",
            "offset exceeds the verified 16-bit field",
        )),
        method if (0x2390..0x2400).contains(&method) && method.is_multiple_of(4) => {
            if candidate
                .operation_state()
                .render_enable()
                .conditional_load_constant_buffer()
                .value()
                == Some(&MaxwellThreeDConditionalLoadConstantBuffer::Enabled)
            {
                return Err(
                    MaxwellEngineDispatchError::UnsupportedConditionalConstantBufferLoad { source },
                );
            }
            let bindings = candidate.operation_state().shader_bindings();
            let selector = bindings.selector();
            let address = selector.address().ok_or_else(|| {
                invalid_encoding(
                    source,
                    "LOAD_CONSTANT_BUFFER",
                    "data load requires a complete selector address",
                )
            })?;
            let size = *selector.size().value().ok_or_else(|| {
                invalid_encoding(
                    source,
                    "LOAD_CONSTANT_BUFFER",
                    "data load requires a selector size",
                )
            })?;
            let offset = bindings.constant_buffer_offset().ok_or_else(|| {
                invalid_encoding(
                    source,
                    "LOAD_CONSTANT_BUFFER",
                    "data load requires LOAD_CONSTANT_BUFFER_OFFSET",
                )
            })?;
            let next_offset = offset.checked_add(4).ok_or_else(|| {
                invalid_encoding(
                    source,
                    "LOAD_CONSTANT_BUFFER",
                    "inline upload offset overflows",
                )
            })?;
            if next_offset > size {
                return Err(invalid_encoding(
                    source,
                    "LOAD_CONSTANT_BUFFER",
                    "inline upload exceeds the selected constant-buffer size",
                ));
            }
            if address
                .get()
                .checked_add(u64::from(next_offset))
                .is_none_or(|end| end > (1_u64 << 40))
            {
                return Err(invalid_encoding(
                    source,
                    "LOAD_CONSTANT_BUFFER",
                    "inline upload GPU range overflows",
                ));
            }
            let upload = MaxwellThreeDInlineConstantBufferUpload::new(address, offset, raw, source);
            Ok(Some((
                MaxwellThreeDStateWrite::ShaderBinding(B::ConstantBufferLoadData {
                    next_offset,
                    source,
                }),
                "LOAD_CONSTANT_BUFFER",
                Some(upload),
            )))
        }
        _ => Ok(None),
    }
}

fn preflight_vertex_and_binding_state(
    source: crate::MaxwellMethodSource,
    candidate: &MaxwellThreeDFrontendState,
) -> Result<Option<(MaxwellThreeDStateWrite, &'static str)>, MaxwellEngineDispatchError> {
    use MaxwellThreeDShaderBindingWrite as B;
    use MaxwellThreeDVertexInputWrite as V;
    let method = source.method().0;
    let raw = source.argument();
    if (0x2400..0x2500).contains(&method) && (method - 0x2400) % 0x20 == 0x10 {
        let group = ((method - 0x2400) / 0x20) as u8;
        if raw & !0x1f1 != 0 {
            return Err(invalid_encoding(
                source,
                "BIND_GROUP_CONSTANT_BUFFER",
                "undefined constant-buffer binding bits",
            ));
        }
        let enabled = raw & 1 != 0;
        let slot = ((raw >> 4) & 0x1f) as u8;
        let selector = candidate.operation_state().shader_bindings().selector();
        let (address, size) = if enabled {
            let address = selector.address().ok_or_else(|| {
                invalid_encoding(
                    source,
                    "BIND_GROUP_CONSTANT_BUFFER",
                    "enabled binding requires a complete selector address",
                )
            })?;
            let size = *selector.size().value().ok_or_else(|| {
                invalid_encoding(
                    source,
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
                    source,
                    "BIND_GROUP_CONSTANT_BUFFER",
                    "constant-buffer range is empty or overflows",
                ));
            }
            (Some(address), Some(size))
        } else {
            (None, None)
        };
        return Ok(Some((
            MaxwellThreeDStateWrite::ShaderBinding(B::BindConstantBuffer {
                group,
                slot,
                enabled,
                address,
                size,
                source,
            }),
            "BIND_GROUP_CONSTANT_BUFFER",
        )));
    }
    Ok(match method {
        0x1614 => Some((
            MaxwellThreeDStateWrite::VertexInput(V::End {
                value: checked_bool(source, "END")?,
                source,
            }),
            "END",
        )),
        0x1618 => Some((
            MaxwellThreeDStateWrite::VertexInput(V::Begin {
                value: MaxwellThreeDBegin::parse(raw).ok_or_else(|| {
                    invalid_encoding(source, "BEGIN", "invalid topology or begin modifier")
                })?,
                source,
            }),
            "BEGIN",
        )),
        _ => None,
    })
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

fn checked_bool(
    source: crate::MaxwellMethodSource,
    name: &'static str,
) -> Result<bool, MaxwellEngineDispatchError> {
    match source.argument() {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(invalid_encoding(source, name, "expected boolean 0 or 1")),
    }
}

fn rectangle(raw: u32) -> Option<MaxwellThreeDRectangle> {
    let value = MaxwellThreeDRectangle {
        min: raw as u16,
        max: (raw >> 16) as u16,
    };
    (value.min <= value.max).then_some(value)
}

fn preflight_clear(
    source: crate::MaxwellMethodSource,
) -> Result<Option<PreparedMethod>, MaxwellEngineDispatchError> {
    if source.method().0 != 0x19d0 {
        return Ok(None);
    }
    let value = MaxwellThreeDClearSurface::parse(source.argument()).ok_or_else(|| {
        invalid_encoding(
            source,
            "CLEAR_SURFACE",
            "invalid target, layer, or reserved bits",
        )
    })?;
    let (operation, writes_state) = if value.depth() || value.stencil() || value.color_mask() != 0 {
        state_operation(MaxwellThreeDOperationTrigger::ClearSurface { source })
    } else {
        state_write()
    };
    Ok(Some(PreparedMethod::new(
        MaxwellEngineMethodMetadata::new(CLASS, CLASS_NAME, source.method(), "CLEAR_SURFACE"),
        operation,
        writes_state,
    )))
}
