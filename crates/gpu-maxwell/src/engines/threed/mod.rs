//! GM20B `MAXWELL_B` 3D class methods and register transitions.

mod bindings;
mod color_reduction;
mod constant_color;
mod counters;
mod coverage;
mod draw;
mod falcon;
mod inline_to_memory;
mod instrumentation;
mod l2_cache;
mod line;
mod mme;
mod operations;
mod output;
mod render_enable;
mod render_targets;
mod resource;
mod shader_execution;
mod state;
mod tiled_cache;
mod vertex;
mod zcull;

pub use bindings::{
    MAXWELL_BIND_GROUP_COUNT, MAXWELL_CONSTANT_BUFFER_SLOT_COUNT, MAXWELL_PIPELINE_SHADER_COUNT,
    MAXWELL_TESSELLATION_LOD_COUNT, MaxwellThreeDBindGroupState,
    MaxwellThreeDConstantBufferBinding, MaxwellThreeDConstantBufferLoadState,
    MaxwellThreeDConstantBufferSelectorState, MaxwellThreeDDescriptorPoolState,
    MaxwellThreeDInlineConstantBufferUpload, MaxwellThreeDPipelineBindingState,
    MaxwellThreeDProgramRegionState, MaxwellThreeDSamplerBindingMode,
    MaxwellThreeDShaderBindingState, MaxwellThreeDShaderBindingWrite, MaxwellThreeDShaderStage,
    MaxwellThreeDTessellationLod,
};
pub use color_reduction::{
    MaxwellThreeDColorReductionFp16Threshold, MaxwellThreeDColorReductionSrgb8Threshold,
    MaxwellThreeDColorReductionState, MaxwellThreeDColorReductionStateWrite,
    MaxwellThreeDColorReductionThresholdsEnable, MaxwellThreeDColorReductionThresholdsFp16,
    MaxwellThreeDColorReductionThresholdsSrgb8, MaxwellThreeDColorReductionThresholdsUnorm8,
    MaxwellThreeDColorReductionThresholdsUnorm10, MaxwellThreeDColorReductionThresholdsUnorm16,
    MaxwellThreeDUnorm8,
};
pub use constant_color::{
    MaxwellThreeDConstantColorComponent, MaxwellThreeDConstantColorRenderingState,
    MaxwellThreeDConstantColorRenderingStateWrite, MaxwellThreeDConstantColorValue,
};
pub use counters::{
    MaxwellThreeDCounterState, MaxwellThreeDCounterStateWrite, MaxwellThreeDZPassPixelCountEnable,
};
pub use coverage::{
    MAXWELL_SAMPLE_LOCATION_GROUP_COUNT, MAXWELL_SAMPLE_LOCATIONS_PER_GROUP,
    MaxwellThreeDAlphaToCoverageOverride, MaxwellThreeDCoverageState,
    MaxwellThreeDCoverageStateWrite, MaxwellThreeDCoverageToColor, MaxwellThreeDCsaaEnable,
    MaxwellThreeDHybridAntiAliasCentroid, MaxwellThreeDHybridAntiAliasControl,
    MaxwellThreeDPostZPixelShaderImask, MaxwellThreeDPsOutputSampleMaskUsage,
    MaxwellThreeDSampleLocation, MaxwellThreeDSampleLocationGroup, MaxwellThreeDTirControl,
    MaxwellThreeDTirMode, MaxwellThreeDTirModulationComponentSelect,
    MaxwellThreeDTirModulationFunction,
};
pub(crate) use draw::lower_maxwell_three_d_operation_into_cache;
pub use draw::{
    MaxwellThreeDLoweredWork, MaxwellThreeDLoweringCache, MaxwellThreeDLoweringError,
    MaxwellThreeDOperationTrigger, MaxwellThreeDShaderResourceUse, MaxwellThreeDTranslatedShader,
    MaxwellThreeDTranslatedShaders, lower_maxwell_three_d_operation,
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
pub use instrumentation::{
    MaxwellThreeDInstrumentationState, MaxwellThreeDInstrumentationStateWrite,
    MaxwellThreeDInstrumentationValue,
};
pub use line::{
    MaxwellThreeDAliasedLineWidthEnable, MaxwellThreeDAntiAliasedLineEnable,
    MaxwellThreeDLineState, MaxwellThreeDLineStateWrite, MaxwellThreeDLineStippleParameters,
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
    MaxwellThreeDReportSemaphorePipelineLocation, MaxwellThreeDReportSemaphoreRelease,
    MaxwellThreeDReportSemaphoreState, MaxwellThreeDReportSemaphoreStateWrite,
    MaxwellThreeDReportSemaphoreStructureSize, MaxwellThreeDShaderCacheInvalidation,
    MaxwellThreeDSynchronizationError, MaxwellThreeDSynchronizationOperation,
    MaxwellThreeDSynchronizationPlan, MaxwellThreeDSynchronizationTrigger,
    MaxwellThreeDSyncpointCondition, MaxwellThreeDSyncpointIncrement,
    MaxwellThreeDTextureCacheInvalidation, MaxwellThreeDTextureCacheLines,
    MaxwellThreeDTextureCacheTarget, MaxwellThreeDTiledCacheFlushMode,
    lower_maxwell_three_d_synchronization,
};

pub use l2_cache::{
    MaxwellThreeDL2CacheEvictionPolicy, MaxwellThreeDL2CacheState, MaxwellThreeDL2CacheStateWrite,
    MaxwellThreeDRopL2CacheRequest, MaxwellThreeDSystemMemoryVolatile,
    MaxwellThreeDVafL2CacheControl,
};
pub use output::{
    MAXWELL_SCISSOR_COUNT, MAXWELL_VIEWPORT_COUNT, MAXWELL_WINDOW_CLIP_COUNT,
    MaxwellThreeDBlendControlState, MaxwellThreeDBlendEnableCommon, MaxwellThreeDBlendFactor,
    MaxwellThreeDBlendFloatPixelKillEnable, MaxwellThreeDBlendOp,
    MaxwellThreeDBlendPerFormatEnable, MaxwellThreeDBlendZeroTimesAnythingIsZero,
    MaxwellThreeDClipIdTestEnable, MaxwellThreeDColorMask, MaxwellThreeDCompareOp,
    MaxwellThreeDCullFace, MaxwellThreeDFixedFunctionRegister, MaxwellThreeDFixedFunctionState,
    MaxwellThreeDFixedFunctionValue, MaxwellThreeDFixedFunctionWrite, MaxwellThreeDFrontFace,
    MaxwellThreeDIteratedBlend, MaxwellThreeDIteratedBlendPassCount, MaxwellThreeDLogicOp,
    MaxwellThreeDPixelShaderClampRange, MaxwellThreeDPixelShaderSaturate, MaxwellThreeDPolygonMode,
    MaxwellThreeDProvokingVertex, MaxwellThreeDSampleMode, MaxwellThreeDScissorState,
    MaxwellThreeDShadeMode, MaxwellThreeDStencilOp, MaxwellThreeDSurfaceClipAxis,
    MaxwellThreeDViewportClipControl, MaxwellThreeDViewportCoordinateSwizzle,
    MaxwellThreeDViewportScaleOffsetEnable, MaxwellThreeDViewportSwizzleComponent,
    MaxwellThreeDViewportTransformState, MaxwellThreeDWindowClipState, MaxwellThreeDWindowClipType,
};
pub use render_enable::{
    MaxwellThreeDConditionalLoadConstantBuffer, MaxwellThreeDRenderEnableMode,
    MaxwellThreeDRenderEnableState, MaxwellThreeDRenderEnableStateWrite,
};
pub use render_targets::{
    MAXWELL_COLOR_TARGET_COUNT, MaxwellThreeDAttachmentReadiness, MaxwellThreeDClearState,
    MaxwellThreeDClearSurface, MaxwellThreeDClearSurfaceControl, MaxwellThreeDColorCompressionMode,
    MaxwellThreeDColorTargetFormat, MaxwellThreeDColorTargetSelection,
    MaxwellThreeDColorTargetState, MaxwellThreeDCompressionThreshold,
    MaxwellThreeDDepthStencilFormat, MaxwellThreeDDepthStencilTargetState,
    MaxwellThreeDDepthTargetCount, MaxwellThreeDImageKind, MaxwellThreeDImageLayout,
    MaxwellThreeDRawValue, MaxwellThreeDRectangle, MaxwellThreeDRenderTargetIndexOffset,
    MaxwellThreeDRenderTargetLayer, MaxwellThreeDRenderTargetLayerControl,
    MaxwellThreeDRenderTargetState, MaxwellThreeDRenderTargetWrite,
    MaxwellThreeDSeparateFragmentData, MaxwellThreeDZCompressionMode,
};
pub(crate) use resource::MaxwellThreeDResolvedResourceCache;
#[cfg(test)]
#[allow(unused_imports)]
pub(crate) use resource::resolve_maxwell_three_d_resources_for_roles_with_staged_writes;
pub use resource::{
    MaxwellThreeDDirtySubresource, MaxwellThreeDDirtySubresources, MaxwellThreeDGuestImageFormat,
    MaxwellThreeDMappingReference, MaxwellThreeDPreservedImageLayout, MaxwellThreeDResolvedBuffer,
    MaxwellThreeDResolvedImage, MaxwellThreeDResolvedResource, MaxwellThreeDResolvedResources,
    MaxwellThreeDResolvedSampler, MaxwellThreeDResourceAccess, MaxwellThreeDResourceAlias,
    MaxwellThreeDResourceError, MaxwellThreeDResourceRole, MaxwellThreeDTextureDimension,
    MaxwellThreeDTextureReference, resolve_maxwell_three_d_resources,
    resolve_maxwell_three_d_resources_for_roles,
};
pub use shader_execution::{
    MAXWELL_THREE_D_SHADER_LOCAL_MEMORY_PER_WARP_SIZE_MAX,
    MAXWELL_THREE_D_SM_TIMEOUT_COUNTER_BIT_MAX, MaxwellThreeDApiMandatedEarlyZ,
    MaxwellThreeDDirectlyAddressableMemory, MaxwellThreeDPixelShaderInterlockControl,
    MaxwellThreeDPixelShaderInterlockFragmentOrder, MaxwellThreeDPixelShaderInterlockMode,
    MaxwellThreeDPixelShaderInterlockTileSize, MaxwellThreeDShaderExceptionsEnable,
    MaxwellThreeDShaderExecutionState, MaxwellThreeDShaderExecutionStateWrite,
    MaxwellThreeDShaderLocalMemoryPerWarpSize, MaxwellThreeDShaderLocalMemoryState,
    MaxwellThreeDShaderWatermarkRange, MaxwellThreeDShaderWatermarkTarget,
    MaxwellThreeDSmTimeoutCounterBit, MaxwellThreeDSubtilingPerfKnobA,
    MaxwellThreeDSubtilingPerfKnobB, MaxwellThreeDVisibleCallLimit,
};
pub use state::{
    MAXWELL_POLYGON_STIPPLE_PATTERN_WORD_COUNT, MaxwellThreeDAlphaFraction,
    MaxwellThreeDAttributePointSize, MaxwellThreeDConservativeRasterEnable, MaxwellThreeDEdgeFlag,
    MaxwellThreeDFillViaTriangleMode, MaxwellThreeDPointCenterMode, MaxwellThreeDPointSize,
    MaxwellThreeDPointSpriteOrigin, MaxwellThreeDPointSpriteRMode, MaxwellThreeDPointSpriteSelect,
    MaxwellThreeDRasterBoundingBox, MaxwellThreeDRasterBoundingBoxMode, MaxwellThreeDRasterState,
    MaxwellThreeDRegister, MaxwellThreeDRegisterOrigin, MaxwellThreeDState,
    MaxwellThreeDStateWrite, MaxwellThreeDViewportPixelCenter, MaxwellThreeDViewportState,
    MaxwellThreeDViewportZClipRange,
};
pub(crate) use state::{
    MaxwellThreeDFrontendState, MaxwellThreeDResourceStateIdentity,
    MaxwellThreeDShaderStateIdentity,
};
pub use tiled_cache::{
    MaxwellThreeDTiledCacheState, MaxwellThreeDTiledCacheStateWrite,
    MaxwellThreeDTiledCacheTileSize, MaxwellThreeDTiledCacheUnknownConfig,
};
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
pub use zcull::{
    MaxwellThreeDZCullBounds, MaxwellThreeDZCullCriterion, MaxwellThreeDZCullEnable,
    MaxwellThreeDZCullRegionId, MaxwellThreeDZCullState, MaxwellThreeDZCullStateWrite,
    MaxwellThreeDZCullStatsEnable, MaxwellThreeDZCullStencilFunction,
};

use nixe_gpu::{GpuClassId, GpuMethodId};

use mme::{MaxwellThreeDMmeHost, MaxwellThreeDMmeRunError};

use super::{
    AppliedMethod, MaxwellEngineCapability, MaxwellEngineDispatchError,
    MaxwellEngineMethodDispatch, MaxwellEngineMethodMetadata, MaxwellEngineOperation,
    MaxwellInlineToMemoryUpload, MaxwellShaderCacheInvalidation, MaxwellThreeDTriggeredOperation,
};
use crate::{
    MaxwellAamVersion, MaxwellAamVersionRange, MaxwellGpuProfile, MaxwellMethodDispatch,
    MaxwellShaderProgramHeaderVersion, MaxwellShaderProgramHeaderVersionRange, MaxwellSpaVersion,
};

pub(super) const CLASS: GpuClassId = GpuClassId(0xb197);
const CLASS_NAME: &str = "MAXWELL_B";

#[derive(Clone, Copy)]
enum PendingOperation {
    ThreeD(MaxwellThreeDOperationTrigger),
    Synchronization(MaxwellThreeDSynchronizationTrigger),
    InlineToMemory(MaxwellInlineToMemoryUpload),
    InlineConstantBuffer(MaxwellThreeDInlineConstantBufferUpload),
}

struct PreparedMethod {
    method: MaxwellMethodDispatch,
    metadata: MaxwellEngineMethodMetadata,
    operation: Option<PendingOperation>,
    writes_state: bool,
}

impl PreparedMethod {
    const fn new(
        method: MaxwellMethodDispatch,
        metadata: MaxwellEngineMethodMetadata,
        operation: Option<PendingOperation>,
        writes_state: bool,
    ) -> Self {
        Self {
            method,
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
    NoOperation,
    DecompressSurface,
    WaitForIdle,
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
    TiledCacheUnknownConfig(u8),
    TiledCacheFlush,
    ShaderWatermarks(MaxwellThreeDShaderWatermarkTarget),
    L1Configuration,
    ColorReductionThresholdsEnable,
    ColorReductionThresholdsUnorm8,
    ColorReductionThresholdsUnorm10,
    ColorReductionThresholdsUnorm16,
    ColorReductionThresholdsFp16,
    ColorReductionThresholdsSrgb8,
    ApiMandatedEarlyZ,
    PostZPixelShaderImask,
    PixelShaderInterlockControl,
    ConstantColorRenderingEnable,
    ConstantColorRenderingComponent(MaxwellThreeDConstantColorComponent),
    AlphaFraction,
    TirMode,
    TirControl,
    TirModulation,
    TirModulationFunction,
    CoverageToColor,
    AlphaToCoverageOverride,
    HybridAntiAliasControl,
    SampleLocations(u8),
    RasterBoundingBox,
    CheckSphVersion,
    CheckAamVersion,
    VafL2CacheControl,
    RopL2CacheControl(MaxwellThreeDRopL2CacheRequest),
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
    RenderEnableMode,
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
    ZCullStatsEnable,
    ZPassPixelCountEnable,
    ZCullCriterion,
    ZCullEnable,
    ZCullBounds,
    DrawVertexArray,
    Unsupported,
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
        MethodAction::ConstantColorRenderingComponent(MaxwellThreeDConstantColorComponent::Red)
    ),
    SET_CONSTANT_COLOR_RENDERING_GREEN => (
        0x0f48,
        "SET_CONSTANT_COLOR_RENDERING_GREEN",
        u32::MAX,
        MethodAction::ConstantColorRenderingComponent(MaxwellThreeDConstantColorComponent::Green)
    ),
    SET_CONSTANT_COLOR_RENDERING_BLUE => (
        0x0f4c,
        "SET_CONSTANT_COLOR_RENDERING_BLUE",
        u32::MAX,
        MethodAction::ConstantColorRenderingComponent(MaxwellThreeDConstantColorComponent::Blue)
    ),
    SET_CONSTANT_COLOR_RENDERING_ALPHA => (
        0x0f50,
        "SET_CONSTANT_COLOR_RENDERING_ALPHA",
        u32::MAX,
        MethodAction::ConstantColorRenderingComponent(MaxwellThreeDConstantColorComponent::Alpha)
    ),
    SET_API_MANDATED_EARLY_Z => (
        0x0210,
        "SET_API_MANDATED_EARLY_Z",
        0x0000_0001,
        MethodAction::ApiMandatedEarlyZ
    ),
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
        MethodAction::TiledCacheUnknownConfig(0)
    ),
    SET_TILED_CACHE_UNKNOWN_CONFIG_1 => (
        0x0f6c,
        "SET_TILED_CACHE_UNKNOWN_CONFIG_1",
        u32::MAX,
        MethodAction::TiledCacheUnknownConfig(1)
    ),
    SET_TILED_CACHE_UNKNOWN_CONFIG_2 => (
        0x1108,
        "SET_TILED_CACHE_UNKNOWN_CONFIG_2",
        u32::MAX,
        MethodAction::TiledCacheUnknownConfig(2)
    ),
    SET_TILED_CACHE_UNKNOWN_CONFIG_3 => (
        0x0f70,
        "SET_TILED_CACHE_UNKNOWN_CONFIG_3",
        u32::MAX,
        MethodAction::TiledCacheUnknownConfig(3)
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
        MethodAction::ShaderWatermarks(
            MaxwellThreeDShaderWatermarkTarget::VertexTessellationGeometryWarps
        )
    ),
    SET_PS_WARP_WATERMARKS => (
        0x1450,
        "SET_PS_WARP_WATERMARKS",
        u32::MAX,
        MethodAction::ShaderWatermarks(MaxwellThreeDShaderWatermarkTarget::PixelWarps)
    ),
    SET_PS_REGISTER_WATERMARKS => (
        0x1454,
        "SET_PS_REGISTER_WATERMARKS",
        u32::MAX,
        MethodAction::ShaderWatermarks(MaxwellThreeDShaderWatermarkTarget::PixelRegisters)
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
        MethodAction::SampleLocations(0)
    ),
    SAMPLE_LOCATIONS_1 => (
        0x11e4,
        "SAMPLE_LOCATIONS(1)",
        u32::MAX,
        MethodAction::SampleLocations(1)
    ),
    SAMPLE_LOCATIONS_2 => (
        0x11e8,
        "SAMPLE_LOCATIONS(2)",
        u32::MAX,
        MethodAction::SampleLocations(2)
    ),
    SAMPLE_LOCATIONS_3 => (
        0x11ec,
        "SAMPLE_LOCATIONS(3)",
        u32::MAX,
        MethodAction::SampleLocations(3)
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
        MethodAction::RopL2CacheControl(MaxwellThreeDRopL2CacheRequest::PrefetchRead)
    ),
    SET_L2_CACHE_CONTROL_FOR_ROP_NONINTERLOCKED_READ_REQUESTS => (
        0x10fc,
        "SET_L2_CACHE_CONTROL_FOR_ROP_NONINTERLOCKED_READ_REQUESTS",
        0x0000_0030,
        MethodAction::RopL2CacheControl(MaxwellThreeDRopL2CacheRequest::NoninterlockedRead)
    ),
    SET_L2_CACHE_CONTROL_FOR_ROP_INTERLOCKED_READ_REQUESTS => (
        0x1290,
        "SET_L2_CACHE_CONTROL_FOR_ROP_INTERLOCKED_READ_REQUESTS",
        0x0000_0030,
        MethodAction::RopL2CacheControl(MaxwellThreeDRopL2CacheRequest::InterlockedRead)
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
    SET_L2_CACHE_CONTROL_FOR_ROP_NONINTERLOCKED_WRITE_REQUESTS => (
        0x12d8,
        "SET_L2_CACHE_CONTROL_FOR_ROP_NONINTERLOCKED_WRITE_REQUESTS",
        0x0000_0030,
        MethodAction::RopL2CacheControl(MaxwellThreeDRopL2CacheRequest::NoninterlockedWrite)
    ),
    SET_L2_CACHE_CONTROL_FOR_ROP_INTERLOCKED_WRITE_REQUESTS => (
        0x12dc,
        "SET_L2_CACHE_CONTROL_FOR_ROP_INTERLOCKED_WRITE_REQUESTS",
        0x0000_0030,
        MethodAction::RopL2CacheControl(MaxwellThreeDRopL2CacheRequest::InterlockedWrite)
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
        MethodAction::Unsupported
    ),
    WAIT_FOR_IDLE => (
        0x0110,
        "WAIT_FOR_IDLE",
        u32::MAX,
        MethodAction::WaitForIdle
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

pub(super) struct MaxwellThreeDMmePreflight {
    pub methods: Box<[MaxwellEngineMethodDispatch]>,
    pub ordered_operations: Box<[MaxwellEngineOperation]>,
}

pub(super) fn preflight_mme_call(
    profile: MaxwellGpuProfile,
    methods: &[MaxwellMethodDispatch],
    candidate: &mut MaxwellThreeDFrontendState,
) -> Result<MaxwellThreeDMmePreflight, MaxwellEngineDispatchError> {
    let first = methods[0];
    let source = first.source();
    let offset = source.method().0 - 0x3800;
    if offset & 7 != 0 {
        return Err(MaxwellEngineDispatchError::MmeExecution {
            source,
            error: MaxwellThreeDMmeExecutionError::DataWithoutCall,
        });
    }
    let data_method = source.method().0 + 4;
    if methods.iter().skip(1).any(|method| {
        let current = method.source().method().0;
        current != source.method().0 && current != data_method
    }) {
        return Err(MaxwellEngineDispatchError::InvalidMethodEncoding {
            source,
            method_name: "CALL_MME_MACRO",
            reason: "macro parameters target a different indexed aperture",
        });
    }
    let macro_index = (offset / 8) as u8;
    let parameters = methods
        .iter()
        .map(|method| method.source().argument())
        .collect::<Vec<_>>();
    let program = candidate.mme().program();
    let ordered_operations = {
        let mut host = MmeDispatchHost {
            profile,
            source,
            candidate,
            ordered_operations: Vec::new(),
        };
        match program.execute(macro_index, &parameters, &mut host) {
            Ok(()) => {}
            Err(MaxwellThreeDMmeRunError::Execution(error)) => {
                return Err(MaxwellEngineDispatchError::MmeExecution { source, error });
            }
            Err(MaxwellThreeDMmeRunError::Host(error)) => return Err(error),
        };
        host.ordered_operations
    };
    let mut dispatches = Vec::new();
    dispatches
        .try_reserve_exact(methods.len())
        .map_err(|_| MaxwellEngineDispatchError::ResourceExhausted)?;
    for (index, method) in methods.iter().copied().enumerate() {
        let source = method.source();
        let metadata = MaxwellEngineMethodMetadata::new(
            CLASS,
            CLASS_NAME,
            source.method(),
            if index == 0 {
                "CALL_MME_MACRO"
            } else {
                "CALL_MME_DATA"
            },
        );
        dispatches.push(MaxwellEngineMethodDispatch::new(method, metadata));
    }
    Ok(MaxwellThreeDMmePreflight {
        methods: dispatches.into_boxed_slice(),
        ordered_operations: ordered_operations.into_boxed_slice(),
    })
}

struct MmeDispatchHost<'a> {
    profile: MaxwellGpuProfile,
    source: crate::MaxwellMethodSource,
    candidate: &'a mut MaxwellThreeDFrontendState,
    ordered_operations: Vec<MaxwellEngineOperation>,
}

impl MaxwellThreeDMmeHost for MmeDispatchHost<'_> {
    type Error = MaxwellEngineDispatchError;

    fn read_register(&self, method_dword: u16) -> Result<u32, Self::Error> {
        let method = GpuMethodId(u32::from(method_dword) * 4);
        self.candidate
            .raw_register(method)
            .and_then(MaxwellThreeDRegister::raw)
            .ok_or(MaxwellEngineDispatchError::MmeExecution {
                source: self.source,
                error: MaxwellThreeDMmeExecutionError::RegisterReadUnavailable { method_dword },
            })
    }

    fn emit_method(&mut self, method_dword: u16, argument: u32) -> Result<(), Self::Error> {
        let method = GpuMethodId(u32::from(method_dword) * 4);
        if is_mme_aperture(method) {
            return Err(MaxwellEngineDispatchError::MmeExecution {
                source: self.source,
                error: MaxwellThreeDMmeExecutionError::RecursiveMacroCall { method_dword },
            });
        }
        let source = self.source.emitted_by_mme(method, argument);
        let dispatch = MaxwellMethodDispatch::emitted_by_mme(source, CLASS);
        // MME-emitted methods target the live register file directly and do
        // not pass through pushbuffer shadow-RAM tracking/replay.
        let prepared = preflight_with_shadow(self.profile, dispatch, self.candidate, false)?;
        if let Some(operation) = lower_pending_operation(prepared.operation, self.candidate) {
            self.ordered_operations
                .try_reserve(1)
                .map_err(|_| MaxwellEngineDispatchError::ResourceExhausted)?;
            self.ordered_operations.push(operation);
        }
        Ok(())
    }
}

pub(super) fn preflight(
    profile: MaxwellGpuProfile,
    method: MaxwellMethodDispatch,
    candidate: &mut MaxwellThreeDFrontendState,
) -> Result<AppliedMethod, MaxwellEngineDispatchError> {
    let prepared = preflight_with_shadow(profile, method, candidate, true)?;
    let operation = lower_pending_operation(prepared.operation, candidate);
    Ok(AppliedMethod::new(
        prepared.method,
        prepared.metadata,
        operation,
    ))
}

fn lower_pending_operation(
    operation: Option<PendingOperation>,
    state: &MaxwellThreeDFrontendState,
) -> Option<MaxwellEngineOperation> {
    operation.map(|operation| match operation {
        PendingOperation::ThreeD(trigger) => {
            MaxwellEngineOperation::ThreeD(Box::new(MaxwellThreeDTriggeredOperation {
                trigger,
                state: state.operation_snapshot(),
            }))
        }
        PendingOperation::Synchronization(trigger) => {
            MaxwellEngineOperation::ThreeDSynchronization(Box::new(
                MaxwellThreeDSynchronizationOperation::new(trigger, state.operation_snapshot()),
            ))
        }
        PendingOperation::InlineToMemory(upload) => MaxwellEngineOperation::InlineToMemory(upload),
        PendingOperation::InlineConstantBuffer(upload) => {
            MaxwellEngineOperation::ThreeDInlineConstantBuffer(upload)
        }
    })
}

fn preflight_with_shadow(
    profile: MaxwellGpuProfile,
    mut method: MaxwellMethodDispatch,
    candidate: &mut MaxwellThreeDFrontendState,
    apply_shadow_ram: bool,
) -> Result<PreparedMethod, MaxwellEngineDispatchError> {
    let source = method.source();
    if is_mme_aperture(source.method()) {
        return Err(MaxwellEngineDispatchError::MmeExecution {
            source,
            error: MaxwellThreeDMmeExecutionError::DataWithoutCall,
        });
    }
    let shadow_control = candidate.mme().shadow_ram_control().value().copied();
    if apply_shadow_ram {
        let effective_argument = candidate
            .mme()
            .resolve_shadow_argument(source.method(), source.argument())
            .map_err(|error| MaxwellEngineDispatchError::MmeShadowRam { source, error })?;
        method = method.with_effective_argument(effective_argument);
    }
    let register_changed = candidate
        .raw_register(method.source().method())
        .is_none_or(|register| register.raw() != Some(method.source().argument()));
    let prepared = preflight_register(profile, method, candidate)?;
    if prepared.writes_state {
        candidate.refresh_semantic_identities(register_changed);
        candidate.record_raw_register(prepared.method.source());
    }
    if apply_shadow_ram {
        candidate
            .mme_mut()
            .track_shadow_register(shadow_control, prepared.method.source());
    }
    Ok(prepared)
}

fn preflight_register(
    profile: MaxwellGpuProfile,
    method: MaxwellMethodDispatch,
    candidate: &mut MaxwellThreeDFrontendState,
) -> Result<PreparedMethod, MaxwellEngineDispatchError> {
    let source = method.source();
    if let Some((write, method_name, upload)) =
        inline_to_memory::preflight(source, candidate.operation_state().inline_to_memory())?
    {
        candidate.apply(write);
        let metadata =
            MaxwellEngineMethodMetadata::new(CLASS, CLASS_NAME, source.method(), method_name);
        let (operation, writes_state) = upload.map_or(state_write(), state_inline_to_memory);
        return Ok(PreparedMethod::new(
            method,
            metadata,
            operation,
            writes_state,
        ));
    }
    if let Some((write, method_name)) = preflight_mme_state(source, candidate)? {
        candidate.apply(write);
        let metadata =
            MaxwellEngineMethodMetadata::new(CLASS, CLASS_NAME, source.method(), method_name);
        return Ok(PreparedMethod::new(method, metadata, None, true));
    }
    if let Some((write, method_name, upload)) = preflight_constant_buffer_load(source, candidate)? {
        candidate.apply(write);
        let metadata =
            MaxwellEngineMethodMetadata::new(CLASS, CLASS_NAME, source.method(), method_name);
        let (operation, writes_state) = upload.map_or(state_write(), state_inline_constant_buffer);
        return Ok(PreparedMethod::new(
            method,
            metadata,
            operation,
            writes_state,
        ));
    }
    if let Some((write, method_name)) = preflight_vertex_and_binding_state(source, candidate)? {
        candidate.apply(write);
        let metadata =
            MaxwellEngineMethodMetadata::new(CLASS, CLASS_NAME, source.method(), method_name);
        return Ok(PreparedMethod::new(method, metadata, None, true));
    }
    if let Some((write, method_name)) = preflight_output_state(source)? {
        candidate.apply(write);
        let metadata =
            MaxwellEngineMethodMetadata::new(CLASS, CLASS_NAME, source.method(), method_name);
        let (operation, writes_state) = if matches!(
            write,
            MaxwellThreeDStateWrite::RenderTarget(
                MaxwellThreeDRenderTargetWrite::ClearSurface { .. }
            )
        ) {
            state_operation(MaxwellThreeDOperationTrigger::ClearSurface { source })
        } else {
            state_write()
        };
        return Ok(PreparedMethod::new(
            method,
            metadata,
            operation,
            writes_state,
        ));
    }
    let Some(declaration) = METHODS
        .iter()
        .find(|declaration| declaration.metadata.method() == source.method())
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
    let (operation, writes_state) =
        match declaration.action {
            MethodAction::NoOperation => no_operation(),
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
            MethodAction::InstrumentationHeader => {
                let value = MaxwellThreeDInstrumentationValue::from_bits(source.argument());
                let write = MaxwellThreeDStateWrite::Instrumentation(
                    MaxwellThreeDInstrumentationStateWrite::Header { value, source },
                );
                candidate.apply(write);
                state_write()
            }
            MethodAction::InstrumentationData => {
                let value = MaxwellThreeDInstrumentationValue::from_bits(source.argument());
                let write = MaxwellThreeDStateWrite::Instrumentation(
                    MaxwellThreeDInstrumentationStateWrite::Data { value, source },
                );
                candidate.apply(write);
                state_write()
            }
            MethodAction::IteratedBlend => {
                let value = MaxwellThreeDIteratedBlend::parse(source.argument()).ok_or(
                    MaxwellEngineDispatchError::InvalidMethodValue {
                        source,
                        metadata: declaration.metadata,
                        defined_mask: declaration.defined_mask,
                    },
                )?;
                let write = MaxwellThreeDStateWrite::FixedFunction(
                    MaxwellThreeDFixedFunctionWrite::IteratedBlend { value, source },
                );
                candidate.apply(write);
                state_write()
            }
            MethodAction::IteratedBlendPassCount => {
                let value = MaxwellThreeDIteratedBlendPassCount::new(source.argument() as u8);
                let write = MaxwellThreeDStateWrite::FixedFunction(
                    MaxwellThreeDFixedFunctionWrite::IteratedBlendPassCount { value, source },
                );
                candidate.apply(write);
                state_write()
            }
            MethodAction::WaitForIdle => {
                synchronization_operation(MaxwellThreeDSynchronizationTrigger::WaitForIdle {
                    value: source.argument(),
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
                let write = MaxwellThreeDStateWrite::Mme(
                    MaxwellThreeDMmeStateWrite::MutableMethodControl { value, source },
                );
                candidate.apply(write);
                state_write()
            }
            MethodAction::FalconFirmwareCall4 => {
                let address = MaxwellThreeDFalconRegisterAddress::try_new(source.argument())
                    .ok_or(MaxwellEngineDispatchError::FalconFirmware {
                        source,
                        error: MaxwellThreeDFalconError::UnalignedRegisterAddress {
                            address: source.argument(),
                        },
                    })?;
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
            MethodAction::SubtilingPerfKnobA => {
                let value = MaxwellThreeDSubtilingPerfKnobA::parse(source.argument());
                let write = MaxwellThreeDStateWrite::ShaderExecution(
                    MaxwellThreeDShaderExecutionStateWrite::SubtilingPerfKnobA { value, source },
                );
                candidate.apply(write);
                state_write()
            }
            MethodAction::SubtilingPerfKnobB => {
                let value = MaxwellThreeDSubtilingPerfKnobB::new(source.argument()).ok_or(
                    MaxwellEngineDispatchError::InvalidMethodValue {
                        source,
                        metadata: declaration.metadata,
                        defined_mask: declaration.defined_mask,
                    },
                )?;
                let write = MaxwellThreeDStateWrite::ShaderExecution(
                    MaxwellThreeDShaderExecutionStateWrite::SubtilingPerfKnobB { value, source },
                );
                candidate.apply(write);
                state_write()
            }
            MethodAction::TiledCacheEnable => {
                let value = source.argument() != 0;
                let write = MaxwellThreeDStateWrite::TiledCache(
                    MaxwellThreeDTiledCacheStateWrite::Enable { value, source },
                );
                candidate.apply(write);
                state_write()
            }
            MethodAction::TiledCacheTileSize => {
                let value = MaxwellThreeDTiledCacheTileSize::parse(source.argument());
                let write = MaxwellThreeDStateWrite::TiledCache(
                    MaxwellThreeDTiledCacheStateWrite::TileSize { value, source },
                );
                candidate.apply(write);
                state_write()
            }
            MethodAction::TiledCacheUnknownConfig(index) => {
                let value = MaxwellThreeDTiledCacheUnknownConfig::from_bits(source.argument());
                let write = MaxwellThreeDStateWrite::TiledCache(
                    MaxwellThreeDTiledCacheStateWrite::UnknownConfig {
                        index,
                        value,
                        source,
                    },
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
            MethodAction::ShaderWatermarks(target) => {
                let value = MaxwellThreeDShaderWatermarkRange::parse(source.argument());
                let write = MaxwellThreeDStateWrite::ShaderExecution(
                    MaxwellThreeDShaderExecutionStateWrite::ShaderWatermarks {
                        target,
                        value,
                        source,
                    },
                );
                candidate.apply(write);
                state_write()
            }
            MethodAction::L1Configuration => {
                let value = MaxwellThreeDDirectlyAddressableMemory::parse(source.argument())
                    .ok_or(MaxwellEngineDispatchError::InvalidMethodValue {
                        source,
                        metadata: declaration.metadata,
                        defined_mask: declaration.defined_mask,
                    })?;
                let write = MaxwellThreeDStateWrite::ShaderExecution(
                    MaxwellThreeDShaderExecutionStateWrite::L1Configuration { value, source },
                );
                candidate.apply(write);
                state_write()
            }
            MethodAction::ColorReductionThresholdsEnable => {
                let value = MaxwellThreeDColorReductionThresholdsEnable::parse(source.argument())
                    .ok_or(MaxwellEngineDispatchError::InvalidMethodValue {
                    source,
                    metadata: declaration.metadata,
                    defined_mask: declaration.defined_mask,
                })?;
                let write = MaxwellThreeDStateWrite::ColorReduction(
                    MaxwellThreeDColorReductionStateWrite::Enable { value, source },
                );
                candidate.apply(write);
                state_write()
            }
            MethodAction::ColorReductionThresholdsUnorm8 => {
                let value = MaxwellThreeDColorReductionThresholdsUnorm8::parse(source.argument())
                    .ok_or(MaxwellEngineDispatchError::InvalidMethodValue {
                    source,
                    metadata: declaration.metadata,
                    defined_mask: declaration.defined_mask,
                })?;
                let write = MaxwellThreeDStateWrite::ColorReduction(
                    MaxwellThreeDColorReductionStateWrite::ThresholdsUnorm8 { value, source },
                );
                candidate.apply(write);
                state_write()
            }
            MethodAction::ColorReductionThresholdsUnorm10 => {
                let value = MaxwellThreeDColorReductionThresholdsUnorm10::parse(source.argument())
                    .ok_or(MaxwellEngineDispatchError::InvalidMethodValue {
                        source,
                        metadata: declaration.metadata,
                        defined_mask: declaration.defined_mask,
                    })?;
                let write = MaxwellThreeDStateWrite::ColorReduction(
                    MaxwellThreeDColorReductionStateWrite::ThresholdsUnorm10 { value, source },
                );
                candidate.apply(write);
                state_write()
            }
            MethodAction::ColorReductionThresholdsUnorm16 => {
                let value = MaxwellThreeDColorReductionThresholdsUnorm16::parse(source.argument())
                    .ok_or(MaxwellEngineDispatchError::InvalidMethodValue {
                        source,
                        metadata: declaration.metadata,
                        defined_mask: declaration.defined_mask,
                    })?;
                let write = MaxwellThreeDStateWrite::ColorReduction(
                    MaxwellThreeDColorReductionStateWrite::ThresholdsUnorm16 { value, source },
                );
                candidate.apply(write);
                state_write()
            }
            MethodAction::ColorReductionThresholdsFp16 => {
                let value = MaxwellThreeDColorReductionThresholdsFp16::parse(source.argument())
                    .ok_or(MaxwellEngineDispatchError::InvalidMethodValue {
                        source,
                        metadata: declaration.metadata,
                        defined_mask: declaration.defined_mask,
                    })?;
                let write = MaxwellThreeDStateWrite::ColorReduction(
                    MaxwellThreeDColorReductionStateWrite::ThresholdsFp16 { value, source },
                );
                candidate.apply(write);
                state_write()
            }
            MethodAction::ColorReductionThresholdsSrgb8 => {
                let value = MaxwellThreeDColorReductionThresholdsSrgb8::parse(source.argument())
                    .ok_or(MaxwellEngineDispatchError::InvalidMethodValue {
                        source,
                        metadata: declaration.metadata,
                        defined_mask: declaration.defined_mask,
                    })?;
                let write = MaxwellThreeDStateWrite::ColorReduction(
                    MaxwellThreeDColorReductionStateWrite::ThresholdsSrgb8 { value, source },
                );
                candidate.apply(write);
                state_write()
            }
            MethodAction::ConstantColorRenderingEnable => {
                let value = source.argument() != 0;
                let write = MaxwellThreeDStateWrite::ConstantColorRendering(
                    MaxwellThreeDConstantColorRenderingStateWrite::Enable { value, source },
                );
                candidate.apply(write);
                state_write()
            }
            MethodAction::ConstantColorRenderingComponent(component) => {
                let value = MaxwellThreeDConstantColorValue::from_bits(source.argument());
                let write = MaxwellThreeDStateWrite::ConstantColorRendering(
                    MaxwellThreeDConstantColorRenderingStateWrite::Component {
                        component,
                        value,
                        source,
                    },
                );
                candidate.apply(write);
                state_write()
            }
            MethodAction::ApiMandatedEarlyZ => {
                let value = MaxwellThreeDApiMandatedEarlyZ::parse(source.argument()).ok_or(
                    MaxwellEngineDispatchError::InvalidMethodValue {
                        source,
                        metadata: declaration.metadata,
                        defined_mask: declaration.defined_mask,
                    },
                )?;
                let write = MaxwellThreeDStateWrite::ShaderExecution(
                    MaxwellThreeDShaderExecutionStateWrite::ApiMandatedEarlyZ { value, source },
                );
                candidate.apply(write);
                state_write()
            }
            MethodAction::PostZPixelShaderImask => {
                let value = MaxwellThreeDPostZPixelShaderImask::parse(source.argument()).ok_or(
                    MaxwellEngineDispatchError::InvalidMethodValue {
                        source,
                        metadata: declaration.metadata,
                        defined_mask: declaration.defined_mask,
                    },
                )?;
                let write = MaxwellThreeDStateWrite::Coverage(
                    MaxwellThreeDCoverageStateWrite::PostZPixelShaderImask { value, source },
                );
                candidate.apply(write);
                state_write()
            }
            MethodAction::PixelShaderInterlockControl => {
                let value = MaxwellThreeDPixelShaderInterlockControl::parse(source.argument())
                    .ok_or(MaxwellEngineDispatchError::InvalidMethodValue {
                        source,
                        metadata: declaration.metadata,
                        defined_mask: declaration.defined_mask,
                    })?;
                let write = MaxwellThreeDStateWrite::ShaderExecution(
                    MaxwellThreeDShaderExecutionStateWrite::PixelShaderInterlockControl {
                        value,
                        source,
                    },
                );
                candidate.apply(write);
                state_write()
            }
            MethodAction::AlphaFraction => {
                let write = MaxwellThreeDStateWrite::AlphaFraction {
                    value: MaxwellThreeDAlphaFraction::new(source.argument() as u8),
                    source,
                };
                candidate.apply(write);
                state_write()
            }
            MethodAction::RasterBoundingBox => {
                let write = MaxwellThreeDStateWrite::RasterBoundingBox {
                    value: MaxwellThreeDRasterBoundingBox::parse(source.argument()),
                    source,
                };
                candidate.apply(write);
                state_write()
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
            MethodAction::VafL2CacheControl => {
                let value = MaxwellThreeDVafL2CacheControl::parse(source.argument()).ok_or(
                    MaxwellEngineDispatchError::InvalidMethodValue {
                        source,
                        metadata: declaration.metadata,
                        defined_mask: declaration.defined_mask,
                    },
                )?;
                let write =
                    MaxwellThreeDStateWrite::L2Cache(MaxwellThreeDL2CacheStateWrite::VafControl {
                        value,
                        source,
                    });
                candidate.apply(write);
                state_write()
            }
            MethodAction::RopL2CacheControl(request) => {
                let value = MaxwellThreeDL2CacheEvictionPolicy::parse(source.argument()).ok_or(
                    MaxwellEngineDispatchError::InvalidMethodValue {
                        source,
                        metadata: declaration.metadata,
                        defined_mask: declaration.defined_mask,
                    },
                )?;
                let write =
                    MaxwellThreeDStateWrite::L2Cache(MaxwellThreeDL2CacheStateWrite::RopPolicy {
                        request,
                        value,
                        source,
                    });
                candidate.apply(write);
                state_write()
            }
            MethodAction::ReportSemaphoreAddressUpper => {
                let write = MaxwellThreeDStateWrite::ReportSemaphore(
                    MaxwellThreeDReportSemaphoreStateWrite::AddressUpper {
                        value: source.argument() as u8,
                        source,
                    },
                );
                candidate.apply(write);
                state_write()
            }
            MethodAction::ReportSemaphoreAddressLower => {
                let write = MaxwellThreeDStateWrite::ReportSemaphore(
                    MaxwellThreeDReportSemaphoreStateWrite::AddressLower {
                        value: source.argument(),
                        source,
                    },
                );
                candidate.apply(write);
                state_write()
            }
            MethodAction::ReportSemaphorePayload => {
                let write = MaxwellThreeDStateWrite::ReportSemaphore(
                    MaxwellThreeDReportSemaphoreStateWrite::Payload {
                        value: source.argument(),
                        source,
                    },
                );
                candidate.apply(write);
                state_write()
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
            MethodAction::PointSize => {
                let write = MaxwellThreeDStateWrite::PointSize {
                    value: MaxwellThreeDPointSize::from_bits(source.argument()),
                    source,
                };
                candidate.apply(write);
                state_write()
            }
            MethodAction::PointSpriteSelect => {
                let value = MaxwellThreeDPointSpriteSelect::parse(source.argument()).ok_or(
                    MaxwellEngineDispatchError::InvalidMethodValue {
                        source,
                        metadata: declaration.metadata,
                        defined_mask: declaration.defined_mask,
                    },
                )?;
                let write = MaxwellThreeDStateWrite::PointSpriteSelect { value, source };
                candidate.apply(write);
                state_write()
            }
            MethodAction::PointCenterMode => {
                let value = MaxwellThreeDPointCenterMode::parse(source.argument()).ok_or(
                    MaxwellEngineDispatchError::InvalidMethodValue {
                        source,
                        metadata: declaration.metadata,
                        defined_mask: declaration.defined_mask,
                    },
                )?;
                let write = MaxwellThreeDStateWrite::PointCenterMode { value, source };
                candidate.apply(write);
                state_write()
            }
            MethodAction::EdgeFlag => {
                let value = MaxwellThreeDEdgeFlag::parse(source.argument()).ok_or(
                    MaxwellEngineDispatchError::InvalidMethodValue {
                        source,
                        metadata: declaration.metadata,
                        defined_mask: declaration.defined_mask,
                    },
                )?;
                let write = MaxwellThreeDStateWrite::EdgeFlag { value, source };
                candidate.apply(write);
                state_write()
            }
            MethodAction::InvalidateShaderCachesNoWfi => {
                let caches = MaxwellShaderCacheInvalidation::new(
                    source.argument() & 1 != 0,
                    source.argument() & (1 << 4) != 0,
                    source.argument() & (1 << 12) != 0,
                );
                synchronization_operation(
                    MaxwellThreeDSynchronizationTrigger::InvalidateShaderCachesNoWfi {
                        caches,
                        source,
                    },
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
            MethodAction::ViewportZClip => {
                let value = MaxwellThreeDViewportZClipRange::parse(source.argument()).ok_or(
                    MaxwellEngineDispatchError::InvalidMethodValue {
                        source,
                        metadata: declaration.metadata,
                        defined_mask: declaration.defined_mask,
                    },
                )?;
                let write = MaxwellThreeDStateWrite::ViewportZClip { value, source };
                candidate.apply(write);
                state_write()
            }
            MethodAction::RenderEnableMode => {
                let value = MaxwellThreeDRenderEnableMode::parse(source.argument()).ok_or(
                    MaxwellEngineDispatchError::InvalidMethodValue {
                        source,
                        metadata: declaration.metadata,
                        defined_mask: declaration.defined_mask,
                    },
                )?;
                let write = MaxwellThreeDStateWrite::RenderEnable(
                    MaxwellThreeDRenderEnableStateWrite::Mode { value, source },
                );
                candidate.apply(write);
                state_write()
            }
            MethodAction::PsOutputSampleMaskUsage => {
                let value = MaxwellThreeDPsOutputSampleMaskUsage::parse(source.argument()).ok_or(
                    MaxwellEngineDispatchError::InvalidMethodValue {
                        source,
                        metadata: declaration.metadata,
                        defined_mask: declaration.defined_mask,
                    },
                )?;
                let write = MaxwellThreeDStateWrite::Coverage(
                    MaxwellThreeDCoverageStateWrite::PsOutputSampleMaskUsage { value, source },
                );
                candidate.apply(write);
                state_write()
            }
            MethodAction::HybridAntiAliasControl => {
                let value = MaxwellThreeDHybridAntiAliasControl::parse(source.argument()).ok_or(
                    MaxwellEngineDispatchError::InvalidMethodValue {
                        source,
                        metadata: declaration.metadata,
                        defined_mask: declaration.defined_mask,
                    },
                )?;
                let write = MaxwellThreeDStateWrite::Coverage(
                    MaxwellThreeDCoverageStateWrite::HybridAntiAliasControl { value, source },
                );
                candidate.apply(write);
                state_write()
            }
            MethodAction::TirMode => {
                let value = MaxwellThreeDTirMode::parse(source.argument()).ok_or(
                    MaxwellEngineDispatchError::InvalidMethodValue {
                        source,
                        metadata: declaration.metadata,
                        defined_mask: declaration.defined_mask,
                    },
                )?;
                let write =
                    MaxwellThreeDStateWrite::Coverage(MaxwellThreeDCoverageStateWrite::TirMode {
                        value,
                        source,
                    });
                candidate.apply(write);
                state_write()
            }
            MethodAction::TirControl => {
                let value = MaxwellThreeDTirControl::parse(source.argument()).ok_or(
                    MaxwellEngineDispatchError::InvalidMethodValue {
                        source,
                        metadata: declaration.metadata,
                        defined_mask: declaration.defined_mask,
                    },
                )?;
                let write = MaxwellThreeDStateWrite::Coverage(
                    MaxwellThreeDCoverageStateWrite::TirControl { value, source },
                );
                candidate.apply(write);
                state_write()
            }
            MethodAction::TirModulation => {
                let value = MaxwellThreeDTirModulationComponentSelect::parse(source.argument())
                    .ok_or(MaxwellEngineDispatchError::InvalidMethodValue {
                        source,
                        metadata: declaration.metadata,
                        defined_mask: declaration.defined_mask,
                    })?;
                let write = MaxwellThreeDStateWrite::Coverage(
                    MaxwellThreeDCoverageStateWrite::TirModulation { value, source },
                );
                candidate.apply(write);
                state_write()
            }
            MethodAction::TirModulationFunction => {
                let value = MaxwellThreeDTirModulationFunction::parse(source.argument()).ok_or(
                    MaxwellEngineDispatchError::InvalidMethodValue {
                        source,
                        metadata: declaration.metadata,
                        defined_mask: declaration.defined_mask,
                    },
                )?;
                let write = MaxwellThreeDStateWrite::Coverage(
                    MaxwellThreeDCoverageStateWrite::TirModulationFunction { value, source },
                );
                candidate.apply(write);
                state_write()
            }
            MethodAction::CoverageToColor => {
                let value = MaxwellThreeDCoverageToColor::parse(source.argument()).ok_or(
                    MaxwellEngineDispatchError::InvalidMethodValue {
                        source,
                        metadata: declaration.metadata,
                        defined_mask: declaration.defined_mask,
                    },
                )?;
                let write = MaxwellThreeDStateWrite::Coverage(
                    MaxwellThreeDCoverageStateWrite::CoverageToColor { value, source },
                );
                candidate.apply(write);
                state_write()
            }
            MethodAction::AlphaToCoverageOverride => {
                let value = MaxwellThreeDAlphaToCoverageOverride::parse(source.argument()).ok_or(
                    MaxwellEngineDispatchError::InvalidMethodValue {
                        source,
                        metadata: declaration.metadata,
                        defined_mask: declaration.defined_mask,
                    },
                )?;
                let write = MaxwellThreeDStateWrite::Coverage(
                    MaxwellThreeDCoverageStateWrite::AlphaToCoverageOverride { value, source },
                );
                candidate.apply(write);
                state_write()
            }
            MethodAction::SampleLocations(group) => {
                let value = MaxwellThreeDSampleLocationGroup::parse(source.argument());
                let write = MaxwellThreeDStateWrite::Coverage(
                    MaxwellThreeDCoverageStateWrite::SampleLocations {
                        group,
                        value,
                        source,
                    },
                );
                candidate.apply(write);
                state_write()
            }
            MethodAction::RenderEnableControl => {
                let value = MaxwellThreeDConditionalLoadConstantBuffer::parse(source.argument())
                    .ok_or(MaxwellEngineDispatchError::InvalidMethodValue {
                        source,
                        metadata: declaration.metadata,
                        defined_mask: declaration.defined_mask,
                    })?;
                let write = MaxwellThreeDStateWrite::RenderEnable(
                    MaxwellThreeDRenderEnableStateWrite::ConditionalLoadConstantBuffer {
                        value,
                        source,
                    },
                );
                candidate.apply(write);
                state_write()
            }
            MethodAction::VisibleCallLimit => {
                let value = MaxwellThreeDVisibleCallLimit::parse(source.argument()).ok_or(
                    MaxwellEngineDispatchError::InvalidMethodValue {
                        source,
                        metadata: declaration.metadata,
                        defined_mask: declaration.defined_mask,
                    },
                )?;
                let write = MaxwellThreeDStateWrite::ShaderExecution(
                    MaxwellThreeDShaderExecutionStateWrite::VisibleCallLimit { value, source },
                );
                candidate.apply(write);
                state_write()
            }
            MethodAction::SmTimeoutCounterBit => {
                let value = MaxwellThreeDSmTimeoutCounterBit::new(source.argument()).ok_or(
                    MaxwellEngineDispatchError::InvalidMethodValue {
                        source,
                        metadata: declaration.metadata,
                        defined_mask: declaration.defined_mask,
                    },
                )?;
                let write = MaxwellThreeDStateWrite::ShaderExecution(
                    MaxwellThreeDShaderExecutionStateWrite::SmTimeoutCounterBit { value, source },
                );
                candidate.apply(write);
                state_write()
            }
            MethodAction::ShaderExceptionsEnable => {
                let value = MaxwellThreeDShaderExceptionsEnable::parse(source.argument()).ok_or(
                    MaxwellEngineDispatchError::InvalidMethodValue {
                        source,
                        metadata: declaration.metadata,
                        defined_mask: declaration.defined_mask,
                    },
                )?;
                let write = MaxwellThreeDStateWrite::ShaderExecution(
                    MaxwellThreeDShaderExecutionStateWrite::ShaderExceptionsEnable {
                        value,
                        source,
                    },
                );
                candidate.apply(write);
                state_write()
            }
            MethodAction::ShaderLocalMemoryWindowBaseAddress => {
                let write = MaxwellThreeDStateWrite::ShaderExecution(
                    MaxwellThreeDShaderExecutionStateWrite::ShaderLocalMemoryWindowBaseAddress {
                        value: source.argument(),
                        source,
                    },
                );
                candidate.apply(write);
                state_write()
            }
            MethodAction::ShaderLocalMemoryAddressUpper => {
                let write = MaxwellThreeDStateWrite::ShaderExecution(
                    MaxwellThreeDShaderExecutionStateWrite::ShaderLocalMemoryAddressUpper {
                        value: source.argument() as u8,
                        source,
                    },
                );
                candidate.apply(write);
                state_write()
            }
            MethodAction::ShaderLocalMemoryAddressLower => {
                let write = MaxwellThreeDStateWrite::ShaderExecution(
                    MaxwellThreeDShaderExecutionStateWrite::ShaderLocalMemoryAddressLower {
                        value: source.argument(),
                        source,
                    },
                );
                candidate.apply(write);
                state_write()
            }
            MethodAction::ShaderLocalMemorySizeUpper => {
                let write = MaxwellThreeDStateWrite::ShaderExecution(
                    MaxwellThreeDShaderExecutionStateWrite::ShaderLocalMemorySizeUpper {
                        value: source.argument() as u8,
                        source,
                    },
                );
                candidate.apply(write);
                state_write()
            }
            MethodAction::ShaderLocalMemorySizeLower => {
                let write = MaxwellThreeDStateWrite::ShaderExecution(
                    MaxwellThreeDShaderExecutionStateWrite::ShaderLocalMemorySizeLower {
                        value: source.argument(),
                        source,
                    },
                );
                candidate.apply(write);
                state_write()
            }
            MethodAction::ShaderLocalMemoryDefaultSizePerWarp => {
                let value = MaxwellThreeDShaderLocalMemoryPerWarpSize::new(source.argument())
                    .ok_or(MaxwellEngineDispatchError::InvalidMethodValue {
                        source,
                        metadata: declaration.metadata,
                        defined_mask: declaration.defined_mask,
                    })?;
                let write = MaxwellThreeDStateWrite::ShaderExecution(
                    MaxwellThreeDShaderExecutionStateWrite::ShaderLocalMemoryDefaultSizePerWarp {
                        value,
                        source,
                    },
                );
                candidate.apply(write);
                state_write()
            }
            MethodAction::CsaaEnable => {
                let value = MaxwellThreeDCsaaEnable::parse(source.argument()).ok_or(
                    MaxwellEngineDispatchError::InvalidMethodValue {
                        source,
                        metadata: declaration.metadata,
                        defined_mask: declaration.defined_mask,
                    },
                )?;
                let write = MaxwellThreeDStateWrite::Coverage(
                    MaxwellThreeDCoverageStateWrite::CsaaEnable { value, source },
                );
                candidate.apply(write);
                state_write()
            }
            MethodAction::AliasedLineWidthEnable => {
                let value = MaxwellThreeDAliasedLineWidthEnable::parse(source.argument()).ok_or(
                    MaxwellEngineDispatchError::InvalidMethodValue {
                        source,
                        metadata: declaration.metadata,
                        defined_mask: declaration.defined_mask,
                    },
                )?;
                let write = MaxwellThreeDStateWrite::Line(
                    MaxwellThreeDLineStateWrite::AliasedLineWidthEnable { value, source },
                );
                candidate.apply(write);
                state_write()
            }
            MethodAction::ActiveZCullRegion => {
                let value = MaxwellThreeDZCullRegionId::new(source.argument()).ok_or(
                    MaxwellEngineDispatchError::InvalidMethodValue {
                        source,
                        metadata: declaration.metadata,
                        defined_mask: declaration.defined_mask,
                    },
                )?;
                let write =
                    MaxwellThreeDStateWrite::ZCull(MaxwellThreeDZCullStateWrite::ActiveRegion {
                        value,
                        source,
                    });
                candidate.apply(write);
                state_write()
            }
            MethodAction::ZCullStatsEnable => {
                let value = MaxwellThreeDZCullStatsEnable::parse(source.argument()).ok_or(
                    MaxwellEngineDispatchError::InvalidMethodValue {
                        source,
                        metadata: declaration.metadata,
                        defined_mask: declaration.defined_mask,
                    },
                )?;
                let write =
                    MaxwellThreeDStateWrite::ZCull(MaxwellThreeDZCullStateWrite::StatsEnable {
                        value,
                        source,
                    });
                candidate.apply(write);
                state_write()
            }
            MethodAction::ZPassPixelCountEnable => {
                let value = MaxwellThreeDZPassPixelCountEnable::parse(source.argument()).ok_or(
                    MaxwellEngineDispatchError::InvalidMethodValue {
                        source,
                        metadata: declaration.metadata,
                        defined_mask: declaration.defined_mask,
                    },
                )?;
                let write = MaxwellThreeDStateWrite::Counter(
                    MaxwellThreeDCounterStateWrite::ZPassPixelCountEnable { value, source },
                );
                candidate.apply(write);
                state_write()
            }
            MethodAction::ZCullCriterion => {
                let value = MaxwellThreeDZCullCriterion::parse(source.argument()).ok_or(
                    MaxwellEngineDispatchError::InvalidMethodValue {
                        source,
                        metadata: declaration.metadata,
                        defined_mask: declaration.defined_mask,
                    },
                )?;
                let write =
                    MaxwellThreeDStateWrite::ZCull(MaxwellThreeDZCullStateWrite::Criterion {
                        value,
                        source,
                    });
                candidate.apply(write);
                state_write()
            }
            MethodAction::ZCullEnable => {
                let value = MaxwellThreeDZCullEnable::parse(source.argument()).ok_or(
                    MaxwellEngineDispatchError::InvalidMethodValue {
                        source,
                        metadata: declaration.metadata,
                        defined_mask: declaration.defined_mask,
                    },
                )?;
                let write = MaxwellThreeDStateWrite::ZCull(MaxwellThreeDZCullStateWrite::Enable {
                    value,
                    source,
                });
                candidate.apply(write);
                state_write()
            }
            MethodAction::ZCullBounds => {
                let value = MaxwellThreeDZCullBounds::parse(source.argument()).ok_or(
                    MaxwellEngineDispatchError::InvalidMethodValue {
                        source,
                        metadata: declaration.metadata,
                        defined_mask: declaration.defined_mask,
                    },
                )?;
                let write = MaxwellThreeDStateWrite::ZCull(MaxwellThreeDZCullStateWrite::Bounds {
                    value,
                    source,
                });
                candidate.apply(write);
                state_write()
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
            MethodAction::Unsupported => {
                return Err(MaxwellEngineDispatchError::UnsupportedMethod {
                    source,
                    metadata: declaration.metadata,
                });
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
        method,
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
            let offset = bindings
                .constant_buffer_load()
                .next_offset()
                .ok_or_else(|| {
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
                    value: raw,
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

    let vertex = if (0x1160..0x11e0).contains(&method) && method & 3 == 0 {
        let attribute = ((method - 0x1160) / 4) as u8;
        let value = MaxwellThreeDVertexAttributeFormat::parse(raw).ok_or_else(|| {
            invalid_encoding(
                source,
                "SET_VERTEX_ATTRIBUTE",
                "invalid stream, component-width, or numerical-type encoding",
            )
        })?;
        Some((
            V::Attribute {
                attribute,
                value,
                source,
            },
            "SET_VERTEX_ATTRIBUTE",
        ))
    } else if (0x1880..0x1900).contains(&method) && method & 3 == 0 {
        let stream = ((method - 0x1880) / 4) as u8;
        Some((
            V::StreamInstanced {
                stream,
                value: checked_bool(source, "SET_VERTEX_STREAM_INSTANCE")?,
                source,
            },
            "SET_VERTEX_STREAM_INSTANCE",
        ))
    } else if (0x1c00..0x1e00).contains(&method) {
        let stream = ((method - 0x1c00) / 0x10) as u8;
        let field = (method - 0x1c00) % 0x10;
        let (write, name) = match field {
            0 => (
                V::StreamFormat {
                    stream,
                    value: MaxwellThreeDVertexStreamFormat::parse(raw).ok_or_else(|| {
                        invalid_encoding(
                            source,
                            "SET_VERTEX_STREAM_FORMAT",
                            "undefined vertex-stream format bits",
                        )
                    })?,
                    source,
                },
                "SET_VERTEX_STREAM_FORMAT",
            ),
            4 if raw <= 0xff => (
                V::StreamAddressUpper {
                    stream,
                    value: raw as u8,
                    source,
                },
                "SET_VERTEX_STREAM_LOCATION_A",
            ),
            8 => (
                V::StreamAddressLower {
                    stream,
                    value: raw,
                    source,
                },
                "SET_VERTEX_STREAM_LOCATION_B",
            ),
            12 => (
                V::StreamFrequency {
                    stream,
                    value: raw,
                    source,
                },
                "SET_VERTEX_STREAM_FREQUENCY",
            ),
            4 => {
                return Err(invalid_encoding(
                    source,
                    "SET_VERTEX_STREAM_LOCATION_A",
                    "GPU address exceeds the 40-bit field",
                ));
            }
            _ => return Ok(None),
        };
        Some((write, name))
    } else if (0x1f00..0x2000).contains(&method) && method & 3 == 0 {
        let stream = if method < 0x1f80 {
            ((method - 0x1f00) / 8) as u8
        } else {
            16 + ((method - 0x1f80) / 8) as u8
        };
        let upper = method & 7 == 0;
        if upper && raw > 0xff {
            return Err(invalid_encoding(
                source,
                "SET_VERTEX_STREAM_LIMIT",
                "GPU address exceeds the 40-bit field",
            ));
        }
        Some(if upper {
            (
                V::StreamLimitUpper {
                    stream,
                    value: raw as u8,
                    source,
                },
                "SET_VERTEX_STREAM_LIMIT_A",
            )
        } else {
            (
                V::StreamLimitLower {
                    stream,
                    value: raw,
                    source,
                },
                "SET_VERTEX_STREAM_LIMIT_B",
            )
        })
    } else {
        match method {
            0x0374 => Some((
                V::BalancedPrimitiveWorkload {
                    value: MaxwellThreeDBalancedPrimitiveWorkload::parse(raw).ok_or_else(|| {
                        invalid_encoding(
                            source,
                            "SET_BALANCED_PRIMITIVE_WORKLOAD",
                            "reserved bits are set",
                        )
                    })?,
                    source,
                },
                "SET_BALANCED_PRIMITIVE_WORKLOAD",
            )),
            0x0f84 if raw <= 0xff => Some((
                V::StreamSubstituteAddressUpper {
                    value: raw as u8,
                    source,
                },
                "SET_VERTEX_STREAM_SUBSTITUTE_A",
            )),
            0x0f84 => {
                return Err(invalid_encoding(
                    source,
                    "SET_VERTEX_STREAM_SUBSTITUTE_A",
                    "GPU address exceeds the 40-bit field",
                ));
            }
            0x0f88 => Some((
                V::StreamSubstituteAddressLower { value: raw, source },
                "SET_VERTEX_STREAM_SUBSTITUTE_B",
            )),
            0x02d0 => Some((
                V::PrimitiveCircularBufferThrottle {
                    value: MaxwellThreeDPrimitiveCircularBufferThrottle::new(raw).ok_or_else(
                        || {
                            invalid_encoding(
                                source,
                                "SET_PRIM_CIRCULAR_BUFFER_THROTTLE",
                                "reserved bits are set",
                            )
                        },
                    )?,
                    source,
                },
                "SET_PRIM_CIRCULAR_BUFFER_THROTTLE",
            )),
            0x0dcc if raw <= 0xff => Some((
                V::PatchSize {
                    value: MaxwellThreeDPatchSize::new(raw as u8),
                    source,
                },
                "SET_PATCH",
            )),
            0x0dcc => {
                return Err(invalid_encoding(
                    source,
                    "SET_PATCH",
                    "reserved bits are set",
                ));
            }
            0x0de8 => Some((
                V::VertexArrayPrimitiveRestartEnable {
                    value: MaxwellThreeDVertexArrayPrimitiveRestartEnable::parse(raw).ok_or_else(
                        || {
                            invalid_encoding(
                                source,
                                "SET_DA_PRIMITIVE_RESTART_VERTEX_ARRAY",
                                "undefined boolean encoding or reserved bits",
                            )
                        },
                    )?,
                    source,
                },
                "SET_DA_PRIMITIVE_RESTART_VERTEX_ARRAY",
            )),
            0x0d74 => Some((
                V::VertexArrayStart { value: raw, source },
                "SET_VERTEX_ARRAY_START",
            )),
            // NVIDIA publishes both global draw-index registers as complete
            // 32-bit fields.
            // https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L2633-L2637
            0x1434 => Some((
                V::GlobalBaseVertexIndex { value: raw, source },
                "SET_GLOBAL_BASE_VERTEX_INDEX",
            )),
            0x1438 => Some((
                V::GlobalBaseInstanceIndex { value: raw, source },
                "SET_GLOBAL_BASE_INSTANCE_INDEX",
            )),
            0x1610 => Some((
                V::AttributeDefaults {
                    value: MaxwellThreeDAttributeDefaults::parse(raw).ok_or_else(|| {
                        invalid_encoding(
                            source,
                            "SET_ATTRIBUTE_DEFAULT",
                            "reserved attribute-default bits are set",
                        )
                    })?,
                    source,
                },
                "SET_ATTRIBUTE_DEFAULT",
            )),
            // NVIDIA defines END as a one-bit method immediately before BEGIN.
            // Retaining it as a state transition closes the current primitive
            // sequence without mutating draw snapshots captured earlier.
            // https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L3022-L3025
            0x1614 => Some((
                V::End {
                    value: checked_bool(source, "END")?,
                    source,
                },
                "END",
            )),
            0x1618 => Some((
                V::Begin {
                    value: MaxwellThreeDBegin::parse(raw).ok_or_else(|| {
                        invalid_encoding(source, "BEGIN", "invalid topology or begin modifier")
                    })?,
                    source,
                },
                "BEGIN",
            )),
            0x1644 => Some((
                V::PrimitiveRestartEnable {
                    value: checked_bool(source, "SET_DA_PRIMITIVE_RESTART")?,
                    source,
                },
                "SET_DA_PRIMITIVE_RESTART",
            )),
            0x1648 => Some((
                V::PrimitiveRestartIndex { value: raw, source },
                "SET_DA_PRIMITIVE_RESTART_INDEX",
            )),
            0x164c => Some((
                V::VertexIdUsesArrayStart {
                    value: MaxwellThreeDVertexIdUsesArrayStart::parse(raw).ok_or_else(|| {
                        invalid_encoding(
                            source,
                            "SET_DA_OUTPUT",
                            "undefined boolean encoding or reserved bits",
                        )
                    })?,
                    source,
                },
                "SET_DA_OUTPUT",
            )),
            0x17c8 if raw <= 0xff => Some((
                V::IndexAddressUpper {
                    value: raw as u8,
                    source,
                },
                "SET_INDEX_BUFFER_A",
            )),
            0x17cc => Some((
                V::IndexAddressLower { value: raw, source },
                "SET_INDEX_BUFFER_B",
            )),
            0x17d0 if raw <= 0xff => Some((
                V::IndexLimitUpper {
                    value: raw as u8,
                    source,
                },
                "SET_INDEX_BUFFER_C",
            )),
            0x17d4 => Some((
                V::IndexLimitLower { value: raw, source },
                "SET_INDEX_BUFFER_D",
            )),
            0x17d8 => Some((
                V::IndexElementSize {
                    value: MaxwellThreeDIndexElementSize::parse(raw).ok_or_else(|| {
                        invalid_encoding(source, "SET_INDEX_BUFFER_E", "invalid index element size")
                    })?,
                    source,
                },
                "SET_INDEX_BUFFER_E",
            )),
            0x17dc => Some((V::IndexFirst { value: raw, source }, "SET_INDEX_BUFFER_F")),
            0x1948 => Some((
                V::TopologyOverride {
                    value: checked_bool(source, "SET_PRIMITIVE_TOPOLOGY_CONTROL")?,
                    source,
                },
                "SET_PRIMITIVE_TOPOLOGY_CONTROL",
            )),
            0x1970 => Some((
                V::Topology {
                    value: MaxwellThreeDPrimitiveTopology::parse(raw).ok_or_else(|| {
                        invalid_encoding(
                            source,
                            "SET_PRIMITIVE_TOPOLOGY",
                            "unknown primitive topology",
                        )
                    })?,
                    source,
                },
                "SET_PRIMITIVE_TOPOLOGY",
            )),
            0x17c8 | 0x17d0 => {
                return Err(invalid_encoding(
                    source,
                    "SET_INDEX_BUFFER",
                    "GPU address exceeds the 40-bit field",
                ));
            }
            _ => None,
        }
    };
    if let Some((write, name)) = vertex {
        return Ok(Some((MaxwellThreeDStateWrite::VertexInput(write), name)));
    }

    let binding = if (0x2000..0x2180).contains(&method) {
        let pipeline = ((method - 0x2000) / 0x40) as u8;
        let field = (method - 0x2000) % 0x40;
        match field {
            0 if raw & !0x71 == 0 => {
                let stage = MaxwellThreeDShaderStage::parse((raw >> 4) & 7).ok_or_else(|| {
                    invalid_encoding(source, "SET_PIPELINE_SHADER", "unknown shader stage")
                })?;
                Some((
                    B::PipelineShader {
                        pipeline,
                        enabled: raw & 1 != 0,
                        stage,
                        source,
                    },
                    "SET_PIPELINE_SHADER",
                ))
            }
            0 => {
                return Err(invalid_encoding(
                    source,
                    "SET_PIPELINE_SHADER",
                    "undefined pipeline shader bits",
                ));
            }
            // NVIDIA defines one full-width program offset and one eight-bit
            // register count for each of the six pipeline slots.
            // https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L3947-L3954
            0x04 => Some((
                B::PipelineProgram {
                    pipeline,
                    offset: raw,
                    source,
                },
                "SET_PIPELINE_PROGRAM",
            )),
            0x0c if raw <= u32::from(u8::MAX) => Some((
                B::PipelineRegisterCount {
                    pipeline,
                    count: raw as u8,
                    source,
                },
                "SET_PIPELINE_REGISTER_COUNT",
            )),
            0x0c => {
                return Err(invalid_encoding(
                    source,
                    "SET_PIPELINE_REGISTER_COUNT",
                    "register count exceeds the eight-bit field",
                ));
            }
            0x10 if raw <= 7 => Some((
                B::PipelineGroup {
                    pipeline,
                    group: raw as u8,
                    source,
                },
                "SET_PIPELINE_BINDING",
            )),
            0x10 => {
                return Err(invalid_encoding(
                    source,
                    "SET_PIPELINE_BINDING",
                    "binding group exceeds the three-bit field",
                ));
            }
            _ => None,
        }
    } else if (0x0324..=0x0338).contains(&method) && method.is_multiple_of(4) {
        Some((
            B::TessellationLod {
                level: MaxwellThreeDTessellationLod::from_index(((method - 0x0324) / 4) as u8),
                value: raw,
                source,
            },
            "SET_TESSELLATION_LOD",
        ))
    } else if (0x2400..0x2500).contains(&method) && (method - 0x2400) % 0x20 == 0x10 {
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
        Some((
            B::BindConstantBuffer {
                group,
                slot,
                enabled,
                address,
                size,
                source,
            },
            "BIND_GROUP_CONSTANT_BUFFER",
        ))
    } else {
        match method {
            0x0310 => Some((
                B::SpaVersion {
                    value: MaxwellSpaVersion::parse(raw).ok_or_else(|| {
                        invalid_encoding(source, "SET_SPA_VERSION", "reserved bits are set")
                    })?,
                    source,
                },
                "SET_SPA_VERSION",
            )),
            0x1608 if raw <= 0xff => Some((
                B::ProgramRegionAddressUpper {
                    value: raw as u8,
                    source,
                },
                "SET_PROGRAM_REGION_A",
            )),
            0x1608 => {
                return Err(invalid_encoding(
                    source,
                    "SET_PROGRAM_REGION_A",
                    "GPU address exceeds the 40-bit field",
                ));
            }
            0x160c => Some((
                B::ProgramRegionAddressLower { value: raw, source },
                "SET_PROGRAM_REGION_B",
            )),
            0x0f10 => Some((
                B::MaxwellTextureHeaders {
                    value: checked_bool(source, "SET_SELECT_MAXWELL_TEXTURE_HEADERS")?,
                    source,
                },
                "SET_SELECT_MAXWELL_TEXTURE_HEADERS",
            )),
            0x1234 if raw <= 1 => Some((
                B::SamplerBinding {
                    value: if raw == 0 {
                        MaxwellThreeDSamplerBindingMode::Independent
                    } else {
                        MaxwellThreeDSamplerBindingMode::ViaTextureHeader
                    },
                    source,
                },
                "SET_SAMPLER_BINDING",
            )),
            0x155c if raw <= 0xff => Some((
                B::SamplerAddressUpper {
                    value: raw as u8,
                    source,
                },
                "SET_TEX_SAMPLER_POOL_A",
            )),
            0x1560 => Some((
                B::SamplerAddressLower { value: raw, source },
                "SET_TEX_SAMPLER_POOL_B",
            )),
            0x1564 if raw <= 0x0f_ffff => Some((
                B::SamplerMaximumIndex { value: raw, source },
                "SET_TEX_SAMPLER_POOL_C",
            )),
            0x1574 if raw <= 0xff => Some((
                B::TextureHeaderAddressUpper {
                    value: raw as u8,
                    source,
                },
                "SET_TEX_HEADER_POOL_A",
            )),
            0x1578 => Some((
                B::TextureHeaderAddressLower { value: raw, source },
                "SET_TEX_HEADER_POOL_B",
            )),
            0x157c if raw <= 0x3f_ffff => Some((
                B::TextureHeaderMaximumIndex { value: raw, source },
                "SET_TEX_HEADER_POOL_C",
            )),
            0x2380 if raw <= 0x1_ffff => Some((
                B::SelectorSize { size: raw, source },
                "SET_CONSTANT_BUFFER_SELECTOR_A",
            )),
            0x2384 if raw <= 0xff => Some((
                B::SelectorAddressUpper {
                    value: raw as u8,
                    source,
                },
                "SET_CONSTANT_BUFFER_SELECTOR_B",
            )),
            0x2388 => Some((
                B::SelectorAddressLower { value: raw, source },
                "SET_CONSTANT_BUFFER_SELECTOR_C",
            )),
            0x2608 if raw <= 0x1f => Some((
                B::BindlessTextureSlot {
                    value: raw as u8,
                    source,
                },
                "SET_BINDLESS_TEXTURE",
            )),
            0x1234 | 0x155c | 0x1564 | 0x1574 | 0x157c | 0x2380 | 0x2384 | 0x2608 => {
                return Err(invalid_encoding(
                    source,
                    "SHADER_BINDING",
                    "argument exceeds its verified field",
                ));
            }
            _ => None,
        }
    };
    Ok(binding.map(|(write, name)| (MaxwellThreeDStateWrite::ShaderBinding(write), name)))
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

fn preflight_output_state(
    source: crate::MaxwellMethodSource,
) -> Result<Option<(MaxwellThreeDStateWrite, &'static str)>, MaxwellEngineDispatchError> {
    let method = source.method().0;
    let raw = source.argument();

    let line_write = match method {
        // NVIDIA publishes these line-rasterization registers in the pinned
        // MAXWELL_B class header.
        // https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L2610-L2611
        // https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L2785-L2788
        // https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L3108-L3121
        0x0f8c => Some((
            MaxwellThreeDLineStateWrite::PolygonClipGeneratedEdge {
                value: MaxwellThreeDPolygonClipGeneratedEdge::parse(raw).ok_or_else(|| {
                    invalid_encoding(
                        source,
                        "SET_LINE_MODE_POLYGON_CLIP",
                        "expected generated-edge selector 0 or 1",
                    )
                })?,
                source,
            },
            "SET_LINE_MODE_POLYGON_CLIP",
        )),
        0x13b4 => Some((
            MaxwellThreeDLineStateWrite::AliasedLineWidth {
                value: MaxwellThreeDRawValue::new(raw),
                source,
            },
            "SET_ALIASED_LINE_WIDTH_FLOAT",
        )),
        0x1570 => Some((
            MaxwellThreeDLineStateWrite::AntiAliasedLineEnable {
                value: MaxwellThreeDAntiAliasedLineEnable::parse(raw).ok_or_else(|| {
                    invalid_encoding(source, "SET_ANTI_ALIASED_LINE", "expected boolean 0 or 1")
                })?,
                source,
            },
            "SET_ANTI_ALIASED_LINE",
        )),
        0x166c => Some((
            MaxwellThreeDLineStateWrite::StippleEnable {
                value: checked_bool(source, "SET_LINE_STIPPLE")?,
                source,
            },
            "SET_LINE_STIPPLE",
        )),
        0x1680 => Some((
            MaxwellThreeDLineStateWrite::StippleParameters {
                value: MaxwellThreeDLineStippleParameters::parse(raw).ok_or_else(|| {
                    invalid_encoding(
                        source,
                        "SET_LINE_STIPPLE_PARAMETERS",
                        "reserved bits are set",
                    )
                })?,
                source,
            },
            "SET_LINE_STIPPLE_PARAMETERS",
        )),
        _ => None,
    };
    if let Some((write, name)) = line_write {
        return Ok(Some((MaxwellThreeDStateWrite::Line(write), name)));
    }

    let point_write = match method {
        // NVIDIA publishes the fill-via-triangle modes and conservative-raster
        // boolean in the pinned public MAXWELL_B class header.
        // https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L1822-L1840
        0x113c => Some((
            MaxwellThreeDStateWrite::FillViaTriangle {
                value: MaxwellThreeDFillViaTriangleMode::parse(raw).ok_or_else(|| {
                    invalid_encoding(source, "SET_FILL_VIA_TRIANGLE", "expected mode 0, 1, or 2")
                })?,
                source,
            },
            "SET_FILL_VIA_TRIANGLE",
        )),
        0x1148 => Some((
            MaxwellThreeDStateWrite::ConservativeRaster {
                value: MaxwellThreeDConservativeRasterEnable::parse(raw).ok_or_else(|| {
                    invalid_encoding(source, "SET_CONSERVATIVE_RASTER", "expected boolean 0 or 1")
                })?,
                source,
            },
            "SET_CONSERVATIVE_RASTER",
        )),
        // NVIDIA publishes polygon smoothing, stipple enable, and all 32
        // pattern words in its pinned public MAXWELL_B class header.
        // https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L1011-L1014
        // https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L3133-L3136
        // https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L3169-L3170
        0x0db4 => Some((
            MaxwellThreeDStateWrite::PolygonSmoothEnable {
                value: checked_bool(source, "SET_POLY_SMOOTH")?,
                source,
            },
            "SET_POLY_SMOOTH",
        )),
        0x168c => Some((
            MaxwellThreeDStateWrite::PolygonStippleEnable {
                value: checked_bool(source, "SET_POLYGON_STIPPLE")?,
                source,
            },
            "SET_POLYGON_STIPPLE",
        )),
        0x1700..=0x177c if method & 3 == 0 => Some((
            MaxwellThreeDStateWrite::PolygonStipplePattern {
                word: ((method - 0x1700) / 4) as u8,
                value: raw,
                source,
            },
            "SET_POLYGON_STIPPLE_PATTERN",
        )),
        // NVIDIA publishes this pixel-center selector in the pinned public
        // MAXWELL_B class header.
        // https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L3362-L3365
        0x1924 => Some((
            MaxwellThreeDStateWrite::ViewportPixelCenter {
                value: MaxwellThreeDViewportPixelCenter::parse(raw).ok_or_else(|| {
                    invalid_encoding(source, "SET_VIEWPORT_PIXEL", "expected center mode 0 or 1")
                })?,
                source,
            },
            "SET_VIEWPORT_PIXEL",
        )),
        0x1520 => Some((
            MaxwellThreeDStateWrite::PointSpriteEnable {
                value: checked_bool(source, "SET_POINT_SPRITE")?,
                source,
            },
            "SET_POINT_SPRITE",
        )),
        0x1658 => Some((
            MaxwellThreeDStateWrite::AntiAliasedPointEnable {
                value: checked_bool(source, "SET_ANTI_ALIASED_POINT")?,
                source,
            },
            "SET_ANTI_ALIASED_POINT",
        )),
        0x1910 => Some((
            MaxwellThreeDStateWrite::AttributePointSize {
                value: MaxwellThreeDAttributePointSize::parse(raw).ok_or_else(|| {
                    invalid_encoding(source, "SET_ATTRIBUTE_POINT_SIZE", "reserved bits are set")
                })?,
                source,
            },
            "SET_ATTRIBUTE_POINT_SIZE",
        )),
        _ => None,
    };
    if let Some((write, name)) = point_write {
        return Ok(Some((write, name)));
    }

    if (0x0800..0x0a00).contains(&method) {
        let target = ((method - 0x0800) / 0x40) as u8;
        let offset = (method - 0x0800) % 0x40;
        if target as usize >= MAXWELL_COLOR_TARGET_COUNT {
            return Ok(None);
        }
        let (write, name) = match offset {
            0x00 if raw <= 0xff => (
                MaxwellThreeDRenderTargetWrite::ColorAddressUpper {
                    target,
                    value: raw as u8,
                    source,
                },
                "SET_COLOR_TARGET_A",
            ),
            0x04 => (
                MaxwellThreeDRenderTargetWrite::ColorAddressLower {
                    target,
                    value: raw,
                    source,
                },
                "SET_COLOR_TARGET_B",
            ),
            0x08 if raw <= 0x0fff_ffff => (
                MaxwellThreeDRenderTargetWrite::ColorWidth {
                    target,
                    value: raw,
                    source,
                },
                "SET_COLOR_TARGET_WIDTH",
            ),
            0x0c if raw <= 0x1ffff => (
                MaxwellThreeDRenderTargetWrite::ColorHeight {
                    target,
                    value: raw,
                    source,
                },
                "SET_COLOR_TARGET_HEIGHT",
            ),
            0x10 => {
                let value = MaxwellThreeDColorTargetFormat::parse(raw).ok_or_else(|| {
                    invalid_encoding(
                        source,
                        "SET_COLOR_TARGET_FORMAT",
                        "unknown public color format",
                    )
                })?;
                (
                    MaxwellThreeDRenderTargetWrite::ColorFormat {
                        target,
                        value,
                        source,
                    },
                    "SET_COLOR_TARGET_FORMAT",
                )
            }
            0x14 if raw & !0x0001_1fff == 0 => {
                let block_width = raw & 0xf;
                let block_height = ((raw >> 4) & 0xf) as u8;
                let block_depth = ((raw >> 8) & 0xf) as u8;
                if block_width != 0 || block_height > 5 || block_depth > 5 {
                    return Err(invalid_encoding(
                        source,
                        "SET_COLOR_TARGET_MEMORY",
                        "invalid public GOB block size",
                    ));
                }
                let layout = if raw & 0x1000 != 0 {
                    if block_height != 0 || block_depth != 0 {
                        return Err(invalid_encoding(
                            source,
                            "SET_COLOR_TARGET_MEMORY",
                            "pitch layout contradicts non-unit GOB dimensions",
                        ));
                    }
                    MaxwellThreeDImageLayout::PitchLinear
                } else {
                    MaxwellThreeDImageLayout::BlockLinear {
                        block_height_log2: block_height,
                        block_depth_log2: block_depth,
                    }
                };
                let kind = if raw & 0x1_0000 != 0 {
                    MaxwellThreeDImageKind::ThreeDimensional
                } else {
                    MaxwellThreeDImageKind::Array
                };
                (
                    MaxwellThreeDRenderTargetWrite::ColorLayout {
                        target,
                        layout,
                        kind,
                        source,
                    },
                    "SET_COLOR_TARGET_MEMORY",
                )
            }
            0x18 if raw <= 0x0fff_ffff => (
                MaxwellThreeDRenderTargetWrite::ColorThirdDimension {
                    target,
                    value: raw,
                    source,
                },
                "SET_COLOR_TARGET_THIRD_DIMENSION",
            ),
            0x1c => (
                MaxwellThreeDRenderTargetWrite::ColorArrayPitch {
                    target,
                    value: raw,
                    source,
                },
                "SET_COLOR_TARGET_ARRAY_PITCH",
            ),
            0x20 if raw <= 0xffff => (
                MaxwellThreeDRenderTargetWrite::ColorLayer {
                    target,
                    value: raw as u16,
                    source,
                },
                "SET_COLOR_TARGET_LAYER",
            ),
            0x00 | 0x08 | 0x0c | 0x14 | 0x18 | 0x20 => {
                return Err(invalid_encoding(
                    source,
                    "SET_COLOR_TARGET",
                    "reserved bits are set",
                ));
            }
            _ => return Ok(None),
        };
        return Ok(Some((MaxwellThreeDStateWrite::RenderTarget(write), name)));
    }

    let render_write = match method {
        0x0d6c | 0x0d70 => {
            let value = rectangle(raw).ok_or_else(|| {
                invalid_encoding(
                    source,
                    if method == 0x0d6c {
                        "SET_CLEAR_RECT_HORIZONTAL"
                    } else {
                        "SET_CLEAR_RECT_VERTICAL"
                    },
                    "rectangle minimum exceeds maximum",
                )
            })?;
            Some((
                if method == 0x0d6c {
                    MaxwellThreeDRenderTargetWrite::ClearHorizontal { value, source }
                } else {
                    MaxwellThreeDRenderTargetWrite::ClearVertical { value, source }
                },
                if method == 0x0d6c {
                    "SET_CLEAR_RECT_HORIZONTAL"
                } else {
                    "SET_CLEAR_RECT_VERTICAL"
                },
            ))
        }
        0x0d80..=0x0d8c if method & 3 == 0 => Some((
            MaxwellThreeDRenderTargetWrite::ClearColor {
                component: ((method - 0x0d80) / 4) as u8,
                value: MaxwellThreeDRawValue::new(raw),
                source,
            },
            "SET_COLOR_CLEAR_VALUE",
        )),
        0x0d90 => Some((
            MaxwellThreeDRenderTargetWrite::ClearDepth {
                value: MaxwellThreeDRawValue::new(raw),
                source,
            },
            "SET_Z_CLEAR_VALUE",
        )),
        0x0da0 if raw <= 0xff => Some((
            MaxwellThreeDRenderTargetWrite::ClearStencil {
                value: raw as u8,
                source,
            },
            "SET_STENCIL_CLEAR_VALUE",
        )),
        0x0da0 => {
            return Err(invalid_encoding(
                source,
                "SET_STENCIL_CLEAR_VALUE",
                "reserved bits are set",
            ));
        }
        0x0fac => {
            let value = MaxwellThreeDSeparateFragmentData::parse(raw).ok_or_else(|| {
                invalid_encoding(
                    source,
                    "SET_CT_MRT_ENABLE",
                    "undefined boolean encoding or reserved bits",
                )
            })?;
            Some((
                MaxwellThreeDRenderTargetWrite::SeparateFragmentData { value, source },
                "SET_CT_MRT_ENABLE",
            ))
        }
        0x11f0 => {
            let value = MaxwellThreeDRenderTargetIndexOffset::parse(raw).ok_or_else(|| {
                invalid_encoding(
                    source,
                    "SET_OFFSET_RENDER_TARGET_INDEX",
                    "undefined boolean encoding or reserved bits",
                )
            })?;
            Some((
                MaxwellThreeDRenderTargetWrite::RenderTargetIndexOffset { value, source },
                "SET_OFFSET_RENDER_TARGET_INDEX",
            ))
        }
        0x0fe0 if raw <= 0xff => Some((
            MaxwellThreeDRenderTargetWrite::DepthAddressUpper {
                value: raw as u8,
                source,
            },
            "SET_ZT_A",
        )),
        0x0fe0 => {
            return Err(invalid_encoding(
                source,
                "SET_ZT_A",
                "reserved address bits are set",
            ));
        }
        0x0fe4 => Some((
            MaxwellThreeDRenderTargetWrite::DepthAddressLower { value: raw, source },
            "SET_ZT_B",
        )),
        0x0fe8 => {
            let value = MaxwellThreeDDepthStencilFormat::parse(raw).ok_or_else(|| {
                invalid_encoding(
                    source,
                    "SET_ZT_FORMAT",
                    "unknown public depth/stencil format",
                )
            })?;
            Some((
                MaxwellThreeDRenderTargetWrite::DepthFormat { value, source },
                "SET_ZT_FORMAT",
            ))
        }
        0x0fec
            if raw & !0x0fff == 0
                && raw & 0xf == 0
                && ((raw >> 4) & 0xf) <= 5
                && ((raw >> 8) & 0xf) == 0 =>
        {
            Some((
                MaxwellThreeDRenderTargetWrite::DepthLayout {
                    value: MaxwellThreeDImageLayout::BlockLinear {
                        block_height_log2: ((raw >> 4) & 0xf) as u8,
                        block_depth_log2: 0,
                    },
                    source,
                },
                "SET_ZT_BLOCK_SIZE",
            ))
        }
        0x0fec => {
            return Err(invalid_encoding(
                source,
                "SET_ZT_BLOCK_SIZE",
                "invalid public GOB block size",
            ));
        }
        0x0ff0 => Some((
            MaxwellThreeDRenderTargetWrite::DepthArrayPitch { value: raw, source },
            "SET_ZT_ARRAY_PITCH",
        )),
        // NVIDIA's public MAXWELL_B class header defines SET_ZT_LAYER as one
        // 16-bit array-layer offset and leaves the upper half reserved.
        // https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L3180-L3181
        0x179c if raw <= u16::MAX as u32 => Some((
            MaxwellThreeDRenderTargetWrite::DepthLayer {
                value: raw as u16,
                source,
            },
            "SET_ZT_LAYER",
        )),
        0x179c => {
            return Err(invalid_encoding(
                source,
                "SET_ZT_LAYER",
                "reserved bits are set",
            ));
        }
        0x1228 if raw <= 0x0fff_ffff => Some((
            MaxwellThreeDRenderTargetWrite::DepthWidth { value: raw, source },
            "SET_ZT_SIZE_A",
        )),
        0x121c => {
            let value = MaxwellThreeDColorTargetSelection::parse(raw).ok_or_else(|| {
                invalid_encoding(
                    source,
                    "SET_CT_SELECT",
                    "reserved bits are set or target count exceeds exposed selectors",
                )
            })?;
            Some((
                MaxwellThreeDRenderTargetWrite::ColorTargetSelection { value, source },
                "SET_CT_SELECT",
            ))
        }
        0x1220 => {
            let value = MaxwellThreeDCompressionThreshold::parse(raw).ok_or_else(|| {
                invalid_encoding(
                    source,
                    "SET_COMPRESSION_THRESHOLD",
                    "sample threshold is undefined or reserved bits are set",
                )
            })?;
            Some((
                MaxwellThreeDRenderTargetWrite::CompressionThreshold { value, source },
                "SET_COMPRESSION_THRESHOLD",
            ))
        }
        0x15cc if raw & !0x0001_ffff == 0 => Some((
            MaxwellThreeDRenderTargetWrite::RenderTargetLayer {
                value: MaxwellThreeDRenderTargetLayer::parse(raw),
                source,
            },
            "SET_RT_LAYER",
        )),
        0x15cc => {
            return Err(invalid_encoding(
                source,
                "SET_RT_LAYER",
                "reserved bits are set",
            ));
        }
        0x1538 => {
            let value = MaxwellThreeDDepthTargetCount::parse(raw).ok_or_else(|| {
                invalid_encoding(
                    source,
                    "SET_ZT_SELECT",
                    "target count exceeds the single exposed depth/stencil target",
                )
            })?;
            Some((
                MaxwellThreeDRenderTargetWrite::DepthTargetCount { value, source },
                "SET_ZT_SELECT",
            ))
        }
        0x122c if raw <= 0x1ffff => Some((
            MaxwellThreeDRenderTargetWrite::DepthHeight { value: raw, source },
            "SET_ZT_SIZE_B",
        )),
        0x19cc => {
            let value = MaxwellThreeDZCompressionMode::parse(raw).ok_or_else(|| {
                invalid_encoding(source, "SET_Z_COMPRESSION", "reserved bits are set")
            })?;
            Some((
                MaxwellThreeDRenderTargetWrite::DepthCompression { value, source },
                "SET_Z_COMPRESSION",
            ))
        }
        0x19e0..=0x19fc if method & 3 == 0 => {
            let target = ((method - 0x19e0) / 4) as u8;
            let value = MaxwellThreeDColorCompressionMode::parse(raw).ok_or_else(|| {
                invalid_encoding(source, "SET_COLOR_COMPRESSION", "reserved bits are set")
            })?;
            Some((
                MaxwellThreeDRenderTargetWrite::ColorCompression {
                    target,
                    value,
                    source,
                },
                "SET_COLOR_COMPRESSION",
            ))
        }
        0x1230 if raw & !0x1ffff == 0 => {
            let kind = if raw & 0x1_0000 != 0 {
                MaxwellThreeDImageKind::Array
            } else {
                MaxwellThreeDImageKind::ThreeDimensional
            };
            Some((
                MaxwellThreeDRenderTargetWrite::DepthThirdDimension {
                    value: raw as u16,
                    kind,
                    source,
                },
                "SET_ZT_SIZE_C",
            ))
        }
        0x1228 | 0x122c | 0x1230 => {
            return Err(invalid_encoding(
                source,
                "SET_ZT_SIZE",
                "reserved dimension bits are set",
            ));
        }
        0x10f8 => {
            let value = MaxwellThreeDClearSurfaceControl::parse(raw).ok_or_else(|| {
                invalid_encoding(
                    source,
                    "SET_CLEAR_SURFACE_CONTROL",
                    "reserved control bits are set",
                )
            })?;
            Some((
                MaxwellThreeDRenderTargetWrite::ClearSurfaceControl { value, source },
                "SET_CLEAR_SURFACE_CONTROL",
            ))
        }
        0x19d0 => {
            let value = MaxwellThreeDClearSurface::parse(raw).ok_or_else(|| {
                invalid_encoding(
                    source,
                    "CLEAR_SURFACE",
                    "invalid target, layer, or reserved bits",
                )
            })?;
            Some((
                MaxwellThreeDRenderTargetWrite::ClearSurface { value, source },
                "CLEAR_SURFACE",
            ))
        }
        _ => None,
    };
    if let Some((write, name)) = render_write {
        return Ok(Some((MaxwellThreeDStateWrite::RenderTarget(write), name)));
    }

    if matches!(method, 0x0ff4 | 0x0ff8) {
        let vertical = method == 0x0ff8;
        let write = MaxwellThreeDFixedFunctionWrite::SurfaceClip {
            vertical,
            value: MaxwellThreeDSurfaceClipAxis::parse(raw),
            source,
        };
        return Ok(Some((
            MaxwellThreeDStateWrite::FixedFunction(write),
            if vertical {
                "SET_SURFACE_CLIP_VERTICAL"
            } else {
                "SET_SURFACE_CLIP_HORIZONTAL"
            },
        )));
    }

    if (0x0a00..0x0c00).contains(&method) {
        let viewport = ((method - 0x0a00) / 0x20) as u8;
        let field = ((method - 0x0a00) % 0x20) / 4;
        if field <= 5 {
            let write = MaxwellThreeDFixedFunctionWrite::ViewportFloat {
                viewport,
                field: field as u8,
                value: MaxwellThreeDRawValue::new(raw),
                source,
            };
            return Ok(Some((
                MaxwellThreeDStateWrite::FixedFunction(write),
                "SET_VIEWPORT_SCALE_OR_OFFSET",
            )));
        }
        if field == 6 {
            let value = MaxwellThreeDViewportCoordinateSwizzle::parse(raw).ok_or_else(|| {
                invalid_encoding(
                    source,
                    "SET_VIEWPORT_COORDINATE_SWIZZLE",
                    "reserved bits are set",
                )
            })?;
            return Ok(Some((
                MaxwellThreeDStateWrite::FixedFunction(
                    MaxwellThreeDFixedFunctionWrite::ViewportCoordinateSwizzle {
                        viewport,
                        value,
                        source,
                    },
                ),
                "SET_VIEWPORT_COORDINATE_SWIZZLE",
            )));
        }
    }
    if (0x0c00..0x0d00).contains(&method) {
        let viewport = ((method - 0x0c00) / 0x10) as u8;
        let field = (method - 0x0c00) % 0x10;
        let write = match field {
            0 | 4 => {
                let value = rectangle(raw).ok_or_else(|| {
                    invalid_encoding(
                        source,
                        "SET_VIEWPORT_CLIP",
                        "rectangle minimum exceeds maximum",
                    )
                })?;
                MaxwellThreeDFixedFunctionWrite::ViewportRectangle {
                    viewport,
                    vertical: field == 4,
                    value,
                    source,
                }
            }
            8 | 12 => MaxwellThreeDFixedFunctionWrite::ViewportDepth {
                viewport,
                maximum: field == 12,
                value: MaxwellThreeDRawValue::new(raw),
                source,
            },
            _ => return Ok(None),
        };
        return Ok(Some((
            MaxwellThreeDStateWrite::FixedFunction(write),
            "SET_VIEWPORT_CLIP",
        )));
    }
    if (0x0d00..0x0d40).contains(&method) {
        let region = ((method - 0x0d00) / 8) as u8;
        let vertical = (method - 0x0d00) % 8 == 4;
        let method_name = if vertical {
            "SET_WINDOW_CLIP_VERTICAL"
        } else {
            "SET_WINDOW_CLIP_HORIZONTAL"
        };
        let value = rectangle(raw).ok_or_else(|| {
            invalid_encoding(source, method_name, "rectangle minimum exceeds maximum")
        })?;
        let write = MaxwellThreeDFixedFunctionWrite::WindowClipRectangle {
            region,
            vertical,
            value,
            source,
        };
        return Ok(Some((
            MaxwellThreeDStateWrite::FixedFunction(write),
            method_name,
        )));
    }
    if (0x0e00..0x0f00).contains(&method) {
        let scissor = ((method - 0x0e00) / 0x10) as u8;
        let field = (method - 0x0e00) % 0x10;
        let write = match field {
            0 => MaxwellThreeDFixedFunctionWrite::ScissorEnable {
                scissor,
                value: checked_bool(source, "SET_SCISSOR_ENABLE")?,
                source,
            },
            4 | 8 => {
                let value = rectangle(raw).ok_or_else(|| {
                    invalid_encoding(source, "SET_SCISSOR", "rectangle minimum exceeds maximum")
                })?;
                MaxwellThreeDFixedFunctionWrite::ScissorRectangle {
                    scissor,
                    vertical: field == 8,
                    value,
                    source,
                }
            }
            _ => return Ok(None),
        };
        return Ok(Some((
            MaxwellThreeDStateWrite::FixedFunction(write),
            "SET_SCISSOR",
        )));
    }
    if (0x1360..0x1380).contains(&method) && method & 3 == 0 {
        let target = ((method - 0x1360) / 4) as u8;
        let value = checked_bool(source, "SET_BLEND")?;
        return Ok(Some((
            MaxwellThreeDStateWrite::FixedFunction(MaxwellThreeDFixedFunctionWrite::BlendEnable {
                target,
                value,
                source,
            }),
            "SET_BLEND",
        )));
    }
    if (0x1a00..0x1a20).contains(&method) && method & 3 == 0 {
        let target = ((method - 0x1a00) / 4) as u8;
        let value = MaxwellThreeDColorMask::parse(raw).ok_or_else(|| {
            invalid_encoding(source, "SET_CT_WRITE", "reserved color-mask bits are set")
        })?;
        return Ok(Some((
            MaxwellThreeDStateWrite::FixedFunction(MaxwellThreeDFixedFunctionWrite::ColorMask {
                target,
                value,
                source,
            }),
            "SET_CT_WRITE",
        )));
    }
    if (0x1e00..0x1f00).contains(&method) {
        let target = ((method - 0x1e00) / 0x20) as u8;
        let field = ((method - 0x1e00) % 0x20) / 4;
        let value = match field {
            0 => MaxwellThreeDFixedFunctionValue::Boolean(checked_bool(
                source,
                "SET_BLEND_PER_TARGET_SEPARATE_FOR_ALPHA",
            )?),
            1 | 4 => MaxwellThreeDFixedFunctionValue::BlendOp(
                MaxwellThreeDBlendOp::parse(raw).ok_or_else(|| {
                    invalid_encoding(source, "SET_BLEND_PER_TARGET_OP", "unknown blend operation")
                })?,
            ),
            2 | 3 | 5 | 6 => MaxwellThreeDFixedFunctionValue::BlendFactor(
                MaxwellThreeDBlendFactor::parse(raw).ok_or_else(|| {
                    invalid_encoding(
                        source,
                        "SET_BLEND_PER_TARGET_COEFF",
                        "unknown blend coefficient",
                    )
                })?,
            ),
            _ => return Ok(None),
        };
        let write = MaxwellThreeDFixedFunctionWrite::BlendState {
            target,
            field: field as u8,
            value,
            source,
        };
        return Ok(Some((
            MaxwellThreeDStateWrite::FixedFunction(write),
            "SET_BLEND_PER_TARGET_STATE",
        )));
    }

    let blend_control = match method {
        0x0fdc => Some((
            MaxwellThreeDFixedFunctionWrite::BlendFloatPixelKillEnable {
                value: MaxwellThreeDBlendFloatPixelKillEnable::parse(raw).ok_or_else(|| {
                    invalid_encoding(
                        source,
                        "SET_BLEND_OPT_CONTROL",
                        "undefined boolean encoding or reserved bits",
                    )
                })?,
                source,
            },
            "SET_BLEND_OPT_CONTROL",
        )),
        0x1140 => Some((
            MaxwellThreeDFixedFunctionWrite::BlendPerFormatEnable {
                value: MaxwellThreeDBlendPerFormatEnable::parse(raw).ok_or_else(|| {
                    invalid_encoding(
                        source,
                        "SET_BLEND_PER_FORMAT_ENABLE",
                        "undefined boolean encoding or reserved bits",
                    )
                })?,
                source,
            },
            "SET_BLEND_PER_FORMAT_ENABLE",
        )),
        0x19c0 => Some((
            MaxwellThreeDFixedFunctionWrite::BlendZeroTimesAnythingIsZero {
                value: MaxwellThreeDBlendZeroTimesAnythingIsZero::parse(raw).ok_or_else(|| {
                    invalid_encoding(
                        source,
                        "SET_BLEND_FLOAT_OPTION",
                        "undefined boolean encoding or reserved bits",
                    )
                })?,
                source,
            },
            "SET_BLEND_FLOAT_OPTION",
        )),
        _ => None,
    };
    if let Some((write, name)) = blend_control {
        return Ok(Some((MaxwellThreeDStateWrite::FixedFunction(write), name)));
    }

    use MaxwellThreeDFixedFunctionRegister as R;
    use MaxwellThreeDFixedFunctionValue as V;
    let fixed = match method {
        0x135c => {
            let value = MaxwellThreeDBlendEnableCommon::parse(raw).ok_or_else(|| {
                invalid_encoding(
                    source,
                    "SET_BLEND_ENABLE_COMMON",
                    "undefined boolean encoding or reserved bits",
                )
            })?;
            return Ok(Some((
                MaxwellThreeDStateWrite::FixedFunction(
                    MaxwellThreeDFixedFunctionWrite::BlendEnableCommon { value, source },
                ),
                "SET_BLEND_ENABLE_COMMON",
            )));
        }
        0x037c => (
            R::RasterEnable,
            V::Boolean(checked_bool(source, "SET_RASTER_ENABLE")?),
            "SET_RASTER_ENABLE",
        ),
        0x0dac | 0x0db0 => {
            let value = MaxwellThreeDPolygonMode::parse(raw).ok_or_else(|| {
                invalid_encoding(
                    source,
                    if method == 0x0dac {
                        "SET_FRONT_POLYGON_MODE"
                    } else {
                        "SET_BACK_POLYGON_MODE"
                    },
                    "unknown polygon mode",
                )
            })?;
            (
                if method == 0x0dac {
                    R::FrontPolygonMode
                } else {
                    R::BackPolygonMode
                },
                V::PolygonMode(value),
                if method == 0x0dac {
                    "SET_FRONT_POLYGON_MODE"
                } else {
                    "SET_BACK_POLYGON_MODE"
                },
            )
        }
        0x0dc0 | 0x0dc4 | 0x0dc8 => (
            match method {
                0x0dc0 => R::PolygonOffsetPointEnable,
                0x0dc4 => R::PolygonOffsetLineEnable,
                _ => R::PolygonOffsetFillEnable,
            },
            V::Boolean(checked_bool(source, "SET_POLY_OFFSET")?),
            "SET_POLY_OFFSET",
        ),
        0x0df8 if raw <= 0x1ffff => (R::WindowOffsetX, V::Mask(raw), "SET_WINDOW_OFFSET_X"),
        0x0dfc if raw <= 0x3ffff => (R::WindowOffsetY, V::Mask(raw), "SET_WINDOW_OFFSET_Y"),
        0x0df8 | 0x0dfc => {
            return Err(invalid_encoding(
                source,
                "SET_WINDOW_OFFSET",
                "reserved bits are set",
            ));
        }
        0x0f54 | 0x0f58 | 0x0f5c if raw <= 0xff => (
            match method {
                0x0f54 => R::BackStencilReference,
                0x0f58 => R::BackStencilWriteMask,
                _ => R::BackStencilCompareMask,
            },
            V::Mask(raw),
            "SET_BACK_STENCIL_MASK_OR_REFERENCE",
        ),
        0x0f54 | 0x0f58 | 0x0f5c => {
            return Err(invalid_encoding(
                source,
                "SET_BACK_STENCIL_MASK_OR_REFERENCE",
                "reserved bits are set",
            ));
        }
        0x0f9c => (
            R::DepthBoundsMin,
            V::FloatBits(MaxwellThreeDRawValue::new(raw)),
            "SET_DEPTH_BOUNDS_MIN",
        ),
        0x0fa0 => (
            R::DepthBoundsMax,
            V::FloatBits(MaxwellThreeDRawValue::new(raw)),
            "SET_DEPTH_BOUNDS_MAX",
        ),
        0x0fa4 if raw & !0x11 == 0 => (R::SampleMaskControl, V::Mask(raw), "SET_SAMPLE_MASK"),
        0x0fa4 => {
            return Err(invalid_encoding(
                source,
                "SET_SAMPLE_MASK",
                "reserved bits are set",
            ));
        }
        0x0fb8 => {
            let value = MaxwellThreeDSampleMode::parse(raw)
                .filter(|value| {
                    matches!(
                        value,
                        MaxwellThreeDSampleMode::Samples1x1
                            | MaxwellThreeDSampleMode::Samples2x2
                            | MaxwellThreeDSampleMode::Samples4x2D3D
                            | MaxwellThreeDSampleMode::Samples2x1D3D
                            | MaxwellThreeDSampleMode::Samples4x4
                    )
                })
                .ok_or_else(|| {
                    invalid_encoding(
                        source,
                        "SET_ANTI_ALIAS_RASTER",
                        "unsupported raster sample encoding",
                    )
                })?;
            (
                R::RasterSampleMode,
                V::SampleMode(value),
                "SET_ANTI_ALIAS_RASTER",
            )
        }
        0x0fbc..=0x0fc8 if method & 3 == 0 && raw <= 0xffff => {
            let slot = ((method - 0x0fbc) / 4) as usize;
            (
                [
                    R::SampleMask0,
                    R::SampleMask1,
                    R::SampleMask2,
                    R::SampleMask3,
                ][slot],
                V::Mask(raw),
                "SET_SAMPLE_MASK_QUADRANT",
            )
        }
        0x0f90 => (
            R::SingleColorTargetWriteControl,
            V::Boolean(checked_bool(source, "SET_SINGLE_CT_WRITE_CONTROL")?),
            "SET_SINGLE_CT_WRITE_CONTROL",
        ),
        0x12cc => (
            R::DepthTestEnable,
            V::Boolean(checked_bool(source, "SET_DEPTH_TEST")?),
            "SET_DEPTH_TEST",
        ),
        0x12d4 => {
            let value = MaxwellThreeDShadeMode::parse(raw)
                .ok_or_else(|| invalid_encoding(source, "SET_SHADE_MODE", "unknown shade mode"))?;
            (R::ShadeMode, V::ShadeMode(value), "SET_SHADE_MODE")
        }
        0x12e4 => (
            R::BlendPerTargetEnable,
            V::Boolean(checked_bool(source, "SET_BLEND_STATE_PER_TARGET")?),
            "SET_BLEND_STATE_PER_TARGET",
        ),
        0x12e8 => (
            R::DepthWriteEnable,
            V::Boolean(checked_bool(source, "SET_DEPTH_WRITE")?),
            "SET_DEPTH_WRITE",
        ),
        // NVIDIA's pinned MAXWELL_B header publishes the enable, IEEE-754
        // reference bits, and all OGL/D3D comparison encodings as one family.
        // https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L2155-L2158
        // https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L2195-L2215
        0x12ec => (
            R::AlphaTestEnable,
            V::Boolean(checked_bool(source, "SET_ALPHA_TEST")?),
            "SET_ALPHA_TEST",
        ),
        0x130c => {
            let value = MaxwellThreeDCompareOp::parse(raw).ok_or_else(|| {
                invalid_encoding(source, "SET_DEPTH_FUNC", "unknown compare operation")
            })?;
            (R::DepthCompare, V::Compare(value), "SET_DEPTH_FUNC")
        }
        0x1310 => (
            R::AlphaTestReference,
            V::FloatBits(MaxwellThreeDRawValue::new(raw)),
            "SET_ALPHA_REF",
        ),
        0x1314 => {
            let value = MaxwellThreeDCompareOp::parse(raw).ok_or_else(|| {
                invalid_encoding(source, "SET_ALPHA_FUNC", "unknown compare operation")
            })?;
            (R::AlphaTestFunction, V::Compare(value), "SET_ALPHA_FUNC")
        }
        0x131c..=0x1328 if method & 3 == 0 => (
            [
                R::BlendConstantRed,
                R::BlendConstantGreen,
                R::BlendConstantBlue,
                R::BlendConstantAlpha,
            ][((method - 0x131c) / 4) as usize],
            V::FloatBits(MaxwellThreeDRawValue::new(raw)),
            "SET_BLEND_CONST",
        ),
        0x133c => (
            R::BlendSeparateAlpha,
            V::Boolean(checked_bool(source, "SET_BLEND_SEPARATE_FOR_ALPHA")?),
            "SET_BLEND_SEPARATE_FOR_ALPHA",
        ),
        0x1340 | 0x134c => {
            let value = MaxwellThreeDBlendOp::parse(raw).ok_or_else(|| {
                invalid_encoding(source, "SET_BLEND_OP", "unknown blend operation")
            })?;
            (
                if method == 0x1340 {
                    R::BlendColorOp
                } else {
                    R::BlendAlphaOp
                },
                V::BlendOp(value),
                "SET_BLEND_OP",
            )
        }
        0x1344 | 0x1348 | 0x1350 | 0x1358 => {
            let value = MaxwellThreeDBlendFactor::parse(raw).ok_or_else(|| {
                invalid_encoding(source, "SET_BLEND_COEFF", "unknown blend coefficient")
            })?;
            let register = match method {
                0x1344 => R::BlendColorSource,
                0x1348 => R::BlendColorDestination,
                0x1350 => R::BlendAlphaSource,
                _ => R::BlendAlphaDestination,
            };
            (register, V::BlendFactor(value), "SET_BLEND_COEFF")
        }
        0x1380 => (
            R::StencilTestEnable,
            V::Boolean(checked_bool(source, "SET_STENCIL_TEST")?),
            "SET_STENCIL_TEST",
        ),
        0x1594 => (
            R::TwoSidedStencilTestEnable,
            V::Boolean(checked_bool(source, "SET_TWO_SIDED_STENCIL_TEST")?),
            "SET_TWO_SIDED_STENCIL_TEST",
        ),
        0x1384 | 0x1388 | 0x138c | 0x1598 | 0x159c | 0x15a0 => {
            let value = MaxwellThreeDStencilOp::parse(raw).ok_or_else(|| {
                invalid_encoding(source, "SET_STENCIL_OP", "unknown stencil operation")
            })?;
            let register = match method {
                0x1384 => R::FrontStencilFail,
                0x1388 => R::FrontStencilDepthFail,
                0x138c => R::FrontStencilPass,
                0x1598 => R::BackStencilFail,
                0x159c => R::BackStencilDepthFail,
                _ => R::BackStencilPass,
            };
            (register, V::StencilOp(value), "SET_STENCIL_OP")
        }
        0x1390 | 0x15a4 => {
            let value = MaxwellThreeDCompareOp::parse(raw).ok_or_else(|| {
                invalid_encoding(source, "SET_STENCIL_FUNC", "unknown compare operation")
            })?;
            (
                if method == 0x1390 {
                    R::FrontStencilCompare
                } else {
                    R::BackStencilCompare
                },
                V::Compare(value),
                "SET_STENCIL_FUNC",
            )
        }
        0x1394 | 0x1398 | 0x139c if raw <= 0xff => (
            match method {
                0x1394 => R::FrontStencilReference,
                0x1398 => R::FrontStencilCompareMask,
                _ => R::FrontStencilWriteMask,
            },
            V::Mask(raw),
            "SET_STENCIL_MASK_OR_REFERENCE",
        ),
        0x1394 | 0x1398 | 0x139c => {
            return Err(invalid_encoding(
                source,
                "SET_STENCIL_MASK_OR_REFERENCE",
                "reserved bits are set",
            ));
        }
        0x13a8 => (
            R::PixelShaderSaturate,
            V::PixelShaderSaturate(MaxwellThreeDPixelShaderSaturate::parse(raw).ok_or_else(
                || invalid_encoding(source, "SET_PS_SATURATE", "reserved output bits are set"),
            )?),
            "SET_PS_SATURATE",
        ),
        0x13ac if raw & !0x11 == 0 => (R::WindowOrigin, V::Mask(raw), "SET_WINDOW_ORIGIN"),
        0x13ac => {
            return Err(invalid_encoding(
                source,
                "SET_WINDOW_ORIGIN",
                "reserved bits are set",
            ));
        }
        0x13b0 => (
            R::LineWidth,
            V::FloatBits(MaxwellThreeDRawValue::new(raw)),
            "SET_LINE_WIDTH_FLOAT",
        ),
        0x1510 if raw <= 0xff => (R::UserClipEnable, V::Mask(raw), "SET_USER_CLIP_ENABLE"),
        0x1510 => {
            return Err(invalid_encoding(
                source,
                "SET_USER_CLIP_ENABLE",
                "reserved clip-plane bits are set",
            ));
        }
        0x1534 => (
            R::AntiAliasEnable,
            V::Boolean(checked_bool(source, "SET_ANTI_ALIAS_ENABLE")?),
            "SET_ANTI_ALIAS_ENABLE",
        ),
        0x153c if raw & !0x11 == 0 => (
            R::AlphaToCoverageEnable,
            V::AlphaControl {
                alpha_to_coverage: raw & 1 != 0,
                alpha_to_one: raw & 0x10 != 0,
            },
            "SET_ANTI_ALIAS_ALPHA_CONTROL",
        ),
        0x153c => {
            return Err(invalid_encoding(
                source,
                "SET_ANTI_ALIAS_ALPHA_CONTROL",
                "reserved bits are set",
            ));
        }
        0x15d0 => {
            let value = MaxwellThreeDSampleMode::parse(raw).ok_or_else(|| {
                invalid_encoding(source, "SET_ANTI_ALIAS", "unknown sample encoding")
            })?;
            (R::SampleMode, V::SampleMode(value), "SET_ANTI_ALIAS")
        }
        0x1684 => {
            let value = MaxwellThreeDProvokingVertex::parse(raw).ok_or_else(|| {
                invalid_encoding(source, "SET_PROVOKING_VERTEX", "expected FIRST or LAST")
            })?;
            (
                R::ProvokingVertex,
                V::ProvokingVertex(value),
                "SET_PROVOKING_VERTEX",
            )
        }
        // The adjacent fixed-function lighting selector is also a one-bit
        // field in NVIDIA's pinned MAXWELL_B header.
        // https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L3128-L3131
        0x1688 => (
            R::TwoSidedLightEnable,
            V::Boolean(checked_bool(source, "SET_TWO_SIDED_LIGHT")?),
            "SET_TWO_SIDED_LIGHT",
        ),
        0x1918 => (
            R::CullEnable,
            V::Boolean(checked_bool(source, "OGL_SET_CULL")?),
            "OGL_SET_CULL",
        ),
        0x191c => {
            let value = MaxwellThreeDFrontFace::parse(raw)
                .ok_or_else(|| invalid_encoding(source, "OGL_SET_FRONT_FACE", "unknown winding"))?;
            (R::FrontFace, V::FrontFace(value), "OGL_SET_FRONT_FACE")
        }
        0x1920 => {
            let value = MaxwellThreeDCullFace::parse(raw).ok_or_else(|| {
                invalid_encoding(source, "OGL_SET_CULL_FACE", "unknown cull face")
            })?;
            (R::CullFace, V::CullFace(value), "OGL_SET_CULL_FACE")
        }
        0x192c => {
            let value = MaxwellThreeDViewportScaleOffsetEnable::parse(raw).ok_or_else(|| {
                invalid_encoding(
                    source,
                    "SET_VIEWPORT_SCALE_OFFSET",
                    "expected boolean 0 or 1",
                )
            })?;
            (
                R::ViewportScaleOffsetEnable,
                V::ViewportScaleOffsetEnable(value),
                "SET_VIEWPORT_SCALE_OFFSET",
            )
        }
        0x193c => {
            let value = MaxwellThreeDViewportClipControl::parse(raw).ok_or_else(|| {
                invalid_encoding(
                    source,
                    "SET_VIEWPORT_CLIP_CONTROL",
                    "invalid clip-control fields",
                )
            })?;
            (
                R::ViewportClipControl,
                V::ClipControl(value),
                "SET_VIEWPORT_CLIP_CONTROL",
            )
        }
        0x1940 if raw & !0x1111_1111 == 0 => {
            (R::UserClipOperation, V::Mask(raw), "SET_USER_CLIP_OP")
        }
        0x1940 => {
            return Err(invalid_encoding(
                source,
                "SET_USER_CLIP_OP",
                "reserved clip operation bits are set",
            ));
        }
        0x194c => (
            R::WindowClipEnable,
            V::Boolean(checked_bool(source, "SET_WINDOW_CLIP_ENABLE")?),
            "SET_WINDOW_CLIP_ENABLE",
        ),
        0x1950 => (
            R::WindowClipType,
            V::WindowClipType(MaxwellThreeDWindowClipType::parse(raw).ok_or_else(|| {
                invalid_encoding(source, "SET_WINDOW_CLIP_TYPE", "unknown window clip type")
            })?),
            "SET_WINDOW_CLIP_TYPE",
        ),
        0x197c => (
            R::ClipIdTestEnable,
            V::ClipIdTestEnable(MaxwellThreeDClipIdTestEnable::parse(raw).ok_or_else(|| {
                invalid_encoding(source, "SET_CLIP_ID_TEST", "expected boolean 0 or 1")
            })?),
            "SET_CLIP_ID_TEST",
        ),
        0x19bc => (
            R::DepthBoundsEnable,
            V::Boolean(checked_bool(source, "SET_DEPTH_BOUNDS_TEST")?),
            "SET_DEPTH_BOUNDS_TEST",
        ),
        0x19c4 => (
            R::LogicOpEnable,
            V::Boolean(checked_bool(source, "SET_LOGIC_OP")?),
            "SET_LOGIC_OP",
        ),
        0x19c8 => (
            R::LogicOpFunction,
            V::LogicOp(MaxwellThreeDLogicOp::parse(raw).ok_or_else(|| {
                invalid_encoding(source, "SET_LOGIC_OP_FUNC", "unknown logic operation")
            })?),
            "SET_LOGIC_OP_FUNC",
        ),
        // https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L4100-L4103
        0x2600 => (
            R::ColorClampEnable,
            V::Boolean(checked_bool(source, "SET_COLOR_CLAMP")?),
            "SET_COLOR_CLAMP",
        ),
        _ => return Ok(None),
    };
    let write = MaxwellThreeDFixedFunctionWrite::Register {
        register: fixed.0,
        value: fixed.1,
        source,
    };
    Ok(Some((
        MaxwellThreeDStateWrite::FixedFunction(write),
        fixed.2,
    )))
}
