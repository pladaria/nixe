//! Consumed Maxwell graphics state and native frontend preparation.
pub mod resolve;
use super::threed::*;
use nixe_gpu::*;
use std::fmt::{Display, Formatter};

#[derive(Clone, Debug)]
pub enum MaxwellLoweringError {
    RendererUnavailable,
    TriggerStateMismatch,
    ComputeLaunch(crate::MaxwellComputeLaunchError),
    ImageBacking(String),
    UnsupportedMultisampleState(&'static str),
    BlitSourceNotResident,
    BlitFormatMismatch,
    UnsupportedBlit(&'static str),
    UnsupportedPolygonRasterization(&'static str),
    UnsupportedWindowOrigin(u32),
    ContradictoryState {
        reason: &'static str,
    },
    UnsupportedConditionalLoadConstantBufferSemantics,
    UnsupportedStreamOutputSemantics,
    VisibleCallLimitExceeded {
        stage: ShaderStage,
        required: u16,
        limit: u16,
    },
    UnsupportedColorReductionSemantics,
    UnsupportedPostPsInitialCoverageSemantics,
    UnsupportedViewportPixelDepthBounds,
    UnsupportedWindowOriginFragmentPosition,
    UnsupportedPostZPixelShaderImaskSemantics,
    UnsupportedPixelShaderInterlockSemantics(MaxwellThreeDPixelShaderInterlockControl),
    UnsupportedCsaaSemantics,
    UnsupportedAntiAliasAlphaControl {
        alpha_to_coverage: bool,
        alpha_to_one: bool,
    },
    UnsupportedCoverageToColorSemantics(MaxwellThreeDCoverageToColor),
    UnsupportedAlphaToCoverageOverrideSemantics(MaxwellThreeDAlphaToCoverageOverride),
    UnsupportedTirSemantics {
        control: Option<MaxwellThreeDTirControl>,
    },
    UnsupportedHybridAntiAliasSemantics(MaxwellThreeDHybridAntiAliasControl),
    UnsupportedSampleLocationsSemantics {
        group: u8,
        value: MaxwellThreeDSampleLocationGroup,
    },
    UnsupportedPsOutputSampleMaskSemantics,
    UnsupportedReplicatedColorTargetOutputSemantics,
    UnsupportedRenderTargetIndexOffsetSemantics(MaxwellThreeDRenderTargetIndexOffset),
    UnsupportedRenderTargetLayerSemantics(MaxwellThreeDRenderTargetLayer),
    UnsupportedShaderLocalMemorySemantics {
        default_size_per_warp: MaxwellThreeDShaderLocalMemoryPerWarpSize,
    },
    UnsupportedViewportPixelCenterSemantics(MaxwellThreeDViewportPixelCenter),
    UnsupportedViewportSnapGridPrecision {
        viewport: u8,
        precision: [u8; 2],
    },
    UnsupportedViewportCoordinateSwizzleSemantics {
        viewport: u8,
        swizzle: MaxwellThreeDViewportCoordinateSwizzle,
    },
    UnsupportedWindowClipSemantics,
    UnsupportedClipIdTestSemantics,
    UnsupportedClearStencilMaskSemantics,
    UnsupportedAliasedLineWidthSemantics,
    UnsupportedLineStippleSemantics {
        factor: u8,
        pattern: u16,
    },
    UnsupportedPolygonClipGeneratedEdgeSemantics,
    UnsupportedSkippedVertexComponent {
        attribute: u8,
        component: u8,
    },
    InvalidPatchSize(MaxwellThreeDPatchSize),
    TessellationMode {
        value: super::threed::MaxwellThreeDTessellationMode,
        reason: super::threed::MaxwellTessellationModeError,
    },
    UnsupportedPointSpriteCoordinatesSemantics(MaxwellThreeDPointSpriteSelect),
    UnsupportedAttributePointSizeSemantics {
        slot: u8,
    },
    UnsupportedPointSpriteSemantics,
    UnsupportedAntiAliasedPointSemantics,
    UnsupportedPointCenterSemantics(MaxwellThreeDPointCenterMode),
    UnsupportedFillViaTriangleSemantics(MaxwellThreeDFillViaTriangleMode),
    UnsupportedFillRectangleDraw(&'static str),
    UnsupportedConservativeRasterSemantics,
    UnsupportedPolygonSmoothSemantics,
    UnsupportedPolygonStippleSemantics,
    UnsupportedEdgeFlagSemantics(MaxwellThreeDEdgeFlag),
    UnsupportedShadeModeSemantics(MaxwellThreeDShadeMode),
    UnsupportedProvokingVertexSemantics(MaxwellThreeDProvokingVertex),
    UnsupportedTwoSidedLightSemantics,
    UnsupportedPixelShaderSaturateSemantics {
        output: u8,
        range: MaxwellThreeDPixelShaderClampRange,
    },
    IncompleteAlphaTestState(&'static str),
    CompressedDepthImportRequired {
        kind: u8,
    },
    CompressedColorImportRequired {
        target: u8,
    },
    CompressedSampledImageImportRequired {
        role: MaxwellThreeDResourceRole,
        kind: u8,
    },
    ShaderTranslationRequired,
    InvalidTranslatedShaders,
    TranslatedShaderStageMismatch,
    TranslatedShaderMemoryConfigurationMismatch {
        stage: ShaderStage,
        configured: MaxwellThreeDDirectlyAddressableMemory,
        required: MaxwellThreeDDirectlyAddressableMemory,
    },
    UnsupportedShaderStage(MaxwellShaderStage),
    InvalidShaderResourceUse {
        role: MaxwellThreeDResourceRole,
    },
    InvalidResolvedView {
        role: MaxwellThreeDResourceRole,
    },
    AllocationDescriptionChanged {
        allocation: nixe_gpu::GpuAllocationId,
    },
    IncompleteClear(&'static str),
    EmptyClearRectangle,
    PartialColorClearUnsupported {
        mask: u8,
    },
    ClearOutsideAttachment,
    IncompleteDraw(&'static str),
    InvalidTransition,
    InvalidResourceCreation,
    Command(CommandDescriptionError),
    ResourceExhausted,
}

impl Display for MaxwellLoweringError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RendererUnavailable => formatter.write_str("GPU renderer is not configured"),
            Self::ComputeLaunch(error) => Display::fmt(error, formatter),
            Self::ImageBacking(reason) => write!(formatter, "Maxwell image backing cannot be lowered: {reason}"),
            Self::UnsupportedBlit(reason) => write!(formatter, "unsupported Maxwell 2D blit: {reason}"),
            Self::BlitFormatMismatch => formatter.write_str("2D image copy requires matching packed source and destination formats"),
            Self::BlitSourceNotResident => formatter.write_str("Maxwell color blit requires a current, fully initialized resident source"),
            Self::UnsupportedMultisampleState(reason) => write!(formatter, "Maxwell multisample state has no neutral lowering: {reason}"),
            Self::UnsupportedPolygonRasterization(reason) => write!(formatter, "MAXWELL_B polygon rasterization is unsupported: {reason}"),
            Self::UnsupportedWindowOrigin(value) => write!(formatter,
                "MAXWELL_B window origin/flip has no neutral coordinate lowering: value=0x{value:x}"),
            Self::ContradictoryState { reason } => {
                write!(formatter, "contradictory Maxwell 3D state: {reason}")
            }
            Self::TriggerStateMismatch => formatter.write_str("3D trigger does not match its consumed state"),
            Self::UnsupportedStreamOutputSemantics => formatter.write_str(
                "MAXWELL_B enabled stream output requires vertex capture to guest memory",
            ),
            Self::UnsupportedConditionalLoadConstantBufferSemantics => formatter.write_str(
                "MAXWELL_B conditional constant-buffer load has no verified execution semantics",
            ),
            Self::VisibleCallLimitExceeded {
                stage,
                required,
                limit,
            } => write!(
                formatter,
                "translated Maxwell shader exceeds SET_API_VISIBLE_CALL_LIMIT: stage={stage:?} required={required} limit={limit}"
            ),
            Self::UnsupportedColorReductionSemantics => formatter.write_str(
                "MAXWELL_B enabled color reduction has no verified neutral threshold evaluation or color-output semantics",
            ),
            Self::UnsupportedPostPsInitialCoverageSemantics => formatter.write_str("MAXWELL_B pre-shader initial coverage for post-shader tests is not represented by the neutral pipeline"),
            Self::UnsupportedViewportPixelDepthBounds => formatter.write_str("MAXWELL_B pixel depth bounds narrow the affine viewport range and require explicit clip/clamp lowering"),
            Self::UnsupportedWindowOriginFragmentPosition => formatter.write_str("MAXWELL_B lower-left fragment position Y requires shader coordinate lowering"),
            Self::UnsupportedPostZPixelShaderImaskSemantics => formatter.write_str(
                "MAXWELL_B post-Z pixel-shader invocation mask is not represented by the neutral pipeline",
            ),
            Self::UnsupportedPixelShaderInterlockSemantics(value) => write!(
                formatter,
                "MAXWELL_B pixel-shader interlock is not represented by the neutral pipeline: control={value:?}"
            ),
            Self::UnsupportedCsaaSemantics => formatter.write_str(
                "MAXWELL_B enabled CSAA has no verified coverage sampling, resolve, capability, or coherency semantics",
            ),
            Self::UnsupportedAntiAliasAlphaControl { alpha_to_coverage, alpha_to_one } => write!(
                formatter,
                "MAXWELL_B alpha-to-coverage/dithering or alpha-to-one is not represented by the neutral pipeline: alpha-to-coverage={alpha_to_coverage} alpha-to-one={alpha_to_one}"
            ),
            Self::UnsupportedCoverageToColorSemantics(value) => write!(
                formatter,
                "MAXWELL_B coverage-to-color output is not represented by the neutral pipeline: color-target={}",
                value.color_target()
            ),
            Self::UnsupportedAlphaToCoverageOverrideSemantics(value) => write!(
                formatter,
                "MAXWELL_B alpha-to-coverage override qualification is not represented by the neutral pipeline: qualify-by-aa={} qualify-by-ps-sample-mask={}",
                value.qualify_by_anti_alias_enable(),
                value.qualify_by_pixel_shader_sample_mask()
            ),
            Self::UnsupportedTirSemantics { control } => write!(
                formatter,
                "MAXWELL_B enabled target-independent rasterization has no neutral raster, coverage, alpha-to-coverage, or query representation: control={control:?}"
            ),
            Self::UnsupportedHybridAntiAliasSemantics(value) => write!(
                formatter,
                "MAXWELL_B hybrid antialiasing is not represented by the neutral raster pipeline: passes={} centroid={:?} passes-extended={}",
                value.passes(),
                value.centroid(),
                value.passes_extended()
            ),
            Self::UnsupportedSampleLocationsSemantics { group, value } => write!(
                formatter,
                "MAXWELL_B custom sample locations are not represented by the neutral raster pipeline: group={group} raw=0x{:08x}",
                value.raw()
            ),
            Self::UnsupportedPsOutputSampleMaskSemantics => formatter.write_str(
                "MAXWELL_B effective pixel-shader sample-mask output has no shader translation or neutral backend representation",
            ),
            Self::UnsupportedReplicatedColorTargetOutputSemantics => formatter.write_str(
                "MAXWELL_B disabled separate MRT fragment data requires replicating fragment color output zero to every active color target",
            ),
            Self::UnsupportedRenderTargetIndexOffsetSemantics(value) => write!(
                formatter,
                "MAXWELL_B viewport-index render-target routing is not represented by the neutral attachment contract: mode={value:?}"
            ),
            Self::UnsupportedRenderTargetLayerSemantics(value) => write!(
                formatter,
                "MAXWELL_B render-target layer routing is not represented by shader translation or the neutral attachment contract: layer={} control={:?}",
                value.layer(),
                value.control()
            ),
            Self::UnsupportedShaderLocalMemorySemantics {
                default_size_per_warp,
            } => write!(
                formatter,
                "MAXWELL_B active shader-local-memory allocation has no translated-shader or neutral backend representation: default-size-per-warp={}",
                default_size_per_warp.bytes()
            ),
            Self::UnsupportedViewportPixelCenterSemantics(center) => write!(
                formatter,
                "MAXWELL_B viewport pixel-center convention is not represented by the neutral pipeline contract: center={center:?}"
            ),
            Self::UnsupportedViewportSnapGridPrecision { viewport, precision } => write!(formatter, "MAXWELL_B viewport {viewport} increased snap-grid precision is unsupported: X={} Y={}", precision[0], precision[1]),
            Self::UnsupportedViewportCoordinateSwizzleSemantics { viewport, swizzle } => write!(
                formatter,
                "MAXWELL_B viewport coordinate swizzle is not represented by the neutral pipeline contract: viewport={viewport} components={:?}",
                swizzle.components()
            ),
            Self::UnsupportedWindowClipSemantics => formatter.write_str(
                "MAXWELL_B enabled window clipping has no neutral pipeline or backend rasterization semantics",
            ),
            Self::UnsupportedClipIdTestSemantics => formatter.write_str(
                "MAXWELL_B enabled clip-ID testing has no implemented extent, surface-ID, comparison, or backend rasterization semantics",
            ),

            Self::UnsupportedClearStencilMaskSemantics => formatter.write_str(
                "MAXWELL_B partial stencil write mask on clear has no neutral backend representation",
            ),
            Self::UnsupportedAliasedLineWidthSemantics => formatter.write_str(
                "MAXWELL_B aliased line-width selection has no represented width register or host rasterization semantics",
            ),
            Self::UnsupportedLineStippleSemantics { factor, pattern } => write!(
                formatter,
                "MAXWELL_B line stippling has no neutral backend representation: factor={factor} pattern=0x{pattern:04x}"
            ),
            Self::UnsupportedPolygonClipGeneratedEdgeSemantics => formatter.write_str(
                "MAXWELL_B suppression of polygon-clip-generated edges has no neutral backend representation",
            ),
            Self::UnsupportedSkippedVertexComponent { attribute, component } => write!(formatter,
                "vertex shader consumes a skipped DA attribute component: attribute={attribute} component={component}; DA default input lowering is not implemented"),
            Self::InvalidPatchSize(size) => write!(
                formatter,
                "MAXWELL_B patch draw has an invalid control-point count: {}",
                size.control_points()
            ),
            Self::TessellationMode { value, reason } => write!(formatter, "MAXWELL_B tessellation mode cannot be consumed: value={:#x} reason={reason:?}", value.raw()),
            Self::UnsupportedPointSpriteCoordinatesSemantics(select) => write!(
                formatter,
                "MAXWELL_B generated point-sprite coordinates are not represented by shader translation or the neutral pipeline contract: texture-mask=0x{:03x} r-mode={:?} origin={:?}",
                select.generated_texture_mask(),
                select.r_mode(),
                select.origin()
            ),
            Self::UnsupportedAttributePointSizeSemantics { slot } => write!(
                formatter,
                "MAXWELL_B shader-provided point size is not represented by shader or neutral backend lowering: slot={slot}"
            ),
            Self::UnsupportedPointSpriteSemantics => formatter.write_str(
                "MAXWELL_B enabled point-sprite rasterization is not represented by the neutral backend",
            ),
            Self::UnsupportedAntiAliasedPointSemantics => formatter.write_str(
                "MAXWELL_B anti-aliased point rasterization is not represented by the neutral backend",
            ),
            Self::UnsupportedPointCenterSemantics(mode) => write!(
                formatter,
                "MAXWELL_B point-center convention is not represented by the neutral pipeline contract: mode={mode:?}"
            ),
            Self::UnsupportedFillViaTriangleSemantics(mode) => write!(
                formatter,
                "MAXWELL_B fill-via-triangle mode is not represented by the neutral pipeline contract: mode={mode:?}"
            ),
            Self::UnsupportedFillRectangleDraw(reason) => write!(
                formatter,
                "MAXWELL_B fill-rectangle draw is not representable: {reason}"
            ),
            Self::UnsupportedConservativeRasterSemantics => formatter.write_str(
                "MAXWELL_B conservative rasterization is not represented by the neutral pipeline contract",
            ),
            Self::UnsupportedPolygonSmoothSemantics => formatter.write_str(
                "MAXWELL_B polygon smoothing is not represented by the neutral pipeline contract",
            ),
            Self::UnsupportedPolygonStippleSemantics => formatter.write_str(
                "MAXWELL_B polygon stippling is not represented by the neutral pipeline contract",
            ),
            Self::UnsupportedEdgeFlagSemantics(flag) => write!(
                formatter,
                "MAXWELL_B disabled polygon edge flag is not represented by the neutral pipeline contract: flag={flag:?}"
            ),
            Self::UnsupportedShadeModeSemantics(mode) => write!(
                formatter,
                "MAXWELL_B shade mode is not representable in the neutral pipeline contract: mode={mode:?}"
            ),
            Self::UnsupportedProvokingVertexSemantics(vertex) => write!(
                formatter,
                "MAXWELL_B provoking vertex is not representable in the neutral pipeline or shader interpolation contract: vertex={vertex:?}"
            ),
            Self::UnsupportedTwoSidedLightSemantics => formatter.write_str(
                "MAXWELL_B enabled two-sided fixed-function lighting is not represented by shader or neutral backend lowering",
            ),
            Self::UnsupportedPixelShaderSaturateSemantics { output, range } => write!(
                formatter,
                "MAXWELL_B pixel-shader output saturation is not represented by shader or neutral backend lowering: output={output} range={range:?}"
            ),
            Self::IncompleteAlphaTestState(field) => write!(
                formatter,
                "MAXWELL_B enabled alpha testing requires SET_ALPHA_{field}"
            ),
            Self::CompressedDepthImportRequired { kind } => write!(
                formatter,
                "Maxwell compressed depth contents require materialization before use: kind=0x{kind:02x}"
            ),
            Self::CompressedSampledImageImportRequired { role, kind } => write!(
                formatter,
                "Maxwell compressed sampled image has no current materialized resident image: role={role:?} kind={kind:#04x}"
            ),
            Self::CompressedColorImportRequired { target } => write!(
                formatter,
                "Maxwell compressed color contents require materialization before use: target={target}"
            ),
            Self::ShaderTranslationRequired => {
                formatter.write_str("Maxwell shader translation is required before draw lowering")
            }
            Self::InvalidTranslatedShaders => {
                formatter.write_str("translated shader evidence is empty, duplicated, or invalid")
            }
            Self::TranslatedShaderStageMismatch => formatter
                .write_str("translated shader stages do not match enabled Maxwell pipeline stages"),
            Self::TranslatedShaderMemoryConfigurationMismatch {
                stage,
                configured,
                required,
            } => write!(
                formatter,
                "translated shader requires a different Maxwell directly addressable memory configuration: stage={stage:?} configured-bytes={} translated-for-bytes={}",
                configured.bytes(),
                required.bytes()
            ),
            Self::UnsupportedShaderStage(stage) => write!(
                formatter,
                "Maxwell shader stage has no neutral lowering: {stage:?}"
            ),
            Self::InvalidShaderResourceUse { role } => write!(
                formatter,
                "translated shader declares an invalid resource use: role={role:?}"
            ),
            Self::InvalidResolvedView { role } => write!(
                formatter,
                "resolved 3D view cannot be re-identified neutrally: role={role:?}"
            ),
            Self::AllocationDescriptionChanged { allocation } => write!(
                formatter,
                "cached GPU allocation changed immutable description: {allocation}"
            ),
            Self::IncompleteClear(field) => {
                write!(formatter, "clear state is incomplete: missing={field}")
            }
            Self::EmptyClearRectangle => formatter.write_str(
                "clear rectangle, scissor, and viewport-clip intersection is empty",
            ),
            Self::PartialColorClearUnsupported { mask } => write!(
                formatter,
                "partial color-channel clear is not represented yet: mask={mask:#x}"
            ),
            Self::ClearOutsideAttachment => {
                formatter.write_str("clear rectangle or layer lies outside the resolved attachment")
            }
            Self::IncompleteDraw(field) => {
                write!(formatter, "draw state is incomplete: missing={field}")
            }
            Self::InvalidTransition => {
                formatter.write_str("derived neutral resource transition is invalid")
            }
            Self::InvalidResourceCreation => {
                formatter.write_str("derived neutral resource creation is invalid")
            }
            Self::Command(error) => {
                write!(formatter, "neutral command construction failed: {error}")
            }
            Self::ResourceExhausted => {
                formatter.write_str("GPU lowering exhausted host resources or identities")
            }
        }
    }
}

impl std::error::Error for MaxwellLoweringError {}
