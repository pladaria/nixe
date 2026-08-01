//! GM20B `MAXWELL_B` 3D class methods and register transitions.

mod bindings;
mod coverage;
mod draw;
mod line;
mod output;
mod render_enable;
mod render_targets;
mod resource;
mod shader_execution;
mod state;
mod vertex;

pub use bindings::{
    MAXWELL_BIND_GROUP_COUNT, MAXWELL_CONSTANT_BUFFER_SLOT_COUNT, MAXWELL_PIPELINE_SHADER_COUNT,
    MaxwellThreeDBindGroupState, MaxwellThreeDConstantBufferBinding,
    MaxwellThreeDConstantBufferSelectorState, MaxwellThreeDDescriptorPoolState,
    MaxwellThreeDPipelineBindingState, MaxwellThreeDSamplerBindingMode,
    MaxwellThreeDShaderBindingState, MaxwellThreeDShaderBindingWrite, MaxwellThreeDShaderStage,
};
pub use coverage::{
    MaxwellThreeDCoverageState, MaxwellThreeDCoverageStateWrite, MaxwellThreeDCsaaEnable,
};
pub use draw::{
    MaxwellThreeDLoweredWork, MaxwellThreeDLoweringCache, MaxwellThreeDLoweringError,
    MaxwellThreeDLoweringPlan, MaxwellThreeDOperationTrigger, MaxwellThreeDShaderResourceUse,
    MaxwellThreeDTranslatedShader, MaxwellThreeDTranslatedShaders,
    preflight_maxwell_three_d_operation,
};
pub use line::{
    MaxwellThreeDAliasedLineWidthEnable, MaxwellThreeDLineState, MaxwellThreeDLineStateWrite,
};

pub use output::{
    MAXWELL_SCISSOR_COUNT, MAXWELL_VIEWPORT_COUNT, MaxwellThreeDBlendFactor, MaxwellThreeDBlendOp,
    MaxwellThreeDColorMask, MaxwellThreeDCompareOp, MaxwellThreeDCullFace,
    MaxwellThreeDFixedFunctionRegister, MaxwellThreeDFixedFunctionState,
    MaxwellThreeDFixedFunctionValue, MaxwellThreeDFixedFunctionWrite, MaxwellThreeDFrontFace,
    MaxwellThreeDPolygonMode, MaxwellThreeDSampleMode, MaxwellThreeDScissorState,
    MaxwellThreeDStencilOp, MaxwellThreeDViewportClipControl, MaxwellThreeDViewportTransformState,
};
pub use render_enable::{
    MaxwellThreeDRenderEnableMode, MaxwellThreeDRenderEnableState,
    MaxwellThreeDRenderEnableStateWrite,
};
pub use render_targets::{
    MAXWELL_COLOR_TARGET_COUNT, MaxwellThreeDAttachmentReadiness, MaxwellThreeDClearState,
    MaxwellThreeDClearSurface, MaxwellThreeDColorCompressionMode, MaxwellThreeDColorTargetFormat,
    MaxwellThreeDColorTargetSelection, MaxwellThreeDColorTargetState,
    MaxwellThreeDDepthStencilFormat, MaxwellThreeDDepthStencilTargetState, MaxwellThreeDImageKind,
    MaxwellThreeDImageLayout, MaxwellThreeDRawValue, MaxwellThreeDRectangle,
    MaxwellThreeDRenderTargetState, MaxwellThreeDRenderTargetWrite, MaxwellThreeDZCompressionMode,
};
pub use resource::{
    MaxwellThreeDDirtySubresource, MaxwellThreeDDirtySubresources, MaxwellThreeDMappingReference,
    MaxwellThreeDPreservedImageLayout, MaxwellThreeDResolvedBuffer, MaxwellThreeDResolvedImage,
    MaxwellThreeDResolvedResource, MaxwellThreeDResolvedResources, MaxwellThreeDResourceAccess,
    MaxwellThreeDResourceAlias, MaxwellThreeDResourceError, MaxwellThreeDResourceRole,
    resolve_maxwell_three_d_resources,
};
pub use shader_execution::{
    MAXWELL_THREE_D_SM_TIMEOUT_COUNTER_BIT_MAX, MaxwellThreeDShaderExecutionState,
    MaxwellThreeDShaderExecutionStateWrite, MaxwellThreeDSmTimeoutCounterBit,
};
pub use state::{
    MaxwellThreeDPointSize, MaxwellThreeDRasterState, MaxwellThreeDRegister,
    MaxwellThreeDRegisterOrigin, MaxwellThreeDState, MaxwellThreeDStateWrite,
    MaxwellThreeDViewportState, MaxwellThreeDViewportZClipRange,
};
pub use vertex::{
    MAXWELL_VERTEX_ATTRIBUTE_COUNT, MAXWELL_VERTEX_STREAM_COUNT, MaxwellThreeDBegin,
    MaxwellThreeDIndexBufferState, MaxwellThreeDIndexElementSize, MaxwellThreeDPrimitiveState,
    MaxwellThreeDPrimitiveTopology, MaxwellThreeDUnresolvedAddress,
    MaxwellThreeDVertexAttributeFormat, MaxwellThreeDVertexComponentWidths,
    MaxwellThreeDVertexInputState, MaxwellThreeDVertexInputWrite, MaxwellThreeDVertexNumericalType,
    MaxwellThreeDVertexStreamFormat, MaxwellThreeDVertexStreamState,
};

use nixe_gpu::{GpuClassId, GpuMethodId};

use super::{
    MaxwellEngineCapability, MaxwellEngineDispatchError, MaxwellEngineMethodDispatch,
    MaxwellEngineMethodEffect, MaxwellEngineMethodMetadata,
};
use crate::MaxwellMethodDispatch;

pub(super) const CLASS: GpuClassId = GpuClassId(0xb197);
const CLASS_NAME: &str = "MAXWELL_B";

#[derive(Clone, Copy)]
enum MethodAction {
    NoOperation,
    PointSize,
    ViewportZClip,
    RenderEnableMode,
    SmTimeoutCounterBit,
    CsaaEnable,
    AliasedLineWidthEnable,
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
// SET_RENDER_ENABLE_A/B/C and all five C modes are defined at:
// https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L2759-L2771
// SET_SM_TIMEOUT_INTERVAL.COUNTER_BIT is defined at bits 5:0 here; this does
// not document a duration formula or watchdog behavior:
// https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L1079-L1090
// SET_Z_COMPRESSION.ENABLE and its FALSE/TRUE values are defined here:
// https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L3545-L3548
// SET_COLOR_COMPRESSION(i).ENABLE and its FALSE/TRUE values are defined here:
// https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L3575-L3578
// SET_CT_SELECT's count and all eight target selectors are defined here:
// https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L2003-L2012
// NVIDIA's public MAXWELL_B header leaves address 0x15b4 unnamed; that omission
// is not evidence that the method is a no-op or reserved:
// https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L2864-L2888
// The pinned envytools register database identifies 0x15b4 as CSAA_ENABLE and
// publishes its FALSE/TRUE values. It does not establish the later execution
// semantics of enabled coverage sampling:
// https://github.com/envytools/envytools/blob/f102b82381f3f11cee113d16374c87091db039d9/rnndb/graph/gf100_3d.xml#L831-L838
// SET_ALIASED_LINE_WIDTH_ENABLE and its FALSE/TRUE encodings are defined here:
// https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h#L244-L247
// The pinned envytools database independently calls the same selector
// LINE_WIDTH_SEPARATE; that name does not establish broader raster semantics:
// https://github.com/envytools/envytools/blob/f102b82381f3f11cee113d16374c87091db039d9/rnndb/graph/gf100_3d.xml#L60-L68
methods!(
    NO_OPERATION => (0x0100, "NO_OPERATION", u32::MAX, MethodAction::NoOperation),
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
        MethodAction::Missing(MaxwellEngineCapability::NeutralExecution)
    ),
    SET_MME_SHADOW_RAM_CONTROL => (
        0x0124,
        "SET_MME_SHADOW_RAM_CONTROL",
        0x0000_0003,
        MethodAction::Missing(MaxwellEngineCapability::NeutralExecution)
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
    SET_RENDER_ENABLE_C => (
        0x1558,
        "SET_RENDER_ENABLE_C",
        0x0000_0007,
        MethodAction::RenderEnableMode
    ),
    CSAA_ENABLE => (
        0x15b4,
        "CSAA_ENABLE",
        0x0000_0001,
        MethodAction::CsaaEnable
    ),
);

pub(super) fn preflight(
    method: MaxwellMethodDispatch,
    candidate: &mut MaxwellThreeDState,
) -> Result<MaxwellEngineMethodDispatch, MaxwellEngineDispatchError> {
    let source = method.source();
    if let Some((write, method_name)) = preflight_vertex_and_binding_state(source, candidate)? {
        candidate.apply(write);
        let metadata =
            MaxwellEngineMethodMetadata::new(CLASS, CLASS_NAME, source.method(), method_name);
        return Ok(MaxwellEngineMethodDispatch::new(
            method,
            metadata,
            MaxwellEngineMethodEffect::ThreeDState(write),
        ));
    }
    if let Some((write, method_name)) = preflight_output_state(source)? {
        candidate.apply(write);
        let metadata =
            MaxwellEngineMethodMetadata::new(CLASS, CLASS_NAME, source.method(), method_name);
        let effect = if matches!(
            write,
            MaxwellThreeDStateWrite::RenderTarget(
                MaxwellThreeDRenderTargetWrite::ClearSurface { .. }
            )
        ) {
            MaxwellEngineMethodEffect::ThreeDStateAndTrigger {
                state: write,
                trigger: MaxwellThreeDOperationTrigger::ClearSurface { source },
            }
        } else {
            MaxwellEngineMethodEffect::ThreeDState(write)
        };
        return Ok(MaxwellEngineMethodDispatch::new(method, metadata, effect));
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
    let effect =
        match declaration.action {
            MethodAction::NoOperation => MaxwellEngineMethodEffect::NoOperation,
            MethodAction::PointSize => {
                let write = MaxwellThreeDStateWrite::PointSize {
                    value: MaxwellThreeDPointSize::from_bits(source.argument()),
                    source,
                };
                candidate.apply(write);
                MaxwellEngineMethodEffect::ThreeDState(write)
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
                MaxwellEngineMethodEffect::ThreeDState(write)
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
                MaxwellEngineMethodEffect::ThreeDState(write)
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
                MaxwellEngineMethodEffect::ThreeDState(write)
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
                MaxwellEngineMethodEffect::ThreeDState(write)
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
                MaxwellEngineMethodEffect::ThreeDState(write)
            }
            MethodAction::DrawVertexArray => {
                if source.argument() == 0 {
                    return Err(invalid_encoding(
                        source,
                        "DRAW_VERTEX_ARRAY",
                        "vertex count is zero",
                    ));
                }
                MaxwellEngineMethodEffect::ThreeDTrigger(
                    MaxwellThreeDOperationTrigger::DrawVertexArray {
                        source,
                        vertex_count: source.argument(),
                    },
                )
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
    Ok(MaxwellEngineMethodDispatch::new(
        method,
        *declaration.metadata,
        effect,
    ))
}

fn preflight_vertex_and_binding_state(
    source: crate::MaxwellMethodSource,
    candidate: &MaxwellThreeDState,
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
            0x0d74 => Some((
                V::VertexArrayStart { value: raw, source },
                "SET_VERTEX_ARRAY_START",
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
        let selector = candidate.shader_bindings().selector();
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

    use MaxwellThreeDFixedFunctionRegister as R;
    use MaxwellThreeDFixedFunctionValue as V;
    let fixed = match method {
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
        0x12cc => (
            R::DepthTestEnable,
            V::Boolean(checked_bool(source, "SET_DEPTH_TEST")?),
            "SET_DEPTH_TEST",
        ),
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
        0x130c => {
            let value = MaxwellThreeDCompareOp::parse(raw).ok_or_else(|| {
                invalid_encoding(source, "SET_DEPTH_FUNC", "unknown compare operation")
            })?;
            (R::DepthCompare, V::Compare(value), "SET_DEPTH_FUNC")
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
        0x1950 if raw <= 2 => (R::WindowClipType, V::Mask(raw), "SET_WINDOW_CLIP_TYPE"),
        0x1950 => {
            return Err(invalid_encoding(
                source,
                "SET_WINDOW_CLIP_TYPE",
                "unknown window clip type",
            ));
        }
        0x19bc => (
            R::DepthBoundsEnable,
            V::Boolean(checked_bool(source, "SET_DEPTH_BOUNDS_TEST")?),
            "SET_DEPTH_BOUNDS_TEST",
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
