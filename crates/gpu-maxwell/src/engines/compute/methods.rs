//! `MAXWELL_COMPUTE_B` method declarations and validated register writes.

use nixe_gpu::GpuMethodId;

use super::{
    CLASS, MaxwellComputeBindlessTextureConstantBufferSlot, MaxwellComputeCwdRefCounterIndex,
    MaxwellComputeCwdRefCounterValue, MaxwellComputeInlineToMemoryLaunch,
    MaxwellComputeInlineToMemoryPendingTransfer, MaxwellComputeInlineToMemoryUpload,
    MaxwellComputeOperationTrigger, MaxwellComputeShaderCacheInvalidation, MaxwellComputeSmCount,
    MaxwellComputeSpaVersion, MaxwellComputeState, MaxwellComputeStateWrite,
    MaxwellComputeTriggeredOperation,
};
use crate::engines::{
    AppliedMethod, MaxwellEngineDispatchError, MaxwellEngineMethodMetadata, MaxwellEngineOperation,
};
use crate::{MaxwellMethodDispatch, MaxwellMethodSource};

const CLASS_NAME: &str = "MAXWELL_COMPUTE_B";

#[derive(Clone, Copy)]
enum MethodAction {
    AddressUpper,
    AddressLower,
    NonThrottledSizeUpper,
    NonThrottledSizeLower,
    NonThrottledMaxSmCount,
    ThrottledSizeUpper,
    ThrottledSizeLower,
    ThrottledMaxSmCount,
    LocalWindowBase,
    SharedWindowBase,
    InlineToMemoryLineLength,
    InlineToMemoryLineCount,
    InlineToMemoryAddressUpper,
    InlineToMemoryAddressLower,
    InlineToMemoryLaunch,
    InlineToMemoryData,
    ProgramRegionAddressUpper,
    ProgramRegionAddressLower,
    SpaVersion,
    TextureHeaderAddressUpper,
    TextureHeaderAddressLower,
    TextureHeaderMaximumIndex,
    SamplerAddressUpper,
    SamplerAddressLower,
    SamplerMaximumIndex,
    BindlessTextureConstantBufferSlot,
    CwdReferenceCounter,
    WaitForIdle,
    InvalidateShaderCachesNoWfi,
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

// Field widths are pinned to NVIDIA's public MAXWELL_COMPUTE_B header. It
// publishes no reset values for these registers, so all state starts unset.
// https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/compute/clb1c0.h#L206-L207
// https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/compute/clb1c0.h#L364-L380
// https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/compute/clb1c0.h#L482-L489
// https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/compute/clb1c0.h#L382-L384
// https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/compute/clb1c0.h#L625-L629
// https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/compute/clb1c0.h#L607-L623
// https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/compute/clb1c0.h#L266-L268
// https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/compute/clb1c0.h#L51-L52
// https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/compute/clb1c0.h#L701-L702
// https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/compute/clb1c0.h#L83-L170
// https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/compute/clb1c0.h#L631-L640
methods!(
    WAIT_FOR_IDLE => (0x0110, "WAIT_FOR_IDLE", u32::MAX, MethodAction::WaitForIdle),
    LINE_LENGTH_IN => (0x0180, "LINE_LENGTH_IN", u32::MAX, MethodAction::InlineToMemoryLineLength),
    LINE_COUNT => (0x0184, "LINE_COUNT", u32::MAX, MethodAction::InlineToMemoryLineCount),
    OFFSET_OUT_UPPER => (0x0188, "OFFSET_OUT_UPPER", 0x0000_00ff, MethodAction::InlineToMemoryAddressUpper),
    OFFSET_OUT => (0x018c, "OFFSET_OUT", u32::MAX, MethodAction::InlineToMemoryAddressLower),
    LAUNCH_DMA => (0x01b0, "LAUNCH_DMA", 0x0000_f37f, MethodAction::InlineToMemoryLaunch),
    LOAD_INLINE_DATA => (0x01b4, "LOAD_INLINE_DATA", u32::MAX, MethodAction::InlineToMemoryData),
    SET_SHADER_SHARED_MEMORY_WINDOW => (0x0214, "SET_SHADER_SHARED_MEMORY_WINDOW", u32::MAX, MethodAction::SharedWindowBase),
    SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A => (0x02e4, "SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A", 0x0000_00ff, MethodAction::NonThrottledSizeUpper),
    SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_B => (0x02e8, "SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_B", u32::MAX, MethodAction::NonThrottledSizeLower),
    SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_C => (0x02ec, "SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_C", 0x0000_01ff, MethodAction::NonThrottledMaxSmCount),
    SET_SHADER_LOCAL_MEMORY_THROTTLED_A => (0x02f0, "SET_SHADER_LOCAL_MEMORY_THROTTLED_A", 0x0000_00ff, MethodAction::ThrottledSizeUpper),
    SET_SHADER_LOCAL_MEMORY_THROTTLED_B => (0x02f4, "SET_SHADER_LOCAL_MEMORY_THROTTLED_B", u32::MAX, MethodAction::ThrottledSizeLower),
    SET_SHADER_LOCAL_MEMORY_THROTTLED_C => (0x02f8, "SET_SHADER_LOCAL_MEMORY_THROTTLED_C", 0x0000_01ff, MethodAction::ThrottledMaxSmCount),
    SET_SHADER_LOCAL_MEMORY_WINDOW => (0x077c, "SET_SHADER_LOCAL_MEMORY_WINDOW", u32::MAX, MethodAction::LocalWindowBase),
    SET_SHADER_LOCAL_MEMORY_A => (0x0790, "SET_SHADER_LOCAL_MEMORY_A", 0x0000_00ff, MethodAction::AddressUpper),
    SET_SHADER_LOCAL_MEMORY_B => (0x0794, "SET_SHADER_LOCAL_MEMORY_B", u32::MAX, MethodAction::AddressLower),
    SET_SPA_VERSION => (0x0310, "SET_SPA_VERSION", 0x0000_ffff, MethodAction::SpaVersion),
    SET_PROGRAM_REGION_A => (0x1608, "SET_PROGRAM_REGION_A", 0x0000_00ff, MethodAction::ProgramRegionAddressUpper),
    SET_PROGRAM_REGION_B => (0x160c, "SET_PROGRAM_REGION_B", u32::MAX, MethodAction::ProgramRegionAddressLower),
    INVALIDATE_SHADER_CACHES_NO_WFI => (0x1698, "INVALIDATE_SHADER_CACHES_NO_WFI", 0x0000_1011, MethodAction::InvalidateShaderCachesNoWfi),
    SET_TEX_SAMPLER_POOL_A => (0x155c, "SET_TEX_SAMPLER_POOL_A", 0x0000_00ff, MethodAction::SamplerAddressUpper),
    SET_TEX_SAMPLER_POOL_B => (0x1560, "SET_TEX_SAMPLER_POOL_B", u32::MAX, MethodAction::SamplerAddressLower),
    SET_TEX_SAMPLER_POOL_C => (0x1564, "SET_TEX_SAMPLER_POOL_C", 0x000f_ffff, MethodAction::SamplerMaximumIndex),
    SET_TEX_HEADER_POOL_A => (0x1574, "SET_TEX_HEADER_POOL_A", 0x0000_00ff, MethodAction::TextureHeaderAddressUpper),
    SET_TEX_HEADER_POOL_B => (0x1578, "SET_TEX_HEADER_POOL_B", u32::MAX, MethodAction::TextureHeaderAddressLower),
    SET_TEX_HEADER_POOL_C => (0x157c, "SET_TEX_HEADER_POOL_C", 0x003f_ffff, MethodAction::TextureHeaderMaximumIndex),
    SET_BINDLESS_TEXTURE => (0x2608, "SET_BINDLESS_TEXTURE", 0x0000_0007, MethodAction::BindlessTextureConstantBufferSlot),
    SET_CWD_REF_COUNTER => (0x0248, "SET_CWD_REF_COUNTER", 0x00ff_ff3f, MethodAction::CwdReferenceCounter),
);

pub(super) fn preflight(
    method: MaxwellMethodDispatch,
    candidate: &mut MaxwellComputeState,
) -> Result<AppliedMethod, MaxwellEngineDispatchError> {
    let source = method.source();
    let declaration = METHODS
        .iter()
        .find(|declaration| declaration.metadata.method() == source.method())
        .ok_or(MaxwellEngineDispatchError::UnknownMethod {
            source,
            class_name: CLASS_NAME,
        })?;
    if source.argument() & !declaration.defined_mask != 0 {
        return Err(invalid_value(source, declaration));
    }

    let raw = source.argument();
    let trigger = match declaration.action {
        MethodAction::WaitForIdle => {
            Some(MaxwellComputeOperationTrigger::WaitForIdle { value: raw, source })
        }
        MethodAction::InvalidateShaderCachesNoWfi => Some(
            MaxwellComputeOperationTrigger::InvalidateShaderCachesNoWfi {
                caches: MaxwellComputeShaderCacheInvalidation::new(
                    raw & 1 != 0,
                    raw & 0x10 != 0,
                    raw & 0x1000 != 0,
                ),
                source,
            },
        ),
        _ => None,
    };
    if let Some(trigger) = trigger {
        return Ok(AppliedMethod::new(
            method,
            *declaration.metadata,
            Some(MaxwellEngineOperation::ComputeSynchronization(Box::new(
                MaxwellComputeTriggeredOperation::new(trigger, candidate.clone()),
            ))),
        ));
    }
    if matches!(declaration.action, MethodAction::InlineToMemoryData) {
        let pending = candidate.inline_to_memory().pending().ok_or_else(|| {
            invalid_encoding(
                source,
                declaration.metadata.method_name(),
                "inline data requires an armed LAUNCH_DMA transfer",
            )
        })?;
        let next_offset = pending.next_offset().checked_add(4).ok_or_else(|| {
            invalid_encoding(
                source,
                declaration.metadata.method_name(),
                "inline upload cursor overflows",
            )
        })?;
        if next_offset > pending.byte_length() {
            return Err(invalid_encoding(
                source,
                declaration.metadata.method_name(),
                "inline data exceeds the armed transfer length",
            ));
        }
        let write = MaxwellComputeStateWrite::InlineToMemoryData {
            value: raw,
            next_offset,
            source,
        };
        let upload = MaxwellComputeInlineToMemoryUpload::new(
            pending.address(),
            pending.next_offset(),
            raw,
            source,
        );
        candidate.apply(write);
        return Ok(AppliedMethod::new(
            method,
            *declaration.metadata,
            Some(MaxwellEngineOperation::ComputeInlineToMemory(upload)),
        ));
    }
    let write = match declaration.action {
        MethodAction::AddressUpper => MaxwellComputeStateWrite::AddressUpper {
            value: raw as u8,
            source,
        },
        MethodAction::AddressLower => MaxwellComputeStateWrite::AddressLower { value: raw, source },
        MethodAction::NonThrottledSizeUpper => MaxwellComputeStateWrite::NonThrottledSizeUpper {
            value: raw as u8,
            source,
        },
        MethodAction::NonThrottledSizeLower => {
            MaxwellComputeStateWrite::NonThrottledSizeLower { value: raw, source }
        }
        MethodAction::NonThrottledMaxSmCount => MaxwellComputeStateWrite::NonThrottledMaxSmCount {
            value: MaxwellComputeSmCount::parse(raw)
                .ok_or_else(|| invalid_value(source, declaration))?,
            source,
        },
        MethodAction::ThrottledSizeUpper => MaxwellComputeStateWrite::ThrottledSizeUpper {
            value: raw as u8,
            source,
        },
        MethodAction::ThrottledSizeLower => {
            MaxwellComputeStateWrite::ThrottledSizeLower { value: raw, source }
        }
        MethodAction::ThrottledMaxSmCount => MaxwellComputeStateWrite::ThrottledMaxSmCount {
            value: MaxwellComputeSmCount::parse(raw)
                .ok_or_else(|| invalid_value(source, declaration))?,
            source,
        },
        MethodAction::LocalWindowBase => {
            MaxwellComputeStateWrite::LocalWindowBase { value: raw, source }
        }
        MethodAction::SharedWindowBase => {
            MaxwellComputeStateWrite::SharedWindowBase { value: raw, source }
        }
        MethodAction::InlineToMemoryLineLength => {
            MaxwellComputeStateWrite::InlineToMemoryLineLength { value: raw, source }
        }
        MethodAction::InlineToMemoryLineCount => {
            MaxwellComputeStateWrite::InlineToMemoryLineCount { value: raw, source }
        }
        MethodAction::InlineToMemoryAddressUpper => {
            MaxwellComputeStateWrite::InlineToMemoryAddressUpper {
                value: raw as u8,
                source,
            }
        }
        MethodAction::InlineToMemoryAddressLower => {
            MaxwellComputeStateWrite::InlineToMemoryAddressLower { value: raw, source }
        }
        MethodAction::InlineToMemoryLaunch => {
            if raw != 0x0000_0041 {
                return Err(invalid_encoding(
                    source,
                    declaration.metadata.method_name(),
                    "only the captured pitch, no-reduction, no-completion inline upload is implemented",
                ));
            }
            let inline = candidate.inline_to_memory();
            if inline.pending().is_some() {
                return Err(invalid_encoding(
                    source,
                    declaration.metadata.method_name(),
                    "cannot replace an incomplete inline upload",
                ));
            }
            let address = inline.address().ok_or_else(|| {
                invalid_encoding(
                    source,
                    declaration.metadata.method_name(),
                    "launch requires a complete destination address",
                )
            })?;
            let line_length = *inline.line_length().value().ok_or_else(|| {
                invalid_encoding(
                    source,
                    declaration.metadata.method_name(),
                    "launch requires LINE_LENGTH_IN",
                )
            })?;
            let line_count = *inline.line_count().value().ok_or_else(|| {
                invalid_encoding(
                    source,
                    declaration.metadata.method_name(),
                    "launch requires LINE_COUNT",
                )
            })?;
            if line_length == 0 || !line_length.is_multiple_of(4) {
                return Err(invalid_encoding(
                    source,
                    declaration.metadata.method_name(),
                    "captured inline upload length must be nonzero and word-aligned",
                ));
            }
            if line_count != 1 {
                return Err(invalid_encoding(
                    source,
                    declaration.metadata.method_name(),
                    "multi-line pitch uploads require PITCH_OUT semantics",
                ));
            }
            if address
                .get()
                .checked_add(u64::from(line_length))
                .is_none_or(|end| end > (1_u64 << 40))
            {
                return Err(invalid_encoding(
                    source,
                    declaration.metadata.method_name(),
                    "inline upload GPU range overflows",
                ));
            }
            MaxwellComputeStateWrite::InlineToMemoryLaunch {
                value: MaxwellComputeInlineToMemoryLaunch::captured_pitch(),
                pending: MaxwellComputeInlineToMemoryPendingTransfer::new(address, line_length),
                source,
            }
        }
        MethodAction::ProgramRegionAddressUpper => {
            MaxwellComputeStateWrite::ProgramRegionAddressUpper {
                value: raw as u8,
                source,
            }
        }
        MethodAction::ProgramRegionAddressLower => {
            MaxwellComputeStateWrite::ProgramRegionAddressLower { value: raw, source }
        }
        MethodAction::SpaVersion => MaxwellComputeStateWrite::SpaVersion {
            value: MaxwellComputeSpaVersion::parse(raw)
                .ok_or_else(|| invalid_value(source, declaration))?,
            source,
        },
        MethodAction::TextureHeaderAddressUpper => {
            MaxwellComputeStateWrite::TextureHeaderAddressUpper {
                value: raw as u8,
                source,
            }
        }
        MethodAction::TextureHeaderAddressLower => {
            MaxwellComputeStateWrite::TextureHeaderAddressLower { value: raw, source }
        }
        MethodAction::TextureHeaderMaximumIndex => {
            MaxwellComputeStateWrite::TextureHeaderMaximumIndex { value: raw, source }
        }
        MethodAction::SamplerAddressUpper => MaxwellComputeStateWrite::SamplerAddressUpper {
            value: raw as u8,
            source,
        },
        MethodAction::SamplerAddressLower => {
            MaxwellComputeStateWrite::SamplerAddressLower { value: raw, source }
        }
        MethodAction::SamplerMaximumIndex => {
            MaxwellComputeStateWrite::SamplerMaximumIndex { value: raw, source }
        }
        MethodAction::BindlessTextureConstantBufferSlot => {
            MaxwellComputeStateWrite::BindlessTextureConstantBufferSlot {
                value: MaxwellComputeBindlessTextureConstantBufferSlot::parse(raw)
                    .ok_or_else(|| invalid_value(source, declaration))?,
                source,
            }
        }
        MethodAction::CwdReferenceCounter => MaxwellComputeStateWrite::CwdReferenceCounter {
            index: MaxwellComputeCwdRefCounterIndex::new((raw & 0x3f) as u8)
                .expect("six-bit CWD reference-counter selector is always in range"),
            value: MaxwellComputeCwdRefCounterValue::new((raw >> 8) as u16),
            source,
        },
        MethodAction::WaitForIdle => unreachable!("WAIT_FOR_IDLE returns before state decoding"),
        MethodAction::InvalidateShaderCachesNoWfi => {
            unreachable!("INVALIDATE_SHADER_CACHES_NO_WFI returns before state decoding")
        }
        MethodAction::InlineToMemoryData => {
            unreachable!("LOAD_INLINE_DATA returns before state decoding")
        }
    };
    candidate.apply(write);
    Ok(AppliedMethod::new(method, *declaration.metadata, None))
}

fn invalid_encoding(
    source: MaxwellMethodSource,
    method_name: &'static str,
    reason: &'static str,
) -> MaxwellEngineDispatchError {
    MaxwellEngineDispatchError::InvalidComputeMethodEncoding {
        source,
        method_name,
        reason,
    }
}

fn invalid_value(
    source: MaxwellMethodSource,
    declaration: &MethodDeclaration,
) -> MaxwellEngineDispatchError {
    MaxwellEngineDispatchError::InvalidMethodValue {
        source,
        metadata: declaration.metadata,
        defined_mask: declaration.defined_mask,
    }
}
