//! `FERMI_TWOD_A` method declarations and validated register writes.

use nixe_gpu::GpuMethodId;

use super::{
    CLASS, MaxwellTwoDBeta1, MaxwellTwoDBeta4, MaxwellTwoDBetaStateWrite, MaxwellTwoDClipEnable,
    MaxwellTwoDColorKeyEnable, MaxwellTwoDNotifyAddressLower, MaxwellTwoDNotifyAddressUpper,
    MaxwellTwoDNotifyStateWrite, MaxwellTwoDOperation, MaxwellTwoDPixelsFromMemoryCorralSize,
    MaxwellTwoDPixelsFromMemorySafeOverlap, MaxwellTwoDPixelsFromMemoryStateWrite,
    MaxwellTwoDProcessingClusters, MaxwellTwoDRenderEnableMode, MaxwellTwoDRenderEnableStateWrite,
    MaxwellTwoDState, MaxwellTwoDStateWrite,
};
use crate::engines::{
    MaxwellEngineDispatchError, MaxwellEngineMethodDispatch, MaxwellEngineMethodEffect,
    MaxwellEngineMethodMetadata,
};
use crate::{MaxwellMethodDispatch, MaxwellMethodSource};

const CLASS_NAME: &str = "FERMI_TWOD_A";

#[derive(Clone, Copy)]
enum MethodAction {
    ProcessingClusters,
    Operation,
    ClipEnable,
    ColorKeyEnable,
    Beta1,
    Beta4,
    PixelsFromMemoryCorralSize,
    PixelsFromMemorySafeOverlap,
    RenderEnableMode,
    NotifyAddressUpper,
    NotifyAddressLower,
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

// Method fields and enum values are pinned to NVIDIA's generated public
// FERMI_TWOD_A header. It publishes no reset values for these registers, so
// state begins explicitly unset rather than assuming zero.
// https://github.com/NVIDIA/open-gpu-doc/blob/9e6d83fe0770bc8644850a0b1bf5ddb1519905ba/classes/twod/cl902d.h#L493-L579
// SET_COLOR_KEY_ENABLE and its False/True values are defined specifically at:
// https://github.com/NVIDIA/open-gpu-doc/blob/9e6d83fe0770bc8644850a0b1bf5ddb1519905ba/classes/twod/cl902d.h#L551-L557
// SET_BETA1 accepts all 32 bits while SET_BETA4 defines four byte-wide B/G/R/A
// components. Neither write independently triggers a 2D operation:
// https://github.com/NVIDIA/open-gpu-doc/blob/9e6d83fe0770bc8644850a0b1bf5ddb1519905ba/classes/twod/cl902d.h#L562-L571
// SET_PIXELS_FROM_MEMORY_CORRAL_SIZE is a 10-bit field. The header publishes
// no reset value, unit, or execution semantics for the value:
// https://github.com/NVIDIA/open-gpu-doc/blob/9e6d83fe0770bc8644850a0b1bf5ddb1519905ba/classes/twod/cl902d.h#L931-L932
// SET_PIXELS_FROM_MEMORY_SAFE_OVERLAP defines False=0 and True=1, but selecting
// it does not identify or execute a pixels-from-memory trigger:
// https://github.com/NVIDIA/open-gpu-doc/blob/9e6d83fe0770bc8644850a0b1bf5ddb1519905ba/classes/twod/cl902d.h#L934-L937
// SET_RENDER_ENABLE_C defines five modes in bits 2:0. SET_RENDER_ENABLE_A/B
// provide address state for modes that consume it, but C is independently
// programmable and publishes no reset value:
// https://github.com/NVIDIA/open-gpu-doc/blob/9e6d83fe0770bc8644850a0b1bf5ddb1519905ba/classes/twod/cl902d.h#L498-L510
// SET_NOTIFY_A holds only the 25-bit upper address fragment. SET_NOTIFY_B and
// NOTIFY remain separate methods, so accepting A neither constructs an address
// nor performs a notification:
// https://github.com/NVIDIA/open-gpu-doc/blob/9e6d83fe0770bc8644850a0b1bf5ddb1519905ba/classes/twod/cl902d.h#L234-L247
methods!(
    SET_NOTIFY_A => (
        0x0104,
        "SET_NOTIFY_A",
        0x01ff_ffff,
        MethodAction::NotifyAddressUpper
    ),
    SET_NOTIFY_B => (
        0x0108,
        "SET_NOTIFY_B",
        u32::MAX,
        MethodAction::NotifyAddressLower
    ),
    SET_NUM_PROCESSING_CLUSTERS => (
        0x0260,
        "SET_NUM_PROCESSING_CLUSTERS",
        0x0000_0001,
        MethodAction::ProcessingClusters
    ),
    SET_RENDER_ENABLE_C => (
        0x026c,
        "SET_RENDER_ENABLE_C",
        0x0000_0007,
        MethodAction::RenderEnableMode
    ),
    SET_CLIP_ENABLE => (
        0x0290,
        "SET_CLIP_ENABLE",
        0x0000_0001,
        MethodAction::ClipEnable
    ),
    SET_COLOR_KEY_ENABLE => (
        0x029c,
        "SET_COLOR_KEY_ENABLE",
        0x0000_0001,
        MethodAction::ColorKeyEnable
    ),
    SET_BETA1 => (
        0x02a4,
        "SET_BETA1",
        u32::MAX,
        MethodAction::Beta1
    ),
    SET_BETA4 => (
        0x02a8,
        "SET_BETA4",
        u32::MAX,
        MethodAction::Beta4
    ),
    SET_OPERATION => (
        0x02ac,
        "SET_OPERATION",
        0x0000_0007,
        MethodAction::Operation
    ),
    SET_PIXELS_FROM_MEMORY_CORRAL_SIZE => (
        0x0884,
        "SET_PIXELS_FROM_MEMORY_CORRAL_SIZE",
        0x0000_03ff,
        MethodAction::PixelsFromMemoryCorralSize
    ),
    SET_PIXELS_FROM_MEMORY_SAFE_OVERLAP => (
        0x0888,
        "SET_PIXELS_FROM_MEMORY_SAFE_OVERLAP",
        0x0000_0001,
        MethodAction::PixelsFromMemorySafeOverlap
    ),
);

pub(super) fn preflight(
    method: MaxwellMethodDispatch,
    candidate: &mut MaxwellTwoDState,
) -> Result<MaxwellEngineMethodDispatch, MaxwellEngineDispatchError> {
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

    let write = match declaration.action {
        MethodAction::ProcessingClusters => MaxwellTwoDStateWrite::ProcessingClusters {
            value: MaxwellTwoDProcessingClusters::parse(source.argument())
                .ok_or_else(|| invalid_value(source, declaration))?,
            source,
        },
        MethodAction::Operation => MaxwellTwoDStateWrite::Operation {
            value: MaxwellTwoDOperation::parse(source.argument())
                .ok_or_else(|| invalid_value(source, declaration))?,
            source,
        },
        MethodAction::ClipEnable => MaxwellTwoDStateWrite::ClipEnable {
            value: MaxwellTwoDClipEnable::parse(source.argument())
                .ok_or_else(|| invalid_value(source, declaration))?,
            source,
        },
        MethodAction::ColorKeyEnable => MaxwellTwoDStateWrite::ColorKeyEnable {
            value: MaxwellTwoDColorKeyEnable::parse(source.argument())
                .ok_or_else(|| invalid_value(source, declaration))?,
            source,
        },
        MethodAction::Beta1 => MaxwellTwoDStateWrite::Beta(MaxwellTwoDBetaStateWrite::Beta1 {
            value: MaxwellTwoDBeta1::new(source.argument()),
            source,
        }),
        MethodAction::Beta4 => MaxwellTwoDStateWrite::Beta(MaxwellTwoDBetaStateWrite::Beta4 {
            value: MaxwellTwoDBeta4::from_raw(source.argument()),
            source,
        }),
        MethodAction::PixelsFromMemoryCorralSize => MaxwellTwoDStateWrite::PixelsFromMemory(
            MaxwellTwoDPixelsFromMemoryStateWrite::CorralSize {
                value: MaxwellTwoDPixelsFromMemoryCorralSize::parse(source.argument())
                    .ok_or_else(|| invalid_value(source, declaration))?,
                source,
            },
        ),
        MethodAction::PixelsFromMemorySafeOverlap => MaxwellTwoDStateWrite::PixelsFromMemory(
            MaxwellTwoDPixelsFromMemoryStateWrite::SafeOverlap {
                value: MaxwellTwoDPixelsFromMemorySafeOverlap::parse(source.argument())
                    .ok_or_else(|| invalid_value(source, declaration))?,
                source,
            },
        ),
        MethodAction::RenderEnableMode => {
            MaxwellTwoDStateWrite::RenderEnable(MaxwellTwoDRenderEnableStateWrite::Mode {
                value: MaxwellTwoDRenderEnableMode::parse(source.argument())
                    .ok_or_else(|| invalid_value(source, declaration))?,
                source,
            })
        }
        MethodAction::NotifyAddressUpper => {
            MaxwellTwoDStateWrite::Notify(MaxwellTwoDNotifyStateWrite::AddressUpper {
                value: MaxwellTwoDNotifyAddressUpper::parse(source.argument())
                    .ok_or_else(|| invalid_value(source, declaration))?,
                source,
            })
        }
        MethodAction::NotifyAddressLower => {
            MaxwellTwoDStateWrite::Notify(MaxwellTwoDNotifyStateWrite::AddressLower {
                value: MaxwellTwoDNotifyAddressLower::new(source.argument()),
                source,
            })
        }
    };

    candidate.apply(write);
    Ok(MaxwellEngineMethodDispatch::new(
        method,
        *declaration.metadata,
        MaxwellEngineMethodEffect::TwoDState(write),
    ))
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
