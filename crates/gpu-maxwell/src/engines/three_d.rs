//! GM20B `MAXWELL_B` 3D class methods reached at the T7 boundary.

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

// Class and method values are pinned to NVIDIA's generated public MAXWELL_B
// header. The declarative list stays deliberately small: a method is added
// only when Nixe can name its next semantic boundary without guessing.
// https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/clb197.h
methods!(
    NO_OPERATION => (0x0100, "NO_OPERATION", u32::MAX, MethodAction::NoOperation),
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
        MethodAction::Missing(MaxwellEngineCapability::HostBackend)
    ),
);

pub(super) fn preflight(
    method: MaxwellMethodDispatch,
) -> Result<MaxwellEngineMethodDispatch, MaxwellEngineDispatchError> {
    let source = method.source();
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
    match declaration.action {
        MethodAction::NoOperation => Ok(MaxwellEngineMethodDispatch::new(
            method,
            *declaration.metadata,
            MaxwellEngineMethodEffect::NoOperation,
        )),
        MethodAction::Unsupported => Err(MaxwellEngineDispatchError::UnsupportedMethod {
            source,
            metadata: declaration.metadata,
        }),
        MethodAction::Missing(capability) => Err(MaxwellEngineDispatchError::MissingCapability {
            source,
            metadata: declaration.metadata,
            capability,
        }),
    }
}
