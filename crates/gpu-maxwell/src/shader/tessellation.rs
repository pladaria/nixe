//! SPH and direct patch operations. ISBE addressing is a separate SASS boundary:
//! an ALD vertex operand is not automatically a neutral control-point index.
//!
//! Public producer ABI and instruction fields:
//! https://github.com/devkitPro/uam/blob/master/source/compiler_iface.cpp#L476-L487
//! https://github.com/devkitPro/uam/blob/master/source/nv_attributes.h
//! https://github.com/devkitPro/uam/blob/master/mesa-imported/codegen/nv50_ir_emit_gm107.cpp#L2487-L2506
use super::binary::MaxwellShaderProgramHeader;
use super::decode::{allocate_shader_temporary, validate_register_range};
use super::error::{MaxwellShaderTranslationError, malformed};
use super::interface::{attribute_location, interface_element};
use crate::MaxwellShaderStage;
use nixe_gpu::{
    ShaderInstruction, ShaderInterfaceElement, ShaderIoLocation, ShaderOperation, ShaderPredicate,
    ShaderRegister, ShaderScalarType,
};

#[cfg(test)]
mod patch_order_tests;

pub(super) fn header_inputs(
    header: MaxwellShaderProgramHeader,
    inputs: &mut Vec<ShaderInterfaceElement>,
) {
    if !matches!(
        header.stage,
        MaxwellShaderStage::TessellationInit | MaxwellShaderStage::Tessellation
    ) {
        return;
    }
    if header.bit(184) {
        declare_input(inputs, ShaderIoLocation::PrimitiveId);
    }
    if header.bit(187) {
        inputs.push(interface_element(ShaderIoLocation::PointSize, 0, None));
    }
    if header.stage == MaxwellShaderStage::Tessellation {
        for index in 0..6 {
            if header.bit(164 + index) {
                let (location, component) = patch_location((index * 4) as u16).unwrap();
                inputs.push(interface_element(location, component, None));
            }
        }
    }
}

pub(super) fn header_outputs(
    header: MaxwellShaderProgramHeader,
    outputs: &mut Vec<ShaderInterfaceElement>,
) {
    if header.stage == MaxwellShaderStage::TessellationInit {
        // SPH allocates scalar slots, not vec4s. Slots 6/7 are padding before
        // user patch data, while 0..5 hold the six tessellator levels.
        for slot in 0..header.bits(56, 8) {
            if let Some((location, component)) = patch_location((slot * 4) as u16) {
                outputs.push(interface_element(location, component, None));
            }
        }
    }
}

pub(super) fn patch_location(address: u16) -> Option<(ShaderIoLocation, u8)> {
    if !address.is_multiple_of(4) {
        return None;
    }
    match address {
        0..=12 => Some((ShaderIoLocation::TessLevelOuter, (address / 4) as u8)),
        16..=20 => Some((ShaderIoLocation::TessLevelInner, ((address - 16) / 4) as u8)),
        32..=1020 => Some((
            ShaderIoLocation::Patch(((address - 32) / 16) as u8),
            ((address - 32) % 16 / 4) as u8,
        )),
        _ => None,
    }
}

/// Preserve Maxwell's warp-synchronous patch-output accesses on hosts that do
/// not execute the whole patch in lockstep. UAM removes TCS BAR.SYNC entirely:
/// https://github.com/devkitPro/uam/blob/master/mesa-imported/codegen/nv50_ir_lowering_nvc0.cpp#L831-L835
/// RAW, WAR and WAW conflicts need a rendezvous; independent slots and repeated
/// reads do not. One barrier orders all earlier accesses. This runs only during
/// shader translation, with constant-time scalar-slot membership tests.
pub(super) fn order_patch_outputs(instructions: &mut Vec<ShaderInstruction>) {
    let mut reads = [0_u64; 4];
    let mut writes = [0_u64; 4];
    let mut barriers = Vec::new();
    for (index, instruction) in instructions.iter().enumerate() {
        let (location, component, count, write) = match instruction.operation() {
            ShaderOperation::LoadPatchOutput {
                location,
                component,
                ..
            } => (*location, *component, 1, false),
            ShaderOperation::StoreOutput {
                location,
                first_component,
                sources,
                ..
            } => (*location, *first_component, sources.len(), true),
            _ => continue,
        };
        let base = match location {
            ShaderIoLocation::TessLevelOuter => 0,
            ShaderIoLocation::TessLevelInner => 4,
            ShaderIoLocation::Patch(n) => 8 + usize::from(n) * 4,
            _ => continue,
        } + usize::from(component);
        let conflict = (base..base + count).any(|slot| {
            let mask = 1 << (slot % 64);
            writes[slot / 64] & mask != 0 || (write && reads[slot / 64] & mask != 0)
        });
        if conflict {
            barriers.push(index);
            reads.fill(0);
            writes.fill(0);
        }
        let accessed = if write { &mut writes } else { &mut reads };
        for slot in base..base + count {
            accessed[slot / 64] |= 1 << (slot % 64);
        }
    }
    if barriers.is_empty() {
        return;
    }
    let original = std::mem::take(instructions);
    instructions.reserve(original.len() + barriers.len());
    let mut barriers = barriers.into_iter().peekable();
    for (index, instruction) in original.into_iter().enumerate() {
        if barriers.peek() == Some(&index) {
            barriers.next();
            // Predicated stores still need every invocation to rendezvous.
            // The neutral verifier rejects branches/early exits for which
            // reaching this point uniformly has not been proven.
            instructions.push(ShaderInstruction::new(
                instruction.source(),
                ShaderPredicate::Always,
                ShaderOperation::PatchBarrier,
            ));
        }
        instructions.push(instruction);
    }
}

fn declare_input(inputs: &mut Vec<ShaderInterfaceElement>, location: ShaderIoLocation) {
    if !inputs.iter().any(|input| input.location() == location) {
        inputs.push(
            ShaderInterfaceElement::new(location, 0, ShaderScalarType::Unsigned32, None).unwrap(),
        );
    }
}

pub(super) const fn is_system_register_read(encoding: u64) -> bool {
    encoding >> 48 == 0xf0c8
}

pub(super) fn decode_system_register(
    stage: MaxwellShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
    inputs: &mut Vec<ShaderInterfaceElement>,
) -> Result<ShaderOperation, MaxwellShaderTranslationError> {
    if encoding & !0xffff_0000_0fff_00ff != 0 {
        return Err(malformed(
            stage,
            offset,
            encoding,
            "S2R reserved field is nonzero",
        ));
    }
    // S2R selector: public GM107 emitSYS/emitS2R, not a shader fingerprint.
    // https://github.com/devkitPro/uam/blob/master/mesa-imported/codegen/nv50_ir_emit_gm107.cpp#L267-L274
    let location = match ((encoding >> 20) & 0xff, stage) {
        (0x11, MaxwellShaderStage::TessellationInit) => ShaderIoLocation::InvocationId,
        (0x10, MaxwellShaderStage::TessellationInit | MaxwellShaderStage::Tessellation) => {
            ShaderIoLocation::PatchVertices
        }
        _ => {
            return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
                stage,
                instruction_offset: offset,
                encoding,
                detail: "S2R system value requires unsupported stage or invocation/warp ABI semantics",
            });
        }
    };
    let destination = encoding as u8;
    validate_register_range(stage, offset, encoding, destination, 1, register_count)?;
    declare_input(inputs, location);
    Ok(ShaderOperation::LoadInput {
        destinations: vec![ShaderRegister::new(u16::from(destination))].into_boxed_slice(),
        location,
        first_component: 0,
        scalar_type: ShaderScalarType::Unsigned32,
    })
}

pub(super) fn decode_control_store(
    stage: MaxwellShaderStage,
    offset: u32,
    encoding: u64,
    register_count: u8,
    next_temporary: &mut u16,
    inputs: &mut Vec<ShaderInterfaceElement>,
) -> Result<Vec<ShaderOperation>, MaxwellShaderTranslationError> {
    let source = encoding as u8;
    let count = (((encoding >> 47) & 3) + 1) as u8;
    validate_register_range(stage, offset, encoding, source, count, register_count)?;
    if ((encoding >> 8) & 0xff) != 0xff || ((encoding >> 39) & 0xff) != 0xff {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage,
            instruction_offset: offset,
            encoding,
            detail: "indexed AST requires attribute/ISBE address lowering",
        });
    }
    let first_address = ((encoding >> 20) & 0x3ff) as u16;
    let patch = encoding & (1 << 31) != 0;
    let mut operations = Vec::new();
    let vertex = if patch {
        None
    } else {
        let vertex = allocate_shader_temporary(
            stage,
            offset,
            encoding,
            "control-point index temporary exceeds register bank",
            next_temporary,
        )?;
        declare_input(inputs, ShaderIoLocation::InvocationId);
        operations.push(ShaderOperation::LoadInput {
            destinations: vec![vertex].into_boxed_slice(),
            location: ShaderIoLocation::InvocationId,
            first_component: 0,
            scalar_type: ShaderScalarType::Unsigned32,
        });
        Some(vertex)
    };
    for lane in 0..count {
        let address = first_address + u16::from(lane) * 4;
        let (location, component) = if patch {
            patch_location(address).ok_or_else(|| {
                malformed(
                    stage,
                    offset,
                    encoding,
                    "AST.P reserved or misaligned patch attribute",
                )
            })?
        } else {
            attribute_location(stage, offset, encoding, address)?
        };
        let register = ShaderRegister::new(u16::from(source + lane));
        operations.push(if let Some(vertex) = vertex {
            ShaderOperation::StoreControlPoint {
                source: register,
                vertex,
                location,
                component,
            }
        } else {
            ShaderOperation::StoreOutput {
                sources: vec![register].into_boxed_slice(),
                location,
                first_component: component,
                scalar_type: ShaderScalarType::Float32,
            }
        });
    }
    Ok(operations)
}

#[cfg(test)]
mod tests;
