//! Shared graphics and compute SASS translation and instruction-stream state.

use super::binary::{
    MAXWELL_INSTRUCTION_SIZE, MAXWELL_SCHEDULE_BUNDLE_SIZE, MAXWELL_SCHEDULE_CONTROL_SIZE,
    MaxwellShaderBinary, MaxwellShaderMetadata,
};
use super::control_flow::{
    decode_shader_control_target, is_branch, is_exit, is_set_sync_point, is_synchronize,
};
use super::conversion::{
    decode_float_to_float, decode_float_to_integer, decode_integer_to_float, is_float_to_float,
    is_float_to_integer, is_integer_to_float,
};
use super::data::{
    decode_constant_buffer_load, decode_move, decode_move_immediate, is_constant_buffer_load,
    is_move, is_move_immediate,
};
use super::decode::decode_predicate;
use super::error::{MaxwellShaderTranslationError, malformed, unsupported_instruction};
use super::float::{
    decode_float_add, decode_float_fused_multiply_add, decode_float_min_max, decode_float_multiply,
    decode_float_set_predicate, is_float_add, is_float_fused_multiply_add, is_float_min_max,
    is_float_multiply, is_float_set_predicate,
};
use super::integer::{decode_shift_left, is_shift_left};
use super::interface::{
    append_implicit_outputs, decode_attribute_load, decode_attribute_store, decode_header_inputs,
    decode_header_outputs, decode_interpolate, is_attribute_load, is_attribute_store,
    is_interpolate, neutral_stage, preload_vertex_inputs,
};
use super::special::{
    decode_mufu, decode_range_reduced_mufu, decode_range_reduction, is_compatible_mufu, is_mufu,
    is_range_reduction,
};
use super::texture::{
    MaxwellTextureResourceBinding, decode_texture_access_simplified, is_texture_access_simplified,
};
use super::{compute, global_memory, integer, patch_address, tessellation};
use crate::MaxwellShaderStage;
use nixe_gpu::{
    ShaderInstruction, ShaderInterfaceElement, ShaderIoLocation, ShaderIr, ShaderOperation,
    ShaderPredicate, ShaderResourceAccess, ShaderResourceKind, ShaderScalarType,
    ShaderSourceLocation,
};
use std::collections::{BTreeMap, BTreeSet};

pub(super) struct TranslatedShaderIr {
    pub(super) ir: ShaderIr,
    pub(super) texture_bindings: Box<[MaxwellTextureResourceBinding]>,
    pub(super) global_buffers: global_memory::GlobalBufferBindings,
}

/// Translate one immutable Maxwell program snapshot using shared SM50 semantics.
///
/// Attribute addresses follow Mesa NAK's pinned public Maxwell ABI constants:
/// https://gitlab.freedesktop.org/mesa/mesa/-/blob/2c9073912232b93eb9b60486edbd72d53e5f3d26/src/nouveau/compiler/nak_private.h#L45-L57
///
/// The implemented EXIT, BRA, SSY, SYNC, ALD, AST, MOV32I, IPA, RRO/MUFU, FMUL,
/// FFMA, FADD, and FSETP encodings are derived from Mesa NAK's pinned SM50 encoder and
/// opcode tables, rather than from the captured shader binaries:
/// https://gitlab.freedesktop.org/mesa/mesa/-/blob/2c9073912232b93eb9b60486edbd72d53e5f3d26/src/nouveau/compiler/nak/sm50.rs
pub(super) fn translate_shader_binary(
    binary: &MaxwellShaderBinary,
    register_count: u8,
    vertex_input_types: &BTreeMap<ShaderIoLocation, ShaderScalarType>,
) -> Result<TranslatedShaderIr, MaxwellShaderTranslationError> {
    let stage = binary.stage();
    let neutral_stage = neutral_stage(stage);
    let (mut inputs, outputs) = match binary.metadata {
        MaxwellShaderMetadata::Graphics(header) => (
            decode_header_inputs(header, vertex_input_types)?,
            decode_header_outputs(header)?,
        ),
        MaxwellShaderMetadata::Compute { .. } => (Vec::new(), Vec::new()),
    };
    let mut instructions = preload_vertex_inputs(neutral_stage, &inputs);
    let mut constant_buffer_bindings = BTreeSet::new();
    let mut texture_bindings = BTreeMap::new();
    let mut next_temporary = u16::from(register_count);
    let mut explicitly_stored = BTreeSet::new();
    let mut active_reconvergence_targets = Vec::new();
    let mut pending_range_reduction = None;
    let mut exited = false;
    let mut patch_addresses = patch_address::PatchAddresses::default();
    let mut integer_carry = None;
    let mut global_memory =
        (stage == MaxwellShaderStage::Compute).then(global_memory::GlobalMemory::default);
    let code_size = u32::try_from(binary.bundles().len() * MAXWELL_SCHEDULE_BUNDLE_SIZE)
        .expect("bounded Maxwell shader code size fits u32");

    'bundles: for bundle in binary.bundles() {
        for (slot, encoding) in bundle.instructions.iter().copied().enumerate() {
            let offset = bundle.offset
                + MAXWELL_SCHEDULE_CONTROL_SIZE as u32
                + (slot * MAXWELL_INSTRUCTION_SIZE) as u32;
            let source = ShaderSourceLocation::new(offset);
            let predicate = decode_predicate(encoding);
            let translated_start = instructions.len();
            if let Some(memory) = &mut global_memory
                && (is_branch(encoding) || is_set_sync_point(encoding) || is_synchronize(encoding))
            {
                memory.observe_control_flow(offset, encoding)?;
            }

            if let Some(range_reduction) = pending_range_reduction.as_ref()
                && !is_compatible_mufu(range_reduction, encoding, predicate)
            {
                return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
                    stage,
                    instruction_offset: range_reduction.offset,
                    encoding: range_reduction.encoding,
                    detail: "RRO result is not consumed by an adjacent compatible MUFU",
                });
            }

            // A reconvergence target describes a structured control-flow
            // region, not a one-shot SSY/SYNC pair. Several mutually
            // exclusive paths may end in SYNC instructions which all branch
            // to the same target. Retire the region only when the linear
            // translation reaches its reconvergence point.
            active_reconvergence_targets
                .retain(|target: &ShaderSourceLocation| target.byte_offset() > offset);

            if is_exit(encoding) {
                if matches!(
                    stage,
                    MaxwellShaderStage::TessellationInit
                        | MaxwellShaderStage::Tessellation
                        | MaxwellShaderStage::Compute
                ) && predicate != ShaderPredicate::Always
                {
                    return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
                        stage,
                        instruction_offset: offset,
                        encoding,
                        detail: "conditional EXIT requires complete shader control-flow discovery",
                    });
                }
                append_implicit_outputs(
                    neutral_stage,
                    source,
                    &outputs,
                    &explicitly_stored,
                    &mut instructions,
                )?;
                instructions.push(ShaderInstruction::new(
                    source,
                    predicate,
                    ShaderOperation::Exit,
                ));
                patch_addresses.ordinary(stage, encoding, &instructions[translated_start..])?;
                exited = true;
                break 'bundles;
            }

            if is_set_sync_point(encoding) {
                // SSY is warp reconvergence metadata, not a per-invocation
                // operation. Keep its normalized target active for the whole
                // region so every path-local SYNC can reference it.
                active_reconvergence_targets.push(decode_shader_control_target(
                    stage, offset, encoding, code_size,
                )?);
                continue;
            }

            if is_synchronize(encoding) {
                let target = active_reconvergence_targets
                    .last()
                    .copied()
                    .ok_or_else(|| {
                        malformed(
                            stage,
                            offset,
                            encoding,
                            "SYNC has no matching SSY reconvergence target",
                        )
                    })?;
                if predicate != ShaderPredicate::Never {
                    instructions.push(ShaderInstruction::new(
                        source,
                        predicate,
                        ShaderOperation::Branch { target },
                    ));
                }
                patch_addresses.ordinary(stage, encoding, &instructions[translated_start..])?;
                continue;
            }

            if predicate == ShaderPredicate::Never {
                // Predicated-false instructions have no architectural data,
                // interface, or control-flow effect. The encoding family is
                // still classified so random data cannot hide as dead code.
                if !is_supported_family(encoding) {
                    return Err(unsupported_instruction(binary, offset, encoding));
                }
                continue;
            }

            if is_range_reduction(encoding) {
                pending_range_reduction = Some(decode_range_reduction(
                    stage,
                    offset,
                    encoding,
                    predicate,
                    register_count,
                    &mut next_temporary,
                )?);
                continue;
            }

            if let Some(operations) = patch_addresses.lower(
                stage,
                offset,
                encoding,
                register_count,
                &mut next_temporary,
                &mut inputs,
            )? {
                instructions.extend(
                    operations
                        .into_iter()
                        .map(|operation| ShaderInstruction::new(source, predicate, operation)),
                );
                continue;
            }
            'instruction: {
                let operation = if global_memory::is_store(encoding) {
                    let memory = global_memory
                        .as_mut()
                        .ok_or_else(|| unsupported_instruction(binary, offset, encoding))?;
                    let operations =
                        memory.store(offset, encoding, register_count, &mut next_temporary)?;
                    append_expanded_operations(&mut instructions, source, predicate, operations);
                    break 'instruction;
                } else if is_branch(encoding) {
                    ShaderOperation::Branch {
                        target: decode_shader_control_target(stage, offset, encoding, code_size)?,
                    }
                } else if tessellation::is_system_register_read(encoding) {
                    if stage == MaxwellShaderStage::Compute {
                        compute::system_register(offset, encoding, register_count)?
                    } else {
                        tessellation::decode_system_register(
                            stage,
                            offset,
                            encoding,
                            register_count,
                            &mut inputs,
                        )?
                    }
                } else if is_attribute_load(encoding) {
                    let operations = decode_attribute_load(
                        stage,
                        offset,
                        encoding,
                        register_count,
                        vertex_input_types,
                    )?;
                    for operation in &operations {
                        if let ShaderOperation::LoadInput {
                            location:
                                location @ (ShaderIoLocation::VertexId | ShaderIoLocation::InstanceId),
                            scalar_type,
                            ..
                        } = operation
                            && !inputs.iter().any(|input| input.location() == *location)
                        {
                            inputs.push(
                                ShaderInterfaceElement::new(*location, 0, *scalar_type, None)
                                    .expect("Maxwell vertex system values are scalar inputs"),
                            );
                        }
                    }
                    instructions.extend(
                        operations
                            .into_iter()
                            .map(|operation| ShaderInstruction::new(source, predicate, operation)),
                    );
                    break 'instruction;
                } else if is_attribute_store(encoding) {
                    if stage == MaxwellShaderStage::TessellationInit {
                        let operations = tessellation::decode_control_store(
                            stage,
                            offset,
                            encoding,
                            register_count,
                            &mut next_temporary,
                            &mut inputs,
                        )?;
                        instructions.extend(operations.into_iter().map(|operation| {
                            let predicate =
                                if matches!(operation, ShaderOperation::LoadInput { .. }) {
                                    ShaderPredicate::Always
                                } else {
                                    predicate
                                };
                            ShaderInstruction::new(source, predicate, operation)
                        }));
                        break 'instruction;
                    }
                    let operation =
                        decode_attribute_store(stage, offset, encoding, register_count)?;
                    if let ShaderOperation::StoreOutput {
                        location,
                        first_component,
                        sources,
                        ..
                    } = &operation
                    {
                        for component in 0..sources.len() {
                            explicitly_stored.insert((
                                *location,
                                first_component.saturating_add(component as u8),
                            ));
                        }
                    }
                    operation
                } else if is_move_immediate(encoding) {
                    decode_move_immediate(stage, offset, encoding, register_count)?
                } else if is_move(encoding) {
                    let decoded =
                        decode_move(stage, offset, encoding, register_count, &mut next_temporary)?;
                    if let Some(binding) = decoded.constant_buffer_binding {
                        constant_buffer_bindings.insert(binding);
                    }
                    append_expanded_operations(
                        &mut instructions,
                        source,
                        predicate,
                        decoded.operations,
                    );
                    break 'instruction;
                } else if is_shift_left(encoding) {
                    let decoded = decode_shift_left(
                        stage,
                        offset,
                        encoding,
                        register_count,
                        &mut next_temporary,
                    )?;
                    if let Some(binding) = decoded.constant_buffer_binding {
                        constant_buffer_bindings.insert(binding);
                    }
                    append_expanded_operations(
                        &mut instructions,
                        source,
                        predicate,
                        decoded.operations,
                    );
                    break 'instruction;
                } else if is_integer_to_float(encoding) {
                    let decoded = decode_integer_to_float(
                        stage,
                        offset,
                        encoding,
                        register_count,
                        &mut next_temporary,
                    )?;
                    if let Some(binding) = decoded.constant_buffer_binding {
                        constant_buffer_bindings.insert(binding);
                    }
                    append_expanded_operations(
                        &mut instructions,
                        source,
                        predicate,
                        decoded.operations,
                    );
                    break 'instruction;
                } else if is_float_to_float(encoding) {
                    let decoded = decode_float_to_float(
                        stage,
                        offset,
                        encoding,
                        register_count,
                        &mut next_temporary,
                    )?;
                    if let Some(binding) = decoded.constant_buffer_binding {
                        constant_buffer_bindings.insert(binding);
                    }
                    append_expanded_operations(
                        &mut instructions,
                        source,
                        predicate,
                        decoded.operations,
                    );
                    break 'instruction;
                } else if is_float_to_integer(encoding) {
                    let decoded = decode_float_to_integer(
                        stage,
                        offset,
                        encoding,
                        register_count,
                        &mut next_temporary,
                    )?;
                    if let Some(binding) = decoded.constant_buffer_binding {
                        constant_buffer_bindings.insert(binding);
                    }
                    append_expanded_operations(
                        &mut instructions,
                        source,
                        predicate,
                        decoded.operations,
                    );
                    break 'instruction;
                } else if is_constant_buffer_load(encoding) {
                    let decoded =
                        decode_constant_buffer_load(stage, offset, encoding, register_count)?;
                    constant_buffer_bindings.insert(decoded.constant_buffer_binding);
                    instructions.push(ShaderInstruction::new(source, predicate, decoded.operation));
                    break 'instruction;
                } else if is_texture_access_simplified(encoding) {
                    if stage == MaxwellShaderStage::Compute {
                        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
                            stage,
                            instruction_offset: offset,
                            encoding,
                            detail: "compute texture binding ABI is not implemented",
                        });
                    }
                    decode_texture_access_simplified(
                        stage,
                        offset,
                        encoding,
                        register_count,
                        &mut texture_bindings,
                    )?
                } else if is_interpolate(encoding) {
                    decode_interpolate(stage, offset, encoding, register_count, &inputs)?
                } else if is_mufu(encoding) {
                    if let Some(range_reduction) = pending_range_reduction.take() {
                        if let Some(binding) = range_reduction.constant_buffer_binding {
                            constant_buffer_bindings.insert(binding);
                        }
                        let operation = decode_range_reduced_mufu(
                            stage,
                            offset,
                            encoding,
                            register_count,
                            &range_reduction,
                        )?;
                        append_expanded_operations(
                            &mut instructions,
                            range_reduction.source,
                            range_reduction.predicate,
                            range_reduction.preparation,
                        );
                        instructions.push(ShaderInstruction::new(source, predicate, operation));
                        break 'instruction;
                    }
                    let operations =
                        decode_mufu(stage, offset, encoding, register_count, &mut next_temporary)?;
                    append_expanded_operations(&mut instructions, source, predicate, operations);
                    break 'instruction;
                } else if is_float_min_max(encoding) {
                    let decoded = decode_float_min_max(
                        stage,
                        offset,
                        encoding,
                        register_count,
                        &mut next_temporary,
                    )?;
                    if let Some(binding) = decoded.constant_buffer_binding {
                        constant_buffer_bindings.insert(binding);
                    }
                    append_expanded_operations(
                        &mut instructions,
                        source,
                        predicate,
                        decoded.operations,
                    );
                    break 'instruction;
                } else if is_float_multiply(encoding) {
                    let decoded = decode_float_multiply(
                        stage,
                        offset,
                        encoding,
                        register_count,
                        &mut next_temporary,
                    )?;
                    if let Some(binding) = decoded.constant_buffer_binding {
                        constant_buffer_bindings.insert(binding);
                    }
                    append_expanded_operations(
                        &mut instructions,
                        source,
                        predicate,
                        decoded.operations,
                    );
                    break 'instruction;
                } else if is_float_fused_multiply_add(encoding) {
                    let decoded = decode_float_fused_multiply_add(
                        stage,
                        offset,
                        encoding,
                        register_count,
                        &mut next_temporary,
                    )?;
                    if let Some(binding) = decoded.constant_buffer_binding {
                        constant_buffer_bindings.insert(binding);
                    }
                    append_expanded_operations(
                        &mut instructions,
                        source,
                        predicate,
                        decoded.operations,
                    );
                    break 'instruction;
                } else if is_float_add(encoding) {
                    let decoded = decode_float_add(
                        stage,
                        offset,
                        encoding,
                        register_count,
                        &mut next_temporary,
                    )?;
                    if let Some(binding) = decoded.constant_buffer_binding {
                        constant_buffer_bindings.insert(binding);
                    }
                    append_expanded_operations(
                        &mut instructions,
                        source,
                        predicate,
                        decoded.operations,
                    );
                    break 'instruction;
                } else if integer::is_bitwise(encoding)
                    || integer::is_add(encoding)
                    || integer::is_shift_add(encoding)
                    || integer::is_set_predicate(encoding)
                {
                    let decoded = if integer::is_add(encoding) || integer::is_shift_add(encoding) {
                        integer::decode_add(
                            stage,
                            offset,
                            encoding,
                            register_count,
                            &mut next_temporary,
                            &mut integer_carry,
                        )?
                    } else {
                        let decode = if integer::is_bitwise(encoding) {
                            integer::decode_bitwise
                        } else {
                            integer::decode_set_predicate
                        };
                        decode(stage, offset, encoding, register_count, &mut next_temporary)?
                    };
                    if let Some(binding) = decoded.constant_buffer_binding {
                        constant_buffer_bindings.insert(binding);
                    }
                    // Every source preparation inherits the predicate as well: a
                    // source may itself only be defined under that condition.
                    instructions.extend(
                        decoded
                            .operations
                            .into_iter()
                            .map(|operation| ShaderInstruction::new(source, predicate, operation)),
                    );
                    break 'instruction;
                } else if is_float_set_predicate(encoding) {
                    let decoded = decode_float_set_predicate(
                        stage,
                        offset,
                        encoding,
                        register_count,
                        &mut next_temporary,
                    )?;
                    if let Some(binding) = decoded.constant_buffer_binding {
                        constant_buffer_bindings.insert(binding);
                    }
                    append_expanded_operations(
                        &mut instructions,
                        source,
                        predicate,
                        decoded.operations,
                    );
                    break 'instruction;
                } else {
                    return Err(unsupported_instruction(binary, offset, encoding));
                };
                instructions.push(ShaderInstruction::new(source, predicate, operation));
            }
            patch_addresses.ordinary(stage, encoding, &instructions[translated_start..])?;
            if let Some(memory) = &mut global_memory {
                memory.observe(
                    &mut instructions,
                    translated_start,
                    encoding,
                    &mut next_temporary,
                )?;
            }
        }
    }
    debug_assert!(
        exited,
        "bounded reader only returns programs containing EXIT"
    );

    let mut resources = constant_buffer_bindings
        .into_iter()
        .map(|binding| {
            ShaderResourceAccess::new(binding, ShaderResourceKind::ConstantBuffer, true, false)
                .expect("read-only constant-buffer access is valid")
        })
        .collect::<Vec<_>>();
    let global_buffers = global_memory
        .map(|memory| memory.bindings)
        .unwrap_or_default();
    for buffer in &global_buffers.buffers {
        resources.push(
            ShaderResourceAccess::new(
                buffer.binding,
                ShaderResourceKind::StorageBuffer,
                false,
                true,
            )
            .expect("storage buffer writes are valid"),
        );
    }
    for binding in texture_bindings.values().copied() {
        resources.push(
            ShaderResourceAccess::new(binding.image_binding, binding.image_kind, true, false)
                .expect("read-only sampled-image access is valid"),
        );
        if let Some(sampler_binding) = binding.sampler_binding {
            resources.push(
                ShaderResourceAccess::new(
                    sampler_binding,
                    ShaderResourceKind::Sampler,
                    true,
                    false,
                )
                .expect("read-only sampler access is valid"),
            );
        }
    }
    if stage == MaxwellShaderStage::TessellationInit {
        tessellation::order_patch_outputs(&mut instructions);
    }
    let mut ir = ShaderIr::new(neutral_stage, inputs, outputs, resources, instructions)
        .with_tessellation_control_points(
            (stage == MaxwellShaderStage::TessellationInit)
                .then(|| binary.header().bits(88, 8) as u32),
        );
    if let MaxwellShaderMetadata::Compute { workgroup_size, .. } = binary.metadata {
        ir = ir.with_workgroup_size(workgroup_size);
    }
    Ok(TranslatedShaderIr {
        ir,
        texture_bindings: texture_bindings.values().copied().collect(),
        global_buffers,
    })
}

fn append_expanded_operations(
    instructions: &mut Vec<ShaderInstruction>,
    source: ShaderSourceLocation,
    predicate: ShaderPredicate,
    operations: Vec<ShaderOperation>,
) {
    let last = operations.len().saturating_sub(1);
    instructions.extend(
        operations
            .into_iter()
            .enumerate()
            .map(|(index, operation)| {
                ShaderInstruction::new(
                    source,
                    if index == last {
                        predicate
                    } else {
                        ShaderPredicate::Always
                    },
                    operation,
                )
            }),
    );
}

const fn is_supported_family(encoding: u64) -> bool {
    is_exit(encoding)
        || global_memory::is_store(encoding)
        || tessellation::is_system_register_read(encoding)
        || is_branch(encoding)
        || is_set_sync_point(encoding)
        || is_synchronize(encoding)
        || is_attribute_load(encoding)
        || is_attribute_store(encoding)
        || is_move_immediate(encoding)
        || is_move(encoding)
        || is_shift_left(encoding)
        || is_integer_to_float(encoding)
        || is_float_to_float(encoding)
        || is_float_to_integer(encoding)
        || is_constant_buffer_load(encoding)
        || is_texture_access_simplified(encoding)
        || is_interpolate(encoding)
        || is_range_reduction(encoding)
        || is_mufu(encoding)
        || is_float_min_max(encoding)
        || is_float_multiply(encoding)
        || is_float_fused_multiply_add(encoding)
        || is_float_add(encoding)
        || is_float_set_predicate(encoding)
        || integer::is_set_predicate(encoding)
        || integer::is_bitwise(encoding)
        || integer::is_add(encoding)
        || integer::is_shift_add(encoding)
        || patch_address::is_supported_family(encoding)
}

#[cfg(test)]
mod tests;
