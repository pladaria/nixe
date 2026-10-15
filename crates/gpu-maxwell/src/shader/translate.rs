//! Shared graphics and compute SASS translation and instruction-stream state.

use super::binary::{
    MAXWELL_INSTRUCTION_SIZE, MAXWELL_SCHEDULE_BUNDLE_SIZE, MAXWELL_SCHEDULE_CONTROL_SIZE,
    MaxwellShaderBinary, MaxwellShaderMetadata,
};
use super::control_flow::{
    decode_shader_control_target, discover_shader_control_flow, is_branch, is_dependency_barrier,
    is_exit, is_loop_branch, is_loop_push, is_set_sync_point, is_synchronize,
    validate_dependency_barrier,
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
use super::decoded::ResourceAccess as ShaderResourceAccess;
use super::error::{MaxwellShaderTranslationError, malformed, unsupported_instruction};
use super::float::{
    decode_float_add, decode_float_fused_multiply_add, decode_float_min_max, decode_float_multiply,
    decode_float_set, decode_float_set_predicate, is_float_add, is_float_fused_multiply_add,
    is_float_min_max, is_float_multiply, is_float_set, is_float_set_predicate,
};
use super::half::{
    decode_half_add, decode_half_fused_multiply_add, decode_half_multiply,
    decode_half_set_predicate, is_half_add, is_half_fused_multiply_add, is_half_multiply,
    is_half_set_predicate,
};
use super::integer::{decode_shift_left, decode_shift_right, is_shift_left, is_shift_right};
use super::interface::{
    append_implicit_outputs, decode_attribute_load, decode_attribute_store, decode_header_inputs,
    decode_header_outputs, decode_interpolate, is_attribute_load, is_attribute_store,
    is_interpolate, neutral_stage, oracle_preload_vertex_inputs,
};
use super::special::{
    decode_mufu, decode_range_reduced_mufu, decode_range_reduction, freeze_range_reduction_input,
    is_mufu, is_range_reduction,
};
use super::texture::{
    MaxwellTextureResourceBinding, decode_texture_access_simplified, decode_texture_query,
    decode_texture_sample, is_texture_access_simplified, is_texture_query, is_texture_sample,
};
use super::{compute, global_memory, integer, patch_address, quad, tessellation, xmad};
use crate::MaxwellShaderStage;
use crate::shader::decoded::{Instruction as ShaderInstruction, Operation as ShaderOperation};
use nixe_gpu::{
    ShaderInterfaceElement, ShaderIoLocation, ShaderIr, ShaderPredicate, ShaderResourceKind,
    ShaderScalarType, ShaderSourceLocation,
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
pub(super) fn decode_shader_binary(
    binary: &MaxwellShaderBinary,
    register_count: u8,
    vertex_input_types: &BTreeMap<ShaderIoLocation, ShaderScalarType>,
    y_negate: bool,
) -> Result<DecodedProgram, MaxwellShaderTranslationError> {
    let stage = binary.stage();
    let neutral_stage = neutral_stage(stage);
    // The reciprocal/IPA fusion below uses straight-line register definitions.
    // Branching programs retain the explicit arithmetic rather than assuming
    // a definition dominates every path into the interpolation instruction.
    let flow = discover_shader_control_flow(binary)?;
    let straight_line_fragment = stage == MaxwellShaderStage::Pixel
        && !binary.bundles().iter().any(|bundle| {
            bundle
                .instructions
                .iter()
                .enumerate()
                .any(|(slot, &encoding)| {
                    flow.reachable
                        .contains(&(bundle.offset + 8 + slot as u32 * 8))
                        && (is_branch(encoding)
                            || is_synchronize(encoding)
                            || encoding >> 48 == 0xe250
                            || is_loop_branch(encoding)
                            || is_loop_push(encoding)
                            || (is_exit(encoding)
                                && decode_predicate(encoding) != ShaderPredicate::Always))
                })
        });
    let (mut inputs, mut outputs) = match binary.metadata {
        MaxwellShaderMetadata::Graphics(header) => (
            decode_header_inputs(header, vertex_input_types)?,
            decode_header_outputs(header)?,
        ),
        MaxwellShaderMetadata::Compute { .. } => (Vec::new(), Vec::new()),
    };
    let mut instructions = Vec::new();
    let mut constant_buffer_bindings = BTreeSet::new();
    let mut texture_bindings = BTreeMap::new();
    let mut texture_queries = BTreeMap::new();
    let mut next_temporary = u32::from(register_count);
    let mut explicitly_stored = BTreeSet::new();
    if stage == MaxwellShaderStage::Vertex {
        // Maxwell's position export starts at (0, 0, 0, 1). A shader may
        // export XY alone; zero-initialized host W would clip every primitive.
        // Eden/yuzu's output prologue implements the same Maxwell contract:
        // https://github.com/eden-emulator/mirror/blob/67bada77f8a43a90da2e94e89b8e7da73c256989/src/shader_recompiler/backend/glsl/emit_glsl_special.cpp#L16-L22
        for component in 0..4 {
            if !outputs.iter().any(|element| {
                element.location() == ShaderIoLocation::Position && element.component() == component
            }) {
                outputs.push(
                    ShaderInterfaceElement::new(
                        ShaderIoLocation::Position,
                        component,
                        ShaderScalarType::Float32,
                        None,
                    )
                    .expect("valid position interface"),
                );
            }
            explicitly_stored.insert((ShaderIoLocation::Position, component));
        }
    }
    let stack_targets = &flow.stack_targets;
    let mut pending_range_reductions = BTreeMap::<u8, super::special::PendingRangeReduction>::new();
    let mut patch_addresses = patch_address::PatchAddresses::default();
    let mut integer_flags = integer::IntegerConditionCodes::default();
    let mut global_memory = global_memory::GlobalMemory::new(stage);
    let code_size = u32::try_from(binary.bundles().len() * MAXWELL_SCHEDULE_BUNDLE_SIZE)
        .expect("bounded Maxwell shader code size fits u32");

    let mut block_targets = BTreeSet::new();
    for bundle in binary.bundles() {
        for (slot, encoding) in bundle.instructions.iter().copied().enumerate() {
            if !flow
                .reachable
                .contains(&(bundle.offset + 8 + slot as u32 * 8))
            {
                continue;
            }
            if is_branch(encoding) || is_set_sync_point(encoding) || is_loop_push(encoding) {
                let offset = bundle.offset + 8 + slot as u32 * 8;
                block_targets.insert(
                    decode_shader_control_target(stage, offset, encoding, code_size)?.byte_offset(),
                );
            }
        }
    }

    block_targets.extend(
        binary
            .indirect_tables
            .iter()
            .flat_map(|table| table.targets.iter().map(|(_, target)| target.byte_offset())),
    );
    block_targets.extend(stack_targets.values().map(|target| target.byte_offset()));

    for bundle in binary.bundles() {
        for (slot, encoding) in bundle.instructions.iter().copied().enumerate() {
            let offset = bundle.offset
                + MAXWELL_SCHEDULE_CONTROL_SIZE as u32
                + (slot * MAXWELL_INSTRUCTION_SIZE) as u32;
            if !flow.reachable.contains(&offset) {
                continue;
            }
            let source = ShaderSourceLocation::new(offset);
            let predicate = decode_predicate(encoding);
            let translated_start = instructions.len();
            if block_targets.contains(&offset)
                || is_branch(encoding)
                || is_set_sync_point(encoding)
                || is_loop_push(encoding)
                || encoding >> 48 == 0xe250
                || is_loop_branch(encoding)
                || is_synchronize(encoding)
            {
                global_memory.begin_block();
            }

            if block_targets.contains(&offset)
                || is_branch(encoding)
                || is_set_sync_point(encoding)
                || is_loop_push(encoding)
                || encoding >> 48 == 0xe250
                || is_loop_branch(encoding)
                || is_synchronize(encoding)
            {
                for range in pending_range_reductions.values_mut() {
                    range.fusion_valid = false;
                }
            }
            // SM50 KIL has no data operands; its execution predicate controls
            // fragment termination. The header enables KIL independently.
            // https://gitlab.freedesktop.org/mesa/mesa/-/blob/2c9073912232b93eb9b60486edbd72d53e5f3d26/src/nouveau/compiler/nak/sm50.rs (OpKill)
            if encoding >> 48 == 0xe330 {
                if stage != MaxwellShaderStage::Pixel || !binary.header().bit(15) {
                    return Err(unsupported_instruction(binary, offset, encoding));
                }
                let predicate_bits = 0xf_u64 << 16;
                if encoding & !(0xffff_u64 << 48 | predicate_bits) != 0xf {
                    return Err(malformed(
                        stage,
                        offset,
                        encoding,
                        "unsupported KIL operand fields",
                    ));
                }
                instructions.push(ShaderInstruction::new(
                    source,
                    predicate,
                    ShaderOperation::Discard,
                ));
                continue;
            }
            if is_exit(encoding) {
                append_implicit_outputs(
                    neutral_stage,
                    source,
                    predicate,
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
                continue;
            }

            if encoding >> 48 == 0xe250 {
                let table = binary
                    .indirect_tables
                    .iter()
                    .find(|table| table.offset == offset)
                    .ok_or_else(|| {
                        malformed(
                            stage,
                            offset,
                            encoding,
                            "BRX table targets were not captured",
                        )
                    })?;
                super::decode::validate_register_range(
                    stage,
                    offset,
                    encoding,
                    table.register,
                    1,
                    register_count,
                )?;
                instructions.push(ShaderInstruction::new(
                    source,
                    predicate,
                    ShaderOperation::BranchTable {
                        value: crate::shader::decoded::Register::new(u32::from(table.register)),
                        targets: table.targets.clone(),
                    },
                ));
                patch_addresses.ordinary(stage, encoding, &instructions[translated_start..])?;
                continue;
            }
            if is_loop_push(encoding) || is_set_sync_point(encoding) {
                decode_shader_control_target(stage, offset, encoding, code_size)?;
                instructions.push(ShaderInstruction::new(
                    source,
                    ShaderPredicate::Always,
                    ShaderOperation::Nop,
                ));
                continue;
            }
            if is_loop_branch(encoding) || is_synchronize(encoding) {
                if let Some(&target) = stack_targets.get(&offset) {
                    instructions.push(ShaderInstruction::new(
                        source,
                        predicate,
                        ShaderOperation::Branch { target },
                    ));
                } else {
                    // No CFG edge reaches this instruction.
                    instructions.push(ShaderInstruction::new(
                        source,
                        ShaderPredicate::Always,
                        ShaderOperation::Nop,
                    ));
                }
                patch_addresses.ordinary(stage, encoding, &instructions[translated_start..])?;
                continue;
            }
            if is_dependency_barrier(encoding) {
                validate_dependency_barrier(stage, offset, encoding)?;
                instructions.push(ShaderInstruction::new(
                    source,
                    predicate,
                    ShaderOperation::Nop,
                ));
                continue;
            }

            if predicate == ShaderPredicate::Never {
                // Predicated-false instructions have no architectural data,
                // interface, or control-flow effect. The encoding family is
                // still classified so random data cannot hide as dead code.
                if !is_supported_family(encoding) {
                    return Err(unsupported_instruction(binary, offset, encoding));
                }
                instructions.push(ShaderInstruction::new(
                    source,
                    ShaderPredicate::Always,
                    ShaderOperation::Nop,
                ));
                continue;
            }

            if is_range_reduction(encoding) {
                let mut range = decode_range_reduction(
                    stage,
                    offset,
                    encoding,
                    predicate,
                    register_count,
                    &mut next_temporary,
                )?;
                freeze_range_reduction_input(&mut range, stage, &mut next_temporary)?;
                if let Some(binding) = range.constant_buffer_binding {
                    constant_buffer_bindings.insert(binding);
                }
                instructions.extend(
                    std::mem::take(&mut range.preparation)
                        .into_iter()
                        .map(|operation| ShaderInstruction::new(source, predicate, operation)),
                );
                for instruction in &instructions[translated_start..] {
                    if instruction
                        .operation()
                        .source_registers()
                        .iter()
                        .any(|register| {
                            register.index() < u32::from(register_count)
                                && pending_range_reductions.contains_key(&(register.index() as u8))
                        })
                    {
                        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
                            stage,
                            instruction_offset: offset,
                            encoding,
                            detail: "nested RRO reduced representation is not implemented",
                        });
                    }
                }
                patch_addresses.ordinary(stage, encoding, &instructions[translated_start..])?;
                let destination =
                    crate::shader::decoded::Register::new(u32::from(range.destination));
                patch_addresses.remove(destination);
                global_memory.observe(
                    &mut instructions,
                    translated_start,
                    encoding,
                    &mut next_temporary,
                )?;
                global_memory.forget(destination);
                pending_range_reductions.insert(range.destination, range);
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
                let operation = if global_memory::is_store(encoding)
                    || global_memory::is_load(encoding)
                {
                    let operations = if global_memory::is_load(encoding) {
                        global_memory.load(offset, encoding, register_count, &mut next_temporary)?
                    } else {
                        global_memory.store(
                            offset,
                            encoding,
                            register_count,
                            &mut next_temporary,
                        )?
                    };
                    append_expanded_operations(&mut instructions, source, predicate, operations);
                    break 'instruction;
                } else if is_branch(encoding) {
                    ShaderOperation::Branch {
                        target: decode_shader_control_target(stage, offset, encoding, code_size)?,
                    }
                } else if quad::is_swizzle_add(encoding) {
                    quad::decode_swizzle_add(stage, offset, encoding, register_count)?
                } else if quad::is_shuffle(encoding) {
                    let operations = quad::decode_shuffle(stage, offset, encoding, register_count)?;
                    append_expanded_operations(&mut instructions, source, predicate, operations);
                    break 'instruction;
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
                            y_negate,
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
                                location @ (ShaderIoLocation::VertexId
                                | ShaderIoLocation::InstanceId
                                | ShaderIoLocation::FrontFacing),
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
                    let operations = decode_attribute_store(
                        stage,
                        offset,
                        encoding,
                        register_count,
                        &mut next_temporary,
                    )?;
                    for operation in &operations {
                        if let ShaderOperation::StoreOutput {
                            location,
                            first_component,
                            sources,
                            ..
                        } = operation
                        {
                            for component in 0..sources.len() {
                                explicitly_stored.insert((
                                    *location,
                                    first_component.saturating_add(component as u8),
                                ));
                            }
                        }
                    }
                    append_expanded_operations(&mut instructions, source, predicate, operations);
                    break 'instruction;
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
                } else if is_shift_left(encoding) || is_shift_right(encoding) {
                    let decoder = if is_shift_right(encoding) {
                        decode_shift_right
                    } else {
                        decode_shift_left
                    };
                    let decoded =
                        decoder(stage, offset, encoding, register_count, &mut next_temporary)?;
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
                    let decoded = decode_constant_buffer_load(
                        stage,
                        offset,
                        encoding,
                        register_count,
                        &mut next_temporary,
                    )?;
                    constant_buffer_bindings.insert(decoded.constant_buffer_binding);
                    // All words of one LDC share its predicate; none may write
                    // an architectural destination when that predicate is false.
                    instructions.extend(
                        decoded
                            .operations
                            .into_iter()
                            .map(|operation| ShaderInstruction::new(source, predicate, operation)),
                    );
                    break 'instruction;
                } else if is_texture_query(encoding) {
                    if stage == MaxwellShaderStage::Compute {
                        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
                            stage,
                            instruction_offset: offset,
                            encoding,
                            detail: "compute texture binding ABI is not implemented",
                        });
                    }
                    let query = decode_texture_query(stage, offset, encoding, register_count)?;
                    // Resolve after texture accesses have established descriptor
                    // dimensionality; no assumed image type escapes translation.
                    texture_queries.insert(offset, (query.descriptor, encoding));
                    instructions.push(ShaderInstruction::new(
                        source,
                        predicate,
                        ShaderOperation::QueryTextureLevels {
                            destination: query.destination,
                            image_binding: u32::MAX,
                        },
                    ));
                    break 'instruction;
                } else if is_texture_access_simplified(encoding) || is_texture_sample(encoding) {
                    if stage == MaxwellShaderStage::Compute {
                        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
                            stage,
                            instruction_offset: offset,
                            encoding,
                            detail: "compute texture binding ABI is not implemented",
                        });
                    }
                    let operations = if is_texture_sample(encoding) {
                        decode_texture_sample(
                            stage,
                            offset,
                            encoding,
                            register_count,
                            &mut texture_bindings,
                            &mut next_temporary,
                            &instructions,
                        )?
                    } else {
                        decode_texture_access_simplified(
                            stage,
                            offset,
                            encoding,
                            register_count,
                            &mut texture_bindings,
                            &mut next_temporary,
                        )?
                    };
                    // A texture instruction may write several architectural registers;
                    // preserve its predicate on both writes and the sample.
                    instructions.extend(
                        operations
                            .into_iter()
                            .map(|operation| ShaderInstruction::new(source, predicate, operation)),
                    );
                    break 'instruction;
                } else if is_interpolate(encoding) {
                    let operations = decode_interpolate(
                        stage,
                        offset,
                        encoding,
                        register_count,
                        &mut inputs,
                        &mut next_temporary,
                        straight_line_fragment
                            && ((encoding >> 54) & 3) == 1
                            && super::interface::is_reciprocal_fragment_w(
                                &instructions,
                                crate::shader::decoded::Register::new(u32::from(
                                    ((encoding >> 20) & 0xff) as u16,
                                )),
                            ),
                    )?;
                    append_expanded_operations(&mut instructions, source, predicate, operations);
                    break 'instruction;
                } else if is_mufu(encoding) {
                    if let Some(range_reduction) =
                        pending_range_reductions.get(&((encoding >> 8) as u8))
                    {
                        let operation = decode_range_reduced_mufu(
                            stage,
                            offset,
                            encoding,
                            register_count,
                            range_reduction,
                        )?;
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
                } else if is_half_set_predicate(encoding) {
                    let operations = decode_half_set_predicate(
                        stage,
                        offset,
                        encoding,
                        register_count,
                        &mut next_temporary,
                    )?;
                    instructions.extend(
                        operations
                            .into_iter()
                            .map(|operation| ShaderInstruction::new(source, predicate, operation)),
                    );
                    break 'instruction;
                } else if is_half_add(encoding)
                    || is_half_multiply(encoding)
                    || is_half_fused_multiply_add(encoding)
                {
                    let decode = if is_half_add(encoding) {
                        decode_half_add
                    } else if is_half_multiply(encoding) {
                        decode_half_multiply
                    } else {
                        decode_half_fused_multiply_add
                    };
                    let decoded =
                        decode(stage, offset, encoding, register_count, &mut next_temporary)?;
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
                } else if xmad::is_xmad(encoding) {
                    let decoded = xmad::decode_xmad(
                        stage,
                        offset,
                        encoding,
                        register_count,
                        &mut next_temporary,
                    )?;
                    if let Some(binding) = decoded.constant_buffer_binding {
                        constant_buffer_bindings.insert(binding);
                    }
                    instructions.extend(
                        decoded
                            .operations
                            .into_iter()
                            .map(|operation| ShaderInstruction::new(source, predicate, operation)),
                    );
                    break 'instruction;
                } else if integer::is_register_to_predicate(encoding) {
                    let operations = integer::decode_register_to_predicate(
                        stage,
                        offset,
                        encoding,
                        register_count,
                        &mut next_temporary,
                    )?;
                    append_expanded_operations(&mut instructions, source, predicate, operations);
                    break 'instruction;
                } else if integer::is_condition_set_predicate(encoding) {
                    let decoded = integer::decode_condition_set_predicate(
                        stage,
                        offset,
                        encoding,
                        &mut next_temporary,
                        &integer_flags,
                    )?;
                    append_expanded_operations(
                        &mut instructions,
                        source,
                        predicate,
                        decoded.operations,
                    );
                    break 'instruction;
                } else if integer::is_predicate_boolean(encoding) {
                    integer::decode_predicate_boolean(stage, offset, encoding)?
                } else if integer::is_integer_to_integer(encoding)
                    || integer::is_bitfield_extract(encoding)
                    || integer::is_bitfield_insert(encoding)
                    || integer::is_integer_min_max(encoding)
                    || integer::is_integer_compare(encoding)
                    || integer::is_select(encoding)
                    || integer::is_add_three(encoding)
                    || integer::is_bitwise(encoding)
                    || integer::is_add(encoding)
                    || integer::is_shift_add(encoding)
                    || integer::is_set(encoding)
                    || integer::is_set_predicate(encoding)
                {
                    let decoded = if integer::is_integer_to_integer(encoding) {
                        integer::decode_integer_to_integer(
                            stage,
                            offset,
                            encoding,
                            register_count,
                            &mut next_temporary,
                            &mut integer_flags,
                        )?
                    } else if integer::is_add(encoding) || integer::is_shift_add(encoding) {
                        if encoding
                            & (1 << if encoding >> 48 & 0xfe00 == 0x1c00 {
                                52
                            } else {
                                47
                            })
                            != 0
                        {
                            // This decoder currently defines only carry. Other flag
                            // consumers remain unsupported until they have a definition.
                            integer_flags.zero = None;
                            integer_flags.sign = None;
                            integer_flags.overflow = None;
                        }
                        integer::decode_add(
                            stage,
                            offset,
                            encoding,
                            register_count,
                            &mut next_temporary,
                            &mut integer_flags.carry,
                        )?
                    } else {
                        let decode = if integer::is_bitfield_extract(encoding) {
                            integer::decode_bitfield_extract
                        } else if integer::is_bitfield_insert(encoding) {
                            integer::decode_bitfield_insert
                        } else if integer::is_add_three(encoding) {
                            integer::decode_add_three
                        } else if integer::is_select(encoding) {
                            integer::decode_select
                        } else if integer::is_integer_compare(encoding) {
                            integer::decode_integer_compare
                        } else if integer::is_integer_min_max(encoding) {
                            integer::decode_integer_min_max
                        } else if integer::is_bitwise(encoding) {
                            integer::decode_bitwise
                        } else if integer::is_set(encoding) {
                            integer::decode_set
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
                } else if is_float_set(encoding) {
                    let decoded = decode_float_set(
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
            for instruction in &instructions[translated_start..] {
                for register in instruction.operation().source_registers() {
                    if let Some(range) = pending_range_reductions.get(&(register.index() as u8))
                        && register.index() < u32::from(register_count)
                    {
                        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
                            stage,
                            instruction_offset: offset,
                            encoding: range.encoding,
                            detail: "RRO reduced representation is consumed outside a compatible MUFU",
                        });
                    }
                }
                instruction
                    .operation()
                    .visit_destination_registers(|register| {
                        if register.index() < u32::from(register_count) {
                            if predicate == ShaderPredicate::Always {
                                pending_range_reductions.remove(&(register.index() as u8));
                            } else if let Some(range) =
                                pending_range_reductions.get_mut(&(register.index() as u8))
                            {
                                range.fusion_valid = false;
                            }
                        }
                    });
            }
            for instruction in &instructions[translated_start..] {
                let written = match instruction.operation() {
                    ShaderOperation::SetPredicateFloat32 { destination, .. } => vec![*destination],
                    ShaderOperation::SetPredicatesFromBits { write_mask, .. } => (0..7)
                        .filter(|index| write_mask & (1 << index) != 0)
                        .collect(),
                    ShaderOperation::SetPredicateBoolean { destinations, .. }
                    | ShaderOperation::SetPredicateInteger32 { destinations, .. } => {
                        destinations.iter().flatten().copied().collect()
                    }
                    _ => Vec::new(),
                };
                for range in pending_range_reductions.values_mut() {
                    if matches!(range.predicate, ShaderPredicate::Register { register, .. } if written.contains(&register))
                    {
                        range.fusion_valid = false;
                    }
                }
            }
            patch_addresses.ordinary(stage, encoding, &instructions[translated_start..])?;
            global_memory.observe(
                &mut instructions,
                translated_start,
                encoding,
                &mut next_temporary,
            )?;
        }
    }
    for instruction in &mut instructions {
        if let ShaderOperation::QueryTextureLevels { destination, .. } = instruction.operation() {
            let destination = *destination;
            let offset = instruction.source().byte_offset();
            let (descriptor, encoding) = texture_queries[&offset];
            let image_binding = texture_bindings
                .get(&descriptor)
                .ok_or(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
                    stage,
                    instruction_offset: offset,
                    encoding,
                    detail: "TXQ-only descriptor dimensionality requires TIC specialization",
                })?
                .image_binding;
            *instruction = ShaderInstruction::new(
                instruction.source(),
                instruction.predicate(),
                ShaderOperation::QueryTextureLevels {
                    destination,
                    image_binding,
                },
            );
        }
    }
    let mut resources = constant_buffer_bindings
        .into_iter()
        .map(|binding| {
            ShaderResourceAccess::new(
                binding.into(),
                ShaderResourceKind::ConstantBuffer,
                true,
                false,
            )
            .expect("read-only constant-buffer access is valid")
        })
        .collect::<Vec<_>>();
    let mut global_buffers = global_memory.bindings;
    // Texture pairs occupy local bindings starting at 32. Place storage
    // bindings after the complete texture set before verification/linking.
    if !global_buffers.buffers.is_empty() && !texture_bindings.is_empty() {
        let shift = texture_bindings.len() as u32 * 2;
        for buffer in &mut global_buffers.buffers {
            buffer.binding = buffer
                .binding
                .checked_add(shift)
                .ok_or(MaxwellShaderTranslationError::ResourceBindingExhausted)?;
        }
        for instruction in &mut instructions {
            let operation = match instruction.operation() {
                ShaderOperation::LoadStorageBuffer32 {
                    destination,
                    binding,
                    word_index,
                } => ShaderOperation::LoadStorageBuffer32 {
                    destination: *destination,
                    binding: binding
                        .checked_add(shift)
                        .ok_or(MaxwellShaderTranslationError::ResourceBindingExhausted)?,
                    word_index: *word_index,
                },
                ShaderOperation::StoreStorageBuffer32 {
                    source,
                    binding,
                    word_index,
                } => ShaderOperation::StoreStorageBuffer32 {
                    source: *source,
                    binding: binding
                        .checked_add(shift)
                        .ok_or(MaxwellShaderTranslationError::ResourceBindingExhausted)?,
                    word_index: *word_index,
                },
                _ => continue,
            };
            *instruction =
                ShaderInstruction::new(instruction.source(), instruction.predicate(), operation);
        }
    }
    for buffer in &global_buffers.buffers {
        resources.push(
            ShaderResourceAccess::new(
                buffer.binding,
                ShaderResourceKind::StorageBuffer,
                buffer.read,
                buffer.write,
            )
            .expect("global storage access is valid"),
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
                    if matches!(
                        binding.image_kind,
                        ShaderResourceKind::SampledDepthImage
                            | ShaderResourceKind::SampledDepthImage2DArray
                    ) {
                        ShaderResourceKind::ComparisonSampler
                    } else {
                        ShaderResourceKind::Sampler
                    },
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
    // Guest GPRs have undefined initial contents, unlike compiler temporaries.
    // Emit only potentially live-in hardware registers. Block-local definite
    // writes eliminate unnecessary entry definitions in ordinary shaders.
    // https://github.com/eden-emulator/mirror/blob/master/src/shader_recompiler/ir_opt/ssa_rewrite_pass.cpp (ReadVariable)
    let mut defined = BTreeSet::new();
    let mut live_in = BTreeSet::new();
    let mut previous_source = None;
    for instruction in &instructions {
        if previous_source != Some(instruction.source())
            && block_targets.contains(&instruction.source().byte_offset())
        {
            defined.clear();
        }
        previous_source = Some(instruction.source());
        for register in instruction.operation().source_registers() {
            if register.index() < u32::from(register_count) && !defined.contains(&register) {
                live_in.insert(register);
            }
        }
        if instruction.predicate() == ShaderPredicate::Always {
            instruction
                .operation()
                .visit_destination_registers(|register| {
                    defined.insert(register);
                });
        }
        if matches!(
            instruction.operation(),
            ShaderOperation::Branch { .. } | ShaderOperation::BranchTable { .. }
        ) {
            defined.clear();
        }
    }
    instructions.splice(
        0..0,
        live_in.into_iter().map(|destination| {
            ShaderInstruction::new(
                ShaderSourceLocation::new(0),
                ShaderPredicate::Always,
                ShaderOperation::Undefined32 { destination },
            )
        }),
    );
    if stage == MaxwellShaderStage::Vertex {
        // One compiler scratch register suffices for the output prologue. It is
        // allocated after translation and has no overlap with guest temporaries.
        let register = crate::shader::decoded::Register::new(next_temporary);
        let source = ShaderSourceLocation::new(0);
        let prologue = (0..4).flat_map(|component| {
            [
                ShaderInstruction::new(
                    source,
                    ShaderPredicate::Always,
                    ShaderOperation::MoveImmediate32 {
                        destination: register,
                        bits: if component == 3 { 1.0_f32.to_bits() } else { 0 },
                        scalar_type: ShaderScalarType::Float32,
                    },
                ),
                ShaderInstruction::new(
                    source,
                    ShaderPredicate::Always,
                    ShaderOperation::StoreOutput {
                        sources: vec![register].into_boxed_slice(),
                        location: ShaderIoLocation::Position,
                        first_component: component,
                        scalar_type: ShaderScalarType::Float32,
                    },
                ),
            ]
        });
        let entry = instructions
            .iter()
            .position(|instruction| instruction.source().byte_offset() != 0)
            .unwrap_or(instructions.len());
        instructions.splice(entry..entry, prologue);
    }
    if stage == MaxwellShaderStage::Pixel {
        // SPH configures interpolators; only actual attribute reads belong in
        // the linked host interface. Eden also uses IR loads, not every SPH bit:
        // https://github.com/eden-emulator/mirror/blob/d16735f5b618942136d6ab53466e3be0a382c30a/src/shader_recompiler/backend/spirv/spirv_emit_context.cpp
        let mut read = BTreeSet::new();
        for instruction in &instructions {
            if instruction.predicate() == ShaderPredicate::Never {
                continue;
            }
            match instruction.operation() {
                ShaderOperation::InterpolateInput {
                    location,
                    component,
                    ..
                } => {
                    read.insert((*location, *component));
                }
                ShaderOperation::LoadInput {
                    destinations,
                    location,
                    first_component,
                    ..
                } => {
                    read.extend(
                        (0..destinations.len())
                            .map(|index| (*location, *first_component + index as u8)),
                    );
                }
                _ => {}
            }
        }
        inputs.retain(|input| read.contains(&(input.location(), input.component())));
    }
    Ok(DecodedProgram {
        stage: neutral_stage,
        inputs,
        outputs,
        resources,
        instructions,
        workgroup_size: match binary.metadata {
            MaxwellShaderMetadata::Compute { workgroup_size, .. } => Some(workgroup_size),
            _ => None,
        },
        tessellation_control_points: (stage == MaxwellShaderStage::TessellationInit)
            .then(|| binary.header().bits(88, 8) as u32),
        temporary_count: next_temporary,
        texture_bindings: texture_bindings.values().copied().collect(),
        global_buffers,
    })
}

pub(super) struct DecodedProgram {
    pub(super) stage: nixe_gpu::ShaderStage,
    pub(super) inputs: Vec<ShaderInterfaceElement>,
    pub(super) outputs: Vec<ShaderInterfaceElement>,
    pub(super) resources: Vec<ShaderResourceAccess>,
    pub(super) instructions: Vec<ShaderInstruction>,
    pub(super) workgroup_size: Option<[u32; 3]>,
    pub(super) tessellation_control_points: Option<u32>,
    pub(super) temporary_count: u32,
    pub(super) texture_bindings: Box<[MaxwellTextureResourceBinding]>,
    pub(super) global_buffers: global_memory::GlobalBufferBindings,
}

/// Retiring oracle export. The native compiler calls decode_shader_binary and
/// consumes frontend instructions directly; it never calls this adapter.
pub(super) fn translate_shader_binary(
    binary: &MaxwellShaderBinary,
    register_count: u8,
    vertex_input_types: &BTreeMap<ShaderIoLocation, ShaderScalarType>,
    y_negate: bool,
) -> Result<TranslatedShaderIr, MaxwellShaderTranslationError> {
    let mut decoded = decode_shader_binary(binary, register_count, vertex_input_types, y_negate)?;
    // Freeze the reference renderer's historical launch bridge only here. It
    // is not an ISA rule and never enters native SSA compilation.
    if decoded.temporary_count > u32::from(u16::MAX) {
        return Err(MaxwellShaderTranslationError::UnsupportedSemanticDetail {
            stage: binary.stage(),
            instruction_offset: 0,
            encoding: 0,
            detail: "reference renderer temporary namespace is limited to u16",
        });
    }
    let preload = oracle_preload_vertex_inputs(decoded.stage, &decoded.inputs);
    let mut launch_registers = BTreeSet::new();
    for i in &preload {
        i.operation().visit_destination_registers(|r| {
            launch_registers.insert(r);
        });
    }
    decoded.instructions.retain(|i|i.source().byte_offset()!=0||!matches!(i.operation(),ShaderOperation::Undefined32 {destination} if launch_registers.contains(destination)));
    decoded.instructions.splice(0..0, preload);
    let mut ir = ShaderIr::new(
        decoded.stage,
        decoded.inputs,
        decoded.outputs,
        decoded
            .resources
            .into_iter()
            .map(ShaderResourceAccess::oracle)
            .collect::<Result<_, _>>()?,
        decoded.instructions.into_iter().map(Into::into).collect(),
    )
    .with_tessellation_control_points(decoded.tessellation_control_points);
    if let Some(size) = decoded.workgroup_size {
        ir = ir.with_workgroup_size(size);
    }
    Ok(TranslatedShaderIr {
        ir,
        texture_bindings: decoded.texture_bindings,
        global_buffers: decoded.global_buffers,
    })
}

fn append_expanded_operations(
    instructions: &mut Vec<ShaderInstruction>,
    source: ShaderSourceLocation,
    predicate: ShaderPredicate,
    operations: Vec<ShaderOperation>,
) {
    instructions.extend(
        operations
            .into_iter()
            .map(|operation| ShaderInstruction::new(source, predicate, operation)),
    );
}

const fn is_supported_family(encoding: u64) -> bool {
    is_exit(encoding)
        || encoding >> 48 == 0xe330
        || global_memory::is_load(encoding)
        || global_memory::is_store(encoding)
        || tessellation::is_system_register_read(encoding)
        || quad::is_shuffle(encoding)
        || quad::is_swizzle_add(encoding)
        || is_branch(encoding)
        || is_set_sync_point(encoding)
        || is_loop_push(encoding)
        || encoding >> 48 == 0xe250
        || is_loop_branch(encoding)
        || is_synchronize(encoding)
        || is_dependency_barrier(encoding)
        || is_attribute_load(encoding)
        || is_attribute_store(encoding)
        || is_move_immediate(encoding)
        || is_move(encoding)
        || is_shift_left(encoding)
        || is_shift_right(encoding)
        || is_integer_to_float(encoding)
        || is_float_to_float(encoding)
        || is_float_to_integer(encoding)
        || is_constant_buffer_load(encoding)
        || is_texture_access_simplified(encoding)
        || is_texture_sample(encoding)
        || is_texture_query(encoding)
        || is_interpolate(encoding)
        || is_range_reduction(encoding)
        || is_mufu(encoding)
        || is_float_min_max(encoding)
        || is_half_set_predicate(encoding)
        || is_half_add(encoding)
        || is_half_multiply(encoding)
        || is_half_fused_multiply_add(encoding)
        || is_float_multiply(encoding)
        || is_float_fused_multiply_add(encoding)
        || is_float_add(encoding)
        || is_float_set(encoding)
        || is_float_set_predicate(encoding)
        || integer::is_set(encoding)
        || integer::is_set_predicate(encoding)
        || integer::is_integer_to_integer(encoding)
        || integer::is_condition_set_predicate(encoding)
        || integer::is_predicate_boolean(encoding)
        || integer::is_register_to_predicate(encoding)
        || xmad::is_xmad(encoding)
        || integer::is_integer_min_max(encoding)
        || integer::is_integer_compare(encoding)
        || integer::is_select(encoding)
        || integer::is_add_three(encoding)
        || integer::is_bitfield_insert(encoding)
        || integer::is_bitwise(encoding)
        || integer::is_add(encoding)
        || integer::is_shift_add(encoding)
        || patch_address::is_supported_family(encoding)
}

#[cfg(test)]
mod tests;
