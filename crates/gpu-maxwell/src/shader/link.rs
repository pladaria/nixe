//! Graphics stage linking, resource binding remapping, and verified backend programs.

use super::error::MaxwellShaderTranslationError;
use super::interface::neutral_stage;
use super::source::{MaxwellShaderProgramTranslationInput, MaxwellShaderTranslationInputs};
use super::texture::MaxwellTextureResourceBinding;
use super::translate::{TranslatedShaderIr, translate_shader_binary};
use crate::{MaxwellShaderStage, MaxwellThreeDDirectlyAddressableMemory};
use nixe_gpu::{
    ShaderBackendModule, ShaderInstruction, ShaderInterfaceElement, ShaderInterpolation,
    ShaderIoLocation, ShaderIr, ShaderOperation, ShaderResourceAccess, ShaderResourceKind,
    ShaderScalarType, ShaderStage, VerifiedShaderIr,
};
use std::collections::BTreeMap;
use std::hash::Hash;
use std::sync::Arc;

/// Complete identity of one translation input, independent from GPU VA reuse.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct MaxwellShaderTranslationKey {
    input: Arc<MaxwellShaderProgramTranslationInput>,
    resource_binding_remap: Box<[(u8, u8)]>,
    linked_output_interpolation: Box<[((ShaderIoLocation, u8), ShaderInterpolation)]>,
    prune_raster_outputs: bool,
    defaulted_fragment_inputs: Box<[(ShaderIoLocation, u8)]>,
}

/// One verified neutral program and its portable backend module.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct MaxwellTranslatedShaderProgram {
    key: MaxwellShaderTranslationKey,
    fingerprint: u128,
    module: ShaderBackendModule,
    directly_addressable_memory: Option<MaxwellThreeDDirectlyAddressableMemory>,
    maximum_api_visible_calls: u16,
    texture_bindings: Box<[MaxwellTextureResourceBinding]>,
    global_buffers: super::global_memory::GlobalBufferBindings,
}

impl MaxwellTranslatedShaderProgram {
    #[cfg(debug_assertions)]
    pub(crate) const fn key(&self) -> &MaxwellShaderTranslationKey {
        &self.key
    }

    pub(crate) const fn fingerprint(&self) -> u128 {
        self.fingerprint
    }

    pub(crate) fn stage(&self) -> ShaderStage {
        self.module.stage()
    }

    pub(crate) fn bind_group(&self) -> Option<u8> {
        self.key.input.effective_group
    }

    pub(crate) fn resources(&self) -> &[ShaderResourceAccess] {
        self.module.ir().ir().resources()
    }

    pub(crate) fn local_resource_binding(&self, neutral_binding: u8) -> Option<u8> {
        self.key
            .resource_binding_remap
            .iter()
            .find_map(|(local, neutral)| (*neutral == neutral_binding).then_some(*local))
    }

    pub(crate) fn global_buffer(
        &self,
        binding: u8,
    ) -> Option<&super::global_memory::GlobalBufferBinding> {
        self.global_buffers
            .buffers
            .iter()
            .find(|buffer| buffer.binding == binding.into())
    }

    pub(crate) fn buffer_size(&self, binding: u8) -> crate::engines::MaxwellShaderBufferSize {
        use crate::engines::MaxwellShaderBufferSize;
        if let Some((index, buffer)) = self
            .global_buffers
            .buffers
            .iter()
            .enumerate()
            .find(|(_, b)| b.binding == binding.into())
        {
            return if buffer.runtime_extent {
                MaxwellShaderBufferSize::Declared
            } else {
                MaxwellShaderBufferSize::Proven(
                    self.global_buffers.byte_extents([1; 3], [1; 3])[index],
                )
            };
        }
        match self.constant_buffer_read_end(binding) {
            u32::MAX => MaxwellShaderBufferSize::Declared,
            size => MaxwellShaderBufferSize::Proven(u64::from(size)),
        }
    }

    /// Only proven accesses limit imports. Unknown indexed loads retain the
    /// declared binding size; BRX uses the same dominating bounds as capture.
    /// https://github.com/eden-emulator/mirror/blob/master/src/video_core/buffer_cache/buffer_cache.h#L929-L935
    pub(crate) fn constant_buffer_read_end(&self, binding: u8) -> u32 {
        self.module
            .ir()
            .ir()
            .instructions()
            .iter()
            .fold(0, |end, instruction| {
                let read_end = match instruction.operation() {
                    ShaderOperation::LoadConstantBuffer32 {
                        binding: current,
                        byte_offset,
                        ..
                    } if *current == binding => byte_offset.saturating_add(4),
                    ShaderOperation::LoadConstantBufferIndexed32 {
                        binding: current, ..
                    } if *current == binding => self
                        .key
                        .input
                        .binary
                        .indirect_tables
                        .iter()
                        .filter(|table| table.load_offset == instruction.source().byte_offset())
                        .map(|table| table.read_end)
                        .max()
                        .unwrap_or(u32::MAX),
                    _ => 0,
                };
                end.max(read_end)
            })
    }

    pub(crate) const fn module(&self) -> &ShaderBackendModule {
        &self.module
    }

    pub(crate) const fn maximum_api_visible_calls(&self) -> u16 {
        self.maximum_api_visible_calls
    }

    /// Returns the Maxwell local/shared-memory partition consumed by this
    /// translation, if any. Register, attribute, constant-buffer, and texture
    /// operations do not consume `SET_L1_CONFIGURATION`.
    pub(crate) const fn directly_addressable_memory(
        &self,
    ) -> Option<MaxwellThreeDDirectlyAddressableMemory> {
        self.directly_addressable_memory
    }

    pub(crate) fn texture_constant_buffer_slot(&self) -> Option<u8> {
        self.key.input.texture_constant_buffer_slot
    }

    pub(crate) fn texture_bindings(&self) -> &[MaxwellTextureResourceBinding] {
        &self.texture_bindings
    }
}

pub(crate) fn translate_prepared_maxwell_shader_programs(
    inputs: &MaxwellShaderTranslationInputs,
) -> Result<Vec<MaxwellTranslatedShaderProgram>, MaxwellShaderTranslationError> {
    let mut translated = Vec::with_capacity(inputs.programs.len());
    for input in &inputs.programs {
        let vertex_input_types = input.vertex_input_types.iter().copied().collect();
        translated.push(translate_shader_binary(
            &input.binary,
            input.register_count,
            &vertex_input_types,
            input.y_negate,
        )?);
    }

    let defaulted_fragment_inputs = specialize_unwritten_fragment_inputs(&mut translated);
    validate_graphics_stage_interfaces(&translated)?;
    let global_bindings = graphics_resource_bindings(inputs, &translated)?;
    let linked_interpolation = graphics_output_interpolation(&translated);
    let has_fragment = translated
        .iter()
        .any(|program| program.ir.stage() == ShaderStage::Fragment);
    let final_producer = translated
        .iter()
        .map(|program| program.ir.stage())
        .filter(|stage| {
            matches!(
                stage,
                ShaderStage::Vertex | ShaderStage::TessellationEvaluation | ShaderStage::Geometry
            )
        })
        .max_by_key(|stage| match stage {
            ShaderStage::Geometry => 2,
            ShaderStage::TessellationEvaluation => 1,
            _ => 0,
        });
    let mut programs = Vec::with_capacity(translated.len());
    for (input, mut translated) in inputs.programs.iter().zip(translated) {
        let stage = input.binary.stage();
        let local_bindings = program_resource_bindings(input, &translated, &global_bindings)?;
        for buffer in &mut translated.global_buffers.buffers {
            buffer.binding =
                remapped_oracle_metadata_binding(&local_bindings, stage, buffer.binding)?;
        }
        for texture in &mut translated.texture_bindings {
            texture.image_binding =
                remapped_oracle_metadata_binding(&local_bindings, stage, texture.image_binding)?;
            texture.sampler_binding = texture
                .sampler_binding
                .map(|binding| remapped_oracle_metadata_binding(&local_bindings, stage, binding))
                .transpose()?;
        }
        let output_interpolation = if Some(neutral_stage(stage)) == final_producer {
            linked_interpolation.as_slice()
        } else {
            &[]
        };
        let mut ir =
            finalize_shader_ir(translated.ir, stage, &local_bindings, output_interpolation)?;
        // Fixed-function validation rejects transform feedback; only fragment
        // consumers and raster builtins can observe this final stage's outputs.
        let prune_raster_outputs = has_fragment && Some(neutral_stage(stage)) == final_producer;
        if prune_raster_outputs {
            ir = ir
                .prune_raster_outputs(output_interpolation.iter().map(|(component, _)| *component));
        }
        // Output pruning can eliminate every access to a constant buffer.
        // Remove its declaration too, instead of importing an empty range or
        // demanding an unused guest binding merely to build a host layout.
        let used_constant_buffers: std::collections::BTreeSet<_> = ir
            .ir()
            .instructions()
            .iter()
            .filter_map(|instruction| match instruction.operation() {
                ShaderOperation::LoadConstantBuffer32 { binding, .. }
                | ShaderOperation::LoadConstantBufferIndexed32 { binding, .. } => Some(*binding),
                _ => None,
            })
            .collect();
        let resources: Vec<_> = ir
            .ir()
            .resources()
            .iter()
            .filter(|resource| {
                resource.kind() != ShaderResourceKind::ConstantBuffer
                    || used_constant_buffers.contains(&resource.binding())
            })
            .copied()
            .collect();
        if resources.len() != ir.ir().resources().len() {
            ir = VerifiedShaderIr::verify(ir.ir().clone().with_resources(resources))?;
        }
        if !translated.texture_bindings.is_empty() && input.texture_constant_buffer_slot.is_none() {
            return Err(MaxwellShaderTranslationError::IncompletePipelineBinding {
                pipeline: input.pipeline,
                field: "SET_BINDLESS_TEXTURE_CONSTANT_BUFFER_SLOT",
            });
        }
        let module = ShaderBackendModule::new(ir);
        let key = MaxwellShaderTranslationKey {
            input: Arc::clone(input),
            resource_binding_remap: local_bindings.into_iter().collect(),
            linked_output_interpolation: output_interpolation.into(),
            prune_raster_outputs,
            defaulted_fragment_inputs: if stage == MaxwellShaderStage::Pixel {
                defaulted_fragment_inputs.clone().into()
            } else {
                Box::new([])
            },
        };
        programs.push(MaxwellTranslatedShaderProgram {
            fingerprint: nixe_gpu::cache_fingerprint(&key),
            key,
            module,
            // No currently translated SASS operation addresses Maxwell
            // local/shared memory. Keep this explicit so adding that family
            // must also declare its concrete partition requirement instead of
            // silently consuming unrelated or absent class state.
            directly_addressable_memory: None,
            maximum_api_visible_calls: 0,
            texture_bindings: translated.texture_bindings,
            global_buffers: translated.global_buffers,
        });
    }
    Ok(programs)
}

/// Maxwell permits a fragment program to read a generic attribute that the
/// preceding stage does not export, including holes within a partially written
/// vector. Specialize each disabled component before
/// applying the neutral backend's stricter stage-interface contract.
/// Ryujinx initializes each consumed output component to (0, 0, 0, 1):
/// https://git.axenov.dev/Museum/ryujinx/src/commit/17a1cab5d29787f3b41f40a69b530dae4e3cb25f/Ryujinx.Graphics.Shader/Translation/Translator.cs
/// Eden likewise defaults disabled/unwritten generic attributes:
/// https://github.com/eden-emulator/mirror/blob/master/src/shader_recompiler/backend/spirv/emit_spirv_context_get_set.cpp#L306-L315
/// https://github.com/eden-emulator/mirror/blob/master/src/shader_recompiler/backend/spirv/spirv_emit_context.cpp#L1563-L1578
fn specialize_unwritten_fragment_inputs(
    programs: &mut [TranslatedShaderIr],
) -> Vec<(ShaderIoLocation, u8)> {
    let Some(fragment) = programs
        .iter()
        .position(|program| program.ir.stage() == ShaderStage::Fragment)
    else {
        return Vec::new();
    };
    let Some(producer) = [
        ShaderStage::Geometry,
        ShaderStage::TessellationEvaluation,
        ShaderStage::Vertex,
    ]
    .into_iter()
    .find_map(|stage| programs.iter().find(|program| program.ir.stage() == stage)) else {
        return Vec::new();
    };
    let missing: Vec<_> = programs[fragment]
        .ir
        .inputs()
        .iter()
        .filter(|input| {
            matches!(input.location(), ShaderIoLocation::Generic(_))
                && input.scalar_type() == ShaderScalarType::Float32
                && !producer.ir.outputs().iter().any(|output| {
                    output.location() == input.location() && output.component() == input.component()
                })
        })
        .map(|input| (input.location(), input.component()))
        .collect();
    if missing.is_empty() {
        return missing;
    }
    let ir = &programs[fragment].ir;
    let mut instructions = Vec::with_capacity(ir.instructions().len());
    for instruction in ir.instructions() {
        let make = |operation| {
            ShaderInstruction::new(instruction.source(), instruction.predicate(), operation)
        };
        let constant = |destination, component| ShaderOperation::MoveImmediate32 {
            destination,
            bits: if component == 3 { 1.0_f32.to_bits() } else { 0 },
            scalar_type: ShaderScalarType::Float32,
        };
        match instruction.operation() {
            ShaderOperation::InterpolateInput {
                destination,
                location,
                component,
                ..
            } if missing.contains(&(*location, *component)) => {
                instructions.push(make(constant(*destination, *component)));
            }
            ShaderOperation::LoadInput {
                destinations,
                location,
                first_component,
                scalar_type,
            } if destinations.iter().enumerate().any(|(index, _)| {
                missing.contains(&(*location, *first_component + index as u8))
            }) =>
            {
                for (index, destination) in destinations.iter().enumerate() {
                    let component = *first_component + index as u8;
                    instructions.push(make(if missing.contains(&(*location, component)) {
                        constant(*destination, component)
                    } else {
                        ShaderOperation::LoadInput {
                            destinations: vec![*destination].into(),
                            location: *location,
                            first_component: component,
                            scalar_type: *scalar_type,
                        }
                    }));
                }
            }
            _ => instructions.push(instruction.clone()),
        }
    }
    programs[fragment].ir = ShaderIr::new(
        ir.stage(),
        ir.inputs()
            .iter()
            .filter(|input| !missing.contains(&(input.location(), input.component())))
            .copied()
            .collect(),
        ir.outputs().to_vec(),
        ir.resources().to_vec(),
        instructions,
    );
    missing
}

/// Maxwell records interpolation on fragment `IPA` inputs. Copy that linked
/// contract onto matching final pre-rasterization outputs before backend lowering so derived
/// rasterization paths can preserve the same interpolation planes.
pub(super) fn graphics_output_interpolation(
    programs: &[TranslatedShaderIr],
) -> Vec<((ShaderIoLocation, u8), ShaderInterpolation)> {
    programs
        .iter()
        .find(|program| program.ir.stage() == ShaderStage::Fragment)
        .map(|fragment| {
            fragment
                .ir
                .inputs()
                .iter()
                .filter_map(|input| {
                    input
                        .interpolation()
                        .map(|interpolation| ((input.location(), input.component()), interpolation))
                })
                .collect::<BTreeMap<_, _>>()
                .into_iter()
                .collect()
        })
        .unwrap_or_default()
}

/// Maxwell resource numbers are local to a shader binding group, whereas the
/// neutral backend exposes one descriptor namespace shared by all stages.
/// Allocate one stable host binding for each `(group, kind, local binding)` and
/// rewrite both the verified IR and texture metadata before WGSL lowering.
type GraphicsResourceIdentity = (u8, ShaderResourceKind, u8, u32);

fn graphics_resource_bindings(
    inputs: &MaxwellShaderTranslationInputs,
    programs: &[TranslatedShaderIr],
) -> Result<BTreeMap<GraphicsResourceIdentity, u8>, MaxwellShaderTranslationError> {
    let mut global = BTreeMap::<GraphicsResourceIdentity, u8>::new();
    for (input, program) in inputs.programs.iter().zip(programs) {
        let Some(group) = input.effective_group else {
            if program.ir.resources().is_empty() {
                continue;
            }
            return Err(MaxwellShaderTranslationError::IncompletePipelineBinding {
                pipeline: input.pipeline,
                field: "effective shader binding group",
            });
        };
        for resource in program.ir.resources() {
            let identity = graphics_resource_identity(group, program, resource)?;
            if global.contains_key(&identity) {
                continue;
            }
            let binding = u8::try_from(global.len())
                .map_err(|_| MaxwellShaderTranslationError::ResourceBindingExhausted)?;
            global.insert(identity, binding);
        }
    }
    Ok(global)
}

fn graphics_resource_identity(
    group: u8,
    program: &TranslatedShaderIr,
    resource: &ShaderResourceAccess,
) -> Result<GraphicsResourceIdentity, MaxwellShaderTranslationError> {
    if resource.kind() == ShaderResourceKind::StorageBuffer {
        let buffer = program
            .global_buffers
            .buffers
            .iter()
            .find(|buffer| buffer.binding == resource.binding().into())
            .ok_or(MaxwellShaderTranslationError::MissingResourceBindingRemap {
                stage: match program.ir.stage() {
                    ShaderStage::Vertex => MaxwellShaderStage::Vertex,
                    ShaderStage::Fragment => MaxwellShaderStage::Pixel,
                    ShaderStage::Geometry => MaxwellShaderStage::Geometry,
                    ShaderStage::TessellationControl => MaxwellShaderStage::TessellationInit,
                    ShaderStage::TessellationEvaluation => MaxwellShaderStage::Tessellation,
                    ShaderStage::Compute => MaxwellShaderStage::Compute,
                },
                binding: resource.binding(),
            })?;
        Ok((
            group,
            resource.kind(),
            buffer.constant_buffer,
            buffer.byte_offset,
        ))
    } else {
        Ok((group, resource.kind(), resource.binding(), 0))
    }
}

fn program_resource_bindings(
    input: &MaxwellShaderProgramTranslationInput,
    program: &TranslatedShaderIr,
    global: &BTreeMap<GraphicsResourceIdentity, u8>,
) -> Result<BTreeMap<u8, u8>, MaxwellShaderTranslationError> {
    let Some(group) = input.effective_group else {
        return Ok(BTreeMap::new());
    };
    program
        .ir
        .resources()
        .iter()
        .map(|resource| {
            global
                .get(&graphics_resource_identity(group, program, resource)?)
                .copied()
                .ok_or(MaxwellShaderTranslationError::MissingResourceBindingRemap {
                    stage: input.binary.stage(),
                    binding: resource.binding(),
                })
                .map(|binding| (resource.binding(), binding))
        })
        .collect()
}

pub(super) fn finalize_shader_ir(
    ir: ShaderIr,
    stage: MaxwellShaderStage,
    bindings: &BTreeMap<u8, u8>,
    output_interpolation: &[((ShaderIoLocation, u8), ShaderInterpolation)],
) -> Result<VerifiedShaderIr, MaxwellShaderTranslationError> {
    let resources = ir
        .resources()
        .iter()
        .map(|resource| {
            Ok(ShaderResourceAccess::new(
                remapped_binding(bindings, stage, resource.binding())?,
                resource.kind(),
                resource.readable(),
                resource.writable(),
            )
            .expect("remapping preserves non-empty resource access"))
        })
        .collect::<Result<Vec<_>, MaxwellShaderTranslationError>>()?;
    let instructions = ir
        .instructions()
        .iter()
        .map(|instruction| {
            let operation = match instruction.operation() {
                ShaderOperation::LoadConstantBuffer32 {
                    destination,
                    binding,
                    byte_offset,
                    scalar_type,
                } => ShaderOperation::LoadConstantBuffer32 {
                    destination: *destination,
                    binding: remapped_binding(bindings, stage, *binding)?,
                    byte_offset: *byte_offset,
                    scalar_type: *scalar_type,
                },
                ShaderOperation::LoadConstantBufferIndexed32 {
                    destination,
                    binding,
                    base_byte_offset,
                    dynamic_byte_offset,
                    scalar_type,
                } => ShaderOperation::LoadConstantBufferIndexed32 {
                    destination: *destination,
                    binding: remapped_binding(bindings, stage, *binding)?,
                    base_byte_offset: *base_byte_offset,
                    dynamic_byte_offset: *dynamic_byte_offset,
                    scalar_type: *scalar_type,
                },
                ShaderOperation::LoadStorageBuffer32 {
                    destination,
                    binding,
                    word_index,
                } => ShaderOperation::LoadStorageBuffer32 {
                    destination: *destination,
                    binding: remapped_binding(bindings, stage, *binding)?,
                    word_index: *word_index,
                },
                ShaderOperation::StoreStorageBuffer32 {
                    source,
                    binding,
                    word_index,
                } => ShaderOperation::StoreStorageBuffer32 {
                    source: *source,
                    binding: remapped_binding(bindings, stage, *binding)?,
                    word_index: *word_index,
                },
                ShaderOperation::QueryTextureLevels {
                    destination,
                    image_binding,
                } => ShaderOperation::QueryTextureLevels {
                    destination: *destination,
                    image_binding: remapped_binding(bindings, stage, *image_binding)?,
                },
                ShaderOperation::LoadTexture2D {
                    outputs,
                    coordinates,
                    image_binding,
                    mip_level,
                } => ShaderOperation::LoadTexture2D {
                    outputs: outputs.clone(),
                    coordinates: *coordinates,
                    image_binding: remapped_binding(bindings, stage, *image_binding)?,
                    mip_level: *mip_level,
                },
                ShaderOperation::SampleDepthCompare2D {
                    offset,
                    outputs,
                    coordinates,
                    array_index,
                    reference,
                    image_binding,
                    sampler_binding,
                    level_zero,
                } => ShaderOperation::SampleDepthCompare2D {
                    offset: *offset,
                    outputs: outputs.clone(),
                    coordinates: *coordinates,
                    array_index: *array_index,
                    reference: *reference,
                    image_binding: remapped_binding(bindings, stage, *image_binding)?,
                    sampler_binding: remapped_binding(bindings, stage, *sampler_binding)?,
                    level_zero: *level_zero,
                },
                ShaderOperation::SampleTexture2D {
                    offset,
                    outputs,
                    coordinates,
                    image_binding,
                    sampler_binding,
                    lod,
                } => ShaderOperation::SampleTexture2D {
                    offset: *offset,
                    outputs: outputs.clone(),
                    coordinates: *coordinates,
                    image_binding: remapped_binding(bindings, stage, *image_binding)?,
                    lod: *lod,
                    sampler_binding: remapped_binding(bindings, stage, *sampler_binding)?,
                },
                ShaderOperation::SampleTextureCube {
                    outputs,
                    coordinates,
                    image_binding,
                    sampler_binding,
                    lod,
                } => ShaderOperation::SampleTextureCube {
                    outputs: outputs.clone(),
                    coordinates: *coordinates,
                    image_binding: remapped_binding(bindings, stage, *image_binding)?,
                    sampler_binding: remapped_binding(bindings, stage, *sampler_binding)?,
                    lod: *lod,
                },
                ShaderOperation::SampleTexture2DArray {
                    offset,
                    outputs,
                    coordinates,
                    array_index,
                    image_binding,
                    sampler_binding,
                    lod,
                } => ShaderOperation::SampleTexture2DArray {
                    offset: *offset,
                    outputs: outputs.clone(),
                    coordinates: *coordinates,
                    array_index: *array_index,
                    image_binding: remapped_binding(bindings, stage, *image_binding)?,
                    lod: *lod,
                    sampler_binding: remapped_binding(bindings, stage, *sampler_binding)?,
                },
                operation => operation.clone(),
            };
            Ok(ShaderInstruction::new(
                instruction.source(),
                instruction.predicate(),
                operation,
            ))
        })
        .collect::<Result<Vec<_>, MaxwellShaderTranslationError>>()?;

    let interpolation = output_interpolation
        .iter()
        .copied()
        .collect::<BTreeMap<_, _>>();
    let outputs = ir
        .outputs()
        .iter()
        .map(|output| {
            ShaderInterfaceElement::new(
                output.location(),
                output.component(),
                output.scalar_type(),
                interpolation
                    .get(&(output.location(), output.component()))
                    .copied(),
            )
            .expect("linking preserves the decoded interface shape")
        })
        .collect();
    VerifiedShaderIr::verify(
        ShaderIr::new(
            ir.stage(),
            ir.inputs().to_vec(),
            outputs,
            resources,
            instructions,
        )
        .with_tessellation_control_points(ir.tessellation_control_points()),
    )
    .map_err(MaxwellShaderTranslationError::from)
}

fn remapped_binding(
    bindings: &BTreeMap<u8, u8>,
    stage: MaxwellShaderStage,
    binding: u8,
) -> Result<u8, MaxwellShaderTranslationError> {
    bindings
        .get(&binding)
        .copied()
        .ok_or(MaxwellShaderTranslationError::MissingResourceBindingRemap { stage, binding })
}

pub(super) fn validate_graphics_stage_interfaces(
    programs: &[TranslatedShaderIr],
) -> Result<(), MaxwellShaderTranslationError> {
    let mut producer: Option<&ShaderIr> = None;
    for stage in [
        ShaderStage::Vertex,
        ShaderStage::TessellationControl,
        ShaderStage::TessellationEvaluation,
        ShaderStage::Geometry,
        ShaderStage::Fragment,
    ] {
        let Some(consumer) = programs
            .iter()
            .find(|program| program.ir.stage() == stage)
            .map(|program| &program.ir)
        else {
            continue;
        };
        let Some(previous) = producer.replace(consumer) else {
            continue;
        };
        nixe_gpu::validate_shader_stage_link(previous, consumer).map_err(|error| {
            MaxwellShaderTranslationError::StageInterfaceMismatch {
                producer: error.producer,
                consumer: error.consumer,
                location: error.location,
                component: error.component,
                reason: error.reason,
            }
        })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;

fn remapped_oracle_metadata_binding(
    remap: &BTreeMap<u8, u8>,
    stage: MaxwellShaderStage,
    binding: u32,
) -> Result<u32, MaxwellShaderTranslationError> {
    let binding = u8::try_from(binding)
        .map_err(|_| MaxwellShaderTranslationError::ResourceBindingExhausted)?;
    remapped_binding(remap, stage, binding).map(u32::from)
}
