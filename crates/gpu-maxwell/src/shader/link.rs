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
    ShaderStage, VerifiedShaderIr,
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
        )?);
    }

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
        for texture in &mut translated.texture_bindings {
            texture.image_binding =
                remapped_binding(&local_bindings, stage, texture.image_binding)?;
            texture.sampler_binding = texture
                .sampler_binding
                .map(|binding| remapped_binding(&local_bindings, stage, binding))
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
        });
    }
    Ok(programs)
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
fn graphics_resource_bindings(
    inputs: &MaxwellShaderTranslationInputs,
    programs: &[TranslatedShaderIr],
) -> Result<BTreeMap<(u8, ShaderResourceKind, u8), u8>, MaxwellShaderTranslationError> {
    let mut global = BTreeMap::<(u8, ShaderResourceKind, u8), u8>::new();
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
            let identity = (group, resource.kind(), resource.binding());
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

fn program_resource_bindings(
    input: &MaxwellShaderProgramTranslationInput,
    program: &TranslatedShaderIr,
    global: &BTreeMap<(u8, ShaderResourceKind, u8), u8>,
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
                .get(&(group, resource.kind(), resource.binding()))
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
                ShaderOperation::SampleTexture2D {
                    outputs,
                    coordinates,
                    image_binding,
                    sampler_binding,
                } => ShaderOperation::SampleTexture2D {
                    outputs: outputs.clone(),
                    coordinates: *coordinates,
                    image_binding: remapped_binding(bindings, stage, *image_binding)?,
                    sampler_binding: remapped_binding(bindings, stage, *sampler_binding)?,
                },
                ShaderOperation::SampleTexture2DArray {
                    outputs,
                    coordinates,
                    array_index,
                    image_binding,
                    sampler_binding,
                } => ShaderOperation::SampleTexture2DArray {
                    outputs: outputs.clone(),
                    coordinates: *coordinates,
                    array_index: *array_index,
                    image_binding: remapped_binding(bindings, stage, *image_binding)?,
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
