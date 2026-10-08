//! Graphics shader source identity, immutable translation inputs, and cache dependencies.

use super::binary::{
    MAXWELL_SCHEDULE_BUNDLE_SIZE, MAXWELL_SHADER_PROGRAM_HEADER_SIZE, MAXWELL_SHADER_READ_LIMIT,
    MaxwellShaderBinary, MaxwellShaderMemoryView, MaxwellStagedShaderWrite,
    canonical_shader_writes, read_shader_binary, validate_program_header,
};
use super::error::MaxwellShaderTranslationError;
use super::interface::neutral_stage;
use crate::{
    MAXWELL_PIPELINE_SHADER_COUNT, MAXWELL_VERTEX_ATTRIBUTE_COUNT, MaxwellGpuAddressSpace,
    MaxwellShaderStage, MaxwellThreeDState, MaxwellThreeDVertexNumericalType,
};
use nixe_gpu::{ShaderIoLocation, ShaderScalarType, ShaderStage};
use nixe_memory::CanonicalCpuWriteDependency;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// Immutable inputs captured before semantic translation and backend lowering.
///
/// Reading guest code is deliberately separate from translating it: callers
/// can compare this exact, versioned snapshot with a retained cache entry
/// before rebuilding IR or WGSL.
#[derive(Clone, Debug)]
pub(crate) struct MaxwellShaderTranslationInputs {
    pub(super) fingerprint: u128,
    pub(super) programs: Box<[Arc<MaxwellShaderProgramTranslationInput>]>,
}

impl MaxwellShaderTranslationInputs {
    pub(crate) const fn fingerprint(&self) -> u128 {
        self.fingerprint
    }
}

impl PartialEq for MaxwellShaderTranslationInputs {
    fn eq(&self, other: &Self) -> bool {
        self.programs == other.programs
    }
}

impl Eq for MaxwellShaderTranslationInputs {}

impl Hash for MaxwellShaderTranslationInputs {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.programs.hash(state);
    }
}

impl MaxwellShaderTranslationInputs {
    pub(crate) fn source_is_current(&self, address_space: &MaxwellGpuAddressSpace) -> bool {
        self.programs.iter().all(|program| {
            program
                .binary
                .source_mappings
                .iter()
                .all(|mapping| address_space.retained_mapping_is_current(mapping))
                && program
                    .binary
                    .source_cpu_writes
                    .iter()
                    .all(CanonicalCpuWriteDependency::remains_current)
        })
    }

    pub(crate) fn staged_writes_are_irrelevant(&self, writes: &[MaxwellStagedShaderWrite]) -> bool {
        writes.iter().all(|write| {
            self.programs.iter().all(|program| {
                let binary = &program.binary;
                let size = MAXWELL_SHADER_PROGRAM_HEADER_SIZE as u64
                    + binary.bundles.len() as u64 * MAXWELL_SCHEDULE_BUNDLE_SIZE as u64;
                let end = binary.address.saturating_add(size);
                write.address.saturating_add(4) <= binary.address || write.address >= end
            })
        })
    }
}

#[derive(Clone, Debug)]
pub(crate) struct MaxwellShaderTranslationSource {
    programs: Box<[MaxwellShaderSourceProgram]>,
    texture_constant_buffer_slot: Option<u8>,
    vertex_input_types: Box<[(ShaderIoLocation, ShaderScalarType)]>,
    staged_writes: Box<[MaxwellStagedShaderWrite]>,
}

impl PartialEq for MaxwellShaderTranslationSource {
    fn eq(&self, other: &Self) -> bool {
        self.programs == other.programs
            && self.texture_constant_buffer_slot == other.texture_constant_buffer_slot
            && self.vertex_input_types == other.vertex_input_types
            && self.staged_writes == other.staged_writes
    }
}

impl Eq for MaxwellShaderTranslationSource {}

impl Hash for MaxwellShaderTranslationSource {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.programs.len().hash(state);
        for program in &self.programs {
            program.hash(state);
        }
        self.texture_constant_buffer_slot.hash(state);
        self.vertex_input_types.len().hash(state);
        for input in &self.vertex_input_types {
            input.hash(state);
        }
        for write in &self.staged_writes {
            write.hash(state);
        }
        self.staged_writes.len().hash(state);
    }
}

/// Allocation-free semantic lookup key for the frontend shader-source cache.
/// Owned storage is materialized only after a miss or stale source snapshot.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MaxwellShaderTranslationSourceKey<'a> {
    programs: [Option<MaxwellShaderSourceProgram>; MAXWELL_PIPELINE_SHADER_COUNT],
    program_count: usize,
    texture_constant_buffer_slot: Option<u8>,
    vertex_input_types:
        [Option<(ShaderIoLocation, ShaderScalarType)>; MAXWELL_VERTEX_ATTRIBUTE_COUNT],
    vertex_input_count: usize,
    staged_writes: &'a [MaxwellStagedShaderWrite],
}

impl MaxwellShaderTranslationSourceKey<'_> {
    pub(crate) fn fingerprint(&self) -> u128 {
        nixe_gpu::cache_fingerprint(&self)
    }

    #[cfg(any(debug_assertions, test))]
    pub(crate) fn matches(&self, source: &MaxwellShaderTranslationSource) -> bool {
        self.programs().eq(source.programs.iter().copied())
            && self.texture_constant_buffer_slot == source.texture_constant_buffer_slot
            && self
                .vertex_input_types()
                .eq(source.vertex_input_types.iter().copied())
            && self
                .relevant_staged_writes()
                .eq(source.staged_writes.iter().copied())
    }

    pub(crate) fn materialize(&self) -> MaxwellShaderTranslationSource {
        MaxwellShaderTranslationSource {
            programs: self.programs().collect(),
            texture_constant_buffer_slot: self.texture_constant_buffer_slot,
            vertex_input_types: self.vertex_input_types().collect(),
            staged_writes: self.relevant_staged_writes().collect(),
        }
    }

    fn programs(&self) -> impl Iterator<Item = MaxwellShaderSourceProgram> + '_ {
        self.programs[..self.program_count]
            .iter()
            .map(|program| program.expect("bounded shader source key is densely populated"))
    }

    fn vertex_input_types(
        &self,
    ) -> impl Iterator<Item = (ShaderIoLocation, ShaderScalarType)> + '_ {
        self.vertex_input_types[..self.vertex_input_count]
            .iter()
            .map(|input| input.expect("bounded vertex-input key is densely populated"))
    }

    fn relevant_staged_writes(&self) -> impl Iterator<Item = MaxwellStagedShaderWrite> + '_ {
        self.staged_writes
            .iter()
            .copied()
            .filter(|write| self.shader_write_is_relevant(*write))
    }

    fn shader_write_is_relevant(&self, write: MaxwellStagedShaderWrite) -> bool {
        self.programs().any(|program| {
            let end = program
                .address
                .saturating_add(MAXWELL_SHADER_READ_LIMIT as u64);
            write.address < end && write.address.saturating_add(4) > program.address
        })
    }
}

impl Hash for MaxwellShaderTranslationSourceKey<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.program_count.hash(state);
        for program in self.programs() {
            program.hash(state);
        }
        self.texture_constant_buffer_slot.hash(state);
        self.vertex_input_count.hash(state);
        for input in self.vertex_input_types() {
            input.hash(state);
        }
        let mut count = 0_usize;
        for write in self.relevant_staged_writes() {
            write.hash(state);
            count += 1;
        }
        count.hash(state);
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct MaxwellShaderSourceProgram {
    pipeline: u8,
    stage: MaxwellShaderStage,
    address: u64,
    register_count: u8,
    effective_group: Option<u8>,
}

#[derive(Clone, Debug)]
pub(super) struct MaxwellShaderProgramTranslationInput {
    pub(super) pipeline: u8,
    pub(super) register_count: u8,
    pub(super) effective_group: Option<u8>,
    pub(super) texture_constant_buffer_slot: Option<u8>,
    pub(super) vertex_input_types: Box<[(ShaderIoLocation, ShaderScalarType)]>,
    pub(super) binary: MaxwellShaderBinary,
}

impl PartialEq for MaxwellShaderProgramTranslationInput {
    fn eq(&self, other: &Self) -> bool {
        self.register_count == other.register_count
            && self.effective_group == other.effective_group
            && self.texture_constant_buffer_slot == other.texture_constant_buffer_slot
            && self.vertex_input_types == other.vertex_input_types
            && self.binary == other.binary
    }
}

impl Eq for MaxwellShaderProgramTranslationInput {}

impl Hash for MaxwellShaderProgramTranslationInput {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.register_count.hash(state);
        self.effective_group.hash(state);
        self.texture_constant_buffer_slot.hash(state);
        self.vertex_input_types.hash(state);
        self.binary.hash(state);
    }
}

pub(crate) fn prepare_maxwell_shader_translation_source<'a>(
    state: &MaxwellThreeDState,
    staged_writes: &'a [MaxwellStagedShaderWrite],
) -> Result<MaxwellShaderTranslationSourceKey<'a>, MaxwellShaderTranslationError> {
    let bindings = state.shader_bindings();
    if !bindings
        .pipeline()
        .iter()
        .any(|pipeline| pipeline.enabled().value() == Some(&true))
    {
        return Err(MaxwellShaderTranslationError::MissingEnabledShader);
    }
    let program_region = bindings
        .program_region()
        .address()
        .ok_or(MaxwellShaderTranslationError::MissingProgramRegion)?
        .get();
    let mut programs = [None; MAXWELL_PIPELINE_SHADER_COUNT];
    let mut program_count = 0;
    for (pipeline_index, pipeline) in bindings.pipeline().iter().enumerate() {
        if pipeline.enabled().value() != Some(&true) {
            continue;
        }
        let pipeline_index = pipeline_index as u8;
        let stage = pipeline.stage().value().copied().ok_or(
            MaxwellShaderTranslationError::IncompletePipelineBinding {
                pipeline: pipeline_index,
                field: "SET_PIPELINE_SHADER stage",
            },
        )?;
        let offset = pipeline.program_offset().value().copied().ok_or(
            MaxwellShaderTranslationError::IncompletePipelineBinding {
                pipeline: pipeline_index,
                field: "SET_PIPELINE_PROGRAM",
            },
        )?;
        let register_count = pipeline.register_count().value().copied().ok_or(
            MaxwellShaderTranslationError::IncompletePipelineBinding {
                pipeline: pipeline_index,
                field: "SET_PIPELINE_REGISTER_COUNT",
            },
        )?;
        let address = program_region.checked_add(u64::from(offset)).ok_or(
            MaxwellShaderTranslationError::AddressOverflow {
                pipeline: pipeline_index,
            },
        )?;
        programs[program_count] = Some(MaxwellShaderSourceProgram {
            pipeline: pipeline_index,
            stage,
            address,
            register_count,
            effective_group: pipeline.effective_group(),
        });
        program_count += 1;
    }
    let mut vertex_input_types = [None; MAXWELL_VERTEX_ATTRIBUTE_COUNT];
    let mut vertex_input_count = 0;
    for input in maxwell_vertex_input_types_iter(state) {
        vertex_input_types[vertex_input_count] = Some(input);
        vertex_input_count += 1;
    }
    Ok(MaxwellShaderTranslationSourceKey {
        programs,
        program_count,
        texture_constant_buffer_slot: bindings
            .bindless_texture_constant_buffer_slot()
            .value()
            .copied(),
        vertex_input_types,
        staged_writes,
        vertex_input_count,
    })
}

pub(crate) fn prepare_maxwell_shader_translation_inputs_from_source(
    source: &MaxwellShaderTranslationSource,
    address_space: &MaxwellGpuAddressSpace,
) -> Result<MaxwellShaderTranslationInputs, MaxwellShaderTranslationError> {
    let mapping_generation = address_space.mapping_generation();
    let stage = source
        .programs
        .first()
        .ok_or(MaxwellShaderTranslationError::MissingEnabledShader)?
        .stage;
    let staged = canonical_shader_writes(address_space, &source.staged_writes, stage)?;
    let memory = MaxwellShaderMemoryView::new(address_space, &staged);
    let mut programs = Vec::with_capacity(source.programs.len());
    for program in &source.programs {
        let binary = read_shader_binary(&memory, program.stage, program.address)?;
        validate_program_header(program.stage, binary.header())?;
        let vertex_input_types: &[(ShaderIoLocation, ShaderScalarType)] =
            if neutral_stage(program.stage) == ShaderStage::Vertex {
                &source.vertex_input_types
            } else {
                &[]
            };
        programs.push(MaxwellShaderProgramTranslationInput {
            pipeline: program.pipeline,
            register_count: program.register_count,
            effective_group: program.effective_group,
            texture_constant_buffer_slot: source.texture_constant_buffer_slot,
            vertex_input_types: vertex_input_types.into(),
            binary,
        });
    }
    if mapping_generation != address_space.mapping_generation() {
        let program = source
            .programs
            .first()
            .expect("shader source contains at least one enabled program");
        return Err(MaxwellShaderTranslationError::SourceChangedDuringRead {
            stage: program.stage,
            address: program.address,
        });
    }
    let programs: Box<[_]> = programs.into_iter().map(Arc::new).collect();
    Ok(MaxwellShaderTranslationInputs {
        fingerprint: nixe_gpu::cache_fingerprint(&programs),
        programs,
    })
}

fn maxwell_vertex_input_types_iter(
    state: &MaxwellThreeDState,
) -> impl Iterator<Item = (ShaderIoLocation, ShaderScalarType)> + '_ {
    // The SPH input map describes component occupancy; the vertex attribute's
    // NUM_* field supplies the numerical interpretation. Field definitions:
    // https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/3d/cl9097.h#L1044-L1055
    state
        .vertex_input()
        .attributes()
        .iter()
        .enumerate()
        .filter_map(|(index, register)| {
            let format = register.value().copied()?;
            if !format.enabled() {
                return None;
            }
            let scalar_type = match format.numerical_type()? {
                MaxwellThreeDVertexNumericalType::SignedInteger => ShaderScalarType::Signed32,
                MaxwellThreeDVertexNumericalType::UnsignedInteger => ShaderScalarType::Unsigned32,
                MaxwellThreeDVertexNumericalType::SignedNormalized
                | MaxwellThreeDVertexNumericalType::UnsignedNormalized
                | MaxwellThreeDVertexNumericalType::UnsignedScaled
                | MaxwellThreeDVertexNumericalType::SignedScaled
                | MaxwellThreeDVertexNumericalType::Float => ShaderScalarType::Float32,
            };
            Some((
                ShaderIoLocation::Generic(
                    u8::try_from(index).expect("Maxwell vertex attribute count fits u8"),
                ),
                scalar_type,
            ))
        })
}

#[cfg(test)]
mod tests;
