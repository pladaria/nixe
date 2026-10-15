//! Retained frontend facts for native descriptor preparation. These are guest
//! identities, not Vulkan bindings or old renderer modules.
use super::*;
use nixe_gpu::ShaderResourceKind;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum MaxwellSsaBindingSource {
    ConstantBuffer {
        slot: u8,
    },
    GlobalBuffer {
        constant_buffer: u8,
        byte_offset: u32,
        runtime_extent: bool,
    },
    Image {
        handle_byte_offset: u32,
        kind: ShaderResourceKind,
    },
    Sampler {
        handle_byte_offset: u32,
        comparison: bool,
    },
}
impl MaxwellSsaBindingSource {
    fn identity(self) -> (ShaderResourceKind, u8, u32) {
        match self {
            Self::ConstantBuffer { slot } => (ShaderResourceKind::ConstantBuffer, slot, 0),
            Self::GlobalBuffer {
                constant_buffer,
                byte_offset,
                ..
            } => (
                ShaderResourceKind::StorageBuffer,
                constant_buffer,
                byte_offset,
            ),
            Self::Image {
                handle_byte_offset,
                kind,
            } => (kind, 0, handle_byte_offset),
            Self::Sampler {
                handle_byte_offset,
                comparison,
            } => (
                if comparison {
                    ShaderResourceKind::ComparisonSampler
                } else {
                    ShaderResourceKind::Sampler
                },
                0,
                handle_byte_offset,
            ),
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellSsaBinding {
    pub resource: ResourceId,
    pub source: MaxwellSsaBindingSource,
    pub binding_group: Option<u8>,
    pub texture_constant_buffer_slot: Option<u8>,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxwellSsaSource {
    pub stage: MaxwellShaderStage,
    pub address: u64,
    pub pc_base: u32,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaxwellSsaProgram {
    pub function: Function,
    pub bindings: Box<[MaxwellSsaBinding]>,
    pub stage: MaxwellShaderStage,
    pub sources: Box<[MaxwellSsaSource]>,
    global_buffers: Box<
        [(
            Option<u8>,
            super::super::global_memory::GlobalBufferBindings,
        )],
    >,
}
impl MaxwellSsaProgram {
    pub(in crate::shader) fn new(
        binary: &MaxwellShaderBinary,
        decoded: super::super::translate::DecodedProgram,
    ) -> Result<Self, MaxwellShaderTranslationError> {
        let mut bindings = Vec::new();
        for resource in &decoded.resources {
            let source = match resource.kind() {
                ShaderResourceKind::ConstantBuffer => MaxwellSsaBindingSource::ConstantBuffer {
                    slot: u8::try_from(resource.binding())
                        .expect("ISA constant buffer slot is bounded"),
                },
                ShaderResourceKind::StorageBuffer => {
                    let buffer = decoded
                        .global_buffers
                        .buffers
                        .iter()
                        .find(|b| b.binding == resource.binding())
                        .expect("decoded global buffer identity");
                    MaxwellSsaBindingSource::GlobalBuffer {
                        constant_buffer: buffer.constant_buffer,
                        byte_offset: buffer.byte_offset,
                        runtime_extent: buffer.runtime_extent,
                    }
                }
                kind if matches!(
                    kind,
                    ShaderResourceKind::Sampler | ShaderResourceKind::ComparisonSampler
                ) =>
                {
                    let texture = decoded
                        .texture_bindings
                        .iter()
                        .find(|t| t.sampler_binding == Some(resource.binding()))
                        .expect("decoded sampler identity");
                    MaxwellSsaBindingSource::Sampler {
                        handle_byte_offset: texture.constant_buffer_byte_offset,
                        comparison: kind == ShaderResourceKind::ComparisonSampler,
                    }
                }
                kind => {
                    let texture = decoded
                        .texture_bindings
                        .iter()
                        .find(|t| t.image_binding == resource.binding())
                        .expect("decoded image identity");
                    MaxwellSsaBindingSource::Image {
                        handle_byte_offset: texture.constant_buffer_byte_offset,
                        kind,
                    }
                }
            };
            bindings.push(MaxwellSsaBinding {
                resource: ResourceId::new(resource.binding()),
                source,
                binding_group: None,
                texture_constant_buffer_slot: None,
            });
        }
        let global_buffers = decoded.global_buffers.clone();
        let function = lower_decoded(binary, decoded)?;
        let mut result = Self {
            function,
            bindings: bindings.into_boxed_slice(),
            stage: binary.stage(),
            sources: vec![MaxwellSsaSource {
                stage: binary.stage(),
                address: binary.address,
                pc_base: 0,
            }]
            .into_boxed_slice(),
            global_buffers: vec![(None, global_buffers)].into_boxed_slice(),
        };
        result.retain_live_bindings();
        Ok(result)
    }
    /// Map a linked SSA diagnostic back to its actual Maxwell binary/PC.
    pub fn locate_source(
        &self,
        location: ShaderSourceLocation,
    ) -> Option<(MaxwellShaderStage, u64, u32)> {
        let source = self
            .sources
            .iter()
            .rev()
            .find(|s| location.byte_offset() >= s.pc_base)?;
        Some((
            source.stage,
            source.address,
            location.byte_offset() - source.pc_base,
        ))
    }
    pub(super) fn append_buffers(&mut self, other: &Self) {
        self.global_buffers = self
            .global_buffers
            .iter()
            .chain(other.global_buffers.iter())
            .cloned()
            .collect();
    }
    pub(super) fn retain_live_bindings(&mut self) {
        let mut live = std::collections::BTreeSet::new();
        let mut inputs = std::collections::BTreeSet::new();
        for block in &self.function.blocks {
            for inst in &block.instructions {
                match inst.op {
                    Op::Input {
                        location,
                        component,
                        ..
                    }
                    | Op::Interpolate {
                        location,
                        component,
                        ..
                    }
                    | Op::ControlPointLoad {
                        location,
                        component,
                        output: false,
                    } => {
                        inputs.insert((location, component));
                    }
                    _ => {}
                }
            }
        }
        self.function
            .inputs
            .retain(|e| inputs.contains(&(e.location(), u32::from(e.component()))));
        for block in &self.function.blocks {
            for inst in &block.instructions {
                match inst.op {
                    Op::Load {
                        resource: Some(id), ..
                    }
                    | Op::Store {
                        resource: Some(id), ..
                    }
                    | Op::TextureLevels { image: id } => {
                        live.insert(id);
                    }
                    Op::TextureSample { image, sampler, .. } => {
                        live.insert(image);
                        if let Some(sampler) = sampler {
                            live.insert(sampler);
                        }
                    }
                    _ => {}
                }
            }
        }
        let mut bindings = BTreeMap::<ResourceId, MaxwellSsaBinding>::new();
        for binding in self
            .bindings
            .iter()
            .copied()
            .filter(|b| live.contains(&b.resource))
        {
            if let Some(old) = bindings.get_mut(&binding.resource) {
                if let (
                    MaxwellSsaBindingSource::GlobalBuffer { runtime_extent, .. },
                    MaxwellSsaBindingSource::GlobalBuffer {
                        runtime_extent: extra,
                        ..
                    },
                ) = (&mut old.source, binding.source)
                {
                    *runtime_extent |= extra;
                }
            } else {
                bindings.insert(binding.resource, binding);
            }
        }
        self.bindings = bindings.into_values().collect();
        self.function.resources.retain(|r| live.contains(&r.id));
    }
    /// Captured affine LDG/STG bounds, evaluated for the consumed dispatch.
    /// Descriptor-backed buffers retain their declared guest extent.
    pub fn global_buffer_extents(
        &self,
        workgroup_size: [u32; 3],
        group_count: [u32; 3],
    ) -> Box<[(ResourceId, u64)]> {
        let extents: Vec<_> = self
            .global_buffers
            .iter()
            .map(|(group, buffers)| {
                (
                    *group,
                    buffers,
                    buffers.byte_extents(group_count, workgroup_size),
                )
            })
            .collect();
        self.bindings
            .iter()
            .filter_map(|binding| {
                let MaxwellSsaBindingSource::GlobalBuffer {
                    constant_buffer,
                    byte_offset,
                    ..
                } = binding.source
                else {
                    return None;
                };
                let extent = extents
                    .iter()
                    .filter(|(group, _, _)| *group == binding.binding_group)
                    .filter_map(|(_, buffers, extents)| {
                        let index = buffers.buffers.iter().position(|b| {
                            b.constant_buffer == constant_buffer && b.byte_offset == byte_offset
                        })?;
                        Some(extents[index])
                    })
                    .max()?;
                Some((binding.resource, extent))
            })
            .collect()
    }

    /// Live constant loads with proven byte addresses need only their maximum
    /// consumed end. An unproved address keeps the whole declared guest binding;
    /// shader code is inspected once at compilation, never on the draw path.
    pub(crate) fn constant_buffer_extents(&self) -> BTreeMap<ResourceId, Option<u64>> {
        let mut extents: BTreeMap<_, _> = self
            .bindings
            .iter()
            .filter_map(|binding| {
                matches!(
                    binding.source,
                    MaxwellSsaBindingSource::ConstantBuffer { .. }
                )
                .then_some((binding.resource, Some(0)))
            })
            .collect();
        for instruction in self.function.blocks.iter().flat_map(|b| &b.instructions) {
            let Op::Load {
                space: MemorySpace::Constant,
                resource: Some(resource),
                address_type,
                ty,
            } = instruction.op
            else {
                continue;
            };
            let Some(extent) = extents.get_mut(&resource) else {
                continue;
            };
            let nixe_gpu::ssa::Definition::Instruction { block, index, .. } =
                self.function.values[instruction.arguments[0].index()].definition
            else {
                *extent = None;
                continue;
            };
            let end = match self.function.blocks[block.index()].instructions[index as usize].op {
                Op::Constant(c) if c.ty == Type::U32 && address_type == Type::U32 => {
                    let bytes = match ty {
                        Type::U32 | Type::I32 | Type::F32 => Some(4),
                        Type::U64 | Type::I64 | Type::F64 => Some(8),
                        _ => None,
                    };
                    bytes.and_then(|bytes| c.bits.checked_add(bytes))
                }
                _ => None,
            };
            *extent = match (*extent, end) {
                (Some(old), Some(end)) => Some(old.max(end)),
                _ => None,
            };
        }
        extents
    }
}
/// Linked native stages, ready for SSA-derived SPIR-V and pipeline cache keys.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MaxwellSsaPipeline {
    pub programs: Box<[MaxwellSsaProgram]>,
    inputs: super::super::source::MaxwellShaderSnapshot,
}
impl MaxwellSsaPipeline {
    pub fn source_identity(&self) -> nixe_gpu::ssa::artifact::ContentId {
        super::cache::snapshot_identity(&self.inputs)
    }
    pub fn source_is_current(&self, address_space: &MaxwellGpuAddressSpace) -> bool {
        self.inputs.source_is_current(address_space)
    }
    pub fn staged_writes_are_irrelevant(&self, writes: &CanonicalWriteBatch) -> bool {
        self.inputs.staged_writes_are_irrelevant(writes)
    }
    pub fn indirect_bindings_current(
        &self,
        bindings: &crate::MaxwellThreeDShaderBindingState,
    ) -> bool {
        self.inputs.indirect_bindings_current(bindings)
    }
}
/// Consume the actual graphics source state on a cache miss. Snapshot capture,
/// including proven BRX tables, is shared frontend work, not an old-IR import.
pub fn compile_maxwell_pipeline_ssa(
    state: &crate::MaxwellThreeDState,
    address_space: &MaxwellGpuAddressSpace,
    writes: &CanonicalWriteBatch,
) -> Result<MaxwellSsaPipeline, MaxwellShaderTranslationError> {
    let source =
        super::super::source::prepare_maxwell_shader_translation_source(state)?.materialize();
    let inputs =
        super::super::source::capture_maxwell_shader_snapshot(&source, address_space, writes)?;
    compile_snapshot(inputs)
}
pub(super) fn compile_snapshot(
    inputs: super::super::source::MaxwellShaderSnapshot,
) -> Result<MaxwellSsaPipeline, MaxwellShaderTranslationError> {
    let mut programs = Vec::new();
    for input in &inputs.programs {
        let types = input.vertex_input_types.iter().copied().collect();
        let decoded = super::super::translate::decode_shader_binary(
            &input.binary,
            input.register_count,
            &types,
            input.y_negate,
        )?;
        programs.push(MaxwellSsaProgram::new(&input.binary, decoded)?);
    }
    // Namespace live resource identities by actual guest binding group and
    // descriptor address. Local texture enumeration order is not an identity.
    let mut identities = BTreeMap::new();
    for (program, input) in programs.iter_mut().zip(&inputs.programs) {
        for (group, _) in &mut program.global_buffers {
            *group = input.effective_group;
        }
        let mut remap = BTreeMap::new();
        for binding in &mut program.bindings {
            binding.binding_group = input.effective_group;
            binding.texture_constant_buffer_slot = input.texture_constant_buffer_slot;
            let group = if input.binary.stage() == MaxwellShaderStage::Compute {
                0
            } else {
                input.effective_group.ok_or(
                    MaxwellShaderTranslationError::IncompletePipelineBinding {
                        pipeline: input.pipeline,
                        field: "effective shader binding group",
                    },
                )?
            };
            let texture_slot = if matches!(
                binding.source,
                MaxwellSsaBindingSource::Image { .. } | MaxwellSsaBindingSource::Sampler { .. }
            ) {
                Some(input.texture_constant_buffer_slot.ok_or(
                    MaxwellShaderTranslationError::IncompletePipelineBinding {
                        pipeline: input.pipeline,
                        field: "SET_BINDLESS_TEXTURE_CONSTANT_BUFFER_SLOT",
                    },
                )?)
            } else {
                None
            };
            let next = ResourceId::new(identities.len() as u32);
            let id = *identities
                .entry((group, texture_slot, binding.source.identity()))
                .or_insert(next);
            remap.insert(binding.resource, id);
            binding.resource = id;
        }
        for block in &mut program.function.blocks {
            for inst in &mut block.instructions {
                match &mut inst.op {
                    Op::Load {
                        resource: Some(id), ..
                    }
                    | Op::Store {
                        resource: Some(id), ..
                    }
                    | Op::TextureLevels { image: id } => *id = remap[id],
                    Op::TextureSample { image, sampler, .. } => {
                        *image = remap[image];
                        if let Some(sampler) = sampler {
                            *sampler = remap[sampler];
                        }
                    }
                    _ => {}
                }
            }
        }
        for resource in &mut program.function.resources {
            resource.id = remap[&resource.id];
        }
    }
    super::link::link(&mut programs)?;
    Ok(MaxwellSsaPipeline {
        programs: programs.into_boxed_slice(),
        inputs,
    })
}
