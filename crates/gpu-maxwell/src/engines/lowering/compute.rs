//! QMD dispatch lowering shares all resource identities with graphics.
use super::*;
use crate::MaxwellGpuAddressSpace;
use crate::engines::compute::MaxwellResolvedComputeLaunch;
use crate::shader::MaxwellComputeProgram;
use nixe_gpu::BackingView;
use nixe_memory::CanonicalWriteBatch;

#[derive(Debug)]
pub(super) struct ComputeShaderRecord {
    key: (u64, u8, [u32; 3]),
    program: Arc<MaxwellComputeProgram>,
    id: ShaderId,
}

impl MaxwellLoweringCache {
    pub(crate) fn lower_compute(
        &mut self,
        launch: &MaxwellResolvedComputeLaunch,
        address_space: &MaxwellGpuAddressSpace,
        writes: &CanonicalWriteBatch,
        frontend: FrontendSubmissionId,
        predecessors: Vec<FrontendSubmissionId>,
    ) -> Result<MaxwellLoweredWork, MaxwellLoweringError> {
        use MaxwellLoweringError as E;
        launch.validate_execution().map_err(E::ComputeLaunch)?;
        let key = launch.kernel_key();
        let fingerprint = nixe_gpu::cache_fingerprint(&key);
        let mut creations = Vec::new();
        let mut invalidations = std::mem::take(&mut self.retired_resources);
        let cached = self
            .compute_shaders
            .get(fingerprint)
            .filter(|record| {
                record.key == key && record.program.source_is_current(address_space, writes)
            })
            .map(|record| (record.id, Arc::clone(&record.program)));
        let (shader, program) = if let Some(cached) = cached {
            cached
        } else {
            let program = Arc::new(
                launch
                    .translate_kernel(address_space, writes)
                    .map_err(E::ComputeShader)?,
            );
            let id = ShaderId::new(take_identity(self)?);
            if let Some(old) = self.compute_shaders.get(fingerprint) {
                invalidations.push(ResourceDependency::Shader(old.id));
            }
            creations.push(BackendResourceCreateInfo::Shader {
                id,
                description: ShaderDescription {
                    stage: ShaderStage::Compute,
                },
                module: program.module.clone(),
            });
            self.compute_shaders.replace(
                fingerprint,
                ComputeShaderRecord {
                    key,
                    program: Arc::clone(&program),
                    id,
                },
            );
            while self.compute_shaders.len() > self.configuration.shader_entries() {
                let (_, old) = self.compute_shaders.remove_lru();
                invalidations.push(ResourceDependency::Shader(old.id));
            }
            (id, program)
        };
        let resources = launch
            .resolve_resources(&program, address_space, writes)
            .map_err(E::ComputeLaunch)?;
        let mut bindings = Vec::with_capacity(resources.len());
        let mut dependencies = Vec::with_capacity(resources.len());
        let mut accesses = Vec::with_capacity(resources.len());
        let mut retained = Vec::<(BackingView, bool)>::with_capacity(resources.len());
        for (binding, range) in resources {
            let buffer::RetainedBuffer {
                description,
                allocation_description,
                backing,
                mappings,
            } = buffer::retain(&range)?;
            let resource = program
                .module
                .ir()
                .ir()
                .resources()
                .iter()
                .find(|r| r.binding() == binding)
                .expect("resolved shader binding");
            let storage = resource.kind() == ShaderResourceKind::StorageBuffer;
            // Separate backend buffers cannot preserve intra-dispatch aliases.
            // Cross-dispatch aliases are reconciled by canonical ownership.
            if retained
                .iter()
                .any(|(other, writable)| (storage || *writable) && backing.overlaps(other))
            {
                return Err(E::BufferBacking(
                    "overlapping writable compute bindings require a shared host buffer".into(),
                ));
            }
            retained.push((backing.clone(), storage));
            let dependency = buffer::prepare_buffer(
                description,
                allocation_description,
                backing,
                mappings,
                self,
                &mut creations,
                &mut invalidations,
                dependencies.iter().copied(),
            )?;
            let buffer = buffer_dependency(dependency)?;
            accesses.push(ResourceAccess::new(
                AccessTarget::Buffer {
                    buffer,
                    range: BufferRange::new(0, range.size()).map_err(|_| E::ResourceExhausted)?,
                },
                AccessScope::new(
                    PipelineStages::COMPUTE_SHADER,
                    if storage {
                        AccessMode::Write
                    } else {
                        AccessMode::Read
                    },
                    if storage {
                        ResourceUsage::StorageBuffer
                    } else {
                        ResourceUsage::UniformBuffer
                    },
                )
                .map_err(|_| E::InvalidTransition)?,
            ));
            bindings.push(DescriptorTableBinding {
                binding,
                resource: dependency,
            });
            dependencies.push(dependency);
        }
        let kinds = vec![DescriptorKind::Buffer; bindings.len()];
        let tables = prepare_descriptors(bindings, kinds, self, &mut creations)?;
        let pipeline = if let Some(pipeline) = self.compute_pipeline {
            pipeline
        } else {
            let id = PipelineId::new(take_identity(self)?);
            creations.push(BackendResourceCreateInfo::Pipeline {
                id,
                description: PipelineDescription {
                    kind: PipelineKind::Compute,
                },
            });
            self.compute_pipeline = Some(id);
            id
        };
        let dispatch =
            nixe_gpu::DispatchOperation::new(pipeline, shader, tables, launch.workgroups())
                .map_err(E::Command)?;
        let operation = GpuOperation::new(
            GpuCommand::Dispatch(dispatch),
            accesses,
            dependencies,
            CapabilityRequirements::new([nixe_gpu::CapabilityRequirement::ShaderStage(
                ShaderStage::Compute,
            )]),
        );
        finish_lowered_work(
            self,
            frontend,
            predecessors,
            creations,
            invalidations,
            [operation],
            Arc::from([]),
        )
    }
}
