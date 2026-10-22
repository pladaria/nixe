use super::*;
use nixe_gpu::shader_compilation::{Configuration, ShaderInputs, StageSource};
use std::sync::Arc;

/// A command-processor capture retains freshness proofs separately from worker inputs.
#[derive(Clone, Debug)]
pub struct CapturedProgram {
    pub inputs: Arc<ShaderInputs>,
    pub stages: Box<[CapturedStage]>,
}
impl CapturedProgram {
    pub fn retained_bytes(&self) -> usize {
        use nixe_gpu::shader_compilation::{environment_heap_bytes, info_heap_bytes};
        std::mem::size_of::<Self>()
            + self.inputs.retained_bytes()
            + self
                .stages
                .iter()
                .map(|stage| {
                    std::mem::size_of::<CapturedStage>()
                        + environment_heap_bytes(&stage.environment)
                        + info_heap_bytes(&stage.info)
                        + (stage.dependencies.len() + stage.code_dependencies.len())
                            * std::mem::size_of::<MaxwellShaderRead>()
                        + stage
                            .code_dependencies
                            .iter()
                            .chain(stage.dependencies.iter())
                            .map(|read| {
                                read.bytes.len()
                            + read.mappings.len() * std::mem::size_of::<crate::MaxwellGpuMapping>()
                            + read.physical.metadata_bytes()
                            // Shared page tracking and mapping references. This
                            // deliberately overcharges tiny capture reads.
                            + 512
                            })
                            .sum::<usize>()
                })
                .sum::<usize>()
    }

    pub fn refresh(
        &self,
        requests: &[(Option<u8>, CaptureRequest)],
        space: &MaxwellGpuMemory,
        memory: &mut MemoryCache,
        access: &mut nixe_gpu::MemoryAccess<'_>,
    ) -> Result<Option<Self>> {
        if requests.len() != self.stages.len() {
            return Ok(None);
        }
        let mut stages = Vec::with_capacity(self.stages.len());
        for (stage, (_, request)) in self.stages.iter().zip(requests) {
            let Some(stage) = stage.refresh(request, space, memory, access)? else {
                return Ok(None);
            };
            stages.push(stage);
        }
        Ok(Some(Self {
            inputs: Arc::clone(&self.inputs),
            stages: stages.into(),
        }))
    }
}

pub fn capture_program(
    compiler: &mut Compiler,
    space: &MaxwellGpuMemory,
    requests: impl IntoIterator<Item = (Option<u8>, CaptureRequest)>,
    configuration: &Configuration,
    memory: &mut MemoryCache,
    access: &mut nixe_gpu::MemoryAccess<'_>,
) -> Result<CapturedProgram> {
    compiler.reset()?;
    let mut stages = Vec::new();
    let mut inputs = Vec::new();
    let mut vertex_a = None;
    let mut previous = None;
    for (group, request) in requests {
        let stage = request.metadata.stage;
        let order = if stage == Stage::VertexA {
            0
        } else {
            stage as u32 + 1
        };
        if previous.is_some_and(|old| old >= order)
            || (stage == Stage::Compute && !stages.is_empty())
        {
            return Err(invalid(
                "Capture stages must be unique and ordered, with compute isolated",
            ));
        }
        previous = Some(order);
        let (captured, id) = capture(
            compiler,
            space,
            request,
            &configuration.host,
            &configuration.settings,
            memory,
            access,
        )?;
        let mut info = (*captured.info).clone();
        if stage == Stage::VertexA {
            vertex_a = Some((id, group));
        } else if stage == Stage::VertexB
            && let Some((a, a_group)) = vertex_a.take()
        {
            if a_group != group {
                return Err(invalid(
                    "Dual vertex stages require the same constant-buffer group",
                ));
            }
            let merged = compiler.merge_dual_vertex(
                a,
                id,
                captured.environment.as_ref(),
                &configuration.settings,
            )?;
            info = compiler.inspect_program(
                merged,
                nixe_gpu::shader_compilation::resources::resource_info,
            )?;
        }
        inputs.push(StageSource {
            environment: (*captured.environment).clone(),
            info,
            binding_group: group,
        });
        stages.push(captured);
    }
    if stages.is_empty() || vertex_a.is_some() {
        return Err(invalid(
            "Capture requires an emitted stage and Vertex A requires Vertex B",
        ));
    }
    Ok(CapturedProgram {
        inputs: Arc::new(ShaderInputs::new(inputs.into(), configuration)),
        stages: stages.into(),
    })
}

impl CapturedStage {
    /// A changed location requires a fresh memory proof even for identical code.
    pub fn request_is_current(&self, request: &CaptureRequest, space: &MaxwellGpuMemory) -> bool {
        self.address == request.address
            && self.request_metadata == request.metadata
            && self.is_current(space, &request.bindings)
    }

    pub fn memory_is_current(&self, space: &MaxwellGpuMemory) -> bool {
        self.code_dependencies
            .iter()
            .chain(self.dependencies.iter())
            .all(|read| read.is_current(space))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linked_capture_merges_vertex_metadata_and_replays_after_scratch_reset() {
        let (allocation, space, base) = crate::shader::test_support::mapped_memory();
        for offset in [0, 0x100, 0x200] {
            crate::test_memory::write(
                &allocation,
                offset + 0x50,
                &0xe300_0000_0007_000fu64.to_le_bytes(),
            )
            .unwrap();
        }
        let configuration = Configuration {
            profile: nixe_shader_recompiler::Profile {
                supported_spirv: 0x10500,
                unified_descriptor_binding: true,
                ..Default::default()
            },
            ..Default::default()
        };
        let request = |offset, stage| CaptureRequest {
            address: base + offset,
            metadata: EnvironmentMetadata {
                stage,
                ..Default::default()
            },
            bindings: CaptureBindings {
                viewport_transform: Some(1),
                ..Default::default()
            },
        };
        let mut compiler = Compiler::new().unwrap();

        let capture = capture_program(
            &mut compiler,
            space.memory(),
            [
                (Some(0), request(0, Stage::VertexA)),
                (Some(0), request(0x100, Stage::VertexB)),
                (Some(4), request(0x200, Stage::Fragment)),
            ],
            &configuration,
            &mut MemoryCache::default(),
            &mut crate::test_memory::access(space.memory().machine()),
        )
        .unwrap();
        assert_eq!(capture.inputs.stages.len(), 3);
        assert!(capture.stages[0].request_is_current(&request(0, Stage::VertexA), space.memory()));
        assert!(
            !capture.stages[0].request_is_current(&request(0x100, Stage::VertexA), space.memory())
        );
        compiler.reset().unwrap();
        let output = capture
            .inputs
            .compile(&mut compiler, &configuration, Some(&Default::default()))
            .unwrap();
        assert_eq!(output.len(), 2);
        assert_eq!(output[0].stage, Stage::VertexB);
        assert_eq!(
            output[0].execution.provenance,
            nixe_shader_recompiler::artifact::StageProvenance::MergedVertex
        );
        assert_eq!(output[1].stage, Stage::Fragment);
        crate::shader::test_support::write_shader_words(space.memory(), &[(base + 0x50, 0)]);
        assert!(!capture.stages[0].is_current(space.memory(), &capture.stages[0].bindings));
        super::super::tests::refresh_unchanged(&capture.stages[1], space.memory());
    }
}
