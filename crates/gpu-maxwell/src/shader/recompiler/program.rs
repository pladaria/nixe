use super::*;
use nixe_gpu::shader_compilation::{Configuration, ShaderInputs, StageSource};

/// Capture only when creating a pipeline. No mapping leases or per-draw
/// freshness records accompany these owned compiler/file environments.
pub fn capture_program(
    compiler: &mut Compiler,
    space: &MaxwellGpuMemory,
    requests: impl IntoIterator<Item = (Option<u8>, CaptureRequest, usize)>,
    configuration: &Configuration,
    access: &mut nixe_gpu::MemoryAccess<'_>,
) -> Result<Arc<ShaderInputs>> {
    nixe_gpu::trace::event("shader_capture", 1, 0, 0);
    let _trace = nixe_gpu::trace::span("shader_capture_translate", 0, 0, 0);
    compiler.reset()?;
    let mut stage_count = 0;
    let mut inputs = Vec::new();
    let mut vertex_a = None;
    let mut previous = None;
    for (group, request, size) in requests {
        let stage = request.metadata.stage;
        let order = if stage == Stage::VertexA {
            0
        } else {
            stage as u32 + 1
        };
        if previous.is_some_and(|old| old >= order) || (stage == Stage::Compute && stage_count != 0)
        {
            return Err(invalid(
                "Capture stages must be unique and ordered, with compute isolated",
            ));
        }
        previous = Some(order);
        let mut source = CaptureEnvironment::new(space, request, Some(size), access)?;
        let mut recorder = RecordingEnvironment::new(&mut source);
        let id = compiler.translate(&mut recorder, &configuration.host, &configuration.settings)?;
        let environment = Arc::new(recorder.into_snapshot());
        let mut info =
            Arc::new(compiler.inspect_program(id, |_, program| Ok(program.info.clone()))?);
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
            let merged =
                compiler.merge_dual_vertex(a, id, environment.as_ref(), &configuration.settings)?;
            info =
                Arc::new(compiler.inspect_program(merged, |_, program| Ok(program.info.clone()))?);
        }
        inputs.push(StageSource {
            environment,
            info,
            binding_group: group,
        });
        stage_count += 1;
    }
    if stage_count == 0 || vertex_a.is_some() {
        return Err(invalid(
            "Capture requires an emitted stage and Vertex A requires Vertex B",
        ));
    }
    Ok(Arc::new(ShaderInputs::new(inputs.into(), configuration)))
}
