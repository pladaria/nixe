use super::*;

impl MaxwellNativeFrontend {
    pub(super) fn capture_compute(
        &mut self,
        launch: &super::super::super::compute::MaxwellResolvedComputeLaunch,
        space: &MaxwellGpuMemory,
        access: &mut nixe_gpu::MemoryAccess<'_>,
    ) -> Result<Arc<crate::recompiler::CapturedProgram>, Error> {
        let request = launch.capture_request();
        let current = |capture: &crate::recompiler::CapturedProgram| {
            capture.stages[0].request_is_current(&request, space)
        };
        if let Some(capture) = &self.channel.compute
            && current(capture)
        {
            return Ok(Arc::clone(capture));
        }
        let m = &request.metadata;
        let key = (
            space.id(),
            request.address,
            m.workgroup_size,
            m.local_memory_size,
            m.shared_memory_size,
            m.texture_bound_buffer.0,
        );
        let cached = self.compute_sources.get(&key).cloned();
        let refreshed = match &cached {
            Some(capture) if !current(capture) => capture
                .refresh(
                    &[(None, launch.capture_request())],
                    space,
                    &mut self.capture_memory,
                    access,
                )
                .map_err(compilation)?,
            _ => None,
        };
        let capture = if let Some(capture) = cached.filter(|capture| current(capture)) {
            capture
        } else if let Some(capture) = refreshed {
            self.statistics.revalidated_captures += 1;
            let capture = Arc::new(capture);
            self.compute_sources
                .insert(key, Arc::clone(&capture), capture.retained_bytes());
            capture
        } else {
            let requests = [(None, request)];
            let restored = self.capture_archive.as_ref().and_then(|archive| {
                archive.restore(&requests, space, &mut self.capture_memory, access)
            });
            let capture = if let Some(capture) = restored {
                self.statistics.restored_captures += 1;
                Arc::new(capture)
            } else {
                if self.capture_compiler.is_none() {
                    self.capture_compiler =
                        Some(nixe_shader_recompiler::Compiler::new().map_err(compilation)?);
                }
                let capture = Arc::new(
                    crate::recompiler::capture_program(
                        self.capture_compiler.as_mut().unwrap(),
                        space,
                        requests,
                        &self.configuration,
                        &mut self.capture_memory,
                        access,
                    )
                    .map_err(compilation)?,
                );
                self.statistics.shader_captures += 1;
                capture
            };
            if let Some(archive) = &mut self.capture_archive {
                archive.remember(&capture.inputs);
            }
            if self.captured_codes.get(&capture.inputs.identity).is_none() {
                self.captured_codes.insert(
                    capture.inputs.identity,
                    Arc::clone(&capture.inputs),
                    capture.inputs.retained_bytes(),
                );
            }
            let weight = capture.retained_bytes();
            self.compute_sources
                .insert(key, Arc::clone(&capture), weight);
            capture
        };
        self.channel.compute = Some(Arc::clone(&capture));
        Ok(capture)
    }
}
