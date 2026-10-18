use super::*;

impl MaxwellNativeFrontend {
    pub(super) fn capture_compute(
        &mut self,
        launch: &super::super::super::compute::MaxwellResolvedComputeLaunch,
        space: &MaxwellGpuAddressSpace,
    ) -> Result<Arc<crate::recompiler::CapturedProgram>, Error> {
        let request = launch.capture_request();
        let current = |capture: &crate::recompiler::CapturedProgram| {
            capture.stages[0].request_is_current(&request, space)
        };
        if let Some(capture) = &self.compute
            && current(capture)
        {
            return Ok(Arc::clone(capture));
        }
        let m = &request.metadata;
        let key = (
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
            if self.capture_compiler.is_none() {
                self.capture_compiler =
                    Some(nixe_shader_recompiler::Compiler::new().map_err(compilation)?);
            }
            let capture = Arc::new(
                crate::recompiler::capture_program(
                    self.capture_compiler.as_mut().unwrap(),
                    space,
                    [(None, request)],
                    &self.configuration,
                    &mut self.capture_memory,
                )
                .map_err(compilation)?,
            );
            self.statistics.shader_captures += 1;
            let weight = capture.retained_bytes();
            self.compute_sources
                .insert(key, Arc::clone(&capture), weight);
            capture
        };
        self.compute = Some(Arc::clone(&capture));
        Ok(capture)
    }
}
