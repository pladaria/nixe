//! Ordinary compute execution shares the device, queue and resource caches with graphics.
use super::*;
use nixe_gpu::DispatchOperation;

pub(super) struct CachedComputePipeline {
    pipeline: ComputePipeline,
    serial: u64,
    last_used: u64,
}

impl WgpuBackendDriver {
    pub(super) fn encode_dispatch(
        &mut self,
        encoder: &mut CommandEncoder,
        dependencies: &ResolvedBackendResources,
        dispatch: &DispatchOperation,
    ) -> Result<(), BackendDriverError> {
        let (pipeline, serial) = self.compute_pipeline(dependencies, dispatch)?;
        let mut groups = std::mem::take(&mut self.compute_bind_groups);
        groups.clear();
        let result = self.create_descriptor_bind_groups(
            dependencies,
            &dispatch.descriptor_tables,
            serial,
            |group| pipeline.get_bind_group_layout(group),
            &mut groups,
        );
        if result.is_ok() {
            // wgpu tracks storage hazards between passes in the existing encoder.
            // No submit, host wait or readback is required at a dispatch boundary.
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Nixe neutral compute dispatch"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            for (group, bindings) in groups.iter().enumerate() {
                pass.set_bind_group(group as u32, bindings, &[]);
            }
            pass.dispatch_workgroups(
                dispatch.workgroups[0],
                dispatch.workgroups[1],
                dispatch.workgroups[2],
            );
        }
        groups.clear();
        self.compute_bind_groups = groups;
        result
    }

    fn compute_pipeline(
        &mut self,
        dependencies: &ResolvedBackendResources,
        dispatch: &DispatchOperation,
    ) -> Result<(ComputePipeline, u64), BackendDriverError> {
        let pipeline_handle = dependency_handle(
            dependencies,
            ResourceDependency::Pipeline(dispatch.pipeline),
        )?;
        let shader_handle =
            dependency_handle(dependencies, ResourceDependency::Shader(dispatch.shader))?;
        let cache_use = self.take_cache_use()?;
        let capacity = self.cache_configuration.pipeline_variants_per_resource();
        let Some(Resource::Pipeline {
            description,
            compute,
            ..
        }) = self.resource_record_mut(pipeline_handle)?.host.as_mut()
        else {
            return Err(kind_mismatch(pipeline_handle));
        };
        if description.kind != PipelineKind::Compute {
            return Err(unsupported("dispatch requires a compute pipeline"));
        }
        if let Some(cached) = compute.get_mut(&shader_handle) {
            cached.last_used = cache_use;
            return Ok((cached.pipeline.clone(), cached.serial));
        }
        let Resource::Shader { neutral, .. } = self.resource(shader_handle)? else {
            return Err(kind_mismatch(shader_handle));
        };
        if neutral.stage() != ShaderStage::Compute {
            return Err(unsupported("dispatch requires a compute shader"));
        }
        let size = neutral
            .ir()
            .ir()
            .workgroup_size()
            .expect("verified compute metadata");
        let limits = self.device.limits();
        if size
            .iter()
            .zip([
                limits.max_compute_workgroup_size_x,
                limits.max_compute_workgroup_size_y,
                limits.max_compute_workgroup_size_z,
            ])
            .any(|(&value, max)| value > max)
            || size
                .into_iter()
                .try_fold(1u32, u32::checked_mul)
                .is_none_or(|total| total > limits.max_compute_invocations_per_workgroup)
        {
            return Err(BackendDriverError::failure(format!(
                "compute workgroup size exceeds host limits: {size:?}"
            )));
        }
        let (module, _) = self.compiled_shader(shader_handle)?;
        let scope = self.device.push_error_scope(ErrorFilter::Validation);
        let pipeline = self
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Nixe neutral compute pipeline"),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: PipelineCompilationOptions::default(),
                cache: self.pipeline_cache.as_ref(),
            });
        self.capture_error_scope(scope)?;
        let serial = self.next_pipeline_serial;
        self.next_pipeline_serial = serial
            .checked_add(1)
            .ok_or_else(|| unsupported("pipeline serial overflow"))?;
        let Some(Resource::Pipeline { compute, .. }) =
            self.resource_record_mut(pipeline_handle)?.host.as_mut()
        else {
            return Err(kind_mismatch(pipeline_handle));
        };
        if compute.len() == capacity {
            let key = least_recent_key(compute, |cached| cached.last_used)
                .expect("nonempty bounded pipeline cache");
            compute.remove(&key);
        }
        compute.insert(
            shader_handle,
            CachedComputePipeline {
                pipeline: pipeline.clone(),
                serial,
                last_used: cache_use,
            },
        );
        Ok((pipeline, serial))
    }
}
