use super::*;

impl crate::native::NativeExecution<'_> {
    fn compute_constant_address(
        launch: &model::MaxwellResolvedComputeLaunch,
        space: &MaxwellGpuMemory,
        slot: u32,
        offset: u32,
        bytes: u64,
    ) -> Result<nixe_gpu::GpuVirtualAddress, crate::VulkanError> {
        use crate::VulkanError as V;
        let (address, size) = launch
            .constant_buffer(slot as u8)
            .map_err(|reason| V::Compilation(format!("compute cbuf {slot}: {reason}").into()))?;
        if u64::from(offset) + bytes > size {
            return Err(V::InvalidResource(
                "compute descriptor exceeds constant buffer",
            ));
        }
        space
            .address(address + u64::from(offset))
            .map_err(|error| V::Compilation(error.to_string().into()))
    }
}

impl crate::native::pipeline_cache::ComputePipeline {
    fn configure_resources(
        &self,
        execution: &mut crate::native::NativeExecution<'_>,
        launch: &model::MaxwellResolvedComputeLaunch,
        space: &MaxwellGpuMemory,
    ) -> Result<
        smallvec::SmallVec<[crate::residency::execution::ShaderImage; 64]>,
        crate::VulkanError,
    > {
        use crate::VulkanError as V;
        let via_header = launch.texture_pool.is_some_and(|p| p.via_header_index);
        let sampler_pool = launch.sampler_pool.map(|mut sampler| {
            if via_header {
                sampler.maximum_index = launch.texture_pool.unwrap().maximum_index;
            }
            sampler
        });
        execution.synchronize_shader_descriptors(true, space, launch.texture_pool, sampler_pool);
        for slot in &self.resources.stages[5].uniforms {
            let (address, size) = launch.constant_buffer(slot.slot as u8).map_err(|reason| {
                V::Compilation(format!("compute cbuf {}: {reason}", slot.slot).into())
            })?;
            let address = space
                .address(address)
                .map_err(|e| V::Compilation(e.to_string().into()))?;
            let address = space
                .device_address(address)
                .map_err(|e| V::Compilation(e.to_string().into()))?;
            execution
                .residency
                .runtime
                .buffer_bindings
                .bind_compute_uniform(slot.slot, GuestBufferRange { address, size });
        }
        for slot in &self.resources.stages[5].storage {
            let descriptor = crate::native::NativeExecution::compute_constant_address(
                launch,
                space,
                slot.cbuf,
                slot.offset,
                16,
            )?;
            let source = execution.storage_binding(space, descriptor, slot.written)?;
            execution
                .residency
                .runtime
                .buffer_bindings
                .bind_storage_buffer(5, slot.slot, source);
        }
        let mut textures =
            smallvec::SmallVec::<[super::textures::ShaderTexture; 64]>::with_capacity(
                self.resources.texture_count,
            );
        for slot in &self.resources.stages[5].images {
            let read_handle = |source: &crate::native::pipeline_resources::ImageHandle| {
                let address = crate::native::NativeExecution::compute_constant_address(
                    launch,
                    space,
                    source.cbuf,
                    source.offset,
                    4,
                )?;
                space
                    .read_u32(address)
                    .map(|value| value << source.shift)
                    .map_err(|error| V::Compilation(error.to_string().into()))
            };
            let mut handle = read_handle(&slot.primary)?;
            if let Some(secondary) = &slot.secondary {
                handle |= read_handle(secondary)?;
            }
            let reference = model::engines::threed::MaxwellThreeDTextureReference::new(
                0,
                slot.primary.cbuf as u8,
                slot.primary.offset,
            );
            let sampler = if slot.sampler {
                Some(execution.shader_sampler(
                    true,
                    space,
                    reference,
                    if via_header {
                        handle & 0xfffff
                    } else {
                        handle >> 20
                    },
                )?)
            } else {
                None
            };
            textures.push(super::textures::ShaderTexture {
                index: handle & 0xfffff,
                reference,
                dimension: slot.dimension,
                descriptor: slot.descriptor,
                sampler,
            });
        }
        execution.fill_shader_images(true, space, &textures)
    }
    pub(in crate::native) fn dispatch(
        &self,
        execution: &mut crate::native::NativeExecution<'_>,
        launch: &model::MaxwellResolvedComputeLaunch,
        space: &MaxwellGpuMemory,
        groups: [u32; 3],
    ) -> Result<bool, crate::VulkanError> {
        use crate::native::Computing;
        execution.flush_work()?;
        let pipeline = &self.pipeline;
        execution
            .residency
            .runtime
            .device
            .realize_deferred_clear()?;
        let previous = execution.state.context.computing.take();
        let changed = previous
            .as_ref()
            .is_none_or(|old| !Arc::ptr_eq(&old.pipeline, pipeline));
        let mut computing = previous.unwrap_or_else(|| Computing {
            pipeline: Arc::clone(pipeline),
            push: Vec::new(),
        });
        if changed {
            crate::native::state::update_guest_push(
                pipeline,
                &self.resources,
                None,
                &mut computing.push,
            )?;
            computing.pipeline = Arc::clone(pipeline);
        }
        let images = self.configure_resources(execution, launch, space)?;
        execution
            .residency
            .update_compute_buffers(&self.resources.stages[5], space)?;
        execution.state.context.computing = Some(computing);
        let computing = execution.state.context.computing.as_ref().unwrap();
        execution.residency.dispatch_guest(
            pipeline,
            &self.resources,
            &images,
            groups,
            &computing.push,
        )?;
        execution.state.statistics.dispatches += 1;
        Ok(true)
    }
}
