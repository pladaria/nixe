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
        images: &mut smallvec::SmallVec<[crate::residency::execution::ShaderImage; 64]>,
    ) -> Result<(), crate::VulkanError> {
        use crate::VulkanError as V;
        let via_header = launch.texture_pool.is_some_and(|p| p.via_header_index);
        let sampler_pool = launch.sampler_pool.map(|mut sampler| {
            if via_header {
                sampler.maximum_index = launch.texture_pool.unwrap().maximum_index;
            }
            sampler
        });
        execution.synchronize_shader_descriptors(
            true,
            space.id().get(),
            launch.texture_pool.map(|p| (p.address, p.maximum_index)),
            sampler_pool.map(|p| (p.address, p.maximum_index)),
        );
        let mut computed = smallvec::SmallVec::<[u32; 64]>::new();
        computed.resize(self.resources.computed_handle_count, 0);
        if !computed.is_empty() {
            super::handles::HandleReader::new(
                execution,
                space,
                &self.resources.stages[5],
                super::handles::HandleBuffers::Compute(launch),
            )
            .evaluate(&mut computed)?;
        }
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
            execution.bind_guest_storage_buffer(space, descriptor, slot.written, 5, slot.slot)?;
        }
        let mut textures =
            smallvec::SmallVec::<[super::textures::ShaderTexture; 64]>::with_capacity(
                self.resources.texture_count,
            );
        let buffers = super::handles::HandleBuffers::Compute(launch);
        for slot in &self.resources.stages[5].images {
            let (handle, consumed) =
                super::handles::read_handle(space, buffers, &slot.handle, &computed)?;
            let sampler = if slot.sampler {
                Some(execution.shader_sampler(
                    true,
                    space,
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
                handle: consumed,
                index: handle & 0xfffff,
                dimension: slot.dimension,
                descriptor: slot.descriptor,
                sampler,
                storage_format: slot.storage_format,
            });
        }
        execution.fill_shader_images(true, space, &textures, images)
    }
    pub(in crate::native) fn dispatch(
        &self,
        execution: &mut crate::native::NativeExecution<'_>,
        launch: &model::MaxwellResolvedComputeLaunch,
        space: &MaxwellGpuMemory,
        groups: [u32; 3],
    ) -> Result<(), crate::VulkanError> {
        execution.flush_work()?;
        let pipeline = &self.pipeline;
        execution
            .residency
            .runtime
            .device
            .realize_deferred_clear()?;
        let mut images = smallvec::SmallVec::new();
        self.configure_resources(execution, launch, space, &mut images)?;
        execution
            .residency
            .update_compute_buffers(&self.resources.stages[5], space)?;
        let push = crate::native::state::ShaderConstants::resources(
            &self.resources,
            &images,
            &execution.residency,
        );
        execution.residency.dispatch_guest(
            pipeline,
            &self.resources,
            &images,
            groups,
            push.as_bytes(),
        )?;
        execution.state.statistics.dispatches += 1;
        Ok(())
    }
}
