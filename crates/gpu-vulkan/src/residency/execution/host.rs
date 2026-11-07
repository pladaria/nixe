//! Owned inputs for native host utility shaders. Guest cache banks and descriptor
//! queues do not pass through this capture API.
use super::*;
impl Residency<'_> {
    fn new_host_program(&self, pipeline: Arc<NativePipeline>) -> Result<HostProgram, VulkanError> {
        if !Arc::ptr_eq(&pipeline.owner, &self.runtime.device.owner) {
            return Err(VulkanError::InvalidResource(
                "program pipeline belongs to another device",
            ));
        }
        Ok(HostProgram {
            binding_revision: self.texture_cache.binding_revision.into(),
            dynamic_offsets: vec![0; pipeline.layout.dynamic_bindings.len()],
            pipeline,
            record: Default::default(),
            bindings: Vec::new(),
            unaligned_uniforms: 0,
        })
    }
    pub fn prepare_host_program(
        &self,
        pipeline: Arc<NativePipeline>,
        bindings: &[HostBinding],
    ) -> Result<HostProgram, VulkanError> {
        let mut program = self.new_host_program(pipeline)?;
        // Explicit host inputs retain their native captures until GPU completion.
        for (index, binding) in bindings.iter().enumerate() {
            self.capture_host_binding(&mut program, index, binding)?;
        }
        self.finish_host_capture(&mut program)?;
        Ok(program)
    }
    /// Like Eden's binding updates, changing a range does not rebuild an access
    /// list. The recorder consumes the native slots and compiled shader roles.
    /// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/buffer_cache/buffer_cache.h
    fn capture_host_binding(
        &self,
        program: &mut HostProgram,
        index: usize,
        binding: &HostBinding,
    ) -> Result<(), VulkanError> {
        let pipeline = &program.pipeline;
        let reflected =
            pipeline
                .layout
                .key
                .bindings
                .get(index)
                .ok_or(VulkanError::InvalidResource(
                    "program binding slot exceeds its layout",
                ))?;
        let uniform = matches!(
            reflected.kind,
            vk::DescriptorType::UNIFORM_BUFFER | vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
        );
        let limits = &self.runtime.device.profile().limits;
        let alignment = limits.min_uniform_buffer_offset_alignment;
        let unaligned = |binding: &HostBinding| {
            uniform
                && matches!(binding, HostBinding::Buffer { offset, .. } if !offset.is_multiple_of(alignment))
        };
        let was_unaligned = program.bindings.get(index).is_some_and(unaligned);
        let now_unaligned = unaligned(binding);
        if !self.input_binding_current(binding) {
            return Err(VulkanError::InvalidResource(
                "program selects a retired native binding",
            ));
        }
        self.validate_binding(reflected.kind, binding)?;
        if !uniform && let Ok(slot) = pipeline.layout.dynamic_bindings.binary_search(&index) {
            program.dynamic_offsets[slot] = match binding {
                HostBinding::Buffer {
                    buffer,
                    offset,
                    size,
                } => {
                    pipeline
                        .layout
                        .split_buffer_offset(
                            index,
                            self.buffer_cache.get(buffer.id).native.size(),
                            *offset,
                            *size,
                            limits,
                        )?
                        .1
                }
                HostBinding::NullBuffer => 0,
                _ => {
                    return Err(VulkanError::InvalidResource(
                        "dynamic binding requires a buffer",
                    ));
                }
            };
        }
        if index == program.bindings.len() {
            program.bindings.push(binding.clone());
        } else {
            *program
                .bindings
                .get_mut(index)
                .ok_or(VulkanError::InvalidResource(
                    "initial program slots are not in order",
                ))? = binding.clone();
        }
        program.unaligned_uniforms =
            program.unaligned_uniforms + usize::from(now_unaligned) - usize::from(was_unaligned);
        Ok(())
    }
    fn finish_host_capture(&self, program: &mut HostProgram) -> Result<(), VulkanError> {
        if program.bindings.len() != program.pipeline.layout.key.bindings.len()
            || !self.host_program_current(program)
        {
            return Err(VulkanError::InvalidResource(
                "program has incomplete or retired binding slots",
            ));
        }
        program
            .binding_revision
            .store(self.texture_cache.binding_revision, Ordering::Relaxed);
        Ok(())
    }
    fn validate_binding(
        &self,
        kind: vk::DescriptorType,
        binding: &HostBinding,
    ) -> Result<(), VulkanError> {
        match binding {
            HostBinding::NullBuffer => {
                if !self.runtime.device.profile().null_descriptor
                    || !matches!(
                        kind,
                        vk::DescriptorType::STORAGE_BUFFER
                            | vk::DescriptorType::STORAGE_BUFFER_DYNAMIC
                            | vk::DescriptorType::UNIFORM_BUFFER
                            | vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
                    )
                {
                    return Err(VulkanError::InvalidResource(
                        "null buffer descriptor is not supported for this binding",
                    ));
                }
            }
            HostBinding::Buffer {
                buffer,
                offset,
                size,
            }
            | HostBinding::TexelBuffer {
                buffer,
                offset,
                size,
                ..
            } => {
                self.check_buffer_range(buffer.id, *offset, *size)?;
            }
            HostBinding::Image(view) | HostBinding::CombinedImageSampler { image: view, .. } => {
                let image = self.texture_cache.get(view.image);
                let description = &self.texture_cache.views[view.id].description;
                if kind == vk::DescriptorType::STORAGE_IMAGE
                    && description.swizzle != [vk::ComponentSwizzle::IDENTITY; 4]
                {
                    return Err(VulkanError::InvalidResource(
                        "storage image descriptors require identity component mapping",
                    ));
                }
                if description.aspects
                    == vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
                {
                    return Err(VulkanError::InvalidResource(
                        "shader descriptors require an individual depth or stencil view aspect",
                    ));
                }
                let usage = if kind == vk::DescriptorType::STORAGE_IMAGE {
                    vk::ImageUsageFlags::STORAGE
                } else {
                    vk::ImageUsageFlags::SAMPLED
                };
                if !image.description.usage.contains(usage) {
                    return Err(VulkanError::InvalidResource(
                        "shader image requires sampled/storage usage",
                    ));
                }
            }
            HostBinding::Sampler(_) => {}
        }
        Ok(())
    }
}
