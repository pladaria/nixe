//! Vulkan descriptor updates consume current Maxwell stage slots directly.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/vk_graphics_pipeline.cpp
use super::*;
use crate::native::state;
use crate::{Draw, PreparedTargets, VulkanError};
use ash::vk;
impl crate::native::NativeExecution<'_> {
    fn graphics_constant_address(
        state: &MaxwellThreeDState,
        space: &MaxwellGpuMemory,
        group: u8,
        slot: u32,
        offset: u32,
        size: u64,
    ) -> Result<nixe_gpu::GpuVirtualAddress, crate::VulkanError> {
        use crate::VulkanError as V;
        let binding = state
            .shader_bindings()
            .groups()
            .get(group as usize)
            .and_then(|g| g.constant_buffers().get(slot as usize))
            .copied()
            .flatten()
            .ok_or(V::InvalidResource(
                "shader reads a disabled constant buffer",
            ))?;
        if u64::from(offset) + size > u64::from(binding.size()) {
            return Err(V::InvalidResource(
                "shader descriptor exceeds its constant buffer",
            ));
        }
        space
            .address(binding.address().get() + u64::from(offset))
            .map_err(|error| V::Compilation(error.to_string().into()))
    }
    /// StorageBufferBinding reads its actual NVN pointer descriptor. Bounds
    /// come from MemoryManager's same-kind layout, exactly as in Eden.
    /// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/buffer_cache/buffer_cache.h#L2006-L2065
    pub(in crate::native) fn storage_binding(
        &mut self,
        space: &MaxwellGpuMemory,
        descriptor: nixe_gpu::GpuVirtualAddress,
        written: bool,
    ) -> Result<Option<crate::residency::StorageBuffer>, crate::VulkanError> {
        use crate::VulkanError as V;
        let fail = |error: model::MaxwellGpuAccessError| V::Compilation(error.to_string().into());
        let target = space.read_u64(descriptor).map_err(fail)?;
        if target == 0 {
            return Ok(None);
        }
        let address = space
            .address(target)
            .map_err(|e| V::Compilation(e.to_string().into()))?;
        let layout = u64::from(space.memory_layout_size(address).map_err(fail)? as u32);
        let size_address = space
            .checked_add(descriptor, 8)
            .map_err(|error| V::Compilation(error.to_string().into()))?;
        let size_word = space.read_u64(size_address).map_err(fail)?;
        let packed = size_word as u32;
        let size = if size_word >> 32 == 0 && packed != 0 && u64::from(packed) <= layout {
            u64::from(packed)
        } else {
            layout.min(8 * 1024 * 1024)
        };
        if size == 0 {
            return Ok(None);
        }
        let alignment = self
            .residency
            .device()
            .profile()
            .limits
            .min_storage_buffer_offset_alignment;
        let aligned = target / alignment * alignment;
        let aligned_gpu = space
            .address(aligned)
            .map_err(|e| V::Compilation(e.to_string().into()))?;
        let device = match space.device_address(aligned_gpu) {
            Ok(address) => address,
            Err(model::MaxwellGpuAccessError::UnmappedAddress { .. }) => return Ok(None),
            Err(error) => return Err(fail(error)),
        };
        let binding_size = if written {
            target - aligned + size
        } else {
            let begin = space.device_address(address).map_err(fail)?;
            (begin + size).div_ceil(nixe_memory::PAGE_SIZE) * nixe_memory::PAGE_SIZE - device
        };
        Ok(Some(crate::residency::StorageBuffer {
            source: GuestBufferRange {
                address: device,
                size: binding_size,
            },
            gpu_address: aligned_gpu,
            native: None,
            written,
        }))
    }
}

// ConfigureImpl's enabled stages are compile-time choices, selected once when
// the graphics pipeline is created, as in Eden's ConfigureFunc/FindSpec.
macro_rules! graphics_stages {
    ($mask:ident, |$stage:ident| $body:block) => {
        if $mask & 1 != 0 {
            const $stage: usize = 0;
            $body
        }
        if $mask & 2 != 0 {
            const $stage: usize = 1;
            $body
        }
        if $mask & 4 != 0 {
            const $stage: usize = 2;
            $body
        }
        if $mask & 8 != 0 {
            const $stage: usize = 3;
            $body
        }
        if $mask & 16 != 0 {
            const $stage: usize = 4;
            $body
        }
    };
}

impl crate::native::pipeline_cache::GraphicsPipeline {
    pub(in crate::native) fn configure_func(
        shader: &crate::cache::ShaderArtifact,
        resources: &crate::native::pipeline_resources::PipelineResources,
    ) -> crate::native::pipeline_cache::ConfigureGraphics {
        // The image/texel-buffer resource classes are rejected when building
        // PipelineResources. The supported specs are Eden's SimpleVertex,
        // SimpleVertexFragment, SimpleStorage and Default, in that order.
        // https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/renderer_vulkan/vk_graphics_pipeline.cpp#L155-L246
        let stages = shader
            .stages
            .iter()
            .fold(0u32, |mask, s| mask | (1 << s.stage as u32));
        let storage = resources.stages[..5].iter().any(|s| !s.storage.is_empty());
        if !storage && stages & !1 == 0 {
            Self::configure_impl::<1, false>
        } else if !storage && stages & !17 == 0 {
            Self::configure_impl::<17, false>
        } else if stages & !17 == 0 {
            Self::configure_impl::<17, true>
        } else {
            Self::configure_impl::<31, true>
        }
    }
    #[allow(clippy::too_many_arguments)]
    fn configure_stage<const STAGE: usize, const STORAGE: bool>(
        &self,
        execution: &mut crate::native::NativeExecution<'_>,
        state: &MaxwellThreeDState,
        space: &MaxwellGpuMemory,
        group: Option<u8>,
        via_header: bool,
        textures: &mut smallvec::SmallVec<[super::textures::ShaderTexture; 64]>,
    ) -> Result<(), VulkanError> {
        let resources = &self.resources.stages[STAGE];
        if resources.storage.is_empty() && resources.images.is_empty() {
            return Ok(());
        }
        let group = group.ok_or(VulkanError::InvalidResource(
            "graphics resource has no hardware cbuf group",
        ))?;
        if STORAGE {
            for slot in &resources.storage {
                let descriptor = crate::native::NativeExecution::graphics_constant_address(
                    state,
                    space,
                    group,
                    slot.cbuf,
                    slot.offset,
                    16,
                )?;
                let source = execution.storage_binding(space, descriptor, slot.written)?;
                execution
                    .residency
                    .runtime
                    .buffer_bindings
                    .bind_storage_buffer(STAGE, slot.slot, source);
            }
        }
        for slot in &resources.images {
            let read_handle = |source: &crate::native::pipeline_resources::ImageHandle| {
                let address = crate::native::NativeExecution::graphics_constant_address(
                    state,
                    space,
                    group,
                    source.cbuf,
                    source.offset,
                    4,
                )?;
                space
                    .read_u32(address)
                    .map(|value| value << source.shift)
                    .map_err(|error| VulkanError::Compilation(error.to_string().into()))
            };
            let mut handle = read_handle(&slot.primary)?;
            if let Some(secondary) = &slot.secondary {
                handle |= read_handle(secondary)?;
            }
            let reference = model::engines::threed::MaxwellThreeDTextureReference::new(
                group,
                slot.primary.cbuf as u8,
                slot.primary.offset,
            );
            let sampler = if slot.sampler {
                Some(execution.shader_sampler(
                    false,
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
                storage_format: slot.storage_format,
            });
        }
        Ok(())
    }
    fn update_stage_buffers<const STAGE: usize, const STORAGE: bool>(
        &self,
        execution: &mut crate::native::NativeExecution<'_>,
        space: &MaxwellGpuMemory,
        group: Option<u8>,
    ) -> Result<(), VulkanError> {
        let resources = &self.resources.stages[STAGE];
        for uniform in &resources.uniforms {
            execution.residency.graphics_uniform(
                group.ok_or(VulkanError::InvalidResource(
                    "active uniform stage has no constant-buffer group",
                ))?,
                uniform.slot as u8,
            )?;
        }
        if STORAGE {
            for storage in &resources.storage {
                execution
                    .residency
                    .update_storage_buffer(STAGE, storage.slot, space)?;
            }
        }
        Ok(())
    }
    fn configure_resources<const STAGES: u8, const STORAGE: bool>(
        &self,
        execution: &mut crate::native::NativeExecution<'_>,
        state: &MaxwellThreeDState,
        space: &MaxwellGpuMemory,
        groups: [Option<u8>; 5],
        images: &mut smallvec::SmallVec<[crate::residency::execution::ShaderImage; 64]>,
    ) -> Result<(), crate::VulkanError> {
        let bindings = state.shader_bindings();
        let via_header = bindings.sampler_binding().value()
            == Some(&model::engines::threed::MaxwellThreeDSamplerBindingMode::ViaTextureHeader);
        let pool = |address: Option<u64>, maximum: Option<u32>| {
            address
                .zip(maximum)
                .map(|(a, maximum_index)| model::recompiler::TexturePool {
                    address: a,
                    maximum_index,
                    via_header_index: via_header,
                })
        };
        let image_pool = pool(
            bindings.texture_headers().address().map(|a| a.get()),
            bindings.texture_headers().maximum_index().value().copied(),
        );
        let sampler_pool = pool(
            bindings.samplers().address().map(|a| a.get()),
            if via_header {
                bindings.texture_headers().maximum_index().value().copied()
            } else {
                bindings.samplers().maximum_index().value().copied()
            },
        );
        execution.synchronize_shader_descriptors(false, space, image_pool, sampler_pool);
        let mut textures = smallvec::SmallVec::with_capacity(self.resources.texture_count);
        graphics_stages!(STAGES, |STAGE| {
            self.configure_stage::<STAGE, STORAGE>(
                execution,
                state,
                space,
                groups[STAGE],
                via_header,
                &mut textures,
            )?;
        });
        execution.fill_shader_images(false, space, &textures, images)
    }
    #[allow(clippy::too_many_arguments)]
    fn configure_impl<const STAGES: u8, const STORAGE: bool>(
        &self,
        execution: &mut crate::native::NativeExecution<'_>,
        registers: &nixe_gpu_maxwell::engines::threed::MaxwellThreeDState,
        draw: &nixe_gpu_maxwell::MaxwellThreeDDrawState,
        space: &nixe_gpu_maxwell::MaxwellGpuMemory,
        groups: [Option<u8>; 5],
        arguments: Draw,
        dirty: u128,
    ) -> Result<(), VulkanError> {
        loop {
            let pipeline = &self.pipeline;
            let vertex_attributes = self.vertex_attributes;
            // Keep draw-local views and their inline payload in this frame
            // instead of moving them through large Result values between
            // preparation and recording.
            let mut images = smallvec::SmallVec::new();
            loop {
                execution.resolve_render_targets(registers, space)?;
                self.configure_resources::<STAGES, STORAGE>(
                    execution,
                    registers,
                    space,
                    groups,
                    &mut images,
                )?;
                execution.resolve_render_targets(registers, space)?;
                if !execution
                    .residency
                    .texture_cache
                    .bindings
                    .has_deleted_images
                {
                    break;
                }
            }
            let mut geometry_changed = false;
            loop {
                execution
                    .residency
                    .runtime
                    .buffer_bindings
                    .channel
                    .has_deleted_buffers = false;
                geometry_changed |= execution
                    .residency
                    .update_geometry(registers, draw, space)?;
                graphics_stages!(STAGES, |STAGE| {
                    self.update_stage_buffers::<STAGE, STORAGE>(execution, space, groups[STAGE])?;
                });
                if !execution
                    .residency
                    .runtime
                    .buffer_bindings
                    .channel
                    .has_deleted_buffers
                {
                    break;
                }
            }
            let (arguments, native_index) = execution.residency.assemble_quads(
                draw.topology,
                arguments,
                draw.inline_indices(),
            )?;
            let channel = execution
                .state
                .current_context
                .expect("selected guest channel");
            let enabled = execution
                .residency
                .runtime
                .buffer_bindings
                .channel
                .enabled_vertex_buffers;
            execution.residency.bind_host_geometry_buffers(
                Some(channel),
                enabled,
                arguments,
                native_index.as_ref(),
                |binding| registers.register_words()[0x1c00 / 4 + binding as usize * 4] & 0xfff,
            )?;
            let device = execution.residency.device();
            let slot = device.recording_slot()?;
            let pipeline_changed =
                device.submission.slots[slot].bound_graphics != pipeline.identity();
            if geometry_changed || pipeline_changed || dirty & GraphicsDirty::VERTEX_INPUT != 0 {
                let words = registers.register_words();
                let mut attributes = vertex_attributes;
                while attributes != 0 {
                    let location = attributes.trailing_zeros() as usize;
                    attributes &= attributes - 1;
                    let attribute = words[0x1160 / 4 + location];
                    if attribute & 64 == 0
                        && execution
                            .residency
                            .runtime
                            .buffer_bindings
                            .channel
                            .vertex_buffers
                            .get((attribute & 31) as usize)
                            .copied()
                            .flatten()
                            .is_none()
                    {
                        return Err(VulkanError::Compilation(
                        format!(
                            "vertex attribute {location}: shader consumes null vertex binding {}",
                            attribute & 31
                        )
                        .into(),
                    ));
                    }
                }
            }
            if execution
                .residency
                .texture_cache
                .bindings
                .has_deleted_images
            {
                continue;
            }
            let descriptors = if arguments.is_empty() {
                crate::residency::execution::DescriptorQueue {
                    record: None,
                    dynamic_offsets: None,
                    update: None,
                }
            } else {
                execution.residency.begin_guest_descriptors(pipeline)?;
                if STORAGE {
                    graphics_stages!(STAGES, |STAGE| {
                        execution
                            .residency
                            .import_guest_stage_storage::<STAGE>(&self.resources.stages[STAGE])?;
                    });
                }
                let mut image_index = 0;
                graphics_stages!(STAGES, |STAGE| {
                    let resources = &self.resources.stages[STAGE];
                    let end = image_index + resources.images.len();
                    execution
                        .residency
                        .configure_guest_stage::<STAGE, STORAGE>(
                            pipeline,
                            resources,
                            groups[STAGE],
                            &images[image_index..end],
                        )?;
                    image_index = end;
                });
                debug_assert_eq!(image_index, images.len());
                execution.residency.guest_descriptor_update()
            };
            execution.update_render_targets(registers, space, false)?;
            if execution
                .residency
                .texture_cache
                .bindings
                .has_deleted_images
            {
                continue;
            }
            let targets = execution.residency.current_framebuffer()?;
            return self.record(
                execution,
                registers,
                draw,
                &targets,
                arguments,
                &images,
                descriptors,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::native) fn draw(
        &self,
        execution: &mut crate::native::NativeExecution<'_>,
        registers: &nixe_gpu_maxwell::engines::threed::MaxwellThreeDState,
        draw: &nixe_gpu_maxwell::MaxwellThreeDDrawState,
        space: &nixe_gpu_maxwell::MaxwellGpuMemory,
        groups: [Option<u8>; 5],
    ) -> Result<(), VulkanError> {
        execution.flush_work()?;
        let arguments = registers::draw_arguments(registers, draw)?;
        (self.configure)(
            self,
            execution,
            registers,
            draw,
            space,
            groups,
            arguments,
            registers.graphics_dirty().bits(GraphicsDirty::PARAMETERS),
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn record(
        &self,
        execution: &mut crate::native::NativeExecution<'_>,
        registers: &nixe_gpu_maxwell::engines::threed::MaxwellThreeDState,
        guest_draw: &nixe_gpu_maxwell::MaxwellThreeDDrawState,
        targets: &Arc<PreparedTargets>,
        draw: Draw,
        images: &[crate::residency::execution::ShaderImage],
        descriptors: crate::residency::execution::DescriptorQueue,
    ) -> Result<(), VulkanError> {
        let pipeline = &self.pipeline;
        let push = state::ShaderConstants::graphics(
            &self.resources,
            registers,
            images,
            &execution.residency,
        )?;
        execution.residency.record_draw_inner(
            pipeline,
            targets,
            images,
            descriptors,
            draw,
            push.as_bytes(),
            registers.register_words()[0x1514 / 4] != 0,
            |aspect| {
                let r = registers.register_words();
                Ok(if aspect == vk::ImageAspectFlags::DEPTH {
                    r[0x12e8 / 4] == 0
                } else if aspect == vk::ImageAspectFlags::STENCIL {
                    !registers::stencil_enabled(r)
                        || registers::stencil_faces(r)?
                            .iter()
                            .all(|face| face & 0x1ff == 0)
                } else {
                    false
                })
            },
            |residency| {
                super::dynamic_state::record(
                    residency,
                    &mut execution.state.dynamic,
                    registers,
                    guest_draw,
                    pipeline,
                    &mut execution.state.statistics,
                )
            },
        )?;
        if !draw.is_empty() {
            execution
                .residency
                .finish_guest_writes(pipeline, &self.resources)?;
        }
        execution.state.statistics.draws += 1;
        Ok(())
    }
}
