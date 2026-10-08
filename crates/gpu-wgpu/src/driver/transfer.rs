use super::*;
use nixe_gpu::{BufferTransform, TransferComponent, TransferLayout};
use wgpu::util::DeviceExt;
fn layout_words(layout: TransferLayout) -> [u32; 6] {
    match layout {
        TransferLayout::Pitch { pitch } => [pitch, 0, 0, 0, 0, 0],
        TransferLayout::BlockLinear {
            row_bytes,
            origin_x_bytes,
            origin_y,
            block_height_log2,
        } => [
            0,
            row_bytes,
            origin_x_bytes,
            origin_y,
            u32::from(block_height_log2),
            1,
        ],
    }
}
impl WgpuBackendDriver {
    pub(super) fn encode_buffer_transform(
        &mut self,
        encoder: &mut CommandEncoder,
        dependencies: &ResolvedBackendResources<'_>,
        copy: &BufferTransform,
    ) -> Result<(), BackendDriverError> {
        copy.validate().map_err(unsupported)?;
        let source = self
            .buffer(dependency_handle(
                dependencies,
                ResourceDependency::Buffer(copy.source.buffer),
            )?)?
            .clone();
        let destination = self
            .buffer(dependency_handle(
                dependencies,
                ResourceDependency::Buffer(copy.destination.buffer),
            )?)?
            .clone();
        if copy.source.range.end() > source.size()
            || copy.destination.range.end() > destination.size()
        {
            return Err(unsupported("device transfer exceeds a resident buffer"));
        }
        let from = copy.source.range.offset() / 4 * 4;
        let size = align_u64(copy.source.range.end(), 4)? - from;
        let destination_from = copy.destination.range.offset()
            / u64::from(self.device.limits().min_storage_buffer_offset_alignment)
            * u64::from(self.device.limits().min_storage_buffer_offset_alignment);
        let destination_size = align_u64(copy.destination.range.end(), 4)? - destination_from;
        if size > self.device.limits().max_storage_buffer_binding_size
            || destination_size > self.device.limits().max_storage_buffer_binding_size
        {
            return Err(unsupported(
                "device transfer exceeds storage buffer binding limits",
            ));
        }
        let mut words = [0_u32; 32];
        words[..12].copy_from_slice(&[
            (copy.source.range.offset() - from) as u32,
            copy.source.range.size() as u32,
            u32::try_from(copy.destination.range.offset() - destination_from)
                .map_err(|_| unsupported("device copy destination offset"))?,
            u32::try_from(copy.destination.range.size())
                .map_err(|_| unsupported("device copy destination size"))?,
            copy.width,
            copy.height,
            u32::from(copy.component_bytes) * u32::from(copy.source_components),
            u32::from(copy.component_bytes) * u32::from(copy.destination_components),
            u32::from(copy.component_bytes),
            u32::from(copy.destination_components),
            copy.constant_a,
            copy.constant_b,
        ]);
        for (word, component) in words[12..16].iter_mut().zip(copy.components) {
            *word = match component {
                TransferComponent::Source(index) => u32::from(index),
                TransferComponent::ConstantA => 4,
                TransferComponent::ConstantB => 5,
                TransferComponent::Preserve => 6,
            };
        }
        words[16..22].copy_from_slice(&layout_words(copy.source_layout));
        words[22..28].copy_from_slice(&layout_words(copy.destination_layout));
        if self.transfer_pipeline.is_none() {
            let scope = self.device.push_error_scope(ErrorFilter::Validation);
            let module = self.device.create_shader_module(ShaderModuleDescriptor {
                label: Some("Nixe byte transfer shader"),
                source: ShaderSource::Wgsl(include_str!("transfer.wgsl").into()),
            });
            let pipeline = self
                .device
                .create_compute_pipeline(&ComputePipelineDescriptor {
                    label: Some("Nixe byte transfer pipeline"),
                    layout: None,
                    module: &module,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: self.pipeline_cache.as_ref(),
                });
            self.capture_error_scope(scope)?;
            self.transfer_pipeline = Some(pipeline);
        }
        if self
            .transfer_scratch
            .as_ref()
            .is_none_or(|buffer| buffer.size() < size)
        {
            let old = self.transfer_scratch.as_ref().map_or(0, Buffer::size);
            self.ensure_residency_budget(usize::from(old == 0), size - old, None)?;
            self.transfer_scratch = Some(self.device.create_buffer(&BufferDescriptor {
                label: Some("Nixe overlapping transfer snapshot"),
                size,
                usage: BufferUsages::COPY_DST | BufferUsages::STORAGE,
                mapped_at_creation: false,
            }));
            self.resident_resources += usize::from(old == 0);
            self.resident_resource_bytes += size - old;
        }
        let snapshot = self.transfer_scratch.as_ref().unwrap();
        // A source snapshot also provides memmove semantics for physical aliases.
        encoder.copy_buffer_to_buffer(&source, from, snapshot, 0, size);
        let mut bytes = [0_u8; 128];
        for (to, word) in bytes.chunks_exact_mut(4).zip(words) {
            to.copy_from_slice(&word.to_ne_bytes());
        }
        let parameters = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Nixe byte transfer parameters"),
                contents: &bytes,
                usage: BufferUsages::UNIFORM,
            });
        let pipeline = self.transfer_pipeline.as_ref().unwrap();
        let bindings = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("Nixe byte transfer bindings"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: snapshot.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &destination,
                        offset: destination_from,
                        size: wgpu::BufferSize::new(destination_size),
                    }),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: parameters.as_entire_binding(),
                },
            ],
        });
        let count =
            align_u64(copy.destination.range.end(), 4)? / 4 - copy.destination.range.offset() / 4;
        let groups = count.div_ceil(64);
        let width = groups.min(u64::from(
            self.device.limits().max_compute_workgroups_per_dimension,
        ));
        let height = groups.div_ceil(width);
        if height > u64::from(self.device.limits().max_compute_workgroups_per_dimension) {
            return Err(unsupported(
                "device transfer exceeds compute dispatch limits",
            ));
        }
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Nixe ordered byte transfer"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bindings, &[]);
        pass.dispatch_workgroups(width as u32, height as u32, 1);
        nixe_trace::event("gpu.copy_device_bytes", 0, copy.source.range.size());
        Ok(())
    }
}
