//! Native set-zero ABI over the driver's resident buffers. No upload/coherence
//! owner or resource table is duplicated here. Sets are immutable once recorded.
use super::*;
use ash::vk::Handle;
use nixe_gpu::{PipelineStages, ShaderResourceKind, SpirvPipelineBinding};

pub(super) fn layout(
    bindings: &[SpirvPipelineBinding],
    limits: crate::VulkanGraphicsLimits,
) -> Result<
    (
        Vec<vk::DescriptorSetLayoutBinding<'static>>,
        vk::PipelineStageFlags,
    ),
    BackendDriverError,
> {
    if bindings.len() > limits.storage_buffers_per_set as usize {
        return Err(unsupported(
            "native descriptor set exceeds physical storage-buffer limit",
        ));
    }
    let stages = [
        (
            PipelineStages::VERTEX_SHADER,
            vk::ShaderStageFlags::VERTEX,
            vk::PipelineStageFlags::VERTEX_SHADER,
        ),
        (
            PipelineStages::TESSELLATION_CONTROL_SHADER,
            vk::ShaderStageFlags::TESSELLATION_CONTROL,
            vk::PipelineStageFlags::TESSELLATION_CONTROL_SHADER,
        ),
        (
            PipelineStages::TESSELLATION_EVALUATION_SHADER,
            vk::ShaderStageFlags::TESSELLATION_EVALUATION,
            vk::PipelineStageFlags::TESSELLATION_EVALUATION_SHADER,
        ),
        (
            PipelineStages::FRAGMENT_SHADER,
            vk::ShaderStageFlags::FRAGMENT,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
        ),
    ];
    let mut counts = [0; 4];
    let mut accesses = vk::PipelineStageFlags::empty();
    let mut layouts = Vec::with_capacity(bindings.len());
    for binding in bindings {
        let resource = binding.resource;
        if resource.kind() != ShaderResourceKind::ConstantBuffer
            || !resource.readable()
            || resource.writable()
        {
            return Err(unsupported(
                "native descriptors currently require read-only constant buffers",
            ));
        }
        let mut flags = vk::ShaderStageFlags::empty();
        for (i, (neutral, shader, stage)) in stages.iter().enumerate() {
            if binding.stages.contains(*neutral) {
                counts[i] += 1;
                // This path has one color attachment, also counted among FS
                // resources. Other stages consume only the live storage blocks.
                // https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceLimits.html
                if counts[i] > limits.storage_buffers_per_stage
                    || counts[i] + u32::from(i == 3) > limits.resources_per_stage
                {
                    return Err(error(format!(
                        "native stage {shader:?} needs {} storage buffers; limits: buffers={}, resources={}",
                        counts[i], limits.storage_buffers_per_stage, limits.resources_per_stage
                    )));
                }
                flags |= *shader;
                accesses |= *stage;
            }
        }
        if flags.is_empty() {
            return Err(unsupported(
                "native descriptor has no graphics-stage visibility",
            ));
        }
        layouts.push(
            vk::DescriptorSetLayoutBinding::default()
                .binding(u32::from(resource.binding()))
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .stage_flags(flags),
        );
    }
    Ok((layouts, accesses))
}

// Pools are allocated in batches, without individual free/reset of live sets.
// A full page retires when its last cached/submitted user disappears. Allocation
// locks only on cache misses; no descriptor update/allocation/lock on warm hits.
// https://docs.vulkan.org/refpages/latest/refpages/source/VkDescriptorPoolCreateInfo.html
const SETS_PER_PAGE: usize = 32;
struct DescriptorPage {
    _device: Device,
    raw: ash::Device,
    pool: vk::DescriptorPool,
    sets: Vec<vk::DescriptorSet>,
}
impl Drop for DescriptorPage {
    fn drop(&mut self) {
        unsafe {
            self.raw.destroy_descriptor_pool(self.pool, None);
        }
    }
}
#[derive(Default)]
pub(super) struct DescriptorArena {
    page: Option<Arc<DescriptorPage>>,
    next: usize,
}
impl DescriptorArena {
    fn allocate(
        &mut self,
        pipeline: &NativePipeline,
        device: &Device,
    ) -> Result<(Arc<DescriptorPage>, vk::DescriptorSet), BackendDriverError> {
        if self.page.is_none() || self.next == SETS_PER_PAGE {
            let sizes = [vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: (pipeline.shaders.bindings().len() * SETS_PER_PAGE) as u32,
            }];
            let pool = unsafe {
                pipeline.raw.create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(SETS_PER_PAGE as u32)
                        .pool_sizes(&sizes),
                    None,
                )
            }
            .map_err(error)?;
            let mut page = DescriptorPage {
                _device: device.clone(),
                raw: pipeline.raw.clone(),
                pool,
                sets: Vec::new(),
            };
            let layouts = [pipeline.set_layout; SETS_PER_PAGE];
            page.sets = unsafe {
                pipeline.raw.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(pool)
                        .set_layouts(&layouts),
                )
            }
            .map_err(error)?;
            self.page = Some(Arc::new(page));
            self.next = 0;
        }
        let page = self.page.as_ref().unwrap();
        let set = page.sets[self.next];
        self.next += 1;
        Ok((Arc::clone(page), set))
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(super) struct BufferKey {
    pub(super) handle: BackendResourceHandle,
    // Host identity distinguishes residency recreation within one generation.
    pub(super) buffer: Buffer,
}
pub(super) struct NativeBindings {
    pub(super) pipeline: Arc<NativePipeline>,
    _page: Arc<DescriptorPage>,
    pub(super) set: vk::DescriptorSet,
    pub(super) buffers: Box<[BufferKey]>,
}

fn same_descriptor_identity<P, B: PartialEq>(
    cached_pipeline: &Arc<P>,
    pipeline: &Arc<P>,
    cached_buffers: &[B],
    buffers: &[B],
) -> bool {
    Arc::ptr_eq(cached_pipeline, pipeline) && cached_buffers == buffers
}

impl WgpuBackendDriver {
    pub(super) fn native_bindings(
        &mut self,
        dependencies: &ResolvedBackendResources,
        draw: &DrawOperation,
        pipeline: &Arc<NativePipeline>,
    ) -> Result<Option<Arc<NativeBindings>>, BackendDriverError> {
        let abi = pipeline.shaders.bindings();
        if abi.is_empty() {
            return Ok(None);
        }
        // The neutral native ABI occupies set zero only. Extra unused tables
        // are harmless, but must never be mistaken for another set's resources.
        let table = draw
            .prepared
            .descriptor_tables
            .first()
            .ok_or_else(|| unsupported("native shader needs descriptor set zero"))?;
        let table = dependency_handle(dependencies, ResourceDependency::DescriptorTable(*table))?;
        let Some(Resource::DescriptorTable {
            bindings,
            native_indices,
            ..
        }) = self.resource_record_mut(table)?.host.as_mut()
        else {
            return Err(kind_mismatch(table));
        };
        if native_indices.is_none() {
            let mut indices = Box::new([u16::MAX; 256]);
            for (i, binding) in bindings.iter().enumerate() {
                indices[usize::from(binding.binding)] = i as u16;
            }
            *native_indices = Some(indices);
        }
        let mut key = std::mem::take(&mut self.native.binding_key);
        key.clear();
        let Resource::DescriptorTable {
            bindings,
            native_indices,
            ..
        } = self.resource(table)?
        else {
            unreachable!()
        };
        let indices = native_indices.as_ref().unwrap();
        let limit = self
            .native
            .capabilities
            .unwrap()
            .graphics_limits
            .storage_buffer_range;
        for binding in abi {
            let number = binding.resource.binding();
            let resource = bindings
                .get(usize::from(indices[usize::from(number)]))
                .ok_or_else(|| error(format!("missing constant-buffer binding {number}")))?
                .resource;
            let handle = dependency_handle(dependencies, resource)?;
            let record = self.resource_record(handle)?;
            let Some(Resource::Buffer { buffer, view }) = &record.host else {
                return Err(kind_mismatch(handle));
            };
            let view = view.as_ref().ok_or_else(|| {
                error(format!(
                    "constant-buffer binding {number} has no canonical backing"
                ))
            })?;
            // Upload_inputs already applies canonical CPU/GPU coherence. Raw
            // reads may not bypass initialization or expose an unbacked suffix.
            if !record.content.as_ref().is_some_and(|c| c.initialized)
                || view.buffer_offset() != 0
                || view.size() != buffer.size()
                || !buffer.size().is_multiple_of(4)
                || buffer.size() > u64::from(limit)
            {
                return Err(error(format!(
                    "constant-buffer binding {number} needs fully initialized, word-sized backing within physical range {limit}"
                )));
            }
            key.push(BufferKey {
                handle,
                buffer: buffer.clone(),
            });
        }
        let fingerprint = nixe_gpu::cache_fingerprint(&(pipeline.pipeline.as_raw(), &key));
        let serial = self.take_cache_use()?;
        let cached = self
            .native
            .bindings
            .get_mut(&fingerprint)
            .and_then(|bucket| {
                bucket.iter_mut().find(|(cached, _)| {
                    same_descriptor_identity(&cached.pipeline, pipeline, &cached.buffers, &key)
                })
            });
        let bindings = if let Some((cached, used)) = cached {
            *used = serial;
            Arc::clone(cached)
        } else {
            let (page, set) = pipeline
                .descriptors
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .allocate(pipeline, &self.device)?;
            let infos: Vec<_> = key
                .iter()
                .map(|entry| {
                    let buffer = unsafe {
                        entry
                            .buffer
                            .as_hal::<wgpu::hal::api::Vulkan>()
                            .unwrap()
                            .raw_handle()
                    };
                    vk::DescriptorBufferInfo {
                        buffer,
                        offset: 0,
                        range: entry.buffer.size(),
                    }
                })
                .collect();
            let writes: Vec<_> = abi
                .iter()
                .zip(&infos)
                .map(|(binding, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(u32::from(binding.resource.binding()))
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(info))
                })
                .collect();
            // Never update a recorded/cached descriptor set in place.
            // https://docs.vulkan.org/refpages/latest/refpages/source/vkUpdateDescriptorSets.html
            unsafe {
                pipeline.raw.update_descriptor_sets(&writes, &[]);
            }
            let bindings = Arc::new(NativeBindings {
                pipeline: Arc::clone(pipeline),
                _page: page,
                set,
                buffers: key.clone().into_boxed_slice(),
            });
            self.native
                .evict_binding_if_full(self.cache_configuration.bind_groups_per_descriptor_table());
            self.native
                .bindings
                .entry(fingerprint)
                .or_default()
                .push((Arc::clone(&bindings), serial));
            self.native.binding_count += 1;
            bindings
        };
        key.clear(); // Scratch storage must not retain resources after eviction.
        self.native.binding_key = key;
        Ok(Some(bindings))
    }
}

#[cfg(test)]
#[path = "descriptor_tests.rs"]
mod tests;
