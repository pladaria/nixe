//! Reflected set-zero ABI. Shared layouts are created on changed-state/miss paths.
use super::*;
use crate::{VulkanDevice, VulkanError, device::DeviceOwner};
use ash::vk;
use rustc_hash::FxHashMap as HashMap;
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct LayoutBinding {
    pub binding: u32,
    pub kind: vk::DescriptorType,
    pub stages: vk::ShaderStageFlags,
}
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct LayoutKey {
    pub bindings: Box<[LayoutBinding]>,
    pub push_bytes: u32,
    pub push_stages: vk::ShaderStageFlags,
}
impl LayoutKey {
    fn select_dynamic_buffers(
        &mut self,
        maximum: u32,
        uniform_maximum: u32,
    ) -> Result<(), VulkanError> {
        let explicit = self
            .bindings
            .iter()
            .filter(|b| b.kind == vk::DescriptorType::STORAGE_BUFFER_DYNAMIC)
            .count();
        let mut available =
            maximum
                .checked_sub(explicit as u32)
                .ok_or(VulkanError::InvalidResource(
                    "native dynamic storage descriptors exceed device limit",
                ))?;
        for binding in &mut self.bindings {
            if binding.kind == vk::DescriptorType::STORAGE_BUFFER && available != 0 {
                binding.kind = vk::DescriptorType::STORAGE_BUFFER_DYNAMIC;
                available -= 1;
            }
        }
        let explicit = self
            .bindings
            .iter()
            .filter(|b| b.kind == vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
            .count();
        let mut available =
            uniform_maximum
                .checked_sub(explicit as u32)
                .ok_or(VulkanError::InvalidResource(
                    "native dynamic uniform descriptors exceed device limit",
                ))?;
        for binding in &mut self.bindings {
            if binding.kind == vk::DescriptorType::UNIFORM_BUFFER && available != 0 {
                binding.kind = vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC;
                available -= 1;
            }
        }
        Ok(())
    }
    fn dynamic_bindings(&self) -> Box<[usize]> {
        let mut indices: Vec<_> = self
            .bindings
            .iter()
            .enumerate()
            .filter_map(|(i, b)| {
                matches!(
                    b.kind,
                    vk::DescriptorType::STORAGE_BUFFER_DYNAMIC
                        | vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
                )
                .then_some(i)
            })
            .collect();
        // Vulkan consumes offsets in ascending binding order, regardless of
        // the order in which a caller described the layout.
        indices.sort_unstable_by_key(|&i| self.bindings[i].binding);
        indices.into_boxed_slice()
    }

    pub fn reflect(shader: &ShaderArtifact) -> Result<Self, VulkanError> {
        let mut bindings = BTreeMap::<u32, LayoutBinding>::new();
        let mut push_bytes = 0;
        let mut push_stages = vk::ShaderStageFlags::empty();
        for stage in &shader.stages {
            let flags = stage_flags(stage.stage);
            if stage.module.push_constant_bytes != 0 {
                push_bytes = push_bytes.max(stage.module.push_constant_bytes);
                push_stages |= flags;
            }
            for binding in &stage.module.bindings {
                let kind = descriptor_type(binding.kind);
                if let Some(old) = bindings.get_mut(&binding.binding) {
                    if old.kind != kind {
                        return Err(VulkanError::InvalidResource(
                            "linked descriptor class mismatch",
                        ));
                    }
                    old.stages |= flags;
                } else {
                    bindings.insert(
                        binding.binding,
                        LayoutBinding {
                            binding: binding.binding,
                            kind,
                            stages: flags,
                        },
                    );
                }
            }
        }
        Ok(Self {
            bindings: bindings.into_values().collect(),
            push_bytes,
            push_stages,
        })
    }
}
pub struct PipelineLayout {
    pub(crate) owner: Arc<DeviceOwner>,
    pub(crate) handle: vk::PipelineLayout,
    pub(crate) set: vk::DescriptorSetLayout,
    pub key: LayoutKey,
    pub(crate) dynamic_bindings: Box<[usize]>,
}
impl PipelineLayout {
    /// Separate the stable descriptor base from the command's uint32 offset.
    /// Explicit ranges preserve shader bounds, including buffers above 4 GiB.
    // https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdBindDescriptorSets.html
    pub(crate) fn split_dynamic_offsets(
        &self,
        resources: &mut [DescriptorResource],
        limits: &vk::PhysicalDeviceLimits,
        mut offsets: Vec<u32>,
    ) -> Result<Vec<u32>, VulkanError> {
        offsets.clear();
        offsets.reserve(self.dynamic_bindings.len());
        for &index in &self.dynamic_bindings {
            let Some(DescriptorResource::Buffer {
                buffer,
                offset,
                range,
            }) = resources.get_mut(index)
            else {
                return Err(VulkanError::InvalidResource(
                    "dynamic storage binding requires a buffer",
                ));
            };
            let alignment = buffer_alignment(self.key.bindings[index].kind, limits);
            if !offset.is_multiple_of(alignment)
                || offset
                    .checked_add(*range)
                    .is_none_or(|end| end > buffer.size())
            {
                return Err(VulkanError::InvalidResource(
                    "dynamic buffer range/alignment exceeds host contract",
                ));
            }
            offsets.push(*offset as u32);
            *offset &= !u64::from(u32::MAX);
        }
        Ok(offsets)
    }
}
impl Drop for PipelineLayout {
    fn drop(&mut self) {
        unsafe {
            self.owner.device.destroy_pipeline_layout(self.handle, None);
            self.owner
                .device
                .destroy_descriptor_set_layout(self.set, None);
        }
    }
}
struct LayoutState {
    entries: HashMap<LayoutKey, (Arc<PipelineLayout>, u64)>,
    clock: u64,
}
pub(super) struct LayoutCache {
    owner: Arc<DeviceOwner>,
    limits: vk::PhysicalDeviceLimits,
    entries: usize,
    state: Mutex<LayoutState>,
}
impl LayoutCache {
    pub fn new(device: &VulkanDevice, entries: usize) -> Self {
        Self {
            owner: Arc::clone(&device.owner),
            limits: device.profile().limits,
            entries,
            state: Mutex::new(LayoutState {
                entries: HashMap::default(),
                clock: 0,
            }),
        }
    }
    pub fn get(&self, mut key: LayoutKey) -> Result<Arc<PipelineLayout>, VulkanError> {
        // Dynamic offsets keep arena positions out of descriptor identities.
        // Choose the ABI once per layout, within the host's dynamic-binding limit.
        // https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceLimits.html
        key.select_dynamic_buffers(
            self.limits.max_descriptor_set_storage_buffers_dynamic,
            self.limits.max_descriptor_set_uniform_buffers_dynamic,
        )?;
        let mut state = self.state.lock().unwrap();
        state.clock = state.clock.wrapping_add(1);
        let clock = state.clock;
        if let Some((layout, touched)) = state.entries.get_mut(&key) {
            *touched = clock;
            return Ok(Arc::clone(layout));
        }
        if key.push_bytes > self.limits.max_push_constants_size || !key.push_bytes.is_multiple_of(4)
        {
            return Err(VulkanError::InvalidResource(
                "native push constants exceed host limit or alignment",
            ));
        }
        for flag in [
            vk::ShaderStageFlags::VERTEX,
            vk::ShaderStageFlags::TESSELLATION_CONTROL,
            vk::ShaderStageFlags::TESSELLATION_EVALUATION,
            vk::ShaderStageFlags::FRAGMENT,
            vk::ShaderStageFlags::COMPUTE,
        ] {
            let counts = key
                .bindings
                .iter()
                .filter(|b| b.stages.contains(flag))
                .fold([0u32; 5], |mut c, b| {
                    c[match b.kind {
                        vk::DescriptorType::UNIFORM_BUFFER
                        | vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC => 0,
                        vk::DescriptorType::STORAGE_BUFFER
                        | vk::DescriptorType::STORAGE_BUFFER_DYNAMIC => 1,
                        vk::DescriptorType::SAMPLED_IMAGE => 2,
                        vk::DescriptorType::STORAGE_IMAGE => 3,
                        vk::DescriptorType::SAMPLER => 4,
                        _ => unreachable!(),
                    }] += 1;
                    c
                });
            let limits = [
                self.limits.max_per_stage_descriptor_uniform_buffers,
                self.limits.max_per_stage_descriptor_storage_buffers,
                self.limits.max_per_stage_descriptor_sampled_images,
                self.limits.max_per_stage_descriptor_storage_images,
                self.limits.max_per_stage_descriptor_samplers,
            ];
            if counts.iter().zip(limits).any(|(&n, max)| n > max)
                || counts.iter().sum::<u32>() > self.limits.max_per_stage_resources
            {
                return Err(VulkanError::InvalidResource(
                    "native descriptors exceed per-stage device limits",
                ));
            }
        }
        let counts =
            key.bindings.iter().fold([0u32; 5], |mut c, b| {
                c[match b.kind {
                    vk::DescriptorType::UNIFORM_BUFFER
                    | vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC => 0,
                    vk::DescriptorType::STORAGE_BUFFER
                    | vk::DescriptorType::STORAGE_BUFFER_DYNAMIC => 1,
                    vk::DescriptorType::SAMPLED_IMAGE => 2,
                    vk::DescriptorType::STORAGE_IMAGE => 3,
                    vk::DescriptorType::SAMPLER => 4,
                    _ => unreachable!(),
                }] += 1;
                c
            });
        let limits = [
            self.limits.max_descriptor_set_uniform_buffers,
            self.limits.max_descriptor_set_storage_buffers,
            self.limits.max_descriptor_set_sampled_images,
            self.limits.max_descriptor_set_storage_images,
            self.limits.max_descriptor_set_samplers,
        ];
        if counts.iter().zip(limits).any(|(&n, max)| n > max) {
            return Err(VulkanError::InvalidResource(
                "native descriptors exceed set device limits",
            ));
        }
        let bindings: Vec<_> = key
            .bindings
            .iter()
            .map(|b| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(b.binding)
                    .descriptor_type(b.kind)
                    .descriptor_count(1)
                    .stage_flags(b.stages)
            })
            .collect();
        let set = unsafe {
            self.owner.device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
        }
        .map_err(|e| VulkanError::vk("native descriptor layout", e))?;
        let mut layout = PipelineLayout {
            owner: Arc::clone(&self.owner),
            handle: vk::PipelineLayout::null(),
            set,
            dynamic_bindings: key.dynamic_bindings(),
            key,
        };
        let sets = [set];
        let push = [vk::PushConstantRange::default()
            .offset(0)
            .size(layout.key.push_bytes)
            .stage_flags(layout.key.push_stages)];
        layout.handle = unsafe {
            self.owner.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&sets)
                    .push_constant_ranges(if layout.key.push_bytes == 0 {
                        &[]
                    } else {
                        &push
                    }),
                None,
            )
        }
        .map_err(|e| VulkanError::vk("native pipeline layout", e))?;
        // Eviction drops only cache ownership; actual pipelines/batches retain
        // their layouts. One LRU victim avoids whole-cache churn under pressure.
        let layout = Arc::new(layout);
        if self.entries != 0 {
            if state.entries.len() >= self.entries {
                let key = state
                    .entries
                    .iter()
                    .min_by_key(|(_, (_, tick))| *tick)
                    .map(|(key, _)| key.clone())
                    .unwrap();
                state.entries.remove(&key);
            }
            state
                .entries
                .insert(layout.key.clone(), (Arc::clone(&layout), clock));
        }
        Ok(layout)
    }
}
pub(super) fn stage_flags(stage: nixe_gpu::ShaderStage) -> vk::ShaderStageFlags {
    use nixe_gpu::ShaderStage::*;
    match stage {
        Vertex => vk::ShaderStageFlags::VERTEX,
        TessellationControl => vk::ShaderStageFlags::TESSELLATION_CONTROL,
        TessellationEvaluation => vk::ShaderStageFlags::TESSELLATION_EVALUATION,
        Geometry => vk::ShaderStageFlags::GEOMETRY,
        Fragment => vk::ShaderStageFlags::FRAGMENT,
        Compute => vk::ShaderStageFlags::COMPUTE,
    }
}
pub(super) fn buffer_alignment(kind: vk::DescriptorType, limits: &vk::PhysicalDeviceLimits) -> u64 {
    if matches!(
        kind,
        vk::DescriptorType::UNIFORM_BUFFER | vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
    ) {
        limits.min_uniform_buffer_offset_alignment
    } else {
        limits.min_storage_buffer_offset_alignment
    }
}
pub(super) fn descriptor_type(kind: nixe_gpu::ShaderResourceKind) -> vk::DescriptorType {
    use nixe_gpu::ShaderResourceKind::*;
    match kind {
        ConstantBuffer => vk::DescriptorType::UNIFORM_BUFFER,
        StorageBuffer => vk::DescriptorType::STORAGE_BUFFER,
        Sampler | ComparisonSampler => vk::DescriptorType::SAMPLER,
        StorageImage => vk::DescriptorType::STORAGE_IMAGE,
        SampledImage
        | SampledImage2DArray
        | SampledDepthImage
        | SampledDepthImage2DArray
        | SampledImageCube => vk::DescriptorType::SAMPLED_IMAGE,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn uniform_and_storage_dynamic_budgets_are_independent_and_offsets_follow_binding_order() {
        let mut key = LayoutKey {
            bindings: vec![
                LayoutBinding {
                    binding: 4,
                    kind: vk::DescriptorType::UNIFORM_BUFFER,
                    stages: vk::ShaderStageFlags::VERTEX,
                },
                LayoutBinding {
                    binding: 1,
                    kind: vk::DescriptorType::STORAGE_BUFFER,
                    stages: vk::ShaderStageFlags::VERTEX,
                },
                LayoutBinding {
                    binding: 2,
                    kind: vk::DescriptorType::UNIFORM_BUFFER,
                    stages: vk::ShaderStageFlags::FRAGMENT,
                },
            ]
            .into_boxed_slice(),
            push_bytes: 0,
            push_stages: vk::ShaderStageFlags::empty(),
        };
        key.select_dynamic_buffers(1, 1).unwrap();
        assert_eq!(&*key.dynamic_bindings(), &[1, 0]);
        assert_eq!(key.bindings[2].kind, vk::DescriptorType::UNIFORM_BUFFER);
        assert!(key.select_dynamic_buffers(1, 0).is_err());
        let limits = vk::PhysicalDeviceLimits {
            min_uniform_buffer_offset_alignment: 256,
            min_storage_buffer_offset_alignment: 16,
            ..Default::default()
        };
        assert_eq!(buffer_alignment(key.bindings[0].kind, &limits), 256);
        assert_eq!(buffer_alignment(key.bindings[1].kind, &limits), 16);
    }
    #[test]
    fn dynamic_storage_selection_obeys_host_limits_and_binding_order() {
        let make = || LayoutKey {
            bindings: [3, 1, 2]
                .map(|binding| LayoutBinding {
                    binding,
                    kind: vk::DescriptorType::STORAGE_BUFFER,
                    stages: vk::ShaderStageFlags::VERTEX,
                })
                .into(),
            push_bytes: 0,
            push_stages: vk::ShaderStageFlags::empty(),
        };
        let mut no_dynamic = make();
        no_dynamic.select_dynamic_buffers(0, 0).unwrap();
        assert!(no_dynamic.dynamic_bindings().is_empty());
        let mut mixed = make();
        mixed.select_dynamic_buffers(2, 0).unwrap();
        assert_eq!(&*mixed.dynamic_bindings(), &[1, 0]);
        assert_eq!(mixed.bindings[2].kind, vk::DescriptorType::STORAGE_BUFFER);
        mixed.select_dynamic_buffers(2, 0).unwrap();
        assert_eq!(&*mixed.dynamic_bindings(), &[1, 0]);
        assert!(mixed.select_dynamic_buffers(1, 0).is_err());
    }
}
