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
    pub fn get(&self, key: LayoutKey) -> Result<Arc<PipelineLayout>, VulkanError> {
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
                        vk::DescriptorType::UNIFORM_BUFFER => 0,
                        vk::DescriptorType::STORAGE_BUFFER => 1,
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
        let counts = key.bindings.iter().fold([0u32; 5], |mut c, b| {
            c[match b.kind {
                vk::DescriptorType::UNIFORM_BUFFER => 0,
                vk::DescriptorType::STORAGE_BUFFER => 1,
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
pub(super) fn descriptor_type(kind: nixe_gpu::ShaderResourceKind) -> vk::DescriptorType {
    use nixe_gpu::ShaderResourceKind::*;
    match kind {
        // The native emitter declares both as std430 word-addressed SSBOs,
        // including dynamically indexed guest constant data. Reflect the actual
        // SPIR-V storage class; "constant" is a guest read-only semantic role.
        ConstantBuffer | StorageBuffer => vk::DescriptorType::STORAGE_BUFFER,
        Sampler | ComparisonSampler => vk::DescriptorType::SAMPLER,
        StorageImage => vk::DescriptorType::STORAGE_IMAGE,
        SampledImage
        | SampledImage2DArray
        | SampledDepthImage
        | SampledDepthImage2DArray
        | SampledImageCube => vk::DescriptorType::SAMPLED_IMAGE,
    }
}
