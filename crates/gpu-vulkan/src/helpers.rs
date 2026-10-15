//! Native helper programs; generated SPIR-V is checked in, compiled offline.
use crate::{VulkanError, device::DeviceOwner};
use ash::vk;
use std::sync::Arc;
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum BufferBinding {
    DescriptorSet,
    DeviceAddress,
    DescriptorBuffer,
}
pub(crate) struct Pipeline {
    pub owner: Arc<DeviceOwner>,
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub set_layout: vk::DescriptorSetLayout,
}
impl Drop for Pipeline {
    fn drop(&mut self) {
        unsafe {
            self.owner.device.destroy_pipeline(self.pipeline, None);
            self.owner.device.destroy_pipeline_layout(self.layout, None);
            self.owner
                .device
                .destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}
struct Module {
    owner: Arc<DeviceOwner>,
    handle: vk::ShaderModule,
}
impl Module {
    fn new(owner: &Arc<DeviceOwner>, bytes: &[u8]) -> Result<Self, VulkanError> {
        let words: Vec<_> = bytes
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes(word.try_into().expect("SPIR-V word")))
            .collect();
        let handle = unsafe {
            owner
                .device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
        }
        .map_err(|e| VulkanError::vk("vkCreateShaderModule(helper)", e))?;
        Ok(Self {
            owner: Arc::clone(owner),
            handle,
        })
    }
}
impl Drop for Module {
    fn drop(&mut self) {
        unsafe {
            self.owner.device.destroy_shader_module(self.handle, None);
        }
    }
}
#[derive(Default)]
pub(crate) struct Helpers {
    compute: [Option<Pipeline>; 3],
}
impl Helpers {
    pub fn compute(
        &mut self,
        owner: &Arc<DeviceOwner>,
        binding: BufferBinding,
    ) -> Result<&Pipeline, VulkanError> {
        let index = binding as usize;
        if self.compute[index].is_none() {
            let mut result = Pipeline {
                owner: Arc::clone(owner),
                pipeline: vk::Pipeline::null(),
                layout: vk::PipelineLayout::null(),
                set_layout: vk::DescriptorSetLayout::null(),
            };
            if binding != BufferBinding::DeviceAddress {
                let bindings = [vk::DescriptorSetLayoutBinding::default()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)];
                let flags = if binding == BufferBinding::DescriptorBuffer {
                    vk::DescriptorSetLayoutCreateFlags::DESCRIPTOR_BUFFER_EXT
                } else {
                    vk::DescriptorSetLayoutCreateFlags::empty()
                };
                result.set_layout = unsafe {
                    owner.device.create_descriptor_set_layout(
                        &vk::DescriptorSetLayoutCreateInfo::default()
                            .flags(flags)
                            .bindings(&bindings),
                        None,
                    )
                }
                .map_err(|e| VulkanError::vk("helper descriptor layout", e))?;
            }
            let sets = [result.set_layout];
            let ranges = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .size(if binding == BufferBinding::DeviceAddress {
                    16
                } else {
                    8
                })];
            let info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(if binding == BufferBinding::DeviceAddress {
                    &[]
                } else {
                    &sets
                })
                .push_constant_ranges(&ranges);
            result.layout = unsafe { owner.device.create_pipeline_layout(&info, None) }
                .map_err(|e| VulkanError::vk("helper pipeline layout", e))?;
            let module = Module::new(
                owner,
                if binding == BufferBinding::DeviceAddress {
                    include_bytes!("../shaders/spv/swap_rb_bda.comp.spv")
                } else {
                    include_bytes!("../shaders/spv/swap_rb.comp.spv")
                },
            )?;
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module.handle)
                .name(c"main");
            let flags = if binding == BufferBinding::DescriptorBuffer {
                vk::PipelineCreateFlags::DESCRIPTOR_BUFFER_EXT
            } else {
                vk::PipelineCreateFlags::empty()
            };
            let infos = [vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(result.layout)
                .flags(flags)];
            result.pipeline = unsafe {
                owner
                    .device
                    .create_compute_pipelines(vk::PipelineCache::null(), &infos, None)
            }
            .map_err(|(pipelines, e)| {
                for pipeline in pipelines {
                    unsafe {
                        owner.device.destroy_pipeline(pipeline, None);
                    }
                }
                VulkanError::vk("helper compute pipeline", e)
            })?[0];
            self.compute[index] = Some(result);
        }
        Ok(self.compute[index].as_ref().expect("compute helper"))
    }
}
