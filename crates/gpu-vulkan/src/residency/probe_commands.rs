//! Binding ABI experiment, available only with the explicit binding-probe feature.
use super::*;
use crate::{BufferBinding, device::DeviceOwner};
use std::collections::HashMap;

const DESCRIPTOR_CAPACITY: usize = 256;
#[derive(Clone, Copy)]
enum BindingData {
    Set(vk::DescriptorSet),
    Buffer(vk::DeviceAddress, u64),
}
pub(crate) struct Descriptors {
    pool: vk::DescriptorPool,
    cache: HashMap<(vk::DescriptorSetLayout, vk::Buffer, u64), BindingData>,
    arena: Option<crate::queue::Arena>,
    address: vk::DeviceAddress,
    last: Option<((vk::DescriptorSetLayout, vk::Buffer, u64), BindingData)>,
}
impl Descriptors {
    pub fn new() -> Self {
        Self {
            pool: vk::DescriptorPool::null(),
            cache: HashMap::new(),
            arena: None,
            address: 0,
            last: None,
        }
    }
    pub fn reset(&mut self, owner: &DeviceOwner) -> Result<(), VulkanError> {
        if self.pool != vk::DescriptorPool::null() {
            unsafe {
                owner
                    .device
                    .reset_descriptor_pool(self.pool, vk::DescriptorPoolResetFlags::empty())
            }
            .map_err(|e| VulkanError::vk("vkResetDescriptorPool", e))?;
        }
        self.last = None;
        self.cache.clear();
        if let Some(arena) = &mut self.arena {
            arena.cursor = 0;
        }
        Ok(())
    }
    pub fn destroy(&mut self, owner: &DeviceOwner) {
        unsafe {
            owner.device.destroy_descriptor_pool(self.pool, None);
        }
        self.pool = vk::DescriptorPool::null();
    }
}
impl VulkanResidency {
    pub fn swap_rgba8_words(
        &mut self,
        buffer: BufferId,
        offset: u64,
        count: u32,
    ) -> Result<WorkTicket, VulkanError> {
        self.swap_rgba8_words_binding(buffer, offset, count, BufferBinding::DescriptorSet)
    }
    /// Binding candidates for the focused probe. Guest shader ABI selection is
    /// frozen before compilation; this helper does not select a guest backend.
    pub fn swap_rgba8_words_binding(
        &mut self,
        id: BufferId,
        offset: u64,
        count: u32,
        binding: BufferBinding,
    ) -> Result<WorkTicket, VulkanError> {
        self.ensure_buffer(id)?;
        let size = u64::from(count) * 4;
        self.check_buffer_range(id, offset, size)?;
        if !offset.is_multiple_of(4) {
            return Err(VulkanError::InvalidResource(
                "word conversion requires a word-aligned interval",
            ));
        }
        if binding == BufferBinding::DeviceAddress
            && (!self.device.profile().buffer_device_address
                || self.device.profile().core.shader_int64 == 0)
            || binding == BufferBinding::DescriptorBuffer
                && !self.device.profile().descriptor_buffer
        {
            return Err(VulkanError::InvalidResource(
                "binding probe candidate is unsupported by the selected device",
            ));
        }
        if binding != BufferBinding::DescriptorSet
            && !self
                .buffer(id)?
                .usage
                .contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
        {
            return Err(VulkanError::InvalidResource(
                "address binding requires a device-address buffer",
            ));
        }
        if !self
            .buffer(id)?
            .usage
            .contains(vk::BufferUsageFlags::STORAGE_BUFFER)
        {
            return Err(VulkanError::InvalidResource(
                "compute conversion requires storage-buffer usage",
            ));
        }
        if self.buffer(id)?.native.size()
            > u64::from(self.device.profile().limits.max_storage_buffer_range)
            && binding != BufferBinding::DeviceAddress
        {
            return Err(VulkanError::InvalidResource(
                "helper descriptor window exceeds maxStorageBufferRange; record an aligned bounded window",
            ));
        }
        let groups = count.div_ceil(64);
        if groups > self.device.profile().limits.max_compute_work_group_count[0] {
            return Err(VulkanError::InvalidResource(
                "conversion dispatch exceeds host workgroup count; split the word interval",
            ));
        }
        self.prepare_write(Resource::Buffer(id), offset, size)?;
        let owner = Arc::clone(&self.device.owner);
        let program = self.helpers.compute(&owner, binding)?;
        let (pipeline, layout, set_layout) = (program.pipeline, program.layout, program.set_layout);
        let buffer = self.buffer(id)?.native.handle;
        let index = self.device.begin_recording()?;
        let range = self.buffer(id)?.native.size();
        let key = (set_layout, buffer, range);
        let descriptors = &self.device.submission.slots[index].descriptors;
        let mut data = if binding == BufferBinding::DeviceAddress {
            None
        } else if let Some((previous, data)) = descriptors.last
            && previous == key
        {
            Some(data)
        } else {
            descriptors.cache.get(&key).copied()
        };
        if binding != BufferBinding::DeviceAddress && data.is_none() {
            if self.device.submission.slots[index].descriptors.cache.len() == DESCRIPTOR_CAPACITY {
                return Err(VulkanError::DescriptorPressure);
            }
            if binding == BufferBinding::DescriptorSet {
                let descriptors = &mut self.device.submission.slots[index].descriptors;
                if descriptors.pool == vk::DescriptorPool::null() {
                    let sizes = [vk::DescriptorPoolSize::default()
                        .ty(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(DESCRIPTOR_CAPACITY as u32)];
                    descriptors.pool = unsafe {
                        owner.device.create_descriptor_pool(
                            &vk::DescriptorPoolCreateInfo::default()
                                .max_sets(DESCRIPTOR_CAPACITY as u32)
                                .pool_sizes(&sizes),
                            None,
                        )
                    }
                    .map_err(|e| VulkanError::vk("helper descriptor pool", e))?;
                }
                let layouts = [set_layout];
                let set = unsafe {
                    owner.device.allocate_descriptor_sets(
                        &vk::DescriptorSetAllocateInfo::default()
                            .descriptor_pool(descriptors.pool)
                            .set_layouts(&layouts),
                    )
                }
                .map_err(|e| VulkanError::vk("helper descriptor allocation", e))?[0];
                let info = [vk::DescriptorBufferInfo::default()
                    .buffer(buffer)
                    .range(range)];
                unsafe {
                    owner.device.update_descriptor_sets(
                        &[vk::WriteDescriptorSet::default()
                            .dst_set(set)
                            .dst_binding(0)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(&info)],
                        &[],
                    );
                }
                data = Some(BindingData::Set(set));
            } else {
                let extension = self
                    .descriptor_dispatch
                    .as_ref()
                    .expect("enabled descriptor dispatch");
                let size = unsafe { extension.get_descriptor_set_layout_size(set_layout) };
                let binding_offset =
                    unsafe { extension.get_descriptor_set_layout_binding_offset(set_layout, 0) };
                let descriptor_size = self.device.profile().storage_buffer_descriptor_size;
                let mut descriptor = vec![0; descriptor_size];
                let address = self.buffer(id)?.address;
                let address_info = vk::DescriptorAddressInfoEXT::default()
                    .address(address)
                    .range(range);
                let info = vk::DescriptorGetInfoEXT::default()
                    .ty(vk::DescriptorType::STORAGE_BUFFER)
                    .data(vk::DescriptorDataEXT {
                        p_storage_buffer: &address_info,
                    });
                unsafe {
                    extension.get_descriptor(&info, &mut descriptor);
                }
                let alignment = self.device.profile().descriptor_buffer_offset_alignment;
                if self.device.submission.slots[index]
                    .descriptors
                    .arena
                    .is_none()
                {
                    let stride = size
                        .max(alignment)
                        .max(owner.non_coherent_atom_size)
                        .next_power_of_two();
                    self.constructing.push(Resource::Buffer(id));
                    let allocated = allocate_buffer_charged(
                        &owner,
                        stride * DESCRIPTOR_CAPACITY as u64,
                        vk::BufferUsageFlags::RESOURCE_DESCRIPTOR_BUFFER_EXT
                            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                        MemoryLocation::CpuToGpu,
                        AllocationMode::Dedicated,
                        "helper descriptor arena",
                        |bytes| self.reserve(bytes),
                    );
                    self.unpin(Resource::Buffer(id));
                    let allocated = allocated?;
                    let address = unsafe {
                        owner.device.get_buffer_device_address(
                            &vk::BufferDeviceAddressInfo::default().buffer(allocated.handle),
                        )
                    };
                    if !address.is_multiple_of(alignment) {
                        return Err(VulkanError::InvalidResource(
                            "descriptor arena address violates host alignment",
                        ));
                    }
                    let descriptors = &mut self.device.submission.slots[index].descriptors;
                    descriptors.address = address;
                    descriptors.arena = Some(crate::queue::Arena {
                        buffer: allocated,
                        cursor: 0,
                    });
                }
                let descriptors = &mut self.device.submission.slots[index].descriptors;
                let arena = descriptors.arena.as_mut().expect("descriptor arena");
                let offset = arena
                    .reserve_aligned(size, alignment)
                    .ok_or(VulkanError::DescriptorPressure)?;
                // Atom-isolated descriptor bytes are immutable until this slot completes.
                unsafe {
                    arena
                        .buffer
                        .write_range(offset + binding_offset, &descriptor)
                }?;
                data = Some(BindingData::Buffer(descriptors.address, offset));
            }
            self.device.submission.slots[index]
                .descriptors
                .cache
                .insert(key, data.expect("created binding"));
        }
        if let Some(data) = data {
            self.device.submission.slots[index].descriptors.last = Some((key, data));
        }
        self.buffer_access_range(
            id,
            offset,
            size,
            AccessScope::new(
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
            ),
        )?;
        let address = self.buffer(id)?.address;
        self.barriers.emit(&mut self.device);
        let slot = &mut self.device.submission.slots[index];
        let command = slot.command;
        unsafe {
            if slot.bound_compute != pipeline {
                owner
                    .device
                    .cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, pipeline);
                slot.bound_compute = pipeline;
                slot.bound_set = vk::DescriptorSet::null();
                slot.bound_descriptor_buffer = 0;
                slot.bound_descriptor_offset = u64::MAX;
            }
            match binding {
                BufferBinding::DeviceAddress => {
                    let address = address + offset;
                    let mut push = [0; 16];
                    push[..8].copy_from_slice(&address.to_ne_bytes());
                    push[12..].copy_from_slice(&count.to_ne_bytes());
                    owner.device.cmd_push_constants(
                        command,
                        layout,
                        vk::ShaderStageFlags::COMPUTE,
                        0,
                        &push,
                    );
                }
                _ => {
                    match data.expect("descriptor binding") {
                        BindingData::Set(set) => {
                            if slot.bound_set != set {
                                owner.device.cmd_bind_descriptor_sets(
                                    command,
                                    vk::PipelineBindPoint::COMPUTE,
                                    layout,
                                    0,
                                    &[set],
                                    &[],
                                );
                                slot.bound_set = set;
                            }
                        }
                        BindingData::Buffer(address, offset) => {
                            let extension = self
                                .descriptor_dispatch
                                .as_ref()
                                .expect("enabled descriptor dispatch");
                            if slot.bound_descriptor_buffer != address {
                                extension.cmd_bind_descriptor_buffers(
                                    command,
                                    &[vk::DescriptorBufferBindingInfoEXT::default()
                                        .address(address)
                                        .usage(
                                            vk::BufferUsageFlags::RESOURCE_DESCRIPTOR_BUFFER_EXT,
                                        )],
                                );
                            }
                            if slot.bound_descriptor_offset != offset
                                || slot.bound_descriptor_buffer != address
                            {
                                extension.cmd_set_descriptor_buffer_offsets(
                                    command,
                                    vk::PipelineBindPoint::COMPUTE,
                                    layout,
                                    0,
                                    &[0],
                                    &[offset],
                                );
                                slot.bound_descriptor_buffer = address;
                                slot.bound_descriptor_offset = offset;
                            }
                        }
                    }
                    let mut push = [0; 8];
                    push[..4].copy_from_slice(&(offset as u32 / 4).to_ne_bytes());
                    push[4..].copy_from_slice(&count.to_ne_bytes());
                    owner.device.cmd_push_constants(
                        command,
                        layout,
                        vk::ShaderStageFlags::COMPUTE,
                        0,
                        &push,
                    );
                }
            }
            owner.device.cmd_dispatch(command, groups, 1, 1);
        }
        self.record_authority(Resource::Buffer(id), offset, size)?;
        self.device.work_ticket()
    }
}
