//! Exported device-local transfer storage for hardware-decoded NV12.
use ash::vk;
use nixe_gpu::{BackendDriverError, DeviceImageFrame, ExternalImageBuffer};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};

fn error(e: impl std::fmt::Display) -> BackendDriverError {
    BackendDriverError::failure(format!("device video transfer: {e}"))
}

pub(crate) fn device_uuid(device: &wgpu::Device) -> Option<[u8; 16]> {
    let hal = unsafe { device.as_hal::<wgpu::hal::api::Vulkan>() }?;
    if !hal
        .enabled_device_extensions()
        .contains(&ash::khr::external_memory_fd::NAME)
        || !hal
            .enabled_device_extensions()
            .contains(&ash::khr::external_semaphore_fd::NAME)
    {
        return None;
    }
    let mut id = vk::PhysicalDeviceIDProperties::default();
    let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut id);
    unsafe {
        hal.shared_instance()
            .raw_instance()
            .get_physical_device_properties2(hal.raw_physical_device(), &mut properties);
    }
    (properties.properties.vendor_id == 0x10de).then_some(id.device_uuid)
}

struct Allocation {
    _device: wgpu::Device,
    raw: ash::Device,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    semaphore: vk::Semaphore,
}
impl Drop for Allocation {
    fn drop(&mut self) {
        unsafe {
            self.raw.destroy_semaphore(self.semaphore, None);
            self.raw.destroy_buffer(self.buffer, None);
            if self.memory != vk::DeviceMemory::null() {
                self.raw.free_memory(self.memory, None);
            }
        }
    }
}

pub(crate) struct VideoBuffer {
    pub buffer: wgpu::Buffer,
    fd: OwnedFd,
    semaphore_fd: OwnedFd,
    semaphore: vk::Semaphore,
    pub allocation_size: u64,
    pub size: u64,
    pub pending: Option<nixe_gpu::BackendSubmissionToken>,
}
impl VideoBuffer {
    pub fn fd(&self) -> i32 {
        self.fd.as_raw_fd()
    }
    pub fn semaphore_fd(&self) -> i32 {
        self.semaphore_fd.as_raw_fd()
    }
    pub fn wait_cuda(
        &self,
        device: &wgpu::Device,
        access: &crate::WgpuQueueAccess,
    ) -> Result<(), BackendDriverError> {
        let hal = unsafe { device.as_hal::<wgpu::hal::api::Vulkan>() }
            .ok_or_else(|| error("Vulkan required"))?;
        let raw = hal.raw_device();
        let waits = [self.semaphore];
        let stages = [vk::PipelineStageFlags::TRANSFER];
        let submit = [vk::SubmitInfo::default()
            .wait_semaphores(&waits)
            .wait_dst_stage_mask(&stages)];
        let _guard = access.lock();
        unsafe { raw.queue_submit(raw.get_device_queue(0, 0), &submit, vk::Fence::null()) }
            .map_err(error)
    }
    pub fn new(
        device: &wgpu::Device,
        size: u64,
        frame: &DeviceImageFrame,
        pitch: u64,
        chroma_offset: u64,
    ) -> Result<Self, BackendDriverError> {
        let uuid = device_uuid(device).ok_or_else(|| error("external memory device UUID"))?;
        let hal = unsafe { device.as_hal::<wgpu::hal::api::Vulkan>() }
            .ok_or_else(|| error("Vulkan required"))?;
        let raw = hal.raw_device().clone();
        let instance = hal.shared_instance().raw_instance();
        let physical = hal.raw_physical_device();
        let memory_types = unsafe { instance.get_physical_device_memory_properties(physical) };
        let export = vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD;
        let mut external = vk::ExternalMemoryBufferCreateInfo::default().handle_types(export);
        let buffer = unsafe {
            raw.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .push_next(&mut external),
                None,
            )
        }
        .map_err(error)?;
        let mut owner = Allocation {
            _device: device.clone(),
            raw: raw.clone(),
            buffer,
            memory: vk::DeviceMemory::null(),
            semaphore: vk::Semaphore::null(),
        };
        let required = unsafe { raw.get_buffer_memory_requirements(buffer) };
        let memory_type_index = (0..memory_types.memory_type_count)
            .find(|&i| {
                required.memory_type_bits & (1 << i) != 0
                    && memory_types.memory_types[i as usize]
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            })
            .ok_or_else(|| error("no device-local export memory"))?;
        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().buffer(buffer);
        let mut exported = vk::ExportMemoryAllocateInfo::default().handle_types(export);
        owner.memory = unsafe {
            raw.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(required.size)
                    .memory_type_index(memory_type_index)
                    .push_next(&mut dedicated)
                    .push_next(&mut exported),
                None,
            )
        }
        .map_err(error)?;
        unsafe { raw.bind_buffer_memory(buffer, owner.memory, 0) }.map_err(error)?;
        let extension = ash::khr::external_memory_fd::Device::new(instance, &raw);
        let fd = unsafe {
            extension.get_memory_fd(
                &vk::MemoryGetFdInfoKHR::default()
                    .memory(owner.memory)
                    .handle_type(export),
            )
        }
        .map_err(error)?;
        let fd = unsafe { OwnedFd::from_raw_fd(fd) };
        let mut export_sem = vk::ExportSemaphoreCreateInfo::default()
            .handle_types(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD);
        owner.semaphore = unsafe {
            raw.create_semaphore(
                &vk::SemaphoreCreateInfo::default().push_next(&mut export_sem),
                None,
            )
        }
        .map_err(error)?;
        let sem_extension = ash::khr::external_semaphore_fd::Device::new(instance, &raw);
        let sem_fd = unsafe {
            sem_extension.get_semaphore_fd(
                &vk::SemaphoreGetFdInfoKHR::default()
                    .semaphore(owner.semaphore)
                    .handle_type(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD),
            )
        }
        .map_err(error)?;
        let semaphore_fd = unsafe { OwnedFd::from_raw_fd(sem_fd) };
        let semaphore = owner.semaphore;
        frame
            .source
            .copy_nv12(ExternalImageBuffer {
                fd: fd.as_raw_fd(),
                semaphore_fd: semaphore_fd.as_raw_fd(),
                allocation_size: required.size,
                device_uuid: uuid,
                row_pitch: pitch,
                chroma_offset,
                initialize: true,
            })
            .map_err(error)?;
        // HAL retires the callback after the last real use of the imported buffer.
        let imported = unsafe {
            wgpu::hal::vulkan::Buffer::from_raw_externally_owned(
                buffer,
                Box::new(move || drop(owner)),
            )
        };
        drop(hal);
        let buffer = unsafe {
            device.create_buffer_from_hal::<wgpu::hal::api::Vulkan>(
                imported,
                &wgpu::BufferDescriptor {
                    label: Some("Nixe hardware video transfer"),
                    size,
                    usage: wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                },
            )
        };
        Ok(Self {
            buffer,
            fd,
            semaphore_fd,
            semaphore,
            allocation_size: required.size,
            size,
            pending: None,
        })
    }
}
