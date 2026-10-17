//! Persistent device-local transfer allocations. CUDA writes visible NV12 bytes;
//! Vulkan waits on their semaphore and publishes ordinary canonical producers.
//! https://docs.vulkan.org/spec/latest/chapters/synchronization.html#synchronization-queue-transfers
use super::*;
use nixe_gpu::{DeviceImageFrame, DeviceImageSource, DeviceImageTransfer, ExternalImageBuffer};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};

pub(crate) fn device_uuid(device: &VulkanDevice) -> Option<[u8; 16]> {
    if !device.profile().external_memory_fd || !device.profile().external_semaphore_fd {
        return None;
    }
    let description = vk::PhysicalDeviceExternalBufferInfo::default()
        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
        .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
    let mut properties = vk::ExternalBufferProperties::default();
    let semaphore = vk::PhysicalDeviceExternalSemaphoreInfo::default()
        .handle_type(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD);
    let mut semaphore_properties = vk::ExternalSemaphoreProperties::default();
    unsafe {
        device
            .owner
            .instance
            .instance
            .get_physical_device_external_buffer_properties(
                device.owner.physical,
                &description,
                &mut properties,
            );
        device
            .owner
            .instance
            .instance
            .get_physical_device_external_semaphore_properties(
                device.owner.physical,
                &semaphore,
                &mut semaphore_properties,
            );
    }
    (properties
        .external_memory_properties
        .external_memory_features
        .contains(vk::ExternalMemoryFeatureFlags::EXPORTABLE)
        && semaphore_properties
            .external_semaphore_features
            .contains(vk::ExternalSemaphoreFeatureFlags::EXPORTABLE))
    .then_some(device.profile().device_uuid)
}

struct Allocation {
    owner: Arc<crate::device::DeviceOwner>,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    ready: vk::Semaphore,
    charge: Option<crate::memory::AllocationCharge>,
}
impl Drop for Allocation {
    fn drop(&mut self) {
        unsafe {
            self.owner.device.destroy_semaphore(self.ready, None);
            self.owner.device.destroy_buffer(self.buffer, None);
            self.owner.device.free_memory(self.memory, None);
        }
    }
}
struct Transfer {
    // Drop imports before their exported resources and their source frame.
    target: Box<dyn DeviceImageTransfer>,
    allocation: Allocation,
    source: Option<Arc<dyn DeviceImageSource>>,
    context: usize,
    dimensions: [u32; 2],
    pitch: u64,
    chroma: u64,
    pending: Option<WorkTicket>,
}
impl Transfer {
    fn new(cache: &mut VulkanResidency, frame: &DeviceImageFrame) -> Result<Self, VulkanError> {
        let device = &cache.device;
        let uuid = device_uuid(device).ok_or(VulkanError::InvalidResource(
            "device does not support external video memory/semaphore export",
        ))?;
        let pitch = u64::from(frame.width.div_ceil(2) * 2).next_multiple_of(4);
        let chroma = (pitch * u64::from(frame.height)).next_multiple_of(4);
        let size = chroma + pitch * u64::from(frame.height.div_ceil(2));
        let raw = &device.owner.device;
        let mut external = vk::ExternalMemoryBufferCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
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
        .map_err(|error| VulkanError::vk("create external video buffer", error))?;
        let mut allocation = Allocation {
            owner: device.owner.clone(),
            buffer,
            memory: vk::DeviceMemory::null(),
            ready: vk::Semaphore::null(),
            charge: None,
        };
        let requirements = unsafe { raw.get_buffer_memory_requirements(buffer) };
        allocation.charge = cache.budget.reserve(requirements.size);
        if allocation.charge.is_none() {
            return Err(VulkanError::ResidencyPressure {
                requested: requirements.size,
                allocated: cache.budget.used(),
                budget: cache.budget.limit,
            });
        }
        let types = unsafe {
            device
                .owner
                .instance
                .instance
                .get_physical_device_memory_properties(device.owner.physical)
        };
        let memory_type = (0..types.memory_type_count)
            .find(|&index| {
                requirements.memory_type_bits & (1 << index) != 0
                    && types.memory_types[index as usize]
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            })
            .ok_or(VulkanError::InvalidResource(
                "no compatible device-local external video memory type",
            ))?;
        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().buffer(buffer);
        let mut export = vk::ExportMemoryAllocateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
        allocation.memory = unsafe {
            raw.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(memory_type)
                    .push_next(&mut dedicated)
                    .push_next(&mut export),
                None,
            )
        }
        .map_err(|error| VulkanError::vk("allocate external video memory", error))?;
        unsafe { raw.bind_buffer_memory(buffer, allocation.memory, 0) }
            .map_err(|error| VulkanError::vk("bind external video memory", error))?;
        let memory_api =
            ash::khr::external_memory_fd::Device::new(&device.owner.instance.instance, raw);
        let fd = unsafe {
            memory_api.get_memory_fd(
                &vk::MemoryGetFdInfoKHR::default()
                    .memory(allocation.memory)
                    .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD),
            )
        }
        .map_err(|error| VulkanError::vk("export video memory FD", error))?;
        let memory_fd = unsafe { OwnedFd::from_raw_fd(fd) };
        let mut export = vk::ExportSemaphoreCreateInfo::default()
            .handle_types(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD);
        allocation.ready = unsafe {
            raw.create_semaphore(
                &vk::SemaphoreCreateInfo::default().push_next(&mut export),
                None,
            )
        }
        .map_err(|error| VulkanError::vk("create external video semaphore", error))?;
        let semaphore_api =
            ash::khr::external_semaphore_fd::Device::new(&device.owner.instance.instance, raw);
        let fd = unsafe {
            semaphore_api.get_semaphore_fd(
                &vk::SemaphoreGetFdInfoKHR::default()
                    .semaphore(allocation.ready)
                    .handle_type(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD),
            )
        }
        .map_err(|error| VulkanError::vk("export video semaphore FD", error))?;
        let semaphore_fd = unsafe { OwnedFd::from_raw_fd(fd) };
        let target = frame
            .source
            .import_nv12(ExternalImageBuffer {
                fd: memory_fd.as_raw_fd(),
                semaphore_fd: semaphore_fd.as_raw_fd(),
                allocation_size: requirements.size,
                device_uuid: uuid,
                row_pitch: pitch,
                chroma_offset: chroma,
                // Copies consume visible bytes only; allocation/pitch padding is
                // neither initialized speculatively nor read by Vulkan.
                initialize: false,
            })
            .map_err(VulkanError::ExternalVideo)?;
        Ok(Self {
            target,
            allocation,
            source: None,
            context: frame.source.transfer_context(),
            dimensions: [frame.width, frame.height],
            pitch,
            chroma,
            pending: None,
        })
    }
}
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct VideoStatistics {
    pub frames: u64,
    pub imports: u64,
    pub reused_imports: u64,
    pub pressure_waits: u64,
    pub transferred_bytes: u64,
}
#[derive(Default)]
pub(crate) struct VideoTransfers {
    slots: Vec<Transfer>,
    pub statistics: VideoStatistics,
}
impl VideoTransfers {
    pub(crate) fn import(
        &mut self,
        cache: &mut VulkanResidency,
        images: [ImageId; 2],
        frame: &DeviceImageFrame,
    ) -> Result<(), VulkanError> {
        if frame.width == 0
            || frame.height == 0
            || frame.width > cache.device.profile().limits.max_image_dimension2_d
            || frame.height > cache.device.profile().limits.max_image_dimension2_d
        {
            return Err(VulkanError::InvalidResource(
                "invalid decoded frame dimensions",
            ));
        }
        for (index, image) in images.iter().enumerate() {
            let description = cache.image(*image)?.description;
            let (width, height, format) = if index == 0 {
                (frame.width, frame.height, vk::Format::R8_UNORM)
            } else {
                (
                    frame.width.div_ceil(2),
                    frame.height.div_ceil(2),
                    vk::Format::R8G8_UNORM,
                )
            };
            if description.format != format
                || description.extent.width < width
                || description.extent.height != height
                || description.samples != vk::SampleCountFlags::TYPE_1
                || description.mip_levels != 1
                || description.array_layers != 1
            {
                return Err(VulkanError::InvalidResource(
                    "decoded video destinations differ from the NV12 dimensions/format",
                ));
            }
        }
        let completed = cache.device.poll_progress()?;
        let idle = |slot: &Transfer| slot.pending.is_none_or(|work| work.tick() <= completed);
        let reusable = self.slots.iter().position(|slot| {
            slot.context == frame.source.transfer_context()
                && slot.dimensions == [frame.width, frame.height]
                && idle(slot)
        });
        let index = if let Some(index) = reusable {
            self.statistics.reused_imports += 1;
            index
        } else {
            if self.slots.len() == cache.device.submission.configuration.command_slots {
                let index = self.slots.iter().position(idle).unwrap_or_else(|| {
                    self.slots
                        .iter()
                        .enumerate()
                        .min_by_key(|(_, slot)| slot.pending.unwrap().tick())
                        .unwrap()
                        .0
                });
                if !idle(&self.slots[index]) {
                    self.statistics.pressure_waits += 1;
                    cache.device.wait_for_work(
                        self.slots[index].pending.unwrap(),
                        std::time::Duration::MAX,
                    )?;
                }
                self.slots.swap_remove(index);
            }
            self.slots.push(Transfer::new(cache, frame)?);
            self.statistics.imports += 1;
            self.slots.len() - 1
        };
        // Reserve the consumer before starting asynchronous external writes.
        // An error while recording still submits the recorded prefix with its
        // wait, so ordinary shutdown cannot race the external producer.
        cache.device.begin_recording()?;
        let transfer = &mut self.slots[index];
        transfer.source = Some(Arc::clone(&frame.source));
        frame
            .source
            .enqueue_nv12(transfer.target.as_mut())
            .map_err(VulkanError::ExternalVideo)?;
        let family = cache.device.queues().graphics_compute_family;
        let acquire = vk::BufferMemoryBarrier2::default()
            .buffer(transfer.allocation.buffer)
            .offset(0)
            .size(vk::WHOLE_SIZE)
            .src_queue_family_index(vk::QUEUE_FAMILY_EXTERNAL)
            .dst_queue_family_index(family)
            .dst_stage_mask(vk::PipelineStageFlags2::COPY)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_READ);
        let recorded = (|| -> Result<(), VulkanError> {
            cache.barriers.queue_buffer(acquire);
            for (plane, image) in images.into_iter().enumerate() {
                let width = if plane == 0 {
                    frame.width
                } else {
                    frame.width.div_ceil(2)
                };
                let height = if plane == 0 {
                    frame.height
                } else {
                    frame.height.div_ceil(2)
                };
                unsafe {
                    cache.video_plane_from_buffer(
                        image,
                        transfer.allocation.buffer,
                        if plane == 0 { 0 } else { transfer.chroma },
                        transfer.pitch,
                        vk::Rect2D {
                            offset: vk::Offset2D::default(),
                            extent: vk::Extent2D { width, height },
                        },
                    )?;
                }
            }
            let release = vk::BufferMemoryBarrier2::default()
                .buffer(transfer.allocation.buffer)
                .offset(0)
                .size(vk::WHOLE_SIZE)
                .src_queue_family_index(family)
                .dst_queue_family_index(vk::QUEUE_FAMILY_EXTERNAL)
                .src_stage_mask(vk::PipelineStageFlags2::COPY)
                .src_access_mask(vk::AccessFlags2::TRANSFER_READ);
            cache.barriers.queue_buffer(release);
            Ok(())
        })();
        let wait = vk::SemaphoreSubmitInfo::default()
            .semaphore(transfer.allocation.ready)
            .stage_mask(vk::PipelineStageFlags2::COPY);
        let work = cache
            .flush_with_sync(&[wait], &[])?
            .ok_or(VulkanError::InvalidResource(
                "external video transfer has no consumer submission",
            ))?;
        transfer.pending = Some(work);
        recorded?;
        self.statistics.frames += 1;
        self.statistics.transferred_bytes += u64::from(frame.width) * u64::from(frame.height)
            + u64::from(frame.width.div_ceil(2) * 2) * u64::from(frame.height.div_ceil(2));
        Ok(())
    }
}
