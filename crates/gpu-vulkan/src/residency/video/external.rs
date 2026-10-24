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
        let owner = cache.device.owner.clone();
        let uuid = device_uuid(&cache.device).ok_or(VulkanError::InvalidResource(
            "device does not support external video memory/semaphore export",
        ))?;
        let pitch = u64::from(frame.width.div_ceil(2) * 2).next_multiple_of(4);
        let chroma = (pitch * u64::from(frame.height)).next_multiple_of(4);
        let size = chroma + pitch * u64::from(frame.height.div_ceil(2));
        let raw = &owner.device;
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
            owner: owner.clone(),
            buffer,
            memory: vk::DeviceMemory::null(),
            ready: vk::Semaphore::null(),
            charge: None,
        };
        let requirements = unsafe { raw.get_buffer_memory_requirements(buffer) };
        allocation.charge = cache.reserve(requirements.size)?;
        let types = unsafe {
            owner
                .instance
                .instance
                .get_physical_device_memory_properties(owner.physical)
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
        let memory_api = ash::khr::external_memory_fd::Device::new(&owner.instance.instance, raw);
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
            ash::khr::external_semaphore_fd::Device::new(&owner.instance.instance, raw);
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
    pub(crate) fn retire_sources(&mut self, completed: u64) {
        for transfer in &mut self.slots {
            if transfer
                .pending
                .is_some_and(|work| work.tick() <= completed)
            {
                // The import retains the device context independently. Only
                // the pending copy needs the decoder's mapped output surface.
                // Eden releases its hardware frame after transferring pixels:
                // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/host1x/ffmpeg.cpp
                transfer.source = None;
                transfer.pending = None;
            }
        }
    }
    pub(crate) fn evict_idle(&mut self, completed: u64) -> bool {
        self.retire_sources(completed);
        let Some(index) = self.slots.iter().position(|slot| slot.pending.is_none()) else {
            return false;
        };
        self.slots.swap_remove(index);
        true
    }
}
impl VulkanResidency {
    pub(crate) fn import_decoded_planes(
        &mut self,
        images: [ImageId; 2],
        frame: &DeviceImageFrame,
    ) -> Result<(), VulkanError> {
        if frame.width == 0
            || frame.height == 0
            || frame.width > self.device.profile().limits.max_image_dimension2_d
            || frame.height > self.device.profile().limits.max_image_dimension2_d
        {
            return Err(VulkanError::InvalidResource(
                "invalid decoded frame dimensions",
            ));
        }
        for (index, image) in images.iter().enumerate() {
            let description = self.image(*image)?.description;
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
        let completed = self.device.poll_progress()?;
        let idle = |slot: &Transfer| slot.pending.is_none_or(|work| work.tick() <= completed);
        let reusable = self.video.slots.iter().position(|slot| {
            slot.context == frame.source.transfer_context()
                && slot.dimensions == [frame.width, frame.height]
                && idle(slot)
        });
        let index = if let Some(index) = reusable {
            self.video.statistics.reused_imports += 1;
            index
        } else {
            if self.video.slots.len() == self.device.submission.configuration.command_slots {
                let index = self.video.slots.iter().position(idle).unwrap_or_else(|| {
                    self.video
                        .slots
                        .iter()
                        .enumerate()
                        .min_by_key(|(_, slot)| slot.pending.unwrap().tick())
                        .unwrap()
                        .0
                });
                if !idle(&self.video.slots[index]) {
                    self.video.statistics.pressure_waits += 1;
                    self.device.wait_for_work(
                        self.video.slots[index].pending.unwrap(),
                        std::time::Duration::MAX,
                    )?;
                }
                self.video.slots.swap_remove(index);
            }
            let transfer = Transfer::new(self, frame)?;
            self.video.slots.push(transfer);
            self.video.statistics.imports += 1;
            self.video.slots.len() - 1
        };
        // Reserve the consumer before starting asynchronous external writes.
        // An error while recording still submits the recorded prefix with its
        // wait, so ordinary shutdown cannot race the external producer.
        self.device.begin_recording()?;
        let mut transfer = self.video.slots.swap_remove(index);
        let result = (|| {
            transfer.source = Some(Arc::clone(&frame.source));
            frame
                .source
                .enqueue_nv12(transfer.target.as_mut())
                .map_err(VulkanError::ExternalVideo)?;
            let (buffer, ready, pitch, chroma) = (
                transfer.allocation.buffer,
                transfer.allocation.ready,
                transfer.pitch,
                transfer.chroma,
            );
            let family = self.device.queues().graphics_compute_family;
            let acquire = vk::BufferMemoryBarrier2::default()
                .buffer(buffer)
                .offset(0)
                .size(vk::WHOLE_SIZE)
                .src_queue_family_index(vk::QUEUE_FAMILY_EXTERNAL)
                .dst_queue_family_index(family)
                .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_READ);
            let recorded = (|| -> Result<(), VulkanError> {
                self.barriers.queue_buffer(acquire);
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
                        self.video_plane_from_buffer(
                            image,
                            buffer,
                            if plane == 0 { 0 } else { chroma },
                            pitch,
                            vk::Rect2D {
                                offset: vk::Offset2D::default(),
                                extent: vk::Extent2D { width, height },
                            },
                        )?;
                    }
                }
                let release = vk::BufferMemoryBarrier2::default()
                    .buffer(buffer)
                    .offset(0)
                    .size(vk::WHOLE_SIZE)
                    .src_queue_family_index(family)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_EXTERNAL)
                    .src_stage_mask(vk::PipelineStageFlags2::COPY)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_READ);
                self.barriers.queue_buffer(release);
                Ok(())
            })();
            let wait = vk::SemaphoreSubmitInfo::default()
                .semaphore(ready)
                .stage_mask(vk::PipelineStageFlags2::COPY);
            let work = self
                .flush_with_sync(&[wait], &[])?
                .ok_or(VulkanError::InvalidResource(
                    "external video transfer has no consumer submission",
                ))?;
            transfer.pending = Some(work);
            recorded?;
            self.video.statistics.frames += 1;
            self.video.statistics.transferred_bytes += u64::from(frame.width)
                * u64::from(frame.height)
                + u64::from(frame.width.div_ceil(2) * 2) * u64::from(frame.height.div_ceil(2));
            Ok(())
        })();
        self.video.slots.push(transfer);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct Source;
    #[derive(Debug)]
    struct Import;
    impl DeviceImageTransfer for Import {
        fn wait(&mut self) -> Result<(), Box<str>> {
            Ok(())
        }
    }
    impl DeviceImageSource for Source {
        fn import_nv12(
            &self,
            _: ExternalImageBuffer,
        ) -> Result<Box<dyn DeviceImageTransfer>, Box<str>> {
            Ok(Box::new(Import))
        }
    }

    #[test]
    fn external_video_and_native_buffers_share_pressure_reclamation() {
        let Some(mut cache) =
            crate::residency::tests::cache(crate::ApiPreference::Auto, 8 * 1024 * 1024)
        else {
            return;
        };
        if device_uuid(&cache.device).is_none() {
            return;
        }
        let validation = cache.device.validation_monitor();
        let frame = DeviceImageFrame {
            width: 64,
            height: 64,
            source: Arc::new(Source),
        };
        let before = cache.allocated_bytes();
        let transfer = Transfer::new(&mut cache, &frame).unwrap();
        let transfer_bytes = cache.allocated_bytes() - before;
        drop(transfer);
        let (_allocation, backing) =
            crate::residency::tests::backing_allocation(&cache, 16384, 4096, 0);
        let buffer = cache
            .buffer_for_deferred_import(backing.clone(), vk::BufferUsageFlags::STORAGE_BUFFER)
            .unwrap()
            .buffer;
        let buffer_bytes = cache.allocated_bytes() - before;
        cache.budget.limit = before + transfer_bytes.max(buffer_bytes);
        let transfer = Transfer::new(&mut cache, &frame).unwrap();
        assert!(
            cache.buffer(buffer).is_err(),
            "external reservation evicts an idle native allocation"
        );
        cache.video.slots.push(transfer);
        cache
            .buffer_for_deferred_import(backing, vk::BufferUsageFlags::STORAGE_BUFFER)
            .unwrap();
        assert!(
            cache.video.slots.is_empty(),
            "ordinary allocations reclaim idle external transfers"
        );
        assert!(cache.allocated_bytes() <= cache.budget.limit);
        drop(cache);
        validation.check().unwrap();
    }

    #[test]
    fn completed_video_copies_release_decoder_surfaces_and_keep_reusable_imports() {
        let Some(mut cache) =
            crate::residency::tests::cache(crate::ApiPreference::Auto, 8 * 1024 * 1024)
        else {
            return;
        };
        if device_uuid(&cache.device).is_none() {
            return;
        }
        let validation = cache.device.validation_monitor();
        let mut sources = Vec::new();
        let mut tickets = Vec::new();
        for _ in 0..2 {
            let source: Arc<dyn DeviceImageSource> = Arc::new(Source);
            sources.push(Arc::downgrade(&source));
            let frame = DeviceImageFrame {
                width: 2,
                height: 2,
                source,
            };
            let mut transfer = Transfer::new(&mut cache, &frame).unwrap();
            let work = cache.device.work_ticket().unwrap();
            transfer.pending = Some(work);
            transfer.source = Some(Arc::clone(&frame.source));
            cache.video.slots.push(transfer);
            cache.flush().unwrap();
            tickets.push(work);
        }
        cache.video.retire_sources(tickets[0].tick() - 1);
        assert!(sources.iter().all(|source| source.upgrade().is_some()));
        for (index, work) in tickets.into_iter().enumerate() {
            assert!(
                cache
                    .device
                    .wait_for_work(work, std::time::Duration::from_secs(5))
                    .unwrap()
            );
            cache.video.retire_sources(work.tick());
            assert!(sources[index].upgrade().is_none());
            if index == 0 {
                assert!(sources[1].upgrade().is_some());
            }
            assert_eq!(
                cache.video.slots.len(),
                2,
                "external allocations stay reusable"
            );
        }
        drop(cache);
        validation.check().unwrap();
    }
}
