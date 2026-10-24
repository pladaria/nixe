use super::*;
use ash::vk;
use nixe_gpu::{
    ImageDescription, ImageDimension, ImageExtent, ImageFormat, ImageKind,
    PresentationImageFormat as F, PresentationImageId, PresentationLease, PresentationRelease,
    ResidentImage, SampleCount, native::NativePresentationImage,
};
pub(crate) struct Snapshot {
    pub view: Arc<AllocatedImageView>,
    pub format: vk::Format,
    pub extent: [u32; 2],
    pub encoding: F,
    pub work: WorkTicket,
    last_use: WorkTicket,
    pub version: u64,
    generation: u32,
    leases: usize,
}
pub(crate) struct Frames {
    snapshots: Vec<Option<Snapshot>>,
    generation: u32,
    sender: mpsc::Sender<PresentationRelease>,
    releases: mpsc::Receiver<PresentationRelease>,
}
impl Frames {
    pub fn new() -> Self {
        let (sender, releases) = mpsc::channel();
        Self {
            snapshots: Vec::new(),
            generation: 0,
            sender,
            releases,
        }
    }
    fn drain(&mut self) {
        while let Ok(release) = self.releases.try_recv() {
            if let Some(Some(frame)) = self.snapshots.get_mut(release.image.slot as usize)
                && frame.generation == release.image.generation
                && frame.version == release.content_version
                && frame.work.submission() == release.producer
            {
                frame.leases = frame
                    .leases
                    .checked_sub(1)
                    .expect("one release per frame lease");
            }
        }
    }
    pub fn get(&self, image: &ResidentImage) -> Result<&Snapshot, VulkanError> {
        let lease = image.lease();
        let frame = self
            .snapshots
            .get(lease.image().slot as usize)
            .and_then(Option::as_ref)
            .ok_or(VulkanError::InvalidResource("foreign presentation slot"))?;
        if frame.generation != lease.image().generation
            || frame.version != lease.content_version()
            || frame.work.submission() != lease.producer()
            || frame.leases == 0
        {
            return Err(VulkanError::InvalidResource(
                "stale or foreign presentation version",
            ));
        }
        Ok(frame)
    }
    pub fn used(&mut self, image: &ResidentImage, work: WorkTicket) -> Result<(), VulkanError> {
        let slot = image.lease().image().slot as usize;
        self.get(image)?;
        // The lease's producer stays immutable; actual last use is tracked separately.
        self.snapshots[slot]
            .as_mut()
            .expect("validated frame")
            .last_use = work;
        Ok(())
    }
    pub fn capture(
        &mut self,
        cache: &mut VulkanResidency,
        source: NativePresentationImage,
        descriptors: &mut crate::cache::DescriptorCache,
    ) -> Result<ResidentImage, VulkanError> {
        self.drain();
        let format = match source.format {
            F::Rgba8 | F::Rgbx8 => vk::Format::R8G8B8A8_UNORM,
            F::Bgra8 => vk::Format::B8G8R8A8_UNORM,
            F::Rgb565 => vk::Format::R16_UINT,
            F::Rgba4444 => vk::Format::R16_UINT,
        };
        cache.device_mut().poll_progress()?;
        // Wait for and release incompatible unleased storage before allocating
        // either the new guest source or its snapshot. Live leases stay intact.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/present/layer.cpp
        for entry in &mut self.snapshots {
            if entry
                .as_ref()
                .is_some_and(|s| s.leases == 0 && (s.extent != source.extent || s.format != format))
            {
                let last_use = entry.as_ref().expect("incompatible snapshot").last_use;
                if last_use.tick() > cache.device().completed_tick() {
                    cache.flush()?;
                    cache
                        .device_mut()
                        .wait_for_work(last_use, std::time::Duration::MAX)?;
                }
                let old = entry.take().expect("completed snapshot");
                descriptors.forget_image(&old.view);
            }
        }
        let image = cache.presentation_source(&source, format)?;
        // The actual native use retains the selected source while its snapshot
        // allocation can invoke residency eviction.
        let (native, layout) = cache.host_source(image, source.extent)?;
        let completed = cache.device().completed_tick();
        let slot = self.snapshots.iter().position(|entry| {
            entry.as_ref().is_some_and(|s| {
                s.leases == 0
                    && s.last_use.tick() <= completed
                    && s.extent == source.extent
                    && s.format == format
            })
        });
        let slot = if let Some(slot) = slot {
            slot
        } else {
            let slot = self
                .snapshots
                .iter()
                .position(Option::is_none)
                .unwrap_or(self.snapshots.len());
            if slot == 12 {
                return Err(VulkanError::InvalidResource(
                    "presentation snapshot budget is saturated",
                ));
            }
            let generation = self
                .generation
                .checked_add(1)
                .ok_or(VulkanError::InvalidResource(
                    "presentation generation exhausted",
                ))?;
            let native = cache.allocate_host_image(source.extent, format)?;
            let view = super::render::view(cache.device(), native, format)?;
            let work = cache.device_mut().work_ticket()?;
            let snapshot = Snapshot {
                view,
                format,
                extent: source.extent,
                encoding: source.format,
                work,
                last_use: work,
                version: 0,
                generation,
                leases: 0,
            };
            // Vacant slots survive allocation failure. Every replacement gets a
            // fresh generation, even after a failed allocation or a slot change.
            self.generation = generation;
            if slot == self.snapshots.len() {
                self.snapshots.push(Some(snapshot));
            } else {
                self.snapshots[slot] = Some(snapshot);
            }
            slot
        };
        let snapshot = self.snapshots[slot].as_mut().expect("allocated snapshot");
        cache.device_mut().end_rendering();
        let index = cache.device_mut().begin_recording()?;
        let device = cache.device_mut();
        let command = device.submission.slots[index].command;
        super::render::transition(
            device,
            command,
            snapshot.view.image.handle,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::PipelineStageFlags2::NONE,
            vk::AccessFlags2::NONE,
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_WRITE,
        );
        let subresource = vk::ImageSubresourceLayers::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .layer_count(1);
        unsafe {
            device.dispatch.cmd_copy_image(
                command,
                native.handle,
                layout,
                snapshot.view.image.handle,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::ImageCopy::default()
                    .src_subresource(subresource)
                    .dst_subresource(subresource)
                    .extent(vk::Extent3D {
                        width: source.extent[0],
                        height: source.extent[1],
                        depth: 1,
                    })],
            );
        }
        super::render::transition(
            device,
            command,
            snapshot.view.image.handle,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_WRITE,
            vk::PipelineStageFlags2::FRAGMENT_SHADER,
            vk::AccessFlags2::SHADER_SAMPLED_READ,
        );
        let work = cache.device_mut().work_ticket()?;
        snapshot.work = work;
        snapshot.last_use = work;
        snapshot.encoding = source.format;
        snapshot.version = snapshot
            .version
            .checked_add(1)
            .expect("frame version exhausted");
        snapshot.leases = 1;
        // The producer ticket owns the destination even if the pool or its
        // consumer is dropped before this copy completes.
        cache
            .device_mut()
            .retire_image(Arc::clone(&snapshot.view.image), work)?;
        cache.flush()?;
        let description = ImageDescription::new(
            ImageDimension::Two,
            ImageExtent::new(source.extent[0], source.extent[1], 1)
                .map_err(|_| VulkanError::InvalidResource("invalid framebuffer extent"))?,
            ImageFormat::Rgba8Unorm,
            ImageKind::Color,
            1,
            1,
            SampleCount::One,
        )
        .map_err(|_| VulkanError::InvalidResource("invalid framebuffer description"))?;
        Ok(ResidentImage::from_lease(
            description,
            PresentationLease::new(
                PresentationImageId {
                    slot: slot as u64,
                    generation: snapshot.generation,
                },
                snapshot.version,
                work.submission(),
                self.sender.clone(),
            ),
        ))
    }
}
