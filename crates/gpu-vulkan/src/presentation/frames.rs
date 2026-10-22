use super::*;
use ash::vk;
use nixe_gpu::{
    ImageDescription, ImageDimension, ImageExtent, ImageFormat, ImageKind,
    PresentationImageFormat as F, PresentationImageId, PresentationLease, PresentationRelease,
    ResidentImage, SampleCount, native::NativePresentationImage,
};
struct Import {
    planes: Arc<[nixe_gpu::native::GuestImagePlane]>,
    extent: [u32; 2],
    format: F,
    image: ImageId,
}
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
    imports: Vec<Import>,
    snapshots: Vec<Snapshot>,
    sender: mpsc::Sender<PresentationRelease>,
    releases: mpsc::Receiver<PresentationRelease>,
}
impl Frames {
    pub fn new() -> Self {
        let (sender, releases) = mpsc::channel();
        Self {
            imports: Vec::new(),
            snapshots: Vec::new(),
            sender,
            releases,
        }
    }
    fn drain(&mut self) {
        while let Ok(release) = self.releases.try_recv() {
            if let Some(frame) = self.snapshots.get_mut(release.image.slot as usize)
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
        self.snapshots[slot].last_use = work;
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
        let image = if let Some(import) = self.imports.iter().find(|i| {
            Arc::ptr_eq(&i.planes, &source.planes)
                && i.extent == source.extent
                && i.format == source.format
                && cache
                    .require_frame_source(i.image, source.extent, source.allow_canonical_import)
                    .is_ok()
        }) {
            import.image
        } else {
            if source.planes.len() != 1 {
                return Err(VulkanError::InvalidResource(
                    "framebuffer requires one color plane",
                ));
            }
            let plane = &source.planes[0];
            let description = NativeImageDescription {
                image_type: vk::ImageType::TYPE_2D,
                format,
                extent: vk::Extent3D {
                    width: source.extent[0],
                    height: source.extent[1],
                    depth: 1,
                },
                mip_levels: 1,
                array_layers: 1,
                samples: vk::SampleCountFlags::TYPE_1,
                usage: vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::SAMPLED,
                flags: vk::ImageCreateFlags::empty(),
            };
            let planes = vec![ImagePlane {
                offset: 0,
                row_pitch: plane.row_pitch,
                mip_level: 0,
                array_layer: 0,
                extent: description.extent,
                aspect: vk::ImageAspectFlags::COLOR,
                block_extent: vk::Extent2D {
                    width: 1,
                    height: 1,
                },
                bytes_per_block: plane.bytes_per_block,
            }];
            log::trace!(
                "native framebuffer import: extent={:?} format={:?} segments={}",
                source.extent,
                source.format,
                plane.footprint.mapping().spans().len()
            );
            let image = if let Some(image) =
                cache.resident_frame_source(&plane.footprint, description, &planes)
            {
                image
            } else if source.allow_canonical_import {
                cache.image_for(vec![Arc::clone(&plane.footprint)], description, planes)?
            } else {
                return Err(VulkanError::InvalidResource(
                    "opaque framebuffer requires a matching native resident representation",
                ));
            };
            log::trace!("native framebuffer resident source: {image:?}");
            self.imports
                .retain(|i| !Arc::ptr_eq(&i.planes, &source.planes));
            if self.imports.len() == 16 {
                self.imports.remove(0);
            }
            self.imports.push(Import {
                planes: Arc::clone(&source.planes),
                extent: source.extent,
                format: source.format,
                image,
            });
            image
        };
        if !source.allow_canonical_import {
            cache.require_resident_frame(image, source.extent)?;
        }
        // Keep the source allocation alive while charging/allocating its
        // snapshot; residency pressure must not evict a selected CPU mirror.
        let (native, layout) = cache.host_source(image, source.extent)?;
        cache.device_mut().poll_progress()?;
        let completed = cache.device().queue_usage().completed_tick;
        let slot = self.snapshots.iter().position(|s| {
            s.leases == 0
                && s.last_use.tick() <= completed
                && s.extent == source.extent
                && s.format == format
        });
        let slot = if let Some(slot) = slot {
            slot
        } else {
            let reusable = self
                .snapshots
                .iter()
                .position(|s| s.leases == 0 && s.last_use.tick() <= completed);
            if reusable.is_none() && self.snapshots.len() == 12 {
                return Err(VulkanError::InvalidResource(
                    "presentation snapshot budget is saturated",
                ));
            }
            let generation = if let Some(slot) = reusable {
                descriptors.forget_image(&self.snapshots[slot].view);
                self.snapshots[slot]
                    .generation
                    .checked_add(1)
                    .expect("presentation generation exhausted")
            } else {
                1
            };
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
            if let Some(slot) = reusable {
                self.snapshots[slot] = snapshot;
                slot
            } else {
                self.snapshots.push(snapshot);
                self.snapshots.len() - 1
            }
        };
        let snapshot = &mut self.snapshots[slot];
        cache.device_mut().end_rendering();
        let index = cache.device_mut().begin_recording()?;
        let device = cache.device();
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
