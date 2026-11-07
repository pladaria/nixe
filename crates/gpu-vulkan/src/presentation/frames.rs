//! Composite directly from the current texture-cache image before later guest work.
//! Only completed composition outputs cross the presentation boundary.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/renderer_vulkan.cpp#L204-L248
use super::*;
use crate::cache::{DescriptorCache, PipelineCompiler};
use ash::vk;
use nixe_gpu::{
    ImageDescription, ImageDimension, ImageExtent, ImageFormat, ImageKind,
    PresentationImageFormat as F, PresentationImageId, PresentationLease, PresentationRelease,
    ResidentImage, SampleCount, native::NativePresentationImage,
};
use std::time::Duration;

pub(crate) struct FrameStorage {
    pub copy: Arc<wsi::CopyCommands>,
    pub view: Arc<AllocatedImageView>,
    pub framebuffer: Arc<crate::render_pass::Framebuffer>,
    pub extent: [u32; 2],
    pub format: vk::Format,
}
#[derive(Clone)]
pub(crate) struct PresentFrame {
    pub storage: Arc<FrameStorage>,
    pub work: WorkTicket,
    pub sequence: Option<u64>,
}
pub(crate) struct Frame {
    pub storage: Arc<FrameStorage>,
    pub work: WorkTicket,
    last_use: WorkTicket,
    pub content: nixe_gpu::FrameCrop,
    pub display_extent: [u32; 2],
    pub version: u64,
    generation: u32,
    leases: usize,
}
impl Frame {
    pub fn layer(&self) -> render::LayerImage {
        render::LayerImage {
            view: Arc::clone(&self.storage.view),
            extent: self.storage.extent,
            format: self.storage.framebuffer.key.colors[0],
            encoding: F::Rgba8,
            crop: self.content,
            transform: nixe_gpu::FrameTransform::default(),
        }
    }
}
struct RawImage {
    view: Arc<AllocatedImageView>,
    image: Arc<AllocatedImage>,
    extent: [u32; 2],
    format: vk::Format,
    last_use: WorkTicket,
}
pub(crate) struct Frames {
    frames: Vec<Option<Frame>>,
    raw_image: Option<RawImage>,
    pub renderer: ScreenRenderer,
    generation: u32,
    sender: mpsc::Sender<PresentationRelease>,
    releases: mpsc::Receiver<PresentationRelease>,
}
impl Frames {
    pub fn new(device: &VulkanDevice) -> Result<Self, VulkanError> {
        let (sender, releases) = mpsc::channel();
        Ok(Self {
            frames: Vec::new(),
            raw_image: None,
            renderer: ScreenRenderer::new(device)?,
            generation: 0,
            sender,
            releases,
        })
    }
    fn drain(&mut self) {
        while let Ok(release) = self.releases.try_recv() {
            if let Some(Some(frame)) = self.frames.get_mut(release.image.slot as usize)
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
    pub fn get(&self, image: &ResidentImage) -> Result<&Frame, VulkanError> {
        let lease = image.lease();
        let frame = self
            .frames
            .get(lease.image().slot as usize)
            .and_then(Option::as_ref)
            .ok_or(VulkanError::InvalidResource("foreign composition slot"))?;
        if frame.generation != lease.image().generation
            || frame.version != lease.content_version()
            || frame.work.submission() != lease.producer()
            || frame.leases == 0
        {
            return Err(VulkanError::InvalidResource(
                "stale or foreign composed frame",
            ));
        }
        Ok(frame)
    }
    pub fn used(&mut self, image: &ResidentImage, work: WorkTicket) -> Result<(), VulkanError> {
        self.get(image)?;
        self.frames[image.lease().image().slot as usize]
            .as_mut()
            .unwrap()
            .last_use = work;
        Ok(())
    }
    fn storage(
        &mut self,
        cache: &mut crate::residency::Residency<'_>,
        extent: [u32; 2],
        format: vk::Format,
    ) -> Result<usize, VulkanError> {
        self.drain();
        cache.device_mut().poll_progress()?;
        let completed = cache.device().completed_tick();
        if let Some(slot) = self.frames.iter().position(|f| {
            f.as_ref().is_some_and(|f| {
                f.leases == 0
                    && Arc::strong_count(&f.storage) == 1
                    && f.last_use.tick() <= completed
                    && f.storage.extent == extent
                    && f.storage.format == format
            })
        }) {
            return Ok(slot);
        }
        // Resize only unleased storage after both graphics and WSI have released it.
        // The presentation worker holds the storage Arc until its copy fence signals.
        for entry in &mut self.frames {
            if entry.as_ref().is_some_and(|f| {
                f.leases == 0
                    && Arc::strong_count(&f.storage) == 1
                    && (f.storage.extent != extent || f.storage.format != format)
            }) {
                let work = entry.as_ref().unwrap().last_use;
                if work.tick() > completed {
                    cache.flush()?;
                    cache.device_mut().wait_for_work(work, Duration::MAX)?;
                }
                entry.take();
            }
        }
        let image = cache.allocate_host_image(extent, format)?;
        let view_format = match format {
            vk::Format::R8G8B8A8_SRGB => vk::Format::R8G8B8A8_UNORM,
            vk::Format::B8G8R8A8_SRGB => vk::Format::B8G8R8A8_UNORM,
            f => f,
        };
        let view = image.view_2d(view_format)?;
        let framebuffer = crate::render_pass::Framebuffer::color(
            &view,
            view_format,
            vk::Extent2D {
                width: extent[0],
                height: extent[1],
            },
        )?;
        let work = cache.device_mut().current_work();
        self.generation = self
            .generation
            .checked_add(1)
            .ok_or(VulkanError::InvalidResource(
                "composition generation exhausted",
            ))?;
        let copy = wsi::CopyCommands::new(cache.device())?;
        let frame = Frame {
            storage: Arc::new(FrameStorage {
                copy,
                view,
                framebuffer,
                extent,
                format,
            }),
            work,
            last_use: work,
            content: nixe_gpu::FrameCrop {
                left: 0,
                top: 0,
                width: extent[0],
                height: extent[1],
            },
            display_extent: extent,
            version: 0,
            generation: self.generation,
            leases: 0,
        };
        let slot = self
            .frames
            .iter()
            .position(Option::is_none)
            .unwrap_or(self.frames.len());
        if slot == self.frames.len() {
            self.frames.push(Some(frame));
        } else {
            self.frames[slot] = Some(frame);
        }
        Ok(slot)
    }
    /// Eden keeps raw presentation storage outside the guest texture cache.
    /// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/present/layer.cpp#L335-L389
    fn upload_raw(
        &mut self,
        cache: &mut crate::residency::Residency<'_>,
        source: &NativePresentationImage,
        format: vk::Format,
    ) -> Result<Arc<AllocatedImage>, VulkanError> {
        use nixe_gpu::{
            ImageMemoryLayout,
            physical::{ByteRect, ImageFootprint, ImageStorage, WriteDomain},
        };
        if self
            .raw_image
            .as_ref()
            .is_some_and(|image| image.extent != source.extent || image.format != format)
        {
            let last_use = self.raw_image.as_ref().unwrap().last_use;
            if last_use.tick() > cache.device().completed_tick() {
                cache.flush()?;
                cache.device_mut().wait_for_work(last_use, Duration::MAX)?;
            }
            self.raw_image.take();
        }
        let initialize = self.raw_image.is_none();
        if self.raw_image.is_none() {
            let image = cache.allocate_host_image(source.extent, format)?;
            let view = Arc::clone(&image).view_2d(format)?;
            self.raw_image = Some(RawImage {
                view,
                image,
                extent: source.extent,
                format,
                last_use: cache.device_mut().current_work(),
            });
        }
        let image = Arc::clone(&self.raw_image.as_ref().unwrap().image);
        let bytes_per_pixel = if matches!(source.format, F::Rgb565 | F::Rgba4444) {
            2
        } else {
            4
        };
        let row_pitch = u64::from(source.extent[0]) * u64::from(bytes_per_pixel);
        let size = row_pitch * u64::from(source.extent[1]);
        let storage = match source.layout {
            ImageMemoryLayout::PitchLinear { row_pitch, .. } => ImageStorage::Pitch { row_pitch },
            ImageMemoryLayout::BlockLinear(block) => ImageStorage::BlockLinear {
                block_height_log2: block.block_height_log2,
                block_depth_log2: block.block_depth_log2,
            },
        };
        let domain = WriteDomain::Image(ImageFootprint {
            depth_slice: 0,
            base: 0,
            columns: source.row_pitch / bytes_per_pixel,
            rows: source.extent[1],
            packed_bytes: bytes_per_pixel,
            aspect_offset: 0,
            aspect_bytes: bytes_per_pixel,
            storage,
            native_base: 0,
            native_element_bytes: bytes_per_pixel,
            native_row_pitch: row_pitch,
            rect: ByteRect {
                x: 0,
                y: 0,
                width: source.extent[0],
                height: source.extent[1],
            },
        });
        cache.device_mut().end_rendering()?;
        let runtime = &mut *cache.runtime;
        let memory = &runtime.device_reader;
        let device = &mut runtime.device;
        let slice =
            device
                .submission
                .reserve_slice(size, crate::queue::staging::Usage::Upload, 4)?;
        // SAFETY: this fresh reservation is retained until its native copy completes.
        unsafe {
            device.submission.staging.write_with(&slice, |output| {
                domain
                    .capture(
                        |offset, destination| {
                            if offset
                                .checked_add(destination.len() as u64)
                                .is_none_or(|end| end > source.size)
                            {
                                return Err(nixe_memory::MemoryError::Invalid(
                                    "framebuffer read exceeds its NvMap plane",
                                ));
                            }
                            memory.read_block_unsafe(source.address + offset, destination)
                        },
                        0,
                        output,
                    )
                    .map_err(VulkanError::from)
            })?;
        }
        let range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);
        if initialize {
            super::render::transition(
                device,
                image.handle,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
                vk::PipelineStageFlags2::NONE,
                vk::AccessFlags2::NONE,
                vk::PipelineStageFlags2::ALL_COMMANDS,
                vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
            );
        }

        let copy = vk::BufferImageCopy::default()
            .buffer_offset(slice.offset)
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1),
            )
            .image_extent(vk::Extent3D {
                width: source.extent[0],
                height: source.extent[1],
                depth: 1,
            });
        unsafe {
            device.dispatch.copy_buffer_to_image(
                device.submission.staging_buffer(&slice).handle,
                image.handle,
                range,
                &[copy],
            );
        }

        let work = device.current_work();
        device.retire_image(Arc::clone(&image), work)?;
        self.raw_image.as_mut().unwrap().last_use = work;
        cache.record_framebuffer_upload(size);
        Ok(image)
    }
    fn render_layer(
        &mut self,
        cache: &mut crate::residency::Residency<'_>,
        pipelines: &PipelineCompiler,
        descriptors: &mut DescriptorCache,
        layer: &render::LayerImage,
        extent: [u32; 2],
        format: vk::Format,
    ) -> Result<usize, VulkanError> {
        let slot = self.storage(cache, extent, format)?;
        let frame = self.frames[slot].as_mut().unwrap();
        let view_format = frame.storage.framebuffer.key.colors[0];
        let pipeline = self.renderer.pipeline(
            pipelines,
            view_format,
            u8::from(layer.format == vk::Format::R16_UINT),
        )?;
        cache.device_mut().end_rendering()?;
        render::transition(
            cache.device_mut(),
            frame.storage.view.image.handle,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
            vk::PipelineStageFlags2::NONE,
            vk::AccessFlags2::NONE,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        );
        self.renderer.screen(
            cache,
            descriptors,
            pipeline,
            layer,
            &frame.storage.framebuffer,
            vk::Extent2D {
                width: extent[0],
                height: extent[1],
            },
            view_format,
            true,
        )?;
        frame.content = layer.content_rect(extent);
        frame.display_extent = layer.dimensions();
        frame.work = cache.device_mut().current_work();
        frame.last_use = frame.work;
        frame.version = frame
            .version
            .checked_add(1)
            .expect("composition version exhausted");
        cache
            .device_mut()
            .retire_image(Arc::clone(&frame.storage.view.image), frame.work)?;
        Ok(slot)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn compose(
        &mut self,
        cache: &mut crate::residency::Residency<'_>,
        pipelines: &PipelineCompiler,
        descriptors: &mut DescriptorCache,
        host: Option<&mut Host>,
        source: NativePresentationImage,
        extent: [u32; 2],
        format: vk::Format,
    ) -> Result<ResidentImage, VulkanError> {
        let source_format = match source.format {
            F::Rgba8 | F::Rgbx8 => vk::Format::R8G8B8A8_UNORM,
            F::Bgra8 => vk::Format::B8G8R8A8_UNORM,
            F::Rgb565 | F::Rgba4444 => vk::Format::R16_UINT,
        };
        let id = cache.presentation_source(&source, source_format)?;
        let raw_source = id.is_none();
        // Keep the actual source view alive while offscreen allocation can evict caches.
        let native = if let Some(id) = id {
            cache.host_source(id, source.extent)?
        } else {
            self.upload_raw(cache, &source, source_format)?
        };
        let view = if let Some(id) = id {
            cache.presentation_view(id, source.extent, source_format)?
        } else {
            Arc::clone(&self.raw_image.as_ref().unwrap().view)
        };
        let layer_image = render::LayerImage {
            view,
            extent: source.extent,
            format: source_format,
            encoding: source.format,
            crop: source.crop,
            transform: source.transform,
        };
        if let Some(host) = host {
            host.capture_source(
                cache,
                &mut self.renderer,
                pipelines,
                descriptors,
                &layer_image,
                layer_image.dimensions(),
            )?;
        }
        let layer =
            self.render_layer(cache, pipelines, descriptors, &layer_image, extent, format)?;
        let work = self.frames[layer].as_ref().unwrap().work;
        cache.device_mut().retire_image(native, work)?;
        if raw_source {
            self.raw_image.as_mut().unwrap().last_use = work;
        }
        let frame = self.frames[layer].as_mut().unwrap();
        frame.leases = 1;
        let description = ImageDescription::new(
            ImageDimension::Two,
            ImageExtent::new(extent[0], extent[1], 1)
                .map_err(|_| VulkanError::InvalidResource("invalid composition extent"))?,
            ImageFormat::Rgba8Unorm,
            ImageKind::Color,
            1,
            1,
            SampleCount::One,
        )
        .map_err(|_| VulkanError::InvalidResource("invalid composition description"))?;
        let image = ResidentImage::from_lease(
            description,
            PresentationLease::new(
                PresentationImageId {
                    slot: layer as u64,
                    generation: frame.generation,
                },
                frame.version,
                frame.work.submission(),
                self.sender.clone(),
            ),
        );
        cache.flush()?;
        Ok(image)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_present(
        &mut self,
        cache: &mut crate::residency::Residency<'_>,
        pipelines: &PipelineCompiler,
        descriptors: &mut DescriptorCache,
        image: Option<&ResidentImage>,
        mut ui: Option<&mut UiFrame>,
        extent: [u32; 2],
        format: vk::Format,
        sequence: Option<u64>,
    ) -> Result<PresentFrame, VulkanError> {
        if ui.as_ref().is_none_or(|u| u.primitives.is_empty())
            && let Some(image) = image
        {
            let frame = self.get(image)?;
            if frame.storage.extent == extent && frame.storage.format == format {
                return Ok(PresentFrame {
                    storage: Arc::clone(&frame.storage),
                    work: frame.work,
                    sequence,
                });
            }
        }
        let slot = if let Some(image) = image {
            let layer = self.get(image)?.layer();
            let slot = self.render_layer(cache, pipelines, descriptors, &layer, extent, format)?;
            let work = self.frames[slot].as_ref().unwrap().work;
            self.used(image, work)?;
            slot
        } else {
            let slot = self.storage(cache, extent, format)?;
            let frame = self.frames[slot].as_ref().unwrap();
            cache.device_mut().end_rendering()?;
            render::transition(
                cache.device_mut(),
                frame.storage.view.image.handle,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
                vk::PipelineStageFlags2::NONE,
                vk::AccessFlags2::NONE,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            );
            let mut key = frame.storage.framebuffer.key;
            key.color_clear = 1;
            cache.device_mut().begin_render_pass(
                &frame.storage.framebuffer,
                key,
                vk::Rect2D {
                    offset: Default::default(),
                    extent: vk::Extent2D {
                        width: extent[0],
                        height: extent[1],
                    },
                },
                &[vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0., 0., 0., 1.],
                    },
                }],
            )?;
            unsafe {
                cache.device_mut().dispatch.end_rendering();
            }
            slot
        };
        let frame = self.frames[slot].as_mut().unwrap();
        if let Some(ui) = ui.as_mut()
            && !ui.primitives.is_empty()
        {
            let view_format = frame.storage.framebuffer.key.colors[0];
            let pipeline = self.renderer.pipeline(pipelines, view_format, 2)?;
            self.renderer.ui(
                cache,
                descriptors,
                pipeline,
                ui,
                &frame.storage.framebuffer,
                vk::Extent2D {
                    width: extent[0],
                    height: extent[1],
                },
                view_format,
            )?;
        }
        frame.work = cache.device_mut().current_work();
        frame.last_use = frame.work;
        cache
            .device_mut()
            .retire_image(Arc::clone(&frame.storage.view.image), frame.work)?;
        cache.flush()?;
        Ok(PresentFrame {
            storage: Arc::clone(&frame.storage),
            work: frame.work,
            sequence,
        })
    }
}
