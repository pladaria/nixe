use super::*;
use crate::cache::{DescriptorCache, PipelineCompiler};
use ash::vk;
use std::time::Duration;
pub(crate) struct Host {
    pub client: PresentationClient,
    chains: Presenter,
    source: Option<Source>,
    ui: Option<UiFrame>,
    size: [u32; 2],
    dirty: bool,
    screenshot: Option<mpsc::Sender<Result<Screenshot, String>>>,
}
impl Host {
    pub fn new(
        device: &VulkanDevice,
        client: PresentationClient,
        vsync: bool,
    ) -> Result<Self, VulkanError> {
        Ok(Self {
            client: client.clone(),
            chains: Presenter::new(device, client.clone(), vsync)?,
            source: None,
            ui: None,
            size: [0, 0],
            dirty: false,
            screenshot: None,
        })
    }
    pub fn output_format(&self) -> vk::Format {
        self.chains.format()
    }
    pub fn service(
        &mut self,
        cache: &mut crate::residency::Residency<'_>,
        frames: &mut Frames,
        pipelines: &PipelineCompiler,
        descriptors: &mut DescriptorCache,
    ) -> Result<(), VulkanError> {
        self.chains.poll()?;
        if self.client.stopped() {
            self.source = None;
            if let Some(mut ui) = self.ui.take() {
                ui.textures.clear();
            }
            return Ok(());
        }
        if let Some(mut packet) = self.client.take() {
            self.size = packet.size;
            if let Some(frame) = packet.frame.take() {
                self.source = Some(Source::from_frame(&frame));
                // The immutable composed frame retains pixels. Release the
                // Horizon slot lease now, independently of WSI availability.
            }
            // Coalescing an unpresented UI frame must still consume its frees.
            // Its texture uploads already ran on this owner, and any submitted
            // draw retains its view through the ordinary graphics timeline.
            if let Some(mut old_ui) = self.ui.take() {
                frames.renderer.free_textures(&mut old_ui);
            }
            if let Some(ui) = &mut packet.ui {
                frames.renderer.update_textures(cache, ui)?;
            }
            self.ui = packet.ui.take();
            if packet.screenshot.is_some() {
                self.screenshot = packet.screenshot.take();
            }
            self.dirty = true;
        }
        // A host-paused redraw has no new Composite. Capture the retained
        // composition, without rereading mutable guest memory or keeping a raw snapshot.
        if self.screenshot.is_some()
            && let Some(image) = self.source.as_ref().map(|s| s.image.clone())
        {
            let frame = frames.get(&image)?;
            let extent = frame.display_extent;
            let layer = frame.layer();
            if let Some(work) = self.capture_source(
                cache,
                &mut frames.renderer,
                pipelines,
                descriptors,
                &layer,
                extent,
            )? {
                frames.used(&image, work)?;
            }
        }
        if !self.dirty || self.size.contains(&0) || !self.chains.ready()? {
            return Ok(());
        }
        let frame = frames.prepare_present(
            cache,
            pipelines,
            descriptors,
            self.source.as_ref().map(|s| &s.image),
            self.ui.as_mut(),
            self.size,
            self.chains.format(),
            self.source.as_ref().map(|s| s.sequence),
        )?;
        self.chains.present(frame)?;
        self.dirty = false;
        Ok(())
    }
    pub fn capture_source(
        &mut self,
        cache: &mut crate::residency::Residency<'_>,
        renderer: &mut ScreenRenderer,
        pipelines: &PipelineCompiler,
        descriptors: &mut DescriptorCache,
        layer: &render::LayerImage,
        extent: [u32; 2],
    ) -> Result<Option<WorkTicket>, VulkanError> {
        let mut use_ticket = None;
        if self.screenshot.is_some() {
            let pipeline = renderer.pipeline(
                pipelines,
                vk::Format::R8G8B8A8_UNORM,
                u8::from(layer.format == vk::Format::R16_UINT),
            )?;
            {
                let image = cache.allocate_host_image(extent, vk::Format::R8G8B8A8_UNORM)?;
                let view = image.view_2d(vk::Format::R8G8B8A8_UNORM)?;
                cache.device_mut().end_rendering()?;

                render::transition(
                    cache.device_mut(),
                    view.image.handle,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::GENERAL,
                    vk::PipelineStageFlags2::NONE,
                    vk::AccessFlags2::NONE,
                    vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                    vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                );
                let framebuffer = crate::render_pass::Framebuffer::color(
                    &view,
                    vk::Format::R8G8B8A8_UNORM,
                    vk::Extent2D {
                        width: extent[0],
                        height: extent[1],
                    },
                )?;
                renderer.screen(
                    cache,
                    descriptors,
                    pipeline,
                    layer,
                    &framebuffer,
                    vk::Extent2D {
                        width: extent[0],
                        height: extent[1],
                    },
                    vk::Format::R8G8B8A8_UNORM,
                    false,
                )?;
                render::transition(
                    cache.device_mut(),
                    view.image.handle,
                    vk::ImageLayout::GENERAL,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                    vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                    vk::PipelineStageFlags2::TRANSFER,
                    vk::AccessFlags2::TRANSFER_READ,
                );
                let work = cache.device_mut().current_work();
                cache
                    .device_mut()
                    .retire_image_views(vec![Arc::clone(&view)], work)?;
                let device = cache.device_mut();
                let (slice, ticket) = device
                    .submission
                    .reserve_readback(u64::from(extent[0]) * u64::from(extent[1]) * 4)?;
                let device = cache.device_mut();
                let region = vk::BufferImageCopy::default()
                    .buffer_offset(slice.offset)
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .layer_count(1),
                    )
                    .image_extent(vk::Extent3D {
                        width: extent[0],
                        height: extent[1],
                        depth: 1,
                    });
                unsafe {
                    device.dispatch.cmd_copy_image_to_buffer(
                        view.image.handle,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        device.submission.staging_buffer(&slice).handle,
                        &[region],
                    );
                }
                let work = ticket.work();
                use_ticket = Some(work);
                cache
                    .device_mut()
                    .retire_image(Arc::clone(&view.image), work)?;
                // Eden RenderToBuffer performs one complete download and Finish before
                // publishing the screenshot. Completion needs no later guest frame.
                // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/renderer_vulkan.cpp#L270-L312
                cache.flush()?;
                cache.device_mut().wait_for_work(work, Duration::MAX)?;
                let rgba = ticket.completed()?.to_vec();
                let _ = self.screenshot.take().unwrap().send(Ok(Screenshot {
                    width: extent[0],
                    height: extent[1],
                    rgba,
                }));
            }
        }
        Ok(use_ticket)
    }
    pub fn shutdown(
        &mut self,
        cache: &mut crate::residency::Residency<'_>,
    ) -> Result<(), VulkanError> {
        self.client.close();
        if let Some(ui) = &mut self.ui {
            ui.textures.clear();
        }
        cache.flush()?;
        self.chains.shutdown()
    }
}

impl Drop for Host {
    fn drop(&mut self) {
        self.client.close();
        if let Some(ui) = &mut self.ui {
            ui.textures.clear();
        }
    }
}
