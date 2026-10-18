use super::*;
use crate::cache::{DescriptorCache, PipelineCompiler};
use ash::vk;
use std::collections::VecDeque;
struct Capture {
    view: Arc<AllocatedImageView>,
    extent: [u32; 2],
    next_row: u32,
    pending: VecDeque<(ReadbackTicket, u32)>,
    bytes: Vec<u8>,
    reply: mpsc::Sender<Result<Screenshot, String>>,
}
pub(crate) struct Host {
    pub client: PresentationClient,
    chains: Presenter,
    renderer: ScreenRenderer,
    source: Option<Source>,
    ui: Option<UiFrame>,
    size: [u32; 2],
    dirty: bool,
    screenshot: Option<mpsc::Sender<Result<Screenshot, String>>>,
    capture: Option<Capture>,
    pending_surface: Option<wsi::Acquired>,
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
            renderer: ScreenRenderer::new(device)?,
            source: None,
            ui: None,
            size: [0, 0],
            dirty: false,
            screenshot: None,
            capture: None,
            pending_surface: None,
        })
    }
    pub fn pending(&self) -> bool {
        self.dirty || self.capture.is_some() || self.client.0.lock().unwrap().pending.is_some()
    }
    pub fn service(
        &mut self,
        cache: &mut VulkanResidency,
        frames: &mut Frames,
        pipelines: &PipelineCompiler,
        descriptors: &mut DescriptorCache,
        samplers: &mut crate::cache::SamplerCache,
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
                // The immutable native snapshot retains pixels. Release the
                // Horizon slot lease now, independently of WSI availability.
            }
            // Coalescing an unpresented UI frame must still consume its frees.
            // Its texture uploads already ran on this owner, and any submitted
            // draw retains its view through the ordinary graphics timeline.
            if let Some(mut old_ui) = self.ui.take() {
                self.renderer.free_textures(&mut old_ui, descriptors);
            }
            if let Some(ui) = &mut packet.ui {
                self.renderer
                    .update_textures(cache, ui, descriptors, samplers)?;
                cache.flush()?;
            }
            self.ui = packet.ui.take();
            if packet.screenshot.is_some() {
                self.screenshot = packet.screenshot.take();
            }
            self.dirty = true;
        }
        self.service_capture(cache)?;
        if self.capture.is_none()
            && self.screenshot.is_some()
            && let Some(source) = &self.source
        {
            let frame = frames.get(&source.image)?;
            let kind = u8::from(frame.format == vk::Format::R16_UINT);
            if let Some(pipeline) =
                self.renderer
                    .pipeline(pipelines, vk::Format::R8G8B8A8_UNORM, kind)?
            {
                let extent = source.dimensions();
                let image = cache.allocate_host_image(extent, vk::Format::R8G8B8A8_UNORM)?;
                let view = render::view(cache.device(), image, vk::Format::R8G8B8A8_UNORM)?;
                cache.device_mut().end_rendering();
                let slot = cache.device_mut().begin_recording()?;
                let command = cache.device().submission.slots[slot].command;
                render::transition(
                    cache.device(),
                    command,
                    view.image.handle,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    vk::PipelineStageFlags2::NONE,
                    vk::AccessFlags2::NONE,
                    vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                    vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                );
                self.renderer.screen(
                    cache,
                    descriptors,
                    pipeline,
                    frame,
                    source,
                    view.handle,
                    vk::Extent2D {
                        width: extent[0],
                        height: extent[1],
                    },
                    vk::Format::R8G8B8A8_UNORM,
                    false,
                )?;
                render::transition(
                    cache.device(),
                    command,
                    view.image.handle,
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                    vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                    vk::PipelineStageFlags2::TRANSFER,
                    vk::AccessFlags2::TRANSFER_READ,
                );
                let work = cache.device_mut().work_ticket()?;
                frames.used(&source.image, work)?;
                cache
                    .device_mut()
                    .retire_image_views(vec![Arc::clone(&view)], work)?;
                self.capture = Some(Capture {
                    view,
                    extent,
                    next_row: 0,
                    pending: VecDeque::new(),
                    bytes: Vec::with_capacity(extent[0] as usize * extent[1] as usize * 4),
                    reply: self.screenshot.take().unwrap(),
                });
                self.service_capture(cache)?;
            }
        }
        if !self.dirty {
            return Ok(());
        }
        let Some(acquired) = self.chains.acquire(self.size)? else {
            return Ok(());
        };
        let format = acquired.format;
        let screen = if let Some(source) = &self.source {
            let kind = u8::from(frames.get(&source.image)?.format == vk::Format::R16_UINT);
            let Some(pipeline) = self.renderer.pipeline(pipelines, format, kind)? else {
                return Ok(());
            };
            Some(pipeline)
        } else {
            None
        };
        let ui_pipeline = if self.ui.is_some() {
            let Some(pipeline) = self.renderer.pipeline(pipelines, format, 2)? else {
                return Ok(());
            };
            Some(pipeline)
        } else {
            None
        };
        cache.device_mut().end_rendering();
        let slot = cache.device_mut().begin_recording()?;
        self.pending_surface = Some(acquired);
        let command = cache.device().submission.slots[slot].command;
        render::transition(
            cache.device(),
            command,
            acquired.image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            // Chain the transition with the acquire semaphore's wait.
            // https://docs.vulkan.org/guide/latest/synchronization_examples.html
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::NONE,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        );
        if let (Some(source), Some(pipeline)) = (&self.source, screen) {
            let frame = frames.get(&source.image)?;
            self.renderer.screen(
                cache,
                descriptors,
                pipeline,
                frame,
                source,
                acquired.view,
                acquired.extent,
                format,
                true,
            )?;
        } else {
            let attachments = [vk::RenderingAttachmentInfo::default()
                .image_view(acquired.view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0., 0., 0., 1.],
                    },
                })];
            let info = vk::RenderingInfo::default()
                .render_area(vk::Rect2D {
                    offset: Default::default(),
                    extent: acquired.extent,
                })
                .layer_count(1)
                .color_attachments(&attachments);
            unsafe {
                cache.device().dispatch.begin_rendering(command, &info);
                cache.device().dispatch.end_rendering(command);
            }
        }
        if let (Some(ui), Some(pipeline)) = (&mut self.ui, ui_pipeline) {
            self.renderer.ui(
                cache,
                descriptors,
                pipeline,
                ui,
                acquired.view,
                acquired.extent,
                format,
            )?;
        }
        render::transition(
            cache.device(),
            command,
            acquired.image,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            vk::PipelineStageFlags2::NONE,
            vk::AccessFlags2::NONE,
        );
        let work = cache.device_mut().work_ticket()?;
        if let Some(source) = &self.source {
            frames.used(&source.image, work)?;
        }
        let wait = [vk::SemaphoreSubmitInfo::default()
            .semaphore(acquired.wait)
            .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)];
        let signal = [vk::SemaphoreSubmitInfo::default()
            .semaphore(acquired.signal)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
        cache.flush_with_sync(&wait, &signal)?;
        self.pending_surface = None;
        self.chains
            .present(acquired, work, self.source.as_ref().map(|s| s.sequence))?;
        self.dirty = false;
        Ok(())
    }
    fn service_capture(&mut self, cache: &mut VulkanResidency) -> Result<(), VulkanError> {
        let Some(capture) = &mut self.capture else {
            return Ok(());
        };
        cache.device_mut().poll_progress()?;
        while capture.pending.front().is_some_and(|(ticket, _)| {
            ticket.work().tick() <= cache.device().queue_usage().completed_tick
        }) {
            let (ticket, _) = capture.pending.pop_front().unwrap();
            capture.bytes.extend(cache.device_mut().readback(ticket)?);
        }
        while capture.next_row < capture.extent[1] && capture.pending.len() < 2 {
            let rows = (1024 * 1024 / (capture.extent[0] as usize * 4))
                .max(1)
                .min((capture.extent[1] - capture.next_row) as usize) as u32;
            let size = u64::from(rows) * u64::from(capture.extent[0]) * 4;
            cache.device_mut().end_rendering();
            let (slot, slice, ticket) = match cache.device_mut().submission.reserve_readback(size) {
                Ok(value) => value,
                Err(VulkanError::QueuePressure { .. }) => {
                    cache.flush()?;
                    break;
                }
                Err(error) => return Err(error),
            };
            let device = cache.device();
            let command = device.submission.slots[slot].command;
            let copy = vk::BufferImageCopy::default()
                .buffer_offset(slice.offset)
                .image_offset(vk::Offset3D {
                    x: 0,
                    y: capture.next_row as i32,
                    z: 0,
                })
                .image_extent(vk::Extent3D {
                    width: capture.extent[0],
                    height: rows,
                    depth: 1,
                })
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .layer_count(1),
                );
            unsafe {
                device.dispatch.cmd_copy_image_to_buffer(
                    command,
                    capture.view.image.handle,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    device.submission.slots[slot].readback_buffer(slice).handle,
                    &[copy],
                );
            }
            cache
                .device_mut()
                .retire_image(Arc::clone(&capture.view.image), ticket.work())?;
            capture.pending.push_back((ticket, capture.next_row));
            capture.next_row += rows;
        }
        cache.flush()?;
        if capture.next_row == capture.extent[1] && capture.pending.is_empty() {
            let capture = self.capture.take().unwrap();
            let _ = capture.reply.send(Ok(Screenshot {
                width: capture.extent[0],
                height: capture.extent[1],
                rgba: capture.bytes,
            }));
        }
        Ok(())
    }
    pub fn shutdown(&mut self, cache: &mut VulkanResidency) -> Result<(), VulkanError> {
        self.client.close();
        if let Some(ui) = &mut self.ui {
            ui.textures.clear();
        }
        if let Some(frame) = self.pending_surface.take() {
            // An interrupted recording still consumes acquisition before any
            // submitted surface access. Track only real, accepted work in WSI.
            let waits = [vk::SemaphoreSubmitInfo::default()
                .semaphore(frame.wait)
                .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)];
            if let Some(work) = cache.flush_with_sync(&waits, &[])? {
                self.chains.graphics(work)?;
            }
        } else {
            cache.flush()?;
        }
        if let Some(capture) = self.capture.take() {
            for (ticket, _) in capture.pending {
                cache.device_mut().discard_readback(ticket)?;
            }
            let _ = capture.reply.send(Err(
                "presentation stopped before screenshot completion".into()
            ));
            let work = cache.device_mut().work_ticket()?;
            cache
                .device_mut()
                .retire_image_views(vec![capture.view], work)?;
            cache.flush()?;
        }
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
