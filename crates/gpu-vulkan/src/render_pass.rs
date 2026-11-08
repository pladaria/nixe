//! Native render passes and framebuffer ownership, as in Eden's RenderPassCache
//! and texture-cache Framebuffer. Pipeline compatibility uses formats/samples;
//! load/store and layout variants share the framebuffer and compiled pipeline.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/vk_render_pass_cache.cpp
use crate::{AllocatedImageView, VulkanError, device::DeviceOwner};
use ash::vk;
use rustc_hash::FxHashMap;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct Key {
    pub colors: [vk::Format; 8],
    pub depth: vk::Format,
    pub samples: vk::SampleCountFlags,
    pub layouts: [vk::ImageLayout; 9],
    pub color_clear: u8,
    pub color_discard: u8,
    pub color_dont_care: u8,
    pub depth_load: vk::AttachmentLoadOp,
    pub depth_store: vk::AttachmentStoreOp,
    pub stencil_load: vk::AttachmentLoadOp,
    pub stencil_store: vk::AttachmentStoreOp,
}
impl Default for Key {
    fn default() -> Self {
        Self {
            colors: [vk::Format::UNDEFINED; 8],
            depth: vk::Format::UNDEFINED,
            samples: vk::SampleCountFlags::TYPE_1,
            layouts: [vk::ImageLayout::GENERAL; 9],
            color_clear: 0,
            color_discard: 0,
            color_dont_care: 0,
            depth_load: vk::AttachmentLoadOp::LOAD,
            depth_store: vk::AttachmentStoreOp::STORE,
            stencil_load: vk::AttachmentLoadOp::LOAD,
            stencil_store: vk::AttachmentStoreOp::STORE,
        }
    }
}
impl Key {
    pub fn color_count(&self) -> usize {
        self.colors
            .iter()
            .rposition(|&f| f != vk::Format::UNDEFINED)
            .map_or(0, |i| i + 1)
    }
    pub fn color(format: vk::Format) -> Self {
        let mut key = Self::default();
        key.colors[0] = format;
        key
    }
}
pub(crate) type Cache = FxHashMap<Key, vk::RenderPass>;
impl DeviceOwner {
    pub(crate) fn render_pass(&self, key: Key) -> Result<vk::RenderPass, VulkanError> {
        let mut cache = self.render_passes.lock().unwrap();
        if let Some(&handle) = cache.get(&key) {
            return Ok(handle);
        }
        let mut attachments = smallvec::SmallVec::<[_; 9]>::new();
        let mut colors = [vk::AttachmentReference {
            attachment: vk::ATTACHMENT_UNUSED,
            layout: vk::ImageLayout::GENERAL,
        }; 8];
        for (i, &format) in key.colors.iter().enumerate() {
            if format == vk::Format::UNDEFINED {
                continue;
            }
            colors[i] = vk::AttachmentReference {
                attachment: attachments.len() as u32,
                layout: key.layouts[i],
            };
            attachments.push(
                vk::AttachmentDescription::default()
                    .format(format)
                    .samples(key.samples)
                    .load_op(if key.color_clear & (1 << i) != 0 {
                        vk::AttachmentLoadOp::CLEAR
                    } else if key.color_dont_care & (1 << i) != 0 {
                        vk::AttachmentLoadOp::DONT_CARE
                    } else {
                        vk::AttachmentLoadOp::LOAD
                    })
                    .store_op(if key.color_discard & (1 << i) != 0 {
                        vk::AttachmentStoreOp::DONT_CARE
                    } else {
                        vk::AttachmentStoreOp::STORE
                    })
                    .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .initial_layout(key.layouts[i])
                    .final_layout(key.layouts[i]),
            );
        }
        let depth_ref = vk::AttachmentReference {
            attachment: attachments.len() as u32,
            layout: key.layouts[8],
        };
        if key.depth != vk::Format::UNDEFINED {
            attachments.push(
                vk::AttachmentDescription::default()
                    .format(key.depth)
                    .samples(key.samples)
                    .load_op(key.depth_load)
                    .store_op(key.depth_store)
                    .stencil_load_op(key.stencil_load)
                    .stencil_store_op(key.stencil_store)
                    .initial_layout(key.layouts[8])
                    .final_layout(key.layouts[8]),
            );
        }
        let mut subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&colors[..key.color_count()]);
        if key.depth != vk::Format::UNDEFINED {
            subpass = subpass.depth_stencil_attachment(&depth_ref);
        }
        let subpasses = [subpass];
        let dependencies = [vk::SubpassDependency::default()
            .src_subpass(0)
            .dst_subpass(0)
            .src_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            )
            .dst_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
            .src_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            )
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .dependency_flags(
                vk::DependencyFlags::BY_REGION
                    | if self.attachment_feedback_loop {
                        vk::DependencyFlags::FEEDBACK_LOOP_EXT
                    } else {
                        vk::DependencyFlags::empty()
                    },
            )];
        let handle = unsafe {
            self.device.create_render_pass(
                &vk::RenderPassCreateInfo::default()
                    .attachments(&attachments)
                    .subpasses(&subpasses)
                    .dependencies(&dependencies),
                None,
            )
        }
        .map_err(|e| VulkanError::vk("native render pass", e))?;
        cache.insert(key, handle);
        Ok(handle)
    }
}
pub(crate) struct Framebuffer {
    _owner: Arc<DeviceOwner>,
    pub handle: vk::Framebuffer,
    pub render_pass: vk::RenderPass,
    pub key: Key,
    pub layers: u32,
    // Guest/owned host images stay alive through the native framebuffer. WSI
    // destroys its framebuffer before destroying its externally owned views.
    _views: Vec<Arc<AllocatedImageView>>,
}
impl Framebuffer {
    /// Raw views must outlive this framebuffer, including GPU use.
    pub unsafe fn new(
        owner: Arc<DeviceOwner>,
        key: Key,
        views: &[vk::ImageView],
        extent: vk::Extent2D,
        retained: Vec<Arc<AllocatedImageView>>,
    ) -> Result<Arc<Self>, VulkanError> {
        let pass = owner.render_pass(key)?;
        // Eden Framebuffer retains the layer extent of its attachment views.
        // https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/renderer_vulkan/vk_texture_cache.cpp
        let layers = retained
            .iter()
            .map(|view| view.range.layer_count)
            .max()
            .unwrap_or(1);
        let handle = unsafe {
            owner.device.create_framebuffer(
                &vk::FramebufferCreateInfo::default()
                    .render_pass(pass)
                    .attachments(views)
                    .width(extent.width.max(1))
                    .height(extent.height.max(1))
                    .layers(layers),
                None,
            )
        }
        .map_err(|e| VulkanError::vk("native framebuffer", e))?;
        Ok(Arc::new(Self {
            _owner: owner,
            handle,
            render_pass: pass,
            key,
            layers,
            _views: retained,
        }))
    }
    pub fn color(
        view: &Arc<AllocatedImageView>,
        format: vk::Format,
        extent: vk::Extent2D,
    ) -> Result<Arc<Self>, VulkanError> {
        unsafe {
            Self::new(
                Arc::clone(&view.image.owner),
                Key::color(format),
                &[view.handle],
                extent,
                vec![Arc::clone(view)],
            )
        }
    }
}
impl Drop for Framebuffer {
    fn drop(&mut self) {
        // Queue slots retain the framebuffer through its actual completion.
        // Attachment views remain alive until after this native handle dies.
        unsafe { self._owner.device.destroy_framebuffer(self.handle, None) };
    }
}
impl crate::VulkanDevice {
    pub(crate) fn begin_render_pass(
        &mut self,
        framebuffer: &Arc<Framebuffer>,
        key: Key,
        area: vk::Rect2D,
        clear: &[vk::ClearValue],
    ) -> Result<vk::RenderPass, VulkanError> {
        let pass = if key == framebuffer.key {
            framebuffer.render_pass
        } else {
            self.owner.render_pass(key)?
        };
        let slot = self.recording_slot()?;
        let retained = &mut self.submission.slots[slot];
        if retained.framebuffer_handles.insert(framebuffer.handle) {
            retained.framebuffers.push(Arc::clone(framebuffer));
        }
        unsafe {
            self.dispatch.begin_render_pass(
                &vk::RenderPassBeginInfo::default()
                    .render_pass(pass)
                    .framebuffer(framebuffer.handle)
                    .render_area(area)
                    .clear_values(clear),
            );
        }
        Ok(pass)
    }
}
