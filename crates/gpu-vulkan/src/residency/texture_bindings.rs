//! The texture cache owns current image slots and framebuffers. CPU-side
//! deletion invalidates selections in every affected cache-owned channel, under
//! the texture mutex, without sharing the GPU engine's register flags.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/texture_cache/texture_cache.h
use super::*;
pub(crate) struct TextureEntry {
    pub(crate) bytes: [u8; 32],
    pub(crate) info: Option<nixe_gpu_maxwell::engines::threed::ImageInfo>,
}
/// Current TIC entries belong to the channel, not to a shader's descriptor ABI.
/// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/texture_cache/texture_cache.h
#[derive(Default)]
pub(crate) struct ImageDescriptors {
    pub(crate) table: Option<u64>,
    pub(crate) address_space: Option<nixe_gpu_maxwell::MaxwellAddressSpaceId>,
    pub(crate) maximum: u32,
    pub(crate) entries: Vec<Option<ImageDescriptor>>,
}
pub(crate) struct ImageDescriptor {
    pub(crate) views: [Option<ImageViewId>; 2],
    pub(crate) raw: TextureEntry,
}
#[derive(Default)]
pub(crate) struct SamplerDescriptors {
    pub(crate) table: Option<u64>,
    pub(crate) address_space: Option<nixe_gpu_maxwell::MaxwellAddressSpaceId>,
    pub(crate) maximum: u32,
    pub(crate) entries: Vec<Option<SamplerDescriptor>>,
}
pub(crate) struct SamplerDescriptor {
    pub(crate) description: nixe_gpu::SamplerDescription,
    /// Original reduction and the variant for views without min/max filtering.
    pub(crate) variants: [Option<vk::Sampler>; 2],
    pub(crate) bytes: [u8; 32],
}
/// Channel selections and texture contents share the actual texture-cache
/// mutex, including CPU coherence and native image deletion.
/// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/control/channel_state_cache.h
#[derive(Default)]
pub(crate) struct TextureBindings {
    pub(crate) channel: Box<TextureChannel>,
    channels: HashMap<u64, Option<Box<TextureChannel>>>,
    current_channel: Option<u64>,
    pub(crate) has_deleted_images: bool,
}
#[derive(Default)]
pub(crate) struct TextureChannel {
    pub(crate) rescaled: bool,
    // Only texture-cache operations touch this mask, under its existing mutex.
    // A cleared native selection must be resolved even without a register write.
    pub(crate) invalidated_targets: u16,
    pub(crate) image_descriptors: [ImageDescriptors; 2],
    pub(crate) sampler_descriptors: [SamplerDescriptors; 2],
    pub(crate) render_targets: FramebufferKey,
    pub(crate) render_targets_serial: u64,
    pub(crate) last_framebuffer_serial: u64,
    pub(crate) last_framebuffer_id: Option<SlotId>,
    pub(crate) address_space: Option<u64>,
}
/// RenderTargets is the framebuffer cache identity, as in Eden. Native
/// attachment metadata and handles are created only for a new combination.
/// https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/texture_cache/render_targets.h
#[derive(Clone, Copy, Default, Hash, PartialEq, Eq)]
pub(crate) struct FramebufferKey {
    pub(crate) views: [Option<ImageViewId>; 9],
    pub(crate) draw_buffers: [u8; 8],
    pub(crate) size: [u32; 2],
}
impl TextureChannel {
    pub(crate) fn render_targets_changed(&mut self) {
        self.render_targets_serial = self
            .render_targets_serial
            .checked_add(1)
            .expect("render target serial exhausted");
    }
}
impl TextureBindings {
    pub(crate) fn create_channel(&mut self, id: u64) {
        assert!(self.channels.insert(id, Some(Box::default())).is_none());
    }
    pub(crate) fn bind_channel(&mut self, id: u64) {
        if self.current_channel == Some(id) {
            return;
        }
        let next = self
            .channels
            .get_mut(&id)
            .expect("registered texture channel")
            .take()
            .expect("inactive texture channel");
        let old = std::mem::replace(&mut self.channel, next);
        if let Some(previous) = self.current_channel.replace(id) {
            *self.channels.get_mut(&previous).unwrap() = Some(old);
        }
    }
    pub(crate) fn close_channel(&mut self, id: u64) {
        self.channels.remove(&id);
        if self.current_channel == Some(id) {
            self.current_channel = None;
            *self.channel = Default::default();
        }
    }
    pub(super) fn invalidate_image_views(&mut self, removed: &[ImageViewId]) {
        self.has_deleted_images = true;
        let invalidate = |bank: &mut TextureChannel| {
            for table in &mut bank.image_descriptors {
                for entry in table.entries.iter_mut().flatten() {
                    for view in &mut entry.views {
                        if view.is_some_and(|id| removed.contains(&id)) {
                            *view = None;
                        }
                    }
                }
            }
            let mut flags = 0;
            for (slot, current) in bank.render_targets.views.iter_mut().enumerate() {
                if current.is_some_and(|id| removed.contains(&id)) {
                    *current = None;
                    flags |= 1 << slot;
                }
            }
            bank.invalidated_targets |= flags;
            if flags != 0 {
                bank.last_framebuffer_id = None;
                bank.render_targets_changed();
            }
        };
        invalidate(&mut self.channel);
        for bank in self.channels.values_mut().flatten() {
            invalidate(bank);
        }
    }
}
impl Residency<'_> {
    /// GetFramebuffer returns the slot selected at the current render-target
    /// serial. Changed combinations resolve through TextureCache's single map.
    /// https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/texture_cache/texture_cache.h#L529-L582
    pub(crate) fn current_framebuffer(
        &mut self,
    ) -> Result<Arc<execution::PreparedTargets>, VulkanError> {
        let bank = &self.texture_cache.bindings.channel;
        if let Some(id) = bank.last_framebuffer_id
            && bank.last_framebuffer_serial == bank.render_targets_serial
        {
            return Ok(Arc::clone(&self.texture_cache.slot_framebuffers[id]));
        }
        let id = self.framebuffer_id(bank.render_targets)?;
        let bank = &mut self.texture_cache.bindings.channel;
        bank.last_framebuffer_id = Some(id);
        bank.last_framebuffer_serial = bank.render_targets_serial;
        Ok(Arc::clone(&self.texture_cache.slot_framebuffers[id]))
    }
    /// RenderTargetFromImage selects the full view extent. The operation's
    /// destination rectangle is dynamic viewport/scissor state, not cache identity.
    /// https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/texture_cache/texture_cache.h#L2660-L2684
    pub(crate) fn framebuffer_for_view(
        &mut self,
        view: ImageViewId,
    ) -> Result<Arc<execution::PreparedTargets>, VulkanError> {
        let description = &self.texture_cache.views[view].description;
        let mut key = FramebufferKey {
            size: [description.size.width, description.size.height],
            ..Default::default()
        };
        let slot = if description.aspects.contains(vk::ImageAspectFlags::COLOR) {
            0
        } else {
            8
        };
        key.views[slot] = Some(view);
        let id = self.framebuffer_id(key)?;
        Ok(Arc::clone(&self.texture_cache.slot_framebuffers[id]))
    }
    fn framebuffer_id(&mut self, key: FramebufferKey) -> Result<SlotId, VulkanError> {
        if let Some(&id) = self.texture_cache.framebuffers.get(&key) {
            return Ok(id);
        }
        self.runtime.device.realize_deferred_clear()?;
        let retained = self.runtime.constructing.len();
        for id in key.views.iter().copied().flatten() {
            self.runtime
                .constructing
                .push(Resource::Image(self.texture_cache.views[id].image));
        }
        let result = self.framebuffer_from_targets(key);
        self.runtime.constructing.truncate(retained);
        let native = result?;
        let id = self.texture_cache.slot_framebuffers.insert(native);
        self.texture_cache.framebuffers.insert(key, id);
        Ok(id)
    }
}
impl TextureCache {
    /// RemoveFramebuffers unregisters every combination containing a removed
    /// view. Delayed native destruction is owned by the existing retirement ring.
    /// https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/video_core/texture_cache/texture_cache.h#L2444-L2460
    pub(super) fn remove_framebuffers(&mut self, views: &[ImageViewId]) {
        if views.is_empty() {
            return;
        }
        let removed: smallvec::SmallVec<[_; 8]> = self
            .framebuffers
            .iter()
            .filter(|(key, _)| key.views.iter().flatten().any(|view| views.contains(view)))
            .map(|(&key, &id)| (key, id))
            .collect();
        for (key, id) in removed {
            self.framebuffers.remove(&key);
            self.retired
                .push_framebuffer(self.slot_framebuffers.erase(id));
        }
    }
}
