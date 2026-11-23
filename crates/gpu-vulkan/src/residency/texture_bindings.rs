//! The texture cache owns current image slots and framebuffers. CPU-side
//! deletion invalidates selections in every affected cache-owned channel, under
//! the texture mutex, without sharing the GPU engine's register flags.
//! https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/texture_cache/texture_cache.h
use super::*;
/// Current TIC entries belong to the channel, not to a shader's descriptor ABI.
/// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/texture_cache/texture_cache.h
pub(crate) struct DescriptorTable<T> {
    pub(crate) table: Option<u64>,
    pub(crate) address_space: Option<u64>,
    pub(crate) maximum: u32,
    pub(crate) changes: u64,
    pub(crate) next_revision: u64,
    pub(crate) entries: Vec<DescriptorRecord<T>>,
}
impl<T> Default for DescriptorTable<T> {
    fn default() -> Self {
        Self {
            table: None,
            address_space: None,
            maximum: 0,
            changes: 1,
            next_revision: 0,
            entries: Vec::new(),
        }
    }
}
impl<T> DescriptorTable<T> {
    pub(crate) fn synchronize(&mut self, space: u64, pool: Option<(u64, u32)>) {
        let table = pool.map(|p| p.0);
        let maximum = pool.map_or(0, |p| p.1);
        if self.table != table || self.address_space != Some(space) || self.maximum != maximum {
            // A different pool no longer owns these retained fragmented inputs.
            // Keep the allocation capacity, but release obsolete physical pins.
            self.entries.clear();
            self.table = table;
            self.address_space = Some(space);
            self.maximum = maximum;
            self.invalidate();
        }
    }
    pub(crate) fn invalidate(&mut self) {
        self.changes = self
            .changes
            .checked_add(1)
            .expect("descriptor table revision exhausted");
    }
}
impl<T: BoundDescriptor> DescriptorTable<T> {
    /// Preparing one image can invalidate other views after their descriptor
    /// bytes were read. Validate the bound selection again at its actual use.
    pub(crate) fn selection(&mut self, index: u32) -> Option<&mut T> {
        let entry = &mut self.entries[index as usize];
        if entry.changes != self.changes {
            if let Some(value) = &mut entry.value {
                value.invalidate_selection();
            }
            self.next_revision = self
                .next_revision
                .checked_add(1)
                .expect("descriptor revision exhausted");
            entry.revision = self.next_revision;
            entry.changes = self.changes;
        }
        entry.value.as_mut()
    }
}
/// A channel/pool index owns one live descriptor and its physical read location.
/// Pipeline bindings keep revisions, not copies of these bytes or translations.
pub(crate) struct DescriptorRecord<T> {
    pub(crate) source: Option<DescriptorSource>,
    pub(crate) mapping: DescriptorMapping,
    pub(crate) changes: u64,
    pub(crate) revision: u64,
    pub(crate) bytes: [u8; 32],
    pub(crate) value: Option<T>,
}
impl<T> Default for DescriptorRecord<T> {
    fn default() -> Self {
        Self {
            source: None,
            mapping: DescriptorMapping::Guarded(0),
            changes: 0,
            revision: 0,
            bytes: [0; 32],
            value: None,
        }
    }
}
/// The pool's synchronized address space owns its records. A guarded mapping
/// pins VA and device backing together. Shared page tables can change during a read.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum DescriptorMapping {
    Guarded(u64),
    Concurrent { gpu: u64, device: u64 },
}
#[derive(Clone)]
pub(crate) enum DescriptorSource {
    Contiguous {
        device: u64,
        host: nixe_memory::HostAddress,
    },
    Captured(Arc<nixe_memory::CapturedMemory>),
    /// Fragmented or partly unmapped GPU memory. Its generation-specific reader
    /// preserves the memory manager's hole semantics under the consuming owner.
    Virtual(u64),
}
pub(crate) trait BoundDescriptor {
    #[cfg(feature = "performance-counters")]
    const DIAGNOSTIC_INDEX: usize;
    fn invalidate_selection(&mut self);
}
pub(crate) struct ImageDescriptor {
    /// Array/non-array selections for ordinary and depth-comparison sampling.
    pub(crate) views: [Option<ImageViewId>; 4],
    pub(crate) info: Option<nixe_gpu::native::ImageInfo>,
}
impl BoundDescriptor for ImageDescriptor {
    #[cfg(feature = "performance-counters")]
    const DIAGNOSTIC_INDEX: usize = 0;
    fn invalidate_selection(&mut self) {
        self.views = [None; 4];
    }
}
pub(crate) struct SamplerDescriptor {
    pub(crate) description: nixe_gpu::SamplerDescription,
    /// Independent reduction, comparison-disable and integer-filter variants.
    pub(crate) variants: [Option<vk::Sampler>; 8],
}
impl BoundDescriptor for SamplerDescriptor {
    #[cfg(feature = "performance-counters")]
    const DIAGNOSTIC_INDEX: usize = 1;
    fn invalidate_selection(&mut self) {}
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
    pub(crate) image_descriptors: [DescriptorTable<ImageDescriptor>; 2],
    pub(crate) sampler_descriptors: [DescriptorTable<SamplerDescriptor>; 2],
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
    pub(super) fn reset_resolution(&mut self) {
        for bank in std::iter::once(&mut *self.channel)
            .chain(self.channels.values_mut().flatten().map(Box::as_mut))
        {
            bank.rescaled = false;
            bank.invalidated_targets = 0x1ff;
            bank.last_framebuffer_id = None;
            bank.render_targets_changed();
        }
    }
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
                // Deletion can invalidate many channels repeatedly in one draw.
                // Clear bound selections lazily on their next descriptor read.
                table.invalidate();
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
        let d = &self.texture_cache.views[view].description;
        let slot = if d.aspects.contains(vk::ImageAspectFlags::COLOR) {
            0
        } else {
            8
        };
        let mut views = [None; 9];
        views[slot] = Some(view);
        self.framebuffer_for_views(views, [d.size.width, d.size.height])
    }
    pub(crate) fn framebuffer_for_views(
        &mut self,
        views: [Option<ImageViewId>; 9],
        size: [u32; 2],
    ) -> Result<Arc<execution::PreparedTargets>, VulkanError> {
        let key = FramebufferKey {
            views,
            size,
            draw_buffers: std::array::from_fn(|i| i as u8),
        };
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
