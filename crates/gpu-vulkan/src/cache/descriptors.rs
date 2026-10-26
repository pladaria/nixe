//! Immutable descriptor records. Content uploads are not binding identities.
//! Pooled sets are never individually freed or updated while in flight.
use super::*;
use crate::{AllocatedBuffer, AllocatedImageView, VulkanDevice, VulkanError, device::DeviceOwner};
use ash::{vk, vk::Handle};
use rustc_hash::FxHashMap as HashMap;
use std::{
    hash::{Hash, Hasher},
    sync::Arc,
};

pub struct Sampler {
    owner: Arc<DeviceOwner>,
    handle: vk::Sampler,
}
impl Sampler {
    pub fn new(
        device: &VulkanDevice,
        info: &vk::SamplerCreateInfo<'_>,
    ) -> Result<Arc<Self>, VulkanError> {
        if info.anisotropy_enable != 0
            && (device.profile().core.sampler_anisotropy == 0
                || info.max_anisotropy > device.profile().limits.max_sampler_anisotropy)
        {
            return Err(VulkanError::InvalidResource(
                "sampler anisotropy exceeds enabled host support",
            ));
        }
        let handle = unsafe { device.owner.device.create_sampler(info, None) }
            .map_err(|e| VulkanError::vk("native sampler", e))?;
        Ok(Arc::new(Self {
            owner: Arc::clone(&device.owner),
            handle,
        }))
    }
}
impl Drop for Sampler {
    fn drop(&mut self) {
        unsafe {
            self.owner.device.destroy_sampler(self.handle, None);
        }
    }
}
#[derive(Clone)]
pub enum DescriptorResource {
    NullBuffer,
    TexelBuffer {
        buffer: Arc<AllocatedBuffer>,
        offset: u64,
        range: u64,
        format: vk::Format,
        element_bytes: u32,
    },
    Buffer {
        buffer: Arc<AllocatedBuffer>,
        offset: u64,
        range: u64,
    },
    Image {
        view: Arc<AllocatedImageView>,
        layout: vk::ImageLayout,
    },
    CombinedImageSampler {
        view: Arc<AllocatedImageView>,
        layout: vk::ImageLayout,
        sampler: Arc<Sampler>,
    },
    Sampler(Arc<Sampler>),
}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum ResourceKey {
    NullBuffer,
    TexelBuffer(u64, u64, u64, vk::Format, u32),
    Buffer(u64, u64, u64),
    Image(u64, vk::ImageLayout),
    CombinedImageSampler(u64, vk::ImageLayout, u64),
    Sampler(u64),
}
impl DescriptorResource {
    fn key(&self) -> ResourceKey {
        match self {
            Self::NullBuffer => ResourceKey::NullBuffer,
            Self::TexelBuffer {
                buffer,
                offset,
                range,
                format,
                element_bytes,
            } => ResourceKey::TexelBuffer(
                buffer.handle.as_raw(),
                *offset,
                *range,
                *format,
                *element_bytes,
            ),
            Self::Buffer {
                buffer,
                offset,
                range,
            } => ResourceKey::Buffer(buffer.handle.as_raw(), *offset, *range),
            Self::Image { view, layout } => ResourceKey::Image(view.handle.as_raw(), *layout),
            Self::CombinedImageSampler {
                view,
                layout,
                sampler,
            } => ResourceKey::CombinedImageSampler(
                view.handle.as_raw(),
                *layout,
                sampler.handle.as_raw(),
            ),
            Self::Sampler(s) => ResourceKey::Sampler(s.handle.as_raw()),
        }
    }
    fn owner(&self) -> Option<&Arc<DeviceOwner>> {
        Some(match self {
            Self::NullBuffer => return None,
            Self::Buffer { buffer, .. } | Self::TexelBuffer { buffer, .. } => &buffer.owner,
            Self::Image { view, .. } | Self::CombinedImageSampler { view, .. } => &view.image.owner,
            Self::Sampler(s) => &s.owner,
        })
    }
}
// Borrowed lookups allocate and retain resources only on a real miss.
struct Key<'a> {
    layout: &'a LayoutKey,
    resources: &'a [DescriptorResource],
}
impl Hash for Key<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.layout.hash(state);
        self.resources.len().hash(state);
        for resource in self.resources {
            resource.key().hash(state);
        }
    }
}
struct Page {
    owner: Arc<DeviceOwner>,
    handle: vk::DescriptorPool,
    key: LayoutKey,
    capacity: u32,
}
impl Drop for Page {
    fn drop(&mut self) {
        unsafe {
            self.owner.device.destroy_descriptor_pool(self.handle, None);
        }
    }
}
struct PageSlot {
    page: Arc<Page>,
    allocated: u32,
}
pub struct BindingRecord {
    pub(crate) set: vk::DescriptorSet,
    pub layout: Arc<PipelineLayout>,
    _page: Arc<Page>,
    pub(crate) resources: Box<[DescriptorResource]>,
    _texel_views: Vec<BufferView>,
}
struct BufferView {
    owner: Arc<DeviceOwner>,
    handle: vk::BufferView,
    _buffer: Arc<AllocatedBuffer>,
}
impl Drop for BufferView {
    fn drop(&mut self) {
        unsafe {
            self.owner.device.destroy_buffer_view(self.handle, None);
        }
    }
}
#[derive(Clone, Copy, Debug, Default)]
pub struct DescriptorStatistics {
    pub hits: u64,
    pub allocated: u64,
    pub updates: u64,
    pub resets: u64,
}
struct Entry {
    record: Arc<BindingRecord>,
    touched: u64,
}
pub struct DescriptorCache {
    owner: Arc<DeviceOwner>,
    null_descriptor: bool,
    limits: vk::PhysicalDeviceLimits,
    maximum: usize,
    pages: Vec<PageSlot>,
    records: HashMap<u128, Entry>,
    clock: u64,
    statistics: DescriptorStatistics,
    counters: nixe_gpu::cache_counters::CacheCounters,
}
impl DescriptorCache {
    pub fn new(device: &VulkanDevice, maximum: usize) -> Self {
        Self::from_limits(
            Arc::clone(&device.owner),
            device.profile().limits,
            maximum,
            device.profile().null_descriptor,
        )
    }
    pub(crate) fn from_limits(
        owner: Arc<DeviceOwner>,
        limits: vk::PhysicalDeviceLimits,
        maximum: usize,
        null_descriptor: bool,
    ) -> Self {
        Self {
            owner,
            null_descriptor,
            limits,
            maximum,
            pages: Vec::new(),
            records: HashMap::default(),
            clock: 0,
            statistics: Default::default(),
            counters: nixe_gpu::cache_counters::CacheCounters::new(maximum, 0),
        }
    }
    pub fn statistics(&self) -> DescriptorStatistics {
        self.statistics
    }
    pub(crate) fn cache_counters(&self) -> nixe_gpu::cache_counters::CacheCounters {
        self.counters.clone()
    }
    /// A host texture's allocation has ended. Pending command slots retain
    /// their binding records; remove only the obsolete future-lookup entries.
    pub(crate) fn forget_image(&mut self, view: &AllocatedImageView) {
        self.records.retain(|_, entry| {
            !entry.record.resources.iter().any(
                |r| matches!(r.key(), ResourceKey::Image(handle, _) | ResourceKey::CombinedImageSampler(handle, _, _) if handle == view.handle.as_raw()),
            )
        });
    }
    pub fn get(
        &mut self,
        layout: &Arc<PipelineLayout>,
        resources: &[DescriptorResource],
    ) -> Result<Arc<BindingRecord>, VulkanError> {
        if !Arc::ptr_eq(&layout.owner, &self.owner)
            || resources.len() != layout.key.bindings.len()
            || resources.is_empty()
        {
            return Err(VulkanError::InvalidResource(
                "descriptor layout/resources are foreign, empty or have mismatched arity",
            ));
        }
        if resources
            .iter()
            .any(|r| r.owner().is_some_and(|owner| !Arc::ptr_eq(owner, &self.owner))
                || matches!(r, DescriptorResource::CombinedImageSampler { sampler, .. } if !Arc::ptr_eq(&sampler.owner, &self.owner)))
        {
            return Err(VulkanError::InvalidResource("foreign descriptor resource"));
        }
        let key = super::fingerprint::cache_fingerprint(&Key {
            layout: &layout.key,
            resources,
        });
        self.clock = self.clock.wrapping_add(1);
        if let Some(entry) = self.records.get_mut(&key) {
            debug_assert!(
                entry.record.layout.key == layout.key
                    && entry
                        .record
                        .resources
                        .iter()
                        .map(DescriptorResource::key)
                        .eq(resources.iter().map(DescriptorResource::key)),
                "XXH3-128 collision or incomplete descriptor key"
            );
            entry.touched = self.clock;
            self.statistics.hits += 1;
            return Ok(Arc::clone(&entry.record));
        }
        for (b, r) in layout.key.bindings.iter().zip(resources) {
            match (b.kind, r) {
                (
                    vk::DescriptorType::STORAGE_BUFFER
                    | vk::DescriptorType::STORAGE_BUFFER_DYNAMIC
                    | vk::DescriptorType::UNIFORM_BUFFER
                    | vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
                    DescriptorResource::NullBuffer,
                ) if self.null_descriptor => {}
                (
                    kind @ (vk::DescriptorType::UNIFORM_TEXEL_BUFFER
                    | vk::DescriptorType::STORAGE_TEXEL_BUFFER),
                    DescriptorResource::TexelBuffer {
                        buffer,
                        offset,
                        range,
                        format,
                        element_bytes,
                    },
                ) => {
                    let (usage, feature) = if kind == vk::DescriptorType::UNIFORM_TEXEL_BUFFER {
                        (
                            vk::BufferUsageFlags::UNIFORM_TEXEL_BUFFER,
                            vk::FormatFeatureFlags::UNIFORM_TEXEL_BUFFER,
                        )
                    } else {
                        (
                            vk::BufferUsageFlags::STORAGE_TEXEL_BUFFER,
                            vk::FormatFeatureFlags::STORAGE_TEXEL_BUFFER,
                        )
                    };
                    let properties = unsafe {
                        self.owner
                            .instance
                            .instance
                            .get_physical_device_format_properties(self.owner.physical, *format)
                    };
                    if !buffer.usage.contains(usage)
                        || !properties.buffer_features.contains(feature)
                        || *element_bytes == 0
                        || *range == 0
                        || !range.is_multiple_of(u64::from(*element_bytes))
                        || range / u64::from((*element_bytes).max(1))
                            > u64::from(self.limits.max_texel_buffer_elements)
                        || !offset
                            .is_multiple_of(self.limits.min_texel_buffer_offset_alignment.max(1))
                        || offset
                            .checked_add(*range)
                            .is_none_or(|end| end > buffer.size())
                    {
                        return Err(VulkanError::InvalidResource(
                            "texel buffer format, usage, range or alignment exceeds host support",
                        ));
                    }
                }
                (
                    vk::DescriptorType::STORAGE_BUFFER
                    | vk::DescriptorType::STORAGE_BUFFER_DYNAMIC
                    | vk::DescriptorType::UNIFORM_BUFFER
                    | vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
                    DescriptorResource::Buffer {
                        buffer,
                        offset,
                        range,
                    },
                ) => {
                    let usage = if matches!(
                        b.kind,
                        vk::DescriptorType::UNIFORM_BUFFER
                            | vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
                    ) {
                        vk::BufferUsageFlags::UNIFORM_BUFFER
                    } else {
                        vk::BufferUsageFlags::STORAGE_BUFFER
                    };
                    if !buffer.usage.contains(usage) {
                        return Err(VulkanError::InvalidResource(
                            "buffer descriptor usage is not enabled",
                        ));
                    }
                    let (alignment, maximum) = if matches!(
                        b.kind,
                        vk::DescriptorType::UNIFORM_BUFFER
                            | vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
                    ) {
                        (
                            self.limits.min_uniform_buffer_offset_alignment,
                            self.limits.max_uniform_buffer_range,
                        )
                    } else {
                        (
                            self.limits.min_storage_buffer_offset_alignment,
                            self.limits.max_storage_buffer_range,
                        )
                    };
                    if *range == 0
                        || *range > u64::from(maximum)
                        || offset
                            .checked_add(*range)
                            .is_none_or(|end| end > buffer.size())
                        || !offset.is_multiple_of(alignment)
                    {
                        return Err(VulkanError::InvalidResource(
                            "buffer descriptor range/alignment exceeds host contract",
                        ));
                    }
                }
                (
                    vk::DescriptorType::SAMPLED_IMAGE | vk::DescriptorType::STORAGE_IMAGE,
                    DescriptorResource::Image { layout, .. },
                )
                | (
                    vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    DescriptorResource::CombinedImageSampler { layout, .. },
                ) => {
                    if b.kind == vk::DescriptorType::STORAGE_IMAGE
                        && *layout != vk::ImageLayout::GENERAL
                    {
                        return Err(VulkanError::InvalidResource(
                            "storage image descriptor must use GENERAL",
                        ));
                    }
                    if !matches!(
                        *layout,
                        vk::ImageLayout::GENERAL
                            | vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                            | vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL
                            | vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL
                            | vk::ImageLayout::STENCIL_READ_ONLY_OPTIMAL
                            | vk::ImageLayout::READ_ONLY_OPTIMAL
                            | vk::ImageLayout::ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT
                    ) {
                        return Err(VulkanError::InvalidResource(
                            "sampled image descriptor requires a shader-readable layout",
                        ));
                    }
                }
                (vk::DescriptorType::SAMPLER, DescriptorResource::Sampler(_)) => {}
                _ => {
                    return Err(VulkanError::InvalidResource(
                        "descriptor resource class mismatch",
                    ));
                }
            }
        }
        let page = self.page_for(layout)?;
        let layouts = [layout.set];
        let set = unsafe {
            self.owner.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(self.pages[page].page.handle)
                    .set_layouts(&layouts),
            )
        }
        .map_err(|e| VulkanError::vk("native descriptor allocation", e))?[0];
        self.pages[page].allocated += 1;
        self.counters.add_entries(1);
        self.counters.miss();
        let buffers: Vec<_> = resources
            .iter()
            .map(|r| match r {
                DescriptorResource::NullBuffer => {
                    vk::DescriptorBufferInfo::default().range(vk::WHOLE_SIZE)
                }
                DescriptorResource::Buffer {
                    buffer,
                    offset,
                    range,
                } => vk::DescriptorBufferInfo::default()
                    .buffer(buffer.handle)
                    .offset(*offset)
                    .range(*range),
                _ => Default::default(),
            })
            .collect();
        let images: Vec<_> = resources
            .iter()
            .map(|r| match r {
                DescriptorResource::Image { view, layout } => vk::DescriptorImageInfo::default()
                    .image_view(view.handle)
                    .image_layout(*layout),
                DescriptorResource::Sampler(s) => {
                    vk::DescriptorImageInfo::default().sampler(s.handle)
                }
                DescriptorResource::CombinedImageSampler {
                    view,
                    layout,
                    sampler,
                } => vk::DescriptorImageInfo::default()
                    .image_view(view.handle)
                    .image_layout(*layout)
                    .sampler(sampler.handle),
                _ => Default::default(),
            })
            .collect();
        let mut texel_views = Vec::new();
        let mut texels = vec![vk::BufferView::null(); resources.len()];
        for (index, resource) in resources.iter().enumerate() {
            if let DescriptorResource::TexelBuffer {
                buffer,
                offset,
                range,
                format,
                ..
            } = resource
            {
                let handle = unsafe {
                    self.owner.device.create_buffer_view(
                        &vk::BufferViewCreateInfo::default()
                            .buffer(buffer.handle)
                            .format(*format)
                            .offset(*offset)
                            .range(*range),
                        None,
                    )
                }
                .map_err(|e| VulkanError::vk("native texel buffer view", e))?;
                texel_views.push(BufferView {
                    owner: Arc::clone(&self.owner),
                    handle,
                    _buffer: Arc::clone(buffer),
                });
                texels[index] = handle;
            }
        }
        let writes: Vec<_> = layout
            .key
            .bindings
            .iter()
            .enumerate()
            .map(|(i, b)| {
                let write = vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(b.binding)
                    .dst_array_element(b.element)
                    .descriptor_type(b.kind);
                if matches!(
                    resources[i],
                    DescriptorResource::Buffer { .. } | DescriptorResource::NullBuffer
                ) {
                    write.buffer_info(&buffers[i..i + 1])
                } else if matches!(resources[i], DescriptorResource::TexelBuffer { .. }) {
                    write.texel_buffer_view(&texels[i..i + 1])
                } else {
                    write.image_info(&images[i..i + 1])
                }
            })
            .collect();
        unsafe {
            self.owner.device.update_descriptor_sets(&writes, &[]);
        }
        self.statistics.allocated += 1;
        self.statistics.updates += 1;
        let record = Arc::new(BindingRecord {
            set,
            layout: Arc::clone(layout),
            _page: Arc::clone(&self.pages[page].page),
            resources: resources.into(),
            _texel_views: texel_views,
        });
        self.records.insert(
            key,
            Entry {
                record: Arc::clone(&record),
                touched: self.clock,
            },
        );
        Ok(record)
    }
    fn page_for(&mut self, layout: &PipelineLayout) -> Result<usize, VulkanError> {
        if let Some(i) = self
            .pages
            .iter()
            .position(|p| p.page.key == layout.key && p.allocated < p.page.capacity)
        {
            return Ok(i);
        }
        loop {
            if let Some(i) = self.pages.iter().position(|p| {
                Arc::strong_count(&p.page) == 1 && (p.allocated != 0 || p.page.key != layout.key)
            }) {
                // Every record and GPU batch has released this page. The reset
                // never depends on a guessed frame count or an idle wait.
                unsafe {
                    self.owner.device.reset_descriptor_pool(
                        self.pages[i].page.handle,
                        vk::DescriptorPoolResetFlags::empty(),
                    )
                }
                .map_err(|e| VulkanError::vk("recycle descriptor page", e))?;
                self.statistics.resets += 1;
                self.counters
                    .remove_entries(self.pages[i].allocated as usize);
                self.pages[i].allocated = 0;
                if self.pages[i].page.key == layout.key {
                    return Ok(i);
                }
                self.pages.remove(i);
                continue;
            }
            let capacity: usize = self.pages.iter().map(|p| p.page.capacity as usize).sum();
            if capacity < self.maximum {
                // Sparse interfaces do not reserve 64 sets before a second
                // layout can bind. Hot layouts grow geometrically; total native
                // pool capacity still obeys the configured hard budget.
                let growth = self
                    .pages
                    .iter()
                    .filter(|p| p.page.key == layout.key)
                    .map(|p| p.page.capacity)
                    .max()
                    .map_or(4, |count| count.saturating_mul(2).min(64));
                let count = (self.maximum - capacity).min(growth as usize) as u32;
                let mut counts = HashMap::<vk::DescriptorType, u32>::default();
                for b in &layout.key.bindings {
                    *counts.entry(b.kind).or_default() += count;
                }
                let sizes: Vec<_> = counts
                    .into_iter()
                    .map(|(kind, count)| {
                        vk::DescriptorPoolSize::default()
                            .ty(kind)
                            .descriptor_count(count)
                    })
                    .collect();
                let handle = unsafe {
                    self.owner.device.create_descriptor_pool(
                        &vk::DescriptorPoolCreateInfo::default()
                            .max_sets(count)
                            .pool_sizes(&sizes),
                        None,
                    )
                }
                .map_err(|e| VulkanError::vk("native descriptor page", e))?;
                self.pages.push(PageSlot {
                    page: Arc::new(Page {
                        owner: Arc::clone(&self.owner),
                        handle,
                        key: layout.key.clone(),
                        capacity: count,
                    }),
                    allocated: 0,
                });
                return Ok(self.pages.len() - 1);
            }
            let Some(key) = self
                .records
                .iter()
                .filter(|(_, e)| Arc::strong_count(&e.record) == 1)
                .min_by_key(|(_, e)| e.touched)
                .map(|(&k, _)| k)
            else {
                return Err(VulkanError::DescriptorPressure);
            };
            self.records.remove(&key);
            self.counters.evict(1);
        }
    }
    /// Drop only reconstructible index ownership under residency pressure.
    /// Prepared and in-flight records keep their resource/pool leases alive.
    pub fn trim(&mut self) {
        let previous = self.records.len();
        self.records.retain(|_, e| Arc::strong_count(&e.record) > 1);
        self.counters.evict((previous - self.records.len()) as u64);
    }
}

impl VulkanDevice {
    pub fn bind_native_pipeline(
        &mut self,
        pipeline: &Arc<NativePipeline>,
    ) -> Result<(), VulkanError> {
        if !Arc::ptr_eq(&self.owner, &pipeline.owner) {
            return Err(VulkanError::InvalidResource("foreign native pipeline"));
        }
        let index = self.begin_recording()?;
        let slot = &mut self.submission.slots[index];
        let bound = if pipeline.bind_point == vk::PipelineBindPoint::COMPUTE {
            &mut slot.bound_compute
        } else {
            &mut slot.bound_graphics
        };
        // Slot retention prevents this identity from being recycled while any
        // queued bind or native use still refers to the pipeline object.
        let identity = Arc::as_ptr(pipeline) as usize;
        if *bound == identity {
            return Ok(());
        }
        unsafe {
            self.dispatch
                .cmd_bind_pipeline(pipeline.bind_point, Arc::clone(pipeline));
        }
        self.submission.statistics.pipeline_binds += 1;
        *bound = identity;
        if slot.pipeline_handles.insert(identity) {
            slot.native_pipelines.push(Arc::clone(pipeline));
        }
        Ok(())
    }
    pub fn bind_native_descriptors(
        &mut self,
        pipeline: &NativePipeline,
        record: &Arc<BindingRecord>,
        dynamic_offsets: &[u32],
    ) -> Result<(), VulkanError> {
        if !Arc::ptr_eq(&self.owner, &record.layout.owner)
            || !Arc::ptr_eq(&self.owner, &pipeline.owner)
        {
            return Err(VulkanError::InvalidResource(
                "foreign native descriptor/pipeline",
            ));
        }
        let limits = self.profile().limits;
        let index = self.begin_recording()?;
        let slot = &mut self.submission.slots[index];
        let bound = if pipeline.bind_point == vk::PipelineBindPoint::COMPUTE {
            &mut slot.bound_set
        } else {
            &mut slot.bound_graphics_set
        };
        let bound_layout = if pipeline.bind_point == vk::PipelineBindPoint::COMPUTE {
            &mut slot.bound_compute_layout
        } else {
            &mut slot.bound_graphics_layout
        };
        let bound_offsets = if pipeline.bind_point == vk::PipelineBindPoint::COMPUTE {
            &mut slot.bound_compute_offsets
        } else {
            &mut slot.bound_graphics_offsets
        };
        if *bound == record.set
            && *bound_layout == pipeline.layout.handle
            && *bound_offsets == dynamic_offsets
        {
            return Ok(());
        }
        if record.layout.key != pipeline.layout.key {
            return Err(VulkanError::InvalidResource(
                "native descriptor/pipeline ABI mismatch",
            ));
        }
        if dynamic_offsets.len() != record.layout.dynamic_bindings.len() {
            return Err(VulkanError::InvalidResource(
                "native dynamic descriptor offset count mismatch",
            ));
        }
        for (&index, &dynamic) in record.layout.dynamic_bindings.iter().zip(dynamic_offsets) {
            if matches!(record.resources[index], DescriptorResource::NullBuffer) && dynamic == 0 {
                continue;
            }
            let DescriptorResource::Buffer {
                buffer,
                offset,
                range,
            } = &record.resources[index]
            else {
                unreachable!()
            };
            let alignment =
                layout::buffer_alignment(record.layout.key.bindings[index].kind, &limits);
            if !u64::from(dynamic).is_multiple_of(alignment)
                || offset
                    .checked_add(u64::from(dynamic))
                    .and_then(|n| n.checked_add(*range))
                    .is_none_or(|end| end > buffer.size())
            {
                return Err(VulkanError::InvalidResource(
                    "dynamic descriptor offset exceeds buffer bounds/alignment",
                ));
            }
        }
        unsafe {
            self.dispatch.cmd_bind_descriptor_sets(
                pipeline.bind_point,
                pipeline.layout.handle,
                0,
                &[record.set],
                dynamic_offsets,
            );
        }
        self.submission.statistics.descriptor_binds += 1;
        bound_offsets.clear();
        bound_offsets.extend_from_slice(dynamic_offsets);
        *bound = record.set;
        *bound_layout = pipeline.layout.handle;
        if slot.binding_handles.insert(record.set) {
            slot.native_bindings.push(Arc::clone(record));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct SamplerKey {
    flags: vk::SamplerCreateFlags,
    mag: vk::Filter,
    min: vk::Filter,
    mip: vk::SamplerMipmapMode,
    addresses: [vk::SamplerAddressMode; 3],
    bias: u32,
    anisotropy: bool,
    maximum: u32,
    compare: bool,
    operation: vk::CompareOp,
    lod: [u32; 2],
    border: vk::BorderColor,
    unnormalized: bool,
    extensions: SamplerExtensions,
}
/// Immutable sampler objects share the same completion lifetime through binding
/// records. Hashing/creation only occurs when sampler state changes.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CustomBorderColor {
    /// Raw components preserve float NaNs/signed zero and integer border colors.
    pub components: [u32; 4],
    pub format: vk::Format,
}
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SamplerExtensions {
    pub custom_border: Option<CustomBorderColor>,
    pub reduction: vk::SamplerReductionMode,
}
impl Default for SamplerExtensions {
    fn default() -> Self {
        Self {
            custom_border: None,
            reduction: vk::SamplerReductionMode::WEIGHTED_AVERAGE,
        }
    }
}
pub struct SamplerCache {
    owner: Arc<DeviceOwner>,
    features: vk::PhysicalDeviceFeatures,
    limits: vk::PhysicalDeviceLimits,
    entries: HashMap<SamplerKey, (Arc<Sampler>, u64)>,
    maximum: usize,
    clock: u64,
    custom_border: bool,
    custom_without_format: bool,
    reduction: bool,
}
impl SamplerCache {
    pub fn new(device: &VulkanDevice, maximum: usize) -> Self {
        Self {
            owner: Arc::clone(&device.owner),
            features: device.profile().core,
            limits: device.profile().limits,
            entries: HashMap::default(),
            maximum,
            clock: 0,
            custom_border: device.profile().custom_border_color,
            custom_without_format: device.profile().custom_border_color_without_format,
            reduction: device.profile().sampler_filter_minmax,
        }
    }
    pub fn get(&mut self, info: &vk::SamplerCreateInfo<'_>) -> Result<Arc<Sampler>, VulkanError> {
        self.get_extended(info, SamplerExtensions::default())
    }
    pub fn get_extended(
        &mut self,
        info: &vk::SamplerCreateInfo<'_>,
        extensions: SamplerExtensions,
    ) -> Result<Arc<Sampler>, VulkanError> {
        // Extension state is owned and included in the key, never pointer-hashed.
        if !info.p_next.is_null() {
            return Err(VulkanError::InvalidResource(
                "sampler extension state is not represented by the native sampler key",
            ));
        }
        let custom = matches!(
            info.border_color,
            vk::BorderColor::FLOAT_CUSTOM_EXT | vk::BorderColor::INT_CUSTOM_EXT
        );
        if custom != extensions.custom_border.is_some()
            || custom
                && (!self.custom_border
                    || extensions.custom_border.unwrap().format == vk::Format::UNDEFINED
                        && !self.custom_without_format)
        {
            return Err(VulkanError::InvalidResource(
                "custom border requires enabled customBorderColors and an explicit supported format",
            ));
        }
        if extensions.reduction != vk::SamplerReductionMode::WEIGHTED_AVERAGE
            && (!self.reduction
                || info.compare_enable != 0
                || info.unnormalized_coordinates != 0
                || !matches!(
                    extensions.reduction,
                    vk::SamplerReductionMode::MIN | vk::SamplerReductionMode::MAX
                ))
        {
            return Err(VulkanError::InvalidResource(
                "sampler reduction requires enabled samplerFilterMinmax and normalized non-comparison state",
            ));
        }
        let key = SamplerKey {
            flags: info.flags,
            mag: info.mag_filter,
            min: info.min_filter,
            mip: info.mipmap_mode,
            addresses: [
                info.address_mode_u,
                info.address_mode_v,
                info.address_mode_w,
            ],
            bias: info.mip_lod_bias.to_bits(),
            anisotropy: info.anisotropy_enable != 0,
            maximum: if info.anisotropy_enable == 0 {
                0
            } else {
                info.max_anisotropy.to_bits()
            },
            compare: info.compare_enable != 0,
            operation: if info.compare_enable == 0 {
                vk::CompareOp::ALWAYS
            } else {
                info.compare_op
            },
            lod: [info.min_lod.to_bits(), info.max_lod.to_bits()],
            border: info.border_color,
            unnormalized: info.unnormalized_coordinates != 0,
            extensions,
        };
        self.clock = self.clock.wrapping_add(1);
        if let Some((sampler, touched)) = self.entries.get_mut(&key) {
            *touched = self.clock;
            return Ok(Arc::clone(sampler));
        }
        if !info.mip_lod_bias.is_finite()
            || info.mip_lod_bias.abs() > self.limits.max_sampler_lod_bias
            || info.anisotropy_enable != 0
                && (self.features.sampler_anisotropy == 0
                    || !info.max_anisotropy.is_finite()
                    || info.max_anisotropy < 1.
                    || info.max_anisotropy > self.limits.max_sampler_anisotropy)
        {
            return Err(VulkanError::InvalidResource(
                "sampler state exceeds enabled host numerical/anisotropy limits",
            ));
        }
        if !info.min_lod.is_finite()
            || !info.max_lod.is_finite()
            || info.max_lod < info.min_lod
            || info.unnormalized_coordinates != 0
                && (info.min_filter != info.mag_filter
                    || info.mipmap_mode != vk::SamplerMipmapMode::NEAREST
                    || info.min_lod != 0.
                    || info.max_lod != 0.
                    || info.anisotropy_enable != 0
                    || info.compare_enable != 0
                    || [info.address_mode_u, info.address_mode_v].iter().any(|m| {
                        !matches!(
                            *m,
                            vk::SamplerAddressMode::CLAMP_TO_EDGE
                                | vk::SamplerAddressMode::CLAMP_TO_BORDER
                        )
                    }))
        {
            return Err(VulkanError::InvalidResource(
                "sampler LOD or unnormalized filtering/addressing state violates the native contract",
            ));
        }
        let mut custom_info = vk::SamplerCustomBorderColorCreateInfoEXT::default();
        if let Some(border) = extensions.custom_border {
            custom_info = custom_info
                .custom_border_color(vk::ClearColorValue {
                    uint32: border.components,
                })
                .format(border.format);
        }
        let mut reduction =
            vk::SamplerReductionModeCreateInfo::default().reduction_mode(extensions.reduction);
        let mut info = *info;
        if extensions.custom_border.is_some() {
            info = info.push_next(&mut custom_info);
        }
        if extensions.reduction != vk::SamplerReductionMode::WEIGHTED_AVERAGE {
            info = info.push_next(&mut reduction);
        }
        let handle = unsafe { self.owner.device.create_sampler(&info, None) }
            .map_err(|e| VulkanError::vk("cached native sampler", e))?;
        let sampler = Arc::new(Sampler {
            owner: Arc::clone(&self.owner),
            handle,
        });
        if self.maximum != 0 {
            if self.entries.len() >= self.maximum {
                let key = self
                    .entries
                    .iter()
                    .min_by_key(|(_, (_, tick))| *tick)
                    .map(|(key, _)| key.clone())
                    .unwrap();
                self.entries.remove(&key);
            }
            self.entries.insert(key, (Arc::clone(&sampler), self.clock));
        }
        Ok(sampler)
    }
    pub fn usage(&self) -> usize {
        self.entries.len()
    }
}
