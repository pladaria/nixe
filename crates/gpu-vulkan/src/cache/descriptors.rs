//! Owned descriptor captures; native allocation/update runs on the recording worker.
//! Native set history contains handles and completion ticks, never resource payloads.
use super::*;
use crate::{AllocatedBuffer, AllocatedImageView, VulkanDevice, VulkanError, device::DeviceOwner};
use ash::{vk, vk::Handle};
use rustc_hash::FxHashMap as HashMap;
use std::sync::{
    Arc, Weak,
    atomic::{AtomicU64, Ordering},
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
// A cached descriptor identity keeps weak object identities, not CPU reservations.
// The weak control blocks prevent handle/address reuse from matching dead objects.
struct CachedResource {
    key: ResourceKey,
    references: ResourceReferences,
}
enum ResourceReferences {
    Null,
    Buffer(Weak<AllocatedBuffer>),
    Image(Weak<AllocatedImageView>),
    Combined(Weak<AllocatedImageView>, Weak<Sampler>),
    Sampler(Weak<Sampler>),
}
impl CachedResource {
    fn new(resource: &DescriptorResource) -> Self {
        let references = match resource {
            DescriptorResource::NullBuffer => ResourceReferences::Null,
            DescriptorResource::Buffer { buffer, .. }
            | DescriptorResource::TexelBuffer { buffer, .. } => {
                ResourceReferences::Buffer(Arc::downgrade(buffer))
            }
            DescriptorResource::Image { view, .. } => {
                ResourceReferences::Image(Arc::downgrade(view))
            }
            DescriptorResource::CombinedImageSampler { view, sampler, .. } => {
                ResourceReferences::Combined(Arc::downgrade(view), Arc::downgrade(sampler))
            }
            DescriptorResource::Sampler(sampler) => {
                ResourceReferences::Sampler(Arc::downgrade(sampler))
            }
        };
        Self {
            key: resource.key(),
            references,
        }
    }
    fn matches(&self, resource: &DescriptorResource) -> bool {
        self.key == resource.key()
            && match (&self.references, resource) {
                (ResourceReferences::Null, DescriptorResource::NullBuffer) => true,
                (
                    ResourceReferences::Buffer(weak),
                    DescriptorResource::Buffer { buffer, .. }
                    | DescriptorResource::TexelBuffer { buffer, .. },
                ) => weak.as_ptr() == Arc::as_ptr(buffer),
                (ResourceReferences::Image(weak), DescriptorResource::Image { view, .. }) => {
                    weak.as_ptr() == Arc::as_ptr(view)
                }
                (
                    ResourceReferences::Combined(weak_view, weak_sampler),
                    DescriptorResource::CombinedImageSampler { view, sampler, .. },
                ) => {
                    weak_view.as_ptr() == Arc::as_ptr(view)
                        && weak_sampler.as_ptr() == Arc::as_ptr(sampler)
                }
                (ResourceReferences::Sampler(weak), DescriptorResource::Sampler(sampler)) => {
                    weak.as_ptr() == Arc::as_ptr(sampler)
                }
                _ => false,
            }
    }
}
struct LastSet {
    resources: Vec<CachedResource>,
    index: usize,
}
// Native sets grow in layout banks and recycle only after their real GPU use.
// https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/vk_descriptor_pool.cpp
const SETS_GROW: usize = 16;
pub(crate) struct DescriptorPool {
    owner: Arc<DeviceOwner>,
    handle: vk::DescriptorPool,
}
impl Drop for DescriptorPool {
    fn drop(&mut self) {
        unsafe { self.owner.device.destroy_descriptor_pool(self.handle, None) };
    }
}
struct PageSlot {
    page: Arc<DescriptorPool>,
    allocated: usize,
}
struct LayoutSets {
    layout: Weak<PipelineLayout>,
    bank: usize,
    sets: Vec<NativeSet>,
    next: usize,
    last: Option<LastSet>,
}
struct Bank {
    sizes: Vec<vk::DescriptorPoolSize>,
    score: u32,
    pages: Vec<PageSlot>,
}
impl Bank {
    fn supports(&self, requested: &[vk::DescriptorPoolSize], score: u32) -> bool {
        self.score.abs_diff(score) < 3
            && requested.iter().all(|size| {
                self.sizes.iter().any(|available| {
                    available.ty == size.ty && available.descriptor_count >= size.descriptor_count
                })
            })
    }
}
pub(crate) struct NativeSet {
    pub handle: vk::DescriptorSet,
    pub pool: Arc<DescriptorPool>,
    tick: u64,
    counters: nixe_gpu::cache_counters::CacheCounters,
}
impl Drop for NativeSet {
    fn drop(&mut self) {
        self.counters.remove_entries(1);
    }
}
/// Compact native data consumed by the layout's cold update template.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) union DescriptorEntry {
    buffer: vk::DescriptorBufferInfo,
    image: vk::DescriptorImageInfo,
    texel: vk::BufferView,
}
pub struct BindingRecord {
    pub layout: Arc<PipelineLayout>,
    pub(crate) resources: Vec<DescriptorResource>,
    entries: Vec<DescriptorEntry>,
    metrics: Arc<NativeMetrics>,
    counters: nixe_gpu::cache_counters::CacheCounters,
    sets_per_pool: usize,
}
#[derive(Default)]
struct NativeMetrics {
    allocated: AtomicU64,
    updates: AtomicU64,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct DescriptorStatistics {
    pub hits: u64,
    pub allocated: u64,
    pub updates: u64,
}
pub struct DescriptorCache {
    owner: Arc<DeviceOwner>,
    null_descriptor: bool,
    limits: vk::PhysicalDeviceLimits,
    sets_per_pool: usize,
    last: HashMap<vk::DescriptorSetLayout, Weak<BindingRecord>>,
    formats: HashMap<vk::Format, vk::FormatFeatureFlags>,
    hits: u64,
    metrics: Arc<NativeMetrics>,
    counters: nixe_gpu::cache_counters::CacheCounters,
}
/// Mutated exclusively by the Vulkan recording worker.
#[derive(Default)]
pub(crate) struct DescriptorAllocator {
    layouts: HashMap<vk::DescriptorSetLayout, LayoutSets>,
    banks: Vec<Bank>,
}
impl DescriptorCache {
    pub fn new(device: &VulkanDevice) -> Self {
        Self {
            owner: Arc::clone(&device.owner),
            null_descriptor: device.profile().null_descriptor,
            limits: device.profile().limits,
            // Match the reference pool sizes, including AMD's larger banks.
            sets_per_pool: if device.profile().vendor_id == 0x1002 {
                96
            } else {
                64
            },
            last: HashMap::default(),
            formats: HashMap::default(),
            hits: 0,
            metrics: Arc::new(NativeMetrics::default()),
            counters: nixe_gpu::cache_counters::CacheCounters::new(0, 0),
        }
    }
    pub(crate) fn hits(&self) -> u64 {
        self.hits
    }
    pub fn statistics(&self) -> DescriptorStatistics {
        DescriptorStatistics {
            hits: self.hits,
            allocated: self.metrics.allocated.load(Ordering::Relaxed),
            updates: self.metrics.updates.load(Ordering::Relaxed),
        }
    }
    pub(crate) fn cache_counters(&self) -> nixe_gpu::cache_counters::CacheCounters {
        self.counters.clone()
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
        if let Some(record) = self.last.get(&layout.set).and_then(Weak::upgrade)
            && Arc::ptr_eq(&record.layout, layout)
            && record.resources.len() == resources.len()
            && record
                .resources
                .iter()
                .map(DescriptorResource::key)
                .eq(resources.iter().map(DescriptorResource::key))
        {
            self.hits += 1;
            return Ok(record);
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
                    let features = *self.formats.entry(*format).or_insert_with(|| unsafe {
                        self.owner
                            .instance
                            .instance
                            .get_physical_device_format_properties(self.owner.physical, *format)
                            .buffer_features
                    });
                    if !buffer.usage.contains(usage)
                        || !features.contains(feature)
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
        self.counters.miss();
        let mut entries = Vec::with_capacity(resources.len());
        for resource in resources {
            let mut entry = DescriptorEntry {
                buffer: vk::DescriptorBufferInfo::default(),
            };
            match resource {
                DescriptorResource::NullBuffer => {
                    entry.buffer = vk::DescriptorBufferInfo::default().range(vk::WHOLE_SIZE);
                }
                DescriptorResource::Buffer {
                    buffer,
                    offset,
                    range,
                } => {
                    entry.buffer = vk::DescriptorBufferInfo::default()
                        .buffer(buffer.handle)
                        .offset(*offset)
                        .range(*range);
                }
                DescriptorResource::Image { view, layout } => {
                    entry.image = vk::DescriptorImageInfo::default()
                        .image_view(view.handle)
                        .image_layout(*layout);
                }
                DescriptorResource::CombinedImageSampler {
                    view,
                    layout,
                    sampler,
                } => {
                    entry.image = vk::DescriptorImageInfo::default()
                        .image_view(view.handle)
                        .image_layout(*layout)
                        .sampler(sampler.handle);
                }
                DescriptorResource::Sampler(sampler) => {
                    entry.image = vk::DescriptorImageInfo::default().sampler(sampler.handle);
                }
                DescriptorResource::TexelBuffer {
                    buffer,
                    offset,
                    range,
                    format,
                    ..
                } => {
                    entry.texel = buffer.texel_view(*offset, *range, *format)?;
                }
            }
            entries.push(entry);
        }
        let record = Arc::new(BindingRecord {
            layout: Arc::clone(layout),
            resources: resources.to_vec(),
            entries,
            metrics: Arc::clone(&self.metrics),
            counters: self.counters.clone(),
            sets_per_pool: self.sets_per_pool,
        });
        // Prune dead weak keys on insertion rather than retaining obsolete layouts.
        if !self.last.contains_key(&layout.set) {
            self.last.retain(|_, last| last.strong_count() != 0);
        }
        self.last.insert(layout.set, Arc::downgrade(&record));
        Ok(record)
    }
}
impl DescriptorAllocator {
    pub(crate) fn resolve(
        &mut self,
        record: &BindingRecord,
        tick: u64,
        progress: &crate::queue::progress::SubmissionProgress,
        timeline: vk::Semaphore,
    ) -> Result<&NativeSet, crate::queue::progress::NativeFailure> {
        let layout = record.layout.set;
        if let Some(sets) = self.layouts.get(&layout)
            && sets.layout.as_ptr() == Arc::as_ptr(&record.layout)
            && let Some(last) = &sets.last
            && last.resources.len() == record.resources.len()
            && last
                .resources
                .iter()
                .zip(&record.resources)
                .all(|(cached, resource)| cached.matches(resource))
        {
            let index = last.index;
            let native = &mut self.layouts.get_mut(&layout).unwrap().sets[index];
            native.tick = tick;
            return Ok(native);
        }
        let index = self.commit(record, progress, timeline)?;
        let sets = self.layouts.get_mut(&layout).unwrap();
        let native = &mut sets.sets[index];
        // CPU captures own descriptor data, not physical sets. The worker
        // recycles by actual tick and updates again when an old capture returns.
        // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/vk_graphics_pipeline.cpp
        unsafe {
            record
                .layout
                .owner
                .device
                .update_descriptor_set_with_template(
                    native.handle,
                    record.layout.update_template,
                    record.entries.as_ptr().cast(),
                );
        }
        native.tick = tick;
        record.metrics.updates.fetch_add(1, Ordering::Relaxed);
        let last = sets.last.get_or_insert_with(|| LastSet {
            resources: Vec::new(),
            index,
        });
        last.index = index;
        last.resources.clear();
        last.resources
            .extend(record.resources.iter().map(CachedResource::new));
        Ok(&sets.sets[index])
    }
    fn commit(
        &mut self,
        record: &BindingRecord,
        progress: &crate::queue::progress::SubmissionProgress,
        timeline: vk::Semaphore,
    ) -> Result<usize, crate::queue::progress::NativeFailure> {
        let layout = &record.layout;
        let owner = &layout.owner;
        let sets_per_pool = record.sets_per_pool;
        if self
            .layouts
            .get(&layout.set)
            .is_none_or(|sets| sets.layout.strong_count() == 0)
        {
            self.layouts
                .retain(|_, sets| sets.layout.strong_count() != 0);
            for bank in &mut self.banks {
                bank.pages.retain(|page| Arc::strong_count(&page.page) > 1);
            }
        }
        if !self.layouts.contains_key(&layout.set) {
            let mut counts = HashMap::<vk::DescriptorType, u32>::default();
            for binding in &layout.key.bindings {
                *counts.entry(binding.kind).or_default() += 1;
            }
            let sizes: Vec<_> = counts
                .into_iter()
                .map(|(ty, descriptor_count)| vk::DescriptorPoolSize {
                    ty,
                    descriptor_count,
                })
                .collect();
            let score = sizes.iter().map(|size| size.descriptor_count).sum();
            let bank = self
                .banks
                .iter()
                .position(|bank| bank.supports(&sizes, score))
                .unwrap_or_else(|| {
                    self.banks.push(Bank {
                        sizes,
                        score,
                        pages: Vec::new(),
                    });
                    self.banks.len() - 1
                });
            self.layouts.insert(
                layout.set,
                LayoutSets {
                    bank,
                    layout: Arc::downgrade(layout),
                    sets: Vec::new(),
                    next: 0,
                    last: None,
                },
            );
        }
        let allocator = self.layouts.get_mut(&layout.set).unwrap();
        let find = |sets: &LayoutSets, completed: u64| {
            (sets.next..sets.sets.len())
                .chain(0..sets.next)
                .find(|&i| sets.sets[i].tick <= completed)
        };
        let completed = progress.completed();
        let mut available = find(allocator, completed);
        if available.is_none() && completed < progress.submitted() {
            // Refresh only on pool overflow, then search again before growing.
            // The worker can recycle actual completed sets before owner retirement.
            // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/renderer_vulkan/vk_resource_pool.cpp
            let actual = unsafe { owner.device.get_semaphore_counter_value(timeline) }.map_err(
                |result| crate::queue::progress::NativeFailure::Vulkan {
                    operation: "descriptor pool timeline refresh",
                    result,
                },
            )?;
            progress.observe_completed(actual.min(progress.submitted()));
            available = find(allocator, progress.completed());
        }
        if let Some(index) = available {
            allocator.next = (index + 1) % allocator.sets.len();
            return Ok(index);
        }
        let bank = &mut self.banks[allocator.bank];
        if bank
            .pages
            .last()
            .is_none_or(|page| page.allocated + SETS_GROW > sets_per_pool)
        {
            Self::grow_pool(owner, bank, sets_per_pool)?;
        }
        let layouts = [layout.set; SETS_GROW];
        let allocate = |page: &DescriptorPool| unsafe {
            owner.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(page.handle)
                    .set_layouts(&layouts),
            )
        };
        let sets = match allocate(&bank.pages.last().unwrap().page) {
            Ok(sets) => sets,
            Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY | vk::Result::ERROR_FRAGMENTED_POOL) => {
                Self::grow_pool(owner, bank, sets_per_pool)?;
                allocate(&bank.pages.last().unwrap().page).map_err(|e| {
                    crate::queue::progress::NativeFailure::Vulkan {
                        operation: "vkAllocateDescriptorSets",
                        result: e,
                    }
                })?
            }
            Err(e) => {
                return Err(crate::queue::progress::NativeFailure::Vulkan {
                    operation: "vkAllocateDescriptorSets",
                    result: e,
                });
            }
        };
        let first = allocator.sets.len();
        let page = bank.pages.last_mut().unwrap();
        page.allocated += sets.len();
        for set in sets {
            allocator.sets.push(NativeSet {
                handle: set,
                pool: Arc::clone(&page.page),
                tick: 0,
                counters: record.counters.clone(),
            });
        }
        allocator.next = first + 1;
        record
            .metrics
            .allocated
            .fetch_add(SETS_GROW as u64, Ordering::Relaxed);
        record.counters.add_entries(SETS_GROW);
        Ok(first)
    }
    fn grow_pool(
        owner: &Arc<DeviceOwner>,
        bank: &mut Bank,
        count: usize,
    ) -> Result<(), crate::queue::progress::NativeFailure> {
        let sizes: Vec<_> = bank
            .sizes
            .iter()
            .map(|size| {
                vk::DescriptorPoolSize::default()
                    .ty(size.ty)
                    .descriptor_count(size.descriptor_count * count as u32)
            })
            .collect();
        let handle = unsafe {
            owner.device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(count as u32)
                    .pool_sizes(&sizes),
                None,
            )
        }
        .map_err(|e| crate::queue::progress::NativeFailure::Vulkan {
            operation: "vkCreateDescriptorPool",
            result: e,
        })?;
        bank.pages.push(PageSlot {
            page: Arc::new(DescriptorPool {
                owner: Arc::clone(owner),
                handle,
            }),
            allocated: 0,
        });
        Ok(())
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
            &mut slot.bound_compute_binding
        } else {
            &mut slot.bound_graphics_binding
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
        let identity = Arc::as_ptr(record) as usize;
        if *bound == identity
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
            self.dispatch.bind_descriptors(
                pipeline.bind_point,
                Arc::clone(record),
                dynamic_offsets,
            );
        }
        self.submission.statistics.descriptor_binds += 1;
        bound_offsets.clear();
        bound_offsets.extend_from_slice(dynamic_offsets);
        *bound = identity;
        *bound_layout = pipeline.layout.handle;
        if slot.binding_handles.insert(identity) {
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sampler_changes_reuse_texel_views_owned_by_the_buffer() {
        let Some(mut device) = crate::test_device::device_for_test(crate::ApiPreference::Auto)
        else {
            return;
        };
        let monitor = device.validation_monitor();
        let layouts = crate::cache::layout::LayoutCache::new(&device, 8);
        let layout = layouts
            .get(LayoutKey {
                bindings: vec![
                    LayoutBinding {
                        binding: 0,
                        element: 0,
                        kind: vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
                        stages: vk::ShaderStageFlags::COMPUTE,
                    },
                    LayoutBinding {
                        binding: 1,
                        element: 0,
                        kind: vk::DescriptorType::SAMPLER,
                        stages: vk::ShaderStageFlags::COMPUTE,
                    },
                ]
                .into(),
                push_bytes: 0,
                push_stages: vk::ShaderStageFlags::empty(),
            })
            .unwrap();
        let buffer = Arc::new(
            device
                .allocate_buffer(
                    16,
                    vk::BufferUsageFlags::UNIFORM_TEXEL_BUFFER,
                    crate::MemoryLocation::CpuToGpu,
                    crate::AllocationMode::Managed,
                    "cached texel view",
                )
                .unwrap(),
        );
        let first_sampler = Sampler::new(&device, &vk::SamplerCreateInfo::default()).unwrap();
        let second_sampler = Sampler::new(
            &device,
            &vk::SamplerCreateInfo::default().mag_filter(vk::Filter::LINEAR),
        )
        .unwrap();
        let resources = |sampler: Arc<Sampler>| {
            [
                DescriptorResource::TexelBuffer {
                    buffer: Arc::clone(&buffer),
                    offset: 0,
                    range: 16,
                    format: vk::Format::R32_UINT,
                    element_bytes: 4,
                },
                DescriptorResource::Sampler(sampler),
            ]
        };
        let mut cache = DescriptorCache::new(&device);
        let a = cache.get(&layout, &resources(first_sampler)).unwrap();
        let b = cache.get(&layout, &resources(second_sampler)).unwrap();
        // Both prepared records use the exact same buffer-owned native view.
        assert_eq!(unsafe { a.entries[0].texel }, unsafe { b.entries[0].texel });
        assert_eq!(
            cache.statistics().updates,
            0,
            "preparation does not update native descriptors"
        );
        assert_eq!(
            cache.formats.len(),
            1,
            "immutable capabilities are cached per format"
        );
        device.begin_recording().unwrap();
        unsafe {
            device
                .dispatch
                .bind_descriptors(vk::PipelineBindPoint::COMPUTE, Arc::clone(&a), &[]);
            device
                .dispatch
                .bind_descriptors(vk::PipelineBindPoint::COMPUTE, Arc::clone(&b), &[]);
        }
        let work = device.flush().unwrap().unwrap();
        assert!(
            device
                .wait_for_work(work, std::time::Duration::from_secs(10))
                .unwrap()
        );
        assert_eq!(
            cache.statistics().updates,
            2,
            "the worker updates both changed sampler payloads"
        );
        drop(a);
        // Destruction of one descriptor payload must not destroy the cached view.
        let retained = unsafe { b.entries[0].texel };
        assert_eq!(
            buffer.texel_view(0, 16, vk::Format::R32_UINT).unwrap(),
            retained
        );
        drop(b);
        drop(cache);
        drop(buffer);
        drop(layout);
        drop(layouts);
        drop(device);
        monitor.check().unwrap();
    }
}
