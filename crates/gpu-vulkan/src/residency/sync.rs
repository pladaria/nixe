//! Ordered native hazards, independent of canonical guest-byte provenance.
//! https://docs.vulkan.org/guide/latest/synchronization_examples.html
use super::*;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AccessScope {
    pub stages: vk::PipelineStageFlags2,
    pub access: vk::AccessFlags2,
}
impl AccessScope {
    pub const fn new(stages: vk::PipelineStageFlags2, access: vk::AccessFlags2) -> Self {
        Self { stages, access }
    }
    pub(super) fn buffer_alignment(self, limits: &vk::PhysicalDeviceLimits) -> u64 {
        let uniform = self.access.contains(vk::AccessFlags2::UNIFORM_READ);
        let storage = self.access.intersects(
            vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
        );
        let mut alignment = 4;
        if uniform {
            alignment = alignment.max(limits.min_uniform_buffer_offset_alignment);
        }
        if storage || !uniform {
            alignment = alignment.max(limits.min_storage_buffer_offset_alignment);
        }
        alignment
    }
    pub fn writes(self) -> bool {
        self.access.intersects(
            vk::AccessFlags2::SHADER_WRITE
                | vk::AccessFlags2::SHADER_STORAGE_WRITE
                | vk::AccessFlags2::TRANSFER_WRITE
                | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
                | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE
                | vk::AccessFlags2::HOST_WRITE
                | vk::AccessFlags2::MEMORY_WRITE,
        )
    }
    fn union(&mut self, scope: Self) {
        self.stages |= scope.stages;
        self.access |= scope.access;
    }
}
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(super) struct AccessState {
    writer: AccessScope,
    readers: AccessScope,
    visible: Vec<AccessScope>,
}
#[derive(Clone, Copy, Debug)]
pub(super) struct Dependency {
    pub source: AccessScope,
    pub destination: AccessScope,
}
impl AccessState {
    pub(super) fn preceding(&self) -> AccessScope {
        let mut scope = self.writer;
        scope.stages |= self.readers.stages;
        scope
    }
    pub fn access(&mut self, next: AccessScope) -> Option<Dependency> {
        if next.writes() {
            let mut source = self.writer;
            // WAR needs execution ordering, not visibility of previous reads.
            source.stages |= self.readers.stages;
            *self = Self {
                writer: next,
                ..Self::default()
            };
            (!source.stages.is_empty()).then_some(Dependency {
                source,
                destination: next,
            })
        } else {
            self.readers.union(next);
            if self.writer.stages.is_empty()
                || self.visible.iter().any(|scope| {
                    scope.stages.contains(next.stages) && scope.access.contains(next.access)
                })
            {
                return None;
            }
            self.visible.push(next);
            Some(Dependency {
                source: self.writer,
                destination: next,
            })
        }
    }
}
#[derive(Clone, Debug)]
struct BufferSpan {
    start: u64,
    end: u64,
    state: AccessState,
}
pub(super) struct BufferAccess {
    spans: Vec<BufferSpan>,
}
impl BufferAccess {
    pub fn new(size: u64) -> Self {
        Self {
            spans: vec![BufferSpan {
                start: 0,
                end: size,
                state: AccessState::default(),
            }],
        }
    }
    pub fn access(
        &mut self,
        offset: u64,
        size: u64,
        next: AccessScope,
        mut emit: impl FnMut(u64, u64, Dependency),
    ) {
        let end = offset + size;
        let first = self.spans.partition_point(|span| span.end <= offset);
        let last = self.spans.partition_point(|span| span.start < end);
        // Repeated reads do not change interval boundaries or visibility.
        if !next.writes()
            && self.spans[first..last].iter().all(|span| {
                let state = &span.state;
                state.readers.stages.contains(next.stages)
                    && state.readers.access.contains(next.access)
                    && (state.writer.stages.is_empty()
                        || state.visible.iter().any(|scope| {
                            scope.stages.contains(next.stages) && scope.access.contains(next.access)
                        }))
            })
        {
            return;
        }
        for boundary in [offset, end] {
            let index = self.spans.partition_point(|span| span.end <= boundary);
            if let Some(span) = self.spans.get_mut(index)
                && span.start < boundary
                && boundary < span.end
            {
                let right = BufferSpan {
                    start: boundary,
                    end: span.end,
                    state: span.state.clone(),
                };
                span.end = boundary;
                self.spans.insert(index + 1, right);
            }
        }
        let first = self.spans.partition_point(|span| span.end <= offset);
        let last = self.spans.partition_point(|span| span.start < end);
        for span in &mut self.spans[first..last] {
            if let Some(dependency) = span.state.access(next) {
                emit(span.start, span.end - span.start, dependency);
            }
        }
        // Only the affected intervals and their neighbors can now coalesce.
        // Compact once; repeated Vec::remove would shift the tail quadratically.
        let mut write = first.saturating_sub(1);
        let stop = (last + 1).min(self.spans.len());
        for read in write + 1..stop {
            if self.spans[write].state == self.spans[read].state {
                self.spans[write].end = self.spans[read].end;
            } else {
                write += 1;
                if write != read {
                    self.spans.swap(write, read);
                }
            }
        }
        self.spans.drain(write + 1..stop);
    }
}
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum LayoutPolicy {
    #[default]
    Automatic,
    Conventional,
}
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum BarrierPolicy {
    #[default]
    ScopedMemory,
    PerResource,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct RecordingPolicy {
    pub layouts: LayoutPolicy,
    pub barriers: BarrierPolicy,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct BarrierStatistics {
    pub calls: u64,
    pub memory: u64,
    pub buffers: u64,
    pub images: u64,
}
#[derive(Clone, Debug)]
pub(super) struct ImageAccess {
    pub aspect: vk::ImageAspectFlags,
    pub mip: u32,
    pub layer: u32,
    pub layout: vk::ImageLayout,
    pub state: AccessState,
    pub ownership: super::ownership::Ownership,
}
pub(super) fn image_states(
    planes: &[ImagePlane],
    format: vk::Format,
    separate: bool,
) -> Vec<ImageAccess> {
    let combined = !separate
        && super::format::copy_block(format, vk::ImageAspectFlags::STENCIL).is_some()
        && super::format::copy_block(format, vk::ImageAspectFlags::DEPTH).is_some();
    let mut states: Vec<ImageAccess> = Vec::new();
    for plane in planes {
        let aspect = if combined {
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
        } else {
            plane.aspect
        };
        if !states.iter().any(|cell| {
            cell.aspect == aspect && cell.mip == plane.mip_level && cell.layer == plane.array_layer
        }) {
            states.push(ImageAccess {
                aspect,
                mip: plane.mip_level,
                layer: plane.array_layer,
                layout: vk::ImageLayout::UNDEFINED,
                state: AccessState::default(),
                ownership: super::ownership::Ownership::Local,
            });
        }
    }
    states
}
#[derive(Default)]
pub(super) struct Barriers {
    memory: Vec<vk::MemoryBarrier2<'static>>,
    buffers: Vec<vk::BufferMemoryBarrier2<'static>>,
    images: Vec<vk::ImageMemoryBarrier2<'static>>,
    pub statistics: BarrierStatistics,
}
impl Barriers {
    fn dependency(
        &mut self,
        policy: BarrierPolicy,
        buffer: vk::Buffer,
        offset: u64,
        size: u64,
        d: Dependency,
    ) {
        if policy == BarrierPolicy::ScopedMemory {
            self.memory_dependency(d);
        } else {
            self.buffers.push(
                vk::BufferMemoryBarrier2::default()
                    .src_stage_mask(d.source.stages)
                    .src_access_mask(d.source.access)
                    .dst_stage_mask(d.destination.stages)
                    .dst_access_mask(d.destination.access)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .buffer(buffer)
                    .offset(offset)
                    .size(size),
            );
        }
    }
    fn memory_dependency(&mut self, d: Dependency) {
        if let Some(last) = self.memory.last_mut()
            && last.src_stage_mask == d.source.stages
            && last.dst_stage_mask == d.destination.stages
        {
            last.src_access_mask |= d.source.access;
            last.dst_access_mask |= d.destination.access;
        } else {
            self.memory.push(
                vk::MemoryBarrier2::default()
                    .src_stage_mask(d.source.stages)
                    .src_access_mask(d.source.access)
                    .dst_stage_mask(d.destination.stages)
                    .dst_access_mask(d.destination.access),
            );
        }
    }
    pub fn queue_buffer(&mut self, barrier: vk::BufferMemoryBarrier2<'static>) {
        self.buffers.push(barrier);
    }
    pub fn queue_image(&mut self, barrier: vk::ImageMemoryBarrier2<'static>) {
        self.images.push(barrier);
    }
    pub(super) fn is_empty(&self) -> bool {
        self.memory.is_empty() && self.buffers.is_empty() && self.images.is_empty()
    }
    pub fn emit(&mut self, device: &mut VulkanDevice) {
        // Transfers, compute, and barriers are recorded outside dynamic rendering.
        // Graphics alone may retain the scope when no dependency is pending.
        device.end_rendering();
        if self.is_empty() {
            return;
        }
        let index = device
            .submission
            .recording
            .expect("barriers have a recording slot");
        unsafe {
            device.dispatch.barrier(
                device.submission.slots[index].command,
                &vk::DependencyInfo::default()
                    .memory_barriers(&self.memory)
                    .buffer_memory_barriers(&self.buffers)
                    .image_memory_barriers(&self.images),
            );
        }
        self.statistics.calls += 1;
        self.statistics.memory += self.memory.len() as u64;
        self.statistics.buffers += self.buffers.len() as u64;
        self.statistics.images += self.images.len() as u64;
        self.memory.clear();
        self.buffers.clear();
        self.images.clear();
    }
}
impl VulkanResidency {
    /// Guest WFI and ordered completion writes are device dependencies. Queue
    /// submission order alone would not establish availability/visibility.
    /// https://docs.vulkan.org/guide/latest/synchronization_examples.html
    pub(crate) fn order_device(&mut self) -> Result<(), VulkanError> {
        self.barriers.memory_dependency(Dependency {
            source: AccessScope::new(
                vk::PipelineStageFlags2::ALL_COMMANDS,
                vk::AccessFlags2::MEMORY_WRITE,
            ),
            destination: AccessScope::new(
                vk::PipelineStageFlags2::ALL_COMMANDS,
                vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
            ),
        });
        if self.device.submission.recording.is_some() {
            self.barriers.emit(&mut self.device);
        }
        Ok(())
    }
    pub fn with_recording_policy(
        memory: Machine,
        device: VulkanDevice,
        bytes: u64,
        policy: RecordingPolicy,
    ) -> Result<Self, VulkanError> {
        let mut cache = Self::new(memory, device, bytes)?;
        cache.unified_layouts = cache.device.profile().unified_image_layouts
            && policy.layouts == LayoutPolicy::Automatic;
        cache.barrier_policy = policy.barriers;
        Ok(cache)
    }
    pub fn barrier_statistics(&self) -> BarrierStatistics {
        self.barriers.statistics
    }
    pub(super) fn buffer_access_range(
        &mut self,
        id: BufferId,
        offset: u64,
        size: u64,
        scope: AccessScope,
    ) -> Result<vk::Buffer, VulkanError> {
        self.check_buffer_range(id, offset, size)?;
        // Logical completion effects retain an exact literal and producer
        // ticket. First physical use materializes the literal in queue order;
        // subsequent partial writers therefore preserve the remaining bytes.
        // Clear the flag before upload_buffer re-enters buffer_access_range.
        let pending: Vec<_> = self.buffers[id.slot]
            .entry
            .as_mut()
            .expect("buffer")
            .known
            .iter_mut()
            .filter_map(|known| {
                if !known.pending_upload
                    || known.offset >= offset + size
                    || known.offset + known.bytes.len() as u64 <= offset
                {
                    return None;
                }
                known.pending_upload = false;
                Some((known.offset, Arc::clone(&known.bytes)))
            })
            .collect();
        for (offset, bytes) in pending {
            self.upload_buffer(id, offset, &bytes)?;
        }
        self.device.begin_recording()?;
        let work = self.device.work_ticket()?;
        let entry = self.buffers[id.slot]
            .entry
            .as_mut()
            .expect("validated buffer");
        let handle = entry.native.handle;
        entry
            .access
            .access(offset, size, scope, |offset, size, dependency| {
                self.barriers
                    .dependency(self.barrier_policy, handle, offset, size, dependency)
            });
        entry.contents.last_use = Some(work);
        self.touch(Resource::Buffer(id));
        Ok(handle)
    }
    pub(super) fn image_access_subresources(
        &mut self,
        id: ImageId,
        range: vk::ImageSubresourceRange,
        scope: AccessScope,
        conventional: vk::ImageLayout,
    ) -> Result<(vk::Image, vk::ImageLayout), VulkanError> {
        self.image(id)?;
        if scope
            .access
            .intersects(vk::AccessFlags2::SHADER_SAMPLED_READ)
            && scope.access.intersects(
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
                    | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            )
        {
            return Err(VulkanError::InvalidResource(
                "attachment feedback needs an explicit supported feedback-loop layout or an exact snapshot lowering",
            ));
        }
        self.device.begin_recording()?;
        let work = self.device.work_ticket()?;
        let entry = self.images[id.slot].entry.as_mut().expect("image");
        if range.level_count == 0
            || range.layer_count == 0
            || range
                .base_mip_level
                .checked_add(range.level_count)
                .is_none_or(|end| end > entry.description.mip_levels)
            || range
                .base_array_layer
                .checked_add(range.layer_count)
                .is_none_or(|end| end > entry.description.array_layers)
        {
            return Err(VulkanError::InvalidResource(
                "image access exceeds subresources",
            ));
        }
        let layout = if self.unified_layouts && conventional != vk::ImageLayout::PRESENT_SRC_KHR {
            vk::ImageLayout::GENERAL
        } else {
            conventional
        };
        for cell in &mut entry.access {
            if !range.aspect_mask.intersects(cell.aspect)
                || cell.mip < range.base_mip_level
                || cell.mip >= range.base_mip_level + range.level_count
                || cell.layer < range.base_array_layer
                || cell.layer >= range.base_array_layer + range.layer_count
            {
                continue;
            }
            if cell.ownership != super::ownership::Ownership::Local {
                return Err(VulkanError::InvalidResource(
                    "image access requires acquiring its foreign queue ownership",
                ));
            }
            let preceding = cell.state.preceding();
            let dependency = cell.state.access(scope);
            if cell.layout == layout && self.barrier_policy == BarrierPolicy::ScopedMemory {
                if let Some(dependency) = dependency {
                    self.barriers.memory_dependency(dependency);
                }
            } else if cell.layout != layout || dependency.is_some() {
                let source = if cell.layout != layout {
                    preceding
                } else {
                    dependency.map_or_else(AccessScope::default, |d| d.source)
                };
                self.barriers.images.push(
                    vk::ImageMemoryBarrier2::default()
                        .old_layout(cell.layout)
                        .new_layout(layout)
                        .src_stage_mask(source.stages)
                        .src_access_mask(source.access)
                        .dst_stage_mask(scope.stages)
                        .dst_access_mask(scope.access)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .image(entry.native.handle)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(cell.aspect)
                                .base_mip_level(cell.mip)
                                .level_count(1)
                                .base_array_layer(cell.layer)
                                .layer_count(1),
                        ),
                );
            }
            cell.layout = layout;
        }
        entry.contents.last_use = Some(work);
        let handle = entry.native.handle;
        self.touch(Resource::Image(id));
        Ok((handle, layout))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn scope(stage: vk::PipelineStageFlags2, access: vk::AccessFlags2) -> AccessScope {
        AccessScope::new(stage, access)
    }
    #[test]
    fn split_and_coalesced_hazards_match_independent_byte_states() {
        let scopes = [
            scope(
                vk::PipelineStageFlags2::COPY,
                vk::AccessFlags2::TRANSFER_WRITE,
            ),
            scope(
                vk::PipelineStageFlags2::COPY,
                vk::AccessFlags2::TRANSFER_READ,
            ),
            scope(
                vk::PipelineStageFlags2::VERTEX_SHADER,
                vk::AccessFlags2::UNIFORM_READ,
            ),
            scope(
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
            ),
        ];
        let mut state = BufferAccess::new(64);
        let mut bytes = vec![AccessState::default(); 64];
        let mut random = 0x42bc_a981u64;
        for _ in 0..2000 {
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            let start = (random as usize >> 8) % 64;
            let end = (start + 1 + (random as usize >> 16) % 64).min(64);
            let next = scopes[random as usize % scopes.len()];
            let mut actual = vec![None; 64];
            state.access(
                start as u64,
                (end - start) as u64,
                next,
                |offset, size, dependency| {
                    actual[offset as usize..(offset + size) as usize]
                        .fill(Some((dependency.source, dependency.destination)));
                },
            );
            let mut expected = vec![None; 64];
            for i in start..end {
                expected[i] = bytes[i]
                    .access(next)
                    .map(|dependency| (dependency.source, dependency.destination));
            }
            assert_eq!(actual, expected);
        }
    }
    #[test]
    fn independent_ranges_skip_barriers_and_writer_visibility_survives_reads() {
        let mut state = BufferAccess::new(1024);
        let writer = scope(
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_STORAGE_WRITE,
        );
        state.access(0, 16, writer, |_, _, _| panic!("first producer"));
        state.access(64, 16, writer, |_, _, _| panic!("disjoint producer"));
        let copy = scope(
            vk::PipelineStageFlags2::COPY,
            vk::AccessFlags2::TRANSFER_READ,
        );
        let mut dependencies = Vec::new();
        state.access(0, 16, copy, |start, size, d| {
            dependencies.push((start, size, d))
        });
        assert_eq!(dependencies.len(), 1);
        assert_eq!(dependencies[0].0, 0);
        assert_eq!(dependencies[0].1, 16);
        state.access(0, 16, copy, |_, _, _| panic!("RAR already visible"));
        let fragment = scope(
            vk::PipelineStageFlags2::FRAGMENT_SHADER,
            vk::AccessFlags2::SHADER_STORAGE_READ,
        );
        state.access(0, 16, fragment, |_, _, d| assert_eq!(d.source, writer));
        state.access(0, 16, copy, |_, _, _| {
            panic!("earlier reader visibility remains valid")
        });
        state.access(64, 16, copy, |start, _, d| {
            assert_eq!(start, 64);
            assert_eq!(d.source, writer);
        });
    }
    #[test]
    fn war_orders_all_reader_stages_and_host_writes_require_visibility() {
        let mut state = AccessState::default();
        let copy = scope(
            vk::PipelineStageFlags2::COPY,
            vk::AccessFlags2::TRANSFER_READ,
        );
        let shader = scope(
            vk::PipelineStageFlags2::VERTEX_SHADER,
            vk::AccessFlags2::SHADER_STORAGE_READ,
        );
        assert!(state.access(copy).is_none());
        assert!(state.access(shader).is_none());
        let writer = scope(
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_STORAGE_WRITE,
        );
        let dependency = state.access(writer).unwrap();
        assert_eq!(dependency.source.stages, copy.stages | shader.stages);
        assert!(dependency.source.access.is_empty());
        let host = scope(vk::PipelineStageFlags2::HOST, vk::AccessFlags2::HOST_WRITE);
        let mut state = AccessState::default();
        assert!(state.access(host).is_none());
        assert_eq!(state.access(copy).unwrap().source, host);
    }
    #[test]
    fn each_access_kind_retains_its_own_visibility_pair() {
        let mut state = AccessState::default();
        let writer = scope(
            vk::PipelineStageFlags2::COPY,
            vk::AccessFlags2::TRANSFER_WRITE,
        );
        let _ = state.access(writer);
        let compute = scope(
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_STORAGE_READ,
        );
        let fragment = scope(
            vk::PipelineStageFlags2::FRAGMENT_SHADER,
            vk::AccessFlags2::UNIFORM_READ,
        );
        assert!(state.access(compute).is_some());
        assert!(state.access(fragment).is_some());
        assert!(
            state
                .access(scope(
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::UNIFORM_READ
                ))
                .is_some(),
            "union of unrelated stage/access pairs cannot fabricate visibility"
        );
    }
}
