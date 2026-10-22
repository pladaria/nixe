//! Physical buffer interpretations, immutable uploads and native alias reconstruction.
use super::*;
#[derive(Clone, Eq, Hash, PartialEq)]
pub(super) struct BufferKey {
    pub geometry: Vec<Arc<Footprint>>,
    pub bytes: u64,
}
#[derive(Clone, Copy)]
pub(super) struct BufferStreamRequest {
    pub id: BufferId,
    pub offset: u64,
    pub size: u64,
    pub alignment: u64,
}
pub(super) type BufferStream = Option<(Arc<AllocatedBuffer>, u64)>;
pub(super) struct StreamedBuffer {
    work: WorkTicket,
    window: (u64, u64),
    buffer: Arc<AllocatedBuffer>,
    offset: u64,
}
impl VulkanResidency {
    pub fn buffer_for(
        &mut self,
        backing: Arc<PhysicalMapping>,
        usage: vk::BufferUsageFlags,
    ) -> Result<BufferId, VulkanError> {
        self.create_buffer(backing, usage, true)
    }
    /// Prepare identity/allocation without reading guest bytes or recording a
    /// transfer. The ordered draw/dispatch consumer calls ensure_buffer once.
    pub fn buffer_for_deferred_import(
        &mut self,
        backing: Arc<PhysicalMapping>,
        usage: vk::BufferUsageFlags,
    ) -> Result<BufferId, VulkanError> {
        self.create_buffer(backing, usage, false)
    }
    fn create_buffer(
        &mut self,
        backing: Arc<PhysicalMapping>,
        usage: vk::BufferUsageFlags,
        initialize: bool,
    ) -> Result<BufferId, VulkanError> {
        let bytes = backing.bytes();
        let footprint = Footprint::new(
            backing,
            nixe_gpu::physical::WriteDomain::Buffer { start: 0, bytes },
        )?;
        self.buffer_for_geometry(vec![footprint], bytes, usage, initialize)
    }
    pub(super) fn buffer_for_geometry(
        &mut self,
        geometry: Vec<Arc<Footprint>>,
        bytes: u64,
        usage: vk::BufferUsageFlags,
        initialize: bool,
    ) -> Result<BufferId, VulkanError> {
        self.drain_notifications();
        if usage.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
            && !self.device.profile().buffer_device_address
        {
            return Err(VulkanError::InvalidResource(
                "buffer device address not enabled",
            ));
        }
        let usage = usage | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
        let key = BufferKey {
            geometry: geometry.clone(),
            bytes,
        };
        let candidate = self.buffer_views.get(&key).and_then(|views| {
            views.iter().find_map(|&id| {
                let entry = self.buffer(id).ok()?;
                entry.usage.contains(usage).then_some(id)
            })
        });
        if let Some(id) = candidate {
            if initialize {
                self.ensure_buffer(id)?;
            }
            self.touch(Resource::Buffer(id));
            self.statistics.cache_hits += 1;
            return Ok(id);
        }
        if geometry.is_empty()
            || geometry
                .iter()
                .any(|f| f.mapping().machine() != self.memory.identity())
        {
            return Err(VulkanError::InvalidResource(
                "buffer has no geometry or names another physical machine",
            ));
        }
        let native = self.allocate_buffer(bytes, usage)?;
        #[cfg(feature = "binding-probe")]
        let address = if usage.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS) {
            unsafe {
                self.device.owner.device.get_buffer_device_address(
                    &vk::BufferDeviceAddressInfo::default().buffer(native.handle),
                )
            }
        } else {
            0
        };
        let slot = self.free_buffers.pop().unwrap_or(self.buffers.len());
        if slot == self.buffers.len() {
            self.buffers.push(Slot {
                generation: 0,
                entry: None,
            });
        }
        let generation =
            self.buffers[slot]
                .generation
                .checked_add(1)
                .ok_or(VulkanError::InvalidResource(
                    "native buffer generation exhausted",
                ))?;
        let id = BufferId {
            instance: self.instance,
            slot,
            generation,
        };
        self.buffers[slot] = Slot {
            generation,
            entry: Some(BufferEntry {
                known: None,
                streamed: None,
                write_plans: Vec::new(),
                contents: Contents {
                    invalid: vec![(0, bytes)],
                    geometry: geometry.clone(),
                    cpu: Vec::new(),
                    device: Vec::new(),
                    last_use: None,
                    touched: 0,
                },
                native: Arc::new(native),
                usage,
                #[cfg(feature = "binding-probe")]
                address,
                access: sync::BufferAccess::new(bytes),
            }),
        };
        self.authority
            .register(Resource::Buffer(id), geometry.clone())
            .map_err(memory_error)?;
        let observations = geometry
            .into_iter()
            .map(|f| self.authority.observe_native(Resource::Buffer(id), f))
            .collect::<Result<Vec<_>, _>>()?;
        self.buffers[id.slot]
            .entry
            .as_mut()
            .expect("buffer")
            .contents
            .device = observations;
        self.buffer_views.entry(key).or_default().push(id);
        if initialize {
            self.ensure_buffer(id)?;
        }
        self.touch(Resource::Buffer(id));
        Ok(id)
    }
    /// Frequently rewritten CPU uniforms receive immutable upload slices. They
    /// stay dirty and do not install page observers or quiesce running cores.
    pub(super) fn stream_buffers(
        &mut self,
        requests: &[Option<BufferStreamRequest>],
    ) -> Result<Vec<BufferStream>, VulkanError> {
        let mut streams = vec![None; requests.len()];
        for (index, request) in requests.iter().enumerate() {
            let Some(request) = request else { continue };
            self.check_buffer_range(request.id, request.offset, request.size)?;
            if request.size > 64 * 1024
                || request.size > self.device.submission.configuration.arena_bytes
            {
                continue;
            }
            let entry = self.buffer(request.id)?;
            let [full] = entry.contents.geometry.as_slice() else {
                continue;
            };
            if !matches!(
                full.domain(),
                nixe_gpu::physical::WriteDomain::Buffer { .. }
            ) {
                continue;
            }
            let footprint = Footprint::new(
                Arc::clone(full.mapping()),
                nixe_gpu::physical::WriteDomain::Buffer {
                    start: request.offset,
                    bytes: request.size,
                },
            )?;
            if self
                .authority
                .resolve_footprint(&footprint)?
                .iter()
                .any(|p| p.producer.is_some())
            {
                continue;
            }
            let work = self.device.work_ticket()?;
            let entry = self.buffer(request.id)?;
            if entry.contents.current()
                && !entry.contents.cpu.is_empty()
                && let Some(stream) = &entry.streamed
                && stream.work == work
                && stream.window == (request.offset, request.size)
                && stream.offset.is_multiple_of(request.alignment)
            {
                streams[index] = Some((Arc::clone(&stream.buffer), stream.offset));
                self.statistics.cpu_stream_hits += 1;
                let slot = self.device.submission.recording.expect("stream recording");
                self.device.submission.retain_stream_upload(slot);
                continue;
            }
            let inputs = footprint.mapping().pin(request.offset, request.size)?;
            let mut bytes = vec![0; request.size as usize];
            self.memory.capture_stream(&inputs, &mut bytes)?;
            let (slot, slice) = self.reserve_upload(&bytes, request.alignment.max(4))?;
            self.device.submission.retain_stream_upload(slot);
            let native = self.device.submission.slots[slot].upload_lease(slice);
            let work = self.device.work_ticket()?;
            let entry = self.buffers[request.id.slot]
                .entry
                .as_mut()
                .expect("buffer");
            entry.streamed = Some(StreamedBuffer {
                work,
                window: (request.offset, request.size),
                buffer: Arc::clone(&native),
                offset: slice.offset,
            });
            entry.contents.last_use = Some(work);
            streams[index] = Some((native, slice.offset));
            self.statistics.cpu_streamed_bytes += request.size;
        }
        Ok(streams)
    }
    pub fn ensure_buffer(&mut self, id: BufferId) -> Result<(), VulkanError> {
        if self.buffer(id)?.contents.current() {
            return Ok(());
        }
        self.constructing.push(Resource::Buffer(id));
        let result = self.refresh_buffer(id, true);
        self.unpin(Resource::Buffer(id));
        result
    }
    pub(super) fn refresh_buffer(
        &mut self,
        id: BufferId,
        upload_cpu: bool,
    ) -> Result<(), VulkanError> {
        let _trace = nixe_gpu::trace::span(
            "buffer_refresh",
            id.slot as u64,
            self.buffer(id)?.native.size(),
            0,
        );
        let geometry = self.buffer(id)?.contents.geometry.clone();
        let mut parts = Vec::new();
        for footprint in &geometry {
            parts.extend(self.authority.resolve_footprint(footprint)?);
        }
        parts.sort_unstable_by_key(|part| part.logical);
        let device = geometry
            .iter()
            .map(|f| {
                self.authority
                    .observe_native(Resource::Buffer(id), Arc::clone(f))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut copies = Vec::new();
        let mut windows: Vec<(u64, usize, usize)> = Vec::new();
        let mut packed = 0;
        let unavailable: Vec<_> = if upload_cpu {
            Vec::new()
        } else {
            parts
                .iter()
                .filter(|part| part.producer.is_none())
                .map(|part| (part.logical, part.bytes))
                .collect()
        };
        let invalid = self.buffer(id)?.contents.invalid.clone();
        for part in parts.iter().filter(|p| upload_cpu && p.producer.is_none()) {
            for &(offset, bytes) in &invalid {
                let begin = part.logical.max(offset);
                let end = (part.logical + part.bytes).min(offset + bytes);
                if begin >= end {
                    continue;
                }
                copies.push(nixe_memory::PhysicalCapture {
                    source: nixe_memory::Extent {
                        start: nixe_memory::ArenaOffset::new(
                            part.start.get() + begin - part.logical,
                        ),
                        bytes: end - begin,
                    },
                    destination: packed,
                });
                if let Some((offset, _, bytes)) = windows.last_mut()
                    && *offset + *bytes as u64 == begin
                {
                    *bytes += (end - begin) as usize;
                } else {
                    windows.push((begin, packed, (end - begin) as usize));
                }
                packed += (end - begin) as usize;
            }
        }
        let mut cpu = Vec::new();
        let mut captured = vec![0; packed];
        if !copies.is_empty() {
            let mut pages = Vec::new();
            for footprint in &geometry {
                footprint.mapping().observed_pages(footprint, &mut pages)?;
            }
            let inputs = nixe_gpu::physical::pin_selected_pages(pages)?;
            let observation =
                self.memory
                    .capture_clean_scattered(&inputs, &copies, &mut captured)?;
            cpu.push(observation);
        }
        let self_alias = parts.iter().any(|part| {
            part.producer
                .is_some_and(|p| p.resource == Resource::Buffer(id) && p.native != part.logical)
        });
        let preserved = if self_alias {
            let bytes = self.buffer(id)?.native.size();
            let scratch = self.temporary_buffer(bytes, &[Resource::Buffer(id)])?;
            let source = self.buffer_access_range(
                id,
                0,
                bytes,
                AccessScope::new(
                    vk::PipelineStageFlags2::COPY,
                    vk::AccessFlags2::TRANSFER_READ,
                ),
            )?;
            let slot = self.device.begin_recording()?;
            self.barriers.emit(&mut self.device);
            unsafe {
                let command = self.device.submission.slots[slot].command;
                self.device.dispatch.cmd_copy_buffer(
                    command,
                    source,
                    scratch.handle,
                    &[vk::BufferCopy::default().size(bytes)],
                );
                super::recording::scratch_read_barrier(
                    &self.device,
                    command,
                    scratch.handle,
                    bytes,
                );
            }
            Some(scratch)
        } else {
            None
        };
        for (offset, start, bytes) in windows {
            self.upload_buffer(id, offset, &captured[start..start + bytes])?;
        }
        let mut images = rustc_hash::FxHashMap::<ImageId, Vec<(u64, u64, u64)>>::default();
        let mut buffers: Vec<(Resource, u64, u64, u64)> = Vec::new();
        for part in parts {
            let Some(source) = part.producer else {
                continue;
            };
            if source.resource == Resource::Buffer(id) && source.native == part.logical {
                continue;
            }
            if let Resource::Image(image) = source.resource {
                let copies = images.entry(image).or_default();
                if let Some((from, to, bytes)) = copies.last_mut()
                    && *from + *bytes == source.native
                    && *to + *bytes == part.logical
                {
                    *bytes += part.bytes;
                } else {
                    copies.push((source.native, part.logical, part.bytes));
                }
            } else if let Some((resource, from, to, bytes)) = buffers.last_mut()
                && *resource == source.resource
                && *from + *bytes == source.native
                && *to + *bytes == part.logical
            {
                *bytes += part.bytes;
            } else {
                buffers.push((source.resource, source.native, part.logical, part.bytes));
            }
        }
        for (source, from, to, bytes) in buffers {
            if source == Resource::Buffer(id) {
                let scratch = preserved.as_ref().expect("preserved native self alias");
                let target = self.buffer_access_range(
                    id,
                    to,
                    bytes,
                    AccessScope::new(
                        vk::PipelineStageFlags2::COPY,
                        vk::AccessFlags2::TRANSFER_WRITE,
                    ),
                )?;
                let slot = self.device.begin_recording()?;
                self.barriers.emit(&mut self.device);
                unsafe {
                    self.device.dispatch.cmd_copy_buffer(
                        self.device.submission.slots[slot].command,
                        scratch.handle,
                        target,
                        &[vk::BufferCopy::default()
                            .src_offset(from)
                            .dst_offset(to)
                            .size(bytes)],
                    );
                }
            } else {
                self.copy_source(source, from, id, to, bytes)?;
            }
        }
        for (image, copies) in images {
            self.copy_image_buffer_regions(image, id, &copies)?;
        }
        if let Some(scratch) = preserved {
            let work = self.device.work_ticket()?;
            self.device.retire_buffer(scratch, work)?;
        }
        let entry = self.buffers[id.slot].entry.as_mut().expect("buffer");
        entry.contents.cpu = cpu;
        entry.contents.device = device;
        entry.contents.invalid = unavailable;
        Ok(())
    }
    pub fn buffer_lease(&self, id: BufferId) -> Result<Arc<AllocatedBuffer>, VulkanError> {
        Ok(Arc::clone(&self.buffer(id)?.native))
    }
}
