//! Native image representations, compatible views and GPU-only byte realignment.
use super::recording::scratch_read_barrier;
use super::*;
mod aliases;
impl VulkanResidency {
    // Image interpretations belong to their resident entry and retire with it.
    // https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/texture_cache/texture_cache.h
    pub(crate) fn guest_image(
        &mut self,
        guest: &nixe_gpu::native::GuestImage,
        view_format: Option<vk::Format>,
    ) -> Result<Option<(ImageId, [u32; 2])>, VulkanError> {
        let Some(plane) = guest.planes.first() else {
            return Ok(None);
        };
        let footprint = &plane.footprint;
        let start = footprint.bounds().start;
        let spans = footprint.mapping().spans();
        let first = &spans[spans.partition_point(|span| span.logical + span.bytes <= start)];
        let mut found = None;
        let mut failure = None;
        self.authority.visit_resource_overlaps(
            nixe_memory::Extent {
                start: nixe_memory::ArenaOffset::new(first.start.get() + start - first.logical),
                bytes: 1,
            },
            |resource, _, _| {
                let Resource::Image(id) = resource else {
                    return;
                };
                let Ok(entry) = self.image(id) else { return };
                if !entry.binding_valid {
                    return;
                }
                let Some(source) = &entry.guest else { return };
                if source.same_storage(guest) {
                    found = Some((id, [0, 0]));
                    return;
                }
                if found.is_some() {
                    return;
                }
                let Some(format) = view_format else { return };
                let a = source.description;
                let b = guest.description;
                if a.dimension() != b.dimension()
                    || a.kind() != b.kind()
                    || a.samples() != b.samples()
                    || source.sample_grid != guest.sample_grid
                    || source.canonical_texels != guest.canonical_texels
                    || (a.format() != b.format()
                        && (entry.decoded.is_some()
                            || !entry
                                .description
                                .flags
                                .contains(vk::ImageCreateFlags::MUTABLE_FORMAT)
                            || !format::view_compatible(entry.description.format, format)))
                {
                    return;
                }
                let Some(first) = source.planes.iter().find(|p| same_plane_storage(p, plane))
                else {
                    return;
                };
                let Some(base_mip) = first.mip_level.checked_sub(plane.mip_level) else {
                    return;
                };
                let Some(base_layer) = first.array_layer.checked_sub(plane.array_layer) else {
                    return;
                };
                let end_mip = base_mip + u32::from(b.mip_levels());
                let end_layer = base_layer + u32::from(b.array_layers());
                if a.mip_extent(base_mip as u8) != Some(b.extent())
                    || end_mip > u32::from(a.mip_levels())
                    || end_layer > u32::from(a.array_layers())
                {
                    return;
                }
                let mut stored = source.planes.iter().filter(|p| {
                    (base_mip..end_mip).contains(&p.mip_level)
                        && (base_layer..end_layer).contains(&p.array_layer)
                });
                if !guest.planes.iter().all(|p| {
                    stored.next().is_some_and(|s| {
                        s.mip_level == p.mip_level + base_mip
                            && s.array_layer == p.array_layer + base_layer
                            && same_plane_storage(s, p)
                    })
                }) || stored.next().is_some()
                {
                    return;
                }
                if a.format() != b.format() && !entry.views.iter().any(|view| view.format == format)
                {
                    match unsafe {
                        self.device
                            .owner
                            .instance
                            .instance
                            .get_physical_device_image_format_properties(
                                self.device.owner.physical,
                                format,
                                entry.description.image_type,
                                vk::ImageTiling::OPTIMAL,
                                entry.description.usage,
                                entry.description.flags,
                            )
                    } {
                        Ok(_) => (),
                        Err(vk::Result::ERROR_FORMAT_NOT_SUPPORTED) => return,
                        Err(error) => {
                            failure = Some(VulkanError::vk("compatible image view format", error));
                            return;
                        }
                    }
                }
                found = Some((id, [base_mip, base_layer]));
            },
        );
        if let Some(error) = failure {
            return Err(error);
        }
        if let Some((id, _)) = found {
            self.touch(Resource::Image(id));
            self.statistics.cache_hits += 1;
        }
        Ok(found)
    }
    pub(crate) fn set_guest_image(&mut self, id: ImageId, guest: &nixe_gpu::native::GuestImage) {
        self.images[id.slot]
            .entry
            .as_mut()
            .expect("registered image")
            .guest = Some(guest.clone());
    }
    pub(crate) fn image_binding_current(&self, id: ImageId) -> bool {
        self.image(id).is_ok_and(|entry| entry.binding_valid)
    }

    /// Register canonical storage; actual LOAD or shader consumption imports it.
    pub fn image_for(
        &mut self,
        geometry: Vec<Arc<Footprint>>,
        description: NativeImageDescription,
        planes: Vec<ImagePlane>,
    ) -> Result<ImageId, VulkanError> {
        self.create_image_representation(geometry, description, planes, true, None, [1, 1])
    }
    /// Allocate an undefined native target for an actual device producer.
    pub fn image_for_device_write(
        &mut self,
        geometry: Vec<Arc<Footprint>>,
        description: NativeImageDescription,
        planes: Vec<ImagePlane>,
    ) -> Result<ImageId, VulkanError> {
        self.create_image_representation(geometry, description, planes, false, None, [1, 1])
    }
    /// Device-authored samples share canonical ownership with ordinary images.
    /// Their sample grid is supplied by the frontend, independently of host layout.
    pub fn multisample_image_for(
        &mut self,
        geometry: Vec<Arc<Footprint>>,
        mut description: NativeImageDescription,
        planes: Vec<ImagePlane>,
        sample_grid: [u32; 2],
    ) -> Result<ImageId, VulkanError> {
        description.usage |= vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
        // CPU sample visibility reads encoded bits through a compatible UNORM
        // view, avoiding an approximate inverse of the sampler's sRGB decode.
        if matches!(
            description.format,
            vk::Format::R8G8B8A8_SRGB | vk::Format::B8G8R8A8_SRGB
        ) {
            description.flags |= vk::ImageCreateFlags::MUTABLE_FORMAT;
        }
        self.create_image_representation(geometry, description, planes, false, None, sample_grid)
    }
    pub(super) fn create_image_representation(
        &mut self,
        geometry: Vec<Arc<Footprint>>,
        mut description: NativeImageDescription,
        planes: Vec<ImagePlane>,
        initialize: bool,
        decoded: Option<super::decoded_texture::DecodedImage>,
        sample_grid: [u32; 2],
    ) -> Result<ImageId, VulkanError> {
        description.usage |= vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
        if self.device.profile().attachment_feedback_loop
            && description.usage.contains(vk::ImageUsageFlags::SAMPLED)
            && !description.usage.contains(vk::ImageUsageFlags::STORAGE)
            && description.usage.intersects(
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            )
        {
            description.usage |= vk::ImageUsageFlags::ATTACHMENT_FEEDBACK_LOOP_EXT;
        }
        if description.samples != vk::SampleCountFlags::TYPE_1 && initialize {
            return Err(VulkanError::InvalidResource(
                "multisample canonical import requires sample conversion",
            ));
        }
        if let Some(decoded) = &decoded {
            let mut encoded = description;
            encoded.format = decoded.format;
            validate_planes(&geometry, encoded, &decoded.planes, [1, 1])?;
        } else {
            validate_planes(&geometry, description, &planes, sample_grid)?;
        }
        if geometry
            .iter()
            .any(|footprint| footprint.mapping().machine() != self.memory.identity())
        {
            return Err(VulkanError::InvalidResource(
                "image belongs to another physical machine",
            ));
        }
        let candidate = self
            .authority
            .resource_candidates(&geometry[0])
            .into_iter()
            .find_map(|resource| {
                let Resource::Image(id) = resource else {
                    return None;
                };
                let entry = self.image(id).ok()?;
                let compatible = NativeImageDescription {
                    usage: entry.description.usage,
                    flags: entry.description.flags,
                    ..description
                };
                (entry.binding_valid
                    && entry.description == compatible
                    && entry.description.usage.contains(description.usage)
                    && entry.description.flags.contains(description.flags)
                    && entry.decoded == decoded
                    && entry.canonical_import == initialize
                    && entry.sample_grid == sample_grid
                    && entry.planes.as_ref() == planes
                    && entry.contents.geometry == geometry)
                    .then_some(id)
            });
        if let Some(id) = candidate {
            self.touch(Resource::Image(id));
            self.statistics.cache_hits += 1;
            return Ok(id);
        }
        let owner = Arc::clone(&self.device.owner);
        let native = allocate_image_charged(
            &owner,
            description,
            AllocationMode::Managed,
            "resident image",
            |bytes| self.reserve(bytes),
        )?;
        let slot = self.free_images.pop().unwrap_or(self.images.len());
        if slot == self.images.len() {
            self.images.push(Slot {
                generation: 0,
                entry: None,
            });
        }
        let generation =
            self.images[slot]
                .generation
                .checked_add(1)
                .ok_or(VulkanError::InvalidResource(
                    "native image generation exhausted",
                ))?;
        let id = ImageId {
            instance: self.instance,
            slot,
            generation,
        };
        self.images[slot] = Slot {
            generation,
            entry: Some(ImageEntry {
                synchronized: false,
                synchronized_sequence: 0,
                cpu_tracked: false,
                guest: None,
                binding_valid: true,
                contents: Contents {
                    lru: None,
                    geometry: geometry.clone(),
                    device: Vec::new(),
                    last_use: None,
                    touched: 0,
                },
                native: Arc::new(native),
                description,
                planes: planes.clone().into(),
                sample_grid,
                views: Vec::new(),
                native_views: Vec::new(),
                access: sync::image_states(
                    &planes,
                    description.format,
                    self.device.profile().separate_depth_stencil_layouts,
                ),
                version: 0,
                producer: None,
                decoded,
                canonical_import: initialize,
                write_plans: Default::default(),
                discarded: planes
                    .iter()
                    .enumerate()
                    .map(|(index, &plane)| (index, super::transfers::full_rect(plane)))
                    .collect(),
            }),
        };
        self.authority
            .register(Resource::Image(id), geometry.clone())
            .map_err(memory_error)?;
        let observations = vec![
            self.authority
                .observe_native_resource(Resource::Image(id))?,
        ];
        self.images[id.slot]
            .entry
            .as_mut()
            .expect("image")
            .contents
            .device = observations;
        self.statistics.image_allocations += 1;
        self.statistics.image_layout_plans += self.image(id)?.contents.geometry.len() as u64;
        self.touch(Resource::Image(id));
        Ok(id)
    }
    pub(super) fn prepare_image_write(
        &mut self,
        id: ImageId,
        plane: ImagePlane,
        rect: vk::Rect2D,
    ) -> Result<WriterId, VulkanError> {
        let entry = self.image(id)?;
        let index =
            entry
                .planes
                .iter()
                .position(|p| *p == plane)
                .ok_or(VulkanError::InvalidResource(
                    "write plane is absent from image",
                ))?;
        let geometry = &entry.contents.geometry[index];
        let nixe_gpu::physical::WriteDomain::Image(mut domain) = geometry.domain() else {
            return Err(VulkanError::InvalidResource(
                "image writer has no image footprint",
            ));
        };
        let stored = entry.storage_rect(rect);
        let x = stored.offset.x as u32 / plane.block_extent.width;
        let y = stored.offset.y as u32 / plane.block_extent.height;
        domain.rect = nixe_gpu::physical::ByteRect {
            x,
            y,
            width: (stored.offset.x as u32 + stored.extent.width)
                .div_ceil(plane.block_extent.width)
                - x,
            height: (stored.offset.y as u32 + stored.extent.height)
                .div_ceil(plane.block_extent.height)
                - y,
        };
        let key = (index, domain.rect);
        let writer = if let Some(&writer) = entry.write_plans.get(&key) {
            writer
        } else {
            let footprint = Footprint::new(
                Arc::clone(geometry.mapping()),
                nixe_gpu::physical::WriteDomain::Image(domain),
            )?;
            let writer = self
                .authority
                .create_writer(Resource::Image(id), footprint)?;
            self.images[id.slot]
                .entry
                .as_mut()
                .expect("image")
                .write_plans
                .insert(key, writer);
            writer
        };
        self.wait_writer(writer)
    }
    pub fn ensure_image(&mut self, id: ImageId) -> Result<(), VulkanError> {
        self.invalidate_cpu_resources(&[])?;
        let current = self.image(id)?.current();
        if current {
            return Ok(());
        }
        if self.refresh_native_alias(id)? {
            self.track_image_cpu(id)?;
            self.images[id.slot]
                .entry
                .as_mut()
                .unwrap()
                .synchronized_sequence = self.authority.modification_sequence().get();
            return self.invalidate_cpu_resources(&[]);
        }
        let entry = self.image(id)?;
        if entry.description.samples != vk::SampleCountFlags::TYPE_1 {
            return Err(VulkanError::InvalidResource(
                "multisample reconstruction needs a compatible native alias or sample conversion",
            ));
        }
        self.constructing.push(Resource::Image(id));
        let result = self
            .refresh_image(id)
            .and_then(|()| self.track_image_cpu(id));
        self.unpin(Resource::Image(id));
        result?;
        self.images[id.slot]
            .entry
            .as_mut()
            .unwrap()
            .synchronized_sequence = self.authority.modification_sequence().get();
        self.invalidate_cpu_resources(&[])
    }
    pub(super) fn ensure_image_initialized(
        &mut self,
        id: ImageId,
        plane: ImagePlane,
        rect: vk::Rect2D,
    ) -> Result<(), VulkanError> {
        self.ensure_image(id)?;
        self.require_image_initialized(id, plane, rect)
    }
    fn refresh_native_alias(&mut self, id: ImageId) -> Result<bool, VulkanError> {
        let target = self.image(id)?;
        if target.decoded.is_some() {
            return Ok(false);
        }
        let geometry = target.contents.geometry.clone();
        let planes = target.planes.clone();
        let synchronized = target.synchronized;
        let since = if synchronized {
            target.synchronized_sequence
        } else {
            0
        };
        let canonical_import = target.canonical_import;
        let grid = target.sample_grid;
        let mut copies = rustc_hash::FxHashMap::<
            (ImageId, u64, u64),
            (ImagePlane, ImagePlane, Vec<vk::ImageCopy>),
        >::default();
        let mut undefined = Vec::new();
        for (&destination, footprint) in planes.iter().zip(&geometry) {
            let Some(changes) =
                self.authority
                    .image_changes(Resource::Image(id), footprint, since)?
            else {
                nixe_gpu::trace::detail_event(
                    "image_reinterpret",
                    id.slot as u64,
                    destination.offset,
                    since,
                );
                return Ok(false);
            };
            if !synchronized {
                if canonical_import && !changes.unowned.is_empty() {
                    return Ok(false);
                }
                for rect in changes.unowned {
                    undefined.push((destination, native_rect(destination, grid, rect)));
                }
            }
            for copy in changes.copies {
                let Resource::Image(source) = copy.resource else {
                    return Ok(false);
                };
                if source == id && copy.native_base == destination.offset {
                    continue;
                }
                if source == id {
                    return Ok(false);
                }
                let source_entry = self.image(source)?;
                let Some(&from) = source_entry
                    .planes
                    .iter()
                    .find(|plane| plane.offset == copy.native_base)
                else {
                    return Ok(false);
                };
                if source_entry.decoded.is_some()
                    || source_entry.sample_grid != grid
                    || source_entry.description.samples != self.image(id)?.description.samples
                    || from.aspect != destination.aspect
                    || !format::copy_compatible(
                        source_entry.description.format,
                        self.image(id)?.description.format,
                        destination.aspect,
                    )
                {
                    return Ok(false);
                }
                let from_rect = native_rect(from, grid, copy.rect);
                let to_rect = native_rect(destination, grid, copy.rect);
                if from_rect.extent != to_rect.extent {
                    return Ok(false);
                }
                let region = vk::ImageCopy::default()
                    .src_subresource(from.layers())
                    .dst_subresource(destination.layers())
                    .src_offset(vk::Offset3D {
                        x: from_rect.offset.x,
                        y: from_rect.offset.y,
                        z: from.depth_slice as i32,
                    })
                    .dst_offset(vk::Offset3D {
                        x: to_rect.offset.x,
                        y: to_rect.offset.y,
                        z: destination.depth_slice as i32,
                    })
                    .extent(vk::Extent3D {
                        width: to_rect.extent.width,
                        height: to_rect.extent.height,
                        depth: 1,
                    });
                copies
                    .entry((source, from.offset, destination.offset))
                    .or_insert_with(|| (from, destination, Vec::new()))
                    .2
                    .push(region);
            }
        }
        let copied = !copies.is_empty();
        let observed = vec![
            self.authority
                .observe_native_resource(Resource::Image(id))?,
        ];
        let retained = self.constructing.len();
        self.constructing.push(Resource::Image(id));
        self.constructing
            .extend(copies.keys().map(|&(source, ..)| Resource::Image(source)));
        let result: Result<(), VulkanError> = (|| {
            for ((source, ..), (from, destination, regions)) in copies {
                self.copy_alias_regions(source, from, id, destination, &regions)?;
            }
            for (plane, rect) in undefined {
                self.mark_image_discarded(id, plane, rect);
            }
            Ok(())
        })();
        self.constructing.truncate(retained);
        result?;
        let work = if copied {
            Some(self.device.work_ticket()?)
        } else {
            None
        };
        let entry = self.images[id.slot].entry.as_mut().expect("image");
        entry.contents.device = observed;
        entry.synchronized = true;
        if let Some(work) = work {
            entry.producer = Some(work);
            entry.version = entry
                .version
                .checked_add(1)
                .ok_or(VulkanError::InvalidResource(
                    "image content version exhausted",
                ))?;
        }
        Ok(true)
    }
    pub(super) fn track_image_cpu(&mut self, id: ImageId) -> Result<(), VulkanError> {
        if self.image(id)?.cpu_tracked {
            return Ok(());
        }
        let mut pages = Vec::new();
        for footprint in &self.image(id)?.contents.geometry {
            footprint.mapping().observed_pages(footprint, &mut pages)?;
        }
        let inputs = nixe_gpu::physical::pin_selected_pages(pages)?;
        self.memory
            .capture_resource_scattered(&inputs, &[], &mut [])?;
        self.images[id.slot].entry.as_mut().unwrap().cpu_tracked = true;
        Ok(())
    }
    fn refresh_image(&mut self, id: ImageId) -> Result<(), VulkanError> {
        let extent = self.image(id)?.description.extent;
        let _trace = nixe_gpu::trace::span(
            "image_refresh",
            id.slot as u64,
            u64::from(extent.width),
            u64::from(extent.height),
        );
        if self.image(id)?.decoded.is_some() {
            return self.refresh_decoded_image(id);
        }
        let entry = self.image(id)?;
        let geometry = entry.contents.geometry.clone();
        let planes = entry.planes.clone();
        let canonical_import = entry.canonical_import;
        let bytes = planes
            .iter()
            .map(|p| p.end().expect("validated image plane"))
            .max()
            .expect("image planes");
        // RAM-only textures need an unswizzle, not a byte-wise competition
        // between native producers. Keep full resolution for actual aliases.
        // https://github.com/eden-emulator/mirror/blob/master/src/video_core/texture_cache/texture_cache.h
        let dense_rows = planes.iter().all(|p| {
            p.row_pitch == p.row_bytes()
                && p.row_pitch <= self.device.submission.configuration.arena_bytes
        });
        let cpu_only = canonical_import
            && dense_rows
            && geometry.iter().all(|f| !self.authority.has_writers(f));
        let mut parts = Vec::new();
        if !cpu_only {
            for footprint in &geometry {
                parts.extend(self.authority.resolve_footprint(footprint)?);
            }
        }
        let device = vec![
            self.authority
                .observe_native_resource(Resource::Image(id))?,
        ];
        // An opaque view's own writes are already in its native image. Unknown
        // CPU holes remain undefined; neither case needs a buffer round trip.
        if !canonical_import
            && parts.iter().all(|part| {
                part.producer.is_none_or(|source| {
                    source.resource == Resource::Image(id) && source.native == part.logical
                })
            })
        {
            self.discard_cpu_image_holes(id, &parts)?;
            let entry = self.images[id.slot].entry.as_mut().expect("image");
            entry.contents.device = device;
            entry.synchronized = true;
            return Ok(());
        }
        if canonical_import && parts.iter().all(|p| p.producer.is_none()) && dense_rows {
            let mut inputs = Vec::new();
            for footprint in &geometry {
                footprint.mapping().observed_pages(footprint, &mut inputs)?;
            }
            let inputs = nixe_gpu::physical::pin_selected_pages(inputs)?;
            let mut copies = Vec::new();
            if cpu_only {
                for footprint in &geometry {
                    for (index, span) in footprint.mapping().spans().iter().enumerate() {
                        footprint.visit(index, span.start, span.bytes, |part| {
                            copies.push(nixe_memory::PhysicalCapture {
                                source: nixe_memory::Extent {
                                    start: part.start,
                                    bytes: part.bytes,
                                },
                                destination: part.native as usize,
                            });
                            true
                        })?;
                    }
                }
            } else {
                copies.extend(parts.iter().map(|p| nixe_memory::PhysicalCapture {
                    source: nixe_memory::Extent {
                        start: p.start,
                        bytes: p.bytes,
                    },
                    destination: p.logical as usize,
                }));
            }
            let mut captured = vec![0; bytes as usize];
            self.memory
                .capture_resource_scattered(&inputs, &copies, &mut captured)?;
            for plane in planes.iter() {
                let end = plane.end().expect("validated plane") as usize;
                self.upload_cpu_image_rows(id, *plane, 0, &captured[plane.offset as usize..end])?;
            }
        } else {
            self.refresh_image_intersections(id, &parts, bytes)?;
        }
        // A copy of a partially covered transfer block must not validate CPU
        // bytes which this opaque representation cannot import.
        if !canonical_import {
            self.discard_cpu_image_holes(id, &parts)?;
        }
        let work = self.device.work_ticket()?;
        let entry = self.images[id.slot].entry.as_mut().expect("image");
        entry.contents.device = device;
        entry.synchronized = true;
        entry.producer = Some(work);
        entry.version = entry
            .version
            .checked_add(1)
            .ok_or(VulkanError::InvalidResource(
                "image content version exhausted",
            ))?;
        Ok(())
    }
    fn discard_cpu_image_holes(
        &mut self,
        id: ImageId,
        parts: &[nixe_gpu::authority::ResolvedBytes<Resource>],
    ) -> Result<(), VulkanError> {
        let planes = self.image(id)?.planes.clone();
        let mut holes: Vec<_> = parts
            .iter()
            .filter(|p| p.producer.is_none())
            .map(|p| (p.logical, p.bytes))
            .collect();
        normalize(&mut holes);
        for (start, bytes) in holes {
            for &plane in planes.iter() {
                let begin = start.max(plane.offset);
                let end = (start + bytes).min(plane.end().unwrap());
                if begin >= end {
                    continue;
                }
                let block = u64::from(plane.bytes_per_block);
                let first = (begin - plane.offset) / plane.row_pitch;
                let last = (end - plane.offset).div_ceil(plane.row_pitch);
                let mut row = first;
                while row < last {
                    let base = plane.offset + row * plane.row_pitch;
                    let low = begin.max(base);
                    let high = end.min(base + plane.row_bytes());
                    if low >= high {
                        row += 1;
                        continue;
                    }
                    let rows = if low == base && high == base + plane.row_bytes() {
                        (1 + (end - high) / plane.row_pitch).min(last - row)
                    } else {
                        1
                    };
                    let x = ((low - base) / block) as u32 * plane.block_extent.width;
                    let stop = (((high - base).div_ceil(block)) as u32 * plane.block_extent.width)
                        .min(plane.extent.width);
                    let y = row as u32 * plane.block_extent.height;
                    self.mark_image_discarded(
                        id,
                        plane,
                        vk::Rect2D {
                            offset: vk::Offset2D {
                                x: x as i32,
                                y: y as i32,
                            },
                            extent: vk::Extent2D {
                                width: stop - x,
                                height: (rows as u32 * plane.block_extent.height)
                                    .min(plane.extent.height - y),
                            },
                        },
                    );
                    row += rows;
                }
            }
        }
        Ok(())
    }
    fn upload_cpu_image_rows(
        &mut self,
        id: ImageId,
        plane: ImagePlane,
        offset: u64,
        bytes: &[u8],
    ) -> Result<(), VulkanError> {
        debug_assert!(offset.is_multiple_of(plane.row_pitch));
        debug_assert!((bytes.len() as u64).is_multiple_of(plane.row_pitch));
        let rows = self.device.submission.configuration.arena_bytes / plane.row_pitch;
        let chunk_bytes = (rows * plane.row_pitch) as usize;
        for (chunk, bytes) in bytes.chunks(chunk_bytes).enumerate() {
            let row = (offset + (chunk * chunk_bytes) as u64) / plane.row_pitch;
            let y = row as u32 * plane.block_extent.height;
            let height = (bytes.len() as u64 / plane.row_pitch) as u32 * plane.block_extent.height;
            let height = height.min(plane.extent.height - y);
            let (slot, slice) =
                self.reserve_upload(bytes, u64::from(plane.bytes_per_block).max(4))?;
            let (image, layout) = self.image_access_subresources(
                id,
                commands::plane_range(plane),
                AccessScope::new(
                    vk::PipelineStageFlags2::COPY,
                    vk::AccessFlags2::TRANSFER_WRITE,
                ),
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            )?;
            let region = vk::BufferImageCopy::default()
                .buffer_offset(slice.offset)
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(plane.aspect)
                        .mip_level(plane.mip_level)
                        .base_array_layer(plane.array_layer)
                        .layer_count(1),
                )
                .image_offset(vk::Offset3D {
                    x: 0,
                    y: y as i32,
                    z: plane.depth_slice as i32,
                })
                .image_extent(vk::Extent3D {
                    width: plane.extent.width,
                    height,
                    depth: 1,
                });
            self.barriers.emit(&mut self.device);
            unsafe {
                self.device.dispatch.cmd_copy_buffer_to_image(
                    self.device.submission.slots[slot]
                        .upload_buffer(slice)
                        .handle,
                    image,
                    layout,
                    &[region],
                );
            }
            self.define_image_rect(
                id,
                plane,
                vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: y as i32 },
                    extent: vk::Extent2D {
                        width: plane.extent.width,
                        height,
                    },
                },
            );
            self.statistics.cpu_uploaded_bytes += bytes.len() as u64;
        }
        Ok(())
    }
    pub(super) fn copy_rows_to_image(
        &mut self,
        source: &Arc<AllocatedBuffer>,
        source_offset: u64,
        image: ImageId,
        dirty: &[(u64, u64)],
    ) -> Result<(), VulkanError> {
        let planes = self.image(image)?.planes.clone();
        let mut pieces = Vec::new();
        for (index, plane) in planes.iter().copied().enumerate() {
            let nixe_gpu::physical::WriteDomain::Image(domain) =
                self.image(image)?.contents.geometry[index].domain()
            else {
                unreachable!()
            };
            let Some(&(first, _)) = dirty.first() else {
                break;
            };
            let &(last, bytes) = dirty.last().unwrap();
            let first_row =
                (first.saturating_sub(plane.offset) / plane.row_pitch).min(u64::from(plane.rows()));
            let last_row = ((last + bytes)
                .saturating_sub(plane.offset)
                .div_ceil(plane.row_pitch))
            .min(u64::from(plane.rows()));
            for row in first_row as u32..last_row as u32 {
                let offset = plane.offset + plane.row_pitch * u64::from(row);
                let first = dirty.partition_point(|&(start, bytes)| start + bytes <= offset);
                for &(start, length) in &dirty[first..] {
                    if start >= offset + plane.row_bytes() {
                        break;
                    }
                    let begin = start.max(offset);
                    let end = (start + length).min(offset + plane.row_bytes());
                    if begin >= end {
                        continue;
                    }
                    let block = u64::from(plane.bytes_per_block);
                    let first = (begin - offset).div_ceil(block);
                    let stop = (end - offset + block - u64::from(domain.aspect_bytes)) / block;
                    if first >= stop {
                        continue;
                    }
                    let logical = offset + first * block;
                    let length = (stop - first) * block;
                    pieces.push(ImagePiece {
                        plane,
                        row,
                        rows: 1,
                        first: first as u32,
                        blocks: (stop - first) as u32,
                        skip: 0,
                        logical,
                        length,
                    });
                }
            }
        }
        let pieces = coalesce_dense_rows(pieces);
        if pieces.is_empty() {
            return Ok(());
        }
        let scratch = if pieces
            .iter()
            .all(|piece| piece.direct(piece.logical - source_offset))
        {
            None
        } else {
            let (size, slices) = packed_slices(&pieces)?;
            let scratch = self.temporary_buffer(size, &[Resource::Image(image)])?;
            Some((scratch, slices))
        };
        let src = source.handle;
        let (dst, layout) =
            self.image_regions_access(image, &pieces, vk::AccessFlags2::TRANSFER_WRITE)?;
        self.device.begin_recording()?;

        let work = self.device.work_ticket()?;
        self.device.retire_buffer(Arc::clone(source), work)?;
        let regions = if let Some((scratch, slices)) = &scratch {
            let copies: Vec<_> = pieces
                .iter()
                .zip(slices)
                .map(|(piece, &start)| {
                    vk::BufferCopy::default()
                        .src_offset(piece.logical - source_offset)
                        .dst_offset(start)
                        .size(piece.length)
                })
                .collect();
            self.barriers.emit(&mut self.device);
            unsafe {
                self.device
                    .dispatch
                    .cmd_copy_buffer(src, scratch.handle, &copies);
                scratch_read_barrier(&mut self.device, scratch.handle, scratch.size());
            }
            pieces
                .iter()
                .zip(slices)
                .map(|(piece, &start)| piece.region(start))
                .collect::<Vec<_>>()
        } else {
            pieces
                .iter()
                .map(|piece| piece.region(piece.logical - source_offset))
                .collect()
        };
        self.barriers.emit(&mut self.device);
        unsafe {
            self.device.dispatch.cmd_copy_buffer_to_image(
                scratch.as_ref().map_or(src, |(buffer, _)| buffer.handle),
                dst,
                layout,
                &regions,
            );
        }
        if let Some((scratch, _)) = scratch {
            let work = self.device.work_ticket()?;
            self.device.retire_buffer(scratch, work)?;
        }
        for piece in &pieces {
            self.define_image_rect(
                image,
                piece.plane,
                vk::Rect2D {
                    offset: vk::Offset2D {
                        x: (piece.first * piece.plane.block_extent.width) as i32,
                        y: (piece.row * piece.plane.block_extent.height) as i32,
                    },
                    extent: vk::Extent2D {
                        width: (piece.blocks * piece.plane.block_extent.width).min(
                            piece.plane.extent.width - piece.first * piece.plane.block_extent.width,
                        ),
                        height: (piece.rows * piece.plane.block_extent.height).min(
                            piece.plane.extent.height - piece.row * piece.plane.block_extent.height,
                        ),
                    },
                },
            );
        }
        self.statistics.gpu_copied_bytes += pieces.iter().map(|piece| piece.length).sum::<u64>();
        Ok(())
    }
    pub fn image_view(
        &mut self,
        image: ImageId,
        description: ImageViewDescription,
    ) -> Result<ImageViewId, VulkanError> {
        let entry = self.image(image)?;
        if let Some(view) = entry.views.iter().position(|view| *view == description) {
            return Ok(ImageViewId { image, view });
        }
        let d = description;
        let same_format = d.format == entry.description.format;
        if !same_format
            && (!format::view_compatible(entry.description.format, d.format)
                || !entry
                    .description
                    .flags
                    .contains(vk::ImageCreateFlags::MUTABLE_FORMAT))
        {
            return Err(VulkanError::InvalidResource(
                "image view format requires a supported mutable-format compatibility class",
            ));
        }
        if d.mip_count == 0
            || d.layer_count == 0
            || d.aspects.is_empty()
            || d.base_mip
                .checked_add(d.mip_count)
                .is_none_or(|end| end > entry.description.mip_levels)
            || d.base_layer.checked_add(d.layer_count).is_none_or(|end| {
                end > if entry.description.image_type == vk::ImageType::TYPE_3D
                    && d.view_type != vk::ImageViewType::TYPE_3D
                {
                    (entry.description.extent.depth >> d.base_mip).max(1)
                } else {
                    entry.description.array_layers
                }
            })
        {
            return Err(VulkanError::InvalidResource(
                "image view exceeds resident subresources",
            ));
        }
        let aspects = entry
            .planes
            .iter()
            .fold(vk::ImageAspectFlags::empty(), |aspects, plane| {
                aspects | plane.aspect
            });
        if !aspects.contains(d.aspects) {
            return Err(VulkanError::InvalidResource(
                "image view aspect has no native plane",
            ));
        }
        for level in d.base_mip..d.base_mip + d.mip_count {
            for layer in d.base_layer..d.base_layer + d.layer_count {
                for aspect in [
                    vk::ImageAspectFlags::COLOR,
                    vk::ImageAspectFlags::DEPTH,
                    vk::ImageAspectFlags::STENCIL,
                ] {
                    if d.aspects.contains(aspect)
                        && !entry.planes.iter().any(|plane| {
                            plane.mip_level == level
                                && (if entry.description.image_type == vk::ImageType::TYPE_3D
                                    && d.view_type != vk::ImageViewType::TYPE_3D
                                {
                                    plane.depth_slice
                                } else {
                                    plane.array_layer
                                }) == layer
                                && plane.aspect == aspect
                        })
                    {
                        return Err(VulkanError::InvalidResource(
                            "image view references an uninitialized native subresource",
                        ));
                    }
                }
            }
        }
        match d.view_type {
            vk::ImageViewType::TYPE_2D if d.layer_count == 1 => {}
            vk::ImageViewType::TYPE_2D_ARRAY => {}
            vk::ImageViewType::TYPE_3D
                if entry.description.image_type == vk::ImageType::TYPE_3D
                    && d.base_layer == 0
                    && d.layer_count == 1 => {}
            vk::ImageViewType::CUBE | vk::ImageViewType::CUBE_ARRAY
                if entry
                    .description
                    .flags
                    .contains(vk::ImageCreateFlags::CUBE_COMPATIBLE)
                    && d.layer_count.is_multiple_of(6)
                    && (d.view_type != vk::ImageViewType::CUBE || d.layer_count == 6) => {}
            _ => {
                return Err(VulkanError::InvalidResource(
                    "image view type is incompatible with resident image/layers",
                ));
            }
        }
        if self
            .device
            .profile()
            .portability
            .as_ref()
            .is_some_and(|p| !p.image_view_format_swizzle)
            && d.swizzle != [vk::ComponentSwizzle::IDENTITY; 4]
        {
            return Err(VulkanError::InvalidResource(
                "component swizzle requires imageViewFormatSwizzle on this portability device",
            ));
        }
        let info = vk::ImageViewCreateInfo::default()
            .image(entry.native.handle)
            .view_type(d.view_type)
            .format(d.format)
            .components(vk::ComponentMapping {
                r: d.swizzle[0],
                g: d.swizzle[1],
                b: d.swizzle[2],
                a: d.swizzle[3],
            })
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(d.aspects)
                    .base_mip_level(d.base_mip)
                    .level_count(d.mip_count)
                    .base_array_layer(d.base_layer)
                    .layer_count(d.layer_count),
            );
        // Alternate views must support the inherited usage on this device.
        // https://docs.vulkan.org/refpages/latest/refpages/source/VkImageViewCreateInfo.html
        let _supported = unsafe {
            self.device
                .owner
                .instance
                .instance
                .get_physical_device_image_format_properties(
                    self.device.owner.physical,
                    d.format,
                    entry.description.image_type,
                    vk::ImageTiling::OPTIMAL,
                    entry.description.usage,
                    entry.description.flags,
                )
        }
        .map_err(|e| VulkanError::vk("image view format usage", e))?;
        let handle = unsafe { self.device.owner.device.create_image_view(&info, None) }
            .map_err(|e| VulkanError::vk("vkCreateImageView", e))?;
        let entry = self.images[image.slot].entry.as_mut().expect("image");
        let view = entry.views.len();
        entry.views.push(description);
        entry
            .native_views
            .push(Arc::new(crate::AllocatedImageView::from_handle(
                Arc::clone(&entry.native),
                handle,
            )));
        self.statistics.image_view_allocations += 1;
        Ok(ImageViewId { image, view })
    }
    pub(super) fn image_pieces(
        &self,
        id: ImageId,
        offset: u64,
        length: u64,
    ) -> Result<Vec<ImagePiece>, VulkanError> {
        if self.image(id)?.description.samples != vk::SampleCountFlags::TYPE_1 {
            return Err(VulkanError::InvalidResource(
                "multisample samples cannot be copied as ordinary buffer texels",
            ));
        }
        self.image_storage_pieces(id, offset, length)
    }
    /// Byte coordinates describe every stored sample. CPU download converts
    /// those samples explicitly; ordinary image/buffer copies require TYPE_1.
    fn image_storage_pieces(
        &self,
        id: ImageId,
        offset: u64,
        length: u64,
    ) -> Result<Vec<ImagePiece>, VulkanError> {
        let entry = self.image(id)?;
        let end = offset
            .checked_add(length)
            .ok_or(VulkanError::InvalidResource("image interval overflow"))?;
        let mut result = Vec::new();
        let mut consumed = 0;
        for &native_plane in entry.planes.iter() {
            let plane = entry.storage_plane(native_plane);
            // Authority often names only a few bytes of one packed texel.
            // Locate its rows arithmetically instead of scanning the full mip
            // again for every physical fragment of a CPU visibility request.
            if end <= plane.offset {
                continue;
            }
            let first_row = offset.saturating_sub(plane.offset) / plane.row_pitch;
            let last_row = (end - plane.offset).div_ceil(plane.row_pitch);
            for row in first_row.min(u64::from(plane.rows())) as u32
                ..last_row.min(u64::from(plane.rows())) as u32
            {
                let row_start = plane.offset + plane.row_pitch * u64::from(row);
                let start = offset.max(row_start);
                let finish = end.min(row_start + plane.row_bytes());
                if start >= finish {
                    continue;
                }
                let block = u64::from(plane.bytes_per_block);
                let first = (start - row_start) / block;
                let last = (finish - row_start).div_ceil(block);
                result.push(ImagePiece {
                    plane,
                    row,
                    rows: 1,
                    first: first as u32,
                    blocks: (last - first) as u32,
                    skip: start - row_start - first * block,
                    logical: start,
                    length: finish - start,
                });
                consumed += finish - start;
            }
        }
        for piece in &result {
            let plane = entry
                .planes
                .iter()
                .find(|plane| {
                    plane.aspect == piece.plane.aspect
                        && plane.mip_level == piece.plane.mip_level
                        && plane.array_layer == piece.plane.array_layer
                        && plane.depth_slice == piece.plane.depth_slice
                })
                .expect("stored image plane");
            let x = piece.first * piece.plane.block_extent.width;
            let y = piece.row * piece.plane.block_extent.height;
            let stop_x =
                (x + piece.blocks * piece.plane.block_extent.width).min(piece.plane.extent.width);
            let stop_y = (y + piece.plane.block_extent.height).min(piece.plane.extent.height);
            let start_x = x / entry.sample_grid[0];
            let start_y = y / entry.sample_grid[1];
            self.require_image_initialized(
                id,
                *plane,
                vk::Rect2D {
                    offset: vk::Offset2D {
                        x: start_x as i32,
                        y: start_y as i32,
                    },
                    extent: vk::Extent2D {
                        width: stop_x.div_ceil(entry.sample_grid[0]) - start_x,
                        height: stop_y.div_ceil(entry.sample_grid[1]) - start_y,
                    },
                },
            )?;
        }
        result.sort_unstable_by_key(|piece| piece.logical);
        if consumed != length {
            return Err(VulkanError::InvalidResource(
                "image producer interval crosses padding or missing plane",
            ));
        }
        Ok(coalesce_dense_rows(result))
    }
    pub(super) fn image_regions_access(
        &mut self,
        id: ImageId,
        pieces: &[ImagePiece],
        access: vk::AccessFlags2,
    ) -> Result<(vk::Image, vk::ImageLayout), VulkanError> {
        let mut result = (vk::Image::null(), vk::ImageLayout::UNDEFINED);
        let conventional = if access == vk::AccessFlags2::TRANSFER_READ {
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL
        } else {
            vk::ImageLayout::TRANSFER_DST_OPTIMAL
        };
        for piece in pieces {
            result = self.image_access_subresources(
                id,
                super::commands::plane_range(piece.plane),
                AccessScope::new(vk::PipelineStageFlags2::COPY, access),
                conventional,
            )?;
        }
        Ok(result)
    }
    /// Download whole transfer blocks once, then scatter only authoritative
    /// bytes on the CPU. D24 padding and independent stencil/alias producers
    /// must never become canonical merely because a native transfer reads them.
    pub(super) fn image_read_pieces(
        &self,
        id: ImageId,
        writes: &[nixe_gpu::authority::ResolvedBytes<Resource>],
        limit: u64,
    ) -> Result<Vec<ImagePiece>, VulkanError> {
        let mut pieces = Vec::new();
        for write in writes {
            for piece in self.image_storage_pieces(
                id,
                write.producer.expect("device readback source").native,
                write.bytes,
            )? {
                let block = u64::from(piece.plane.bytes_per_block);
                for row in piece.row..piece.row + piece.rows {
                    pieces.push(ImagePiece {
                        row,
                        rows: 1,
                        skip: 0,
                        logical: piece.plane.offset
                            + u64::from(row) * piece.plane.row_pitch
                            + u64::from(piece.first) * block,
                        length: u64::from(piece.blocks) * block,
                        ..piece
                    });
                }
            }
        }
        pieces.sort_unstable_by_key(|piece| piece.logical);
        let mut merged: Vec<ImagePiece> = Vec::new();
        for piece in pieces {
            if let Some(last) = merged.last_mut()
                && last.plane == piece.plane
                && last.row == piece.row
                && piece.first <= last.first + last.blocks
            {
                last.blocks =
                    (piece.first + piece.blocks).max(last.first + last.blocks) - last.first;
                last.length = u64::from(last.blocks) * u64::from(last.plane.bytes_per_block);
            } else {
                merged.push(piece);
            }
        }
        let mut bounded = Vec::new();
        for piece in merged {
            let block = u64::from(piece.plane.bytes_per_block);
            let blocks = (limit / block).min(u64::from(u32::MAX)) as u32;
            if blocks == 0 {
                return Err(VulkanError::InvalidResource(
                    "readback arena cannot hold an image block",
                ));
            }
            let mut consumed = 0;
            while consumed < piece.blocks {
                let count = blocks.min(piece.blocks - consumed);
                bounded.push(ImagePiece {
                    first: piece.first + consumed,
                    blocks: count,
                    logical: piece.logical + u64::from(consumed) * block,
                    length: u64::from(count) * block,
                    ..piece
                });
                consumed += count;
            }
        }
        Ok(bounded)
    }
    pub(super) fn image_pieces_buffer(
        &mut self,
        id: ImageId,
        pieces: &[ImagePiece],
        references: &[Resource],
    ) -> Result<(vk::Buffer, Vec<u64>), VulkanError> {
        let (size, slices) = packed_slices(pieces)?;
        let mut references = references.to_vec();
        references.push(Resource::Image(id));
        let scratch = self.temporary_buffer(size, &references)?;
        let mut source = vk::Image::null();
        let mut layout = vk::ImageLayout::UNDEFINED;
        for piece in pieces {
            (source, layout) = self.image_access_subresources(
                id,
                super::commands::plane_range(piece.plane),
                AccessScope::new(
                    vk::PipelineStageFlags2::COPY,
                    vk::AccessFlags2::TRANSFER_READ,
                ),
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            )?;
        }
        self.device.begin_recording()?;

        let regions: Vec<_> = pieces
            .iter()
            .zip(&slices)
            .map(|(piece, &offset)| piece.region(offset))
            .collect();
        self.barriers.emit(&mut self.device);
        unsafe {
            self.device
                .dispatch
                .cmd_copy_image_to_buffer(source, layout, scratch.handle, &regions);
            scratch_read_barrier(&mut self.device, scratch.handle, size);
        }
        let handle = scratch.handle;
        let work = self.device.work_ticket()?;
        self.device.retire_buffer(scratch, work)?;
        Ok((handle, slices))
    }
}

#[derive(Clone, Copy)]
pub(super) struct ImagePiece {
    pub(super) plane: ImagePlane,
    pub(super) row: u32,
    pub(super) rows: u32,
    pub(super) first: u32,
    pub(super) blocks: u32,
    pub(super) skip: u64,
    pub(super) logical: u64,
    pub(super) length: u64,
}
impl ImagePiece {
    pub(super) fn direct(self, offset: u64) -> bool {
        self.skip == 0
            && self.length
                == u64::from(self.blocks)
                    * u64::from(self.plane.bytes_per_block)
                    * u64::from(self.rows)
            && offset.is_multiple_of(u64::from(self.plane.bytes_per_block).max(4))
    }
    pub(super) fn region(self, offset: u64) -> vk::BufferImageCopy {
        let plane = self.plane;
        let x = self.first * plane.block_extent.width;
        let y = self.row * plane.block_extent.height;
        vk::BufferImageCopy::default()
            .buffer_offset(offset)
            .image_subresource(plane.layers())
            .image_offset(vk::Offset3D {
                x: x as i32,
                y: y as i32,
                z: plane.depth_slice as i32,
            })
            .image_extent(vk::Extent3D {
                width: (self.blocks * plane.block_extent.width).min(plane.extent.width - x),
                height: (plane.block_extent.height * self.rows).min(plane.extent.height - y),
                depth: 1,
            })
    }
}
fn coalesce_dense_rows(pieces: Vec<ImagePiece>) -> Vec<ImagePiece> {
    let mut merged: Vec<ImagePiece> = Vec::with_capacity(pieces.len());
    for piece in pieces {
        if let Some(last) = merged.last_mut()
            && last.plane == piece.plane
            && last.first == 0
            && piece.first == 0
            && last.skip == 0
            && piece.skip == 0
            && last.blocks == piece.blocks
            && last.plane.row_pitch == last.plane.row_bytes()
            && last.length == last.plane.row_bytes() * u64::from(last.rows)
            && piece.length == piece.plane.row_bytes()
            && last.row + last.rows == piece.row
            && last.logical + last.length == piece.logical
        {
            last.rows += piece.rows;
            last.length += piece.length;
        } else {
            merged.push(piece);
        }
    }
    merged
}
fn packed_slices(pieces: &[ImagePiece]) -> Result<(u64, Vec<u64>), VulkanError> {
    let mut size = 0;
    let mut slices = Vec::with_capacity(pieces.len());
    for piece in pieces {
        size = crate::queue::align_up(size, u64::from(piece.plane.bytes_per_block).max(4)).ok_or(
            VulkanError::InvalidResource("packed image copy size overflow"),
        )?;
        slices.push(size);
        size = size
            .checked_add(
                u64::from(piece.blocks)
                    * u64::from(piece.plane.bytes_per_block)
                    * u64::from(piece.rows),
            )
            .ok_or(VulkanError::InvalidResource(
                "packed image copy size overflow",
            ))?;
    }
    Ok((size, slices))
}

fn validate_planes(
    geometry: &[Arc<Footprint>],
    description: NativeImageDescription,
    planes: &[ImagePlane],
    sample_grid: [u32; 2],
) -> Result<(), VulkanError> {
    if planes.is_empty() || geometry.len() != planes.len() {
        return Err(VulkanError::InvalidResource(
            "image residency needs native copy planes",
        ));
    }
    if !matches!(
        description.image_type,
        vk::ImageType::TYPE_2D | vk::ImageType::TYPE_3D
    ) || sample_grid[0] == 0
        || sample_grid[1] == 0
        || sample_grid[0].checked_mul(sample_grid[1]) != Some(description.samples.as_raw())
        || description.samples.as_raw().count_ones() != 1
        || description.samples != vk::SampleCountFlags::TYPE_1 && description.mip_levels != 1
    {
        return Err(VulkanError::InvalidResource(
            "image residency requires 2D planes and a valid frontend sample grid",
        ));
    }
    let mut prior = 0;
    let mut subresources = std::collections::HashSet::new();
    for (plane, footprint) in planes.iter().zip(geometry) {
        let nixe_gpu::physical::WriteDomain::Image(physical) = footprint.domain() else {
            return Err(VulkanError::InvalidResource(
                "native image requires image footprint geometry",
            ));
        };
        if !subresources.insert((
            plane.mip_level,
            plane.array_layer,
            plane.depth_slice,
            plane.aspect.as_raw(),
        )) {
            return Err(VulkanError::InvalidResource(
                "duplicate native image subresource plane",
            ));
        }
        if plane.extent.width == 0
            || plane.extent.height == 0
            || plane.extent.depth != 1
            || plane.block_extent.width == 0
            || plane.block_extent.height == 0
            || plane.bytes_per_block == 0
            || plane.aspect.as_raw().count_ones() != 1
            || plane.mip_level >= description.mip_levels
            || plane.array_layer >= description.array_layers
            || plane.depth_slice >= (description.extent.depth >> plane.mip_level).max(1)
            || (description.image_type != vk::ImageType::TYPE_3D && plane.depth_slice != 0)
        {
            return Err(VulkanError::InvalidResource(
                "invalid native image copy plane",
            ));
        }
        if format::copy_block(description.format, plane.aspect)
            != Some((plane.block_extent, plane.bytes_per_block))
        {
            return Err(VulkanError::InvalidResource(
                "native copy plane geometry does not match format/aspect; packed or unsupported encodings require conversion recording",
            ));
        }
        let width = description
            .extent
            .width
            .checked_shr(plane.mip_level)
            .unwrap_or(0)
            .max(1);
        let height = description
            .extent
            .height
            .checked_shr(plane.mip_level)
            .unwrap_or(0)
            .max(1);
        if plane.extent.width != width
            || plane.extent.height != height
            || plane.mip_level >= u32::BITS
        {
            return Err(VulkanError::InvalidResource(
                "image copy plane extent does not match native mip extent",
            ));
        }
        let storage = ImagePlane {
            extent: vk::Extent3D {
                width: width
                    .checked_mul(sample_grid[0])
                    .ok_or(VulkanError::InvalidResource("sample width overflow"))?,
                height: height
                    .checked_mul(sample_grid[1])
                    .ok_or(VulkanError::InvalidResource("sample height overflow"))?,
                depth: 1,
            },
            ..*plane
        };
        if plane.row_pitch < storage.row_bytes()
            || plane.offset < prior
            || storage.end().is_none()
            || physical.depth_slice != plane.depth_slice
            || physical.native_base != plane.offset
            || physical.native_row_pitch != plane.row_pitch
            || physical.native_element_bytes != plane.bytes_per_block
            || physical.columns != storage.extent.width.div_ceil(plane.block_extent.width)
            || physical.rows != storage.extent.height.div_ceil(plane.block_extent.height)
            || physical.rect.x != 0
            || physical.rect.y != 0
            || physical.rect.width != physical.columns
            || physical.rect.height != physical.rows
        {
            return Err(VulkanError::InvalidResource(
                "overlapping or out-of-bounds image sample planes",
            ));
        }
        prior = storage.end().expect("checked sample extent");
    }
    Ok(())
}

impl VulkanResidency {
    pub fn image_view_lease(
        &self,
        id: ImageViewId,
    ) -> Result<Arc<crate::AllocatedImageView>, VulkanError> {
        self.image(id.image)?
            .native_views
            .get(id.view)
            .cloned()
            .ok_or(VulkanError::InvalidResource("stale image view"))
    }
}

#[cfg(test)]
mod tests;

// Compare physical placement without requiring the two mappings to share a
// logical origin. A view of a mip/layer commonly starts at a sliced mapping.
fn native_rect(
    plane: ImagePlane,
    samples: [u32; 2],
    rect: nixe_gpu::physical::ByteRect,
) -> vk::Rect2D {
    let x = rect.x * plane.block_extent.width / samples[0];
    let y = rect.y * plane.block_extent.height / samples[1];
    let end_x = ((rect.x + rect.width) * plane.block_extent.width)
        .div_ceil(samples[0])
        .min(plane.extent.width);
    let end_y = ((rect.y + rect.height) * plane.block_extent.height)
        .div_ceil(samples[1])
        .min(plane.extent.height);
    vk::Rect2D {
        offset: vk::Offset2D {
            x: x as i32,
            y: y as i32,
        },
        extent: vk::Extent2D {
            width: end_x - x,
            height: end_y - y,
        },
    }
}

fn same_plane_storage(
    a: &nixe_gpu::native::GuestImagePlane,
    b: &nixe_gpu::native::GuestImagePlane,
) -> bool {
    if a.aspect != b.aspect || a.row_pitch != b.row_pitch || a.bytes_per_block != b.bytes_per_block
    {
        return false;
    }
    if a.footprint == b.footprint {
        return true;
    }
    let (
        nixe_gpu::physical::WriteDomain::Image(mut x),
        nixe_gpu::physical::WriteDomain::Image(mut y),
    ) = (a.footprint.domain(), b.footprint.domain())
    else {
        return false;
    };
    x.base = 0;
    y.base = 0;
    x.native_base = 0;
    y.native_base = 0;
    if x != y || a.footprint.mapping().machine() != b.footprint.mapping().machine() {
        return false;
    }
    let ar = a.footprint.bounds();
    let br = b.footprint.bounds();
    if ar.end - ar.start != br.end - br.start {
        return false;
    }
    let aa = a.footprint.mapping().spans();
    let bb = b.footprint.mapping().spans();
    let mut i = aa.partition_point(|s| s.logical + s.bytes <= ar.start);
    let mut j = bb.partition_point(|s| s.logical + s.bytes <= br.start);
    let mut offset = 0;
    while offset < ar.end - ar.start {
        let (x, y) = (&aa[i], &bb[j]);
        let xs = ar.start + offset;
        let ys = br.start + offset;
        if x.allocation != y.allocation
            || x.start.get() + xs - x.logical != y.start.get() + ys - y.logical
        {
            return false;
        }
        offset += (x.logical + x.bytes - xs)
            .min(y.logical + y.bytes - ys)
            .min(ar.end - xs);
        if ar.start + offset == x.logical + x.bytes {
            i += 1;
        }
        if br.start + offset == y.logical + y.bytes {
            j += 1;
        }
    }
    true
}
