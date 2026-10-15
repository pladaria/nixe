//! Prepared physical write coverage. Guest layout permutations are immutable;
//! publication updates producer epochs without reconstructing each texel range.
use super::index::Spans;
use super::*;
use rustc_hash::FxHashMap;
struct PagePlan {
    page: CanonicalPageId,
    spans: Arc<Spans>,
}
pub(super) struct ImageWritePlan {
    ownership: CanonicalBackingRange,
    pages: Box<[PagePlan]>,
    logical: Box<[(u64, u64)]>,
    repeated: bool,
    aliases: Option<AliasCoverage>,
    queued: Option<WorkTicket>,
}
type AliasCoverage = (Box<[u64]>, Box<[AliasWindows]>);
struct AliasWindows {
    resource: Resource,
    windows: Box<[(u64, u64)]>,
}
impl VulkanResidency {
    /// Bindings and producer footprints describe the same meaningful native
    /// bytes. Build their immutable physical mapping once and retain it for
    /// full-plane clears/draws instead of legalizing packed aspects twice.
    pub(super) fn index_image_planes(
        &mut self,
        id: ImageId,
        planes: &[ImagePlane],
    ) -> Result<(), VulkanError> {
        for &plane in planes {
            let rect = transfers::full_rect(plane);
            let plan = self.write_plan(id, plane, rect)?;
            for p in &plan.pages {
                self.index
                    .insert_image_spans(Resource::Image(id), p.page, Arc::clone(&p.spans));
            }
            self.retain_write_plan(id, plane, rect, plan);
        }
        Ok(())
    }
    fn write_plan(
        &mut self,
        id: ImageId,
        plane: ImagePlane,
        rect: vk::Rect2D,
    ) -> Result<Box<ImageWritePlan>, VulkanError> {
        if let Some(index) = self
            .image(id)?
            .write_plans
            .iter()
            .position(|(p, r, _)| *p == plane && *r == rect)
        {
            return Ok(self.images[id.slot]
                .entry
                .as_mut()
                .unwrap()
                .write_plans
                .swap_remove(index)
                .2);
        }
        let storage_plane = self.image(id)?.storage_plane(plane);
        let storage_rect = self.image(id)?.storage_rect(rect);
        let logical = if rect == transfers::full_rect(plane) {
            // Native image contents contain no pitch padding or gaps between
            // planes. A full-plane producer validates that whole partition;
            // exact canonical authority below still excludes all padding.
            let entry = self.image(id)?;
            let planes = entry.decoded.as_ref().map_or(&entry.planes, |d| &d.planes);
            let index = planes
                .iter()
                .position(|p| *p == plane)
                .expect("image plane");
            let start = if index == 0 { 0 } else { plane.offset };
            let end = planes
                .get(index + 1)
                .map_or(entry.contents.backing.size(), |p| p.offset);
            vec![(start, end - start)]
        } else {
            Self::image_rect_bytes(storage_plane, storage_rect)
        };
        let backing = self
            .image(id)?
            .contents
            .backing
            .logical_view(plane.offset, storage_plane.end().unwrap() - plane.offset)
            .map_err(memory_error)?;
        let meaningful = if plane.aspect == vk::ImageAspectFlags::DEPTH
            && matches!(
                self.image(id)?.description.format,
                vk::Format::D24_UNORM_S8_UINT | vk::Format::X8_D24_UNORM_PACK32
            ) {
            3
        } else {
            plane.bytes_per_block
        };
        let repeated = backing.may_repeat_bytes();
        let mut touches = Vec::new();
        let partial = rect != transfers::full_rect(plane);
        let pages = if partial {
            let full = self
                .image(id)?
                .write_plans
                .iter()
                .find(|(p, r, _)| *p == plane && *r == transfers::full_rect(plane))
                .expect("full-plane indexing retains its immutable footprint");
            full.2
                .pages
                .iter()
                .filter_map(|page| {
                    let spans = Arc::new(page.spans.clipped_image_rect(storage_rect)?);
                    // The conservative page bound avoids expanding every page of
                    // a large texture for each small upload stripe. Exact coverage
                    // still decides ownership and excludes untouched bytes.
                    let first = spans.exact().first().copied()?;
                    touches.extend_from_slice(
                        backing
                            .snapshot_subrange(first.logical - plane.offset, 1)
                            .expect("validated image footprint")
                            .segments(),
                    );
                    Some(PagePlan {
                        page: page.page,
                        spans,
                    })
                })
                .collect()
        } else {
            backing
                .pages()
                .map(|page| PagePlan {
                    page: page.identity(),
                    spans: Arc::new(Spans::image(
                        backing.clone(),
                        page.identity(),
                        storage_plane,
                        storage_rect,
                        meaningful,
                        page.size() as u64,
                        repeated,
                    )),
                })
                .collect()
        };
        // Visibility is page-granular; exact authority remains in page spans.
        let ownership = if partial {
            CanonicalBackingRange::new(touches).map_err(memory_error)?
        } else {
            backing
        };
        let plan = Box::new(ImageWritePlan {
            ownership,
            pages,
            logical: logical.into_boxed_slice(),
            repeated,
            aliases: None,
            queued: None,
        });
        Ok(plan)
    }
    fn retain_write_plan(
        &mut self,
        id: ImageId,
        plane: ImagePlane,
        rect: vk::Rect2D,
        plan: Box<ImageWritePlan>,
    ) {
        let plans = &mut self.images[id.slot].entry.as_mut().unwrap().write_plans;
        // Full-plane footprints are also the index's immutable mapping basis.
        // Bound only partial rectangles; evicting a full plane would repeatedly
        // rebuild its page bounds during a long sequence of striped uploads.
        if plans
            .iter()
            .filter(|(p, r, _)| *r != transfers::full_rect(*p))
            .count()
            >= 32
            && let Some(oldest) = plans
                .iter()
                .position(|(p, r, _)| *r != transfers::full_rect(*p))
        {
            let _ = plans.remove(oldest);
        }
        plans.push((plane, rect, plan));
    }
    pub(super) fn prepare_image_write(
        &mut self,
        id: ImageId,
        plane: ImagePlane,
        rect: vk::Rect2D,
    ) -> Result<(), VulkanError> {
        let plan = self.write_plan(id, plane, rect)?;
        let tick = self.device.work_ticket()?.tick();
        let declaration = DeviceAccessDeclaration::write(
            NonCpuDeviceId::new(self.instance.get()),
            DeviceVisibilityPoint::new(tick),
            DeviceVisibilityPoint::new(tick),
        )
        .map_err(memory_error)?;
        let result = CanonicalBackingRange::prepare_resident_device_accesses(
            [(&plan.ownership, declaration)],
            self.visibility.clone(),
        )
        .map_err(memory_error);
        self.retain_write_plan(id, plane, rect, plan);
        result
    }
    pub(super) fn publish_image_write(
        &mut self,
        id: ImageId,
        plane: ImagePlane,
        rect: vk::Rect2D,
    ) -> Result<(), VulkanError> {
        let mut plan = self.write_plan(id, plane, rect)?;
        if plan.repeated {
            for (offset, size) in self.canonical_image_rect_bytes(id, plane, rect)? {
                self.record_authority(Resource::Image(id), offset, size)?;
            }
            self.retain_write_plan(id, plane, rect, plan);
            return Ok(());
        }
        let work = self.device.work_ticket()?;
        let point = DeviceVisibilityPoint::new(work.tick());
        let source = Resource::Image(id);
        // Layout changes invalidate alias coverage; producer epochs do not.
        // This metadata is owned by the recorder and never rebuilt per unchanged draw.
        if plan.aliases.as_ref().is_none_or(|(revisions, _)| {
            revisions
                .iter()
                .zip(&plan.pages)
                .any(|(revision, p)| *revision != self.index.pages[&p.page].revision)
        }) {
            let mut aliases = FxHashMap::<Resource, Vec<(u64, u64)>>::default();
            for p in plan.pages.iter() {
                let page = self.index.pages.get(&p.page).expect("indexed write plan");
                for group in page.binding_groups() {
                    if (group.resource == source
                        && (Arc::ptr_eq(group.spans, &p.spans)
                            || group.spans.same_mapping(&p.spans)))
                        || group.spans.disjoint(&p.spans)
                    {
                        continue;
                    }
                    // Dirtiness can be conservative; producer authority cannot.
                    // Shared physical pages identify candidate aliases in O(1).
                    // Exact byte maps are needed only when an alias is consumed.
                    if group.resource != source {
                        let windows = aliases.entry(group.resource).or_default();
                        if matches!(group.resource, Resource::Buffer(_)) {
                            // Buffer bindings already contain contiguous physical
                            // runs. Invalidate their intersections with this page,
                            // without expanding the image's texel permutation or
                            // invalidating unrelated pages of a large buffer.
                            windows.extend(
                                group
                                    .spans
                                    .exact_iter()
                                    .map(|span| (span.logical, span.end - span.start)),
                            );
                        }
                        continue;
                    }
                    // Different subresources of this same image can map to
                    // the same physical bytes. Invalidate only the other
                    // logical interpretation, preserving the current producer.
                    let exact = p.spans.exact();
                    let mut first = 0;
                    for binding in group.spans.exact_iter() {
                        while first < exact.len() && exact[first].end <= binding.start {
                            first += 1;
                        }
                        for span in &exact[first..] {
                            if span.start >= binding.end {
                                break;
                            }
                            let begin = span.start.max(binding.start);
                            let end = span.end.min(binding.end);
                            if begin < end
                                && binding.logical + begin - binding.start
                                    != span.logical + begin - span.start
                            {
                                aliases.entry(source).or_default().push(
                                    self.image_invalidation_window(
                                        id,
                                        binding.logical + begin - binding.start,
                                        end - begin,
                                    ),
                                );
                            }
                        }
                    }
                }
            }
            plan.aliases = Some((
                plan.pages
                    .iter()
                    .map(|p| self.index.pages[&p.page].revision)
                    .collect(),
                aliases
                    .into_iter()
                    .map(|(resource, mut windows)| {
                        // Buffer dirtiness is conservative; authority remains byte-exact.
                        // One invalid interval avoids a growing list per packed depth texel.
                        if resource != source && matches!(resource, Resource::Image(_)) {
                            windows.push((0, self.contents(resource).unwrap().backing.size()));
                        } else {
                            normalize(&mut windows);
                        }
                        AliasWindows {
                            resource,
                            windows: windows.into_boxed_slice(),
                        }
                    })
                    .collect(),
            ));
        }
        for alias in plan.aliases.as_ref().unwrap().1.iter() {
            let contents = self.contents_mut(alias.resource);
            if contents.invalid.as_slice() == alias.windows.as_ref() {
                continue;
            }
            contents.invalidate_intervals(&alias.windows);
        }
        for p in plan.pages.iter() {
            self.index
                .pages
                .get_mut(&p.page)
                .unwrap()
                .record_image_write(source, &p.spans, point);
        }

        // One accepted recording tick needs this immutable coverage once.
        // Exact producer authority above still changes on every write. A CPU
        // demand necessarily flushes the recorder before materializing bytes,
        // so a following writer has a new ticket and publishes ownership again.
        if plan.queued != Some(work) {
            self.pending_writes.push(plan.ownership.clone());
            plan.queued = Some(work);
        }
        let entry = self.images[id.slot].entry.as_mut().unwrap();
        entry.contents.last_use = Some(work);
        if !entry.contents.invalid.is_empty() {
            entry.contents.validate_intervals(&plan.logical);
        }
        self.retain_write_plan(id, plane, rect, plan);
        Ok(())
    }
}
