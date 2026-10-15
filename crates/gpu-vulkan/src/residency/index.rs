//! One canonical index holds overlap lookup and exact byte authority.
mod spans;
use super::{BufferId, ImageId};
use nixe_memory::{CanonicalBackingRange, CanonicalPageId, DeviceVisibilityPoint};
use rustc_hash::FxHashMap;
pub(super) use spans::Spans;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) enum Resource {
    Buffer(BufferId),
    Image(ImageId),
}
/// An immutable image footprint is shared by indexing and producer authority.
#[derive(Clone, Copy, Debug)]
pub(super) struct Span {
    pub start: u64,
    pub end: u64,
    pub logical: u64,
}
pub(super) struct BindingGroup<'a> {
    pub resource: Resource,
    pub spans: &'a Arc<Spans>,
}
#[derive(Clone, Copy, Debug)]
pub(super) struct Binding {
    pub resource: Resource,
    pub start: u64,
    pub end: u64,
    pub offset: u64,
}
#[derive(Clone, Copy, Debug)]
pub(super) struct Authority {
    pub start: u64,
    pub end: u64,
    pub source: Resource,
    pub offset: u64,
    pub point: DeviceVisibilityPoint,
}
struct ImageWrite {
    source: Resource,
    spans: Arc<Spans>,
    point: DeviceVisibilityPoint,
}
#[derive(Default)]
pub(super) struct Page {
    /// Resource bindings, independent of producer changes.
    pub revision: u64,
    bindings: FxHashMap<Resource, Vec<Arc<Spans>>>,
    shared_writes: Vec<ImageWrite>,
    pub writes: Vec<Authority>,
    /// All byte producers belong to this image and share its latest actual use.
    pub image_epoch: Option<(Resource, DeviceVisibilityPoint)>,
}
impl Page {
    pub fn resources(&self) -> impl Iterator<Item = &Resource> {
        self.bindings.keys()
    }
    pub fn bindings_for(&self, resource: Resource) -> impl Iterator<Item = &Arc<Spans>> {
        self.bindings.get(&resource).into_iter().flatten()
    }
    pub fn binding_groups(&self) -> impl Iterator<Item = BindingGroup<'_>> {
        self.bindings.iter().flat_map(|(&resource, groups)| {
            groups
                .iter()
                .map(move |spans| BindingGroup { resource, spans })
        })
    }
    pub fn bindings(&self) -> impl Iterator<Item = Binding> + '_ {
        self.binding_groups().flat_map(|group| {
            group.spans.exact_iter().map(move |span| Binding {
                resource: group.resource,
                start: span.start,
                end: span.end,
                offset: span.logical,
            })
        })
    }
    pub fn has_writes(&self) -> bool {
        !self.writes.is_empty() || !self.shared_writes.is_empty()
    }
    pub fn has_source(&self, source: Resource) -> bool {
        self.shared_writes
            .iter()
            .any(|write| write.source == source)
            || self.writes.iter().any(|write| write.source == source)
    }
    pub fn has_complete_producer(&mut self, spans: &Spans) -> bool {
        if self
            .shared_writes
            .iter()
            .any(|write| write.spans.covers(spans))
        {
            return true;
        }
        self.materialize_writes();
        let mut first = 0;
        for span in spans.exact_iter() {
            while first < self.writes.len() && self.writes[first].end <= span.start {
                first += 1;
            }
            let mut cursor = span.start;
            for write in &self.writes[first..] {
                if write.start > cursor {
                    break;
                }
                cursor = cursor.max(write.end);
                if cursor >= span.end {
                    break;
                }
            }
            if cursor < span.end {
                return false;
            }
        }
        true
    }
    /// Native image producers remain ordered compact layers, including foreign
    /// images. Only a byte consumer or buffer write resolves their intersections.
    pub fn record_image_write(
        &mut self,
        source: Resource,
        spans: &Arc<Spans>,
        point: DeviceVisibilityPoint,
    ) {
        if spans.covers_physical_page() {
            // A complete physical overwrite retires exact buffer/foreign-image
            // authorities too. Keeping them made every subsequent draw scan
            // hundreds of already-superseded intervals on each attachment page.
            self.writes.clear();
            if let [write] = self.shared_writes.as_mut_slice()
                && Arc::ptr_eq(&write.spans, spans)
            {
                write.source = source;
                write.point = point;
            } else {
                self.shared_writes.clear();
                self.shared_writes.push(ImageWrite {
                    source,
                    spans: Arc::clone(spans),
                    point,
                });
            }
            self.image_epoch = Some((source, point));
            return;
        }
        if let Some((image, epoch)) = self.image_epoch
            && image != source
        {
            for write in &mut self.writes {
                write.point = epoch;
            }
            for write in &mut self.shared_writes {
                write.point = epoch;
            }
            self.image_epoch = None;
        }
        let prior = self
            .shared_writes
            .iter()
            .position(|old| Arc::ptr_eq(&old.spans, spans));
        if let Some(index) = prior {
            let reorder = self.shared_writes[index + 1..]
                .iter()
                .any(|old| !old.spans.disjoint(spans));
            if !reorder {
                self.shared_writes[index].source = source;
                self.shared_writes[index].point = point;
                if self.image_epoch.is_some_and(|(image, _)| image == source)
                    || (self.writes.iter().all(|w| w.source == source)
                        && self.shared_writes.iter().all(|w| w.source == source))
                {
                    self.image_epoch = Some((source, point));
                }
                return;
            }
            self.shared_writes.remove(index);
        }
        // A full write of an equivalent physical interpretation supersedes
        // that old layer, even when the guest destroyed/recreated its image.
        self.shared_writes.retain(|old| !spans.covers(&old.spans));
        // Bound dependency fan-in under repeated incompatible reinterpretation.
        // Resolving one physical page keeps memory and demand cost bounded.
        if self.shared_writes.len() >= 16 {
            self.materialize_writes();
        }
        self.shared_writes.push(ImageWrite {
            source,
            spans: Arc::clone(spans),
            point,
        });
        if self.writes.iter().all(|w| w.source == source)
            && self.shared_writes.iter().all(|w| w.source == source)
        {
            self.image_epoch = Some((source, point));
        }
    }
    pub fn materialize_writes(&mut self) {
        // Resolve the lazy epoch once. Keeping it after expansion revisits
        // every packed byte interval for each later fragment of the same page.
        let epoch = self.image_epoch.take();
        if let Some((_, point)) = epoch {
            for write in &mut self.writes {
                write.point = point;
            }
        }
        for write in self.shared_writes.drain(..) {
            let point = epoch.map_or(write.point, |(_, point)| point);
            CanonicalIndex::overlay_sorted(
                &mut self.writes,
                write.spans.exact_iter().map(|span| Authority {
                    source: write.source,
                    point,
                    start: span.start,
                    end: span.end,
                    offset: span.logical,
                }),
            );
        }
    }
    pub fn resolved_writes(&self) -> Vec<Authority> {
        let mut writes = self.writes.clone();
        if let Some((_, point)) = self.image_epoch {
            for write in &mut writes {
                write.point = point;
            }
        }
        for write in &self.shared_writes {
            let point = self.image_epoch.map_or(write.point, |(_, point)| point);
            CanonicalIndex::overlay_sorted(
                &mut writes,
                write.spans.exact_iter().map(|span| Authority {
                    source: write.source,
                    point,
                    start: span.start,
                    end: span.end,
                    offset: span.logical,
                }),
            );
        }
        writes
    }
    pub fn clear_writes(&mut self) {
        self.writes.clear();
        self.shared_writes.clear();
        self.image_epoch = None;
    }
    pub fn acknowledge(&mut self, point: DeviceVisibilityPoint) {
        if let Some((_, written)) = self.image_epoch {
            if written <= point {
                self.clear_writes();
            }
        } else {
            self.writes.retain(|w| w.point > point);
            self.shared_writes.retain(|w| w.point > point);
        }
    }
    pub fn has_unsubmitted_write(&self, submitted: u64) -> bool {
        self.image_epoch.map_or_else(
            || {
                self.writes.iter().any(|w| w.point.get() > submitted)
                    || self.shared_writes.iter().any(|w| w.point.get() > submitted)
            },
            |(_, point)| point.get() > submitted,
        )
    }
    #[cfg(test)]
    pub fn write_layer_count(&self) -> usize {
        self.shared_writes.len()
    }
}
#[derive(Default)]
pub(super) struct CanonicalIndex {
    pub pages: FxHashMap<CanonicalPageId, Page>,
}
impl CanonicalIndex {
    pub fn insert(&mut self, resource: Resource, range: &CanonicalBackingRange) {
        let mut offset = 0;
        for segment in range.segments() {
            self.insert_span(
                resource,
                segment.page(),
                segment.offset(),
                segment.offset() + segment.size(),
                offset,
            );
            offset += segment.size();
        }
        for backing in range.pages() {
            if let Some(groups) = self
                .pages
                .get_mut(&backing.identity())
                .and_then(|page| page.bindings.get_mut(&resource))
            {
                for spans in groups {
                    Arc::make_mut(spans).sort_buffer();
                }
            }
        }
    }
    pub fn insert_span(
        &mut self,
        resource: Resource,
        page: CanonicalPageId,
        start: u64,
        end: u64,
        offset: u64,
    ) {
        let page = self.register(page);
        let groups = page.bindings.entry(resource).or_default();
        if groups.is_empty() {
            groups.push(Arc::new(Vec::new().into()));
        }
        Arc::make_mut(&mut groups[0]).push(Span {
            start,
            end,
            logical: offset,
        });
    }
    fn register(&mut self, page: CanonicalPageId) -> &mut Page {
        let page = self.pages.entry(page).or_default();
        page.revision = page
            .revision
            .checked_add(1)
            .expect("canonical page binding revision exhausted");
        page
    }
    pub fn insert_image_spans(
        &mut self,
        resource: Resource,
        page: CanonicalPageId,
        spans: Arc<Spans>,
    ) {
        self.register(page)
            .bindings
            .entry(resource)
            .or_default()
            .push(spans);
    }
    pub fn remove(&mut self, resource: Resource, range: &CanonicalBackingRange) {
        for backing in range.pages() {
            if let Some(page) = self.pages.get_mut(&backing.identity()) {
                page.revision = page
                    .revision
                    .checked_add(1)
                    .expect("canonical page binding revision exhausted");
                page.bindings.remove(&resource);
                debug_assert!(!page.has_source(resource));
                if page.bindings.is_empty() && !page.has_writes() {
                    self.pages.remove(&backing.identity());
                }
            }
        }
    }
    /// Overlay a sorted non-overlapping write footprint in one linear sweep.
    /// Swizzled depth planes have thousands of spans per page; replacing each
    /// span separately would repeatedly rebuild the same authority list.
    fn overlay_sorted(writes: &mut Vec<Authority>, replacements: impl Iterator<Item = Authority>) {
        let mut old = std::mem::take(writes).into_iter();
        let mut next = old.next();
        for replacement in replacements {
            while let Some(write) = next {
                if write.end <= replacement.start {
                    writes.push(write);
                    next = old.next();
                    continue;
                }
                if write.start >= replacement.end {
                    break;
                }
                if write.start < replacement.start {
                    writes.push(Authority {
                        end: replacement.start,
                        ..write
                    });
                }
                if write.end > replacement.end {
                    next = Some(Authority {
                        start: replacement.end,
                        offset: write.offset + replacement.end - write.start,
                        ..write
                    });
                    break;
                }
                next = old.next();
            }
            writes.push(replacement);
        }
        if let Some(write) = next {
            writes.push(write);
        }
        writes.extend(old);
        let mut used = 0;
        for index in 0..writes.len() {
            let write = writes[index];
            if used > 0
                && writes[used - 1].end == write.start
                && writes[used - 1].source == write.source
                && writes[used - 1].point == write.point
                && writes[used - 1].offset + writes[used - 1].end - writes[used - 1].start
                    == write.offset
            {
                writes[used - 1].end = write.end;
            } else {
                writes[used] = write;
                used += 1;
            }
        }
        writes.truncate(used);
    }
    /// Ordered repeated physical offsets overwrite only their intersection.
    pub fn replace(page: &mut Page, replacement: Authority) {
        page.materialize_writes();
        let first = page
            .writes
            .partition_point(|write| write.end <= replacement.start);
        if let Some(old) = page.writes.get_mut(first)
            && old.start == replacement.start
            && old.end == replacement.end
            && old.source == replacement.source
            && old.offset == replacement.offset
        {
            old.point = replacement.point;
            return;
        }
        let mut writes = Vec::with_capacity(page.writes.len() + 2);
        for write in page.writes.drain(..) {
            if write.end <= replacement.start || replacement.end <= write.start {
                writes.push(write);
                continue;
            }
            if write.start < replacement.start {
                writes.push(Authority {
                    end: replacement.start,
                    ..write
                });
            }
            if replacement.end < write.end {
                writes.push(Authority {
                    start: replacement.end,
                    offset: write.offset + replacement.end - write.start,
                    ..write
                });
            }
        }
        writes.push(replacement);
        writes.sort_unstable_by_key(|write| write.start);
        // Coalesce only contiguous bytes from the same exact producer/version.
        for write in writes {
            if let Some(last) = page.writes.last_mut()
                && last.end == write.start
                && last.source == write.source
                && last.point == write.point
                && last.offset + last.end - last.start == write.offset
            {
                last.end = write.end;
            } else {
                page.writes.push(write);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nixe_gpu::BackendInstanceId;
    fn source(slot: usize) -> Resource {
        Resource::Buffer(BufferId {
            instance: BackendInstanceId::new(1),
            slot,
            generation: 1,
        })
    }
    #[test]
    fn complete_producer_proof_rejects_holes_and_combines_adjacent_writers() {
        let spans = Spans::from(vec![Span {
            start: 2,
            end: 14,
            logical: 0,
        }]);
        let mut page = Page::default();
        let write = |start, end| Authority {
            start,
            end,
            source: source(start as usize),
            offset: 0,
            point: DeviceVisibilityPoint::new(1),
        };
        assert!(!page.has_complete_producer(&spans));
        page.writes = vec![write(0, 8), write(9, 16)];
        assert!(!page.has_complete_producer(&spans));
        page.writes = vec![write(0, 8), write(8, 16)];
        assert!(page.has_complete_producer(&spans));
        page.acknowledge(DeviceVisibilityPoint::new(1));
        assert!(!page.has_complete_producer(&spans));
    }
    #[test]
    fn shared_aspects_keep_holes_and_latest_epoch_before_foreign_partial_write() {
        let image = Resource::Image(ImageId {
            instance: BackendInstanceId::new(1),
            slot: 0,
            generation: 1,
        });
        let depth = Arc::new(Spans::from(vec![
            Span {
                start: 1,
                end: 4,
                logical: 0,
            },
            Span {
                start: 5,
                end: 8,
                logical: 4,
            },
        ]));
        let stencil = Arc::new(Spans::from(vec![
            Span {
                start: 0,
                end: 1,
                logical: 8,
            },
            Span {
                start: 4,
                end: 5,
                logical: 9,
            },
        ]));
        let mut page = Page::default();
        page.record_image_write(image, &depth, DeviceVisibilityPoint::new(1));
        assert_eq!(
            page.resolved_writes()
                .iter()
                .map(|w| (w.start, w.end))
                .collect::<Vec<_>>(),
            vec![(1, 4), (5, 8)]
        );
        page.record_image_write(image, &stencil, DeviceVisibilityPoint::new(2));
        page.record_image_write(image, &depth, DeviceVisibilityPoint::new(3));
        assert!(page.writes.is_empty());
        assert_eq!(page.shared_writes.len(), 2);
        assert!(page.has_source(image));
        page.acknowledge(DeviceVisibilityPoint::new(2));
        assert!(page.has_unsubmitted_write(2));
        assert!(page.resolved_writes().iter().all(|w| w.point.get() == 3));
        CanonicalIndex::replace(
            &mut page,
            Authority {
                start: 2,
                end: 6,
                source: source(1),
                offset: 100,
                point: DeviceVisibilityPoint::new(4),
            },
        );
        assert!(page.shared_writes.is_empty());
        assert_eq!(
            page.resolved_writes()
                .iter()
                .map(|w| (w.start, w.end, w.source, w.offset, w.point.get()))
                .collect::<Vec<_>>(),
            vec![
                (0, 1, image, 8, 3),
                (1, 2, image, 0, 3),
                (2, 6, source(1), 100, 4),
                (6, 8, image, 5, 3)
            ]
        );
        page.record_image_write(image, &depth, DeviceVisibilityPoint::new(5));
        page.acknowledge(DeviceVisibilityPoint::new(3));
        assert!(page.has_source(image));
        let writes = page.resolved_writes();
        assert_eq!(
            writes
                .iter()
                .map(|w| (w.start, w.end, w.source))
                .collect::<Vec<_>>(),
            vec![(1, 4, image), (4, 5, source(1)), (5, 8, image)]
        );
        assert!(page.has_source(source(1)));
        page.clear_writes();
        assert!(!page.has_writes());
    }

    #[test]
    fn foreign_image_layers_resolve_last_writer_without_expanding_on_publication() {
        let mut page = Page::default();
        let mut expected = [None; 32];
        let footprints: Vec<_> = (0..24)
            .map(|index| {
                Arc::new(Spans::from(vec![Span {
                    start: index % 17,
                    end: index % 17 + 8,
                    logical: 100 + index * 16,
                }]))
            })
            .collect();
        for tick in 1..=100u64 {
            let spans = &footprints[(tick as usize * 7) % footprints.len()];
            let image = Resource::Image(ImageId {
                instance: BackendInstanceId::new(1),
                slot: tick as usize % 3,
                generation: 1,
            });
            page.record_image_write(image, spans, DeviceVisibilityPoint::new(tick));
            for span in spans.exact_iter() {
                for byte in span.start..span.end {
                    expected[byte as usize] = Some((image, span.logical + byte - span.start));
                }
            }
            let actual = page.resolved_writes();
            for byte in 0..32u64 {
                let actual = actual
                    .iter()
                    .find(|w| w.start <= byte && byte < w.end)
                    .map(|w| (w.source, w.offset + byte - w.start));
                assert_eq!(actual, expected[byte as usize], "tick={tick}, byte={byte}");
            }
            assert!(page.shared_writes.len() <= 16);
        }
        let current = page.resolved_writes();
        page.acknowledge(DeviceVisibilityPoint::new(0));
        assert_eq!(
            page.resolved_writes()
                .iter()
                .map(|w| (w.start, w.end, w.source, w.offset, w.point))
                .collect::<Vec<_>>(),
            current
                .iter()
                .map(|w| (w.start, w.end, w.source, w.offset, w.point))
                .collect::<Vec<_>>()
        );
        page.materialize_writes();
        assert!(page.shared_writes.is_empty());
    }
    #[test]
    fn image_epoch_survives_old_ack_and_expands_before_partial_foreign_write() {
        let a = Resource::Image(ImageId {
            instance: BackendInstanceId::new(1),
            slot: 0,
            generation: 1,
        });
        let b = source(1);
        let mut page = Page {
            writes: vec![
                Authority {
                    start: 0,
                    end: 4,
                    source: a,
                    offset: 10,
                    point: DeviceVisibilityPoint::new(1),
                },
                Authority {
                    start: 8,
                    end: 12,
                    source: a,
                    offset: 18,
                    point: DeviceVisibilityPoint::new(1),
                },
            ],
            image_epoch: Some((a, DeviceVisibilityPoint::new(3))),
            ..Default::default()
        };
        assert!(page.has_unsubmitted_write(2));
        page.materialize_writes();
        assert!(page.image_epoch.is_none());
        assert!(page.writes.iter().all(|write| write.point.get() == 3));
        page.materialize_writes();
        page.acknowledge(DeviceVisibilityPoint::new(2));
        assert_eq!(
            page.resolved_writes()
                .iter()
                .map(|w| w.point.get())
                .collect::<Vec<_>>(),
            vec![3, 3]
        );
        CanonicalIndex::replace(
            &mut page,
            Authority {
                start: 2,
                end: 10,
                source: b,
                offset: 20,
                point: DeviceVisibilityPoint::new(4),
            },
        );
        assert_eq!(
            page.writes
                .iter()
                .map(|w| (w.start, w.end, w.source, w.offset, w.point.get()))
                .collect::<Vec<_>>(),
            vec![(0, 2, a, 10, 3), (2, 10, b, 20, 4), (10, 12, a, 20, 3)]
        );
        assert!(page.image_epoch.is_none());
        page.acknowledge(DeviceVisibilityPoint::new(3));
        assert_eq!(page.writes.len(), 1);
        assert_eq!(page.writes[0].source, b);
        page.acknowledge(DeviceVisibilityPoint::new(4));
        assert!(page.writes.is_empty());
    }
}
