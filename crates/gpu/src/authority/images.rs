//! Native image synchronization retains geometry instead of expanding texels.
use super::*;
use crate::physical::{ByteRect, ImageOverlap, WriteDomain};

pub struct ImageCopy<R> {
    pub resource: R,
    pub native_base: u64,
    pub rect: ByteRect,
}
pub struct ImageChanges<R> {
    pub copies: Vec<ImageCopy<R>>,
    pub unowned: Vec<ByteRect>,
}
impl<R: Copy + Eq + Hash> Authority<R> {
    /// Newest native rectangles win. Unchanged regions stay in the consuming
    /// image; only a first import needs its uncovered RAM/undefined regions.
    /// Noncongruent storage or CPU byte holes require actual reinterpretation.
    /// https://github.com/eden-emulator/mirror/blob/master/src/video_core/texture_cache/texture_cache.h
    pub fn image_changes(
        &self,
        resource: R,
        footprint: &Footprint,
        since: u64,
    ) -> Result<Option<ImageChanges<R>>, MemoryError> {
        let WriteDomain::Image(image) = footprint.domain() else {
            return Ok(None);
        };
        if !footprint.mapping().is_injective() {
            return Ok(None);
        }
        let entry = self
            .resources
            .get(&resource)
            .ok_or(MemoryError::Invalid("native image is not registered"))?;
        let mut writers = Vec::new();
        for candidate in std::iter::once(&resource).chain(&entry.aliases) {
            let alias = &self.resources[candidate];
            if alias.modification <= since {
                continue;
            }
            for &id in &alias.writers {
                let writer = &self.writers[&id];
                if writer.sequence.get() > since {
                    writers.push(id);
                }
            }
        }
        writers.sort_unstable_by_key(|id| std::cmp::Reverse(self.writers[id].sequence));
        let mut remaining = vec![image.rect];
        let mut next = Vec::new();
        let mut copies = Vec::new();
        for id in writers {
            let writer = &self.writers[&id];
            if !footprint.native_cache_compatible(&writer.footprint) {
                continue;
            }
            let rect = match footprint.image_overlap(&writer.footprint) {
                ImageOverlap::Disjoint => continue,
                ImageOverlap::Rect(rect) => rect,
                ImageOverlap::Reinterpret => {
                    if !footprint.intersects(&writer.footprint) {
                        continue;
                    }
                    return Ok(None);
                }
            };
            let WriteDomain::Image(source) = writer.footprint.domain() else {
                unreachable!()
            };
            if writer.visible.is_some() || !writer.holes.is_empty() {
                crate::trace::detail_event(
                    "alias_byte_merge",
                    source.native_base,
                    writer.holes.len() as u64,
                    writer.visible.as_ref().map_or(0, |v| v.len()) as u64,
                );
                return Ok(None);
            }
            next.clear();
            for demand in remaining.drain(..) {
                if let Some(claimed) = intersection(demand, rect) {
                    copies.push(ImageCopy {
                        resource: writer.resource,
                        native_base: source.native_base,
                        rect: claimed,
                    });
                    subtract(demand, claimed, &mut next);
                } else {
                    next.push(demand);
                }
            }
            std::mem::swap(&mut remaining, &mut next);
            if remaining.is_empty() {
                break;
            }
        }
        Ok(Some(ImageChanges {
            copies,
            unowned: if since == 0 { remaining } else { Vec::new() },
        }))
    }
}
fn intersection(a: ByteRect, b: ByteRect) -> Option<ByteRect> {
    let x = a.x.max(b.x);
    let y = a.y.max(b.y);
    let end_x = (a.x + a.width).min(b.x + b.width);
    let end_y = (a.y + a.height).min(b.y + b.height);
    (x < end_x && y < end_y).then(|| ByteRect {
        x,
        y,
        width: end_x - x,
        height: end_y - y,
    })
}
fn subtract(a: ByteRect, b: ByteRect, out: &mut Vec<ByteRect>) {
    for (x, y, width, height) in [
        (a.x, a.y, a.width, b.y - a.y),
        (
            a.x,
            b.y + b.height,
            a.width,
            a.y + a.height - b.y - b.height,
        ),
        (a.x, b.y, b.x - a.x, b.height),
        (b.x + b.width, b.y, a.x + a.width - b.x - b.width, b.height),
    ] {
        if width != 0 && height != 0 {
            out.push(ByteRect {
                x,
                y,
                width,
                height,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physical::{ImageFootprint, ImageStorage};
    use nixe_memory::{AllocationRange, Machine};

    #[test]
    fn whole_native_observation_tracks_the_last_volume_slice() {
        let bytes = 32768;
        let machine = Machine::new(bytes).unwrap();
        let allocation = machine.account(bytes).unwrap().allocate(bytes).unwrap();
        let range = AllocationRange::new(allocation.pin(), 0, bytes).unwrap();
        let mapping = PhysicalMapping::new(std::slice::from_ref(&range)).unwrap();
        let planes: Vec<_> = (0..128)
            .map(|depth_slice| {
                Footprint::new(
                    mapping.clone(),
                    WriteDomain::Image(ImageFootprint {
                        depth_slice,
                        base: 0,
                        columns: 8,
                        rows: 8,
                        packed_bytes: 4,
                        aspect_offset: 0,
                        aspect_bytes: 4,
                        storage: ImageStorage::Pitch { row_pitch: 32 },
                        native_base: u64::from(depth_slice) * 256,
                        native_element_bytes: 4,
                        native_row_pitch: 32,
                        rect: ByteRect {
                            x: 0,
                            y: 0,
                            width: 8,
                            height: 8,
                        },
                    }),
                )
                .unwrap()
            })
            .collect();
        let mut authority = Authority::new(machine.identity());
        authority.register(1, planes.clone()).unwrap();
        authority.register(2, planes.clone()).unwrap();
        let own = authority.observe_native_resource(1).unwrap();
        let alias = authority.observe_native_resource(2).unwrap();
        let writer = authority.create_writer(1, planes[127].clone()).unwrap();
        authority.record_write(writer, NativeTick(1)).unwrap();
        assert!(!own.is_invalid());
        assert!(alias.is_invalid());
        authority
            .cpu_write(&[range.slice(126 * 256, 4).unwrap()])
            .unwrap();
        assert!(own.is_invalid());
    }

    #[test]
    fn native_versions_resolve_large_partial_aliases_as_rectangles() {
        let bytes = 16 * 1024 * 1024;
        let machine = Machine::new(bytes).unwrap();
        let allocation = machine.account(bytes).unwrap().allocate(bytes).unwrap();
        let mapping =
            PhysicalMapping::new(&[AllocationRange::new(allocation.pin(), 0, bytes).unwrap()])
                .unwrap();
        let full = ByteRect {
            x: 0,
            y: 0,
            width: 1920,
            height: 1080,
        };
        let make = |rect| {
            Footprint::new(
                mapping.clone(),
                WriteDomain::Image(ImageFootprint {
                    depth_slice: 0,
                    base: 0,
                    columns: 1920,
                    rows: 1080,
                    packed_bytes: 4,
                    aspect_offset: 0,
                    aspect_bytes: 4,
                    storage: ImageStorage::BlockLinear {
                        block_height_log2: 4,
                        block_depth_log2: 0,
                    },
                    native_base: 0,
                    native_element_bytes: 4,
                    native_row_pitch: 1920 * 4,
                    rect,
                }),
            )
            .unwrap()
        };
        let footprint = make(full);
        let patch = ByteRect {
            x: 17,
            y: 31,
            width: 113,
            height: 127,
        };
        let mut authority = Authority::new(machine.identity());
        for id in 1..=3 {
            authority.register(id, vec![footprint.clone()]).unwrap();
        }
        let writer = authority.create_writer(1, footprint.clone()).unwrap();
        authority.record_write(writer, NativeTick(1)).unwrap();
        let writer = authority.create_writer(2, make(patch)).unwrap();
        authority.record_write(writer, NativeTick(2)).unwrap();
        let changes = authority.image_changes(3, &footprint, 0).unwrap().unwrap();
        assert!(changes.unowned.is_empty());
        assert_eq!(changes.copies.len(), 5);
        assert_eq!(
            changes
                .copies
                .iter()
                .map(|copy| u64::from(copy.rect.width) * u64::from(copy.rect.height))
                .sum::<u64>(),
            1920 * 1080
        );
        assert_eq!(
            changes
                .copies
                .iter()
                .filter(|copy| copy.resource == 2)
                .map(|copy| copy.rect)
                .collect::<Vec<_>>(),
            [patch]
        );
        let delta = authority.image_changes(3, &footprint, 1).unwrap().unwrap();
        assert_eq!(delta.copies.len(), 1);
        assert_eq!(delta.copies[0].resource, 2);
        assert_eq!(delta.copies[0].rect, patch);
        assert!(delta.unowned.is_empty());
        assert!(
            authority
                .image_changes(3, &footprint, 2)
                .unwrap()
                .unwrap()
                .copies
                .is_empty()
        );
    }

    #[test]
    fn incompatible_heap_reuse_preserves_native_aliases_during_compaction() {
        let machine = Machine::new(4096).unwrap();
        let allocation = machine.account(4096).unwrap().allocate(4096).unwrap();
        let range = AllocationRange::new(allocation.pin(), 0, 4096).unwrap();
        let mapping = PhysicalMapping::new(std::slice::from_ref(&range)).unwrap();
        let domain = ImageFootprint {
            depth_slice: 0,
            base: 0,
            columns: 8,
            rows: 8,
            packed_bytes: 4,
            aspect_offset: 0,
            aspect_bytes: 4,
            storage: ImageStorage::Pitch { row_pitch: 32 },
            native_base: 0,
            native_element_bytes: 4,
            native_row_pitch: 32,
            rect: ByteRect {
                x: 0,
                y: 0,
                width: 8,
                height: 8,
            },
        };
        let full = Footprint::new(mapping.clone(), WriteDomain::Image(domain)).unwrap();
        let reused = Footprint::new(
            mapping,
            WriteDomain::Image(ImageFootprint {
                columns: 4,
                rows: 16,
                storage: ImageStorage::Pitch { row_pitch: 16 },
                native_row_pitch: 16,
                rect: ByteRect {
                    x: 0,
                    y: 0,
                    width: 4,
                    height: 16,
                },
                ..domain
            }),
        )
        .unwrap();
        let mut authority = Authority::new(machine.identity());
        authority.register(1, vec![full.clone()]).unwrap();
        authority.register(2, vec![reused.clone()]).unwrap();
        authority.register(3, vec![full.clone()]).unwrap();
        assert!(authority.resources_may_overlap(1, 2));
        assert!(authority.resources_may_overlap(2, 1));
        assert!(!authority.resources[&1].aliases.contains(&2));
        let first = authority.create_writer(1, full.clone()).unwrap();
        authority.record_write(first, NativeTick(1)).unwrap();
        let retained = authority.observe_native(3, full.clone()).unwrap();
        let second = authority.create_writer(2, reused).unwrap();
        authority.record_write(second, NativeTick(2)).unwrap();
        authority.normalize(1).unwrap();
        assert!(!retained.is_invalid());
        let changes = authority.image_changes(3, &full, 0).unwrap().unwrap();
        assert!(changes.unowned.is_empty());
        assert_eq!(changes.copies.len(), 1);
        assert_eq!(changes.copies[0].resource, 1);
        assert_eq!(changes.copies[0].rect, domain.rect);
        let visible = authority.resolve(&[range.slice(0, 256).unwrap()]).unwrap();
        assert!(
            visible
                .iter()
                .all(|part| part.producer.unwrap().resource == 2)
        );
    }

    #[test]
    fn alias_graph_releases_generational_identities_and_cpu_holes_require_merge() {
        let machine = Machine::new(4096).unwrap();
        let allocation = machine.account(4096).unwrap().allocate(4096).unwrap();
        let range = AllocationRange::new(allocation.pin(), 0, 4096).unwrap();
        let mapping = PhysicalMapping::new(std::slice::from_ref(&range)).unwrap();
        let full = Footprint::new(
            mapping,
            WriteDomain::Image(ImageFootprint {
                depth_slice: 0,
                base: 0,
                columns: 8,
                rows: 8,
                packed_bytes: 4,
                aspect_offset: 0,
                aspect_bytes: 4,
                storage: ImageStorage::Pitch { row_pitch: 32 },
                native_base: 0,
                native_element_bytes: 4,
                native_row_pitch: 32,
                rect: ByteRect {
                    x: 0,
                    y: 0,
                    width: 8,
                    height: 8,
                },
            }),
        )
        .unwrap();
        let mut authority = Authority::new(machine.identity());
        for id in 1..=3 {
            authority.register(id, vec![full.clone()]).unwrap();
        }
        authority.unregister(3).unwrap();
        assert_eq!(authority.resources[&1].aliases, [2]);
        assert!(!authority.resources_may_overlap(1, 3));
        assert!(!authority.resources_may_overlap(2, 3));
        authority.register(4, vec![full.clone()]).unwrap();
        let writer = authority.create_writer(1, full.clone()).unwrap();
        authority.record_write(writer, NativeTick(1)).unwrap();
        authority.cpu_write(&[range.slice(0, 4).unwrap()]).unwrap();
        assert!(authority.image_changes(4, &full, 0).unwrap().is_none());
    }
}
