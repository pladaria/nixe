//! Exact byte intervals are a demand result, not the storage of a byte layout.
use super::*;
use crate::residency::ImagePlane;
use ash::vk;
use std::{borrow::Cow, sync::OnceLock};

#[derive(Clone)]
struct ImageSpans {
    backing: CanonicalBackingRange,
    page: CanonicalPageId,
    plane: ImagePlane,
    rect: vk::Rect2D,
    meaningful: u32,
    page_size: u64,
    repeated: bool,
    full_page: OnceLock<bool>,
    logical_bounds: OnceLock<(u64, u64)>,
}
#[derive(Clone)]
enum Footprint {
    Explicit(Vec<Span>),
    Mapped(CanonicalBackingRange, CanonicalPageId),
    Image(ImageSpans),
}
#[derive(Clone)]
pub(in crate::residency) struct Spans {
    footprint: Footprint,
    pub lanes: u64,
}
impl From<Vec<Span>> for Spans {
    fn from(spans: Vec<Span>) -> Self {
        Self {
            footprint: Footprint::Explicit(spans),
            lanes: u64::MAX,
        }
    }
}
impl Spans {
    pub fn mapped(backing: CanonicalBackingRange, page: CanonicalPageId) -> Self {
        Self {
            footprint: Footprint::Mapped(backing, page),
            lanes: u64::MAX,
        }
    }
    fn image_spans(&self) -> Option<&ImageSpans> {
        match &self.footprint {
            Footprint::Image(image) => Some(image),
            _ => None,
        }
    }
    #[cfg(test)]
    fn image_spans_mut(&mut self) -> Option<&mut ImageSpans> {
        match &mut self.footprint {
            Footprint::Image(image) => Some(image),
            _ => None,
        }
    }
    pub fn image(
        backing: CanonicalBackingRange,
        page: CanonicalPageId,
        plane: ImagePlane,
        rect: vk::Rect2D,
        meaningful: u32,
        page_size: u64,
        repeated: bool,
    ) -> Self {
        let lanes = if plane
            .row_pitch
            .is_multiple_of(u64::from(plane.bytes_per_block))
        {
            backing.page_byte_lanes(page, plane.bytes_per_block, meaningful)
        } else {
            u64::MAX
        };
        Self {
            footprint: Footprint::Image(ImageSpans {
                backing,
                page,
                plane,
                rect,
                meaningful,
                page_size,
                repeated,
                full_page: OnceLock::new(),
                logical_bounds: OnceLock::new(),
            }),
            lanes,
        }
    }
    pub fn same_mapping(&self, other: &Self) -> bool {
        match (self.image_spans(), other.image_spans()) {
            (Some(a), Some(b)) => a.plane == b.plane && a.backing.same_byte_mapping(&b.backing),
            _ => false,
        }
    }
    /// Reject pages outside a rectangle using a retained logical bound before
    /// expanding the page's byte permutation. The bound may be conservative.
    pub fn clipped_image_rect(&self, rect: vk::Rect2D) -> Option<Self> {
        let image = self.image_spans().expect("image footprint");
        let p = image.plane;
        debug_assert_eq!(image.rect, crate::residency::transfers::full_rect(p));
        let (start, end) = *image.logical_bounds.get_or_init(|| {
            let mut bounds = (u64::MAX, 0);
            image
                .backing
                .visit_page_mappings(image.page, &mut |logical, _, size| {
                    bounds.0 = bounds.0.min(logical);
                    bounds.1 = bounds.1.max(logical + size);
                });
            bounds
        });
        let first = u64::from(rect.offset.y as u32 / p.block_extent.height) * p.row_pitch
            + u64::from(rect.offset.x as u32 / p.block_extent.width) * u64::from(p.bytes_per_block);
        let last = u64::from(
            (rect.offset.y as u32 + rect.extent.height).div_ceil(p.block_extent.height) - 1,
        ) * p.row_pitch
            + u64::from((rect.offset.x as u32 + rect.extent.width).div_ceil(p.block_extent.width))
                * u64::from(p.bytes_per_block);
        if start >= last || first >= end {
            return None;
        }
        let mut clipped = image.clone();
        clipped.rect = rect;
        clipped.full_page = OnceLock::new();
        Some(Self {
            footprint: Footprint::Image(clipped),
            lanes: self.lanes,
        })
    }
    pub fn covers(&self, other: &Self) -> bool {
        match (self.image_spans(), other.image_spans()) {
            (Some(a), Some(b)) => {
                // Prove physical coverage first. Comparing two complete backing
                // maps once per page makes reinterpretation quadratic in pages.
                (a.page == b.page && a.covers_page())
                    || (a.meaningful == b.meaningful
                        && self.same_mapping(other)
                        && a.rect.offset.x <= b.rect.offset.x
                        && a.rect.offset.y <= b.rect.offset.y
                        && a.rect.offset.x as u32 + a.rect.extent.width
                            >= b.rect.offset.x as u32 + b.rect.extent.width
                        && a.rect.offset.y as u32 + a.rect.extent.height
                            >= b.rect.offset.y as u32 + b.rect.extent.height)
            }
            _ => false,
        }
    }
    pub fn covers_physical_page(&self) -> bool {
        self.image_spans().is_some_and(ImageSpans::covers_page)
    }
    #[cfg(test)]
    pub fn is_compact(&self) -> bool {
        !matches!(self.footprint, Footprint::Explicit(_))
    }
    pub fn push(&mut self, span: Span) {
        let Footprint::Explicit(spans) = &mut self.footprint else {
            unreachable!("explicit buffer footprint")
        };
        spans.push(span);
    }
    pub fn sort_buffer(&mut self) {
        let Footprint::Explicit(spans) = &mut self.footprint else {
            unreachable!("explicit buffer footprint")
        };
        spans.sort_unstable_by_key(|span| (span.start, span.logical));
    }
    /// Exact intervals exist only for this demand; tiled storage stays compact.
    pub fn exact(&self) -> Cow<'_, [Span]> {
        match &self.footprint {
            Footprint::Explicit(spans) => Cow::Borrowed(spans),
            _ => Cow::Owned(self.expand()),
        }
    }
    pub fn exact_iter(&self) -> ExactSpans<'_> {
        match self.exact() {
            Cow::Owned(spans) => ExactSpans::Owned(spans.into_iter()),
            Cow::Borrowed(spans) => ExactSpans::Borrowed(spans.iter()),
        }
    }
    pub fn disjoint(&self, other: &Self) -> bool {
        self.lanes & other.lanes == 0
    }
    fn expand(&self) -> Vec<Span> {
        if let Footprint::Mapped(backing, page) = &self.footprint {
            let mut spans = Vec::new();
            backing.visit_page_mappings(*page, &mut |logical, start, size| {
                spans.push(Span {
                    start,
                    end: start + size,
                    logical,
                });
            });
            spans.sort_unstable_by_key(|s| (s.start, s.logical));
            return spans;
        }
        let Some(image) = self.image_spans() else {
            return Vec::new();
        };
        let p = image.plane;
        let first_row = image.rect.offset.y as u32 / p.block_extent.height;
        let end_row =
            (image.rect.offset.y as u32 + image.rect.extent.height).div_ceil(p.block_extent.height);
        let first_byte = u64::from(image.rect.offset.x as u32 / p.block_extent.width)
            * u64::from(p.bytes_per_block);
        let end_byte = u64::from(
            (image.rect.offset.x as u32 + image.rect.extent.width).div_ceil(p.block_extent.width),
        ) * u64::from(p.bytes_per_block);
        let mut spans = Vec::<Span>::new();
        image
            .backing
            .visit_page_mappings(image.page, &mut |logical, physical, size| {
                let mut at = logical;
                while at < logical + size {
                    let row = at / p.row_pitch;
                    let x = at % p.row_pitch;
                    if x < first_byte {
                        at += (logical + size - at)
                            .min(p.row_pitch - x)
                            .min(first_byte - x);
                        continue;
                    }
                    let byte = (x % u64::from(p.bytes_per_block)) as u32;
                    let length = (logical + size - at).min(p.row_pitch - x);
                    let length = if image.meaningful == p.bytes_per_block {
                        length
                    } else {
                        length.min(u64::from(p.bytes_per_block - byte))
                    };
                    if row >= u64::from(first_row)
                        && row < u64::from(end_row)
                        && x >= first_byte
                        && x < end_byte
                        && byte < image.meaningful
                    {
                        let length = length.min(end_byte - x);
                        let length = if image.meaningful == p.bytes_per_block {
                            length
                        } else {
                            length.min(u64::from(image.meaningful - byte))
                        };
                        spans.push(Span {
                            start: physical + at - logical,
                            end: physical + at - logical + length,
                            logical: p.offset + at,
                        });
                    }
                    at += length;
                }
            });
        spans.sort_unstable_by_key(|s| (s.start, s.logical));
        let mut used = 0;
        for i in 0..spans.len() {
            let span = spans[i];
            if used != 0
                && spans[used - 1].end == span.start
                && spans[used - 1].logical + spans[used - 1].end - spans[used - 1].start
                    == span.logical
            {
                spans[used - 1].end = span.end;
            } else {
                spans[used] = span;
                used += 1;
            }
        }
        spans.truncate(used);
        spans
    }
}
pub(in crate::residency) enum ExactSpans<'a> {
    Owned(std::vec::IntoIter<Span>),
    Borrowed(std::slice::Iter<'a, Span>),
}
impl Iterator for ExactSpans<'_> {
    type Item = Span;
    fn next(&mut self) -> Option<Span> {
        match self {
            Self::Owned(spans) => spans.next(),
            Self::Borrowed(spans) => spans.next().copied(),
        }
    }
}

impl ImageSpans {
    fn covers_page(&self) -> bool {
        *self.full_page.get_or_init(|| {
            let p = self.plane;
            if self.meaningful != p.bytes_per_block
                || self.rect != crate::residency::transfers::full_rect(p)
                || p.row_pitch != p.row_bytes()
                || self.backing.size() != p.row_bytes() * u64::from(p.rows())
                || self.repeated
            {
                return false;
            }
            // An injective projection containing page_size bytes covers the
            // entire physical page, regardless of texel size, sample grid or
            // aspect interpretation. Count runs once without expanding texels.
            let mut bytes = 0;
            self.backing
                .visit_page_mappings(self.page, &mut |_, _, size| bytes += size);
            bytes == self.page_size
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nixe_memory::{CanonicalAllocation, MemoryPermissions};

    #[test]
    fn full_physical_page_write_retires_materialized_foreign_producers() {
        use nixe_gpu::BackendInstanceId;
        let allocation = CanonicalAllocation::zeroed(256, 64).unwrap();
        let backing = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let plane = ImagePlane {
            offset: 0,
            row_pitch: 32,
            mip_level: 0,
            array_layer: 0,
            aspect: vk::ImageAspectFlags::COLOR,
            extent: vk::Extent3D {
                width: 8,
                height: 8,
                depth: 1,
            },
            block_extent: vk::Extent2D {
                width: 1,
                height: 1,
            },
            bytes_per_block: 4,
        };
        let spans = Arc::new(Spans::image(
            backing.clone(),
            backing.pages().next().unwrap().identity(),
            plane,
            crate::residency::transfers::full_rect(plane),
            4,
            64,
            false,
        ));
        let old = Resource::Buffer(BufferId {
            instance: BackendInstanceId::new(1),
            slot: 0,
            generation: 1,
        });
        let image = Resource::Image(ImageId {
            instance: BackendInstanceId::new(1),
            slot: 1,
            generation: 1,
        });
        let mut page = Page::default();
        for byte in 0..64 {
            CanonicalIndex::replace(
                &mut page,
                Authority {
                    start: byte,
                    end: byte + 1,
                    source: old,
                    offset: byte * 2,
                    point: DeviceVisibilityPoint::new(1),
                },
            );
        }
        page.record_image_write(image, &spans, DeviceVisibilityPoint::new(2));
        assert!(
            !page.has_source(old),
            "overwritten providers must be evictable"
        );
        assert!(
            page.writes.is_empty(),
            "full replacement retains compact coverage"
        );
        page.record_image_write(image, &spans, DeviceVisibilityPoint::new(3));
        assert_eq!(page.shared_writes.len(), 1);
        let actual = page.resolved_writes();
        assert_eq!(actual.len(), 1);
        assert_eq!(
            (actual[0].start, actual[0].end, actual[0].point.get()),
            (0, 64, 3)
        );
    }
    #[test]
    fn physical_full_page_coverage_supersedes_other_texel_interpretations() {
        let allocation = CanonicalAllocation::zeroed(256, 64).unwrap();
        let backing = allocation
            .backing_range(MemoryPermissions::READ_WRITE)
            .unwrap();
        let page = backing.pages().next().unwrap().identity();
        let make = |width, height, pitch, meaningful, partial| {
            let plane = ImagePlane {
                offset: 0,
                row_pitch: pitch,
                mip_level: 0,
                array_layer: 0,
                aspect: vk::ImageAspectFlags::COLOR,
                extent: vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                },
                block_extent: vk::Extent2D {
                    width: 1,
                    height: 1,
                },
                bytes_per_block: 4,
            };
            let mut rect = crate::residency::transfers::full_rect(plane);
            if partial {
                rect.extent.width /= 2;
            }
            Spans::image(backing.clone(), page, plane, rect, meaningful, 64, false)
        };
        let candidates = [
            make(8, 8, 32, 4, false),
            make(4, 16, 16, 3, false),
            make(4, 8, 32, 4, false),
            make(8, 8, 32, 4, true),
            make(4, 16, 16, 4, false),
        ];
        assert!(candidates[0].covers(&candidates[1]));
        assert!(candidates[4].covers(&candidates[0]));
        for candidate in [
            &candidates[0],
            &candidates[1],
            &candidates[2],
            &candidates[4],
        ] {
            let p = candidate.image_spans().unwrap().plane;
            for y in 0..p.extent.height {
                for x in 0..p.extent.width {
                    let rect = vk::Rect2D {
                        offset: vk::Offset2D {
                            x: x as i32,
                            y: y as i32,
                        },
                        extent: vk::Extent2D {
                            width: p.extent.width - x,
                            height: 1,
                        },
                    };
                    let mut direct = candidate.clone();
                    direct.image_spans_mut().unwrap().rect = rect;
                    let tuples = |spans: &[Span]| {
                        spans
                            .iter()
                            .map(|s| (s.start, s.end, s.logical))
                            .collect::<Vec<_>>()
                    };
                    let clipped = candidate.clipped_image_rect(rect);
                    assert_eq!(
                        clipped
                            .as_ref()
                            .map_or_else(Vec::new, |c| tuples(&c.exact())),
                        tuples(&direct.exact())
                    );
                }
            }
        }
        for a in &candidates {
            for b in &candidates {
                let mut actual = [false; 64];
                for span in a.exact().iter() {
                    actual[span.start as usize..span.end as usize].fill(true);
                }
                if a.covers(b) {
                    for span in b.exact().iter() {
                        assert!(
                            actual[span.start as usize..span.end as usize]
                                .iter()
                                .all(|b| *b)
                        );
                    }
                }
            }
        }
        // Repeated projections cannot prove full coverage by byte count.
        let mut repeated = candidates[0].clone();
        repeated.image_spans_mut().unwrap().repeated = true;
        repeated.image_spans_mut().unwrap().full_page = OnceLock::new();
        assert!(!repeated.image_spans().unwrap().covers_page());
    }
}
