//! Neutral allocation identities and retained canonical backing views.

use std::{
    fmt::{Display, Formatter},
    sync::Arc,
};

use nixe_memory::{CanonicalBackingRange, CanonicalBackingSegment, CanonicalPageId};

/// Backend-independent identity of one logical GPU backing allocation.
///
/// This identity is not an `nvmap` handle, guest address, host pointer, or
/// concrete graphics-API object. Its owner assigns it and controls its
/// lifetime.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct GpuAllocationId(u64);

impl GpuAllocationId {
    /// Creates an identity from a value assigned by the allocation owner.
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the owner-assigned numeric representation.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl Display for GpuAllocationId {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "gpu-allocation=0x{:016x}", self.0)
    }
}

/// Immutable size and alignment requirements of a logical GPU allocation.
///
/// Canonical guest bytes are deliberately absent. A resource view attaches a
/// checked [`CanonicalBackingRange`] to an allocation only through
/// [`BackingView`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuAllocationDescription {
    size: u64,
    alignment: u64,
}

impl GpuAllocationDescription {
    /// Creates a non-empty allocation description with power-of-two alignment.
    pub const fn new(size: u64, alignment: u64) -> Result<Self, AllocationDescriptionError> {
        if size == 0 {
            return Err(AllocationDescriptionError::Empty);
        }
        if !alignment.is_power_of_two() {
            return Err(AllocationDescriptionError::InvalidAlignment { alignment });
        }
        Ok(Self { size, alignment })
    }

    /// Returns the allocation's logical byte size.
    #[must_use]
    pub const fn size(self) -> u64 {
        self.size
    }

    /// Returns the required byte alignment.
    #[must_use]
    pub const fn alignment(self) -> u64 {
        self.alignment
    }
}

/// Failure to describe a logical GPU allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AllocationDescriptionError {
    /// The allocation has no bytes.
    Empty,
    /// The alignment is zero or is not a power of two.
    InvalidAlignment { alignment: u64 },
}

impl Display for AllocationDescriptionError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => formatter.write_str("GPU allocation size is zero"),
            Self::InvalidAlignment { alignment } => write!(
                formatter,
                "GPU allocation alignment is not a non-zero power of two: alignment={alignment:#x}"
            ),
        }
    }
}

impl std::error::Error for AllocationDescriptionError {}

/// A retained canonical byte range within one logical GPU allocation.
///
/// The range owns canonical page references and remains pointer-free. It does
/// not give the resource a guest CPU or GPU virtual address. Constructors also
/// reject a range which names the same canonical bytes more than once; aliases
/// are represented by distinct views instead.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BackingView {
    allocation: GpuAllocationId,
    allocation_offset: u64,
    range: CanonicalBackingRange,
    canonical_spans: Arc<[CanonicalBackingSpan]>,
}

/// One maximal run of canonical bytes retained by a backing view.
///
/// Consecutive fully covered pages share one span. Boundary offsets preserve
/// exact byte identity for partial first and last pages.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CanonicalBackingSpan {
    first_page: CanonicalPageId,
    last_page: CanonicalPageId,
    first_offset: u64,
    last_end: u64,
    ends_at_page_boundary: bool,
}

impl CanonicalBackingSpan {
    /// Returns the first page covered by this span.
    #[must_use]
    pub const fn first_page(self) -> CanonicalPageId {
        self.first_page
    }

    /// Returns the last page covered by this span.
    #[must_use]
    pub const fn last_page(self) -> CanonicalPageId {
        self.last_page
    }

    /// Returns the first covered byte within [`Self::first_page`].
    #[must_use]
    pub const fn first_offset(self) -> u64 {
        self.first_offset
    }

    /// Returns the exclusive final covered byte within [`Self::last_page`].
    #[must_use]
    pub const fn last_end(self) -> u64 {
        self.last_end
    }

    /// Returns whether this span can no longer overlap a later sorted span.
    #[must_use]
    pub fn ends_before(self, page: CanonicalPageId, offset: u64) -> bool {
        self.last_page < page || (self.last_page == page && self.last_end <= offset)
    }

    /// Returns whether both spans contain any of the same canonical bytes.
    #[must_use]
    pub fn overlaps(self, other: Self) -> bool {
        if self.first_page.store() != other.first_page.store()
            || self.last_page < other.first_page
            || other.last_page < self.first_page
        {
            return false;
        }

        let page = self.first_page.max(other.first_page);
        let last_page = self.last_page.min(other.last_page);
        if page < last_page {
            return true;
        }

        let self_start = if page == self.first_page {
            self.first_offset
        } else {
            0
        };
        let self_end = if page == self.last_page {
            self.last_end
        } else {
            u64::MAX
        };
        let other_start = if page == other.first_page {
            other.first_offset
        } else {
            0
        };
        let other_end = if page == other.last_page {
            other.last_end
        } else {
            u64::MAX
        };
        self_start < other_end && other_start < self_end
    }
}

impl BackingView {
    /// Attaches a complete checked canonical range to an allocation subrange.
    pub fn new(
        allocation: GpuAllocationId,
        description: GpuAllocationDescription,
        allocation_offset: u64,
        range: CanonicalBackingRange,
    ) -> Result<Self, BackingViewError> {
        let end = allocation_offset
            .checked_add(range.size())
            .ok_or(BackingViewError::RangeOverflow)?;
        if end > description.size {
            return Err(BackingViewError::OutOfBounds {
                offset: allocation_offset,
                size: range.size(),
                allocation_size: description.size,
            });
        }
        let canonical_spans = canonical_spans(&range)?;
        Ok(Self {
            allocation,
            allocation_offset,
            range,
            canonical_spans: canonical_spans.into(),
        })
    }

    /// Returns the logical allocation retaining this range.
    #[must_use]
    pub const fn allocation(&self) -> GpuAllocationId {
        self.allocation
    }

    /// Returns the first byte within the logical allocation.
    #[must_use]
    pub const fn allocation_offset(&self) -> u64 {
        self.allocation_offset
    }

    /// Returns the exact retained canonical range.
    #[must_use]
    pub const fn range(&self) -> &CanonicalBackingRange {
        &self.range
    }

    /// Returns the retained byte count.
    #[must_use]
    pub const fn size(&self) -> u64 {
        self.range.size()
    }

    /// Returns the page-ordered, maximally compressed canonical coverage.
    ///
    /// Canonical identity is exposed without revealing host pointers so
    /// callers can index aliases without enumerating every retained page.
    #[must_use]
    pub fn canonical_spans(&self) -> &[CanonicalBackingSpan] {
        &self.canonical_spans
    }

    /// Returns whether both views retain any of the same canonical bytes.
    ///
    /// Each constructor creates a compressed page-ordered index once, so this
    /// comparison depends on discontiguous runs rather than retained pages.
    #[must_use]
    pub fn overlaps(&self, other: &Self) -> bool {
        sorted_spans_overlap(&self.canonical_spans, &other.canonical_spans)
    }
}

/// Failure to attach canonical bytes to a logical GPU allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackingViewError {
    /// Allocation offset plus retained size overflowed.
    RangeOverflow,
    /// The retained range lies outside the described allocation.
    OutOfBounds {
        offset: u64,
        size: u64,
        allocation_size: u64,
    },
    /// Two segments in the range name overlapping bytes of one canonical page.
    OverlappingCanonicalBytes,
}

impl Display for BackingViewError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RangeOverflow => formatter.write_str("GPU backing-view range overflows"),
            Self::OutOfBounds {
                offset,
                size,
                allocation_size,
            } => write!(
                formatter,
                "GPU backing view offset={offset:#x} size={size:#x} exceeds allocation-size={allocation_size:#x}"
            ),
            Self::OverlappingCanonicalBytes => {
                formatter.write_str("GPU backing view contains overlapping canonical page bytes")
            }
        }
    }
}

impl std::error::Error for BackingViewError {}

fn canonical_spans(
    range: &CanonicalBackingRange,
) -> Result<Vec<CanonicalBackingSpan>, BackingViewError> {
    let segments = range.segments();
    let ordered = segments
        .windows(2)
        .all(|pair| canonical_segment_key(&pair[0]) <= canonical_segment_key(&pair[1]));
    let mut spans = Vec::new();
    if ordered {
        for segment in segments {
            append_canonical_span(&mut spans, segment)?;
        }
    } else {
        let mut sorted = segments.iter().collect::<Vec<_>>();
        sorted.sort_unstable_by_key(|segment| canonical_segment_key(segment));
        for segment in sorted {
            append_canonical_span(&mut spans, segment)?;
        }
    }
    Ok(spans)
}

fn canonical_segment_key(segment: &CanonicalBackingSegment) -> (CanonicalPageId, u64, u64) {
    (
        segment.page(),
        segment.offset(),
        segment.offset() + segment.size(),
    )
}

fn append_canonical_span(
    spans: &mut Vec<CanonicalBackingSpan>,
    segment: &CanonicalBackingSegment,
) -> Result<(), BackingViewError> {
    let span = CanonicalBackingSpan {
        first_page: segment.page(),
        last_page: segment.page(),
        first_offset: segment.offset(),
        last_end: segment.offset() + segment.size(),
        ends_at_page_boundary: segment.ends_at_page_boundary(),
    };
    if let Some(previous) = spans.last_mut() {
        if previous.last_page == span.first_page {
            if span.first_offset < previous.last_end {
                return Err(BackingViewError::OverlappingCanonicalBytes);
            }
            if span.first_offset == previous.last_end {
                previous.last_end = span.last_end;
                previous.ends_at_page_boundary = span.ends_at_page_boundary;
                return Ok(());
            }
        } else if previous.last_page.store() == span.first_page.store()
            && previous.last_page.page().get().checked_add(1) == Some(span.first_page.page().get())
            && previous.ends_at_page_boundary
            && span.first_offset == 0
        {
            previous.last_page = span.last_page;
            previous.last_end = span.last_end;
            previous.ends_at_page_boundary = span.ends_at_page_boundary;
            return Ok(());
        }
    }
    spans.push(span);
    Ok(())
}

fn sorted_spans_overlap(left: &[CanonicalBackingSpan], right: &[CanonicalBackingSpan]) -> bool {
    let mut left_index = 0;
    let mut right_index = 0;
    while let (Some(left), Some(right)) = (left.get(left_index), right.get(right_index)) {
        if left.ends_before(right.first_page, right.first_offset) {
            left_index += 1;
        } else if right.ends_before(left.first_page, left.first_offset) {
            right_index += 1;
        } else {
            return left.overlaps(*right);
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use nixe_memory::{
        CanonicalBackingPage, CanonicalBackingSegment, CanonicalBackingStore, ContentGeneration,
        GuestPhysicalPageId, MappingGeneration, MemoryPermissions,
    };

    use super::*;

    fn page() -> CanonicalBackingPage {
        let store = CanonicalBackingStore::allocate().unwrap();
        CanonicalBackingPage::zeroed(
            &store,
            GuestPhysicalPageId::new(1),
            0x1000,
            ContentGeneration::INITIAL,
        )
        .unwrap()
    }

    fn segment(page: &CanonicalBackingPage, offset: u64, size: u64) -> CanonicalBackingSegment {
        CanonicalBackingSegment::new(
            page.clone(),
            offset,
            size,
            MemoryPermissions::READ_WRITE,
            page.content_generation(),
            MappingGeneration::INITIAL,
        )
        .unwrap()
    }

    #[test]
    fn descriptions_reject_empty_or_unaligned_allocations() {
        assert_eq!(
            GpuAllocationDescription::new(0, 0x1000),
            Err(AllocationDescriptionError::Empty)
        );
        assert_eq!(
            GpuAllocationDescription::new(0x1000, 24),
            Err(AllocationDescriptionError::InvalidAlignment { alignment: 24 })
        );
    }

    #[test]
    fn backing_views_are_bounded_and_pointer_free() {
        let page = page();
        let range = CanonicalBackingRange::new(vec![segment(&page, 0, 0x800)]).unwrap();
        let description = GpuAllocationDescription::new(0x1000, 0x100).unwrap();
        let view = BackingView::new(GpuAllocationId::new(7), description, 0x800, range).unwrap();

        assert_eq!(view.allocation(), GpuAllocationId::new(7));
        assert_eq!(view.allocation_offset(), 0x800);
        assert_eq!(view.size(), 0x800);
        assert_eq!(
            view.allocation().to_string(),
            "gpu-allocation=0x0000000000000007"
        );
    }

    #[test]
    fn backing_view_rejects_bounds_overflow_and_duplicate_canonical_bytes() {
        let page = page();
        let description = GpuAllocationDescription::new(0x1000, 1).unwrap();
        let range = CanonicalBackingRange::new(vec![segment(&page, 0, 0x800)]).unwrap();
        assert!(matches!(
            BackingView::new(GpuAllocationId::new(1), description, 0x900, range),
            Err(BackingViewError::OutOfBounds { .. })
        ));

        let range = CanonicalBackingRange::new(vec![
            segment(&page, 0, 0x800),
            segment(&page, 0x400, 0x800),
        ])
        .unwrap();
        assert_eq!(
            BackingView::new(GpuAllocationId::new(1), description, 0, range),
            Err(BackingViewError::OverlappingCanonicalBytes)
        );
    }

    #[test]
    fn backing_view_overlap_index_preserves_exact_interval_semantics() {
        let page = page();
        let description = GpuAllocationDescription::new(0x2000, 1).unwrap();
        let first = BackingView::new(
            GpuAllocationId::new(1),
            description,
            0,
            CanonicalBackingRange::new(vec![
                segment(&page, 0x800, 0x100),
                segment(&page, 0, 0x100),
            ])
            .unwrap(),
        )
        .unwrap();
        let overlapping = BackingView::new(
            GpuAllocationId::new(2),
            description,
            0,
            CanonicalBackingRange::new(vec![segment(&page, 0x80, 0x40)]).unwrap(),
        )
        .unwrap();
        let adjacent = BackingView::new(
            GpuAllocationId::new(3),
            description,
            0,
            CanonicalBackingRange::new(vec![segment(&page, 0x100, 0x100)]).unwrap(),
        )
        .unwrap();

        assert!(first.overlaps(&overlapping));
        assert!(overlapping.overlaps(&first));
        assert!(!first.overlaps(&adjacent));
        assert!(!adjacent.overlaps(&first));
    }

    #[test]
    fn backing_view_compresses_consecutive_pages_without_losing_boundaries() {
        let store = CanonicalBackingStore::allocate().unwrap();
        let pages = [10, 11, 12, 14]
            .into_iter()
            .map(|id| {
                CanonicalBackingPage::zeroed(
                    &store,
                    GuestPhysicalPageId::new(id),
                    0x1000,
                    ContentGeneration::INITIAL,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let description = GpuAllocationDescription::new(0x5000, 1).unwrap();
        let view = BackingView::new(
            GpuAllocationId::new(1),
            description,
            0,
            CanonicalBackingRange::new(vec![
                segment(&pages[0], 0x800, 0x800),
                segment(&pages[1], 0, 0x1000),
                segment(&pages[2], 0, 0x400),
                segment(&pages[3], 0, 0x1000),
            ])
            .unwrap(),
        )
        .unwrap();

        let [span, separate] = view.canonical_spans() else {
            panic!("only consecutive canonical pages may share a span");
        };
        assert_eq!(span.first_page(), pages[0].identity());
        assert_eq!(span.last_page(), pages[2].identity());
        assert_eq!(span.first_offset(), 0x800);
        assert_eq!(span.last_end(), 0x400);
        assert_eq!(separate.first_page(), pages[3].identity());
        assert_eq!(separate.last_page(), pages[3].identity());

        let middle = BackingView::new(
            GpuAllocationId::new(2),
            description,
            0,
            CanonicalBackingRange::new(vec![segment(&pages[1], 0x400, 0x100)]).unwrap(),
        )
        .unwrap();
        let adjacent = BackingView::new(
            GpuAllocationId::new(3),
            description,
            0,
            CanonicalBackingRange::new(vec![segment(&pages[2], 0x400, 0x100)]).unwrap(),
        )
        .unwrap();
        assert!(view.overlaps(&middle));
        assert!(!view.overlaps(&adjacent));
    }
}
