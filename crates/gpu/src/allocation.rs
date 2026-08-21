//! Neutral allocation identities and retained canonical backing views.

use std::{
    fmt::{Display, Formatter},
    sync::Arc,
};

use nixe_memory::{CanonicalBackingRange, CanonicalPageId};

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
    canonical_intervals: Arc<[CanonicalByteInterval]>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CanonicalByteInterval {
    page: CanonicalPageId,
    start: u64,
    end: u64,
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
        let canonical_intervals = canonical_intervals(&range);
        if canonical_intervals
            .windows(2)
            .any(|pair| pair[0].page == pair[1].page && pair[1].start < pair[0].end)
        {
            return Err(BackingViewError::OverlappingCanonicalBytes);
        }
        Ok(Self {
            allocation,
            allocation_offset,
            range,
            canonical_intervals: canonical_intervals.into(),
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

    /// Returns whether both views retain any of the same canonical bytes.
    ///
    /// Each constructor creates a page-ordered interval index once, so this
    /// comparison is linear in the number of segments rather than quadratic.
    #[must_use]
    pub fn overlaps(&self, other: &Self) -> bool {
        sorted_intervals_overlap(&self.canonical_intervals, &other.canonical_intervals)
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

fn canonical_intervals(range: &CanonicalBackingRange) -> Vec<CanonicalByteInterval> {
    let mut intervals = range
        .segments()
        .iter()
        .map(|segment| CanonicalByteInterval {
            page: segment.page(),
            start: segment.offset(),
            end: segment.offset() + segment.size(),
        })
        .collect::<Vec<_>>();
    intervals.sort_unstable_by_key(|interval| (interval.page, interval.start, interval.end));
    intervals
}

fn sorted_intervals_overlap(
    left: &[CanonicalByteInterval],
    right: &[CanonicalByteInterval],
) -> bool {
    let mut left_index = 0;
    let mut right_index = 0;
    while let (Some(left), Some(right)) = (left.get(left_index), right.get(right_index)) {
        if left.page < right.page || (left.page == right.page && left.end <= right.start) {
            left_index += 1;
        } else if right.page < left.page || right.end <= left.start {
            right_index += 1;
        } else {
            return true;
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
}
