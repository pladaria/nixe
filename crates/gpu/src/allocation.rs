//! Neutral allocation identities and weak physical resource views.

use std::{
    fmt::{Display, Formatter},
    sync::Arc,
};

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
/// checked physical mapping to an allocation only through
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

/// Immutable physical interpretation of one logical allocation subrange.
/// Cached views keep weak allocation metadata. In-flight native inputs retain
/// their own exact pins until actual device retirement.
#[derive(Clone)]
pub struct BackingView {
    allocation: GpuAllocationId,
    allocation_offset: u64,
    mapping: Arc<crate::physical::PhysicalMapping>,
    footprint: Arc<crate::physical::Footprint>,
}
impl std::fmt::Debug for BackingView {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BackingView")
            .field("allocation", &self.allocation)
            .field("allocation_offset", &self.allocation_offset)
            .field("mapping", &self.mapping)
            .finish()
    }
}
impl PartialEq for BackingView {
    fn eq(&self, other: &Self) -> bool {
        self.allocation == other.allocation
            && self.allocation_offset == other.allocation_offset
            && self.mapping == other.mapping
    }
}
impl Eq for BackingView {}
impl BackingView {
    pub fn new(
        allocation: GpuAllocationId,
        description: GpuAllocationDescription,
        allocation_offset: u64,
        mapping: Arc<crate::physical::PhysicalMapping>,
    ) -> Result<Self, BackingViewError> {
        let bytes = mapping.bytes();
        let end = allocation_offset
            .checked_add(bytes)
            .ok_or(BackingViewError::RangeOverflow)?;
        if end > description.size {
            return Err(BackingViewError::OutOfBounds {
                offset: allocation_offset,
                size: bytes,
                allocation_size: description.size,
            });
        }
        let footprint = crate::physical::Footprint::new(
            mapping.clone(),
            crate::physical::WriteDomain::Buffer { start: 0, bytes },
        )
        .expect("complete nonempty physical mapping is a valid byte footprint");
        Ok(Self {
            allocation,
            allocation_offset,
            mapping,
            footprint,
        })
    }
    pub fn has_internal_aliases(&self) -> bool {
        !self.mapping.is_injective()
    }
    pub const fn allocation(&self) -> GpuAllocationId {
        self.allocation
    }
    pub const fn allocation_offset(&self) -> u64 {
        self.allocation_offset
    }
    pub fn mapping(&self) -> &Arc<crate::physical::PhysicalMapping> {
        &self.mapping
    }
    pub fn size(&self) -> u64 {
        self.mapping.bytes()
    }
    pub fn overlaps(&self, other: &Self) -> bool {
        crate::metrics::record(crate::metrics::Counter::AliasComparisons, 1);
        crate::metrics::record(
            crate::metrics::Counter::AliasInputSpans,
            (self.mapping.spans().len() + other.mapping.spans().len()) as u64,
        );
        self.footprint.intersects(&other.footprint)
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
        }
    }
}

impl std::error::Error for BackingViewError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physical::PhysicalMapping;
    use nixe_memory::{AllocationRange, Machine, PAGE_SIZE};
    fn view(allocation: &nixe_memory::Allocation, offset: u64, bytes: u64) -> BackingView {
        let range =
            AllocationRange::new(allocation.pin_range(offset, bytes).unwrap(), offset, bytes)
                .unwrap();
        BackingView::new(
            GpuAllocationId::new(7),
            GpuAllocationDescription::new(allocation.bytes(), PAGE_SIZE).unwrap(),
            offset,
            PhysicalMapping::new(&[range]).unwrap(),
        )
        .unwrap()
    }
    #[test]
    fn allocation_descriptions_and_view_bounds_reject_invalid_ranges() {
        assert_eq!(
            GpuAllocationDescription::new(0, PAGE_SIZE),
            Err(AllocationDescriptionError::Empty)
        );
        assert_eq!(
            GpuAllocationDescription::new(PAGE_SIZE, 24),
            Err(AllocationDescriptionError::InvalidAlignment { alignment: 24 })
        );
        let machine = Machine::new(PAGE_SIZE).unwrap();
        let allocation = machine
            .account(PAGE_SIZE)
            .unwrap()
            .allocate(PAGE_SIZE)
            .unwrap();
        let first = view(&allocation, 0x800, 0x800);
        assert_eq!(first.allocation(), GpuAllocationId::new(7));
        assert_eq!(first.allocation_offset(), 0x800);
        assert_eq!(first.size(), 0x800);
        let description = GpuAllocationDescription::new(PAGE_SIZE, 1).unwrap();
        assert!(matches!(
            BackingView::new(
                GpuAllocationId::new(7),
                description,
                0x801,
                first.mapping().clone()
            ),
            Err(BackingViewError::OutOfBounds { .. })
        ));
        assert_eq!(
            BackingView::new(
                GpuAllocationId::new(7),
                description,
                u64::MAX,
                first.mapping().clone()
            ),
            Err(BackingViewError::RangeOverflow)
        );
    }
    #[test]
    fn physical_aliases_overlap_by_exact_bytes_across_logical_allocation_ids() {
        let machine = Machine::new(PAGE_SIZE).unwrap();
        let allocation = machine
            .account(PAGE_SIZE)
            .unwrap()
            .allocate(PAGE_SIZE)
            .unwrap();
        let first = view(&allocation, 128, 512);
        let overlap = view(&allocation, 512, 64);
        let adjacent = view(&allocation, 640, 64);
        assert!(first.overlaps(&overlap));
        assert!(!first.overlaps(&adjacent));
        let alias = BackingView::new(
            GpuAllocationId::new(99),
            GpuAllocationDescription::new(PAGE_SIZE, 1).unwrap(),
            0,
            first.mapping().clone(),
        )
        .unwrap();
        assert!(first.overlaps(&alias));
        let other_machine = Machine::new(PAGE_SIZE).unwrap();
        let other = other_machine
            .account(PAGE_SIZE)
            .unwrap()
            .allocate(PAGE_SIZE)
            .unwrap();
        assert!(!first.overlaps(&view(&other, 128, 512)));
    }
    #[test]
    fn cached_views_keep_metadata_without_pinning_guest_ram() {
        let machine = Machine::new(PAGE_SIZE).unwrap();
        let allocation = machine
            .account(PAGE_SIZE)
            .unwrap()
            .allocate(PAGE_SIZE)
            .unwrap();
        let cached = view(&allocation, 0, PAGE_SIZE);
        assert!(!cached.has_internal_aliases());
        drop(allocation);
        assert_eq!(machine.statistics().allocated_bytes, 0);
        assert!(cached.mapping().pin(0, 1).is_err());
    }
    #[test]
    fn repeated_physical_bytes_are_preserved_as_internal_aliases() {
        let machine = Machine::new(PAGE_SIZE).unwrap();
        let allocation = machine
            .account(PAGE_SIZE)
            .unwrap()
            .allocate(PAGE_SIZE)
            .unwrap();
        let range = AllocationRange::new(allocation.pin(), 0, PAGE_SIZE / 2).unwrap();
        let mapping = PhysicalMapping::new(&[range.clone(), range]).unwrap();
        let repeated = BackingView::new(
            GpuAllocationId::new(7),
            GpuAllocationDescription::new(PAGE_SIZE, 1).unwrap(),
            0,
            mapping,
        )
        .unwrap();
        assert!(repeated.has_internal_aliases());
        assert!(repeated.overlaps(&view(&allocation, 0, 1)));
        assert!(!repeated.overlaps(&view(&allocation, PAGE_SIZE / 2, 1)));
    }
}
