//! Checked, retained and pointer-free canonical backing ranges.

use std::{
    collections::BTreeSet,
    fmt::{Display, Formatter},
    sync::Arc,
};

use crate::{
    AddressSpaceId, CanonicalBackingPage, CanonicalCpuWriteOverlap, CanonicalPageError,
    CanonicalPageId, ContentGeneration, DeviceAccessDeclaration, GuestVirtualAddress,
    MappingGeneration, MemoryPermissions, VisibilityCoordinator, VisibilityError, VisibilityState,
};

/// One contiguous segment of a translated canonical backing range.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CanonicalBackingSegment {
    backing: CanonicalBackingPage,
    offset: u64,
    size: u64,
    permissions: MemoryPermissions,
    content_generation: ContentGeneration,
    mapping_generation: MappingGeneration,
}

impl CanonicalBackingSegment {
    pub(crate) const fn backing(&self) -> &CanonicalBackingPage {
        &self.backing
    }

    /// Creates a checked segment and retains its canonical page.
    pub fn new(
        backing: CanonicalBackingPage,
        offset: u64,
        size: u64,
        permissions: MemoryPermissions,
        content_generation: ContentGeneration,
        mapping_generation: MappingGeneration,
    ) -> Result<Self, CanonicalRangeError> {
        let end = offset
            .checked_add(size)
            .ok_or(CanonicalRangeError::SegmentOverflow)?;
        if size == 0 || end > backing.size() as u64 {
            return Err(CanonicalRangeError::InvalidSegmentBounds);
        }
        if content_generation != backing.content_generation() {
            return Err(CanonicalRangeError::StaleContentGeneration);
        }
        Ok(Self {
            backing,
            offset,
            size,
            permissions,
            content_generation,
            mapping_generation,
        })
    }

    /// Returns the stable page identity, never a host pointer.
    #[must_use]
    pub fn page(&self) -> CanonicalPageId {
        self.backing.identity()
    }

    /// Returns the first byte within the canonical page.
    #[must_use]
    pub const fn offset(&self) -> u64 {
        self.offset
    }

    /// Returns the number of bytes in this segment.
    #[must_use]
    pub const fn size(&self) -> u64 {
        self.size
    }

    /// Returns the permissions of the CPU mapping used for translation.
    #[must_use]
    pub const fn permissions(&self) -> MemoryPermissions {
        self.permissions
    }

    /// Returns the content generation captured during translation.
    #[must_use]
    pub const fn content_generation(&self) -> ContentGeneration {
        self.content_generation
    }

    /// Returns the mapping generation captured during translation.
    #[must_use]
    pub const fn mapping_generation(&self) -> MappingGeneration {
        self.mapping_generation
    }

    /// Returns whether the retained bytes have changed since translation.
    #[must_use]
    pub fn content_is_current(&self) -> bool {
        self.backing.content_generation() == self.content_generation
    }

    /// Returns the conservative visibility authority shared by all aliases.
    #[must_use]
    pub fn visibility_state(&self) -> VisibilityState {
        self.backing.visibility_state()
    }

    /// Reports CPU-side writes overlapping this segment since `generation`.
    pub fn cpu_write_overlap_since(
        &self,
        generation: ContentGeneration,
    ) -> Result<CanonicalCpuWriteOverlap, CanonicalPageError> {
        self.backing
            .cpu_write_overlap_since(generation, self.offset, self.size)
    }
}

/// A validated logical byte range represented by retained page segments.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CanonicalBackingRange {
    segments: Box<[CanonicalBackingSegment]>,
    size: u64,
}

impl CanonicalBackingRange {
    /// Creates a non-empty range and checks its total length.
    pub fn new(segments: Vec<CanonicalBackingSegment>) -> Result<Self, CanonicalRangeError> {
        if segments.is_empty() {
            return Err(CanonicalRangeError::Empty);
        }
        let mut size = 0_u64;
        for segment in &segments {
            size = size
                .checked_add(segment.size)
                .ok_or(CanonicalRangeError::RangeOverflow)?;
        }
        Ok(Self {
            segments: segments.into_boxed_slice(),
            size,
        })
    }

    /// Returns the logical byte length.
    #[must_use]
    pub const fn size(&self) -> u64 {
        self.size
    }

    /// Returns the ordered canonical segments.
    #[must_use]
    pub fn segments(&self) -> &[CanonicalBackingSegment] {
        &self.segments
    }

    /// Captures a checked logical subrange with current content generations.
    ///
    /// The returned range retains the same canonical pages and permissions,
    /// but snapshots each touched page's content generation at this boundary.
    /// This is useful when a long-lived mapping is interpreted as a resource:
    /// content writes do not invalidate the mapping, while the derived view
    /// still needs an exact version key for later invalidation.
    pub fn snapshot_subrange(&self, offset: u64, size: u64) -> Result<Self, CanonicalRangeError> {
        let end = offset
            .checked_add(size)
            .ok_or(CanonicalRangeError::RangeOverflow)?;
        if size == 0 || end > self.size {
            return Err(CanonicalRangeError::InvalidSubrange);
        }

        let mut logical_start = 0_u64;
        let mut captured = Vec::new();
        for segment in &self.segments {
            let logical_end = logical_start
                .checked_add(segment.size)
                .ok_or(CanonicalRangeError::RangeOverflow)?;
            let capture_start = offset.max(logical_start);
            let capture_end = end.min(logical_end);
            if capture_start < capture_end {
                let within_segment = capture_start - logical_start;
                let page_offset = segment
                    .offset
                    .checked_add(within_segment)
                    .ok_or(CanonicalRangeError::SegmentOverflow)?;
                captured.push(CanonicalBackingSegment::new(
                    segment.backing.clone(),
                    page_offset,
                    capture_end - capture_start,
                    segment.permissions,
                    segment.backing.content_generation(),
                    segment.mapping_generation,
                )?);
            }
            logical_start = logical_end;
            if logical_start >= end {
                break;
            }
        }
        Self::new(captured)
    }

    /// Copies a checked logical subrange from retained canonical storage.
    ///
    /// Reads walk canonical page segments directly. A CPU virtual address is
    /// neither required nor reconstructed, so aliases and unmapped-but-retained
    /// storage preserve the same byte identity.
    pub fn read(&self, offset: u64, output: &mut [u8]) -> Result<(), CanonicalRangeAccessError> {
        let output_size =
            u64::try_from(output.len()).map_err(|_| CanonicalRangeAccessError::RangeOverflow)?;
        let end = offset
            .checked_add(output_size)
            .ok_or(CanonicalRangeAccessError::RangeOverflow)?;
        if end > self.size {
            return Err(CanonicalRangeAccessError::OutOfBounds {
                offset,
                size: output_size,
                range_size: self.size,
            });
        }
        if output.is_empty() {
            return Ok(());
        }

        let mut logical_start = 0_u64;
        let mut copied = 0_usize;
        for segment in &self.segments {
            let logical_end = logical_start
                .checked_add(segment.size)
                .ok_or(CanonicalRangeAccessError::RangeOverflow)?;
            let read_start = offset.max(logical_start);
            let read_end = end.min(logical_end);
            if read_start < read_end {
                let within_segment = read_start - logical_start;
                let page_offset = segment
                    .offset
                    .checked_add(within_segment)
                    .ok_or(CanonicalRangeAccessError::RangeOverflow)?;
                let copy_size = usize::try_from(read_end - read_start)
                    .map_err(|_| CanonicalRangeAccessError::RangeOverflow)?;
                let page_offset = usize::try_from(page_offset)
                    .map_err(|_| CanonicalRangeAccessError::RangeOverflow)?;
                let copied_end = copied
                    .checked_add(copy_size)
                    .ok_or(CanonicalRangeAccessError::RangeOverflow)?;
                segment
                    .backing
                    .read(page_offset, &mut output[copied..copied_end])
                    .map_err(CanonicalRangeAccessError::Backing)?;
                copied = copied_end;
            }
            logical_start = logical_end;
            if logical_start >= end {
                break;
            }
        }
        if copied != output.len() {
            return Err(CanonicalRangeAccessError::IncompleteRange);
        }
        Ok(())
    }

    /// Establishes device visibility before executing a declared access.
    ///
    /// The initial implementation transitions complete canonical pages even
    /// when the logical range contains only a page fragment. This deliberately
    /// conservative granularity preserves overlapping-alias correctness.
    pub fn prepare_device_access(
        &self,
        declaration: DeviceAccessDeclaration,
        coordinator: Arc<dyn VisibilityCoordinator>,
    ) -> Result<(), VisibilityError> {
        let mut visited = BTreeSet::new();
        for segment in &self.segments {
            if visited.insert(segment.page()) {
                segment
                    .backing
                    .prepare_device_access(declaration, Arc::clone(&coordinator))?;
            }
        }
        Ok(())
    }

    /// Publishes newer device contents after the declaration's write point.
    ///
    /// Callers must invoke this only after the host operations which make that
    /// point true. It records logical ownership; it does not signal a guest
    /// fence or claim host queue completion.
    pub fn complete_device_write(
        &self,
        declaration: DeviceAccessDeclaration,
        coordinator: Arc<dyn VisibilityCoordinator>,
    ) -> Result<(), VisibilityError> {
        if !declaration.kind().writes() {
            return Err(VisibilityError::DeclarationDoesNotWrite);
        }
        let mut visited = BTreeSet::new();
        for segment in &self.segments {
            if visited.insert(segment.page()) {
                segment
                    .backing
                    .complete_device_write(declaration, Arc::clone(&coordinator))?;
            }
        }
        Ok(())
    }

    /// Marks every retained page invalid after an unrecoverable residency or
    /// visibility failure.
    pub fn invalidate_visibility(&self) -> Result<(), VisibilityError> {
        let mut visited = BTreeSet::new();
        for segment in &self.segments {
            if visited.insert(segment.page()) {
                segment.backing.invalidate_visibility()?;
            }
        }
        Ok(())
    }
}

/// Invalid construction of a canonical range.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CanonicalRangeError {
    Empty,
    InvalidSubrange,
    SegmentOverflow,
    InvalidSegmentBounds,
    StaleContentGeneration,
    RangeOverflow,
}

impl Display for CanonicalRangeError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Empty => "canonical backing range is empty",
            Self::InvalidSubrange => "canonical backing subrange is empty or out of bounds",
            Self::SegmentOverflow => "canonical segment end overflows",
            Self::InvalidSegmentBounds => "canonical segment is outside its retained page",
            Self::StaleContentGeneration => {
                "canonical segment generation changed during construction"
            }
            Self::RangeOverflow => "canonical backing range length overflows",
        })
    }
}

impl std::error::Error for CanonicalRangeError {}

/// Failure while accessing a retained canonical backing range.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CanonicalRangeAccessError {
    RangeOverflow,
    OutOfBounds {
        offset: u64,
        size: u64,
        range_size: u64,
    },
    IncompleteRange,
    Backing(CanonicalPageError),
}

impl Display for CanonicalRangeAccessError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RangeOverflow => formatter.write_str("canonical range access overflows"),
            Self::OutOfBounds {
                offset,
                size,
                range_size,
            } => write!(
                formatter,
                "canonical range access offset={offset:#x} size={size:#x} exceeds \
                 range-size={range_size:#x}"
            ),
            Self::IncompleteRange => {
                formatter.write_str("canonical range segments do not cover the requested bytes")
            }
            Self::Backing(error) => write!(formatter, "canonical backing access failed: {error}"),
        }
    }
}

impl std::error::Error for CanonicalRangeAccessError {}

/// Why a CPU virtual range could not be translated to canonical RAM.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CanonicalRangeTranslationErrorReason {
    Empty,
    AddressOverflow,
    Unmapped,
    PermissionDenied,
    DeviceMemory,
    InconsistentBacking,
    ResourceExhausted,
}

/// Pointer-free failure from a canonical range translation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CanonicalRangeTranslationError {
    pub address_space: AddressSpaceId,
    pub address: GuestVirtualAddress,
    pub reason: CanonicalRangeTranslationErrorReason,
}

impl Display for CanonicalRangeTranslationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "canonical range translation failed in {} at {}: {:?}",
            self.address_space, self.address, self.reason
        )
    }
}

impl std::error::Error for CanonicalRangeTranslationError {}

/// Device-neutral boundary for validated CPU-VA to backing translation.
pub trait CanonicalRangeTranslator {
    /// Translates the complete virtual range or returns its first failing byte.
    fn translate_canonical_range(
        &self,
        address_space: AddressSpaceId,
        address: GuestVirtualAddress,
        size: u64,
        required_permissions: MemoryPermissions,
    ) -> Result<CanonicalBackingRange, CanonicalRangeTranslationError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BackingStoreId, GuestPhysicalPageId};

    #[test]
    fn segment_construction_checks_bounds_and_generation() {
        let page = CanonicalBackingPage::zeroed(
            CanonicalPageId::new(
                BackingStoreId::allocate().unwrap(),
                GuestPhysicalPageId::new(1),
            ),
            0x1000,
            ContentGeneration::INITIAL,
        )
        .unwrap();

        assert_eq!(
            CanonicalBackingSegment::new(
                page.clone(),
                0xfff,
                2,
                MemoryPermissions::READ,
                ContentGeneration::INITIAL,
                MappingGeneration::new(1),
            ),
            Err(CanonicalRangeError::InvalidSegmentBounds)
        );
        assert_eq!(
            CanonicalBackingSegment::new(
                page,
                0,
                1,
                MemoryPermissions::READ,
                ContentGeneration::new(1),
                MappingGeneration::new(1),
            ),
            Err(CanonicalRangeError::StaleContentGeneration)
        );
    }

    #[test]
    fn retained_range_reads_checked_subranges_across_pages() {
        let store = BackingStoreId::allocate().unwrap();
        let first = CanonicalBackingPage::initialized(
            CanonicalPageId::new(store, GuestPhysicalPageId::new(1)),
            &[0x10, 0x11, 0x12, 0x13],
            ContentGeneration::INITIAL,
        )
        .unwrap();
        let second = CanonicalBackingPage::initialized(
            CanonicalPageId::new(store, GuestPhysicalPageId::new(2)),
            &[0x20, 0x21, 0x22, 0x23],
            ContentGeneration::INITIAL,
        )
        .unwrap();
        let range = CanonicalBackingRange::new(vec![
            CanonicalBackingSegment::new(
                first,
                1,
                3,
                MemoryPermissions::READ,
                ContentGeneration::INITIAL,
                MappingGeneration::new(1),
            )
            .unwrap(),
            CanonicalBackingSegment::new(
                second,
                0,
                3,
                MemoryPermissions::READ,
                ContentGeneration::INITIAL,
                MappingGeneration::new(2),
            )
            .unwrap(),
        ])
        .unwrap();

        let mut bytes = [0_u8; 4];
        range.read(1, &mut bytes).unwrap();
        assert_eq!(bytes, [0x12, 0x13, 0x20, 0x21]);
        assert_eq!(
            range.read(5, &mut [0_u8; 2]),
            Err(CanonicalRangeAccessError::OutOfBounds {
                offset: 5,
                size: 2,
                range_size: 6,
            })
        );
    }

    #[test]
    fn snapshot_subrange_retains_exact_bytes_and_current_content_generations() {
        let store = BackingStoreId::allocate().unwrap();
        let first = CanonicalBackingPage::zeroed(
            CanonicalPageId::new(store, GuestPhysicalPageId::new(1)),
            0x1000,
            ContentGeneration::INITIAL,
        )
        .unwrap();
        let second = CanonicalBackingPage::zeroed(
            CanonicalPageId::new(store, GuestPhysicalPageId::new(2)),
            0x1000,
            ContentGeneration::INITIAL,
        )
        .unwrap();
        let range = CanonicalBackingRange::new(vec![
            CanonicalBackingSegment::new(
                first.clone(),
                0,
                0x1000,
                MemoryPermissions::READ_WRITE,
                ContentGeneration::INITIAL,
                MappingGeneration::new(1),
            )
            .unwrap(),
            CanonicalBackingSegment::new(
                second.clone(),
                0,
                0x1000,
                MemoryPermissions::READ_WRITE,
                ContentGeneration::INITIAL,
                MappingGeneration::new(2),
            )
            .unwrap(),
        ])
        .unwrap();

        first.prepare_write().unwrap();
        first
            .write_preflighted(
                0xff0,
                &[0x5a; 0x10],
                ContentGeneration::INITIAL,
                ContentGeneration::new(1),
            )
            .unwrap();
        second.prepare_write().unwrap();
        second
            .write_preflighted(
                0,
                &[0xa5; 0x10],
                ContentGeneration::INITIAL,
                ContentGeneration::new(1),
            )
            .unwrap();
        let snapshot = range.snapshot_subrange(0xff0, 0x20).unwrap();

        assert_eq!(snapshot.size(), 0x20);
        assert_eq!(snapshot.segments().len(), 2);
        assert_eq!(snapshot.segments()[0].offset(), 0xff0);
        assert_eq!(
            snapshot.segments()[0].content_generation(),
            first.content_generation()
        );
        assert_eq!(
            snapshot.segments()[1].content_generation(),
            second.content_generation()
        );
        let mut bytes = [0; 0x20];
        snapshot.read(0, &mut bytes).unwrap();
        assert_eq!(&bytes[..0x10], &[0x5a; 0x10]);
        assert_eq!(&bytes[0x10..], &[0xa5; 0x10]);

        assert_eq!(
            range.snapshot_subrange(0, 0),
            Err(CanonicalRangeError::InvalidSubrange)
        );
        assert_eq!(
            range.snapshot_subrange(0x1ff0, 0x20),
            Err(CanonicalRangeError::InvalidSubrange)
        );
    }
}
