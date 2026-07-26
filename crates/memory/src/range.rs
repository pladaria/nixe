//! Checked, retained and pointer-free canonical backing ranges.

use std::{
    collections::BTreeSet,
    fmt::{Display, Formatter},
    sync::Arc,
};

use crate::{
    AddressSpaceId, CanonicalBackingPage, CanonicalPageId, ContentGeneration,
    DeviceAccessDeclaration, GuestVirtualAddress, MappingGeneration, MemoryPermissions,
    VisibilityCoordinator, VisibilityError, VisibilityState,
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
    SegmentOverflow,
    InvalidSegmentBounds,
    StaleContentGeneration,
    RangeOverflow,
}

impl Display for CanonicalRangeError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Empty => "canonical backing range is empty",
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
}
