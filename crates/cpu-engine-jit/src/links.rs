//! Domain-owned writable link tables kept outside executable memory.

use core::mem::{offset_of, size_of};
use std::ptr;
use std::sync::atomic::{AtomicPtr, Ordering};

use nixe_cpu::location::LocationDescriptor;

use crate::abi::NativeEntryAddress;

pub(crate) const INDIRECT_LINK_WAYS: usize = 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum LinkKind {
    Direct,
    Indirect,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct LinkSiteMetadata {
    pub(crate) kind: LinkKind,
    pub(crate) direct_target: Option<LocationDescriptor>,
}

/// Immutable payload published through one atomic link-cell pointer.
///
/// After unlinking, the domain cache retains each payload until its native
/// reclamation epoch proves that no executor still holds a previously acquired
/// pointer.
#[repr(C)]
pub(crate) struct NativeLinkTarget {
    pub(crate) guest_pc: u64,
    pub(crate) guest_state: u32,
    pub(crate) reserved: u32,
    pub(crate) region_id: u64,
    pub(crate) link_table: usize,
    pub(crate) metadata: usize,
    pub(crate) entry: NativeEntryAddress,
}

#[repr(C)]
pub(crate) struct LinkCell {
    target: AtomicPtr<NativeLinkTarget>,
}

impl LinkCell {
    fn empty() -> Self {
        Self {
            target: AtomicPtr::new(ptr::null_mut()),
        }
    }

    pub(crate) fn publish(&self, target: *mut NativeLinkTarget) -> bool {
        self.target
            .compare_exchange(
                ptr::null_mut(),
                target,
                Ordering::Release,
                Ordering::Acquire,
            )
            .is_ok()
    }

    pub(crate) fn clear_if(&self, expected: *mut NativeLinkTarget) {
        let _ = self.target.compare_exchange(
            expected,
            ptr::null_mut(),
            Ordering::AcqRel,
            Ordering::Acquire,
        );
    }
}

#[repr(C)]
pub(crate) struct LinkSite {
    cells: [LinkCell; INDIRECT_LINK_WAYS],
}

impl LinkSite {
    fn empty() -> Self {
        Self {
            cells: std::array::from_fn(|_| LinkCell::empty()),
        }
    }

    pub(crate) fn cell(&self, way: usize) -> Option<&LinkCell> {
        self.cells.get(way)
    }
}

pub(crate) struct LinkTable {
    sites: Box<[LinkSite]>,
}

impl LinkTable {
    pub(crate) fn new(site_count: usize) -> Self {
        Self {
            sites: std::iter::repeat_with(LinkSite::empty)
                .take(site_count)
                .collect(),
        }
    }

    pub(crate) fn base_address(&self) -> usize {
        self.sites.as_ptr().addr()
    }

    pub(crate) fn site(&self, index: usize) -> Option<&LinkSite> {
        self.sites.get(index)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct LinkOffsets {
    pub(crate) site_size: usize,
    pub(crate) cell_size: usize,
    pub(crate) cell_target: usize,
    pub(crate) target_guest_pc: usize,
    pub(crate) target_guest_state: usize,
    pub(crate) target_region_id: usize,
    pub(crate) target_link_table: usize,
    pub(crate) target_metadata: usize,
    pub(crate) target_entry: usize,
}

pub(crate) const LINK_OFFSETS: LinkOffsets = LinkOffsets {
    site_size: size_of::<LinkSite>(),
    cell_size: size_of::<LinkCell>(),
    cell_target: offset_of!(LinkCell, target),
    target_guest_pc: offset_of!(NativeLinkTarget, guest_pc),
    target_guest_state: offset_of!(NativeLinkTarget, guest_state),
    target_region_id: offset_of!(NativeLinkTarget, region_id),
    target_link_table: offset_of!(NativeLinkTarget, link_table),
    target_metadata: offset_of!(NativeLinkTarget, metadata),
    target_entry: offset_of!(NativeLinkTarget, entry),
};

const _: () = assert!(size_of::<AtomicPtr<NativeLinkTarget>>() == size_of::<usize>());
