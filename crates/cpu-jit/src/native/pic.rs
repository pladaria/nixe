//! Native-readable PIC layout. A way is one pointer into an immutable bridge
//! owner, not an inline copy of its key. Only cold installation/Closed removal
//! writes the table; a hit needs neither synchronization nor a recency update.

use crate::abi::{BlockKey, ExitSiteKey, FpSpecialization};
use nixe_cpu::platform::TargetPlatform;
use std::cell::UnsafeCell;
use std::sync::Arc;

pub(crate) mod probe;

pub(crate) const SETS: usize = 2048;
pub(crate) const WAYS: usize = SETS * 2;

/// Explicit scalar encoding: generated code must not depend on Rust enum or
/// identity-wrapper layouts. Reachability/version validation is cold; the way's
/// strong owner and mandatory Closed unlink protect this exact native target.
#[repr(C)]
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Record {
    pub source: u64,
    pub state_map: u32,
    pub platform: u32,
    pub pc: u64,
    pub address_space: u64,
    pub profile: u64,
    pub fp: u64,
    pub address: usize,
}

impl Record {
    pub fn new(source: ExitSiteKey, target: BlockKey, address: usize) -> Self {
        Self {
            source: source.source.get(),
            state_map: source.state_map,
            platform: match target.platform {
                TargetPlatform::Switch1 => 0,
                TargetPlatform::Switch2 => 1,
            },
            pc: target.pc.get(),
            address_space: target.address_space.get(),
            profile: target.profile.get(),
            fp: match target.fp {
                FpSpecialization::Dynamic => 0,
                FpSpecialization::Exact(fpcr) => (1_u64 << 32) | u64::from(fpcr),
            },
            address,
        }
    }
}

/// Only a set selector, never identity validation. Both ways must compare the
/// complete source-site and target key; misaligned guest PCs cannot match.
pub(crate) fn set_index(source: ExitSiteKey, target: BlockKey) -> usize {
    ((target.pc.get() >> 2) ^ source.source.get() ^ u64::from(source.state_map)) as usize
        & (SETS - 1)
}

pub(crate) struct Table {
    // Arc keeps this allocation shared even while cold code exclusively borrows
    // Pic/Registration to change ownership or backlinks. UnsafeCell permits the
    // explicitly synchronized writes; neither registry growth nor those borrows
    // invalidates a native table pointer. The allocation is never resized.
    ways: Arc<[UnsafeCell<*const Record>]>,
}

// SAFETY: the only Rust access to cells is unsafe `set`. Native reads belong to
// the registered vCPU alone; writes require its suspension/quiescence or Closed.
// Moving cold ownership between threads does not confer native read permission.
unsafe impl Send for Table {}
unsafe impl Sync for Table {}

impl Table {
    // Persistent allocation including Arc's two reference counts. The containing
    // Table value is already included in its registration's metadata charge.
    pub const BYTES: usize = WAYS * size_of::<*const Record>() + 2 * size_of::<usize>();

    pub fn new() -> Self {
        Self {
            ways: (0..WAYS)
                .map(|_| UnsafeCell::new(std::ptr::null()))
                .collect(),
        }
    }

    /// Stable, non-owning view for this vCPU's native invocation. The caller must
    /// keep the registration alive and obey `set`'s exclusion requirements.
    pub fn as_ptr(&self) -> *const *const Record {
        self.ways.as_ptr().cast()
    }

    /// # Safety
    /// No native probe may read this table during the write: the owning vCPU
    /// must be quiescent/suspended, or maintenance must have reached Closed.
    /// A nonnull record must be immutable and strongly owned by this way before
    /// publication; clear the cell before releasing/replacing that owner.
    pub unsafe fn set(&self, slot: usize, record: *const Record) {
        assert!(slot < WAYS);
        // UnsafeCell is layout-transparent; as_ptr addresses its contents, not
        // an ordinary shared pointer array. The caller excludes native reads.
        unsafe { self.as_ptr().add(slot).cast_mut().write(record) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abi::CodeVersion;
    use nixe_cpu::profile::CpuProfileId;
    use nixe_memory::{AddressSpaceId, GuestVirtualAddress};
    use std::mem::offset_of;

    fn key() -> (ExitSiteKey, BlockKey) {
        (
            ExitSiteKey {
                source: CodeVersion::new(1).unwrap(),
                state_map: 7,
            },
            BlockKey {
                address_space: AddressSpaceId::new(1),
                pc: GuestVirtualAddress::new(0x1000),
                profile: CpuProfileId::new(1),
                platform: TargetPlatform::Switch1,
                fp: FpSpecialization::Dynamic,
            },
        )
    }

    #[test]
    fn pic_native_layout_has_two_pointer_ways_and_explicit_scalar_fields() {
        assert_eq!(size_of::<Record>(), 56);
        assert_eq!(offset_of!(Record, source), 0);
        assert_eq!(offset_of!(Record, state_map), 8);
        assert_eq!(offset_of!(Record, platform), 12);
        assert_eq!(offset_of!(Record, pc), 16);
        assert_eq!(offset_of!(Record, address_space), 24);
        assert_eq!(offset_of!(Record, profile), 32);
        assert_eq!(offset_of!(Record, fp), 40);
        assert_eq!(offset_of!(Record, address), 48);
        assert_eq!(Table::BYTES, 32768 + 16);
        let table = Table::new();
        for slot in 0..WAYS {
            assert!(unsafe { (*table.as_ptr().add(slot)).is_null() });
        }
    }

    #[test]
    fn pic_native_record_distinguishes_every_source_and_target_key_field() {
        let (source, target) = key();
        let expected = Record::new(source, target, 123);
        for other in [
            ExitSiteKey {
                source: CodeVersion::new(2).unwrap(),
                ..source
            },
            ExitSiteKey {
                state_map: 8,
                ..source
            },
        ] {
            assert_ne!(Record::new(other, target, 123), expected);
        }
        for other in [
            BlockKey {
                pc: GuestVirtualAddress::new(0x3000),
                ..target
            },
            BlockKey {
                address_space: AddressSpaceId::new(2),
                ..target
            },
            BlockKey {
                profile: CpuProfileId::new(2),
                ..target
            },
            BlockKey {
                platform: TargetPlatform::Switch2,
                ..target
            },
            BlockKey {
                fp: FpSpecialization::Exact(0),
                ..target
            },
            BlockKey {
                fp: FpSpecialization::Exact(u32::MAX),
                ..target
            },
        ] {
            // Deliberately retain the same selector: a hash match alone must
            // never accept another platform/profile/FP key or colliding PC.
            assert_eq!(set_index(source, other), set_index(source, target));
            assert_ne!(Record::new(source, other, 123), expected);
        }
        assert_eq!(Record::new(source, target, 123).fp, 0);
        assert_eq!(
            Record::new(
                source,
                BlockKey {
                    fp: FpSpecialization::Exact(0),
                    ..target
                },
                123
            )
            .fp,
            1_u64 << 32
        );
    }
}
