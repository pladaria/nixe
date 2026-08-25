//! Executor-local software TLB derived from canonical memory leases.

use std::mem::offset_of;

use nixe_cpu::memory::{CpuMemory, DataAccessKind};
use nixe_memory::{AddressSpaceId, CanonicalDirectAccessLease, GuestVirtualAddress};
use nixe_memory::{MemoryInvalidation, MemoryInvalidationKind};

use crate::abi::MemoryAcceleration;

pub(crate) const PAGE_BITS: u32 = 12;
pub(crate) const PAGE_SIZE: u64 = 1 << PAGE_BITS;
const ENTRY_COUNT: usize = 256;

pub(crate) const FLAG_READ: u32 = 1 << 0;
pub(crate) const FLAG_WRITE: u32 = 1 << 1;
pub(crate) const FLAG_WRITE_ARMED: u32 = 1 << 2;
pub(crate) const FLAG_ORDINARY: u32 = 1 << 3;
pub(crate) const FLAG_CPU_VISIBLE: u32 = 1 << 4;

/// Plain derived fields consumed by generated code. The owning lease remains
/// in [`SoftwareTlb`] and is never exposed through the native ABI.
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub(crate) struct NativeTlbEntry {
    pub(crate) guest_page: u64,
    pub(crate) address_space: u64,
    pub(crate) mapping_epoch: u64,
    pub(crate) host_word_base: usize,
    pub(crate) generation_address: usize,
    pub(crate) content_epoch_address: usize,
    pub(crate) cpu_write_epoch_address: usize,
    pub(crate) cpu_writes_active_address: usize,
    pub(crate) write_sequence_address: usize,
    pub(crate) validity_address: usize,
    pub(crate) visibility_epoch: u64,
    pub(crate) flags: u32,
    pub(crate) reserved: u32,
}

pub(crate) struct SoftwareTlb {
    entries: Box<[NativeTlbEntry]>,
    leases: Box<[Option<CanonicalDirectAccessLease>]>,
}

impl SoftwareTlb {
    pub(crate) fn new() -> Self {
        Self {
            entries: vec![NativeTlbEntry::default(); ENTRY_COUNT].into_boxed_slice(),
            leases: vec![None; ENTRY_COUNT].into_boxed_slice(),
        }
    }

    pub(crate) fn acceleration(
        &mut self,
        address_space: AddressSpaceId,
        mapping_epoch: u64,
    ) -> MemoryAcceleration {
        MemoryAcceleration {
            address_space: address_space.get(),
            mapping_epoch,
            tlb_base: self.entries.as_mut_ptr().addr(),
            tlb_entry_count: ENTRY_COUNT as u32,
            tlb_index_mask: (ENTRY_COUNT - 1) as u32,
        }
    }

    pub(crate) fn clear(&mut self) {
        self.entries.fill(NativeTlbEntry::default());
        self.leases.fill(None);
    }

    pub(crate) fn apply_invalidations(&mut self, records: &[MemoryInvalidation]) {
        for record in records {
            if let MemoryInvalidationKind::Mapping {
                address_space,
                start,
                size,
            } = record.kind
            {
                let first = start.get() >> PAGE_BITS;
                let end = start
                    .get()
                    .saturating_add(size)
                    .saturating_add(PAGE_SIZE - 1)
                    >> PAGE_BITS;
                for index in 0..self.entries.len() {
                    let entry = self.entries[index];
                    if entry.address_space == address_space.get()
                        && entry.guest_page >= first
                        && entry.guest_page < end
                    {
                        self.entries[index] = NativeTlbEntry::default();
                        self.leases[index] = None;
                    }
                }
            }
        }
    }

    /// Carries unaffected entries across a globally sequenced mapping epoch.
    /// A mapping record has already removed every changed virtual range.
    pub(crate) fn advance_mapping_epoch(
        &mut self,
        address_space: AddressSpaceId,
        mapping_epoch: u64,
    ) {
        for entry in &mut self.entries {
            if entry.address_space == address_space.get() && entry.host_word_base != 0 {
                entry.mapping_epoch = mapping_epoch;
            }
        }
    }

    pub(crate) fn install(
        &mut self,
        memory: &dyn CpuMemory,
        address_space: AddressSpaceId,
        mapping_epoch: u64,
        address: GuestVirtualAddress,
        kind: DataAccessKind,
    ) {
        let page = GuestVirtualAddress::new(address.get() & !(PAGE_SIZE - 1));
        let Some(lease) = memory.direct_access(address_space, page, kind) else {
            return;
        };
        if lease.size() as u64 != PAGE_SIZE {
            return;
        }
        let guest_page = page.get() >> PAGE_BITS;
        let index = (guest_page as usize) & (ENTRY_COUNT - 1);
        let permissions = lease.permissions();
        let mut flags = FLAG_ORDINARY | FLAG_CPU_VISIBLE;
        if permissions.contains(nixe_memory::MemoryPermissions::READ) {
            flags |= FLAG_READ;
        }
        if permissions.contains(nixe_memory::MemoryPermissions::WRITE) {
            flags |= FLAG_WRITE;
        }
        if matches!(kind, DataAccessKind::Write) {
            flags |= FLAG_WRITE_ARMED;
        }
        self.entries[index] = NativeTlbEntry {
            guest_page,
            address_space: address_space.get(),
            mapping_epoch,
            host_word_base: lease.host_word_base(),
            generation_address: lease.generation_address(),
            content_epoch_address: lease.content_epoch_address(),
            cpu_write_epoch_address: lease.cpu_write_epoch_address(),
            cpu_writes_active_address: lease.cpu_writes_active_address(),
            write_sequence_address: lease.write_sequence_address(),
            validity_address: lease.validity_address(),
            visibility_epoch: lease.visibility_epoch(),
            flags,
            reserved: 0,
        };
        self.leases[index] = Some(lease);
    }
}

pub(crate) struct NativeTlbOffsets {
    pub(crate) guest_page: usize,
    pub(crate) address_space: usize,
    pub(crate) mapping_epoch: usize,
    pub(crate) host_word_base: usize,
    pub(crate) generation_address: usize,
    pub(crate) content_epoch_address: usize,
    pub(crate) cpu_write_epoch_address: usize,
    pub(crate) cpu_writes_active_address: usize,
    pub(crate) write_sequence_address: usize,
    pub(crate) validity_address: usize,
    pub(crate) visibility_epoch: usize,
    pub(crate) flags: usize,
}

pub(crate) const TLB_ENTRY_SIZE: usize = size_of::<NativeTlbEntry>();
pub(crate) const TLB_OFFSETS: NativeTlbOffsets = NativeTlbOffsets {
    guest_page: offset_of!(NativeTlbEntry, guest_page),
    address_space: offset_of!(NativeTlbEntry, address_space),
    mapping_epoch: offset_of!(NativeTlbEntry, mapping_epoch),
    host_word_base: offset_of!(NativeTlbEntry, host_word_base),
    generation_address: offset_of!(NativeTlbEntry, generation_address),
    content_epoch_address: offset_of!(NativeTlbEntry, content_epoch_address),
    cpu_write_epoch_address: offset_of!(NativeTlbEntry, cpu_write_epoch_address),
    cpu_writes_active_address: offset_of!(NativeTlbEntry, cpu_writes_active_address),
    write_sequence_address: offset_of!(NativeTlbEntry, write_sequence_address),
    validity_address: offset_of!(NativeTlbEntry, validity_address),
    visibility_epoch: offset_of!(NativeTlbEntry, visibility_epoch),
    flags: offset_of!(NativeTlbEntry, flags),
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tlb_is_power_of_two_and_native_layout_is_bounded() {
        assert!(ENTRY_COUNT.is_power_of_two());
        assert_eq!(TLB_ENTRY_SIZE, 96);
    }

    #[test]
    fn mapping_ranges_remove_only_overlapping_entries_before_epoch_advance() {
        let space = AddressSpaceId::new(7);
        let mut tlb = SoftwareTlb::new();
        for (index, page) in [1_u64, 2].into_iter().enumerate() {
            tlb.entries[index] = NativeTlbEntry {
                guest_page: page,
                address_space: space.get(),
                mapping_epoch: 4,
                host_word_base: 0x1000 + index,
                ..NativeTlbEntry::default()
            };
        }
        tlb.apply_invalidations(&[MemoryInvalidation {
            cursor: nixe_memory::MemoryInvalidationCursor::new(1),
            kind: MemoryInvalidationKind::Mapping {
                address_space: space,
                start: GuestVirtualAddress::new(PAGE_SIZE),
                size: PAGE_SIZE,
            },
        }]);
        tlb.advance_mapping_epoch(space, 5);

        assert_eq!(tlb.entries[0].host_word_base, 0);
        assert_eq!(tlb.entries[1].mapping_epoch, 5);
        assert_ne!(tlb.entries[1].host_word_base, 0);
    }
}
