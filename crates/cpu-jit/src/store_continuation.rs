//! Cold completion of an ordinary native store instruction after MemoryAbort.
//!
//! Dynarmic commits each pair/structure store before checking the next access.
//! Restarting at the guest PC would replay that prefix after releasing admission.
//! Checked semantics reconstruct addresses/writeback but skip completed stores.
//! vendor/dynarmic/src/dynarmic/frontend/A64/translate/impl/load_store_register_pair.cpp

use nixe_cpu::error::InstructionFetchFault;
use nixe_cpu::exclusive::ExclusiveReservation;
use nixe_cpu::memory::*;
use nixe_memory::{
    AddressSpaceId, GuestVirtualAddress, MemoryInvalidation, MemoryInvalidationCursor,
    MemoryInvalidationError, MemoryInvalidationSource,
};
use std::sync::atomic::{AtomicBool, Ordering};

pub(crate) struct StoreContinuation<'a> {
    memory: &'a ExecutionMemory,
    address: GuestVirtualAddress,
    reached: AtomicBool,
}
impl<'a> StoreContinuation<'a> {
    pub(crate) fn new(memory: &'a ExecutionMemory, address: GuestVirtualAddress) -> Self {
        Self {
            memory,
            address,
            reached: AtomicBool::new(false),
        }
    }
    pub(crate) fn reached(&self) -> bool {
        self.reached.load(Ordering::Relaxed)
    }
}
impl MemoryInvalidationSource for StoreContinuation<'_> {
    fn invalidation_cursor(&self) -> MemoryInvalidationCursor {
        self.memory.invalidation_cursor()
    }
    fn read_invalidations_since(
        &self,
        after: MemoryInvalidationCursor,
        output: &mut Vec<MemoryInvalidation>,
    ) -> Result<MemoryInvalidationCursor, MemoryInvalidationError> {
        self.memory.read_invalidations_since(after, output)
    }
}
impl InstructionMemory for StoreContinuation<'_> {
    fn fetch32(
        &self,
        space: AddressSpaceId,
        address: GuestVirtualAddress,
    ) -> Result<FetchedInstruction, InstructionFetchFault> {
        self.memory.fetch32(space, address)
    }
}
impl CpuMemory for StoreContinuation<'_> {
    fn write(
        &self,
        space: AddressSpaceId,
        address: GuestVirtualAddress,
        access: MemoryAccess,
        value: MemoryValue,
    ) -> Result<DataWriteResult, DataAccessFault> {
        // Dynarmic combines ST1 elements into a vector store; checked semantics
        // may use smaller elements. The faulting start VA identifies the tail.
        if !self.reached() {
            if address != self.address {
                // These operations already completed under the native lease.
                // Do not revalidate, repair or write their possibly remapped VA.
                return Ok(DataWriteResult {
                    region: MemoryRegionKind::Ram,
                });
            }
            self.reached.store(true, Ordering::Relaxed);
        }
        self.memory.write(space, address, access, value)
    }
    fn read(
        &self,
        space: AddressSpaceId,
        address: GuestVirtualAddress,
        access: MemoryAccess,
    ) -> Result<DataReadResult, DataAccessFault> {
        self.memory.read(space, address, access)
    }
    fn atomic_read_modify_write(
        &self,
        space: AddressSpaceId,
        address: GuestVirtualAddress,
        access: MemoryAccess,
        kind: AtomicRmwKind,
        operand: MemoryValue,
    ) -> Result<AtomicMemoryResult, DataAccessFault> {
        self.memory
            .atomic_read_modify_write(space, address, access, kind, operand)
    }
    fn atomic_compare_exchange(
        &self,
        space: AddressSpaceId,
        address: GuestVirtualAddress,
        access: MemoryAccess,
        expected: MemoryValue,
        replacement: MemoryValue,
    ) -> Result<AtomicMemoryResult, DataAccessFault> {
        self.memory
            .atomic_compare_exchange(space, address, access, expected, replacement)
    }
    fn maintain_cache(
        &self,
        space: AddressSpaceId,
        kind: CacheMaintenanceKind,
        address: Option<GuestVirtualAddress>,
    ) -> Result<(), DataAccessFault> {
        self.memory.maintain_cache(space, kind, address)
    }
    fn query_page(
        &self,
        space: AddressSpaceId,
        address: GuestVirtualAddress,
    ) -> Option<(MemoryRegionKind, MemoryMappingProperties)> {
        self.memory.query_page(space, address)
    }
    fn query_memory(
        &self,
        space: AddressSpaceId,
        address: GuestVirtualAddress,
        end: GuestVirtualAddress,
    ) -> Option<MemoryQueryResult> {
        self.memory.query_memory(space, address, end)
    }
    fn load_exclusive(
        &self,
        space: AddressSpaceId,
        address: GuestVirtualAddress,
        access: MemoryAccess,
    ) -> Result<(DataReadResult, ExclusiveReservation), DataAccessFault> {
        self.memory.load_exclusive(space, address, access)
    }
    fn store_exclusive(
        &self,
        space: AddressSpaceId,
        address: GuestVirtualAddress,
        access: MemoryAccess,
        value: MemoryValue,
        reservation: ExclusiveReservation,
    ) -> Result<(DataWriteResult, bool), DataAccessFault> {
        self.memory
            .store_exclusive(space, address, access, value, reservation)
    }
}
