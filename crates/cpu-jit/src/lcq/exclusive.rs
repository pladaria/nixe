//! Physical-reservation slow path for exclusive stores. The ordinary same-VA
//! load/store sequence stays native; aliases and incoming reservations use
//! the memory authority's physical identity, never a virtual-address guess.

use crate::abi::ExclusiveStoreOperation;
use nixe_cpu::{
    exclusive::ExclusiveMonitorState,
    memory::{
        CpuMemory, DataAccessFault, MemoryAccess, MemoryAccessClass, MemoryAlignment,
        MemoryOrdering, MemoryValue,
    },
    state::a64::A64State,
};
use nixe_memory::{AddressSpaceId, GuestVirtualAddress};

impl ExclusiveStoreOperation {
    fn descriptor(self) -> MemoryAccess {
        MemoryAccess::new(
            self.size,
            MemoryAlignment::Natural,
            if self.release {
                MemoryOrdering::Release
            } else {
                MemoryOrdering::Relaxed
            },
            MemoryAccessClass::Exclusive,
        )
    }

    /// Complete a PRE exit after the frame's reservation handoff and release
    /// of the execution lease. A missing reservation fails without accessing
    /// memory, as in the interpreter. A fault consumes the monitor but leaves
    /// Ws and PC unchanged. No instruction is re-decoded or replayed.
    /// https://documentation-service.arm.com/static/67e40f3398aa3c3b6eea6a85#page=986
    /// https://documentation-service.arm.com/static/67e40f3398aa3c3b6eea6a85#page=984
    pub(crate) fn complete(
        self,
        state: &mut A64State,
        memory: &dyn CpuMemory,
        space: AddressSpaceId,
        monitor: &mut ExclusiveMonitorState,
    ) -> Result<(), DataAccessFault> {
        let address = if self.address == 31 {
            *state.stack_pointer_storage_mut()
        } else {
            state.general_register_storage_mut()[self.address as usize]
        };
        let mut read = |register: u8| {
            if register == 31 {
                0
            } else {
                u128::from(state.general_register_storage_mut()[register as usize])
            }
        };
        let low = read(self.source);
        let bits = if let Some(second) = self.second {
            let shift = self.size.bytes() * 4;
            (low & ((1u128 << shift) - 1)) | (read(second) << shift)
        } else {
            low
        };
        let reservation = monitor.reservation();
        monitor.clear();
        let succeeded = if let Some(reservation) = reservation {
            memory
                .store_exclusive(
                    space,
                    GuestVirtualAddress::new(address),
                    self.descriptor(),
                    MemoryValue::from_bits(self.size, bits),
                    reservation,
                )?
                .1
        } else {
            false
        };
        if self.status != 31 {
            state.general_register_storage_mut()[self.status as usize] = u64::from(!succeeded);
        }
        state.set_pc(state.pc().wrapping_add(4));
        Ok(())
    }
}
