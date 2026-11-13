//! Heap growth shares the physical mapping lock with MapPhysicalMemory.
//! https://git.eden-emu.dev/eden-emu/eden/src/commit/d85005b2b27ebdbe785316886cdf04c2894d550e/src/core/hle/kernel/k_page_table_base.cpp
use super::*;

impl ProcessMemory {
    pub async fn resize_heap(
        self: &Arc<Self>,
        bytes: u64,
    ) -> Result<GuestVirtualAddress, MappingError> {
        if !bytes.is_multiple_of(PAGE_SIZE) || bytes > self.heap_region.size() {
            return Err(MappingError::InvalidRange);
        }
        let physical = self.lock_physical().await?;
        let base = self.heap_region.base();
        let old = self.heap_size();
        let mapping = Mapping {
            kind: MemoryKind::Heap,
            permissions: MemoryPermissions::READ_WRITE,
            uncached: false,
            locked: false,
            permission_locked: false,
        };
        if bytes < old {
            self.unmap(base.checked_add(bytes).unwrap(), old - bytes, mapping)?;
        } else if bytes > old {
            let allocation = physical.allocate(&self.account, bytes - old).await?;
            self.map(
                base.checked_add(old).unwrap(),
                allocation,
                0,
                bytes - old,
                mapping,
            )?;
        }
        self.heap_bytes.store(bytes, Ordering::Release);
        Ok(base)
    }
}
