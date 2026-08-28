/// Incremental Horizon physical-memory accounting for one process.
///
/// Horizon reports code and the main-thread stack separately from the page
/// table's normal-memory counter. Keeping those categories here makes all
/// related `GetInfo` queries derive from one coherent snapshot without walking
/// the guest page table on every supervisor call.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProcessMemoryAccounting {
    memory_capacity: u64,
    normal_memory_size: u64,
    code_size: u64,
    main_thread_stack_size: u64,
    heap_size: u64,
    system_resource_size: u64,
    system_resource_used: u64,
    required_secure_memory_size: u64,
}

impl ProcessMemoryAccounting {
    pub(super) fn new(
        memory_capacity: u64,
        initial_mapped_size: u64,
        code_size: u64,
        main_thread_stack_size: u64,
        system_resource_size: u64,
    ) -> Option<Self> {
        let normal_memory_size = initial_mapped_size
            .checked_sub(code_size)?
            .checked_sub(main_thread_stack_size)?;
        // A nonzero NPDM system-resource size creates a process-owned secure
        // resource on Horizon. Nixe does not allocate host backing for its
        // kernel metadata, but the reservation still consumes the process's
        // physical-memory allowance and must be excluded by InfoTypes 21/22.
        let required_secure_memory_size = system_resource_size;
        if initial_mapped_size.checked_add(required_secure_memory_size)? > memory_capacity {
            return None;
        }
        let accounting = Self {
            memory_capacity,
            normal_memory_size,
            code_size,
            main_thread_stack_size,
            heap_size: 0,
            system_resource_size,
            system_resource_used: 0,
            required_secure_memory_size,
        };
        Some(accounting)
    }

    /// Returns the process physical-memory ceiling reported by Horizon.
    #[must_use]
    pub const fn total_user_physical_memory_size(self) -> u64 {
        self.memory_capacity
    }

    /// Returns mapped process memory plus its non-default secure reservation.
    #[must_use]
    pub const fn used_user_physical_memory_size(self) -> u64 {
        self.used_non_system_user_physical_memory_size() + self.required_secure_memory_size
    }

    /// Returns the process ceiling after excluding its secure reservation.
    #[must_use]
    pub const fn total_non_system_user_physical_memory_size(self) -> u64 {
        self.memory_capacity - self.required_secure_memory_size
    }

    /// Mirrors Horizon's normal-memory + code + main-stack calculation.
    #[must_use]
    pub const fn used_non_system_user_physical_memory_size(self) -> u64 {
        self.normal_memory_size + self.code_size + self.main_thread_stack_size
    }

    #[must_use]
    pub const fn total_system_resource_size(self) -> u64 {
        self.system_resource_size
    }

    #[must_use]
    pub const fn used_system_resource_size(self) -> u64 {
        self.system_resource_used
    }

    #[must_use]
    pub const fn heap_size(self) -> u64 {
        self.heap_size
    }

    pub(crate) const fn can_resize_heap(self, new_size: u64) -> bool {
        let Some(non_heap_size) = self
            .used_user_physical_memory_size()
            .checked_sub(self.heap_size)
        else {
            return false;
        };
        match non_heap_size.checked_add(new_size) {
            Some(new_used_size) => new_used_size <= self.memory_capacity,
            None => false,
        }
    }

    pub(crate) fn commit_heap_size(&mut self, new_size: u64) {
        self.normal_memory_size = self
            .normal_memory_size
            .checked_sub(self.heap_size)
            .and_then(|size| size.checked_add(new_size))
            .expect("a validated heap resize preserves process memory accounting");
        self.heap_size = new_size;
    }

    pub(super) const fn can_commit_normal_memory(self, size: u64) -> bool {
        match self.used_user_physical_memory_size().checked_add(size) {
            Some(used) => used <= self.memory_capacity,
            None => false,
        }
    }

    pub(super) fn commit_normal_memory(&mut self, size: u64) {
        self.normal_memory_size = self
            .normal_memory_size
            .checked_add(size)
            .expect("validated normal-memory growth cannot overflow");
    }

    pub(super) fn release_normal_memory(&mut self, size: u64) {
        self.normal_memory_size = self
            .normal_memory_size
            .checked_sub(size)
            .expect("released normal memory was previously committed");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn horizon_memory_views_stay_coherent_across_heap_and_tls_changes() {
        let mut accounting =
            ProcessMemoryAccounting::new(0x100_000, 0x50_000, 0x20_000, 0x10_000, 0x8_000).unwrap();

        assert_eq!(
            accounting.used_non_system_user_physical_memory_size(),
            0x50_000
        );
        assert_eq!(accounting.used_user_physical_memory_size(), 0x58_000);
        assert_eq!(
            accounting.total_non_system_user_physical_memory_size(),
            0xf8_000
        );
        assert_eq!(accounting.total_system_resource_size(), 0x8_000);
        assert_eq!(accounting.used_system_resource_size(), 0);

        assert!(accounting.can_resize_heap(0x20_000));
        accounting.commit_heap_size(0x20_000);
        accounting.commit_normal_memory(0x1_000);
        assert_eq!(accounting.heap_size(), 0x20_000);
        assert_eq!(
            accounting.used_non_system_user_physical_memory_size(),
            0x71_000
        );
        assert_eq!(accounting.used_user_physical_memory_size(), 0x79_000);

        accounting.release_normal_memory(0x1_000);
        accounting.commit_heap_size(0);
        assert_eq!(
            accounting.used_non_system_user_physical_memory_size(),
            0x50_000
        );
    }

    #[test]
    fn construction_and_growth_enforce_the_shared_physical_limit() {
        assert!(ProcessMemoryAccounting::new(0x50_000, 0x50_000, 0x20_000, 0x10_000, 1).is_none());

        let accounting =
            ProcessMemoryAccounting::new(0x60_000, 0x50_000, 0x20_000, 0x10_000, 0x8_000).unwrap();
        assert!(accounting.can_commit_normal_memory(0x8_000));
        assert!(!accounting.can_commit_normal_memory(0x8_001));
        assert!(!accounting.can_resize_heap(0x8_001));
    }
}
