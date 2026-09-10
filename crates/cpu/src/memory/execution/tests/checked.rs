use super::*;
use crate::memory::MemoryAccessSize;

const SPACE: AddressSpaceId = AddressSpaceId::new(1);
const START: GuestVirtualAddress = GuestVirtualAddress::new(0x1ffe);
const OLD_ALIAS: GuestVirtualAddress = GuestVirtualAddress::new(0x5ffe);

fn crossing_access() -> MemoryAccess {
    MemoryAccess::new(
        MemoryAccessSize::Word,
        MemoryAlignment::Unaligned,
        crate::memory::MemoryOrdering::Relaxed,
        MemoryAccessClass::Normal,
    )
}

fn memory(first: u64, second: u64) -> Arc<ExecutionMemory> {
    let mut memory = ExecutionMemory::new();
    for id in [1, 2] {
        let page = GuestPhysicalPageId::new(id);
        assert!(memory.add_ram_page(page));
        memory
            .initialize_ram(page, 0, &vec![id as u8 * 0x11; 4096])
            .unwrap();
    }
    for (address, page) in [(0x1000, first), (0x2000, second), (0x5000, first)] {
        assert!(memory.map_page(
            SPACE,
            GuestVirtualAddress::new(address),
            GuestPhysicalPageId::new(page),
            MemoryPermissions::READ_WRITE
        ));
    }
    Arc::new(memory)
}

struct Download {
    memory: std::sync::Weak<ExecutionMemory>,
    fail: bool,
}

impl VisibilityCoordinator for Download {
    fn make_device_visible(
        &self,
        _: DeviceVisibilityRequest,
        _: &[u8],
    ) -> Result<(), VisibilityCoordinatorError> {
        Ok(())
    }

    fn make_cpu_visible(
        &self,
        _: CpuVisibilityRequest,
    ) -> Result<Box<[u8]>, VisibilityCoordinatorError> {
        let memory = self.memory.upgrade().unwrap();
        assert!(
            memory.inner.try_lock().is_ok(),
            "checked RAM reconciliation retained the mapping lock"
        );
        // This also takes the first page's lock. Both page guards must be gone
        // and no first fragment may have been written before reconciliation.
        assert_eq!(
            memory
                .read(
                    SPACE,
                    OLD_ALIAS,
                    MemoryAccess::normal(MemoryAccessSize::Halfword)
                )
                .unwrap()
                .value,
            MemoryValue::U16(0x1111)
        );
        if self.fail {
            return Err(VisibilityCoordinatorError::new(
                "checked RAM download rejected",
            ));
        }
        let base = GuestVirtualAddress::new(0x1000);
        memory
            .resize_zeroed_mapping(
                SPACE,
                base,
                4096,
                0,
                MemoryPermissions::READ_WRITE,
                MemoryMappingPurpose::Normal,
            )
            .unwrap();
        memory
            .resize_zeroed_mapping(
                SPACE,
                base,
                0,
                4096,
                MemoryPermissions::READ_WRITE,
                MemoryMappingPurpose::Normal,
            )
            .unwrap();
        memory
            .overwrite_mapped_ram(SPACE, START, &[0x66; 2])
            .unwrap();
        Ok(vec![0x77; 4096].into_boxed_slice())
    }
}

fn device_owned_second_page(memory: &Arc<ExecutionMemory>, fail: bool) {
    let range = memory
        .translate_canonical_range(
            SPACE,
            GuestVirtualAddress::new(0x2000),
            4096,
            MemoryPermissions::READ_WRITE,
        )
        .unwrap();
    let device: Arc<dyn VisibilityCoordinator> = Arc::new(Download {
        memory: Arc::downgrade(memory),
        fail,
    });
    let declaration = DeviceAccessDeclaration::write(
        NonCpuDeviceId::new(1),
        DeviceVisibilityPoint::new(1),
        DeviceVisibilityPoint::new(2),
    )
    .unwrap();
    range
        .prepare_device_access(declaration, device.clone())
        .unwrap();
    range.publish_device_write(declaration, device).unwrap();
}

#[test]
fn checked_cross_page_access_retranslates_after_device_callback_remaps_the_first_page() {
    for write in [false, true] {
        let memory = memory(1, 2);
        device_owned_second_page(&memory, false);
        let access = crossing_access();
        if write {
            memory
                .write(SPACE, START, access, MemoryValue::U32(0xaabbccdd))
                .unwrap();
        } else {
            assert_eq!(
                memory.read(SPACE, START, access).unwrap().value,
                MemoryValue::U32(0x77776666)
            );
        }
        assert_eq!(
            memory
                .read(
                    SPACE,
                    OLD_ALIAS,
                    MemoryAccess::normal(MemoryAccessSize::Halfword)
                )
                .unwrap()
                .value,
            MemoryValue::U16(0x1111)
        );
        assert_eq!(
            memory.read(SPACE, START, access).unwrap().value,
            MemoryValue::U32(if write { 0xaabbccdd } else { 0x77776666 })
        );
    }
}

#[test]
fn failed_second_page_download_keeps_first_store_fragment_and_original_diagnostic() {
    for write in [false, true] {
        let memory = memory(1, 2);
        device_owned_second_page(&memory, true);
        let access = crossing_access();
        let error = if write {
            memory
                .write(SPACE, START, access, MemoryValue::U32(0xaabbccdd))
                .unwrap_err()
        } else {
            memory.read(SPACE, START, access).unwrap_err()
        };
        assert_eq!(error.address, START);
        assert!(
            matches!(error.reason, DataAccessFaultReason::HostBacking(ref detail) if detail.contains("checked RAM download rejected"))
        );
        assert_eq!(
            memory
                .read(
                    SPACE,
                    OLD_ALIAS,
                    MemoryAccess::normal(MemoryAccessSize::Halfword)
                )
                .unwrap()
                .value,
            MemoryValue::U16(0x1111)
        );
    }
}

#[test]
fn checked_cross_page_access_locks_reverse_order_and_duplicate_physical_aliases_once() {
    for (first, second) in [(2, 1), (1, 1)] {
        let memory = memory(first, second);
        let access = crossing_access();
        memory
            .write(SPACE, START, access, MemoryValue::U32(0xaabbccdd))
            .unwrap();
        assert_eq!(
            memory.read(SPACE, START, access).unwrap().value,
            MemoryValue::U32(0xaabbccdd)
        );
        assert_eq!(
            memory
                .read(
                    SPACE,
                    OLD_ALIAS,
                    MemoryAccess::normal(MemoryAccessSize::Halfword)
                )
                .unwrap()
                .value,
            MemoryValue::U16(0xccdd)
        );
    }
}
