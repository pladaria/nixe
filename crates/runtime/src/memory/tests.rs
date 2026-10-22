use super::*;
use nixe_memory::{Access, Machine};

const RW: Mapping = Mapping {
    kind: MemoryKind::Heap,
    permissions: MemoryPermissions::READ_WRITE,
    uncached: false,
    locked: false,
    permission_locked: false,
};
fn address(page: usize) -> GuestVirtualAddress {
    GuestVirtualAddress::new(page as u64 * PAGE_SIZE)
}

#[test]
fn kernel_queries_follow_an_independent_page_state_oracle_without_page_metadata() {
    let machine = Machine::new(32 * PAGE_SIZE).unwrap();
    let account = machine.account(32 * PAGE_SIZE).unwrap();
    let memory = ProcessMemory::new(
        account.clone(),
        machine.account(machine.arena().capacity()).unwrap(),
        AddressWidth::Bits32,
        crate::ProcessVirtualRegion::new(GuestVirtualAddress::new(0x1000000), 0x400000),
        0,
    )
    .unwrap();
    let mut expected = [None; 34];
    let mut random = 71u64;
    for _ in 0..1000 {
        random = random.wrapping_mul(6364136223846793005).wrapping_add(1);
        let page = 1 + ((random >> 32) as usize % 32);
        match (expected[page], random & 3) {
            (None, _) => {
                memory
                    .map(
                        address(page),
                        account.allocate(PAGE_SIZE).unwrap(),
                        0,
                        PAGE_SIZE,
                        RW,
                    )
                    .unwrap();
                expected[page] = Some(RW);
            }
            (Some(mapping), 0) => {
                memory.unmap(address(page), PAGE_SIZE, mapping).unwrap();
                expected[page] = None;
            }
            (Some(mut mapping), 1) => {
                let permissions = if mapping.permissions == MemoryPermissions::READ {
                    MemoryPermissions::READ_WRITE
                } else {
                    MemoryPermissions::READ
                };
                let result = memory.protect(address(page), PAGE_SIZE, permissions);
                if mapping.uncached {
                    assert!(matches!(result, Err(MappingError::StateMismatch)));
                } else {
                    result.unwrap();
                    mapping.permissions = permissions;
                    expected[page] = Some(mapping);
                }
            }
            (Some(mut mapping), _) => {
                mapping.uncached = !mapping.uncached;
                memory
                    .set_uncached(address(page), PAGE_SIZE, mapping.uncached)
                    .unwrap();
                expected[page] = Some(mapping);
            }
        }
        assert_eq!(
            account.used_bytes(),
            expected.iter().filter(|mapping| mapping.is_some()).count() as u64 * PAGE_SIZE
        );
        for query in 0..34 {
            let info = memory.query(address(query)).unwrap();
            assert_eq!(info.mapping, expected[query]);
            let mut first = query;
            let mut last = query + 1;
            while first != 0 && expected[first - 1] == expected[query] {
                first -= 1;
            }
            while last < 34 && expected[last] == expected[query] {
                last += 1;
            }
            assert_eq!(info.base, address(first));
            assert_eq!(
                info.bytes,
                if last == 34 {
                    (1u64 << 32) - info.base.get()
                } else {
                    (last - first) as u64 * PAGE_SIZE
                }
            );
        }
    }
}

#[test]
fn shared_system_allocation_maps_into_two_processes_without_duplicate_storage_or_charge() {
    let machine = Machine::new(4 * PAGE_SIZE).unwrap();
    let system = machine.account(2 * PAGE_SIZE).unwrap();
    let first = ProcessMemory::new(
        machine.account(PAGE_SIZE).unwrap(),
        machine.account(machine.arena().capacity()).unwrap(),
        AddressWidth::Bits32,
        crate::ProcessVirtualRegion::new(GuestVirtualAddress::new(0x1000000), 0x400000),
        0,
    )
    .unwrap();
    let second = ProcessMemory::new(
        machine.account(PAGE_SIZE).unwrap(),
        machine.account(machine.arena().capacity()).unwrap(),
        AddressWidth::Bits36,
        crate::ProcessVirtualRegion::new(GuestVirtualAddress::new(0x1000000), 0x400000),
        0,
    )
    .unwrap();
    let allocation = system.allocate(PAGE_SIZE).unwrap();
    let shared = Mapping {
        kind: MemoryKind::Shared,
        ..RW
    };
    let remote = Mapping {
        permissions: MemoryPermissions::READ,
        ..shared
    };
    first
        .map(address(1), allocation.clone(), 0, PAGE_SIZE, shared)
        .unwrap();
    second
        .map(address(2), allocation, 0, PAGE_SIZE, remote)
        .unwrap();
    assert_eq!(system.used_bytes(), PAGE_SIZE);
    assert_eq!(first.account().used_bytes(), 0);
    assert_eq!(second.account().used_bytes(), 0);
    assert!(matches!(
        first
            .space()
            .write_value(address(1), 4, 0x12345678)
            .unwrap(),
        Access::Ready(())
    ));
    assert!(matches!(
        second.space().read_value(address(2), 4).unwrap(),
        Access::Ready(0x12345678)
    ));
    assert!(second.space().write_value(address(2), 4, 1).is_err());
    first.unmap(address(1), PAGE_SIZE, shared).unwrap();
    assert_eq!(system.used_bytes(), PAGE_SIZE);
    second.unmap(address(2), PAGE_SIZE, remote).unwrap();
    assert_eq!(system.used_bytes(), 0);
    assert_eq!(machine.statistics().allocated_bytes, 0);
}

#[test]
fn kernel_alias_states_restore_and_locked_mappings_reject_guest_reprotection() {
    let machine = Machine::new(2 * PAGE_SIZE).unwrap();
    let account = machine.account(2 * PAGE_SIZE).unwrap();
    let memory = ProcessMemory::new(
        account.clone(),
        machine.account(machine.arena().capacity()).unwrap(),
        AddressWidth::Bits32,
        crate::ProcessVirtualRegion::new(GuestVirtualAddress::new(0x1000000), 0x400000),
        0,
    )
    .unwrap();
    memory
        .map(
            address(1),
            account.allocate(2 * PAGE_SIZE).unwrap(),
            0,
            2 * PAGE_SIZE,
            RW,
        )
        .unwrap();
    let locked = Mapping {
        permissions: MemoryPermissions::NONE,
        locked: true,
        ..RW
    };
    let alias = Mapping {
        kind: MemoryKind::Stack,
        ..RW
    };
    memory
        .map_alias(address(1), address(8), PAGE_SIZE, RW, locked, alias)
        .unwrap();
    assert_eq!(memory.query(address(1)).unwrap().mapping, Some(locked));
    assert_eq!(memory.query(address(8)).unwrap().mapping, Some(alias));
    assert!(matches!(
        memory.protect(address(1), PAGE_SIZE, MemoryPermissions::READ_WRITE),
        Err(MappingError::PermissionLocked)
    ));
    assert!(matches!(
        memory.set_uncached(address(1), PAGE_SIZE, true),
        Err(MappingError::PermissionLocked)
    ));
    assert!(matches!(
        memory.protect(address(8), PAGE_SIZE, MemoryPermissions::READ),
        Err(MappingError::StateMismatch)
    ));
    assert!(matches!(
        memory.unmap_alias(address(1), address(8), PAGE_SIZE, RW, RW, alias),
        Err(MappingError::StateMismatch)
    ));
    assert!(matches!(
        memory.space().write_value(address(8), 4, 91).unwrap(),
        Access::Ready(())
    ));
    memory
        .unmap_alias(address(1), address(8), PAGE_SIZE, locked, RW, alias)
        .unwrap();
    assert!(memory.query(address(8)).unwrap().mapping.is_none());
    assert_eq!(memory.query(address(1)).unwrap().bytes, 2 * PAGE_SIZE);
    assert!(matches!(
        memory.space().read_value(address(1), 4).unwrap(),
        Access::Ready(91)
    ));
    assert_eq!(account.used_bytes(), 2 * PAGE_SIZE);
}

#[test]
fn shrinking_a_heap_releases_a_suffix_of_an_earlier_large_allocation() {
    const UNIT: u64 = 2 * 1024 * 1024;
    let machine = Machine::new(2 * UNIT).unwrap();
    let account = machine.account(2 * UNIT).unwrap();
    let memory = ProcessMemory::new(
        account.clone(),
        machine.account(machine.arena().capacity()).unwrap(),
        AddressWidth::Bits32,
        crate::ProcessVirtualRegion::new(GuestVirtualAddress::new(0x1000000), 0x400000),
        0,
    )
    .unwrap();
    let base = GuestVirtualAddress::new(UNIT);
    memory
        .map(base, account.allocate(2 * UNIT).unwrap(), 0, 2 * UNIT, RW)
        .unwrap();
    memory
        .unmap(base.checked_add(UNIT).unwrap(), UNIT, RW)
        .unwrap();
    assert_eq!(
        account.used_bytes(),
        UNIT,
        "unmapped heap bytes cannot remain charged to the guest"
    );
    assert_eq!(machine.statistics().allocated_bytes, UNIT);
    let replacement = account.allocate(UNIT).unwrap();
    assert_eq!(account.used_bytes(), 2 * UNIT);
    drop(replacement);
    memory.close().unwrap();
    assert_eq!(machine.statistics().allocated_bytes, 0);
}

#[test]
fn transfer_memory_locks_original_pages_until_the_last_object_and_restores_fragmented_mappings() {
    use crate::TransferMemoryObject;
    let machine = Machine::new(4 * PAGE_SIZE).unwrap();
    let account = machine.account(4 * PAGE_SIZE).unwrap();
    let owner = Arc::new(
        ProcessMemory::new(
            account.clone(),
            machine.account(4 * PAGE_SIZE).unwrap(),
            AddressWidth::Bits32,
            crate::ProcessVirtualRegion::new(GuestVirtualAddress::new(0x1000000), 0x400000),
            0,
        )
        .unwrap(),
    );
    let allocation = account.allocate(2 * PAGE_SIZE).unwrap();
    owner
        .map(address(1), allocation.clone(), 0, PAGE_SIZE, RW)
        .unwrap();
    owner
        .map(address(2), allocation, PAGE_SIZE, PAGE_SIZE, RW)
        .unwrap();
    assert!(matches!(
        owner.space().write_value(address(1), 4, 0x1234).unwrap(),
        Access::Ready(())
    ));
    let transfer = TransferMemoryObject::new(
        owner.clone(),
        address(1),
        2 * PAGE_SIZE,
        MemoryPermissions::NONE,
    )
    .unwrap();
    assert!(owner.space().read_value(address(1), 4).is_err());
    assert!(matches!(
        owner.protect(address(1), PAGE_SIZE, MemoryPermissions::READ_WRITE),
        Err(MappingError::PermissionLocked)
    ));
    assert!(owner.unmap(address(1), PAGE_SIZE, RW).is_err());
    let duplicate = transfer.clone();
    drop(transfer);
    assert!(owner.space().read_value(address(1), 4).is_err());
    drop(duplicate);
    assert_eq!(owner.query(address(1)).unwrap().mapping, Some(RW));
    assert!(matches!(
        owner.space().read_value(address(1), 4).unwrap(),
        Access::Ready(0x1234)
    ));
    assert_eq!(account.used_bytes(), 2 * PAGE_SIZE);
    let transfer = TransferMemoryObject::new(
        owner.clone(),
        address(1),
        2 * PAGE_SIZE,
        MemoryPermissions::READ,
    )
    .unwrap();
    owner.close().unwrap();
    assert_eq!(account.used_bytes(), 0);
    assert_eq!(machine.statistics().allocated_bytes, 2 * PAGE_SIZE);
    drop(transfer);
    assert_eq!(machine.statistics().allocated_bytes, 0);
}

#[test]
fn shared_unmap_checks_the_handle_allocation_before_removing_any_mapping() {
    let machine = Machine::new(2 * PAGE_SIZE).unwrap();
    let system = machine.account(2 * PAGE_SIZE).unwrap();
    let memory = ProcessMemory::new(
        machine.account(PAGE_SIZE).unwrap(),
        system.clone(),
        AddressWidth::Bits32,
        crate::ProcessVirtualRegion::new(GuestVirtualAddress::new(0x1000000), 0x400000),
        0,
    )
    .unwrap();
    let shared = Mapping {
        kind: MemoryKind::Shared,
        ..RW
    };
    let first = system.allocate(PAGE_SIZE).unwrap();
    let other = system.allocate(PAGE_SIZE).unwrap();
    memory
        .map(address(1), first.clone(), 0, PAGE_SIZE, shared)
        .unwrap();
    assert!(matches!(
        memory.unmap_shared(address(1), &other, PAGE_SIZE, shared),
        Err(MappingError::PhysicalIdentityMismatch)
    ));
    assert_eq!(
        memory
            .space()
            .capture(address(1), 1, MemoryPermissions::READ)
            .unwrap()[0]
            .allocation()
            .identity(),
        first.identity()
    );
    memory
        .unmap_shared(address(1), &first, PAGE_SIZE, shared)
        .unwrap();
    assert!(memory.query(address(1)).unwrap().mapping.is_none());
    assert_eq!(system.used_bytes(), 2 * PAGE_SIZE);
    drop(first);
    drop(other);
    assert_eq!(machine.statistics().allocated_bytes, 0);
}

#[test]
fn permanent_code_permission_locks_are_distinct_from_temporary_transfer_locks() {
    let machine = Machine::new(2 * PAGE_SIZE).unwrap();
    let account = machine.account(2 * PAGE_SIZE).unwrap();
    let memory = Arc::new(
        ProcessMemory::new(
            account.clone(),
            machine.account(2 * PAGE_SIZE).unwrap(),
            AddressWidth::Bits32,
            crate::ProcessVirtualRegion::new(GuestVirtualAddress::new(0x1000000), 0x400000),
            0,
        )
        .unwrap(),
    );
    let code = Mapping {
        kind: MemoryKind::MutableCode,
        ..RW
    };
    memory
        .map(
            address(1),
            account.allocate(PAGE_SIZE).unwrap(),
            0,
            PAGE_SIZE,
            code,
        )
        .unwrap();
    memory
        .map(
            address(2),
            account.allocate(PAGE_SIZE).unwrap(),
            0,
            PAGE_SIZE,
            RW,
        )
        .unwrap();
    assert!(matches!(
        memory.lock_permissions(address(2), PAGE_SIZE),
        Err(MappingError::StateMismatch)
    ));
    memory.lock_permissions(address(1), PAGE_SIZE).unwrap();
    let state = memory.query(address(1)).unwrap().mapping.unwrap();
    assert!(state.permission_locked);
    assert!(!state.locked);
    assert!(matches!(
        memory.protect(address(1), PAGE_SIZE, MemoryPermissions::READ),
        Err(MappingError::PermissionLocked)
    ));
    assert!(
        crate::TransferMemoryObject::new(
            memory.clone(),
            address(1),
            PAGE_SIZE,
            MemoryPermissions::NONE
        )
        .is_err()
    );
    memory.set_uncached(address(1), PAGE_SIZE, true).unwrap();
    assert!(
        memory
            .query(address(1))
            .unwrap()
            .mapping
            .unwrap()
            .permission_locked
    );
    memory.close().unwrap();
    assert_eq!(machine.statistics().allocated_bytes, 0);
}

#[test]
fn a_small_accepted_output_retains_only_its_original_page_after_unmapping() {
    let machine = Machine::new(3 * PAGE_SIZE).unwrap();
    let account = machine.account(3 * PAGE_SIZE).unwrap();
    let memory = ProcessMemory::new(
        account.clone(),
        machine.account(3 * PAGE_SIZE).unwrap(),
        AddressWidth::Bits32,
        crate::ProcessVirtualRegion::new(address(16), 4 * PAGE_SIZE),
        0,
    )
    .unwrap();
    memory
        .map(
            address(1),
            account.allocate(3 * PAGE_SIZE).unwrap(),
            0,
            3 * PAGE_SIZE,
            RW,
        )
        .unwrap();
    let original = memory
        .space()
        .capture(address(1), 3 * PAGE_SIZE, MemoryPermissions::WRITE)
        .unwrap();
    let selected = nixe_memory::CapturedMemory::new(
        nixe_memory::capture_subrange(&original, PAGE_SIZE + 7, 5).unwrap(),
    )
    .unwrap();
    assert_eq!(
        selected[0].mapping().unwrap().start,
        address(2).checked_add(7).unwrap()
    );
    assert_eq!(
        selected[0].mapping().unwrap().generation,
        original[0].mapping().unwrap().generation
    );
    drop(original);
    memory.close().unwrap();
    assert_eq!(account.used_bytes(), 0);
    assert_eq!(machine.statistics().allocated_bytes, PAGE_SIZE);
    assert!(matches!(
        machine
            .write_captured(&selected, 0, &[1, 2, 3, 4, 5])
            .unwrap(),
        Access::Ready(())
    ));
    drop(selected);
    assert_eq!(machine.statistics().allocated_bytes, 0);
}
