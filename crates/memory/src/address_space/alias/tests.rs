use super::*;

#[test]
fn alias_validation_never_changes_a_prefix_or_unmaps_an_unrelated_replacement() {
    let machine = Machine::new(3 * PAGE_SIZE).unwrap();
    let account = machine.account(3 * PAGE_SIZE).unwrap();
    let original = account.allocate(2 * PAGE_SIZE).unwrap();
    let replacement = account.allocate(PAGE_SIZE).unwrap();
    let space = AddressSpace::new(machine, AddressWidth::Bits32).unwrap();
    let source = GuestVirtualAddress::new(PAGE_SIZE);
    let destination = GuestVirtualAddress::new(0x10000);
    space
        .map(
            source,
            original.clone(),
            0,
            2 * PAGE_SIZE,
            MemoryPermissions::READ_WRITE,
        )
        .unwrap();
    let conflict = GuestVirtualAddress::new(destination.get() + PAGE_SIZE);
    space
        .map(
            conflict,
            replacement.clone(),
            0,
            PAGE_SIZE,
            MemoryPermissions::READ_WRITE,
        )
        .unwrap();
    assert!(
        space
            .map_alias(
                source,
                destination,
                2 * PAGE_SIZE,
                MemoryPermissions::NONE,
                MemoryPermissions::READ_WRITE
            )
            .is_err()
    );
    assert!(
        space
            .translate(destination, MemoryPermissions::READ)
            .is_err()
    );
    assert!(
        space
            .translate(source, MemoryPermissions::READ_WRITE)
            .is_ok()
    );
    space.unmap(conflict, PAGE_SIZE).unwrap();
    space
        .map_alias(
            source,
            destination,
            2 * PAGE_SIZE,
            MemoryPermissions::NONE,
            MemoryPermissions::READ_WRITE,
        )
        .unwrap();
    space
        .remap(
            conflict,
            replacement,
            0,
            PAGE_SIZE,
            MemoryPermissions::READ_WRITE,
        )
        .unwrap();
    assert!(
        space
            .unmap_alias(
                source,
                destination,
                2 * PAGE_SIZE,
                MemoryPermissions::READ_WRITE
            )
            .is_err()
    );
    assert!(space.translate(source, MemoryPermissions::READ).is_err());
    assert!(
        space
            .translate(destination, MemoryPermissions::READ_WRITE)
            .is_ok()
    );
    assert!(
        space
            .translate(conflict, MemoryPermissions::READ_WRITE)
            .is_ok()
    );
    space
        .remap(
            conflict,
            original,
            PAGE_SIZE,
            PAGE_SIZE,
            MemoryPermissions::READ_WRITE,
        )
        .unwrap();
    space
        .unmap_alias(
            source,
            destination,
            2 * PAGE_SIZE,
            MemoryPermissions::READ_WRITE,
        )
        .unwrap();
    assert_eq!(account.used_bytes(), 2 * PAGE_SIZE);
}
