use super::*;
use std::sync::{Barrier, atomic::AtomicUsize};

#[test]
fn alias_batch_changes_both_views_once_and_preserves_allocation_identity_and_charge() {
    let machine = Machine::new(3 * PAGE_SIZE).unwrap();
    let account = machine.account(3 * PAGE_SIZE).unwrap();
    let space = AddressSpace::new(machine.clone(), AddressWidth::Bits32).unwrap();
    let source = GuestVirtualAddress::new(PAGE_SIZE);
    let destination = GuestVirtualAddress::new(0x10000);
    let mut allocations = Vec::new();
    for offset in [0, PAGE_SIZE] {
        let allocation = account.allocate(PAGE_SIZE).unwrap();
        space
            .map(
                GuestVirtualAddress::new(source.get() + offset),
                allocation.clone(),
                0,
                PAGE_SIZE,
                MemoryPermissions::READ_WRITE,
            )
            .unwrap();
        allocations.push(allocation);
    }
    let original = space
        .capture(source, 2 * PAGE_SIZE, MemoryPermissions::READ)
        .unwrap();
    let observed: Vec<_> = original
        .iter()
        .map(|range| {
            let observation = crate::Observation::default();
            machine
                .access_captured(
                    std::slice::from_ref(range),
                    crate::AccessOperation::ReadObserved(observation.clone()),
                )
                .unwrap()
                .into_ready()
                .unwrap();
            observation
        })
        .collect();
    let _mapping_watch = space.observe_mapping(&original[0], &observed[0]).unwrap();
    let interrupts = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let barrier = Arc::new(Barrier::new(2));
    let context = space.register_core({
        let interrupts = Arc::clone(&interrupts);
        let stop = Arc::clone(&stop);
        Arc::new(move || {
            interrupts.fetch_add(1, Ordering::Relaxed);
            stop.store(true, Ordering::Release);
        })
    });
    let worker = std::thread::spawn({
        let space = Arc::clone(&space);
        let stop = Arc::clone(&stop);
        let barrier = Arc::clone(&barrier);
        move || {
            let admission = context.enter_native().unwrap();
            barrier.wait();
            while !stop.load(Ordering::Acquire) {
                std::thread::yield_now();
            }
            drop(admission);
            let admission = context.enter_native().unwrap();
            assert!(
                space
                    .check_native_access(&admission, source, MemoryPermissions::READ)
                    .is_err()
            );
            assert!(
                space
                    .check_native_access(&admission, destination, MemoryPermissions::READ_WRITE)
                    .is_ok()
            );
        }
    });
    barrier.wait();
    space
        .map_alias(
            source,
            destination,
            2 * PAGE_SIZE,
            MemoryPermissions::NONE,
            MemoryPermissions::READ_WRITE,
        )
        .unwrap();
    worker.join().unwrap();
    assert_eq!(interrupts.load(Ordering::Relaxed), 1);
    assert!(observed[0].is_invalid());
    assert!(!observed[1].is_invalid());
    assert_eq!(account.used_bytes(), 2 * PAGE_SIZE);
    for offset in [0, PAGE_SIZE] {
        assert_eq!(
            space
                .translate(
                    GuestVirtualAddress::new(destination.get() + offset),
                    MemoryPermissions::READ
                )
                .unwrap()
                .allocation,
            original[(offset / PAGE_SIZE) as usize]
                .allocation()
                .identity()
        );
    }
    assert!(matches!(
        space
            .write_value(
                GuestVirtualAddress::new(destination.get() + PAGE_SIZE),
                4,
                0x12345678
            )
            .unwrap(),
        Access::Ready(())
    ));
    assert!(observed[1].is_invalid());
    assert!(space.read_value(source, 4).is_err());
    space
        .unmap_alias(
            source,
            destination,
            2 * PAGE_SIZE,
            MemoryPermissions::READ_WRITE,
        )
        .unwrap();
    assert!(space.read_value(destination, 4).is_err());
    assert!(matches!(
        space
            .read_value(GuestVirtualAddress::new(source.get() + PAGE_SIZE), 4)
            .unwrap(),
        Access::Ready(0x12345678)
    ));
    assert_eq!(account.used_bytes(), 2 * PAGE_SIZE);
    drop(original);
    drop(allocations);
    drop(space);
    assert_eq!(machine.statistics().allocated_bytes, 0);
}

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
