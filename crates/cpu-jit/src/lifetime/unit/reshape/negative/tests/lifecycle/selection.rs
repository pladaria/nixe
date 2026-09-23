use super::*;

fn page(pc: u64, space: u64) -> Owner {
    let mut key = key(pc);
    key.address_space = AddressSpaceId::new(space);
    Owner::SelectionPage(SelectionPage::of(key))
}

#[test]
fn selection_pages_deduplicate_without_enlarging_unit_identity_payloads() {
    assert!(size_of::<SelectionPage>() <= size_of::<UnitHandle>());
    let process = process();
    let key = boundary(&process, 0);
    let record = record(&process, key, &[page(0, 1), page(4, 1), page(4092, 1)]).unwrap();
    assert_eq!(record.associations.len(), 1);
    assert_ne!(page(0, 1), page(4096, 1));
    assert_ne!(page(0, 1), page(0, 2));
}

#[test]
fn lcq_demand_invalidates_only_its_address_space_page_not_empty_reservations() {
    let process = process();
    let a = boundary(&process, 0);
    let b = boundary(&process, 16);
    let c = boundary(&process, 32);
    install(&process, a, &[page(0, 1)]);
    install(&process, b, &[page(4096, 1)]);
    install(&process, c, &[page(0, 2)]);
    let empty = process.reserve(key(1024)).unwrap();
    assert!(present(&process, a)); // No published LCQ input yet.
    process.retire_dispatch(empty).unwrap();
    assert!(present(&process, a));
    image(&process, 2048, 1, 1);
    assert!(!present(&process, a));
    assert!(present(&process, b));
    assert!(present(&process, c));
    assert!(process.lock().units.negatives.removed.is_none());
}

#[test]
fn lcq_unlink_revokes_page_selection_without_a_unit_association() {
    let process = process();
    let unit = image(&process, 0x1800, 1, 1);
    let key = boundary(&process, 0);
    install(&process, key, &[page(0x1000, 1)]);
    process.retire_unit(unit).unwrap();
    drain(&process);
    assert!(!present(&process, key));
    assert!(process.lock().units.negatives.removed.is_none());
}

#[test]
fn membership_without_a_public_entry_invalidates_on_publication_and_unlink() {
    let process = process();
    publish(&process, &AtomicU64::new(0), &[0x1000], Tier::Lcq);
    publish(&process, &AtomicU64::new(0), &[0x2000], Tier::Lcq);
    let watched = boundary(&process, 0);
    let other = boundary(&process, 16);
    install(&process, watched, &[page(0x2000, 1)]);
    install(&process, other, &[page(0x3000, 1)]);
    let mut candidate = input(&process, &[0x1000, 0x2000], Tier::Hcq);
    candidate.entries = candidate.entries.into_vec().into_iter().take(1).collect();
    let family = process
        .prepare_unit(
            &[process.reserve(key(0x1000)).unwrap()],
            candidate,
            &AtomicU64::new(0),
        )
        .unwrap()
        .publish()
        .unwrap();
    process.try_service_links().unwrap();
    assert!(!present(&process, watched)); // 0x2000 is only coverage, not a published entry.
    assert!(present(&process, other));
    install(&process, watched, &[page(0x2000, 1)]);
    process.retire_unit(family).unwrap();
    drain(&process);
    assert!(!present(&process, watched));
    assert!(present(&process, other));
    assert!(process.lock().units.negatives.removed.is_none());
}
