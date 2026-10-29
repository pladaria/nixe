use super::*;
use crate::physical::WriteDomain;
use nixe_memory::{
    Access, AddressSpace, AddressWidth, GuestVirtualAddress, MemoryPermissions, PAGE_SIZE,
};

#[test]
fn partial_retirement_keeps_live_producers_and_recycled_pages_have_an_independent_lifetime() {
    let machine = Machine::new(3 * PAGE_SIZE).unwrap();
    let account = machine.account(3 * PAGE_SIZE).unwrap();
    let allocation = account.allocate(2 * PAGE_SIZE).unwrap();
    let old_id = allocation.identity();
    let range = AllocationRange::new(allocation.pin(), 0, 2 * PAGE_SIZE).unwrap();
    let mapping = PhysicalMapping::new(std::slice::from_ref(&range)).unwrap();
    let start = mapping.spans()[0].start;
    let full = Footprint::new(
        Arc::clone(&mapping),
        WriteDomain::Buffer {
            start: 0,
            bytes: 2 * PAGE_SIZE,
        },
    )
    .unwrap();
    let prefix = Footprint::new(
        Arc::clone(&mapping),
        WriteDomain::Buffer { start: 0, bytes: 8 },
    )
    .unwrap();
    let space = AddressSpace::new(machine.clone(), AddressWidth::Bits32).unwrap();
    let base = GuestVirtualAddress::new(PAGE_SIZE);
    space
        .map(
            base,
            allocation,
            0,
            2 * PAGE_SIZE,
            MemoryPermissions::READ_WRITE,
        )
        .unwrap();
    let mut authority = Authority::new(machine.identity());
    authority.register(1, vec![Arc::clone(&full)]).unwrap();
    let whole_writer = authority.create_writer(1, Arc::clone(&full)).unwrap();
    let prefix_writer = authority.create_writer(1, Arc::clone(&prefix)).unwrap();
    authority
        .publish_write(&machine, whole_writer, || Ok(((), NativeTick(1))))
        .unwrap();
    authority
        .publish_write(&machine, prefix_writer, || Ok(((), NativeTick(2))))
        .unwrap();
    drop(range);
    space
        .unmap(base.checked_add(PAGE_SIZE).unwrap(), PAGE_SIZE)
        .unwrap();
    assert_eq!(machine.statistics().allocated_bytes, PAGE_SIZE);
    let appended = std::cell::Cell::new(false);
    assert!(
        authority
            .publish_write(&machine, whole_writer, || {
                appended.set(true);
                Ok(((), NativeTick(3)))
            })
            .is_err()
    );
    assert!(!appended.get());
    authority
        .publish_write(&machine, prefix_writer, || Ok(((), NativeTick(3))))
        .unwrap();
    let fresh = account.allocate(PAGE_SIZE).unwrap();
    let fresh_range = AllocationRange::new(fresh.pin(), 0, PAGE_SIZE).unwrap();
    let fresh_mapping = PhysicalMapping::new(std::slice::from_ref(&fresh_range)).unwrap();
    assert_eq!(
        fresh_mapping.spans()[0].start.get(),
        start.get() + PAGE_SIZE
    );
    assert!(
        authority
            .resolve(std::slice::from_ref(&fresh_range))
            .unwrap()
            .iter()
            .all(|part| part.producer.is_none())
    );
    let fresh_domain =
        Footprint::new(fresh_mapping, WriteDomain::Buffer { start: 0, bytes: 8 }).unwrap();
    authority
        .register(2, vec![Arc::clone(&fresh_domain)])
        .unwrap();
    let fresh_writer = authority.create_writer(2, fresh_domain).unwrap();
    authority
        .publish_write(&machine, fresh_writer, || Ok(((), NativeTick(4))))
        .unwrap();
    let suffix = nixe_memory::Extent {
        start: nixe_memory::ArenaOffset::new(start.get() + PAGE_SIZE),
        bytes: PAGE_SIZE,
    };
    authority
        .retire_physical_ranges(machine.identity(), old_id, &[suffix])
        .unwrap();
    let live = space.capture(base, 16, MemoryPermissions::READ).unwrap();
    let current = authority.resolve(&live).unwrap();
    assert_eq!(current.len(), 2);
    assert_eq!(current[0].producer.unwrap().writer, prefix_writer);
    assert_eq!(current[1].producer.unwrap().writer, whole_writer);
    authority
        .publish_readback(&machine, current[0], &[0x11; 8], &mut Vec::new())
        .unwrap();
    authority
        .publish_readback(&machine, current[1], &[0x22; 8], &mut Vec::new())
        .unwrap();
    authority
        .retire_physical_ranges(machine.identity(), old_id, &[suffix])
        .unwrap();
    assert_eq!(
        authority
            .resolve(std::slice::from_ref(&fresh_range))
            .unwrap()[0]
            .producer
            .unwrap()
            .resource,
        2
    );
    drop(live);
    space.close().unwrap();
    authority
        .retire_physical_ranges(
            machine.identity(),
            old_id,
            &[nixe_memory::Extent {
                start,
                bytes: PAGE_SIZE,
            }],
        )
        .unwrap();
    assert_eq!(authority.resource_count(), 1);
    drop((fresh_range, fresh));
    assert_eq!(machine.statistics().allocated_bytes, 0);
}

#[test]
fn a_completed_download_preserves_live_bytes_after_partial_allocation_retirement() {
    let machine = Machine::new(2 * PAGE_SIZE).unwrap();
    let account = machine.account(2 * PAGE_SIZE).unwrap();
    let allocation = account.allocate(2 * PAGE_SIZE).unwrap();
    let live = allocation.pin_range(0, PAGE_SIZE).unwrap();
    let range = AllocationRange::new(allocation.pin(), 0, 2 * PAGE_SIZE).unwrap();
    let mapping = PhysicalMapping::new(std::slice::from_ref(&range)).unwrap();
    let start = mapping.spans()[0].start;
    let footprint = Footprint::new(
        mapping,
        WriteDomain::Buffer {
            start: 0,
            bytes: 2 * PAGE_SIZE,
        },
    )
    .unwrap();
    let mut authority = Authority::new(machine.identity());
    authority.register(1u32, vec![footprint.clone()]).unwrap();
    let writer = authority.create_writer(1, footprint).unwrap();
    authority
        .publish_write(&machine, writer, || Ok(((), NativeTick(1))))
        .unwrap();
    let snapshot = authority.resolve(std::slice::from_ref(&range)).unwrap()[0];
    let id = allocation.identity();
    drop(range);
    drop(allocation);
    authority
        .retire_physical_ranges(
            machine.identity(),
            id,
            &[nixe_memory::Extent {
                start: ArenaOffset::new(start.get() + PAGE_SIZE),
                bytes: PAGE_SIZE,
            }],
        )
        .unwrap();
    let replacement = account.allocate(PAGE_SIZE).unwrap();
    let recycled = AllocationRange::new(replacement.pin(), 0, PAGE_SIZE).unwrap();
    machine
        .write_resolved(
            std::slice::from_ref(&recycled),
            &vec![33; PAGE_SIZE as usize],
        )
        .unwrap();
    let published = authority
        .publish_readback(
            &machine,
            snapshot,
            &vec![11; 2 * PAGE_SIZE as usize],
            &mut Vec::new(),
        )
        .unwrap();
    assert_eq!(
        published.iter().map(|span| span.bytes).sum::<u64>(),
        PAGE_SIZE
    );
    let mut bytes = vec![0; PAGE_SIZE as usize];
    machine
        .capture_stream(
            &[AllocationRange::new(live, 0, PAGE_SIZE).unwrap()],
            &mut bytes,
        )
        .unwrap();
    assert_eq!(bytes, vec![11; PAGE_SIZE as usize]);
    machine.capture_stream(&[recycled], &mut bytes).unwrap();
    assert_eq!(bytes, vec![33; PAGE_SIZE as usize]);
}

#[test]
fn evicting_materialized_bytes_does_not_resurrect_an_older_hidden_producer() {
    let machine = Machine::new(PAGE_SIZE).unwrap();
    let allocation = machine
        .account(PAGE_SIZE)
        .unwrap()
        .allocate(PAGE_SIZE)
        .unwrap();
    let range = AllocationRange::new(allocation.pin(), 0, 8).unwrap();
    let mapping = PhysicalMapping::new(std::slice::from_ref(&range)).unwrap();
    let domain = |start, bytes| {
        Footprint::new(
            Arc::clone(&mapping),
            crate::physical::WriteDomain::Buffer { start, bytes },
        )
        .unwrap()
    };
    let space = AddressSpace::new(machine.clone(), AddressWidth::Bits32).unwrap();
    let base = GuestVirtualAddress::new(PAGE_SIZE);
    space
        .map(
            base,
            allocation,
            0,
            PAGE_SIZE,
            MemoryPermissions::READ_WRITE,
        )
        .unwrap();
    let mut authority = Authority::new(machine.identity());
    for (id, start, bytes) in [(1, 0, 8), (2, 2, 4)] {
        let footprint = domain(start, bytes);
        authority
            .register(id, vec![Arc::clone(&footprint)])
            .unwrap();
        let writer = authority.create_writer(id, footprint).unwrap();
        authority
            .publish_write(&machine, writer, || Ok(((), NativeTick(id as u64))))
            .unwrap();
    }
    assert!(!authority.can_evict(1).unwrap());
    assert!(!authority.can_evict(2).unwrap());
    assert!(authority.evict_materialized(&machine, 2).is_err());
    let current = authority.resolve(std::slice::from_ref(&range)).unwrap();
    let middle = current
        .iter()
        .find(|part| part.producer.unwrap().resource == 2)
        .copied()
        .unwrap();
    authority
        .publish_readback(&machine, middle, &[2; 4], &mut Vec::new())
        .unwrap();
    assert!(authority.can_evict(2).unwrap());
    assert!(!authority.can_evict(1).unwrap());
    authority.evict_materialized(&machine, 2).unwrap();
    assert!(authority.can_evict(2).is_err());
    let current = authority.resolve(std::slice::from_ref(&range)).unwrap();
    assert_eq!(current.len(), 3);
    assert!(current[0].producer.is_some_and(|p| p.resource == 1));
    assert!(current[1].producer.is_none());
    assert!(current[2].producer.is_some_and(|p| p.resource == 1));
    for part in current.into_iter().filter(|part| part.producer.is_some()) {
        authority
            .publish_readback(&machine, part, &[1; 2], &mut Vec::new())
            .unwrap();
    }
    assert!(authority.can_evict(1).unwrap());
    authority.evict_materialized(&machine, 1).unwrap();
    assert_eq!(authority.resource_count(), 0);
    assert_eq!(authority.writer_count(), 0);
    assert!(matches!(
        space.read_value(base, 8).unwrap(),
        Access::Ready(0x0101020202020101)
    ));
}

#[test]
fn delayed_physical_retirement_cannot_discard_a_recycled_allocation() {
    let machine = Machine::new(PAGE_SIZE).unwrap();
    let account = machine.account(PAGE_SIZE).unwrap();
    let mut authority = Authority::new(machine.identity());
    let old = account.allocate(PAGE_SIZE).unwrap();
    let old_id = old.identity();
    let old_range = AllocationRange::new(old.pin(), 0, 8).unwrap();
    let extents = AllocationRange::new(old.pin(), 0, PAGE_SIZE)
        .unwrap()
        .physical_extents()
        .collect::<Vec<_>>();
    let old_footprint = Footprint::new(
        PhysicalMapping::new(std::slice::from_ref(&old_range)).unwrap(),
        WriteDomain::Buffer { start: 0, bytes: 8 },
    )
    .unwrap();
    authority
        .register(1, vec![Arc::clone(&old_footprint)])
        .unwrap();
    let old_writer = authority
        .create_writer(1, Arc::clone(&old_footprint))
        .unwrap();
    authority
        .publish_write(&machine, old_writer, || Ok(((), NativeTick(1))))
        .unwrap();
    let observed = authority.observe(1, old_footprint).unwrap();
    drop(old);
    assert!(
        authority
            .retire_physical_ranges(machine.identity(), old_id, &extents)
            .is_err()
    );
    drop(old_range);
    let new = account.allocate(PAGE_SIZE).unwrap();
    let new_range = AllocationRange::new(new.pin(), 0, 8).unwrap();
    assert_eq!(
        AllocationRange::new(new.pin(), 0, PAGE_SIZE)
            .unwrap()
            .physical_extents()
            .collect::<Vec<_>>(),
        extents
    );
    let new_footprint = Footprint::new(
        PhysicalMapping::new(std::slice::from_ref(&new_range)).unwrap(),
        WriteDomain::Buffer { start: 0, bytes: 8 },
    )
    .unwrap();
    authority
        .register(2, vec![Arc::clone(&new_footprint)])
        .unwrap();
    let new_writer = authority.create_writer(2, new_footprint).unwrap();
    authority
        .publish_write(&machine, new_writer, || Ok(((), NativeTick(2))))
        .unwrap();
    assert_eq!(
        authority
            .retire_physical_ranges(machine.identity(), old_id, &extents)
            .unwrap(),
        [1]
    );
    assert!(observed.is_invalid());
    assert_eq!(authority.resource_count(), 1);
    assert_eq!(authority.writer_count(), 1);
    assert!(
        authority
            .resolve(&[new_range])
            .unwrap()
            .iter()
            .all(|part| part.producer.is_some_and(|p| p.resource == 2))
    );
}

#[test]
fn buffer_relocation_preserves_versions_cpu_holes_and_revalidates_readbacks() {
    let machine = Machine::new(4 * PAGE_SIZE).unwrap();
    let allocation = machine
        .account(4 * PAGE_SIZE)
        .unwrap()
        .allocate(4 * PAGE_SIZE)
        .unwrap();
    let all = AllocationRange::new(allocation.pin(), 0, 4 * PAGE_SIZE).unwrap();
    let mapping = PhysicalMapping::new(&[all]).unwrap();
    let old = mapping.slice(PAGE_SIZE, PAGE_SIZE).unwrap();
    let full = |mapping: Arc<PhysicalMapping>| {
        Footprint::new(
            Arc::clone(&mapping),
            WriteDomain::Buffer {
                start: 0,
                bytes: mapping.bytes(),
            },
        )
        .unwrap()
    };
    let mut authority = Authority::new(machine.identity());
    authority.register(1, vec![full(Arc::clone(&old))]).unwrap();
    let writer = authority.create_writer(1, full(Arc::clone(&old))).unwrap();
    let version = authority
        .publish_write(&machine, writer, || Ok(((), NativeTick(1))))
        .unwrap()
        .1;
    let hole = AllocationRange::new(allocation.pin(), PAGE_SIZE + 16, 16).unwrap();
    authority.cpu_write(&[hole]).unwrap();
    let request = AllocationRange::new(allocation.pin(), PAGE_SIZE, 64).unwrap();
    let before = authority.resolve(std::slice::from_ref(&request)).unwrap();
    let observation = authority.observe(1, full(Arc::clone(&old))).unwrap();
    let alias = full(old);
    authority.register(3, vec![Arc::clone(&alias)]).unwrap();
    authority.register(2, vec![full(mapping)]).unwrap();
    authority.relocate_buffer(1, 2, NativeTick(4)).unwrap();
    assert_eq!(authority.resources[&2].modification, version.get());
    assert_eq!(authority.resources[&3].aliases, [2]);
    assert!(authority.resources_may_overlap(2, 3));
    assert!(authority.resources_may_overlap(3, 2));
    assert!(!authority.resources_may_overlap(2, 1));
    assert!(!authority.resources_may_overlap(3, 1));
    let native = authority.observe_native(3, alias).unwrap();
    assert!(observation.is_invalid());
    assert!(authority.mapping(1, 0).is_none());
    let after = authority.resolve(&[request]).unwrap();
    assert_eq!(after.len(), 3);
    assert!(after[1].producer.is_none());
    for part in [&after[0], &after[2]] {
        let producer = part.producer.unwrap();
        assert_eq!(producer.resource, 2);
        assert_eq!(producer.writer, writer);
        assert_eq!(producer.sequence, version);
        assert_eq!(producer.tick, NativeTick(4));
        assert_eq!(producer.native, PAGE_SIZE + part.logical);
    }
    assert!(authority.current_readback(before[0]).unwrap().is_empty());
    authority
        .publish_write(&machine, writer, || Ok(((), NativeTick(5))))
        .unwrap();
    assert!(native.is_invalid());
    assert_eq!(authority.live_record_count(), 1);
}
