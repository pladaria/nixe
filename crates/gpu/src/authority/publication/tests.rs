use super::*;
use crate::physical::{ByteRect, ImageFootprint, ImageStorage, WriteDomain};
use nixe_memory::{
    Access, AccessResult, AddressSpace, AddressWidth, DeviceAccessEndpoint, DeviceAccessRequest,
    GuestVirtualAddress, MemoryPermissions,
};
use std::sync::mpsc;

struct Endpoint(mpsc::Sender<DeviceAccessRequest>);
impl DeviceAccessEndpoint for Endpoint {
    fn submit(&self, request: DeviceAccessRequest) {
        self.0.send(request).unwrap();
    }
    fn cancel(&self, _: AccessRequestId) {}
    fn allocation_pressure(&self, _: nixe_memory::MachineId, _: u64) {}
    fn allocation_released(&self, _: MachineId, _: AllocationId) {}
    fn physical_retired(&self, _: MachineId, _: AllocationId, _: Box<[nixe_memory::Extent]>) {}
}

#[test]
fn partial_writers_ignore_retired_allocations_outside_their_exact_domain() {
    for image in [false, true] {
        let machine = Machine::new(3 * PAGE_SIZE).unwrap();
        let account = machine.account(3 * PAGE_SIZE).unwrap();
        let unused = account.allocate(PAGE_SIZE).unwrap();
        let written = account.allocate(2 * PAGE_SIZE).unwrap();
        let ranges = [
            AllocationRange::new(written.pin(), 0, PAGE_SIZE).unwrap(),
            AllocationRange::new(unused.pin(), 0, PAGE_SIZE).unwrap(),
            AllocationRange::new(written.pin(), PAGE_SIZE, PAGE_SIZE).unwrap(),
        ];
        let mapping = PhysicalMapping::new(&ranges).unwrap();
        let (full_domain, partial_domain) = if image {
            let full = ImageFootprint {
                depth_slice: 0,
                base: 0,
                columns: 1,
                rows: 2,
                packed_bytes: 4,
                aspect_offset: 0,
                aspect_bytes: 4,
                storage: ImageStorage::Pitch {
                    row_pitch: 2 * PAGE_SIZE,
                },
                native_base: 0,
                native_element_bytes: 4,
                native_row_pitch: 4,
                rect: ByteRect {
                    x: 0,
                    y: 0,
                    width: 1,
                    height: 2,
                },
            };
            (
                WriteDomain::Image(full),
                WriteDomain::Image(ImageFootprint {
                    aspect_offset: 3,
                    aspect_bytes: 1,
                    ..full
                }),
            )
        } else {
            (
                WriteDomain::Buffer {
                    start: 0,
                    bytes: 3 * PAGE_SIZE,
                },
                WriteDomain::Buffer {
                    start: 2 * PAGE_SIZE,
                    bytes: PAGE_SIZE,
                },
            )
        };
        let full = Footprint::new(mapping.clone(), full_domain).unwrap();
        let partial = Footprint::new(mapping.clone(), partial_domain).unwrap();
        let mut authority = Authority::new(machine.identity());
        for id in 1..=3 {
            authority.register(id, vec![full.clone()]).unwrap();
        }
        let writer = authority.create_writer(1, partial.clone()).unwrap();
        authority
            .publish_write(&machine, writer, || Ok(((), NativeTick(1))))
            .unwrap();
        let retired_id = unused.identity();
        let extents = ranges[1].physical_extents().collect::<Vec<_>>();
        drop(ranges);
        drop(unused);
        authority
            .retire_physical_ranges(machine.identity(), retired_id, &extents)
            .unwrap();
        assert!(mapping.pin(2 * PAGE_SIZE, PAGE_SIZE).is_ok());
        let new_writer = authority.create_writer(2, partial.clone()).unwrap();
        for id in [writer, new_writer] {
            authority
                .publish_write(&machine, id, || Ok(((), NativeTick(2))))
                .unwrap();
        }
        drop(written);
        let retired_writer = authority.create_writer(3, partial).unwrap();
        for id in [writer, new_writer, retired_writer] {
            assert!(
                authority
                    .publish_write::<()>(&machine, id, || panic!(
                        "retired written bytes must fail before recording"
                    ))
                    .is_err()
            );
        }
    }
}

#[test]
fn lifetime_subscriptions_use_the_sorted_union_without_broadening_each_writer() {
    let machine = Machine::new(6 * PAGE_SIZE).unwrap();
    let account = machine.account(6 * PAGE_SIZE).unwrap();
    let first = account.allocate(3 * PAGE_SIZE).unwrap();
    let second = account.allocate(3 * PAGE_SIZE).unwrap();
    let a = |page| AllocationRange::new(first.pin(), page * PAGE_SIZE, PAGE_SIZE).unwrap();
    let b = |page| AllocationRange::new(second.pin(), page * PAGE_SIZE, PAGE_SIZE).unwrap();
    // Reverse order, repeated aliases, interleaved allocations and adjacent
    // ranges merged across different writers all use the same final union.
    let mappings = [
        PhysicalMapping::new(&[a(2), a(0), a(2)]).unwrap(),
        PhysicalMapping::new(&[b(2), b(0)]).unwrap(),
        PhysicalMapping::new(&[b(1), a(1), b(0), a(0)]).unwrap(),
    ];
    let mut authority = Authority::new(machine.identity());
    let mut writers = Vec::new();
    for (id, mapping) in mappings.into_iter().enumerate() {
        let bytes = mapping.bytes();
        let footprint = Footprint::new(mapping, WriteDomain::Buffer { start: 0, bytes }).unwrap();
        authority.register(id, vec![footprint.clone()]).unwrap();
        writers.push(authority.create_writer(id, footprint).unwrap());
    }
    authority
        .publish_writes(&machine, &writers, || Ok(((), NativeTick(1))))
        .unwrap();
    let retained = [
        first.pin_range(0, PAGE_SIZE).unwrap(),
        first.pin_range(2 * PAGE_SIZE, PAGE_SIZE).unwrap(),
    ];
    drop(first);
    assert!(
        authority
            .publish_write::<()>(&machine, writers[2], || panic!(
                "the retired middle page must invalidate its writer"
            ))
            .is_err()
    );
    authority
        .publish_write(&machine, writers[0], || Ok(((), NativeTick(2))))
        .unwrap();
    drop(retained);
    for id in [writers[0], writers[2]] {
        assert!(
            authority
                .publish_write::<()>(&machine, id, || panic!(
                    "retired allocation must invalidate its own writers"
                ))
                .is_err()
        );
    }
    authority
        .publish_write(&machine, writers[1], || Ok(((), NativeTick(2))))
        .unwrap();
    drop(second);
    assert!(
        authority
            .publish_write::<()>(&machine, writers[1], || panic!(
                "the remaining writer must observe its own retirement"
            ))
            .is_err()
    );
}

#[test]
fn sliced_writers_depend_only_on_the_allocations_in_their_view() {
    let machine = Machine::new(2 * PAGE_SIZE).unwrap();
    let account = machine.account(2 * PAGE_SIZE).unwrap();
    let first = account.allocate(PAGE_SIZE).unwrap();
    let second = account.allocate(PAGE_SIZE).unwrap();
    let ranges = [
        AllocationRange::new(first.pin(), 0, PAGE_SIZE).unwrap(),
        AllocationRange::new(second.pin(), 0, PAGE_SIZE).unwrap(),
    ];
    let parent = PhysicalMapping::new(&ranges).unwrap();
    let view = parent.slice(PAGE_SIZE, PAGE_SIZE).unwrap();
    assert_eq!(view.allocations().len(), 1);
    assert_eq!(*view, *PhysicalMapping::new(&ranges[1..]).unwrap());
    let footprint = Footprint::new(
        view.clone(),
        WriteDomain::Buffer {
            start: 0,
            bytes: PAGE_SIZE,
        },
    )
    .unwrap();
    let mut authority = Authority::new(machine.identity());
    for resource in 1u32..=3 {
        authority
            .register(resource, vec![footprint.clone()])
            .unwrap();
    }
    let writer = authority.create_writer(1, footprint.clone()).unwrap();
    drop(ranges);
    drop(first);
    let new_writer = authority.create_writer(2, footprint.clone()).unwrap();
    authority
        .publish_write(&machine, writer, || Ok(((), NativeTick(1))))
        .unwrap();
    authority
        .publish_write(&machine, new_writer, || Ok(((), NativeTick(2))))
        .unwrap();
    assert!(view.pin(0, PAGE_SIZE).is_ok());
    drop(second);
    let retired_writer = authority.create_writer(3, footprint).unwrap();
    for id in [writer, retired_writer] {
        assert!(
            authority
                .publish_write::<()>(&machine, id, || panic!(
                    "retired input must fail before recording"
                ))
                .is_err()
        );
    }
}

#[test]
fn abandoned_memory_observations_do_not_outlive_their_consumers() {
    let machine = Machine::new(PAGE_SIZE).unwrap();
    let allocation = machine
        .account(PAGE_SIZE)
        .unwrap()
        .allocate(PAGE_SIZE)
        .unwrap();
    let range = AllocationRange::new(allocation.pin(), 0, PAGE_SIZE).unwrap();
    let footprint = Footprint::new(
        PhysicalMapping::new(std::slice::from_ref(&range)).unwrap(),
        WriteDomain::Buffer {
            start: 0,
            bytes: PAGE_SIZE,
        },
    )
    .unwrap();
    let mut authority = Authority::new(machine.identity());
    authority
        .register(1u32, vec![Arc::clone(&footprint)])
        .unwrap();
    let writer = authority.create_writer(1, footprint).unwrap();
    for _ in 0..1024 {
        let observation = nixe_memory::Observation::default();
        let lifetime = observation.weak_invalidation_flag();
        authority
            .watch_memory(std::slice::from_ref(&range), observation)
            .unwrap();
        assert!(
            lifetime.upgrade().is_none(),
            "the producer retained an abandoned consumer"
        );
    }
    assert!(authority.watched_write_pages(writer).unwrap().is_empty());
    authority.record_write(writer, NativeTick(1)).unwrap();

    let observed = nixe_memory::Observation::default();
    machine
        .access_captured(
            std::slice::from_ref(&range),
            nixe_memory::AccessOperation::ReadObserved(observed.clone()),
        )
        .unwrap()
        .into_ready()
        .unwrap();
    authority
        .watch_memory(std::slice::from_ref(&range), observed.clone())
        .unwrap();
    assert!(!authority.watched_write_pages(writer).unwrap().is_empty());
    authority
        .publish_write(&machine, writer, || {
            assert!(observed.is_invalid());
            Ok(((), NativeTick(2)))
        })
        .unwrap();
    assert!(authority.watched_write_pages(writer).unwrap().is_empty());
}

#[test]
fn collected_native_overwrites_preserve_gpu_neighbors_and_replayed_producer_order() {
    let machine = Machine::new(PAGE_SIZE).unwrap();
    let allocation = machine
        .account(PAGE_SIZE)
        .unwrap()
        .allocate(PAGE_SIZE)
        .unwrap();
    let range = AllocationRange::new(allocation.pin(), 0, PAGE_SIZE).unwrap();
    let mapping = PhysicalMapping::new(std::slice::from_ref(&range)).unwrap();
    let footprint = Footprint::new(
        mapping,
        WriteDomain::Buffer {
            start: 0,
            bytes: 64,
        },
    )
    .unwrap();
    let mut authority = Authority::new(machine.identity());
    authority
        .register(1u32, vec![Arc::clone(&footprint)])
        .unwrap();
    let writer = authority.create_writer(1, footprint).unwrap();
    authority
        .publish_write(&machine, writer, || Ok(((), NativeTick(1))))
        .unwrap();
    let snapshot = authority
        .resolve(std::slice::from_ref(&range.slice(0, 64).unwrap()))
        .unwrap()[0];
    let space = AddressSpace::new(machine.clone(), AddressWidth::Bits32).unwrap();
    space
        .map(
            GuestVirtualAddress::new(PAGE_SIZE),
            allocation,
            0,
            PAGE_SIZE,
            MemoryPermissions::READ_WRITE,
        )
        .unwrap();
    machine.enable_cpu_write_collection().unwrap();
    let core = space.register_core(Arc::new(|| {}));
    let native = core.enter_native().unwrap();
    assert!(
        space
            .write_native(&native, GuestVirtualAddress::new(PAGE_SIZE + 3), &[77, 78])
            .unwrap()
    );
    assert!(
        space
            .write_native(&native, GuestVirtualAddress::new(PAGE_SIZE + 8), &[79])
            .unwrap()
    );
    drop(native);
    let mut dirty = Vec::new();
    // A download already recorded before the CPU writes cannot restore its old
    // bytes over those stores, including bytes sharing a native transfer block.
    authority
        .publish_readback(&machine, snapshot, &[11; 64], &mut dirty)
        .unwrap();
    assert_eq!(
        dirty
            .iter()
            .map(|range| (range.offset(), range.bytes()))
            .collect::<Vec<_>>(),
        [(3, 2), (8, 1)]
    );
    assert!(machine.collect_cpu_writes().unwrap().is_empty());
    let mut actual = [0; 64];
    machine
        .capture_stream(&[range.slice(0, 64).unwrap()], &mut actual)
        .unwrap();
    let mut expected = [11; 64];
    expected[3..5].copy_from_slice(&[77, 78]);
    expected[8] = 79;
    assert_eq!(actual, expected);
    let resolved = authority.resolve(&[range.slice(0, 64).unwrap()]).unwrap();
    assert_eq!(
        resolved
            .iter()
            .filter(|r| r.producer.is_none())
            .map(|r| r.bytes)
            .sum::<u64>(),
        3
    );
    // The next native effect is later than the collected CPU writes and owns
    // its original domain again, without preserving stale CPU holes.
    authority
        .publish_write(&machine, writer, || Ok(((), NativeTick(2))))
        .unwrap();
    let resolved = authority.resolve(&[range.slice(0, 64).unwrap()]).unwrap();
    assert!(resolved.iter().all(|r| {
        r.producer
            .is_some_and(|p| p.tick == NativeTick(2) && !p.materialized)
    }));
    machine.disable_cpu_write_collection();
}

#[test]
fn native_publication_reuses_protection_and_cpu_completion_rearms_cached_domains() {
    let machine = Machine::new(PAGE_SIZE).unwrap();
    let (sender, receiver) = mpsc::channel();
    let endpoint: Arc<dyn DeviceAccessEndpoint> = Arc::new(Endpoint(sender));
    machine.set_device_endpoint(&endpoint);
    let allocation = machine
        .account(PAGE_SIZE)
        .unwrap()
        .allocate(PAGE_SIZE)
        .unwrap();
    let mapping =
        PhysicalMapping::new(&[AllocationRange::new(allocation.pin(), 0, PAGE_SIZE).unwrap()])
            .unwrap();
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
    let mut writers = Vec::new();
    for start in [0, 4] {
        let footprint = Footprint::new(
            Arc::clone(&mapping),
            WriteDomain::Buffer { start, bytes: 4 },
        )
        .unwrap();
        authority
            .register(start, vec![Arc::clone(&footprint)])
            .unwrap();
        writers.push(authority.create_writer(start, footprint).unwrap());
    }
    let mut appended = 0;
    authority
        .publish_write(&machine, writers[0], || {
            appended += 1;
            Ok(((), NativeTick(1)))
        })
        .unwrap();
    let first_protection = machine.tracking_statistics().protection_calls;
    assert_eq!(first_protection, 1);
    for tick in 2..1002 {
        authority
            .publish_write(&machine, writers[(tick % 2) as usize], || {
                appended += 1;
                Ok(((), NativeTick(tick)))
            })
            .unwrap();
    }
    assert_eq!(appended, 1001);
    assert_eq!(
        machine.tracking_statistics().protection_calls,
        first_protection
    );

    let Access::Pending(first_write) = space.write_value(base, 4, 0x11223344).unwrap() else {
        panic!("GPU byte overwrite must enter the owner");
    };
    let first_id = authority.queue_access(receiver.recv().unwrap()).unwrap();
    assert!(authority.prepare_access(first_id).unwrap().is_empty());
    let mut invalid_append = false;
    assert!(
        authority
            .publish_write(&machine, writers[0], || {
                invalid_append = true;
                Ok(((), NativeTick(1002)))
            })
            .is_err()
    );
    assert!(!invalid_append);
    authority.complete_access(&machine, first_id).unwrap();
    assert_eq!(first_write.wait().unwrap(), AccessResult::Written);
    assert_eq!(
        machine.tracking_statistics().protection_calls,
        first_protection
    );

    // A disjoint RAM-only read queued in the same page still prevents reopening
    // that page, even after every GPU-produced byte has been overwritten.
    let Access::Pending(gap_read) = space
        .read_value(GuestVirtualAddress::new(base.get() + 16), 4)
        .unwrap()
    else {
        panic!("the other producer keeps the page protected");
    };
    let gap_id = authority.queue_access(receiver.recv().unwrap()).unwrap();
    assert!(authority.prepare_access(gap_id).unwrap().is_empty());
    let Access::Pending(second_write) = space
        .write_value(GuestVirtualAddress::new(base.get() + 4), 4, 0x55667788)
        .unwrap()
    else {
        panic!("second GPU byte overwrite must enter the owner");
    };
    let second_id = authority.queue_access(receiver.recv().unwrap()).unwrap();
    assert!(authority.prepare_access(second_id).unwrap().is_empty());
    authority.complete_access(&machine, second_id).unwrap();
    assert_eq!(second_write.wait().unwrap(), AccessResult::Written);
    assert_eq!(
        machine.tracking_statistics().protection_calls,
        first_protection
    );
    authority.complete_access(&machine, gap_id).unwrap();
    assert_eq!(
        gap_read.wait().unwrap(),
        AccessResult::Read(Box::new([0; 4]))
    );
    assert_eq!(
        machine.tracking_statistics().protection_calls,
        first_protection + 1
    );
    assert!(matches!(
        space.read_value(base, 8).unwrap(),
        Access::Ready(0x5566778811223344)
    ));
    assert!(!authority.writers[&writers[0]].device_protected);
    assert!(!authority.writers[&writers[1]].device_protected);

    authority
        .publish_write(&machine, writers[0], || Ok(((), NativeTick(1003))))
        .unwrap();
    assert_eq!(
        machine.tracking_statistics().protection_calls,
        first_protection + 2
    );
    let Access::Pending(readback) = space.read_value(base, 4).unwrap() else {
        panic!("cached replay must restore protection");
    };
    let read_id = authority.queue_access(receiver.recv().unwrap()).unwrap();
    let plan = authority.prepare_access(read_id).unwrap();
    assert_eq!(plan.len(), 1);
    authority
        .publish_access_readback(&machine, plan[0], &[9, 8, 7, 6], &mut Vec::new())
        .unwrap();
    authority.complete_access(&machine, read_id).unwrap();
    assert_eq!(
        readback.wait().unwrap(),
        AccessResult::Read(Box::new([9, 8, 7, 6]))
    );
    assert!(matches!(
        space.read_value(base, 4).unwrap(),
        Access::Ready(0x06070809)
    ));
    assert!(matches!(
        space.write_value(base, 4, 0xaabbccdd).unwrap(),
        Access::Ready(())
    ));
    assert!(
        authority
            .resolve(&space.capture(base, 4, MemoryPermissions::READ).unwrap())
            .unwrap()[0]
            .producer
            .is_none()
    );
}

#[test]
fn publication_invalidates_new_code_before_append_without_another_page_publication() {
    let machine = Machine::new(PAGE_SIZE).unwrap();
    let (sender, receiver) = mpsc::channel();
    let endpoint: Arc<dyn DeviceAccessEndpoint> = Arc::new(Endpoint(sender));
    machine.set_device_endpoint(&endpoint);
    let allocation = machine
        .account(PAGE_SIZE)
        .unwrap()
        .allocate(PAGE_SIZE)
        .unwrap();
    let range = AllocationRange::new(allocation.pin(), 0, 4).unwrap();
    let mapping = PhysicalMapping::new(std::slice::from_ref(&range)).unwrap();
    let footprint = Footprint::new(mapping, WriteDomain::Buffer { start: 0, bytes: 4 }).unwrap();
    let mut authority = Authority::new(machine.identity());
    authority
        .register(1u32, vec![Arc::clone(&footprint)])
        .unwrap();
    let writer = authority.create_writer(1, footprint).unwrap();
    authority
        .publish_write(&machine, writer, || Ok(((), NativeTick(1))))
        .unwrap();
    let (Access::Pending(code), observation) = machine.access_code_captured(vec![range]).unwrap()
    else {
        panic!("device code must be captured by the owner");
    };
    let id = authority.queue_access(receiver.recv().unwrap()).unwrap();
    let [plan] = authority.prepare_access(id).unwrap()[..] else {
        panic!("one exact code readback");
    };
    authority
        .publish_access_readback(&machine, plan, &[1, 2, 3, 4], &mut Vec::new())
        .unwrap();
    authority.complete_access(&machine, id).unwrap();
    assert_eq!(
        code.wait().unwrap(),
        AccessResult::Read(Box::new([1, 2, 3, 4]))
    );
    assert!(!observation.is_invalid());
    let protections = machine.tracking_statistics().protection_calls;
    authority
        .publish_write(&machine, writer, || {
            assert!(observation.is_invalid());
            Ok(((), NativeTick(2)))
        })
        .unwrap();
    assert_eq!(machine.tracking_statistics().protection_calls, protections);
}

#[test]
fn cancelling_the_last_exact_dependency_reopens_a_ram_only_page() {
    for cancel_before_completion in [true, false] {
        let machine = Machine::new(PAGE_SIZE).unwrap();
        let (sender, receiver) = mpsc::channel();
        let endpoint: Arc<dyn DeviceAccessEndpoint> = Arc::new(Endpoint(sender));
        machine.set_device_endpoint(&endpoint);
        let allocation = machine
            .account(PAGE_SIZE)
            .unwrap()
            .allocate(PAGE_SIZE)
            .unwrap();
        let mapping =
            PhysicalMapping::new(&[AllocationRange::new(allocation.pin(), 0, 4).unwrap()]).unwrap();
        let footprint =
            Footprint::new(mapping, WriteDomain::Buffer { start: 0, bytes: 4 }).unwrap();
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
        authority
            .register(1u32, vec![Arc::clone(&footprint)])
            .unwrap();
        let writer = authority.create_writer(1, footprint).unwrap();
        authority
            .publish_write(&machine, writer, || Ok(((), NativeTick(1))))
            .unwrap();

        let Access::Pending(gap) = space.read_value(base.checked_add(16).unwrap(), 4).unwrap()
        else {
            panic!("device-affected page bypassed its exact owner");
        };
        let gap_id = authority.queue_access(receiver.recv().unwrap()).unwrap();
        assert!(authority.prepare_access(gap_id).unwrap().is_empty());
        let Access::Pending(overwrite) = space.write_value(base, 4, 93).unwrap() else {
            panic!("device overwrite bypassed its exact owner");
        };
        let overwrite_id = authority.queue_access(receiver.recv().unwrap()).unwrap();
        assert!(authority.prepare_access(overwrite_id).unwrap().is_empty());
        authority.complete_access(&machine, overwrite_id).unwrap();
        assert_eq!(overwrite.wait().unwrap(), AccessResult::Written);
        let before = machine.tracking_statistics().protection_calls;
        drop(gap);
        if cancel_before_completion {
            authority.cancel_access(&machine, gap_id).unwrap();
        } else {
            authority.complete_access(&machine, gap_id).unwrap();
        }
        assert_eq!(machine.tracking_statistics().protection_calls, before + 1);
        assert!(matches!(
            space.read_value(base, 4).unwrap(),
            Access::Ready(93)
        ));
        assert_eq!(authority.pending_access_count(), 0);
        assert!(!authority.writers[&writer].device_protected);
        assert!(receiver.try_recv().is_err());
    }
}

#[test]
fn one_effect_publishes_all_writers_once_and_rejects_a_conflicting_cpu_access_before_append() {
    let machine = Machine::new(PAGE_SIZE * 2).unwrap();
    let allocation = machine
        .account(PAGE_SIZE * 2)
        .unwrap()
        .allocate(PAGE_SIZE * 2)
        .unwrap();
    let ranges =
        [0, PAGE_SIZE].map(|offset| AllocationRange::new(allocation.pin(), offset, 4).unwrap());
    let mut authority = Authority::new(machine.identity());
    let mut writers = Vec::new();
    let observations: Vec<_> = ranges
        .iter()
        .map(|range| {
            let observation = nixe_memory::Observation::default();
            machine
                .access_captured(
                    std::slice::from_ref(range),
                    nixe_memory::AccessOperation::ReadObserved(observation.clone()),
                )
                .unwrap()
                .into_ready()
                .unwrap();
            observation
        })
        .collect();
    for (index, range) in ranges.iter().enumerate() {
        let footprint = Footprint::new(
            PhysicalMapping::new(std::slice::from_ref(range)).unwrap(),
            WriteDomain::Buffer { start: 0, bytes: 4 },
        )
        .unwrap();
        authority
            .register(index, vec![Arc::clone(&footprint)])
            .unwrap();
        writers.push(authority.create_writer(index, footprint).unwrap());
        authority
            .watch_memory(std::slice::from_ref(range), observations[index].clone())
            .unwrap();
    }
    let mut appended = 0;
    authority
        .publish_writes(&machine, &writers, || {
            assert!(observations.iter().all(|value| value.is_invalid()));
            appended += 1;
            Ok(((), NativeTick(1)))
        })
        .unwrap();
    assert_eq!(appended, 1);
    let baseline = machine.tracking_statistics().protection_calls;
    for tick in 2..1002 {
        authority
            .publish_writes(&machine, &writers, || {
                appended += 1;
                Ok(((), NativeTick(tick)))
            })
            .unwrap();
    }
    assert_eq!(appended, 1001);
    assert_eq!(machine.tracking_statistics().protection_calls, baseline);
    let before = authority.resolve(&ranges).unwrap();
    assert_eq!(before.len(), 2);
    assert!(
        before
            .iter()
            .all(|part| part.producer.unwrap().tick == NativeTick(1001))
    );
    let (request, result) = machine
        .prepare_device_access(
            vec![ranges[1].clone()],
            nixe_memory::AccessOperation::Write(Box::new([9; 4])),
        )
        .unwrap();
    let id = authority.queue_access(request).unwrap();
    assert!(
        authority
            .publish_writes(&machine, &writers, || {
                appended += 1;
                Ok(((), NativeTick(1002)))
            })
            .is_err()
    );
    assert_eq!(appended, 1001);
    assert_eq!(authority.resolve(&ranges).unwrap(), before);
    authority.cancel_access(&machine, id).unwrap();
    assert!(result.wait().is_err());
    authority
        .publish_writes(&machine, &writers, || Ok(((), NativeTick(1002))))
        .unwrap();
    assert!(
        authority
            .resolve(&ranges)
            .unwrap()
            .iter()
            .all(|part| part.producer.unwrap().tick == NativeTick(1002))
    );
}
