use super::*;
use crate::physical::WriteDomain;
use nixe_memory::AtomicOperation;
use nixe_memory::{Access, DeviceAccessEndpoint, Extent, PAGE_SIZE, PendingAccess};
use std::sync::mpsc;

struct Endpoint {
    requests: mpsc::Sender<DeviceAccessRequest>,
    cancellations: mpsc::Sender<AccessRequestId>,
}
impl DeviceAccessEndpoint for Endpoint {
    fn submit(&self, request: DeviceAccessRequest) {
        self.requests.send(request).unwrap();
    }
    fn cancel(&self, request: AccessRequestId) {
        let _ = self.cancellations.send(request);
    }
    fn allocation_pressure(&self, _: nixe_memory::MachineId, _: u64) {}
    fn allocation_released(&self, _: MachineId, _: AllocationId) {}
    fn physical_retired(&self, _: MachineId, _: AllocationId, _: Box<[nixe_memory::Extent]>) {}
}
fn endpoint(
    machine: &Machine,
) -> (
    Arc<dyn DeviceAccessEndpoint>,
    mpsc::Receiver<DeviceAccessRequest>,
    mpsc::Receiver<AccessRequestId>,
) {
    let (requests, receiver) = mpsc::channel();
    let (cancellations, cancelled) = mpsc::channel();
    let endpoint: Arc<dyn DeviceAccessEndpoint> = Arc::new(Endpoint {
        requests,
        cancellations,
    });
    machine.set_device_endpoint(&endpoint);
    (endpoint, receiver, cancelled)
}

#[test]
fn indexed_alias_spans_preserve_dependencies_consumers_and_partial_retirement() {
    let machine = Machine::new(3 * PAGE_SIZE).unwrap();
    let (_endpoint, receiver, _) = endpoint(&machine);
    let allocation = machine
        .account(3 * PAGE_SIZE)
        .unwrap()
        .allocate(3 * PAGE_SIZE)
        .unwrap();
    let page = |n| AllocationRange::new(allocation.pin(), n * PAGE_SIZE, PAGE_SIZE).unwrap();
    let alias = PhysicalMapping::new(&[page(2), page(0), page(2)]).unwrap();
    let other = PhysicalMapping::new(&[page(1)]).unwrap();
    let domain = |mapping, start, bytes| {
        Footprint::new(mapping, WriteDomain::Buffer { start, bytes }).unwrap()
    };
    let mut authority = Authority::new(machine.identity());
    authority
        .register(
            1u32,
            vec![
                domain(alias.clone(), 0, 3 * PAGE_SIZE),
                domain(other.clone(), 0, PAGE_SIZE),
            ],
        )
        .unwrap();
    // The first occurrence of page 2 is outside this writer. The index must
    // retain its second occurrence and the separate registered footprint.
    let first = authority
        .create_writer(1, domain(alias, 2 * PAGE_SIZE + 7, 4))
        .unwrap();
    let second = authority
        .create_writer(1, domain(other.clone(), 7, 4))
        .unwrap();
    for writer in [first, second] {
        authority
            .publish_write(&machine, writer, || Ok(((), NativeTick(1))))
            .unwrap();
    }
    let equal_alias = PhysicalMapping::new(&[page(2), page(0), page(2)]).unwrap();
    let affected = authority
        .observe(1, domain(equal_alias, 2 * PAGE_SIZE + 7, 4))
        .unwrap();
    let unaffected = authority.observe(1, domain(other, 7, 4)).unwrap();
    let range = AllocationRange::new(allocation.pin(), 2 * PAGE_SIZE + 7, 4).unwrap();
    let Access::Pending(pending) = machine
        .access_captured(std::slice::from_ref(&range), AccessOperation::Read)
        .unwrap()
    else {
        panic!("device read must defer");
    };
    let request = authority.queue_access(receiver.recv().unwrap()).unwrap();
    assert_eq!(authority.writer_dependencies(first).unwrap(), &[request]);
    assert!(authority.writer_dependencies(second).unwrap().is_empty());
    authority.fail_access(
        request,
        Arc::new(MemoryError::Invalid("test request complete")),
    );
    assert!(pending.wait().is_err());
    authority.cpu_write(std::slice::from_ref(&range)).unwrap();
    assert!(affected.is_invalid());
    assert!(!unaffected.is_invalid());
    assert!(
        authority
            .resolve(std::slice::from_ref(&range))
            .unwrap()
            .iter()
            .all(|p| p.producer.is_none())
    );
    authority.normalize(1).unwrap();
    let retained = allocation.pin_range(PAGE_SIZE, PAGE_SIZE).unwrap();
    let retired = range
        .physical_extents()
        .map(|extent| Extent {
            start: ArenaOffset::new(extent.start.get() & !(PAGE_SIZE - 1)),
            bytes: PAGE_SIZE,
        })
        .collect::<Vec<_>>();
    let identity = allocation.identity();
    assert!(
        authority
            .retire_physical_ranges(machine.identity(), identity, &retired)
            .is_err()
    );
    drop(range);
    drop(allocation);
    authority
        .retire_physical_ranges(machine.identity(), identity, &retired)
        .unwrap();
    let live = AllocationRange::new(retained, PAGE_SIZE + 7, 4).unwrap();
    assert_eq!(
        authority.resolve(&[live]).unwrap()[0]
            .producer
            .unwrap()
            .writer,
        second
    );
    assert_eq!(authority.resource_count(), 1);
}

#[test]
fn native_read_in_a_ram_hole_resolves_its_page_and_invalidates_on_replay() {
    use nixe_memory::{AddressSpace, AddressWidth, GuestVirtualAddress, MemoryPermissions};
    let machine = Machine::new(2 * PAGE_SIZE).unwrap();
    let (_endpoint, receiver, _) = endpoint(&machine);
    let account = machine.account(2 * PAGE_SIZE).unwrap();
    let allocation = account.allocate(PAGE_SIZE).unwrap();
    let range = |start, bytes| AllocationRange::new(allocation.pin(), start, bytes).unwrap();
    let space = AddressSpace::new(machine.clone(), AddressWidth::Bits32).unwrap();
    for base in [PAGE_SIZE, 3 * PAGE_SIZE] {
        space
            .map(
                GuestVirtualAddress::new(base),
                allocation.clone(),
                0,
                PAGE_SIZE,
                MemoryPermissions::READ_WRITE,
            )
            .unwrap();
    }
    machine
        .write_resolved(&[range(128, 4)], &77u32.to_le_bytes())
        .unwrap();
    let mapping = PhysicalMapping::new(&[range(0, PAGE_SIZE)]).unwrap();
    let domain = |start, bytes| {
        Footprint::new(mapping.clone(), WriteDomain::Buffer { start, bytes }).unwrap()
    };
    let mut authority = Authority::new(machine.identity());
    let mut writers = Vec::new();
    for (id, start, size) in [(1, 0, 64), (2, 512, 64), (3, 64, 448)] {
        let footprint = domain(start, size);
        authority.register(id, vec![footprint.clone()]).unwrap();
        writers.push(authority.create_writer(id, footprint).unwrap());
    }
    for writer in &writers[..2] {
        authority
            .publish_write(&machine, *writer, || Ok(((), NativeTick(1))))
            .unwrap();
    }
    let (read, result) = enqueue(
        &machine,
        &receiver,
        &mut authority,
        range(128, 4),
        AccessOperation::ReadNative,
    );
    let plans = authority.prepare_access(read).unwrap();
    assert_eq!(plans.iter().map(|p| p.source.bytes).sum::<u64>(), 128);
    assert!(plans.iter().all(|p| !p.required));
    for plan in plans {
        authority
            .publish_access_readback(
                &machine,
                plan,
                &vec![0x37; plan.source.bytes as usize],
                &mut Vec::new(),
            )
            .unwrap();
    }
    authority.complete_access(&machine, read).unwrap();
    let AccessResult::NativeRead {
        bytes,
        area: Some(area),
    } = result.wait().unwrap()
    else {
        panic!("native read must retain a RAM area");
    };
    assert_eq!(*bytes, 77u32.to_le_bytes());
    let core = space.register_core(Arc::new(|| {}));
    {
        let native = core.enter_native().unwrap();
        for base in [PAGE_SIZE, 3 * PAGE_SIZE] {
            for _ in 0..128 {
                assert_eq!(
                    space
                        .read_native(
                            &native,
                            GuestVirtualAddress::new(base + 128),
                            4,
                            Some(&area)
                        )
                        .unwrap(),
                    Some(77)
                );
            }
            for offset in [0, 60, 512, 572] {
                assert_eq!(
                    space
                        .read_native(&native, GuestVirtualAddress::new(base + offset), 4, None)
                        .unwrap(),
                    Some(0x37373737)
                );
            }
        }
    }
    // A reused virtual address cannot reuse another allocation's RAM proof.
    let replacement = account.allocate(PAGE_SIZE).unwrap();
    let replacement_range = AllocationRange::new(replacement.pin(), 0, PAGE_SIZE).unwrap();
    space
        .unmap(GuestVirtualAddress::new(PAGE_SIZE), PAGE_SIZE)
        .unwrap();
    space
        .map(
            GuestVirtualAddress::new(PAGE_SIZE),
            replacement,
            0,
            PAGE_SIZE,
            MemoryPermissions::READ_WRITE,
        )
        .unwrap();
    machine
        .publish_device_write(&[replacement_range], || Ok(()))
        .unwrap();
    let native = core.enter_native().unwrap();
    assert_eq!(
        space
            .read_native(
                &native,
                GuestVirtualAddress::new(PAGE_SIZE + 128),
                4,
                Some(&area)
            )
            .unwrap(),
        None
    );
    assert!(!area.observation().is_invalid());
    drop(native);
    // Replaying a producer invalidates the area that now includes its bytes.
    authority
        .publish_write(&machine, writers[0], || Ok(((), NativeTick(2))))
        .unwrap();
    assert!(area.observation().is_invalid());
    authority
        .publish_write(&machine, writers[2], || Ok(((), NativeTick(3))))
        .unwrap();
    assert!(area.observation().is_invalid());
    let native = core.enter_native().unwrap();
    assert_eq!(
        space
            .read_native(
                &native,
                GuestVirtualAddress::new(3 * PAGE_SIZE + 128),
                4,
                Some(&area)
            )
            .unwrap(),
        None
    );
    drop(native);
}
fn enqueue(
    machine: &Machine,
    receiver: &mpsc::Receiver<DeviceAccessRequest>,
    authority: &mut Authority<u32>,
    range: AllocationRange,
    operation: AccessOperation,
) -> (AccessRequestId, PendingAccess) {
    let pending = machine
        .request_device_access(vec![range], operation)
        .unwrap();
    let id = authority.queue_access(receiver.recv().unwrap()).unwrap();
    (id, pending)
}

#[test]
fn exact_accesses_order_overlaps_allow_disjoint_work_and_apply_operations_once() {
    let machine = Machine::new(PAGE_SIZE).unwrap();
    let (_endpoint, receiver, cancellations) = endpoint(&machine);
    let allocation = machine
        .account(PAGE_SIZE)
        .unwrap()
        .allocate(PAGE_SIZE)
        .unwrap();
    let range = |start, bytes| AllocationRange::new(allocation.pin(), start, bytes).unwrap();
    let mapping = PhysicalMapping::new(&[range(0, PAGE_SIZE)]).unwrap();
    let domain = |start, bytes| {
        Footprint::new(Arc::clone(&mapping), WriteDomain::Buffer { start, bytes }).unwrap()
    };
    let full = domain(0, 64);
    let mut authority = Authority::new(machine.identity());
    authority.register(1u32, vec![Arc::clone(&full)]).unwrap();
    authority.register(2, vec![domain(128, 8)]).unwrap();
    let writer = authority.create_writer(1, Arc::clone(&full)).unwrap();
    let disjoint = authority.create_writer(2, domain(128, 8)).unwrap();
    authority.record_write(writer, NativeTick(10)).unwrap();
    let (read, read_result) = enqueue(
        &machine,
        &receiver,
        &mut authority,
        range(4, 4),
        AccessOperation::Read,
    );
    let (write, write_result) = enqueue(
        &machine,
        &receiver,
        &mut authority,
        range(6, 2),
        AccessOperation::Write(Box::new([9, 10])),
    );
    assert!(!authority.access_can_prepare(write));
    assert!(authority.record_write(writer, NativeTick(11)).is_err());
    authority.register(3, vec![Arc::clone(&full)]).unwrap();
    let late = authority.create_writer(3, full).unwrap();
    assert!(authority.writer_dependencies(late).unwrap().contains(&read));
    assert!(
        authority
            .writer_dependencies(late)
            .unwrap()
            .contains(&write)
    );
    assert!(authority.record_write(late, NativeTick(11)).is_err());
    let (other, other_result) = enqueue(
        &machine,
        &receiver,
        &mut authority,
        range(128, 1),
        AccessOperation::Write(Box::new([77])),
    );
    assert!(authority.prepare_access(other).unwrap().is_empty());
    authority.complete_access(&machine, other).unwrap();
    assert_eq!(other_result.wait().unwrap(), AccessResult::Written);
    authority.record_write(disjoint, NativeTick(12)).unwrap();
    let [plan]: [AccessReadback<u32>; 1] =
        authority.prepare_access(read).unwrap().try_into().unwrap();
    assert_eq!(plan.source.bytes, 4);
    assert!(!authority.access_can_complete(read));
    assert!(authority.complete_access(&machine, read).is_err());
    assert_eq!(
        authority
            .publish_access_readback(&machine, plan, &[1, 2, 3, 4], &mut Vec::new())
            .unwrap(),
        ReadbackPublication::Published {
            bytes: plan.source.bytes
        }
    );
    assert!(!authority.access_can_prepare(write));
    authority.complete_access(&machine, read).unwrap();
    assert_eq!(
        read_result.wait().unwrap(),
        AccessResult::Read(Box::new([1, 2, 3, 4]))
    );
    assert!(authority.prepare_access(write).unwrap().is_empty());
    authority.complete_access(&machine, write).unwrap();
    assert_eq!(write_result.wait().unwrap(), AccessResult::Written);
    assert!(authority.writer_dependencies(writer).unwrap().is_empty());
    assert!(authority.writer_dependencies(late).unwrap().is_empty());
    let (again, again_result) = enqueue(
        &machine,
        &receiver,
        &mut authority,
        range(4, 4),
        AccessOperation::Read,
    );
    assert!(authority.prepare_access(again).unwrap().is_empty());
    authority.complete_access(&machine, again).unwrap();
    assert_eq!(
        again_result.wait().unwrap(),
        AccessResult::Read(Box::new([1, 2, 9, 10]))
    );
    authority.record_write(writer, NativeTick(9)).unwrap();
    let (cancel, cancel_result) = enqueue(
        &machine,
        &receiver,
        &mut authority,
        range(4, 4),
        AccessOperation::Read,
    );
    let [old_plan]: [AccessReadback<u32>; 1] = authority
        .prepare_access(cancel)
        .unwrap()
        .try_into()
        .unwrap();
    drop(cancel_result);
    assert_eq!(cancellations.recv().unwrap(), cancel);
    // Native completion can be polled before the queued cancellation message.
    // The cancelled completion itself must already prohibit RAM publication.
    assert_eq!(
        authority
            .publish_access_readback(&machine, old_plan, &[98; 4], &mut Vec::new())
            .unwrap(),
        ReadbackPublication::Obsolete
    );
    let mut before_cancel_message = [0; 4];
    machine
        .capture_stream(&[range(4, 4)], &mut before_cancel_message)
        .unwrap();
    assert_eq!(before_cancel_message, [1, 2, 9, 10]);
    assert!(authority.access_requests().any(|id| id == cancel));
    authority.fail_access(
        cancel,
        Arc::new(MemoryError::Invalid("CPU request cancelled")),
    );
    assert_eq!(
        authority
            .publish_access_readback(&machine, old_plan, &[99; 4], &mut Vec::new())
            .unwrap(),
        ReadbackPublication::Obsolete
    );
    let mut bytes = [0; 4];
    machine.capture_stream(&[range(4, 4)], &mut bytes).unwrap();
    assert_eq!(bytes, [1, 2, 9, 10]);
    assert_eq!(authority.pending_access_count(), 0);
    assert!(authority.writer_dependencies(writer).unwrap().is_empty());
}

#[test]
fn failure_preserves_the_original_error_and_releases_all_request_pins() {
    let machine = Machine::new(PAGE_SIZE).unwrap();
    let (_endpoint, receiver, _cancellations) = endpoint(&machine);
    let allocation = machine
        .account(PAGE_SIZE)
        .unwrap()
        .allocate(PAGE_SIZE)
        .unwrap();
    let mut authority = Authority::new(machine.identity());
    let (a, first) = enqueue(
        &machine,
        &receiver,
        &mut authority,
        AllocationRange::new(allocation.pin(), 0, 4).unwrap(),
        AccessOperation::Read,
    );
    let (b, second) = enqueue(
        &machine,
        &receiver,
        &mut authority,
        AllocationRange::new(allocation.pin(), 2, 2).unwrap(),
        AccessOperation::Write(Box::new([7, 8])),
    );
    assert!(authority.access_can_prepare(a));
    assert!(!authority.access_can_prepare(b));
    drop(allocation);
    assert_eq!(machine.statistics().retained_bytes, PAGE_SIZE);
    let cause = Arc::new(MemoryError::Invalid("native submission B failed"));
    authority.fail_accesses(Arc::clone(&cause));
    assert!(Arc::ptr_eq(&first.wait().unwrap_err(), &cause));
    assert!(Arc::ptr_eq(&second.wait().unwrap_err(), &cause));
    assert_eq!(authority.pending_access_count(), 0);
    assert_eq!(machine.statistics().allocated_bytes, 0);
}

#[test]
fn an_owned_atomic_reads_its_current_producer_and_is_never_replayed() {
    use nixe_memory::{AccessOrdering, AtomicAccess, AtomicTransform};
    let machine = Machine::new(PAGE_SIZE).unwrap();
    let (_endpoint, receiver, _cancellations) = endpoint(&machine);
    let allocation = machine
        .account(PAGE_SIZE)
        .unwrap()
        .allocate(PAGE_SIZE)
        .unwrap();
    let range = || AllocationRange::new(allocation.pin(), 16, 4).unwrap();
    let mapping = PhysicalMapping::new(&[range()]).unwrap();
    let footprint = Footprint::new(mapping, WriteDomain::Buffer { start: 0, bytes: 4 }).unwrap();
    let mut authority = Authority::new(machine.identity());
    authority
        .register(1u32, vec![Arc::clone(&footprint)])
        .unwrap();
    let writer = authority.create_writer(1, footprint).unwrap();
    authority.record_write(writer, NativeTick(1)).unwrap();
    let atomic = AtomicAccess {
        bytes: 4,
        ordering: AccessOrdering::AcquireRelease,
        operation: AtomicOperation::ReadModifyWrite {
            transform: AtomicTransform::Add,
            operand: 1,
        },
    };
    let (id, result) = enqueue(
        &machine,
        &receiver,
        &mut authority,
        range(),
        AccessOperation::Atomic(atomic),
    );
    let [plan]: [AccessReadback<u32>; 1] =
        authority.prepare_access(id).unwrap().try_into().unwrap();
    authority
        .publish_access_readback(&machine, plan, &22u32.to_le_bytes(), &mut Vec::new())
        .unwrap();
    authority.complete_access(&machine, id).unwrap();
    assert_eq!(
        result.wait().unwrap(),
        AccessResult::Atomic {
            previous: 22,
            written: true
        }
    );
    assert!(authority.complete_access(&machine, id).is_err());
    let mut bytes = [0; 4];
    machine.capture_stream(&[range()], &mut bytes).unwrap();
    assert_eq!(u32::from_le_bytes(bytes), 23);
    assert!(authority.resolve(&[range()]).unwrap()[0].producer.is_none());
    authority.record_write(writer, NativeTick(2)).unwrap();
    let snapshot = authority.resolve(&[range()]).unwrap().remove(0);
    authority
        .publish_readback(&machine, snapshot, &100u32.to_le_bytes(), &mut Vec::new())
        .unwrap();
    let code = nixe_memory::Observation::default();
    machine
        .apply_resolved_access(
            &[range()],
            &nixe_memory::AccessOperation::ReadCode(code.clone()),
        )
        .unwrap();
    let compare = AtomicAccess {
        operation: AtomicOperation::CompareExchange {
            expected: 99,
            replacement: 101,
        },
        ..atomic
    };
    let (id, result) = enqueue(
        &machine,
        &receiver,
        &mut authority,
        range(),
        AccessOperation::Atomic(compare),
    );
    assert!(authority.prepare_access(id).unwrap().is_empty());
    authority.complete_access(&machine, id).unwrap();
    assert_eq!(
        result.wait().unwrap(),
        AccessResult::Atomic {
            previous: 100,
            written: false
        }
    );
    assert!(!code.is_invalid());
    assert!(authority.resolve(&[range()]).unwrap()[0].producer.is_none());
}

#[test]
fn device_code_capture_is_invalidated_before_replay_without_touching_unwritten_pages() {
    let machine = Machine::new(2 * PAGE_SIZE).unwrap();
    let (_endpoint, receiver, _cancellations) = endpoint(&machine);
    let allocation = machine
        .account(2 * PAGE_SIZE)
        .unwrap()
        .allocate(2 * PAGE_SIZE)
        .unwrap();
    let range = |offset, bytes| AllocationRange::new(allocation.pin(), offset, bytes).unwrap();
    let mapping = PhysicalMapping::new(&[range(0, 2 * PAGE_SIZE)]).unwrap();
    let footprint = Footprint::new(
        mapping,
        WriteDomain::Buffer {
            start: 16,
            bytes: 4,
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
    let (access, code) = machine.access_code_captured(vec![range(16, 4)]).unwrap();
    let nixe_memory::Access::Pending(result) = access else {
        panic!("device code bypassed owner resolution");
    };
    let id = authority.queue_access(receiver.recv().unwrap()).unwrap();
    let [plan]: [AccessReadback<u32>; 1] =
        authority.prepare_access(id).unwrap().try_into().unwrap();
    authority
        .publish_access_readback(
            &machine,
            plan,
            &0x11223344u32.to_le_bytes(),
            &mut Vec::new(),
        )
        .unwrap();
    authority.complete_access(&machine, id).unwrap();
    assert_eq!(
        result.wait().unwrap(),
        AccessResult::Read(Box::new(0x11223344u32.to_le_bytes()))
    );
    // The owner resolved the only producer above; the device page remains protected.
    let root = [range(0, 2 * PAGE_SIZE)];
    let observed = nixe_memory::Observation::default();
    machine
        .apply_resolved_access(
            &root,
            &nixe_memory::AccessOperation::ReadObserved(observed.clone()),
        )
        .unwrap();
    authority.watch_memory(&root, observed.clone()).unwrap();
    let untouched = nixe_memory::Observation::default();
    machine
        .access_captured(
            &[range(PAGE_SIZE, PAGE_SIZE)],
            nixe_memory::AccessOperation::ReadObserved(untouched.clone()),
        )
        .unwrap()
        .into_ready()
        .unwrap();
    assert!(authority.record_write(writer, NativeTick(2)).is_err());
    let inputs =
        crate::physical::pin_selected_pages(authority.watched_write_pages(writer).unwrap())
            .unwrap();
    assert_eq!(inputs.len(), 1);
    assert_eq!((inputs[0].offset(), inputs[0].bytes()), (0, PAGE_SIZE));
    let before = machine.tracking_statistics().protection_calls;
    authority
        .publish_write(&machine, writer, || {
            assert!(code.is_invalid());
            assert!(observed.is_invalid());
            assert!(!untouched.is_invalid());
            Ok(((), NativeTick(2)))
        })
        .unwrap();
    assert!(authority.watched_write_pages(writer).unwrap().is_empty());
    assert_eq!(machine.tracking_statistics().protection_calls, before);
}

#[test]
fn late_readback_does_not_pin_or_overwrite_a_recycled_allocation() {
    let machine = Machine::new(PAGE_SIZE).unwrap();
    let account = machine.account(PAGE_SIZE).unwrap();
    let allocation = account.allocate(PAGE_SIZE).unwrap();
    let range = AllocationRange::new(allocation.pin(), 0, 4).unwrap();
    let mapping = PhysicalMapping::new(std::slice::from_ref(&range)).unwrap();
    let footprint = Footprint::new(mapping, WriteDomain::Buffer { start: 0, bytes: 4 }).unwrap();
    let mut authority = Authority::new(machine.identity());
    authority
        .register(1u32, vec![Arc::clone(&footprint)])
        .unwrap();
    let writer = authority.create_writer(1, footprint).unwrap();
    authority.record_write(writer, NativeTick(1)).unwrap();
    let snapshot = authority
        .resolve(std::slice::from_ref(&range))
        .unwrap()
        .remove(0);
    drop(range);
    drop(allocation);
    assert_eq!(machine.statistics().allocated_bytes, 0);
    let replacement = account.allocate(PAGE_SIZE).unwrap();
    let range = AllocationRange::new(replacement.pin(), 0, 4).unwrap();
    machine
        .write_resolved(std::slice::from_ref(&range), &[77; 4])
        .unwrap();
    assert!(
        authority
            .publish_readback(&machine, snapshot, &[55; 4], &mut Vec::new())
            .unwrap()
            .is_empty()
    );
    let mut bytes = [0; 4];
    machine
        .capture_stream(std::slice::from_ref(&range), &mut bytes)
        .unwrap();
    assert_eq!(bytes, [77; 4]);
    assert!(
        authority
            .resolve(&[range])
            .unwrap()
            .iter()
            .all(|span| span.producer.is_none())
    );
}

#[test]
fn owner_local_operations_apply_directly_without_endpoint_or_reentry() {
    use nixe_memory::{AccessOrdering, AtomicAccess, AtomicTransform};
    let machine = Machine::new(PAGE_SIZE).unwrap();
    let allocation = machine
        .account(PAGE_SIZE)
        .unwrap()
        .allocate(PAGE_SIZE)
        .unwrap();
    let range = || AllocationRange::new(allocation.pin(), 28, 4).unwrap();
    let mut authority = Authority::new(machine.identity());
    let (request, result) = machine
        .prepare_device_access(
            vec![range()],
            AccessOperation::Write(Box::new(41u32.to_le_bytes())),
        )
        .unwrap();
    let write = authority.queue_access(request).unwrap();
    assert!(authority.prepare_access(write).unwrap().is_empty());
    authority.complete_access(&machine, write).unwrap();
    assert_eq!(result.wait().unwrap(), AccessResult::Written);

    // A registered endpoint must not intercept owner-local operations either.
    let (_endpoint, receiver, cancellations) = endpoint(&machine);
    let mapping = PhysicalMapping::new(&[range()]).unwrap();
    let footprint = Footprint::new(mapping, WriteDomain::Buffer { start: 0, bytes: 4 }).unwrap();
    authority
        .register(1u32, vec![Arc::clone(&footprint)])
        .unwrap();
    let writer = authority.create_writer(1, footprint).unwrap();
    authority.record_write(writer, NativeTick(7)).unwrap();
    let (request, result) = machine
        .prepare_device_access(vec![range()], AccessOperation::Read)
        .unwrap();
    let read = authority.queue_access(request).unwrap();
    let [plan]: [AccessReadback<u32>; 1] =
        authority.prepare_access(read).unwrap().try_into().unwrap();
    assert_eq!((plan.source.bytes, plan.source.logical), (4, 0));
    authority
        .publish_access_readback(&machine, plan, &70u32.to_le_bytes(), &mut Vec::new())
        .unwrap();
    authority.complete_access(&machine, read).unwrap();
    assert_eq!(
        result.wait().unwrap(),
        AccessResult::Read(Box::new(70u32.to_le_bytes()))
    );

    let (request, result) = machine
        .prepare_device_access(
            vec![range()],
            AccessOperation::Atomic(AtomicAccess {
                bytes: 4,
                ordering: AccessOrdering::AcquireRelease,
                operation: AtomicOperation::ReadModifyWrite {
                    transform: AtomicTransform::Add,
                    operand: 2,
                },
            }),
        )
        .unwrap();
    let atomic = authority.queue_access(request).unwrap();
    assert!(authority.prepare_access(atomic).unwrap().is_empty());
    authority.complete_access(&machine, atomic).unwrap();
    assert_eq!(
        result.wait().unwrap(),
        AccessResult::Atomic {
            previous: 70,
            written: true
        }
    );
    let mut bytes = [0; 4];
    machine.capture_stream(&[range()], &mut bytes).unwrap();
    assert_eq!(u32::from_le_bytes(bytes), 72);

    let (request, result) = machine
        .prepare_device_access(vec![range()], AccessOperation::Write(Box::new([99; 4])))
        .unwrap();
    let cancelled = authority.queue_access(request).unwrap();
    assert!(authority.prepare_access(cancelled).unwrap().is_empty());
    drop(result);
    // Cancellation at completion is a request outcome, not an owner failure.
    authority.complete_access(&machine, cancelled).unwrap();
    machine.capture_stream(&[range()], &mut bytes).unwrap();
    assert_eq!(u32::from_le_bytes(bytes), 72);
    assert_eq!(authority.pending_access_count(), 0);
    assert!(receiver.try_recv().is_err());
    assert!(cancellations.try_recv().is_err());
}

#[test]
fn native_coherency_prefetch_preserves_cpu_holes_and_skips_newer_neighbour_writes() {
    native_prefetch_with_later_neighbour(64, 1, 1, 0x37);
    native_prefetch_with_later_neighbour(128, 4, 0, 0x55);
}

fn native_prefetch_with_later_neighbour(start: u64, bytes: u64, required: u64, expected: u8) {
    use nixe_memory::{AddressSpace, AddressWidth, GuestVirtualAddress, MemoryPermissions};
    let machine = Machine::new(PAGE_SIZE).unwrap();
    let (_endpoint, receiver, _) = endpoint(&machine);
    let allocation = machine
        .account(PAGE_SIZE)
        .unwrap()
        .allocate(PAGE_SIZE)
        .unwrap();
    let range = |start, bytes| AllocationRange::new(allocation.pin(), start, bytes).unwrap();
    machine
        .write_resolved(&[range(0, PAGE_SIZE)], &vec![0x19; PAGE_SIZE as usize])
        .unwrap();
    let mapping = PhysicalMapping::new(&[range(0, PAGE_SIZE)]).unwrap();
    let mut authority = Authority::new(machine.identity());
    let full = Footprint::new(
        mapping.clone(),
        WriteDomain::Buffer {
            start: 0,
            bytes: PAGE_SIZE,
        },
    )
    .unwrap();
    authority.register(1, vec![full.clone()]).unwrap();
    let original = authority.create_writer(1, full).unwrap();
    authority
        .publish_write(&machine, original, || Ok(((), NativeTick(1))))
        .unwrap();
    machine
        .write_resolved(&[range(128, 4)], &[0x55; 4])
        .unwrap();
    authority.cpu_write(&[range(128, 4)]).unwrap();
    let (id, result) = enqueue(
        &machine,
        &receiver,
        &mut authority,
        range(start, bytes),
        AccessOperation::ReadNative,
    );
    let plans = authority.prepare_access(id).unwrap();
    assert_eq!(
        plans
            .iter()
            .filter(|p| p.required)
            .map(|p| p.source.bytes)
            .sum::<u64>(),
        required
    );
    assert_eq!(
        plans.iter().map(|p| p.source.bytes).sum::<u64>(),
        PAGE_SIZE - 4
    );
    let neighbour = Footprint::new(
        mapping,
        WriteDomain::Buffer {
            start: 256,
            bytes: 4,
        },
    )
    .unwrap();
    authority.register(2, vec![neighbour.clone()]).unwrap();
    let later = authority.create_writer(2, neighbour).unwrap();
    authority
        .publish_write(&machine, later, || Ok(((), NativeTick(2))))
        .unwrap();
    let mut published = 0;
    for plan in plans {
        let ReadbackPublication::Published { bytes } = authority
            .publish_access_readback(
                &machine,
                plan,
                &vec![0x37; plan.source.bytes as usize],
                &mut Vec::new(),
            )
            .unwrap()
        else {
            panic!("newer optional neighbour must not restart the requested read");
        };
        published += bytes;
    }
    assert_eq!(published, PAGE_SIZE - 8);
    assert!(authority.access_can_complete(id));
    authority.complete_access(&machine, id).unwrap();
    let AccessResult::NativeRead {
        bytes,
        area: Some(area),
    } = result.wait().unwrap()
    else {
        panic!("native RAM region");
    };
    assert_eq!(&*bytes, &vec![expected; bytes.len()]);
    let space = AddressSpace::new(machine, AddressWidth::Bits32).unwrap();
    space
        .map(
            GuestVirtualAddress::new(PAGE_SIZE),
            allocation,
            0,
            PAGE_SIZE,
            MemoryPermissions::READ_WRITE,
        )
        .unwrap();
    let core = space.register_core(Arc::new(|| {}));
    let native = core.enter_native().unwrap();
    for (offset, expected) in [(64, 0x37), (127, 0x37), (128, 0x55), (255, 0x37)] {
        assert_eq!(
            space
                .read_native(
                    &native,
                    GuestVirtualAddress::new(PAGE_SIZE + offset),
                    1,
                    Some(&area)
                )
                .unwrap(),
            Some(expected)
        );
    }
    assert_eq!(
        space
            .read_native(
                &native,
                GuestVirtualAddress::new(PAGE_SIZE + 256),
                1,
                Some(&area)
            )
            .unwrap(),
        None
    );
}
