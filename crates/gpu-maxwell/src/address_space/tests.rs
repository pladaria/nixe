use super::*;
use crate::SWITCH_1_GM20B_PROFILE;
use nixe_memory::{Access, AccessOperation, AccessResult, Allocation};

fn address_space() -> MaxwellGpuAddressSpace {
    MaxwellGpuAddressSpace::new(
        Machine::new(0x100_0000).unwrap(),
        MaxwellAddressSpaceId::new(1),
        SWITCH_1_GM20B_PROFILE,
    )
}
fn initialized() -> MaxwellGpuAddressSpace {
    let mut space = address_space();
    space
        .initialize(MaxwellAddressSpaceInitialization::default())
        .unwrap();
    space
}
fn backing(space: &MaxwellGpuAddressSpace, bytes: u64) -> (Allocation, Vec<AllocationRange>) {
    let allocation = space
        .machine
        .account(bytes)
        .unwrap()
        .allocate(bytes)
        .unwrap();
    let ranges = vec![AllocationRange::new(allocation.pin(), 0, bytes).unwrap()];
    (allocation, ranges)
}
fn request(backing: Vec<AllocationRange>, size: u64) -> MaxwellMapRequest {
    MaxwellMapRequest {
        allocation: MaxwellAllocationId::new(1),
        backing,
        backing_offset: 0,
        size,
        allocation_alignment: 0x1000,
        page_size: 0,
        kind: 0,
        cacheable: false,
        permissions: MemoryPermissions::READ_WRITE,
        fixed_offset: None,
    }
}
fn write(machine: &Machine, ranges: &[AllocationRange], bytes: &[u8]) {
    assert!(matches!(
        machine
            .access_captured(ranges, AccessOperation::Write(bytes.into()))
            .unwrap(),
        Access::Ready(AccessResult::Written)
    ));
}
fn read(machine: &Machine, ranges: &[AllocationRange]) -> Vec<u8> {
    match machine
        .access_captured(ranges, AccessOperation::Read)
        .unwrap()
    {
        Access::Ready(AccessResult::Read(bytes)) => bytes.into_vec(),
        other => panic!("unexpected read: {other:?}"),
    }
}

#[test]
fn immediate_reads_cross_mappings_and_keep_original_inputs_during_device_remap() {
    let mut space = initialized();
    let machine = space.machine.clone();
    let (_, first) = backing(&space, 0x1000);
    let (_, second) = backing(&space, 0x1000);
    write(&machine, &first, &[11; 0x1000]);
    write(&machine, &second, &[22; 0x1000]);
    let left = space.map(request(first.clone(), 0x1000)).unwrap();
    let right = space.map(request(second.clone(), 0x1000)).unwrap();
    assert_eq!(right.offset().get(), left.offset().get() + 0x1000);
    let memory = space.memory().clone();
    let address = memory.checked_add(left.offset(), 0xffc).unwrap();
    let mut output = [0; 8];
    memory
        .read_into(address, &mut output, &mut |_, _| {
            panic!("RAM must be consumed directly")
        })
        .unwrap();
    assert_eq!(output, [11, 11, 11, 11, 22, 22, 22, 22]);
    machine.publish_device_write(&second, || Ok(())).unwrap();
    memory
        .read_into(address, &mut output, &mut |ranges, operation| {
            // The callback can suspend and change virtual mappings. Its inputs
            // must still name the physical bytes accepted before this boundary.
            space.unmap(left.offset()).unwrap();
            space.unmap(right.offset()).unwrap();
            let replacement = space.map(request(second.clone(), 0x1000)).unwrap();
            assert_eq!(replacement.offset(), left.offset());
            machine
                .apply_resolved_access(&ranges, &operation)
                .map_err(Arc::new)
        })
        .unwrap();
    assert_eq!(output, [11, 11, 11, 11, 22, 22, 22, 22]);
}

#[test]
fn address_space_retains_identity_and_immutable_profile() {
    let address_space = address_space();

    assert_eq!(address_space.id(), MaxwellAddressSpaceId::new(1));
    assert_eq!(address_space.profile(), SWITCH_1_GM20B_PROFILE);
    assert_eq!(address_space.profile_id(), SWITCH_1_GM20B_PROFILE.id());
    assert_eq!(
        address_space.id().to_string(),
        "gpu-address-space=0x0000000000000001"
    );
}

#[test]
fn address_arithmetic_uses_the_bound_profile_width() {
    let address_space = address_space();
    let last_page = address_space.address(0x00ff_ffff_f000).unwrap();

    assert_eq!(
        address_space.checked_add(last_page, 0x1000),
        Err(GpuVirtualAddressError::AddressOutOfRange {
            value: 0x0100_0000_0000,
            bits: 40,
        })
    );
}

#[test]
fn initialization_derives_two_regions_from_profile_and_state() {
    let mut address_space = address_space();
    address_space
        .initialize(MaxwellAddressSpaceInitialization::default())
        .unwrap();

    assert_eq!(
        address_space.regions().unwrap(),
        [
            MaxwellVaRegion {
                offset: address_space.address(0x0800_0000).unwrap(),
                page_size: 0x1000,
                pages: 0x3f_8000,
            },
            MaxwellVaRegion {
                offset: address_space.address(0x4_0000_0000).unwrap(),
                page_size: 0x2_0000,
                pages: 0xe_0000,
            },
        ]
    );
    assert_eq!(
        address_space.initialize(MaxwellAddressSpaceInitialization::default()),
        Err(MaxwellAddressSpaceError::AlreadyInitialized)
    );
}

#[test]
fn alternate_big_page_size_changes_default_start_and_regions() {
    let mut address_space = address_space();
    address_space
        .initialize(MaxwellAddressSpaceInitialization {
            big_page_size: 0x1_0000,
            ..Default::default()
        })
        .unwrap();
    let regions = address_space.regions().unwrap();

    assert_eq!(regions[0].offset().get(), 0x0400_0000);
    assert_eq!(regions[1].page_size(), 0x1_0000);
    assert_eq!(regions[1].pages(), 0x1c_0000);
}

#[test]
fn reservations_use_aligned_first_fit_and_exact_atomic_free() {
    let mut address_space = address_space();
    address_space
        .initialize(MaxwellAddressSpaceInitialization::default())
        .unwrap();
    let first = address_space.reserve(3, 0x1000, 0, 0x4000).unwrap();
    let fixed_offset = address_space.address(0x0801_0000).unwrap();
    let fixed = address_space
        .reserve(2, 0x1000, ALLOC_SPACE_FIXED, fixed_offset.get())
        .unwrap();
    let second = address_space.reserve(1, 0x1000, 0, 0x1000).unwrap();

    assert_eq!(first.offset().get(), 0x0800_0000);
    assert_eq!(second.offset().get(), 0x0800_3000);
    assert_eq!(fixed.offset(), fixed_offset);
    assert_eq!(address_space.reservation_count(), 3);
    assert_eq!(
        address_space.free(first.offset(), 2, first.page_size()),
        Err(MaxwellAddressSpaceError::ReservationShapeMismatch)
    );
    assert_eq!(address_space.reservation(first.offset()), Some(first));
    assert_eq!(
        address_space
            .free(first.offset(), first.pages(), first.page_size())
            .unwrap(),
        first
    );
    assert_eq!(address_space.reservation(first.offset()), None);
}

#[test]
fn reservation_validation_rejects_partial_or_invalid_state() {
    let mut address_space = address_space();
    assert_eq!(
        address_space.reserve(1, 0x1000, 0, 0x1000),
        Err(MaxwellAddressSpaceError::NotInitialized)
    );
    address_space
        .initialize(MaxwellAddressSpaceInitialization::default())
        .unwrap();

    assert_eq!(
        address_space.reserve(0, 0x1000, 0, 0x1000),
        Err(MaxwellAddressSpaceError::InvalidPageCount)
    );
    assert_eq!(
        address_space.reserve(1, 0x1000, ALLOC_SPACE_SPARSE, 0x1000),
        Err(MaxwellAddressSpaceError::SparseSmallPagesUnsupported)
    );
    assert_eq!(
        address_space.reserve(1, 0x2000, 0, 0x2000),
        Err(MaxwellAddressSpaceError::InvalidPageSize { page_size: 0x2000 })
    );
    assert_eq!(
        address_space.reserve(1, 0x1000, ALLOC_SPACE_FIXED, 0x0800_0001),
        Err(MaxwellAddressSpaceError::MisalignedAddress {
            address: 0x0800_0001,
            alignment: 0x1000,
        })
    );
}

#[test]
fn indexed_translation_uses_actual_physical_pages_across_leaf_boundaries() {
    let mut space = initialized();
    let (allocation, ranges) = backing(&space, 0x4000);
    let reservation = space
        .reserve(4, 0x1000, ALLOC_SPACE_FIXED, 0x081f_f000)
        .unwrap();
    let mut map = request(ranges.clone(), 0x4000);
    map.fixed_offset = Some(reservation.offset());
    map.kind = 0xfe;
    let mapping = space.map(map).unwrap();
    let physical: Vec<_> = ranges[0].physical_extents().collect();
    assert_eq!(physical.len(), 1);
    for offset in [0, 4095, 4096, 8192, 16383] {
        let translated = space
            .memory()
            .translate(
                space.address(mapping.offset.get() + offset).unwrap(),
                MemoryPermissions::READ,
            )
            .unwrap();
        assert_eq!(translated.allocation, allocation.identity());
        assert_eq!(translated.physical.get(), physical[0].start.get() + offset);
        assert_eq!(translated.bytes, 4096 - offset % 4096);
        assert_eq!(translated.kind, 0xfe);
    }
    let metadata = space.memory().translation_metadata_bytes();
    space.unmap(mapping.offset()).unwrap();
    assert_eq!(space.memory().translation_metadata_bytes(), metadata);
    assert!(matches!(
        space
            .memory()
            .translate(mapping.offset(), MemoryPermissions::READ),
        Err(MaxwellGpuAccessError::UnmappedAddress { .. })
    ));
}

#[test]
fn physical_aliases_and_accepted_inputs_survive_unmap_without_retaining_neighbors() {
    let mut space = initialized();
    let machine = space.machine.clone();
    let (allocation, ranges) = backing(&space, 0x4000);
    let mut map = request(ranges.clone(), 0x2000);
    map.backing_offset = 0x1000;
    let first = space.map(map.clone()).unwrap();
    map.allocation = MaxwellAllocationId::new(2);
    let alias = space.map(map).unwrap();
    assert_eq!(
        space
            .memory()
            .translate(first.offset(), MemoryPermissions::READ)
            .unwrap()
            .physical,
        space
            .memory()
            .translate(alias.offset(), MemoryPermissions::READ)
            .unwrap()
            .physical
    );
    let source = space
        .memory()
        .resolve_range(first.offset(), 4, MemoryPermissions::READ_WRITE)
        .unwrap();
    let accepted = space.memory().capture(&source).unwrap();
    write(&machine, &accepted, &[1, 2, 3, 4]);
    drop(ranges);
    drop(allocation);
    assert_eq!(machine.statistics().allocated_bytes, 0x2000);
    space.unmap(first.offset()).unwrap();
    assert!(!space.memory().retained_mapping_is_current(&first));
    assert!(space.memory().retained_mapping_is_current(&alias));
    space.unmap(alias.offset()).unwrap();
    assert_eq!(machine.statistics().allocated_bytes, 0x1000);
    assert_eq!(read(&machine, &accepted), [1, 2, 3, 4]);
    assert!(matches!(
        space.memory().capture(&source),
        Err(MaxwellGpuAccessError::StaleMapping { .. })
    ));
    drop(accepted);
    assert_eq!(machine.statistics().allocated_bytes, 0);
    assert!(first.physical().pin(0x1000, 4).is_err());
}

#[test]
fn cached_mapping_tokens_cannot_authorize_a_reused_slot_or_another_address_space() {
    let mut first_space = initialized();
    let (_, ranges) = backing(&first_space, 0x1000);
    let old = first_space.map(request(ranges.clone(), 0x1000)).unwrap();
    first_space.unmap(old.offset()).unwrap();
    let current = first_space.map(request(ranges.clone(), 0x1000)).unwrap();
    assert_eq!(old.slot, current.slot);
    assert!(!first_space.memory().retained_mapping_is_current(&old));
    let mut second_space = MaxwellGpuAddressSpace::new(
        first_space.machine.clone(),
        first_space.id(),
        first_space.profile(),
    );
    second_space
        .initialize(MaxwellAddressSpaceInitialization::default())
        .unwrap();
    let other = second_space.map(request(ranges, 0x1000)).unwrap();
    assert!(!second_space.memory().retained_mapping_is_current(&current));
    assert!(!first_space.memory().retained_mapping_is_current(&other));
    drop(first_space);
    assert!(!current.current.load(Ordering::Acquire));
}

#[test]
fn remap_changes_new_translation_but_not_an_accepted_physical_operation() {
    let mut space = initialized();
    let (_, first) = backing(&space, 0x1000);
    let (_, replacement) = backing(&space, 0x1000);
    let map = space.map(request(first, 0x1000)).unwrap();
    let source = space
        .memory()
        .resolve_range(map.offset(), 4, MemoryPermissions::READ_WRITE)
        .unwrap();
    let accepted = space.memory().capture(&source).unwrap();
    space.unmap(map.offset()).unwrap();
    let new_map = space.map(request(replacement, 0x1000)).unwrap();
    assert_eq!(new_map.offset(), map.offset());
    write(space.machine(), &accepted, &[7, 6, 5, 4]);
    assert_eq!(read(space.machine(), &accepted), [7, 6, 5, 4]);
    let current = space
        .memory()
        .resolve_range(new_map.offset(), 4, MemoryPermissions::READ)
        .unwrap();
    let mut bytes = [9; 4];
    assert!(matches!(
        space
            .memory()
            .read_resolved(&current, &mut bytes, &mut |ranges, operation| space
                .machine()
                .apply_resolved_access(&ranges, &operation)
                .map_err(Arc::new))
            .unwrap(),
        ()
    ));
    assert_eq!(bytes, [0; 4]);
}

#[test]
fn invalid_mapping_arguments_leave_active_state_unchanged() {
    let mut space = initialized();
    let (_, ranges) = backing(&space, 0x4000);
    let good = space.map(request(ranges.clone(), 0x1000)).unwrap();
    let original = space.mapping_dump(64);
    for bad in [
        MaxwellMapRequest {
            backing_offset: 0x4000,
            ..request(ranges.clone(), 0x1000)
        },
        MaxwellMapRequest {
            backing_offset: 1,
            ..request(ranges.clone(), 0x1000)
        },
        MaxwellMapRequest {
            kind: 0xff,
            ..request(ranges.clone(), 0x1000)
        },
        MaxwellMapRequest {
            permissions: MemoryPermissions::NONE,
            ..request(ranges, 0x1000)
        },
    ] {
        assert!(space.map(bad).is_err());
        assert_eq!(space.mapping_dump(64), original);
        assert!(space.memory().retained_mapping_is_current(&good));
    }
    let other = initialized();
    let (_, wrong_machine) = backing(&other, 0x1000);
    assert_eq!(
        space.map(request(wrong_machine, 0x1000)).unwrap_err(),
        MaxwellAddressSpaceError::InvalidBackingRange
    );
    assert_eq!(space.mapping_dump(64), original);
}

#[test]
fn partial_kind_replacement_invalidates_only_affected_mapping_and_preserves_bytes() {
    let mut space = initialized();
    let (allocation, ranges) = backing(&space, 0x4000);
    let mapping = space.map(request(ranges.clone(), 0x3000)).unwrap();
    let mut adjacent = request(ranges, 0x1000);
    adjacent.backing_offset = 0x3000;
    let unrelated = space.map(adjacent).unwrap();
    let physical = space
        .memory()
        .translate(mapping.offset(), MemoryPermissions::READ)
        .unwrap()
        .physical;
    drop(allocation);
    space
        .modify_mapping(mapping.offset(), 0x1000, 0x1000, 0xfe, true)
        .unwrap();
    assert!(!space.memory().retained_mapping_is_current(&mapping));
    assert!(space.memory().retained_mapping_is_current(&unrelated));
    assert_eq!(space.mapping_count(), 4);
    for (offset, kind) in [(0, 0), (0x1000, 0xfe), (0x2000, 0)] {
        let translated = space
            .memory()
            .translate(
                space.address(mapping.offset().get() + offset).unwrap(),
                MemoryPermissions::READ,
            )
            .unwrap();
        assert_eq!(translated.kind, kind);
        assert_eq!(translated.physical.get(), physical.get() + offset);
    }
    assert_eq!(space.unmap(mapping.offset()).unwrap().len(), 3);
    assert_eq!(space.machine.statistics().allocated_bytes, 0x1000);
    assert!(space.memory().retained_mapping_is_current(&unrelated));
}

#[test]
fn sparse_replacement_splits_only_overlap_and_releases_removed_physical_pages() {
    let mut space = initialized();
    let reservation = space
        .reserve(3, 0x20000, ALLOC_SPACE_SPARSE, 0x20000)
        .unwrap();
    let (allocation, ranges) = backing(&space, 0x60000);
    let sparse = |offset, size, backing, backing_offset| MaxwellSparseRemapRequest {
        offset,
        size,
        mapping: Some(MaxwellSparseMapping {
            allocation: MaxwellAllocationId::new(1),
            backing,
            backing_offset,
            kind: 0xfe,
            cacheable: true,
            permissions: MemoryPermissions::READ_WRITE,
        }),
    };
    space
        .remap_sparse(vec![sparse(
            reservation.offset(),
            0x60000,
            ranges.clone(),
            0,
        )])
        .unwrap();
    let original = space.mapping(reservation.offset()).unwrap();
    drop(ranges);
    drop(allocation);
    let middle = space.address(reservation.offset().get() + 0x20000).unwrap();
    let before = space.mapping_dump(64);
    assert_eq!(
        space.remap_sparse(vec![
            MaxwellSparseRemapRequest {
                offset: middle,
                size: 0x20000,
                mapping: None
            },
            MaxwellSparseRemapRequest {
                offset: middle,
                size: 0x20000,
                mapping: None
            },
        ]),
        Err(MaxwellAddressSpaceError::OverlappingRemapEntries)
    );
    assert_eq!(space.mapping_dump(64), before);
    assert!(space.memory().retained_mapping_is_current(&original));
    space
        .remap_sparse(vec![MaxwellSparseRemapRequest {
            offset: middle,
            size: 0x20000,
            mapping: None,
        }])
        .unwrap();
    assert_eq!(space.mapping_count(), 2);
    assert_eq!(space.machine.statistics().allocated_bytes, 0x40000);
    assert!(!space.memory().retained_mapping_is_current(&original));
    assert!(matches!(
        space
            .memory()
            .resolve_range(reservation.offset(), 0x60000, MemoryPermissions::READ),
        Err(MaxwellGpuAccessError::UnmappedAddress { .. })
    ));
    space.free(reservation.offset(), 3, 0x20000).unwrap();
    assert_eq!(space.machine.statistics().allocated_bytes, 0);
}

#[test]
fn generation_and_identity_exhaustion_leave_mappings_unchanged() {
    let mut space = initialized();
    let (_, ranges) = backing(&space, 0x1000);
    space.mapping_generation = MappingGeneration::MAX;
    assert!(matches!(
        space.map(request(ranges.clone(), 0x1000)),
        Err(MaxwellAddressSpaceError::GenerationExhausted(_))
    ));
    assert_eq!(space.mapping_count(), 0);
    space.mapping_generation = MappingGeneration::INITIAL;
    space.next_mapping_id = u64::MAX;
    assert_eq!(
        space.map(request(ranges, 0x1000)).unwrap_err(),
        MaxwellAddressSpaceError::MappingIdentityExhausted
    );
    assert_eq!(space.mapping_count(), 0);
}

#[test]
fn scatter_gather_reads_preserve_permissions_holes_and_original_physical_order() {
    let mut space = initialized();
    let (_, ranges) = backing(&space, 0x3000);
    let one = space.map(request(ranges.clone(), 0x1000)).unwrap();
    let mut map = request(ranges.clone(), 0x1000);
    map.backing_offset = 0x2000;
    map.permissions = MemoryPermissions::READ;
    let two = space.map(map).unwrap();
    let first = capture_subrange(&ranges, 0xffe, 2).unwrap();
    write(space.machine(), &first, &[1, 2]);
    let last = capture_subrange(&ranges, 0x2000, 2).unwrap();
    write(space.machine(), &last, &[3, 4]);
    let start = space.address(one.offset().get() + 0xffe).unwrap();
    let source = space
        .memory()
        .resolve_range(start, 4, MemoryPermissions::READ)
        .unwrap();
    assert_eq!(source.segments().len(), 2);
    let mut bytes = [0; 4];
    assert!(matches!(
        space
            .memory()
            .read_resolved(&source, &mut bytes, &mut |ranges, operation| space
                .machine()
                .apply_resolved_access(&ranges, &operation)
                .map_err(Arc::new))
            .unwrap(),
        ()
    ));
    assert_eq!(bytes, [1, 2, 3, 4]);
    assert!(matches!(
        space
            .memory()
            .resolve_range(start, 4, MemoryPermissions::WRITE),
        Err(MaxwellGpuAccessError::PermissionDenied { .. })
    ));
    assert!(matches!(
        space
            .memory()
            .translate(two.offset(), MemoryPermissions::WRITE),
        Err(MaxwellGpuAccessError::PermissionDenied { .. })
    ));
    assert!(matches!(
        space
            .memory()
            .read_resolved(&source, &mut [0; 3], &mut |_, _| panic!(
                "size mismatch cannot access memory"
            )),
        Err(MaxwellGpuAccessError::OutputSizeMismatch { .. })
    ));
    space.unmap(two.offset()).unwrap();
    assert!(matches!(
        space
            .memory()
            .read_resolved(&source, &mut [0; 4], &mut |_, _| panic!(
                "stale mapping cannot access memory"
            )),
        Err(MaxwellGpuAccessError::StaleMapping { .. })
    ));
    assert!(matches!(
        space
            .memory()
            .resolve_range(start, 4, MemoryPermissions::READ),
        Err(MaxwellGpuAccessError::UnmappedAddress { .. })
    ));
}

#[test]
fn mapping_diagnostics_are_ordered_bounded_and_do_not_retain_ram() {
    let mut space = initialized();
    let (allocation, ranges) = backing(&space, 0x1000);
    for _ in 0..=MAX_MAPPING_DUMP_ENTRIES {
        space.map(request(ranges.clone(), 0x1000)).unwrap();
    }
    let dump = space.mapping_dump(usize::MAX);
    assert_eq!(dump.entries.len(), MAX_MAPPING_DUMP_ENTRIES);
    assert_eq!(dump.omitted_mappings(), 1);
    assert!(
        dump.entries
            .windows(2)
            .all(|pair| pair[0].offset < pair[1].offset)
    );
    let text = dump.to_string();
    assert!(text.contains("permissions=0x3"));
    assert!(text.contains("first-physical=0x"));
    assert!(text.ends_with("mappings-truncated=1"));
    drop(allocation);
    drop(ranges);
    let machine = space.machine.clone();
    drop(space);
    assert_eq!(machine.statistics().allocated_bytes, 0);
}

#[test]
fn shared_reader_observes_remap_and_never_keeps_retired_mapping_pins_alive() {
    let mut space = initialized();
    let machine = space.machine().clone();
    let (first, inputs) = backing(&space, 0x1000);
    let initial = space.map(request(inputs, 0x1000)).unwrap();
    let reader = space.memory().clone();
    let source = reader
        .resolve_range(initial.offset(), 4, MemoryPermissions::READ)
        .unwrap();
    let accepted = reader.capture(&source).unwrap();
    write(&machine, &accepted, &[1, 2, 3, 4]);
    let old_identity = first.identity();
    drop(first);
    space.unmap(initial.offset()).unwrap();
    let (replacement, inputs) = backing(&space, 0x1000);
    let current = space.map(request(inputs, 0x1000)).unwrap();
    assert_eq!(current.offset(), initial.offset());
    let translated = reader
        .translate(current.offset(), MemoryPermissions::READ)
        .unwrap();
    assert_eq!(translated.allocation, replacement.identity());
    assert_ne!(translated.allocation, old_identity);
    assert!(!reader.retained_mapping_is_current(&initial));
    assert_eq!(read(&machine, &accepted), [1, 2, 3, 4]);
    drop(replacement);
    drop(space);
    assert!(!reader.initialized());
    assert_eq!(machine.statistics().allocated_bytes, 0x1000);
    assert_eq!(read(&machine, &accepted), [1, 2, 3, 4]);
    drop(accepted);
    assert_eq!(machine.statistics().allocated_bytes, 0);
}

#[test]
fn concurrent_translation_never_mixes_identity_permissions_kind_and_physical_page() {
    let mut space = initialized();
    let (first, inputs_one) = backing(&space, 0x1000);
    let (second, inputs_two) = backing(&space, 0x1000);
    let mapping = space.map(request(inputs_one.clone(), 0x1000)).unwrap();
    let address = mapping.offset();
    let reader = space.memory().clone();
    let first_physical = inputs_one[0].physical_extents().next().unwrap().start;
    let second_physical = inputs_two[0].physical_extents().next().unwrap().start;
    let first_id = first.identity();
    let second_id = second.identity();
    let stopped = Arc::new(AtomicBool::new(false));
    let ready = Arc::new(std::sync::Barrier::new(2));
    let done = stopped.clone();
    let barrier = ready.clone();
    let worker = std::thread::spawn(move || {
        let initial = reader.translate(address, MemoryPermissions::READ).unwrap();
        assert_eq!(initial.allocation, first_id);
        assert_eq!(initial.physical, first_physical);
        assert_eq!(initial.permissions, MemoryPermissions::READ_WRITE);
        assert_eq!(initial.kind, 0);
        barrier.wait();
        let mut observed = 1;
        while !done.load(Ordering::Acquire) {
            match reader.translate(address, MemoryPermissions::READ) {
                Ok(value) if value.allocation == first_id => {
                    assert_eq!(value.physical, first_physical);
                    assert_eq!(value.permissions, MemoryPermissions::READ_WRITE);
                    assert_eq!(value.kind, 0);
                    observed += 1;
                }
                Ok(value) => {
                    assert_eq!(value.allocation, second_id);
                    assert_eq!(value.physical, second_physical);
                    assert_eq!(value.permissions, MemoryPermissions::READ);
                    assert_eq!(value.kind, 0x10);
                    observed += 1;
                }
                Err(MaxwellGpuAccessError::UnmappedAddress { .. }) => {}
                Err(error) => panic!("unexpected translation failure: {error}"),
            }
        }
        observed
    });
    ready.wait();
    for index in 0..1000 {
        space.unmap(address).unwrap();
        let mut next = request(
            if index % 2 == 0 {
                inputs_two.clone()
            } else {
                inputs_one.clone()
            },
            0x1000,
        );
        if index % 2 == 0 {
            next.permissions = MemoryPermissions::READ;
            next.kind = 0x10;
        }
        assert_eq!(space.map(next).unwrap().offset(), address);
    }
    stopped.store(true, Ordering::Release);
    assert!(worker.join().unwrap() > 0);
}

#[test]
fn direct_ram_write_validates_the_whole_mapping_before_modifying_any_prefix() {
    let mut space = initialized();
    let (_allocation, ranges) = backing(&space, 0x2000);
    let mapping = space.map(request(ranges.clone(), 0x1000)).unwrap();
    let address = space.address(mapping.offset().get() + 0xffe).unwrap();
    assert!(matches!(
        space
            .memory()
            .write_from(address, &[1, 2, 3, 4], &mut |_| panic!(
                "invalid write cannot escape"
            )),
        Err(MaxwellGpuAccessError::UnmappedAddress { .. })
    ));
    assert_eq!(&read(&space.machine, &ranges)[0xffe..0x1002], &[0; 4]);
    space
        .memory()
        .write_from(mapping.offset(), &[1, 2, 3, 4], &mut |_| {
            panic!("ordinary RAM write must be direct")
        })
        .unwrap();
    assert_eq!(&read(&space.machine, &ranges)[..4], &[1, 2, 3, 4]);
}

#[test]
fn observed_owner_write_invalidates_without_a_retained_request() {
    let mut space = initialized();
    let (_old, ranges) = backing(&space, 0x1000);
    let mapping = space.map(request(ranges.clone(), 0x1000)).unwrap();
    let observation = nixe_memory::Observation::default();
    space
        .machine
        .access_captured(&ranges, AccessOperation::ReadObserved(observation.clone()))
        .unwrap()
        .into_ready()
        .unwrap();
    space
        .memory()
        .write_from(mapping.offset(), &[4, 3, 2, 1], &mut |_| {
            panic!("CPU observation requires no device-authority update")
        })
        .unwrap();
    assert!(observation.is_invalid());
    assert_eq!(&read(&space.machine, &ranges)[..4], &[4, 3, 2, 1]);
}
