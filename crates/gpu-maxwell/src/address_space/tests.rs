use super::*;
use crate::SWITCH_1_GM20B_PROFILE;
use nixe_memory::{Access, AccessOperation, AccessResult, Allocation, capture_subrange};

fn address_space() -> MaxwellGpuAddressSpace {
    MaxwellGpuAddressSpace::new(
        crate::MaxwellDeviceMemory::new(Machine::new(0x100_0000).unwrap()),
        MaxwellAddressSpaceId::new(1),
        SWITCH_1_GM20B_PROFILE,
    )
}
fn initialized() -> (MaxwellGpuAddressSpace, MaxwellGpuMemory) {
    let mut space = address_space();
    space
        .initialize(MaxwellAddressSpaceInitialization::default())
        .unwrap();
    let gpu = MaxwellGpuMemory::new(space.memory().clone());
    (space, gpu)
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
fn request(
    space: &MaxwellGpuAddressSpace,
    backing: Vec<AllocationRange>,
    size: u64,
) -> MaxwellMapRequest {
    MaxwellMapRequest {
        allocation: MaxwellAllocationId::new(1),
        backing: space
            .memory()
            .device_memory()
            .map(backing.into(), None)
            .unwrap(),
        backing_offset: 0,
        size,
        allocation_alignment: 0x1000,
        page_size: 0,
        kind: 0,
        cacheable: false,

        fixed_offset: None,
    }
}
fn write(machine: &Machine, ranges: &[AllocationRange], bytes: &[u8]) {
    assert!(matches!(
        machine
            .access_captured(ranges, AccessOperation::Write(bytes.into()))
            .unwrap(),
        Access::Ready(AccessResult::Written(_))
    ));
}
fn read(machine: &Machine, ranges: &[AllocationRange]) -> Vec<u8> {
    match machine
        .access_captured(ranges, AccessOperation::Read(Vec::new()))
        .unwrap()
    {
        Access::Ready(AccessResult::Read(bytes)) => bytes,
        other => panic!("unexpected read: {other:?}"),
    }
}

#[test]
fn immediate_reads_consume_current_device_tables_after_each_flush() {
    let (mut space, gpu) = initialized();
    let machine = space.machine.clone();
    let (_, first) = backing(&space, 0x1000);
    let (_, second) = backing(&space, 0x1000);
    write(&machine, &first, &[11; 0x1000]);
    write(&machine, &second, &[22; 0x1000]);
    let left = space.map(request(&space, first.clone(), 0x1000)).unwrap();
    let right = space.map(request(&space, second.clone(), 0x1000)).unwrap();
    assert_eq!(right.offset().get(), left.offset().get() + 0x1000);
    let memory = gpu;
    let address = memory.checked_add(left.offset(), 0xffc).unwrap();
    let mut output = [0; 8];
    memory
        .read_into(
            address,
            &mut output,
            &mut crate::test_memory::access(&machine),
        )
        .unwrap();
    assert_eq!(output, [11, 11, 11, 11, 22, 22, 22, 22]);
    let mut first_flush = true;
    memory
        .read_into(
            address,
            &mut output,
            &mut crate::test_memory::safe_read(|_, _| {
                if first_flush {
                    first_flush = false;
                    space.unmap(left.offset()).unwrap();
                    space.unmap(right.offset()).unwrap();
                    let replacement = space.map(request(&space, second.clone(), 0x1000)).unwrap();
                    assert_eq!(replacement.offset(), left.offset());
                }
                Ok(())
            }),
        )
        .unwrap();
    // MemoryManager::ReadBlockImpl resolves DAddr, flushes, then reads its
    // current device PTE. A later hardware page consults the current GPU table.
    assert_eq!(output, [22, 22, 22, 22, 0, 0, 0, 0]);
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
fn device_page_tables_translate_small_page_boundaries_and_unmap() {
    let (mut space, gpu) = initialized();
    let (_allocation, ranges) = backing(&space, 0x4000);
    let reservation = space
        .reserve(4, 0x1000, ALLOC_SPACE_FIXED, 0x081f_f000)
        .unwrap();
    let mut map = request(&space, ranges.clone(), 0x4000);
    map.fixed_offset = Some(reservation.offset());
    map.kind = 0xfe;
    let device = map.backing.address();
    let mapping = space.map(map).unwrap();
    let physical = ranges[0].physical_extents().next().unwrap().start;
    for offset in [0, 4095, 4096, 8192, 16383] {
        let address = space.address(mapping.offset.get() + offset).unwrap();
        let translated = gpu.device_address(address).unwrap();
        assert_eq!(translated, device + offset);
        assert_eq!(
            gpu.device_memory()
                .device()
                .physical_address(translated)
                .unwrap()
                .get(),
            physical.get() + offset
        );
        assert_eq!(gpu.page_kind(address).unwrap(), 0xfe);
    }
    space.unmap(mapping.offset()).unwrap();
    assert!(matches!(
        gpu.device_address(mapping.offset()),
        Err(MaxwellGpuAccessError::UnmappedAddress { .. })
    ));
}

#[test]
fn nvmap_pins_survive_aliases_and_final_unmap_leaves_only_exact_accepted_inputs() {
    let (mut space, gpu) = initialized();
    let machine = space.machine.clone();
    let (allocation, ranges) = backing(&space, 0x4000);
    let mut map = request(&space, ranges.clone(), 0x2000);
    map.backing_offset = 0x1000;
    let first = space.map(map.clone()).unwrap();
    map.allocation = MaxwellAllocationId::new(2);
    let alias = space.map(map).unwrap();
    assert_eq!(
        gpu.device_address(first.offset()).unwrap(),
        gpu.device_address(alias.offset()).unwrap()
    );
    let source = gpu.resolve_range(first.offset(), 4).unwrap();
    let accepted = gpu.capture(&source).unwrap();
    write(&machine, &accepted, &[1, 2, 3, 4]);
    drop(ranges);
    drop(allocation);
    assert_eq!(machine.statistics().allocated_bytes, 0x4000);
    space.unmap(first.offset()).unwrap();
    assert!(!gpu.retained_mapping_is_current(&first));
    assert!(gpu.retained_mapping_is_current(&alias));
    space.unmap(alias.offset()).unwrap();
    assert_eq!(machine.statistics().allocated_bytes, 0x1000);
    assert_eq!(read(&machine, &accepted), [1, 2, 3, 4]);
    assert!(matches!(
        gpu.capture(&source),
        Err(MaxwellGpuAccessError::StaleMapping { .. })
    ));
    drop(accepted);
    assert_eq!(machine.statistics().allocated_bytes, 0);
    assert!(first.physical().pin(0x1000, 4).is_err());
}

#[test]
fn retained_mapping_metadata_rejects_reused_addresses_and_other_address_spaces() {
    let (mut first_space, first_gpu) = initialized();
    let (_, ranges) = backing(&first_space, 0x1000);
    let old = first_space
        .map(request(&first_space, ranges.clone(), 0x1000))
        .unwrap();
    first_space.unmap(old.offset()).unwrap();
    let current = first_space
        .map(request(&first_space, ranges.clone(), 0x1000))
        .unwrap();
    assert_eq!(old.offset(), current.offset());
    assert!(!first_gpu.retained_mapping_is_current(&old));
    let mut second_space = MaxwellGpuAddressSpace::new(
        crate::MaxwellDeviceMemory::new(first_space.machine.clone()),
        first_space.id(),
        first_space.profile(),
    );
    second_space
        .initialize(MaxwellAddressSpaceInitialization::default())
        .unwrap();
    let second_gpu = MaxwellGpuMemory::new(second_space.memory().clone());
    let other = second_space
        .map(request(&second_space, ranges, 0x1000))
        .unwrap();
    assert!(!second_gpu.retained_mapping_is_current(&current));
    assert!(!first_gpu.retained_mapping_is_current(&other));
    drop(first_space);
    assert!(!current.current.load(Ordering::Acquire));
}

#[test]
fn invalid_mapping_arguments_leave_active_state_unchanged() {
    let (mut space, gpu) = initialized();
    let (_, ranges) = backing(&space, 0x4000);
    let good = space.map(request(&space, ranges.clone(), 0x1000)).unwrap();
    let original = space.mapping_dump(64);
    for bad in [
        MaxwellMapRequest {
            backing_offset: 0x4000,
            ..request(&space, ranges.clone(), 0x1000)
        },
        MaxwellMapRequest {
            backing_offset: 1,
            ..request(&space, ranges.clone(), 0x1000)
        },
        MaxwellMapRequest {
            kind: 0xff,
            ..request(&space, ranges.clone(), 0x1000)
        },
    ] {
        assert!(space.map(bad).is_err());
        assert_eq!(space.mapping_dump(64), original);
        assert!(gpu.retained_mapping_is_current(&good));
    }
    let (other, _other_gpu) = initialized();
    let (_, wrong_machine) = backing(&other, 0x1000);
    assert_eq!(
        space
            .map(request(&other, wrong_machine, 0x1000))
            .unwrap_err(),
        MaxwellAddressSpaceError::InvalidBackingRange
    );
    assert_eq!(space.mapping_dump(64), original);
}

#[test]
fn partial_kind_replacement_invalidates_only_affected_mapping_and_preserves_bytes() {
    let (mut space, gpu) = initialized();
    let (allocation, ranges) = backing(&space, 0x4000);
    let mapping = space.map(request(&space, ranges.clone(), 0x3000)).unwrap();
    let mut adjacent = request(&space, ranges, 0x1000);
    adjacent.backing_offset = 0x3000;
    let unrelated = space.map(adjacent).unwrap();
    let device = gpu.device_address(mapping.offset()).unwrap();
    drop(allocation);
    space
        .modify_mapping(mapping.offset(), 0x1000, 0x1000, 0xfe, true)
        .unwrap();
    assert!(!gpu.retained_mapping_is_current(&mapping));
    assert!(gpu.retained_mapping_is_current(&unrelated));
    assert_eq!(space.mapping_count(), 4);
    for (offset, kind) in [(0, 0), (0x1000, 0xfe), (0x2000, 0)] {
        let address = space.address(mapping.offset().get() + offset).unwrap();
        assert_eq!(gpu.page_kind(address).unwrap(), kind);
        assert_eq!(gpu.device_address(address).unwrap(), device + offset);
    }
    assert_eq!(space.unmap(mapping.offset()).unwrap().len(), 3);
    assert_eq!(space.machine.statistics().allocated_bytes, 0x4000);
    assert!(gpu.retained_mapping_is_current(&unrelated));
    space.unmap(unrelated.offset()).unwrap();
    assert_eq!(space.machine.statistics().allocated_bytes, 0);
}

#[test]
fn sparse_replacement_changes_gpu_coverage_and_retains_nvmap_until_final_unmap() {
    let (mut space, gpu) = initialized();
    let reservation = space
        .reserve(3, 0x20000, ALLOC_SPACE_SPARSE, 0x20000)
        .unwrap();
    let (allocation, ranges) = backing(&space, 0x60000);
    let device = space.memory().device_memory().clone();
    let sparse =
        |offset, size, backing: Vec<AllocationRange>, backing_offset| MaxwellSparseRemapRequest {
            offset,
            size,
            mapping: Some(MaxwellSparseMapping {
                allocation: MaxwellAllocationId::new(1),
                backing: device.map(backing.into(), None).unwrap(),
                backing_offset,
                kind: 0xfe,
                cacheable: true,
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
    assert!(gpu.retained_mapping_is_current(&original));
    space
        .remap_sparse(vec![MaxwellSparseRemapRequest {
            offset: middle,
            size: 0x20000,
            mapping: None,
        }])
        .unwrap();
    assert_eq!(space.mapping_count(), 2);
    assert_eq!(space.machine.statistics().allocated_bytes, 0x60000);
    assert!(!gpu.retained_mapping_is_current(&original));
    assert!(matches!(
        gpu.resolve_range(reservation.offset(), 0x60000),
        Err(MaxwellGpuAccessError::UnmappedAddress { .. })
    ));
    space.free(reservation.offset(), 3, 0x20000).unwrap();
    assert_eq!(space.machine.statistics().allocated_bytes, 0);
}

#[test]
fn generation_and_identity_exhaustion_leave_mappings_unchanged() {
    let (mut space, _gpu) = initialized();
    let (_, ranges) = backing(&space, 0x1000);
    space.mapping_generation = MappingGeneration::MAX;
    assert!(matches!(
        space.map(request(&space, ranges.clone(), 0x1000)),
        Err(MaxwellAddressSpaceError::GenerationExhausted(_))
    ));
    assert_eq!(space.mapping_count(), 0);
    space.mapping_generation = MappingGeneration::INITIAL;
    space.next_mapping_id = u64::MAX;
    assert_eq!(
        space.map(request(&space, ranges, 0x1000)).unwrap_err(),
        MaxwellAddressSpaceError::MappingIdentityExhausted
    );
    assert_eq!(space.mapping_count(), 0);
}

#[test]
fn mapping_diagnostics_are_ordered_bounded_and_do_not_retain_ram() {
    let (mut space, _gpu) = initialized();
    let (allocation, ranges) = backing(&space, 0x1000);
    for _ in 0..=MAX_MAPPING_DUMP_ENTRIES {
        space.map(request(&space, ranges.clone(), 0x1000)).unwrap();
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
    let (mut space, gpu) = initialized();
    let machine = space.machine().clone();
    let (first, inputs) = backing(&space, 0x1000);
    let initial = space.map(request(&space, inputs, 0x1000)).unwrap();
    let reader = gpu;
    let source = reader.resolve_range(initial.offset(), 4).unwrap();
    let accepted = reader.capture(&source).unwrap();
    write(&machine, &accepted, &[1, 2, 3, 4]);
    let old_physical = accepted[0].physical_extents().next().unwrap().start;
    drop(first);
    space.unmap(initial.offset()).unwrap();
    let (replacement, inputs) = backing(&space, 0x1000);
    let current = space.map(request(&space, inputs, 0x1000)).unwrap();
    assert_eq!(current.offset(), initial.offset());
    let device = reader.device_address(current.offset()).unwrap();
    let physical = reader
        .device_memory()
        .device()
        .physical_address(device)
        .unwrap();
    let expected = AllocationRange::new(replacement.pin(), 0, 4).unwrap();
    assert_eq!(physical, expected.physical_extents().next().unwrap().start);
    assert_ne!(physical, old_physical);
    drop(expected);
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
fn concurrent_translation_observes_only_published_device_addresses() {
    let (mut space, reader) = initialized();
    let (_first, inputs_one) = backing(&space, 0x1000);
    let (_second, inputs_two) = backing(&space, 0x1000);
    let one = request(&space, inputs_one, 0x1000);
    let mut two = request(&space, inputs_two, 0x1000);
    two.kind = 0x10;
    let device_one = one.backing.address();
    let device_two = two.backing.address();
    let address = space.map(one.clone()).unwrap().offset();
    let stopped = Arc::new(AtomicBool::new(false));
    let ready = Arc::new(std::sync::Barrier::new(2));
    let done = stopped.clone();
    let barrier = ready.clone();
    let worker = std::thread::spawn(move || {
        assert_eq!(reader.device_address(address).unwrap(), device_one);
        barrier.wait();
        let mut observed = 1;
        while !done.load(Ordering::Acquire) {
            match reader.device_address(address) {
                Ok(device) => {
                    assert!(device == device_one || device == device_two);
                    observed += 1;
                }
                Err(MaxwellGpuAccessError::UnmappedAddress { .. }) => (),
                Err(error) => panic!("unexpected translation failure: {error}"),
            }
        }
        observed
    });
    ready.wait();
    for index in 0..1000 {
        space.unmap(address).unwrap();
        assert_eq!(
            space
                .map(if index % 2 == 0 {
                    two.clone()
                } else {
                    one.clone()
                })
                .unwrap()
                .offset(),
            address
        );
    }
    stopped.store(true, Ordering::Release);
    assert!(worker.join().unwrap() > 0);
}

#[test]
fn raw_upload_writes_mapped_prefixes_and_advances_over_gpu_holes() {
    let (mut space, mut gpu) = initialized();
    let (_allocation, ranges) = backing(&space, 0x2000);
    let mapping = space.map(request(&space, ranges.clone(), 0x1000)).unwrap();
    let address = space.address(mapping.offset().get() + 0xffe).unwrap();
    gpu.write_block_cached(address, &[1, 2, 3, 4]).unwrap();
    assert_eq!(&read(&space.machine, &ranges)[0xffe..0x1002], &[1, 2, 0, 0]);
    let mut visible = [0xff; 4];
    gpu.read_into_unsafe(address, &mut visible).unwrap();
    assert_eq!(visible, [1, 2, 0, 0]);
    gpu.write_block_cached(mapping.offset(), &[1, 2, 3, 4])
        .unwrap();
    assert_eq!(&read(&space.machine, &ranges)[..4], &[1, 2, 3, 4]);
}

#[test]
fn continuous_device_range_crosses_gpu_mappings_and_stops_at_holes_or_device_aliases() {
    let (mut space, gpu) = initialized();
    let (_allocation, ranges) = backing(&space, 0x3000);
    let mut first_request = request(&space, ranges, 0x1000);
    let shared = first_request.backing.clone();
    let first = space.map(first_request.clone()).unwrap();
    first_request.backing = shared;
    first_request.backing_offset = 0x1000;
    let second = space.map(first_request).unwrap();
    assert_eq!(second.offset().get(), first.offset().get() + 0x1000);
    let memory = &gpu;
    assert_eq!(
        memory.max_continuous_range(first.offset(), 0x3000).unwrap(),
        0x2000
    );
    assert_eq!(
        memory
            .max_continuous_range(memory.checked_add(first.offset(), 0xff0).unwrap(), 0x30)
            .unwrap(),
        0x30
    );
    let (_other, other_ranges) = backing(&space, 0x1000);
    let third = space.map(request(&space, other_ranges, 0x1000)).unwrap();
    assert_eq!(third.offset().get(), first.offset().get() + 0x2000);
    assert_eq!(
        gpu.max_continuous_range(first.offset(), 0x3000).unwrap(),
        0x2000
    );
    space.unmap(second.offset()).unwrap();
    assert_eq!(
        gpu.max_continuous_range(first.offset(), 0x3000).unwrap(),
        0x1000
    );
    assert_eq!(
        gpu.max_continuous_range(second.offset(), 0x1000).unwrap(),
        0
    );
}

#[derive(Default)]
struct InvalidationObserver {
    invalidations: Vec<Vec<nixe_gpu::native::GuestBufferRange>>,
    inline: Vec<Vec<nixe_gpu::native::GuestBufferRange>>,
}
impl nixe_gpu::OwnerMemory for InvalidationObserver {
    fn invalidate_gpu_cache(&mut self) {}
    fn flush_region(&mut self, _: u64, _: u64) -> Result<(), Arc<MemoryError>> {
        panic!("cached writes must not wait for native downloads");
    }
    fn invalidate_regions(
        &mut self,
        regions: &[nixe_gpu::native::GuestBufferRange],
    ) -> Result<(), Arc<MemoryError>> {
        self.invalidations.push(regions.to_vec());
        Ok(())
    }
    fn inline_memory(
        &mut self,
        regions: &[nixe_gpu::native::GuestBufferRange],
    ) -> Result<(), Arc<MemoryError>> {
        self.inline.push(regions.to_vec());
        Ok(())
    }
    fn access_memory(
        &mut self,
        _: Vec<AllocationRange>,
        _: AccessOperation,
    ) -> Result<AccessResult, Arc<MemoryError>> {
        panic!("the manager writes current RAM directly");
    }
}

#[test]
fn cached_writes_change_ram_immediately_and_group_invalidations_until_consumption() {
    let (mut space, mut gpu) = initialized();
    let (_allocation, ranges) = backing(&space, 0x1000);
    let mapping = space.map(request(&space, ranges, 0x1000)).unwrap();
    let base = mapping.offset().get();
    let device = gpu.device_address(mapping.offset()).unwrap();
    let mut observer = InvalidationObserver::default();
    for (offset, value) in [(64, 1), (68, 2), (96, 3)] {
        gpu.write_block_cached(gpu.address(base + offset).unwrap(), &[value; 4])
            .unwrap();
    }
    let mut current = [0; 4];
    gpu.read_into_unsafe(gpu.address(base + 68).unwrap(), &mut current)
        .unwrap();
    assert_eq!(current, [2; 4]);
    assert!(observer.invalidations.is_empty());
    assert!(observer.inline.is_empty());
    gpu.flush_caching(&mut observer).unwrap();
    assert_eq!(
        observer.invalidations,
        [vec![nixe_gpu::native::GuestBufferRange {
            address: device + 64,
            size: 64
        }]]
    );
    gpu.flush_caching(&mut observer).unwrap();
    assert_eq!(
        observer.invalidations.len(),
        1,
        "an empty consumer does not revisit caches"
    );
    gpu.write_inline(gpu.address(base + 128).unwrap(), &[4; 4], &mut observer)
        .unwrap();
    assert_eq!(
        observer.inline,
        [vec![nixe_gpu::native::GuestBufferRange {
            address: device + 128,
            size: 4
        }]]
    );
    gpu.flush_caching(&mut observer).unwrap();
    assert_eq!(
        observer.invalidations.len(),
        1,
        "inline uploads are not cached CB writes"
    );
}

#[test]
fn cached_invalidation_uses_current_gpu_mappings_and_skips_unmapped_pages() {
    let (mut space, mut gpu) = initialized();
    let (_a, first) = backing(&space, 0x1000);
    let (_b, second) = backing(&space, 0x1000);
    let mapping = space.map(request(&space, first.clone(), 0x1000)).unwrap();
    gpu.write_block_cached(mapping.offset(), &[7; 4]).unwrap();
    space.unmap(mapping.offset()).unwrap();
    let replacement = space.map(request(&space, second.clone(), 0x1000)).unwrap();
    assert_eq!(replacement.offset(), mapping.offset());
    let device = gpu.device_address(replacement.offset()).unwrap();
    let mut observer = InvalidationObserver::default();
    gpu.flush_caching(&mut observer).unwrap();
    assert_eq!(
        observer.invalidations,
        [vec![nixe_gpu::native::GuestBufferRange {
            address: device,
            size: 32
        }]]
    );
    assert_eq!(&read(&space.machine, &first)[..4], &[7; 4]);
    assert_eq!(&read(&space.machine, &second)[..4], &[0; 4]);
    gpu.write_block_cached(replacement.offset(), &[9; 4])
        .unwrap();
    space.unmap(replacement.offset()).unwrap();
    gpu.flush_caching(&mut observer).unwrap();
    assert_eq!(observer.invalidations.last().unwrap(), &[]);
}

#[test]
fn cached_invalidation_splits_sparse_holes_and_uses_remapped_device_addresses() {
    let (mut space, mut gpu) = initialized();
    let reservation = space
        .reserve(3, 0x20000, ALLOC_SPACE_SPARSE, 0x20000)
        .unwrap();
    let (_allocation, ranges) = backing(&space, 0x60000);
    let device = space
        .memory()
        .device_memory()
        .map(ranges.into(), None)
        .unwrap();
    space
        .remap_sparse(vec![MaxwellSparseRemapRequest {
            offset: reservation.offset(),
            size: 0x60000,
            mapping: Some(MaxwellSparseMapping {
                allocation: MaxwellAllocationId::new(1),
                backing: device.clone(),
                backing_offset: 0,
                kind: 0,
                cacheable: false,
            }),
        }])
        .unwrap();
    gpu.write_block_cached(reservation.offset(), &vec![0x23; 0x60000])
        .unwrap();
    let middle = space.address(reservation.offset().get() + 0x20000).unwrap();
    space
        .remap_sparse(vec![MaxwellSparseRemapRequest {
            offset: middle,
            size: 0x20000,
            mapping: None,
        }])
        .unwrap();
    let mut observer = InvalidationObserver::default();
    gpu.flush_caching(&mut observer).unwrap();
    assert_eq!(
        observer.invalidations,
        [vec![
            nixe_gpu::native::GuestBufferRange {
                address: device.address(),
                size: 0x20000
            },
            nixe_gpu::native::GuestBufferRange {
                address: device.address() + 0x40000,
                size: 0x20000
            },
        ]]
    );
    let last = space.address(reservation.offset().get() + 0x40000).unwrap();
    gpu.write_block_cached(last, &vec![0x31; 0x20000]).unwrap();
    space
        .remap_sparse(vec![MaxwellSparseRemapRequest {
            offset: last,
            size: 0x20000,
            mapping: Some(MaxwellSparseMapping {
                allocation: MaxwellAllocationId::new(1),
                backing: device.clone(),
                backing_offset: 0,
                kind: 0,
                cacheable: false,
            }),
        }])
        .unwrap();
    gpu.flush_caching(&mut observer).unwrap();
    assert_eq!(
        observer.invalidations.last().unwrap(),
        &[nixe_gpu::native::GuestBufferRange {
            address: device.address(),
            size: 0x20000
        }]
    );
}

#[test]
fn big_gpu_pages_preserve_discontiguous_physical_runs_for_reads_and_uploads() {
    let (mut space, mut gpu) = initialized();
    let big = u64::from(space.regions().unwrap()[1].page_size());
    let half = big / 2;
    let (_first, first) = backing(&space, half);
    let (_gap, gap) = backing(&space, half);
    let (_second, second) = backing(&space, half);
    write(space.machine(), &first, &vec![0x11; half as usize]);
    write(space.machine(), &gap, &vec![0xcc; half as usize]);
    write(space.machine(), &second, &vec![0x22; half as usize]);
    let mut input = first.clone();
    input.extend(second.clone());
    let mut map = request(&space, input, big);
    map.page_size = big as u32;
    map.allocation_alignment = big as u32;
    let base = space.map(map).unwrap().offset();
    let address = space.address(base.get() + half - 4).unwrap();
    let mut actual = [0; 8];
    gpu.read_into_unsafe(address, &mut actual).unwrap();
    assert_eq!(actual, [0x11, 0x11, 0x11, 0x11, 0x22, 0x22, 0x22, 0x22]);
    gpu.write_block_cached(address, &[1, 2, 3, 4, 5, 6, 7, 8])
        .unwrap();
    assert_eq!(
        &read(space.machine(), &first)[half as usize - 4..],
        &[1, 2, 3, 4]
    );
    assert_eq!(&read(space.machine(), &second)[..4], &[5, 6, 7, 8]);
    assert_eq!(read(space.machine(), &gap), vec![0xcc; half as usize]);
    assert_eq!(gpu.max_continuous_range(base, big).unwrap(), big);
}

#[test]
fn sparse_big_pages_read_zero_discard_uploads_and_replace_live_device_ranges() {
    let (mut space, mut gpu) = initialized();
    let big = u64::from(space.regions().unwrap()[1].page_size());
    let reserved = space
        .reserve(3, big as u32, ALLOC_SPACE_SPARSE, big)
        .unwrap();
    let base = reserved.offset();
    let middle = space.address(base.get() + big).unwrap();
    let (_allocation, ranges) = backing(&space, big);
    write(space.machine(), &ranges, &vec![0x55; big as usize]);
    let device = space
        .memory()
        .device_memory()
        .map(ranges.clone().into(), None)
        .unwrap();
    space
        .remap_sparse(vec![MaxwellSparseRemapRequest {
            offset: middle,
            size: big,
            mapping: Some(MaxwellSparseMapping {
                allocation: MaxwellAllocationId::new(1),
                backing: device,
                backing_offset: 0,
                kind: 0,
                cacheable: false,
            }),
        }])
        .unwrap();
    let start = space.address(middle.get() - 4).unwrap();
    let mut output = vec![0xff; big as usize + 8];
    gpu.read_into_unsafe(start, &mut output).unwrap();
    assert_eq!(&output[..4], &[0; 4]);
    assert!(output[4..4 + big as usize].iter().all(|&byte| byte == 0x55));
    assert_eq!(&output[4 + big as usize..], &[0; 4]);
    gpu.write_block_cached(start, &vec![0x77; big as usize + 8])
        .unwrap();
    assert_eq!(read(space.machine(), &ranges), vec![0x77; big as usize]);
    space
        .remap_sparse(vec![MaxwellSparseRemapRequest {
            offset: middle,
            size: big,
            mapping: None,
        }])
        .unwrap();
    gpu.read_into_unsafe(start, &mut output).unwrap();
    assert!(output.iter().all(|&byte| byte == 0));
    assert!(matches!(
        gpu.device_address(middle),
        Err(MaxwellGpuAccessError::UnmappedAddress { .. })
    ));
    space.free(base, 3, big as u32).unwrap();
}
