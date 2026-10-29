use super::*;
use crate::physical::WriteDomain;
use nixe_memory::{Machine, PAGE_SIZE};

#[test]
fn physical_overlap_relations_follow_registration_retirement_and_replacement() {
    let machine = Machine::new(4 * PAGE_SIZE).unwrap();
    let allocation = machine
        .account(4 * PAGE_SIZE)
        .unwrap()
        .allocate(4 * PAGE_SIZE)
        .unwrap();
    let physical =
        PhysicalMapping::new(&[AllocationRange::new(allocation.pin(), 0, 4 * PAGE_SIZE).unwrap()])
            .unwrap();
    let buffer = |start, bytes| {
        Footprint::new(physical.clone(), WriteDomain::Buffer { start, bytes }).unwrap()
    };
    let mut authority = Authority::new(machine.identity());
    authority
        .register(1u32, vec![buffer(0, PAGE_SIZE)])
        .unwrap();
    authority
        .register(2, vec![buffer(PAGE_SIZE / 2, PAGE_SIZE)])
        .unwrap();
    authority
        .register(3, vec![buffer(2 * PAGE_SIZE, PAGE_SIZE)])
        .unwrap();
    assert!(authority.resources_may_overlap(1, 1));
    assert!(authority.resources_may_overlap(1, 2));
    assert!(authority.resources_may_overlap(2, 1));
    assert!(!authority.resources_may_overlap(1, 3));
    assert!(!authority.resources_may_overlap(3, 2));
    authority.unregister(2).unwrap();
    assert!(!authority.resources_may_overlap(1, 2));
    authority
        .register(2, vec![buffer(3 * PAGE_SIZE, PAGE_SIZE)])
        .unwrap();
    assert!(!authority.resources_may_overlap(1, 2));
    assert!(!authority.resources_may_overlap(2, 1));
}

#[test]
fn accepted_equivalent_mappings_use_registered_spans_and_reuse_writers() {
    let machine = Machine::new(4 * PAGE_SIZE).unwrap();
    let allocation = machine
        .account(4 * PAGE_SIZE)
        .unwrap()
        .allocate(4 * PAGE_SIZE)
        .unwrap();
    for fragmented in [false, true] {
        let ranges: Vec<_> = if fragmented {
            vec![2, 0, 2, 1]
        } else {
            vec![0]
        }
        .into_iter()
        .map(|page| AllocationRange::new(allocation.pin(), page * PAGE_SIZE, PAGE_SIZE).unwrap())
        .collect();
        let registered = PhysicalMapping::new(&ranges).unwrap();
        let equivalent = PhysicalMapping::new(&ranges).unwrap();
        assert_eq!(*registered, *equivalent);
        assert!(!Arc::ptr_eq(&registered, &equivalent));
        let domain = WriteDomain::Buffer {
            start: 0,
            bytes: registered.bytes(),
        };
        let full = Footprint::new(registered.clone(), domain).unwrap();
        let other = Footprint::new(equivalent, domain).unwrap();
        let mut authority = Authority::new(machine.identity());
        authority.register(1u32, vec![full.clone()]).unwrap();
        let writer = authority.create_writer(1, other).unwrap();
        assert_eq!(authority.create_writer(1, full.clone()).unwrap(), writer);
        authority
            .publish_write(&machine, writer, || Ok(((), NativeTick(1))))
            .unwrap();
        let resolved = authority.resolve_footprint(&full).unwrap();
        assert_eq!(
            resolved.iter().map(|part| part.bytes).sum::<u64>(),
            registered.bytes()
        );
        assert!(resolved.iter().all(|part| {
            part.producer
                .is_some_and(|producer| producer.writer == writer)
        }));
    }
}

#[test]
fn compatible_short_mappings_preserve_logical_and_native_coordinates() {
    let machine = Machine::new(2 * PAGE_SIZE).unwrap();
    let allocation = machine
        .account(2 * PAGE_SIZE)
        .unwrap()
        .allocate(2 * PAGE_SIZE)
        .unwrap();
    let range = AllocationRange::new(allocation.pin(), 0, 2 * PAGE_SIZE).unwrap();
    let registered = PhysicalMapping::new(std::slice::from_ref(&range)).unwrap();
    let shorter = PhysicalMapping::new(&[range.slice(0, PAGE_SIZE).unwrap()]).unwrap();
    let full = Footprint::new(
        registered,
        WriteDomain::Buffer {
            start: 0,
            bytes: 2 * PAGE_SIZE,
        },
    )
    .unwrap();
    let partial = Footprint::new(
        shorter,
        WriteDomain::Buffer {
            start: 17,
            bytes: 31,
        },
    )
    .unwrap();
    let mut authority = Authority::new(machine.identity());
    authority.register(1u32, vec![full]).unwrap();
    let writer = authority.create_writer(1, partial).unwrap();
    authority
        .publish_write(&machine, writer, || Ok(((), NativeTick(1))))
        .unwrap();
    let resolved = authority.resolve(&[range.slice(17, 31).unwrap()]).unwrap();
    assert_eq!(resolved.len(), 1);
    assert_eq!(resolved[0].bytes, 31);
    let producer = resolved[0].producer.as_ref().unwrap();
    assert_eq!(producer.writer, writer);
    assert_eq!(producer.native, 17);
    assert!(
        authority
            .resolve(&[range.slice(PAGE_SIZE, 31).unwrap()])
            .unwrap()
            .iter()
            .all(|part| part.producer.is_none())
    );
    let displaced = PhysicalMapping::new(&[range.slice(PAGE_SIZE, PAGE_SIZE).unwrap()]).unwrap();
    let invalid = Footprint::new(
        displaced,
        WriteDomain::Buffer {
            start: 0,
            bytes: 31,
        },
    )
    .unwrap();
    assert!(authority.create_writer(1, invalid).is_err());
}

#[test]
fn image_writer_canonicalization_preserves_partial_geometry_and_native_layout() {
    use crate::physical::{ByteRect, ImageFootprint, ImageStorage};
    let machine = Machine::new(2 * PAGE_SIZE).unwrap();
    let allocation = machine
        .account(2 * PAGE_SIZE)
        .unwrap()
        .allocate(2 * PAGE_SIZE)
        .unwrap();
    let range = AllocationRange::new(allocation.pin(), 0, 2 * PAGE_SIZE).unwrap();
    let small = PhysicalMapping::new(&[range.slice(0, PAGE_SIZE).unwrap()]).unwrap();
    let large = PhysicalMapping::new(std::slice::from_ref(&range)).unwrap();
    let geometry = ImageFootprint {
        depth_slice: 0,
        base: 0,
        columns: 1,
        rows: 1,
        packed_bytes: 4,
        aspect_offset: 0,
        aspect_bytes: 4,
        storage: ImageStorage::Pitch {
            row_pitch: PAGE_SIZE,
        },
        native_base: 0,
        native_element_bytes: 4,
        native_row_pitch: 4,
        rect: ByteRect {
            x: 0,
            y: 0,
            width: 1,
            height: 1,
        },
    };
    let full = Footprint::new(small, WriteDomain::Image(geometry)).unwrap();
    let partial = Footprint::new(
        large,
        WriteDomain::Image(ImageFootprint {
            rows: 2,
            aspect_offset: 3,
            aspect_bytes: 1,
            native_base: 17,
            native_element_bytes: 1,
            native_row_pitch: 1,
            ..geometry
        }),
    )
    .unwrap();
    let mut authority = Authority::new(machine.identity());
    authority.register(1u32, vec![full]).unwrap();
    let writer = authority.create_writer(1, partial).unwrap();
    authority
        .publish_write(&machine, writer, || Ok(((), NativeTick(1))))
        .unwrap();
    let resolved = authority.resolve(&[range.slice(3, 1).unwrap()]).unwrap();
    assert_eq!(resolved.len(), 1);
    assert_eq!(resolved[0].bytes, 1);
    let producer = resolved[0].producer.as_ref().unwrap();
    assert_eq!(producer.writer, writer);
    assert_eq!(producer.native, 17);
    let untouched = authority
        .resolve(&[
            range.slice(0, 3).unwrap(),
            range.slice(PAGE_SIZE, 4).unwrap(),
        ])
        .unwrap();
    assert!(untouched.iter().all(|part| part.producer.is_none()));
}

#[test]
fn observer_pruning_charges_scans_to_insertions_and_reclaims_abandoned_entries() {
    let mut entries = Vec::new();
    let mut next = 16;
    let mut inspections = 0;
    for _ in 0..4096 {
        prune_on_insert(&mut entries, &mut next, |_: &bool| {
            inspections += 1;
            true
        });
        entries.push(true);
    }
    assert!(inspections < 2 * entries.len());
    entries.fill(false);
    for _ in 0..4096 {
        prune_on_insert(&mut entries, &mut next, |&live| live);
        entries.push(false);
    }
    assert!(entries.len() <= 16);
}

#[test]
fn authority_observers_remain_live_until_publication_and_are_reclaimed_afterwards() {
    let machine = Machine::new(PAGE_SIZE).unwrap();
    let allocation = machine
        .account(PAGE_SIZE)
        .unwrap()
        .allocate(PAGE_SIZE)
        .unwrap();
    let ranges = [AllocationRange::new(allocation.pin(), 0, PAGE_SIZE).unwrap()];
    let footprint = Footprint::new(
        PhysicalMapping::new(&ranges).unwrap(),
        WriteDomain::Buffer {
            start: 0,
            bytes: PAGE_SIZE,
        },
    )
    .unwrap();
    let mut authority = Authority::new(machine.identity());
    authority.register(1u32, vec![footprint.clone()]).unwrap();
    authority.register(2, vec![footprint.clone()]).unwrap();
    let writer = authority.create_writer(1, footprint.clone()).unwrap();
    let mut memory = Vec::new();
    let mut native = Vec::new();
    for _ in 0..512 {
        let observation = nixe_memory::Observation::default();
        machine
            .access_captured(
                &ranges,
                nixe_memory::AccessOperation::ReadObserved(observation.clone()),
            )
            .unwrap()
            .into_ready()
            .unwrap();
        authority
            .watch_memory(&ranges, observation.clone())
            .unwrap();
        memory.push(observation);
        native.push(authority.observe_native(2, footprint.clone()).unwrap());
    }
    assert!(memory.iter().all(|observation| !observation.is_invalid()));
    assert!(native.iter().all(|observation| !observation.is_invalid()));
    assert_eq!(authority.writers[&writer].memory.len(), 512);
    assert_eq!(authority.writers[&writer].consumers.len(), 512);
    assert_eq!(authority.resources[&2].consumers.len(), 512);
    authority
        .publish_write(&machine, writer, || Ok(((), NativeTick(1))))
        .unwrap();
    assert!(memory.iter().all(nixe_memory::Observation::is_invalid));
    assert!(native.iter().all(ResourceObservation::is_invalid));
    assert!(authority.writers[&writer].consumers.is_empty());
    for _ in 0..1024 {
        let observation = nixe_memory::Observation::default();
        authority.watch_memory(&ranges, observation).unwrap();
        drop(authority.observe_native(2, footprint.clone()).unwrap());
    }
    assert!(authority.writers[&writer].memory.len() <= 16);
    assert!(authority.writers[&writer].consumers.len() <= 16);
    assert!(authority.resources[&2].consumers.len() <= 16);
}

#[test]
fn modified_intervals_match_a_byte_oracle_across_allocations_and_insert_orders() {
    let mut intervals = PhysicalIntervals::new();
    let mut oracle = [[false; 257]; 3];
    let mut seed = 0x8374aabcu64;
    for _ in 0..1024 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let owner = (seed >> 32) as usize % oracle.len();
        let start = (seed >> 16) as usize % 256;
        let end = (start + 1 + (seed as usize % 13)).min(257);
        oracle[owner][start..end].fill(true);
        add_interval(
            &mut intervals,
            PhysicalInterval {
                allocation: AllocationId::new(owner as u64 + 1),
                start: start as u64,
                end: end as u64,
            },
        );
        let mut actual = [[false; 257]; 3];
        let mut previous: Option<PhysicalInterval> = None;
        for (&(allocation, start), &end) in &intervals {
            if let Some(old) = previous
                && old.allocation == allocation
            {
                assert!(
                    old.end < start,
                    "overlapping or adjacent runs must have merged"
                );
            }
            actual[allocation.get() as usize - 1][start as usize..end as usize].fill(true);
            previous = Some(PhysicalInterval {
                allocation,
                start,
                end,
            });
        }
        assert_eq!(actual, oracle);
        let queried: Vec<_> = overlapping_intervals(
            &intervals,
            AllocationId::new(owner as u64 + 1),
            start as u64,
            end as u64,
        )
        .collect();
        assert!(
            queried
                .iter()
                .all(|part| part.allocation.get() == owner as u64 + 1
                    && part.start < end as u64
                    && part.end > start as u64)
        );
        for (offset, &covered) in oracle[owner][start..end].iter().enumerate() {
            let byte = (start + offset) as u64;
            assert_eq!(
                queried
                    .iter()
                    .any(|part| part.start <= byte && byte < part.end),
                covered
            );
        }
    }
}

#[test]
fn fragmented_packed_aspects_keep_padding_and_merge_a_later_covering_write() {
    let mut intervals = PhysicalIntervals::new();
    let allocation = AllocationId::new(1);
    const PIXELS: u64 = 16_384;
    for pass in 0..2 {
        for index in 0..PIXELS {
            let pixel = if pass == 0 {
                (index * 4051) & (PIXELS - 1)
            } else {
                PIXELS - index - 1
            };
            add_interval(
                &mut intervals,
                PhysicalInterval {
                    allocation,
                    start: pixel * 4,
                    end: pixel * 4 + 3,
                },
            );
        }
        assert_eq!(intervals.len(), PIXELS as usize);
    }
    for (&(owner, start), &end) in &intervals {
        assert_eq!(owner, allocation);
        assert_eq!(start % 4, 0);
        assert_eq!(
            end - start,
            3,
            "the untouched packed stencil byte must remain a hole"
        );
    }
    add_interval(
        &mut intervals,
        PhysicalInterval {
            allocation,
            start: 0,
            end: PIXELS * 4,
        },
    );
    assert_eq!(intervals.len(), 1);
    assert_eq!(intervals[&(allocation, 0)], PIXELS * 4);
    assert!(
        overlapping_intervals(&intervals, AllocationId::new(2), 0, PIXELS * 4)
            .next()
            .is_none()
    );
    assert!(
        overlapping_intervals(&intervals, allocation, PIXELS * 4, PIXELS * 4 + 1)
            .next()
            .is_none()
    );
}

#[test]
fn tiled_demand_sweep_preserves_rectangles_aspects_cpu_holes_and_native_offsets() {
    use crate::physical::{ByteRect, ImageFootprint, ImageStorage};
    let machine = Machine::new(2 * PAGE_SIZE).unwrap();
    let allocation = machine
        .account(2 * PAGE_SIZE)
        .unwrap()
        .allocate(2 * PAGE_SIZE)
        .unwrap();
    let range = AllocationRange::new(allocation.pin(), 0, 2 * PAGE_SIZE).unwrap();
    let mapping = PhysicalMapping::new(std::slice::from_ref(&range)).unwrap();
    let full = ByteRect {
        x: 0,
        y: 0,
        width: 17,
        height: 13,
    };
    let patch = ByteRect {
        x: 3,
        y: 2,
        width: 5,
        height: 6,
    };
    let footprint = |mapping, aspect_offset, aspect_bytes, native_element_bytes, rect| {
        Footprint::new(
            mapping,
            WriteDomain::Image(ImageFootprint {
                depth_slice: 0,
                base: 0,
                columns: 17,
                rows: 13,
                packed_bytes: 4,
                aspect_offset,
                aspect_bytes,
                storage: ImageStorage::BlockLinear {
                    block_height_log2: 2,
                    block_depth_log2: 0,
                },
                native_base: 0,
                native_element_bytes,
                native_row_pitch: 17 * u64::from(native_element_bytes),
                rect,
            }),
        )
        .unwrap()
    };
    let depth = footprint(Arc::clone(&mapping), 0, 3, 4, full);
    // A distinct interpretation object can describe the same physical bytes.
    let stencil = footprint(
        PhysicalMapping::new(std::slice::from_ref(&range)).unwrap(),
        3,
        1,
        1,
        full,
    );
    let changed = footprint(Arc::clone(&mapping), 0, 3, 4, patch);
    let mut authority = Authority::new(machine.identity());
    for (resource, domain) in [(1, &depth), (2, &stencil), (3, &depth)] {
        authority
            .register(resource, vec![Arc::clone(domain)])
            .unwrap();
    }
    for (resource, domain) in [(1, &depth), (2, &stencil), (3, &changed)] {
        let writer = authority
            .create_writer(resource, Arc::clone(domain))
            .unwrap();
        authority
            .record_write(writer, NativeTick(resource as u64))
            .unwrap();
        if resource < 3 {
            assert_eq!(authority.covering_resource(&depth).unwrap(), Some(1));
        }
        if resource == 2 {
            assert_eq!(authority.covering_resource(&stencil).unwrap(), Some(2));
        }
    }
    assert_eq!(authority.covering_resource(&depth).unwrap(), None);
    assert_eq!(authority.covering_resource(&stencil).unwrap(), Some(2));
    // A completed download crosses tile runs and both packed aspects. Later
    // CPU holes must still win over materialized bytes within those runs.
    for part in authority
        .resolve(&[AllocationRange::new(allocation.pin(), 7, 76).unwrap()])
        .unwrap()
    {
        authority
            .publish_readback(
                &machine,
                part,
                &vec![0x6d; part.bytes as usize],
                &mut Vec::new(),
            )
            .unwrap();
    }
    authority
        .cpu_write(&[AllocationRange::new(allocation.pin(), 19, 12).unwrap()])
        .unwrap();
    assert_eq!(authority.covering_resource(&stencil).unwrap(), None);
    for resource in [1, 2, 3] {
        authority.normalize(resource).unwrap();
    }
    for (demand, lane_begin, lane_count, element) in [(&depth, 0, 3, 4), (&stencil, 3, 1, 1)] {
        let mut actual = vec![None; 17 * 13 * element];
        let mut present = vec![false; actual.len()];
        for part in authority.resolve_footprint(demand).unwrap() {
            for offset in 0..part.bytes {
                let index = (part.logical + offset) as usize;
                assert!(!present[index], "demand byte was resolved twice");
                present[index] = true;
                actual[index] = part
                    .producer
                    .map(|p| (p.resource, p.native + offset, p.materialized));
            }
        }
        for y in 0..13usize {
            for x in 0..17usize {
                for lane in 0..element {
                    let index = (y * 17 + x) * element + lane;
                    if lane >= lane_count {
                        assert!(!present[index]);
                        continue;
                    }
                    let byte_x = x * 4 + lane_begin + lane;
                    // Independent Tegra GOB layout, including padded columns.
                    let physical = byte_x / 64 * 2048
                        + y / 8 * 512
                        + byte_x % 64 / 32 * 256
                        + y % 8 / 2 * 64
                        + byte_x % 32 / 16 * 32
                        + y % 2 * 16
                        + byte_x % 16;
                    let expected = if (19..31).contains(&physical) {
                        None
                    } else if lane_begin == 3 {
                        Some((2, index as u64, (7..83).contains(&physical)))
                    } else {
                        Some((
                            if (3..8).contains(&x) && (2..8).contains(&y) {
                                3
                            } else {
                                1
                            },
                            index as u64,
                            (7..83).contains(&physical),
                        ))
                    };
                    assert!(present[index]);
                    assert_eq!(actual[index], expected, "x={x} y={y} lane={lane}");
                }
            }
        }
    }
}

#[test]
fn releasing_domain_users_preserves_live_producers_and_retires_shadowed_records() {
    let machine = Machine::new(PAGE_SIZE).unwrap();
    let allocation = machine
        .account(PAGE_SIZE)
        .unwrap()
        .allocate(PAGE_SIZE)
        .unwrap();
    let range = AllocationRange::new(allocation.pin(), 0, 8).unwrap();
    let mapping = PhysicalMapping::new(std::slice::from_ref(&range)).unwrap();
    let footprint = Footprint::new(
        Arc::clone(&mapping),
        WriteDomain::Buffer { start: 0, bytes: 8 },
    )
    .unwrap();
    let mut authority = Authority::new(machine.identity());
    authority
        .register(1u32, vec![Arc::clone(&footprint)])
        .unwrap();
    authority.register(2, vec![Arc::clone(&footprint)]).unwrap();
    let first = authority.create_writer(1, Arc::clone(&footprint)).unwrap();
    let second_user = authority
        .create_writer(
            1,
            Footprint::new(
                Arc::clone(&mapping),
                WriteDomain::Buffer { start: 0, bytes: 8 },
            )
            .unwrap(),
        )
        .unwrap();
    assert_eq!(first, second_user);
    authority.release_writer(first).unwrap();
    authority.record_write(second_user, NativeTick(1)).unwrap();
    authority.release_writer(second_user).unwrap();
    assert!(authority.record_write(first, NativeTick(2)).is_err());
    assert_eq!(
        authority.resolve(std::slice::from_ref(&range)).unwrap()[0]
            .producer
            .unwrap()
            .resource,
        1
    );
    assert_eq!(authority.retire_unused_writers(1).unwrap(), 0);
    let second = authority.create_writer(2, footprint).unwrap();
    authority.record_write(second, NativeTick(3)).unwrap();
    assert_eq!(authority.retire_unused_writers(1).unwrap(), 1);
    assert!(authority.record_write(first, NativeTick(4)).is_err());
    authority.unregister(1).unwrap();
    assert_eq!(authority.writer_count(), 1);
    assert_eq!(
        authority.resolve(&[range]).unwrap()[0]
            .producer
            .unwrap()
            .resource,
        2
    );
}

#[test]
fn normalization_preserves_partial_versions_cpu_holes_and_reusable_domains() {
    let machine = Machine::new(PAGE_SIZE).unwrap();
    let allocation = machine
        .account(PAGE_SIZE)
        .unwrap()
        .allocate(PAGE_SIZE)
        .unwrap();
    let mapping =
        PhysicalMapping::new(&[AllocationRange::new(allocation.pin(), 0, 128).unwrap()]).unwrap();
    let domain = |start, bytes| {
        Footprint::new(Arc::clone(&mapping), WriteDomain::Buffer { start, bytes }).unwrap()
    };
    let full = domain(0, 128);
    let mut authority = Authority::new(machine.identity());
    authority.register(1u32, vec![Arc::clone(&full)]).unwrap();
    authority.register(2, vec![Arc::clone(&full)]).unwrap();
    let foreign = authority.create_writer(2, domain(12, 12)).unwrap();
    let mut expected = vec![None; 128];
    let mut writers = Vec::new();
    for start in 0..65usize {
        let writer = authority
            .create_writer(1, domain(start as u64, 64))
            .unwrap();
        writers.push(writer);
        let tick = NativeTick(start as u64 + 1);
        let sequence = authority.record_write(writer, tick).unwrap();
        for (byte, value) in expected.iter_mut().enumerate().skip(start).take(64) {
            *value = Some(Producer {
                writer,
                resource: 1,
                sequence,
                tick,
                native: byte as u64,
                materialized: false,
            });
        }
        if start == 40 {
            let tick = NativeTick(199);
            let sequence = authority.record_write(foreign, tick).unwrap();
            for (byte, value) in expected.iter_mut().enumerate().take(24).skip(12) {
                *value = Some(Producer {
                    writer: foreign,
                    resource: 2,
                    sequence,
                    tick,
                    native: byte as u64,
                    materialized: false,
                });
            }
            authority
                .cpu_write(&[AllocationRange::new(allocation.pin(), 5, 1).unwrap()])
                .unwrap();
            expected[5] = None;
        }
    }
    let demand = [AllocationRange::new(allocation.pin(), 0, 128).unwrap()];
    let flatten = |authority: &Authority<u32>| {
        let mut actual = vec![None; 128];
        for span in authority.resolve(&demand).unwrap() {
            for byte in 0..span.bytes {
                actual[(span.logical + byte) as usize] = span.producer.map(|p| Producer {
                    native: p.native + byte,
                    ..p
                });
            }
        }
        actual
    };
    assert_eq!(flatten(&authority), expected);
    let writer = authority.create_writer(1, full).unwrap();
    let sequence = authority.record_write(writer, NativeTick(201)).unwrap();
    for (byte, value) in expected.iter_mut().enumerate() {
        *value = Some(Producer {
            writer,
            resource: 1,
            sequence,
            tick: NativeTick(201),
            native: byte as u64,
            materialized: false,
        });
    }
    assert_eq!(flatten(&authority), expected);
    authority.normalize(1).unwrap();
    authority.unregister(2).unwrap();
    assert_eq!(authority.live_record_count(), 1);
    let sequence = authority.record_write(writers[0], NativeTick(202)).unwrap();
    for (byte, value) in expected[..64].iter_mut().enumerate() {
        *value = Some(Producer {
            writer: writers[0],
            resource: 1,
            sequence,
            tick: NativeTick(202),
            native: byte as u64,
            materialized: false,
        });
    }
    assert_eq!(flatten(&authority), expected);
    assert_eq!(authority.live_record_count(), 2);
    drop(demand);
    drop(allocation);
    assert_eq!(machine.statistics().allocated_bytes, 0);
}

#[test]
fn packed_image_aspects_do_not_claim_padding_or_each_others_bytes() {
    use crate::physical::{ByteRect, ImageFootprint, ImageStorage};
    let machine = Machine::new(PAGE_SIZE).unwrap();
    let allocation = machine
        .account(PAGE_SIZE)
        .unwrap()
        .allocate(PAGE_SIZE)
        .unwrap();
    let mapping =
        PhysicalMapping::new(&[AllocationRange::new(allocation.pin(), 0, PAGE_SIZE).unwrap()])
            .unwrap();
    let make = |aspect_offset, aspect_bytes, native_element_bytes| {
        Footprint::new(
            Arc::clone(&mapping),
            WriteDomain::Image(ImageFootprint {
                depth_slice: 0,
                base: 0,
                columns: 3,
                rows: 2,
                packed_bytes: 4,
                aspect_offset,
                aspect_bytes,
                storage: ImageStorage::Pitch { row_pitch: 16 },
                native_base: 0,
                native_element_bytes,
                native_row_pitch: 3 * u64::from(native_element_bytes),
                rect: ByteRect {
                    x: 0,
                    y: 0,
                    width: 3,
                    height: 2,
                },
            }),
        )
        .unwrap()
    };
    let depth = make(0, 3, 4);
    let stencil = make(3, 1, 1);
    let mut authority = Authority::new(machine.identity());
    authority.register(1, vec![Arc::clone(&depth)]).unwrap();
    authority.register(2, vec![Arc::clone(&stencil)]).unwrap();
    let watched = authority.observe(1, Arc::clone(&depth)).unwrap();
    let dw = authority.create_writer(1, Arc::clone(&depth)).unwrap();
    let sw = authority.create_writer(2, Arc::clone(&stencil)).unwrap();
    assert!(!authority.has_writers(&depth));
    assert!(!authority.has_writers(&stencil));
    authority.record_write(sw, NativeTick(9)).unwrap();
    assert!(!authority.has_writers(&depth));
    assert!(authority.has_writers(&stencil));
    assert!(!watched.is_invalid());
    authority.record_write(dw, NativeTick(2)).unwrap();
    assert!(authority.has_writers(&depth));
    assert!(watched.is_invalid());
    let demand = [AllocationRange::new(allocation.pin(), 0, 32).unwrap()];
    let mut actual = [None; 32];
    for span in authority.resolve(&demand).unwrap() {
        for byte in 0..span.bytes {
            actual[(span.logical + byte) as usize] =
                span.producer.map(|p| (p.resource, p.native + byte));
        }
    }
    for y in 0..2 {
        for x in 0..3 {
            for byte in 0..4 {
                let producer = if byte == 3 {
                    (2, y * 3 + x)
                } else {
                    (1, y * 12 + x * 4 + byte)
                };
                assert_eq!(actual[(y * 16 + x * 4 + byte) as usize], Some(producer));
            }
        }
        assert_eq!(
            &actual[(y * 16 + 12) as usize..(y * 16 + 16) as usize],
            &[None; 4]
        );
    }
    for (footprint, native_bytes, meaningful) in [(&depth, 4, 3), (&stencil, 1, 1)] {
        let resolved = authority.resolve_footprint(footprint).unwrap();
        assert_eq!(
            resolved.iter().map(|part| part.bytes).sum::<u64>(),
            6 * meaningful
        );
        let mut native = [None; 24];
        for part in resolved {
            for byte in 0..part.bytes {
                let producer = part.producer.unwrap();
                native[(part.logical + byte) as usize] =
                    Some((producer.resource, producer.native + byte));
            }
        }
        for element in 0..6 {
            for byte in 0..native_bytes {
                let expected = (byte < meaningful).then_some((
                    if native_bytes == 4 { 1 } else { 2 },
                    element * native_bytes + byte,
                ));
                assert_eq!(native[(element * native_bytes + byte) as usize], expected);
            }
        }
    }
    let watched = authority.observe(1, make(0, 3, 4)).unwrap();
    authority
        .cpu_write(&[AllocationRange::new(allocation.pin(), 7, 1).unwrap()])
        .unwrap();
    assert!(!watched.is_invalid());
    let actual = authority
        .resolve(&[AllocationRange::new(allocation.pin(), 4, 4).unwrap()])
        .unwrap();
    assert_eq!(
        actual
            .iter()
            .map(|s| (s.bytes, s.producer.map(|p| p.resource)))
            .collect::<Vec<_>>(),
        [(3, Some(1)), (1, None)]
    );
    let selected_stencil = authority.resolve_footprint(&stencil).unwrap();
    assert_eq!(
        selected_stencil.iter().map(|part| part.bytes).sum::<u64>(),
        6
    );
    assert_eq!(
        selected_stencil
            .iter()
            .filter(|part| part.producer.is_none())
            .map(|part| part.logical)
            .collect::<Vec<_>>(),
        [1]
    );
    assert!(
        authority
            .resolve_footprint(&depth)
            .unwrap()
            .iter()
            .all(|part| part.producer.is_some_and(|producer| producer.resource == 1))
    );
}

#[test]
fn producer_sequence_cpu_subtraction_and_readback_match_an_independent_byte_model() {
    let machine = Machine::new(PAGE_SIZE).unwrap();
    let allocation = machine
        .account(PAGE_SIZE)
        .unwrap()
        .allocate(PAGE_SIZE)
        .unwrap();
    let ranges = [AllocationRange::new(allocation.pin(), 0, 128).unwrap()];
    let mapping = PhysicalMapping::new(&ranges).unwrap();
    let domain = |start, bytes| {
        Footprint::new(Arc::clone(&mapping), WriteDomain::Buffer { start, bytes }).unwrap()
    };
    let full = domain(0, 64);
    let part = domain(8, 16);
    let other = domain(96, 8);
    let mut authority = Authority::new(machine.identity());
    authority.register(1u32, vec![Arc::clone(&full)]).unwrap();
    authority.register(2, vec![Arc::clone(&full)]).unwrap();
    authority.register(3, vec![Arc::clone(&other)]).unwrap();
    let first = authority.create_writer(1, Arc::clone(&full)).unwrap();
    let second = authority.create_writer(2, Arc::clone(&part)).unwrap();
    let unrelated = authority.observe(3, other).unwrap();
    let observer = authority.observe(2, Arc::clone(&part)).unwrap();
    let sequence = authority.record_write(first, NativeTick(100)).unwrap();
    assert!(observer.is_invalid());
    assert!(!unrelated.is_invalid());
    let demand = [AllocationRange::new(allocation.pin(), 0, 64).unwrap()];
    let old = authority.resolve(&demand).unwrap().remove(0);
    assert_eq!(old.producer.unwrap().sequence, sequence);
    authority.record_write(second, NativeTick(7)).unwrap();
    let current = authority.current_readback(old).unwrap();
    assert_eq!(
        current
            .iter()
            .map(|s| (s.logical, s.bytes))
            .collect::<Vec<_>>(),
        [(0, 8), (24, 40)]
    );
    let snapshot = authority
        .resolve(&[AllocationRange::new(allocation.pin(), 8, 16).unwrap()])
        .unwrap()
        .remove(0);
    let overwritten = [AllocationRange::new(allocation.pin(), 13, 3).unwrap()];
    authority.cpu_write(&overwritten).unwrap();
    assert_eq!(
        authority
            .current_readback(snapshot)
            .unwrap()
            .iter()
            .map(|s| (s.logical, s.bytes))
            .collect::<Vec<_>>(),
        [(0, 5), (8, 8)]
    );
    authority.record_write(first, NativeTick(4)).unwrap();
    assert!(authority.current_readback(old).unwrap().is_empty());
    assert!(authority.current_readback(snapshot).unwrap().is_empty());
    assert!(
        authority
            .resolve(&demand)
            .unwrap()
            .iter()
            .all(|s| s.producer.unwrap().writer == first)
    );

    let mut expected = vec![None; 64];
    let mut random = 0x73ab_c9d5u64;
    for step in 0..400 {
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        match random % 3 {
            0 | 1 => {
                let (id, resource, range) = if random & 1 == 0 {
                    (first, 1, 0..64)
                } else {
                    (second, 2, 8..24)
                };
                let tick = NativeTick(1000 - step);
                let sequence = authority.record_write(id, tick).unwrap();
                for byte in range {
                    expected[byte] = Some(Producer {
                        writer: id,
                        resource,
                        sequence,
                        tick,
                        native: byte as u64,
                        materialized: false,
                    });
                }
            }
            _ => {
                let start = (random >> 8) as usize % 64;
                let end = (start + (random >> 16) as usize % 11 + 1).min(64);
                authority
                    .cpu_write(&[AllocationRange::new(
                        allocation.pin(),
                        start as u64,
                        (end - start) as u64,
                    )
                    .unwrap()])
                    .unwrap();
                expected[start..end].fill(None);
            }
        }
        if step == 0 {
            let sequence = authority.record_write(first, NativeTick(0)).unwrap();
            for (byte, producer) in expected.iter_mut().enumerate() {
                *producer = Some(Producer {
                    writer: first,
                    resource: 1,
                    sequence,
                    tick: NativeTick(0),
                    native: byte as u64,
                    materialized: false,
                });
            }
        }
        if step % 7 == 0 {
            authority.normalize(1).unwrap();
            authority.normalize(2).unwrap();
        }
        let mut actual = vec![None; 64];
        for span in authority.resolve(&demand).unwrap() {
            for offset in 0..span.bytes {
                actual[(span.logical + offset) as usize] = span.producer.map(|p| Producer {
                    native: p.native + offset,
                    ..p
                });
            }
        }
        assert_eq!(actual, expected, "step {step}");
        let mapping = PhysicalMapping::new(&demand).unwrap();
        for start in 0..64 {
            let bytes = (64 - start).min(9);
            assert_eq!(
                authority
                    .has_buffer_producer(&mapping, start, bytes)
                    .unwrap(),
                expected[start as usize..(start + bytes) as usize]
                    .iter()
                    .any(Option::is_some),
                "producer query at step {step}, byte {start}"
            );
        }

        assert!(!unrelated.is_invalid());
    }
    authority.record_write(first, NativeTick(1001)).unwrap();
    assert!(authority.unregister(1).is_err());
    authority.unregister(2).unwrap();
    assert!(authority.record_write(second, NativeTick(1002)).is_err());
    assert_eq!(authority.writer_count(), 1);
}
#[test]
fn late_and_repeated_producers_invalidate_only_actual_overlapping_consumers() {
    let machine = Machine::new(PAGE_SIZE).unwrap();
    let allocation = machine
        .account(PAGE_SIZE)
        .unwrap()
        .allocate(PAGE_SIZE)
        .unwrap();
    let mapping =
        PhysicalMapping::new(&[AllocationRange::new(allocation.pin(), 0, PAGE_SIZE).unwrap()])
            .unwrap();
    let domain = |start, bytes| {
        Footprint::new(Arc::clone(&mapping), WriteDomain::Buffer { start, bytes }).unwrap()
    };
    let full = domain(0, PAGE_SIZE);
    let mut authority = Authority::new(machine.identity());
    authority.register(1, vec![Arc::clone(&full)]).unwrap();
    authority.register(2, vec![Arc::clone(&full)]).unwrap();
    let a = authority.observe(1, domain(8, 8)).unwrap();
    let b = authority.observe(1, domain(24, 8)).unwrap();
    let writer_domain = domain(10, 4);
    let writer = authority
        .create_writer(2, Arc::clone(&writer_domain))
        .unwrap();
    assert!(!a.is_invalid());
    authority.record_write(writer, NativeTick(1)).unwrap();
    assert!(a.is_invalid());
    assert!(!b.is_invalid());
    let c = authority.observe(1, domain(8, 8)).unwrap();
    authority.record_write(writer, NativeTick(2)).unwrap();
    assert!(c.is_invalid());
    assert!(!b.is_invalid());
    let refs = Arc::strong_count(&writer_domain);
    for tick in 3..1000 {
        authority.record_write(writer, NativeTick(tick)).unwrap();
    }
    assert_eq!(authority.writer_count(), 1);
    assert_eq!(Arc::strong_count(&writer_domain), refs);
    assert!(!b.is_invalid());
    authority
        .cpu_write(&[AllocationRange::new(allocation.pin(), 25, 1).unwrap()])
        .unwrap();
    assert!(b.is_invalid());
}

#[test]
fn native_observation_ignores_its_own_updates_but_tracks_aliases_and_cpu_overwrites() {
    let machine = nixe_memory::Machine::new(nixe_memory::PAGE_SIZE).unwrap();
    let allocation = machine
        .account(nixe_memory::PAGE_SIZE)
        .unwrap()
        .allocate(nixe_memory::PAGE_SIZE)
        .unwrap();
    let range = nixe_memory::AllocationRange::new(allocation.pin(), 0, 16).unwrap();
    let mapping = PhysicalMapping::new(std::slice::from_ref(&range)).unwrap();
    let full = Footprint::new(
        mapping,
        WriteDomain::Buffer {
            start: 0,
            bytes: 16,
        },
    )
    .unwrap();
    let mut authority = Authority::new(machine.identity());
    authority.register(1u32, vec![Arc::clone(&full)]).unwrap();
    authority.register(2u32, vec![Arc::clone(&full)]).unwrap();
    let native = authority.observe_native(1, Arc::clone(&full)).unwrap();
    let snapshot = authority.observe(1, Arc::clone(&full)).unwrap();
    let writer = authority.create_writer(1, Arc::clone(&full)).unwrap();
    for tick in 1..101 {
        authority.record_write(writer, NativeTick(tick)).unwrap();
    }
    assert!(!native.is_invalid());
    assert!(snapshot.is_invalid());
    let alias = authority.create_writer(2, Arc::clone(&full)).unwrap();
    authority.record_write(alias, NativeTick(101)).unwrap();
    assert!(native.is_invalid());
    let native = authority.observe_native(1, full).unwrap();
    authority.cpu_write(&[range]).unwrap();
    assert!(native.is_invalid());
}

#[test]
fn native_views_with_repeated_physical_bytes_observe_their_own_alias_writes() {
    let machine = Machine::new(PAGE_SIZE).unwrap();
    let allocation = machine
        .account(PAGE_SIZE)
        .unwrap()
        .allocate(PAGE_SIZE)
        .unwrap();
    let part = AllocationRange::new(allocation.pin(), 0, 16).unwrap();
    let mapping = PhysicalMapping::new(&[part.clone(), part]).unwrap();
    let full = Footprint::new(
        mapping.clone(),
        WriteDomain::Buffer {
            start: 0,
            bytes: 32,
        },
    )
    .unwrap();
    let mut authority = Authority::new(machine.identity());
    authority.register(1, vec![full.clone()]).unwrap();
    let observed = authority.observe_native(1, full.clone()).unwrap();
    let writer = authority
        .create_writer(
            1,
            Footprint::new(
                mapping,
                WriteDomain::Buffer {
                    start: 16,
                    bytes: 16,
                },
            )
            .unwrap(),
        )
        .unwrap();
    authority.record_write(writer, NativeTick(1)).unwrap();
    assert!(
        observed.is_invalid(),
        "a later logical occurrence invalidates the earlier native alias"
    );
    let resolved = authority.resolve_footprint(&full).unwrap();
    assert_eq!(
        resolved
            .iter()
            .map(|part| (part.logical, part.producer.unwrap().native, part.bytes))
            .collect::<Vec<_>>(),
        [(0, 16, 16), (16, 16, 16)]
    );
}
