use nixe_gpu::FrontendSubmissionId;
use nixe_gpu_maxwell::*;
use nixe_memory::{AllocationRange, Machine, MemoryPermissions, PAGE_SIZE};

#[path = "support/memory.rs"]
mod memory;

const ENTRY1_ADDRESS_HIGH_MASK: u32 = 0xff;
const ENTRY1_ALLOW_FLUSH: u32 = 1 << 8;
const ENTRY1_LEVEL_SUBROUTINE: u32 = 1 << 9;
const ENTRY1_WORD_COUNT_SHIFT: u32 = 10;
const ENTRY1_SYNC_WAIT: u32 = 1 << 31;
const SUBMIT_FENCE_WAIT: u32 = 1 << 0;
const SUBMIT_FENCE_GET: u32 = 1 << 1;
const SUBMIT_HARDWARE_FORMAT: u32 = 1 << 2;
const SUBMIT_SYNC_FENCE: u32 = 1 << 3;
const SUBMIT_SUPPRESS_WFI: u32 = 1 << 4;
const SUBMIT_FENCE_INCREMENT_VALUE: u32 = 1 << 8;

fn descriptor(address: u64, words: u32, modes: u32) -> [u8; 8] {
    let word0 = address as u32;
    let word1 = ((address >> 32) as u32 & ENTRY1_ADDRESS_HIGH_MASK)
        | (words << ENTRY1_WORD_COUNT_SHIFT)
        | modes;
    let mut bytes = [0; 8];
    bytes[..4].copy_from_slice(&word0.to_le_bytes());
    bytes[4..].copy_from_slice(&word1.to_le_bytes());
    bytes
}

fn request(entries: u32, flags: u32) -> MaxwellGpfifoSubmitRequest {
    MaxwellGpfifoSubmitRequest {
        entry_count: entries,
        flags,
        fence_id: 7,
        fence_value: 11,
    }
}

fn bound_channel(id: u64, address_space: MaxwellAddressSpaceId) -> MaxwellGpuChannel {
    let mut channel = MaxwellGpuChannel::new(
        MaxwellChannelId::new(id),
        MaxwellChannelOwner::new(1),
        SWITCH_1_GM20B_PROFILE,
    );
    channel.bind_address_space(address_space).unwrap();
    channel
}

#[test]
fn decodes_every_descriptor_and_switch_mode_without_retaining_guest_memory() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&descriptor(0x12_3456_7000, 3, 0));
    bytes.extend_from_slice(&descriptor(
        0x23_4567_8000,
        5,
        ENTRY1_ALLOW_FLUSH | ENTRY1_LEVEL_SUBROUTINE | ENTRY1_SYNC_WAIT,
    ));
    let decoded = decode_gpfifo_submission(
        SWITCH_1_GM20B_PROFILE,
        0x800,
        request(
            2,
            SUBMIT_HARDWARE_FORMAT
                | SUBMIT_FENCE_WAIT
                | SUBMIT_SUPPRESS_WFI
                | SUBMIT_FENCE_INCREMENT_VALUE,
        ),
        &bytes,
    )
    .unwrap();

    assert_eq!(decoded.fence_id(), 7);
    assert_eq!(decoded.fence_value(), 11);
    assert!(decoded.mode().fence_wait());
    assert!(decoded.mode().suppress_wfi());
    assert!(decoded.mode().fence_increment_value());
    assert_eq!(decoded.entries()[0].address().get(), 0x12_3456_7000);
    assert_eq!(decoded.entries()[0].word_count(), 3);
    assert_eq!(decoded.entries()[0].byte_count(), 12);
    assert_eq!(decoded.entries()[1].level(), MaxwellGpfifoLevel::Subroutine);
    assert_eq!(decoded.entries()[1].sync(), MaxwellGpfifoSyncMode::Wait);
    assert!(decoded.entries()[1].allow_flush());
}

#[test]
fn rejects_count_size_and_profile_range_failures() {
    assert_eq!(
        decode_gpfifo_submission(
            SWITCH_1_GM20B_PROFILE,
            1,
            request(2, SUBMIT_HARDWARE_FORMAT),
            &[0; 16],
        ),
        Err(MaxwellGpfifoDecodeError::Invalid(
            MaxwellInvalidGpfifoSubmission::EntryCountExceedsAllocation {
                requested: 2,
                allocated: 1,
            }
        ))
    );
    assert!(matches!(
        decode_gpfifo_submission(
            SWITCH_1_GM20B_PROFILE,
            2,
            request(2, SUBMIT_HARDWARE_FORMAT),
            &[0; 8],
        ),
        Err(MaxwellGpfifoDecodeError::Invalid(
            MaxwellInvalidGpfifoSubmission::EntryByteCountMismatch { .. }
        ))
    ));
    let at_end = descriptor((1_u64 << 40) - 4, 1, 0);
    assert!(matches!(
        decode_gpfifo_submission(
            SWITCH_1_GM20B_PROFILE,
            1,
            request(1, SUBMIT_HARDWARE_FORMAT),
            &at_end,
        ),
        Err(MaxwellGpfifoDecodeError::Invalid(
            MaxwellInvalidGpfifoSubmission::PushbufferRangeOutOfRange { .. }
        ))
    ));
}

#[test]
fn rejects_unsupported_modes_before_returning_any_prefix() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&descriptor(0x1000, 4, 0));
    bytes.extend_from_slice(&descriptor(0x2001, 4, 0));
    assert_eq!(
        decode_gpfifo_submission(
            SWITCH_1_GM20B_PROFILE,
            2,
            request(2, SUBMIT_HARDWARE_FORMAT),
            &bytes,
        ),
        Err(MaxwellGpfifoDecodeError::Unsupported(
            MaxwellUnsupportedGpfifoSubmission::ConditionalFetch { entry: 1 }
        ))
    );
    assert!(matches!(
        decode_gpfifo_submission(
            SWITCH_1_GM20B_PROFILE,
            0,
            request(0, SUBMIT_HARDWARE_FORMAT | SUBMIT_SYNC_FENCE),
            &[],
        ),
        Err(MaxwellGpfifoDecodeError::Unsupported(
            MaxwellUnsupportedGpfifoSubmission::SyncFenceFileDescriptor
        ))
    ));
    assert_eq!(
        decode_gpfifo_submission(
            SWITCH_1_GM20B_PROFILE,
            0,
            MaxwellGpfifoSubmitRequest {
                entry_count: 0,
                flags: SUBMIT_HARDWARE_FORMAT | SUBMIT_FENCE_GET | SUBMIT_FENCE_INCREMENT_VALUE,
                fence_id: 0,
                fence_value: 1,
            },
            &[],
        ),
        Err(MaxwellGpfifoDecodeError::Unsupported(
            MaxwellUnsupportedGpfifoSubmission::ConflictingFenceCompletionModes
        ))
    );
    assert_eq!(
        decode_gpfifo_submission(
            SWITCH_1_GM20B_PROFILE,
            0,
            MaxwellGpfifoSubmitRequest {
                entry_count: 0,
                flags: SUBMIT_HARDWARE_FORMAT | SUBMIT_FENCE_INCREMENT_VALUE,
                fence_id: 0,
                fence_value: 0,
            },
            &[],
        ),
        Err(MaxwellGpfifoDecodeError::Unsupported(
            MaxwellUnsupportedGpfifoSubmission::ZeroFenceIncrementValue
        ))
    );
}

#[test]
fn accepts_a_completely_empty_hardware_format_submission() {
    let decoded = decode_gpfifo_submission(
        SWITCH_1_GM20B_PROFILE,
        0x800,
        request(0, SUBMIT_HARDWARE_FORMAT),
        &[],
    )
    .unwrap();
    assert!(decoded.entries().is_empty());
}

#[test]
fn resolves_complete_ranges_and_retains_exact_mapping_lifetimes() {
    let machine = Machine::new(16 * PAGE_SIZE).unwrap();
    let first_allocation = memory::allocate(&machine, 0x1000);
    let second_allocation = memory::allocate(&machine, 0x1000);
    memory::write(&first_allocation, 0xffc, &[1, 2, 3, 4]);
    memory::write(&second_allocation, 0, &[5, 6, 7, 8, 9, 10, 11, 12]);
    let mut address_space = memory::address_space(&machine, 9);
    let first = memory::map(&mut address_space, &first_allocation, 1);
    let second = memory::map(&mut address_space, &second_allocation, 2);
    assert_eq!(second.offset().get(), first.offset().get() + 0x1000);

    let pushbuffer = first.offset().get() + 0xffc;
    let decoded = decode_gpfifo_submission(
        SWITCH_1_GM20B_PROFILE,
        4,
        request(1, SUBMIT_HARDWARE_FORMAT),
        &descriptor(pushbuffer, 3, 0),
    )
    .unwrap();
    let channel = bound_channel(4, address_space.id());
    let validated = resolve_gpfifo_submission(
        &channel.configuration(),
        FrontendSubmissionId::new(7),
        decoded,
        address_space.memory(),
    )
    .unwrap();

    assert_eq!(validated.pushbuffers()[0].source().segments().len(), 2);
    let capture = validated.capture();
    assert_eq!(capture.channel(), MaxwellChannelId::new(4));
    assert_eq!(capture.frontend(), FrontendSubmissionId::new(7));
    assert_eq!(capture.total_entries(), 1);
    assert_eq!(capture.total_sources(), 2);
    assert_eq!(capture.sources()[0].word_offset, 0);
    assert_eq!(capture.sources()[1].word_offset, 1);
    let mut bytes = [0; 12];
    address_space
        .memory()
        .read_resolved(
            validated.pushbuffers()[0].source(),
            &mut bytes,
            &mut |_, _| panic!("RAM command read entered backend"),
        )
        .unwrap();
    assert_eq!(bytes, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    let accepted = address_space
        .memory()
        .capture(validated.pushbuffers()[0].source())
        .unwrap();
    drop(first_allocation);
    address_space.unmap(first.offset()).unwrap();
    assert!(matches!(
        validated.validate_sources(address_space.memory()),
        Err(MaxwellGpfifoSourceError::StaleMapping { location })
            if location.entry_index == 0
                && location.word_offset == 0
                && location.mapping == first.id()
                && location.generation == first.generation()
    ));
    let accepted = nixe_memory::CapturedMemory::new(accepted).unwrap();
    let mut retained_bytes = [0; 12];
    machine
        .read_captured_into(&accepted, 0, &mut retained_bytes)
        .unwrap()
        .into_ready()
        .unwrap();
    assert_eq!(retained_bytes, bytes);
    assert_eq!(machine.statistics().allocated_bytes, 2 * PAGE_SIZE);
    drop(accepted);
    assert_eq!(machine.statistics().allocated_bytes, PAGE_SIZE);
}

#[test]
fn rejects_a_later_hole_before_returning_a_retained_prefix() {
    let machine = Machine::new(16 * PAGE_SIZE).unwrap();
    let allocation = memory::allocate(&machine, 0x1000);
    let mut address_space = memory::address_space(&machine, 3);
    let mapping = memory::map(&mut address_space, &allocation, 1);
    let mut entries = Vec::new();
    entries.extend_from_slice(&descriptor(mapping.offset().get(), 1, 0));
    entries.extend_from_slice(&descriptor(mapping.offset().get() + 0x2000, 1, 0));
    let decoded = decode_gpfifo_submission(
        SWITCH_1_GM20B_PROFILE,
        4,
        request(2, SUBMIT_HARDWARE_FORMAT),
        &entries,
    )
    .unwrap();

    assert!(matches!(
        resolve_gpfifo_submission(
            &bound_channel(1, address_space.id()),
            FrontendSubmissionId::new(1),
            decoded,
            address_space.memory(),
        ),
        Err(MaxwellGpfifoSourceError::Resolution {
            entry_index: 1,
            error: MaxwellGpuAccessError::UnmappedAddress { .. },
            ..
        })
    ));
}

#[test]
fn multiple_entry_subroutine_capture_replays_the_same_first_packet() {
    let machine = Machine::new(16 * PAGE_SIZE).unwrap();
    let allocation = memory::allocate(&machine, 0x2000);
    memory::write(
        &allocation,
        0,
        &[0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88],
    );
    let mut address_space = memory::address_space(&machine, 5);
    let mapping = memory::map(&mut address_space, &allocation, 3);
    let mut entries = Vec::new();
    entries.extend_from_slice(&descriptor(mapping.offset().get(), 1, 0));
    entries.extend_from_slice(&descriptor(
        mapping.offset().get() + 4,
        1,
        ENTRY1_LEVEL_SUBROUTINE | ENTRY1_SYNC_WAIT,
    ));
    let decoded = decode_gpfifo_submission(
        SWITCH_1_GM20B_PROFILE,
        4,
        request(2, SUBMIT_HARDWARE_FORMAT),
        &entries,
    )
    .unwrap();
    let channel = bound_channel(6, address_space.id());
    let validated = resolve_gpfifo_submission(
        &channel.configuration(),
        FrontendSubmissionId::new(12),
        decoded,
        address_space.memory(),
    )
    .unwrap();

    assert_eq!(validated.pushbuffers().len(), 2);
    assert_eq!(
        validated.pushbuffers()[1].entry().level(),
        MaxwellGpfifoLevel::Subroutine
    );
    assert_eq!(
        validated.pushbuffers()[1].entry().sync(),
        MaxwellGpfifoSyncMode::Wait
    );
    let first_location = validated.first_packet_location().unwrap().unwrap();
    let replay = validated.clone();
    assert_eq!(replay.capture(), validated.capture());
    assert_eq!(
        replay.first_packet_location().unwrap(),
        Some(first_location)
    );
    assert_eq!(first_location.entry_index, 0);
    assert_eq!(first_location.pushbuffer, mapping.offset());
    assert_eq!(first_location.word_offset, 0);
}

#[test]
fn capture_is_hard_bounded_without_changing_submission_identity() {
    let machine = Machine::new(16 * PAGE_SIZE).unwrap();
    let allocation = memory::allocate(&machine, 0x1000);
    let mut address_space = memory::address_space(&machine, 6);
    let mapping = memory::map(&mut address_space, &allocation, 20);
    let entry_count = u32::try_from(MAXWELL_GPFIFO_CAPTURE_SOURCES + 1).unwrap();
    let mut entries = Vec::new();
    for _ in 0..entry_count {
        entries.extend_from_slice(&descriptor(mapping.offset().get(), 1, 0));
    }
    let decoded = decode_gpfifo_submission(
        SWITCH_1_GM20B_PROFILE,
        entry_count,
        request(entry_count, SUBMIT_HARDWARE_FORMAT),
        &entries,
    )
    .unwrap();
    let validated = resolve_gpfifo_submission(
        &bound_channel(7, address_space.id()),
        FrontendSubmissionId::new(14),
        decoded,
        address_space.memory(),
    )
    .unwrap();

    let capture = validated.capture();
    assert_eq!(capture.channel(), validated.channel());
    assert_eq!(capture.frontend(), validated.frontend());
    assert_eq!(capture.total_entries(), entry_count as usize);
    assert_eq!(capture.total_sources(), entry_count as usize);
    assert_eq!(capture.sources().len(), MAXWELL_GPFIFO_CAPTURE_SOURCES);
    assert_eq!(capture.omitted_sources(), 1);
    assert_eq!(
        validated.first_packet_location().unwrap(),
        validated.clone().first_packet_location().unwrap()
    );
}

#[test]
fn remapping_the_same_gpu_va_cannot_retarget_a_retained_submission() {
    let machine = Machine::new(16 * PAGE_SIZE).unwrap();
    let original = memory::allocate(&machine, 0x1000);
    let replacement = memory::allocate(&machine, 0x1000);
    memory::write(&original, 0, &[1, 2, 3, 4]);
    memory::write(&replacement, 0, &[9, 8, 7, 6]);
    let mut address_space = memory::address_space(&machine, 7);
    let first = memory::map(&mut address_space, &original, 4);
    let channel = bound_channel(8, address_space.id());
    let decoded = decode_gpfifo_submission(
        SWITCH_1_GM20B_PROFILE,
        1,
        request(1, SUBMIT_HARDWARE_FORMAT),
        &descriptor(first.offset().get(), 1, 0),
    )
    .unwrap();
    let retained = resolve_gpfifo_submission(
        &channel.configuration(),
        FrontendSubmissionId::new(13),
        decoded,
        address_space.memory(),
    )
    .unwrap();

    address_space.unmap(first.offset()).unwrap();
    let remapped = memory::map(&mut address_space, &replacement, 5);
    assert_eq!(remapped.offset(), first.offset());
    assert_ne!(remapped.id(), first.id());
    assert_ne!(remapped.generation(), first.generation());
    assert!(matches!(
        retained.validate_sources(address_space.memory()),
        Err(MaxwellGpfifoSourceError::StaleMapping { location })
            if location.mapping == first.id()
                && location.generation == first.generation()
    ));

    let mut original_bytes = [0; 4];
    machine
        .capture_stream(
            &[AllocationRange::new(original.pin(), 0, 4).unwrap()],
            &mut original_bytes,
        )
        .unwrap();
    assert_eq!(original_bytes.as_ref(), &[1, 2, 3, 4]);
    let mut replacement_bytes = [0; 4];
    let replacement_range = address_space
        .memory()
        .resolve_range(remapped.offset(), 4, MemoryPermissions::READ)
        .unwrap();
    address_space
        .memory()
        .read_resolved(&replacement_range, &mut replacement_bytes, &mut |_, _| {
            panic!("RAM command read entered backend")
        })
        .unwrap();
    assert_eq!(replacement_bytes, [9, 8, 7, 6]);
}

fn exercise_command_source(split: bool, wait: bool, unmap_during_entry: bool) {
    use nixe_gpu::CacheMaintenanceOperation;
    use std::{cell::Cell, sync::Arc};
    let machine = Machine::new(PAGE_SIZE).unwrap();
    let allocation = memory::allocate(&machine, PAGE_SIZE);
    let mut space = memory::address_space(&machine, 1);
    let mapping = memory::map(&mut space, &allocation, 1);
    let mut channel = bound_channel(1, space.id());
    let header = (1u32 << 29) | (1 << 16) | (6 << 13) | (0x002c / 4);
    let words = [header, 0x8000_0000, header, 0x8000_0000];
    let bytes: Vec<_> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    memory::write(&allocation, 0, &bytes);
    let mut descriptors = Vec::new();
    let entries: u32 = if split { 2 } else { 1 };
    for index in 0..entries {
        let flags = if index == 1 && wait {
            ENTRY1_SYNC_WAIT
        } else {
            0
        };
        descriptors.extend_from_slice(&descriptor(
            mapping.offset().get() + u64::from(index) * 8,
            if split { 2 } else { 4 },
            flags,
        ));
    }
    let decoded = decode_gpfifo_submission(
        SWITCH_1_GM20B_PROFILE,
        entries,
        request(entries, SUBMIT_HARDWARE_FORMAT),
        &descriptors,
    )
    .unwrap();
    let submission = resolve_gpfifo_submission(
        &channel.configuration(),
        FrontendSubmissionId::new(1),
        decoded,
        space.memory(),
    )
    .unwrap();
    let mut scheduler = MaxwellScheduler::default();
    scheduler
        .enqueue(&channel.configuration(), submission, None, None)
        .unwrap();
    let dispatch = scheduler
        .dispatch_next(true, space.memory())
        .unwrap()
        .unwrap();
    let reader = space.memory().clone();
    let captured_before = machine.tracking_statistics().captured_bytes;
    let mut allocation = Some(allocation);
    let waits = Cell::new(0);
    let mut observed = Vec::new();
    let mut access = |ranges: Vec<AllocationRange>, operation| {
        machine
            .apply_resolved_access(&ranges, &operation)
            .map_err(Arc::new)
    };
    let tail = stream_maxwell_frontend(
        &dispatch,
        &mut channel,
        &reader,
        &mut MaxwellNativeFrontend::new(Default::default()),
        &mut MaxwellCommandSourceScratch::default(),
        &mut access,
        &mut || {
            waits.set(waits.get() + 1);
            Ok(())
        },
        &mut |steps| {
            for step in steps {
                let MaxwellSubmissionExecutionStep::Ordering(operation) = step else {
                    panic!("unexpected operation in command capture fixture")
                };
                observed.push(operation);
                if observed.len() == 1 {
                    assert_eq!(waits.get(), 0);
                    memory::write(
                        allocation.as_ref().unwrap(),
                        12,
                        &0x7000_0000u32.to_le_bytes(),
                    );
                    if unmap_during_entry {
                        space.unmap(mapping.offset()).unwrap();
                        drop(allocation.take());
                    }
                }
            }
            Ok(())
        },
    )
    .unwrap();
    assert!(tail.into_steps().is_empty());
    assert_eq!(waits.get(), usize::from(wait));
    assert_eq!(
        observed,
        [
            CacheMaintenanceOperation::FlushDirtyDeviceWrites,
            CacheMaintenanceOperation::InvalidateDeviceReadCaches
        ]
    );
    assert_eq!(
        machine.tracking_statistics().captured_bytes,
        captured_before,
        "contiguous command RAM must not be copied to entry scratch"
    );
    if unmap_during_entry {
        assert_eq!(
            machine.statistics().allocated_bytes,
            0,
            "the consumed entry must release its last physical pin"
        );
    }
}

#[test]
fn contiguous_command_entry_reads_later_words_from_live_ram() {
    exercise_command_source(false, false, false);
}

#[test]
fn active_command_source_survives_unmap_and_releases_ram_after_consumption() {
    exercise_command_source(false, false, true);
}

#[test]
fn following_command_entry_observes_prior_owner_effects() {
    exercise_command_source(true, false, false);
}

#[test]
fn sync_wait_captures_following_sources_after_the_real_dependency() {
    exercise_command_source(true, true, false);
}
