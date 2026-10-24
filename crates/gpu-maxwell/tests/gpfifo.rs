#[path = "support/command.rs"]
mod command;
use nixe_gpu::FrontendSubmissionId;
use nixe_gpu_maxwell::*;
use nixe_memory::{Machine, PAGE_SIZE};

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
fn multiple_entry_subroutine_metadata_survives_submission_validation() {
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
    let validated = validate_gpfifo_submission(
        &channel.configuration(),
        FrontendSubmissionId::new(12),
        decoded,
        address_space.memory(),
    )
    .unwrap();

    assert_eq!(validated.decoded().entries().len(), 2);
    assert_eq!(
        validated.decoded().entries()[1].level(),
        MaxwellGpfifoLevel::Subroutine
    );
    assert_eq!(
        validated.decoded().entries()[1].sync(),
        MaxwellGpfifoSyncMode::Wait
    );
    let replay = validated.clone();
    assert_eq!(replay, validated);
    assert_eq!(replay.decoded().entries()[0].address(), mapping.offset());
}

fn exercise_command_source(split: bool, wait: bool, unmap_during_entry: bool) {
    use nixe_gpu::{CacheMaintenanceOperation, native::NativeCommand};
    use std::cell::RefCell;
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
    let submission = validate_gpfifo_submission(
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
    let mut observed = Vec::new();
    let mut consume = |work: nixe_gpu::native::NativeWork| {
        let NativeCommand::Ordering(operation) = work.command else {
            panic!("unexpected native work in command source fixture")
        };
        observed.push(operation);
        if observed.len() == 1 {
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
    };
    let mut host = command::Observer {
        memory: &machine,
        consume: &mut consume,
        controls: 0,
    };
    let frontend = RefCell::new(MaxwellNativeFrontend::new(Default::default()));
    let mut scratch = MaxwellCommandSourceScratch::default();
    command::ready(stream_maxwell_frontend(
        &dispatch,
        &mut channel,
        &reader,
        &frontend,
        &mut scratch,
        &mut host,
    ))
    .unwrap();
    if wait {
        assert!(host.controls > 0);
    }
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
fn sync_wait_entry_reads_following_source_after_prior_effect() {
    exercise_command_source(true, true, false);
}

#[test]
fn later_entry_fetch_uses_the_mapping_current_after_prior_work() {
    use nixe_gpu::{CacheMaintenanceOperation, native::NativeCommand};
    use std::cell::RefCell;

    let machine = Machine::new(8 * PAGE_SIZE).unwrap();
    let first_bytes = memory::allocate(&machine, PAGE_SIZE);
    let old_bytes = memory::allocate(&machine, PAGE_SIZE);
    let replacement_bytes = memory::allocate(&machine, PAGE_SIZE);
    let header = (1u32 << 29) | (1 << 16) | (6 << 13) | (0x002c / 4);
    for (allocation, argument) in [
        (&first_bytes, 0x8000_0000u32),
        (&old_bytes, 0x8000_0000u32),
        (&replacement_bytes, 0x7000_0000u32),
    ] {
        memory::write(allocation, 0, &header.to_le_bytes());
        memory::write(allocation, 4, &argument.to_le_bytes());
    }
    let mut space = memory::address_space(&machine, 10);
    let first = memory::map(&mut space, &first_bytes, 1);
    let old = memory::map(&mut space, &old_bytes, 2);
    let mut descriptors = Vec::new();
    descriptors.extend_from_slice(&descriptor(first.offset().get(), 2, 0));
    descriptors.extend_from_slice(&descriptor(old.offset().get(), 2, 0));
    let decoded = decode_gpfifo_submission(
        SWITCH_1_GM20B_PROFILE,
        2,
        request(2, SUBMIT_HARDWARE_FORMAT),
        &descriptors,
    )
    .unwrap();
    let mut channel = bound_channel(10, space.id());
    let submission = validate_gpfifo_submission(
        &channel.configuration(),
        FrontendSubmissionId::new(10),
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
    let mut observed = Vec::new();
    let mut consume = |work: nixe_gpu::native::NativeWork| {
        let NativeCommand::Ordering(operation) = work.command else {
            panic!("unexpected native work")
        };
        observed.push(operation);
        if observed.len() == 1 {
            space.unmap(old.offset()).unwrap();
            let replacement = memory::map(&mut space, &replacement_bytes, 3);
            assert_eq!(replacement.offset(), old.offset());
        }
    };
    let mut host = command::Observer {
        memory: &machine,
        consume: &mut consume,
        controls: 0,
    };
    command::ready(stream_maxwell_frontend(
        &dispatch,
        &mut channel,
        &reader,
        &RefCell::new(MaxwellNativeFrontend::new(Default::default())),
        &mut MaxwellCommandSourceScratch::default(),
        &mut host,
    ))
    .unwrap();
    assert_eq!(
        observed,
        [
            CacheMaintenanceOperation::FlushDirtyDeviceWrites,
            CacheMaintenanceOperation::InvalidateDeviceReadCaches,
        ]
    );
}
