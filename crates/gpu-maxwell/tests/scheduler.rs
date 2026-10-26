use nixe_gpu::{FrontendSubmissionId, GuestTimelinePoint};
use nixe_gpu_maxwell::*;
use nixe_memory::{Allocation, Machine, MemoryPermissions, PAGE_SIZE};

#[path = "support/memory.rs"]
mod memory;

const HARDWARE_FORMAT: u32 = 1 << 2;

fn descriptor(address: u64, words: u32) -> [u8; 8] {
    let word0 = address as u32;
    let word1 = ((address >> 32) as u32 & 0xff) | (words << 10);
    let mut bytes = [0; 8];
    bytes[..4].copy_from_slice(&word0.to_le_bytes());
    bytes[4..].copy_from_slice(&word1.to_le_bytes());
    bytes
}

fn setup() -> (Allocation, MaxwellGpuAddressSpace, MaxwellGpuChannel) {
    let machine = Machine::new(16 * PAGE_SIZE).unwrap();
    let allocation = memory::allocate(&machine, 0x1000);
    memory::write(&allocation, 0, &[0x21, 0x43, 0x65, 0x87]);
    let mut address_space = memory::address_space(&machine, 3);
    let mapping = memory::map(&mut address_space, &allocation, 9);
    let mut channel = MaxwellGpuChannel::new(
        MaxwellChannelId::new(7),
        MaxwellChannelOwner::new(1),
        SWITCH_1_GM20B_PROFILE,
    );
    channel.bind_address_space(address_space.id()).unwrap();
    let decoded = decode_gpfifo_submission(
        SWITCH_1_GM20B_PROFILE,
        8,
        MaxwellGpfifoSubmitRequest {
            entry_count: 1,
            flags: HARDWARE_FORMAT,
            fence_id: 0,
            fence_value: 0,
        },
        &descriptor(mapping.offset().get(), 1),
    )
    .unwrap();
    // Ensure the helper itself exercises a valid decoded entry.
    validate_gpfifo_submission(
        &channel.configuration(),
        FrontendSubmissionId::new(0),
        decoded,
        address_space.memory(),
    )
    .unwrap();
    (allocation, address_space, channel)
}

fn submission(
    channel: &MaxwellChannelConfiguration,
    address_space: &MaxwellGpuAddressSpace,
    frontend: u64,
) -> MaxwellValidatedGpfifoSubmission {
    let mapping = address_space.mapping_dump(1).entries[0];
    let decoded = decode_gpfifo_submission(
        SWITCH_1_GM20B_PROFILE,
        8,
        MaxwellGpfifoSubmitRequest {
            entry_count: 1,
            flags: HARDWARE_FORMAT,
            fence_id: 0,
            fence_value: 0,
        },
        &descriptor(mapping.offset.get(), 1),
    )
    .unwrap();
    validate_gpfifo_submission(
        channel,
        FrontendSubmissionId::new(frontend),
        decoded,
        address_space.memory(),
    )
    .unwrap()
}

#[test]
fn global_fifo_order_is_independent_of_channel_identity() {
    let (_allocation, address_space, first_channel) = setup();
    let mut second_channel = MaxwellGpuChannel::new(
        MaxwellChannelId::new(2),
        MaxwellChannelOwner::new(1),
        SWITCH_1_GM20B_PROFILE,
    );
    second_channel
        .bind_address_space(address_space.id())
        .unwrap();
    let mut scheduler = MaxwellScheduler::default();
    assert_eq!(
        scheduler
            .enqueue(
                &first_channel.configuration(),
                submission(&first_channel.configuration(), &address_space, 10),
                None,
                None,
            )
            .unwrap(),
        MaxwellSchedulerSequence::new(1)
    );
    assert_eq!(
        scheduler
            .enqueue(
                &second_channel.configuration(),
                submission(&second_channel.configuration(), &address_space, 11),
                None,
                None,
            )
            .unwrap(),
        MaxwellSchedulerSequence::new(2)
    );

    let first = scheduler
        .dispatch_next(true, address_space.memory())
        .unwrap()
        .unwrap();
    let second = scheduler
        .dispatch_next(true, address_space.memory())
        .unwrap()
        .unwrap();
    assert_eq!(first.scheduled().channel(), first_channel.id());
    assert_eq!(second.scheduled().channel(), second_channel.id());
    assert_eq!(scheduler.pending_count(), 0);
}

#[test]
fn zero_stream_increment_submits_without_a_completion_reservation() {
    let (_allocation, address_space, channel) = setup();
    let mapping = address_space.mapping_dump(1).entries[0];
    let decoded = decode_gpfifo_submission(
        SWITCH_1_GM20B_PROFILE,
        8,
        MaxwellGpfifoSubmitRequest {
            entry_count: 1,
            flags: HARDWARE_FORMAT | (1 << 8),
            fence_id: 0,
            fence_value: 0,
        },
        &descriptor(mapping.offset.get(), 1),
    )
    .unwrap();
    let validated = validate_gpfifo_submission(
        &channel.configuration(),
        FrontendSubmissionId::new(12),
        decoded,
        address_space.memory(),
    )
    .unwrap();
    let mut scheduler = MaxwellScheduler::default();
    scheduler
        .enqueue(&channel.configuration(), validated, None, None)
        .unwrap();
    assert!(
        scheduler
            .dispatch_next(true, address_space.memory())
            .unwrap()
            .unwrap()
            .scheduled()
            .completion()
            .is_none()
    );
}

#[test]
fn dispatch_defers_stale_mapping_detection_to_command_consumption() {
    let (_allocation, mut address_space, channel) = setup();
    let validated = submission(&channel.configuration(), &address_space, 3);
    let mapping = address_space.mapping_dump(1).entries[0];
    let mut scheduler = MaxwellScheduler::default();
    scheduler
        .enqueue(&channel.configuration(), validated, None, None)
        .unwrap();
    address_space.unmap(mapping.offset).unwrap();

    let dispatched = scheduler
        .dispatch_next(true, address_space.memory())
        .unwrap()
        .unwrap();
    assert_eq!(scheduler.pending_count(), 0);
    let entry = dispatched.scheduled().submission().decoded().entries()[0];
    assert!(matches!(
        address_space.memory().resolve_range(
            entry.address(),
            entry.byte_count(),
            MemoryPermissions::READ
        ),
        Err(MaxwellGpuAccessError::UnmappedAddress { .. })
    ));
}

#[test]
fn frontend_boundary_retains_ordering_and_never_claims_completion() {
    let (_allocation, address_space, channel) = setup();
    let mut scheduler = MaxwellScheduler::default();
    scheduler
        .enqueue(
            &channel.configuration(),
            submission(&channel.configuration(), &address_space, 4),
            None,
            None,
        )
        .unwrap();
    let dispatch = scheduler
        .dispatch_next(true, address_space.memory())
        .unwrap()
        .unwrap();
    assert_eq!(
        dispatch.scheduled().stage(),
        MaxwellSubmissionOrderingStage::FrontendDispatched
    );
    assert!(dispatch.scheduled().completion().is_none());
}

#[test]
fn channel_cancellation_preserves_other_channels_and_does_not_dispatch() {
    let (_allocation, address_space, first_channel) = setup();
    let mut second_channel = MaxwellGpuChannel::new(
        MaxwellChannelId::new(8),
        MaxwellChannelOwner::new(1),
        SWITCH_1_GM20B_PROFILE,
    );
    second_channel
        .bind_address_space(address_space.id())
        .unwrap();
    let mut scheduler = MaxwellScheduler::default();
    scheduler
        .enqueue(
            &first_channel.configuration(),
            submission(&first_channel.configuration(), &address_space, 5),
            None,
            None,
        )
        .unwrap();
    scheduler
        .enqueue(
            &second_channel.configuration(),
            submission(&second_channel.configuration(), &address_space, 6),
            None,
            None,
        )
        .unwrap();

    let cancelled = scheduler.cancel_channel(first_channel.id());
    assert_eq!(cancelled.len(), 1);
    assert_eq!(
        cancelled[0].stage(),
        MaxwellSubmissionOrderingStage::Cancelled
    );
    let remaining = scheduler
        .dispatch_next(true, address_space.memory())
        .unwrap()
        .unwrap();
    assert_eq!(remaining.scheduled().channel(), second_channel.id());
}

#[test]
fn pending_dependency_and_teardown_retain_then_release_whole_work() {
    let (_allocation, address_space, channel) = setup();
    let dependency = GuestTimelinePoint::new(
        nixe_gpu::GuestSyncpointId::new(9),
        nixe_gpu::GuestSyncpointValue::new(3),
    );
    let decoded = decode_gpfifo_submission(
        SWITCH_1_GM20B_PROFILE,
        8,
        MaxwellGpfifoSubmitRequest {
            entry_count: 1,
            flags: HARDWARE_FORMAT | 1,
            fence_id: dependency.syncpoint().get(),
            fence_value: dependency.value().get(),
        },
        &descriptor(address_space.mapping_dump(1).entries[0].offset.get(), 1),
    )
    .unwrap();
    let validated = validate_gpfifo_submission(
        &channel.configuration(),
        FrontendSubmissionId::new(8),
        decoded,
        address_space.memory(),
    )
    .unwrap();
    let mut scheduler = MaxwellScheduler::default();
    scheduler
        .enqueue(&channel.configuration(), validated, Some(dependency), None)
        .unwrap();

    assert!(
        matches!(scheduler.dispatch_next(false, address_space.memory()), Err(MaxwellScheduleError::PendingDependency(actual)) if actual == dependency)
    );
    assert_eq!(scheduler.pending_count(), 1);
    assert_eq!(scheduler.clear(), 1);
    assert_eq!(scheduler.pending_count(), 0);
    assert!(matches!(
        scheduler.dispatch_next(true, address_space.memory()),
        Ok(None)
    ));
}
