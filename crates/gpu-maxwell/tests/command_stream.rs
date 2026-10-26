#[path = "support/command.rs"]
mod command;
#[path = "support/stream.rs"]
mod stream;
use nixe_gpu::{CacheMaintenanceOperation, native::NativeCommand};
use nixe_gpu_maxwell::{
    MaxwellFrontendDispatchError, MaxwellPushbufferDecodeError, SWITCH_1_GM20B_PROFILE,
};
use stream::{Fixture, header, packet};

#[test]
fn solid_rectangles_preserve_color_and_zeta_destinations_on_cold_and_cached_paths() {
    use nixe_gpu::{ClearValue, ImageFormat, native::GuestImageAspect};
    let packed = 0x7f12_3456u32;
    for (kind, zeta, expected) in [
        (
            0xdb,
            false,
            ClearValue::Color([0x12, 0x34, 0x56, 0x7f].map(|v| v as f32 / 255.0)),
        ),
        (
            0x17,
            true,
            ClearValue::DepthStencil {
                depth: 0x123456 as f32 / 16777215.0,
                stencil: 0x7f,
            },
        ),
        (
            0x51,
            true,
            ClearValue::DepthStencil {
                depth: 0x7f1234 as f32 / 16777215.0,
                stencil: 0x56,
            },
        ),
    ] {
        let mut base = 0;
        let mut fixture = Fixture::new(|address| {
            let address = address + 64 * nixe_memory::PAGE_SIZE;
            base = address;
            let mut words = Vec::new();
            for (method, value) in [
                (0, SWITCH_1_GM20B_PROFILE.classes().two_d().0),
                (0x0290, 0),
                (0x02ac, 3),
                (0x02b8, u32::from(zeta)),
                (0x0200, 0xcf),
                (0x0204, 0),
                (0x0208, 0),
                (0x020c, 1),
                (0x0210, 0),
                (0x0218, 32),
                (0x021c, 16),
                (0x0220, (address >> 32) as u32),
                (0x0224, address as u32),
                (0x02d4, 1),
                (0x0580, 4),
                (0x0584, 0xcf),
                (0x0540, packed),
                (0x0600, 2),
                (0x0604, 3),
                (0x0608, 30),
                (0x060c, 16),
            ] {
                let at = words.len();
                packet(&mut words, method, &[value]);
                words[at] |= 3 << 13;
            }
            vec![words]
        });
        let image_mapping = fixture
            .space
            .map(nixe_gpu_maxwell::MaxwellMapRequest {
                allocation: nixe_gpu_maxwell::MaxwellAllocationId::new(1),
                backing: vec![
                    nixe_memory::AllocationRange::new(fixture.allocation.pin(), 0, 0x1000).unwrap(),
                ],
                backing_offset: 0,
                size: 0x1000,
                allocation_alignment: 0x1000,
                page_size: 0,
                kind,
                cacheable: false,
                permissions: nixe_memory::MemoryPermissions::READ_WRITE,
                fixed_offset: None,
            })
            .unwrap();
        assert_eq!(image_mapping.offset().get(), base);
        let mut clears = 0;
        for _ in 0..2 {
            fixture
                .run(&mut |work| {
                    let NativeCommand::Clear {
                        image,
                        rectangle,
                        value,
                    } = work.command
                    else {
                        panic!("solid rectangle must emit a native clear");
                    };
                    assert_eq!(value, expected);
                    assert_eq!(
                        (rectangle.x, rectangle.y, rectangle.width, rectangle.height),
                        (2, 3, 28, 13)
                    );
                    assert_eq!(
                        image.description.format(),
                        if zeta {
                            ImageFormat::Depth24UnormStencil8Uint
                        } else {
                            ImageFormat::Bgra8Unorm
                        }
                    );
                    let aspects: Vec<_> = image.planes.iter().map(|plane| plane.aspect).collect();
                    assert_eq!(
                        aspects,
                        if zeta {
                            vec![GuestImageAspect::Depth, GuestImageAspect::Stencil]
                        } else {
                            vec![GuestImageAspect::Color]
                        }
                    );
                    clears += 1;
                })
                .unwrap();
        }
        assert_eq!(clears, 2);
    }
}

#[test]
fn all_packet_modes_consume_repeated_effects_once_across_entry_boundaries() {
    let mut fixture = Fixture::new(|base| {
        let mut first = vec![header(4, 0x50, 73), header(5, 0x28, 3), 0, 0x8000_0000];
        // The increase-once continuation must remain nonincrementing.
        let mut second = vec![0x7000_0000, header(3, 0x78, 2), 0, 0];
        packet(
            &mut second,
            0x10,
            &[(base >> 32) as u32, base as u32, 0x1234_5678, 0x0100_0002],
        );
        first.shrink_to_fit();
        vec![first, second]
    });
    let mut kinds = Vec::new();
    fixture
        .run(&mut |work| match work.command {
            NativeCommand::WaitForIdle => kinds.push(0),
            NativeCommand::Ordering(CacheMaintenanceOperation::FlushDirtyDeviceWrites) => {
                kinds.push(1)
            }
            NativeCommand::Ordering(CacheMaintenanceOperation::InvalidateDeviceReadCaches) => {
                kinds.push(2)
            }
            NativeCommand::ReportMemory { bytes, fence, .. } => {
                assert!(fence);
                assert_eq!(bytes.as_ref(), 0x1234_5678u32.to_le_bytes());
                kinds.push(3);
            }
            other => panic!("unexpected operation: {other:?}"),
        })
        .unwrap();
    assert_eq!(kinds, [0, 1, 2, 0, 0, 3]);
}

#[test]
fn inline_upload_header_and_payload_can_cross_multiple_entries() {
    let mut fixture = Fixture::new(|base| {
        let mut prefix = Vec::new();
        let mut compute = |method, args: &[u32]| {
            let at = prefix.len();
            packet(&mut prefix, method, args);
            prefix[at] |= 1 << 13;
        };
        compute(0, &[SWITCH_1_GM20B_PROFILE.classes().compute().0]);
        compute(0x188, &[(base >> 32) as u32, base as u32]);
        compute(0x180, &[16, 1]);
        compute(0x1b0, &[0x41]);
        prefix.push(header(3, 0x1b4, 4) | (1 << 13));
        vec![prefix, vec![11, 22], vec![33, 44]]
    });
    fixture
        .run(&mut |_| panic!("inline upload is a direct memory effect"))
        .unwrap();
    assert_eq!(
        fixture.read(0, 16),
        [11u32, 22, 33, 44]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>()
    );
}

#[test]
fn embedded_inline_zero_and_partial_lengths_preserve_neighboring_bytes() {
    for (length, payload, expected) in [
        (3, vec![0x4433_2211], vec![0x11, 0x22, 0x33, 0, 0, 0, 0, 0]),
        (
            5,
            vec![0x4433_2211, 0x8877_6655],
            vec![0x11, 0x22, 0x33, 0x44, 0x55, 0, 0, 0],
        ),
    ] {
        let mut fixture = Fixture::new(|base| {
            let mut words = Vec::new();
            packet(
                &mut words,
                0,
                &[SWITCH_1_GM20B_PROFILE.classes().three_d().0],
            );
            packet(
                &mut words,
                0x0180,
                &[0, 1, (base >> 32) as u32, base as u32],
            );
            packet(&mut words, 0x01b0, &[0x1011]);
            packet(&mut words, 0x0180, &[4, 0]);
            packet(&mut words, 0x01b0, &[0x1011]);
            packet(&mut words, 0x0180, &[length, 1]);
            packet(&mut words, 0x01b0, &[0x1011]);
            if payload.len() == 1 {
                packet(&mut words, 0x01b4, &payload);
            } else {
                words.push(header(3, 0x01b4, payload.len() as u32));
                words.extend_from_slice(&payload);
            }
            vec![words]
        });
        fixture
            .run(&mut |_| panic!("inline upload must stay a direct memory effect"))
            .unwrap();
        assert_eq!(fixture.read(0, 8), expected);
    }
}

#[test]
fn malformed_tail_reports_source_after_consuming_the_valid_prefix() {
    let mut fixture = Fixture::new(|_| vec![vec![header(3, 0x78, 3), 0, 0]]);
    let mut waits = 0;
    let error = fixture
        .run(&mut |work| {
            assert!(matches!(work.command, NativeCommand::WaitForIdle));
            waits += 1;
        })
        .unwrap_err();
    assert_eq!(waits, 2);
    assert!(
        matches!(error,MaxwellFrontendDispatchError::PacketDecode(MaxwellPushbufferDecodeError::TruncatedPacket {source,expected_arguments:3,available_arguments:2}) if source.entry_index==0 && source.word_offset==0)
    );
}

#[test]
fn long_plain_packet_services_control_without_materializing_method_records() {
    let mut fixture = Fixture::new(|_| {
        vec![
            std::iter::once(header(3, 8, 8191))
                .chain(std::iter::repeat_n(0, 8191))
                .collect(),
        ]
    });
    assert_eq!(
        fixture.run(&mut |_| panic!("NOP has no effect")).unwrap(),
        3
    );
    let before = fixture.memory.tracking_statistics().captured_bytes;
    fixture.run(&mut |_| panic!("NOP has no effect")).unwrap();
    assert_eq!(
        fixture.memory.tracking_statistics().captured_bytes,
        before,
        "live command RAM is not captured"
    );
}

#[test]
fn end_segment_skips_only_its_own_entry() {
    let mut fixture = Fixture::new(|_| vec![vec![7 << 29, 2 << 29], vec![header(3, 0x78, 1), 0]]);
    let mut waits = 0;
    fixture
        .run(&mut |work| {
            assert!(matches!(work.command, NativeCommand::WaitForIdle));
            waits += 1;
        })
        .unwrap();
    assert_eq!(waits, 1);
}

mod allocation_counter {
    use std::{
        alloc::{GlobalAlloc, Layout, System},
        cell::Cell,
    };
    thread_local! { pub static COUNT: Cell<Option<u64>> = const { Cell::new(None) }; }
    struct Counter;
    #[global_allocator]
    static ALLOCATOR: Counter = Counter;
    unsafe impl GlobalAlloc for Counter {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let _ = COUNT.try_with(|n| {
                if let Some(value) = n.get() {
                    n.set(Some(value + 1));
                }
            });
            unsafe { System.alloc(layout) }
        }
        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
            let _ = COUNT.try_with(|n| {
                if let Some(value) = n.get() {
                    n.set(Some(value + 1));
                }
            });
            unsafe { System.realloc(ptr, layout, size) }
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            unsafe { System.dealloc(ptr, layout) }
        }
    }
}

#[test]
#[ignore = "opt-in command parser CPU/allocation measurement"]
#[cfg(target_os = "linux")]
fn command_stream_throughput() {
    let iterations = std::env::var("NIXE_COMMAND_BENCH_ITERS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(10_000);
    for words in [64, 8191] {
        let mut fixture = Fixture::new(|_| {
            vec![
                std::iter::once(header(3, 8, words))
                    .chain(std::iter::repeat_n(0, words as usize))
                    .collect(),
            ]
        });
        for _ in 0..100 {
            fixture.run(&mut |_| panic!("NOP has no effect")).unwrap();
        }
        let cpu_ns = || {
            std::fs::read_to_string("/proc/thread-self/schedstat")
                .unwrap()
                .split_whitespace()
                .next()
                .unwrap()
                .parse::<u64>()
                .unwrap()
        };
        let captures = fixture.memory.tracking_statistics().captured_bytes;
        let before = cpu_ns();
        allocation_counter::COUNT.with(|n| n.set(Some(0)));
        let started = std::time::Instant::now();
        let mut controls = 0;
        for _ in 0..iterations {
            controls += fixture.run(&mut |_| panic!("NOP has no effect")).unwrap();
        }
        let elapsed = started.elapsed().as_secs_f64();
        let allocations = allocation_counter::COUNT.with(|n| n.replace(None).unwrap());
        let cpu_nanoseconds = cpu_ns() - before;
        let captured_bytes = fixture.memory.tracking_statistics().captured_bytes - captures;
        println!(
            "{{\"workload\":\"command_nops\",\"words\":{words},\"iterations\":{iterations},\"seconds\":{elapsed},\"cpu_nanoseconds\":{cpu_nanoseconds},\"allocations\":{allocations},\"captured_bytes\":{captured_bytes},\"controls\":{controls}}}"
        );
    }
}

#[test]
fn user_and_driver_completion_counts_are_exact_and_consumption_never_signals() {
    use nixe_gpu::{
        GuestSyncpointId, GuestSyncpointValue, GuestTimeline, TimelineInstanceId, TimelineOwnerId,
    };
    for (driver, expected, actual) in [
        (false, 3, 3),
        (false, 3, 2),
        (false, 3, 4),
        (true, 2, 0),
        (true, 5, 3),
    ] {
        let owner = TimelineOwnerId::new(7);
        let mut timeline = GuestTimeline::new(
            GuestSyncpointId::new(1),
            TimelineInstanceId::new(1),
            owner,
            GuestSyncpointValue::new(0),
        );
        let reservation = timeline.reserve(owner, expected).unwrap();
        let mut fixture = Fixture::with_completion(
            4 | if driver { 2 } else { 1 << 8 },
            Some(reservation),
            |_| {
                let mut words = Vec::new();
                packet(
                    &mut words,
                    0,
                    &[SWITCH_1_GM20B_PROFILE.classes().three_d().0],
                );
                for _ in 0..actual {
                    packet(&mut words, 0x2c8, &[1]);
                }
                // A completion-producing method does not terminate consumption.
                packet(&mut words, 0x78, &[0]);
                vec![words]
            },
        );
        let mut effects = 0;
        let result = fixture.run(&mut |_| effects += 1);
        assert_eq!(
            result.is_ok(),
            actual + if driver { 2 } else { 0 } == expected
        );
        if actual <= expected {
            assert_eq!(effects, 1);
        }
        if let Err(error) = result {
            let detail = error.to_string();
            assert!(detail.contains("syncpoint increments"), "{detail}");
        }
        assert_eq!(timeline.current_point().value().get(), 0);
    }
}

#[test]
fn engine_waits_emit_ordering_and_notification_follows_it_without_cpu_publication() {
    let mut fixture = Fixture::new(|base| {
        let mut words = Vec::new();
        packet(
            &mut words,
            0,
            &[SWITCH_1_GM20B_PROFILE.classes().three_d().0],
        );
        packet(&mut words, 0x110, &[0]); // 3D WAIT_FOR_IDLE
        packet(&mut words, 0x104, &[(base >> 32) as u32, base as u32, 0]);
        packet(&mut words, 0x100, &[0]);
        let index = words.len();
        packet(
            &mut words,
            0,
            &[SWITCH_1_GM20B_PROFILE.classes().compute().0],
        );
        words[index] |= 1 << 13;
        let index = words.len();
        packet(&mut words, 0x110, &[0]);
        words[index] |= 1 << 13;
        vec![words]
    });
    let mut kinds = Vec::new();
    fixture
        .run(&mut |work| match work.command {
            NativeCommand::WaitForIdle => kinds.push(0),
            NativeCommand::ReportMemory { bytes, fence, .. } => {
                assert!(fence);
                assert_eq!(*bytes, [0; 16]);
                kinds.push(1);
            }
            other => panic!("unexpected {other:?}"),
        })
        .unwrap();
    assert_eq!(kinds, [0, 1, 0]);
    assert_eq!(fixture.read(0, 16), [0; 16]);
}

#[test]
#[ignore = "opt-in control service latency measurement"]
fn command_control_latency() {
    use nixe_gpu_maxwell::{
        MaxwellCommandHost, MaxwellSubmissionExecutionError, stream_maxwell_frontend,
    };
    struct ClockedControl {
        previous: std::time::Instant,
        maximum: std::time::Duration,
        count: u64,
    }
    impl MaxwellCommandHost for ClockedControl {
        async fn emit(
            &mut self,
            _: nixe_gpu::native::NativeWork,
        ) -> Result<(), MaxwellSubmissionExecutionError> {
            panic!("NOP stream")
        }
        unsafe fn memory_written(
            &mut self,
            _: nixe_memory::MachineId,
            _: &[(nixe_memory::AllocationId, nixe_memory::Extent)],
        ) -> Result<(), std::sync::Arc<nixe_memory::MemoryError>> {
            unreachable!("fixture does not produce native memory")
        }
        fn access(
            &mut self,
            _: Vec<nixe_memory::AllocationRange>,
            _: nixe_memory::AccessOperation,
        ) -> Result<nixe_memory::AccessResult, std::sync::Arc<nixe_memory::MemoryError>> {
            panic!("RAM-current stream")
        }
        fn timestamp(&self) -> u64 {
            0
        }
        fn control(&mut self) -> Result<(), MaxwellSubmissionExecutionError> {
            let now = std::time::Instant::now();
            self.maximum = self.maximum.max(now - self.previous);
            self.previous = now;
            self.count += 1;
            Ok(())
        }
    }
    let mut fixture = Fixture::new(|_| {
        vec![
            std::iter::once(header(3, 8, 8191))
                .chain(std::iter::repeat_n(0, 8191))
                .collect(),
        ]
    });
    let mut host = ClockedControl {
        previous: std::time::Instant::now(),
        maximum: std::time::Duration::ZERO,
        count: 0,
    };
    for _ in 0..1000 {
        host.previous = std::time::Instant::now();
        command::ready(stream_maxwell_frontend(
            &fixture.dispatch,
            &mut fixture.channel,
            fixture.space.memory(),
            &fixture.frontend,
            &mut fixture.scratch,
            &mut host,
        ))
        .unwrap();
    }
    assert_eq!(host.count, 3000);
    println!(
        "control latency: max_ns={} controls={}",
        host.maximum.as_nanos(),
        host.count
    );
}

#[test]
fn scalar_and_bulk_inline_ram_writes_do_not_allocate_per_word() {
    for (count, bulk) in [
        (0, false),
        (1, false),
        (64, false),
        (64, true),
        (65, true),
        (1024, true),
        (4096, true),
    ] {
        let mut fixture = Fixture::new(|base| {
            let base = base + 32 * nixe_memory::PAGE_SIZE;
            let mut words = Vec::new();
            packet(
                &mut words,
                0,
                &[SWITCH_1_GM20B_PROFILE.classes().three_d().0],
            );
            packet(
                &mut words,
                0x2380,
                &[(count * 4).max(256), (base >> 32) as u32, base as u32, 0],
            );
            if bulk {
                words.push(header(3, 0x2390, count));
                words.extend(0..count);
            } else {
                for i in 0..count {
                    words.push(header(4, 0x2390, i));
                }
            }
            vec![words]
        });
        for _ in 0..10 {
            fixture.run(&mut |_| panic!()).unwrap();
        }
        allocation_counter::COUNT.with(|n| n.set(Some(0)));
        let mut controls = 0;
        for _ in 0..100 {
            controls += fixture.run(&mut |_| panic!()).unwrap();
        }
        let allocations = allocation_counter::COUNT.with(|n| n.replace(None).unwrap());
        if count != 0 {
            assert_eq!(
                fixture.read(32 * nixe_memory::PAGE_SIZE, count as usize * 4),
                (0..count).flat_map(u32::to_le_bytes).collect::<Vec<_>>()
            );
        }
        assert_eq!(
            allocations / 100,
            1,
            "only the command-entry pin may allocate"
        );
        eprintln!(
            "inline RAM count={count} bulk={bulk}: allocations/list={} controls/list={}",
            allocations / 100,
            controls / 100
        );
    }
}

#[test]
fn inline_uploads_to_observed_resources_do_not_materialize_a_payload() {
    let mut counts = Vec::new();
    for count in [64u32, 4096] {
        let mut fixture = Fixture::new(|base| {
            let base = base + 32 * nixe_memory::PAGE_SIZE;
            let mut words = Vec::new();
            packet(
                &mut words,
                0,
                &[SWITCH_1_GM20B_PROFILE.classes().three_d().0],
            );
            packet(
                &mut words,
                0x2380,
                &[count * 4, (base >> 32) as u32, base as u32, 0],
            );
            words.push(header(3, 0x2390, count));
            words.extend(0..count);
            vec![words]
        });
        let input = nixe_memory::AllocationRange::new(
            fixture.allocation.pin(),
            32 * nixe_memory::PAGE_SIZE,
            u64::from(count) * 4,
        )
        .unwrap();
        // Warm parser storage, then reinstall the same tracking used by imported buffers.
        fixture.run(&mut |_| panic!()).unwrap();
        fixture
            .memory
            .capture_resource_scattered(std::slice::from_ref(&input), &[], &mut [])
            .unwrap();
        allocation_counter::COUNT.with(|n| n.set(Some(0)));
        fixture.run(&mut |_| panic!()).unwrap();
        let allocations = allocation_counter::COUNT.with(|n| n.replace(None).unwrap());
        counts.push(allocations);
        assert!(!fixture.memory.take_cpu_modified_pages().is_empty());
        assert_eq!(
            fixture.read(32 * nixe_memory::PAGE_SIZE, count as usize * 4),
            (0..count).flat_map(u32::to_le_bytes).collect::<Vec<_>>()
        );
        eprintln!("observed inline count={count}: allocations/list={allocations}");
    }
    assert_eq!(
        counts[0], counts[1],
        "payload size must not add an allocation"
    );
}

#[test]
fn dma_tiling_and_untile_preserve_padding_and_complete_in_canonical_memory() {
    for (block, native) in [0, 2, 4]
        .into_iter()
        .flat_map(|block| [(block, false), (block, true)])
    {
        let mut fixture = Fixture::new(|base| {
            let mut words = Vec::new();
            let addresses = [base + 0x4000, base + 0x8000, base + 0xc000];
            for (method, value) in [
                (0, SWITCH_1_GM20B_PROFILE.classes().dma_copy().0),
                (0x400, (addresses[0] >> 32) as u32),
                (0x404, addresses[0] as u32),
                (0x408, (addresses[1] >> 32) as u32),
                (0x40c, addresses[1] as u32),
                (0x410, 96),
                (0x418, 67),
                (0x41c, 19),
                (0x70c, 0x1000 | block << 4),
                (0x710, 128),
                (0x714, 32),
                (0x718, 1),
                (0x720, 5 | 3 << 16),
                (0x300, 1 | 1 << 7 | 1 << 9),
                (0x400, (addresses[1] >> 32) as u32),
                (0x404, addresses[1] as u32),
                (0x408, (addresses[2] >> 32) as u32),
                (0x40c, addresses[2] as u32),
                (0x414, 96),
                (0x728, 0x1000 | block << 4),
                (0x72c, 128),
                (0x730, 32),
                (0x734, 1),
                (0x73c, 5 | 3 << 16),
                (0x300, 1 | 1 << 8 | 1 << 9),
            ] {
                let at = words.len();
                packet(&mut words, method, &[value]);
                words[at] |= 4 << 13;
            }
            vec![words]
        });
        let source: Vec<u8> = (0..96 * 19).map(|i| (i * 13 + i / 96) as u8).collect();
        fixture
            .memory
            .write_resolved(
                &[nixe_memory::AllocationRange::new(
                    fixture.allocation.pin(),
                    0x4000,
                    source.len() as u64,
                )
                .unwrap()],
                &source,
            )
            .unwrap();
        fixture
            .memory
            .write_resolved(
                &[
                    nixe_memory::AllocationRange::new(fixture.allocation.pin(), 0x8000, 0x8000)
                        .unwrap(),
                ],
                &vec![0xa5; 0x8000],
            )
            .unwrap();
        let tiled =
            nixe_memory::AllocationRange::new(fixture.allocation.pin(), 0x8000, 0x4000).unwrap();
        if native {
            fixture
                .memory
                .publish_device_write(std::slice::from_ref(&tiled), || Ok(()))
                .unwrap();
        }
        let mut native_copies = 0;
        fixture
            .run(&mut |work| {
                assert!(
                    native,
                    "RAM layout conversion must not expand into native buffer copies"
                );
                assert!(matches!(work.command, NativeCommand::CopyBuffer { .. }));
                native_copies += 1;
            })
            .unwrap();
        if native {
            assert_eq!(native_copies, 2);
            fixture.memory.retire_device_pages(&[tiled]).unwrap();
            continue;
        }
        let actual = fixture.read(0xc000, source.len());
        let mut expected = vec![0xa5; source.len()];
        for y in 0..19 {
            expected[y * 96..y * 96 + 67].copy_from_slice(&source[y * 96..y * 96 + 67]);
        }
        assert_eq!(actual, expected, "block-height={block}");
    }
}
