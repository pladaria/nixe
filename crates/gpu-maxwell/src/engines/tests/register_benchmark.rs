//! Ordinary command consumption must not allocate after the entry is retained.
use super::*;
use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
};
thread_local! { pub(crate) static ALLOCATIONS: Cell<Option<u64>> = const {Cell::new(None)}; }
struct CountAllocations;
#[global_allocator]
static ALLOCATOR: CountAllocations = CountAllocations;
unsafe impl GlobalAlloc for CountAllocations {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = ALLOCATIONS.try_with(|n| {
            if let Some(value) = n.get() {
                n.set(Some(value + 1));
            }
        });
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        unsafe { System.dealloc(p, layout) }
    }
    unsafe fn realloc(&self, p: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let _ = ALLOCATIONS.try_with(|n| {
            if let Some(value) = n.get() {
                n.set(Some(value + 1));
            }
        });
        unsafe { System.realloc(p, layout, size) }
    }
}
fn run(
    channel: &mut MaxwellGpuChannel,
    stream: &mut crate::pushbuffer::packet::CommandStream,
    mme: &mut Option<MaxwellMethodDispatch>,
    parameters: &mut Vec<u32>,
) {
    stream.rewind();
    crate::test_memory::ready(stream_maxwell_engine_entry(
        channel,
        stream,
        None,
        mme,
        parameters,
        &mut async |_| Ok::<_, crate::MaxwellFrontendDispatchError>(()),
    ))
    .unwrap_or_else(|_| panic!("benchmark dispatch failed"));
}
#[test]
fn ordinary_register_admission_allocates_nothing_with_a_retained_snapshot() {
    let mut channel = three_d_channel();
    let snapshot = channel.three_d().clone();
    let mut streams = [packet(0x2380 / 4, 32), packet(0x2380 / 4, 64)]
        .map(|p| crate::pushbuffer::packet::CommandStream::fixture(p.packets()[0].view()));
    let mut mme = None;
    let mut parameters = Vec::new();
    ALLOCATIONS.with(|n| n.set(Some(0)));
    for i in 0..1000 {
        run(&mut channel, &mut streams[i % 2], &mut mme, &mut parameters);
    }
    let count = ALLOCATIONS.with(|n| n.replace(None).unwrap());
    assert_eq!(count, 0);
    assert_eq!(
        snapshot.shader_bindings().selector().size().value(),
        Some(&0)
    );
    let (_, state) = channel.three_d_with_profile_mut();
    assert_eq!(
        state
            .operation_state()
            .shader_bindings()
            .selector()
            .size()
            .value(),
        Some(&64)
    );
}

#[test]
#[ignore = "controlled command-consumption measurement"]
fn command_consumption_throughput() {
    use std::hint::black_box;
    use std::time::Instant;
    let iterations: usize = std::env::var("NIXE_COMMAND_BENCH_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(100_000);
    for (name, words, methods) in [
        ("immediate", vec![(4 << 29) | (3 << 16) | (0x3400 / 4)], 1),
        (
            "increasing",
            [
                vec![(1 << 29) | (32 << 16) | (0x3400 / 4)],
                (0..32).collect(),
            ]
            .concat(),
            32,
        ),
        (
            "constant",
            [
                vec![(3 << 29) | (32 << 16) | (0x3400 / 4)],
                (0..32).collect(),
            ]
            .concat(),
            32,
        ),
        ("offset", vec![(4 << 29) | (16 << 16) | (0x238c / 4)], 1),
        (
            "mixed",
            vec![
                (4 << 29) | (0x238c / 4),
                (5 << 29) | (5 << 16) | (0x238c / 4),
                4,
                1,
                2,
                3,
                4,
            ],
            6,
        ),
    ] {
        let mut channel = three_d_channel();
        for (m, v) in [(0x2380, 256), (0x2384, 0), (0x2388, 0x4000), (0x238c, 0)] {
            program_three_d(&mut channel, m, v);
        }
        let origin = word(0, 0).unwrap().location();
        let mut stream = crate::pushbuffer::packet::CommandStream::fixture_words(origin, &words);
        let mut pending = None;
        let mut parameters = Vec::new();
        let warm = Instant::now();
        while warm.elapsed().as_millis() < 100 {
            run(&mut channel, &mut stream, &mut pending, &mut parameters);
        }
        let start = Instant::now();
        for _ in 0..iterations {
            run(&mut channel, &mut stream, &mut pending, &mut parameters);
            black_box(&channel);
        }
        let ns = start.elapsed().as_nanos() as f64 / (iterations * methods) as f64;
        println!("COMMAND_BENCH scenario={name} ns_per_method={ns:.3} iterations={iterations}");
    }
}

#[test]
#[cfg(target_os = "linux")]
#[ignore = "controlled changing-register packet measurement"]
fn changing_register_packet_throughput() {
    use std::hint::black_box;
    use std::time::Instant;
    let iterations: usize = std::env::var("NIXE_COMMAND_BENCH_ITERS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(50_000);
    let cpu_ns = || {
        std::fs::read_to_string("/proc/thread-self/schedstat")
            .unwrap()
            .split_whitespace()
            .next()
            .unwrap()
            .parse::<u64>()
            .unwrap()
    };
    for (name, base, stride, slots, count) in [
        ("vertex_slots", 0x1c00, 0x10, 32, 3),
        ("color_slots", 0x0800, 0x40, 8, 9),
        ("scratch_control", 0x3400, 0x80, 1, 32),
    ] {
        let mut channel = three_d_channel();
        let origin = word(0, 0).unwrap().location();
        let mut streams = [0x10000, 0x20000].map(|seed| {
            let mut words = Vec::new();
            for slot in 0..slots {
                words.push((1 << 29) | (count << 16) | ((base + slot * stride) / 4));
                words.extend((0..count).map(|lane| seed + slot * 0x100 + lane));
            }
            // Flush the real frontend dirty state at an effect boundary. The
            // consumer measures admission only; it does not lower a GPU draw.
            words.push((4 << 29) | (3 << 16) | (0x0d78 / 4));
            crate::pushbuffer::packet::CommandStream::fixture_words(origin, &words)
        });
        let mut pending = None;
        let mut parameters = Vec::new();
        for index in 0..2000 {
            run(
                &mut channel,
                &mut streams[index % 2],
                &mut pending,
                &mut parameters,
            );
        }
        let before = cpu_ns();
        let start = Instant::now();
        ALLOCATIONS.with(|counter| counter.set(Some(0)));
        for index in 0..iterations {
            run(
                &mut channel,
                &mut streams[index % 2],
                &mut pending,
                &mut parameters,
            );
            black_box(&channel);
        }
        let allocations = ALLOCATIONS.with(|counter| counter.replace(None).unwrap());
        let wall_ns = start.elapsed().as_nanos();
        let elapsed_cpu_ns = cpu_ns() - before;
        let last = base + (slots - 1) * stride + (count - 1) * 4;
        let seed = if (iterations - 1).is_multiple_of(2) {
            0x10000
        } else {
            0x20000
        };
        assert_eq!(
            channel
                .three_d_mut()
                .raw_register_value(nixe_gpu::GpuMethodId(last)),
            Some(seed + (slots - 1) * 0x100 + count - 1)
        );
        assert_eq!(allocations, 0);
        let methods = u64::from(slots * count) * iterations as u64;
        println!(
            "CHANGING_REGISTER_BENCH {{\"scenario\":\"{name}\",\"iterations\":{iterations},\"methods\":{methods},\"cpu_ns\":{elapsed_cpu_ns},\"wall_ns\":{wall_ns},\"allocations\":{allocations}}}"
        );
    }
}

#[test]
#[cfg(target_os = "linux")]
#[ignore = "controlled real MME register-admission measurement"]
fn mme_register_admission_throughput() {
    use std::hint::black_box;
    use std::time::Instant;
    println!(
        "MME_ADMISSION_LAYOUT frontend_error={} mme_error={}",
        std::mem::size_of::<crate::MaxwellFrontendDispatchError>(),
        std::mem::size_of::<
            super::super::threed::MaxwellThreeDMmePreflightError<
                crate::MaxwellFrontendDispatchError,
            >,
        >()
    );
    let iterations: usize = std::env::var("NIXE_MME_BENCH_ITERS")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|&value| value > 0)
        .unwrap_or(50_000);
    let cpu_ns = || {
        std::fs::read_to_string("/proc/thread-self/schedstat")
            .unwrap()
            .split_whitespace()
            .next()
            .unwrap()
            .parse::<u64>()
            .unwrap()
    };
    // Read, increment and send through the real channel. The read and backward
    // branch require the ISA interpreter, including its live register semantics.
    // ISA reference: https://github.com/eden-emulator/mirror/blob/de7e6e01d43e8cf2ec193d74fa053b0d7949955f/src/video_core/macro.cpp
    let method = 0x3400 / 4;
    let count = 64;
    let code = [
        1 | (2 << 4) | (method << 14),
        5 | (1 << 4) | (2 << 8) | (method << 14),
        1 | (1 << 4) | (2 << 8) | (2 << 11) | (1 << 14),
        (4 << 4) | (2 << 11),
        1 | (1 << 4) | (1 << 8) | (1 << 11) | ((-1_i32 as u32) << 14),
        7 | (1 << 4) | (1 << 5) | (1 << 11) | ((-4_i32 as u32) << 14),
        1 | (1 << 4) | (1 << 7),
        1 | (1 << 4),
    ];
    for (scenario, control) in [("bypass", 2), ("track", 0), ("replay", 3)] {
        let mut channel = three_d_channel();
        load_mme_program(&mut channel, 0, &code);
        let origin = word(0, 0).unwrap().location();
        let mut stream = crate::pushbuffer::packet::CommandStream::fixture_words(
            origin,
            &[(4 << 29) | (count << 16) | (0x3800 / 4)],
        );
        let mut pending = None;
        let mut parameters = Vec::new();
        program_three_d(&mut channel, 0x124, control);
        for _ in 0..1000 {
            run(&mut channel, &mut stream, &mut pending, &mut parameters);
        }
        program_three_d(&mut channel, 0x124, 0);
        program_three_d(&mut channel, 0x3400, 1);
        program_three_d(&mut channel, 0x124, control);
        let before = cpu_ns();
        let start = Instant::now();
        ALLOCATIONS.with(|counter| counter.set(Some(0)));
        for _ in 0..iterations {
            run(&mut channel, &mut stream, &mut pending, &mut parameters);
            black_box(&channel);
        }
        let allocations = ALLOCATIONS.with(|counter| counter.replace(None).unwrap());
        let wall_ns = start.elapsed().as_nanos();
        let elapsed_cpu_ns = cpu_ns() - before;
        let methods = u64::from(count) * iterations as u64;
        let expected = if control == 3 {
            1
        } else {
            1_u32.wrapping_add(methods as u32)
        };
        assert_eq!(
            channel
                .three_d_mut()
                .raw_register_value(nixe_gpu::GpuMethodId(0x3400)),
            Some(expected)
        );
        assert_eq!(allocations, 0);
        assert!(pending.is_none());
        assert!(parameters.is_empty());
        println!(
            "MME_ADMISSION_BENCH {{\"scenario\":\"{scenario}\",\"iterations\":{iterations},\"methods\":{methods},\"cpu_ns\":{elapsed_cpu_ns},\"wall_ns\":{wall_ns},\"allocations\":{allocations}}}"
        );
    }
}
