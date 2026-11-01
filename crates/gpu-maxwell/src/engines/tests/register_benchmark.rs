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
        &mut async |_| Ok::<_, std::convert::Infallible>(()),
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
